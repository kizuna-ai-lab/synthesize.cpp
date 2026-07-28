// Graph construction for the talker.
//
// The talker's own block is the shared Qwen3 one: its attention differs from the
// code predictor's only in going through the reference's multimodal rope helper,
// and that collapses exactly to plain rope here because all three position rows
// are always identical. See docs/porting/families/qwen3-tts.md.
//
// What is specific to the talker is that two towers meet at every input position
// and are summed: a text token brought down through the projection, and a row of
// the codec embedding.

#include "talker.h"

#include "ggml.h"

namespace synth::qwen3tts {

ggml_tensor * build_text_projection(ggml_context * context, const TalkerWeights & weights, ggml_tensor * token_ids) {
    if (context == nullptr || token_ids == nullptr || token_ids->type != GGML_TYPE_I32 ||
        weights.text_embedding == nullptr || weights.text_projection_1.weight == nullptr ||
        weights.text_projection_1.bias == nullptr || weights.text_projection_2.weight == nullptr ||
        weights.text_projection_2.bias == nullptr) {
        return nullptr;
    }
    ggml_tensor * hidden = ggml_get_rows(context, weights.text_embedding, token_ids);
    hidden = ggml_add(context, ggml_mul_mat(context, weights.text_projection_1.weight, hidden),
                      weights.text_projection_1.bias);
    hidden = ggml_silu(context, hidden);
    return ggml_add(context, ggml_mul_mat(context, weights.text_projection_2.weight, hidden),
                    weights.text_projection_2.bias);
}

ggml_tensor * build_talker_prefill_input(ggml_context *        context,
                                         const TalkerWeights & weights,
                                         ggml_tensor *         text_tokens,
                                         ggml_tensor *         codec_tokens,
                                         int64_t               codec_offset) {
    if (context == nullptr || codec_tokens == nullptr || codec_tokens->type != GGML_TYPE_I32 ||
        weights.codec_embedding == nullptr || codec_offset < 0) {
        return nullptr;
    }
    ggml_tensor * text = build_text_projection(context, weights, text_tokens);
    if (text == nullptr) {
        return nullptr;
    }
    if (codec_offset + codec_tokens->ne[0] != text->ne[1]) {
        return nullptr;
    }
    ggml_tensor * codec = ggml_get_rows(context, weights.codec_embedding, codec_tokens);
    // The codec stream is a tail of the text stream, so it is accumulated into it
    // at an offset rather than scattered row by row.
    return ggml_acc(context, text, codec, text->nb[1], text->nb[2], text->nb[3],
                    static_cast<size_t>(codec_offset) * text->nb[1]);
}

ggml_tensor * build_talker_step_input(ggml_context *        context,
                                     const TalkerWeights & weights,
                                     ggml_tensor *         summed_codes,
                                     ggml_tensor *         text_token) {
    if (context == nullptr || summed_codes == nullptr) {
        return nullptr;
    }
    ggml_tensor * text = build_text_projection(context, weights, text_token);
    if (text == nullptr || text->ne[0] != summed_codes->ne[0] || text->ne[1] != summed_codes->ne[1]) {
        return nullptr;
    }
    return ggml_add(context, summed_codes, text);
}

ggml_tensor * build_talker_step(ggml_context *         context,
                                ggml_cgraph *          graph,
                                ggml_tensor *          input,
                                ggml_tensor *          position_ids,
                                ggml_tensor *          mask,
                                const TalkerWeights &  weights,
                                const AttentionShape & shape,
                                const TalkerCache &    cache,
                                ggml_tensor **         out_hidden) {
    if (context == nullptr || graph == nullptr || input == nullptr || out_hidden == nullptr ||
        weights.norm == nullptr || weights.codec_head == nullptr || weights.layers.empty() ||
        weights.layers.size() != cache.layers.size()) {
        return nullptr;
    }
    *out_hidden = nullptr;

    ggml_tensor * hidden = input;
    for (size_t index = 0; index < weights.layers.size(); ++index) {
        hidden = decoder_layer(context, graph, hidden, position_ids, mask, weights.layers[index], shape,
                               cache.layers[index]);
        if (hidden == nullptr) {
            return nullptr;
        }
    }
    hidden = rms_norm(context, hidden, weights.norm, shape.rms_norm_eps);
    if (hidden == nullptr) {
        return nullptr;
    }

    // Only the last position predicts, and its hidden state is also what the code
    // predictor prefills with, so both come from the same view.
    const int64_t positions = hidden->ne[1];
    ggml_tensor * last =
        ggml_cont(context, ggml_view_2d(context, hidden, hidden->ne[0], 1, hidden->nb[1],
                                        static_cast<size_t>(positions - 1) * hidden->nb[1]));
    *out_hidden = last;
    return ggml_mul_mat(context, weights.codec_head, last);
}

}  // namespace synth::qwen3tts
