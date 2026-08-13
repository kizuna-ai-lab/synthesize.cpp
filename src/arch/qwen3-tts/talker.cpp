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

namespace {

// The ICL block's groups 1..15, already summed per position, laid on top of the
// two ordinary streams. A no-op without one, which is what keeps the Stage 1 and
// Plan 2 paths at the node counts they had. Shapes are checked by the caller,
// once, before either accumulation path forks.
ggml_tensor * accumulate_acoustic(ggml_context * context,
                                  ggml_tensor *  summed,
                                  ggml_tensor *  acoustic_embedding,
                                  int64_t        acoustic_offset) {
    if (acoustic_embedding == nullptr) {
        return summed;
    }
    return ggml_acc(context, summed, acoustic_embedding, summed->nb[1], summed->nb[2], summed->nb[3],
                    static_cast<size_t>(acoustic_offset) * summed->nb[1]);
}

}  // namespace

ggml_tensor * build_text_projection(ggml_context * context, const TalkerWeights & weights, ggml_tensor * token_ids) {
    if (context == nullptr || token_ids == nullptr || token_ids->type != GGML_TYPE_I32 ||
        weights.text_embedding == nullptr || weights.text_projection_1.weight == nullptr ||
        weights.text_projection_1.bias == nullptr || weights.text_projection_2.weight == nullptr ||
        weights.text_projection_2.bias == nullptr) {
        return nullptr;
    }
    ggml_tensor * hidden = ggml_get_rows(context, weights.text_embedding, token_ids);
    hidden               = ggml_add(context, ggml_mul_mat(context, weights.text_projection_1.weight, hidden),
                                    as_f32(context, weights.text_projection_1.bias));
    hidden               = ggml_silu(context, hidden);
    return ggml_add(context, ggml_mul_mat(context, weights.text_projection_2.weight, hidden),
                    as_f32(context, weights.text_projection_2.bias));
}

ggml_tensor * build_talker_prefill_input(ggml_context *        context,
                                         const TalkerWeights & weights,
                                         ggml_tensor *         text_tokens,
                                         ggml_tensor *         codec_tokens,
                                         int64_t               codec_offset,
                                         ggml_tensor *         speaker_embedding,
                                         int64_t               speaker_index,
                                         ggml_tensor *         acoustic_embedding,
                                         int64_t               acoustic_offset) {
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
    if (acoustic_embedding != nullptr) {
        // A tail, like the codec stream: same width, and it must reach the last
        // position. An acoustic block that stops short is a reference whose
        // frames have gone out of step with their text, which no shape check
        // downstream would notice.
        if (acoustic_offset < 0 || acoustic_embedding->ne[0] != text->ne[0] ||
            acoustic_offset + acoustic_embedding->ne[1] != text->ne[1]) {
            return nullptr;
        }
    } else if (acoustic_offset >= 0) {
        // An offset with nothing to place at it means the caller flattened an
        // ICL prompt and then dropped the summed embeddings.
        return nullptr;
    }
    ggml_tensor * codec = ggml_get_rows(context, weights.codec_embedding, codec_tokens);
    if (speaker_embedding == nullptr) {
        // The codec stream is a tail of the text stream, so it is accumulated
        // into it at an offset rather than scattered row by row.
        ggml_tensor * summed = ggml_acc(context, text, codec, text->nb[1], text->nb[2], text->nb[3],
                                        static_cast<size_t>(codec_offset) * text->nb[1]);
        return accumulate_acoustic(context, summed, acoustic_embedding, acoustic_offset);
    }
    if (speaker_index < 0 || speaker_index >= codec_tokens->ne[0] || speaker_embedding->ne[0] != text->ne[0]) {
        return nullptr;
    }
    // The x-vector substitutes for one row of the codec embedding at the
    // position the speaker token occupies. enc_dim equals the talker's hidden
    // size -- weights.h:26-28: "there is no projection between them" -- so it
    // is literally a row of this [hidden_size, N] F32 tensor. Splitting the
    // accumulation around that index is what leaves the placeholder row
    // computed but never read; zeroing it in place would need a second pass
    // over a tensor ggml_acc has already consumed.
    ggml_tensor * head = ggml_view_2d(context, codec, codec->ne[0], speaker_index, codec->nb[1], 0);
    ggml_tensor * tail = ggml_view_2d(context, codec, codec->ne[0], codec->ne[1] - speaker_index - 1, codec->nb[1],
                                      static_cast<size_t>(speaker_index + 1) * codec->nb[1]);
    ggml_tensor * out  = text;
    if (speaker_index > 0) {
        out = ggml_acc(context, out, head, out->nb[1], out->nb[2], out->nb[3],
                       static_cast<size_t>(codec_offset) * out->nb[1]);
    }
    out = ggml_acc(context, out, speaker_embedding, out->nb[1], out->nb[2], out->nb[3],
                   static_cast<size_t>(codec_offset + speaker_index) * out->nb[1]);
    if (tail->ne[1] > 0) {
        out = ggml_acc(context, out, tail, out->nb[1], out->nb[2], out->nb[3],
                       static_cast<size_t>(codec_offset + speaker_index + 1) * out->nb[1]);
    }
    return accumulate_acoustic(context, out, acoustic_embedding, acoustic_offset);
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

ggml_tensor * build_talker_step(ggml_context *               context,
                                ggml_cgraph *                graph,
                                ggml_tensor *                input,
                                ggml_tensor *                position_ids,
                                ggml_tensor *                mask,
                                const TalkerWeights &        weights,
                                const AttentionShape &       shape,
                                const TalkerCache &          cache,
                                ggml_tensor **               out_hidden,
                                std::vector<ggml_tensor *> * out_layers,
                                ggml_tensor **               out_all_hidden) {
    if (context == nullptr || graph == nullptr || input == nullptr || out_hidden == nullptr ||
        weights.norm == nullptr || weights.codec_head == nullptr || weights.layers.empty() ||
        weights.layers.size() != cache.layers.size()) {
        return nullptr;
    }
    *out_hidden = nullptr;
    if (out_layers != nullptr) {
        out_layers->clear();
        out_layers->reserve(weights.layers.size());
    }

    ggml_tensor * hidden = input;
    for (size_t index = 0; index < weights.layers.size(); ++index) {
        hidden = decoder_layer(context, graph, hidden, position_ids, mask, weights.layers[index], shape,
                               cache.layers[index]);
        if (hidden == nullptr) {
            return nullptr;
        }
        if (out_layers != nullptr) {
            out_layers->push_back(hidden);
        }
    }
    hidden = rms_norm(context, hidden, weights.norm, shape.rms_norm_eps);
    if (hidden == nullptr) {
        return nullptr;
    }
    // Every position's normalized state, which port validation compares against
    // the oracle's prefill probe. The step itself only needs the last.
    if (out_all_hidden != nullptr) {
        *out_all_hidden = hidden;
    }

    // Only the last position predicts, and its hidden state is also what the code
    // predictor prefills with, so both come from the same view.
    const int64_t positions = hidden->ne[1];
    ggml_tensor * last      = ggml_cont(context, ggml_view_2d(context, hidden, hidden->ne[0], 1, hidden->nb[1],
                                                              static_cast<size_t>(positions - 1) * hidden->nb[1]));
    *out_hidden             = last;
    return ggml_mul_mat(context, weights.codec_head, last);
}

}  // namespace synth::qwen3tts
