// Graph construction for the code predictor.
//
// The reference reaches this through a full generate() call per frame, which is
// why its cost is dominated by per-step overhead rather than by arithmetic: a
// step is about 161 MFLOP against a 1024-wide hidden state, and there are
// fifteen of them per 80 ms frame. Building the steps directly is the point of
// porting it.

#include "code-predictor.h"

#include "ggml.h"

namespace synth::qwen3tts {

namespace {

bool table_in_range(const CodePredictorWeights & weights, uint32_t table) {
    return table < weights.codec_embedding.size() && weights.codec_embedding[table] != nullptr;
}

}  // namespace

ggml_tensor * code_predictor_embed(ggml_context *               context,
                                   const CodePredictorWeights & weights,
                                   uint32_t                     table,
                                   ggml_tensor *                token_ids) {
    if (context == nullptr || token_ids == nullptr || token_ids->type != GGML_TYPE_I32 ||
        !table_in_range(weights, table)) {
        return nullptr;
    }
    return ggml_get_rows(context, weights.codec_embedding[table], token_ids);
}

ggml_tensor * build_code_predictor(ggml_context *               context,
                                   ggml_cgraph *                graph,
                                   ggml_tensor *                input,
                                   ggml_tensor *                position_ids,
                                   ggml_tensor *                mask,
                                   const CodePredictorWeights & weights,
                                   const AttentionShape &       shape,
                                   uint32_t                     lm_head,
                                   const CodePredictorCache &   cache) {
    if (context == nullptr || graph == nullptr || input == nullptr || weights.norm == nullptr ||
        weights.layers.empty() || weights.layers.size() != cache.layers.size() || lm_head >= weights.lm_head.size() ||
        weights.lm_head[lm_head] == nullptr) {
        return nullptr;
    }
    // A projection is either fully bound or absent; a half-bound one would
    // silently drop the bias.
    if ((weights.input_projection == nullptr) != (weights.input_projection_bias == nullptr)) {
        return nullptr;
    }

    ggml_tensor * hidden = input;
    if (weights.input_projection != nullptr) {
        hidden = ggml_add(context, ggml_mul_mat(context, weights.input_projection, hidden),
                          as_f32(context, weights.input_projection_bias));
    }

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

    // Only the last position predicts. A prefill carries the talker's hidden
    // state and the semantic code's embedding, and it is the code's position
    // that the first acoustic group is read from.
    const int64_t positions = hidden->ne[1];
    ggml_tensor * last =
        ggml_view_2d(context, hidden, hidden->ne[0], 1, hidden->nb[1], (positions - 1) * hidden->nb[1]);
    return ggml_mul_mat(context, weights.lm_head[lm_head], ggml_cont(context, last));
}

ggml_tensor * sum_code_embeddings(ggml_context * context, const CodePredictorWeights & weights, ggml_tensor * codes) {
    if (context == nullptr || codes == nullptr || codes->type != GGML_TYPE_I32 || weights.codec_embedding.empty() ||
        codes->ne[0] != static_cast<int64_t>(weights.codec_embedding.size())) {
        return nullptr;
    }

    ggml_tensor * total = nullptr;
    for (size_t table = 0; table < weights.codec_embedding.size(); ++table) {
        ggml_tensor * one = ggml_view_1d(context, codes, 1, static_cast<size_t>(table) * codes->nb[0]);
        ggml_tensor * row = code_predictor_embed(context, weights, static_cast<uint32_t>(table), one);
        if (row == nullptr) {
            return nullptr;
        }
        total = total == nullptr ? row : ggml_add(context, total, row);
    }
    return total;
}

}  // namespace synth::qwen3tts
