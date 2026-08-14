#pragma once

#include "operations.h"

#include <cstdint>
#include <vector>

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

namespace synth::qwen3tts {

// The multi-token-prediction head that expands one frame's semantic code into
// the frame's remaining acoustic codes.
//
// Every acoustic group has its own embedding table and its own output head, so
// both lists are code_group_count - 1 long: group 0 is the talker's and is
// neither embedded nor predicted here.
struct CodePredictorWeights {
    std::vector<DecoderLayerWeights> layers;
    ggml_tensor *                    norm = nullptr;
    std::vector<ggml_tensor *>       codec_embedding;
    std::vector<ggml_tensor *>       lm_head;
    // The reference projects the talker's hidden state into the predictor's
    // width. This variant's widths agree, so its projection is an Identity and
    // the package carries no tensor; a rung whose widths differ would bind both.
    ggml_tensor *                    input_projection      = nullptr;
    ggml_tensor *                    input_projection_bias = nullptr;
};

// One cache per layer, all of the same capacity. The predictor's cache covers a
// single frame and is reset before each one, because the reference calls the
// predictor afresh per frame with no carried state.
struct CodePredictorCache {
    std::vector<KvCache> layers;
};

// Looks up `token_ids` in acoustic group `table`'s embedding, returning
// [hidden_size, count]. Returns nullptr for a table outside the catalog.
ggml_tensor * code_predictor_embed(ggml_context *               context,
                                   const CodePredictorWeights & weights,
                                   uint32_t                     table,
                                   ggml_tensor *                token_ids);

// Builds one predictor call over `input` ([hidden_size, positions]) and returns
// the logits of the last position under output head `lm_head`, shaped
// [vocab_size, 1].
//
// `cache.layers` must hold one cache per layer, each already advanced to the
// positions written by earlier calls; this call writes its own and leaves
// `filled` for the caller to advance.
//
// Returns nullptr rather than aborting on anything it cannot build.
ggml_tensor * build_code_predictor(ggml_context *               context,
                                   ggml_cgraph *                graph,
                                   ggml_tensor *                input,
                                   ggml_tensor *                position_ids,
                                   ggml_tensor *                mask,
                                   const CodePredictorWeights & weights,
                                   const AttentionShape &       shape,
                                   uint32_t                     lm_head,
                                   const CodePredictorCache &   cache);

// Sums each acoustic code's embedding from its own group's table, for one or
// many frames at once. Returns [hidden_size, frames].
//
// `codes` is a contiguous I32 tensor shaped GGML `[frames, code_group_count-1]`
// -- `ne[0]` is the FRAME index and is fastest, `ne[1]` selects the acoustic
// group and must equal the table count. Group-major, in other words, which is
// what makes each group's ids a contiguous 1-D view: ggml_get_rows cannot read
// a strided index tensor, so the alternative is a copy inside the graph.
//
// A decode step is the frames == 1 case and passes `[1, code_group_count-1]`,
// the same fifteen int32s it always did. The talker adds its own embedding of
// group 0 to whatever comes back; nothing here reads the semantic code.
ggml_tensor * sum_code_embeddings(ggml_context * context, const CodePredictorWeights & weights, ggml_tensor * codes);

}  // namespace synth::qwen3tts
