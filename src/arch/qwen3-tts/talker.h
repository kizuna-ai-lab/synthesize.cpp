#pragma once

#include "catalog.h"
#include "operations.h"

#include <cstdint>
#include <vector>

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

namespace synth::qwen3tts {

// One cache per talker layer. Unlike the code predictor's, this one spans the
// whole utterance and is never reset, which is why the block writes into it
// rather than concatenating onto it.
struct TalkerCache {
    std::vector<KvCache> layers;
};

// Brings text tokens through the text tower: the wide text embedding, then the
// two-layer projection that narrows it to the talker's width. Returns
// [hidden_size, count].
ggml_tensor * build_text_projection(ggml_context * context, const TalkerWeights & weights, ggml_tensor * token_ids);

// The prefill's input embeddings, [hidden_size, positions].
//
// Two streams are summed. Every position carries a text token; the trailing
// `codec_tokens->ne[0]` positions also carry a codec token, and `codec_offset` is
// where that run starts. The layout that makes this a tail rather than a scatter
// is decided and checked in talker-host.h.
//
// Returns nullptr rather than aborting on anything it cannot build.
ggml_tensor * build_talker_prefill_input(ggml_context *        context,
                                         const TalkerWeights & weights,
                                         ggml_tensor *         text_tokens,
                                         ggml_tensor *         codec_tokens,
                                         int64_t               codec_offset);

// A decode step's input, [hidden_size, 1]: the frame's summed code embeddings
// plus the text this step contributes. The sum comes from the code predictor's
// tables and the talker's own for group 0; see sum_code_embeddings.
ggml_tensor * build_talker_step_input(ggml_context *        context,
                                     const TalkerWeights & weights,
                                     ggml_tensor *         summed_codes,
                                     ggml_tensor *         text_token);

// Runs the talker over `input` ([hidden_size, positions]) and returns the logits
// of the last position, [codec_vocab_size, 1]. `out_hidden` receives that
// position's hidden state after the final norm, which is what the code predictor
// prefills with.
ggml_tensor * build_talker_step(ggml_context *         context,
                                ggml_cgraph *          graph,
                                ggml_tensor *          input,
                                ggml_tensor *          position_ids,
                                ggml_tensor *          mask,
                                const TalkerWeights &  weights,
                                const AttentionShape & shape,
                                const TalkerCache &    cache,
                                ggml_tensor **         out_hidden);

}  // namespace synth::qwen3tts
