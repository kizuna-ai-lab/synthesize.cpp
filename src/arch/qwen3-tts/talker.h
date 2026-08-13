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
// `speaker_embedding` and `speaker_index` are the x-vector path: when
// `speaker_embedding` is non-null, the row of the codec embedding at
// `codec_tokens[speaker_index]` is discarded and `speaker_embedding` -- a row
// of the same [hidden_size, 1] shape -- is summed into the text tower at that
// position instead. `speaker_index` indexes `codec_tokens`, matching
// TalkerPrompt::external_speaker_index (talker-host.h). Left at their
// defaults, the Stage 1 path is byte-identical to before this parameter pair
// existed.
//
// `acoustic_embedding` and `acoustic_offset` are the transcript-assisted (ICL)
// path: the reference block's groups 1..15, already summed per position by
// sum_code_embeddings (code-predictor.h) into a [hidden_size, frames] tensor,
// accumulated on top of the two streams starting at POSITION `acoustic_offset`
// -- a position index like `codec_offset`, not an index into `codec_tokens`
// like `speaker_index`. Group 0 is not in it: it reads the talker's own codec
// embedding and therefore travels as an ordinary entry of `codec_tokens`.
// The run must reach the last position, because it is accumulated as a tail.
// Left at their defaults, the Stage 1 and Plan 2 paths are byte-identical to
// before this parameter pair existed.
//
// Returns nullptr rather than aborting on anything it cannot build, including
// a `speaker_index` outside `[0, codec_tokens->ne[0])`, a `speaker_embedding`
// whose width is not the talker's hidden size, or an `acoustic_embedding` whose
// width or placement does not make it a tail.
ggml_tensor * build_talker_prefill_input(ggml_context *        context,
                                         const TalkerWeights & weights,
                                         ggml_tensor *         text_tokens,
                                         ggml_tensor *         codec_tokens,
                                         int64_t               codec_offset,
                                         ggml_tensor *         speaker_embedding  = nullptr,
                                         int64_t               speaker_index      = -1,
                                         ggml_tensor *         acoustic_embedding = nullptr,
                                         int64_t               acoustic_offset    = -1);

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
// `out_layers`, when non-null, receives each layer's output. Port validation
// compares them against the oracle's per-layer probes; nothing else reads them,
// and passing null leaves the graph unchanged.
ggml_tensor * build_talker_step(ggml_context *               context,
                                ggml_cgraph *                graph,
                                ggml_tensor *                input,
                                ggml_tensor *                position_ids,
                                ggml_tensor *                mask,
                                const TalkerWeights &        weights,
                                const AttentionShape &       shape,
                                const TalkerCache &          cache,
                                ggml_tensor **               out_hidden,
                                std::vector<ggml_tensor *> * out_layers     = nullptr,
                                ggml_tensor **               out_all_hidden = nullptr);

}  // namespace synth::qwen3tts
