#pragma once

// catalog.h carries the weight structs and weights.h the HParams-side
// parameter structs (AudioCanvasParams) -- the post-Task-4 roles.
#include "catalog.h"
#include "weights.h"

#include <cstdint>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace synth::omnivoice {

struct AttentionShape {
    uint32_t hidden_size          = 0;
    uint32_t attention_head_count = 0;
    uint32_t key_value_head_count = 0;
    uint32_t head_dim             = 0;
    float    rms_norm_eps         = 0.0f;
    float    rope_theta           = 0.0f;
};

// RMSNorm with a learned gain, which is what every norm in this family is.
ggml_tensor * rms_norm(ggml_context * context, ggml_tensor * input, ggml_tensor * weight, float eps);

// One Qwen3 block over `positions` tokens, bidirectional and cache-free.
//
// This is deliberately NOT qwen3-tts's decoder_layer: that block writes a KV
// cache and orders the writes through the graph, neither of which exists here
// because no position is ever causal. It is the codec_transformer_layer shape
// with the layer scales removed (a Qwen3 block has none) and the per-head q/k
// norms added (a Qwen3 block has them, at head_dim width, applied before rope).
//
// `input` is [hidden, positions]; `position_ids` is I32 [positions]. `mask` may
// be nullptr -- full bidirectional attention, the only mode synthesis uses --
// or an additive F32 [positions, positions] tensor (0 = may attend, -INF = may
// not), which the unit test uses to prove the mask actually gates.
//
// Returns nullptr rather than aborting on a shape it cannot build.
ggml_tensor * generator_layer(ggml_context *                context,
                              ggml_tensor *                 input,
                              ggml_tensor *                 position_ids,
                              ggml_tensor *                 mask,
                              const GeneratorLayerWeights & weights,
                              const AttentionShape &        shape);

// The canvas embedding: text positions read the text table at row 0's ids;
// audio positions read the stacked audio table once per codebook at the
// host-shifted ids (id + codebook * vocab_size) and sum the eight rows. The
// reference `torch.where`-selects between the two streams, so an audio
// position carries the sum ALONE -- the text stream is never added to it.
//
// `text_ids` is I32 [text_count] or nullptr (the unconditional branch has no
// text region); `audio_ids` is I32 [audio_count, num_codebooks], codebook-major
// rows, or nullptr. At least one must be non-null. Returns [hidden, total].
ggml_tensor * build_canvas_embedding(ggml_context *           context,
                                     const GeneratorWeights & weights,
                                     ggml_tensor *            text_ids,
                                     ggml_tensor *            audio_ids,
                                     uint32_t                 num_codebooks);

// The whole generator: embeddings -> every layer -> final norm -> audio heads
// over EVERY position (a mask-predict model predicts the full canvas at once;
// there is no last-position slice). Returns the logits reshaped to
// [vocab, codebooks, positions]: element (v, c, s) is position s's logit for
// code v of codebook c, matching the head's stacked row order c * vocab + v.
//
// `out_layers`, when non-null, receives each layer's output; `out_final`, when
// non-null, receives the hidden state after the final norm. Port validation
// compares them against the oracle's step-0 probes; nothing else reads them.
ggml_tensor * build_generator_forward(ggml_context *               context,
                                      ggml_tensor *                embeddings,
                                      ggml_tensor *                position_ids,
                                      ggml_tensor *                mask,
                                      const GeneratorWeights &     weights,
                                      const AttentionShape &       shape,
                                      const AudioCanvasParams &    canvas,
                                      std::vector<ggml_tensor *> * out_layers,
                                      ggml_tensor **               out_final);

}  // namespace synth::omnivoice
