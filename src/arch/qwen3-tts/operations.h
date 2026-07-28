#pragma once

#include <cstdint>

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

namespace synth::qwen3tts {

// One Qwen3 decoder layer's weights. The talker and the code predictor use the
// same block shape, so this is bound once and built once for both.
struct DecoderLayerWeights {
    ggml_tensor * input_layernorm          = nullptr;
    ggml_tensor * q_proj                   = nullptr;
    ggml_tensor * k_proj                   = nullptr;
    ggml_tensor * v_proj                   = nullptr;
    ggml_tensor * o_proj                   = nullptr;
    // Qwen3 normalizes each head of q and k before rope rather than normalizing
    // the packed projection, so these are head_dim wide, not hidden wide.
    ggml_tensor * q_norm                   = nullptr;
    ggml_tensor * k_norm                   = nullptr;
    ggml_tensor * post_attention_layernorm = nullptr;
    ggml_tensor * gate_proj                = nullptr;
    ggml_tensor * up_proj                  = nullptr;
    ggml_tensor * down_proj                = nullptr;
};

struct AttentionShape {
    uint32_t hidden_size          = 0;
    uint32_t attention_head_count = 0;
    uint32_t key_value_head_count = 0;
    uint32_t head_dim             = 0;
    float    rms_norm_eps         = 0.0f;
    float    rope_theta           = 0.0f;
};

// A layer's key/value cache. `k` and `v` are [head_dim, key_value_head_count,
// capacity] and live in a buffer the graph allocator does not own, because a
// decode step reads what earlier steps wrote.
//
// `filled` counts the positions written before this call. The block writes this
// call's keys and values at that offset and attends over the whole prefix, so
// the cache grows in place rather than by concatenation — which would copy the
// entire cache once per step, and is the difference between a talker that costs
// O(n) per utterance and one that costs O(n^2).
struct KvCache {
    ggml_tensor * k      = nullptr;
    ggml_tensor * v      = nullptr;
    int64_t       filled = 0;
};

// Brings a weight to F32 when it is not already there.
//
// ggml's matrix multiply reads a BF16 weight directly, but its elementwise
// operations do not: a norm gain or a bias arriving as BF16 aborts inside
// binary_op rather than being converted. Every such weight in the talker half is
// BF16, because that is what the checkpoint stores.
ggml_tensor * as_f32(ggml_context * context, ggml_tensor * weight);

// RMSNorm with a learned gain, which is what every norm in this family is.
ggml_tensor * rms_norm(ggml_context * context, ggml_tensor * input, ggml_tensor * weight, float eps);

// SwiGLU: down(silu(gate(x)) * up(x)).
ggml_tensor * swiglu_mlp(ggml_context * context, ggml_tensor * input, const DecoderLayerWeights & weights);

// One decoder layer over `positions` tokens.
//
// `input` is [hidden_size, positions]. `position_ids` is an I32 vector of the
// absolute position of each token, which rope consumes; passing it explicitly
// rather than deriving it is what lets a cached decode step sit at position n
// while holding a single token.
//
// `mask` is added to the pre-softmax scores and so is [cache.filled + positions,
// positions] F32, holding 0 where a query may attend and -INFINITY where it may
// not. A single-token decode step attending to its whole cache needs no mask and
// passes nullptr; a step holding several positions at once must pass a causal
// one, because nothing else stops position 0 from reading position 2.
//
// The cache writes are expanded into `graph` before the reads that depend on
// them, which is what orders the two.
//
// Returns nullptr rather than aborting on a shape the block cannot build.
ggml_tensor * decoder_layer(ggml_context *              context,
                            ggml_cgraph *               graph,
                            ggml_tensor *               input,
                            ggml_tensor *               position_ids,
                            ggml_tensor *               mask,
                            const DecoderLayerWeights & weights,
                            const AttentionShape &      shape,
                            const KvCache &             cache);

}  // namespace synth::qwen3tts
