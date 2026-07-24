#pragma once

#include <cstdint>

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

namespace synth::kokoro {

struct LinearWeights;
struct NormWeights;

// x * gamma + beta over the feature axis, which is ne[0] in this family's
// [features, time] layout.
ggml_tensor * layer_norm(ggml_context * context, ggml_tensor * input, const NormWeights & weights, float epsilon);

ggml_tensor * linear(ggml_context * context, ggml_tensor * input, const LinearWeights & weights);

// Adaptive layer norm: normalize without learned affine, then apply a scale and
// shift projected from the style vector. Upstream stores the projection as one
// linear whose output is split in half, and applies (1 + gamma) rather than
// gamma, so a zero projection is the identity.
ggml_tensor * ada_layer_norm(ggml_context *        context,
                             ggml_tensor *         input,
                             ggml_tensor *         style,
                             const LinearWeights & projection,
                             float                 epsilon);

// Broadcasts a [features] vector across `length` time steps.
ggml_tensor * broadcast_over_time(ggml_context * context, ggml_tensor * vector, int64_t length);

// Weights of one LSTM direction, in the layout PyTorch stores them: the four
// gate blocks are stacked input, forget, cell, output along the row axis.
struct LstmDirectionWeights {
    ggml_tensor * weight_ih = nullptr;  // [input_dim, 4 * hidden]
    ggml_tensor * weight_hh = nullptr;  // [hidden, 4 * hidden]
    ggml_tensor * bias_ih   = nullptr;  // [4 * hidden]
    ggml_tensor * bias_hh   = nullptr;  // [4 * hidden]
};

struct LstmWeights {
    LstmDirectionWeights forward;
    LstmDirectionWeights reverse;
    uint32_t             input_dim = 0;
    uint32_t             hidden    = 0;  // per direction
};

// Scratch a bidirectional LSTM needs from a persistent backend buffer.
//
// `forward_store` and `reverse_store` must be [hidden, length] tensors that the
// graph allocator does not own. Per-step outputs are written into disjoint
// views of them, so nothing the recurrent chain touches can be recycled
// underneath it. `zero_state` is a [hidden] tensor of zeros used as the initial
// hidden and cell state.
//
// Allocator-managed accumulation was measured to be wrong on CUDA while looking
// correct on CPU, so this separation is a correctness requirement rather than
// an optimization. See docs/porting/families/kokoro.md.
struct LstmScratch {
    ggml_tensor * zero_state    = nullptr;
    ggml_tensor * forward_store = nullptr;
    ggml_tensor * reverse_store = nullptr;
};

// Returns the number of graph nodes `build_bidirectional_lstm` will add for a
// sequence of `length` steps, so callers can size their graph up front.
uint64_t bidirectional_lstm_node_count(uint64_t length);

// Unrolls a bidirectional LSTM over `input` of shape [input_dim, length].
//
// The input-side projection is one batched matrix multiply over the whole
// sequence; only the hidden-side projection stays inside the recurrent loop,
// where the four gates are computed by a single fused multiply.
//
// On success the concatenated [2 * hidden, length] result is returned. The
// caller must have expanded the graph with the returned tensor before compute.
ggml_tensor * build_bidirectional_lstm(ggml_context *      context,
                                       ggml_cgraph *       graph,
                                       ggml_tensor *       input,
                                       const LstmWeights & weights,
                                       const LstmScratch & scratch,
                                       uint32_t            length);

}  // namespace synth::kokoro
