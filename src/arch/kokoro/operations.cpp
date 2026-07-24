#include "operations.h"

#include "ggml.h"
#include "weights.h"

#include <cstdio>
#include <initializer_list>

namespace synth::kokoro {

ggml_tensor * layer_norm(ggml_context * context, ggml_tensor * input, const NormWeights & weights, float epsilon) {
    ggml_tensor * normalized = ggml_norm(context, input, epsilon);
    normalized               = ggml_mul(context, normalized, weights.weight);
    return ggml_add(context, normalized, weights.bias);
}

ggml_tensor * linear(ggml_context * context, ggml_tensor * input, const LinearWeights & weights) {
    return ggml_add(context, ggml_mul_mat(context, weights.weight, input), weights.bias);
}

ggml_tensor * broadcast_over_time(ggml_context * context, ggml_tensor * vector, int64_t length) {
    ggml_tensor * shape = ggml_new_tensor_2d(context, GGML_TYPE_F32, vector->ne[0], length);
    return ggml_repeat(context, ggml_reshape_2d(context, vector, vector->ne[0], 1), shape);
}

int64_t depthwise_transpose_conv1d_length(int64_t length,
                                          int64_t kernel,
                                          int64_t stride,
                                          int64_t padding,
                                          int64_t output_padding) {
    return (length - 1) * stride - 2 * padding + kernel + output_padding;
}

ggml_tensor * depthwise_transpose_conv1d(ggml_context * context,
                                         ggml_tensor *  input,
                                         ggml_tensor *  weight,
                                         ggml_tensor *  bias,
                                         int64_t        stride,
                                         int64_t        padding,
                                         int64_t        output_padding) {
    if (context == nullptr || input == nullptr || weight == nullptr || stride < 1) {
        return nullptr;
    }
    const int64_t length   = input->ne[0];
    const int64_t channels = input->ne[1];
    const int64_t kernel   = weight->ne[0];
    if (kernel < 1 || weight->ne[1] != 1 || weight->ne[2] != channels) {
        return nullptr;
    }

    const int64_t out_length = depthwise_transpose_conv1d_length(length, kernel, stride, padding, output_padding);
    const int64_t left       = kernel - 1 - padding;
    const int64_t right      = kernel + output_padding - stride - padding;
    if (out_length < 1 || left < 0 || right < 0) {
        return nullptr;
    }

    // Insert stride - 1 zeros after every sample: reshaping to [1, length,
    // channels] and padding the fastest axis interleaves them exactly.
    ggml_tensor * spread = ggml_reshape_3d(context, input, 1, length, channels);
    if (stride > 1) {
        spread = ggml_pad(context, spread, int(stride - 1), 0, 0, 0);
    }
    spread = ggml_reshape_2d(context, ggml_cont(context, spread), stride * length, channels);

    // Widen so every tap's window stays in range.
    ggml_tensor * padded = ggml_pad_ext(context, spread, int(left), int(right), 0, 0, 0, 0, 0, 0);

    ggml_tensor * accumulated = nullptr;
    for (int64_t tap = 0; tap < kernel; ++tap) {
        ggml_tensor * window = ggml_view_2d(context, padded, out_length, channels, padded->nb[1],
                                            size_t(kernel - 1 - tap) * padded->nb[0]);
        // One scalar per channel for this tap, strided across the kernel axis.
        ggml_tensor * scale =
            ggml_cont(context, ggml_view_2d(context, weight, 1, channels, weight->nb[2], size_t(tap) * weight->nb[0]));
        ggml_tensor * term = ggml_mul(context, ggml_cont(context, window), scale);
        accumulated        = accumulated == nullptr ? term : ggml_add(context, accumulated, term);
    }
    if (bias != nullptr) {
        accumulated = ggml_add(context, accumulated, ggml_reshape_2d(context, bias, 1, channels));
    }
    return accumulated;
}

ggml_tensor * ada_layer_norm(ggml_context *        context,
                             ggml_tensor *         input,
                             ggml_tensor *         style,
                             const LinearWeights & projection,
                             float                 epsilon) {
    const int64_t channels = input->ne[0];
    ggml_tensor * both     = linear(context, style, projection);
    ggml_tensor * gamma    = ggml_view_1d(context, both, channels, 0);
    ggml_tensor * beta     = ggml_view_1d(context, both, channels, size_t(channels) * ggml_element_size(both));

    ggml_tensor * normalized = ggml_norm(context, input, epsilon);
    // Upstream applies (1 + gamma), so the projection biases toward identity.
    normalized               = ggml_add(context, normalized, ggml_mul(context, normalized, gamma));
    return ggml_add(context, normalized, beta);
}

namespace {

// Nodes emitted per unrolled step, per direction. Kept next to the builder so
// the two stay in step; the test asserts the prediction against the real graph.
constexpr uint64_t kNodesPerStep              = 23;
// The batched input projection plus its bias, per direction, and the final
// concatenation of the two directions.
constexpr uint64_t kNodesPerDirectionPrologue = 2;
constexpr uint64_t kNodesEpilogue             = 1;

// Unrolls one direction, writing each step into a disjoint view of `store`.
void build_direction(ggml_context *               context,
                     ggml_cgraph *                graph,
                     ggml_tensor *                input,
                     const LstmDirectionWeights & weights,
                     ggml_tensor *                zero_state,
                     ggml_tensor *                store,
                     uint32_t                     hidden,
                     uint32_t                     length,
                     bool                         reverse) {
    const int64_t h = static_cast<int64_t>(hidden);

    // One matrix multiply covers the whole sequence: the input contribution to
    // every gate at every step, shaped [4 * hidden, length].
    ggml_tensor * gates = ggml_add(context, ggml_mul_mat(context, weights.weight_ih, input), weights.bias_ih);

    ggml_tensor * state_h = zero_state;
    ggml_tensor * state_c = zero_state;

    for (uint32_t step = 0; step < length; ++step) {
        const uint32_t index = reverse ? length - 1 - step : step;

        ggml_tensor * gate_slice = ggml_view_1d(context, gates, 4 * h, static_cast<size_t>(index) * gates->nb[1]);
        // The four gates share one fused hidden-side multiply.
        ggml_tensor * pre        = ggml_add(
            context, gate_slice, ggml_add(context, ggml_mul_mat(context, weights.weight_hh, state_h), weights.bias_hh));

        const size_t  stride = static_cast<size_t>(h) * sizeof(float);
        ggml_tensor * gate_i = ggml_sigmoid(context, ggml_cont(context, ggml_view_1d(context, pre, h, 0)));
        ggml_tensor * gate_f = ggml_sigmoid(context, ggml_cont(context, ggml_view_1d(context, pre, h, stride)));
        ggml_tensor * gate_g = ggml_tanh(context, ggml_cont(context, ggml_view_1d(context, pre, h, 2 * stride)));
        ggml_tensor * gate_o = ggml_sigmoid(context, ggml_cont(context, ggml_view_1d(context, pre, h, 3 * stride)));

        state_c = ggml_add(context, ggml_mul(context, gate_f, state_c), ggml_mul(context, gate_i, gate_g));
        state_h = ggml_mul(context, gate_o, ggml_tanh(context, state_c));

        // Disjoint write into a persistent store. Accumulating through aliased
        // in-place views instead is silently wrong on CUDA.
        ggml_tensor * slot = ggml_view_1d(context, store, h, static_cast<size_t>(index) * store->nb[1]);
        ggml_build_forward_expand(graph, ggml_cpy(context, state_h, slot));
    }
}

}  // namespace

uint64_t bidirectional_lstm_node_count(uint64_t length) {
    return 2 * (kNodesPerDirectionPrologue + length * kNodesPerStep) + kNodesEpilogue;
}

ggml_tensor * build_bidirectional_lstm(ggml_context *      context,
                                       ggml_cgraph *       graph,
                                       ggml_tensor *       input,
                                       const LstmWeights & weights,
                                       const LstmScratch & scratch,
                                       uint32_t            length) {
    if (context == nullptr || graph == nullptr || input == nullptr || length == 0) {
        return nullptr;
    }
    if (weights.hidden == 0 || weights.input_dim == 0) {
        return nullptr;
    }
    for (const ggml_tensor * tensor :
         { weights.forward.weight_ih, weights.forward.weight_hh, weights.forward.bias_ih, weights.forward.bias_hh,
           weights.reverse.weight_ih, weights.reverse.weight_hh, weights.reverse.bias_ih, weights.reverse.bias_hh,
           scratch.zero_state, scratch.forward_store, scratch.reverse_store }) {
        if (tensor == nullptr) {
            return nullptr;
        }
    }

    const int64_t h = static_cast<int64_t>(weights.hidden);
    if (input->ne[0] != static_cast<int64_t>(weights.input_dim) || input->ne[1] != static_cast<int64_t>(length)) {
        std::fprintf(stderr, "kokoro: LSTM input must be [input_dim, length]\n");
        return nullptr;
    }
    if (scratch.zero_state->ne[0] != h) {
        std::fprintf(stderr, "kokoro: LSTM zero state must be [hidden]\n");
        return nullptr;
    }
    for (const ggml_tensor * store : { scratch.forward_store, scratch.reverse_store }) {
        if (store->ne[0] != h || store->ne[1] != static_cast<int64_t>(length)) {
            std::fprintf(stderr, "kokoro: LSTM output store must be [hidden, length]\n");
            return nullptr;
        }
    }

    build_direction(context, graph, input, weights.forward, scratch.zero_state, scratch.forward_store, weights.hidden,
                    length, false);
    build_direction(context, graph, input, weights.reverse, scratch.zero_state, scratch.reverse_store, weights.hidden,
                    length, true);

    // PyTorch concatenates the two directions along the feature axis.
    ggml_tensor * output = ggml_concat(context, scratch.forward_store, scratch.reverse_store, 0);
    ggml_build_forward_expand(graph, output);
    return output;
}

}  // namespace synth::kokoro
