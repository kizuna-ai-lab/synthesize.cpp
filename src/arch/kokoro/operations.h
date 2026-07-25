#pragma once

#include <cstdint>

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

namespace synth::kokoro {

// Every stage keeps tensors as [features, time], matching the VITS family, so
// there is one layout in the codebase. Convolution transposes internally where
// GGML needs the time axis first.

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

// Convolution that preserves F32 weights.
//
// ggml_conv_1d asserts on an F32 kernel because its im2col buffer is F16, so
// this uses the same im2col-and-matrix-multiply decomposition the VITS family
// uses rather than downcasting the accuracy artifact. `bias` may be null.
ggml_tensor * conv1d(ggml_context * context,
                     ggml_tensor *  input,
                     ggml_tensor *  weight,
                     ggml_tensor *  bias,
                     int            stride,
                     int            padding,
                     int            dilation);

// The logical kernel width of a convolution weight.
//
// A block-quantized kernel is stored flattened to [kernel * in_channels,
// out_channels], so its leading extent is no longer the kernel width. Anything
// that needs the width must ask for it rather than read ne[0]. Returns 0 when
// the weight and the channel count disagree.
int64_t conv_kernel_size(const ggml_tensor * weight, int64_t in_channels);

// Transposed 1-D convolution across all channels, the generator's upsampler.
//
// GGML's own transposed convolution has no padding, so this reuses the scatter
// construction the depthwise version documents: interleave `stride - 1` zeros,
// widen the ends, then accumulate one shifted matrix multiply per kernel tap.
// Reading the taps back to front is what supplies the definition's kernel
// reversal, so the stored weights stay exactly as the checkpoint holds them.
//
// `input` is [in_channels, length] and `weight` is [kernel, out_channels,
// in_channels], the layout GGML reports for PyTorch's [in, out, kernel].
// `bias` may be null.
ggml_tensor * transpose_conv1d(ggml_context * context,
                               ggml_tensor *  input,
                               ggml_tensor *  weight,
                               ggml_tensor *  bias,
                               int64_t        stride,
                               int64_t        padding,
                               int64_t        output_padding);

// Output length of transpose_conv1d.
int64_t transpose_conv1d_length(int64_t length,
                                int64_t kernel,
                                int64_t stride,
                                int64_t padding,
                                int64_t output_padding);

// Prepends one frame by reflecting across the first one, so the new leading
// frame is the second frame of the input. This is ReflectionPad1d((1, 0)), which
// the generator applies before its last residual sum.
ggml_tensor * reflect_pad_left_1(ggml_context * context, ggml_tensor * input);

// Instance norm: each channel is normalized across time, which is a different
// axis from layer norm's normalization across channels.
ggml_tensor * instance_norm(ggml_context * context, ggml_tensor * input, float epsilon);

// AdaIN: instance norm without learned affine, then a
// scale and shift projected from the style vector. As with the adaptive layer
// norm, upstream applies (1 + gamma) so a zero projection is the identity.
ggml_tensor * adain(ggml_context *        context,
                    ggml_tensor *         input,
                    ggml_tensor *         style,
                    const LinearWeights & projection,
                    float                 epsilon);

// Nearest-neighbour doubling along the time axis.
ggml_tensor * upsample_nearest_2x(ggml_context * context, ggml_tensor * input);

// The tanh form of GELU, built from primitives rather than taken from
// ggml_gelu.
//
// GGML's CPU GELU reads a half-precision lookup table, which costs about 1e-4
// against the reference and compounds across PL-BERT's twelve replayed layers.
// That error reaches F0, and the harmonic source's phase accumulator amplifies
// an F0 difference into radians by the end of an utterance, so it is worth the
// extra nodes to evaluate the closed form in single precision instead. This is
// the same reasoning that keeps convolution off ggml_conv_1d.
ggml_tensor * gelu_tanh(ggml_context * context, ggml_tensor * input);

// x + (1/a) * sin(a * x)^2, the Snake activation, with one alpha per channel.
ggml_tensor * snake(ggml_context * context, ggml_tensor * input, ggml_tensor * alpha);

// Depthwise transposed 1-D convolution, one filter per channel.
//
// GGML's transposed convolution has no grouping and forbids internal padding,
// so this is built from the scatter definition instead. Inserting `stride - 1`
// zeros between input samples turns the scatter into a plain correlation, and
// each kernel tap then contributes one shifted, per-channel scaled copy. That
// avoids reversing the stored kernel, so the package stays byte-faithful to the
// checkpoint and no weight preparation happens at load time.
//
// `input` is [channels, length] and `weight` is [kernel, 1, channels], the
// layout GGML reports for PyTorch's [channels, 1, kernel]. `bias` may be null.
ggml_tensor * depthwise_transpose_conv1d(ggml_context * context,
                                         ggml_tensor *  input,
                                         ggml_tensor *  weight,
                                         ggml_tensor *  bias,
                                         int64_t        stride,
                                         int64_t        padding,
                                         int64_t        output_padding);

// Output length of depthwise_transpose_conv1d.
int64_t depthwise_transpose_conv1d_length(int64_t length,
                                          int64_t kernel,
                                          int64_t stride,
                                          int64_t padding,
                                          int64_t output_padding);

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
