#pragma once

struct ggml_context;
struct ggml_tensor;

namespace synth::vits {

struct Conv1dWeights;
struct NormWeights;
struct TransposeConv1dWeights;

ggml_tensor * conv1d(ggml_context *        context,
                     ggml_tensor *         input,
                     const Conv1dWeights & weights,
                     int                   padding,
                     int                   dilation = 1);

ggml_tensor * conv1d_without_bias(ggml_context * context,
                                  ggml_tensor *  input,
                                  ggml_tensor *  weight,
                                  int            padding,
                                  int            dilation = 1);

ggml_tensor * depthwise_conv1d(ggml_context *        context,
                               ggml_tensor *         input,
                               const Conv1dWeights & weights,
                               int                   padding,
                               int                   dilation);

// Transposed convolution with the bias left to the caller, expressed as a
// column matrix multiply followed by ggml_col2im_1d rather than the fused
// ggml_conv_transpose_1d. `input` is channel-major [in_channels, length]; the
// result is time-major [length_out, out_channels] with `padding` already
// cropped from both ends. Returns nullptr on a degenerate shape instead of
// letting ggml_col2im_1d abort on it.
ggml_tensor * transpose_conv1d_without_bias(ggml_context * context,
                                            ggml_tensor *  input,
                                            ggml_tensor *  weight,
                                            int            stride,
                                            int            padding);

ggml_tensor * transpose_conv1d(ggml_context *                 context,
                               ggml_tensor *                  input,
                               const TransposeConv1dWeights & weights,
                               int                            stride,
                               int                            padding);

ggml_tensor * layer_norm(ggml_context * context, ggml_tensor * input, const NormWeights & weights, float epsilon);

}  // namespace synth::vits
