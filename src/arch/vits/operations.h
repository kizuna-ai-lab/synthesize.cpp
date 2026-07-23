#pragma once

struct ggml_context;
struct ggml_tensor;

namespace synth::vits {

struct Conv1dWeights;
struct NormWeights;
struct TransposeConv1dWeights;

ggml_tensor * conv1d(ggml_context *       context,
                     ggml_tensor *        input,
                     const Conv1dWeights & weights,
                     int                   padding,
                     int                   dilation = 1);

ggml_tensor * conv1d_without_bias(ggml_context * context,
                                  ggml_tensor *  input,
                                  ggml_tensor *  weight,
                                  int            padding,
                                  int            dilation = 1);

ggml_tensor * depthwise_conv1d(ggml_context *       context,
                               ggml_tensor *        input,
                               const Conv1dWeights & weights,
                               int                   padding,
                               int                   dilation);

ggml_tensor * transpose_conv1d(ggml_context *                context,
                               ggml_tensor *                 input,
                               const TransposeConv1dWeights & weights,
                               int                            stride,
                               int                            padding);

ggml_tensor * layer_norm(ggml_context * context,
                         ggml_tensor *  input,
                         const NormWeights & weights,
                         float         epsilon);

}  // namespace synth::vits
