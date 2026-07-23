#include "operations.h"

#include "ggml.h"
#include "weights.h"

namespace synth::vits {

namespace {

bool packed_kernel_shape(const ggml_tensor * kernel,
                         int64_t             input_channels,
                         int64_t &           kernel_size,
                         int64_t &           output_channels) {
    if (!ggml_is_quantized(kernel->type)) {
        kernel_size     = kernel->ne[0];
        output_channels = kernel->ne[2];
        return kernel->ne[1] == input_channels;
    }
    if (input_channels <= 0 || kernel->ne[0] % input_channels != 0 || kernel->ne[2] != 1 || kernel->ne[3] != 1) {
        return false;
    }
    kernel_size     = kernel->ne[0] / input_channels;
    output_channels = kernel->ne[1];
    return kernel_size > 0 && output_channels > 0;
}

ggml_tensor * conv1d_channels_first(ggml_context * context,
                                    ggml_tensor *  kernel,
                                    ggml_tensor *  data,
                                    int            padding,
                                    int            dilation) {
    int64_t kernel_size     = 0;
    int64_t output_channels = 0;
    if (!packed_kernel_shape(kernel, data->ne[1], kernel_size, output_channels)) {
        return nullptr;
    }
    const bool    packed = ggml_is_quantized(kernel->type);
    ggml_tensor * shape_kernel =
        packed ? ggml_new_tensor_3d(context, GGML_TYPE_F32, kernel_size, data->ne[1], output_channels) : kernel;
    ggml_tensor * columns = ggml_im2col(context, shape_kernel, data, 1, 0, padding, 0, dilation, 0, false,
                                        packed ? GGML_TYPE_F32 : kernel->type);
    ggml_tensor * kernel_2d =
        packed ? kernel : ggml_reshape_2d(context, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]);
    ggml_tensor * output =
        ggml_mul_mat(context, kernel_2d, ggml_reshape_2d(context, columns, columns->ne[0], columns->ne[1]));
    return ggml_reshape_3d(context, output, output_channels, columns->ne[1], 1);
}

}  // namespace

ggml_tensor * conv1d(ggml_context *        context,
                     ggml_tensor *         input,
                     const Conv1dWeights & weights,
                     int                   padding,
                     int                   dilation) {
    int64_t kernel_size     = 0;
    int64_t output_channels = 0;
    if (!packed_kernel_shape(weights.weight, input->ne[0], kernel_size, output_channels)) {
        return nullptr;
    }
    if (kernel_size == 1 && padding == 0 && dilation == 1) {
        ggml_tensor * kernel =
            ggml_is_quantized(weights.weight->type) ?
                weights.weight :
                ggml_reshape_2d(context, weights.weight, weights.weight->ne[1], weights.weight->ne[2]);
        ggml_tensor * output = ggml_mul_mat(context, kernel, input);
        output               = ggml_reshape_3d(context, output, output_channels, input->ne[1], 1);
        ggml_tensor * bias   = ggml_reshape_2d(context, weights.bias, weights.bias->ne[0], 1);
        output               = ggml_add(context, output, bias);
        return ggml_cont(context, output);
    }
    ggml_tensor * time_major = ggml_cont(context, ggml_transpose(context, input));
    ggml_tensor * output     = conv1d_channels_first(context, weights.weight, time_major, padding, dilation);
    if (output == nullptr) {
        return nullptr;
    }
    ggml_tensor * bias = ggml_reshape_2d(context, weights.bias, weights.bias->ne[0], 1);
    output             = ggml_add(context, output, bias);
    return ggml_cont(context, output);
}

ggml_tensor * conv1d_without_bias(ggml_context * context,
                                  ggml_tensor *  input,
                                  ggml_tensor *  weight,
                                  int            padding,
                                  int            dilation) {
    ggml_tensor * time_major = ggml_cont(context, ggml_transpose(context, input));
    ggml_tensor * output     = conv1d_channels_first(context, weight, time_major, padding, dilation);
    if (output == nullptr) {
        return nullptr;
    }
    return ggml_cont(context, output);
}

ggml_tensor * depthwise_conv1d(ggml_context *        context,
                               ggml_tensor *         input,
                               const Conv1dWeights & weights,
                               int                   padding,
                               int                   dilation) {
    ggml_tensor * time_major = ggml_cont(context, ggml_transpose(context, input));
    ggml_tensor * batched    = ggml_reshape_4d(context, time_major, time_major->ne[0], 1, time_major->ne[1], 1);
    ggml_tensor * columns =
        ggml_im2col(context, weights.weight, batched, 1, 0, padding, 0, dilation, 0, false, weights.weight->type);
    ggml_tensor * output = ggml_mul_mat(context, weights.weight, columns);
    output               = ggml_reshape_3d(context, output, output->ne[1], output->ne[2], 1);
    ggml_tensor * bias   = ggml_reshape_2d(context, weights.bias, 1, weights.bias->ne[0]);
    output               = ggml_add(context, output, bias);
    return ggml_cont(context, ggml_transpose(context, output));
}

ggml_tensor * transpose_conv1d(ggml_context *                 context,
                               ggml_tensor *                  input,
                               const TransposeConv1dWeights & weights,
                               int                            stride,
                               int                            padding) {
    if (context == nullptr || input == nullptr || weights.weight == nullptr || weights.bias == nullptr || stride <= 0 ||
        padding < 0) {
        return nullptr;
    }
    ggml_tensor * time_major = ggml_cont(context, ggml_transpose(context, input));
    ggml_tensor * raw        = ggml_conv_transpose_1d(context, weights.weight, time_major, stride, 0, 1);
    raw                      = ggml_reshape_2d(context, raw, raw->ne[0], raw->ne[1]);
    if (raw->ne[0] <= 2LL * padding) {
        return nullptr;
    }
    ggml_tensor * cropped = ggml_view_2d(context, raw, raw->ne[0] - 2LL * padding, raw->ne[1], raw->nb[1],
                                         static_cast<size_t>(padding) * raw->nb[0]);
    cropped               = ggml_cont(context, cropped);
    ggml_tensor * bias    = ggml_reshape_2d(context, weights.bias, 1, weights.bias->ne[0]);
    cropped               = ggml_add(context, cropped, bias);
    return ggml_cont(context, ggml_transpose(context, cropped));
}

ggml_tensor * layer_norm(ggml_context * context, ggml_tensor * input, const NormWeights & weights, float epsilon) {
    ggml_tensor * output = ggml_norm(context, input, epsilon);
    output               = ggml_mul(context, output, weights.weight);
    output               = ggml_add(context, output, weights.bias);
    return output;
}

}  // namespace synth::vits
