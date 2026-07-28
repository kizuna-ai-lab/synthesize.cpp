// The codec decoder's convolution stack.
//
// Layout is channel-major, [channels, length], which is what the rest of this
// family already uses -- the transformer in the middle of the decoder wants
// [hidden, positions], so nothing has to be transposed across that boundary.
//
// Convolutions are built from ggml_im2col and a matrix multiply rather than
// ggml_conv_1d, whose CPU path asserts an F16 kernel and this family's are F32.
// The transposed form uses ggml_col2im_1d rather than the fused
// ggml_conv_transpose_1d, whose CUDA kernel is quadratic in the kernel width;
// see ggml-patches/README.md.
//
// Causality is the recurring hazard: every convolution here sees only the
// present and the past, which is left-only padding. im2col pads symmetrically,
// so each one pads wide and keeps the prefix. An off-by-one there shifts the
// waveform in time and fails nothing.

#include "codec.h"

#include "ggml.h"

namespace synth::qwen3tts {

namespace {

bool bound(const Conv1dWeights & weights) {
    return weights.weight != nullptr && weights.bias != nullptr;
}

// A bias over [channels, length] is per channel, so it broadcasts along the
// length rather than across it.
ggml_tensor * add_channel_bias(ggml_context * context, ggml_tensor * signal, ggml_tensor * bias) {
    return ggml_add(context, signal, ggml_reshape_2d(context, bias, bias->ne[0], 1));
}

// Keeps the first `length` columns of a wider signal, which is what turns a
// symmetrically padded convolution into a causal one.
ggml_tensor * keep_prefix(ggml_context * context, ggml_tensor * signal, int64_t length) {
    if (signal->ne[1] < length) {
        return nullptr;
    }
    if (signal->ne[1] == length) {
        return signal;
    }
    return ggml_cont(context, ggml_view_2d(context, signal, signal->ne[0], length, signal->nb[1], 0));
}

}  // namespace

ggml_tensor * codec_causal_conv1d(ggml_context *        context,
                                  ggml_tensor *         input,
                                  const Conv1dWeights & weights,
                                  int                   dilation) {
    if (context == nullptr || input == nullptr || !bound(weights) || dilation <= 0) {
        return nullptr;
    }
    const int64_t kernel       = weights.weight->ne[0];
    const int64_t in_channels  = weights.weight->ne[1];
    const int64_t out_channels = weights.weight->ne[2];
    const int64_t length       = input->ne[1];
    if (kernel <= 0 || length <= 0 || in_channels != input->ne[0] || out_channels <= 0) {
        return nullptr;
    }

    // Padding on both sides by the whole receptive field puts the causal result
    // in the first `length` columns, followed by the right-padded tail.
    const int     padding    = static_cast<int>((kernel - 1) * dilation);
    ggml_tensor * time_major = ggml_cont(context, ggml_transpose(context, input));
    ggml_tensor * columns    = ggml_im2col(context, weights.weight, time_major, 1, 0, padding, 0, dilation, 0, false,
                                           weights.weight->type);
    ggml_tensor * kernel_2d  = ggml_reshape_2d(context, weights.weight, kernel * in_channels, out_channels);
    ggml_tensor * wide =
        ggml_mul_mat(context, kernel_2d, ggml_reshape_2d(context, columns, columns->ne[0], columns->ne[1]));

    ggml_tensor * causal = keep_prefix(context, wide, length);
    if (causal == nullptr) {
        return nullptr;
    }
    return add_channel_bias(context, causal, weights.bias);
}

ggml_tensor * codec_causal_depthwise_conv1d(ggml_context *        context,
                                            ggml_tensor *         input,
                                            const Conv1dWeights & weights) {
    if (context == nullptr || input == nullptr || !bound(weights)) {
        return nullptr;
    }
    const int64_t kernel   = weights.weight->ne[0];
    const int64_t channels = weights.weight->ne[2];
    const int64_t length   = input->ne[1];
    // One filter per channel: the kernel's input extent is one, not the channel
    // count, and the channel count cannot change across the operator.
    if (kernel <= 0 || length <= 0 || weights.weight->ne[1] != 1 || channels != input->ne[0]) {
        return nullptr;
    }

    const int     padding    = static_cast<int>(kernel - 1);
    ggml_tensor * time_major = ggml_cont(context, ggml_transpose(context, input));
    // Each channel is its own batch entry, which is what keeps the filters from
    // mixing across channels.
    ggml_tensor * batched = ggml_reshape_4d(context, time_major, time_major->ne[0], 1, time_major->ne[1], 1);
    ggml_tensor * columns =
        ggml_im2col(context, weights.weight, batched, 1, 0, padding, 0, 1, 0, false, weights.weight->type);
    ggml_tensor * wide = ggml_mul_mat(context, weights.weight, columns);
    // [1, out_length, channels] down to [out_length, channels], then back to the
    // stack's channel-major layout.
    wide = ggml_reshape_2d(context, wide, wide->ne[1], wide->ne[2]);
    wide = ggml_cont(context, ggml_transpose(context, wide));

    ggml_tensor * causal = keep_prefix(context, wide, length);
    if (causal == nullptr) {
        return nullptr;
    }
    return add_channel_bias(context, causal, weights.bias);
}

ggml_tensor * codec_causal_transpose_conv1d(ggml_context *        context,
                                            ggml_tensor *         input,
                                            const Conv1dWeights & weights,
                                            int                   stride) {
    if (context == nullptr || input == nullptr || !bound(weights) || stride <= 0) {
        return nullptr;
    }
    // ConvTranspose1d stores [in, out, kernel], so ggml reports [kernel, out, in].
    const int64_t kernel       = weights.weight->ne[0];
    const int64_t out_channels = weights.weight->ne[1];
    const int64_t in_channels  = weights.weight->ne[2];
    const int64_t length       = input->ne[1];
    if (kernel < stride || out_channels <= 0 || in_channels != input->ne[0] || length <= 0 ||
        weights.weight->ne[3] != 1) {
        return nullptr;
    }
    if ((length - 1) * stride + kernel < 1) {
        return nullptr;
    }

    // Merging the kernel's leading pair yields exactly the out_channels-major,
    // kernel-minor column order col2im_1d scatters back; the transpose then puts
    // the input channels first so the matrix multiply reduces over them.
    ggml_tensor * columns_weight = ggml_reshape_2d(context, weights.weight, kernel * out_channels, in_channels);
    columns_weight               = ggml_cont(context, ggml_transpose(context, columns_weight));
    ggml_tensor * contiguous     = ggml_is_contiguous(input) ? input : ggml_cont(context, input);
    ggml_tensor * columns        = ggml_mul_mat(context, columns_weight, contiguous);
    // col2im_1d returns [out_length, out_channels].
    ggml_tensor * wide = ggml_col2im_1d(context, columns, stride, static_cast<int>(out_channels), 0);

    // Causality is the crop: the reference drops the trailing `kernel - stride`
    // samples rather than padding symmetrically.
    const int64_t kept = wide->ne[0] - (kernel - stride);
    if (kept <= 0) {
        return nullptr;
    }
    ggml_tensor * causal =
        ggml_cont(context, ggml_view_2d(context, wide, kept, out_channels, wide->nb[1], 0));
    causal = ggml_add(context, causal, ggml_reshape_2d(context, weights.bias, 1, out_channels));
    return ggml_cont(context, ggml_transpose(context, causal));
}

ggml_tensor * codec_snake_beta(ggml_context * context, ggml_tensor * input, const SnakeBetaWeights & weights) {
    if (context == nullptr || input == nullptr || weights.alpha == nullptr || weights.beta == nullptr) {
        return nullptr;
    }
    if (weights.alpha->ne[0] != input->ne[0] || weights.beta->ne[0] != input->ne[0]) {
        return nullptr;
    }
    // Both curves are per channel, and both are stored as logarithms, so neither
    // exponential can be folded into a neighbouring convolution.
    ggml_tensor * alpha = ggml_exp(context, ggml_reshape_2d(context, weights.alpha, weights.alpha->ne[0], 1));
    ggml_tensor * beta  = ggml_exp(context, ggml_reshape_2d(context, weights.beta, weights.beta->ne[0], 1));

    ggml_tensor * squared = ggml_sqr(context, ggml_sin(context, ggml_mul(context, input, alpha)));
    // The reference divides by beta plus a fixed guard, which keeps a zero beta
    // finite. It is part of the function, not a numerical courtesy.
    ggml_tensor * guard = ggml_scale_bias(context, beta, 1.0f, 1e-9f);
    return ggml_add(context, input, ggml_div(context, squared, guard));
}

ggml_tensor * codec_convnext_block(ggml_context * context, ggml_tensor * input, const ConvNeXtWeights & weights) {
    if (context == nullptr || input == nullptr || weights.norm.weight == nullptr || weights.norm.bias == nullptr ||
        weights.pwconv1.weight == nullptr || weights.pwconv1.bias == nullptr ||
        weights.pwconv2.weight == nullptr || weights.pwconv2.bias == nullptr || weights.gamma == nullptr) {
        return nullptr;
    }
    ggml_tensor * hidden = codec_causal_depthwise_conv1d(context, input, weights.dwconv);
    if (hidden == nullptr) {
        return nullptr;
    }

    // The norm and both pointwise layers work across channels, and channels are
    // already this layout's leading axis, so none of it needs transposing.
    hidden = ggml_norm(context, hidden, 1e-6f);
    hidden = ggml_add(context, ggml_mul(context, hidden, weights.norm.weight), weights.norm.bias);
    hidden = ggml_add(context, ggml_mul_mat(context, weights.pwconv1.weight, hidden), weights.pwconv1.bias);
    // nn.GELU() is the exact erf form. ggml_gelu is the tanh approximation, which
    // differs by around 2e-4 here -- small, but larger than everything else in
    // this stack disagrees by, and it compounds over sixty of these blocks.
    hidden = ggml_gelu_erf(context, hidden);
    hidden = ggml_add(context, ggml_mul_mat(context, weights.pwconv2.weight, hidden), weights.pwconv2.bias);
    hidden = ggml_mul(context, hidden, weights.gamma);
    return ggml_add(context, input, hidden);
}

}  // namespace synth::qwen3tts
