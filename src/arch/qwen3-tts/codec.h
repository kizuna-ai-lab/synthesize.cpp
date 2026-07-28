#pragma once

#include "catalog.h"

#include <cstdint>

struct ggml_context;
struct ggml_tensor;

namespace synth::qwen3tts {

// The codec's convolution stack works channel-major -- [channels, length] --
// which is the layout the rest of this family already uses, so the transformer
// in the middle of the decoder needs no transpose around it.
//
// Every convolution in the decode path is causal: it sees only the present and
// the past. That is left-only padding, which ggml's symmetric padding cannot do
// directly, so these helpers pad wide and keep the prefix. Getting it wrong
// shifts the whole waveform in time without failing.

// A causal convolution at stride one. `input` is [in_channels, length]; the
// result is [out_channels, length], the same length the reference produces.
//
// The reference also computes an extra right-hand padding, which is always zero
// at stride one -- every convolution in the decode path is stride one, and the
// only strided operators are the transposed ones below.
ggml_tensor * codec_causal_conv1d(ggml_context *        context,
                                  ggml_tensor *         input,
                                  const Conv1dWeights & weights,
                                  int                   dilation);

// A causal depthwise convolution: one filter per channel, so `weights.weight` is
// [kernel, 1, channels] and the channel count does not change.
ggml_tensor * codec_causal_depthwise_conv1d(ggml_context *        context,
                                            ggml_tensor *         input,
                                            const Conv1dWeights & weights);

// A causal transposed convolution at stride `stride`, kernel `2 * stride` for the
// residual stack and `stride` for the ConvNeXt stages.
//
// The reference convolves and then drops the trailing `kernel - stride` samples,
// which is what makes it causal. Built from a column matrix multiply and
// ggml_col2im_1d rather than the fused ggml_conv_transpose_1d, whose CUDA kernel
// is quadratic in the kernel width; see ggml-patches/README.md.
ggml_tensor * codec_causal_transpose_conv1d(ggml_context *        context,
                                            ggml_tensor *         input,
                                            const Conv1dWeights & weights,
                                            int                   stride);

// SnakeBeta: `x + sin(x * exp(alpha))^2 / (exp(beta) + eps)`.
//
// Both curves are stored as logarithms, so neither may be assumed near one, and
// the exponentials cannot be folded into the neighbouring convolutions.
ggml_tensor * codec_snake_beta(ggml_context * context, ggml_tensor * input, const SnakeBetaWeights & weights);

// One ConvNeXt block: a causal depthwise convolution, then a LayerNorm and a
// two-layer pointwise MLP with GELU applied across channels, scaled by a learned
// per-channel gamma and added back to the input.
ggml_tensor * codec_convnext_block(ggml_context * context, ggml_tensor * input, const ConvNeXtWeights & weights);

}  // namespace synth::qwen3tts
