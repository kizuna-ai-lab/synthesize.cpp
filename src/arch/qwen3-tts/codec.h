#pragma once

#include "catalog.h"
#include "operations.h"

#include <cstdint>

struct ggml_context;
struct ggml_tensor;

namespace synth::qwen3tts {

struct HParams;

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
ggml_tensor * codec_causal_depthwise_conv1d(ggml_context * context, ggml_tensor * input, const Conv1dWeights & weights);

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

// Turns one residual quantizer's codes back into a signal: a table lookup per
// level, summed, then the level width projected back up.
//
// `codes` is an I32 [frames, levels] with each level's ids contiguous, so a level
// is a row. Returns [output_width, frames].
ggml_tensor * codec_quantizer_decode(ggml_context *                context,
                                     const CodecQuantizerWeights & weights,
                                     ggml_tensor *                 codes);

// The mask the codec's transformer attends under: causal *and* limited to a
// sliding window, so a query at position i reads keys in (i - window, i]. Returns
// an F32 [frames, frames] the caller fills; `window` of zero means causal only.
//
// The window is why this cannot reuse the talker's mask: at 72 frames it is
// under six seconds, and an utterance is routinely longer.
void codec_fill_sliding_window_mask(float * mask, int64_t frames, int64_t window);

// One codec transformer layer. This is *not* the shared Qwen3 block: it has no
// per-head norms, and each residual branch is scaled by a learned per-channel
// vector before being added back.
ggml_tensor * codec_transformer_layer(ggml_context *                       context,
                                      ggml_tensor *                        input,
                                      ggml_tensor *                        position_ids,
                                      ggml_tensor *                        mask,
                                      const CodecTransformerLayerWeights & weights,
                                      const AttentionShape &               shape);

// The whole decoder: codes in, one channel of audio out, [1, frames * hop].
//
// `codes` is an I32 [frames, code_group_count] as above, `position_ids` an I32
// [frames], and `mask` the sliding-window mask. Returns nullptr rather than
// aborting on anything it cannot build.
ggml_tensor * build_codec_decoder(ggml_context *              context,
                                  ggml_tensor *               codes,
                                  ggml_tensor *               position_ids,
                                  ggml_tensor *               mask,
                                  const CodecDecoderWeights & weights,
                                  const HParams &             hparams);

}  // namespace synth::qwen3tts
