#pragma once

// catalog.h carries the weight structs and weights.h the HParams the decoder's
// geometry reads -- the post-Task-4 roles.
#include "catalog.h"
#include "weights.h"

#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace synth::omnivoice {

// The Higgs Audio V2 decode path. Layout is channel-major [channels, length]
// throughout, the qwen3-tts codec convention. NOTHING here is causal: DAC pads
// symmetrically, so there is no keep-prefix crop anywhere -- the recurring
// hazard of the qwen3 codec does not exist in this one, and porting its crops
// here would shift the waveform in time and fail nothing.

// Plain Snake, alpha only: x + sin^2(alpha * x) / (alpha + 1e-9). Alpha is
// stored linearly in a [1, channels, 1] tensor (not as a logarithm -- the
// qwen3 SnakeBeta contrast), and the 1e-9 guard is the reference's own
// (transformers Snake1d), part of the function rather than a courtesy.
ggml_tensor * codec_snake(ggml_context * context, ggml_tensor * input, const SnakeWeights & weights);

// Symmetric-padding convolution via im2col + matmul (ggml_conv_1d's CPU path
// asserts an F16 kernel and this family's are F32). With DAC's odd kernels and
// pad = dilation * (kernel - 1) / 2 the output length equals the input length.
//
// `stride` defaults to 1 -- every decoder call site is stride-1 (DAC's own
// convolutions all are; only its ConvTranspose1d resamples, and that is
// codec_transpose_conv1d's own stride parameter below) -- and is Task 12's
// addition for the acoustic ENCODER's own resampling convolutions, DAC's
// mirror of the decoder's transposed one: a plain strided forward Conv1d
// rather than a transposed one, since the encoder downsamples going forward
// through time instead of scattering backward into it.
ggml_tensor * codec_conv1d(ggml_context *        context,
                           ggml_tensor *         input,
                           const Conv1dWeights & weights,
                           int                   dilation,
                           int                   padding,
                           int                   stride = 1);

// Transposed convolution via mul_mat + ggml_col2im_1d (the VITS recipe). Torch
// semantics: out = (L-1)*stride - 2*padding + kernel + output_padding. The
// scatter is computed unpadded and the torch padding becomes a view crop of
// `padding` rows from the left with `padding - output_padding` from the right,
// which is what makes Higgs's output_padding = stride % 2 representable
// without a new operator. Requires 0 <= output_padding <= padding.
ggml_tensor * codec_transpose_conv1d(ggml_context *        context,
                                     ggml_tensor *         input,
                                     const Conv1dWeights & weights,
                                     int                   stride,
                                     int                   padding,
                                     int                   output_padding);

// Residual dequantization: every level looks its codes up in its own codebook
// and projects them out through a biased Linear; the levels are summed in
// ascending order (float addition is not associative and the reference sums
// 0..7). `codes` is I32 [frames, num_quantizers], level-major rows.
//
// Every code must already be a real index into the codebook. Nothing here can
// tell a mask id from a code, so validate_code_grid in codec-host.h is the
// guard, and it runs on the host before the graph is built.
ggml_tensor * codec_rvq_decode(ggml_context *                           context,
                               const std::vector<RvqQuantizerWeights> & quantizers,
                               ggml_tensor *                            codes);

// The whole decode: RVQ sum -> fc2 (concat width -> acoustic width) -> DAC
// conv1 -> per-ratio blocks (snake, transposed conv, three residual units at
// dilations 1/3/9) -> snake -> conv2 to mono. Higgs removes DAC's final tanh,
// so the raw convolution output IS the waveform: no clamp, no activation, and
// no peak normalization -- that last one is the caller's separate branch.
// Returns a 1-D [frames * hop] tensor.
//
// `out_latent` and `out_acoustic`, when non-null, receive the RVQ sum and the
// fc2 output. The unit test reads them so a mismatch names a stage instead of
// naming the waveform; nothing else reads them. Both are meaningful ONLY when
// this returns non-null.
ggml_tensor * build_codec_decoder(ggml_context *       context,
                                  ggml_tensor *        codes,
                                  const ModelWeights & weights,
                                  const HParams &      hparams,
                                  ggml_tensor **       out_latent   = nullptr,
                                  ggml_tensor **       out_acoustic = nullptr);

}  // namespace synth::omnivoice
