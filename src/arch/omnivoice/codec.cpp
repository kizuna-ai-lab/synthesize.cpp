// The Higgs Audio V2 codec's decode graph.
//
// Layout is channel-major, [channels, length], which is what the qwen3-tts
// codec already uses; the shapes and the recipes (im2col + matmul for the
// forward convolution, mul_mat + col2im_1d for the transposed one) are that
// family's, because ggml_conv_1d's CPU path asserts an F16 kernel and this
// family's are F32, and the fused ggml_conv_transpose_1d's CUDA kernel is
// quadratic in the kernel width.
//
// What is NOT that family's is causality. Every convolution in the qwen3 codec
// sees only the present and the past, so each one pads wide and keeps the
// prefix. DAC pads symmetrically -- conv k7 pad 3, res-unit conv k7 dilation d
// pad 3d, conv_t k=2s pad ceil(s/2) -- and a keep-prefix crop copied over from
// there would shift the waveform in time and fail no shape check.
//
// Two Higgs adjustments to stock DAC, both from
// HiggsAudioV2TokenizerModel._adjust_dac_decoder: output_padding = stride % 2
// on every ConvTranspose1d, which is what keeps an odd ratio at exactly
// length * stride; and the final Tanh replaced by Identity, which makes the
// last convolution's raw output the waveform.

#include "arch/omnivoice/codec.h"

#include "arch/omnivoice/catalog.h"
#include "arch/omnivoice/weights.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <iterator>

namespace synth::omnivoice {

namespace {

// The DAC residual stack's fixed dilations, three units per block.
constexpr int kDacDilations[3] = { 1, 3, 9 };

bool bound(const Conv1dWeights & weights) {
    return weights.weight != nullptr && weights.bias != nullptr;
}

// A bias over [channels, length] is per channel, so it broadcasts along the
// length rather than across it.
ggml_tensor * add_channel_bias(ggml_context * context, ggml_tensor * signal, ggml_tensor * bias) {
    return ggml_add(context, signal, ggml_reshape_2d(context, bias, bias->ne[0], 1));
}

}  // namespace

ggml_tensor * codec_snake(ggml_context * context, ggml_tensor * input, const SnakeWeights & weights) {
    if (context == nullptr || input == nullptr || weights.alpha == nullptr) {
        return nullptr;
    }
    // Alpha is stored [1, channels, 1]; one value per channel. The rank is part
    // of the catalog's contract, and a flat curve of the same length would
    // reshape fine here but broadcast differently upstream, so both the width
    // and the emptiness of the other axes are checked.
    if (weights.alpha->ne[1] != input->ne[0] || ggml_nelements(weights.alpha) != weights.alpha->ne[1]) {
        return nullptr;
    }
    ggml_tensor * per_channel = ggml_reshape_1d(context, weights.alpha, weights.alpha->ne[1]);
    ggml_tensor * scaled      = ggml_mul(context, input, per_channel);
    ggml_tensor * sine        = ggml_sin(context, scaled);
    ggml_tensor * squared     = ggml_mul(context, sine, sine);
    // The reference divides by alpha plus a fixed guard (Snake1d's 1e-9). It is
    // part of the function, not a numerical courtesy.
    ggml_tensor * guard       = ggml_scale_bias(context, per_channel, 1.0f, 1e-9f);
    return ggml_add(context, input, ggml_div(context, squared, guard));
}

ggml_tensor * codec_conv1d(ggml_context *        context,
                           ggml_tensor *         input,
                           const Conv1dWeights & weights,
                           int                   dilation,
                           int                   padding) {
    if (context == nullptr || input == nullptr || !bound(weights) || dilation <= 0 || padding < 0) {
        return nullptr;
    }
    const int64_t kernel       = weights.weight->ne[0];
    const int64_t in_channels  = weights.weight->ne[1];
    const int64_t out_channels = weights.weight->ne[2];
    const int64_t length       = input->ne[1];
    if (kernel <= 0 || length <= 0 || in_channels != input->ne[0] || out_channels <= 0 ||
        weights.bias->ne[0] != out_channels) {
        return nullptr;
    }
    ggml_tensor * time_major = ggml_cont(context, ggml_transpose(context, input));
    ggml_tensor * columns =
        ggml_im2col(context, weights.weight, time_major, 1, 0, padding, 0, dilation, 0, false, weights.weight->type);
    ggml_tensor * kernel_2d = ggml_reshape_2d(context, weights.weight, kernel * in_channels, out_channels);
    ggml_tensor * signal =
        ggml_mul_mat(context, kernel_2d, ggml_reshape_2d(context, columns, columns->ne[0], columns->ne[1]));
    return add_channel_bias(context, signal, weights.bias);
}

ggml_tensor * codec_transpose_conv1d(ggml_context *        context,
                                     ggml_tensor *         input,
                                     const Conv1dWeights & weights,
                                     int                   stride,
                                     int                   padding,
                                     int                   output_padding) {
    if (context == nullptr || input == nullptr || !bound(weights) || stride <= 0 || padding < 0 || output_padding < 0 ||
        output_padding > padding) {
        return nullptr;
    }
    // ConvTranspose1d stores [in, out, kernel], so ggml reports [kernel, out, in].
    const int64_t kernel       = weights.weight->ne[0];
    const int64_t out_channels = weights.weight->ne[1];
    const int64_t in_channels  = weights.weight->ne[2];
    const int64_t length       = input->ne[1];
    if (kernel < stride || out_channels <= 0 || in_channels != input->ne[0] || length <= 0 ||
        weights.weight->ne[3] != 1 || weights.bias->ne[0] != out_channels) {
        return nullptr;
    }
    const int64_t out_length = (length - 1) * stride + kernel - 2 * int64_t(padding) + output_padding;
    if (out_length < 1) {
        return nullptr;
    }
    // Merging the kernel's leading pair yields exactly the out_channels-major,
    // kernel-minor column order col2im_1d scatters back; the transpose then
    // puts the input channels first so the matrix multiply reduces over them.
    ggml_tensor * columns_weight = ggml_reshape_2d(context, weights.weight, kernel * out_channels, in_channels);
    columns_weight               = ggml_cont(context, ggml_transpose(context, columns_weight));
    ggml_tensor * contiguous     = ggml_is_contiguous(input) ? input : ggml_cont(context, input);
    ggml_tensor * columns        = ggml_mul_mat(context, columns_weight, contiguous);
    // The FULL scatter, [(length-1)*stride + kernel, out_channels]; the torch
    // padding crops `padding` from the left and `padding - output_padding`
    // from the right of it. Both crops are of real scattered samples, which is
    // why output_padding may not exceed padding: torch's own col2im writes
    // into an output that already includes the output_padding tail, so those
    // samples are the scatter's, not zeros.
    ggml_tensor * wide           = ggml_col2im_1d(context, columns, stride, static_cast<int>(out_channels), 0);
    ggml_tensor * cropped =
        ggml_view_2d(context, wide, out_length, out_channels, wide->nb[1], size_t(padding) * wide->nb[0]);
    cropped = ggml_cont(context, cropped);
    cropped = ggml_add(context, cropped, ggml_reshape_2d(context, weights.bias, 1, out_channels));
    return ggml_cont(context, ggml_transpose(context, cropped));
}

ggml_tensor * codec_rvq_decode(ggml_context *                           context,
                               const std::vector<RvqQuantizerWeights> & quantizers,
                               ggml_tensor *                            codes) {
    if (context == nullptr || codes == nullptr || codes->type != GGML_TYPE_I32 || quantizers.empty()) {
        return nullptr;
    }
    if (codes->ne[1] != int64_t(quantizers.size()) || codes->ne[0] <= 0 || !ggml_is_contiguous(codes)) {
        return nullptr;
    }
    ggml_tensor * total = nullptr;
    for (size_t level = 0; level < quantizers.size(); ++level) {
        const RvqQuantizerWeights & quantizer = quantizers[level];
        if (quantizer.codebook == nullptr || quantizer.output_proj.weight == nullptr ||
            quantizer.output_proj.bias == nullptr) {
            return nullptr;
        }
        // A level whose projection does not span its own codebook, or whose
        // width differs from the levels already summed, is a package that
        // resolved inconsistently; ggml would abort on the assertion instead.
        if (quantizer.output_proj.weight->ne[0] != quantizer.codebook->ne[0] ||
            quantizer.output_proj.weight->ne[1] != quantizer.output_proj.bias->ne[0] ||
            (total != nullptr && total->ne[0] != quantizer.output_proj.weight->ne[1])) {
            return nullptr;
        }
        ggml_tensor * ids       = ggml_view_1d(context, codes, codes->ne[0], size_t(level) * codes->nb[1]);
        ggml_tensor * rows      = ggml_get_rows(context, quantizer.codebook, ids);
        // Unlike most RVQ ports the projection is a biased Linear, not a
        // kernel-one convolution.
        ggml_tensor * projected = ggml_mul_mat(context, quantizer.output_proj.weight, rows);
        projected               = add_channel_bias(context, projected, quantizer.output_proj.bias);
        total                   = total == nullptr ? projected : ggml_add(context, total, projected);
    }
    return total;
}

ggml_tensor * build_codec_decoder(ggml_context *       context,
                                  ggml_tensor *        codes,
                                  const ModelWeights & weights,
                                  const HParams &      hparams,
                                  ggml_tensor **       out_latent,
                                  ggml_tensor **       out_acoustic) {
    if (context == nullptr || codes == nullptr || codes->type != GGML_TYPE_I32 ||
        codes->ne[1] != int64_t(hparams.audio.num_codebooks) || codes->ne[0] <= 0 || weights.fc2.weight == nullptr ||
        weights.fc2.bias == nullptr) {
        return nullptr;
    }
    const AcousticDecoderWeights & decoder = weights.acoustic_decoder;
    // Every block upsamples by the ratio at its own index, so a decoder that
    // resolved a different number of blocks than the package declares ratios
    // would read past the list rather than merely produce the wrong length.
    if (decoder.blocks.empty() || decoder.blocks.size() != hparams.codec.upsampling_ratios.size()) {
        return nullptr;
    }
    const int64_t frames = codes->ne[0];

    ggml_tensor * latent = codec_rvq_decode(context, weights.quantizers, codes);
    if (latent == nullptr) {
        return nullptr;
    }
    // fc2 brings the dequantized concat-width latent down to the acoustic
    // width the DAC decoder consumes.
    if (weights.fc2.weight->ne[0] != latent->ne[0] || weights.fc2.weight->ne[1] != weights.fc2.bias->ne[0]) {
        return nullptr;
    }
    ggml_tensor * acoustic = ggml_mul_mat(context, weights.fc2.weight, latent);
    acoustic               = add_channel_bias(context, acoustic, weights.fc2.bias);

    ggml_tensor * hidden = codec_conv1d(context, acoustic, decoder.conv1, 1, 3);  // kernel 7, pad 3
    if (hidden == nullptr) {
        return nullptr;
    }
    for (size_t block_index = 0; block_index < decoder.blocks.size(); ++block_index) {
        const AcousticDecoderBlock & block  = decoder.blocks[block_index];
        const int                    stride = int(hparams.codec.upsampling_ratios[block_index]);
        if (stride <= 0 || block.res_units.size() != std::size(kDacDilations)) {
            return nullptr;
        }
        hidden = codec_snake(context, hidden, block.snake1);
        if (hidden == nullptr) {
            return nullptr;
        }
        // kernel 2*stride, padding ceil(stride/2), and Higgs's adjustment:
        // output_padding = stride % 2, which keeps every stage at exactly
        // length * stride.
        hidden = codec_transpose_conv1d(context, hidden, block.conv_t1, stride, (stride + 1) / 2, stride % 2);
        if (hidden == nullptr) {
            return nullptr;
        }
        for (size_t unit_index = 0; unit_index < block.res_units.size(); ++unit_index) {
            const DacResidualUnit & unit     = block.res_units[unit_index];
            const int               dilation = kDacDilations[unit_index];
            ggml_tensor *           branch   = codec_snake(context, hidden, unit.snake1);
            if (branch == nullptr) {
                return nullptr;
            }
            branch = codec_conv1d(context, branch, unit.conv1, dilation, 3 * dilation);  // kernel 7
            if (branch == nullptr) {
                return nullptr;
            }
            branch = codec_snake(context, branch, unit.snake2);
            if (branch == nullptr) {
                return nullptr;
            }
            branch = codec_conv1d(context, branch, unit.conv2, 1, 0);  // kernel 1
            if (branch == nullptr) {
                return nullptr;
            }
            // DAC's residual unit is length-preserving because its padding is
            // symmetric, and upstream's own forward crops the skip connection
            // when it is not. Nothing here should ever need that crop, so a
            // branch that came back a different length means the padding is
            // wrong -- refuse rather than let ggml_add abort on it.
            if (branch->ne[0] != hidden->ne[0] || branch->ne[1] != hidden->ne[1]) {
                return nullptr;
            }
            hidden = ggml_add(context, hidden, branch);
        }
    }
    hidden = codec_snake(context, hidden, decoder.snake1);
    if (hidden == nullptr) {
        return nullptr;
    }
    ggml_tensor * wave = codec_conv1d(context, hidden, decoder.conv2, 1, 3);  // kernel 7 -> mono
    if (wave == nullptr || wave->ne[0] != 1 || wave->ne[1] != frames * int64_t(hparams.codec.hop_length)) {
        return nullptr;
    }
    if (out_latent != nullptr) {
        *out_latent = latent;
    }
    if (out_acoustic != nullptr) {
        *out_acoustic = acoustic;
    }
    // Higgs replaces DAC's final tanh with identity: the raw output is the
    // waveform. No clamp -- qwen3-tts clamps because ITS reference does. The
    // signal is mono, so [1, samples] already flattens sample-major and only
    // needs its rank dropped.
    ggml_tensor * flat = ggml_is_contiguous(wave) ? wave : ggml_cont(context, wave);
    return ggml_reshape_1d(context, flat, wave->ne[1]);
}

}  // namespace synth::omnivoice
