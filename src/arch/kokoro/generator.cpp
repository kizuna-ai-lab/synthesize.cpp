#include "generator.h"

#include "ggml.h"
#include "operations.h"
#include "weights.h"

namespace synth::kokoro {

namespace {

// The loop's activation and the one before the final projection differ
// upstream: the loop passes an explicit slope and the last one takes PyTorch's
// default. They are not interchangeable.
constexpr float kLoopSlope = 0.1f;
constexpr float kFinalSlope = 0.01f;

// The noise branch's residual blocks carry their dilations in the upstream
// constructor rather than in the checkpoint's configuration, so they are pinned
// here. Their kernel size is read back from the stored weights instead, since
// it differs between the first stages and the last.
constexpr uint32_t kNoiseDilations[]   = { 1, 3, 5 };
constexpr uint32_t kNoiseDilationCount = 3;

// Padding that keeps the length through a dilated odd-kernel convolution.
int same_padding(int64_t kernel, int64_t dilation) {
    return int(dilation * (kernel - 1) / 2);
}

}  // namespace

ggml_tensor * build_adain_resblock1(ggml_context *                context,
                                    ggml_tensor *                 input,
                                    ggml_tensor *                 style,
                                    const AdaINResBlock1Weights & weights,
                                    const uint32_t *              dilations,
                                    uint32_t                      dilation_count,
                                    uint32_t                      kernel_size,
                                    float                         epsilon) {
    if (context == nullptr || input == nullptr || style == nullptr || dilations == nullptr) {
        return nullptr;
    }
    if (dilation_count == 0 || kernel_size == 0 || kernel_size % 2 == 0) {
        return nullptr;
    }
    if (weights.convs1.size() != dilation_count || weights.convs2.size() != dilation_count ||
        weights.adain1.size() != dilation_count || weights.adain2.size() != dilation_count ||
        weights.alpha1.size() != dilation_count || weights.alpha2.size() != dilation_count) {
        return nullptr;
    }

    ggml_tensor * current = input;
    for (uint32_t branch = 0; branch < dilation_count; ++branch) {
        const int dilation = int(dilations[branch]);

        ggml_tensor * residual = adain(context, current, style, weights.adain1[branch].fc, epsilon);
        residual               = snake(context, residual, weights.alpha1[branch]);
        residual = conv1d(context, residual, weights.convs1[branch].weight, weights.convs1[branch].bias, 1,
                          same_padding(kernel_size, dilation), dilation);
        if (residual == nullptr) {
            return nullptr;
        }

        residual = adain(context, residual, style, weights.adain2[branch].fc, epsilon);
        residual = snake(context, residual, weights.alpha2[branch]);
        // The second convolution is never dilated, whatever the branch.
        residual = conv1d(context, residual, weights.convs2[branch].weight, weights.convs2[branch].bias, 1,
                          same_padding(kernel_size, 1), 1);
        if (residual == nullptr) {
            return nullptr;
        }

        current = ggml_add(context, residual, current);
    }
    return current;
}

uint64_t generator_output_frames(const HParams & hparams, uint64_t length) {
    uint64_t total = 1;
    for (uint32_t rate : hparams.istftnet.upsample_rates) {
        total *= rate;
    }
    // The last stage prepends one reflected frame, which is what lines the
    // upsampled features up with the source spectrum.
    return length * total + 1;
}

ggml_tensor * build_generator(ggml_context *           context,
                              ggml_tensor *            input,
                              ggml_tensor *            style,
                              ggml_tensor *            har,
                              const GeneratorWeights & weights,
                              const HParams &          hparams) {
    if (context == nullptr || input == nullptr || style == nullptr || har == nullptr) {
        return nullptr;
    }
    const auto &   rates        = hparams.istftnet.upsample_rates;
    const auto &   kernels      = hparams.istftnet.upsample_kernel_sizes;
    const auto &   block_sizes  = hparams.istftnet.resblock_kernel_sizes;
    const auto &   dilations    = hparams.istftnet.resblock_dilations;
    const uint32_t stage_count  = uint32_t(rates.size());
    const uint32_t branch_count = uint32_t(block_sizes.size());
    if (stage_count == 0 || branch_count == 0 || kernels.size() != stage_count ||
        dilations.size() != branch_count) {
        return nullptr;
    }
    if (weights.ups.size() != stage_count || weights.noise_convs.size() != stage_count ||
        weights.noise_res.size() != stage_count ||
        weights.resblocks.size() != size_t(stage_count) * branch_count) {
        return nullptr;
    }
    if (weights.conv_post.weight == nullptr) {
        return nullptr;
    }

    ggml_tensor * current = input;
    for (uint32_t stage = 0; stage < stage_count; ++stage) {
        current = ggml_leaky_relu(context, current, kLoopSlope, false);

        // The source spectrum is shared by every stage, so each one resamples
        // it to its own rate: a strided convolution while stages remain, and a
        // plain projection on the last, where the rates already agree.
        const bool     last          = stage + 1 == stage_count;
        uint32_t       source_stride = 1;
        for (uint32_t later = stage + 1; later < stage_count; ++later) {
            source_stride *= rates[later];
        }
        const int noise_stride  = last ? 1 : int(source_stride);
        const int noise_padding = last ? 0 : int((source_stride + 1) / 2);

        ggml_tensor * source = conv1d(context, har, weights.noise_convs[stage].weight,
                                      weights.noise_convs[stage].bias, noise_stride, noise_padding, 1);
        if (source == nullptr || weights.noise_res[stage].convs1.empty() ||
            weights.noise_res[stage].convs1.front().weight == nullptr) {
            return nullptr;
        }
        const uint32_t noise_kernel = uint32_t(weights.noise_res[stage].convs1.front().weight->ne[0]);
        source = build_adain_resblock1(context, source, style, weights.noise_res[stage], kNoiseDilations,
                                       kNoiseDilationCount, noise_kernel, hparams.adain_eps);
        if (source == nullptr) {
            return nullptr;
        }

        const int64_t kernel = int64_t(kernels[stage]);
        const int64_t rate   = int64_t(rates[stage]);
        if (kernel < rate || (kernel - rate) % 2 != 0) {
            return nullptr;
        }
        current = transpose_conv1d(context, current, weights.ups[stage].weight, weights.ups[stage].bias, rate,
                                   (kernel - rate) / 2, 0);
        if (current == nullptr) {
            return nullptr;
        }
        if (last) {
            current = reflect_pad_left_1(context, current);
            if (current == nullptr) {
                return nullptr;
            }
        }
        if (current->ne[1] != source->ne[1]) {
            return nullptr;
        }
        current = ggml_add(context, current, source);

        // Every branch sees the same input; upstream averages their outputs.
        ggml_tensor * summed = nullptr;
        for (uint32_t branch = 0; branch < branch_count; ++branch) {
            const AdaINResBlock1Weights & block = weights.resblocks[size_t(stage) * branch_count + branch];
            ggml_tensor *                 term =
                build_adain_resblock1(context, current, style, block, dilations[branch].data(),
                                      uint32_t(dilations[branch].size()), block_sizes[branch], hparams.adain_eps);
            if (term == nullptr) {
                return nullptr;
            }
            summed = summed == nullptr ? term : ggml_add(context, summed, term);
        }
        current = ggml_scale(context, summed, 1.0f / float(branch_count));
    }

    current = ggml_leaky_relu(context, current, kFinalSlope, false);
    const int64_t post_kernel = weights.conv_post.weight->ne[0];
    if (post_kernel % 2 == 0) {
        return nullptr;
    }
    return conv1d(context, current, weights.conv_post.weight, weights.conv_post.bias, 1,
                  same_padding(post_kernel, 1), 1);
}

}  // namespace synth::kokoro
