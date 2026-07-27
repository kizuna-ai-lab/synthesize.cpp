#include "waveform-decoder.h"

#include "ggml.h"
#include "operations.h"
#include "voice-conditioning.h"
#include "weights.h"

#include <cmath>
#include <cstdio>
#include <limits>

namespace synth::vits {

namespace {

bool conv_is_present(const Conv1dWeights & weights) {
    return weights.weight != nullptr && weights.bias != nullptr;
}

bool decoder_catalog_is_valid(const DecoderWeights & weights, const HParams & hparams) {
    if (!conv_is_present(weights.pre) || weights.post_weight == nullptr ||
        weights.stages.size() != hparams.decoder_upsample_rates.size()) {
        return false;
    }
    for (const DecoderStageWeights & stage : weights.stages) {
        if (stage.transpose_conv.weight == nullptr || stage.transpose_conv.bias == nullptr ||
            stage.resblocks.size() != hparams.decoder_resblock_kernel_sizes.size()) {
            return false;
        }
        for (size_t branch = 0; branch < stage.resblocks.size(); ++branch) {
            const DecoderResBlockWeights & block       = stage.resblocks[branch];
            const size_t                   layer_count = hparams.decoder_resblock_dilations[branch].size();
            if (block.conv1.size() != layer_count || block.conv2.size() != layer_count) {
                return false;
            }
            for (size_t layer = 0; layer < layer_count; ++layer) {
                if (!conv_is_present(block.conv1[layer]) || !conv_is_present(block.conv2[layer])) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool decoder_request_is_valid(const HParams & hparams, int64_t frame_count) {
    if (frame_count <= 0 || hparams.inter_channels == 0 || hparams.decoder_resblock_kernel_sizes.empty() ||
        hparams.decoder_resblock_kernel_sizes.size() != hparams.decoder_resblock_dilations.size() ||
        hparams.decoder_upsample_rates.empty() ||
        hparams.decoder_upsample_rates.size() != hparams.decoder_upsample_kernel_sizes.size() ||
        hparams.decoder_initial_channels == 0 || !std::isfinite(hparams.decoder_leaky_relu_slope) ||
        hparams.decoder_leaky_relu_slope <= 0.0f || hparams.decoder_leaky_relu_slope > 1.0f ||
        hparams.hop_length == 0 ||
        static_cast<uint64_t>(frame_count) >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / hparams.hop_length ||
        static_cast<uint64_t>(frame_count) * hparams.hop_length > hparams.max_output_frames) {
        return false;
    }
    for (size_t branch = 0; branch < hparams.decoder_resblock_kernel_sizes.size(); ++branch) {
        const uint32_t kernel = hparams.decoder_resblock_kernel_sizes[branch];
        if (kernel == 0 || kernel % 2 == 0 || kernel > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
            hparams.decoder_resblock_dilations[branch].size() != 3) {
            return false;
        }
        for (uint32_t dilation : hparams.decoder_resblock_dilations[branch]) {
            const uint64_t padding = static_cast<uint64_t>(kernel - 1) * dilation / 2;
            if (dilation == 0 || dilation > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
                padding > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
                return false;
            }
        }
    }
    uint64_t channels   = hparams.decoder_initial_channels;
    uint64_t hop_length = 1;
    for (size_t stage = 0; stage < hparams.decoder_upsample_rates.size(); ++stage) {
        const uint32_t rate   = hparams.decoder_upsample_rates[stage];
        const uint32_t kernel = hparams.decoder_upsample_kernel_sizes[stage];
        if (rate == 0 || kernel < rate || (kernel - rate) % 2 != 0 ||
            rate > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
            kernel > static_cast<uint32_t>(std::numeric_limits<int>::max()) || channels < 2 || channels % 2 != 0 ||
            hop_length > std::numeric_limits<uint32_t>::max() / rate) {
            return false;
        }
        channels /= 2;
        hop_length *= rate;
    }
    return hop_length == hparams.hop_length;
}

ggml_tensor * conv1d_time_major_f32(ggml_context * context,
                                    ggml_tensor *  kernel,
                                    ggml_tensor *  input,
                                    int            padding,
                                    int            dilation) {
    const bool packed = ggml_is_quantized(kernel->type);
    if (input->ne[1] <= 0 || (packed && kernel->ne[0] % input->ne[1] != 0) ||
        (!packed && kernel->ne[1] != input->ne[1])) {
        return nullptr;
    }
    const int64_t kernel_size     = packed ? kernel->ne[0] / input->ne[1] : kernel->ne[0];
    const int64_t output_channels = packed ? kernel->ne[1] : kernel->ne[2];
    ggml_tensor * shape_kernel =
        packed ? ggml_new_tensor_3d(context, GGML_TYPE_F32, kernel_size, input->ne[1], output_channels) : kernel;
    ggml_tensor * columns = ggml_im2col(context, shape_kernel, input, 1, 0, padding, 0, dilation, 0, false,
                                        packed ? GGML_TYPE_F32 : kernel->type);
    ggml_tensor * kernel_2d =
        packed ? kernel : ggml_reshape_2d(context, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]);
    ggml_tensor * output =
        ggml_mul_mat(context, kernel_2d, ggml_reshape_2d(context, columns, columns->ne[0], columns->ne[1]));
    output = ggml_reshape_3d(context, output, output_channels, columns->ne[1], 1);
    return ggml_cont(context, ggml_transpose(context, output));
}

ggml_tensor * conv1d_time_major(ggml_context *        context,
                                ggml_tensor *         input,
                                const Conv1dWeights & weights,
                                int                   padding,
                                int                   dilation = 1) {
    ggml_tensor * output = conv1d_time_major_f32(context, weights.weight, input, padding, dilation);
    ggml_tensor * bias   = ggml_reshape_2d(context, weights.bias, 1, weights.bias->ne[0]);
    return ggml_add(context, output, bias);
}

ggml_tensor * transpose_conv1d_time_major(ggml_context *                 context,
                                          ggml_tensor *                  input,
                                          const TransposeConv1dWeights & weights,
                                          int                            stride,
                                          int                            padding) {
    if (context == nullptr || input == nullptr || weights.bias == nullptr) {
        return nullptr;
    }
    // The shared helper reduces over the input channels, so hand it the
    // channel-major view of this stage's time-major state.
    ggml_tensor * channel_major = ggml_cont(context, ggml_transpose(context, input));
    ggml_tensor * signal = transpose_conv1d_without_bias(context, channel_major, weights.weight, stride, padding);
    if (signal == nullptr) {
        return nullptr;
    }
    ggml_tensor * bias = ggml_reshape_2d(context, weights.bias, 1, weights.bias->ne[0]);
    return ggml_add(context, signal, bias);
}

ggml_tensor * resblock_time_major(ggml_context *                 context,
                                  ggml_tensor *                  input,
                                  const DecoderResBlockWeights & weights,
                                  uint32_t                       kernel,
                                  const std::vector<uint32_t> &  dilations,
                                  float                          slope) {
    ggml_tensor * state = input;
    for (size_t layer = 0; layer < dilations.size(); ++layer) {
        ggml_tensor * residual = state;
        state                  = ggml_leaky_relu(context, state, slope, false);
        const int padding      = static_cast<int>(static_cast<uint64_t>(kernel - 1) * dilations[layer] / 2);
        state = conv1d_time_major(context, state, weights.conv1[layer], padding, static_cast<int>(dilations[layer]));
        state = ggml_leaky_relu(context, state, slope, false);
        state = conv1d_time_major(context, state, weights.conv2[layer], static_cast<int>((kernel - 1) / 2));
        state = ggml_add(context, state, residual);
    }
    return state;
}

}  // namespace

WaveformDecoderGraph build_waveform_decoder_graph(ggml_context *         context,
                                                  const DecoderWeights & weights,
                                                  const HParams &        hparams,
                                                  int64_t                frame_count) {
    const VoiceWeights no_voice;
    return build_waveform_decoder_graph(context, weights, no_voice, hparams, UINT32_MAX, frame_count);
}

WaveformDecoderGraph build_waveform_decoder_graph(ggml_context *         context,
                                                  const DecoderWeights & weights,
                                                  const VoiceWeights &   voice_weights,
                                                  const HParams &        hparams,
                                                  uint32_t               speaker_index,
                                                  int64_t                frame_count) {
    WaveformDecoderGraph result;
    if (context == nullptr || !decoder_request_is_valid(hparams, frame_count) ||
        !decoder_catalog_is_valid(weights, hparams) ||
        (hparams.conditioning_channels > 0 &&
         (weights.conditioning.weight == nullptr || weights.conditioning.bias == nullptr))) {
        std::fprintf(stderr, "vits: invalid waveform-decoder graph request\n");
        return result;
    }
    const VoiceConditioning voice = build_voice_conditioning(context, voice_weights, hparams, speaker_index);
    if (!voice.valid) {
        std::fprintf(stderr, "vits: invalid waveform-decoder Voice conditioning\n");
        return result;
    }
    result.speaker_index = voice.speaker_index;

    result.z = ggml_new_tensor_2d(context, GGML_TYPE_F32, hparams.inter_channels, frame_count);
    ggml_set_name(result.z, "flow.z.input");
    ggml_set_input(result.z);
    ggml_tensor * state = ggml_cont(context, ggml_transpose(context, result.z));
    state               = conv1d_time_major(context, state, weights.pre, 3);
    if (voice.embedding != nullptr) {
        ggml_tensor * conditioning = conv1d(context, voice.embedding, weights.conditioning, 0);
        conditioning               = ggml_cont(context, ggml_transpose(context, conditioning));
        state                      = ggml_add(context, state, conditioning);
    }
    for (size_t stage_index = 0; stage_index < weights.stages.size(); ++stage_index) {
        const DecoderStageWeights & stage = weights.stages[stage_index];
        state                             = ggml_leaky_relu(context, state, hparams.decoder_leaky_relu_slope, false);
        const int stride                  = static_cast<int>(hparams.decoder_upsample_rates[stage_index]);
        const int kernel                  = static_cast<int>(hparams.decoder_upsample_kernel_sizes[stage_index]);
        const int padding                 = (kernel - stride) / 2;
        state = transpose_conv1d_time_major(context, state, stage.transpose_conv, stride, padding);
        if (state == nullptr) {
            std::fprintf(stderr, "vits: invalid waveform-decoder transpose convolution\n");
            return result;
        }
        ggml_tensor * branch_sum = nullptr;
        for (size_t branch = 0; branch < stage.resblocks.size(); ++branch) {
            ggml_tensor * branch_output = resblock_time_major(
                context, state, stage.resblocks[branch], hparams.decoder_resblock_kernel_sizes[branch],
                hparams.decoder_resblock_dilations[branch], hparams.decoder_leaky_relu_slope);
            branch_sum = branch_sum == nullptr ? branch_output : ggml_add(context, branch_sum, branch_output);
        }
        state = stage.resblocks.size() == 1 ?
                    branch_sum :
                    ggml_scale(context, branch_sum, 1.0f / static_cast<float>(stage.resblocks.size()));
    }
    state      = ggml_leaky_relu(context, state, 0.01f, false);
    state      = conv1d_time_major_f32(context, weights.post_weight, state, 3, 1);
    state      = ggml_cont(context, ggml_transpose(context, state));
    result.pcm = ggml_cont(context, ggml_tanh(context, state));
    ggml_set_name(result.pcm, "audio.pcm");
    result.graph = ggml_new_graph_custom(context, 32768, false);
    ggml_build_forward_expand(result.graph, result.pcm);
    return result;
}

}  // namespace synth::vits
