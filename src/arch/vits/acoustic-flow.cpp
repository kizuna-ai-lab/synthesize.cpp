#include "acoustic-flow.h"

#include "ggml.h"
#include "operations.h"
#include "voice-conditioning.h"
#include "weights.h"

#include <cstdio>
#include <limits>

namespace synth::vits {

namespace {

bool flow_dilations_fit(const HParams & hparams) {
    uint64_t dilation = 1;
    for (uint32_t layer = 0; layer < hparams.flow_wn_layer_count; ++layer) {
        const uint64_t padding = (static_cast<uint64_t>(hparams.flow_kernel_size - 1) * dilation) / 2;
        if (dilation > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
            padding > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            return false;
        }
        if (layer + 1 < hparams.flow_wn_layer_count) {
            if (dilation > static_cast<uint64_t>(std::numeric_limits<int>::max()) / hparams.flow_dilation_rate) {
                return false;
            }
            dilation *= hparams.flow_dilation_rate;
        }
    }
    return true;
}

ggml_tensor * channel_view(ggml_context * context, ggml_tensor * input, int64_t first_channel, int64_t channel_count) {
    ggml_tensor * view = ggml_view_2d(context, input, channel_count, input->ne[1], input->nb[1],
                                      static_cast<size_t>(first_channel) * sizeof(float));
    return ggml_cont(context, view);
}

ggml_tensor * flip_channels(ggml_context * context, ggml_tensor * input, ggml_tensor * channel_indices) {
    ggml_tensor * time_major = ggml_cont(context, ggml_transpose(context, input));
    ggml_tensor * reversed   = ggml_get_rows(context, time_major, channel_indices);
    return ggml_cont(context, ggml_transpose(context, reversed));
}

ggml_tensor * wavenet(ggml_context *           context,
                      ggml_tensor *            input,
                      ggml_tensor *            voice_embedding,
                      const FlowBlockWeights & weights,
                      const HParams &          hparams) {
    ggml_tensor * state  = input;
    ggml_tensor * output = nullptr;
    ggml_tensor * conditioning =
        voice_embedding == nullptr ? nullptr : conv1d(context, voice_embedding, weights.conditioning, 0);
    int64_t dilation = 1;
    for (size_t layer = 0; layer < weights.wn_layers.size(); ++layer) {
        const int64_t padding     = (static_cast<int64_t>(hparams.flow_kernel_size) * dilation - dilation) / 2;
        ggml_tensor * activations = conv1d(context, state, weights.wn_layers[layer].input, static_cast<int>(padding),
                                           static_cast<int>(dilation));
        if (conditioning != nullptr) {
            activations =
                ggml_add(context, activations,
                         channel_view(context, conditioning, 2 * static_cast<int64_t>(hparams.hidden_channels) * layer,
                                      2 * hparams.hidden_channels));
        }
        ggml_tensor * tanh_branch = channel_view(context, activations, 0, hparams.hidden_channels);
        ggml_tensor * sigmoid_branch =
            channel_view(context, activations, hparams.hidden_channels, hparams.hidden_channels);
        activations = ggml_mul(context, ggml_tanh(context, tanh_branch), ggml_sigmoid(context, sigmoid_branch));
        ggml_tensor * residual_skip = conv1d(context, activations, weights.wn_layers[layer].residual_skip, 0);
        if (layer + 1 < weights.wn_layers.size()) {
            state = ggml_add(context, state, channel_view(context, residual_skip, 0, hparams.hidden_channels));
            ggml_tensor * skip = channel_view(context, residual_skip, hparams.hidden_channels, hparams.hidden_channels);
            output             = output == nullptr ? skip : ggml_add(context, output, skip);
        } else {
            output = output == nullptr ? residual_skip : ggml_add(context, output, residual_skip);
        }
        if (layer + 1 < weights.wn_layers.size()) {
            if (dilation > std::numeric_limits<int>::max() / hparams.flow_dilation_rate) {
                return nullptr;
            }
            dilation *= hparams.flow_dilation_rate;
        }
    }
    return output;
}

ggml_tensor * inverse_coupling(ggml_context *           context,
                               ggml_tensor *            input,
                               ggml_tensor *            voice_embedding,
                               const FlowBlockWeights & weights,
                               const HParams &          hparams) {
    const int64_t half   = hparams.inter_channels / 2;
    ggml_tensor * first  = channel_view(context, input, 0, half);
    ggml_tensor * second = channel_view(context, input, half, half);
    ggml_tensor * hidden = conv1d(context, first, weights.pre, 0);
    hidden               = wavenet(context, hidden, voice_embedding, weights, hparams);
    if (hidden == nullptr) {
        return nullptr;
    }
    ggml_tensor * mean = conv1d(context, hidden, weights.projection, 0);
    second             = ggml_sub(context, second, mean);
    return ggml_concat(context, first, second, 0);
}

}  // namespace

AcousticFlowGraph build_acoustic_flow_graph(ggml_context *      context,
                                            const FlowWeights & weights,
                                            const HParams &     hparams,
                                            int64_t             frame_count) {
    const VoiceWeights no_voice;
    return build_acoustic_flow_graph(context, weights, no_voice, hparams, UINT32_MAX, frame_count);
}

AcousticFlowGraph build_acoustic_flow_graph(ggml_context *       context,
                                            const FlowWeights &  weights,
                                            const VoiceWeights & voice_weights,
                                            const HParams &      hparams,
                                            uint32_t             speaker_index,
                                            int64_t              frame_count) {
    AcousticFlowGraph result;
    if (context == nullptr || frame_count <= 0 || hparams.inter_channels == 0 || hparams.inter_channels % 2 != 0 ||
        hparams.hidden_channels == 0 || hparams.flow_block_count == 0 || hparams.flow_kernel_size == 0 ||
        hparams.flow_kernel_size % 2 == 0 || hparams.flow_dilation_rate == 0 || hparams.flow_wn_layer_count == 0 ||
        !hparams.flow_mean_only || !flow_dilations_fit(hparams) || weights.blocks.size() != hparams.flow_block_count) {
        std::fprintf(stderr, "vits: invalid acoustic-flow graph request\n");
        return result;
    }
    for (const FlowBlockWeights & block : weights.blocks) {
        if (block.wn_layers.size() != hparams.flow_wn_layer_count ||
            (hparams.conditioning_channels > 0 &&
             (block.conditioning.weight == nullptr || block.conditioning.bias == nullptr))) {
            std::fprintf(stderr, "vits: invalid acoustic-flow weight catalog\n");
            return result;
        }
    }

    const VoiceConditioning voice = build_voice_conditioning(context, voice_weights, hparams, speaker_index);
    if (!voice.valid) {
        std::fprintf(stderr, "vits: invalid acoustic-flow Voice conditioning\n");
        return result;
    }
    result.speaker_index = voice.speaker_index;

    result.z_p             = ggml_new_tensor_2d(context, GGML_TYPE_F32, hparams.inter_channels, frame_count);
    result.channel_indices = ggml_new_tensor_1d(context, GGML_TYPE_I32, hparams.inter_channels);
    ggml_set_name(result.z_p, "latent.z_p.input");
    ggml_set_name(result.channel_indices, "flow.channel_indices");
    ggml_set_input(result.z_p);
    ggml_set_input(result.channel_indices);
    ggml_tensor * latent = result.z_p;
    for (size_t block = weights.blocks.size(); block-- > 0;) {
        latent = flip_channels(context, latent, result.channel_indices);
        latent = inverse_coupling(context, latent, voice.embedding, weights.blocks[block], hparams);
        if (latent == nullptr) {
            std::fprintf(stderr, "vits: acoustic-flow dilation overflow\n");
            return result;
        }
    }
    result.z = ggml_cont(context, latent);
    ggml_set_name(result.z, "flow.z");
    result.graph = ggml_new_graph_custom(context, 8192, false);
    ggml_build_forward_expand(result.graph, result.z);
    return result;
}

}  // namespace synth::vits
