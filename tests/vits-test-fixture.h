#pragma once

#include "arch/vits/weights.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace synth::test {

struct GgmlContextDeleter {
    void operator()(ggml_context * context) const {
        if (context != nullptr) {
            ggml_free(context);
        }
    }
};

using GgmlContext = std::unique_ptr<ggml_context, GgmlContextDeleter>;

inline GgmlContext make_ggml_context(size_t bytes = 4 * 1024 * 1024) {
    ggml_init_params parameters{};
    parameters.mem_size = bytes;
    parameters.no_alloc = true;
    return GgmlContext(ggml_init(parameters));
}

inline vits::HParams small_vits_hparams() {
    vits::HParams hparams;
    hparams.input_flags                   = SYNTH_INPUT_SUPPORT_TOKEN_IDS;
    hparams.capability_flags              = SYNTH_MODEL_CAPABILITY_SPEAKING_RATE | SYNTH_MODEL_CAPABILITY_STOCHASTIC;
    hparams.output_sample_rate            = 22050;
    hparams.output_channel_count          = 1;
    hparams.vocab_size                    = 5;
    hparams.inter_channels                = 2;
    hparams.hidden_channels               = 4;
    hparams.filter_channels               = 8;
    hparams.text_layer_count              = 2;
    hparams.text_head_count               = 2;
    hparams.text_ffn_kernel_size          = 3;
    hparams.text_attention_window         = 1;
    hparams.duration_dds_layer_count      = 2;
    hparams.duration_flow_count           = 2;
    hparams.duration_spline_bin_count     = 3;
    hparams.flow_block_count              = 2;
    hparams.flow_kernel_size              = 5;
    hparams.flow_dilation_rate            = 1;
    hparams.flow_wn_layer_count           = 2;
    hparams.flow_mean_only                = true;
    hparams.decoder_resblock_kernel_sizes = { 3 };
    hparams.decoder_resblock_dilations    = {
        { 1, 2, 3 }
    };
    hparams.decoder_upsample_rates         = { 4 };
    hparams.decoder_upsample_kernel_sizes  = { 8 };
    hparams.decoder_initial_channels       = 4;
    hparams.decoder_leaky_relu_slope       = 0.1f;
    hparams.hop_length                     = 4;
    hparams.max_input_tokens               = 8;
    hparams.max_output_frames              = 100;
    hparams.layer_norm_epsilon             = 1.0e-5f;
    hparams.embedding_scale                = 2.0f;
    hparams.duration_spline_tail_bound     = 5.0f;
    hparams.duration_spline_min_bin_width  = 1.0e-3f;
    hparams.duration_spline_min_bin_height = 1.0e-3f;
    hparams.duration_spline_min_derivative = 1.0e-3f;
    hparams.duration_noise_scale_w         = 0.8f;
    hparams.latent_noise_scale             = 0.667f;
    hparams.min_speaking_rate              = 0.5f;
    hparams.max_speaking_rate              = 2.0f;
    return hparams;
}

inline vits::HParams small_conditioned_vits_hparams() {
    vits::HParams hparams         = small_vits_hparams();
    hparams.model_variant         = "vits-vctk";
    hparams.has_package_default   = false;
    hparams.speaker_count         = 3;
    hparams.conditioning_channels = 3;
    hparams.preset_voice_ids      = { "speaker-000", "speaker-001", "speaker-002" };
    hparams.preset_voice_flags    = { 0, 0, 0 };
    return hparams;
}

inline void add_named_tensor(ggml_context *       context,
                             const std::string &  name,
                             std::vector<int64_t> dimensions,
                             const std::string &  missing,
                             const std::string &  wrong_type,
                             const std::string &  wrong_shape,
                             ggml_type            wrong_type_value = GGML_TYPE_I32,
                             ggml_type            default_type     = GGML_TYPE_F32) {
    if (name == missing) {
        return;
    }
    if (name == wrong_shape) {
        ++dimensions[0];
    }
    const ggml_type type   = name == wrong_type ? wrong_type_value : default_type;
    ggml_tensor *   tensor = ggml_new_tensor(context, type, static_cast<int>(dimensions.size()), dimensions.data());
    ggml_set_name(tensor, name.c_str());
}

inline void add_conv(ggml_context *      context,
                     const std::string & prefix,
                     int64_t             kernel,
                     int64_t             input_channels,
                     int64_t             output_channels,
                     const std::string & missing,
                     const std::string & wrong_type,
                     const std::string & wrong_shape,
                     ggml_type           wrong_type_value = GGML_TYPE_I32,
                     ggml_type           weight_type      = GGML_TYPE_F32,
                     bool                pack_weight      = false) {
    const std::vector<int64_t> weight_shape = pack_weight ?
                                                  std::vector<int64_t>{ kernel * input_channels, output_channels } :
                                                  std::vector<int64_t>{ kernel, input_channels, output_channels };
    add_named_tensor(context, prefix + ".weight", weight_shape, missing, wrong_type, wrong_shape, wrong_type_value,
                     weight_type);
    add_named_tensor(context, prefix + ".bias", { output_channels }, missing, wrong_type, wrong_shape,
                     wrong_type_value);
}

inline void add_norm(ggml_context *      context,
                     const std::string & prefix,
                     int64_t             channels,
                     const std::string & missing,
                     const std::string & wrong_type,
                     const std::string & wrong_shape,
                     ggml_type           wrong_type_value = GGML_TYPE_I32) {
    add_named_tensor(context, prefix + ".weight", { channels }, missing, wrong_type, wrong_shape, wrong_type_value);
    add_named_tensor(context, prefix + ".bias", { channels }, missing, wrong_type, wrong_shape, wrong_type_value);
}

inline void populate_text_weight_tensors(ggml_context *        context,
                                         const vits::HParams & hparams,
                                         const std::string &   missing          = {},
                                         const std::string &   wrong_type       = {},
                                         const std::string &   wrong_shape      = {},
                                         ggml_type             wrong_type_value = GGML_TYPE_I32) {
    const int64_t hidden        = hparams.hidden_channels;
    const int64_t filter        = hparams.filter_channels;
    const int64_t head_channels = hidden / hparams.text_head_count;
    const int64_t relative_rows = 2 * static_cast<int64_t>(hparams.text_attention_window) + 1;
    add_named_tensor(context, "text_encoder.token_embedding.weight", { hidden, hparams.vocab_size }, missing,
                     wrong_type, wrong_shape, wrong_type_value);

    for (uint32_t layer = 0; layer < hparams.text_layer_count; ++layer) {
        const std::string prefix = "text_encoder.blocks." + std::to_string(layer);
        add_conv(context, prefix + ".attention.query", 1, hidden, hidden, missing, wrong_type, wrong_shape,
                 wrong_type_value);
        add_conv(context, prefix + ".attention.key", 1, hidden, hidden, missing, wrong_type, wrong_shape,
                 wrong_type_value);
        add_conv(context, prefix + ".attention.value", 1, hidden, hidden, missing, wrong_type, wrong_shape,
                 wrong_type_value);
        add_conv(context, prefix + ".attention.output", 1, hidden, hidden, missing, wrong_type, wrong_shape,
                 wrong_type_value);
        add_norm(context, prefix + ".attention_norm", hidden, missing, wrong_type, wrong_shape, wrong_type_value);
        add_conv(context, prefix + ".ffn.input", hparams.text_ffn_kernel_size, hidden, filter, missing, wrong_type,
                 wrong_shape, wrong_type_value);
        add_conv(context, prefix + ".ffn.output", hparams.text_ffn_kernel_size, filter, hidden, missing, wrong_type,
                 wrong_shape, wrong_type_value);
        add_norm(context, prefix + ".ffn_norm", hidden, missing, wrong_type, wrong_shape, wrong_type_value);
        add_named_tensor(context, prefix + ".attention.relative_key.weight", { head_channels, relative_rows }, missing,
                         wrong_type, wrong_shape, wrong_type_value);
        add_named_tensor(context, prefix + ".attention.relative_value.weight", { head_channels, relative_rows },
                         missing, wrong_type, wrong_shape, wrong_type_value);
    }
    add_conv(context, "text_encoder.projection", 1, hidden, 2 * hparams.inter_channels, missing, wrong_type,
             wrong_shape, wrong_type_value);
}

inline void populate_dds_weight_tensors(ggml_context *      context,
                                        const std::string & prefix,
                                        int64_t             channels,
                                        uint32_t            layer_count,
                                        const std::string & missing,
                                        const std::string & wrong_type,
                                        const std::string & wrong_shape,
                                        ggml_type           wrong_type_value = GGML_TYPE_I32) {
    for (uint32_t layer = 0; layer < layer_count; ++layer) {
        const std::string block = prefix + ".blocks." + std::to_string(layer);
        add_conv(context, block + ".depthwise", 3, 1, channels, missing, wrong_type, wrong_shape, wrong_type_value);
        add_norm(context, block + ".depthwise_norm", channels, missing, wrong_type, wrong_shape, wrong_type_value);
        add_conv(context, block + ".pointwise", 1, channels, channels, missing, wrong_type, wrong_shape,
                 wrong_type_value);
        add_norm(context, block + ".pointwise_norm", channels, missing, wrong_type, wrong_shape, wrong_type_value);
    }
}

inline void populate_duration_weight_tensors(ggml_context *        context,
                                             const vits::HParams & hparams,
                                             const std::string &   missing          = {},
                                             const std::string &   wrong_type       = {},
                                             const std::string &   wrong_shape      = {},
                                             ggml_type             wrong_type_value = GGML_TYPE_I32) {
    const int64_t hidden            = hparams.hidden_channels;
    const int64_t spline_parameters = 3 * static_cast<int64_t>(hparams.duration_spline_bin_count) - 1;
    if (hparams.conditioning_channels > 0) {
        add_conv(context, "duration_predictor.conditioning", 1, hparams.conditioning_channels, hidden, missing,
                 wrong_type, wrong_shape, wrong_type_value);
    }
    add_conv(context, "duration_predictor.pre", 1, hidden, hidden, missing, wrong_type, wrong_shape, wrong_type_value);
    populate_dds_weight_tensors(context, "duration_predictor.dds", hidden, hparams.duration_dds_layer_count, missing,
                                wrong_type, wrong_shape, wrong_type_value);
    add_conv(context, "duration_predictor.projection", 1, hidden, hidden, missing, wrong_type, wrong_shape,
             wrong_type_value);
    add_named_tensor(context, "duration_predictor.affine.bias", { 1, 2 }, missing, wrong_type, wrong_shape,
                     wrong_type_value);
    add_named_tensor(context, "duration_predictor.affine.log_scale", { 1, 2 }, missing, wrong_type, wrong_shape,
                     wrong_type_value);

    for (uint32_t flow = 0; flow < hparams.duration_flow_count; ++flow) {
        const std::string prefix = "duration_predictor.flows." + std::to_string(flow);
        add_conv(context, prefix + ".pre", 1, 1, hidden, missing, wrong_type, wrong_shape, wrong_type_value);
        populate_dds_weight_tensors(context, prefix + ".dds", hidden, hparams.duration_dds_layer_count, missing,
                                    wrong_type, wrong_shape, wrong_type_value);
        add_conv(context, prefix + ".projection", 1, hidden, spline_parameters, missing, wrong_type, wrong_shape,
                 wrong_type_value);
    }
}

inline void populate_flow_weight_tensors(ggml_context *        context,
                                         const vits::HParams & hparams,
                                         const std::string &   missing          = {},
                                         const std::string &   wrong_type       = {},
                                         const std::string &   wrong_shape      = {},
                                         ggml_type             wrong_type_value = GGML_TYPE_I32,
                                         ggml_type             learned_type     = GGML_TYPE_F32,
                                         bool                  pack_learned     = false) {
    const int64_t hidden = hparams.hidden_channels;
    const int64_t half   = hparams.inter_channels / 2;
    for (uint32_t block = 0; block < hparams.flow_block_count; ++block) {
        const std::string prefix = "flow.blocks." + std::to_string(block);
        if (hparams.conditioning_channels > 0) {
            add_conv(context, prefix + ".conditioning", 1, hparams.conditioning_channels,
                     2 * hidden * hparams.flow_wn_layer_count, missing, wrong_type, wrong_shape, wrong_type_value,
                     learned_type, pack_learned);
        }
        add_conv(context, prefix + ".pre", 1, half, hidden, missing, wrong_type, wrong_shape, wrong_type_value,
                 learned_type, pack_learned);
        add_conv(context, prefix + ".projection", 1, hidden, half, missing, wrong_type, wrong_shape, wrong_type_value,
                 learned_type, pack_learned);
        for (uint32_t layer = 0; layer < hparams.flow_wn_layer_count; ++layer) {
            const std::string layer_prefix = prefix + ".wn.layers." + std::to_string(layer);
            add_conv(context, layer_prefix + ".input", hparams.flow_kernel_size, hidden, 2 * hidden, missing,
                     wrong_type, wrong_shape, wrong_type_value, learned_type, pack_learned);
            const int64_t residual_skip_channels = layer + 1 < hparams.flow_wn_layer_count ? 2 * hidden : hidden;
            add_conv(context, layer_prefix + ".residual_skip", 1, hidden, residual_skip_channels, missing, wrong_type,
                     wrong_shape, wrong_type_value, learned_type, pack_learned);
        }
    }
}

inline void populate_decoder_weight_tensors(ggml_context *        context,
                                            const vits::HParams & hparams,
                                            const std::string &   missing          = {},
                                            const std::string &   wrong_type       = {},
                                            const std::string &   wrong_shape      = {},
                                            ggml_type             wrong_type_value = GGML_TYPE_I32,
                                            ggml_type             learned_type     = GGML_TYPE_F32,
                                            ggml_type             transpose_type   = GGML_TYPE_F32,
                                            bool                  pack_learned     = false) {
    const int64_t initial = hparams.decoder_initial_channels;
    if (hparams.conditioning_channels > 0) {
        add_conv(context, "decoder.conditioning", 1, hparams.conditioning_channels, initial, missing, wrong_type,
                 wrong_shape, wrong_type_value, learned_type, pack_learned);
    }
    add_conv(context, "decoder.pre", 7, hparams.inter_channels, initial, missing, wrong_type, wrong_shape,
             wrong_type_value, learned_type, pack_learned);
    int64_t channels = initial;
    for (size_t stage = 0; stage < hparams.decoder_upsample_rates.size(); ++stage) {
        const int64_t     next_channels = channels / 2;
        const std::string prefix        = "decoder.upsample." + std::to_string(stage);
        add_named_tensor(context, prefix + ".transpose_conv.weight",
                         { hparams.decoder_upsample_kernel_sizes[stage], next_channels, channels }, missing, wrong_type,
                         wrong_shape, wrong_type_value, transpose_type);
        add_named_tensor(context, prefix + ".transpose_conv.bias", { next_channels }, missing, wrong_type, wrong_shape,
                         wrong_type_value);
        for (size_t branch = 0; branch < hparams.decoder_resblock_kernel_sizes.size(); ++branch) {
            const std::string block = prefix + ".resblocks." + std::to_string(branch);
            for (size_t layer = 0; layer < hparams.decoder_resblock_dilations[branch].size(); ++layer) {
                add_conv(context, block + ".conv1." + std::to_string(layer),
                         hparams.decoder_resblock_kernel_sizes[branch], next_channels, next_channels, missing,
                         wrong_type, wrong_shape, wrong_type_value, learned_type, pack_learned);
                add_conv(context, block + ".conv2." + std::to_string(layer),
                         hparams.decoder_resblock_kernel_sizes[branch], next_channels, next_channels, missing,
                         wrong_type, wrong_shape, wrong_type_value, learned_type, pack_learned);
            }
        }
        channels = next_channels;
    }
    const std::vector<int64_t> post_shape =
        pack_learned ? std::vector<int64_t>{ 7 * channels, 1 } : std::vector<int64_t>{ 7, channels, 1 };
    add_named_tensor(context, "decoder.post.weight", post_shape, missing, wrong_type, wrong_shape, wrong_type_value,
                     learned_type);
}

inline void populate_voice_weight_tensors(ggml_context *        context,
                                          const vits::HParams & hparams,
                                          const std::string &   missing          = {},
                                          const std::string &   wrong_type       = {},
                                          const std::string &   wrong_shape      = {},
                                          ggml_type             wrong_type_value = GGML_TYPE_I32) {
    if (hparams.speaker_count > 0 && hparams.conditioning_channels > 0) {
        add_named_tensor(context, "voice.embedding.weight", { hparams.conditioning_channels, hparams.speaker_count },
                         missing, wrong_type, wrong_shape, wrong_type_value);
    }
}

}  // namespace synth::test
