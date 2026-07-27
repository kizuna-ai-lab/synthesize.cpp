#include "weights.h"

#include "ggml.h"
#include "gguf-metadata.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <string>

namespace synth::vits {

namespace {

bool flow_dilations_fit(uint32_t kernel_size, uint32_t dilation_rate, uint32_t layer_count) {
    if (kernel_size == 0 || dilation_rate == 0 || layer_count == 0) {
        return false;
    }
    uint64_t dilation = 1;
    for (uint32_t layer = 0; layer < layer_count; ++layer) {
        const uint64_t padding = (static_cast<uint64_t>(kernel_size - 1) * dilation) / 2;
        if (dilation > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
            padding > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            return false;
        }
        if (layer + 1 < layer_count) {
            if (dilation > static_cast<uint64_t>(std::numeric_limits<int>::max()) / dilation_rate) {
                return false;
            }
            dilation *= dilation_rate;
        }
    }
    return true;
}

// Typed metadata reads are shared with every other Model Family; these keep the
// family's existing call shape while the validation itself lives in one place.
GgufMetadata metadata(const gguf_context * gguf) {
    return GgufMetadata(gguf, "vits");
}

bool read_u32(const gguf_context * gguf, const char * key, uint32_t & value) {
    return metadata(gguf).u32(key, value);
}

bool read_u64(const gguf_context * gguf, const char * key, uint64_t & value) {
    return metadata(gguf).u64(key, value);
}

bool read_f32(const gguf_context * gguf, const char * key, float & value) {
    return metadata(gguf).f32(key, value);
}

bool read_bool(const gguf_context * gguf, const char * key, bool & value) {
    return metadata(gguf).boolean(key, value);
}

bool read_positive_i32_array(const gguf_context * gguf, const std::string & key, std::vector<uint32_t> & value) {
    return metadata(gguf).positive_i32_array(key, value);
}

bool read_decoder_dilations(const gguf_context *                 gguf,
                            size_t                               branch_count,
                            std::vector<std::vector<uint32_t>> & output) {
    output.clear();
    output.resize(branch_count);
    for (size_t branch = 0; branch < branch_count; ++branch) {
        const std::string key = "synthesize.vits.decoder.resblock_dilations." + std::to_string(branch);
        if (!read_positive_i32_array(gguf, key, output[branch])) {
            return false;
        }
    }
    return true;
}

bool decoder_hparams_fit(const HParams & hparams) {
    if (hparams.decoder_resblock_kernel_sizes.empty() ||
        hparams.decoder_resblock_kernel_sizes.size() != hparams.decoder_resblock_dilations.size() ||
        hparams.decoder_upsample_rates.empty() ||
        hparams.decoder_upsample_rates.size() != hparams.decoder_upsample_kernel_sizes.size() ||
        hparams.decoder_initial_channels == 0 || !std::isfinite(hparams.decoder_leaky_relu_slope) ||
        hparams.decoder_leaky_relu_slope <= 0.0f || hparams.decoder_leaky_relu_slope > 1.0f) {
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

bool require_string(const gguf_context * gguf, const char * key, const char * expected) {
    return metadata(gguf).require_string(key, expected);
}

bool read_string(const gguf_context * gguf, const std::string & key, std::string & value) {
    return metadata(gguf).string(key, value);
}

bool read_string_array(const gguf_context * gguf, const std::string & key, std::vector<std::string> & value) {
    return metadata(gguf).string_array(key, value);
}

bool read_frontend_metadata(const gguf_context * gguf, HParams & hparams) {
    if (!read_bool(gguf, "synthesize.frontend.present", hparams.frontend_present)) {
        return false;
    }
    if (!hparams.frontend_present) {
        hparams.frontend_config = {};
        return hparams.input_flags == SYNTH_INPUT_SUPPORT_TOKEN_IDS;
    }

    std::string mapping_mode;
    std::string lookup_policy;
    bool        upstream_add_blank = false;
    if (hparams.input_flags != (SYNTH_INPUT_SUPPORT_PHONEMES_UTF8 | SYNTH_INPUT_SUPPORT_TOKEN_IDS) ||
        !read_string(gguf, "synthesize.frontend.provider", hparams.frontend_config.provider_id) ||
        !read_u32(gguf, "synthesize.frontend.contract_version", hparams.frontend_config.contract_version) ||
        !read_string(gguf, "synthesize.frontend.phoneme_mapping", mapping_mode) ||
        !read_string_array(gguf, "synthesize.vits.symbols", hparams.frontend_config.symbols) ||
        !read_string(gguf, "synthesize.vits.symbols.lookup_policy", lookup_policy) ||
        !read_u32(gguf, "synthesize.vits.symbols.blank_id", hparams.frontend_config.blank_id) ||
        !read_bool(gguf, "synthesize.vits.symbols.upstream_add_blank", upstream_add_blank) ||
        mapping_mode != "unicode_scalar" || lookup_policy != "last_index_wins" ||
        hparams.frontend_config.symbols.size() != hparams.vocab_size) {
        return false;
    }
    hparams.frontend_config.mapping_mode = SymbolMappingMode::UnicodeScalar;
    hparams.frontend_config.padding_rule =
        upstream_add_blank ? SymbolPaddingRule::InterleavedBlank : SymbolPaddingRule::None;
    std::unique_ptr<TextFrontend> frontend;
    return make_symbol_map_frontend(hparams.frontend_config, frontend) == SYNTH_OK;
}

bool read_voice_metadata(const gguf_context * gguf, HParams & hparams) {
    std::string mode;
    uint32_t    preset_count = 0;
    if (!read_string(gguf, "synthesize.voice.mode", mode) ||
        !read_bool(gguf, "synthesize.voice.has_package_default", hparams.has_package_default) ||
        !read_u32(gguf, "synthesize.voice.preset_count", preset_count)) {
        return false;
    }
    hparams.preset_voice_ids.clear();
    hparams.preset_voice_flags.clear();
    hparams.preset_voice_ids.reserve(preset_count);
    hparams.preset_voice_flags.reserve(preset_count);
    for (uint32_t index = 0; index < preset_count; ++index) {
        const std::string prefix = "synthesize.voice." + std::to_string(index);
        std::string       id;
        uint32_t          flags = 0;
        if (!read_string(gguf, prefix + ".id", id) || id.empty() ||
            !read_u32(gguf, (prefix + ".flags").c_str(), flags)) {
            return false;
        }
        hparams.preset_voice_ids.push_back(std::move(id));
        hparams.preset_voice_flags.push_back(flags);
    }

    if (hparams.model_variant == "vits-ljspeech") {
        return mode == "fixed-default" && hparams.has_package_default && hparams.speaker_count == 0 &&
               hparams.conditioning_channels == 0 && preset_count == 0;
    }
    if (hparams.model_variant != "vits-vctk" || mode != "preset-catalog" || hparams.has_package_default ||
        hparams.speaker_count != 109 || hparams.conditioning_channels == 0 || preset_count != hparams.speaker_count) {
        return false;
    }
    for (uint32_t index = 0; index < preset_count; ++index) {
        char expected[20];
        std::snprintf(expected, sizeof(expected), "speaker-%03u", index);
        if (hparams.preset_voice_ids[index] != expected) {
            return false;
        }
    }
    return true;
}

bool read_quantization_metadata(const gguf_context * gguf, HParams & hparams) {
    std::string profile;
    uint32_t    file_type = 0;
    if (!read_u32(gguf, "general.file_type", file_type) ||
        !read_string(gguf, "synthesize.quantization.profile", profile) ||
        !read_u32(gguf, "synthesize.quantization.profile_version", hparams.quantization_profile_version)) {
        return false;
    }
    if (hparams.quantization_profile_version != 1) {
        std::fprintf(stderr, "vits: unsupported quantization profile version %u\n",
                     hparams.quantization_profile_version);
        return false;
    }
    if (profile == "F32" && file_type == 0) {
        hparams.quantization_profile = QuantizationProfile::F32;
        return true;
    }
    if (profile == "F16" && file_type == 1) {
        hparams.quantization_profile = QuantizationProfile::F16;
        return true;
    }
    if (profile == "Q8_MIXED" && file_type == GGML_FTYPE_MOSTLY_Q8_0) {
        hparams.quantization_profile = QuantizationProfile::Q8Mixed;
        return true;
    }
    std::fprintf(stderr, "vits: unsupported or inconsistent quantization profile %s (file type %u)\n", profile.c_str(),
                 file_type);
    return false;
}

enum class TensorRole {
    MatrixWeight,
    TransposeWeight,
    Sensitive,
};

ggml_tensor * find_tensor(ggml_context *                 context,
                          const HParams &                hparams,
                          const std::string &            name,
                          std::initializer_list<int64_t> expected,
                          TensorRole                     role) {
    ggml_tensor * tensor = ggml_get_tensor(context, name.c_str());
    if (tensor == nullptr) {
        std::fprintf(stderr, "vits: missing tensor %s\n", name.c_str());
        return nullptr;
    }
    // Transposed-convolution weights stay F32 in every profile. They feed the
    // column matrix multiply in transpose_conv1d_without_bias, and CUDA's F16
    // matrix multiply accumulates in half precision -- about 3e-3 relative
    // against the F32 reference on the decoder's upsampling stages, where the
    // fused op this replaced had accumulated in F32. A package built before
    // that change fails here by type rather than drifting quietly.
    ggml_type expected_type = GGML_TYPE_F32;
    if (role == TensorRole::MatrixWeight) {
        if (hparams.quantization_profile == QuantizationProfile::F16) {
            expected_type = GGML_TYPE_F16;
        } else if (hparams.quantization_profile == QuantizationProfile::Q8Mixed) {
            expected_type = GGML_TYPE_Q8_0;
        }
    }
    if (tensor->type != expected_type) {
        std::fprintf(stderr, "vits: tensor %s has type %s, expected %s for package profile\n", name.c_str(),
                     ggml_type_name(tensor->type), ggml_type_name(expected_type));
        return nullptr;
    }
    if (hparams.quantization_profile == QuantizationProfile::Q8Mixed && role == TensorRole::MatrixWeight) {
        if (expected.size() != 3) {
            std::fprintf(stderr, "vits: tensor %s has invalid packed-matrix contract\n", name.c_str());
            return nullptr;
        }
        auto wanted = expected.begin();
        if (wanted[0] <= 0 || wanted[1] <= 0 || wanted[0] > std::numeric_limits<int64_t>::max() / wanted[1] ||
            tensor->ne[0] != wanted[0] * wanted[1] || tensor->ne[1] != wanted[2] || tensor->ne[2] != 1 ||
            tensor->ne[3] != 1) {
            std::fprintf(stderr, "vits: tensor %s packed matrix shape mismatch\n", name.c_str());
            return nullptr;
        }
        return tensor;
    }
    size_t dimension = 0;
    for (int64_t wanted : expected) {
        if (dimension >= GGML_MAX_DIMS || tensor->ne[dimension] != wanted) {
            std::fprintf(stderr, "vits: tensor %s shape mismatch at dimension %zu\n", name.c_str(), dimension);
            return nullptr;
        }
        ++dimension;
    }
    for (; dimension < GGML_MAX_DIMS; ++dimension) {
        if (tensor->ne[dimension] != 1) {
            std::fprintf(stderr, "vits: tensor %s has unexpected rank\n", name.c_str());
            return nullptr;
        }
    }
    return tensor;
}

bool load_conv_with_role(ggml_context *      context,
                         const HParams &     hparams,
                         const std::string & prefix,
                         int64_t             kernel,
                         int64_t             input_channels,
                         int64_t             output_channels,
                         TensorRole          weight_role,
                         Conv1dWeights &     output) {
    output.weight =
        find_tensor(context, hparams, prefix + ".weight", { kernel, input_channels, output_channels }, weight_role);
    output.bias = find_tensor(context, hparams, prefix + ".bias", { output_channels }, TensorRole::Sensitive);
    return output.weight != nullptr && output.bias != nullptr;
}

bool load_conv(ggml_context *      context,
               const HParams &     hparams,
               const std::string & prefix,
               int64_t             kernel,
               int64_t             input_channels,
               int64_t             output_channels,
               Conv1dWeights &     output) {
    return load_conv_with_role(context, hparams, prefix, kernel, input_channels, output_channels,
                               TensorRole::MatrixWeight, output);
}

bool load_sensitive_conv(ggml_context *      context,
                         const HParams &     hparams,
                         const std::string & prefix,
                         int64_t             kernel,
                         int64_t             input_channels,
                         int64_t             output_channels,
                         Conv1dWeights &     output) {
    return load_conv_with_role(context, hparams, prefix, kernel, input_channels, output_channels, TensorRole::Sensitive,
                               output);
}

bool load_norm(ggml_context *      context,
               const HParams &     hparams,
               const std::string & prefix,
               int64_t             channels,
               NormWeights &       output) {
    output.weight = find_tensor(context, hparams, prefix + ".weight", { channels }, TensorRole::Sensitive);
    output.bias   = find_tensor(context, hparams, prefix + ".bias", { channels }, TensorRole::Sensitive);
    return output.weight != nullptr && output.bias != nullptr;
}

bool load_dds(ggml_context *      context,
              const HParams &     hparams,
              const std::string & prefix,
              int64_t             channels,
              uint32_t            layer_count,
              DDSWeights &        output) {
    output.blocks.clear();
    output.blocks.resize(layer_count);
    for (uint32_t layer = 0; layer < layer_count; ++layer) {
        DDSBlockWeights & block        = output.blocks[layer];
        const std::string block_prefix = prefix + ".blocks." + std::to_string(layer);
        if (!load_sensitive_conv(context, hparams, block_prefix + ".depthwise", 3, 1, channels, block.depthwise) ||
            !load_norm(context, hparams, block_prefix + ".depthwise_norm", channels, block.depthwise_norm) ||
            !load_sensitive_conv(context, hparams, block_prefix + ".pointwise", 1, channels, channels,
                                 block.pointwise) ||
            !load_norm(context, hparams, block_prefix + ".pointwise_norm", channels, block.pointwise_norm)) {
            return false;
        }
    }
    return true;
}

}  // namespace

synth_status_t read_hparams(const gguf_context * gguf, HParams & hparams) {
    if (gguf == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    hparams                       = {};
    uint32_t format_version       = 0;
    uint32_t architecture_version = 0;
    if (!require_string(gguf, "general.architecture", "vits") ||
        !require_string(gguf, "synthesize.model_family", "vits") ||
        !read_string(gguf, "synthesize.model_variant", hparams.model_variant) ||
        !read_quantization_metadata(gguf, hparams) || !read_u32(gguf, "synthesize.format_version", format_version) ||
        !read_u32(gguf, "synthesize.vits.architecture_version", architecture_version) ||
        !read_u32(gguf, "synthesize.capabilities.input_flags", hparams.input_flags) ||
        !read_u32(gguf, "synthesize.capabilities.flags", hparams.capability_flags) ||
        !read_u32(gguf, "synthesize.audio.sample_rate_hz", hparams.output_sample_rate) ||
        !read_u32(gguf, "synthesize.audio.channels", hparams.output_channel_count) ||
        !require_string(gguf, "synthesize.audio.sample_format", "f32le") ||
        !read_u32(gguf, "synthesize.vits.vocab_size", hparams.vocab_size) ||
        !read_u32(gguf, "synthesize.vits.inter_channels", hparams.inter_channels) ||
        !read_u32(gguf, "synthesize.vits.hidden_channels", hparams.hidden_channels) ||
        !read_u32(gguf, "synthesize.vits.filter_channels", hparams.filter_channels) ||
        !read_u32(gguf, "synthesize.vits.text.layer_count", hparams.text_layer_count) ||
        !read_u32(gguf, "synthesize.vits.text.head_count", hparams.text_head_count) ||
        !read_u32(gguf, "synthesize.vits.text.ffn_kernel_size", hparams.text_ffn_kernel_size) ||
        !read_u32(gguf, "synthesize.vits.text.attention_window", hparams.text_attention_window) ||
        !require_string(gguf, "synthesize.vits.duration.predictor_type", "stochastic") ||
        !read_u32(gguf, "synthesize.vits.duration.dds_layer_count", hparams.duration_dds_layer_count) ||
        !read_u32(gguf, "synthesize.vits.duration.flow_count", hparams.duration_flow_count) ||
        !read_u32(gguf, "synthesize.vits.duration.spline_bin_count", hparams.duration_spline_bin_count) ||
        !read_u32(gguf, "synthesize.vits.flow.block_count", hparams.flow_block_count) ||
        !read_u32(gguf, "synthesize.vits.flow.kernel_size", hparams.flow_kernel_size) ||
        !read_u32(gguf, "synthesize.vits.flow.dilation_rate", hparams.flow_dilation_rate) ||
        !read_u32(gguf, "synthesize.vits.flow.wn_layer_count", hparams.flow_wn_layer_count) ||
        !read_bool(gguf, "synthesize.vits.flow.mean_only", hparams.flow_mean_only) ||
        !require_string(gguf, "synthesize.vits.decoder.resblock_type", "1") ||
        !read_positive_i32_array(gguf, "synthesize.vits.decoder.resblock_kernel_sizes",
                                 hparams.decoder_resblock_kernel_sizes) ||
        !read_decoder_dilations(gguf, hparams.decoder_resblock_kernel_sizes.size(),
                                hparams.decoder_resblock_dilations) ||
        !read_positive_i32_array(gguf, "synthesize.vits.decoder.upsample_rates", hparams.decoder_upsample_rates) ||
        !read_positive_i32_array(gguf, "synthesize.vits.decoder.upsample_kernel_sizes",
                                 hparams.decoder_upsample_kernel_sizes) ||
        !read_u32(gguf, "synthesize.vits.decoder.upsample_initial_channels", hparams.decoder_initial_channels) ||
        !read_f32(gguf, "synthesize.vits.decoder.leaky_relu_slope", hparams.decoder_leaky_relu_slope) ||
        !require_string(gguf, "synthesize.vits.decoder.output_activation", "tanh") ||
        !read_u32(gguf, "synthesize.vits.hop_length", hparams.hop_length) ||
        !read_u32(gguf, "synthesize.vits.speaker_count", hparams.speaker_count) ||
        !read_u32(gguf, "synthesize.vits.conditioning_channels", hparams.conditioning_channels) ||
        !read_u64(gguf, "synthesize.capabilities.max_input_tokens", hparams.max_input_tokens) ||
        !read_u64(gguf, "synthesize.capabilities.max_output_frames", hparams.max_output_frames) ||
        !read_f32(gguf, "synthesize.vits.text.layer_norm_epsilon", hparams.layer_norm_epsilon) ||
        !read_f32(gguf, "synthesize.vits.text.embedding_scale", hparams.embedding_scale) ||
        !read_f32(gguf, "synthesize.vits.duration.spline_tail_bound", hparams.duration_spline_tail_bound) ||
        !read_f32(gguf, "synthesize.vits.duration.spline_min_bin_width", hparams.duration_spline_min_bin_width) ||
        !read_f32(gguf, "synthesize.vits.duration.spline_min_bin_height", hparams.duration_spline_min_bin_height) ||
        !read_f32(gguf, "synthesize.vits.duration.spline_min_derivative", hparams.duration_spline_min_derivative) ||
        !read_f32(gguf, "synthesize.vits.inference.noise_scale", hparams.latent_noise_scale) ||
        !read_f32(gguf, "synthesize.vits.inference.noise_scale_w", hparams.duration_noise_scale_w) ||
        !read_f32(gguf, "synthesize.capabilities.min_speaking_rate", hparams.min_speaking_rate) ||
        !read_f32(gguf, "synthesize.capabilities.max_speaking_rate", hparams.max_speaking_rate) ||
        !read_voice_metadata(gguf, hparams) || !read_frontend_metadata(gguf, hparams)) {
        return SYNTH_ERR_GGUF;
    }
    if (format_version != 1 || architecture_version != 1 ||
        hparams.capability_flags != (SYNTH_MODEL_CAPABILITY_SPEAKING_RATE | SYNTH_MODEL_CAPABILITY_STOCHASTIC) ||
        hparams.output_sample_rate == 0 || hparams.output_channel_count != 1 || hparams.vocab_size == 0 ||
        hparams.inter_channels == 0 || hparams.hidden_channels == 0 || hparams.filter_channels == 0 ||
        hparams.text_layer_count == 0 || hparams.text_head_count == 0 ||
        hparams.hidden_channels % hparams.text_head_count != 0 || hparams.text_ffn_kernel_size == 0 ||
        hparams.text_ffn_kernel_size % 2 == 0 || hparams.text_attention_window == 0 ||
        hparams.duration_dds_layer_count == 0 || hparams.duration_flow_count == 0 ||
        hparams.duration_spline_bin_count < 2 || hparams.inter_channels % 2 != 0 || hparams.flow_block_count == 0 ||
        hparams.flow_kernel_size == 0 || hparams.flow_kernel_size % 2 == 0 || hparams.flow_dilation_rate == 0 ||
        hparams.flow_wn_layer_count == 0 ||
        !flow_dilations_fit(hparams.flow_kernel_size, hparams.flow_dilation_rate, hparams.flow_wn_layer_count) ||
        !hparams.flow_mean_only || !decoder_hparams_fit(hparams) || hparams.hop_length == 0 ||
        hparams.max_input_tokens == 0 || hparams.max_output_frames < hparams.hop_length ||
        !std::isfinite(hparams.layer_norm_epsilon) || hparams.layer_norm_epsilon <= 0.0f ||
        !std::isfinite(hparams.embedding_scale) || hparams.embedding_scale <= 0.0f ||
        !std::isfinite(hparams.duration_spline_tail_bound) || hparams.duration_spline_tail_bound <= 0.0f ||
        !std::isfinite(hparams.duration_spline_min_bin_width) || hparams.duration_spline_min_bin_width <= 0.0f ||
        hparams.duration_spline_min_bin_width * hparams.duration_spline_bin_count > 1.0f ||
        !std::isfinite(hparams.duration_spline_min_bin_height) || hparams.duration_spline_min_bin_height <= 0.0f ||
        hparams.duration_spline_min_bin_height * hparams.duration_spline_bin_count > 1.0f ||
        !std::isfinite(hparams.duration_spline_min_derivative) || hparams.duration_spline_min_derivative <= 0.0f ||
        !std::isfinite(hparams.latent_noise_scale) || hparams.latent_noise_scale < 0.0f ||
        !std::isfinite(hparams.duration_noise_scale_w) || hparams.duration_noise_scale_w < 0.0f ||
        !std::isfinite(hparams.min_speaking_rate) || hparams.min_speaking_rate <= 0.0f ||
        !std::isfinite(hparams.max_speaking_rate) || hparams.max_speaking_rate < hparams.min_speaking_rate ||
        hparams.min_speaking_rate > 1.0f || hparams.max_speaking_rate < 1.0f) {
        std::fprintf(stderr, "vits: inconsistent VITS metadata\n");
        return SYNTH_ERR_GGUF;
    }
    return SYNTH_OK;
}

synth_status_t build_voice_weights(ggml_context * context, const HParams & hparams, VoiceWeights & weights) {
    weights = {};
    if (context == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (hparams.speaker_count == 0 && hparams.conditioning_channels == 0) {
        return SYNTH_OK;
    }
    if (hparams.speaker_count == 0 || hparams.conditioning_channels == 0) {
        return SYNTH_ERR_INVALID_ARG;
    }
    weights.embedding = find_tensor(context, hparams, "voice.embedding.weight",
                                    { hparams.conditioning_channels, hparams.speaker_count }, TensorRole::Sensitive);
    return weights.embedding == nullptr ? SYNTH_ERR_GGUF : SYNTH_OK;
}

synth_status_t build_duration_weights(ggml_context * context, const HParams & hparams, DurationWeights & weights) {
    if (context == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    const int64_t hidden            = hparams.hidden_channels;
    const int64_t spline_parameters = 3 * static_cast<int64_t>(hparams.duration_spline_bin_count) - 1;
    weights.conditioning            = {};
    if ((hparams.conditioning_channels > 0 &&
         !load_sensitive_conv(context, hparams, "duration_predictor.conditioning", 1, hparams.conditioning_channels,
                              hidden, weights.conditioning)) ||
        !load_sensitive_conv(context, hparams, "duration_predictor.pre", 1, hidden, hidden, weights.pre) ||
        !load_dds(context, hparams, "duration_predictor.dds", hidden, hparams.duration_dds_layer_count, weights.dds) ||
        !load_sensitive_conv(context, hparams, "duration_predictor.projection", 1, hidden, hidden,
                             weights.projection)) {
        return SYNTH_ERR_GGUF;
    }
    weights.affine_bias =
        find_tensor(context, hparams, "duration_predictor.affine.bias", { 1, 2 }, TensorRole::Sensitive);
    weights.affine_log_scale =
        find_tensor(context, hparams, "duration_predictor.affine.log_scale", { 1, 2 }, TensorRole::Sensitive);
    if (weights.affine_bias == nullptr || weights.affine_log_scale == nullptr) {
        return SYNTH_ERR_GGUF;
    }

    weights.flows.clear();
    weights.flows.resize(hparams.duration_flow_count);
    for (uint32_t flow = 0; flow < hparams.duration_flow_count; ++flow) {
        DurationFlowWeights & item   = weights.flows[flow];
        const std::string     prefix = "duration_predictor.flows." + std::to_string(flow);
        if (!load_sensitive_conv(context, hparams, prefix + ".pre", 1, 1, hidden, item.pre) ||
            !load_dds(context, hparams, prefix + ".dds", hidden, hparams.duration_dds_layer_count, item.dds) ||
            !load_sensitive_conv(context, hparams, prefix + ".projection", 1, hidden, spline_parameters,
                                 item.projection)) {
            return SYNTH_ERR_GGUF;
        }
    }
    return SYNTH_OK;
}

synth_status_t build_flow_weights(ggml_context * context, const HParams & hparams, FlowWeights & weights) {
    if (context == nullptr || hparams.inter_channels == 0 || hparams.inter_channels % 2 != 0 ||
        hparams.hidden_channels == 0 || hparams.flow_block_count == 0 || hparams.flow_kernel_size == 0 ||
        hparams.flow_wn_layer_count == 0 || !hparams.flow_mean_only) {
        return SYNTH_ERR_INVALID_ARG;
    }
    const int64_t hidden = hparams.hidden_channels;
    const int64_t half   = hparams.inter_channels / 2;
    weights.blocks.clear();
    weights.blocks.resize(hparams.flow_block_count);
    for (uint32_t block = 0; block < hparams.flow_block_count; ++block) {
        FlowBlockWeights & item   = weights.blocks[block];
        const std::string  prefix = "flow.blocks." + std::to_string(block);
        item.conditioning         = {};
        if ((hparams.conditioning_channels > 0 &&
             !load_conv(context, hparams, prefix + ".conditioning", 1, hparams.conditioning_channels,
                        2 * hidden * hparams.flow_wn_layer_count, item.conditioning)) ||
            !load_conv(context, hparams, prefix + ".pre", 1, half, hidden, item.pre) ||
            !load_conv(context, hparams, prefix + ".projection", 1, hidden, half, item.projection)) {
            return SYNTH_ERR_GGUF;
        }
        item.wn_layers.resize(hparams.flow_wn_layer_count);
        for (uint32_t layer = 0; layer < hparams.flow_wn_layer_count; ++layer) {
            FlowWNLayerWeights & wn                     = item.wn_layers[layer];
            const std::string    layer_prefix           = prefix + ".wn.layers." + std::to_string(layer);
            const int64_t        residual_skip_channels = layer + 1 < hparams.flow_wn_layer_count ? 2 * hidden : hidden;
            if (!load_conv(context, hparams, layer_prefix + ".input", hparams.flow_kernel_size, hidden, 2 * hidden,
                           wn.input) ||
                !load_conv(context, hparams, layer_prefix + ".residual_skip", 1, hidden, residual_skip_channels,
                           wn.residual_skip)) {
                return SYNTH_ERR_GGUF;
            }
        }
    }
    return SYNTH_OK;
}

synth_status_t build_decoder_weights(ggml_context * context, const HParams & hparams, DecoderWeights & weights) {
    if (context == nullptr || hparams.inter_channels == 0 || !decoder_hparams_fit(hparams)) {
        return SYNTH_ERR_INVALID_ARG;
    }
    const int64_t initial = hparams.decoder_initial_channels;
    weights.conditioning  = {};
    if ((hparams.conditioning_channels > 0 &&
         !load_conv(context, hparams, "decoder.conditioning", 1, hparams.conditioning_channels, initial,
                    weights.conditioning)) ||
        !load_conv(context, hparams, "decoder.pre", 7, hparams.inter_channels, initial, weights.pre)) {
        return SYNTH_ERR_GGUF;
    }
    weights.stages.clear();
    weights.stages.resize(hparams.decoder_upsample_rates.size());
    int64_t channels = initial;
    for (size_t stage = 0; stage < weights.stages.size(); ++stage) {
        DecoderStageWeights & item          = weights.stages[stage];
        const int64_t         next_channels = channels / 2;
        const std::string     prefix        = "decoder.upsample." + std::to_string(stage);
        item.transpose_conv.weight          = find_tensor(
            context, hparams, prefix + ".transpose_conv.weight",
            { hparams.decoder_upsample_kernel_sizes[stage], next_channels, channels }, TensorRole::TransposeWeight);
        item.transpose_conv.bias =
            find_tensor(context, hparams, prefix + ".transpose_conv.bias", { next_channels }, TensorRole::Sensitive);
        if (item.transpose_conv.weight == nullptr || item.transpose_conv.bias == nullptr) {
            return SYNTH_ERR_GGUF;
        }
        item.resblocks.clear();
        item.resblocks.resize(hparams.decoder_resblock_kernel_sizes.size());
        for (size_t branch = 0; branch < item.resblocks.size(); ++branch) {
            DecoderResBlockWeights & block        = item.resblocks[branch];
            const std::string        block_prefix = prefix + ".resblocks." + std::to_string(branch);
            block.conv1.resize(hparams.decoder_resblock_dilations[branch].size());
            block.conv2.resize(hparams.decoder_resblock_dilations[branch].size());
            for (size_t layer = 0; layer < block.conv1.size(); ++layer) {
                const int64_t kernel = hparams.decoder_resblock_kernel_sizes[branch];
                if (!load_conv(context, hparams, block_prefix + ".conv1." + std::to_string(layer), kernel,
                               next_channels, next_channels, block.conv1[layer]) ||
                    !load_conv(context, hparams, block_prefix + ".conv2." + std::to_string(layer), kernel,
                               next_channels, next_channels, block.conv2[layer])) {
                    return SYNTH_ERR_GGUF;
                }
            }
        }
        channels = next_channels;
    }
    weights.post_weight =
        find_tensor(context, hparams, "decoder.post.weight", { 7, channels, 1 }, TensorRole::MatrixWeight);
    return weights.post_weight == nullptr ? SYNTH_ERR_GGUF : SYNTH_OK;
}

synth_status_t build_text_weights(ggml_context * context, const HParams & hparams, TextWeights & weights) {
    if (context == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    const int64_t hidden        = hparams.hidden_channels;
    const int64_t filter        = hparams.filter_channels;
    const int64_t head_channels = hidden / hparams.text_head_count;
    const int64_t relative_rows = 2 * static_cast<int64_t>(hparams.text_attention_window) + 1;

    weights.token_embedding = find_tensor(context, hparams, "text_encoder.token_embedding.weight",
                                          { hidden, hparams.vocab_size }, TensorRole::Sensitive);
    if (weights.token_embedding == nullptr) {
        return SYNTH_ERR_GGUF;
    }

    weights.blocks.clear();
    weights.blocks.resize(hparams.text_layer_count);
    for (uint32_t layer = 0; layer < hparams.text_layer_count; ++layer) {
        TextBlockWeights & block  = weights.blocks[layer];
        const std::string  prefix = "text_encoder.blocks." + std::to_string(layer);
        if (!load_sensitive_conv(context, hparams, prefix + ".attention.query", 1, hidden, hidden, block.query) ||
            !load_sensitive_conv(context, hparams, prefix + ".attention.key", 1, hidden, hidden, block.key) ||
            !load_sensitive_conv(context, hparams, prefix + ".attention.value", 1, hidden, hidden, block.value) ||
            !load_sensitive_conv(context, hparams, prefix + ".attention.output", 1, hidden, hidden, block.output) ||
            !load_norm(context, hparams, prefix + ".attention_norm", hidden, block.attention_norm) ||
            !load_sensitive_conv(context, hparams, prefix + ".ffn.input", hparams.text_ffn_kernel_size, hidden, filter,
                                 block.ffn_input) ||
            !load_sensitive_conv(context, hparams, prefix + ".ffn.output", hparams.text_ffn_kernel_size, filter, hidden,
                                 block.ffn_output) ||
            !load_norm(context, hparams, prefix + ".ffn_norm", hidden, block.ffn_norm)) {
            return SYNTH_ERR_GGUF;
        }
        block.relative_key   = find_tensor(context, hparams, prefix + ".attention.relative_key.weight",
                                           { head_channels, relative_rows }, TensorRole::Sensitive);
        block.relative_value = find_tensor(context, hparams, prefix + ".attention.relative_value.weight",
                                           { head_channels, relative_rows }, TensorRole::Sensitive);
        if (block.relative_key == nullptr || block.relative_value == nullptr) {
            return SYNTH_ERR_GGUF;
        }
    }

    if (!load_sensitive_conv(context, hparams, "text_encoder.projection", 1, hidden, 2 * hparams.inter_channels,
                             weights.projection)) {
        return SYNTH_ERR_GGUF;
    }
    return SYNTH_OK;
}

}  // namespace synth::vits
