#include "weights.h"

#include "gguf-metadata.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>

namespace synth::kokoro {

namespace {

constexpr uint32_t kArchitectureVersion = 1;
constexpr uint32_t kFormatVersion       = 1;
// The pad token wraps the mapped sequence, and the upstream style table is
// indexed by the unpadded phoneme count, so two tokens separate them.
constexpr uint64_t kPadTokenCount       = 2;

GgufMetadata metadata(const gguf_context * gguf) {
    return GgufMetadata(gguf, "kokoro");
}

bool read_identity(const GgufMetadata & meta, HParams & hparams) {
    uint32_t format_version       = 0;
    uint32_t architecture_version = 0;
    if (!meta.require_string("general.architecture", "kokoro") ||
        !meta.require_string("synthesize.model_family", "kokoro") ||
        !meta.string("synthesize.model_variant", hparams.model_variant) ||
        !meta.u32("synthesize.format_version", format_version) ||
        !meta.u32("synthesize.kokoro.architecture_version", architecture_version)) {
        return false;
    }
    if (format_version != kFormatVersion || architecture_version != kArchitectureVersion) {
        std::fprintf(stderr, "kokoro: unsupported format %u or architecture %u\n", format_version,
                     architecture_version);
        return false;
    }
    return !hparams.model_variant.empty();
}

bool read_quantization(const GgufMetadata & meta, HParams & hparams) {
    std::string profile;
    if (!meta.string("synthesize.quantization.profile", profile) ||
        !meta.u32("synthesize.quantization.profile_version", hparams.quantization_profile_version)) {
        return false;
    }
    if (profile == "F32") {
        hparams.quantization_profile = QuantizationProfile::F32;
    } else if (profile == "F16") {
        hparams.quantization_profile = QuantizationProfile::F16;
    } else if (profile == "Q8_MIXED") {
        hparams.quantization_profile = QuantizationProfile::Q8Mixed;
    } else {
        std::fprintf(stderr, "kokoro: unsupported quantization profile %s\n", profile.c_str());
        return false;
    }
    return hparams.quantization_profile_version == 1;
}

bool read_capabilities(const GgufMetadata & meta, HParams & hparams) {
    if (!meta.u32("synthesize.capabilities.input_flags", hparams.input_flags) ||
        !meta.u32("synthesize.capabilities.flags", hparams.capability_flags) ||
        !meta.u64("synthesize.capabilities.max_input_tokens", hparams.max_input_tokens) ||
        !meta.u64("synthesize.capabilities.max_output_frames", hparams.max_output_frames) ||
        !meta.f32("synthesize.capabilities.min_speaking_rate", hparams.min_speaking_rate) ||
        !meta.f32("synthesize.capabilities.max_speaking_rate", hparams.max_speaking_rate) ||
        !meta.u32("synthesize.audio.sample_rate_hz", hparams.output_sample_rate) ||
        !meta.u32("synthesize.audio.channels", hparams.output_channel_count) ||
        !meta.require_string("synthesize.audio.sample_format", "f32le")) {
        return false;
    }
    if (hparams.input_flags != (SYNTH_INPUT_SUPPORT_PHONEMES_UTF8 | SYNTH_INPUT_SUPPORT_TOKEN_IDS)) {
        std::fprintf(stderr, "kokoro: package must advertise phoneme and token input only\n");
        return false;
    }
    // Every Kokoro synthesis consumes randomness through the source module and
    // honors speaking rate, so both flags are mandatory rather than optional.
    const uint32_t required = SYNTH_MODEL_CAPABILITY_SPEAKING_RATE | SYNTH_MODEL_CAPABILITY_STOCHASTIC;
    if ((hparams.capability_flags & required) != required) {
        std::fprintf(stderr, "kokoro: package must declare speaking rate and stochastic capabilities\n");
        return false;
    }
    if (hparams.output_sample_rate == 0 || hparams.output_channel_count != 1) {
        std::fprintf(stderr, "kokoro: native audio must be mono with a positive sample rate\n");
        return false;
    }
    if (hparams.max_input_tokens <= kPadTokenCount || hparams.max_output_frames == 0) {
        std::fprintf(stderr, "kokoro: package limits must leave room for content tokens and output\n");
        return false;
    }
    if (!std::isfinite(hparams.min_speaking_rate) || !std::isfinite(hparams.max_speaking_rate) ||
        hparams.min_speaking_rate <= 0.0f || hparams.min_speaking_rate > 1.0f || hparams.max_speaking_rate < 1.0f) {
        std::fprintf(stderr, "kokoro: speaking-rate range must be positive, ordered, and contain 1.0\n");
        return false;
    }
    return true;
}

bool read_frontend(const GgufMetadata & meta, HParams & hparams) {
    if (!meta.boolean("synthesize.frontend.present", hparams.frontend_present)) {
        return false;
    }
    if (!hparams.frontend_present) {
        std::fprintf(stderr, "kokoro: package must supply the built-in symbol map frontend\n");
        return false;
    }

    std::string mapping_mode;
    std::string padding_rule;
    std::string lookup;
    uint32_t    mapped_count = 0;
    if (!meta.string("synthesize.frontend.provider", hparams.frontend_config.provider_id) ||
        !meta.u32("synthesize.frontend.contract_version", hparams.frontend_config.contract_version) ||
        !meta.string("synthesize.frontend.phoneme_mapping", mapping_mode) ||
        !meta.string("synthesize.frontend.padding_rule", padding_rule) ||
        !meta.string_array("synthesize.kokoro.symbols", hparams.frontend_config.symbols) ||
        !meta.string("synthesize.kokoro.symbols.lookup", lookup) ||
        !meta.u32("synthesize.kokoro.symbols.mapped_count", mapped_count) ||
        !meta.u32("synthesize.kokoro.symbols.pad_id", hparams.frontend_config.blank_id)) {
        return false;
    }
    if (mapping_mode != "unicode_scalar" || padding_rule != "wrap_pad_token" || lookup != "unique_scalar") {
        std::fprintf(stderr, "kokoro: unsupported frontend mapping, padding, or lookup rule\n");
        return false;
    }
    if (hparams.frontend_config.symbols.size() != hparams.n_token) {
        std::fprintf(stderr, "kokoro: symbol table must be dense over the %u embedding rows\n", hparams.n_token);
        return false;
    }
    // The vocabulary is sparse: reserved ids stay empty and are not mappable.
    uint32_t present = 0;
    for (const std::string & symbol : hparams.frontend_config.symbols) {
        if (!symbol.empty()) {
            ++present;
        }
    }
    if (present != mapped_count || mapped_count == 0) {
        std::fprintf(stderr, "kokoro: symbol table has %u mapped entries, metadata declares %u\n", present,
                     mapped_count);
        return false;
    }
    if (hparams.frontend_config.blank_id >= hparams.n_token ||
        !hparams.frontend_config.symbols[hparams.frontend_config.blank_id].empty()) {
        std::fprintf(stderr, "kokoro: the pad id must name a reserved, unmapped row\n");
        return false;
    }
    hparams.frontend_config.mapping_mode = SymbolMappingMode::UnicodeScalar;
    hparams.frontend_config.padding_rule = SymbolPaddingRule::WrapPadToken;

    std::unique_ptr<TextFrontend> frontend;
    return make_symbol_map_frontend(hparams.frontend_config, frontend) == SYNTH_OK;
}

bool read_voices(const GgufMetadata & meta, HParams & hparams) {
    uint32_t preset_count = 0;
    if (!meta.require_string("synthesize.voice.mode", "preset-catalog") ||
        !meta.boolean("synthesize.voice.has_package_default", hparams.has_package_default) ||
        !meta.u32("synthesize.voice.preset_count", preset_count)) {
        return false;
    }
    if (hparams.has_package_default) {
        std::fprintf(stderr, "kokoro: package invents no default Voice; every request selects one\n");
        return false;
    }
    if (preset_count == 0) {
        std::fprintf(stderr, "kokoro: preset catalog must not be empty\n");
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
        if (!meta.string(prefix + ".id", id) || !meta.u32(prefix + ".flags", flags)) {
            return false;
        }
        if (id.empty()) {
            std::fprintf(stderr, "kokoro: preset voice %u has an empty identifier\n", index);
            return false;
        }
        for (const std::string & existing : hparams.preset_voice_ids) {
            if (existing == id) {
                std::fprintf(stderr, "kokoro: duplicate preset voice identifier %s\n", id.c_str());
                return false;
            }
        }
        // No catalog entry may claim the package default, which does not exist.
        if ((flags & SYNTH_PRESET_VOICE_DEFAULT) != 0) {
            std::fprintf(stderr, "kokoro: preset voice %s must not claim the package default\n", id.c_str());
            return false;
        }
        hparams.preset_voice_ids.push_back(id);
        hparams.preset_voice_flags.push_back(flags);
    }
    return true;
}

bool read_dimensions(const GgufMetadata & meta, HParams & hparams) {
    if (!meta.u32("synthesize.kokoro.n_token", hparams.n_token) ||
        !meta.u32("synthesize.kokoro.hidden_dim", hparams.hidden_dim) ||
        !meta.u32("synthesize.kokoro.style_dim", hparams.style_dim) ||
        !meta.u32("synthesize.kokoro.n_layer", hparams.n_layer) ||
        !meta.u32("synthesize.kokoro.n_mels", hparams.n_mels) ||
        !meta.u32("synthesize.kokoro.max_dur", hparams.max_dur) ||
        !meta.u32("synthesize.kokoro.dim_in", hparams.dim_in) ||
        !meta.u32("synthesize.kokoro.text_encoder_kernel_size", hparams.text_encoder_kernel_size) ||
        !meta.u32("synthesize.kokoro.samples_per_frame", hparams.samples_per_frame)) {
        return false;
    }
    if (hparams.n_token == 0 || hparams.hidden_dim == 0 || hparams.style_dim == 0 || hparams.n_layer == 0 ||
        hparams.n_mels == 0 || hparams.max_dur == 0 || hparams.dim_in == 0 || hparams.samples_per_frame == 0) {
        std::fprintf(stderr, "kokoro: model dimensions must be positive\n");
        return false;
    }
    // The prosody LSTMs consume the hidden state concatenated with the style
    // half, and the shared bidirectional split needs an even hidden width.
    if (hparams.hidden_dim % 2 != 0) {
        std::fprintf(stderr, "kokoro: hidden_dim must be even for the bidirectional split\n");
        return false;
    }
    if (hparams.text_encoder_kernel_size == 0 || hparams.text_encoder_kernel_size % 2 == 0) {
        std::fprintf(stderr, "kokoro: text encoder kernel size must be odd\n");
        return false;
    }
    return true;
}

bool read_plbert(const GgufMetadata & meta, PLBertParams & plbert) {
    if (!meta.u32("synthesize.kokoro.plbert.hidden_size", plbert.hidden_size) ||
        !meta.u32("synthesize.kokoro.plbert.num_attention_heads", plbert.num_attention_heads) ||
        !meta.u32("synthesize.kokoro.plbert.intermediate_size", plbert.intermediate_size) ||
        !meta.u32("synthesize.kokoro.plbert.num_hidden_layers", plbert.num_hidden_layers) ||
        !meta.u32("synthesize.kokoro.plbert.max_position_embeddings", plbert.max_position_embeddings) ||
        !meta.u32("synthesize.kokoro.plbert.shared_layer_groups", plbert.shared_layer_groups) ||
        !meta.f32("synthesize.kokoro.plbert.layer_norm_eps", plbert.layer_norm_eps)) {
        return false;
    }
    if (plbert.hidden_size == 0 || plbert.num_attention_heads == 0 || plbert.intermediate_size == 0 ||
        plbert.num_hidden_layers == 0 || plbert.max_position_embeddings == 0) {
        std::fprintf(stderr, "kokoro: PL-BERT dimensions must be positive\n");
        return false;
    }
    if (plbert.hidden_size % plbert.num_attention_heads != 0) {
        std::fprintf(stderr, "kokoro: PL-BERT hidden size must divide across its heads\n");
        return false;
    }
    // Only the single shared ALBERT group is supported; a package that stored a
    // distinct group per layer would need a different weight catalog.
    if (plbert.shared_layer_groups != 1) {
        std::fprintf(stderr, "kokoro: only one shared ALBERT layer group is supported\n");
        return false;
    }
    if (!std::isfinite(plbert.layer_norm_eps) || plbert.layer_norm_eps <= 0.0f) {
        std::fprintf(stderr, "kokoro: PL-BERT layer-norm epsilon must be positive\n");
        return false;
    }
    return true;
}

bool read_istftnet(const GgufMetadata & meta, IStftNetParams & istftnet) {
    if (!meta.positive_i32_array("synthesize.kokoro.istftnet.upsample_rates", istftnet.upsample_rates) ||
        !meta.positive_i32_array("synthesize.kokoro.istftnet.upsample_kernel_sizes", istftnet.upsample_kernel_sizes) ||
        !meta.positive_i32_array("synthesize.kokoro.istftnet.resblock_kernel_sizes", istftnet.resblock_kernel_sizes) ||
        !meta.u32("synthesize.kokoro.istftnet.upsample_initial_channel", istftnet.upsample_initial_channel) ||
        !meta.u32("synthesize.kokoro.istftnet.gen_istft_n_fft", istftnet.gen_istft_n_fft) ||
        !meta.u32("synthesize.kokoro.istftnet.gen_istft_hop_size", istftnet.gen_istft_hop_size) ||
        !meta.require_string("synthesize.kokoro.istftnet.window", "hann_periodic") ||
        !meta.boolean("synthesize.kokoro.istftnet.center", istftnet.center)) {
        return false;
    }
    if (istftnet.upsample_rates.empty() || istftnet.upsample_rates.size() != istftnet.upsample_kernel_sizes.size() ||
        istftnet.resblock_kernel_sizes.empty() || istftnet.upsample_initial_channel == 0) {
        std::fprintf(stderr, "kokoro: iSTFTNet upsample and resblock shapes are inconsistent\n");
        return false;
    }
    istftnet.resblock_dilations.clear();
    istftnet.resblock_dilations.resize(istftnet.resblock_kernel_sizes.size());
    for (size_t branch = 0; branch < istftnet.resblock_kernel_sizes.size(); ++branch) {
        const std::string key = "synthesize.kokoro.istftnet.resblock_dilations." + std::to_string(branch);
        if (!meta.positive_i32_array(key, istftnet.resblock_dilations[branch])) {
            return false;
        }
        const uint32_t kernel = istftnet.resblock_kernel_sizes[branch];
        if (kernel == 0 || kernel % 2 == 0 || istftnet.resblock_dilations[branch].empty()) {
            std::fprintf(stderr, "kokoro: resblock branch %zu needs an odd kernel and dilations\n", branch);
            return false;
        }
        for (uint32_t dilation : istftnet.resblock_dilations[branch]) {
            const uint64_t padding = static_cast<uint64_t>(kernel - 1) * dilation / 2;
            if (padding > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
                return false;
            }
        }
    }
    uint64_t channels = istftnet.upsample_initial_channel;
    for (size_t stage = 0; stage < istftnet.upsample_rates.size(); ++stage) {
        const uint32_t rate   = istftnet.upsample_rates[stage];
        const uint32_t kernel = istftnet.upsample_kernel_sizes[stage];
        if (kernel < rate || (kernel - rate) % 2 != 0 || channels < 2 || channels % 2 != 0) {
            std::fprintf(stderr, "kokoro: upsample stage %zu has an unusable rate or kernel\n", stage);
            return false;
        }
        channels /= 2;
    }
    // A real one-sided spectrum needs an even transform whose hop divides it.
    if (istftnet.gen_istft_n_fft < 2 || istftnet.gen_istft_n_fft % 2 != 0 || istftnet.gen_istft_hop_size == 0 ||
        istftnet.gen_istft_n_fft % istftnet.gen_istft_hop_size != 0) {
        std::fprintf(stderr, "kokoro: iSTFT transform size and hop are unusable\n");
        return false;
    }
    if (!istftnet.center) {
        std::fprintf(stderr, "kokoro: only centered iSTFT framing is supported\n");
        return false;
    }
    return true;
}

bool read_source(const GgufMetadata & meta, SourceParams & source) {
    if (!meta.u32("synthesize.kokoro.source.sampling_rate", source.sampling_rate) ||
        !meta.u32("synthesize.kokoro.source.harmonic_num", source.harmonic_num) ||
        !meta.u32("synthesize.kokoro.source.upsample_scale", source.upsample_scale) ||
        !meta.f32("synthesize.kokoro.source.sine_amp", source.sine_amp) ||
        !meta.f32("synthesize.kokoro.source.noise_std", source.noise_std) ||
        !meta.f32("synthesize.kokoro.source.voiced_threshold", source.voiced_threshold)) {
        return false;
    }
    if (source.sampling_rate == 0 || source.upsample_scale == 0) {
        std::fprintf(stderr, "kokoro: source sampling rate and upsample scale must be positive\n");
        return false;
    }
    if (!std::isfinite(source.sine_amp) || source.sine_amp <= 0.0f || !std::isfinite(source.noise_std) ||
        source.noise_std < 0.0f || !std::isfinite(source.voiced_threshold)) {
        std::fprintf(stderr, "kokoro: source amplitudes must be finite and non-negative\n");
        return false;
    }
    return true;
}

bool read_voice_table(const GgufMetadata & meta, HParams & hparams) {
    if (!meta.u32("synthesize.kokoro.voice.rows", hparams.voice_rows) ||
        !meta.u32("synthesize.kokoro.voice.dim", hparams.voice_dim) ||
        !meta.require_string("synthesize.kokoro.voice.row_rule", "final_token_count_minus_three") ||
        !meta.u32("synthesize.kokoro.voice.decoder_offset", hparams.voice_decoder_offset) ||
        !meta.u32("synthesize.kokoro.voice.prosody_offset", hparams.voice_prosody_offset) ||
        !meta.boolean("synthesize.kokoro.adain.instance_norm_affine", hparams.adain_instance_norm_affine) ||
        !meta.f32("synthesize.kokoro.adain.eps", hparams.adain_eps)) {
        return false;
    }
    if (hparams.voice_rows == 0 || hparams.voice_dim != 2 * hparams.style_dim) {
        std::fprintf(stderr, "kokoro: voice table must hold a decoder and prosody half per row\n");
        return false;
    }
    if (hparams.voice_decoder_offset != 0 || hparams.voice_prosody_offset != hparams.style_dim) {
        std::fprintf(stderr, "kokoro: voice halves must be contiguous decoder-then-prosody\n");
        return false;
    }
    // The checkpoint carries no InstanceNorm affine parameters; a package that
    // claimed otherwise would reference tensors the converter never emits.
    if (hparams.adain_instance_norm_affine) {
        std::fprintf(stderr, "kokoro: affine instance norm is not part of this architecture\n");
        return false;
    }
    if (!std::isfinite(hparams.adain_eps) || hparams.adain_eps <= 0.0f) {
        std::fprintf(stderr, "kokoro: AdaIN epsilon must be positive\n");
        return false;
    }
    // The style table is addressed by unpadded token count, so it must be able
    // to cover every input the package accepts.
    if (static_cast<uint64_t>(hparams.voice_rows) + kPadTokenCount < hparams.max_input_tokens) {
        std::fprintf(stderr, "kokoro: voice table cannot address the declared maximum input\n");
        return false;
    }
    return true;
}

bool derived_frame_chain_fits(const HParams & hparams) {
    uint64_t product = 2;  // the prosody residual block upsamples once
    for (uint32_t rate : hparams.istftnet.upsample_rates) {
        product *= rate;
    }
    product *= hparams.istftnet.gen_istft_hop_size;
    if (product != hparams.samples_per_frame) {
        std::fprintf(stderr, "kokoro: declared %u samples per frame but the chain yields %llu\n",
                     hparams.samples_per_frame, static_cast<unsigned long long>(product));
        return false;
    }
    if (hparams.source.upsample_scale * 2ull != product) {
        std::fprintf(stderr, "kokoro: source upsample scale disagrees with the frame chain\n");
        return false;
    }
    if (hparams.output_sample_rate != hparams.source.sampling_rate) {
        std::fprintf(stderr, "kokoro: source sampling rate must match the native output rate\n");
        return false;
    }
    return true;
}

}  // namespace

bool resolve_style_row(const HParams & hparams, uint64_t final_token_count, uint32_t & row) {
    row = 0;
    if (final_token_count <= kPadTokenCount + 1) {
        return false;
    }
    const uint64_t index = final_token_count - kPadTokenCount - 1;
    if (index >= hparams.voice_rows) {
        return false;
    }
    row = static_cast<uint32_t>(index);
    return true;
}

synth_status_t read_hparams(const gguf_context * gguf, HParams & hparams) {
    if (gguf == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    hparams                 = HParams{};
    const GgufMetadata meta = metadata(gguf);

    if (!read_identity(meta, hparams)) {
        return SYNTH_ERR_UNSUPPORTED_ARCH;
    }
    if (!read_quantization(meta, hparams) || !read_capabilities(meta, hparams) || !read_dimensions(meta, hparams) ||
        !read_plbert(meta, hparams.plbert) || !read_istftnet(meta, hparams.istftnet) ||
        !read_source(meta, hparams.source) || !read_voice_table(meta, hparams) || !read_frontend(meta, hparams) ||
        !read_voices(meta, hparams) || !derived_frame_chain_fits(hparams)) {
        return SYNTH_ERR_GGUF;
    }
    return SYNTH_OK;
}

}  // namespace synth::kokoro
