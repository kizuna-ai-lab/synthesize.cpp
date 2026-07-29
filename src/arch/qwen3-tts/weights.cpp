#include "weights.h"

#include "gguf-metadata.h"
#include "gguf.h"

#include <cstdio>

namespace synth::qwen3tts {

namespace {

constexpr uint32_t kFormatVersion       = 1;
constexpr uint32_t kArchitectureVersion = 1;

GgufMetadata metadata(const gguf_context * gguf) {
    return GgufMetadata(gguf, "qwen3-tts");
}

bool read_identity(const GgufMetadata & meta, HParams & hparams) {
    uint32_t format_version       = 0;
    uint32_t architecture_version = 0;
    if (!meta.require_string("general.architecture", "qwen3-tts") ||
        !meta.require_string("synthesize.model_family", "qwen3-tts") ||
        !meta.string("synthesize.model_variant", hparams.model_variant) ||
        !meta.u32("synthesize.format_version", format_version) ||
        !meta.u32("synthesize.qwen3-tts.architecture_version", architecture_version)) {
        return false;
    }
    if (format_version != kFormatVersion || architecture_version != kArchitectureVersion) {
        std::fprintf(stderr, "qwen3-tts: unsupported format %u or architecture %u\n", format_version,
                     architecture_version);
        return false;
    }
    hparams.architecture_version = architecture_version;
    return !hparams.model_variant.empty();
}

bool read_quantization(const GgufMetadata & meta, HParams & hparams) {
    std::string profile;
    if (!meta.string("synthesize.quantization.profile", profile) ||
        !meta.u32("synthesize.quantization.profile_version", hparams.quantization_profile_version)) {
        return false;
    }
    // BF16 rather than F32 is this family's source profile: the talker
    // checkpoint stores bfloat16 and widening it would describe a different
    // model. See docs/port-validation.md, "Choosing the Oracle's dtype".
    if (profile == "BF16") {
        hparams.quantization_profile = QuantizationProfile::BF16;
    } else if (profile == "F16") {
        hparams.quantization_profile = QuantizationProfile::F16;
    } else if (profile == "Q8_MIXED") {
        hparams.quantization_profile = QuantizationProfile::Q8Mixed;
    } else if (profile == "Q5_K_MIXED") {
        hparams.quantization_profile = QuantizationProfile::Q5KMixed;
    } else {
        std::fprintf(stderr, "qwen3-tts: unsupported quantization profile %s\n", profile.c_str());
        return false;
    }
    return hparams.quantization_profile_version == 1;
}

bool read_capabilities(const GgufMetadata & meta, HParams & hparams) {
    std::string sample_format;
    if (!meta.u32("synthesize.capabilities.input_flags", hparams.input_flags) ||
        !meta.u32("synthesize.capabilities.flags", hparams.capability_flags) ||
        !meta.u64("synthesize.capabilities.max_input_tokens", hparams.max_input_tokens) ||
        !meta.u64("synthesize.capabilities.max_output_frames", hparams.max_output_frames) ||
        !meta.f32("synthesize.capabilities.min_speaking_rate", hparams.min_speaking_rate) ||
        !meta.f32("synthesize.capabilities.max_speaking_rate", hparams.max_speaking_rate) ||
        !meta.u32("synthesize.audio.sample_rate_hz", hparams.output_sample_rate) ||
        !meta.u32("synthesize.audio.channels", hparams.output_channel_count) ||
        !meta.string("synthesize.audio.sample_format", sample_format)) {
        return false;
    }
    if (sample_format != "f32le") {
        std::fprintf(stderr, "qwen3-tts: unsupported sample format %s\n", sample_format.c_str());
        return false;
    }
    if (hparams.output_channel_count != 1 || hparams.output_sample_rate == 0) {
        std::fprintf(stderr, "qwen3-tts: package declares %u channels at %u Hz\n", hparams.output_channel_count,
                     hparams.output_sample_rate);
        return false;
    }
    if (hparams.max_input_tokens == 0 || hparams.max_output_frames == 0) {
        std::fprintf(stderr, "qwen3-tts: package declares a zero input or output limit\n");
        return false;
    }
    if (!meta.f32("synthesize.qwen3-tts.sampling.temperature", hparams.talker_sampling.temperature) ||
        !meta.u32("synthesize.qwen3-tts.sampling.top_k", hparams.talker_sampling.top_k) ||
        !meta.f32("synthesize.qwen3-tts.sampling.top_p", hparams.talker_sampling.top_p) ||
        !meta.f32("synthesize.qwen3-tts.sampling.repetition_penalty", hparams.talker_sampling.repetition_penalty) ||
        !meta.f32("synthesize.qwen3-tts.sampling.predictor.temperature", hparams.predictor_sampling.temperature) ||
        !meta.u32("synthesize.qwen3-tts.sampling.predictor.top_k", hparams.predictor_sampling.top_k) ||
        !meta.f32("synthesize.qwen3-tts.sampling.predictor.top_p", hparams.predictor_sampling.top_p)) {
        std::fprintf(stderr, "qwen3-tts: package declares no sampling defaults; re-cut it\n");
        return false;
    }
    // A package that carries these has to carry usable ones. A temperature of
    // zero would divide, and a penalty below one rewards repetition rather than
    // discouraging it, which is the opposite of what the field means.
    for (const SamplingDefaults & sampling : { hparams.talker_sampling, hparams.predictor_sampling }) {
        if (!(sampling.temperature > 0.0f) || !(sampling.top_p > 0.0f) || sampling.top_p > 1.0f ||
            !(sampling.repetition_penalty >= 1.0f)) {
            std::fprintf(stderr, "qwen3-tts: package declares unusable sampling defaults\n");
            return false;
        }
    }
    // Upstream's entry point has no speed parameter, so this family declares a
    // degenerate range. A package claiming otherwise would promise a control the
    // graph cannot honour.
    if (hparams.min_speaking_rate != 1.0f || hparams.max_speaking_rate != 1.0f) {
        std::fprintf(stderr,
                     "qwen3-tts: speaking rate is not adjustable in this family, "
                     "but the package declares [%f, %f]\n",
                     static_cast<double>(hparams.min_speaking_rate), static_cast<double>(hparams.max_speaking_rate));
        return false;
    }
    return true;
}

bool read_talker(const GgufMetadata & meta, HParams & hparams) {
    TalkerParams & talker = hparams.talker;
    if (!meta.u32("synthesize.qwen3-tts.talker.layer_count", talker.layer_count) ||
        !meta.u32("synthesize.qwen3-tts.talker.hidden_size", talker.hidden_size) ||
        !meta.u32("synthesize.qwen3-tts.talker.attention_head_count", talker.attention_head_count) ||
        !meta.u32("synthesize.qwen3-tts.talker.key_value_head_count", talker.key_value_head_count) ||
        !meta.u32("synthesize.qwen3-tts.talker.head_dim", talker.head_dim) ||
        !meta.u32("synthesize.qwen3-tts.talker.intermediate_size", talker.intermediate_size) ||
        !meta.u32("synthesize.qwen3-tts.talker.codec_vocab_size", talker.codec_vocab_size) ||
        !meta.u32("synthesize.qwen3-tts.talker.text_vocab_size", talker.text_vocab_size) ||
        !meta.u32("synthesize.qwen3-tts.talker.text_hidden_size", talker.text_hidden_size) ||
        !meta.u32("synthesize.qwen3-tts.talker.code_group_count", talker.code_group_count) ||
        !meta.f32("synthesize.qwen3-tts.talker.rms_norm_eps", talker.rms_norm_eps) ||
        !meta.f32("synthesize.qwen3-tts.talker.rope_theta", talker.rope_theta) ||
        !meta.string("synthesize.qwen3-tts.talker.rope_type", talker.rope_type)) {
        return false;
    }
    if (talker.layer_count == 0 || talker.hidden_size == 0 || talker.head_dim == 0 ||
        talker.attention_head_count == 0 || talker.key_value_head_count == 0) {
        std::fprintf(stderr, "qwen3-tts: talker geometry contains a zero dimension\n");
        return false;
    }
    // Grouped-query attention: every key/value head serves a whole number of
    // query heads, and a package violating that would silently mis-broadcast.
    if (talker.attention_head_count % talker.key_value_head_count != 0) {
        std::fprintf(stderr, "qwen3-tts: %u query heads do not divide into %u key/value heads\n",
                     talker.attention_head_count, talker.key_value_head_count);
        return false;
    }
    // The runtime implements plain rope. A package asking for anything else is
    // rejected rather than quietly given the wrong positional encoding.
    if (talker.rope_type != "1d") {
        std::fprintf(stderr, "qwen3-tts: unsupported rope type %s\n", talker.rope_type.c_str());
        return false;
    }
    if (talker.rms_norm_eps <= 0.0f || talker.rope_theta <= 0.0f) {
        std::fprintf(stderr, "qwen3-tts: non-positive rms_norm_eps or rope_theta\n");
        return false;
    }
    return true;
}

bool read_code_predictor(const GgufMetadata & meta, HParams & hparams) {
    CodePredictorParams & cp = hparams.code_predictor;
    if (!meta.u32("synthesize.qwen3-tts.code_predictor.layer_count", cp.layer_count) ||
        !meta.u32("synthesize.qwen3-tts.code_predictor.hidden_size", cp.hidden_size) ||
        !meta.u32("synthesize.qwen3-tts.code_predictor.attention_head_count", cp.attention_head_count) ||
        !meta.u32("synthesize.qwen3-tts.code_predictor.key_value_head_count", cp.key_value_head_count) ||
        !meta.u32("synthesize.qwen3-tts.code_predictor.head_dim", cp.head_dim) ||
        !meta.u32("synthesize.qwen3-tts.code_predictor.vocab_size", cp.vocab_size) ||
        !meta.u32("synthesize.qwen3-tts.code_predictor.code_group_count", cp.code_group_count)) {
        return false;
    }
    if (cp.layer_count == 0 || cp.hidden_size == 0 || cp.vocab_size == 0 || cp.code_group_count == 0) {
        std::fprintf(stderr, "qwen3-tts: code predictor geometry contains a zero dimension\n");
        return false;
    }
    if (cp.attention_head_count % cp.key_value_head_count != 0) {
        std::fprintf(stderr, "qwen3-tts: code predictor query heads do not divide its key/value heads\n");
        return false;
    }
    // Both heads describe the same frame, so a disagreement about how many
    // codes a frame holds would desynchronise the two loops.
    if (cp.code_group_count != hparams.talker.code_group_count) {
        std::fprintf(stderr, "qwen3-tts: talker declares %u code groups, predictor %u\n",
                     hparams.talker.code_group_count, cp.code_group_count);
        return false;
    }
    return true;
}

bool read_codec_decoder(const GgufMetadata & meta, HParams & hparams) {
    CodecDecoderParams & decoder = hparams.codec.decoder;
    const std::string    prefix  = "synthesize.qwen3-tts.codec.decoder.";
    if (!meta.u32(prefix + "latent_dim", decoder.latent_dim) || !meta.u32(prefix + "dim", decoder.dim) ||
        !meta.u32(prefix + "codebook_dim", decoder.codebook_dim) ||
        !meta.u32(prefix + "codebook_size", decoder.codebook_size) ||
        !meta.u32(prefix + "quantizer_count", decoder.quantizer_count) ||
        !meta.u32(prefix + "semantic_quantizer_count", decoder.semantic_quantizer_count) ||
        !meta.u32(prefix + "hidden_size", decoder.hidden_size) ||
        !meta.u32(prefix + "intermediate_size", decoder.intermediate_size) ||
        !meta.u32(prefix + "layer_count", decoder.layer_count) ||
        !meta.u32(prefix + "attention_head_count", decoder.attention_head_count) ||
        !meta.u32(prefix + "key_value_head_count", decoder.key_value_head_count) ||
        !meta.u32(prefix + "head_dim", decoder.head_dim) ||
        !meta.u32(prefix + "sliding_window", decoder.sliding_window) ||
        !meta.f32(prefix + "rms_norm_eps", decoder.rms_norm_eps) ||
        !meta.f32(prefix + "rope_theta", decoder.rope_theta) ||
        !meta.positive_i32_array(prefix + "upsample_rates", decoder.upsample_rates) ||
        !meta.positive_i32_array(prefix + "upsampling_ratios", decoder.upsampling_ratios)) {
        return false;
    }
    if (decoder.latent_dim == 0 || decoder.dim == 0 || decoder.codebook_dim == 0 || decoder.codebook_size == 0 ||
        decoder.hidden_size == 0 || decoder.intermediate_size == 0 || decoder.layer_count == 0 ||
        decoder.head_dim == 0 || decoder.key_value_head_count == 0 || decoder.upsample_rates.empty() ||
        decoder.upsampling_ratios.empty()) {
        std::fprintf(stderr, "qwen3-tts: codec decoder geometry contains a zero dimension\n");
        return false;
    }
    if (decoder.rms_norm_eps <= 0.0f || decoder.rope_theta <= 0.0f) {
        std::fprintf(stderr, "qwen3-tts: codec decoder declares a non-positive rms_norm_eps or rope_theta\n");
        return false;
    }
    if (decoder.attention_head_count % decoder.key_value_head_count != 0) {
        std::fprintf(stderr, "qwen3-tts: codec decoder query heads do not divide its key/value heads\n");
        return false;
    }
    // The quantizer runs at half the codebook dimension, with a projection on
    // either side; an odd width would round the halving and shift every table.
    if (decoder.codebook_dim % 2 != 0) {
        std::fprintf(stderr, "qwen3-tts: codec codebook_dim %u is not even\n", decoder.codebook_dim);
        return false;
    }
    if (decoder.semantic_quantizer_count == 0 || decoder.semantic_quantizer_count >= decoder.quantizer_count) {
        std::fprintf(stderr, "qwen3-tts: %u semantic quantizers leave no acoustic groups behind them\n",
                     decoder.semantic_quantizer_count);
        return false;
    }
    // The talker and the codec must agree on how many codes a frame holds, or
    // the decoder would be handed a frame it cannot reconstruct.
    if (decoder.quantizer_count != hparams.talker.code_group_count) {
        std::fprintf(stderr, "qwen3-tts: the codec takes %u code groups but the talker emits %u\n",
                     decoder.quantizer_count, hparams.talker.code_group_count);
        return false;
    }
    // The residual stack halves its width once per rate, so the narrowest stage
    // must still be a whole number of channels.
    uint32_t narrowest = decoder.dim;
    for (size_t stage = 0; stage < decoder.upsample_rates.size(); ++stage) {
        if (narrowest % 2 != 0) {
            std::fprintf(stderr, "qwen3-tts: codec decoder width %u does not halve %zu times\n", decoder.dim,
                         decoder.upsample_rates.size());
            return false;
        }
        narrowest /= 2;
    }
    // Every upsample factor multiplies out to exactly one frame of samples.
    // Deriving this rather than trusting the hop is what catches a package whose
    // stack and frame geometry disagree, which would drift the output length.
    uint64_t total = 1;
    for (const std::vector<uint32_t> & factors : { decoder.upsample_rates, decoder.upsampling_ratios }) {
        for (uint32_t factor : factors) {
            total *= factor;
        }
    }
    if (total != hparams.codec.hop_length) {
        std::fprintf(stderr, "qwen3-tts: the codec upsample factors multiply to %llu, not the hop %u\n",
                     static_cast<unsigned long long>(total), hparams.codec.hop_length);
        return false;
    }
    return true;
}

bool read_codec(const GgufMetadata & meta, HParams & hparams) {
    CodecParams & codec = hparams.codec;
    if (!meta.u32("synthesize.qwen3-tts.codec.sample_rate", codec.sample_rate) ||
        !meta.u32("synthesize.qwen3-tts.codec.hop_length", codec.hop_length) ||
        !meta.f32("synthesize.qwen3-tts.codec.frame_rate_hz", codec.frame_rate_hz)) {
        return false;
    }
    if (codec.hop_length == 0 || codec.frame_rate_hz <= 0.0f) {
        std::fprintf(stderr, "qwen3-tts: codec declares a zero hop or frame rate\n");
        return false;
    }
    if (codec.sample_rate != hparams.output_sample_rate) {
        std::fprintf(stderr, "qwen3-tts: codec runs at %u Hz but the package declares %u Hz\n", codec.sample_rate,
                     hparams.output_sample_rate);
        return false;
    }
    // One frame is exactly hop_length samples. Deriving the relation rather than
    // trusting three independent numbers is what catches a package whose frame
    // rate and hop disagree, which would drift the audio length per frame.
    const double implied = static_cast<double>(codec.sample_rate) / static_cast<double>(codec.hop_length);
    if (implied < codec.frame_rate_hz - 1e-6 || implied > codec.frame_rate_hz + 1e-6) {
        std::fprintf(stderr, "qwen3-tts: %u Hz over hop %u is %f frames per second, not %f\n", codec.sample_rate,
                     codec.hop_length, implied, static_cast<double>(codec.frame_rate_hz));
        return false;
    }
    return read_codec_decoder(meta, hparams);
}

bool read_tokens(const GgufMetadata & meta, HParams & hparams) {
    SpecialTokens & t = hparams.tokens;
    return meta.u32("synthesize.qwen3-tts.token.tts_bos_token_id", t.tts_bos) &&
           meta.u32("synthesize.qwen3-tts.token.tts_eos_token_id", t.tts_eos) &&
           meta.u32("synthesize.qwen3-tts.token.tts_pad_token_id", t.tts_pad) &&
           meta.u32("synthesize.qwen3-tts.token.im_start_token_id", t.im_start) &&
           meta.u32("synthesize.qwen3-tts.token.im_end_token_id", t.im_end) &&
           meta.u32("synthesize.qwen3-tts.token.assistant_token_id", t.assistant) &&
           meta.u32("synthesize.qwen3-tts.token.codec_bos_id", t.codec_bos) &&
           meta.u32("synthesize.qwen3-tts.token.codec_eos_token_id", t.codec_eos) &&
           meta.u32("synthesize.qwen3-tts.token.codec_pad_id", t.codec_pad) &&
           meta.u32("synthesize.qwen3-tts.token.codec_think_id", t.codec_think) &&
           meta.u32("synthesize.qwen3-tts.token.codec_nothink_id", t.codec_nothink) &&
           meta.u32("synthesize.qwen3-tts.token.codec_think_bos_id", t.codec_think_bos) &&
           meta.u32("synthesize.qwen3-tts.token.codec_think_eos_id", t.codec_think_eos);
}

bool read_voices(const GgufMetadata & meta, HParams & hparams) {
    std::string mode;
    uint32_t    preset_count = 0;
    if (!meta.string("synthesize.voice.mode", mode) ||
        !meta.boolean("synthesize.voice.has_package_default", hparams.has_package_default) ||
        !meta.u32("synthesize.voice.preset_count", preset_count)) {
        return false;
    }
    if (mode != "preset-catalog" || preset_count == 0) {
        std::fprintf(stderr, "qwen3-tts: unsupported voice mode %s with %u presets\n", mode.c_str(), preset_count);
        return false;
    }

    std::vector<std::string> names;
    std::vector<uint32_t>    token_ids;
    std::vector<std::string> dialects;
    if (!meta.string_array("synthesize.qwen3-tts.speakers.names", names) ||
        !meta.positive_i32_array("synthesize.qwen3-tts.speakers.token_ids", token_ids) ||
        !meta.string_array("synthesize.qwen3-tts.speakers.dialect_override", dialects)) {
        return false;
    }
    if (names.size() != token_ids.size() || names.size() != dialects.size() || names.size() != preset_count) {
        std::fprintf(stderr, "qwen3-tts: speaker catalog is %zu/%zu/%zu against %u presets\n", names.size(),
                     token_ids.size(), dialects.size(), preset_count);
        return false;
    }

    hparams.preset_voices.clear();
    hparams.preset_voices.reserve(preset_count);
    for (uint32_t index = 0; index < preset_count; ++index) {
        std::string       declared;
        uint32_t          flags  = 0;
        const std::string prefix = "synthesize.voice." + std::to_string(index) + ".";
        if (!meta.string(prefix + "id", declared) || !meta.u32(prefix + "flags", flags)) {
            return false;
        }
        // The generic Voice list and the family speaker table describe the same
        // catalog. If they disagree the package is internally inconsistent, and
        // trusting either one would pick a different speaker than the other.
        if (declared != names[index]) {
            std::fprintf(stderr, "qwen3-tts: voice %u is %s in the catalog and %s in the speaker table\n", index,
                         declared.c_str(), names[index].c_str());
            return false;
        }
        if (token_ids[index] >= hparams.talker.codec_vocab_size) {
            std::fprintf(stderr, "qwen3-tts: speaker %s has token id %u outside the codec vocabulary\n",
                         names[index].c_str(), token_ids[index]);
            return false;
        }
        hparams.preset_voices.push_back(PresetVoice{ names[index], token_ids[index], dialects[index], flags });
    }
    // This family names no package default; every request selects a Voice.
    return !hparams.has_package_default;
}

bool read_languages(const GgufMetadata & meta, HParams & hparams) {
    if (!meta.string_array("synthesize.qwen3-tts.languages.names", hparams.language_names) ||
        !meta.positive_i32_array("synthesize.qwen3-tts.languages.token_ids", hparams.language_token_ids)) {
        return false;
    }
    if (hparams.language_names.empty() || hparams.language_names.size() != hparams.language_token_ids.size()) {
        std::fprintf(stderr, "qwen3-tts: %zu language names against %zu token ids\n", hparams.language_names.size(),
                     hparams.language_token_ids.size());
        return false;
    }
    // Every dialect a speaker can pin must exist as a language token, or that
    // speaker would resolve to nothing at request time.
    for (const PresetVoice & voice : hparams.preset_voices) {
        if (voice.dialect_override.empty()) {
            continue;
        }
        bool found = false;
        for (const std::string & name : hparams.language_names) {
            found = found || name == voice.dialect_override;
        }
        if (!found) {
            std::fprintf(stderr, "qwen3-tts: speaker %s pins dialect %s, which is not a language\n", voice.id.c_str(),
                         voice.dialect_override.c_str());
            return false;
        }
    }
    return true;
}

bool read_frontend(const GgufMetadata & meta, HParams & hparams) {
    if (!meta.boolean("synthesize.frontend.present", hparams.frontend_present)) {
        return false;
    }
    if (!hparams.frontend_present) {
        std::fprintf(stderr, "qwen3-tts: this family consumes raw text and needs a frontend\n");
        return false;
    }
    if (!meta.string("synthesize.frontend.provider", hparams.frontend_provider) ||
        !meta.u32("synthesize.frontend.contract_version", hparams.frontend_contract_version)) {
        return false;
    }
    if (hparams.frontend_provider != "synthesize.qwen_bpe" || hparams.frontend_contract_version != 1) {
        std::fprintf(stderr, "qwen3-tts: unsupported frontend %s version %u\n", hparams.frontend_provider.c_str(),
                     hparams.frontend_contract_version);
        return false;
    }
    return true;
}

}  // namespace

synth_status_t read_hparams(const gguf_context * gguf, HParams & hparams) {
    if (gguf == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    hparams                 = HParams{};
    const GgufMetadata meta = metadata(gguf);
    const bool ok = read_identity(meta, hparams) && read_quantization(meta, hparams) &&
                    read_capabilities(meta, hparams) && read_talker(meta, hparams) &&
                    read_code_predictor(meta, hparams) && read_codec(meta, hparams) && read_tokens(meta, hparams) &&
                    read_voices(meta, hparams) && read_languages(meta, hparams) && read_frontend(meta, hparams);
    return ok ? SYNTH_OK : SYNTH_ERR_GGUF;
}

bool find_preset_voice(const HParams & hparams, const std::string & id, PresetVoice & voice) {
    for (const PresetVoice & candidate : hparams.preset_voices) {
        if (candidate.id == id) {
            voice = candidate;
            return true;
        }
    }
    return false;
}

bool resolve_language_token(const HParams &     hparams,
                            const std::string & requested,
                            const PresetVoice & voice,
                            uint32_t &          token_id,
                            std::string &       resolved_name) {
    // A dialect speaker overrides the requested language. That is the
    // reference's behaviour, and the two dialects are not otherwise reachable:
    // they exist in the codec language table but not the public language list.
    const std::string & wanted = voice.dialect_override.empty() ? requested : voice.dialect_override;
    for (size_t index = 0; index < hparams.language_names.size(); ++index) {
        if (hparams.language_names[index] == wanted) {
            token_id      = hparams.language_token_ids[index];
            resolved_name = wanted;
            return true;
        }
    }
    return false;
}

}  // namespace synth::qwen3tts
