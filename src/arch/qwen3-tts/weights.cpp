#include "weights.h"

#include "gguf-metadata.h"
#include "gguf.h"

#include <cstdio>

namespace synth::qwen3tts {

namespace {

constexpr uint32_t kFormatVersion        = 1;
constexpr uint32_t kArchitectureVersion  = 1;
constexpr uint32_t kProfileSchemaVersion = 1;

// A Profile Compatibility ID is a sha256 over the family compatibility
// manifest, so it is exactly 32 bytes written as hex.
constexpr size_t kCompatibilityIdChars = 64;

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
    // Both head counts, not only the shapes: the divisibility test below divides
    // by key_value_head_count, and a package declaring zero would raise SIGFPE
    // during load rather than be refused. The talker guards both operands and
    // this half did not.
    if (cp.layer_count == 0 || cp.hidden_size == 0 || cp.vocab_size == 0 || cp.code_group_count == 0 ||
        cp.attention_head_count == 0 || cp.key_value_head_count == 0) {
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

// Every special token id has to index the vocabulary it is used against.
//
// read_voices already refuses a speaker token outside the codec vocabulary; none
// of these thirteen got the same treatment, so a package naming an id past the
// end of either vocabulary loaded cleanly and surfaced as an out-of-range
// embedding row during synthesis -- a wrong answer or a crash, in a place that
// says nothing about the cause.
//
// Which vocabulary each belongs to is read off the package rather than assumed:
// the seven codec ids sit at 2148-2157 against a codec vocabulary of 3072, and
// the six prompt ids at 77091-151673 against a text vocabulary of 151936.
bool check_token_ranges(const HParams & hparams) {
    const SpecialTokens & t = hparams.tokens;

    struct Bound {
        const char * name;
        uint32_t     value;
        uint32_t     limit;
    };

    const Bound bounds[] = {
        { "codec_bos_id",       t.codec_bos,       hparams.talker.codec_vocab_size },
        { "codec_eos_token_id", t.codec_eos,       hparams.talker.codec_vocab_size },
        { "codec_pad_id",       t.codec_pad,       hparams.talker.codec_vocab_size },
        { "codec_think_id",     t.codec_think,     hparams.talker.codec_vocab_size },
        { "codec_nothink_id",   t.codec_nothink,   hparams.talker.codec_vocab_size },
        { "codec_think_bos_id", t.codec_think_bos, hparams.talker.codec_vocab_size },
        { "codec_think_eos_id", t.codec_think_eos, hparams.talker.codec_vocab_size },
        { "tts_bos_token_id",   t.tts_bos,         hparams.talker.text_vocab_size  },
        { "tts_eos_token_id",   t.tts_eos,         hparams.talker.text_vocab_size  },
        { "tts_pad_token_id",   t.tts_pad,         hparams.talker.text_vocab_size  },
        { "im_start_token_id",  t.im_start,        hparams.talker.text_vocab_size  },
        { "im_end_token_id",    t.im_end,          hparams.talker.text_vocab_size  },
        { "assistant_token_id", t.assistant,       hparams.talker.text_vocab_size  },
    };
    for (const Bound & bound : bounds) {
        if (bound.limit == 0 || bound.value >= bound.limit) {
            std::fprintf(stderr, "qwen3-tts: %s is %u, outside a vocabulary of %u\n", bound.name, bound.value,
                         bound.limit);
            return false;
        }
    }
    return true;
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
           meta.u32("synthesize.qwen3-tts.token.codec_think_eos_id", t.codec_think_eos) && check_token_ranges(hparams);
}

bool read_voices(const GgufMetadata & meta, HParams & hparams) {
    std::string mode;
    uint32_t    preset_count = 0;
    if (!meta.string("synthesize.voice.mode", mode) ||
        !meta.boolean("synthesize.voice.has_package_default", hparams.has_package_default) ||
        !meta.u32("synthesize.voice.preset_count", preset_count)) {
        return false;
    }
    if (mode == "preset-catalog") {
        hparams.voice_mode = VoiceMode::PresetCatalog;
        if (preset_count == 0) {
            std::fprintf(stderr, "qwen3-tts: preset-catalog mode with no presets\n");
            return false;
        }
    } else if (mode == "profile-sources") {
        // Base variants carry no selectable Voice at all: upstream ships an
        // empty spk_id table. The package says so positively rather than
        // arriving as a catalog that happens to be empty, so a truncated
        // catalog cannot be mistaken for this.
        //
        // `profile-sources` is the value omnivoice already established
        // (src/arch/omnivoice/weights.cpp) and the manifest schema's own enum.
        // The two families differ on exactly one point: omnivoice REQUIRES a
        // package default because auto-voice is its default Voice, while this
        // variant has none, so the check runs the other way.
        hparams.voice_mode = VoiceMode::ProfileSources;
        if (preset_count != 0) {
            std::fprintf(stderr, "qwen3-tts: profile-sources mode with %u presets\n", preset_count);
            return false;
        }
        hparams.preset_voices.clear();
        if (hparams.has_package_default) {
            std::fprintf(stderr, "qwen3-tts: profile-sources mode names a package default, but this family has none\n");
            return false;
        }
        return true;
    } else {
        std::fprintf(stderr, "qwen3-tts: unsupported voice mode %s\n", mode.c_str());
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

// Whether `value` is 32 bytes of lowercase hex. The converter always emits a
// lowercase hexdigest and the Voice Profile module will byte-compare against
// it, so an uppercase value is refused rather than repaired. Mirrors
// src/arch/omnivoice/weights.cpp's identically-named helper.
bool is_sha256_hex(const std::string & value) {
    if (value.size() != kCompatibilityIdChars) {
        return false;
    }
    for (const char character : value) {
        const bool digit = character >= '0' && character <= '9';
        const bool lower = character >= 'a' && character <= 'f';
        if (!digit && !lower) {
            return false;
        }
    }
    return true;
}

// Nothing consumes a Voice Profile before Plan 2, but a package whose contract
// is wrong cannot be discovered then without re-cutting it, so it is refused
// now. Shaped after omnivoice's read_profile_contract (weights.cpp:524) for
// the schema/version/compatibility-id fields and the reference-bound checks,
// but not identical to it since Stage 3: omnivoice's one variant always
// carries reference audio, so it reads the synthesize.reference.* block
// unconditionally, while this family's Description Text source carries none
// of those keys and accepts either of two schema names depending on what the
// package declared (read_profile_sources, defined further down alongside its
// caller, read_profile_and_speaker_encoder).
bool read_profile_contract(const GgufMetadata & meta, HParams & hparams) {
    ProfileContract & profile = hparams.profile;
    if (!meta.string("synthesize.profile.schema", profile.schema) ||
        !meta.u32("synthesize.profile.schema_version", profile.schema_version) ||
        !meta.string("synthesize.profile.compatibility_id", profile.compatibility_id_hex)) {
        return false;
    }
    const char * expected_schema = (hparams.profile_sources & SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO) != 0 ?
                                       "qwen3-tts-voice-clone" :
                                       "qwen3-tts-voice-design";
    if (profile.schema != expected_schema || profile.schema_version != kProfileSchemaVersion) {
        std::fprintf(stderr, "qwen3-tts: unsupported profile schema %s version %u\n", profile.schema.c_str(),
                     profile.schema_version);
        return false;
    }
    if (!is_sha256_hex(profile.compatibility_id_hex)) {
        std::fprintf(stderr, "qwen3-tts: profile compatibility id %s is not 32 bytes of lowercase hex\n",
                     profile.compatibility_id_hex.c_str());
        return false;
    }
    // Everything below describes REFERENCE AUDIO specifically: the clip limits
    // and the encoder's own front-end rate. A source set with no
    // reference-audio bit carries none of the synthesize.reference.* keys at
    // all -- scripts/convert-qwen3-tts.py's add_metadata only emits that block
    // when the variant carries a speaker encoder -- so reading them
    // unconditionally would refuse every Description Text package on keys it
    // never had reason to write. Gated on has_speaker_encoder, which
    // read_profile_and_speaker_encoder has already proven equals the declared
    // reference-audio bit by the time this function runs.
    if (!hparams.has_speaker_encoder) {
        return true;
    }
    if (!meta.u32("synthesize.reference.target_sample_rate", profile.reference_sample_rate) ||
        !meta.u32("synthesize.reference.target_channels", profile.reference_channels) ||
        !meta.u64("synthesize.reference.min_frames_per_clip", profile.min_frames_per_clip) ||
        !meta.u64("synthesize.reference.max_frames_per_clip", profile.max_frames_per_clip) ||
        !meta.u64("synthesize.reference.max_total_frames", profile.max_total_frames) ||
        !meta.u64("synthesize.reference.max_reference_count", profile.max_reference_count)) {
        return false;
    }
    // Reference audio is resampled to mono at this rate before it reaches the
    // speaker encoder.
    if (profile.reference_sample_rate == 0 || profile.reference_channels != 1) {
        std::fprintf(stderr, "qwen3-tts: reference audio is declared as %u channels at %u Hz\n",
                     profile.reference_channels, profile.reference_sample_rate);
        return false;
    }
    // ...and "before it reaches the speaker encoder" is the whole point: this
    // rate is what create_qwen3_tts_profile_from_reference resamples the
    // caller's clip to (src/voice-profile.cpp, via
    // capabilities.reference_target_sample_rate), while the mel filterbank
    // that then consumes the clip derives its FFT bin frequencies from the
    // ENCODER's own rate (src/arch/qwen3-tts/mel.cpp). Two independently
    // declared package fields describing one physical signal, so they are
    // tied here the way enc_dim is tied to the talker's hidden size in
    // read_speaker_encoder below -- and for a sharper reason: a package
    // declaring 16000 here against a 24000 Hz encoder changes no shape
    // anywhere (the mel's geometry is n_fft/hop_length, the x-vector's width
    // is enc_dim, and the frame bounds are self-consistent against whichever
    // rate they were computed for), so nothing downstream can notice. It
    // would return SYNTH_OK and a frequency-scaled x-vector -- a silent
    // mis-clone. read_speaker_encoder runs before this function so the
    // comparison has something to make; see read_profile_and_speaker_encoder.
    // Unreachable when this package carries no speaker encoder: the early
    // return above already left the function for that case.
    if (profile.reference_sample_rate != hparams.speaker_encoder.sample_rate) {
        std::fprintf(stderr, "qwen3-tts: reference audio is resampled to %u Hz but the speaker encoder runs at %u Hz\n",
                     profile.reference_sample_rate, hparams.speaker_encoder.sample_rate);
        return false;
    }
    // A reference bound of zero would let a Profile be prepared from no audio.
    if (profile.min_frames_per_clip == 0 || profile.min_frames_per_clip > profile.max_frames_per_clip) {
        std::fprintf(stderr, "qwen3-tts: a clip is between %llu and %llu frames, which admits nothing\n",
                     static_cast<unsigned long long>(profile.min_frames_per_clip),
                     static_cast<unsigned long long>(profile.max_frames_per_clip));
        return false;
    }
    if (profile.max_total_frames < profile.max_frames_per_clip) {
        std::fprintf(stderr, "qwen3-tts: a budget of %llu frames admits no %llu-frame clip\n",
                     static_cast<unsigned long long>(profile.max_total_frames),
                     static_cast<unsigned long long>(profile.max_frames_per_clip));
        return false;
    }
    // Exactly one, not merely "at least one". This value is published
    // verbatim as `synth_voice_profile_capabilities_t::max_reference_count`,
    // so it is a promise to the caller about how many Reference Audio clips
    // a Voice Profile may be built from -- and this runtime reads
    // `references[0]` and nothing else (src/voice-profile.cpp's
    // create_qwen3_tts_profile_from_reference, whose own `reference_count !=
    // 1` refusal names that limitation in the code). A package declaring a
    // higher ceiling would advertise a capability every such call is then
    // refused for. Accepting only what the code implements keeps the two
    // sides one statement, and the refusal happens where a wrong package can
    // still be re-cut rather than at the caller's first attempt.
    //
    // It also removes the arithmetic hazard underneath: this field is
    // uint64_t here and in the public struct, and every published package
    // declares 1 (scripts/convert-qwen3-tts.py writes the manifest's own
    // `reference.max_reference_count`; the Base package's Golden Manifest
    // pins 1), so no package that exists is narrowed by anything on the way
    // out.
    if (profile.max_reference_count != 1) {
        std::fprintf(stderr,
                     "qwen3-tts: this runtime builds a Voice Profile from exactly one Reference Audio clip, but the "
                     "package declares %llu\n",
                     static_cast<unsigned long long>(profile.max_reference_count));
        return false;
    }
    return true;
}

// The ECAPA-TDNN speaker encoder's mel front end and output width. enc_dim
// feeds the talker's prompt slot directly -- there is no projection between
// the x-vector and the speaker-token embedding it substitutes for -- so a
// package whose encoder is a different width would build a prompt of the
// wrong shape.
bool read_speaker_encoder(const GgufMetadata & meta, HParams & hparams) {
    SpeakerEncoderParams & encoder = hparams.speaker_encoder;
    const std::string      prefix  = "synthesize.qwen3-tts.speaker_encoder.";
    if (!meta.u32(prefix + "enc_dim", encoder.enc_dim) || !meta.u32(prefix + "sample_rate", encoder.sample_rate) ||
        !meta.u32(prefix + "mel_bins", encoder.mel_bins) || !meta.u32(prefix + "n_fft", encoder.n_fft) ||
        !meta.u32(prefix + "hop_length", encoder.hop_length) || !meta.u32(prefix + "win_length", encoder.win_length) ||
        !meta.f32(prefix + "fmin", encoder.fmin) || !meta.f32(prefix + "fmax", encoder.fmax)) {
        return false;
    }
    // Only the three widths no later rule can catch. `enc_dim`, `sample_rate`
    // and `win_length` were checked here too until the branch's final review
    // showed the mutation is invisible: each is caught below with the same
    // SYNTH_ERR_GGUF and a more specific message -- enc_dim by the
    // hidden-size equality (read_talker already refuses a zero hidden size),
    // sample_rate by the codec-rate equality (read_codec already refuses a
    // codec rate that cannot produce its declared frame rate), and
    // win_length by `hop_length >= win_length`, which any hop satisfies
    // against zero. A rule nobody can see work is not a rule.
    if (encoder.mel_bins == 0 || encoder.n_fft == 0 || encoder.hop_length == 0) {
        std::fprintf(stderr, "qwen3-tts: speaker encoder mel front end has a zero mel_bins/n_fft/hop_length\n");
        return false;
    }
    // The mel front end's FFT is an iterative radix-2 Cooley-Tukey
    // (src/arch/qwen3-tts/mel.cpp), which only accepts a power of two. A
    // load-time contract rather than a runtime surprise mid-enrollment.
    if ((encoder.n_fft & (encoder.n_fft - 1)) != 0) {
        std::fprintf(stderr, "qwen3-tts: speaker encoder n_fft %u is not a power of two\n", encoder.n_fft);
        return false;
    }
    // The other structural constraint the radix-2 transform imposes: the
    // zero-padded-centred window rule (mel.cpp) has no meaning for a window
    // wider than the transform itself. compute_log_mel's own runtime guard
    // for this is the belt to this braces -- the same relationship the
    // power-of-two check above has with compute_log_mel's is_power_of_two
    // check. Without this, hop_length < win_length (checked further down)
    // does not catch it: a win_length past n_fft can still be comfortably
    // past a small hop_length.
    if (encoder.win_length > encoder.n_fft) {
        std::fprintf(stderr, "qwen3-tts: speaker encoder win_length %u exceeds n_fft %u\n", encoder.win_length,
                     encoder.n_fft);
        return false;
    }
    if (encoder.enc_dim != hparams.talker.hidden_size) {
        std::fprintf(stderr, "qwen3-tts: the speaker encoder emits width %u but the talker's hidden size is %u\n",
                     encoder.enc_dim, hparams.talker.hidden_size);
        return false;
    }
    // The reference clip is resampled to the codec's own rate before the mel
    // front end runs on it, so the two must agree.
    if (encoder.sample_rate != hparams.codec.sample_rate) {
        std::fprintf(stderr, "qwen3-tts: the speaker encoder runs at %u Hz but the codec runs at %u Hz\n",
                     encoder.sample_rate, hparams.codec.sample_rate);
        return false;
    }
    if (encoder.hop_length >= encoder.win_length) {
        std::fprintf(stderr, "qwen3-tts: speaker encoder hop_length %u does not fall inside win_length %u\n",
                     encoder.hop_length, encoder.win_length);
        return false;
    }
    if (!(encoder.fmax > encoder.fmin) || encoder.fmax > static_cast<float>(encoder.sample_rate) / 2.0f) {
        std::fprintf(stderr, "qwen3-tts: speaker encoder mel band [%f, %f] is invalid for a %u Hz front end\n",
                     static_cast<double>(encoder.fmin), static_cast<double>(encoder.fmax), encoder.sample_rate);
        return false;
    }
    return true;
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

// The declared source names, mapped to the public bits. An unknown name is a
// refusal rather than a skip: a package written by a newer converter must not
// load with a capability quietly narrower than it claims.
bool read_profile_sources(const GgufMetadata & meta, HParams & hparams) {
    std::vector<std::string> names;
    if (!meta.string_array("synthesize.voice.profile_sources", names) || names.empty()) {
        std::fprintf(stderr, "qwen3-tts: profile-sources mode declares no profile sources\n");
        return false;
    }
    hparams.profile_sources = 0;
    for (const std::string & name : names) {
        if (name == "reference-audio") {
            hparams.profile_sources |= SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO;
        } else if (name == "description-text") {
            hparams.profile_sources |= SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT;
        } else {
            std::fprintf(stderr, "qwen3-tts: unknown profile source %s\n", name.c_str());
            return false;
        }
    }
    return true;
}

// A CustomVoice package carries none of the profile/speaker-encoder/
// profile_sources keys at all -- it has a Preset Catalog and no Voice Profile
// contract -- and must keep loading exactly as it does today. This function
// is gated on the declared Voice Mode (populated by read_voices, which always
// runs first in the chain below) rather than on raw key presence: gating on
// presence alone would let a package that claims profile-sources but never
// actually got any of its synthesize.voice.profile_sources /
// synthesize.profile.* / synthesize.qwen3-tts.speaker_encoder.* keys written
// load clean with a zeroed ProfileContract, no declared sources, and no
// encoder -- a model that says every request must carry a Voice Profile, with
// nothing to validate one against. That is exactly the truncation
// read_voices's own profile-sources comment above claims to rule out; gating
// on the mode closes the gap because a truncated package still fails, now at
// the first step: read_profile_sources refuses a missing or empty
// declaration before either block is even considered.
//
// The declaration and the package's actual shape are cross-checked rather
// than either one alone deciding what to read: Reference Audio needs the
// encoder and the reference limits; Description Text needs neither and must
// not carry them, because a package that ships an encoder while claiming it
// cannot clone disagrees with itself and one of the two statements is wrong.
//
// The speaker encoder is read FIRST when it is read at all, though the
// contract is the more externally visible half: read_profile_contract ties
// the declared reference target rate to the encoder's own rate, and cannot do
// that against a SpeakerEncoderParams nobody has filled in yet. Same ordering
// rule the rest of read_hparams follows -- read_speaker_encoder itself only
// checks enc_dim against the talker and its rate against the codec because
// read_talker and read_codec already ran. Nothing in read_speaker_encoder
// reads hparams.profile, so the pair has exactly one valid order.
bool read_profile_and_speaker_encoder(const GgufMetadata & meta, HParams & hparams) {
    if (!read_profile_sources(meta, hparams)) {
        return false;
    }
    const bool wants_reference = (hparams.profile_sources & SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO) != 0;
    const bool carries_encoder = meta.has("synthesize.qwen3-tts.speaker_encoder.enc_dim");
    if (wants_reference != carries_encoder) {
        std::fprintf(stderr, "qwen3-tts: package declares reference-audio=%d but carries a speaker encoder=%d\n",
                     static_cast<int>(wants_reference), static_cast<int>(carries_encoder));
        return false;
    }
    if (wants_reference) {
        hparams.has_speaker_encoder = true;
        return read_speaker_encoder(meta, hparams) && read_profile_contract(meta, hparams);
    }
    return read_profile_contract(meta, hparams);
}

}  // namespace

synth_status_t read_hparams(const gguf_context * gguf, HParams & hparams) {
    if (gguf == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    hparams                 = HParams{};
    const GgufMetadata meta = metadata(gguf);
    const bool         ok =
        read_identity(meta, hparams) && read_quantization(meta, hparams) && read_capabilities(meta, hparams) &&
        read_talker(meta, hparams) && read_code_predictor(meta, hparams) && read_codec(meta, hparams) &&
        read_tokens(meta, hparams) && read_voices(meta, hparams) &&
        (hparams.voice_mode != VoiceMode::ProfileSources || read_profile_and_speaker_encoder(meta, hparams)) &&
        read_languages(meta, hparams) && read_frontend(meta, hparams);
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

void fill_voice_profile_capability(const HParams & hparams, VoiceProfileInfo & info) {
    info = VoiceProfileInfo{};
    // The gate is the voice mode. `has_speaker_encoder` alone is not enough:
    // an encoder flag says the package carries weights, not that the runtime
    // can prepare anything, and a PresetCatalog package that somehow carried
    // one must still advertise nothing (see
    // qwen3_tts_voice_required_test.cpp's adversarial case). And
    // `has_preset_voice_catalog` is the INVERSE of this condition -- it is
    // true only for CustomVoice -- which is the trap the Plan 1 carryover
    // corrected before Plan 2 was written.
    if (hparams.voice_mode != VoiceMode::ProfileSources) {
        return;
    }
    // The bits follow what the package DECLARED and read_profile_sources
    // already checked against what it carries. Serialized Profile accompanies
    // either source, because every v1 Profile this family can create can be
    // serialized (docs/c-interface.md).
    info.source_flags = hparams.profile_sources | SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE;

    if ((hparams.profile_sources & SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO) == 0) {
        // No recording is taken on this path, so the six reference limits and
        // the transcript requirements describe nothing. Left at their zeroed
        // defaults: a zero here means "not applicable", and copying Base's
        // numbers would state a contract this package cannot honour.
        info.schema         = hparams.profile.schema;
        info.schema_version = hparams.profile.schema_version;
        decode_profile_compatibility_id(hparams.profile.compatibility_id_hex, info.compatibility_id);
        return;
    }
    // OPTIONAL, both of them, since Plan 3 landed the transcript-assisted
    // (ICL) mode next to the x-vector one. BOTH MODES NOW EXIST, and D4 fixes
    // the clone mode at preparation, so the transcript's PRESENCE is what
    // selects between them: absent selects x-vector, present selects ICL
    // (src/voice-profile.cpp's create_from_reference dispatch, the one site in
    // the chain holding a live Model and therefore able to reach the BPE
    // tables). Neither field is REQUIRED -- a caller supplying neither still
    // gets a working x-vector Profile -- and neither is UNSUPPORTED any
    // longer, which it was for the whole of Plan 2 precisely because
    // advertising a mode with no implementation behind it would have invited a
    // caller to pass a transcript and receive the weaker clone it did not ask
    // for. `reference_language` follows the transcript, as it did when it was
    // refused: optional, and validated against this package's declared
    // languages when present.
    //
    // Nothing else here moves. This is a statement about the RUNTIME, not
    // about the package: `source_flags`, the six reference limits, the schema
    // identity and the compatibility id are all read from the same declared
    // ProfileContract they were before, and Description Text and Random Seed
    // stay unadvertised.
    info.reference_transcript = SYNTH_REQUIREMENT_OPTIONAL;
    info.reference_language   = SYNTH_REQUIREMENT_OPTIONAL;

    info.reference_target_sample_rate = hparams.profile.reference_sample_rate;
    info.reference_target_channels    = hparams.profile.reference_channels;
    info.min_frames_per_clip          = hparams.profile.min_frames_per_clip;
    info.max_frames_per_clip          = hparams.profile.max_frames_per_clip;
    info.max_total_frames             = hparams.profile.max_total_frames;
    info.max_reference_count          = hparams.profile.max_reference_count;
    info.schema                       = hparams.profile.schema;
    info.schema_version               = hparams.profile.schema_version;
    decode_profile_compatibility_id(hparams.profile.compatibility_id_hex, info.compatibility_id);
}

}  // namespace synth::qwen3tts
