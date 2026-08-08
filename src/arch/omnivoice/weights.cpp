#include "arch/omnivoice/weights.h"

#include "gguf-metadata.h"
#include "gguf.h"

#include <cstddef>
#include <cstdio>

namespace synth::omnivoice {

namespace {

constexpr uint32_t kFormatVersion              = 1;
constexpr uint32_t kArchitectureVersion        = 1;
constexpr uint32_t kQuantizationProfileVersion = 1;
constexpr uint32_t kProfileSchemaVersion       = 1;
constexpr uint32_t kFrontendContractVersion    = 1;

// A Profile Compatibility ID is a sha256 over the family compatibility
// manifest, so it is exactly 32 bytes written as hex.
constexpr size_t kCompatibilityIdChars = 64;

// Ceiling on any derived tensor dimension. Far past every real model in this
// family's class, and small enough that any product of two guarded values
// stays inside int64_t everywhere the catalog multiplies.
constexpr uint64_t kMaxDimensionProduct = uint64_t(1) << 24;

GgufMetadata metadata(const gguf_context * gguf) {
    return GgufMetadata(gguf, "omnivoice");
}

bool read_identity(const GgufMetadata & meta, HParams & hparams) {
    uint32_t format_version       = 0;
    uint32_t architecture_version = 0;
    if (!meta.require_string("general.architecture", "omnivoice") ||
        !meta.require_string("synthesize.model_family", "omnivoice") ||
        !meta.string("synthesize.model_variant", hparams.model_variant) ||
        !meta.u32("synthesize.format_version", format_version) ||
        !meta.u32("synthesize.omnivoice.architecture_version", architecture_version)) {
        return false;
    }
    if (format_version != kFormatVersion || architecture_version != kArchitectureVersion) {
        std::fprintf(stderr, "omnivoice: unsupported format %u or architecture %u\n", format_version,
                     architecture_version);
        return false;
    }
    hparams.architecture_version = architecture_version;
    if (hparams.model_variant.empty()) {
        std::fprintf(stderr, "omnivoice: the package names no model variant\n");
        return false;
    }
    return true;
}

bool read_quantization(const GgufMetadata & meta, HParams & hparams) {
    std::string profile;
    if (!meta.string("synthesize.quantization.profile", profile) ||
        !meta.u32("synthesize.quantization.profile_version", hparams.quantization_profile_version)) {
        return false;
    }
    // The checkpoint stores F32 in both halves, so F32 is also the source
    // profile. Q8_MIXED and F16 are Plan 4's codec-half Quantization
    // Profiles; Q8_GEN and Q4_K_GEN quantize the generator half instead and
    // leave the codec byte-identical to F32. See quantization.h's QuantRole and
    // ModelHalf for which tensors each actually touches. These strings are
    // the same ones tools/synthesize-quantize/policy.cpp's profile table
    // carries -- that table is where a new name is introduced, and this is
    // where a package carrying it becomes loadable.
    if (profile == "F32") {
        hparams.quantization_profile = QuantizationProfile::F32;
    } else if (profile == "Q8_MIXED") {
        hparams.quantization_profile = QuantizationProfile::Q8Mixed;
    } else if (profile == "F16") {
        hparams.quantization_profile = QuantizationProfile::F16;
    } else if (profile == "Q8_GEN") {
        hparams.quantization_profile = QuantizationProfile::Q8Gen;
    } else if (profile == "Q4_K_GEN") {
        hparams.quantization_profile = QuantizationProfile::Q4KGen;
    } else {
        std::fprintf(stderr, "omnivoice: unsupported quantization profile %s\n", profile.c_str());
        return false;
    }
    if (hparams.quantization_profile_version != kQuantizationProfileVersion) {
        std::fprintf(stderr, "omnivoice: unsupported quantization profile version %u\n",
                     hparams.quantization_profile_version);
        return false;
    }
    return true;
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
        std::fprintf(stderr, "omnivoice: unsupported sample format %s\n", sample_format.c_str());
        return false;
    }
    if (hparams.output_channel_count != 1 || hparams.output_sample_rate == 0) {
        std::fprintf(stderr, "omnivoice: package declares %u channels at %u Hz\n", hparams.output_channel_count,
                     hparams.output_sample_rate);
        return false;
    }
    if (hparams.max_input_tokens == 0 || hparams.max_output_frames == 0) {
        std::fprintf(stderr, "omnivoice: package declares a zero input or output limit\n");
        return false;
    }
    // Exact match, not merely "the TEXT bit is present": this family declares
    // text input ONLY (src/synthesize.cpp's public dispatch comment relies on
    // input_kind being "guaranteed SYNTH_INPUT_TEXT_UTF8" for every omnivoice
    // request), so a package additionally declaring, say,
    // SYNTH_INPUT_SUPPORT_TOKEN_IDS would let a request whose input_kind is
    // SYNTH_INPUT_TOKEN_IDS reach that dispatch and have its int32 token
    // array read as raw UTF-8 text bytes (reviewer FINDING 4). Refused here,
    // at load, rather than trusted at every synthesis call site downstream.
    if (hparams.input_flags != SYNTH_INPUT_SUPPORT_TEXT_UTF8) {
        std::fprintf(stderr, "omnivoice: package declares input_flags %u; this family accepts text input only\n",
                     hparams.input_flags);
        return false;
    }
    // The speaking rate divides the duration estimate, and the neutral rate has
    // to sit inside the declared range or the default request would already be
    // out of bounds against the package's own claim.
    if (!(hparams.min_speaking_rate > 0.0f) || hparams.min_speaking_rate > 1.0f || hparams.max_speaking_rate < 1.0f) {
        std::fprintf(stderr, "omnivoice: unusable speaking rate range [%f, %f]\n",
                     static_cast<double>(hparams.min_speaking_rate), static_cast<double>(hparams.max_speaking_rate));
        return false;
    }
    // The flags are what the public interface reports. An adjustable range the
    // capability does not claim would be a control no caller may reach.
    const bool adjustable = hparams.min_speaking_rate != hparams.max_speaking_rate;
    if (adjustable && (hparams.capability_flags & SYNTH_MODEL_CAPABILITY_SPEAKING_RATE) == 0) {
        std::fprintf(stderr, "omnivoice: speaking rate spans [%f, %f] without the capability that admits it\n",
                     static_cast<double>(hparams.min_speaking_rate), static_cast<double>(hparams.max_speaking_rate));
        return false;
    }
    // Every public synthesis of this family samples: the checkpoint's own
    // position_temperature is 5.0, and only the goldens turn it off. A package
    // that did not declare it would promise reproducibility it cannot keep.
    if ((hparams.capability_flags & SYNTH_MODEL_CAPABILITY_STOCHASTIC) == 0) {
        std::fprintf(stderr, "omnivoice: this family samples but the package claims determinism\n");
        return false;
    }
    return true;
}

// Auto-voice is the package default and there is no preset catalog, so all
// three of these are constants rather than variables. They are read anyway
// because a package that disagrees would put the Voice seam in a state no code
// path serves.
bool read_voice(const GgufMetadata & meta) {
    std::string mode;
    bool        has_package_default = false;
    uint32_t    preset_count        = 0;
    if (!meta.string("synthesize.voice.mode", mode) ||
        !meta.boolean("synthesize.voice.has_package_default", has_package_default) ||
        !meta.u32("synthesize.voice.preset_count", preset_count)) {
        return false;
    }
    if (mode != "profile-sources" || preset_count != 0) {
        std::fprintf(stderr, "omnivoice: unsupported voice mode %s with %u presets\n", mode.c_str(), preset_count);
        return false;
    }
    if (!has_package_default) {
        std::fprintf(stderr, "omnivoice: auto-voice is this family's package default\n");
        return false;
    }
    return true;
}

bool read_generation(const GgufMetadata & meta, HParams & hparams) {
    GenerationDefaults & generation = hparams.generation;
    const std::string    prefix     = "synthesize.omnivoice.generation.";
    if (!meta.u32(prefix + "num_step", generation.num_step) ||
        !meta.f32(prefix + "guidance_scale", generation.guidance_scale) ||
        !meta.f32(prefix + "t_shift", generation.t_shift) ||
        !meta.f32(prefix + "layer_penalty_factor", generation.layer_penalty_factor) ||
        !meta.f32(prefix + "position_temperature", generation.position_temperature) ||
        !meta.f32(prefix + "class_temperature", generation.class_temperature)) {
        std::fprintf(stderr, "omnivoice: package declares no generation defaults; re-cut it\n");
        return false;
    }
    if (generation.num_step == 0) {
        std::fprintf(stderr, "omnivoice: zero mask-predict steps decode nothing\n");
        return false;
    }
    // t_shift divides the schedule; the other three only scale, so zero is a
    // meaningful value for them and a negative one is not.
    if (!(generation.t_shift > 0.0f)) {
        std::fprintf(stderr, "omnivoice: t_shift %f does not shift a schedule\n",
                     static_cast<double>(generation.t_shift));
        return false;
    }
    if (generation.guidance_scale < 0.0f || generation.layer_penalty_factor < 0.0f ||
        generation.position_temperature < 0.0f || generation.class_temperature < 0.0f) {
        std::fprintf(stderr, "omnivoice: package declares a negative guidance, penalty or temperature\n");
        return false;
    }
    return true;
}

bool read_generator(const GgufMetadata & meta, HParams & hparams) {
    GeneratorParams & generator = hparams.generator;
    const std::string prefix    = "synthesize.omnivoice.generator.";
    std::string       attention;
    if (!meta.u32(prefix + "layer_count", generator.layer_count) ||
        !meta.u32(prefix + "hidden_size", generator.hidden_size) ||
        !meta.u32(prefix + "attention_head_count", generator.attention_head_count) ||
        !meta.u32(prefix + "key_value_head_count", generator.key_value_head_count) ||
        !meta.u32(prefix + "head_dim", generator.head_dim) ||
        !meta.u32(prefix + "intermediate_size", generator.intermediate_size) ||
        !meta.u32(prefix + "text_vocab_size", generator.text_vocab_size) ||
        !meta.f32(prefix + "rms_norm_eps", generator.rms_norm_eps) ||
        !meta.f32(prefix + "rope_theta", generator.rope_theta) || !meta.string(prefix + "attention", attention)) {
        return false;
    }
    // The head counts are guarded here rather than by the divisibility test
    // below, which would otherwise divide by zero during load.
    if (generator.layer_count == 0 || generator.hidden_size == 0 || generator.attention_head_count == 0 ||
        generator.key_value_head_count == 0 || generator.head_dim == 0 || generator.intermediate_size == 0 ||
        generator.text_vocab_size == 0) {
        std::fprintf(stderr, "omnivoice: generator geometry contains a zero dimension\n");
        return false;
    }
    // Grouped-query attention: every key/value head serves a whole number of
    // query heads, and a package violating that would silently mis-broadcast.
    if (generator.attention_head_count % generator.key_value_head_count != 0) {
        std::fprintf(stderr, "omnivoice: %u query heads do not divide into %u key/value heads\n",
                     generator.attention_head_count, generator.key_value_head_count);
        return false;
    }
    // Products of these fields feed int64 shape arithmetic in the catalog;
    // guarding the products here means no consumer has to prove overflow
    // freedom case by case. Checked one at a time and named in the message --
    // a single combined condition can only ever report the first field of the
    // five, which is a wrong diagnosis whenever a different one is the actual
    // offender.
    const uint64_t attention_inner = uint64_t(generator.attention_head_count) * generator.head_dim;
    const uint64_t kv_inner        = uint64_t(generator.key_value_head_count) * generator.head_dim;

    const struct {
        const char * field;
        uint64_t     value;
    } guarded_products[] = {
        { "attention_head_count * head_dim", attention_inner                       },
        { "key_value_head_count * head_dim", kv_inner                              },
        { "hidden_size",                     uint64_t(generator.hidden_size)       },
        { "intermediate_size",               uint64_t(generator.intermediate_size) },
        { "text_vocab_size",                 uint64_t(generator.text_vocab_size)   },
    };

    for (const auto & product : guarded_products) {
        if (product.value > kMaxDimensionProduct) {
            std::fprintf(stderr, "omnivoice: generator geometry is implausibly large (%s is %llu)\n", product.field,
                         static_cast<unsigned long long>(product.value));
            return false;
        }
    }
    if (generator.rms_norm_eps <= 0.0f || generator.rope_theta <= 0.0f) {
        std::fprintf(stderr, "omnivoice: non-positive rms_norm_eps or rope_theta\n");
        return false;
    }
    // Mask-predict attends the whole canvas in both directions at every step.
    // A package asking for a causal mask describes a different model, and
    // building one would be wrong at every position rather than at the edges.
    if (attention != "bidirectional") {
        std::fprintf(stderr, "omnivoice: unsupported attention %s; this family is bidirectional\n", attention.c_str());
        return false;
    }
    return true;
}

bool read_audio_canvas(const GgufMetadata & meta, HParams & hparams) {
    AudioCanvasParams & audio  = hparams.audio;
    const std::string   prefix = "synthesize.omnivoice.audio.";
    if (!meta.u32(prefix + "num_codebooks", audio.num_codebooks) ||
        !meta.u32(prefix + "vocab_size", audio.vocab_size) || !meta.u32(prefix + "mask_id", audio.mask_id)) {
        return false;
    }
    if (audio.num_codebooks == 0 || audio.vocab_size == 0) {
        std::fprintf(stderr, "omnivoice: the canvas declares %u codebooks over a vocabulary of %u\n",
                     audio.num_codebooks, audio.vocab_size);
        return false;
    }
    // The mask occupies the last entry of the canvas vocabulary, and the shared
    // embedding table is read at `codebook * vocab_size`. Anywhere else and
    // masking a position would write a real code instead.
    if (audio.mask_id != audio.vocab_size - 1) {
        std::fprintf(stderr, "omnivoice: mask id %u is not the last entry of a %u-entry canvas vocabulary\n",
                     audio.mask_id, audio.vocab_size);
        return false;
    }
    // The stacked audio tables are [num_codebooks * vocab_size] rows; the
    // catalog multiplies these in int64, so bound the product here.
    if (uint64_t(audio.num_codebooks) * audio.vocab_size > kMaxDimensionProduct) {
        std::fprintf(stderr, "omnivoice: a %u x %u canvas table exceeds any real package\n", audio.num_codebooks,
                     audio.vocab_size);
        return false;
    }
    return true;
}

bool read_codec(const GgufMetadata & meta, HParams & hparams) {
    CodecParams &     codec  = hparams.codec;
    const std::string prefix = "synthesize.omnivoice.codec.";
    if (!meta.u32(prefix + "sample_rate", codec.sample_rate) || !meta.u32(prefix + "hop_length", codec.hop_length) ||
        !meta.f32(prefix + "frame_rate_hz", codec.frame_rate_hz) ||
        !meta.positive_i32_array(prefix + "upsampling_ratios", codec.upsampling_ratios) ||
        !meta.u32(prefix + "decoder_hidden_size", codec.decoder_hidden_size) ||
        !meta.u32(prefix + "encoder_hidden_size", codec.encoder_hidden_size) ||
        !meta.u32(prefix + "hidden_size", codec.hidden_size) ||
        !meta.u32(prefix + "codebook_dim", codec.codebook_dim) ||
        !meta.u32(prefix + "codebook_size", codec.codebook_size) ||
        !meta.u32(prefix + "semantic_sample_rate", codec.semantic_sample_rate)) {
        return false;
    }
    if (codec.hop_length == 0 || codec.frame_rate_hz <= 0.0f || codec.decoder_hidden_size == 0 ||
        codec.encoder_hidden_size == 0 || codec.hidden_size == 0 || codec.codebook_dim == 0 ||
        codec.codebook_size == 0 || codec.semantic_sample_rate == 0) {
        std::fprintf(stderr, "omnivoice: codec geometry contains a zero dimension\n");
        return false;
    }
    if (codec.sample_rate != hparams.output_sample_rate) {
        std::fprintf(stderr, "omnivoice: codec runs at %u Hz but the package declares %u Hz\n", codec.sample_rate,
                     hparams.output_sample_rate);
        return false;
    }
    // One frame is exactly hop_length samples. Deriving the relation rather than
    // trusting three independent numbers is what catches a package whose frame
    // rate and hop disagree, which would drift the audio length per frame.
    const double implied = static_cast<double>(codec.sample_rate) / static_cast<double>(codec.hop_length);
    if (implied < codec.frame_rate_hz - 1e-6 || implied > codec.frame_rate_hz + 1e-6) {
        std::fprintf(stderr, "omnivoice: %u Hz over hop %u is %f frames per second, not %f\n", codec.sample_rate,
                     codec.hop_length, implied, static_cast<double>(codec.frame_rate_hz));
        return false;
    }
    // Every upsampling stage multiplies out to exactly one frame of samples.
    // The decoder's stage widths are derived from this stack, so a package whose
    // ratios and hop disagree would build a vocoder of the wrong length.
    if (codec.upsampling_ratios.empty()) {
        std::fprintf(stderr, "omnivoice: the codec declares no upsampling stages\n");
        return false;
    }
    uint64_t total = 1;
    for (uint32_t ratio : codec.upsampling_ratios) {
        total *= ratio;
        // Each ratio is a positive i32, so one multiply can raise `total` by at
        // most 2^31: once it exceeds the hop it can never come back, and
        // stopping here is also what keeps the product from wrapping uint64_t.
        if (total > codec.hop_length) {
            break;
        }
    }
    if (total != codec.hop_length) {
        std::fprintf(stderr, "omnivoice: the upsampling ratios multiply to %llu, not the hop %u\n",
                     static_cast<unsigned long long>(total), codec.hop_length);
        return false;
    }
    // The canvas holds one entry per codec code plus the mask. A disagreement
    // would have the generator predicting codes the codec cannot look up.
    if (hparams.audio.vocab_size != codec.codebook_size + 1) {
        std::fprintf(stderr, "omnivoice: a %u-entry canvas over %u codes leaves no room for exactly one mask\n",
                     hparams.audio.vocab_size, codec.codebook_size);
        return false;
    }
    return true;
}

bool read_semantic(const GgufMetadata & meta, HParams & hparams) {
    SemanticParams &  semantic = hparams.semantic;
    const std::string prefix   = "synthesize.omnivoice.semantic.";
    if (!meta.u32(prefix + "hidden_size", semantic.hidden_size) ||
        !meta.u32(prefix + "layer_count", semantic.layer_count) ||
        !meta.u32(prefix + "attention_head_count", semantic.attention_head_count) ||
        !meta.u32(prefix + "intermediate_size", semantic.intermediate_size) ||
        !meta.f32(prefix + "layer_norm_eps", semantic.layer_norm_eps) ||
        !meta.positive_i32_array(prefix + "conv_dim", semantic.conv_dim) ||
        !meta.positive_i32_array(prefix + "conv_kernel", semantic.conv_kernel) ||
        !meta.positive_i32_array(prefix + "conv_stride", semantic.conv_stride)) {
        return false;
    }
    if (semantic.hidden_size == 0 || semantic.layer_count == 0 || semantic.attention_head_count == 0 ||
        semantic.intermediate_size == 0) {
        std::fprintf(stderr, "omnivoice: semantic geometry contains a zero dimension\n");
        return false;
    }
    if (semantic.hidden_size % semantic.attention_head_count != 0) {
        std::fprintf(stderr, "omnivoice: a semantic width of %u does not split into %u whole heads\n",
                     semantic.hidden_size, semantic.attention_head_count);
        return false;
    }
    if (semantic.layer_norm_eps <= 0.0f) {
        std::fprintf(stderr, "omnivoice: non-positive semantic layer_norm_eps\n");
        return false;
    }
    // The three arrays describe the same feature-extractor layers one field at
    // a time. A length disagreement would silently truncate the stack.
    if (semantic.conv_dim.empty() || semantic.conv_dim.size() != semantic.conv_kernel.size() ||
        semantic.conv_dim.size() != semantic.conv_stride.size()) {
        std::fprintf(stderr, "omnivoice: the semantic conv arrays are %zu/%zu/%zu long\n", semantic.conv_dim.size(),
                     semantic.conv_kernel.size(), semantic.conv_stride.size());
        return false;
    }
    return true;
}

// Every marker indexes the text embedding table, and the prompt builder
// assembles a request out of these by value. An id past the end of the
// vocabulary is an out-of-range row during synthesis, and two markers sharing
// an id make the slots they mark indistinguishable in the built prompt --
// neither of which says anything about the package when it surfaces.
bool check_token_values(const HParams & hparams) {
    const SpecialTokens & t = hparams.tokens;

    struct Marker {
        const char * name;
        uint32_t     value;
    };

    const Marker markers[] = {
        { "denoise",        t.denoise        },
        { "lang_start",     t.lang_start     },
        { "lang_end",       t.lang_end       },
        { "instruct_start", t.instruct_start },
        { "instruct_end",   t.instruct_end   },
        { "text_start",     t.text_start     },
        { "text_end",       t.text_end       },
        { "eos",            t.eos            },
        { "pad",            t.pad            },
    };
    const size_t count = sizeof(markers) / sizeof(markers[0]);
    for (size_t index = 0; index < count; ++index) {
        if (markers[index].value >= hparams.generator.text_vocab_size) {
            std::fprintf(stderr, "omnivoice: token %s is %u, outside a vocabulary of %u\n", markers[index].name,
                         markers[index].value, hparams.generator.text_vocab_size);
            return false;
        }
        for (size_t other = index + 1; other < count; ++other) {
            if (markers[index].value == markers[other].value) {
                std::fprintf(stderr, "omnivoice: tokens %s and %s share id %u\n", markers[index].name,
                             markers[other].name, markers[index].value);
                return false;
            }
        }
    }
    return true;
}

bool read_tokens(const GgufMetadata & meta, HParams & hparams) {
    SpecialTokens &   t      = hparams.tokens;
    const std::string prefix = "synthesize.omnivoice.token.";
    return meta.u32(prefix + "denoise", t.denoise) && meta.u32(prefix + "lang_start", t.lang_start) &&
           meta.u32(prefix + "lang_end", t.lang_end) && meta.u32(prefix + "instruct_start", t.instruct_start) &&
           meta.u32(prefix + "instruct_end", t.instruct_end) && meta.u32(prefix + "text_start", t.text_start) &&
           meta.u32(prefix + "text_end", t.text_end) && meta.u32(prefix + "eos", t.eos) &&
           meta.u32(prefix + "pad", t.pad) && check_token_values(hparams);
}

bool read_languages(const GgufMetadata & meta, HParams & hparams) {
    if (!meta.string_array("synthesize.omnivoice.languages.tags", hparams.language_tags)) {
        return false;
    }
    if (hparams.language_tags.empty()) {
        std::fprintf(stderr, "omnivoice: the package declares no languages\n");
        return false;
    }
    // The tag is also the literal text the prompt's language slot carries, so an
    // empty one would mark the slot with nothing, and a repeat would publish the
    // same capability twice.
    for (size_t index = 0; index < hparams.language_tags.size(); ++index) {
        if (hparams.language_tags[index].empty()) {
            std::fprintf(stderr, "omnivoice: language %zu carries an empty tag\n", index);
            return false;
        }
        for (size_t other = index + 1; other < hparams.language_tags.size(); ++other) {
            if (hparams.language_tags[index] == hparams.language_tags[other]) {
                std::fprintf(stderr, "omnivoice: language tag %s is declared twice\n",
                             hparams.language_tags[index].c_str());
                return false;
            }
        }
    }
    return true;
}

// Whether `value` is 32 bytes of lowercase hex. The converter always emits a
// lowercase hexdigest and Plan 3 will byte-compare against it, so an uppercase
// value is refused rather than repaired.
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

bool read_profile_contract(const GgufMetadata & meta, HParams & hparams) {
    ProfileContract & profile = hparams.profile;
    if (!meta.string("synthesize.profile.schema", profile.schema) ||
        !meta.u32("synthesize.profile.schema_version", profile.schema_version) ||
        !meta.string("synthesize.profile.compatibility_id", profile.compatibility_id_hex) ||
        !meta.u32("synthesize.reference.target_sample_rate", profile.reference_sample_rate) ||
        !meta.u32("synthesize.reference.target_channels", profile.reference_channels) ||
        !meta.u64("synthesize.reference.min_frames_per_clip", profile.min_frames_per_clip) ||
        !meta.u64("synthesize.reference.max_frames_per_clip", profile.max_frames_per_clip) ||
        !meta.u64("synthesize.reference.max_total_frames", profile.max_total_frames) ||
        !meta.u64("synthesize.reference.max_reference_count", profile.max_reference_count)) {
        return false;
    }
    // Nothing consumes a Serialized Profile before Plan 3, but a package whose
    // contract is wrong cannot be discovered then without re-cutting it, so it
    // is refused now.
    if (profile.schema != "omnivoice-clone-prompt" || profile.schema_version != kProfileSchemaVersion) {
        std::fprintf(stderr, "omnivoice: unsupported profile schema %s version %u\n", profile.schema.c_str(),
                     profile.schema_version);
        return false;
    }
    if (!is_sha256_hex(profile.compatibility_id_hex)) {
        std::fprintf(stderr, "omnivoice: profile compatibility id %s is not 32 bytes of lowercase hex\n",
                     profile.compatibility_id_hex.c_str());
        return false;
    }
    // The encode path takes one mono stream at the codec's own rate; every clip
    // is resampled to these before it reaches the semantic branch.
    if (profile.reference_sample_rate == 0 || profile.reference_channels != 1) {
        std::fprintf(stderr, "omnivoice: reference audio is declared as %u channels at %u Hz\n",
                     profile.reference_channels, profile.reference_sample_rate);
        return false;
    }
    if (profile.min_frames_per_clip == 0 || profile.min_frames_per_clip > profile.max_frames_per_clip) {
        std::fprintf(stderr, "omnivoice: a clip is between %llu and %llu frames, which admits nothing\n",
                     static_cast<unsigned long long>(profile.min_frames_per_clip),
                     static_cast<unsigned long long>(profile.max_frames_per_clip));
        return false;
    }
    if (profile.max_total_frames < profile.max_frames_per_clip || profile.max_reference_count == 0) {
        std::fprintf(stderr, "omnivoice: a budget of %llu frames over %llu references admits no clip\n",
                     static_cast<unsigned long long>(profile.max_total_frames),
                     static_cast<unsigned long long>(profile.max_reference_count));
        return false;
    }
    return true;
}

bool read_frontend(const GgufMetadata & meta, HParams & hparams) {
    if (!meta.boolean("synthesize.frontend.present", hparams.frontend_present)) {
        return false;
    }
    if (!hparams.frontend_present) {
        std::fprintf(stderr, "omnivoice: this family consumes raw text and needs a frontend\n");
        return false;
    }
    if (!meta.string("synthesize.frontend.provider", hparams.frontend_provider) ||
        !meta.u32("synthesize.frontend.contract_version", hparams.frontend_contract_version)) {
        return false;
    }
    if (hparams.frontend_provider != "synthesize.qwen_bpe" ||
        hparams.frontend_contract_version != kFrontendContractVersion) {
        std::fprintf(stderr, "omnivoice: unsupported frontend %s version %u\n", hparams.frontend_provider.c_str(),
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
                    read_capabilities(meta, hparams) && read_voice(meta) && read_generation(meta, hparams) &&
                    read_generator(meta, hparams) && read_audio_canvas(meta, hparams) && read_codec(meta, hparams) &&
                    read_semantic(meta, hparams) && read_tokens(meta, hparams) && read_languages(meta, hparams) &&
                    read_profile_contract(meta, hparams) && read_frontend(meta, hparams);
    return ok ? SYNTH_OK : SYNTH_ERR_GGUF;
}

}  // namespace synth::omnivoice
