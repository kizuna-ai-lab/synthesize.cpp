#include "arch/omnivoice/profile.h"
#include "audio-normalizer.h"
#include "bcp47.h"
#include "model-handle.h"
#include "model-info.h"
#include "synthesize.h"
#include "voice-profile-handle.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

struct ByteBufferStorage {
    synth_byte_buffer_t        public_value{};
    std::unique_ptr<uint8_t[]> data;
};

template <typename T> void initialize_struct(T * output, uint64_t struct_size) {
    if (output == nullptr || struct_size == 0) {
        return;
    }
    std::memset(output, 0, std::min(static_cast<size_t>(struct_size), sizeof(*output)));
    if (struct_size >= sizeof(output->struct_size)) {
        output->struct_size = struct_size;
    }
}

template <typename T, typename V> V read_visible(const T * input, size_t offset, V default_value) {
    if (input == nullptr || input->struct_size < offset + sizeof(V)) {
        return default_value;
    }
    V value;
    std::memcpy(&value, reinterpret_cast<const unsigned char *>(input) + offset, sizeof(value));
    return value;
}

template <typename T> void write_visible(T * output, size_t offset, const void * value, size_t value_size) {
    if (output == nullptr || output->struct_size < offset + value_size) {
        return;
    }
    std::memcpy(reinterpret_cast<unsigned char *>(output) + offset, value, value_size);
}

bool valid_diagnostic_sink(const synth_diagnostic_sink_t * sink) {
    return sink == nullptr ||
           (sink->struct_size >= offsetof(synth_diagnostic_sink_t, emit) + sizeof(sink->emit) && sink->emit != nullptr);
}

template <typename T> synth_status_t validate_unsupported_params(const T * params, size_t diagnostics_offset) {
    if (params == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (params->struct_size < sizeof(uint64_t)) {
        return SYNTH_ERR_BAD_STRUCT_SIZE;
    }
    const synth_diagnostic_sink_t * diagnostics =
        read_visible(params, diagnostics_offset, static_cast<const synth_diagnostic_sink_t *>(nullptr));
    return valid_diagnostic_sink(diagnostics) ? SYNTH_ERR_UNSUPPORTED_VOICE : SYNTH_ERR_INVALID_ARG;
}

// Duplicated from synthesize.cpp's own local copy rather than shared: this is
// this file's first diagnostic emission (every rejection before Task 14
// returned a bare status), and a second small copy is the codebase's own
// tolerance -- `valid_diagnostic_sink` above already exists in both files for
// the same reason. A third copy would be the signal to hoist both into one
// place.
void emit_diagnostic(const synth_diagnostic_sink_t * sink,
                     synth_status_t                  status,
                     const char *                    code,
                     const char *                    message) {
    if (sink == nullptr || !valid_diagnostic_sink(sink) || code == nullptr || message == nullptr) {
        return;
    }
    synth_diagnostic_t diagnostic{};
    diagnostic.struct_size  = sizeof(diagnostic);
    diagnostic.level        = SYNTH_DIAGNOSTIC_ERROR;
    diagnostic.status       = status;
    diagnostic.code         = code;
    diagnostic.code_size    = std::strlen(code);
    diagnostic.message      = message;
    diagnostic.message_size = std::strlen(message);
    try {
        sink->emit(sink->user_data, &diagnostic);
    } catch (...) {
        // Foreign callbacks may not unwind through the public C ABI.
    }
}

// ---------------------------------------------------------------------------
// BCP-47 shape and declared-language matching, duplicated from
// synthesis-request.cpp's own (anonymous-namespace-local, so not reusable
// without exposing it) `valid_bcp47_shape`/`declared_language`: the Reference
// Audio descriptor's optional `language_tag` gets the exact same validation a
// synthesis request's does, per Task 14's brief ("mirror").
// ---------------------------------------------------------------------------

bool ascii_alpha(char value) {
    const unsigned char character = static_cast<unsigned char>(value);
    return std::isalpha(character) != 0 && character < 128;
}

bool ascii_digit(char value) {
    const unsigned char character = static_cast<unsigned char>(value);
    return std::isdigit(character) != 0 && character < 128;
}

bool ascii_alnum(char value) {
    return ascii_alpha(value) || ascii_digit(value);
}

bool valid_bcp47_shape(const char * value, size_t size) {
    if (value == nullptr || size < 2 || size > 63) {
        return false;
    }
    size_t subtag_start = 0;
    size_t subtag_count = 0;
    for (size_t index = 0; index <= size; ++index) {
        if (index != size && value[index] != '-') {
            if (!ascii_alnum(value[index])) {
                return false;
            }
            continue;
        }
        const size_t subtag_size = index - subtag_start;
        if (subtag_size == 0 || subtag_size > 8) {
            return false;
        }
        if (subtag_count == 0) {
            if (subtag_size < 2) {
                return false;
            }
            for (size_t item = subtag_start; item < index; ++item) {
                if (!ascii_alpha(value[item])) {
                    return false;
                }
            }
        }
        ++subtag_count;
        subtag_start = index + 1;
    }
    return true;
}

bool equals_ascii_case(const char * value, size_t size, const std::string & expected) {
    if (expected.size() != size) {
        return false;
    }
    for (size_t index = 0; index < size; ++index) {
        const unsigned char left  = static_cast<unsigned char>(value[index]);
        const unsigned char right = static_cast<unsigned char>(expected[index]);
        if (std::tolower(left) != std::tolower(right)) {
            return false;
        }
    }
    return true;
}

// Exact match, then the primary subtag for entries that accept a region --
// synthesis-request.cpp's own `declared_language`, minus the "which entry
// answered" return value this call site does not need. The region-shape
// check itself is bcp47.h's shared `is_bcp47_region_subtag` (PR #6 triage
// FIX 5) -- see that header's own comment for why this used to be a third
// independent (and identically incomplete) copy.
bool declared_language(const std::vector<synth::LanguageCapability> & languages, const char * value, size_t size) {
    for (const synth::LanguageCapability & entry : languages) {
        if (equals_ascii_case(value, size, entry.tag)) {
            return true;
        }
    }
    const char * separator = static_cast<const char *>(std::memchr(value, '-', size));
    if (separator == nullptr) {
        return false;
    }
    const size_t primary       = static_cast<size_t>(separator - value);
    const size_t suffix_offset = primary + 1;
    if (!synth::is_bcp47_region_subtag(value + suffix_offset, size - suffix_offset)) {
        return false;
    }
    for (const synth::LanguageCapability & entry : languages) {
        if ((entry.flags & SYNTH_LANGUAGE_REGIONAL_FALLBACK) != 0 && equals_ascii_case(value, primary, entry.tag)) {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// OmniVoice's create_from_reference handler (Task 14): the first Voice
// Profile source any family implements for real. `model`/`params` are
// already known non-null with a params struct_size of at least
// sizeof(uint64_t) by the caller below.
// ---------------------------------------------------------------------------

synth_status_t create_omnivoice_profile_from_reference(const synth_model_t *                  model,
                                                       const synth_voice_reference_params_t * params,
                                                       synth_voice_profile_t **               out_profile) {
    const synth_diagnostic_sink_t * diagnostics =
        read_visible(params, offsetof(synth_voice_reference_params_t, diagnostics),
                     static_cast<const synth_diagnostic_sink_t *>(nullptr));
    if (!valid_diagnostic_sink(diagnostics)) {
        return SYNTH_ERR_INVALID_ARG;
    }

    const uint64_t reference_count =
        read_visible(params, offsetof(synth_voice_reference_params_t, reference_count), uint64_t(0));
    const uint64_t reference_stride =
        read_visible(params, offsetof(synth_voice_reference_params_t, reference_stride), uint64_t(0));
    const synth_voice_reference_t * references =
        read_visible(params, offsetof(synth_voice_reference_params_t, references),
                     static_cast<const synth_voice_reference_t *>(nullptr));

    const synth::VoiceProfileInfo & capabilities = model->info.voice_profile;

    if (references == nullptr || reference_count == 0) {
        return SYNTH_ERR_INVALID_ARG;
    }
    // Step 1 (brief order): exactly one clip. `max_reference_count` is this
    // package's declared ceiling (1 for OmniVoice); more than that is refused
    // by name rather than silently fused or truncated to the first one.
    if (reference_count > capabilities.max_reference_count) {
        emit_diagnostic(diagnostics, SYNTH_ERR_INVALID_ARG, "voice_profile.too_many_references",
                        "this package accepts at most one Reference Audio clip per Voice Profile");
        return SYNTH_ERR_INVALID_ARG;
    }
    // This family's own implementation only ever reads `references[0]`
    // below -- `reference_stride` is validated (the per-descriptor
    // struct_size floor a moment below) but never used to walk to a SECOND
    // descriptor. That is safe today only because every OmniVoice package
    // that exists declares `max_reference_count == 1`, so the check above
    // already narrows `reference_count` to exactly 1 by the time this line
    // runs (0 was refused above, `references == nullptr` too) -- but that is
    // a PACKAGE METADATA value, not a compile-time constant of this code. A
    // future package declaring a higher ceiling would silently drop every
    // clip past the first rather than fusing or refusing them by name. This
    // refusal names the real limitation in the code itself, rather than
    // leaning on every future package happening to agree with it.
    if (reference_count != 1) {
        emit_diagnostic(diagnostics, SYNTH_ERR_INVALID_ARG, "voice_profile.multi_reference_unsupported",
                        "this package's Voice Profile creation reads a single Reference Audio clip; it does not fuse "
                        "or select among multiple clips");
        return SYNTH_ERR_INVALID_ARG;
    }

    const auto * reference = reinterpret_cast<const synth_voice_reference_t *>(references);
    if (reference->struct_size < sizeof(uint64_t)) {
        return SYNTH_ERR_BAD_STRUCT_SIZE;
    }
    // docs/c-interface.md: reference_stride "must cover every descriptor's
    // declared struct_size".
    if (reference_stride < reference->struct_size) {
        return SYNTH_ERR_INVALID_ARG;
    }

    const float * samples =
        read_visible(reference, offsetof(synth_voice_reference_t, samples), static_cast<const float *>(nullptr));
    const uint64_t frame_count = read_visible(reference, offsetof(synth_voice_reference_t, frame_count), uint64_t(0));
    const uint32_t sample_rate = read_visible(reference, offsetof(synth_voice_reference_t, sample_rate), uint32_t(0));
    const uint32_t channel_count =
        read_visible(reference, offsetof(synth_voice_reference_t, channel_count), uint32_t(0));
    const char * transcript =
        read_visible(reference, offsetof(synth_voice_reference_t, transcript), static_cast<const char *>(nullptr));
    const uint64_t transcript_size =
        read_visible(reference, offsetof(synth_voice_reference_t, transcript_size), uint64_t(0));
    const char * language_tag =
        read_visible(reference, offsetof(synth_voice_reference_t, language_tag), static_cast<const char *>(nullptr));
    const uint64_t language_size =
        read_visible(reference, offsetof(synth_voice_reference_t, language_tag_size), uint64_t(0));

    if ((transcript == nullptr) != (transcript_size == 0) || (language_tag == nullptr) != (language_size == 0)) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (transcript_size > std::numeric_limits<size_t>::max() || language_size > std::numeric_limits<size_t>::max()) {
        return SYNTH_ERR_INVALID_ARG;
    }

    // Step 2: transcript required.
    if (capabilities.reference_transcript == SYNTH_REQUIREMENT_REQUIRED &&
        (transcript == nullptr || transcript_size == 0)) {
        emit_diagnostic(diagnostics, SYNTH_ERR_INVALID_ARG, "voice_profile.transcript_required",
                        "this package requires a transcript for Reference Audio cloning; none was supplied");
        return SYNTH_ERR_INVALID_ARG;
    }
    const std::string transcript_text(transcript != nullptr ? transcript : "", static_cast<size_t>(transcript_size));

    // Step 3: language optional, validated against the package's declared
    // languages when present -- mirroring synthesis-request.cpp's own
    // request-language validation.
    std::string language_text;
    if (language_tag != nullptr) {
        if (!valid_bcp47_shape(language_tag, static_cast<size_t>(language_size))) {
            return SYNTH_ERR_INVALID_ARG;
        }
        if (!declared_language(model->info.languages, language_tag, static_cast<size_t>(language_size))) {
            return SYNTH_ERR_UNSUPPORTED_LANGUAGE;
        }
        language_text.assign(language_tag, static_cast<size_t>(language_size));
    }

    // Step 4: the Audio Normalizer, RFE-prechecked against this package's
    // declared limits before any resample allocation (docs/c-interface.md:
    // 486-496).
    if (samples == nullptr || frame_count == 0) {
        return SYNTH_ERR_INVALID_ARG;
    }
    // Format precondition, checked BEFORE reference_frame_equivalent() below
    // ever runs (reviewer FINDING 1): an out-of-contract sample
    // rate or channel count must never be shadowed by whatever the RFE math
    // derives from it. Before this check existed here, rate 0 always
    // produced "reference_too_short" (INVALID_ARG) -- reference_frame_equivalent
    // special-cases a zero rate to 0, which is always below
    // min_frames_per_clip -- and rate 4000 produced three DIFFERENT statuses
    // depending on clip length, because the RFE precheck happily computed
    // something plausible-looking from the invalid rate before this ran.
    // synth::validate_reference_format is the SAME check
    // audio-normalizer.cpp's own normalize_reference below applies again,
    // later, on the same inputs -- audio-normalizer.h is the authority on
    // this range, not a second set of literals here.
    const synth_status_t format_status = synth::validate_reference_format(sample_rate, channel_count);
    if (format_status != SYNTH_OK) {
        return format_status;
    }
    const uint64_t rfe =
        synth::reference_frame_equivalent(frame_count, sample_rate, capabilities.reference_target_sample_rate);
    if (rfe > capabilities.max_frames_per_clip || rfe > capabilities.max_total_frames) {
        emit_diagnostic(diagnostics, SYNTH_ERR_INPUT_TOO_LONG, "voice_profile.reference_too_long",
                        "the reference clip exceeds this package's maximum Reference Audio length");
        return SYNTH_ERR_INPUT_TOO_LONG;
    }
    if (rfe < capabilities.min_frames_per_clip) {
        emit_diagnostic(diagnostics, SYNTH_ERR_INVALID_ARG, "voice_profile.reference_too_short",
                        "the reference clip is shorter than this package's minimum Reference Audio length");
        return SYNTH_ERR_INVALID_ARG;
    }

    synth::NormalizedReference normalized;
    const synth_status_t       normalize_status = synth::normalize_reference(
        samples, frame_count, sample_rate, channel_count, capabilities.reference_target_sample_rate,
        capabilities.reference_target_channels, normalized);
    if (normalize_status != SYNTH_OK) {
        return normalize_status;
    }

    // Step 5: the family's own encode chain, plus the silent-reference
    // rejection (jiangzhuo's 2026-08-01 ruling) -- both inside
    // create_clone_prompt.
    std::shared_ptr<const synth::omnivoice::ClonePrompt> clone_prompt;
    const char *                                         diagnostic_code    = nullptr;
    const char *                                         diagnostic_message = nullptr;
    const synth_status_t                                 clone_status =
        synth::omnivoice::create_clone_prompt(*model->omnivoice, normalized.pcm, transcript_text, language_text, 0,
                                              clone_prompt, diagnostic_code, diagnostic_message);
    if (clone_status != SYNTH_OK) {
        emit_diagnostic(diagnostics, clone_status, diagnostic_code, diagnostic_message);
        return clone_status;
    }

    auto profile        = std::make_unique<synth_voice_profile>();
    profile->model      = model;
    profile->family_tag = synth::ProfileFamilyTag::OmnivoiceClone;
    profile->payload    = std::move(clone_prompt);
    *out_profile        = profile.release();
    return SYNTH_OK;
}

// ---------------------------------------------------------------------------
// OmniVoice's create_from_description handler (Task 15): the second Voice
// Profile source this family implements for real. `model`/`params` are
// already known non-null with a params struct_size of at least
// sizeof(uint64_t) by the caller below.
// ---------------------------------------------------------------------------

synth_status_t create_omnivoice_profile_from_description(const synth_model_t *                    model,
                                                         const synth_voice_description_params_t * params,
                                                         synth_voice_profile_t **                 out_profile) {
    const synth_diagnostic_sink_t * diagnostics =
        read_visible(params, offsetof(synth_voice_description_params_t, diagnostics),
                     static_cast<const synth_diagnostic_sink_t *>(nullptr));
    if (!valid_diagnostic_sink(diagnostics)) {
        return SYNTH_ERR_INVALID_ARG;
    }

    const char *   description = read_visible(params, offsetof(synth_voice_description_params_t, description),
                                              static_cast<const char *>(nullptr));
    const uint64_t description_size =
        read_visible(params, offsetof(synth_voice_description_params_t, description_size), uint64_t(0));
    const char *   language_tag = read_visible(params, offsetof(synth_voice_description_params_t, language_tag),
                                               static_cast<const char *>(nullptr));
    const uint64_t language_size =
        read_visible(params, offsetof(synth_voice_description_params_t, language_tag_size), uint64_t(0));
    const uint64_t seed = read_visible(params, offsetof(synth_voice_description_params_t, seed), uint64_t(0));

    if ((description == nullptr) != (description_size == 0) || (language_tag == nullptr) != (language_size == 0)) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (description_size > std::numeric_limits<size_t>::max() || language_size > std::numeric_limits<size_t>::max()) {
        return SYNTH_ERR_INVALID_ARG;
    }

    // Step 1: description required, non-empty (docs/c-interface.md:566:
    // "description is required, non-empty, length-delimited UTF-8").
    if (description == nullptr || description_size == 0) {
        emit_diagnostic(diagnostics, SYNTH_ERR_INVALID_ARG, "voice_profile.description_required",
                        "a Description Text Voice Profile requires a non-empty description");
        return SYNTH_ERR_INVALID_ARG;
    }
    const std::string description_text(description, static_cast<size_t>(description_size));

    // Step 2: the preparation seed must be concrete (docs/c-interface.md:570,
    // "v1 profile preparation rejects SYNTH_SEED_RANDOM"; docs/voice-conditioning.md:57,
    // "a concrete preparation seed"). This family's Description Text
    // preparation is itself deterministic -- pure vocabulary resolution, no
    // sampling -- so the seed is accepted here and otherwise unused;
    // rejecting the sentinel is what stops a caller from building a Voice
    // Profile it could never reproduce, not a randomness concern of this
    // family's own. The Python binding
    // (bindings/python/src/synthesize_cpp/voice_profiles.py's
    // _concrete_profile_seed) already refuses the sentinel before it ever
    // reaches this function; a direct C caller has no such gate, so this
    // port enforces it here too.
    if (seed == SYNTH_SEED_RANDOM) {
        emit_diagnostic(diagnostics, SYNTH_ERR_INVALID_ARG, "voice_profile.seed_must_be_concrete",
                        "Voice Profile preparation does not accept SYNTH_SEED_RANDOM; generate a concrete seed first");
        return SYNTH_ERR_INVALID_ARG;
    }

    // Step 3: description_language, validated against this family's own
    // description-language support (en/zh -- upstream's trained set) rather
    // than the synthesis Language Capability Catalog (docs/c-interface.md:568:
    // "validated against the Model Variant's description-language support
    // rather than its synthesis Language Capability Catalog"). A null tag
    // selects the Model Package's declared default, a FIXED "en" -- NEVER
    // detected from the description's own text: docs/c-interface.md is
    // explicit ("The implementation never detects the description language
    // from its text"), and this port follows that literally even though it
    // means a Chinese-script description with no explicit tag resolves as
    // "en" until the caller says otherwise. (This is a deliberate reading of
    // a looser earlier brief that suggested sniffing the description for
    // CJK content for this default; the confirmed contract's explicit
    // prohibition governs.)
    bool use_zh = false;  // the "en" default
    if (language_tag != nullptr) {
        if (!valid_bcp47_shape(language_tag, static_cast<size_t>(language_size))) {
            return SYNTH_ERR_INVALID_ARG;
        }
        const bool is_en = equals_ascii_case(language_tag, static_cast<size_t>(language_size), "en");
        const bool is_zh = equals_ascii_case(language_tag, static_cast<size_t>(language_size), "zh");
        if (!is_en && !is_zh) {
            emit_diagnostic(
                diagnostics, SYNTH_ERR_UNSUPPORTED_INPUT, "voice_profile.description_language_unsupported",
                "this package's Description Text only supports the \"en\" and \"zh\" description languages");
            return SYNTH_ERR_UNSUPPORTED_INPUT;
        }
        use_zh = is_zh;
    }

    // Step 4: the vocabulary resolution itself (profile.cpp). `use_zh` here
    // is the unification BASELINE only -- resolve_instruct's own
    // dialect/accent overrides (a dialect item forces Chinese, an accent
    // item forces English) still apply on top of it, exactly as upstream's
    // own override does on top of whatever baseline its call site computed.
    std::shared_ptr<const synth::omnivoice::DesignInstruct> design;
    const char *                                            diagnostic_code = nullptr;
    std::string                                             diagnostic_message;
    const synth_status_t                                    resolve_status =
        synth::omnivoice::resolve_instruct(description_text, use_zh, design, diagnostic_code, diagnostic_message);
    if (resolve_status != SYNTH_OK) {
        emit_diagnostic(diagnostics, resolve_status, diagnostic_code, diagnostic_message.c_str());
        return resolve_status;
    }

    auto profile        = std::make_unique<synth_voice_profile>();
    profile->model      = model;
    profile->family_tag = synth::ProfileFamilyTag::OmnivoiceDesign;
    profile->payload    = std::move(design);
    *out_profile        = profile.release();
    return SYNTH_OK;
}

// ---------------------------------------------------------------------------
// OmniVoice's serialize / load_from_memory dispatch (Task 16): the v1
// Serialized Profile round trip, ADR 0008 / docs/c-interface.md's "v1
// Serialized Profile GGUF Contract". The GGUF envelope writer/reader itself
// lives in arch/omnivoice/profile.cpp; this file only owns dispatch (which
// family, which of that family's two payload shapes) and the public
// `synth_byte_buffer_t` allocation.
// ---------------------------------------------------------------------------

synth_status_t serialize_omnivoice_profile(const synth_voice_profile *                    profile,
                                           const synth_voice_profile_serialize_params_t * params,
                                           synth_byte_buffer_t **                         out_data) {
    const synth_diagnostic_sink_t * diagnostics =
        read_visible(params, offsetof(synth_voice_profile_serialize_params_t, diagnostics),
                     static_cast<const synth_diagnostic_sink_t *>(nullptr));
    if (!valid_diagnostic_sink(diagnostics)) {
        return SYNTH_ERR_INVALID_ARG;
    }

    const uint8_t (&compatibility_id)[32] = profile->model->info.voice_profile.compatibility_id;

    std::vector<uint8_t> bytes;
    synth_status_t       status;
    if (profile->family_tag == synth::ProfileFamilyTag::OmnivoiceClone) {
        const auto & clone = *static_cast<const synth::omnivoice::ClonePrompt *>(profile->payload.get());
        status             = synth::omnivoice::serialize_clone_prompt(clone, compatibility_id, bytes);
    } else {
        const auto & design = *static_cast<const synth::omnivoice::DesignInstruct *>(profile->payload.get());
        status              = synth::omnivoice::serialize_design_instruct(design, compatibility_id, bytes);
    }
    if (status != SYNTH_OK) {
        return status;
    }

    auto storage  = std::make_unique<ByteBufferStorage>();
    storage->data = std::make_unique<uint8_t[]>(bytes.size());
    if (!bytes.empty()) {
        std::memcpy(storage->data.get(), bytes.data(), bytes.size());
    }
    storage->public_value.struct_size = sizeof(storage->public_value);
    storage->public_value.data        = storage->data.get();
    storage->public_value.data_size   = bytes.size();
    *out_data                         = reinterpret_cast<synth_byte_buffer_t *>(storage.release());
    return SYNTH_OK;
}

synth_status_t load_omnivoice_profile_from_memory(const synth_model_t *                     model,
                                                  const synth_voice_profile_load_params_t * params,
                                                  synth_voice_profile_t **                  out_profile) {
    const synth_diagnostic_sink_t * diagnostics =
        read_visible(params, offsetof(synth_voice_profile_load_params_t, diagnostics),
                     static_cast<const synth_diagnostic_sink_t *>(nullptr));
    if (!valid_diagnostic_sink(diagnostics)) {
        return SYNTH_ERR_INVALID_ARG;
    }

    const uint8_t * data =
        read_visible(params, offsetof(synth_voice_profile_load_params_t, data), static_cast<const uint8_t *>(nullptr));
    const uint64_t data_size =
        read_visible(params, offsetof(synth_voice_profile_load_params_t, data_size), uint64_t(0));
    // docs/c-interface.md: "Loading requires non-null, non-empty bytes."
    if (data == nullptr || data_size == 0) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (data_size > std::numeric_limits<size_t>::max()) {
        return SYNTH_ERR_INVALID_ARG;
    }

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                diagnostic_code    = nullptr;
    const char *                diagnostic_message = nullptr;
    const synth_status_t        status             = synth::omnivoice::load_profile_from_memory(
        *model->omnivoice, data, static_cast<size_t>(data_size), model->info.voice_profile.compatibility_id,
        model->info.voice_profile.max_total_frames, family_tag, payload, diagnostic_code, diagnostic_message);
    if (status != SYNTH_OK) {
        emit_diagnostic(diagnostics, status, diagnostic_code, diagnostic_message);
        return status;
    }

    auto profile        = std::make_unique<synth_voice_profile>();
    profile->model      = model;
    profile->family_tag = family_tag;
    profile->payload    = std::move(payload);
    *out_profile        = profile.release();
    return SYNTH_OK;
}

}  // namespace

void synth_voice_profile_capabilities_init(synth_voice_profile_capabilities_t * capabilities, uint64_t struct_size) {
    initialize_struct(capabilities, struct_size);
}

synth_status_t synth_model_get_voice_profile_capabilities(const synth_model_t *                model,
                                                          synth_voice_profile_capabilities_t * out_capabilities) {
    if (model == nullptr || out_capabilities == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (out_capabilities->struct_size < sizeof(uint64_t)) {
        return SYNTH_ERR_BAD_STRUCT_SIZE;
    }
    const uint64_t struct_size = out_capabilities->struct_size;
    synth_voice_profile_capabilities_init(out_capabilities, struct_size);

    // Generic over every family: a family with no Voice Profile support at
    // all leaves model->info.voice_profile at its all-zero default
    // (VoiceProfileInfo's own default member initializers), which already IS
    // the all-`SYNTH_REQUIREMENT_UNSUPPORTED`-and-zero shape
    // docs/c-interface.md requires for an unsupported source -- so there is
    // nothing family-specific to branch on here.
    const synth::VoiceProfileInfo & profile = model->info.voice_profile;
    write_visible(out_capabilities, offsetof(synth_voice_profile_capabilities_t, source_flags), &profile.source_flags,
                  sizeof(profile.source_flags));
    write_visible(out_capabilities, offsetof(synth_voice_profile_capabilities_t, reference_transcript),
                  &profile.reference_transcript, sizeof(profile.reference_transcript));
    write_visible(out_capabilities, offsetof(synth_voice_profile_capabilities_t, reference_language),
                  &profile.reference_language, sizeof(profile.reference_language));
    write_visible(out_capabilities, offsetof(synth_voice_profile_capabilities_t, description_language),
                  &profile.description_language, sizeof(profile.description_language));
    const uint64_t max_reference_count = profile.max_reference_count;
    write_visible(out_capabilities, offsetof(synth_voice_profile_capabilities_t, max_reference_count),
                  &max_reference_count, sizeof(max_reference_count));
    write_visible(out_capabilities, offsetof(synth_voice_profile_capabilities_t, reference_target_sample_rate),
                  &profile.reference_target_sample_rate, sizeof(profile.reference_target_sample_rate));
    write_visible(out_capabilities, offsetof(synth_voice_profile_capabilities_t, reference_target_channel_count),
                  &profile.reference_target_channels, sizeof(profile.reference_target_channels));
    write_visible(out_capabilities, offsetof(synth_voice_profile_capabilities_t, min_reference_frames_per_clip),
                  &profile.min_frames_per_clip, sizeof(profile.min_frames_per_clip));
    write_visible(out_capabilities, offsetof(synth_voice_profile_capabilities_t, max_reference_frames_per_clip),
                  &profile.max_frames_per_clip, sizeof(profile.max_frames_per_clip));
    write_visible(out_capabilities, offsetof(synth_voice_profile_capabilities_t, max_reference_total_frames),
                  &profile.max_total_frames, sizeof(profile.max_total_frames));

    // profile_schema/profile_schema_size/profile_schema_version/
    // profile_compatibility_id stay at the initializer's null/zero default
    // until SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE is also set (Task 16):
    // docs/c-interface.md's "When Serialized Profile is unsupported ... the
    // 32 ID bytes are zero" rule, which a model can be on the near side of
    // even once it stores real compatibility_id bytes internally
    // (VoiceProfileInfo::compatibility_id, filled from Task 14 on). Once the
    // flag is set, the schema string is borrowed from `model->info` itself
    // (VoiceProfileInfo::schema, a std::string owned by the Loaded Model),
    // which is exactly what lets it "remain valid until synth_model_free()"
    // per that same doc section.
    if ((profile.source_flags & SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE) != 0) {
        const char *   schema_data = profile.schema.c_str();
        const uint64_t schema_size = profile.schema.size();
        write_visible(out_capabilities, offsetof(synth_voice_profile_capabilities_t, profile_schema), &schema_data,
                      sizeof(schema_data));
        write_visible(out_capabilities, offsetof(synth_voice_profile_capabilities_t, profile_schema_size), &schema_size,
                      sizeof(schema_size));
        write_visible(out_capabilities, offsetof(synth_voice_profile_capabilities_t, profile_schema_version),
                      &profile.schema_version, sizeof(profile.schema_version));
        write_visible(out_capabilities, offsetof(synth_voice_profile_capabilities_t, profile_compatibility_id),
                      &profile.compatibility_id, sizeof(profile.compatibility_id));
    }
    return SYNTH_OK;
}

void synth_voice_reference_init(synth_voice_reference_t * reference, uint64_t struct_size) {
    initialize_struct(reference, struct_size);
}

void synth_voice_reference_params_init(synth_voice_reference_params_t * params, uint64_t struct_size) {
    initialize_struct(params, struct_size);
}

synth_status_t synth_voice_profile_create_from_reference(const synth_model_t *                  model,
                                                         const synth_voice_reference_params_t * params,
                                                         synth_voice_profile_t **               out_profile) {
    if (out_profile == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    *out_profile = nullptr;
    if (model == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    // Every other family still takes the generic "unsupported" fallback
    // (struct_size/null checks plus the valid-sink UNSUPPORTED_VOICE-or-
    // INVALID_ARG split), unchanged from before Task 14.
    if (model->info.family != synth::ModelFamily::Omnivoice) {
        return validate_unsupported_params(params, offsetof(synth_voice_reference_params_t, diagnostics));
    }
    if (params == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (params->struct_size < sizeof(uint64_t)) {
        return SYNTH_ERR_BAD_STRUCT_SIZE;
    }
    try {
        return create_omnivoice_profile_from_reference(model, params, out_profile);
    } catch (const std::bad_alloc &) {
        return SYNTH_ERR_OOM;
    } catch (...) {
        return SYNTH_ERR_INTERNAL;
    }
}

void synth_voice_description_params_init(synth_voice_description_params_t * params, uint64_t struct_size) {
    initialize_struct(params, struct_size);
}

synth_status_t synth_voice_profile_create_from_description(const synth_model_t *                    model,
                                                           const synth_voice_description_params_t * params,
                                                           synth_voice_profile_t **                 out_profile) {
    if (out_profile == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    *out_profile = nullptr;
    if (model == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    // Every other family still takes the generic "unsupported" fallback,
    // unchanged from before Task 15 -- the same split
    // create_from_reference's own dispatcher above uses.
    if (model->info.family != synth::ModelFamily::Omnivoice) {
        return validate_unsupported_params(params, offsetof(synth_voice_description_params_t, diagnostics));
    }
    if (params == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (params->struct_size < sizeof(uint64_t)) {
        return SYNTH_ERR_BAD_STRUCT_SIZE;
    }
    try {
        return create_omnivoice_profile_from_description(model, params, out_profile);
    } catch (const std::bad_alloc &) {
        return SYNTH_ERR_OOM;
    } catch (...) {
        return SYNTH_ERR_INTERNAL;
    }
}

void synth_voice_random_params_init(synth_voice_random_params_t * params, uint64_t struct_size) {
    initialize_struct(params, struct_size);
}

synth_status_t synth_voice_profile_create_random(const synth_model_t *               model,
                                                 const synth_voice_random_params_t * params,
                                                 synth_voice_profile_t **            out_profile) {
    if (out_profile == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    *out_profile = nullptr;
    if (model == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    return validate_unsupported_params(params, offsetof(synth_voice_random_params_t, diagnostics));
}

void synth_voice_profile_load_params_init(synth_voice_profile_load_params_t * params, uint64_t struct_size) {
    initialize_struct(params, struct_size);
}

void synth_voice_profile_serialize_params_init(synth_voice_profile_serialize_params_t * params, uint64_t struct_size) {
    initialize_struct(params, struct_size);
}

synth_status_t synth_voice_profile_load_from_memory(const synth_model_t *                     model,
                                                    const synth_voice_profile_load_params_t * params,
                                                    synth_voice_profile_t **                  out_profile) {
    if (out_profile == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    *out_profile = nullptr;
    if (model == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (params == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (params->struct_size < sizeof(uint64_t)) {
        return SYNTH_ERR_BAD_STRUCT_SIZE;
    }
    // `params` is fully validated (non-null, adequately sized) before `model`
    // is ever dereferenced below -- tests/voice_profile_api_test.c drives
    // this exact function with a dummy, never-dereferenced
    // `(synth_model_t *) 1` specifically to pin a too-small params struct
    // returning BAD_STRUCT_SIZE without needing a real model behind it; that
    // ordering has to survive Task 16 wiring a real family dispatch in here.
    if (model->info.family != synth::ModelFamily::Omnivoice) {
        // Every other family still takes the generic "unsupported" fallback,
        // the same one synth_voice_profile_create_from_reference/
        // create_from_description use for a non-OmniVoice model.
        return validate_unsupported_params(params, offsetof(synth_voice_profile_load_params_t, diagnostics));
    }
    try {
        return load_omnivoice_profile_from_memory(model, params, out_profile);
    } catch (const std::bad_alloc &) {
        return SYNTH_ERR_OOM;
    } catch (...) {
        return SYNTH_ERR_INTERNAL;
    }
}

synth_status_t synth_voice_profile_serialize(const synth_voice_profile_t *                  profile,
                                             const synth_voice_profile_serialize_params_t * params,
                                             synth_byte_buffer_t **                         out_data) {
    if (out_data == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    *out_data = nullptr;
    if (profile == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (params != nullptr && params->struct_size < sizeof(uint64_t)) {
        return SYNTH_ERR_BAD_STRUCT_SIZE;
    }
    // `params` is fully validated before `profile` is ever dereferenced
    // below, for the same reason and against the same test as
    // synth_voice_profile_load_from_memory's own ordering comment above --
    // tests/voice_profile_api_test.c's dummy `(synth_voice_profile_t *) 1`
    // case for THIS function relies on it too.
    if (profile->family_tag == synth::ProfileFamilyTag::OmnivoiceClone ||
        profile->family_tag == synth::ProfileFamilyTag::OmnivoiceDesign) {
        try {
            return serialize_omnivoice_profile(profile, params, out_data);
        } catch (const std::bad_alloc &) {
            return SYNTH_ERR_OOM;
        } catch (...) {
            return SYNTH_ERR_INTERNAL;
        }
    }
    // No family currently produces any other tag, but the generic fallback
    // stays here rather than being narrowed to an assert: a Voice Profile's
    // own family_tag is the one thing this dispatcher must never
    // misinterpret as OmniVoice's.
    const synth_diagnostic_sink_t * diagnostics =
        read_visible(params, offsetof(synth_voice_profile_serialize_params_t, diagnostics),
                     static_cast<const synth_diagnostic_sink_t *>(nullptr));
    return valid_diagnostic_sink(diagnostics) ? SYNTH_ERR_UNSUPPORTED_VOICE : SYNTH_ERR_INVALID_ARG;
}

void synth_byte_buffer_free(synth_byte_buffer_t * buffer) {
    if (buffer == nullptr) {
        return;
    }
    auto * storage = reinterpret_cast<ByteBufferStorage *>(buffer);
    delete storage;
}

void synth_voice_profile_free(synth_voice_profile_t * profile) {
    delete profile;
}
