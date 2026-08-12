// The real Base package, through the public C interface only.
//
// Task 9 built the capability snapshot (fill_voice_profile_capability) and
// the Catalog-less refusal (has_preset_voice_catalog) as pure functions of
// synthetic HParams, unit-tested without a loaded Model -- deliberately,
// per the controller's ruling, because a `unit`-labelled test cannot depend
// on this family's ~2.5 GB real package (docs/testing.md). That ruling left
// one seam uncovered on purpose until preparation existed to make it
// observable: src/synthesize.cpp's `shared_info(qwen3tts::ModelInfo, ...)`
// copies `info.voice_profile` into the public `synth::ModelInfo` with a
// single line,
//
//     shared.voice_profile = info.voice_profile;
//
// and while the snapshot stayed all-zero for every variant, nothing in the
// unit gate could notice if that line were deleted -- Task 9's Plan 1 review
// confirmed deleting it left `synthesize-check-unit` unchanged. Stage 2 Plan
// 2 is what puts a nonzero field behind it: Base (ProfileSources) now
// publishes SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO |
// SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE with six real reference limits, a
// real schema identity, and a nonzero compatibility id, so this file's own
// check_capabilities below is that seam line's coverage returning for real --
// deleting the line now fails this test rather than passing it silently.
//
// This file is also the only tier that can prove three more things a
// synthetic-HParams unit test cannot: that create_from_reference ->
// serialize -> load_from_memory actually works end to end against the real
// 894-tensor Base GGUF and its real speaker encoder, that every one of this
// family's new public refusal paths (a transcript, a reference_language tag
// -- declared or not, per the Task 9 review's own finding that this rung
// must refuse the field it advertises UNSUPPORTED -- too many clips, a clip
// outside the declared frame bounds, an out-of-contract rate or channel
// count) fires with the real package's own declared limits rather than a
// fixture's, and that a real Preset Voice Catalog request still refuses.
// What it does not duplicate is the family-layer resolve_voice/
// fill_voice_profile_capability logic itself, or the CustomVoice-model
// dispatch guard, both already covered against synthetic HParams/a hand-built
// `synth_model` in tests/qwen3_tts_voice_required_test.cpp.

#include "synthesize.h"
#include "test-assert.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

// A profile-sources package still resolves through the ordinary
// synth_request_t / synth_synthesize_to_buffer seam -- there is no separate
// "voice profile capability probe" request shape. Both calls below use this
// same short sentence; each exercises a different refusal path before
// either one would touch the codec (see the two call sites for which path).
const char * const kText = "Hi.";

int check_capabilities(synth_model_t * model, synth_voice_profile_capabilities_t & capabilities) {
    // The Preset Voice Catalog is empty by design: identity for this variant
    // arrives through a prepared Voice Profile, not a name.
    uint64_t preset_count = 1;
    SYNTH_TEST_CHECK(synth_model_get_preset_voice_count(model, &preset_count) == SYNTH_OK);
    SYNTH_TEST_CHECK(preset_count == 0);

    synth_voice_profile_capabilities_init(&capabilities, sizeof(capabilities));
    SYNTH_TEST_CHECK(synth_model_get_voice_profile_capabilities(model, &capabilities) == SYNTH_OK);

    // Stage 2 Plan 2 landed preparation, so the advertisement is now true --
    // exactly these two bits, not "at least these two", and not the all-zero
    // shape an earlier revision of this file asserted for every variant.
    // src/voice-profile.cpp's create_from_reference/load_from_memory
    // dispatchers now check this Loaded Model's own published
    // SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO/SERIALIZED_PROFILE bits (not just
    // `family == Qwen3Tts`, which the CustomVoice variant shares), so this
    // capability query and what the calls below actually do have to agree.
    SYNTH_TEST_CHECK(capabilities.source_flags ==
                     (SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO | SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE));
    // UNSUPPORTED, not OPTIONAL: this rung implements the x-vector clone mode
    // only, and a transcript names the transcript-assisted mode Plan 3 adds.
    SYNTH_TEST_CHECK(capabilities.reference_transcript == SYNTH_REQUIREMENT_UNSUPPORTED);
    SYNTH_TEST_CHECK(capabilities.reference_language == SYNTH_REQUIREMENT_UNSUPPORTED);
    // Never claimed for this family: there is no Description Text path at
    // any stage of the Reference Model Variant Ladder's second rung.
    SYNTH_TEST_CHECK(capabilities.description_language == SYNTH_REQUIREMENT_UNSUPPORTED);

    // The real package's own declared Voice Profile contract
    // (tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-base.manifest.json's
    // package_contract.profile), not a synthetic fixture's -- this is the one
    // thing only this integration tier can prove.
    SYNTH_TEST_CHECK(capabilities.reference_target_sample_rate == 24000);
    SYNTH_TEST_CHECK(capabilities.reference_target_channel_count == 1);
    SYNTH_TEST_CHECK(capabilities.min_reference_frames_per_clip == 24000);
    SYNTH_TEST_CHECK(capabilities.max_reference_frames_per_clip == 720000);
    SYNTH_TEST_CHECK(capabilities.max_reference_total_frames == 720000);
    SYNTH_TEST_CHECK(capabilities.max_reference_count == 1);

    // Serialized Profile is advertised now, so docs/c-interface.md requires
    // the schema identity to be present rather than absent.
    SYNTH_TEST_CHECK(capabilities.profile_schema != nullptr && capabilities.profile_schema_size > 0);
    SYNTH_TEST_CHECK(std::string(capabilities.profile_schema, size_t(capabilities.profile_schema_size)) ==
                     "qwen3-tts-voice-clone");
    SYNTH_TEST_CHECK(capabilities.profile_schema_version == 1);
    bool compatibility_id_nonzero = false;
    for (uint8_t byte : capabilities.profile_compatibility_id) {
        compatibility_id_nonzero = compatibility_id_nonzero || (byte != 0);
    }
    SYNTH_TEST_CHECK(compatibility_id_nonzero);
    return 0;
}

// The last diagnostic a request emitted, captured through the public sink.
struct SeenDiagnostic {
    bool           seen   = false;
    synth_status_t status = SYNTH_OK;
    std::string    code;
};

void SYNTH_CALL record_diagnostic(void * user_data, const synth_diagnostic_t * diagnostic) {
    SeenDiagnostic * target = static_cast<SeenDiagnostic *>(user_data);
    target->seen            = true;
    target->status          = diagnostic->status;
    target->code.assign(diagnostic->code != nullptr ? diagnostic->code : "", size_t(diagnostic->code_size));
}

// ---------------------------------------------------------------------------
// Voice Profile helpers (Stage 2 Plan 2 Task 9): every call below drives the
// PUBLIC create_from_reference/serialize/load_from_memory seam against the
// real Base package's own declared limits -- no synthetic package, no
// hand-built payload. There is no pinned reference WAV for this family the
// way tests/omnivoice_profile_test.cpp uses one; a short synthesized tone is
// enough to prove the wiring and the refusal paths, since none of the checks
// below are about voice quality.
// ---------------------------------------------------------------------------

// A short, non-silent tone at `sample_rate` -- enough that the ECAPA-TDNN
// speaker encoder measures a real, strictly positive ref_rms rather than
// tripping Task 7's "voice_profile.reference_silent" gate. This test proves
// the public dispatch wiring, not any property of a particular voice, so the
// content itself is arbitrary as long as it is not digital silence.
std::vector<float> make_tone(uint64_t frame_count, uint32_t sample_rate) {
    constexpr double   kFrequencyHz = 220.0;
    constexpr double   kPi          = 3.14159265358979323846;
    std::vector<float> pcm(static_cast<size_t>(frame_count));
    for (uint64_t index = 0; index < frame_count; ++index) {
        const double t                  = double(index) / double(sample_rate);
        pcm[static_cast<size_t>(index)] = float(0.2 * std::sin(2.0 * kPi * kFrequencyHz * t));
    }
    return pcm;
}

synth_voice_reference_t make_reference(const std::vector<float> & pcm, uint32_t sample_rate, uint32_t channel_count) {
    synth_voice_reference_t reference;
    synth_voice_reference_init(&reference, sizeof(reference));
    reference.samples       = pcm.data();
    reference.frame_count   = pcm.size();
    reference.sample_rate   = sample_rate;
    reference.channel_count = channel_count;
    return reference;
}

synth_voice_reference_params_t make_reference_params(const synth_voice_reference_t * references,
                                                     uint64_t                        count,
                                                     const synth_diagnostic_sink_t * diagnostics) {
    synth_voice_reference_params_t params;
    synth_voice_reference_params_init(&params, sizeof(params));
    params.references       = references;
    params.reference_count  = count;
    params.reference_stride = sizeof(synth_voice_reference_t);
    params.diagnostics      = diagnostics;
    return params;
}

// Create -> serialize -> load -> the loaded Profile is usable. Every one of
// these calls returned SYNTH_ERR_UNSUPPORTED_VOICE before this task.
int check_reference_profile_round_trip(synth_model_t * model, const synth_voice_profile_capabilities_t & capabilities) {
    const std::vector<float> pcm =
        make_tone(capabilities.min_reference_frames_per_clip, capabilities.reference_target_sample_rate);
    const synth_voice_reference_t reference =
        make_reference(pcm, capabilities.reference_target_sample_rate, capabilities.reference_target_channel_count);
    const synth_voice_reference_params_t params = make_reference_params(&reference, 1, nullptr);

    synth_voice_profile_t * profile = nullptr;
    SYNTH_TEST_CHECK(synth_voice_profile_create_from_reference(model, &params, &profile) == SYNTH_OK);
    SYNTH_TEST_CHECK(profile != nullptr);

    synth_voice_profile_serialize_params_t serialize_params;
    synth_voice_profile_serialize_params_init(&serialize_params, sizeof(serialize_params));
    synth_byte_buffer_t * bytes = nullptr;
    SYNTH_TEST_CHECK(synth_voice_profile_serialize(profile, &serialize_params, &bytes) == SYNTH_OK);
    SYNTH_TEST_CHECK(bytes != nullptr);
    SYNTH_TEST_CHECK(bytes->data != nullptr && bytes->data_size > 0);

    synth_voice_profile_load_params_t load_params;
    synth_voice_profile_load_params_init(&load_params, sizeof(load_params));
    load_params.data      = bytes->data;
    load_params.data_size = bytes->data_size;

    synth_voice_profile_t * reloaded = nullptr;
    SYNTH_TEST_CHECK(synth_voice_profile_load_from_memory(model, &load_params, &reloaded) == SYNTH_OK);
    SYNTH_TEST_CHECK(reloaded != nullptr);

    synth_voice_profile_free(reloaded);
    synth_byte_buffer_free(bytes);
    synth_voice_profile_free(profile);
    return 0;
}

// A reference transcript names the transcript-assisted mode this rung has no
// implementation for, so it is refused by name rather than silently
// downgraded to the x-vector mode it did not ask for.
int check_reference_transcript_refused(synth_model_t * model, const synth_voice_profile_capabilities_t & capabilities) {
    const std::vector<float> pcm =
        make_tone(capabilities.min_reference_frames_per_clip, capabilities.reference_target_sample_rate);
    synth_voice_reference_t reference =
        make_reference(pcm, capabilities.reference_target_sample_rate, capabilities.reference_target_channel_count);
    const char * transcript   = "hello there";
    reference.transcript      = transcript;
    reference.transcript_size = std::strlen(transcript);

    SeenDiagnostic          diagnostic;
    synth_diagnostic_sink_t sink;
    synth_diagnostic_sink_init(&sink, sizeof(sink));
    sink.emit      = record_diagnostic;
    sink.user_data = &diagnostic;

    const synth_voice_reference_params_t params  = make_reference_params(&reference, 1, &sink);
    synth_voice_profile_t *              profile = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
    SYNTH_TEST_CHECK(synth_voice_profile_create_from_reference(model, &params, &profile) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(profile == nullptr);
    SYNTH_TEST_CHECK(diagnostic.seen);
    SYNTH_TEST_CHECK(diagnostic.status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(diagnostic.code == "voice_profile.transcript_unsupported");
    return 0;
}

// reference_language is advertised SYNTH_REQUIREMENT_UNSUPPORTED
// (check_capabilities above), and docs/c-interface.md requires a caller
// supplying a field declared unsupported to be refused rather than silently
// accepted. Both a real, package-declared tag ("en") and one this package
// does not declare at all ("zz-ZZ") are refused identically and by name --
// the point is that NO language tag is accepted at this rung, not that an
// undeclared one is treated differently from a declared one the way
// OmniVoice's own (fully supported) reference_language handling would.
// Before this task's review fix, both of these returned SYNTH_OK -- "en"
// was written verbatim into the serialized envelope, and "zz-ZZ" was
// accepted despite naming no language this package has ever heard of.
int check_reference_language_refused(synth_model_t * model, const synth_voice_profile_capabilities_t & capabilities) {
    const std::vector<float> pcm =
        make_tone(capabilities.min_reference_frames_per_clip, capabilities.reference_target_sample_rate);

    for (const char * language_tag : { "en", "zz-ZZ" }) {
        synth_voice_reference_t reference =
            make_reference(pcm, capabilities.reference_target_sample_rate, capabilities.reference_target_channel_count);
        reference.language_tag      = language_tag;
        reference.language_tag_size = std::strlen(language_tag);

        SeenDiagnostic          diagnostic;
        synth_diagnostic_sink_t sink;
        synth_diagnostic_sink_init(&sink, sizeof(sink));
        sink.emit      = record_diagnostic;
        sink.user_data = &diagnostic;

        const synth_voice_reference_params_t params  = make_reference_params(&reference, 1, &sink);
        synth_voice_profile_t *              profile = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
        SYNTH_TEST_CHECK(synth_voice_profile_create_from_reference(model, &params, &profile) == SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(profile == nullptr);
        SYNTH_TEST_CHECK(diagnostic.seen);
        SYNTH_TEST_CHECK(diagnostic.status == SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(diagnostic.code == "voice_profile.reference_language_unsupported");
    }
    return 0;
}

// Two clips -> INVALID_ARG (max_reference_count is 1). Neither descriptor
// needs real content: the count is refused before either one is ever read.
int check_two_reference_clips_refused(synth_model_t * model) {
    synth_voice_reference_t references[2];
    synth_voice_reference_init(&references[0], sizeof(references[0]));
    synth_voice_reference_init(&references[1], sizeof(references[1]));
    const synth_voice_reference_params_t params  = make_reference_params(references, 2, nullptr);
    synth_voice_profile_t *              profile = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
    SYNTH_TEST_CHECK(synth_voice_profile_create_from_reference(model, &params, &profile) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(profile == nullptr);
    return 0;
}

// A clip one frame below min_reference_frames_per_clip -> INVALID_ARG (too
// short), and one frame above max_reference_frames_per_clip -> INPUT_TOO_LONG
// (too long) -- at this package's own valid target rate/channels, so the
// refusal is decided from length alone, not shadowed by a format problem.
int check_reference_clip_length_bounds_refused(synth_model_t *                            model,
                                               const synth_voice_profile_capabilities_t & capabilities) {
    const uint32_t rate     = capabilities.reference_target_sample_rate;
    const uint32_t channels = capabilities.reference_target_channel_count;

    const std::vector<float>             too_short(size_t(capabilities.min_reference_frames_per_clip - 1), 0.0f);
    const synth_voice_reference_t        short_reference = make_reference(too_short, rate, channels);
    const synth_voice_reference_params_t short_params    = make_reference_params(&short_reference, 1, nullptr);
    synth_voice_profile_t *              short_profile   = nullptr;
    SYNTH_TEST_CHECK(synth_voice_profile_create_from_reference(model, &short_params, &short_profile) ==
                     SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(short_profile == nullptr);

    const std::vector<float>             too_long(size_t(capabilities.max_reference_frames_per_clip + 1), 0.0f);
    const synth_voice_reference_t        long_reference = make_reference(too_long, rate, channels);
    const synth_voice_reference_params_t long_params    = make_reference_params(&long_reference, 1, nullptr);
    synth_voice_profile_t *              long_profile   = nullptr;
    SYNTH_TEST_CHECK(synth_voice_profile_create_from_reference(model, &long_params, &long_profile) ==
                     SYNTH_ERR_INPUT_TOO_LONG);
    SYNTH_TEST_CHECK(long_profile == nullptr);
    return 0;
}

// The eight-assertion format-gate matrix tests/omnivoice_profile_test.cpp's
// own reviewer FINDING 1 arm already established: every bad rate/channel
// value paired with a clip at BOTH this package's minimum and maximum
// declared length, so a length-dependent regression cannot hide behind only
// one of the two.
int check_reference_format_gate_matrix_refused(synth_model_t *                            model,
                                               const synth_voice_profile_capabilities_t & capabilities) {
    const std::vector<float> clip_min(size_t(capabilities.min_reference_frames_per_clip), 0.0f);
    const std::vector<float> clip_max(size_t(capabilities.max_reference_frames_per_clip), 0.0f);
    const uint32_t           valid_rate = capabilities.reference_target_sample_rate;

    const auto status_for = [&](const std::vector<float> & pcm, uint32_t rate, uint32_t channels) {
        const synth_voice_reference_t        reference = make_reference(pcm, rate, channels);
        const synth_voice_reference_params_t params    = make_reference_params(&reference, 1, nullptr);
        synth_voice_profile_t *              profile   = nullptr;
        const synth_status_t status = synth_voice_profile_create_from_reference(model, &params, &profile);
        if (profile != nullptr) {
            synth_voice_profile_free(profile);
        }
        return status;
    };

    // rate 0.
    SYNTH_TEST_CHECK(status_for(clip_min, 0, 1) == SYNTH_ERR_UNSUPPORTED_INPUT);
    SYNTH_TEST_CHECK(status_for(clip_max, 0, 1) == SYNTH_ERR_UNSUPPORTED_INPUT);
    // rate 7999: one below SYNTH_REFERENCE_SAMPLE_RATE_MIN.
    SYNTH_TEST_CHECK(status_for(clip_min, 7999, 1) == SYNTH_ERR_UNSUPPORTED_INPUT);
    SYNTH_TEST_CHECK(status_for(clip_max, 7999, 1) == SYNTH_ERR_UNSUPPORTED_INPUT);
    // rate 192001: one above SYNTH_REFERENCE_SAMPLE_RATE_MAX.
    SYNTH_TEST_CHECK(status_for(clip_min, 192001, 1) == SYNTH_ERR_UNSUPPORTED_INPUT);
    SYNTH_TEST_CHECK(status_for(clip_max, 192001, 1) == SYNTH_ERR_UNSUPPORTED_INPUT);
    // 3 channels: one above SYNTH_REFERENCE_CHANNELS_MAX, at this package's
    // own valid target sample rate.
    SYNTH_TEST_CHECK(status_for(clip_min, valid_rate, 3) == SYNTH_ERR_UNSUPPORTED_INPUT);
    SYNTH_TEST_CHECK(status_for(clip_max, valid_rate, 3) == SYNTH_ERR_UNSUPPORTED_INPUT);
    return 0;
}

// A request naming no Voice at all. The generic core's preset-lookup guard
// (src/synthesis-request.cpp) does not fire for an empty voice_id against an
// empty catalog -- there is nothing to have missed a name in -- so this
// reaches src/synthesize.cpp's Qwen3Tts branch and then
// Model::resolve_voice (src/arch/qwen3-tts/model.cpp), whose preset lookup
// refuses because a profile-sources package reaches it with an empty catalog.
//
// This is also the only path that reaches src/synthesize.cpp's
// `synthesis.voice_unsupported` diagnostic, so the code is asserted here: the
// ABI has exactly one voice-error status, which makes the diagnostic code the
// only thing that tells a caller a Voice refusal from a codec that failed to
// run. Both used to arrive as `synthesis.graph_failed`.
int check_unnamed_voice_refused(synth_context_t * context) {
    SeenDiagnostic          diagnostic;
    synth_diagnostic_sink_t sink;
    synth_diagnostic_sink_init(&sink, sizeof(sink));
    sink.emit      = record_diagnostic;
    sink.user_data = &diagnostic;

    synth_request_t request;
    synth_request_init(&request, sizeof(request));
    request.input_kind  = SYNTH_INPUT_TEXT_UTF8;
    request.input_data  = kText;
    request.input_count = std::strlen(kText);
    request.diagnostics = &sink;

    synth_audio_buffer_t * audio = nullptr;
    synth_result_t         result;
    synth_result_init(&result, sizeof(result));
    const synth_status_t status = synth_synthesize_to_buffer(context, &request, &audio, &result);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_UNSUPPORTED_VOICE);
    SYNTH_TEST_CHECK(audio == nullptr);
    SYNTH_TEST_CHECK(diagnostic.seen);
    SYNTH_TEST_CHECK(diagnostic.status == SYNTH_ERR_UNSUPPORTED_VOICE);
    SYNTH_TEST_CHECK(diagnostic.code == "synthesis.voice_unsupported");
    synth_audio_buffer_free(audio);
    return 0;
}

// A request naming a voice_id that could never exist in an empty catalog.
// This is the *other* refusal path: the generic core's own preset lookup
// (src/synthesis-request.cpp) finds no match and returns
// SYNTH_ERR_UNSUPPORTED_VOICE before the request ever reaches this family's
// code at all. docs/c-interface.md defines exactly one voice-error status,
// so a caller cannot tell "no catalog" from "wrong name" apart by status
// code -- both collapse to the same refusal, which is the property this
// checks.
int check_named_voice_refused(synth_context_t * context) {
    const char * voice_id = "aiden";  // a real CustomVoice speaker; absent here on purpose.

    synth_request_t request;
    synth_request_init(&request, sizeof(request));
    request.input_kind    = SYNTH_INPUT_TEXT_UTF8;
    request.input_data    = kText;
    request.input_count   = std::strlen(kText);
    request.voice_id      = voice_id;
    request.voice_id_size = std::strlen(voice_id);

    synth_audio_buffer_t * audio = nullptr;
    synth_result_t         result;
    synth_result_init(&result, sizeof(result));
    const synth_status_t status = synth_synthesize_to_buffer(context, &request, &audio, &result);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_UNSUPPORTED_VOICE);
    SYNTH_TEST_CHECK(audio == nullptr);
    synth_audio_buffer_free(audio);
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    SYNTH_TEST_CHECK(argc == 2);
    const char * model_path = argv[1];

    synth_model_load_params_t load_params;
    synth_model_load_params_init(&load_params, sizeof(load_params));
    load_params.backend = SYNTH_BACKEND_CPU;

    synth_model_t * model = nullptr;
    SYNTH_TEST_CHECK(synth_model_load(model_path, &load_params, &model) == SYNTH_OK);
    SYNTH_TEST_CHECK(model != nullptr);

    synth_voice_profile_capabilities_t capabilities;
    SYNTH_TEST_CHECK(check_capabilities(model, capabilities) == 0);

    SYNTH_TEST_CHECK(check_reference_profile_round_trip(model, capabilities) == 0);
    SYNTH_TEST_CHECK(check_reference_transcript_refused(model, capabilities) == 0);
    SYNTH_TEST_CHECK(check_reference_language_refused(model, capabilities) == 0);
    SYNTH_TEST_CHECK(check_two_reference_clips_refused(model) == 0);
    SYNTH_TEST_CHECK(check_reference_clip_length_bounds_refused(model, capabilities) == 0);
    SYNTH_TEST_CHECK(check_reference_format_gate_matrix_refused(model, capabilities) == 0);

    synth_context_t * context = nullptr;
    SYNTH_TEST_CHECK(synth_context_create(model, &context) == SYNTH_OK);
    SYNTH_TEST_CHECK(context != nullptr);

    SYNTH_TEST_CHECK(check_unnamed_voice_refused(context) == 0);
    SYNTH_TEST_CHECK(check_named_voice_refused(context) == 0);

    synth_context_free(context);
    synth_model_free(model);
    return 0;
}
