// The real Base package, through the public C interface only.
//
// Task 9 built the capability snapshot (fill_voice_profile_capability) and
// the Catalog-less refusal (has_preset_voice_catalog) as pure functions of
// synthetic HParams, unit-tested without a loaded Model -- deliberately,
// per the controller's ruling, because a `unit`-labelled test cannot depend
// on this family's ~2.5 GB real package (docs/testing.md). That ruling left
// one seam uncovered on purpose: src/synthesize.cpp's
// `shared_info(qwen3tts::ModelInfo, ...)` copies `info.voice_profile` into
// the public `synth::ModelInfo` with a single line,
//
//     shared.voice_profile = info.voice_profile;
//
// and nothing in the unit gate reaches far enough to notice if that line is
// deleted: Task 9's own review confirmed deleting it left
// `synthesize-check-unit` at 89/91, unchanged.
//
// This file was written as that line's coverage, and the branch's final
// review then removed the thing it covered: the capability snapshot this
// family reports is all-zero for every variant until Plan 2 can actually
// prepare a Profile (see check_capabilities below for the contract that
// decides it). An all-zero copy of an all-zero struct is unobservable, so
// the seam line is uncovered again and will stay so until Plan 2 puts a
// nonzero field in it. What this file does still prove -- and only the
// public seam can -- is that the real 894-tensor Base GGUF loads, that it
// reports an empty Preset Voice Catalog, and that requesting a Voice from
// it, named or not, refuses rather than guesses.
//
// Deliberately thin otherwise: it does not attempt a real synthesis (Plan 2
// is what makes one succeed) and does not duplicate the family-layer
// resolve_voice/fill_voice_profile_capability tests
// (tests/qwen3_tts_voice_required_test.cpp), which already exercise both
// functions' logic against synthetic HParams covering CustomVoice, Base, and
// the adversarial in-between states. What only this file can prove is that
// the real package's on-disk metadata reaches the public struct at all.

#include "synthesize.h"
#include "test-assert.h"

#include <cstdint>
#include <cstring>
#include <string>

namespace {

// A profile-sources package still resolves through the ordinary
// synth_request_t / synth_synthesize_to_buffer seam -- there is no separate
// "voice profile capability probe" request shape. Both calls below use this
// same short sentence; each exercises a different refusal path before
// either one would touch the codec (see the two call sites for which path).
const char * const kText = "Hi.";

int check_capabilities(synth_model_t * model) {
    // The Preset Voice Catalog is empty by design: identity for this variant
    // arrives through a prepared Voice Profile, not a name.
    uint64_t preset_count = 1;
    SYNTH_TEST_CHECK(synth_model_get_preset_voice_count(model, &preset_count) == SYNTH_OK);
    SYNTH_TEST_CHECK(preset_count == 0);

    synth_voice_profile_capabilities_t capabilities;
    synth_voice_profile_capabilities_init(&capabilities, sizeof(capabilities));
    SYNTH_TEST_CHECK(synth_model_get_voice_profile_capabilities(model, &capabilities) == SYNTH_OK);

    // Exactly zero -- not "at least these bits", and not the two bits an
    // earlier revision of this file asserted. The Base package really does
    // carry a Voice Profile contract (`synthesize.profile.*` plus
    // `synthesize.reference.*`, written by Task 6 and validated at load time
    // by read_profile_contract), and the runtime really cannot honour it:
    // src/voice-profile.cpp guards every Profile source on
    // `family != ModelFamily::Omnivoice`, so nothing in this build can
    // create or consume a Qwen3-TTS Profile. docs/c-interface.md settles
    // which of those two facts this query reports -- "A Model without runtime
    // Voice Profile support reports zero flags", and for an unsupported
    // source every field describing it is UNSUPPORTED or zero, down to the
    // null schema pointer and the 32 zero ID bytes below. The package
    // declaring a contract and the runtime advertising a capability are
    // different statements; only the second belongs here, and only the second
    // would be false today.
    //
    // Plan 2 is what flips this: when the speaker encoder and Profile
    // preparation land, this assertion becomes
    // REFERENCE_AUDIO | SERIALIZED_PROFILE with the six reference limits the
    // package already carries, and the schema identity below becomes present.
    //
    // Stated plainly, because it is a real cost of that decision: while the
    // snapshot is all-zero, this file no longer covers
    // src/synthesize.cpp's `shared.voice_profile = info.voice_profile;` --
    // deleting that line produces the identical all-zero answer. It is a
    // no-op copy today and stays only because Plan 2 makes it carry
    // something; the coverage it was written for returns with the first
    // nonzero field. What this file still proves at the ABI is everything
    // below: the empty Preset Voice Catalog, and both refusal paths.
    SYNTH_TEST_CHECK(capabilities.source_flags == 0);
    SYNTH_TEST_CHECK(capabilities.reference_transcript == SYNTH_REQUIREMENT_UNSUPPORTED);
    SYNTH_TEST_CHECK(capabilities.reference_language == SYNTH_REQUIREMENT_UNSUPPORTED);
    // Never claimed for this family: there is no Description Text path at
    // any stage of the Reference Model Variant Ladder's second rung.
    SYNTH_TEST_CHECK(capabilities.description_language == SYNTH_REQUIREMENT_UNSUPPORTED);

    SYNTH_TEST_CHECK(capabilities.reference_target_sample_rate == 0);
    SYNTH_TEST_CHECK(capabilities.reference_target_channel_count == 0);
    SYNTH_TEST_CHECK(capabilities.min_reference_frames_per_clip == 0);
    SYNTH_TEST_CHECK(capabilities.max_reference_frames_per_clip == 0);
    SYNTH_TEST_CHECK(capabilities.max_reference_total_frames == 0);
    SYNTH_TEST_CHECK(capabilities.max_reference_count == 0);

    // Serialized Profile is unadvertised, so docs/c-interface.md requires the
    // schema identity to be absent rather than merely unused.
    SYNTH_TEST_CHECK(capabilities.profile_schema == nullptr);
    SYNTH_TEST_CHECK(capabilities.profile_schema_size == 0);
    SYNTH_TEST_CHECK(capabilities.profile_schema_version == 0);
    for (uint8_t byte : capabilities.profile_compatibility_id) {
        SYNTH_TEST_CHECK(byte == 0);
    }
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

    SYNTH_TEST_CHECK(check_capabilities(model) == 0);

    synth_context_t * context = nullptr;
    SYNTH_TEST_CHECK(synth_context_create(model, &context) == SYNTH_OK);
    SYNTH_TEST_CHECK(context != nullptr);

    SYNTH_TEST_CHECK(check_unnamed_voice_refused(context) == 0);
    SYNTH_TEST_CHECK(check_named_voice_refused(context) == 0);

    synth_context_free(context);
    synth_model_free(model);
    return 0;
}
