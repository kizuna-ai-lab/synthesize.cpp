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
// `synthesize-check-unit` at 89/91, unchanged. This file is the coverage
// that line was missing -- it asks the real Base GGUF, through
// synth_model_get_voice_profile_capabilities and synth_model_get_preset_voice_count,
// the two questions only the public seam can answer honestly: does the
// package advertise Reference Audio support, and does requesting a Voice
// from it -- named or not -- refuse rather than guess.
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

// The six reference-format/limit values Task 6 shipped into the Base
// package's `synthesize.reference.*` metadata, and Task 9's capability
// snapshot re-reports through `synth_model_get_voice_profile_capabilities`.
// docs/porting/families/qwen3-tts.md's Stage 2 section records these as
// safety ceilings pulled from the plan, not as perceptually validated
// bounds -- this test asserts the ABI reports them faithfully, not that they
// are the right numbers.
constexpr uint32_t kReferenceSampleRate     = 24000;
constexpr uint32_t kReferenceChannelCount   = 1;
constexpr uint64_t kMinReferenceFrames      = 24000;
constexpr uint64_t kMaxReferenceFrames      = 720000;
constexpr uint64_t kMaxReferenceTotalFrames = 720000;
constexpr uint64_t kMaxReferenceCount       = 1;

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

    // Exactly these two bits -- not "at least" -- because the all-zero shape
    // this same query returns for a preset-catalog package (CustomVoice) is
    // itself a claim (docs/c-interface.md's "no Voice Profile support at
    // all"), and the deleted-seam-line guard experiment this file exists for
    // turns every one of these into that same all-zero shape.
    SYNTH_TEST_CHECK(capabilities.source_flags ==
                     (SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO | SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE));
    SYNTH_TEST_CHECK(capabilities.reference_transcript == SYNTH_REQUIREMENT_OPTIONAL);
    SYNTH_TEST_CHECK(capabilities.reference_language == SYNTH_REQUIREMENT_OPTIONAL);
    // Never claimed for this family: there is no Description Text path at
    // any stage of the Reference Model Variant Ladder's second rung.
    SYNTH_TEST_CHECK(capabilities.description_language == SYNTH_REQUIREMENT_UNSUPPORTED);

    SYNTH_TEST_CHECK(capabilities.reference_target_sample_rate == kReferenceSampleRate);
    SYNTH_TEST_CHECK(capabilities.reference_target_channel_count == kReferenceChannelCount);
    SYNTH_TEST_CHECK(capabilities.min_reference_frames_per_clip == kMinReferenceFrames);
    SYNTH_TEST_CHECK(capabilities.max_reference_frames_per_clip == kMaxReferenceFrames);
    SYNTH_TEST_CHECK(capabilities.max_reference_total_frames == kMaxReferenceTotalFrames);
    SYNTH_TEST_CHECK(capabilities.max_reference_count == kMaxReferenceCount);

    // Serialized Profile is claimed alongside Reference Audio, so the schema
    // identity it promises must actually be present too.
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

// A request naming no Voice at all. The generic core's preset-lookup guard
// (src/synthesis-request.cpp) does not fire for an empty voice_id against an
// empty catalog -- there is nothing to have missed a name in -- so this
// reaches src/synthesize.cpp's Qwen3Tts branch and then
// Model::resolve_voice's own has_preset_voice_catalog gate
// (src/arch/qwen3-tts/model.cpp), the Catalog-less refusal Task 9 built.
int check_unnamed_voice_refused(synth_context_t * context) {
    synth_request_t request;
    synth_request_init(&request, sizeof(request));
    request.input_kind  = SYNTH_INPUT_TEXT_UTF8;
    request.input_data  = kText;
    request.input_count = std::strlen(kText);

    synth_audio_buffer_t * audio = nullptr;
    synth_result_t         result;
    synth_result_init(&result, sizeof(result));
    const synth_status_t status = synth_synthesize_to_buffer(context, &request, &audio, &result);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_UNSUPPORTED_VOICE);
    SYNTH_TEST_CHECK(audio == nullptr);
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
