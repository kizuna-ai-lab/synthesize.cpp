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
// family's public refusal paths (a blank transcript, a malformed or
// undeclared reference_language tag, too many clips, a clip outside the
// declared frame bounds, an out-of-contract rate or channel count) fires with
// the real package's own declared limits rather than a fixture's, and that a
// real Preset Voice Catalog request still refuses. What it does not duplicate
// is the family-layer resolve_voice/fill_voice_profile_capability logic
// itself, or the CustomVoice-model dispatch guard, both already covered
// against synthetic HParams/a hand-built `synth_model` in
// tests/qwen3_tts_voice_required_test.cpp.
//
// Stage 2 Plan 3's Task 10 adds the thing this tier alone can prove about the
// CLONE MODE. `reference_transcript` became SYNTH_REQUIREMENT_OPTIONAL in the
// same change that landed ICL, and the transcript's presence became the mode
// selector (D4) -- so the two checks that used to assert a transcript and a
// language tag were REFUSED now assert what they select and how they are
// validated. Selecting the mode needs a live Model, because the transcript is
// tokenized against BPE tables only a loaded package carries, which is why
// src/voice-profile.cpp's ICL arm has no `unit`-layer coverage at all and
// check_transcript_selects_icl_mode below is where it gets covered.
//
// Task 11 adds a fourth thing only this tier proves: that a Profile prepared
// against the real Base package actually SYNTHESIZES -- src/synthesize.cpp's
// Qwen3Tts branch refused every non-null `prepared.voice_profile`
// unconditionally before this task, so check_profile_synthesizes below is
// the first test anywhere to reach model.cpp's x-vector prompt-substitution
// path through the public seam -- and that a Profile presented to a
// synthesis request bound to a DIFFERENT Loaded Model (the same package,
// loaded a second time, so its `synth_model_t*` differs while its weights do
// not) is refused as a Voice error rather than silently misread or left to
// whatever a mismatched enc_dim would do downstream.

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
    // OPTIONAL since Plan 3 landed ICL: both clone modes exist and the
    // transcript's presence selects between them. This is the on-disk
    // package's answer rather than a fixture's, and what the two mode-selector
    // checks below actually do has to agree with it.
    SYNTH_TEST_CHECK(capabilities.reference_transcript == SYNTH_REQUIREMENT_OPTIONAL);
    SYNTH_TEST_CHECK(capabilities.reference_language == SYNTH_REQUIREMENT_OPTIONAL);
    // Never claimed for this family: there is no Description Text path at
    // any stage of the Reference Model Variant Ladder's second rung. Unchanged
    // by Plan 3 -- only the two fields above moved.
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
// The message is captured as well as the code because this family has more
// than one refusal behind a single code: `synthesis.voice_unsupported` covers
// the Catalog-less case, the cross-Model case AND the wrong-clone-mode case
// (the ABI defines exactly one voice-error status, so the code cannot split
// further), and the mode-selector check below needs to tell the third from the
// other two.
struct SeenDiagnostic {
    bool           seen   = false;
    synth_status_t status = SYNTH_OK;
    std::string    code;
    std::string    message;
};

void SYNTH_CALL record_diagnostic(void * user_data, const synth_diagnostic_t * diagnostic) {
    SeenDiagnostic * target = static_cast<SeenDiagnostic *>(user_data);
    target->seen            = true;
    target->status          = diagnostic->status;
    target->code.assign(diagnostic->code != nullptr ? diagnostic->code : "", size_t(diagnostic->code_size));
    target->message.assign(diagnostic->message != nullptr ? diagnostic->message : "", size_t(diagnostic->message_size));
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

// One create_from_reference call over this package's own minimum-length tone,
// optionally carrying a transcript and/or a language tag. `out_profile` is
// pre-poisoned so "left untouched" and "set to null on failure" are
// distinguishable, the same way the refusal checks below rely on.
synth_status_t create_with(synth_model_t *                            model,
                           const synth_voice_profile_capabilities_t & capabilities,
                           const char *                               transcript,
                           const char *                               language_tag,
                           SeenDiagnostic &                           diagnostic,
                           synth_voice_profile_t *&                   out_profile) {
    const std::vector<float> pcm =
        make_tone(capabilities.min_reference_frames_per_clip, capabilities.reference_target_sample_rate);
    synth_voice_reference_t reference =
        make_reference(pcm, capabilities.reference_target_sample_rate, capabilities.reference_target_channel_count);
    if (transcript != nullptr) {
        reference.transcript      = transcript;
        reference.transcript_size = std::strlen(transcript);
    }
    if (language_tag != nullptr) {
        reference.language_tag      = language_tag;
        reference.language_tag_size = std::strlen(language_tag);
    }

    synth_diagnostic_sink_t sink;
    synth_diagnostic_sink_init(&sink, sizeof(sink));
    sink.emit      = record_diagnostic;
    sink.user_data = &diagnostic;

    const synth_voice_reference_params_t params = make_reference_params(&reference, 1, &sink);
    out_profile                                 = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
    return synth_voice_profile_create_from_reference(model, &params, &out_profile);
}

// Presents `profile` to a synthesis request on `context` and reports what came
// back. Used twice below with two Profiles that differ ONLY in their clone
// mode.
int synthesize_with(synth_context_t *       context,
                    synth_voice_profile_t * profile,
                    SeenDiagnostic &        diagnostic,
                    synth_status_t &        out_status) {
    synth_diagnostic_sink_t sink;
    synth_diagnostic_sink_init(&sink, sizeof(sink));
    sink.emit      = record_diagnostic;
    sink.user_data = &diagnostic;

    synth_request_t request;
    synth_request_init(&request, sizeof(request));
    request.input_kind    = SYNTH_INPUT_TEXT_UTF8;
    request.input_data    = kText;
    request.input_count   = std::strlen(kText);
    request.voice_profile = profile;
    request.diagnostics   = &sink;

    synth_audio_buffer_t * audio = nullptr;
    synth_result_t         result;
    synth_result_init(&result, sizeof(result));
    out_status = synth_synthesize_to_buffer(context, &request, &audio, &result);
    if (out_status == SYNTH_OK) {
        SYNTH_TEST_CHECK(audio != nullptr && audio->frame_count > 0);
    } else {
        SYNTH_TEST_CHECK(audio == nullptr);
    }
    synth_audio_buffer_free(audio);
    return 0;
}

// Serializes `profile` and loads the bytes straight back. Returns the envelope
// size so the caller can compare two kinds.
int round_trip(synth_model_t *          model,
               synth_voice_profile_t *  profile,
               uint64_t &               out_size,
               synth_voice_profile_t *& out_reloaded) {
    synth_voice_profile_serialize_params_t serialize_params;
    synth_voice_profile_serialize_params_init(&serialize_params, sizeof(serialize_params));
    synth_byte_buffer_t * bytes = nullptr;
    SYNTH_TEST_CHECK(synth_voice_profile_serialize(profile, &serialize_params, &bytes) == SYNTH_OK);
    SYNTH_TEST_CHECK(bytes != nullptr && bytes->data != nullptr && bytes->data_size > 0);
    out_size = bytes->data_size;

    synth_voice_profile_load_params_t load_params;
    synth_voice_profile_load_params_init(&load_params, sizeof(load_params));
    load_params.data      = bytes->data;
    load_params.data_size = bytes->data_size;

    out_reloaded = nullptr;
    SYNTH_TEST_CHECK(synth_voice_profile_load_from_memory(model, &load_params, &out_reloaded) == SYNTH_OK);
    SYNTH_TEST_CHECK(out_reloaded != nullptr);
    synth_byte_buffer_free(bytes);
    return 0;
}

// THE MODE SELECTOR, at the seam that publishes it (Task 10). This is the
// assertion Task 8's unit test could not make: selecting the mode needs a live
// Model, because the transcript has to be tokenized against the BPE tables
// that only a loaded package carries. The SAME
// synth_voice_profile_create_from_reference call runs twice, differing ONLY in
// whether `reference.transcript` is set --
//
//     transcript absent  -> the x-vector mode Plan 2 shipped
//     transcript present -> the transcript-assisted (ICL) mode Plan 3 adds
//
// -- and BOTH succeed, which is the whole of the flip: a transcript was
// SYNTH_ERR_INVALID_ARG ("voice_profile.transcript_unsupported") for the whole
// of Plan 2, and reporting reference_transcript OPTIONAL while still refusing
// one would be the same capability lie in the opposite direction.
//
// A Profile handle is opaque, so neither `kind` is directly readable here.
// Both are observed through what the runtime does with the payload, in two
// independent ways:
//
//   * SERIALIZE. src/voice-profile.cpp's serialize dispatch reads the payload
//     back through an `XVectorProfile *` and picks its writer from the
//     CloneMode it finds there, and BOTH writers refuse a payload whose mode
//     names the other kind (Task 9), so a wrong branch is an error and never a
//     silent downgrade. A serialize that SUCCEEDS on the transcript-carrying
//     Profile is therefore that dispatch's ICL arm running -- the one line in
//     that file no `unit` test can reach, since a `unit` test may not load
//     this family's 2.5 GB package and nothing smaller can build an ICL
//     payload. The ICL envelope is also strictly the larger of the two: it
//     carries the [16, T] reference codes and the reference text ids on top of
//     everything the x-vector envelope carries.
//   * SYNTHESIZE. src/synthesize.cpp refuses a payload whose CloneMode is not
//     XVector, by message. Plan 2 wrote that guard against a mode that could
//     not yet exist and its own comment says so ("unreachable today"); this is
//     the first test anywhere to reach it. The message is what identifies it:
//     the ABI defines exactly one voice-error status and this file already
//     asserts `synthesis.voice_unsupported` for two OTHER refusals, so the
//     code alone would not tell the three apart.
//
// Task 11 turns that refusal into the ICL prompt and will rewrite the second
// half of this check with it; until it does, the refusal is what the runtime
// honestly reports.
int check_transcript_selects_icl_mode(synth_model_t *                            model,
                                      synth_context_t *                          context,
                                      const synth_voice_profile_capabilities_t & capabilities) {
    SeenDiagnostic          x_vector_diagnostic;
    synth_voice_profile_t * x_vector_profile = nullptr;
    SYNTH_TEST_CHECK(create_with(model, capabilities, nullptr, nullptr, x_vector_diagnostic, x_vector_profile) ==
                     SYNTH_OK);
    SYNTH_TEST_CHECK(x_vector_profile != nullptr);

    SeenDiagnostic          icl_diagnostic;
    synth_voice_profile_t * icl_profile = nullptr;
    SYNTH_TEST_CHECK(create_with(model, capabilities, "hello there", nullptr, icl_diagnostic, icl_profile) == SYNTH_OK);
    SYNTH_TEST_CHECK(icl_profile != nullptr);
    // Nothing was refused on the way: the transcript selected a mode rather
    // than tripping a diagnostic.
    SYNTH_TEST_CHECK(!icl_diagnostic.seen);

    uint64_t                x_vector_size     = 0;
    uint64_t                icl_size          = 0;
    synth_voice_profile_t * x_vector_reloaded = nullptr;
    synth_voice_profile_t * icl_reloaded      = nullptr;
    SYNTH_TEST_CHECK(round_trip(model, x_vector_profile, x_vector_size, x_vector_reloaded) == 0);
    SYNTH_TEST_CHECK(round_trip(model, icl_profile, icl_size, icl_reloaded) == 0);
    SYNTH_TEST_CHECK(icl_size > x_vector_size);

    // The contrast, on the four Profiles: the two x-vector ones synthesize,
    // the two ICL ones are refused BY MODE. The reloaded pair is what proves
    // the envelope carried the kind across the round trip rather than the
    // reader defaulting to one of them.
    for (synth_voice_profile_t * profile : { x_vector_profile, x_vector_reloaded }) {
        SeenDiagnostic diagnostic;
        synth_status_t status = SYNTH_OK;
        SYNTH_TEST_CHECK(synthesize_with(context, profile, diagnostic, status) == 0);
        SYNTH_TEST_CHECK(status == SYNTH_OK);
    }
    for (synth_voice_profile_t * profile : { icl_profile, icl_reloaded }) {
        SeenDiagnostic diagnostic;
        synth_status_t status = SYNTH_OK;
        SYNTH_TEST_CHECK(synthesize_with(context, profile, diagnostic, status) == 0);
        SYNTH_TEST_CHECK(status == SYNTH_ERR_UNSUPPORTED_VOICE);
        SYNTH_TEST_CHECK(diagnostic.seen);
        SYNTH_TEST_CHECK(diagnostic.status == SYNTH_ERR_UNSUPPORTED_VOICE);
        SYNTH_TEST_CHECK(diagnostic.code == "synthesis.voice_unsupported");
        SYNTH_TEST_CHECK(diagnostic.message == "this build supports x-vector Voice Profiles only");
    }

    synth_voice_profile_free(icl_reloaded);
    synth_voice_profile_free(x_vector_reloaded);
    synth_voice_profile_free(icl_profile);
    synth_voice_profile_free(x_vector_profile);
    return 0;
}

// A blank transcript names the transcript-assisted mode and then supplies
// nothing to assist with, so it is SYNTH_ERR_INVALID_ARG rather than a quiet
// fallback to the x-vector mode the caller did not ask for (the design's
// section 9 error table). "Blank" is one state covering empty and
// whitespace-only alike; a NULL transcript is the absent case the check above
// proves selects x-vector, and is not blank.
//
// The zero-size-but-non-null descriptor is the third shape and has its own
// refusal: the paired-null rule refuses it before the mode is ever selected.
//
// THE CODE IS ASSERTED, NOT JUST THE STATUS, AND THAT IS THE WHOLE POINT.
// Three checks apply the same blankness predicate on this path (the dispatch
// in src/voice-profile.cpp, bpe.cpp's qwen_reference_transcript_ids, and
// create_icl_profile), so deleting the first leaves the status at
// SYNTH_ERR_INVALID_ARG and changes only which diagnostic is named -- measured
// by making that deletion. A status-only assertion here would have been the
// fourteenth check in this repository unable to fail.
int check_blank_transcript_refused(synth_model_t * model, const synth_voice_profile_capabilities_t & capabilities) {
    for (const char * transcript : { " ", "\t\n  ", "\r\n" }) {
        SeenDiagnostic          diagnostic;
        synth_voice_profile_t * profile = nullptr;
        SYNTH_TEST_CHECK(create_with(model, capabilities, transcript, nullptr, diagnostic, profile) ==
                         SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(profile == nullptr);
        SYNTH_TEST_CHECK(diagnostic.seen);
        SYNTH_TEST_CHECK(diagnostic.status == SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(diagnostic.code == "voice_profile.transcript_blank");
    }
    return 0;
}

// reference_language is SYNTH_REQUIREMENT_OPTIONAL now, which means VALIDATED,
// not accepted. Plan 2 refused every tag identically; replacing that blanket
// refusal with a blanket acceptance would reintroduce the exact defect the
// refusal closed (a reviewer measured `language_tag="zz-ZZ"` returning
// SYNTH_OK and being written verbatim into the serialized envelope). So the
// two halves OmniVoice's own handler applies apply here, and they refuse
// DIFFERENT things with DIFFERENT statuses:
//
//   * a tag that is not BCP-47-SHAPED     -> SYNTH_ERR_INVALID_ARG
//   * a well-formed tag this package does
//     not DECLARE                          -> SYNTH_ERR_UNSUPPORTED_LANGUAGE
//
// WHAT THIS SEAM DECLARES IS THE BCP-47 TAG, NOT THE PACKAGE'S OWN NAME FOR
// THE LANGUAGE -- and the two really are different strings here, which is the
// trap worth naming. The package's `general.languages` and
// `synthesize.qwen3-tts.languages.names` hold FULL ENGLISH NAMES (`chinese
// english french ...`), and so does the Golden Manifest's `language_tags`
// (which also lists `auto`), because both describe the upstream oracle's
// vocabulary rather than this library's. `src/arch/qwen3-tts/model.cpp`
// bridges them -- "the public interface speaks BCP-47 and the package names
// its languages in full" -- and `declared_language` validates against the
// PUBLIC side of that bridge, the same list `synthesis-request.cpp` validates
// a request's language against. Validating against the package's own names
// instead would make one string legal in a Reference Audio descriptor and
// illegal in the synthesis request it clones for.
//
// So `"en"` is the accepted case and `"english"` is refused as undeclared.
// That is the opposite of what this task's own brief asserted, so the premise
// is ASSERTED below through the public language enumeration rather than
// restated in prose: a comment about which tags a package declares is exactly
// the thing that gets copied forward wrong.
int check_reference_language_validated(synth_model_t * model, const synth_voice_profile_capabilities_t & capabilities) {
    // The premise, measured from the package through the public seam:
    // `en de es zh ja fr ko ru it pt`, and no full name among them.
    bool     declares_english = false;
    bool     declares_en      = false;
    uint64_t language_count   = 0;
    SYNTH_TEST_CHECK(synth_model_get_language_count(model, &language_count) == SYNTH_OK);
    SYNTH_TEST_CHECK(language_count > 0);
    for (uint64_t index = 0; index < language_count; ++index) {
        synth_language_capability_t language;
        synth_language_capability_init(&language, sizeof(language));
        SYNTH_TEST_CHECK(synth_model_get_language(model, index, &language) == SYNTH_OK);
        const std::string tag(language.tag != nullptr ? language.tag : "", size_t(language.tag_size));
        declares_english = declares_english || tag == "english";
        declares_en      = declares_en || tag == "en";
    }
    SYNTH_TEST_CHECK(declares_en);
    SYNTH_TEST_CHECK(!declares_english);

    // Accepted: declared, and it survives the round trip our own reader would
    // otherwise have to refuse.
    {
        SeenDiagnostic          diagnostic;
        synth_voice_profile_t * profile = nullptr;
        SYNTH_TEST_CHECK(create_with(model, capabilities, nullptr, "en", diagnostic, profile) == SYNTH_OK);
        SYNTH_TEST_CHECK(profile != nullptr);
        SYNTH_TEST_CHECK(!diagnostic.seen);

        uint64_t                size     = 0;
        synth_voice_profile_t * reloaded = nullptr;
        SYNTH_TEST_CHECK(round_trip(model, profile, size, reloaded) == 0);
        synth_voice_profile_free(reloaded);
        synth_voice_profile_free(profile);
    }

    // Well-formed, undeclared. "english" is here because it is the package's
    // OWN name for a language it really does carry -- undeclared at this seam
    // all the same; "zz-ZZ" names no language on either side of the bridge.
    for (const char * language_tag : { "english", "zz-ZZ" }) {
        SeenDiagnostic          diagnostic;
        synth_voice_profile_t * profile = nullptr;
        SYNTH_TEST_CHECK(create_with(model, capabilities, nullptr, language_tag, diagnostic, profile) ==
                         SYNTH_ERR_UNSUPPORTED_LANGUAGE);
        SYNTH_TEST_CHECK(profile == nullptr);
    }

    // Malformed shape, two different sub-rules of it: "e" is below the
    // two-character minimum, and "en_US" carries a byte that is neither
    // alphanumeric nor the subtag separator. Both must come back INVALID_ARG
    // and not UNSUPPORTED_LANGUAGE -- which is what makes the shape half and
    // the declared half independently deletable rather than masking each
    // other. Delete the declared half and the pair above starts succeeding;
    // delete the shape half and these two change status instead of passing.
    for (const char * language_tag : { "e", "en_US" }) {
        SeenDiagnostic          diagnostic;
        synth_voice_profile_t * profile = nullptr;
        SYNTH_TEST_CHECK(create_with(model, capabilities, nullptr, language_tag, diagnostic, profile) ==
                         SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(profile == nullptr);
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

// Builds a fresh x-vector Profile from a short, non-silent tone against
// `model`'s own declared reference limits -- the same shape
// check_reference_profile_round_trip already proves is CREATABLE. The two
// tests below are what prove it is USABLE.
int create_test_profile(synth_model_t *                            model,
                        const synth_voice_profile_capabilities_t & capabilities,
                        synth_voice_profile_t *&                   out_profile) {
    const std::vector<float> pcm =
        make_tone(capabilities.min_reference_frames_per_clip, capabilities.reference_target_sample_rate);
    const synth_voice_reference_t reference =
        make_reference(pcm, capabilities.reference_target_sample_rate, capabilities.reference_target_channel_count);
    const synth_voice_reference_params_t params = make_reference_params(&reference, 1, nullptr);

    out_profile = nullptr;
    SYNTH_TEST_CHECK(synth_voice_profile_create_from_reference(model, &params, &out_profile) == SYNTH_OK);
    SYNTH_TEST_CHECK(out_profile != nullptr);
    return 0;
}

// The whole point of Plan 2, at the public seam: a Profile prepared against
// THIS Model synthesizes rather than refusing. `request.voice_id` is left
// unset -- prepare_synthesis_request never resolves a preset voice id for a
// profile-carrying request against this family's empty Preset Voice
// Catalog (src/synthesis-request.cpp) -- so the x-vector is the request's
// only speaker source, exercising model.cpp's `external` prompt-substitution
// path for the first time through the public seam.
int check_profile_synthesizes(synth_context_t * context, synth_voice_profile_t * profile) {
    synth_request_t request;
    synth_request_init(&request, sizeof(request));
    request.input_kind    = SYNTH_INPUT_TEXT_UTF8;
    request.input_data    = kText;
    request.input_count   = std::strlen(kText);
    request.voice_profile = profile;

    synth_audio_buffer_t * audio = nullptr;
    synth_result_t         result;
    synth_result_init(&result, sizeof(result));
    const synth_status_t status = synth_synthesize_to_buffer(context, &request, &audio, &result);
    SYNTH_TEST_CHECK(status == SYNTH_OK);
    SYNTH_TEST_CHECK(audio != nullptr);
    SYNTH_TEST_CHECK(audio->frame_count > 0);
    synth_audio_buffer_free(audio);
    return 0;
}

// A Profile prepared against THIS Model, presented to a synthesis request
// bound to a DIFFERENT Loaded Model's context, is refused -- and the
// refusal is a Voice refusal rather than a graph failure: the ABI has one
// voice-error status, so the diagnostic code is the only thing that tells a
// caller a Voice refusal apart from a codec that failed to run (the same
// property check_unnamed_voice_refused above asserts for the Catalog-less
// case). The second Model is the same real Base package loaded a SECOND
// time: the identity a Voice Profile is bound to is the in-memory
// `synth_model_t*` src/voice-profile-handle.h stores, not any property of
// the package's bytes on disk, so two loads of the identical file already
// give this check two Models to tell apart.
int check_cross_model_profile_refused(const char * model_path, synth_voice_profile_t * profile) {
    synth_model_load_params_t load_params;
    synth_model_load_params_init(&load_params, sizeof(load_params));
    load_params.backend = SYNTH_BACKEND_CPU;

    synth_model_t * other_model = nullptr;
    SYNTH_TEST_CHECK(synth_model_load(model_path, &load_params, &other_model) == SYNTH_OK);
    SYNTH_TEST_CHECK(other_model != nullptr);

    synth_context_t * other_context = nullptr;
    SYNTH_TEST_CHECK(synth_context_create(other_model, &other_context) == SYNTH_OK);
    SYNTH_TEST_CHECK(other_context != nullptr);

    SeenDiagnostic          diagnostic;
    synth_diagnostic_sink_t sink;
    synth_diagnostic_sink_init(&sink, sizeof(sink));
    sink.emit      = record_diagnostic;
    sink.user_data = &diagnostic;

    synth_request_t request;
    synth_request_init(&request, sizeof(request));
    request.input_kind    = SYNTH_INPUT_TEXT_UTF8;
    request.input_data    = kText;
    request.input_count   = std::strlen(kText);
    request.voice_profile = profile;
    request.diagnostics   = &sink;

    synth_audio_buffer_t * audio = nullptr;
    synth_result_t         result;
    synth_result_init(&result, sizeof(result));
    const synth_status_t status = synth_synthesize_to_buffer(other_context, &request, &audio, &result);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_UNSUPPORTED_VOICE);
    SYNTH_TEST_CHECK(audio == nullptr);
    SYNTH_TEST_CHECK(diagnostic.seen);
    SYNTH_TEST_CHECK(diagnostic.status == SYNTH_ERR_UNSUPPORTED_VOICE);
    SYNTH_TEST_CHECK(diagnostic.code == "synthesis.voice_unsupported");

    synth_context_free(other_context);
    synth_model_free(other_model);
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
    SYNTH_TEST_CHECK(check_blank_transcript_refused(model, capabilities) == 0);
    SYNTH_TEST_CHECK(check_reference_language_validated(model, capabilities) == 0);
    SYNTH_TEST_CHECK(check_two_reference_clips_refused(model) == 0);
    SYNTH_TEST_CHECK(check_reference_clip_length_bounds_refused(model, capabilities) == 0);
    SYNTH_TEST_CHECK(check_reference_format_gate_matrix_refused(model, capabilities) == 0);

    synth_context_t * context = nullptr;
    SYNTH_TEST_CHECK(synth_context_create(model, &context) == SYNTH_OK);
    SYNTH_TEST_CHECK(context != nullptr);

    SYNTH_TEST_CHECK(check_unnamed_voice_refused(context) == 0);
    SYNTH_TEST_CHECK(check_named_voice_refused(context) == 0);
    // Needs the context as well as the Model: the clone mode a Profile fixes
    // is only observable through what the runtime does with the payload.
    SYNTH_TEST_CHECK(check_transcript_selects_icl_mode(model, context, capabilities) == 0);

    synth_voice_profile_t * clone_profile = nullptr;
    SYNTH_TEST_CHECK(create_test_profile(model, capabilities, clone_profile) == 0);
    SYNTH_TEST_CHECK(check_profile_synthesizes(context, clone_profile) == 0);
    SYNTH_TEST_CHECK(check_cross_model_profile_refused(model_path, clone_profile) == 0);
    synth_voice_profile_free(clone_profile);

    synth_context_free(context);
    synth_model_free(model);
    return 0;
}
