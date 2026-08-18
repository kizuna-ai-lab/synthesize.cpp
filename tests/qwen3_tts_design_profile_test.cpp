// The Description Text Voice Profile payload.
//
// It holds the instruct string and nothing else, mirroring
// src/arch/omnivoice/profile.h's struct of the same name -- and deliberately
// NOT its validation. OmniVoice can reject an instruct because upstream defines
// a closed attribute vocabulary; this family's is free-form and upstream accepts
// any string, so a vocabulary invented here would reject input upstream accepts.
// Design D2.

#include "arch/qwen3-tts/profile.h"
#include "arch/qwen3-tts/weights.h"
#include "model-handle.h"
#include "synthesize.h"
#include "test-assert.h"
#include "voice-profile-handle.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using synth::qwen3tts::CloneMode;
using synth::qwen3tts::create_design_profile;
using synth::qwen3tts::DesignInstruct;
using synth::qwen3tts::HParams;
using synth::qwen3tts::kMaxDesignInstructBytes;
using synth::qwen3tts::load_profile_from_memory;
using synth::qwen3tts::serialize_design_profile;
using synth::qwen3tts::serialize_x_vector_profile;
using synth::qwen3tts::XVectorProfile;
using synth::qwen3tts::testing::make_model_for_testing;

namespace {

HParams voice_design_hparams() {
    HParams hparams;
    hparams.model_variant   = "qwen3-tts-12hz-1-7b-voicedesign";
    hparams.profile_sources = SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT;
    return hparams;
}

// A Base-shaped package: ProfileSources mode, REFERENCE_AUDIO declared instead
// of DESCRIPTION_TEXT. Mirrors tests/qwen3_tts_voice_required_test.cpp's own
// base_hparams() at the level this file's dispatch actually reads --
// abbreviated to the one field create_qwen3_tts_profile_from_description
// consults, since the other Reference Audio contract fields that file's own
// fixture carries are irrelevant to a Description Text refusal.
HParams base_hparams() {
    HParams hparams;
    hparams.model_variant   = "qwen3-tts-12hz-1-7b-base";
    hparams.voice_mode      = synth::qwen3tts::VoiceMode::ProfileSources;
    hparams.profile_sources = SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO;
    return hparams;
}

// A CustomVoice-shaped package: a Preset Voice Catalog, no Voice Profile
// contract at all. Mirrors tests/qwen3_tts_voice_required_test.cpp's own
// customvoice_hparams() at the level this file's dispatch reads.
HParams customvoice_hparams() {
    HParams hparams;
    hparams.model_variant = "qwen3-tts-12hz-1-7b-customvoice";
    hparams.voice_mode    = synth::qwen3tts::VoiceMode::PresetCatalog;
    return hparams;
}

// Wraps `hparams` in a synth_model whose family is Qwen3Tts and whose
// qwen3_tts pointer is a REAL (if otherwise empty) Model -- built through the
// narrow `friend` factory testing::make_model_for_testing rather than
// Model::load/load_cpu, because this family has no synthetic-package test
// harness a `unit` test may depend on (that factory's own header comment,
// ahead of `class Model` in qwen3-tts.h, says why, and why it is temporary).
// This is what lets the three tests below exercise the PUBLIC SEAM
// (synth_voice_profile_create_from_description) with a Model whose declared
// sources they control, rather than a hand-built handle with qwen3_tts left
// null the way tests/qwen3_tts_voice_required_test.cpp's cross-family guard
// uses for create_from_reference/load_from_memory -- that trick would only
// prove the defensive null-check inside create_qwen3_tts_profile_from_description,
// not the declared-sources gate these tests are actually aimed at.
synth_model make_model(const HParams & hparams) {
    synth_model model;
    model.info.family = synth::ModelFamily::Qwen3Tts;
    model.qwen3_tts   = make_model_for_testing(hparams);
    return model;
}

synth_voice_description_params_t description_params(const std::string & description) {
    synth_voice_description_params_t params;
    synth_voice_description_params_init(&params, sizeof(params));
    params.description      = description.data();
    params.description_size = description.size();
    return params;
}

}  // namespace

int test_design_profile_holds_the_string_verbatim() {
    const HParams     hparams = voice_design_hparams();
    DesignInstruct    payload;
    const std::string instruct = "A warm, low voice, unhurried, with a slight rasp.";
    SYNTH_TEST_CHECK(create_design_profile(hparams, instruct, payload) == SYNTH_OK);
    // VERBATIM: no canonicalisation, no trimming, no reordering. Upstream
    // tokenizes whatever it is given, so anything done here would be a
    // difference from upstream that no tolerance would show.
    SYNTH_TEST_CHECK(payload.instruct == instruct);
    return 0;
}

int test_an_empty_instruct_is_accepted() {
    // Design D3: upstream's `instruct_ids.append(None)` path. An empty
    // description is a legal request for an unconditioned voice, not an error.
    const HParams  hparams = voice_design_hparams();
    DesignInstruct payload;
    SYNTH_TEST_CHECK(create_design_profile(hparams, "", payload) == SYNTH_OK);
    SYNTH_TEST_CHECK(payload.instruct.empty());
    return 0;
}

int test_invalid_utf8_is_refused() {
    const HParams  hparams = voice_design_hparams();
    DesignInstruct payload;
    // A lone continuation byte, a truncated three-byte sequence, an overlong
    // encoding of NUL, a UTF-16 surrogate, and a code point beyond U+10FFFF --
    // the four rejection classes is_well_formed_utf8's own header comment
    // claims ("over-long encodings, surrogates and out-of-range code points
    // as well as truncated sequences"). A reviewer found the first version of
    // this test covered only two of the four: disabling only the overlong
    // check left the suite 100% green. The implementation itself is correct
    // -- a standalone probe confirmed it rejects all four -- this closes the
    // coverage gap.
    for (const std::string bad : { std::string("\x80"), std::string("\xE2\x82"), std::string("\xC0\x80"),
                                   std::string("\xED\xA0\x80"), std::string("\xF4\x90\x80\x80") }) {
        payload.instruct = "untouched";
        SYNTH_TEST_CHECK(create_design_profile(hparams, bad, payload) == SYNTH_ERR_INVALID_ARG);
        // On refusal the output is left alone rather than half-written.
        SYNTH_TEST_CHECK(payload.instruct == "untouched");
    }
    // And valid multi-byte UTF-8 is NOT refused -- without this the check could
    // pass by rejecting everything non-ASCII, which would reject the Chinese and
    // Japanese this package declares.
    payload.instruct.clear();
    SYNTH_TEST_CHECK(create_design_profile(hparams, "温かく低い声、少しかすれた", payload) == SYNTH_OK);
    SYNTH_TEST_CHECK(!payload.instruct.empty());
    return 0;
}

int test_an_over_long_instruct_is_refused_at_the_boundary() {
    const HParams  hparams = voice_design_hparams();
    DesignInstruct payload;
    // Exactly at the bound is accepted; one byte over is refused. A test that
    // only checked a wildly long string would pass on an off-by-one.
    SYNTH_TEST_CHECK(create_design_profile(hparams, std::string(kMaxDesignInstructBytes, 'a'), payload) == SYNTH_OK);
    SYNTH_TEST_CHECK(create_design_profile(hparams, std::string(kMaxDesignInstructBytes + 1, 'a'), payload) ==
                     SYNTH_ERR_INVALID_ARG);
    return 0;
}

// =============================================================================
// Task 2: synth_voice_profile_create_from_description's Qwen3-TTS arm
// (src/voice-profile.cpp). The tests below use the real public entry point,
// through make_model()'s make_model_for_testing-backed fixture -- see that
// helper's own comment for why a hand-built handle with qwen3_tts left null
// is not enough here, unlike the reference/load-from-memory guard
// tests/qwen3_tts_voice_required_test.cpp already carries.
// =============================================================================

// The public seam. A Model whose package declares description-text gets a
// Profile; the other two variants get the unsupported-source refusal, which is
// the same answer they gave before this arm existed and must keep giving.
int test_create_from_description_accepts_a_voicedesign_model() {
    synth_model                            model       = make_model(voice_design_hparams());
    const std::string                      description = "A warm, low voice, unhurried, with a slight rasp.";
    const synth_voice_description_params_t params      = description_params(description);

    synth_voice_profile_t * profile = nullptr;
    SYNTH_TEST_CHECK(synth_voice_profile_create_from_description(&model, &params, &profile) == SYNTH_OK);
    SYNTH_TEST_CHECK(profile != nullptr);
    SYNTH_TEST_CHECK(profile->family_tag == synth::ProfileFamilyTag::Qwen3TtsDesign);
    // model is filled too, not just family_tag/payload -- create_from_reference's
    // own create_qwen3_tts_profile_from_reference sets it (voice-profile.cpp:701)
    // and this arm must not skip it, a half-filled handle being the trap this
    // task's own brief warned about.
    SYNTH_TEST_CHECK(profile->model == &model);
    synth_voice_profile_free(profile);
    // synth_voice_profile_free is safe on null too -- proves the free path
    // itself does not depend on this handle's particular shape, so a caller
    // freeing twice by mistake (or freeing an already-null profile from a
    // failed call) cannot crash.
    synth_voice_profile_free(nullptr);
    return 0;
}

int test_create_from_description_refuses_the_clone_variants() {
    const std::string                      description = "irrelevant -- refused before this is ever read";
    const synth_voice_description_params_t params      = description_params(description);

    for (const HParams & hparams : { base_hparams(), customvoice_hparams() }) {
        synth_model             model   = make_model(hparams);
        // A non-null sentinel rather than a pre-nulled pointer: the assertion
        // below must prove the dispatch itself wrote null, not merely that it
        // left an already-null pointer alone. Mirrors
        // tests/qwen3_tts_voice_required_test.cpp's own
        // test_customvoice_model_refuses_public_profile_calls.
        synth_voice_profile_t * profile = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
        SYNTH_TEST_CHECK(synth_voice_profile_create_from_description(&model, &params, &profile) ==
                         SYNTH_ERR_UNSUPPORTED_VOICE);
        SYNTH_TEST_CHECK(profile == nullptr);
    }
    return 0;
}

// Invalid UTF-8 through the public entry point returns SYNTH_ERR_INVALID_ARG
// rather than reaching the payload -- this pins that Task 1's validation is
// actually wired into the dispatch and not bypassed. Uses a VoiceDesign Model:
// a Base or CustomVoice Model would refuse this same description with
// SYNTH_ERR_UNSUPPORTED_VOICE regardless of its validity, which would prove
// nothing about the UTF-8 check specifically.
int test_create_from_description_refuses_a_malformed_description() {
    synth_model                            model  = make_model(voice_design_hparams());
    // A lone continuation byte -- the same malformed sequence
    // test_invalid_utf8_is_refused above pins at create_design_profile's own
    // level; here it is pinned again one layer up, through the dispatch.
    const std::string                      bad    = "\x80";
    const synth_voice_description_params_t params = description_params(bad);

    synth_voice_profile_t * profile = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
    SYNTH_TEST_CHECK(synth_voice_profile_create_from_description(&model, &params, &profile) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(profile == nullptr);
    return 0;
}

// Round-1 review finding (Important #1/#2): D3's "an empty instruct is legal"
// must be reachable through BOTH of its natural spellings -- a null
// description pointer with a zero size, and a non-null pointer to a
// zero-length buffer, which is what bindings/python/src/native_loader.c's
// `PyBytes_AsString(b"")` hands this entry point (never null, even for an
// empty bytes object). An earlier version of this arm copied OmniVoice's own
// pairing check verbatim -- correct there, because OmniVoice refuses empty
// regardless of spelling -- which refused the second spelling here with
// SYNTH_ERR_INVALID_ARG and had zero test coverage: the reviewer's injection
// of OmniVoice's exact "description required, non-empty" refusal left this
// target 100% green, because nothing exercised D3 through the public seam at
// all (Task 1's test_an_empty_instruct_is_accepted covers create_design_profile
// directly, not this dispatch). Both spellings are asserted here so that
// injection -- or the narrower pairing-check regression that motivated this
// test -- fails loudly.
int test_create_from_description_accepts_both_empty_description_spellings() {
    synth_voice_description_params_t null_spelling;
    synth_voice_description_params_init(&null_spelling, sizeof(null_spelling));
    null_spelling.description      = nullptr;
    null_spelling.description_size = 0;

    synth_voice_description_params_t nonnull_spelling;
    synth_voice_description_params_init(&nonnull_spelling, sizeof(nonnull_spelling));
    // A string literal, not empty.data(): non-null is the property under
    // test, and a literal makes that true by construction rather than by an
    // implementation guarantee about std::string.
    nonnull_spelling.description      = "";
    nonnull_spelling.description_size = 0;

    for (const synth_voice_description_params_t & params : { null_spelling, nonnull_spelling }) {
        synth_model             model   = make_model(voice_design_hparams());
        synth_voice_profile_t * profile = nullptr;
        SYNTH_TEST_CHECK(synth_voice_profile_create_from_description(&model, &params, &profile) == SYNTH_OK);
        SYNTH_TEST_CHECK(profile != nullptr);
        SYNTH_TEST_CHECK(profile->family_tag == synth::ProfileFamilyTag::Qwen3TtsDesign);
        synth_voice_profile_free(profile);
    }
    // What description == nullptr still refuses: a null pointer CLAIMING a
    // nonzero size, which has nothing to read. Not a spelling of "empty" --
    // a caller error.
    synth_model                      model = make_model(voice_design_hparams());
    synth_voice_description_params_t mismatched;
    synth_voice_description_params_init(&mismatched, sizeof(mismatched));
    mismatched.description          = nullptr;
    mismatched.description_size     = 4;
    synth_voice_profile_t * profile = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
    SYNTH_TEST_CHECK(synth_voice_profile_create_from_description(&model, &mismatched, &profile) ==
                     SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(profile == nullptr);
    return 0;
}

// Round-1 review finding (Important #3): docs/c-interface.md:570's ABI-wide
// "v1 profile preparation rejects SYNTH_SEED_RANDOM" contract, which
// OmniVoice's own create_from_description arm above enforces. This family's
// Description Text preparation has nothing seed-dependent to fix -- unlike
// OmniVoice's own arm, whose comment explains what ITS seed is for -- but
// that is a fact about what the seed is used FOR, not licence to skip an
// ABI-wide contract this file's sibling arm already enforces.
int test_create_from_description_refuses_a_random_seed() {
    synth_model                      model       = make_model(voice_design_hparams());
    const std::string                description = "a description, otherwise unremarkable";
    synth_voice_description_params_t params      = description_params(description);
    params.seed                                  = SYNTH_SEED_RANDOM;

    synth_voice_profile_t * profile = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
    SYNTH_TEST_CHECK(synth_voice_profile_create_from_description(&model, &params, &profile) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(profile == nullptr);
    return 0;
}

// =============================================================================
// Task 3: Serialization, without loosening the x-vector check
// (src/arch/qwen3-tts/profile.cpp). These exercise the FAMILY-LEVEL envelope
// functions directly -- serialize_design_profile and load_profile_from_memory
// -- the same level tests/qwen3_tts_profile_test.cpp exercises for the clone
// envelope, rather than the public C seam: synth_voice_profile_serialize and
// synth_voice_profile_load_from_memory do not yet route a design Profile at
// all (Task 5 republishes the SERIALIZED_PROFILE capability bit those
// dispatchers gate on for a VoiceDesign Model).
// =============================================================================

namespace {

void fill_compatibility_id(uint8_t (&id)[32]) {
    // Arbitrary but non-zero, so a mismatch is never mistaken for an
    // unset/all-zero placeholder on either side of a comparison.
    for (size_t index = 0; index < sizeof(id); ++index) {
        id[index] = uint8_t(index + 7);
    }
}

// The clone (x-vector) side's own package width -- enough for
// load_profile_from_memory's x-vector-only path, which never reads past
// hparams.speaker_encoder.enc_dim before it returns.
HParams clone_envelope_hparams(uint32_t enc_dim) {
    HParams hparams;
    hparams.speaker_encoder.enc_dim = enc_dim;
    return hparams;
}

std::shared_ptr<XVectorProfile> make_x_vector_profile(uint32_t enc_dim) {
    auto profile  = std::make_shared<XVectorProfile>();
    profile->mode = CloneMode::XVector;
    profile->x_vector.resize(enc_dim);
    for (uint32_t index = 0; index < enc_dim; ++index) {
        // Finite and never identically zero -- load_profile_from_memory's own
        // payload-value parity check refuses an all-zero x-vector.
        profile->x_vector[index] = 1.0f + float(index);
    }
    profile->ref_rms      = 0.5f;
    profile->language_tag = "en-US";
    return profile;
}

}  // namespace

// Round-trip: serialize a design Profile, load the bytes back, and get the same
// string with the same tag. Per design D6 the envelope stores the TEXT and the
// loader re-tokenizes -- there are no ids in it to drift.
int test_design_profile_round_trips() {
    uint8_t compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    DesignInstruct payload;
    payload.instruct = "A warm, low voice, unhurried, with a slight rasp.";

    std::vector<uint8_t> bytes;
    SYNTH_TEST_CHECK(serialize_design_profile(voice_design_hparams(), payload, compatibility_id, bytes) == SYNTH_OK);
    SYNTH_TEST_CHECK(!bytes.empty());

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> loaded;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = load_profile_from_memory(voice_design_hparams(), bytes.data(), bytes.size(),
                                                                   compatibility_id, family_tag, loaded, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_OK);
    SYNTH_TEST_CHECK(family_tag == synth::ProfileFamilyTag::Qwen3TtsDesign);
    SYNTH_TEST_CHECK(loaded != nullptr);
    SYNTH_TEST_CHECK(code == nullptr);
    SYNTH_TEST_CHECK(message == nullptr);

    const auto * reloaded = static_cast<const DesignInstruct *>(loaded.get());
    SYNTH_TEST_CHECK(reloaded->instruct == payload.instruct);
    return 0;
}

// A clone envelope must not load as a design Profile and a design envelope must
// not load as a clone. The tag is what a consumer switches on, so a
// misidentified payload is a type confusion, not a wrong answer.
int test_the_two_envelope_kinds_are_not_confusable() {
    uint8_t compatibility_id[32];
    fill_compatibility_id(compatibility_id);

    // A real design envelope loads as Design, never as Clone.
    DesignInstruct design_payload;
    design_payload.instruct = "a description, unremarkable on purpose";
    std::vector<uint8_t> design_bytes;
    SYNTH_TEST_CHECK(serialize_design_profile(voice_design_hparams(), design_payload, compatibility_id, design_bytes) ==
                     SYNTH_OK);

    synth::ProfileFamilyTag     design_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> design_loaded;
    const char *                design_code    = nullptr;
    const char *                design_message = nullptr;
    SYNTH_TEST_CHECK(load_profile_from_memory(voice_design_hparams(), design_bytes.data(), design_bytes.size(),
                                              compatibility_id, design_tag, design_loaded, design_code,
                                              design_message) == SYNTH_OK);
    SYNTH_TEST_CHECK(design_tag == synth::ProfileFamilyTag::Qwen3TtsDesign);

    // A real clone (x-vector) envelope loads as Clone, never as Design.
    constexpr uint32_t                    kEncDim       = 11;
    const std::shared_ptr<XVectorProfile> clone_payload = make_x_vector_profile(kEncDim);
    std::vector<uint8_t>                  clone_bytes;
    SYNTH_TEST_CHECK(serialize_x_vector_profile(*clone_payload, compatibility_id, clone_bytes) == SYNTH_OK);

    synth::ProfileFamilyTag     clone_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> clone_loaded;
    const char *                clone_code    = nullptr;
    const char *                clone_message = nullptr;
    SYNTH_TEST_CHECK(load_profile_from_memory(clone_envelope_hparams(kEncDim), clone_bytes.data(), clone_bytes.size(),
                                              compatibility_id, clone_tag, clone_loaded, clone_code,
                                              clone_message) == SYNTH_OK);
    SYNTH_TEST_CHECK(clone_tag == synth::ProfileFamilyTag::Qwen3TtsClone);
    return 0;
}

// The x-vector size check is UNCHANGED: a clone envelope whose element count
// disagrees with enc_dim is still refused. Pin it here because this task edits
// the function that performs it.
int test_a_clone_envelope_with_the_wrong_size_is_still_refused() {
    uint8_t compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    constexpr uint32_t kDeclaredEncDim = 11;

    // Both directions: an envelope with FEWER elements than the package
    // declares, and one with MORE. The second is the direction that matters
    // for THIS task specifically -- weakening `element_count == enc_dim` to
    // `element_count >= enc_dim` (Step 4's own fault injection) would let a
    // too-LARGE envelope straight through while still refusing a too-small
    // one, so a test that only tried the "too small" direction could not see
    // that exact mutation. Pinning both is what makes the injection below
    // meaningful rather than accidental.
    for (uint32_t envelope_enc_dim : { kDeclaredEncDim - 1, kDeclaredEncDim + 1 }) {
        const std::shared_ptr<XVectorProfile> profile = make_x_vector_profile(envelope_enc_dim);
        std::vector<uint8_t>                  bytes;
        SYNTH_TEST_CHECK(serialize_x_vector_profile(*profile, compatibility_id, bytes) == SYNTH_OK);

        synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
        std::shared_ptr<const void> loaded;
        const char *                code    = nullptr;
        const char *                message = nullptr;
        const synth_status_t        status =
            load_profile_from_memory(clone_envelope_hparams(kDeclaredEncDim), bytes.data(), bytes.size(),
                                     compatibility_id, family_tag, loaded, code, message);
        SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(loaded == nullptr);
    }
    return 0;
}

// An empty instruct round-trips as an empty instruct, not as an absent field.
int test_an_empty_instruct_round_trips() {
    uint8_t compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    DesignInstruct payload;
    SYNTH_TEST_CHECK(payload.instruct.empty());

    std::vector<uint8_t> bytes;
    SYNTH_TEST_CHECK(serialize_design_profile(voice_design_hparams(), payload, compatibility_id, bytes) == SYNTH_OK);

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> loaded;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    SYNTH_TEST_CHECK(load_profile_from_memory(voice_design_hparams(), bytes.data(), bytes.size(), compatibility_id,
                                              family_tag, loaded, code, message) == SYNTH_OK);
    SYNTH_TEST_CHECK(family_tag == synth::ProfileFamilyTag::Qwen3TtsDesign);
    const auto * reloaded = static_cast<const DesignInstruct *>(loaded.get());
    SYNTH_TEST_CHECK(reloaded->instruct.empty());
    return 0;
}

int main() {
    SYNTH_TEST_CHECK(test_design_profile_holds_the_string_verbatim() == 0);
    SYNTH_TEST_CHECK(test_an_empty_instruct_is_accepted() == 0);
    SYNTH_TEST_CHECK(test_invalid_utf8_is_refused() == 0);
    SYNTH_TEST_CHECK(test_an_over_long_instruct_is_refused_at_the_boundary() == 0);
    SYNTH_TEST_CHECK(test_create_from_description_accepts_a_voicedesign_model() == 0);
    SYNTH_TEST_CHECK(test_create_from_description_refuses_the_clone_variants() == 0);
    SYNTH_TEST_CHECK(test_create_from_description_refuses_a_malformed_description() == 0);
    SYNTH_TEST_CHECK(test_create_from_description_accepts_both_empty_description_spellings() == 0);
    SYNTH_TEST_CHECK(test_create_from_description_refuses_a_random_seed() == 0);
    SYNTH_TEST_CHECK(test_design_profile_round_trips() == 0);
    SYNTH_TEST_CHECK(test_the_two_envelope_kinds_are_not_confusable() == 0);
    SYNTH_TEST_CHECK(test_a_clone_envelope_with_the_wrong_size_is_still_refused() == 0);
    SYNTH_TEST_CHECK(test_an_empty_instruct_round_trips() == 0);
    return 0;
}
