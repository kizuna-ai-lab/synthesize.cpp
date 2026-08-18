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
#include "gguf.h"
#include "model-handle.h"
#include "sha256.h"
#include "synthesize.h"
#include "test-assert.h"
#include "voice-profile-handle.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <vector>

using synth::qwen3tts::CloneMode;
using synth::qwen3tts::create_design_profile;
using synth::qwen3tts::DesignInstruct;
using synth::qwen3tts::HParams;
using synth::qwen3tts::kMaxDesignInstructBytes;
using synth::qwen3tts::kPrescanDesignKnownKeyCount;
using synth::qwen3tts::kPrescanDesignKnownKeys;
using synth::qwen3tts::kPrescanKvCountDesign;
using synth::qwen3tts::load_profile_from_memory;
using synth::qwen3tts::serialize_design_profile;
using synth::qwen3tts::serialize_x_vector_profile;
using synth::qwen3tts::XVectorProfile;

namespace {

HParams voice_design_hparams() {
    HParams hparams;
    hparams.model_variant   = "qwen3-tts-12hz-1-7b-voicedesign";
    hparams.profile_sources = SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT;
    return hparams;
}

// Three hand-built `synth_model` fixtures, one per variant, each shaped like
// the RUNTIME's published capability snapshot (`model->info.voice_profile.
// source_flags`) that a real Loaded Model of that variant would carry --
// fill_voice_profile_capability's own actual output, pinned independently by
// tests/qwen3_tts_voice_required_test.cpp's
// test_base_capability_publishes_both_sources_together and
// test_capability_follows_the_declared_sources. `qwen3_tts` is left null on
// all three: since Task 5 moved create_qwen3_tts_profile_from_description's
// gate onto this published bit, that function never dereferences
// `model->qwen3_tts` (see its own header comment, voice-profile.cpp), so the
// three tests below can exercise the PUBLIC SEAM
// (synth_voice_profile_create_from_description) without a live Model at
// all -- the same hand-built-handle pattern
// tests/qwen3_tts_voice_required_test.cpp's own cross-family guard already
// uses for create_from_reference/load_from_memory
// (test_customvoice_model_refuses_public_profile_calls), rather than the
// `friend`-gated live-Model factory this file used before Task 5 removed it.

synth_model voice_design_model() {
    synth_model model;
    model.info.family = synth::ModelFamily::Qwen3Tts;
    model.info.voice_profile.source_flags =
        SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT | SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE;
    return model;
}

// Base-shaped: REFERENCE_AUDIO | SERIALIZED_PROFILE, no DESCRIPTION_TEXT --
// what create_qwen3_tts_profile_from_description must still refuse.
synth_model base_model() {
    synth_model model;
    model.info.family = synth::ModelFamily::Qwen3Tts;
    model.info.voice_profile.source_flags =
        SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO | SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE;
    return model;
}

// CustomVoice-shaped: source_flags == 0, VoiceProfileInfo's own all-zero "no
// runtime Voice Profile support" default -- left untouched below.
synth_model customvoice_model() {
    synth_model model;
    model.info.family = synth::ModelFamily::Qwen3Tts;
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
// through the hand-built `synth_model` fixtures above -- Task 5 moved this
// arm's gate onto the published capability bit, so a hand-built handle with
// `qwen3_tts` left null is now enough here too, the same as the
// reference/load-from-memory guard tests/qwen3_tts_voice_required_test.cpp
// already carries.
// =============================================================================

// The public seam. A Model whose package declares description-text gets a
// Profile; the other two variants get the unsupported-source refusal, which is
// the same answer they gave before this arm existed and must keep giving.
int test_create_from_description_accepts_a_voicedesign_model() {
    synth_model                            model       = voice_design_model();
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

    // synth_model holds unique_ptr members, so it is move-only -- a vector of
    // moved-in fixtures stands in for the braced-initializer-list loop the
    // (copyable) HParams version used before Task 5.
    std::vector<synth_model> models;
    models.push_back(base_model());
    models.push_back(customvoice_model());
    for (synth_model & model : models) {
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

// Design section 6.4's FIRST refusal, pinned directly: a VoiceDesign package
// refuses create_from_reference, because it has no speaker encoder.
//
// It was already true, but only by COMPOSITION of two other facts held in two
// other files -- that create_from_reference is gated on the published
// SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO bit (tests/qwen3_tts_voice_required_test.
// cpp's test_customvoice_model_refuses_public_profile_calls, which proves the
// gate with an all-zero capability snapshot), and that a VoiceDesign package
// publishes DESCRIPTION_TEXT without REFERENCE_AUDIO (the same file's
// test_capability_follows_the_declared_sources). Neither one names this
// refusal, and a regression that made the reference gate accept any nonzero
// source_flags would leave both of them green while opening exactly the hole
// 6.4's first bullet is about. The final whole-branch review of Stage 3 Plan 2
// asked for it here, where voice_design_model() -- the same fixture the
// create_from_description tests above use, so the two directions are asserted
// against ONE description of what a VoiceDesign Model looks like -- already
// exists.
//
// Zero references, deliberately: the package-support gate in src/voice-profile.
// cpp runs before reference_count is ever consulted, so an empty reference
// array reaches the same refusal a real recording would, and the assertion
// cannot be satisfied by an unrelated "at least one reference" complaint.
// scripts/validate-qwen3-tts-public.py's own `probe:reference` check against a
// real loaded package is built on the identical observation.
int test_create_from_reference_refuses_a_voicedesign_model() {
    synth_model                    model = voice_design_model();
    synth_voice_reference_params_t params;
    synth_voice_reference_params_init(&params, sizeof(params));

    // A non-null sentinel, for the reason the clone-variant test above states.
    synth_voice_profile_t * profile = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
    SYNTH_TEST_CHECK(synth_voice_profile_create_from_reference(&model, &params, &profile) ==
                     SYNTH_ERR_UNSUPPORTED_VOICE);
    SYNTH_TEST_CHECK(profile == nullptr);
    return 0;
}

// Invalid UTF-8 through the public entry point returns SYNTH_ERR_INVALID_ARG
// rather than reaching the payload -- this pins that Task 1's validation is
// actually wired into the dispatch and not bypassed. Uses a VoiceDesign Model:
// a Base or CustomVoice Model would refuse this same description with
// SYNTH_ERR_UNSUPPORTED_VOICE regardless of its validity, which would prove
// nothing about the UTF-8 check specifically.
int test_create_from_description_refuses_a_malformed_description() {
    synth_model                            model  = voice_design_model();
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
        synth_model             model   = voice_design_model();
        synth_voice_profile_t * profile = nullptr;
        SYNTH_TEST_CHECK(synth_voice_profile_create_from_description(&model, &params, &profile) == SYNTH_OK);
        SYNTH_TEST_CHECK(profile != nullptr);
        SYNTH_TEST_CHECK(profile->family_tag == synth::ProfileFamilyTag::Qwen3TtsDesign);
        synth_voice_profile_free(profile);
    }
    // What description == nullptr still refuses: a null pointer CLAIMING a
    // nonzero size, which has nothing to read. Not a spelling of "empty" --
    // a caller error.
    synth_model                      model = voice_design_model();
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
    synth_model                      model       = voice_design_model();
    const std::string                description = "a description, otherwise unremarkable";
    synth_voice_description_params_t params      = description_params(description);
    params.seed                                  = SYNTH_SEED_RANDOM;

    synth_voice_profile_t * profile = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
    SYNTH_TEST_CHECK(synth_voice_profile_create_from_description(&model, &params, &profile) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(profile == nullptr);
    return 0;
}

// Round-1 review finding (I3): description_language is
// SYNTH_REQUIREMENT_UNSUPPORTED for this family (this file's own
// voice_design_model() fixture doesn't need to say so -- the dispatch itself
// never validates language_tag against any declared vocabulary, unlike
// OmniVoice's own arm -- but tests/qwen3_tts_voice_required_test.cpp's
// test_capability_follows_the_declared_sources pins the published capability
// value this refusal is required by, docs/c-interface.md:478). A
// WELL-FORMED tag must still be refused, not merely a malformed one --
// test_create_from_description_refuses_a_malformed_description above only
// proves the shape check; this proves the support check runs after it.
int test_create_from_description_refuses_a_language_tag() {
    synth_model                      model       = voice_design_model();
    const std::string                description = "a description, otherwise unremarkable";
    const std::string                language    = "ja";
    synth_voice_description_params_t params      = description_params(description);
    params.language_tag                          = language.data();
    params.language_tag_size                     = language.size();

    synth_voice_profile_t * profile = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
    SYNTH_TEST_CHECK(synth_voice_profile_create_from_description(&model, &params, &profile) ==
                     SYNTH_ERR_UNSUPPORTED_INPUT);
    SYNTH_TEST_CHECK(profile == nullptr);
    return 0;
}

// =============================================================================
// Task 5: Republish the capability bit -- the PUBLIC-SEAM serialize/load
// round trip, closing the asymmetry Task 5's own brief named: before this
// task, synth_voice_profile_serialize on a design Profile already succeeded
// (it gates on the PROFILE's own family_tag, set by
// create_qwen3_tts_profile_from_description on every successful call, not on
// the model's published bit) while synth_voice_profile_load_from_memory
// refused with SYNTH_ERR_UNSUPPORTED_VOICE (its own dispatch gates on
// model->info.voice_profile.source_flags's SERIALIZED_PROFILE bit, which
// fill_voice_profile_capability published as 0 for a VoiceDesign package
// until this task). The seam could produce an envelope it could not consume,
// and nothing exercised either half AT THE SEAM: Task 3's own round-trip
// tests above (test_design_profile_round_trips and friends) drive
// load_profile_from_memory directly, one level below synth_voice_profile_
// load_from_memory's own dispatch. This test is the level above.
// =============================================================================

int test_serialize_and_load_round_trip_through_the_public_seam() {
    synth_model                            model         = voice_design_model();
    const std::string                      description   = "A warm, low voice, unhurried, with a slight rasp.";
    const synth_voice_description_params_t create_params = description_params(description);

    synth_voice_profile_t * created = nullptr;
    SYNTH_TEST_CHECK(synth_voice_profile_create_from_description(&model, &create_params, &created) == SYNTH_OK);
    SYNTH_TEST_CHECK(created != nullptr);

    synth_voice_profile_serialize_params_t serialize_params;
    synth_voice_profile_serialize_params_init(&serialize_params, sizeof(serialize_params));
    synth_byte_buffer_t * bytes = nullptr;
    // Already worked before this task (family_tag-gated) -- pinned here as
    // the round trip's first half, not as new coverage of serialize itself.
    SYNTH_TEST_CHECK(synth_voice_profile_serialize(created, &serialize_params, &bytes) == SYNTH_OK);
    SYNTH_TEST_CHECK(bytes != nullptr);
    SYNTH_TEST_CHECK(bytes->data_size > 0);
    synth_voice_profile_free(created);

    synth_voice_profile_load_params_t load_params;
    synth_voice_profile_load_params_init(&load_params, sizeof(load_params));
    load_params.data      = bytes->data;
    load_params.data_size = bytes->data_size;

    synth_voice_profile_t * loaded = nullptr;
    // THIS is what Task 5 fixes: before it, this call returned
    // SYNTH_ERR_UNSUPPORTED_VOICE regardless of the bytes' own validity,
    // because synth_voice_profile_load_from_memory's dispatch never reached
    // load_qwen3_tts_profile_from_memory at all for a VoiceDesign model.
    SYNTH_TEST_CHECK(synth_voice_profile_load_from_memory(&model, &load_params, &loaded) == SYNTH_OK);
    SYNTH_TEST_CHECK(loaded != nullptr);
    SYNTH_TEST_CHECK(loaded->family_tag == synth::ProfileFamilyTag::Qwen3TtsDesign);
    SYNTH_TEST_CHECK(loaded->model == &model);

    const auto * reloaded = static_cast<const DesignInstruct *>(loaded->payload.get());
    SYNTH_TEST_CHECK(reloaded->instruct == description);

    synth_byte_buffer_free(bytes);
    synth_voice_profile_free(loaded);
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

// ---------------------------------------------------------------------------
// Reviewer follow-up (fix round 1): low-level raw-byte helpers for the
// shared-header tamper matrix and the two design-only load-side refusals
// below. Written out here rather than shared with
// tests/qwen3_tts_profile_test.cpp's own file-local copies of the same
// shapes (`find_string_value`, `find_u32_value`, `find_u8_32_value`,
// `reseal`, `emitted_keys_of`, `put`/`put_bytes`/`put_gguf_string`) -- this
// project's own house convention for test-file-local utilities, the same
// reason profile.cpp's own envelope framing is duplicated rather than
// shared across families (see this file's earlier comment on
// prescan_design_buffer).
// ---------------------------------------------------------------------------

void put_bytes(std::vector<uint8_t> & out, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    out.insert(out.end(), bytes, bytes + size);
}

template <typename T> void put(std::vector<uint8_t> & out, T value) {
    put_bytes(out, &value, sizeof(value));
}

void put_gguf_string(std::vector<uint8_t> & out, const std::string & value) {
    put<uint64_t>(out, uint64_t(value.size()));
    put_bytes(out, value.data(), value.size());
}

// Locates the byte OFFSET of a STRING KV's own VALUE bytes (after its own
// length prefix), by searching for that entry's on-disk encoding prefix.
bool find_string_value(const std::vector<uint8_t> & bytes,
                       const std::string &          key,
                       size_t &                     out_offset,
                       size_t &                     out_length) {
    std::vector<uint8_t> needle;
    put<uint64_t>(needle, uint64_t(key.size()));
    put_bytes(needle, key.data(), key.size());
    put<int32_t>(needle, int32_t(GGUF_TYPE_STRING));
    const auto found = std::search(bytes.begin(), bytes.end(), needle.begin(), needle.end());
    if (found == bytes.end()) {
        return false;
    }
    size_t offset = size_t(found - bytes.begin()) + needle.size();
    if (offset + sizeof(uint64_t) > bytes.size()) {
        return false;
    }
    uint64_t length = 0;
    std::memcpy(&length, bytes.data() + offset, sizeof(length));
    offset += sizeof(length);
    if (offset + length > bytes.size()) {
        return false;
    }
    out_offset = offset;
    out_length = size_t(length);
    return true;
}

// Locates the byte OFFSET of a UINT32 KV's own VALUE bytes.
bool find_u32_value(const std::vector<uint8_t> & bytes, const std::string & key, size_t & out_offset) {
    std::vector<uint8_t> needle;
    put<uint64_t>(needle, uint64_t(key.size()));
    put_bytes(needle, key.data(), key.size());
    put<int32_t>(needle, int32_t(GGUF_TYPE_UINT32));
    const auto found = std::search(bytes.begin(), bytes.end(), needle.begin(), needle.end());
    if (found == bytes.end()) {
        return false;
    }
    const size_t offset = size_t(found - bytes.begin()) + needle.size();
    if (offset + sizeof(uint32_t) > bytes.size()) {
        return false;
    }
    out_offset = offset;
    return true;
}

// Locates the byte OFFSET of a 32-element uint8 array KV's own VALUE bytes.
bool find_u8_32_value(const std::vector<uint8_t> & bytes, const std::string & key, size_t & out_offset) {
    std::vector<uint8_t> needle;
    put<uint64_t>(needle, uint64_t(key.size()));
    put_bytes(needle, key.data(), key.size());
    put<int32_t>(needle, int32_t(GGUF_TYPE_ARRAY));
    put<int32_t>(needle, int32_t(GGUF_TYPE_UINT8));
    put<uint64_t>(needle, uint64_t(32));
    const auto found = std::search(bytes.begin(), bytes.end(), needle.begin(), needle.end());
    if (found == bytes.end()) {
        return false;
    }
    const size_t offset = size_t(found - bytes.begin()) + needle.size();
    if (offset + 32 > bytes.size()) {
        return false;
    }
    out_offset = offset;
    return true;
}

// Re-computes `content_sha256` over a buffer whose bytes were edited after
// sealing (hand-built or writer-produced) -- the same two-step
// docs/c-interface.md prescribes (hash with that field zeroed, then write
// the digest back), so a value-level tamper reaches the check aimed at it
// instead of stopping at the digest comparison in front of it.
bool reseal(std::vector<uint8_t> & bytes) {
    size_t offset = 0;
    if (!find_u8_32_value(bytes, "synthesize.voice_profile.content_sha256", offset)) {
        return false;
    }
    std::memset(bytes.data() + offset, 0, 32);
    uint8_t digest[32];
    synth::sha256(bytes.data(), bytes.size(), digest);
    std::memcpy(bytes.data() + offset, digest, sizeof(digest));
    return true;
}

// Reads back the metadata key set of an envelope this file's real writers
// produced, through ggml's own parser rather than through profile.cpp's
// hand-rolled prescan walk -- so writer and whitelist are compared across
// two independent decoders. Mirrors
// tests/qwen3_tts_profile_test.cpp's own helper of the same name.
std::set<std::string> emitted_keys_of(const std::vector<uint8_t> & bytes) {
    gguf_init_params init_params{};
    init_params.no_alloc = true;
    init_params.ctx      = nullptr;
    gguf_context * ctx   = gguf_init_from_buffer(bytes.data(), bytes.size(), init_params);
    if (ctx == nullptr) {
        return {};
    }
    std::set<std::string> keys;
    const int64_t         n_kv = gguf_get_n_kv(ctx);
    for (int64_t index = 0; index < n_kv; ++index) {
        keys.insert(gguf_get_key(ctx, index));
    }
    gguf_free(ctx);
    return keys;
}

void put_kv_string(std::vector<uint8_t> & out, const std::string & key, const std::string & value) {
    put_gguf_string(out, key);
    put<int32_t>(out, int32_t(GGUF_TYPE_STRING));
    put_gguf_string(out, value);
}

void put_kv_u32(std::vector<uint8_t> & out, const std::string & key, uint32_t value) {
    put_gguf_string(out, key);
    put<int32_t>(out, int32_t(GGUF_TYPE_UINT32));
    put<uint32_t>(out, value);
}

void put_kv_u8_array32(std::vector<uint8_t> & out, const std::string & key, const uint8_t (&value)[32]) {
    put_gguf_string(out, key);
    put<int32_t>(out, int32_t(GGUF_TYPE_ARRAY));
    put<int32_t>(out, int32_t(GGUF_TYPE_UINT8));
    put<uint64_t>(out, uint64_t(32));
    put_bytes(out, value, 32);
}

// Hand-builds a complete, UNSEALED design envelope carrying an ARBITRARY
// `instruct` value -- bytes serialize_design_profile itself would refuse to
// produce (over kMaxDesignInstructBytes, or not well-formed UTF-8), needed
// to exercise load_profile_from_memory's OWN independent re-check of both
// invariants rather than only the writer's. Mirrors
// tests/qwen3_tts_profile_test.cpp's own assemble_hand_built /
// common_kv_bytes technique, at this envelope's own eight-key, zero-tensor
// shape (kPrescanDesignKnownKeys, profile.h). `content_sha256` is written as
// 32 zero bytes -- callers must `reseal()` before loading, the same
// two-step every caller of tests/qwen3_tts_profile_test.cpp's own
// hand-built buffers performs explicitly rather than having it happen
// silently inside the builder.
std::vector<uint8_t> hand_build_design_bytes(const uint8_t (&compatibility_id)[32], const std::string & instruct) {
    std::vector<uint8_t> kv;
    put_kv_string(kv, "general.architecture", "synthprofile");
    put_kv_u32(kv, "synthesize.voice_profile.format_version", 1);
    put_kv_string(kv, "synthesize.voice_profile.model_family", "qwen3-tts");
    put_kv_string(kv, "synthesize.voice_profile.schema", "qwen3-tts-voice-design");
    put_kv_u32(kv, "synthesize.voice_profile.schema_version", 1);
    put_kv_u8_array32(kv, "synthesize.voice_profile.compatibility_id", compatibility_id);
    const uint8_t zero_digest[32] = {};
    put_kv_u8_array32(kv, "synthesize.voice_profile.content_sha256", zero_digest);
    put_kv_string(kv, "synthesize.voice_profile.instruct", instruct);

    std::vector<uint8_t> bytes;
    put_bytes(bytes, GGUF_MAGIC, 4);
    put<uint32_t>(bytes, uint32_t(GGUF_VERSION));
    put<int64_t>(bytes, int64_t(0));  // n_tensors -- a design envelope carries none
    put<int64_t>(bytes, kPrescanKvCountDesign);
    put_bytes(bytes, kv.data(), kv.size());
    while (bytes.size() % GGUF_DEFAULT_ALIGNMENT != 0) {
        bytes.push_back(0);
    }
    return bytes;
}

// ---------------------------------------------------------------------------
// The shared-header tamper matrix (reviewer's own table, fix round 1): the
// seven metadata keys set_common_metadata (clone side) and
// hand_build_design_bytes's own header emit alike, each tampered one at a
// time and checked against the SAME expected status on BOTH envelope kinds.
// This is the guard for the ~40 lines of header-validation logic duplicated
// between load_profile_from_memory's design branch and its clone branch: a
// divergence between the two copies becomes a failing assertion here instead
// of a comment nobody re-checks.
// ---------------------------------------------------------------------------

enum class TamperKind { kString, kU32, kU8Array32 };

struct HeaderTamperCase {
    const char *   key;
    TamperKind     kind;
    synth_status_t expected_status;
};

constexpr HeaderTamperCase kSharedHeaderTamperCases[] = {
    { "general.architecture",                      TamperKind::kString,    SYNTH_ERR_INVALID_ARG       },
    { "synthesize.voice_profile.model_family",     TamperKind::kString,    SYNTH_ERR_UNSUPPORTED_VOICE },
    { "synthesize.voice_profile.schema",           TamperKind::kString,    SYNTH_ERR_UNSUPPORTED_VOICE },
    { "synthesize.voice_profile.format_version",   TamperKind::kU32,       SYNTH_ERR_INVALID_ARG       },
    { "synthesize.voice_profile.schema_version",   TamperKind::kU32,       SYNTH_ERR_UNSUPPORTED_VOICE },
    { "synthesize.voice_profile.compatibility_id", TamperKind::kU8Array32, SYNTH_ERR_UNSUPPORTED_VOICE },
    { "synthesize.voice_profile.content_sha256",   TamperKind::kU8Array32, SYNTH_ERR_INVALID_ARG       },
};

// Corrupts `bytes` in place at the named key's own value bytes -- one byte
// flipped for a STRING or a U8[32] array (never touching the length prefix,
// so the buffer's own structure and every OTHER offset stay intact), a
// fixed wrong sentinel for a UINT32 (both this schema's real values are 1,
// so any value other than 1 disagrees). No digest reseal: every one of
// these seven fields is read and compared BEFORE the digest check in both
// load_profile_from_memory branches except content_sha256 itself, which the
// digest check is aimed at directly.
bool apply_header_tamper(std::vector<uint8_t> & bytes, const HeaderTamperCase & tamper_case) {
    switch (tamper_case.kind) {
        case TamperKind::kString:
            {
                size_t offset = 0, length = 0;
                if (!find_string_value(bytes, tamper_case.key, offset, length) || length == 0) {
                    return false;
                }
                bytes[offset] ^= 0xFF;
                return true;
            }
        case TamperKind::kU32:
            {
                size_t offset = 0;
                if (!find_u32_value(bytes, tamper_case.key, offset)) {
                    return false;
                }
                const uint32_t wrong_value = 0xFFFFFFFFu;
                std::memcpy(bytes.data() + offset, &wrong_value, sizeof(wrong_value));
                return true;
            }
        case TamperKind::kU8Array32:
            {
                size_t offset = 0;
                if (!find_u8_32_value(bytes, tamper_case.key, offset)) {
                    return false;
                }
                bytes[offset] ^= 0xFF;
                return true;
            }
    }
    return false;
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

// =============================================================================
// Fix round 1 (reviewer follow-up on Task 3's review): the design branch's
// own copy of the shared header validation had zero committed coverage.
// This is the parameterised guard the reviewer designed for it -- one table,
// run against a REAL design envelope and a REAL clone envelope alike, each
// field's status checked identically on both. It closes two of the
// reviewer's four named refusals (content_sha256 / digest mismatch,
// compatibility_id mismatch) as shared cases; the other two
// (kMaxDesignInstructBytes at load, invalid UTF-8 at load) have no clone-side
// counterpart to parameterise against -- an XVectorProfile carries no
// `instruct` -- and get their own dedicated tests below instead.
//
// This is also the guard that converts "the ~40 duplicated lines are a
// deliberate trade-off" from a comment into something that fails when the
// trade-off stops holding: see the report's own fault-injection evidence for
// this specific test catching a status divergence between the two branches.
// =============================================================================

int test_shared_header_tampers_produce_identical_statuses_for_both_envelope_kinds() {
    uint8_t compatibility_id[32];
    fill_compatibility_id(compatibility_id);

    DesignInstruct design_payload;
    design_payload.instruct = "a description, unremarkable on purpose";
    std::vector<uint8_t> design_base;
    SYNTH_TEST_CHECK(serialize_design_profile(voice_design_hparams(), design_payload, compatibility_id, design_base) ==
                     SYNTH_OK);
    SYNTH_TEST_CHECK(!design_base.empty());

    constexpr uint32_t                    kEncDim       = 11;
    const std::shared_ptr<XVectorProfile> clone_payload = make_x_vector_profile(kEncDim);
    std::vector<uint8_t>                  clone_base;
    SYNTH_TEST_CHECK(serialize_x_vector_profile(*clone_payload, compatibility_id, clone_base) == SYNTH_OK);
    SYNTH_TEST_CHECK(!clone_base.empty());

    for (const HeaderTamperCase & tamper_case : kSharedHeaderTamperCases) {
        std::vector<uint8_t> design_bytes = design_base;
        std::vector<uint8_t> clone_bytes  = clone_base;
        SYNTH_TEST_CHECK(apply_header_tamper(design_bytes, tamper_case));
        SYNTH_TEST_CHECK(apply_header_tamper(clone_bytes, tamper_case));

        synth::ProfileFamilyTag     design_tag = synth::ProfileFamilyTag::None;
        std::shared_ptr<const void> design_loaded;
        const char *                design_code    = nullptr;
        const char *                design_message = nullptr;
        const synth_status_t        design_status =
            load_profile_from_memory(voice_design_hparams(), design_bytes.data(), design_bytes.size(), compatibility_id,
                                     design_tag, design_loaded, design_code, design_message);

        synth::ProfileFamilyTag     clone_tag = synth::ProfileFamilyTag::None;
        std::shared_ptr<const void> clone_loaded;
        const char *                clone_code    = nullptr;
        const char *                clone_message = nullptr;
        const synth_status_t        clone_status =
            load_profile_from_memory(clone_envelope_hparams(kEncDim), clone_bytes.data(), clone_bytes.size(),
                                     compatibility_id, clone_tag, clone_loaded, clone_code, clone_message);

        SYNTH_TEST_CHECK(design_status == tamper_case.expected_status);
        SYNTH_TEST_CHECK(clone_status == tamper_case.expected_status);
        SYNTH_TEST_CHECK(design_loaded == nullptr);
        SYNTH_TEST_CHECK(clone_loaded == nullptr);
    }
    return 0;
}

// The remaining two of the reviewer's four named refusals: neither has a
// clone-side counterpart (an XVectorProfile carries no `instruct`), so each
// gets its own dedicated test against a HAND-BUILT, re-sealed envelope --
// bytes serialize_design_profile itself would refuse to produce, needed to
// reach load_profile_from_memory's OWN independent re-check rather than only
// the writer's (the same "our own writer must never emit what our own reader
// refuses" reasoning kMaxLanguageTagLength's own header comment records for
// the clone path, mirrored here on the read side instead).

int test_design_instruct_over_length_is_refused_at_load() {
    uint8_t compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    const std::string    oversized(size_t(kMaxDesignInstructBytes) + 1, 'a');
    std::vector<uint8_t> bytes = hand_build_design_bytes(compatibility_id, oversized);
    SYNTH_TEST_CHECK(reseal(bytes));

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> loaded;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = load_profile_from_memory(voice_design_hparams(), bytes.data(), bytes.size(),
                                                                   compatibility_id, family_tag, loaded, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(loaded == nullptr);
    return 0;
}

int test_design_instruct_invalid_utf8_is_refused_at_load() {
    uint8_t compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    // A lone continuation byte -- structurally a valid (short) GGUF string,
    // and well within kMaxDesignInstructBytes, so this arm and the length
    // arm above are independent: neither can pass by accident of the other.
    std::vector<uint8_t> bytes = hand_build_design_bytes(compatibility_id, "\x80");
    SYNTH_TEST_CHECK(reseal(bytes));

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> loaded;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = load_profile_from_memory(voice_design_hparams(), bytes.data(), bytes.size(),
                                                                   compatibility_id, family_tag, loaded, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(loaded == nullptr);
    return 0;
}

// =============================================================================
// Unblocked writer-agreement test (reviewer follow-up): kPrescanDesignKnownKeys
// moved from profile.cpp's anonymous namespace to profile.h so this test can
// reach it -- modelled on
// tests/qwen3_tts_profile_test.cpp's own test_writer_emits_exactly_the_whitelisted_keys
// / test_the_icl_writer_emits_exactly_the_whitelisted_keys. A key REMOVED
// from serialize_design_profile while left in kPrescanDesignKnownKeys would
// be a silently too-permissive whitelist that nothing else in this file
// would notice.
// =============================================================================

int test_the_design_writer_emits_exactly_the_whitelisted_keys() {
    uint8_t compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    DesignInstruct payload;
    payload.instruct = "a description, unremarkable on purpose";
    std::vector<uint8_t> bytes;
    SYNTH_TEST_CHECK(serialize_design_profile(voice_design_hparams(), payload, compatibility_id, bytes) == SYNTH_OK);
    SYNTH_TEST_CHECK(!bytes.empty());
    const std::set<std::string> emitted_keys = emitted_keys_of(bytes);

    std::set<std::string> expected_keys;
    for (size_t index = 0; index < kPrescanDesignKnownKeyCount; ++index) {
        expected_keys.insert(kPrescanDesignKnownKeys[index].key);
    }

    SYNTH_TEST_CHECK(emitted_keys == expected_keys);
    SYNTH_TEST_CHECK(int64_t(emitted_keys.size()) == kPrescanKvCountDesign);
    return 0;
}

int main() {
    SYNTH_TEST_CHECK(test_design_profile_holds_the_string_verbatim() == 0);
    SYNTH_TEST_CHECK(test_an_empty_instruct_is_accepted() == 0);
    SYNTH_TEST_CHECK(test_invalid_utf8_is_refused() == 0);
    SYNTH_TEST_CHECK(test_an_over_long_instruct_is_refused_at_the_boundary() == 0);
    SYNTH_TEST_CHECK(test_create_from_description_accepts_a_voicedesign_model() == 0);
    SYNTH_TEST_CHECK(test_create_from_description_refuses_the_clone_variants() == 0);
    SYNTH_TEST_CHECK(test_create_from_reference_refuses_a_voicedesign_model() == 0);
    SYNTH_TEST_CHECK(test_create_from_description_refuses_a_malformed_description() == 0);
    SYNTH_TEST_CHECK(test_create_from_description_accepts_both_empty_description_spellings() == 0);
    SYNTH_TEST_CHECK(test_create_from_description_refuses_a_random_seed() == 0);
    SYNTH_TEST_CHECK(test_create_from_description_refuses_a_language_tag() == 0);
    SYNTH_TEST_CHECK(test_serialize_and_load_round_trip_through_the_public_seam() == 0);
    SYNTH_TEST_CHECK(test_design_profile_round_trips() == 0);
    SYNTH_TEST_CHECK(test_the_two_envelope_kinds_are_not_confusable() == 0);
    SYNTH_TEST_CHECK(test_a_clone_envelope_with_the_wrong_size_is_still_refused() == 0);
    SYNTH_TEST_CHECK(test_an_empty_instruct_round_trips() == 0);
    SYNTH_TEST_CHECK(test_shared_header_tampers_produce_identical_statuses_for_both_envelope_kinds() == 0);
    SYNTH_TEST_CHECK(test_design_instruct_over_length_is_refused_at_load() == 0);
    SYNTH_TEST_CHECK(test_design_instruct_invalid_utf8_is_refused_at_load() == 0);
    SYNTH_TEST_CHECK(test_the_design_writer_emits_exactly_the_whitelisted_keys() == 0);
    return 0;
}
