// The Base package is the project's first that carries no Preset Voice
// Catalog at all: every request must supply a prepared Voice Profile instead
// (Plan 2). This exercises the two family-layer rules that make that honest
// rather than a silent guess:
//
//   1. weights.h's has_preset_voice_catalog, the declared-mode discriminator,
//      and the empty catalog it implies: read_voices refuses a
//      profile-sources package that declares any preset at all and clears the
//      vector, so Model::resolve_voice's find_preset_voice (model.cpp)
//      refuses with SYNTH_ERR_UNSUPPORTED_VOICE whether a request names no
//      voice_id or names one that could never exist. Both collapse to the
//      same public status; what would distinguish "no voice_id" from
//      "unknown voice_id" is a diagnostic concern, not a status code.
//
//   2. weights.h's fill_voice_profile_capability, which Model::get_info
//      (model.cpp) uses to build the capability snapshot a caller reads
//      before ever trying to synthesize. As of Stage 2 Plan 2, Base
//      (ProfileSources) publishes SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO |
//      SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE with the six reference limits;
//      CustomVoice (PresetCatalog) still reports nothing, because it has no
//      speaker encoder and nothing in the runtime can prepare anything from
//      it. The gate is `voice_mode`, not `has_preset_voice_catalog` -- see
//      that predicate's own header comment and
//      test_base_capability_publishes_both_sources_together below for why
//      the inverse would have been a real bug, caught here.
//
// A `unit`-labelled test cannot depend on the real ~2.5 GB Base package
// (docs/testing.md), so both rules are exercised here directly against
// synthetic HParams, the way qwen3_tts_metadata_test.cpp and
// qwen3_tts_catalog_test.cpp already do -- no GGUF file and no loaded Model
// involved. Proving the real on-disk metadata reaches the public struct is
// tests/qwen3_tts_base_load_real.cpp's job instead (see that file's own
// header comment for why a synthetic fixture cannot substitute for it); the
// public-seam dispatch gate that keeps a CustomVoice Loaded Model out of this
// family's new create/load handlers is exercised here too
// (test_customvoice_model_refuses_public_profile_calls below), since it needs
// no GGUF either -- only a hand-built `synth_model` with its family pointer
// left null, the same pattern tests/omnivoice_serialize_test.cpp's own
// cross-family guard arm already uses.

#include "arch/qwen3-tts/weights.h"
#include "model-handle.h"
#include "synthesize.h"
#include "test-assert.h"

#include <cstdint>
#include <string>

namespace {

// A CustomVoice-shaped package: a Preset Voice Catalog, no speaker encoder,
// no Voice Profile contract. Mirrors qwen3_tts_metadata_test.cpp's
// valid_metadata() at the HParams level.
synth::qwen3tts::HParams customvoice_hparams() {
    synth::qwen3tts::HParams h;
    h.voice_mode          = synth::qwen3tts::VoiceMode::PresetCatalog;
    h.has_speaker_encoder = false;
    synth::qwen3tts::PresetVoice voice;
    voice.id       = "aiden";
    voice.token_id = 2861;
    h.preset_voices.push_back(voice);
    return h;
}

// A Base-shaped package: no preset voices, a speaker encoder, and the Voice
// Profile contract Task 7 validates at load time. Mirrors
// qwen3_tts_metadata_test.cpp's base_metadata() at the HParams level.
synth::qwen3tts::HParams base_hparams() {
    synth::qwen3tts::HParams h;
    h.voice_mode                    = synth::qwen3tts::VoiceMode::ProfileSources;
    h.has_speaker_encoder           = true;
    h.profile.schema                = "qwen3-tts-voice-clone";
    h.profile.schema_version        = 1;
    h.profile.compatibility_id_hex  = std::string(64, 'a');
    h.profile.reference_sample_rate = 24000;
    h.profile.reference_channels    = 1;
    h.profile.min_frames_per_clip   = 24000;
    h.profile.max_frames_per_clip   = 720000;
    h.profile.max_total_frames      = 720000;
    h.profile.max_reference_count   = 1;
    return h;
}

int test_customvoice_package_has_a_selectable_catalog() {
    const synth::qwen3tts::HParams h = customvoice_hparams();
    SYNTH_TEST_CHECK(synth::qwen3tts::has_preset_voice_catalog(h));
    return 0;
}

// The rule Model::resolve_voice actually relies on is voice_mode, not "the
// catalog happens to be empty": a profile-sources package can never carry
// preset voices for real (qwen3_tts_metadata_test.cpp's
// test_profile_only_mode_with_presets_is_refused proves read_hparams refuses
// that combination), but this HParams is built directly rather than through
// read_hparams, so it can hold both at once -- proving has_preset_voice_catalog
// answers from voice_mode alone, not by noticing the list is empty.
int test_profile_sources_package_has_no_catalog_even_with_entries() {
    synth::qwen3tts::HParams     h = base_hparams();
    synth::qwen3tts::PresetVoice stray;
    stray.id       = "somebody";
    stray.token_id = 1;
    h.preset_voices.push_back(stray);

    SYNTH_TEST_CHECK(!synth::qwen3tts::has_preset_voice_catalog(h));
    // The entry really is there to find -- proving the gate, not an empty
    // list, is what would make Model::resolve_voice refuse.
    synth::qwen3tts::PresetVoice found;
    SYNTH_TEST_CHECK(synth::qwen3tts::find_preset_voice(h, "somebody", found));
    return 0;
}

// docs/c-interface.md's "no runtime Voice Profile support" shape, in full:
// zero source flags, and with them zero (or SYNTH_REQUIREMENT_UNSUPPORTED) in
// every field that describes a source -- including the Serialized Profile
// identity, which the same document requires to be an empty schema, a zero
// version and 32 zero ID bytes when that bit is clear. It is
// VoiceProfileInfo's own default, which is what makes this checkable field by
// field rather than by trusting the default.
int check_reports_no_voice_profile_support(const synth::VoiceProfileInfo & info) {
    SYNTH_TEST_CHECK(info.source_flags == 0);
    SYNTH_TEST_CHECK(info.reference_transcript == SYNTH_REQUIREMENT_UNSUPPORTED);
    SYNTH_TEST_CHECK(info.reference_language == SYNTH_REQUIREMENT_UNSUPPORTED);
    SYNTH_TEST_CHECK(info.description_language == SYNTH_REQUIREMENT_UNSUPPORTED);
    SYNTH_TEST_CHECK(info.reference_target_sample_rate == 0);
    SYNTH_TEST_CHECK(info.reference_target_channels == 0);
    SYNTH_TEST_CHECK(info.min_frames_per_clip == 0);
    SYNTH_TEST_CHECK(info.max_frames_per_clip == 0);
    SYNTH_TEST_CHECK(info.max_total_frames == 0);
    SYNTH_TEST_CHECK(info.max_reference_count == 0);
    SYNTH_TEST_CHECK(info.schema.empty());
    SYNTH_TEST_CHECK(info.schema_version == 0);
    for (uint8_t byte : info.compatibility_id) {
        SYNTH_TEST_CHECK(byte == 0);
    }
    return 0;
}

int test_customvoice_capability_reports_no_voice_profile_support() {
    const synth::qwen3tts::HParams h = customvoice_hparams();
    synth::VoiceProfileInfo        info;
    synth::qwen3tts::fill_voice_profile_capability(h, info);
    return check_reports_no_voice_profile_support(info);
}

// A future variant could in principle set has_speaker_encoder without also
// being profile-sources -- unreachable via read_hparams today (the profile
// contract and speaker encoder are only ever read together, gated on
// voice_mode == ProfileSources), but this HParams is built directly rather
// than through read_hparams, so it can hold both at once. Nothing about an
// encoder flag makes a runtime able to prepare a Profile, so this shape
// advertises nothing either.
int test_speaker_encoder_without_profile_sources_advertises_nothing() {
    synth::qwen3tts::HParams h = customvoice_hparams();
    h.has_speaker_encoder      = true;  // adversarial: profile contract left unset
    synth::VoiceProfileInfo info;
    synth::qwen3tts::fill_voice_profile_capability(h, info);
    return check_reports_no_voice_profile_support(info);
}

// The Base shape, through the production discriminator rather than by
// inspecting the fixture: a profile-sources package carries no Preset Voice
// Catalog, so a Voice id has nothing to resolve against.
int test_base_package_carries_no_preset_voice_catalog() {
    const synth::qwen3tts::HParams h = base_hparams();
    SYNTH_TEST_CHECK(!synth::qwen3tts::has_preset_voice_catalog(h));
    synth::qwen3tts::PresetVoice found;
    SYNTH_TEST_CHECK(!synth::qwen3tts::find_preset_voice(h, "aiden", found));
    return 0;
}

// Plan 2 landed preparation, so the advertisement is now true. The gate is
// the voice mode, not has_preset_voice_catalog: that predicate is true only
// for PresetCatalog -- CustomVoice, the variant with no speaker encoder --
// and gating on it would advertise Reference Audio on the variant that cannot
// prepare anything. The carryover records the correction; the assertion below
// is what would have caught it.
int test_base_capability_publishes_both_sources_together() {
    const synth::qwen3tts::HParams h = base_hparams();
    synth::VoiceProfileInfo        info;
    synth::qwen3tts::fill_voice_profile_capability(h, info);

    SYNTH_TEST_CHECK((info.source_flags & SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO) != 0);
    SYNTH_TEST_CHECK((info.source_flags & SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE) != 0);
    // Description Text is Stage 3's and Random Seed is nobody's: exactly two
    // bits, not "at least these two".
    SYNTH_TEST_CHECK(info.source_flags ==
                     (SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO | SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE));

    // Plan 2 implements one of two modes, so a transcript names a mode with no
    // implementation behind it. Plan 3 flips both of these to OPTIONAL in the
    // same change that lands ICL.
    SYNTH_TEST_CHECK(info.reference_transcript == SYNTH_REQUIREMENT_UNSUPPORTED);
    SYNTH_TEST_CHECK(info.reference_language == SYNTH_REQUIREMENT_UNSUPPORTED);
    SYNTH_TEST_CHECK(info.description_language == SYNTH_REQUIREMENT_UNSUPPORTED);

    SYNTH_TEST_CHECK(info.reference_target_sample_rate == 24000);
    SYNTH_TEST_CHECK(info.reference_target_channels == 1);
    SYNTH_TEST_CHECK(info.min_frames_per_clip == 24000);
    SYNTH_TEST_CHECK(info.max_frames_per_clip == 720000);
    SYNTH_TEST_CHECK(info.max_total_frames == 720000);
    SYNTH_TEST_CHECK(info.max_reference_count == 1);
    SYNTH_TEST_CHECK(info.schema == "qwen3-tts-voice-clone");
    SYNTH_TEST_CHECK(info.schema_version == 1);
    bool any_nonzero = false;
    for (uint8_t byte : info.compatibility_id) {
        any_nonzero = any_nonzero || byte != 0;
    }
    SYNTH_TEST_CHECK(any_nonzero);
    return 0;
}

// The reference count reaches the snapshot at its declared width.
//
// fill_voice_profile_capability is a pure function of HParams with no load
// rule in front of it -- that is exactly why this file exists (model-info.h's
// own comment on the two routes into VoiceProfileInfo) -- so it can be handed
// a value read_profile_contract would refuse, and that is the point here.
// 2^32 is the specific value that used to be destroyed: VoiceProfileInfo held
// this field as a uint32_t until 2026-08-13, so a package declaring it landed
// in the snapshot as 0, which docs/c-interface.md reads as "no Reference
// Audio clip may be used at all". Both families now refuse anything but 1 at
// load, and this pins the hop underneath that rule rather than through it, so
// widening the rule later cannot silently reintroduce the narrowing.
//
// The 2 case is here for the same reason at a value a caller might plausibly
// see if the load rule ever widens: it is not a boundary of any integer type,
// so only a faithful copy produces it.
int test_a_declared_reference_count_is_published_unnarrowed() {
    for (uint64_t declared : { uint64_t(2), uint64_t(1) << 32, ~uint64_t(0) }) {
        synth::qwen3tts::HParams h    = base_hparams();
        h.profile.max_reference_count = declared;
        synth::VoiceProfileInfo info;
        synth::qwen3tts::fill_voice_profile_capability(h, info);
        SYNTH_TEST_CHECK(info.max_reference_count == declared);
    }
    return 0;
}

// The public-seam counterpart of the rule above: a `synth_model` whose family
// is Qwen3Tts but whose capability snapshot is the CustomVoice all-zero shape
// must take the SAME generic "unsupported" fallback every non-participating
// family already takes at src/voice-profile.cpp's three dispatchers -- both
// before this family had ANY dispatch arm (an unconditional
// `family != Omnivoice` check) and after (a per-model check on
// `info.voice_profile.source_flags`, added alongside the family check by this
// same task). This is the trap the brief warns about made concrete: gating
// solely on `family == Qwen3Tts` -- the natural first draft once a Qwen3Tts
// branch exists at all -- would route a CustomVoice Loaded Model into a
// handler that dereferences `model->qwen3_tts`, which a hand-built
// `synth_model` like this one leaves null, exactly as
// tests/omnivoice_serialize_test.cpp's own cross-family guard arm relies on
// for `ModelFamily::Vits`.
int test_customvoice_model_refuses_public_profile_calls() {
    synth_model fake_model;
    fake_model.info.family        = synth::ModelFamily::Qwen3Tts;
    fake_model.info.voice_profile = synth::VoiceProfileInfo{};  // all-zero: CustomVoice's own shape

    synth_voice_reference_params_t reference_params;
    synth_voice_reference_params_init(&reference_params, sizeof(reference_params));
    synth_voice_profile_t * profile = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
    SYNTH_TEST_CHECK(synth_voice_profile_create_from_reference(&fake_model, &reference_params, &profile) ==
                     SYNTH_ERR_UNSUPPORTED_VOICE);
    SYNTH_TEST_CHECK(profile == nullptr);

    synth_voice_profile_load_params_t load_params;
    synth_voice_profile_load_params_init(&load_params, sizeof(load_params));
    profile = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
    SYNTH_TEST_CHECK(synth_voice_profile_load_from_memory(&fake_model, &load_params, &profile) ==
                     SYNTH_ERR_UNSUPPORTED_VOICE);
    SYNTH_TEST_CHECK(profile == nullptr);
    return 0;
}

}  // namespace

int main() {
    SYNTH_TEST_CHECK(test_customvoice_package_has_a_selectable_catalog() == 0);
    SYNTH_TEST_CHECK(test_profile_sources_package_has_no_catalog_even_with_entries() == 0);
    SYNTH_TEST_CHECK(test_customvoice_capability_reports_no_voice_profile_support() == 0);
    SYNTH_TEST_CHECK(test_speaker_encoder_without_profile_sources_advertises_nothing() == 0);
    SYNTH_TEST_CHECK(test_base_package_carries_no_preset_voice_catalog() == 0);
    SYNTH_TEST_CHECK(test_base_capability_publishes_both_sources_together() == 0);
    SYNTH_TEST_CHECK(test_a_declared_reference_count_is_published_unnarrowed() == 0);
    SYNTH_TEST_CHECK(test_customvoice_model_refuses_public_profile_calls() == 0);
    return 0;
}
