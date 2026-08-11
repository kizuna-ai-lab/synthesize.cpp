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
//      before ever trying to synthesize. As of Plan 1 that snapshot is
//      all-zero for EVERY variant of this family -- no preset voices on Base,
//      and no advertised Voice Profile source on either -- because nothing in
//      the runtime can create or consume a Profile yet
//      (src/voice-profile.cpp dispatches OmniVoice only). The Base package's
//      own ProfileContract is still read and validated at load time; what is
//      withheld is the runtime's claim, not the package's declaration.
//
// A `unit`-labelled test cannot depend on the real ~2.5 GB Base package
// (docs/testing.md), so both rules are exercised here directly against
// synthetic HParams, the way qwen3_tts_metadata_test.cpp and
// qwen3_tts_catalog_test.cpp already do -- no GGUF file and no loaded Model
// involved. The end-to-end refusal through synth_synthesize on a real loaded
// package is Task 10's integration step.

#include "arch/qwen3-tts/weights.h"
#include "synthesize.h"
#include "test-assert.h"

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

// The Critical finding of this branch's whole-branch review, inverted into a
// test. base_hparams() carries a fully populated, load-time-validated
// ProfileContract -- 24 kHz mono, real per-clip and total limits, a real
// schema and compatibility id -- and the capability snapshot still advertises
// NOTHING, because src/voice-profile.cpp cannot prepare, consume or
// serialize a Profile for this family: every source there is guarded on
// `family != ModelFamily::Omnivoice`. docs/c-interface.md is the contract
// that decides this ("A Model without runtime Voice Profile support reports
// zero flags"), and a package that carries a contract nothing can honour is
// not a Model with runtime support.
//
// Plan 2 is what flips this: when preparation exists, this same fixture must
// report SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO | SERIALIZED_PROFILE and the
// six limits below, and this test is expected to be rewritten there rather
// than deleted.
int test_base_capability_advertises_nothing_until_preparation_exists() {
    const synth::qwen3tts::HParams h = base_hparams();
    // The contract really is there to publish, which is what makes the
    // all-zero answer a decision rather than an empty struct.
    SYNTH_TEST_CHECK(h.profile.schema == "qwen3-tts-voice-clone");
    SYNTH_TEST_CHECK(h.profile.reference_sample_rate == 24000);
    SYNTH_TEST_CHECK(h.profile.max_reference_count == 1);

    synth::VoiceProfileInfo info;
    synth::qwen3tts::fill_voice_profile_capability(h, info);
    return check_reports_no_voice_profile_support(info);
}

}  // namespace

int main() {
    SYNTH_TEST_CHECK(test_customvoice_package_has_a_selectable_catalog() == 0);
    SYNTH_TEST_CHECK(test_profile_sources_package_has_no_catalog_even_with_entries() == 0);
    SYNTH_TEST_CHECK(test_customvoice_capability_reports_no_voice_profile_support() == 0);
    SYNTH_TEST_CHECK(test_speaker_encoder_without_profile_sources_advertises_nothing() == 0);
    SYNTH_TEST_CHECK(test_base_package_carries_no_preset_voice_catalog() == 0);
    SYNTH_TEST_CHECK(test_base_capability_advertises_nothing_until_preparation_exists() == 0);
    return 0;
}
