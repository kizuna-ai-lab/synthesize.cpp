// The Base package is the project's first that carries no Preset Voice
// Catalog at all: every request must supply a prepared Voice Profile instead
// (Plan 2). This exercises the two family-layer rules that make that honest
// rather than a silent guess:
//
//   1. weights.h's has_preset_voice_catalog, which Model::resolve_voice
//      (model.cpp) consults BEFORE find_preset_voice ever runs -- a
//      profile-sources package refuses with SYNTH_ERR_UNSUPPORTED_VOICE
//      whether a request names no voice_id at all or names one that could
//      never exist, because there is no catalog to have missed a voice in
//      either way. Both collapse to the same public status; what would
//      distinguish "no voice_id" from "unknown voice_id" is a diagnostic
//      concern, not a status code (see weights.h's find_preset_voice).
//
//   2. weights.h's fill_voice_profile_capability, which Model::get_info
//      (model.cpp) uses to build the capability snapshot a caller reads
//      before ever trying to synthesize: no preset voices, but Reference
//      Audio and Serialized Profile support declared honestly through
//      synth::VoiceProfileInfo.
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

int test_customvoice_capability_reports_no_voice_profile_support() {
    const synth::qwen3tts::HParams h = customvoice_hparams();
    synth::VoiceProfileInfo        info;
    synth::qwen3tts::fill_voice_profile_capability(h, info);

    // The "no Voice Profile support" shape docs/c-interface.md requires:
    // VoiceProfileInfo's own all-zero default, untouched.
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
    return 0;
}

int test_base_capability_reports_no_preset_voices() {
    const synth::qwen3tts::HParams h = base_hparams();
    SYNTH_TEST_CHECK(h.preset_voices.empty());
    return 0;
}

int test_base_capability_reports_reference_audio_and_serialized_profile() {
    const synth::qwen3tts::HParams h = base_hparams();
    synth::VoiceProfileInfo        info;
    synth::qwen3tts::fill_voice_profile_capability(h, info);

    SYNTH_TEST_CHECK((info.source_flags & SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO) != 0);
    SYNTH_TEST_CHECK((info.source_flags & SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE) != 0);
    // Nothing else: this package has no Description Text or Random Seed path.
    SYNTH_TEST_CHECK((info.source_flags & SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT) == 0);
    SYNTH_TEST_CHECK((info.source_flags & SYNTH_PROFILE_SOURCE_RANDOM_SEED) == 0);

    SYNTH_TEST_CHECK(info.reference_transcript == SYNTH_REQUIREMENT_OPTIONAL);
    SYNTH_TEST_CHECK(info.reference_language == SYNTH_REQUIREMENT_OPTIONAL);
    // Never claimed: this family has no Description Text path.
    SYNTH_TEST_CHECK(info.description_language == SYNTH_REQUIREMENT_UNSUPPORTED);

    SYNTH_TEST_CHECK(info.reference_target_sample_rate == 24000);
    SYNTH_TEST_CHECK(info.reference_target_channels == 1);
    SYNTH_TEST_CHECK(info.min_frames_per_clip == 24000);
    SYNTH_TEST_CHECK(info.max_frames_per_clip == 720000);
    SYNTH_TEST_CHECK(info.max_total_frames == 720000);
    SYNTH_TEST_CHECK(info.max_reference_count == 1);

    SYNTH_TEST_CHECK(info.schema == "qwen3-tts-voice-clone");
    SYNTH_TEST_CHECK(info.schema_version == 1);
    for (uint8_t byte : info.compatibility_id) {
        SYNTH_TEST_CHECK(byte == 0xaa);
    }
    return 0;
}

}  // namespace

int main() {
    SYNTH_TEST_CHECK(test_customvoice_package_has_a_selectable_catalog() == 0);
    SYNTH_TEST_CHECK(test_profile_sources_package_has_no_catalog_even_with_entries() == 0);
    SYNTH_TEST_CHECK(test_customvoice_capability_reports_no_voice_profile_support() == 0);
    SYNTH_TEST_CHECK(test_base_capability_reports_no_preset_voices() == 0);
    SYNTH_TEST_CHECK(test_base_capability_reports_reference_audio_and_serialized_profile() == 0);
    return 0;
}
