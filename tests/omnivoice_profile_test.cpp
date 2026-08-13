// Reference Audio profiles end to end (Plan 3's Task 14), against the real
// omnivoice package and its pinned reference clip.
//
// This is a model-guarded integration test, not a unit test: it drives the
// PUBLIC C seam's create_from_reference dispatch (src/voice-profile.cpp) --
// the first Voice Profile source any family implements for real -- with the
// real package's declared Reference Audio limits (min/max frames per clip,
// max_reference_count) and a real transcript. The exact-token-grid and
// ref_rms assertions additionally call synth::omnivoice::Model::encode_reference
// directly (bypassing the opaque public profile handle, which has no
// accessor for its contents): the SAME function voice-profile.cpp's
// dispatcher calls internally (via arch/omnivoice/profile.cpp's
// create_clone_prompt) on the SAME normalized PCM -- the pinned wav is
// already 24 kHz mono, the package's own target format, so
// synth::normalize_reference is an identity pass-through and the two paths
// see byte-identical input.

#include "arch/omnivoice/omnivoice.h"
#include "arch/omnivoice/profile.h"
#include "omnivoice_synthetic_package.h"
#include "synthesize.h"
#include "test-assert.h"
#include "voice-profile-handle.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// A minimal RIFF/WAVE reader. The pinned reference
// (models/omnivoice-reference-audio/seedtts_ref_en_1.wav) is confirmed
// mono, 16-bit PCM, 24 kHz by direct inspection (Python's stdlib `wave`
// module against the file itself) -- this reads exactly that shape and
// refuses anything else rather than silently reinterpreting it.
// ---------------------------------------------------------------------------
bool read_wav_mono16(const std::string & path, std::vector<float> & pcm, uint32_t & sample_rate) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return false;
    }
    // Every chunk_size in the loop below is an untrusted 32-bit field: it can
    // claim up to 4 GiB no matter how many bytes the file actually holds. The
    // file's own length is the only honest bound to size an allocation
    // against, so it is measured once, here, rather than trusted per chunk --
    // the same shape qwen3_tts_mel_driver.cpp's own copy carries.
    const std::streamoff file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (file_size < 0 || !file) {
        return false;
    }

    char riff_id[4];
    file.read(riff_id, 4);
    uint32_t riff_size = 0;
    file.read(reinterpret_cast<char *>(&riff_size), 4);
    char wave_id[4];
    file.read(wave_id, 4);
    if (!file || std::memcmp(riff_id, "RIFF", 4) != 0 || std::memcmp(wave_id, "WAVE", 4) != 0) {
        return false;
    }

    bool                 found_fmt       = false;
    bool                 found_data      = false;
    uint16_t             channels        = 0;
    uint16_t             bits_per_sample = 0;
    std::vector<int16_t> raw_samples;
    while (file && !(found_fmt && found_data)) {
        char chunk_id[4];
        file.read(chunk_id, 4);
        uint32_t chunk_size = 0;
        file.read(reinterpret_cast<char *>(&chunk_size), 4);
        if (!file) {
            break;
        }
        // Before anything is sized from it: a chunk cannot be longer than
        // what is left of the file. Without this the raw_samples.resize below
        // is sized from the claim alone, and the short read that follows
        // leaves the tail zero-filled -- fabricated samples a caller cannot
        // tell from real ones, on top of a 2 GiB allocation from a 4-byte
        // field.
        //
        // The span is the chunk PLUS its pad byte: RIFF keeps chunks
        // word-aligned, so an odd-sized chunk is followed by one padding byte
        // that this loop skips at the bottom. Counting it here is what makes
        // that skip safe -- an odd final chunk whose pad is missing would
        // otherwise be accepted and then seek past the end. The addition is
        // done in uint64_t: at chunk_size 0xFFFFFFFF a 32-bit +1 wraps to 0
        // and would admit the single largest claim there is.
        const uint64_t       chunk_span = uint64_t(chunk_size) + (chunk_size & 1u);
        const std::streamoff position   = file.tellg();
        if (position < 0 || chunk_span > uint64_t(file_size - position)) {
            return false;
        }
        constexpr uint32_t kConsumed = 16;
        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            // The floor BEFORE the fixed reads, not after them. Those 16
            // bytes are read unconditionally, so a `fmt ` declaring fewer
            // consumes bytes belonging to the next chunk and leaves the walk
            // one chunk out of step -- and the compensating seekg below only
            // runs when chunk_size is LARGER than 16, so nothing puts it back.
            if (chunk_size < kConsumed) {
                return false;
            }
            uint16_t audio_format = 0;
            uint32_t byte_rate    = 0;
            uint16_t block_align  = 0;
            file.read(reinterpret_cast<char *>(&audio_format), 2);
            file.read(reinterpret_cast<char *>(&channels), 2);
            file.read(reinterpret_cast<char *>(&sample_rate), 4);
            file.read(reinterpret_cast<char *>(&byte_rate), 4);
            file.read(reinterpret_cast<char *>(&block_align), 2);
            file.read(reinterpret_cast<char *>(&bits_per_sample), 2);
            if (chunk_size > kConsumed) {
                file.seekg(chunk_size - kConsumed, std::ios::cur);
            }
            found_fmt = (audio_format == 1);  // 1 == PCM, not float or compressed
        } else if (std::memcmp(chunk_id, "data", 4) == 0) {
            // An odd `data` size is not a whole number of 16-bit samples, and
            // reading it as one overflows the heap: the resize below
            // TRUNCATES (5 / 2 == 2, a four-byte buffer) while the read that
            // follows asks for all five. The file-length bound above does not
            // help -- the fifth byte really is in the file. Refused here,
            // before the resize, rather than by any later sample-count check,
            // because the overflow happens on the next line.
            if (chunk_size % sizeof(int16_t) != 0) {
                return false;
            }
            raw_samples.resize(chunk_size / sizeof(int16_t));
            file.read(reinterpret_cast<char *>(raw_samples.data()), chunk_size);
            found_data = true;
        } else {
            file.seekg(chunk_size, std::ios::cur);
        }
        // RIFF pads an odd-sized chunk to a word boundary. This reader never
        // skipped that byte, so a single odd chunk left every following chunk
        // id misaligned by one -- the three sibling readers in this tree all
        // carried the skip and this one did not.
        if (chunk_size % 2 == 1) {
            file.seekg(1, std::ios::cur);  // chunks are word-aligned
        }
    }
    if (!found_fmt || !found_data || channels != 1 || bits_per_sample != 16) {
        return false;
    }
    pcm.resize(raw_samples.size());
    for (size_t index = 0; index < raw_samples.size(); ++index) {
        pcm[index] = static_cast<float>(raw_samples[index]) / 32768.0f;
    }
    return true;
}

bool read_i32(const std::string & path, std::vector<int32_t> & values) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return false;
    }
    const std::streamsize size = file.tellg();
    if (size < 0 || size % sizeof(int32_t) != 0) {
        return false;
    }
    file.seekg(0);
    values.resize(static_cast<size_t>(size) / sizeof(int32_t));
    return bool(file.read(reinterpret_cast<char *>(values.data()), size));
}

// The `omni-clone-en` golden case's own pinned transcript
// (tests/golden/omnivoice/omnivoice-0-6b.manifest.json:
// cases[id=="omni-clone-en"].input.reference.transcript), transcribed here
// with this provenance comment rather than parsed from the manifest at test
// run time.
constexpr const char * kPinnedTranscript =
    "Some call me nature. Others call me Mother Nature. I've been here for over four point and "
    "five billion years, twenty-two thousand five hundred times longer than you.";

// tests/golden/omnivoice/omnivoice-0-6b.manifest.json's own `omni-design-en`
// case (voice.description); this is also the exact instruct the vocabulary
// transcription's own fixed-point unit case pins
// (tests/omnivoice_design_test.cpp), transcribed here with this provenance
// comment for the same reason kPinnedTranscript above is.
constexpr const char * kGoldenDesignInstruct = "female, young adult, high pitch";

struct SeenDiagnostic {
    std::string code;
    std::string message;
    bool        seen = false;
};

void SYNTH_CALL record_diagnostic(void * user_data, const synth_diagnostic_t * diagnostic) {
    auto * seen = static_cast<SeenDiagnostic *>(user_data);
    seen->code.assign(diagnostic->code, static_cast<size_t>(diagnostic->code_size));
    seen->message.assign(diagnostic->message, static_cast<size_t>(diagnostic->message_size));
    seen->seen = true;
}

synth_diagnostic_sink_t make_sink(SeenDiagnostic & target) {
    synth_diagnostic_sink_t sink;
    synth_diagnostic_sink_init(&sink, sizeof(sink));
    sink.emit      = record_diagnostic;
    sink.user_data = &target;
    return sink;
}

synth_voice_reference_t make_reference(const std::vector<float> & pcm,
                                       uint32_t                   sample_rate,
                                       const char *               transcript,
                                       const char *               language_tag) {
    synth_voice_reference_t reference;
    synth_voice_reference_init(&reference, sizeof(reference));
    reference.samples       = pcm.data();
    reference.frame_count   = pcm.size();
    reference.sample_rate   = sample_rate;
    reference.channel_count = 1;
    if (transcript != nullptr) {
        reference.transcript      = transcript;
        reference.transcript_size = std::strlen(transcript);
    }
    if (language_tag != nullptr) {
        reference.language_tag      = language_tag;
        reference.language_tag_size = std::strlen(language_tag);
    }
    return reference;
}

synth_voice_reference_params_t make_params(const synth_voice_reference_t * references,
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

// Drives synth_voice_profile_create_from_reference with an explicit
// (possibly out-of-contract) sample_rate/channel_count and transcript, and
// frees the resulting profile handle itself -- every caller below only cares
// about the returned status, and optionally the diagnostic emitted alongside
// it. Shared by the RFE-vs-format-check ordering arms (reviewer FINDING 1)
// and the transcript length cap arm (reviewer FINDING 2).
synth_status_t create_reference_profile_status(synth_model_t *            model,
                                               const std::vector<float> & pcm,
                                               uint32_t                   sample_rate,
                                               uint32_t                   channel_count,
                                               const char *               transcript,
                                               const char *               language_tag,
                                               SeenDiagnostic *           diagnostic = nullptr) {
    synth_voice_reference_t reference = make_reference(pcm, sample_rate, transcript, language_tag);
    reference.channel_count           = channel_count;
    synth_diagnostic_sink_t sink{};
    if (diagnostic != nullptr) {
        sink = make_sink(*diagnostic);
    }
    const synth_voice_reference_params_t params  = make_params(&reference, 1, diagnostic != nullptr ? &sink : nullptr);
    synth_voice_profile_t *              profile = nullptr;
    const synth_status_t                 status  = synth_voice_profile_create_from_reference(model, &params, &profile);
    if (profile != nullptr) {
        synth_voice_profile_free(profile);
    }
    return status;
}

// A sink `synth_synthesize` must never write through: used only by the
// cross-model check below, whose whole point is that the call is refused
// before any synthesis work happens.
synth_sink_result_t SYNTH_CALL refuse_audio(void * user_data, const synth_audio_chunk_t * chunk) {
    (void) chunk;
    *static_cast<bool *>(user_data) = true;
    return SYNTH_SINK_CONTINUE;
}

// Synthesizes `text` at a fixed `seed` against `profile` and returns the raw
// PCM plus its channel count, for the Serialized Profile round-trip's own
// byte-identical comparison below (Task 16).
bool synthesize_pcm(synth_model_t *         model,
                    synth_voice_profile_t * profile,
                    const char *            text,
                    uint64_t                seed,
                    std::vector<float> &    out_samples,
                    uint32_t &              out_channels) {
    synth_context_t * context = nullptr;
    if (synth_context_create(model, &context) != SYNTH_OK) {
        return false;
    }

    synth_request_t request;
    synth_request_init(&request, sizeof(request));
    request.input_kind        = SYNTH_INPUT_TEXT_UTF8;
    request.input_data        = text;
    request.input_count       = std::strlen(text);
    request.language_tag      = "en";
    request.language_tag_size = 2;
    request.voice_profile     = profile;
    request.seed              = seed;

    synth_audio_buffer_t * audio = nullptr;
    synth_result_t         result;
    synth_result_init(&result, sizeof(result));
    const bool ok = synth_synthesize_to_buffer(context, &request, &audio, &result) == SYNTH_OK && audio != nullptr;
    if (ok) {
        out_channels                = audio->channel_count;
        const uint64_t sample_count = audio->frame_count * audio->channel_count;
        out_samples.assign(audio->samples, audio->samples + sample_count);
    }
    if (audio != nullptr) {
        synth_audio_buffer_free(audio);
    }
    synth_context_free(context);
    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s <model.gguf> <reference.wav> <golden-tokens.i32-or-missing> <scratch-dir>\n",
                     argv[0]);
        return 2;
    }
    const std::string model_path         = argv[1];
    const std::string reference_wav      = argv[2];
    const std::string golden_tokens_path = argv[3];
    const std::string scratch_dir        = argv[4];

    std::vector<float> pcm;
    uint32_t           sample_rate = 0;
    SYNTH_TEST_CHECK(read_wav_mono16(reference_wav, pcm, sample_rate));
    SYNTH_TEST_CHECK(sample_rate == 24000);
    SYNTH_TEST_CHECK(!pcm.empty());

    synth_model_load_params_t load_params;
    synth_model_load_params_init(&load_params, sizeof(load_params));
    load_params.backend   = SYNTH_BACKEND_CPU;
    synth_model_t * model = nullptr;
    SYNTH_TEST_CHECK(synth_model_load(model_path.c_str(), &load_params, &model) == SYNTH_OK);

    // --- Capabilities: a profile-capable model returns REAL capabilities,
    // not the all-zero stub tests/voice_profile_api_test.c pins for a model
    // that never loads (that C-level test's arm is deliberately untouched by
    // Task 14; this is the capable-model arm it names).
    synth_voice_profile_capabilities_t capabilities;
    synth_voice_profile_capabilities_init(&capabilities, sizeof(capabilities));
    SYNTH_TEST_CHECK(synth_model_get_voice_profile_capabilities(model, &capabilities) == SYNTH_OK);
    SYNTH_TEST_CHECK((capabilities.source_flags & SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO) != 0);
    SYNTH_TEST_CHECK((capabilities.source_flags & SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT) != 0);
    // Serialized Profile (Task 16): claimed alongside REFERENCE_AUDIO/
    // DESCRIPTION_TEXT now that a Voice Profile prepared either way can be
    // serialized -- docs/c-interface.md: "Any Loaded Model that creates a
    // Profile from Reference Audio, Description Text, or Random Seed also
    // sets SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE". The schema string and
    // compatibility id now come out of the same HParams that were already
    // internally populated before this task, just not exposed until now.
    SYNTH_TEST_CHECK((capabilities.source_flags & SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE) != 0);
    SYNTH_TEST_CHECK(capabilities.reference_transcript == SYNTH_REQUIREMENT_REQUIRED);
    SYNTH_TEST_CHECK(capabilities.reference_language == SYNTH_REQUIREMENT_OPTIONAL);
    SYNTH_TEST_CHECK(capabilities.description_language == SYNTH_REQUIREMENT_OPTIONAL);
    SYNTH_TEST_CHECK(capabilities.max_reference_count == 1);
    SYNTH_TEST_CHECK(capabilities.reference_target_sample_rate == 24000);
    SYNTH_TEST_CHECK(capabilities.reference_target_channel_count == 1);
    SYNTH_TEST_CHECK(capabilities.min_reference_frames_per_clip == 24000);
    SYNTH_TEST_CHECK(capabilities.max_reference_frames_per_clip == 480000);
    SYNTH_TEST_CHECK(capabilities.max_reference_total_frames == 480000);
    SYNTH_TEST_CHECK(capabilities.profile_schema != nullptr && capabilities.profile_schema_size > 0);
    SYNTH_TEST_CHECK(std::string(capabilities.profile_schema, size_t(capabilities.profile_schema_size)) ==
                     "omnivoice-clone-prompt");
    SYNTH_TEST_CHECK(capabilities.profile_schema_version == 1);
    bool compatibility_id_nonzero = false;
    for (uint8_t byte : capabilities.profile_compatibility_id) {
        compatibility_id_nonzero = compatibility_id_nonzero || (byte != 0);
    }
    SYNTH_TEST_CHECK(compatibility_id_nonzero);

    // --- Missing transcript -> INVALID_ARG, named diagnostic.
    {
        const synth_voice_reference_t        reference = make_reference(pcm, sample_rate, nullptr, nullptr);
        SeenDiagnostic                       diagnostic;
        const synth_diagnostic_sink_t        sink    = make_sink(diagnostic);
        const synth_voice_reference_params_t params  = make_params(&reference, 1, &sink);
        synth_voice_profile_t *              profile = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
        SYNTH_TEST_CHECK(synth_voice_profile_create_from_reference(model, &params, &profile) == SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(profile == nullptr);
        SYNTH_TEST_CHECK(diagnostic.seen);
        SYNTH_TEST_CHECK(diagnostic.code == "voice_profile.transcript_required");
    }

    // --- Two clips -> INVALID_ARG (max_reference_count is 1). Neither
    // descriptor needs real content: the count is refused before either one
    // is ever read.
    {
        synth_voice_reference_t references[2];
        synth_voice_reference_init(&references[0], sizeof(references[0]));
        synth_voice_reference_init(&references[1], sizeof(references[1]));
        const synth_voice_reference_params_t params  = make_params(references, 2, nullptr);
        synth_voice_profile_t *              profile = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
        SYNTH_TEST_CHECK(synth_voice_profile_create_from_reference(model, &params, &profile) == SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(profile == nullptr);
    }

    // --- Silent-reference rejection (jiangzhuo's ruling, 2026-08-01): a
    // buffer of exact digital silence at exactly the package's minimum clip
    // length (so the RFE precheck admits it) is refused once encode_reference
    // measures ref_rms == 0.0f.
    {
        const std::vector<float>             silence(capabilities.min_reference_frames_per_clip, 0.0f);
        const synth_voice_reference_t        reference = make_reference(silence, sample_rate, kPinnedTranscript, "en");
        SeenDiagnostic                       diagnostic;
        const synth_diagnostic_sink_t        sink    = make_sink(diagnostic);
        const synth_voice_reference_params_t params  = make_params(&reference, 1, &sink);
        synth_voice_profile_t *              profile = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
        SYNTH_TEST_CHECK(synth_voice_profile_create_from_reference(model, &params, &profile) == SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(profile == nullptr);
        SYNTH_TEST_CHECK(diagnostic.seen);
        SYNTH_TEST_CHECK(diagnostic.code == "voice_profile.reference_silent");
    }

    // --- Reviewer FINDING 1: an out-of-contract sample rate or channel
    // count is refused with SYNTH_ERR_UNSUPPORTED_INPUT BEFORE the RFE
    // precheck ever runs, regardless of clip length. The reviewer's own
    // probe: rate 0 always produced "reference_too_short" (INVALID_ARG);
    // rate 4000 produced THREE different statuses depending on clip length,
    // because reference_frame_equivalent() happily computed something
    // plausible-looking from the invalid rate before the format check ever
    // ran. Every arm below pairs a bad rate/channel-count with a clip at
    // BOTH this package's minimum and maximum declared length, so a length-
    // dependent regression cannot hide behind only one of the two -- these
    // are exactly the reviewer's own probe values, the spec for this fix.
    {
        const std::vector<float> clip_min(capabilities.min_reference_frames_per_clip, 0.0f);
        const std::vector<float> clip_max(capabilities.max_reference_frames_per_clip, 0.0f);

        // rate 0.
        SYNTH_TEST_CHECK(create_reference_profile_status(model, clip_min, 0, 1, kPinnedTranscript, "en") ==
                         SYNTH_ERR_UNSUPPORTED_INPUT);
        SYNTH_TEST_CHECK(create_reference_profile_status(model, clip_max, 0, 1, kPinnedTranscript, "en") ==
                         SYNTH_ERR_UNSUPPORTED_INPUT);
        // rate 7999: one below SYNTH_REFERENCE_SAMPLE_RATE_MIN.
        SYNTH_TEST_CHECK(create_reference_profile_status(model, clip_min, 7999, 1, kPinnedTranscript, "en") ==
                         SYNTH_ERR_UNSUPPORTED_INPUT);
        SYNTH_TEST_CHECK(create_reference_profile_status(model, clip_max, 7999, 1, kPinnedTranscript, "en") ==
                         SYNTH_ERR_UNSUPPORTED_INPUT);
        // rate 192001: one above SYNTH_REFERENCE_SAMPLE_RATE_MAX.
        SYNTH_TEST_CHECK(create_reference_profile_status(model, clip_min, 192001, 1, kPinnedTranscript, "en") ==
                         SYNTH_ERR_UNSUPPORTED_INPUT);
        SYNTH_TEST_CHECK(create_reference_profile_status(model, clip_max, 192001, 1, kPinnedTranscript, "en") ==
                         SYNTH_ERR_UNSUPPORTED_INPUT);
        // 3 channels: one above SYNTH_REFERENCE_CHANNELS_MAX, at this
        // package's own valid target sample rate.
        SYNTH_TEST_CHECK(create_reference_profile_status(model, clip_min, sample_rate, 3, kPinnedTranscript, "en") ==
                         SYNTH_ERR_UNSUPPORTED_INPUT);
        SYNTH_TEST_CHECK(create_reference_profile_status(model, clip_max, sample_rate, 3, kPinnedTranscript, "en") ==
                         SYNTH_ERR_UNSUPPORTED_INPUT);
    }

    // --- Reviewer FINDING 2: a transcript over this family's transcript
    // length cap (kMaxClonePromptTranscriptLength, arch/omnivoice/profile.h)
    // is refused at CREATION with a named diagnostic, rather than silently
    // accepted, serialized, and only THEN rejected by our own loader's
    // prescan whitelist with a bare INVALID_ARG. Exactly one byte over the
    // cap; the pinned reference clip (already exercised above) is reused
    // since only the transcript is under test here.
    {
        std::string over_cap(size_t(synth::omnivoice::kMaxClonePromptTranscriptLength) + 1, 'a');
        over_cap.back() = '.';  // already end-punctuation: add_punctuation() will not extend this further
        SeenDiagnostic diagnostic;
        SYNTH_TEST_CHECK(create_reference_profile_status(model, pcm, sample_rate, 1, over_cap.c_str(), "en",
                                                         &diagnostic) == SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(diagnostic.seen);
        SYNTH_TEST_CHECK(diagnostic.code == "voice_profile.transcript_too_long");
    }

    // --- The real clip, real transcript: profile creation succeeds through
    // the public seam.
    {
        const synth_voice_reference_t        reference = make_reference(pcm, sample_rate, kPinnedTranscript, "en");
        const synth_voice_reference_params_t params    = make_params(&reference, 1, nullptr);
        synth_voice_profile_t *              profile   = nullptr;
        SYNTH_TEST_CHECK(synth_voice_profile_create_from_reference(model, &params, &profile) == SYNTH_OK);
        SYNTH_TEST_CHECK(profile != nullptr);
        synth_voice_profile_free(profile);
    }

    // --- ref_rms and the token grid: since the opaque public profile has no
    // accessor for its contents, this calls the family's own
    // Model::encode_reference directly (the exact function
    // arch/omnivoice/profile.cpp's create_clone_prompt calls internally) on
    // the SAME pcm the public path above just normalized (an identity
    // pass-through: the wav is already the package's 24 kHz mono target
    // format).
    {
        std::unique_ptr<synth::omnivoice::Model> family_model;
        SYNTH_TEST_CHECK(synth::omnivoice::Model::load_cpu(model_path, family_model) == SYNTH_OK);
        synth::omnivoice::ReferenceEncoding encoding;
        SYNTH_TEST_CHECK(family_model->encode_reference(pcm, 0, encoding) == SYNTH_OK);

        // reference-encoder-host.h's own measured figure for this exact
        // file, cited in clip_and_boost_reference's header comment:
        // 0.1229146420955658 on the full 337726-sample wav.
        const double kExpectedRefRms = 0.1229146420955658;
        SYNTH_TEST_CHECK(std::fabs(double(encoding.ref_rms) - kExpectedRefRms) < 1e-4);

        std::vector<int32_t> expected_tokens;
        if (read_i32(golden_tokens_path, expected_tokens)) {
            SYNTH_TEST_CHECK(encoding.tokens == expected_tokens);
        } else {
            std::fprintf(stderr,
                         "omnivoice-profile-test: no golden token grid at %s (uncommitted oracle artifact); "
                         "skipping the exact-grid comparison\n",
                         golden_tokens_path.c_str());
        }
    }

    // --- Cross-model: a profile created against the real model, presented to
    // a DIFFERENT model's synthesis request, is refused with
    // UNSUPPORTED_VOICE rather than silently accepted or misread. The other
    // model is the small synthetic omnivoice package
    // (omnivoice_synthetic_package.h) every earlier omnivoice unit test
    // already builds and loads -- tiny (kilobytes, not the real package's
    // gigabytes of F32 weights), and it declares the same real ProfileContract
    // shape (schema, schema_version, target format, reference limits) this
    // family's loader requires of every package, real or synthetic.
    {
        const synth_voice_reference_t        reference = make_reference(pcm, sample_rate, kPinnedTranscript, "en");
        const synth_voice_reference_params_t params    = make_params(&reference, 1, nullptr);
        synth_voice_profile_t *              profile   = nullptr;
        SYNTH_TEST_CHECK(synth_voice_profile_create_from_reference(model, &params, &profile) == SYNTH_OK);
        SYNTH_TEST_CHECK(profile != nullptr);

        const std::string                                  synthetic_path = scratch_dir + "/cross-model.gguf";
        synth::omnivoice::testing::SyntheticPackageOptions synthetic_options;
        synthetic_options.ascii_text_vocab = true;  // enough to tokenize "a" below
        SYNTH_TEST_CHECK(synth::omnivoice::testing::write_synthetic_package(synthetic_path, synthetic_options));

        synth_model_t * other_model = nullptr;
        SYNTH_TEST_CHECK(synth_model_load(synthetic_path.c_str(), &load_params, &other_model) == SYNTH_OK);

        synth_context_t * context = nullptr;
        SYNTH_TEST_CHECK(synth_context_create(other_model, &context) == SYNTH_OK);

        const char *       text        = "a";
        bool               wrote_audio = false;
        synth_audio_sink_t sink;
        synth_audio_sink_init(&sink, sizeof(sink));
        sink.write     = refuse_audio;
        sink.user_data = &wrote_audio;

        synth_request_t request;
        synth_request_init(&request, sizeof(request));
        request.input_kind    = SYNTH_INPUT_TEXT_UTF8;
        request.input_data    = text;
        request.input_count   = std::strlen(text);
        request.voice_profile = profile;

        synth_result_t result;
        synth_result_init(&result, sizeof(result));
        SYNTH_TEST_CHECK(synth_synthesize(context, &request, &sink, &result) == SYNTH_ERR_UNSUPPORTED_VOICE);
        SYNTH_TEST_CHECK(!wrote_audio);

        synth_context_free(context);
        synth_model_free(other_model);
        synth_voice_profile_free(profile);
    }

    // --- Description Text ("voice design"), Task 15: the golden-instruct
    // profile through the public seam, then a real synthesis producing
    // finite PCM. The vocabulary transcription itself (every rejection arm,
    // the fixed-point cases, width-comma splitting) is
    // tests/omnivoice_design_test.cpp's job, against a synthetic package --
    // this only proves the wiring reaches the real package end to end.
    {
        synth_voice_description_params_t params;
        synth_voice_description_params_init(&params, sizeof(params));
        params.description      = kGoldenDesignInstruct;
        params.description_size = std::strlen(kGoldenDesignInstruct);

        synth_voice_profile_t * profile = nullptr;
        SYNTH_TEST_CHECK(synth_voice_profile_create_from_description(model, &params, &profile) == SYNTH_OK);
        SYNTH_TEST_CHECK(profile != nullptr);

        synth_context_t * context = nullptr;
        SYNTH_TEST_CHECK(synth_context_create(model, &context) == SYNTH_OK);

        const char *    text = "OmniVoice speaks with one voice.";
        synth_request_t request;
        synth_request_init(&request, sizeof(request));
        request.input_kind        = SYNTH_INPUT_TEXT_UTF8;
        request.input_data        = text;
        request.input_count       = std::strlen(text);
        request.language_tag      = "en";
        request.language_tag_size = 2;
        request.voice_profile     = profile;
        request.seed              = 0;

        synth_audio_buffer_t * audio = nullptr;
        synth_result_t         result;
        synth_result_init(&result, sizeof(result));
        SYNTH_TEST_CHECK(synth_synthesize_to_buffer(context, &request, &audio, &result) == SYNTH_OK);
        SYNTH_TEST_CHECK(audio != nullptr);
        SYNTH_TEST_CHECK(audio->frame_count > 0);
        const uint64_t sample_count = audio->frame_count * audio->channel_count;
        for (uint64_t index = 0; index < sample_count; ++index) {
            SYNTH_TEST_CHECK(std::isfinite(audio->samples[index]));
        }

        synth_audio_buffer_free(audio);
        synth_context_free(context);
        synth_voice_profile_free(profile);
    }

    // --- Serialized Profiles (Plan 3's Task 16): the ClonePrompt round
    // trip against the real package -- serialize, load_from_memory, then
    // synthesize at a fixed seed from BOTH the original and the reloaded
    // profile and require byte-identical PCM. Two serialize calls over the
    // same profile are also compared byte-for-byte (determinism).
    {
        const synth_voice_reference_t        reference = make_reference(pcm, sample_rate, kPinnedTranscript, "en");
        const synth_voice_reference_params_t params    = make_params(&reference, 1, nullptr);
        synth_voice_profile_t *              original  = nullptr;
        SYNTH_TEST_CHECK(synth_voice_profile_create_from_reference(model, &params, &original) == SYNTH_OK);

        synth_voice_profile_serialize_params_t serialize_params;
        synth_voice_profile_serialize_params_init(&serialize_params, sizeof(serialize_params));
        synth_byte_buffer_t * buffer_a = nullptr;
        synth_byte_buffer_t * buffer_b = nullptr;
        SYNTH_TEST_CHECK(synth_voice_profile_serialize(original, &serialize_params, &buffer_a) == SYNTH_OK);
        SYNTH_TEST_CHECK(synth_voice_profile_serialize(original, &serialize_params, &buffer_b) == SYNTH_OK);
        SYNTH_TEST_CHECK(buffer_a->data_size == buffer_b->data_size);
        SYNTH_TEST_CHECK(std::memcmp(buffer_a->data, buffer_b->data, buffer_a->data_size) == 0);

        synth_voice_profile_load_params_t load_params;
        synth_voice_profile_load_params_init(&load_params, sizeof(load_params));
        load_params.data                 = buffer_a->data;
        load_params.data_size            = buffer_a->data_size;
        synth_voice_profile_t * reloaded = nullptr;
        SYNTH_TEST_CHECK(synth_voice_profile_load_from_memory(model, &load_params, &reloaded) == SYNTH_OK);
        SYNTH_TEST_CHECK(reloaded != nullptr);
        SYNTH_TEST_CHECK(reloaded->model == model);
        SYNTH_TEST_CHECK(reloaded->family_tag == synth::ProfileFamilyTag::OmnivoiceClone);

        // Short on purpose (fix round 1, reviewer FINDING 3): the identity
        // claim ("original and reloaded produce the same PCM") does not
        // need a long sentence to be meaningful, and a short one keeps this
        // real forward pass affordable under the sanitizer build -- see
        // this test's own TIMEOUT comment in tests/CMakeLists.txt for the
        // measurements this shortening was based on. "Hi." is the same
        // minimal text tests/CMakeLists.txt's own cleanup-test invocation
        // for this family already exercises successfully against the real
        // package.
        const char *       text = "Hi.";
        std::vector<float> pcm_original;
        std::vector<float> pcm_reloaded;
        uint32_t           channels_original = 0;
        uint32_t           channels_reloaded = 0;
        SYNTH_TEST_CHECK(synthesize_pcm(model, original, text, 7, pcm_original, channels_original));
        SYNTH_TEST_CHECK(synthesize_pcm(model, reloaded, text, 7, pcm_reloaded, channels_reloaded));
        SYNTH_TEST_CHECK(channels_original == channels_reloaded);
        SYNTH_TEST_CHECK(!pcm_original.empty());
        SYNTH_TEST_CHECK(pcm_original.size() == pcm_reloaded.size());
        SYNTH_TEST_CHECK(std::memcmp(pcm_original.data(), pcm_reloaded.data(), pcm_original.size() * sizeof(float)) ==
                         0);

        synth_byte_buffer_free(buffer_a);
        synth_byte_buffer_free(buffer_b);
        synth_voice_profile_free(original);
        synth_voice_profile_free(reloaded);
    }

    // --- Reviewer FINDING 2, the cheap-and-valuable complement to the
    // "one byte over the cap is refused at creation" arm above: a transcript
    // AT the cap (not over it) round-trips through create -> serialize ->
    // load without ever hitting the loader's own prescan whitelist --
    // exactly the boundary kPrescanMaxStringLength and
    // kMaxClonePromptTranscriptLength are tied together to keep passable.
    {
        std::string at_cap(size_t(synth::omnivoice::kMaxClonePromptTranscriptLength) - 1, 'a');
        at_cap += '.';  // canonical_transcript.size() == kMaxClonePromptTranscriptLength exactly
        const synth_voice_reference_t        reference = make_reference(pcm, sample_rate, at_cap.c_str(), "en");
        const synth_voice_reference_params_t params    = make_params(&reference, 1, nullptr);
        synth_voice_profile_t *              original  = nullptr;
        SYNTH_TEST_CHECK(synth_voice_profile_create_from_reference(model, &params, &original) == SYNTH_OK);

        synth_voice_profile_serialize_params_t serialize_params;
        synth_voice_profile_serialize_params_init(&serialize_params, sizeof(serialize_params));
        synth_byte_buffer_t * buffer = nullptr;
        SYNTH_TEST_CHECK(synth_voice_profile_serialize(original, &serialize_params, &buffer) == SYNTH_OK);

        synth_voice_profile_load_params_t load_params;
        synth_voice_profile_load_params_init(&load_params, sizeof(load_params));
        load_params.data                 = buffer->data;
        load_params.data_size            = buffer->data_size;
        synth_voice_profile_t * reloaded = nullptr;
        SYNTH_TEST_CHECK(synth_voice_profile_load_from_memory(model, &load_params, &reloaded) == SYNTH_OK);
        SYNTH_TEST_CHECK(reloaded != nullptr);
        SYNTH_TEST_CHECK(reloaded->family_tag == synth::ProfileFamilyTag::OmnivoiceClone);

        synth_byte_buffer_free(buffer);
        synth_voice_profile_free(original);
        synth_voice_profile_free(reloaded);
    }

    // --- Serialized Profiles: the DesignInstruct round trip, same proof.
    {
        synth_voice_description_params_t params;
        synth_voice_description_params_init(&params, sizeof(params));
        params.description               = kGoldenDesignInstruct;
        params.description_size          = std::strlen(kGoldenDesignInstruct);
        synth_voice_profile_t * original = nullptr;
        SYNTH_TEST_CHECK(synth_voice_profile_create_from_description(model, &params, &original) == SYNTH_OK);

        synth_voice_profile_serialize_params_t serialize_params;
        synth_voice_profile_serialize_params_init(&serialize_params, sizeof(serialize_params));
        synth_byte_buffer_t * buffer = nullptr;
        SYNTH_TEST_CHECK(synth_voice_profile_serialize(original, &serialize_params, &buffer) == SYNTH_OK);

        synth_voice_profile_load_params_t load_params;
        synth_voice_profile_load_params_init(&load_params, sizeof(load_params));
        load_params.data                 = buffer->data;
        load_params.data_size            = buffer->data_size;
        synth_voice_profile_t * reloaded = nullptr;
        SYNTH_TEST_CHECK(synth_voice_profile_load_from_memory(model, &load_params, &reloaded) == SYNTH_OK);
        SYNTH_TEST_CHECK(reloaded != nullptr);
        SYNTH_TEST_CHECK(reloaded->family_tag == synth::ProfileFamilyTag::OmnivoiceDesign);

        // Short on purpose -- same reasoning as the ClonePrompt round
        // trip's own comment above.
        const char *       text = "Hi.";
        std::vector<float> pcm_original;
        std::vector<float> pcm_reloaded;
        uint32_t           channels_original = 0;
        uint32_t           channels_reloaded = 0;
        SYNTH_TEST_CHECK(synthesize_pcm(model, original, text, 7, pcm_original, channels_original));
        SYNTH_TEST_CHECK(synthesize_pcm(model, reloaded, text, 7, pcm_reloaded, channels_reloaded));
        SYNTH_TEST_CHECK(channels_original == channels_reloaded);
        SYNTH_TEST_CHECK(!pcm_original.empty());
        SYNTH_TEST_CHECK(pcm_original.size() == pcm_reloaded.size());
        SYNTH_TEST_CHECK(std::memcmp(pcm_original.data(), pcm_reloaded.data(), pcm_original.size() * sizeof(float)) ==
                         0);

        synth_byte_buffer_free(buffer);
        synth_voice_profile_free(original);
        synth_voice_profile_free(reloaded);
    }

    synth_model_free(model);
    return 0;
}
