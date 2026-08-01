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
#include "omnivoice_synthetic_package.h"
#include "synthesize.h"
#include "test-assert.h"

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
    std::ifstream file(path, std::ios::binary);
    if (!file) {
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
        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            uint16_t audio_format = 0;
            uint32_t byte_rate    = 0;
            uint16_t block_align  = 0;
            file.read(reinterpret_cast<char *>(&audio_format), 2);
            file.read(reinterpret_cast<char *>(&channels), 2);
            file.read(reinterpret_cast<char *>(&sample_rate), 4);
            file.read(reinterpret_cast<char *>(&byte_rate), 4);
            file.read(reinterpret_cast<char *>(&block_align), 2);
            file.read(reinterpret_cast<char *>(&bits_per_sample), 2);
            constexpr uint32_t kConsumed = 16;
            if (chunk_size > kConsumed) {
                file.seekg(chunk_size - kConsumed, std::ios::cur);
            }
            found_fmt = (audio_format == 1);  // 1 == PCM, not float or compressed
        } else if (std::memcmp(chunk_id, "data", 4) == 0) {
            raw_samples.resize(chunk_size / sizeof(int16_t));
            file.read(reinterpret_cast<char *>(raw_samples.data()), chunk_size);
            found_data = true;
        } else {
            file.seekg(chunk_size, std::ios::cur);
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

// A sink `synth_synthesize` must never write through: used only by the
// cross-model check below, whose whole point is that the call is refused
// before any synthesis work happens.
synth_sink_result_t SYNTH_CALL refuse_audio(void * user_data, const synth_audio_chunk_t * chunk) {
    (void) chunk;
    *static_cast<bool *>(user_data) = true;
    return SYNTH_SINK_CONTINUE;
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
    // Serialized Profile is Task 16's; not claimed yet, so the schema/version/
    // compatibility-id fields all stay at their unsupported default even
    // though this model's HParams already carries real values internally.
    SYNTH_TEST_CHECK((capabilities.source_flags & SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE) == 0);
    SYNTH_TEST_CHECK(capabilities.reference_transcript == SYNTH_REQUIREMENT_REQUIRED);
    SYNTH_TEST_CHECK(capabilities.reference_language == SYNTH_REQUIREMENT_OPTIONAL);
    SYNTH_TEST_CHECK(capabilities.description_language == SYNTH_REQUIREMENT_OPTIONAL);
    SYNTH_TEST_CHECK(capabilities.max_reference_count == 1);
    SYNTH_TEST_CHECK(capabilities.reference_target_sample_rate == 24000);
    SYNTH_TEST_CHECK(capabilities.reference_target_channel_count == 1);
    SYNTH_TEST_CHECK(capabilities.min_reference_frames_per_clip == 24000);
    SYNTH_TEST_CHECK(capabilities.max_reference_frames_per_clip == 480000);
    SYNTH_TEST_CHECK(capabilities.max_reference_total_frames == 480000);
    SYNTH_TEST_CHECK(capabilities.profile_schema == nullptr && capabilities.profile_schema_size == 0);
    SYNTH_TEST_CHECK(capabilities.profile_schema_version == 0);
    for (uint8_t byte : capabilities.profile_compatibility_id) {
        SYNTH_TEST_CHECK(byte == 0);
    }

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

    synth_model_free(model);
    return 0;
}
