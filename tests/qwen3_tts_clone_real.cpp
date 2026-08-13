// Reference audio in, cloned audio out: the x-vector clone path against the
// real Base package and its pinned reference clip, entirely through the
// public C seam (Task 11 opened that seam; nothing here reaches around it
// except the one direct family-level call assertion 1 needs).
//
// This is Stage 2 Plan 2's completion gate
// (docs/superpowers/plans/2026-08-12-qwen3-tts-stage-2-xvector-path.md,
// spec Sec.8): "reference audio in, cloned audio out on CPU; a usable clone
// capability exists." On the pattern of tests/omnivoice_profile_test.cpp.
//
// Task 11's own review proved the embedding's VALUES reach the graph by
// hand -- negating the x-vector in place changed the audio, and two
// different references produced different audio, with a determinism
// control -- but nothing committed asserted it: the existing synthesis
// checks (tests/qwen3_tts_base_load_real.cpp's check_profile_synthesizes)
// only check SYNTH_OK and frame_count > 0, which a Profile the graph
// silently ignored would also satisfy. This file closes that gap with six
// assertions, each its own block below:
//
//   1. a Profile prepared from the real clip has enc_dim floats and matches
//      speaker/x_vector.f32 above the tolerance Task 6 committed;
//   2. synthesis with that Profile produces finite, non-silent PCM at 24 kHz;
//   3. the same Profile and seed produce identical PCM twice (request
//      repeatability);
//   4. two different reference clips produce different PCM -- the one
//      assertion that actually gates "cloned" as opposed to merely
//      "synthesized": a Profile the graph ignored would produce identical
//      audio here too, since the text and seed are unchanged from (2)/(3);
//   5. serialize -> free -> load -> synthesize reproduces the same PCM the
//      original Profile produced -- what makes the Serialized Profile a real
//      artifact rather than a round-trippable blob;
//   6. a Profile presented to a second Loaded Model refuses with
//      SYNTH_ERR_UNSUPPORTED_VOICE and writes no audio;
//   7. a transcript-assisted (ICL) Profile prepared from the SAME clip and
//      its own real transcript synthesizes finite, non-silent PCM that
//      differs from (2)'s -- Plan 3's Task 11, and the only place the real
//      clip's real transcript is used for a real synthesis.
//
// Only one real reference clip is materialized in this tree
// (SYNTH_QWEN3_TTS_REFERENCE_WAV, models/qwen3-tts-reference-audio/clone.wav
// by default -- fetched, not committed, since models/ is gitignored);
// assertion 4's "two different reference clips" therefore pairs it with a
// synthesized 440 Hz tone rather than a second real recording, the same
// substitution tests/qwen3_tts_base_load_real.cpp's own make_tone already
// makes for every wiring/refusal check that needs "a valid, non-silent
// reference" without caring what voice it names. That is enough for what
// assertion 4 decides -- the two inputs differ, so the two outputs must --
// but it is NOT evidence that the clone resembles the speaker in the clip.
// An audit on 2026-08-13 found no obvious regression in x-vector mode; one
// listener, one source clip — evidence of resemblance, not a port property —
// while Quality Evaluation per ADR 0017 stays unrun.
//
// This test is registration-gated on the oracle x-vector dump as well as on
// the package and the clip: assertion 1 is the only enforcement of the
// committed speaker.x_vector tolerance, and it refuses rather than skips when
// the dump is unreadable. See tests/CMakeLists.txt beside its add_test.
//
// What this file does NOT re-test: the capability snapshot, the full
// refusal matrix (bad rate/channels, too many clips, out-of-bounds length,
// an unsupported transcript/language tag), and the Catalog-less no-Profile
// refusal -- tests/qwen3_tts_base_load_real.cpp already covers all of those
// against the real package. This file's scope is exactly the six assertions
// above.

#include "arch/qwen3-tts/qwen3-tts.h"
#include "arch/qwen3-tts/weights.h"
#include "synthesize.h"
#include "test-assert.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

// A minimal RIFF/WAVE reader for the two shapes this file ever needs to
// read: the pinned reference clip (models/qwen3-tts-reference-audio/
// clone.wav), confirmed by direct inspection (Python's soundfile against the
// file itself) to be mono, 24 kHz, IEEE float32 -- and, defensively, 16-bit
// PCM mono, the shape tests/omnivoice_profile_test.cpp's own read_wav_mono16
// reads exclusively. Refuses multi-channel audio and any other sample
// format rather than silently reinterpreting it, the same contract that
// reader states for its own (narrower) shape.
bool read_wav_mono(const std::string & path, std::vector<float> & pcm, uint32_t & sample_rate) {
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

    bool              found_fmt       = false;
    bool              found_data      = false;
    uint16_t          audio_format    = 0;
    uint16_t          channels        = 0;
    uint16_t          bits_per_sample = 0;
    std::vector<char> data_bytes;
    while (file && !(found_fmt && found_data)) {
        char chunk_id[4];
        file.read(chunk_id, 4);
        uint32_t chunk_size = 0;
        file.read(reinterpret_cast<char *>(&chunk_size), 4);
        if (!file) {
            break;
        }
        // Before anything is sized from it: a chunk cannot be longer than
        // what is left of the file. Without this the data_bytes.resize below
        // is sized from the claim alone, and the short read that follows
        // leaves the tail zero-filled -- fabricated samples a caller cannot
        // tell from real ones, on top of a 4 GiB allocation from a 4-byte
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
            uint32_t rate        = 0;
            uint32_t byte_rate   = 0;
            uint16_t block_align = 0;
            file.read(reinterpret_cast<char *>(&audio_format), 2);
            file.read(reinterpret_cast<char *>(&channels), 2);
            file.read(reinterpret_cast<char *>(&rate), 4);
            file.read(reinterpret_cast<char *>(&byte_rate), 4);
            file.read(reinterpret_cast<char *>(&block_align), 2);
            file.read(reinterpret_cast<char *>(&bits_per_sample), 2);
            sample_rate = rate;
            if (chunk_size > kConsumed) {
                file.seekg(chunk_size - kConsumed, std::ios::cur);
            }
            found_fmt = (audio_format == 1 && bits_per_sample == 16) ||  // PCM16
                        (audio_format == 3 && bits_per_sample == 32);    // IEEE float32
        } else if (std::memcmp(chunk_id, "data", 4) == 0) {
            data_bytes.resize(chunk_size);
            file.read(data_bytes.data(), std::streamsize(chunk_size));
            found_data = true;
        } else {
            file.seekg(chunk_size, std::ios::cur);
        }
        if (chunk_size % 2 == 1) {
            file.seekg(1, std::ios::cur);  // chunks are word-aligned
        }
    }
    if (!found_fmt || !found_data || channels != 1) {
        return false;
    }

    if (audio_format == 1) {
        if (data_bytes.size() % sizeof(int16_t) != 0) {
            return false;
        }
        const size_t count = data_bytes.size() / sizeof(int16_t);
        pcm.resize(count);
        for (size_t index = 0; index < count; ++index) {
            int16_t sample = 0;
            std::memcpy(&sample, data_bytes.data() + index * sizeof(int16_t), sizeof(int16_t));
            pcm[index] = float(sample) / 32768.0f;
        }
    } else {
        if (data_bytes.size() % sizeof(float) != 0) {
            return false;
        }
        pcm.resize(data_bytes.size() / sizeof(float));
        std::memcpy(pcm.data(), data_bytes.data(), data_bytes.size());
    }
    return true;
}

bool read_f32(const std::string & path, std::vector<float> & values) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return false;
    }
    const std::streamsize size = file.tellg();
    if (size < 0 || size % std::streamsize(sizeof(float)) != 0) {
        return false;
    }
    file.seekg(0);
    values.resize(static_cast<size_t>(size) / sizeof(float));
    return bool(file.read(reinterpret_cast<char *>(values.data()), size));
}

bool write_f32(const std::string & path, const std::vector<float> & values) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    if (!values.empty()) {
        file.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size() * sizeof(float)));
    }
    return bool(file);
}

// Cosine similarity and elementwise max-abs difference, the same two
// quantities tests/qwen3_tts_xvector_driver.cpp's own committed measurement
// (Task 6) reports for this exact clip.
void compare_vectors(const std::vector<float> & a,
                     const std::vector<float> & b,
                     double &                   out_cosine,
                     float &                    out_max_abs) {
    double dot     = 0.0;
    double norm_a  = 0.0;
    double norm_b  = 0.0;
    float  max_abs = 0.0f;
    for (size_t index = 0; index < a.size(); ++index) {
        dot += double(a[index]) * double(b[index]);
        norm_a += double(a[index]) * double(a[index]);
        norm_b += double(b[index]) * double(b[index]);
        max_abs = std::max(max_abs, std::fabs(a[index] - b[index]));
    }
    out_cosine  = dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    out_max_abs = max_abs;
}

// A short, non-silent tone, distinct in content from the pinned reference
// clip -- for assertion 4, the second "different reference clip" this test
// needs and the tree does not otherwise commit. Same shape
// tests/qwen3_tts_base_load_real.cpp's own make_tone already uses (220 Hz);
// a different frequency here (440 Hz, one octave up) is not load-bearing --
// any non-silent, in-bounds clip works -- but keeps the two audibly and
// numerically distinguishable inputs obviously distinct on inspection.
std::vector<float> make_tone(uint64_t frame_count, uint32_t sample_rate) {
    constexpr double   kFrequencyHz = 440.0;
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

// Synthesizes `text` at a fixed `seed` against `profile` and returns the raw
// PCM plus its channel count and sample rate.
bool synthesize_pcm(synth_model_t *         model,
                    synth_voice_profile_t * profile,
                    const char *            text,
                    uint64_t                seed,
                    std::vector<float> &    out_samples,
                    uint32_t &              out_channels,
                    uint32_t &              out_sample_rate) {
    synth_context_t * context = nullptr;
    if (synth_context_create(model, &context) != SYNTH_OK) {
        return false;
    }

    synth_request_t request;
    synth_request_init(&request, sizeof(request));
    request.input_kind    = SYNTH_INPUT_TEXT_UTF8;
    request.input_data    = text;
    request.input_count   = std::strlen(text);
    request.voice_profile = profile;
    request.seed          = seed;

    synth_audio_buffer_t * audio = nullptr;
    synth_result_t         result;
    synth_result_init(&result, sizeof(result));
    const bool ok = synth_synthesize_to_buffer(context, &request, &audio, &result) == SYNTH_OK && audio != nullptr;
    if (ok) {
        out_channels                = audio->channel_count;
        out_sample_rate             = audio->sample_rate;
        const uint64_t sample_count = audio->frame_count * audio->channel_count;
        out_samples.assign(audio->samples, audio->samples + sample_count);
    }
    if (audio != nullptr) {
        synth_audio_buffer_free(audio);
    }
    synth_context_free(context);
    return ok;
}

struct SeenDiagnostic {
    bool           seen   = false;
    synth_status_t status = SYNTH_OK;
    std::string    code;
};

void SYNTH_CALL record_diagnostic(void * user_data, const synth_diagnostic_t * diagnostic) {
    auto * target  = static_cast<SeenDiagnostic *>(user_data);
    target->seen   = true;
    target->status = diagnostic->status;
    target->code.assign(diagnostic->code != nullptr ? diagnostic->code : "", size_t(diagnostic->code_size));
}

synth_diagnostic_sink_t make_sink(SeenDiagnostic & target) {
    synth_diagnostic_sink_t sink;
    synth_diagnostic_sink_init(&sink, sizeof(sink));
    sink.emit      = record_diagnostic;
    sink.user_data = &target;
    return sink;
}

// A sink `synth_synthesize` must never write through: assertion 6's whole
// point is that the call is refused before any synthesis work happens.
synth_sink_result_t SYNTH_CALL refuse_audio(void * user_data, const synth_audio_chunk_t * chunk) {
    (void) chunk;
    *static_cast<bool *>(user_data) = true;
    return SYNTH_SINK_CONTINUE;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s <model.gguf> <reference.wav> <golden-x-vector.f32> <scratch-dir>\n", argv[0]);
        return 2;
    }
    const std::string model_path          = argv[1];
    const std::string reference_wav       = argv[2];
    const std::string golden_xvector_path = argv[3];
    const std::string scratch_dir         = argv[4];

    std::vector<float> pcm;
    uint32_t           sample_rate = 0;
    SYNTH_TEST_CHECK(read_wav_mono(reference_wav, pcm, sample_rate));
    SYNTH_TEST_CHECK(sample_rate == 24000);
    SYNTH_TEST_CHECK(!pcm.empty());

    synth_model_load_params_t load_params;
    synth_model_load_params_init(&load_params, sizeof(load_params));
    load_params.backend   = SYNTH_BACKEND_CPU;
    synth_model_t * model = nullptr;
    SYNTH_TEST_CHECK(synth_model_load(model_path.c_str(), &load_params, &model) == SYNTH_OK);

    synth_voice_profile_capabilities_t capabilities;
    synth_voice_profile_capabilities_init(&capabilities, sizeof(capabilities));
    SYNTH_TEST_CHECK(synth_model_get_voice_profile_capabilities(model, &capabilities) == SYNTH_OK);
    SYNTH_TEST_CHECK(sample_rate == capabilities.reference_target_sample_rate);

    // --- Assertion 1: the Profile's x-vector has enc_dim floats and matches
    // the oracle above Task 6's committed tolerance. The opaque public
    // Profile handle has no accessor for its raw floats (the same reason
    // tests/omnivoice_profile_test.cpp reaches its own family's Model
    // directly for its ref_rms/token-grid check), so this calls
    // Model::prepare_x_vector directly -- the exact function
    // src/voice-profile.cpp's create_from_reference dispatcher calls
    // internally -- on the SAME already-24kHz-mono pcm the public path below
    // uses unmodified.
    {
        std::unique_ptr<synth::qwen3tts::Model> family_model;
        SYNTH_TEST_CHECK(synth::qwen3tts::Model::load_cpu(model_path, family_model) == SYNTH_OK);

        synth::qwen3tts::XVectorEncoding encoding;
        const char *                     diagnostic_code    = nullptr;
        const char *                     diagnostic_message = nullptr;
        SYNTH_TEST_CHECK(family_model->prepare_x_vector(pcm, 0, encoding, diagnostic_code, diagnostic_message) ==
                         SYNTH_OK);

        const uint32_t enc_dim = family_model->hparams().speaker_encoder.enc_dim;
        SYNTH_TEST_CHECK(enc_dim > 0);
        SYNTH_TEST_CHECK(encoding.x_vector.size() == size_t(enc_dim));

        // Written unconditionally for post-mortem inspection, the same role
        // tests/qwen3_tts_xvector_driver.cpp's own committed output serves --
        // captured here instead of requiring a separate driver invocation.
        SYNTH_TEST_CHECK(write_f32(scratch_dir + "/x_vector.f32", encoding.x_vector));

        // Required, not optional. This was a skip until 2026-08-12: an absent
        // payload (or a size mismatch) printed to stderr and returned 0, so
        // the committed min_cosine below -- the only enforcement of the whole
        // mel/ECAPA/x-vector numerical claim anywhere -- was checked by
        // nothing on any machine that had not run the oracle dump, and the
        // suite was green. The payload is deliberately uncommitted
        // (build/goldens/... is ignored, "commit golden contracts, not golden
        // payloads"), so its absence is handled where docs/testing.md says to
        // handle it: at REGISTRATION. tests/CMakeLists.txt gates this test on
        // the same file existing, which makes an unmaterialized oracle a
        // missing test rather than a passing one. Here, having been given the
        // path, the only correct response to not being able to read it is to
        // fail.
        std::vector<float> golden;
        if (!read_f32(golden_xvector_path, golden)) {
            std::fprintf(stderr,
                         "qwen3-tts-clone-real: cannot read the oracle x-vector at %s. It is an uncommitted "
                         "dump artifact -- run scripts/dump_reference_qwen3_tts_speaker.py for case "
                         "base-xvector-en, then re-configure so this test registers again.\n",
                         golden_xvector_path.c_str());
            return 1;
        }
        SYNTH_TEST_CHECK(golden.size() == encoding.x_vector.size());

        double cosine  = 0.0;
        float  max_abs = 0.0f;
        compare_vectors(encoding.x_vector, golden, cosine, max_abs);
        // tests/tolerances/qwen3-tts.json,
        // variants.qwen3-tts-12hz-0-6b-base.profiles.BF16.stages.replay.
        // probes.speaker.x_vector.min_cosine -- Task 6, measured
        // 2026-08-12 on CPU (build/, Release) over base-xvector-en and
        // base-xvector-zh against this exact clip
        // (SYNTH_QWEN3_TTS_REFERENCE_WAV, models/qwen3-tts-reference-audio/
        // clone.wav by default). Committed observed_min_cosine there is
        // 0.9999953552841363 with observed_max_abs 0.021114349365234375; the
        // residual is bfloat16 quantization of the oracle's own tensors (that
        // file's own note), not gated on max_abs for the same reason the
        // talker probes above it are not: an x-vector is a direction consumed
        // by a dot product, not compared elementwise.
        constexpr double kMinCosine = 0.9999767764206815;
        std::fprintf(stderr, "qwen3-tts-clone-real: oracle cosine %.10f (gate %.10f), max_abs %.9g\n", cosine,
                     kMinCosine, double(max_abs));
        SYNTH_TEST_CHECK(cosine >= kMinCosine);
    }

    // The Profile every remaining assertion needs, through the public seam --
    // the first Voice Profile source this family implements for real.
    const synth_voice_reference_t reference =
        make_reference(pcm, sample_rate, capabilities.reference_target_channel_count);
    const synth_voice_reference_params_t params  = make_params(&reference, 1, nullptr);
    synth_voice_profile_t *              profile = nullptr;
    SYNTH_TEST_CHECK(synth_voice_profile_create_from_reference(model, &params, &profile) == SYNTH_OK);
    SYNTH_TEST_CHECK(profile != nullptr);

    // kText/kSeed are shared by assertions 2 through 5: a short sentence
    // (same bound-the-cost reasoning tests/omnivoice_profile_test.cpp's own
    // Serialized Profile round trip states for its own "Hi." text) and an
    // arbitrary fixed seed, so every comparison below is a comparison of the
    // SAME nominal request against different Profiles or Profile lifecycles.
    const char *   kText = "Hi.";
    const uint64_t kSeed = 7;

    // --- Assertion 2: synthesis with the real Profile produces finite,
    // non-silent PCM at 24 kHz.
    std::vector<float> pcm_first;
    uint32_t           channels_first = 0;
    uint32_t           rate_first     = 0;
    SYNTH_TEST_CHECK(synthesize_pcm(model, profile, kText, kSeed, pcm_first, channels_first, rate_first));
    SYNTH_TEST_CHECK(!pcm_first.empty());
    SYNTH_TEST_CHECK(rate_first == 24000);
    float max_abs_first = 0.0f;
    for (float sample : pcm_first) {
        SYNTH_TEST_CHECK(std::isfinite(sample));
        max_abs_first = std::max(max_abs_first, std::fabs(sample));
    }
    // Well below any plausible speech amplitude -- this only needs to catch
    // exact or near-exact digital silence (the shape a Profile that produced
    // a degenerate all-zero prompt substitution would emit), not bound
    // loudness.
    constexpr float kNonSilentThreshold = 1e-4f;
    SYNTH_TEST_CHECK(max_abs_first > kNonSilentThreshold);

    // --- Assertion 3: the same Profile and seed reproduce identical PCM
    // (request repeatability).
    std::vector<float> pcm_repeat;
    uint32_t           channels_repeat = 0;
    uint32_t           rate_repeat     = 0;
    SYNTH_TEST_CHECK(synthesize_pcm(model, profile, kText, kSeed, pcm_repeat, channels_repeat, rate_repeat));
    SYNTH_TEST_CHECK(channels_repeat == channels_first && rate_repeat == rate_first);
    SYNTH_TEST_CHECK(pcm_repeat.size() == pcm_first.size());
    SYNTH_TEST_CHECK(std::memcmp(pcm_repeat.data(), pcm_first.data(), pcm_first.size() * sizeof(float)) == 0);

    // --- Assertion 4: two different reference clips produce different PCM.
    // The one assertion that actually gates "cloned" as opposed to merely
    // "synthesized" -- a Profile the graph ignored would produce identical
    // audio here too, since kText and kSeed are unchanged from assertions
    // 2/3 above, and nothing else in this plan can see the difference.
    {
        const std::vector<float> tone =
            make_tone(capabilities.min_reference_frames_per_clip, capabilities.reference_target_sample_rate);
        const synth_voice_reference_t tone_reference = make_reference(tone, capabilities.reference_target_sample_rate,
                                                                      capabilities.reference_target_channel_count);
        const synth_voice_reference_params_t tone_params  = make_params(&tone_reference, 1, nullptr);
        synth_voice_profile_t *              tone_profile = nullptr;
        SYNTH_TEST_CHECK(synth_voice_profile_create_from_reference(model, &tone_params, &tone_profile) == SYNTH_OK);
        SYNTH_TEST_CHECK(tone_profile != nullptr);

        std::vector<float> pcm_tone;
        uint32_t           channels_tone = 0;
        uint32_t           rate_tone     = 0;
        SYNTH_TEST_CHECK(synthesize_pcm(model, tone_profile, kText, kSeed, pcm_tone, channels_tone, rate_tone));
        SYNTH_TEST_CHECK(!pcm_tone.empty());

        const size_t compared_length = std::min(pcm_tone.size(), pcm_first.size());
        const bool   differs = pcm_tone.size() != pcm_first.size() ||
                               std::memcmp(pcm_tone.data(), pcm_first.data(), compared_length * sizeof(float)) != 0;
        SYNTH_TEST_CHECK(differs);

        synth_voice_profile_free(tone_profile);
    }

    // --- Assertion 5: serialize -> free -> load -> synthesize reproduces
    // the same PCM the original Profile produced (pcm_first above). What
    // makes the Serialized Profile a real artifact rather than a
    // round-trippable blob.
    synth_voice_profile_serialize_params_t serialize_params;
    synth_voice_profile_serialize_params_init(&serialize_params, sizeof(serialize_params));
    synth_byte_buffer_t * bytes = nullptr;
    SYNTH_TEST_CHECK(synth_voice_profile_serialize(profile, &serialize_params, &bytes) == SYNTH_OK);
    SYNTH_TEST_CHECK(bytes != nullptr && bytes->data != nullptr && bytes->data_size > 0);

    // Freed before the reload below: the loaded Profile must stand on its
    // own reconstruction from the serialized bytes, not merely alias memory
    // the original Profile still owns.
    synth_voice_profile_free(profile);
    profile = nullptr;

    synth_voice_profile_load_params_t voice_load_params;
    synth_voice_profile_load_params_init(&voice_load_params, sizeof(voice_load_params));
    voice_load_params.data           = bytes->data;
    voice_load_params.data_size      = bytes->data_size;
    synth_voice_profile_t * reloaded = nullptr;
    SYNTH_TEST_CHECK(synth_voice_profile_load_from_memory(model, &voice_load_params, &reloaded) == SYNTH_OK);
    SYNTH_TEST_CHECK(reloaded != nullptr);
    synth_byte_buffer_free(bytes);

    std::vector<float> pcm_reloaded;
    uint32_t           channels_reloaded = 0;
    uint32_t           rate_reloaded     = 0;
    SYNTH_TEST_CHECK(synthesize_pcm(model, reloaded, kText, kSeed, pcm_reloaded, channels_reloaded, rate_reloaded));
    SYNTH_TEST_CHECK(channels_reloaded == channels_first && rate_reloaded == rate_first);
    SYNTH_TEST_CHECK(pcm_reloaded.size() == pcm_first.size());
    SYNTH_TEST_CHECK(std::memcmp(pcm_reloaded.data(), pcm_first.data(), pcm_first.size() * sizeof(float)) == 0);

    // --- Assertion 7: the transcript-assisted (ICL) mode, from the same clip
    // and the same nominal request. The transcript is the pinned clip's own,
    // as recorded in tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-base.
    // manifest.json for every reference case in it -- the one place in this
    // tree where an ICL synthesis runs against a reference whose transcript
    // is actually what the reference says.
    //
    // The assertion is the DIFFERENCE, not the success. Both Profiles come
    // from the same 24 kHz PCM, so create_icl_profile ran the same
    // encode_speaker_reference over the same samples and both carry
    // bit-identical x-vectors; kText and kSeed are unchanged from assertion
    // 2. The reference block -- the codes and the reference text ids the
    // dispatch passes only in the ICL arm -- is therefore the only thing left
    // that can move a sample. A dispatch that dropped either field would
    // reproduce pcm_first exactly.
    //
    // What this does NOT claim: that the ICL clone resembles the speaker more
    // closely than the x-vector one does. That is a Quality Evaluation claim
    // (ADR 0017, unrun), and this file's own header already states the same
    // limit for assertion 4.
    {
        const char * kTranscript =
            "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.";
        synth_voice_reference_t icl_reference =
            make_reference(pcm, sample_rate, capabilities.reference_target_channel_count);
        icl_reference.transcript      = kTranscript;
        icl_reference.transcript_size = std::strlen(kTranscript);

        const synth_voice_reference_params_t icl_params  = make_params(&icl_reference, 1, nullptr);
        synth_voice_profile_t *              icl_profile = nullptr;
        SYNTH_TEST_CHECK(synth_voice_profile_create_from_reference(model, &icl_params, &icl_profile) == SYNTH_OK);
        SYNTH_TEST_CHECK(icl_profile != nullptr);

        std::vector<float> pcm_icl;
        uint32_t           channels_icl = 0;
        uint32_t           rate_icl     = 0;
        SYNTH_TEST_CHECK(synthesize_pcm(model, icl_profile, kText, kSeed, pcm_icl, channels_icl, rate_icl));
        SYNTH_TEST_CHECK(!pcm_icl.empty());
        SYNTH_TEST_CHECK(channels_icl == channels_first && rate_icl == rate_first);
        float max_abs_icl = 0.0f;
        for (float sample : pcm_icl) {
            SYNTH_TEST_CHECK(std::isfinite(sample));
            max_abs_icl = std::max(max_abs_icl, std::fabs(sample));
        }
        SYNTH_TEST_CHECK(max_abs_icl > kNonSilentThreshold);

        const size_t compared_length = std::min(pcm_icl.size(), pcm_first.size());
        const bool   differs = pcm_icl.size() != pcm_first.size() ||
                               std::memcmp(pcm_icl.data(), pcm_first.data(), compared_length * sizeof(float)) != 0;
        SYNTH_TEST_CHECK(differs);

        synth_voice_profile_free(icl_profile);
    }

    // --- Assertion 6: a Profile presented to a second Loaded Model refuses
    // with SYNTH_ERR_UNSUPPORTED_VOICE and writes no audio. The second Model
    // is the same real Base package loaded a second time -- a different
    // in-memory synth_model_t*, identical bytes on disk -- the same
    // cross-model shape tests/qwen3_tts_base_load_real.cpp's own
    // check_cross_model_profile_refused already proves for a synthetic-tone
    // Profile; this proves it for the one this file actually cloned with.
    synth_model_t * other_model = nullptr;
    SYNTH_TEST_CHECK(synth_model_load(model_path.c_str(), &load_params, &other_model) == SYNTH_OK);
    synth_context_t * other_context = nullptr;
    SYNTH_TEST_CHECK(synth_context_create(other_model, &other_context) == SYNTH_OK);

    SeenDiagnostic          diagnostic;
    synth_diagnostic_sink_t sink = make_sink(diagnostic);

    bool               wrote_audio = false;
    synth_audio_sink_t audio_sink;
    synth_audio_sink_init(&audio_sink, sizeof(audio_sink));
    audio_sink.write     = refuse_audio;
    audio_sink.user_data = &wrote_audio;

    synth_request_t request;
    synth_request_init(&request, sizeof(request));
    request.input_kind    = SYNTH_INPUT_TEXT_UTF8;
    request.input_data    = kText;
    request.input_count   = std::strlen(kText);
    request.voice_profile = reloaded;
    request.diagnostics   = &sink;

    synth_result_t result;
    synth_result_init(&result, sizeof(result));
    SYNTH_TEST_CHECK(synth_synthesize(other_context, &request, &audio_sink, &result) == SYNTH_ERR_UNSUPPORTED_VOICE);
    SYNTH_TEST_CHECK(!wrote_audio);
    SYNTH_TEST_CHECK(diagnostic.seen);
    SYNTH_TEST_CHECK(diagnostic.status == SYNTH_ERR_UNSUPPORTED_VOICE);
    SYNTH_TEST_CHECK(diagnostic.code == "synthesis.voice_unsupported");

    synth_context_free(other_context);
    synth_model_free(other_model);
    synth_voice_profile_free(reloaded);
    synth_model_free(model);
    return 0;
}
