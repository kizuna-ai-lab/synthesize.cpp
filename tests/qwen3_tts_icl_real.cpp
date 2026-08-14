// Reference audio AND TRANSCRIPT in, cloned audio out: the transcript-assisted
// (ICL) clone path against the real Base package, its pinned reference clip and
// that clip's own transcript, through the public C seam -- with one family-level
// reach-around for the assertion the opaque Profile handle cannot answer.
//
// This is Stage 2 Plan 3's completion gate for the end-to-end half of the
// spec's Sec.8 ("ICL clone runs end to end"). The numerical halves belong
// elsewhere and are NOT re-derived here: the codec encoder's stage-wise
// comparison is scripts/validate-qwen3-tts-codec_encoder.py's -- which nothing
// registers with CTest, so assertion 1 below is the ONLY registered consumer of
// the two codec.rvq_reconstruction gates -- and the two-track prompt's is
// tests/qwen3_tts_icl_prompt_real.cpp's, which IS registered.
//
// WHAT AN END-TO-END TEST HERE CAN AND CANNOT ESTABLISH. Read this before
// adding an assertion, and before reading any assertion below as an alignment
// gate.
//
// Task 11 measured it on this port rather than arguing it, and it was
// re-measured against THIS file on 2026-08-14. With the ICL prompt's codec
// track rotated one frame -- talker-host.cpp's append_icl_block reading frame
// `index` where it must read `index - 1` -- the public seam still returns
// SYNTH_OK with plausible speech, and this test PASSES, at 19,200 PCM frames
// where correct is 24,960 (0.8000 s against 1.0400 s). The frame count is ON
// THE API SURFACE and no assertion reads it: the accurate phrasing is
// OBSERVABLE, NEVER DETECTABLE. (Under the same rotation
// synthesize-qwen3-tts-icl-prompt-real fails, codec 7.945e-01 against its
// 2.0e-2 gate -- run, not cited.) That this file passes is a property of what
// an end-to-end differential can decide, not a defect in the file:
//
//   * assertions 2-7 are differential or self-consistent (ICL against
//     x-vector, a run against its own repeat, a Profile against its own
//     serialized reload). A rotation moves every arm of every one of those
//     comparisons by the same amount, so none of them can see it.
//   * assertion 1 IS oracle-anchored, and it is anchored on the PROFILE's code
//     grid -- reference audio to [16, T] codes. Rotate THAT and assertion 1
//     fails. The prompt assembly that consumes the grid is a different stage,
//     and tests/qwen3_tts_icl_prompt_real.cpp is what pins it (the same
//     rotation moves its codec track from 2.232e-03 to 7.945e-01 against a
//     2.0e-2 gate).
//
// So: this file's oracle anchor covers waveform -> codes. Alignment is
// icl-prompt-real's. An assertion added here that would pass under a one-frame
// rotation is not an alignment check no matter what its comment says.
//
// THE INPUT IS THE WAV, AND THE ORACLE'S INPUT IS NOT. Measured here on
// 2026-08-14 and disclosed rather than absorbed: the oracle's own
// `codec_encoder/waveform.f32` is EXACTLY `bfloat16(clone.wav)` -- verified
// element for element, `torch.equal` -- because the reference model loads in
// bfloat16. The stage-wise validator hands the port that rounded waveform, so
// its committed observations isolate the port's arithmetic with the input
// held identical. This file hands the port the WAV, because that is what a
// caller hands it, so its comparison carries one extra input rounding
// (rel_absmax 2.954e-03 on the waveform, 0.76 of one bf16 unit). That is not
// free and the block below records what it costs.
//
// KNOWN UPSTREAM BEHAVIOUR THIS FILE IS BUILT AROUND, so that nobody
// "simplifies" a fixture into it. Pairing a reference clip with a transcript
// that does NOT match it makes synthesis run to the 2048-frame ceiling --
// about eight minutes, non-OK, zero audio, reported as `synthesis.output_limit`
// naming "a reference transcript that does not match its reference audio". It
// is upstream's pathology, measured on the PyTorch reference (Task 12), not a
// port defect. Every ICL Profile below therefore uses the clip's OWN
// transcript, read out of the oracle's own dump rather than retyped, and no
// assertion here perturbs a reference grid or pairs the clip with foreign text.
// The same input at different seeds also produces either near-zero output
// (4-12 frames) or the runaway, so no assertion here has a frame count for its
// expected value.
//
// The seven assertions, each its own block below:
//
//   1. a Profile prepared WITH a transcript reports CloneMode::Icl and carries
//      a [16, T] grid whose T is the clip's own frame arithmetic, every code
//      in [0, codebook_size); its dequantized reconstruction matches the
//      oracle's rvq_reconstruction.f32 inside the committed
//      codec.rvq_reconstruction.{semantic,acoustic} p95 gates; and preparing
//      the same clip twice yields the identical grid, through the production
//      `codes_equal`. The code agreement rate against codes/reference.i32 is
//      RECORDED and gates nothing -- see the block for why;
//   2. synthesis with it produces finite, non-silent PCM at 24 kHz;
//   3. the same Profile and seed produce identical PCM twice;
//   4. the same clip WITH and WITHOUT a transcript produces different PCM --
//      the assertion that gates "ICL" rather than merely "cloned";
//   5. serialize -> free -> load -> synthesize reproduces the ICL PCM exactly,
//      and the envelope carries the ICL kind's own markers;
//   6. an ICL Profile presented to a second Loaded Model refuses with
//      SYNTH_ERR_UNSUPPORTED_VOICE and writes no audio;
//   7. an x-vector Profile serializes to a Plan 2-shaped envelope -- the ICL
//      kind's keys and tensors ABSENT, the "x-vector" kind present -- and
//      still loads and still synthesizes.
//
// WHAT THIS FILE DOES NOT CLAIM. That the ICL clone resembles the speaker, or
// resembles them more closely than the x-vector clone does. That is a Quality
// Evaluation claim (ADR 0017, unrun); tests/qwen3_tts_clone_real.cpp's own
// header states the same limit for its own assertion 4.

#include "arch/qwen3-tts/codec-encoder-host.h"
#include "arch/qwen3-tts/profile.h"
#include "arch/qwen3-tts/qwen3-tts.h"
#include "arch/qwen3-tts/weights.h"
#include "synthesize.h"
#include "test-assert.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

// A minimal RIFF/WAVE reader for the two shapes this file ever reads: the
// pinned reference clip (mono, 24 kHz, IEEE float32) and, defensively, 16-bit
// PCM mono. A copy of tests/qwen3_tts_clone_real.cpp's, deliberately: these
// readers are per-test-binary in this tree (the mel driver carries its own
// too), and the bound that matters -- every chunk_size is an untrusted 32-bit
// field, so the file's own length is the only honest bound to size an
// allocation against -- is carried with the copy rather than left behind.
bool read_wav_mono(const std::string & path, std::vector<float> & pcm, uint32_t & sample_rate) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return false;
    }
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
        // The span is the chunk PLUS its pad byte, computed in uint64_t so that
        // chunk_size 0xFFFFFFFF cannot wrap +1 to zero and admit the single
        // largest claim there is.
        const uint64_t       chunk_span = uint64_t(chunk_size) + (chunk_size & 1u);
        const std::streamoff position   = file.tellg();
        if (position < 0 || chunk_span > uint64_t(file_size - position)) {
            return false;
        }
        constexpr uint32_t kConsumed = 16;
        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            // The floor BEFORE the fixed reads: those 16 bytes are read
            // unconditionally, so a `fmt ` declaring fewer would consume bytes
            // belonging to the next chunk and leave the walk out of step.
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

template <typename T> bool read_binary(const std::string & path, std::vector<T> & values) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return false;
    }
    const std::streamsize size = file.tellg();
    if (size < 0 || size % std::streamsize(sizeof(T)) != 0) {
        return false;
    }
    file.seekg(0);
    values.resize(static_cast<size_t>(size) / sizeof(T));
    return bool(file.read(reinterpret_cast<char *>(values.data()), size));
}

bool read_text(const std::string & path, std::string & text) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    text.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    return true;
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

// numpy.percentile's default ("linear") interpolation, over a copy that this
// function is free to sort. The statistic is transcribed from
// scripts/validate-qwen3-tts-codec_encoder.py rather than approximated: that
// script is where the committed gate was measured, and a nearest-rank
// percentile here would compare a different number against it.
double percentile_linear(std::vector<double> values, double percent) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double position = percent / 100.0 * double(values.size() - 1);
    const size_t low      = size_t(std::floor(position));
    const size_t high     = size_t(std::ceil(position));
    if (low == high) {
        return values[low];
    }
    return values[low] + (values[high] - values[low]) * (position - double(low));
}

// The p95 of the per-frame L2 deviation, relative to the REFERENCE frame's own
// L2 norm -- one branch of the [2, frames, projected] reconstruction. Same
// statistic, same operand order (`b` is the oracle and is the denominator) as
// scripts/validate-qwen3-tts-codec_encoder.py:243-252.
double reconstruction_p95_relative(const float * port, const float * oracle, size_t frames, size_t projected) {
    std::vector<double> relative;
    relative.reserve(frames);
    for (size_t frame = 0; frame < frames; ++frame) {
        double difference = 0.0;
        double reference  = 0.0;
        for (size_t column = 0; column < projected; ++column) {
            const double a = double(port[frame * projected + column]);
            const double b = double(oracle[frame * projected + column]);
            difference += (a - b) * (a - b);
            reference += b * b;
        }
        relative.push_back(std::sqrt(difference) / std::sqrt(reference));
    }
    return percentile_linear(relative, 95.0);
}

// Whether `needle` occurs verbatim in `bytes`. GGUF stores metadata keys and
// tensor names as length-prefixed UTF-8 with no compression, so a key's or a
// tensor's presence in an envelope is exactly the presence of its name as a
// byte substring -- which is what assertions 5 and 7 need and all they need.
bool contains_bytes(const uint8_t * bytes, size_t size, const char * needle) {
    const size_t needle_size = std::strlen(needle);
    if (needle_size == 0 || size < needle_size) {
        return false;
    }
    for (size_t offset = 0; offset + needle_size <= size; ++offset) {
        if (std::memcmp(bytes + offset, needle, needle_size) == 0) {
            return true;
        }
    }
    return false;
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

synth_voice_reference_params_t make_params(const synth_voice_reference_t * references, uint64_t count) {
    synth_voice_reference_params_t params;
    synth_voice_reference_params_init(&params, sizeof(params));
    params.references       = references;
    params.reference_count  = count;
    params.reference_stride = sizeof(synth_voice_reference_t);
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

bool same_pcm(const std::vector<float> & left, const std::vector<float> & right) {
    return left.size() == right.size() && std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) == 0;
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

// A sink `synth_synthesize` must never write through: assertion 6's whole point
// is that the call is refused before any synthesis work happens.
synth_sink_result_t SYNTH_CALL refuse_audio(void * user_data, const synth_audio_chunk_t * chunk) {
    (void) chunk;
    *static_cast<bool *>(user_data) = true;
    return SYNTH_SINK_CONTINUE;
}

// Serializes `profile` into `bytes`, which the caller owns and must free.
bool serialize_profile(synth_voice_profile_t * profile, synth_byte_buffer_t *& bytes) {
    synth_voice_profile_serialize_params_t serialize_params;
    synth_voice_profile_serialize_params_init(&serialize_params, sizeof(serialize_params));
    bytes = nullptr;
    return synth_voice_profile_serialize(profile, &serialize_params, &bytes) == SYNTH_OK && bytes != nullptr &&
           bytes->data != nullptr && bytes->data_size > 0;
}

bool load_profile(synth_model_t * model, const synth_byte_buffer_t * bytes, synth_voice_profile_t *& out_profile) {
    synth_voice_profile_load_params_t load_params;
    synth_voice_profile_load_params_init(&load_params, sizeof(load_params));
    load_params.data      = bytes->data;
    load_params.data_size = bytes->data_size;
    out_profile           = nullptr;
    return synth_voice_profile_load_from_memory(model, &load_params, &out_profile) == SYNTH_OK &&
           out_profile != nullptr;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 9) {
        std::fprintf(stderr,
                     "usage: %s <model.gguf> <reference.wav> <transcript.txt> <rvq_reconstruction.f32> "
                     "<reference-codes.i32> <semantic-p95> <acoustic-p95> <scratch-dir>\n",
                     argv[0]);
        return 2;
    }
    const std::string model_path        = argv[1];
    const std::string reference_wav     = argv[2];
    const std::string transcript_path   = argv[3];
    const std::string oracle_recon_path = argv[4];
    const std::string oracle_codes_path = argv[5];
    const double      semantic_p95_gate = std::strtod(argv[6], nullptr);
    const double      acoustic_p95_gate = std::strtod(argv[7], nullptr);
    const std::string scratch_dir       = argv[8];

    // Both gates arrive from tests/tolerances/qwen3-tts.json through
    // tests/CMakeLists.txt, never as a literal here -- the same rule
    // tests/qwen3_tts_icl_prompt_real.cpp's own argv[2] follows, and for the
    // same reason: a tolerance restated in a test file is a description of the
    // test rather than its source, and editing the JSON would change nothing
    // that runs. A bound that did not survive the trip is refused, so that a
    // mis-wired registration cannot become a test that gates on 0.
    SYNTH_TEST_CHECK(semantic_p95_gate > 0.0 && semantic_p95_gate < 1.0);
    SYNTH_TEST_CHECK(acoustic_p95_gate > 0.0 && acoustic_p95_gate < 1.0);

    std::vector<float> pcm;
    uint32_t           sample_rate = 0;
    SYNTH_TEST_CHECK(read_wav_mono(reference_wav, pcm, sample_rate));
    SYNTH_TEST_CHECK(sample_rate == 24000);
    SYNTH_TEST_CHECK(!pcm.empty());

    // The transcript comes out of the oracle's own dump for the case whose
    // codes and reconstruction this file compares against
    // (alignment.json's inputs.ref_text, written to a fixture by
    // tests/CMakeLists.txt). Retyping it here would let the transcript and the
    // artifacts it must match drift apart -- and a transcript that does not
    // match its audio is the eight-minute runaway this file's header records.
    std::string transcript;
    SYNTH_TEST_CHECK(read_text(transcript_path, transcript));
    SYNTH_TEST_CHECK(!transcript.empty());

    synth_model_load_params_t load_params;
    synth_model_load_params_init(&load_params, sizeof(load_params));
    load_params.backend   = SYNTH_BACKEND_CPU;
    synth_model_t * model = nullptr;
    SYNTH_TEST_CHECK(synth_model_load(model_path.c_str(), &load_params, &model) == SYNTH_OK);

    synth_voice_profile_capabilities_t capabilities;
    synth_voice_profile_capabilities_init(&capabilities, sizeof(capabilities));
    SYNTH_TEST_CHECK(synth_model_get_voice_profile_capabilities(model, &capabilities) == SYNTH_OK);
    SYNTH_TEST_CHECK(sample_rate == capabilities.reference_target_sample_rate);
    // The seam this whole plan opened. Task 10 flipped it from UNSUPPORTED to
    // OPTIONAL in the same change that landed ICL; asserted here so that a
    // build advertising the capability without this file's remaining six
    // assertions holding is not a passing configuration.
    //
    // MASKED, and measured to be: reverting weights.cpp's assignment to
    // UNSUPPORTED fails this line AND
    // tests/qwen3_tts_base_load_real.cpp:107, which asserts the same
    // enumerator. Deleting this line alone changes nothing that test does not
    // already report.
    SYNTH_TEST_CHECK(capabilities.reference_transcript == SYNTH_REQUIREMENT_OPTIONAL);

    // --- Assertion 1: the prepared ICL Profile, against the oracle.
    //
    // The public Profile handle is opaque and has no accessor for its code
    // grid, so this block calls the family's own create_icl_profile -- the
    // exact function src/voice-profile.cpp's create_from_reference dispatcher
    // calls, with the same arguments it passes -- on the SAME already-24kHz
    // mono pcm the public path below uses unmodified. That is the same
    // reach-around tests/qwen3_tts_clone_real.cpp makes for its own assertion 1
    // and tests/omnivoice_profile_test.cpp for its token-grid check.
    {
        std::unique_ptr<synth::qwen3tts::Model> family_model;
        SYNTH_TEST_CHECK(synth::qwen3tts::Model::load_cpu(model_path, family_model) == SYNTH_OK);
        const synth::qwen3tts::HParams & hparams = family_model->hparams();

        std::vector<int32_t> reference_text_ids;
        SYNTH_TEST_CHECK(family_model->tokenize_reference_transcript(transcript, reference_text_ids) == SYNTH_OK);
        SYNTH_TEST_CHECK(!reference_text_ids.empty());

        std::shared_ptr<const synth::qwen3tts::IclProfile> icl;
        const char *                                       diagnostic_code    = nullptr;
        const char *                                       diagnostic_message = nullptr;
        SYNTH_TEST_CHECK(synth::qwen3tts::create_icl_profile(hparams, family_model->speaker_encoder_weights(),
                                                             family_model->codec_encoder_weights(), pcm, transcript,
                                                             reference_text_ids, std::string(), 0, icl, diagnostic_code,
                                                             diagnostic_message) == SYNTH_OK);
        SYNTH_TEST_CHECK(icl != nullptr);

        // The kind, at the layer that decides it. D4 fixes the mode at
        // preparation: a transcript names the transcript-assisted mode, and a
        // Profile that came back meaning the other one would be the silent
        // downgrade the whole D4 ruling exists to prevent.
        SYNTH_TEST_CHECK(icl->speaker.mode == synth::qwen3tts::CloneMode::Icl);
        SYNTH_TEST_CHECK(icl->reference_text_ids == reference_text_ids);
        SYNTH_TEST_CHECK(!icl->speaker.x_vector.empty());

        // T, from two independent sources, neither of them the geometry walker
        // that produced it. The package's own declared hop gives the ceiling
        // divide; the oracle's committed grid gives the frame count upstream
        // actually produced for this clip. A port that agreed with itself and
        // with neither would pass a check written against
        // codec_encoder_geometry.
        const uint64_t groups = hparams.codec.decoder.quantizer_count;
        SYNTH_TEST_CHECK(groups > 1);
        SYNTH_TEST_CHECK(hparams.codec.hop_length > 0);
        const uint64_t frames_from_hop = (pcm.size() + hparams.codec.hop_length - 1) / hparams.codec.hop_length;

        std::vector<int32_t> oracle_codes;
        if (!read_binary(oracle_codes_path, oracle_codes)) {
            std::fprintf(stderr,
                         "qwen3-tts-icl-real: cannot read the oracle reference codes at %s. It is an uncommitted "
                         "dump artifact -- run scripts/dump_reference_qwen3_tts_base.py for case base-icl-en, then "
                         "re-configure so this test registers again.\n",
                         oracle_codes_path.c_str());
            return 1;
        }
        SYNTH_TEST_CHECK(oracle_codes.size() % size_t(groups) == 0);
        const uint64_t frames_from_oracle = uint64_t(oracle_codes.size()) / groups;

        SYNTH_TEST_CHECK(icl->groups == groups);
        SYNTH_TEST_CHECK(icl->frames == frames_from_hop);
        SYNTH_TEST_CHECK(icl->frames == frames_from_oracle);
        SYNTH_TEST_CHECK(icl->codes.size() == size_t(groups) * size_t(icl->frames));

        const int32_t codebook_size = int32_t(hparams.codec.decoder.codebook_size);
        SYNTH_TEST_CHECK(codebook_size == 2048);
        for (int32_t code : icl->codes) {
            SYNTH_TEST_CHECK(code >= 0 && code < codebook_size);
        }

        // Preparing the same clip twice yields the identical grid. This runs
        // the whole encode chain a SECOND time through the other production
        // entry point (Model::prepare_codec_reference, which is what the
        // committed codec-encoder driver drives) and compares through
        // codec-encoder-host.h's own `codes_equal` -- a production comparison
        // with three callers rather than a helper local to this file, so
        // inverting it fails all three. The port against itself, where
        // exactness is free and mandatory.
        synth::qwen3tts::CodecEncoding encoding;
        SYNTH_TEST_CHECK(family_model->prepare_codec_reference(pcm, 0, encoding, diagnostic_code, diagnostic_message) ==
                         SYNTH_OK);
        SYNTH_TEST_CHECK(synth::qwen3tts::codes_equal(encoding, icl->codes.data(), int64_t(icl->frames)));

        // THE ORACLE CHECK IS THE RECONSTRUCTION, NOT THE CODES. The design's
        // fourth erratum (2026-08-13): the oracle's RVQ codebook is bfloat16
        // and the converter's is float32, upstream disagrees with ITSELF
        // depending on whether `dtype` is passed at load, and base-icl-en is
        // one of the cases the two tables disagree on -- an equality gate on
        // codes/reference.i32 would test a load-time keyword argument rather
        // than this port. What is compared instead is the discrete decision
        // converted back into the continuous quantity it was rounded from,
        // per branch, at the p95 of the per-frame relative L2, against the
        // gates tests/tolerances/qwen3-tts.json commits under
        // codec.rvq_reconstruction.{semantic,acoustic}.
        //
        // A PERCENTILE AND NEVER A MAXIMUM, and never averaged across the two
        // branches: one flipped code displaces the reconstruction by about a
        // full codebook-row separation, so no bf16-scale maximum survives, and
        // the two branches are an order of magnitude apart in rms. Both facts
        // are that file's own, measured; this test transcribes the statistic
        // rather than inventing a second one.
        //
        // THE GATE IS ENFORCED ON A DIFFERENT INPUT THAN IT WAS CALIBRATED ON,
        // and the difference is one bf16 rounding of the waveform (see this
        // file's header). Measured on 2026-08-14, base-icl-en, CPU:
        //
        //   branch    | fed waveform.f32 (committed) | fed clone.wav (here)
        //   semantic  | 0.004103                     | 0.004033   gate 2.0e-2
        //   acoustic  | 0.2491                       | 0.304719   gate 5.0e-1
        //
        // So the acoustic branch spends 22% more of its budget here and clears
        // by 1.64x rather than the committed 2.0x. Recorded in
        // tests/tolerances/qwen3-tts.json beside both probes, NOT accommodated:
        // nothing was widened, and the injected fault those cells record (1.741
        // acoustic, 1.601 semantic) still sits far above the unchanged
        // thresholds. The alternative -- feeding this test the oracle's
        // bfloat16-rounded waveform so the two tiers compare the same
        // quantity -- was rejected because no caller hands the library that,
        // and this tier's whole claim is "reference audio in".
        //
        // The recorded code agreement below moves for the same reason and by
        // the same mechanism: 796 of 1616 differ from the WAV where 760 differ
        // from waveform.f32.
        const size_t projected = size_t(encoding.projected);
        SYNTH_TEST_CHECK(projected > 0);
        SYNTH_TEST_CHECK(encoding.reconstruction.size() == 2u * size_t(icl->frames) * projected);

        std::vector<float> oracle_reconstruction;
        if (!read_binary(oracle_recon_path, oracle_reconstruction)) {
            std::fprintf(stderr,
                         "qwen3-tts-icl-real: cannot read the oracle reconstruction at %s. It is an uncommitted "
                         "dump artifact -- run scripts/dump_reference_qwen3_tts_codec_encoder.py for case "
                         "base-icl-en, then re-configure so this test registers again.\n",
                         oracle_recon_path.c_str());
            return 1;
        }
        SYNTH_TEST_CHECK(oracle_reconstruction.size() == encoding.reconstruction.size());

        SYNTH_TEST_CHECK(write_f32(scratch_dir + "/rvq_reconstruction.f32", encoding.reconstruction));

        const size_t branch_stride = size_t(icl->frames) * projected;
        const double semantic_p95  = reconstruction_p95_relative(
            encoding.reconstruction.data(), oracle_reconstruction.data(), size_t(icl->frames), projected);
        const double acoustic_p95 =
            reconstruction_p95_relative(encoding.reconstruction.data() + branch_stride,
                                        oracle_reconstruction.data() + branch_stride, size_t(icl->frames), projected);
        std::fprintf(stderr,
                     "qwen3-tts-icl-real: reconstruction p95 relative L2 -- semantic %.6g (gate %.3g), "
                     "acoustic %.6g (gate %.3g) over %llu frames x %zu\n",
                     semantic_p95, semantic_p95_gate, acoustic_p95, acoustic_p95_gate, (unsigned long long) icl->frames,
                     projected);
        SYNTH_TEST_CHECK(semantic_p95 <= semantic_p95_gate);
        SYNTH_TEST_CHECK(acoustic_p95 <= acoustic_p95_gate);

        // RECORDED, GATING NOTHING -- and printed on a passing run as well as a
        // failing one, which is the whole point: the committed rate is ~53%
        // (the bf16 codebook's doing, with port-vs-upstream-f32 at 100.000%),
        // so a regression that took it to a few per cent would be invisible to
        // every gate above and visible here. Do not turn this into an
        // assertion; the erratum removed exactly that.
        size_t differing = 0;
        for (size_t index = 0; index < icl->codes.size(); ++index) {
            differing += size_t(icl->codes[index] != oracle_codes[index]);
        }
        std::fprintf(stderr,
                     "qwen3-tts-icl-real: code agreement against the bf16 oracle %.3f%% (%zu of %zu differ) "
                     "-- RECORDED, gating nothing (the design's fourth erratum)\n",
                     100.0 * (1.0 - double(differing) / double(icl->codes.size())), differing, icl->codes.size());
    }

    // kText/kSeed are shared by assertions 2 through 7, so that every
    // comparison below is the SAME nominal request against different Profiles
    // or Profile lifecycles. A short sentence bounds the cost of five
    // autoregressive passes over the real package.
    const char *   kText = "Hi.";
    const uint64_t kSeed = 7;

    // The two Profiles, through the public seam: the same clip, the same
    // normalization, differing only in whether a transcript was supplied. That
    // difference is what selects the mode (D4), and it is what assertion 4
    // reads.
    synth_voice_reference_t icl_reference =
        make_reference(pcm, sample_rate, capabilities.reference_target_channel_count);
    icl_reference.transcript                         = transcript.c_str();
    icl_reference.transcript_size                    = transcript.size();
    const synth_voice_reference_params_t icl_params  = make_params(&icl_reference, 1);
    synth_voice_profile_t *              icl_profile = nullptr;
    SYNTH_TEST_CHECK(synth_voice_profile_create_from_reference(model, &icl_params, &icl_profile) == SYNTH_OK);
    SYNTH_TEST_CHECK(icl_profile != nullptr);

    const synth_voice_reference_t x_vector_reference =
        make_reference(pcm, sample_rate, capabilities.reference_target_channel_count);
    const synth_voice_reference_params_t x_vector_params  = make_params(&x_vector_reference, 1);
    synth_voice_profile_t *              x_vector_profile = nullptr;
    SYNTH_TEST_CHECK(synth_voice_profile_create_from_reference(model, &x_vector_params, &x_vector_profile) == SYNTH_OK);
    SYNTH_TEST_CHECK(x_vector_profile != nullptr);

    // --- Assertion 2: synthesis with the ICL Profile produces finite,
    // non-silent PCM at 24 kHz.
    std::vector<float> pcm_icl;
    uint32_t           channels_icl = 0;
    uint32_t           rate_icl     = 0;
    SYNTH_TEST_CHECK(synthesize_pcm(model, icl_profile, kText, kSeed, pcm_icl, channels_icl, rate_icl));
    SYNTH_TEST_CHECK(!pcm_icl.empty());
    SYNTH_TEST_CHECK(rate_icl == 24000);
    SYNTH_TEST_CHECK(channels_icl == 1);
    float max_abs_icl = 0.0f;
    for (float sample : pcm_icl) {
        SYNTH_TEST_CHECK(std::isfinite(sample));
        max_abs_icl = std::max(max_abs_icl, std::fabs(sample));
    }
    // Well below any plausible speech amplitude: this only needs to catch exact
    // or near-exact digital silence, not bound loudness.
    constexpr float kNonSilentThreshold = 1e-4f;
    SYNTH_TEST_CHECK(max_abs_icl > kNonSilentThreshold);

    // --- Assertion 3: the same Profile and seed reproduce identical PCM.
    std::vector<float> pcm_icl_repeat;
    uint32_t           channels_icl_repeat = 0;
    uint32_t           rate_icl_repeat     = 0;
    SYNTH_TEST_CHECK(
        synthesize_pcm(model, icl_profile, kText, kSeed, pcm_icl_repeat, channels_icl_repeat, rate_icl_repeat));
    SYNTH_TEST_CHECK(channels_icl_repeat == channels_icl && rate_icl_repeat == rate_icl);
    SYNTH_TEST_CHECK(same_pcm(pcm_icl_repeat, pcm_icl));

    // --- Assertion 4: the same clip WITH and WITHOUT a transcript produces
    // different PCM. The assertion that gates "ICL" as opposed to "cloned".
    //
    // Both Profiles come from the same 24 kHz PCM, so both ran the same
    // encode_speaker_reference over the same samples and carry bit-identical
    // x-vectors; kText and kSeed are unchanged. The reference block -- the
    // codes and the reference text ids the dispatch passes only in the ICL arm
    // (src/synthesize.cpp's CloneMode::Icl case) -- is the only thing left that
    // can move a sample, so a dispatch that dropped either field would
    // reproduce the x-vector run exactly.
    //
    // MASKED, and measured to be, which corrects the plan's own wording that
    // "nothing else in this plan sees that from the outside". Deleting the
    // three assignments in src/synthesize.cpp's CloneMode::Icl arm fails this
    // line AND tests/qwen3_tts_clone_real.cpp:624 -- Task 11 added an
    // equivalent ICL-against-x-vector differential there as a stopgap, in the
    // x-vector plan's own gate, because this file did not exist yet, and its
    // own comment says so. Both were run under the deletion on 2026-08-14 and
    // both failed. Deleting THIS check alone changes nothing that test does
    // not already report; what this file adds around it is the oracle anchor
    // in assertion 1 and the envelope checks in assertions 5 and 7, none of
    // which clone-real makes. If clone-real's assertion 7 is ever retired now
    // that the ICL path has its own end-to-end gate, this becomes the sole
    // owner of the claim -- retire it there, not here.
    std::vector<float> pcm_x_vector;
    uint32_t           channels_x_vector = 0;
    uint32_t           rate_x_vector     = 0;
    SYNTH_TEST_CHECK(
        synthesize_pcm(model, x_vector_profile, kText, kSeed, pcm_x_vector, channels_x_vector, rate_x_vector));
    SYNTH_TEST_CHECK(!pcm_x_vector.empty());
    SYNTH_TEST_CHECK(channels_x_vector == channels_icl && rate_x_vector == rate_icl);
    SYNTH_TEST_CHECK(!same_pcm(pcm_x_vector, pcm_icl));

    // Both durations, on the record. They are the quantity Task 11 showed to be
    // observable-but-never-detected -- a one-frame codec-track rotation moves
    // the ICL figure and no assertion in this file reads it (see the header).
    // Printed rather than asserted, deliberately: the same input at different
    // seeds gives either near-zero output or the 2048-frame runaway, so a frame
    // count is not an expected value this suite may pin.
    std::fprintf(stderr, "qwen3-tts-icl-real: PCM frames -- icl %zu (%.4f s), x-vector %zu (%.4f s)\n", pcm_icl.size(),
                 double(pcm_icl.size()) / double(rate_icl), pcm_x_vector.size(),
                 double(pcm_x_vector.size()) / double(rate_x_vector));

    // --- Assertion 5: serialize -> free -> load -> synthesize reproduces the
    // ICL PCM exactly, and the envelope is the ICL kind's.
    //
    // The envelope check is not decoration. Task 8's review recorded the exact
    // defect it closes: serialize_x_vector_profile does not consult `mode`, and
    // composition makes an XVectorProfile{mode==Icl} constructible through
    // IclProfile::speaker, so a serialize dispatch that picked the wrong writer
    // would emit a perfectly valid, digest-correct kind="x-vector" envelope
    // that loads and synthesizes as an x-vector Profile -- the silent downgrade
    // D4 exists to prevent. A PCM comparison cannot see that (it would compare
    // an x-vector run against an x-vector run); the kind and the two ICL-only
    // tensors can.
    synth_byte_buffer_t * icl_bytes = nullptr;
    SYNTH_TEST_CHECK(serialize_profile(icl_profile, icl_bytes));
    {
        const auto * bytes = static_cast<const uint8_t *>(icl_bytes->data);
        const size_t size  = size_t(icl_bytes->data_size);
        SYNTH_TEST_CHECK(contains_bytes(bytes, size, "profile.codes"));
        SYNTH_TEST_CHECK(contains_bytes(bytes, size, "profile.reference_text_ids"));
        SYNTH_TEST_CHECK(contains_bytes(bytes, size, "synthesize.voice_profile.code_groups"));
        SYNTH_TEST_CHECK(contains_bytes(bytes, size, "synthesize.voice_profile.reference_frames"));
        // The kind VALUE. "x-vector" occurs in an x-vector envelope only as
        // that value: the shared speaker tensor is "profile.x_vector", with an
        // underscore, so this substring cannot be satisfied by it.
        SYNTH_TEST_CHECK(!contains_bytes(bytes, size, "x-vector"));
    }

    // Freed before the reload: the loaded Profile must stand on its own
    // reconstruction from the serialized bytes rather than alias memory the
    // original still owns.
    synth_voice_profile_free(icl_profile);
    icl_profile = nullptr;

    synth_voice_profile_t * icl_reloaded = nullptr;
    SYNTH_TEST_CHECK(load_profile(model, icl_bytes, icl_reloaded));
    synth_byte_buffer_free(icl_bytes);

    std::vector<float> pcm_icl_reloaded;
    uint32_t           channels_icl_reloaded = 0;
    uint32_t           rate_icl_reloaded     = 0;
    SYNTH_TEST_CHECK(
        synthesize_pcm(model, icl_reloaded, kText, kSeed, pcm_icl_reloaded, channels_icl_reloaded, rate_icl_reloaded));
    SYNTH_TEST_CHECK(channels_icl_reloaded == channels_icl && rate_icl_reloaded == rate_icl);
    SYNTH_TEST_CHECK(same_pcm(pcm_icl_reloaded, pcm_icl));

    // --- Assertion 7: an x-vector Profile still serializes to a Plan 2-shaped
    // envelope, and still loads and still synthesizes.
    //
    // WHAT MAKES THIS A COMPATIBILITY CHECK RATHER THAN A ROUND TRIP. A writer
    // and a reader changed together stay agreeing with each other, so a cycle
    // through both cannot notice that "a Plan 2 profile" has been quietly
    // redefined -- Task 8's review said exactly that about its own round trip.
    // The half that a future writer change cannot compensate for is pinned at
    // the unit tier, where tests/qwen3_tts_profile_test.cpp holds 800 bytes
    // produced by the PLAN 2 WRITER at commit 50912d9 and runs only the reader
    // over them. This tier adds the half that needs the real package: that the
    // ICL kind's arrival did not put ICL keys or ICL tensors into an x-vector
    // envelope, and that such an envelope still synthesizes against a real
    // 2.5 GB Model. Neither is expressible on the synthetic fixture, and the
    // committed byte array cannot be regenerated for this package without a
    // Plan 2 build.
    synth_byte_buffer_t * x_vector_bytes = nullptr;
    SYNTH_TEST_CHECK(serialize_profile(x_vector_profile, x_vector_bytes));
    {
        const auto * bytes = static_cast<const uint8_t *>(x_vector_bytes->data);
        const size_t size  = size_t(x_vector_bytes->data_size);
        SYNTH_TEST_CHECK(contains_bytes(bytes, size, "x-vector"));
        SYNTH_TEST_CHECK(contains_bytes(bytes, size, "profile.x_vector"));
        SYNTH_TEST_CHECK(!contains_bytes(bytes, size, "profile.codes"));
        SYNTH_TEST_CHECK(!contains_bytes(bytes, size, "profile.reference_text_ids"));
        SYNTH_TEST_CHECK(!contains_bytes(bytes, size, "synthesize.voice_profile.code_groups"));
        SYNTH_TEST_CHECK(!contains_bytes(bytes, size, "synthesize.voice_profile.reference_frames"));
    }

    synth_voice_profile_free(x_vector_profile);
    x_vector_profile = nullptr;

    synth_voice_profile_t * x_vector_reloaded = nullptr;
    SYNTH_TEST_CHECK(load_profile(model, x_vector_bytes, x_vector_reloaded));
    synth_byte_buffer_free(x_vector_bytes);

    std::vector<float> pcm_x_vector_reloaded;
    uint32_t           channels_x_vector_reloaded = 0;
    uint32_t           rate_x_vector_reloaded     = 0;
    SYNTH_TEST_CHECK(synthesize_pcm(model, x_vector_reloaded, kText, kSeed, pcm_x_vector_reloaded,
                                    channels_x_vector_reloaded, rate_x_vector_reloaded));
    SYNTH_TEST_CHECK(channels_x_vector_reloaded == channels_x_vector && rate_x_vector_reloaded == rate_x_vector);
    SYNTH_TEST_CHECK(same_pcm(pcm_x_vector_reloaded, pcm_x_vector));

    // --- Assertion 6: an ICL Profile presented to a second Loaded Model
    // refuses with SYNTH_ERR_UNSUPPORTED_VOICE and writes no audio.
    //
    // MASKING, DISCLOSED. The rule this fires is
    // `prepared.voice_profile->model != context->model` in src/synthesize.cpp,
    // which sits ABOVE the CloneMode switch -- so it is shared with
    // tests/qwen3_tts_clone_real.cpp's own assertion 6 (an x-vector Profile)
    // and tests/qwen3_tts_base_load_real.cpp's check_cross_model_profile_refused
    // (a tone Profile). Deleting the rule fails all three; deleting THIS check
    // alone changes nothing the other two do not already report. It is kept
    // because the refusal has to hold for a Profile carrying an ICL payload
    // too, and a future reader must know that this is where the ICL kind is
    // covered -- but it is not independent evidence, and a report that counted
    // it as such would be wrong.
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
    request.voice_profile = icl_reloaded;
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
    synth_voice_profile_free(icl_reloaded);
    synth_voice_profile_free(x_vector_reloaded);
    synth_model_free(model);
    return 0;
}
