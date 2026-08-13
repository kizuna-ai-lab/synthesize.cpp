// Runs Model::prepare_x_vector over a real WAV file against a real package,
// so scripts/validate-qwen3-tts-replay.py --compare-x-vector can diff the
// result against Task 1's oracle dump (speaker/x_vector.f32).
//
// This is an adapter, not a test: it asserts nothing and reports what it
// produced, the same shape as qwen3_tts_mel_driver.cpp and
// qwen3_tts_replay_real.cpp. The comparison and its tolerance live in the
// Python validator, not here.
//
// Unlike the mel driver, this one needs a real package: the ECAPA graph is
// resolved against the package's own speaker_encoder weights
// (Model::load_cpu), not built by hand. Point it at the Base variant, not
// CustomVoice -- CustomVoice resolves no SpeakerEncoderWeights at all and
// prepare_x_vector refuses it with SYNTH_ERR_UNSUPPORTED_VOICE.
//
// The reference clip (models/qwen3-tts-reference-audio/clone.wav) is already
// 24 kHz mono float32, the rate prepare_x_vector's pcm_24k parameter assumes;
// this driver does not resample, the same restriction qwen3_tts_mel_driver.cpp
// states for the mel front end alone.

#include "arch/qwen3-tts/qwen3-tts.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

struct WavAudio {
    std::vector<float> samples;  // mono, already float in [-1, 1] range
    uint32_t           sample_rate = 0;
};

// Just enough of RIFF/WAVE to read the one shape this validator's reference
// clip actually is (PCM float32 or int16, mono or interleaved-and-averaged).
// Duplicated from qwen3_tts_mel_driver.cpp rather than shared: every small
// driver in this tree that reads a WAV carries its own copy (see also
// omnivoice_profile_test.cpp), and a WAV reader belongs in validation tooling
// rather than in src/ (no production entry point reads a WAV file today, and
// this driver does not add one).
bool read_wav(const std::string & path, WavAudio & out, std::string & error) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        error = "could not open " + path;
        return false;
    }
    // Every chunk_size in the loop below is an untrusted 32-bit field: it can
    // claim up to 4 GiB no matter how many bytes the file actually holds. The
    // file's own length is the only honest bound to size an allocation
    // against, so it is measured once, here, rather than trusted per chunk.
    const std::streamoff file_size = input.tellg();
    input.seekg(0, std::ios::beg);
    if (file_size < 0 || !input) {
        error = "could not measure " + path;
        return false;
    }

    char riff[4];
    input.read(riff, 4);
    uint32_t riff_size = 0;
    input.read(reinterpret_cast<char *>(&riff_size), 4);
    char wave[4];
    input.read(wave, 4);
    // The stream state before the contents: a file shorter than this 12-byte
    // header leaves `riff`/`wave` untouched, and comparing them would read
    // indeterminate stack bytes. The chunk loop below already checks after
    // every read; this header was the one place that did not.
    if (!input) {
        error = path + " is shorter than a 12-byte RIFF/WAVE header";
        return false;
    }
    if (std::memcmp(riff, "RIFF", 4) != 0 || std::memcmp(wave, "WAVE", 4) != 0) {
        error = path + " is not a RIFF/WAVE file";
        return false;
    }

    uint16_t          format_tag      = 0;
    uint16_t          channel_count   = 0;
    uint32_t          sample_rate     = 0;
    uint16_t          bits_per_sample = 0;
    bool              have_fmt        = false;
    std::vector<char> data_bytes;
    bool              have_data = false;

    while (input) {
        char     chunk_id[4];
        uint32_t chunk_size = 0;
        input.read(chunk_id, 4);
        if (!input) {
            break;
        }
        input.read(reinterpret_cast<char *>(&chunk_size), 4);
        if (!input) {
            break;
        }
        // Before anything is sized from it: a chunk cannot be longer than
        // what is left of the file. Without this the two allocations below
        // are sized from the claim alone, and the short read that follows
        // leaves the tail zero-filled -- fabricated samples a caller cannot
        // tell from real ones, on top of a 4 GiB allocation from a 4-byte
        // field.
        const std::streamoff position = input.tellg();
        if (position < 0 || uint64_t(chunk_size) > uint64_t(file_size - position)) {
            error = path + " declares a chunk larger than the file holds";
            return false;
        }
        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            // The floor before the allocation, not after it: the read below
            // is what the memcpy offsets depend on.
            if (chunk_size < 16) {
                error = "fmt chunk too small";
                return false;
            }
            std::vector<char> fmt(chunk_size);
            input.read(fmt.data(), std::streamsize(chunk_size));
            std::memcpy(&format_tag, fmt.data() + 0, 2);
            std::memcpy(&channel_count, fmt.data() + 2, 2);
            std::memcpy(&sample_rate, fmt.data() + 4, 4);
            std::memcpy(&bits_per_sample, fmt.data() + 14, 2);
            have_fmt = true;
        } else if (std::memcmp(chunk_id, "data", 4) == 0) {
            data_bytes.resize(chunk_size);
            input.read(data_bytes.data(), std::streamsize(chunk_size));
            have_data = true;
        } else {
            input.seekg(chunk_size, std::ios::cur);
        }
        if (chunk_size % 2 == 1) {
            input.seekg(1, std::ios::cur);  // chunks are word-aligned
        }
    }

    if (!have_fmt || !have_data) {
        error = path + " has no fmt/data chunk";
        return false;
    }
    if (channel_count == 0) {
        error = "zero channels";
        return false;
    }

    const size_t bytes_per_sample = size_t(bits_per_sample) / 8;
    if (bytes_per_sample == 0 || data_bytes.size() % (bytes_per_sample * channel_count) != 0) {
        error = "data chunk size is not a whole number of interleaved frames";
        return false;
    }
    const size_t frame_count = data_bytes.size() / (bytes_per_sample * channel_count);

    std::vector<float> mono(frame_count, 0.0f);
    for (size_t frame = 0; frame < frame_count; ++frame) {
        double sum = 0.0;
        for (uint16_t channel = 0; channel < channel_count; ++channel) {
            const size_t offset = (frame * channel_count + channel) * bytes_per_sample;
            double       value  = 0.0;
            if (format_tag == 3 && bits_per_sample == 32) {  // IEEE float32
                float sample;
                std::memcpy(&sample, data_bytes.data() + offset, 4);
                value = double(sample);
            } else if (format_tag == 1 && bits_per_sample == 16) {  // signed PCM16
                int16_t sample;
                std::memcpy(&sample, data_bytes.data() + offset, 2);
                value = double(sample) / 32768.0;
            } else {
                error = "unsupported WAV format_tag/bits_per_sample combination";
                return false;
            }
            sum += value;
        }
        mono[frame] = float(sum / double(channel_count));
    }

    out.samples     = std::move(mono);
    out.sample_rate = sample_rate;
    return true;
}

bool write_f32(const std::string & path, const std::vector<float> & values) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        return false;
    }
    if (!values.empty()) {
        output.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size() * sizeof(float)));
    }
    return bool(output);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <input.wav> <output-x_vector.f32>\n", argv[0]);
        return 2;
    }
    const std::string model_path(argv[1]);
    const std::string input_path(argv[2]);
    const std::string output_path(argv[3]);

    WavAudio    wav;
    std::string error;
    if (!read_wav(input_path, wav, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    // prepare_x_vector's pcm_24k parameter documents the package's declared
    // reference rate as an input contract, not something it resamples to --
    // the Audio Normalizer's job upstream of this call at the real seam
    // (src/audio-normalizer.h's validate_reference_format), which this
    // stand-alone driver bypasses the same way qwen3_tts_replay_real.cpp
    // bypasses the public C seam.
    if (wav.sample_rate != 24000) {
        std::fprintf(stderr,
                     "%s is %u Hz but prepare_x_vector expects the package's declared reference rate "
                     "(24000 Hz for this family) -- this driver does not resample\n",
                     input_path.c_str(), wav.sample_rate);
        return 1;
    }

    std::unique_ptr<synth::qwen3tts::Model> model;
    synth_status_t                          status = synth::qwen3tts::Model::load_cpu(model_path, model);
    if (status != SYNTH_OK) {
        std::fprintf(stderr, "load_cpu -> %d\n", int(status));
        return 1;
    }

    synth::qwen3tts::XVectorEncoding output;
    const char *                     diagnostic_code    = nullptr;
    const char *                     diagnostic_message = nullptr;
    status = model->prepare_x_vector(wav.samples, /*threads=*/0, output, diagnostic_code, diagnostic_message);
    if (status != SYNTH_OK) {
        std::fprintf(stderr, "prepare_x_vector -> %d (%s: %s)\n", int(status),
                     diagnostic_code != nullptr ? diagnostic_code : "",
                     diagnostic_message != nullptr ? diagnostic_message : "");
        return 1;
    }
    if (!write_f32(output_path, output.x_vector)) {
        std::fprintf(stderr, "could not write %s\n", output_path.c_str());
        return 1;
    }

    std::printf(
        "{\"wav_sample_rate\": %u, \"wav_samples\": %zu, \"ref_rms\": %.9g, \"mel_frames\": %llu, "
        "\"x_vector_size\": %zu}\n",
        wav.sample_rate, wav.samples.size(), double(output.ref_rms), (unsigned long long) output.mel_frames,
        output.x_vector.size());
    return 0;
}
