// Runs compute_log_mel over a real WAV file and writes the log-mel it
// produces, in the oracle's own on-disk layout, for
// scripts/validate-qwen3-tts-replay.py's --compare-mel mode to diff against
// Task 1's speaker/mel.f32.
//
// This is an adapter, not a test: it asserts nothing and reports what it
// produced, the same shape as qwen3_tts_replay_real.cpp. The comparison and
// its tolerance live in the Python validator, not here.
//
// The reference clip (models/qwen3-tts-reference-audio/clone.wav) is already
// 24 kHz mono float32 -- confirmed directly (soundfile: samplerate 24000 Hz,
// channels 1, subtype FLOAT) -- so upstream's librosa.load(sr=None) applied
// no resampling to produce the 193920-sample array mel_spectrogram actually
// saw. This driver therefore only needs to parse the WAV's samples, not
// resample them; a WAV reader belongs here, in validation tooling, and not in
// src/ (docs/testing.md's port-validation split -- no production entry point
// reads a WAV file today, and this driver does not add one).
#include "arch/qwen3-tts/mel.h"
#include "arch/qwen3-tts/weights.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct WavAudio {
    std::vector<float> samples;  // mono, already float in [-1, 1] range
    uint32_t           sample_rate = 0;
};

// Just enough of RIFF/WAVE to read the one shape this validator's reference
// clip actually is (PCM float32 or int16, mono or interleaved-and-averaged).
// Not a general-purpose decoder: no compressed formats, no extensible fmt
// chunk fields beyond what selects the sample layout.
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
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <input.wav> <output-mel.f32>\n", argv[0]);
        return 1;
    }
    const std::string input_path  = argv[1];
    const std::string output_path = argv[2];

    // The package's own pinned speaker-encoder parameters
    // (qwen3-tts-12hz-0-6b-base's speaker_encoder metadata), not read from a
    // GGUF here: this driver's job is the mel front end alone, and Task 2's
    // brief settles these as production values.
    synth::qwen3tts::SpeakerEncoderParams params;
    params.enc_dim     = 1024;
    params.sample_rate = 24000;
    params.mel_bins    = 128;
    params.n_fft       = 1024;
    params.hop_length  = 256;
    params.win_length  = 1024;
    params.fmin        = 0.0f;
    params.fmax        = 12000.0f;

    WavAudio    wav;
    std::string error;
    if (!read_wav(input_path, wav, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    if (wav.sample_rate != params.sample_rate) {
        std::fprintf(stderr, "%s is %u Hz but the speaker encoder runs at %u Hz -- this driver does not resample\n",
                     input_path.c_str(), wav.sample_rate, params.sample_rate);
        return 1;
    }

    synth::qwen3tts::MelSpectrogram mel;
    const synth_status_t            status = synth::qwen3tts::compute_log_mel(params, wav.samples, mel);
    if (status != SYNTH_OK) {
        std::fprintf(stderr, "compute_log_mel failed: status %d\n", int(status));
        return 1;
    }

    // Transpose from this port's [frame * bins + bin] layout to the oracle's
    // own [bin * frames + frame] row-major layout (conventions.json's
    // `mel_layout`), so the Python side can diff two flat float32 buffers
    // with no reshape of its own.
    std::vector<float> oracle_order(mel.values.size());
    for (uint64_t frame = 0; frame < mel.frames; ++frame) {
        for (uint32_t bin = 0; bin < mel.bins; ++bin) {
            oracle_order[size_t(bin) * mel.frames + frame] = mel.values[size_t(frame) * mel.bins + bin];
        }
    }
    if (!write_f32(output_path, oracle_order)) {
        std::fprintf(stderr, "could not write %s\n", output_path.c_str());
        return 1;
    }

    std::printf("{\"bins\": %u, \"frames\": %llu, \"input_samples\": %zu}\n", mel.bins, (unsigned long long) mel.frames,
                wav.samples.size());
    return 0;
}
