// Thin adapter that runs one Kokoro stage against a real package and writes
// its outputs as raw binary files.
//
// It is shared by the Python Golden validators and by manual diagnostics, and
// is deliberately not itself a test. Unlike the VITS family, which has one
// adapter per stage, Kokoro has a single one selected by name: its stages take
// the same arguments, so seven near-identical binaries would only duplicate the
// argument handling.

#include "arch/kokoro/kokoro.h"
#include "synthesize.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

template <typename T> bool read_values(const std::filesystem::path & path, std::vector<T> & output) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        return false;
    }
    const std::streamoff bytes = input.tellg();
    if (bytes <= 0 || bytes % static_cast<std::streamoff>(sizeof(T)) != 0) {
        return false;
    }
    output.resize(static_cast<size_t>(bytes) / sizeof(T));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(output.data()), bytes);
    return input.good();
}

template <typename T> bool write_values(const std::filesystem::path & path, const std::vector<T> & values) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        return false;
    }
    output.write(reinterpret_cast<const char *>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(T)));
    return output.good();
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 9) {
        std::cerr << "usage: synthesize-kokoro-stages STAGE MODEL.gguf token_ids.i32 RANDOM_DIR OUT_DIR "
                     "VOICE_INDEX SPEAKING_RATE THREADS\n";
        return 2;
    }
    const std::string           stage       = argv[1];
    const std::filesystem::path model_path  = argv[2];
    const std::filesystem::path tokens_path = argv[3];
    const std::filesystem::path random_dir  = argv[4];
    const std::filesystem::path out_dir     = argv[5];

    uint32_t voice_index   = 0;
    float    speaking_rate = 1.0f;
    int      threads       = 1;
    try {
        voice_index   = static_cast<uint32_t>(std::stoul(argv[6]));
        speaking_rate = std::stof(argv[7]);
        threads       = std::stoi(argv[8]);
    } catch (...) {
        return 2;
    }

    std::vector<int32_t> token_ids;
    if (!read_values(tokens_path, token_ids)) {
        std::cerr << "failed to read token IDs\n";
        return 3;
    }

    std::unique_ptr<synth::kokoro::Model> model;
    const synth_status_t                  loaded = synth::kokoro::Model::load_cpu(model_path.string(), model);
    if (loaded != SYNTH_OK || model == nullptr) {
        std::cerr << "failed to load model: status " << loaded << "\n";
        return 4;
    }

    // The Voice row is chosen by input length, so the runner resolves it the
    // same way synthesis would rather than taking it as an argument.
    uint32_t voice_row = 0;
    if (!model->resolve_voice_row(token_ids.size(), voice_row)) {
        std::cerr << "no style row for " << token_ids.size() << " tokens\n";
        return 5;
    }

    synth::kokoro::SourceRandom random;
    if (stage == "source" || stage == "decoder" || stage == "waveform") {
        if (!read_values(random_dir / "rand_ini.f32", random.rand_ini) ||
            !read_values(random_dir / "source_noise.f32", random.noise)) {
            std::cerr << "failed to read the replay tensors\n";
            return 3;
        }
    }

    std::error_code error;
    std::filesystem::create_directories(out_dir, error);
    if (error) {
        return 6;
    }

    synth_status_t status = SYNTH_OK;
    if (stage == "plbert") {
        synth::kokoro::PLBertOutput output;
        status = model->run_plbert(token_ids, threads, output);
        if (status == SYNTH_OK) {
            write_values(out_dir / "hidden.f32", output.hidden);
            write_values(out_dir / "d_en.f32", output.projected);
        }
    } else if (stage == "duration") {
        synth::kokoro::DurationOutput output;
        status = model->run_duration(token_ids, voice_index, voice_row, speaking_rate, threads, output);
        if (status == SYNTH_OK) {
            write_values(out_dir / "logits.f32", output.logits);
            write_values(out_dir / "d.f32", output.encoded);
            write_values(out_dir / "alignment.f32", output.alignment);
            write_values(out_dir / "pred_dur.i64", output.durations);
            const std::vector<int64_t> y_length{ static_cast<int64_t>(output.frame_count) };
            write_values(out_dir / "y_length.i64", y_length);
        }
    } else if (stage == "prosody") {
        synth::kokoro::ProsodyOutput output;
        status = model->run_prosody(token_ids, voice_index, voice_row, speaking_rate, threads, output);
        if (status == SYNTH_OK) {
            write_values(out_dir / "en.f32", output.expanded);
            write_values(out_dir / "f0.f32", output.f0);
            write_values(out_dir / "n.f32", output.energy);
        }
    } else if (stage == "text-encoder") {
        synth::kokoro::TextEncoderOutput output;
        status = model->run_text_encoder(token_ids, voice_index, voice_row, speaking_rate, threads, output);
        if (status == SYNTH_OK) {
            write_values(out_dir / "t_en.f32", output.encoded);
            write_values(out_dir / "asr.f32", output.aligned);
        }
    } else if (stage == "source") {
        synth::kokoro::SourceOutput output;
        status = model->run_source(token_ids, voice_index, voice_row, speaking_rate, random, threads, output);
        if (status == SYNTH_OK) {
            write_values(out_dir / "har.f32", output.spectrum);
            write_values(out_dir / "har_source.f32", output.excitation);
        }
    } else if (stage == "decoder") {
        synth::kokoro::DecoderOutput output;
        status = model->run_decoder(token_ids, voice_index, voice_row, speaking_rate, random, threads, output);
        if (status == SYNTH_OK) {
            write_values(out_dir / "spectrum.f32", output.spectrum);
        }
    } else if (stage == "waveform") {
        synth::kokoro::WaveformOutput output;
        status = model->run_waveform(token_ids, voice_index, voice_row, speaking_rate, random, threads, output);
        if (status == SYNTH_OK) {
            write_values(out_dir / "pcm.f32", output.pcm);
        }
    } else {
        std::cerr << "unknown stage " << stage << "\n";
        return 2;
    }

    if (status != SYNTH_OK) {
        std::cerr << "stage " << stage << " failed: status " << status << "\n";
        return 7;
    }
    return 0;
}
