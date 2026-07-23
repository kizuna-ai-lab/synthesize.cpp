#include "arch/vits/vits.h"
#include "synthesize.h"
#include "vits-runner-backend.h"
#include "vits-runner-voice.h"

#include <cstdint>
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
    if (argc != 7 && argc != 8 && argc != 9) {
        std::cerr << "usage: synthesize-vits-duration-predictor MODEL.gguf token_ids.i32 duration_noise.f32 "
                     "OUT_DIR THREADS NOISE_SCALE_W [BACKEND [SPEAKER_INDEX]]\n";
        return 2;
    }
    int   threads       = 0;
    float noise_scale_w = 0.0f;
    try {
        threads       = std::stoi(argv[5]);
        noise_scale_w = std::stof(argv[6]);
    } catch (...) {
        return 2;
    }
    uint32_t speaker_index = UINT32_MAX;
    if (argc == 9 && !parse_vits_runner_speaker_index(argv[8], speaker_index)) {
        return 2;
    }

    std::vector<int32_t> token_ids;
    std::vector<float>   duration_noise;
    if (!read_values(argv[2], token_ids) || !read_values(argv[3], duration_noise)) {
        std::cerr << "failed to read duration-predictor inputs\n";
        return 3;
    }

    std::unique_ptr<synth::vits::Model> model;
    synth_status_t                      status = load_vits_runner_model(argv[1], argc >= 8 ? argv[7] : "cpu", model);
    if (status != SYNTH_OK) {
        std::cerr << "model load failed: " << synth_status_string(status) << '\n';
        return 4;
    }

    synth::vits::DurationPredictorOutput result;
    status = model->run_duration_predictor(token_ids, duration_noise, noise_scale_w, threads, result, speaker_index);
    if (status != SYNTH_OK) {
        std::cerr << "duration predictor failed: " << synth_status_string(status) << '\n';
        return 5;
    }

    const std::filesystem::path output_dir = argv[4];
    std::error_code             error;
    std::filesystem::create_directories(output_dir, error);
    if (error || !write_values(output_dir / "logw.f32", result.logw)) {
        std::cerr << "failed to write duration-predictor output\n";
        return 6;
    }

    std::cout << "tokens=" << result.token_count << '\n';
    return 0;
}
