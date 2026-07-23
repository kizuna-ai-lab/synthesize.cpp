#include "arch/vits/vits.h"
#include "synthesize.h"
#include "vits-runner-backend.h"
#include "vits-runner-voice.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
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
    if (argc != 8 && argc != 9 && argc != 10) {
        std::cerr << "usage: synthesize-vits-duration-path MODEL.gguf token_ids.i32 duration_noise.f32 "
                     "OUT_DIR THREADS NOISE_SCALE_W SPEAKING_RATE [BACKEND [SPEAKER_INDEX]]\n";
        return 2;
    }
    int   threads       = 0;
    float noise_scale_w = 0.0f;
    float speaking_rate = 0.0f;
    try {
        threads       = std::stoi(argv[5]);
        noise_scale_w = std::stof(argv[6]);
        speaking_rate = std::stof(argv[7]);
    } catch (...) {
        return 2;
    }
    uint32_t speaker_index = UINT32_MAX;
    if (argc == 10 && !parse_vits_runner_speaker_index(argv[9], speaker_index)) {
        return 2;
    }

    std::vector<int32_t> token_ids;
    std::vector<float>   duration_noise;
    if (!read_values(argv[2], token_ids) || !read_values(argv[3], duration_noise)) {
        std::cerr << "failed to read duration inputs\n";
        return 3;
    }

    std::unique_ptr<synth::vits::Model> model;
    synth_status_t                      status = load_vits_runner_model(argv[1], argc >= 9 ? argv[8] : "cpu", model);
    if (status != SYNTH_OK) {
        std::cerr << "model load failed: " << synth_status_string(status) << '\n';
        return 4;
    }

    synth::vits::DurationOutput result;
    status =
        model->run_duration(token_ids, duration_noise, noise_scale_w, speaking_rate, threads, result, speaker_index);
    if (status != SYNTH_OK) {
        std::cerr << "duration path failed: " << synth_status_string(status) << '\n';
        return 5;
    }
    if (result.frame_count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return 5;
    }

    const std::filesystem::path output_dir = argv[4];
    const std::vector<int64_t>  y_length   = { static_cast<int64_t>(result.frame_count) };
    std::error_code             error;
    std::filesystem::create_directories(output_dir, error);
    if (error || !write_values(output_dir / "w_ceil.f32", result.w_ceil) ||
        !write_values(output_dir / "y_length.i64", y_length) ||
        !write_values(output_dir / "attention.f32", result.attention)) {
        std::cerr << "failed to write duration-path outputs\n";
        return 6;
    }

    std::cout << "tokens=" << result.token_count << " frames=" << result.frame_count << '\n';
    return 0;
}
