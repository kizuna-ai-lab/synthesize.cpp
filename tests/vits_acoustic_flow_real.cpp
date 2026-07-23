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
    if (argc != 10 && argc != 11 && argc != 12) {
        std::cerr << "usage: synthesize-vits-acoustic-flow MODEL.gguf token_ids.i32 duration_noise.f32 "
                     "latent_noise.f32 OUT_DIR THREADS NOISE_SCALE NOISE_SCALE_W SPEAKING_RATE "
                     "[BACKEND [SPEAKER_INDEX]]\n";
        return 2;
    }
    int   threads       = 0;
    float noise_scale   = 0.0f;
    float noise_scale_w = 0.0f;
    float speaking_rate = 0.0f;
    try {
        threads       = std::stoi(argv[6]);
        noise_scale   = std::stof(argv[7]);
        noise_scale_w = std::stof(argv[8]);
        speaking_rate = std::stof(argv[9]);
    } catch (...) {
        return 2;
    }
    uint32_t speaker_index = UINT32_MAX;
    if (argc == 12 && !parse_vits_runner_speaker_index(argv[11], speaker_index)) {
        return 2;
    }

    std::vector<int32_t> token_ids;
    std::vector<float>   duration_noise;
    std::vector<float>   latent_noise;
    if (!read_values(argv[2], token_ids) || !read_values(argv[3], duration_noise) ||
        !read_values(argv[4], latent_noise)) {
        std::cerr << "failed to read acoustic-flow inputs\n";
        return 3;
    }

    std::unique_ptr<synth::vits::Model> model;
    synth_status_t                      status = load_vits_runner_model(argv[1], argc >= 11 ? argv[10] : "cpu", model);
    if (status != SYNTH_OK) {
        std::cerr << "model load failed: " << synth_status_string(status) << '\n';
        return 4;
    }

    synth::vits::AcousticFlowOutput result;
    status = model->run_acoustic_flow(token_ids, duration_noise, latent_noise, noise_scale, noise_scale_w,
                                      speaking_rate, threads, result, speaker_index);
    if (status != SYNTH_OK) {
        std::cerr << "acoustic flow failed: " << synth_status_string(status) << '\n';
        return 5;
    }

    const std::filesystem::path output_dir = argv[5];
    std::error_code             error;
    std::filesystem::create_directories(output_dir, error);
    if (error || !write_values(output_dir / "z.f32", result.z)) {
        std::cerr << "failed to write acoustic-flow output\n";
        return 6;
    }

    std::cout << "channels=" << result.channels << " frames=" << result.frame_count << '\n';
    return 0;
}
