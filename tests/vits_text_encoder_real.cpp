#include "arch/vits/vits.h"
#include "synthesize.h"
#include "vits-runner-backend.h"

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
    if (argc != 5 && argc != 6) {
        std::cerr << "usage: synthesize-vits-text-encoder MODEL.gguf token_ids.i32 OUT_DIR THREADS [BACKEND]\n";
        return 2;
    }
    int threads = 0;
    try {
        threads = std::stoi(argv[4]);
    } catch (...) {
        return 2;
    }

    std::vector<int32_t> token_ids;
    if (!read_values(argv[2], token_ids)) {
        std::cerr << "failed to read token IDs\n";
        return 3;
    }

    std::unique_ptr<synth::vits::Model> model;
    synth_status_t                      status = load_vits_runner_model(argv[1], argc == 6 ? argv[5] : "cpu", model);
    if (status != SYNTH_OK) {
        std::cerr << "model load failed: " << synth_status_string(status) << "\n";
        return 4;
    }

    synth::vits::TextEncoderOutput result;
    status = model->run_text_encoder(token_ids, threads, result);
    if (status != SYNTH_OK) {
        std::cerr << "text encoder failed: " << synth_status_string(status) << "\n";
        return 5;
    }

    const std::filesystem::path output_dir = argv[3];
    std::error_code             error;
    std::filesystem::create_directories(output_dir, error);
    if (error || !write_values(output_dir / "m_p.f32", result.m_p) ||
        !write_values(output_dir / "logs_p.f32", result.logs_p) ||
        !write_values(output_dir / "mask.f32", result.mask)) {
        std::cerr << "failed to write text-encoder outputs\n";
        return 6;
    }

    std::cout << "tokens=" << result.token_count << " channels=" << result.channels << '\n';
    return 0;
}
