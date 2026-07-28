// Replays the oracle's cases through the port and writes the artifacts the
// Stage 5 validator compares.
//
// This is an adapter, not a test: it asserts nothing and reports what it
// produced. The comparison and its thresholds live in
// scripts/validate-qwen3-tts-replay.py, so a tolerance is a reviewed number in a
// committed file rather than a constant here.
//
// Replay is the Port Validation Contract's seam. The oracle sampled its codes
// from PyTorch's generator and this port draws from its own, so parity feeds the
// oracle's codes back in rather than trying to reproduce them; every probe
// downstream of the draw is then compared on identical inputs.

#include "arch/qwen3-tts/qwen3-tts.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

bool read_file(const std::string & path, std::vector<char> & bytes) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        return false;
    }
    const std::streamsize size = input.tellg();
    input.seekg(0);
    bytes.resize(size_t(size));
    return bool(input.read(bytes.data(), size));
}

bool read_i32(const std::string & path, std::vector<int32_t> & values) {
    std::vector<char> bytes;
    if (!read_file(path, bytes) || bytes.size() % sizeof(int32_t) != 0) {
        return false;
    }
    values.resize(bytes.size() / sizeof(int32_t));
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return true;
}

bool write_f32(const std::string & path, const std::vector<float> & values) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        return false;
    }
    output.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size() * sizeof(float)));
    return bool(output);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 6) {
        std::fprintf(stderr,
                     "usage: %s <model.gguf> <case-dir> <out-dir> <voice-id> <language-tag> [probe-layers...]\n",
                     argv[0]);
        return 2;
    }
    const std::string model_path(argv[1]);
    const std::string case_dir(argv[2]);
    const std::string out_dir(argv[3]);

    std::vector<int32_t> token_ids;
    std::vector<int32_t> semantic;
    std::vector<int32_t> acoustic;
    if (!read_i32(case_dir + "/input/token_ids.i32", token_ids) ||
        !read_i32(case_dir + "/codes/semantic.i32", semantic) ||
        !read_i32(case_dir + "/codes/acoustic.i32", acoustic)) {
        std::fprintf(stderr, "cannot read the oracle's artifacts under %s\n", case_dir.c_str());
        return 2;
    }
    const uint64_t frames = semantic.size();
    if (frames == 0 || acoustic.size() % frames != 0) {
        std::fprintf(stderr, "%llu semantic codes against %zu acoustic\n", (unsigned long long) frames,
                     acoustic.size());
        return 2;
    }
    const size_t groups = acoustic.size() / frames + 1;

    // The oracle splits column 0 from columns 1..15 of a [frames, 16] block, so
    // the acoustic stream is frame-major and not level-major. Reading it the
    // other way transposes the codes, which decodes to audio rather than to an
    // error.
    std::vector<int32_t> replay(size_t(frames) * groups);
    for (uint64_t frame = 0; frame < frames; ++frame) {
        replay[size_t(frame) * groups] = semantic[size_t(frame)];
        for (size_t level = 0; level + 1 < groups; ++level) {
            replay[size_t(frame) * groups + level + 1] = acoustic[size_t(frame) * (groups - 1) + level];
        }
    }

    std::unique_ptr<synth::qwen3tts::Model> model;
    synth_status_t                          status = synth::qwen3tts::Model::load_cpu(model_path, model);
    if (status != SYNTH_OK) {
        std::fprintf(stderr, "load -> %d\n", int(status));
        return 1;
    }

    synth::qwen3tts::SynthesisRequest request;
    request.token_ids     = token_ids;
    request.voice_id      = argv[4];
    request.language      = argv[5];
    request.replay_codes  = &replay;
    request.replay_frames = frames;
    request.max_frames    = frames;
    request.threads       = 0;
    for (int index = 6; index < argc; ++index) {
        request.probe_layers.push_back(uint32_t(std::atoi(argv[index])));
    }

    synth::qwen3tts::SynthesisOutput output;
    const auto                       started = std::chrono::steady_clock::now();
    status                                   = model->run_synthesis(request, output);
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    if (status != SYNTH_OK) {
        std::fprintf(stderr, "run_synthesis -> %d\n", int(status));
        return 1;
    }

    bool ok = write_f32(out_dir + "/pcm.f32", output.audio) &&
              write_f32(out_dir + "/talker_logits.f32", output.talker_logits) &&
              write_f32(out_dir + "/talker_final.f32", output.talker_final);
    for (size_t index = 0; index < output.talker_layers.size() && ok; ++index) {
        ok = write_f32(out_dir + "/talker_l" + std::to_string(request.probe_layers[index]) + ".f32",
                       output.talker_layers[index]);
    }
    if (!ok) {
        std::fprintf(stderr, "cannot write under %s\n", out_dir.c_str());
        return 2;
    }
    const double total = output.talker_seconds + output.predictor_seconds + output.codec_seconds;
    std::printf(
        "{\"frames\": %llu, \"samples\": %zu, \"probe_layers\": %zu, "
        "\"talker_seconds\": %.4f, \"predictor_seconds\": %.4f, \"codec_seconds\": %.4f, "
        "\"codec_share\": %.4f, \"predictor_setup_seconds\": %.4f, \"wall_seconds\": %.4f}\n",
        (unsigned long long) output.frame_count, output.audio.size(), output.talker_layers.size(),
        output.talker_seconds, output.predictor_seconds, output.codec_seconds,
        total > 0.0 ? output.codec_seconds / total : 0.0, output.predictor_setup_seconds, wall);
    return 0;
}
