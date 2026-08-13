// Thin adapter that runs the codec encoder graph against a real package and
// writes every stage tap as a raw binary file, in the oracle's own on-disk
// layout, so Task 1's `codec_encoder/*.f32` dumps can be diffed against them.
//
// It is shared by manual diagnostics and by whatever validator Task 12 settles
// on, and is deliberately not itself a test -- the same role
// `kokoro_stages_real.cpp` and the `vits_*_real.cpp` adapters play. The
// comparison and its tolerance live outside it.
//
// IT TAKES THE ORACLE'S `waveform.f32`, NOT A WAV FILE, and that is a
// correctness requirement rather than a convenience. The oracle casts the clip
// to bfloat16 before its first convolution
// (qwen_tts/inference/qwen3_tts_tokenizer.py:248, `.to(self.model.dtype)`), and
// `waveform.f32` is captured after that cast. Measured on
// models/qwen3-tts-reference-audio/clone.wav: the WAV's own float32 samples
// differ from `waveform.f32` by up to 2.6% of the clip's rms. Feeding the WAV
// puts that perturbation through eleven convolutions before the comparison
// starts, and it swamps what the comparison is trying to see. There is
// therefore no WAV reader here, unlike every other driver in this directory.
//
// Weights come from a real Loaded Model; the graph runs on a plain CPU backend
// of this file's own, because the encoder's tensors are CPU-resident (they are
// bound against the package context, never a twin -- see catalog.cpp and
// model.cpp's twin pass) and the intermediates are all this allocator owns.

#include "arch/qwen3-tts/catalog.h"
#include "arch/qwen3-tts/codec-encoder.h"
#include "arch/qwen3-tts/qwen3-tts.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr size_t kNodeBudget = 8192;

struct ContextDeleter {
    void operator()(ggml_context * context) const {
        if (context != nullptr) {
            ggml_free(context);
        }
    }
};

using Context = std::unique_ptr<ggml_context, ContextDeleter>;

Context make_context(size_t bytes) {
    ggml_init_params parameters{};
    parameters.mem_size = bytes;
    parameters.no_alloc = true;
    return Context(ggml_init(parameters));
}

bool read_f32(const std::string & path, std::vector<float> & values) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        return false;
    }
    const std::streamoff bytes = input.tellg();
    if (bytes <= 0 || bytes % std::streamoff(sizeof(float)) != 0) {
        return false;
    }
    values.resize(size_t(bytes) / sizeof(float));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(values.data()), bytes);
    return input.good();
}

// Writes one tap in the layout the oracle's dumper produced.
//
// The graph is [channels, length] with CHANNELS contiguous. The oracle's
// convolution artifacts are [channels, length] with LENGTH contiguous, so they
// need transposing; its `transformer_l*.f32` are [length, channels] with
// channels contiguous -- upstream transposes to [B, T, C] around its
// transformer (modeling_mimi.py:1460, :1466) and this port does not -- so those
// are already in the graph's own order and must NOT be transposed.
bool write_tap(const std::filesystem::path & path, ggml_tensor * tensor, bool oracle_is_channel_last) {
    std::vector<float> values(size_t(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));

    const int64_t      channels = tensor->ne[0];
    const int64_t      length   = tensor->ne[1];
    std::vector<float> ordered(values.size());
    if (oracle_is_channel_last) {
        ordered = values;
    } else {
        for (int64_t position = 0; position < length; ++position) {
            for (int64_t channel = 0; channel < channels; ++channel) {
                ordered[size_t(channel * length + position)] = values[size_t(position * channels + channel)];
            }
        }
    }

    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char *>(ordered.data()), std::streamsize(ordered.size() * sizeof(float)));
    if (!output) {
        return false;
    }
    std::printf("  %-18s [%lld, %lld]\n", path.filename().string().c_str(), (long long) channels, (long long) length);
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr,
                     "usage: %s <model.gguf> <oracle-waveform.f32> <output-dir>\n"
                     "  the second argument is the oracle's own waveform.f32, not a WAV -- see this file's header\n",
                     argv[0]);
        return 2;
    }
    const std::string           model_path(argv[1]);
    const std::string           waveform_path(argv[2]);
    const std::filesystem::path out_dir(argv[3]);

    std::vector<float> pcm;
    if (!read_f32(waveform_path, pcm)) {
        std::fprintf(stderr, "could not read %s as a raw float32 buffer\n", waveform_path.c_str());
        return 1;
    }

    std::unique_ptr<synth::qwen3tts::Model> model;
    synth_status_t                          status = synth::qwen3tts::Model::load_cpu(model_path, model);
    if (status != SYNTH_OK) {
        std::fprintf(stderr, "load_cpu -> %d\n", int(status));
        return 1;
    }
    const synth::qwen3tts::CodecEncoderWeights & weights = model->codec_encoder_weights();

    synth::qwen3tts::CodecEncoderGeometry geometry;
    status = synth::qwen3tts::codec_encoder_check_waveform(weights, pcm.data(), pcm.size(), geometry);
    if (status != SYNTH_OK) {
        std::fprintf(stderr, "codec_encoder_check_waveform -> %d (a CustomVoice package carries no encoder)\n",
                     int(status));
        return 1;
    }

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (backend == nullptr) {
        std::fprintf(stderr, "no CPU backend\n");
        return 1;
    }

    Context       input_ctx = make_context(ggml_tensor_overhead() * 8);
    ggml_tensor * waveform  = ggml_new_tensor_2d(input_ctx.get(), GGML_TYPE_F32, 1, int64_t(pcm.size()));
    ggml_tensor * positions = ggml_new_tensor_1d(input_ctx.get(), GGML_TYPE_I32, geometry.transformer_positions);
    ggml_backend_buffer_t input_buffer = ggml_backend_alloc_ctx_tensors(input_ctx.get(), backend);
    if (input_buffer == nullptr) {
        std::fprintf(stderr, "input alloc failed\n");
        return 1;
    }
    ggml_backend_tensor_set(waveform, pcm.data(), 0, ggml_nbytes(waveform));
    std::vector<int32_t> sequential(size_t(geometry.transformer_positions), 0);
    for (int64_t index = 0; index < geometry.transformer_positions; ++index) {
        sequential[size_t(index)] = int32_t(index);
    }
    ggml_backend_tensor_set(positions, sequential.data(), 0, ggml_nbytes(positions));

    Context graph_ctx =
        make_context(ggml_tensor_overhead() * (kNodeBudget + 256) + ggml_graph_overhead_custom(kNodeBudget, false));
    ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx.get(), kNodeBudget, false);

    // The taps are ggml_set_output by the builder, so the allocator cannot
    // recycle them out from under the reads below.
    synth::qwen3tts::CodecEncoderTaps taps;
    ggml_tensor * latents = synth::qwen3tts::build_codec_encoder(graph_ctx.get(), waveform, positions, weights, &taps);
    if (latents == nullptr) {
        std::fprintf(stderr, "build_codec_encoder returned nullptr\n");
        return 1;
    }
    ggml_build_forward_expand(graph, latents);
    for (ggml_tensor * tap : taps.seanet_stages) {
        ggml_build_forward_expand(graph, tap);
    }
    ggml_build_forward_expand(graph, taps.seanet_tail);
    for (ggml_tensor * tap : taps.transformer_layers) {
        ggml_build_forward_expand(graph, tap);
    }

    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (allocator == nullptr || !ggml_gallocr_alloc_graph(allocator, graph)) {
        std::fprintf(stderr, "graph alloc failed\n");
        return 1;
    }
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "compute failed\n");
        return 1;
    }

    std::error_code ignored;
    std::filesystem::create_directories(out_dir, ignored);
    bool ok = true;
    for (size_t stage = 0; stage < taps.seanet_stages.size(); ++stage) {
        ok = write_tap(out_dir / ("seanet_stage" + std::to_string(stage) + ".f32"), taps.seanet_stages[stage], false) &&
             ok;
    }
    ok = write_tap(out_dir / "seanet_tail.f32", taps.seanet_tail, false) && ok;
    for (size_t layer = 0; layer < taps.transformer_layers.size(); ++layer) {
        ok = write_tap(out_dir / ("transformer_l" + std::to_string(layer) + ".f32"), taps.transformer_layers[layer],
                       true) &&
             ok;
    }
    // `latents.f32` is byte-identical to `downsample.f32` upstream -- the
    // quantizer is handed the downsampler's output with nothing in between
    // (modeling_mimi.py:1467-1469), asserted per case by the oracle's dumper.
    // Only one file is written here; a comparison may diff it against either.
    ok = write_tap(out_dir / "downsample.f32", latents, false) && ok;

    ggml_gallocr_free(allocator);
    ggml_backend_buffer_free(input_buffer);
    ggml_backend_free(backend);
    if (!ok) {
        std::fprintf(stderr, "a stage could not be written\n");
        return 1;
    }

    std::printf(
        "{\"samples\": %lld, \"samples_per_frame\": %lld, \"transformer_positions\": %lld, \"frames\": %lld, "
        "\"nodes\": %d}\n",
        (long long) geometry.samples, (long long) geometry.samples_per_frame,
        (long long) geometry.transformer_positions, (long long) geometry.frames, ggml_graph_n_nodes(graph));
    return 0;
}
