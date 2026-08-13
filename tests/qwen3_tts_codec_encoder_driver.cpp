// Thin adapter that runs the codec encoder against a real package and writes
// every stage tap, every RVQ intermediate and the reference codes as raw binary
// files, in the oracle's own on-disk layout, so Task 1's `codec_encoder/*.f32`
// dumps can be diffed against them.
//
// IT RUNS THE GRAPH TWICE, deliberately. The stage taps come from a graph this
// file builds itself, because a tap is an intermediate `ggml_tensor` that dies
// with its context and cannot be handed back through a value type; the codes
// and every RVQ quantity come from the REAL production entry point
// (`Model::prepare_codec_reference`), because an adapter that reimplemented the
// quantizer would measure the adapter. The two runs are compared for BIT
// equality on the latents before anything is written -- same input, same
// deterministic CPU F32 graph -- so a divergence between what the stage
// artifacts describe and what the codes were computed from is reported here
// rather than absorbed into a tolerance.
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
// WHAT THE TWO ORACLE CASES ACTUALLY COVER, recorded here because this is the
// tool a coverage claim will be made from. `base-ref-min`'s waveform.f32 is a
// BYTE-EXACT PREFIX of `base-icl-en`'s -- the first 24,000 of its 193,920
// samples, checked on the bytes -- and `base-text-short` is the same clip again
// at full length, differing only in synthesis text the encoder never reads. So
// the manifest's three cases are ONE RECORDING AT TWO LENGTHS: one speaker, one
// microphone, one sample rate. Running the driver on both is a real check of
// the length-dependent geometry (the shorter one is the only case where the
// frame downsampler's right-hand extra_padding is non-zero) and is NOT
// independent corroboration of a per-stage tolerance. A second speaker would
// be; there is not one.
//
// Weights come from a real Loaded Model; the graph runs on a plain CPU backend
// of this file's own, because the encoder's tensors are CPU-resident (they are
// bound against the package context, never a twin -- see catalog.cpp and
// model.cpp's twin pass) and the intermediates are all this allocator owns.

#include "arch/qwen3-tts/catalog.h"
#include "arch/qwen3-tts/codec-encoder-host.h"
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

// A buffer already in the oracle's own element order, written verbatim.
template <typename Element>
bool write_raw(const std::filesystem::path & path, const std::vector<Element> & values, const char * shape) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size() * sizeof(Element)));
    if (!output) {
        return false;
    }
    std::printf("  %-24s %s\n", path.filename().string().c_str(), shape);
    return true;
}

// `[fast, slow]` in GGML index order to `[slow, fast]` in numpy's, which is the
// same transpose write_tap performs and is spelled separately because these
// buffers arrive as plain vectors rather than as tensors.
std::vector<float> transposed(const std::vector<float> & values, int64_t fast, int64_t slow) {
    std::vector<float> ordered(values.size());
    for (int64_t outer = 0; outer < slow; ++outer) {
        for (int64_t inner = 0; inner < fast; ++inner) {
            ordered[size_t(inner * slow + outer)] = values[size_t(outer * fast + inner)];
        }
    }
    return ordered;
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
    ok = write_tap(out_dir / "downsample.f32", latents, false) && ok;

    // The second run: the real production entry point, which is what the codes
    // and every RVQ quantity must come from.
    synth::qwen3tts::CodecEncoding encoding;
    const char *                   diagnostic_code    = nullptr;
    const char *                   diagnostic_message = nullptr;
    status = model->prepare_codec_reference(pcm, 0, encoding, diagnostic_code, diagnostic_message);
    if (status != SYNTH_OK) {
        std::fprintf(stderr, "prepare_codec_reference -> %d (%s: %s)\n", int(status),
                     diagnostic_code != nullptr ? diagnostic_code : "-",
                     diagnostic_message != nullptr ? diagnostic_message : "-");
        return 1;
    }

    // Bit equality, not a tolerance: the same clip through the same
    // deterministic CPU F32 graph twice. Anything else means the stage
    // artifacts above and the codes below describe different runs, and a
    // stage-wise comparison built on them would be comparing across a seam it
    // could not see.
    std::vector<float> tap_latents(size_t(ggml_nelements(latents)));
    ggml_backend_tensor_get(latents, tap_latents.data(), 0, ggml_nbytes(latents));
    if (tap_latents.size() != encoding.latents.size()) {
        std::fprintf(stderr, "latent element count %zu from the tap run, %zu from prepare_codec_reference\n",
                     tap_latents.size(), encoding.latents.size());
        return 1;
    }
    for (size_t index = 0; index < tap_latents.size(); ++index) {
        if (tap_latents[index] != encoding.latents[index]) {
            std::fprintf(stderr, "the two runs disagree on latent %zu: %.9g vs %.9g\n", index, tap_latents[index],
                         encoding.latents[index]);
            return 1;
        }
    }

    const int64_t frames    = int64_t(encoding.frames);
    const int64_t groups    = int64_t(encoding.groups);
    const int64_t projected = int64_t(encoding.projected);
    char          shape[128];

    // `latents.f32`: the oracle's is C-order [channels, frames], the port's is
    // GGML [channels, frames] with channels fastest, so this one transposes.
    std::snprintf(shape, sizeof(shape), "[%lld, %lld]", (long long) encoding.latent_width, (long long) frames);
    ok = write_raw(out_dir / "latents.f32", transposed(encoding.latents, int64_t(encoding.latent_width), frames),
                   shape) &&
         ok;

    // `rvq_residual_sNN.f32`: [frames, projected] each, already the oracle's
    // own order slice for slice -- no transpose.
    for (int64_t group = 0; group < groups; ++group) {
        char name[64];
        std::snprintf(name, sizeof(name), "rvq_residual_s%02lld.f32", (long long) group);
        const size_t       stride = size_t(frames) * size_t(projected);
        std::vector<float> slice(encoding.residuals.begin() + std::ptrdiff_t(size_t(group) * stride),
                                 encoding.residuals.begin() + std::ptrdiff_t(size_t(group + 1) * stride));
        std::snprintf(shape, sizeof(shape), "[%lld, %lld]", (long long) frames, (long long) projected);
        ok = write_raw(out_dir / name, slice, shape) && ok;
    }

    // `rvq_reconstruction.f32`: [2, frames, projected], semantic branch first,
    // already the oracle's order.
    std::snprintf(shape, sizeof(shape), "[2, %lld, %lld]", (long long) frames, (long long) projected);
    ok = write_raw(out_dir / "rvq_reconstruction.f32", encoding.reconstruction, shape) && ok;

    // `rvq_distance_margin.f32`: the oracle's is C-order [groups, frames], i.e.
    // STAGE-major; the port's gap grid shares the codes' group-fastest layout.
    // So this one transposes and the codes below do not -- see
    // codec-encoder-host.h, which states that asymmetry because a reader would
    // otherwise assume it away.
    std::snprintf(shape, sizeof(shape), "[%lld, %lld]", (long long) groups, (long long) frames);
    ok = write_raw(out_dir / "rvq_distance_margin.f32", transposed(encoding.gaps, groups, frames), shape) && ok;

    // `codes.i32`: GGML [16, T] with ne[0] = 16 IS the oracle's C-order
    // [frames, 16], byte for byte. Written verbatim. A transpose here is the
    // one mistake this artifact invites.
    std::snprintf(shape, sizeof(shape), "[%lld, %lld]", (long long) frames, (long long) groups);
    ok = write_raw(out_dir / "codes.i32", encoding.codes, shape) && ok;

    ggml_gallocr_free(allocator);
    ggml_backend_buffer_free(input_buffer);
    ggml_backend_free(backend);
    if (!ok) {
        std::fprintf(stderr, "an artifact could not be written\n");
        return 1;
    }

    std::printf(
        "{\"samples\": %lld, \"samples_per_frame\": %lld, \"transformer_positions\": %lld, \"frames\": %lld, "
        "\"nodes\": %d, \"groups\": %lld, \"projected\": %lld, \"ref_rms\": %.9g, \"narrowest_gap\": %.9g}\n",
        (long long) geometry.samples, (long long) geometry.samples_per_frame,
        (long long) geometry.transformer_positions, (long long) geometry.frames, ggml_graph_n_nodes(graph),
        (long long) groups, (long long) projected, double(encoding.ref_rms), double(encoding.narrowest_gap));
    return 0;
}
