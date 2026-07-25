#include "decoder.h"

#include "generator.h"
#include "ggml.h"
#include "operations.h"
#include "prosody.h"
#include "source.h"
#include "weights.h"

#include <cstdio>

namespace synth::kokoro {

namespace {

// Generous per-block allowances; the graph is sized from these and then given
// headroom, so they only have to be upper bounds.
constexpr uint64_t kNodesPerAdainBlock   = 64;
constexpr uint64_t kNodesPerSnakeBlock   = 128;
constexpr uint64_t kNodesPerTransposeTap = 8;
constexpr uint64_t kNodesEpilogue        = 64;

// The curves reach the decoder at twice the frame rate and are halved by a
// stride-two convolution whose kernel is three wide.
constexpr int kCurveStride  = 2;
constexpr int kCurvePadding = 1;

}  // namespace

uint64_t decoder_graph_node_count(const HParams & hparams, uint32_t frame_count) {
    const uint64_t stages   = hparams.istftnet.upsample_rates.size();
    const uint64_t branches = hparams.istftnet.resblock_kernel_sizes.size();

    uint64_t total = kNodesEpilogue;
    // encode plus the four decode blocks.
    total += kNodesPerAdainBlock * 5;
    // Per stage: one noise residual block, the branch blocks, and the taps of
    // the transposed convolution.
    total += stages * kNodesPerSnakeBlock;
    total += stages * branches * kNodesPerSnakeBlock;
    for (size_t stage = 0; stage < hparams.istftnet.upsample_kernel_sizes.size(); ++stage) {
        total += uint64_t(hparams.istftnet.upsample_kernel_sizes[stage]) * kNodesPerTransposeTap;
    }
    // The pool inside the upsampling decode block is one tap per kernel entry.
    total += 3 * kNodesPerTransposeTap;
    // Frame count only enters through the concatenations, which are constant,
    // but a longer sequence costs no extra nodes.
    (void) frame_count;
    return total;
}

DecoderGraph build_decoder_graph(ggml_context *         context,
                                 const DecoderWeights & weights,
                                 const HParams &        hparams,
                                 uint32_t               frame_count) {
    DecoderGraph out;
    if (context == nullptr || frame_count == 0 || hparams.dim_in == 0 || hparams.style_dim == 0) {
        return out;
    }
    if (weights.decode.size() != 4 || weights.f0_conv.weight == nullptr || weights.n_conv.weight == nullptr ||
        weights.asr_res.weight == nullptr) {
        return out;
    }

    const uint64_t har_frames = generator_output_frames(hparams, uint64_t(frame_count) * 2);
    const uint64_t har_bins   = uint64_t(hparams.istftnet.gen_istft_n_fft) + 2;
    if (har_bins < 3) {
        return out;
    }

    const uint64_t budget = decoder_graph_node_count(hparams, frame_count) + 256;
    ggml_cgraph *  graph  = ggml_new_graph_custom(context, budget, false);
    if (graph == nullptr) {
        return out;
    }

    ggml_tensor * asr = ggml_new_tensor_2d(context, GGML_TYPE_F32, hparams.dim_in, frame_count);
    ggml_set_name(asr, "input.asr");
    ggml_set_input(asr);
    ggml_tensor * f0_curve = ggml_new_tensor_2d(context, GGML_TYPE_F32, 1, int64_t(frame_count) * 2);
    ggml_set_name(f0_curve, "input.f0");
    ggml_set_input(f0_curve);
    ggml_tensor * energy_curve = ggml_new_tensor_2d(context, GGML_TYPE_F32, 1, int64_t(frame_count) * 2);
    ggml_set_name(energy_curve, "input.energy");
    ggml_set_input(energy_curve);
    ggml_tensor * har = ggml_new_tensor_2d(context, GGML_TYPE_F32, int64_t(har_bins), int64_t(har_frames));
    ggml_set_name(har, "input.har");
    ggml_set_input(har);
    ggml_tensor * style = ggml_new_tensor_1d(context, GGML_TYPE_F32, hparams.style_dim);
    ggml_set_name(style, "input.style");
    ggml_set_input(style);

    ggml_tensor * f0 =
        conv1d(context, f0_curve, weights.f0_conv.weight, weights.f0_conv.bias, kCurveStride, kCurvePadding, 1);
    ggml_tensor * energy =
        conv1d(context, energy_curve, weights.n_conv.weight, weights.n_conv.bias, kCurveStride, kCurvePadding, 1);
    if (f0 == nullptr || energy == nullptr || f0->ne[1] != frame_count) {
        return DecoderGraph{};
    }

    ggml_tensor * current = ggml_concat(context, ggml_concat(context, asr, f0, 0), energy, 0);
    current               = build_adain_res_block(context, current, style, weights.encode, false, hparams.adain_eps);
    if (current == nullptr) {
        return DecoderGraph{};
    }

    // A narrow projection of the encoder features is re-supplied to every
    // decode block, which is what keeps the phonetic identity from washing out
    // as the style conditioning accumulates.
    ggml_tensor * asr_residual = conv1d(context, asr, weights.asr_res.weight, weights.asr_res.bias, 1, 0, 1);
    if (asr_residual == nullptr) {
        return DecoderGraph{};
    }

    for (size_t block = 0; block < weights.decode.size(); ++block) {
        const bool upsample = block + 1 == weights.decode.size();
        current             = ggml_concat(context, current, asr_residual, 0);
        current             = ggml_concat(context, ggml_concat(context, current, f0, 0), energy, 0);
        current = build_adain_res_block(context, current, style, weights.decode[block], upsample, hparams.adain_eps);
        if (current == nullptr) {
            return DecoderGraph{};
        }
    }

    ggml_tensor * spectrum = build_generator(context, current, style, har, weights.generator, hparams);
    if (spectrum == nullptr || spectrum->ne[1] != int64_t(har_frames)) {
        std::fprintf(stderr, "kokoro: generator produced a spectrum of the wrong length\n");
        return DecoderGraph{};
    }

    ggml_set_name(spectrum, "decoder.spectrum");
    ggml_set_output(spectrum);
    ggml_build_forward_expand(graph, spectrum);

    out.graph    = graph;
    out.asr      = asr;
    out.f0       = f0_curve;
    out.energy   = energy_curve;
    out.har      = har;
    out.style    = style;
    out.spectrum = spectrum;
    return out;
}

}  // namespace synth::kokoro
