#pragma once

#include <cstdint>

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

namespace synth::kokoro {

struct DecoderWeights;
struct HParams;

struct DecoderGraph {
    ggml_cgraph * graph    = nullptr;
    ggml_tensor * asr      = nullptr;  // [dim_in, frame_count]
    ggml_tensor * f0       = nullptr;  // [1, 2 * frame_count]
    ggml_tensor * energy   = nullptr;  // [1, 2 * frame_count]
    ggml_tensor * har      = nullptr;  // [n_fft + 2, output_frames]
    ggml_tensor * style    = nullptr;  // [style_dim]
    // [n_fft + 2, output_frames]: the magnitude half is still a logarithm and
    // the phase half is still a pre-sine angle, both applied by the host next
    // to the inverse transform.
    ggml_tensor * spectrum = nullptr;
};

uint64_t decoder_graph_node_count(const HParams & hparams, uint32_t frame_count);

// Builds the decoder and the iSTFTNet generator as one graph.
//
// The F0 and energy curves arrive at twice the frame rate, as the prosody stage
// leaves them; the decoder's own strided convolutions halve them back before
// they are concatenated onto the encoder features. The generator's harmonic
// source spectrum is supplied from the host seam rather than built here.
DecoderGraph build_decoder_graph(ggml_context *         context,
                                 const DecoderWeights & weights,
                                 const HParams &        hparams,
                                 uint32_t               frame_count);

}  // namespace synth::kokoro
