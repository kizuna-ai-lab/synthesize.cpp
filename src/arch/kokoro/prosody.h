#pragma once

#include "operations.h"

#include <cstdint>
#include <vector>

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

namespace synth::kokoro {

struct AdainResBlockWeights;
struct HParams;
struct ProsodyPredictorWeights;

// Applies one AdainResBlk1d.
//
// The residual branch is norm, activation, optional upsampling pool, then two
// convolutions with a second norm between them. The shortcut upsamples and,
// when the channel count changes, projects. Upstream scales their sum by
// 1/sqrt(2) so the block preserves variance.
ggml_tensor * build_adain_res_block(ggml_context *               context,
                                    ggml_tensor *                input,
                                    ggml_tensor *                style,
                                    const AdainResBlockWeights & weights,
                                    bool                         upsample,
                                    float                        epsilon);

// Persistent scratch for the prosody stage's shared LSTM.
struct ProsodyScratch {
    LstmScratch shared;
};

struct ProsodyGraph {
    ggml_cgraph * graph    = nullptr;
    ggml_tensor * expanded = nullptr;  // [hidden_dim + style_dim, frame_count]
    ggml_tensor * style    = nullptr;  // [style_dim]
    ggml_tensor * f0       = nullptr;  // [2 * frame_count]
    ggml_tensor * energy   = nullptr;  // [2 * frame_count]
};

uint64_t prosody_graph_node_count(uint32_t frame_count);

// Builds the F0 and energy curves from the alignment-expanded encoder state.
//
// Both curves leave this stage at twice the frame rate: the middle residual
// block upsamples, and the decoder later halves them again with a strided
// convolution.
ProsodyGraph build_prosody_graph(ggml_context *                  context,
                                 const ProsodyPredictorWeights & weights,
                                 const HParams &                 hparams,
                                 const ProsodyScratch &          scratch,
                                 uint32_t                        frame_count);

}  // namespace synth::kokoro
