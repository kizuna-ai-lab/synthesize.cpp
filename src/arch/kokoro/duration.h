#pragma once

#include "operations.h"

#include <cstdint>
#include <vector>

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

namespace synth::kokoro {

struct HParams;
struct ProsodyPredictorWeights;

// Persistent scratch for the duration path's unrolled LSTMs. Every store must
// live in a backend buffer the graph allocator does not own; see
// docs/porting/families/kokoro.md for why allocator-managed stores are unsafe.
struct DurationScratch {
    ggml_tensor *            zero_state = nullptr;  // [hidden_dim / 2]
    std::vector<LstmScratch> encoder;               // one per DurationEncoder layer
    LstmScratch              predictor;
};

struct DurationGraph {
    ggml_cgraph * graph  = nullptr;
    ggml_tensor * input  = nullptr;  // [hidden_dim, token_count], the projected PL-BERT state
    ggml_tensor * style  = nullptr;  // [style_dim], the prosody half of the Voice
    ggml_tensor * hidden = nullptr;  // [hidden_dim + style_dim, token_count]
    ggml_tensor * logits = nullptr;  // [max_dur, token_count]
};

// Nodes the duration graph adds, so callers can size their graph up front.
uint64_t duration_graph_node_count(const HParams & hparams, uint32_t token_count);

// Builds the DurationEncoder and the duration projection.
//
// The encoder alternates a bidirectional LSTM with an adaptive layer norm,
// re-concatenating the style vector after each norm, which is why its output is
// `hidden_dim + style_dim` wide. `hidden` is the tensor the prosody stage later
// expands by the resolved alignment; `logits` feeds the host duration seam.
DurationGraph build_duration_graph(ggml_context *                  context,
                                   const ProsodyPredictorWeights & weights,
                                   const HParams &                 hparams,
                                   const DurationScratch &         scratch,
                                   uint32_t                        token_count);

}  // namespace synth::kokoro
