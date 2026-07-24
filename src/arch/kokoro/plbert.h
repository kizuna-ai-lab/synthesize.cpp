#pragma once

#include <cstdint>

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

namespace synth::kokoro {

struct HParams;
struct PLBertWeights;

struct PLBertGraph {
    ggml_cgraph * graph     = nullptr;
    ggml_tensor * token_ids = nullptr;  // I32 [token_count]
    ggml_tensor * hidden    = nullptr;  // F32 [plbert.hidden_size, token_count]
};

// Builds the PL-BERT encoder over a single complete token sequence.
//
// The package stores one ALBERT layer group and replays it for every hidden
// layer, so the graph reuses the same weights `num_hidden_layers` times.
//
// No attention mask is built. Kokoro consumes one unpadded sequence per
// synthesis, so upstream's mask is all-visible and contributes nothing; the
// host seam is responsible for never presenting a padded batch.
PLBertGraph build_plbert_graph(ggml_context *        context,
                               const PLBertWeights & weights,
                               const HParams &       hparams,
                               uint32_t              token_count);

}  // namespace synth::kokoro
