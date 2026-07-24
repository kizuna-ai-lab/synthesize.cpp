#pragma once

#include "operations.h"

#include <cstdint>

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

namespace synth::kokoro {

struct HParams;
struct TextEncoderWeights;

struct TextEncoderScratch {
    LstmScratch lstm;
};

struct TextEncoderGraph {
    ggml_cgraph * graph     = nullptr;
    ggml_tensor * token_ids = nullptr;  // I32 [token_count]
    ggml_tensor * encoded   = nullptr;  // [hidden_dim, token_count]
};

uint64_t text_encoder_node_count(const HParams & hparams, uint32_t token_count);

// Builds the acoustic text encoder: an embedding, then convolution blocks with
// a layer norm and leaky activation, then a bidirectional LSTM.
//
// This is a separate path from PL-BERT. PL-BERT conditions prosody, while this
// encoder produces the features the decoder consumes after alignment expansion.
TextEncoderGraph build_text_encoder_graph(ggml_context *             context,
                                          const TextEncoderWeights & weights,
                                          const HParams &            hparams,
                                          const TextEncoderScratch & scratch,
                                          uint32_t                   token_count);

}  // namespace synth::kokoro
