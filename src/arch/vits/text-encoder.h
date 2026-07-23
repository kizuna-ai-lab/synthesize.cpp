#pragma once

#include "weights.h"

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

namespace synth::vits {

struct TextEncoderGraph {
    ggml_cgraph * graph            = nullptr;
    ggml_tensor * token_ids        = nullptr;
    ggml_tensor * relative_indices = nullptr;
    ggml_tensor * encoded          = nullptr;
    ggml_tensor * m_p              = nullptr;
    ggml_tensor * logs_p           = nullptr;
};

TextEncoderGraph build_text_encoder_graph(ggml_context *      context,
                                          const TextWeights & weights,
                                          const HParams &     hparams,
                                          int64_t             token_count);

}  // namespace synth::vits
