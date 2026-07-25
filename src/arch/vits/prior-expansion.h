#pragma once

#include <cstdint>

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

namespace synth::vits {

struct PriorExpansionGraph {
    ggml_cgraph * graph           = nullptr;
    ggml_tensor * m_p             = nullptr;
    ggml_tensor * logs_p          = nullptr;
    ggml_tensor * attention       = nullptr;
    ggml_tensor * m_p_expanded    = nullptr;
    ggml_tensor * logs_p_expanded = nullptr;
};

PriorExpansionGraph build_prior_expansion_graph(ggml_context * context,
                                                int64_t        channels,
                                                int64_t        token_count,
                                                int64_t        frame_count);

}  // namespace synth::vits
