#include "prior-expansion.h"

#include "ggml.h"

#include <cstdio>

namespace synth::vits {

PriorExpansionGraph build_prior_expansion_graph(ggml_context * context,
                                                int64_t        channels,
                                                int64_t        token_count,
                                                int64_t        frame_count) {
    PriorExpansionGraph result;
    if (context == nullptr || channels <= 0 || token_count <= 0 || frame_count <= 0) {
        std::fprintf(stderr, "vits: invalid prior-expansion graph request\n");
        return result;
    }

    result.m_p       = ggml_new_tensor_2d(context, GGML_TYPE_F32, channels, token_count);
    result.logs_p    = ggml_new_tensor_2d(context, GGML_TYPE_F32, channels, token_count);
    result.attention = ggml_new_tensor_2d(context, GGML_TYPE_F32, token_count, frame_count);
    ggml_set_name(result.m_p, "text.m_p.input");
    ggml_set_name(result.logs_p, "text.logs_p.input");
    ggml_set_name(result.attention, "duration.attention.input");
    ggml_set_input(result.m_p);
    ggml_set_input(result.logs_p);
    ggml_set_input(result.attention);

    ggml_tensor * m_p_transposed    = ggml_cont(context, ggml_transpose(context, result.m_p));
    ggml_tensor * logs_p_transposed = ggml_cont(context, ggml_transpose(context, result.logs_p));
    result.m_p_expanded =
        ggml_cont(context, ggml_transpose(context, ggml_mul_mat(context, result.attention, m_p_transposed)));
    result.logs_p_expanded =
        ggml_cont(context, ggml_transpose(context, ggml_mul_mat(context, result.attention, logs_p_transposed)));
    ggml_set_name(result.m_p_expanded, "prior.m_p_expanded");
    ggml_set_name(result.logs_p_expanded, "prior.logs_p_expanded");

    result.graph = ggml_new_graph_custom(context, 256, false);
    ggml_build_forward_expand(result.graph, result.m_p_expanded);
    ggml_build_forward_expand(result.graph, result.logs_p_expanded);
    return result;
}

}  // namespace synth::vits
