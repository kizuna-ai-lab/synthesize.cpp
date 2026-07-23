#include "text-encoder.h"

#include "ggml.h"
#include "operations.h"

#include <cmath>
#include <cstdio>

namespace synth::vits {

namespace {

ggml_tensor * attention(ggml_context *           context,
                        ggml_tensor *            input,
                        ggml_tensor *            relative_indices,
                        const TextBlockWeights & weights,
                        const HParams &          hparams,
                        int64_t                  token_count) {
    const int64_t hidden        = hparams.hidden_channels;
    const int64_t heads         = hparams.text_head_count;
    const int64_t head_channels = hidden / heads;

    ggml_tensor * query = conv1d(context, input, weights.query, 0);
    ggml_tensor * key   = conv1d(context, input, weights.key, 0);
    ggml_tensor * value = conv1d(context, input, weights.value, 0);

    query = ggml_cont(
        context, ggml_permute(context, ggml_reshape_3d(context, query, head_channels, heads, token_count), 0, 2, 1, 3));
    key = ggml_cont(
        context, ggml_permute(context, ggml_reshape_3d(context, key, head_channels, heads, token_count), 0, 2, 1, 3));
    value = ggml_cont(
        context, ggml_permute(context, ggml_reshape_3d(context, value, head_channels, heads, token_count), 0, 2, 1, 3));

    ggml_tensor * scores = ggml_mul_mat(context, key, query);

    ggml_tensor * relative_key_table = ggml_pad(context, weights.relative_key, 0, 1, 0, 0);
    ggml_tensor * relative_key_rows  = ggml_get_rows(context, relative_key_table, relative_indices);
    relative_key_rows = ggml_reshape_3d(context, relative_key_rows, head_channels, token_count, token_count);
    ggml_tensor * query_per_position = ggml_reshape_4d(context, query, head_channels, 1, token_count, heads);
    ggml_tensor * relative_scores    = ggml_mul_mat(context, relative_key_rows, query_per_position);
    relative_scores                  = ggml_reshape_3d(context, relative_scores, token_count, token_count, heads);
    scores                           = ggml_add(context, scores, relative_scores);

    const float   scale         = 1.0f / std::sqrt(static_cast<float>(head_channels));
    ggml_tensor * probabilities = ggml_soft_max_ext(context, scores, nullptr, scale, 0.0f);

    ggml_tensor * value_transposed = ggml_cont(context, ggml_permute(context, value, 1, 0, 2, 3));
    ggml_tensor * output           = ggml_mul_mat(context, value_transposed, probabilities);

    ggml_tensor * relative_value_table = ggml_pad(context, weights.relative_value, 0, 1, 0, 0);
    ggml_tensor * relative_value_rows  = ggml_get_rows(context, relative_value_table, relative_indices);
    relative_value_rows = ggml_reshape_3d(context, relative_value_rows, head_channels, token_count, token_count);
    relative_value_rows = ggml_cont(context, ggml_permute(context, relative_value_rows, 1, 0, 2, 3));
    ggml_tensor * probabilities_per_position =
        ggml_reshape_4d(context, probabilities, token_count, 1, token_count, heads);
    ggml_tensor * relative_output = ggml_mul_mat(context, relative_value_rows, probabilities_per_position);
    relative_output               = ggml_reshape_3d(context, relative_output, head_channels, token_count, heads);
    output                        = ggml_add(context, output, relative_output);

    output = ggml_cont(context, ggml_permute(context, output, 0, 2, 1, 3));
    output = ggml_reshape_2d(context, output, hidden, token_count);
    return conv1d(context, output, weights.output, 0);
}

ggml_tensor * feed_forward(ggml_context *           context,
                           ggml_tensor *            input,
                           const TextBlockWeights & weights,
                           const HParams &          hparams) {
    const int     padding = static_cast<int>((hparams.text_ffn_kernel_size - 1) / 2);
    ggml_tensor * output  = conv1d(context, input, weights.ffn_input, padding);
    output                = ggml_relu(context, output);
    return conv1d(context, output, weights.ffn_output, padding);
}

}  // namespace

TextEncoderGraph build_text_encoder_graph(ggml_context *      context,
                                          const TextWeights & weights,
                                          const HParams &     hparams,
                                          int64_t             token_count) {
    TextEncoderGraph result;
    if (context == nullptr || token_count <= 0 || static_cast<uint64_t>(token_count) > hparams.max_input_tokens ||
        weights.blocks.size() != hparams.text_layer_count) {
        std::fprintf(stderr, "vits: invalid text-encoder graph request\n");
        return result;
    }

    result.token_ids        = ggml_new_tensor_1d(context, GGML_TYPE_I32, token_count);
    result.relative_indices = ggml_new_tensor_1d(context, GGML_TYPE_I32, token_count * token_count);
    ggml_set_name(result.token_ids, "input.token_ids");
    ggml_set_name(result.relative_indices, "text.relative_indices");
    ggml_set_input(result.token_ids);
    ggml_set_input(result.relative_indices);

    ggml_tensor * output = ggml_get_rows(context, weights.token_embedding, result.token_ids);
    output               = ggml_scale(context, output, hparams.embedding_scale);

    for (const TextBlockWeights & block : weights.blocks) {
        ggml_tensor * branch = attention(context, output, result.relative_indices, block, hparams, token_count);
        output =
            layer_norm(context, ggml_add(context, output, branch), block.attention_norm, hparams.layer_norm_epsilon);

        branch = feed_forward(context, output, block, hparams);
        output = layer_norm(context, ggml_add(context, output, branch), block.ffn_norm, hparams.layer_norm_epsilon);
    }
    result.encoded = output;
    ggml_set_name(result.encoded, "text.encoded");

    ggml_tensor * stats    = conv1d(context, output, weights.projection, 0);
    const int64_t channels = hparams.inter_channels;
    result.m_p             = ggml_cont(context, ggml_view_2d(context, stats, channels, token_count, stats->nb[1], 0));
    result.logs_p =
        ggml_cont(context, ggml_view_2d(context, stats, channels, token_count, stats->nb[1], channels * sizeof(float)));
    ggml_set_name(result.m_p, "text.m_p");
    ggml_set_name(result.logs_p, "text.logs_p");

    result.graph = ggml_new_graph_custom(context, 4096, false);
    ggml_build_forward_expand(result.graph, result.m_p);
    ggml_build_forward_expand(result.graph, result.logs_p);
    return result;
}

}  // namespace synth::vits
