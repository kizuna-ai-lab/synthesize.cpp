#include "plbert.h"

#include "ggml.h"
#include "operations.h"
#include "weights.h"

#include <cmath>
#include <cstdio>

namespace synth::kokoro {

namespace {

// One ALBERT layer: post-norm attention, then post-norm feed-forward.
ggml_tensor * albert_layer(ggml_context *             context,
                           ggml_tensor *              hidden,
                           const AlbertLayerWeights & weights,
                           uint32_t                   head_count,
                           uint32_t                   token_count,
                           float                      epsilon) {
    const int64_t hidden_size = hidden->ne[0];
    const int64_t head_size   = hidden_size / head_count;
    const float   scale       = 1.0f / std::sqrt(static_cast<float>(head_size));

    ggml_tensor * query = linear(context, hidden, weights.query);
    ggml_tensor * key   = linear(context, hidden, weights.key);
    ggml_tensor * value = linear(context, hidden, weights.value);

    // [head_size, token, head] so each head's scores are one matrix multiply.
    query = ggml_permute(context, ggml_reshape_3d(context, query, head_size, head_count, token_count), 0, 2, 1, 3);
    key   = ggml_permute(context, ggml_reshape_3d(context, key, head_size, head_count, token_count), 0, 2, 1, 3);

    ggml_tensor * scores = ggml_mul_mat(context, key, query);
    scores               = ggml_soft_max_ext(context, scores, nullptr, scale, 0.0f);

    // Values are consumed transposed, giving [token, head_size, head].
    ggml_tensor * value_t = ggml_cont(
        context,
        ggml_permute(context, ggml_reshape_3d(context, value, head_size, head_count, token_count), 1, 2, 0, 3));

    ggml_tensor * attended = ggml_mul_mat(context, value_t, scores);
    attended               = ggml_permute(context, attended, 0, 2, 1, 3);
    attended               = ggml_cont_2d(context, attended, hidden_size, token_count);

    ggml_tensor * projected = linear(context, attended, weights.dense);
    ggml_tensor * attention =
        layer_norm(context, ggml_add(context, hidden, projected), weights.attention_norm, epsilon);

    // The package's activation is gelu_new, the tanh form, evaluated here in
    // single precision rather than through ggml_gelu's half-precision table.
    ggml_tensor * feed = linear(context, attention, weights.ffn);
    feed               = gelu_tanh(context, feed);
    feed               = linear(context, feed, weights.ffn_output);
    return layer_norm(context, ggml_add(context, feed, attention), weights.output_norm, epsilon);
}

bool weights_complete(const PLBertWeights & weights) {
    const ggml_tensor * required[] = {
        weights.word_embeddings,       weights.position_embeddings,     weights.token_type_embeddings,
        weights.embedding_norm.weight, weights.embedding_norm.bias,     weights.hidden_mapping.weight,
        weights.hidden_mapping.bias,   weights.layer.query.weight,      weights.layer.key.weight,
        weights.layer.value.weight,    weights.layer.dense.weight,      weights.layer.attention_norm.weight,
        weights.layer.ffn.weight,      weights.layer.ffn_output.weight, weights.layer.output_norm.weight
    };
    for (const ggml_tensor * tensor : required) {
        if (tensor == nullptr) {
            return false;
        }
    }
    return true;
}

}  // namespace

PLBertGraph build_plbert_graph(ggml_context *        context,
                               const PLBertWeights & weights,
                               const HParams &       hparams,
                               uint32_t              token_count) {
    PLBertGraph out;
    if (context == nullptr || token_count == 0 || !weights_complete(weights)) {
        return out;
    }
    if (token_count > hparams.plbert.max_position_embeddings) {
        std::fprintf(stderr, "kokoro: %u tokens exceed the %u learned positions\n", token_count,
                     hparams.plbert.max_position_embeddings);
        return out;
    }
    if (hparams.plbert.num_attention_heads == 0 ||
        hparams.plbert.hidden_size % hparams.plbert.num_attention_heads != 0) {
        return out;
    }

    ggml_cgraph * graph = ggml_new_graph(context);
    if (graph == nullptr) {
        return out;
    }

    ggml_tensor * token_ids = ggml_new_tensor_1d(context, GGML_TYPE_I32, token_count);
    ggml_set_name(token_ids, "input.token_ids");
    ggml_set_input(token_ids);

    const int64_t embed_dim = weights.word_embeddings->ne[0];

    ggml_tensor * embedded   = ggml_get_rows(context, weights.word_embeddings, token_ids);
    // Absolute positions are the first `token_count` learned rows.
    ggml_tensor * positions  = ggml_view_2d(context, weights.position_embeddings, embed_dim, token_count,
                                            weights.position_embeddings->nb[1], 0);
    // Every token is type zero, so one row broadcasts across the sequence.
    ggml_tensor * token_type = ggml_view_1d(context, weights.token_type_embeddings, embed_dim, 0);

    ggml_tensor * sum = ggml_add(context, embedded, positions);
    sum               = ggml_add(context, sum, token_type);
    sum               = layer_norm(context, sum, weights.embedding_norm, hparams.plbert.layer_norm_eps);

    ggml_tensor * hidden = linear(context, sum, weights.hidden_mapping);
    for (uint32_t layer = 0; layer < hparams.plbert.num_hidden_layers; ++layer) {
        hidden = albert_layer(context, hidden, weights.layer, hparams.plbert.num_attention_heads, token_count,
                              hparams.plbert.layer_norm_eps);
    }

    ggml_set_name(hidden, "bert.hidden");
    ggml_set_output(hidden);
    ggml_build_forward_expand(graph, hidden);

    out.graph     = graph;
    out.token_ids = token_ids;
    out.hidden    = hidden;
    return out;
}

}  // namespace synth::kokoro
