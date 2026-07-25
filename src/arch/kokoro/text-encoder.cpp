#include "text-encoder.h"

#include "ggml.h"
#include "weights.h"

#include <cstdio>

namespace synth::kokoro {

namespace {

// The encoder's own layer norm epsilon, fixed by the upstream module.
constexpr float    kLayerNormEpsilon = 1e-5f;
constexpr float    kLeakySlope       = 0.2f;
constexpr uint64_t kNodesPerBlock    = 24;
constexpr uint64_t kNodesEpilogue    = 8;

LstmWeights lstm_from(const LstmTensors & tensors, uint32_t input_dim, uint32_t hidden) {
    LstmWeights weights;
    weights.input_dim         = input_dim;
    weights.hidden            = hidden;
    weights.forward.weight_ih = tensors.forward.weight_ih;
    weights.forward.weight_hh = tensors.forward.weight_hh;
    weights.forward.bias_ih   = tensors.forward.bias_ih;
    weights.forward.bias_hh   = tensors.forward.bias_hh;
    weights.reverse.weight_ih = tensors.reverse.weight_ih;
    weights.reverse.weight_hh = tensors.reverse.weight_hh;
    weights.reverse.bias_ih   = tensors.reverse.bias_ih;
    weights.reverse.bias_hh   = tensors.reverse.bias_hh;
    return weights;
}

}  // namespace

uint64_t text_encoder_node_count(const HParams & hparams, uint32_t token_count) {
    return hparams.n_layer * kNodesPerBlock + bidirectional_lstm_node_count(token_count) + kNodesEpilogue;
}

TextEncoderGraph build_text_encoder_graph(ggml_context *             context,
                                          const TextEncoderWeights & weights,
                                          const HParams &            hparams,
                                          const TextEncoderScratch & scratch,
                                          uint32_t                   token_count) {
    TextEncoderGraph out;
    if (context == nullptr || token_count == 0 || hparams.hidden_dim == 0 || hparams.n_layer == 0) {
        return out;
    }
    if (hparams.hidden_dim % 2 != 0 || hparams.text_encoder_kernel_size % 2 == 0) {
        return out;
    }
    if (weights.embedding == nullptr || weights.cnn.size() != hparams.n_layer ||
        weights.cnn_norm.size() != hparams.n_layer) {
        return out;
    }
    if (scratch.lstm.zero_state == nullptr || scratch.lstm.forward_store == nullptr ||
        scratch.lstm.reverse_store == nullptr) {
        std::fprintf(stderr, "kokoro: text encoder scratch must supply a persistent store for its LSTM\n");
        return out;
    }

    const uint64_t budget = text_encoder_node_count(hparams, token_count) + 64;
    ggml_cgraph *  graph  = ggml_new_graph_custom(context, budget, false);
    if (graph == nullptr) {
        return out;
    }

    ggml_tensor * token_ids = ggml_new_tensor_1d(context, GGML_TYPE_I32, token_count);
    ggml_set_name(token_ids, "input.token_ids");
    ggml_set_input(token_ids);

    ggml_tensor * current = ggml_get_rows(context, weights.embedding, token_ids);

    // Padding keeps the sequence length through every convolution block.
    const int padding = int(hparams.text_encoder_kernel_size / 2);
    for (uint32_t block = 0; block < hparams.n_layer; ++block) {
        current = conv1d(context, current, weights.cnn[block].weight, weights.cnn[block].bias, 1, padding, 1);
        if (current == nullptr) {
            return TextEncoderGraph{};
        }
        current = layer_norm(context, current, weights.cnn_norm[block], kLayerNormEpsilon);
        current = ggml_leaky_relu(context, current, kLeakySlope, false);
    }

    ggml_tensor * encoded = build_bidirectional_lstm(
        context, graph, current, lstm_from(weights.lstm, hparams.hidden_dim, hparams.hidden_dim / 2), scratch.lstm,
        token_count);
    if (encoded == nullptr) {
        return TextEncoderGraph{};
    }

    ggml_set_name(encoded, "text.t_en");
    ggml_set_output(encoded);
    ggml_build_forward_expand(graph, encoded);

    out.graph     = graph;
    out.token_ids = token_ids;
    out.encoded   = encoded;
    return out;
}

}  // namespace synth::kokoro
