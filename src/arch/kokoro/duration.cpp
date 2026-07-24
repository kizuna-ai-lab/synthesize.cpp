#include "duration.h"

#include "ggml.h"
#include "weights.h"

#include <cstdio>

namespace synth::kokoro {

namespace {

// The adaptive layer norm's own epsilon, fixed by the upstream module rather
// than declared by the package.
constexpr float kAdaLayerNormEpsilon = 1e-5f;

// Concatenation, the style broadcast, and the duration projection.
constexpr uint64_t kNodesPerEncoderLayer = 8;
constexpr uint64_t kNodesPrologue        = 3;
constexpr uint64_t kNodesEpilogue        = 3;

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

bool scratch_usable(const DurationScratch & scratch, uint32_t layers) {
    if (scratch.zero_state == nullptr || scratch.encoder.size() != layers) {
        return false;
    }
    for (const LstmScratch & entry : scratch.encoder) {
        if (entry.zero_state == nullptr || entry.forward_store == nullptr || entry.reverse_store == nullptr) {
            return false;
        }
    }
    return scratch.predictor.zero_state != nullptr && scratch.predictor.forward_store != nullptr &&
           scratch.predictor.reverse_store != nullptr;
}

}  // namespace

uint64_t duration_graph_node_count(const HParams & hparams, uint32_t token_count) {
    const uint64_t per_lstm = bidirectional_lstm_node_count(token_count);
    return kNodesPrologue + hparams.n_layer * (per_lstm + kNodesPerEncoderLayer) + per_lstm + kNodesEpilogue;
}

DurationGraph build_duration_graph(ggml_context *                  context,
                                   const ProsodyPredictorWeights & weights,
                                   const HParams &                 hparams,
                                   const DurationScratch &         scratch,
                                   uint32_t                        token_count) {
    DurationGraph out;
    if (context == nullptr || token_count == 0 || hparams.n_layer == 0 || hparams.hidden_dim == 0 ||
        hparams.style_dim == 0 || hparams.max_dur == 0) {
        return out;
    }
    if (hparams.hidden_dim % 2 != 0) {
        return out;
    }
    if (weights.text_encoder.lstms.size() != hparams.n_layer ||
        weights.text_encoder.ada_norm.size() != hparams.n_layer) {
        return out;
    }
    if (!scratch_usable(scratch, hparams.n_layer)) {
        std::fprintf(stderr, "kokoro: duration scratch must supply a persistent store per LSTM\n");
        return out;
    }
    if (weights.duration_proj.weight == nullptr || weights.duration_proj.bias == nullptr) {
        return out;
    }

    const uint64_t budget = duration_graph_node_count(hparams, token_count) + 64;
    ggml_cgraph *  graph  = ggml_new_graph_custom(context, budget, false);
    if (graph == nullptr) {
        return out;
    }

    const int64_t hidden    = hparams.hidden_dim;
    const int64_t style_dim = hparams.style_dim;
    const int64_t combined  = hidden + style_dim;
    const int64_t half      = hidden / 2;

    ggml_tensor * input = ggml_new_tensor_2d(context, GGML_TYPE_F32, hidden, token_count);
    ggml_set_name(input, "text.d_en");
    ggml_set_input(input);

    ggml_tensor * style = ggml_new_tensor_1d(context, GGML_TYPE_F32, style_dim);
    ggml_set_name(style, "style.prosody");
    ggml_set_input(style);

    // The style vector rides alongside the features through every layer.
    ggml_tensor * style_rows = broadcast_over_time(context, style, token_count);
    ggml_tensor * current    = ggml_concat(context, input, style_rows, 0);

    for (uint32_t layer = 0; layer < hparams.n_layer; ++layer) {
        ggml_tensor * encoded = build_bidirectional_lstm(
            context, graph, current, lstm_from(weights.text_encoder.lstms[layer], uint32_t(combined), uint32_t(half)),
            scratch.encoder[layer], token_count);
        if (encoded == nullptr) {
            return DurationGraph{};
        }
        ggml_tensor * normalized =
            ada_layer_norm(context, encoded, style, weights.text_encoder.ada_norm[layer], kAdaLayerNormEpsilon);
        current = ggml_concat(context, normalized, style_rows, 0);
    }

    ggml_set_name(current, "duration.d");
    ggml_set_output(current);

    ggml_tensor * predicted =
        build_bidirectional_lstm(context, graph, current, lstm_from(weights.lstm, uint32_t(combined), uint32_t(half)),
                                 scratch.predictor, token_count);
    if (predicted == nullptr) {
        return DurationGraph{};
    }

    ggml_tensor * logits = linear(context, predicted, weights.duration_proj);
    ggml_set_name(logits, "duration.logits");
    ggml_set_output(logits);

    ggml_build_forward_expand(graph, current);
    ggml_build_forward_expand(graph, logits);

    out.graph  = graph;
    out.input  = input;
    out.style  = style;
    out.hidden = current;
    out.logits = logits;
    return out;
}

}  // namespace synth::kokoro
