#include "prosody.h"

#include "ggml.h"
#include "weights.h"

#include <cmath>
#include <cstdio>

namespace synth::kokoro {

namespace {

// Upstream's AdaIN normalizes with the InstanceNorm1d default epsilon.
constexpr float kAdaINEpsilon = 1e-5f;
constexpr float kLeakySlope   = 0.2f;
// The block averages its two branches by scaling their sum.
const float     kBranchScale  = 1.0f / std::sqrt(2.0f);

// The upsampling pool is a depthwise transposed convolution that doubles the
// length: kernel 3, stride 2, padding 1, output padding 1.
constexpr int64_t kPoolKernel        = 3;
constexpr int64_t kPoolStride        = 2;
constexpr int64_t kPoolPadding       = 1;
constexpr int64_t kPoolOutputPadding = 1;

constexpr uint64_t kNodesPerResBlock = 64;
constexpr uint64_t kNodesEpilogue    = 16;

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

ggml_tensor * build_adain_res_block(ggml_context *               context,
                                    ggml_tensor *                input,
                                    ggml_tensor *                style,
                                    const AdainResBlockWeights & weights,
                                    bool                         upsample,
                                    float                        epsilon) {
    if (context == nullptr || input == nullptr || style == nullptr || weights.conv1.weight == nullptr ||
        weights.conv2.weight == nullptr || weights.norm1.fc.weight == nullptr || weights.norm2.fc.weight == nullptr) {
        return nullptr;
    }
    if (upsample && weights.pool.weight == nullptr) {
        return nullptr;
    }

    ggml_tensor * residual = adain(context, input, style, weights.norm1.fc, epsilon);
    residual               = ggml_leaky_relu(context, residual, kLeakySlope, false);
    if (upsample) {
        residual = depthwise_transpose_conv1d(context, residual, weights.pool.weight, weights.pool.bias, kPoolStride,
                                              kPoolPadding, kPoolOutputPadding);
        if (residual == nullptr) {
            return nullptr;
        }
    }
    residual = conv1d(context, residual, weights.conv1.weight, weights.conv1.bias, 1, 1, 1);
    residual = adain(context, residual, style, weights.norm2.fc, epsilon);
    residual = ggml_leaky_relu(context, residual, kLeakySlope, false);
    residual = conv1d(context, residual, weights.conv2.weight, weights.conv2.bias, 1, 1, 1);

    ggml_tensor * shortcut = upsample ? upsample_nearest_2x(context, input) : input;
    if (weights.conv1x1 != nullptr) {
        shortcut = conv1d(context, shortcut, weights.conv1x1, nullptr, 1, 0, 1);
    }

    return ggml_scale(context, ggml_add(context, residual, shortcut), kBranchScale);
}

uint64_t prosody_graph_node_count(uint32_t frame_count) {
    // Two three-block stacks plus the shared LSTM over the expanded frames.
    return bidirectional_lstm_node_count(frame_count) + 2 * 3 * kNodesPerResBlock + kNodesEpilogue;
}

ProsodyGraph build_prosody_graph(ggml_context *                  context,
                                 const ProsodyPredictorWeights & weights,
                                 const HParams &                 hparams,
                                 const ProsodyScratch &          scratch,
                                 uint32_t                        frame_count) {
    ProsodyGraph out;
    if (context == nullptr || frame_count == 0 || hparams.hidden_dim == 0 || hparams.style_dim == 0) {
        return out;
    }
    if (hparams.hidden_dim % 2 != 0 || weights.f0.size() != 3 || weights.n.size() != 3) {
        return out;
    }
    if (scratch.shared.zero_state == nullptr || scratch.shared.forward_store == nullptr ||
        scratch.shared.reverse_store == nullptr) {
        std::fprintf(stderr, "kokoro: prosody scratch must supply a persistent store for the shared LSTM\n");
        return out;
    }
    if (weights.f0_proj.weight == nullptr || weights.n_proj.weight == nullptr) {
        return out;
    }

    const int64_t  hidden   = hparams.hidden_dim;
    const int64_t  combined = hidden + hparams.style_dim;
    const uint64_t budget   = prosody_graph_node_count(frame_count) + 128;

    ggml_cgraph * graph = ggml_new_graph_custom(context, budget, false);
    if (graph == nullptr) {
        return out;
    }

    ggml_tensor * expanded = ggml_new_tensor_2d(context, GGML_TYPE_F32, combined, frame_count);
    ggml_set_name(expanded, "prosody.en");
    ggml_set_input(expanded);

    ggml_tensor * style = ggml_new_tensor_1d(context, GGML_TYPE_F32, hparams.style_dim);
    ggml_set_name(style, "style.prosody");
    ggml_set_input(style);

    ggml_tensor * shared = build_bidirectional_lstm(context, graph, expanded,
                                                    lstm_from(weights.shared, uint32_t(combined), uint32_t(hidden / 2)),
                                                    scratch.shared, frame_count);
    if (shared == nullptr) {
        return ProsodyGraph{};
    }

    // F0 and energy run the same three-block shape over the shared state; only
    // the middle block upsamples.
    const struct {
        const std::vector<AdainResBlockWeights> * blocks;
        const Conv1dWeights *                     projection;
        const char *                              name;
        ggml_tensor **                            slot;
    } branches[] = {
        { &weights.f0, &weights.f0_proj, "prosody.f0", &out.f0     },
        { &weights.n,  &weights.n_proj,  "prosody.n",  &out.energy },
    };

    for (const auto & branch : branches) {
        ggml_tensor * current = shared;
        for (size_t block = 0; block < branch.blocks->size(); ++block) {
            current =
                build_adain_res_block(context, current, style, (*branch.blocks)[block], block == 1, kAdaINEpsilon);
            if (current == nullptr) {
                return ProsodyGraph{};
            }
        }
        current = conv1d(context, current, branch.projection->weight, branch.projection->bias, 1, 0, 1);
        if (current == nullptr) {
            return ProsodyGraph{};
        }
        // The projection leaves a single channel; the curve is that row.
        current = ggml_reshape_1d(context, current, current->ne[1]);
        ggml_set_name(current, branch.name);
        ggml_set_output(current);
        ggml_build_forward_expand(graph, current);
        *branch.slot = current;
    }

    out.graph    = graph;
    out.expanded = expanded;
    out.style    = style;
    return out;
}

}  // namespace synth::kokoro
