// Graph construction for the HuBERT semantic branch and the codec's own
// SemanticEncoder conv stack -- see reference-encoder.h for the full pipeline
// and its citations against the pinned transformers source.
//
// Attention delta list against src/arch/omnivoice/generator.cpp's
// generator_layer, the structural donor this file transcribes from:
//   - NO rope. HuBERT's only positional signal is the single additive
//     grouped-conv embedding build_semantic_branch adds once before the
//     layer stack (pos_conv_embed below); nothing here rotates q or k.
//   - NO per-head q/k RMSNorm. Qwen3 normalizes each head of q and k before
//     rope; HubertAttention applies no such step (eager_attention_forward
//     runs directly on the raw projections).
//   - every projection is BIASED (q/k/v/out), where generator_layer's are
//     not -- HubertAttention's four nn.Linear all default to bias=True.
//   - NO grouped-query attention: HuBERT's key/value head count always
//     equals its query head count, so there is no kv-head broadcast to
//     account for (mul_mat's own broadcast, which generator_layer relies on
//     for its GQA, degenerates to the ordinary case here).
//   - the residual/norm ordering is POST-LN, not pre-LN: HubertEncoderLayer
//     (do_stable_layer_norm = False) feeds `hidden_states` to attention
//     WITHOUT normalizing it first -- there is no analogue of
//     generator_layer's `input_layernorm` at all -- and normalizes only
//     AFTER each residual add (attn -> +residual -> LayerNorm -> ff ->
//     +residual -> LayerNorm). generator_layer normalizes BEFORE each branch
//     and adds the branch to an unnormalized residual.
//   - the feed-forward is a plain two-layer MLP with a GELU(erf) in between
//     (HubertFeedForward), not generator_layer's SwiGLU-style gated one.
//   - the norm itself is LayerNorm (mean AND variance, plus a bias), not
//     generator_layer's RMSNorm (variance only, no bias, no mean
//     subtraction).
// The softmax scale mechanism (`ggml_soft_max_ext`'s own scale argument,
// 1/sqrt(head_dim)) and the reshape-to-heads convention (head_dim the
// fastest-varying sub-index of the hidden axis) are NOT deltas: both sides
// use the same technique.

#include "arch/omnivoice/reference-encoder.h"

#include "arch/omnivoice/catalog.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>

namespace synth::omnivoice {

namespace {

// HiggsAudioV2TokenizerModel._extract_semantic_features pads the RAW 16 kHz
// waveform by this many zero samples on each side, unconditionally --
// modeling_higgs_audio_v2_tokenizer.py:499, `F.pad(input_values, (160, 160))`.
// See reference-encoder.h's top comment for why this is a literal rather than
// something derived from HParams.
constexpr int kSemanticPadSamples = 160;

// torch.nn.GroupNorm's own default (HubertGroupNormConvLayer passes no `eps`
// keyword), distinct from `semantic.layer_norm_eps` (config.layer_norm_eps,
// 1e-5 for this checkpoint too, but a different field upstream).
constexpr float kGroupNormEps = 1e-5f;

// This checkpoint's codec.encoder_semantic.downsample_factor
// (HiggsAudioV2TokenizerConfig.semantic_downsample_factor = hop_length /
// (sample_rate / semantic_sample_rate) / downsample_factor = 960 / 1.5 / 320
// = 2 exactly), likewise not threaded through HParams -- see reference-
// encoder.h.
constexpr int64_t kSemanticDownsampleFactor = 2;

bool bound(const Conv1dWeights & weights, bool biased) {
    return weights.weight != nullptr && (!biased || weights.bias != nullptr);
}

bool bound(const LinearWeights & weights) {
    return weights.weight != nullptr && weights.bias != nullptr;
}

bool bound(const LayerNormWeights & weights) {
    return weights.weight != nullptr && weights.bias != nullptr;
}

// A bias over [channels, length] is per channel, so it broadcasts along the
// length rather than across it -- codec.cpp's add_channel_bias, duplicated
// here because that one is file-local to codec.cpp.
ggml_tensor * add_channel_bias(ggml_context * context, ggml_tensor * signal, ggml_tensor * bias) {
    return ggml_add(context, signal, ggml_reshape_2d(context, bias, bias->ne[0], 1));
}

ggml_tensor * linear(ggml_context * context, ggml_tensor * input, const LinearWeights & weights) {
    if (context == nullptr || input == nullptr || !bound(weights) || weights.weight->ne[0] != input->ne[0] ||
        weights.weight->ne[1] != weights.bias->ne[0]) {
        return nullptr;
    }
    return add_channel_bias(context, ggml_mul_mat(context, weights.weight, input), weights.bias);
}

// Normalizes along ne0 (the feature axis in this file's [features, positions]
// convention), which is exactly what every LayerNorm in this module needs:
// mean/variance over the hidden width, independently per position. Mirrors
// vits/operations.cpp's layer_norm; not shared because that one is file-local
// to the vits family.
ggml_tensor * layer_norm(ggml_context * context, ggml_tensor * input, const LayerNormWeights & weights, float eps) {
    if (context == nullptr || input == nullptr || !bound(weights) || weights.weight->ne[0] != input->ne[0]) {
        return nullptr;
    }
    ggml_tensor * output = ggml_norm(context, input, eps);
    output               = ggml_mul(context, output, weights.weight);
    output               = ggml_add(context, output, weights.bias);
    return output;
}

// HubertGroupNormConvLayer's GroupNorm(num_groups = channels, num_channels =
// channels): every channel is its own group, so this normalizes each channel
// independently over TIME rather than over the feature axis -- the opposite
// reduction from `layer_norm` above, and ggml_norm only ever reduces ne0. The
// channel-major convention this file shares with codec.cpp puts channels on
// ne0, so the tensor is transposed to time-major, normalized (now a per-
// channel reduction over ne0 = time), affine-scaled with the weight/bias
// reshaped to broadcast along ne1 = channels, and transposed back.
ggml_tensor * group_norm_per_channel(ggml_context * context, ggml_tensor * input, const LayerNormWeights & weights,
                                     float eps) {
    if (context == nullptr || input == nullptr || !bound(weights) || weights.weight->ne[0] != input->ne[0]) {
        return nullptr;
    }
    const int64_t channels    = input->ne[0];
    ggml_tensor * time_major  = ggml_cont(context, ggml_transpose(context, input));  // [T, C]
    ggml_tensor * normed      = ggml_norm(context, time_major, eps);
    ggml_tensor * gain        = ggml_reshape_2d(context, weights.weight, 1, channels);
    ggml_tensor * shift       = ggml_reshape_2d(context, weights.bias, 1, channels);
    ggml_tensor * scaled      = ggml_mul(context, normed, gain);
    ggml_tensor * shifted     = ggml_add(context, scaled, shift);
    return ggml_cont(context, ggml_transpose(context, shifted));  // back to [C, T]
}

// im2col + matmul, the codec.cpp recipe (ggml_conv_1d's CPU path asserts an
// F16 kernel and this family's are F32), generalized over stride and an
// optional bias -- codec_conv1d never needs either, since DAC's convolutions
// are all stride-1 and biased. `weight` is a Conv1dWeights.weight, ne =
// [kernel, in_channels, out_channels]; `bias`, when non-null, must be
// [out_channels].
ggml_tensor * conv1d(ggml_context * context, ggml_tensor * input, ggml_tensor * weight, ggml_tensor * bias,
                     int stride, int padding, int dilation) {
    if (context == nullptr || input == nullptr || weight == nullptr || stride <= 0 || padding < 0 || dilation <= 0) {
        return nullptr;
    }
    const int64_t kernel       = weight->ne[0];
    const int64_t in_channels  = weight->ne[1];
    const int64_t out_channels = weight->ne[2];
    const int64_t length       = input->ne[1];
    if (kernel <= 0 || length <= 0 || in_channels != input->ne[0] || out_channels <= 0 ||
        (bias != nullptr && bias->ne[0] != out_channels)) {
        return nullptr;
    }
    ggml_tensor * time_major = ggml_cont(context, ggml_transpose(context, input));
    ggml_tensor * columns =
        ggml_im2col(context, weight, time_major, stride, 0, padding, 0, dilation, 0, false, weight->type);
    ggml_tensor * kernel_2d = ggml_reshape_2d(context, weight, kernel * in_channels, out_channels);
    ggml_tensor * signal =
        ggml_mul_mat(context, kernel_2d, ggml_reshape_2d(context, columns, columns->ne[0], columns->ne[1]));
    return bias != nullptr ? add_channel_bias(context, signal, bias) : signal;
}

ggml_tensor * conv1d(ggml_context * context, ggml_tensor * input, const Conv1dWeights & weights, int stride,
                     int padding, int dilation) {
    return conv1d(context, input, weights.weight, weights.bias, stride, padding, dilation);
}

// HubertPositionalConvEmbedding's grouped convolution: `groups` even
// divisions of the channel axis, each an independent (ordinary) convolution
// over its own input slice into its own output slice. Neither ggml_conv_1d
// (no groups parameter) nor ggml_conv_1d_dw (groups == channels only) serves
// this, and codec.cpp already avoids the fused ops for other reasons, so this
// is `conv1d` run once per group with `ggml_view_*` slices of the weight and
// input, concatenated back together -- the number of groups here is small
// (16 at real scale) and this graph is built once per cloning request, not
// once per step, so the loop costs nothing worth avoiding.
ggml_tensor * grouped_conv1d(ggml_context * context, ggml_tensor * input, const Conv1dWeights & weights,
                             int padding) {
    if (context == nullptr || input == nullptr || !bound(weights, true)) {
        return nullptr;
    }
    const int64_t kernel      = weights.weight->ne[0];
    const int64_t in_per_group = weights.weight->ne[1];
    const int64_t out_total   = weights.weight->ne[2];
    const int64_t in_total    = input->ne[0];
    if (kernel <= 0 || in_per_group <= 0 || out_total <= 0 || in_total <= 0 || in_total % in_per_group != 0) {
        return nullptr;
    }
    const int64_t groups = in_total / in_per_group;
    if (groups <= 0 || out_total % groups != 0 || weights.bias->ne[0] != out_total) {
        return nullptr;
    }
    const int64_t out_per_group = out_total / groups;

    ggml_tensor * result = nullptr;
    for (int64_t group = 0; group < groups; ++group) {
        ggml_tensor * group_weight =
            ggml_view_3d(context, weights.weight, kernel, in_per_group, out_per_group, weights.weight->nb[1],
                        weights.weight->nb[2], size_t(group) * size_t(out_per_group) * weights.weight->nb[2]);
        ggml_tensor * group_input = ggml_view_2d(context, input, in_per_group, input->ne[1], input->nb[1],
                                                 size_t(group) * size_t(in_per_group) * input->nb[0]);
        ggml_tensor * group_bias =
            ggml_view_1d(context, weights.bias, out_per_group, size_t(group) * size_t(out_per_group) * weights.bias->nb[0]);
        ggml_tensor * group_output = conv1d(context, group_input, group_weight, group_bias, 1, padding, 1);
        if (group_output == nullptr) {
            return nullptr;
        }
        result = result == nullptr ? group_output : ggml_concat(context, result, group_output, 0);
    }
    return result;
}

// HubertPositionalConvEmbedding.forward: the grouped conv (padding =
// kernel/2), HubertSamePadLayer's trim (drop the LAST time step when the
// kernel is even -- odd kernels need no trim, since `kernel/2` integer
// division already yields a length-preserving pad on both sides), then
// GELU(erf) (feat_extract_activation, "gelu" -> exact erf, never the tanh
// approximation -- see reference-encoder.h and generator.cpp's own precedent
// of citing the exact activation used).
ggml_tensor * pos_conv_embed(ggml_context * context, ggml_tensor * hidden, const Conv1dWeights & weights) {
    if (context == nullptr || hidden == nullptr || weights.weight == nullptr) {
        return nullptr;
    }
    const int64_t kernel = weights.weight->ne[0];
    if (kernel <= 0) {
        return nullptr;
    }
    ggml_tensor * output = grouped_conv1d(context, hidden, weights, int(kernel / 2));
    if (output == nullptr) {
        return nullptr;
    }
    if (kernel % 2 == 0) {
        if (output->ne[1] < 2) {
            return nullptr;
        }
        // Truncating the trailing sample of the outermost (time) axis keeps
        // the view's own strides length-preserving-contiguous (nb1 is
        // unchanged, and the contiguity identity nb1 == nb0 * ne0 does not
        // depend on ne1), so no ggml_cont is needed here.
        output = ggml_view_2d(context, output, output->ne[0], output->ne[1] - 1, output->nb[1], 0);
    }
    return ggml_gelu_erf(context, output);
}

// HubertAttention + eager_attention_forward, bidirectional and cache-free
// (there is no other mode HuBERT ever runs in): q/k/v/out all biased and
// square, no rope, no per-head norm, no GQA. See this file's top comment for
// the full delta list against generator.cpp's generator_layer.
ggml_tensor * semantic_attention(ggml_context * context, ggml_tensor * hidden, const SemanticLayerWeights & weights,
                                 uint32_t heads, uint32_t head_dim) {
    if (context == nullptr || hidden == nullptr || !bound(weights.q_proj) || !bound(weights.k_proj) ||
        !bound(weights.v_proj) || !bound(weights.out_proj) || heads == 0 || head_dim == 0) {
        return nullptr;
    }
    const int64_t positions = hidden->ne[1];
    if (positions <= 0 || hidden->ne[0] != int64_t(heads) * int64_t(head_dim)) {
        return nullptr;
    }

    ggml_tensor * q = linear(context, hidden, weights.q_proj);
    ggml_tensor * k = linear(context, hidden, weights.k_proj);
    ggml_tensor * v = linear(context, hidden, weights.v_proj);
    if (q == nullptr || k == nullptr || v == nullptr) {
        return nullptr;
    }

    q = ggml_reshape_3d(context, q, int64_t(head_dim), int64_t(heads), positions);
    k = ggml_reshape_3d(context, k, int64_t(head_dim), int64_t(heads), positions);
    v = ggml_reshape_3d(context, v, int64_t(head_dim), int64_t(heads), positions);

    ggml_tensor * q_hd = ggml_cont(context, ggml_permute(context, q, 0, 2, 1, 3));
    ggml_tensor * k_hd = ggml_cont(context, ggml_permute(context, k, 0, 2, 1, 3));
    ggml_tensor * v_hd = ggml_cont(context, ggml_permute(context, v, 0, 2, 1, 3));

    ggml_tensor * scores = ggml_mul_mat(context, k_hd, q_hd);
    // HubertAttention's own scaling, head_dim**-0.5 -- full bidirectional
    // attention, so mask is null exactly as it is at generator_layer's own
    // call site when no mask is asked for.
    scores = ggml_soft_max_ext(context, scores, nullptr, 1.0f / std::sqrt(float(head_dim)), 0.0f);

    ggml_tensor * v_t      = ggml_cont(context, ggml_permute(context, v_hd, 1, 0, 2, 3));
    ggml_tensor * attended = ggml_mul_mat(context, v_t, scores);
    attended               = ggml_cont(context, ggml_permute(context, attended, 0, 2, 1, 3));
    attended               = ggml_reshape_2d(context, attended, int64_t(head_dim) * int64_t(heads), positions);

    return linear(context, attended, weights.out_proj);
}

// HubertFeedForward: intermediate_dense -> GELU(erf) -> output_dense (both
// dropouts are no-ops in eval).
ggml_tensor * semantic_feed_forward(ggml_context * context, ggml_tensor * hidden, const SemanticLayerWeights & weights) {
    if (context == nullptr || hidden == nullptr || !bound(weights.inter_dense) || !bound(weights.output_dense)) {
        return nullptr;
    }
    ggml_tensor * inter = linear(context, hidden, weights.inter_dense);
    if (inter == nullptr) {
        return nullptr;
    }
    inter = ggml_gelu_erf(context, inter);
    return linear(context, inter, weights.output_dense);
}

// HubertEncoderLayer.forward (do_stable_layer_norm = False): attention runs
// on the UNNORMALIZED input -- there is no pre-attention norm in this variant
// at all -- then +residual, THEN layer_norm; feed_forward runs on that
// already-normalized state, then +residual, then final_layer_norm. Contrast
// generator_layer, which normalizes BEFORE each branch and never normalizes
// the branch's own output.
ggml_tensor * semantic_layer(ggml_context * context, ggml_tensor * hidden, const SemanticLayerWeights & weights,
                             uint32_t heads, uint32_t head_dim, float eps) {
    if (context == nullptr || hidden == nullptr) {
        return nullptr;
    }
    ggml_tensor * attended = semantic_attention(context, hidden, weights, heads, head_dim);
    if (attended == nullptr) {
        return nullptr;
    }
    ggml_tensor * after_attention = layer_norm(context, ggml_add(context, hidden, attended), weights.layer_norm, eps);
    if (after_attention == nullptr) {
        return nullptr;
    }
    ggml_tensor * feed_forward = semantic_feed_forward(context, after_attention, weights);
    if (feed_forward == nullptr) {
        return nullptr;
    }
    return layer_norm(context, ggml_add(context, after_attention, feed_forward), weights.final_layer_norm, eps);
}

// HiggsAudioV2TokenizerResidualUnit.forward: ELU -> bias-free kernel-3 conv
// (dilation 1, pad 1 -- this checkpoint's block_dilations are fixed to
// [1, 1] by catalog.cpp's own resolution) -> ELU -> bias-free kernel-1 conv,
// added back to the branch's input.
ggml_tensor * semantic_encoder_res_unit(ggml_context * context, ggml_tensor * hidden,
                                        const SemanticEncoderResUnit & weights) {
    if (context == nullptr || hidden == nullptr || !bound(weights.conv1, false) || !bound(weights.conv2, false)) {
        return nullptr;
    }
    ggml_tensor * branch = ggml_elu(context, hidden);
    branch                = conv1d(context, branch, weights.conv1, 1, int((weights.conv1.weight->ne[0] - 1) / 2), 1);
    if (branch == nullptr) {
        return nullptr;
    }
    branch = ggml_elu(context, branch);
    branch = conv1d(context, branch, weights.conv2, 1, int((weights.conv2.weight->ne[0] - 1) / 2), 1);
    if (branch == nullptr || branch->ne[0] != hidden->ne[0] || branch->ne[1] != hidden->ne[1]) {
        return nullptr;
    }
    return ggml_add(context, hidden, branch);
}

// HiggsAudioV2TokenizerSemanticEncoderBlock.forward: every residual unit,
// then the block's own biased exit convolution (kernel 3, stride 1, pad 1
// for this checkpoint's fixed strides == [1, 1] -- catalog.cpp's
// kSemanticEncoderKernel).
ggml_tensor * semantic_encoder_block(ggml_context * context, ggml_tensor * hidden, const SemanticEncoderBlock & weights) {
    if (context == nullptr || hidden == nullptr || weights.res_units.empty() || !bound(weights.conv, true)) {
        return nullptr;
    }
    ggml_tensor * current = hidden;
    for (const SemanticEncoderResUnit & unit : weights.res_units) {
        current = semantic_encoder_res_unit(context, current, unit);
        if (current == nullptr) {
            return nullptr;
        }
    }
    return conv1d(context, current, weights.conv, 1, int((weights.conv.weight->ne[0] - 1) / 2), 1);
}

}  // namespace

ggml_tensor * build_semantic_branch(ggml_context * context, ggml_tensor * pcm_16k, const ModelWeights & weights,
                                    const HParams & hparams, std::vector<ggml_tensor *> * out_hidden_states,
                                    ggml_tensor ** out_mean, ggml_tensor ** out_downsampled) {
    if (out_hidden_states != nullptr) {
        out_hidden_states->clear();
    }
    if (out_mean != nullptr) {
        *out_mean = nullptr;
    }
    if (out_downsampled != nullptr) {
        *out_downsampled = nullptr;
    }
    if (context == nullptr || pcm_16k == nullptr || pcm_16k->type != GGML_TYPE_F32 || pcm_16k->ne[1] != 1 ||
        pcm_16k->ne[2] != 1 || pcm_16k->ne[3] != 1 || pcm_16k->ne[0] <= 0) {
        return nullptr;
    }

    const SemanticParams &         s      = hparams.semantic;
    const SemanticModelWeights &   hubert = weights.semantic_model;
    const SemanticEncoderWeights & sem_enc = weights.encoder_semantic;
    if (s.hidden_size == 0 || s.attention_head_count == 0 || s.hidden_size % s.attention_head_count != 0 ||
        s.layer_count == 0 || hubert.feat_conv.size() != s.conv_dim.size() ||
        s.conv_dim.size() != s.conv_kernel.size() || s.conv_dim.size() != s.conv_stride.size() ||
        hubert.feat_conv.empty() || hubert.layers.size() != s.layer_count || !bound(hubert.feat_conv_norm) ||
        !bound(hubert.feature_projection_norm) || !bound(hubert.feature_projection) ||
        !bound(hubert.pos_conv, true) || !bound(hubert.encoder_norm)) {
        return nullptr;
    }
    const uint32_t head_dim = s.hidden_size / s.attention_head_count;

    // HiggsAudioV2TokenizerModel._extract_semantic_features: pad, then read
    // the padded waveform as a single input channel.
    ggml_tensor * padded = ggml_pad_ext(context, pcm_16k, kSemanticPadSamples, kSemanticPadSamples, 0, 0, 0, 0, 0, 0);
    ggml_tensor * hidden = ggml_reshape_2d(context, padded, 1, padded->ne[0]);

    // HubertFeatureEncoder: one bias-free conv per layer, GroupNorm + GELU on
    // layer 0 only (feat_extract_norm = "group"), GELU alone thereafter.
    for (size_t index = 0; index < hubert.feat_conv.size(); ++index) {
        if (!bound(hubert.feat_conv[index], false)) {
            return nullptr;
        }
        hidden = conv1d(context, hidden, hubert.feat_conv[index], int(s.conv_stride[index]), 0, 1);
        if (hidden == nullptr) {
            return nullptr;
        }
        if (index == 0) {
            hidden = group_norm_per_channel(context, hidden, hubert.feat_conv_norm, kGroupNormEps);
            if (hidden == nullptr) {
                return nullptr;
            }
        }
        hidden = ggml_gelu_erf(context, hidden);
    }

    // HubertFeatureProjection: LayerNorm then a biased Linear (dropout is a
    // no-op in eval).
    hidden = layer_norm(context, hidden, hubert.feature_projection_norm, s.layer_norm_eps);
    if (hidden == nullptr) {
        return nullptr;
    }
    hidden = linear(context, hidden, hubert.feature_projection);
    if (hidden == nullptr) {
        return nullptr;
    }

    // HubertEncoder.forward (do_stable_layer_norm = False): add the
    // positional embedding, THEN this encoder's own layer_norm -- before the
    // layer stack, which is what distinguishes this variant from the stable-
    // layer-norm one this port does not implement.
    ggml_tensor * position_embedding = pos_conv_embed(context, hidden, hubert.pos_conv);
    if (position_embedding == nullptr || position_embedding->ne[0] != hidden->ne[0] ||
        position_embedding->ne[1] != hidden->ne[1]) {
        return nullptr;
    }
    hidden = ggml_add(context, hidden, position_embedding);
    hidden = layer_norm(context, hidden, hubert.encoder_norm, s.layer_norm_eps);
    if (hidden == nullptr) {
        return nullptr;
    }

    // The pre-layer-stack state is hidden state 0; every layer's own output
    // follows, matching HubertEncoder.forward's all_hidden_states order
    // exactly (see reference-encoder.h's pipeline comment for the telescoping
    // argument).
    std::vector<ggml_tensor *> states;
    states.reserve(size_t(s.layer_count) + 1);
    states.push_back(hidden);
    for (uint32_t layer = 0; layer < s.layer_count; ++layer) {
        hidden = semantic_layer(context, hidden, hubert.layers[layer], s.attention_head_count, head_dim,
                                s.layer_norm_eps);
        if (hidden == nullptr) {
            return nullptr;
        }
        states.push_back(hidden);
    }
    if (out_hidden_states != nullptr) {
        *out_hidden_states = states;
    }

    ggml_tensor * sum = states[0];
    for (size_t index = 1; index < states.size(); ++index) {
        sum = ggml_add(context, sum, states[index]);
    }
    ggml_tensor * mean = ggml_scale(context, sum, 1.0f / float(states.size()));
    if (out_mean != nullptr) {
        *out_mean = mean;
    }

    // Python's `[:, ::2, :]`: a strided view (every other position), taking
    // the FIRST of each pair -- ceil(T / 2) positions survive when T is odd.
    const int64_t total       = mean->ne[1];
    const int64_t downsampled_length = (total + kSemanticDownsampleFactor - 1) / kSemanticDownsampleFactor;
    ggml_tensor * downsampled =
        ggml_cont(context, ggml_view_2d(context, mean, mean->ne[0], downsampled_length,
                                        size_t(kSemanticDownsampleFactor) * mean->nb[1], 0));
    if (out_downsampled != nullptr) {
        *out_downsampled = downsampled;
    }

    // SemanticEncoder.forward: the bias-free entry convolution, then every
    // block in order.
    if (!bound(sem_enc.conv, false)) {
        return nullptr;
    }
    ggml_tensor * encoded = conv1d(context, downsampled, sem_enc.conv, 1, int((sem_enc.conv.weight->ne[0] - 1) / 2), 1);
    for (const SemanticEncoderBlock & block : sem_enc.blocks) {
        if (encoded == nullptr) {
            return nullptr;
        }
        encoded = semantic_encoder_block(context, encoded, block);
    }
    return encoded;
}

}  // namespace synth::omnivoice
