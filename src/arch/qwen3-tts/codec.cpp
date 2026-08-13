// The codec decoder's convolution stack.
//
// Layout is channel-major, [channels, length], which is what the rest of this
// family already uses -- the transformer in the middle of the decoder wants
// [hidden, positions], so nothing has to be transposed across that boundary.
//
// Convolutions are built from ggml_im2col and a matrix multiply rather than
// ggml_conv_1d, whose CPU path asserts an F16 kernel and this family's are F32.
// The transposed form uses ggml_col2im_1d rather than the fused
// ggml_conv_transpose_1d, whose CUDA kernel is quadratic in the kernel width;
// see ggml-patches/README.md.
//
// THIS FILE ASSUMES ITS WEIGHTS ARE F32, in more places than the two obvious
// ones. Both ggml_im2col calls below request a destination of
// `weights.weight->type`, and add_channel_bias adds a raw bias to an F32
// signal; ggml_compute_forward_im2col's CPU switch implements only an F16 or
// F32 destination and the CPU binary_op has no F32+BF16 case, so either one
// ABORTS rather than degrades on a lower-precision weight. The same latent
// assumption is at roughly a dozen sites here, not two -- add_channel_bias
// itself plus every ggml_mul_mat whose second operand is weight-derived --
// so whoever wakes this must sweep the file rather than patch the pair.
//
// It is dormant, not fixed. The quantization policy holds every `codec.*`
// tensor at the package's reference dtype under every profile
// (tools/synthesize-quantize/policy.cpp, pinned by
// tests/qwen3_tts_quantization_policy_test.cpp), and that reference dtype is
// F32 for this checkpoint's codec half: all 416 `codec.*` tensors in the real
// Base package are F32, measured with a GGUF read rather than assumed. What
// would wake it is a package -- or a future profile -- that stores a codec
// weight as anything else. That is exactly what the speaker encoder's BF16
// tensors did to the same two assumptions in speaker-encoder.cpp, where they
// were found by aborting during Task 5 rather than by reading; see that
// file's own add_channel_bias and same_conv1d for the shape of the fix.
//
// Causality is the recurring hazard: every convolution here sees only the
// present and the past, which is left-only padding. im2col pads symmetrically,
// so each one pads wide and keeps the prefix. An off-by-one there shifts the
// waveform in time and fails nothing.

#include "codec.h"

#include "ggml.h"
#include "weights.h"

#include <cmath>
#include <iterator>

namespace synth::qwen3tts {

namespace {

bool bound(const Conv1dWeights & weights) {
    return weights.weight != nullptr && weights.bias != nullptr;
}

// A bias over [channels, length] is per channel, so it broadcasts along the
// length rather than across it.
ggml_tensor * add_channel_bias(ggml_context * context, ggml_tensor * signal, ggml_tensor * bias) {
    return ggml_add(context, signal, ggml_reshape_2d(context, bias, bias->ne[0], 1));
}

// Keeps the first `length` columns of a wider signal, which is what turns a
// symmetrically padded convolution into a causal one.
ggml_tensor * keep_prefix(ggml_context * context, ggml_tensor * signal, int64_t length) {
    if (signal->ne[1] < length) {
        return nullptr;
    }
    if (signal->ne[1] == length) {
        return signal;
    }
    return ggml_cont(context, ggml_view_2d(context, signal, signal->ne[0], length, signal->nb[1], 0));
}

}  // namespace

ggml_tensor * codec_causal_conv1d(ggml_context *        context,
                                  ggml_tensor *         input,
                                  const Conv1dWeights & weights,
                                  int                   dilation) {
    if (context == nullptr || input == nullptr || !bound(weights) || dilation <= 0) {
        return nullptr;
    }
    const int64_t kernel       = weights.weight->ne[0];
    const int64_t in_channels  = weights.weight->ne[1];
    const int64_t out_channels = weights.weight->ne[2];
    const int64_t length       = input->ne[1];
    if (kernel <= 0 || length <= 0 || in_channels != input->ne[0] || out_channels <= 0) {
        return nullptr;
    }

    // Padding on both sides by the whole receptive field puts the causal result
    // in the first `length` columns, followed by the right-padded tail.
    const int     padding    = static_cast<int>((kernel - 1) * dilation);
    ggml_tensor * time_major = ggml_cont(context, ggml_transpose(context, input));
    ggml_tensor * columns =
        ggml_im2col(context, weights.weight, time_major, 1, 0, padding, 0, dilation, 0, false, weights.weight->type);
    ggml_tensor * kernel_2d = ggml_reshape_2d(context, weights.weight, kernel * in_channels, out_channels);
    ggml_tensor * wide =
        ggml_mul_mat(context, kernel_2d, ggml_reshape_2d(context, columns, columns->ne[0], columns->ne[1]));

    ggml_tensor * causal = keep_prefix(context, wide, length);
    if (causal == nullptr) {
        return nullptr;
    }
    return add_channel_bias(context, causal, weights.bias);
}

ggml_tensor * codec_causal_depthwise_conv1d(ggml_context *        context,
                                            ggml_tensor *         input,
                                            const Conv1dWeights & weights) {
    if (context == nullptr || input == nullptr || !bound(weights)) {
        return nullptr;
    }
    const int64_t kernel   = weights.weight->ne[0];
    const int64_t channels = weights.weight->ne[2];
    const int64_t length   = input->ne[1];
    // One filter per channel: the kernel's input extent is one, not the channel
    // count, and the channel count cannot change across the operator.
    if (kernel <= 0 || length <= 0 || weights.weight->ne[1] != 1 || channels != input->ne[0]) {
        return nullptr;
    }

    const int     padding    = static_cast<int>(kernel - 1);
    ggml_tensor * time_major = ggml_cont(context, ggml_transpose(context, input));
    // Each channel is its own batch entry, which is what keeps the filters from
    // mixing across channels.
    ggml_tensor * batched    = ggml_reshape_4d(context, time_major, time_major->ne[0], 1, time_major->ne[1], 1);
    ggml_tensor * columns =
        ggml_im2col(context, weights.weight, batched, 1, 0, padding, 0, 1, 0, false, weights.weight->type);
    ggml_tensor * wide = ggml_mul_mat(context, weights.weight, columns);
    // [1, out_length, channels] down to [out_length, channels], then back to the
    // stack's channel-major layout.
    wide               = ggml_reshape_2d(context, wide, wide->ne[1], wide->ne[2]);
    wide               = ggml_cont(context, ggml_transpose(context, wide));

    ggml_tensor * causal = keep_prefix(context, wide, length);
    if (causal == nullptr) {
        return nullptr;
    }
    return add_channel_bias(context, causal, weights.bias);
}

ggml_tensor * codec_causal_transpose_conv1d(ggml_context *        context,
                                            ggml_tensor *         input,
                                            const Conv1dWeights & weights,
                                            int                   stride) {
    if (context == nullptr || input == nullptr || !bound(weights) || stride <= 0) {
        return nullptr;
    }
    // ConvTranspose1d stores [in, out, kernel], so ggml reports [kernel, out, in].
    const int64_t kernel       = weights.weight->ne[0];
    const int64_t out_channels = weights.weight->ne[1];
    const int64_t in_channels  = weights.weight->ne[2];
    const int64_t length       = input->ne[1];
    if (kernel < stride || out_channels <= 0 || in_channels != input->ne[0] || length <= 0 ||
        weights.weight->ne[3] != 1) {
        return nullptr;
    }
    if ((length - 1) * stride + kernel < 1) {
        return nullptr;
    }

    // Merging the kernel's leading pair yields exactly the out_channels-major,
    // kernel-minor column order col2im_1d scatters back; the transpose then puts
    // the input channels first so the matrix multiply reduces over them.
    ggml_tensor * columns_weight = ggml_reshape_2d(context, weights.weight, kernel * out_channels, in_channels);
    columns_weight               = ggml_cont(context, ggml_transpose(context, columns_weight));
    ggml_tensor * contiguous     = ggml_is_contiguous(input) ? input : ggml_cont(context, input);
    ggml_tensor * columns        = ggml_mul_mat(context, columns_weight, contiguous);
    // col2im_1d returns [out_length, out_channels].
    ggml_tensor * wide           = ggml_col2im_1d(context, columns, stride, static_cast<int>(out_channels), 0);

    // Causality is the crop: the reference drops the trailing `kernel - stride`
    // samples rather than padding symmetrically.
    const int64_t kept = wide->ne[0] - (kernel - stride);
    if (kept <= 0) {
        return nullptr;
    }
    ggml_tensor * causal = ggml_cont(context, ggml_view_2d(context, wide, kept, out_channels, wide->nb[1], 0));
    causal               = ggml_add(context, causal, ggml_reshape_2d(context, weights.bias, 1, out_channels));
    return ggml_cont(context, ggml_transpose(context, causal));
}

ggml_tensor * codec_snake_beta(ggml_context * context, ggml_tensor * input, const SnakeBetaWeights & weights) {
    if (context == nullptr || input == nullptr || weights.alpha == nullptr || weights.beta == nullptr) {
        return nullptr;
    }
    if (weights.alpha->ne[0] != input->ne[0] || weights.beta->ne[0] != input->ne[0]) {
        return nullptr;
    }
    // Both curves are per channel, and both are stored as logarithms, so neither
    // exponential can be folded into a neighbouring convolution.
    ggml_tensor * alpha = ggml_exp(context, ggml_reshape_2d(context, weights.alpha, weights.alpha->ne[0], 1));
    ggml_tensor * beta  = ggml_exp(context, ggml_reshape_2d(context, weights.beta, weights.beta->ne[0], 1));

    ggml_tensor * squared = ggml_sqr(context, ggml_sin(context, ggml_mul(context, input, alpha)));
    // The reference divides by beta plus a fixed guard, which keeps a zero beta
    // finite. It is part of the function, not a numerical courtesy.
    ggml_tensor * guard   = ggml_scale_bias(context, beta, 1.0f, 1e-9f);
    return ggml_add(context, input, ggml_div(context, squared, guard));
}

ggml_tensor * codec_convnext_block(ggml_context * context, ggml_tensor * input, const ConvNeXtWeights & weights) {
    if (context == nullptr || input == nullptr || weights.norm.weight == nullptr || weights.norm.bias == nullptr ||
        weights.pwconv1.weight == nullptr || weights.pwconv1.bias == nullptr || weights.pwconv2.weight == nullptr ||
        weights.pwconv2.bias == nullptr || weights.gamma == nullptr) {
        return nullptr;
    }
    ggml_tensor * hidden = codec_causal_depthwise_conv1d(context, input, weights.dwconv);
    if (hidden == nullptr) {
        return nullptr;
    }

    // The norm and both pointwise layers work across channels, and channels are
    // already this layout's leading axis, so none of it needs transposing.
    hidden = ggml_norm(context, hidden, 1e-6f);
    hidden = ggml_add(context, ggml_mul(context, hidden, weights.norm.weight), weights.norm.bias);
    hidden = ggml_add(context, ggml_mul_mat(context, weights.pwconv1.weight, hidden), weights.pwconv1.bias);
    // nn.GELU() is the exact erf form. ggml_gelu is the tanh approximation, which
    // differs by around 2e-4 here -- small, but larger than everything else in
    // this stack disagrees by, and it compounds over sixty of these blocks.
    hidden = ggml_gelu_erf(context, hidden);
    hidden = ggml_add(context, ggml_mul_mat(context, weights.pwconv2.weight, hidden), weights.pwconv2.bias);
    hidden = ggml_mul(context, hidden, weights.gamma);
    return ggml_add(context, input, hidden);
}

ggml_tensor * codec_quantizer_decode(ggml_context *                context,
                                     const CodecQuantizerWeights & weights,
                                     ggml_tensor *                 codes) {
    if (context == nullptr || codes == nullptr || codes->type != GGML_TYPE_I32 || weights.output_proj == nullptr ||
        weights.codebooks.empty()) {
        return nullptr;
    }
    if (codes->ne[1] != static_cast<int64_t>(weights.codebooks.size())) {
        return nullptr;
    }

    // Residual quantization: every level refines the one before it, so the
    // levels are summed rather than concatenated.
    ggml_tensor * total = nullptr;
    for (size_t level = 0; level < weights.codebooks.size(); ++level) {
        if (weights.codebooks[level] == nullptr) {
            return nullptr;
        }
        ggml_tensor * ids = ggml_view_1d(context, codes, codes->ne[0], static_cast<size_t>(level) * codes->nb[1]);
        ggml_tensor * row = ggml_get_rows(context, weights.codebooks[level], ids);
        total             = total == nullptr ? row : ggml_add(context, total, row);
    }

    // The output projection is a kernel-one convolution, which at that width is a
    // matrix multiply; it carries no bias.
    ggml_tensor * projection =
        ggml_reshape_2d(context, weights.output_proj, weights.output_proj->ne[1], weights.output_proj->ne[2]);
    return ggml_mul_mat(context, projection, total);
}

void codec_fill_sliding_window_mask(float * mask, int64_t frames, int64_t window) {
    if (mask == nullptr || frames <= 0) {
        return;
    }
    for (int64_t query = 0; query < frames; ++query) {
        for (int64_t key = 0; key < frames; ++key) {
            // Causal, and no further back than the window. A window of zero is
            // taken as no window at all rather than as a query that sees nothing.
            const bool causal          = key <= query;
            const bool near            = window <= 0 || key > query - window;
            mask[query * frames + key] = causal && near ? 0.0f : -INFINITY;
        }
    }
}

ggml_tensor * codec_transformer_layer(ggml_context *                       context,
                                      ggml_tensor *                        input,
                                      ggml_tensor *                        position_ids,
                                      ggml_tensor *                        mask,
                                      const CodecTransformerLayerWeights & weights,
                                      const AttentionShape &               shape) {
    if (context == nullptr || input == nullptr || position_ids == nullptr || mask == nullptr ||
        weights.input_layernorm == nullptr || weights.q_proj == nullptr || weights.k_proj == nullptr ||
        weights.v_proj == nullptr || weights.o_proj == nullptr || weights.self_attn_layer_scale == nullptr ||
        weights.post_attention_layernorm == nullptr || weights.gate_proj == nullptr || weights.up_proj == nullptr ||
        weights.down_proj == nullptr || weights.mlp_layer_scale == nullptr) {
        return nullptr;
    }
    const int64_t hidden    = static_cast<int64_t>(shape.hidden_size);
    const int64_t head_dim  = static_cast<int64_t>(shape.head_dim);
    const int64_t q_heads   = static_cast<int64_t>(shape.attention_head_count);
    const int64_t kv_heads  = static_cast<int64_t>(shape.key_value_head_count);
    const int64_t positions = input->ne[1];
    if (hidden <= 0 || head_dim <= 0 || q_heads <= 0 || kv_heads <= 0 || positions <= 0 || input->ne[0] != hidden ||
        q_heads % kv_heads != 0) {
        return nullptr;
    }
    if (position_ids->type != GGML_TYPE_I32 || position_ids->ne[0] != positions || mask->type != GGML_TYPE_F32 ||
        mask->ne[0] != positions || mask->ne[1] < positions) {
        return nullptr;
    }

    ggml_tensor * residual = input;
    ggml_tensor * normed   = rms_norm(context, input, weights.input_layernorm, shape.rms_norm_eps);

    ggml_tensor * q = ggml_mul_mat(context, weights.q_proj, normed);
    ggml_tensor * k = ggml_mul_mat(context, weights.k_proj, normed);
    ggml_tensor * v = ggml_mul_mat(context, weights.v_proj, normed);

    q = ggml_reshape_3d(context, q, head_dim, q_heads, positions);
    k = ggml_reshape_3d(context, k, head_dim, kv_heads, positions);
    v = ggml_reshape_3d(context, v, head_dim, kv_heads, positions);

    // No per-head norms here: this block is not the Qwen3 one, and adding them
    // would renormalize vectors the reference leaves alone.
    q = ggml_rope_ext(context, q, position_ids, nullptr, static_cast<int>(head_dim), GGML_ROPE_TYPE_NEOX, 0,
                      shape.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(context, k, position_ids, nullptr, static_cast<int>(head_dim), GGML_ROPE_TYPE_NEOX, 0,
                      shape.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    ggml_tensor * q_hd = ggml_cont(context, ggml_permute(context, q, 0, 2, 1, 3));
    ggml_tensor * k_hd = ggml_cont(context, ggml_permute(context, k, 0, 2, 1, 3));
    ggml_tensor * v_hd = ggml_cont(context, ggml_permute(context, v, 0, 2, 1, 3));

    ggml_tensor * scores = ggml_mul_mat(context, k_hd, q_hd);
    scores = ggml_soft_max_ext(context, scores, mask, 1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f);

    ggml_tensor * v_t      = ggml_cont(context, ggml_permute(context, v_hd, 1, 0, 2, 3));
    ggml_tensor * attended = ggml_mul_mat(context, v_t, scores);
    attended               = ggml_cont(context, ggml_permute(context, attended, 0, 2, 1, 3));
    attended               = ggml_reshape_2d(context, attended, head_dim * q_heads, positions);

    // Each residual branch is scaled by a learned per-channel vector before it is
    // added back, which is what a Qwen3 block does not do.
    ggml_tensor * projected = ggml_mul_mat(context, weights.o_proj, attended);
    ggml_tensor * after_attention =
        ggml_add(context, residual, ggml_mul(context, projected, weights.self_attn_layer_scale));

    ggml_tensor * mlp_in = rms_norm(context, after_attention, weights.post_attention_layernorm, shape.rms_norm_eps);
    ggml_tensor * gate   = ggml_silu(context, ggml_mul_mat(context, weights.gate_proj, mlp_in));
    ggml_tensor * up     = ggml_mul_mat(context, weights.up_proj, mlp_in);
    ggml_tensor * mlp    = ggml_mul_mat(context, weights.down_proj, ggml_mul(context, gate, up));
    return ggml_add(context, after_attention, ggml_mul(context, mlp, weights.mlp_layer_scale));
}

namespace {

// A Linear over [in, positions] with its bias.
ggml_tensor * linear(ggml_context * context, ggml_tensor * input, const LinearWeights & weights) {
    if (weights.weight == nullptr || weights.bias == nullptr) {
        return nullptr;
    }
    return ggml_add(context, ggml_mul_mat(context, weights.weight, input), weights.bias);
}

}  // namespace

ggml_tensor * build_codec_decoder(ggml_context *              context,
                                  ggml_tensor *               codes,
                                  ggml_tensor *               position_ids,
                                  ggml_tensor *               mask,
                                  const CodecDecoderWeights & weights,
                                  const HParams &             hparams) {
    const CodecDecoderParams & p = hparams.codec.decoder;
    if (context == nullptr || codes == nullptr || codes->type != GGML_TYPE_I32 ||
        codes->ne[1] != static_cast<int64_t>(p.quantizer_count) ||
        weights.pre_transformer.layers.size() != p.layer_count ||
        weights.upsample.size() != p.upsampling_ratios.size() || weights.stages.size() != p.upsample_rates.size() ||
        weights.pre_transformer.norm == nullptr) {
        return nullptr;
    }

    // The semantic quantizer covers code group 0 and the acoustic one the rest,
    // and the two are summed -- the same split the talker and the predictor make.
    const int64_t semantic_count = static_cast<int64_t>(p.semantic_quantizer_count);
    ggml_tensor * semantic_codes = ggml_view_2d(context, codes, codes->ne[0], semantic_count, codes->nb[1], 0);
    ggml_tensor * acoustic_codes = ggml_view_2d(context, codes, codes->ne[0], codes->ne[1] - semantic_count,
                                                codes->nb[1], static_cast<size_t>(semantic_count) * codes->nb[1]);
    ggml_tensor * semantic       = codec_quantizer_decode(context, weights.semantic, semantic_codes);
    ggml_tensor * acoustic       = codec_quantizer_decode(context, weights.acoustic, acoustic_codes);
    if (semantic == nullptr || acoustic == nullptr) {
        return nullptr;
    }
    ggml_tensor * hidden = ggml_add(context, semantic, acoustic);

    hidden = codec_causal_conv1d(context, hidden, weights.pre_conv, 1);
    if (hidden == nullptr) {
        return nullptr;
    }

    // The transformer runs at its own narrower width, with a projection on either
    // side of it.
    AttentionShape shape;
    shape.hidden_size          = p.hidden_size;
    shape.attention_head_count = p.attention_head_count;
    shape.key_value_head_count = p.key_value_head_count;
    shape.head_dim             = p.head_dim;
    shape.rms_norm_eps         = p.rms_norm_eps;
    shape.rope_theta           = p.rope_theta;

    hidden = linear(context, hidden, weights.pre_transformer.input_proj);
    if (hidden == nullptr) {
        return nullptr;
    }
    for (const CodecTransformerLayerWeights & layer : weights.pre_transformer.layers) {
        hidden = codec_transformer_layer(context, hidden, position_ids, mask, layer, shape);
        if (hidden == nullptr) {
            return nullptr;
        }
    }
    hidden = rms_norm(context, hidden, weights.pre_transformer.norm, p.rms_norm_eps);
    hidden = linear(context, hidden, weights.pre_transformer.output_proj);
    if (hidden == nullptr) {
        return nullptr;
    }

    // Two ConvNeXt stages at the latent width, each upsampling by its ratio.
    for (size_t stage = 0; stage < weights.upsample.size(); ++stage) {
        hidden = codec_causal_transpose_conv1d(context, hidden, weights.upsample[stage].transpose_conv,
                                               static_cast<int>(p.upsampling_ratios[stage]));
        if (hidden == nullptr) {
            return nullptr;
        }
        hidden = codec_convnext_block(context, hidden, weights.upsample[stage].convnext);
        if (hidden == nullptr) {
            return nullptr;
        }
    }

    // Then the residual stack: one block per upsample rate, each halving the
    // channel width while multiplying the length.
    hidden = codec_causal_conv1d(context, hidden, weights.input_conv, 1);
    if (hidden == nullptr) {
        return nullptr;
    }
    for (size_t stage = 0; stage < weights.stages.size(); ++stage) {
        const CodecResidualStage & residual = weights.stages[stage];
        hidden                              = codec_snake_beta(context, hidden, residual.act);
        if (hidden == nullptr) {
            return nullptr;
        }
        hidden = codec_causal_transpose_conv1d(context, hidden, residual.transpose_conv,
                                               static_cast<int>(p.upsample_rates[stage]));
        if (hidden == nullptr) {
            return nullptr;
        }
        // Three units at dilations 1, 3 and 9, each adding back to its input.
        static constexpr int kDilations[] = { 1, 3, 9 };
        if (residual.units.size() != std::size(kDilations)) {
            return nullptr;
        }
        for (size_t unit = 0; unit < residual.units.size(); ++unit) {
            const CodecResidualUnit & target = residual.units[unit];
            ggml_tensor *             branch = codec_snake_beta(context, hidden, target.act1);
            if (branch == nullptr) {
                return nullptr;
            }
            branch = codec_causal_conv1d(context, branch, target.conv1, kDilations[unit]);
            if (branch == nullptr) {
                return nullptr;
            }
            branch = codec_snake_beta(context, branch, target.act2);
            if (branch == nullptr) {
                return nullptr;
            }
            branch = codec_causal_conv1d(context, branch, target.conv2, 1);
            if (branch == nullptr) {
                return nullptr;
            }
            hidden = ggml_add(context, hidden, branch);
        }
    }

    hidden = codec_snake_beta(context, hidden, weights.output_act);
    if (hidden == nullptr) {
        return nullptr;
    }
    hidden = codec_causal_conv1d(context, hidden, weights.output_conv, 1);
    if (hidden == nullptr) {
        return nullptr;
    }
    // The reference clamps, which is what keeps a diverged frame from leaving the
    // sample range rather than a courtesy.
    return ggml_clamp(context, hidden, -1.0f, 1.0f);
}

}  // namespace synth::qwen3tts
