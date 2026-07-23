#include "duration-predictor.h"

#include "ggml.h"
#include "operations.h"
#include "voice-conditioning.h"
#include "weights.h"

#include <cmath>
#include <cstdio>
#include <limits>

namespace synth::vits {

namespace {

ggml_tensor * channel_view(ggml_context * context, ggml_tensor * input, int64_t first_channel, int64_t channel_count) {
    ggml_tensor * view = ggml_view_2d(context, input, channel_count, input->ne[1], input->nb[1],
                                      static_cast<size_t>(first_channel) * sizeof(float));
    return ggml_cont(context, view);
}

ggml_tensor * constant_like(ggml_context * context, ggml_tensor * input, float value) {
    return ggml_scale_bias(context, input, 0.0f, value);
}

ggml_tensor * dds(ggml_context *     context,
                  ggml_tensor *      input,
                  ggml_tensor *      conditioning,
                  const DDSWeights & weights,
                  float              epsilon) {
    ggml_tensor * output   = conditioning == nullptr ? input : ggml_add(context, input, conditioning);
    int           dilation = 1;
    for (const DDSBlockWeights & block : weights.blocks) {
        ggml_tensor * branch = depthwise_conv1d(context, output, block.depthwise, dilation, dilation);
        branch               = layer_norm(context, branch, block.depthwise_norm, epsilon);
        branch               = ggml_gelu_erf(context, branch);
        branch               = conv1d(context, branch, block.pointwise, 0);
        branch               = layer_norm(context, branch, block.pointwise_norm, epsilon);
        branch               = ggml_gelu_erf(context, branch);
        output               = ggml_add(context, output, branch);
        dilation *= 3;
    }
    return output;
}

ggml_tensor * select_bins(ggml_context * context, ggml_tensor * values, ggml_tensor * selection) {
    return ggml_sum_rows(context, ggml_mul(context, values, selection));
}

ggml_tensor * inverse_spline(ggml_context *  context,
                             ggml_tensor *   input,
                             ggml_tensor *   parameters,
                             const HParams & hparams) {
    const int64_t bins            = hparams.duration_spline_bin_count;
    const float   left            = -hparams.duration_spline_tail_bound;
    const float   span            = 2.0f * hparams.duration_spline_tail_bound;
    const float   parameter_scale = 1.0f / std::sqrt(static_cast<float>(hparams.hidden_channels));

    ggml_tensor * width_logits      = channel_view(context, parameters, 0, bins);
    ggml_tensor * height_logits     = channel_view(context, parameters, bins, bins);
    ggml_tensor * derivative_logits = channel_view(context, parameters, 2 * bins, bins - 1);

    ggml_tensor * widths = ggml_soft_max(context, ggml_scale(context, width_logits, parameter_scale));
    widths               = ggml_scale_bias(context, widths, 1.0f - hparams.duration_spline_min_bin_width * bins,
                                           hparams.duration_spline_min_bin_width);
    widths               = ggml_scale(context, widths, span);
    ggml_tensor * cumulative_widths = ggml_scale_bias(context, ggml_cumsum(context, widths), 1.0f, left);
    ggml_tensor * lower_widths      = ggml_concat(context, constant_like(context, input, left),
                                                  channel_view(context, cumulative_widths, 0, bins - 1), 0);

    ggml_tensor * heights = ggml_soft_max(context, ggml_scale(context, height_logits, parameter_scale));
    heights               = ggml_scale_bias(context, heights, 1.0f - hparams.duration_spline_min_bin_height * bins,
                                            hparams.duration_spline_min_bin_height);
    heights               = ggml_scale(context, heights, span);
    ggml_tensor * cumulative_heights = ggml_scale_bias(context, ggml_cumsum(context, heights), 1.0f, left);
    ggml_tensor * lower_heights      = ggml_concat(context, constant_like(context, input, left),
                                                   channel_view(context, cumulative_heights, 0, bins - 1), 0);

    const float   endpoint    = std::log(std::expm1(1.0f - hparams.duration_spline_min_derivative));
    ggml_tensor * derivatives = ggml_concat(context, constant_like(context, input, endpoint), derivative_logits, 0);
    derivatives               = ggml_concat(context, derivatives, constant_like(context, input, endpoint), 0);
    derivatives =
        ggml_scale_bias(context, ggml_softplus(context, derivatives), 1.0f, hparams.duration_spline_min_derivative);
    ggml_tensor * lower_derivatives = channel_view(context, derivatives, 0, bins);
    ggml_tensor * upper_derivatives = channel_view(context, derivatives, 1, bins);

    ggml_tensor * repeated_input = ggml_repeat(context, input, lower_heights);
    ggml_tensor * after_lower =
        ggml_step(context, ggml_scale_bias(context, ggml_sub(context, repeated_input, lower_heights), 1.0f, 1.0e-6f));
    ggml_tensor * after_upper = ggml_step(
        context, ggml_scale_bias(context, ggml_sub(context, repeated_input, cumulative_heights), 1.0f, 1.0e-6f));
    ggml_tensor * selection = ggml_sub(context, after_lower, after_upper);
    ggml_tensor * inside    = ggml_sum_rows(context, selection);
    ggml_tensor * outside   = ggml_scale_bias(context, inside, -1.0f, 1.0f);
    selection = ggml_add(context, selection, ggml_pad(context, outside, static_cast<int>(bins - 1), 0, 0, 0));

    ggml_tensor * input_cumulative_width  = select_bins(context, lower_widths, selection);
    ggml_tensor * input_width             = select_bins(context, widths, selection);
    ggml_tensor * input_cumulative_height = select_bins(context, lower_heights, selection);
    ggml_tensor * input_height            = select_bins(context, heights, selection);
    ggml_tensor * delta                   = ggml_div(context, heights, widths);
    ggml_tensor * input_delta             = select_bins(context, delta, selection);
    ggml_tensor * input_derivative        = select_bins(context, lower_derivatives, selection);
    ggml_tensor * input_derivative_next   = select_bins(context, upper_derivatives, selection);

    ggml_tensor * input_offset = ggml_sub(context, input, input_cumulative_height);
    ggml_tensor * common       = ggml_sub(context, ggml_add(context, input_derivative, input_derivative_next),
                                          ggml_scale(context, input_delta, 2.0f));
    ggml_tensor * a = ggml_add(context, ggml_mul(context, input_offset, common),
                               ggml_mul(context, input_height, ggml_sub(context, input_delta, input_derivative)));
    ggml_tensor * b =
        ggml_sub(context, ggml_mul(context, input_height, input_derivative), ggml_mul(context, input_offset, common));
    ggml_tensor * c = ggml_neg(context, ggml_mul(context, input_delta, input_offset));
    ggml_tensor * discriminant =
        ggml_sub(context, ggml_sqr(context, b), ggml_scale(context, ggml_mul(context, a, c), 4.0f));
    discriminant              = ggml_clamp(context, discriminant, 0.0f, std::numeric_limits<float>::max());
    ggml_tensor * denominator = ggml_sub(context, ggml_neg(context, b), ggml_sqrt(context, discriminant));
    ggml_tensor * root        = ggml_div(context, ggml_scale(context, c, 2.0f), denominator);
    ggml_tensor * transformed = ggml_add(context, ggml_mul(context, root, input_width), input_cumulative_width);
    return ggml_add(context, ggml_mul(context, inside, transformed), ggml_mul(context, outside, input));
}

ggml_tensor * flip_channels(ggml_context * context, ggml_tensor * input) {
    return ggml_concat(context, channel_view(context, input, 1, 1), channel_view(context, input, 0, 1), 0);
}

ggml_tensor * inverse_conv_flow(ggml_context *              context,
                                ggml_tensor *               input,
                                ggml_tensor *               conditioning,
                                const DurationFlowWeights & weights,
                                const HParams &             hparams) {
    ggml_tensor * first  = channel_view(context, input, 0, 1);
    ggml_tensor * second = channel_view(context, input, 1, 1);
    ggml_tensor * hidden = conv1d(context, first, weights.pre, 0);
    hidden               = dds(context, hidden, conditioning, weights.dds, hparams.layer_norm_epsilon);
    hidden               = conv1d(context, hidden, weights.projection, 0);
    second               = inverse_spline(context, second, hidden, hparams);
    return ggml_concat(context, first, second, 0);
}

}  // namespace

DurationPredictorGraph build_duration_predictor_graph(ggml_context *          context,
                                                      ggml_tensor *           encoded,
                                                      const DurationWeights & weights,
                                                      const HParams &         hparams,
                                                      int64_t                 token_count,
                                                      float                   noise_scale_w) {
    const VoiceWeights no_voice;
    return build_duration_predictor_graph(context, encoded, weights, no_voice, hparams, UINT32_MAX, token_count,
                                          noise_scale_w);
}

DurationPredictorGraph build_duration_predictor_graph(ggml_context *          context,
                                                      ggml_tensor *           encoded,
                                                      const DurationWeights & weights,
                                                      const VoiceWeights &    voice_weights,
                                                      const HParams &         hparams,
                                                      uint32_t                speaker_index,
                                                      int64_t                 token_count,
                                                      float                   noise_scale_w) {
    DurationPredictorGraph result;
    if (context == nullptr || encoded == nullptr || encoded->type != GGML_TYPE_F32 ||
        encoded->ne[0] != hparams.hidden_channels || encoded->ne[1] != token_count || token_count <= 0 ||
        static_cast<uint64_t>(token_count) > hparams.max_input_tokens || !std::isfinite(noise_scale_w) ||
        noise_scale_w < 0.0f || weights.dds.blocks.size() != hparams.duration_dds_layer_count ||
        weights.flows.size() != hparams.duration_flow_count) {
        std::fprintf(stderr, "vits: invalid duration-predictor graph request\n");
        return result;
    }
    for (const DurationFlowWeights & flow : weights.flows) {
        if (flow.dds.blocks.size() != hparams.duration_dds_layer_count) {
            std::fprintf(stderr, "vits: invalid duration flow weight catalog\n");
            return result;
        }
    }

    const VoiceConditioning voice = build_voice_conditioning(context, voice_weights, hparams, speaker_index);
    if (!voice.valid || (voice.embedding != nullptr &&
                         (weights.conditioning.weight == nullptr || weights.conditioning.bias == nullptr))) {
        std::fprintf(stderr, "vits: invalid duration-predictor Voice conditioning\n");
        return result;
    }
    result.speaker_index = voice.speaker_index;

    ggml_tensor * conditioning = conv1d(context, encoded, weights.pre, 0);
    if (voice.embedding != nullptr) {
        conditioning = ggml_add(context, conditioning, conv1d(context, voice.embedding, weights.conditioning, 0));
    }
    conditioning = dds(context, conditioning, nullptr, weights.dds, hparams.layer_norm_epsilon);
    conditioning = conv1d(context, conditioning, weights.projection, 0);

    result.duration_noise = ggml_new_tensor_2d(context, GGML_TYPE_F32, 2, token_count);
    ggml_set_name(result.duration_noise, "random.duration_noise");
    ggml_set_input(result.duration_noise);
    ggml_tensor * latent = ggml_scale(context, result.duration_noise, noise_scale_w);
    for (size_t flow = weights.flows.size(); flow-- > 0;) {
        latent = flip_channels(context, latent);
        latent = inverse_conv_flow(context, latent, conditioning, weights.flows[flow], hparams);
    }
    latent                         = flip_channels(context, latent);
    ggml_tensor * affine_bias      = ggml_reshape_2d(context, weights.affine_bias, 2, 1);
    ggml_tensor * affine_log_scale = ggml_reshape_2d(context, weights.affine_log_scale, 2, 1);
    latent                         = ggml_mul(context, ggml_sub(context, latent, affine_bias),
                                              ggml_exp(context, ggml_neg(context, affine_log_scale)));

    result.logw = ggml_cont(context, channel_view(context, latent, 0, 1));
    ggml_set_name(result.logw, "duration.logw");
    result.graph = ggml_new_graph_custom(context, 16384, false);
    ggml_build_forward_expand(result.graph, result.logw);
    return result;
}

}  // namespace synth::vits
