// The speech tokenizer's encoder half: waveform in, pre-quantization latents
// out. See codec-encoder.h for the three provenances this file transcribes and
// why naming them matters.
//
// Every convention below comes from Task 1's `conventions.json`, which resolved
// them once and verified them at run time against upstream's own modules --
// including running the padding formula against
// `_get_extra_padding_for_conv1d` for all fifteen convolutions at the real
// input lengths. The `modeling_mimi.py` line citations are carried through from
// there and re-checked against transformers==4.57.3 while this was written.
//
// Convolutions are built from ggml_im2col and a matrix multiply rather than
// ggml_conv_1d, whose CPU path asserts an F16 kernel and this family's are F32
// -- the same reason codec.cpp and speaker-encoder.cpp build theirs that way.
//
// The three hazards here, in the order they cost time:
//
//   * Causality is asymmetric. Upstream puts ALL of `padding_total` on the LEFT
//     and only the ragged `extra_padding` on the right (:331-333), which is not
//     the decoder's "pad symmetrically and keep the prefix". A symmetric pad
//     builds the same shapes from the same weights and a different encoder, and
//     nothing downstream can see it.
//   * The frame downsampler REPLICATE-pads (:1406-1415), alone on this path.
//     Zero-padding it corrupts exactly the first frame of every clip.
//   * The frame downsampler runs AFTER the transformer (:1456-1467), so the
//     transformer's sequence length is twice the frame count, not equal to it.
//     The transformer itself is the next commit; the geometry already accounts
//     for where it sits, because getting that wrong is a shape error and would
//     be baked in here otherwise.
//
// Nothing here names a channel width. Every extent is read off the weights,
// which catalog.cpp already checked against the package's own hyper-parameters.
// The scalars that ARE named are the ones the package publishes nowhere -- it
// carries no `codec.encoder.*` metadata namespace at all -- and each cites the
// checkpoint field it was read from.

#include "codec-encoder.h"

#include "ggml.h"
#include "operations.h"

#include <cmath>
#include <cstddef>
#include <vector>

namespace synth::qwen3tts {

namespace {

// speech_tokenizer/config.json's `encoder_config`, which the package does not
// republish. Read from the checkpoint at revision
// 5d83992436eae1d760afd27aff78a71d676296fc.
// One convolution's geometry, in the order the encode path applies them.
struct ConvStep {
    ggml_tensor * weight        = nullptr;
    ggml_tensor * bias          = nullptr;
    int64_t       stride        = 1;
    int64_t       dilation      = 1;
    bool          replicate_pad = false;
};

// A strided MimiConv1d is always built with `kernel_size=ratio * 2,
// stride=ratio` (modeling_mimi.py:465 for the SEANet stack, :1406-1415 for the
// frame downsampler), so the stride is half the kernel and the package needs to
// declare no strides at all. The catalog resolves the kernels {8, 10, 12, 16}
// and 4; halving them gives the strides {4, 5, 6, 8} and 2 --
// `reversed(config.upsampling_ratios)` at :456, which is the DECODER's list
// walked backwards. Reversed the other way the total downsampling is still 960
// and everything else differs, which the frame count cannot see.
int64_t stride_of(ggml_tensor * weight) {
    return weight == nullptr ? 0 : weight->ne[0] / 2;
}

// Every convolution on the encode path, in order. False when any is unbound.
//
// The residual bottleneck's dilations are both 1, at every stage. The
// checkpoint declares `dilation_growth_rate: 2`, and it is inert:
// MimiEncoder:461 passes `[config.dilation_growth_rate ** j, 1]` with j from
// `range(config.num_residual_layers)`, and num_residual_layers is 1, so j is
// only ever 0 and 2**0 is 1. A port that reads "growth rate 2" as per-stage
// dilations 1/2/4/8 gets a different encoder from the same weights and every
// shape still resolves.
bool encode_path(const CodecEncoderWeights & weights, std::vector<ConvStep> & steps, bool include_downsample) {
    steps.clear();
    auto push = [&steps](ggml_tensor * weight, ggml_tensor * bias, int64_t stride, bool replicate) {
        steps.push_back(ConvStep{ weight, bias, stride, 1, replicate });
        return weight != nullptr && stride > 0 && weight->ne[0] > 0;
    };

    bool ok = push(weights.stem.weight, weights.stem.bias, 1, false);
    if (weights.stages.empty()) {
        return false;
    }
    for (const CodecEncoderStage & stage : weights.stages) {
        ok = push(stage.bottleneck_in.weight, stage.bottleneck_in.bias, 1, false) && ok;
        ok = push(stage.bottleneck_out.weight, stage.bottleneck_out.bias, 1, false) && ok;
        ok = push(stage.stride_conv.weight, stage.stride_conv.bias, stride_of(stage.stride_conv.weight), false) && ok;
    }
    ok = push(weights.tail.weight, weights.tail.bias, 1, false) && ok;
    if (include_downsample) {
        // No bias: modeling_mimi.py:1406-1415 constructs it with `bias=False`,
        // which is why the catalog holds it as a bare tensor rather than a
        // Conv1dWeights.
        ok = push(weights.downsample, nullptr, stride_of(weights.downsample), true) && ok;
    }
    // Every convolution but the frame downsampler carries a bias.
    for (size_t index = 0; index + (include_downsample ? 1u : 0u) < steps.size(); ++index) {
        ok = steps[index].bias != nullptr && ok;
    }
    return ok;
}

// Upstream's own arithmetic, transcribed rather than simplified.
//
// `padding_total = effective_kernel - stride` is registered in MimiConv1d's
// constructor (modeling_mimi.py:237-246); `extra_padding` is
// _get_extra_padding_for_conv1d (:263-273); the output length is
// _get_output_length (:295-310).
//
// Upstream writes the frame count as `ceil((L - k + p)/s + 1) - 1`. The `+ 1`
// is INSIDE the ceiling and the `- 1` outside it, and since 1 is an integer the
// two cancel to a single ceiling. Transcribing the outer `- 1` without the
// inner `+ 1` yields a formula one whole stride short -- it makes extra_padding
// come out -1 for the stem where the true value is 0, turning a pad into a
// truncation. That is a mistake this plan has already made once, in Task 1's
// docstring, which is why the cancelled form is spelled out here.
struct ConvPadding {
    int64_t effective_kernel = 0;
    int64_t padding_left     = 0;
    int64_t padding_right    = 0;
    int64_t output_length    = 0;
};

int64_t ceil_div(int64_t numerator, int64_t denominator) {
    // Correct for a negative numerator too: `length - stride` goes negative for
    // a clip shorter than one stride, and a truncating C++ division would round
    // it the wrong way.
    const int64_t quotient = numerator / denominator;
    return (numerator % denominator != 0 && numerator > 0) ? quotient + 1 : quotient;
}

bool conv_padding(int64_t length, const ConvStep & step, ConvPadding & padding) {
    if (length <= 0 || step.stride <= 0 || step.dilation <= 0 || step.weight == nullptr) {
        return false;
    }
    const int64_t kernel = step.weight->ne[0];
    if (kernel <= 0) {
        return false;
    }
    padding.effective_kernel = (kernel - 1) * step.dilation + 1;
    const int64_t total      = padding.effective_kernel - step.stride;
    if (total < 0) {
        return false;
    }
    const int64_t frames       = ceil_div(length - padding.effective_kernel + total, step.stride);
    const int64_t ideal_length = frames * step.stride + padding.effective_kernel - total;
    const int64_t extra        = ideal_length - length;
    if (extra < 0) {
        return false;
    }
    padding.padding_left  = total;
    padding.padding_right = extra;
    padding.output_length = (length + total + extra - padding.effective_kernel) / step.stride + 1;
    // The cross-check on the transcription above: the whole expression collapses
    // to ceil(length / stride), which is also what makes the composition over
    // the five strides come out at ceil(samples / 1920). Checked rather than
    // substituted -- if the two ever disagree, the transcription is what is
    // wrong and this refuses instead of encoding it.
    return padding.output_length == ceil_div(length, step.stride) && padding.output_length > 0;
}

// Replicate padding along the length axis of a time-major [length, channels]
// signal: the first frame repeated `left` times and the last `right` times.
// `nn.functional.pad(..., mode="replicate")` edge-extends BOTH sides, so the
// right-hand `extra_padding` repeats the last frame rather than zero-filling.
ggml_tensor * replicate_pad_edge(ggml_context * context, ggml_tensor * time_major, int64_t left, int64_t right) {
    const int64_t length   = time_major->ne[0];
    const int64_t channels = time_major->ne[1];
    ggml_tensor * padded   = time_major;
    if (left > 0) {
        ggml_tensor * first = ggml_cont(context, ggml_view_2d(context, time_major, 1, channels, time_major->nb[1], 0));
        padded              = ggml_concat(context, ggml_repeat_4d(context, first, left, channels, 1, 1), padded, 0);
    }
    if (right > 0) {
        ggml_tensor * last = ggml_cont(context, ggml_view_2d(context, time_major, 1, channels, time_major->nb[1],
                                                             size_t(length - 1) * time_major->nb[0]));
        padded             = ggml_concat(context, padded, ggml_repeat_4d(context, last, right, channels, 1, 1), 0);
    }
    return padded;
}

// A bias over [channels, length] is per channel, so it broadcasts along the
// length rather than across it.
//
// `as_f32` rather than a raw add: every activation here is F32 and ggml's CPU
// binary_op has no F32+BF16 case, so a package storing this half at lower
// precision would abort rather than degrade. All 161 `codec.encoder.*` tensors
// in the real Base package are F32 today (measured with a GGUF read), so the
// cast is a no-op -- but that is exactly what was true of speaker-encoder.cpp's
// weights in its own unit test, and false of the package, and it aborted.
ggml_tensor * add_channel_bias(ggml_context * context, ggml_tensor * signal, ggml_tensor * bias) {
    ggml_tensor * bias_f32 = as_f32(context, bias);
    return ggml_add(context, signal, ggml_reshape_2d(context, bias_f32, bias_f32->ne[0], 1));
}

// One residual block: ELU, kernel-3 convolution down to dim/2, ELU, kernel-1
// convolution back up, added to the block's own unmodified input.
//
// The shortcut is an nn.Identity, not a 1x1 convolution: `use_conv_shortcut` is
// false for this checkpoint (modeling_mimi.py:422-425), so the skip is the
// input itself. The add closes the block at :441, BEFORE the stage's stride, so
// the residual is added at the stage's input rate.
ggml_tensor * residual_block(ggml_context * context, ggml_tensor * input, const CodecEncoderStage & stage) {
    ggml_tensor * hidden = codec_encoder_causal_conv1d(context, ggml_elu(context, input), stage.bottleneck_in.weight,
                                                       stage.bottleneck_in.bias, 1, 1, false);
    if (hidden == nullptr) {
        return nullptr;
    }
    hidden = codec_encoder_causal_conv1d(context, ggml_elu(context, hidden), stage.bottleneck_out.weight,
                                         stage.bottleneck_out.bias, 1, 1, false);
    if (hidden == nullptr || hidden->ne[0] != input->ne[0] || hidden->ne[1] != input->ne[1]) {
        return nullptr;
    }
    return ggml_add(context, input, hidden);
}

}  // namespace

bool codec_encoder_geometry(const CodecEncoderWeights & weights, int64_t samples, CodecEncoderGeometry & geometry) {
    geometry = CodecEncoderGeometry{};
    std::vector<ConvStep> steps;
    if (!encode_path(weights, steps, true) || samples <= 0) {
        return false;
    }

    int64_t samples_per_frame = 1;
    for (const ConvStep & step : steps) {
        samples_per_frame *= step.stride;
    }
    // A clip that does not fill one frame is refused rather than padded up to
    // one: upstream would happily encode forty milliseconds into a single
    // frame, and a caller that asks for that has made a mistake worth reporting
    // instead of a frame worth fabricating.
    if (samples < samples_per_frame) {
        return false;
    }

    int64_t length = samples;
    for (size_t index = 0; index < steps.size(); ++index) {
        ConvPadding padding;
        if (!conv_padding(length, steps[index], padding)) {
            return false;
        }
        // The transformer runs on the SEANet stack's output, which is every
        // convolution except the last -- the frame downsampler.
        if (index + 1 == steps.size()) {
            geometry.transformer_positions = length;
        }
        length = padding.output_length;
    }

    geometry.samples           = samples;
    geometry.samples_per_frame = samples_per_frame;
    geometry.frames            = length;
    return geometry.frames > 0 && geometry.transformer_positions > 0;
}

synth_status_t codec_encoder_check_waveform(const CodecEncoderWeights & weights,
                                            const float *               samples,
                                            size_t                      count,
                                            CodecEncoderGeometry &      geometry) {
    geometry = CodecEncoderGeometry{};
    if (samples == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (!codec_encoder_geometry(weights, int64_t(count), geometry)) {
        return SYNTH_ERR_INVALID_ARG;
    }
    for (size_t index = 0; index < count; ++index) {
        if (!std::isfinite(samples[index])) {
            geometry = CodecEncoderGeometry{};
            return SYNTH_ERR_INVALID_ARG;
        }
    }
    return SYNTH_OK;
}

ggml_tensor * codec_encoder_causal_conv1d(ggml_context * context,
                                          ggml_tensor *  input,
                                          ggml_tensor *  weight,
                                          ggml_tensor *  bias,
                                          int64_t        stride,
                                          int64_t        dilation,
                                          bool           replicate_pad) {
    if (context == nullptr || input == nullptr || weight == nullptr || stride <= 0 || dilation <= 0) {
        return nullptr;
    }
    const int64_t kernel       = weight->ne[0];
    const int64_t in_channels  = weight->ne[1];
    const int64_t out_channels = weight->ne[2];
    const int64_t length       = input->ne[1];
    if (kernel <= 0 || out_channels <= 0 || in_channels != input->ne[0] || input->ne[2] != 1 || input->ne[3] != 1) {
        return nullptr;
    }
    if (bias != nullptr && bias->ne[0] != out_channels) {
        return nullptr;
    }

    ConvStep step;
    step.weight   = weight;
    step.stride   = stride;
    step.dilation = dilation;
    ConvPadding padding;
    if (!conv_padding(length, step, padding)) {
        return nullptr;
    }

    // ggml pads and gathers along ne[0], so the signal goes time-major for the
    // padding and the im2col and comes back channel-major from the matrix
    // multiply.
    ggml_tensor * time_major = ggml_cont(context, ggml_transpose(context, input));
    ggml_tensor * padded     = time_major;
    if (replicate_pad) {
        padded = replicate_pad_edge(context, time_major, padding.padding_left, padding.padding_right);
    } else if (padding.padding_left > 0 || padding.padding_right > 0) {
        padded =
            ggml_pad_ext(context, time_major, int(padding.padding_left), int(padding.padding_right), 0, 0, 0, 0, 0, 0);
    }

    // Padded already, so im2col pads by nothing of its own. Its destination is
    // always F32, never `weight->type`: ggml_compute_forward_im2col's CPU switch
    // implements an F16 or F32 destination and nothing else, and the matrix
    // multiply below needs an F32 second operand regardless.
    ggml_tensor * columns =
        ggml_im2col(context, weight, padded, int(stride), 0, 0, 0, int(dilation), 0, false, GGML_TYPE_F32);
    ggml_tensor * kernel_2d = ggml_reshape_2d(context, weight, kernel * in_channels, out_channels);
    ggml_tensor * wide =
        ggml_mul_mat(context, kernel_2d, ggml_reshape_2d(context, columns, columns->ne[0], columns->ne[1]));
    // The padding arithmetic and im2col's own output-size arithmetic are two
    // independent derivations of the same number; a disagreement is a
    // transcription error, not a shape to accept.
    if (wide->ne[1] != padding.output_length) {
        return nullptr;
    }
    return bias == nullptr ? wide : add_channel_bias(context, wide, bias);
}

ggml_tensor * build_codec_encoder_seanet(ggml_context *              context,
                                         ggml_tensor *               waveform,
                                         const CodecEncoderWeights & weights,
                                         CodecEncoderTaps *          taps) {
    std::vector<ConvStep> steps;
    if (context == nullptr || waveform == nullptr || !encode_path(weights, steps, false)) {
        return nullptr;
    }
    if (waveform->type != GGML_TYPE_F32 || waveform->ne[0] != 1 || waveform->ne[2] != 1 || waveform->ne[3] != 1) {
        return nullptr;
    }

    // No activation before the stem: MimiEncoder opens with a bare MimiConv1d
    // (modeling_mimi.py:449) and the first nn.ELU() is not appended until :463.
    ggml_tensor * hidden =
        codec_encoder_causal_conv1d(context, waveform, weights.stem.weight, weights.stem.bias, 1, 1, false);
    for (const CodecEncoderStage & stage : weights.stages) {
        if (hidden == nullptr) {
            return nullptr;
        }
        hidden = residual_block(context, hidden, stage);
        if (hidden == nullptr) {
            return nullptr;
        }
        // ELU, then the stride: :463-465 appends them in that order, after the
        // residual block of the same stage.
        hidden = codec_encoder_causal_conv1d(context, ggml_elu(context, hidden), stage.stride_conv.weight,
                                             stage.stride_conv.bias, stride_of(stage.stride_conv.weight), 1, false);
        if (taps != nullptr) {
            taps->seanet_stages.push_back(hidden);
        }
    }
    if (hidden == nullptr) {
        return nullptr;
    }
    // One more ELU and the tail convolution (:468-470), and no activation after
    // it.
    ggml_tensor * tail = codec_encoder_causal_conv1d(context, ggml_elu(context, hidden), weights.tail.weight,
                                                     weights.tail.bias, 1, 1, false);
    if (taps != nullptr) {
        taps->seanet_tail = tail;
    }
    return tail;
}

ggml_tensor * build_codec_encoder_downsample(ggml_context *              context,
                                             ggml_tensor *               input,
                                             const CodecEncoderWeights & weights) {
    if (context == nullptr || input == nullptr || weights.downsample == nullptr) {
        return nullptr;
    }
    return codec_encoder_causal_conv1d(context, input, weights.downsample, nullptr, stride_of(weights.downsample), 1,
                                       true);
}

}  // namespace synth::qwen3tts
