// The ECAPA-TDNN speaker encoder: a mel spectrogram in, a fixed-width speaker
// embedding out.
//
// Transcribed from the pinned upstream revision
// 022e286b98fbec7e1e916cb940cdf532cd9f488e, file
// qwen_tts/core/models/modeling_qwen3_tts.py:
//
//   :247-267  TimeDelayNetBlock -- Conv1d(padding="same", padding_mode="reflect")
//             followed by ReLU. Every convolution in this graph is one of these
//             or a bare Conv1d with the same padding; there is not one
//             normalization tensor among the 76, which is a fact about the
//             architecture and not an omission from the package.
//   :95-126   Res2NetBlock -- `scale` equal splits along channels. Split 0
//             passes through UNCONVOLVED and leads the concatenation; split 1
//             is convolved alone; every later split is convolved after the
//             PREVIOUS split's output is added to it. Hence `scale - 1`
//             convolutions rather than `scale`.
//   :129-156  SqueezeExcitationBlock -- mean over time, 1x1 down to the
//             bottleneck, ReLU, 1x1 back up, sigmoid, multiplied into the
//             signal along time.
//   :269-308  SqueezeExcitationRes2NetBlock -- tdnn1, res2net, tdnn2, squeeze
//             excite, and then the block's own input ADDED BACK as a residual.
//   :159-245  AttentiveStatisticsPooling -- see the pooling section below.
//   :311-393  Qwen3TTSSpeakerEncoder -- the stem, three SE-Res2Net blocks, a
//             multi-layer aggregation of blocks 1..3 only (the stem's output is
//             dropped from the concatenation), pooling, then fc.
//   configuration_qwen3_tts.py:47-67 -- enc_kernel_sizes [5,3,3,3,1] and
//             enc_dilations [1,2,3,4,1]. A dilation is not a weight, so the
//             package cannot declare one and the resolver cannot see one; a
//             dilation guessed wrong still yields a finite, plausible x-vector.
//
// The reading above was confirmed against upstream's own modules rather than
// taken off the page: a numpy transcription of exactly this graph reproduced
// Qwen3TTSSpeakerEncoder's output on random weights to 1.2e-6.
//
// Two things ggml does not hand over for free:
//
//   * `padding_mode="reflect"` is not zero padding. Every convolution here
//     reflect-pads first and then convolves with no padding of its own. Zero
//     padding would be a silent error: it changes only the edge frames, which
//     is a fraction of a percent of a long clip's pooled statistics and none of
//     its plausibility.
//   * `ggml_conv_1d`'s CPU path asserts an F16 kernel, so convolutions are
//     built from ggml_im2col and a matrix multiply instead, the way codec.cpp
//     builds the decoder's. A kernel of one needs neither: it is already a
//     matrix multiply over channels, and 16 of this graph's 38 convolutions
//     have one, so im2col is built 22 times rather than 38. The 22 are the
//     5-wide stem and the three blocks' seven 3-wide res2net splits apiece;
//     counted by instrumenting same_conv1d, not by reading the list.
//   * im2col's own destination type must be F32, not the kernel's storage
//     type: this package's speaker_encoder tensors are BF16 (measured with
//     gguf-dump against the real Base package; only the codec decoder is kept
//     F32 by this checkpoint's own quantization policy), and
//     ggml_compute_forward_im2col's CPU switch only implements an F16 or F32
//     destination -- a BF16 one hits its default case and aborts. Task 5 is
//     what first ran this graph against real, quantized weights rather than
//     Task 4's own synthetic F32 fixture, which is why this went unnoticed
//     until then. The fix generalizes the kernel-of-one path's own contract
//     (ggml_mul_mat's second operand must be F32; only the first may be a
//     lower-precision weight) to the im2col path: same_conv1d's mel-derived
//     activation is always F32 already, so requesting an F32 im2col output
//     and multiplying it against a same-dtype-as-package `kernel_2d` is the
//     same mixed-precision matmul the kernel-of-one branch already relies on,
//     not a new one.
//
// Layout is channel-major throughout, [channels, length], as the rest of this
// family is. Only the reflect padding and the pooling statistics work
// time-major, because ggml pads and reduces along ne[0].
//
// Nothing here names a width. Every extent is read off the weights, which
// catalog.cpp already checked against the package's own hyper-parameters, so
// there is no second place for a shape to be wrong.

#include "speaker-encoder.h"

#include "ggml.h"

#include <cmath>
#include <vector>

namespace synth::qwen3tts {

namespace {

// Upstream's own floor, applied to the VARIANCE before the square root
// (modeling_qwen3_tts.py:211, `.clamp(self.eps)` with eps set at :167).
//
// It is here for agreement with upstream, not to catch a NaN: the variance is
// summed from squares scaled by softmax weights, so it cannot come out negative
// on any backend and the square root is safe with or without it. What it does
// change is the one case where the variance is exactly zero -- a clip whose
// channel never varies -- where upstream reports 1e-6 and an unfloored port
// would report 0. That difference is a millionth, far under any tolerance a
// test could credibly assert, which is precisely why it has to be transcribed
// rather than reasoned about.
constexpr float kVarianceFloor = 1e-12f;

bool bound(const Conv1dWeights & weights) {
    return weights.weight != nullptr && weights.bias != nullptr;
}

// A bias over [channels, length] is per channel, so it broadcasts along the
// length rather than across it.
//
// `signal` is always F32 (every activation in this graph is); `bias` is
// whatever the package stores, which is BF16 for this checkpoint's
// speaker_encoder tensors. ggml's CPU binary_op has no F32+BF16 case -- only
// matching types or specific promotions -- so a raw ggml_add aborts exactly
// the way the im2col call above did before its own fix, and for the same
// underlying reason: Task 4's own unit test only ever exercised F32 weights.
// Cast only when needed (`bias->type != GGML_TYPE_F32`) rather than
// unconditionally: an unconditional ggml_cast inserts a CPY node even when
// `bias` is already F32, and this function runs once per convolution (38
// times), which would have pushed the graph's node count past
// tests/qwen3_tts_speaker_encoder_test.cpp's own 450-node ceiling for no
// reason on the synthetic F32 fixture that test still uses.
ggml_tensor * add_channel_bias(ggml_context * context, ggml_tensor * signal, ggml_tensor * bias) {
    ggml_tensor * bias_f32 = bias->type == GGML_TYPE_F32 ? bias : ggml_cast(context, bias, GGML_TYPE_F32);
    return ggml_add(context, signal, ggml_reshape_2d(context, bias_f32, bias_f32->ne[0], 1));
}

// One convolution at PyTorch's `padding="same", padding_mode="reflect"`.
//
// `input` is [in_channels, length]; the result is [out_channels, length]. Only
// odd kernels are accepted: PyTorch splits an even kernel's padding
// asymmetrically, and this family carries no even one, so an even kernel here
// means a weight was resolved into the wrong slot rather than a case to
// support.
ggml_tensor * same_conv1d(ggml_context *        context,
                          ggml_tensor *         input,
                          const Conv1dWeights & weights,
                          int64_t               dilation) {
    if (context == nullptr || input == nullptr || !bound(weights) || dilation <= 0) {
        return nullptr;
    }
    const int64_t kernel       = weights.weight->ne[0];
    const int64_t in_channels  = weights.weight->ne[1];
    const int64_t out_channels = weights.weight->ne[2];
    const int64_t length       = input->ne[1];
    if (kernel <= 0 || (kernel % 2) == 0 || length <= 0 || in_channels != input->ne[0] || out_channels <= 0) {
        return nullptr;
    }
    if (weights.bias->ne[0] != out_channels) {
        return nullptr;
    }

    if (kernel == 1) {
        // No neighbourhood to gather: the convolution is a matrix multiply over
        // channels, and `same` padding of a 1-wide kernel is no padding.
        ggml_tensor * dense       = ggml_reshape_2d(context, weights.weight, in_channels, out_channels);
        ggml_tensor * dense_input = ggml_is_contiguous(input) ? input : ggml_cont(context, input);
        return add_channel_bias(context, ggml_mul_mat(context, dense, dense_input), weights.bias);
    }

    // A reflection cannot reach past the signal it reflects; ggml asserts on it
    // rather than returning, and so does upstream's own F.pad.
    const int64_t padding = (kernel - 1) * dilation / 2;
    if (padding >= length) {
        return nullptr;
    }

    ggml_tensor * time_major = ggml_cont(context, ggml_transpose(context, input));
    ggml_tensor * padded     = ggml_pad_reflect_1d(context, time_major, int(padding), int(padding));
    // Padded already, so im2col pads by nothing of its own. The destination
    // type is always F32, not weights.weight->type: `padded` is already F32
    // (this graph's mel input and every activation derived from it stay F32
    // throughout), ggml_compute_forward_im2col's CPU path has no BF16 case to
    // fall into, and the mul_mat below needs an F32 second operand regardless
    // -- see this file's header comment for why a package whose kernels are
    // not F32 (this checkpoint's speaker_encoder tensors are BF16) makes this
    // the only correct choice rather than an optimization to skip.
    ggml_tensor * columns =
        ggml_im2col(context, weights.weight, padded, 1, 0, 0, 0, int(dilation), 0, false, GGML_TYPE_F32);
    ggml_tensor * kernel_2d = ggml_reshape_2d(context, weights.weight, kernel * in_channels, out_channels);
    ggml_tensor * wide =
        ggml_mul_mat(context, kernel_2d, ggml_reshape_2d(context, columns, columns->ne[0], columns->ne[1]));
    return add_channel_bias(context, wide, weights.bias);
}

// TimeDelayNetBlock: the convolution above, then ReLU.
ggml_tensor * tdnn(ggml_context * context, ggml_tensor * input, const Conv1dWeights & weights, int64_t dilation) {
    ggml_tensor * convolved = same_conv1d(context, input, weights, dilation);
    return convolved == nullptr ? nullptr : ggml_relu(context, convolved);
}

// The mean over time of a [channels, length] signal, as [channels, 1].
//
// ggml reduces along ne[0], which is the channel axis here, so the reduction
// runs on a transposed copy. The result comes back by reshape rather than by a
// second transpose: a contiguous [1, channels] and a contiguous [channels, 1]
// hold the same elements in the same order.
ggml_tensor * mean_over_time(ggml_context * context, ggml_tensor * signal) {
    ggml_tensor * time_major = ggml_cont(context, ggml_transpose(context, signal));
    ggml_tensor * mean       = ggml_mean(context, time_major);
    return ggml_reshape_2d(context, mean, mean->ne[1], 1);
}

// Res2Net at `weights.size() + 1` splits: split 0 passes through and leads the
// concatenation, and each later split is convolved after the previous split's
// output has been added to it. The accumulation is what makes the receptive
// field grow across splits; dropping it leaves a plausible encoder that is not
// this one.
ggml_tensor * res2net(ggml_context *                     context,
                      ggml_tensor *                      input,
                      const std::vector<Conv1dWeights> & weights,
                      int64_t                            dilation) {
    const int64_t scale = int64_t(weights.size()) + 1;
    if (scale < 2) {
        return nullptr;
    }
    const int64_t channels = input->ne[0];
    const int64_t length   = input->ne[1];
    if (channels % scale != 0) {
        return nullptr;
    }
    const int64_t width = channels / scale;

    ggml_tensor * joined   = nullptr;
    ggml_tensor * previous = nullptr;
    for (int64_t split = 0; split < scale; ++split) {
        // A split is a strided window over the channel axis, made contiguous
        // here rather than left for whichever operator reads it next.
        ggml_tensor * part =
            ggml_cont(context, ggml_view_2d(context, input, width, length, input->nb[1], split * width * input->nb[0]));
        ggml_tensor * out = nullptr;
        if (split == 0) {
            out = part;
        } else {
            ggml_tensor * fed = split == 1 ? part : ggml_add(context, part, previous);
            out               = tdnn(context, fed, weights[size_t(split - 1)], dilation);
        }
        if (out == nullptr) {
            return nullptr;
        }
        joined   = joined == nullptr ? out : ggml_concat(context, joined, out, 0);
        previous = out;
    }
    return joined;
}

// Squeeze-and-excitation: one gain per channel, learned from that channel's
// mean over time, multiplied back into every frame.
ggml_tensor * squeeze_excite(ggml_context * context, ggml_tensor * input, const SpeakerEncoderBlockWeights & weights) {
    ggml_tensor * pooled = mean_over_time(context, input);
    ggml_tensor * narrow = tdnn(context, pooled, weights.se1, 1);
    if (narrow == nullptr) {
        return nullptr;
    }
    ggml_tensor * wide = same_conv1d(context, narrow, weights.se2, 1);
    if (wide == nullptr || wide->ne[0] != input->ne[0]) {
        return nullptr;
    }
    // One gain per channel against every frame: a broadcast along the length,
    // not a per-frame gain.
    return ggml_mul(context, input, ggml_sigmoid(context, wide));
}

// One SE-Res2Net block, residual included.
ggml_tensor * se_res2net_block(ggml_context *                     context,
                               ggml_tensor *                      input,
                               const SpeakerEncoderBlockWeights & weights,
                               int64_t                            dilation) {
    ggml_tensor * hidden = tdnn(context, input, weights.tdnn1, 1);
    if (hidden == nullptr) {
        return nullptr;
    }
    hidden = res2net(context, hidden, weights.res2net, dilation);
    if (hidden == nullptr) {
        return nullptr;
    }
    hidden = tdnn(context, hidden, weights.tdnn2, 1);
    if (hidden == nullptr) {
        return nullptr;
    }
    hidden = squeeze_excite(context, hidden, weights);
    // The residual is an add, which aborts rather than fails on a width
    // mismatch, so the widths are compared before it is built.
    if (hidden == nullptr || hidden->ne[0] != input->ne[0]) {
        return nullptr;
    }
    return ggml_add(context, hidden, input);
}

// The standard deviation over time of a [length, channels] time-major signal
// weighted by `weight` ([length, channels] or [1, channels]), as [1, channels].
ggml_tensor * weighted_deviation(ggml_context * context,
                                 ggml_tensor *  time_major,
                                 ggml_tensor *  mean,
                                 ggml_tensor *  weight) {
    ggml_tensor * deviation = ggml_sqr(context, ggml_sub(context, time_major, mean));
    ggml_tensor * variance  = weight == nullptr ? ggml_mean(context, deviation) :
                                                  ggml_sum_rows(context, ggml_mul(context, deviation, weight));
    return ggml_sqrt(context, ggml_clamp(context, variance, kVarianceFloor, INFINITY));
}

}  // namespace

ggml_tensor * build_speaker_encoder(ggml_context * context, const SpeakerEncoderWeights & weights, ggml_tensor * mel) {
    if (context == nullptr || mel == nullptr || weights.blocks.empty()) {
        return nullptr;
    }
    // A mel of the wrong type reaches ggml_pad_reflect_1d's F32 assertion, and a
    // batched one would be convolved as if the batch were more frames.
    if (mel->type != GGML_TYPE_F32 || mel->ne[2] != 1 || mel->ne[3] != 1) {
        return nullptr;
    }
    if (mel->ne[1] <= kSpeakerEncoderDeepestReflection) {
        return nullptr;
    }

    // The stem: mel bins up to the encoder's working width. Its declared input
    // extent is what a mel's bin count has to agree with.
    ggml_tensor * hidden = tdnn(context, mel, weights.stem, 1);
    if (hidden == nullptr) {
        return nullptr;
    }
    std::vector<ggml_tensor *> taps;

    // The three SE-Res2Net blocks at enc_dilations[1..3]. The stem's own output
    // is deliberately not a tap: upstream aggregates blocks 1..3 only.
    for (size_t index = 0; index < weights.blocks.size(); ++index) {
        // enc_dilations is [1, 2, 3, 4, 1]: the stem and mfa run undilated and
        // block i dilates by i + 2.
        hidden = se_res2net_block(context, hidden, weights.blocks[index], int64_t(index) + 2);
        if (hidden == nullptr) {
            return nullptr;
        }
        taps.push_back(hidden);
    }

    // Multi-layer feature aggregation over the three blocks' outputs.
    ggml_tensor * aggregated = taps.front();
    for (size_t index = 1; index < taps.size(); ++index) {
        aggregated = ggml_concat(context, aggregated, taps[index], 0);
    }
    ggml_tensor * features = tdnn(context, aggregated, weights.mfa, 1);
    if (features == nullptr) {
        return nullptr;
    }

    // Attentive statistics pooling (modeling_qwen3_tts.py:214-245). The mask
    // upstream builds is all ones for a single un-padded clip, so `mask / total`
    // is a uniform 1/length and the unweighted statistics below are plain means;
    // the masked_fill before the softmax is likewise a no-op. Reproducing the
    // mask would add a tensor whose every element is one.
    const int64_t channels   = features->ne[0];
    const int64_t frames     = features->ne[1];
    ggml_tensor * time_major = ggml_cont(context, ggml_transpose(context, features));
    ggml_tensor * mean       = ggml_mean(context, time_major);
    ggml_tensor * deviation  = weighted_deviation(context, time_major, mean, nullptr);

    // The attention sees the frame beside the whole clip's mean and deviation,
    // which is what lets a per-frame weight depend on global properties -- and
    // is why asp.tdnn's input extent is three times the aggregated width.
    ggml_tensor * mean_wide      = ggml_repeat(context, ggml_reshape_2d(context, mean, channels, 1), features);
    ggml_tensor * deviation_wide = ggml_repeat(context, ggml_reshape_2d(context, deviation, channels, 1), features);
    ggml_tensor * context_wide = ggml_concat(context, ggml_concat(context, features, mean_wide, 0), deviation_wide, 0);

    // Two activations, not one: TimeDelayNetBlock's own ReLU and then the
    // pooling layer's tanh (:234, `self.conv(self.tanh(self.tdnn(attention)))`).
    ggml_tensor * narrow = tdnn(context, context_wide, weights.asp_tdnn, 1);
    if (narrow == nullptr) {
        return nullptr;
    }
    ggml_tensor * logits = same_conv1d(context, ggml_tanh(context, narrow), weights.asp, 1);
    if (logits == nullptr || logits->ne[0] != channels || logits->ne[1] != frames) {
        return nullptr;
    }

    // The softmax is over time, per channel: every channel gets its own weighting
    // of the clip. ggml normalizes along ne[0], which is time in this layout.
    ggml_tensor * attention      = ggml_soft_max(context, ggml_cont(context, ggml_transpose(context, logits)));
    ggml_tensor * weighted_mean  = ggml_sum_rows(context, ggml_mul(context, time_major, attention));
    ggml_tensor * weighted_scale = weighted_deviation(context, time_major, weighted_mean, attention);

    // fc reads the weighted mean and deviation concatenated, which is why its
    // input extent is twice the aggregated width.
    ggml_tensor * statistics = ggml_concat(context, ggml_reshape_2d(context, weighted_mean, channels, 1),
                                           ggml_reshape_2d(context, weighted_scale, channels, 1), 0);
    ggml_tensor * embedding  = same_conv1d(context, statistics, weights.fc, 1);
    if (embedding == nullptr) {
        return nullptr;
    }
    return ggml_reshape_1d(context, embedding, embedding->ne[0]);
}

}  // namespace synth::qwen3tts
