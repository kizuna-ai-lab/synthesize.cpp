#pragma once

#include "catalog.h"
#include "synthesize.h"

#include <cstddef>
#include <cstdint>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace synth::qwen3tts {

// The speech tokenizer's encoder half: a waveform in, the pre-quantization
// latents out. Base variants carry it; it is what the ICL path clones a voice
// from, and it runs in the opposite direction from codec.h's decoder.
//
// ITS REFERENCE IMPLEMENTATION IS NOT A QWEN FILE. Qwen contributes
// `Qwen3TTSTokenizerV2Encoder(MimiModel)`, whose __init__ nulls `upsample`,
// `decoder_transformer` and `decoder` and adds two post-steps; everything from
// the waveform to the codes is unmodified `transformers.MimiModel`
// (transformers==4.57.3, `transformers/models/mimi/modeling_mimi.py`). The
// checkpoint contributes the widths. Three provenances, and a convention read
// off the wrong one still produces finite latents that decode to plausible
// audio -- which is why every rule below cites the file it came from, and why
// the values themselves are taken from the oracle's own machine-checked
// `conventions.json` rather than re-derived here.
//
// Layout is channel-major, [channels, length], as the rest of this family is.
// That is also the transformer's own layout (ne[0] is the hidden width), so
// unlike upstream -- which transposes to [B, T, C] at modeling_mimi.py:1460 and
// back at :1466 -- nothing has to be transposed across that boundary. The
// oracle's `transformer_l*.f32` artifacts are dumped in upstream's transposed
// order and a comparison has to transpose them, not this graph.

// What a clip of `samples` samples turns into. Derived by walking the real
// convolutions rather than dividing by a constant: the strides ARE the model,
// and 1920 is their product, not an independent fact about it.
struct CodecEncoderGeometry {
    int64_t samples               = 0;
    // The product of every stride on the encode path: 4 * 5 * 6 * 8 (the SEANet
    // stack) * 2 (the frame downsampler) = 1920 at the production geometry.
    int64_t samples_per_frame     = 0;
    // The SEANet stack's output length, which is what the transformer runs at.
    // TWICE the frame count, near enough: the frame downsampler comes AFTER the
    // transformer, not before it (modeling_mimi.py:1456-1467).
    int64_t transformer_positions = 0;
    // After the frame downsampler. `ceil(samples / samples_per_frame)`.
    int64_t frames                = 0;
};

// Fills `geometry` for a clip of `samples` samples. False when the weights are
// not bound, or when the clip does not fill one whole frame -- a sub-frame clip
// is refused rather than padded up to one, because a caller that hands over
// forty milliseconds of audio has made a mistake that a single fabricated frame
// would hide.
//
// Every convolution's output length is upstream's own
// `MimiConv1d._get_output_length` (modeling_mimi.py:295-310) over its
// `_get_extra_padding_for_conv1d` (:263-273), transcribed in
// codec-encoder.cpp. It collapses to `ceil(length / stride)` per convolution,
// and the composition of those ceilings to `ceil(samples / 1920)` -- but the
// collapse is a cross-check on the transcription, not a substitute for it.
bool codec_encoder_geometry(const CodecEncoderWeights & weights, int64_t samples, CodecEncoderGeometry & geometry);

// The two rules a graph cannot enforce, because a graph builder sees shapes and
// never values: the clip must fill at least one frame, and every sample must be
// finite. A NaN sample propagates into every frame the transformer touches, so
// it must be refused at the seam rather than found in the codes.
synth_status_t codec_encoder_check_waveform(const CodecEncoderWeights & weights,
                                            const float *               samples,
                                            size_t                      count,
                                            CodecEncoderGeometry &      geometry);

// The encoder transformer's geometry.
//
// Deliberately NOT operations.h's AttentionShape: that struct names its epsilon
// `rms_norm_eps`, and this block does not RMS-norm. It uses a standard
// LayerNorm with a weight AND a bias (modeling_mimi.py:930-931,
// `nn.LayerNorm(config.hidden_size, eps=config.norm_eps)`). Reusing a struct
// whose field name asserts the other norm is exactly the sort of quiet
// mis-transcription this port keeps paying for.
struct CodecEncoderAttentionShape {
    int64_t hidden         = 0;
    int64_t head_count     = 0;
    int64_t head_dim       = 0;
    float   layer_norm_eps = 0.0f;
    float   rope_theta     = 0.0f;
};

// Where the stage-wise comparison against the oracle reads.
//
// There is one entry here for every artifact Task 1's dumper writes
// (`seanet_stage0..3`, `seanet_tail`, `transformer_l0..l7`, `downsample`), and
// filling them costs a pointer copy: the tensors are the graph's own, not
// copies of it. That matters more than it looks. The alternative -- a validator
// that rebuilds the stages out of the exposed operators to get at the
// intermediates -- measures the validator, and would have agreed with the
// oracle stage by stage while the encoder Task 5 actually runs disagreed.
//
// Every tap is in the graph's channel-major [channels, length] layout,
// INCLUDING the transformer ones. The oracle's `transformer_l*.f32` are
// channel-last, because upstream transposes around its transformer and this
// port does not; a comparison transposes them.
//
// A CALLER MUST ggml_set_output EACH TAP BEFORE ALLOCATING THE GRAPH.
// ggml_gallocr recycles an intermediate's buffer the moment nothing left to
// run reads it, so a tap read back after compute is whatever later node landed
// on top of it -- plausible floats, right shape, wrong tensor. Building the tap
// into the graph with ggml_build_forward_expand is NOT enough; only the output
// flag pins the memory. This cost the first run of the stage-wise comparison,
// which reported every stage disagreeing with the oracle by 15-70x when the
// graph was already correct to 2e-4.
struct CodecEncoderTaps {
    std::vector<ggml_tensor *> seanet_stages;       // one per downsampling stage, after its strided convolution
    ggml_tensor *              seanet_tail = nullptr;
    std::vector<ggml_tensor *> transformer_layers;  // one per layer, after the layer's MLP residual
    ggml_tensor *              downsample = nullptr;
};

// One causal convolution, at `stride` and `dilation`.
//
// `input` is [in_channels, length]; `weight` is [kernel, in_channels,
// out_channels] as ggml reports a Conv1d; `bias` is [out_channels] or nullptr
// (the frame downsampler carries none). The result is [out_channels,
// ceil(length / stride)].
//
// Causal padding here is NOT the decoder's "pad wide and keep the prefix". It
// is upstream's own asymmetric split: ALL of `padding_total` on the LEFT and
// only the ragged `extra_padding` on the right (modeling_mimi.py:331-333).
// :249-250 does compute a symmetric split, but that is the non-causal branch at
// :336-338 and `use_causal_conv` is true for this checkpoint, so it is never
// reached. A port that pads symmetrically builds the same shapes from the same
// weights and a different encoder.
//
// `replicate_pad` selects `pad_mode="replicate"` over the checkpoint's
// `"constant"`. Exactly one convolution on this path needs it -- the frame
// downsampler, which modeling_mimi.py:1406-1415 constructs with an explicit
// `pad_mode="replicate"` keyword that :222 lets override the config. Padding it
// with zeros like its neighbours corrupts precisely the first frame of every
// clip, which no summary statistic over 101 frames will show.
ggml_tensor * codec_encoder_causal_conv1d(ggml_context * context,
                                          ggml_tensor *  input,
                                          ggml_tensor *  weight,
                                          ggml_tensor *  bias,
                                          int64_t        stride,
                                          int64_t        dilation,
                                          bool           replicate_pad);

// The SEANet stack: an F32 [1, samples] waveform in, [codebook_dim,
// transformer_positions] out.
//
// [1, samples] and not [samples]: this family's layout is channel-major, so a
// mono clip is one channel of `samples`, exactly the shape build_codec_decoder
// hands back at the other end of the codec. A bare 1-D [samples] tensor is
// ne[0] = samples, which this would read as a `samples`-channel clip one sample
// long, so it is refused rather than transposed on the caller's behalf.
//
// Stem convolution, then four stages of `residual block -> ELU -> strided
// convolution`, then a final ELU and the tail convolution
// (modeling_mimi.py:443-484). ELU precedes every convolution and follows none
// (:418-419, :463, :468); there is no activation before the stem and none after
// the tail.
ggml_tensor * build_codec_encoder_seanet(ggml_context *              context,
                                         ggml_tensor *               waveform,
                                         const CodecEncoderWeights & weights,
                                         CodecEncoderTaps *          taps = nullptr);

// One encoder transformer layer over [hidden, positions].
//
// `position_ids` is an I32 [positions] that rope consumes. The causal mask is
// built inside, with ggml_diag_mask_inf, rather than taken from the caller: it
// is fully determined by the sequence length and there is nothing for a caller
// to decide.
//
// THERE IS NO SLIDING WINDOW, and that is a measured claim rather than an
// omission. `encoder_config.sliding_window` is 250 and MimiAttention stores it
// (modeling_mimi.py:644), but the only forward pass that reads it is
// MimiFlashAttention2's (:810). MimiTransformerModel builds its mask with
// `create_causal_mask` (:1099-1106), which is transformers' PLAIN causal mask
// -- `create_sliding_window_causal_mask` is a different function it does not
// call -- so under `eager` and `sdpa` alike the window never reaches the mask.
// The oracle ran `sdpa` (conventions.json's attn_implementation_observed).
// Verified by running MimiTransformerModel at sliding_window=2 over 8
// positions: position 7 still depends on position 0, and create_causal_mask
// returns a full lower-triangular mask. See this task's report.
ggml_tensor * codec_encoder_transformer_layer(ggml_context *                              context,
                                              ggml_tensor *                               input,
                                              ggml_tensor *                               position_ids,
                                              const CodecEncoderTransformerLayerWeights & weights,
                                              const CodecEncoderAttentionShape &          shape);

// The eight-layer encoder transformer. There is no final norm after it: the
// package carries none, and modeling_mimi.py's MimiTransformerModel has none.
ggml_tensor * build_codec_encoder_transformer(ggml_context *              context,
                                              ggml_tensor *               input,
                                              ggml_tensor *               position_ids,
                                              const CodecEncoderWeights & weights,
                                              CodecEncoderTaps *          taps = nullptr);

// The frame downsampler: kernel 4, stride 2, no bias, replicate padding.
// [hidden, positions] in, [hidden, ceil(positions / 2)] out.
ggml_tensor * build_codec_encoder_downsample(ggml_context *              context,
                                             ggml_tensor *               input,
                                             const CodecEncoderWeights & weights);

// The whole encoder: an F32 [1, samples] waveform in, [codebook_dim, frames] of
// pre-quantization latents out. Task 5's host wrapper runs this and quantizes.
//
// `position_ids` is an I32 [transformer_positions] -- the SEANet stack's output
// length, NOT the frame count, because the transformer runs before the frame
// downsampler. Take it from codec_encoder_geometry rather than computing it.
//
// Returns nullptr rather than aborting on anything it cannot build.
ggml_tensor * build_codec_encoder(ggml_context *              context,
                                  ggml_tensor *               waveform,
                                  ggml_tensor *               position_ids,
                                  const CodecEncoderWeights & weights,
                                  CodecEncoderTaps *          taps = nullptr);

}  // namespace synth::qwen3tts
