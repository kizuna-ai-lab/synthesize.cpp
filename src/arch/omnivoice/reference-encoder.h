#pragma once

// catalog.h carries the weight structs and weights.h the HParams the branch's
// geometry reads -- the post-Task-4 roles.
#include "catalog.h"
#include "weights.h"

#include <cstdint>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace synth::omnivoice {

// The HuBERT semantic branch (transformers HubertModel, do_stable_layer_norm
// = False -- this checkpoint's variant) plus the codec's own SemanticEncoder
// conv stack (codec.encoder_semantic.*): the path
// HiggsAudioV2TokenizerModel.encode calls to build the cloning path's
// semantic latent. Every citation below is against
// modeling_higgs_audio_v2_tokenizer.py and modeling_hubert.py as pinned in
// scripts/envs/omnivoice's locked venv (transformers 5.14.1).
//
// `pcm_16k` is 1-D [samples], the ALREADY-RESAMPLED-TO-16kHz mono waveform
// (reference-encoder-host.h's resample_24k_to_16k), PRE-pad: this builder
// applies a (160, 160) zero pad internally, matching
// HiggsAudioV2TokenizerModel._extract_semantic_features's own
// `F.pad(input_values, (160, 160))` (modeling_higgs_audio_v2_tokenizer.py:499)
// -- a LITERAL constant in the reference, unconditional on any config field
// (the line above it, commented out, shows the formula it replaced:
// `self.pad = hop_length // 2`, i.e. 480 for this checkpoint's 960-sample
// hop -- NOT what actually runs), so it is hardcoded here too rather than
// threaded through HParams, identically at real and miniature scale.
//
// Pipeline (HubertModel.forward + HubertEncoder.forward, do_stable_layer_norm
// = False, then HiggsAudioV2TokenizerModel.encode's
// `encoder_semantic(e_semantic_input.transpose(1, 2))`):
//   pad -> feature extractor: one bias-free Conv1d per `semantic.conv_dim`
//     entry (kernel/stride from `conv_kernel`/`conv_stride`, valid padding),
//     GroupNorm(groups = channels, i.e. per-channel over time -- NOT the
//     stock affine-per-channel-over-features GroupNorm) + GELU(erf) on the
//     FIRST convolution only (`feat_extract_norm = "group"`), GELU(erf) alone
//     on the rest
//   -> feature_projection: LayerNorm(conv_dim.back()) then a biased Linear to
//     `semantic.hidden_size` (dropout is a no-op in eval)
//   -> + a grouped positional convolution (kernel/groups fixed by the
//     checkpoint, weight-norm already folded into a plain kernel by the
//     converter -- see catalog.cpp's kPosConvKernel/kPosConvGroups comment),
//     HubertSamePadLayer's trailing-sample trim when that kernel is even,
//     GELU(erf)
//   -> encoder.layer_norm (BEFORE the layer stack in this do_stable_layer_norm
//     = False variant -- the stable-layer-norm variant this port does NOT
//     implement puts it after)
//   -> `semantic.layer_count` post-LN attention blocks (see
//     reference-encoder.cpp's semantic_layer for the full delta list against
//     generator.cpp's generator_layer, the structural donor)
//   -> mean over ALL layer_count + 1 hidden states: the state entering the
//     layer stack (post pos-conv, post encoder.layer_norm) plus every layer's
//     own output, matching HubertEncoder.forward's `all_hidden_states` order
//     exactly (append the pre-layer state each iteration, then the
//     post-loop final state -- which telescopes to "the initial state, then
//     each layer's output")
//   -> [::2] stride-2 downsample (the codec's own `downsample_factor`,
//     likewise not threaded through HParams: for this checkpoint's config it
//     evaluates to exactly 2 -- HiggsAudioV2TokenizerConfig.semantic_downsample_factor
//     = hop_length / (sample_rate / semantic_sample_rate) / downsample_factor
//     = 960 / 1.5 / 320 = 2 -- and this port has no path that ever needs a
//     different checkpoint's value)
//   -> SemanticEncoder (codec.encoder_semantic.*): a bias-free entry Conv1d
//     (kernel 3, pad 1), then per block (two of them, both stride 1 for this
//     checkpoint) two bias-free dilation-1 residual units
//     (ELU -> kernel-3 conv -> ELU -> kernel-1 conv, added back) around a
//     biased kernel-3 exit Conv1d.
//
// LayerDrop: HubertEncoder.forward draws `torch.rand([])` every layer
// UNCONDITIONALLY, even in eval, but only acts on it when `self.training`
// (`skip_the_layer = self.training and dropout_probability < layerdrop`) --
// eval mode never skips regardless of the draw. This port draws nothing and
// skips nothing: a divergence-of-no-consequence, since eval-mode upstream's
// own behavior is "never skip" too.
//
// `out_hidden_states`, when non-null, receives all `layer_count + 1` hidden-
// state tensors in that same order (the pre-layer-stack state first) -- a
// debugging tap, the way generator.cpp's `out_layers` exposes each block's
// output for the same purpose.
// `out_mean`, when non-null, receives the mean BEFORE the downsample: this is
// the oracle's own `ref/semantic_mean.f32` probe (see
// scripts/dump_reference_omnivoice_pytorch.py's SemanticMeanProbe and its
// metadata note -- captured before the [::2] downsample, not after).
// `out_downsampled`, when non-null, receives the mean AFTER the downsample,
// i.e. SemanticEncoder's own input.
//
// Returns the SemanticEncoder's output, [semantic.hidden_size,
// downsampled_length], or nullptr on any shape this builder cannot serve.
// Every out-parameter is meaningful ONLY when this returns non-null.
ggml_tensor * build_semantic_branch(ggml_context *               context,
                                    ggml_tensor *                pcm_16k,
                                    const ModelWeights &         weights,
                                    const HParams &              hparams,
                                    std::vector<ggml_tensor *> * out_hidden_states = nullptr,
                                    ggml_tensor **               out_mean          = nullptr,
                                    ggml_tensor **               out_downsampled   = nullptr);

// The DAC-style acoustic encoder (transformers DacEncoder, reached through
// HiggsAudioV2TokenizerModel's own `self.acoustic_encoder = acoustic_model.encoder`)
// over a 24 kHz mono input tensor [samples]: an entry Conv1d (kernel 7, pad
// 3, stride 1, 1 channel -> hparams.codec.encoder_hidden_size; every
// convolution in this branch IS biased, unlike the semantic branch's
// bias-free HuBERT feature extractor -- catalog.cpp's resolve_acoustic_encoder
// resolves both weight and bias for every one of the 110
// codec.acoustic_encoder.* tensors), then
// one AcousticEncoderBlock per hparams.codec.upsampling_ratios entry (three
// residual units at dilations 1/3/9 FIRST, matching DacEncoderBlock.forward's
// own order -- res_unit1, res_unit2, snake1(res_unit3(...)), conv1 -- then a
// block-level Snake and a STRIDED Conv1d that DOUBLES the channel width
// LAST, the decoder's AcousticDecoderBlock in reverse: residual units before
// the resampling step there, after it here), then a final Snake and a
// kernel-3 exit Conv1d down to hparams.codec.hidden_size (256 for this
// checkpoint -- the DAC encoder/decoder's own shared bottleneck width, NOT
// the 1024-wide fused latent build_reference_fusion below produces). Every
// convolution and Snake here is codec.h's codec_conv1d/codec_snake -- the DAC
// DECODER's own building blocks (build_codec_decoder, codec.cpp), reused
// rather than duplicated, since this branch's convolutions are biased
// exactly like the decoder's; codec_conv1d's `stride` parameter (defaulting
// 1, every decoder call site unaffected) is this task's addition, needed only
// by this branch's block-exit resampling convolutions.
//
// RATIO ORDER: not the decoder's ratios reversed, despite this branch
// mirroring the decoder in every other respect (residual units and
// resampling swap places; width doubles instead of halves). transformers'
// DacEncoder.__init__ walks `config.downsampling_ratios` in the SAME plain
// enumeration order DacDecoder.__init__ walks `config.upsampling_ratios`, and
// this checkpoint's own acoustic_model_config states both lists as the
// LITERAL SAME ARRAY, [8, 5, 4, 2, 3]
// (models/omnivoice-0-6b/audio_tokenizer/config.json) -- confirmed not just
// by reading the config file (DacConfig.__post_init__ would otherwise
// unconditionally DERIVE upsampling_ratios as downsampling_ratios reversed,
// which turns out to only be the *default* when a caller omits the field;
// the checkpoint's own raw config dict overrides it explicitly) but by
// directly instantiating this checkpoint's real AutoModel.from_config(
// acoustic_model_config) and reading its own encoder.block[i].conv1 stride
// off the live module: block 0 is stride 8, block 1 stride 5, ..., block 4
// stride 3 -- exactly forward order, not reversed. catalog.cpp's own
// resolve_acoustic_encoder (catalog.cpp:364-387) already resolves this
// checkpoint's tensors against hparams.codec.upsampling_ratios directly, in
// the identical order the decoder reads it, so this builder needs no
// separate HParams field and no reversal.
//
// UNPADDED, always. HiggsAudioV2TokenizerModel.encode conditionally pads the
// RAW input by hop_length/2 (480) per side before this branch ever runs
// (modeling_higgs_audio_v2_tokenizer.py:549-552), but only when
// `_get_conv1d_output_lengths` -- a pure arithmetic replay of every Conv1d in
// this branch's stack, transformers/audio_utils.py:378's formula applied in
// forward order -- disagrees with the semantic branch's own frame count.
// RESOLVED for every hop-aligned input this port ever builds (Task 8's hop
// clip is what makes every input hop-aligned by construction):
//   - closed form: every residual-unit convolution here is length-preserving
//     (kernel 7 dilation d pad 3d, or kernel 1 pad 0), and so are this
//     branch's own entry/exit convolutions (kernel 7 pad 3; kernel 3 pad 1)
//     -- the only length-changing steps are the five STRIDED convolutions,
//     kernel = 2*ratio, stride = ratio, padding = ceil(ratio/2). Threading
//     conv1d_output_length's formula through ratios [8, 5, 4, 2, 3] in order
//     reduces algebraically, for any T that is an exact multiple of 960
//     (hop_length), to EXACTLY T / 960 frames -- no remainder, no rounding.
//   - measured: the oracle's own omni-clone-en fixture has
//     ref/pcm_24k.f32 = 336960 samples = 351 * 960 exactly; the closed form
//     above gives 351 frames unpadded; ref/fused_latent.f32 is
//     359424 = 1024 * 351 floats -- 351 frames, matching the semantic
//     branch's own measured downsampled length for the identical clip
//     (Task 11's report: 351). The two lengths already agree without the
//     pad, so the conditional pad branch does not fire for this (or any
//     hop-aligned) input, and this builder does not implement it.
// A caller that reaches this with a genuinely non-hop-aligned input gets a
// length disagreement build_reference_fusion refuses (nullptr) rather than
// one this builder silently pads around -- a wiring defect upstream of this
// file (a hop clip that let a misaligned length through), not a runtime case
// this file's own contract covers.
//
// Returns [hparams.codec.hidden_size, T] (T = pcm_24k's own sample count
// divided exactly by hop_length for a hop-aligned input), or nullptr on any
// shape this builder cannot serve.
ggml_tensor * build_acoustic_encoder(ggml_context *       context,
                                     ggml_tensor *        pcm_24k,
                                     const ModelWeights & weights,
                                     const HParams &      hparams);

// Fuses the acoustic and semantic branches:
// HiggsAudioV2TokenizerModel.encode's `embeddings = torch.cat([e_acoustic,
// e_semantic], dim=1)` (channel-wise concatenation: 256 + 768 = 1024 for this
// checkpoint) then `self.fc(embeddings.transpose(1, 2)).transpose(1, 2)` --
// codec.fc's Linear (1024 -> 1024, resolved by catalog.cpp's own
// `resolver.linear("codec.fc", concat, concat, weights.fc)`, `concat` being
// exactly hparams.codec.hidden_size + hparams.semantic.hidden_size) applied
// PER FRAME. The transpose dance in the reference exists only because torch
// keeps channels on dim=1 and nn.Linear reduces over the LAST axis; this
// file's [channels, length] convention already puts channels on ne0, the
// axis ggml_mul_mat reduces over, so applying the Linear needs no transpose
// here at all.
//
// `acoustic` and `semantic` must already share the same frame count (ne[1])
// -- see build_acoustic_encoder's own comment for why a caller that manages
// to violate this reached a wiring defect this refuses rather than a runtime
// case to pad around.
//
// Returns [hparams.codec.hidden_size + hparams.semantic.hidden_size, T] (the
// RVQ quantizer's own input width, Task 13's concern), or nullptr on any
// shape this builder cannot serve.
ggml_tensor * build_reference_fusion(ggml_context *       context,
                                     ggml_tensor *        acoustic,
                                     ggml_tensor *        semantic,
                                     const ModelWeights & weights);

}  // namespace synth::omnivoice
