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

}  // namespace synth::omnivoice
