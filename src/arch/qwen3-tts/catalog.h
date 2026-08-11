#pragma once

#include "code-predictor.h"
#include "operations.h"
#include "synthesize.h"

#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace synth::qwen3tts {

struct HParams;

struct LinearWeights {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias   = nullptr;
};

// GGML reports a Conv1d kernel stored as [out, in, kernel] in reverse, so its ne
// is [kernel, in, out]; a ConvTranspose1d stores [in, out, kernel] and so
// reports [kernel, out, in]. The two are easy to mistake for each other and the
// mistake only shows up as noise, so they are resolved by separate helpers.
struct Conv1dWeights {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias   = nullptr;
};

struct LayerNormWeights {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias   = nullptr;
};

// SnakeBeta's two curves are stored as logarithms and exponentiated in the
// forward pass, which is why neither may be assumed to be near one.
struct SnakeBetaWeights {
    ggml_tensor * alpha = nullptr;
    ggml_tensor * beta  = nullptr;
};

// The autoregressive language model that emits one semantic code per frame.
struct TalkerWeights {
    ggml_tensor *                    text_embedding = nullptr;
    // Two layers wide-then-narrow, which is what brings the text tower's width
    // down to the talker's.
    LinearWeights                    text_projection_1;
    LinearWeights                    text_projection_2;
    ggml_tensor *                    codec_embedding = nullptr;
    ggml_tensor *                    codec_head      = nullptr;
    std::vector<DecoderLayerWeights> layers;
    ggml_tensor *                    norm = nullptr;
};

// The codec's own transformer, which is not a Qwen3 block: it carries per-branch
// layer scales and no per-head norms.
struct CodecTransformerLayerWeights {
    ggml_tensor * input_layernorm          = nullptr;
    ggml_tensor * q_proj                   = nullptr;
    ggml_tensor * k_proj                   = nullptr;
    ggml_tensor * v_proj                   = nullptr;
    ggml_tensor * o_proj                   = nullptr;
    ggml_tensor * self_attn_layer_scale    = nullptr;
    ggml_tensor * post_attention_layernorm = nullptr;
    ggml_tensor * gate_proj                = nullptr;
    ggml_tensor * up_proj                  = nullptr;
    ggml_tensor * down_proj                = nullptr;
    ggml_tensor * mlp_layer_scale          = nullptr;
};

struct CodecTransformerWeights {
    LinearWeights                             input_proj;
    LinearWeights                             output_proj;
    std::vector<CodecTransformerLayerWeights> layers;
    ggml_tensor *                             norm = nullptr;
};

// One residual vector quantizer. The projections are kernel-1 convolutions
// rather than linears and carry no bias.
struct CodecQuantizerWeights {
    ggml_tensor *              input_proj  = nullptr;
    ggml_tensor *              output_proj = nullptr;
    std::vector<ggml_tensor *> codebooks;
};

struct ConvNeXtWeights {
    // Depthwise: one filter per channel, so the kernel's input extent is 1.
    Conv1dWeights    dwconv;
    LayerNormWeights norm;
    LinearWeights    pwconv1;
    LinearWeights    pwconv2;
    ggml_tensor *    gamma = nullptr;
};

struct CodecUpsampleStage {
    Conv1dWeights   transpose_conv;
    ConvNeXtWeights convnext;
};

struct CodecResidualUnit {
    SnakeBetaWeights act1;
    Conv1dWeights    conv1;
    SnakeBetaWeights act2;
    Conv1dWeights    conv2;
};

struct CodecResidualStage {
    SnakeBetaWeights               act;
    Conv1dWeights                  transpose_conv;
    std::vector<CodecResidualUnit> units;
};

struct CodecDecoderWeights {
    // The semantic quantizer covers code group 0 and the acoustic one the rest,
    // which is the same split the talker and the code predictor make.
    CodecQuantizerWeights           semantic;
    CodecQuantizerWeights           acoustic;
    Conv1dWeights                   pre_conv;
    CodecTransformerWeights         pre_transformer;
    std::vector<CodecUpsampleStage> upsample;
    Conv1dWeights                   input_conv;
    std::vector<CodecResidualStage> stages;
    SnakeBetaWeights                output_act;
    Conv1dWeights                   output_conv;
};

struct ModelWeights {
    TalkerWeights        talker;
    CodePredictorWeights code_predictor;
    CodecDecoderWeights  codec;
};

// Resolves the whole catalog against a loaded package.
//
// Every shape is derived from the package's own hyper-parameters rather than
// hardcoded, so a package whose metadata and tensors disagree is refused here
// instead of producing wrong audio later. Afterwards the package is swept: a
// tensor the catalog never asked for is an error, not something to ignore,
// because a name nobody resolves is a name nobody checked.
//
// When `hparams.has_speaker_encoder` is set (Base variants), the catalog also
// covers `speaker_encoder.*` and `codec.encoder.*`: the ECAPA-TDNN speaker
// encoder and the speech tokenizer's encoder half. Neither has a graph builder
// yet -- Plans 2 and 3 add those -- so they are resolved here purely to bring
// their names into the sweep; a Base package that carries them uncatalogued is
// refused rather than silently accepted.
// `codec_context`, when non-null, holds same-named twins of the codec half and
// the codec is bound against those instead. That is what lets the codec run on
// an accelerator while the talker and the code predictor stay on the CPU, which
// docs/backends.md requires of them: their output feeds a sampled code.
//
// The sweep still covers `context` alone, because that is the package. A twin is
// a placement detail and not a tensor the package carries.
synth_status_t build_model_weights(ggml_context *  context,
                                   ggml_context *  codec_context,
                                   const HParams & hparams,
                                   ModelWeights &  weights);

// The number of tensors a package for these hyper-parameters must contain.
// Exposed so a caller can size a context before resolving anything.
uint64_t expected_tensor_count(const HParams & hparams);

}  // namespace synth::qwen3tts
