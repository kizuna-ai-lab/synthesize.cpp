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

// One SE-Res2Net block: a TDNN 1x1 in, a scale-8 res2net body whose FIRST
// split passes through unconvolved and leads the concatenation (hence seven
// convolutions, not eight), a squeeze-excite bottleneck pair, and a TDNN 1x1
// out. The block's own input is then added back as a residual.
//
// Which split passes through does not change the tensor count, so it does not
// change what this struct resolves -- but it does change the forward pass, and
// speaker-encoder.cpp is the file that reads these pointers. See
// modeling_qwen3_tts.py:115-126, where split 0 is `output_part = hidden_part`
// and the outputs are concatenated in split order.
struct SpeakerEncoderBlockWeights {
    Conv1dWeights              tdnn1;
    std::vector<Conv1dWeights> res2net;
    Conv1dWeights              se1;
    Conv1dWeights              se2;
    Conv1dWeights              tdnn2;
};

// The ECAPA-TDNN speaker encoder Base variants carry. Plan 1 resolved these
// 76 names into a discarded scratch struct because no graph could reach them;
// Plan 2's speaker-encoder.cpp is that graph, so the pointers are kept.
struct SpeakerEncoderWeights {
    Conv1dWeights                           stem;      // blocks.0.conv, kernel 5
    std::vector<SpeakerEncoderBlockWeights> blocks;    // three
    Conv1dWeights                           mfa;       // multi-layer feature aggregation
    Conv1dWeights                           asp_tdnn;  // attention bottleneck
    Conv1dWeights                           asp;       // attention logits
    Conv1dWeights                           fc;        // pooled statistics -> enc_dim
};

// One downsampling stage of the codec encoder's convolutional stem: a
// narrow-then-wide residual bottleneck (`layers.<3n+1>.block.1` and `.3`)
// followed by a strided convolution (`layers.<3n+3>`) that widens the channel
// count while shortening the sequence. The activation-only position between
// them carries no tensor.
struct CodecEncoderStage {
    Conv1dWeights bottleneck_in;   // kernel 3, width -> width/2
    Conv1dWeights bottleneck_out;  // kernel 1, width/2 -> width
    Conv1dWeights stride_conv;     // kernel {8,10,12,16}, width -> {128,256,512,1024}
};

// The codec encoder's transformer layer. Standard LayerNorm (weight *and*
// bias) and a plain two-layer MLP, unlike CodecTransformerLayerWeights above,
// which is RMSNorm and gated: the two are built from the same
// per-branch-scale idea but are not the same block, so they do not share a
// struct.
struct CodecEncoderTransformerLayerWeights {
    LayerNormWeights input_layernorm;
    ggml_tensor *    q_proj                = nullptr;
    ggml_tensor *    k_proj                = nullptr;
    ggml_tensor *    v_proj                = nullptr;
    ggml_tensor *    o_proj                = nullptr;
    ggml_tensor *    self_attn_layer_scale = nullptr;
    LayerNormWeights post_attention_layernorm;
    ggml_tensor *    fc1             = nullptr;
    ggml_tensor *    fc2             = nullptr;
    ggml_tensor *    mlp_layer_scale = nullptr;
};

// The speech tokenizer's encoder half, which Base variants carry: waveform in,
// codes out. Plan 1 resolved these 161 names into a discarded scratch struct
// because no graph could reach them; Plan 3's ICL path is that graph, so the
// pointers are kept.
//
// The quantizer cascades reuse CodecQuantizerWeights: the encoder's tables have
// the decoder's shape and differ only in where the name puts them (`layers.<n>`
// rather than `vq.layers.<n>`), which is the resolver's problem and not the
// struct's. The acoustic cascade runs deeper here than the decoder ever reads
// back -- 31 stages against `quantizer_count - semantic_quantizer_count` -- so
// its length is not derived from the decoder's.
struct CodecEncoderWeights {
    Conv1dWeights                                    stem;    // encoder.layers.0.conv, kernel 7, 1 -> 64
    std::vector<CodecEncoderStage>                   stages;  // four
    Conv1dWeights                                    tail;    // encoder.layers.14.conv, kernel 3, -> codebook_dim
    // codec.encoder.downsample.conv.weight: the frame downsampler, kernel 4,
    // hidden -> hidden. It carries no bias, so there is no Conv1dWeights here.
    ggml_tensor *                                    downsample = nullptr;
    std::vector<CodecEncoderTransformerLayerWeights> layers;  // eight
    CodecQuantizerWeights                            semantic;
    CodecQuantizerWeights                            acoustic;
};

struct ModelWeights {
    TalkerWeights         talker;
    CodePredictorWeights  code_predictor;
    CodecDecoderWeights   codec;
    // Both empty (default-constructed) for a CustomVoice package -- see
    // build_model_weights's doc comment below.
    SpeakerEncoderWeights speaker_encoder;
    CodecEncoderWeights   codec_encoder;
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
// encoder and the speech tokenizer's encoder half. Each is resolved into its
// own member -- `ModelWeights::speaker_encoder` for speaker-encoder.cpp's
// graph (Plan 2), `ModelWeights::codec_encoder` for the ICL path's (Plan 3) --
// and a Base package that carries either uncatalogued is refused rather than
// silently accepted.
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
