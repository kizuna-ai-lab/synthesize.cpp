#pragma once

#include "synthesize.h"

#include <cstdint>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace synth::omnivoice {

struct HParams;

struct LinearWeights {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias   = nullptr;
};

// GGML reports a Conv1d kernel stored as [out, in, kernel] in reverse, so its ne
// is [kernel, in, out]; a ConvTranspose1d stores [in, out, kernel] and so
// reports [kernel, out, in]. The two are easy to mistake for each other and the
// mistake only shows up as noise, so they are resolved by separate helpers.
//
// Several of this family's convolutions carry no bias -- the HuBERT feature
// extractor's and the codec's semantic encoder's -- and leave `bias` null.
struct Conv1dWeights {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias   = nullptr;
};

struct LayerNormWeights {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias   = nullptr;
};

// Higgs DAC uses plain Snake, not SnakeBeta: one curve per channel and no second
// one, so a port that assumed the beta pair would look for a tensor that is not
// there.
struct SnakeWeights {
    ggml_tensor * alpha = nullptr;
};

// One Qwen3 block of the mask-predict generator. It is the ordinary block --
// per-head q/k/v norms, gated feed-forward -- run bidirectionally, which changes
// the graph but not the tensors.
struct GeneratorLayerWeights {
    ggml_tensor * input_layernorm = nullptr;
    ggml_tensor * q_proj = nullptr, *k_proj = nullptr, *v_proj = nullptr, *o_proj = nullptr;
    ggml_tensor * q_norm = nullptr, *k_norm = nullptr;  // per-head, width head_dim
    ggml_tensor * post_attention_layernorm = nullptr;
    ggml_tensor * gate_proj = nullptr, *up_proj = nullptr, *down_proj = nullptr;
};

struct GeneratorWeights {
    ggml_tensor *                      text_embedding = nullptr;  // llm.embed_tokens.weight; tied -- no lm_head exists
    std::vector<GeneratorLayerWeights> layers;
    ggml_tensor *                      norm             = nullptr;
    // The eight codebooks are stacked into one table each, so a canvas slot
    // indexes codebook * vocab_size + code.
    ggml_tensor *                      audio_embeddings = nullptr;
    ggml_tensor *                      audio_heads      = nullptr;
};

// One residual vector quantizer. Unlike most RVQ ports the projections here are
// Linears with biases, not kernel-1 convolutions.
struct RvqQuantizerWeights {
    LinearWeights input_proj;
    LinearWeights output_proj;
    ggml_tensor * codebook = nullptr;
};

// The DAC residual unit, dilated 1, 3 and 9 across the three of a block. The
// dilation changes no shape, so only the count reaches the catalog.
struct DacResidualUnit {
    SnakeWeights  snake1;
    Conv1dWeights conv1;
    SnakeWeights  snake2;
    Conv1dWeights conv2;
};

// Activation, transposed convolution that upsamples by the block's ratio, then
// the three residual units at the halved width.
struct AcousticDecoderBlock {
    SnakeWeights                 snake1;
    Conv1dWeights                conv_t1;
    std::vector<DacResidualUnit> res_units;
};

struct AcousticDecoderWeights {
    Conv1dWeights                     conv1;  // hidden_size -> decoder_hidden_size
    std::vector<AcousticDecoderBlock> blocks;
    SnakeWeights                      snake1;
    Conv1dWeights                     conv2;  // -> one channel, the mono waveform
};

// The encoder block is the decoder's mirror in order as well as in shape: the
// residual units come first, at the block's input width, and the strided
// convolution that doubles the width comes last.
struct AcousticEncoderBlock {
    std::vector<DacResidualUnit> res_units;
    SnakeWeights                 snake1;
    Conv1dWeights                conv1;
};

struct AcousticEncoderWeights {
    Conv1dWeights                     conv1;  // one channel in -> encoder_hidden_size
    std::vector<AcousticEncoderBlock> blocks;
    SnakeWeights                      snake1;
    Conv1dWeights                     conv2;  // -> hidden_size
};

// One HuBERT encoder layer. It is a post-norm BERT block, not a Qwen3 one: every
// projection is biased and square, and the two norms sit after their branches.
struct SemanticLayerWeights {
    LinearWeights    q_proj;
    LinearWeights    k_proj;
    LinearWeights    v_proj;
    LinearWeights    out_proj;
    LayerNormWeights layer_norm;
    LinearWeights    inter_dense;
    LinearWeights    output_dense;
    LayerNormWeights final_layer_norm;
};

// The HuBERT semantic branch, reached only when preparing a cloning profile.
struct SemanticModelWeights {
    // Bias-free, one per conv_dim entry; the first reads the raw waveform.
    std::vector<Conv1dWeights>        feat_conv;
    // feat_extract_norm is "group", so only the first convolution is normalized.
    LayerNormWeights                  feat_conv_norm;
    LayerNormWeights                  feature_projection_norm;
    LinearWeights                     feature_projection;
    Conv1dWeights                     pos_conv;  // grouped, and weight-norm already folded
    std::vector<SemanticLayerWeights> layers;
    LayerNormWeights                  encoder_norm;
};

// Two bias-free convolutions around an ELU, added back to the input.
struct SemanticEncoderResUnit {
    Conv1dWeights conv1;
    Conv1dWeights conv2;
};

struct SemanticEncoderBlock {
    std::vector<SemanticEncoderResUnit> res_units;
    Conv1dWeights                       conv;
};

// The codec's own encoder over HuBERT features, whose output is concatenated
// with the acoustic encoder's before the quantizer.
struct SemanticEncoderWeights {
    Conv1dWeights                     conv;  // bias-free
    std::vector<SemanticEncoderBlock> blocks;
};

struct ModelWeights {
    GeneratorWeights                 generator;
    std::vector<RvqQuantizerWeights> quantizers;  // order = codebook index
    LinearWeights                    fc;          // encode-path concat projection
    LinearWeights                    fc2;         // decode-path concat width -> hidden_size
    AcousticDecoderWeights           acoustic_decoder;
    AcousticEncoderWeights           acoustic_encoder;
    SemanticModelWeights             semantic_model;
    SemanticEncoderWeights           encoder_semantic;
};

// Resolves the whole catalog against a loaded package.
//
// Every shape is derived from the package's own hyper-parameters rather than
// hardcoded, so a package whose metadata and tensors disagree is refused here
// instead of producing wrong audio later. Afterwards the package is swept: a
// tensor the catalog never asked for is an error, not something to ignore,
// because a name nobody resolves is a name nobody checked.
//
// Plan 1 is CPU-only, so there is no accelerator-twin context parameter yet; the
// backends stage adds that seam once there is a placement to make.
synth_status_t build_model_weights(ggml_context * context, const HParams & hparams, ModelWeights & weights);

// The number of tensors a package for these hyper-parameters must contain.
// Exposed so a caller can size a context before resolving anything.
uint64_t expected_tensor_count(const HParams & hparams);

}  // namespace synth::omnivoice
