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

// Resolves the whole catalog against a loaded package. Always against
// `context` alone -- see bind_decode_weights below for the codec's
// accelerator twin, which is a materially different binding from this one and
// not a variant of it.
//
// Every shape is derived from the package's own hyper-parameters rather than
// hardcoded, so a package whose metadata and tensors disagree is refused here
// instead of producing wrong audio later. Afterwards the package is swept: a
// tensor the catalog never asked for is an error, not something to ignore,
// because a name nobody resolves is a name nobody checked.
synth_status_t build_model_weights(ggml_context * context, const HParams & hparams, ModelWeights & weights);

// Binds the codec's DECODE path against an accelerator twin -- narrower than
// qwen3-tts's own twin, which mirrors its whole codec half because nothing
// else under `codec.` there is CPU-held. Byte counts below are this
// checkpoint's F32 sizes, from the per-tensor `bytes` field of
// reports/convert/omnivoice/omnivoice-0-6b-F32.json -- the same file this
// catalog's layout comment above cites as ground truth for names.
//
//   MOVABLE (read only by build_codec_decoder, via Model::decode_codes):
//     codec.acoustic_decoder.*   110 tensors   77.20 MiB
//     codec.quantizer.*           40 tensors    6.03 MiB
//     codec.fc2.*                  2 tensors    1.00 MiB
//                                 ---           84.24 MiB total
//
//   NOT MOVABLE (read only by the CPU-held clone-encode chain --
//   reference-encoder.cpp, via reference-encoder-host.cpp):
//     codec.acoustic_encoder.*   110 tensors  195.75 MiB
//     codec.semantic_model.*     209 tensors  360.00 MiB
//     codec.encoder_semantic.*    13 tensors   56.26 MiB
//     codec.fc.*                   2 tensors    4.00 MiB
//                                 ---          616.01 MiB total
//
// The NOT MOVABLE half's own output is continuous, but what reads it --
// rvq_encode's host-side nearest-neighbour argmax (reference-encoder-host.h)
// -- is a discrete decision, so docs/backends.md's discrete-outputs rule holds
// that whole chain on the CPU the same as the generator. Twinning it anyway,
// the way a filter copied from qwen3-tts's blanket `codec.` prefix would,
// moves ~616 MiB to the primary backend that no primary-side graph ever
// reads.
//
// THE SECOND-CONSUMER TRAP (found in review of this task's first draft, which
// had this function overwrite `weights`'s own quantizer/fc2/acoustic_decoder
// pointers in place -- docs/backends.md:114-119's own worked case): unlike
// every other movable group, `codec.quantizer.*` is not read by the decode
// graph alone. rvq_encode (reference-encoder-host.h), the CPU-held clone
// ENCODE direction's host-side nearest-neighbour argmax, reads the exact same
// projections and codebook through `ModelWeights::quantizers` too. Overwriting
// `weights.quantizers` to point at the twin would have made that host loop
// silently issue `ggml_backend_tensor_get` against a CUDA-resident tensor on
// every cloning request the moment Task 11 lands a CUDA primary -- correct
// (that call is backend-aware, not a raw `->data` dereference) but exactly the
// violation docs/backends.md's "give any group a second view" sentence warns
// against: a stage that must stay off the accelerator quietly paying a
// per-request PCIe round trip for weights that were supposed to be
// CPU-resident, with no test or comment anywhere naming it as intentional.
//
// The fix: `weights` (built by build_model_weights above) is NEVER mutated
// here. `decode_weights` starts as a copy of it -- so every non-movable field,
// and every movable field when `codec_context` is null, is pointer-identical
// to `weights`'s own -- and only its quantizers/fc2/acoustic_decoder are
// re-resolved against `codec_context` when present. The host clone-encode
// path keeps reading `weights` (CPU, always); only Model::decode_codes reads
// `decode_weights` (the twin, when one exists). Neither consumer can silently
// read the other's copy: the invariant tests/omnivoice_catalog_test.cpp's
// check_twin_resolution pins by pointer identity in both directions.
synth_status_t bind_decode_weights(ggml_context *       codec_context,
                                   const HParams &      hparams,
                                   const ModelWeights & weights,
                                   ModelWeights &       decode_weights);

// Binds the GENERATOR against an accelerator twin (Plan 5 Task 1) -- the
// mask-predict denoising loop's own weights, all 312 generator tensors:
// `llm.*`, `audio_embeddings.weight`, `audio_heads.weight`. Byte total from
// the same reports/convert/omnivoice/omnivoice-0-6b-F32.json this file's
// other comments cite as ground truth: 2,450,309,120 bytes, 2,336.80 MiB.
//
// Unlike bind_decode_weights above, the whole group is movable -- there is
// no NOT-MOVABLE remainder to carve out, because the generator has no
// sibling stage that reads a subset of its own tensors the way the
// clone-encode chain reads codec.acoustic_encoder/semantic_model/
// encoder_semantic. And unlike codec.quantizer.* in bind_decode_weights,
// GeneratorWeights has no second host-side consumer at all: grepping every
// reader of `weights.generator` (Plan 5 Task 1's own pre-flight check) finds
// exactly two functions, build_canvas_embedding and build_generator_forward
// (generator.cpp), both reached only through model.cpp's file-local
// generator_branch_forward, itself called only from Model::run_synthesis.
// There is no rvq_encode-shaped trap here to close.
//
// The split is still built the second-consumer-safe way regardless, for two
// reasons rather than one: a future reader that bypasses
// generator_branch_forward must keep seeing the CPU-resident package by
// construction, not by continued vigilance that this comment's "no second
// consumer today" claim stays true; and tests/omnivoice_catalog_test.cpp's
// own unit tests call build_model_weights directly and must keep observing
// its output unaffected by whatever this function does elsewhere. `weights`
// (built by build_model_weights) is NEVER mutated here -- `generator_weights`
// starts as a copy of it, so every non-generator field, and the generator
// field itself when `generator_context` is null, is pointer-identical to
// `weights`'s own; only `generator_weights.generator` is re-resolved against
// `generator_context` when present. Only Model::generator_branch_forward
// (via Model::run_synthesis) reads `generator_weights`; every other reader
// keeps reading `weights`. tests/omnivoice_catalog_test.cpp's
// check_generator_twin_resolution pins the pointer-identity invariant in
// both directions, mirroring check_twin_resolution's own shape.
synth_status_t bind_generator_weights(ggml_context *       generator_context,
                                      const HParams &      hparams,
                                      const ModelWeights & weights,
                                      ModelWeights &       generator_weights);

// The number of tensors a package for these hyper-parameters must contain.
// Exposed so a caller can size a context before resolving anything.
uint64_t expected_tensor_count(const HParams & hparams);

}  // namespace synth::omnivoice
