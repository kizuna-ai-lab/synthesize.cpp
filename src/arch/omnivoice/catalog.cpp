// Resolution of the OmniVoice tensor catalog.
//
// Every entry is looked up by its canonical name and checked for storage type
// and shape, with the shape derived from the package hyper-parameters. A package
// whose metadata and tensors disagree is rejected at load rather than producing
// wrong audio, and a tensor the catalog never asks for is rejected too: a name
// nobody resolves is a name nobody checked.
//
// GGML reports dimensions in the reverse of PyTorch's order. A Linear stored as
// [out, in] has ne = [in, out]; a Conv1d stored as [out, in, kernel] has ne =
// [kernel, in, out]; a ConvTranspose1d stored as [in, out, kernel] has ne =
// [kernel, out, in]. The last two differ only in the order of the trailing pair,
// which is why they have separate helpers here rather than one with a flag.
//
// The layout below is transcribed from the emitted names in
// reports/convert/omnivoice/omnivoice-0-6b-F32.json, which is this catalog's
// ground truth: the converter prefixes the codec half with `codec.`, shortens
// four infixes (`.attention.` -> `.attn.`, `.feed_forward.` -> `.ff.`,
// `.intermediate_dense.` -> `.inter_dense.`, `.feature_extractor.conv_layers.`
// -> `.feat_conv.`) to stay inside GGML's 64-byte name field, and drops the
// training-only modules. 798 names survive:
//
//   generator, 312
//     llm.embed_tokens.weight                          [hidden, text_vocab]
//     llm.layers.{0..L-1}.                             11 each
//       input_layernorm.weight                         [hidden]
//       self_attn.{q,k,v}_proj.weight                  [hidden, heads*head_dim]
//       self_attn.o_proj.weight                        [heads*head_dim, hidden]
//       self_attn.{q,k}_norm.weight                    [head_dim]
//       post_attention_layernorm.weight                [hidden]
//       mlp.{gate,up}_proj.weight                      [hidden, intermediate]
//       mlp.down_proj.weight                           [intermediate, hidden]
//     llm.norm.weight                                  [hidden]
//     audio_{embeddings,heads}.weight                  [hidden, codebooks*vocab]
//
//   codec.quantizer, 40 -- one biased Linear pair and one table per codebook
//     quantizers.{0..7}.project_in.{weight,bias}       [concat, dim] / [dim]
//     quantizers.{0..7}.project_out.{weight,bias}      [dim, concat] / [concat]
//     quantizers.{0..7}.codebook.embed                 [dim, codebook_size]
//
//   codec.fc / codec.fc2, 4 -- the concatenation projection and its inverse
//
//   codec.acoustic_decoder, 110
//     conv1.{weight,bias}                              [7, hidden, dec_hidden]
//     block.{0..R-1}.                                  21 each, width halving
//       snake1.alpha                                   [1, width, 1]
//       conv_t1.{weight,bias}                          [2*ratio, narrower, width]
//       res_unit{1,2,3}.                               6 each
//         snake1.alpha, conv1.{weight,bias}            [1, w, 1] / [7, w, w]
//         snake2.alpha, conv2.{weight,bias}            [1, w, 1] / [1, w, w]
//     snake1.alpha, conv2.{weight,bias}                [7, width, 1] -- mono out
//
//   codec.acoustic_encoder, 110 -- the decoder mirrored, width doubling, the
//     residual units first and a strided conv1 last, exiting through a kernel-3
//     conv2 into the acoustic width
//
//   codec.semantic_model, 209 -- HuBERT
//     feat_conv.{0..C-1}.conv.weight                   [kernel[i], in, dim[i]], no bias
//     feat_conv.0.layer_norm.{weight,bias}             group norm, first layer only
//     feature_projection.layer_norm.{weight,bias}      [dim.back()]
//     feature_projection.projection.{weight,bias}      [dim.back(), semantic]
//     encoder.pos_conv_embed.conv.{weight,bias}        [128, semantic/16, semantic]
//     encoder.layers.{0..11}.                          16 each
//       attn.{q,k,v,out}_proj.{weight,bias}            square and biased
//       layer_norm.{weight,bias}
//       ff.inter_dense.{weight,bias}                   [semantic, intermediate]
//       ff.output_dense.{weight,bias}                  [intermediate, semantic]
//       final_layer_norm.{weight,bias}
//     encoder.layer_norm.{weight,bias}
//
//   codec.encoder_semantic, 13
//     conv.weight                                      [3, semantic, semantic], no bias
//     conv_blocks.{0,1}.                               6 each
//       res_units.{0,1}.conv1.weight                   [3, semantic, semantic], no bias
//       res_units.{0,1}.conv2.weight                   [1, semantic, semantic], no bias
//       conv.{weight,bias}                             [3, semantic, semantic]

#include "arch/omnivoice/catalog.h"

#include "arch/omnivoice/quantization.h"
#include "arch/omnivoice/weights.h"
#include "ggml.h"

#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <initializer_list>
#include <set>
#include <string>
#include <vector>

namespace synth::omnivoice {

namespace {

// The DAC residual stack runs three units per block at dilations 1, 3 and 9. The
// dilation changes no shape, so only the count reaches the catalog.
constexpr size_t  kDacUnitCount              = 3;
// The wide convolution of a residual unit, and the pointwise one that follows.
constexpr int64_t kDacWideKernel             = 7;
constexpr int64_t kDacPointKernel            = 1;
// The stack's own entry and exit convolutions.
constexpr int64_t kDacOuterKernel            = 7;
// The acoustic encoder's exit convolution, the one kernel-3 in that half.
constexpr int64_t kAcousticEncoderExitKernel = 3;

// HuBERT's positional convolution: 128 taps in 16 groups, so each filter sees a
// sixteenth of the width. Neither number is in the package -- they are fixed by
// the HuBERT configuration this checkpoint's semantic model was trained under --
// and a checkpoint that moved either is refused here by shape.
constexpr int64_t kPosConvKernel = 128;
constexpr int64_t kPosConvGroups = 16;

// The codec's semantic encoder: two stride-1 blocks of two dilation-1 residual
// units, all at the HuBERT width. The checkpoint's codec config states these as
// `strides`, `channel_ratios` and `block_dilations`, each [1, 1], and the
// package does not carry them; a checkpoint that changed them is refused by the
// post-resolution sweep or by shape rather than silently mis-bound.
constexpr size_t  kSemanticEncoderBlocks    = 2;
constexpr size_t  kSemanticEncoderUnits     = 2;
constexpr int64_t kSemanticEncoderKernel    = 3;
constexpr int64_t kSemanticEncoderPointwise = 1;

// A resolution pass, so the failure is reported once with its reason rather than
// unwound through a chain of booleans.
class Resolver {
  public:
    Resolver(ggml_context * context, const HParams & hparams) : context_(context), hparams_(hparams) {}

    bool ok() const { return ok_; }

    const std::set<std::string> & resolved() const { return resolved_; }

    ggml_tensor * find(const std::string & name, std::initializer_list<int64_t> expected) {
        if (!ok_) {
            return nullptr;
        }
        ggml_tensor * tensor = ggml_get_tensor(context_, name.c_str());
        if (tensor == nullptr) {
            return fail("missing tensor %s", name.c_str());
        }
        const ggml_type want_type = expected_type(name, tensor->ne);
        if (tensor->type != want_type) {
            return fail("tensor %s has type %s, expected %s under the %s profile", name.c_str(),
                        ggml_type_name(tensor->type), ggml_type_name(want_type), profile_name());
        }
        // A convolution kernel that a Q8_MIXED profile packed collapses its
        // logical [kernel, in, out] shape into the flattened
        // [kernel * in, out] a matrix multiply consumes -- the offline
        // quantizer's own layout for a MatrixWeight tensor
        // (tools/synthesize-quantize/quantize.cpp). Every caller here still
        // supplies the logical three-axis shape, so a quantized tensor whose
        // caller asked for three axes is checked against the packed row
        // instead of axis by axis.
        if (hparams_.quantization_profile == QuantizationProfile::Q8Mixed && ggml_is_quantized(tensor->type) &&
            expected.size() == 3) {
            const auto *  want   = expected.begin();
            const int64_t packed = want[0] * want[1];
            if (want[0] <= 0 || want[1] <= 0 || tensor->ne[0] != packed || tensor->ne[1] != want[2] ||
                tensor->ne[2] != 1 || tensor->ne[3] != 1) {
                return fail("tensor %s does not have the packed shape [%lld, %lld]", name.c_str(), (long long) packed,
                            (long long) want[2]);
            }
            resolved_.insert(name);
            return tensor;
        }
        size_t axis = 0;
        for (int64_t want : expected) {
            if (axis >= GGML_MAX_DIMS || want <= 0 || tensor->ne[axis] != want) {
                return fail("tensor %s has shape [%lld, %lld, %lld, %lld], expected axis %zu to be %lld", name.c_str(),
                            (long long) tensor->ne[0], (long long) tensor->ne[1], (long long) tensor->ne[2],
                            (long long) tensor->ne[3], axis, (long long) want);
            }
            ++axis;
        }
        for (; axis < GGML_MAX_DIMS; ++axis) {
            if (tensor->ne[axis] != 1) {
                return fail("tensor %s has an unexpected extent on axis %zu", name.c_str(), axis);
            }
        }
        resolved_.insert(name);
        return tensor;
    }

    // A Linear's weight is [in, out] once reversed.
    bool linear(const std::string & prefix, int64_t in, int64_t out, LinearWeights & target) {
        target.weight = find(prefix + ".weight", { in, out });
        target.bias   = find(prefix + ".bias", { out });
        return ok_;
    }

    bool conv(const std::string & prefix, int64_t kernel, int64_t in, int64_t out, Conv1dWeights & target) {
        target.weight = find(prefix + ".weight", { kernel, in, out });
        target.bias   = find(prefix + ".bias", { out });
        return ok_;
    }

    // The HuBERT feature extractor's convolutions and the codec's semantic
    // encoder's inner ones store no bias, so `bias` stays null.
    bool bare_conv(const std::string & prefix, int64_t kernel, int64_t in, int64_t out, Conv1dWeights & target) {
        target.weight = find(prefix + ".weight", { kernel, in, out });
        return ok_;
    }

    bool transpose_conv(const std::string & prefix, int64_t kernel, int64_t in, int64_t out, Conv1dWeights & target) {
        target.weight = find(prefix + ".weight", { kernel, out, in });
        target.bias   = find(prefix + ".bias", { out });
        return ok_;
    }

    bool layer_norm(const std::string & prefix, int64_t channels, LayerNormWeights & target) {
        target.weight = find(prefix + ".weight", { channels });
        target.bias   = find(prefix + ".bias", { channels });
        return ok_;
    }

    // Plain Snake, one curve per channel, stored as [1, channels, 1] rather than
    // flat. The rank is part of the contract: a flattened curve broadcasts
    // differently and would be silently wrong.
    bool snake(const std::string & prefix, int64_t channels, SnakeWeights & target) {
        target.alpha = find(prefix + ".alpha", { 1, channels, 1 });
        return ok_;
    }

    bool refuse(const char * format, ...) {
        va_list arguments;
        va_start(arguments, format);
        report(format, arguments);
        va_end(arguments);
        ok_ = false;
        return false;
    }

  private:
    // `name`/`ne` feed classify_tensor, the same classifier the offline
    // quantizer dispatches from (quantization.h), so an offline packing
    // decision and this load-time expectation cannot drift apart. Every role
    // this catalog resolves today classifies from the name alone (see that
    // header's own comment), so `ne` matters only if a future tensor's role
    // ever needs it.
    ggml_type expected_type(const std::string & name, const int64_t * ne) const {
        switch (hparams_.quantization_profile) {
            case QuantizationProfile::F32:
                // The source profile carries the checkpoint through unchanged,
                // and both halves were already F32.
                return GGML_TYPE_F32;
            case QuantizationProfile::Q8Mixed:
                // Codec-only: only a MatrixWeight tensor is ever packed.
                // TransposeWeight and Sensitive both stay F32 -- the former
                // for the same col2im_1d/CUDA-F16 reason VITS and Qwen3-TTS's
                // decoder do, the latter because jiangzhuo's 2026-08-06 ruling
                // keeps the whole generator and the RVQ exact. An Unknown role
                // here means a catalog name the classifier does not
                // recognise, which the type check below still catches as a
                // mismatch against whatever the tensor actually is.
                return classify_tensor(name, ne) == QuantRole::MatrixWeight ? GGML_TYPE_Q8_0 : GGML_TYPE_F32;
        }
        return GGML_TYPE_F32;
    }

    const char * profile_name() const {
        switch (hparams_.quantization_profile) {
            case QuantizationProfile::F32:
                return "F32";
            case QuantizationProfile::Q8Mixed:
                return "Q8_MIXED";
        }
        return "unknown";
    }

    void report(const char * format, va_list arguments) {
        if (!ok_) {
            return;
        }
        std::fputs("omnivoice: ", stderr);
        std::vfprintf(stderr, format, arguments);
        std::fputc('\n', stderr);
    }

    ggml_tensor * fail(const char * format, ...) {
        va_list arguments;
        va_start(arguments, format);
        report(format, arguments);
        va_end(arguments);
        ok_ = false;
        return nullptr;
    }

    ggml_context *        context_;
    const HParams &       hparams_;
    bool                  ok_ = true;
    std::set<std::string> resolved_;
};

std::string index_of(const std::string & prefix, size_t index, const std::string & suffix) {
    return prefix + std::to_string(index) + suffix;
}

// The RVQ works on the acoustic latent and the HuBERT latent concatenated, which
// is what the reference calls the codec's hidden size: it computes that as
// acoustic + semantic and never stores it, so the package does not carry it
// either. `fc` projects the concatenation in place ahead of the quantizer and
// `fc2` brings the dequantized latent back down to the acoustic width.
int64_t concat_width(const HParams & hparams) {
    return int64_t(hparams.codec.hidden_size) + hparams.semantic.hidden_size;
}

bool resolve_generator(Resolver & resolver, const HParams & hparams, GeneratorWeights & generator) {
    const GeneratorParams & g = hparams.generator;

    // The embedding is tied to the text head, so no lm_head exists to resolve.
    generator.text_embedding = resolver.find("llm.embed_tokens.weight", { g.hidden_size, g.text_vocab_size });

    const int64_t attention_inner = int64_t(g.attention_head_count) * g.head_dim;
    const int64_t kv_inner        = int64_t(g.key_value_head_count) * g.head_dim;
    generator.layers.resize(g.layer_count);
    for (uint32_t index = 0; index < g.layer_count; ++index) {
        const std::string       base  = index_of("llm.layers.", index, ".");
        GeneratorLayerWeights & layer = generator.layers[index];
        layer.input_layernorm         = resolver.find(base + "input_layernorm.weight", { g.hidden_size });
        layer.q_proj = resolver.find(base + "self_attn.q_proj.weight", { g.hidden_size, attention_inner });
        layer.k_proj = resolver.find(base + "self_attn.k_proj.weight", { g.hidden_size, kv_inner });
        layer.v_proj = resolver.find(base + "self_attn.v_proj.weight", { g.hidden_size, kv_inner });
        layer.o_proj = resolver.find(base + "self_attn.o_proj.weight", { attention_inner, g.hidden_size });
        // Qwen3 normalizes each head at head_dim, not the packed projection, so
        // these are narrow where a per-projection norm would be hidden-wide.
        layer.q_norm = resolver.find(base + "self_attn.q_norm.weight", { g.head_dim });
        layer.k_norm = resolver.find(base + "self_attn.k_norm.weight", { g.head_dim });
        layer.post_attention_layernorm = resolver.find(base + "post_attention_layernorm.weight", { g.hidden_size });
        layer.gate_proj = resolver.find(base + "mlp.gate_proj.weight", { g.hidden_size, g.intermediate_size });
        layer.up_proj   = resolver.find(base + "mlp.up_proj.weight", { g.hidden_size, g.intermediate_size });
        layer.down_proj = resolver.find(base + "mlp.down_proj.weight", { g.intermediate_size, g.hidden_size });
    }
    generator.norm = resolver.find("llm.norm.weight", { g.hidden_size });

    // Every codebook is stacked into one table, so a canvas slot indexes
    // codebook * vocab_size + code. The heads are a second table of the same
    // shape rather than a tie of the first.
    const int64_t canvas       = int64_t(hparams.audio.num_codebooks) * hparams.audio.vocab_size;
    generator.audio_embeddings = resolver.find("audio_embeddings.weight", { g.hidden_size, canvas });
    generator.audio_heads      = resolver.find("audio_heads.weight", { g.hidden_size, canvas });
    return resolver.ok();
}

bool resolve_quantizers(Resolver & resolver, const HParams & hparams, std::vector<RvqQuantizerWeights> & quantizers) {
    const int64_t concat = concat_width(hparams);
    const int64_t dim    = hparams.codec.codebook_dim;
    quantizers.resize(hparams.audio.num_codebooks);
    for (uint32_t index = 0; index < hparams.audio.num_codebooks; ++index) {
        const std::string     base      = index_of("codec.quantizer.quantizers.", index, ".");
        RvqQuantizerWeights & quantizer = quantizers[index];
        resolver.linear(base + "project_in", concat, dim, quantizer.input_proj);
        resolver.linear(base + "project_out", dim, concat, quantizer.output_proj);
        // The live table only. Its EMA numerator and denominator are training
        // state the converter drops, so asking for them here would fail.
        quantizer.codebook = resolver.find(base + "codebook.embed", { dim, hparams.codec.codebook_size });
    }
    return resolver.ok();
}

bool resolve_residual_units(Resolver &                     resolver,
                            const std::string &            base,
                            int64_t                        width,
                            std::vector<DacResidualUnit> & units) {
    units.resize(kDacUnitCount);
    for (size_t index = 0; index < units.size(); ++index) {
        const std::string unit_base = index_of(base + "res_unit", index + 1, ".");
        DacResidualUnit & unit      = units[index];
        resolver.snake(unit_base + "snake1", width, unit.snake1);
        resolver.conv(unit_base + "conv1", kDacWideKernel, width, width, unit.conv1);
        resolver.snake(unit_base + "snake2", width, unit.snake2);
        resolver.conv(unit_base + "conv2", kDacPointKernel, width, width, unit.conv2);
    }
    return resolver.ok();
}

bool resolve_acoustic_decoder(Resolver & resolver, const HParams & hparams, AcousticDecoderWeights & decoder) {
    const CodecParams & c = hparams.codec;
    resolver.conv("codec.acoustic_decoder.conv1", kDacOuterKernel, c.hidden_size, c.decoder_hidden_size, decoder.conv1);

    // One block per upsampling ratio, halving the width each time.
    int64_t width = c.decoder_hidden_size;
    decoder.blocks.resize(c.upsampling_ratios.size());
    for (size_t index = 0; index < c.upsampling_ratios.size(); ++index) {
        const std::string      base     = index_of("codec.acoustic_decoder.block.", index, ".");
        AcousticDecoderBlock & block    = decoder.blocks[index];
        const int64_t          narrower = width / 2;
        resolver.snake(base + "snake1", width, block.snake1);
        // The transposed kernel is twice the rate it upsamples by.
        resolver.transpose_conv(base + "conv_t1", 2 * int64_t(c.upsampling_ratios[index]), width, narrower,
                                block.conv_t1);
        resolve_residual_units(resolver, base, narrower, block.res_units);
        width = narrower;
    }

    resolver.snake("codec.acoustic_decoder.snake1", width, decoder.snake1);
    // One channel out: the codec emits a mono waveform.
    resolver.conv("codec.acoustic_decoder.conv2", kDacOuterKernel, width, 1, decoder.conv2);
    return resolver.ok();
}

bool resolve_acoustic_encoder(Resolver & resolver, const HParams & hparams, AcousticEncoderWeights & encoder) {
    const CodecParams & c = hparams.codec;
    // One channel in: the encoder reads the mono reference waveform.
    resolver.conv("codec.acoustic_encoder.conv1", kDacOuterKernel, 1, c.encoder_hidden_size, encoder.conv1);

    // The mirror of the decoder, in the same ratio order: the checkpoint states
    // its downsampling ratios separately and they are the upsampling ones
    // exactly, so one list drives both halves.
    int64_t width = c.encoder_hidden_size;
    encoder.blocks.resize(c.upsampling_ratios.size());
    for (size_t index = 0; index < c.upsampling_ratios.size(); ++index) {
        const std::string      base  = index_of("codec.acoustic_encoder.block.", index, ".");
        AcousticEncoderBlock & block = encoder.blocks[index];
        const int64_t          wider = width * 2;
        resolve_residual_units(resolver, base, width, block.res_units);
        resolver.snake(base + "snake1", width, block.snake1);
        resolver.conv(base + "conv1", 2 * int64_t(c.upsampling_ratios[index]), width, wider, block.conv1);
        width = wider;
    }

    resolver.snake("codec.acoustic_encoder.snake1", width, encoder.snake1);
    resolver.conv("codec.acoustic_encoder.conv2", kAcousticEncoderExitKernel, width, c.hidden_size, encoder.conv2);
    return resolver.ok();
}

bool resolve_semantic_model(Resolver & resolver, const HParams & hparams, SemanticModelWeights & semantic) {
    const SemanticParams & s = hparams.semantic;
    if (s.conv_dim.empty() || s.conv_dim.size() != s.conv_kernel.size()) {
        return resolver.refuse("the semantic feature extractor declares %zu widths against %zu kernels",
                               s.conv_dim.size(), s.conv_kernel.size());
    }
    if (s.hidden_size == 0 || s.hidden_size % kPosConvGroups != 0) {
        return resolver.refuse("the semantic width %u is not a multiple of the positional convolution's %lld groups",
                               s.hidden_size, (long long) kPosConvGroups);
    }

    // The feature extractor reads the raw waveform, so the first convolution has
    // one input channel. None of them carries a bias.
    semantic.feat_conv.resize(s.conv_dim.size());
    for (size_t index = 0; index < s.conv_dim.size(); ++index) {
        const int64_t in = index == 0 ? 1 : int64_t(s.conv_dim[index - 1]);
        resolver.bare_conv(index_of("codec.semantic_model.feat_conv.", index, ".conv"), s.conv_kernel[index], in,
                           s.conv_dim[index], semantic.feat_conv[index]);
    }
    // feat_extract_norm is "group": only the first convolution is normalized,
    // which is why there is one norm here and not one per layer.
    resolver.layer_norm("codec.semantic_model.feat_conv.0.layer_norm", s.conv_dim[0], semantic.feat_conv_norm);

    const int64_t features = s.conv_dim.back();
    resolver.layer_norm("codec.semantic_model.feature_projection.layer_norm", features,
                        semantic.feature_projection_norm);
    resolver.linear("codec.semantic_model.feature_projection.projection", features, s.hidden_size,
                    semantic.feature_projection);

    // Grouped, and the checkpoint's weight-norm parametrization already folded
    // into a plain kernel by the converter.
    resolver.conv("codec.semantic_model.encoder.pos_conv_embed.conv", kPosConvKernel, s.hidden_size / kPosConvGroups,
                  s.hidden_size, semantic.pos_conv);

    semantic.layers.resize(s.layer_count);
    for (uint32_t index = 0; index < s.layer_count; ++index) {
        const std::string      base  = index_of("codec.semantic_model.encoder.layers.", index, ".");
        SemanticLayerWeights & layer = semantic.layers[index];
        // HuBERT splits the width across heads rather than declaring a head
        // width, so every attention projection here is square and biased.
        resolver.linear(base + "attn.q_proj", s.hidden_size, s.hidden_size, layer.q_proj);
        resolver.linear(base + "attn.k_proj", s.hidden_size, s.hidden_size, layer.k_proj);
        resolver.linear(base + "attn.v_proj", s.hidden_size, s.hidden_size, layer.v_proj);
        resolver.linear(base + "attn.out_proj", s.hidden_size, s.hidden_size, layer.out_proj);
        resolver.layer_norm(base + "layer_norm", s.hidden_size, layer.layer_norm);
        resolver.linear(base + "ff.inter_dense", s.hidden_size, s.intermediate_size, layer.inter_dense);
        resolver.linear(base + "ff.output_dense", s.intermediate_size, s.hidden_size, layer.output_dense);
        resolver.layer_norm(base + "final_layer_norm", s.hidden_size, layer.final_layer_norm);
    }
    resolver.layer_norm("codec.semantic_model.encoder.layer_norm", s.hidden_size, semantic.encoder_norm);
    return resolver.ok();
}

bool resolve_encoder_semantic(Resolver & resolver, const HParams & hparams, SemanticEncoderWeights & encoder) {
    const int64_t width = hparams.semantic.hidden_size;
    resolver.bare_conv("codec.encoder_semantic.conv", kSemanticEncoderKernel, width, width, encoder.conv);

    encoder.blocks.resize(kSemanticEncoderBlocks);
    for (size_t index = 0; index < encoder.blocks.size(); ++index) {
        const std::string      base  = index_of("codec.encoder_semantic.conv_blocks.", index, ".");
        SemanticEncoderBlock & block = encoder.blocks[index];
        block.res_units.resize(kSemanticEncoderUnits);
        for (size_t unit = 0; unit < block.res_units.size(); ++unit) {
            const std::string        unit_base = index_of(base + "res_units.", unit, ".");
            SemanticEncoderResUnit & target    = block.res_units[unit];
            resolver.bare_conv(unit_base + "conv1", kSemanticEncoderKernel, width, width, target.conv1);
            resolver.bare_conv(unit_base + "conv2", kSemanticEncoderPointwise, width, width, target.conv2);
        }
        // The block's own convolution is the only biased one in this module, and
        // it is kernel-3 because its stride is 1.
        resolver.conv(base + "conv", kSemanticEncoderKernel, width, width, block.conv);
    }
    return resolver.ok();
}

}  // namespace

uint64_t expected_tensor_count(const HParams & hparams) {
    // 11 per generator layer: the input norm, four projections, two per-head
    // norms, the post-attention norm, three feed-forward matrices.
    constexpr uint64_t kPerGeneratorLayer = 11;
    // 6 per DAC residual unit: two Snake curves and two biased convolutions.
    constexpr uint64_t kPerResidualUnit   = 6;
    // 21 per DAC block: three residual units, one Snake curve and one biased
    // resampling convolution.
    constexpr uint64_t kPerDacBlock       = kDacUnitCount * kPerResidualUnit + 1 + 2;
    // 16 per HuBERT layer: four biased attention projections, two biased
    // feed-forward matrices, two layer norms.
    constexpr uint64_t kPerSemanticLayer  = 4 * 2 + 2 * 2 + 2 * 2;

    // 312 real: 1 + 28*11 + 1 + 2.
    uint64_t generator = 1 + uint64_t(hparams.generator.layer_count) * kPerGeneratorLayer + 1 + 2;

    // 40 real: 8 * (2 + 2 + 1).
    uint64_t quantizers  = uint64_t(hparams.audio.num_codebooks) * (2 + 2 + 1);
    // fc and fc2, both biased.
    uint64_t projections = 4;

    // 110 real each: 2 + 5*21 + 1 + 2. The two halves are the same shape of
    // module walked in opposite directions, so they count the same.
    const uint64_t blocks           = hparams.codec.upsampling_ratios.size();
    uint64_t       acoustic_decoder = 2 + blocks * kPerDacBlock + 1 + 2;
    uint64_t       acoustic_encoder = 2 + blocks * kPerDacBlock + 1 + 2;

    // 209 real: 7 bias-free feature convolutions, the group norm on the first of
    // them, the feature projection's norm and matrix, the positional
    // convolution, 12*16 layers, the encoder's output norm.
    uint64_t semantic_model = uint64_t(hparams.semantic.conv_dim.size()) + 2 + 4 + 2 +
                              uint64_t(hparams.semantic.layer_count) * kPerSemanticLayer + 2;

    // 13 real: one bias-free entry convolution, then per block two bias-free
    // residual units of two convolutions each and one biased convolution.
    uint64_t encoder_semantic = 1 + uint64_t(kSemanticEncoderBlocks) * (kSemanticEncoderUnits * 2 + 2);

    // 798 real: 312 generator + 486 codec.
    return generator + quantizers + projections + acoustic_decoder + acoustic_encoder + semantic_model +
           encoder_semantic;
}

synth_status_t build_model_weights(ggml_context * context, const HParams & hparams, ModelWeights & weights) {
    if (context == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    weights = ModelWeights{};

    Resolver resolver(context, hparams);
    if (!resolve_generator(resolver, hparams, weights.generator) ||
        !resolve_quantizers(resolver, hparams, weights.quantizers)) {
        return SYNTH_ERR_GGUF;
    }
    const int64_t concat = concat_width(hparams);
    resolver.linear("codec.fc", concat, concat, weights.fc);
    resolver.linear("codec.fc2", concat, hparams.codec.hidden_size, weights.fc2);
    if (!resolver.ok() || !resolve_acoustic_decoder(resolver, hparams, weights.acoustic_decoder) ||
        !resolve_acoustic_encoder(resolver, hparams, weights.acoustic_encoder) ||
        !resolve_semantic_model(resolver, hparams, weights.semantic_model) ||
        !resolve_encoder_semantic(resolver, hparams, weights.encoder_semantic)) {
        return SYNTH_ERR_GGUF;
    }

    // A tensor nobody looked up is a tensor nobody checked, so the package is
    // swept rather than trusted. This also catches a catalog that resolved the
    // same name twice and left another unread.
    for (ggml_tensor * tensor = ggml_get_first_tensor(context); tensor != nullptr;
         tensor               = ggml_get_next_tensor(context, tensor)) {
        if (resolver.resolved().count(tensor->name) == 0) {
            std::fprintf(stderr, "omnivoice: tensor %s is outside the catalog\n", tensor->name);
            return SYNTH_ERR_GGUF;
        }
    }
    return SYNTH_OK;
}

}  // namespace synth::omnivoice
