// Resolution of the Qwen3-TTS tensor catalog.
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

#include "catalog.h"

#include "ggml.h"
#include "weights.h"

#include <cstdarg>
#include <cstdio>
#include <initializer_list>
#include <set>
#include <string>

namespace synth::qwen3tts {

namespace {

// Under the source profile the expected type follows the half: the talker's
// carries the checkpoint's own BF16 and the codec's the speech tokenizer's F32.
// Every other profile is decided by what the tensor *is*, which is why the
// resolver carries a role rather than deriving one from the name -- the
// quantizer classifies by name and the two must not drift apart.
enum class Half { Talker, Codec };

enum class Role {
    // Everything a profile is allowed to halve or pack.
    Matrix,
    // A transposed convolution, which runs as a column matrix multiply into
    // col2im_1d. CUDA's F16 multiply accumulates in half precision, so these
    // stay F32 under every profile.
    Transpose,
    // Norms, biases, layer scales, SnakeBeta curves, the quantizer's tables:
    // each multiplies or seeds a whole branch, and together they are a rounding
    // error of the file.
    Sensitive,
};

// A resolution pass, so the failure is reported once with its reason rather than
// unwound through a chain of booleans.
class Resolver {
  public:
    Resolver(ggml_context * context, const HParams & hparams) : context_(context), hparams_(hparams) {}

    bool ok() const { return ok_; }

    const std::set<std::string> & resolved() const { return resolved_; }

    ggml_tensor * find(const std::string & name, std::initializer_list<int64_t> expected, Role role = Role::Sensitive) {
        if (!ok_) {
            return nullptr;
        }
        ggml_tensor * tensor = ggml_get_tensor(context_, name.c_str());
        if (tensor == nullptr) {
            return fail("missing tensor %s", name.c_str());
        }
        if (tensor->type != expected_type(name, role)) {
            return fail("tensor %s has type %s, expected %s under the %s profile", name.c_str(),
                        ggml_type_name(tensor->type), ggml_type_name(expected_type(name, role)), profile_name());
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
    bool linear(const std::string & prefix, int64_t in, int64_t out, LinearWeights & target, bool with_bias = true) {
        target.weight = find(prefix + ".weight", { in, out }, Role::Matrix);
        if (with_bias) {
            target.bias = find(prefix + ".bias", { out });
        }
        return ok_;
    }

    bool conv(const std::string & prefix, int64_t kernel, int64_t in, int64_t out, Conv1dWeights & target) {
        target.weight = find(prefix + ".weight", { kernel, in, out }, Role::Matrix);
        target.bias   = find(prefix + ".bias", { out });
        return ok_;
    }

    // Depthwise: one filter per channel, so the kernel's input extent is 1
    // rather than the channel count.
    bool depthwise_conv(const std::string & prefix, int64_t kernel, int64_t channels, Conv1dWeights & target) {
        target.weight = find(prefix + ".weight", { kernel, 1, channels });
        target.bias   = find(prefix + ".bias", { channels });
        return ok_;
    }

    bool transpose_conv(const std::string & prefix, int64_t kernel, int64_t in, int64_t out, Conv1dWeights & target) {
        target.weight = find(prefix + ".weight", { kernel, out, in }, Role::Transpose);
        target.bias   = find(prefix + ".bias", { out });
        return ok_;
    }

    bool layer_norm(const std::string & prefix, int64_t channels, LayerNormWeights & target) {
        target.weight = find(prefix + ".weight", { channels });
        target.bias   = find(prefix + ".bias", { channels });
        return ok_;
    }

    bool snake_beta(const std::string & prefix, int64_t channels, SnakeBetaWeights & target) {
        target.alpha = find(prefix + ".alpha", { channels });
        target.beta  = find(prefix + ".beta", { channels });
        return ok_;
    }

  private:
    ggml_type expected_type(const std::string & name, Role role) const {
        const Half half = name.compare(0, 6, "codec.") == 0 ? Half::Codec : Half::Talker;
        switch (hparams_.quantization_profile) {
            case QuantizationProfile::BF16:
                // The source profile carries both halves through unchanged, so
                // the role does not enter: what the checkpoint stored is what
                // the package holds.
                return half == Half::Codec ? GGML_TYPE_F32 : GGML_TYPE_BF16;
            case QuantizationProfile::F16:
            case QuantizationProfile::Q8Mixed:
            case QuantizationProfile::Q5KMixed:
                // The codec half is never halved. Its convolutions run through
                // im2col into a matrix multiply, and ggml's F16 path there is
                // slower on CPU than its F32 one -- measured at 1.75 times on a
                // 37-frame case -- so halving 457 MB costs more time than the
                // space is worth. See the quantizer's policy, which classifies
                // the same way.
                if (half == Half::Codec || role != Role::Matrix) {
                    return GGML_TYPE_F32;
                }
                switch (hparams_.quantization_profile) {
                    case QuantizationProfile::F16:
                        return GGML_TYPE_F16;
                    case QuantizationProfile::Q8Mixed:
                        return GGML_TYPE_Q8_0;
                    default:
                        return GGML_TYPE_Q5_K;
                }
        }
        return GGML_TYPE_F32;
    }

    const char * profile_name() const {
        switch (hparams_.quantization_profile) {
            case QuantizationProfile::BF16:
                return "BF16";
            case QuantizationProfile::F16:
                return "F16";
            case QuantizationProfile::Q8Mixed:
                return "Q8_MIXED";
            case QuantizationProfile::Q5KMixed:
                return "Q5_K_MIXED";
        }
        return "unknown";
    }

    ggml_tensor * fail(const char * format, ...) {
        if (ok_) {
            std::fputs("qwen3-tts: ", stderr);
            va_list arguments;
            va_start(arguments, format);
            std::vfprintf(stderr, format, arguments);
            va_end(arguments);
            std::fputc('\n', stderr);
        }
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

// The talker and the code predictor are the same block, so one helper binds
// both; only the prefix and the layer count differ.
bool resolve_decoder_layers(Resolver &                         resolver,
                            const std::string &                prefix,
                            uint32_t                           layer_count,
                            uint32_t                           hidden,
                            uint32_t                           intermediate,
                            uint32_t                           head_dim,
                            uint32_t                           heads,
                            uint32_t                           kv_heads,
                            std::vector<DecoderLayerWeights> & layers) {
    layers.resize(layer_count);
    const int64_t attention_inner = int64_t(heads) * head_dim;
    const int64_t kv_inner        = int64_t(kv_heads) * head_dim;
    for (uint32_t index = 0; index < layer_count; ++index) {
        const std::string     base  = index_of(prefix, index, ".");
        DecoderLayerWeights & layer = layers[index];
        layer.input_layernorm       = resolver.find(base + "input_layernorm.weight", { hidden });
        layer.q_proj = resolver.find(base + "self_attn.q_proj.weight", { hidden, attention_inner }, Role::Matrix);
        layer.k_proj = resolver.find(base + "self_attn.k_proj.weight", { hidden, kv_inner }, Role::Matrix);
        layer.v_proj = resolver.find(base + "self_attn.v_proj.weight", { hidden, kv_inner }, Role::Matrix);
        layer.o_proj = resolver.find(base + "self_attn.o_proj.weight", { attention_inner, hidden }, Role::Matrix);
        // Qwen3 normalizes each head at head_dim, not the packed projection, so
        // these are narrow where a per-projection norm would be hidden-wide.
        layer.q_norm = resolver.find(base + "self_attn.q_norm.weight", { head_dim });
        layer.k_norm = resolver.find(base + "self_attn.k_norm.weight", { head_dim });
        layer.post_attention_layernorm = resolver.find(base + "post_attn_norm.weight", { hidden });
        layer.gate_proj = resolver.find(base + "mlp.gate_proj.weight", { hidden, intermediate }, Role::Matrix);
        layer.up_proj   = resolver.find(base + "mlp.up_proj.weight", { hidden, intermediate }, Role::Matrix);
        layer.down_proj = resolver.find(base + "mlp.down_proj.weight", { intermediate, hidden }, Role::Matrix);
    }
    return resolver.ok();
}

bool resolve_talker(Resolver & resolver, const HParams & hparams, TalkerWeights & talker) {
    const TalkerParams & p = hparams.talker;
    talker.text_embedding =
        resolver.find("talker.model.text_embedding.weight", { p.text_hidden_size, p.text_vocab_size }, Role::Matrix);
    resolver.linear("talker.text_projection.linear_fc1", p.text_hidden_size, p.text_hidden_size,
                    talker.text_projection_1);
    resolver.linear("talker.text_projection.linear_fc2", p.text_hidden_size, p.hidden_size, talker.text_projection_2);
    talker.codec_embedding =
        resolver.find("talker.model.codec_embedding.weight", { p.hidden_size, p.codec_vocab_size }, Role::Matrix);
    talker.codec_head = resolver.find("talker.codec_head.weight", { p.hidden_size, p.codec_vocab_size }, Role::Matrix);
    resolve_decoder_layers(resolver, "talker.model.layers.", p.layer_count, p.hidden_size, p.intermediate_size,
                           p.head_dim, p.attention_head_count, p.key_value_head_count, talker.layers);
    talker.norm = resolver.find("talker.model.norm.weight", { p.hidden_size });
    return resolver.ok();
}

bool resolve_code_predictor(Resolver & resolver, const HParams & hparams, CodePredictorWeights & predictor) {
    const CodePredictorParams & p = hparams.code_predictor;
    // The predictor shares the talker's feed-forward width; the package does not
    // declare a separate one, so it is checked against the talker's.
    resolve_decoder_layers(resolver, "talker.code_predictor.model.layers.", p.layer_count, p.hidden_size,
                           hparams.talker.intermediate_size, p.head_dim, p.attention_head_count, p.key_value_head_count,
                           predictor.layers);
    predictor.norm = resolver.find("talker.code_predictor.model.norm.weight", { p.hidden_size });

    // One private table and one private head per acoustic group. Group 0 is the
    // talker's, which is why both lists are one shorter than the group count.
    const uint32_t acoustic = p.code_group_count - 1;
    predictor.codec_embedding.assign(acoustic, nullptr);
    predictor.lm_head.assign(acoustic, nullptr);
    for (uint32_t group = 0; group < acoustic; ++group) {
        predictor.codec_embedding[group] =
            resolver.find(index_of("talker.code_predictor.model.codec_embedding.", group, ".weight"),
                          { p.hidden_size, p.vocab_size }, Role::Matrix);
        predictor.lm_head[group] = resolver.find(index_of("talker.code_predictor.lm_head.", group, ".weight"),
                                                 { p.hidden_size, p.vocab_size }, Role::Matrix);
    }

    // The reference projects the talker's hidden state into the predictor's
    // width and drops the projection entirely when the widths agree. A package
    // whose widths differ would need one, and this variant's do not.
    if (p.hidden_size != hparams.talker.hidden_size) {
        std::fprintf(stderr,
                     "qwen3-tts: the code predictor is %u wide against the talker's %u, which needs an input "
                     "projection this package does not carry\n",
                     p.hidden_size, hparams.talker.hidden_size);
        return false;
    }
    return resolver.ok();
}

bool resolve_quantizer(Resolver &                 resolver,
                       const std::string &        prefix,
                       const CodecDecoderParams & p,
                       uint32_t                   codebook_count,
                       CodecQuantizerWeights &    quantizer) {
    // The quantizer works at half the codebook dimension, with a kernel-1
    // convolution on either side changing the width. Neither carries a bias.
    const int64_t inner   = p.codebook_dim / 2;
    quantizer.input_proj  = resolver.find(prefix + "input_proj.weight", { 1, p.codebook_dim, inner });
    quantizer.output_proj = resolver.find(prefix + "output_proj.weight", { 1, inner, p.codebook_dim });
    quantizer.codebooks.assign(codebook_count, nullptr);
    for (uint32_t index = 0; index < codebook_count; ++index) {
        quantizer.codebooks[index] =
            resolver.find(index_of(prefix + "vq.layers.", index, ".codebook"), { inner, p.codebook_size });
    }
    return resolver.ok();
}

bool resolve_codec_transformer(Resolver &                 resolver,
                               const CodecDecoderParams & p,
                               CodecTransformerWeights &  transformer) {
    const std::string prefix = "codec.decoder.pre_transformer.";
    resolver.linear(prefix + "input_proj", p.latent_dim, p.hidden_size, transformer.input_proj);
    resolver.linear(prefix + "output_proj", p.hidden_size, p.latent_dim, transformer.output_proj);

    const int64_t attention_inner = int64_t(p.attention_head_count) * p.head_dim;
    const int64_t kv_inner        = int64_t(p.key_value_head_count) * p.head_dim;
    transformer.layers.resize(p.layer_count);
    for (uint32_t index = 0; index < p.layer_count; ++index) {
        const std::string              base  = index_of(prefix + "layers.", index, ".");
        CodecTransformerLayerWeights & layer = transformer.layers[index];
        layer.input_layernorm                = resolver.find(base + "input_layernorm.weight", { p.hidden_size });
        layer.q_proj =
            resolver.find(base + "self_attn.q_proj.weight", { p.hidden_size, attention_inner }, Role::Matrix);
        layer.k_proj = resolver.find(base + "self_attn.k_proj.weight", { p.hidden_size, kv_inner }, Role::Matrix);
        layer.v_proj = resolver.find(base + "self_attn.v_proj.weight", { p.hidden_size, kv_inner }, Role::Matrix);
        layer.o_proj =
            resolver.find(base + "self_attn.o_proj.weight", { attention_inner, p.hidden_size }, Role::Matrix);
        // Per-branch layer scales, which a Qwen3 block does not have; leaving
        // them unbound would run the residual branches at unit gain.
        layer.self_attn_layer_scale    = resolver.find(base + "self_attn_scale.scale", { p.hidden_size });
        layer.post_attention_layernorm = resolver.find(base + "post_attn_norm.weight", { p.hidden_size });
        layer.gate_proj =
            resolver.find(base + "mlp.gate_proj.weight", { p.hidden_size, p.intermediate_size }, Role::Matrix);
        layer.up_proj =
            resolver.find(base + "mlp.up_proj.weight", { p.hidden_size, p.intermediate_size }, Role::Matrix);
        layer.down_proj =
            resolver.find(base + "mlp.down_proj.weight", { p.intermediate_size, p.hidden_size }, Role::Matrix);
        layer.mlp_layer_scale = resolver.find(base + "mlp_scale.scale", { p.hidden_size });
    }
    transformer.norm = resolver.find(prefix + "norm.weight", { p.hidden_size });
    return resolver.ok();
}

bool resolve_codec(Resolver & resolver, const HParams & hparams, CodecDecoderWeights & codec) {
    const CodecDecoderParams & p = hparams.codec.decoder;

    resolve_quantizer(resolver, "codec.decoder.quantizer.rvq_first.", p, p.semantic_quantizer_count, codec.semantic);
    resolve_quantizer(resolver, "codec.decoder.quantizer.rvq_rest.", p, p.quantizer_count - p.semantic_quantizer_count,
                      codec.acoustic);

    resolver.conv("codec.decoder.pre_conv.conv", 3, p.codebook_dim, p.latent_dim, codec.pre_conv);
    resolve_codec_transformer(resolver, p, codec.pre_transformer);

    // One ConvNeXt stage per upsampling ratio, each a transposed convolution
    // whose kernel is the ratio followed by a depthwise residual block.
    codec.upsample.resize(p.upsampling_ratios.size());
    for (size_t stage = 0; stage < p.upsampling_ratios.size(); ++stage) {
        const std::string    base = index_of("codec.decoder.upsample.", stage, ".");
        CodecUpsampleStage & up   = codec.upsample[stage];
        resolver.transpose_conv(base + "0.conv", p.upsampling_ratios[stage], p.latent_dim, p.latent_dim,
                                up.transpose_conv);
        resolver.depthwise_conv(base + "1.dwconv.conv", 7, p.latent_dim, up.convnext.dwconv);
        resolver.layer_norm(base + "1.norm", p.latent_dim, up.convnext.norm);
        resolver.linear(base + "1.pwconv1", p.latent_dim, 4 * int64_t(p.latent_dim), up.convnext.pwconv1);
        resolver.linear(base + "1.pwconv2", 4 * int64_t(p.latent_dim), p.latent_dim, up.convnext.pwconv2);
        up.convnext.gamma = resolver.find(base + "1.gamma", { p.latent_dim });
    }

    // The residual stack is one flat ModuleList: an input convolution, one block
    // per upsample rate, then the output activation and convolution. The indices
    // are positional, so they are derived here rather than written out.
    resolver.conv("codec.decoder.decoder.0.conv", 7, p.latent_dim, p.dim, codec.input_conv);
    codec.stages.resize(p.upsample_rates.size());
    int64_t width = p.dim;
    for (size_t stage = 0; stage < p.upsample_rates.size(); ++stage) {
        const std::string    base     = index_of("codec.decoder.decoder.", stage + 1, ".block.");
        CodecResidualStage & residual = codec.stages[stage];
        const int64_t        narrower = width / 2;
        resolver.snake_beta(base + "0", width, residual.act);
        // The transposed kernel is twice the rate it upsamples by.
        resolver.transpose_conv(base + "1.conv", 2 * int64_t(p.upsample_rates[stage]), width, narrower,
                                residual.transpose_conv);
        // Three dilated units per stage, at dilations 1, 3 and 9. The dilation
        // does not change any shape, so only the count matters here.
        residual.units.resize(3);
        for (size_t unit = 0; unit < residual.units.size(); ++unit) {
            const std::string   unit_base = index_of(base, unit + 2, ".");
            CodecResidualUnit & target    = residual.units[unit];
            resolver.snake_beta(unit_base + "act1", narrower, target.act1);
            resolver.conv(unit_base + "conv1.conv", 7, narrower, narrower, target.conv1);
            resolver.snake_beta(unit_base + "act2", narrower, target.act2);
            resolver.conv(unit_base + "conv2.conv", 1, narrower, narrower, target.conv2);
        }
        width = narrower;
    }
    const size_t tail = p.upsample_rates.size() + 1;
    resolver.snake_beta(index_of("codec.decoder.decoder.", tail, ""), width, codec.output_act);
    // One channel out: the codec emits a mono waveform.
    resolver.conv(index_of("codec.decoder.decoder.", tail + 1, ".conv"), 7, width, 1, codec.output_conv);
    return resolver.ok();
}

}  // namespace

uint64_t expected_tensor_count(const HParams & hparams) {
    const CodecDecoderParams & codec = hparams.codec.decoder;

    // 11 per decoder layer, shared by the talker and the predictor.
    constexpr uint64_t kPerDecoderLayer = 11;

    uint64_t talker = 1 + 4 + 2 + 1;  // text embedding, projection, codec pair, norm
    talker += uint64_t(hparams.talker.layer_count) * kPerDecoderLayer;

    uint64_t predictor = 1 + uint64_t(hparams.code_predictor.layer_count) * kPerDecoderLayer;
    predictor += 2ull * (hparams.code_predictor.code_group_count - 1);

    uint64_t quantizers  = 2 * 2 + codec.quantizer_count;             // two projections each, plus codebooks
    uint64_t transformer = 4 + 1 + uint64_t(codec.layer_count) * 11;  // projections, norm, layers
    uint64_t upsample    = uint64_t(codec.upsampling_ratios.size()) * 11;
    uint64_t residual    = 2;                                         // the stack's input convolution
    residual += uint64_t(codec.upsample_rates.size()) * (2 + 2 + 3 * 8);
    residual += 2 + 2;                                                // output activation and convolution
    return talker + predictor + quantizers + 2 + transformer + upsample + residual;
}

synth_status_t build_model_weights(ggml_context *  context,
                                   ggml_context *  codec_context,
                                   const HParams & hparams,
                                   ModelWeights &  weights) {
    if (context == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    weights = ModelWeights{};

    Resolver resolver(context, hparams);
    if (!resolve_talker(resolver, hparams, weights.talker) ||
        !resolve_code_predictor(resolver, hparams, weights.code_predictor)) {
        return SYNTH_ERR_GGUF;
    }
    // The codec is resolved twice when a twin context exists: once against the
    // package so the sweep below sees its names, and once against the twins,
    // which is what the graph binds to.
    if (!resolve_codec(resolver, hparams, weights.codec)) {
        return SYNTH_ERR_GGUF;
    }
    if (codec_context != nullptr) {
        Resolver twins(codec_context, hparams);
        if (!resolve_codec(twins, hparams, weights.codec)) {
            return SYNTH_ERR_GGUF;
        }
    }

    // A tensor nobody looked up is a tensor nobody checked, so the package is
    // swept rather than trusted. This also catches a catalog that resolved the
    // same name twice and left another unread.
    for (ggml_tensor * tensor = ggml_get_first_tensor(context); tensor != nullptr;
         tensor               = ggml_get_next_tensor(context, tensor)) {
        if (resolver.resolved().count(tensor->name) == 0) {
            std::fprintf(stderr, "qwen3-tts: tensor %s is outside the catalog\n", tensor->name);
            return SYNTH_ERR_GGUF;
        }
    }
    return SYNTH_OK;
}

}  // namespace synth::qwen3tts
