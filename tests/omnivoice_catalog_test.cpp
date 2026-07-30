// The OmniVoice tensor catalog is resolved from a small synthetic package.
//
// The entry list below is a second, independent statement of the contract: it is
// written out by hand from the checkpoint's layout, while the catalog derives
// the same names and shapes from the hyper-parameters. Agreeing is the point.
// The real 798-tensor package is exercised by the integration tier.
//
// Most of the value here is in the rejections. A package whose metadata and
// tensors disagree must be refused at load: past that point the mistake becomes
// wrong audio rather than an error.

#include "arch/omnivoice/catalog.h"
#include "arch/omnivoice/weights.h"
#include "ggml.h"
#include "test-assert.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

struct ContextDeleter {
    void operator()(ggml_context * context) const {
        if (context != nullptr) {
            ggml_free(context);
        }
    }
};

using Context = std::unique_ptr<ggml_context, ContextDeleter>;

Context make_context() {
    ggml_init_params parameters{};
    parameters.mem_size = ggml_tensor_overhead() * 4096;
    parameters.no_alloc = true;
    return Context(ggml_init(parameters));
}

struct Entry {
    std::string          name;
    std::vector<int64_t> ne;
};

void add(std::vector<Entry> & out, const std::string & name, std::vector<int64_t> ne) {
    out.push_back(Entry{ name, std::move(ne) });
}

void add_linear(std::vector<Entry> & out, const std::string & prefix, int64_t in, int64_t o) {
    add(out, prefix + ".weight", { in, o });
    add(out, prefix + ".bias", { o });
}

void add_conv(std::vector<Entry> & out, const std::string & prefix, int64_t k, int64_t in, int64_t o) {
    add(out, prefix + ".weight", { k, in, o });
    add(out, prefix + ".bias", { o });
}

// The HuBERT feature extractor and the codec's semantic encoder store their
// convolutions without a bias.
void add_bare_conv(std::vector<Entry> & out, const std::string & prefix, int64_t k, int64_t in, int64_t o) {
    add(out, prefix + ".weight", { k, in, o });
}

// ConvTranspose1d stores [in, out, kernel], so the trailing pair is reversed
// against a plain convolution's.
void add_transpose_conv(std::vector<Entry> & out, const std::string & prefix, int64_t k, int64_t in, int64_t o) {
    add(out, prefix + ".weight", { k, o, in });
    add(out, prefix + ".bias", { o });
}

void add_layer_norm(std::vector<Entry> & out, const std::string & prefix, int64_t c) {
    add(out, prefix + ".weight", { c });
    add(out, prefix + ".bias", { c });
}

// Plain Snake: one curve, stored channel-wise inside a [1, C, 1] tensor.
void add_snake(std::vector<Entry> & out, const std::string & prefix, int64_t c) {
    add(out, prefix + ".alpha", { 1, c, 1 });
}

void add_residual_units(std::vector<Entry> & out, const std::string & base, int64_t width) {
    for (int unit = 1; unit <= 3; ++unit) {
        const std::string prefix = base + "res_unit" + std::to_string(unit) + ".";
        add_snake(out, prefix + "snake1", width);
        add_conv(out, prefix + "conv1", 7, width, width);
        add_snake(out, prefix + "snake2", width);
        add_conv(out, prefix + "conv2", 1, width, width);
    }
}

// Structurally faithful at reduced widths: two generator layers, two codebooks,
// two acoustic blocks at ratios {2, 3}, two HuBERT layers and three feature
// convolutions. Every cross-check the metadata reader makes still holds -- the
// ratios multiply to the hop, 150 Hz over hop 6 is the 25 Hz frame rate, the
// mask id is the last canvas slot, and the canvas vocabulary is one past the
// codebook size -- so this package is a legal one, only small.
synth::omnivoice::HParams small_hparams() {
    synth::omnivoice::HParams h;
    h.model_variant        = "synthetic";
    h.quantization_profile = synth::omnivoice::QuantizationProfile::F32;
    h.output_sample_rate   = 150;
    h.output_channel_count = 1;

    h.generator.layer_count          = 2;
    h.generator.hidden_size          = 8;
    h.generator.attention_head_count = 4;
    h.generator.key_value_head_count = 2;
    h.generator.head_dim             = 4;
    h.generator.intermediate_size    = 16;
    h.generator.text_vocab_size      = 40;
    h.generator.rms_norm_eps         = 1e-6f;
    h.generator.rope_theta           = 10000.0f;

    h.audio.num_codebooks = 2;
    h.audio.vocab_size    = 5;
    h.audio.mask_id       = 4;

    h.codec.sample_rate          = 150;
    h.codec.hop_length           = 6;
    h.codec.frame_rate_hz        = 25.0f;
    h.codec.decoder_hidden_size  = 8;
    h.codec.encoder_hidden_size  = 2;
    h.codec.hidden_size          = 4;
    h.codec.codebook_dim         = 3;
    h.codec.codebook_size        = 4;
    h.codec.semantic_sample_rate = 100;
    h.codec.upsampling_ratios    = { 2, 3 };

    // The positional convolution is grouped by sixteen, so the semantic width
    // has to stay a multiple of that for the package to be representable at all.
    h.semantic.hidden_size          = 32;
    h.semantic.layer_count          = 2;
    h.semantic.attention_head_count = 4;
    h.semantic.intermediate_size    = 12;
    h.semantic.layer_norm_eps       = 1e-5f;
    h.semantic.conv_dim             = { 6, 6, 6 };
    h.semantic.conv_kernel          = { 5, 3, 2 };
    h.semantic.conv_stride          = { 2, 2, 2 };
    return h;
}

std::vector<Entry> expected_entries(const synth::omnivoice::HParams & h) {
    std::vector<Entry> out;

    const int64_t hidden = h.generator.hidden_size;
    const int64_t attn   = int64_t(h.generator.attention_head_count) * h.generator.head_dim;
    const int64_t kv     = int64_t(h.generator.key_value_head_count) * h.generator.head_dim;
    add(out, "llm.embed_tokens.weight", { hidden, h.generator.text_vocab_size });
    for (uint32_t layer = 0; layer < h.generator.layer_count; ++layer) {
        const std::string base = "llm.layers." + std::to_string(layer) + ".";
        add(out, base + "input_layernorm.weight", { hidden });
        add(out, base + "self_attn.q_proj.weight", { hidden, attn });
        add(out, base + "self_attn.k_proj.weight", { hidden, kv });
        add(out, base + "self_attn.v_proj.weight", { hidden, kv });
        add(out, base + "self_attn.o_proj.weight", { attn, hidden });
        add(out, base + "self_attn.q_norm.weight", { h.generator.head_dim });
        add(out, base + "self_attn.k_norm.weight", { h.generator.head_dim });
        add(out, base + "post_attention_layernorm.weight", { hidden });
        add(out, base + "mlp.gate_proj.weight", { hidden, h.generator.intermediate_size });
        add(out, base + "mlp.up_proj.weight", { hidden, h.generator.intermediate_size });
        add(out, base + "mlp.down_proj.weight", { h.generator.intermediate_size, hidden });
    }
    add(out, "llm.norm.weight", { hidden });
    const int64_t canvas = int64_t(h.audio.num_codebooks) * h.audio.vocab_size;
    add(out, "audio_embeddings.weight", { hidden, canvas });
    add(out, "audio_heads.weight", { hidden, canvas });

    // The quantizer runs on the concatenation of the acoustic and the semantic
    // latents, which is the codec's own top-level width.
    const int64_t concat = int64_t(h.codec.hidden_size) + h.semantic.hidden_size;
    for (uint32_t index = 0; index < h.audio.num_codebooks; ++index) {
        const std::string base = "codec.quantizer.quantizers." + std::to_string(index) + ".";
        add_linear(out, base + "project_in", concat, h.codec.codebook_dim);
        add_linear(out, base + "project_out", h.codec.codebook_dim, concat);
        add(out, base + "codebook.embed", { h.codec.codebook_dim, h.codec.codebook_size });
    }
    add_linear(out, "codec.fc", concat, concat);
    add_linear(out, "codec.fc2", concat, h.codec.hidden_size);

    add_conv(out, "codec.acoustic_decoder.conv1", 7, h.codec.hidden_size, h.codec.decoder_hidden_size);
    int64_t width = h.codec.decoder_hidden_size;
    for (size_t index = 0; index < h.codec.upsampling_ratios.size(); ++index) {
        const std::string base     = "codec.acoustic_decoder.block." + std::to_string(index) + ".";
        const int64_t     narrower = width / 2;
        add_snake(out, base + "snake1", width);
        add_transpose_conv(out, base + "conv_t1", 2 * int64_t(h.codec.upsampling_ratios[index]), width, narrower);
        add_residual_units(out, base, narrower);
        width = narrower;
    }
    add_snake(out, "codec.acoustic_decoder.snake1", width);
    add_conv(out, "codec.acoustic_decoder.conv2", 7, width, 1);

    add_conv(out, "codec.acoustic_encoder.conv1", 7, 1, h.codec.encoder_hidden_size);
    width = h.codec.encoder_hidden_size;
    for (size_t index = 0; index < h.codec.upsampling_ratios.size(); ++index) {
        const std::string base  = "codec.acoustic_encoder.block." + std::to_string(index) + ".";
        const int64_t     wider = width * 2;
        add_residual_units(out, base, width);
        add_snake(out, base + "snake1", width);
        add_conv(out, base + "conv1", 2 * int64_t(h.codec.upsampling_ratios[index]), width, wider);
        width = wider;
    }
    add_snake(out, "codec.acoustic_encoder.snake1", width);
    add_conv(out, "codec.acoustic_encoder.conv2", 3, width, h.codec.hidden_size);

    const int64_t semantic = h.semantic.hidden_size;
    for (size_t index = 0; index < h.semantic.conv_dim.size(); ++index) {
        const int64_t in = index == 0 ? 1 : int64_t(h.semantic.conv_dim[index - 1]);
        add_bare_conv(out, "codec.semantic_model.feat_conv." + std::to_string(index) + ".conv",
                      h.semantic.conv_kernel[index], in, h.semantic.conv_dim[index]);
    }
    add_layer_norm(out, "codec.semantic_model.feat_conv.0.layer_norm", h.semantic.conv_dim[0]);
    const int64_t features = h.semantic.conv_dim.back();
    add_layer_norm(out, "codec.semantic_model.feature_projection.layer_norm", features);
    add_linear(out, "codec.semantic_model.feature_projection.projection", features, semantic);
    add_conv(out, "codec.semantic_model.encoder.pos_conv_embed.conv", 128, semantic / 16, semantic);
    for (uint32_t layer = 0; layer < h.semantic.layer_count; ++layer) {
        const std::string base = "codec.semantic_model.encoder.layers." + std::to_string(layer) + ".";
        add_linear(out, base + "attn.q_proj", semantic, semantic);
        add_linear(out, base + "attn.k_proj", semantic, semantic);
        add_linear(out, base + "attn.v_proj", semantic, semantic);
        add_linear(out, base + "attn.out_proj", semantic, semantic);
        add_layer_norm(out, base + "layer_norm", semantic);
        add_linear(out, base + "ff.inter_dense", semantic, h.semantic.intermediate_size);
        add_linear(out, base + "ff.output_dense", h.semantic.intermediate_size, semantic);
        add_layer_norm(out, base + "final_layer_norm", semantic);
    }
    add_layer_norm(out, "codec.semantic_model.encoder.layer_norm", semantic);

    add_bare_conv(out, "codec.encoder_semantic.conv", 3, semantic, semantic);
    for (size_t block = 0; block < 2; ++block) {
        const std::string base = "codec.encoder_semantic.conv_blocks." + std::to_string(block) + ".";
        for (size_t unit = 0; unit < 2; ++unit) {
            const std::string prefix = base + "res_units." + std::to_string(unit) + ".";
            add_bare_conv(out, prefix + "conv1", 3, semantic, semantic);
            add_bare_conv(out, prefix + "conv2", 1, semantic, semantic);
        }
        add_conv(out, base + "conv", 3, semantic, semantic);
    }
    return out;
}

// `mutate` may rewrite one entry's shape or type, or drop it by clearing `ne`.
using Mutation = std::function<bool(const Entry &, Entry &, ggml_type &)>;

void populate(ggml_context * context, const std::vector<Entry> & entries, const Mutation & mutate) {
    for (const Entry & entry : entries) {
        Entry     effective = entry;
        ggml_type type      = GGML_TYPE_F32;
        if (mutate && mutate(entry, effective, type)) {
            if (effective.ne.empty()) {
                continue;
            }
        }
        int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
        for (size_t axis = 0; axis < effective.ne.size() && axis < GGML_MAX_DIMS; ++axis) {
            ne[axis] = effective.ne[axis];
        }
        ggml_tensor * tensor =
            ggml_new_tensor(context, type, int(std::min<size_t>(effective.ne.size(), GGML_MAX_DIMS)), ne);
        ggml_set_name(tensor, effective.name.c_str());
    }
}

int check_resolution(const synth::omnivoice::HParams & h, const std::vector<Entry> & entries) {
    Context                        context = make_context();
    synth::omnivoice::ModelWeights weights;
    populate(context.get(), entries, nullptr);
    SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, weights) == SYNTH_OK);

    // The count the catalog resolves and the count derived by arithmetic are two
    // independent statements of the same package, so they must agree.
    SYNTH_TEST_CHECK(synth::omnivoice::expected_tensor_count(h) == entries.size());

    SYNTH_TEST_CHECK(weights.generator.layers.size() == h.generator.layer_count);
    SYNTH_TEST_CHECK(weights.generator.text_embedding != nullptr);
    SYNTH_TEST_CHECK(weights.generator.norm != nullptr);
    // Qwen3 normalizes each head at head_dim, not the packed projection.
    SYNTH_TEST_CHECK(weights.generator.layers[0].q_norm != nullptr);
    SYNTH_TEST_CHECK(weights.generator.layers[0].q_norm->ne[0] == h.generator.head_dim);
    // The canvas tables stack every codebook into one table.
    SYNTH_TEST_CHECK(weights.generator.audio_embeddings != nullptr);
    SYNTH_TEST_CHECK(weights.generator.audio_heads->ne[1] ==
                     int64_t(h.audio.num_codebooks) * int64_t(h.audio.vocab_size));

    SYNTH_TEST_CHECK(weights.quantizers.size() == h.audio.num_codebooks);
    SYNTH_TEST_CHECK(weights.quantizers[1].codebook != nullptr);
    SYNTH_TEST_CHECK(weights.quantizers[1].codebook->ne[0] == h.codec.codebook_dim);
    // fc2 brings the concatenated latent back down to the acoustic width.
    SYNTH_TEST_CHECK(weights.fc2.weight != nullptr);
    SYNTH_TEST_CHECK(weights.fc2.weight->ne[1] == h.codec.hidden_size);
    SYNTH_TEST_CHECK(weights.fc.weight->ne[0] == int64_t(h.codec.hidden_size) + h.semantic.hidden_size);

    SYNTH_TEST_CHECK(weights.acoustic_decoder.blocks.size() == h.codec.upsampling_ratios.size());
    SYNTH_TEST_CHECK(weights.acoustic_decoder.blocks[0].res_units.size() == 3);
    // Plain Snake, so there is an alpha and nothing beside it.
    SYNTH_TEST_CHECK(weights.acoustic_decoder.blocks[0].res_units[2].snake2.alpha != nullptr);
    // The decoder halves once per block, so the last width is the narrowest and
    // the exit convolution emits one channel.
    SYNTH_TEST_CHECK(weights.acoustic_decoder.snake1.alpha->ne[1] ==
                     int64_t(h.codec.decoder_hidden_size) >> h.codec.upsampling_ratios.size());
    SYNTH_TEST_CHECK(weights.acoustic_decoder.conv2.weight->ne[2] == 1);

    SYNTH_TEST_CHECK(weights.acoustic_encoder.blocks.size() == h.codec.upsampling_ratios.size());
    SYNTH_TEST_CHECK(weights.acoustic_encoder.conv1.weight->ne[1] == 1);
    SYNTH_TEST_CHECK(weights.acoustic_encoder.conv2.weight->ne[2] == h.codec.hidden_size);

    SYNTH_TEST_CHECK(weights.semantic_model.feat_conv.size() == h.semantic.conv_dim.size());
    // The feature extractor's convolutions carry no bias, and only the first is
    // normalized.
    SYNTH_TEST_CHECK(weights.semantic_model.feat_conv[0].bias == nullptr);
    SYNTH_TEST_CHECK(weights.semantic_model.feat_conv_norm.weight != nullptr);
    SYNTH_TEST_CHECK(weights.semantic_model.layers.size() == h.semantic.layer_count);
    SYNTH_TEST_CHECK(weights.semantic_model.layers[1].out_proj.bias != nullptr);
    // Grouped by sixteen, so each filter sees a sixteenth of the width.
    SYNTH_TEST_CHECK(weights.semantic_model.pos_conv.weight->ne[1] == int64_t(h.semantic.hidden_size) / 16);

    SYNTH_TEST_CHECK(weights.encoder_semantic.blocks.size() == 2);
    SYNTH_TEST_CHECK(weights.encoder_semantic.blocks[1].res_units.size() == 2);
    SYNTH_TEST_CHECK(weights.encoder_semantic.blocks[1].res_units[0].conv1.bias == nullptr);
    SYNTH_TEST_CHECK(weights.encoder_semantic.conv.bias == nullptr);
    return 0;
}

int check_rejections(const synth::omnivoice::HParams & h, const std::vector<Entry> & entries) {
    synth::omnivoice::ModelWeights weights;
    SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(nullptr, h, weights) == SYNTH_ERR_INVALID_ARG);

    // Any single missing entry is a package defect. One is taken from each
    // region so a whole region cannot go unresolved unnoticed.
    for (const std::string & missing :
         { std::string("llm.layers.1.self_attn.k_norm.weight"), std::string("audio_heads.weight"),
           std::string("codec.quantizer.quantizers.1.codebook.embed"), std::string("codec.fc2.bias"),
           std::string("codec.acoustic_decoder.block.1.res_unit3.conv2.bias"),
           std::string("codec.acoustic_encoder.block.0.snake1.alpha"),
           std::string("codec.semantic_model.feat_conv.0.layer_norm.bias"),
           std::string("codec.semantic_model.encoder.layers.1.ff.output_dense.weight"),
           std::string("codec.semantic_model.encoder.pos_conv_embed.conv.weight"),
           std::string("codec.encoder_semantic.conv_blocks.1.res_units.0.conv2.weight") }) {
        Context                        context = make_context();
        synth::omnivoice::ModelWeights parsed;
        populate(context.get(), entries, [&](const Entry & entry, Entry & effective, ggml_type &) {
            if (entry.name == missing) {
                effective.ne.clear();
                return true;
            }
            return false;
        });
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, parsed) == SYNTH_ERR_GGUF);
    }

    // A shape that disagrees with the declared hyper-parameters.
    for (const std::string & wrong :
         { std::string("llm.embed_tokens.weight"), std::string("audio_embeddings.weight"),
           std::string("codec.quantizer.quantizers.0.project_in.weight"), std::string("codec.fc.weight"),
           std::string("codec.acoustic_decoder.conv1.weight"),
           std::string("codec.acoustic_encoder.block.1.conv1.weight"),
           std::string("codec.semantic_model.feat_conv.1.conv.weight"),
           std::string("codec.encoder_semantic.conv.weight") }) {
        Context                        context = make_context();
        synth::omnivoice::ModelWeights parsed;
        populate(context.get(), entries, [&](const Entry & entry, Entry & effective, ggml_type &) {
            if (entry.name == wrong) {
                effective.ne[0] += 1;
                return true;
            }
            return false;
        });
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, parsed) == SYNTH_ERR_GGUF);
    }

    // A transposed convolution stored in a plain convolution's order. The two
    // differ only in the order of the trailing pair, so a package built by a
    // converter that confused them still has the right element count.
    {
        Context                        context = make_context();
        synth::omnivoice::ModelWeights parsed;
        populate(context.get(), entries, [](const Entry & entry, Entry & effective, ggml_type &) {
            if (entry.name == "codec.acoustic_decoder.block.0.conv_t1.weight") {
                std::swap(effective.ne[1], effective.ne[2]);
                return true;
            }
            return false;
        });
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, parsed) == SYNTH_ERR_GGUF);
    }

    // This profile carries the checkpoint through unchanged, so every tensor is
    // F32 and anything halved is a package defect.
    for (const std::string & recast :
         { std::string("llm.norm.weight"), std::string("codec.acoustic_decoder.block.0.res_unit1.conv1.weight") }) {
        Context                        context = make_context();
        synth::omnivoice::ModelWeights parsed;
        populate(context.get(), entries, [&](const Entry & entry, Entry &, ggml_type & type) {
            if (entry.name == recast) {
                type = GGML_TYPE_F16;
                return true;
            }
            return false;
        });
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, parsed) == SYNTH_ERR_GGUF);
    }

    // A trailing axis the catalog does not expect, so a silently reshaped tensor
    // cannot slip through with the right element count.
    {
        Context                        context = make_context();
        synth::omnivoice::ModelWeights parsed;
        populate(context.get(), entries, [](const Entry & entry, Entry & effective, ggml_type &) {
            if (entry.name == "llm.norm.weight") {
                effective.ne.push_back(2);
                return true;
            }
            return false;
        });
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, parsed) == SYNTH_ERR_GGUF);
    }

    // A tensor nobody looks up is a tensor nobody checks. This is what would
    // otherwise let a skipped training buffer -- an EMA table, the semantic
    // decoder -- ride along unread.
    {
        Context                        context = make_context();
        synth::omnivoice::ModelWeights parsed;
        populate(context.get(), entries, nullptr);
        ggml_tensor * stray = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 4);
        ggml_set_name(stray, "codec.decoder_semantic.conv1.weight");
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, parsed) == SYNTH_ERR_GGUF);
    }

    // The quantizer's width is the acoustic and semantic widths added together
    // rather than a number the package states, so a semantic width that moved
    // is refused even though every name is still present.
    {
        Context                        context = make_context();
        synth::omnivoice::HParams      other   = h;
        synth::omnivoice::ModelWeights parsed;
        other.semantic.hidden_size = h.semantic.hidden_size + 16;
        populate(context.get(), entries, nullptr);
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), other, parsed) == SYNTH_ERR_GGUF);
    }

    // Hyper-parameters the resolver would have to index past: refused rather
    // than read out of bounds.
    {
        Context                        context = make_context();
        synth::omnivoice::HParams      other   = h;
        synth::omnivoice::ModelWeights parsed;
        other.semantic.conv_kernel.pop_back();
        populate(context.get(), entries, nullptr);
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), other, parsed) == SYNTH_ERR_GGUF);
    }
    return 0;
}

// The real package's shape, to catch an arithmetic change that the synthetic
// package is too small to notice. 798 is the emitted tensor count in
// reports/convert/omnivoice/omnivoice-0-6b-F32.json: 312 generator plus 486
// codec, after the converter's declared skips.
int check_real_package_count() {
    synth::omnivoice::HParams h;
    h.generator.layer_count   = 28;
    h.audio.num_codebooks     = 8;
    h.codec.upsampling_ratios = { 8, 5, 4, 2, 3 };
    h.semantic.layer_count    = 12;
    h.semantic.conv_dim       = { 512, 512, 512, 512, 512, 512, 512 };
    SYNTH_TEST_CHECK(synth::omnivoice::expected_tensor_count(h) == 798);
    return 0;
}

}  // namespace

int main() {
    const synth::omnivoice::HParams h       = small_hparams();
    const std::vector<Entry>        entries = expected_entries(h);

    SYNTH_TEST_CHECK(check_resolution(h, entries) == 0);
    SYNTH_TEST_CHECK(check_rejections(h, entries) == 0);
    SYNTH_TEST_CHECK(check_real_package_count() == 0);
    return 0;
}
