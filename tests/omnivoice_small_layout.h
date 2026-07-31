#pragma once

// The small OmniVoice package layout shared by the catalog test and the
// synthetic on-disk package builder: structurally faithful at reduced widths,
// so both consumers describe the exact same package rather than two
// independent approximations of one.

#include "arch/omnivoice/weights.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace synth::omnivoice::testing {

struct Entry {
    std::string          name;
    std::vector<int64_t> ne;
};

inline void add(std::vector<Entry> & out, const std::string & name, std::vector<int64_t> ne) {
    out.push_back(Entry{ name, std::move(ne) });
}

inline void add_linear(std::vector<Entry> & out, const std::string & prefix, int64_t in, int64_t o) {
    add(out, prefix + ".weight", { in, o });
    add(out, prefix + ".bias", { o });
}

inline void add_conv(std::vector<Entry> & out, const std::string & prefix, int64_t k, int64_t in, int64_t o) {
    add(out, prefix + ".weight", { k, in, o });
    add(out, prefix + ".bias", { o });
}

// The HuBERT feature extractor and the codec's semantic encoder store their
// convolutions without a bias.
inline void add_bare_conv(std::vector<Entry> & out, const std::string & prefix, int64_t k, int64_t in, int64_t o) {
    add(out, prefix + ".weight", { k, in, o });
}

// ConvTranspose1d stores [in, out, kernel], so the trailing pair is reversed
// against a plain convolution's.
inline void add_transpose_conv(std::vector<Entry> & out, const std::string & prefix, int64_t k, int64_t in,
                                int64_t o) {
    add(out, prefix + ".weight", { k, o, in });
    add(out, prefix + ".bias", { o });
}

inline void add_layer_norm(std::vector<Entry> & out, const std::string & prefix, int64_t c) {
    add(out, prefix + ".weight", { c });
    add(out, prefix + ".bias", { c });
}

// Plain Snake: one curve, stored channel-wise inside a [1, C, 1] tensor.
inline void add_snake(std::vector<Entry> & out, const std::string & prefix, int64_t c) {
    add(out, prefix + ".alpha", { 1, c, 1 });
}

inline void add_residual_units(std::vector<Entry> & out, const std::string & base, int64_t width) {
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
inline synth::omnivoice::HParams small_hparams() {
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

inline std::vector<Entry> expected_entries(const synth::omnivoice::HParams & h) {
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

}  // namespace synth::omnivoice::testing
