// OmniVoice's tensor->role classifier for quantization.
//
// This classifier is the single source of truth shared by the offline
// quantizer and the runtime's catalog, so what it decides is a package
// contract, not an implementation detail. Real tensor names and shapes below
// are read out of src/arch/omnivoice/catalog.cpp's own registration (cross-
// checked against reports/porting/omnivoice/omnivoice-0-6b/tensor-inventory.json,
// the real checkpoint's tensor shapes) rather than invented, so a mismatch
// between this test and the catalog's actual layout would show up as a
// spurious failure here rather than a name this classifier silently mishandles.
//
// Two things beyond the per-tensor cases matter more than any single one of
// them: the completeness case proves every tensor the real catalog can ever
// register resolves to something other than Unknown, and the count case
// proves each profile's headline "N tensors quantized" number is what the
// catalog actually contains rather than a number nobody re-derived.

#include "arch/omnivoice/catalog.h"
#include "arch/omnivoice/quantization.h"
#include "arch/omnivoice/weights.h"
#include "omnivoice_small_layout.h"
#include "test-assert.h"

#include <cstdint>
#include <string>
#include <vector>

using synth::omnivoice::classify_tensor;
using synth::omnivoice::classify_tensor_for_half;
using synth::omnivoice::ModelHalf;
using synth::omnivoice::QuantRole;
using synth::omnivoice::tensor_half;
using synth::omnivoice::testing::Entry;

namespace {

void fill_ne(const std::vector<int64_t> & shape, int64_t ne[4]) {
    for (size_t axis = 0; axis < 4; ++axis) {
        ne[axis] = axis < shape.size() ? shape[axis] : 1;
    }
}

QuantRole classify(const std::string & name, std::vector<int64_t> shape) {
    int64_t ne[4] = { 1, 1, 1, 1 };
    fill_ne(shape, ne);
    return classify_tensor(name, ne);
}

QuantRole classify_for(ModelHalf half, const std::string & name, std::vector<int64_t> shape) {
    int64_t ne[4] = { 1, 1, 1, 1 };
    fill_ne(shape, ne);
    return classify_tensor_for_half(name, ne, half);
}

uint64_t count_role(const std::vector<Entry> & entries, QuantRole role) {
    uint64_t count = 0;
    for (const Entry & entry : entries) {
        if (classify(entry.name, entry.ne) == role) {
            ++count;
        }
    }
    return count;
}

uint64_t count_role_for(const std::vector<Entry> & entries, ModelHalf half, QuantRole role) {
    uint64_t count = 0;
    for (const Entry & entry : entries) {
        if (classify_for(half, entry.name, entry.ne) == role) {
            ++count;
        }
    }
    return count;
}

std::vector<Entry> filter_prefix(const std::vector<Entry> & entries, const std::string & prefix) {
    std::vector<Entry> out;
    for (const Entry & entry : entries) {
        if (entry.name.compare(0, prefix.size(), prefix) == 0) {
            out.push_back(entry);
        }
    }
    return out;
}

// The real checkpoint's topology and widths, read out of
// reports/porting/omnivoice/omnivoice-0-6b/tensor-inventory.json: 28 generator
// layers, 8 codebooks, five upsampling blocks at ratios {8,5,4,2,3}, 12 HuBERT
// layers over 7 feature convolutions. No classification here reads a width --
// every role follows from the name -- but the counts below are per-layer and
// per-block totals, so the topology has to be the real one.
synth::omnivoice::HParams real_hparams() {
    synth::omnivoice::HParams h;
    h.model_variant        = "omnivoice-0-6b";
    h.quantization_profile = synth::omnivoice::QuantizationProfile::F32;

    h.generator.layer_count          = 28;
    h.generator.hidden_size          = 1024;
    h.generator.attention_head_count = 16;
    h.generator.key_value_head_count = 8;
    h.generator.head_dim             = 128;
    h.generator.intermediate_size    = 3072;
    h.generator.text_vocab_size      = 151676;
    h.generator.rms_norm_eps         = 1e-6f;
    h.generator.rope_theta           = 1000000.0f;

    h.audio.num_codebooks = 8;
    h.audio.vocab_size    = 1025;
    h.audio.mask_id       = 1024;

    h.codec.sample_rate          = 16000;
    h.codec.hop_length           = 960;
    h.codec.frame_rate_hz        = 16.6667f;
    h.codec.decoder_hidden_size  = 1024;
    h.codec.encoder_hidden_size  = 64;
    h.codec.hidden_size          = 256;
    h.codec.codebook_dim         = 64;
    h.codec.codebook_size        = 1024;
    h.codec.semantic_sample_rate = 16000;
    h.codec.upsampling_ratios    = { 8, 5, 4, 2, 3 };

    h.semantic.hidden_size          = 768;
    h.semantic.layer_count          = 12;
    h.semantic.attention_head_count = 12;
    h.semantic.intermediate_size    = 3072;
    h.semantic.layer_norm_eps       = 1e-5f;
    h.semantic.conv_dim             = { 512, 512, 512, 512, 512, 512, 512 };
    h.semantic.conv_kernel          = { 10, 3, 3, 3, 3, 2, 2 };
    h.semantic.conv_stride          = { 5, 2, 2, 2, 2, 2, 2 };
    return h;
}

// Every group the generator half contains, at the real checkpoint's shapes
// (tensor-inventory.json). These are the *architectural* roles: what a
// profile that quantizes this half would do with each tensor. What a
// codec-half profile does with them is check_the_half_split's question.
int check_generator_groups() {
    // The seven projections per layer: four attention, three MLP. Every one
    // is a plain ggml_mul_mat operand, and every row (ne[0]) here is 1024,
    // 2048 or 3072 -- divisible by 32 and by 256, so Q8_0 and every k-quant
    // clear the offline quantizer's block-size check without an exception.
    SYNTH_TEST_CHECK(classify("llm.layers.0.self_attn.q_proj.weight", { 1024, 2048 }) == QuantRole::MatrixWeight);
    SYNTH_TEST_CHECK(classify("llm.layers.0.self_attn.k_proj.weight", { 1024, 1024 }) == QuantRole::MatrixWeight);
    SYNTH_TEST_CHECK(classify("llm.layers.0.self_attn.v_proj.weight", { 1024, 1024 }) == QuantRole::MatrixWeight);
    SYNTH_TEST_CHECK(classify("llm.layers.0.self_attn.o_proj.weight", { 2048, 1024 }) == QuantRole::MatrixWeight);
    SYNTH_TEST_CHECK(classify("llm.layers.27.mlp.gate_proj.weight", { 1024, 3072 }) == QuantRole::MatrixWeight);
    SYNTH_TEST_CHECK(classify("llm.layers.27.mlp.up_proj.weight", { 1024, 3072 }) == QuantRole::MatrixWeight);
    SYNTH_TEST_CHECK(classify("llm.layers.27.mlp.down_proj.weight", { 3072, 1024 }) == QuantRole::MatrixWeight);

    // Every norm in the half. Note the name: this family's converter emits
    // `post_attention_layernorm`, the Hugging Face name, where qwen3-tts's
    // emits `post_attn_norm`. A classifier copied across without changing it
    // would silently drop this tensor to Unknown.
    SYNTH_TEST_CHECK(classify("llm.layers.0.input_layernorm.weight", { 1024 }) == QuantRole::Sensitive);
    SYNTH_TEST_CHECK(classify("llm.layers.0.post_attention_layernorm.weight", { 1024 }) == QuantRole::Sensitive);
    SYNTH_TEST_CHECK(classify("llm.layers.0.post_attn_norm.weight", { 1024 }) == QuantRole::Unknown);
    // Per-head norms are head_dim wide -- 128 values against a projection's
    // two million -- and scale every head before rope.
    SYNTH_TEST_CHECK(classify("llm.layers.0.self_attn.q_norm.weight", { 128 }) == QuantRole::Sensitive);
    SYNTH_TEST_CHECK(classify("llm.layers.0.self_attn.k_norm.weight", { 128 }) == QuantRole::Sensitive);
    SYNTH_TEST_CHECK(classify("llm.norm.weight", { 1024 }) == QuantRole::Sensitive);

    // The two `ggml_get_rows` tables (generator.cpp:123, 136), and the third
    // tensor of the same shape that is *not* one of them: `audio_heads` is a
    // plain mul_mat (generator.cpp:193), not the tied transpose of
    // `audio_embeddings` it resembles.
    SYNTH_TEST_CHECK(classify("llm.embed_tokens.weight", { 1024, 151676 }) == QuantRole::RowLookup);
    SYNTH_TEST_CHECK(classify("audio_embeddings.weight", { 1024, 8200 }) == QuantRole::RowLookup);
    SYNTH_TEST_CHECK(classify("audio_heads.weight", { 1024, 8200 }) == QuantRole::MatrixWeight);
    return 0;
}

// A generator name the catalog cannot produce is Unknown, not quietly
// absorbed by a prefix rule. Each of these is one plausible edit away from a
// real name.
int check_generator_strays_are_unknown() {
    for (const std::string & name : {
             std::string("llm"),
             std::string("llm.weight"),
             std::string("llm.embed_tokens"),
             std::string("llm.embed_tokens.bias"),
             std::string("llm.norm.bias"),
             std::string("llm.layers.0.self_attn.q_proj.bias"),
             std::string("llm.layers.0.self_attn.qkv_proj.weight"),
             std::string("llm.layers.0.mlp.fc.weight"),
             std::string("llm.layers.x.mlp.up_proj.weight"),
             std::string("llm.layers.0.input_layernorm.bias"),
             std::string("llmx.norm.weight"),
             std::string("audio_embeddings.bias"),
             std::string("audio_headsx.weight"),
             // The tied text head that does not exist in this package: the
             // embedding is tied, so a converter emitting one would be a
             // change this classifier must not absorb.
             std::string("llm.lm_head.weight"),
         }) {
        SYNTH_TEST_CHECK(classify(name, {}) == QuantRole::Unknown);
    }
    return 0;
}

// The half split, which is what makes one classifier serve profiles that
// quantize opposite halves. Two properties matter and neither is obvious:
// a tensor outside the quantized half reports Sensitive whatever its
// architectural role, and an Unknown name stays Unknown even when it falls
// in the half a profile is leaving alone.
int check_the_half_split() {
    SYNTH_TEST_CHECK(tensor_half("llm.layers.0.self_attn.q_proj.weight") == ModelHalf::Generator);
    SYNTH_TEST_CHECK(tensor_half("audio_heads.weight") == ModelHalf::Generator);
    SYNTH_TEST_CHECK(tensor_half("codec.acoustic_decoder.conv1.weight") == ModelHalf::Codec);

    // Generator-half profile: the generator carries its architectural role,
    // the codec is held exact -- including the codec's own MatrixWeight and
    // TransposeWeight tensors.
    SYNTH_TEST_CHECK(classify_for(ModelHalf::Generator, "llm.layers.0.self_attn.q_proj.weight", { 1024, 2048 }) ==
                     QuantRole::MatrixWeight);
    SYNTH_TEST_CHECK(classify_for(ModelHalf::Generator, "llm.embed_tokens.weight", { 1024, 151676 }) ==
                     QuantRole::RowLookup);
    SYNTH_TEST_CHECK(classify_for(ModelHalf::Generator, "llm.norm.weight", { 1024 }) == QuantRole::Sensitive);
    SYNTH_TEST_CHECK(classify_for(ModelHalf::Generator, "codec.acoustic_decoder.conv1.weight", { 7, 256, 1024 }) ==
                     QuantRole::Sensitive);
    SYNTH_TEST_CHECK(classify_for(ModelHalf::Generator, "codec.semantic_model.encoder.layers.0.attn.q_proj.weight",
                                  { 768, 768 }) == QuantRole::Sensitive);
    SYNTH_TEST_CHECK(classify_for(ModelHalf::Generator, "codec.acoustic_decoder.block.0.conv_t1.weight",
                                  { 16, 512, 1024 }) == QuantRole::Sensitive);

    // Codec-half profile: the mirror image. The decoder's input convolution
    // is ConvKernel since the conv-exempt policy; a HuBERT Linear is what
    // MatrixWeight now means on this side.
    SYNTH_TEST_CHECK(classify_for(ModelHalf::Codec, "codec.acoustic_decoder.conv1.weight", { 7, 256, 1024 }) ==
                     QuantRole::ConvKernel);
    SYNTH_TEST_CHECK(classify_for(ModelHalf::Codec, "codec.semantic_model.encoder.layers.0.attn.q_proj.weight",
                                  { 768, 768 }) == QuantRole::MatrixWeight);
    SYNTH_TEST_CHECK(classify_for(ModelHalf::Codec, "codec.acoustic_decoder.block.0.conv_t1.weight",
                                  { 16, 512, 1024 }) == QuantRole::TransposeWeight);
    SYNTH_TEST_CHECK(classify_for(ModelHalf::Codec, "llm.layers.0.self_attn.q_proj.weight", { 1024, 2048 }) ==
                     QuantRole::Sensitive);
    SYNTH_TEST_CHECK(classify_for(ModelHalf::Codec, "llm.embed_tokens.weight", { 1024, 151676 }) ==
                     QuantRole::Sensitive);
    SYNTH_TEST_CHECK(classify_for(ModelHalf::Codec, "audio_heads.weight", { 1024, 8200 }) == QuantRole::Sensitive);

    // Unknown survives the half test from both sides. Without this, a stray
    // generator name would be silently reported as Sensitive under a
    // codec-half profile and the quantizer's fail-by-name would never fire.
    SYNTH_TEST_CHECK(classify_for(ModelHalf::Codec, "llm.layers.0.mlp.fc.weight", { 1024, 3072 }) ==
                     QuantRole::Unknown);
    SYNTH_TEST_CHECK(classify_for(ModelHalf::Generator, "codec.unknown_module.conv.weight", { 3, 8, 8 }) ==
                     QuantRole::Unknown);
    SYNTH_TEST_CHECK(classify_for(ModelHalf::Codec, "voice.af_heart", {}) == QuantRole::Unknown);
    SYNTH_TEST_CHECK(classify_for(ModelHalf::Generator, "voice.af_heart", {}) == QuantRole::Unknown);
    return 0;
}

int check_rvq_and_concat_projections_are_sensitive() {
    // codec.quantizer.*: matches qwen3-tts's own RVQ rule (policy.cpp:258-265).
    SYNTH_TEST_CHECK(classify("codec.quantizer.quantizers.0.project_in.weight", { 1024, 64 }) == QuantRole::Sensitive);
    SYNTH_TEST_CHECK(classify("codec.quantizer.quantizers.0.project_in.bias", { 64 }) == QuantRole::Sensitive);
    SYNTH_TEST_CHECK(classify("codec.quantizer.quantizers.0.project_out.weight", { 64, 1024 }) == QuantRole::Sensitive);
    SYNTH_TEST_CHECK(classify("codec.quantizer.quantizers.0.project_out.bias", { 1024 }) == QuantRole::Sensitive);
    SYNTH_TEST_CHECK(classify("codec.quantizer.quantizers.7.codebook.embed", { 64, 1024 }) == QuantRole::Sensitive);

    // codec.fc / codec.fc2: the concatenation projection and its inverse.
    SYNTH_TEST_CHECK(classify("codec.fc.weight", { 1024, 1024 }) == QuantRole::Sensitive);
    SYNTH_TEST_CHECK(classify("codec.fc.bias", { 1024 }) == QuantRole::Sensitive);
    SYNTH_TEST_CHECK(classify("codec.fc2.weight", { 1024, 256 }) == QuantRole::Sensitive);
    SYNTH_TEST_CHECK(classify("codec.fc2.bias", { 256 }) == QuantRole::Sensitive);

    // "fc2" must not be reached by a "fc" prefix that stops short of the full
    // token, nor the reverse -- a substring classifier would conflate them.
    SYNTH_TEST_CHECK(classify("codec.fc21.weight", { 1024, 1024 }) == QuantRole::Unknown);
    SYNTH_TEST_CHECK(classify("codec.fcx.weight", { 1024, 1024 }) == QuantRole::Unknown);
    return 0;
}

int check_transpose_weight_override() {
    // codec.acoustic_decoder.block.<i>.conv_t1.weight: the standing
    // transposed-convolution override (policy.cpp:390-395). Real shape at
    // block 0 (ratio 8, so kernel 16, width 1024 narrowing to 512).
    SYNTH_TEST_CHECK(classify("codec.acoustic_decoder.block.0.conv_t1.weight", { 16, 512, 1024 }) ==
                     QuantRole::TransposeWeight);
    SYNTH_TEST_CHECK(classify("codec.acoustic_decoder.block.4.conv_t1.weight", { 6, 32, 64 }) ==
                     QuantRole::TransposeWeight);
    // Its bias is still a bias -- Sensitive, not TransposeWeight.
    SYNTH_TEST_CHECK(classify("codec.acoustic_decoder.block.0.conv_t1.bias", { 512 }) == QuantRole::Sensitive);

    // The encoder has no conv_t1 at all -- its resampling convolution is a
    // regular strided one named "conv1", not "conv_t1". A classifier that
    // matched "acoustic_decoder" as a substring prefix of "acoustic_encoder"
    // (or vice versa), or that matched "conv1" against the "conv_t1" pattern,
    // would misfile this as a decoder tensor or a transpose weight.
    SYNTH_TEST_CHECK(classify("codec.acoustic_encoder.block.0.conv1.weight", { 16, 64, 128 }) == QuantRole::ConvKernel);
    return 0;
}

int check_structural_exceptions() {
    // codec.acoustic_encoder.conv1.weight: reads the raw mono waveform, so its
    // packed row is kernel * in = 7 * 1 = 7 -- real shape from the checkpoint.
    SYNTH_TEST_CHECK(classify("codec.acoustic_encoder.conv1.weight", { 7, 1, 64 }) == QuantRole::Sensitive);

    // codec.semantic_model.feat_conv.0.conv.weight: the HuBERT feature
    // extractor's own raw-waveform convolution, packed row 10 * 1 = 10.
    SYNTH_TEST_CHECK(classify("codec.semantic_model.feat_conv.0.conv.weight", { 10, 1, 512 }) == QuantRole::Sensitive);
    // Every later feature convolution reads the previous layer's 512-wide
    // output, not the raw waveform, so only index 0 is a named exception --
    // and the difference that exception now makes is F32 against ConvKernel's
    // F16, not exact against block-quantized.
    SYNTH_TEST_CHECK(classify("codec.semantic_model.feat_conv.1.conv.weight", { 3, 512, 512 }) ==
                     QuantRole::ConvKernel);
    SYNTH_TEST_CHECK(classify("codec.semantic_model.feat_conv.6.conv.weight", { 2, 512, 512 }) ==
                     QuantRole::ConvKernel);

    // codec.semantic_model.encoder.pos_conv_embed.conv.weight: grouped by
    // sixteen and sliced per group by ggml_view_3d in reference-encoder.cpp's
    // grouped_conv1d. Its packed row (128 * 48 = 6144) is itself divisible by
    // 32, proving the exception is about the grouped view, not the size.
    SYNTH_TEST_CHECK(classify("codec.semantic_model.encoder.pos_conv_embed.conv.weight", { 128, 48, 768 }) ==
                     QuantRole::Sensitive);
    SYNTH_TEST_CHECK(classify("codec.semantic_model.encoder.pos_conv_embed.conv.bias", { 768 }) ==
                     QuantRole::Sensitive);
    return 0;
}

int check_snake_alphas_and_biases_are_sensitive() {
    SYNTH_TEST_CHECK(classify("codec.acoustic_decoder.block.0.res_unit1.snake1.alpha", { 1, 512, 1 }) ==
                     QuantRole::Sensitive);
    SYNTH_TEST_CHECK(classify("codec.acoustic_decoder.snake1.alpha", { 1, 32, 1 }) == QuantRole::Sensitive);
    SYNTH_TEST_CHECK(classify("codec.acoustic_encoder.snake1.alpha", { 1, 2048, 1 }) == QuantRole::Sensitive);

    SYNTH_TEST_CHECK(classify("codec.acoustic_decoder.conv1.bias", { 1024 }) == QuantRole::Sensitive);
    SYNTH_TEST_CHECK(classify("codec.acoustic_encoder.block.0.conv1.bias", { 128 }) == QuantRole::Sensitive);
    SYNTH_TEST_CHECK(classify("codec.semantic_model.encoder.layers.0.attn.q_proj.bias", { 768 }) ==
                     QuantRole::Sensitive);
    SYNTH_TEST_CHECK(classify("codec.encoder_semantic.conv_blocks.0.conv.bias", { 768 }) == QuantRole::Sensitive);
    return 0;
}

int check_one_dimensional_norms_are_sensitive() {
    // All five norm sites in the codec, by name -- the feature group norm,
    // the feature projection's norm, each encoder layer's two norms, and the
    // encoder's own exit norm.
    SYNTH_TEST_CHECK(classify("codec.semantic_model.feat_conv.0.layer_norm.weight", { 512 }) == QuantRole::Sensitive);
    SYNTH_TEST_CHECK(classify("codec.semantic_model.feat_conv.0.layer_norm.bias", { 512 }) == QuantRole::Sensitive);
    SYNTH_TEST_CHECK(classify("codec.semantic_model.feature_projection.layer_norm.weight", { 512 }) ==
                     QuantRole::Sensitive);
    SYNTH_TEST_CHECK(classify("codec.semantic_model.encoder.layers.0.layer_norm.weight", { 768 }) ==
                     QuantRole::Sensitive);
    SYNTH_TEST_CHECK(classify("codec.semantic_model.encoder.layers.0.final_layer_norm.weight", { 768 }) ==
                     QuantRole::Sensitive);
    SYNTH_TEST_CHECK(classify("codec.semantic_model.encoder.layer_norm.weight", { 768 }) == QuantRole::Sensitive);
    // Its neighbor, the feature projection's matrix, is not a norm and is not
    // named "layer_norm" -- it must still resolve to MatrixWeight.
    SYNTH_TEST_CHECK(classify("codec.semantic_model.feature_projection.projection.weight", { 512, 768 }) ==
                     QuantRole::MatrixWeight);
    return 0;
}

// The conv-exempt split, which is what the codec half's policy now IS: a
// convolution kernel is ConvKernel and is never block-quantized, and the only
// MatrixWeight tensors left in the codec are the HuBERT Linears.
int check_the_conv_exempt_split() {
    // codec.acoustic_decoder. -- the path every synthesis runs, cloning or
    // not, and under this policy none of it is ever quantized.
    SYNTH_TEST_CHECK(classify("codec.acoustic_decoder.conv1.weight", { 7, 256, 1024 }) == QuantRole::ConvKernel);
    SYNTH_TEST_CHECK(classify("codec.acoustic_decoder.block.0.res_unit1.conv1.weight", { 7, 512, 512 }) ==
                     QuantRole::ConvKernel);
    SYNTH_TEST_CHECK(classify("codec.acoustic_decoder.block.0.res_unit1.conv2.weight", { 1, 512, 512 }) ==
                     QuantRole::ConvKernel);
    // The mono exit convolution at ne = [7, 32, 1] -- the tensor a
    // shape-based conv-exempt rule would get wrong, because `ggml_n_dims`
    // collapses its trailing unit axis and reports it as two-dimensional,
    // exactly like a Linear. It is ConvKernel because the classifier reads
    // the name, never the shape.
    SYNTH_TEST_CHECK(classify("codec.acoustic_decoder.conv2.weight", { 7, 32, 1 }) == QuantRole::ConvKernel);

    // codec.acoustic_encoder.
    SYNTH_TEST_CHECK(classify("codec.acoustic_encoder.block.0.res_unit1.conv1.weight", { 7, 64, 64 }) ==
                     QuantRole::ConvKernel);
    SYNTH_TEST_CHECK(classify("codec.acoustic_encoder.conv2.weight", { 3, 2048, 256 }) == QuantRole::ConvKernel);

    // codec.semantic_model: the seven Linear module names, and nothing else.
    SYNTH_TEST_CHECK(classify("codec.semantic_model.encoder.layers.0.attn.q_proj.weight", { 768, 768 }) ==
                     QuantRole::MatrixWeight);
    SYNTH_TEST_CHECK(classify("codec.semantic_model.encoder.layers.0.attn.k_proj.weight", { 768, 768 }) ==
                     QuantRole::MatrixWeight);
    SYNTH_TEST_CHECK(classify("codec.semantic_model.encoder.layers.0.attn.v_proj.weight", { 768, 768 }) ==
                     QuantRole::MatrixWeight);
    SYNTH_TEST_CHECK(classify("codec.semantic_model.encoder.layers.11.attn.out_proj.weight", { 768, 768 }) ==
                     QuantRole::MatrixWeight);
    SYNTH_TEST_CHECK(classify("codec.semantic_model.encoder.layers.0.ff.inter_dense.weight", { 768, 3072 }) ==
                     QuantRole::MatrixWeight);
    SYNTH_TEST_CHECK(classify("codec.semantic_model.encoder.layers.0.ff.output_dense.weight", { 3072, 768 }) ==
                     QuantRole::MatrixWeight);
    SYNTH_TEST_CHECK(classify("codec.semantic_model.feature_projection.projection.weight", { 512, 768 }) ==
                     QuantRole::MatrixWeight);
    // The whitelist is scoped to semantic_model: the same module name
    // anywhere else in the codec would be a convolution's neighbour, not a
    // HuBERT Linear, and must not be quantized on the strength of its
    // spelling alone.
    SYNTH_TEST_CHECK(classify("codec.acoustic_encoder.q_proj.weight", { 768, 768 }) == QuantRole::ConvKernel);

    // codec.encoder_semantic.
    SYNTH_TEST_CHECK(classify("codec.encoder_semantic.conv.weight", { 3, 768, 768 }) == QuantRole::ConvKernel);
    SYNTH_TEST_CHECK(classify("codec.encoder_semantic.conv_blocks.0.res_units.0.conv1.weight", { 3, 768, 768 }) ==
                     QuantRole::ConvKernel);
    SYNTH_TEST_CHECK(classify("codec.encoder_semantic.conv_blocks.1.conv.weight", { 3, 768, 768 }) ==
                     QuantRole::ConvKernel);

    // The generator's own projections are unaffected: the policy is about
    // convolutions, and that half has none.
    SYNTH_TEST_CHECK(classify("llm.layers.0.self_attn.q_proj.weight", { 1024, 2048 }) == QuantRole::MatrixWeight);
    SYNTH_TEST_CHECK(classify("audio_heads.weight", { 1024, 8200 }) == QuantRole::MatrixWeight);
    return 0;
}

int check_unknown_outside_the_catalog() {
    for (const std::string & name : {
             std::string(""),
             std::string("codec"),
             std::string("llm"),
             std::string("codec.acoustic_decoderx.conv1.weight"),
             std::string("codec.acoustic_decoder"),
             std::string("codec.quantizerx.quantizers.0.project_in.weight"),
             std::string("codec.unknown_module.conv.weight"),
             // The exact stray name omnivoice_catalog_test.cpp uses to prove
             // the runtime's own sweep rejects an unresolved tensor.
             std::string("codec.decoder_semantic.conv1.weight"),
             std::string("voice.af_heart"),
         }) {
        SYNTH_TEST_CHECK(classify(name, {}) == QuantRole::Unknown);
    }
    return 0;
}

// Walks the small synthetic package's full registration -- structurally
// faithful to the real catalog at reduced widths (omnivoice_small_layout.h) --
// and asserts nothing classifies as Unknown. This is the test that would catch
// a future catalog addition escaping the policy silently: a new tensor name
// the classifier does not recognize resolves to Unknown here immediately,
// rather than only failing much later when a package built from it refuses to
// load. The small layout is enough for this: completeness depends only on
// which *name patterns* the catalog can register, and the small layout
// exercises every one of them (every module, every per-layer and per-block
// site) at reduced counts, not reduced variety. It is also the cheapest
// faithful source already in the test tree.
int check_completeness_against_the_small_layout() {
    const synth::omnivoice::HParams h       = synth::omnivoice::testing::small_hparams();
    const std::vector<Entry>        entries = synth::omnivoice::testing::expected_entries(h);
    SYNTH_TEST_CHECK(!entries.empty());
    for (const Entry & entry : entries) {
        SYNTH_TEST_CHECK(classify(entry.name, entry.ne) != QuantRole::Unknown);
    }
    return 0;
}

// The counts that decide what each profile actually touches, over the real
// 798-tensor package. Unlike the completeness case, this one needs the *real*
// topology and widths, not the small layout's -- a count is a property of how
// many blocks, layers and feature convolutions the real checkpoint has, not
// of name patterns alone. `real_hparams()` above is that topology, read out
// of the real checkpoint's own tensor inventory (see the file comment).
//
// 158 (codec) and 197 (generator) are the numbers verified against
// catalog.cpp; every other figure here is derived from the catalog through
// the classifier rather than restated.
int check_matrix_weight_count() {
    const synth::omnivoice::HParams h       = real_hparams();
    const std::vector<Entry>        entries = synth::omnivoice::testing::expected_entries(h);

    // The catalog's own arithmetic and this test's hand-built entry list are
    // two independent statements of the same 798-tensor package; they must
    // agree before the role counts below mean anything.
    SYNTH_TEST_CHECK(entries.size() == synth::omnivoice::expected_tensor_count(h));
    SYNTH_TEST_CHECK(entries.size() == 798);

    // What a codec-half profile block-quantizes under the conv-exempt policy:
    // the 73 HuBERT Linears and nothing else. The 158 that used to be
    // quantized split 73 Linears / 85 convolution kernels, and this pair of
    // assertions is what pins that split -- 158 is still the count of codec
    // tensors read through a matrix multiply, but only 73 of them now carry a
    // block-quantized type.
    constexpr uint64_t kCodecMatrixWeightCount = 73;
    constexpr uint64_t kCodecConvKernelCount   = 85;
    SYNTH_TEST_CHECK(count_role_for(entries, ModelHalf::Codec, QuantRole::MatrixWeight) == kCodecMatrixWeightCount);
    SYNTH_TEST_CHECK(count_role_for(entries, ModelHalf::Codec, QuantRole::ConvKernel) == kCodecConvKernelCount);
    SYNTH_TEST_CHECK(kCodecMatrixWeightCount + kCodecConvKernelCount == 158);

    // The per-module breakdown, and the finding the policy turns on: every
    // block-quantized codec tensor is in `codec.semantic_model`, which is the
    // clone-encode path. The acoustic decoder -- the path every synthesis
    // runs -- keeps all 32 of its matrix-read weights at the profile's halved
    // type, so a codec profile now buys nothing at all on the decode side.
    SYNTH_TEST_CHECK(count_role(filter_prefix(entries, "codec.acoustic_decoder."), QuantRole::MatrixWeight) == 0);
    SYNTH_TEST_CHECK(count_role(filter_prefix(entries, "codec.acoustic_encoder."), QuantRole::MatrixWeight) == 0);
    SYNTH_TEST_CHECK(count_role(filter_prefix(entries, "codec.encoder_semantic."), QuantRole::MatrixWeight) == 0);
    SYNTH_TEST_CHECK(count_role(filter_prefix(entries, "codec.semantic_model."), QuantRole::MatrixWeight) ==
                     kCodecMatrixWeightCount);
    SYNTH_TEST_CHECK(count_role(filter_prefix(entries, "codec.acoustic_decoder."), QuantRole::ConvKernel) == 32);
    SYNTH_TEST_CHECK(count_role(filter_prefix(entries, "codec.acoustic_encoder."), QuantRole::ConvKernel) == 36);
    SYNTH_TEST_CHECK(count_role(filter_prefix(entries, "codec.semantic_model."), QuantRole::ConvKernel) == 6);
    SYNTH_TEST_CHECK(count_role(filter_prefix(entries, "codec.encoder_semantic."), QuantRole::ConvKernel) == 11);
    // 73 = four attention projections and two feed-forward matrices across
    // each HuBERT layer, plus the feature projection.
    SYNTH_TEST_CHECK(kCodecMatrixWeightCount == 6 * uint64_t(h.semantic.layer_count) + 1);
    // No codec tensor is ever a RowLookup: the only two `ggml_get_rows`
    // weights in the package are the generator's canvas tables.
    SYNTH_TEST_CHECK(count_role_for(entries, ModelHalf::Codec, QuantRole::RowLookup) == 0);
    // And no generator tensor is ever a ConvKernel: that half emits no
    // convolution at all.
    SYNTH_TEST_CHECK(count_role_for(entries, ModelHalf::Generator, QuantRole::ConvKernel) == 0);

    // What a generator-half profile (Q8) quantizes: 199 two-dimensional
    // weights, of which 2 are the `ggml_get_rows` tables and 197 are matrix
    // multiply operands -- 7 projections across each of 28 layers, plus
    // `audio_heads`.
    constexpr uint64_t kGeneratorMatrixWeightCount = 197;
    SYNTH_TEST_CHECK(count_role_for(entries, ModelHalf::Generator, QuantRole::MatrixWeight) ==
                     kGeneratorMatrixWeightCount);
    SYNTH_TEST_CHECK(kGeneratorMatrixWeightCount == 7 * uint64_t(h.generator.layer_count) + 1);
    SYNTH_TEST_CHECK(count_role_for(entries, ModelHalf::Generator, QuantRole::RowLookup) == 2);
    // Four norms per layer plus the exit norm, and nothing else exact.
    SYNTH_TEST_CHECK(count_role(filter_prefix(entries, "llm."), QuantRole::Sensitive) ==
                     4 * uint64_t(h.generator.layer_count) + 1);
    // The generator emits no convolution, so it contributes no transpose
    // weight -- every one of them is the codec decoder's, one per upsampling
    // ratio.
    SYNTH_TEST_CHECK(count_role_for(entries, ModelHalf::Generator, QuantRole::TransposeWeight) == 0);
    SYNTH_TEST_CHECK(count_role_for(entries, ModelHalf::Codec, QuantRole::TransposeWeight) ==
                     h.codec.upsampling_ratios.size());

    // Every tensor lands somewhere under every profile, and each half's roles
    // partition the whole package.
    SYNTH_TEST_CHECK(count_role(entries, QuantRole::Unknown) == 0);
    for (ModelHalf half : { ModelHalf::Generator, ModelHalf::Codec }) {
        SYNTH_TEST_CHECK(count_role_for(entries, half, QuantRole::Unknown) == 0);
        SYNTH_TEST_CHECK(count_role_for(entries, half, QuantRole::MatrixWeight) +
                             count_role_for(entries, half, QuantRole::RowLookup) +
                             count_role_for(entries, half, QuantRole::ConvKernel) +
                             count_role_for(entries, half, QuantRole::TransposeWeight) +
                             count_role_for(entries, half, QuantRole::Sensitive) ==
                         entries.size());
    }

    // The 312/486 split the reference port publishes as two separate GGUFs is
    // this classifier's own half boundary, tensor for tensor.
    uint64_t generator_tensors = 0;
    for (const Entry & entry : entries) {
        generator_tensors += tensor_half(entry.name) == ModelHalf::Generator ? 1 : 0;
    }
    SYNTH_TEST_CHECK(generator_tensors == 312);
    SYNTH_TEST_CHECK(entries.size() - generator_tensors == 486);
    return 0;
}

}  // namespace

int main() {
    SYNTH_TEST_CHECK(check_generator_groups() == 0);
    SYNTH_TEST_CHECK(check_generator_strays_are_unknown() == 0);
    SYNTH_TEST_CHECK(check_the_half_split() == 0);
    SYNTH_TEST_CHECK(check_rvq_and_concat_projections_are_sensitive() == 0);
    SYNTH_TEST_CHECK(check_transpose_weight_override() == 0);
    SYNTH_TEST_CHECK(check_structural_exceptions() == 0);
    SYNTH_TEST_CHECK(check_snake_alphas_and_biases_are_sensitive() == 0);
    SYNTH_TEST_CHECK(check_one_dimensional_norms_are_sensitive() == 0);
    SYNTH_TEST_CHECK(check_the_conv_exempt_split() == 0);
    SYNTH_TEST_CHECK(check_unknown_outside_the_catalog() == 0);
    SYNTH_TEST_CHECK(check_completeness_against_the_small_layout() == 0);
    SYNTH_TEST_CHECK(check_matrix_weight_count() == 0);
    return 0;
}
