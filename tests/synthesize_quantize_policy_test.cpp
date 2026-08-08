#include "ggml.h"
#include "policy.h"
#include "test-assert.h"

#include <string>

using synth::quantize::find_profile;
using synth::quantize::resolve_omnivoice_target_spec;
using synth::quantize::resolve_vits_target_spec;
using synth::quantize::resolve_vits_target_type;
using synth::quantize::TensorLayout;

namespace {

int expect_type(const char * name, ggml_type expected) {
    ggml_type actual = GGML_TYPE_COUNT;
    SYNTH_TEST_CHECK(resolve_vits_target_type(*find_profile("F16"), name, actual));
    SYNTH_TEST_CHECK(actual == expected);
    return 0;
}

int expect_spec(const char * profile_name, const char * name, ggml_type expected_type, TensorLayout expected_layout) {
    const auto * profile = find_profile(profile_name);
    SYNTH_TEST_CHECK(profile != nullptr);
    synth::quantize::TargetSpec actual{};
    SYNTH_TEST_CHECK(resolve_vits_target_spec(*profile, name, actual));
    SYNTH_TEST_CHECK(actual.type == expected_type);
    SYNTH_TEST_CHECK(actual.layout == expected_layout);
    return 0;
}

int expect_omnivoice(const char * profile_name,
                     const char * name,
                     ggml_type    expected_type,
                     TensorLayout expected_layout) {
    const auto * profile = find_profile(profile_name);
    SYNTH_TEST_CHECK(profile != nullptr);
    synth::quantize::TargetSpec actual{};
    SYNTH_TEST_CHECK(resolve_omnivoice_target_spec(*profile, name, actual));
    SYNTH_TEST_CHECK(actual.type == expected_type);
    SYNTH_TEST_CHECK(actual.layout == expected_layout);
    return 0;
}

// OmniVoice is the only family whose profiles do not all quantize the same
// half of the package: Q8_GEN and Q4_K_GEN take the generator and leave the
// codec byte-identical to F32, where Q8_MIXED and F16 do the reverse. Both
// directions are asserted here because the failure that matters is the
// silent one -- a Q8_MIXED package that started quantizing the generator, or
// a Q8_GEN package that reached into the codec, would still load.
int check_omnivoice_halves() {
    // Q8_GEN: the generator's matrices are Q8_0 and Native. Native, not
    // packed, is a real assertion and not a formality: the profile row says
    // PackedMatrix, and it is quantize.cpp's two-dimensional demotion that
    // makes this Native. Packing a [1024, 3072] projection would flatten it
    // into one row of 3,145,728.
    SYNTH_TEST_CHECK(expect_omnivoice("Q8_GEN", "llm.layers.0.self_attn.q_proj.weight", GGML_TYPE_Q8_0,
                                      TensorLayout::PackedMatrix) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q8_GEN", "llm.layers.27.mlp.down_proj.weight", GGML_TYPE_Q8_0,
                                      TensorLayout::PackedMatrix) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q8_GEN", "audio_heads.weight", GGML_TYPE_Q8_0, TensorLayout::PackedMatrix) == 0);
    // The two `ggml_get_rows` tables take the profile's row-lookup type and
    // are never packed, whatever the matrix layout says.
    SYNTH_TEST_CHECK(expect_omnivoice("Q8_GEN", "llm.embed_tokens.weight", GGML_TYPE_Q8_0, TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q8_GEN", "audio_embeddings.weight", GGML_TYPE_Q8_0, TensorLayout::Native) == 0);
    // Every generator norm, and the whole codec half.
    SYNTH_TEST_CHECK(expect_omnivoice("Q8_GEN", "llm.norm.weight", GGML_TYPE_F32, TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(
        expect_omnivoice("Q8_GEN", "llm.layers.0.self_attn.q_norm.weight", GGML_TYPE_F32, TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(
        expect_omnivoice("Q8_GEN", "codec.acoustic_decoder.conv1.weight", GGML_TYPE_F32, TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q8_GEN", "codec.semantic_model.encoder.layers.0.attn.q_proj.weight",
                                      GGML_TYPE_F32, TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q8_GEN", "codec.acoustic_decoder.block.0.conv_t1.weight", GGML_TYPE_F32,
                                      TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q8_GEN", "codec.quantizer.quantizers.0.codebook.embed", GGML_TYPE_F32,
                                      TensorLayout::Native) == 0);

    // Q4_K_GEN: the same half, and the profile where the RowLookup role stops
    // being cosmetic. Under Q8_GEN above, a matrix and a lookup table both
    // land on Q8_0 and a resolver that had forgotten the role entirely would
    // pass every assertion. Here they must differ, and the direction is not a
    // preference: CUDA's GET_ROWS accepts no k-quant
    // (ggml/src/ggml-cuda/ggml-cuda.cu:5190-5207) where its matrix multiply
    // accepts every one, so a Q4_K lookup table does not fail -- it quietly
    // runs on the CPU. These four checks are the whole argument for the pin.
    SYNTH_TEST_CHECK(expect_omnivoice("Q4_K_GEN", "llm.layers.0.self_attn.q_proj.weight", GGML_TYPE_Q4_K,
                                      TensorLayout::PackedMatrix) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q4_K_GEN", "llm.layers.27.mlp.down_proj.weight", GGML_TYPE_Q4_K,
                                      TensorLayout::PackedMatrix) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q4_K_GEN", "llm.embed_tokens.weight", GGML_TYPE_Q8_0, TensorLayout::Native) ==
                     0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q4_K_GEN", "audio_embeddings.weight", GGML_TYPE_Q8_0, TensorLayout::Native) ==
                     0);
    // `audio_heads` reads like an embedding table and is not one: it is a
    // plain `ggml_mul_mat` (generator.cpp), so it is unconstrained and takes
    // the matrix type. Getting this one wrong costs nothing at load time and
    // 27 MiB of file, so it is asserted rather than assumed.
    SYNTH_TEST_CHECK(expect_omnivoice("Q4_K_GEN", "audio_heads.weight", GGML_TYPE_Q4_K, TensorLayout::PackedMatrix) ==
                     0);
    // Norms and the whole codec half, exactly as under Q8_GEN.
    SYNTH_TEST_CHECK(expect_omnivoice("Q4_K_GEN", "llm.norm.weight", GGML_TYPE_F32, TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(
        expect_omnivoice("Q4_K_GEN", "llm.layers.0.self_attn.q_norm.weight", GGML_TYPE_F32, TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(
        expect_omnivoice("Q4_K_GEN", "codec.acoustic_decoder.conv1.weight", GGML_TYPE_F32, TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q4_K_GEN", "codec.semantic_model.encoder.layers.0.attn.q_proj.weight",
                                      GGML_TYPE_F32, TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q4_K_GEN", "codec.quantizer.quantizers.0.codebook.embed", GGML_TYPE_F32,
                                      TensorLayout::Native) == 0);

    // Q8_MIXED and F16, unchanged by the generator acquiring real roles.
    SYNTH_TEST_CHECK(expect_omnivoice("Q8_MIXED", "codec.acoustic_decoder.conv1.weight", GGML_TYPE_Q8_0,
                                      TensorLayout::PackedMatrix) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q8_MIXED", "codec.acoustic_decoder.block.0.conv_t1.weight", GGML_TYPE_F32,
                                      TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(
        expect_omnivoice("Q8_MIXED", "llm.layers.0.self_attn.q_proj.weight", GGML_TYPE_F32, TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q8_MIXED", "llm.embed_tokens.weight", GGML_TYPE_F32, TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q8_MIXED", "audio_heads.weight", GGML_TYPE_F32, TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(
        expect_omnivoice("F16", "codec.acoustic_decoder.conv1.weight", GGML_TYPE_F16, TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("F16", "llm.layers.0.mlp.up_proj.weight", GGML_TYPE_F32, TensorLayout::Native) ==
                     0);

    // A stray name is fatal under every profile, including one that leaves
    // the half it would land in untouched.
    synth::quantize::TargetSpec ignored{};
    for (const char * profile_name : { "Q8_GEN", "Q4_K_GEN", "Q8_MIXED", "F16" }) {
        const auto * profile = find_profile(profile_name);
        SYNTH_TEST_CHECK(profile != nullptr);
        SYNTH_TEST_CHECK(!resolve_omnivoice_target_spec(*profile, "llm.layers.0.mlp.fc.weight", ignored));
        SYNTH_TEST_CHECK(!resolve_omnivoice_target_spec(*profile, "codec.unknown_module.conv.weight", ignored));
        SYNTH_TEST_CHECK(!resolve_omnivoice_target_spec(*profile, "llm", ignored));
    }
    return 0;
}

}  // namespace

int main() {
    const auto * f16 = find_profile("F16");
    SYNTH_TEST_CHECK(f16 != nullptr);
    SYNTH_TEST_CHECK(find_profile("f16") == f16);
    SYNTH_TEST_CHECK(find_profile("F32") == nullptr);
    const auto * q8_mixed = find_profile("Q8_MIXED");
    SYNTH_TEST_CHECK(q8_mixed != nullptr);
    SYNTH_TEST_CHECK(find_profile("q8_mixed") == q8_mixed);
    const auto * q8_gen = find_profile("Q8_GEN");
    SYNTH_TEST_CHECK(q8_gen != nullptr);
    SYNTH_TEST_CHECK(find_profile("q8_gen") == q8_gen);
    SYNTH_TEST_CHECK(q8_gen != q8_mixed);
    const auto * q4_k_gen = find_profile("Q4_K_GEN");
    SYNTH_TEST_CHECK(q4_k_gen != nullptr);
    SYNTH_TEST_CHECK(find_profile("q4_k_gen") == q4_k_gen);
    SYNTH_TEST_CHECK(q4_k_gen != q8_gen);
    // The pin, stated once at the table row itself: the two columns differ,
    // which they do for no other profile in the table.
    SYNTH_TEST_CHECK(q4_k_gen->matrix_weight_type == GGML_TYPE_Q4_K);
    SYNTH_TEST_CHECK(q4_k_gen->row_lookup_type == GGML_TYPE_Q8_0);
    SYNTH_TEST_CHECK(q8_gen->matrix_weight_type == q8_gen->row_lookup_type);
    // `Q4_K_M` is llama.cpp's mixture name for a Q4_K/Q6_K blend. This project
    // implements no such mixture, and a package asking for one must be refused
    // rather than quietly served the pure-Q4_K profile that happens to be
    // spelled similarly.
    SYNTH_TEST_CHECK(find_profile("Q4_K_M") == nullptr);
    SYNTH_TEST_CHECK(find_profile("Q4_K") == nullptr);
    SYNTH_TEST_CHECK(find_profile(nullptr) == nullptr);
    SYNTH_TEST_CHECK(check_omnivoice_halves() == 0);

    SYNTH_TEST_CHECK(expect_type("voice.embedding.weight", GGML_TYPE_F32) == 0);
    SYNTH_TEST_CHECK(expect_type("text_encoder.token_embedding.weight", GGML_TYPE_F32) == 0);
    SYNTH_TEST_CHECK(expect_type("text_encoder.blocks.5.attention.relative_key.weight", GGML_TYPE_F32) == 0);
    SYNTH_TEST_CHECK(expect_type("text_encoder.blocks.3.ffn.output.weight", GGML_TYPE_F32) == 0);
    SYNTH_TEST_CHECK(expect_type("duration_predictor.flows.2.dds.blocks.1.pointwise.weight", GGML_TYPE_F32) == 0);
    SYNTH_TEST_CHECK(expect_type("flow.blocks.3.wn.layers.2.residual_skip.weight", GGML_TYPE_F16) == 0);
    // Transposed-convolution weights are the one role VITS keeps at full
    // precision in every profile; see resolve_vits_target_spec.
    SYNTH_TEST_CHECK(expect_type("decoder.upsample.3.transpose_conv.weight", GGML_TYPE_F32) == 0);
    SYNTH_TEST_CHECK(expect_type("decoder.upsample.2.resblocks.1.conv2.2.weight", GGML_TYPE_F16) == 0);

    SYNTH_TEST_CHECK(expect_spec("Q8_MIXED", "flow.blocks.3.wn.layers.2.residual_skip.weight", GGML_TYPE_Q8_0,
                                 TensorLayout::PackedMatrix) == 0);
    SYNTH_TEST_CHECK(expect_spec("Q8_MIXED", "decoder.pre.weight", GGML_TYPE_Q8_0, TensorLayout::PackedMatrix) == 0);
    SYNTH_TEST_CHECK(expect_spec("Q8_MIXED", "decoder.post.weight", GGML_TYPE_Q8_0, TensorLayout::PackedMatrix) == 0);
    SYNTH_TEST_CHECK(
        expect_spec("Q8_MIXED", "decoder.upsample.3.transpose_conv.weight", GGML_TYPE_F32, TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(expect_spec("Q8_MIXED", "duration_predictor.pre.weight", GGML_TYPE_F32, TensorLayout::Native) ==
                     0);

    SYNTH_TEST_CHECK(expect_type("text_encoder.blocks.0.attention.query.bias", GGML_TYPE_F32) == 0);
    SYNTH_TEST_CHECK(expect_type("text_encoder.blocks.0.attention_norm.weight", GGML_TYPE_F32) == 0);
    SYNTH_TEST_CHECK(expect_type("duration_predictor.dds.blocks.2.pointwise_norm.weight", GGML_TYPE_F32) == 0);
    SYNTH_TEST_CHECK(expect_type("duration_predictor.affine.bias", GGML_TYPE_F32) == 0);
    SYNTH_TEST_CHECK(expect_type("duration_predictor.affine.log_scale", GGML_TYPE_F32) == 0);
    SYNTH_TEST_CHECK(expect_type("decoder.upsample.0.transpose_conv.bias", GGML_TYPE_F32) == 0);

    ggml_type actual = GGML_TYPE_COUNT;
    SYNTH_TEST_CHECK(!resolve_vits_target_type(*f16, "unknown.weight", actual));
    SYNTH_TEST_CHECK(!resolve_vits_target_type(*f16, "decoder.future_module.weight", actual));
    SYNTH_TEST_CHECK(!resolve_vits_target_type(*f16, "text_encoder.blocks.x.ffn.input.weight", actual));
    SYNTH_TEST_CHECK(!resolve_vits_target_type(*f16, "decoder.post.bias", actual));
    return 0;
}
