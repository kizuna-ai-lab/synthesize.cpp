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
// half of the package, which since 2026-08-09 is exactly what its profile
// names say: Q8, Q4_K, F16 and BF16 take the generator and leave the codec
// byte-identical to F32, where Q8_CODEC_MIXED and F16_CODEC do the reverse.
// Both directions are asserted here because the failure that matters is the
// silent one -- a Q8_CODEC_MIXED package that started quantizing the
// generator, or a Q8 package that reached into the codec, would still load.
int check_omnivoice_halves() {
    // Q8: the generator's matrices are Q8_0 and Native. Native, not
    // packed, is a real assertion and not a formality: the profile row says
    // PackedMatrix, and it is quantize.cpp's two-dimensional demotion that
    // makes this Native. Packing a [1024, 3072] projection would flatten it
    // into one row of 3,145,728.
    SYNTH_TEST_CHECK(expect_omnivoice("Q8", "llm.layers.0.self_attn.q_proj.weight", GGML_TYPE_Q8_0,
                                      TensorLayout::PackedMatrix) == 0);
    SYNTH_TEST_CHECK(
        expect_omnivoice("Q8", "llm.layers.27.mlp.down_proj.weight", GGML_TYPE_Q8_0, TensorLayout::PackedMatrix) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q8", "audio_heads.weight", GGML_TYPE_Q8_0, TensorLayout::PackedMatrix) == 0);
    // The two `ggml_get_rows` tables take the profile's row-lookup type and
    // are never packed, whatever the matrix layout says.
    SYNTH_TEST_CHECK(expect_omnivoice("Q8", "llm.embed_tokens.weight", GGML_TYPE_Q8_0, TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q8", "audio_embeddings.weight", GGML_TYPE_Q8_0, TensorLayout::Native) == 0);
    // Every generator norm, and the whole codec half.
    SYNTH_TEST_CHECK(expect_omnivoice("Q8", "llm.norm.weight", GGML_TYPE_F32, TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(
        expect_omnivoice("Q8", "llm.layers.0.self_attn.q_norm.weight", GGML_TYPE_F32, TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(
        expect_omnivoice("Q8", "codec.acoustic_decoder.conv1.weight", GGML_TYPE_F32, TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q8", "codec.semantic_model.encoder.layers.0.attn.q_proj.weight", GGML_TYPE_F32,
                                      TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q8", "codec.acoustic_decoder.block.0.conv_t1.weight", GGML_TYPE_F32,
                                      TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q8", "codec.quantizer.quantizers.0.codebook.embed", GGML_TYPE_F32,
                                      TensorLayout::Native) == 0);

    // Q4_K: the same half, and the profile where the RowLookup role stops
    // being cosmetic. Under Q8 above, a matrix and a lookup table both
    // land on Q8_0 and a resolver that had forgotten the role entirely would
    // pass every assertion. Here they must differ, and the direction is not a
    // preference: CUDA's GET_ROWS accepts no k-quant
    // (ggml/src/ggml-cuda/ggml-cuda.cu:5190-5207) where its matrix multiply
    // accepts every one, so a Q4_K lookup table does not fail -- it quietly
    // runs on the CPU. These four checks are the whole argument for the pin.
    SYNTH_TEST_CHECK(expect_omnivoice("Q4_K", "llm.layers.0.self_attn.q_proj.weight", GGML_TYPE_Q4_K,
                                      TensorLayout::PackedMatrix) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q4_K", "llm.layers.27.mlp.down_proj.weight", GGML_TYPE_Q4_K,
                                      TensorLayout::PackedMatrix) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q4_K", "llm.embed_tokens.weight", GGML_TYPE_Q8_0, TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q4_K", "audio_embeddings.weight", GGML_TYPE_Q8_0, TensorLayout::Native) == 0);
    // `audio_heads` reads like an embedding table and is not one: it is a
    // plain `ggml_mul_mat` (generator.cpp), so it is unconstrained and takes
    // the matrix type. Getting this one wrong costs nothing at load time and
    // 27 MiB of file, so it is asserted rather than assumed.
    SYNTH_TEST_CHECK(expect_omnivoice("Q4_K", "audio_heads.weight", GGML_TYPE_Q4_K, TensorLayout::PackedMatrix) == 0);
    // Norms and the whole codec half, exactly as under Q8.
    SYNTH_TEST_CHECK(expect_omnivoice("Q4_K", "llm.norm.weight", GGML_TYPE_F32, TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(
        expect_omnivoice("Q4_K", "llm.layers.0.self_attn.q_norm.weight", GGML_TYPE_F32, TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(
        expect_omnivoice("Q4_K", "codec.acoustic_decoder.conv1.weight", GGML_TYPE_F32, TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q4_K", "codec.semantic_model.encoder.layers.0.attn.q_proj.weight", GGML_TYPE_F32,
                                      TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q4_K", "codec.quantizer.quantizers.0.codebook.embed", GGML_TYPE_F32,
                                      TensorLayout::Native) == 0);

    // F16 and BF16: the same
    // generator half again, narrowed to a reference dtype with a native CUDA
    // MUL_MAT and GET_ROWS path instead of block-quantized. Native layout
    // this time not because of the two-dimensional demotion Q8 needs, but
    // because the profile row itself says Native: neither format is
    // `ggml_is_quantized`, so there is nothing quantize.cpp's demotion needs
    // to catch. row_lookup_type equals matrix_weight_type for both, the same
    // as Q8 and unlike Q4_K, because CUDA's GET_ROWS accepts F16 and
    // BF16 directly (ggml/src/ggml-cuda/ggml-cuda.cu:5190-5207).
    for (const char * profile_name : { "F16", "BF16" }) {
        const ggml_type narrow = std::string(profile_name) == "F16" ? GGML_TYPE_F16 : GGML_TYPE_BF16;
        SYNTH_TEST_CHECK(
            expect_omnivoice(profile_name, "llm.layers.0.self_attn.q_proj.weight", narrow, TensorLayout::Native) == 0);
        SYNTH_TEST_CHECK(
            expect_omnivoice(profile_name, "llm.layers.27.mlp.down_proj.weight", narrow, TensorLayout::Native) == 0);
        SYNTH_TEST_CHECK(expect_omnivoice(profile_name, "audio_heads.weight", narrow, TensorLayout::Native) == 0);
        SYNTH_TEST_CHECK(expect_omnivoice(profile_name, "llm.embed_tokens.weight", narrow, TensorLayout::Native) == 0);
        SYNTH_TEST_CHECK(expect_omnivoice(profile_name, "audio_embeddings.weight", narrow, TensorLayout::Native) == 0);
        SYNTH_TEST_CHECK(expect_omnivoice(profile_name, "llm.norm.weight", GGML_TYPE_F32, TensorLayout::Native) == 0);
        SYNTH_TEST_CHECK(expect_omnivoice(profile_name, "llm.layers.0.self_attn.q_norm.weight", GGML_TYPE_F32,
                                          TensorLayout::Native) == 0);
        SYNTH_TEST_CHECK(expect_omnivoice(profile_name, "codec.acoustic_decoder.conv1.weight", GGML_TYPE_F32,
                                          TensorLayout::Native) == 0);
        SYNTH_TEST_CHECK(expect_omnivoice(profile_name, "codec.semantic_model.encoder.layers.0.attn.q_proj.weight",
                                          GGML_TYPE_F32, TensorLayout::Native) == 0);
        SYNTH_TEST_CHECK(expect_omnivoice(profile_name, "codec.acoustic_decoder.block.0.conv_t1.weight", GGML_TYPE_F32,
                                          TensorLayout::Native) == 0);
        SYNTH_TEST_CHECK(expect_omnivoice(profile_name, "codec.quantizer.quantizers.0.codebook.embed", GGML_TYPE_F32,
                                          TensorLayout::Native) == 0);
    }

    // Q8_CODEC_MIXED, under the conv-exempt codec policy of 2026-08-09. The single
    // most load-bearing assertion in this function is the first one: the
    // decoder's input convolution used to resolve Q8_0/PackedMatrix and now
    // resolves F16/Native, which is the whole policy in one line. Nothing
    // three-dimensional is packed any more, and the only codec tensors that
    // still carry a block-quantized type are the HuBERT Linears.
    SYNTH_TEST_CHECK(expect_omnivoice("Q8_CODEC_MIXED", "codec.acoustic_decoder.conv1.weight", GGML_TYPE_F16,
                                      TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q8_CODEC_MIXED", "codec.acoustic_decoder.conv2.weight", GGML_TYPE_F16,
                                      TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q8_CODEC_MIXED", "codec.acoustic_encoder.block.0.res_unit1.conv1.weight",
                                      GGML_TYPE_F16, TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q8_CODEC_MIXED", "codec.semantic_model.feat_conv.1.conv.weight", GGML_TYPE_F16,
                                      TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q8_CODEC_MIXED", "codec.encoder_semantic.conv.weight", GGML_TYPE_F16,
                                      TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q8_CODEC_MIXED", "codec.semantic_model.encoder.layers.0.attn.q_proj.weight",
                                      GGML_TYPE_Q8_0, TensorLayout::PackedMatrix) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q8_CODEC_MIXED", "codec.semantic_model.feature_projection.projection.weight",
                                      GGML_TYPE_Q8_0, TensorLayout::PackedMatrix) == 0);
    // The three named Sensitive convolutions stay F32, not F16: the policy
    // would have halved them, and they are held exact so the already-cut F16
    // and Q8_CODEC_MIXED packages keep the same bytes there.
    SYNTH_TEST_CHECK(expect_omnivoice("Q8_CODEC_MIXED", "codec.acoustic_encoder.conv1.weight", GGML_TYPE_F32,
                                      TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q8_CODEC_MIXED", "codec.semantic_model.feat_conv.0.conv.weight", GGML_TYPE_F32,
                                      TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q8_CODEC_MIXED", "codec.semantic_model.encoder.pos_conv_embed.conv.weight",
                                      GGML_TYPE_F32, TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q8_CODEC_MIXED", "codec.acoustic_decoder.block.0.conv_t1.weight", GGML_TYPE_F32,
                                      TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q8_CODEC_MIXED", "llm.layers.0.self_attn.q_proj.weight", GGML_TYPE_F32,
                                      TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(
        expect_omnivoice("Q8_CODEC_MIXED", "llm.embed_tokens.weight", GGML_TYPE_F32, TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("Q8_CODEC_MIXED", "audio_heads.weight", GGML_TYPE_F32, TensorLayout::Native) ==
                     0);
    // F16_CODEC is byte-identical across the policy change: its matrix weight
    // type and its halved fallback column are both F16, so a Linear and a
    // convolution land on the same type either way. The already-measured
    // codec-half F16 package's figures stand because of this.
    SYNTH_TEST_CHECK(
        expect_omnivoice("F16_CODEC", "codec.acoustic_decoder.conv1.weight", GGML_TYPE_F16, TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(expect_omnivoice("F16_CODEC", "codec.semantic_model.encoder.layers.0.attn.q_proj.weight",
                                      GGML_TYPE_F16, TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(
        expect_omnivoice("F16_CODEC", "llm.layers.0.mlp.up_proj.weight", GGML_TYPE_F32, TensorLayout::Native) == 0);

    // The naming rule made concrete: the plain `F16` is the GENERATOR half for
    // this family, so it must resolve the mirror image of `F16_CODEC` above --
    // the generator's Linear narrowed and the codec left exact. These two
    // checks are the reason a stale codec-half package cut under the old
    // meaning of `F16` cannot be read as if nothing changed.
    SYNTH_TEST_CHECK(expect_omnivoice("F16", "llm.layers.0.mlp.up_proj.weight", GGML_TYPE_F16, TensorLayout::Native) ==
                     0);
    SYNTH_TEST_CHECK(
        expect_omnivoice("F16", "codec.acoustic_decoder.conv1.weight", GGML_TYPE_F32, TensorLayout::Native) == 0);

    // A stray name is fatal under every profile, including one that leaves
    // the half it would land in untouched.
    synth::quantize::TargetSpec ignored{};
    for (const char * profile_name : { "Q8", "Q4_K", "F16", "BF16", "Q8_CODEC_MIXED", "F16_CODEC" }) {
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
    // `F16` is one row serving two meanings, which is safe only because the
    // half is resolved per-architecture: VITS, Kokoro and Qwen3-TTS cut their
    // codec/decoder halves with it, and omnivoice cuts its GENERATOR half with
    // it (policy.cpp's omnivoice_quantized_half). There is deliberately no
    // second F16 row.
    const auto * f16 = find_profile("F16");
    SYNTH_TEST_CHECK(f16 != nullptr);
    SYNTH_TEST_CHECK(find_profile("f16") == f16);
    SYNTH_TEST_CHECK(f16->matrix_weight_type == GGML_TYPE_F16);
    SYNTH_TEST_CHECK(f16->matrix_weight_type == f16->row_lookup_type);
    SYNTH_TEST_CHECK(find_profile("F32") == nullptr);
    // The sibling families' shared mixed row, and omnivoice's own codec-half
    // rows. `Q8_MIXED` must survive the rename untouched -- it names published
    // VITS and Qwen3-TTS packages -- and must be a different row from the
    // `_CODEC`-qualified name omnivoice now uses.
    const auto * q8_mixed = find_profile("Q8_MIXED");
    SYNTH_TEST_CHECK(q8_mixed != nullptr);
    SYNTH_TEST_CHECK(find_profile("q8_mixed") == q8_mixed);
    const auto * q8_codec_mixed = find_profile("Q8_CODEC_MIXED");
    SYNTH_TEST_CHECK(q8_codec_mixed != nullptr);
    SYNTH_TEST_CHECK(find_profile("q8_codec_mixed") == q8_codec_mixed);
    SYNTH_TEST_CHECK(q8_codec_mixed != q8_mixed);
    const auto * f16_codec = find_profile("F16_CODEC");
    SYNTH_TEST_CHECK(f16_codec != nullptr);
    SYNTH_TEST_CHECK(find_profile("f16_codec") == f16_codec);
    SYNTH_TEST_CHECK(f16_codec != f16);
    const auto * q8 = find_profile("Q8");
    SYNTH_TEST_CHECK(q8 != nullptr);
    SYNTH_TEST_CHECK(find_profile("q8") == q8);
    SYNTH_TEST_CHECK(q8 != q8_codec_mixed);
    // `Q4_K` is a real profile since the 2026-08-09 rename -- it is what the
    // former `Q4_K_GEN` is called now -- so this lookup must resolve rather
    // than being refused as the near-miss spelling it used to be.
    const auto * q4_k = find_profile("Q4_K");
    SYNTH_TEST_CHECK(q4_k != nullptr);
    SYNTH_TEST_CHECK(find_profile("q4_k") == q4_k);
    SYNTH_TEST_CHECK(q4_k != q8);
    // The pin, stated once at the table row itself: the two columns differ,
    // which they do for no other profile in the table.
    SYNTH_TEST_CHECK(q4_k->matrix_weight_type == GGML_TYPE_Q4_K);
    SYNTH_TEST_CHECK(q4_k->row_lookup_type == GGML_TYPE_Q8_0);
    SYNTH_TEST_CHECK(q8->matrix_weight_type == q8->row_lookup_type);
    // BF16: same lookup reasoning as Q8 -- row_lookup_type agrees with
    // matrix_weight_type because CUDA's GET_ROWS takes the format directly.
    const auto * bf16 = find_profile("BF16");
    SYNTH_TEST_CHECK(bf16 != nullptr);
    SYNTH_TEST_CHECK(find_profile("bf16") == bf16);
    SYNTH_TEST_CHECK(bf16->matrix_weight_type == GGML_TYPE_BF16);
    SYNTH_TEST_CHECK(bf16->matrix_weight_type == bf16->row_lookup_type);
    SYNTH_TEST_CHECK(bf16 != f16);
    // `Q4_K_M` is llama.cpp's mixture name for a Q4_K/Q6_K blend. This project
    // implements no such mixture, and a package asking for one must be refused
    // rather than quietly served the pure-Q4_K profile that is now spelled
    // exactly that way.
    SYNTH_TEST_CHECK(find_profile("Q4_K_M") == nullptr);
    // The `_GEN`-suffixed names are gone and must not resolve. A build that
    // kept an alias would let a stale script keep cutting packages under a
    // name no reader will find documented.
    SYNTH_TEST_CHECK(find_profile("Q8_GEN") == nullptr);
    SYNTH_TEST_CHECK(find_profile("Q4_K_GEN") == nullptr);
    SYNTH_TEST_CHECK(find_profile("F16_GEN") == nullptr);
    SYNTH_TEST_CHECK(find_profile("BF16_GEN") == nullptr);
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

    SYNTH_TEST_CHECK(expect_spec("Q8_CODEC_MIXED", "flow.blocks.3.wn.layers.2.residual_skip.weight", GGML_TYPE_Q8_0,
                                 TensorLayout::PackedMatrix) == 0);
    SYNTH_TEST_CHECK(expect_spec("Q8_CODEC_MIXED", "decoder.pre.weight", GGML_TYPE_Q8_0, TensorLayout::PackedMatrix) ==
                     0);
    SYNTH_TEST_CHECK(expect_spec("Q8_CODEC_MIXED", "decoder.post.weight", GGML_TYPE_Q8_0, TensorLayout::PackedMatrix) ==
                     0);
    SYNTH_TEST_CHECK(expect_spec("Q8_CODEC_MIXED", "decoder.upsample.3.transpose_conv.weight", GGML_TYPE_F32,
                                 TensorLayout::Native) == 0);
    SYNTH_TEST_CHECK(
        expect_spec("Q8_CODEC_MIXED", "duration_predictor.pre.weight", GGML_TYPE_F32, TensorLayout::Native) == 0);

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
