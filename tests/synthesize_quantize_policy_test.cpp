#include "ggml.h"
#include "policy.h"
#include "test-assert.h"

#include <string>

using synth::quantize::find_profile;
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

}  // namespace

int main() {
    const auto * f16 = find_profile("F16");
    SYNTH_TEST_CHECK(f16 != nullptr);
    SYNTH_TEST_CHECK(find_profile("f16") == f16);
    SYNTH_TEST_CHECK(find_profile("F32") == nullptr);
    const auto * q8_mixed = find_profile("Q8_MIXED");
    SYNTH_TEST_CHECK(q8_mixed != nullptr);
    SYNTH_TEST_CHECK(find_profile("q8_mixed") == q8_mixed);
    SYNTH_TEST_CHECK(find_profile(nullptr) == nullptr);

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
