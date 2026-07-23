#include "arch/vits/weights.h"
#include "test-assert.h"
#include "vits-test-fixture.h"

int main() {
    const synth::vits::HParams hparams = synth::test::small_vits_hparams();
    synth::vits::FlowWeights   weights;
    SYNTH_TEST_CHECK(synth::vits::build_flow_weights(nullptr, hparams, weights) == SYNTH_ERR_INVALID_ARG);

    synth::test::GgmlContext valid = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(valid != nullptr);
    synth::test::populate_flow_weight_tensors(valid.get(), hparams);
    SYNTH_TEST_CHECK(synth::vits::build_flow_weights(valid.get(), hparams, weights) == SYNTH_OK);
    SYNTH_TEST_CHECK(weights.blocks.size() == hparams.flow_block_count);
    SYNTH_TEST_CHECK(weights.blocks[0].pre.weight != nullptr && weights.blocks[0].projection.bias != nullptr);
    SYNTH_TEST_CHECK(weights.blocks[0].wn_layers.size() == hparams.flow_wn_layer_count);
    SYNTH_TEST_CHECK(weights.blocks[0].wn_layers[0].input.weight != nullptr);
    SYNTH_TEST_CHECK(weights.blocks[0].wn_layers[1].residual_skip.bias != nullptr);

    synth::test::GgmlContext missing = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(missing != nullptr);
    synth::test::populate_flow_weight_tensors(missing.get(), hparams, "flow.blocks.0.pre.bias");
    SYNTH_TEST_CHECK(synth::vits::build_flow_weights(missing.get(), hparams, weights) == SYNTH_ERR_GGUF);

    synth::test::GgmlContext wrong_type = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(wrong_type != nullptr);
    synth::test::populate_flow_weight_tensors(wrong_type.get(), hparams, {}, "flow.blocks.1.wn.layers.0.input.weight");
    SYNTH_TEST_CHECK(synth::vits::build_flow_weights(wrong_type.get(), hparams, weights) == SYNTH_ERR_GGUF);

    synth::test::GgmlContext mismatched_f16_weight = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(mismatched_f16_weight != nullptr);
    synth::test::populate_flow_weight_tensors(mismatched_f16_weight.get(), hparams, {},
                                              "flow.blocks.1.wn.layers.0.input.weight", {}, GGML_TYPE_F16);
    SYNTH_TEST_CHECK(synth::vits::build_flow_weights(mismatched_f16_weight.get(), hparams, weights) == SYNTH_ERR_GGUF);

    synth::vits::HParams f16_hparams     = hparams;
    f16_hparams.quantization_profile     = synth::vits::QuantizationProfile::F16;
    synth::test::GgmlContext f16_weights = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(f16_weights != nullptr);
    synth::test::populate_flow_weight_tensors(f16_weights.get(), f16_hparams, {}, {}, {}, GGML_TYPE_I32, GGML_TYPE_F16);
    SYNTH_TEST_CHECK(synth::vits::build_flow_weights(f16_weights.get(), f16_hparams, weights) == SYNTH_OK);
    SYNTH_TEST_CHECK(weights.blocks[1].wn_layers[0].input.weight->type == GGML_TYPE_F16);

    synth::vits::HParams q8_hparams     = hparams;
    q8_hparams.quantization_profile     = synth::vits::QuantizationProfile::Q8Mixed;
    q8_hparams.inter_channels           = 64;
    q8_hparams.hidden_channels          = 32;
    synth::test::GgmlContext q8_weights = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(q8_weights != nullptr);
    synth::test::populate_flow_weight_tensors(q8_weights.get(), q8_hparams, {}, {}, {}, GGML_TYPE_I32, GGML_TYPE_Q8_0,
                                              true);
    SYNTH_TEST_CHECK(synth::vits::build_flow_weights(q8_weights.get(), q8_hparams, weights) == SYNTH_OK);
    SYNTH_TEST_CHECK(weights.blocks[0].pre.weight->type == GGML_TYPE_Q8_0);
    SYNTH_TEST_CHECK(weights.blocks[0].pre.weight->ne[0] == 32 && weights.blocks[0].pre.weight->ne[1] == 32);

    synth::test::GgmlContext q8_native_shape = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(q8_native_shape != nullptr);
    synth::test::populate_flow_weight_tensors(q8_native_shape.get(), q8_hparams, {}, {}, {}, GGML_TYPE_I32,
                                              GGML_TYPE_Q8_0, false);
    SYNTH_TEST_CHECK(synth::vits::build_flow_weights(q8_native_shape.get(), q8_hparams, weights) == SYNTH_ERR_GGUF);

    synth::test::GgmlContext mismatched_f32_weight = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(mismatched_f32_weight != nullptr);
    synth::test::populate_flow_weight_tensors(mismatched_f32_weight.get(), f16_hparams, {},
                                              "flow.blocks.1.wn.layers.0.input.weight", {}, GGML_TYPE_F32,
                                              GGML_TYPE_F16);
    SYNTH_TEST_CHECK(synth::vits::build_flow_weights(mismatched_f32_weight.get(), f16_hparams, weights) ==
                     SYNTH_ERR_GGUF);

    synth::test::GgmlContext f16_bias = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(f16_bias != nullptr);
    synth::test::populate_flow_weight_tensors(f16_bias.get(), hparams, {}, "flow.blocks.0.pre.bias", {}, GGML_TYPE_F16);
    SYNTH_TEST_CHECK(synth::vits::build_flow_weights(f16_bias.get(), hparams, weights) == SYNTH_ERR_GGUF);

    synth::test::GgmlContext wrong_shape = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(wrong_shape != nullptr);
    synth::test::populate_flow_weight_tensors(wrong_shape.get(), hparams, {}, {},
                                              "flow.blocks.0.wn.layers.1.residual_skip.weight");
    SYNTH_TEST_CHECK(synth::vits::build_flow_weights(wrong_shape.get(), hparams, weights) == SYNTH_ERR_GGUF);

    const synth::vits::HParams conditioned         = synth::test::small_conditioned_vits_hparams();
    synth::test::GgmlContext   conditioned_context = synth::test::make_ggml_context();
    synth::test::populate_flow_weight_tensors(conditioned_context.get(), conditioned);
    SYNTH_TEST_CHECK(synth::vits::build_flow_weights(conditioned_context.get(), conditioned, weights) == SYNTH_OK);
    SYNTH_TEST_CHECK(weights.blocks[0].conditioning.weight != nullptr &&
                     weights.blocks[1].conditioning.bias != nullptr);
    return 0;
}
