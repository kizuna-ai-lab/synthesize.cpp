#include "arch/vits/weights.h"
#include "test-assert.h"
#include "vits-test-fixture.h"

int main() {
    const synth::vits::HParams   hparams = synth::test::small_vits_hparams();
    synth::vits::DurationWeights weights;
    SYNTH_TEST_CHECK(synth::vits::build_duration_weights(nullptr, hparams, weights) == SYNTH_ERR_INVALID_ARG);

    synth::test::GgmlContext valid = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(valid != nullptr);
    synth::test::populate_duration_weight_tensors(valid.get(), hparams);
    SYNTH_TEST_CHECK(synth::vits::build_duration_weights(valid.get(), hparams, weights) == SYNTH_OK);
    SYNTH_TEST_CHECK(weights.pre.weight != nullptr && weights.projection.bias != nullptr);
    SYNTH_TEST_CHECK(weights.dds.blocks.size() == hparams.duration_dds_layer_count);
    SYNTH_TEST_CHECK(weights.affine_bias != nullptr && weights.affine_log_scale != nullptr);
    SYNTH_TEST_CHECK(weights.flows.size() == hparams.duration_flow_count);
    SYNTH_TEST_CHECK(weights.flows[0].dds.blocks.size() == hparams.duration_dds_layer_count);

    synth::test::GgmlContext missing = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(missing != nullptr);
    synth::test::populate_duration_weight_tensors(missing.get(), hparams, "duration_predictor.affine.bias");
    SYNTH_TEST_CHECK(synth::vits::build_duration_weights(missing.get(), hparams, weights) == SYNTH_ERR_GGUF);

    synth::test::GgmlContext wrong_type = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(wrong_type != nullptr);
    synth::test::populate_duration_weight_tensors(wrong_type.get(), hparams, {},
                                                  "duration_predictor.dds.blocks.0.depthwise.weight");
    SYNTH_TEST_CHECK(synth::vits::build_duration_weights(wrong_type.get(), hparams, weights) == SYNTH_ERR_GGUF);

    synth::test::GgmlContext f16_weight = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(f16_weight != nullptr);
    synth::test::populate_duration_weight_tensors(
        f16_weight.get(), hparams, {}, "duration_predictor.dds.blocks.0.depthwise.weight", {}, GGML_TYPE_F16);
    SYNTH_TEST_CHECK(synth::vits::build_duration_weights(f16_weight.get(), hparams, weights) == SYNTH_ERR_GGUF);

    synth::test::GgmlContext f16_affine = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(f16_affine != nullptr);
    synth::test::populate_duration_weight_tensors(f16_affine.get(), hparams, {}, "duration_predictor.affine.log_scale",
                                                  {}, GGML_TYPE_F16);
    SYNTH_TEST_CHECK(synth::vits::build_duration_weights(f16_affine.get(), hparams, weights) == SYNTH_ERR_GGUF);

    synth::test::GgmlContext wrong_shape = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(wrong_shape != nullptr);
    synth::test::populate_duration_weight_tensors(wrong_shape.get(), hparams, {}, {},
                                                  "duration_predictor.flows.1.projection.weight");
    SYNTH_TEST_CHECK(synth::vits::build_duration_weights(wrong_shape.get(), hparams, weights) == SYNTH_ERR_GGUF);

    const synth::vits::HParams conditioned         = synth::test::small_conditioned_vits_hparams();
    synth::test::GgmlContext   conditioned_context = synth::test::make_ggml_context();
    synth::test::populate_duration_weight_tensors(conditioned_context.get(), conditioned);
    SYNTH_TEST_CHECK(synth::vits::build_duration_weights(conditioned_context.get(), conditioned, weights) == SYNTH_OK);
    SYNTH_TEST_CHECK(weights.conditioning.weight != nullptr && weights.conditioning.bias != nullptr);
    return 0;
}
