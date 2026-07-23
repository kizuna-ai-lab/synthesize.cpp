#include "arch/vits/weights.h"
#include "test-assert.h"
#include "vits-test-fixture.h"

int main() {
    const synth::vits::HParams  hparams = synth::test::small_vits_hparams();
    synth::vits::DecoderWeights weights;
    SYNTH_TEST_CHECK(synth::vits::build_decoder_weights(nullptr, hparams, weights) == SYNTH_ERR_INVALID_ARG);

    synth::test::GgmlContext valid = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(valid != nullptr);
    synth::test::populate_decoder_weight_tensors(valid.get(), hparams);
    SYNTH_TEST_CHECK(synth::vits::build_decoder_weights(valid.get(), hparams, weights) == SYNTH_OK);
    SYNTH_TEST_CHECK(weights.pre.weight != nullptr && weights.pre.bias != nullptr);
    SYNTH_TEST_CHECK(weights.stages.size() == 1);
    SYNTH_TEST_CHECK(weights.stages[0].transpose_conv.weight != nullptr);
    SYNTH_TEST_CHECK(weights.stages[0].transpose_conv.bias != nullptr);
    SYNTH_TEST_CHECK(weights.stages[0].resblocks.size() == 1);
    SYNTH_TEST_CHECK(weights.stages[0].resblocks[0].conv1.size() == 3);
    SYNTH_TEST_CHECK(weights.stages[0].resblocks[0].conv2.size() == 3);
    SYNTH_TEST_CHECK(weights.stages[0].resblocks[0].conv1[2].bias != nullptr);
    SYNTH_TEST_CHECK(weights.post_weight != nullptr);

    synth::test::GgmlContext missing = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(missing != nullptr);
    synth::test::populate_decoder_weight_tensors(missing.get(), hparams, "decoder.pre.bias");
    SYNTH_TEST_CHECK(synth::vits::build_decoder_weights(missing.get(), hparams, weights) == SYNTH_ERR_GGUF);

    synth::test::GgmlContext wrong_type = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(wrong_type != nullptr);
    synth::test::populate_decoder_weight_tensors(wrong_type.get(), hparams, {},
                                                 "decoder.upsample.0.resblocks.0.conv2.1.weight");
    SYNTH_TEST_CHECK(synth::vits::build_decoder_weights(wrong_type.get(), hparams, weights) == SYNTH_ERR_GGUF);

    synth::test::GgmlContext mismatched_f16_weight = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(mismatched_f16_weight != nullptr);
    synth::test::populate_decoder_weight_tensors(mismatched_f16_weight.get(), hparams, {},
                                                 "decoder.upsample.0.transpose_conv.weight", {}, GGML_TYPE_F16);
    SYNTH_TEST_CHECK(synth::vits::build_decoder_weights(mismatched_f16_weight.get(), hparams, weights) ==
                     SYNTH_ERR_GGUF);

    synth::vits::HParams f16_hparams    = hparams;
    f16_hparams.quantization_profile    = synth::vits::QuantizationProfile::F16;
    synth::test::GgmlContext f16_weight = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(f16_weight != nullptr);
    synth::test::populate_decoder_weight_tensors(f16_weight.get(), f16_hparams, {}, {}, {}, GGML_TYPE_I32,
                                                 GGML_TYPE_F16, GGML_TYPE_F16);
    SYNTH_TEST_CHECK(synth::vits::build_decoder_weights(f16_weight.get(), f16_hparams, weights) == SYNTH_OK);
    SYNTH_TEST_CHECK(weights.stages[0].transpose_conv.weight->type == GGML_TYPE_F16);

    synth::vits::HParams q8_hparams     = hparams;
    q8_hparams.quantization_profile     = synth::vits::QuantizationProfile::Q8Mixed;
    q8_hparams.inter_channels           = 32;
    q8_hparams.decoder_initial_channels = 64;
    synth::test::GgmlContext q8_weights = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(q8_weights != nullptr);
    synth::test::populate_decoder_weight_tensors(q8_weights.get(), q8_hparams, {}, {}, {}, GGML_TYPE_I32,
                                                 GGML_TYPE_Q8_0, GGML_TYPE_F16, true);
    SYNTH_TEST_CHECK(synth::vits::build_decoder_weights(q8_weights.get(), q8_hparams, weights) == SYNTH_OK);
    SYNTH_TEST_CHECK(weights.pre.weight->type == GGML_TYPE_Q8_0 && weights.pre.weight->ne[0] == 7 * 32);
    SYNTH_TEST_CHECK(weights.stages[0].transpose_conv.weight->type == GGML_TYPE_F16);
    SYNTH_TEST_CHECK(weights.post_weight->type == GGML_TYPE_Q8_0 && weights.post_weight->ne[0] == 7 * 32);

    synth::test::GgmlContext f16_bias = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(f16_bias != nullptr);
    synth::test::populate_decoder_weight_tensors(f16_bias.get(), hparams, {}, "decoder.upsample.0.transpose_conv.bias",
                                                 {}, GGML_TYPE_F16);
    SYNTH_TEST_CHECK(synth::vits::build_decoder_weights(f16_bias.get(), hparams, weights) == SYNTH_ERR_GGUF);

    synth::test::GgmlContext wrong_shape = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(wrong_shape != nullptr);
    synth::test::populate_decoder_weight_tensors(wrong_shape.get(), hparams, {}, {}, "decoder.post.weight");
    SYNTH_TEST_CHECK(synth::vits::build_decoder_weights(wrong_shape.get(), hparams, weights) == SYNTH_ERR_GGUF);

    const synth::vits::HParams conditioned         = synth::test::small_conditioned_vits_hparams();
    synth::test::GgmlContext   conditioned_context = synth::test::make_ggml_context();
    synth::test::populate_decoder_weight_tensors(conditioned_context.get(), conditioned);
    SYNTH_TEST_CHECK(synth::vits::build_decoder_weights(conditioned_context.get(), conditioned, weights) == SYNTH_OK);
    SYNTH_TEST_CHECK(weights.conditioning.weight != nullptr && weights.conditioning.bias != nullptr);
    return 0;
}
