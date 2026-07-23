#include "arch/vits/weights.h"
#include "test-assert.h"
#include "vits-test-fixture.h"

int main() {
    const synth::vits::HParams hparams = synth::test::small_vits_hparams();
    synth::vits::TextWeights   weights;
    SYNTH_TEST_CHECK(synth::vits::build_text_weights(nullptr, hparams, weights) == SYNTH_ERR_INVALID_ARG);

    synth::test::GgmlContext valid = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(valid != nullptr);
    synth::test::populate_text_weight_tensors(valid.get(), hparams);
    SYNTH_TEST_CHECK(synth::vits::build_text_weights(valid.get(), hparams, weights) == SYNTH_OK);
    SYNTH_TEST_CHECK(weights.token_embedding != nullptr);
    SYNTH_TEST_CHECK(weights.blocks.size() == hparams.text_layer_count);
    SYNTH_TEST_CHECK(weights.projection.weight != nullptr && weights.projection.bias != nullptr);

    synth::test::GgmlContext missing = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(missing != nullptr);
    synth::test::populate_text_weight_tensors(missing.get(), hparams, "text_encoder.projection.bias");
    SYNTH_TEST_CHECK(synth::vits::build_text_weights(missing.get(), hparams, weights) == SYNTH_ERR_GGUF);

    synth::test::GgmlContext wrong_type = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(wrong_type != nullptr);
    synth::test::populate_text_weight_tensors(wrong_type.get(), hparams, {}, "text_encoder.token_embedding.weight");
    SYNTH_TEST_CHECK(synth::vits::build_text_weights(wrong_type.get(), hparams, weights) == SYNTH_ERR_GGUF);

    synth::test::GgmlContext f16_weight = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(f16_weight != nullptr);
    synth::test::populate_text_weight_tensors(f16_weight.get(), hparams, {}, "text_encoder.token_embedding.weight", {},
                                              GGML_TYPE_F16);
    SYNTH_TEST_CHECK(synth::vits::build_text_weights(f16_weight.get(), hparams, weights) == SYNTH_ERR_GGUF);

    synth::test::GgmlContext f16_norm = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(f16_norm != nullptr);
    synth::test::populate_text_weight_tensors(f16_norm.get(), hparams, {},
                                              "text_encoder.blocks.0.attention_norm.weight", {}, GGML_TYPE_F16);
    SYNTH_TEST_CHECK(synth::vits::build_text_weights(f16_norm.get(), hparams, weights) == SYNTH_ERR_GGUF);

    synth::test::GgmlContext f16_relative = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(f16_relative != nullptr);
    synth::test::populate_text_weight_tensors(f16_relative.get(), hparams, {},
                                              "text_encoder.blocks.0.attention.relative_key.weight", {}, GGML_TYPE_F16);
    SYNTH_TEST_CHECK(synth::vits::build_text_weights(f16_relative.get(), hparams, weights) == SYNTH_ERR_GGUF);

    synth::test::GgmlContext wrong_shape = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(wrong_shape != nullptr);
    synth::test::populate_text_weight_tensors(wrong_shape.get(), hparams, {}, {}, "text_encoder.projection.weight");
    SYNTH_TEST_CHECK(synth::vits::build_text_weights(wrong_shape.get(), hparams, weights) == SYNTH_ERR_GGUF);

    const synth::vits::HParams conditioned   = synth::test::small_conditioned_vits_hparams();
    synth::test::GgmlContext   voice_context = synth::test::make_ggml_context();
    synth::test::populate_voice_weight_tensors(voice_context.get(), conditioned);
    synth::vits::VoiceWeights voice;
    SYNTH_TEST_CHECK(synth::vits::build_voice_weights(voice_context.get(), conditioned, voice) == SYNTH_OK);
    SYNTH_TEST_CHECK(voice.embedding != nullptr);
    synth::test::GgmlContext f16_voice = synth::test::make_ggml_context();
    synth::test::populate_voice_weight_tensors(f16_voice.get(), conditioned, {}, "voice.embedding.weight", {},
                                               GGML_TYPE_F16);
    SYNTH_TEST_CHECK(synth::vits::build_voice_weights(f16_voice.get(), conditioned, voice) == SYNTH_ERR_GGUF);
    synth::test::GgmlContext missing_voice = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(synth::vits::build_voice_weights(missing_voice.get(), conditioned, voice) == SYNTH_ERR_GGUF);
    return 0;
}
