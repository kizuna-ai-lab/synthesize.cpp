// The Kokoro quantization split.
//
// This classifier is the single source of truth shared by the offline
// quantizer and the runtime's catalog, so what it decides is a package
// contract, not an implementation detail. The point of the test is the split
// itself: everything that decides F0 or the durations stays at the reference
// dtype, because the harmonic source turns a small relative F0 difference into
// radians of phase over an utterance.
//
// That the real 511-tensor package classifies completely is proved elsewhere —
// the runtime rejects any tensor outside the catalog, so the integration tier
// loading a package is that check.

#include "arch/kokoro/quantization.h"
#include "test-assert.h"

#include <string>

using synth::kokoro::tensor_role;
using synth::kokoro::TensorRole;

int main() {
    // The structural path is never quantized, whatever the profile.
    for (const char * name : {
             "bert.embeddings.word_embeddings.weight",
             "bert.layer.attention.query.weight",
             "bert.layer.ffn.weight",
             "bert_encoder.weight",
             "text_encoder.embedding.weight",
             "text_encoder.cnn.0.0.weight",
             "text_encoder.cnn.2.1.gamma",
             "text_encoder.lstm.weight_ih_l0",
             "text_encoder.lstm.weight_hh_l0_reverse",
             "predictor.lstm.weight_ih_l0",
             "predictor.shared.weight_hh_l0_reverse",
             "predictor.text_encoder.lstms.1.fc.weight",
             "predictor.duration_proj.linear_layer.weight",
             "predictor.F0.0.conv1.weight",
             "predictor.N.2.norm2.fc.weight",
             "predictor.F0_proj.weight",
             "voice.af_heart",
             "voice.zm_yunyang",
         }) {
        SYNTH_TEST_CHECK(tensor_role(name) == TensorRole::Sensitive);
    }

    // The decoder and the generator are what gets quantized.
    for (const char * name : {
             "decoder.encode.conv1.weight",
             "decoder.encode.conv1x1.weight",
             "decoder.encode.norm1.fc.weight",
             "decoder.decode.3.conv2.weight",
             "decoder.decode.0.norm2.fc.weight",
             "decoder.asr_res.0.weight",
             "decoder.generator.conv_post.weight",
             "decoder.generator.noise_convs.1.weight",
             "decoder.generator.resblocks.5.convs1.2.weight",
             "decoder.generator.resblocks.0.adain2.1.fc.weight",
             "decoder.generator.noise_res.1.convs2.0.weight",
         }) {
        SYNTH_TEST_CHECK(tensor_role(name) == TensorRole::MatrixWeight);
    }

    // Biases stay exact everywhere, including inside the quantized stages.
    for (const char * name : {
             "decoder.encode.conv1.bias",
             "decoder.generator.conv_post.bias",
             "decoder.generator.resblocks.5.convs1.2.bias",
             "decoder.generator.ups.0.bias",
             "decoder.asr_res.0.bias",
         }) {
        SYNTH_TEST_CHECK(tensor_role(name) == TensorRole::Sensitive);
    }

    // The dense upsamplers are sliced and permuted by this runtime, so they can
    // be halved but not block-quantized.
    SYNTH_TEST_CHECK(tensor_role("decoder.generator.ups.0.weight") == TensorRole::TransposeWeight);
    SYNTH_TEST_CHECK(tensor_role("decoder.generator.ups.1.weight") == TensorRole::TransposeWeight);

    // The depthwise pools are read one tap at a time as a per-channel scale,
    // which only works at the reference dtype, so they are not transpose
    // weights despite also being transposed convolutions.
    SYNTH_TEST_CHECK(tensor_role("decoder.decode.3.pool.weight") == TensorRole::Sensitive);
    SYNTH_TEST_CHECK(tensor_role("predictor.F0.1.pool.weight") == TensorRole::Sensitive);

    // Snake alphas are divided by, so they never leave the reference dtype.
    SYNTH_TEST_CHECK(tensor_role("decoder.generator.resblocks.2.alpha1.0") == TensorRole::Sensitive);
    SYNTH_TEST_CHECK(tensor_role("decoder.generator.noise_res.0.alpha2.2") == TensorRole::Sensitive);

    // These two halve the prosody curves before the decoder reads them, so they
    // sit on the F0 path rather than the audio path.
    SYNTH_TEST_CHECK(tensor_role("decoder.F0_conv.weight") == TensorRole::Sensitive);
    SYNTH_TEST_CHECK(tensor_role("decoder.N_conv.weight") == TensorRole::Sensitive);

    // Nine numbers that feed the phase accumulator.
    SYNTH_TEST_CHECK(tensor_role("decoder.generator.m_source.l_linear.weight") == TensorRole::Sensitive);

    // Anything outside the catalog is unknown rather than defaulted, so a
    // converter or runtime change cannot quietly acquire a storage type.
    for (const char * name : {
             "",
             "decoder",
             "decoder.encode.conv3.weight",
             "decoder.decode.03.conv1.weight",
             "decoder.generator.resblocks.5.convs3.2.weight",
             "decoder.generator.ups.0.scale",
             "decoder.generator.unknown.0.weight",
             "voice",
             "voice.af_heart.extra",
             "unexpected.tensor",
         }) {
        SYNTH_TEST_CHECK(tensor_role(name) == TensorRole::Unknown);
    }
    return 0;
}
