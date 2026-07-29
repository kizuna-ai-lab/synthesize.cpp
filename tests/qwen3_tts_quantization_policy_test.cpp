// The Qwen3-TTS quantization split.
//
// The package has two halves that quantize very differently. The
// autoregressive half -- talker and code predictor, 1728 MiB of a 2164 MiB
// package -- is entirely two-dimensional matrices with rows of 1024, 2048 and
// 3072, so it takes a block quantization without argument. The codec half is
// convolutions whose fastest dimension is the kernel: rows of 7, 16, 3 and 1,
// which no block size divides.
//
// The codec is therefore held at the reference dtype under every profile, and
// that is a measurement rather than caution -- halving it made the codec 1.75
// times slower, because its convolutions run through im2col into a matrix
// multiply where ggml's F16 path is slower than its F32 one.
//
// This test pins the split. It exists because Q8_MIXED was refused outright for
// this family on the belief that its convolution kernels would be packed and
// could not be; they are never packed, because they never reach a quantized
// type at all. Nothing failed when that belief was wrong, because nothing
// checked it.

#include "ggml.h"
#include "policy.h"
#include "test-assert.h"

#include <string>

using synth::quantize::find_profile;
using synth::quantize::Profile;
using synth::quantize::resolve_qwen3_tts_target_spec;
using synth::quantize::TargetSpec;
using synth::quantize::TensorLayout;

namespace {

// An unresolved name yields a spec that matches nothing the tests assert, so a
// classifier that stops recognising a tensor fails on the type check rather than
// passing silently.
TargetSpec resolve(const Profile & profile, const char * name) {
    TargetSpec spec{};
    if (!resolve_qwen3_tts_target_spec(profile, name, spec)) {
        return TargetSpec{ GGML_TYPE_COUNT, TensorLayout::PackedMatrix };
    }
    return spec;
}

}  // namespace

int main() {
    const Profile * f16 = find_profile("F16");
    const Profile * q8  = find_profile("Q8_MIXED");
    const Profile * q5  = find_profile("Q5_K_MIXED");
    SYNTH_TEST_CHECK(f16 != nullptr && q8 != nullptr && q5 != nullptr);

    // The autoregressive half quantizes, and takes the profile's own type.
    //
    // Only the type is asserted here. This classifier sees a name and not a
    // tensor, so the layout it returns is the profile's default and the decision
    // that matters -- a two-dimensional weight is never packed, because packing
    // would flatten a [1024, 3072] projection into one row of 3145728 -- is made
    // in the quantizer, which has the dimensions. Reading this layer's layout as
    // final is precisely the mistake that had Q8_MIXED refused for this family.
    for (const char * name : {
             "talker.model.text_embedding.weight",
             "talker.model.layers.0.self_attn.q_proj.weight",
             "talker.model.layers.13.self_attn.k_proj.weight",
             "talker.model.layers.27.mlp.down_proj.weight",
             "talker.code_predictor.model.layers.0.mlp.gate_proj.weight",
             "talker.code_predictor.lm_head.7.weight",
             "talker.code_predictor.model.codec_embedding.3.weight",
         }) {
        SYNTH_TEST_CHECK(resolve(*q8, name).type == GGML_TYPE_Q8_0);
        SYNTH_TEST_CHECK(resolve(*q5, name).type == GGML_TYPE_Q5_K);
        SYNTH_TEST_CHECK(resolve(*f16, name).type == GGML_TYPE_F16);
    }

    // Norms and biases stay at the reference dtype in both halves: they are
    // one-dimensional, and a block quantization has nothing to divide.
    for (const char * name : {
             "talker.model.norm.weight",
             "talker.model.layers.5.input_layernorm.weight",
             "talker.model.layers.5.self_attn.q_norm.weight",
             "talker.code_predictor.model.layers.1.post_attn_norm.weight",
         }) {
        SYNTH_TEST_CHECK(resolve(*q8, name).type == GGML_TYPE_F32);
        SYNTH_TEST_CHECK(resolve(*q5, name).type == GGML_TYPE_F32);
        SYNTH_TEST_CHECK(resolve(*f16, name).type == GGML_TYPE_F32);
    }

    // The codec half never quantizes and never halves, under either profile.
    // The convolutions are the reason the kernel cannot take a block; the
    // quantizer's codebooks are a separate reason, since a residual level's
    // later entries carry small magnitudes and a relative error there is a large
    // one against the residual it corrects.
    for (const char * name : {
             "codec.decoder.decoder.0.conv.weight",
             "codec.decoder.decoder.1.block.1.conv.weight",
             "codec.decoder.decoder.1.block.2.conv1.conv.weight",
             "codec.decoder.upsample.0.1.pwconv1.weight",
             "codec.decoder.upsample.0.1.pwconv2.weight",
             "codec.decoder.pre_conv.conv.weight",
             "codec.decoder.quantizer.rvq_first.vq.layers.0.codebook",
         }) {
        SYNTH_TEST_CHECK(resolve(*q8, name).type == GGML_TYPE_F32);
        SYNTH_TEST_CHECK(resolve(*q8, name).layout == TensorLayout::Native);
        SYNTH_TEST_CHECK(resolve(*q5, name).type == GGML_TYPE_F32);
        SYNTH_TEST_CHECK(resolve(*f16, name).type == GGML_TYPE_F32);
    }

    // Q5_K needs a row divisible by 256 where Q8_0 needs 32. Everything this
    // family quantizes clears that -- rows of 1024, 2048 and 3072 -- which is
    // why the profile exists for it at all and why a family whose matrix weights
    // are packed convolution kernels cannot take it.
    SYNTH_TEST_CHECK(ggml_blck_size(GGML_TYPE_Q5_K) == 256);
    SYNTH_TEST_CHECK(ggml_blck_size(GGML_TYPE_Q8_0) == 32);
    for (const int64_t row : { int64_t(1024), int64_t(2048), int64_t(3072) }) {
        SYNTH_TEST_CHECK(row % ggml_blck_size(GGML_TYPE_Q5_K) == 0);
    }

    // A name outside the catalog is an error rather than a default. The runtime
    // refuses any tensor the catalog does not resolve, so a silent fallback here
    // would produce a package it then declines to load.
    TargetSpec ignored{};
    SYNTH_TEST_CHECK(!resolve_qwen3_tts_target_spec(*q8, "talker.model.layers.0.not_a_tensor", ignored));
    SYNTH_TEST_CHECK(!resolve_qwen3_tts_target_spec(*q8, "codec.decoder.nonsense.weight", ignored));
    SYNTH_TEST_CHECK(!resolve_qwen3_tts_target_spec(*q8, "", ignored));

    return 0;
}
