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

#include <set>
#include <string>

using synth::quantize::find_profile;
using synth::quantize::Profile;
using synth::quantize::profile_applies_to_architecture;
using synth::quantize::qwen3_tts_tensor_is_conv_kernel;
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

// ---------------------------------------------------------------------------
// The Base package's 237 tensors, every one by name.
//
// Until Plan 4 Task 6 the quantizer could not cut a Base package AT ALL: these
// 237 names classified Unknown and synthesize-quantize failed on the first one
// it met. The count is exactly the Base/CustomVoice difference, 894 - 657, and
// it is 76 speaker_encoder + 161 codec.encoder.
//
// Listed by name rather than matched by prefix, and the lists below are
// transcribed from the real BF16 package rather than from the catalog, so a
// classifier that recognises a PREFIX while mis-shaping a member of it still
// fails here. The standing example for why this is by-name: a resolver that
// aliased all 31 acoustic codebooks to slot 0 left every name resolved, every
// pointer non-null and a by-name test passing -- only a set insertion caught
// it, which is why the counts below are asserted through std::set.
// kSpeakerEncoderConvWeights: 38
const char * const kSpeakerEncoderConvWeights[] = {
    "speaker_encoder.asp.conv.weight",
    "speaker_encoder.asp.tdnn.conv.weight",
    "speaker_encoder.blocks.0.conv.weight",
    "speaker_encoder.blocks.1.res2net_block.blocks.0.conv.weight",
    "speaker_encoder.blocks.1.res2net_block.blocks.1.conv.weight",
    "speaker_encoder.blocks.1.res2net_block.blocks.2.conv.weight",
    "speaker_encoder.blocks.1.res2net_block.blocks.3.conv.weight",
    "speaker_encoder.blocks.1.res2net_block.blocks.4.conv.weight",
    "speaker_encoder.blocks.1.res2net_block.blocks.5.conv.weight",
    "speaker_encoder.blocks.1.res2net_block.blocks.6.conv.weight",
    "speaker_encoder.blocks.1.se_block.conv1.weight",
    "speaker_encoder.blocks.1.se_block.conv2.weight",
    "speaker_encoder.blocks.1.tdnn1.conv.weight",
    "speaker_encoder.blocks.1.tdnn2.conv.weight",
    "speaker_encoder.blocks.2.res2net_block.blocks.0.conv.weight",
    "speaker_encoder.blocks.2.res2net_block.blocks.1.conv.weight",
    "speaker_encoder.blocks.2.res2net_block.blocks.2.conv.weight",
    "speaker_encoder.blocks.2.res2net_block.blocks.3.conv.weight",
    "speaker_encoder.blocks.2.res2net_block.blocks.4.conv.weight",
    "speaker_encoder.blocks.2.res2net_block.blocks.5.conv.weight",
    "speaker_encoder.blocks.2.res2net_block.blocks.6.conv.weight",
    "speaker_encoder.blocks.2.se_block.conv1.weight",
    "speaker_encoder.blocks.2.se_block.conv2.weight",
    "speaker_encoder.blocks.2.tdnn1.conv.weight",
    "speaker_encoder.blocks.2.tdnn2.conv.weight",
    "speaker_encoder.blocks.3.res2net_block.blocks.0.conv.weight",
    "speaker_encoder.blocks.3.res2net_block.blocks.1.conv.weight",
    "speaker_encoder.blocks.3.res2net_block.blocks.2.conv.weight",
    "speaker_encoder.blocks.3.res2net_block.blocks.3.conv.weight",
    "speaker_encoder.blocks.3.res2net_block.blocks.4.conv.weight",
    "speaker_encoder.blocks.3.res2net_block.blocks.5.conv.weight",
    "speaker_encoder.blocks.3.res2net_block.blocks.6.conv.weight",
    "speaker_encoder.blocks.3.se_block.conv1.weight",
    "speaker_encoder.blocks.3.se_block.conv2.weight",
    "speaker_encoder.blocks.3.tdnn1.conv.weight",
    "speaker_encoder.blocks.3.tdnn2.conv.weight",
    "speaker_encoder.fc.weight",
    "speaker_encoder.mfa.conv.weight",
};

// kSpeakerEncoderBiases: 38
const char * const kSpeakerEncoderBiases[] = {
    "speaker_encoder.asp.conv.bias",
    "speaker_encoder.asp.tdnn.conv.bias",
    "speaker_encoder.blocks.0.conv.bias",
    "speaker_encoder.blocks.1.res2net_block.blocks.0.conv.bias",
    "speaker_encoder.blocks.1.res2net_block.blocks.1.conv.bias",
    "speaker_encoder.blocks.1.res2net_block.blocks.2.conv.bias",
    "speaker_encoder.blocks.1.res2net_block.blocks.3.conv.bias",
    "speaker_encoder.blocks.1.res2net_block.blocks.4.conv.bias",
    "speaker_encoder.blocks.1.res2net_block.blocks.5.conv.bias",
    "speaker_encoder.blocks.1.res2net_block.blocks.6.conv.bias",
    "speaker_encoder.blocks.1.se_block.conv1.bias",
    "speaker_encoder.blocks.1.se_block.conv2.bias",
    "speaker_encoder.blocks.1.tdnn1.conv.bias",
    "speaker_encoder.blocks.1.tdnn2.conv.bias",
    "speaker_encoder.blocks.2.res2net_block.blocks.0.conv.bias",
    "speaker_encoder.blocks.2.res2net_block.blocks.1.conv.bias",
    "speaker_encoder.blocks.2.res2net_block.blocks.2.conv.bias",
    "speaker_encoder.blocks.2.res2net_block.blocks.3.conv.bias",
    "speaker_encoder.blocks.2.res2net_block.blocks.4.conv.bias",
    "speaker_encoder.blocks.2.res2net_block.blocks.5.conv.bias",
    "speaker_encoder.blocks.2.res2net_block.blocks.6.conv.bias",
    "speaker_encoder.blocks.2.se_block.conv1.bias",
    "speaker_encoder.blocks.2.se_block.conv2.bias",
    "speaker_encoder.blocks.2.tdnn1.conv.bias",
    "speaker_encoder.blocks.2.tdnn2.conv.bias",
    "speaker_encoder.blocks.3.res2net_block.blocks.0.conv.bias",
    "speaker_encoder.blocks.3.res2net_block.blocks.1.conv.bias",
    "speaker_encoder.blocks.3.res2net_block.blocks.2.conv.bias",
    "speaker_encoder.blocks.3.res2net_block.blocks.3.conv.bias",
    "speaker_encoder.blocks.3.res2net_block.blocks.4.conv.bias",
    "speaker_encoder.blocks.3.res2net_block.blocks.5.conv.bias",
    "speaker_encoder.blocks.3.res2net_block.blocks.6.conv.bias",
    "speaker_encoder.blocks.3.se_block.conv1.bias",
    "speaker_encoder.blocks.3.se_block.conv2.bias",
    "speaker_encoder.blocks.3.tdnn1.conv.bias",
    "speaker_encoder.blocks.3.tdnn2.conv.bias",
    "speaker_encoder.fc.bias",
    "speaker_encoder.mfa.conv.bias",
};

// kCodecEncoderTensors: 161
const char * const kCodecEncoderTensors[] = {
    "codec.encoder.downsample.conv.weight",
    "codec.encoder.enc_transformer.layers.0.input_layernorm.bias",
    "codec.encoder.enc_transformer.layers.0.input_layernorm.weight",
    "codec.encoder.enc_transformer.layers.0.mlp.fc1.weight",
    "codec.encoder.enc_transformer.layers.0.mlp.fc2.weight",
    "codec.encoder.enc_transformer.layers.0.mlp_scale.scale",
    "codec.encoder.enc_transformer.layers.0.post_attn_norm.bias",
    "codec.encoder.enc_transformer.layers.0.post_attn_norm.weight",
    "codec.encoder.enc_transformer.layers.0.self_attn.k_proj.weight",
    "codec.encoder.enc_transformer.layers.0.self_attn.o_proj.weight",
    "codec.encoder.enc_transformer.layers.0.self_attn.q_proj.weight",
    "codec.encoder.enc_transformer.layers.0.self_attn.v_proj.weight",
    "codec.encoder.enc_transformer.layers.0.self_attn_scale.scale",
    "codec.encoder.enc_transformer.layers.1.input_layernorm.bias",
    "codec.encoder.enc_transformer.layers.1.input_layernorm.weight",
    "codec.encoder.enc_transformer.layers.1.mlp.fc1.weight",
    "codec.encoder.enc_transformer.layers.1.mlp.fc2.weight",
    "codec.encoder.enc_transformer.layers.1.mlp_scale.scale",
    "codec.encoder.enc_transformer.layers.1.post_attn_norm.bias",
    "codec.encoder.enc_transformer.layers.1.post_attn_norm.weight",
    "codec.encoder.enc_transformer.layers.1.self_attn.k_proj.weight",
    "codec.encoder.enc_transformer.layers.1.self_attn.o_proj.weight",
    "codec.encoder.enc_transformer.layers.1.self_attn.q_proj.weight",
    "codec.encoder.enc_transformer.layers.1.self_attn.v_proj.weight",
    "codec.encoder.enc_transformer.layers.1.self_attn_scale.scale",
    "codec.encoder.enc_transformer.layers.2.input_layernorm.bias",
    "codec.encoder.enc_transformer.layers.2.input_layernorm.weight",
    "codec.encoder.enc_transformer.layers.2.mlp.fc1.weight",
    "codec.encoder.enc_transformer.layers.2.mlp.fc2.weight",
    "codec.encoder.enc_transformer.layers.2.mlp_scale.scale",
    "codec.encoder.enc_transformer.layers.2.post_attn_norm.bias",
    "codec.encoder.enc_transformer.layers.2.post_attn_norm.weight",
    "codec.encoder.enc_transformer.layers.2.self_attn.k_proj.weight",
    "codec.encoder.enc_transformer.layers.2.self_attn.o_proj.weight",
    "codec.encoder.enc_transformer.layers.2.self_attn.q_proj.weight",
    "codec.encoder.enc_transformer.layers.2.self_attn.v_proj.weight",
    "codec.encoder.enc_transformer.layers.2.self_attn_scale.scale",
    "codec.encoder.enc_transformer.layers.3.input_layernorm.bias",
    "codec.encoder.enc_transformer.layers.3.input_layernorm.weight",
    "codec.encoder.enc_transformer.layers.3.mlp.fc1.weight",
    "codec.encoder.enc_transformer.layers.3.mlp.fc2.weight",
    "codec.encoder.enc_transformer.layers.3.mlp_scale.scale",
    "codec.encoder.enc_transformer.layers.3.post_attn_norm.bias",
    "codec.encoder.enc_transformer.layers.3.post_attn_norm.weight",
    "codec.encoder.enc_transformer.layers.3.self_attn.k_proj.weight",
    "codec.encoder.enc_transformer.layers.3.self_attn.o_proj.weight",
    "codec.encoder.enc_transformer.layers.3.self_attn.q_proj.weight",
    "codec.encoder.enc_transformer.layers.3.self_attn.v_proj.weight",
    "codec.encoder.enc_transformer.layers.3.self_attn_scale.scale",
    "codec.encoder.enc_transformer.layers.4.input_layernorm.bias",
    "codec.encoder.enc_transformer.layers.4.input_layernorm.weight",
    "codec.encoder.enc_transformer.layers.4.mlp.fc1.weight",
    "codec.encoder.enc_transformer.layers.4.mlp.fc2.weight",
    "codec.encoder.enc_transformer.layers.4.mlp_scale.scale",
    "codec.encoder.enc_transformer.layers.4.post_attn_norm.bias",
    "codec.encoder.enc_transformer.layers.4.post_attn_norm.weight",
    "codec.encoder.enc_transformer.layers.4.self_attn.k_proj.weight",
    "codec.encoder.enc_transformer.layers.4.self_attn.o_proj.weight",
    "codec.encoder.enc_transformer.layers.4.self_attn.q_proj.weight",
    "codec.encoder.enc_transformer.layers.4.self_attn.v_proj.weight",
    "codec.encoder.enc_transformer.layers.4.self_attn_scale.scale",
    "codec.encoder.enc_transformer.layers.5.input_layernorm.bias",
    "codec.encoder.enc_transformer.layers.5.input_layernorm.weight",
    "codec.encoder.enc_transformer.layers.5.mlp.fc1.weight",
    "codec.encoder.enc_transformer.layers.5.mlp.fc2.weight",
    "codec.encoder.enc_transformer.layers.5.mlp_scale.scale",
    "codec.encoder.enc_transformer.layers.5.post_attn_norm.bias",
    "codec.encoder.enc_transformer.layers.5.post_attn_norm.weight",
    "codec.encoder.enc_transformer.layers.5.self_attn.k_proj.weight",
    "codec.encoder.enc_transformer.layers.5.self_attn.o_proj.weight",
    "codec.encoder.enc_transformer.layers.5.self_attn.q_proj.weight",
    "codec.encoder.enc_transformer.layers.5.self_attn.v_proj.weight",
    "codec.encoder.enc_transformer.layers.5.self_attn_scale.scale",
    "codec.encoder.enc_transformer.layers.6.input_layernorm.bias",
    "codec.encoder.enc_transformer.layers.6.input_layernorm.weight",
    "codec.encoder.enc_transformer.layers.6.mlp.fc1.weight",
    "codec.encoder.enc_transformer.layers.6.mlp.fc2.weight",
    "codec.encoder.enc_transformer.layers.6.mlp_scale.scale",
    "codec.encoder.enc_transformer.layers.6.post_attn_norm.bias",
    "codec.encoder.enc_transformer.layers.6.post_attn_norm.weight",
    "codec.encoder.enc_transformer.layers.6.self_attn.k_proj.weight",
    "codec.encoder.enc_transformer.layers.6.self_attn.o_proj.weight",
    "codec.encoder.enc_transformer.layers.6.self_attn.q_proj.weight",
    "codec.encoder.enc_transformer.layers.6.self_attn.v_proj.weight",
    "codec.encoder.enc_transformer.layers.6.self_attn_scale.scale",
    "codec.encoder.enc_transformer.layers.7.input_layernorm.bias",
    "codec.encoder.enc_transformer.layers.7.input_layernorm.weight",
    "codec.encoder.enc_transformer.layers.7.mlp.fc1.weight",
    "codec.encoder.enc_transformer.layers.7.mlp.fc2.weight",
    "codec.encoder.enc_transformer.layers.7.mlp_scale.scale",
    "codec.encoder.enc_transformer.layers.7.post_attn_norm.bias",
    "codec.encoder.enc_transformer.layers.7.post_attn_norm.weight",
    "codec.encoder.enc_transformer.layers.7.self_attn.k_proj.weight",
    "codec.encoder.enc_transformer.layers.7.self_attn.o_proj.weight",
    "codec.encoder.enc_transformer.layers.7.self_attn.q_proj.weight",
    "codec.encoder.enc_transformer.layers.7.self_attn.v_proj.weight",
    "codec.encoder.enc_transformer.layers.7.self_attn_scale.scale",
    "codec.encoder.encoder.layers.0.conv.bias",
    "codec.encoder.encoder.layers.0.conv.weight",
    "codec.encoder.encoder.layers.1.block.1.conv.bias",
    "codec.encoder.encoder.layers.1.block.1.conv.weight",
    "codec.encoder.encoder.layers.1.block.3.conv.bias",
    "codec.encoder.encoder.layers.1.block.3.conv.weight",
    "codec.encoder.encoder.layers.10.block.1.conv.bias",
    "codec.encoder.encoder.layers.10.block.1.conv.weight",
    "codec.encoder.encoder.layers.10.block.3.conv.bias",
    "codec.encoder.encoder.layers.10.block.3.conv.weight",
    "codec.encoder.encoder.layers.12.conv.bias",
    "codec.encoder.encoder.layers.12.conv.weight",
    "codec.encoder.encoder.layers.14.conv.bias",
    "codec.encoder.encoder.layers.14.conv.weight",
    "codec.encoder.encoder.layers.3.conv.bias",
    "codec.encoder.encoder.layers.3.conv.weight",
    "codec.encoder.encoder.layers.4.block.1.conv.bias",
    "codec.encoder.encoder.layers.4.block.1.conv.weight",
    "codec.encoder.encoder.layers.4.block.3.conv.bias",
    "codec.encoder.encoder.layers.4.block.3.conv.weight",
    "codec.encoder.encoder.layers.6.conv.bias",
    "codec.encoder.encoder.layers.6.conv.weight",
    "codec.encoder.encoder.layers.7.block.1.conv.bias",
    "codec.encoder.encoder.layers.7.block.1.conv.weight",
    "codec.encoder.encoder.layers.7.block.3.conv.bias",
    "codec.encoder.encoder.layers.7.block.3.conv.weight",
    "codec.encoder.encoder.layers.9.conv.bias",
    "codec.encoder.encoder.layers.9.conv.weight",
    "codec.encoder.quantizer.acoustic_rvq.input_proj.weight",
    "codec.encoder.quantizer.acoustic_rvq.layers.0.codebook",
    "codec.encoder.quantizer.acoustic_rvq.layers.1.codebook",
    "codec.encoder.quantizer.acoustic_rvq.layers.10.codebook",
    "codec.encoder.quantizer.acoustic_rvq.layers.11.codebook",
    "codec.encoder.quantizer.acoustic_rvq.layers.12.codebook",
    "codec.encoder.quantizer.acoustic_rvq.layers.13.codebook",
    "codec.encoder.quantizer.acoustic_rvq.layers.14.codebook",
    "codec.encoder.quantizer.acoustic_rvq.layers.15.codebook",
    "codec.encoder.quantizer.acoustic_rvq.layers.16.codebook",
    "codec.encoder.quantizer.acoustic_rvq.layers.17.codebook",
    "codec.encoder.quantizer.acoustic_rvq.layers.18.codebook",
    "codec.encoder.quantizer.acoustic_rvq.layers.19.codebook",
    "codec.encoder.quantizer.acoustic_rvq.layers.2.codebook",
    "codec.encoder.quantizer.acoustic_rvq.layers.20.codebook",
    "codec.encoder.quantizer.acoustic_rvq.layers.21.codebook",
    "codec.encoder.quantizer.acoustic_rvq.layers.22.codebook",
    "codec.encoder.quantizer.acoustic_rvq.layers.23.codebook",
    "codec.encoder.quantizer.acoustic_rvq.layers.24.codebook",
    "codec.encoder.quantizer.acoustic_rvq.layers.25.codebook",
    "codec.encoder.quantizer.acoustic_rvq.layers.26.codebook",
    "codec.encoder.quantizer.acoustic_rvq.layers.27.codebook",
    "codec.encoder.quantizer.acoustic_rvq.layers.28.codebook",
    "codec.encoder.quantizer.acoustic_rvq.layers.29.codebook",
    "codec.encoder.quantizer.acoustic_rvq.layers.3.codebook",
    "codec.encoder.quantizer.acoustic_rvq.layers.30.codebook",
    "codec.encoder.quantizer.acoustic_rvq.layers.4.codebook",
    "codec.encoder.quantizer.acoustic_rvq.layers.5.codebook",
    "codec.encoder.quantizer.acoustic_rvq.layers.6.codebook",
    "codec.encoder.quantizer.acoustic_rvq.layers.7.codebook",
    "codec.encoder.quantizer.acoustic_rvq.layers.8.codebook",
    "codec.encoder.quantizer.acoustic_rvq.layers.9.codebook",
    "codec.encoder.quantizer.acoustic_rvq.output_proj.weight",
    "codec.encoder.quantizer.semantic_rvq.input_proj.weight",
    "codec.encoder.quantizer.semantic_rvq.layers.0.codebook",
    "codec.encoder.quantizer.semantic_rvq.output_proj.weight",
};

// totals: 38 + 38 + 161 = 237

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
             // The width bridge a package carries only when the predictor's
             // hidden size differs from the talker's -- absent from every
             // 0.6B package (Base, CustomVoice), present on the 1.7B
             // VoiceDesign checkpoint this task's own quantization run first
             // met. Real name, from src/arch/qwen3-tts/catalog.cpp's own
             // resolution of it (Role::Matrix) and confirmed against a live
             // `--quant F16` run of the VoiceDesign package, which refused
             // with "unknown qwen3-tts tensor" on this exact name before this
             // classifier arm existed.
             "talker.code_predictor.small_to_mtp_projection.weight",
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
             // small_to_mtp_projection's own bias, the Sensitive half of the
             // pair added above.
             "talker.code_predictor.small_to_mtp_projection.bias",
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

    // ---- the Base package's 237, by name --------------------------------

    // The 38 speaker-encoder convolution WEIGHTS are held at the profile's
    // halved fallback -- F16 under all three -- at native three-axis shape.
    //
    // Never the block type and never packed, and that is arithmetic. A block
    // runs along ne[0], which for these is the KERNEL extent: 1, 3 and 5 across
    // the 38, against Q8_0's block of 32 and Q5_K's super-block of 256. The
    // quantizer's own row-size check refuses every one of them, so "Q8_0 at
    // native shape" is not a worse option here, it is an impossible one. The
    // packed [kernel * in, out] alternative is what Kokoro emits, and this
    // family implements neither half of it.
    std::set<std::string> speaker_weight_names;
    for (const char * name : kSpeakerEncoderConvWeights) {
        SYNTH_TEST_CHECK(speaker_weight_names.insert(name).second);
        SYNTH_TEST_CHECK(resolve(*f16, name).type == GGML_TYPE_F16);
        SYNTH_TEST_CHECK(resolve(*q8, name).type == GGML_TYPE_F16);
        SYNTH_TEST_CHECK(resolve(*q5, name).type == GGML_TYPE_F16);
        SYNTH_TEST_CHECK(resolve(*f16, name).layout == TensorLayout::Native);
        SYNTH_TEST_CHECK(resolve(*q8, name).layout == TensorLayout::Native);
        SYNTH_TEST_CHECK(resolve(*q5, name).layout == TensorLayout::Native);
    }
    SYNTH_TEST_CHECK(speaker_weight_names.size() == 38);

    // Their 38 biases are per-channel vectors and stay F32, which is what
    // src/arch/qwen3-tts/catalog.cpp's Resolver::conv already expects for them.
    std::set<std::string> speaker_bias_names;
    for (const char * name : kSpeakerEncoderBiases) {
        SYNTH_TEST_CHECK(speaker_bias_names.insert(name).second);
        SYNTH_TEST_CHECK(resolve(*f16, name).type == GGML_TYPE_F32);
        SYNTH_TEST_CHECK(resolve(*q8, name).type == GGML_TYPE_F32);
        SYNTH_TEST_CHECK(resolve(*q5, name).type == GGML_TYPE_F32);
        SYNTH_TEST_CHECK(resolve(*q8, name).layout == TensorLayout::Native);
    }
    SYNTH_TEST_CHECK(speaker_bias_names.size() == 38);

    // All 161 codec.encoder tensors stay F32 under every profile, which is the
    // same answer the runtime already gives them: src/arch/qwen3-tts/catalog.cpp
    // splits halves on the `codec.` prefix, so the encoder was never on the
    // disagreeing side of the contradiction Task 6 settled. What was missing was
    // RECOGNITION -- they classified Unknown and stopped the tool.
    std::set<std::string> codec_encoder_names;
    for (const char * name : kCodecEncoderTensors) {
        SYNTH_TEST_CHECK(codec_encoder_names.insert(name).second);
        SYNTH_TEST_CHECK(resolve(*f16, name).type == GGML_TYPE_F32);
        SYNTH_TEST_CHECK(resolve(*q8, name).type == GGML_TYPE_F32);
        SYNTH_TEST_CHECK(resolve(*q5, name).type == GGML_TYPE_F32);
        SYNTH_TEST_CHECK(resolve(*q8, name).layout == TensorLayout::Native);
    }
    SYNTH_TEST_CHECK(codec_encoder_names.size() == 161);

    // 76 + 161 = 237 = 894 - 657, the Base/CustomVoice tensor difference, and
    // all distinct.
    std::set<std::string> all_new_names;
    all_new_names.insert(speaker_weight_names.begin(), speaker_weight_names.end());
    all_new_names.insert(speaker_bias_names.begin(), speaker_bias_names.end());
    all_new_names.insert(codec_encoder_names.begin(), codec_encoder_names.end());
    SYNTH_TEST_CHECK(all_new_names.size() == 237);

    // Recognition under the two new prefixes is by name, not by prefix: a
    // plausible-looking tensor that is not one of the 237 is still an error.
    // Without this, a classifier could pass everything above by returning a
    // role for anything starting `speaker_encoder.`.
    SYNTH_TEST_CHECK(!resolve_qwen3_tts_target_spec(*q8, "speaker_encoder.nonsense.weight", ignored));
    SYNTH_TEST_CHECK(!resolve_qwen3_tts_target_spec(*q8, "speaker_encoder.blocks.0.conv.gamma", ignored));
    SYNTH_TEST_CHECK(!resolve_qwen3_tts_target_spec(*q8, "speaker_encoder.blocks.0.not_conv.weight", ignored));
    SYNTH_TEST_CHECK(!resolve_qwen3_tts_target_spec(*q8, "speaker_encoder.blocks.x.conv.weight", ignored));
    SYNTH_TEST_CHECK(!resolve_qwen3_tts_target_spec(*q8, "codec.encoder.nonsense.weight", ignored));
    SYNTH_TEST_CHECK(
        !resolve_qwen3_tts_target_spec(*q8, "codec.encoder.quantizer.mystery_rvq.input_proj.weight", ignored));
    SYNTH_TEST_CHECK(
        !resolve_qwen3_tts_target_spec(*q8, "codec.encoder.enc_transformer.layers.0.mlp.fc3.weight", ignored));

    // The halved fallback the ConvKernel role reads is the profile's
    // `transpose_weight_type` column, and for all three halved profiles that is
    // F16. Asserted against the profile table rather than restated, so a row
    // that changed would fail here instead of silently moving 38 tensors.
    SYNTH_TEST_CHECK(f16->transpose_weight_type == GGML_TYPE_F16);
    SYNTH_TEST_CHECK(q8->transpose_weight_type == GGML_TYPE_F16);
    SYNTH_TEST_CHECK(q5->transpose_weight_type == GGML_TYPE_F16);

    // And the arithmetic that forces it: no speaker-encoder kernel extent
    // clears either block size, so a native block quantization of these is
    // impossible rather than merely undesirable.
    for (const int64_t kernel : { int64_t(1), int64_t(3), int64_t(5) }) {
        SYNTH_TEST_CHECK(kernel % ggml_blck_size(GGML_TYPE_Q8_0) != 0);
        SYNTH_TEST_CHECK(kernel % ggml_blck_size(GGML_TYPE_Q5_K) != 0);
    }

    // BF16 IS NOT A CUT TARGET FOR THIS FAMILY, and the tool has to say so
    // before it reads a tensor. Every column disagrees with the runtime here,
    // not just the ConvKernel one: the profile row's sensitive column is F32
    // while src/arch/qwen3-tts/catalog.cpp holds the whole talker half at BF16
    // regardless of role. Measured 2026-08-17 -- a `--quant BF16` cut of the
    // shipped CustomVoice package succeeded, wrote 2.2 GB, and was rejected at
    // load on `talker.text_projection.linear_fc1.bias`, a Sensitive tensor on a
    // package that holds no ConvKernel tensors at all.
    const Profile * bf16 = find_profile("BF16");
    SYNTH_TEST_CHECK(bf16 != nullptr);
    SYNTH_TEST_CHECK(bf16->sensitive_type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(bf16->transpose_weight_type == GGML_TYPE_F32);
    std::string reason;
    SYNTH_TEST_CHECK(!profile_applies_to_architecture("qwen3-tts", *bf16, reason));
    SYNTH_TEST_CHECK(!reason.empty());

    // The three profiles this family DOES cut are unaffected, so the guard
    // cannot be passing by refusing everything.
    for (const Profile * profile : { f16, q8, q5 }) {
        SYNTH_TEST_CHECK(profile_applies_to_architecture("qwen3-tts", *profile, reason));
        SYNTH_TEST_CHECK(reason.empty());
    }

    // And BF16 stays available where it means something: it is one of
    // OmniVoice's four generator-half profiles. A guard keyed on the profile
    // alone rather than on the pair would have taken that away.
    SYNTH_TEST_CHECK(profile_applies_to_architecture("omnivoice", *bf16, reason));
    SYNTH_TEST_CHECK(reason.empty());

    // Design spec section 7: "a package with no speaker encoder resolves no
    // ConvKernel". Asserted on the ROLE, not on the resolved type: under F16
    // the ConvKernel column (profile.transpose_weight_type) and the
    // MatrixWeight column are both GGML_TYPE_F16, so a type-only check cannot
    // tell a misclassified tensor from a correct one and would pass on the
    // bug it exists to catch.
    //
    // The names below are real tensor names from the shipped CustomVoice
    // package (reports/porting/qwen3-tts/qwen3-tts-12hz-0-6b-customvoice/intake.json
    // and src/arch/qwen3-tts/catalog.cpp), covering a talker block, the
    // talker's own output head, a code predictor block, the code predictor's
    // per-group output head, and the codec decoder's input and output
    // convolutions -- no speaker_encoder.* tensor anywhere. VoiceDesign
    // carries this same shape; Base is the only variant with a speaker
    // encoder.
    for (const char * name : {
             "talker.model.layers.0.self_attn.q_proj.weight",
             "talker.model.layers.12.mlp.gate_proj.weight",
             "talker.codec_head.weight",
             "talker.code_predictor.model.layers.2.self_attn.k_proj.weight",
             "talker.code_predictor.lm_head.3.weight",
             "codec.decoder.decoder.0.conv.weight",
             "codec.decoder.decoder.6.conv.weight",
         }) {
        SYNTH_TEST_CHECK(!qwen3_tts_tensor_is_conv_kernel(name));
    }

    // Three more negatives the loop above does not reach, each failing for a
    // DIFFERENT reason, so a regression in any one of the three guards shows
    // up here rather than hiding behind the others:
    //
    //   - `talker.code_predictor.small_to_mtp_projection.weight` is this
    //     branch's own new classifier arm (src/arch/qwen3-tts/catalog.cpp
    //     resolves it at Role::Matrix; classify_qwen3_talker returns
    //     MatrixWeight). It is a Linear weight that reaches ConvKernel only
    //     if a future edit widens the conv rule past the talker guard. The
    //     positive half of this pair is already asserted above -- the arm
    //     resolves to Q8_0/Q5_K/F16 under the three profiles -- so this is
    //     the assertion that it is not ALSO taken for a convolution.
    //
    //   - The empty name. split_name("") yields one empty token, which every
    //     one of the three classifiers rejects on its own size or prefix
    //     guard (talker needs >= 2 tokens, speaker_encoder needs tokens[0] to
    //     match, codec needs >= 3). A classifier that indexed before checking
    //     size would fault here rather than return false.
    //
    //   - `speaker_encoder.blocks.x.conv.weight` is a genuine near-miss: it
    //     has the right prefix, the right `.conv.` wrapper segment and the
    //     right `weight` leaf, and differs from the real
    //     `speaker_encoder.blocks.1.conv.weight` stem ONLY in that `x` is not
    //     an index. is_index() rejects it, so classify_qwen3_speaker_encoder
    //     falls through every arm and returns Unknown -- NOT ConvKernel. This
    //     is what keeps the "requiring it is what keeps a future
    //     non-convolution tensor under this prefix an error rather than
    //     something that inherits a convolution's role by position" comment
    //     in policy.cpp true: position alone must not confer the role.
    //
    // MUTATION-TESTER'S NOTE, on the order of the two guards for that last
    // name. `speaker_encoder.blocks.x.conv.weight` is ALREADY asserted
    // earlier in this same test, through resolve_qwen3_tts_target_spec in
    // the "plausible-looking tensor that is not one of the 237" block. That
    // assertion runs FIRST, so a mutation to is_index() kills the test there
    // and never reaches this line -- crediting the kill to this assertion is
    // a misattribution (PR #16's fix round made exactly that mistake). This
    // assertion is independently load-bearing only once the earlier one is
    // neutralized, and it tests a different predicate: the earlier block
    // asserts the name RESOLVES to nothing, this one asserts it is not
    // classified ConvKernel specifically. Both are wanted; neutralize the
    // earlier one before mutation-testing this one.
    for (const char * name : {
             "talker.code_predictor.small_to_mtp_projection.weight",
             "",
             "speaker_encoder.blocks.x.conv.weight",
         }) {
        SYNTH_TEST_CHECK(!qwen3_tts_tensor_is_conv_kernel(name));
    }

    // The control, a real speaker-encoder convolution weight from
    // kSpeakerEncoderConvWeights above. Without it the loop above passes on a
    // build where the function always returns false, which is the same
    // failure the loop is written to catch, spelled the other way.
    SYNTH_TEST_CHECK(qwen3_tts_tensor_is_conv_kernel("speaker_encoder.blocks.1.tdnn1.conv.weight"));

    return 0;
}
