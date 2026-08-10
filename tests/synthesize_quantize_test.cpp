#include "ggml.h"
#include "gguf.h"
#include "quantize.h"
#include "test-assert.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

bool write_fixture(const std::string & path, bool include_unknown) {
    ggml_init_params params{};
    params.mem_size    = 1024 * 1024;
    params.no_alloc    = false;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }
    gguf_context * gguf = gguf_init_empty();
    if (gguf == nullptr) {
        ggml_free(ctx);
        return false;
    }
    gguf_set_val_str(gguf, "general.architecture", "vits");
    gguf_set_val_u32(gguf, "general.file_type", 0);
    gguf_set_val_str(gguf, "synthesize.quantization.profile", "F32");
    gguf_set_val_u32(gguf, "synthesize.quantization.profile_version", 1);

    auto add = [&](const char * name, int64_t ne0, int64_t ne1) {
        ggml_tensor * tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
        ggml_set_name(tensor, name);
        auto * values = static_cast<float *>(tensor->data);
        for (int64_t i = 0; i < ne0 * ne1; ++i) {
            values[i] = static_cast<float>(i + 1) / 7.0f;
        }
        gguf_add_tensor(gguf, tensor);
    };
    auto add3 = [&](const char * name, int64_t ne0, int64_t ne1, int64_t ne2) {
        ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2);
        ggml_set_name(tensor, name);
        auto * values = static_cast<float *>(tensor->data);
        for (int64_t i = 0; i < ne0 * ne1 * ne2; ++i) {
            values[i] = static_cast<float>(i + 1) / 7.0f;
        }
        gguf_add_tensor(gguf, tensor);
    };

    add("voice.embedding.weight", 4, 3);
    add("text_encoder.blocks.0.attention_norm.weight", 4, 1);
    add("duration_predictor.affine.log_scale", 1, 2);
    add3("decoder.post.weight", 7, 32, 1);
    add3("decoder.upsample.0.transpose_conv.weight", 8, 2, 4);
    if (include_unknown) {
        add("decoder.future_module.weight", 4, 4);
    }

    const bool ok = gguf_write_to_file(gguf, path.c_str(), false);
    gguf_free(gguf);
    ggml_free(ctx);
    return ok;
}

// Task 2 (Plan 4): a minimal omnivoice package exercising every role
// resolve_omnivoice_target_spec's dispatch can return, plus the one hazard
// shape (`codec.acoustic_decoder.conv2.weight`, [7, 32, 1]) that decides
// whether omnivoice belongs in quantize.cpp's matrix_family demotion list.
// Unlike VITS's own decoder.post.weight (the same [7, 32, 1] shape, and the
// reason that check exists at all), omnivoice also has genuine 2-D Linears
// among its MatrixWeight tensors (q_proj here stands in for all of HuBERT's
// attention/feed-forward projections), so it cannot simply sit outside
// matrix_family the way VITS does -- q_proj would otherwise be packed into a
// meaningless single row. It is IN matrix_family, with conv2.weight named
// out of the demotion instead; see quantize.cpp's own comment for why.
bool write_omnivoice_fixture(const std::string & path, bool include_unknown) {
    ggml_init_params params{};
    params.mem_size    = 1024 * 1024;
    params.no_alloc    = false;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }
    gguf_context * gguf = gguf_init_empty();
    if (gguf == nullptr) {
        ggml_free(ctx);
        return false;
    }
    gguf_set_val_str(gguf, "general.architecture", "omnivoice");
    gguf_set_val_u32(gguf, "general.file_type", 0);
    gguf_set_val_str(gguf, "synthesize.quantization.profile", "F32");
    gguf_set_val_u32(gguf, "synthesize.quantization.profile_version", 1);

    auto add = [&](const char * name, int64_t ne0, int64_t ne1) {
        ggml_tensor * tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
        ggml_set_name(tensor, name);
        auto * values = static_cast<float *>(tensor->data);
        for (int64_t i = 0; i < ne0 * ne1; ++i) {
            values[i] = static_cast<float>(i + 1) / 7.0f;
        }
        gguf_add_tensor(gguf, tensor);
    };
    auto add1 = [&](const char * name, int64_t ne0) {
        ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
        ggml_set_name(tensor, name);
        auto * values = static_cast<float *>(tensor->data);
        for (int64_t i = 0; i < ne0; ++i) {
            values[i] = static_cast<float>(i + 1) / 7.0f;
        }
        gguf_add_tensor(gguf, tensor);
    };
    auto add3 = [&](const char * name, int64_t ne0, int64_t ne1, int64_t ne2) {
        ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2);
        ggml_set_name(tensor, name);
        auto * values = static_cast<float *>(tensor->data);
        for (int64_t i = 0; i < ne0 * ne1 * ne2; ++i) {
            values[i] = static_cast<float>(i + 1) / 7.0f;
        }
        gguf_add_tensor(gguf, tensor);
    };

    // The generator half. Widths are 256 because that is Q4_K's super-block
    // and Q4_K quantizes these; Q8_0's block of 32 divides it, so the same
    // rows serve Q8. The real package's rows are 1024, 2048 and 3072,
    // which clear both with room to spare -- a narrower fixture would let a
    // Q4_K profile fail the row-size check here for a reason the real package
    // never has.
    add1("llm.norm.weight", 4);              // a norm: Sensitive under every profile
    add("llm.embed_tokens.weight", 256, 3);  // RowLookup: never packed
    add("llm.layers.0.self_attn.q_proj.weight", 256, 3);
    add1("llm.layers.0.self_attn.q_norm.weight", 4);
    add("audio_embeddings.weight", 256, 3);  // the second RowLookup table
    add("audio_heads.weight", 256, 3);       // same shape, but a plain mul_mat

    // A genuine 2-D Linear (HuBERT's attention projections stand in for the
    // whole 73-tensor group, and since the conv-exempt policy they are the
    // *only* codec tensors a profile still block-quantizes): MatrixWeight,
    // but must land Native, not packed -- the reason omnivoice needs
    // matrix_family at all.
    add("codec.semantic_model.encoder.layers.0.attn.q_proj.weight", 32, 3);
    add1("codec.semantic_model.encoder.layers.0.attn.q_proj.bias", 3);

    // An ordinary 3-D convolution kernel: ConvKernel since the conv-exempt
    // policy of 2026-08-09, so halved at its native shape rather than packed.
    add3("codec.acoustic_decoder.conv1.weight", 3, 32, 4);
    add1("codec.acoustic_decoder.conv1.bias", 4);

    // The hazard shape: out_channels == 1 collapses ggml_n_dims to 2, so this
    // convolution kernel is indistinguishable from a Linear by shape alone --
    // the case that decides the conv-exempt rule must classify by name. It
    // must come out F16 and unpacked like every other convolution, not Q8_0.
    add3("codec.acoustic_decoder.conv2.weight", 7, 32, 1);
    add1("codec.acoustic_decoder.conv2.bias", 1);

    // TransposeWeight: the same F32-at-every-profile override VITS and
    // Qwen3-TTS's decoder make, not Kokoro's halved type.
    add3("codec.acoustic_decoder.block.0.conv_t1.weight", 4, 3, 2);
    add1("codec.acoustic_decoder.block.0.conv_t1.bias", 3);

    if (include_unknown) {
        // Three tensors, and only two of them refuse anything. The
        // differences are worth naming because none of them is visible from
        // the tensor names.
        //
        // The first is NOT unknown to the classifier and no longer refuses
        // anything at all. Anything ending `.weight` under one of the four
        // codec modules is recognised by design -- that region is
        // deliberately permissive, with its exceptions named individually
        // (quantization.cpp). Before the conv-exempt policy it was
        // MatrixWeight and a codec-half profile stopped on its row size;
        // now it is not one of the seven whitelisted semantic-model Linear
        // module names, so it is ConvKernel, halved at its native shape, and
        // every profile walks past it. That is the policy failing safe: a
        // codec weight nobody anticipated is not block-quantized on the
        // strength of a catch-all.
        add("codec.acoustic_decoder.future_module.weight", 4, 4);
        // The second is a genuine semantic-model Linear with a row no block
        // divides: 17 is not a whole number of Q8_0's 32. It is what still
        // exercises the row-size refusal under a codec-half profile now that
        // convolutions never reach it. A generator-half profile holds it at
        // F32 and walks past.
        add("codec.semantic_model.encoder.layers.0.ff.inter_dense.weight", 17, 3);
        // The third genuinely is unknown -- `codec.unknown_module` matches
        // no module -- so it stops every profile, whichever half it lands in.
        add("codec.unknown_module.conv.weight", 32, 4);
    }

    const bool ok = gguf_write_to_file(gguf, path.c_str(), false);
    gguf_free(gguf);
    ggml_free(ctx);
    return ok;
}

std::vector<char> read_bytes(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    return { std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>() };
}

float tensor_value(const ggml_tensor * tensor, int64_t index) {
    if (tensor->type == GGML_TYPE_F32) {
        return static_cast<const float *>(tensor->data)[index];
    }
    if (tensor->type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(static_cast<const ggml_fp16_t *>(tensor->data)[index]);
    }
    return NAN;
}

int check_output(const std::string & path) {
    ggml_context *   ctx = nullptr;
    gguf_init_params params{};
    params.no_alloc     = false;
    params.ctx          = &ctx;
    gguf_context * gguf = gguf_init_from_file(path.c_str(), params);
    SYNTH_TEST_CHECK(gguf != nullptr);
    SYNTH_TEST_CHECK(ctx != nullptr);

    const int64_t file_type = gguf_find_key(gguf, "general.file_type");
    const int64_t profile   = gguf_find_key(gguf, "synthesize.quantization.profile");
    const int64_t version   = gguf_find_key(gguf, "synthesize.quantization.profile_version");
    SYNTH_TEST_CHECK(file_type >= 0);
    SYNTH_TEST_CHECK(profile >= 0);
    SYNTH_TEST_CHECK(version >= 0);
    SYNTH_TEST_CHECK(gguf_get_val_u32(gguf, file_type) == 1);
    SYNTH_TEST_CHECK(std::strcmp(gguf_get_val_str(gguf, profile), "F16") == 0);
    SYNTH_TEST_CHECK(gguf_get_val_u32(gguf, version) == 1);

    ggml_tensor * embedding = ggml_get_tensor(ctx, "voice.embedding.weight");
    ggml_tensor * norm      = ggml_get_tensor(ctx, "text_encoder.blocks.0.attention_norm.weight");
    ggml_tensor * affine    = ggml_get_tensor(ctx, "duration_predictor.affine.log_scale");
    ggml_tensor * post      = ggml_get_tensor(ctx, "decoder.post.weight");
    ggml_tensor * transpose = ggml_get_tensor(ctx, "decoder.upsample.0.transpose_conv.weight");
    SYNTH_TEST_CHECK(embedding != nullptr && embedding->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(norm != nullptr && norm->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(affine != nullptr && affine->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(post != nullptr && post->type == GGML_TYPE_F16);
    SYNTH_TEST_CHECK(post->ne[0] == 7 && post->ne[1] == 32 && post->ne[2] == 1);
    SYNTH_TEST_CHECK(transpose != nullptr && transpose->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(tensor_value(embedding, 6) == 1.0f);
    SYNTH_TEST_CHECK(tensor_value(norm, 0) == 1.0f / 7.0f);

    gguf_free(gguf);
    ggml_free(ctx);
    return 0;
}

int check_q8_output(const std::string & path) {
    ggml_context *   ctx = nullptr;
    gguf_init_params params{};
    params.no_alloc     = false;
    params.ctx          = &ctx;
    gguf_context * gguf = gguf_init_from_file(path.c_str(), params);
    SYNTH_TEST_CHECK(gguf != nullptr && ctx != nullptr);

    const int64_t file_type = gguf_find_key(gguf, "general.file_type");
    const int64_t profile   = gguf_find_key(gguf, "synthesize.quantization.profile");
    const int64_t version   = gguf_find_key(gguf, "synthesize.quantization.profile_version");
    SYNTH_TEST_CHECK(file_type >= 0 && gguf_get_val_u32(gguf, file_type) == GGML_FTYPE_MOSTLY_Q8_0);
    SYNTH_TEST_CHECK(profile >= 0 && std::strcmp(gguf_get_val_str(gguf, profile), "Q8_MIXED") == 0);
    SYNTH_TEST_CHECK(version >= 0 && gguf_get_val_u32(gguf, version) == 1);

    ggml_tensor * post      = ggml_get_tensor(ctx, "decoder.post.weight");
    ggml_tensor * transpose = ggml_get_tensor(ctx, "decoder.upsample.0.transpose_conv.weight");
    ggml_tensor * embedding = ggml_get_tensor(ctx, "voice.embedding.weight");
    SYNTH_TEST_CHECK(post != nullptr && post->type == GGML_TYPE_Q8_0);
    SYNTH_TEST_CHECK(post->ne[0] == 7 * 32 && post->ne[1] == 1 && post->ne[2] == 1);
    SYNTH_TEST_CHECK(transpose != nullptr && transpose->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(transpose->ne[0] == 8 && transpose->ne[1] == 2 && transpose->ne[2] == 4);
    SYNTH_TEST_CHECK(embedding != nullptr && embedding->type == GGML_TYPE_F32);

    const ggml_type_traits * traits = ggml_get_type_traits(post->type);
    SYNTH_TEST_CHECK(traits != nullptr && traits->to_float != nullptr);
    std::vector<float> dequantized(static_cast<size_t>(post->ne[0]));
    traits->to_float(post->data, dequantized.data(), post->ne[0]);
    const int64_t probe = 100;
    SYNTH_TEST_CHECK(std::fabs(dequantized[probe] - static_cast<float>(probe + 1) / 7.0f) < 0.2f);

    gguf_free(gguf);
    ggml_free(ctx);
    return 0;
}

// Task 2 (Plan 4): the dispatch wiring and the matrix_family demotion
// decision, checked against the fixture write_omnivoice_fixture builds.
int check_omnivoice_q8_output(const std::string & path) {
    ggml_context *   ctx = nullptr;
    gguf_init_params params{};
    params.no_alloc     = false;
    params.ctx          = &ctx;
    gguf_context * gguf = gguf_init_from_file(path.c_str(), params);
    SYNTH_TEST_CHECK(gguf != nullptr && ctx != nullptr);

    const int64_t file_type = gguf_find_key(gguf, "general.file_type");
    const int64_t profile   = gguf_find_key(gguf, "synthesize.quantization.profile");
    SYNTH_TEST_CHECK(file_type >= 0 && gguf_get_val_u32(gguf, file_type) == GGML_FTYPE_MOSTLY_Q8_0);
    SYNTH_TEST_CHECK(profile >= 0 && std::strcmp(gguf_get_val_str(gguf, profile), "Q8_CODEC_MIXED") == 0);

    // The whole generator half is held at F32 by a codec-half profile, and
    // that includes the two tensors a shape-only rule would happily pack.
    for (const char * name : {
             "llm.norm.weight",
             "llm.embed_tokens.weight",
             "llm.layers.0.self_attn.q_proj.weight",
             "llm.layers.0.self_attn.q_norm.weight",
             "audio_embeddings.weight",
             "audio_heads.weight",
         }) {
        ggml_tensor * generator = ggml_get_tensor(ctx, name);
        SYNTH_TEST_CHECK(generator != nullptr && generator->type == GGML_TYPE_F32);
    }

    // A genuine 2-D Linear: MatrixWeight, but must stay Native -- packing it
    // would flatten [32, 3] into a meaningless single row of 96.
    ggml_tensor * linear = ggml_get_tensor(ctx, "codec.semantic_model.encoder.layers.0.attn.q_proj.weight");
    SYNTH_TEST_CHECK(linear != nullptr && linear->type == GGML_TYPE_Q8_0);
    SYNTH_TEST_CHECK(linear->ne[0] == 32 && linear->ne[1] == 3 && linear->ne[2] == 1);
    ggml_tensor * linear_bias = ggml_get_tensor(ctx, "codec.semantic_model.encoder.layers.0.attn.q_proj.bias");
    SYNTH_TEST_CHECK(linear_bias != nullptr && linear_bias->type == GGML_TYPE_F32);

    // An ordinary 3-D convolution kernel. This is the assertion the
    // conv-exempt policy of 2026-08-09 turns on: before it, this came back
    // Q8_0 packed to [kernel * in_channels, out_channels] = [96, 4]; now it
    // is F16 at its declared [3, 32, 4].
    ggml_tensor * conv1 = ggml_get_tensor(ctx, "codec.acoustic_decoder.conv1.weight");
    SYNTH_TEST_CHECK(conv1 != nullptr && conv1->type == GGML_TYPE_F16);
    SYNTH_TEST_CHECK(conv1->ne[0] == 3 && conv1->ne[1] == 32 && conv1->ne[2] == 4);

    // The hazard shape: a convolution kernel whose out_channels == 1
    // collapses ggml_n_dims to 2, exactly what a genuine Linear reports. It
    // must come out F16 and unpacked like every other convolution, which is
    // the case that proves the conv-exempt rule reads the name and not the
    // shape. `ggml_n_dims` collapsing the trailing axis is why its stored
    // rank is 2 while its extents are still [7, 32, 1].
    ggml_tensor * conv2 = ggml_get_tensor(ctx, "codec.acoustic_decoder.conv2.weight");
    SYNTH_TEST_CHECK(conv2 != nullptr && conv2->type == GGML_TYPE_F16);
    SYNTH_TEST_CHECK(conv2->ne[0] == 7 && conv2->ne[1] == 32 && conv2->ne[2] == 1);
    ggml_tensor * conv2_bias = ggml_get_tensor(ctx, "codec.acoustic_decoder.conv2.bias");
    SYNTH_TEST_CHECK(conv2_bias != nullptr && conv2_bias->type == GGML_TYPE_F32);

    // TransposeWeight: the same F32-at-every-profile override VITS and
    // Qwen3-TTS's decoder make, not Kokoro's halved type -- left Native at
    // its own declared 3-D shape, never packed. It is F32 where an ordinary
    // convolution is now F16, which is the one difference the two
    // convolution roles still make.
    ggml_tensor * transpose = ggml_get_tensor(ctx, "codec.acoustic_decoder.block.0.conv_t1.weight");
    SYNTH_TEST_CHECK(transpose != nullptr && transpose->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(transpose->ne[0] == 4 && transpose->ne[1] == 3 && transpose->ne[2] == 2);

    // Round-trip the one tensor this profile still block-quantizes, to prove
    // the bytes are a real quantization of the source values and not merely
    // the right shape and type -- the same proof check_q8_output runs for
    // VITS's own decoder.post.weight.
    const ggml_type_traits * traits = ggml_get_type_traits(linear->type);
    SYNTH_TEST_CHECK(traits != nullptr && traits->to_float != nullptr);
    std::vector<float> dequantized(static_cast<size_t>(linear->ne[0]));
    traits->to_float(linear->data, dequantized.data(), linear->ne[0]);
    const int64_t probe = 20;
    SYNTH_TEST_CHECK(std::fabs(dequantized[probe] - static_cast<float>(probe + 1) / 7.0f) < 0.2f);

    // And round-trip the halved convolution, so "F16" here means the values
    // survived the conversion rather than the tensor merely carrying the
    // type. Its packed predecessor was the tensor this proof used to cover.
    const ggml_type_traits * conv_traits = ggml_get_type_traits(conv2->type);
    SYNTH_TEST_CHECK(conv_traits != nullptr && conv_traits->to_float != nullptr);
    std::vector<float> conv_values(static_cast<size_t>(conv2->ne[0] * conv2->ne[1]));
    conv_traits->to_float(conv2->data, conv_values.data(), conv2->ne[0] * conv2->ne[1]);
    const int64_t conv_probe = 100;
    SYNTH_TEST_CHECK(std::fabs(conv_values[conv_probe] - static_cast<float>(conv_probe + 1) / 7.0f) < 0.2f);

    gguf_free(gguf);
    ggml_free(ctx);
    return 0;
}

// Task 1 (Plan 6): the mirror image, and the assertion that a future edit to
// quantize.cpp's two-dimensional demotion cannot silently start packing the
// generator. Every tensor Q8 quantizes is two-dimensional, so if that
// demotion stopped applying to omnivoice, `q_proj` here would come back as a
// single packed row of 96 instead of [32, 3] -- shaped nothing like what
// catalog.cpp resolves and nothing like what `ggml_mul_mat` wants.
int check_omnivoice_q8_output_generator(const std::string & path) {
    ggml_context *   ctx = nullptr;
    gguf_init_params params{};
    params.no_alloc     = false;
    params.ctx          = &ctx;
    gguf_context * gguf = gguf_init_from_file(path.c_str(), params);
    SYNTH_TEST_CHECK(gguf != nullptr && ctx != nullptr);

    const int64_t file_type = gguf_find_key(gguf, "general.file_type");
    const int64_t profile   = gguf_find_key(gguf, "synthesize.quantization.profile");
    SYNTH_TEST_CHECK(file_type >= 0 && gguf_get_val_u32(gguf, file_type) == GGML_FTYPE_MOSTLY_Q8_0);
    SYNTH_TEST_CHECK(profile >= 0 && std::strcmp(gguf_get_val_str(gguf, profile), "Q8") == 0);

    // The generator's matrices and both lookup tables: Q8_0 at their declared
    // two-dimensional shape.
    for (const char * name : {
             "llm.embed_tokens.weight",
             "llm.layers.0.self_attn.q_proj.weight",
             "audio_embeddings.weight",
             "audio_heads.weight",
         }) {
        ggml_tensor * quantized = ggml_get_tensor(ctx, name);
        SYNTH_TEST_CHECK(quantized != nullptr && quantized->type == GGML_TYPE_Q8_0);
        SYNTH_TEST_CHECK(quantized->ne[0] == 256 && quantized->ne[1] == 3 && quantized->ne[2] == 1);
    }

    // Every generator norm stays exact.
    for (const char * name : { "llm.norm.weight", "llm.layers.0.self_attn.q_norm.weight" }) {
        ggml_tensor * norm = ggml_get_tensor(ctx, name);
        SYNTH_TEST_CHECK(norm != nullptr && norm->type == GGML_TYPE_F32);
    }

    // The whole codec half is untouched -- including the tensors a codec-half
    // profile would pack, halve or leave native, each for its own reason.
    for (const char * name : {
             "codec.semantic_model.encoder.layers.0.attn.q_proj.weight",
             "codec.semantic_model.encoder.layers.0.attn.q_proj.bias",
             "codec.acoustic_decoder.conv1.weight",
             "codec.acoustic_decoder.conv2.weight",
             "codec.acoustic_decoder.block.0.conv_t1.weight",
         }) {
        ggml_tensor * codec = ggml_get_tensor(ctx, name);
        SYNTH_TEST_CHECK(codec != nullptr && codec->type == GGML_TYPE_F32);
    }
    // Untouched means the shapes are the source's too, not merely the type:
    // a conv kernel that reached the packing branch would come back
    // two-dimensional even at F32.
    ggml_tensor * conv2 = ggml_get_tensor(ctx, "codec.acoustic_decoder.conv2.weight");
    SYNTH_TEST_CHECK(conv2 != nullptr && conv2->ne[0] == 7 && conv2->ne[1] == 32 && conv2->ne[2] == 1);

    // Round-trip a quantized generator matrix to prove the bytes are a real
    // quantization of the source values, not merely the right shape and type.
    ggml_tensor * q_proj = ggml_get_tensor(ctx, "llm.layers.0.self_attn.q_proj.weight");
    SYNTH_TEST_CHECK(q_proj != nullptr);
    const ggml_type_traits * traits = ggml_get_type_traits(q_proj->type);
    SYNTH_TEST_CHECK(traits != nullptr && traits->to_float != nullptr);
    std::vector<float> dequantized(static_cast<size_t>(q_proj->ne[0]));
    traits->to_float(q_proj->data, dequantized.data(), q_proj->ne[0]);
    const int64_t probe = 20;
    SYNTH_TEST_CHECK(std::fabs(dequantized[probe] - static_cast<float>(probe + 1) / 7.0f) < 0.2f);

    gguf_free(gguf);
    ggml_free(ctx);
    return 0;
}

// Task 2 (Plan 6): the Q4 profile, whose whole point is that two tensors do
// NOT follow the rest. Everything the generator multiplies by goes to Q4_K;
// `llm.embed_tokens.weight` and `audio_embeddings.weight` stay Q8_0 because
// CUDA's GET_ROWS accepts no k-quant, and a package that k-quantized them
// would load, run, and silently execute its embedding lookups on the CPU.
// That failure is invisible to a type check, so it is asserted here at the
// only place the bytes are actually produced.
int check_omnivoice_q4_k_output(const std::string & path) {
    ggml_context *   ctx = nullptr;
    gguf_init_params params{};
    params.no_alloc     = false;
    params.ctx          = &ctx;
    gguf_context * gguf = gguf_init_from_file(path.c_str(), params);
    SYNTH_TEST_CHECK(gguf != nullptr && ctx != nullptr);

    const int64_t file_type = gguf_find_key(gguf, "general.file_type");
    const int64_t profile   = gguf_find_key(gguf, "synthesize.quantization.profile");
    SYNTH_TEST_CHECK(file_type >= 0 && gguf_get_val_u32(gguf, file_type) == GGML_FTYPE_MOSTLY_Q4_K);
    SYNTH_TEST_CHECK(profile >= 0 && std::strcmp(gguf_get_val_str(gguf, profile), "Q4_K") == 0);

    // The matrices: Q4_K at their declared two-dimensional shape. `audio_heads`
    // is in this list and not the one below because it is a plain
    // `ggml_mul_mat`, not a lookup table, despite reading like one.
    for (const char * name : { "llm.layers.0.self_attn.q_proj.weight", "audio_heads.weight" }) {
        ggml_tensor * quantized = ggml_get_tensor(ctx, name);
        SYNTH_TEST_CHECK(quantized != nullptr && quantized->type == GGML_TYPE_Q4_K);
        SYNTH_TEST_CHECK(quantized->ne[0] == 256 && quantized->ne[1] == 3 && quantized->ne[2] == 1);
    }

    // The pin. These two differ from every other quantized tensor in the file.
    for (const char * name : { "llm.embed_tokens.weight", "audio_embeddings.weight" }) {
        ggml_tensor * table = ggml_get_tensor(ctx, name);
        SYNTH_TEST_CHECK(table != nullptr && table->type == GGML_TYPE_Q8_0);
        SYNTH_TEST_CHECK(table->ne[0] == 256 && table->ne[1] == 3 && table->ne[2] == 1);
    }

    // Norms exact, and the whole codec half untouched at its source shapes.
    for (const char * name : { "llm.norm.weight", "llm.layers.0.self_attn.q_norm.weight" }) {
        ggml_tensor * norm = ggml_get_tensor(ctx, name);
        SYNTH_TEST_CHECK(norm != nullptr && norm->type == GGML_TYPE_F32);
    }
    for (const char * name : {
             "codec.semantic_model.encoder.layers.0.attn.q_proj.weight",
             "codec.acoustic_decoder.conv1.weight",
             "codec.acoustic_decoder.conv2.weight",
             "codec.acoustic_decoder.block.0.conv_t1.weight",
         }) {
        ggml_tensor * codec = ggml_get_tensor(ctx, name);
        SYNTH_TEST_CHECK(codec != nullptr && codec->type == GGML_TYPE_F32);
    }
    ggml_tensor * conv2 = ggml_get_tensor(ctx, "codec.acoustic_decoder.conv2.weight");
    SYNTH_TEST_CHECK(conv2 != nullptr && conv2->ne[0] == 7 && conv2->ne[1] == 32 && conv2->ne[2] == 1);

    // Round-trip a Q4_K matrix: real quantized values, not merely the right
    // shape and type. The fixture's row rises linearly from 1/7 to 256/7, so a
    // Q4_K sub-block of 32 spans about 4.4 across 15 levels -- a step near 0.3,
    // which is why this tolerance is looser than the Q8_0 checks above rather
    // than because anything is being waved through.
    ggml_tensor * q_proj = ggml_get_tensor(ctx, "llm.layers.0.self_attn.q_proj.weight");
    SYNTH_TEST_CHECK(q_proj != nullptr);
    const ggml_type_traits * traits = ggml_get_type_traits(q_proj->type);
    SYNTH_TEST_CHECK(traits != nullptr && traits->to_float != nullptr);
    std::vector<float> dequantized(static_cast<size_t>(q_proj->ne[0]));
    traits->to_float(q_proj->data, dequantized.data(), q_proj->ne[0]);
    for (const int64_t probe : { int64_t(20), int64_t(137), int64_t(250) }) {
        SYNTH_TEST_CHECK(std::fabs(dequantized[probe] - static_cast<float>(probe + 1) / 7.0f) < 0.4f);
    }

    gguf_free(gguf);
    ggml_free(ctx);
    return 0;
}

// F16 and BF16 (src/arch/omnivoice/weights.h):
// the same generator half as Q8 and Q4_K, narrowed to a reference
// dtype instead of block-quantized. `narrow`/`file_type`/`profile_name`
// parameterize the one difference between the two profiles; everything else
// -- which tensors move, which stay F32, the shapes -- is identical, because
// both give `row_lookup_type` the same value as `matrix_weight_type` (neither
// format needs Q4_K's pin) and both keep `matrix_weight_layout` Native at
// the profile row itself rather than by the two-dimensional demotion Q8
// and Q4_K rely on -- neither F16 nor BF16 is `ggml_is_quantized`, so
// there is nothing for that demotion to catch.
int check_omnivoice_narrowed_output(const std::string & path,
                                    ggml_type           narrow,
                                    uint32_t            file_type,
                                    const char *        profile_name,
                                    float               tolerance) {
    ggml_context *   ctx = nullptr;
    gguf_init_params params{};
    params.no_alloc     = false;
    params.ctx          = &ctx;
    gguf_context * gguf = gguf_init_from_file(path.c_str(), params);
    SYNTH_TEST_CHECK(gguf != nullptr && ctx != nullptr);

    const int64_t file_type_key = gguf_find_key(gguf, "general.file_type");
    const int64_t profile_key   = gguf_find_key(gguf, "synthesize.quantization.profile");
    SYNTH_TEST_CHECK(file_type_key >= 0 && gguf_get_val_u32(gguf, file_type_key) == file_type);
    SYNTH_TEST_CHECK(profile_key >= 0 && std::strcmp(gguf_get_val_str(gguf, profile_key), profile_name) == 0);

    // The generator's matrices and both lookup tables: narrowed at their
    // declared two-dimensional shape, never packed.
    for (const char * name : {
             "llm.embed_tokens.weight",
             "llm.layers.0.self_attn.q_proj.weight",
             "audio_embeddings.weight",
             "audio_heads.weight",
         }) {
        ggml_tensor * quantized = ggml_get_tensor(ctx, name);
        SYNTH_TEST_CHECK(quantized != nullptr && quantized->type == narrow);
        SYNTH_TEST_CHECK(quantized->ne[0] == 256 && quantized->ne[1] == 3 && quantized->ne[2] == 1);
    }

    // Every generator norm stays exact.
    for (const char * name : { "llm.norm.weight", "llm.layers.0.self_attn.q_norm.weight" }) {
        ggml_tensor * norm = ggml_get_tensor(ctx, name);
        SYNTH_TEST_CHECK(norm != nullptr && norm->type == GGML_TYPE_F32);
    }

    // The whole codec half is untouched -- including the tensors a codec-half
    // profile would pack, halve or leave native, each for its own reason.
    for (const char * name : {
             "codec.semantic_model.encoder.layers.0.attn.q_proj.weight",
             "codec.semantic_model.encoder.layers.0.attn.q_proj.bias",
             "codec.acoustic_decoder.conv1.weight",
             "codec.acoustic_decoder.conv2.weight",
             "codec.acoustic_decoder.block.0.conv_t1.weight",
         }) {
        ggml_tensor * codec = ggml_get_tensor(ctx, name);
        SYNTH_TEST_CHECK(codec != nullptr && codec->type == GGML_TYPE_F32);
    }
    ggml_tensor * conv2 = ggml_get_tensor(ctx, "codec.acoustic_decoder.conv2.weight");
    SYNTH_TEST_CHECK(conv2 != nullptr && conv2->ne[0] == 7 && conv2->ne[1] == 32 && conv2->ne[2] == 1);

    // Round-trip a narrowed generator matrix to prove the bytes are a real
    // narrowing of the source values, not merely the right shape and type.
    ggml_tensor * q_proj = ggml_get_tensor(ctx, "llm.layers.0.self_attn.q_proj.weight");
    SYNTH_TEST_CHECK(q_proj != nullptr);
    const ggml_type_traits * traits = ggml_get_type_traits(q_proj->type);
    SYNTH_TEST_CHECK(traits != nullptr && traits->to_float != nullptr);
    std::vector<float> dequantized(static_cast<size_t>(q_proj->ne[0]));
    traits->to_float(q_proj->data, dequantized.data(), q_proj->ne[0]);
    const int64_t probe = 20;
    SYNTH_TEST_CHECK(std::fabs(dequantized[probe] - static_cast<float>(probe + 1) / 7.0f) < tolerance);

    gguf_free(gguf);
    ggml_free(ctx);
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    SYNTH_TEST_CHECK(argc == 2);
    const std::string root                     = argv[1];
    const std::string input                    = root + "/quantize-input.gguf";
    const std::string first                    = root + "/quantize-first.gguf";
    const std::string second                   = root + "/quantize-second.gguf";
    const std::string invalid_input            = root + "/quantize-invalid.gguf";
    const std::string invalid_output           = root + "/quantize-invalid-output.gguf";
    const std::string q8_first                 = root + "/quantize-q8-first.gguf";
    const std::string q8_second                = root + "/quantize-q8-second.gguf";
    const std::string omnivoice_input          = root + "/quantize-omnivoice-input.gguf";
    const std::string omnivoice_invalid_input  = root + "/quantize-omnivoice-invalid.gguf";
    const std::string omnivoice_invalid_output = root + "/quantize-omnivoice-invalid-output.gguf";
    const std::string omnivoice_q8_first       = root + "/quantize-omnivoice-q8-first.gguf";
    const std::string omnivoice_q8_second      = root + "/quantize-omnivoice-q8-second.gguf";
    const std::string omnivoice_gen_first      = root + "/quantize-omnivoice-q8gen-first.gguf";
    const std::string omnivoice_gen_second     = root + "/quantize-omnivoice-q8gen-second.gguf";
    const std::string omnivoice_q4_first       = root + "/quantize-omnivoice-q4kgen-first.gguf";
    const std::string omnivoice_q4_second      = root + "/quantize-omnivoice-q4kgen-second.gguf";
    const std::string omnivoice_f16gen_first   = root + "/quantize-omnivoice-f16gen-first.gguf";
    const std::string omnivoice_f16gen_second  = root + "/quantize-omnivoice-f16gen-second.gguf";
    const std::string omnivoice_bf16gen_first  = root + "/quantize-omnivoice-bf16gen-first.gguf";
    const std::string omnivoice_bf16gen_second = root + "/quantize-omnivoice-bf16gen-second.gguf";
    std::remove(first.c_str());
    std::remove(second.c_str());
    std::remove(invalid_output.c_str());
    std::remove(q8_first.c_str());
    std::remove(q8_second.c_str());
    std::remove(omnivoice_invalid_output.c_str());
    std::remove(omnivoice_q8_first.c_str());
    std::remove(omnivoice_q8_second.c_str());
    std::remove(omnivoice_gen_first.c_str());
    std::remove(omnivoice_gen_second.c_str());
    std::remove(omnivoice_q4_first.c_str());
    std::remove(omnivoice_q4_second.c_str());
    std::remove(omnivoice_f16gen_first.c_str());
    std::remove(omnivoice_f16gen_second.c_str());
    std::remove(omnivoice_bf16gen_first.c_str());
    std::remove(omnivoice_bf16gen_second.c_str());

    SYNTH_TEST_CHECK(write_fixture(input, false));
    std::string error;
    SYNTH_TEST_CHECK(synth::quantize::quantize_file(input, first, "F16", error));
    SYNTH_TEST_CHECK(error.empty());
    SYNTH_TEST_CHECK(check_output(first) == 0);
    SYNTH_TEST_CHECK(synth::quantize::quantize_file(input, second, "f16", error));
    SYNTH_TEST_CHECK(read_bytes(first) == read_bytes(second));

    SYNTH_TEST_CHECK(write_fixture(invalid_input, true));
    SYNTH_TEST_CHECK(!synth::quantize::quantize_file(invalid_input, invalid_output, "F16", error));
    SYNTH_TEST_CHECK(error.find("decoder.future_module.weight") != std::string::npos);
    std::ifstream absent(invalid_output, std::ios::binary);
    SYNTH_TEST_CHECK(!absent.good());

    SYNTH_TEST_CHECK(synth::quantize::quantize_file(input, q8_first, "Q8_MIXED", error));
    SYNTH_TEST_CHECK(error.empty());
    SYNTH_TEST_CHECK(check_q8_output(q8_first) == 0);
    SYNTH_TEST_CHECK(synth::quantize::quantize_file(input, q8_second, "q8_mixed", error));
    SYNTH_TEST_CHECK(read_bytes(q8_first) == read_bytes(q8_second));

    SYNTH_TEST_CHECK(write_omnivoice_fixture(omnivoice_input, false));
    SYNTH_TEST_CHECK(synth::quantize::quantize_file(omnivoice_input, omnivoice_q8_first, "Q8_CODEC_MIXED", error));
    SYNTH_TEST_CHECK(error.empty());
    SYNTH_TEST_CHECK(check_omnivoice_q8_output(omnivoice_q8_first) == 0);
    SYNTH_TEST_CHECK(synth::quantize::quantize_file(omnivoice_input, omnivoice_q8_second, "q8_codec_mixed", error));
    SYNTH_TEST_CHECK(read_bytes(omnivoice_q8_first) == read_bytes(omnivoice_q8_second));

    SYNTH_TEST_CHECK(synth::quantize::quantize_file(omnivoice_input, omnivoice_gen_first, "Q8", error));
    SYNTH_TEST_CHECK(error.empty());
    SYNTH_TEST_CHECK(check_omnivoice_q8_output_generator(omnivoice_gen_first) == 0);
    SYNTH_TEST_CHECK(synth::quantize::quantize_file(omnivoice_input, omnivoice_gen_second, "q8", error));
    SYNTH_TEST_CHECK(read_bytes(omnivoice_gen_first) == read_bytes(omnivoice_gen_second));

    SYNTH_TEST_CHECK(synth::quantize::quantize_file(omnivoice_input, omnivoice_q4_first, "Q4_K", error));
    SYNTH_TEST_CHECK(error.empty());
    SYNTH_TEST_CHECK(check_omnivoice_q4_k_output(omnivoice_q4_first) == 0);
    SYNTH_TEST_CHECK(synth::quantize::quantize_file(omnivoice_input, omnivoice_q4_second, "q4_k", error));
    SYNTH_TEST_CHECK(read_bytes(omnivoice_q4_first) == read_bytes(omnivoice_q4_second));
    // The two generator profiles differ in the file, not merely in metadata --
    // a Q4_K that had silently fallen back to its sibling's types would
    // pass every per-tensor check above only if the check list were wrong, and
    // this catches the case where it is.
    SYNTH_TEST_CHECK(read_bytes(omnivoice_q4_first) != read_bytes(omnivoice_gen_first));
    SYNTH_TEST_CHECK(read_bytes(omnivoice_q4_first).size() < read_bytes(omnivoice_gen_first).size());

    SYNTH_TEST_CHECK(synth::quantize::quantize_file(omnivoice_input, omnivoice_f16gen_first, "F16", error));
    SYNTH_TEST_CHECK(error.empty());
    SYNTH_TEST_CHECK(check_omnivoice_narrowed_output(omnivoice_f16gen_first, GGML_TYPE_F16, 1, "F16", 0.01f) == 0);
    SYNTH_TEST_CHECK(synth::quantize::quantize_file(omnivoice_input, omnivoice_f16gen_second, "f16", error));
    SYNTH_TEST_CHECK(read_bytes(omnivoice_f16gen_first) == read_bytes(omnivoice_f16gen_second));

    SYNTH_TEST_CHECK(synth::quantize::quantize_file(omnivoice_input, omnivoice_bf16gen_first, "BF16", error));
    SYNTH_TEST_CHECK(error.empty());
    SYNTH_TEST_CHECK(check_omnivoice_narrowed_output(omnivoice_bf16gen_first, GGML_TYPE_BF16, GGML_FTYPE_MOSTLY_BF16,
                                                     "BF16", 0.05f) == 0);
    SYNTH_TEST_CHECK(synth::quantize::quantize_file(omnivoice_input, omnivoice_bf16gen_second, "bf16", error));
    SYNTH_TEST_CHECK(read_bytes(omnivoice_bf16gen_first) == read_bytes(omnivoice_bf16gen_second));
    // F16 and BF16 differ from each other and from their k-quant/Q8_0
    // siblings in the file, not merely in metadata -- the same non-fallback
    // proof the Q4_K/Q8 pair gets above.
    SYNTH_TEST_CHECK(read_bytes(omnivoice_f16gen_first) != read_bytes(omnivoice_bf16gen_first));
    SYNTH_TEST_CHECK(read_bytes(omnivoice_f16gen_first) != read_bytes(omnivoice_gen_first));
    SYNTH_TEST_CHECK(read_bytes(omnivoice_bf16gen_first) != read_bytes(omnivoice_gen_first));

    SYNTH_TEST_CHECK(write_omnivoice_fixture(omnivoice_invalid_input, true));
    SYNTH_TEST_CHECK(
        !synth::quantize::quantize_file(omnivoice_invalid_input, omnivoice_invalid_output, "Q8_CODEC_MIXED", error));
    // The row-size refusal, not the unknown-tensor one -- see the fixture.
    // The tensor named is the semantic-model Linear with the 17-wide row:
    // since the conv-exempt policy the Linears are the only codec tensors
    // that can reach the row-size check at all.
    SYNTH_TEST_CHECK(error.find("row size is incompatible") != std::string::npos);
    SYNTH_TEST_CHECK(error.find("codec.semantic_model.encoder.layers.0.ff.inter_dense.weight") != std::string::npos);
    std::ifstream omnivoice_absent(omnivoice_invalid_output, std::ios::binary);
    SYNTH_TEST_CHECK(!omnivoice_absent.good());

    // A genuinely unrecognised name is fatal under Q8 too, even though
    // the half it lands in is the one this profile leaves alone -- that is
    // the Unknown-first ordering in classify_tensor_for_half. Without it the
    // name would be reported as Sensitive and written out as an F32 tensor
    // nobody recognises. Q8 walks past both `future_module` and the
    // 17-wide Linear (F32, no row-size check to fail) and stops on
    // `unknown_module` instead, which is why the expected message differs
    // from the Q8_CODEC_MIXED case above.
    for (const char * generator_profile : { "Q8", "Q4_K", "F16", "BF16" }) {
        const std::string invalid = root + "/quantize-omnivoice-" + generator_profile + "-invalid.gguf";
        std::remove(invalid.c_str());
        SYNTH_TEST_CHECK(!synth::quantize::quantize_file(omnivoice_invalid_input, invalid, generator_profile, error));
        SYNTH_TEST_CHECK(error.find("unknown omnivoice tensor: codec.unknown_module.conv.weight") != std::string::npos);
        std::ifstream omnivoice_gen_absent(invalid, std::ios::binary);
        SYNTH_TEST_CHECK(!omnivoice_gen_absent.good());
    }
    return 0;
}
