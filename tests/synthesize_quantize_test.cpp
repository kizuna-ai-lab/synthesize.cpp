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

    add1("llm.norm.weight", 4);  // generator: Sensitive whatever its shape

    // A genuine 2-D Linear (HuBERT's attention projections stand in for the
    // whole 73-tensor matmul-consumed group): MatrixWeight, but must land
    // Native, not packed -- the reason omnivoice needs matrix_family at all.
    add("codec.semantic_model.encoder.layers.0.attn.q_proj.weight", 32, 3);
    add1("codec.semantic_model.encoder.layers.0.attn.q_proj.bias", 3);

    // An ordinary 3-D convolution kernel: MatrixWeight, packed normally.
    add3("codec.acoustic_decoder.conv1.weight", 3, 32, 4);
    add1("codec.acoustic_decoder.conv1.bias", 4);

    // The hazard shape: out_channels == 1 collapses ggml_n_dims to 2, the
    // same shape class as VITS's decoder.post.weight -- must stay PACKED.
    add3("codec.acoustic_decoder.conv2.weight", 7, 32, 1);
    add1("codec.acoustic_decoder.conv2.bias", 1);

    // TransposeWeight: the same F32-at-every-profile override VITS and
    // Qwen3-TTS's decoder make, not Kokoro's halved type.
    add3("codec.acoustic_decoder.block.0.conv_t1.weight", 4, 3, 2);
    add1("codec.acoustic_decoder.block.0.conv_t1.bias", 3);

    if (include_unknown) {
        add("codec.acoustic_decoder.future_module.weight", 4, 4);
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
    SYNTH_TEST_CHECK(profile >= 0 && std::strcmp(gguf_get_val_str(gguf, profile), "Q8_MIXED") == 0);

    ggml_tensor * sensitive = ggml_get_tensor(ctx, "llm.norm.weight");
    SYNTH_TEST_CHECK(sensitive != nullptr && sensitive->type == GGML_TYPE_F32);

    // A genuine 2-D Linear: MatrixWeight, but must stay Native -- packing it
    // would flatten [32, 3] into a meaningless single row of 96.
    ggml_tensor * linear = ggml_get_tensor(ctx, "codec.semantic_model.encoder.layers.0.attn.q_proj.weight");
    SYNTH_TEST_CHECK(linear != nullptr && linear->type == GGML_TYPE_Q8_0);
    SYNTH_TEST_CHECK(linear->ne[0] == 32 && linear->ne[1] == 3 && linear->ne[2] == 1);
    ggml_tensor * linear_bias = ggml_get_tensor(ctx, "codec.semantic_model.encoder.layers.0.attn.q_proj.bias");
    SYNTH_TEST_CHECK(linear_bias != nullptr && linear_bias->type == GGML_TYPE_F32);

    // An ordinary 3-D convolution kernel: MatrixWeight, packed to
    // [kernel * in_channels, out_channels] = [96, 4].
    ggml_tensor * conv1 = ggml_get_tensor(ctx, "codec.acoustic_decoder.conv1.weight");
    SYNTH_TEST_CHECK(conv1 != nullptr && conv1->type == GGML_TYPE_Q8_0);
    SYNTH_TEST_CHECK(conv1->ne[0] == 3 * 32 && conv1->ne[1] == 4 && conv1->ne[2] == 1);

    // The VITS-shaped hazard: a convolution kernel whose out_channels == 1
    // collapses ggml_n_dims to 2, the same shape a genuine Linear would
    // report. It must still be PACKED ([224, 1]), not demoted to a native
    // row of 7 -- the one named exception in quantize.cpp's demotion rule.
    ggml_tensor * conv2 = ggml_get_tensor(ctx, "codec.acoustic_decoder.conv2.weight");
    SYNTH_TEST_CHECK(conv2 != nullptr && conv2->type == GGML_TYPE_Q8_0);
    SYNTH_TEST_CHECK(conv2->ne[0] == 7 * 32 && conv2->ne[1] == 1 && conv2->ne[2] == 1);
    ggml_tensor * conv2_bias = ggml_get_tensor(ctx, "codec.acoustic_decoder.conv2.bias");
    SYNTH_TEST_CHECK(conv2_bias != nullptr && conv2_bias->type == GGML_TYPE_F32);

    // TransposeWeight: the same F32-at-every-profile override VITS and
    // Qwen3-TTS's decoder make, not Kokoro's halved type -- left Native at
    // its own declared 3-D shape, never packed.
    ggml_tensor * transpose = ggml_get_tensor(ctx, "codec.acoustic_decoder.block.0.conv_t1.weight");
    SYNTH_TEST_CHECK(transpose != nullptr && transpose->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(transpose->ne[0] == 4 && transpose->ne[1] == 3 && transpose->ne[2] == 2);

    // Round-trip the packed conv2.weight to prove the bytes are a real
    // quantization of the source values, not merely the right shape/type --
    // the same proof check_q8_output runs for VITS's own decoder.post.weight.
    const ggml_type_traits * traits = ggml_get_type_traits(conv2->type);
    SYNTH_TEST_CHECK(traits != nullptr && traits->to_float != nullptr);
    std::vector<float> dequantized(static_cast<size_t>(conv2->ne[0]));
    traits->to_float(conv2->data, dequantized.data(), conv2->ne[0]);
    const int64_t probe = 100;
    SYNTH_TEST_CHECK(std::fabs(dequantized[probe] - static_cast<float>(probe + 1) / 7.0f) < 0.2f);

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
    std::remove(first.c_str());
    std::remove(second.c_str());
    std::remove(invalid_output.c_str());
    std::remove(q8_first.c_str());
    std::remove(q8_second.c_str());
    std::remove(omnivoice_invalid_output.c_str());
    std::remove(omnivoice_q8_first.c_str());
    std::remove(omnivoice_q8_second.c_str());

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
    SYNTH_TEST_CHECK(synth::quantize::quantize_file(omnivoice_input, omnivoice_q8_first, "Q8_MIXED", error));
    SYNTH_TEST_CHECK(error.empty());
    SYNTH_TEST_CHECK(check_omnivoice_q8_output(omnivoice_q8_first) == 0);
    SYNTH_TEST_CHECK(synth::quantize::quantize_file(omnivoice_input, omnivoice_q8_second, "q8_mixed", error));
    SYNTH_TEST_CHECK(read_bytes(omnivoice_q8_first) == read_bytes(omnivoice_q8_second));

    SYNTH_TEST_CHECK(write_omnivoice_fixture(omnivoice_invalid_input, true));
    SYNTH_TEST_CHECK(
        !synth::quantize::quantize_file(omnivoice_invalid_input, omnivoice_invalid_output, "Q8_MIXED", error));
    SYNTH_TEST_CHECK(error.find("codec.acoustic_decoder.future_module.weight") != std::string::npos);
    std::ifstream omnivoice_absent(omnivoice_invalid_output, std::ios::binary);
    SYNTH_TEST_CHECK(!omnivoice_absent.good());
    return 0;
}
