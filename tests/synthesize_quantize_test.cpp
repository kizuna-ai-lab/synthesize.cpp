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

}  // namespace

int main(int argc, char ** argv) {
    SYNTH_TEST_CHECK(argc == 2);
    const std::string root           = argv[1];
    const std::string input          = root + "/quantize-input.gguf";
    const std::string first          = root + "/quantize-first.gguf";
    const std::string second         = root + "/quantize-second.gguf";
    const std::string invalid_input  = root + "/quantize-invalid.gguf";
    const std::string invalid_output = root + "/quantize-invalid-output.gguf";
    const std::string q8_first       = root + "/quantize-q8-first.gguf";
    const std::string q8_second      = root + "/quantize-q8-second.gguf";
    std::remove(first.c_str());
    std::remove(second.c_str());
    std::remove(invalid_output.c_str());
    std::remove(q8_first.c_str());
    std::remove(q8_second.c_str());

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
    return 0;
}
