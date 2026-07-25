#include "policy.h"

#include "arch/kokoro/quantization.h"

#include <cctype>
#include <initializer_list>
#include <string_view>
#include <vector>

namespace synth::quantize {

namespace {

const Profile kProfiles[] = {
    { "F16",      GGML_TYPE_F16,  TensorLayout::Native,       GGML_TYPE_F16, GGML_TYPE_F32, 1,                      1 },
    { "Q8_MIXED", GGML_TYPE_Q8_0, TensorLayout::PackedMatrix, GGML_TYPE_F16, GGML_TYPE_F32, GGML_FTYPE_MOSTLY_Q8_0, 1 },
};

bool iequals(const char * lhs, const char * rhs) {
    if (lhs == nullptr || rhs == nullptr) {
        return false;
    }
    while (*lhs != '\0' && *rhs != '\0') {
        if (std::tolower(static_cast<unsigned char>(*lhs)) != std::tolower(static_cast<unsigned char>(*rhs))) {
            return false;
        }
        ++lhs;
        ++rhs;
    }
    return *lhs == '\0' && *rhs == '\0';
}

enum class CatalogRole {
    Unknown,
    MatrixWeight,
    TransposeWeight,
    Sensitive,
};

std::vector<std::string_view> split_name(const std::string & name) {
    std::vector<std::string_view> tokens;
    size_t                        begin = 0;
    while (begin <= name.size()) {
        const size_t end = name.find('.', begin);
        tokens.emplace_back(name.data() + begin, (end == std::string::npos ? name.size() : end) - begin);
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return tokens;
}

bool one_of(std::string_view value, std::initializer_list<std::string_view> choices) {
    for (std::string_view choice : choices) {
        if (value == choice) {
            return true;
        }
    }
    return false;
}

bool is_index(std::string_view value) {
    if (value.empty() || (value.size() > 1 && value.front() == '0')) {
        return false;
    }
    for (char character : value) {
        if (character < '0' || character > '9') {
            return false;
        }
    }
    return true;
}

CatalogRole classify_text(const std::vector<std::string_view> & tokens) {
    if (tokens.size() == 3 && tokens[0] == "text_encoder" &&
        ((tokens[1] == "token_embedding" && tokens[2] == "weight") ||
         (tokens[1] == "projection" && one_of(tokens[2], { "weight", "bias" })))) {
        return CatalogRole::Sensitive;
    }
    if (tokens.size() == 5 && tokens[0] == "text_encoder" && tokens[1] == "blocks" && is_index(tokens[2]) &&
        one_of(tokens[3], { "attention_norm", "ffn_norm" }) && one_of(tokens[4], { "weight", "bias" })) {
        return CatalogRole::Sensitive;
    }
    if (tokens.size() == 6 && tokens[0] == "text_encoder" && tokens[1] == "blocks" && is_index(tokens[2])) {
        const bool attention = tokens[3] == "attention" && one_of(tokens[4], { "query", "key", "value", "output" }) &&
                               one_of(tokens[5], { "weight", "bias" });
        const bool relative  = tokens[3] == "attention" && one_of(tokens[4], { "relative_key", "relative_value" }) &&
                               tokens[5] == "weight";
        const bool ffn =
            tokens[3] == "ffn" && one_of(tokens[4], { "input", "output" }) && one_of(tokens[5], { "weight", "bias" });
        if (attention || relative || ffn) {
            return CatalogRole::Sensitive;
        }
    }
    return CatalogRole::Unknown;
}

CatalogRole classify_duration(const std::vector<std::string_view> & tokens) {
    if (tokens.empty() || tokens[0] != "duration_predictor") {
        return CatalogRole::Unknown;
    }
    if (tokens.size() == 3 &&
        ((one_of(tokens[1], { "conditioning", "pre", "projection" }) && one_of(tokens[2], { "weight", "bias" })) ||
         (tokens[1] == "affine" && one_of(tokens[2], { "bias", "log_scale" })))) {
        return CatalogRole::Sensitive;
    }
    if (tokens.size() == 6 && tokens[1] == "dds" && tokens[2] == "blocks" && is_index(tokens[3]) &&
        one_of(tokens[4], { "depthwise", "pointwise", "depthwise_norm", "pointwise_norm" }) &&
        one_of(tokens[5], { "weight", "bias" })) {
        return CatalogRole::Sensitive;
    }
    if (tokens.size() == 5 && tokens[1] == "flows" && is_index(tokens[2]) &&
        one_of(tokens[3], { "pre", "projection" }) && one_of(tokens[4], { "weight", "bias" })) {
        return CatalogRole::Sensitive;
    }
    if (tokens.size() == 8 && tokens[1] == "flows" && is_index(tokens[2]) && tokens[3] == "dds" &&
        tokens[4] == "blocks" && is_index(tokens[5]) &&
        one_of(tokens[6], { "depthwise", "pointwise", "depthwise_norm", "pointwise_norm" }) &&
        one_of(tokens[7], { "weight", "bias" })) {
        return CatalogRole::Sensitive;
    }
    return CatalogRole::Unknown;
}

CatalogRole classify_flow(const std::vector<std::string_view> & tokens) {
    if (tokens.size() == 5 && tokens[0] == "flow" && tokens[1] == "blocks" && is_index(tokens[2]) &&
        one_of(tokens[3], { "conditioning", "pre", "projection" }) && one_of(tokens[4], { "weight", "bias" })) {
        return tokens[4] == "weight" ? CatalogRole::MatrixWeight : CatalogRole::Sensitive;
    }
    if (tokens.size() == 8 && tokens[0] == "flow" && tokens[1] == "blocks" && is_index(tokens[2]) &&
        tokens[3] == "wn" && tokens[4] == "layers" && is_index(tokens[5]) &&
        one_of(tokens[6], { "input", "residual_skip" }) && one_of(tokens[7], { "weight", "bias" })) {
        return tokens[7] == "weight" ? CatalogRole::MatrixWeight : CatalogRole::Sensitive;
    }
    return CatalogRole::Unknown;
}

CatalogRole classify_decoder(const std::vector<std::string_view> & tokens) {
    if (tokens.size() == 3 && tokens[0] == "decoder") {
        if (one_of(tokens[1], { "conditioning", "pre" }) && one_of(tokens[2], { "weight", "bias" })) {
            return tokens[2] == "weight" ? CatalogRole::MatrixWeight : CatalogRole::Sensitive;
        }
        if (tokens[1] == "post" && tokens[2] == "weight") {
            return CatalogRole::MatrixWeight;
        }
    }
    if (tokens.size() == 5 && tokens[0] == "decoder" && tokens[1] == "upsample" && is_index(tokens[2]) &&
        tokens[3] == "transpose_conv" && one_of(tokens[4], { "weight", "bias" })) {
        return tokens[4] == "weight" ? CatalogRole::TransposeWeight : CatalogRole::Sensitive;
    }
    if (tokens.size() == 8 && tokens[0] == "decoder" && tokens[1] == "upsample" && is_index(tokens[2]) &&
        tokens[3] == "resblocks" && is_index(tokens[4]) && one_of(tokens[5], { "conv1", "conv2" }) &&
        is_index(tokens[6]) && one_of(tokens[7], { "weight", "bias" })) {
        return tokens[7] == "weight" ? CatalogRole::MatrixWeight : CatalogRole::Sensitive;
    }
    return CatalogRole::Unknown;
}

CatalogRole classify_vits_tensor(const std::string & name) {
    const std::vector<std::string_view> tokens = split_name(name);
    if (tokens.size() == 3 && tokens[0] == "voice" && tokens[1] == "embedding" && tokens[2] == "weight") {
        return CatalogRole::Sensitive;
    }
    for (CatalogRole role :
         { classify_text(tokens), classify_duration(tokens), classify_flow(tokens), classify_decoder(tokens) }) {
        if (role != CatalogRole::Unknown) {
            return role;
        }
    }
    return CatalogRole::Unknown;
}

}  // namespace

const Profile * find_profile(const char * name) {
    for (const Profile & profile : kProfiles) {
        if (iequals(profile.name, name)) {
            return &profile;
        }
    }
    return nullptr;
}

bool resolve_vits_target_spec(const Profile & profile, const std::string & name, TargetSpec & spec_out) {
    switch (classify_vits_tensor(name)) {
        case CatalogRole::MatrixWeight:
            spec_out = { profile.matrix_weight_type, profile.matrix_weight_layout };
            return true;
        case CatalogRole::TransposeWeight:
            spec_out = { profile.transpose_weight_type, TensorLayout::Native };
            return true;
        case CatalogRole::Sensitive:
            spec_out = { profile.sensitive_type, TensorLayout::Native };
            return true;
        case CatalogRole::Unknown:
            return false;
    }
    return false;
}

bool resolve_kokoro_target_spec(const Profile & profile, const std::string & name, TargetSpec & spec_out) {
    // The classifier lives in the family module so the runtime's catalog and
    // this tool cannot disagree about a tensor.
    switch (synth::kokoro::tensor_role(name)) {
        case synth::kokoro::TensorRole::MatrixWeight:
            spec_out = { profile.matrix_weight_type, profile.matrix_weight_layout };
            return true;
        case synth::kokoro::TensorRole::TransposeWeight:
            spec_out = { profile.transpose_weight_type, TensorLayout::Native };
            return true;
        case synth::kokoro::TensorRole::Sensitive:
            spec_out = { profile.sensitive_type, TensorLayout::Native };
            return true;
        case synth::kokoro::TensorRole::Unknown:
            return false;
    }
    return false;
}

bool resolve_kokoro_target_type(const Profile & profile, const std::string & name, ggml_type & type_out) {
    TargetSpec spec{};
    if (!resolve_kokoro_target_spec(profile, name, spec)) {
        return false;
    }
    type_out = spec.type;
    return true;
}

bool resolve_vits_target_type(const Profile & profile, const std::string & name, ggml_type & type_out) {
    TargetSpec spec{};
    if (!resolve_vits_target_spec(profile, name, spec)) {
        return false;
    }
    type_out = spec.type;
    return true;
}

}  // namespace synth::quantize
