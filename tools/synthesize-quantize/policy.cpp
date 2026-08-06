#include "policy.h"

#include "arch/kokoro/quantization.h"
#include "arch/omnivoice/quantization.h"

#include <cctype>
#include <initializer_list>
#include <string_view>
#include <vector>

namespace synth::quantize {

namespace {

const Profile kProfiles[] = {
    { "F16",        GGML_TYPE_F16,  TensorLayout::Native,       GGML_TYPE_F16, GGML_TYPE_F32, 1,                      1 },
    { "Q8_MIXED",   GGML_TYPE_Q8_0, TensorLayout::PackedMatrix, GGML_TYPE_F16, GGML_TYPE_F32, GGML_FTYPE_MOSTLY_Q8_0, 1 },
    // Q5_K is a super-block of 256, so it needs a row four times longer than
    // Q8_0 does. Qwen3-TTS clears that everywhere it quantizes -- its rows are
    // 1024, 2048 and 3072 -- while a family whose matrix weights are packed
    // convolution kernels will not, and is refused by the row-size check with
    // the tensor named rather than by a rule here.
    { "Q5_K_MIXED", GGML_TYPE_Q5_K, TensorLayout::PackedMatrix, GGML_TYPE_F16, GGML_TYPE_F32, GGML_FTYPE_MOSTLY_Q5_K,
     1                                                                                                                  },
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

// One Qwen3 decoder block, shared by the talker and the code predictor. The
// projections are the parameters; the norms are four narrow vectors a profile
// gains nothing by halving and can lose accuracy to.
CatalogRole classify_qwen3_block(const std::vector<std::string_view> & tokens, size_t offset) {
    if (tokens.size() == offset + 2 && one_of(tokens[offset], { "input_layernorm", "post_attn_norm" }) &&
        tokens[offset + 1] == "weight") {
        return CatalogRole::Sensitive;
    }
    if (tokens.size() == offset + 3 && tokens[offset] == "self_attn" && tokens[offset + 2] == "weight") {
        if (one_of(tokens[offset + 1], { "q_proj", "k_proj", "v_proj", "o_proj" })) {
            return CatalogRole::MatrixWeight;
        }
        // Per-head norms are head_dim wide -- 128 values against a projection's
        // two million -- and they scale every head before rope.
        if (one_of(tokens[offset + 1], { "q_norm", "k_norm" })) {
            return CatalogRole::Sensitive;
        }
    }
    if (tokens.size() == offset + 3 && tokens[offset] == "mlp" && tokens[offset + 2] == "weight" &&
        one_of(tokens[offset + 1], { "gate_proj", "up_proj", "down_proj" })) {
        return CatalogRole::MatrixWeight;
    }
    return CatalogRole::Unknown;
}

CatalogRole classify_qwen3_talker(const std::vector<std::string_view> & tokens) {
    if (tokens.size() < 2 || tokens[0] != "talker") {
        return CatalogRole::Unknown;
    }
    // talker.codec_head.weight
    if (tokens.size() == 3 && tokens[1] == "codec_head" && tokens[2] == "weight") {
        return CatalogRole::MatrixWeight;
    }
    // talker.text_projection.linear_fcN.{weight,bias}
    if (tokens.size() == 4 && tokens[1] == "text_projection" && one_of(tokens[2], { "linear_fc1", "linear_fc2" }) &&
        one_of(tokens[3], { "weight", "bias" })) {
        return tokens[3] == "weight" ? CatalogRole::MatrixWeight : CatalogRole::Sensitive;
    }
    if (tokens.size() == 4 && tokens[1] == "model" && one_of(tokens[2], { "text_embedding", "codec_embedding" }) &&
        tokens[3] == "weight") {
        return CatalogRole::MatrixWeight;
    }
    if (tokens.size() == 4 && tokens[1] == "model" && tokens[2] == "norm" && tokens[3] == "weight") {
        return CatalogRole::Sensitive;
    }
    if (tokens.size() > 4 && tokens[1] == "model" && tokens[2] == "layers" && is_index(tokens[3])) {
        return classify_qwen3_block(tokens, 4);
    }

    // The code predictor sits under the talker and repeats the same block, plus
    // one private embedding table and one private head per acoustic group.
    if (tokens.size() > 2 && tokens[1] == "code_predictor") {
        if (tokens.size() == 5 && tokens[2] == "lm_head" && is_index(tokens[3]) && tokens[4] == "weight") {
            return CatalogRole::MatrixWeight;
        }
        if (tokens.size() == 6 && tokens[2] == "model" && tokens[3] == "codec_embedding" && is_index(tokens[4]) &&
            tokens[5] == "weight") {
            return CatalogRole::MatrixWeight;
        }
        if (tokens.size() == 5 && tokens[2] == "model" && tokens[3] == "norm" && tokens[4] == "weight") {
            return CatalogRole::Sensitive;
        }
        if (tokens.size() > 5 && tokens[2] == "model" && tokens[3] == "layers" && is_index(tokens[4])) {
            return classify_qwen3_block(tokens, 5);
        }
    }
    return CatalogRole::Unknown;
}

CatalogRole classify_qwen3_codec_shape(const std::vector<std::string_view> & tokens);

// The codec half stays at the reference dtype under every profile, and that is a
// measurement rather than caution. Halving it made the codec 1.75 times slower on
// CPU -- 4.2 seconds to 7.3 on a 37-frame case -- because its convolutions run
// through im2col into a matrix multiply and ggml's F16 path there is slower than
// its F32 one. The codec is 457 MB of a 2 GB package, so the size it would give
// back is not worth the time it costs. The talker half is where both the
// parameters and the win are: halving it took the talker from 13.7 seconds to 1.7
// and the predictor from 25.1 to 3.7.
CatalogRole classify_qwen3_codec(const std::vector<std::string_view> & tokens) {
    if (tokens.size() < 3 || tokens[0] != "codec" || tokens[1] != "decoder") {
        return CatalogRole::Unknown;
    }
    // Recognised below for the catalog's sake, then reported as sensitive so no
    // profile halves it. Splitting the recognition from the decision keeps an
    // unknown codec tensor an error rather than something that slips through as
    // "sensitive by default".
    const CatalogRole recognised = classify_qwen3_codec_shape(tokens);
    return recognised == CatalogRole::Unknown ? CatalogRole::Unknown : CatalogRole::Sensitive;
}

CatalogRole classify_qwen3_codec_shape(const std::vector<std::string_view> & tokens) {
    // The quantizer's tables and its two kernel-one projections stay at the
    // reference dtype. A residual codebook's later levels carry small
    // magnitudes, so a relative error there is a large one against the residual
    // it is meant to correct, and the whole quantizer is under 35 MB.
    if (tokens[2] == "quantizer") {
        return CatalogRole::Sensitive;
    }
    if (tokens.size() == 5 && tokens[2] == "pre_conv" && tokens[3] == "conv" &&
        one_of(tokens[4], { "weight", "bias" })) {
        return tokens[4] == "weight" ? CatalogRole::MatrixWeight : CatalogRole::Sensitive;
    }
    if (tokens.size() > 3 && tokens[2] == "pre_transformer") {
        if (tokens.size() == 5 && one_of(tokens[3], { "input_proj", "output_proj" }) &&
            one_of(tokens[4], { "weight", "bias" })) {
            return tokens[4] == "weight" ? CatalogRole::MatrixWeight : CatalogRole::Sensitive;
        }
        if (tokens.size() == 5 && tokens[3] == "norm" && tokens[4] == "weight") {
            return CatalogRole::Sensitive;
        }
        if (tokens.size() > 5 && tokens[3] == "layers" && is_index(tokens[4])) {
            // The layer scales multiply a whole residual branch and start near
            // 0.01, so they stay exact.
            if (tokens.size() == 7 && one_of(tokens[5], { "self_attn_scale", "mlp_scale" }) && tokens[6] == "scale") {
                return CatalogRole::Sensitive;
            }
            if (tokens.size() == 7 && one_of(tokens[5], { "input_layernorm", "post_attn_norm" }) &&
                tokens[6] == "weight") {
                return CatalogRole::Sensitive;
            }
            if (tokens.size() == 8 && tokens[5] == "self_attn" && tokens[7] == "weight" &&
                one_of(tokens[6], { "q_proj", "k_proj", "v_proj", "o_proj" })) {
                return CatalogRole::MatrixWeight;
            }
            if (tokens.size() == 8 && tokens[5] == "mlp" && tokens[7] == "weight" &&
                one_of(tokens[6], { "gate_proj", "up_proj", "down_proj" })) {
                return CatalogRole::MatrixWeight;
            }
        }
        return CatalogRole::Unknown;
    }
    if (tokens.size() > 4 && tokens[2] == "upsample" && is_index(tokens[3])) {
        // The transposed convolution that upsamples, then a ConvNeXt block.
        if (tokens.size() == 7 && tokens[4] == "0" && tokens[5] == "conv" && one_of(tokens[6], { "weight", "bias" })) {
            return tokens[6] == "weight" ? CatalogRole::TransposeWeight : CatalogRole::Sensitive;
        }
        if (tokens[4] != "1") {
            return CatalogRole::Unknown;
        }
        // A depthwise kernel is one filter per channel; packing it would flatten
        // the channels together, and it is 28 kB.
        if (tokens.size() == 8 && tokens[5] == "dwconv" && tokens[6] == "conv" &&
            one_of(tokens[7], { "weight", "bias" })) {
            return CatalogRole::Sensitive;
        }
        if (tokens.size() == 7 && tokens[5] == "norm" && one_of(tokens[6], { "weight", "bias" })) {
            return CatalogRole::Sensitive;
        }
        if (tokens.size() == 6 && tokens[5] == "gamma") {
            return CatalogRole::Sensitive;
        }
        if (tokens.size() == 7 && one_of(tokens[5], { "pwconv1", "pwconv2" }) &&
            one_of(tokens[6], { "weight", "bias" })) {
            return tokens[6] == "weight" ? CatalogRole::MatrixWeight : CatalogRole::Sensitive;
        }
        return CatalogRole::Unknown;
    }
    if (tokens.size() > 4 && tokens[2] == "decoder" && is_index(tokens[3])) {
        // A flat ModuleList: an input convolution, one block per upsample rate,
        // then the output activation and convolution.
        if (tokens.size() == 6 && tokens[4] == "conv" && one_of(tokens[5], { "weight", "bias" })) {
            return tokens[5] == "weight" ? CatalogRole::MatrixWeight : CatalogRole::Sensitive;
        }
        if (tokens.size() == 5 && one_of(tokens[4], { "alpha", "beta" })) {
            return CatalogRole::Sensitive;
        }
        if (tokens.size() > 5 && tokens[4] == "block" && is_index(tokens[5])) {
            if (tokens.size() == 7 && one_of(tokens[6], { "alpha", "beta" })) {
                return CatalogRole::Sensitive;
            }
            if (tokens.size() == 8 && tokens[6] == "conv" && one_of(tokens[7], { "weight", "bias" })) {
                return tokens[7] == "weight" ? CatalogRole::TransposeWeight : CatalogRole::Sensitive;
            }
            if (tokens.size() == 8 && one_of(tokens[6], { "act1", "act2" }) && one_of(tokens[7], { "alpha", "beta" })) {
                return CatalogRole::Sensitive;
            }
            if (tokens.size() == 9 && one_of(tokens[6], { "conv1", "conv2" }) && tokens[7] == "conv" &&
                one_of(tokens[8], { "weight", "bias" })) {
                return tokens[8] == "weight" ? CatalogRole::MatrixWeight : CatalogRole::Sensitive;
            }
        }
        return CatalogRole::Unknown;
    }
    return CatalogRole::Unknown;
}

CatalogRole classify_qwen3_tts_tensor(const std::string & name) {
    const std::vector<std::string_view> tokens = split_name(name);
    const CatalogRole                   talker = classify_qwen3_talker(tokens);
    return talker != CatalogRole::Unknown ? talker : classify_qwen3_codec(tokens);
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

bool resolve_qwen3_tts_target_spec(const Profile & profile, const std::string & name, TargetSpec & spec_out) {
    switch (classify_qwen3_tts_tensor(name)) {
        case CatalogRole::MatrixWeight:
            spec_out = { profile.matrix_weight_type, profile.matrix_weight_layout };
            return true;
        case CatalogRole::TransposeWeight:
            // The same override VITS makes and for the same reason: these run as
            // a column matrix multiply into col2im_1d, and CUDA's F16 matrix
            // multiply accumulates in half precision. There are six of them.
            spec_out = { GGML_TYPE_F32, TensorLayout::Native };
            return true;
        case CatalogRole::Sensitive:
            spec_out = { profile.sensitive_type, TensorLayout::Native };
            return true;
        case CatalogRole::Unknown:
            return false;
    }
    return false;
}

bool resolve_qwen3_tts_target_type(const Profile & profile, const std::string & name, ggml_type & type_out) {
    TargetSpec spec{};
    if (!resolve_qwen3_tts_target_spec(profile, name, spec)) {
        return false;
    }
    type_out = spec.type;
    return true;
}

bool resolve_vits_target_spec(const Profile & profile, const std::string & name, TargetSpec & spec_out) {
    switch (classify_vits_tensor(name)) {
        case CatalogRole::MatrixWeight:
            spec_out = { profile.matrix_weight_type, profile.matrix_weight_layout };
            return true;
        case CatalogRole::TransposeWeight:
            // VITS overrides the profile here. Its transposed convolutions run
            // as a column matrix multiply plus col2im_1d, and CUDA's F16 matrix
            // multiply accumulates in half precision, so halving these eight
            // tensors would cost about 3e-3 relative on the decoder's output to
            // save roughly 5 MB. Kokoro keeps the profile type: its per-tap
            // decomposition already shipped with F16 weights and measured
            // tolerances to match.
            spec_out = { GGML_TYPE_F32, TensorLayout::Native };
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

bool resolve_omnivoice_target_spec(const Profile & profile, const std::string & name, TargetSpec & spec_out) {
    // The classifier lives in the family module so the runtime's catalog and
    // this tool cannot disagree about a tensor. `ne` is unused by every role
    // this catalog resolves today (see quantization.h's own header comment),
    // so a placeholder shape is enough here -- this dispatch, like every
    // sibling family's, has no tensor shape on hand at this call site.
    const int64_t ne[4] = { 1, 1, 1, 1 };
    switch (synth::omnivoice::classify_tensor(name, ne)) {
        case synth::omnivoice::QuantRole::MatrixWeight:
            spec_out = { profile.matrix_weight_type, profile.matrix_weight_layout };
            return true;
        case synth::omnivoice::QuantRole::TransposeWeight:
            // The same override VITS and Qwen3-TTS's decoder make, and for
            // the same reason: this runs as a column matrix multiply into
            // col2im_1d, and CUDA's F16 matrix multiply accumulates in half
            // precision (quantization.h's TransposeWeight comment).
            spec_out = { GGML_TYPE_F32, TensorLayout::Native };
            return true;
        case synth::omnivoice::QuantRole::Sensitive:
            spec_out = { profile.sensitive_type, TensorLayout::Native };
            return true;
        case synth::omnivoice::QuantRole::Unknown:
            return false;
    }
    return false;
}

bool resolve_omnivoice_target_type(const Profile & profile, const std::string & name, ggml_type & type_out) {
    TargetSpec spec{};
    if (!resolve_omnivoice_target_spec(profile, name, spec)) {
        return false;
    }
    type_out = spec.type;
    return true;
}

}  // namespace synth::quantize
