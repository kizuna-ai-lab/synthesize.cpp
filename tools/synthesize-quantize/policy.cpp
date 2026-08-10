#include "policy.h"

#include "arch/kokoro/quantization.h"
#include "arch/omnivoice/quantization.h"

#include <cctype>
#include <initializer_list>
#include <string_view>
#include <vector>

namespace synth::quantize {

namespace {

// The `row_lookup_type` column is read only by a family whose classifier
// reports QuantRole::RowLookup, which today is omnivoice alone; for the other
// three it is inert (policy.h).
//
// Naming, ruled by jiangzhuo on 2026-08-09: for omnivoice -- the one family
// whose profiles do not all quantize the same half -- **the half a profile
// quantizes determines its name**. The generator half takes the plain name
// (`F16`, `Q8`, `Q4_K`, `BF16`) and the codec half takes a `_CODEC` qualifier
// (`F16_CODEC`, `Q8_CODEC_MIXED`). This lands omnivoice's shipping profiles on
// the names the three sibling families already publish, and it is a rule about
// the artifact rather than about ship status, so a codec profile becoming
// shippable later does not force another rename. The former `_GEN`-suffixed
// names are gone; see the porting log's 2026-08-09 entry for the full mapping.
//
// Which rows are shared and which are one family's is load-bearing here. `F16`,
// `Q8_MIXED` and `Q5_K_MIXED` predate omnivoice and are what VITS, Kokoro and
// Qwen3-TTS publish; their names are a public contract with already-shipped
// packages and cannot move. So omnivoice's codec-half profiles get their own
// rows below rather than renaming those, while omnivoice's generator-half F16
// reuses the shared `F16` row -- see omnivoice_quantized_half for why that row
// serves both meanings without ambiguity.
const Profile kProfiles[] = {
    // Shared. For VITS/Kokoro/Qwen3-TTS this is the codec/decoder-half F16 it
    // has always been. For omnivoice it is the **generator** half, per the
    // naming rule above; `transpose_weight_type` is inert on that path, so the
    // one row means both things without a second F16 entry. See
    // omnivoice_quantized_half.
    { "F16",            GGML_TYPE_F16,  TensorLayout::Native,       GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_F16,  1, 1 },
    // Shared, and NOT an omnivoice profile: omnivoice's codec-half mixed
    // profile is `Q8_CODEC_MIXED` below. Cutting an omnivoice package with this
    // name still produces a codec-half package -- omnivoice_quantized_half's
    // default -- but the omnivoice runtime refuses to load the resulting
    // profile string, which is the same loud refusal `Q5_K_MIXED` already gets
    // from that family.
    { "Q8_MIXED",       GGML_TYPE_Q8_0, TensorLayout::PackedMatrix, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q8_0,
     GGML_FTYPE_MOSTLY_Q8_0,                                                                                         1 },
    // Q5_K is a super-block of 256, so it needs a row four times longer than
    // Q8_0 does. Qwen3-TTS clears that everywhere it quantizes -- its rows are
    // 1024, 2048 and 3072 -- while a family whose matrix weights are packed
    // convolution kernels will not, and is refused by the row-size check with
    // the tensor named rather than by a rule here.
    { "Q5_K_MIXED",     GGML_TYPE_Q5_K, TensorLayout::PackedMatrix, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q8_0,
     GGML_FTYPE_MOSTLY_Q5_K,                                                                                         1 },
    // OmniVoice's two codec-half profiles. Field-for-field these duplicate the
    // shared `F16` and `Q8_MIXED` rows above; only the name differs, and the
    // name is the whole point -- it is what omnivoice_quantized_half reads to
    // decide the half, and what the package's
    // `synthesize.quantization.profile` string then carries to a reader. They
    // exist as separate rows rather than as a rename of the shared pair
    // because those two names belong to three other families' published
    // packages. Both are BLOCKED and unpublished for omnivoice: the codec is
    // only 23.1% of this package's tensor bytes, so no codec-only profile
    // reaches the sizes this family needs.
    { "F16_CODEC",      GGML_TYPE_F16,  TensorLayout::Native,       GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_F16,  1, 1 },
    { "Q8_CODEC_MIXED", GGML_TYPE_Q8_0, TensorLayout::PackedMatrix, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q8_0,
     GGML_FTYPE_MOSTLY_Q8_0,                                                                                         1 },
    // OmniVoice's generator-half Q8_0 profile, and the first profile in this
    // table that quantizes something other than a codec/decoder. Its matrix
    // weight layout says PackedMatrix like its siblings, but nothing under it
    // is ever actually packed: every generator matrix is two-dimensional and
    // quantize.cpp demotes a two-dimensional matrix_family tensor to Native,
    // because packing a [1024, 3072] projection would flatten it into one
    // meaningless row of 3,145,728. The codec stays F32 not because
    // `sensitive_type` says so but because this profile's classifier arm holds
    // the whole codec half at Sensitive -- see resolve_omnivoice_target_spec.
    { "Q8",             GGML_TYPE_Q8_0, TensorLayout::PackedMatrix, GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_Q8_0,
     GGML_FTYPE_MOSTLY_Q8_0,                                                                                         1 },
    // The same half as Q8, four bits deep -- and the first profile in this
    // table where `row_lookup_type` differs from `matrix_weight_type` rather
    // than agreeing with it by coincidence. Q4_K is a k-quant, and CUDA's
    // GET_ROWS accepts none: its type list is F16/F32/BF16/I32/Q1_0/Q4_0/Q4_1/
    // Q5_0/Q5_1/Q8_0 (ggml/src/ggml-cuda/ggml-cuda.cu:5190-5207), where its
    // matrix multiply takes every k-quant. A k-quant on `llm.embed_tokens.
    // weight` or `audio_embeddings.weight` therefore does not fail loudly --
    // the scheduler silently places those nodes on the CPU. So the two lookup
    // tables are pinned to Q8_0, at a measured cost of 81,856,512 bytes
    // (78.06 MiB) against a package that k-quantized them too.
    //
    // Q4_K's super-block is 256 elements against Q8_0's 32, which this
    // family's generator clears everywhere: all 199 of its two-dimensional
    // weights have rows of 1024, 2048 or 3072. A row that did not would be
    // refused by quantize.cpp's row-size check with the tensor named, not
    // silently demoted.
    { "Q4_K",           GGML_TYPE_Q4_K, TensorLayout::PackedMatrix, GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_Q8_0,
     GGML_FTYPE_MOSTLY_Q4_K,                                                                                         1 },
    // There is deliberately no fourth generator row for F16. The same
    // generator half at a narrowed reference dtype rather than a
    // block-quantized one is cut with the shared `F16` row at the top of this
    // table, which omnivoice_quantized_half reads as the generator half. That
    // row's only disagreement with a hypothetical omnivoice-specific one is
    // `transpose_weight_type` (F16 there, F32 here), and that column is never
    // read on a generator-half omnivoice cut: TransposeWeight hardcodes F32 in
    // resolve_omnivoice_target_spec, and the ConvKernel arm that does read the
    // column cannot be reached, because classify_tensor_for_half holds the
    // whole codec half -- where every convolution in this family lives -- at
    // Sensitive. Verified by re-cutting rather than by reading: against the
    // package the removed `F16_GEN` row produced, the only differing GGUF KV
    // value is `synthesize.quantization.profile` itself, the 798 tensor infos
    // are identical, and no byte at or after the tensor-data offset differs.
    //
    // F16 is viable on this backend for the two ops this half uses: it has a
    // native CUDA matrix-multiply path (ggml/src/ggml-cuda/mmf.cu's
    // GGML_TYPE_F16 case, MMA-backed on Ampere+) and a native GET_ROWS path
    // (ggml/src/ggml-cuda/getrows.cu), so unlike Q8_0/Q4_K it never
    // dequantizes before either.
    //
    // bfloat16 rather than IEEE half: wider exponent, fewer mantissa bits, no
    // dynamic-range rescaling needed going in or out. Verified viable on this
    // backend by reading ggml-cuda.cu's own `supports_op` before writing this
    // row rather than assuming it from F16's presence: MUL_MAT accepts
    // GGML_TYPE_BF16 (ggml/src/ggml-cuda/ggml-cuda.cu:5121-5187, checked at
    // line 5182) and so does GET_ROWS
    // (ggml/src/ggml-cuda/ggml-cuda.cu:5190-5207, checked at line 5195) --
    // the same file the RowLookup role's own doc comment cites for the
    // narrower list Q8_0 and Q4_K sit inside. mmf.cu's
    // `ggml_cuda_should_use_mmf` further confirms BF16 reaches the same
    // MMA-backed kernel F16 does on Ampere-and-later compute capability, not
    // a dequantize-then-F32 fallback. `matrix_weight_layout` is Native because
    // BF16, like F16, is not `ggml_is_quantized`.
    { "BF16",           GGML_TYPE_BF16, TensorLayout::Native,       GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_BF16,
     GGML_FTYPE_MOSTLY_BF16,                                                                                         1 },
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

namespace {

// Which half of the package a profile quantizes. Deliberately keyed on the
// profile's own name rather than on a `Profile` field: the half is meaningless
// for the other three families, and this family is the only reason it exists.
// The runtime states the same mapping over its own profile enum
// (src/arch/omnivoice/catalog.cpp's expected_type) -- the pairing is the same
// one weights.cpp already maintains between these names and that enum.
//
// Since jiangzhuo's naming ruling of 2026-08-09 the name IS the half, so this
// function reads as the rule rather than as a lookup table that happens to
// agree with one: the four plain names are the generator, and anything else --
// `F16_CODEC`, `Q8_CODEC_MIXED`, and the sibling families' `Q8_MIXED` and
// `Q5_K_MIXED` -- is the codec. Note `F16` is listed here and is also the row
// VITS, Kokoro and Qwen3-TTS cut their codec/decoder halves with; that is not
// a contradiction, because this function is only ever consulted on the
// omnivoice dispatch path (quantize.cpp keys on general.architecture).
//
// The two sibling names falling through to Codec is a reachable but harmless
// state: cutting an omnivoice package as `Q8_MIXED` writes a codec-half
// package whose profile string the omnivoice runtime then refuses by name
// (weights.cpp's read_quantization), which is the same loud refusal
// `Q5_K_MIXED` has always got from this family.
synth::omnivoice::ModelHalf omnivoice_quantized_half(const Profile & profile) {
    return iequals(profile.name, "Q8") || iequals(profile.name, "Q4_K") || iequals(profile.name, "F16") ||
                   iequals(profile.name, "BF16") ?
               synth::omnivoice::ModelHalf::Generator :
               synth::omnivoice::ModelHalf::Codec;
}

}  // namespace

bool resolve_omnivoice_target_spec(const Profile & profile, const std::string & name, TargetSpec & spec_out) {
    // The classifier lives in the family module so the runtime's catalog and
    // this tool cannot disagree about a tensor. `ne` is unused by every role
    // this catalog resolves today (see quantization.h's own header comment),
    // so a placeholder shape is enough here -- this dispatch, like every
    // sibling family's, has no tensor shape on hand at this call site.
    const int64_t ne[4] = { 1, 1, 1, 1 };
    switch (synth::omnivoice::classify_tensor_for_half(name, ne, omnivoice_quantized_half(profile))) {
        case synth::omnivoice::QuantRole::MatrixWeight:
            spec_out = { profile.matrix_weight_type, profile.matrix_weight_layout };
            return true;
        case synth::omnivoice::QuantRole::RowLookup:
            // Never packed: `ggml_get_rows` indexes whole rows of the
            // tensor's declared shape, and the type is the profile's own
            // narrower row-lookup column because CUDA's GET_ROWS accepts no
            // k-quant (quantization.h's RowLookup).
            spec_out = { profile.row_lookup_type, TensorLayout::Native };
            return true;
        case synth::omnivoice::QuantRole::TransposeWeight:
            // The same override VITS and Qwen3-TTS's decoder make, and for
            // the same reason: this runs as a column matrix multiply into
            // col2im_1d, and CUDA's F16 matrix multiply accumulates in half
            // precision (quantization.h's TransposeWeight comment).
            spec_out = { GGML_TYPE_F32, TensorLayout::Native };
            return true;
        case synth::omnivoice::QuantRole::ConvKernel:
            // The conv-exempt codec policy: never block-quantized, never
            // packed, held at the profile's halved fallback column at its
            // native three-axis shape. That column is the same one Kokoro's
            // quantizer falls back to for a matrix it will not block-quantize
            // (quantize.cpp's matrix_is_block_quantizable branch), which is
            // what it means here too -- not "this is a transposed
            // convolution". Under `Q8_CODEC_MIXED` and `F16_CODEC` it is F16;
            // under the four generator-half profiles the codec is all
            // Sensitive and this arm is never reached.
            spec_out = { profile.transpose_weight_type, TensorLayout::Native };
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
