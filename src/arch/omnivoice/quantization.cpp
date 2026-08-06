#include "quantization.h"

#include <initializer_list>
#include <string_view>
#include <vector>

namespace synth::omnivoice {

namespace {

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

// The mask-predict generator: the Qwen3 layer stack (`llm.*`) plus its two
// canvas tables, which are a second table each rather than a tied pair.
// jiangzhuo's ruling of 2026-08-06 keeps all of it at the reference dtype
// regardless of shape: a reference port measured exact-token agreement
// collapsing from 100% to roughly 7% with an F16 generator, and this
// family's headline claim is exact tokens.
bool is_generator_tensor(const std::vector<std::string_view> & tokens, const std::string & name) {
    // "llm.*" requires at least one child segment; a bare "llm" names nothing
    // in the catalog and falls through to Unknown like any other stray name.
    if (tokens.size() >= 2 && tokens[0] == "llm") {
        return true;
    }
    return name == "audio_embeddings.weight" || name == "audio_heads.weight";
}

// codec.quantizer.*, codec.fc.* and codec.fc2.*: the RVQ and the two
// projections that concatenate the acoustic and semantic latents around it.
// The RVQ rule matches qwen3-tts's own (policy.cpp:258-265): a residual
// codebook's later levels carry small magnitudes, so a relative error there
// is large against the residual it corrects, and the whole quantizer here is
// smaller still (40 tensors against qwen3-tts's own codebook set).
bool is_blanket_sensitive_codec_prefix(const std::vector<std::string_view> & tokens) {
    return tokens.size() >= 2 && one_of(tokens[1], { "quantizer", "fc", "fc2" });
}

// codec.acoustic_decoder., codec.acoustic_encoder., codec.semantic_model. and
// codec.encoder_semantic.: the four codec modules whose parameters are read
// through a matrix multiply. `tokens[0]` is "codec" and `tokens[1]` is the
// module name; every decision below reads only the tokens past that, so it
// applies identically to all four.
QuantRole classify_codec_matrix_region(const std::vector<std::string_view> & tokens, const std::string & name) {
    const std::string_view suffix = tokens.back();
    // Every bias is added elementwise, and every Snake curve is divided by
    // elementwise; neither is ever read through a matrix multiply.
    if (suffix == "bias" || suffix == "alpha") {
        return QuantRole::Sensitive;
    }
    if (suffix != "weight") {
        return QuantRole::Unknown;
    }

    // A LayerNorm's weight is one curve per channel, read elementwise rather
    // than through a matrix multiply. This one token check covers all five
    // norm sites in this region -- the HuBERT feature group norm, the
    // feature projection's norm, each encoder layer's two norms, and the
    // encoder's own exit norm -- instead of five separate name patterns.
    if (tokens.size() >= 2) {
        const std::string_view parent = tokens[tokens.size() - 2];
        if (parent == "layer_norm" || parent == "final_layer_norm") {
            return QuantRole::Sensitive;
        }
    }

    // Three tensors are Sensitive for reasons specific to their shape or to
    // how this runtime reads them, named individually rather than folded
    // into a generic rule -- a future reader must not "fix" any of them.
    if (name == "codec.acoustic_encoder.conv1.weight") {
        // Reads the raw mono reference waveform, so its packed row is
        // kernel * in_channels = 7 * 1 = 7: seven elements, never a multiple
        // of the 32-element block. The input channel count is architectural
        // (this convolution's whole purpose is to read one channel), not a
        // hyperparameter that could grow at another scale.
        return QuantRole::Sensitive;
    }
    if (name == "codec.semantic_model.feat_conv.0.conv.weight") {
        // The HuBERT feature extractor's first convolution also reads a raw
        // single-channel waveform, so its packed row is kernel * in_channels
        // = 10 * 1 = 10 -- again never a multiple of 32, for the same reason.
        return QuantRole::Sensitive;
    }
    if (name == "codec.semantic_model.encoder.pos_conv_embed.conv.weight") {
        // Grouped by sixteen. reference-encoder.cpp's grouped_conv1d slices
        // this kernel per group with ggml_view_3d, which walks the tensor's
        // native [kernel, in, out] layout; a block-quantized or packed
        // layout has no such view to offer it. Its packed row (128 * 48 =
        // 6144) is itself divisible by 32, so nothing about its size would
        // otherwise stop a future reader from "optimizing" it away -- the
        // grouped view this runtime takes of it is the whole reason.
        return QuantRole::Sensitive;
    }

    // codec.acoustic_decoder.block.<i>.conv_t1.weight: the transposed
    // convolution that upsamples each decoder block by its ratio. The same
    // override VITS and Qwen3-TTS's decoder make, and for the same reason
    // (policy.cpp:390-395, docs/quantization.md:52-56).
    if (tokens.size() == 6 && tokens[1] == "acoustic_decoder" && tokens[2] == "block" && is_index(tokens[3]) &&
        tokens[4] == "conv_t1") {
        return QuantRole::TransposeWeight;
    }

    return QuantRole::MatrixWeight;
}

}  // namespace

QuantRole classify_tensor(const std::string & name, const int64_t ne[4]) {
    // Every role this catalog resolves today follows from the name alone; see
    // the header comment for why the parameter still exists.
    (void) ne;

    const std::vector<std::string_view> tokens = split_name(name);
    if (is_generator_tensor(tokens, name)) {
        return QuantRole::Sensitive;
    }
    if (tokens.size() < 2 || tokens[0] != "codec") {
        return QuantRole::Unknown;
    }
    if (is_blanket_sensitive_codec_prefix(tokens)) {
        return QuantRole::Sensitive;
    }
    if (one_of(tokens[1], { "acoustic_decoder", "acoustic_encoder", "semantic_model", "encoder_semantic" })) {
        return classify_codec_matrix_region(tokens, name);
    }
    return QuantRole::Unknown;
}

}  // namespace synth::omnivoice
