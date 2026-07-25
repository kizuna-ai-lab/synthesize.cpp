#include "quantization.h"

#include <initializer_list>
#include <string_view>
#include <vector>

namespace synth::kokoro {

namespace {

using CatalogRole = TensorRole;

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

// One AdaINResBlock1 branch: three dilated convolution pairs, each with its own
// AdaIN projection and Snake alpha. The alphas are divided by, so they never
// leave the reference dtype however the rest of the block is stored.
CatalogRole classify_kokoro_snake_block(const std::vector<std::string_view> & tokens, size_t offset) {
    const size_t remaining = tokens.size() - offset;
    if (remaining == 2 && one_of(tokens[offset], { "alpha1", "alpha2" }) && is_index(tokens[offset + 1])) {
        return CatalogRole::Sensitive;
    }
    if (remaining == 3 && one_of(tokens[offset], { "convs1", "convs2" }) && is_index(tokens[offset + 1]) &&
        one_of(tokens[offset + 2], { "weight", "bias" })) {
        return tokens[offset + 2] == "weight" ? CatalogRole::MatrixWeight : CatalogRole::Sensitive;
    }
    if (remaining == 4 && one_of(tokens[offset], { "adain1", "adain2" }) && is_index(tokens[offset + 1]) &&
        tokens[offset + 2] == "fc" && one_of(tokens[offset + 3], { "weight", "bias" })) {
        return tokens[offset + 3] == "weight" ? CatalogRole::MatrixWeight : CatalogRole::Sensitive;
    }
    return CatalogRole::Unknown;
}

// One AdainResBlk1d: two convolutions, two AdaIN projections, an optional
// channel-matching projection, and an optional upsampling pool. The pool is a
// depthwise transposed convolution whose taps this runtime reads one at a time
// as a scale, which only works at the reference dtype.
CatalogRole classify_kokoro_adain_block(const std::vector<std::string_view> & tokens, size_t offset) {
    const size_t remaining = tokens.size() - offset;
    if (remaining == 2 && one_of(tokens[offset], { "conv1", "conv2", "pool" }) &&
        one_of(tokens[offset + 1], { "weight", "bias" })) {
        if (tokens[offset] == "pool") {
            return CatalogRole::Sensitive;
        }
        return tokens[offset + 1] == "weight" ? CatalogRole::MatrixWeight : CatalogRole::Sensitive;
    }
    if (remaining == 2 && tokens[offset] == "conv1x1" && tokens[offset + 1] == "weight") {
        return CatalogRole::MatrixWeight;
    }
    if (remaining == 3 && one_of(tokens[offset], { "norm1", "norm2" }) && tokens[offset + 1] == "fc" &&
        one_of(tokens[offset + 2], { "weight", "bias" })) {
        return tokens[offset + 2] == "weight" ? CatalogRole::MatrixWeight : CatalogRole::Sensitive;
    }
    return CatalogRole::Unknown;
}

CatalogRole classify_kokoro_generator(const std::vector<std::string_view> & tokens) {
    // Two tokens in: decoder.generator.<rest>
    const size_t base = 2;
    if (tokens.size() == 4 && tokens[base] == "conv_post" && one_of(tokens[3], { "weight", "bias" })) {
        return tokens[3] == "weight" ? CatalogRole::MatrixWeight : CatalogRole::Sensitive;
    }
    // The nine-to-one projection that merges the harmonics is nine numbers, and
    // it feeds the phase accumulator; it stays exact.
    if (tokens.size() == 5 && tokens[base] == "m_source" && tokens[3] == "l_linear" &&
        one_of(tokens[4], { "weight", "bias" })) {
        return CatalogRole::Sensitive;
    }
    if (tokens.size() == 5 && tokens[base] == "ups" && is_index(tokens[3]) &&
        one_of(tokens[4], { "weight", "bias" })) {
        return tokens[4] == "weight" ? CatalogRole::TransposeWeight : CatalogRole::Sensitive;
    }
    if (tokens.size() == 5 && tokens[base] == "noise_convs" && is_index(tokens[3]) &&
        one_of(tokens[4], { "weight", "bias" })) {
        return tokens[4] == "weight" ? CatalogRole::MatrixWeight : CatalogRole::Sensitive;
    }
    if (tokens.size() >= 6 && one_of(tokens[base], { "noise_res", "resblocks" }) && is_index(tokens[3])) {
        return classify_kokoro_snake_block(tokens, 4);
    }
    return CatalogRole::Unknown;
}

CatalogRole classify_kokoro_decoder(const std::vector<std::string_view> & tokens) {
    if (tokens.empty() || tokens[0] != "decoder") {
        return CatalogRole::Unknown;
    }
    if (tokens.size() >= 3 && tokens[1] == "generator") {
        return classify_kokoro_generator(tokens);
    }
    if (tokens.size() == 3 && one_of(tokens[1], { "F0_conv", "N_conv" }) && one_of(tokens[2], { "weight", "bias" })) {
        // These two halve the prosody curves before anything else reads them,
        // so they sit on the F0 path rather than the audio path.
        return CatalogRole::Sensitive;
    }
    if (tokens.size() == 4 && tokens[1] == "asr_res" && is_index(tokens[2]) &&
        one_of(tokens[3], { "weight", "bias" })) {
        return tokens[3] == "weight" ? CatalogRole::MatrixWeight : CatalogRole::Sensitive;
    }
    if (tokens.size() >= 3 && tokens[1] == "encode") {
        return classify_kokoro_adain_block(tokens, 2);
    }
    if (tokens.size() >= 4 && tokens[1] == "decode" && is_index(tokens[2])) {
        return classify_kokoro_adain_block(tokens, 3);
    }
    return CatalogRole::Unknown;
}

// Everything that decides F0 or the durations. A relative F0 difference of a
// few parts in ten thousand becomes radians of excitation phase by the end of
// an utterance, so none of this is quantized at any profile.
CatalogRole classify_kokoro_structural(const std::vector<std::string_view> & tokens) {
    if (tokens.empty()) {
        return CatalogRole::Unknown;
    }
    if (tokens[0] == "voice" && tokens.size() == 2 && !tokens[1].empty()) {
        return CatalogRole::Sensitive;
    }
    if (tokens[0] == "bert" || tokens[0] == "bert_encoder" || tokens[0] == "predictor" ||
        tokens[0] == "text_encoder") {
        return CatalogRole::Sensitive;
    }
    return CatalogRole::Unknown;
}

CatalogRole classify(const std::string & name) {
    const std::vector<std::string_view> tokens = split_name(name);
    for (CatalogRole role : { classify_kokoro_structural(tokens), classify_kokoro_decoder(tokens) }) {
        if (role != CatalogRole::Unknown) {
            return role;
        }
    }
    return CatalogRole::Unknown;
}

}  // namespace

TensorRole tensor_role(const std::string & name) {
    return classify(name);
}

}  // namespace synth::kokoro
