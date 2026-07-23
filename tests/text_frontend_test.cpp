#include "text-frontend.h"
#include "test-assert.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

synth::SymbolMapFrontendConfig valid_config() {
    synth::SymbolMapFrontendConfig config;
    config.provider_id      = "synthesize.symbol_map";
    config.contract_version = 1;
    config.mapping_mode     = synth::SymbolMappingMode::UnicodeScalar;
    config.symbols          = { "_", "a", "ˈ", "ɪ", "t", "s", "a" };
    config.blank_id         = 0;
    config.add_blank        = true;
    return config;
}

}  // namespace

int main() {
    std::unique_ptr<synth::TextFrontend> frontend;
    synth::SymbolMapFrontendConfig       config = valid_config();
    SYNTH_TEST_CHECK(synth::make_symbol_map_frontend(config, frontend) == SYNTH_OK);
    SYNTH_TEST_CHECK(frontend != nullptr);
    SYNTH_TEST_CHECK(frontend->input_flags() == SYNTH_INPUT_SUPPORT_PHONEMES_UTF8);

    const std::string phonemes = "aˈɪts";
    std::vector<int32_t> tokens;
    SYNTH_TEST_CHECK(frontend->prepare(SYNTH_INPUT_PHONEMES_UTF8, phonemes.data(), phonemes.size(), 64, tokens) ==
                     SYNTH_OK);
    const std::vector<int32_t> expected = { 0, 6, 0, 2, 0, 3, 0, 4, 0, 5, 0 };
    SYNTH_TEST_CHECK(tokens == expected);

    // TOKEN_IDS bypasses the frontend, and this adapter does not claim G2P.
    SYNTH_TEST_CHECK(frontend->prepare(SYNTH_INPUT_TEXT_UTF8, phonemes.data(), phonemes.size(), 64, tokens) ==
                     SYNTH_ERR_UNSUPPORTED_INPUT);
    SYNTH_TEST_CHECK(tokens.empty());
    SYNTH_TEST_CHECK(frontend->prepare(SYNTH_INPUT_TOKEN_IDS, phonemes.data(), phonemes.size(), 64, tokens) ==
                     SYNTH_ERR_UNSUPPORTED_INPUT);
    SYNTH_TEST_CHECK(tokens.empty());

    const char invalid_utf8[] = { static_cast<char>(0xc0), static_cast<char>(0xaf) };
    SYNTH_TEST_CHECK(frontend->prepare(SYNTH_INPUT_PHONEMES_UTF8, invalid_utf8, sizeof(invalid_utf8), 64, tokens) ==
                     SYNTH_ERR_TEXT_FRONTEND);
    SYNTH_TEST_CHECK(tokens.empty());

    const std::string unknown = "z";
    SYNTH_TEST_CHECK(frontend->prepare(SYNTH_INPUT_PHONEMES_UTF8, unknown.data(), unknown.size(), 64, tokens) ==
                     SYNTH_ERR_TEXT_FRONTEND);
    SYNTH_TEST_CHECK(tokens.empty());

    SYNTH_TEST_CHECK(frontend->prepare(SYNTH_INPUT_PHONEMES_UTF8, nullptr, 1, 64, tokens) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(frontend->prepare(SYNTH_INPUT_PHONEMES_UTF8, phonemes.data(), 0, 64, tokens) ==
                     SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(frontend->prepare(SYNTH_INPUT_PHONEMES_UTF8, phonemes.data(), phonemes.size(), 10, tokens) ==
                     SYNTH_ERR_INPUT_TOO_LONG);

    config = valid_config();
    config.provider_id = "other.symbol_map";
    SYNTH_TEST_CHECK(synth::make_symbol_map_frontend(config, frontend) == SYNTH_ERR_TEXT_FRONTEND);
    SYNTH_TEST_CHECK(frontend == nullptr);

    config                     = valid_config();
    config.contract_version    = 2;
    SYNTH_TEST_CHECK(synth::make_symbol_map_frontend(config, frontend) == SYNTH_ERR_TEXT_FRONTEND);
    SYNTH_TEST_CHECK(frontend == nullptr);

    config              = valid_config();
    config.symbols[2]   = "\xc0\xaf";
    SYNTH_TEST_CHECK(synth::make_symbol_map_frontend(config, frontend) == SYNTH_ERR_TEXT_FRONTEND);
    SYNTH_TEST_CHECK(frontend == nullptr);

    config          = valid_config();
    config.blank_id = static_cast<uint32_t>(config.symbols.size());
    SYNTH_TEST_CHECK(synth::make_symbol_map_frontend(config, frontend) == SYNTH_ERR_TEXT_FRONTEND);
    SYNTH_TEST_CHECK(frontend == nullptr);
    return 0;
}
