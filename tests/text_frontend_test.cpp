#include "test-assert.h"
#include "text-frontend.h"

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
    config.padding_rule     = synth::SymbolPaddingRule::InterleavedBlank;
    return config;
}

}  // namespace

int main() {
    std::unique_ptr<synth::TextFrontend> frontend;
    synth::SymbolMapFrontendConfig       config = valid_config();
    SYNTH_TEST_CHECK(synth::make_symbol_map_frontend(config, frontend) == SYNTH_OK);
    SYNTH_TEST_CHECK(frontend != nullptr);
    SYNTH_TEST_CHECK(frontend->input_flags() == SYNTH_INPUT_SUPPORT_PHONEMES_UTF8);

    const std::string    phonemes = "aˈɪts";
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

    config             = valid_config();
    config.provider_id = "other.symbol_map";
    SYNTH_TEST_CHECK(synth::make_symbol_map_frontend(config, frontend) == SYNTH_ERR_TEXT_FRONTEND);
    SYNTH_TEST_CHECK(frontend == nullptr);

    config                  = valid_config();
    config.contract_version = 2;
    SYNTH_TEST_CHECK(synth::make_symbol_map_frontend(config, frontend) == SYNTH_ERR_TEXT_FRONTEND);
    SYNTH_TEST_CHECK(frontend == nullptr);

    config            = valid_config();
    config.symbols[2] = "\xc0\xaf";
    SYNTH_TEST_CHECK(synth::make_symbol_map_frontend(config, frontend) == SYNTH_ERR_TEXT_FRONTEND);
    SYNTH_TEST_CHECK(frontend == nullptr);

    config          = valid_config();
    config.blank_id = static_cast<uint32_t>(config.symbols.size());
    SYNTH_TEST_CHECK(synth::make_symbol_map_frontend(config, frontend) == SYNTH_ERR_TEXT_FRONTEND);
    SYNTH_TEST_CHECK(frontend == nullptr);

    // A package may pad by wrapping the mapped sequence with one pad token
    // instead of interleaving blanks.
    config              = valid_config();
    config.padding_rule = synth::SymbolPaddingRule::WrapPadToken;
    SYNTH_TEST_CHECK(synth::make_symbol_map_frontend(config, frontend) == SYNTH_OK);
    SYNTH_TEST_CHECK(frontend->prepare(SYNTH_INPUT_PHONEMES_UTF8, phonemes.data(), phonemes.size(), 64, tokens) ==
                     SYNTH_OK);
    const std::vector<int32_t> wrapped = { 0, 6, 2, 3, 4, 5, 0 };
    SYNTH_TEST_CHECK(tokens == wrapped);

    // The limit applies to the final wrapped length, so a budget one short of
    // the trailing pad is rejected rather than silently truncated.
    SYNTH_TEST_CHECK(frontend->prepare(SYNTH_INPUT_PHONEMES_UTF8, phonemes.data(), phonemes.size(), wrapped.size(),
                                       tokens) == SYNTH_OK);
    SYNTH_TEST_CHECK(tokens.size() == wrapped.size());
    SYNTH_TEST_CHECK(frontend->prepare(SYNTH_INPUT_PHONEMES_UTF8, phonemes.data(), phonemes.size(), wrapped.size() - 1,
                                       tokens) == SYNTH_ERR_INPUT_TOO_LONG);
    SYNTH_TEST_CHECK(tokens.empty());
    SYNTH_TEST_CHECK(frontend->prepare(SYNTH_INPUT_PHONEMES_UTF8, phonemes.data(), phonemes.size(), 1, tokens) ==
                     SYNTH_ERR_INPUT_TOO_LONG);

    // No padding at all still maps every scalar exactly once.
    config              = valid_config();
    config.padding_rule = synth::SymbolPaddingRule::None;
    SYNTH_TEST_CHECK(synth::make_symbol_map_frontend(config, frontend) == SYNTH_OK);
    SYNTH_TEST_CHECK(frontend->prepare(SYNTH_INPUT_PHONEMES_UTF8, phonemes.data(), phonemes.size(), 64, tokens) ==
                     SYNTH_OK);
    const std::vector<int32_t> bare = { 6, 2, 3, 4, 5 };
    SYNTH_TEST_CHECK(tokens == bare);

    // A sparse vocabulary leaves unmapped ids empty. They are reserved
    // embedding rows, not mappable symbols, so they are skipped rather than
    // rejected, and the surviving ids keep their table positions.
    config              = valid_config();
    config.padding_rule = synth::SymbolPaddingRule::WrapPadToken;
    config.symbols      = { "", "a", "", "", "t", "", "s" };
    SYNTH_TEST_CHECK(synth::make_symbol_map_frontend(config, frontend) == SYNTH_OK);
    const std::string sparse_input = "ats";
    SYNTH_TEST_CHECK(
        frontend->prepare(SYNTH_INPUT_PHONEMES_UTF8, sparse_input.data(), sparse_input.size(), 64, tokens) == SYNTH_OK);
    const std::vector<int32_t> sparse_expected = { 0, 1, 4, 6, 0 };
    SYNTH_TEST_CHECK(tokens == sparse_expected);

    // An id that the sparse table leaves empty is still unmappable input.
    const std::string reserved = "\xCB\x88";  // U+02C8, absent from this table
    SYNTH_TEST_CHECK(frontend->prepare(SYNTH_INPUT_PHONEMES_UTF8, reserved.data(), reserved.size(), 64, tokens) ==
                     SYNTH_ERR_TEXT_FRONTEND);

    // A table with no mappable entry at all is a malformed package.
    config         = valid_config();
    config.symbols = { "", "", "" };
    SYNTH_TEST_CHECK(synth::make_symbol_map_frontend(config, frontend) == SYNTH_ERR_TEXT_FRONTEND);
    SYNTH_TEST_CHECK(frontend == nullptr);

    config              = valid_config();
    config.padding_rule = static_cast<synth::SymbolPaddingRule>(99);
    SYNTH_TEST_CHECK(synth::make_symbol_map_frontend(config, frontend) == SYNTH_ERR_TEXT_FRONTEND);
    SYNTH_TEST_CHECK(frontend == nullptr);
    return 0;
}
