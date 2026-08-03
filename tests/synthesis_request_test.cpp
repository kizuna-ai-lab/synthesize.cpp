#include "synthesis-request.h"
#include "test-assert.h"

#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

synth::ModelInfo model_info() {
    synth::ModelInfo info;
    info.input_flags          = SYNTH_INPUT_SUPPORT_TOKEN_IDS;
    info.capability_flags     = SYNTH_MODEL_CAPABILITY_SPEAKING_RATE | SYNTH_MODEL_CAPABILITY_STOCHASTIC;
    info.output_sample_rate   = 22050;
    info.output_channel_count = 1;
    info.vocab_size           = 5;
    info.max_input_tokens     = 4;
    info.max_output_frames    = 100;
    info.min_speaking_rate    = 0.8f;
    info.max_speaking_rate    = 1.25f;
    // An English-only model, declared rather than assumed. The validator used to
    // hardcode this test; now the model has to say it.
    info.languages            = {
        { "en", SYNTH_LANGUAGE_DEFAULT | SYNTH_LANGUAGE_REGIONAL_FALLBACK }
    };
    return info;
}

// A model that declares several languages, which is what Qwen3-TTS is. `ko` is
// deliberately absent so the suite can tell "declared" from "well-formed".
synth::ModelInfo multilingual_model_info() {
    synth::ModelInfo info = model_info();
    info.languages        = {
        { "en", SYNTH_LANGUAGE_REGIONAL_FALLBACK },
        { "zh", SYNTH_LANGUAGE_REGIONAL_FALLBACK },
        { "ja", 0                                }
    };
    return info;
}

synth::ModelInfo multispeaker_model_info() {
    synth::ModelInfo info = model_info();
    for (int index = 0; index < 109; ++index) {
        char id[12];
        std::snprintf(id, sizeof(id), "speaker-%03d", index);
        info.preset_voice_ids.emplace_back(id);
    }
    return info;
}

synth::ModelInfo frontend_model_info() {
    synth::ModelInfo info = model_info();
    info.input_flags      = SYNTH_INPUT_SUPPORT_PHONEMES_UTF8 | SYNTH_INPUT_SUPPORT_TOKEN_IDS;
    synth::SymbolMapFrontendConfig config;
    config.provider_id      = "synthesize.symbol_map";
    config.contract_version = 1;
    config.mapping_mode     = synth::SymbolMappingMode::UnicodeScalar;
    config.symbols          = { "_", "a", "ˈ", "ɪ", "t", "s", "a" };
    config.blank_id         = 0;
    config.padding_rule     = synth::SymbolPaddingRule::InterleavedBlank;
    std::unique_ptr<synth::TextFrontend> frontend;
    if (synth::make_symbol_map_frontend(config, frontend) != SYNTH_OK) {
        return {};
    }
    info.text_frontend    = std::shared_ptr<const synth::TextFrontend>(std::move(frontend));
    info.vocab_size       = config.symbols.size();
    info.max_input_tokens = 16;
    return info;
}

synth_request_t valid_request(const std::vector<int32_t> & tokens) {
    synth_request_t request;
    synth_request_init(&request, sizeof(request));
    request.input_kind  = SYNTH_INPUT_TOKEN_IDS;
    request.input_data  = tokens.data();
    request.input_count = tokens.size();
    request.seed        = 42;
    return request;
}

}  // namespace

int main() {
    const synth::ModelInfo          info    = model_info();
    const std::vector<int32_t>      tokens  = { 1, 2, 3 };
    synth_request_t                 request = valid_request(tokens);
    synth::PreparedSynthesisRequest prepared;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(info, &request, prepared) == SYNTH_OK);
    SYNTH_TEST_CHECK(prepared.token_ids == tokens && prepared.seed == 42 && prepared.speaking_rate == 1.0f);
    SYNTH_TEST_CHECK(prepared.effective_frame_limit == 100);
    SYNTH_TEST_CHECK(prepared.resolved_language_size == 2);

    request.max_output_frames = 80;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(info, &request, prepared) == SYNTH_OK);
    SYNTH_TEST_CHECK(prepared.effective_frame_limit == 80);
    request.max_output_frames = 0;

    request.language_tag      = "EN";
    request.language_tag_size = 2;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(info, &request, prepared) == SYNTH_OK);
    request.language_tag      = "en-US";
    request.language_tag_size = 5;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(info, &request, prepared) == SYNTH_OK);
    // PR #6 triage FIX 5: SYNTH_LANGUAGE_REGIONAL_FALLBACK permits ONLY a
    // primary language plus ONE region subtag (two ASCII letters or three
    // ASCII digits) -- not a script, variant, extension, or private-use
    // suffix (docs/c-interface.md:341). Before this fix, this call site
    // (like its two now-hoisted siblings) accepted ANY suffix after the
    // first '-' as long as the primary matched: "en-Latn" (a script
    // subtag, still a well-formed BCP-47 tag on its own) and "en-USA"
    // (three letters, not a real ISO 3166-1 region) both used to fall back
    // to "en" here.
    request.language_tag      = "en-Latn";
    request.language_tag_size = 7;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(info, &request, prepared) == SYNTH_ERR_UNSUPPORTED_LANGUAGE);
    request.language_tag      = "en-USA";
    request.language_tag_size = 6;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(info, &request, prepared) == SYNTH_ERR_UNSUPPORTED_LANGUAGE);
    // A genuine three-ASCII-digit UN M.49 region subtag is still accepted --
    // the fix narrows the suffix SHAPE, not its length class.
    request.language_tag      = "en-419";
    request.language_tag_size = 6;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(info, &request, prepared) == SYNTH_OK);
    request.language_tag      = "zh-CN";
    request.language_tag_size = 5;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(info, &request, prepared) == SYNTH_ERR_UNSUPPORTED_LANGUAGE);
    request.language_tag      = "en_US";
    request.language_tag_size = 5;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(info, &request, prepared) == SYNTH_ERR_INVALID_ARG);

    // A multilingual model answers for what it declared, and only that. Before
    // the validator read the model's own list, every one of these but "en" was
    // refused no matter what the package said.
    const synth::ModelInfo many = multilingual_model_info();
    request.language_tag        = "zh";
    request.language_tag_size   = 2;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(many, &request, prepared) == SYNTH_OK);
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(info, &request, prepared) == SYNTH_ERR_UNSUPPORTED_LANGUAGE);
    request.language_tag      = "ZH-cn";
    request.language_tag_size = 5;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(many, &request, prepared) == SYNTH_OK);
    // Declared without regional fallback: the bare tag is served, a region is not.
    request.language_tag      = "ja";
    request.language_tag_size = 2;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(many, &request, prepared) == SYNTH_OK);
    request.language_tag      = "ja-JP";
    request.language_tag_size = 5;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(many, &request, prepared) == SYNTH_ERR_UNSUPPORTED_LANGUAGE);
    // Well-formed and undeclared is a language error, not an argument error.
    request.language_tag      = "ko";
    request.language_tag_size = 2;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(many, &request, prepared) == SYNTH_ERR_UNSUPPORTED_LANGUAGE);

    // The resolved language is the capability that answered, not a constant. This
    // was hardcoded "en" for every request, so a model that accepted zh reported
    // English back -- docs/languages.md requires the resolved language and any
    // fallback to be reported to the caller.
    request.language_tag      = "zh";
    request.language_tag_size = 2;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(many, &request, prepared) == SYNTH_OK);
    SYNTH_TEST_CHECK(prepared.resolved_language_size == 2);
    SYNTH_TEST_CHECK(std::strncmp(prepared.resolved_language_tag, "zh", 2) == 0);
    // A regional request reports the tag it fell back to, which is the fallback
    // being reported rather than hidden.
    request.language_tag      = "en-GB";
    request.language_tag_size = 5;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(many, &request, prepared) == SYNTH_OK);
    SYNTH_TEST_CHECK(prepared.resolved_language_size == 2);
    SYNTH_TEST_CHECK(std::strncmp(prepared.resolved_language_tag, "en", 2) == 0);
    // No tag: the package's declared default, and nothing when it declares none.
    request.language_tag      = nullptr;
    request.language_tag_size = 0;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(info, &request, prepared) == SYNTH_OK);
    SYNTH_TEST_CHECK(prepared.resolved_language_size == 2);
    SYNTH_TEST_CHECK(std::strncmp(prepared.resolved_language_tag, "en", 2) == 0);
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(many, &request, prepared) == SYNTH_OK);
    SYNTH_TEST_CHECK(prepared.resolved_language_size == 0 && prepared.resolved_language_tag == nullptr);
    // Restore an explicit, undeclared tag for the case below.
    request.language_tag      = "ko";
    request.language_tag_size = 2;

    // A model declaring nothing accepts no explicit tag, but naming no language
    // at all still works: that path never consults the list.
    synth::ModelInfo silent = model_info();
    silent.languages.clear();
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(silent, &request, prepared) == SYNTH_ERR_UNSUPPORTED_LANGUAGE);

    request.language_tag      = nullptr;
    request.language_tag_size = 0;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(silent, &request, prepared) == SYNTH_OK);

    request.input_kind = SYNTH_INPUT_TEXT_UTF8;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(info, &request, prepared) == SYNTH_ERR_UNSUPPORTED_INPUT);
    request.input_kind = SYNTH_INPUT_TOKEN_IDS;
    request.input_data = nullptr;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(info, &request, prepared) == SYNTH_ERR_INVALID_ARG);
    request.input_data  = tokens.data();
    request.input_count = 0;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(info, &request, prepared) == SYNTH_ERR_INVALID_ARG);
    request.input_count = 5;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(info, &request, prepared) == SYNTH_ERR_INPUT_TOO_LONG);
    request.input_count = tokens.size();

    std::vector<int32_t> bad_tokens = tokens;
    bad_tokens[1]                   = 5;
    request.input_data              = bad_tokens.data();
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(info, &request, prepared) == SYNTH_ERR_INVALID_ARG);
    request.input_data = tokens.data();

    request.speaking_rate = 0.7f;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(info, &request, prepared) == SYNTH_ERR_INVALID_ARG);
    request.speaking_rate = std::numeric_limits<float>::quiet_NaN();
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(info, &request, prepared) == SYNTH_ERR_INVALID_ARG);
    request.speaking_rate = 1.0f;

    request.voice_id      = "voice";
    request.voice_id_size = 5;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(info, &request, prepared) == SYNTH_ERR_UNSUPPORTED_VOICE);
    request.voice_id      = nullptr;
    request.voice_id_size = 1;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(info, &request, prepared) == SYNTH_ERR_INVALID_ARG);
    request.voice_id_size = 0;

    const synth::ModelInfo multispeaker = multispeaker_model_info();
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(multispeaker, &request, prepared) == SYNTH_ERR_UNSUPPORTED_VOICE);
    request.voice_id      = "speaker-004";
    request.voice_id_size = 11;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(multispeaker, &request, prepared) == SYNTH_OK);
    SYNTH_TEST_CHECK(prepared.speaker_index == 4 && prepared.resolved_voice_size == 11);
    SYNTH_TEST_CHECK(std::memcmp(prepared.resolved_voice_id, "speaker-004", 11) == 0);
    request.voice_id      = "speaker-109";
    request.voice_id_size = 11;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(multispeaker, &request, prepared) == SYNTH_ERR_UNSUPPORTED_VOICE);
    request.voice_id      = nullptr;
    request.voice_id_size = 0;

    const synth::ModelInfo frontend_info = frontend_model_info();
    const std::string      phonemes      = "aˈɪts";
    request.input_kind                   = SYNTH_INPUT_PHONEMES_UTF8;
    request.input_data                   = phonemes.data();
    request.input_count                  = phonemes.size();
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(frontend_info, &request, prepared) == SYNTH_OK);
    SYNTH_TEST_CHECK(prepared.token_ids == std::vector<int32_t>({ 0, 6, 0, 2, 0, 3, 0, 4, 0, 5, 0 }));

    synth::ModelInfo short_frontend_info = frontend_info;
    short_frontend_info.max_input_tokens = 10;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(short_frontend_info, &request, prepared) ==
                     SYNTH_ERR_INPUT_TOO_LONG);
    const char invalid_utf8[] = { static_cast<char>(0xc0), static_cast<char>(0xaf) };
    request.input_data        = invalid_utf8;
    request.input_count       = sizeof(invalid_utf8);
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(frontend_info, &request, prepared) == SYNTH_ERR_TEXT_FRONTEND);
    request.input_kind  = SYNTH_INPUT_TEXT_UTF8;
    request.input_data  = phonemes.data();
    request.input_count = phonemes.size();
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(frontend_info, &request, prepared) ==
                     SYNTH_ERR_UNSUPPORTED_INPUT);

    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(info, nullptr, prepared) == SYNTH_ERR_INVALID_ARG);
    request.struct_size = sizeof(uint64_t) - 1;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(info, &request, prepared) == SYNTH_ERR_BAD_STRUCT_SIZE);
    return 0;
}
