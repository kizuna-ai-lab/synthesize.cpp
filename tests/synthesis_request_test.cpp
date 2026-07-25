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
    request.language_tag      = "zh-CN";
    request.language_tag_size = 5;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(info, &request, prepared) == SYNTH_ERR_UNSUPPORTED_LANGUAGE);
    request.language_tag      = "en_US";
    request.language_tag_size = 5;
    SYNTH_TEST_CHECK(synth::prepare_synthesis_request(info, &request, prepared) == SYNTH_ERR_INVALID_ARG);
    request.language_tag      = nullptr;
    request.language_tag_size = 0;

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
