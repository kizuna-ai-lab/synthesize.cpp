#include "synthesize.h"
#include "test-assert.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

std::vector<int32_t> read_tokens(const char * path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        return {};
    }
    const std::streamoff bytes = input.tellg();
    if (bytes <= 0 || bytes % static_cast<std::streamoff>(sizeof(int32_t)) != 0) {
        return {};
    }
    std::vector<int32_t> tokens(static_cast<size_t>(bytes) / sizeof(int32_t));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(tokens.data()), bytes);
    return input ? tokens : std::vector<int32_t>{};
}

bool check_voice(const synth_model_t * model, uint64_t index, const char * expected) {
    synth_preset_voice_t voice;
    synth_preset_voice_init(&voice, sizeof(voice));
    return synth_model_get_preset_voice(model, index, &voice) == SYNTH_OK && voice.id != nullptr &&
           voice.id_size == std::strlen(expected) && std::memcmp(voice.id, expected, voice.id_size) == 0 &&
           voice.flags == 0;
}

}  // namespace

int main(int argc, char ** argv) {
    SYNTH_TEST_CHECK(argc == 3 || argc == 4);
    const std::vector<int32_t> tokens = read_tokens(argv[2]);
    SYNTH_TEST_CHECK(!tokens.empty());

    synth_model_load_params_t load;
    synth_model_load_params_init(&load, sizeof(load));
    load.backend          = argc == 4 && std::strcmp(argv[3], "cuda") == 0 ? SYNTH_BACKEND_CUDA : SYNTH_BACKEND_CPU;
    synth_model_t * model = nullptr;
    SYNTH_TEST_CHECK(synth_model_load(argv[1], &load, &model) == SYNTH_OK && model != nullptr);
    synth_model_capabilities_t capabilities;
    synth_model_capabilities_init(&capabilities, sizeof(capabilities));
    SYNTH_TEST_CHECK(synth_model_get_capabilities(model, &capabilities) == SYNTH_OK);
    SYNTH_TEST_CHECK(
        capabilities.input_flags == (SYNTH_INPUT_SUPPORT_PHONEMES_UTF8 | SYNTH_INPUT_SUPPORT_TOKEN_IDS));

    uint64_t voice_count = 0;
    SYNTH_TEST_CHECK(synth_model_get_preset_voice_count(model, &voice_count) == SYNTH_OK && voice_count == 109);
    SYNTH_TEST_CHECK(check_voice(model, 0, "speaker-000"));
    SYNTH_TEST_CHECK(check_voice(model, 54, "speaker-054"));
    SYNTH_TEST_CHECK(check_voice(model, 108, "speaker-108"));
    synth_preset_voice_t voice;
    synth_preset_voice_init(&voice, sizeof(voice));
    SYNTH_TEST_CHECK(synth_model_get_preset_voice(model, 109, &voice) == SYNTH_ERR_INVALID_ARG);

    synth_context_t * context = nullptr;
    SYNTH_TEST_CHECK(synth_context_create(model, &context) == SYNTH_OK && context != nullptr);
    synth_request_t request;
    synth_request_init(&request, sizeof(request));
    request.input_kind  = SYNTH_INPUT_TOKEN_IDS;
    request.input_data  = tokens.data();
    request.input_count = tokens.size();
    request.seed        = 42;

    synth_audio_buffer_t * audio = reinterpret_cast<synth_audio_buffer_t *>(uintptr_t(1));
    SYNTH_TEST_CHECK(synth_synthesize_to_buffer(context, &request, &audio, nullptr) == SYNTH_ERR_UNSUPPORTED_VOICE &&
                     audio == nullptr);
    request.voice_id      = "speaker-109";
    request.voice_id_size = 11;
    audio                 = reinterpret_cast<synth_audio_buffer_t *>(uintptr_t(1));
    SYNTH_TEST_CHECK(synth_synthesize_to_buffer(context, &request, &audio, nullptr) == SYNTH_ERR_UNSUPPORTED_VOICE &&
                     audio == nullptr);

    request.voice_id = "speaker-004";
    synth_result_t first_result;
    synth_result_init(&first_result, sizeof(first_result));
    SYNTH_TEST_CHECK(synth_synthesize_to_buffer(context, &request, &audio, &first_result) == SYNTH_OK);
    SYNTH_TEST_CHECK(audio != nullptr && audio->frame_count > 0 && first_result.frames_emitted == audio->frame_count);
    SYNTH_TEST_CHECK(first_result.resolved_voice_id_size == 11 &&
                     std::memcmp(first_result.resolved_voice_id, "speaker-004", 11) == 0);

    const char phonemes[] = "ˈeɪ.";
    request.input_kind    = SYNTH_INPUT_PHONEMES_UTF8;
    request.input_data    = phonemes;
    request.input_count   = sizeof(phonemes) - 1;
    synth_audio_buffer_t * from_phonemes = nullptr;
    SYNTH_TEST_CHECK(synth_synthesize_to_buffer(context, &request, &from_phonemes, nullptr) == SYNTH_OK);
    SYNTH_TEST_CHECK(from_phonemes != nullptr && from_phonemes->frame_count == audio->frame_count);
    SYNTH_TEST_CHECK(std::memcmp(from_phonemes->samples, audio->samples, audio->frame_count * sizeof(float)) == 0);

    request.input_kind  = SYNTH_INPUT_TOKEN_IDS;
    request.input_data  = tokens.data();
    request.input_count = tokens.size();
    synth_audio_buffer_t * repeat = nullptr;
    synth_result_t         repeat_result;
    synth_result_init(&repeat_result, sizeof(repeat_result));
    SYNTH_TEST_CHECK(synth_synthesize_to_buffer(context, &request, &repeat, &repeat_result) == SYNTH_OK);
    SYNTH_TEST_CHECK(repeat != nullptr && repeat->frame_count == audio->frame_count);
    SYNTH_TEST_CHECK(std::memcmp(repeat->samples, audio->samples, audio->frame_count * sizeof(float)) == 0);

    request.voice_id                 = "speaker-054";
    synth_audio_buffer_t * different = nullptr;
    SYNTH_TEST_CHECK(synth_synthesize_to_buffer(context, &request, &different, nullptr) == SYNTH_OK);
    SYNTH_TEST_CHECK(different != nullptr);
    const bool different_shape = different->frame_count != audio->frame_count;
    const bool different_pcm =
        !different_shape && std::memcmp(different->samples, audio->samples, audio->frame_count * sizeof(float)) != 0;
    SYNTH_TEST_CHECK(different_shape || different_pcm);

    synth_audio_buffer_free(different);
    synth_audio_buffer_free(repeat);
    synth_audio_buffer_free(from_phonemes);
    synth_audio_buffer_free(audio);
    synth_context_free(context);
    synth_model_free(model);
    return 0;
}
