#include "synthesize.h"
#include "test-assert.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

namespace {

std::vector<int32_t> read_tokens(const char * path) {
    std::ifstream input(path, std::ios::binary);
    input.seekg(0, std::ios::end);
    const std::streamoff bytes = input.tellg();
    input.seekg(0, std::ios::beg);
    if (!input || bytes <= 0 || bytes % static_cast<std::streamoff>(sizeof(int32_t)) != 0) {
        return {};
    }
    std::vector<int32_t> tokens(static_cast<size_t>(bytes) / sizeof(int32_t));
    input.read(reinterpret_cast<char *>(tokens.data()), bytes);
    return input ? tokens : std::vector<int32_t>{};
}

struct AudioCapture {
    int                calls = 0;
    bool               valid = true;
    std::vector<float> samples;
};

synth_sink_result_t SYNTH_CALL capture_audio(void * user_data, const synth_audio_chunk_t * chunk) {
    auto * capture = static_cast<AudioCapture *>(user_data);
    ++capture->calls;
    if (chunk == nullptr || chunk->struct_size != sizeof(*chunk) ||
        chunk->frame_offset != capture->samples.size() || chunk->sample_rate != 22050 ||
        chunk->channel_count != 1) {
        capture->valid = false;
        return SYNTH_SINK_ERROR;
    }
    capture->samples.insert(capture->samples.end(), chunk->samples, chunk->samples + chunk->frame_count);
    return SYNTH_SINK_CONTINUE;
}

synth_bool_t SYNTH_CALL cancel_now(void *) {
    return SYNTH_TRUE;
}

bool all_finite(const float * samples, uint64_t count) {
    for (uint64_t index = 0; index < count; ++index) {
        if (!std::isfinite(samples[index])) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    SYNTH_TEST_CHECK(argc == 3 || argc == 4);
    const std::vector<int32_t> tokens = read_tokens(argv[2]);
    SYNTH_TEST_CHECK(!tokens.empty());

    synth_model_load_params_t params;
    synth_model_load_params_init(&params, sizeof(params));
    params.backend = argc == 4 && std::strcmp(argv[3], "cuda") == 0 ? SYNTH_BACKEND_CUDA : SYNTH_BACKEND_CPU;
    synth_model_t * model = nullptr;
    SYNTH_TEST_CHECK(synth_model_load(argv[1], &params, &model) == SYNTH_OK && model != nullptr);
    synth_model_capabilities_t capabilities;
    synth_model_capabilities_init(&capabilities, sizeof(capabilities));
    SYNTH_TEST_CHECK(synth_model_get_capabilities(model, &capabilities) == SYNTH_OK);
    SYNTH_TEST_CHECK(
        capabilities.input_flags == (SYNTH_INPUT_SUPPORT_PHONEMES_UTF8 | SYNTH_INPUT_SUPPORT_TOKEN_IDS));
    synth_context_t * context = nullptr;
    SYNTH_TEST_CHECK(synth_context_create(model, &context) == SYNTH_OK && context != nullptr);

    synth_request_t request;
    synth_request_init(&request, sizeof(request));
    request.input_kind  = SYNTH_INPUT_TOKEN_IDS;
    request.input_data  = tokens.data();
    request.input_count = tokens.size();
    request.seed        = 42;

    synth_result_t first_result;
    synth_result_init(&first_result, sizeof(first_result));
    synth_audio_buffer_t * first = nullptr;
    SYNTH_TEST_CHECK(synth_synthesize_to_buffer(context, &request, &first, &first_result) == SYNTH_OK);
    SYNTH_TEST_CHECK(first != nullptr && first->struct_size == sizeof(*first));
    SYNTH_TEST_CHECK(first->samples != nullptr && first->frame_count > 0);
    SYNTH_TEST_CHECK(first->sample_rate == 22050 && first->channel_count == 1);
    SYNTH_TEST_CHECK(first_result.frames_emitted == first->frame_count);
    SYNTH_TEST_CHECK(first_result.actual_seed == 42 && first_result.flags == SYNTH_RESULT_SEED_USED);
    SYNTH_TEST_CHECK(first_result.sample_rate == 22050 && first_result.channel_count == 1);
    SYNTH_TEST_CHECK(first_result.resolved_language_tag_size == 2);
    SYNTH_TEST_CHECK(std::memcmp(first_result.resolved_language_tag, "en", 2) == 0);
    SYNTH_TEST_CHECK(first_result.resolved_voice_id == nullptr && first_result.resolved_voice_id_size == 0);
    SYNTH_TEST_CHECK(all_finite(first->samples, first->frame_count));

    const char phonemes[] = "ˈeɪ.";
    request.input_kind    = SYNTH_INPUT_PHONEMES_UTF8;
    request.input_data    = phonemes;
    request.input_count   = sizeof(phonemes) - 1;
    synth_audio_buffer_t * from_phonemes = nullptr;
    SYNTH_TEST_CHECK(synth_synthesize_to_buffer(context, &request, &from_phonemes, nullptr) == SYNTH_OK);
    SYNTH_TEST_CHECK(from_phonemes != nullptr && from_phonemes->frame_count == first->frame_count);
    SYNTH_TEST_CHECK(std::memcmp(from_phonemes->samples, first->samples, first->frame_count * sizeof(float)) == 0);

    request.input_kind  = SYNTH_INPUT_TOKEN_IDS;
    request.input_data  = tokens.data();
    request.input_count = tokens.size();
    synth_result_t repeat_result;
    synth_result_init(&repeat_result, sizeof(repeat_result));
    synth_audio_buffer_t * repeat = nullptr;
    SYNTH_TEST_CHECK(synth_synthesize_to_buffer(context, &request, &repeat, &repeat_result) == SYNTH_OK);
    SYNTH_TEST_CHECK(repeat != nullptr && repeat->frame_count == first->frame_count);
    SYNTH_TEST_CHECK(std::memcmp(repeat->samples, first->samples, first->frame_count * sizeof(float)) == 0);

    request.seed = 43;
    AudioCapture different;
    synth_audio_sink_t sink;
    synth_audio_sink_init(&sink, sizeof(sink));
    sink.write     = capture_audio;
    sink.user_data = &different;
    synth_result_t different_result;
    synth_result_init(&different_result, sizeof(different_result));
    SYNTH_TEST_CHECK(synth_synthesize(context, &request, &sink, &different_result) == SYNTH_OK);
    SYNTH_TEST_CHECK(different.valid && different.calls == 1 && !different.samples.empty());
    SYNTH_TEST_CHECK(different_result.actual_seed == 43);
    const bool different_shape = different.samples.size() != first->frame_count;
    const bool different_pcm = !different_shape &&
                               std::memcmp(different.samples.data(), first->samples,
                                           first->frame_count * sizeof(float)) != 0;
    SYNTH_TEST_CHECK(different_shape || different_pcm);

    request.seed              = 42;
    request.max_output_frames = 1;
    AudioCapture limited;
    sink.user_data = &limited;
    synth_result_t limited_result;
    synth_result_init(&limited_result, sizeof(limited_result));
    SYNTH_TEST_CHECK(synth_synthesize(context, &request, &sink, &limited_result) == SYNTH_ERR_OUTPUT_LIMIT);
    SYNTH_TEST_CHECK(limited.calls == 0 && limited_result.frames_emitted == 0);

    request.max_output_frames = 0;
    request.should_cancel     = cancel_now;
    synth_audio_buffer_t * cancelled = reinterpret_cast<synth_audio_buffer_t *>(uintptr_t(1));
    synth_result_t cancelled_result;
    synth_result_init(&cancelled_result, sizeof(cancelled_result));
    SYNTH_TEST_CHECK(synth_synthesize_to_buffer(context, &request, &cancelled, &cancelled_result) ==
                     SYNTH_ERR_CANCELLED);
    SYNTH_TEST_CHECK(cancelled == nullptr && cancelled_result.frames_emitted == 0);

    request.should_cancel = nullptr;
    request.input_kind    = SYNTH_INPUT_TEXT_UTF8;
    const char text[]     = "test";
    request.input_data    = text;
    request.input_count   = sizeof(text) - 1;
    synth_result_t invalid_result;
    synth_result_init(&invalid_result, sizeof(invalid_result));
    SYNTH_TEST_CHECK(synth_synthesize(context, &request, &sink, &invalid_result) == SYNTH_ERR_UNSUPPORTED_INPUT);
    SYNTH_TEST_CHECK(invalid_result.frames_emitted == 0 && invalid_result.flags == 0);

    synth_audio_buffer_free(repeat);
    synth_audio_buffer_free(from_phonemes);
    synth_audio_buffer_free(first);
    synth_context_free(context);
    synth_model_free(model);
    return 0;
}
