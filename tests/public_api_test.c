#include "synthesize.h"

#include <stddef.h>
#include <string.h>

#define CHECK(condition)     \
    do {                     \
        if (!(condition)) {  \
            return __LINE__; \
        }                    \
    } while (0)

static synth_sink_result_t SYNTH_CALL accept_audio(void * user_data, const synth_audio_chunk_t * chunk) {
    (void) user_data;
    (void) chunk;
    return SYNTH_SINK_CONTINUE;
}

int main(void) {
    CHECK(SYNTH_FALSE == 0 && SYNTH_TRUE == 1);
    CHECK(SYNTH_INPUT_SUPPORT_TOKEN_IDS == 4);
    CHECK(SYNTH_MODEL_CAPABILITY_SPEAKING_RATE == 1);
    CHECK(SYNTH_MODEL_CAPABILITY_STOCHASTIC == 2);

    synth_diagnostic_sink_t diagnostic;
    memset(&diagnostic, 0xa5, sizeof(diagnostic));
    synth_diagnostic_sink_init(&diagnostic, sizeof(diagnostic));
    CHECK(diagnostic.struct_size == sizeof(diagnostic));
    CHECK(diagnostic.emit == NULL && diagnostic.user_data == NULL);

    synth_model_load_params_t load;
    memset(&load, 0xa5, sizeof(load));
    synth_model_load_params_init(&load, sizeof(load));
    CHECK(load.struct_size == sizeof(load));
    CHECK(load.backend == SYNTH_BACKEND_AUTO && load.device_index == -1 && load.diagnostics == NULL);

    synth_model_capabilities_t capabilities;
    memset(&capabilities, 0xa5, sizeof(capabilities));
    synth_model_capabilities_init(&capabilities, sizeof(capabilities));
    CHECK(capabilities.struct_size == sizeof(capabilities));
    CHECK(capabilities.input_flags == 0 && capabilities.output_sample_rate == 0);

    synth_preset_voice_t voice;
    synth_preset_voice_init(&voice, sizeof(voice));
    CHECK(voice.struct_size == sizeof(voice) && voice.id == NULL && voice.flags == 0);

    synth_language_capability_t language;
    synth_language_capability_init(&language, sizeof(language));
    CHECK(language.struct_size == sizeof(language) && language.tag == NULL && language.flags == 0);

    synth_request_t request;
    memset(&request, 0xa5, sizeof(request));
    synth_request_init(&request, sizeof(request));
    CHECK(request.struct_size == sizeof(request));
    CHECK(request.input_kind == SYNTH_INPUT_TEXT_UTF8 && request.input_data == NULL && request.input_count == 0);
    CHECK(request.seed == 0 && request.speaking_rate == 1.0f && request.max_output_frames == 0);
    CHECK(request.should_cancel == NULL && request.diagnostics == NULL && request.voice_profile == NULL);

    synth_audio_sink_t sink;
    synth_audio_sink_init(&sink, sizeof(sink));
    CHECK(sink.struct_size == sizeof(sink) && sink.write == NULL && sink.user_data == NULL);
    sink.write = accept_audio;

    synth_result_t result;
    memset(&result, 0xa5, sizeof(result));
    synth_result_init(&result, sizeof(result));
    CHECK(result.struct_size == sizeof(result));
    CHECK(result.frames_emitted == 0 && result.actual_seed == 0 && result.flags == 0);
    CHECK(result.resolved_language_tag == NULL && result.resolved_voice_id == NULL);

    synth_model_t * model = (synth_model_t *) (uintptr_t) 1;
    CHECK(synth_model_load(NULL, NULL, &model) == SYNTH_ERR_INVALID_ARG && model == NULL);
    CHECK(synth_model_load("", NULL, &model) == SYNTH_ERR_INVALID_ARG && model == NULL);
    CHECK(synth_model_load("does-not-exist.gguf", NULL, &model) == SYNTH_ERR_FILE_NOT_FOUND && model == NULL);
    CHECK(synth_model_load("does-not-exist.gguf", NULL, NULL) == SYNTH_ERR_INVALID_ARG);

    synth_context_t * context = (synth_context_t *) (uintptr_t) 1;
    CHECK(synth_context_create(NULL, &context) == SYNTH_ERR_INVALID_ARG && context == NULL);
    CHECK(synth_context_create(NULL, NULL) == SYNTH_ERR_INVALID_ARG);
    CHECK(synth_model_get_capabilities(NULL, &capabilities) == SYNTH_ERR_INVALID_ARG);
    CHECK(synth_model_get_preset_voice_count(NULL, NULL) == SYNTH_ERR_INVALID_ARG);
    CHECK(synth_model_get_language_count(NULL, NULL) == SYNTH_ERR_INVALID_ARG);
    CHECK(synth_synthesize(NULL, &request, &sink, &result) == SYNTH_ERR_INVALID_ARG);

    synth_audio_buffer_t * audio = (synth_audio_buffer_t *) (uintptr_t) 1;
    CHECK(synth_synthesize_to_buffer(NULL, &request, &audio, &result) == SYNTH_ERR_INVALID_ARG && audio == NULL);
    CHECK(synth_synthesize_to_buffer(NULL, &request, NULL, &result) == SYNTH_ERR_INVALID_ARG);

    synth_audio_buffer_free(NULL);
    synth_context_free(NULL);
    synth_model_free(NULL);
    return 0;
}
