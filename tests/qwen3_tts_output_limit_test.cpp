// What a request that hits the output limit reports back.
//
// docs/c-interface.md, v1 Synthesis Result: "On success, cancellation,
// output-limit termination, or sink error, the implementation fills every result
// field it has resolved and every frame it has emitted." A limit stop is one of
// the four fill cases, not a failure that may leave the result blank.
//
// This family is the only one that carries the request's limit into its own
// generation loop, so it is the only one whose limit stop arrives at the generic
// error branch rather than at a delivering check. It took that branch, which
// zeroed every resolved field and also reported the stop as
// `synthesis.graph_failed`. An output limit is not a graph failure.
//
// The sibling families' tests did not catch the class: the VITS one asserts only
// that no chunk arrived and that `frames_emitted` is zero, which a blank result
// satisfies. So this asserts the fields that were actually wrong.

#include "synthesize.h"
#include "test-assert.h"

#include <cstring>
#include <string>

namespace {

constexpr uint64_t kSeed = 1234;

struct Sink {
    int calls = 0;
};

synth_sink_result_t SYNTH_CALL count_chunks(void * user_data, const synth_audio_chunk_t * chunk) {
    auto * sink = static_cast<Sink *>(user_data);
    // The ABI has no completion callback and the contract forbids a zero-length
    // final chunk, so every call here must carry frames.
    if (sink == nullptr || chunk == nullptr || chunk->frame_count == 0) {
        return SYNTH_SINK_ERROR;
    }
    ++sink->calls;
    return SYNTH_SINK_CONTINUE;
}

}  // namespace

int main(int argc, char ** argv) {
    SYNTH_TEST_CHECK(argc == 2);

    synth_model_load_params_t load_params;
    synth_model_load_params_init(&load_params, sizeof(load_params));
    load_params.backend = SYNTH_BACKEND_CPU;

    synth_model_t * model = nullptr;
    SYNTH_TEST_CHECK(synth_model_load(argv[1], &load_params, &model) == SYNTH_OK && model != nullptr);
    synth_context_t * context = nullptr;
    SYNTH_TEST_CHECK(synth_context_create(model, &context) == SYNTH_OK && context != nullptr);

    const char * text  = "Hi.";
    auto         build = [&](uint64_t limit) {
        synth_request_t request;
        synth_request_init(&request, sizeof(request));
        request.input_kind        = SYNTH_INPUT_TEXT_UTF8;
        request.input_data        = text;
        request.input_count       = std::strlen(text);
        request.voice_id          = "aiden";
        request.voice_id_size     = 5;
        request.language_tag      = "en";
        request.language_tag_size = 2;
        request.seed              = kSeed;
        request.max_output_frames = limit;
        return request;
    };

    // One PCM frame is far below one codec frame of 1920, so this cannot be met.
    {
        Sink               sink_state;
        synth_audio_sink_t sink;
        synth_audio_sink_init(&sink, sizeof(sink));
        sink.write     = count_chunks;
        sink.user_data = &sink_state;

        synth_request_t request = build(1920);
        synth_result_t  result;
        synth_result_init(&result, sizeof(result));
        SYNTH_TEST_CHECK(synth_synthesize(context, &request, &sink, &result) == SYNTH_ERR_OUTPUT_LIMIT);

        // Nothing was delivered, and nothing was delivered emptily either.
        SYNTH_TEST_CHECK(sink_state.calls == 0);
        SYNTH_TEST_CHECK(result.frames_emitted == 0);

        // Everything the request resolved is reported anyway. These were all zero
        // or empty before the limit stop was separated from a graph failure.
        SYNTH_TEST_CHECK(result.actual_seed == kSeed);
        SYNTH_TEST_CHECK((result.flags & SYNTH_RESULT_SEED_USED) != 0);
        SYNTH_TEST_CHECK(result.sample_rate != 0);
        SYNTH_TEST_CHECK(result.channel_count == 1);
        SYNTH_TEST_CHECK(result.resolved_voice_id != nullptr && result.resolved_voice_id_size == 5);
        SYNTH_TEST_CHECK(std::string(result.resolved_voice_id, result.resolved_voice_id_size) == "aiden");
        SYNTH_TEST_CHECK(result.resolved_language_tag != nullptr && result.resolved_language_tag_size == 2);
        SYNTH_TEST_CHECK(std::string(result.resolved_language_tag, result.resolved_language_tag_size) == "en");
    }

    // A request naming no limit must synthesize. The package declares a ceiling of
    // fifteen million frames as a generic "no limit", and the core normalises an
    // unset request to it; this family then sized its talker cache from that and
    // asked for three terabytes, so every default request -- including every one
    // the CLI makes -- returned OOM.
    {
        synth_request_t        request = build(0);
        synth_audio_buffer_t * audio   = nullptr;
        synth_result_t         result;
        synth_result_init(&result, sizeof(result));
        SYNTH_TEST_CHECK(synth_synthesize_to_buffer(context, &request, &audio, &result) == SYNTH_OK);
        SYNTH_TEST_CHECK(audio != nullptr && audio->frame_count > 0);
        SYNTH_TEST_CHECK(result.frames_emitted == audio->frame_count);
        synth_audio_buffer_free(audio);
    }

    // A limit the request can meet is honoured rather than exceeded: the public
    // field counts PCM frames and this family counts codec frames of 1920, and
    // passing one as the other let a request for 600 emit 21120.
    {
        synth_request_t        request = build(48000);
        synth_audio_buffer_t * audio   = nullptr;
        synth_result_t         result;
        synth_result_init(&result, sizeof(result));
        SYNTH_TEST_CHECK(synth_synthesize_to_buffer(context, &request, &audio, &result) == SYNTH_OK);
        SYNTH_TEST_CHECK(audio != nullptr);
        SYNTH_TEST_CHECK(audio->frame_count > 0 && audio->frame_count <= 48000);
        synth_audio_buffer_free(audio);
    }

    synth_context_free(context);
    synth_model_free(model);
    return 0;
}
