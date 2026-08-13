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
//
// Plan 3's Task 11 added the other half: removing the wrong code left NO code,
// so a limit stop reached the caller as a bare status. It now carries
// `synthesis.output_limit` and one of two messages, and this file pins both --
// the preflight refusal and the stop during generation. The third message, the
// transcript-assisted one, needs a Voice Profile and lives in
// tests/qwen3_tts_base_load_real.cpp.

#include "synthesize.h"
#include "test-assert.h"

#include <cstring>
#include <string>

namespace {

constexpr uint64_t kSeed = 1234;

struct Sink {
    int calls = 0;
};

// What the sink is told, so "a limit stop is not a graph failure" can be
// asserted as the presence of the RIGHT diagnostic rather than as the absence
// of any. Plan 3's Task 11 review found the absence: separating the limit stop
// from the graph-failure branch removed the wrong code and left a caller with
// a bare status, which for a transcript-assisted request arrives after minutes
// of CPU and no audio.
struct SeenDiagnostic {
    bool           seen   = false;
    synth_status_t status = SYNTH_OK;
    std::string    code;
    std::string    message;
};

void SYNTH_CALL record_diagnostic(void * user_data, const synth_diagnostic_t * diagnostic) {
    auto * target  = static_cast<SeenDiagnostic *>(user_data);
    target->seen   = true;
    target->status = diagnostic->status;
    target->code.assign(diagnostic->code != nullptr ? diagnostic->code : "", size_t(diagnostic->code_size));
    target->message.assign(diagnostic->message != nullptr ? diagnostic->message : "", size_t(diagnostic->message_size));
}

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

    const char *            text = "Hi.";
    SeenDiagnostic          seen;
    synth_diagnostic_sink_t diagnostics;
    synth_diagnostic_sink_init(&diagnostics, sizeof(diagnostics));
    diagnostics.emit      = record_diagnostic;
    diagnostics.user_data = &seen;

    auto build = [&](uint64_t limit) {
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
        request.diagnostics       = &diagnostics;
        return request;
    };

    // A limit of exactly one codec hop: the loop is allowed its one frame, does
    // not reach a stop code inside it, and stops on the limit having delivered
    // nothing.
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

        // And the caller is told what happened, by its own name. `graph_failed`
        // was wrong here and was removed; nothing replaced it until Plan 3's
        // Task 11, so a limit stop reported no code at all.
        //
        // 1920 PCM frames is exactly ONE codec hop, so this takes the
        // during-generation route rather than the preflight one: the loop runs
        // its one permitted frame, does not reach a stop code, and stops on
        // the limit. The preflight route is the case below.
        SYNTH_TEST_CHECK(seen.seen);
        SYNTH_TEST_CHECK(seen.status == SYNTH_ERR_OUTPUT_LIMIT);
        SYNTH_TEST_CHECK(seen.code == "synthesis.output_limit");
        SYNTH_TEST_CHECK(seen.message.find("before the model produced its stop condition") != std::string::npos);
        // Not the transcript-assisted variant of the message: this request
        // carries a Preset Voice and no reference at all, so naming a
        // reference transcript here would be misdirection.
        // tests/qwen3_tts_base_load_real.cpp pins the ICL branch.
        SYNTH_TEST_CHECK(seen.message.find("reference transcript") == std::string::npos);
    }

    // The preflight route: a limit smaller than one codec hop cannot be met by
    // emitting anything, so it is refused before the graph runs and carries a
    // different message. Two routes reach report_output_limit for this family
    // and a caller can act on only one of them by shrinking its text.
    {
        seen                           = SeenDiagnostic{};
        synth_request_t        request = build(1919);
        synth_audio_buffer_t * audio   = nullptr;
        synth_result_t         result;
        synth_result_init(&result, sizeof(result));
        SYNTH_TEST_CHECK(synth_synthesize_to_buffer(context, &request, &audio, &result) == SYNTH_ERR_OUTPUT_LIMIT);
        SYNTH_TEST_CHECK(audio == nullptr);
        SYNTH_TEST_CHECK(seen.seen);
        SYNTH_TEST_CHECK(seen.status == SYNTH_ERR_OUTPUT_LIMIT);
        SYNTH_TEST_CHECK(seen.code == "synthesis.output_limit");
        SYNTH_TEST_CHECK(seen.message.find("smaller than one") != std::string::npos);
    }

    // Nothing below reaches the limit, so nothing below may report one. Reset
    // rather than trust: the sink is shared, and a stale hit would make the
    // two success cases look like limit stops to a later reader.
    seen = SeenDiagnostic{};

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
    // Neither success case emitted anything. Without this the assertion above
    // would only prove that SOME call diagnosed, not that the limit stop did.
    SYNTH_TEST_CHECK(!seen.seen);

    synth_context_free(context);
    synth_model_free(model);
    return 0;
}
