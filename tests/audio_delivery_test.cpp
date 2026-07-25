#include "audio-delivery.h"
#include "test-assert.h"

#include <array>
#include <cstdint>

namespace {

struct Capture {
    synth_sink_result_t  return_value = SYNTH_SINK_CONTINUE;
    int                  calls        = 0;
    synth_audio_chunk_t  chunk{};
    std::array<float, 4> samples{};
};

synth_sink_result_t SYNTH_CALL capture_audio(void * user_data, const synth_audio_chunk_t * chunk) {
    auto * capture = static_cast<Capture *>(user_data);
    ++capture->calls;
    capture->chunk = *chunk;
    for (size_t index = 0; index < capture->samples.size(); ++index) {
        capture->samples[index] = chunk->samples[index];
    }
    return capture->return_value;
}

synth_sink_result_t SYNTH_CALL throw_audio(void *, const synth_audio_chunk_t *) {
    throw 1;
}

}  // namespace

int main() {
    const std::array<float, 4> pcm = { -0.5f, 0.25f, 0.75f, -1.0f };
    synth::AudioDeliveryInfo   info;
    info.actual_seed            = 42;
    info.sample_rate            = 22050;
    info.channel_count          = 1;
    info.result_flags           = SYNTH_RESULT_SEED_USED;
    info.resolved_language_tag  = "en";
    info.resolved_language_size = 2;

    Capture            capture;
    synth_audio_sink_t sink;
    synth_audio_sink_init(&sink, sizeof(sink));
    sink.write     = capture_audio;
    sink.user_data = &capture;

    synth_result_t result;
    synth_result_init(&result, sizeof(result));
    SYNTH_TEST_CHECK(synth::valid_audio_sink(&sink));
    SYNTH_TEST_CHECK(synth::deliver_complete_audio(pcm.data(), pcm.size(), info, &sink, &result) == SYNTH_OK);
    SYNTH_TEST_CHECK(capture.calls == 1);
    SYNTH_TEST_CHECK(capture.chunk.struct_size == sizeof(synth_audio_chunk_t));
    SYNTH_TEST_CHECK(capture.chunk.frame_count == pcm.size() && capture.chunk.frame_offset == 0);
    SYNTH_TEST_CHECK(capture.chunk.sample_rate == 22050 && capture.chunk.channel_count == 1);
    SYNTH_TEST_CHECK(capture.samples == pcm);
    SYNTH_TEST_CHECK(result.frames_emitted == pcm.size());
    SYNTH_TEST_CHECK(result.actual_seed == 42 && result.flags == SYNTH_RESULT_SEED_USED);
    SYNTH_TEST_CHECK(result.sample_rate == 22050 && result.channel_count == 1);
    SYNTH_TEST_CHECK(result.resolved_language_tag == info.resolved_language_tag);
    SYNTH_TEST_CHECK(result.resolved_language_tag_size == 2);
    SYNTH_TEST_CHECK(result.resolved_voice_id == nullptr && result.resolved_voice_id_size == 0);

    capture.return_value = SYNTH_SINK_CANCEL;
    capture.calls        = 0;
    synth_result_init(&result, sizeof(result));
    SYNTH_TEST_CHECK(synth::deliver_complete_audio(pcm.data(), pcm.size(), info, &sink, &result) ==
                     SYNTH_ERR_CANCELLED);
    SYNTH_TEST_CHECK(capture.calls == 1 && result.frames_emitted == pcm.size());

    capture.return_value = SYNTH_SINK_ERROR;
    synth_result_init(&result, sizeof(result));
    SYNTH_TEST_CHECK(synth::deliver_complete_audio(pcm.data(), pcm.size(), info, &sink, &result) == SYNTH_ERR_SINK);
    SYNTH_TEST_CHECK(result.frames_emitted == pcm.size());

    capture.return_value = 99;
    synth_result_init(&result, sizeof(result));
    SYNTH_TEST_CHECK(synth::deliver_complete_audio(pcm.data(), pcm.size(), info, &sink, &result) == SYNTH_ERR_SINK);

    sink.write = throw_audio;
    synth_result_init(&result, sizeof(result));
    SYNTH_TEST_CHECK(synth::deliver_complete_audio(pcm.data(), pcm.size(), info, &sink, &result) == SYNTH_ERR_SINK);
    SYNTH_TEST_CHECK(result.frames_emitted == pcm.size());
    sink.write = capture_audio;

    synth_audio_sink_t invalid_sink{};
    SYNTH_TEST_CHECK(!synth::valid_audio_sink(nullptr));
    SYNTH_TEST_CHECK(!synth::valid_audio_sink(&invalid_sink));
    synth_audio_sink_init(&invalid_sink, sizeof(invalid_sink));
    SYNTH_TEST_CHECK(!synth::valid_audio_sink(&invalid_sink));
    SYNTH_TEST_CHECK(synth::deliver_complete_audio(pcm.data(), pcm.size(), info, &invalid_sink, &result) ==
                     SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(synth::deliver_complete_audio(nullptr, pcm.size(), info, &sink, &result) == SYNTH_ERR_INVALID_ARG);

    capture.calls = 0;
    synth_result_init(&result, sizeof(result));
    SYNTH_TEST_CHECK(synth::deliver_complete_audio(nullptr, 0, info, &sink, &result) == SYNTH_OK);
    SYNTH_TEST_CHECK(capture.calls == 0 && result.frames_emitted == 0);
    return 0;
}
