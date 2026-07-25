#include "audio-delivery.h"

#include <cstddef>
#include <cstring>

namespace synth {
namespace {

template <typename Value> void write_result_field(synth_result_t * result, size_t offset, const Value & value) {
    if (result != nullptr && result->struct_size >= offset + sizeof(value)) {
        std::memcpy(reinterpret_cast<unsigned char *>(result) + offset, &value, sizeof(value));
    }
}

void write_result_metadata(synth_result_t * result, const AudioDeliveryInfo & info) {
    write_result_field(result, offsetof(synth_result_t, actual_seed), info.actual_seed);
    write_result_field(result, offsetof(synth_result_t, sample_rate), info.sample_rate);
    write_result_field(result, offsetof(synth_result_t, channel_count), info.channel_count);
    write_result_field(result, offsetof(synth_result_t, flags), info.result_flags);
    write_result_field(result, offsetof(synth_result_t, resolved_language_tag), info.resolved_language_tag);
    write_result_field(result, offsetof(synth_result_t, resolved_language_tag_size), info.resolved_language_size);
    write_result_field(result, offsetof(synth_result_t, resolved_voice_id), info.resolved_voice_id);
    write_result_field(result, offsetof(synth_result_t, resolved_voice_id_size), info.resolved_voice_size);
}

}  // namespace

bool valid_audio_sink(const synth_audio_sink_t * sink) {
    return sink != nullptr && sink->struct_size >= offsetof(synth_audio_sink_t, write) + sizeof(sink->write) &&
           sink->write != nullptr;
}

synth_status_t deliver_complete_audio(const float *              samples,
                                      uint64_t                   frame_count,
                                      const AudioDeliveryInfo &  info,
                                      const synth_audio_sink_t * sink,
                                      synth_result_t *           out_result) {
    if (!valid_audio_sink(sink) || (samples == nullptr && frame_count != 0) || info.sample_rate == 0 ||
        info.channel_count == 0) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (out_result != nullptr && out_result->struct_size < sizeof(uint64_t)) {
        return SYNTH_ERR_BAD_STRUCT_SIZE;
    }

    write_result_metadata(out_result, info);
    if (frame_count == 0) {
        return SYNTH_OK;
    }

    synth_audio_chunk_t chunk{};
    chunk.struct_size   = sizeof(chunk);
    chunk.samples       = samples;
    chunk.frame_count   = frame_count;
    chunk.frame_offset  = 0;
    chunk.sample_rate   = info.sample_rate;
    chunk.channel_count = info.channel_count;

    synth_sink_result_t callback_result = SYNTH_SINK_ERROR;
    try {
        callback_result = sink->write(sink->user_data, &chunk);
    } catch (...) {
        callback_result = SYNTH_SINK_ERROR;
    }
    write_result_field(out_result, offsetof(synth_result_t, frames_emitted), frame_count);
    if (callback_result == SYNTH_SINK_CONTINUE) {
        return SYNTH_OK;
    }
    if (callback_result == SYNTH_SINK_CANCEL) {
        return SYNTH_ERR_CANCELLED;
    }
    return SYNTH_ERR_SINK;
}

}  // namespace synth
