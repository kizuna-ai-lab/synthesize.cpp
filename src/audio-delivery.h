#pragma once

#include "synthesize.h"

#include <cstdint>

namespace synth {

struct AudioDeliveryInfo {
    uint64_t     actual_seed            = 0;
    uint32_t     sample_rate            = 0;
    uint32_t     channel_count          = 0;
    synth_result_flags_t result_flags   = 0;
    const char * resolved_language_tag  = nullptr;
    uint64_t     resolved_language_size = 0;
    const char * resolved_voice_id      = nullptr;
    uint64_t     resolved_voice_size    = 0;
};

bool valid_audio_sink(const synth_audio_sink_t * sink);

synth_status_t deliver_complete_audio(const float *                samples,
                                      uint64_t                     frame_count,
                                      const AudioDeliveryInfo &    info,
                                      const synth_audio_sink_t *   sink,
                                      synth_result_t *             out_result);

}  // namespace synth
