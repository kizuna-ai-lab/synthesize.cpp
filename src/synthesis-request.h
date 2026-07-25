#pragma once

#include "model-info.h"
#include "synthesize.h"

#include <cstdint>
#include <vector>

namespace synth {

struct PreparedSynthesisRequest {
    std::vector<int32_t>            token_ids;
    uint64_t                        seed                   = 0;
    float                           speaking_rate          = 1.0f;
    uint64_t                        effective_frame_limit  = 0;
    synth_cancel_callback_t         should_cancel          = nullptr;
    void *                          cancel_user_data       = nullptr;
    const synth_diagnostic_sink_t * diagnostics            = nullptr;
    const char *                    resolved_language_tag  = nullptr;
    uint64_t                        resolved_language_size = 0;
    uint32_t                        speaker_index          = UINT32_MAX;
    const char *                    resolved_voice_id      = nullptr;
    uint64_t                        resolved_voice_size    = 0;
};

synth_status_t prepare_synthesis_request(const ModelInfo &          info,
                                         const synth_request_t *    request,
                                         PreparedSynthesisRequest & output);

}  // namespace synth
