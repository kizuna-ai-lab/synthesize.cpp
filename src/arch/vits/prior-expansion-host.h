#pragma once

#include "synthesize.h"

#include <cstdint>
#include <vector>

namespace synth::vits {

struct DurationOutput;
struct PriorExpansionOutput;
struct TextEncoderOutput;

struct PreparedPriorExpansionInput {
    uint32_t           channels    = 0;
    uint64_t           token_count = 0;
    uint64_t           frame_count = 0;
    std::vector<float> m_p_channel_fastest;
    std::vector<float> logs_p_channel_fastest;
};

synth_status_t prepare_prior_expansion_input(const TextEncoderOutput &     text,
                                             const DurationOutput &        duration,
                                             PreparedPriorExpansionInput & output);

synth_status_t finalize_prior_expansion_output(uint32_t                   channels,
                                               uint64_t                   frame_count,
                                               const std::vector<float> & m_p_channel_fastest,
                                               const std::vector<float> & logs_p_channel_fastest,
                                               PriorExpansionOutput &     output);

}  // namespace synth::vits
