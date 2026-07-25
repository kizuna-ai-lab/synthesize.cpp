#pragma once

#include "synthesize.h"

#include <cstddef>
#include <vector>

namespace synth::vits {

struct DurationPredictorOutput;

struct PreparedDurationInput {
    std::vector<float> noise_channel_fastest;
};

synth_status_t prepare_duration_input(size_t                     token_count,
                                      const std::vector<float> & duration_noise_channel_major,
                                      float                      noise_scale_w,
                                      PreparedDurationInput &    output);

synth_status_t finalize_duration_output(size_t                     token_count,
                                        const std::vector<float> & logw,
                                        DurationPredictorOutput &  output);

}  // namespace synth::vits
