#pragma once

#include "synthesize.h"

#include <cstdint>
#include <vector>

namespace synth::vits {

struct LatentSamplingOutput;
struct PriorExpansionOutput;

struct PreparedLatentSamplingInput {
    uint32_t           channels    = 0;
    uint64_t           frame_count = 0;
    float              noise_scale = 0.0f;
    std::vector<float> m_p_channel_fastest;
    std::vector<float> logs_p_channel_fastest;
    std::vector<float> noise_channel_fastest;
};

synth_status_t prepare_latent_sampling_input(const PriorExpansionOutput &  prior,
                                             const std::vector<float> &    latent_noise,
                                             float                         noise_scale,
                                             PreparedLatentSamplingInput & output);

synth_status_t finalize_latent_sampling_output(uint32_t                   channels,
                                               uint64_t                   frame_count,
                                               const std::vector<float> & z_p_channel_fastest,
                                               LatentSamplingOutput &     output);

}  // namespace synth::vits
