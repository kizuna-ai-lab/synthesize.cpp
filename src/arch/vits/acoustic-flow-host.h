#pragma once

#include "synthesize.h"

#include <cstdint>
#include <vector>

namespace synth::vits {

struct AcousticFlowOutput;
struct LatentSamplingOutput;

struct PreparedAcousticFlowInput {
    uint32_t           channels    = 0;
    uint64_t           frame_count = 0;
    std::vector<float> z_p_channel_fastest;
};

synth_status_t prepare_acoustic_flow_input(const LatentSamplingOutput & latent, PreparedAcousticFlowInput & output);

synth_status_t finalize_acoustic_flow_output(uint32_t                   channels,
                                             uint64_t                   frame_count,
                                             const std::vector<float> & z_channel_fastest,
                                             AcousticFlowOutput &       output);

}  // namespace synth::vits
