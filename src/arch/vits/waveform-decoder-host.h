#pragma once

#include "synthesize.h"

#include <cstdint>
#include <vector>

namespace synth::vits {

struct AcousticFlowOutput;
struct WaveformDecoderOutput;

struct PreparedWaveformDecoderInput {
    uint32_t           channels    = 0;
    uint64_t           frame_count = 0;
    std::vector<float> z_channel_fastest;
};

synth_status_t prepare_waveform_decoder_input(const AcousticFlowOutput & flow, PreparedWaveformDecoderInput & output);

synth_status_t finalize_waveform_decoder_output(uint64_t                   sample_count,
                                                const std::vector<float> & pcm,
                                                WaveformDecoderOutput &    output);

}  // namespace synth::vits
