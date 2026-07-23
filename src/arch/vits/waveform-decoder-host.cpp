#include "waveform-decoder-host.h"

#include "vits.h"

#include <cmath>
#include <cstddef>
#include <limits>

namespace synth::vits {

namespace {

bool checked_elements(uint64_t first, uint64_t second, size_t & output) {
    if (first == 0 || second == 0 || first > std::numeric_limits<size_t>::max() ||
        second > std::numeric_limits<size_t>::max() ||
        static_cast<size_t>(first) > std::numeric_limits<size_t>::max() / static_cast<size_t>(second)) {
        return false;
    }
    output = static_cast<size_t>(first) * static_cast<size_t>(second);
    return true;
}

bool all_finite(const std::vector<float> & values) {
    for (float value : values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

}  // namespace

synth_status_t prepare_waveform_decoder_input(const AcousticFlowOutput & flow,
                                               PreparedWaveformDecoderInput & output) {
    output          = {};
    size_t elements = 0;
    if (!checked_elements(flow.channels, flow.frame_count, elements) || flow.z.size() != elements ||
        !all_finite(flow.z)) {
        return SYNTH_ERR_INVALID_ARG;
    }
    output.channels    = flow.channels;
    output.frame_count = flow.frame_count;
    output.z_channel_fastest.resize(elements);
    for (size_t channel = 0; channel < flow.channels; ++channel) {
        for (size_t frame = 0; frame < flow.frame_count; ++frame) {
            const size_t source = frame + static_cast<size_t>(flow.frame_count) * channel;
            const size_t destination = channel + static_cast<size_t>(flow.channels) * frame;
            output.z_channel_fastest[destination] = flow.z[source];
        }
    }
    return SYNTH_OK;
}

synth_status_t finalize_waveform_decoder_output(uint64_t                   sample_count,
                                                const std::vector<float> & pcm,
                                                WaveformDecoderOutput &     output) {
    output = {};
    if (sample_count == 0 || sample_count > std::numeric_limits<size_t>::max() ||
        pcm.size() != static_cast<size_t>(sample_count) || !all_finite(pcm)) {
        return SYNTH_ERR_INTERNAL;
    }
    output.sample_count = sample_count;
    output.pcm          = pcm;
    return SYNTH_OK;
}

}  // namespace synth::vits
