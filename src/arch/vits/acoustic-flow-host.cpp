#include "acoustic-flow-host.h"

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

synth_status_t prepare_acoustic_flow_input(const LatentSamplingOutput & latent, PreparedAcousticFlowInput & output) {
    output          = {};
    size_t elements = 0;
    if (!checked_elements(latent.channels, latent.frame_count, elements) || latent.z_p.size() != elements ||
        !all_finite(latent.z_p)) {
        return SYNTH_ERR_INVALID_ARG;
    }
    output.channels    = latent.channels;
    output.frame_count = latent.frame_count;
    output.z_p_channel_fastest.resize(elements);
    for (size_t channel = 0; channel < latent.channels; ++channel) {
        for (size_t frame = 0; frame < latent.frame_count; ++frame) {
            const size_t source                     = frame + static_cast<size_t>(latent.frame_count) * channel;
            const size_t destination                = channel + static_cast<size_t>(latent.channels) * frame;
            output.z_p_channel_fastest[destination] = latent.z_p[source];
        }
    }
    return SYNTH_OK;
}

synth_status_t finalize_acoustic_flow_output(uint32_t                   channels,
                                             uint64_t                   frame_count,
                                             const std::vector<float> & z_channel_fastest,
                                             AcousticFlowOutput &       output) {
    output          = {};
    size_t elements = 0;
    if (!checked_elements(channels, frame_count, elements) || z_channel_fastest.size() != elements ||
        !all_finite(z_channel_fastest)) {
        return SYNTH_ERR_INTERNAL;
    }
    output.channels    = channels;
    output.frame_count = frame_count;
    output.z.resize(elements);
    for (size_t channel = 0; channel < channels; ++channel) {
        for (size_t frame = 0; frame < frame_count; ++frame) {
            const size_t source      = channel + static_cast<size_t>(channels) * frame;
            const size_t destination = frame + static_cast<size_t>(frame_count) * channel;
            output.z[destination]    = z_channel_fastest[source];
        }
    }
    return SYNTH_OK;
}

}  // namespace synth::vits
