#include "latent-sampling-host.h"

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

synth_status_t prepare_latent_sampling_input(const PriorExpansionOutput &  prior,
                                             const std::vector<float> &    latent_noise,
                                             float                         noise_scale,
                                             PreparedLatentSamplingInput & output) {
    output          = {};
    size_t elements = 0;
    if (!checked_elements(prior.channels, prior.frame_count, elements) || prior.m_p.size() != elements ||
        prior.logs_p.size() != elements || latent_noise.size() != elements || !std::isfinite(noise_scale) ||
        noise_scale < 0.0f || !all_finite(prior.m_p) || !all_finite(prior.logs_p) || !all_finite(latent_noise)) {
        return SYNTH_ERR_INVALID_ARG;
    }

    output.channels    = prior.channels;
    output.frame_count = prior.frame_count;
    output.noise_scale = noise_scale;
    output.m_p_channel_fastest.resize(elements);
    output.logs_p_channel_fastest.resize(elements);
    output.noise_channel_fastest.resize(elements);
    for (size_t channel = 0; channel < prior.channels; ++channel) {
        for (size_t frame = 0; frame < prior.frame_count; ++frame) {
            const size_t source                        = frame + static_cast<size_t>(prior.frame_count) * channel;
            const size_t destination                   = channel + static_cast<size_t>(prior.channels) * frame;
            output.m_p_channel_fastest[destination]    = prior.m_p[source];
            output.logs_p_channel_fastest[destination] = prior.logs_p[source];
            output.noise_channel_fastest[destination]  = latent_noise[source];
        }
    }
    return SYNTH_OK;
}

synth_status_t finalize_latent_sampling_output(uint32_t                   channels,
                                               uint64_t                   frame_count,
                                               const std::vector<float> & z_p_channel_fastest,
                                               LatentSamplingOutput &     output) {
    output          = {};
    size_t elements = 0;
    if (!checked_elements(channels, frame_count, elements) || z_p_channel_fastest.size() != elements ||
        !all_finite(z_p_channel_fastest)) {
        return SYNTH_ERR_INTERNAL;
    }

    output.channels    = channels;
    output.frame_count = frame_count;
    output.z_p.resize(elements);
    for (size_t channel = 0; channel < channels; ++channel) {
        for (size_t frame = 0; frame < frame_count; ++frame) {
            const size_t source      = channel + static_cast<size_t>(channels) * frame;
            const size_t destination = frame + static_cast<size_t>(frame_count) * channel;
            output.z_p[destination]  = z_p_channel_fastest[source];
        }
    }
    return SYNTH_OK;
}

}  // namespace synth::vits
