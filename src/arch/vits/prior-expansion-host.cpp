#include "prior-expansion-host.h"

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

synth_status_t prepare_prior_expansion_input(const TextEncoderOutput &     text,
                                             const DurationOutput &        duration,
                                             PreparedPriorExpansionInput & output) {
    output                    = {};
    size_t text_elements      = 0;
    size_t attention_elements = 0;
    if (text.channels == 0 || text.token_count == 0 || text.token_count != duration.token_count ||
        duration.frame_count == 0 || !checked_elements(text.channels, text.token_count, text_elements) ||
        !checked_elements(duration.frame_count, duration.token_count, attention_elements) ||
        text.m_p.size() != text_elements || text.logs_p.size() != text_elements ||
        duration.attention.size() != attention_elements || !all_finite(text.m_p) || !all_finite(text.logs_p) ||
        !all_finite(duration.attention)) {
        return SYNTH_ERR_INVALID_ARG;
    }

    output.channels    = text.channels;
    output.token_count = text.token_count;
    output.frame_count = duration.frame_count;
    output.m_p_channel_fastest.resize(text_elements);
    output.logs_p_channel_fastest.resize(text_elements);
    for (size_t channel = 0; channel < text.channels; ++channel) {
        for (size_t token = 0; token < text.token_count; ++token) {
            const size_t source                        = token + static_cast<size_t>(text.token_count) * channel;
            const size_t destination                   = channel + static_cast<size_t>(text.channels) * token;
            output.m_p_channel_fastest[destination]    = text.m_p[source];
            output.logs_p_channel_fastest[destination] = text.logs_p[source];
        }
    }
    return SYNTH_OK;
}

synth_status_t finalize_prior_expansion_output(uint32_t                   channels,
                                               uint64_t                   frame_count,
                                               const std::vector<float> & m_p_channel_fastest,
                                               const std::vector<float> & logs_p_channel_fastest,
                                               PriorExpansionOutput &     output) {
    output          = {};
    size_t elements = 0;
    if (!checked_elements(channels, frame_count, elements) || m_p_channel_fastest.size() != elements ||
        logs_p_channel_fastest.size() != elements || !all_finite(m_p_channel_fastest) ||
        !all_finite(logs_p_channel_fastest)) {
        return SYNTH_ERR_INTERNAL;
    }

    output.channels    = channels;
    output.frame_count = frame_count;
    output.m_p.resize(elements);
    output.logs_p.resize(elements);
    for (size_t channel = 0; channel < channels; ++channel) {
        for (size_t frame = 0; frame < frame_count; ++frame) {
            const size_t source        = channel + static_cast<size_t>(channels) * frame;
            const size_t destination   = frame + static_cast<size_t>(frame_count) * channel;
            output.m_p[destination]    = m_p_channel_fastest[source];
            output.logs_p[destination] = logs_p_channel_fastest[source];
        }
    }
    return SYNTH_OK;
}

}  // namespace synth::vits
