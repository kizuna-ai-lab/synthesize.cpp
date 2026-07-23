#include "duration-predictor-host.h"

#include "vits.h"

#include <cmath>
#include <limits>

namespace synth::vits {

synth_status_t prepare_duration_input(size_t                     token_count,
                                      const std::vector<float> & duration_noise_channel_major,
                                      float                      noise_scale_w,
                                      PreparedDurationInput &    output) {
    output = {};
    if (token_count == 0 || token_count > std::numeric_limits<size_t>::max() / 2 ||
        duration_noise_channel_major.size() != 2 * token_count || !std::isfinite(noise_scale_w) ||
        noise_scale_w < 0.0f) {
        return SYNTH_ERR_INVALID_ARG;
    }
    output.noise_channel_fastest.resize(2 * token_count);
    for (size_t channel = 0; channel < 2; ++channel) {
        for (size_t token = 0; token < token_count; ++token) {
            const float value = duration_noise_channel_major[token + token_count * channel];
            if (!std::isfinite(value)) {
                output = {};
                return SYNTH_ERR_INVALID_ARG;
            }
            output.noise_channel_fastest[channel + 2 * token] = value;
        }
    }
    return SYNTH_OK;
}

synth_status_t finalize_duration_output(size_t                     token_count,
                                       const std::vector<float> & logw,
                                       DurationPredictorOutput &  output) {
    output = {};
    if (token_count == 0 || logw.size() != token_count) {
        return SYNTH_ERR_INTERNAL;
    }
    for (float value : logw) {
        if (!std::isfinite(value)) {
            return SYNTH_ERR_INTERNAL;
        }
    }
    output.token_count = token_count;
    output.logw        = logw;
    return SYNTH_OK;
}

}  // namespace synth::vits
