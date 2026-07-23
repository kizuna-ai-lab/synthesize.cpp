#include "duration-path.h"

#include "weights.h"

#include <cmath>
#include <cstddef>
#include <limits>

namespace synth::vits {

synth_status_t resolve_duration_path(const HParams &            hparams,
                                     const std::vector<float> & logw,
                                     float                      speaking_rate,
                                     ResolvedDurationPath &     output) {
    output = {};
    if (logw.empty() || logw.size() > hparams.max_input_tokens || hparams.hop_length == 0 ||
        hparams.max_output_frames < hparams.hop_length || !std::isfinite(hparams.min_speaking_rate) ||
        !std::isfinite(hparams.max_speaking_rate) || hparams.min_speaking_rate <= 0.0f ||
        hparams.max_speaking_rate < hparams.min_speaking_rate || !std::isfinite(speaking_rate) ||
        speaking_rate < hparams.min_speaking_rate || speaking_rate > hparams.max_speaking_rate) {
        return SYNTH_ERR_INVALID_ARG;
    }

    const uint64_t max_duration_frames = hparams.max_output_frames / hparams.hop_length;
    const float    length_scale        = 1.0f / speaking_rate;
    uint64_t       frame_sum           = 0;
    output.w_ceil.reserve(logw.size());
    for (float value : logw) {
        if (!std::isfinite(value)) {
            output = {};
            return SYNTH_ERR_INVALID_ARG;
        }
        const float duration = std::exp(value) * length_scale;
        const float rounded  = std::ceil(duration);
        if (!std::isfinite(rounded) || rounded < 0.0f ||
            static_cast<long double>(rounded) > static_cast<long double>(max_duration_frames)) {
            output = {};
            return SYNTH_ERR_OUTPUT_LIMIT;
        }
        const uint64_t frames = static_cast<uint64_t>(rounded);
        if (frames > max_duration_frames - frame_sum) {
            output = {};
            return SYNTH_ERR_OUTPUT_LIMIT;
        }
        frame_sum += frames;
        output.w_ceil.push_back(rounded);
    }

    output.frame_count = frame_sum > 0 ? frame_sum : 1;
    if (output.frame_count > std::numeric_limits<size_t>::max() / logw.size()) {
        output = {};
        return SYNTH_ERR_OUTPUT_LIMIT;
    }
    output.attention.assign(static_cast<size_t>(output.frame_count) * logw.size(), 0.0f);
    uint64_t frame = 0;
    for (size_t token = 0; token < output.w_ceil.size(); ++token) {
        const uint64_t end = frame + static_cast<uint64_t>(output.w_ceil[token]);
        for (; frame < end; ++frame) {
            output.attention[static_cast<size_t>(frame) * logw.size() + token] = 1.0f;
        }
    }
    return SYNTH_OK;
}

}  // namespace synth::vits
