#include "duration-host.h"

#include "weights.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <new>

namespace synth::kokoro {

namespace {

float sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

}  // namespace

synth_status_t resolve_durations(const std::vector<float> & logits,
                                 const HParams &            hparams,
                                 uint64_t                   token_count,
                                 float                      speaking_rate,
                                 uint64_t                   max_frames,
                                 DurationResult &           output) {
    output = DurationResult{};
    if (token_count == 0 || hparams.max_dur == 0 || hparams.samples_per_frame == 0) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (!std::isfinite(speaking_rate) || speaking_rate <= 0.0f) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (logits.size() != token_count * hparams.max_dur) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (max_frames == 0) {
        return SYNTH_ERR_INVALID_ARG;
    }

    try {
        output.pred_dur.resize(token_count);
    } catch (const std::bad_alloc &) {
        return SYNTH_ERR_OOM;
    }

    uint64_t total = 0;
    for (uint64_t token = 0; token < token_count; ++token) {
        const float * row = logits.data() + token * hparams.max_dur;
        float         sum = 0.0f;
        for (uint32_t bin = 0; bin < hparams.max_dur; ++bin) {
            if (!std::isfinite(row[bin])) {
                std::fprintf(stderr, "kokoro: duration logits contain a non-finite value\n");
                return SYNTH_ERR_INTERNAL;
            }
            sum += sigmoid(row[bin]);
        }
        // A faster rate divides the predicted duration, exactly as upstream does.
        const float scaled  = sum / speaking_rate;
        const float rounded = std::round(scaled);
        // Every token occupies at least one step, so a token is never dropped.
        int64_t     steps   = rounded < 1.0f ? 1 : static_cast<int64_t>(rounded);
        if (!std::isfinite(scaled) || steps < 1) {
            steps = 1;
        }
        output.pred_dur[token] = steps;
        total += static_cast<uint64_t>(steps);
    }

    if (total == 0) {
        return SYNTH_ERR_INTERNAL;
    }
    // Reject before allocating the alignment, which is quadratic in the inputs.
    if (total > max_frames / hparams.samples_per_frame) {
        std::fprintf(stderr, "kokoro: resolved %llu frames exceeds the effective output limit\n",
                     static_cast<unsigned long long>(total));
        return SYNTH_ERR_OUTPUT_LIMIT;
    }
    if (total > std::numeric_limits<uint64_t>::max() / token_count) {
        return SYNTH_ERR_OUTPUT_LIMIT;
    }

    output.y_length = total;
    try {
        output.alignment.assign(token_count * total, 0.0f);
    } catch (const std::bad_alloc &) {
        output = DurationResult{};
        return SYNTH_ERR_OOM;
    }

    uint64_t frame = 0;
    for (uint64_t token = 0; token < token_count; ++token) {
        const uint64_t steps = static_cast<uint64_t>(output.pred_dur[token]);
        for (uint64_t step = 0; step < steps; ++step) {
            output.alignment[token * total + frame] = 1.0f;
            ++frame;
        }
    }
    return SYNTH_OK;
}

}  // namespace synth::kokoro
