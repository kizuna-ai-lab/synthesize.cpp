#include "arch/omnivoice/codec-host.h"

#include <cmath>
#include <cstddef>

namespace synth::omnivoice {

synth_status_t validate_code_grid(const std::vector<int32_t> & codes,
                                  uint64_t                     frame_count,
                                  uint32_t                     num_codebooks,
                                  uint32_t                     codebook_size) {
    if (frame_count == 0 || num_codebooks == 0 || codebook_size == 0 ||
        codes.size() != size_t(num_codebooks) * frame_count) {
        return SYNTH_ERR_INVALID_ARG;
    }
    for (int32_t code : codes) {
        if (code < 0 || uint32_t(code) >= codebook_size) {
            return SYNTH_ERR_INVALID_ARG;
        }
    }
    return SYNTH_OK;
}

void apply_no_reference_volume(std::vector<float> & audio) {
    float peak = 0.0f;
    for (float value : audio) {
        peak = std::fmax(peak, std::fabs(value));
    }
    if (peak <= 1e-6f) {
        return;
    }
    for (float & value : audio) {
        value = value / peak * 0.5f;
    }
}

void apply_reference_volume(std::vector<float> & audio, float ref_rms) {
    if (ref_rms <= 0.0f || ref_rms >= 0.1f) {
        return;
    }
    const float scale = ref_rms / 0.1f;
    for (float & value : audio) {
        value *= scale;
    }
}

}  // namespace synth::omnivoice
