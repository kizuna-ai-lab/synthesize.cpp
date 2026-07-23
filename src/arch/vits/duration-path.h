#pragma once

#include "synthesize.h"

#include <cstdint>
#include <vector>

namespace synth::vits {

struct HParams;

struct ResolvedDurationPath {
    uint64_t           frame_count = 0;
    std::vector<float> w_ceil;
    // Logical [frame_count, token_count], with token index contiguous.
    std::vector<float> attention;
};

synth_status_t resolve_duration_path(const HParams &            hparams,
                                     const std::vector<float> & logw,
                                     float                      speaking_rate,
                                     ResolvedDurationPath &     output);

}  // namespace synth::vits
