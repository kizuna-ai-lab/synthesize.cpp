#pragma once

#include "synthesize.h"

#include <cstdint>
#include <vector>

namespace synth::kokoro {

struct HParams;

struct DurationResult {
    std::vector<int64_t> pred_dur;  // one rounded step count per token
    uint64_t             y_length = 0;
    // Logical [token_count, y_length] one-hot alignment with the frame index
    // contiguous, matching the reference probe layout.
    std::vector<float>   alignment;
};

// Resolves per-token durations from the graph's logits and builds the expansion
// alignment.
//
// This is a host seam because it turns a distribution into a concrete output
// length, which the following graphs need as a static shape. Upstream sums the
// sigmoid over the duration bins, divides by the speaking rate, rounds to the
// nearest integer and clamps to at least one step per token.
//
// `logits` is the graph output in [max_dur, token_count] layout. `max_frames`
// is the effective output limit in native frames; a resolved length that would
// exceed it returns SYNTH_ERR_OUTPUT_LIMIT before any alignment is allocated.
synth_status_t resolve_durations(const std::vector<float> & logits,
                                 const HParams &            hparams,
                                 uint64_t                   token_count,
                                 float                      speaking_rate,
                                 uint64_t                   max_frames,
                                 DurationResult &           output);

}  // namespace synth::kokoro
