#pragma once

#include "synthesize.h"

#include <cstdint>
#include <vector>

namespace synth::omnivoice {

// Host-side guards around the codec, per the discrete-outputs rule: the grid
// is discrete, so its validation never rides the graph.

// Every value must be a real code in [0, codebook_size): the mask id or
// anything past the table means the loop (or a replay input) is broken, and
// get_rows would read a row that exists but means nothing.
synth_status_t validate_code_grid(const std::vector<int32_t> & codes,
                                  uint64_t                     frame_count,
                                  uint32_t                     num_codebooks,
                                  uint32_t                     codebook_size);

// The ungated no-reference volume branch measured at intake: when the peak
// exceeds 1e-6, every sample becomes sample / peak * 0.5 -- two float
// operations per element, numpy's order, so the oracle's bytes reproduce.
// Applies to auto-voice and voice-design output; the clone branches are
// Plan 3's.
void apply_no_reference_volume(std::vector<float> & audio);

}  // namespace synth::omnivoice
