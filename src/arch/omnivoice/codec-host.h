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

// The clone-request volume branch (Task 14), the sibling
// `apply_no_reference_volume` above never takes: upstream's
// `_post_process_audio` (omnivoice/models/omnivoice.py:898-899) --
//   if ref_rms is not None and ref_rms < 0.1:
//       generated_audio = generated_audio * ref_rms / 0.1
// -- transcribed as `rms >= 0.1` -> no change (the chain falls through);
// `0 < rms < 0.1` -> every sample scaled by `rms / 0.1`. It is the exact
// inverse of the quiet-reference BOOST `clip_and_boost_reference`
// (reference-encoder-host.h) applies when the same reference is encoded
// (`ref_wav *= 0.1 / ref_rms`), so a quiet voice comes back quiet rather than
// inheriting the boost's loudness.
//
// `ref_rms == 0.0f` is unreachable in a real synthesis: voice-profile.cpp's
// create_from_reference handler rejects a digitally silent reference
// (jiangzhuo's ruling, 2026-08-01 -- docs/porting/families/omnivoice.md's
// "Silent-Reference Rejection" note) before a ClonePrompt can ever exist to
// carry one here. Guarded anyway (`<= 0.0f`, not `== 0.0f`) so a defect
// upstream of this call -- including Model::synthesize's own "no reference"
// sentinel of -1.0f -- degrades to a no-op instead of scaling by a
// nonsensical or divide-prone value.
void apply_reference_volume(std::vector<float> & audio, float ref_rms);

}  // namespace synth::omnivoice
