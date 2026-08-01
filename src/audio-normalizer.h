#pragma once

#include "synthesize.h"

#include <cstdint>
#include <vector>

namespace synth {

// ADR 0009 (docs/adr/0009-pin-reference-audio-normalization.md): the private,
// family-independent Audio Normalizer behind Reference Audio Profile
// preparation. Fixed order: validate, convert channels, then resample, using
// libsamplerate 0.2.2's SRC_SINC_BEST_QUALITY converter (vendored in
// third_party/libsamplerate/, never a system library). Matching input and
// target format is a plain copy with no resampler invocation at all, which is
// what makes it provably bit-identical; every other conversion goes through
// internal storage and is never padded, cropped, or trimmed to force a
// particular output length.
struct NormalizedReference {
    std::vector<float> pcm;         // Interleaved F32 PCM at the target format.
    uint64_t           frames = 0;  // Frames actually produced, at target_rate.
};

// Reference Frame Equivalent (docs/c-interface.md:486-496):
// ceil(input_frames * target_rate / input_rate), computed with checked uint64
// arithmetic before any allocation. This is a target-rate duration measure
// for preflight limit checks, not a promise that normalize_reference() below
// returns exactly this many frames -- the resampler's actual drained output
// can differ by a frame or two either way, and this module never pads or
// crops its result to force a match.
uint64_t reference_frame_equivalent(uint64_t input_frames, uint32_t input_rate, uint32_t target_rate);

// Normalizes one Reference Audio clip to a Loaded Model's declared target
// sample rate and channel count.
//
// Error mapping (docs/c-interface.md:486-496):
//   - input_rate outside [8000, 192000], or input_channels outside [1, 2]:
//     SYNTH_ERR_UNSUPPORTED_INPUT.
//   - null pcm, zero frames, a non-finite sample, or frame-count arithmetic
//     that would overflow: SYNTH_ERR_INVALID_ARG.
// Per-clip and total Reference Frame Equivalent length limits are the
// caller's concern, not this module's: it normalizes whatever finite input it
// is given and lets the caller decide whether the result was too long.
synth_status_t normalize_reference(const float *         pcm,
                                   uint64_t              frames,
                                   uint32_t              input_rate,
                                   uint32_t              input_channels,
                                   uint32_t              target_rate,
                                   uint32_t              target_channels,
                                   NormalizedReference & output);

}  // namespace synth
