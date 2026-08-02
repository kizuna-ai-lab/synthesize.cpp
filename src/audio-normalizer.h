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
//
// IMPORTANT: this function does not itself validate `input_rate` against the
// Reference Audio format contract below -- a rate of 0 or one outside
// [SYNTH_REFERENCE_SAMPLE_RATE_MIN, SYNTH_REFERENCE_SAMPLE_RATE_MAX] still
// produces SOME number here (0 for a zero rate; a real, otherwise-unbounded
// value for an out-of-range one). A caller that derives anything from this
// (in particular, a length-based limit decision) before calling
// validate_reference_format() below risks shadowing an out-of-contract rate
// with a length-derived status instead (reviewer FINDING 1) -- always call
// validate_reference_format() first.
uint64_t reference_frame_equivalent(uint64_t input_frames, uint32_t input_rate, uint32_t target_rate);

// The format half of docs/c-interface.md:486-496's Reference Audio contract,
// pulled out of normalize_reference() below so a caller can apply it BEFORE
// deriving anything (in particular, a Reference Frame Equivalent) from a
// caller-supplied sample rate: `input_rate` outside
// [SYNTH_REFERENCE_SAMPLE_RATE_MIN, SYNTH_REFERENCE_SAMPLE_RATE_MAX] or
// `input_channels` outside [1, SYNTH_REFERENCE_CHANNELS_MAX] (both in
// include/synthesize.h, the single source of truth for these limits) is
// SYNTH_ERR_UNSUPPORTED_INPUT, independent of clip length.
// normalize_reference() calls this itself before touching a single sample;
// voice-profile.cpp's create_omnivoice_profile_from_reference calls this
// SAME function a second, EARLIER time, before reference_frame_equivalent()
// ever runs, so an out-of-contract rate can never be shadowed by whatever
// the RFE math derives from it (reviewer FINDING 1: rate 0 always produced
// "too_short"; rate 4000 produced three different statuses depending on
// clip length, because the RFE precheck ran first and happened to compute
// something plausible-looking from the invalid rate).
synth_status_t validate_reference_format(uint32_t input_rate, uint32_t input_channels);

// Normalizes one Reference Audio clip to a Loaded Model's declared target
// sample rate and channel count.
//
// Error mapping (docs/c-interface.md:486-496):
//   - input_rate or input_channels outside validate_reference_format()'s own
//     range (above): SYNTH_ERR_UNSUPPORTED_INPUT.
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
