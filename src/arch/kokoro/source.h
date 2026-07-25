#pragma once

#include "synthesize.h"

#include <cstdint>
#include <vector>

namespace synth::kokoro {

struct HParams;

// The two random draws the pinned harmonic source consumes. Port validation
// replays the oracle's exact tensors through this seam; ordinary synthesis
// fills them from the project's own seeded stream.
struct SourceRandomInputs {
    // Initial phase per harmonic, whose first entry upstream forces to zero.
    std::vector<float> rand_ini;
    // Additive noise, laid out frame-major as [upsampled_frames, harmonics].
    std::vector<float> noise;
};

struct SourceResult {
    std::vector<float> har_source;  // [upsampled_frames], the merged excitation
    // [2 * bins, frames] with magnitude bins first and phase bins second, in
    // the family's [channels, time] order.
    std::vector<float> har;
    uint64_t           frames = 0;
    uint64_t           bins   = 0;
};

// Number of harmonics the source generates, which is the fundamental plus its
// overtones.
uint32_t source_harmonic_count(const HParams & hparams);

// Samples the source module consumes for a given F0 curve length.
uint64_t source_upsampled_length(const HParams & hparams, uint64_t f0_length);

// Builds the harmonic-plus-noise excitation and its short-time spectrum.
//
// This is a host seam. The module has one tiny learned projection and is
// otherwise phase accumulation and resampling, which GGML expresses poorly, and
// it sits between two graph stages that both need concrete shapes. Keeping it
// on the host also gives port validation a place to inject recorded randomness.
//
// `f0` is the prosody stage's curve at twice the frame rate. `merge_weight` and
// `merge_bias` are the stored 9-to-1 projection that sums the harmonics.
synth_status_t build_harmonic_source(const HParams &            hparams,
                                     const std::vector<float> & f0,
                                     const std::vector<float> & merge_weight,
                                     float                      merge_bias,
                                     const SourceRandomInputs & random,
                                     SourceResult &             output);

}  // namespace synth::kokoro
