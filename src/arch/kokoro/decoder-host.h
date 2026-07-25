#pragma once

#include "synthesize.h"

#include <cstdint>
#include <vector>

namespace synth::kokoro {

struct HParams;

// Samples the inverse transform produces for a given frame count.
uint64_t inverse_stft_length(const HParams & hparams, uint64_t frames);

// Turns the generator's spectrum into the waveform.
//
// This is the third host seam. The inverse transform is an overlap-add divided
// by the overlap-added squared window, which GGML has no operation for, and it
// is the last step before the audio leaves the family, so nothing downstream
// needs it inside a graph.
//
// `spectrum` is the generator's [n_fft + 2, frames] output as the graph leaves
// it: the first `n_fft / 2 + 1` rows are logarithmic magnitudes and the rest
// are pre-sine angles. Applying those two activations here keeps them next to
// the transform that consumes them.
synth_status_t inverse_stft(const HParams &            hparams,
                            const std::vector<float> & spectrum,
                            uint64_t                   frames,
                            std::vector<float> &       audio);

}  // namespace synth::kokoro
