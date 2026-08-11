#pragma once

#include "synthesize.h"

#include <cstdint>
#include <vector>

namespace synth::qwen3tts {

struct SpeakerEncoderParams;

// A log-mel spectrogram, mel-major: `values[frame * bins + bin]`. That is the
// layout the ECAPA graph reads directly as a ggml [bins, frames] F32 tensor,
// so nothing transposes between here and the graph.
struct MelSpectrogram {
    std::vector<float> values;
    uint32_t           bins   = 0;
    uint64_t           frames = 0;
};

// 24 kHz mono F32 PCM to log-mel at the package's own pinned parameters.
//
// Host DSP rather than a GGML graph, per the Stage 2 design's section 5: it
// runs once per enrollment, its cost is negligible beside the ECAPA forward,
// and keeping it on the host avoids a backend-placement question for a stage
// with no downstream shape dependence.
//
// Every conversion this performs -- mel scale, filterbank normalization,
// magnitude-vs-power, log floor, window shape, centring -- is transcribed
// from the pinned upstream source and recorded in the Task 1 oracle's
// conventions.json. None of the six is derivable from the package's eight
// metadata fields, and each one silently changes the answer rather than
// failing, which is why they are pinned by a dump rather than by a comment.
//
// Errors:
//   SYNTH_ERR_UNSUPPORTED_INPUT -- n_fft is not a power of two (the transform
//                                  is radix-2), or a parameter is zero.
//   SYNTH_ERR_INVALID_ARG       -- a non-finite sample, or a clip too short to
//                                  produce a single frame.
synth_status_t compute_log_mel(const SpeakerEncoderParams & params,
                               const std::vector<float> &   pcm,
                               MelSpectrogram &             output);

}  // namespace synth::qwen3tts
