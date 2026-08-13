#pragma once

#include "synthesize.h"

#include <cstdint>
#include <vector>

namespace synth::qwen3tts {

struct HParams;
struct SpeakerEncoderWeights;

// What one enrollment produced. `ref_rms` is instrumentation and a gate: a
// digitally silent clip produces an embedding that is finite, stable and
// meaningless, so it is refused by name rather than cloned.
struct XVectorEncoding {
    std::vector<float> x_vector;  // enc_dim floats
    float              ref_rms    = 0.0f;
    uint64_t           mel_frames = 0;
};

// PCM at the package's declared reference rate to a speaker embedding: mel
// front end, ECAPA graph, one shot on the CPU.
//
// The speaker encoder stays on the CPU in Plan 2. The catalog's twin-context
// note (src/arch/qwen3-tts/catalog.cpp:657-662) says the encoder half "is
// resolved against the package only, and stays on the CPU until a graph
// exists that reads it"; a graph now exists, and moving it to an accelerator
// is a measured decision Plan 4 makes, not a side effect of this one.
synth_status_t encode_speaker_reference(const HParams &               hparams,
                                        const SpeakerEncoderWeights & weights,
                                        const std::vector<float> &    pcm,
                                        int                           threads,
                                        XVectorEncoding &             output,
                                        const char *&                 out_diagnostic_code,
                                        const char *&                 out_diagnostic_message);

}  // namespace synth::qwen3tts
