#pragma once

#include <cstdint>

struct ggml_context;
struct ggml_tensor;

namespace synth::vits {

struct HParams;
struct VoiceWeights;

struct VoiceConditioning {
    bool          valid         = false;
    ggml_tensor * speaker_index = nullptr;
    ggml_tensor * embedding     = nullptr;
};

VoiceConditioning build_voice_conditioning(ggml_context *       context,
                                           const VoiceWeights & weights,
                                           const HParams &      hparams,
                                           uint32_t             speaker_index);

}  // namespace synth::vits
