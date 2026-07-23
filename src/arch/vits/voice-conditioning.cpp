#include "voice-conditioning.h"

#include "ggml.h"
#include "weights.h"

#include <climits>

namespace synth::vits {

VoiceConditioning build_voice_conditioning(ggml_context *       context,
                                           const VoiceWeights & weights,
                                           const HParams &      hparams,
                                           uint32_t             speaker_index) {
    VoiceConditioning result;
    if (context == nullptr) {
        return result;
    }
    if (hparams.speaker_count == 0 && hparams.conditioning_channels == 0) {
        result.valid = speaker_index == UINT32_MAX && weights.embedding == nullptr;
        return result;
    }
    if (speaker_index >= hparams.speaker_count || hparams.conditioning_channels == 0 || weights.embedding == nullptr) {
        return result;
    }

    result.speaker_index = ggml_new_tensor_1d(context, GGML_TYPE_I32, 1);
    ggml_set_name(result.speaker_index, "voice.speaker_index");
    ggml_set_input(result.speaker_index);
    result.embedding = ggml_get_rows(context, weights.embedding, result.speaker_index);
    ggml_set_name(result.embedding, "voice.embedding");
    result.valid = true;
    return result;
}

}  // namespace synth::vits
