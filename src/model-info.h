#pragma once

#include "text-frontend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace synth {

enum class ModelFamily {
    Vits,
    Kokoro,
    Qwen3Tts,
};

// What the core runtime needs from a Loaded Model, with nothing family-specific
// in it.
//
// Each family also reports quantities only its own graphs use — VITS's latent
// channel count, for instance — and those stay in the family's own info struct.
// This is the part the public interface, the request validator, and the audio
// delivery path read, and it is deliberately the same shape for every family so
// that adding one does not reach into those.
struct ModelInfo {
    ModelFamily                         family              = ModelFamily::Vits;
    bool                                has_package_default = false;
    std::vector<std::string>            preset_voice_ids;
    std::vector<uint32_t>               preset_voice_flags;
    std::shared_ptr<const TextFrontend> text_frontend;
    uint32_t                            input_flags          = 0;
    uint32_t                            capability_flags     = 0;
    uint32_t                            output_sample_rate   = 0;
    uint32_t                            output_channel_count = 0;
    uint32_t                            vocab_size           = 0;
    // Output samples one predicted frame becomes. VITS calls this the hop
    // length; Kokoro's duration steps are 600 samples each.
    uint32_t                            samples_per_frame    = 0;
    uint64_t                            max_input_tokens     = 0;
    uint64_t                            max_output_frames    = 0;
    float                               min_speaking_rate    = 0.0f;
    float                               max_speaking_rate    = 0.0f;
};

}  // namespace synth
