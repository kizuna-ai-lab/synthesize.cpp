#pragma once

#include "text-frontend.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace synth {

enum class ModelFamily {
    Vits,
    Kokoro,
    Qwen3Tts,
    Omnivoice,
};

// One language a package declares a request may ask for, named the way the
// public interface names languages rather than the way the package does.
//
// The tag is BCP-47. Families whose packages name languages in full -- Qwen3-TTS
// carries "chinese", not "zh" -- translate on the way out, because the bridge
// belongs with whoever knows the package's vocabulary. `flags` carries
// SYNTH_LANGUAGE_* bits; REGIONAL_FALLBACK means a request may carry a region
// this entry answers for, so "en-GB" is served by "en".
struct LanguageCapability {
    std::string tag;
    uint32_t    flags = 0;
};

// What the public Voice Profile capability query (`synth_voice_profile_capabilities_t`,
// include/synthesize.h) reports about a Loaded Model, filled by each family's
// own `shared_info` -- today only OmniVoice's, from its ProfileContract
// (src/arch/omnivoice/weights.h). A family with no Voice Profile support at
// all leaves this at its all-zero default, which is exactly the
// "SYNTH_REQUIREMENT_UNSUPPORTED, zero everything else" shape
// docs/c-interface.md requires for an unsupported source.
//
// `compatibility_id` is decoded here (decode_profile_compatibility_id below)
// but is exposed through the public query ONLY once `source_flags` also
// carries SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE (Task 16) --
// docs/c-interface.md: "When Serialized Profile is unsupported ... the 32 ID
// bytes are zero" -- so a Task 14 model carries real bytes here without yet
// being allowed to hand them out.
struct VoiceProfileInfo {
    uint32_t            source_flags                 = 0;  // bitmask of SYNTH_PROFILE_SOURCE_*
    synth_requirement_t reference_transcript         = SYNTH_REQUIREMENT_UNSUPPORTED;
    synth_requirement_t reference_language           = SYNTH_REQUIREMENT_UNSUPPORTED;
    synth_requirement_t description_language         = SYNTH_REQUIREMENT_UNSUPPORTED;
    uint32_t            reference_target_sample_rate = 0;
    uint32_t            reference_target_channels    = 0;
    uint64_t            min_frames_per_clip          = 0;
    uint64_t            max_frames_per_clip          = 0;
    uint64_t            max_total_frames             = 0;
    uint32_t            max_reference_count          = 0;
    uint8_t             compatibility_id[32]         = {};  // decoded from the package hex
};

// Decodes a 64-character lowercase-hex Profile Compatibility ID -- the shape
// a package's GGUF metadata stores it in (`ProfileContract::compatibility_id_hex`,
// validated at load time by weights.cpp's own `is_sha256_hex`) -- into the 32
// raw bytes `synth_voice_profile_capabilities_t::profile_compatibility_id`
// exposes. Returns false and leaves `bytes` untouched for any input that is
// not exactly 64 characters of `[0-9a-f]`, so a caller that skipped
// validation fails safely rather than reading whatever partial decode it
// stopped at.
inline bool decode_profile_compatibility_id(const std::string & hex, uint8_t (&bytes)[32]) {
    if (hex.size() != 64) {
        return false;
    }
    const auto nibble = [](char character, uint8_t & value) {
        if (character >= '0' && character <= '9') {
            value = static_cast<uint8_t>(character - '0');
            return true;
        }
        if (character >= 'a' && character <= 'f') {
            value = static_cast<uint8_t>(character - 'a' + 10);
            return true;
        }
        return false;
    };
    uint8_t decoded[32];
    for (size_t index = 0; index < 32; ++index) {
        uint8_t high = 0;
        uint8_t low  = 0;
        if (!nibble(hex[index * 2], high) || !nibble(hex[index * 2 + 1], low)) {
            return false;
        }
        decoded[index] = static_cast<uint8_t>((high << 4) | low);
    }
    std::memcpy(bytes, decoded, sizeof(decoded));
    return true;
}

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
    // Every language this model may be asked for. The request validator matches
    // against it and the public enumeration reports it, so a family that leaves
    // it empty accepts no explicit language tag at all.
    std::vector<LanguageCapability>     languages;
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
    // Zero-valued (VoiceProfileInfo's own default) for every family that has
    // not filled it in; only OmniVoice's `shared_info` does, as of Task 14.
    VoiceProfileInfo                    voice_profile;
};

}  // namespace synth
