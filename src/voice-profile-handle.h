#pragma once

#include "synthesize.h"

#include <memory>

namespace synth {

// Which family produced a Voice Profile's payload, and which of that
// family's own payload shapes it is -- not just which family, because a
// single family may grow more than one profile-preparation source over more
// than one payload type (OmniVoice's Reference Audio ClonePrompt as of Task
// 14; its Description Text DesignInstruct from Plan 3's Task 15 on) and the
// two must never be read back through each other's pointer cast.
enum class ProfileFamilyTag : uint32_t {
    None = 0,
    OmnivoiceClone,
    // Description Text ("voice design"): wraps a
    // synth::omnivoice::DesignInstruct payload (profile.h, Task 15).
    OmnivoiceDesign,
    // Wraps a synth::qwen3tts::XVectorProfile payload
    // (arch/qwen3-tts/profile.h). One tag covers both of this family's clone
    // modes: the payload's own CloneMode discriminates, so Plan 3's ICL
    // Profiles reuse this tag rather than adding a second one that every
    // switch would have to learn.
    Qwen3TtsClone,
};

}  // namespace synth

// The private side of the opaque `synth_voice_profile_t` handle: the Loaded
// Model that produced it -- so a profile presented to a different model's
// synthesis request can be refused rather than silently misread -- and a
// type-erased pointer to the owning family's own prepared payload, per the
// core/family firewall: core never names `synth::omnivoice::ClonePrompt`
// itself, only `family_tag` and a `void` pointer it hands back to the family
// that produced it.
//
// Shared between src/voice-profile.cpp (construction, on every successful
// profile-preparation call) and src/synthesize.cpp (reads `family_tag` and
// `payload` to recover the family's own type before a clone synthesis call)
// -- the same reason `struct synth_model` itself moved out of synthesize.cpp
// into model-handle.h.
struct synth_voice_profile {
    const synth_model_t *       model      = nullptr;
    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
};
