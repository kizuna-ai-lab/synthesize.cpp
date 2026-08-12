#pragma once

#include "synthesize.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace synth::qwen3tts {

struct HParams;
struct SpeakerEncoderWeights;

// Which of the two upstream clone modes this Profile fixes. D4 (this plan's
// own ruling, restated in the brief this file implements): the mode is
// decided at preparation, not at synthesis, which is what gives a Serialized
// Profile one unambiguous meaning -- a caller who asked for one mode can
// never be handed back a Profile that quietly means the other. Plan 2
// produces only XVector; Plan 3 adds Icl and the fields it needs, without a
// schema version bump, because the envelope discriminates on a `kind` key
// rather than on its schema version (Task 8).
enum class CloneMode : uint32_t {
    XVector = 0,
    // Icl -- Plan 3.
};

// The prepared clone payload a Reference Audio Voice Profile carries once
// create_x_vector_profile below has run the whole x-vector encode chain
// (Model::prepare_x_vector / encode_speaker_reference) over an
// already-normalized, already length-checked reference clip. Immutable after
// construction, the same convention omnivoice::ClonePrompt uses -- a Loaded
// Model's opaque `synth_voice_profile_t` wraps one of these behind a
// type-erased `std::shared_ptr<void>` plus `ProfileFamilyTag::Qwen3TtsClone`
// (voice-profile-handle.h).
struct XVectorProfile {
    CloneMode          mode = CloneMode::XVector;
    std::vector<float> x_vector;  // enc_dim floats; substitutes for the prompt's speaker embedding
    float              ref_rms = 0.0f;
    // The reference clip's own optional declared language (not the target
    // synthesis language, which the core resolves separately from the
    // request). May be empty. Stored verbatim; Plan 2 does nothing with it
    // yet (there is no transcript to pair it with), but the field exists now
    // so Plan 3's Icl payload does not need a shape change to add it.
    std::string        language_tag;
};

// Prepares an x-vector Voice Profile from one already-normalized (this
// package's declared Reference Audio target format -- 24 kHz mono),
// already length-checked Reference Audio clip.
//
// Unlike omnivoice::create_clone_prompt, which takes the family's `Model &`
// directly, this takes the two pieces of it Model::prepare_x_vector itself
// reads (`HParams`, `SpeakerEncoderWeights`) rather than a `Model` reference.
// `synth::qwen3tts::Model` can only be constructed through `Model::load`/
// `load_cpu`, both of which require a real GGUF on disk; this family has no
// synthetic-package test harness yet (unlike OmniVoice's
// tests/omnivoice_synthetic_package.h), and building one is disproportionate
// to a Voice Profile preparation task. Taking the two structs directly keeps
// this function testable against synthetic HParams and in-memory LCG weights
// -- the same shape tests/qwen3_tts_speaker_encoder_host_test.cpp (Task 5)
// already relies on -- while still reproducing Model::prepare_x_vector's own
// two behaviors exactly: the `has_speaker_encoder` gate below, and the
// encode_speaker_reference call it wraps. Task 9's dispatch arm, which does
// hold a real `Model &`, is free to add a small accessor (or keep calling
// `model.prepare_x_vector` itself and pass the pieces through) once it wires
// the public seam -- that plumbing choice belongs to the task that opens it.
//
// `transcript` non-empty (including a whitespace-only string) is REJECTED in
// Plan 2 with "voice_profile.transcript_unsupported": this rung implements
// the x-vector mode only, and D4 fixes the mode at preparation, so accepting
// a transcript and silently building the x-vector Profile anyway would hand
// back a weaker clone than the caller asked for -- the same capability lie
// the erratum removed from the source flags. Checked FIRST, before the
// (expensive) encode chain ever runs. Plan 3 is the change that turns this
// rejection into the mode selector.
//
// Order: reject a non-empty transcript; reject a package with no speaker
// encoder (`SYNTH_ERR_UNSUPPORTED_VOICE`, mirroring
// Model::prepare_x_vector's own CustomVoice guard); run
// encode_speaker_reference, whose own refusals (including the
// "voice_profile.reference_silent" digitally-silent-reference rejection)
// propagate unchanged; build the payload.
//
// `threads` follows encode_speaker_reference's own convention: 0 selects
// default_synthesis_threads().
//
// On any non-OK return `output` is left untouched (reset to null).
// `out_diagnostic_code`/`out_diagnostic_message` are set to non-null static
// strings only for the two refusals this function itself names
// ("voice_profile.transcript_unsupported" and whatever
// encode_speaker_reference itself sets); otherwise both stay null and the
// returned status is specific enough on its own.
synth_status_t create_x_vector_profile(const HParams &                         hparams,
                                       const SpeakerEncoderWeights &           speaker_encoder,
                                       const std::vector<float> &              pcm_24k,
                                       const std::string &                     transcript,
                                       const std::string &                     language_tag,
                                       int                                     threads,
                                       std::shared_ptr<const XVectorProfile> & output,
                                       const char *&                           out_diagnostic_code,
                                       const char *&                           out_diagnostic_message);

}  // namespace synth::qwen3tts
