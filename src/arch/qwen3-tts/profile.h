#pragma once

#include "gguf.h"
#include "synthesize.h"
#include "voice-profile-handle.h"

#include <cstddef>
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

// ---------------------------------------------------------------------------
// Task 8: the v1 Serialized Voice Profile envelope (docs/c-interface.md's "v1
// Serialized Profile GGUF Contract"; this plan's own design doc records the
// format decision) -- little-endian GGUF v3, alignment
// GGUF_DEFAULT_ALIGNMENT, `general.architecture = "synthprofile"`. ONE
// schema, "qwen3-tts-voice-clone" -- the SAME string every already-converted
// Base package's own ProfileContract already declares and weights.cpp's
// read_profile_contract already requires (so a second schema id would need a
// package re-cut) -- covers BOTH of this family's clone modes;
// `synthesize.voice_profile.kind` ("x-vector" | a Plan 3 "icl") picks
// between them, exactly the split omnivoice::profile.h's own header comment
// explains for its two kinds. Plan 2 implements only the "x-vector" kind:
// Plan 3 adds "icl" and whatever ICL-only metadata it needs on top of the 8
// common keys below, without a schema version bump -- discriminating on
// `kind` rather than on `schema_version` is the entire point.
// ---------------------------------------------------------------------------

// Which kind(s) a kPrescanKnownKeys entry is ever emitted for -- the same
// role omnivoice::PrescanKeyScope plays for its own two kinds (see that
// enum's own header comment for why a flat key-set check alone misses a key
// MOVED between kinds, or a per-kind count left stale after a key is added
// to one kind and the whitelist but not to its own count constant). Plan 2
// emits only kCommon and kXVectorOnly; a Plan 3 "icl" kind adds its own
// kIclOnly member here when it lands, without touching either existing
// value.
enum class PrescanKeyScope {
    kCommon,
    kXVectorOnly,
};

// The exact, closed set of metadata keys this family's writer
// (set_common_metadata/serialize_x_vector_profile, both in profile.cpp) ever
// emits for the "x-vector" kind: the 8 keys every kind shares, plus
// serialize_x_vector_profile's 2 x-vector-only keys -- 10 entries total.
// profile.cpp's own prescan_buffer (the untrusted-buffer pre-scan guarding
// load_profile_from_memory) uses this SAME table as its positive-validation
// whitelist: a key outside this set, or a known key declared with the wrong
// type/array-ness/count, is refused before gguf_init_from_buffer ever runs.
//
// Declared here in the header -- `inline constexpr` at namespace scope, the
// same pattern omnivoice::profile.h uses its own kPrescanKnownKeys for --
// rather than file-local (anonymous-namespace) to profile.cpp: a
// writer-agreement test (tests/qwen3_tts_profile_test.cpp) drives the REAL
// serialize_x_vector_profile and checks its real output's key set against
// this SAME table, rather than a second hand-transcription of it -- a key
// REMOVED from the writer while left in this table would otherwise be a
// silently too-permissive whitelist that no other fast test would notice.
struct PrescanKeySpec {
    const char *    key;
    gguf_type       type;
    bool            is_array;
    uint64_t        count;
    PrescanKeyScope scope;
};

inline constexpr PrescanKeySpec kPrescanKnownKeys[] = {
    // set_common_metadata (8 keys, every kind).
    { "general.architecture",                      GGUF_TYPE_STRING,  false, 0,  PrescanKeyScope::kCommon      },
    { "synthesize.voice_profile.format_version",   GGUF_TYPE_UINT32,  false, 0,  PrescanKeyScope::kCommon      },
    { "synthesize.voice_profile.model_family",     GGUF_TYPE_STRING,  false, 0,  PrescanKeyScope::kCommon      },
    { "synthesize.voice_profile.schema",           GGUF_TYPE_STRING,  false, 0,  PrescanKeyScope::kCommon      },
    { "synthesize.voice_profile.schema_version",   GGUF_TYPE_UINT32,  false, 0,  PrescanKeyScope::kCommon      },
    { "synthesize.voice_profile.compatibility_id", GGUF_TYPE_UINT8,   true,  32, PrescanKeyScope::kCommon      },
    { "synthesize.voice_profile.content_sha256",   GGUF_TYPE_UINT8,   true,  32, PrescanKeyScope::kCommon      },
    { "synthesize.voice_profile.kind",             GGUF_TYPE_STRING,  false, 0,  PrescanKeyScope::kCommon      },
    // serialize_x_vector_profile (2 more keys, "x-vector" only).
    { "synthesize.voice_profile.ref_rms",          GGUF_TYPE_FLOAT32, false, 0,  PrescanKeyScope::kXVectorOnly },
    { "synthesize.voice_profile.language_tag",     GGUF_TYPE_STRING,  false, 0,  PrescanKeyScope::kXVectorOnly },
};
inline constexpr size_t kPrescanKnownKeyCount = sizeof(kPrescanKnownKeys) / sizeof(kPrescanKnownKeys[0]);

// n_kv is exactly this value for every envelope Plan 2 ever writes -- not a
// generous ceiling, an exact enumeration, since the "x-vector" kind is the
// only one this writer produces yet. A Plan 3 "icl" kind adds its own
// kPrescanKvCountIcl alongside this constant, the same way
// omnivoice::kPrescanKvCountDesign/kPrescanKvCountClone sit side by side for
// omnivoice's own two kinds; prescan_buffer's own `n_kv` gate then widens to
// accept either value, exactly as omnivoice's already does.
inline constexpr int64_t kPrescanKvCountXVector = 10;

// Serializes `profile` into a fresh v1 envelope. `compatibility_id` is the
// Loaded Model's own 32-byte Profile Compatibility ID (already decoded from
// the package's hex metadata by model-handle.h's
// decode_profile_compatibility_id / model-info.h's VoiceProfileInfo) and is
// copied into the envelope verbatim.
//
// Deterministic: the same `profile` and `compatibility_id` always produce
// byte-identical output -- the metadata key insertion order below is fixed,
// and nothing here reads the wall clock, an address, or any other
// environment-derived value.
//
// Returns SYNTH_ERR_INVALID_ARG if `profile.x_vector` is empty: a
// zero-length tensor is not a shape this family's own reader (or any real
// Profile create_x_vector_profile ever builds) could load back, so writing
// one out is this writer's own defect to refuse rather than a caller mistake
// to diagnose further.
synth_status_t serialize_x_vector_profile(const XVectorProfile & profile,
                                          const uint8_t (&compatibility_id)[32],
                                          std::vector<uint8_t> & out_bytes);

// Parses a v1 envelope out of untrusted `data`/`data_size` against the
// caller's own `compatibility_id` and declared x-vector width (`enc_dim`,
// this package's `SpeakerEncoderParams::enc_dim` -- the exact width, not a
// ceiling, since a Profile of the wrong width builds a wrong-shaped prompt).
//
// Unlike omnivoice::load_profile_from_memory, this takes `enc_dim` by value
// rather than a `Model &`: every check this function needs -- the tensor's
// own declared element count -- comes from that one scalar, and this
// family's XVectorProfile carries no discrete token stream to range-check
// against a vocabulary/mask id, no transcript to re-tokenize through a text
// frontend, and no declared-language validation of its own (`language_tag`
// is stored verbatim here, exactly as XVectorProfile's own header comment
// says create_x_vector_profile itself does -- validating it is the caller's
// job, the same way profile.cpp's own header comment defers language
// validation for the in-memory Profile). None of the reasons
// omnivoice::load_profile_from_memory needs a live `Model &` apply here, so
// this takes the one scalar it actually needs, the same substitution
// create_x_vector_profile's own header comment already makes (and
// justifies) for `HParams`/`SpeakerEncoderWeights` in place of a `Model &`.
//
// Status mapping (docs/c-interface.md: "Incompatible Model data returns
// SYNTH_ERR_UNSUPPORTED_VOICE; malformed, truncated, or corrupt data returns
// SYNTH_ERR_INVALID_ARG"):
//   * structurally broken bytes (gguf_init_from_buffer itself refuses them,
//     a required key is missing/wrongly typed/duplicated, the tensor shape
//     or byte range does not fit `data`, the x-vector's declared length does
//     not equal `enc_dim`, or the content digest does not match) ->
//     SYNTH_ERR_INVALID_ARG;
//   * a structurally well-formed envelope for a DIFFERENT model_family,
//     schema, schema_version, or exact compatibility_id -> SYNTH_ERR_UNSUPPORTED_VOICE
//     (this loader understands the envelope, just not for this Model);
//   * an unrecognized `kind` value (Plan 2: anything other than "x-vector",
//     including a Plan 3 "icl" envelope this build does not implement yet)
//     -> SYNTH_ERR_INVALID_ARG: unlike the three mismatches above, `kind` is
//     this ONE schema's own internal tag, not a different schema/version/
//     model -- a value this build does not recognize means the payload
//     structure the rest of the bytes describe cannot be interpreted at
//     all, which this project treats as malformed rather than merely
//     unsupported. This is also the exact rejection Plan 3 turns into a
//     success once it adds the "icl" arm.
//
// Size arithmetic runs BEFORE any allocation sized from the untrusted bytes:
// the declared tensor element count is checked against `enc_dim` using only
// cheap GGUF metadata/tensor-info getters (no tensor payload is read) before
// `std::vector<float>` ever sizes itself from it.
//
// `out_diagnostic_code`/`out_diagnostic_message` are left null: this
// function names no refusal of its own that needs one (unlike
// omnivoice::load_profile_from_memory's re-tokenization failure, which has
// no counterpart here -- XVectorProfile carries no transcript to
// re-tokenize). Both out-parameters exist only so this signature matches the
// same shape Task 9's dispatch already expects from the family loader.
synth_status_t load_profile_from_memory(uint32_t        enc_dim,
                                        const uint8_t * data,
                                        size_t          data_size,
                                        const uint8_t (&compatibility_id)[32],
                                        synth::ProfileFamilyTag &     out_family_tag,
                                        std::shared_ptr<const void> & out_payload,
                                        const char *&                 out_diagnostic_code,
                                        const char *&                 out_diagnostic_message);

}  // namespace synth::qwen3tts
