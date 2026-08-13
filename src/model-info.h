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

// Whether a Model Family's own graphs place real work on an EXPLICITLY
// requested execution backend (SYNTH_BACKEND_*, include/synthesize.h). This is
// a per-family fact independent of any one Loaded Model instance -- unlike the
// rest of what a family reports (VoiceProfileInfo below, for instance), which
// only exists once a specific package has finished loading -- because the load
// path (src/synthesize.cpp's synth_model_load) must refuse an unsupported
// backend BEFORE calling that family's loader. Handing a family a device it
// cannot really use would otherwise "load" onto that device's handle while
// every graph quietly ran somewhere else: docs/backends.md's "a backend that
// is present is not a backend that ran."
//
// Never call this for SYNTH_BACKEND_AUTO. AUTO always resolves to a CPU device
// today (docs/backends.md's v1 Selection Policy) and every family places real
// work on CPU, so AUTO is accepted before this function is ever reached rather
// than by adding a case here.
//
// OmniVoice claims CUDA as of Plan 4 Task 11, AFTER the evidence rather than
// before it (accumulated requirement 3): the replay runner
// (tests/omnivoice_replay_real.cpp) calls Model::load directly and bypasses
// this seam entirely, so the sweep measured real placement on the actual GB10
// device while this function still said false. Task 9 gave the codec's
// decode path an accelerator twin (src/arch/omnivoice/model.cpp's
// `Model::Impl::codec_context`), built whenever `BackendPlan::primary()` is
// not the CPU backend; Task 10 taught the runner and the validator to check
// where the nodes actually landed; Task 11 is the sweep itself. Twenty golden
// cases, `--accelerate`: every codec node (8,440 of 8,440 across the suite)
// left the CPU, every generator node (880,032 of 880,032) did not, and the
// seventeen greedy cases' token grids stayed byte-exact against the CPU
// baseline -- docs/backends.md's discrete-outputs rule holding under TF32
// exactly as it must, since the generator is what draws the codes and it
// never moved. Only the codec's own waveform shows CUDA's TF32 arithmetic
// (tests/tolerances/omnivoice.json's `backends.CUDA` cell). Full figures:
// docs/porting/families/omnivoice.md's Execution Backends section.
//
// VITS, Kokoro, and Qwen3-TTS already place real graph work on CUDA
// (docs/backends.md) and keep exactly that.
//
// CPU_ACCEL keeps CPU as the primary backend and only adds optional
// host-memory accelerators (BLAS/AMX), degrading to plain CPU when none is
// registered (docs/backends.md's v1 Selection Policy) -- the same
// "everything on CPU" promise a CPU-only family already keeps -- so every
// family accepts it.
//
// Metal and Vulkan are unavailable for every family today; that is unrelated
// to this per-family split and this function reports it as such (false) for
// all of them.
inline bool family_supports_explicit_backend(ModelFamily family, synth_backend_request_t backend) {
    switch (backend) {
        case SYNTH_BACKEND_CPU:
        case SYNTH_BACKEND_CPU_ACCEL:
            return true;
        case SYNTH_BACKEND_CUDA:
            return true;
        default:
            return false;
    }
}

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
// include/synthesize.h) reports about a Loaded Model. Two families fill this
// today, by deliberately opposite routes:
//   - OmniVoice keeps its family ModelInfo to the raw ProfileContract
//     (src/arch/omnivoice/weights.h) and src/synthesize.cpp's own
//     `shared_info` assembles this struct inline, at the family/core seam.
//   - Qwen3-TTS (its Base variant, Plan 1 Task 9) assembles the whole struct
//     in the family layer instead (src/arch/qwen3-tts/weights.h's
//     fill_voice_profile_capability, called from Model::get_info), and its
//     own `shared_info` just copies the finished result through.
// The second route exists so the assembly logic -- which Voice Profile
// sources a package supports and why -- is a pure function of the family's
// own HParams, testable at the `unit` tier without a loaded Model: neither
// family's real package is small enough for a unit test to depend on
// (docs/testing.md). Pick whichever route suits a new family; both produce
// the same field. A family with no Voice Profile support at all leaves this
// at its all-zero default, which is exactly the "SYNTH_REQUIREMENT_UNSUPPORTED,
// zero everything else" shape docs/c-interface.md requires for an
// unsupported source.
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
    // uint64_t, matching both ends it sits between: the package's own
    // `synthesize.reference.max_reference_count` (a GGUF u64, so a
    // ProfileContract u64) and the public
    // `synth_voice_profile_capabilities_t::max_reference_count`. It was
    // uint32_t until 2026-08-13, which made this the one narrowing hop in an
    // otherwise 64-bit path -- a declared 2^32 arrived here as 0, the value
    // that means "no Reference Audio clip may be used at all". Both families
    // now refuse anything but 1 at load (weights.cpp's
    // read_profile_contract), so nothing reaches this field that a uint32_t
    // could not have held either; the type simply stops being the place a
    // future widening of that rule would have to remember to look.
    uint64_t            max_reference_count          = 0;
    uint8_t             compatibility_id[32]         = {};  // decoded from the package hex
    // The package's own declared Serialized Profile schema identity (the
    // same "synthesize.profile.schema"/"schema_version" ProfileContract pair
    // weights.cpp already validates at load time) -- exposed through
    // `synth_voice_profile_capabilities_t::profile_schema`/
    // `profile_schema_version` once `source_flags` also carries
    // SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE (Task 16). `schema` is owned by
    // this Loaded Model's own `ModelInfo` for its whole lifetime, which is
    // what lets the public query hand back a borrowed pointer that "remains
    // valid until synth_model_free()" (docs/c-interface.md).
    std::string         schema;
    uint32_t            schema_version = 0;
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
    // not filled it in. OmniVoice's `shared_info` fills it as of Task 14 and
    // is the only family that does; Qwen3-TTS routes through its own family
    // layer (fill_voice_profile_capability) but deliberately reports the
    // all-zero shape until it can actually prepare a Profile -- see
    // VoiceProfileInfo's own doc comment above for the two fill routes.
    VoiceProfileInfo                    voice_profile;
};

}  // namespace synth
