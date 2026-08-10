#pragma once

#include "gguf.h"
#include "synthesize.h"
#include "voice-profile-handle.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace synth::omnivoice {

class Model;

// The maximum length, in bytes, of a ClonePrompt's canonical
// `transcript_text` this family will ever CREATE or ACCEPT. Tied by name to
// profile.cpp's own `kPrescanMaxStringLength` -- the Serialized Profile
// loader's positive-validation prescan whitelist refuses any GGUF string
// value longer than that -- so the two can never drift apart: a transcript
// create_clone_prompt accepts here is guaranteed to round-trip through
// serialize_clone_prompt/load_profile_from_memory without ever being
// rejected by our OWN writer's output (reviewer FINDING 2: before this cap
// existed, a >1 MiB transcript sailed through creation and serialization,
// then hit the prescan's string-length ceiling on load with a bare
// INVALID_ARG -- our own writer's output, rejected by our own reader). A
// Voice Reference clip is capped at this family's own declared
// `max_reference_frames_per_clip` (20 s of audio for the shipped package),
// so no real transcript approaches this bound; it exists to give a
// deterministic, named refusal to a pathological caller instead of a
// silently-broken round trip.
constexpr uint64_t kMaxClonePromptTranscriptLength = 1u << 20;  // 1 MiB

// The prepared clone prompt a Reference Audio Voice Profile carries once
// voice-profile.cpp's create_from_reference handler has validated its
// inputs, normalized the caller's clip, and run the whole encode chain
// (Model::encode_reference) over it. Immutable after construction; a Loaded
// Model's opaque `synth_voice_profile_t` wraps one of these (or, from Task
// 15's DesignInstruct on, one of that struct instead) behind a type-erased
// `std::shared_ptr<void>` plus a family/payload tag
// (voice-profile-handle.h's `ProfileFamilyTag`) -- the core/family firewall's
// rule that family internals never cross the public seam, restated here for
// the payload rather than for a graph.
//
// The unit Serialized Profiles round-trip (Task 16): `reference_tokens`
// becomes the payload's tensor, `transcript_text`/`ref_rms`/`language_tag`
// its metadata, and `transcript_ids` is NOT serialized -- it is re-tokenized
// on load (determinism guaranteed by the frozen BPE vocabulary), so it costs
// nothing to omit and everything to trust across a package upgrade.
struct ClonePrompt {
    // 8 x T_ref, codebook-major (level l, frame t at l * T_ref + t) --
    // Model::encode_reference's own ReferenceEncoding::tokens layout exactly,
    // and SynthesisRequest::reference_tokens' own layout, so this is threaded
    // through Model::synthesize with no reshaping.
    std::vector<int32_t> reference_tokens;
    // transcript_text tokenized through the package's own frontend, UNWRAPPED
    // (no style markers, no combine_text join) -- a cache Task 16's
    // serialized payload can reuse without touching the frontend again.
    // Model::synthesize retokenizes the WRAPPED prompt itself
    // (assemble_prompt_ids, which calls combine_text(transcript_text, text)
    // internally) and never reads this field; it exists for round-tripping,
    // not for synthesis.
    std::vector<int32_t> transcript_ids;
    // Canonical, post-punctuation transcript (frontend-host.h's
    // add_punctuation) -- conditioning text, not source material:
    // combine_text() joins it with the request's own target text at
    // synthesis time, so THIS is what is canonical and transcript_ids above
    // is a derived cache, not the other way around.
    std::string          transcript_text;
    // The reference's loudness, measured on the FULL input before the
    // quiet-reference boost or the hop-clip touch it
    // (ReferenceEncoding::ref_rms's own contract). Always strictly positive
    // for a ClonePrompt that exists at all: create_clone_prompt below
    // refuses a digitally silent reference (ref_rms == 0.0f) before one can
    // ever be built. Drives Task 14's volume arms
    // (codec-host.h's apply_reference_volume).
    float                ref_rms = 0.0f;
    // The reference clip's own optional declared language (not the target
    // synthesis language, which the core resolves separately from the
    // request). May be empty.
    std::string          language_tag;
};

// Builds a ClonePrompt from an already-normalized (the package's declared
// Reference Audio target format -- 24 kHz mono for this family), already
// length-checked Reference Audio clip and its transcript.
//
// `transcript` is the caller's raw, unpunctuated text -- voice-profile.cpp's
// dispatcher has already checked it is present and non-empty per this
// package's `reference_transcript == SYNTH_REQUIREMENT_REQUIRED` capability,
// but has not touched its content otherwise. `language_tag` is the
// reference clip's own optional declared language, stored verbatim.
// `threads` follows Model::encode_reference's own convention: 0 selects
// default_synthesis_threads().
//
// Runs, in order: Model::encode_reference (the whole cloning encode chain:
// hop-clip, ref_rms measurement, quiet-reference boost, resample, the
// semantic and acoustic branches, RVQ encode); the silent-reference rejection
// jiangzhuo's 2026-08-01 ruling requires
// (docs/porting/families/omnivoice.md's "Silent-Reference Rejection" note);
// then the punctuation rule (frontend-host.h's add_punctuation) and a plain
// (unwrapped) tokenization of the result for the ClonePrompt::transcript_ids
// cache.
//
// Returns whatever encode_reference itself returns on its own refusals.
// Returns SYNTH_ERR_INVALID_ARG, with `out_diagnostic_code`/
// `out_diagnostic_message` set to non-null static strings, for the two
// refusals this function itself names: a digitally silent reference
// (`"voice_profile.reference_silent"`) and a transcript the package's own
// frontend cannot tokenize (`"voice_profile.transcript_unencodable"`) --
// voice-profile.cpp's dispatcher passes those straight to its diagnostic
// sink. For every other non-OK return, both out-parameters are left null:
// the underlying status is specific enough on its own.
//
// `output` is left untouched on any non-OK return.
synth_status_t create_clone_prompt(Model &                              model,
                                   const std::vector<float> &           pcm_24k,
                                   const std::string &                  transcript,
                                   const std::string &                  language_tag,
                                   int                                  threads,
                                   std::shared_ptr<const ClonePrompt> & output,
                                   const char *&                        out_diagnostic_code,
                                   const char *&                        out_diagnostic_message);

// ClonePrompt's sibling for a Description Text ("voice design") Voice
// Profile (Task 15): the validated, canonical instruct string
// resolve_instruct() below produces. Unlike ClonePrompt, this payload needs
// no Model at all -- vocabulary resolution is pure string processing, never
// a tensor -- so it carries nothing else.
struct DesignInstruct {
    // EN- or ZH-unified, joined with ", " or "，" -- upstream's own
    // _resolve_instruct() return value shape
    // (omnivoice/utils/voice_design.py + omnivoice/models/omnivoice.py's
    // own _resolve_instruct, lines 1492-1621, both at the pinned revision),
    // reproduced by resolve_instruct() below. Flows into Model::synthesize's
    // assemble_prompt_ids call as the instruct slot verbatim. May be empty:
    // a description that validates to zero attribute items (e.g. one made
    // only of separators) resolves the same way upstream's own empty-result
    // path does -- see resolve_instruct's own comment.
    std::string instruct;
};

// Validates and canonicalizes a Description Text ("voice design") instruct
// string against upstream's CLOSED attribute vocabulary --
// omnivoice/utils/voice_design.py:31-97 and omnivoice/models/omnivoice.py's
// own _resolve_instruct (1492-1621), both at the pinned revision.
//
// This is a closed vocabulary, not a free-form/attribute hybrid: transcription
// (see profile.cpp's own citations) confirmed upstream raises ValueError on
// any comma-separated item that is not one of the fixed gender/age/pitch/
// style/accent/dialect terms below, so this port rejects a non-member the
// same way, by name, rather than accepting it as free text. The design
// spec's "attribute / free-form instruct vocabulary" phrasing is resolved
// against the pinned source in the vocabulary's favor.
//
// `description` is the caller's raw, non-empty string -- voice-profile.cpp's
// dispatcher has already checked presence per docs/c-interface.md
// ("description is required, non-empty"), but has not touched its content
// otherwise.
//
// `use_zh` is the language-unification baseline BEFORE this function's own
// dialect/accent overrides (a dialect item present forces Chinese, an accent
// item present forces English, exactly as upstream's own override does).
// Upstream computes its OWN baseline from the target TEXT being synthesized
// (omnivoice.py:1068's `use_zh = bool(text_list[i] and _ZH_RE.search(text_list[i]))`),
// which does not exist yet at Voice Profile creation time -- this port's
// caller (voice-profile.cpp) substitutes the request's resolved
// `description_language` instead, a deliberate, documented divergence
// forced by the public Interface's own shape (Description Text profiles
// are prepared independently of any later synthesis text).
//
// Returns SYNTH_ERR_INVALID_ARG, with `out_diagnostic_code` set to one of
// three non-null static strings and `out_diagnostic_message` filled with a
// diagnostic naming the offending item(s) -- for the three rejections
// _resolve_instruct itself raises on:
//   * an item outside the closed vocabulary ("voice_profile.instruct_unknown_item"),
//     named verbatim as the caller wrote it (no difflib "did you mean"
//     suggestion: a diagnostic, not a search engine);
//   * two or more items from the same mutually-exclusive category
//     ("voice_profile.instruct_category_conflict"), naming every conflicting
//     item;
//   * a Chinese dialect item combined with an English accent item in the
//     same instruct ("voice_profile.instruct_dialect_accent_mix").
// Unlike create_clone_prompt above, the message is an owned `std::string`
// rather than a static `const char *`: two of these three diagnostics name
// caller-controlled content, which a static string cannot hold.
//
// `output` is left untouched (reset to null) on any non-OK return.
synth_status_t resolve_instruct(const std::string &                     description,
                                bool                                    use_zh,
                                std::shared_ptr<const DesignInstruct> & output,
                                const char *&                           out_diagnostic_code,
                                std::string &                           out_diagnostic_message);

// ---------------------------------------------------------------------------
// Task 16: the v1 Serialized Voice Profile envelope (ADR 0008,
// docs/c-interface.md's "v1 Serialized Profile GGUF Contract" and
// docs/c-interface.md:654-674) -- little-endian GGUF v3, alignment 32,
// `general.architecture = "synthprofile"`. ONE family schema
// ("omnivoice-clone-prompt", the same string the package's own
// ProfileContract already declares) covers BOTH of this family's payload
// shapes; `synthesize.voice_profile.kind` ("clone-prompt" | "design-instruct")
// picks between them. A second schema id would be truthful too, but would
// need a package re-cut to declare (every already-shipped package's
// `synthesize.profile.schema` is frozen at "omnivoice-clone-prompt"); one
// schema with an internal kind tag is available without one.
// ---------------------------------------------------------------------------

// Which kind(s) a kPrescanKnownKeys entry is ever emitted for --
// `kCommon` for the 8 keys set_common_metadata writes into every envelope,
// `kCloneOnly`/`kDesignOnly` for the keys only serialize_clone_prompt/
// serialize_design_instruct add on top. Fix-round-1 addition: without a
// per-entry scope, a test could only check the FLAT 12-key set, which a key
// MOVED between the two writers (say `language_tag` from clone to design)
// leaves untouched -- the union of what both kinds emit is unchanged, so a
// flat-set check alone cannot see the move. With `scope` on each entry, the
// expected per-kind key set is a direct filter over this SAME table (no
// second, position- or count-based reconstruction of "which keys belong to
// which kind" -- filtering by a field this table already carries is the
// smallest addition that makes the per-kind question directly answerable).
enum class PrescanKeyScope {
    kCommon,
    kCloneOnly,
    kDesignOnly,
};

// The exact, closed set of metadata keys this family's writer
// (set_common_metadata/serialize_clone_prompt/serialize_design_instruct, all
// in profile.cpp) ever emits: the 8 keys every kind shares (including
// "general.alignment"'s absence -- this writer never sets that key, so it is
// simply not a member here), plus serialize_clone_prompt's 3 ClonePrompt-only
// keys, plus serialize_design_instruct's 1 DesignInstruct-only key -- 12
// entries total. profile.cpp's own prescan_buffer (the untrusted-buffer
// pre-scan guarding load_profile_from_memory; see that function's own header
// comment for why it exists) uses this SAME table as its positive-validation
// whitelist: a key outside this set, or a known key declared with the wrong
// type/array-ness/count, is refused before gguf_init_from_buffer ever runs.
// prescan_buffer itself never reads `scope` (which SPECIFIC keys are
// required for a given `kind` is load_profile_from_memory's own job,
// unchanged by this field) -- `scope` exists purely so
// tests/omnivoice_serialize_writer_agreement_test.cpp can compute an exact
// per-kind expected key set from this table alone.
//
// Declared here in the header -- `inline constexpr` at namespace scope, the
// same pattern src/unicode-ranges.h already uses for its own cross-TU
// constant tables -- rather than file-local (anonymous-namespace) to
// profile.cpp, where the rest of the pre-scan machinery still lives, for
// exactly one reason: tests/omnivoice_serialize_writer_agreement_test.cpp
// drives the REAL serialize_clone_prompt/serialize_design_instruct and checks
// their real output's key set against this SAME table, rather than yet
// another hand-transcription of it -- the carry-over item this test closes:
// a key REMOVED from the writer while left in this table is a silently
// too-permissive whitelist that no other fast test would ever notice.
// CLAUDE.md's "family internals are private; tests may reach them" rule,
// applied to this one table.
//
// `is_array`/`count` are only meaningful together; a scalar entry's `count`
// is ignored. The two array-typed entries (compatibility_id, content_sha256)
// are spelled as raw literals here, matching how the other ten entries below
// already are -- profile.cpp's own kKeyCompatibilityId/kKeyContentSha256
// constants exist for the call sites that build an actual value FROM them
// (read_u8_32_array, find_u8_32_value_offset), not for this table, which
// never referenced the other ten keys' own literals by a shared constant
// either.
struct PrescanKeySpec {
    const char *    key;
    gguf_type       type;
    bool            is_array;
    uint64_t        count;
    PrescanKeyScope scope;
};

inline constexpr PrescanKeySpec kPrescanKnownKeys[] = {
    // set_common_metadata (8 keys, every kind).
    { "general.architecture",                      GGUF_TYPE_STRING,  false, 0,  PrescanKeyScope::kCommon     },
    { "synthesize.voice_profile.format_version",   GGUF_TYPE_UINT32,  false, 0,  PrescanKeyScope::kCommon     },
    { "synthesize.voice_profile.model_family",     GGUF_TYPE_STRING,  false, 0,  PrescanKeyScope::kCommon     },
    { "synthesize.voice_profile.schema",           GGUF_TYPE_STRING,  false, 0,  PrescanKeyScope::kCommon     },
    { "synthesize.voice_profile.schema_version",   GGUF_TYPE_UINT32,  false, 0,  PrescanKeyScope::kCommon     },
    { "synthesize.voice_profile.compatibility_id", GGUF_TYPE_UINT8,   true,  32, PrescanKeyScope::kCommon     },
    { "synthesize.voice_profile.content_sha256",   GGUF_TYPE_UINT8,   true,  32, PrescanKeyScope::kCommon     },
    { "synthesize.voice_profile.kind",             GGUF_TYPE_STRING,  false, 0,  PrescanKeyScope::kCommon     },
    // serialize_clone_prompt (3 more keys, "clone-prompt" only).
    { "synthesize.voice_profile.transcript_text",  GGUF_TYPE_STRING,  false, 0,  PrescanKeyScope::kCloneOnly  },
    { "synthesize.voice_profile.ref_rms",          GGUF_TYPE_FLOAT32, false, 0,  PrescanKeyScope::kCloneOnly  },
    { "synthesize.voice_profile.language_tag",     GGUF_TYPE_STRING,  false, 0,  PrescanKeyScope::kCloneOnly  },
    // serialize_design_instruct (1 more key, "design-instruct" only).
    { "synthesize.voice_profile.instruct",         GGUF_TYPE_STRING,  false, 0,  PrescanKeyScope::kDesignOnly },
};
inline constexpr size_t kPrescanKnownKeyCount = sizeof(kPrescanKnownKeys) / sizeof(kPrescanKnownKeys[0]);

// n_kv is exactly one of these two values: 8 common + 1 ("instruct") for
// DesignInstruct, 8 common + 3 (transcript_text/ref_rms/language_tag) for
// ClonePrompt -- not a generous ceiling, an exact enumeration, since this
// writer never produces anything else. Which SPECIFIC keys are required for
// a given `kind` is still load_profile_from_memory's own job afterward
// (GgufMetadata's per-field reads already fail closed on a missing field);
// profile.cpp's own prescan_buffer only bounds the total count against these
// two values and rejects any key outside kPrescanKnownKeys above.
//
// Declared here (`inline constexpr`, same reasoning as kPrescanKnownKeys
// above) rather than file-local to profile.cpp: fix-round-1 addition --
// tests/omnivoice_serialize_writer_agreement_test.cpp also pins the REAL
// writer's per-kind key COUNT against these SAME two constants, independent
// of (and a cheaper, coarser complement to) the per-kind key SET check
// `scope` above enables. This matters because the two checks catch DIFFERENT
// drifts: a key added to a writer AND correctly added to kPrescanKnownKeys
// with the correct `scope`, but whose matching count constant here is not
// bumped, passes the per-kind SET check (scope-filtered kPrescanKnownKeys
// now genuinely matches what the writer emits) while still being exactly the
// bug prescan_buffer's own n_kv gate (profile.cpp) would reject every real
// envelope of that kind over -- only a dedicated count check catches it.
inline constexpr int64_t kPrescanKvCountDesign = 9;
inline constexpr int64_t kPrescanKvCountClone  = 11;

// Serializes `prompt` into a fresh v1 envelope. `compatibility_id` is the
// Loaded Model's own 32-byte Profile Compatibility ID (already decoded from
// the package's hex metadata by model-handle.h's
// decode_profile_compatibility_id / model-info.h's VoiceProfileInfo) and is
// copied into the envelope verbatim.
//
// `prompt.transcript_ids` is NOT written: it is a re-tokenization cache
// (profile.h's own ClonePrompt comment), and load_profile_from_memory below
// rebuilds it from `prompt.transcript_text` through the Loaded Model's own
// frontend, which this project's frozen BPE vocabulary guarantees is
// deterministic across the round trip.
//
// Deterministic: the same `prompt` and `compatibility_id` always produce
// byte-identical output -- the metadata key insertion order below is fixed,
// and nothing here reads the wall clock, an address, or any other
// environment-derived value. The round-trip and byte-determinism tests
// (tests/omnivoice_profile_test.cpp, tests/omnivoice_serialize_test.cpp)
// depend on this.
//
// Returns SYNTH_ERR_INVALID_ARG if `prompt.reference_tokens` is not a
// non-empty multiple of 8 (profile.h's own "8 x T_ref" contract) -- a
// ClonePrompt this project itself ever builds should never violate that, so
// reaching this path is itself evidence of a defect upstream of this call,
// not a caller mistake to diagnose further.
synth_status_t serialize_clone_prompt(const ClonePrompt & prompt,
                                      const uint8_t (&compatibility_id)[32],
                                      std::vector<uint8_t> & out_bytes);

// DesignInstruct's sibling: no tensor, one `synthesize.voice_profile.instruct`
// metadata string.
synth_status_t serialize_design_instruct(const DesignInstruct & instruct,
                                         const uint8_t (&compatibility_id)[32],
                                         std::vector<uint8_t> & out_bytes);

// Parses a v1 envelope out of untrusted `data`/`data_size` against `model`'s
// own compatibility id and Reference Audio token budget (`max_total_frames`,
// this package's `max_total_frames` from ProfileContract -- the ceiling on
// `reference_tokens.size() / 8`), and reconstructs whichever payload the
// envelope's own `kind` metadata names.
//
// Status mapping (docs/c-interface.md: "Incompatible Model data returns
// SYNTH_ERR_UNSUPPORTED_VOICE; malformed, truncated, or corrupt data returns
// SYNTH_ERR_INVALID_ARG"):
//   * structurally broken bytes (gguf_init_from_buffer itself refuses them,
//     a required key is missing/wrongly typed/duplicated, the tensor shape
//     or byte range does not fit `data`, a token is outside
//     [0, mask_id) \ {mask_id}, or the content digest does not match) ->
//     SYNTH_ERR_INVALID_ARG;
//   * a structurally well-formed envelope for a DIFFERENT model_family,
//     schema, schema_version, or exact compatibility_id -> SYNTH_ERR_UNSUPPORTED_VOICE
//     (this loader understands the envelope, just not for this Model);
//   * an unrecognized `kind` value -> SYNTH_ERR_INVALID_ARG: unlike the three
//     mismatches above, `kind` is this ONE schema's own internal tag, not a
//     different schema/version/model -- a value outside its two-member enum
//     means the payload structure the rest of the bytes describe cannot be
//     interpreted at all, which is this project's definition of malformed,
//     not merely unsupported.
//
// Size arithmetic runs BEFORE any allocation sized from the untrusted bytes:
// the declared tensor element count is checked against `max_total_frames * 8`
// using only cheap GGUF metadata/tensor-info getters (no tensor payload is
// read) before `std::vector<int32_t>` ever sizes itself from it -- the
// qwen3-3TB lesson (docs referenced in this project's own Task 16 brief)
// applies here too.
//
// On a refusal this function itself names (currently only an unencodable
// re-tokenized transcript), `out_diagnostic_code`/`out_diagnostic_message`
// are set to non-null static strings; otherwise both stay null and the
// returned status is specific enough on its own.
synth_status_t load_profile_from_memory(Model &         model,
                                        const uint8_t * data,
                                        size_t          data_size,
                                        const uint8_t (&compatibility_id)[32],
                                        uint64_t                      max_total_frames,
                                        synth::ProfileFamilyTag &     out_family_tag,
                                        std::shared_ptr<const void> & out_payload,
                                        const char *&                 out_diagnostic_code,
                                        const char *&                 out_diagnostic_message);

}  // namespace synth::omnivoice
