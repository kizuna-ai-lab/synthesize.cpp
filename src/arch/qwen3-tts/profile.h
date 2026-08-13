#pragma once

#include "gguf.h"
#include "synthesize.h"
#include "voice-profile-handle.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace synth::qwen3tts {

struct CodecEncoderWeights;
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
    // Transcript-assisted (in-context learning), landed by Plan 3's Task 8 in
    // place of the commented-out placeholder that had sat here since Plan 2.
    // This is the discriminator voice-profile-handle.h's own ProfileFamilyTag
    // comment defers to ("the payload's own CloneMode discriminates") rather
    // than adding a second family tag -- see IclProfile below for the layout
    // rule that makes reading it off a type-erased payload well-defined. The
    // number itself crosses no boundary: a Serialized Profile discriminates on
    // the `synthesize.voice_profile.kind` STRING (Task 9), not on this value.
    Icl     = 1,
};

// The maximum length, in bytes, of an `XVectorProfile::language_tag` this
// family will ever CREATE or ACCEPT. Tied by name to profile.cpp's own
// `kPrescanMaxStringLength` -- the Serialized Profile loader's
// positive-validation prescan whitelist refuses any GGUF string value
// longer than that -- so the two can never drift apart, the same tie
// omnivoice::kMaxClonePromptTranscriptLength holds with its own
// `kPrescanMaxStringLength` (see that constant's own header comment). Before
// this tie existed, a reviewer measured the exact defect that comment
// warns about: a hand-built profile with a 1 MiB + 8 byte `language_tag`
// serialized fine (1,049,376 bytes) and then failed to load with a bare
// INVALID_ARG -- our own writer's output, rejected by our own reader.
// `create_x_vector_profile` and `serialize_x_vector_profile` below both
// refuse a `language_tag` over this bound, so the defect cannot recur from
// either entry point. A real BCP-47 tag (Task 9's own shape validation,
// `valid_bcp47_shape`-style, capping at a few dozen characters) never
// approaches this bound; it exists only to give a pathological caller a
// deterministic, named refusal instead of a silently-broken round trip.
constexpr uint64_t kMaxLanguageTagLength = 1u << 20;  // 1 MiB

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
// `transcript` non-empty (including a whitespace-only string) is REJECTED
// with "voice_profile.transcript_unsupported": this function implements the
// x-vector mode only, and D4 fixes the mode at preparation, so accepting
// a transcript and silently building the x-vector Profile anyway would hand
// back a weaker clone than the caller asked for -- the same capability lie
// the erratum removed from the source flags. Checked FIRST, before the
// (expensive) encode chain ever runs.
//
// THE REFUSAL SURVIVED PLAN 3 rather than becoming the mode selector, which
// is what an earlier revision of this sentence predicted. This function takes
// `HParams`/`SpeakerEncoderWeights` and not a `Model &` precisely so it stays
// unit-testable -- so it cannot tokenize, and therefore cannot produce the
// `reference_text_ids` create_icl_profile below requires. The mode SELECTION
// lives at the one site in the chain that holds a live Model and can reach
// the BPE tables (src/voice-profile.cpp's dispatch, Task 10); this function
// keeps refusing a transcript, and Task 8's own unit test pins that refusal
// directly rather than letting it become an unasserted leftover.
//
// Order: reject a non-empty transcript; reject a `language_tag` over
// `kMaxLanguageTagLength` with "voice_profile.language_tag_too_long" (a
// reviewer-measured Task 8 finding: our own writer must never produce an
// envelope our own reader refuses -- see that constant's own header comment
// -- checked here, cheaply, before the expensive encode chain, the same
// reasoning the transcript check above already uses); reject a package with
// no speaker encoder (`SYNTH_ERR_UNSUPPORTED_VOICE`, mirroring
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
// strings only for the three refusals this function itself names
// ("voice_profile.transcript_unsupported", "voice_profile.language_tag_too_long",
// and whatever encode_speaker_reference itself sets); otherwise both stay
// null and the returned status is specific enough on its own.
synth_status_t create_x_vector_profile(const HParams &                         hparams,
                                       const SpeakerEncoderWeights &           speaker_encoder,
                                       const std::vector<float> &              pcm_24k,
                                       const std::string &                     transcript,
                                       const std::string &                     language_tag,
                                       int                                     threads,
                                       std::shared_ptr<const XVectorProfile> & output,
                                       const char *&                           out_diagnostic_code,
                                       const char *&                           out_diagnostic_message);

// The prepared clone payload a transcript-assisted (ICL) Reference Audio
// Voice Profile carries. D5 (the design's own ruling) tabulates
// exactly three rows for this mode, and this struct is those three rows: the
// `[1024]` speaker embedding, which the table marks `yes` in BOTH columns
// because upstream inserts it regardless of mode; the reference codes; and
// the reference text token ids. An ICL Profile is an x-vector Profile PLUS
// two things, not an alternative to one -- which is why `speaker` below is a
// whole XVectorProfile rather than a re-listing of its fields.
//
// THE FIRST MEMBER IS LOAD-BEARING AND MUST STAY FIRST. One
// ProfileFamilyTag::Qwen3TtsClone covers both of this family's clone modes,
// and voice-profile-handle.h's own comment says why: "the payload's own
// CloneMode discriminates". The consumer that acts on that
// (src/synthesize.cpp:1073-1086) reads the type-erased `shared_ptr<void>`
// payload back as an `XVectorProfile *` and inspects `.mode` BEFORE it knows
// which mode it has -- so for an ICL payload that read has to be well-defined
// rather than merely usually-right. It is: both structs are standard-layout,
// and a standard-layout object is pointer-interconvertible with its first
// non-static data member ([basic.compound]/4), so converting `void *` (which
// holds this object's address) to `const XVectorProfile *` yields `speaker`
// itself ([expr.static.cast]/13). The static_asserts below hold that property
// at compile time; move `speaker` off offset 0, or give either struct a base
// class or a virtual, and the build stops instead of the read going quietly
// wrong at runtime.
//
// WHY THE IDS AND NOT THE STRING, which is a deliberate divergence from the
// sibling family rather than an oversight. omnivoice::ClonePrompt serializes
// its canonical `transcript_text` and RE-TOKENIZES on load
// (arch/omnivoice/profile.h:46-50 and its `transcript_ids` member at :64;
// arch/omnivoice/profile.cpp:1464-1476), on the reasoning that ids "cost
// nothing to omit and everything to trust across a package upgrade". D5 rules
// the other way for this family: the ICL Profile carries reference text token
// ids and DECLARES them recoverable, because byte-level BPE is invertible and
// pretending a token stream is not its transcript would be the dishonest half
// of that trade. Do not "fix" this into consistency with OmniVoice; the two
// families answered the same question differently, on the record, and this
// one's answer is the design's.
struct IclProfile {
    // MUST BE THE FIRST MEMBER -- see this struct's own header comment.
    // `speaker.mode` is CloneMode::Icl for every IclProfile that exists at
    // all; the x-vector, ref_rms and language_tag it carries are the same
    // three quantities an x-vector Profile carries, prepared the same way by
    // the same encode_speaker_reference call.
    XVectorProfile       speaker;
    // `groups * frames` reference codes in codec-encoder-host.h's settled
    // GGML `[16, T]` GROUP-FASTEST order -- `ne[0] = groups` is the
    // quantizer-group index and varies fastest. Stored exactly as
    // CodecEncoding::codes hands it over: THERE IS NO TRANSPOSE, here or
    // anywhere downstream (that header's own comment records why the next
    // reader will want to add one, and why it is wrong).
    std::vector<int32_t> codes;
    // `quantizer_count`: 1 semantic + 15 acoustic at the production geometry.
    // Carried rather than re-derived so `codes` can be indexed without
    // reaching back into HParams.
    uint64_t             groups = 0;
    // `ceil(samples / samples_per_frame)` for the reference clip, from
    // CodecEncoderGeometry -- 13 for the shortest clip this package accepts
    // (24,000 samples) and 101 for a 193,920-sample one.
    uint64_t             frames = 0;
    // The reference transcript as the ids upstream passes as `ref_id`: the
    // reference TURN's own markers are already sliced off (bpe.h's
    // qwen_reference_transcript_ids). Never empty for an IclProfile that
    // exists at all -- create_icl_profile below refuses the half-present
    // state.
    std::vector<int32_t> reference_text_ids;
};

static_assert(std::is_standard_layout<XVectorProfile>::value,
              "XVectorProfile must stay standard-layout: IclProfile's first-member cast depends on it");
static_assert(std::is_standard_layout<IclProfile>::value,
              "IclProfile must stay standard-layout: src/synthesize.cpp reads an ICL payload's CloneMode back "
              "through an XVectorProfile pointer, which is only defined for pointer-interconvertible objects");
static_assert(offsetof(IclProfile, speaker) == 0, "IclProfile::speaker must stay the FIRST member");

// Prepares a transcript-assisted (ICL) Voice Profile from one
// already-normalized (24 kHz mono), already length-checked Reference Audio
// clip, its transcript, and that transcript's already-produced token ids.
//
// Takes the same two structs create_x_vector_profile takes -- `HParams` and
// `SpeakerEncoderWeights` -- plus `CodecEncoderWeights`, and NOT a `Model &`,
// for the reason that function's own header comment above records: a `Model`
// can only be built from a real 2.5 GB GGUF, and this family has no
// synthetic-package harness, so a signature naming one would make every test
// of this function an integration test.
//
// `reference_text_ids` ARRIVES ALREADY TOKENIZED, AND THAT IS THE POINT. The
// BPE tables live on `Model`; tokenizing here would drag that `Model &` in
// through the back door. Task 6's Model::tokenize_reference_transcript
// therefore runs at the dispatch site in src/voice-profile.cpp, which already
// holds a live Model, and passes the ids down. `transcript` is still passed,
// for two judgements that need no tables: this function decides "blank"
// itself, at the same point in its own order that create_x_vector_profile
// makes its own (weaker, "is one present at all") transcript judgement, and it
// refuses the HALF-PRESENT state (a transcript with no ids, or ids with no
// transcript) rather than trusting its caller.
//
// Order, cheapest refusal first, so nothing pays for a graph pass over
// reference audio for a request about to be refused:
//   * a blank `transcript` -- empty OR whitespace-only, which is one state and
//     one refusal, "voice_profile.transcript_blank" (the design's section 9
//     error table: "Transcript present but empty or whitespace-only ->
//     SYNTH_ERR_INVALID_ARG"). An ABSENT transcript names the OTHER clone
//     mode, and D4 fixes the mode at preparation, so this function refuses it
//     rather than quietly preparing the x-vector Profile the caller did not
//     ask for -- the mirror of create_x_vector_profile's own refusal above,
//     and the same capability lie both are written against. Blankness is
//     decided by bpe.h's qwen_transcript_is_blank, the SAME predicate
//     qwen_reference_transcript_ids applies before tokenizing, so this
//     function and the tokenizer can never disagree about which transcripts
//     exist;
//   * a non-blank `transcript` with empty `reference_text_ids`
//     ("voice_profile.reference_text_ids_missing"): all-or-nothing, the same
//     rule Task 11 enforces again at the synthesis seam. Half-present ICL
//     inputs are what produce a plausible-sounding wrong prompt;
//   * a `language_tag` over `kMaxLanguageTagLength`
//     ("voice_profile.language_tag_too_long"): the tie that constant's own
//     header comment documents applies to both kinds or it applies to
//     neither;
//   * a package with no speaker encoder -> SYNTH_ERR_UNSUPPORTED_VOICE. One
//     flag covers both encoders: the catalog resolves `speaker_encoder.*` and
//     `codec.encoder.*` under `has_speaker_encoder` alike, which is why
//     Model::prepare_codec_reference guards on that same flag (model.cpp);
//   * encode_speaker_reference, then encode_codec_reference, whose own
//     refusals (including the shared "voice_profile.reference_silent"
//     digitally-silent-reference rejection, which both raise identically on
//     the same clip) propagate unchanged.
//
// `threads` follows encode_speaker_reference's own convention: 0 selects
// default_synthesis_threads().
//
// On any non-OK return `output` is left untouched (reset to null), and
// `out_diagnostic_code`/`out_diagnostic_message` are set to non-null static
// strings only for the refusals named above.
synth_status_t create_icl_profile(const HParams &                     hparams,
                                  const SpeakerEncoderWeights &       speaker_encoder,
                                  const CodecEncoderWeights &         codec_encoder,
                                  const std::vector<float> &          pcm_24k,
                                  const std::string &                 transcript,
                                  const std::vector<int32_t> &        reference_text_ids,
                                  const std::string &                 language_tag,
                                  int                                 threads,
                                  std::shared_ptr<const IclProfile> & output,
                                  const char *&                       out_diagnostic_code,
                                  const char *&                       out_diagnostic_message);

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
// Returns SYNTH_ERR_INVALID_ARG if `profile.x_vector` is empty (a
// zero-length tensor is not a shape this family's own reader, or any real
// Profile create_x_vector_profile ever builds, could load back) or if
// `profile.language_tag` exceeds `kMaxLanguageTagLength` (that constant's
// own header comment records the exact defect this refusal closes). Both
// are this writer's own defects to refuse, independently of whatever
// `create_x_vector_profile` itself already checked -- this function has no
// way to know whether `profile` reached it through that path or was built
// by hand (as, deliberately, every round-trip test in
// tests/qwen3_tts_profile_test.cpp does), so it re-asserts both invariants
// itself rather than trusting a caller-specific history.
synth_status_t serialize_x_vector_profile(const XVectorProfile & profile,
                                          const uint8_t (&compatibility_id)[32],
                                          std::vector<uint8_t> & out_bytes);

// Parses a v1 envelope out of untrusted `data`/`data_size` against the
// caller's own `compatibility_id` and declared x-vector width (`enc_dim`,
// this package's `SpeakerEncoderParams::enc_dim` -- the exact width, not a
// ceiling, since a Profile of the wrong width builds a wrong-shaped prompt).
//
// Unlike omnivoice::load_profile_from_memory, this takes `enc_dim` by value
// rather than a `Model &`. This is NOT because no payload-VALUE invariant
// needs re-checking here -- it does (see the ref_rms/x_vector paragraph
// below), the same way omnivoice's own ref_rms>0 parity check does for
// ClonePrompt. It is because every check this function needs, INCLUDING
// that one, is answerable from data already in hand: the tensor's own
// declared element count against the one scalar `enc_dim`, and ref_rms/
// x_vector against invariants that are properties of the stored FLOATS
// themselves, needing no Model lookup at all. What genuinely has no
// counterpart here is the set of checks that DO need a live Model:
// range-checking a discrete token stream against a vocabulary/mask id,
// re-tokenizing a transcript through a text frontend, and matching a
// declared language tag against the Model's own declared list --
// XVectorProfile carries no token stream, no transcript, and (Task 9's own
// job, not this function's) no validated language tag; `language_tag` is
// stored verbatim here, exactly as XVectorProfile's own header comment says
// create_x_vector_profile itself does. That is the actual substitution this
// signature makes, the same one create_x_vector_profile's own header
// comment already makes (and justifies) for `HParams`/`SpeakerEncoderWeights`
// in place of a `Model &`.
//
// Payload-value parity (a reviewer finding on this task): a loaded
// XVectorProfile satisfies the same invariants a CREATED one does, not just
// the same shape. `ref_rms` must be finite and strictly positive --
// encode_speaker_reference's own "voice_profile.reference_silent" rejection
// (speaker-encoder-host.cpp) refuses a silent reference at EXACTLY
// ref_rms == 0.0f, and reference_rms() -- a sqrt of a sum of squares over
// real PCM -- can never itself produce a negative, infinite, or NaN result,
// so any of those four states in a loaded envelope is the untrusted-bytes
// counterpart of the same rejection, checked without needing a Model.
// `x_vector` must have every element finite (encode_speaker_reference's own
// graph-level check, mirrored here as a malformed-input INVALID_ARG rather
// than that function's internal SYNTH_ERR_INTERNAL) and must not be
// identically zero (this network's own bias terms make an exactly-all-zero
// output indistinguishable from a corrupted or hand-forged payload, silent
// reference or not). Before this check existed, a reviewer measured all
// four ref_rms states plus an all-zero x_vector loading with SYNTH_OK.
//
// Status mapping (docs/c-interface.md: "Incompatible Model data returns
// SYNTH_ERR_UNSUPPORTED_VOICE; malformed, truncated, or corrupt data returns
// SYNTH_ERR_INVALID_ARG"):
//   * structurally broken bytes (gguf_init_from_buffer itself refuses them,
//     a required key is missing/wrongly typed/duplicated, the tensor shape
//     or byte range does not fit `data`, the x-vector's declared length does
//     not equal `enc_dim`, the content digest does not match, or a
//     payload-value invariant above is violated) -> SYNTH_ERR_INVALID_ARG;
//   * a structurally well-formed envelope for a DIFFERENT model_family,
//     schema, schema_version, or exact compatibility_id -> SYNTH_ERR_UNSUPPORTED_VOICE
//     (this loader understands the envelope, just not for this Model);
//   * an unrecognized `kind` value (Plan 2: anything other than "x-vector")
//     -> SYNTH_ERR_INVALID_ARG: unlike the three mismatches above, `kind` is
//     this ONE schema's own internal tag, not a different schema/version/
//     model -- a value this build does not recognize means the payload
//     structure the rest of the bytes describe cannot be interpreted at
//     all, which this project treats as malformed rather than merely
//     unsupported. This branch is reachable only for a HAND-BUILT buffer
//     that keeps Plan 2's own key set and count and swaps out just the
//     `kind` string (exactly what this task's own tamper matrix does) -- a
//     REAL Plan 3 "icl" envelope, carrying its own additional keys, is
//     rejected earlier by the prescan whitelist/`n_kv` gate below, which
//     Plan 3 must widen (kPrescanKnownKeys/kPrescanKvCountXVector's own
//     header comments already say so) before this `kind` branch becomes the
//     operative rejection for a genuine "icl" envelope, let alone a
//     successful load.
//
//     Naming the cost this trade carries: because an unrecognized field is
//     refused by a POSITIVE whitelist rather than accepted and ignored, a
//     structurally sound Plan 3 envelope this build does not yet understand
//     is reported as SYNTH_ERR_INVALID_ARG -- "malformed" -- rather than
//     SYNTH_ERR_UNSUPPORTED_VOICE -- "newer than this build", the status a
//     schema_version bump would have produced instead (see the
//     schema_version row above). This project accepts that trade
//     deliberately (this section's own top comment: discriminating on
//     `kind` rather than `schema_version` is what lets a Plan 2 profile
//     stay loadable under a Plan 3 build with no re-cut), but the trade is
//     not free, and a future reader of this file should not have to
//     rediscover the cost by reading a status code as a bug report.
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
// re-tokenize; the payload-value checks above return a bare INVALID_ARG,
// the same way omnivoice::load_profile_from_memory's own ref_rms>0 parity
// check does). Both out-parameters exist only so this signature matches the
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
