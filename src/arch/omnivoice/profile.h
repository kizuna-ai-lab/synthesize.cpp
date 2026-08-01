#pragma once

#include "synthesize.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace synth::omnivoice {

class Model;

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

}  // namespace synth::omnivoice
