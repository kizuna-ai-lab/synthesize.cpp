#pragma once

#include "synthesize.h"
#include "text-frontend.h"

#include <cstdint>
#include <string>
#include <vector>

namespace synth::omnivoice {

// The host side of this family's text path: the two strings the prompt is built
// out of, the tag-aware tokenization of one of them, and the canvas length.
//
// None of it touches a tensor, and none of it can fail loudly. A wrong cleanup
// rule or a wrong character weight produces a prompt the model still accepts and
// a canvas the decoder still fills, so every rule here is a transcription of
// `omnivoice/models/omnivoice.py` and `omnivoice/utils/duration.py` at the
// pinned source revision rather than a reimplementation of their intent.

// Joins the reference text and the target text into the single string the
// `<|text_start|>`/`<|text_end|>` pair wraps, then applies the reference's four
// cleanup rules in its order.
//
// An empty `ref_text` means no reference, which is the auto-voice and
// voice-design case; the target alone is used. This is `_combine_text`.
std::string combine_text(const std::string & ref_text, const std::string & text);

// Builds the style prefix that precedes the wrapped text.
//
// `denoise` is the reference's `denoise and ref_audio_tokens is not None`
// already resolved: the `<|denoise|>` marker belongs to a clone request only.
// An empty language tag or instruction becomes the literal string "None", which
// is what selects the language-agnostic and instruction-free paths -- it is a
// value the model was trained on, not a placeholder.
std::string style_text(bool denoise, const std::string & language_tag, const std::string & instruct);

// Tokenizes an already-wrapped prompt string, splitting the thirteen bracketed
// nonverbal tags out first and tokenizing each on its own.
//
// The split exists so a tag receives the same ids whatever language surrounds
// it: tokenized in place, a tag merges with its neighbours and the ids move.
// `ids` is cleared first and left empty on failure.
synth_status_t tokenize_wrapped_text(const TextFrontend &   frontend,
                                     const std::string &    wrapped,
                                     std::vector<int32_t> & ids);

// The number of nonverbal tags, exposed so a caller can state the count it
// expects rather than trusting this file's table silently.
constexpr size_t kNonverbalTagCount = 13;

// The canvas length is not a model output. It is fixed before the first forward
// by this rule-based estimator, so it must reproduce upstream exactly: every
// downstream token index shifts with it.
class DurationEstimator {
  public:
    // Sums each character's phonetic weight. The unit is "one Latin letter".
    //
    // The summation is compensated (Neumaier), which is not decoration: the
    // reference sums with Python's `sum()`, and CPython 3.12 -- the version the
    // pinned oracle environment runs -- uses Neumaier compensation for float
    // sequences. A plain running total differs in the last ulp, and the result
    // feeds a truncation to an integer frame count.
    double total_weight(const std::string & utf8) const;

    // Scales the reference's measured duration by the weight ratio, then pulls
    // short estimates back up a cube-root curve. `ref_duration` is in audio
    // tokens, so the result is in frames. Zero when there is nothing to scale
    // from, matching the reference's own guards.
    double estimate_duration(const std::string & target, const std::string & ref, double ref_duration) const;

    // The canvas length for one request: the estimate divided by the speaking
    // rate and truncated toward zero, never below one frame.
    //
    // With no reference -- an empty `ref_text` or no reference frames -- the
    // anchor pair ("Nice to meet you.", 25 frames) stands in. That anchor is
    // part of this family's contract, not a default. The reference distinguishes
    // "no reference frames" from "zero reference frames"; this port does not,
    // because a reference with no audio behind it is not a reference and the
    // load path refuses one.
    uint64_t estimate_target_frames(const std::string & text,
                                    const std::string & ref_text,
                                    uint64_t            ref_frames,
                                    float               speaking_rate) const;
};

}  // namespace synth::omnivoice
