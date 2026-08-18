#pragma once

#include "bpe-frontend.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace synth::qwen3tts {

// The shared byte-level BPE, re-exported under this family's namespace so the
// family's own files and tests keep reading naturally.
using synth::BpeFrontendConfig;
using synth::make_bpe_frontend;
using synth::qwen_pretokenize;

// The assistant turn the reference wraps every request in, which is a fixed
// string rather than a template the package carries:
//
//     <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
//
// `_build_assistant_text`, qwen3_tts_model.py:269-270.
std::string qwen_assistant_turn(const std::string & text);

// The turn the reference wraps a *reference transcript* in for
// transcript-assisted cloning, which is a different string from the one above:
// it stops at the closing markers instead of opening a second assistant turn.
//
//     <|im_start|>assistant\n{text}<|im_end|>\n
//
// `_build_ref_text`, qwen3_tts_model.py:272-273.
std::string qwen_reference_turn(const std::string & text);

// How many tokens the *assistant* turn's role prefix and closing markers
// occupy, so a caller can split a tokenized turn back into its parts without
// re-tokenizing. Upstream slices the same way, at 3 and -5. "Upstream" and not
// "the reference": kReferenceSuffixTokens below is about the reference *turn*,
// which slices at 3 and -2, and the two senses of the word sit four lines
// apart.
constexpr size_t kAssistantRolePrefixTokens = 3;
constexpr size_t kAssistantSuffixTokens     = 5;

// The reference turn's closing markers -- `<|im_end|>` and the newline after
// it, two tokens where the assistant turn spends five. Its role prefix is the
// same `<|im_start|>assistant\n` and costs the same three tokens, so
// kAssistantRolePrefixTokens serves both and there is no second prefix
// constant. Upstream slices `ref_id[:, 3:-2]` beside `text_id[:, 3:-5]` at one
// call site, modeling_qwen3_tts.py:2190-2191.
constexpr size_t kReferenceSuffixTokens = 2;

// Whether `transcript` names no speech: empty, or nothing but ASCII
// whitespace. The design's section 9 error table gives those two
// states one row and one status, so this predicate gives them one name.
//
// ASCII whitespace only, deliberately: what this refuses is a transcript a
// caller passed through without content. A transcript made entirely of U+3000
// ideographic spaces passes here and then tokenizes to real ids -- it is not
// the case that row is about, and widening this to Unicode whitespace would
// mean carrying a table for it.
//
// EXPORTED WITH TWO CALLERS ON PURPOSE, rather than kept file-local to
// bpe.cpp where it started: qwen_reference_transcript_ids below applies it
// before tokenizing, and profile.cpp's create_icl_profile applies it before
// preparing an ICL Voice Profile -- and those two must never disagree about
// which transcripts exist. A second, independently-written "is it blank?"
// test in profile.cpp would be free to drift, and the drift would show up as
// a Profile that this file's own tokenizer had already refused to make ids
// for (or the reverse).
bool qwen_transcript_is_blank(const std::string & transcript);

// A reference transcript to the ids upstream passes as `ref_id`: wrap it in the
// reference turn, tokenize the whole string, then cut the role prefix and the
// closing markers back off (qwen3_tts_model.py:598, then the slice at
// modeling_qwen3_tts.py:2191).
//
// Wrap-then-slice is not a roundabout way of tokenizing the bare transcript,
// and the difference is not a rounding-sized one. The pre-tokenizer's newline
// branch is greedy, so a transcript that opens with a newline has that newline
// swallowed by the turn's own: measured on the shipped Base package's
// vocabulary, "\nHello" tokenizes through this path to the single id 9707,
// which is what bare "Hello" gives, while tokenizing "\nHello" on its own gives
// 198, 9707. A prompt off by one token still synthesizes speech, in the wrong
// voice or the wrong language (talker-host.h), so this is the kind of
// difference nothing downstream would report.
//
// `frontend` must apply no turn of its own -- this applies the reference turn
// itself, and a frontend carrying the assistant turn (which is the one the core
// runs for every request) would wrap the result a second time.
//
// SYNTH_ERR_INVALID_ARG for an empty or whitespace-only transcript, which is
// the design's §9 error table. SYNTH_ERR_INPUT_TOO_LONG when the wrapped turn
// exceeds `max_tokens`, counting the turn's own markers exactly as
// Model::tokenize_request's limit does. SYNTH_ERR_TEXT_FRONTEND when the
// frontend returns fewer tokens than the turn's own markers, which is the
// precondition above going unmet rather than anything a caller did.
synth_status_t qwen_reference_transcript_ids(const TextFrontend &   frontend,
                                             const std::string &    transcript,
                                             uint64_t               max_tokens,
                                             std::vector<int32_t> & token_ids);

// The turn a Description Text instruct is wrapped in before tokenizing, a
// third and different turn from the two above -- a USER turn, not an
// assistant one:
//
//     <|im_start|>user\n{instruct}<|im_end|>\n
//
// `_build_instruct_text`, qwen3_tts_model.py:275-276.
std::string qwen_instruct_turn(const std::string & instruct);

// An instruct string to the token ids upstream calls `instruct_ids`: wrap it in
// the instruct turn and tokenize the WHOLE wrapped string.
//
// UNLIKE qwen_reference_transcript_ids, upstream applies NO slice here --
// `instruct_ids.append(self._tokenize_texts([self._build_instruct_text(ins)])[0])`
// (qwen3_tts_model.py:715) keeps the wrapper's own role-marker tokens in the
// result, because they are meant to reach the talker: modeling_qwen3_tts.py:
// 2076-2080 embeds `instruct_id` whole, with no slice of its own either.
//
// An empty `instruct` produces an EMPTY `token_ids` and SYNTH_OK -- design
// decision D3's `if ins is None or ins == "": instruct_ids.append(None)`
// (qwen3_tts_model.py:712-713): no instruct block at all, not a
// wrapped-empty-string block. Tokenizing "" through the turn would still
// yield the role markers' own (non-empty) tokens, which is a real instruct
// block and not what upstream does for this case -- so the empty check runs
// BEFORE wrapping, the same ordering qwen_reference_transcript_ids' own blank
// check uses for a different reason (there, blank is refused; here, blank is
// accepted and turned into absence).
//
// `frontend` must apply no turn of its own, the same requirement
// qwen_reference_transcript_ids states and for the same reason: this applies
// the instruct turn itself, and a frontend already carrying a turn would wrap
// the result a second time.
synth_status_t qwen_instruct_ids(const TextFrontend &   frontend,
                                 const std::string &    instruct,
                                 uint64_t               max_tokens,
                                 std::vector<int32_t> & token_ids);

}  // namespace synth::qwen3tts
