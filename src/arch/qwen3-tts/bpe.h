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

// How many tokens the role prefix and the closing markers occupy, so a caller
// can split a tokenized turn back into its parts without re-tokenizing. The
// reference slices the same way, at 3 and -5.
constexpr size_t kAssistantRolePrefixTokens = 3;
constexpr size_t kAssistantSuffixTokens     = 5;

// The reference turn's closing markers -- `<|im_end|>` and the newline after
// it, two tokens where the assistant turn spends five. Its role prefix is the
// same `<|im_start|>assistant\n` and costs the same three tokens, so
// kAssistantRolePrefixTokens serves both and there is no second prefix
// constant. Upstream slices `ref_id[:, 3:-2]` beside `text_id[:, 3:-5]` at one
// call site, modeling_qwen3_tts.py:2190-2191.
constexpr size_t kReferenceSuffixTokens = 2;

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

}  // namespace synth::qwen3tts
