#include "bpe.h"

#include <cctype>

namespace synth::qwen3tts {

namespace {

// ASCII whitespace only, deliberately: what this refuses is a transcript that
// names no speech, and the case that reaches it is an empty or blank string a
// caller passed through. A transcript made entirely of U+3000 ideographic
// spaces would pass here and then tokenize to real ids -- it is not the case
// the design's §9 row is about, and widening this to Unicode whitespace would
// mean carrying a table for it.
bool is_blank(const std::string & text) {
    for (const char byte : text) {
        if (std::isspace(static_cast<unsigned char>(byte)) == 0) {
            return false;
        }
    }
    return true;
}

}  // namespace

std::string qwen_assistant_turn(const std::string & text) {
    return "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
}

std::string qwen_reference_turn(const std::string & text) {
    return "<|im_start|>assistant\n" + text + "<|im_end|>\n";
}

synth_status_t qwen_reference_transcript_ids(const TextFrontend &   frontend,
                                             const std::string &    transcript,
                                             uint64_t               max_tokens,
                                             std::vector<int32_t> & token_ids) {
    token_ids.clear();
    if (is_blank(transcript)) {
        return SYNTH_ERR_INVALID_ARG;
    }

    const std::string    turn = qwen_reference_turn(transcript);
    std::vector<int32_t> wrapped;
    const synth_status_t status =
        frontend.prepare(SYNTH_INPUT_TEXT_UTF8, turn.data(), turn.size(), max_tokens, wrapped);
    if (status != SYNTH_OK) {
        return status;
    }

    // The markers, plus at least one token of transcript. No *vocabulary* can
    // land here, and saying why is the point: the pre-tokenizer splits
    // "assistant\n{transcript}" into at least three pieces, because its letter
    // branch cannot take a newline and its newline branch cannot take a
    // letter, and a non-blank transcript always puts a non-whitespace byte in
    // a piece of its own -- so a frontend that encodes at all returns at least
    // six tokens here. What this guards is the precondition on `frontend`:
    // hand it something that is not this family's byte-level BPE and the two
    // iterators below cross, which assign() turns into a length_error and an
    // abort -- measured, by deleting this branch and running the test that
    // reaches it, in both the Release and the sanitizer tree.
    if (wrapped.size() < kAssistantRolePrefixTokens + kReferenceSuffixTokens + 1) {
        return SYNTH_ERR_TEXT_FRONTEND;
    }
    token_ids.assign(wrapped.begin() + kAssistantRolePrefixTokens, wrapped.end() - kReferenceSuffixTokens);
    return SYNTH_OK;
}

}  // namespace synth::qwen3tts
