#include "bpe.h"

namespace synth::qwen3tts {

// Moved out of this file's anonymous namespace and declared in bpe.h when
// Plan 3's Task 8 gave it a second caller (profile.cpp's create_icl_profile);
// the rationale for ASCII-only, and for the two callers sharing one
// predicate, is on the declaration there.
// The set is written out rather than delegated to std::isspace, which answers
// against the global C locale: a host whose locale classifies an additional
// byte as space would make this predicate -- and with it the
// "voice_profile.transcript_blank" refusal and the mode selection behind it --
// depend on the environment the process happens to start in. These six are
// what the "C" locale means by space, and they are what this family's
// pre-tokenizer treats as whitespace. Naming them keeps the answer identical
// on every host, which is what the ASCII-only rule on the declaration in bpe.h
// is for.
bool qwen_transcript_is_blank(const std::string & transcript) {
    for (const char byte : transcript) {
        if (byte != ' ' && byte != '\t' && byte != '\n' && byte != '\v' && byte != '\f' && byte != '\r') {
            return false;
        }
    }
    return true;
}

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
    if (qwen_transcript_is_blank(transcript)) {
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

std::string qwen_instruct_turn(const std::string & instruct) {
    return "<|im_start|>user\n" + instruct + "<|im_end|>\n";
}

synth_status_t qwen_instruct_ids(const TextFrontend &   frontend,
                                 const std::string &    instruct,
                                 uint64_t               max_tokens,
                                 std::vector<int32_t> & token_ids) {
    token_ids.clear();
    // Design D3: `if ins is None or ins == "": instruct_ids.append(None)`
    // (qwen3_tts_model.py:712-713) -- no instruct block at all. This has to
    // run BEFORE wrapping: tokenizing the wrapped empty string would still
    // yield the turn's own (non-empty) role-marker tokens, which is a real
    // instruct block and not what upstream does for this case.
    if (instruct.empty()) {
        return SYNTH_OK;
    }

    const std::string turn = qwen_instruct_turn(instruct);
    // UNLIKE qwen_reference_transcript_ids, no slice: upstream tokenizes the
    // whole wrapped turn and hands it to the talker as-is
    // (`_tokenize_texts([self._build_instruct_text(ins)])[0]`,
    // qwen3_tts_model.py:715; embedded whole at modeling_qwen3_tts.py:
    // 2076-2080, with no slice there either), so this function's own output
    // IS the frontend's tokenization of `turn`, unmodified.
    return frontend.prepare(SYNTH_INPUT_TEXT_UTF8, turn.data(), turn.size(), max_tokens, token_ids);
}

}  // namespace synth::qwen3tts
