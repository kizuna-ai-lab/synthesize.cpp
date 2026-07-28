#pragma once

#include "synthesize.h"
#include "text-frontend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace synth::qwen3tts {

// A byte-level byte-pair tokenizer, which is what this family's text tower
// expects and what no existing frontend in this project provides: the symbol-map
// frontend maps one symbol to one id, and this maps a byte sequence to a
// sequence of merged pieces.
struct BpeFrontendConfig {
    std::string provider_id;
    uint32_t    contract_version = 0;
    // Dense table indexed by token id, in the byte-level alphabet -- the vocab
    // entries are not UTF-8 text, they are bytes remapped into printable code
    // points so that every byte has a character.
    std::vector<std::string> vocab;
    // Ranked merges, most preferred first, each a pair separated by one space.
    std::vector<std::string> merges;
    // Matched literally and never split, which is what lets the prompt
    // template's markers survive tokenization.
    //
    // Each carries its own id because these are *added* tokens: `<|im_start|>` is
    // 151644 against a vocabulary of 151643, so it cannot be looked up there.
    std::vector<std::pair<std::string, int32_t>> special_tokens;
};

synth_status_t make_bpe_frontend(const BpeFrontendConfig & config, std::unique_ptr<TextFrontend> & output);

// Splits text the way the reference's pre-tokenizer pattern does, before any
// merging. Exposed because this is the stage that decides the tokenization and
// the only one that can be checked without the package's 151k-entry vocabulary:
// a wrong split produces different ids and therefore different, still-plausible
// speech.
std::vector<std::string> qwen_pretokenize(const std::string & text);

// The assistant turn the reference wraps every request in, which is a fixed
// string rather than a template the package carries:
//
//     <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
//
// The talker's prompt layout then takes the first three tokens as the role
// prefix and the text between them; see talker-host.h.
std::string qwen_assistant_turn(const std::string & text);

// How many tokens the role prefix and the closing markers occupy, so a caller
// can split a tokenized turn back into its parts without re-tokenizing. The
// reference slices the same way, at 3 and -5.
constexpr size_t kAssistantRolePrefixTokens = 3;
constexpr size_t kAssistantSuffixTokens     = 5;

}  // namespace synth::qwen3tts
