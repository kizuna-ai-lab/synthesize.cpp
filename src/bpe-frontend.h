#pragma once

#include "synthesize.h"
#include "text-frontend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace synth {

// A byte-level byte-pair tokenizer, which is what this family's text tower
// expects and what no existing frontend in this project provides: the symbol-map
// frontend maps one symbol to one id, and this maps a byte sequence to a
// sequence of merged pieces.
struct BpeFrontendConfig {
    std::string                                  provider_id;
    uint32_t                                     contract_version = 0;
    // Dense table indexed by token id, in the byte-level alphabet -- the vocab
    // entries are not UTF-8 text, they are bytes remapped into printable code
    // points so that every byte has a character.
    std::vector<std::string>                     vocab;
    // Ranked merges, most preferred first, each a pair separated by one space.
    std::vector<std::string>                     merges;
    // Matched literally and never split, which is what lets the prompt
    // template's markers survive tokenization.
    //
    // Each carries its own id because these are *added* tokens: `<|im_start|>` is
    // 151644 against a vocabulary of 151643, so it cannot be looked up there.
    std::vector<std::pair<std::string, int32_t>> special_tokens;
    // Wrapped around the input before tokenizing, when a family's reference
    // does that as part of turning text into ids. qwen3-tts wraps every
    // request in a fixed assistant turn (see synth::qwen3tts::qwen_assistant_turn);
    // omnivoice leaves both empty, because its prompt is assembled by the
    // synthesis path rather than at tokenize time.
    std::string                                  prefix;
    std::string                                  suffix;
};

synth_status_t make_bpe_frontend(const BpeFrontendConfig & config, std::unique_ptr<TextFrontend> & output);

// Splits text the way the reference's pre-tokenizer pattern does, before any
// merging. Exposed because this is the stage that decides the tokenization and
// the only one that can be checked without the package's 151k-entry vocabulary:
// a wrong split produces different ids and therefore different, still-plausible
// speech.
//
// The name is kept as `qwen_pretokenize` rather than something family-neutral
// because it names what the pattern is, not who owns it: this is the Qwen2
// pre-tokenizer pattern, and it is shared verbatim by every Qwen-family model in
// this project, not specific to any one of them.
std::vector<std::string> qwen_pretokenize(const std::string & text);

}  // namespace synth
