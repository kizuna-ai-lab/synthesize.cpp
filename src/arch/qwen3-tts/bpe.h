#pragma once

#include "bpe-frontend.h"

#include <cstddef>
#include <string>

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
std::string qwen_assistant_turn(const std::string & text);

constexpr size_t kAssistantRolePrefixTokens = 3;
constexpr size_t kAssistantSuffixTokens     = 5;

}  // namespace synth::qwen3tts
