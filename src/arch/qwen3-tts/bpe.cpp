#include "bpe.h"

namespace synth::qwen3tts {

std::string qwen_assistant_turn(const std::string & text) {
    return "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
}

}  // namespace synth::qwen3tts
