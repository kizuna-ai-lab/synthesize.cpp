// The repetition penalty, which this port shipped without.
//
// The checkpoint carries `repetition_penalty: 1.05` in its generation config and
// the reference samples with it. This port reimplemented the filter chain from
// temperature through top-k to top-p and simply omitted the penalty, so long
// inputs stopped mid-sentence: the talker ends an utterance by drawing the codec
// end token, and everything in front of that draw decides when it does.
//
// Nothing caught it. The Port Validation Contract's replay seam feeds the
// oracle's codes precisely so that comparison is deterministic, which means the
// sampler is the one stage eighteen Golden cases never execute.

#include "arch/qwen3-tts/code-predictor-host.h"
#include "test-assert.h"

#include <cmath>
#include <vector>

int main() {
    // Hugging Face's rule: a positive logit is divided and a negative one
    // multiplied, so both move toward zero. Rewarding a negative score for
    // having appeared would be the opposite of a penalty.
    {
        std::vector<float>         logits  = { 2.0f, -2.0f, 0.5f, -0.5f };
        const std::vector<int32_t> history = { 0, 1 };
        synth::qwen3tts::apply_repetition_penalty(logits, history, 2.0f);
        SYNTH_TEST_CHECK(std::fabs(logits[0] - 1.0f) < 1e-6f);
        SYNTH_TEST_CHECK(std::fabs(logits[1] + 4.0f) < 1e-6f);
        // Untouched entries keep their value exactly.
        SYNTH_TEST_CHECK(logits[2] == 0.5f && logits[3] == -0.5f);
    }

    // A code drawn many times is penalised once, not once per occurrence.
    // Compounding would make the penalty a function of repetition count, and the
    // reference processor walks the sequence dividing a fresh copy of the logits.
    {
        std::vector<float>         once   = { 4.0f };
        std::vector<float>         thrice = { 4.0f };
        synth::qwen3tts::apply_repetition_penalty(once, { 0 }, 2.0f);
        synth::qwen3tts::apply_repetition_penalty(thrice, { 0, 0, 0 }, 2.0f);
        SYNTH_TEST_CHECK(once[0] == thrice[0]);
        SYNTH_TEST_CHECK(std::fabs(once[0] - 2.0f) < 1e-6f);
    }

    // A penalty of one is the identity, which is what a package declaring no
    // penalty must produce -- not a silently different distribution.
    {
        std::vector<float> logits = { 1.5f, -3.0f, 0.0f };
        synth::qwen3tts::apply_repetition_penalty(logits, { 0, 1, 2 }, 1.0f);
        SYNTH_TEST_CHECK(logits[0] == 1.5f && logits[1] == -3.0f && logits[2] == 0.0f);
    }

    // An empty history leaves everything alone: the first frame of an utterance
    // has nothing to suppress.
    {
        std::vector<float> logits = { 1.0f, 2.0f };
        synth::qwen3tts::apply_repetition_penalty(logits, {}, 1.05f);
        SYNTH_TEST_CHECK(logits[0] == 1.0f && logits[1] == 2.0f);
    }

    // Out-of-range codes are ignored rather than read. A history entry can only
    // be malformed if something upstream is wrong, and reading past the logits
    // would turn that into memory corruption.
    {
        std::vector<float> logits = { 1.0f, 2.0f };
        synth::qwen3tts::apply_repetition_penalty(logits, { -1, 2, 99 }, 2.0f);
        SYNTH_TEST_CHECK(logits[0] == 1.0f && logits[1] == 2.0f);
    }

    return 0;
}
