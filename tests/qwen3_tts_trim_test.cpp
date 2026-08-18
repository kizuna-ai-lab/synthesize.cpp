// The reference-clip trim, under the `unit` label.
//
// WHY THIS EXISTS. tests/qwen3_tts_trim.h was added by Plan 4 Task 4 and called
// from exactly one place -- tests/qwen3_tts_icl_real.cpp, which is
// integration-tier and which `synthesize-check-unit` does not build. So a
// transcription of ANOTHER implementation's arithmetic shipped with no test
// that runs in the standard gate, which is the same shape the p95 statistic was
// in until Plan 4 Task 2 gave it a cross-check.
//
// WHAT IT PINS, and the order matters: the OVER-LENGTH case loops rather than
// truncating. base-ref-max asks for 30.0 s of an 8.08 s recording and the
// oracle answers with np.tile. A driver that truncated where the oracle loops
// produces a comparison failure that is not the port's arithmetic -- a false
// signal about the port, from the test harness.
//
// The expectations here are the oracle's arithmetic restated by hand, not
// computed with the function under test. There is no fixture because there is
// nothing to cross-check against: the oracle is five lines of Python
// transcribed into the header's own comment, and these cases are that Python
// evaluated by hand.

#include "qwen3_tts_trim.h"

#include "test-assert.h"

#include <cstdio>
#include <vector>

namespace {

using synth::qwen3_tts::testing::trim_reference;
using synth::qwen3_tts::testing::TrimOutcome;

std::vector<float> ramp(size_t count) {
    std::vector<float> samples;
    samples.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        samples.push_back(float(i + 1));
    }
    return samples;
}

}  // namespace

int main() {
    // ---- refusals -------------------------------------------------------
    // Each returns false and each leaves `samples` untouched, so a caller that
    // ignored the return value gets the untrimmed clip rather than a mangled
    // one.
    {
        TrimOutcome outcome;
        for (const double seconds : { 0.0, -1.0, -0.001 }) {
            std::vector<float> samples = ramp(8);
            SYNTH_TEST_CHECK(!trim_reference(samples, seconds, 24000, outcome));
            SYNTH_TEST_CHECK(samples.size() == 8);
            SYNTH_TEST_CHECK(!outcome.applied);
        }

        // Zero sample rate: the oracle has no branch for it because a WAV that
        // parsed carries one. Refused rather than producing target 0.
        std::vector<float> samples = ramp(8);
        SYNTH_TEST_CHECK(!trim_reference(samples, 1.0, 0, outcome));
        SYNTH_TEST_CHECK(samples.size() == 8);

        // Empty input: this is the one refusal that is NOT transcribed, and it
        // cannot be -- the loop arm divides by the sample count. Documented as
        // a deliberate departure in the header.
        std::vector<float> nothing;
        SYNTH_TEST_CHECK(!trim_reference(nothing, 1.0, 24000, outcome));
        SYNTH_TEST_CHECK(nothing.empty());

        // A positive `trim_seconds` so short it rounds to zero samples. The
        // oracle raises on `target <= 0` AFTER rounding, not before, so this
        // must be refused even though the request itself was positive.
        std::vector<float> tiny = ramp(8);
        SYNTH_TEST_CHECK(!trim_reference(tiny, 1e-9, 24000, outcome));
        SYNTH_TEST_CHECK(tiny.size() == 8);
    }

    // ---- truncation -----------------------------------------------------
    {
        std::vector<float> samples = ramp(10);
        TrimOutcome        outcome;
        SYNTH_TEST_CHECK(trim_reference(samples, 4.0, 1, outcome));
        SYNTH_TEST_CHECK(outcome.applied);
        SYNTH_TEST_CHECK(!outcome.looped);
        SYNTH_TEST_CHECK(outcome.repeats == 1);
        SYNTH_TEST_CHECK(outcome.samples == 4);
        SYNTH_TEST_CHECK(samples.size() == 4);
        // A PREFIX, not a decimation: the manifest's base-ref-min is described
        // as a byte-exact 24,000-sample prefix of base-icl-en.
        for (size_t i = 0; i < samples.size(); ++i) {
            SYNTH_TEST_CHECK(samples[i] == float(i + 1));
        }
    }

    // The boundary between the two arms: target == have takes the truncating
    // branch and must not loop. `target <= have` in the oracle, not `<`.
    {
        std::vector<float> samples = ramp(6);
        TrimOutcome        outcome;
        SYNTH_TEST_CHECK(trim_reference(samples, 6.0, 1, outcome));
        SYNTH_TEST_CHECK(!outcome.looped);
        SYNTH_TEST_CHECK(outcome.repeats == 1);
        SYNTH_TEST_CHECK(samples.size() == 6);
        SYNTH_TEST_CHECK(samples[5] == 6.0f);
    }

    // ---- looping --------------------------------------------------------
    // THE HALF A READER WOULD GET WRONG FROM THE FLAG'S NAME. 8 samples asked
    // of a 3-sample clip: np.tile three times gives 9, sliced back to 8.
    {
        std::vector<float> samples = { 1.0f, 2.0f, 3.0f };
        TrimOutcome        outcome;
        SYNTH_TEST_CHECK(trim_reference(samples, 8.0, 1, outcome));
        SYNTH_TEST_CHECK(outcome.applied);
        SYNTH_TEST_CHECK(outcome.looped);
        SYNTH_TEST_CHECK(outcome.repeats == 3);
        SYNTH_TEST_CHECK(outcome.samples == 8);
        const float expected[8] = { 1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f, 1.0f, 2.0f };
        SYNTH_TEST_CHECK(samples.size() == 8);
        for (size_t i = 0; i < samples.size(); ++i) {
            SYNTH_TEST_CHECK(samples[i] == expected[i]);
        }
    }

    // An exact multiple: the ceiling division must not add a fourth repeat that
    // the slice then discards. 9 of 3 is 3, not 4.
    {
        std::vector<float> samples = { 1.0f, 2.0f, 3.0f };
        TrimOutcome        outcome;
        SYNTH_TEST_CHECK(trim_reference(samples, 9.0, 1, outcome));
        SYNTH_TEST_CHECK(outcome.looped);
        SYNTH_TEST_CHECK(outcome.repeats == 3);
        SYNTH_TEST_CHECK(samples.size() == 9);
        SYNTH_TEST_CHECK(samples[8] == 3.0f);
    }

    // One sample past the end takes the loop arm, which is where an
    // off-by-one between `<` and `<=` would show.
    {
        std::vector<float> samples = ramp(4);
        TrimOutcome        outcome;
        SYNTH_TEST_CHECK(trim_reference(samples, 5.0, 1, outcome));
        SYNTH_TEST_CHECK(outcome.looped);
        SYNTH_TEST_CHECK(outcome.repeats == 2);
        SYNTH_TEST_CHECK(samples.size() == 5);
        SYNTH_TEST_CHECK(samples[4] == 1.0f);
    }

    // ---- the rounding ---------------------------------------------------
    // TIES GO TO EVEN, because Python's `round` does and this is a
    // transcription of Python. std::llround would round half AWAY FROM ZERO and
    // answer 3 to the first case below -- one sample of difference, on an input
    // no committed case reaches today, which is exactly why it would go
    // unnoticed. Both products are exact in binary, so this is a real tie and
    // not a floating-point near-miss.
    {
        std::vector<float> samples = ramp(8);
        TrimOutcome        outcome;
        SYNTH_TEST_CHECK(trim_reference(samples, 1.25, 2, outcome));  // 2.5 -> 2
        SYNTH_TEST_CHECK(outcome.samples == 2);
    }
    {
        std::vector<float> samples = ramp(8);
        TrimOutcome        outcome;
        SYNTH_TEST_CHECK(trim_reference(samples, 1.75, 2, outcome));  // 3.5 -> 4
        SYNTH_TEST_CHECK(outcome.samples == 4);
    }
    // And an ordinary non-tie still rounds to nearest rather than truncating.
    {
        std::vector<float> samples = ramp(8);
        TrimOutcome        outcome;
        SYNTH_TEST_CHECK(trim_reference(samples, 2.6, 1, outcome));  // -> 3
        SYNTH_TEST_CHECK(outcome.samples == 3);
    }

    // ---- the shapes the Golden Manifest actually asks for ----------------
    // base-ref-min is trim_seconds 1.0 and base-ref-max is 30.0, both at
    // 24000 Hz against clone.wav's 193,920 samples. One truncates, one loops --
    // which is the pairing the manifest chose and the reason this header
    // exists.
    {
        const size_t       clip = 193920;  // clone.wav, 8.08 s at 24 kHz
        std::vector<float> samples(clip, 0.5f);
        TrimOutcome        outcome;
        SYNTH_TEST_CHECK(trim_reference(samples, 1.0, 24000, outcome));
        SYNTH_TEST_CHECK(!outcome.looped);
        SYNTH_TEST_CHECK(outcome.samples == 24000);

        std::vector<float> again(clip, 0.5f);
        SYNTH_TEST_CHECK(trim_reference(again, 30.0, 24000, outcome));
        SYNTH_TEST_CHECK(outcome.looped);
        SYNTH_TEST_CHECK(outcome.samples == 720000);
        SYNTH_TEST_CHECK(outcome.repeats == 4);  // ceil(720000 / 193920) = 4
    }

    std::fprintf(stderr, "qwen3-tts-trim: refusals, truncation, looping and ties-to-even all hold\n");
    return 0;
}
