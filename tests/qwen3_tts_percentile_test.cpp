// The C++ half of the p95 cross-check.
//
// WHY THIS EXISTS. tests/tolerances/qwen3-tts.json records the hole this test
// closes, in its own words: the end-to-end test "transcribes the p95 statistic
// from the validator by hand (numpy-linear percentile of the per-frame L2
// relative to the oracle frame's own norm), and the two implementations have NO
// registered cross-check -- they were compared once, manually, on the same
// buffers, agreeing to every printed digit, and could drift apart silently
// afterwards." The two sites are
// scripts/validate-qwen3-tts-codec_encoder.py's `np.percentile(relative, 95)`
// and tests/qwen3_tts_percentile.h's `percentile_linear`, which says in its own
// comment that it is deliberately reimplementing numpy's "linear"
// interpolation. `load_gates`'s fatal-disagreement discipline covers
// THRESHOLDS, not IMPLEMENTATIONS, so it does not reach this.
//
// NEITHER IMPLEMENTATION IS THE SPECIFICATION.
// tests/fixtures/qwen3-tts/percentile-cross-check.json is: every `expected` in
// it was produced by numpy once, at generation time, and committed. This test
// computes with tests/qwen3_tts_percentile.h and
// tests/python/test_percentile_agreement.py computes with the validator's own
// numpy expression, and both assert against the committed values. A test that
// recomputed its expectation with the implementation it is testing would prove
// nothing; the fixture is the third party.
//
// HOW THE FIXTURE REACHES THIS FILE. CMake reads the committed JSON at
// configure time and writes the flat projection this test reads, which is the
// same shape tests/qwen3_tts_icl_prompt_real.cpp already uses for
// alignment.json's token ids -- there is no JSON parser in this tree and this
// test does not add one. The JSON stays the single committed source; the flat
// file is never hand-written. tests/CMakeLists.txt lists the JSON in
// CMAKE_CONFIGURE_DEPENDS, so editing the fixture re-runs configure and this
// test sees the edit on a plain rebuild.
//
// THE FLAT LAYOUT, which tests/CMakeLists.txt writes and this file reads, one
// whitespace-separated token per position:
//
//     percent
//     percentile_declared_count      <- the fixture's own `entry_count`
//     percentile_actual_count        <- the LENGTH of its `entries` array
//     for each of percentile_actual_count entries:
//         n
//         expected
//         n values
//     reconstruction_declared_count
//     reconstruction_actual_count
//     for each of reconstruction_actual_count entries:
//         frames
//         projected
//         expected_p95_relative
//         expected_p95_relative_operands_swapped
//         frames * projected port values
//         frames * projected oracle values
//
// The declared and actual counts are BOTH carried, and asserted equal, so that
// deleting an entry from the JSON fails this test on the count rather than
// quietly testing fewer cases -- which is the presence inversion Plan 4 Task 2
// Step 4 requires to have a target.

#include "qwen3_tts_percentile.h"

#include "test-assert.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

using synth::qwen3_tts::testing::percentile_linear;
using synth::qwen3_tts::testing::reconstruction_p95_relative;

// The fixture's values are decimal transcriptions of IEEE doubles that
// round-trip, and both sides compute in double, so agreement here is exact up
// to the last operation's rounding rather than approximate. This bound is
// three orders of magnitude tighter than any difference an interpolation
// change produces -- nearest-rank moves `n2` from 2.9 to 3.0 -- so it
// separates "the same statistic" from "a different one" without pinning bits.
constexpr double kTolerance = 1e-12;

bool close(double a, double b) {
    const double scale = std::fmax(1.0, std::fmax(std::fabs(a), std::fabs(b)));
    return std::fabs(a - b) <= kTolerance * scale;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <percentile-cross-check.txt>\n", argv[0]);
        return 1;
    }

    std::ifstream fixture(argv[1]);
    if (!fixture) {
        std::fprintf(stderr,
                     "qwen3-tts-percentile: cannot open the flat fixture at %s. It is written by "
                     "tests/CMakeLists.txt from tests/fixtures/qwen3-tts/percentile-cross-check.json "
                     "at configure time -- re-run cmake.\n",
                     argv[1]);
        return 1;
    }

    double percent = 0.0;
    SYNTH_TEST_CHECK(bool(fixture >> percent));
    SYNTH_TEST_CHECK(percent == 95.0);

    long declared = 0;
    long actual   = 0;
    SYNTH_TEST_CHECK(bool(fixture >> declared));
    SYNTH_TEST_CHECK(bool(fixture >> actual));
    // The fixture declares its own size and the projection carries what was
    // really there. A disagreement means an entry was added or removed without
    // the count moving with it.
    SYNTH_TEST_CHECK(declared > 0);
    SYNTH_TEST_CHECK(declared == actual);

    long consumed = 0;
    for (long index = 0; index < actual; ++index) {
        long   n        = 0;
        double expected = 0.0;
        SYNTH_TEST_CHECK(bool(fixture >> n));
        SYNTH_TEST_CHECK(bool(fixture >> expected));
        SYNTH_TEST_CHECK(n > 0);

        std::vector<double> values;
        values.reserve(size_t(n));
        for (long i = 0; i < n; ++i) {
            double value = 0.0;
            SYNTH_TEST_CHECK(bool(fixture >> value));
            values.push_back(value);
        }

        const double measured = percentile_linear(values, percent);
        if (!close(measured, expected)) {
            std::fprintf(stderr,
                         "qwen3-tts-percentile: entry %ld (n = %ld): percentile_linear returned %.17g, "
                         "the committed numpy value is %.17g\n",
                         index, n, measured, expected);
            return 1;
        }
        ++consumed;
    }
    // Asserted against the DECLARED count, not against the loop bound it was
    // just derived from: a test that iterates an array and checks nothing about
    // its length passes trivially when the array shrinks.
    SYNTH_TEST_CHECK(consumed == declared);

    long recon_declared = 0;
    long recon_actual   = 0;
    SYNTH_TEST_CHECK(bool(fixture >> recon_declared));
    SYNTH_TEST_CHECK(bool(fixture >> recon_actual));
    SYNTH_TEST_CHECK(recon_declared > 0);
    SYNTH_TEST_CHECK(recon_declared == recon_actual);

    long recon_consumed = 0;
    for (long index = 0; index < recon_actual; ++index) {
        long   frames    = 0;
        long   projected = 0;
        double forward   = 0.0;
        double swapped   = 0.0;
        SYNTH_TEST_CHECK(bool(fixture >> frames));
        SYNTH_TEST_CHECK(bool(fixture >> projected));
        SYNTH_TEST_CHECK(bool(fixture >> forward));
        SYNTH_TEST_CHECK(bool(fixture >> swapped));
        SYNTH_TEST_CHECK(frames > 0);
        SYNTH_TEST_CHECK(projected > 0);

        const size_t       count = size_t(frames) * size_t(projected);
        std::vector<float> port(count, 0.0f);
        std::vector<float> oracle(count, 0.0f);
        for (size_t i = 0; i < count; ++i) {
            double value = 0.0;
            SYNTH_TEST_CHECK(bool(fixture >> value));
            port[i] = float(value);
        }
        for (size_t i = 0; i < count; ++i) {
            double value = 0.0;
            SYNTH_TEST_CHECK(bool(fixture >> value));
            oracle[i] = float(value);
        }

        // The forward direction: the denominator is the ORACLE's norm, which is
        // the right operand.
        const double measured =
            reconstruction_p95_relative(port.data(), oracle.data(), size_t(frames), size_t(projected));
        if (!close(measured, forward)) {
            std::fprintf(stderr, "qwen3-tts-percentile: reconstruction entry %ld: got %.17g, committed %.17g\n", index,
                         measured, forward);
            return 1;
        }

        // And the fixture's own proof that the operand order is observable:
        // swapping the arguments must produce the other committed value. This
        // is what gives Step 4's order inversion a target that cannot pass by
        // accident -- the two numbers differ by construction, asserted at
        // generation time.
        const double reversed =
            reconstruction_p95_relative(oracle.data(), port.data(), size_t(frames), size_t(projected));
        if (!close(reversed, swapped)) {
            std::fprintf(stderr, "qwen3-tts-percentile: reconstruction entry %ld swapped: got %.17g, committed %.17g\n",
                         index, reversed, swapped);
            return 1;
        }
        SYNTH_TEST_CHECK(!close(forward, swapped));
        ++recon_consumed;
    }
    SYNTH_TEST_CHECK(recon_consumed == recon_declared);

    // Nothing may follow the last value: a projection that carried more than
    // the counts describe would mean CMake and this reader disagree about the
    // layout, and every assertion above would have been reading the wrong
    // offsets while still passing.
    double trailing = 0.0;
    SYNTH_TEST_CHECK(!(fixture >> trailing));

    std::fprintf(stderr,
                 "qwen3-tts-percentile: %ld percentile entries and %ld reconstruction entries agree with "
                 "the committed numpy values\n",
                 consumed, recon_consumed);
    return 0;
}
