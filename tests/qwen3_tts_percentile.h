#pragma once

// The p95 reconstruction statistic, in one place so both of its consumers can
// be held to the same arithmetic.
//
// These two functions lived inside tests/qwen3_tts_icl_real.cpp until Plan 4
// Task 2. That file is integration-tier -- `synthesize-check-unit` does not
// build it -- so nothing under the `unit` label could reach them, and the
// tolerance file recorded the consequence in its own words
// (tests/tolerances/qwen3-tts.json): the C++ arm "transcribes the p95 statistic
// from the validator by hand ... and the two implementations have NO registered
// cross-check -- they were compared once, manually, on the same buffers,
// agreeing to every printed digit, and could drift apart silently afterwards."
//
// The Python half is scripts/validate-qwen3-tts-codec_encoder.py. Neither side
// is the specification: tests/fixtures/qwen3-tts/percentile-cross-check.json is,
// and tests/qwen3_tts_percentile_test.cpp and
// tests/python/test_percentile_agreement.py each compute against it with their
// own implementation. A test that recomputed the expectation with the
// implementation under test would prove nothing, which is why the fixture is a
// third party to both.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace synth::qwen3_tts::testing {

// numpy.percentile's default ("linear") interpolation, over a copy that this
// function is free to sort. The statistic is transcribed from
// scripts/validate-qwen3-tts-codec_encoder.py rather than approximated: that
// script is where the committed gate was measured, and a nearest-rank
// percentile here would compare a different number against it.
inline double percentile_linear(std::vector<double> values, double percent) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double position = percent / 100.0 * double(values.size() - 1);
    const size_t low      = size_t(std::floor(position));
    const size_t high     = size_t(std::ceil(position));
    if (low == high) {
        return values[low];
    }
    return values[low] + (values[high] - values[low]) * (position - double(low));
}

// The p95 of the per-frame L2 deviation, relative to the REFERENCE frame's own
// L2 norm -- one branch of the [2, frames, projected] reconstruction. Same
// statistic, same operand order (`oracle` is the right operand and is the
// denominator) as scripts/validate-qwen3-tts-codec_encoder.py:251-255, where
// `norms` is taken over `b`. The operand order is load-bearing and invisible to
// a fixture of pre-computed percentile inputs, which is why the cross-check
// fixture carries a {frames, projected} port/oracle pair of its own.
inline double reconstruction_p95_relative(const float * port, const float * oracle, size_t frames, size_t projected) {
    std::vector<double> relative;
    relative.reserve(frames);
    for (size_t frame = 0; frame < frames; ++frame) {
        double difference = 0.0;
        double reference  = 0.0;
        for (size_t column = 0; column < projected; ++column) {
            const double a = double(port[frame * projected + column]);
            const double b = double(oracle[frame * projected + column]);
            difference += (a - b) * (a - b);
            reference += b * b;
        }
        relative.push_back(std::sqrt(difference) / std::sqrt(reference));
    }
    return percentile_linear(relative, 95.0);
}

// The same statistic over a SUBSET of frames, which is what Plan 4 Task 3 made
// the gated quantity.
//
// `keep[frame]` selects the frames that count. The mask the committed gate uses
// is "upstream-f32 and the bf16 oracle chose the same code here", which contains
// no port at all -- see the redesign note in
// scripts/validate-qwen3-tts-codec_encoder.py for why a mask that mentions the
// port measures 1x discrimination and is refused.
//
// Returns NaN when the mask keeps nothing. That is deliberate and the caller
// must treat it as a refusal: a NaN compares false against every bound, so a
// caller that let it through would have a gate that silently stopped gating.
inline double reconstruction_p95_relative_masked(const float *             port,
                                                 const float *             oracle,
                                                 size_t                    frames,
                                                 size_t                    projected,
                                                 const std::vector<char> & keep) {
    std::vector<double> relative;
    relative.reserve(frames);
    for (size_t frame = 0; frame < frames; ++frame) {
        if (frame >= keep.size() || keep[frame] == 0) {
            continue;
        }
        double difference = 0.0;
        double reference  = 0.0;
        for (size_t column = 0; column < projected; ++column) {
            const double a = double(port[frame * projected + column]);
            const double b = double(oracle[frame * projected + column]);
            difference += (a - b) * (a - b);
            reference += b * b;
        }
        relative.push_back(std::sqrt(difference) / std::sqrt(reference));
    }
    if (relative.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return percentile_linear(relative, 95.0);
}

// The codebook mask itself: keep[frame] is 1 where upstream-f32 and the oracle
// agree on this branch's codes. `groups` is 16 for this family; column 0 is the
// semantic branch and columns 1..groups-1 are the acoustic one, so a frame is
// acoustically flipped if ANY of the fifteen differs.
inline std::vector<char> codebook_keep_mask(const std::vector<int32_t> & upstream,
                                            const std::vector<int32_t> & oracle,
                                            size_t                       frames,
                                            size_t                       groups,
                                            bool                         semantic) {
    std::vector<char> keep(frames, 0);
    if (upstream.size() < frames * groups || oracle.size() < frames * groups || groups == 0) {
        return keep;
    }
    for (size_t frame = 0; frame < frames; ++frame) {
        bool flipped = false;
        if (semantic) {
            flipped = upstream[frame * groups] != oracle[frame * groups];
        } else {
            for (size_t group = 1; group < groups && !flipped; ++group) {
                flipped = upstream[frame * groups + group] != oracle[frame * groups + group];
            }
        }
        keep[frame] = flipped ? 0 : 1;
    }
    return keep;
}

}  // namespace synth::qwen3_tts::testing
