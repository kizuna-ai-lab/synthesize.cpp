#pragma once

// The reference-clip trim, transcribed from the oracle rather than inferred
// from the flag's name.
//
// scripts/dump_reference_qwen3_tts_base.py takes `--trim-seconds` and the
// Golden Manifest uses it -- `trim_seconds: 1.0` for base-ref-min and `30.0`
// for base-ref-max -- but until Plan 4 Task 4 the C++ drivers had no trim at
// all. The consequence is recorded in tests/tolerances/qwen3-tts.json: the only
// two cases that would add an x-vector data point are exactly the two this
// comparison could not drive, because "run_compare_x_vector hands the driver
// the WAV path and the driver has no trim, so the port would encode the full
// clip while the oracle encoded a trimmed one, and the comparison would fail
// for a reason that is not the port's arithmetic."
//
// THE OVER-LENGTH CASE LOOPS, IT DOES NOT TRUNCATE, and that is the half a
// reader would get wrong from the flag's name. base-ref-max asks for 30.0 s of
// an 8.08 s recording; the oracle answers with np.tile
// (dump_reference_qwen3_tts_base.py:339-342), roughly 3.7 repeats sliced back
// to the requested length. A driver that truncated where the oracle loops
// produces a comparison failure that is not the port's arithmetic -- the same
// class of false signal the missing flag produced.
//
// Transcribed, line for line:
//
//     target = int(round(trim_seconds * sr))
//     if target <= 0:            raise
//     if target <= wav.shape[0]: wav = wav[:target]
//     else:                      repeats = ceil(target / len(wav))
//                                wav = np.tile(wav, repeats)[:target]
//
// One shared copy rather than one per driver. The four WAV readers in this
// directory are duplicated by a recorded decision, but that decision is about
// a reader whose behaviour is pinned by a format; this is a transcription of
// another implementation's arithmetic, which is the shape that drifted apart
// silently in the p95 statistic until Plan 4 Task 2 gave it a cross-check.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace synth::qwen3_tts::testing {

struct TrimOutcome {
    bool    applied = false;
    bool    looped  = false;
    int64_t repeats = 1;
    size_t  samples = 0;
};

// Returns false when `trim_seconds` is non-positive, which the oracle treats as
// an error rather than as "no trim". A caller that wants no trim does not call
// this at all.
inline bool trim_reference(std::vector<float> & samples,
                           double               trim_seconds,
                           uint32_t             sample_rate,
                           TrimOutcome &        outcome) {
    if (!(trim_seconds > 0.0) || sample_rate == 0 || samples.empty()) {
        return false;
    }
    const int64_t target = int64_t(std::llround(trim_seconds * double(sample_rate)));
    if (target <= 0) {
        return false;
    }
    const int64_t have = int64_t(samples.size());
    if (target <= have) {
        samples.resize(size_t(target));
        outcome = TrimOutcome{ true, false, 1, samples.size() };
        return true;
    }
    // Ceiling division, matching numpy's repeat count before the slice.
    const int64_t      repeats = (target + have - 1) / have;
    std::vector<float> tiled;
    tiled.reserve(size_t(repeats) * samples.size());
    for (int64_t repeat = 0; repeat < repeats; ++repeat) {
        tiled.insert(tiled.end(), samples.begin(), samples.end());
    }
    tiled.resize(size_t(target));
    samples.swap(tiled);
    outcome = TrimOutcome{ true, true, repeats, samples.size() };
    return true;
}

}  // namespace synth::qwen3_tts::testing
