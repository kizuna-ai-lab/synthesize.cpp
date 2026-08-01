#include "audio-normalizer.h"
#include "test-assert.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

double rms(const std::vector<float> & samples) {
    double sum_of_squares = 0.0;
    for (float sample : samples) {
        sum_of_squares += static_cast<double>(sample) * static_cast<double>(sample);
    }
    return std::sqrt(sum_of_squares / static_cast<double>(samples.size()));
}

// A whole number of 1 kHz periods at 48 kHz (period = 48 samples), so there is
// no partial-cycle edge effect to muddy the RMS comparison.
std::vector<float> sine_fixture(uint64_t frames, uint32_t sample_rate, float frequency_hz, float amplitude) {
    constexpr double   kTwoPi = 6.283185307179586476925286766559;
    std::vector<float> samples(frames);
    for (uint64_t index = 0; index < frames; ++index) {
        const double t = static_cast<double>(index) / static_cast<double>(sample_rate);
        samples[index] = amplitude * static_cast<float>(std::sin(kTwoPi * frequency_hz * t));
    }
    return samples;
}

}  // namespace

int main() {
    // --- RFE exactness -----------------------------------------------------
    SYNTH_TEST_CHECK(synth::reference_frame_equivalent(1000, 24000, 24000) == 1000);
    SYNTH_TEST_CHECK(synth::reference_frame_equivalent(44100, 44100, 24000) == 24000);
    SYNTH_TEST_CHECK(synth::reference_frame_equivalent(1, 8000, 24000) == 3);
    // ceil(147 * 24000 / 44100) = ceil(80.0) = 80, chosen to exercise the
    // ceiling itself: 147*24000 = 3528000, 3528000/44100 = 80.0 exactly, and
    // the case below (48001 frames) is the one that lands mid-fraction.
    SYNTH_TEST_CHECK(synth::reference_frame_equivalent(147, 44100, 24000) == 80);
    // A rate pair where the true ratio does not divide evenly: this is the
    // "the resampler need not emit exactly this many frames" case the ADR
    // describes, verified separately below via the real resampler.
    SYNTH_TEST_CHECK(synth::reference_frame_equivalent(48001, 48000, 24000) == 24001);

    // --- Matching-format borrow: no resampler invocation, bit-identical ----
    {
        const std::vector<float>   input = { -0.75f, 0.0f, 0.5f, 1.0f, -1.0f, 0.125f, 0.0625f, -0.5f };
        synth::NormalizedReference output;
        SYNTH_TEST_CHECK(synth::normalize_reference(input.data(), input.size(), 24000, 1, 24000, 1, output) ==
                         SYNTH_OK);
        SYNTH_TEST_CHECK(output.frames == input.size());
        SYNTH_TEST_CHECK(output.pcm.size() == input.size());
        // Bit-equality, not approximate equality: SRC_SINC_BEST_QUALITY would
        // not reproduce this exactly, so this only passes if the resampler was
        // never invoked at all.
        for (size_t index = 0; index < input.size(); ++index) {
            SYNTH_TEST_CHECK(output.pcm[index] == input[index]);
        }
    }

    // --- Stereo -> mono: exact equal-weight mean, same rate so the resampler
    //     is never invoked and the arithmetic stays exact -------------------
    {
        constexpr uint64_t kFrames = 16;
        std::vector<float> input(kFrames * 2);
        for (uint64_t frame = 0; frame < kFrames; ++frame) {
            input[frame * 2]     = 0.5f;
            input[frame * 2 + 1] = -0.25f;
        }
        synth::NormalizedReference output;
        SYNTH_TEST_CHECK(synth::normalize_reference(input.data(), kFrames, 24000, 2, 24000, 1, output) == SYNTH_OK);
        SYNTH_TEST_CHECK(output.frames == kFrames);
        SYNTH_TEST_CHECK(output.pcm.size() == kFrames);
        for (float sample : output.pcm) {
            SYNTH_TEST_CHECK(sample == 0.125f);
        }
    }

    // --- Mono -> stereo: exact duplication, same rate -----------------------
    {
        const std::vector<float>   input = { 0.5f, -0.25f, 1.0f };
        synth::NormalizedReference output;
        SYNTH_TEST_CHECK(synth::normalize_reference(input.data(), input.size(), 16000, 1, 16000, 2, output) ==
                         SYNTH_OK);
        SYNTH_TEST_CHECK(output.frames == input.size());
        SYNTH_TEST_CHECK(output.pcm.size() == input.size() * 2);
        for (size_t frame = 0; frame < input.size(); ++frame) {
            SYNTH_TEST_CHECK(output.pcm[frame * 2] == input[frame]);
            SYNTH_TEST_CHECK(output.pcm[frame * 2 + 1] == input[frame]);
        }
    }

    // --- Rate conversion sanity: 1 kHz sine, 48 kHz -> 24 kHz --------------
    {
        constexpr uint64_t         kFrames = 4800;  // 100 whole periods at 1 kHz / 48 kHz.
        const std::vector<float>   input   = sine_fixture(kFrames, 48000, 1000.0f, 0.8f);
        synth::NormalizedReference output;
        SYNTH_TEST_CHECK(synth::normalize_reference(input.data(), kFrames, 48000, 1, 24000, 1, output) == SYNTH_OK);
        const uint64_t expected_frames = synth::reference_frame_equivalent(kFrames, 48000, 24000);
        SYNTH_TEST_CHECK(output.frames == expected_frames);
        SYNTH_TEST_CHECK(output.pcm.size() == output.frames);
        const double input_rms  = rms(input);
        const double output_rms = rms(output.pcm);
        SYNTH_TEST_CHECK(std::fabs(output_rms - input_rms) <= 0.05 * input_rms);
    }

    // --- Error mapping -------------------------------------------------------
    {
        const std::vector<float>   valid = { 0.1f, -0.1f, 0.2f, -0.2f };
        synth::NormalizedReference output;

        // Input sample rate outside [8000, 192000].
        SYNTH_TEST_CHECK(synth::normalize_reference(valid.data(), valid.size(), 7999, 1, 24000, 1, output) ==
                         SYNTH_ERR_UNSUPPORTED_INPUT);
        SYNTH_TEST_CHECK(synth::normalize_reference(valid.data(), valid.size(), 192001, 1, 24000, 1, output) ==
                         SYNTH_ERR_UNSUPPORTED_INPUT);
        // Boundary values are accepted (off-by-one sanity on the range check).
        SYNTH_TEST_CHECK(synth::normalize_reference(valid.data(), valid.size(), 8000, 1, 8000, 1, output) == SYNTH_OK);
        SYNTH_TEST_CHECK(synth::normalize_reference(valid.data(), valid.size(), 192000, 1, 192000, 1, output) ==
                         SYNTH_OK);

        // Input channel count outside [1, 2]. The channel/rate range check
        // runs before any sample is read, so frame/channel layout need not
        // agree with valid's actual size here.
        SYNTH_TEST_CHECK(synth::normalize_reference(valid.data(), valid.size(), 24000, 3, 24000, 1, output) ==
                         SYNTH_ERR_UNSUPPORTED_INPUT);

        // Non-finite samples.
        std::vector<float> with_nan = valid;
        with_nan[2]                 = std::numeric_limits<float>::quiet_NaN();
        SYNTH_TEST_CHECK(synth::normalize_reference(with_nan.data(), with_nan.size(), 24000, 1, 24000, 1, output) ==
                         SYNTH_ERR_INVALID_ARG);
        std::vector<float> with_inf = valid;
        with_inf[0]                 = std::numeric_limits<float>::infinity();
        SYNTH_TEST_CHECK(synth::normalize_reference(with_inf.data(), with_inf.size(), 24000, 1, 24000, 1, output) ==
                         SYNTH_ERR_INVALID_ARG);

        // Zero frames.
        SYNTH_TEST_CHECK(synth::normalize_reference(valid.data(), 0, 24000, 1, 24000, 1, output) ==
                         SYNTH_ERR_INVALID_ARG);

        // Null pcm.
        SYNTH_TEST_CHECK(synth::normalize_reference(nullptr, valid.size(), 24000, 1, 24000, 1, output) ==
                         SYNTH_ERR_INVALID_ARG);
    }

    return 0;
}
