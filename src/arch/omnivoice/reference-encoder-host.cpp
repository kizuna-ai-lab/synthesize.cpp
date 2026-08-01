#include "arch/omnivoice/reference-encoder-host.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace synth::omnivoice {

namespace {

// The fixed 24000 -> 16000 pair, reduced by gcd(24000, 16000) = 8000. See the
// header for the citations behind every constant and formula here.
constexpr int    kOrigFreq           = 3;  // 24000 / 8000
constexpr int    kNewFreq            = 2;  // 16000 / 8000
constexpr int    kLowpassFilterWidth = 6;
constexpr double kRolloff            = 0.99;
constexpr double kPi                 = 3.14159265358979323846;

// base_freq = min(orig_freq, new_freq) * rolloff = min(3, 2) * 0.99 = 1.98.
constexpr double kBaseFreq = double(kOrigFreq < kNewFreq ? kOrigFreq : kNewFreq) * kRolloff;
// scale = base_freq / orig_freq = 1.98 / 3 = 0.66.
constexpr double kScale    = kBaseFreq / double(kOrigFreq);

// width = ceil(lowpass_filter_width * orig_freq / base_freq)
//       = ceil(6 * 3 / 1.98) = ceil(9.0909...) = 10 (measured against the
// pinned torchaudio in the locked venv -- see the header comment).
int sinc_kernel_width() {
    return static_cast<int>(std::ceil(double(kLowpassFilterWidth) * double(kOrigFreq) / kBaseFreq));
}

// The lazily-built polyphase kernel table: kNewFreq phases, each
// 2*width + kOrigFreq taps (23 with the pinned constants above, so 46
// coefficients total). Row-major: taps[phase * taps_per_phase + k].
struct SincKernel {
    int                width          = 0;
    int                taps_per_phase = 0;
    std::vector<float> taps;
};

const SincKernel & sinc_kernel() {
    static const SincKernel kernel = [] {
        SincKernel built;
        built.width          = sinc_kernel_width();
        built.taps_per_phase = 2 * built.width + kOrigFreq;
        built.taps.resize(size_t(kNewFreq) * size_t(built.taps_per_phase));

        for (int phase = 0; phase < kNewFreq; ++phase) {
            for (int k = 0; k < built.taps_per_phase; ++k) {
                // idx[k] = float32((k - width) / orig_freq)          (functional.py:1376)
                const float idx = float(double(k - built.width) / double(kOrigFreq));
                // t = float32(-phase / new_freq) + idx[k]            (functional.py:1378)
                float       t   = float(double(-phase) / double(kNewFreq)) + idx;
                // t *= base_freq                                    (functional.py:1379)
                t               = float(double(t) * kBaseFreq);
                // t = clamp(t, -lowpass_filter_width, lowpass_filter_width) (functional.py:1380)
                t               = std::clamp(t, -float(kLowpassFilterWidth), float(kLowpassFilterWidth));

                // window = cos(((t * pi) / lowpass_filter_width) / 2) ^ 2   (functional.py:1385)
                const float t_pi   = float(double(t) * kPi);
                const float w_arg1 = float(double(t_pi) / double(kLowpassFilterWidth));
                const float w_arg2 = float(double(w_arg1) / 2.0);
                const float cosine = std::cos(w_arg2);
                const float window = cosine * cosine;

                // t_rad = t * pi (functional.py:1393 reassigns t -- the same
                // expression as t_pi above, so reused rather than recomputed).
                const float t_rad = t_pi;
                // sinc = t_rad == 0 ? 1.0 : sin(t_rad) / t_rad       (functional.py:1396)
                const float sinc  = (t_rad == 0.0f) ? 1.0f : std::sin(t_rad) / t_rad;

                // kernel = sinc * (window * scale)                  (functional.py:1397)
                const float scaled_window                                            = float(double(window) * kScale);
                built.taps[size_t(phase) * size_t(built.taps_per_phase) + size_t(k)] = sinc * scaled_window;
            }
        }
        return built;
    }();
    return kernel;
}

}  // namespace

bool resample_24k_to_16k(const std::vector<float> & input, std::vector<float> & output) {
    output.clear();
    const size_t length = input.size();
    if (length == 0) {
        return false;
    }

    const SincKernel & kernel = sinc_kernel();
    const int          width  = kernel.width;
    const int          taps   = kernel.taps_per_phase;

    // Pad `width` zeros on the left and `width + orig_freq` on the right
    // (functional.py:1424).
    const size_t       left_pad      = size_t(width);
    const size_t       right_pad     = size_t(width + kOrigFreq);
    const size_t       padded_length = length + left_pad + right_pad;
    std::vector<float> padded(padded_length, 0.0f);
    std::copy(input.begin(), input.end(), padded.begin() + std::ptrdiff_t(left_pad));

    // target_length = ceil(new_freq * length / orig_freq) (functional.py:1427).
    const uint64_t target_length = uint64_t(std::ceil(double(kNewFreq) * double(length) / double(kOrigFreq)));

    output.assign(size_t(target_length), 0.0f);
    for (uint64_t j = 0; j < target_length; ++j) {
        const size_t block = size_t(j / uint64_t(kNewFreq));
        const size_t phase = size_t(j % uint64_t(kNewFreq));
        const size_t base  = block * size_t(kOrigFreq);
        if (base + size_t(taps) > padded_length) {
            // Cannot happen for the formulas above (conv1d's own output
            // length identity guarantees enough padded samples remain for
            // every j < target_length); guarded rather than assumed.
            output.clear();
            return false;
        }
        const float * tap_row = kernel.taps.data() + phase * size_t(taps);
        float         acc     = 0.0f;
        for (int k = 0; k < taps; ++k) {
            acc += tap_row[k] * padded[base + size_t(k)];
        }
        output[size_t(j)] = acc;
    }
    return true;
}

}  // namespace synth::omnivoice
