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
constexpr double kRolloffD           = 0.99;

// base_freq = min(orig_freq, new_freq) * rolloff = min(3, 2) * 0.99 = 1.98.
// This is plain Python arithmetic in torchaudio (functional.py:1345-1350) --
// no tensor exists yet at this point in _get_sinc_resample_kernel -- so it
// stays double-precision here too, exactly like width below.
constexpr double kBaseFreqD = double(kOrigFreq < kNewFreq ? kOrigFreq : kNewFreq) * kRolloffD;

// width = ceil(lowpass_filter_width * orig_freq / base_freq)
//       = ceil(6 * 3 / 1.98) = ceil(9.0909...) = 10 (measured against the
// pinned torchaudio in the locked venv -- see the header comment). Also
// plain double-precision Python arithmetic (functional.py:1369).
int sinc_kernel_width() {
    return static_cast<int>(std::ceil(double(kLowpassFilterWidth) * double(kOrigFreq) / kBaseFreqD));
}

// From here down, every one of these constants is combined with a float32
// "tensor" value inside _get_sinc_resample_kernel. PyTorch's tensor-scalar
// op for float32_tensor op python_number rounds the scalar down to float32
// FIRST, then computes entirely in float32 -- there is no double
// intermediate anywhere in the kernel-construction loop below (see the
// header for how this was verified and why an earlier revision of this file
// had it backwards). kBaseFreqF and kScaleF are exactly kBaseFreqD and
// scale rounded to float32 once, at the point they stop being pure-Python
// scalars and start being multiplied into tensor-shaped values.
constexpr float kBaseFreqF = float(kBaseFreqD);
// scale = base_freq / orig_freq = 1.98 / 3 = 0.66, likewise computed in
// double (pure Python, functional.py:1395) then rounded to float32 once.
constexpr float kScaleF    = float(kBaseFreqD / double(kOrigFreq));
constexpr float kPiF       = 3.14159265358979323846f;

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
                // idx[k] = float32((k - width) / orig_freq) (functional.py:1376).
                // orig_freq (a Python int) is exactly representable in
                // float32, so casting it to float here IS "round to float32
                // first" -- no separate rounding step is needed.
                const float idx = float(k - built.width) / float(kOrigFreq);
                // t = float32(-phase / new_freq) + idx[k] (functional.py:1378).
                float       t   = float(-phase) / float(kNewFreq) + idx;
                // t *= base_freq (functional.py:1379) -- float32 * float32.
                t               = t * kBaseFreqF;
                // t = clamp(t, -lowpass_filter_width, lowpass_filter_width) (functional.py:1380).
                t               = std::clamp(t, -float(kLowpassFilterWidth), float(kLowpassFilterWidth));

                // window = cos(((t * pi) / lowpass_filter_width) / 2) ^ 2 (functional.py:1385).
                const float t_pi   = t * kPiF;
                const float w_arg1 = t_pi / float(kLowpassFilterWidth);
                const float w_arg2 = w_arg1 / 2.0f;
                const float cosine = std::cos(w_arg2);
                const float window = cosine * cosine;

                // t_rad = t * pi (functional.py:1393 reassigns t -- the same
                // expression as t_pi above, so reused rather than recomputed).
                const float t_rad = t_pi;
                // sinc = t_rad == 0 ? 1.0 : sin(t_rad) / t_rad (functional.py:1396).
                const float sinc  = (t_rad == 0.0f) ? 1.0f : std::sin(t_rad) / t_rad;

                // kernel = sinc * (window * scale) (functional.py:1397) -- both
                // multiplies are float32 * float32.
                const float scaled_window                                            = window * kScaleF;
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

    // target_length = torch.ceil(torch.as_tensor(new_freq * length /
    // orig_freq)).long() (functional.py:1427). `new_freq * length /
    // orig_freq` is plain Python int/true-division arithmetic (double
    // precision) -- but torch.as_tensor's default dtype is float32, so the
    // pre-ceil value is downcast to float32 BEFORE ceiling, not ceiled
    // directly in double. Checked against a plain double-then-ceil model
    // for every length in [1, 200000] and a sparse sample up to 3,000,000:
    // zero divergence (this module's own reference clips cap well under
    // 500,000 samples), but the downcast is transcribed anyway so this file
    // carries no unverified promotion gap.
    const double   pre_ceil_d    = double(kNewFreq) * double(length) / double(kOrigFreq);
    const float    pre_ceil_f    = float(pre_ceil_d);
    const uint64_t target_length = uint64_t(std::ceil(pre_ceil_f));

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
