#include "decoder-host.h"

#include "weights.h"

#include <cmath>
#include <cstdio>
#include <new>

namespace synth::kokoro {

namespace {

constexpr double kPi = 3.14159265358979323846;

// Below this the overlap-added window carries no signal to recover, which is
// the same guard PyTorch's inverse transform applies.
constexpr float kEnvelopeFloor = 1e-11f;

}  // namespace

uint64_t inverse_stft_length(const HParams & hparams, uint64_t frames) {
    if (frames == 0) {
        return 0;
    }
    return (frames - 1) * hparams.istftnet.gen_istft_hop_size;
}

synth_status_t inverse_stft(const HParams &            hparams,
                            const std::vector<float> & spectrum,
                            uint64_t                   frames,
                            std::vector<float> &       audio) {
    audio.clear();

    const uint32_t n_fft = hparams.istftnet.gen_istft_n_fft;
    const uint32_t hop   = hparams.istftnet.gen_istft_hop_size;
    if (frames == 0 || n_fft < 2 || n_fft % 2 != 0 || hop == 0) {
        return SYNTH_ERR_INVALID_ARG;
    }
    const uint32_t bins = n_fft / 2 + 1;
    const uint32_t rows = n_fft + 2;
    if (spectrum.size() != size_t(frames) * rows) {
        std::fprintf(stderr, "kokoro: inverse transform received a spectrum of the wrong size\n");
        return SYNTH_ERR_INVALID_ARG;
    }

    const uint64_t total = uint64_t(n_fft) + uint64_t(hop) * (frames - 1);
    const uint32_t half  = n_fft / 2;
    const uint64_t kept  = inverse_stft_length(hparams, frames);
    if (total < uint64_t(n_fft) || kept == 0) {
        return SYNTH_ERR_INVALID_ARG;
    }

    try {
        std::vector<float> window(n_fft);
        for (uint32_t index = 0; index < n_fft; ++index) {
            window[index] = float(0.5 * (1.0 - std::cos(2.0 * kPi * double(index) / double(n_fft))));
        }

        std::vector<double> accumulated(size_t(total), 0.0);
        std::vector<double> envelope(size_t(total), 0.0);
        std::vector<double> frame(n_fft);

        for (uint64_t index = 0; index < frames; ++index) {
            const float * row = spectrum.data() + size_t(index) * rows;
            // The generator emits a logarithm and an angle; both activations
            // belong with the transform that consumes them.
            for (uint32_t sample = 0; sample < n_fft; ++sample) {
                double value = 0.0;
                for (uint32_t bin = 0; bin < bins; ++bin) {
                    const double magnitude = std::exp(double(row[bin]));
                    const double angle     = std::sin(double(row[bins + bin]));
                    const double turn      = 2.0 * kPi * double(bin) * double(sample) / double(n_fft);
                    const double real      = magnitude * std::cos(angle);
                    const double imaginary = magnitude * std::sin(angle);
                    // Bins zero and the Nyquist bin have no mirrored partner,
                    // so they contribute once and only through their real part.
                    const double weight    = (bin == 0 || bin == bins - 1) ? 1.0 : 2.0;
                    value += weight * (real * std::cos(turn) - imaginary * std::sin(turn));
                }
                frame[sample] = value / double(n_fft);
            }
            for (uint32_t sample = 0; sample < n_fft; ++sample) {
                const size_t position = size_t(index) * hop + sample;
                accumulated[position] += frame[sample] * double(window[sample]);
                envelope[position] += double(window[sample]) * double(window[sample]);
            }
        }

        audio.assign(size_t(kept), 0.0f);
        for (uint64_t sample = 0; sample < kept; ++sample) {
            const size_t position = size_t(sample) + half;
            const double divisor  = envelope[position];
            audio[size_t(sample)] =
                std::fabs(divisor) < double(kEnvelopeFloor) ? 0.0f : float(accumulated[position] / divisor);
        }
    } catch (const std::bad_alloc &) {
        audio.clear();
        return SYNTH_ERR_OOM;
    }
    return SYNTH_OK;
}

}  // namespace synth::kokoro
