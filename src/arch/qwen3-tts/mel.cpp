// A 128-bin log-mel front end for the ECAPA-TDNN speaker encoder.
//
// Every convention below is pinned by Task 1's oracle dump
// (build/goldens/qwen3-tts/qwen3-tts-12hz-0-6b-base/base-xvector-en/speaker/
// conventions.json), which ran upstream's own mel_spectrogram and recorded
// what it actually does -- not what a reader might assume from "mel
// spectrogram" alone. None of these six conversions is derivable from the
// package's eight SpeakerEncoderParams fields:
//
//   - mel scale:              Slaney (librosa htk=False), not HTK's log scale.
//   - filterbank norm:        Slaney area normalization (librosa norm='slaney').
//   - spectrum magnitude:     amplitude |X| = sqrt(re^2 + im^2 + 1e-9), not power.
//   - log:                    natural log, floor 1e-5, no pre-log scale.
//   - window:                 periodic Hann, zero-padded and centred into an
//                              n_fft buffer when win_length < n_fft.
//   - padding/centring:       upstream reflect-pads (n_fft - hop_length) // 2
//                              samples per side by hand, then calls
//                              torch.stft(center=False) on the padded signal --
//                              a third convention, distinct from both a naive
//                              center=True and a plainly uncentred transform.
//
// Host DSP rather than a GGML graph: see mel.h's doc comment and the Stage 2
// design's section 5.

#include "mel.h"

#include "weights.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

namespace synth::qwen3tts {

namespace {

// Pinned by conventions.json; none of these six is a SpeakerEncoderParams
// field, and each one silently changes the answer rather than failing.
constexpr double kMagnitudeEpsilon  = 1e-9;  // spectrum_magnitude / magnitude_epsilon
constexpr double kLogFloor          = 1e-5;  // log_floor
// log_scale_before_log: a no-op at this value, kept named so a future package
// that carries a different value is not silently dropped.
constexpr double kLogScaleBeforeLog = 1.0;

bool is_power_of_two(uint32_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

// Iterative radix-2 Cooley-Tukey, in place, on split real/imaginary buffers.
// `n` is a power of two, checked by the caller. Kokoro's two transforms are
// naive O(n^2) DFTs (src/arch/kokoro/source.cpp:163-179,
// src/arch/kokoro/decoder-host.cpp:77); at n_fft 1024 that is about 100x the
// work per frame, which is why this is written rather than reused -- along
// with the fact that neither of them has a mel filterbank, a log, a
// win_length, or a PCM entry point.
void fft_in_place(std::vector<float> & re, std::vector<float> & im) {
    const size_t n = re.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const double angle = -2.0 * M_PI / double(len);
        for (size_t start = 0; start < n; start += len) {
            for (size_t k = 0; k < len / 2; ++k) {
                const double theta = angle * double(k);
                const double wr    = std::cos(theta);
                const double wi    = std::sin(theta);
                const size_t a     = start + k;
                const size_t b     = a + len / 2;
                const double xr    = re[b] * wr - im[b] * wi;
                const double xi    = re[b] * wi + im[b] * wr;
                re[b]              = float(re[a] - xr);
                im[b]              = float(im[a] - xi);
                re[a]              = float(re[a] + xr);
                im[a]              = float(im[a] + xi);
            }
        }
    }
}

// Hz to mel and back, Slaney's piecewise linear-then-log scale -- librosa's
// filters.mel(..., htk=False) default, transcribed from librosa 0.11.0's
// core/convert.py (hz_to_mel/mel_to_hz). conventions.json's `mel_scale` is
// "slaney", not HTK's single 2595*log10(1+f/700): the two differ by several
// bins at 12 kHz and both look like a plausible spectrogram, which is exactly
// why this is pinned by a dump rather than assumed.
constexpr double kMelBreakFrequencyHz = 1000.0;
constexpr double kMelFSp              = 200.0 / 3.0;

double slaney_log_step() {
    return std::log(6.4) / 27.0;
}

double hz_to_mel(double hz) {
    const double min_log_mel = kMelBreakFrequencyHz / kMelFSp;
    if (hz >= kMelBreakFrequencyHz) {
        return min_log_mel + std::log(hz / kMelBreakFrequencyHz) / slaney_log_step();
    }
    return hz / kMelFSp;
}

double mel_to_hz(double mel) {
    const double min_log_mel = kMelBreakFrequencyHz / kMelFSp;
    if (mel >= min_log_mel) {
        return kMelBreakFrequencyHz * std::exp(slaney_log_step() * (mel - min_log_mel));
    }
    return kMelFSp * mel;
}

// Triangular filters over the FFT bins, area-normalized (conventions.json's
// `filterbank_normalization`: "slaney_area" -- librosa's norm='slaney'
// default, which divides each filter by the width of the mel band it covers).
// Transcribed from librosa 0.11.0's filters.mel: fmin/fmax select which
// filters exist at all, so narrowing the band changes every value in the
// filterbank, not its shape.
std::vector<float> build_mel_filterbank(const SpeakerEncoderParams & params) {
    const uint32_t n_freqs = params.n_fft / 2 + 1;
    const uint32_t n_mels  = params.mel_bins;

    std::vector<double> fft_freqs(n_freqs);
    for (uint32_t k = 0; k < n_freqs; ++k) {
        fft_freqs[k] = double(k) * double(params.sample_rate) / double(params.n_fft);
    }

    // librosa.mel_frequencies(n_mels + 2, fmin, fmax): evenly spaced in mel,
    // converted back to Hz. The two extra points are the edges the first and
    // last triangle need beyond their own centre.
    std::vector<double> mel_f(size_t(n_mels) + 2);
    const double        min_mel = hz_to_mel(double(params.fmin));
    const double        max_mel = hz_to_mel(double(params.fmax));
    for (size_t i = 0; i < mel_f.size(); ++i) {
        const double mel = min_mel + (max_mel - min_mel) * double(i) / double(n_mels + 1);
        mel_f[i]         = mel_to_hz(mel);
    }

    std::vector<double> fdiff(n_mels + 1);
    for (uint32_t i = 0; i < n_mels + 1; ++i) {
        fdiff[i] = mel_f[i + 1] - mel_f[i];
    }

    std::vector<float> weights(size_t(n_mels) * n_freqs, 0.0f);
    for (uint32_t bin = 0; bin < n_mels; ++bin) {
        // Slaney area normalization: each filter is scaled by 2 / (width of
        // the mel band it covers), so a wider filter (as at the high end of
        // the Slaney scale) is not simply taller than a narrow one.
        const double enorm = 2.0 / (mel_f[bin + 2] - mel_f[bin]);
        for (uint32_t k = 0; k < n_freqs; ++k) {
            const double lower                 = (fft_freqs[k] - mel_f[bin]) / fdiff[bin];
            const double upper                 = (mel_f[bin + 2] - fft_freqs[k]) / fdiff[bin + 1];
            const double value                 = std::max(0.0, std::min(lower, upper));
            weights[size_t(bin) * n_freqs + k] = float(value * enorm);
        }
    }
    return weights;
}

}  // namespace

synth_status_t compute_log_mel(const SpeakerEncoderParams & params,
                               const std::vector<float> &   pcm,
                               MelSpectrogram &             output) {
    if (params.n_fft == 0 || params.hop_length == 0 || params.win_length == 0 || params.mel_bins == 0) {
        return SYNTH_ERR_UNSUPPORTED_INPUT;
    }
    // The radix-2 transform is the whole reason n_fft must be a power of two;
    // read_speaker_encoder's load-time check is the belt to this braces.
    if (!is_power_of_two(params.n_fft)) {
        return SYNTH_ERR_UNSUPPORTED_INPUT;
    }
    // A window (or a hop) wider than the transform itself has no meaning for
    // the zero-padded-centred rule below.
    if (params.win_length > params.n_fft || params.hop_length > params.n_fft) {
        return SYNTH_ERR_UNSUPPORTED_INPUT;
    }

    // Reject a non-finite input sample before any of it: the Audio Normalizer
    // already refuses non-finite PCM on the ordinary synthesis path, but this
    // function is also called directly by tests and by Plan 3's Voice Profile
    // preparation, which do not necessarily go through that gate first.
    for (float sample : pcm) {
        if (!std::isfinite(sample)) {
            return SYNTH_ERR_INVALID_ARG;
        }
    }

    // conventions.json's `manual_pad_each_side_formula`. Upstream's own
    // reflect pad (F.pad(y, (pad, pad), mode='reflect')) requires the input
    // to strictly exceed `pad` -- not merely reach it -- on pain of raising;
    // conventions.json's `min_pcm_samples` records the boundary it observed
    // (384/385 at the production hop). Rejected here, as a domain error,
    // rather than let our own reflect-pad implementation fail however it
    // fails, and before this ever reaches a frame-count computation.
    const uint32_t pad = (params.n_fft - params.hop_length) / 2;
    if (pcm.size() <= size_t(pad)) {
        return SYNTH_ERR_INVALID_ARG;
    }

    // Manual reflect pad, matching upstream's F.pad(y, (pad, pad),
    // mode='reflect') ahead of an uncentred torch.stft -- see
    // conventions.json's `padding_note`. This is neither a naive centred STFT
    // (which would reflect-pad n_fft/2 each side) nor a plainly uncentred one
    // (no pad at all).
    std::vector<float> padded(pcm.size() + size_t(2) * pad);
    for (size_t i = 0; i < pcm.size(); ++i) {
        padded[pad + i] = pcm[i];
    }
    for (uint32_t j = 0; j < pad; ++j) {
        padded[j] = pcm[pad - j];
    }
    for (uint32_t k = 0; k < pad; ++k) {
        padded[pad + pcm.size() + k] = pcm[pcm.size() - 2 - k];
    }
    // Belt-and-braces for a parameter combination the min_pcm_samples check
    // above does not cover (e.g. hop_length close to n_fft, where `pad` is
    // small): the padded signal must still reach a whole n_fft window.
    if (padded.size() < params.n_fft) {
        return SYNTH_ERR_INVALID_ARG;
    }

    // conventions.json's `frames_formula`, with `centered: false` on the
    // already-padded signal.
    const uint64_t frames = 1 + (uint64_t(padded.size()) - uint64_t(params.n_fft)) / uint64_t(params.hop_length);

    // Periodic Hann (torch.hann_window's default, `window_periodic: true`),
    // zero-padded and centred into an n_fft buffer when win_length < n_fft --
    // torch.stft's own documented contract, not something this checkpoint's
    // own call (win_length == n_fft always) exercises, but a win_length < n_fft
    // window is a zero-padded window, not a shorter transform.
    std::vector<float> window(params.n_fft, 0.0f);
    const uint32_t     window_pad_left = (params.n_fft - params.win_length) / 2;
    for (uint32_t i = 0; i < params.win_length; ++i) {
        const double phase          = 2.0 * M_PI * double(i) / double(params.win_length);
        window[window_pad_left + i] = float(0.5 - 0.5 * std::cos(phase));
    }

    const std::vector<float> filterbank = build_mel_filterbank(params);
    const uint32_t           n_freqs    = params.n_fft / 2 + 1;

    output.bins   = params.mel_bins;
    output.frames = frames;
    output.values.assign(size_t(output.frames) * output.bins, 0.0f);

    std::vector<float> re(params.n_fft);
    std::vector<float> im(params.n_fft);
    std::vector<float> magnitude(n_freqs);

    for (uint64_t frame = 0; frame < frames; ++frame) {
        const size_t start = size_t(frame) * params.hop_length;
        for (uint32_t i = 0; i < params.n_fft; ++i) {
            re[i] = padded[start + i] * window[i];
            im[i] = 0.0f;
        }
        fft_in_place(re, im);

        // spectrum_magnitude: amplitude |X|, not power |X|^2, with epsilon
        // 1e-9 inside the sqrt (conventions.json's `magnitude_epsilon`).
        for (uint32_t k = 0; k < n_freqs; ++k) {
            const double amplitude_sq = double(re[k]) * double(re[k]) + double(im[k]) * double(im[k]);
            magnitude[k]              = float(std::sqrt(amplitude_sq + kMagnitudeEpsilon));
        }

        for (uint32_t bin = 0; bin < params.mel_bins; ++bin) {
            double        energy     = 0.0;
            const float * filter_row = filterbank.data() + size_t(bin) * n_freqs;
            for (uint32_t k = 0; k < n_freqs; ++k) {
                energy += double(filter_row[k]) * double(magnitude[k]);
            }
            // dynamic_range_compression_torch(x, C=1, clip_val=1e-5) =
            // log(clamp(x * C, min=clip_val)): natural log, floor 1e-5. This
            // is the only thing standing between digital silence and -inf.
            const double compressed = std::log(std::max(energy * kLogScaleBeforeLog, kLogFloor));
            output.values[size_t(frame) * output.bins + bin] = float(compressed);
        }
    }
    return SYNTH_OK;
}

}  // namespace synth::qwen3tts
