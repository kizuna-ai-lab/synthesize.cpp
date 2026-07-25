#include "source.h"

#include "weights.h"

#include <cmath>
#include <cstdio>
#include <new>

namespace synth::kokoro {

namespace {

constexpr double kPi = 3.14159265358979323846;

// Nearest-neighbour upsampling, which is what nn.Upsample defaults to.
void upsample_nearest(const std::vector<float> & input, uint64_t scale, std::vector<float> & output) {
    output.assign(input.size() * scale, 0.0f);
    for (size_t index = 0; index < input.size(); ++index) {
        for (uint64_t step = 0; step < scale; ++step) {
            output[index * scale + step] = input[index];
        }
    }
}

// Linear resampling with PyTorch's align_corners=false mapping, verified
// against F.interpolate: source position is (o + 0.5) * in / out - 0.5.
void resample_linear(const std::vector<float> & input,
                     uint64_t                   in_length,
                     uint64_t                   out_length,
                     uint32_t                   channels,
                     std::vector<float> &       output) {
    output.assign(size_t(out_length) * channels, 0.0f);
    const double ratio = double(in_length) / double(out_length);
    for (uint64_t out_index = 0; out_index < out_length; ++out_index) {
        double position = (double(out_index) + 0.5) * ratio - 0.5;
        if (position < 0.0) {
            position = 0.0;
        }
        const uint64_t low    = uint64_t(position);
        const uint64_t high   = low + 1 < in_length ? low + 1 : in_length - 1;
        const float    weight = float(position - double(low));
        for (uint32_t channel = 0; channel < channels; ++channel) {
            const float a                                  = input[size_t(low) * channels + channel];
            const float b                                  = input[size_t(high) * channels + channel];
            output[size_t(out_index) * channels + channel] = a * (1.0f - weight) + b * weight;
        }
    }
}

}  // namespace

uint32_t source_harmonic_count(const HParams & hparams) {
    return hparams.source.harmonic_num + 1;
}

uint64_t source_upsampled_length(const HParams & hparams, uint64_t f0_length) {
    return f0_length * hparams.source.upsample_scale;
}

synth_status_t build_harmonic_source(const HParams &            hparams,
                                     const std::vector<float> & f0,
                                     const std::vector<float> & merge_weight,
                                     float                      merge_bias,
                                     const SourceRandomInputs & random,
                                     SourceResult &             output) {
    output = SourceResult{};

    const uint32_t harmonics = source_harmonic_count(hparams);
    const uint32_t scale     = hparams.source.upsample_scale;
    const uint32_t n_fft     = hparams.istftnet.gen_istft_n_fft;
    const uint32_t hop       = hparams.istftnet.gen_istft_hop_size;
    if (f0.empty() || harmonics == 0 || scale == 0 || n_fft < 2 || hop == 0) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (hparams.source.sampling_rate == 0) {
        return SYNTH_ERR_INVALID_ARG;
    }
    const uint64_t upsampled = source_upsampled_length(hparams, f0.size());
    if (merge_weight.size() != harmonics || random.rand_ini.size() != harmonics ||
        random.noise.size() != size_t(upsampled) * harmonics) {
        std::fprintf(stderr, "kokoro: harmonic source received replay tensors of the wrong size\n");
        return SYNTH_ERR_INVALID_ARG;
    }

    try {
        std::vector<float> f0_up;
        upsample_nearest(f0, scale, f0_up);

        // Phase increment per sample for each harmonic, wrapped to one turn.
        std::vector<float> radians(size_t(upsampled) * harmonics);
        const float        rate = float(hparams.source.sampling_rate);
        for (uint64_t frame = 0; frame < upsampled; ++frame) {
            for (uint32_t harmonic = 0; harmonic < harmonics; ++harmonic) {
                const float value                             = f0_up[frame] * float(harmonic + 1) / rate;
                radians[size_t(frame) * harmonics + harmonic] = value - std::floor(value);
            }
        }
        // Only the first sample carries the random initial phase.
        for (uint32_t harmonic = 0; harmonic < harmonics; ++harmonic) {
            radians[harmonic] += random.rand_ini[harmonic];
        }

        // Accumulate phase on the coarse grid, then restore the sample rate.
        // Upstream multiplies by the scale because each coarse step stands for
        // that many samples of advance.
        std::vector<float> coarse;
        resample_linear(radians, upsampled, f0.size(), harmonics, coarse);
        std::vector<float> phase(coarse.size());
        std::vector<float> running(harmonics, 0.0f);
        for (uint64_t frame = 0; frame < f0.size(); ++frame) {
            for (uint32_t harmonic = 0; harmonic < harmonics; ++harmonic) {
                running[harmonic] += coarse[size_t(frame) * harmonics + harmonic];
                phase[size_t(frame) * harmonics + harmonic] =
                    float(double(running[harmonic]) * 2.0 * kPi * double(scale));
            }
        }
        std::vector<float> fine;
        resample_linear(phase, f0.size(), upsampled, harmonics, fine);

        // Voiced frames use the small noise floor; unvoiced ones are noise only.
        std::vector<float> excitation(upsampled, 0.0f);
        for (uint64_t frame = 0; frame < upsampled; ++frame) {
            const bool  voiced     = f0_up[frame] > hparams.source.voiced_threshold;
            const float uv         = voiced ? 1.0f : 0.0f;
            const float noise_gain = uv * hparams.source.noise_std + (1.0f - uv) * hparams.source.sine_amp / 3.0f;
            float       merged     = merge_bias;
            for (uint32_t harmonic = 0; harmonic < harmonics; ++harmonic) {
                const size_t offset = size_t(frame) * harmonics + harmonic;
                const float  sine   = std::sin(fine[offset]) * hparams.source.sine_amp;
                const float  value  = sine * uv + noise_gain * random.noise[offset];
                merged += merge_weight[harmonic] * value;
            }
            excitation[frame] = std::tanh(merged);
        }
        output.har_source = excitation;

        // Short-time spectrum: reflect padding, a periodic Hann window, and a
        // real discrete transform, matching torch.stft with center framing.
        const uint32_t half   = n_fft / 2;
        const uint64_t frames = upsampled / hop + 1;
        const uint32_t bins   = half + 1;
        output.frames         = frames;
        output.bins           = bins;

        std::vector<float> padded(upsampled + 2 * half);
        for (uint64_t index = 0; index < padded.size(); ++index) {
            int64_t       source = int64_t(index) - int64_t(half);
            // Reflect without repeating the edge sample.
            const int64_t last   = int64_t(upsampled) - 1;
            while (source < 0 || source > last) {
                source = source < 0 ? -source : 2 * last - source;
            }
            padded[index] = excitation[size_t(source)];
        }

        std::vector<float> window(n_fft);
        for (uint32_t index = 0; index < n_fft; ++index) {
            window[index] = float(0.5 * (1.0 - std::cos(2.0 * kPi * double(index) / double(n_fft))));
        }

        output.har.assign(size_t(frames) * 2 * bins, 0.0f);
        std::vector<float> segment(n_fft);
        for (uint64_t frame = 0; frame < frames; ++frame) {
            for (uint32_t index = 0; index < n_fft; ++index) {
                segment[index] = padded[size_t(frame) * hop + index] * window[index];
            }
            for (uint32_t bin = 0; bin < bins; ++bin) {
                double real = 0.0;
                double imag = 0.0;
                for (uint32_t index = 0; index < n_fft; ++index) {
                    const double angle = -2.0 * kPi * double(bin) * double(index) / double(n_fft);
                    real += double(segment[index]) * std::cos(angle);
                    imag += double(segment[index]) * std::sin(angle);
                }
                const size_t base             = size_t(frame) * 2 * bins;
                output.har[base + bin]        = float(std::hypot(real, imag));
                output.har[base + bins + bin] = float(std::atan2(imag, real));
            }
        }
    } catch (const std::bad_alloc &) {
        output = SourceResult{};
        return SYNTH_ERR_OOM;
    }
    return SYNTH_OK;
}

}  // namespace synth::kokoro
