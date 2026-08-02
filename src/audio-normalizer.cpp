#include "audio-normalizer.h"

#include <samplerate.h>

#include <cmath>
#include <limits>
#include <utility>

namespace synth {

namespace {

bool would_overflow_multiply(uint64_t lhs, uint64_t rhs) {
    return rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs;
}

// samplerate.h's SRC_DATA counts frames with a plain `long`, which is 32 bits
// under Windows' LLP64 model even in a 64-bit build. A frame count that would
// not fit is treated the same as any other frame-count overflow the contract
// names (docs/c-interface.md:486-496): SYNTH_ERR_INVALID_ARG, rather than a
// silently truncated call into the resampler.
constexpr uint64_t kMaxSrcFrames = static_cast<uint64_t>(std::numeric_limits<long>::max());

// Mono<->stereo conversion only: ADR 0009 declares no other channel count
// supported, and the caller has already validated input_channels is in
// {1, 2}; target_channels is a Loaded Model's own declared capability and
// validated the same way just below.
std::vector<float> convert_channels(const float * pcm,
                                    uint64_t      frames,
                                    uint32_t      input_channels,
                                    uint32_t      target_channels) {
    std::vector<float> converted(frames * target_channels);
    if (input_channels == 2 && target_channels == 1) {
        for (uint64_t frame = 0; frame < frames; ++frame) {
            converted[frame] = (pcm[frame * 2] + pcm[frame * 2 + 1]) * 0.5f;
        }
    } else if (input_channels == 1 && target_channels == 2) {
        for (uint64_t frame = 0; frame < frames; ++frame) {
            converted[frame * 2]     = pcm[frame];
            converted[frame * 2 + 1] = pcm[frame];
        }
    }
    return converted;
}

}  // namespace

uint64_t reference_frame_equivalent(uint64_t input_frames, uint32_t input_rate, uint32_t target_rate) {
    if (input_frames == 0 || input_rate == 0 || target_rate == 0) {
        return 0;
    }
    const uint64_t rate   = static_cast<uint64_t>(input_rate);
    const uint64_t target = static_cast<uint64_t>(target_rate);
    const uint64_t product =
        would_overflow_multiply(input_frames, target) ? std::numeric_limits<uint64_t>::max() : input_frames * target;
    // ceil(product / rate) without letting the "+ rate - 1" rounding term overflow.
    const uint64_t rounding_room = std::numeric_limits<uint64_t>::max() - (rate - 1);
    const uint64_t rounded = product > rounding_room ? std::numeric_limits<uint64_t>::max() : product + (rate - 1);
    return rounded / rate;
}

synth_status_t validate_reference_format(uint32_t input_rate, uint32_t input_channels) {
    if (input_rate < SYNTH_REFERENCE_SAMPLE_RATE_MIN || input_rate > SYNTH_REFERENCE_SAMPLE_RATE_MAX ||
        input_channels < 1 || input_channels > SYNTH_REFERENCE_CHANNELS_MAX) {
        return SYNTH_ERR_UNSUPPORTED_INPUT;
    }
    return SYNTH_OK;
}

synth_status_t normalize_reference(const float *         pcm,
                                   uint64_t              frames,
                                   uint32_t              input_rate,
                                   uint32_t              input_channels,
                                   uint32_t              target_rate,
                                   uint32_t              target_channels,
                                   NormalizedReference & output) {
    if (pcm == nullptr || frames == 0) {
        return SYNTH_ERR_INVALID_ARG;
    }
    const synth_status_t format_status = validate_reference_format(input_rate, input_channels);
    if (format_status != SYNTH_OK) {
        return format_status;
    }
    // A Loaded Model's own declared target format is not caller-supplied
    // input, so a malformed one is this module's own precondition failure
    // rather than a rejected Reference Audio clip.
    if (target_rate == 0 || target_channels < 1 || target_channels > 2) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (would_overflow_multiply(frames, input_channels) || frames > kMaxSrcFrames) {
        return SYNTH_ERR_INVALID_ARG;
    }
    const uint64_t total_input_samples = frames * input_channels;
    for (uint64_t index = 0; index < total_input_samples; ++index) {
        if (!std::isfinite(pcm[index])) {
            return SYNTH_ERR_INVALID_ARG;
        }
    }

    const bool channels_match = input_channels == target_channels;
    const bool rate_matches   = input_rate == target_rate;

    if (channels_match && rate_matches) {
        // Both later pipeline steps are no-ops: a plain copy, never routed
        // through the resampler, which is what makes this provably
        // bit-identical to the input.
        output.pcm.assign(pcm, pcm + total_input_samples);
        output.frames = frames;
        return SYNTH_OK;
    }

    // Fixed order (ADR 0009): validate (done above), convert channels, then
    // resample. A step that is already satisfied is skipped rather than run
    // as a costly no-op -- in particular, running a matching sample rate
    // through the sinc filter would not be an identity operation in floating
    // point, and the borrow/exact-arithmetic guarantees above depend on never
    // doing that.
    std::vector<float> channel_converted;
    const float *      stage_pcm      = pcm;
    uint32_t           stage_channels = input_channels;
    if (!channels_match) {
        channel_converted = convert_channels(pcm, frames, input_channels, target_channels);
        stage_pcm         = channel_converted.data();
        stage_channels    = target_channels;
    }

    if (rate_matches) {
        output.pcm.assign(stage_pcm, stage_pcm + frames * stage_channels);
        output.frames = frames;
        return SYNTH_OK;
    }

    const uint64_t rfe = reference_frame_equivalent(frames, input_rate, target_rate);
    if (rfe > kMaxSrcFrames || would_overflow_multiply(rfe, stage_channels)) {
        return SYNTH_ERR_INVALID_ARG;
    }

    // A single src_simple() call, sized to the Reference Frame Equivalent,
    // always fully drains the input for every (rate, channel) combination
    // this module supports: RFE is an upper bound on the frames the sinc
    // converter needs to consume everything at this ratio, never a lower
    // one. Sizing the buffer any smaller risks a partial drain that would be
    // indistinguishable from silently cropping the tail.
    std::vector<float> resampled(rfe * stage_channels);
    SRC_DATA           data{};
    data.data_in       = stage_pcm;
    data.data_out      = resampled.data();
    data.input_frames  = static_cast<long>(frames);
    data.output_frames = static_cast<long>(rfe);
    data.src_ratio     = static_cast<double>(target_rate) / static_cast<double>(input_rate);

    const int error = src_simple(&data, SRC_SINC_BEST_QUALITY, static_cast<int>(stage_channels));
    if (error != 0) {
        return SYNTH_ERR_INTERNAL;
    }

    // The sweep that justified sizing the output buffer to the Reference
    // Frame Equivalent (tests/audio_normalizer_test.cpp, task-8-report.md) is
    // evidence about today's libsamplerate on the rate/channel combinations it
    // covered, not a guarantee this code can lean on forever. If a future
    // libsamplerate version, or some untested rate pair, ever left part of the
    // input undrained, returning SYNTH_OK here would silently crop the
    // signal -- exactly what ADR 0009 forbids ("never trims, truncates, or
    // discards"). Fail loudly instead. This is INTERNAL rather than
    // INVALID_ARG/UNSUPPORTED_INPUT because it is the *library* breaking its
    // own drain contract, not a malformed caller input -- every input shape
    // that could cause that was already rejected above.
    if (static_cast<uint64_t>(data.input_frames_used) != frames) {
        return SYNTH_ERR_INTERNAL;
    }
    // Symmetric check on the other half of the same contract:
    // output_frames_gen must never exceed the capacity (rfe) we gave it.
    // SRC_DATA.output_frames_gen is a signed `long`; a negative value here
    // (impossible per the library's documented contract, but not something
    // this code should trust blindly) would silently become a huge
    // uint64_t once cast below and turn the resize() into a multi-exabyte
    // allocation attempt instead of a clean, diagnosable failure. Note that
    // gen < rfe is expected and fine (ADR 0009: the resampler need not emit
    // exactly the Reference Frame Equivalent) -- only gen > rfe or gen < 0
    // indicate the library returned something this code's own buffer sizing
    // cannot have produced.
    if (data.output_frames_gen < 0 || static_cast<uint64_t>(data.output_frames_gen) > rfe) {
        return SYNTH_ERR_INTERNAL;
    }

    // ADR 0009: retain the actual drained output rather than padding or
    // cropping it to force the Reference Frame Equivalent length.
    const uint64_t produced_frames = static_cast<uint64_t>(data.output_frames_gen);
    resampled.resize(produced_frames * stage_channels);
    output.pcm    = std::move(resampled);
    output.frames = produced_frames;
    return SYNTH_OK;
}

}  // namespace synth
