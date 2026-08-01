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
    if (input_rate < 8000 || input_rate > 192000 || input_channels < 1 || input_channels > 2) {
        return SYNTH_ERR_UNSUPPORTED_INPUT;
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

    // ADR 0009: retain the actual drained output rather than padding or
    // cropping it to force the Reference Frame Equivalent length.
    const uint64_t produced_frames = static_cast<uint64_t>(data.output_frames_gen);
    resampled.resize(produced_frames * stage_channels);
    output.pcm    = std::move(resampled);
    output.frames = produced_frames;
    return SYNTH_OK;
}

}  // namespace synth
