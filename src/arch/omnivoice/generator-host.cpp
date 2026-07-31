#include "arch/omnivoice/generator-host.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace synth::omnivoice {

synth_status_t build_prompt_grid(const std::vector<int32_t> & text_ids,
                                 const std::vector<int32_t> & reference_tokens,
                                 uint64_t                     target_frames,
                                 uint32_t                     num_codebooks,
                                 uint32_t                     mask_id,
                                 PromptLayout &               output) {
    output = PromptLayout{};
    if (text_ids.empty() || target_frames == 0 || num_codebooks == 0) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (reference_tokens.size() % num_codebooks != 0) {
        return SYNTH_ERR_INVALID_ARG;
    }
    const uint64_t reference_frames = reference_tokens.size() / num_codebooks;
    for (int32_t token : reference_tokens) {
        // A reference token is a committed code; the mask id inside one would
        // mean the encoder emitted a hole, which is a caller bug, not a state.
        if (token < 0 || uint32_t(token) >= mask_id) {
            return SYNTH_ERR_INVALID_ARG;
        }
    }

    output.text_length      = text_ids.size();
    output.reference_frames = reference_frames;
    output.target_frames    = target_frames;
    output.num_codebooks    = num_codebooks;
    const uint64_t total    = output.total();
    output.grid.assign(size_t(num_codebooks) * total, int32_t(mask_id));
    for (uint32_t codebook = 0; codebook < num_codebooks; ++codebook) {
        int32_t * row = output.grid.data() + size_t(codebook) * total;
        // Every row of the codebook dimension repeats the same text ids; the
        // rows differ only where audio tokens live.
        std::copy(text_ids.begin(), text_ids.end(), row);
        const int32_t * reference = reference_tokens.data() + size_t(codebook) * reference_frames;
        std::copy(reference, reference + reference_frames, row + output.text_length);
        // The target region stays at mask_id from the assign above.
    }
    return SYNTH_OK;
}

std::vector<double> shifted_timesteps(uint32_t num_step, double t_shift) {
    std::vector<double> timesteps(size_t(num_step) + 1);
    for (uint32_t index = 0; index <= num_step; ++index) {
        const double t = num_step == 0 ? 0.0 : double(index) / double(num_step);
        // The formula's true value at t = 1 is t_shift / t_shift = 1 for any
        // nonzero t_shift, but the naive division loses that exactly: most
        // t_shift values (0.1 included) are not exact in binary floating
        // point, so `1.0 + (t_shift - 1.0)` does not recover t_shift bit for
        // bit and the quotient lands a couple of ulps off 1.0. The endpoint
        // is pinned directly rather than left to drift.
        if (t == 1.0 && t_shift != 0.0) {
            timesteps[index] = 1.0;
        } else {
            timesteps[index] = t_shift * t / (1.0 + (t_shift - 1.0) * t);
        }
    }
    return timesteps;
}

std::vector<uint64_t> commit_schedule(uint64_t total_mask, uint32_t num_step, double t_shift) {
    const std::vector<double> timesteps = shifted_timesteps(num_step, t_shift);
    std::vector<uint64_t>     schedule(num_step, 0);
    uint64_t                  remaining = total_mask;
    for (uint32_t step = 0; step < num_step; ++step) {
        uint64_t count;
        if (step + 1 == num_step) {
            // The final step commits the entire remainder, so the grid is
            // always fully committed regardless of rounding.
            count = remaining;
        } else {
            const double share = double(total_mask) * (timesteps[step + 1] - timesteps[step]);
            count              = std::min(uint64_t(std::ceil(share)), remaining);
        }
        schedule[step] = count;
        remaining -= count;
    }
    return schedule;
}

namespace {

// Numerically stable log-softmax into `output`. The max subtraction and the
// double accumulator keep the exp sum honest; the argmax downstream depends
// only on monotone shifts, so this is about the confidence value, not the
// winner.
void log_softmax(const float * input, uint32_t count, float * output) {
    float highest = -std::numeric_limits<float>::infinity();
    for (uint32_t index = 0; index < count; ++index) {
        highest = std::max(highest, input[index]);
    }
    double total = 0.0;
    for (uint32_t index = 0; index < count; ++index) {
        total += std::exp(double(input[index]) - highest);
    }
    const float log_total = float(std::log(total));
    for (uint32_t index = 0; index < count; ++index) {
        output[index] = input[index] - highest - log_total;
    }
}

}  // namespace

void choose_token(const float * cond,
                  const float * uncond,
                  uint32_t      vocab_size,
                  uint32_t      mask_id,
                  float         guidance_scale,
                  int32_t &     token,
                  float &       log_prob) {
    thread_local std::vector<float> cond_lp;
    thread_local std::vector<float> uncond_lp;
    thread_local std::vector<float> guided;
    cond_lp.resize(vocab_size);
    guided.resize(vocab_size);

    log_softmax(cond, vocab_size, cond_lp.data());
    if (guidance_scale != 0.0f && uncond != nullptr) {
        uncond_lp.resize(vocab_size);
        log_softmax(uncond, vocab_size, uncond_lp.data());
        // log_softmax(log_pc + s * (log_pc - log_pu)): the second log-softmax
        // is a constant shift per position, which the confidence value (not
        // the argmax) depends on.
        for (uint32_t index = 0; index < vocab_size; ++index) {
            guided[index] = cond_lp[index] + guidance_scale * (cond_lp[index] - uncond_lp[index]);
        }
        log_softmax(guided.data(), vocab_size, guided.data());
    } else {
        std::copy(cond_lp.begin(), cond_lp.end(), guided.begin());
    }

    // The mask id is banned AFTER the combination, so the model can never
    // commit a mask; everything else competes. Guarded: this function is
    // unit-tested standalone with toy vocabularies, and a reader's guarantee
    // that mask_id < vocab_size does not apply here.
    if (mask_id < vocab_size) {
        guided[mask_id] = -std::numeric_limits<float>::infinity();
    }

    int32_t best       = 0;
    float   best_value = -std::numeric_limits<float>::infinity();
    for (uint32_t index = 0; index < vocab_size; ++index) {
        if (guided[index] > best_value) {
            best_value = guided[index];
            best       = int32_t(index);
        }
    }
    token    = best;
    log_prob = best_value;
}

size_t select_commits(std::vector<MaskedCandidate> & candidates, uint64_t budget) {
    const size_t keep   = size_t(std::min<uint64_t>(budget, candidates.size()));
    const auto   before = [](const MaskedCandidate & left, const MaskedCandidate & right) {
        if (left.score != right.score) {
            return left.score > right.score;
        }
        if (left.codebook != right.codebook) {
            return left.codebook < right.codebook;
        }
        return left.frame < right.frame;
    };
    std::partial_sort(candidates.begin(), candidates.begin() + keep, candidates.end(), before);
    return keep;
}

void fill_shifted_audio_ids(const int32_t *        grid,
                            uint64_t               row_stride,
                            uint64_t               offset,
                            uint64_t               count,
                            uint32_t               num_codebooks,
                            uint32_t               vocab_size,
                            std::vector<int32_t> & output) {
    output.resize(size_t(num_codebooks) * count);
    for (uint32_t codebook = 0; codebook < num_codebooks; ++codebook) {
        const int32_t shift = int32_t(codebook * vocab_size);
        for (uint64_t index = 0; index < count; ++index) {
            output[size_t(codebook) * count + index] = grid[size_t(codebook) * row_stride + offset + index] + shift;
        }
    }
}

}  // namespace synth::omnivoice
