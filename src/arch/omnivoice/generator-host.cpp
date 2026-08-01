#include "arch/omnivoice/generator-host.h"

#include "random-stream.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
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
        // Upstream's `_get_time_steps` runs this formula over a
        // `torch.linspace(0, 1, num_step + 1)` tensor, which is float32 by
        // default (torch 2.13.0) -- reproducing that dtype is not optional.
        // Sweeping total_mask over [1, 6000] found the double-precision and
        // float32 schedules disagree at roughly 1% of canvas lengths (e.g.
        // num_step 32: T in {205, 247, 295, 407, 410, ...}); a divergent step
        // commits a different set of positions, which is a different token
        // grid, and no committed Golden case happens to land on a divergent
        // T, so only this fixture -- not the exact-token gate -- would ever
        // catch a regression here. So every arithmetic step below is float32,
        // cast to double only once at the very end for commit_schedule's own
        // (double) arithmetic.
        //
        // Caveat: `float(index / num_step)` coincides with torch.linspace's
        // own start + step * i kernel only because 1 / num_step is exact in
        // binary floating point for this family's num_step values (16, 32).
        // If num_step ever becomes a free knob, this must switch to
        // linspace's own construction (or refuse a num_step for which
        // 1 / num_step is not exact) rather than trust this shortcut.
        const float t           = num_step == 0 ? 0.0f : float(double(index) / double(num_step));
        const float numerator   = float(t_shift) * t;
        const float denominator = 1.0f + float(t_shift - 1.0) * t;
        timesteps[index]        = double(numerator / denominator);
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

// The guided combination shared by `choose_token` and `choose_token_sampled`:
// log_softmax -> combine -> log_softmax -> mask ban. `guided` must already be
// sized to the vocabulary (its size is the only place vocab_size comes from
// here); this is the one place either caller computes the guided array, so
// the two decode paths cannot drift apart from each other.
//
// Carryover item 1: `uncond` may be nullptr only when guidance_scale == 0
// (the reference's own `guidance_scale != 0` branch). This used to be prose
// on choose_token's header comment alone; asserting it here enforces the
// same contract for both callers instead of documenting it once and hoping.
void build_guided(const float *        cond,
                  const float *        uncond,
                  uint32_t             mask_id,
                  float                guidance_scale,
                  std::vector<float> & guided) {
    assert(!(guidance_scale != 0.0f && uncond == nullptr) &&
           "choose_token contract: uncond must be non-null when guidance_scale != 0");
    const uint32_t vocab_size = uint32_t(guided.size());

    thread_local std::vector<float> cond_lp;
    thread_local std::vector<float> uncond_lp;
    cond_lp.resize(vocab_size);

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
    // commit a mask; everything else competes. Guarded: this helper is
    // unit-tested standalone with toy vocabularies, and a reader's guarantee
    // that mask_id < vocab_size does not apply here.
    if (mask_id < vocab_size) {
        guided[mask_id] = -std::numeric_limits<float>::infinity();
    }
}

}  // namespace

void choose_token(const float * cond,
                  const float * uncond,
                  uint32_t      vocab_size,
                  uint32_t      mask_id,
                  float         guidance_scale,
                  int32_t &     token,
                  float &       log_prob,
                  float *       runner_up_gap) {
    thread_local std::vector<float> guided;
    guided.resize(vocab_size);
    build_guided(cond, uncond, mask_id, guidance_scale, guided);

    int32_t best         = 0;
    float   best_value   = -std::numeric_limits<float>::infinity();
    float   second_value = -std::numeric_limits<float>::infinity();
    for (uint32_t index = 0; index < vocab_size; ++index) {
        if (guided[index] > best_value) {
            second_value = best_value;
            best_value   = guided[index];
            best         = int32_t(index);
        } else if (guided[index] > second_value) {
            second_value = guided[index];
        }
    }
    token    = best;
    log_prob = best_value;
    if (runner_up_gap != nullptr) {
        // Two -inf values subtract to NaN and a lone finite best has no rival
        // to be close to; both mean "nothing contested this token", which reads
        // as an infinite gap rather than as a zero one.
        *runner_up_gap = (std::isfinite(best_value) && std::isfinite(second_value)) ?
                             best_value - second_value :
                             std::numeric_limits<float>::infinity();
    }
}

float gumbel_perturb(float logit, float temperature, float uniform) {
    // Upstream's `_gumbel_sample` (omnivoice.py:1632-1636), one value: scaled
    // = logit / temperature; g = -log(-log(u + 1e-10) + 1e-10); result =
    // scaled + g. Float32 throughout -- no double intermediate anywhere in
    // this expression, and the grouping matches upstream's line-for-line
    // rather than an algebraically equivalent rearrangement, because a
    // reordering changes the rounding and therefore which token an argmax
    // over several perturbed values picks.
    const float scaled = logit / temperature;
    const float noise  = -std::log(-std::log(uniform + 1e-10f) + 1e-10f);
    return scaled + noise;
}

synth_status_t choose_token_sampled(const float *        cond,
                                    const float *        uncond,
                                    uint32_t             vocab_size,
                                    uint32_t             mask_id,
                                    float                guidance_scale,
                                    float                class_temperature,
                                    NormalRandomStream & stream,
                                    int32_t &            token,
                                    float &              log_prob) {
    if (class_temperature == 0.0f) {
        // Matches upstream's `if class_temperature > 0.0` split exactly: no
        // top-k filter, no stream draws, the existing greedy path decides.
        choose_token(cond, uncond, vocab_size, mask_id, guidance_scale, token, log_prob);
        return SYNTH_OK;
    }

    thread_local std::vector<float> guided;
    guided.resize(vocab_size);
    build_guided(cond, uncond, mask_id, guidance_scale, guided);

    // Upstream's `_filter_top_k`: keep = ceil(0.1 * vocab_size) largest
    // guided values (the mask entry is already -inf from build_guided's ban,
    // so it can never be among them). Computed as integer ceiling division
    // by 10 rather than double(0.1) * vocab_size, so the result is exact for
    // every integer vocab_size instead of depending on how 0.1's binary
    // rounding happens to fall relative to a .5 boundary.
    const uint32_t keep = std::min((vocab_size + 9) / 10, vocab_size);

    // Ties break toward the lower class id, matching this port's argmax
    // first-maximal convention elsewhere (see choose_token above).
    thread_local std::vector<uint32_t> order;
    order.resize(vocab_size);
    for (uint32_t index = 0; index < vocab_size; ++index) {
        order[index] = index;
    }
    // `guided` has thread-local storage duration, so it needs no capture --
    // it is already reachable from the lambda body the way a global would be.
    const auto ranks_before = [](uint32_t left, uint32_t right) {
        return guided[left] != guided[right] ? guided[left] > guided[right] : left < right;
    };
    std::partial_sort(order.begin(), order.begin() + keep, order.end(), ranks_before);
    // The survivors, now in ascending class-id order for the draw: a
    // port-defined order, since upstream draws one dense `rand_like` over the
    // whole row at once rather than one uniform per surviving class.
    std::sort(order.begin(), order.begin() + keep);

    int32_t best        = -1;
    float   best_score  = -std::numeric_limits<float>::infinity();
    float   best_guided = -std::numeric_limits<float>::infinity();
    for (uint32_t index = 0; index < keep; ++index) {
        const uint32_t class_id = order[index];
        const float    uniform  = stream.next_uniform();
        const float    score    = gumbel_perturb(guided[class_id], class_temperature, uniform);
        if (score > best_score) {
            best_score  = score;
            best        = int32_t(class_id);
            best_guided = guided[class_id];
        }
    }
    token    = best;
    log_prob = best_guided;
    return SYNTH_OK;
}

bool commits_before(const MaskedCandidate & left, const MaskedCandidate & right) {
    // A NaN score compares false against everything, including itself
    // (`left.score != right.score` is true for a NaN vs. anything, even
    // another NaN), so without this guard the plain score branch below
    // makes this order non-transitive across a NaN candidate -- undefined
    // behavior for std::partial_sort, not just a wrong order. Route NaN
    // scores after every real one, and let two NaN-scored candidates fall
    // through to the codebook/frame tie-break as if they were equal.
    const bool left_nan  = left.score != left.score;
    const bool right_nan = right.score != right.score;
    if (left_nan != right_nan) {
        return right_nan;
    }
    if (!left_nan && left.score != right.score) {
        return left.score > right.score;
    }
    if (left.codebook != right.codebook) {
        return left.codebook < right.codebook;
    }
    return left.frame < right.frame;
}

size_t select_commits(std::vector<MaskedCandidate> & candidates, uint64_t budget) {
    const size_t keep = size_t(std::min<uint64_t>(budget, candidates.size()));
    std::partial_sort(candidates.begin(), candidates.begin() + keep, candidates.end(), commits_before);
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

void note_margin(MarginReport &     report,
                 MarginReport::Kind kind,
                 float              value,
                 uint32_t           step,
                 uint32_t           codebook,
                 uint64_t           frame) {
    if (report.measured) {
        const bool already_broken = std::isnan(report.value);
        const bool narrower       = std::isnan(value) || value < report.value;
        if (already_broken || !narrower) {
            return;
        }
    }
    report.measured = true;
    report.kind     = kind;
    report.value    = value;
    report.step     = step;
    report.codebook = codebook;
    report.frame    = frame;
}

std::string margin_value_json(float value) {
    if (std::isnan(value)) {
        return "NaN";
    }
    if (std::isinf(value)) {
        return value > 0.0f ? "Infinity" : "-Infinity";
    }
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.9g", double(value));
    return buffer;
}

}  // namespace synth::omnivoice
