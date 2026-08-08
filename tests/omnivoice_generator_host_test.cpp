// The generator host: prompt grid, timestep schedule, CFG scoring and the
// flat top-k commit -- everything discrete in the mask-predict decode loop,
// off the graph. Every expected value here is hand-computable and is
// transcribed rather than derived, per the discrete-outputs placement rule:
// a wrong rule produces a grid the codec still decodes, and only the
// exact-token gate would ever notice a wrong value slipping through.

#include "arch/omnivoice/generator-host.h"
#include "test-assert.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace {

using synth::omnivoice::MaskedCandidate;
using synth::omnivoice::PromptLayout;

// --------------------------------------------------------------------------
// commit_schedule / shifted_timesteps
// --------------------------------------------------------------------------

// t' = 0.5t/(1 - 0.5t) over linspace(0,1,5): {0, 1/7, 1/3, 3/5, 1}.
// ceil(10*delta) gives {2, 2, 3}, and the final step takes the remainder 3.
int check_schedule_hand_fixture() {
    const std::vector<uint64_t> schedule = synth::omnivoice::commit_schedule(10, 4, 0.5);
    SYNTH_TEST_CHECK(schedule.size() == 4);
    SYNTH_TEST_CHECK(schedule[0] == 2);
    SYNTH_TEST_CHECK(schedule[1] == 2);
    SYNTH_TEST_CHECK(schedule[2] == 3);
    SYNTH_TEST_CHECK(schedule[3] == 3);
    uint64_t sum = 0;
    for (uint64_t count : schedule) {
        sum += count;
    }
    SYNTH_TEST_CHECK(sum == 10);
    return 0;
}

// t_shift 1.0 is the identity shift: timesteps {0, .25, .5, .75, 1}, so every
// non-final delta is 0.25 and ceil(2*0.25) = 1. total_mask 2 runs out after
// the first two steps, so the mid-schedule clamp to the remainder is what
// keeps steps 2 and 3 at zero rather than going negative.
int check_schedule_clamps() {
    const std::vector<uint64_t> schedule = synth::omnivoice::commit_schedule(2, 4, 1.0);
    SYNTH_TEST_CHECK(schedule.size() == 4);
    SYNTH_TEST_CHECK(schedule[0] == 1);
    SYNTH_TEST_CHECK(schedule[1] == 1);
    SYNTH_TEST_CHECK(schedule[2] == 0);
    SYNTH_TEST_CHECK(schedule[3] == 0);
    return 0;
}

// The real decode-loop parameters: a 50-frame canvas (8 codebooks * 50 = 400
// masked positions), 32 steps, t_shift 0.1.
int check_schedule_real_parameters() {
    constexpr uint64_t kTotalMask = 400;
    constexpr uint32_t kNumStep   = 32;
    constexpr double   kTShift    = 0.1;

    const std::vector<double>   timesteps = synth::omnivoice::shifted_timesteps(kNumStep, kTShift);
    const std::vector<uint64_t> schedule  = synth::omnivoice::commit_schedule(kTotalMask, kNumStep, kTShift);

    SYNTH_TEST_CHECK(timesteps.size() == kNumStep + 1);
    SYNTH_TEST_CHECK(timesteps.front() == 0.0);
    // Not 1.0: computed in float32 to match upstream's torch.linspace tensor
    // dtype, t_shift 0.1 is not exact in binary floating point, and the
    // float32 quotient at t = 1 lands a few ulps short of 1.0 rather than
    // recovering it exactly. Independently recomputed (Python, struct-based
    // float32 simulation of the exact formula below) and its double-to-hex
    // round-trip verified bit-exact against this literal.
    SYNTH_TEST_CHECK(timesteps.back() == 0.9999997615814209);

    SYNTH_TEST_CHECK(schedule.size() == kNumStep);
    uint64_t sum = 0;
    for (uint64_t count : schedule) {
        sum += count;
    }
    SYNTH_TEST_CHECK(sum == kTotalMask);

    // t_shift 0.1 back-loads the budget: the intervals grow monotonically, so
    // the last step commits the most.
    uint64_t largest = 0;
    for (uint64_t count : schedule) {
        largest = std::max(largest, count);
    }
    SYNTH_TEST_CHECK(schedule.back() == largest);
    return 0;
}

// A regression fixture anchored to the float32/float64 divergence: at these
// parameters, computing shifted_timesteps in double instead of float32 (this
// family's actual failure mode, caught in review) changes the last two
// entries from {256, 384} to {257, 383} -- a different commit count at step
// 30, which is a different set of committed positions and a different token
// grid downstream. Independently recomputed (Python, struct-based float32
// simulation of the exact formula in generator-host.cpp): float32 gives
// {..., 131, 178, 256, 384}; a double-precision version of the same formula
// gives {..., 131, 178, 257, 383}. Both sum to 1640, so the sum invariant
// alone cannot see this defect -- only the exact per-step values can.
int check_schedule_upstream_regression() {
    const std::vector<uint64_t> schedule = synth::omnivoice::commit_schedule(1640, 32, 0.1);
    SYNTH_TEST_CHECK(schedule.size() == 32);
    SYNTH_TEST_CHECK(schedule[28] == 131);
    SYNTH_TEST_CHECK(schedule[29] == 178);
    SYNTH_TEST_CHECK(schedule[30] == 256);
    SYNTH_TEST_CHECK(schedule[31] == 384);
    uint64_t sum = 0;
    for (uint64_t count : schedule) {
        sum += count;
    }
    SYNTH_TEST_CHECK(sum == 1640);
    return 0;
}

// The OTHER step count this family runs. `num_step` is a Synthesis Request
// field (omnivoice.h) that falls back to the package's embedded default (32)
// when it is 0; the Golden suite's `omni-fast-mode` case sets it to 16, and
// 16 is the value a fast profile would use. It is pinned here on the canvas
// this project actually measures the step count against -- `omni-long-boundary`,
// 719 frames x 8 codebooks = 5,752 masked positions -- because the 16-step
// schedule has no other fixture and the 32-step one cannot stand in for it:
// shifted_timesteps' float32 shortcut is justified per-num_step (see
// generator-host.cpp's comment), so a regression could move one schedule and
// not the other.
//
// Every entry is transcribed from an independent recomputation (Python,
// struct-based float32 simulation of the exact formula in generator-host.cpp)
// rather than from a run of this code.
//
// What the values say about the mechanism, recorded because no other artifact
// in the tree does: halving the step count does not halve the risk evenly. At
// 16 steps the FINAL forward jointly commits 2,294 of 5,752 positions (39.88%)
// from one set of logits with no further refinement, against 1,388 (24.13%) at
// 32; the last two steps commit 58.71% against 39.44%. The schedule is more
// lopsided, not merely shorter.
int check_schedule_sixteen_step_real_parameters() {
    const std::vector<uint64_t> schedule = synth::omnivoice::commit_schedule(5752, 16, 0.1);
    const uint64_t expected[16] = { 39, 43, 49, 56, 65, 76, 90, 108, 133, 167, 216, 291, 412, 630, 1083, 2294 };
    SYNTH_TEST_CHECK(schedule.size() == 16);
    uint64_t sum = 0;
    for (size_t step = 0; step < schedule.size(); ++step) {
        SYNTH_TEST_CHECK(schedule[step] == expected[step]);
        sum += schedule[step];
    }
    SYNTH_TEST_CHECK(sum == 5752);
    // Back-loaded, like the 32-step schedule: monotone growth, last step largest.
    for (size_t step = 1; step < schedule.size(); ++step) {
        SYNTH_TEST_CHECK(schedule[step] >= schedule[step - 1]);
    }

    // The 50-frame canvas is `omni-fast-mode`'s own (8 x 50 = 400), the one
    // case in the suite whose oracle is a 16-step dump. That case's committed
    // grid is the only end-to-end check the 16-step path has, so its schedule
    // is pinned here beside it.
    const std::vector<uint64_t> fast_mode         = synth::omnivoice::commit_schedule(400, 16, 0.1);
    const uint64_t              fast_expected[16] = { 3, 3, 4, 4, 5, 6, 7, 8, 10, 12, 15, 21, 29, 44, 76, 153 };
    SYNTH_TEST_CHECK(fast_mode.size() == 16);
    uint64_t fast_sum = 0;
    for (size_t step = 0; step < fast_mode.size(); ++step) {
        SYNTH_TEST_CHECK(fast_mode[step] == fast_expected[step]);
        fast_sum += fast_mode[step];
    }
    SYNTH_TEST_CHECK(fast_sum == 400);

    // 16 is one of the two num_step values for which shifted_timesteps'
    // `float(index / num_step)` shortcut is exact (1/16 and 1/32 are both
    // exact in binary floating point). The endpoint therefore lands the same
    // few ulps short of 1.0 that the 32-step schedule's does, and for the same
    // reason -- t_shift 0.1 is not exact, not the linspace.
    const std::vector<double> timesteps = synth::omnivoice::shifted_timesteps(16, 0.1);
    SYNTH_TEST_CHECK(timesteps.size() == 17);
    SYNTH_TEST_CHECK(timesteps.front() == 0.0);
    SYNTH_TEST_CHECK(timesteps.back() == 0.9999997615814209);
    return 0;
}

// The 16-step twin of check_schedule_upstream_regression, and it exists for
// the same reason: the sum invariant cannot see a float32/float64 swap, only
// the exact per-step values can, and the 32-step anchor at total_mask 1640
// does NOT diverge at 16 steps -- so without this fixture the 16-step
// schedule has no guard against that specific defect at all.
//
// Independently recomputed by sweeping total_mask over [1, 6000]: the
// float32 and double schedules disagree at 85 canvas lengths (1.42%) at 16
// steps, of which 7 are whole 8-codebook canvases. 1,208 (151 frames) is the
// smallest of those seven. float32 gives {8, ..., 477}; a double-precision
// version of the same formula gives {9, ..., 476}. Both sum to 1,208.
//
// Note which lengths are NOT in that set: neither 5,752 nor 400 diverges, so
// every 16-step number measured elsewhere in this project is safe -- and that
// is exactly why the guard has to be a length nobody would otherwise run.
int check_schedule_sixteen_step_upstream_regression() {
    const std::vector<uint64_t> schedule = synth::omnivoice::commit_schedule(1208, 16, 0.1);
    SYNTH_TEST_CHECK(schedule.size() == 16);
    SYNTH_TEST_CHECK(schedule[0] == 8);
    SYNTH_TEST_CHECK(schedule[15] == 477);
    uint64_t sum = 0;
    for (uint64_t count : schedule) {
        sum += count;
    }
    SYNTH_TEST_CHECK(sum == 1208);
    return 0;
}

// --------------------------------------------------------------------------
// choose_token
// --------------------------------------------------------------------------

constexpr uint32_t kToyVocab  = 4;
constexpr uint32_t kToyMaskId = 3;

// cond and uncond are already normalized probability distributions, so their
// log-softmax is themselves: log_softmax(log p) = log p - log(sum p) = log p.
int check_choose_token_guided() {
    const float cond[kToyVocab]   = { std::log(0.5f), std::log(0.25f), std::log(0.125f), std::log(0.125f) };
    const float uncond[kToyVocab] = { std::log(0.25f), std::log(0.25f), std::log(0.25f), std::log(0.25f) };

    int32_t token    = -1;
    float   log_prob = 0.0f;
    synth::omnivoice::choose_token(cond, uncond, kToyVocab, kToyMaskId, 2.0f, token, log_prob);

    // Guided pre-norm is 3*log_pc - 2*log_pu, whose exps are {2, 1/4, 1/32,
    // 1/32}; after the second log-softmax and the mask ban (index 3), the
    // argmax is token 0 with log-prob ln(2 / 2.3125).
    SYNTH_TEST_CHECK(token == 0);
    SYNTH_TEST_CHECK(std::fabs(log_prob - (-0.14518201f)) < 1e-6f);
    return 0;
}

// The mask id's logit dominates the conditional row, but the ban still
// applies after the guided combination, so the argmax must land elsewhere.
int check_choose_token_bans_mask() {
    const float cond[kToyVocab]   = { 0.0f, 0.0f, 0.0f, 100.0f };
    const float uncond[kToyVocab] = { 0.0f, 0.0f, 0.0f, 0.0f };

    int32_t token    = -1;
    float   log_prob = 0.0f;
    synth::omnivoice::choose_token(cond, uncond, kToyVocab, kToyMaskId, 1.0f, token, log_prob);
    SYNTH_TEST_CHECK(token != int32_t(kToyMaskId));
    return 0;
}

// scale 0.0 and uncond == nullptr: token is the argmax of cond alone, and
// log_prob is its plain log-softmax value.
int check_choose_token_unguided() {
    const float cond[kToyVocab] = { 1.0f, 2.0f, 0.5f, -1.0f };

    int32_t token    = -1;
    float   log_prob = 0.0f;
    synth::omnivoice::choose_token(cond, nullptr, kToyVocab, kToyMaskId, 0.0f, token, log_prob);
    SYNTH_TEST_CHECK(token == 1);

    // Plain log-softmax of cond at the argmax: x_best - highest - log(sum_j
    // exp(x_j - highest)), with highest = cond[1] = 2.0f, so the first two
    // terms cancel and only the log-sum-exp survives.
    double total = 0.0;
    for (float value : cond) {
        total += std::exp(double(value) - 2.0);
    }
    const float expected = float(-std::log(total));
    SYNTH_TEST_CHECK(std::fabs(log_prob - expected) < 1e-6f);
    return 0;
}

// guidance_scale == 0.0f && uncond == nullptr is the one combination the
// choose_token contract explicitly allows -- everything else pairing a null
// uncond with a nonzero guidance_scale is now an assert inside the shared
// build_guided helper (Task 3's carryover item 1: the header comment used to
// be the only place this rule lived; the assert is not currently exercised
// by anything short of a death-test harness, which this project's unit
// runner does not have, so its presence is reviewed by hand rather than
// tripped by a test). This case exists to pin the legal combination
// explicitly, separate from check_choose_token_unguided incidentally using
// it: this is the assert's precondition being satisfied, not just a
// convenient fixture.
int check_choose_token_zero_guidance_null_uncond_is_legal() {
    const float cond[kToyVocab] = { 0.2f, -0.5f, 1.5f, 0.0f };

    int32_t token    = -1;
    float   log_prob = 0.0f;
    synth::omnivoice::choose_token(cond, nullptr, kToyVocab, kToyMaskId, 0.0f, token, log_prob);
    SYNTH_TEST_CHECK(token == 2);
    SYNTH_TEST_CHECK(std::isfinite(log_prob));
    return 0;
}

// The runner-up gap the margin report screens on: the distance between the
// argmax and the best entry that lost to it, over the POST-ban vocabulary.
int check_choose_token_runner_up_gap() {
    // Plain log-softmax (scale 0, no uncond), so the gap is a difference of two
    // raw logits: the softmax's shared normalizer cancels exactly.
    const float cond[kToyVocab] = { 1.0f, 2.0f, 0.5f, -1.0f };

    int32_t token    = -1;
    float   log_prob = 0.0f;
    float   gap      = -1.0f;
    synth::omnivoice::choose_token(cond, nullptr, kToyVocab, kToyMaskId, 0.0f, token, log_prob, &gap);
    SYNTH_TEST_CHECK(token == 1);
    SYNTH_TEST_CHECK(std::fabs(gap - (2.0f - 1.0f)) < 1e-6f);

    // The banned mask id is the largest raw logit here. It must not be the
    // runner-up: the gap is measured against index 0, not against the ban.
    const float dominated[kToyVocab] = { 1.0f, 2.0f, 0.5f, 100.0f };
    synth::omnivoice::choose_token(dominated, nullptr, kToyVocab, kToyMaskId, 0.0f, token, log_prob, &gap);
    SYNTH_TEST_CHECK(token == 1);
    SYNTH_TEST_CHECK(std::fabs(gap - 1.0f) < 1e-6f);

    // An exact tie is a zero gap -- the coin flip the screen exists to catch,
    // reported as such rather than rounded away.
    const float tied[kToyVocab] = { 2.0f, 2.0f, 0.5f, -1.0f };
    synth::omnivoice::choose_token(tied, nullptr, kToyVocab, kToyMaskId, 0.0f, token, log_prob, &gap);
    SYNTH_TEST_CHECK(gap == 0.0f);

    // No rival at all: a two-entry vocabulary whose second entry IS the mask.
    // -inf leaves nothing to be close to, which reads as an infinite gap and
    // must never read as a zero one.
    const float alone[2] = { 1.0f, 5.0f };
    synth::omnivoice::choose_token(alone, nullptr, /* vocab_size */ 2, /* mask_id */ 1, 0.0f, token, log_prob, &gap);
    SYNTH_TEST_CHECK(token == 0);
    SYNTH_TEST_CHECK(std::isinf(gap) && gap > 0.0f);
    return 0;
}

// --------------------------------------------------------------------------
// select_commits
// --------------------------------------------------------------------------

// commits_before is select_commits' own order, exported so the margin report
// can name the best rejected candidate. Whatever it says must agree with where
// select_commits actually puts things, or a reported margin is measured
// against a candidate the selection never considered.
int check_commits_before_matches_select_commits() {
    std::vector<MaskedCandidate> candidates = {
        { /* codebook */ 1, /* frame */ 0, /* token */ 0, /* score */ 5.0f },
        { /* codebook */ 0, /* frame */ 5, /* token */ 0, /* score */ 4.0f },
        { /* codebook */ 0, /* frame */ 2, /* token */ 0, /* score */ 4.0f },
        { /* codebook */ 2, /* frame */ 1, /* token */ 0, /* score */ 1.0f },
    };
    // Score first.
    SYNTH_TEST_CHECK(synth::omnivoice::commits_before(candidates[0], candidates[1]));
    SYNTH_TEST_CHECK(!synth::omnivoice::commits_before(candidates[1], candidates[0]));
    // Equal scores fall to the lower frame (same codebook).
    SYNTH_TEST_CHECK(synth::omnivoice::commits_before(candidates[2], candidates[1]));
    // A candidate never precedes itself.
    SYNTH_TEST_CHECK(!synth::omnivoice::commits_before(candidates[0], candidates[0]));

    // The rejected tail after a partial commit: its best under this order must
    // be the very next candidate a larger budget would have taken.
    std::vector<MaskedCandidate> partial = candidates;
    const size_t                 kept    = synth::omnivoice::select_commits(partial, 2);
    SYNTH_TEST_CHECK(kept == 2);
    const auto rival = std::min_element(partial.begin() + 2, partial.end(), synth::omnivoice::commits_before);
    std::vector<MaskedCandidate> one_more = candidates;
    SYNTH_TEST_CHECK(synth::omnivoice::select_commits(one_more, 3) == 3);
    SYNTH_TEST_CHECK(rival->codebook == one_more[2].codebook && rival->frame == one_more[2].frame);
    SYNTH_TEST_CHECK(rival->score == one_more[2].score);
    return 0;
}

int check_select_commits() {
    std::vector<MaskedCandidate> candidates = {
        { /* codebook */ 1, /* frame */ 0, /* token */ 0, /* score */ 5.0f },
        { /* codebook */ 0, /* frame */ 5, /* token */ 0, /* score */ 4.0f },
        { /* codebook */ 0, /* frame */ 2, /* token */ 0, /* score */ 4.0f },
        { /* codebook */ 0, /* frame */ 0, /* token */ 0, /* score */ 3.0f },
    };

    std::vector<MaskedCandidate> two      = candidates;
    const size_t                 kept_two = synth::omnivoice::select_commits(two, 2);
    SYNTH_TEST_CHECK(kept_two == 2);
    // Highest score first; the tie inside score 4 breaks on the lower frame.
    SYNTH_TEST_CHECK(two[0].score == 5.0f && two[0].codebook == 1 && two[0].frame == 0);
    SYNTH_TEST_CHECK(two[1].score == 4.0f && two[1].codebook == 0 && two[1].frame == 2);

    std::vector<MaskedCandidate> all      = candidates;
    const size_t                 kept_all = synth::omnivoice::select_commits(all, 10);
    SYNTH_TEST_CHECK(kept_all == 4);
    return 0;
}

// A NaN score is not something honest logits produce, but the comparator
// must stay a strict weak ordering even so: std::partial_sort's introspective
// algorithm relies on that guarantee, and a violated one is undefined
// behavior -- memory corruption, not just a wrong order. This does not prove
// UB is absent on every implementation; it proves the comparator itself is
// well-defined (a NaN-scored candidate sorts after every real one, and two
// NaN-scored candidates fall back to the codebook/frame tie-break), which is
// what makes the algorithm's precondition hold regardless of implementation.
int check_select_commits_nan_guard() {
    const float                  nan_value  = std::numeric_limits<float>::quiet_NaN();
    std::vector<MaskedCandidate> candidates = {
        { /* codebook */ 0, /* frame */ 0, /* token */ 0, /* score */ nan_value },
        { /* codebook */ 1, /* frame */ 0, /* token */ 0, /* score */ 5.0f      },
        { /* codebook */ 2, /* frame */ 0, /* token */ 0, /* score */ nan_value },
        { /* codebook */ 3, /* frame */ 0, /* token */ 0, /* score */ 2.0f      },
    };
    const size_t kept = synth::omnivoice::select_commits(candidates, 4);
    SYNTH_TEST_CHECK(kept == 4);
    // The two real scores sort first, highest first; the two NaN-scored
    // candidates come last, tie-broken by codebook.
    SYNTH_TEST_CHECK(candidates[0].codebook == 1 && candidates[0].score == 5.0f);
    SYNTH_TEST_CHECK(candidates[1].codebook == 3 && candidates[1].score == 2.0f);
    SYNTH_TEST_CHECK(candidates[2].codebook == 0 && std::isnan(candidates[2].score));
    SYNTH_TEST_CHECK(candidates[3].codebook == 2 && std::isnan(candidates[3].score));
    return 0;
}

// --------------------------------------------------------------------------
// build_prompt_grid
// --------------------------------------------------------------------------

int check_prompt_grid() {
    const std::vector<int32_t> text_ids         = { 7, 8 };
    const std::vector<int32_t> reference_tokens = { 10, 11, 20, 21 };  // 2 codebooks x 2 frames
    constexpr uint64_t         kTargetFrames    = 3;
    constexpr uint32_t         kCodebooks       = 2;
    constexpr uint32_t         kMaskId          = 1024;

    PromptLayout         layout;
    const synth_status_t status =
        synth::omnivoice::build_prompt_grid(text_ids, reference_tokens, kTargetFrames, kCodebooks, kMaskId, layout);
    SYNTH_TEST_CHECK(status == SYNTH_OK);
    SYNTH_TEST_CHECK(layout.total() == 7);
    SYNTH_TEST_CHECK(layout.text_length == 2);
    SYNTH_TEST_CHECK(layout.reference_frames == 2);
    SYNTH_TEST_CHECK(layout.target_frames == 3);
    SYNTH_TEST_CHECK(layout.audio_start() == 2);
    SYNTH_TEST_CHECK(layout.audio_length() == 5);
    SYNTH_TEST_CHECK(layout.grid.size() == 14);

    const std::vector<int32_t> expected_row0 = { 7, 8, 10, 11, 1024, 1024, 1024 };
    const std::vector<int32_t> expected_row1 = { 7, 8, 20, 21, 1024, 1024, 1024 };
    for (size_t index = 0; index < expected_row0.size(); ++index) {
        SYNTH_TEST_CHECK(layout.grid[index] == expected_row0[index]);
        SYNTH_TEST_CHECK(layout.grid[7 + index] == expected_row1[index]);
    }

    // Empty text.
    PromptLayout rejected;
    SYNTH_TEST_CHECK(synth::omnivoice::build_prompt_grid({}, reference_tokens, kTargetFrames, kCodebooks, kMaskId,
                                                         rejected) == SYNTH_ERR_INVALID_ARG);
    // Reference size 3 is not a whole number of rows for 2 codebooks.
    SYNTH_TEST_CHECK(synth::omnivoice::build_prompt_grid(text_ids, { 1, 2, 3 }, kTargetFrames, kCodebooks, kMaskId,
                                                         rejected) == SYNTH_ERR_INVALID_ARG);
    // A reference token equal to the mask id is not a committed code.
    SYNTH_TEST_CHECK(synth::omnivoice::build_prompt_grid(text_ids, { 10, 11, 20, 1024 }, kTargetFrames, kCodebooks,
                                                         kMaskId, rejected) == SYNTH_ERR_INVALID_ARG);
    // Zero target frames.
    SYNTH_TEST_CHECK(synth::omnivoice::build_prompt_grid(text_ids, reference_tokens, 0, kCodebooks, kMaskId,
                                                         rejected) == SYNTH_ERR_INVALID_ARG);
    return 0;
}

// --------------------------------------------------------------------------
// fill_shifted_audio_ids
// --------------------------------------------------------------------------

int check_shifted_ids() {
    const std::vector<int32_t> text_ids         = { 7, 8 };
    const std::vector<int32_t> reference_tokens = { 10, 11, 20, 21 };
    constexpr uint64_t         kTargetFrames    = 3;
    constexpr uint32_t         kCodebooks       = 2;
    constexpr uint32_t         kMaskId          = 1024;
    constexpr uint32_t         kVocabSize       = 1025;

    PromptLayout layout;
    SYNTH_TEST_CHECK(synth::omnivoice::build_prompt_grid(text_ids, reference_tokens, kTargetFrames, kCodebooks, kMaskId,
                                                         layout) == SYNTH_OK);

    std::vector<int32_t> output;
    synth::omnivoice::fill_shifted_audio_ids(layout.grid.data(), /* row_stride */ 7, /* offset */ 2, /* count */ 5,
                                             kCodebooks, kVocabSize, output);
    SYNTH_TEST_CHECK(output.size() == 10);

    const std::vector<int32_t> expected_row0 = { 10, 11, 1024, 1024, 1024 };
    const std::vector<int32_t> expected_row1 = { 20 + 1025, 21 + 1025, 1024 + 1025, 1024 + 1025, 1024 + 1025 };
    for (size_t index = 0; index < expected_row0.size(); ++index) {
        SYNTH_TEST_CHECK(output[index] == expected_row0[index]);
        SYNTH_TEST_CHECK(output[5 + index] == expected_row1[index]);
    }
    return 0;
}

// --------------------------------------------------------------------------
// note_margin / margin_value_json
// --------------------------------------------------------------------------

// The update rule the greedy loop's margin report is built on: keep the
// narrowest measurement, except once a NaN has been recorded nothing may ever
// displace it again. Untested before this task because note_margin lived as
// a file-local helper in model.cpp, reachable only through a full Model.
int check_note_margin_nan_wins() {
    using synth::omnivoice::MarginReport;
    const float nan_value = std::numeric_limits<float>::quiet_NaN();

    MarginReport report;
    SYNTH_TEST_CHECK(!report.measured);

    // The first call always records, whatever it sees.
    synth::omnivoice::note_margin(report, MarginReport::Kind::selection, 5.0f, /*step*/ 1, /*codebook*/ 0,
                                  /*frame*/ 0);
    SYNTH_TEST_CHECK(report.measured);
    SYNTH_TEST_CHECK(report.value == 5.0f);

    // A narrower real value overwrites.
    synth::omnivoice::note_margin(report, MarginReport::Kind::argmax, 2.0f, /*step*/ 2, /*codebook*/ 1,
                                  /*frame*/ 7);
    SYNTH_TEST_CHECK(report.value == 2.0f);
    SYNTH_TEST_CHECK(report.kind == MarginReport::Kind::argmax);
    SYNTH_TEST_CHECK(report.step == 2 && report.codebook == 1 && report.frame == 7);

    // A wider real value does not.
    synth::omnivoice::note_margin(report, MarginReport::Kind::selection, 9.0f, /*step*/ 3, /*codebook*/ 0,
                                  /*frame*/ 0);
    SYNTH_TEST_CHECK(report.value == 2.0f);
    SYNTH_TEST_CHECK(report.step == 2);

    // NaN wins even though it is not numerically "narrower" than 2.0 -- the
    // scoring broke, and that must read as broken rather than as a healthy
    // 2.0 f run. The position it is recorded at moves to where the break
    // happened.
    synth::omnivoice::note_margin(report, MarginReport::Kind::argmax, nan_value, /*step*/ 4, /*codebook*/ 3,
                                  /*frame*/ 9);
    SYNTH_TEST_CHECK(std::isnan(report.value));
    SYNTH_TEST_CHECK(report.step == 4 && report.codebook == 3 && report.frame == 9);

    // Once broken, NOTHING displaces it -- not a tiny real value, and not a
    // second NaN at a different position (which would otherwise silently
    // move the reported location of the break).
    synth::omnivoice::note_margin(report, MarginReport::Kind::selection, 1e-9f, /*step*/ 5, /*codebook*/ 0,
                                  /*frame*/ 0);
    SYNTH_TEST_CHECK(std::isnan(report.value));
    SYNTH_TEST_CHECK(report.step == 4);
    synth::omnivoice::note_margin(report, MarginReport::Kind::argmax, nan_value, /*step*/ 6, /*codebook*/ 0,
                                  /*frame*/ 0);
    SYNTH_TEST_CHECK(std::isnan(report.value));
    SYNTH_TEST_CHECK(report.step == 4);
    return 0;
}

// The JSON spellings for values a bare printf would render as `nan`/`inf`,
// which is not JSON and which Python's own parser rejects.
int check_margin_value_json() {
    SYNTH_TEST_CHECK(synth::omnivoice::margin_value_json(std::numeric_limits<float>::quiet_NaN()) == "NaN");
    SYNTH_TEST_CHECK(synth::omnivoice::margin_value_json(std::numeric_limits<float>::infinity()) == "Infinity");
    SYNTH_TEST_CHECK(synth::omnivoice::margin_value_json(-std::numeric_limits<float>::infinity()) == "-Infinity");
    // A finite value round-trips as an ordinary JSON number, not one of the
    // three non-finite spellings.
    const std::string finite = synth::omnivoice::margin_value_json(6.1e-04f);
    SYNTH_TEST_CHECK(finite != "NaN" && finite != "Infinity" && finite != "-Infinity");
    SYNTH_TEST_CHECK(std::fabs(std::strtof(finite.c_str(), nullptr) - 6.1e-04f) < 1e-9f);
    return 0;
}

}  // namespace

int main() {
    SYNTH_TEST_CHECK(check_schedule_hand_fixture() == 0);
    SYNTH_TEST_CHECK(check_schedule_clamps() == 0);
    SYNTH_TEST_CHECK(check_schedule_real_parameters() == 0);
    SYNTH_TEST_CHECK(check_schedule_upstream_regression() == 0);
    SYNTH_TEST_CHECK(check_schedule_sixteen_step_real_parameters() == 0);
    SYNTH_TEST_CHECK(check_schedule_sixteen_step_upstream_regression() == 0);
    SYNTH_TEST_CHECK(check_choose_token_guided() == 0);
    SYNTH_TEST_CHECK(check_choose_token_bans_mask() == 0);
    SYNTH_TEST_CHECK(check_choose_token_unguided() == 0);
    SYNTH_TEST_CHECK(check_choose_token_zero_guidance_null_uncond_is_legal() == 0);
    SYNTH_TEST_CHECK(check_choose_token_runner_up_gap() == 0);
    SYNTH_TEST_CHECK(check_commits_before_matches_select_commits() == 0);
    SYNTH_TEST_CHECK(check_select_commits() == 0);
    SYNTH_TEST_CHECK(check_select_commits_nan_guard() == 0);
    SYNTH_TEST_CHECK(check_prompt_grid() == 0);
    SYNTH_TEST_CHECK(check_shifted_ids() == 0);
    SYNTH_TEST_CHECK(check_note_margin_nan_wins() == 0);
    SYNTH_TEST_CHECK(check_margin_value_json() == 0);
    return 0;
}
