#pragma once

#include "synthesize.h"

#include <cstdint>
#include <vector>

namespace synth::omnivoice {

// The host side of the mask-predict decode loop: the prompt grid, the commit
// schedule, the guided scoring and the flat top-k commit. None of it touches a
// tensor. Everything here transcribes omnivoice/models/omnivoice.py at the
// pinned revision -- _prepare_inference_inputs, _get_time_steps,
// _generate_iterative and _predict_tokens_with_scoring -- because a wrong rule
// here produces a grid the codec still decodes, and only the exact-token gate
// would ever notice.

// The conditional prompt as an 8-row grid, codebook-major:
// grid[c * total() + s]. Text region first (every row repeats the same ids),
// then the reference audio tokens, then target_frames of the mask id.
struct PromptLayout {
    std::vector<int32_t> grid;
    uint64_t             text_length      = 0;
    uint64_t             reference_frames = 0;
    uint64_t             target_frames    = 0;
    uint32_t             num_codebooks    = 0;

    uint64_t total() const { return text_length + reference_frames + target_frames; }

    uint64_t audio_start() const { return text_length; }

    uint64_t audio_length() const { return reference_frames + target_frames; }
};

// Builds the conditional grid. `text_ids` is row 0's text region (style markers
// plus wrapped text, already tokenized); `reference_tokens` is codebook-major
// [num_codebooks * reference_frames], empty when there is no reference.
// Refuses empty text, zero target frames, a reference stream that is not a
// whole number of codebook rows, or any reference token outside [0, mask_id).
synth_status_t build_prompt_grid(const std::vector<int32_t> & text_ids,
                                 const std::vector<int32_t> & reference_tokens,
                                 uint64_t                     target_frames,
                                 uint32_t                     num_codebooks,
                                 uint32_t                     mask_id,
                                 PromptLayout &               output);

// num_step + 1 points linearly spaced on [0, 1], each shifted by
// t' = t_shift * t / (1 + (t_shift - 1) * t). t_shift 0.1 makes the early
// intervals small: the loop commits little while everything is masked and most
// near the end.
//
// Computed in float32, matching the dtype of upstream's own
// `torch.linspace(0, 1, num_step + 1)` tensor -- not an approximation of
// convenience. A double-precision computation of the same formula disagrees
// with the float32 one at roughly 1% of canvas lengths (see generator-host.cpp),
// each disagreement changing which positions a step commits.
std::vector<double> shifted_timesteps(uint32_t num_step, double t_shift);

// Per-step commit budgets over total_mask = 8 * T positions. Step s commits
// ceil(total_mask * (t'[s+1] - t'[s])) clamped to what remains; the FINAL step
// commits the entire remainder, so the grid is always fully committed after
// num_step steps regardless of rounding. Sums to exactly total_mask.
std::vector<uint64_t> commit_schedule(uint64_t total_mask, uint32_t num_step, double t_shift);

// One masked position's decision from its conditional and unconditional logit
// rows (each vocab_size floats). With guidance_scale != 0 the combination is
// log_softmax(log_pc + s * (log_pc - log_pu)) over log-softmaxed inputs -- the
// double log-softmax is the reference's, not an accident. The mask id is
// banned AFTER the combination; `token` is the argmax over what remains and
// `log_prob` its guided log-probability. `uncond` may be nullptr only when
// guidance_scale == 0 (the reference's `guidance_scale != 0` branch).
//
// A non-null `runner_up_gap` receives `log_prob` minus the second-highest
// guided value over the same post-ban vocabulary: how much slack the argmax
// had, which is what the margin report screens candidate golden cases on. It
// reads +inf when no rival was selectable at all (a one-entry vocabulary, or
// every other entry at -inf) -- the opposite of a near-tie, and it must not be
// read as one.
void choose_token(const float * cond,
                  const float * uncond,
                  uint32_t      vocab_size,
                  uint32_t      mask_id,
                  float         guidance_scale,
                  int32_t &     token,
                  float &       log_prob,
                  float *       runner_up_gap = nullptr);

// One still-masked canvas position, scored. `score` is the guided log-prob
// minus codebook * layer_penalty_factor -- the bias that commits coarse
// codebooks first.
struct MaskedCandidate {
    uint32_t codebook   = 0;
    uint64_t frame      = 0;
    int32_t  token      = 0;
    float    score      = 0.0f;
    // choose_token's `runner_up_gap` for this position, when the caller asked
    // for a margin report. Never read by the ordering below: two positions are
    // ranked on `score` alone, exactly as upstream's topk ranks them.
    float    argmax_gap = 0.0f;
};

// The order select_commits keeps: higher score first, ties broken on lower
// codebook then lower frame, NaN scores after everything real. Exposed because
// the margin report has to find the best REJECTED candidate after the partial
// sort has left the tail unordered, and a second copy of this rule that drifted
// from the first would report a margin against a candidate the selection never
// considered.
bool commits_before(const MaskedCandidate & left, const MaskedCandidate & right);

// Flat top-k over the candidates: partitions the `budget` highest scores to
// the front and returns how many were kept (min(budget, size)). Ties break on
// (higher score, then lower codebook, then lower frame) -- torch.topk's tie
// order is unspecified, so this port pins its own; an exact tie in F32 scores
// between real logits has never been observed and the exact-token gate would
// surface one instantly.
size_t select_commits(std::vector<MaskedCandidate> & candidates, uint64_t budget);

// Shifts a codebook-major id region into the stacked audio-embedding table's
// row space: output[c * count + i] = grid[c * row_stride + offset + i]
// + c * vocab_size. `output` is sized by the callee.
void fill_shifted_audio_ids(const int32_t *        grid,
                            uint64_t               row_stride,
                            uint64_t               offset,
                            uint64_t               count,
                            uint32_t               num_codebooks,
                            uint32_t               vocab_size,
                            std::vector<int32_t> & output);

}  // namespace synth::omnivoice
