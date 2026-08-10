#pragma once

#include "synthesize.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace synth {
class NormalRandomStream;
}

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
// guidance_scale == 0 (the reference's `guidance_scale != 0` branch) -- this
// is asserted, not just documented: the shared `build_guided` helper behind
// both this function and `choose_token_sampled` refuses
// `guidance_scale != 0.0f && uncond == nullptr`. The project's unit harness
// has no death-test mechanism, so the assert's presence is reviewed rather
// than exercised by a test that expects it to fire.
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

// Upstream's `_gumbel_sample` (omnivoice/models/omnivoice.py:1632-1636),
// applied to one value: scaled = logit / temperature; g = -log(-log(u +
// 1e-10) + 1e-10); result = scaled + g. Float32 throughout -- the port's
// tests pin the exact expression shape, not just a tolerance, because a
// double intermediate or a reordering of the sum changes which token an
// argmax over several perturbed values picks. `uniform` comes from this
// port's own seeded stream; upstream's `_gumbel_sample` has no seed
// parameter of its own, so the seed contract here is this port's, not a
// transcription.
float gumbel_perturb(float logit, float temperature, float uniform);

// How many classes upstream's `_filter_top_k` keeps for a vocabulary of
// `vocab_size` -- and therefore EXACTLY how many uniforms `choose_token_sampled`
// consumes for one position when `class_temperature > 0`, since its draw loop
// runs once per survivor with no early exit. Integer ceiling division by 10
// rather than `0.1 * vocab_size`, so the answer is exact for every vocab_size
// instead of depending on how 0.1's binary rounding falls near a .5 boundary.
//
// Exposed (rather than left inline in the sampler) so a caller that pre-draws a
// whole batch of positions' randomness up front can size that buffer from the
// same formula the consumer uses, instead of restating it and risking a stride
// that silently disagrees with the number of draws actually made.
inline uint32_t topk_keep(uint32_t vocab_size) {
    return std::min((vocab_size + 9) / 10, vocab_size);
}

// The class-branch companion to `choose_token`, transcribing upstream's
// `if class_temperature > 0.0: ... _gumbel_sample(filtered, class_temperature
// ).argmax(-1)` split (omnivoice.py:1443-1448). `class_temperature == 0.0`
// short-circuits to the plain greedy `choose_token` -- no top-k filter, no
// stream draws -- matching upstream's own branch exactly.
//
// Otherwise: builds the same guided array `choose_token` would (shared via
// `build_guided`, so the two paths cannot drift), applies upstream's
// `_filter_top_k` (keep the `ceil(0.1 * vocab_size)` largest guided values;
// the mask id is already -inf from the ban inside `build_guided`, so it can
// never survive), then draws one uniform per SURVIVING class from `stream`,
// in ascending class-id order -- a port-defined draw order, since upstream
// draws a dense `rand_like` over the whole row at once and this port does
// not reproduce that shape. `token` is the argmax of
// `gumbel_perturb(guided[c], class_temperature, u_c)` over the survivors
// (ties keep the lower class id, matching this port's argmax first-maximal
// convention elsewhere).
//
// `log_prob` is upstream's `confidence_scores = log_probs.max(dim=-1)[0]`
// (omnivoice.py:1449): the max over the FULL guided array (post mask-ban,
// pre top-k filter) -- the same value `choose_token` would report as its own
// argmax's log_prob for this row -- NOT `guided[token]`. Upstream computes
// this confidence from `log_probs` independently of which token
// `pred_tokens` names, so whenever the Gumbel draw picks a survivor other
// than the row's true argmax (routine once more than one class survives the
// filter), `log_prob` and the guided value of the chosen `token` diverge by
// design and must not be conflated.
//
// Preconditions identical to `choose_token`, including the
// `guidance_scale != 0.0f && uncond == nullptr` assert (carryover item 1,
// now enforced for both paths through the shared `build_guided`).
//
// Two overloads, and the pre-drawn one is the primitive: the stream overload
// draws `topk_keep(vocab_size)` uniforms into a buffer and delegates, so the
// two cannot describe different draw counts or a different consumption order.
// A caller that scores many positions CONCURRENTLY must use the pre-drawn
// overload -- a shared stream is a sequential accumulator, and consuming it
// from several threads makes which uniform a position receives depend on
// thread scheduling. Drawing the whole batch serially up front and handing
// each position its own slice reproduces the single-threaded stream exactly,
// which is only possible because the count per position is fixed by
// `topk_keep` and does not depend on the logits.
//
// `uniforms` must point at `topk_keep(vocab_size)` values when
// `class_temperature > 0`, and is ignored (may be null) when it is 0.0 -- the
// short-circuit branch draws nothing.
synth_status_t choose_token_sampled(const float * cond,
                                    const float * uncond,
                                    uint32_t      vocab_size,
                                    uint32_t      mask_id,
                                    float         guidance_scale,
                                    float         class_temperature,
                                    const float * uniforms,
                                    int32_t &     token,
                                    float &       log_prob);

synth_status_t choose_token_sampled(const float *        cond,
                                    const float *        uncond,
                                    uint32_t             vocab_size,
                                    uint32_t             mask_id,
                                    float                guidance_scale,
                                    float                class_temperature,
                                    NormalRandomStream & stream,
                                    int32_t &            token,
                                    float &              log_prob);

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

// The canonical scan over still-masked canvas positions: codebook-major,
// frame-minor. This walk fixes two things at once -- the candidate order that
// `select_commits` later ranks, and the indexing of the block of uniforms the
// step pre-draws -- so moving a position in it moves which random draw that
// position receives.
//
// It lives here, rather than inline in the decode loop, so that the
// differential test of the parallelised scan exercises THIS walk instead of a
// retyped copy of it. A test carrying its own enumeration keeps agreeing with
// itself after production's order changes, which is the one regression such a
// test exists to catch. `candidate_uniform_offset` is exposed beside it for
// the same reason.
void enumerate_masked_candidates(const int32_t *                canvas,
                                 uint32_t                       codebooks,
                                 uint64_t                       frames,
                                 int32_t                        mask_id,
                                 std::vector<MaskedCandidate> & out);

// How many uniforms one candidate consumes: its class draws, then its single
// position draw when position_temperature is on.
inline size_t uniform_draws_per_slot(uint32_t class_draws, bool position_draw) {
    return size_t(class_draws) + (position_draw ? size_t(1) : size_t(0));
}

// Candidate `index` owns [offset, offset + draws_per_slot): class draws first,
// then the position draw. Drawing the whole block in one sequential pass and
// consuming it by this offset reproduces the order an inline per-candidate
// draw used to produce, which is what makes the parallel scan bit-identical.
inline size_t candidate_uniform_offset(size_t index, size_t draws_per_slot) {
    return index * draws_per_slot;
}

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

// The narrowest decision the greedy loop made, filled when a caller asks for
// a margin report (SynthesisRequest::margin_report in omnivoice.h).
//
// Two kinds of decision can be narrow, and this port's two knife-edge cases at
// the slice-5 gate were one of each. A `selection` margin is the guided-score
// gap between the last candidate a step committed and the best one it rejected:
// close it and a different POSITION is committed, which changes every later
// step's context. An `argmax` margin is the gap between a committed position's
// token and its runner-up: close it and a different TOKEN lands in that slot.
//
// A step that commits everything still masked rejects nothing, so it can only
// contribute argmax margins; a step that rejects something contributes its
// selection margin. `value` is the smallest margin over the whole run and the
// remaining fields name the position it was measured at. Note the scope: argmax
// margins are collected on full-commit steps only, which is where the flip that
// motivated this instrument happened -- a narrow argmax on a partially
// committing step is not screened.
struct MarginReport {
    enum class Kind { selection, argmax };

    bool     measured = false;
    Kind     kind     = Kind::selection;
    float    value    = 0.0f;
    uint32_t step     = 0;
    uint32_t codebook = 0;
    uint64_t frame    = 0;
};

// Keeps the narrowest margin measurement seen so far in `report`. A NaN value
// means the scoring broke, so it wins the comparison rather than losing to
// every real number and leaving the report reading like a healthy run; once
// recorded it locks the report -- broken is broken, and a later real number
// must not quietly displace it. The first call always records, whatever it
// sees.
void note_margin(MarginReport &     report,
                 MarginReport::Kind kind,
                 float              value,
                 uint32_t           step,
                 uint32_t           codebook,
                 uint64_t           frame);

// A margin value the way JSON must see it. `NaN`/`Infinity`/`-Infinity` are
// not valid JSON tokens by the spec, but they are exactly what Python's `json`
// module (and most other parsers) accept and round-trip; the alternative --
// `nan`/`inf`, what a plain printf produces -- is what those parsers reject
// outright, turning a broken run's report unreadable rather than visible.
std::string margin_value_json(float value);

}  // namespace synth::omnivoice
