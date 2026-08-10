// The decode loop's per-step candidate scan, serial against parallel.
//
// WHY THIS FILE EXISTS. run_synthesis scores every still-masked canvas position
// of a step on several threads. That is only sound because the scan was made
// order-independent first, and the two halves of that argument are invisible to
// every gate this project already has:
//
//   * The exact-token Golden gate runs all 17 manifest cases at
//     position_temperature = 0.0 AND class_temperature = 0.0. Those cases draw
//     ZERO uniforms, so the entire RNG hazard -- the one thing parallelising
//     this loop genuinely breaks if done naively -- is not merely under-covered
//     there, it is unreachable. A wrong stride, a draw left inside the parallel
//     region, an off-by-one in the per-candidate slice: the Golden suite passes
//     on all of them.
//   * The sanitizer gate is ASan+UBSan; there is no TSan option in this build,
//     so a genuine data race is not detected either.
//
// So the equivalence is asserted here directly, on both branches, by replaying
// the SAME step two ways and comparing the results field by field:
//
//   serial   -- one thread, candidates visited in scan order, each drawing its
//               own uniforms from a shared stream as it goes. This is literally
//               the loop shape that existed before the change.
//   parallel -- every uniform the step consumes drawn UP FRONT in scan order
//               from an identically seeded stream, then the scoring spread over
//               N workers, each candidate reading its own slice by index.
//
// Equality of `token` is exact by definition; `score` is compared BITWISE (via
// its integer image), not with a tolerance -- a tolerance would accept a
// differently-rounded score, and a differently-rounded score is a different
// commit order, which is a different token grid. That is the failure this file
// is here to catch, so it must not be able to pass through it.
//
// What this file is NOT: a check of the scoring rules themselves. What
// choose_token and choose_token_sampled compute is pinned by
// tests/omnivoice_generator_host_test.cpp and tests/omnivoice_sampler_test.cpp
// against hand-computable fixtures. Here the arithmetic is only required to
// agree with itself.

#include "arch/omnivoice/generator-host.h"
#include "cpu-parallelism.h"
#include "random-stream.h"
#include "test-assert.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

// Shape: small enough to stay quick under ASan, large enough that the candidate
// count clears parallel_for's internal serial floor by a wide margin and every
// worker gets several chunks. 4 * 80 = 320 candidates, vocabulary 101 (so
// topk_keep is 11 -- more than one survivor, which is what makes the class draw
// order observable at all).
constexpr uint32_t kCodebooks = 4;
constexpr uint32_t kFrames    = 80;
constexpr uint32_t kVocab     = 101;
constexpr uint32_t kMaskId    = 100;
constexpr float    kGuidance  = 2.0f;
constexpr float    kPenalty   = 0.05f;

// Deterministic, spread-out logits. Constant or tied logits would make every
// candidate agree trivially and hide exactly the ordering bug this file hunts,
// so the values must genuinely differ across both position and class.
std::vector<float> make_logits(uint64_t seed, size_t count) {
    std::vector<float> values(count);
    uint64_t           state = seed;
    for (size_t index = 0; index < count; ++index) {
        state         = state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
        // [-8, 8), fine-grained: 13 bits of mantissa is plenty to make ties
        // vanishingly unlikely while keeping the values printable.
        values[index] = float(int32_t((state >> 33) & 0x1FFF) - 4096) / 512.0f;
    }
    return values;
}

struct Scored {
    std::vector<int32_t> tokens;
    std::vector<float>   scores;
};

bool scores_are_bitwise_equal(const std::vector<float> & left, const std::vector<float> & right) {
    if (left.size() != right.size()) {
        return false;
    }
    // Through the integer image rather than `==`: this must reject a score that
    // differs in the last bit, and it must also treat two identical NaNs as
    // equal (a NaN score is a real possibility the ordering code guards for).
    for (size_t index = 0; index < left.size(); ++index) {
        uint32_t left_bits  = 0;
        uint32_t right_bits = 0;
        std::memcpy(&left_bits, &left[index], sizeof(left_bits));
        std::memcpy(&right_bits, &right[index], sizeof(right_bits));
        if (left_bits != right_bits) {
            return false;
        }
    }
    return true;
}

// The pre-change loop, verbatim in shape: one thread, scan order, draws taken
// from the stream inline at the moment each candidate needs them.
Scored score_serial(const std::vector<float> & cond,
                    const std::vector<float> & uncond,
                    float                      position_temperature,
                    float                      class_temperature,
                    uint64_t                   seed) {
    Scored result;
    result.tokens.resize(size_t(kCodebooks) * kFrames);
    result.scores.resize(size_t(kCodebooks) * kFrames);

    synth::NormalRandomStream stream(seed);
    const size_t              row  = size_t(kCodebooks) * kVocab;
    size_t                    slot = 0;
    for (uint32_t codebook = 0; codebook < kCodebooks; ++codebook) {
        for (uint32_t frame = 0; frame < kFrames; ++frame) {
            const size_t offset   = size_t(frame) * row + size_t(codebook) * kVocab;
            int32_t      token    = 0;
            float        log_prob = 0.0f;
            if (class_temperature > 0.0f) {
                synth::omnivoice::choose_token_sampled(cond.data() + offset, uncond.data() + offset, kVocab, kMaskId,
                                                       kGuidance, class_temperature, stream, token, log_prob);
            } else {
                synth::omnivoice::choose_token(cond.data() + offset, uncond.data() + offset, kVocab, kMaskId, kGuidance,
                                               token, log_prob);
            }
            float score = log_prob - float(codebook) * kPenalty;
            if (position_temperature > 0.0f) {
                score = synth::omnivoice::gumbel_perturb(score, position_temperature, stream.next_uniform());
            }
            result.tokens[slot] = token;
            result.scores[slot] = score;
            ++slot;
        }
    }
    return result;
}

// The post-change loop: pre-draw the whole step serially, then score in
// parallel, each candidate reading its own slice.
Scored score_parallel(const std::vector<float> & cond,
                      const std::vector<float> & uncond,
                      float                      position_temperature,
                      float                      class_temperature,
                      uint64_t                   seed,
                      int                        workers) {
    Scored       result;
    const size_t slots = size_t(kCodebooks) * kFrames;
    result.tokens.assign(slots, -1);
    result.scores.assign(slots, 0.0f);

    const uint32_t class_draws    = class_temperature > 0.0f ? synth::omnivoice::topk_keep(kVocab) : 0u;
    const size_t   draws_per_slot = size_t(class_draws) + (position_temperature > 0.0f ? 1u : 0u);

    std::vector<float> uniforms;
    if (draws_per_slot != 0) {
        synth::NormalRandomStream stream(seed);
        uniforms.resize(slots * draws_per_slot);
        stream.fill_uniform(uniforms.data(), uniforms.size());
    }

    const size_t row = size_t(kCodebooks) * kVocab;
    synth::parallel_for(slots, workers, [&](size_t begin, size_t end) {
        for (size_t slot = begin; slot < end; ++slot) {
            const uint32_t codebook = uint32_t(slot / kFrames);
            const uint32_t frame    = uint32_t(slot % kFrames);
            const size_t   offset   = size_t(frame) * row + size_t(codebook) * kVocab;
            int32_t        token    = 0;
            float          log_prob = 0.0f;
            if (class_temperature > 0.0f) {
                synth::omnivoice::choose_token_sampled(cond.data() + offset, uncond.data() + offset, kVocab, kMaskId,
                                                       kGuidance, class_temperature,
                                                       uniforms.data() + slot * draws_per_slot, token, log_prob);
            } else {
                synth::omnivoice::choose_token(cond.data() + offset, uncond.data() + offset, kVocab, kMaskId, kGuidance,
                                               token, log_prob);
            }
            float score = log_prob - float(codebook) * kPenalty;
            if (position_temperature > 0.0f) {
                score = synth::omnivoice::gumbel_perturb(score, position_temperature,
                                                         uniforms[slot * draws_per_slot + class_draws]);
            }
            result.tokens[slot] = token;
            result.scores[slot] = score;
        }
    });
    return result;
}

int check_one_configuration(const char *               name,
                            const std::vector<float> & cond,
                            const std::vector<float> & uncond,
                            float                      position_temperature,
                            float                      class_temperature) {
    const Scored reference = score_serial(cond, uncond, position_temperature, class_temperature, 4242);
    // A degenerate reference would make every comparison below vacuous: at
    // least two distinct tokens must actually have been chosen.
    bool         varied    = false;
    for (int32_t token : reference.tokens) {
        SYNTH_TEST_CHECK(token >= 0 && uint32_t(token) < kVocab);
        SYNTH_TEST_CHECK(uint32_t(token) != kMaskId);  // the mask is banned before the argmax
        varied = varied || token != reference.tokens[0];
    }
    if (!varied) {
        std::cerr << name << ": fixture is degenerate -- every candidate chose the same token\n";
        return 1;
    }

    for (int workers : { 1, 2, 3, 4, 8, 20 }) {
        const Scored candidate = score_parallel(cond, uncond, position_temperature, class_temperature, 4242, workers);
        if (candidate.tokens != reference.tokens) {
            std::cerr << name << ": tokens differ at " << workers << " workers\n";
            return 1;
        }
        if (!scores_are_bitwise_equal(reference.scores, candidate.scores)) {
            std::cerr << name << ": scores differ bitwise at " << workers << " workers\n";
            return 1;
        }
    }
    return 0;
}

// The load-bearing arithmetic behind the pre-draw: the stream overload consumes
// EXACTLY topk_keep(vocab) uniforms per position and not one more. The whole
// per-candidate slice arithmetic is built on that number, and nothing else in
// the tree would notice if the sampler's draw loop grew or lost an iteration --
// the Golden gate never enters this branch at all.
int check_draw_count_matches_topk_keep() {
    const std::vector<float> cond   = make_logits(11, kVocab);
    const std::vector<float> uncond = make_logits(12, kVocab);

    synth::NormalRandomStream through_sampler(99);
    int32_t                   token    = 0;
    float                     log_prob = 0.0f;
    SYNTH_TEST_CHECK(synth::omnivoice::choose_token_sampled(cond.data(), uncond.data(), kVocab, kMaskId, kGuidance,
                                                            1.25f, through_sampler, token, log_prob) == SYNTH_OK);

    synth::NormalRandomStream by_hand(99);
    std::vector<float>        drawn(synth::omnivoice::topk_keep(kVocab));
    by_hand.fill_uniform(drawn.data(), drawn.size());

    // Both streams have now been advanced by what each believes the sampler
    // consumes; if those disagree, their next values do.
    SYNTH_TEST_CHECK(through_sampler.next_uniform() == by_hand.next_uniform());

    // And the pre-drawn overload, handed exactly those uniforms, reaches the
    // same decision -- so the wrapper really is a wrapper.
    int32_t pre_token    = 0;
    float   pre_log_prob = 0.0f;
    SYNTH_TEST_CHECK(synth::omnivoice::choose_token_sampled(cond.data(), uncond.data(), kVocab, kMaskId, kGuidance,
                                                            1.25f, drawn.data(), pre_token, pre_log_prob) == SYNTH_OK);
    SYNTH_TEST_CHECK(pre_token == token);
    SYNTH_TEST_CHECK(pre_log_prob == log_prob);

    // class_temperature 0.0 short-circuits and must draw nothing at all, with
    // or without a uniforms buffer -- the zero-RNG property depends on it.
    synth::NormalRandomStream untouched(7);
    int32_t                   greedy_token    = 0;
    float                     greedy_log_prob = 0.0f;
    SYNTH_TEST_CHECK(synth::omnivoice::choose_token_sampled(cond.data(), uncond.data(), kVocab, kMaskId, kGuidance,
                                                            0.0f, untouched, greedy_token,
                                                            greedy_log_prob) == SYNTH_OK);
    synth::NormalRandomStream fresh(7);
    SYNTH_TEST_CHECK(untouched.next_uniform() == fresh.next_uniform());
    SYNTH_TEST_CHECK(synth::omnivoice::choose_token_sampled(cond.data(), uncond.data(), kVocab, kMaskId, kGuidance,
                                                            0.0f, static_cast<const float *>(nullptr), greedy_token,
                                                            greedy_log_prob) == SYNTH_OK);
    return 0;
}

}  // namespace

// The differential checks above prove parallel == serial. They cannot prove
// the ENUMERATION is the one production uses: both arms now call the shared
// walk, so flipping it flips both and they keep agreeing. That order is a
// contract in its own right -- it fixes which pre-drawn uniform each canvas
// position receives, so swapping the two loops silently re-assigns every
// Gumbel draw and changes which positions a step commits. Pin it directly.
int check_enumeration_is_codebook_major() {
    // 3 codebooks x 4 frames, with a mask pattern that would look identical
    // under either loop order if only the COUNT were checked.
    constexpr uint32_t   codebooks = 3;
    constexpr uint64_t   frames    = 4;
    constexpr int32_t    mask      = 99;
    std::vector<int32_t> canvas(size_t(codebooks) * size_t(frames), 0);
    canvas[0 * frames + 1] = mask;  // (cb 0, fr 1)
    canvas[0 * frames + 3] = mask;  // (cb 0, fr 3)
    canvas[2 * frames + 0] = mask;  // (cb 2, fr 0)
    canvas[1 * frames + 2] = mask;  // (cb 1, fr 2)

    std::vector<synth::omnivoice::MaskedCandidate> out;
    synth::omnivoice::enumerate_masked_candidates(canvas.data(), codebooks, frames, mask, out);

    // Codebook-major, frame-minor. Frame-major would give
    // (2,0) (0,1) (1,2) (0,3) -- a different order over the same four slots.
    const uint32_t want_codebook[] = { 0, 0, 1, 2 };
    const uint64_t want_frame[]    = { 1, 3, 2, 0 };
    SYNTH_TEST_CHECK(out.size() == 4);
    for (size_t index = 0; index < out.size(); ++index) {
        SYNTH_TEST_CHECK(out[index].codebook == want_codebook[index]);
        SYNTH_TEST_CHECK(out[index].frame == want_frame[index]);
    }

    // A committed slot is never revisited, whatever its value.
    std::vector<synth::omnivoice::MaskedCandidate> none;
    std::vector<int32_t>                           committed(size_t(codebooks) * size_t(frames), 7);
    synth::omnivoice::enumerate_masked_candidates(committed.data(), codebooks, frames, mask, none);
    SYNTH_TEST_CHECK(none.empty());

    // The slice arithmetic the walk's order indexes into, from the same header
    // production reads it from.
    SYNTH_TEST_CHECK(synth::omnivoice::uniform_draws_per_slot(0, false) == 0);
    SYNTH_TEST_CHECK(synth::omnivoice::uniform_draws_per_slot(0, true) == 1);
    SYNTH_TEST_CHECK(synth::omnivoice::uniform_draws_per_slot(103, true) == 104);
    SYNTH_TEST_CHECK(synth::omnivoice::candidate_uniform_offset(0, 104) == 0);
    SYNTH_TEST_CHECK(synth::omnivoice::candidate_uniform_offset(3, 104) == 312);
    return 0;
}

int main() {
    const size_t             row    = size_t(kCodebooks) * kVocab;
    const std::vector<float> cond   = make_logits(0x51ED, size_t(kFrames) * row);
    const std::vector<float> uncond = make_logits(0xC0FFEE, size_t(kFrames) * row);

    if (check_enumeration_is_codebook_major() != 0) {
        return 1;
    }
    if (check_draw_count_matches_topk_keep() != 0) {
        return 1;
    }
    // Fully greedy: no draws at all. Proves the scan is thread-safe on its own
    // -- the shared thread_local scratch inside build_guided, choose_token and
    // the sampler -- independently of the RNG question.
    if (check_one_configuration("greedy", cond, uncond, 0.0f, 0.0f) != 0) {
        return 1;
    }
    // The PUBLIC default for this family: position_temperature 5.0, class 0.0.
    // One draw per candidate, so a stride error of one shows up immediately.
    if (check_one_configuration("position-sampled", cond, uncond, 5.0f, 0.0f) != 0) {
        return 1;
    }
    // Class draws only: topk_keep(vocab) uniforms per candidate, no position
    // draw -- the slice is all class, so an off-by-one in the class count is
    // not masked by a trailing position slot.
    if (check_one_configuration("class-sampled", cond, uncond, 0.0f, 1.25f) != 0) {
        return 1;
    }
    // Both: class draws then the position draw, the full stride.
    if (check_one_configuration("both-sampled", cond, uncond, 5.0f, 1.25f) != 0) {
        return 1;
    }
    return 0;
}
