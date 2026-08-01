// The Gumbel sampler host logic: gumbel_perturb and choose_token_sampled's
// class branch. Transcribed from omnivoice/models/omnivoice.py at the pinned
// revision 468e927b -- `_gumbel_sample` (1632-1636) and the class-branch call
// site (1443-1448). The Port Validation Contract's replay seam feeds the
// oracle's codes precisely so comparison stays deterministic, which means
// this sampler is a stage the exact-token gate never exercises -- exactly
// the situation that let qwen3-tts's repetition penalty go untested (see
// tests/qwen3_tts_sampling_test.cpp). These fixtures exist so the same gap
// does not open here.

#include "arch/omnivoice/generator-host.h"
#include "random-stream.h"
#include "test-assert.h"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

// Bit-exact comparison: gumbel_perturb's contract is the *expression shape*
// (float32 throughout, this exact grouping), not "close enough". A silent
// promotion to double or a reordering of the sum changes the low bits without
// tripping a tolerance-based check.
bool bits_equal(float a, float b) {
    uint32_t bits_a = 0;
    uint32_t bits_b = 0;
    std::memcpy(&bits_a, &a, sizeof(bits_a));
    std::memcpy(&bits_b, &b, sizeof(bits_b));
    return bits_a == bits_b;
}

// --------------------------------------------------------------------------
// gumbel_perturb
// --------------------------------------------------------------------------

// Upstream's `_gumbel_sample`, one value: scaled = logit / temperature;
// g = -log(-log(u + 1e-10) + 1e-10); result = scaled + g. Each expected value
// below is the same float32 expression, written out longhand rather than
// pinned as a literal, so this test pins the *shape* of the computation: a
// refactor to double intermediates or a reordering of the sum would change
// the low bits and fail bit-exact comparison even though the "obvious"
// result stays close.
int check_gumbel_perturb_exactness() {
    struct Case {
        float logit;
        float temperature;
        float uniform;
    };

    const Case cases[] = {
        { 0.0f,   5.0f, 0.5f       },
        { -3.25f, 5.0f, 0.0f       }, // u = 0: the 1e-10 guard must keep this finite.
        { 2.5f,   1.0f, 0.9999999f }, // u -> 1: the other 1e-10 guard must keep this finite.
        { -1e30f, 5.0f, 0.25f      },
    };
    for (const Case & c : cases) {
        const float scaled   = c.logit / c.temperature;
        const float noise    = -std::log(-std::log(c.uniform + 1e-10f) + 1e-10f);
        const float expected = scaled + noise;
        SYNTH_TEST_CHECK(std::isfinite(expected));
        const float actual = synth::omnivoice::gumbel_perturb(c.logit, c.temperature, c.uniform);
        SYNTH_TEST_CHECK(std::isfinite(actual));
        SYNTH_TEST_CHECK(bits_equal(actual, expected));
    }
    return 0;
}

// --------------------------------------------------------------------------
// choose_token_sampled -- class branch
// --------------------------------------------------------------------------

// Same seed, two independent NormalRandomStream instances: the implementation
// must be deterministic in the seed alone, with no hidden global state
// leaking between calls (the greedy path's `thread_local` buffers are exactly
// the kind of bug this would catch if it were reused carelessly here).
int check_choose_token_sampled_determinism() {
    constexpr uint32_t kVocab       = 8;
    constexpr uint32_t kMaskId      = 999;  // Out of range: the ban never applies to this fixture.
    const float        cond[kVocab] = { 0.5f, 1.2f, -0.3f, 2.0f, 0.1f, -1.0f, 0.7f, 3.0f };

    synth::NormalRandomStream stream_a(7);
    int32_t                   token_a    = -1;
    float                     log_prob_a = 0.0f;
    SYNTH_TEST_CHECK(synth::omnivoice::choose_token_sampled(cond, nullptr, kVocab, kMaskId, 0.0f, 5.0f, stream_a,
                                                            token_a, log_prob_a) == SYNTH_OK);

    synth::NormalRandomStream stream_b(7);
    int32_t                   token_b    = -1;
    float                     log_prob_b = 0.0f;
    SYNTH_TEST_CHECK(synth::omnivoice::choose_token_sampled(cond, nullptr, kVocab, kMaskId, 0.0f, 5.0f, stream_b,
                                                            token_b, log_prob_b) == SYNTH_OK);
    SYNTH_TEST_CHECK(token_a == token_b);
    SYNTH_TEST_CHECK(log_prob_a == log_prob_b);

    // Seed 8 draws a different raw sequence than seed 7 -- proven directly
    // against the stream, not through the token: with kVocab == 8, keep =
    // ceil(0.1*8) = 1 survivor, so the token is forced to the argmax
    // regardless of which uniform lands, and asserting token inequality here
    // would test nothing (tokens may legitimately coincide across seeds).
    // What must differ is the actual randomness drawn.
    synth::NormalRandomStream seed7_raw(7);
    synth::NormalRandomStream seed8_raw(8);
    SYNTH_TEST_CHECK(seed7_raw.next_uniform() != seed8_raw.next_uniform());
    return 0;
}

// keep = ceil(0.1 * 10) = 1: only the argmax class ever survives the filter,
// so it must be the class drawn regardless of temperature or which uniform
// the stream produces.
int check_choose_token_sampled_topk_boundary() {
    constexpr uint32_t kVocab       = 10;
    constexpr uint32_t kMaskId      = 999;
    constexpr int32_t  kExpected    = 6;
    const float        cond[kVocab] = { 0.0f, 1.0f, 0.5f, -2.0f, 0.2f, -0.5f, 10.0f, -1.0f, 0.3f, -0.2f };

    const float temperatures[] = { 0.01f, 1.0f, 5.0f, 100.0f };
    for (float temperature : temperatures) {
        for (uint64_t seed = 0; seed < 8; ++seed) {
            synth::NormalRandomStream stream(seed);
            int32_t                   token    = -1;
            float                     log_prob = 0.0f;
            SYNTH_TEST_CHECK(synth::omnivoice::choose_token_sampled(cond, nullptr, kVocab, kMaskId, 0.0f, temperature,
                                                                    stream, token, log_prob) == SYNTH_OK);
            SYNTH_TEST_CHECK(token == kExpected);
        }
    }
    return 0;
}

// The mask id holds the largest raw logit, so it would win an unfiltered
// argmax outright -- but build_guided bans it to -inf before the top-k filter
// even looks at the array, so it can never be among the survivors a Gumbel
// draw might pick. Swept across 64 seeds so the ban's persistence is checked
// against many different draws, not just one lucky one.
int check_choose_token_sampled_bans_mask() {
    constexpr uint32_t kVocab  = 20;
    constexpr uint32_t kMaskId = 19;
    float              cond[kVocab];
    for (uint32_t index = 0; index < kVocab; ++index) {
        cond[index] = float(index);  // Ascending, so index 18 is the highest non-mask value.
    }
    cond[kMaskId] = 1000.0f;         // The mask holds the largest raw logit in the row.

    for (uint64_t seed = 0; seed < 64; ++seed) {
        synth::NormalRandomStream stream(seed);
        int32_t                   token    = -1;
        float                     log_prob = 0.0f;
        SYNTH_TEST_CHECK(synth::omnivoice::choose_token_sampled(cond, nullptr, kVocab, kMaskId, 0.0f, 5.0f, stream,
                                                                token, log_prob) == SYNTH_OK);
        SYNTH_TEST_CHECK(token != int32_t(kMaskId));
    }
    return 0;
}

// Reuses the guided-combination fixture from
// tests/omnivoice_generator_host_test.cpp's check_choose_token_guided: guided
// pre-norm exps are {2, 1/4, 1/32, 1/32}, so after the mask ban (index 3) the
// greedy argmax is token 0. With vocab_size 4, keep = ceil(0.1*4) = 1, so
// choose_token_sampled's surviving top-k set is exactly that same singleton
// and the sampled token must equal the greedy one regardless of the Gumbel
// draw, on every seed -- proof the two paths share one guided array rather
// than two that could drift apart.
int check_choose_token_sampled_guidance_consistency() {
    constexpr uint32_t kVocab         = 4;
    constexpr uint32_t kMaskId        = 3;
    const float        cond[kVocab]   = { std::log(0.5f), std::log(0.25f), std::log(0.125f), std::log(0.125f) };
    const float        uncond[kVocab] = { std::log(0.25f), std::log(0.25f), std::log(0.25f), std::log(0.25f) };

    int32_t greedy_token    = -1;
    float   greedy_log_prob = 0.0f;
    synth::omnivoice::choose_token(cond, uncond, kVocab, kMaskId, 2.0f, greedy_token, greedy_log_prob);
    SYNTH_TEST_CHECK(greedy_token == 0);

    for (uint64_t seed = 0; seed < 8; ++seed) {
        synth::NormalRandomStream stream(seed);
        int32_t                   token    = -1;
        float                     log_prob = 0.0f;
        SYNTH_TEST_CHECK(synth::omnivoice::choose_token_sampled(cond, uncond, kVocab, kMaskId, 2.0f, 5.0f, stream,
                                                                token, log_prob) == SYNTH_OK);
        SYNTH_TEST_CHECK(token == greedy_token);
        // log_prob is the guided log-probability of the chosen token, read
        // from the same guided array the greedy path used -- unaffected by
        // which Gumbel draw won the argmax.
        SYNTH_TEST_CHECK(std::fabs(log_prob - greedy_log_prob) < 1e-6f);
    }
    return 0;
}

// class_temperature == 0.0 must short-circuit to the existing greedy
// choose_token: same token, same log_prob, AND no top-k, no draws -- the
// stream must come out of the call exactly as it went in.
int check_choose_token_sampled_zero_temperature_short_circuits() {
    constexpr uint32_t kVocab       = 4;
    constexpr uint32_t kMaskId      = 3;
    const float        cond[kVocab] = { 1.0f, 2.0f, 0.5f, -1.0f };

    int32_t greedy_token    = -1;
    float   greedy_log_prob = 0.0f;
    synth::omnivoice::choose_token(cond, nullptr, kVocab, kMaskId, 0.0f, greedy_token, greedy_log_prob);

    synth::NormalRandomStream stream(42);
    // An untouched stream with the same seed: its first draw is what `stream`
    // must still produce afterward if choose_token_sampled drew nothing.
    synth::NormalRandomStream witness(42);
    const float               witness_first_draw = witness.next_uniform();

    int32_t token    = -1;
    float   log_prob = 0.0f;
    SYNTH_TEST_CHECK(synth::omnivoice::choose_token_sampled(cond, nullptr, kVocab, kMaskId, 0.0f, 0.0f, stream, token,
                                                            log_prob) == SYNTH_OK);
    SYNTH_TEST_CHECK(token == greedy_token);
    SYNTH_TEST_CHECK(log_prob == greedy_log_prob);
    SYNTH_TEST_CHECK(stream.next_uniform() == witness_first_draw);
    return 0;
}

}  // namespace

int main() {
    SYNTH_TEST_CHECK(check_gumbel_perturb_exactness() == 0);
    SYNTH_TEST_CHECK(check_choose_token_sampled_determinism() == 0);
    SYNTH_TEST_CHECK(check_choose_token_sampled_topk_boundary() == 0);
    SYNTH_TEST_CHECK(check_choose_token_sampled_bans_mask() == 0);
    SYNTH_TEST_CHECK(check_choose_token_sampled_guidance_consistency() == 0);
    SYNTH_TEST_CHECK(check_choose_token_sampled_zero_temperature_short_circuits() == 0);
    return 0;
}
