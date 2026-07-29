#pragma once

#include <cstdint>
#include <vector>

namespace synth {
class NormalRandomStream;
}

namespace synth::qwen3tts {

// What one call of the predictor does within a frame.
//
// The mapping is easy to get off by one and impossible to notice afterwards: a
// wrong table or head still produces plausible codes, and the audio only sounds
// subtly wrong. The reference's prefill covers two positions -- the talker's
// hidden state and the semantic code's embedding -- and reads its logits from
// head 0; each later call embeds the code the previous one produced through the
// table of *that* code's group and reads the next head.
struct CodePredictorStep {
    // Index into codec_embedding, or kPrefillStep for the first call, which is
    // handed its input by the talker rather than embedding one.
    uint32_t embedding_table = 0;
    uint32_t lm_head         = 0;
    int64_t  first_position  = 0;
    int64_t  position_count  = 0;
};

constexpr uint32_t kPrefillStep = UINT32_MAX;

// The full per-frame schedule: one entry per predictor call, producing
// code_group_count - 1 acoustic codes over code_group_count positions. Returns
// empty for a group count below two, which would leave nothing to predict.
std::vector<CodePredictorStep> code_predictor_schedule(uint32_t code_group_count);

// The knobs the package's generation defaults carry for this head. They are read
// from the package rather than defaulted here: what they are is the checkpoint's
// business, and a port that keeps its own copy drifts from it silently.
struct SamplingParams {
    bool     enabled     = true;
    float    temperature = 0.9f;
    uint32_t top_k       = 50;
    float    top_p       = 1.0f;
};

// Divides the logits of codes already drawn, which is what the checkpoint's
// `repetition_penalty` asks for and what this port did not do.
//
// Hugging Face's rule, kept exactly: a positive logit is divided by the penalty
// and a negative one multiplied, so both move toward zero rather than a negative
// score being rewarded. A penalty of 1 is the identity.
//
// `history` is the codes drawn so far in this utterance. The end token is never
// among them -- drawing it ends the utterance -- so this only ever suppresses
// speech, never the model's ability to stop.
void apply_repetition_penalty(std::vector<float> & logits, const std::vector<int32_t> & history, float penalty);

// The distribution a draw is made from, once the filters have run.
//
// Split out of select_code so it can be compared against the reference's own
// logits processors on identical inputs. That comparison is the only way the
// suite can see this stage at all: the Port Validation Contract replays the
// oracle's codes, so eighteen Golden cases never execute a draw, and the missing
// repetition penalty lived here unnoticed until a listener heard a sentence stop
// early.
//
// `order` is the survivors ranked by score, `weights` their unnormalised mass
// indexed by code, `survivors` how many of `order` top-p kept, and `retained`
// the mass across those. A code outside the first `survivors` entries of `order`
// cannot be drawn.
struct SamplingDistribution {
    std::vector<size_t> order;
    std::vector<double> weights;
    size_t              survivors = 0;
    double              retained  = 0.0;

    // The probability this distribution gives a code, which is what the draw
    // actually uses and what a comparison against another implementation should
    // be made on -- scores differ in how they spell a rejected entry.
    double probability(size_t code) const {
        if (!(retained > 0.0) || code >= weights.size()) {
            return 0.0;
        }
        for (size_t rank = 0; rank < survivors && rank < order.size(); ++rank) {
            if (order[rank] == code) {
                return weights[code] / retained;
            }
        }
        return 0.0;
    }
};

SamplingDistribution sampling_distribution(const std::vector<float> & logits, const SamplingParams & params);

// Selects one code from a step's logits.
//
// This is a host seam because it is where a distribution becomes a discrete
// value: the result indexes an embedding table on the next step, and a value
// that differs by one between backends changes the whole rest of the frame. See
// docs/backends.md.
//
// The filter order is the reference's: temperature, then top-k, then top-p over
// the softmax of what survives, then a draw. `stream` is unused when sampling is
// disabled, which selects the largest logit instead.
uint32_t select_code(const std::vector<float> & logits, const SamplingParams & params, NormalRandomStream & stream);

}  // namespace synth::qwen3tts
