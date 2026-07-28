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

// The knobs the package's generation defaults carry for this head. Sampling is
// on with temperature 0.9 and top-k 50; top-p 1.0 keeps the whole distribution
// and so is inert unless a request overrides it.
struct SamplingParams {
    bool     enabled     = true;
    float    temperature = 0.9f;
    uint32_t top_k       = 50;
    float    top_p       = 1.0f;
};

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
