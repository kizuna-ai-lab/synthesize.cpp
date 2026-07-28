// Host-side orchestration for the code predictor: the per-frame step schedule,
// and the point where a distribution becomes a code.

#include "code-predictor-host.h"

#include "random-stream.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace synth::qwen3tts {

std::vector<CodePredictorStep> code_predictor_schedule(uint32_t code_group_count) {
    std::vector<CodePredictorStep> steps;
    if (code_group_count < 2) {
        return steps;
    }
    steps.reserve(code_group_count - 1);

    // The prefill holds two positions and still predicts only one code: the
    // logits come from the last position, which is the semantic code's.
    CodePredictorStep prefill;
    prefill.embedding_table = kPrefillStep;
    prefill.lm_head         = 0;
    prefill.first_position  = 0;
    prefill.position_count  = 2;
    steps.push_back(prefill);

    for (uint32_t group = 1; group + 1 < code_group_count; ++group) {
        CodePredictorStep step;
        // The code produced by the previous call belongs to group `group - 1`,
        // and is embedded through that group's table rather than this one's.
        step.embedding_table = group - 1;
        step.lm_head         = group;
        step.first_position  = int64_t(group) + 1;
        step.position_count  = 1;
        steps.push_back(step);
    }
    return steps;
}

namespace {

size_t argmax(const std::vector<float> & values) {
    return size_t(std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

}  // namespace

uint32_t select_code(const std::vector<float> & logits, const SamplingParams & params, NormalRandomStream & stream) {
    if (logits.empty()) {
        return 0;
    }
    if (!params.enabled || !(params.temperature > 0.0f)) {
        return uint32_t(argmax(logits));
    }

    std::vector<float> scores(logits.size());
    for (size_t index = 0; index < logits.size(); ++index) {
        scores[index] = logits[index] / params.temperature;
    }

    // top-k first, matching the reference's warper order. A k at or above the
    // vocabulary keeps everything, which is what the reference's clamp does.
    const size_t keep = params.top_k == 0 ? scores.size() : std::min<size_t>(params.top_k, scores.size());
    std::vector<size_t> order(scores.size());
    std::iota(order.begin(), order.end(), size_t(0));
    // Descending by score, ties broken by the lower index, so the survivors are
    // the same set the reference's topk returns.
    std::stable_sort(order.begin(), order.end(),
                     [&scores](size_t left, size_t right) { return scores[left] > scores[right]; });

    std::vector<float> filtered(scores.size(), -std::numeric_limits<float>::infinity());
    for (size_t rank = 0; rank < keep; ++rank) {
        filtered[order[rank]] = scores[order[rank]];
    }

    // Softmax over what survived, then top-p: walking the survivors from most
    // to least likely, everything past the point where the mass reaches top_p is
    // dropped. The most likely code is always kept, which is what the
    // reference's min_tokens_to_keep guarantees.
    const float highest = filtered[order[0]];
    double      total   = 0.0;
    std::vector<double> weights(scores.size(), 0.0);
    for (size_t rank = 0; rank < keep; ++rank) {
        const size_t index = order[rank];
        weights[index]     = std::exp(double(filtered[index] - highest));
        total += weights[index];
    }
    if (!(total > 0.0)) {
        return uint32_t(order[0]);
    }

    const double threshold  = double(std::clamp(params.top_p, 0.0f, 1.0f));
    double       cumulative = 0.0;
    size_t       survivors  = keep;
    for (size_t rank = 0; rank < keep; ++rank) {
        cumulative += weights[order[rank]] / total;
        if (cumulative >= threshold) {
            survivors = rank + 1;
            break;
        }
    }

    double retained = 0.0;
    for (size_t rank = 0; rank < survivors; ++rank) {
        retained += weights[order[rank]];
    }
    if (!(retained > 0.0)) {
        return uint32_t(order[0]);
    }

    const double draw       = double(stream.next_uniform()) * retained;
    double       accumulated = 0.0;
    for (size_t rank = 0; rank < survivors; ++rank) {
        accumulated += weights[order[rank]];
        if (draw < accumulated) {
            return uint32_t(order[rank]);
        }
    }
    // Only reachable when the draw lands on the far edge of the last interval.
    return uint32_t(order[survivors - 1]);
}

}  // namespace synth::qwen3tts
