#pragma once

#include <cstdint>

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

namespace synth::vits {

struct DurationWeights;
struct HParams;
struct VoiceWeights;

struct DurationPredictorGraph {
    ggml_cgraph * graph          = nullptr;
    ggml_tensor * speaker_index  = nullptr;
    ggml_tensor * duration_noise = nullptr;
    ggml_tensor * logw           = nullptr;
};

DurationPredictorGraph build_duration_predictor_graph(ggml_context *          context,
                                                      ggml_tensor *           encoded,
                                                      const DurationWeights & weights,
                                                      const HParams &         hparams,
                                                      int64_t                 token_count,
                                                      float                   noise_scale_w);
DurationPredictorGraph build_duration_predictor_graph(ggml_context *          context,
                                                      ggml_tensor *           encoded,
                                                      const DurationWeights & weights,
                                                      const VoiceWeights &    voice_weights,
                                                      const HParams &         hparams,
                                                      uint32_t                speaker_index,
                                                      int64_t                 token_count,
                                                      float                   noise_scale_w);

}  // namespace synth::vits
