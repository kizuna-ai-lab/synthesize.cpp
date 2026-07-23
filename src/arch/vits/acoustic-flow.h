#pragma once

#include <cstdint>

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

namespace synth::vits {

struct FlowWeights;
struct HParams;
struct VoiceWeights;

struct AcousticFlowGraph {
    ggml_cgraph * graph           = nullptr;
    ggml_tensor * speaker_index   = nullptr;
    ggml_tensor * z_p             = nullptr;
    ggml_tensor * channel_indices = nullptr;
    ggml_tensor * z               = nullptr;
};

AcousticFlowGraph build_acoustic_flow_graph(ggml_context *      context,
                                            const FlowWeights & weights,
                                            const HParams &     hparams,
                                            int64_t             frame_count);
AcousticFlowGraph build_acoustic_flow_graph(ggml_context *       context,
                                            const FlowWeights &  weights,
                                            const VoiceWeights & voice_weights,
                                            const HParams &      hparams,
                                            uint32_t             speaker_index,
                                            int64_t              frame_count);

}  // namespace synth::vits
