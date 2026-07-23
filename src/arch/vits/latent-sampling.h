#pragma once

#include <cstdint>

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

namespace synth::vits {

struct LatentSamplingGraph {
    ggml_cgraph * graph        = nullptr;
    ggml_tensor * m_p          = nullptr;
    ggml_tensor * logs_p       = nullptr;
    ggml_tensor * latent_noise = nullptr;
    ggml_tensor * z_p          = nullptr;
};

LatentSamplingGraph build_latent_sampling_graph(ggml_context * context,
                                                int64_t        channels,
                                                int64_t        frame_count,
                                                float          noise_scale);

}  // namespace synth::vits
