#include "latent-sampling.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>

namespace synth::vits {

LatentSamplingGraph build_latent_sampling_graph(ggml_context * context,
                                                int64_t        channels,
                                                int64_t        frame_count,
                                                float          noise_scale) {
    LatentSamplingGraph result;
    if (context == nullptr || channels <= 0 || frame_count <= 0 || !std::isfinite(noise_scale) || noise_scale < 0.0f) {
        std::fprintf(stderr, "vits: invalid latent-sampling graph request\n");
        return result;
    }

    result.m_p          = ggml_new_tensor_2d(context, GGML_TYPE_F32, channels, frame_count);
    result.logs_p       = ggml_new_tensor_2d(context, GGML_TYPE_F32, channels, frame_count);
    result.latent_noise = ggml_new_tensor_2d(context, GGML_TYPE_F32, channels, frame_count);
    ggml_set_name(result.m_p, "prior.m_p_expanded.input");
    ggml_set_name(result.logs_p, "prior.logs_p_expanded.input");
    ggml_set_name(result.latent_noise, "random.latent_noise.input");
    ggml_set_input(result.m_p);
    ggml_set_input(result.logs_p);
    ggml_set_input(result.latent_noise);

    ggml_tensor * standard_deviation = ggml_exp(context, result.logs_p);
    ggml_tensor * scaled_noise =
        ggml_scale(context, ggml_mul(context, result.latent_noise, standard_deviation), noise_scale);
    result.z_p = ggml_add(context, result.m_p, scaled_noise);
    ggml_set_name(result.z_p, "latent.z_p");

    result.graph = ggml_new_graph_custom(context, 64, false);
    ggml_build_forward_expand(result.graph, result.z_p);
    return result;
}

}  // namespace synth::vits
