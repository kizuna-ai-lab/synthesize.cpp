#pragma once

#include "synthesize.h"
#include "text-frontend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_backend_device;

namespace synth::vits {

struct TextEncoderOutput {
    uint32_t           channels    = 0;
    uint64_t           token_count = 0;
    // Logical [channels, token_count] row-major arrays: time is contiguous
    // within each channel, matching the canonical PyTorch probe artifacts.
    std::vector<float> m_p;
    std::vector<float> logs_p;
    std::vector<float> mask;
};

struct DurationPredictorOutput {
    size_t             token_count = 0;
    std::vector<float> logw;
};

struct DurationOutput {
    size_t             token_count = 0;
    uint64_t           frame_count = 0;
    std::vector<float> logw;
    std::vector<float> w_ceil;
    // Logical [frame_count, token_count], with token index contiguous.
    std::vector<float> attention;
};

struct PriorExpansionOutput {
    uint32_t           channels    = 0;
    uint64_t           frame_count = 0;
    // Logical [channels, frame_count], with frame index contiguous.
    std::vector<float> m_p;
    std::vector<float> logs_p;
};

struct LatentSamplingOutput {
    uint32_t           channels    = 0;
    uint64_t           frame_count = 0;
    // Logical [channels, frame_count], with frame index contiguous.
    std::vector<float> z_p;
};

struct AcousticFlowOutput {
    uint32_t           channels    = 0;
    uint64_t           frame_count = 0;
    // Logical [channels, frame_count], with frame index contiguous.
    std::vector<float> z;
};

struct WaveformDecoderOutput {
    uint64_t           sample_count = 0;
    std::vector<float> pcm;
};

struct ModelInfo {
    bool                                has_package_default = false;
    std::vector<std::string>            preset_voice_ids;
    std::vector<uint32_t>               preset_voice_flags;
    std::shared_ptr<const TextFrontend> text_frontend;
    uint32_t                            input_flags          = 0;
    uint32_t                            capability_flags     = 0;
    uint32_t                            output_sample_rate   = 0;
    uint32_t                            output_channel_count = 0;
    uint32_t                            inter_channels       = 0;
    uint32_t                            vocab_size           = 0;
    uint32_t                            hop_length           = 0;
    uint64_t                            max_input_tokens     = 0;
    uint64_t                            max_output_frames    = 0;
    float                               min_speaking_rate    = 0.0f;
    float                               max_speaking_rate    = 0.0f;
    float                               latent_noise_scale   = 0.0f;
    float                               duration_noise_scale = 0.0f;
};

class Model {
  public:
    static synth_status_t load_cpu(const std::string & path, std::unique_ptr<Model> & output);
    static synth_status_t load_cpu(const std::string &      path,
                                   ggml_backend_device *    device,
                                   std::unique_ptr<Model> & output);
    static synth_status_t load(const std::string &      path,
                               ggml_backend_device *    primary_device,
                               bool                     include_accelerators,
                               std::unique_ptr<Model> & output);

    synth_status_t        get_info(ModelInfo & output) const;
    ggml_backend_device * primary_device() const;

    ~Model();
    Model(const Model &)             = delete;
    Model & operator=(const Model &) = delete;
    Model(Model &&)                  = delete;
    Model & operator=(Model &&)      = delete;

    synth_status_t run_text_encoder(const std::vector<int32_t> & token_ids,
                                    int                          threads,
                                    TextEncoderOutput &          output) const;

    synth_status_t run_duration_predictor(const std::vector<int32_t> & token_ids,
                                          const std::vector<float> &   duration_noise,
                                          float                        noise_scale_w,
                                          int                          threads,
                                          DurationPredictorOutput &    output,
                                          uint32_t                     speaker_index = UINT32_MAX) const;

    synth_status_t run_duration(const std::vector<int32_t> & token_ids,
                                const std::vector<float> &   duration_noise,
                                float                        noise_scale_w,
                                float                        speaking_rate,
                                int                          threads,
                                DurationOutput &             output,
                                uint32_t                     speaker_index = UINT32_MAX) const;

    synth_status_t run_prior_expansion(const std::vector<int32_t> & token_ids,
                                       const std::vector<float> &   duration_noise,
                                       float                        noise_scale_w,
                                       float                        speaking_rate,
                                       int                          threads,
                                       PriorExpansionOutput &       output,
                                       uint32_t                     speaker_index = UINT32_MAX) const;

    synth_status_t run_latent_sampling(const std::vector<int32_t> & token_ids,
                                       const std::vector<float> &   duration_noise,
                                       const std::vector<float> &   latent_noise,
                                       float                        noise_scale,
                                       float                        noise_scale_w,
                                       float                        speaking_rate,
                                       int                          threads,
                                       LatentSamplingOutput &       output,
                                       uint32_t                     speaker_index = UINT32_MAX) const;

    synth_status_t run_acoustic_flow(const std::vector<int32_t> & token_ids,
                                     const std::vector<float> &   duration_noise,
                                     const std::vector<float> &   latent_noise,
                                     float                        noise_scale,
                                     float                        noise_scale_w,
                                     float                        speaking_rate,
                                     int                          threads,
                                     AcousticFlowOutput &         output,
                                     uint32_t                     speaker_index = UINT32_MAX) const;

    synth_status_t run_waveform_decoder(const std::vector<int32_t> & token_ids,
                                        const std::vector<float> &   duration_noise,
                                        const std::vector<float> &   latent_noise,
                                        float                        noise_scale,
                                        float                        noise_scale_w,
                                        float                        speaking_rate,
                                        int                          threads,
                                        WaveformDecoderOutput &      output,
                                        uint32_t                     speaker_index = UINT32_MAX) const;

  private:
    struct Impl;
    explicit Model(std::unique_ptr<Impl> implementation);
    std::unique_ptr<Impl> implementation_;
};

}  // namespace synth::vits
