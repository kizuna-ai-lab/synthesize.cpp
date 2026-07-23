#pragma once

#include "synthesize.h"
#include "text-frontend.h"

#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct gguf_context;

namespace synth::vits {

enum class QuantizationProfile : uint32_t {
    F32,
    F16,
    Q8Mixed,
};

struct HParams {
    std::string                        model_variant;
    QuantizationProfile                quantization_profile         = QuantizationProfile::F32;
    uint32_t                           quantization_profile_version = 1;
    bool                               has_package_default          = false;
    uint32_t                           speaker_count                = 0;
    uint32_t                           conditioning_channels        = 0;
    std::vector<std::string>           preset_voice_ids;
    std::vector<uint32_t>              preset_voice_flags;
    bool                               frontend_present          = false;
    SymbolMapFrontendConfig            frontend_config;
    uint32_t                           input_flags               = 0;
    uint32_t                           capability_flags          = 0;
    uint32_t                           output_sample_rate        = 0;
    uint32_t                           output_channel_count      = 0;
    uint32_t                           vocab_size                = 0;
    uint32_t                           inter_channels            = 0;
    uint32_t                           hidden_channels           = 0;
    uint32_t                           filter_channels           = 0;
    uint32_t                           text_layer_count          = 0;
    uint32_t                           text_head_count           = 0;
    uint32_t                           text_ffn_kernel_size      = 0;
    uint32_t                           text_attention_window     = 0;
    uint32_t                           duration_dds_layer_count  = 0;
    uint32_t                           duration_flow_count       = 0;
    uint32_t                           duration_spline_bin_count = 0;
    uint32_t                           flow_block_count          = 0;
    uint32_t                           flow_kernel_size          = 0;
    uint32_t                           flow_dilation_rate        = 0;
    uint32_t                           flow_wn_layer_count       = 0;
    bool                               flow_mean_only            = false;
    std::vector<uint32_t>              decoder_resblock_kernel_sizes;
    std::vector<std::vector<uint32_t>> decoder_resblock_dilations;
    std::vector<uint32_t>              decoder_upsample_rates;
    std::vector<uint32_t>              decoder_upsample_kernel_sizes;
    uint32_t                           decoder_initial_channels       = 0;
    float                              decoder_leaky_relu_slope       = 0.0f;
    uint32_t                           hop_length                     = 0;
    uint64_t                           max_input_tokens               = 0;
    uint64_t                           max_output_frames              = 0;
    float                              layer_norm_epsilon             = 0.0f;
    float                              embedding_scale                = 0.0f;
    float                              duration_spline_tail_bound     = 0.0f;
    float                              duration_spline_min_bin_width  = 0.0f;
    float                              duration_spline_min_bin_height = 0.0f;
    float                              duration_spline_min_derivative = 0.0f;
    float                              duration_noise_scale_w         = 0.0f;
    float                              latent_noise_scale             = 0.0f;
    float                              min_speaking_rate              = 0.0f;
    float                              max_speaking_rate              = 0.0f;
};

struct Conv1dWeights {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias   = nullptr;
};

struct NormWeights {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias   = nullptr;
};

struct TextBlockWeights {
    Conv1dWeights query;
    Conv1dWeights key;
    Conv1dWeights value;
    Conv1dWeights output;
    ggml_tensor * relative_key   = nullptr;
    ggml_tensor * relative_value = nullptr;
    NormWeights   attention_norm;
    Conv1dWeights ffn_input;
    Conv1dWeights ffn_output;
    NormWeights   ffn_norm;
};

struct TextWeights {
    ggml_tensor *                 token_embedding = nullptr;
    std::vector<TextBlockWeights> blocks;
    Conv1dWeights                 projection;
};

struct VoiceWeights {
    ggml_tensor * embedding = nullptr;
};

struct DDSBlockWeights {
    Conv1dWeights depthwise;
    NormWeights   depthwise_norm;
    Conv1dWeights pointwise;
    NormWeights   pointwise_norm;
};

struct DDSWeights {
    std::vector<DDSBlockWeights> blocks;
};

struct DurationFlowWeights {
    Conv1dWeights pre;
    DDSWeights    dds;
    Conv1dWeights projection;
};

struct DurationWeights {
    Conv1dWeights                    conditioning;
    Conv1dWeights                    pre;
    DDSWeights                       dds;
    Conv1dWeights                    projection;
    ggml_tensor *                    affine_bias      = nullptr;
    ggml_tensor *                    affine_log_scale = nullptr;
    std::vector<DurationFlowWeights> flows;
};

struct FlowWNLayerWeights {
    Conv1dWeights input;
    Conv1dWeights residual_skip;
};

struct FlowBlockWeights {
    Conv1dWeights                   conditioning;
    Conv1dWeights                   pre;
    std::vector<FlowWNLayerWeights> wn_layers;
    Conv1dWeights                   projection;
};

struct FlowWeights {
    std::vector<FlowBlockWeights> blocks;
};

struct TransposeConv1dWeights {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias   = nullptr;
};

struct DecoderResBlockWeights {
    std::vector<Conv1dWeights> conv1;
    std::vector<Conv1dWeights> conv2;
};

struct DecoderStageWeights {
    TransposeConv1dWeights              transpose_conv;
    std::vector<DecoderResBlockWeights> resblocks;
};

struct DecoderWeights {
    Conv1dWeights                    conditioning;
    Conv1dWeights                    pre;
    std::vector<DecoderStageWeights> stages;
    ggml_tensor *                    post_weight = nullptr;
};

synth_status_t read_hparams(const gguf_context * gguf, HParams & hparams);
synth_status_t build_voice_weights(ggml_context * context, const HParams & hparams, VoiceWeights & weights);
synth_status_t build_text_weights(ggml_context * context, const HParams & hparams, TextWeights & weights);
synth_status_t build_duration_weights(ggml_context * context, const HParams & hparams, DurationWeights & weights);
synth_status_t build_flow_weights(ggml_context * context, const HParams & hparams, FlowWeights & weights);
synth_status_t build_decoder_weights(ggml_context * context, const HParams & hparams, DecoderWeights & weights);

}  // namespace synth::vits
