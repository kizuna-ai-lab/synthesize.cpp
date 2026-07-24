#pragma once

#include "synthesize.h"
#include "text-frontend.h"

#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct gguf_context;

namespace synth::kokoro {

enum class QuantizationProfile : uint32_t {
    F32,
    F16,
    Q8Mixed,
};

// Constants of the pinned harmonic-plus-noise source module. They are read from
// the package rather than hardcoded so a later Model Variant cannot silently
// change excitation behavior without a metadata change.
struct SourceParams {
    uint32_t sampling_rate    = 0;
    uint32_t harmonic_num     = 0;
    uint32_t upsample_scale   = 0;
    float    sine_amp         = 0.0f;
    float    noise_std        = 0.0f;
    float    voiced_threshold = 0.0f;
};

struct PLBertParams {
    uint32_t hidden_size             = 0;
    uint32_t num_attention_heads     = 0;
    uint32_t intermediate_size       = 0;
    uint32_t num_hidden_layers       = 0;
    uint32_t max_position_embeddings = 0;
    // Kokoro's PL-BERT is an ALBERT: one weight group is replayed for every
    // hidden layer, so the package stores a single layer rather than twelve.
    uint32_t shared_layer_groups     = 0;
    float    layer_norm_eps          = 0.0f;
};

struct IStftNetParams {
    std::vector<uint32_t>              upsample_rates;
    std::vector<uint32_t>              upsample_kernel_sizes;
    std::vector<uint32_t>              resblock_kernel_sizes;
    std::vector<std::vector<uint32_t>> resblock_dilations;
    uint32_t                           upsample_initial_channel = 0;
    uint32_t                           gen_istft_n_fft          = 0;
    uint32_t                           gen_istft_hop_size       = 0;
    bool                               center                   = false;
};

struct HParams {
    std::string         model_variant;
    QuantizationProfile quantization_profile         = QuantizationProfile::F32;
    uint32_t            quantization_profile_version = 1;

    bool                     has_package_default = false;
    std::vector<std::string> preset_voice_ids;
    std::vector<uint32_t>    preset_voice_flags;

    bool                    frontend_present = false;
    SymbolMapFrontendConfig frontend_config;

    uint32_t input_flags          = 0;
    uint32_t capability_flags     = 0;
    uint32_t output_sample_rate   = 0;
    uint32_t output_channel_count = 0;
    uint64_t max_input_tokens     = 0;
    uint64_t max_output_frames    = 0;
    float    min_speaking_rate    = 0.0f;
    float    max_speaking_rate    = 0.0f;

    uint32_t n_token                  = 0;
    uint32_t hidden_dim               = 0;
    uint32_t style_dim                = 0;
    uint32_t n_layer                  = 0;
    uint32_t n_mels                   = 0;
    uint32_t max_dur                  = 0;
    uint32_t dim_in                   = 0;
    uint32_t text_encoder_kernel_size = 0;
    uint32_t samples_per_frame        = 0;

    PLBertParams   plbert;
    IStftNetParams istftnet;
    SourceParams   source;

    // A Voice is one row of a [rows, dim] style table selected by input length.
    uint32_t voice_rows                 = 0;
    uint32_t voice_dim                  = 0;
    uint32_t voice_decoder_offset       = 0;
    uint32_t voice_prosody_offset       = 0;
    // The checkpoint carries no InstanceNorm affine parameters, so AdaIN
    // normalizes without them and applies only its style-projected scale.
    bool     adain_instance_norm_affine = false;
    float    adain_eps                  = 0.0f;
};

// Resolves the style-table row for a final token count, including the two pad
// tokens. Returns false when the request cannot address a row.
bool resolve_style_row(const HParams & hparams, uint64_t final_token_count, uint32_t & row);

synth_status_t read_hparams(const gguf_context * gguf, HParams & hparams);

// ---------------------------------------------------------------------------
// Tensor catalog
//
// Canonical GGUF names are the upstream module paths with the DataParallel
// `module.` prefix stripped, so every entry is traceable to a checkpoint entry.
// ---------------------------------------------------------------------------

struct LinearWeights {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias   = nullptr;
};

struct NormWeights {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias   = nullptr;
};

struct Conv1dWeights {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias   = nullptr;
};

// One ALBERT layer. Kokoro stores a single group and replays it for every
// hidden layer, so this appears once rather than per layer.
struct AlbertLayerWeights {
    LinearWeights query;
    LinearWeights key;
    LinearWeights value;
    LinearWeights dense;
    NormWeights   attention_norm;
    LinearWeights ffn;
    LinearWeights ffn_output;
    NormWeights   output_norm;
};

struct PLBertWeights {
    ggml_tensor *      word_embeddings       = nullptr;
    ggml_tensor *      position_embeddings   = nullptr;
    ggml_tensor *      token_type_embeddings = nullptr;
    NormWeights        embedding_norm;
    LinearWeights      hidden_mapping;
    AlbertLayerWeights layer;
};

// AdaIN projects the style vector to a scale and shift; the normalization
// itself has no learned parameters in this checkpoint.
struct AdaINWeights {
    LinearWeights fc;
};

// AdainResBlk1d: two convolutions with AdaIN, an optional learned shortcut,
// and an optional depthwise transposed convolution when the block upsamples.
struct AdainResBlockWeights {
    Conv1dWeights conv1;
    Conv1dWeights conv2;
    AdaINWeights  norm1;
    AdaINWeights  norm2;
    ggml_tensor * conv1x1 = nullptr;  // present only when the channel count changes
    Conv1dWeights pool;               // present only when the block upsamples
};

// AdaINResBlock1: the generator's residual stack, with a Snake alpha per layer.
struct AdaINResBlock1Weights {
    std::vector<Conv1dWeights> convs1;
    std::vector<Conv1dWeights> convs2;
    std::vector<AdaINWeights>  adain1;
    std::vector<AdaINWeights>  adain2;
    std::vector<ggml_tensor *> alpha1;
    std::vector<ggml_tensor *> alpha2;
};

struct LstmDirectionTensors {
    ggml_tensor * weight_ih = nullptr;
    ggml_tensor * weight_hh = nullptr;
    ggml_tensor * bias_ih   = nullptr;
    ggml_tensor * bias_hh   = nullptr;
};

struct LstmTensors {
    LstmDirectionTensors forward;
    LstmDirectionTensors reverse;
};

struct TextEncoderWeights {
    ggml_tensor *              embedding = nullptr;
    std::vector<Conv1dWeights> cnn;
    std::vector<NormWeights>   cnn_norm;
    LstmTensors                lstm;
};

// DurationEncoder alternates LSTMs with adaptive layer norms.
struct DurationEncoderWeights {
    std::vector<LstmTensors>   lstms;
    std::vector<LinearWeights> ada_norm;
};

struct ProsodyPredictorWeights {
    DurationEncoderWeights            text_encoder;
    LstmTensors                       lstm;
    LinearWeights                     duration_proj;
    LstmTensors                       shared;
    std::vector<AdainResBlockWeights> f0;
    std::vector<AdainResBlockWeights> n;
    Conv1dWeights                     f0_proj;
    Conv1dWeights                     n_proj;
};

struct GeneratorWeights {
    LinearWeights                      source_linear;
    std::vector<Conv1dWeights>         ups;
    std::vector<Conv1dWeights>         noise_convs;
    std::vector<AdaINResBlock1Weights> noise_res;
    std::vector<AdaINResBlock1Weights> resblocks;
    Conv1dWeights                      conv_post;
};

struct DecoderWeights {
    Conv1dWeights                     f0_conv;
    Conv1dWeights                     n_conv;
    Conv1dWeights                     asr_res;
    AdainResBlockWeights              encode;
    std::vector<AdainResBlockWeights> decode;
    GeneratorWeights                  generator;
};

// One [voice_dim, voice_rows] style table per Preset Voice, in catalog order.
struct VoiceWeights {
    std::vector<ggml_tensor *> packs;
};

struct ModelWeights {
    PLBertWeights           bert;
    LinearWeights           bert_encoder;
    TextEncoderWeights      text_encoder;
    ProsodyPredictorWeights predictor;
    DecoderWeights          decoder;
    VoiceWeights            voices;
};

// Resolves every catalog entry from `context`, validating name, type and shape.
synth_status_t build_model_weights(ggml_context * context, const HParams & hparams, ModelWeights & weights);

}  // namespace synth::kokoro
