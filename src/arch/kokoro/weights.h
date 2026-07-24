#pragma once

#include "synthesize.h"
#include "text-frontend.h"

#include <cstdint>
#include <string>
#include <vector>

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

}  // namespace synth::kokoro
