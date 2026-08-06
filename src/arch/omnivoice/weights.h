#pragma once

#include "synthesize.h"

#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;

namespace synth::omnivoice {

// F32 is the source profile: the checkpoint stores both halves in it, and the
// converter never produces anything else. Q8Mixed and F16 are Plan 4's
// codec-only Quantization Profiles -- every generator tensor and the RVQ
// stay at F32 regardless (src/arch/omnivoice/quantization.h's QuantRole), so
// this enum governs the codec's own matrix weights only. Q8Mixed packs a
// MatrixWeight conv kernel's [kernel, in, out] into a flattened
// [kernel * in, out] row before quantizing; F16 does not -- the tool's
// profile table gives it TensorLayout::Native (tools/synthesize-quantize/
// policy.cpp:14-24), so an F16 MatrixWeight tensor keeps its native
// three-axis shape, just halved, and never exercises the packed-shape branch
// catalog.cpp's find() or reference-encoder.cpp's feat_conv check carry for
// Q8Mixed.
enum class QuantizationProfile : uint32_t {
    F32,
    Q8Mixed,
    F16,
};

// The mask-predict generator: a Qwen3 block stack run bidirectionally over the
// whole canvas. No KV cache exists because no position is ever causal.
struct GeneratorParams {
    uint32_t layer_count          = 0;
    uint32_t hidden_size          = 0;
    uint32_t attention_head_count = 0;
    uint32_t key_value_head_count = 0;
    uint32_t head_dim             = 0;
    uint32_t intermediate_size    = 0;
    uint32_t text_vocab_size      = 0;
    float    rms_norm_eps         = 0.0f;
    float    rope_theta           = 0.0f;
};

// The 8-codebook token canvas the generator predicts into.
struct AudioCanvasParams {
    uint32_t num_codebooks = 0;
    uint32_t vocab_size    = 0;  // 1025: 1024 codes plus the mask id
    uint32_t mask_id       = 0;
};

// The Higgs Audio V2 codec halves this package carries in full: DAC decoder
// (the vocoder), and the encode path for cloning.
struct CodecParams {
    uint32_t              sample_rate          = 0;
    uint32_t              hop_length           = 0;
    float                 frame_rate_hz        = 0.0f;
    uint32_t              decoder_hidden_size  = 0;
    uint32_t              encoder_hidden_size  = 0;
    uint32_t              hidden_size          = 0;  // fc2's output width
    uint32_t              codebook_dim         = 0;
    uint32_t              codebook_size        = 0;
    uint32_t              semantic_sample_rate = 0;
    std::vector<uint32_t> upsampling_ratios;
};

// The HuBERT semantic branch, used only when preparing a cloning profile.
struct SemanticParams {
    uint32_t              hidden_size          = 0;
    uint32_t              layer_count          = 0;
    uint32_t              attention_head_count = 0;
    uint32_t              intermediate_size    = 0;
    float                 layer_norm_eps       = 0.0f;
    std::vector<uint32_t> conv_dim;
    std::vector<uint32_t> conv_kernel;
    std::vector<uint32_t> conv_stride;
};

// The seven prompt markers plus eos/pad, by value. All are added tokens past
// the base vocabulary.
struct SpecialTokens {
    uint32_t denoise        = 0;
    uint32_t lang_start     = 0;
    uint32_t lang_end       = 0;
    uint32_t instruct_start = 0;
    uint32_t instruct_end   = 0;
    uint32_t text_start     = 0;
    uint32_t text_end       = 0;
    uint32_t eos            = 0;
    uint32_t pad            = 0;
};

// The checkpoint's own decoding defaults, carried in the package rather than
// restated here. position_temperature 5.0 means the public path samples;
// both temperatures zero is the deterministic mode the goldens use.
struct GenerationDefaults {
    uint32_t num_step             = 0;
    float    guidance_scale       = 0.0f;
    float    t_shift              = 0.0f;
    float    layer_penalty_factor = 0.0f;
    float    position_temperature = 0.0f;
    float    class_temperature    = 0.0f;
};

// The Reference Audio limits and Serialized Profile identity the Voice Profile
// module enforces from Plan 3 on; read and validated from Plan 1 so a package
// is whole from its first cut.
struct ProfileContract {
    std::string schema;
    uint32_t    schema_version = 0;
    std::string compatibility_id_hex;  // 64 hex chars = 32 bytes
    uint32_t    reference_sample_rate = 0;
    uint32_t    reference_channels    = 0;
    uint64_t    min_frames_per_clip   = 0;
    uint64_t    max_frames_per_clip   = 0;
    uint64_t    max_total_frames      = 0;
    uint64_t    max_reference_count   = 0;
};

struct HParams {
    std::string         model_variant;
    QuantizationProfile quantization_profile         = QuantizationProfile::F32;
    uint32_t            quantization_profile_version = 1;
    uint32_t            architecture_version         = 1;

    uint32_t input_flags          = 0;
    uint32_t capability_flags     = 0;
    uint32_t output_sample_rate   = 0;
    uint32_t output_channel_count = 0;
    uint64_t max_input_tokens     = 0;
    uint64_t max_output_frames    = 0;
    float    min_speaking_rate    = 0.0f;
    float    max_speaking_rate    = 0.0f;

    GeneratorParams    generator;
    AudioCanvasParams  audio;
    CodecParams        codec;
    SemanticParams     semantic;
    SpecialTokens      tokens;
    GenerationDefaults generation;
    ProfileContract    profile;

    // Validated languages as BCP-47 tags. For this family the tag is also the
    // exact text the prompt's language slot carries, so no name bridge exists.
    std::vector<std::string> language_tags;

    bool        frontend_present = false;
    std::string frontend_provider;
    uint32_t    frontend_contract_version = 0;
};

synth_status_t read_hparams(const gguf_context * gguf, HParams & hparams);

}  // namespace synth::omnivoice
