#pragma once

#include "synthesize.h"

#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;

namespace synth::qwen3tts {

enum class QuantizationProfile : uint32_t {
    BF16,
    F16,
    Q8Mixed,
};

// The autoregressive language model that emits one semantic code per frame.
struct TalkerParams {
    uint32_t    layer_count          = 0;
    uint32_t    hidden_size          = 0;
    uint32_t    attention_head_count = 0;
    uint32_t    key_value_head_count = 0;
    uint32_t    head_dim             = 0;
    uint32_t    intermediate_size    = 0;
    uint32_t    codec_vocab_size     = 0;
    uint32_t    text_vocab_size      = 0;
    uint32_t    text_hidden_size     = 0;
    uint32_t    code_group_count     = 0;
    float       rms_norm_eps         = 0.0f;
    float       rope_theta           = 0.0f;
    // The package declares "1d" because the checkpoint's own mrope_section is
    // inert: every position_ids path in the reference yields three identical
    // rows, which makes the interleaved form exactly plain rope. Loading
    // refuses any other value rather than silently building the wrong graph.
    std::string rope_type;
};

// The multi-token-prediction head that expands one semantic code into the
// remaining acoustic codes of the same frame. It runs `code_group_count - 1`
// sequential steps per frame, so it, not the talker, is the hot loop.
struct CodePredictorParams {
    uint32_t layer_count          = 0;
    uint32_t hidden_size          = 0;
    uint32_t attention_head_count = 0;
    uint32_t key_value_head_count = 0;
    uint32_t head_dim             = 0;
    uint32_t vocab_size           = 0;
    uint32_t code_group_count     = 0;
};

struct CodecParams {
    uint32_t sample_rate   = 0;
    uint32_t hop_length    = 0;
    float    frame_rate_hz = 0.0f;
};

// Token ids the graph needs by value rather than by name.
struct SpecialTokens {
    uint32_t tts_bos   = 0;
    uint32_t tts_eos   = 0;
    uint32_t tts_pad   = 0;
    uint32_t im_start  = 0;
    uint32_t im_end    = 0;
    uint32_t assistant = 0;
    uint32_t codec_bos = 0;
    uint32_t codec_eos = 0;
    uint32_t codec_pad = 0;
};

// A preset Voice is a codec-vocabulary token id, not a row of an embedding
// table, which is why this variant carries no speaker encoder. Two of the nine
// speakers additionally pin a dialect language token regardless of the
// requested language; `dialect_override` is empty for the rest.
struct PresetVoice {
    std::string id;
    uint32_t    token_id = 0;
    std::string dialect_override;
    uint32_t    flags = 0;
};

struct HParams {
    std::string         model_variant;
    QuantizationProfile quantization_profile         = QuantizationProfile::BF16;
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

    TalkerParams        talker;
    CodePredictorParams code_predictor;
    CodecParams         codec;
    SpecialTokens       tokens;

    bool                     has_package_default = false;
    std::vector<PresetVoice> preset_voices;

    std::vector<std::string> language_names;
    std::vector<uint32_t>    language_token_ids;

    bool        frontend_present = false;
    std::string frontend_provider;
    uint32_t    frontend_contract_version = 0;
};

synth_status_t read_hparams(const gguf_context * gguf, HParams & hparams);

// Resolves a preset Voice id to its catalog entry. Returns false when the id is
// not in the package, which the caller maps to SYNTH_ERR_VOICE_NOT_FOUND.
bool find_preset_voice(const HParams & hparams, const std::string & id, PresetVoice & voice);

// Resolves the codec language token for a request. A speaker carrying a dialect
// override wins over the requested language, because that is what the reference
// does; `resolved_name` reports which language was actually used.
bool resolve_language_token(const HParams &     hparams,
                            const std::string & requested,
                            const PresetVoice & voice,
                            uint32_t &          token_id,
                            std::string &       resolved_name);

}  // namespace synth::qwen3tts
