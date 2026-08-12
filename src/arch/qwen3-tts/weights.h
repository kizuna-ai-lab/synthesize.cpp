#pragma once

#include "model-info.h"
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
    Q5KMixed,
};

enum class VoiceMode : uint32_t {
    PresetCatalog,   // speakers are codec-vocabulary token ids
    ProfileSources,  // no selectable Voice; every request carries a Voice Profile
};

// The ECAPA-TDNN speaker encoder Base variants carry. Its output width equals
// the talker's hidden size because the x-vector substitutes directly for the
// prompt's speaker embedding -- there is no projection between them.
struct SpeakerEncoderParams {
    uint32_t enc_dim     = 0;
    uint32_t sample_rate = 0;
    uint32_t mel_bins    = 0;
    uint32_t n_fft       = 0;
    uint32_t hop_length  = 0;
    uint32_t win_length  = 0;
    float    fmin        = 0.0f;
    float    fmax        = 0.0f;
};

// Read and validated here so the package is whole from its first cut; the Voice
// Profile module enforces these from Plan 2 on.
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

// The codec decoder's geometry. The speech tokenizer's encoder half is
// deliberately absent from the package: synthesis runs codes to audio only, this
// checkpoint carries no speaker encoder to clone with, and the encoder's first
// sixteen codebooks duplicated the decoder's exactly. Every codec tensor's
// expected shape is derived from these numbers.
struct CodecDecoderParams {
    uint32_t              latent_dim               = 0;
    // The residual stack's widest point, halved once per upsample rate.
    uint32_t              dim                      = 0;
    uint32_t              codebook_dim             = 0;
    uint32_t              codebook_size            = 0;
    uint32_t              quantizer_count          = 0;
    uint32_t              semantic_quantizer_count = 0;
    uint32_t              hidden_size              = 0;
    uint32_t              intermediate_size        = 0;
    uint32_t              layer_count              = 0;
    uint32_t              attention_head_count     = 0;
    uint32_t              key_value_head_count     = 0;
    uint32_t              head_dim                 = 0;
    uint32_t              sliding_window           = 0;
    float                 rms_norm_eps             = 0.0f;
    float                 rope_theta               = 0.0f;
    // One transposed convolution per rate in the residual stack, and one
    // ConvNeXt stage per ratio ahead of it. Together they multiply to the hop.
    std::vector<uint32_t> upsample_rates;
    std::vector<uint32_t> upsampling_ratios;
};

struct CodecParams {
    uint32_t sample_rate   = 0;
    uint32_t hop_length    = 0;
    float    frame_rate_hz = 0.0f;

    CodecDecoderParams decoder;
};

// Token ids the graph needs by value rather than by name.
struct SpecialTokens {
    uint32_t tts_bos         = 0;
    uint32_t tts_eos         = 0;
    uint32_t tts_pad         = 0;
    uint32_t im_start        = 0;
    uint32_t im_end          = 0;
    uint32_t assistant       = 0;
    uint32_t codec_bos       = 0;
    uint32_t codec_eos       = 0;
    uint32_t codec_pad       = 0;
    // The prompt's codec side opens with one of these: think when the request
    // names a language, nothink when it asks for auto. The two are not
    // interchangeable -- the nothink prompt carries no language token at all
    // rather than a default one, so it is one position shorter.
    uint32_t codec_think     = 0;
    uint32_t codec_nothink   = 0;
    uint32_t codec_think_bos = 0;
    uint32_t codec_think_eos = 0;
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

// The checkpoint's own decoding defaults, carried in the package rather than
// written into this port. They decide what the model says: the talker ends an
// utterance by sampling the codec end token, so the filters in front of that
// draw are part of the model's contract, and a port that reimplements them from
// memory drifts silently. This one did -- repetition_penalty was simply absent.
//
// The predictor is configured separately upstream and has no penalty of its own.
struct SamplingDefaults {
    float    temperature        = 0.0f;
    uint32_t top_k              = 0;
    float    top_p              = 0.0f;
    float    repetition_penalty = 1.0f;
};

struct HParams {
    std::string         model_variant;
    QuantizationProfile quantization_profile         = QuantizationProfile::BF16;
    uint32_t            quantization_profile_version = 1;
    uint32_t            architecture_version         = 1;

    uint32_t         input_flags          = 0;
    uint32_t         capability_flags     = 0;
    uint32_t         output_sample_rate   = 0;
    uint32_t         output_channel_count = 0;
    SamplingDefaults talker_sampling;
    SamplingDefaults predictor_sampling;
    uint64_t         max_input_tokens  = 0;
    uint64_t         max_output_frames = 0;
    float            min_speaking_rate = 0.0f;
    float            max_speaking_rate = 0.0f;

    TalkerParams        talker;
    CodePredictorParams code_predictor;
    CodecParams         codec;
    SpecialTokens       tokens;

    VoiceMode                voice_mode          = VoiceMode::PresetCatalog;
    bool                     has_package_default = false;
    std::vector<PresetVoice> preset_voices;

    // Only present for a variant with a speaker encoder (currently just Base).
    bool                 has_speaker_encoder = false;
    SpeakerEncoderParams speaker_encoder;
    ProfileContract      profile;

    std::vector<std::string> language_names;
    std::vector<uint32_t>    language_token_ids;

    bool        frontend_present = false;
    std::string frontend_provider;
    uint32_t    frontend_contract_version = 0;
};

synth_status_t read_hparams(const gguf_context * gguf, HParams & hparams);

// Resolves a preset Voice id to its catalog entry. Returns false when the id is
// not in the package, which the caller maps to SYNTH_ERR_UNSUPPORTED_VOICE --
// the ABI's only voice-error status (include/synthesize.h).
bool find_preset_voice(const HParams & hparams, const std::string & id, PresetVoice & voice);

// Resolves the codec language token for a request. A speaker carrying a dialect
// override wins over the requested language, because that is what the reference
// does; `resolved_name` reports which language was actually used.
bool resolve_language_token(const HParams &     hparams,
                            const std::string & requested,
                            const PresetVoice & voice,
                            uint32_t &          token_id,
                            std::string &       resolved_name);

// Whether this package carries a Preset Voice Catalog at all. False for a
// profile-sources package (this family's Base variant): read_voices refuses
// such a package unless its preset_count is zero and clears `preset_voices`,
// so there is no catalog to resolve a Voice id against and
// Model::resolve_voice's lookup refuses every request, named or not.
//
// Deliberately answered from `voice_mode` rather than from
// `preset_voices.empty()`: the mode is what the package DECLARES, and a
// truncated catalog that merely arrived empty is a different thing that must
// not read as this one.
//
// It has no production caller. Its two Plan 1 callers -- the capability gate
// in fill_voice_profile_capability and the catalog-less refusal in
// Model::resolve_voice -- were both removed by that branch's final review,
// each for the same reason: neither could be observed doing anything the code
// around it did not already do. Plan 2's capability gate is specified against
// `hparams.voice_mode == VoiceMode::ProfileSources` DIRECTLY, not against this
// predicate: this predicate is that condition's INVERSE (true only for
// PresetCatalog, i.e. CustomVoice, the variant with no speaker encoder that
// can prepare nothing) -- an earlier draft of the carryover named this
// predicate as the gate, backwards, and the carryover
// (docs/superpowers/plans/2026-08-12-qwen3-tts-stage-2-plan-1-carryover.md
// §1.4) records the correction that was made before Plan 2 was written. Kept,
// with no production caller, because the unit tests assert the discriminator
// itself (tests/qwen3_tts_voice_required_test.cpp).
inline bool has_preset_voice_catalog(const HParams & hparams) {
    return hparams.voice_mode == VoiceMode::PresetCatalog;
}

// Fills the Voice Profile capability fields this package supports -- what
// Model::get_info reports through synth::VoiceProfileInfo. Unlike OmniVoice,
// which keeps its family ModelInfo to the raw ProfileContract and lets
// src/synthesize.cpp's shared_info assemble VoiceProfileInfo inline at the
// seam, this family assembles the whole struct here, in the family layer, and
// shared_info just copies the result through -- see src/model-info.h's own
// VoiceProfileInfo doc comment for why the two routes differ
// (unit-testability without a loaded Model).
//
// Gated on `hparams.voice_mode == VoiceMode::ProfileSources`, NOT on
// `has_preset_voice_catalog(hparams)` above -- that predicate is this
// condition's INVERSE (true only for PresetCatalog, i.e. CustomVoice, the
// variant with no speaker encoder that can prepare nothing). An earlier draft
// of this plan named the wrong one; the carryover
// (docs/superpowers/plans/2026-08-12-qwen3-tts-stage-2-plan-1-carryover.md
// §1.4) records the correction made before this function was written this
// way.
//
// A ProfileSources package (Base) publishes
// SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO with
// SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE alongside it -- docs/c-interface.md
// requires the second bit on any Model that can create a v1 Profile, because
// every successfully prepared v1 Profile can be serialized -- published from
// hparams.profile's already-validated limits. `reference_transcript` and
// `reference_language` stay SYNTH_REQUIREMENT_UNSUPPORTED: this rung
// implements the x-vector clone mode only, and D4 (this plan's own ruling)
// fixes the clone mode at preparation, so an optional transcript would invite
// a caller to pass one and receive the weaker clone it did not ask for. Plan
// 3 flips both to OPTIONAL in the same change that lands transcript-assisted
// (ICL) cloning.
//
// A CustomVoice package (PresetCatalog) reports NOTHING: zero source flags
// and, with them, zero in every field that describes a source, per
// docs/c-interface.md's required shape for a Model with no runtime Voice
// Profile support -- it has no speaker encoder to prepare anything from. A
// Base package's ProfileContract and speaker-encoder metadata are still read
// and validated in full at load time regardless (read_hparams): the package
// declaring a contract and the runtime advertising a capability are different
// statements. Description Text and Random Seed stay unadvertised at every
// stage of this family's ladder; neither has an implementation here.
void fill_voice_profile_capability(const HParams & hparams, VoiceProfileInfo & info);

}  // namespace synth::qwen3tts
