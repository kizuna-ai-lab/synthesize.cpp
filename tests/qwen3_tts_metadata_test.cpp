// Package-metadata contract for the qwen3-tts family.
//
// Every check here is one a malformed package could otherwise pass silently.
// The family's own hazards are the ones worth naming: the source profile is
// BF16 rather than F32 because the talker checkpoint stores bfloat16; the
// declared rope must be plain, because the checkpoint's mrope_section is inert
// and building the sectioned form would be wrong; the two heads must agree on
// how many codes a frame holds; and a dialect speaker must resolve to a
// language token that exists.

#include "arch/qwen3-tts/weights.h"
#include "gguf.h"
#include "test-assert.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace {

struct GgufDeleter {
    void operator()(gguf_context * context) const {
        if (context != nullptr) {
            gguf_free(context);
        }
    }
};

using GgufContext = std::unique_ptr<gguf_context, GgufDeleter>;

constexpr uint32_t kCodecVocab = 3072;
constexpr uint32_t kSampleRate = 24000;
constexpr uint32_t kHopLength  = 1920;
constexpr uint32_t kCodeGroups = 16;

void set_string_array(gguf_context * g, const char * key, const std::vector<std::string> & values) {
    std::vector<const char *> raw;
    raw.reserve(values.size());
    for (const std::string & value : values) {
        raw.push_back(value.c_str());
    }
    gguf_set_arr_str(g, key, raw.data(), static_cast<int>(raw.size()));
}

// A small but structurally faithful package: nine speakers of which two pin a
// dialect, twelve codec languages, and a frame of exactly 1920 samples at
// 12.5 Hz over 24 kHz.
GgufContext valid_metadata() {
    GgufContext    c(gguf_init_empty());
    gguf_context * g = c.get();

    gguf_set_val_str(g, "general.architecture", "qwen3-tts");
    gguf_set_val_str(g, "synthesize.model_family", "qwen3-tts");
    gguf_set_val_str(g, "synthesize.model_variant", "qwen3-tts-12hz-0-6b-customvoice");
    gguf_set_val_u32(g, "synthesize.format_version", 1);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.architecture_version", 1);
    gguf_set_val_str(g, "synthesize.quantization.profile", "BF16");
    gguf_set_val_u32(g, "synthesize.quantization.profile_version", 1);

    gguf_set_val_u32(g, "synthesize.capabilities.input_flags", 1);
    gguf_set_val_u32(g, "synthesize.capabilities.flags", 2);
    gguf_set_val_u64(g, "synthesize.capabilities.max_input_tokens", 1024);
    gguf_set_val_u64(g, "synthesize.capabilities.max_output_frames", 15728640);
    // The checkpoint's shipped decoding defaults. They are required rather than
    // defaulted: this port reimplemented the filter chain and left the
    // repetition penalty out, which truncated long inputs mid-sentence.
    gguf_set_val_f32(g, "synthesize.qwen3-tts.sampling.temperature", 0.9f);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.sampling.top_k", 50);
    gguf_set_val_f32(g, "synthesize.qwen3-tts.sampling.top_p", 1.0f);
    gguf_set_val_f32(g, "synthesize.qwen3-tts.sampling.repetition_penalty", 1.05f);
    gguf_set_val_f32(g, "synthesize.qwen3-tts.sampling.predictor.temperature", 0.9f);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.sampling.predictor.top_k", 50);
    gguf_set_val_f32(g, "synthesize.qwen3-tts.sampling.predictor.top_p", 1.0f);
    gguf_set_val_f32(g, "synthesize.capabilities.min_speaking_rate", 1.0f);
    gguf_set_val_f32(g, "synthesize.capabilities.max_speaking_rate", 1.0f);
    gguf_set_val_u32(g, "synthesize.audio.sample_rate_hz", kSampleRate);
    gguf_set_val_u32(g, "synthesize.audio.channels", 1);
    gguf_set_val_str(g, "synthesize.audio.sample_format", "f32le");

    gguf_set_val_u32(g, "synthesize.qwen3-tts.talker.layer_count", 28);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.talker.hidden_size", 1024);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.talker.attention_head_count", 16);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.talker.key_value_head_count", 8);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.talker.head_dim", 128);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.talker.intermediate_size", 3072);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.talker.codec_vocab_size", kCodecVocab);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.talker.text_vocab_size", 151936);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.talker.text_hidden_size", 2048);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.talker.code_group_count", kCodeGroups);
    gguf_set_val_f32(g, "synthesize.qwen3-tts.talker.rms_norm_eps", 1e-6f);
    gguf_set_val_f32(g, "synthesize.qwen3-tts.talker.rope_theta", 1000000.0f);
    gguf_set_val_str(g, "synthesize.qwen3-tts.talker.rope_type", "1d");

    gguf_set_val_u32(g, "synthesize.qwen3-tts.code_predictor.layer_count", 5);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.code_predictor.hidden_size", 1024);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.code_predictor.attention_head_count", 16);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.code_predictor.key_value_head_count", 8);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.code_predictor.head_dim", 128);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.code_predictor.vocab_size", 2048);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.code_predictor.code_group_count", kCodeGroups);

    gguf_set_val_u32(g, "synthesize.qwen3-tts.codec.sample_rate", kSampleRate);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.codec.hop_length", kHopLength);
    gguf_set_val_f32(g, "synthesize.qwen3-tts.codec.frame_rate_hz", 12.5f);

    // The decoder's real geometry: four residual stages at 8, 5, 4 and 3, two
    // ConvNeXt stages at 2, and 8 x 5 x 4 x 3 x 2 x 2 = 1920 samples a frame.
    gguf_set_val_u32(g, "synthesize.qwen3-tts.codec.decoder.latent_dim", 1024);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.codec.decoder.dim", 1536);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.codec.decoder.codebook_dim", 512);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.codec.decoder.codebook_size", 2048);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.codec.decoder.quantizer_count", kCodeGroups);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.codec.decoder.semantic_quantizer_count", 1);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.codec.decoder.hidden_size", 512);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.codec.decoder.intermediate_size", 1024);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.codec.decoder.layer_count", 8);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.codec.decoder.attention_head_count", 16);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.codec.decoder.key_value_head_count", 16);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.codec.decoder.head_dim", 64);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.codec.decoder.sliding_window", 72);
    gguf_set_val_f32(g, "synthesize.qwen3-tts.codec.decoder.rms_norm_eps", 1e-5f);
    gguf_set_val_f32(g, "synthesize.qwen3-tts.codec.decoder.rope_theta", 10000.0f);
    const int32_t rates[]  = { 8, 5, 4, 3 };
    const int32_t ratios[] = { 2, 2 };
    gguf_set_arr_data(g, "synthesize.qwen3-tts.codec.decoder.upsample_rates", GGUF_TYPE_INT32, rates, 4);
    gguf_set_arr_data(g, "synthesize.qwen3-tts.codec.decoder.upsampling_ratios", GGUF_TYPE_INT32, ratios, 2);

    gguf_set_val_u32(g, "synthesize.qwen3-tts.token.tts_bos_token_id", 151672);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.token.tts_eos_token_id", 151673);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.token.tts_pad_token_id", 151671);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.token.im_start_token_id", 151644);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.token.im_end_token_id", 151645);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.token.assistant_token_id", 77091);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.token.codec_bos_id", 2149);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.token.codec_eos_token_id", 2150);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.token.codec_pad_id", 2148);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.token.codec_think_id", 2154);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.token.codec_nothink_id", 2155);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.token.codec_think_bos_id", 2156);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.token.codec_think_eos_id", 2157);

    const std::vector<std::string> speakers    = { "aiden",    "sohee", "ono_anna", "eric",  "dylan",
                                                   "uncle_fu", "ryan",  "vivian",   "serena" };
    const std::vector<int32_t>     speaker_ids = { 2861, 2864, 2873, 2875, 2878, 3010, 3061, 3065, 3066 };
    const std::vector<std::string> dialects    = { "", "", "", "sichuan_dialect", "beijing_dialect", "", "", "", "" };
    set_string_array(g, "synthesize.qwen3-tts.speakers.names", speakers);
    gguf_set_arr_data(g, "synthesize.qwen3-tts.speakers.token_ids", GGUF_TYPE_INT32, speaker_ids.data(),
                      static_cast<int>(speaker_ids.size()));
    set_string_array(g, "synthesize.qwen3-tts.speakers.dialect_override", dialects);

    gguf_set_val_str(g, "synthesize.voice.mode", "preset-catalog");
    gguf_set_val_bool(g, "synthesize.voice.has_package_default", false);
    gguf_set_val_u32(g, "synthesize.voice.preset_count", static_cast<uint32_t>(speakers.size()));
    for (size_t index = 0; index < speakers.size(); ++index) {
        const std::string prefix = "synthesize.voice." + std::to_string(index) + ".";
        gguf_set_val_str(g, (prefix + "id").c_str(), speakers[index].c_str());
        gguf_set_val_u32(g, (prefix + "flags").c_str(), 0);
    }

    const std::vector<std::string> languages    = { "english",  "german",  "spanish",         "chinese",
                                                    "japanese", "french",  "sichuan_dialect", "korean",
                                                    "russian",  "italian", "portuguese",      "beijing_dialect" };
    const std::vector<int32_t>     language_ids = {
        2050, 2053, 2054, 2055, 2058, 2061, 2062, 2064, 2069, 2070, 2071, 2074
    };
    set_string_array(g, "synthesize.qwen3-tts.languages.names", languages);
    gguf_set_arr_data(g, "synthesize.qwen3-tts.languages.token_ids", GGUF_TYPE_INT32, language_ids.data(),
                      static_cast<int>(language_ids.size()));

    gguf_set_val_bool(g, "synthesize.frontend.present", true);
    gguf_set_val_str(g, "synthesize.frontend.provider", "synthesize.qwen_bpe");
    gguf_set_val_u32(g, "synthesize.frontend.contract_version", 1);
    return c;
}

// A Base package: no speakers at all, and a Voice Profile contract instead.
GgufContext base_metadata() {
    GgufContext    c = valid_metadata();
    gguf_context * g = c.get();
    gguf_set_val_str(g, "synthesize.model_variant", "qwen3-tts-12hz-0-6b-base");
    gguf_set_val_str(g, "synthesize.voice.mode", "profile-sources");
    gguf_set_val_u32(g, "synthesize.voice.preset_count", 0);
    set_string_array(g, "synthesize.qwen3-tts.speakers.names", {});
    set_string_array(g, "synthesize.voice.profile_sources", { "reference-audio" });
    gguf_set_val_str(g, "synthesize.profile.schema", "qwen3-tts-voice-clone");
    gguf_set_val_u32(g, "synthesize.profile.schema_version", 1);
    gguf_set_val_str(g, "synthesize.profile.compatibility_id", std::string(64, 'a').c_str());
    gguf_set_val_u32(g, "synthesize.reference.target_sample_rate", 24000);
    gguf_set_val_u32(g, "synthesize.reference.target_channels", 1);
    gguf_set_val_u64(g, "synthesize.reference.min_frames_per_clip", 24000);
    gguf_set_val_u64(g, "synthesize.reference.max_frames_per_clip", 720000);
    gguf_set_val_u64(g, "synthesize.reference.max_total_frames", 720000);
    gguf_set_val_u64(g, "synthesize.reference.max_reference_count", 1);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.speaker_encoder.enc_dim", 1024);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.speaker_encoder.sample_rate", 24000);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.speaker_encoder.mel_bins", 128);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.speaker_encoder.n_fft", 1024);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.speaker_encoder.hop_length", 256);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.speaker_encoder.win_length", 1024);
    gguf_set_val_f32(g, "synthesize.qwen3-tts.speaker_encoder.fmin", 0.0f);
    gguf_set_val_f32(g, "synthesize.qwen3-tts.speaker_encoder.fmax", 12000.0f);
    return c;
}

// A Base-shaped package that never got its Voice Profile contract or speaker
// encoder written: everything base_metadata() sets, minus the
// synthesize.profile.* and synthesize.qwen3-tts.speaker_encoder.* blocks.
// Exercises the truncation read_voices's profile-sources comment warns
// against -- a package that claims every request must carry a Voice Profile
// but carries nothing to validate one against.
GgufContext truncated_base_metadata() {
    GgufContext    c = valid_metadata();
    gguf_context * g = c.get();
    gguf_set_val_str(g, "synthesize.model_variant", "qwen3-tts-12hz-0-6b-base");
    gguf_set_val_str(g, "synthesize.voice.mode", "profile-sources");
    gguf_set_val_u32(g, "synthesize.voice.preset_count", 0);
    set_string_array(g, "synthesize.qwen3-tts.speakers.names", {});
    return c;
}

// A voice_design package: profile-sources mode, zero presets, doubled talker
// dimensions, no speaker encoder, and a design Profile schema.
GgufContext voice_design_metadata() {
    GgufContext    c = valid_metadata();
    gguf_context * g = c.get();
    gguf_set_val_str(g, "synthesize.model_variant", "qwen3-tts-12hz-1-7b-voicedesign");
    gguf_set_val_str(g, "synthesize.voice.mode", "profile-sources");
    gguf_set_val_bool(g, "synthesize.voice.has_package_default", false);
    gguf_set_val_u32(g, "synthesize.voice.preset_count", 0);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.talker.hidden_size", 2048);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.talker.intermediate_size", 6144);
    // valid_metadata() writes a nine-entry Preset Voice catalog; a
    // profile-sources package has none, and the converter cannot emit these
    // keys for a variant with a zero preset_count. Removed rather than left
    // in place, so the fixture describes a package the converter could
    // actually produce.
    gguf_remove_key(g, "synthesize.qwen3-tts.speakers.names");
    gguf_remove_key(g, "synthesize.qwen3-tts.speakers.token_ids");
    gguf_remove_key(g, "synthesize.qwen3-tts.speakers.dialect_override");
    for (int index = 0; index < 9; ++index) {
        const std::string prefix = "synthesize.voice." + std::to_string(index) + ".";
        gguf_remove_key(g, (prefix + "id").c_str());
        gguf_remove_key(g, (prefix + "flags").c_str());
    }
    set_string_array(g, "synthesize.voice.profile_sources", { "description-text" });
    gguf_set_val_str(g, "synthesize.profile.schema", "qwen3-tts-voice-design");
    gguf_set_val_u32(g, "synthesize.profile.schema_version", 1);
    gguf_set_val_str(g, "synthesize.profile.compatibility_id",
                     "0000000000000000000000000000000000000000000000000000000000000001");
    return c;
}

int expect_rejected(const std::function<void(gguf_context *)> & mutate, const char * label) {
    GgufContext context = valid_metadata();
    SYNTH_TEST_CHECK(context != nullptr);
    mutate(context.get());
    synth::qwen3tts::HParams hparams;
    const synth_status_t     status = synth::qwen3tts::read_hparams(context.get(), hparams);
    if (status == SYNTH_OK) {
        std::fprintf(stderr, "expected rejection: %s\n", label);
        return 1;
    }
    return 0;
}

// Same as expect_rejected, but mutating a Base package instead of the
// CustomVoice fixture -- the profile and speaker-encoder keys only exist on
// the Base package, so their rejection paths can only be exercised from here.
int expect_base_rejected(const std::function<void(gguf_context *)> & mutate, const char * label) {
    GgufContext context = base_metadata();
    SYNTH_TEST_CHECK(context != nullptr);
    mutate(context.get());
    synth::qwen3tts::HParams hparams;
    const synth_status_t     status = synth::qwen3tts::read_hparams(context.get(), hparams);
    if (status == SYNTH_OK) {
        std::fprintf(stderr, "expected rejection: %s\n", label);
        return 1;
    }
    return 0;
}

// code_predictor.intermediate_size is the one code_predictor.* key
// valid_metadata() deliberately does not set (see its own block above,
// lines 96-102) -- every package converted before this key existed never
// declared it either, and read_code_predictor falls back to the talker's
// value for exactly that package. This asserts the fallback rather than
// merely relying on read_hparams returning SYNTH_OK, which the fallback
// value itself never gates: a wrong fallback (the talker's value plus one,
// say) still loads a structurally sound synthetic package clean, and only a
// real, gitignored package whose predictor and talker genuinely disagree
// would notice -- which is exactly why this needs its own assertion rather
// than trusting run_valid_package()'s existing SYNTH_OK check.
int test_code_predictor_intermediate_size_falls_back_to_the_talkers_when_absent() {
    GgufContext c = valid_metadata();
    // gguf_find_key returns -1 for "not found", not a falsy 0/false -- a
    // plain `!gguf_find_key(...)` would be true only when the key sits at
    // index 0, which is nearly the opposite check.
    SYNTH_TEST_CHECK(gguf_find_key(c.get(), "synthesize.qwen3-tts.code_predictor.intermediate_size") == -1);
    synth::qwen3tts::HParams hparams;
    SYNTH_TEST_CHECK(synth::qwen3tts::read_hparams(c.get(), hparams) == SYNTH_OK);
    SYNTH_TEST_CHECK(hparams.code_predictor.intermediate_size == hparams.talker.intermediate_size);
    SYNTH_TEST_CHECK(hparams.code_predictor.intermediate_size == 3072);
    return 0;
}

// The other half: a package that DOES declare the key is honoured even when
// it disagrees with the talker's -- the case a rung with genuinely disjoint
// talker/predictor widths (VoiceDesign's 1.7B) needs, and the one the
// fallback above must NOT silently override.
int test_code_predictor_intermediate_size_declared_is_honoured() {
    GgufContext c = valid_metadata();
    // valid_metadata()'s talker.intermediate_size is 3072; declare something
    // different so "read the declared value" and "fell back to the talker's"
    // are distinguishable outcomes rather than coincidentally equal.
    gguf_set_val_u32(c.get(), "synthesize.qwen3-tts.code_predictor.intermediate_size", 6144);
    synth::qwen3tts::HParams hparams;
    SYNTH_TEST_CHECK(synth::qwen3tts::read_hparams(c.get(), hparams) == SYNTH_OK);
    SYNTH_TEST_CHECK(hparams.code_predictor.intermediate_size == 6144);
    SYNTH_TEST_CHECK(hparams.code_predictor.intermediate_size != hparams.talker.intermediate_size);
    return 0;
}

int test_base_package_loads_without_any_preset_voice() {
    GgufContext              c = base_metadata();
    synth::qwen3tts::HParams hparams;
    SYNTH_TEST_CHECK(synth::qwen3tts::read_hparams(c.get(), hparams) == SYNTH_OK);
    SYNTH_TEST_CHECK(hparams.preset_voices.empty());
    SYNTH_TEST_CHECK(hparams.voice_mode == synth::qwen3tts::VoiceMode::ProfileSources);
    SYNTH_TEST_CHECK(hparams.profile.max_reference_count == 1);
    return 0;
}

// enc_dim feeds the talker's prompt slot directly. A package whose speaker
// encoder is a different width would build a prompt of the wrong shape.
int test_speaker_embedding_width_must_equal_the_talker_hidden_size() {
    GgufContext c = base_metadata();
    gguf_set_val_u32(c.get(), "synthesize.qwen3-tts.speaker_encoder.enc_dim", 512);
    synth::qwen3tts::HParams hparams;
    SYNTH_TEST_CHECK(synth::qwen3tts::read_hparams(c.get(), hparams) != SYNTH_OK);
    return 0;
}

int test_profile_only_mode_with_presets_is_refused() {
    GgufContext c = base_metadata();
    gguf_set_val_u32(c.get(), "synthesize.voice.preset_count", 9);
    synth::qwen3tts::HParams hparams;
    SYNTH_TEST_CHECK(synth::qwen3tts::read_hparams(c.get(), hparams) != SYNTH_OK);
    return 0;
}

int test_preset_catalog_mode_with_no_presets_is_still_refused() {
    GgufContext c = valid_metadata();
    gguf_set_val_u32(c.get(), "synthesize.voice.preset_count", 0);
    synth::qwen3tts::HParams hparams;
    SYNTH_TEST_CHECK(synth::qwen3tts::read_hparams(c.get(), hparams) != SYNTH_OK);
    return 0;
}

// A reference bound of zero would let a Profile be prepared from no audio.
int test_zero_reference_bounds_are_refused() {
    GgufContext c = base_metadata();
    gguf_set_val_u64(c.get(), "synthesize.reference.min_frames_per_clip", 0);
    synth::qwen3tts::HParams hparams;
    SYNTH_TEST_CHECK(synth::qwen3tts::read_hparams(c.get(), hparams) != SYNTH_OK);
    return 0;
}

// This variant's one deliberate divergence from omnivoice: omnivoice REQUIRES
// a package default (auto-voice is its default Voice), while a profile-sources
// Base package must have none, because every request has to carry a Voice
// Profile. Flipping weights.cpp's `return !hparams.has_package_default;` to
// `return true;` would leave the rest of the suite green -- this is the test
// that would catch it.
int test_profile_sources_names_a_package_default_is_refused() {
    GgufContext c = base_metadata();
    gguf_set_val_bool(c.get(), "synthesize.voice.has_package_default", true);
    synth::qwen3tts::HParams hparams;
    SYNTH_TEST_CHECK(synth::qwen3tts::read_hparams(c.get(), hparams) != SYNTH_OK);
    return 0;
}

// A package that declares profile-sources mode but never got its Voice
// Profile contract or speaker encoder written must not load clean with a
// zeroed contract and no encoder -- there would be nothing to validate a
// Voice Profile against for a model that requires one on every request.
int test_truncated_profile_sources_package_is_refused() {
    GgufContext              c = truncated_base_metadata();
    synth::qwen3tts::HParams hparams;
    SYNTH_TEST_CHECK(synth::qwen3tts::read_hparams(c.get(), hparams) != SYNTH_OK);
    return 0;
}

// Stage 3: a package declares which Profile sources it implements, and the
// loader demands exactly the blocks that declaration implies. Before this,
// `profile-sources` meant Base and therefore meant reference audio; it now
// means two variants whose sources are disjoint.
int test_profile_sources_are_declared_not_inferred() {
    // A voice_design package: description-text, no encoder block, no reference
    // limits. This must LOAD -- before Stage 3 it failed on the missing
    // speaker-encoder keys.
    {
        GgufContext              c = voice_design_metadata();
        synth::qwen3tts::HParams hparams;
        SYNTH_TEST_CHECK(synth::qwen3tts::read_hparams(c.get(), hparams) == SYNTH_OK);
        SYNTH_TEST_CHECK(hparams.voice_mode == synth::qwen3tts::VoiceMode::ProfileSources);
        SYNTH_TEST_CHECK(!hparams.has_speaker_encoder);
        SYNTH_TEST_CHECK(hparams.profile_sources == SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT);
        SYNTH_TEST_CHECK(hparams.profile.schema == "qwen3-tts-voice-design");
    }
    // Declaring description-text while carrying a speaker encoder is a package
    // that disagrees with itself. Refused, not reconciled.
    {
        GgufContext c = voice_design_metadata();
        gguf_set_val_u32(c.get(), "synthesize.qwen3-tts.speaker_encoder.enc_dim", 2048);
        synth::qwen3tts::HParams hparams;
        SYNTH_TEST_CHECK(synth::qwen3tts::read_hparams(c.get(), hparams) != SYNTH_OK);
    }
    // Declaring reference-audio without the encoder block is the same fault
    // from the other side, and was already impossible; it must stay so.
    //
    // This is a regression guard, not a witness for the `wants_reference !=
    // carries_encoder` check above it: delete that check and this case is
    // still refused, because `wants_reference` is still true here and
    // `read_speaker_encoder` still runs and still fails on the missing
    // `enc_dim` -- the package carries none of the eight speaker-encoder
    // keys, not just an inconsistent one. There is no way to construct "no
    // encoder block at all" so that only the mismatch check catches it; the
    // case above (description-text declared, `enc_dim` present) is the one
    // that actually isolates that check, by the same reasoning
    // read_speaker_encoder's own comments apply to their zero-checks.
    {
        GgufContext c = voice_design_metadata();
        set_string_array(c.get(), "synthesize.voice.profile_sources", { "reference-audio" });
        synth::qwen3tts::HParams hparams;
        SYNTH_TEST_CHECK(synth::qwen3tts::read_hparams(c.get(), hparams) != SYNTH_OK);
    }
    // An unknown source name is refused rather than ignored: a package from a
    // future converter must not load with a silently narrower capability.
    {
        GgufContext c = voice_design_metadata();
        set_string_array(c.get(), "synthesize.voice.profile_sources", { "telepathy" });
        synth::qwen3tts::HParams hparams;
        SYNTH_TEST_CHECK(synth::qwen3tts::read_hparams(c.get(), hparams) != SYNTH_OK);
    }
    // A truncated package: profile-sources mode with the declaration missing
    // entirely. It says every request must carry a Profile and offers no way
    // to make one. Tested by REMOVING the key rather than writing an empty
    // array, because a missing key is the shape a truncated package actually
    // takes and `gguf_set_arr_str` with n = 0 passes a null data pointer.
    {
        GgufContext c = voice_design_metadata();
        gguf_remove_key(c.get(), "synthesize.voice.profile_sources");
        synth::qwen3tts::HParams hparams;
        SYNTH_TEST_CHECK(synth::qwen3tts::read_hparams(c.get(), hparams) != SYNTH_OK);
    }
    return 0;
}

// External review of Stage 3 Plan 2's sibling PR, 2026-08-19: read_profile_sources
// ORs the declared names' bits together rather than refusing a combination, so
// a package declaring BOTH "reference-audio" and "description-text" -- with a
// speaker encoder attached, so the pre-existing `wants_reference ==
// carries_encoder` cross-check is satisfied -- loaded clean. Built from
// base_metadata() specifically, not voice_design_metadata(): base_metadata()
// already carries a full, self-consistent speaker encoder and reference
// contract (enc_dim 1024 == the talker's own hidden size, schema
// "qwen3-tts-voice-clone"), so this fixture isolates the exactly-one-source
// rule -- every OTHER check `read_profile_and_speaker_encoder` performs would
// pass on it unmodified; only adding "description-text" to its
// profile_sources should be what makes it fail. No real converter emits this
// shape (scripts/convert-qwen3-tts.py's profile_source_names always returns
// exactly one name), but a positively-declaring loader exists to refuse
// exactly the malformed shapes a converter would never write.
int test_mixed_profile_sources_are_refused() {
    GgufContext c = base_metadata();
    set_string_array(c.get(), "synthesize.voice.profile_sources", { "reference-audio", "description-text" });
    synth::qwen3tts::HParams hparams;
    SYNTH_TEST_CHECK(synth::qwen3tts::read_hparams(c.get(), hparams) != SYNTH_OK);
    return 0;
}

// The final whole-branch review of Stage 3 Plan 2, 2026-08-19, on the very
// check the test above pins: it was written as a comparison of the ORed bitmask
// against each single flag, which counts DISTINCT sources, while its own message
// said "declares more than one profile source". A repeated name ORs back to one
// bit, so ["description-text", "description-text"] loaded clean -- the message
// and the code disagreed, and the code was the wrong one. Both duplications are
// tested, one per source name, because the two names take different arms of the
// mapping loop and a check written on either arm's bit alone would still let one
// of them through. Each fixture is the one that ALREADY matches that name's own
// downstream shape (voice_design_metadata for description-text, base_metadata
// for reference-audio), so nothing but the duplication can be what fails: the
// single-name spelling of each is asserted to LOAD elsewhere in this file, by
// test_profile_sources_are_declared_not_inferred and by base_metadata()'s own
// use throughout.
int test_a_duplicated_profile_source_is_refused() {
    {
        GgufContext c = voice_design_metadata();
        set_string_array(c.get(), "synthesize.voice.profile_sources", { "description-text", "description-text" });
        synth::qwen3tts::HParams hparams;
        SYNTH_TEST_CHECK(synth::qwen3tts::read_hparams(c.get(), hparams) != SYNTH_OK);
    }
    {
        GgufContext c = base_metadata();
        set_string_array(c.get(), "synthesize.voice.profile_sources", { "reference-audio", "reference-audio" });
        synth::qwen3tts::HParams hparams;
        SYNTH_TEST_CHECK(synth::qwen3tts::read_hparams(c.get(), hparams) != SYNTH_OK);
    }
    return 0;
}

// The other half of the same review finding: a Description Text package that
// still carries synthesize.reference.* keys -- e.g. a converter regression
// that fails to omit them for a variant with no speaker encoder -- must be
// refused rather than silently ignored. read_profile_contract's own
// has_speaker_encoder-gated early return never inspects them, so before this
// check they passed through unread. Each of the six reference-only keys is
// tested individually, added one at a time to an otherwise-valid voice_design
// package, because a converter regression is more likely to leave a SINGLE
// stray key (a partial revert, a merge conflict) than the whole block -- and
// the loader must catch that shape too, not only a fully-reconstructed
// reference contract.
int test_description_text_package_with_surplus_reference_keys_is_refused() {
    {
        GgufContext c = voice_design_metadata();
        gguf_set_val_u32(c.get(), "synthesize.reference.target_sample_rate", 24000);
        synth::qwen3tts::HParams hparams;
        SYNTH_TEST_CHECK(synth::qwen3tts::read_hparams(c.get(), hparams) != SYNTH_OK);
    }
    {
        GgufContext c = voice_design_metadata();
        gguf_set_val_u32(c.get(), "synthesize.reference.target_channels", 1);
        synth::qwen3tts::HParams hparams;
        SYNTH_TEST_CHECK(synth::qwen3tts::read_hparams(c.get(), hparams) != SYNTH_OK);
    }
    {
        GgufContext c = voice_design_metadata();
        gguf_set_val_u64(c.get(), "synthesize.reference.min_frames_per_clip", 24000);
        synth::qwen3tts::HParams hparams;
        SYNTH_TEST_CHECK(synth::qwen3tts::read_hparams(c.get(), hparams) != SYNTH_OK);
    }
    {
        GgufContext c = voice_design_metadata();
        gguf_set_val_u64(c.get(), "synthesize.reference.max_frames_per_clip", 720000);
        synth::qwen3tts::HParams hparams;
        SYNTH_TEST_CHECK(synth::qwen3tts::read_hparams(c.get(), hparams) != SYNTH_OK);
    }
    {
        GgufContext c = voice_design_metadata();
        gguf_set_val_u64(c.get(), "synthesize.reference.max_total_frames", 720000);
        synth::qwen3tts::HParams hparams;
        SYNTH_TEST_CHECK(synth::qwen3tts::read_hparams(c.get(), hparams) != SYNTH_OK);
    }
    {
        GgufContext c = voice_design_metadata();
        gguf_set_val_u64(c.get(), "synthesize.reference.max_reference_count", 1);
        synth::qwen3tts::HParams hparams;
        SYNTH_TEST_CHECK(synth::qwen3tts::read_hparams(c.get(), hparams) != SYNTH_OK);
    }
    // The full block together must be refused too, not only each key alone.
    {
        GgufContext c = voice_design_metadata();
        gguf_set_val_u32(c.get(), "synthesize.reference.target_sample_rate", 24000);
        gguf_set_val_u32(c.get(), "synthesize.reference.target_channels", 1);
        gguf_set_val_u64(c.get(), "synthesize.reference.min_frames_per_clip", 24000);
        gguf_set_val_u64(c.get(), "synthesize.reference.max_frames_per_clip", 720000);
        gguf_set_val_u64(c.get(), "synthesize.reference.max_total_frames", 720000);
        gguf_set_val_u64(c.get(), "synthesize.reference.max_reference_count", 1);
        synth::qwen3tts::HParams hparams;
        SYNTH_TEST_CHECK(synth::qwen3tts::read_hparams(c.get(), hparams) != SYNTH_OK);
    }
    return 0;
}

// --- Model Variant kind against the package's own Voice declarations ---
//
// `synthesize.model_variant` and the Voice Mode / `synthesize.voice.profile_sources`
// pair are two independent statements about the same package, written from two
// different origins in the converter: the variant string comes from the intake
// manifest's `variant` field (scripts/convert-qwen3-tts.py), the sources come
// from the checkpoint's own `tts_model_type` (its `profile_source_names`).
// Nothing made them agree, so a package could call itself CustomVoice while
// declaring `description-text` and load clean, leaving `ModelInfo::variant` --
// which a caller may route on -- saying something the rest of the package
// contradicts.
//
// What this pins is the refusal of a MISLABELLED package, not of an
// unimplemented one. The variant string is not the implementation: Description
// Text has no tensor footprint of its own, so no string comparison can
// establish that an implementation is present. Two declarations agreeing is
// all this is and all it claims.
//
// Failures are COUNTED rather than returned at the first one, which
// SYNTH_TEST_CHECK would do. Disabling the guard in weights.cpp must make
// EVERY case below report, not just the first; a helper that aborted early
// would hide whether the remaining cases were being caught by this guard or by
// some earlier unrelated validation that happened to reject the fixture.
int expect_variant_rejected(GgufContext context, const char * variant, const char * label) {
    if (context == nullptr) {
        std::fprintf(stderr, "fixture failed to build: %s\n", label);
        return 1;
    }
    gguf_set_val_str(context.get(), "synthesize.model_variant", variant);
    synth::qwen3tts::HParams hparams;
    if (synth::qwen3tts::read_hparams(context.get(), hparams) == SYNTH_OK) {
        std::fprintf(stderr, "expected rejection: %s (variant %s)\n", label, variant);
        return 1;
    }
    return 0;
}

int expect_variant_accepted(GgufContext context, const char * variant, const char * label) {
    if (context == nullptr) {
        std::fprintf(stderr, "fixture failed to build: %s\n", label);
        return 1;
    }
    gguf_set_val_str(context.get(), "synthesize.model_variant", variant);
    synth::qwen3tts::HParams hparams;
    if (synth::qwen3tts::read_hparams(context.get(), hparams) != SYNTH_OK) {
        std::fprintf(stderr, "expected acceptance: %s (variant %s)\n", label, variant);
        return 1;
    }
    if (hparams.model_variant != variant) {
        std::fprintf(stderr, "variant not carried through: %s\n", label);
        return 1;
    }
    return 0;
}

// Every ordered pair of (fixture, wrong kind). Each fixture is otherwise
// untouched and is asserted to LOAD under its own kind by the test below, so
// the variant string is the only thing that differs between the two verdicts
// -- nothing else in the fixture can be what fails.
int test_a_variant_kind_that_contradicts_the_package_is_refused() {
    int failures = 0;

    // The reviewer's own case, and the one the Stage 3 design's 2026-08-19
    // erratum measured as still returning SYNTH_OK: a profile-sources package
    // declaring description-text that calls itself CustomVoice. The
    // preset-catalog refusal that landed in the sibling PR cannot reach this
    // -- that one fires on `voice_mode == PresetCatalog`, and this package's
    // mode is `profile-sources`.
    failures += expect_variant_rejected(voice_design_metadata(), "qwen3-tts-12hz-0-6b-customvoice",
                                        "a CustomVoice kind over a profile-sources package");
    failures += expect_variant_rejected(base_metadata(), "qwen3-tts-12hz-0-6b-customvoice",
                                        "a CustomVoice kind over a reference-audio package");

    // A Base kind names reference-audio; these two declare something else.
    failures += expect_variant_rejected(voice_design_metadata(), "qwen3-tts-12hz-0-6b-base",
                                        "a Base kind over a description-text declaration");
    failures += expect_variant_rejected(valid_metadata(), "qwen3-tts-12hz-0-6b-base",
                                        "a Base kind over a preset-catalog package");

    // A VoiceDesign kind names description-text; these two declare something else.
    failures += expect_variant_rejected(base_metadata(), "qwen3-tts-12hz-1-7b-voicedesign",
                                        "a VoiceDesign kind over a reference-audio declaration");
    failures += expect_variant_rejected(valid_metadata(), "qwen3-tts-12hz-1-7b-voicedesign",
                                        "a VoiceDesign kind over a preset-catalog package");

    // The same contradictions spelled WITHOUT a separator. Until 2026-08-20
    // the loader returned early on these while the converter refused them, so
    // the two ends disagreed about whether a bare kind name is a kind. It is.
    failures += expect_variant_rejected(voice_design_metadata(), "base",
                                        "a bare Base kind over a description-text declaration");
    failures += expect_variant_rejected(base_metadata(), "voicedesign",
                                        "a bare VoiceDesign kind over a reference-audio declaration");
    failures += expect_variant_rejected(voice_design_metadata(), "customvoice",
                                        "a bare CustomVoice kind over a profile-sources package");

    SYNTH_TEST_CHECK(failures == 0);
    return 0;
}

// The other half, and the reason the rule reads a KIND rather than a whole
// string: a future package that changes only its frame rate or parameter count
// must still load, and a kind this build has never heard of must pass through
// untouched rather than being refused for being unknown. Counted, not
// short-circuited, for the same reason as above.
int test_variant_strings_this_rule_does_not_govern_still_load() {
    int failures = 0;

    // The three committed variants, each on its own fixture.
    failures += expect_variant_accepted(valid_metadata(), "qwen3-tts-12hz-0-6b-customvoice",
                                        "the committed CustomVoice package");
    failures += expect_variant_accepted(base_metadata(), "qwen3-tts-12hz-0-6b-base", "the committed Base package");
    failures += expect_variant_accepted(voice_design_metadata(), "qwen3-tts-12hz-1-7b-voicedesign",
                                        "the committed VoiceDesign package");

    // A future SIZE and frame rate under a kind this build knows: recognized,
    // agrees, loads. A whole-string table would refuse all three of these.
    failures += expect_variant_accepted(valid_metadata(), "qwen3-tts-24hz-3b-customvoice", "a future CustomVoice size");
    failures += expect_variant_accepted(base_metadata(), "qwen3-tts-24hz-3b-base", "a future Base size");
    failures +=
        expect_variant_accepted(voice_design_metadata(), "qwen3-tts-24hz-3b-voicedesign", "a future VoiceDesign size");

    // A kind this build does not recognize is governed by nothing, so it loads
    // on whichever package shape it arrives with -- including shapes a
    // recognized kind would have been refused for.
    failures +=
        expect_variant_accepted(valid_metadata(), "qwen3-tts-24hz-3b-dialogue", "an unknown kind, preset-catalog");
    failures +=
        expect_variant_accepted(base_metadata(), "qwen3-tts-24hz-3b-dialogue", "an unknown kind, reference-audio");
    failures += expect_variant_accepted(voice_design_metadata(), "qwen3-tts-24hz-3b-dialogue",
                                        "an unknown kind, description-text");

    // A variant string with no separator is its own final segment, so it is
    // read as a kind. "synthetic" is not a known one, so it stays governed by
    // nothing; the two below are known kinds arriving on their OWN shape, so
    // they agree and load. Both changed meaning on 2026-08-20, when the loader
    // stopped returning early on a missing separator and started agreeing with
    // the converter about what such a string names.
    failures += expect_variant_accepted(voice_design_metadata(), "synthetic", "an unknown kind, no separator");
    failures += expect_variant_accepted(base_metadata(), "base", "a known kind, no separator, on its own shape");
    failures +=
        expect_variant_accepted(voice_design_metadata(), "voicedesign", "a known kind, no separator, on its own shape");

    SYNTH_TEST_CHECK(failures == 0);
    return 0;
}

int run_valid_package() {
    GgufContext context = valid_metadata();
    SYNTH_TEST_CHECK(context != nullptr);
    synth::qwen3tts::HParams hparams;
    SYNTH_TEST_CHECK(synth::qwen3tts::read_hparams(context.get(), hparams) == SYNTH_OK);

    SYNTH_TEST_CHECK(hparams.model_variant == "qwen3-tts-12hz-0-6b-customvoice");
    SYNTH_TEST_CHECK(hparams.quantization_profile == synth::qwen3tts::QuantizationProfile::BF16);
    SYNTH_TEST_CHECK(hparams.talker.layer_count == 28 && hparams.talker.head_dim == 128);
    SYNTH_TEST_CHECK(hparams.code_predictor.layer_count == 5);
    SYNTH_TEST_CHECK(hparams.codec.hop_length == kHopLength);
    SYNTH_TEST_CHECK(hparams.preset_voices.size() == 9);
    SYNTH_TEST_CHECK(hparams.language_names.size() == 12);
    SYNTH_TEST_CHECK(!hparams.has_package_default);
    // A CustomVoice package carries none of the profile/speaker-encoder keys:
    // it must keep loading exactly as it did before this task, with the
    // preset-catalog Voice Mode and no speaker encoder recorded.
    SYNTH_TEST_CHECK(hparams.voice_mode == synth::qwen3tts::VoiceMode::PresetCatalog);
    SYNTH_TEST_CHECK(!hparams.has_speaker_encoder);
    return 0;
}

int run_voice_and_language_routing() {
    GgufContext              context = valid_metadata();
    synth::qwen3tts::HParams hparams;
    SYNTH_TEST_CHECK(synth::qwen3tts::read_hparams(context.get(), hparams) == SYNTH_OK);

    synth::qwen3tts::PresetVoice voice;
    SYNTH_TEST_CHECK(synth::qwen3tts::find_preset_voice(hparams, "aiden", voice));
    SYNTH_TEST_CHECK(voice.token_id == 2861 && voice.dialect_override.empty());
    SYNTH_TEST_CHECK(!synth::qwen3tts::find_preset_voice(hparams, "nobody", voice));

    // An ordinary speaker uses the requested language.
    SYNTH_TEST_CHECK(synth::qwen3tts::find_preset_voice(hparams, "aiden", voice));
    uint32_t    token = 0;
    std::string resolved;
    SYNTH_TEST_CHECK(synth::qwen3tts::resolve_language_token(hparams, "english", voice, token, resolved));
    SYNTH_TEST_CHECK(token == 2050 && resolved == "english");

    // A dialect speaker overrides it, which is the only way those tokens are
    // reachable: they are not in the public language list.
    SYNTH_TEST_CHECK(synth::qwen3tts::find_preset_voice(hparams, "eric", voice));
    SYNTH_TEST_CHECK(synth::qwen3tts::resolve_language_token(hparams, "english", voice, token, resolved));
    SYNTH_TEST_CHECK(token == 2062 && resolved == "sichuan_dialect");

    SYNTH_TEST_CHECK(synth::qwen3tts::find_preset_voice(hparams, "dylan", voice));
    SYNTH_TEST_CHECK(synth::qwen3tts::resolve_language_token(hparams, "chinese", voice, token, resolved));
    SYNTH_TEST_CHECK(token == 2074 && resolved == "beijing_dialect");

    // An unknown language is a miss rather than a silent default.
    SYNTH_TEST_CHECK(synth::qwen3tts::find_preset_voice(hparams, "aiden", voice));
    SYNTH_TEST_CHECK(!synth::qwen3tts::resolve_language_token(hparams, "klingon", voice, token, resolved));
    return 0;
}

int run_rejections() {
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_str(g, "synthesize.quantization.profile", "F32"); },
                        "F32 is not this family's source profile") == 0);

    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_str(g, "synthesize.qwen3-tts.talker.rope_type", "mrope"); },
                        "sectioned rope is not implemented") == 0);

    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { gguf_set_val_u32(g, "synthesize.qwen3-tts.talker.key_value_head_count", 5); },
            "query heads must divide by key/value heads") == 0);

    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { gguf_set_val_u32(g, "synthesize.qwen3-tts.code_predictor.code_group_count", 15); },
            "the two heads disagree about the codes per frame") == 0);

    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.qwen3-tts.codec.hop_length", 1024); },
                        "hop and frame rate must agree with the sample rate") == 0);

    // A zero head count on the code predictor divided by zero on load rather
    // than being refused: the guard covered the shapes and not the heads, while
    // the talker's covered both.
    SYNTH_TEST_CHECK(expect_rejected(
                         [](gguf_context * g) {
                             gguf_set_val_u32(g, "synthesize.qwen3-tts.code_predictor.key_value_head_count", 0);
                         },
                         "a zero key/value head count is refused rather than divided by") == 0);

    // Special token ids have to index the vocabulary they are used against.
    // Thirteen of them were read with no range check at all, so a package naming
    // an id past the end loaded and failed later at an embedding row.
    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { gguf_set_val_u32(g, "synthesize.qwen3-tts.token.codec_eos_token_id", 4096); },
            "a codec token id beyond the codec vocabulary is refused") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { gguf_set_val_u32(g, "synthesize.qwen3-tts.token.im_start_token_id", 999999); },
            "a prompt token id beyond the text vocabulary is refused") == 0);

    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.qwen3-tts.codec.sample_rate", 16000); },
                        "codec rate must match the declared output rate") == 0);

    // Every codec tensor's shape is derived from the decoder geometry, so a
    // package whose geometry is wrong builds the wrong graph rather than failing.
    SYNTH_TEST_CHECK(expect_rejected(
                         [](gguf_context * g) {
                             const int32_t rates[] = { 8, 5, 4, 2 };
                             gguf_set_arr_data(g, "synthesize.qwen3-tts.codec.decoder.upsample_rates", GGUF_TYPE_INT32,
                                               rates, 4);
                         },
                         "the upsample factors must multiply to exactly one frame of samples") == 0);

    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.qwen3-tts.codec.decoder.dim", 1000); },
                        "the residual stack must halve once per stage") == 0);

    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { gguf_set_val_u32(g, "synthesize.qwen3-tts.codec.decoder.quantizer_count", 12); },
            "the codec must take the number of code groups the talker emits") == 0);

    SYNTH_TEST_CHECK(expect_rejected(
                         [](gguf_context * g) {
                             gguf_set_val_u32(g, "synthesize.qwen3-tts.codec.decoder.semantic_quantizer_count", 16);
                         },
                         "the semantic quantizers must leave acoustic groups behind them") == 0);

    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { gguf_set_val_u32(g, "synthesize.qwen3-tts.codec.decoder.codebook_dim", 511); },
            "the quantizer runs at half the codebook width, so it must be even") == 0);

    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { gguf_set_val_u32(g, "synthesize.qwen3-tts.codec.decoder.key_value_head_count", 5); },
            "codec query heads must divide its key/value heads") == 0);

    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { gguf_set_val_f32(g, "synthesize.capabilities.max_speaking_rate", 1.25f); },
            "this family has no speaking-rate control to promise") == 0);

    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_str(g, "synthesize.voice.3.id", "someone-else"); },
                        "voice catalog and speaker table must name the same voices") == 0);

    SYNTH_TEST_CHECK(expect_rejected(
                         [](gguf_context * g) {
                             const std::vector<std::string> shortened = { "english", "chinese" };
                             set_string_array(g, "synthesize.qwen3-tts.languages.names", shortened);
                         },
                         "a pinned dialect must exist as a language token") == 0);

    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_bool(g, "synthesize.frontend.present", false); },
                        "a raw-text family needs a frontend") == 0);

    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_str(g, "synthesize.audio.sample_format", "s16le"); },
                        "only f32le is produced") == 0);

    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.qwen3-tts.architecture_version", 2); },
                        "an unknown architecture version") == 0);

    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_bool(g, "synthesize.voice.has_package_default", true); },
                        "this family names no package default voice") == 0);

    // A preset-catalog package (valid_metadata()'s own shape) has no Voice
    // Profile contract, so a `synthesize.voice.profile_sources` declaration on
    // one is a surplus claim nothing downstream ever reads: read_hparams only
    // calls read_profile_sources when the mode is ProfileSources, so this key
    // would otherwise be silently ignored regardless of what it names -- a
    // package could claim `description-text` with no speaker encoder, no
    // codec encoder, and no ProfileContract to check the claim against, and
    // still load clean. Adding the key at all is refused, independent of its
    // contents: `description-text` and `reference-audio` both exercised, so
    // this is not merely catching an unknown-source-name typo.
    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { set_string_array(g, "synthesize.voice.profile_sources", { "description-text" }); },
            "a preset-catalog package may not declare profile_sources at all") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { set_string_array(g, "synthesize.voice.profile_sources", { "reference-audio" }); },
            "not even a source this package could plausibly carry structural evidence for") == 0);
    return 0;
}

// Every validation rule Task 7 adds for the Voice Profile contract and the
// speaker encoder, exercised from the Base fixture (the only one that carries
// these keys at all). Each case below proves one rejection branch actually
// fires, rather than trusting the prose description of the rule.
int run_base_package_rejections() {
    // --- Profile contract: schema identity ---
    SYNTH_TEST_CHECK(
        expect_base_rejected(
            [](gguf_context * g) { gguf_set_val_str(g, "synthesize.profile.schema", "omnivoice-clone-prompt"); },
            "the profile schema must be qwen3-tts-voice-clone") == 0);

    SYNTH_TEST_CHECK(
        expect_base_rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.profile.schema_version", 2); },
                             "an unknown profile schema version") == 0);

    // --- Profile contract: compatibility id is 32 bytes of lowercase hex ---
    SYNTH_TEST_CHECK(expect_base_rejected(
                         [](gguf_context * g) {
                             gguf_set_val_str(g, "synthesize.profile.compatibility_id", std::string(63, 'a').c_str());
                         },
                         "a compatibility id shorter than 64 hex chars") == 0);

    SYNTH_TEST_CHECK(expect_base_rejected(
                         [](gguf_context * g) {
                             gguf_set_val_str(g, "synthesize.profile.compatibility_id", std::string(64, 'A').c_str());
                         },
                         "an uppercase compatibility id is not lowercase hex") == 0);

    // --- Profile contract: reference audio format ---
    SYNTH_TEST_CHECK(
        expect_base_rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.reference.target_channels", 2); },
                             "reference audio must be single-channel") == 0);

    SYNTH_TEST_CHECK(expect_base_rejected(
                         [](gguf_context * g) { gguf_set_val_u32(g, "synthesize.reference.target_sample_rate", 0); },
                         "a zero reference sample rate is refused") == 0);

    // --- Profile contract: min <= max <= total, and max_reference_count == 1 ---
    SYNTH_TEST_CHECK(
        expect_base_rejected(
            [](gguf_context * g) { gguf_set_val_u64(g, "synthesize.reference.min_frames_per_clip", 800000); },
            "min_frames_per_clip may not exceed max_frames_per_clip") == 0);

    SYNTH_TEST_CHECK(expect_base_rejected(
                         [](gguf_context * g) { gguf_set_val_u64(g, "synthesize.reference.max_total_frames", 100000); },
                         "max_frames_per_clip may not exceed max_total_frames") == 0);

    SYNTH_TEST_CHECK(expect_base_rejected(
                         [](gguf_context * g) { gguf_set_val_u64(g, "synthesize.reference.max_reference_count", 0); },
                         "a zero reference count admits no clip") == 0);

    // The other side of the same rule, which nothing pinned until
    // 2026-08-13: this runtime reads references[0] and nothing else, so a
    // package declaring a HIGHER ceiling would publish a capability that
    // create_qwen3_tts_profile_from_reference then refuses every call for
    // (src/voice-profile.cpp's `reference_count != 1` gate). Two cases, both
    // above the ceiling and neither reachable by the zero check above:
    //
    // - 2, the smallest over-declaration, which nothing else in
    //   read_profile_contract has any opinion about (max_total_frames and
    //   max_frames_per_clip are untouched and still consistent, so this case
    //   isolates the reference-count rule as the one doing the rejecting);
    // - 2^32, historically the worst over-declaration, kept as a regression
    //   value rather than as its own mechanism. It takes the same `!= 1`
    //   branch as the 2 above -- the label below says only that it is
    //   refused, which is all this arm checks. What it used to do (arrive in
    //   the capability snapshot as 0, from a package that had just been
    //   accepted for declaring a NON-zero count) is a property of the
    //   internal field width, and pinning that is
    //   qwen3_tts_voice_required_test.cpp's
    //   test_a_declared_reference_count_is_published_unnarrowed, which
    //   asserts the value rather than a rejection.
    SYNTH_TEST_CHECK(expect_base_rejected(
                         [](gguf_context * g) { gguf_set_val_u64(g, "synthesize.reference.max_reference_count", 2); },
                         "this runtime reads exactly one reference clip") == 0);

    SYNTH_TEST_CHECK(expect_base_rejected(
                         [](gguf_context * g) {
                             gguf_set_val_u64(g, "synthesize.reference.max_reference_count", uint64_t(1) << 32);
                         },
                         "a reference count of 2^32 is over the ceiling like any other") == 0);

    // --- Profile contract: the reference target rate is the encoder's rate ---
    // The one mismatch in this file that changes no shape and so could only
    // ever be caught here: reference audio is resampled to
    // target_sample_rate, and the mel filterbank derives its bin frequencies
    // from the speaker encoder's sample_rate. Declare 16000 against the
    // fixture's 24000 encoder and every later rule still passes -- the frame
    // bounds are frame counts, the mel's geometry is n_fft/hop_length, the
    // x-vector's width is enc_dim -- so the package would load, advertise
    // 16000, and hand the encoder a frequency-scaled spectrum.
    //
    // 16000 rather than 0: a zero would also trip the reference_sample_rate
    // zero-check just above, which would make this case pass whether or not
    // the rate equality exists. Isolating it needs a rate that is valid on
    // its own terms and merely disagrees.
    SYNTH_TEST_CHECK(
        expect_base_rejected(
            [](gguf_context * g) { gguf_set_val_u32(g, "synthesize.reference.target_sample_rate", 16000); },
            "the reference target rate must equal the speaker encoder's rate") == 0);

    // --- Speaker encoder: the three widths only the zero-check catches ---
    // read_speaker_encoder's zero-check covers mel_bins, n_fft and hop_length
    // and nothing else: for each of these, deleting its term from that
    // condition makes the package load. The three that used to sit alongside
    // them (enc_dim, sample_rate, win_length) are checked further down
    // instead, and are exercised by their own cases below.
    SYNTH_TEST_CHECK(expect_base_rejected(
                         [](gguf_context * g) { gguf_set_val_u32(g, "synthesize.qwen3-tts.speaker_encoder.n_fft", 0); },
                         "n_fft must be non-zero") == 0);

    // The mel front end's FFT is radix-2 (src/arch/qwen3-tts/mel.cpp); a
    // non-power-of-two n_fft must be refused at load time rather than
    // reaching compute_log_mel's own runtime check.
    //
    // win_length comes down to 512 alongside it, for the same reason the
    // sibling case in tests/qwen3_tts_mel_test.cpp does: base_metadata()
    // leaves win_length at 1024, and at n_fft 1000 the separate
    // win_length-vs-n_fft rule below (1024 > 1000) rejects the very same
    // package with the very same SYNTH_ERR_GGUF -- so a case that left
    // win_length alone still passed with the power-of-two rule deleted,
    // proving nothing. 512 stays under both 1000 and the fixture's hop of
    // 256 stays under 512, so this case now isolates the power-of-two rule
    // as the one doing the rejecting.
    SYNTH_TEST_CHECK(expect_base_rejected(
                         [](gguf_context * g) {
                             gguf_set_val_u32(g, "synthesize.qwen3-tts.speaker_encoder.n_fft", 1000);
                             gguf_set_val_u32(g, "synthesize.qwen3-tts.speaker_encoder.win_length", 512);
                         },
                         "non-power-of-two n_fft") == 0);

    // The other structural constraint the radix-2 transform imposes: a
    // window wider than the transform has no meaning for the zero-padded-
    // centred rule. n_fft stays at the fixture's own 1024 (a valid power of
    // two) and hop_length stays at 256 (comfortably under win_length here
    // too), so this case isolates win_length > n_fft as the one rule doing
    // the rejecting, rather than incidentally tripping n_fft's own checks.
    SYNTH_TEST_CHECK(
        expect_base_rejected(
            [](gguf_context * g) { gguf_set_val_u32(g, "synthesize.qwen3-tts.speaker_encoder.win_length", 2048); },
            "win_length exceeding n_fft") == 0);

    SYNTH_TEST_CHECK(
        expect_base_rejected(
            [](gguf_context * g) { gguf_set_val_u32(g, "synthesize.qwen3-tts.speaker_encoder.mel_bins", 0); },
            "mel_bins must be non-zero") == 0);

    // A zero hop advances the mel window by nothing and would frame forever.
    // Distinct from the hop_length >= win_length case below: at hop 0 that
    // rule is satisfied (0 < 1024), so only the zero-check refuses this.
    SYNTH_TEST_CHECK(
        expect_base_rejected(
            [](gguf_context * g) { gguf_set_val_u32(g, "synthesize.qwen3-tts.speaker_encoder.hop_length", 0); },
            "a zero hop_length is refused even though it is below win_length") == 0);

    // --- Speaker encoder: width must equal the talker's hidden size ---
    // Zero included: read_talker already refuses a zero hidden size, so a
    // zero enc_dim can only ever be unequal to it.
    SYNTH_TEST_CHECK(
        expect_base_rejected(
            [](gguf_context * g) { gguf_set_val_u32(g, "synthesize.qwen3-tts.speaker_encoder.enc_dim", 0); },
            "a zero speaker-encoder enc_dim cannot equal the talker hidden size") == 0);

    // --- Speaker encoder: rate must match the codec's own rate ---
    // Zero included, for the same reason: read_codec already refuses a codec
    // rate of zero (it could not produce the declared frame rate).
    SYNTH_TEST_CHECK(
        expect_base_rejected(
            [](gguf_context * g) { gguf_set_val_u32(g, "synthesize.qwen3-tts.speaker_encoder.sample_rate", 16000); },
            "the speaker encoder must run at the codec's sample rate") == 0);

    SYNTH_TEST_CHECK(
        expect_base_rejected(
            [](gguf_context * g) { gguf_set_val_u32(g, "synthesize.qwen3-tts.speaker_encoder.sample_rate", 0); },
            "a zero speaker-encoder sample rate cannot equal the codec's") == 0);

    // --- Speaker encoder: hop_length < win_length ---
    SYNTH_TEST_CHECK(
        expect_base_rejected(
            [](gguf_context * g) { gguf_set_val_u32(g, "synthesize.qwen3-tts.speaker_encoder.hop_length", 1024); },
            "hop_length equal to win_length is refused") == 0);

    // Zero win_length is this same rule, not a separate zero-check: every
    // hop_length is >= 0.
    SYNTH_TEST_CHECK(
        expect_base_rejected(
            [](gguf_context * g) { gguf_set_val_u32(g, "synthesize.qwen3-tts.speaker_encoder.win_length", 0); },
            "a zero win_length leaves no hop inside it") == 0);

    // --- Speaker encoder: fmax > fmin and fmax <= sample_rate / 2 ---
    SYNTH_TEST_CHECK(
        expect_base_rejected(
            [](gguf_context * g) { gguf_set_val_f32(g, "synthesize.qwen3-tts.speaker_encoder.fmax", 0.0f); },
            "fmax must be strictly greater than fmin") == 0);

    SYNTH_TEST_CHECK(
        expect_base_rejected(
            [](gguf_context * g) { gguf_set_val_f32(g, "synthesize.qwen3-tts.speaker_encoder.fmax", 13000.0f); },
            "fmax may not exceed sample_rate / 2") == 0);

    return 0;
}

}  // namespace

int main() {
    synth::qwen3tts::HParams hparams;
    SYNTH_TEST_CHECK(synth::qwen3tts::read_hparams(nullptr, hparams) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(run_valid_package() == 0);
    SYNTH_TEST_CHECK(test_code_predictor_intermediate_size_falls_back_to_the_talkers_when_absent() == 0);
    SYNTH_TEST_CHECK(test_code_predictor_intermediate_size_declared_is_honoured() == 0);
    SYNTH_TEST_CHECK(run_voice_and_language_routing() == 0);
    SYNTH_TEST_CHECK(run_rejections() == 0);
    SYNTH_TEST_CHECK(run_base_package_rejections() == 0);
    SYNTH_TEST_CHECK(test_base_package_loads_without_any_preset_voice() == 0);
    SYNTH_TEST_CHECK(test_speaker_embedding_width_must_equal_the_talker_hidden_size() == 0);
    SYNTH_TEST_CHECK(test_profile_only_mode_with_presets_is_refused() == 0);
    SYNTH_TEST_CHECK(test_preset_catalog_mode_with_no_presets_is_still_refused() == 0);
    SYNTH_TEST_CHECK(test_zero_reference_bounds_are_refused() == 0);
    SYNTH_TEST_CHECK(test_profile_sources_names_a_package_default_is_refused() == 0);
    SYNTH_TEST_CHECK(test_truncated_profile_sources_package_is_refused() == 0);
    SYNTH_TEST_CHECK(test_profile_sources_are_declared_not_inferred() == 0);
    SYNTH_TEST_CHECK(test_mixed_profile_sources_are_refused() == 0);
    SYNTH_TEST_CHECK(test_a_duplicated_profile_source_is_refused() == 0);
    SYNTH_TEST_CHECK(test_description_text_package_with_surplus_reference_keys_is_refused() == 0);
    SYNTH_TEST_CHECK(test_a_variant_kind_that_contradicts_the_package_is_refused() == 0);
    SYNTH_TEST_CHECK(test_variant_strings_this_rule_does_not_govern_still_load() == 0);
    return 0;
}
