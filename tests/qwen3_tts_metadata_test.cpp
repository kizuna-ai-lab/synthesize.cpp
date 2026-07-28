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
    return 0;
}

}  // namespace

int main() {
    synth::qwen3tts::HParams hparams;
    SYNTH_TEST_CHECK(synth::qwen3tts::read_hparams(nullptr, hparams) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(run_valid_package() == 0);
    SYNTH_TEST_CHECK(run_voice_and_language_routing() == 0);
    SYNTH_TEST_CHECK(run_rejections() == 0);
    return 0;
}
