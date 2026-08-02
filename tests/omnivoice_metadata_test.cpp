// Package-metadata contract for the omnivoice family.
//
// Every check here is one a malformed package could otherwise pass silently.
// The family's own hazards are the ones worth naming: the generator is
// bidirectional over a fixed canvas, so a package declaring causal attention
// would build a graph that is wrong everywhere rather than late; the canvas's
// mask id is the last entry of its own vocabulary and one past the codec's
// codebook, so a disagreement puts the mask on a real code; the codec's
// upsampling ratios must multiply to exactly one frame of samples; and this
// family clones from Reference Audio, so the Serialized Profile contract --
// schema, compatibility id, clip limits -- has to be whole from the first cut
// even though nothing consumes it before Plan 3.

#include "arch/omnivoice/weights.h"
#include "gguf.h"
#include "test-assert.h"

#include <cstdint>
#include <cstdio>
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

constexpr uint32_t kSampleRate   = 24000;
constexpr uint32_t kHopLength    = 960;
constexpr uint32_t kTextVocab    = 151676;
constexpr uint32_t kAudioVocab   = 1025;
constexpr uint32_t kMaskId       = 1024;
constexpr uint32_t kCodebookSize = 1024;

// A sha256 hexdigest is 64 lowercase hex characters. The value itself is
// opaque to the reader; its shape is not.
constexpr const char * kCompatibilityId = "9f2c1b0a7d4e6538aa10cc93bf7712d4e08a5c63b91d47fe2085730c6ad1e4b8";

void set_string_array(gguf_context * g, const char * key, const std::vector<std::string> & values) {
    std::vector<const char *> raw;
    raw.reserve(values.size());
    for (const std::string & value : values) {
        raw.push_back(value.c_str());
    }
    gguf_set_arr_str(g, key, raw.data(), raw.size());
}

void set_i32_array(gguf_context * g, const char * key, const std::vector<int32_t> & values) {
    gguf_set_arr_data(g, key, GGUF_TYPE_INT32, values.data(), values.size());
}

// The real omnivoice-0-6b package, key for key: every `synthesize.*` key
// `scripts/convert-omnivoice.py` writes, carrying the value the pinned
// checkpoint produced.
//
// Nine of those keys are set here but deliberately unread by read_hparams: the
// six `synthesize.source.*` provenance strings and `synthesize.converter` are
// records rather than contract, and the two frontend payload arrays belong to
// the Text Frontend the load path builds, not to the hyperparameters. They are
// present so this fixture stays a faithful miniature of the package.
GgufContext valid_metadata() {
    GgufContext    c(gguf_init_empty());
    gguf_context * g = c.get();

    gguf_set_val_str(g, "general.architecture", "omnivoice");
    gguf_set_val_u32(g, "synthesize.format_version", 1);
    gguf_set_val_str(g, "synthesize.model_family", "omnivoice");
    gguf_set_val_str(g, "synthesize.model_variant", "omnivoice-0-6b");
    gguf_set_val_str(g, "synthesize.quantization.profile", "F32");
    gguf_set_val_u32(g, "synthesize.quantization.profile_version", 1);
    gguf_set_val_str(g, "synthesize.source.repository", "https://github.com/k2-fsa/OmniVoice");
    gguf_set_val_str(g, "synthesize.source.revision", "468e927ba3716cd8dd86421148dfb3046e9f9d7b");
    gguf_set_val_str(g, "synthesize.source.checkpoint.sha256", kCompatibilityId);
    gguf_set_val_str(g, "synthesize.source.codec.sha256", kCompatibilityId);
    gguf_set_val_str(g, "synthesize.source.config.sha256", kCompatibilityId);
    gguf_set_val_str(g, "synthesize.source.checkpoint.license_status",
                     "cc-by-nc (LM, version unstated) + boson-higgs-audio-2-community (codec) + apache-2.0 (code)");
    gguf_set_val_str(g, "synthesize.converter", "scripts/convert-omnivoice.py");
    gguf_set_val_u32(g, "synthesize.omnivoice.architecture_version", 1);

    // The checkpoint's own decoding defaults, from OmniVoiceGenerationConfig.
    gguf_set_val_u32(g, "synthesize.omnivoice.generation.num_step", 32);
    gguf_set_val_f32(g, "synthesize.omnivoice.generation.guidance_scale", 2.0f);
    gguf_set_val_f32(g, "synthesize.omnivoice.generation.t_shift", 0.1f);
    gguf_set_val_f32(g, "synthesize.omnivoice.generation.layer_penalty_factor", 5.0f);
    gguf_set_val_f32(g, "synthesize.omnivoice.generation.position_temperature", 5.0f);
    gguf_set_val_f32(g, "synthesize.omnivoice.generation.class_temperature", 0.0f);

    gguf_set_val_u32(g, "synthesize.capabilities.input_flags", 1);
    gguf_set_val_u32(g, "synthesize.capabilities.flags", 3);
    gguf_set_val_u64(g, "synthesize.capabilities.max_input_tokens", 2048);
    gguf_set_val_u64(g, "synthesize.capabilities.max_output_frames", 750);
    gguf_set_val_f32(g, "synthesize.capabilities.min_speaking_rate", 0.5f);
    gguf_set_val_f32(g, "synthesize.capabilities.max_speaking_rate", 2.0f);

    gguf_set_val_u32(g, "synthesize.audio.sample_rate_hz", kSampleRate);
    gguf_set_val_u32(g, "synthesize.audio.channels", 1);
    gguf_set_val_str(g, "synthesize.audio.sample_format", "f32le");

    gguf_set_val_str(g, "synthesize.voice.mode", "profile-sources");
    gguf_set_val_bool(g, "synthesize.voice.has_package_default", true);
    gguf_set_val_u32(g, "synthesize.voice.preset_count", 0);

    gguf_set_val_u32(g, "synthesize.omnivoice.generator.layer_count", 28);
    gguf_set_val_u32(g, "synthesize.omnivoice.generator.hidden_size", 1024);
    gguf_set_val_u32(g, "synthesize.omnivoice.generator.attention_head_count", 16);
    gguf_set_val_u32(g, "synthesize.omnivoice.generator.key_value_head_count", 8);
    gguf_set_val_u32(g, "synthesize.omnivoice.generator.head_dim", 128);
    gguf_set_val_u32(g, "synthesize.omnivoice.generator.intermediate_size", 3072);
    gguf_set_val_u32(g, "synthesize.omnivoice.generator.text_vocab_size", kTextVocab);
    gguf_set_val_f32(g, "synthesize.omnivoice.generator.rms_norm_eps", 1e-6f);
    gguf_set_val_f32(g, "synthesize.omnivoice.generator.rope_theta", 1000000.0f);
    gguf_set_val_str(g, "synthesize.omnivoice.generator.attention", "bidirectional");

    gguf_set_val_u32(g, "synthesize.omnivoice.audio.num_codebooks", 8);
    gguf_set_val_u32(g, "synthesize.omnivoice.audio.vocab_size", kAudioVocab);
    gguf_set_val_u32(g, "synthesize.omnivoice.audio.mask_id", kMaskId);

    // 8 x 5 x 4 x 2 x 3 = 960 samples a frame, 25 frames a second at 24 kHz.
    gguf_set_val_u32(g, "synthesize.omnivoice.codec.sample_rate", kSampleRate);
    gguf_set_val_u32(g, "synthesize.omnivoice.codec.hop_length", kHopLength);
    gguf_set_val_f32(g, "synthesize.omnivoice.codec.frame_rate_hz", 25.0f);
    set_i32_array(g, "synthesize.omnivoice.codec.upsampling_ratios", { 8, 5, 4, 2, 3 });
    gguf_set_val_u32(g, "synthesize.omnivoice.codec.decoder_hidden_size", 1024);
    gguf_set_val_u32(g, "synthesize.omnivoice.codec.encoder_hidden_size", 64);
    gguf_set_val_u32(g, "synthesize.omnivoice.codec.hidden_size", 256);
    gguf_set_val_u32(g, "synthesize.omnivoice.codec.codebook_dim", 64);
    gguf_set_val_u32(g, "synthesize.omnivoice.codec.codebook_size", kCodebookSize);
    gguf_set_val_u32(g, "synthesize.omnivoice.codec.semantic_sample_rate", 16000);

    gguf_set_val_u32(g, "synthesize.omnivoice.semantic.hidden_size", 768);
    gguf_set_val_u32(g, "synthesize.omnivoice.semantic.layer_count", 12);
    gguf_set_val_u32(g, "synthesize.omnivoice.semantic.attention_head_count", 12);
    gguf_set_val_u32(g, "synthesize.omnivoice.semantic.intermediate_size", 3072);
    set_i32_array(g, "synthesize.omnivoice.semantic.conv_dim", { 512, 512, 512, 512, 512, 512, 512 });
    set_i32_array(g, "synthesize.omnivoice.semantic.conv_kernel", { 10, 3, 3, 3, 3, 2, 2 });
    set_i32_array(g, "synthesize.omnivoice.semantic.conv_stride", { 5, 2, 2, 2, 2, 2, 2 });
    gguf_set_val_f32(g, "synthesize.omnivoice.semantic.layer_norm_eps", 1e-5f);

    gguf_set_val_u32(g, "synthesize.omnivoice.token.denoise", 151669);
    gguf_set_val_u32(g, "synthesize.omnivoice.token.lang_start", 151670);
    gguf_set_val_u32(g, "synthesize.omnivoice.token.lang_end", 151671);
    gguf_set_val_u32(g, "synthesize.omnivoice.token.instruct_start", 151672);
    gguf_set_val_u32(g, "synthesize.omnivoice.token.instruct_end", 151673);
    gguf_set_val_u32(g, "synthesize.omnivoice.token.text_start", 151674);
    gguf_set_val_u32(g, "synthesize.omnivoice.token.text_end", 151675);
    gguf_set_val_u32(g, "synthesize.omnivoice.token.eos", 151645);
    gguf_set_val_u32(g, "synthesize.omnivoice.token.pad", 151643);

    set_string_array(g, "synthesize.omnivoice.languages.tags", { "en", "zh", "ja" });

    gguf_set_val_str(g, "synthesize.profile.schema", "omnivoice-clone-prompt");
    gguf_set_val_u32(g, "synthesize.profile.schema_version", 1);
    gguf_set_val_str(g, "synthesize.profile.compatibility_id", kCompatibilityId);
    gguf_set_val_u32(g, "synthesize.reference.target_sample_rate", kSampleRate);
    gguf_set_val_u32(g, "synthesize.reference.target_channels", 1);
    gguf_set_val_u64(g, "synthesize.reference.min_frames_per_clip", 24000);
    gguf_set_val_u64(g, "synthesize.reference.max_frames_per_clip", 480000);
    gguf_set_val_u64(g, "synthesize.reference.max_total_frames", 480000);
    gguf_set_val_u64(g, "synthesize.reference.max_reference_count", 1);

    // Stand-ins for the 151,676-entry vocabulary and its 151,387 merges. The
    // load path reads those; read_hparams does not.
    set_string_array(g, "synthesize.omnivoice.frontend.vocab", { "!", "\"", "#" });
    set_string_array(g, "synthesize.omnivoice.frontend.merges", { "\304 \240", "\304 \243" });
    gguf_set_val_bool(g, "synthesize.frontend.present", true);
    gguf_set_val_str(g, "synthesize.frontend.provider", "synthesize.qwen_bpe");
    gguf_set_val_u32(g, "synthesize.frontend.contract_version", 1);
    return c;
}

int expect_rejected(const std::function<void(gguf_context *)> & mutate, const char * label) {
    GgufContext context = valid_metadata();
    SYNTH_TEST_CHECK(context != nullptr);
    mutate(context.get());
    synth::omnivoice::HParams hparams;
    const synth_status_t      status = synth::omnivoice::read_hparams(context.get(), hparams);
    if (status == SYNTH_OK) {
        std::fprintf(stderr, "expected rejection: %s\n", label);
        return 1;
    }
    return 0;
}

int run_valid_package() {
    GgufContext context = valid_metadata();
    SYNTH_TEST_CHECK(context != nullptr);
    synth::omnivoice::HParams hparams;
    SYNTH_TEST_CHECK(synth::omnivoice::read_hparams(context.get(), hparams) == SYNTH_OK);

    SYNTH_TEST_CHECK(hparams.model_variant == "omnivoice-0-6b");
    SYNTH_TEST_CHECK(hparams.quantization_profile == synth::omnivoice::QuantizationProfile::F32);
    SYNTH_TEST_CHECK(hparams.quantization_profile_version == 1 && hparams.architecture_version == 1);

    SYNTH_TEST_CHECK(hparams.input_flags == SYNTH_INPUT_SUPPORT_TEXT_UTF8);
    SYNTH_TEST_CHECK(hparams.capability_flags ==
                     (SYNTH_MODEL_CAPABILITY_SPEAKING_RATE | SYNTH_MODEL_CAPABILITY_STOCHASTIC));
    SYNTH_TEST_CHECK(hparams.output_sample_rate == kSampleRate && hparams.output_channel_count == 1);
    SYNTH_TEST_CHECK(hparams.max_input_tokens == 2048 && hparams.max_output_frames == 750);
    SYNTH_TEST_CHECK(hparams.min_speaking_rate == 0.5f && hparams.max_speaking_rate == 2.0f);

    SYNTH_TEST_CHECK(hparams.generator.layer_count == 28 && hparams.generator.hidden_size == 1024);
    SYNTH_TEST_CHECK(hparams.generator.attention_head_count == 16 && hparams.generator.key_value_head_count == 8);
    SYNTH_TEST_CHECK(hparams.generator.head_dim == 128 && hparams.generator.intermediate_size == 3072);
    SYNTH_TEST_CHECK(hparams.generator.text_vocab_size == kTextVocab);
    SYNTH_TEST_CHECK(hparams.generator.rope_theta == 1000000.0f);

    SYNTH_TEST_CHECK(hparams.audio.num_codebooks == 8 && hparams.audio.vocab_size == kAudioVocab);
    SYNTH_TEST_CHECK(hparams.audio.mask_id == kMaskId);

    SYNTH_TEST_CHECK(hparams.codec.sample_rate == kSampleRate && hparams.codec.hop_length == kHopLength);
    SYNTH_TEST_CHECK(hparams.codec.frame_rate_hz == 25.0f);
    SYNTH_TEST_CHECK(hparams.codec.upsampling_ratios == std::vector<uint32_t>({ 8, 5, 4, 2, 3 }));
    SYNTH_TEST_CHECK(hparams.codec.decoder_hidden_size == 1024 && hparams.codec.encoder_hidden_size == 64);
    SYNTH_TEST_CHECK(hparams.codec.hidden_size == 256 && hparams.codec.codebook_dim == 64);
    SYNTH_TEST_CHECK(hparams.codec.codebook_size == kCodebookSize && hparams.codec.semantic_sample_rate == 16000);

    SYNTH_TEST_CHECK(hparams.semantic.hidden_size == 768 && hparams.semantic.layer_count == 12);
    SYNTH_TEST_CHECK(hparams.semantic.attention_head_count == 12 && hparams.semantic.intermediate_size == 3072);
    SYNTH_TEST_CHECK(hparams.semantic.conv_dim.size() == 7 && hparams.semantic.conv_kernel.size() == 7);
    SYNTH_TEST_CHECK(hparams.semantic.conv_stride == std::vector<uint32_t>({ 5, 2, 2, 2, 2, 2, 2 }));

    SYNTH_TEST_CHECK(hparams.tokens.denoise == 151669 && hparams.tokens.text_end == 151675);
    SYNTH_TEST_CHECK(hparams.tokens.eos == 151645 && hparams.tokens.pad == 151643);

    SYNTH_TEST_CHECK(hparams.generation.num_step == 32 && hparams.generation.guidance_scale == 2.0f);
    SYNTH_TEST_CHECK(hparams.generation.t_shift == 0.1f && hparams.generation.layer_penalty_factor == 5.0f);
    SYNTH_TEST_CHECK(hparams.generation.position_temperature == 5.0f && hparams.generation.class_temperature == 0.0f);

    SYNTH_TEST_CHECK(hparams.profile.schema == "omnivoice-clone-prompt" && hparams.profile.schema_version == 1);
    SYNTH_TEST_CHECK(hparams.profile.compatibility_id_hex == kCompatibilityId);
    SYNTH_TEST_CHECK(hparams.profile.reference_sample_rate == kSampleRate && hparams.profile.reference_channels == 1);
    SYNTH_TEST_CHECK(hparams.profile.min_frames_per_clip == 24000 && hparams.profile.max_frames_per_clip == 480000);
    SYNTH_TEST_CHECK(hparams.profile.max_total_frames == 480000 && hparams.profile.max_reference_count == 1);

    SYNTH_TEST_CHECK(hparams.language_tags == std::vector<std::string>({ "en", "zh", "ja" }));

    SYNTH_TEST_CHECK(hparams.frontend_present && hparams.frontend_provider == "synthesize.qwen_bpe");
    SYNTH_TEST_CHECK(hparams.frontend_contract_version == 1);
    return 0;
}

int run_identity_rejections() {
    SYNTH_TEST_CHECK(expect_rejected([](gguf_context * g) { gguf_set_val_str(g, "general.architecture", "qwen3-tts"); },
                                     "another family's package is not this family's") == 0);
    SYNTH_TEST_CHECK(expect_rejected([](gguf_context * g) { gguf_set_val_str(g, "synthesize.model_family", "kokoro"); },
                                     "the family tag must agree with the architecture") == 0);
    SYNTH_TEST_CHECK(expect_rejected([](gguf_context * g) { gguf_set_val_str(g, "synthesize.model_variant", ""); },
                                     "a package names its variant") == 0);
    SYNTH_TEST_CHECK(expect_rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.format_version", 2); },
                                     "an unknown package format") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.omnivoice.architecture_version", 2); },
                        "an unknown architecture version") == 0);

    // F32 is this family's only source profile: the checkpoint stores F32
    // throughout, so a BF16 package would describe weights that never existed.
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_str(g, "synthesize.quantization.profile", "BF16"); },
                        "BF16 is not a profile this family cuts") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.quantization.profile_version", 2); },
                        "an unknown quantization profile version") == 0);

    // Absence is a contract violation rather than a defaulted value, for every
    // key the reader touches; one removal stands for all of them.
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_remove_key(g, "synthesize.omnivoice.codec.hop_length"); },
                        "a missing key is refused rather than defaulted") == 0);
    return 0;
}

int run_capability_rejections() {
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.capabilities.input_flags", 0); },
                        "a text-to-speech package accepts text") == 0);
    // Reviewer FINDING 4: this family declares text input ONLY --
    // src/synthesize.cpp's public dispatch relies on input_kind being
    // "guaranteed SYNTH_INPUT_TEXT_UTF8" for every omnivoice request, which
    // is only true if the loader refuses any OTHER input_flags bit, not
    // merely requires the TEXT bit to be present. Before this fix, a package
    // declaring TEXT_UTF8 | TOKEN_IDS loaded successfully, which would have
    // let a request whose input_kind is SYNTH_INPUT_TOKEN_IDS reach that
    // dispatch and have its int32 token array read as raw UTF-8 text bytes.
    SYNTH_TEST_CHECK(expect_rejected(
                         [](gguf_context * g) {
                             gguf_set_val_u32(g, "synthesize.capabilities.input_flags",
                                              SYNTH_INPUT_SUPPORT_TEXT_UTF8 | SYNTH_INPUT_SUPPORT_TOKEN_IDS);
                         },
                         "this family accepts text input only, no other input_flags bit") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_u64(g, "synthesize.capabilities.max_input_tokens", 0); },
                        "a zero input limit accepts nothing") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_u64(g, "synthesize.capabilities.max_output_frames", 0); },
                        "a zero output limit produces nothing") == 0);

    // The neutral rate has to be inside the range, or the default request would
    // be out of bounds against the package's own declaration.
    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { gguf_set_val_f32(g, "synthesize.capabilities.min_speaking_rate", 1.5f); },
            "a minimum above the neutral rate") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { gguf_set_val_f32(g, "synthesize.capabilities.max_speaking_rate", 0.5f); },
            "a maximum below the neutral rate") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { gguf_set_val_f32(g, "synthesize.capabilities.min_speaking_rate", 0.0f); },
            "the speaking rate divides the duration estimate, so zero is not a rate") == 0);

    // The flags are the promise the public interface reports; a range the
    // capability does not claim would be a control no caller may reach.
    SYNTH_TEST_CHECK(expect_rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.capabilities.flags", 2); },
                                     "an adjustable range without the speaking-rate capability") == 0);
    SYNTH_TEST_CHECK(expect_rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.capabilities.flags", 1); },
                                     "this family samples, so it declares the stochastic capability") == 0);

    SYNTH_TEST_CHECK(expect_rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.audio.channels", 2); },
                                     "the codec produces one channel") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_str(g, "synthesize.audio.sample_format", "s16le"); },
                        "only f32le is produced") == 0);

    // Auto-voice is this family's package default and it has no preset catalog:
    // every alternative here would put the Voice seam in a state no code path
    // serves.
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_str(g, "synthesize.voice.mode", "preset-catalog"); },
                        "this family clones from profile sources") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_bool(g, "synthesize.voice.has_package_default", false); },
                        "auto-voice is the package default") == 0);
    SYNTH_TEST_CHECK(expect_rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.voice.preset_count", 1); },
                                     "there is no preset catalog to count") == 0);
    return 0;
}

int run_generation_rejections() {
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.omnivoice.generation.num_step", 0); },
                        "zero mask-predict steps decode nothing") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { gguf_set_val_f32(g, "synthesize.omnivoice.generation.guidance_scale", -1.0f); },
            "a negative guidance scale inverts the conditional") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_f32(g, "synthesize.omnivoice.generation.t_shift", 0.0f); },
                        "t_shift divides the schedule") == 0);
    SYNTH_TEST_CHECK(expect_rejected(
                         [](gguf_context * g) {
                             gguf_set_val_f32(g, "synthesize.omnivoice.generation.position_temperature", -1.0f);
                         },
                         "a negative temperature scales the Gumbel noise backwards") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { gguf_set_val_f32(g, "synthesize.omnivoice.generation.class_temperature", -1.0f); },
            "a negative class temperature scales the Gumbel noise backwards") == 0);
    return 0;
}

int run_generator_rejections() {
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.omnivoice.generator.layer_count", 0); },
                        "a stack of no layers") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.omnivoice.generator.head_dim", 0); },
                        "a zero head width") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { gguf_set_val_u32(g, "synthesize.omnivoice.generator.key_value_head_count", 5); },
            "query heads must divide into key/value heads") == 0);
    // Guarded before the divisibility test rather than by it: a zero divisor
    // would raise SIGFPE during load instead of being refused.
    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { gguf_set_val_u32(g, "synthesize.omnivoice.generator.key_value_head_count", 0); },
            "a zero key/value head count is refused rather than divided by") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { gguf_set_val_u32(g, "synthesize.omnivoice.generator.text_vocab_size", 0); },
            "an empty text vocabulary") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { gguf_set_val_f32(g, "synthesize.omnivoice.generator.rms_norm_eps", 0.0f); },
            "a zero rms_norm_eps divides by the norm alone") == 0);

    // The whole canvas is attended in both directions at every step. A causal
    // package would be silently building a different model.
    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { gguf_set_val_str(g, "synthesize.omnivoice.generator.attention", "causal"); },
            "this family's generator is bidirectional") == 0);

    // Products of these fields feed int64 shape arithmetic in the catalog; a
    // package this large is not a model, it is an overflow attempt.
    SYNTH_TEST_CHECK(expect_rejected(
                         [](gguf_context * g) {
                             gguf_set_val_u32(g, "synthesize.omnivoice.generator.attention_head_count", 65536);
                             gguf_set_val_u32(g, "synthesize.omnivoice.generator.key_value_head_count", 65536);
                             gguf_set_val_u32(g, "synthesize.omnivoice.generator.head_dim", 65536);
                         },
                         "attention geometry whose products leave shape arithmetic") == 0);

    // The five guarded products are now checked one at a time so the
    // diagnostic names the actual offender; a combined condition could only
    // ever report attention_inner regardless of which field broke the
    // ceiling (T5 review). Each case below pushes exactly one field over it
    // while the other four stay small, which the mutation above -- all three
    // attention fields huge at once -- cannot distinguish.
    //
    // key_value_head_count has no isolated case: the GQA divisibility rule
    // above requires key_value_head_count <= attention_head_count whenever
    // attention_head_count > 0, and both multiply the SAME head_dim, so
    // kv_inner can never exceed the ceiling while attention_inner does not.
    //
    // Each value sits one past kMaxDimensionProduct in weights.cpp (1 << 24),
    // so what these cases pin is the boundary itself: if the ceiling ever
    // moves, every case here goes stale together and loudly, rather than one
    // at a time as values chosen above the old ceiling happen to straddle the
    // new one.
    constexpr uint32_t kOverCeiling = (1u << 24) + 1;
    SYNTH_TEST_CHECK(expect_rejected(
                         [](gguf_context * g) {
                             // (2^22 + 1) * 4 = 2^24 + 4; odd, so only kv 1
                             // divides it, and kv_inner stays at 4.
                             gguf_set_val_u32(g, "synthesize.omnivoice.generator.attention_head_count", (1u << 22) + 1);
                             gguf_set_val_u32(g, "synthesize.omnivoice.generator.key_value_head_count", 1);
                             gguf_set_val_u32(g, "synthesize.omnivoice.generator.head_dim", 4);
                         },
                         "attention_head_count * head_dim alone over the ceiling") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { gguf_set_val_u32(g, "synthesize.omnivoice.generator.hidden_size", kOverCeiling); },
            "hidden_size alone over the ceiling") == 0);
    SYNTH_TEST_CHECK(expect_rejected(
                         [](gguf_context * g) {
                             gguf_set_val_u32(g, "synthesize.omnivoice.generator.intermediate_size", kOverCeiling);
                         },
                         "intermediate_size alone over the ceiling") == 0);
    SYNTH_TEST_CHECK(expect_rejected(
                         [](gguf_context * g) {
                             gguf_set_val_u32(g, "synthesize.omnivoice.generator.text_vocab_size", kOverCeiling);
                         },
                         "text_vocab_size alone over the ceiling") == 0);
    return 0;
}

int run_canvas_and_codec_rejections() {
    // The mask occupies the last entry of the canvas vocabulary. Anywhere else
    // and masking a position would write a real code.
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.omnivoice.audio.mask_id", 1023); },
                        "the mask id is the last canvas entry") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.omnivoice.audio.num_codebooks", 0); },
                        "a canvas of no codebooks") == 0);
    // Products of these fields feed int64 shape arithmetic in the catalog; a
    // canvas this large is not a model, it is an overflow attempt. Keeps
    // mask_id == vocab_size - 1 and vocab_size == codebook_size + 1 satisfied
    // so only the new size rule can fire.
    SYNTH_TEST_CHECK(expect_rejected(
                         [](gguf_context * g) {
                             gguf_set_val_u32(g, "synthesize.omnivoice.audio.num_codebooks", 1u << 16);
                             gguf_set_val_u32(g, "synthesize.omnivoice.audio.vocab_size", (1u << 16) + 1);
                             gguf_set_val_u32(g, "synthesize.omnivoice.audio.mask_id", 1u << 16);
                             gguf_set_val_u32(g, "synthesize.omnivoice.codec.codebook_size", 1u << 16);
                         },
                         "a canvas whose embedding table exceeds any real package") == 0);

    SYNTH_TEST_CHECK(expect_rejected(
                         [](gguf_context * g) {
                             set_i32_array(g, "synthesize.omnivoice.codec.upsampling_ratios", { 8, 5, 4, 2, 2 });
                         },
                         "the upsampling ratios must multiply to exactly one frame of samples") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { set_i32_array(g, "synthesize.omnivoice.codec.upsampling_ratios", {}); },
                        "an empty ratio stack upsamples nothing") == 0);
    // Before the early-exceed check this product could wrap uint64_t; whether
    // it then collided with the hop was luck, not a rule.
    SYNTH_TEST_CHECK(expect_rejected(
                         [](gguf_context * g) {
                             set_i32_array(g, "synthesize.omnivoice.codec.upsampling_ratios",
                                           { 2147483647, 2147483647, 2147483647 });
                         },
                         "upsampling ratios that would wrap the hop product") == 0);
    // One rule per mutation: each of these breaks exactly one reader check, so
    // a reordering of read_codec cannot silently change which rule a test
    // exercises.
    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { gguf_set_val_f32(g, "synthesize.omnivoice.codec.frame_rate_hz", 30.0f); },
            "hop and frame rate must agree with the sample rate") == 0);
    // Fires the codec-vs-declared-output rule; the frame relation is checked
    // later and never reached.
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.omnivoice.codec.sample_rate", 16000); },
                        "codec rate must match the declared output rate") == 0);
    // 16000 is the semantic branch's rate, not a second output rate; a package
    // that swapped the two would resample every clone reference wrongly.
    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { gguf_set_val_u32(g, "synthesize.omnivoice.codec.semantic_sample_rate", 0); },
            "the semantic branch runs at a rate") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { gguf_set_val_u32(g, "synthesize.omnivoice.codec.decoder_hidden_size", 0); },
            "a zero decoder width") == 0);
    // The canvas holds one entry per codec code plus the mask. A disagreement
    // would have the generator predicting codes the codec cannot look up.
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.omnivoice.codec.codebook_size", 512); },
                        "the canvas vocabulary is the codebook plus the mask") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.omnivoice.codec.codebook_dim", 0); },
                        "a zero codebook width") == 0);
    return 0;
}

int run_semantic_rejections() {
    SYNTH_TEST_CHECK(expect_rejected(
                         [](gguf_context * g) {
                             set_i32_array(g, "synthesize.omnivoice.semantic.conv_stride", { 5, 2, 2, 2, 2, 2 });
                         },
                         "the three conv arrays describe the same seven layers") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { set_i32_array(g, "synthesize.omnivoice.semantic.conv_dim", {}); },
                        "a feature extractor of no layers") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.omnivoice.semantic.hidden_size", 0); },
                        "a zero semantic width") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.omnivoice.semantic.hidden_size", 770); },
                        "the semantic width must split into whole heads") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { gguf_set_val_f32(g, "synthesize.omnivoice.semantic.layer_norm_eps", 0.0f); },
            "a zero layer_norm_eps divides by the norm alone") == 0);
    return 0;
}

int run_token_rejections() {
    // Every marker indexes the text embedding table. One past the end is an
    // out-of-range row at synthesis time, in a place that says nothing about
    // the cause.
    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { gguf_set_val_u32(g, "synthesize.omnivoice.token.text_end", kTextVocab); },
            "a marker id past the end of the text vocabulary") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.omnivoice.token.pad", 999999); },
                        "a pad id past the end of the text vocabulary") == 0);
    // The prompt is assembled from these by value; two markers sharing an id
    // would make the two slots indistinguishable in the built prompt.
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.omnivoice.token.lang_end", 151670); },
                        "two markers may not share an id") == 0);
    return 0;
}

int run_language_and_profile_rejections() {
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { set_string_array(g, "synthesize.omnivoice.languages.tags", {}); },
                        "a package serves at least one language") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { set_string_array(g, "synthesize.omnivoice.languages.tags", { "en", "", "ja" }); },
            "an empty tag names no language") == 0);
    // The tag is also the literal text the prompt's language slot carries, so a
    // duplicate would publish the same capability twice.
    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { set_string_array(g, "synthesize.omnivoice.languages.tags", { "en", "zh", "en" }); },
            "a duplicated language tag") == 0);

    SYNTH_TEST_CHECK(expect_rejected([](gguf_context * g) { gguf_set_val_str(g, "synthesize.profile.schema", ""); },
                                     "a profile contract names its schema") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { gguf_set_val_str(g, "synthesize.profile.schema", "kokoro-style-vector"); },
            "a Serialized Profile schema this family cannot parse") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.profile.schema_version", 2); },
                        "an unknown profile schema version") == 0);
    // The Profile Compatibility ID is a sha256 over the family compatibility
    // manifest. A short or non-hex value could never match one.
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_str(g, "synthesize.profile.compatibility_id", "abc123"); },
                        "a compatibility id that is not 32 bytes of hex") == 0);
    SYNTH_TEST_CHECK(expect_rejected(
                         [](gguf_context * g) {
                             gguf_set_val_str(g, "synthesize.profile.compatibility_id",
                                              "zzzz1b0a7d4e6538aa10cc93bf7712d4e08a5c63b91d47fe2085730c6ad1e4b8");
                         },
                         "a compatibility id with non-hex characters") == 0);
    // The converter always emits a lowercase hexdigest; refuse-don't-repair
    // means an otherwise-valid id with one uppercased hex letter is rejected
    // rather than normalized.
    SYNTH_TEST_CHECK(expect_rejected(
                         [](gguf_context * g) {
                             std::string mutated = kCompatibilityId;
                             for (char & character : mutated) {
                                 if (character >= 'a' && character <= 'f') {
                                     character = static_cast<char>(character - 'a' + 'A');
                                     break;
                                 }
                             }
                             gguf_set_val_str(g, "synthesize.profile.compatibility_id", mutated.c_str());
                         },
                         "compatibility id must be lowercase hex") == 0);

    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.reference.target_sample_rate", 0); },
                        "reference audio is resampled to a rate") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.reference.target_channels", 2); },
                        "the encode path takes mono") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_u64(g, "synthesize.reference.min_frames_per_clip", 0); },
                        "a clip of no frames carries no voice") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { gguf_set_val_u64(g, "synthesize.reference.min_frames_per_clip", 960000); },
            "the minimum clip length must fit inside the maximum") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_u64(g, "synthesize.reference.max_total_frames", 0); },
                        "a total budget of no frames admits no clip") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_u64(g, "synthesize.reference.max_reference_count", 0); },
                        "a clone package accepts at least one reference") == 0);
    return 0;
}

int run_frontend_rejections() {
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_bool(g, "synthesize.frontend.present", false); },
                        "a raw-text family needs a frontend") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected(
            [](gguf_context * g) { gguf_set_val_str(g, "synthesize.frontend.provider", "synthesize.espeak"); },
            "this family tokenizes with the byte-level BPE frontend") == 0);
    SYNTH_TEST_CHECK(
        expect_rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.frontend.contract_version", 2); },
                        "an unknown frontend contract version") == 0);
    return 0;
}

}  // namespace

int main() {
    synth::omnivoice::HParams hparams;
    SYNTH_TEST_CHECK(synth::omnivoice::read_hparams(nullptr, hparams) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(run_valid_package() == 0);
    SYNTH_TEST_CHECK(run_identity_rejections() == 0);
    SYNTH_TEST_CHECK(run_capability_rejections() == 0);
    SYNTH_TEST_CHECK(run_generation_rejections() == 0);
    SYNTH_TEST_CHECK(run_generator_rejections() == 0);
    SYNTH_TEST_CHECK(run_canvas_and_codec_rejections() == 0);
    SYNTH_TEST_CHECK(run_semantic_rejections() == 0);
    SYNTH_TEST_CHECK(run_token_rejections() == 0);
    SYNTH_TEST_CHECK(run_language_and_profile_rejections() == 0);
    SYNTH_TEST_CHECK(run_frontend_rejections() == 0);
    return 0;
}
