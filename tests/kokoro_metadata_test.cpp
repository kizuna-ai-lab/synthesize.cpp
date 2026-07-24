#include "arch/kokoro/weights.h"
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

constexpr uint32_t kNToken     = 8;
constexpr uint32_t kStyleDim   = 4;
constexpr uint32_t kVoiceRows  = 16;
constexpr uint32_t kMaxTokens  = 12;
constexpr uint32_t kHopSize    = 5;
constexpr uint32_t kSampleRate = 24000;

// A small but structurally faithful package: sparse symbol table with a
// reserved pad row, one shared ALBERT group, and a frame chain of
// 2 * 10 * 6 * 5 = 600 samples per duration step.
GgufContext valid_metadata() {
    GgufContext    c(gguf_init_empty());
    gguf_context * g = c.get();

    gguf_set_val_str(g, "general.architecture", "kokoro");
    gguf_set_val_str(g, "synthesize.model_family", "kokoro");
    gguf_set_val_str(g, "synthesize.model_variant", "kokoro-v1-0");
    gguf_set_val_u32(g, "synthesize.format_version", 1);
    gguf_set_val_u32(g, "synthesize.kokoro.architecture_version", 1);
    gguf_set_val_str(g, "synthesize.quantization.profile", "F32");
    gguf_set_val_u32(g, "synthesize.quantization.profile_version", 1);

    gguf_set_val_u32(g, "synthesize.capabilities.input_flags",
                     SYNTH_INPUT_SUPPORT_PHONEMES_UTF8 | SYNTH_INPUT_SUPPORT_TOKEN_IDS);
    gguf_set_val_u32(g, "synthesize.capabilities.flags",
                     SYNTH_MODEL_CAPABILITY_SPEAKING_RATE | SYNTH_MODEL_CAPABILITY_STOCHASTIC);
    gguf_set_val_u64(g, "synthesize.capabilities.max_input_tokens", kMaxTokens);
    gguf_set_val_u64(g, "synthesize.capabilities.max_output_frames", 1440000);
    gguf_set_val_f32(g, "synthesize.capabilities.min_speaking_rate", 0.8f);
    gguf_set_val_f32(g, "synthesize.capabilities.max_speaking_rate", 1.25f);
    gguf_set_val_u32(g, "synthesize.audio.sample_rate_hz", kSampleRate);
    gguf_set_val_u32(g, "synthesize.audio.channels", 1);
    gguf_set_val_str(g, "synthesize.audio.sample_format", "f32le");

    gguf_set_val_u32(g, "synthesize.kokoro.n_token", kNToken);
    gguf_set_val_u32(g, "synthesize.kokoro.hidden_dim", 512);
    gguf_set_val_u32(g, "synthesize.kokoro.style_dim", kStyleDim);
    gguf_set_val_u32(g, "synthesize.kokoro.n_layer", 3);
    gguf_set_val_u32(g, "synthesize.kokoro.n_mels", 80);
    gguf_set_val_u32(g, "synthesize.kokoro.max_dur", 50);
    gguf_set_val_u32(g, "synthesize.kokoro.dim_in", 64);
    gguf_set_val_u32(g, "synthesize.kokoro.text_encoder_kernel_size", 5);
    gguf_set_val_u32(g, "synthesize.kokoro.samples_per_frame", 600);

    gguf_set_val_u32(g, "synthesize.kokoro.plbert.hidden_size", 768);
    gguf_set_val_u32(g, "synthesize.kokoro.plbert.num_attention_heads", 12);
    gguf_set_val_u32(g, "synthesize.kokoro.plbert.intermediate_size", 2048);
    gguf_set_val_u32(g, "synthesize.kokoro.plbert.num_hidden_layers", 12);
    gguf_set_val_u32(g, "synthesize.kokoro.plbert.max_position_embeddings", 512);
    gguf_set_val_u32(g, "synthesize.kokoro.plbert.shared_layer_groups", 1);
    gguf_set_val_f32(g, "synthesize.kokoro.plbert.layer_norm_eps", 1e-12f);

    const int32_t rates[]   = { 10, 6 };
    const int32_t kernels[] = { 20, 12 };
    const int32_t rb[]      = { 3, 7, 11 };
    const int32_t dil[]     = { 1, 3, 5 };
    gguf_set_arr_data(g, "synthesize.kokoro.istftnet.upsample_rates", GGUF_TYPE_INT32, rates, 2);
    gguf_set_arr_data(g, "synthesize.kokoro.istftnet.upsample_kernel_sizes", GGUF_TYPE_INT32, kernels, 2);
    gguf_set_arr_data(g, "synthesize.kokoro.istftnet.resblock_kernel_sizes", GGUF_TYPE_INT32, rb, 3);
    for (int branch = 0; branch < 3; ++branch) {
        const std::string key = "synthesize.kokoro.istftnet.resblock_dilations." + std::to_string(branch);
        gguf_set_arr_data(g, key.c_str(), GGUF_TYPE_INT32, dil, 3);
    }
    gguf_set_val_u32(g, "synthesize.kokoro.istftnet.upsample_initial_channel", 512);
    gguf_set_val_u32(g, "synthesize.kokoro.istftnet.gen_istft_n_fft", 20);
    gguf_set_val_u32(g, "synthesize.kokoro.istftnet.gen_istft_hop_size", kHopSize);
    gguf_set_val_str(g, "synthesize.kokoro.istftnet.window", "hann_periodic");
    gguf_set_val_bool(g, "synthesize.kokoro.istftnet.center", true);

    gguf_set_val_u32(g, "synthesize.kokoro.source.sampling_rate", kSampleRate);
    gguf_set_val_u32(g, "synthesize.kokoro.source.harmonic_num", 8);
    gguf_set_val_u32(g, "synthesize.kokoro.source.upsample_scale", 300);
    gguf_set_val_f32(g, "synthesize.kokoro.source.sine_amp", 0.1f);
    gguf_set_val_f32(g, "synthesize.kokoro.source.noise_std", 0.003f);
    gguf_set_val_f32(g, "synthesize.kokoro.source.voiced_threshold", 10.0f);

    gguf_set_val_u32(g, "synthesize.kokoro.voice.rows", kVoiceRows);
    gguf_set_val_u32(g, "synthesize.kokoro.voice.dim", 2 * kStyleDim);
    gguf_set_val_str(g, "synthesize.kokoro.voice.row_rule", "final_token_count_minus_three");
    gguf_set_val_u32(g, "synthesize.kokoro.voice.decoder_offset", 0);
    gguf_set_val_u32(g, "synthesize.kokoro.voice.prosody_offset", kStyleDim);
    gguf_set_val_bool(g, "synthesize.kokoro.adain.instance_norm_affine", false);
    gguf_set_val_f32(g, "synthesize.kokoro.adain.eps", 1e-5f);

    gguf_set_val_bool(g, "synthesize.frontend.present", true);
    gguf_set_val_str(g, "synthesize.frontend.provider", "synthesize.symbol_map");
    gguf_set_val_u32(g, "synthesize.frontend.contract_version", 1);
    gguf_set_val_str(g, "synthesize.frontend.phoneme_mapping", "unicode_scalar");
    gguf_set_val_str(g, "synthesize.frontend.padding_rule", "wrap_pad_token");
    // Ids 0, 2 and 5 stay reserved, mirroring the real sparse vocabulary.
    const char * symbols[kNToken] = { "", "a", "", "t", "s", "", "i", "o" };
    gguf_set_arr_str(g, "synthesize.kokoro.symbols", symbols, kNToken);
    gguf_set_val_str(g, "synthesize.kokoro.symbols.lookup", "unique_scalar");
    gguf_set_val_u32(g, "synthesize.kokoro.symbols.mapped_count", 5);
    gguf_set_val_u32(g, "synthesize.kokoro.symbols.pad_id", 0);

    gguf_set_val_str(g, "synthesize.voice.mode", "preset-catalog");
    gguf_set_val_bool(g, "synthesize.voice.has_package_default", false);
    gguf_set_val_u32(g, "synthesize.voice.preset_count", 2);
    gguf_set_val_str(g, "synthesize.voice.0.id", "af_heart");
    gguf_set_val_u32(g, "synthesize.voice.0.flags", 0);
    gguf_set_val_str(g, "synthesize.voice.1.id", "bm_george");
    gguf_set_val_u32(g, "synthesize.voice.1.flags", 0);
    return c;
}

bool rejected(const std::function<void(gguf_context *)> & mutate) {
    GgufContext            context = valid_metadata();
    synth::kokoro::HParams hparams;
    mutate(context.get());
    return synth::kokoro::read_hparams(context.get(), hparams) != SYNTH_OK;
}

}  // namespace

int main() {
    synth::kokoro::HParams hparams;
    GgufContext            context = valid_metadata();
    SYNTH_TEST_CHECK(synth::kokoro::read_hparams(context.get(), hparams) == SYNTH_OK);

    SYNTH_TEST_CHECK(hparams.model_variant == "kokoro-v1-0");
    SYNTH_TEST_CHECK(hparams.quantization_profile == synth::kokoro::QuantizationProfile::F32);
    SYNTH_TEST_CHECK(hparams.output_sample_rate == kSampleRate && hparams.output_channel_count == 1);
    SYNTH_TEST_CHECK(hparams.n_token == kNToken && hparams.style_dim == kStyleDim);
    SYNTH_TEST_CHECK(hparams.samples_per_frame == 600);
    SYNTH_TEST_CHECK(hparams.plbert.shared_layer_groups == 1 && hparams.plbert.num_hidden_layers == 12);
    SYNTH_TEST_CHECK(hparams.istftnet.upsample_rates.size() == 2 && hparams.istftnet.gen_istft_hop_size == kHopSize);
    SYNTH_TEST_CHECK(hparams.istftnet.resblock_dilations.size() == 3);
    SYNTH_TEST_CHECK(hparams.source.harmonic_num == 8 && hparams.source.upsample_scale == 300);
    SYNTH_TEST_CHECK(hparams.voice_rows == kVoiceRows && hparams.voice_dim == 2 * kStyleDim);
    SYNTH_TEST_CHECK(hparams.voice_prosody_offset == kStyleDim && !hparams.adain_instance_norm_affine);
    SYNTH_TEST_CHECK(!hparams.has_package_default);
    SYNTH_TEST_CHECK(hparams.preset_voice_ids.size() == 2 && hparams.preset_voice_ids[1] == "bm_george");
    SYNTH_TEST_CHECK(hparams.frontend_present);
    SYNTH_TEST_CHECK(hparams.frontend_config.padding_rule == synth::SymbolPaddingRule::WrapPadToken);
    SYNTH_TEST_CHECK(hparams.frontend_config.symbols.size() == kNToken);
    SYNTH_TEST_CHECK(hparams.frontend_config.blank_id == 0);

    // The style row is the unpadded token count minus one, so the shortest
    // usable request lands on row zero and the table bounds the longest.
    uint32_t row = 99;
    SYNTH_TEST_CHECK(!synth::kokoro::resolve_style_row(hparams, 0, row));
    SYNTH_TEST_CHECK(!synth::kokoro::resolve_style_row(hparams, 3, row));
    SYNTH_TEST_CHECK(synth::kokoro::resolve_style_row(hparams, 4, row) && row == 1);
    SYNTH_TEST_CHECK(synth::kokoro::resolve_style_row(hparams, 12, row) && row == 9);
    SYNTH_TEST_CHECK(synth::kokoro::resolve_style_row(hparams, kVoiceRows + 2, row) && row == kVoiceRows - 1);
    SYNTH_TEST_CHECK(!synth::kokoro::resolve_style_row(hparams, kVoiceRows + 3, row));

    SYNTH_TEST_CHECK(synth::kokoro::read_hparams(nullptr, hparams) == SYNTH_ERR_INVALID_ARG);

    // Identity drift is an architecture error rather than a package error.
    {
        GgufContext            other = valid_metadata();
        synth::kokoro::HParams parsed;
        gguf_set_val_str(other.get(), "general.architecture", "vits");
        SYNTH_TEST_CHECK(synth::kokoro::read_hparams(other.get(), parsed) == SYNTH_ERR_UNSUPPORTED_ARCH);
    }

    SYNTH_TEST_CHECK(rejected([](gguf_context * g) { gguf_set_val_str(g, "synthesize.model_family", "vits"); }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.format_version", 2); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.kokoro.architecture_version", 2); }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * g) { gguf_set_val_str(g, "synthesize.model_variant", ""); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * g) { gguf_set_val_str(g, "synthesize.quantization.profile", "Q2_K"); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.quantization.profile_version", 2); }));

    // Capabilities: text input is not claimed, and both capability bits are
    // mandatory because every synthesis is stochastic and rate-aware.
    SYNTH_TEST_CHECK(rejected([](gguf_context * g) {
        gguf_set_val_u32(g, "synthesize.capabilities.input_flags",
                         SYNTH_INPUT_SUPPORT_TEXT_UTF8 | SYNTH_INPUT_SUPPORT_TOKEN_IDS);
    }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * g) {
        gguf_set_val_u32(g, "synthesize.capabilities.flags", SYNTH_MODEL_CAPABILITY_SPEAKING_RATE);
    }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.audio.channels", 2); }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.audio.sample_rate_hz", 0); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * g) { gguf_set_val_str(g, "synthesize.audio.sample_format", "s16le"); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * g) { gguf_set_val_u64(g, "synthesize.capabilities.max_input_tokens", 2); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * g) { gguf_set_val_u64(g, "synthesize.capabilities.max_output_frames", 0); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * g) { gguf_set_val_f32(g, "synthesize.capabilities.min_speaking_rate", 1.5f); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * g) { gguf_set_val_f32(g, "synthesize.capabilities.max_speaking_rate", 0.5f); }));

    // Dimensions.
    SYNTH_TEST_CHECK(rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.kokoro.n_token", 0); }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.kokoro.hidden_dim", 511); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.kokoro.text_encoder_kernel_size", 4); }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.kokoro.max_dur", 0); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.kokoro.samples_per_frame", 601); }));

    // PL-BERT.
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.kokoro.plbert.shared_layer_groups", 12); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.kokoro.plbert.num_attention_heads", 7); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * g) { gguf_set_val_f32(g, "synthesize.kokoro.plbert.layer_norm_eps", 0.0f); }));

    // iSTFTNet: framing, parity, and stage shapes.
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.kokoro.istftnet.gen_istft_n_fft", 21); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.kokoro.istftnet.gen_istft_hop_size", 0); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.kokoro.istftnet.gen_istft_hop_size", 7); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * g) { gguf_set_val_bool(g, "synthesize.kokoro.istftnet.center", false); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * g) { gguf_set_val_str(g, "synthesize.kokoro.istftnet.window", "hamming"); }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * g) {
        const int32_t mismatched[] = { 10 };
        gguf_set_arr_data(g, "synthesize.kokoro.istftnet.upsample_rates", GGUF_TYPE_INT32, mismatched, 1);
    }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * g) {
        const int32_t even[] = { 4, 7, 11 };
        gguf_set_arr_data(g, "synthesize.kokoro.istftnet.resblock_kernel_sizes", GGUF_TYPE_INT32, even, 3);
    }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * g) {
        const int32_t negative[] = { 1, -3, 5 };
        gguf_set_arr_data(g, "synthesize.kokoro.istftnet.resblock_dilations.0", GGUF_TYPE_INT32, negative, 3);
    }));

    // Source module.
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.kokoro.source.upsample_scale", 150); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.kokoro.source.sampling_rate", 22050); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * g) { gguf_set_val_f32(g, "synthesize.kokoro.source.sine_amp", 0.0f); }));

    // Voice table and AdaIN.
    SYNTH_TEST_CHECK(rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.kokoro.voice.dim", kStyleDim); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.kokoro.voice.prosody_offset", 0); }));
    SYNTH_TEST_CHECK(rejected(
        [](gguf_context * g) { gguf_set_val_str(g, "synthesize.kokoro.voice.row_rule", "final_token_count"); }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.kokoro.voice.rows", 4); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * g) { gguf_set_val_bool(g, "synthesize.kokoro.adain.instance_norm_affine", true); }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * g) { gguf_set_val_f32(g, "synthesize.kokoro.adain.eps", -1.0f); }));

    // Frontend contract.
    SYNTH_TEST_CHECK(rejected([](gguf_context * g) { gguf_set_val_bool(g, "synthesize.frontend.present", false); }));
    SYNTH_TEST_CHECK(rejected(
        [](gguf_context * g) { gguf_set_val_str(g, "synthesize.frontend.padding_rule", "interleaved_blank"); }));
    SYNTH_TEST_CHECK(rejected(
        [](gguf_context * g) { gguf_set_val_str(g, "synthesize.frontend.phoneme_mapping", "delimited_symbol"); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * g) { gguf_set_val_str(g, "synthesize.kokoro.symbols.lookup", "last_index_wins"); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.kokoro.symbols.mapped_count", 4); }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * g) {
        // The pad id must name a reserved row, never a mapped symbol.
        gguf_set_val_u32(g, "synthesize.kokoro.symbols.pad_id", 1);
    }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * g) {
        const char * shorter[] = { "", "a", "" };
        gguf_set_arr_str(g, "synthesize.kokoro.symbols", shorter, 3);
    }));

    // Voice catalog.
    SYNTH_TEST_CHECK(rejected([](gguf_context * g) { gguf_set_val_str(g, "synthesize.voice.mode", "fixed-default"); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * g) { gguf_set_val_bool(g, "synthesize.voice.has_package_default", true); }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.voice.preset_count", 0); }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * g) { gguf_set_val_str(g, "synthesize.voice.1.id", "af_heart"); }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * g) { gguf_set_val_str(g, "synthesize.voice.0.id", ""); }));
    SYNTH_TEST_CHECK(rejected(
        [](gguf_context * g) { gguf_set_val_u32(g, "synthesize.voice.0.flags", SYNTH_PRESET_VOICE_DEFAULT); }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * g) { gguf_set_val_u32(g, "synthesize.voice.preset_count", 3); }));

    return 0;
}
