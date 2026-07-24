#include "arch/vits/weights.h"
#include "gguf.h"
#include "test-assert.h"
#include "vits-test-fixture.h"

#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <string>

namespace {

struct GgufDeleter {
    void operator()(gguf_context * context) const {
        if (context != nullptr) {
            gguf_free(context);
        }
    }
};

using GgufContext = std::unique_ptr<gguf_context, GgufDeleter>;

GgufContext valid_metadata() {
    const synth::vits::HParams hparams = synth::test::small_vits_hparams();
    GgufContext                context(gguf_init_empty());
    gguf_set_val_str(context.get(), "general.architecture", "vits");
    gguf_set_val_str(context.get(), "synthesize.model_family", "vits");
    gguf_set_val_str(context.get(), "synthesize.model_variant", "vits-ljspeech");
    gguf_set_val_u32(context.get(), "general.file_type", 0);
    gguf_set_val_str(context.get(), "synthesize.quantization.profile", "F32");
    gguf_set_val_u32(context.get(), "synthesize.quantization.profile_version", 1);
    gguf_set_val_u32(context.get(), "synthesize.format_version", 1);
    gguf_set_val_u32(context.get(), "synthesize.vits.architecture_version", 1);
    gguf_set_val_u32(context.get(), "synthesize.capabilities.input_flags",
                     SYNTH_INPUT_SUPPORT_PHONEMES_UTF8 | SYNTH_INPUT_SUPPORT_TOKEN_IDS);
    gguf_set_val_u32(context.get(), "synthesize.capabilities.flags", hparams.capability_flags);
    gguf_set_val_u32(context.get(), "synthesize.audio.sample_rate_hz", hparams.output_sample_rate);
    gguf_set_val_u32(context.get(), "synthesize.audio.channels", hparams.output_channel_count);
    gguf_set_val_str(context.get(), "synthesize.audio.sample_format", "f32le");
    gguf_set_val_u32(context.get(), "synthesize.vits.vocab_size", hparams.vocab_size);
    gguf_set_val_bool(context.get(), "synthesize.frontend.present", true);
    gguf_set_val_str(context.get(), "synthesize.frontend.provider", "synthesize.symbol_map");
    gguf_set_val_u32(context.get(), "synthesize.frontend.contract_version", 1);
    gguf_set_val_str(context.get(), "synthesize.frontend.phoneme_mapping", "unicode_scalar");
    const char * symbols[] = { "_", "a", "b", "c", "d" };
    gguf_set_arr_str(context.get(), "synthesize.vits.symbols", symbols, 5);
    gguf_set_val_str(context.get(), "synthesize.vits.symbols.lookup_policy", "last_index_wins");
    gguf_set_val_u32(context.get(), "synthesize.vits.symbols.blank_id", 0);
    gguf_set_val_bool(context.get(), "synthesize.vits.symbols.upstream_add_blank", true);
    gguf_set_val_u32(context.get(), "synthesize.vits.inter_channels", hparams.inter_channels);
    gguf_set_val_u32(context.get(), "synthesize.vits.hidden_channels", hparams.hidden_channels);
    gguf_set_val_u32(context.get(), "synthesize.vits.filter_channels", hparams.filter_channels);
    gguf_set_val_u32(context.get(), "synthesize.vits.text.layer_count", hparams.text_layer_count);
    gguf_set_val_u32(context.get(), "synthesize.vits.text.head_count", hparams.text_head_count);
    gguf_set_val_u32(context.get(), "synthesize.vits.text.ffn_kernel_size", hparams.text_ffn_kernel_size);
    gguf_set_val_u32(context.get(), "synthesize.vits.text.attention_window", hparams.text_attention_window);
    gguf_set_val_str(context.get(), "synthesize.vits.duration.predictor_type", "stochastic");
    gguf_set_val_u32(context.get(), "synthesize.vits.duration.dds_layer_count", hparams.duration_dds_layer_count);
    gguf_set_val_u32(context.get(), "synthesize.vits.duration.flow_count", hparams.duration_flow_count);
    gguf_set_val_u32(context.get(), "synthesize.vits.duration.spline_bin_count", hparams.duration_spline_bin_count);
    gguf_set_val_u32(context.get(), "synthesize.vits.flow.block_count", hparams.flow_block_count);
    gguf_set_val_u32(context.get(), "synthesize.vits.flow.kernel_size", hparams.flow_kernel_size);
    gguf_set_val_u32(context.get(), "synthesize.vits.flow.dilation_rate", hparams.flow_dilation_rate);
    gguf_set_val_u32(context.get(), "synthesize.vits.flow.wn_layer_count", hparams.flow_wn_layer_count);
    gguf_set_val_bool(context.get(), "synthesize.vits.flow.mean_only", hparams.flow_mean_only);
    gguf_set_val_str(context.get(), "synthesize.vits.decoder.resblock_type", "1");
    const std::vector<int32_t> resblock_kernels(hparams.decoder_resblock_kernel_sizes.begin(),
                                                hparams.decoder_resblock_kernel_sizes.end());
    gguf_set_arr_data(context.get(), "synthesize.vits.decoder.resblock_kernel_sizes", GGUF_TYPE_INT32,
                      resblock_kernels.data(), resblock_kernels.size());
    for (size_t branch = 0; branch < hparams.decoder_resblock_dilations.size(); ++branch) {
        const std::string          key = "synthesize.vits.decoder.resblock_dilations." + std::to_string(branch);
        const std::vector<int32_t> dilations(hparams.decoder_resblock_dilations[branch].begin(),
                                             hparams.decoder_resblock_dilations[branch].end());
        gguf_set_arr_data(context.get(), key.c_str(), GGUF_TYPE_INT32, dilations.data(), dilations.size());
    }
    const std::vector<int32_t> upsample_rates(hparams.decoder_upsample_rates.begin(),
                                              hparams.decoder_upsample_rates.end());
    const std::vector<int32_t> upsample_kernels(hparams.decoder_upsample_kernel_sizes.begin(),
                                                hparams.decoder_upsample_kernel_sizes.end());
    gguf_set_arr_data(context.get(), "synthesize.vits.decoder.upsample_rates", GGUF_TYPE_INT32, upsample_rates.data(),
                      upsample_rates.size());
    gguf_set_arr_data(context.get(), "synthesize.vits.decoder.upsample_kernel_sizes", GGUF_TYPE_INT32,
                      upsample_kernels.data(), upsample_kernels.size());
    gguf_set_val_u32(context.get(), "synthesize.vits.decoder.upsample_initial_channels",
                     hparams.decoder_initial_channels);
    gguf_set_val_f32(context.get(), "synthesize.vits.decoder.leaky_relu_slope", hparams.decoder_leaky_relu_slope);
    gguf_set_val_str(context.get(), "synthesize.vits.decoder.output_activation", "tanh");
    gguf_set_val_u32(context.get(), "synthesize.vits.hop_length", hparams.hop_length);
    gguf_set_val_u32(context.get(), "synthesize.vits.speaker_count", 0);
    gguf_set_val_u32(context.get(), "synthesize.vits.conditioning_channels", 0);
    gguf_set_val_str(context.get(), "synthesize.voice.mode", "fixed-default");
    gguf_set_val_bool(context.get(), "synthesize.voice.has_package_default", true);
    gguf_set_val_u32(context.get(), "synthesize.voice.preset_count", 0);
    gguf_set_val_u64(context.get(), "synthesize.capabilities.max_input_tokens", hparams.max_input_tokens);
    gguf_set_val_u64(context.get(), "synthesize.capabilities.max_output_frames", hparams.max_output_frames);
    gguf_set_val_f32(context.get(), "synthesize.vits.text.layer_norm_epsilon", hparams.layer_norm_epsilon);
    gguf_set_val_f32(context.get(), "synthesize.vits.text.embedding_scale", hparams.embedding_scale);
    gguf_set_val_f32(context.get(), "synthesize.vits.duration.spline_tail_bound", hparams.duration_spline_tail_bound);
    gguf_set_val_f32(context.get(), "synthesize.vits.duration.spline_min_bin_width",
                     hparams.duration_spline_min_bin_width);
    gguf_set_val_f32(context.get(), "synthesize.vits.duration.spline_min_bin_height",
                     hparams.duration_spline_min_bin_height);
    gguf_set_val_f32(context.get(), "synthesize.vits.duration.spline_min_derivative",
                     hparams.duration_spline_min_derivative);
    gguf_set_val_f32(context.get(), "synthesize.vits.inference.noise_scale_w", hparams.duration_noise_scale_w);
    gguf_set_val_f32(context.get(), "synthesize.vits.inference.noise_scale", hparams.latent_noise_scale);
    gguf_set_val_f32(context.get(), "synthesize.capabilities.min_speaking_rate", hparams.min_speaking_rate);
    gguf_set_val_f32(context.get(), "synthesize.capabilities.max_speaking_rate", hparams.max_speaking_rate);
    return context;
}

bool rejected(const std::function<void(gguf_context *)> & mutate) {
    GgufContext context = valid_metadata();
    mutate(context.get());
    synth::vits::HParams output;
    return synth::vits::read_hparams(context.get(), output) == SYNTH_ERR_GGUF;
}

GgufContext valid_vctk_metadata() {
    GgufContext context = valid_metadata();
    gguf_set_val_str(context.get(), "synthesize.model_variant", "vits-vctk");
    gguf_set_val_u32(context.get(), "synthesize.vits.speaker_count", 109);
    gguf_set_val_u32(context.get(), "synthesize.vits.conditioning_channels", 256);
    gguf_set_val_str(context.get(), "synthesize.voice.mode", "preset-catalog");
    gguf_set_val_bool(context.get(), "synthesize.voice.has_package_default", false);
    gguf_set_val_u32(context.get(), "synthesize.voice.preset_count", 109);
    for (uint32_t index = 0; index < 109; ++index) {
        char id[12];
        std::snprintf(id, sizeof(id), "speaker-%03u", index);
        const std::string prefix = "synthesize.voice." + std::to_string(index);
        gguf_set_val_str(context.get(), (prefix + ".id").c_str(), id);
        gguf_set_val_u32(context.get(), (prefix + ".flags").c_str(), 0);
    }
    return context;
}

}  // namespace

int main() {
    synth::vits::HParams output;
    SYNTH_TEST_CHECK(synth::vits::read_hparams(nullptr, output) == SYNTH_ERR_INVALID_ARG);

    GgufContext empty(gguf_init_empty());
    SYNTH_TEST_CHECK(synth::vits::read_hparams(empty.get(), output) == SYNTH_ERR_GGUF);

    GgufContext valid = valid_metadata();
    SYNTH_TEST_CHECK(synth::vits::read_hparams(valid.get(), output) == SYNTH_OK);
    const synth::vits::HParams expected = synth::test::small_vits_hparams();
    SYNTH_TEST_CHECK(output.quantization_profile == synth::vits::QuantizationProfile::F32);
    SYNTH_TEST_CHECK(output.quantization_profile_version == 1);
    SYNTH_TEST_CHECK(output.input_flags == (SYNTH_INPUT_SUPPORT_PHONEMES_UTF8 | SYNTH_INPUT_SUPPORT_TOKEN_IDS));
    SYNTH_TEST_CHECK(output.frontend_present);
    SYNTH_TEST_CHECK(output.frontend_config.provider_id == "synthesize.symbol_map");
    SYNTH_TEST_CHECK(output.frontend_config.contract_version == 1);
    SYNTH_TEST_CHECK(output.frontend_config.mapping_mode == synth::SymbolMappingMode::UnicodeScalar);
    SYNTH_TEST_CHECK(output.frontend_config.symbols == std::vector<std::string>({ "_", "a", "b", "c", "d" }));
    SYNTH_TEST_CHECK(output.frontend_config.blank_id == 0 &&
                     output.frontend_config.padding_rule == synth::SymbolPaddingRule::InterleavedBlank);
    SYNTH_TEST_CHECK(output.capability_flags == expected.capability_flags);
    SYNTH_TEST_CHECK(output.output_sample_rate == expected.output_sample_rate);
    SYNTH_TEST_CHECK(output.output_channel_count == expected.output_channel_count);
    SYNTH_TEST_CHECK(output.vocab_size == expected.vocab_size);
    SYNTH_TEST_CHECK(output.inter_channels == expected.inter_channels);
    SYNTH_TEST_CHECK(output.hidden_channels == expected.hidden_channels);
    SYNTH_TEST_CHECK(output.filter_channels == expected.filter_channels);
    SYNTH_TEST_CHECK(output.text_layer_count == expected.text_layer_count);
    SYNTH_TEST_CHECK(output.text_head_count == expected.text_head_count);
    SYNTH_TEST_CHECK(output.text_ffn_kernel_size == expected.text_ffn_kernel_size);
    SYNTH_TEST_CHECK(output.text_attention_window == expected.text_attention_window);
    SYNTH_TEST_CHECK(output.duration_dds_layer_count == expected.duration_dds_layer_count);
    SYNTH_TEST_CHECK(output.duration_flow_count == expected.duration_flow_count);
    SYNTH_TEST_CHECK(output.duration_spline_bin_count == expected.duration_spline_bin_count);
    SYNTH_TEST_CHECK(output.flow_block_count == expected.flow_block_count);
    SYNTH_TEST_CHECK(output.flow_kernel_size == expected.flow_kernel_size);
    SYNTH_TEST_CHECK(output.flow_dilation_rate == expected.flow_dilation_rate);
    SYNTH_TEST_CHECK(output.flow_wn_layer_count == expected.flow_wn_layer_count);
    SYNTH_TEST_CHECK(output.flow_mean_only == expected.flow_mean_only);
    SYNTH_TEST_CHECK(output.decoder_resblock_kernel_sizes == expected.decoder_resblock_kernel_sizes);
    SYNTH_TEST_CHECK(output.decoder_resblock_dilations == expected.decoder_resblock_dilations);
    SYNTH_TEST_CHECK(output.decoder_upsample_rates == expected.decoder_upsample_rates);
    SYNTH_TEST_CHECK(output.decoder_upsample_kernel_sizes == expected.decoder_upsample_kernel_sizes);
    SYNTH_TEST_CHECK(output.decoder_initial_channels == expected.decoder_initial_channels);
    SYNTH_TEST_CHECK(output.decoder_leaky_relu_slope == expected.decoder_leaky_relu_slope);
    SYNTH_TEST_CHECK(output.hop_length == expected.hop_length);
    SYNTH_TEST_CHECK(output.max_input_tokens == expected.max_input_tokens);
    SYNTH_TEST_CHECK(output.max_output_frames == expected.max_output_frames);
    SYNTH_TEST_CHECK(output.layer_norm_epsilon == expected.layer_norm_epsilon);
    SYNTH_TEST_CHECK(output.embedding_scale == expected.embedding_scale);
    SYNTH_TEST_CHECK(output.duration_spline_tail_bound == expected.duration_spline_tail_bound);
    SYNTH_TEST_CHECK(output.duration_spline_min_bin_width == expected.duration_spline_min_bin_width);
    SYNTH_TEST_CHECK(output.duration_spline_min_bin_height == expected.duration_spline_min_bin_height);
    SYNTH_TEST_CHECK(output.duration_spline_min_derivative == expected.duration_spline_min_derivative);
    SYNTH_TEST_CHECK(output.duration_noise_scale_w == expected.duration_noise_scale_w);
    SYNTH_TEST_CHECK(output.latent_noise_scale == expected.latent_noise_scale);
    SYNTH_TEST_CHECK(output.min_speaking_rate == expected.min_speaking_rate);
    SYNTH_TEST_CHECK(output.max_speaking_rate == expected.max_speaking_rate);

    GgufContext vctk = valid_vctk_metadata();
    SYNTH_TEST_CHECK(synth::vits::read_hparams(vctk.get(), output) == SYNTH_OK);
    SYNTH_TEST_CHECK(output.model_variant == "vits-vctk");
    SYNTH_TEST_CHECK(output.speaker_count == 109 && output.conditioning_channels == 256);
    SYNTH_TEST_CHECK(!output.has_package_default && output.preset_voice_ids.size() == 109);
    SYNTH_TEST_CHECK(output.preset_voice_ids.front() == "speaker-000");
    SYNTH_TEST_CHECK(output.preset_voice_ids.back() == "speaker-108");

    GgufContext q8_mixed = valid_metadata();
    gguf_set_val_u32(q8_mixed.get(), "general.file_type", GGML_FTYPE_MOSTLY_Q8_0);
    gguf_set_val_str(q8_mixed.get(), "synthesize.quantization.profile", "Q8_MIXED");
    SYNTH_TEST_CHECK(synth::vits::read_hparams(q8_mixed.get(), output) == SYNTH_OK);
    SYNTH_TEST_CHECK(output.quantization_profile == synth::vits::QuantizationProfile::Q8Mixed);
    SYNTH_TEST_CHECK(output.quantization_profile_version == 1);

    GgufContext bad_vctk = valid_vctk_metadata();
    gguf_set_val_u32(bad_vctk.get(), "synthesize.vits.speaker_count", 108);
    SYNTH_TEST_CHECK(synth::vits::read_hparams(bad_vctk.get(), output) == SYNTH_ERR_GGUF);

    SYNTH_TEST_CHECK(
        rejected([](gguf_context * context) { gguf_set_val_str(context, "synthesize.model_variant", "vits-vctk"); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * context) { gguf_set_val_u64(context, "synthesize.vits.vocab_size", 5); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * context) { gguf_set_val_u32(context, "synthesize.format_version", 2); }));
    SYNTH_TEST_CHECK(rejected(
        [](gguf_context * context) { gguf_set_val_str(context, "synthesize.quantization.profile", "Q4_K_M"); }));
    SYNTH_TEST_CHECK(rejected(
        [](gguf_context * context) { gguf_set_val_u32(context, "synthesize.quantization.profile_version", 2); }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * context) { gguf_set_val_u32(context, "general.file_type", 1); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * context) { gguf_set_val_u32(context, "synthesize.capabilities.input_flags", 0); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * context) { gguf_set_val_str(context, "synthesize.frontend.provider", "unknown"); }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * context) {
        gguf_set_val_str(context, "synthesize.frontend.phoneme_mapping", "delimited_symbol");
    }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * context) { gguf_set_val_u32(context, "synthesize.capabilities.flags", 0); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * context) { gguf_set_val_u32(context, "synthesize.audio.sample_rate_hz", 0); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * context) { gguf_set_val_u32(context, "synthesize.audio.channels", 2); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * context) { gguf_set_val_str(context, "synthesize.audio.sample_format", "s16le"); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * context) { gguf_set_val_u32(context, "synthesize.vits.hidden_channels", 5); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * context) { gguf_set_val_u32(context, "synthesize.vits.text.ffn_kernel_size", 2); }));
    SYNTH_TEST_CHECK(rejected(
        [](gguf_context * context) { gguf_set_val_u32(context, "synthesize.vits.text.attention_window", 0); }));
    SYNTH_TEST_CHECK(rejected(
        [](gguf_context * context) { gguf_set_val_u64(context, "synthesize.capabilities.max_input_tokens", 0); }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * context) {
        gguf_set_val_f32(context, "synthesize.vits.text.layer_norm_epsilon", std::numeric_limits<float>::quiet_NaN());
    }));
    SYNTH_TEST_CHECK(rejected(
        [](gguf_context * context) { gguf_set_val_f32(context, "synthesize.vits.text.embedding_scale", 0.0f); }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * context) {
        gguf_set_val_str(context, "synthesize.vits.duration.predictor_type", "deterministic");
    }));
    SYNTH_TEST_CHECK(rejected(
        [](gguf_context * context) { gguf_set_val_u32(context, "synthesize.vits.duration.spline_bin_count", 1); }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * context) {
        gguf_set_val_f32(context, "synthesize.vits.duration.spline_min_bin_width", 0.5f);
    }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * context) {
        gguf_set_val_f32(context, "synthesize.vits.duration.spline_tail_bound",
                         std::numeric_limits<float>::quiet_NaN());
    }));
    SYNTH_TEST_CHECK(rejected(
        [](gguf_context * context) { gguf_set_val_f32(context, "synthesize.vits.inference.noise_scale_w", -0.1f); }));
    SYNTH_TEST_CHECK(rejected(
        [](gguf_context * context) { gguf_set_val_f32(context, "synthesize.vits.inference.noise_scale", -0.1f); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * context) { gguf_set_val_u32(context, "synthesize.vits.flow.block_count", 0); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * context) { gguf_set_val_u32(context, "synthesize.vits.flow.kernel_size", 4); }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * context) {
        gguf_set_val_u32(context, "synthesize.vits.flow.dilation_rate", std::numeric_limits<uint32_t>::max());
    }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * context) { gguf_set_val_bool(context, "synthesize.vits.flow.mean_only", false); }));
    SYNTH_TEST_CHECK(rejected(
        [](gguf_context * context) { gguf_set_val_str(context, "synthesize.vits.decoder.resblock_type", "2"); }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * context) {
        const int32_t values[] = { 4 };
        gguf_set_arr_data(context, "synthesize.vits.decoder.resblock_kernel_sizes", GGUF_TYPE_INT32, values, 1);
    }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * context) {
        const int32_t values[] = { 1, 0, 3 };
        gguf_set_arr_data(context, "synthesize.vits.decoder.resblock_dilations.0", GGUF_TYPE_INT32, values, 3);
    }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * context) {
        const int32_t values[] = { 4, 2 };
        gguf_set_arr_data(context, "synthesize.vits.decoder.upsample_rates", GGUF_TYPE_INT32, values, 2);
    }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * context) {
        const int32_t values[] = { 7 };
        gguf_set_arr_data(context, "synthesize.vits.decoder.upsample_kernel_sizes", GGUF_TYPE_INT32, values, 1);
    }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * context) {
        gguf_set_val_u32(context, "synthesize.vits.decoder.upsample_initial_channels", 3);
    }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * context) {
        gguf_set_val_f32(context, "synthesize.vits.decoder.leaky_relu_slope", std::numeric_limits<float>::quiet_NaN());
    }));
    SYNTH_TEST_CHECK(rejected([](gguf_context * context) {
        gguf_set_val_str(context, "synthesize.vits.decoder.output_activation", "identity");
    }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * context) { gguf_set_val_u32(context, "synthesize.vits.hop_length", 0); }));
    SYNTH_TEST_CHECK(
        rejected([](gguf_context * context) { gguf_set_val_u32(context, "synthesize.vits.hop_length", 8); }));
    SYNTH_TEST_CHECK(rejected(
        [](gguf_context * context) { gguf_set_val_u64(context, "synthesize.capabilities.max_output_frames", 0); }));
    SYNTH_TEST_CHECK(rejected(
        [](gguf_context * context) { gguf_set_val_f32(context, "synthesize.capabilities.min_speaking_rate", 0.0f); }));
    SYNTH_TEST_CHECK(rejected(
        [](gguf_context * context) { gguf_set_val_f32(context, "synthesize.capabilities.min_speaking_rate", 2.5f); }));
    return 0;
}
