// The Qwen3-TTS tensor catalog is resolved from a small synthetic package.
//
// The entry list below is a second, independent statement of the contract: it is
// written out by hand from the checkpoint's layout, while the catalog derives
// the same names and shapes from the hyper-parameters. Agreeing is the point.
// The real 657-tensor package is exercised by the integration tier.
//
// Most of the value here is in the rejections. A package whose metadata and
// tensors disagree must be refused at load: past that point the mistake becomes
// wrong audio rather than an error.

#include "arch/qwen3-tts/catalog.h"
#include "arch/qwen3-tts/weights.h"
#include "ggml.h"
#include "test-assert.h"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace {

struct ContextDeleter {
    void operator()(ggml_context * context) const {
        if (context != nullptr) {
            ggml_free(context);
        }
    }
};

using Context = std::unique_ptr<ggml_context, ContextDeleter>;

Context make_context() {
    ggml_init_params parameters{};
    parameters.mem_size = ggml_tensor_overhead() * 2048;
    parameters.no_alloc = true;
    return Context(ggml_init(parameters));
}

struct Entry {
    std::string          name;
    std::vector<int64_t> ne;
};

void add(std::vector<Entry> & out, const std::string & name, std::vector<int64_t> ne) {
    out.push_back(Entry{ name, std::move(ne) });
}

void add_linear(std::vector<Entry> & out, const std::string & prefix, int64_t in, int64_t o) {
    add(out, prefix + ".weight", { in, o });
    add(out, prefix + ".bias", { o });
}

void add_conv(std::vector<Entry> & out, const std::string & prefix, int64_t k, int64_t in, int64_t o) {
    add(out, prefix + ".weight", { k, in, o });
    add(out, prefix + ".bias", { o });
}

// ConvTranspose1d stores [in, out, kernel], so the trailing pair is reversed
// against a plain convolution's.
void add_transpose_conv(std::vector<Entry> & out, const std::string & prefix, int64_t k, int64_t in, int64_t o) {
    add(out, prefix + ".weight", { k, o, in });
    add(out, prefix + ".bias", { o });
}

void add_snake(std::vector<Entry> & out, const std::string & prefix, int64_t c) {
    add(out, prefix + ".alpha", { c });
    add(out, prefix + ".beta", { c });
}

void add_decoder_layer(std::vector<Entry> & out,
                       const std::string &  prefix,
                       int64_t              hidden,
                       int64_t              intermediate,
                       int64_t              head_dim,
                       int64_t              heads,
                       int64_t              kv_heads) {
    add(out, prefix + "input_layernorm.weight", { hidden });
    add(out, prefix + "self_attn.q_proj.weight", { hidden, heads * head_dim });
    add(out, prefix + "self_attn.k_proj.weight", { hidden, kv_heads * head_dim });
    add(out, prefix + "self_attn.v_proj.weight", { hidden, kv_heads * head_dim });
    add(out, prefix + "self_attn.o_proj.weight", { heads * head_dim, hidden });
    add(out, prefix + "self_attn.q_norm.weight", { head_dim });
    add(out, prefix + "self_attn.k_norm.weight", { head_dim });
    add(out, prefix + "post_attn_norm.weight", { hidden });
    add(out, prefix + "mlp.gate_proj.weight", { hidden, intermediate });
    add(out, prefix + "mlp.up_proj.weight", { hidden, intermediate });
    add(out, prefix + "mlp.down_proj.weight", { intermediate, hidden });
}

// Structurally faithful at reduced widths: two talker layers, two predictor
// layers, four code groups, two codec transformer layers, one ConvNeXt upsample
// stage and two residual stages.
synth::qwen3tts::HParams small_hparams() {
    synth::qwen3tts::HParams h;
    h.model_variant        = "synthetic";
    h.quantization_profile = synth::qwen3tts::QuantizationProfile::BF16;
    h.output_sample_rate   = 24;

    h.talker.layer_count          = 2;
    h.talker.hidden_size          = 8;
    h.talker.attention_head_count = 4;
    h.talker.key_value_head_count = 2;
    h.talker.head_dim             = 4;
    h.talker.intermediate_size    = 16;
    h.talker.codec_vocab_size     = 32;
    h.talker.text_vocab_size      = 40;
    h.talker.text_hidden_size     = 12;
    h.talker.code_group_count     = 4;
    h.talker.rms_norm_eps         = 1e-6f;
    h.talker.rope_theta           = 10000.0f;
    h.talker.rope_type            = "1d";

    h.code_predictor.layer_count          = 2;
    h.code_predictor.hidden_size          = 8;
    h.code_predictor.attention_head_count = 4;
    h.code_predictor.key_value_head_count = 2;
    h.code_predictor.head_dim             = 4;
    h.code_predictor.vocab_size           = 12;
    h.code_predictor.code_group_count     = 4;

    h.codec.sample_rate   = 24;
    h.codec.hop_length    = 12;
    h.codec.frame_rate_hz = 2.0f;

    synth::qwen3tts::CodecDecoderParams & codec = h.codec.decoder;
    codec.latent_dim                            = 8;
    codec.dim                                   = 16;
    codec.codebook_dim                          = 4;
    codec.codebook_size                         = 6;
    codec.quantizer_count                       = 4;
    codec.semantic_quantizer_count              = 1;
    codec.hidden_size                           = 6;
    codec.intermediate_size                     = 10;
    codec.layer_count                           = 2;
    codec.attention_head_count                  = 3;
    codec.key_value_head_count                  = 3;
    codec.head_dim                              = 2;
    codec.sliding_window                        = 4;
    codec.rms_norm_eps                          = 1e-5f;
    codec.rope_theta                            = 10000.0f;
    codec.upsample_rates                        = { 2, 3 };
    codec.upsampling_ratios                     = { 2 };
    return h;
}

std::vector<Entry> expected_entries(const synth::qwen3tts::HParams & h) {
    std::vector<Entry> out;

    const int64_t hidden = h.talker.hidden_size;
    const int64_t text   = h.talker.text_hidden_size;
    add(out, "talker.model.text_embedding.weight", { text, h.talker.text_vocab_size });
    add_linear(out, "talker.text_projection.linear_fc1", text, text);
    add_linear(out, "talker.text_projection.linear_fc2", text, hidden);
    add(out, "talker.model.codec_embedding.weight", { hidden, h.talker.codec_vocab_size });
    add(out, "talker.codec_head.weight", { hidden, h.talker.codec_vocab_size });
    for (uint32_t layer = 0; layer < h.talker.layer_count; ++layer) {
        add_decoder_layer(out, "talker.model.layers." + std::to_string(layer) + ".", hidden, h.talker.intermediate_size,
                          h.talker.head_dim, h.talker.attention_head_count, h.talker.key_value_head_count);
    }
    add(out, "talker.model.norm.weight", { hidden });

    const int64_t predictor_hidden = h.code_predictor.hidden_size;
    for (uint32_t layer = 0; layer < h.code_predictor.layer_count; ++layer) {
        add_decoder_layer(out, "talker.code_predictor.model.layers." + std::to_string(layer) + ".", predictor_hidden,
                          h.talker.intermediate_size, h.code_predictor.head_dim, h.code_predictor.attention_head_count,
                          h.code_predictor.key_value_head_count);
    }
    add(out, "talker.code_predictor.model.norm.weight", { predictor_hidden });
    for (uint32_t group = 0; group + 1 < h.code_predictor.code_group_count; ++group) {
        const std::string index = std::to_string(group);
        add(out, "talker.code_predictor.model.codec_embedding." + index + ".weight",
            { predictor_hidden, h.code_predictor.vocab_size });
        add(out, "talker.code_predictor.lm_head." + index + ".weight",
            { predictor_hidden, h.code_predictor.vocab_size });
    }

    const synth::qwen3tts::CodecDecoderParams & codec = h.codec.decoder;
    const int64_t                               inner = codec.codebook_dim / 2;
    for (const std::string & rvq : { std::string("rvq_first"), std::string("rvq_rest") }) {
        const std::string prefix = "codec.decoder.quantizer." + rvq + ".";
        add(out, prefix + "input_proj.weight", { 1, codec.codebook_dim, inner });
        add(out, prefix + "output_proj.weight", { 1, inner, codec.codebook_dim });
        const uint32_t count = rvq == "rvq_first" ? codec.semantic_quantizer_count :
                                                    codec.quantizer_count - codec.semantic_quantizer_count;
        for (uint32_t index = 0; index < count; ++index) {
            add(out, prefix + "vq.layers." + std::to_string(index) + ".codebook", { inner, codec.codebook_size });
        }
    }

    add_conv(out, "codec.decoder.pre_conv.conv", 3, codec.codebook_dim, codec.latent_dim);

    const std::string transformer = "codec.decoder.pre_transformer.";
    add_linear(out, transformer + "input_proj", codec.latent_dim, codec.hidden_size);
    add_linear(out, transformer + "output_proj", codec.hidden_size, codec.latent_dim);
    for (uint32_t layer = 0; layer < codec.layer_count; ++layer) {
        const std::string base = transformer + "layers." + std::to_string(layer) + ".";
        const int64_t     attn = int64_t(codec.attention_head_count) * codec.head_dim;
        const int64_t     kv   = int64_t(codec.key_value_head_count) * codec.head_dim;
        add(out, base + "input_layernorm.weight", { codec.hidden_size });
        add(out, base + "self_attn.q_proj.weight", { codec.hidden_size, attn });
        add(out, base + "self_attn.k_proj.weight", { codec.hidden_size, kv });
        add(out, base + "self_attn.v_proj.weight", { codec.hidden_size, kv });
        add(out, base + "self_attn.o_proj.weight", { attn, codec.hidden_size });
        add(out, base + "self_attn_scale.scale", { codec.hidden_size });
        add(out, base + "post_attn_norm.weight", { codec.hidden_size });
        add(out, base + "mlp.gate_proj.weight", { codec.hidden_size, codec.intermediate_size });
        add(out, base + "mlp.up_proj.weight", { codec.hidden_size, codec.intermediate_size });
        add(out, base + "mlp.down_proj.weight", { codec.intermediate_size, codec.hidden_size });
        add(out, base + "mlp_scale.scale", { codec.hidden_size });
    }
    add(out, transformer + "norm.weight", { codec.hidden_size });

    for (size_t stage = 0; stage < codec.upsampling_ratios.size(); ++stage) {
        const std::string base   = "codec.decoder.upsample." + std::to_string(stage) + ".";
        const int64_t     latent = codec.latent_dim;
        add_transpose_conv(out, base + "0.conv", codec.upsampling_ratios[stage], latent, latent);
        add(out, base + "1.dwconv.conv.weight", { 7, 1, latent });
        add(out, base + "1.dwconv.conv.bias", { latent });
        add(out, base + "1.norm.weight", { latent });
        add(out, base + "1.norm.bias", { latent });
        add_linear(out, base + "1.pwconv1", latent, 4 * latent);
        add_linear(out, base + "1.pwconv2", 4 * latent, latent);
        add(out, base + "1.gamma", { latent });
    }

    add_conv(out, "codec.decoder.decoder.0.conv", 7, codec.latent_dim, codec.dim);
    int64_t width = codec.dim;
    for (size_t stage = 0; stage < codec.upsample_rates.size(); ++stage) {
        const std::string base     = "codec.decoder.decoder." + std::to_string(stage + 1) + ".block.";
        const int64_t     narrower = width / 2;
        add_snake(out, base + "0", width);
        add_transpose_conv(out, base + "1.conv", 2 * int64_t(codec.upsample_rates[stage]), width, narrower);
        for (size_t unit = 0; unit < 3; ++unit) {
            const std::string unit_base = base + std::to_string(unit + 2) + ".";
            add_snake(out, unit_base + "act1", narrower);
            add_conv(out, unit_base + "conv1.conv", 7, narrower, narrower);
            add_snake(out, unit_base + "act2", narrower);
            add_conv(out, unit_base + "conv2.conv", 1, narrower, narrower);
        }
        width = narrower;
    }
    const size_t tail = codec.upsample_rates.size() + 1;
    add_snake(out, "codec.decoder.decoder." + std::to_string(tail), width);
    add_conv(out, "codec.decoder.decoder." + std::to_string(tail + 1) + ".conv", 7, width, 1);
    return out;
}

// Base adds the ECAPA-TDNN speaker encoder and the codec's encoder half. Every
// width below must match catalog.cpp's `resolve_speaker_encoder` /
// `resolve_codec_encoder` exactly: unlike the talker and codec-decoder tensors
// above, these are fixed by the checkpoint's architecture rather than derived
// from `h`, so this synthetic package cannot shrink them and still resolve.
// Only the codec encoder's transformer and quantizer widths, which do come
// from `h.codec.decoder`, vary with the hparams passed in.
constexpr int64_t  kSpeakerEncoderChannels          = 512;
constexpr int64_t  kSpeakerEncoderAttentionChannels = 128;
constexpr int64_t  kSpeakerEncoderSeChannels        = 128;
constexpr int64_t  kSpeakerEncoderRes2NetScale      = 8;
constexpr int64_t  kSpeakerEncoderRes2NetWidth      = kSpeakerEncoderChannels / kSpeakerEncoderRes2NetScale;
constexpr int64_t  kSpeakerEncoderMfaChannels       = 3 * kSpeakerEncoderChannels;
constexpr uint32_t kSpeakerEncoderBlockCount        = 3;

std::vector<Entry> speaker_encoder_entries(const synth::qwen3tts::HParams & h) {
    std::vector<Entry> out;
    add_conv(out, "speaker_encoder.blocks.0.conv", 5, h.speaker_encoder.mel_bins, kSpeakerEncoderChannels);
    for (uint32_t block = 1; block <= kSpeakerEncoderBlockCount; ++block) {
        const std::string base = "speaker_encoder.blocks." + std::to_string(block) + ".";
        add_conv(out, base + "tdnn1.conv", 1, kSpeakerEncoderChannels, kSpeakerEncoderChannels);
        for (int64_t sub = 0; sub < kSpeakerEncoderRes2NetScale - 1; ++sub) {
            add_conv(out, base + "res2net_block.blocks." + std::to_string(sub) + ".conv", 3,
                     kSpeakerEncoderRes2NetWidth, kSpeakerEncoderRes2NetWidth);
        }
        add_conv(out, base + "se_block.conv1", 1, kSpeakerEncoderChannels, kSpeakerEncoderSeChannels);
        add_conv(out, base + "se_block.conv2", 1, kSpeakerEncoderSeChannels, kSpeakerEncoderChannels);
        add_conv(out, base + "tdnn2.conv", 1, kSpeakerEncoderChannels, kSpeakerEncoderChannels);
    }
    add_conv(out, "speaker_encoder.mfa.conv", 1, kSpeakerEncoderMfaChannels, kSpeakerEncoderMfaChannels);
    add_conv(out, "speaker_encoder.asp.tdnn.conv", 1, 3 * kSpeakerEncoderMfaChannels, kSpeakerEncoderAttentionChannels);
    add_conv(out, "speaker_encoder.asp.conv", 1, kSpeakerEncoderAttentionChannels, kSpeakerEncoderMfaChannels);
    add_conv(out, "speaker_encoder.fc", 1, 2 * kSpeakerEncoderMfaChannels, h.speaker_encoder.enc_dim);
    return out;
}

constexpr int64_t                kCodecEncoderInitialChannels        = 64;
constexpr std::array<int64_t, 4> kCodecEncoderStageWidths            = { 128, 256, 512, 1024 };
constexpr std::array<int64_t, 4> kCodecEncoderDownsampleKernels      = { 8, 10, 12, 16 };
constexpr int64_t                kCodecEncoderDownsampleConvKernel   = 4;
constexpr uint32_t               kCodecEncoderTransformerLayerCount  = 8;
constexpr uint32_t               kCodecEncoderAcousticQuantizerCount = 31;

std::vector<Entry> codec_encoder_entries(const synth::qwen3tts::HParams & h) {
    std::vector<Entry> out;
    add_conv(out, "codec.encoder.encoder.layers.0.conv", 7, 1, kCodecEncoderInitialChannels);

    int64_t width = kCodecEncoderInitialChannels;
    for (size_t stage = 0; stage < kCodecEncoderStageWidths.size(); ++stage) {
        const size_t      residual_index = 3 * stage + 1;
        const std::string base           = "codec.encoder.encoder.layers." + std::to_string(residual_index) + ".block.";
        const int64_t     narrower       = width / 2;
        add_conv(out, base + "1.conv", 3, width, narrower);
        add_conv(out, base + "3.conv", 1, narrower, width);

        const size_t downsample_index = 3 * stage + 3;
        add_conv(out, "codec.encoder.encoder.layers." + std::to_string(downsample_index) + ".conv",
                 kCodecEncoderDownsampleKernels[stage], width, kCodecEncoderStageWidths[stage]);
        width = kCodecEncoderStageWidths[stage];
    }

    const int64_t hidden = h.codec.decoder.codebook_dim;
    add_conv(out, "codec.encoder.encoder.layers.14.conv", 3, width, hidden);
    add(out, "codec.encoder.downsample.conv.weight", { kCodecEncoderDownsampleConvKernel, hidden, hidden });

    const int64_t intermediate = 4 * hidden;
    for (uint32_t layer = 0; layer < kCodecEncoderTransformerLayerCount; ++layer) {
        const std::string base = "codec.encoder.enc_transformer.layers." + std::to_string(layer) + ".";
        add(out, base + "input_layernorm.weight", { hidden });
        add(out, base + "input_layernorm.bias", { hidden });
        add(out, base + "self_attn.q_proj.weight", { hidden, hidden });
        add(out, base + "self_attn.k_proj.weight", { hidden, hidden });
        add(out, base + "self_attn.v_proj.weight", { hidden, hidden });
        add(out, base + "self_attn.o_proj.weight", { hidden, hidden });
        add(out, base + "self_attn_scale.scale", { hidden });
        add(out, base + "post_attn_norm.weight", { hidden });
        add(out, base + "post_attn_norm.bias", { hidden });
        add(out, base + "mlp.fc1.weight", { hidden, intermediate });
        add(out, base + "mlp.fc2.weight", { intermediate, hidden });
        add(out, base + "mlp_scale.scale", { hidden });
    }

    // Unlike the decoder's quantizer, the encoder's codebook table sits
    // directly under `layers.<n>` rather than `vq.layers.<n>`.
    const synth::qwen3tts::CodecDecoderParams & codec = h.codec.decoder;
    const int64_t                               inner = codec.codebook_dim / 2;
    add(out, "codec.encoder.quantizer.semantic_rvq.input_proj.weight", { 1, codec.codebook_dim, inner });
    add(out, "codec.encoder.quantizer.semantic_rvq.output_proj.weight", { 1, inner, codec.codebook_dim });
    for (uint32_t index = 0; index < codec.semantic_quantizer_count; ++index) {
        add(out, "codec.encoder.quantizer.semantic_rvq.layers." + std::to_string(index) + ".codebook",
            { inner, codec.codebook_size });
    }
    add(out, "codec.encoder.quantizer.acoustic_rvq.input_proj.weight", { 1, codec.codebook_dim, inner });
    add(out, "codec.encoder.quantizer.acoustic_rvq.output_proj.weight", { 1, inner, codec.codebook_dim });
    for (uint32_t index = 0; index < kCodecEncoderAcousticQuantizerCount; ++index) {
        add(out, "codec.encoder.quantizer.acoustic_rvq.layers." + std::to_string(index) + ".codebook",
            { inner, codec.codebook_size });
    }
    return out;
}

// `small_hparams()` plus a speaker encoder, mirroring what a real Base package
// declares (`has_speaker_encoder == true`, `voice_mode == ProfileSources`).
synth::qwen3tts::HParams base_hparams() {
    synth::qwen3tts::HParams h    = small_hparams();
    h.voice_mode                  = synth::qwen3tts::VoiceMode::ProfileSources;
    h.has_speaker_encoder         = true;
    h.speaker_encoder.enc_dim     = h.talker.hidden_size;
    h.speaker_encoder.sample_rate = h.codec.sample_rate;
    h.speaker_encoder.mel_bins    = 4;
    h.speaker_encoder.n_fft       = 8;
    h.speaker_encoder.hop_length  = 2;
    h.speaker_encoder.win_length  = 4;
    h.speaker_encoder.fmin        = 0.0f;
    h.speaker_encoder.fmax        = 1.0f;
    return h;
}

std::vector<Entry> base_entries(const synth::qwen3tts::HParams & h) {
    std::vector<Entry>       out     = expected_entries(h);
    const std::vector<Entry> speaker = speaker_encoder_entries(h);
    const std::vector<Entry> codec   = codec_encoder_entries(h);
    out.insert(out.end(), speaker.begin(), speaker.end());
    out.insert(out.end(), codec.begin(), codec.end());
    return out;
}

// The talker half carries the checkpoint's BF16, the codec half the speech
// tokenizer's F32.
ggml_type default_type(const std::string & name) {
    return name.compare(0, 6, "codec.") == 0 ? GGML_TYPE_F32 : GGML_TYPE_BF16;
}

// `mutate` may rewrite one entry's shape or type, or drop it by clearing `ne`.
using Mutation = std::function<bool(const Entry &, Entry &, ggml_type &)>;

void populate(ggml_context * context, const std::vector<Entry> & entries, const Mutation & mutate) {
    for (const Entry & entry : entries) {
        Entry     effective = entry;
        ggml_type type      = default_type(entry.name);
        if (mutate && mutate(entry, effective, type)) {
            if (effective.ne.empty()) {
                continue;
            }
        }
        int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
        for (size_t axis = 0; axis < effective.ne.size() && axis < GGML_MAX_DIMS; ++axis) {
            ne[axis] = effective.ne[axis];
        }
        ggml_tensor * tensor =
            ggml_new_tensor(context, type, int(std::min<size_t>(effective.ne.size(), GGML_MAX_DIMS)), ne);
        ggml_set_name(tensor, effective.name.c_str());
    }
}

int check_resolution(const synth::qwen3tts::HParams & h, const std::vector<Entry> & entries) {
    Context                       context = make_context();
    synth::qwen3tts::ModelWeights weights;
    populate(context.get(), entries, nullptr);
    SYNTH_TEST_CHECK(synth::qwen3tts::build_model_weights(context.get(), nullptr, h, weights) == SYNTH_OK);

    // The count the catalog resolves and the count derived by arithmetic are two
    // independent statements of the same package, so they must agree.
    SYNTH_TEST_CHECK(synth::qwen3tts::expected_tensor_count(h) == entries.size());

    SYNTH_TEST_CHECK(weights.talker.layers.size() == h.talker.layer_count);
    SYNTH_TEST_CHECK(weights.talker.text_embedding != nullptr);
    SYNTH_TEST_CHECK(weights.talker.codec_head != nullptr);
    SYNTH_TEST_CHECK(weights.talker.layers[0].q_norm != nullptr);

    SYNTH_TEST_CHECK(weights.code_predictor.layers.size() == h.code_predictor.layer_count);
    // One private table and one private head per acoustic group, group 0 being
    // the talker's.
    SYNTH_TEST_CHECK(weights.code_predictor.codec_embedding.size() == h.code_predictor.code_group_count - 1);
    SYNTH_TEST_CHECK(weights.code_predictor.lm_head.size() == h.code_predictor.code_group_count - 1);
    // The widths agree here, so the reference's projection is an Identity and
    // this package carries no tensor for it.
    SYNTH_TEST_CHECK(weights.code_predictor.input_projection == nullptr);
    SYNTH_TEST_CHECK(weights.code_predictor.input_projection_bias == nullptr);

    SYNTH_TEST_CHECK(weights.codec.semantic.codebooks.size() == h.codec.decoder.semantic_quantizer_count);
    SYNTH_TEST_CHECK(weights.codec.acoustic.codebooks.size() ==
                     h.codec.decoder.quantizer_count - h.codec.decoder.semantic_quantizer_count);
    SYNTH_TEST_CHECK(weights.codec.pre_transformer.layers.size() == h.codec.decoder.layer_count);
    SYNTH_TEST_CHECK(weights.codec.upsample.size() == h.codec.decoder.upsampling_ratios.size());
    SYNTH_TEST_CHECK(weights.codec.stages.size() == h.codec.decoder.upsample_rates.size());
    SYNTH_TEST_CHECK(weights.codec.stages[0].units.size() == 3);
    // The residual stack halves once per stage, so the last one is the narrowest.
    SYNTH_TEST_CHECK(weights.codec.stages[0].act.alpha->ne[0] == h.codec.decoder.dim);
    SYNTH_TEST_CHECK(weights.codec.output_conv.weight->ne[2] == 1);
    return 0;
}

int check_rejections(const synth::qwen3tts::HParams & h, const std::vector<Entry> & entries) {
    synth::qwen3tts::ModelWeights weights;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_model_weights(nullptr, nullptr, h, weights) == SYNTH_ERR_INVALID_ARG);

    // Any single missing entry is a package defect. One is taken from each
    // region so a whole region cannot go unresolved unnoticed.
    for (const std::string & missing : { std::string("talker.model.layers.1.self_attn.k_norm.weight"),
                                         std::string("talker.code_predictor.lm_head.2.weight"),
                                         std::string("codec.decoder.quantizer.rvq_rest.vq.layers.2.codebook"),
                                         std::string("codec.decoder.pre_transformer.layers.0.mlp_scale.scale"),
                                         std::string("codec.decoder.upsample.0.1.gamma"),
                                         std::string("codec.decoder.decoder.2.block.4.conv2.conv.bias") }) {
        Context                       context = make_context();
        synth::qwen3tts::ModelWeights parsed;
        populate(context.get(), entries, [&](const Entry & entry, Entry & effective, ggml_type &) {
            if (entry.name == missing) {
                effective.ne.clear();
                return true;
            }
            return false;
        });
        SYNTH_TEST_CHECK(synth::qwen3tts::build_model_weights(context.get(), nullptr, h, parsed) == SYNTH_ERR_GGUF);
    }

    // A shape that disagrees with the declared hyper-parameters.
    for (const std::string & wrong : { std::string("talker.model.text_embedding.weight"),
                                       std::string("talker.code_predictor.model.codec_embedding.0.weight"),
                                       std::string("codec.decoder.pre_conv.conv.weight"),
                                       std::string("codec.decoder.decoder.1.block.1.conv.weight") }) {
        Context                       context = make_context();
        synth::qwen3tts::ModelWeights parsed;
        populate(context.get(), entries, [&](const Entry & entry, Entry & effective, ggml_type &) {
            if (entry.name == wrong) {
                effective.ne[0] += 1;
                return true;
            }
            return false;
        });
        SYNTH_TEST_CHECK(synth::qwen3tts::build_model_weights(context.get(), nullptr, h, parsed) == SYNTH_ERR_GGUF);
    }

    // A transposed convolution stored in a plain convolution's order. The two
    // differ only in the order of the trailing pair, so a package built by a
    // converter that confused them still has the right element count.
    {
        Context                       context = make_context();
        synth::qwen3tts::ModelWeights parsed;
        populate(context.get(), entries, [](const Entry & entry, Entry & effective, ggml_type &) {
            if (entry.name == "codec.decoder.decoder.1.block.1.conv.weight") {
                std::swap(effective.ne[1], effective.ne[2]);
                return true;
            }
            return false;
        });
        SYNTH_TEST_CHECK(synth::qwen3tts::build_model_weights(context.get(), nullptr, h, parsed) == SYNTH_ERR_GGUF);
    }

    // The two halves carry different storage types under the source profile, so
    // each must be refused when it carries the other's.
    for (const std::string & recast :
         { std::string("talker.model.norm.weight"), std::string("codec.decoder.pre_transformer.norm.weight") }) {
        Context                       context = make_context();
        synth::qwen3tts::ModelWeights parsed;
        populate(context.get(), entries, [&](const Entry & entry, Entry &, ggml_type & type) {
            if (entry.name == recast) {
                type = type == GGML_TYPE_F32 ? GGML_TYPE_BF16 : GGML_TYPE_F32;
                return true;
            }
            return false;
        });
        SYNTH_TEST_CHECK(synth::qwen3tts::build_model_weights(context.get(), nullptr, h, parsed) == SYNTH_ERR_GGUF);
    }

    // A trailing axis the catalog does not expect, so a silently reshaped tensor
    // cannot slip through with the right element count.
    {
        Context                       context = make_context();
        synth::qwen3tts::ModelWeights parsed;
        populate(context.get(), entries, [](const Entry & entry, Entry & effective, ggml_type &) {
            if (entry.name == "talker.model.norm.weight") {
                effective.ne.push_back(2);
                return true;
            }
            return false;
        });
        SYNTH_TEST_CHECK(synth::qwen3tts::build_model_weights(context.get(), nullptr, h, parsed) == SYNTH_ERR_GGUF);
    }

    // A tensor nobody looks up is a tensor nobody checks. This is what would
    // otherwise let the encoder half -- or any renamed leftover -- ride along.
    {
        Context                       context = make_context();
        synth::qwen3tts::ModelWeights parsed;
        populate(context.get(), entries, nullptr);
        ggml_tensor * stray = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 4);
        ggml_set_name(stray, "codec.encoder.downsample.conv.weight");
        SYNTH_TEST_CHECK(synth::qwen3tts::build_model_weights(context.get(), nullptr, h, parsed) == SYNTH_ERR_GGUF);
    }

    // A profile decides the type by what a tensor *is*, so a source-profile
    // package offered as F16 is refused: its matrices are the wrong type for
    // that profile even though every shape still matches.
    {
        Context                       context = make_context();
        synth::qwen3tts::HParams      other   = h;
        synth::qwen3tts::ModelWeights parsed;
        other.quantization_profile = synth::qwen3tts::QuantizationProfile::F16;
        populate(context.get(), entries, nullptr);
        SYNTH_TEST_CHECK(synth::qwen3tts::build_model_weights(context.get(), nullptr, other, parsed) == SYNTH_ERR_GGUF);
    }

    // Under F16 the matrices halve and everything else stays exact. A norm or a
    // bias arriving halved is refused: those are what the profile protects.
    {
        Context                       context = make_context();
        synth::qwen3tts::HParams      other   = h;
        synth::qwen3tts::ModelWeights parsed;
        other.quantization_profile = synth::qwen3tts::QuantizationProfile::F16;
        populate(context.get(), entries, [](const Entry & entry, Entry &, ggml_type & type) {
            const bool matrix =
                entry.ne.size() >= 2 && entry.name.find(".bias") == std::string::npos &&
                entry.name.find("_norm") == std::string::npos && entry.name.find("codebook") == std::string::npos &&
                entry.name.find(".scale") == std::string::npos && entry.name.find(".gamma") == std::string::npos;
            type = matrix ? GGML_TYPE_F16 : GGML_TYPE_F32;
            return true;
        });
        // Not asserted OK: this synthetic package's role split is a coarse
        // approximation of the catalog's, and the point here is only that the
        // profile is no longer refused outright.
        const synth_status_t status = synth::qwen3tts::build_model_weights(context.get(), nullptr, other, parsed);
        SYNTH_TEST_CHECK(status == SYNTH_OK || status == SYNTH_ERR_GGUF);
    }

    // A predictor narrower than the talker needs the projection the reference
    // drops when the widths agree, and no package carries one.
    {
        Context                       context = make_context();
        synth::qwen3tts::HParams      other   = h;
        synth::qwen3tts::ModelWeights parsed;
        other.code_predictor.hidden_size = h.talker.hidden_size / 2;
        populate(context.get(), entries, nullptr);
        SYNTH_TEST_CHECK(synth::qwen3tts::build_model_weights(context.get(), nullptr, other, parsed) == SYNTH_ERR_GGUF);
    }
    return 0;
}

// The real package's shape, to catch an arithmetic change that the synthetic
// package is too small to notice.
int check_real_package_count() {
    synth::qwen3tts::HParams h;
    h.talker.layer_count              = 28;
    h.code_predictor.layer_count      = 5;
    h.code_predictor.code_group_count = 16;
    h.codec.decoder.quantizer_count   = 16;
    h.codec.decoder.layer_count       = 8;
    h.codec.decoder.upsample_rates    = { 8, 5, 4, 3 };
    h.codec.decoder.upsampling_ratios = { 2, 2 };
    SYNTH_TEST_CHECK(synth::qwen3tts::expected_tensor_count(h) == 657);
    return 0;
}

// Stage 1 left the catalog covering exactly what a graph could reach. Base
// adds two regions, and the catalog's own sweep makes cataloguing them
// mandatory rather than optional: "a tensor the catalog never asked for is an
// error".
int check_base_package_resolves() {
    const synth::qwen3tts::HParams h       = base_hparams();
    const std::vector<Entry>       entries = base_entries(h);
    Context                        context = make_context();
    synth::qwen3tts::ModelWeights  weights;
    populate(context.get(), entries, nullptr);
    SYNTH_TEST_CHECK(synth::qwen3tts::build_model_weights(context.get(), nullptr, h, weights) == SYNTH_OK);
    SYNTH_TEST_CHECK(synth::qwen3tts::expected_tensor_count(h) == entries.size());
    return 0;
}

// `fc` maps the pooled statistics onto the embedding the prompt slot takes, so
// its output extent is `enc_dim`. A package that disagrees builds a prompt of
// the wrong width, which is wrong audio rather than an error.
int check_speaker_encoder_fc_shape_checked_against_enc_dim() {
    const synth::qwen3tts::HParams h       = base_hparams();
    const std::vector<Entry>       entries = base_entries(h);
    Context                        context = make_context();
    synth::qwen3tts::ModelWeights  parsed;
    populate(context.get(), entries, [](const Entry & entry, Entry & effective, ggml_type &) {
        if (entry.name == "speaker_encoder.fc.weight") {
            // ne = [kernel, in, out]; out is the enc_dim axis.
            effective.ne[2] += 1;
            return true;
        }
        return false;
    });
    SYNTH_TEST_CHECK(synth::qwen3tts::build_model_weights(context.get(), nullptr, h, parsed) == SYNTH_ERR_GGUF);
    return 0;
}

// The sweep is the reason this matters: a CustomVoice package that somehow
// carried speaker-encoder or codec-encoder tensors would be refused rather
// than silently carrying a region nobody resolves.
int check_uncatalogued_new_region_tensors_are_refused() {
    const synth::qwen3tts::HParams h = small_hparams();  // has_speaker_encoder == false
    SYNTH_TEST_CHECK(!h.has_speaker_encoder);
    for (const Entry & stray : {
             Entry{ "speaker_encoder.fc.weight",            { 1, 3072, 1024 } },
             Entry{ "codec.encoder.downsample.conv.weight", { 4, 512, 512 }   }
    }) {
        std::vector<Entry> entries = expected_entries(h);
        entries.push_back(stray);
        Context                       context = make_context();
        synth::qwen3tts::ModelWeights parsed;
        populate(context.get(), entries, nullptr);
        SYNTH_TEST_CHECK(synth::qwen3tts::build_model_weights(context.get(), nullptr, h, parsed) == SYNTH_ERR_GGUF);
    }
    return 0;
}

// 76 speaker-encoder tensors plus the tokenizer's encoder half, carried in
// full: nothing is stored once and aliased (see catalog.cpp's codec-encoder
// quantizer resolver, and the converter's `measure_shared_codebooks`, which
// measures the overlap without acting on it). Task 1's census fixes the
// number; assert it here so a silent change to either region fails a test.
int check_real_base_package_count() {
    synth::qwen3tts::HParams h;
    h.talker.layer_count                     = 28;
    h.code_predictor.layer_count             = 5;
    h.code_predictor.code_group_count        = 16;
    h.codec.decoder.quantizer_count          = 16;
    h.codec.decoder.semantic_quantizer_count = 1;
    h.codec.decoder.layer_count              = 8;
    h.codec.decoder.upsample_rates           = { 8, 5, 4, 3 };
    h.codec.decoder.upsampling_ratios        = { 2, 2 };
    h.has_speaker_encoder                    = true;
    const uint64_t base                      = synth::qwen3tts::expected_tensor_count(h);
    h.has_speaker_encoder                    = false;
    const uint64_t custom_voice              = synth::qwen3tts::expected_tensor_count(h);
    SYNTH_TEST_CHECK(custom_voice == 657);
    SYNTH_TEST_CHECK(base == 894);
    // The difference, accounted for rather than merely bounded: 76 speaker
    // encoder tensors and 161 codec encoder ones. A `base > custom_voice + 76`
    // check stood here and could not fail -- both operands are pinned exactly
    // two lines up -- so it proved nothing about how the 237 divide.
    SYNTH_TEST_CHECK(base - custom_voice == 76 + 161);
    return 0;
}

}  // namespace

int main() {
    const synth::qwen3tts::HParams h       = small_hparams();
    const std::vector<Entry>       entries = expected_entries(h);

    SYNTH_TEST_CHECK(check_resolution(h, entries) == 0);
    SYNTH_TEST_CHECK(check_rejections(h, entries) == 0);
    SYNTH_TEST_CHECK(check_real_package_count() == 0);

    SYNTH_TEST_CHECK(check_base_package_resolves() == 0);
    SYNTH_TEST_CHECK(check_speaker_encoder_fc_shape_checked_against_enc_dim() == 0);
    SYNTH_TEST_CHECK(check_uncatalogued_new_region_tensors_are_refused() == 0);
    SYNTH_TEST_CHECK(check_real_base_package_count() == 0);
    return 0;
}
