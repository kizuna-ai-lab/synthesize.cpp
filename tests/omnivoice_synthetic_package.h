#pragma once

// Writes a miniature but fully legal omnivoice package to disk: every metadata
// key the reader demands, every tensor the catalog resolves (the small layout
// the catalog test already proves legal), real F32 payloads, and a minimal
// frontend. Options knock out exactly one thing so a test can prove one
// refusal.

#include "ggml.h"
#include "gguf.h"
#include "omnivoice_small_layout.h"

#include <cstdint>
#include <string>
#include <vector>

namespace synth::omnivoice::testing {

struct SyntheticPackageOptions {
    bool  omit_frontend_vocab  = false;
    bool  omit_frontend_merges = false;
    // 2.0 matches the small layout's transcribed default (see
    // omnivoice_decode_loop_test.cpp's file header): non-zero, so the
    // unconditional branch runs by default, matching every existing caller.
    // Override to 0.0f to exercise the loop's `guidance == 0` branch, which
    // skips the unconditional forward entirely.
    float guidance_scale       = 2.0f;
    // Constant fill (the default, false) is what makes every candidate's
    // guided log-prob tie across the ENTIRE vocabulary at every position --
    // deliberate elsewhere in this file (check_grid's argmax-0 property), but
    // it also means no seed or draw sequence can ever show up in a committed
    // grid, so a test that needs to prove run_synthesis actually READS
    // request.seed (rather than, say, hardcoding a constant) cannot use the
    // default package. Set this to true to fill every generator/codec weight
    // tensor with a fixed, non-constant, reproducible LCG walk instead, so
    // logits genuinely vary across positions and vocabulary entries.
    bool  varied_weights       = false;
};

// A tiny deterministic LCG -- not synth::NormalRandomStream, which is
// production code belonging to a synthesis' own draw contract, not to a test
// fixture's weights. Only used when SyntheticPackageOptions::varied_weights
// is set. The output is scaled to a modest range: two transformer layers of
// wide, unscaled random weights would risk overflow inside the attention
// softmax, and nothing here needs realism, only variety.
inline float lcg_next_weight(uint64_t & state) {
    state                      = state * 6364136223846793005ULL + 1442695040888963407ULL;
    const uint32_t bits        = uint32_t(state >> 32);
    const float    unit_signed = float(bits) / float(0xFFFFFFFFu) * 2.0f - 1.0f;  // [-1, 1]
    return unit_signed * 0.1f;                                                    // [-0.1, 0.1]
}

inline void set_string_array(gguf_context * g, const char * key, const std::vector<std::string> & values) {
    std::vector<const char *> pointers;
    pointers.reserve(values.size());
    for (const std::string & value : values) {
        pointers.push_back(value.c_str());
    }
    gguf_set_arr_str(g, key, pointers.data(), int(pointers.size()));
}

inline void set_i32_array(gguf_context * g, const char * key, const std::vector<int32_t> & values) {
    gguf_set_arr_data(g, key, GGUF_TYPE_INT32, values.data(), values.size());
}

inline bool write_synthetic_package(const std::string & path, const SyntheticPackageOptions & options) {
    const synth::omnivoice::HParams h       = small_hparams();
    const std::vector<Entry>        entries = expected_entries(h);

    // Enough for every small tensor's data plus headers; the largest entry is
    // the [8, 40] text embedding at reduced widths.
    ggml_init_params parameters{};
    parameters.mem_size    = 8u * 1024 * 1024;
    parameters.no_alloc    = false;
    ggml_context * context = ggml_init(parameters);
    if (context == nullptr) {
        return false;
    }
    gguf_context * gguf = gguf_init_empty();
    if (gguf == nullptr) {
        ggml_free(context);
        return false;
    }

    // --- Metadata: every key read_hparams demands, at the small layout's
    // values. Kept in one place so the builder and small_hparams cannot drift:
    // the values below are small_hparams()'s, transcribed.
    gguf_set_val_str(gguf, "general.architecture", "omnivoice");
    gguf_set_val_u32(gguf, "synthesize.format_version", 1);
    gguf_set_val_str(gguf, "synthesize.model_family", "omnivoice");
    gguf_set_val_str(gguf, "synthesize.model_variant", "synthetic");
    gguf_set_val_str(gguf, "synthesize.quantization.profile", "F32");
    gguf_set_val_u32(gguf, "synthesize.quantization.profile_version", 1);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.architecture_version", 1);

    gguf_set_val_u32(gguf, "synthesize.capabilities.input_flags", 1u << 0);
    gguf_set_val_u32(gguf, "synthesize.capabilities.flags", (1u << 0) | (1u << 1));
    gguf_set_val_u64(gguf, "synthesize.capabilities.max_input_tokens", 64);
    gguf_set_val_u64(gguf, "synthesize.capabilities.max_output_frames", 16);
    gguf_set_val_f32(gguf, "synthesize.capabilities.min_speaking_rate", 0.5f);
    gguf_set_val_f32(gguf, "synthesize.capabilities.max_speaking_rate", 2.0f);
    gguf_set_val_u32(gguf, "synthesize.audio.sample_rate_hz", 150);
    gguf_set_val_u32(gguf, "synthesize.audio.channels", 1);
    gguf_set_val_str(gguf, "synthesize.audio.sample_format", "f32le");
    gguf_set_val_str(gguf, "synthesize.voice.mode", "profile-sources");
    gguf_set_val_bool(gguf, "synthesize.voice.has_package_default", true);
    gguf_set_val_u32(gguf, "synthesize.voice.preset_count", 0);

    gguf_set_val_u32(gguf, "synthesize.omnivoice.generation.num_step", 4);
    gguf_set_val_f32(gguf, "synthesize.omnivoice.generation.guidance_scale", options.guidance_scale);
    gguf_set_val_f32(gguf, "synthesize.omnivoice.generation.t_shift", 0.1f);
    gguf_set_val_f32(gguf, "synthesize.omnivoice.generation.layer_penalty_factor", 5.0f);
    gguf_set_val_f32(gguf, "synthesize.omnivoice.generation.position_temperature", 5.0f);
    gguf_set_val_f32(gguf, "synthesize.omnivoice.generation.class_temperature", 0.0f);

    gguf_set_val_u32(gguf, "synthesize.omnivoice.generator.layer_count", h.generator.layer_count);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.generator.hidden_size", h.generator.hidden_size);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.generator.attention_head_count", h.generator.attention_head_count);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.generator.key_value_head_count", h.generator.key_value_head_count);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.generator.head_dim", h.generator.head_dim);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.generator.intermediate_size", h.generator.intermediate_size);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.generator.text_vocab_size", h.generator.text_vocab_size);
    gguf_set_val_f32(gguf, "synthesize.omnivoice.generator.rms_norm_eps", h.generator.rms_norm_eps);
    gguf_set_val_f32(gguf, "synthesize.omnivoice.generator.rope_theta", h.generator.rope_theta);
    gguf_set_val_str(gguf, "synthesize.omnivoice.generator.attention", "bidirectional");

    gguf_set_val_u32(gguf, "synthesize.omnivoice.audio.num_codebooks", h.audio.num_codebooks);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.audio.vocab_size", h.audio.vocab_size);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.audio.mask_id", h.audio.mask_id);

    gguf_set_val_u32(gguf, "synthesize.omnivoice.codec.sample_rate", h.codec.sample_rate);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.codec.hop_length", h.codec.hop_length);
    gguf_set_val_f32(gguf, "synthesize.omnivoice.codec.frame_rate_hz", h.codec.frame_rate_hz);
    set_i32_array(gguf, "synthesize.omnivoice.codec.upsampling_ratios",
                  std::vector<int32_t>(h.codec.upsampling_ratios.begin(), h.codec.upsampling_ratios.end()));
    gguf_set_val_u32(gguf, "synthesize.omnivoice.codec.decoder_hidden_size", h.codec.decoder_hidden_size);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.codec.encoder_hidden_size", h.codec.encoder_hidden_size);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.codec.hidden_size", h.codec.hidden_size);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.codec.codebook_dim", h.codec.codebook_dim);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.codec.codebook_size", h.codec.codebook_size);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.codec.semantic_sample_rate", h.codec.semantic_sample_rate);

    gguf_set_val_u32(gguf, "synthesize.omnivoice.semantic.hidden_size", h.semantic.hidden_size);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.semantic.layer_count", h.semantic.layer_count);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.semantic.attention_head_count", h.semantic.attention_head_count);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.semantic.intermediate_size", h.semantic.intermediate_size);
    gguf_set_val_f32(gguf, "synthesize.omnivoice.semantic.layer_norm_eps", h.semantic.layer_norm_eps);
    set_i32_array(gguf, "synthesize.omnivoice.semantic.conv_dim",
                  std::vector<int32_t>(h.semantic.conv_dim.begin(), h.semantic.conv_dim.end()));
    set_i32_array(gguf, "synthesize.omnivoice.semantic.conv_kernel",
                  std::vector<int32_t>(h.semantic.conv_kernel.begin(), h.semantic.conv_kernel.end()));
    set_i32_array(gguf, "synthesize.omnivoice.semantic.conv_stride",
                  std::vector<int32_t>(h.semantic.conv_stride.begin(), h.semantic.conv_stride.end()));

    // Seven markers, eos and pad, all below the small text vocabulary (40).
    gguf_set_val_u32(gguf, "synthesize.omnivoice.token.denoise", 30);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.token.lang_start", 31);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.token.lang_end", 32);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.token.instruct_start", 33);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.token.instruct_end", 34);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.token.text_start", 35);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.token.text_end", 36);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.token.eos", 37);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.token.pad", 38);

    set_string_array(gguf, "synthesize.omnivoice.languages.tags", { "en", "zh", "ja" });

    gguf_set_val_str(gguf, "synthesize.profile.schema", "omnivoice-clone-prompt");
    gguf_set_val_u32(gguf, "synthesize.profile.schema_version", 1);
    gguf_set_val_str(gguf, "synthesize.profile.compatibility_id",
                     "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");
    gguf_set_val_u32(gguf, "synthesize.reference.target_sample_rate", 150);
    gguf_set_val_u32(gguf, "synthesize.reference.target_channels", 1);
    gguf_set_val_u64(gguf, "synthesize.reference.min_frames_per_clip", 150);
    gguf_set_val_u64(gguf, "synthesize.reference.max_frames_per_clip", 3000);
    gguf_set_val_u64(gguf, "synthesize.reference.max_total_frames", 3000);
    gguf_set_val_u64(gguf, "synthesize.reference.max_reference_count", 1);

    gguf_set_val_bool(gguf, "synthesize.frontend.present", true);
    gguf_set_val_str(gguf, "synthesize.frontend.provider", "synthesize.qwen_bpe");
    gguf_set_val_u32(gguf, "synthesize.frontend.contract_version", 1);
    if (!options.omit_frontend_vocab) {
        // make_bpe_frontend requires only a non-empty dense table; loading does
        // not tokenize, so token text is arbitrary here.
        std::vector<std::string> vocab;
        for (uint32_t index = 0; index < h.generator.text_vocab_size; ++index) {
            vocab.push_back("tok" + std::to_string(index));
        }
        set_string_array(gguf, "synthesize.omnivoice.frontend.vocab", vocab);
    }
    if (!options.omit_frontend_merges) {
        set_string_array(gguf, "synthesize.omnivoice.frontend.merges", { "tok1 tok2" });
    }

    // --- Tensors: the small layout with real F32 payloads. One LCG state
    // walks across every tensor's every element (rather than resetting per
    // tensor), so varied_weights produces a genuinely varied pattern rather
    // than the same short repeating sequence in every tensor.
    uint64_t lcg_state = 0x2545f4914f6cdd1dULL;  // an arbitrary fixed seed
    for (const Entry & entry : entries) {
        ggml_tensor * tensor = ggml_new_tensor(context, GGML_TYPE_F32, int(entry.ne.size()), entry.ne.data());
        if (tensor == nullptr) {
            gguf_free(gguf);
            ggml_free(context);
            return false;
        }
        ggml_set_name(tensor, entry.name.c_str());
        float * data = static_cast<float *>(tensor->data);
        for (int64_t index = 0; index < ggml_nelements(tensor); ++index) {
            // any finite constant; loading never computes -- unless
            // varied_weights asks for a real seed effect to be observable.
            data[index] = options.varied_weights ? lcg_next_weight(lcg_state) : 0.03125f;
        }
        gguf_add_tensor(gguf, tensor);
    }

    const bool written = gguf_write_to_file(gguf, path.c_str(), /*only_meta=*/false);
    gguf_free(gguf);
    ggml_free(context);
    return written;
}

}  // namespace synth::omnivoice::testing
