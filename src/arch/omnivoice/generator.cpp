// Graph construction for the mask-predict generator.
//
// The block is the ordinary Qwen3 one -- per-head q/k norms, gated feed-forward
// -- run bidirectionally over the whole canvas. There is no KV cache and no
// causal mask anywhere in this family: every layer is full attention, so the
// cache-free codec_transformer_layer shape from qwen3-tts is the precedent, not
// its decoder_layer. See docs/porting/families/omnivoice.md, "Architecture".

#include "arch/omnivoice/generator.h"

#include "arch/omnivoice/catalog.h"
#include "ggml.h"

#include <cmath>

namespace synth::omnivoice {

ggml_tensor * rms_norm(ggml_context * context, ggml_tensor * input, ggml_tensor * weight, float eps) {
    if (context == nullptr || input == nullptr || weight == nullptr) {
        return nullptr;
    }
    return ggml_mul(context, ggml_rms_norm(context, input, eps), weight);
}

ggml_tensor * generator_layer(ggml_context *                context,
                              ggml_tensor *                 input,
                              ggml_tensor *                 position_ids,
                              ggml_tensor *                 mask,
                              const GeneratorLayerWeights & weights,
                              const AttentionShape &        shape) {
    if (context == nullptr || input == nullptr || position_ids == nullptr || weights.input_layernorm == nullptr ||
        weights.q_proj == nullptr || weights.k_proj == nullptr || weights.v_proj == nullptr ||
        weights.o_proj == nullptr || weights.q_norm == nullptr || weights.k_norm == nullptr ||
        weights.post_attention_layernorm == nullptr || weights.gate_proj == nullptr || weights.up_proj == nullptr ||
        weights.down_proj == nullptr) {
        return nullptr;
    }
    const int64_t hidden    = static_cast<int64_t>(shape.hidden_size);
    const int64_t head_dim  = static_cast<int64_t>(shape.head_dim);
    const int64_t q_heads   = static_cast<int64_t>(shape.attention_head_count);
    const int64_t kv_heads  = static_cast<int64_t>(shape.key_value_head_count);
    const int64_t positions = input->ne[1];
    if (hidden <= 0 || head_dim <= 0 || q_heads <= 0 || kv_heads <= 0 || positions <= 0 || input->ne[0] != hidden ||
        q_heads % kv_heads != 0) {
        return nullptr;
    }
    if (position_ids->type != GGML_TYPE_I32 || position_ids->ne[0] != positions) {
        return nullptr;
    }
    // Delta against codec_transformer_layer: the mask is OPTIONAL. Bidirectional
    // attention over one un-padded sequence needs none, and that is the only
    // mode synthesis uses; a non-null mask exists so a test can prove gating.
    if (mask != nullptr && (mask->type != GGML_TYPE_F32 || mask->ne[0] != positions || mask->ne[1] < positions)) {
        return nullptr;
    }

    ggml_tensor * residual = input;
    ggml_tensor * normed   = rms_norm(context, input, weights.input_layernorm, shape.rms_norm_eps);

    ggml_tensor * q = ggml_mul_mat(context, weights.q_proj, normed);
    ggml_tensor * k = ggml_mul_mat(context, weights.k_proj, normed);
    ggml_tensor * v = ggml_mul_mat(context, weights.v_proj, normed);

    q = ggml_reshape_3d(context, q, head_dim, q_heads, positions);
    k = ggml_reshape_3d(context, k, head_dim, kv_heads, positions);
    v = ggml_reshape_3d(context, v, head_dim, kv_heads, positions);

    // Delta against codec_transformer_layer: Qwen3 normalizes each head of q
    // and k before rope rather than the packed projection; doing it after
    // would rotate an unnormalized vector.
    q = rms_norm(context, q, weights.q_norm, shape.rms_norm_eps);
    k = rms_norm(context, k, weights.k_norm, shape.rms_norm_eps);

    q = ggml_rope_ext(context, q, position_ids, nullptr, static_cast<int>(head_dim), GGML_ROPE_TYPE_NEOX, 0,
                      shape.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(context, k, position_ids, nullptr, static_cast<int>(head_dim), GGML_ROPE_TYPE_NEOX, 0,
                      shape.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    ggml_tensor * q_hd = ggml_cont(context, ggml_permute(context, q, 0, 2, 1, 3));
    ggml_tensor * k_hd = ggml_cont(context, ggml_permute(context, k, 0, 2, 1, 3));
    ggml_tensor * v_hd = ggml_cont(context, ggml_permute(context, v, 0, 2, 1, 3));

    // Grouped-query attention comes out of ggml_mul_mat's own broadcast rather
    // than a materialized repeat of the kv side, exactly as in qwen3-tts.
    ggml_tensor * scores = ggml_mul_mat(context, k_hd, q_hd);
    scores = ggml_soft_max_ext(context, scores, mask, 1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f);

    ggml_tensor * v_t      = ggml_cont(context, ggml_permute(context, v_hd, 1, 0, 2, 3));
    ggml_tensor * attended = ggml_mul_mat(context, v_t, scores);
    attended               = ggml_cont(context, ggml_permute(context, attended, 0, 2, 1, 3));
    attended               = ggml_reshape_2d(context, attended, head_dim * q_heads, positions);

    // Delta against codec_transformer_layer: no layer scales -- a Qwen3 block
    // adds its branches back unscaled.
    ggml_tensor * projected       = ggml_mul_mat(context, weights.o_proj, attended);
    ggml_tensor * after_attention = ggml_add(context, residual, projected);

    ggml_tensor * mlp_in = rms_norm(context, after_attention, weights.post_attention_layernorm, shape.rms_norm_eps);
    ggml_tensor * gate   = ggml_silu(context, ggml_mul_mat(context, weights.gate_proj, mlp_in));
    ggml_tensor * up     = ggml_mul_mat(context, weights.up_proj, mlp_in);
    ggml_tensor * mlp    = ggml_mul_mat(context, weights.down_proj, ggml_mul(context, gate, up));
    return ggml_add(context, after_attention, mlp);
}

ggml_tensor * build_canvas_embedding(ggml_context *           context,
                                     const GeneratorWeights & weights,
                                     ggml_tensor *            text_ids,
                                     ggml_tensor *            audio_ids,
                                     uint32_t                 num_codebooks) {
    if (context == nullptr || weights.text_embedding == nullptr || weights.audio_embeddings == nullptr ||
        num_codebooks == 0 || (text_ids == nullptr && audio_ids == nullptr)) {
        return nullptr;
    }
    if (text_ids != nullptr && (text_ids->type != GGML_TYPE_I32 || text_ids->ne[1] != 1)) {
        return nullptr;
    }
    if (audio_ids != nullptr && (audio_ids->type != GGML_TYPE_I32 || audio_ids->ne[1] != int64_t(num_codebooks))) {
        return nullptr;
    }

    ggml_tensor * text = nullptr;
    if (text_ids != nullptr) {
        text = ggml_get_rows(context, weights.text_embedding, text_ids);
    }

    ggml_tensor * audio = nullptr;
    if (audio_ids != nullptr) {
        // Residual of nothing: the eight codebook embeddings are summed, one
        // get_rows per codebook over the host-shifted ids, mirroring
        // codec_quantizer_decode's level loop. Ascending order matters: float
        // addition is not associative and the reference sums 0..7.
        const int64_t count = audio_ids->ne[0];
        for (uint32_t codebook = 0; codebook < num_codebooks; ++codebook) {
            ggml_tensor * ids  = ggml_view_1d(context, audio_ids, count, size_t(codebook) * audio_ids->nb[1]);
            ggml_tensor * rows = ggml_get_rows(context, weights.audio_embeddings, ids);
            audio              = audio == nullptr ? rows : ggml_add(context, audio, rows);
        }
    }

    if (text == nullptr) {
        return audio;
    }
    if (audio == nullptr) {
        return text;
    }
    // The reference torch.where-selects per position; regions are contiguous,
    // so selection is concatenation along the sequence axis.
    return ggml_concat(context, text, audio, 1);
}

ggml_tensor * build_generator_forward(ggml_context *               context,
                                      ggml_tensor *                embeddings,
                                      ggml_tensor *                position_ids,
                                      ggml_tensor *                mask,
                                      const GeneratorWeights &     weights,
                                      const AttentionShape &       shape,
                                      const AudioCanvasParams &    canvas,
                                      std::vector<ggml_tensor *> * out_layers,
                                      ggml_tensor **               out_final) {
    if (context == nullptr || embeddings == nullptr || position_ids == nullptr || weights.norm == nullptr ||
        weights.audio_heads == nullptr || weights.layers.empty()) {
        return nullptr;
    }
    if (out_layers != nullptr) {
        out_layers->clear();
        out_layers->reserve(weights.layers.size());
    }
    if (out_final != nullptr) {
        *out_final = nullptr;
    }

    ggml_tensor * hidden = embeddings;
    for (const GeneratorLayerWeights & layer : weights.layers) {
        hidden = generator_layer(context, hidden, position_ids, mask, layer, shape);
        if (hidden == nullptr) {
            return nullptr;
        }
        if (out_layers != nullptr) {
            out_layers->push_back(hidden);
        }
    }
    hidden = rms_norm(context, hidden, weights.norm, shape.rms_norm_eps);
    if (hidden == nullptr) {
        return nullptr;
    }
    if (out_final != nullptr) {
        *out_final = hidden;
    }

    // Every position predicts: [codebooks * vocab, positions], then the split
    // that names the stacked row order c * vocab + v explicitly.
    ggml_tensor * logits = ggml_mul_mat(context, weights.audio_heads, hidden);
    if (logits->ne[0] != int64_t(canvas.num_codebooks) * canvas.vocab_size) {
        return nullptr;
    }
    return ggml_reshape_3d(context, logits, canvas.vocab_size, canvas.num_codebooks, logits->ne[1]);
}

}  // namespace synth::omnivoice
