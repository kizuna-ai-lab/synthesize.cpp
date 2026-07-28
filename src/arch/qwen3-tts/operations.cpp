#include "operations.h"

#include "ggml.h"

#include <cmath>
#include <initializer_list>

namespace synth::qwen3tts {

namespace {

// ggml's rope reads positions from an I32 vector and expects
// [head_dim, head_count, positions]. NEOX mode rotates the two halves of each
// head against each other, which is what Qwen3 does; the interleaved GPT-J
// layout would pair adjacent elements instead and quietly change every angle.
constexpr int kRopeModeNeox = GGML_ROPE_TYPE_NEOX;

}  // namespace

ggml_tensor * rms_norm(ggml_context * context, ggml_tensor * input, ggml_tensor * weight, float eps) {
    if (context == nullptr || input == nullptr || weight == nullptr) {
        return nullptr;
    }
    return ggml_mul(context, ggml_rms_norm(context, input, eps), weight);
}

ggml_tensor * swiglu_mlp(ggml_context * context, ggml_tensor * input, const DecoderLayerWeights & weights) {
    if (context == nullptr || input == nullptr || weights.gate_proj == nullptr || weights.up_proj == nullptr ||
        weights.down_proj == nullptr) {
        return nullptr;
    }
    ggml_tensor * gate = ggml_silu(context, ggml_mul_mat(context, weights.gate_proj, input));
    ggml_tensor * up   = ggml_mul_mat(context, weights.up_proj, input);
    return ggml_mul_mat(context, weights.down_proj, ggml_mul(context, gate, up));
}

ggml_tensor * decoder_layer(ggml_context *              context,
                            ggml_cgraph *               graph,
                            ggml_tensor *               input,
                            ggml_tensor *               position_ids,
                            ggml_tensor *               mask,
                            const DecoderLayerWeights & weights,
                            const AttentionShape &      shape,
                            const KvCache &             cache) {
    if (context == nullptr || graph == nullptr || input == nullptr || position_ids == nullptr || cache.k == nullptr ||
        cache.v == nullptr || weights.input_layernorm == nullptr || weights.q_proj == nullptr ||
        weights.k_proj == nullptr || weights.v_proj == nullptr || weights.o_proj == nullptr ||
        weights.q_norm == nullptr || weights.k_norm == nullptr || weights.post_attention_layernorm == nullptr ||
        weights.gate_proj == nullptr || weights.up_proj == nullptr || weights.down_proj == nullptr) {
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
    const int64_t total = cache.filled + positions;
    for (const ggml_tensor * slot : { cache.k, cache.v }) {
        if (slot->type != GGML_TYPE_F32 || slot->ne[0] != head_dim || slot->ne[1] != kv_heads || slot->ne[2] < total ||
            slot->ne[3] != 1) {
            return nullptr;
        }
    }
    if (cache.filled < 0) {
        return nullptr;
    }
    if (mask != nullptr && (mask->type != GGML_TYPE_F32 || mask->ne[0] != total || mask->ne[1] < positions)) {
        return nullptr;
    }

    ggml_tensor * residual = input;
    ggml_tensor * normed   = rms_norm(context, input, weights.input_layernorm, shape.rms_norm_eps);

    // Project, then split into heads. Qwen3 normalizes each head of q and k
    // before rope; doing it after would rotate an unnormalized vector.
    ggml_tensor * q = ggml_mul_mat(context, weights.q_proj, normed);
    ggml_tensor * k = ggml_mul_mat(context, weights.k_proj, normed);
    ggml_tensor * v = ggml_mul_mat(context, weights.v_proj, normed);

    q = ggml_reshape_3d(context, q, head_dim, q_heads, positions);
    k = ggml_reshape_3d(context, k, head_dim, kv_heads, positions);
    v = ggml_reshape_3d(context, v, head_dim, kv_heads, positions);

    q = rms_norm(context, q, weights.q_norm, shape.rms_norm_eps);
    k = rms_norm(context, k, weights.k_norm, shape.rms_norm_eps);

    q = ggml_rope_ext(context, q, position_ids, nullptr, static_cast<int>(head_dim), kRopeModeNeox, 0, shape.rope_theta,
                      1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(context, k, position_ids, nullptr, static_cast<int>(head_dim), kRopeModeNeox, 0, shape.rope_theta,
                      1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // Write this step's keys and values into the cache, then attend over the
    // whole filled prefix. Expanding the copies here, before the reads below are
    // expanded, is what puts them earlier in the graph's execution order.
    ggml_build_forward_expand(
        graph, ggml_cpy(context, k,
                        ggml_view_3d(context, cache.k, head_dim, kv_heads, positions, cache.k->nb[1], cache.k->nb[2],
                                     cache.filled * cache.k->nb[2])));
    ggml_build_forward_expand(
        graph, ggml_cpy(context, v,
                        ggml_view_3d(context, cache.v, head_dim, kv_heads, positions, cache.v->nb[1], cache.v->nb[2],
                                     cache.filled * cache.v->nb[2])));

    k = ggml_view_3d(context, cache.k, head_dim, kv_heads, total, cache.k->nb[1], cache.k->nb[2], 0);
    v = ggml_view_3d(context, cache.v, head_dim, kv_heads, total, cache.v->nb[1], cache.v->nb[2], 0);

    // [head_dim, heads, positions] -> [head_dim, positions, heads] so the
    // matrix multiply reduces over head_dim with the heads batched.
    ggml_tensor * q_hd = ggml_cont(context, ggml_permute(context, q, 0, 2, 1, 3));
    ggml_tensor * k_hd = ggml_cont(context, ggml_permute(context, k, 0, 2, 1, 3));
    ggml_tensor * v_hd = ggml_cont(context, ggml_permute(context, v, 0, 2, 1, 3));

    // Grouped-query attention comes out of ggml_mul_mat's own broadcast rather
    // than a materialized repeat of the kv side: with q batched over q_heads and
    // kv over kv_heads it reads kv head i/(q_heads/kv_heads) for query head i,
    // which is the blocked grouping the reference's repeat_interleave produces.
    ggml_tensor * scores = ggml_mul_mat(context, k_hd, q_hd);
    const float   scale  = 1.0f / std::sqrt(static_cast<float>(head_dim));
    scores               = ggml_soft_max_ext(context, scores, mask, scale, 0.0f);

    // [total, positions, heads] against [head_dim, total, heads].
    ggml_tensor * v_t      = ggml_cont(context, ggml_permute(context, v_hd, 1, 0, 2, 3));
    ggml_tensor * attended = ggml_mul_mat(context, v_t, scores);

    attended = ggml_cont(context, ggml_permute(context, attended, 0, 2, 1, 3));
    attended = ggml_reshape_2d(context, attended, head_dim * q_heads, positions);

    ggml_tensor * projected       = ggml_mul_mat(context, weights.o_proj, attended);
    ggml_tensor * after_attention = ggml_add(context, residual, projected);

    ggml_tensor * mlp_in = rms_norm(context, after_attention, weights.post_attention_layernorm, shape.rms_norm_eps);
    ggml_tensor * mlp    = swiglu_mlp(context, mlp_in, weights);
    return ggml_add(context, after_attention, mlp);
}

}  // namespace synth::qwen3tts
