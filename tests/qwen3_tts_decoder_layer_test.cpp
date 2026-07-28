// The shared Qwen3 decoder block is compared against the reference
// implementation's own Qwen3TTSDecoderLayer.
//
// The talker and the code predictor are built from the same block, so a defect
// here is a defect in both. The reference values come from
// scripts/dump_reference_qwen3_tts_decoder_layer.py, which runs the real layer
// class over four positions with a causal mask. This test reproduces those four
// positions two ways: as a three-position prefill, and as a cached single-token
// step at position three. Both must land on the same reference, which is what
// makes the key/value cache and the causal mask checkable rather than assumed.
//
// Weights are not carried in either file. Both sides draw them from the same
// 64-bit LCG in the same order, so only the outputs are pinned.

#include "arch/qwen3-tts/operations.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "test-assert.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

namespace {

constexpr uint32_t kHidden       = 8;
constexpr uint32_t kHeads        = 4;
constexpr uint32_t kKvHeads      = 2;
constexpr uint32_t kHeadDim      = 4;
constexpr uint32_t kIntermediate = 16;
constexpr uint32_t kPositions    = 4;
constexpr uint32_t kPrefill      = 3;
constexpr float    kRmsNormEps   = 1e-6f;
constexpr float    kRopeTheta    = 1000000.0f;
constexpr uint64_t kSeed         = 20260728u;

// 4 positions x 8 hidden, position-major.
constexpr float kExpectedHidden[] = {
    -0.143100798f, 1.05902505f,  1.13132095f,  2.02069712f,  -1.90775514f,  -0.991743386f, -1.49144959f,  -1.66155756f,
    0.469848841f,  0.263561219f, 1.36933374f,  2.16476631f,  -1.00003767f,  -0.356122792f, -1.26636958f,  -0.329140633f,
    0.64148581f,   0.391249448f, 0.120782375f, 0.221909165f, -0.311113715f, -0.196477801f, 0.0848978162f, -0.968586683f,
    0.42898649f,   1.1952492f,   0.931253195f, 1.30417335f,  -1.0430665f,   -0.94602108f,  -0.610675633f, -1.12255645f,
};

// The twin of LcgStream in the reference script. Every value is a 24-bit
// numerator over 2^23, so both sides start from bit-identical inputs.
class LcgStream {
  public:
    explicit LcgStream(uint64_t seed) : state_(seed) {}

    float next() {
        state_ = state_ * 6364136223846793005ull + 1442695040888963407ull;
        return float(state_ >> 40) / 8388608.0f - 1.0f;
    }

    std::vector<float> fill(size_t count, float scale, float offset) {
        std::vector<float> values(count);
        for (float & value : values) {
            value = next() * scale + offset;
        }
        return values;
    }

  private:
    uint64_t state_;
};

struct ContextDeleter {
    void operator()(ggml_context * context) const { ggml_free(context); }
};

using Context = std::unique_ptr<ggml_context, ContextDeleter>;

Context make_context(size_t bytes, bool no_alloc) {
    ggml_init_params parameters = {};
    parameters.mem_size         = bytes;
    parameters.no_alloc         = no_alloc;
    return Context(ggml_init(parameters));
}

synth::qwen3tts::AttentionShape make_shape() {
    synth::qwen3tts::AttentionShape shape;
    shape.hidden_size          = kHidden;
    shape.attention_head_count = kHeads;
    shape.key_value_head_count = kKvHeads;
    shape.head_dim             = kHeadDim;
    shape.rms_norm_eps         = kRmsNormEps;
    shape.rope_theta           = kRopeTheta;
    return shape;
}

// Runs both the prefill and the cached step on one device and returns the
// largest deviation from the reference across all four positions.
bool run_case(ggml_backend_dev_t device, float & max_diff) {
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (backend == nullptr) {
        return false;
    }

    Context        persistent = make_context(ggml_tensor_overhead() * 32, true);
    ggml_context * pctx       = persistent.get();

    ggml_tensor * t_input_norm = ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kHidden);
    ggml_tensor * t_q_proj     = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kHeads * kHeadDim);
    ggml_tensor * t_k_proj     = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kKvHeads * kHeadDim);
    ggml_tensor * t_v_proj     = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kKvHeads * kHeadDim);
    ggml_tensor * t_o_proj     = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHeads * kHeadDim, kHidden);
    ggml_tensor * t_q_norm     = ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kHeadDim);
    ggml_tensor * t_k_norm     = ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kHeadDim);
    ggml_tensor * t_post_norm  = ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kHidden);
    ggml_tensor * t_gate_proj  = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kIntermediate);
    ggml_tensor * t_up_proj    = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kIntermediate);
    ggml_tensor * t_down_proj  = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kIntermediate, kHidden);

    ggml_tensor * t_prefill      = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kPrefill);
    ggml_tensor * t_step         = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, 1);
    ggml_tensor * t_prefill_pos  = ggml_new_tensor_1d(pctx, GGML_TYPE_I32, kPrefill);
    ggml_tensor * t_step_pos     = ggml_new_tensor_1d(pctx, GGML_TYPE_I32, 1);
    ggml_tensor * t_prefill_mask = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kPrefill, kPrefill);

    // The cache outlives both graphs, so it is allocated here rather than by the
    // graph allocator, and is sized for every position the frame will hold.
    ggml_tensor * t_cache_k = ggml_new_tensor_3d(pctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kPositions);
    ggml_tensor * t_cache_v = ggml_new_tensor_3d(pctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kPositions);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(pctx, backend);
    if (buffer == nullptr) {
        ggml_backend_free(backend);
        return false;
    }

    LcgStream stream(kSeed);

    struct Assignment {
        ggml_tensor * tensor;
        float         scale;
        float         offset;
    };

    // The order and the scales are the reference script's, not a convention:
    // reordering either line silently changes every weight.
    const Assignment assignments[] = {
        { t_input_norm, 0.25f, 1.0f },
        { t_q_proj,     0.5f,  0.0f },
        { t_k_proj,     0.5f,  0.0f },
        { t_v_proj,     0.5f,  0.0f },
        { t_o_proj,     0.5f,  0.0f },
        { t_q_norm,     0.25f, 1.0f },
        { t_k_norm,     0.25f, 1.0f },
        { t_post_norm,  0.25f, 1.0f },
        { t_gate_proj,  0.5f,  0.0f },
        { t_up_proj,    0.5f,  0.0f },
        { t_down_proj,  0.5f,  0.0f },
    };
    for (const Assignment & assignment : assignments) {
        const std::vector<float> values =
            stream.fill(size_t(ggml_nelements(assignment.tensor)), assignment.scale, assignment.offset);
        ggml_backend_tensor_set(assignment.tensor, values.data(), 0, ggml_nbytes(assignment.tensor));
    }

    const std::vector<float> hidden = stream.fill(size_t(kPositions) * kHidden, 0.5f, 0.0f);
    ggml_backend_tensor_set(t_prefill, hidden.data(), 0, ggml_nbytes(t_prefill));
    ggml_backend_tensor_set(t_step, hidden.data() + size_t(kPrefill) * kHidden, 0, ggml_nbytes(t_step));

    const int32_t prefill_positions[kPrefill] = { 0, 1, 2 };
    const int32_t step_position[1]            = { int32_t(kPrefill) };
    ggml_backend_tensor_set(t_prefill_pos, prefill_positions, 0, ggml_nbytes(t_prefill_pos));
    ggml_backend_tensor_set(t_step_pos, step_position, 0, ggml_nbytes(t_step_pos));

    // Additive causal mask: query q may read key k only when k <= q.
    std::vector<float> mask(size_t(kPrefill) * kPrefill, 0.0f);
    for (uint32_t query = 0; query < kPrefill; ++query) {
        for (uint32_t key = 0; key < kPrefill; ++key) {
            mask[size_t(query) * kPrefill + key] = key <= query ? 0.0f : -std::numeric_limits<float>::infinity();
        }
    }
    ggml_backend_tensor_set(t_prefill_mask, mask.data(), 0, ggml_nbytes(t_prefill_mask));

    constexpr size_t kNodeBudget = 512;
    Context          graph_ctx   = make_context(
        ggml_tensor_overhead() * (kNodeBudget + 64) + ggml_graph_overhead_custom(kNodeBudget, false), true);
    ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx.get(), kNodeBudget, false);

    synth::qwen3tts::DecoderLayerWeights weights;
    weights.input_layernorm          = t_input_norm;
    weights.q_proj                   = t_q_proj;
    weights.k_proj                   = t_k_proj;
    weights.v_proj                   = t_v_proj;
    weights.o_proj                   = t_o_proj;
    weights.q_norm                   = t_q_norm;
    weights.k_norm                   = t_k_norm;
    weights.post_attention_layernorm = t_post_norm;
    weights.gate_proj                = t_gate_proj;
    weights.up_proj                  = t_up_proj;
    weights.down_proj                = t_down_proj;

    const synth::qwen3tts::AttentionShape shape = make_shape();

    synth::qwen3tts::KvCache cache;
    cache.k      = t_cache_k;
    cache.v      = t_cache_v;
    cache.filled = 0;

    ggml_tensor * prefill_out = synth::qwen3tts::decoder_layer(graph_ctx.get(), graph, t_prefill, t_prefill_pos,
                                                               t_prefill_mask, weights, shape, cache);

    // The cached step passes no mask: a single query attending to its whole
    // cache is causal by construction.
    cache.filled = kPrefill;
    ggml_tensor * step_out =
        synth::qwen3tts::decoder_layer(graph_ctx.get(), graph, t_step, t_step_pos, nullptr, weights, shape, cache);

    if (prefill_out == nullptr || step_out == nullptr || prefill_out->ne[0] != int64_t(kHidden) ||
        prefill_out->ne[1] != int64_t(kPrefill) || step_out->ne[1] != 1) {
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        return false;
    }

    ggml_build_forward_expand(graph, prefill_out);
    ggml_build_forward_expand(graph, step_out);

    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    bool           ok        = allocator != nullptr && ggml_gallocr_alloc_graph(allocator, graph);
    if (ok) {
        ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    }

    if (ok) {
        std::vector<float> prefill_values(size_t(kPrefill) * kHidden);
        std::vector<float> step_values(kHidden);
        ggml_backend_tensor_get(prefill_out, prefill_values.data(), 0, ggml_nbytes(prefill_out));
        ggml_backend_tensor_get(step_out, step_values.data(), 0, ggml_nbytes(step_out));

        max_diff = 0.0f;
        for (size_t index = 0; index < prefill_values.size(); ++index) {
            max_diff = std::fmax(max_diff, std::fabs(prefill_values[index] - kExpectedHidden[index]));
        }
        for (size_t index = 0; index < step_values.size(); ++index) {
            const float expected = kExpectedHidden[size_t(kPrefill) * kHidden + index];
            max_diff             = std::fmax(max_diff, std::fabs(step_values[index] - expected));
        }
    }

    if (allocator != nullptr) {
        ggml_gallocr_free(allocator);
    }
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    return ok;
}

// A shape the block cannot build is a wiring defect, so it returns nullptr
// rather than aborting inside ggml on an assertion the caller cannot catch.
int check_rejections() {
    Context        context = make_context(ggml_tensor_overhead() * 64 + ggml_graph_overhead() + 4096, true);
    ggml_context * ctx     = context.get();

    synth::qwen3tts::DecoderLayerWeights weights;
    weights.input_layernorm          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHidden);
    weights.q_proj                   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, kHeads * kHeadDim);
    weights.k_proj                   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, kKvHeads * kHeadDim);
    weights.v_proj                   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, kKvHeads * kHeadDim);
    weights.o_proj                   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHeads * kHeadDim, kHidden);
    weights.q_norm                   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHeadDim);
    weights.k_norm                   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHeadDim);
    weights.post_attention_layernorm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHidden);
    weights.gate_proj                = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, kIntermediate);
    weights.up_proj                  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, kIntermediate);
    weights.down_proj                = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kIntermediate, kHidden);

    ggml_tensor * input     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, kPrefill);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kPrefill);
    ggml_tensor * mask      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kPrefill, kPrefill);

    const synth::qwen3tts::AttentionShape shape = make_shape();

    synth::qwen3tts::KvCache cache;
    cache.k      = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kPositions);
    cache.v      = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kPositions);
    cache.filled = 0;

    // A graph the rejected calls must be left out of entirely: a builder that
    // validated after expanding its cache writes would leave nodes behind.
    ggml_cgraph * graph = ggml_new_graph(ctx);

    // A missing per-head norm is the defect this family invites: Qwen2 has no
    // q_norm, so a block copied from a Qwen2 port would leave it null.
    synth::qwen3tts::DecoderLayerWeights no_q_norm = weights;
    no_q_norm.q_norm                               = nullptr;
    SYNTH_TEST_CHECK(synth::qwen3tts::decoder_layer(ctx, graph, input, positions, mask, no_q_norm, shape, cache) ==
                     nullptr);

    // Positions must be I32; an F32 vector would be read as garbage indices.
    ggml_tensor * float_positions = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kPrefill);
    SYNTH_TEST_CHECK(synth::qwen3tts::decoder_layer(ctx, graph, input, float_positions, mask, weights, shape, cache) ==
                     nullptr);

    // One position id per token, no more and no fewer.
    ggml_tensor * short_positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kPrefill - 1);
    SYNTH_TEST_CHECK(synth::qwen3tts::decoder_layer(ctx, graph, input, short_positions, mask, weights, shape, cache) ==
                     nullptr);

    // A mask narrower than the key count would be applied to the wrong keys.
    ggml_tensor * short_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kPrefill - 1, kPrefill);
    SYNTH_TEST_CHECK(synth::qwen3tts::decoder_layer(ctx, graph, input, positions, short_mask, weights, shape, cache) ==
                     nullptr);

    // Grouped attention needs the head counts to divide.
    synth::qwen3tts::AttentionShape ungrouped = shape;
    ungrouped.key_value_head_count            = 3;
    SYNTH_TEST_CHECK(synth::qwen3tts::decoder_layer(ctx, graph, input, positions, mask, weights, ungrouped, cache) ==
                     nullptr);

    // An input whose width disagrees with the declared hidden size.
    ggml_tensor * wide_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden + 1, kPrefill);
    SYNTH_TEST_CHECK(synth::qwen3tts::decoder_layer(ctx, graph, wide_input, positions, mask, weights, shape, cache) ==
                     nullptr);

    // A cache too short for the positions this call would write.
    synth::qwen3tts::KvCache overflowing = cache;
    overflowing.filled                   = kPositions - 1;
    SYNTH_TEST_CHECK(
        synth::qwen3tts::decoder_layer(ctx, graph, input, positions, nullptr, weights, shape, overflowing) == nullptr);

    // No cache at all: the block has nowhere to put the keys it just projected.
    synth::qwen3tts::KvCache absent;
    SYNTH_TEST_CHECK(synth::qwen3tts::decoder_layer(ctx, graph, input, positions, mask, weights, shape, absent) ==
                     nullptr);

    // Not one rejected call may have touched the graph.
    SYNTH_TEST_CHECK(ggml_graph_n_nodes(graph) == 0);
    return 0;
}

}  // namespace

int main() {
    SYNTH_TEST_CHECK(check_rejections() == 0);

    // Every registered device. The block is the family's hot path on both the
    // talker and the predictor, so a backend that disagrees is worth catching
    // here rather than in a whole-model comparison.
    const size_t device_count = ggml_backend_dev_count();
    SYNTH_TEST_CHECK(device_count > 0);

    size_t exercised = 0;
    for (size_t index = 0; index < device_count; ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (device == nullptr) {
            continue;
        }
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(device);
        if (type != GGML_BACKEND_DEVICE_TYPE_CPU && type != GGML_BACKEND_DEVICE_TYPE_GPU &&
            type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }

        float max_diff = 0.0f;
        SYNTH_TEST_CHECK(run_case(device, max_diff));

        // Accelerators run F32 matmuls through tensor cores at reduced mantissa
        // width, so they are held to a looser bound than the CPU.
        const float tolerance = type == GGML_BACKEND_DEVICE_TYPE_CPU ? 1e-4f : 5e-3f;
        std::printf("qwen3-tts-decoder-layer: %s (device type %d) max_diff %.3g\n", ggml_backend_dev_name(device),
                    int(type), double(max_diff));
        SYNTH_TEST_CHECK(max_diff < tolerance);
        ++exercised;
    }
    SYNTH_TEST_CHECK(exercised > 0);
    return 0;
}
