// The bidirectional LSTM builder is compared against a host reference of
// PyTorch's LSTM semantics.
//
// This test exists because the accumulation strategy is a correctness hazard,
// not a performance detail: writing per-step outputs through allocator-managed
// in-place views produced correct CPU results and wrong CUDA results. It must
// therefore be run on every Execution Backend the package claims, not only on
// the CPU gate. See docs/porting/families/kokoro.md.

#include "arch/kokoro/operations.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "test-assert.h"

#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

namespace {

constexpr uint32_t kInputDim = 12;
constexpr uint32_t kHidden   = 8;

struct DirectionData {
    std::vector<float> weight_ih;  // [4H, D]
    std::vector<float> weight_hh;  // [4H, H]
    std::vector<float> bias_ih;    // [4H]
    std::vector<float> bias_hh;    // [4H]
};

DirectionData make_direction(std::mt19937 & rng) {
    std::normal_distribution<float> dist(0.0f, 0.4f);
    DirectionData                   data;
    data.weight_ih.resize(size_t(4 * kHidden) * kInputDim);
    data.weight_hh.resize(size_t(4 * kHidden) * kHidden);
    data.bias_ih.resize(4 * kHidden);
    data.bias_hh.resize(4 * kHidden);
    for (float & v : data.weight_ih) {
        v = dist(rng);
    }
    for (float & v : data.weight_hh) {
        v = dist(rng);
    }
    for (float & v : data.bias_ih) {
        v = dist(rng);
    }
    for (float & v : data.bias_hh) {
        v = dist(rng);
    }
    return data;
}

float sigmoidf(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// PyTorch LSTM: gates stacked input, forget, cell, output.
void reference_direction(const DirectionData &      data,
                         const std::vector<float> & input,
                         uint32_t                   length,
                         bool                       reverse,
                         std::vector<float> &       out) {
    const int H = int(kHidden);
    const int D = int(kInputDim);
    out.assign(size_t(H) * length, 0.0f);
    std::vector<float> state_h(H, 0.0f), state_c(H, 0.0f), pre(4 * H);
    for (uint32_t step = 0; step < length; ++step) {
        const uint32_t index = reverse ? length - 1 - step : step;
        const float *  x     = input.data() + size_t(index) * D;
        for (int row = 0; row < 4 * H; ++row) {
            float acc = data.bias_ih[row] + data.bias_hh[row];
            for (int k = 0; k < D; ++k) {
                acc += data.weight_ih[size_t(row) * D + k] * x[k];
            }
            for (int k = 0; k < H; ++k) {
                acc += data.weight_hh[size_t(row) * H + k] * state_h[k];
            }
            pre[row] = acc;
        }
        for (int k = 0; k < H; ++k) {
            const float i = sigmoidf(pre[k]);
            const float f = sigmoidf(pre[H + k]);
            const float g = std::tanh(pre[2 * H + k]);
            const float o = sigmoidf(pre[3 * H + k]);
            state_c[k]    = f * state_c[k] + i * g;
            state_h[k]    = o * std::tanh(state_c[k]);
        }
        std::memcpy(out.data() + size_t(index) * H, state_h.data(), size_t(H) * sizeof(float));
    }
}

struct ContextDeleter {
    void operator()(ggml_context * context) const {
        if (context != nullptr) {
            ggml_free(context);
        }
    }
};

using Context = std::unique_ptr<ggml_context, ContextDeleter>;

Context make_context(size_t bytes) {
    ggml_init_params parameters{};
    parameters.mem_size = bytes;
    parameters.no_alloc = true;
    return Context(ggml_init(parameters));
}

// Runs the builder for one sequence length and returns the largest deviation
// from the reference, plus the realized node count.
bool run_case(uint32_t length, float & max_diff, int & nodes) {
    std::mt19937                    rng(20260725u + length);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    const DirectionData             forward = make_direction(rng);
    const DirectionData             reverse = make_direction(rng);
    std::vector<float>              input(size_t(kInputDim) * length);
    for (float & v : input) {
        v = dist(rng);
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) {
        return false;
    }

    // Weights, the zero seed and both output stores live in a persistent buffer
    // so the graph allocator never owns what the recurrent chain writes.
    Context        persistent = make_context(ggml_tensor_overhead() * 32);
    ggml_context * pctx       = persistent.get();
    ggml_tensor *  t_input    = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kInputDim, length);
    ggml_tensor *  t_wih_f    = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kInputDim, 4 * kHidden);
    ggml_tensor *  t_whh_f    = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, 4 * kHidden);
    ggml_tensor *  t_bih_f    = ggml_new_tensor_1d(pctx, GGML_TYPE_F32, 4 * kHidden);
    ggml_tensor *  t_bhh_f    = ggml_new_tensor_1d(pctx, GGML_TYPE_F32, 4 * kHidden);
    ggml_tensor *  t_wih_r    = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kInputDim, 4 * kHidden);
    ggml_tensor *  t_whh_r    = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, 4 * kHidden);
    ggml_tensor *  t_bih_r    = ggml_new_tensor_1d(pctx, GGML_TYPE_F32, 4 * kHidden);
    ggml_tensor *  t_bhh_r    = ggml_new_tensor_1d(pctx, GGML_TYPE_F32, 4 * kHidden);
    ggml_tensor *  t_zero     = ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kHidden);
    ggml_tensor *  t_store_f  = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, length);
    ggml_tensor *  t_store_r  = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, length);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(pctx, backend);
    if (buffer == nullptr) {
        ggml_backend_free(backend);
        return false;
    }

    ggml_backend_tensor_set(t_input, input.data(), 0, ggml_nbytes(t_input));
    ggml_backend_tensor_set(t_wih_f, forward.weight_ih.data(), 0, ggml_nbytes(t_wih_f));
    ggml_backend_tensor_set(t_whh_f, forward.weight_hh.data(), 0, ggml_nbytes(t_whh_f));
    ggml_backend_tensor_set(t_bih_f, forward.bias_ih.data(), 0, ggml_nbytes(t_bih_f));
    ggml_backend_tensor_set(t_bhh_f, forward.bias_hh.data(), 0, ggml_nbytes(t_bhh_f));
    ggml_backend_tensor_set(t_wih_r, reverse.weight_ih.data(), 0, ggml_nbytes(t_wih_r));
    ggml_backend_tensor_set(t_whh_r, reverse.weight_hh.data(), 0, ggml_nbytes(t_whh_r));
    ggml_backend_tensor_set(t_bih_r, reverse.bias_ih.data(), 0, ggml_nbytes(t_bih_r));
    ggml_backend_tensor_set(t_bhh_r, reverse.bias_hh.data(), 0, ggml_nbytes(t_bhh_r));
    const std::vector<float> zeros(size_t(kHidden) * length, 0.0f);
    ggml_backend_tensor_set(t_zero, zeros.data(), 0, ggml_nbytes(t_zero));
    ggml_backend_tensor_set(t_store_f, zeros.data(), 0, ggml_nbytes(t_store_f));
    ggml_backend_tensor_set(t_store_r, zeros.data(), 0, ggml_nbytes(t_store_r));

    const uint64_t predicted = synth::kokoro::bidirectional_lstm_node_count(length);
    const size_t   budget    = size_t(predicted) + 64;
    Context        graph_ctx =
        make_context(ggml_tensor_overhead() * (budget + 64) + ggml_graph_overhead_custom(budget, false));
    ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx.get(), budget, false);

    synth::kokoro::LstmWeights weights;
    weights.input_dim         = kInputDim;
    weights.hidden            = kHidden;
    weights.forward.weight_ih = t_wih_f;
    weights.forward.weight_hh = t_whh_f;
    weights.forward.bias_ih   = t_bih_f;
    weights.forward.bias_hh   = t_bhh_f;
    weights.reverse.weight_ih = t_wih_r;
    weights.reverse.weight_hh = t_whh_r;
    weights.reverse.bias_ih   = t_bih_r;
    weights.reverse.bias_hh   = t_bhh_r;

    synth::kokoro::LstmScratch scratch;
    scratch.zero_state    = t_zero;
    scratch.forward_store = t_store_f;
    scratch.reverse_store = t_store_r;

    ggml_tensor * output =
        synth::kokoro::build_bidirectional_lstm(graph_ctx.get(), graph, t_input, weights, scratch, length);
    if (output == nullptr || output->ne[0] != 2 * int64_t(kHidden) || output->ne[1] != int64_t(length)) {
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        return false;
    }
    nodes = ggml_graph_n_nodes(graph);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_reserve(alloc, graph);
    ggml_gallocr_alloc_graph(alloc, graph);
    ggml_backend_cpu_set_n_threads(backend, 2);
    const bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;

    std::vector<float> got(size_t(2 * kHidden) * length);
    if (ok) {
        ggml_backend_tensor_get(output, got.data(), 0, ggml_nbytes(output));
    }

    std::vector<float> want_f, want_r;
    reference_direction(forward, input, length, false, want_f);
    reference_direction(reverse, input, length, true, want_r);

    max_diff = 0.0f;
    for (uint32_t t = 0; t < length && ok; ++t) {
        for (uint32_t k = 0; k < kHidden; ++k) {
            const float f = got[size_t(t) * 2 * kHidden + k];
            const float r = got[size_t(t) * 2 * kHidden + kHidden + k];
            max_diff      = std::max(max_diff, std::fabs(f - want_f[size_t(t) * kHidden + k]));
            max_diff      = std::max(max_diff, std::fabs(r - want_r[size_t(t) * kHidden + k]));
        }
    }

    ggml_gallocr_free(alloc);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    return ok;
}

}  // namespace

int main() {
    // A single step exercises the seed state; longer runs exercise the
    // recurrence and, for the reverse direction, the descending write order.
    for (uint32_t length : { 1u, 2u, 5u, 37u }) {
        float max_diff = 0.0f;
        int   nodes    = 0;
        SYNTH_TEST_CHECK(run_case(length, max_diff, nodes));
        SYNTH_TEST_CHECK(max_diff < 1e-5f);
        // The advertised node count must match the graph the builder emits, so
        // callers can size their graphs without guessing.
        SYNTH_TEST_CHECK(nodes == int(synth::kokoro::bidirectional_lstm_node_count(length)));
    }

    // Contract failures return null rather than building a partial graph.
    const size_t reject_budget = 512;
    Context      ctx =
        make_context(ggml_tensor_overhead() * (reject_budget + 64) + ggml_graph_overhead_custom(reject_budget, false));
    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), reject_budget, false);
    ggml_tensor * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, kInputDim, 4);

    synth::kokoro::LstmWeights weights;
    weights.input_dim         = kInputDim;
    weights.hidden            = kHidden;
    weights.forward.weight_ih = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, kInputDim, 4 * kHidden);
    weights.forward.weight_hh = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, kHidden, 4 * kHidden);
    weights.forward.bias_ih   = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4 * kHidden);
    weights.forward.bias_hh   = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4 * kHidden);
    weights.reverse           = weights.forward;

    synth::kokoro::LstmScratch scratch;
    scratch.zero_state    = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, kHidden);
    scratch.forward_store = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, kHidden, 4);
    scratch.reverse_store = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, kHidden, 4);

    SYNTH_TEST_CHECK(synth::kokoro::build_bidirectional_lstm(nullptr, graph, input, weights, scratch, 4) == nullptr);
    SYNTH_TEST_CHECK(synth::kokoro::build_bidirectional_lstm(ctx.get(), nullptr, input, weights, scratch, 4) ==
                     nullptr);
    SYNTH_TEST_CHECK(synth::kokoro::build_bidirectional_lstm(ctx.get(), graph, nullptr, weights, scratch, 4) ==
                     nullptr);
    SYNTH_TEST_CHECK(synth::kokoro::build_bidirectional_lstm(ctx.get(), graph, input, weights, scratch, 0) == nullptr);

    // Length must agree with the input and with both stores.
    SYNTH_TEST_CHECK(synth::kokoro::build_bidirectional_lstm(ctx.get(), graph, input, weights, scratch, 3) == nullptr);

    synth::kokoro::LstmScratch short_store = scratch;
    short_store.reverse_store              = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, kHidden, 3);
    SYNTH_TEST_CHECK(synth::kokoro::build_bidirectional_lstm(ctx.get(), graph, input, weights, short_store, 4) ==
                     nullptr);

    synth::kokoro::LstmScratch wrong_seed = scratch;
    wrong_seed.zero_state                 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, kHidden + 1);
    SYNTH_TEST_CHECK(synth::kokoro::build_bidirectional_lstm(ctx.get(), graph, input, weights, wrong_seed, 4) ==
                     nullptr);

    synth::kokoro::LstmScratch missing = scratch;
    missing.forward_store              = nullptr;
    SYNTH_TEST_CHECK(synth::kokoro::build_bidirectional_lstm(ctx.get(), graph, input, weights, missing, 4) == nullptr);

    synth::kokoro::LstmWeights incomplete = weights;
    incomplete.forward.weight_hh          = nullptr;
    SYNTH_TEST_CHECK(synth::kokoro::build_bidirectional_lstm(ctx.get(), graph, input, incomplete, scratch, 4) ==
                     nullptr);

    synth::kokoro::LstmWeights zero_dims = weights;
    zero_dims.hidden                     = 0;
    SYNTH_TEST_CHECK(synth::kokoro::build_bidirectional_lstm(ctx.get(), graph, input, zero_dims, scratch, 4) ==
                     nullptr);
    return 0;
}
