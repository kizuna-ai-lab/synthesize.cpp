// The duration path: the DurationEncoder graph against a host reference, and
// the host seam that turns its logits into a concrete alignment.

#include "arch/kokoro/duration-host.h"
#include "arch/kokoro/duration.h"
#include "arch/kokoro/weights.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "test-assert.h"

#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kTokens   = 4;
constexpr uint32_t kHidden   = 8;
constexpr uint32_t kStyleDim = 4;
constexpr uint32_t kLayers   = 2;
constexpr uint32_t kMaxDur   = 5;
constexpr float    kAdaEps   = 1e-5f;

struct ContextDeleter {
    void operator()(ggml_context * context) const {
        if (context != nullptr) {
            ggml_free(context);
        }
    }
};

using Context = std::unique_ptr<ggml_context, ContextDeleter>;

Context make_context(size_t tensors, size_t graph_nodes = 0) {
    ggml_init_params parameters{};
    parameters.mem_size =
        ggml_tensor_overhead() * tensors + (graph_nodes > 0 ? ggml_graph_overhead_custom(graph_nodes, false) : 0);
    parameters.no_alloc = true;
    return Context(ggml_init(parameters));
}

synth::kokoro::HParams hparams() {
    synth::kokoro::HParams h;
    h.hidden_dim        = kHidden;
    h.style_dim         = kStyleDim;
    h.n_layer           = kLayers;
    h.max_dur           = kMaxDur;
    h.samples_per_frame = 600;
    return h;
}

using Vec = std::vector<float>;

Vec random_vec(size_t n, std::mt19937 & rng, float sigma = 0.4f) {
    std::normal_distribution<float> dist(0.0f, sigma);
    Vec                             out(n);
    for (float & v : out) {
        v = dist(rng);
    }
    return out;
}

struct LstmData {
    Vec wih_f, whh_f, bih_f, bhh_f;
    Vec wih_r, whh_r, bih_r, bhh_r;
};

LstmData make_lstm(size_t in, size_t hidden, std::mt19937 & rng) {
    LstmData d;
    d.wih_f = random_vec(4 * hidden * in, rng);
    d.whh_f = random_vec(4 * hidden * hidden, rng);
    d.bih_f = random_vec(4 * hidden, rng);
    d.bhh_f = random_vec(4 * hidden, rng);
    d.wih_r = random_vec(4 * hidden * in, rng);
    d.whh_r = random_vec(4 * hidden * hidden, rng);
    d.bih_r = random_vec(4 * hidden, rng);
    d.bhh_r = random_vec(4 * hidden, rng);
    return d;
}

float sigmoidf(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// One direction over `x` laid out [in_dim, tokens] by step.
void reference_lstm_direction(const Vec & wih,
                              const Vec & whh,
                              const Vec & bih,
                              const Vec & bhh,
                              const Vec & x,
                              size_t      in,
                              size_t      hidden,
                              size_t      tokens,
                              bool        reverse,
                              Vec &       out) {
    out.assign(hidden * tokens, 0.0f);
    Vec h(hidden, 0.0f), c(hidden, 0.0f), z(4 * hidden);
    for (size_t step = 0; step < tokens; ++step) {
        const size_t  t  = reverse ? tokens - 1 - step : step;
        const float * xt = x.data() + t * in;
        for (size_t r = 0; r < 4 * hidden; ++r) {
            float acc = bih[r] + bhh[r];
            for (size_t k = 0; k < in; ++k) {
                acc += wih[r * in + k] * xt[k];
            }
            for (size_t k = 0; k < hidden; ++k) {
                acc += whh[r * hidden + k] * h[k];
            }
            z[r] = acc;
        }
        for (size_t k = 0; k < hidden; ++k) {
            const float i = sigmoidf(z[k]);
            const float f = sigmoidf(z[hidden + k]);
            const float g = std::tanh(z[2 * hidden + k]);
            const float o = sigmoidf(z[3 * hidden + k]);
            c[k]          = f * c[k] + i * g;
            h[k]          = o * std::tanh(c[k]);
        }
        for (size_t k = 0; k < hidden; ++k) {
            out[t * hidden + k] = h[k];
        }
    }
}

void reference_ada_layer_norm(Vec &       x,
                              const Vec & fc_w,
                              const Vec & fc_b,
                              const Vec & style,
                              size_t      channels,
                              size_t      tokens) {
    // gamma and beta are the two halves of one projection of the style vector.
    Vec projected(2 * channels);
    for (size_t o = 0; o < 2 * channels; ++o) {
        float acc = fc_b[o];
        for (size_t i = 0; i < style.size(); ++i) {
            acc += fc_w[o * style.size() + i] * style[i];
        }
        projected[o] = acc;
    }
    for (size_t t = 0; t < tokens; ++t) {
        float * row  = x.data() + t * channels;
        float   mean = 0.0f;
        for (size_t i = 0; i < channels; ++i) {
            mean += row[i];
        }
        mean /= float(channels);
        float variance = 0.0f;
        for (size_t i = 0; i < channels; ++i) {
            variance += (row[i] - mean) * (row[i] - mean);
        }
        variance /= float(channels);
        const float inv = 1.0f / std::sqrt(variance + kAdaEps);
        for (size_t i = 0; i < channels; ++i) {
            const float normalized = (row[i] - mean) * inv;
            row[i]                 = (1.0f + projected[i]) * normalized + projected[channels + i];
        }
    }
}

}  // namespace

int main() {
    const synth::kokoro::HParams h        = hparams();
    const size_t                 combined = kHidden + kStyleDim;
    const size_t                 half     = kHidden / 2;

    std::mt19937 rng(20260726u);
    const Vec    input_data = random_vec(size_t(kHidden) * kTokens, rng, 1.0f);
    const Vec    style_data = random_vec(kStyleDim, rng, 1.0f);

    std::vector<LstmData> encoder_lstms;
    std::vector<Vec>      ada_w, ada_b;
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
        encoder_lstms.push_back(make_lstm(combined, half, rng));
        ada_w.push_back(random_vec(2 * kHidden * kStyleDim, rng));
        ada_b.push_back(random_vec(2 * kHidden, rng));
    }
    const LstmData predictor_lstm = make_lstm(combined, half, rng);
    const Vec      proj_w         = random_vec(size_t(kMaxDur) * kHidden, rng);
    const Vec      proj_b         = random_vec(kMaxDur, rng);

    ggml_backend_t backend = ggml_backend_cpu_init();
    SYNTH_TEST_CHECK(backend != nullptr);

    Context        wctx_holder = make_context(256);
    ggml_context * wctx        = wctx_holder.get();

    auto make_lstm_tensors = [&](size_t in) {
        synth::kokoro::LstmTensors t;
        t.forward.weight_ih = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, in, 4 * half);
        t.forward.weight_hh = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, half, 4 * half);
        t.forward.bias_ih   = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 4 * half);
        t.forward.bias_hh   = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 4 * half);
        t.reverse.weight_ih = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, in, 4 * half);
        t.reverse.weight_hh = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, half, 4 * half);
        t.reverse.bias_ih   = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 4 * half);
        t.reverse.bias_hh   = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 4 * half);
        return t;
    };

    synth::kokoro::ProsodyPredictorWeights weights;
    weights.text_encoder.lstms.resize(kLayers);
    weights.text_encoder.ada_norm.resize(kLayers);
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
        weights.text_encoder.lstms[layer]           = make_lstm_tensors(combined);
        weights.text_encoder.ada_norm[layer].weight = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, kStyleDim, 2 * kHidden);
        weights.text_encoder.ada_norm[layer].bias   = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 2 * kHidden);
    }
    weights.lstm                 = make_lstm_tensors(combined);
    weights.duration_proj.weight = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, kHidden, kMaxDur);
    weights.duration_proj.bias   = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, kMaxDur);

    synth::kokoro::DurationScratch scratch;
    scratch.zero_state = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, half);
    scratch.encoder.resize(kLayers);
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
        scratch.encoder[layer].zero_state    = scratch.zero_state;
        scratch.encoder[layer].forward_store = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, half, kTokens);
        scratch.encoder[layer].reverse_store = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, half, kTokens);
    }
    scratch.predictor.zero_state    = scratch.zero_state;
    scratch.predictor.forward_store = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, half, kTokens);
    scratch.predictor.reverse_store = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, half, kTokens);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(wctx, backend);
    SYNTH_TEST_CHECK(buffer != nullptr);

    auto upload = [](ggml_tensor * tensor, const Vec & data) {
        ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
    };
    auto upload_lstm = [&](const synth::kokoro::LstmTensors & t, const LstmData & d) {
        upload(t.forward.weight_ih, d.wih_f);
        upload(t.forward.weight_hh, d.whh_f);
        upload(t.forward.bias_ih, d.bih_f);
        upload(t.forward.bias_hh, d.bhh_f);
        upload(t.reverse.weight_ih, d.wih_r);
        upload(t.reverse.weight_hh, d.whh_r);
        upload(t.reverse.bias_ih, d.bih_r);
        upload(t.reverse.bias_hh, d.bhh_r);
    };
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
        upload_lstm(weights.text_encoder.lstms[layer], encoder_lstms[layer]);
        upload(weights.text_encoder.ada_norm[layer].weight, ada_w[layer]);
        upload(weights.text_encoder.ada_norm[layer].bias, ada_b[layer]);
    }
    upload_lstm(weights.lstm, predictor_lstm);
    upload(weights.duration_proj.weight, proj_w);
    upload(weights.duration_proj.bias, proj_b);
    const Vec zeros(size_t(half) * kTokens, 0.0f);
    upload(scratch.zero_state, zeros);
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
        upload(scratch.encoder[layer].forward_store, zeros);
        upload(scratch.encoder[layer].reverse_store, zeros);
    }
    upload(scratch.predictor.forward_store, zeros);
    upload(scratch.predictor.reverse_store, zeros);

    const uint64_t               budget    = synth::kokoro::duration_graph_node_count(h, kTokens) + 128;
    Context                      graph_ctx = make_context(budget + 256, budget);
    synth::kokoro::DurationGraph graph =
        synth::kokoro::build_duration_graph(graph_ctx.get(), weights, h, scratch, kTokens);
    SYNTH_TEST_CHECK(graph.graph != nullptr);
    SYNTH_TEST_CHECK(graph.hidden->ne[0] == int64_t(combined) && graph.hidden->ne[1] == kTokens);
    SYNTH_TEST_CHECK(graph.logits->ne[0] == kMaxDur && graph.logits->ne[1] == kTokens);
    SYNTH_TEST_CHECK(ggml_graph_n_nodes(graph.graph) <= int(budget));

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    SYNTH_TEST_CHECK(ggml_gallocr_alloc_graph(alloc, graph.graph));
    upload(graph.input, input_data);
    upload(graph.style, style_data);
    ggml_backend_cpu_set_n_threads(backend, 2);
    SYNTH_TEST_CHECK(ggml_backend_graph_compute(backend, graph.graph) == GGML_STATUS_SUCCESS);

    // ---- host reference -----------------------------------------------------
    Vec current(combined * kTokens);
    for (size_t t = 0; t < kTokens; ++t) {
        for (size_t i = 0; i < kHidden; ++i) {
            current[t * combined + i] = input_data[t * kHidden + i];
        }
        for (size_t i = 0; i < kStyleDim; ++i) {
            current[t * combined + kHidden + i] = style_data[i];
        }
    }
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
        const LstmData & d = encoder_lstms[layer];
        Vec              fwd, rev;
        reference_lstm_direction(d.wih_f, d.whh_f, d.bih_f, d.bhh_f, current, combined, half, kTokens, false, fwd);
        reference_lstm_direction(d.wih_r, d.whh_r, d.bih_r, d.bhh_r, current, combined, half, kTokens, true, rev);
        Vec merged(size_t(kHidden) * kTokens);
        for (size_t t = 0; t < kTokens; ++t) {
            for (size_t k = 0; k < half; ++k) {
                merged[t * kHidden + k]        = fwd[t * half + k];
                merged[t * kHidden + half + k] = rev[t * half + k];
            }
        }
        reference_ada_layer_norm(merged, ada_w[layer], ada_b[layer], style_data, kHidden, kTokens);
        for (size_t t = 0; t < kTokens; ++t) {
            for (size_t i = 0; i < kHidden; ++i) {
                current[t * combined + i] = merged[t * kHidden + i];
            }
            for (size_t i = 0; i < kStyleDim; ++i) {
                current[t * combined + kHidden + i] = style_data[i];
            }
        }
    }

    std::vector<float> produced_hidden(combined * kTokens);
    ggml_backend_tensor_get(graph.hidden, produced_hidden.data(), 0, ggml_nbytes(graph.hidden));
    float hidden_diff = 0.0f;
    for (size_t i = 0; i < current.size(); ++i) {
        hidden_diff = std::max(hidden_diff, std::fabs(current[i] - produced_hidden[i]));
    }
    SYNTH_TEST_CHECK(hidden_diff < 1e-5f);

    Vec fwd, rev;
    reference_lstm_direction(predictor_lstm.wih_f, predictor_lstm.whh_f, predictor_lstm.bih_f, predictor_lstm.bhh_f,
                             current, combined, half, kTokens, false, fwd);
    reference_lstm_direction(predictor_lstm.wih_r, predictor_lstm.whh_r, predictor_lstm.bih_r, predictor_lstm.bhh_r,
                             current, combined, half, kTokens, true, rev);
    Vec expected_logits(size_t(kMaxDur) * kTokens);
    for (size_t t = 0; t < kTokens; ++t) {
        Vec merged(kHidden);
        for (size_t k = 0; k < half; ++k) {
            merged[k]        = fwd[t * half + k];
            merged[half + k] = rev[t * half + k];
        }
        for (size_t o = 0; o < kMaxDur; ++o) {
            float acc = proj_b[o];
            for (size_t i = 0; i < kHidden; ++i) {
                acc += proj_w[o * kHidden + i] * merged[i];
            }
            expected_logits[t * kMaxDur + o] = acc;
        }
    }

    std::vector<float> produced_logits(size_t(kMaxDur) * kTokens);
    ggml_backend_tensor_get(graph.logits, produced_logits.data(), 0, ggml_nbytes(graph.logits));
    float logits_diff = 0.0f;
    for (size_t i = 0; i < expected_logits.size(); ++i) {
        logits_diff = std::max(logits_diff, std::fabs(expected_logits[i] - produced_logits[i]));
    }
    SYNTH_TEST_CHECK(logits_diff < 1e-5f);

    // ---- host seam ----------------------------------------------------------
    synth::kokoro::DurationResult resolved;
    SYNTH_TEST_CHECK(synth::kokoro::resolve_durations(produced_logits, h, kTokens, 1.0f, 1440000, resolved) ==
                     SYNTH_OK);
    SYNTH_TEST_CHECK(resolved.pred_dur.size() == kTokens);
    uint64_t summed = 0;
    for (int64_t steps : resolved.pred_dur) {
        SYNTH_TEST_CHECK(steps >= 1);  // no token is ever dropped
        summed += uint64_t(steps);
    }
    SYNTH_TEST_CHECK(summed == resolved.y_length);
    SYNTH_TEST_CHECK(resolved.alignment.size() == kTokens * resolved.y_length);

    // The alignment is one-hot per frame, in ascending token order.
    for (uint64_t frame = 0; frame < resolved.y_length; ++frame) {
        int hits = 0;
        for (uint64_t token = 0; token < kTokens; ++token) {
            hits += resolved.alignment[token * resolved.y_length + frame] != 0.0f ? 1 : 0;
        }
        SYNTH_TEST_CHECK(hits == 1);
    }
    uint64_t cursor = 0;
    for (uint64_t token = 0; token < kTokens; ++token) {
        for (int64_t step = 0; step < resolved.pred_dur[token]; ++step) {
            SYNTH_TEST_CHECK(resolved.alignment[token * resolved.y_length + cursor] == 1.0f);
            ++cursor;
        }
    }

    // A faster rate shortens the output, a slower rate lengthens it.
    synth::kokoro::DurationResult fast, slow;
    SYNTH_TEST_CHECK(synth::kokoro::resolve_durations(produced_logits, h, kTokens, 1.25f, 1440000, fast) == SYNTH_OK);
    SYNTH_TEST_CHECK(synth::kokoro::resolve_durations(produced_logits, h, kTokens, 0.8f, 1440000, slow) == SYNTH_OK);
    SYNTH_TEST_CHECK(fast.y_length <= resolved.y_length);
    SYNTH_TEST_CHECK(resolved.y_length <= slow.y_length);

    // The limit is enforced before the quadratic alignment is allocated.
    synth::kokoro::DurationResult limited;
    SYNTH_TEST_CHECK(synth::kokoro::resolve_durations(produced_logits, h, kTokens, 1.0f, h.samples_per_frame,
                                                      limited) == SYNTH_ERR_OUTPUT_LIMIT);
    SYNTH_TEST_CHECK(limited.alignment.empty());

    // Contract failures.
    synth::kokoro::DurationResult rejected;
    SYNTH_TEST_CHECK(synth::kokoro::resolve_durations(produced_logits, h, 0, 1.0f, 1440000, rejected) ==
                     SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(synth::kokoro::resolve_durations(produced_logits, h, kTokens, 0.0f, 1440000, rejected) ==
                     SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(synth::kokoro::resolve_durations(produced_logits, h, kTokens, 1.0f, 0, rejected) ==
                     SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(synth::kokoro::resolve_durations(Vec(3, 0.0f), h, kTokens, 1.0f, 1440000, rejected) ==
                     SYNTH_ERR_INVALID_ARG);
    Vec not_finite = produced_logits;
    not_finite[0]  = std::numeric_limits<float>::quiet_NaN();
    SYNTH_TEST_CHECK(synth::kokoro::resolve_durations(not_finite, h, kTokens, 1.0f, 1440000, rejected) ==
                     SYNTH_ERR_INTERNAL);

    // Graph contract failures return an empty graph.
    {
        Context ctx = make_context(budget + 256, budget);
        SYNTH_TEST_CHECK(synth::kokoro::build_duration_graph(nullptr, weights, h, scratch, kTokens).graph == nullptr);
        SYNTH_TEST_CHECK(synth::kokoro::build_duration_graph(ctx.get(), weights, h, scratch, 0).graph == nullptr);

        synth::kokoro::DurationScratch missing = scratch;
        missing.encoder.pop_back();
        SYNTH_TEST_CHECK(synth::kokoro::build_duration_graph(ctx.get(), weights, h, missing, kTokens).graph == nullptr);

        synth::kokoro::DurationScratch unowned = scratch;
        unowned.predictor.forward_store        = nullptr;
        SYNTH_TEST_CHECK(synth::kokoro::build_duration_graph(ctx.get(), weights, h, unowned, kTokens).graph == nullptr);

        synth::kokoro::ProsodyPredictorWeights incomplete = weights;
        incomplete.duration_proj.weight                   = nullptr;
        SYNTH_TEST_CHECK(synth::kokoro::build_duration_graph(ctx.get(), incomplete, h, scratch, kTokens).graph ==
                         nullptr);

        synth::kokoro::HParams odd = h;
        odd.hidden_dim             = 7;
        SYNTH_TEST_CHECK(synth::kokoro::build_duration_graph(ctx.get(), weights, odd, scratch, kTokens).graph ==
                         nullptr);
    }

    ggml_gallocr_free(alloc);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    return 0;
}
