// The acoustic text encoder against a host reference.
//
// This path is distinct from PL-BERT: PL-BERT conditions prosody, while this
// encoder feeds the decoder. Its layer norm normalizes over channels within a
// frame, which is the opposite axis from the prosody stack's AdaIN, so the
// reference checks the arithmetic rather than only the shapes.

#include "arch/kokoro/text-encoder.h"
#include "arch/kokoro/weights.h"
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

constexpr uint32_t kTokens = 6;
constexpr uint32_t kVocab  = 11;
constexpr uint32_t kHidden = 8;
constexpr uint32_t kLayers = 3;
constexpr uint32_t kKernel = 5;
constexpr float    kEps    = 1e-5f;
constexpr float    kSlope  = 0.2f;

struct ContextDeleter {
    void operator()(ggml_context * context) const {
        if (context != nullptr) {
            ggml_free(context);
        }
    }
};

using Context = std::unique_ptr<ggml_context, ContextDeleter>;
using Vec     = std::vector<float>;

Context make_context(size_t tensors, size_t nodes) {
    ggml_init_params parameters{};
    parameters.mem_size = ggml_tensor_overhead() * tensors + ggml_graph_overhead_custom(nodes, false);
    parameters.no_alloc = true;
    return Context(ggml_init(parameters));
}

Vec random_vec(size_t n, std::mt19937 & rng, float sigma = 0.4f) {
    std::normal_distribution<float> dist(0.0f, sigma);
    Vec                             out(n);
    for (float & v : out) {
        v = dist(rng);
    }
    return out;
}

float sigmoidf(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// Tensors are [channels, time]: value(c, t) lives at t * channels + c.
Vec reference_conv1d(const Vec & x,
                     const Vec & w,
                     const Vec & b,
                     size_t      channels,
                     size_t      time,
                     size_t      kernel,
                     size_t      padding) {
    Vec out(channels * time, 0.0f);
    for (size_t t = 0; t < time; ++t) {
        for (size_t oc = 0; oc < channels; ++oc) {
            float acc = b[oc];
            for (size_t ic = 0; ic < channels; ++ic) {
                for (size_t k = 0; k < kernel; ++k) {
                    const long src = long(t) + long(k) - long(padding);
                    if (src < 0 || src >= long(time)) {
                        continue;
                    }
                    acc += w[(oc * channels + ic) * kernel + k] * x[size_t(src) * channels + ic];
                }
            }
            out[t * channels + oc] = acc;
        }
    }
    return out;
}

// Normalizes over channels within each frame, with learned scale and shift.
void reference_layer_norm(Vec & x, const Vec & gamma, const Vec & beta, size_t channels, size_t time) {
    for (size_t t = 0; t < time; ++t) {
        float * frame = x.data() + t * channels;
        float   mean  = 0.0f;
        for (size_t c = 0; c < channels; ++c) {
            mean += frame[c];
        }
        mean /= float(channels);
        float variance = 0.0f;
        for (size_t c = 0; c < channels; ++c) {
            variance += (frame[c] - mean) * (frame[c] - mean);
        }
        variance /= float(channels);
        const float inv = 1.0f / std::sqrt(variance + kEps);
        for (size_t c = 0; c < channels; ++c) {
            frame[c] = (frame[c] - mean) * inv * gamma[c] + beta[c];
        }
    }
}

struct LstmData {
    Vec wih_f, whh_f, bih_f, bhh_f;
    Vec wih_r, whh_r, bih_r, bhh_r;
};

void reference_lstm_direction(const Vec & wih,
                              const Vec & whh,
                              const Vec & bih,
                              const Vec & bhh,
                              const Vec & x,
                              size_t      in,
                              size_t      hidden,
                              size_t      time,
                              bool        reverse,
                              Vec &       out) {
    out.assign(hidden * time, 0.0f);
    Vec h(hidden, 0.0f), c(hidden, 0.0f), z(4 * hidden);
    for (size_t step = 0; step < time; ++step) {
        const size_t t = reverse ? time - 1 - step : step;
        for (size_t r = 0; r < 4 * hidden; ++r) {
            float acc = bih[r] + bhh[r];
            for (size_t k = 0; k < in; ++k) {
                acc += wih[r * in + k] * x[t * in + k];
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

}  // namespace

int main() {
    synth::kokoro::HParams h;
    h.hidden_dim               = kHidden;
    h.n_layer                  = kLayers;
    h.n_token                  = kVocab;
    h.text_encoder_kernel_size = kKernel;

    std::mt19937     rng(20260728u);
    const Vec        embedding = random_vec(size_t(kVocab) * kHidden, rng, 1.0f);
    std::vector<Vec> conv_w, conv_b, norm_g, norm_b;
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
        conv_w.push_back(random_vec(size_t(kHidden) * kHidden * kKernel, rng));
        conv_b.push_back(random_vec(kHidden, rng));
        norm_g.push_back(random_vec(kHidden, rng, 0.3f));
        norm_b.push_back(random_vec(kHidden, rng, 0.3f));
    }
    const size_t half = kHidden / 2;
    LstmData     lstm;
    lstm.wih_f = random_vec(4 * half * kHidden, rng);
    lstm.whh_f = random_vec(4 * half * half, rng);
    lstm.bih_f = random_vec(4 * half, rng);
    lstm.bhh_f = random_vec(4 * half, rng);
    lstm.wih_r = random_vec(4 * half * kHidden, rng);
    lstm.whh_r = random_vec(4 * half * half, rng);
    lstm.bih_r = random_vec(4 * half, rng);
    lstm.bhh_r = random_vec(4 * half, rng);

    ggml_backend_t backend = ggml_backend_cpu_init();
    SYNTH_TEST_CHECK(backend != nullptr);

    Context        holder = make_context(128, 64);
    ggml_context * wctx   = holder.get();

    synth::kokoro::TextEncoderWeights weights;
    weights.embedding = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, kHidden, kVocab);
    weights.cnn.resize(kLayers);
    weights.cnn_norm.resize(kLayers);
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
        weights.cnn[layer].weight      = ggml_new_tensor_3d(wctx, GGML_TYPE_F32, kKernel, kHidden, kHidden);
        weights.cnn[layer].bias        = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, kHidden);
        weights.cnn_norm[layer].weight = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, kHidden);
        weights.cnn_norm[layer].bias   = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, kHidden);
    }
    weights.lstm.forward.weight_ih = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, kHidden, 4 * half);
    weights.lstm.forward.weight_hh = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, half, 4 * half);
    weights.lstm.forward.bias_ih   = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 4 * half);
    weights.lstm.forward.bias_hh   = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 4 * half);
    weights.lstm.reverse.weight_ih = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, kHidden, 4 * half);
    weights.lstm.reverse.weight_hh = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, half, 4 * half);
    weights.lstm.reverse.bias_ih   = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 4 * half);
    weights.lstm.reverse.bias_hh   = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 4 * half);

    synth::kokoro::TextEncoderScratch scratch;
    scratch.lstm.zero_state    = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, half);
    scratch.lstm.forward_store = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, half, kTokens);
    scratch.lstm.reverse_store = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, half, kTokens);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(wctx, backend);
    SYNTH_TEST_CHECK(buffer != nullptr);

    auto upload = [](ggml_tensor * tensor, const Vec & data) {
        ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
    };
    upload(weights.embedding, embedding);
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
        upload(weights.cnn[layer].weight, conv_w[layer]);
        upload(weights.cnn[layer].bias, conv_b[layer]);
        upload(weights.cnn_norm[layer].weight, norm_g[layer]);
        upload(weights.cnn_norm[layer].bias, norm_b[layer]);
    }
    upload(weights.lstm.forward.weight_ih, lstm.wih_f);
    upload(weights.lstm.forward.weight_hh, lstm.whh_f);
    upload(weights.lstm.forward.bias_ih, lstm.bih_f);
    upload(weights.lstm.forward.bias_hh, lstm.bhh_f);
    upload(weights.lstm.reverse.weight_ih, lstm.wih_r);
    upload(weights.lstm.reverse.weight_hh, lstm.whh_r);
    upload(weights.lstm.reverse.bias_ih, lstm.bih_r);
    upload(weights.lstm.reverse.bias_hh, lstm.bhh_r);
    const Vec zeros(size_t(half) * kTokens, 0.0f);
    upload(scratch.lstm.zero_state, zeros);
    upload(scratch.lstm.forward_store, zeros);
    upload(scratch.lstm.reverse_store, zeros);

    const uint64_t                  budget    = synth::kokoro::text_encoder_node_count(h, kTokens) + 128;
    Context                         graph_ctx = make_context(budget + 256, budget);
    synth::kokoro::TextEncoderGraph graph =
        synth::kokoro::build_text_encoder_graph(graph_ctx.get(), weights, h, scratch, kTokens);
    SYNTH_TEST_CHECK(graph.graph != nullptr);
    SYNTH_TEST_CHECK(graph.encoded->ne[0] == kHidden && graph.encoded->ne[1] == kTokens);
    SYNTH_TEST_CHECK(std::strcmp(graph.encoded->name, "text.t_en") == 0);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    SYNTH_TEST_CHECK(ggml_gallocr_alloc_graph(alloc, graph.graph));
    const std::vector<int32_t> tokens = { 0, 4, 10, 2, 7, 1 };
    ggml_backend_tensor_set(graph.token_ids, tokens.data(), 0, ggml_nbytes(graph.token_ids));
    ggml_backend_cpu_set_n_threads(backend, 2);
    SYNTH_TEST_CHECK(ggml_backend_graph_compute(backend, graph.graph) == GGML_STATUS_SUCCESS);

    // ---- host reference -----------------------------------------------------
    Vec current(size_t(kHidden) * kTokens);
    for (size_t t = 0; t < kTokens; ++t) {
        std::memcpy(current.data() + t * kHidden, embedding.data() + size_t(tokens[t]) * kHidden,
                    kHidden * sizeof(float));
    }
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
        current = reference_conv1d(current, conv_w[layer], conv_b[layer], kHidden, kTokens, kKernel, kKernel / 2);
        reference_layer_norm(current, norm_g[layer], norm_b[layer], kHidden, kTokens);
        for (float & v : current) {
            v = v >= 0.0f ? v : kSlope * v;
        }
    }
    Vec fwd, rev;
    reference_lstm_direction(lstm.wih_f, lstm.whh_f, lstm.bih_f, lstm.bhh_f, current, kHidden, half, kTokens, false,
                             fwd);
    reference_lstm_direction(lstm.wih_r, lstm.whh_r, lstm.bih_r, lstm.bhh_r, current, kHidden, half, kTokens, true,
                             rev);
    Vec expected(size_t(kHidden) * kTokens);
    for (size_t t = 0; t < kTokens; ++t) {
        for (size_t k = 0; k < half; ++k) {
            expected[t * kHidden + k]        = fwd[t * half + k];
            expected[t * kHidden + half + k] = rev[t * half + k];
        }
    }

    Vec produced(size_t(kHidden) * kTokens);
    ggml_backend_tensor_get(graph.encoded, produced.data(), 0, ggml_nbytes(graph.encoded));
    float max_diff = 0.0f;
    for (size_t i = 0; i < expected.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(expected[i] - produced[i]));
    }
    SYNTH_TEST_CHECK(max_diff < 1e-5f);

    // Contract failures return an empty graph.
    {
        Context ctx = make_context(budget + 256, budget);
        SYNTH_TEST_CHECK(synth::kokoro::build_text_encoder_graph(nullptr, weights, h, scratch, kTokens).graph ==
                         nullptr);
        SYNTH_TEST_CHECK(synth::kokoro::build_text_encoder_graph(ctx.get(), weights, h, scratch, 0).graph == nullptr);

        synth::kokoro::TextEncoderScratch unowned = scratch;
        unowned.lstm.forward_store                = nullptr;
        SYNTH_TEST_CHECK(synth::kokoro::build_text_encoder_graph(ctx.get(), weights, h, unowned, kTokens).graph ==
                         nullptr);

        synth::kokoro::TextEncoderWeights incomplete = weights;
        incomplete.embedding                         = nullptr;
        SYNTH_TEST_CHECK(synth::kokoro::build_text_encoder_graph(ctx.get(), incomplete, h, scratch, kTokens).graph ==
                         nullptr);

        synth::kokoro::HParams even   = h;
        even.text_encoder_kernel_size = 4;  // an even kernel cannot preserve length
        SYNTH_TEST_CHECK(synth::kokoro::build_text_encoder_graph(ctx.get(), weights, even, scratch, kTokens).graph ==
                         nullptr);
    }

    ggml_gallocr_free(alloc);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    return 0;
}
