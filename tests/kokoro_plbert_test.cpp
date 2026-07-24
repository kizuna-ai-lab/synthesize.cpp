// The PL-BERT graph is compared against a host reference of ALBERT semantics.
//
// The reference is written from the pinned transformers implementation: shared
// embeddings with absolute positions and token type zero, a projection to the
// hidden width, then post-norm attention and a gelu_new feed-forward, with the
// single stored layer group replayed for every hidden layer.

#include "arch/kokoro/plbert.h"
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
#include <string>
#include <vector>

namespace {

constexpr uint32_t kTokens    = 5;
constexpr uint32_t kVocab     = 9;
constexpr uint32_t kEmbedDim  = 4;
constexpr uint32_t kHidden    = 6;
constexpr uint32_t kHeads     = 3;
constexpr uint32_t kIntermed  = 8;
constexpr uint32_t kLayers    = 3;
constexpr uint32_t kPositions = 12;
constexpr float    kEpsilon   = 1e-12f;

struct ContextDeleter {
    void operator()(ggml_context * context) const {
        if (context != nullptr) {
            ggml_free(context);
        }
    }
};

using Context = std::unique_ptr<ggml_context, ContextDeleter>;

Context make_context(size_t tensors) {
    ggml_init_params parameters{};
    parameters.mem_size = ggml_tensor_overhead() * tensors + ggml_graph_overhead();
    parameters.no_alloc = true;
    return Context(ggml_init(parameters));
}

synth::kokoro::HParams hparams() {
    synth::kokoro::HParams h;
    h.plbert.hidden_size             = kHidden;
    h.plbert.num_attention_heads     = kHeads;
    h.plbert.intermediate_size       = kIntermed;
    h.plbert.num_hidden_layers       = kLayers;
    h.plbert.max_position_embeddings = kPositions;
    h.plbert.shared_layer_groups     = 1;
    h.plbert.layer_norm_eps          = kEpsilon;
    h.n_token                        = kVocab;
    return h;
}

// ---------------------------------------------------------------------------
// Host reference
// ---------------------------------------------------------------------------

using Matrix = std::vector<float>;  // row-major [rows, cols]

Matrix random_matrix(size_t rows, size_t cols, std::mt19937 & rng, float sigma = 0.5f) {
    std::normal_distribution<float> dist(0.0f, sigma);
    Matrix                          out(rows * cols);
    for (float & v : out) {
        v = dist(rng);
    }
    return out;
}

// out[t] = W * in[t] + b, where W is [out_dim, in_dim] row-major.
void apply_linear(const Matrix & w,
                  const Matrix & b,
                  const Matrix & in,
                  size_t         in_dim,
                  size_t         out_dim,
                  size_t         tokens,
                  Matrix &       out) {
    out.assign(tokens * out_dim, 0.0f);
    for (size_t t = 0; t < tokens; ++t) {
        for (size_t o = 0; o < out_dim; ++o) {
            float acc = b[o];
            for (size_t i = 0; i < in_dim; ++i) {
                acc += w[o * in_dim + i] * in[t * in_dim + i];
            }
            out[t * out_dim + o] = acc;
        }
    }
}

void apply_layer_norm(Matrix & x, const Matrix & gamma, const Matrix & beta, size_t dim, size_t tokens) {
    for (size_t t = 0; t < tokens; ++t) {
        float * row  = x.data() + t * dim;
        float   mean = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            mean += row[i];
        }
        mean /= float(dim);
        float variance = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            variance += (row[i] - mean) * (row[i] - mean);
        }
        variance /= float(dim);
        const float inv = 1.0f / std::sqrt(variance + kEpsilon);
        for (size_t i = 0; i < dim; ++i) {
            row[i] = (row[i] - mean) * inv * gamma[i] + beta[i];
        }
    }
}

float gelu_new(float x) {
    const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(inner));
}

// GGML's CPU GELU is a half-precision lookup table, so it evaluates
// f16(gelu(f16(x))) rather than the exact activation. Modelling that is how the
// test separates a real graph defect from the backend's known approximation.
float gelu_ggml_table(float x) {
    if (x <= -10.0f) {
        return 0.0f;
    }
    if (x >= 10.0f) {
        return x;
    }
    // Mirror GGML's own association order, which rounds differently from the
    // algebraically identical form above.
    const float rounded = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
    const float value =
        0.5f * rounded *
        (1.0f + std::tanh(0.79788456080286535587989211986876f * rounded * (1.0f + 0.044715f * rounded * rounded)));
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
}

struct LayerMatrices {
    Matrix wq, bq, wk, bk, wv, bv, wd, bd;
    Matrix attn_gamma, attn_beta;
    Matrix wffn, bffn, wffn_out, bffn_out;
    Matrix out_gamma, out_beta;
};

Matrix reference_forward(const Matrix &               word,
                         const Matrix &               position,
                         const Matrix &               token_type,
                         const Matrix &               embed_gamma,
                         const Matrix &               embed_beta,
                         const Matrix &               map_w,
                         const Matrix &               map_b,
                         const LayerMatrices &        layer,
                         const std::vector<int32_t> & tokens,
                         bool                         model_ggml_gelu_table) {
    const size_t count = tokens.size();

    Matrix x(count * kEmbedDim);
    for (size_t t = 0; t < count; ++t) {
        for (size_t i = 0; i < kEmbedDim; ++i) {
            x[t * kEmbedDim + i] =
                word[size_t(tokens[t]) * kEmbedDim + i] + position[t * kEmbedDim + i] + token_type[i];
        }
    }
    apply_layer_norm(x, embed_gamma, embed_beta, kEmbedDim, count);

    Matrix hidden;
    apply_linear(map_w, map_b, x, kEmbedDim, kHidden, count, hidden);

    const size_t head_size = kHidden / kHeads;
    const float  scale     = 1.0f / std::sqrt(float(head_size));

    for (uint32_t depth = 0; depth < kLayers; ++depth) {
        Matrix q, k, v;
        apply_linear(layer.wq, layer.bq, hidden, kHidden, kHidden, count, q);
        apply_linear(layer.wk, layer.bk, hidden, kHidden, kHidden, count, k);
        apply_linear(layer.wv, layer.bv, hidden, kHidden, kHidden, count, v);

        Matrix attended(count * kHidden, 0.0f);
        for (size_t head = 0; head < kHeads; ++head) {
            const size_t base = head * head_size;
            for (size_t qi = 0; qi < count; ++qi) {
                std::vector<float> scores(count);
                float              max_score = -INFINITY;
                for (size_t ki = 0; ki < count; ++ki) {
                    float dot = 0.0f;
                    for (size_t d = 0; d < head_size; ++d) {
                        dot += q[qi * kHidden + base + d] * k[ki * kHidden + base + d];
                    }
                    scores[ki] = dot * scale;
                    max_score  = std::max(max_score, scores[ki]);
                }
                float total = 0.0f;
                for (float & score : scores) {
                    score = std::exp(score - max_score);
                    total += score;
                }
                for (size_t d = 0; d < head_size; ++d) {
                    float acc = 0.0f;
                    for (size_t ki = 0; ki < count; ++ki) {
                        acc += (scores[ki] / total) * v[ki * kHidden + base + d];
                    }
                    attended[qi * kHidden + base + d] = acc;
                }
            }
        }

        Matrix projected;
        apply_linear(layer.wd, layer.bd, attended, kHidden, kHidden, count, projected);
        for (size_t i = 0; i < projected.size(); ++i) {
            projected[i] += hidden[i];
        }
        apply_layer_norm(projected, layer.attn_gamma, layer.attn_beta, kHidden, count);

        Matrix feed;
        apply_linear(layer.wffn, layer.bffn, projected, kHidden, kIntermed, count, feed);
        for (float & value : feed) {
            value = model_ggml_gelu_table ? gelu_ggml_table(value) : gelu_new(value);
        }
        Matrix feed_out;
        apply_linear(layer.wffn_out, layer.bffn_out, feed, kIntermed, kHidden, count, feed_out);
        for (size_t i = 0; i < feed_out.size(); ++i) {
            feed_out[i] += projected[i];
        }
        apply_layer_norm(feed_out, layer.out_gamma, layer.out_beta, kHidden, count);
        hidden = feed_out;
    }
    return hidden;
}

}  // namespace

int main() {
    const synth::kokoro::HParams h = hparams();
    std::mt19937                 rng(20260725u);

    const Matrix word        = random_matrix(kVocab, kEmbedDim, rng);
    const Matrix position    = random_matrix(kPositions, kEmbedDim, rng);
    const Matrix token_type  = random_matrix(2, kEmbedDim, rng);
    const Matrix embed_gamma = random_matrix(1, kEmbedDim, rng, 0.2f);
    const Matrix embed_beta  = random_matrix(1, kEmbedDim, rng, 0.2f);
    const Matrix map_w       = random_matrix(kHidden, kEmbedDim, rng);
    const Matrix map_b       = random_matrix(1, kHidden, rng);

    LayerMatrices layer;
    layer.wq         = random_matrix(kHidden, kHidden, rng);
    layer.bq         = random_matrix(1, kHidden, rng);
    layer.wk         = random_matrix(kHidden, kHidden, rng);
    layer.bk         = random_matrix(1, kHidden, rng);
    layer.wv         = random_matrix(kHidden, kHidden, rng);
    layer.bv         = random_matrix(1, kHidden, rng);
    layer.wd         = random_matrix(kHidden, kHidden, rng);
    layer.bd         = random_matrix(1, kHidden, rng);
    layer.attn_gamma = random_matrix(1, kHidden, rng, 0.2f);
    layer.attn_beta  = random_matrix(1, kHidden, rng, 0.2f);
    layer.wffn       = random_matrix(kIntermed, kHidden, rng);
    layer.bffn       = random_matrix(1, kIntermed, rng);
    layer.wffn_out   = random_matrix(kHidden, kIntermed, rng);
    layer.bffn_out   = random_matrix(1, kHidden, rng);
    layer.out_gamma  = random_matrix(1, kHidden, rng, 0.2f);
    layer.out_beta   = random_matrix(1, kHidden, rng, 0.2f);

    ggml_backend_t backend = ggml_backend_cpu_init();
    SYNTH_TEST_CHECK(backend != nullptr);

    Context        weights_ctx = make_context(64);
    ggml_context * wctx        = weights_ctx.get();

    synth::kokoro::PLBertWeights weights;
    weights.word_embeddings             = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, kEmbedDim, kVocab);
    weights.position_embeddings         = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, kEmbedDim, kPositions);
    weights.token_type_embeddings       = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, kEmbedDim, 2);
    weights.embedding_norm.weight       = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, kEmbedDim);
    weights.embedding_norm.bias         = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, kEmbedDim);
    weights.hidden_mapping.weight       = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, kEmbedDim, kHidden);
    weights.hidden_mapping.bias         = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, kHidden);
    weights.layer.query.weight          = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, kHidden, kHidden);
    weights.layer.query.bias            = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, kHidden);
    weights.layer.key.weight            = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, kHidden, kHidden);
    weights.layer.key.bias              = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, kHidden);
    weights.layer.value.weight          = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, kHidden, kHidden);
    weights.layer.value.bias            = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, kHidden);
    weights.layer.dense.weight          = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, kHidden, kHidden);
    weights.layer.dense.bias            = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, kHidden);
    weights.layer.attention_norm.weight = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, kHidden);
    weights.layer.attention_norm.bias   = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, kHidden);
    weights.layer.ffn.weight            = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, kHidden, kIntermed);
    weights.layer.ffn.bias              = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, kIntermed);
    weights.layer.ffn_output.weight     = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, kIntermed, kHidden);
    weights.layer.ffn_output.bias       = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, kHidden);
    weights.layer.output_norm.weight    = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, kHidden);
    weights.layer.output_norm.bias      = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, kHidden);

    ggml_backend_buffer_t weight_buffer = ggml_backend_alloc_ctx_tensors(wctx, backend);
    SYNTH_TEST_CHECK(weight_buffer != nullptr);

    const struct {
        ggml_tensor *  tensor;
        const Matrix * source;
    } uploads[] = {
        { weights.word_embeddings,             &word             },
        { weights.position_embeddings,         &position         },
        { weights.token_type_embeddings,       &token_type       },
        { weights.embedding_norm.weight,       &embed_gamma      },
        { weights.embedding_norm.bias,         &embed_beta       },
        { weights.hidden_mapping.weight,       &map_w            },
        { weights.hidden_mapping.bias,         &map_b            },
        { weights.layer.query.weight,          &layer.wq         },
        { weights.layer.query.bias,            &layer.bq         },
        { weights.layer.key.weight,            &layer.wk         },
        { weights.layer.key.bias,              &layer.bk         },
        { weights.layer.value.weight,          &layer.wv         },
        { weights.layer.value.bias,            &layer.bv         },
        { weights.layer.dense.weight,          &layer.wd         },
        { weights.layer.dense.bias,            &layer.bd         },
        { weights.layer.attention_norm.weight, &layer.attn_gamma },
        { weights.layer.attention_norm.bias,   &layer.attn_beta  },
        { weights.layer.ffn.weight,            &layer.wffn       },
        { weights.layer.ffn.bias,              &layer.bffn       },
        { weights.layer.ffn_output.weight,     &layer.wffn_out   },
        { weights.layer.ffn_output.bias,       &layer.bffn_out   },
        { weights.layer.output_norm.weight,    &layer.out_gamma  },
        { weights.layer.output_norm.bias,      &layer.out_beta   },
    };

    for (const auto & upload : uploads) {
        ggml_backend_tensor_set(upload.tensor, upload.source->data(), 0, ggml_nbytes(upload.tensor));
    }

    Context                    graph_ctx = make_context(2048);
    synth::kokoro::PLBertGraph graph     = synth::kokoro::build_plbert_graph(graph_ctx.get(), weights, h, kTokens);
    SYNTH_TEST_CHECK(graph.graph != nullptr);
    SYNTH_TEST_CHECK(graph.token_ids != nullptr && graph.token_ids->type == GGML_TYPE_I32);
    SYNTH_TEST_CHECK(graph.token_ids->ne[0] == kTokens);
    SYNTH_TEST_CHECK(graph.hidden != nullptr && graph.hidden->ne[0] == kHidden && graph.hidden->ne[1] == kTokens);
    SYNTH_TEST_CHECK(std::strcmp(graph.hidden->name, "bert.hidden") == 0);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    SYNTH_TEST_CHECK(ggml_gallocr_alloc_graph(alloc, graph.graph));

    const std::vector<int32_t> tokens = { 0, 3, 8, 1, 5 };
    ggml_backend_tensor_set(graph.token_ids, tokens.data(), 0, ggml_nbytes(graph.token_ids));
    ggml_backend_cpu_set_n_threads(backend, 2);
    SYNTH_TEST_CHECK(ggml_backend_graph_compute(backend, graph.graph) == GGML_STATUS_SUCCESS);

    std::vector<float> produced(size_t(kHidden) * kTokens);
    ggml_backend_tensor_get(graph.hidden, produced.data(), 0, ggml_nbytes(graph.hidden));

    auto deviation = [&](const Matrix & reference) {
        float worst = 0.0f;
        for (size_t i = 0; i < reference.size(); ++i) {
            worst = std::max(worst, std::fabs(reference[i] - produced[i]));
        }
        return worst;
    };

    // Against the exact activation the graph is close, but not to float
    // precision: GGML's CPU GELU is a half-precision lookup table.
    const Matrix exact =
        reference_forward(word, position, token_type, embed_gamma, embed_beta, map_w, map_b, layer, tokens, false);
    SYNTH_TEST_CHECK(exact.size() == produced.size());
    SYNTH_TEST_CHECK(deviation(exact) < 5e-3f);

    // Modelling that table shrinks the difference by roughly seven times, which
    // shows the residual is the backend's approximation rather than a graph
    // defect. What remains is ordinary accumulation-order noise carried through
    // three layers.
    const Matrix tabulated =
        reference_forward(word, position, token_type, embed_gamma, embed_beta, map_w, map_b, layer, tokens, true);
    SYNTH_TEST_CHECK(deviation(tabulated) < 5e-4f);
    SYNTH_TEST_CHECK(deviation(tabulated) * 3.0f < deviation(exact));

    // Contract failures return an empty graph rather than a partial one.
    {
        Context ctx = make_context(2048);
        SYNTH_TEST_CHECK(synth::kokoro::build_plbert_graph(nullptr, weights, h, kTokens).graph == nullptr);
        SYNTH_TEST_CHECK(synth::kokoro::build_plbert_graph(ctx.get(), weights, h, 0).graph == nullptr);
        // A sequence longer than the learned position table has no embedding.
        SYNTH_TEST_CHECK(synth::kokoro::build_plbert_graph(ctx.get(), weights, h, kPositions + 1).graph == nullptr);

        synth::kokoro::HParams uneven     = h;
        uneven.plbert.num_attention_heads = 4;  // does not divide the hidden width
        SYNTH_TEST_CHECK(synth::kokoro::build_plbert_graph(ctx.get(), weights, uneven, kTokens).graph == nullptr);

        synth::kokoro::HParams headless     = h;
        headless.plbert.num_attention_heads = 0;
        SYNTH_TEST_CHECK(synth::kokoro::build_plbert_graph(ctx.get(), weights, headless, kTokens).graph == nullptr);

        synth::kokoro::PLBertWeights incomplete = weights;
        incomplete.layer.ffn_output.weight      = nullptr;
        SYNTH_TEST_CHECK(synth::kokoro::build_plbert_graph(ctx.get(), incomplete, h, kTokens).graph == nullptr);
    }

    ggml_gallocr_free(alloc);
    ggml_backend_buffer_free(weight_buffer);
    ggml_backend_free(backend);
    return 0;
}
