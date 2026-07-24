// The AdaIN residual block against a host reference, and the prosody graph's
// shape and rejection contract.
//
// The block is where the new operators meet: instance norm on a different axis
// from layer norm, a depthwise transposed pool that doubles the length, a
// nearest-neighbour shortcut that must double it identically, and a 1/sqrt(2)
// branch average. Tensors are [channels, time], so element (c, t) sits at
// t * channels + c.

#include "arch/kokoro/prosody.h"
#include "arch/kokoro/weights.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "test-assert.h"

#include <cmath>
#include <memory>
#include <random>
#include <vector>

namespace {

constexpr float kEpsilon = 1e-5f;
constexpr float kSlope   = 0.2f;

struct ContextDeleter {
    void operator()(ggml_context * context) const {
        if (context != nullptr) {
            ggml_free(context);
        }
    }
};

using Context = std::unique_ptr<ggml_context, ContextDeleter>;
using Vec     = std::vector<float>;

Context make_context(size_t tensors, size_t nodes = 0) {
    ggml_init_params parameters{};
    parameters.mem_size = ggml_tensor_overhead() * tensors +
                          (nodes > 0 ? ggml_graph_overhead_custom(nodes, false) : ggml_graph_overhead());
    parameters.no_alloc = true;
    return Context(ggml_init(parameters));
}

Vec random_vec(size_t n, std::mt19937 & rng, float sigma = 0.5f) {
    std::normal_distribution<float> dist(0.0f, sigma);
    Vec                             out(n);
    for (float & v : out) {
        v = dist(rng);
    }
    return out;
}

// All host helpers address value(c, t) as x[t * channels + c], which is how a
// [channels, time] ggml tensor lays out in memory.

void reference_adain(Vec & x, const Vec & fc_w, const Vec & fc_b, const Vec & style, size_t channels, size_t time) {
    Vec projected(2 * channels);
    for (size_t o = 0; o < 2 * channels; ++o) {
        float acc = fc_b[o];
        for (size_t i = 0; i < style.size(); ++i) {
            acc += fc_w[o * style.size() + i] * style[i];
        }
        projected[o] = acc;
    }
    for (size_t c = 0; c < channels; ++c) {
        float mean = 0.0f;
        for (size_t t = 0; t < time; ++t) {
            mean += x[t * channels + c];
        }
        mean /= float(time);
        float variance = 0.0f;
        for (size_t t = 0; t < time; ++t) {
            const float d = x[t * channels + c] - mean;
            variance += d * d;
        }
        variance /= float(time);
        const float inv = 1.0f / std::sqrt(variance + kEpsilon);
        for (size_t t = 0; t < time; ++t) {
            const float normalized = (x[t * channels + c] - mean) * inv;
            x[t * channels + c]    = (1.0f + projected[c]) * normalized + projected[channels + c];
        }
    }
}

void reference_leaky_relu(Vec & x) {
    for (float & v : x) {
        v = v >= 0.0f ? v : kSlope * v;
    }
}

Vec reference_conv1d(const Vec & x,
                     const Vec & w,
                     const Vec * b,
                     size_t      in_c,
                     size_t      out_c,
                     size_t      time,
                     size_t      kernel,
                     size_t      padding) {
    const size_t out_time = time + 2 * padding - kernel + 1;
    Vec          out(out_c * out_time, 0.0f);
    for (size_t oc = 0; oc < out_c; ++oc) {
        for (size_t t = 0; t < out_time; ++t) {
            float acc = b != nullptr ? (*b)[oc] : 0.0f;
            for (size_t ic = 0; ic < in_c; ++ic) {
                for (size_t k = 0; k < kernel; ++k) {
                    const long src = long(t) + long(k) - long(padding);
                    if (src < 0 || src >= long(time)) {
                        continue;
                    }
                    acc += w[(oc * in_c + ic) * kernel + k] * x[size_t(src) * in_c + ic];
                }
            }
            out[t * out_c + oc] = acc;
        }
    }
    return out;
}

Vec reference_depthwise_transpose(const Vec & x, const Vec & w, const Vec & b, size_t channels, size_t time) {
    const size_t kernel = 3, stride = 2, padding = 1, output_padding = 1;
    const size_t out_time = (time - 1) * stride - 2 * padding + kernel + output_padding;
    Vec          out(channels * out_time, 0.0f);
    for (size_t c = 0; c < channels; ++c) {
        for (size_t i = 0; i < time; ++i) {
            for (size_t k = 0; k < kernel; ++k) {
                const long o = long(i) * long(stride) - long(padding) + long(k);
                if (o < 0 || o >= long(out_time)) {
                    continue;
                }
                out[size_t(o) * channels + c] += x[i * channels + c] * w[c * kernel + k];
            }
        }
        for (size_t o = 0; o < out_time; ++o) {
            out[o * channels + c] += b[c];
        }
    }
    return out;
}

Vec reference_upsample(const Vec & x, size_t channels, size_t time) {
    Vec out(channels * time * 2);
    for (size_t c = 0; c < channels; ++c) {
        for (size_t t = 0; t < time; ++t) {
            out[(2 * t) * channels + c]     = x[t * channels + c];
            out[(2 * t + 1) * channels + c] = x[t * channels + c];
        }
    }
    return out;
}

struct BlockData {
    Vec conv1_w, conv1_b, conv2_w, conv2_b;
    Vec norm1_w, norm1_b, norm2_w, norm2_b;
    Vec conv1x1_w, pool_w, pool_b;
};

Vec reference_block(const Vec &       x,
                    const BlockData & d,
                    const Vec &       style,
                    size_t            in_c,
                    size_t            out_c,
                    size_t            time,
                    bool              upsample) {
    Vec residual = x;
    reference_adain(residual, d.norm1_w, d.norm1_b, style, in_c, time);
    reference_leaky_relu(residual);

    size_t current_time = time;
    if (upsample) {
        residual     = reference_depthwise_transpose(residual, d.pool_w, d.pool_b, in_c, time);
        current_time = time * 2;
    }
    residual = reference_conv1d(residual, d.conv1_w, &d.conv1_b, in_c, out_c, current_time, 3, 1);
    reference_adain(residual, d.norm2_w, d.norm2_b, style, out_c, current_time);
    reference_leaky_relu(residual);
    residual = reference_conv1d(residual, d.conv2_w, &d.conv2_b, out_c, out_c, current_time, 3, 1);

    Vec shortcut = upsample ? reference_upsample(x, in_c, time) : x;
    if (in_c != out_c) {
        shortcut = reference_conv1d(shortcut, d.conv1x1_w, nullptr, in_c, out_c, current_time, 1, 0);
    }

    const float scale = 1.0f / std::sqrt(2.0f);
    Vec         out(out_c * current_time);
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = (residual[i] + shortcut[i]) * scale;
    }
    return out;
}

bool run_block_case(size_t in_c, size_t out_c, size_t time, bool upsample, float & max_diff) {
    std::mt19937 rng(20260727u + unsigned(in_c * 131 + out_c * 17 + time + (upsample ? 1 : 0)));
    const size_t style_dim = 4;

    BlockData d;
    d.conv1_w       = random_vec(out_c * in_c * 3, rng);
    d.conv1_b       = random_vec(out_c, rng);
    d.conv2_w       = random_vec(out_c * out_c * 3, rng);
    d.conv2_b       = random_vec(out_c, rng);
    d.norm1_w       = random_vec(2 * in_c * style_dim, rng);
    d.norm1_b       = random_vec(2 * in_c, rng);
    d.norm2_w       = random_vec(2 * out_c * style_dim, rng);
    d.norm2_b       = random_vec(2 * out_c, rng);
    d.conv1x1_w     = random_vec(out_c * in_c, rng);
    d.pool_w        = random_vec(in_c * 3, rng);
    d.pool_b        = random_vec(in_c, rng);
    const Vec x     = random_vec(in_c * time, rng, 1.0f);
    const Vec style = random_vec(style_dim, rng, 1.0f);

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) {
        return false;
    }

    Context        holder = make_context(64);
    ggml_context * wctx   = holder.get();

    synth::kokoro::AdainResBlockWeights weights;
    weights.conv1.weight    = ggml_new_tensor_3d(wctx, GGML_TYPE_F32, 3, in_c, out_c);
    weights.conv1.bias      = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, out_c);
    weights.conv2.weight    = ggml_new_tensor_3d(wctx, GGML_TYPE_F32, 3, out_c, out_c);
    weights.conv2.bias      = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, out_c);
    weights.norm1.fc.weight = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, style_dim, 2 * in_c);
    weights.norm1.fc.bias   = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 2 * in_c);
    weights.norm2.fc.weight = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, style_dim, 2 * out_c);
    weights.norm2.fc.bias   = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 2 * out_c);
    if (in_c != out_c) {
        weights.conv1x1 = ggml_new_tensor_3d(wctx, GGML_TYPE_F32, 1, in_c, out_c);
    }
    if (upsample) {
        weights.pool.weight = ggml_new_tensor_3d(wctx, GGML_TYPE_F32, 3, 1, in_c);
        weights.pool.bias   = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, in_c);
    }
    ggml_tensor * input        = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, in_c, time);
    ggml_tensor * style_tensor = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, style_dim);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(wctx, backend);
    if (buffer == nullptr) {
        ggml_backend_free(backend);
        return false;
    }
    auto upload = [](ggml_tensor * tensor, const Vec & data) {
        ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
    };
    upload(weights.conv1.weight, d.conv1_w);
    upload(weights.conv1.bias, d.conv1_b);
    upload(weights.conv2.weight, d.conv2_w);
    upload(weights.conv2.bias, d.conv2_b);
    upload(weights.norm1.fc.weight, d.norm1_w);
    upload(weights.norm1.fc.bias, d.norm1_b);
    upload(weights.norm2.fc.weight, d.norm2_w);
    upload(weights.norm2.fc.bias, d.norm2_b);
    if (weights.conv1x1 != nullptr) {
        upload(weights.conv1x1, d.conv1x1_w);
    }
    if (upsample) {
        upload(weights.pool.weight, d.pool_w);
        upload(weights.pool.bias, d.pool_b);
    }
    upload(input, x);
    upload(style_tensor, style);

    Context       graph_holder = make_context(1024, 2048);
    ggml_cgraph * graph        = ggml_new_graph_custom(graph_holder.get(), 2048, false);
    ggml_tensor * out =
        synth::kokoro::build_adain_res_block(graph_holder.get(), input, style_tensor, weights, upsample, kEpsilon);
    const size_t out_time = upsample ? time * 2 : time;
    if (out == nullptr || out->ne[0] != int64_t(out_c) || out->ne[1] != int64_t(out_time)) {
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        return false;
    }
    ggml_build_forward_expand(graph, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, graph);
    ggml_backend_cpu_set_n_threads(backend, 2);
    const bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;

    Vec produced(out_c * out_time);
    if (ok) {
        ggml_backend_tensor_get(out, produced.data(), 0, ggml_nbytes(out));
    }
    const Vec expected = reference_block(x, d, style, in_c, out_c, time, upsample);

    max_diff = 0.0f;
    for (size_t i = 0; i < expected.size() && ok; ++i) {
        max_diff = std::max(max_diff, std::fabs(expected[i] - produced[i]));
    }

    ggml_gallocr_free(alloc);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    return ok;
}

}  // namespace

int main() {
    // The three shapes the prosody stack actually uses, plus a plain block whose
    // channel count changes without upsampling.
    const struct {
        size_t in_c, out_c, time;
        bool   upsample;
    } cases[] = {
        { 6, 6, 7,  false },
        { 6, 3, 7,  true  },
        { 3, 3, 14, false },
        { 4, 8, 5,  false },
    };

    for (const auto & shape : cases) {
        float max_diff = 0.0f;
        SYNTH_TEST_CHECK(run_block_case(shape.in_c, shape.out_c, shape.time, shape.upsample, max_diff));
        SYNTH_TEST_CHECK(max_diff < 1e-4f);
    }

    // A block that upsamples but has no pool is a catalog defect, not something
    // to silently skip.
    {
        Context                             holder = make_context(64);
        ggml_context *                      ctx    = holder.get();
        synth::kokoro::AdainResBlockWeights weights;
        weights.conv1.weight    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 4, 4);
        weights.conv2.weight    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 4, 4);
        weights.norm1.fc.weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8);
        weights.norm2.fc.weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8);
        ggml_tensor * input     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 5);
        ggml_tensor * style     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        SYNTH_TEST_CHECK(synth::kokoro::build_adain_res_block(ctx, input, style, weights, true, kEpsilon) == nullptr);
        SYNTH_TEST_CHECK(synth::kokoro::build_adain_res_block(nullptr, input, style, weights, false, kEpsilon) ==
                         nullptr);

        synth::kokoro::AdainResBlockWeights incomplete = weights;
        incomplete.conv2.weight                        = nullptr;
        SYNTH_TEST_CHECK(synth::kokoro::build_adain_res_block(ctx, input, style, incomplete, false, kEpsilon) ==
                         nullptr);
    }

    // The prosody graph rejects an unusable contract before building.
    {
        const uint32_t                         frames = 6;
        Context                                holder = make_context(4096, 8192);
        synth::kokoro::ProsodyPredictorWeights weights;
        synth::kokoro::ProsodyScratch          scratch;
        synth::kokoro::HParams                 h;
        h.hidden_dim = 8;
        h.style_dim  = 4;
        SYNTH_TEST_CHECK(synth::kokoro::build_prosody_graph(holder.get(), weights, h, scratch, frames).graph ==
                         nullptr);
        SYNTH_TEST_CHECK(synth::kokoro::build_prosody_graph(holder.get(), weights, h, scratch, 0).graph == nullptr);
        SYNTH_TEST_CHECK(synth::kokoro::build_prosody_graph(nullptr, weights, h, scratch, frames).graph == nullptr);
    }
    return 0;
}
