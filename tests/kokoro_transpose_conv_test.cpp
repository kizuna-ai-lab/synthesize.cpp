// The depthwise transposed convolution is compared against a host reference of
// PyTorch's ConvTranspose1d scatter definition.
//
// GGML has no grouped transposed convolution and forbids internal padding, so
// this operation is assembled from a zero-interleave and per-tap shifts. The
// reference is the definition itself, which is what makes the assembly
// checkable rather than merely plausible.

#include "arch/kokoro/operations.h"
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
    parameters.mem_size = ggml_tensor_overhead() * 512 + ggml_graph_overhead();
    parameters.no_alloc = true;
    return Context(ggml_init(parameters));
}

// y[c][o] = sum over i, k of x[c][i] * w[c][k] where o == i * stride - padding + k.
// Tensors are [channels, time], so element (c, t) sits at t * channels + c.
std::vector<float> reference(const std::vector<float> & x,
                             const std::vector<float> & w,
                             int64_t                    length,
                             int64_t                    channels,
                             int64_t                    kernel,
                             int64_t                    stride,
                             int64_t                    padding,
                             int64_t                    output_padding,
                             int64_t                    out_length) {
    std::vector<float> out(size_t(out_length) * channels, 0.0f);
    for (int64_t c = 0; c < channels; ++c) {
        for (int64_t i = 0; i < length; ++i) {
            for (int64_t k = 0; k < kernel; ++k) {
                const int64_t o = i * stride - padding + k;
                if (o < 0 || o >= out_length) {
                    continue;
                }
                out[size_t(o) * channels + c] += x[size_t(i) * channels + c] * w[size_t(c) * kernel + k];
            }
        }
    }
    return out;
}

// The same scatter, now across all channels:
// y[o][n] = sum over i, t, k of x[i][t] * w[i][o][k] where n == t * stride - padding + k.
// The weight is [kernel, out_channels, in_channels] in GGML's reported order.
std::vector<float> reference_dense(const std::vector<float> & x,
                                   const std::vector<float> & w,
                                   int64_t                    length,
                                   int64_t                    in_channels,
                                   int64_t                    out_channels,
                                   int64_t                    kernel,
                                   int64_t                    stride,
                                   int64_t                    padding,
                                   int64_t                    out_length) {
    std::vector<float> out(size_t(out_length) * out_channels, 0.0f);
    for (int64_t t = 0; t < length; ++t) {
        for (int64_t k = 0; k < kernel; ++k) {
            const int64_t n = t * stride - padding + k;
            if (n < 0 || n >= out_length) {
                continue;
            }
            for (int64_t i = 0; i < in_channels; ++i) {
                const float value = x[size_t(t) * in_channels + i];
                for (int64_t o = 0; o < out_channels; ++o) {
                    const size_t weight_index = size_t(i) * out_channels * kernel + size_t(o) * kernel + size_t(k);
                    out[size_t(n) * out_channels + o] += value * w[weight_index];
                }
            }
        }
    }
    return out;
}

struct DenseCase {
    int64_t length;
    int64_t in_channels;
    int64_t out_channels;
    int64_t kernel;
    int64_t stride;
    int64_t padding;
    int64_t output_padding;
    bool    with_bias;
};

bool run_dense_case(const DenseCase & shape, float & max_diff) {
    const int64_t out_length = synth::kokoro::transpose_conv1d_length(shape.length, shape.kernel, shape.stride,
                                                                      shape.padding, shape.output_padding);

    std::mt19937                    rng(20260727u + unsigned(shape.length * 31 + shape.kernel));
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float>              x(size_t(shape.length) * shape.in_channels);
    std::vector<float>              w(size_t(shape.kernel) * shape.in_channels * shape.out_channels);
    std::vector<float>              b(size_t(shape.out_channels));
    for (float & v : x) {
        v = dist(rng);
    }
    for (float & v : w) {
        v = dist(rng);
    }
    for (float & v : b) {
        v = dist(rng);
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) {
        return false;
    }

    Context        holder = make_context();
    ggml_context * ctx    = holder.get();
    ggml_tensor *  input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, shape.in_channels, shape.length);
    ggml_tensor *  weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, shape.kernel, shape.out_channels, shape.in_channels);
    ggml_tensor *  bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, shape.out_channels);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_backend_free(backend);
        return false;
    }
    ggml_backend_tensor_set(input, x.data(), 0, ggml_nbytes(input));
    ggml_backend_tensor_set(weight, w.data(), 0, ggml_nbytes(weight));
    ggml_backend_tensor_set(bias, b.data(), 0, ggml_nbytes(bias));

    Context       graph_holder = make_context();
    ggml_cgraph * graph        = ggml_new_graph(graph_holder.get());
    ggml_tensor * out =
        synth::kokoro::transpose_conv1d(graph_holder.get(), input, weight, shape.with_bias ? bias : nullptr,
                                        shape.stride, shape.padding, shape.output_padding);
    if (out == nullptr || out->ne[0] != shape.out_channels || out->ne[1] != out_length) {
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        return false;
    }
    ggml_build_forward_expand(graph, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, graph);
    ggml_backend_cpu_set_n_threads(backend, 2);
    const bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;

    std::vector<float> produced(size_t(out_length) * shape.out_channels);
    if (ok) {
        ggml_backend_tensor_get(out, produced.data(), 0, ggml_nbytes(out));
    }

    std::vector<float> expected = reference_dense(x, w, shape.length, shape.in_channels, shape.out_channels,
                                                  shape.kernel, shape.stride, shape.padding, out_length);
    if (shape.with_bias) {
        for (int64_t n = 0; n < out_length; ++n) {
            for (int64_t o = 0; o < shape.out_channels; ++o) {
                expected[size_t(n) * shape.out_channels + o] += b[size_t(o)];
            }
        }
    }

    max_diff = 0.0f;
    for (size_t i = 0; i < expected.size() && ok; ++i) {
        max_diff = std::max(max_diff, std::fabs(expected[i] - produced[i]));
    }

    ggml_gallocr_free(alloc);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    return ok;
}

bool run_reflect_pad(float & max_diff) {
    const int64_t channels = 3;
    const int64_t length   = 6;

    std::vector<float> x(size_t(length) * channels);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = float(i) * 0.25f - 1.0f;
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) {
        return false;
    }
    Context               holder = make_context();
    ggml_tensor *         input  = ggml_new_tensor_2d(holder.get(), GGML_TYPE_F32, channels, length);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(holder.get(), backend);
    if (buffer == nullptr) {
        ggml_backend_free(backend);
        return false;
    }
    ggml_backend_tensor_set(input, x.data(), 0, ggml_nbytes(input));

    Context       graph_holder = make_context();
    ggml_cgraph * graph        = ggml_new_graph(graph_holder.get());
    ggml_tensor * out          = synth::kokoro::reflect_pad_left_1(graph_holder.get(), input);
    if (out == nullptr || out->ne[0] != channels || out->ne[1] != length + 1) {
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        return false;
    }
    ggml_build_forward_expand(graph, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, graph);
    const bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;

    std::vector<float> produced(size_t(length + 1) * channels);
    if (ok) {
        ggml_backend_tensor_get(out, produced.data(), 0, ggml_nbytes(out));
    }

    // The new leading frame mirrors across frame zero, so it is frame one.
    std::vector<float> expected(size_t(length + 1) * channels);
    for (int64_t c = 0; c < channels; ++c) {
        expected[size_t(c)] = x[size_t(channels) + size_t(c)];
    }
    for (int64_t t = 0; t < length; ++t) {
        for (int64_t c = 0; c < channels; ++c) {
            expected[size_t(t + 1) * channels + size_t(c)] = x[size_t(t) * channels + size_t(c)];
        }
    }

    max_diff = 0.0f;
    for (size_t i = 0; i < expected.size() && ok; ++i) {
        max_diff = std::max(max_diff, std::fabs(expected[i] - produced[i]));
    }

    ggml_gallocr_free(alloc);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    return ok;
}

struct Case {
    int64_t length;
    int64_t channels;
    int64_t kernel;
    int64_t stride;
    int64_t padding;
    int64_t output_padding;
    bool    with_bias;
};

bool run_case(const Case & shape, float & max_diff) {
    const int64_t out_length = synth::kokoro::depthwise_transpose_conv1d_length(
        shape.length, shape.kernel, shape.stride, shape.padding, shape.output_padding);

    std::mt19937                    rng(20260726u + unsigned(shape.length * 31 + shape.kernel));
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float>              x(size_t(shape.length) * shape.channels);
    std::vector<float>              w(size_t(shape.kernel) * shape.channels);
    std::vector<float>              b(size_t(shape.channels));
    for (float & v : x) {
        v = dist(rng);
    }
    for (float & v : w) {
        v = dist(rng);
    }
    for (float & v : b) {
        v = dist(rng);
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) {
        return false;
    }

    Context        holder = make_context();
    ggml_context * ctx    = holder.get();
    ggml_tensor *  input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, shape.channels, shape.length);
    ggml_tensor *  weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, shape.kernel, 1, shape.channels);
    ggml_tensor *  bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, shape.channels);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_backend_free(backend);
        return false;
    }
    ggml_backend_tensor_set(input, x.data(), 0, ggml_nbytes(input));
    ggml_backend_tensor_set(weight, w.data(), 0, ggml_nbytes(weight));
    ggml_backend_tensor_set(bias, b.data(), 0, ggml_nbytes(bias));

    Context       graph_holder = make_context();
    ggml_cgraph * graph        = ggml_new_graph(graph_holder.get());
    ggml_tensor * out =
        synth::kokoro::depthwise_transpose_conv1d(graph_holder.get(), input, weight, shape.with_bias ? bias : nullptr,
                                                  shape.stride, shape.padding, shape.output_padding);
    if (out == nullptr || out->ne[0] != shape.channels || out->ne[1] != out_length) {
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        return false;
    }
    ggml_build_forward_expand(graph, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, graph);
    ggml_backend_cpu_set_n_threads(backend, 2);
    const bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;

    std::vector<float> produced(size_t(out_length) * shape.channels);
    if (ok) {
        ggml_backend_tensor_get(out, produced.data(), 0, ggml_nbytes(out));
    }

    std::vector<float> expected = reference(x, w, shape.length, shape.channels, shape.kernel, shape.stride,
                                            shape.padding, shape.output_padding, out_length);
    if (shape.with_bias) {
        for (int64_t o = 0; o < out_length; ++o) {
            for (int64_t c = 0; c < shape.channels; ++c) {
                expected[size_t(o) * shape.channels + c] += b[size_t(c)];
            }
        }
    }

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
    // The first case is Kokoro's own upsampling pool: kernel 3, stride 2,
    // padding 1, output padding 1, which exactly doubles the length. The rest
    // vary each parameter so the assembly is not fitted to one shape.
    const Case cases[] = {
        { 7,  5, 3, 2, 1, 1, true  },
        { 7,  5, 3, 2, 1, 1, false },
        { 1,  3, 3, 2, 1, 1, true  },
        { 12, 4, 3, 1, 0, 0, true  },
        { 6,  2, 5, 2, 2, 1, true  },
        { 9,  3, 4, 3, 1, 2, true  },
        { 4,  6, 1, 1, 0, 0, true  },
    };

    for (const Case & shape : cases) {
        float max_diff = 0.0f;
        SYNTH_TEST_CHECK(run_case(shape, max_diff));
        SYNTH_TEST_CHECK(max_diff < 1e-5f);
    }

    // Kokoro's pool doubles the frame count, which the prosody and decoder
    // stages depend on for their static shapes.
    SYNTH_TEST_CHECK(synth::kokoro::depthwise_transpose_conv1d_length(100, 3, 2, 1, 1) == 200);
    SYNTH_TEST_CHECK(synth::kokoro::depthwise_transpose_conv1d_length(1, 3, 2, 1, 1) == 2);

    // Contract failures return null rather than a wrong shape.
    Context        holder = make_context();
    ggml_context * ctx    = holder.get();
    ggml_tensor *  input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
    ggml_tensor *  weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 1, 4);
    SYNTH_TEST_CHECK(synth::kokoro::depthwise_transpose_conv1d(nullptr, input, weight, nullptr, 2, 1, 1) == nullptr);
    SYNTH_TEST_CHECK(synth::kokoro::depthwise_transpose_conv1d(ctx, nullptr, weight, nullptr, 2, 1, 1) == nullptr);
    SYNTH_TEST_CHECK(synth::kokoro::depthwise_transpose_conv1d(ctx, input, nullptr, nullptr, 2, 1, 1) == nullptr);
    SYNTH_TEST_CHECK(synth::kokoro::depthwise_transpose_conv1d(ctx, input, weight, nullptr, 0, 1, 1) == nullptr);

    // A kernel whose channel count disagrees with the input is a catalog defect.
    ggml_tensor * mismatched = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 1, 5);
    SYNTH_TEST_CHECK(synth::kokoro::depthwise_transpose_conv1d(ctx, input, mismatched, nullptr, 2, 1, 1) == nullptr);

    // Padding wider than the kernel would need a negative window, which the
    // operation refuses rather than silently clamping.
    SYNTH_TEST_CHECK(synth::kokoro::depthwise_transpose_conv1d(ctx, input, weight, nullptr, 2, 5, 1) == nullptr);

    // The generator's upsamplers are the dense form. Its two real shapes are
    // kernel 20 stride 10 and kernel 12 stride 6, both with padding (k - u) / 2,
    // which multiplies the length by exactly the stride.
    const DenseCase dense_cases[] = {
        { 5, 4, 6, 20, 10, 5, 0, true  },
        { 5, 4, 6, 12, 6,  3, 0, true  },
        { 5, 4, 6, 12, 6,  3, 0, false },
        { 3, 2, 2, 3,  1,  0, 0, true  },
        { 4, 3, 5, 1,  1,  0, 0, true  },
        { 6, 5, 3, 4,  3,  1, 2, true  },
        { 1, 3, 4, 20, 10, 5, 0, true  },
    };
    for (const DenseCase & shape : dense_cases) {
        float max_diff = 0.0f;
        SYNTH_TEST_CHECK(run_dense_case(shape, max_diff));
        SYNTH_TEST_CHECK(max_diff < 1e-4f);
    }

    SYNTH_TEST_CHECK(synth::kokoro::transpose_conv1d_length(100, 20, 10, 5, 0) == 1000);
    SYNTH_TEST_CHECK(synth::kokoro::transpose_conv1d_length(100, 12, 6, 3, 0) == 600);

    ggml_tensor * dense_weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 6, 8);
    SYNTH_TEST_CHECK(synth::kokoro::transpose_conv1d(nullptr, input, dense_weight, nullptr, 2, 1, 1) == nullptr);
    SYNTH_TEST_CHECK(synth::kokoro::transpose_conv1d(ctx, nullptr, dense_weight, nullptr, 2, 1, 1) == nullptr);
    SYNTH_TEST_CHECK(synth::kokoro::transpose_conv1d(ctx, input, nullptr, nullptr, 2, 1, 1) == nullptr);
    SYNTH_TEST_CHECK(synth::kokoro::transpose_conv1d(ctx, input, dense_weight, nullptr, 0, 1, 1) == nullptr);
    // A weight whose input-channel axis disagrees with the input is a catalog defect.
    ggml_tensor * wrong_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 6, 7);
    SYNTH_TEST_CHECK(synth::kokoro::transpose_conv1d(ctx, input, wrong_in, nullptr, 2, 1, 1) == nullptr);
    SYNTH_TEST_CHECK(synth::kokoro::transpose_conv1d(ctx, input, dense_weight, nullptr, 2, 5, 1) == nullptr);

    float pad_diff = 0.0f;
    SYNTH_TEST_CHECK(run_reflect_pad(pad_diff));
    SYNTH_TEST_CHECK(pad_diff == 0.0f);
    SYNTH_TEST_CHECK(synth::kokoro::reflect_pad_left_1(nullptr, input) == nullptr);
    SYNTH_TEST_CHECK(synth::kokoro::reflect_pad_left_1(ctx, nullptr) == nullptr);
    // Reflecting needs a second frame to mirror.
    ggml_tensor * single = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
    SYNTH_TEST_CHECK(synth::kokoro::reflect_pad_left_1(ctx, single) == nullptr);
    return 0;
}
