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
                out[size_t(c) * out_length + o] += x[size_t(c) * length + i] * w[size_t(c) * kernel + k];
            }
        }
    }
    return out;
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
    ggml_tensor *  input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, shape.length, shape.channels);
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
    if (out == nullptr || out->ne[0] != out_length || out->ne[1] != shape.channels) {
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
        for (int64_t c = 0; c < shape.channels; ++c) {
            for (int64_t o = 0; o < out_length; ++o) {
                expected[size_t(c) * out_length + o] += b[size_t(c)];
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
    return 0;
}
