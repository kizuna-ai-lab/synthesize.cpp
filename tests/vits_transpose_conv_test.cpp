#include "arch/vits/operations.h"
#include "arch/vits/weights.h"
#include "backend-device.h"
#include "backend-plan.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "test-assert.h"
#include "vits-test-fixture.h"

#include <memory>
#include <vector>

namespace {

int run_transpose_conv_case(ggml_backend_dev_t         device,
                            const std::vector<float> & kernel,
                            const std::vector<float> & input_values,
                            int                        stride,
                            int                        padding,
                            const std::vector<float> & expected,
                            ggml_type                  weight_type = GGML_TYPE_F32) {
    synth::test::GgmlContext weights_context = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(weights_context != nullptr);
    synth::test::add_named_tensor(weights_context.get(), "transpose.weight",
                                  { static_cast<int64_t>(kernel.size()), 1, 1 }, {}, "transpose.weight", {},
                                  weight_type);
    synth::test::add_named_tensor(weights_context.get(), "transpose.bias", { 1 }, {}, {}, {});
    synth::vits::TransposeConv1dWeights weights{
        ggml_get_tensor(weights_context.get(), "transpose.weight"),
        ggml_get_tensor(weights_context.get(), "transpose.bias"),
    };

    synth::test::GgmlContext graph_context = synth::test::make_ggml_context(4 * 1024 * 1024);
    SYNTH_TEST_CHECK(graph_context != nullptr);
    ggml_tensor * input =
        ggml_new_tensor_2d(graph_context.get(), GGML_TYPE_F32, 1, static_cast<int64_t>(input_values.size()));
    ggml_set_input(input);
    ggml_tensor * output = synth::vits::transpose_conv1d(graph_context.get(), input, weights, stride, padding);
    SYNTH_TEST_CHECK(output != nullptr && output->ne[0] == 1 && output->ne[1] == static_cast<int64_t>(expected.size()));
    ggml_cgraph * graph = ggml_new_graph_custom(graph_context.get(), 64, false);
    ggml_build_forward_expand(graph, output);

    std::unique_ptr<synth::BackendPlan> plan;
    SYNTH_TEST_CHECK(synth::BackendPlan::create(device, false, plan) == SYNTH_OK);
    SYNTH_TEST_CHECK(plan != nullptr);
    ggml_backend_buffer_t weights_buffer = ggml_backend_alloc_ctx_tensors(weights_context.get(), plan->primary());
    SYNTH_TEST_CHECK(weights_buffer != nullptr);
    const std::vector<float> bias(1, 0.0f);
    std::vector<ggml_fp16_t> kernel_f16;
    if (weight_type == GGML_TYPE_F16) {
        kernel_f16.resize(kernel.size());
        ggml_fp32_to_fp16_row(kernel.data(), kernel_f16.data(), static_cast<int64_t>(kernel.size()));
        ggml_backend_tensor_set(weights.weight, kernel_f16.data(), 0, kernel_f16.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(weights.weight, kernel.data(), 0, kernel.size() * sizeof(float));
    }
    ggml_backend_tensor_set(weights.bias, bias.data(), 0, bias.size() * sizeof(float));

    ggml_backend_sched_t scheduler = plan->create_scheduler(64);
    SYNTH_TEST_CHECK(scheduler != nullptr);
    SYNTH_TEST_CHECK(plan->assign_to_primary(scheduler, output));
    SYNTH_TEST_CHECK(ggml_backend_sched_alloc_graph(scheduler, graph));
    const synth::BackendPlacement placement = plan->inspect_placement(scheduler, graph);
    SYNTH_TEST_CHECK(placement.cpu_fallback_node_count == 0);
    ggml_backend_tensor_set(input, input_values.data(), 0, input_values.size() * sizeof(float));
    plan->set_threads(1);
    SYNTH_TEST_CHECK(ggml_backend_sched_graph_compute(scheduler, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    SYNTH_TEST_CHECK(actual == expected);

    ggml_backend_sched_free(scheduler);
    ggml_backend_buffer_free(weights_buffer);
    return 0;
}

// The transposed convolution written out from its definition, so the graph is
// checked against the semantics rather than against a second implementation of
// the same decomposition. Layouts are ggml's: the kernel is [kernel_size,
// out_channels, in_channels], the input [in_channels, length], the result
// [out_channels, length_out].
std::vector<float> reference_transpose_conv(const std::vector<float> & kernel,
                                            const std::vector<float> & input,
                                            int64_t                    kernel_size,
                                            int64_t                    out_channels,
                                            int64_t                    in_channels,
                                            int64_t                    length,
                                            int64_t                    stride,
                                            int64_t                    padding) {
    const int64_t      raw_length = (length - 1) * stride + kernel_size;
    const int64_t      out_length = raw_length - 2 * padding;
    std::vector<float> raw(static_cast<size_t>(raw_length * out_channels), 0.0f);
    for (int64_t channel = 0; channel < in_channels; ++channel) {
        for (int64_t step = 0; step < length; ++step) {
            const float value = input[static_cast<size_t>(step * in_channels + channel)];
            for (int64_t out = 0; out < out_channels; ++out) {
                for (int64_t tap = 0; tap < kernel_size; ++tap) {
                    const size_t weight_index = static_cast<size_t>((channel * out_channels + out) * kernel_size + tap);
                    raw[static_cast<size_t>(out * raw_length + step * stride + tap)] += value * kernel[weight_index];
                }
            }
        }
    }
    std::vector<float> expected(static_cast<size_t>(out_length * out_channels));
    for (int64_t out = 0; out < out_channels; ++out) {
        for (int64_t step = 0; step < out_length; ++step) {
            expected[static_cast<size_t>(step * out_channels + out)] =
                raw[static_cast<size_t>(out * raw_length + step + padding)];
        }
    }
    return expected;
}

// The single-channel cases above cannot see a channel-ordering mistake, and the
// column decomposition this operation is built on lives or dies by that
// ordering: the kernel's leading pair is merged into one axis, so a swap of
// out_channels and kernel_size would still produce the right shape.
int run_transpose_conv_multichannel_case(ggml_backend_dev_t device,
                                         int64_t            kernel_size,
                                         int64_t            out_channels,
                                         int64_t            in_channels,
                                         int64_t            length,
                                         int                stride,
                                         int                padding,
                                         ggml_type          weight_type = GGML_TYPE_F32) {
    std::vector<float> kernel(static_cast<size_t>(kernel_size * out_channels * in_channels));
    for (size_t i = 0; i < kernel.size(); ++i) {
        // Small distinct integers: exact in F16 and in every accumulation
        // order, so the comparison below can stay exact.
        kernel[i] = static_cast<float>((i % 7) + 1);
    }
    std::vector<float> input_values(static_cast<size_t>(in_channels * length));
    for (size_t i = 0; i < input_values.size(); ++i) {
        input_values[i] = static_cast<float>((i % 5) + 1);
    }
    const std::vector<float> expected =
        reference_transpose_conv(kernel, input_values, kernel_size, out_channels, in_channels, length, stride, padding);

    synth::test::GgmlContext weights_context = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(weights_context != nullptr);
    synth::test::add_named_tensor(weights_context.get(), "transpose.weight", { kernel_size, out_channels, in_channels },
                                  {}, "transpose.weight", {}, weight_type);
    synth::test::add_named_tensor(weights_context.get(), "transpose.bias", { out_channels }, {}, {}, {});
    synth::vits::TransposeConv1dWeights weights{
        ggml_get_tensor(weights_context.get(), "transpose.weight"),
        ggml_get_tensor(weights_context.get(), "transpose.bias"),
    };

    synth::test::GgmlContext graph_context = synth::test::make_ggml_context(8 * 1024 * 1024);
    SYNTH_TEST_CHECK(graph_context != nullptr);
    ggml_tensor * input = ggml_new_tensor_2d(graph_context.get(), GGML_TYPE_F32, in_channels, length);
    ggml_set_input(input);
    ggml_tensor * output = synth::vits::transpose_conv1d(graph_context.get(), input, weights, stride, padding);
    SYNTH_TEST_CHECK(output != nullptr && output->ne[0] == out_channels &&
                     output->ne[1] * output->ne[0] == static_cast<int64_t>(expected.size()));
    ggml_cgraph * graph = ggml_new_graph_custom(graph_context.get(), 64, false);
    ggml_build_forward_expand(graph, output);

    std::unique_ptr<synth::BackendPlan> plan;
    SYNTH_TEST_CHECK(synth::BackendPlan::create(device, false, plan) == SYNTH_OK);
    SYNTH_TEST_CHECK(plan != nullptr);
    ggml_backend_buffer_t weights_buffer = ggml_backend_alloc_ctx_tensors(weights_context.get(), plan->primary());
    SYNTH_TEST_CHECK(weights_buffer != nullptr);
    const std::vector<float> bias(static_cast<size_t>(out_channels), 0.0f);
    std::vector<ggml_fp16_t> kernel_f16;
    if (weight_type == GGML_TYPE_F16) {
        kernel_f16.resize(kernel.size());
        ggml_fp32_to_fp16_row(kernel.data(), kernel_f16.data(), static_cast<int64_t>(kernel.size()));
        ggml_backend_tensor_set(weights.weight, kernel_f16.data(), 0, kernel_f16.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(weights.weight, kernel.data(), 0, kernel.size() * sizeof(float));
    }
    ggml_backend_tensor_set(weights.bias, bias.data(), 0, bias.size() * sizeof(float));

    ggml_backend_sched_t scheduler = plan->create_scheduler(64);
    SYNTH_TEST_CHECK(scheduler != nullptr);
    SYNTH_TEST_CHECK(plan->assign_to_primary(scheduler, output));
    SYNTH_TEST_CHECK(ggml_backend_sched_alloc_graph(scheduler, graph));
    const synth::BackendPlacement placement = plan->inspect_placement(scheduler, graph);
    SYNTH_TEST_CHECK(placement.cpu_fallback_node_count == 0);
    ggml_backend_tensor_set(input, input_values.data(), 0, input_values.size() * sizeof(float));
    plan->set_threads(1);
    SYNTH_TEST_CHECK(ggml_backend_sched_graph_compute(scheduler, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    SYNTH_TEST_CHECK(actual == expected);

    ggml_backend_sched_free(scheduler);
    ggml_backend_buffer_free(weights_buffer);
    return 0;
}

int run_transpose_conv(ggml_backend_dev_t device) {
    SYNTH_TEST_CHECK(run_transpose_conv_case(device, std::vector<float>(4, 1.0f), { 1.0f, 2.0f }, 2, 1,
                                             { 1.0f, 3.0f, 3.0f, 2.0f }) == 0);
    SYNTH_TEST_CHECK(run_transpose_conv_case(device, std::vector<float>(4, 1.0f), { 1.0f, 2.0f }, 2, 1,
                                             { 1.0f, 3.0f, 3.0f, 2.0f }, GGML_TYPE_F16) == 0);
    SYNTH_TEST_CHECK(
        run_transpose_conv_case(
            device,
            { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f },
            { 1.0f, 2.0f, 3.0f }, 8, 4,
            { 5.0f,  6.0f,  7.0f,  8.0f,  11.0f, 14.0f, 17.0f, 20.0f, 23.0f, 26.0f, 29.0f, 32.0f,
              21.0f, 26.0f, 31.0f, 36.0f, 41.0f, 46.0f, 51.0f, 56.0f, 27.0f, 30.0f, 33.0f, 36.0f }) == 0);
    // Several channels in both directions, in the decoder's own kernel/stride
    // shape (kernel = 2 * stride) and in a shape where the padding does not
    // consume a whole stride.
    for (ggml_type weight_type : { GGML_TYPE_F32, GGML_TYPE_F16 }) {
        SYNTH_TEST_CHECK(run_transpose_conv_multichannel_case(device, 16, 3, 4, 5, 8, 4, weight_type) == 0);
        SYNTH_TEST_CHECK(run_transpose_conv_multichannel_case(device, 4, 5, 2, 7, 2, 1, weight_type) == 0);
        SYNTH_TEST_CHECK(run_transpose_conv_multichannel_case(device, 3, 2, 3, 6, 1, 0, weight_type) == 0);
    }
    return 0;
}

}  // namespace

int main() {
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    SYNTH_TEST_CHECK(cpu != nullptr);
    SYNTH_TEST_CHECK(run_transpose_conv(cpu) == 0);

    ggml_backend_dev_t cuda = nullptr;
    if (synth::resolve_requested_device(SYNTH_BACKEND_CUDA, -1, &cuda) == SYNTH_OK) {
        SYNTH_TEST_CHECK(run_transpose_conv(cuda) == 0);
    }
    return 0;
}
