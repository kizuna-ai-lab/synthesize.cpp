#include "arch/vits/operations.h"
#include "arch/vits/weights.h"
#include "backend-device.h"
#include "backend-plan.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "test-assert.h"
#include "vits-test-fixture.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace {

ggml_tensor * conv1d_time_major_reference(ggml_context *                     context,
                                          ggml_tensor *                      input,
                                          const synth::vits::Conv1dWeights & weights,
                                          int                                padding,
                                          int                                dilation) {
    ggml_tensor * columns =
        ggml_im2col(context, weights.weight, input, 1, 0, padding, 0, dilation, 0, false, weights.weight->type);
    ggml_tensor * kernel =
        ggml_reshape_2d(context, weights.weight, weights.weight->ne[0] * weights.weight->ne[1], weights.weight->ne[2]);
    ggml_tensor * output =
        ggml_mul_mat(context, kernel, ggml_reshape_2d(context, columns, columns->ne[0], columns->ne[1]));
    output             = ggml_reshape_3d(context, output, weights.weight->ne[2], columns->ne[1], 1);
    output             = ggml_cont(context, ggml_transpose(context, output));
    ggml_tensor * bias = ggml_reshape_2d(context, weights.bias, 1, weights.bias->ne[0]);
    return ggml_add(context, output, bias);
}

int run_conv_case(ggml_backend_dev_t         device,
                  int64_t                    kernel_size,
                  int64_t                    input_channels,
                  int64_t                    output_channels,
                  const std::vector<float> & kernel,
                  const std::vector<float> & bias,
                  const std::vector<float> & input_values,
                  int                        padding,
                  int                        dilation,
                  const std::vector<float> & expected,
                  ggml_type                  weight_type = GGML_TYPE_F32) {
    SYNTH_TEST_CHECK(kernel.size() == static_cast<size_t>(kernel_size * input_channels * output_channels));
    SYNTH_TEST_CHECK(bias.size() == static_cast<size_t>(output_channels));
    SYNTH_TEST_CHECK(input_values.size() % static_cast<size_t>(input_channels) == 0);

    synth::test::GgmlContext weights_context = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(weights_context != nullptr);
    const std::vector<int64_t> weight_shape =
        weight_type == GGML_TYPE_Q8_0 ? std::vector<int64_t>{ kernel_size * input_channels, output_channels } :
                                        std::vector<int64_t>{ kernel_size, input_channels, output_channels };
    synth::test::add_named_tensor(weights_context.get(), "conv.weight", weight_shape, {}, "conv.weight", {},
                                  weight_type);
    synth::test::add_named_tensor(weights_context.get(), "conv.bias", { output_channels }, {}, {}, {});
    synth::vits::Conv1dWeights weights{
        ggml_get_tensor(weights_context.get(), "conv.weight"),
        ggml_get_tensor(weights_context.get(), "conv.bias"),
    };

    const int64_t            input_frames  = static_cast<int64_t>(input_values.size()) / input_channels;
    synth::test::GgmlContext graph_context = synth::test::make_ggml_context(4 * 1024 * 1024);
    SYNTH_TEST_CHECK(graph_context != nullptr);
    ggml_tensor * input = ggml_new_tensor_2d(graph_context.get(), GGML_TYPE_F32, input_channels, input_frames);
    ggml_set_input(input);
    ggml_tensor * output = synth::vits::conv1d(graph_context.get(), input, weights, padding, dilation);
    SYNTH_TEST_CHECK(output != nullptr && output->ne[0] == output_channels &&
                     ggml_nelements(output) == static_cast<int64_t>(expected.size()));
    ggml_cgraph * graph = ggml_new_graph_custom(graph_context.get(), 64, false);
    ggml_build_forward_expand(graph, output);

    std::unique_ptr<synth::BackendPlan> plan;
    SYNTH_TEST_CHECK(synth::BackendPlan::create(device, false, plan) == SYNTH_OK);
    SYNTH_TEST_CHECK(plan != nullptr);
    ggml_backend_buffer_t weights_buffer = ggml_backend_alloc_ctx_tensors(weights_context.get(), plan->primary());
    SYNTH_TEST_CHECK(weights_buffer != nullptr);
    std::vector<ggml_fp16_t> kernel_f16;
    std::vector<uint8_t>     kernel_quantized;
    if (weight_type == GGML_TYPE_F16) {
        kernel_f16.resize(kernel.size());
        ggml_fp32_to_fp16_row(kernel.data(), kernel_f16.data(), static_cast<int64_t>(kernel.size()));
        ggml_backend_tensor_set(weights.weight, kernel_f16.data(), 0, kernel_f16.size() * sizeof(ggml_fp16_t));
    } else if (weight_type == GGML_TYPE_Q8_0) {
        kernel_quantized.resize(ggml_nbytes(weights.weight));
        const size_t written = ggml_quantize_chunk(weight_type, kernel.data(), kernel_quantized.data(), 0,
                                                   output_channels, kernel_size * input_channels, nullptr);
        SYNTH_TEST_CHECK(written == kernel_quantized.size());
        ggml_backend_tensor_set(weights.weight, kernel_quantized.data(), 0, kernel_quantized.size());
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
    if (weight_type == GGML_TYPE_Q8_0) {
        for (size_t index = 0; index < actual.size(); ++index) {
            SYNTH_TEST_CHECK(std::fabs(actual[index] - expected[index]) < 0.5f);
        }
    } else {
        SYNTH_TEST_CHECK(actual == expected);
    }

    ggml_backend_sched_free(scheduler);
    ggml_backend_buffer_free(weights_buffer);
    return 0;
}

int run_im2col_case(ggml_backend_dev_t device, int64_t input_frames) {
    constexpr int64_t kernel_size    = 7;
    constexpr int64_t input_channels = 31;
    constexpr int64_t padding        = 9;
    constexpr int64_t dilation       = 3;

    synth::test::GgmlContext graph_context = synth::test::make_ggml_context(4 * 1024 * 1024);
    SYNTH_TEST_CHECK(graph_context != nullptr);
    ggml_tensor * kernel = ggml_new_tensor_3d(graph_context.get(), GGML_TYPE_F32, kernel_size, input_channels, 1);
    ggml_tensor * input  = ggml_new_tensor_3d(graph_context.get(), GGML_TYPE_F32, input_frames, input_channels, 1);
    ggml_set_input(kernel);
    ggml_set_input(input);
    ggml_tensor * columns =
        ggml_im2col(graph_context.get(), kernel, input, 1, 0, padding, 0, dilation, 0, false, GGML_TYPE_F32);
    SYNTH_TEST_CHECK(columns != nullptr && columns->ne[0] == kernel_size * input_channels &&
                     columns->ne[1] == input_frames && columns->ne[2] == 1);
    ggml_cgraph * graph = ggml_new_graph_custom(graph_context.get(), 16, false);
    ggml_build_forward_expand(graph, columns);

    std::vector<float> input_values(static_cast<size_t>(input_channels * input_frames));
    for (int64_t channel = 0; channel < input_channels; ++channel) {
        for (int64_t frame = 0; frame < input_frames; ++frame) {
            input_values[static_cast<size_t>(channel * input_frames + frame)] =
                static_cast<float>(channel * 1000 + frame + 1);
        }
    }
    std::vector<float> expected(static_cast<size_t>(ggml_nelements(columns)), 0.0f);
    for (int64_t output_frame = 0; output_frame < input_frames; ++output_frame) {
        for (int64_t channel = 0; channel < input_channels; ++channel) {
            for (int64_t kernel_index = 0; kernel_index < kernel_size; ++kernel_index) {
                const int64_t input_frame = output_frame + kernel_index * dilation - padding;
                if (input_frame >= 0 && input_frame < input_frames) {
                    const size_t output_index = static_cast<size_t>(output_frame * input_channels * kernel_size +
                                                                    channel * kernel_size + kernel_index);
                    expected[output_index]    = input_values[static_cast<size_t>(channel * input_frames + input_frame)];
                }
            }
        }
    }

    std::unique_ptr<synth::BackendPlan> plan;
    SYNTH_TEST_CHECK(synth::BackendPlan::create(device, false, plan) == SYNTH_OK);
    SYNTH_TEST_CHECK(plan != nullptr);
    ggml_backend_sched_t scheduler = plan->create_scheduler(16);
    SYNTH_TEST_CHECK(scheduler != nullptr);
    SYNTH_TEST_CHECK(plan->assign_to_primary(scheduler, columns));
    SYNTH_TEST_CHECK(ggml_backend_sched_alloc_graph(scheduler, graph));
    const synth::BackendPlacement placement = plan->inspect_placement(scheduler, graph);
    SYNTH_TEST_CHECK(placement.cpu_fallback_node_count == 0);
    const std::vector<float> kernel_values(static_cast<size_t>(ggml_nelements(kernel)), 0.0f);
    ggml_backend_tensor_set(kernel, kernel_values.data(), 0, kernel_values.size() * sizeof(float));
    ggml_backend_tensor_set(input, input_values.data(), 0, input_values.size() * sizeof(float));
    plan->set_threads(1);
    SYNTH_TEST_CHECK(ggml_backend_sched_graph_compute(scheduler, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(columns, actual.data(), 0, actual.size() * sizeof(float));
    SYNTH_TEST_CHECK(actual == expected);

    ggml_backend_sched_free(scheduler);
    return 0;
}

int run_time_major_chain_case(ggml_backend_dev_t device) {
    constexpr int64_t input_channels  = 17;
    constexpr int64_t hidden_channels = 31;
    constexpr int64_t output_channels = 5;
    constexpr int64_t frames          = 65;
    constexpr int64_t kernel_size     = 7;

    synth::test::GgmlContext weights_context = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(weights_context != nullptr);
    synth::test::add_named_tensor(weights_context.get(), "first.weight",
                                  { kernel_size, input_channels, hidden_channels }, {}, {}, {});
    synth::test::add_named_tensor(weights_context.get(), "first.bias", { hidden_channels }, {}, {}, {});
    synth::test::add_named_tensor(weights_context.get(), "second.weight",
                                  { kernel_size, hidden_channels, output_channels }, {}, {}, {});
    synth::test::add_named_tensor(weights_context.get(), "second.bias", { output_channels }, {}, {}, {});
    const synth::vits::Conv1dWeights first{
        ggml_get_tensor(weights_context.get(), "first.weight"),
        ggml_get_tensor(weights_context.get(), "first.bias"),
    };
    const synth::vits::Conv1dWeights second{
        ggml_get_tensor(weights_context.get(), "second.weight"),
        ggml_get_tensor(weights_context.get(), "second.bias"),
    };

    synth::test::GgmlContext graph_context = synth::test::make_ggml_context(32 * 1024 * 1024);
    SYNTH_TEST_CHECK(graph_context != nullptr);
    ggml_tensor * input = ggml_new_tensor_2d(graph_context.get(), GGML_TYPE_F32, input_channels, frames);
    ggml_set_input(input);

    ggml_tensor * legacy = synth::vits::conv1d(graph_context.get(), input, first, 9, 3);
    legacy               = ggml_leaky_relu(graph_context.get(), legacy, 0.1f, false);
    legacy               = synth::vits::conv1d(graph_context.get(), legacy, second, 3, 1);

    ggml_tensor * time_major = ggml_cont(graph_context.get(), ggml_transpose(graph_context.get(), input));
    time_major               = conv1d_time_major_reference(graph_context.get(), time_major, first, 9, 3);
    time_major               = ggml_leaky_relu(graph_context.get(), time_major, 0.1f, false);
    time_major               = conv1d_time_major_reference(graph_context.get(), time_major, second, 3, 1);
    ggml_tensor * optimized  = ggml_cont(graph_context.get(), ggml_transpose(graph_context.get(), time_major));

    SYNTH_TEST_CHECK(legacy != nullptr && optimized != nullptr);
    SYNTH_TEST_CHECK(legacy->ne[0] == output_channels && legacy->ne[1] == frames);
    SYNTH_TEST_CHECK(ggml_nelements(legacy) == ggml_nelements(optimized));
    ggml_cgraph * graph = ggml_new_graph_custom(graph_context.get(), 256, false);
    ggml_build_forward_expand(graph, legacy);
    ggml_build_forward_expand(graph, optimized);

    std::vector<float> first_weight(static_cast<size_t>(ggml_nelements(first.weight)));
    std::vector<float> first_bias(static_cast<size_t>(ggml_nelements(first.bias)));
    std::vector<float> second_weight(static_cast<size_t>(ggml_nelements(second.weight)));
    std::vector<float> second_bias(static_cast<size_t>(ggml_nelements(second.bias)));
    std::vector<float> input_values(static_cast<size_t>(input_channels * frames));
    for (size_t index = 0; index < first_weight.size(); ++index) {
        first_weight[index] = static_cast<float>(static_cast<int>(index % 17) - 8) * 0.0025f;
    }
    for (size_t index = 0; index < second_weight.size(); ++index) {
        second_weight[index] = static_cast<float>(static_cast<int>(index % 13) - 6) * 0.003f;
    }
    for (size_t index = 0; index < first_bias.size(); ++index) {
        first_bias[index] = static_cast<float>(static_cast<int>(index % 7) - 3) * 0.01f;
    }
    for (size_t index = 0; index < second_bias.size(); ++index) {
        second_bias[index] = static_cast<float>(static_cast<int>(index % 5) - 2) * 0.02f;
    }
    for (size_t index = 0; index < input_values.size(); ++index) {
        input_values[index] = static_cast<float>(static_cast<int>(index % 29) - 14) * 0.05f;
    }

    std::unique_ptr<synth::BackendPlan> plan;
    SYNTH_TEST_CHECK(synth::BackendPlan::create(device, false, plan) == SYNTH_OK);
    SYNTH_TEST_CHECK(plan != nullptr);
    ggml_backend_buffer_t weights_buffer = ggml_backend_alloc_ctx_tensors(weights_context.get(), plan->primary());
    SYNTH_TEST_CHECK(weights_buffer != nullptr);
    ggml_backend_tensor_set(first.weight, first_weight.data(), 0, first_weight.size() * sizeof(float));
    ggml_backend_tensor_set(first.bias, first_bias.data(), 0, first_bias.size() * sizeof(float));
    ggml_backend_tensor_set(second.weight, second_weight.data(), 0, second_weight.size() * sizeof(float));
    ggml_backend_tensor_set(second.bias, second_bias.data(), 0, second_bias.size() * sizeof(float));

    ggml_backend_sched_t scheduler = plan->create_scheduler(256);
    SYNTH_TEST_CHECK(scheduler != nullptr);
    SYNTH_TEST_CHECK(plan->assign_to_primary(scheduler, legacy));
    SYNTH_TEST_CHECK(plan->assign_to_primary(scheduler, optimized));
    SYNTH_TEST_CHECK(ggml_backend_sched_alloc_graph(scheduler, graph));
    const synth::BackendPlacement placement = plan->inspect_placement(scheduler, graph);
    SYNTH_TEST_CHECK(placement.cpu_fallback_node_count == 0);
    ggml_backend_tensor_set(input, input_values.data(), 0, input_values.size() * sizeof(float));
    plan->set_threads(1);
    SYNTH_TEST_CHECK(ggml_backend_sched_graph_compute(scheduler, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> legacy_values(static_cast<size_t>(ggml_nelements(legacy)));
    std::vector<float> optimized_values(legacy_values.size());
    ggml_backend_tensor_get(legacy, legacy_values.data(), 0, legacy_values.size() * sizeof(float));
    ggml_backend_tensor_get(optimized, optimized_values.data(), 0, optimized_values.size() * sizeof(float));
    SYNTH_TEST_CHECK(legacy_values == optimized_values);

    ggml_backend_sched_free(scheduler);
    ggml_backend_buffer_free(weights_buffer);
    return 0;
}

int run_conv(ggml_backend_dev_t device) {
    SYNTH_TEST_CHECK(
        run_conv_case(device, 1, 2, 3, { 1.0f, 2.0f, -1.0f, 0.5f, 3.0f, -2.0f }, { 0.5f, -1.0f, 2.0f },
                      { 1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 5.0f, 2.0f, -2.0f }, 0, 1,
                      { 5.5f, -1.0f, 1.0f, 11.5f, -2.0f, 3.0f, 9.5f, 2.5f, -11.0f, -1.5f, -4.0f, 12.0f }) == 0);
    SYNTH_TEST_CHECK(run_conv_case(device, 3, 1, 1, { 1.0f, 2.0f, 3.0f }, { 0.25f }, { 1.0f, 2.0f, 3.0f, 4.0f }, 2, 2,
                                   { 11.25f, 16.25f, 7.25f, 10.25f }) == 0);
    SYNTH_TEST_CHECK(run_conv_case(device, 1, 2, 3, { 1.0f, 2.0f, -1.0f, 0.5f, 3.0f, -2.0f }, { 0.5f, -1.0f, 2.0f },
                                   { 1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 5.0f, 2.0f, -2.0f }, 0, 1,
                                   { 5.5f, -1.0f, 1.0f, 11.5f, -2.0f, 3.0f, 9.5f, 2.5f, -11.0f, -1.5f, -4.0f, 12.0f },
                                   GGML_TYPE_F16) == 0);
    SYNTH_TEST_CHECK(run_conv_case(device, 3, 1, 1, { 1.0f, 2.0f, 3.0f }, { 0.25f }, { 1.0f, 2.0f, 3.0f, 4.0f }, 2, 2,
                                   { 11.25f, 16.25f, 7.25f, 10.25f }, GGML_TYPE_F16) == 0);
    std::vector<float> q8_kernel(64);
    std::fill(q8_kernel.begin(), q8_kernel.begin() + 32, 1.0f);
    std::fill(q8_kernel.begin() + 32, q8_kernel.end(), -0.5f);
    std::vector<float> q8_input(96);
    std::vector<float> q8_expected(6);
    for (int64_t frame = 0; frame < 3; ++frame) {
        float sum = 0.0f;
        for (int64_t channel = 0; channel < 32; ++channel) {
            const float value                                   = static_cast<float>(frame + channel + 1);
            q8_input[static_cast<size_t>(frame * 32 + channel)] = value;
            sum += value;
        }
        q8_expected[static_cast<size_t>(frame * 2)]     = sum + 0.25f;
        q8_expected[static_cast<size_t>(frame * 2 + 1)] = -0.5f * sum - 0.75f;
    }
    SYNTH_TEST_CHECK(run_conv_case(device, 1, 32, 2, q8_kernel, { 0.25f, -0.75f }, q8_input, 0, 1, q8_expected,
                                   GGML_TYPE_Q8_0) == 0);
    const std::vector<float> q8_spatial_kernel(3 * 32, 1.0f);
    const std::vector<float> q8_spatial_input(4 * 32, 1.0f);
    SYNTH_TEST_CHECK(run_conv_case(device, 3, 32, 1, q8_spatial_kernel, { 0.5f }, q8_spatial_input, 1, 1,
                                   { 64.5f, 96.5f, 96.5f, 64.5f }, GGML_TYPE_Q8_0) == 0);
    SYNTH_TEST_CHECK(run_im2col_case(device, 65) == 0);
    SYNTH_TEST_CHECK(run_im2col_case(device, 257) == 0);
    SYNTH_TEST_CHECK(run_time_major_chain_case(device) == 0);
    return 0;
}

}  // namespace

int main() {
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    SYNTH_TEST_CHECK(cpu != nullptr);
    SYNTH_TEST_CHECK(run_conv(cpu) == 0);

    ggml_backend_dev_t cuda = nullptr;
    if (synth::resolve_requested_device(SYNTH_BACKEND_CUDA, -1, &cuda) == SYNTH_OK) {
        SYNTH_TEST_CHECK(run_conv(cuda) == 0);
    }
    return 0;
}
