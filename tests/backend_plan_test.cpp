#include "backend-plan.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "test-assert.h"

#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

int main() {
    std::unique_ptr<synth::BackendPlan> plan;
    SYNTH_TEST_CHECK(synth::BackendPlan::create(nullptr, false, plan) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(plan == nullptr);

    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    SYNTH_TEST_CHECK(cpu != nullptr);
    SYNTH_TEST_CHECK(synth::BackendPlan::create(cpu, false, plan) == SYNTH_OK);
    SYNTH_TEST_CHECK(plan != nullptr);
    SYNTH_TEST_CHECK(plan->primary_device() == cpu);
    SYNTH_TEST_CHECK(plan->scheduler_backend_count() == 1);
    SYNTH_TEST_CHECK(plan->scheduler_device(0) == cpu);
    SYNTH_TEST_CHECK(plan->scheduler_device(1) == nullptr);

    ggml_init_params parameters{};
    parameters.mem_size    = 1024 * 1024;
    parameters.no_alloc    = true;
    ggml_context * context = ggml_init(parameters);
    SYNTH_TEST_CHECK(context != nullptr);
    ggml_tensor * left  = ggml_new_tensor_1d(context, GGML_TYPE_F32, 1);
    ggml_tensor * right = ggml_new_tensor_1d(context, GGML_TYPE_F32, 1);
    ggml_tensor * sum   = ggml_add(context, left, right);
    ggml_cgraph * graph = ggml_new_graph_custom(context, 16, false);
    ggml_build_forward_expand(graph, sum);

    ggml_backend_sched_t scheduler = plan->create_scheduler(16);
    SYNTH_TEST_CHECK(scheduler != nullptr);
    SYNTH_TEST_CHECK(!plan->assign_to_primary(nullptr, sum));
    SYNTH_TEST_CHECK(!plan->assign_to_primary(scheduler, nullptr));
    SYNTH_TEST_CHECK(plan->assign_to_primary(scheduler, sum));
    SYNTH_TEST_CHECK(ggml_backend_sched_alloc_graph(scheduler, graph));
    const synth::BackendPlacement placement = plan->inspect_placement(scheduler, graph);
    SYNTH_TEST_CHECK(placement.node_count == 1);
    SYNTH_TEST_CHECK(placement.primary_node_count == 1);
    SYNTH_TEST_CHECK(placement.cpu_fallback_node_count == 0);
    SYNTH_TEST_CHECK(placement.accelerator_node_count == 0);
    SYNTH_TEST_CHECK(placement.other_node_count == 0);
    SYNTH_TEST_CHECK(placement.unassigned_node_count == 0);
    SYNTH_TEST_CHECK(placement.split_count == 1);
    const float left_value  = 1.25f;
    const float right_value = 2.75f;
    ggml_backend_tensor_set(left, &left_value, 0, sizeof(left_value));
    ggml_backend_tensor_set(right, &right_value, 0, sizeof(right_value));
    plan->set_threads(2);
    SYNTH_TEST_CHECK(ggml_backend_sched_graph_compute(scheduler, graph) == GGML_STATUS_SUCCESS);
    float sum_value = 0.0f;
    ggml_backend_tensor_get(sum, &sum_value, 0, sizeof(sum_value));
    SYNTH_TEST_CHECK(std::fabs(sum_value - 4.0f) < 1.0e-6f);
    ggml_backend_sched_free(scheduler);
    ggml_free(context);

    ggml_context * view_context = ggml_init(parameters);
    SYNTH_TEST_CHECK(view_context != nullptr);
    ggml_tensor * view_input = ggml_new_tensor_2d(view_context, GGML_TYPE_F32, 2, 3);
    ggml_set_input(view_input);
    ggml_tensor * transpose  = ggml_transpose(view_context, view_input);
    ggml_tensor * contiguous = ggml_cont(view_context, transpose);
    ggml_cgraph * view_graph = ggml_new_graph_custom(view_context, 16, false);
    ggml_build_forward_expand(view_graph, contiguous);
    ggml_backend_sched_t view_scheduler = plan->create_scheduler(16);
    SYNTH_TEST_CHECK(view_scheduler != nullptr);
    SYNTH_TEST_CHECK(ggml_backend_sched_alloc_graph(view_scheduler, view_graph));
    const synth::BackendPlacement view_placement = plan->inspect_placement(view_scheduler, view_graph);
    SYNTH_TEST_CHECK(view_placement.node_count == 2);
    SYNTH_TEST_CHECK(view_placement.view_node_count == 1);
    SYNTH_TEST_CHECK(view_placement.primary_node_count == 1);
    SYNTH_TEST_CHECK(view_placement.cpu_fallback_node_count == 0);
    ggml_backend_sched_free(view_scheduler);
    ggml_free(view_context);

    std::unique_ptr<synth::BackendPlan> accelerated;
    SYNTH_TEST_CHECK(synth::BackendPlan::create(cpu, true, accelerated) == SYNTH_OK);
    SYNTH_TEST_CHECK(accelerated != nullptr && accelerated->primary_device() == cpu);
    SYNTH_TEST_CHECK(accelerated->scheduler_backend_count() >= 1);
    const size_t last = accelerated->scheduler_backend_count() - 1;
    SYNTH_TEST_CHECK(accelerated->scheduler_device(last) == cpu);
    for (size_t i = 0; i < last; ++i) {
        SYNTH_TEST_CHECK(ggml_backend_dev_type(accelerated->scheduler_device(i)) == GGML_BACKEND_DEVICE_TYPE_ACCEL);
    }

    ggml_backend_dev_t cuda = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t candidate     = ggml_backend_dev_get(i);
        ggml_backend_reg_t registry      = ggml_backend_dev_backend_reg(candidate);
        const char *       registry_name = registry == nullptr ? nullptr : ggml_backend_reg_name(registry);
        if ((ggml_backend_dev_type(candidate) == GGML_BACKEND_DEVICE_TYPE_GPU ||
             ggml_backend_dev_type(candidate) == GGML_BACKEND_DEVICE_TYPE_IGPU) &&
            registry_name != nullptr && std::strncmp(registry_name, "CUDA", 4) == 0) {
            cuda = candidate;
            break;
        }
    }
    if (cuda != nullptr) {
        std::unique_ptr<synth::BackendPlan> gpu;
        SYNTH_TEST_CHECK(synth::BackendPlan::create(cuda, false, gpu) == SYNTH_OK);
        SYNTH_TEST_CHECK(gpu != nullptr && gpu->primary_device() == cuda);
        SYNTH_TEST_CHECK(gpu->scheduler_backend_count() == 2);
        SYNTH_TEST_CHECK(gpu->scheduler_device(0) == cuda);
        SYNTH_TEST_CHECK(ggml_backend_dev_type(gpu->scheduler_device(1)) == GGML_BACKEND_DEVICE_TYPE_CPU);

        ggml_context * gpu_context = ggml_init(parameters);
        SYNTH_TEST_CHECK(gpu_context != nullptr);
        ggml_tensor * gpu_left  = ggml_new_tensor_1d(gpu_context, GGML_TYPE_F32, 1);
        ggml_tensor * gpu_right = ggml_new_tensor_1d(gpu_context, GGML_TYPE_F32, 1);
        ggml_set_input(gpu_left);
        ggml_set_input(gpu_right);
        ggml_tensor * gpu_sum   = ggml_add(gpu_context, gpu_left, gpu_right);
        ggml_cgraph * gpu_graph = ggml_new_graph_custom(gpu_context, 16, false);
        ggml_build_forward_expand(gpu_graph, gpu_sum);
        ggml_backend_sched_t gpu_scheduler = gpu->create_scheduler(16);
        SYNTH_TEST_CHECK(gpu_scheduler != nullptr);
        SYNTH_TEST_CHECK(gpu->assign_to_primary(gpu_scheduler, gpu_sum));
        SYNTH_TEST_CHECK(ggml_backend_sched_alloc_graph(gpu_scheduler, gpu_graph));
        const synth::BackendPlacement gpu_placement = gpu->inspect_placement(gpu_scheduler, gpu_graph);
        SYNTH_TEST_CHECK(gpu_placement.node_count == 1);
        SYNTH_TEST_CHECK(gpu_placement.primary_node_count == 1);
        SYNTH_TEST_CHECK(gpu_placement.cpu_fallback_node_count == 0);
        ggml_backend_sched_free(gpu_scheduler);
        ggml_free(gpu_context);

        constexpr int64_t matrix_size       = 64;
        ggml_context *    precision_context = ggml_init(parameters);
        SYNTH_TEST_CHECK(precision_context != nullptr);
        ggml_tensor * precision_left  = ggml_new_tensor_2d(precision_context, GGML_TYPE_F32, matrix_size, matrix_size);
        ggml_tensor * precision_right = ggml_new_tensor_2d(precision_context, GGML_TYPE_F32, matrix_size, matrix_size);
        ggml_set_input(precision_left);
        ggml_set_input(precision_right);
        ggml_tensor * precision_product = ggml_mul_mat(precision_context, precision_left, precision_right);
        ggml_cgraph * precision_graph   = ggml_new_graph_custom(precision_context, 16, false);
        ggml_build_forward_expand(precision_graph, precision_product);
        ggml_backend_sched_t precision_scheduler = gpu->create_scheduler(16);
        SYNTH_TEST_CHECK(precision_scheduler != nullptr);
        SYNTH_TEST_CHECK(gpu->assign_to_primary(precision_scheduler, precision_product));
        SYNTH_TEST_CHECK(ggml_backend_sched_alloc_graph(precision_scheduler, precision_graph));
        const std::vector<float> precision_input(matrix_size * matrix_size, 1.0001f);
        ggml_backend_tensor_set(precision_left, precision_input.data(), 0, precision_input.size() * sizeof(float));
        ggml_backend_tensor_set(precision_right, precision_input.data(), 0, precision_input.size() * sizeof(float));
        SYNTH_TEST_CHECK(ggml_backend_sched_graph_compute(precision_scheduler, precision_graph) == GGML_STATUS_SUCCESS);
        float precision_value = 0.0f;
        ggml_backend_tensor_get(precision_product, &precision_value, 0, sizeof(precision_value));
        // CUDA F32 matrix multiplies compute at TF32 precision: cuBLAS runs in
        // CUBLAS_TF32_TENSOR_OP_MATH and ggml's mmf tile path uses tf32 MMA, so
        // the bound here is TF32's ~1e-3 relative error with margin rather than
        // FP32's ~1e-7. It stays a real check: a misplaced or broken matmul is
        // wrong by orders of magnitude, not by one part in a thousand.
        const float precision_expected = matrix_size * 1.0001f * 1.0001f;
        SYNTH_TEST_CHECK(std::isfinite(precision_value));
        SYNTH_TEST_CHECK(std::fabs(precision_value - precision_expected) < 5.0e-3f * precision_expected);
        ggml_backend_sched_free(precision_scheduler);
        ggml_free(precision_context);

        // A stage whose output is discrete is held on CPU deliberately, so that a
        // rounded integer cannot depend on which backend produced the value it was
        // rounded from. create_cpu_scheduler is what holds it there.
        //
        // The single-split assertion is the load-bearing one. Forcing nodes onto
        // CPU while their operands stay in the primary buffer also places the work
        // on CPU, but it cost 2,372 splits on Kokoro's duration graph and ran five
        // times slower end to end than leaving the stage on the GPU. A CPU-only
        // scheduler over CPU-resident operands is one split; a regression that
        // reintroduced cross-backend operands would show up here as more.
        SYNTH_TEST_CHECK(gpu->cpu_backend() != nullptr);
        SYNTH_TEST_CHECK(ggml_backend_dev_type(ggml_backend_get_device(gpu->cpu_backend())) ==
                         GGML_BACKEND_DEVICE_TYPE_CPU);
        SYNTH_TEST_CHECK(gpu->cpu_backend() != gpu->primary());
        SYNTH_TEST_CHECK(gpu->create_cpu_scheduler(0) == nullptr);

        ggml_context * held_context = ggml_init(parameters);
        SYNTH_TEST_CHECK(held_context != nullptr);
        ggml_tensor * held_left  = ggml_new_tensor_2d(held_context, GGML_TYPE_F32, 4, 4);
        ggml_tensor * held_right = ggml_new_tensor_2d(held_context, GGML_TYPE_F32, 4, 4);
        ggml_set_input(held_left);
        ggml_set_input(held_right);
        ggml_tensor * held_product = ggml_mul_mat(held_context, held_left, held_right);
        ggml_tensor * held_scaled  = ggml_scale(held_context, held_product, 2.0f);
        ggml_cgraph * held_graph   = ggml_new_graph_custom(held_context, 16, false);
        ggml_build_forward_expand(held_graph, held_scaled);

        ggml_backend_sched_t held_scheduler = gpu->create_cpu_scheduler(32);
        SYNTH_TEST_CHECK(held_scheduler != nullptr);
        SYNTH_TEST_CHECK(ggml_backend_sched_alloc_graph(held_scheduler, held_graph));
        const synth::BackendPlacement held = gpu->inspect_placement(held_scheduler, held_graph);
        SYNTH_TEST_CHECK(held.node_count == 2);
        SYNTH_TEST_CHECK(held.primary_node_count == 0);
        SYNTH_TEST_CHECK(held.cpu_fallback_node_count == 2);
        SYNTH_TEST_CHECK(held.accelerator_node_count == 0);
        SYNTH_TEST_CHECK(held.unassigned_node_count == 0);
        SYNTH_TEST_CHECK(held.split_count == 1);

        // And it still computes: the values come back exact, because CPU F32 has
        // no TF32 tile path to fall into.
        const std::vector<float> held_input(16, 1.0f);
        ggml_backend_tensor_set(held_left, held_input.data(), 0, held_input.size() * sizeof(float));
        ggml_backend_tensor_set(held_right, held_input.data(), 0, held_input.size() * sizeof(float));
        SYNTH_TEST_CHECK(ggml_backend_sched_graph_compute(held_scheduler, held_graph) == GGML_STATUS_SUCCESS);
        float held_value = 0.0f;
        ggml_backend_tensor_get(held_scaled, &held_value, 0, sizeof(held_value));
        SYNTH_TEST_CHECK(held_value == 8.0f);

        ggml_backend_sched_free(held_scheduler);
        ggml_free(held_context);
    }

    // With CPU as the primary there is no boundary to hold anything away from, so
    // the CPU backend a plan reports is the primary itself.
    SYNTH_TEST_CHECK(plan->cpu_backend() == plan->primary());
    return 0;
}
