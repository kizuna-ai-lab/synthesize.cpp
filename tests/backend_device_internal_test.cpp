#include "backend-device.h"
#include "test-assert.h"

#include <cstring>

int main() {
    using synth::BackendKind;

    SYNTH_TEST_CHECK(synth::classify_backend_type(GGML_BACKEND_DEVICE_TYPE_CPU, "CUDA") == BackendKind::Cpu);
    SYNTH_TEST_CHECK(synth::classify_backend_type(GGML_BACKEND_DEVICE_TYPE_ACCEL, "BLAS") == BackendKind::Accel);
    SYNTH_TEST_CHECK(synth::classify_backend_type(GGML_BACKEND_DEVICE_TYPE_GPU, "CUDA0") == BackendKind::Cuda);
    SYNTH_TEST_CHECK(synth::classify_backend_type(GGML_BACKEND_DEVICE_TYPE_IGPU, "MTL") == BackendKind::Metal);
    SYNTH_TEST_CHECK(synth::classify_backend_type(GGML_BACKEND_DEVICE_TYPE_GPU, "Metal") == BackendKind::Metal);
    SYNTH_TEST_CHECK(synth::classify_backend_type(GGML_BACKEND_DEVICE_TYPE_GPU, "Vulkan0") == BackendKind::Vulkan);
    SYNTH_TEST_CHECK(synth::classify_backend_type(GGML_BACKEND_DEVICE_TYPE_GPU, "SYCL0") == BackendKind::Sycl);
    SYNTH_TEST_CHECK(synth::classify_backend_type(GGML_BACKEND_DEVICE_TYPE_GPU, "RPC") == BackendKind::OtherGpu);
    SYNTH_TEST_CHECK(synth::classify_backend_type(GGML_BACKEND_DEVICE_TYPE_META, "META") == BackendKind::Unknown);
    SYNTH_TEST_CHECK(std::strcmp(synth::backend_kind_name(BackendKind::OtherGpu), "gpu") == 0);
    SYNTH_TEST_CHECK(std::strcmp(synth::backend_kind_name(BackendKind::Unknown), "unknown") == 0);

    SYNTH_TEST_CHECK(synth::public_device_type(GGML_BACKEND_DEVICE_TYPE_CPU) == SYNTH_DEVICE_TYPE_CPU);
    SYNTH_TEST_CHECK(synth::public_device_type(GGML_BACKEND_DEVICE_TYPE_GPU) == SYNTH_DEVICE_TYPE_GPU);
    SYNTH_TEST_CHECK(synth::public_device_type(GGML_BACKEND_DEVICE_TYPE_IGPU) == SYNTH_DEVICE_TYPE_IGPU);
    SYNTH_TEST_CHECK(synth::public_device_type(GGML_BACKEND_DEVICE_TYPE_ACCEL) == SYNTH_DEVICE_TYPE_ACCEL);
    SYNTH_TEST_CHECK(synth::public_device_type(GGML_BACKEND_DEVICE_TYPE_META) == SYNTH_DEVICE_TYPE_GPU);

    SYNTH_TEST_CHECK(synth::device_memory_flags(BackendKind::Cuda, GGML_BACKEND_DEVICE_TYPE_GPU, 0) == 0);
    SYNTH_TEST_CHECK(synth::device_memory_flags(BackendKind::Cuda, GGML_BACKEND_DEVICE_TYPE_GPU, 1) ==
                     SYNTH_DEVICE_MEMORY_INFO_VALID);
    SYNTH_TEST_CHECK(synth::device_memory_flags(BackendKind::Cpu, GGML_BACKEND_DEVICE_TYPE_CPU, 1) ==
                     (SYNTH_DEVICE_MEMORY_INFO_VALID | SYNTH_DEVICE_MEMORY_INFO_APPROXIMATE));
    SYNTH_TEST_CHECK(
        synth::device_memory_flags(BackendKind::Cuda, GGML_BACKEND_DEVICE_TYPE_IGPU, 1) ==
        (SYNTH_DEVICE_MEMORY_INFO_VALID | SYNTH_DEVICE_MEMORY_SHARED | SYNTH_DEVICE_MEMORY_INFO_APPROXIMATE));
    SYNTH_TEST_CHECK(
        synth::device_memory_flags(BackendKind::Metal, GGML_BACKEND_DEVICE_TYPE_GPU, 1) ==
        (SYNTH_DEVICE_MEMORY_INFO_VALID | SYNTH_DEVICE_MEMORY_SHARED | SYNTH_DEVICE_MEMORY_INFO_APPROXIMATE));

    ggml_backend_dev_t selected = nullptr;
    SYNTH_TEST_CHECK(synth::resolve_cpu_device(SYNTH_BACKEND_AUTO, -1, &selected) == SYNTH_OK);
    SYNTH_TEST_CHECK(selected != nullptr && ggml_backend_dev_type(selected) == GGML_BACKEND_DEVICE_TYPE_CPU);
    uint32_t cpu_index = UINT32_MAX;
    for (uint32_t i = 0; i < synth::backend_device_count(); ++i) {
        if (ggml_backend_dev_get(i) == selected) {
            cpu_index = i;
            break;
        }
    }
    SYNTH_TEST_CHECK(cpu_index != UINT32_MAX);
    selected = nullptr;
    SYNTH_TEST_CHECK(synth::resolve_cpu_device(SYNTH_BACKEND_CPU, static_cast<int32_t>(cpu_index), &selected) ==
                     SYNTH_OK);
    SYNTH_TEST_CHECK(selected == ggml_backend_dev_get(cpu_index));
    SYNTH_TEST_CHECK(synth::resolve_cpu_device(SYNTH_BACKEND_CPU, -2, &selected) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(synth::resolve_cpu_device(SYNTH_BACKEND_CPU, static_cast<int32_t>(synth::backend_device_count()),
                                               &selected) == SYNTH_ERR_BACKEND);
    SYNTH_TEST_CHECK(synth::resolve_cpu_device(SYNTH_BACKEND_CUDA, -1, &selected) == SYNTH_ERR_BACKEND);
    SYNTH_TEST_CHECK(synth::resolve_cpu_device(SYNTH_BACKEND_CPU, -1, nullptr) == SYNTH_ERR_INVALID_ARG);

    selected = nullptr;
    SYNTH_TEST_CHECK(synth::resolve_requested_device(SYNTH_BACKEND_CPU, -1, &selected) == SYNTH_OK);
    SYNTH_TEST_CHECK(selected != nullptr && synth::classify_backend_device(selected) == BackendKind::Cpu);
    const synth_status_t cuda_status = synth::resolve_requested_device(SYNTH_BACKEND_CUDA, -1, &selected);
    if (synth::backend_available(SYNTH_BACKEND_CUDA) == SYNTH_TRUE) {
        SYNTH_TEST_CHECK(cuda_status == SYNTH_OK);
        SYNTH_TEST_CHECK(selected != nullptr && synth::classify_backend_device(selected) == BackendKind::Cuda);
        SYNTH_TEST_CHECK(synth::resolve_requested_device(SYNTH_BACKEND_CUDA, static_cast<int32_t>(cpu_index),
                                                         &selected) == SYNTH_ERR_BACKEND);
    } else {
        SYNTH_TEST_CHECK(cuda_status == SYNTH_ERR_BACKEND && selected == nullptr);
    }
    return 0;
}
