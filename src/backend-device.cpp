#include "backend-device.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>

namespace synth {

namespace {

// The GGML registry-name classification follows transcribe.cpp's MIT-licensed
// backend wrapper. See THIRD_PARTY_NOTICES.md for attribution.
bool starts_with(const char * value, const char * prefix) {
    return value != nullptr && prefix != nullptr && std::strncmp(value, prefix, std::strlen(prefix)) == 0;
}

template <typename T> void copy_visible(T * output, const T & staged, uint64_t caller_size) {
    std::memcpy(output, &staged, std::min(static_cast<size_t>(caller_size), sizeof(T)));
}

}  // namespace

const char * backend_kind_name(BackendKind kind) {
    switch (kind) {
        case BackendKind::Cpu:
            return "cpu";
        case BackendKind::Accel:
            return "accel";
        case BackendKind::Cuda:
            return "cuda";
        case BackendKind::Metal:
            return "metal";
        case BackendKind::Vulkan:
            return "vulkan";
        case BackendKind::Sycl:
            return "sycl";
        case BackendKind::OtherGpu:
            return "gpu";
        case BackendKind::Unknown:
        default:
            return "unknown";
    }
}

BackendKind classify_backend_type(enum ggml_backend_dev_type device_type, const char * registry_name) {
    if (device_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return BackendKind::Cpu;
    }
    if (device_type == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
        return BackendKind::Accel;
    }
    if (device_type != GGML_BACKEND_DEVICE_TYPE_GPU && device_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
        return BackendKind::Unknown;
    }
    if (starts_with(registry_name, "CUDA")) {
        return BackendKind::Cuda;
    }
    if (starts_with(registry_name, "MTL") || starts_with(registry_name, "Metal")) {
        return BackendKind::Metal;
    }
    if (starts_with(registry_name, "Vulkan")) {
        return BackendKind::Vulkan;
    }
    if (starts_with(registry_name, "SYCL")) {
        return BackendKind::Sycl;
    }
    return BackendKind::OtherGpu;
}

BackendKind classify_backend_device(ggml_backend_dev_t device) {
    if (device == nullptr) {
        return BackendKind::Unknown;
    }
    ggml_backend_reg_t registry      = ggml_backend_dev_backend_reg(device);
    const char *       registry_name = registry == nullptr ? nullptr : ggml_backend_reg_name(registry);
    return classify_backend_type(ggml_backend_dev_type(device), registry_name);
}

synth_device_type_t public_device_type(enum ggml_backend_dev_type device_type) {
    switch (device_type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:
            return SYNTH_DEVICE_TYPE_CPU;
        case GGML_BACKEND_DEVICE_TYPE_IGPU:
            return SYNTH_DEVICE_TYPE_IGPU;
        case GGML_BACKEND_DEVICE_TYPE_ACCEL:
            return SYNTH_DEVICE_TYPE_ACCEL;
        case GGML_BACKEND_DEVICE_TYPE_GPU:
        default:
            return SYNTH_DEVICE_TYPE_GPU;
    }
}

synth_device_flags_t device_memory_flags(BackendKind                kind,
                                         enum ggml_backend_dev_type device_type,
                                         uint64_t                   memory_total) {
    if (memory_total == 0) {
        return 0;
    }
    synth_device_flags_t flags = SYNTH_DEVICE_MEMORY_INFO_VALID;
    if (device_type == GGML_BACKEND_DEVICE_TYPE_IGPU || kind == BackendKind::Metal || kind == BackendKind::Accel) {
        flags |= SYNTH_DEVICE_MEMORY_SHARED | SYNTH_DEVICE_MEMORY_INFO_APPROXIMATE;
    } else if (device_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
        // GGML's CPU value is system RAM; on POSIX its current "free" value
        // is deliberately an approximation equal to total RAM.
        flags |= SYNTH_DEVICE_MEMORY_INFO_APPROXIMATE;
    }
    return flags;
}

uint32_t backend_device_count() {
    const size_t count = ggml_backend_dev_count();
    return count > std::numeric_limits<uint32_t>::max() ? std::numeric_limits<uint32_t>::max() :
                                                          static_cast<uint32_t>(count);
}

synth_status_t get_backend_device(ggml_backend_dev_t device, uint64_t caller_size, synth_backend_device_t * output) {
    if (device == nullptr || output == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (caller_size < sizeof(uint64_t)) {
        return SYNTH_ERR_BAD_STRUCT_SIZE;
    }

    ggml_backend_dev_props properties{};
    ggml_backend_dev_get_props(device, &properties);
    const BackendKind kind      = classify_backend_device(device);
    const char *      kind_name = backend_kind_name(kind);
    const char *      name      = ggml_backend_dev_name(device);
    if (name == nullptr || name[0] == '\0') {
        name = properties.name != nullptr && properties.name[0] != '\0' ? properties.name : kind_name;
    }
    const char * description = ggml_backend_dev_description(device);
    if (description == nullptr) {
        description = properties.description == nullptr ? "" : properties.description;
    }

    const auto                 type = public_device_type(properties.type);
    const synth_device_flags_t flags =
        device_memory_flags(kind, properties.type, static_cast<uint64_t>(properties.memory_total));
    synth_backend_device_t staged{};
    staged.struct_size = caller_size;
    staged.name        = name;
    staged.description = description;
    staged.kind        = kind_name;
    staged.device_id   = properties.device_id;
    staged.memory_total =
        (flags & SYNTH_DEVICE_MEMORY_INFO_VALID) == 0 ? 0 : static_cast<uint64_t>(properties.memory_total);
    staged.memory_free =
        (flags & SYNTH_DEVICE_MEMORY_INFO_VALID) == 0 ? 0 : static_cast<uint64_t>(properties.memory_free);
    staged.device_type = type;
    staged.flags       = flags;
    copy_visible(output, staged, caller_size);
    return SYNTH_OK;
}

synth_status_t get_backend_device(uint32_t index, uint64_t caller_size, synth_backend_device_t * output) {
    if (output == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (caller_size < sizeof(uint64_t)) {
        return SYNTH_ERR_BAD_STRUCT_SIZE;
    }
    if (static_cast<size_t>(index) >= ggml_backend_dev_count()) {
        return SYNTH_ERR_INVALID_ARG;
    }
    return get_backend_device(ggml_backend_dev_get(index), caller_size, output);
}

synth_bool_t backend_available(synth_backend_request_t request) {
    const size_t count = ggml_backend_dev_count();
    if (request == SYNTH_BACKEND_AUTO) {
        return count == 0 ? SYNTH_FALSE : SYNTH_TRUE;
    }

    BackendKind wanted = BackendKind::Unknown;
    switch (request) {
        case SYNTH_BACKEND_CPU:
        case SYNTH_BACKEND_CPU_ACCEL:
            wanted = BackendKind::Cpu;
            break;
        case SYNTH_BACKEND_CUDA:
            wanted = BackendKind::Cuda;
            break;
        case SYNTH_BACKEND_METAL:
            wanted = BackendKind::Metal;
            break;
        case SYNTH_BACKEND_VULKAN:
            wanted = BackendKind::Vulkan;
            break;
        default:
            return SYNTH_FALSE;
    }
    for (size_t i = 0; i < count; ++i) {
        if (classify_backend_device(ggml_backend_dev_get(i)) == wanted) {
            return SYNTH_TRUE;
        }
    }
    return SYNTH_FALSE;
}

synth_status_t resolve_cpu_device(synth_backend_request_t request,
                                  int32_t                 requested_index,
                                  ggml_backend_dev_t *    output) {
    if (request != SYNTH_BACKEND_AUTO && request != SYNTH_BACKEND_CPU && request != SYNTH_BACKEND_CPU_ACCEL) {
        if (output != nullptr) {
            *output = nullptr;
        }
        return SYNTH_ERR_BACKEND;
    }
    return resolve_requested_device(request, requested_index, output);
}

synth_status_t resolve_requested_device(synth_backend_request_t request,
                                        int32_t                 requested_index,
                                        ggml_backend_dev_t *    output) {
    if (output == nullptr || requested_index < -1) {
        return SYNTH_ERR_INVALID_ARG;
    }
    *output = nullptr;

    BackendKind wanted = BackendKind::Unknown;
    switch (request) {
        case SYNTH_BACKEND_AUTO:
        case SYNTH_BACKEND_CPU:
        case SYNTH_BACKEND_CPU_ACCEL:
            // AUTO remains CPU-conservative until each Model Family has a
            // validated GPU preference. Explicit GPU requests are separate.
            wanted = BackendKind::Cpu;
            break;
        case SYNTH_BACKEND_CUDA:
            wanted = BackendKind::Cuda;
            break;
        case SYNTH_BACKEND_METAL:
            wanted = BackendKind::Metal;
            break;
        case SYNTH_BACKEND_VULKAN:
            wanted = BackendKind::Vulkan;
            break;
        default:
            return SYNTH_ERR_INVALID_ARG;
    }

    if (requested_index >= 0) {
        const size_t index = static_cast<size_t>(requested_index);
        if (index >= ggml_backend_dev_count()) {
            return SYNTH_ERR_BACKEND;
        }
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (classify_backend_device(device) != wanted) {
            return SYNTH_ERR_BACKEND;
        }
        *output = device;
        return SYNTH_OK;
    }

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (classify_backend_device(device) == wanted) {
            *output = device;
            return SYNTH_OK;
        }
    }
    return SYNTH_ERR_BACKEND;
}

}  // namespace synth
