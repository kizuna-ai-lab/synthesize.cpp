#pragma once

#include "ggml-backend.h"
#include "synthesize.h"

#include <cstdint>

namespace synth {

enum class BackendKind {
    Unknown = 0,
    Cpu,
    Accel,
    Cuda,
    Metal,
    Vulkan,
    Sycl,
    OtherGpu,
};

const char * backend_kind_name(BackendKind kind);
BackendKind classify_backend_type(enum ggml_backend_dev_type device_type, const char * registry_name);
BackendKind classify_backend_device(ggml_backend_dev_t device);
synth_device_type_t public_device_type(enum ggml_backend_dev_type device_type);
synth_device_flags_t device_memory_flags(BackendKind                  kind,
                                         enum ggml_backend_dev_type device_type,
                                         uint64_t                    memory_total);

uint32_t backend_device_count();
synth_status_t get_backend_device(uint32_t index, uint64_t caller_size, synth_backend_device_t * output);
synth_status_t get_backend_device(ggml_backend_dev_t device,
                                  uint64_t           caller_size,
                                  synth_backend_device_t * output);
synth_bool_t backend_available(synth_backend_request_t request);

// Resolution honors the public global registry index exactly, so a strict
// backend request cannot silently substitute a different device or backend.
synth_status_t resolve_cpu_device(synth_backend_request_t request,
                                  int32_t                 requested_index,
                                  ggml_backend_dev_t *    output);
synth_status_t resolve_requested_device(synth_backend_request_t request,
                                        int32_t                 requested_index,
                                        ggml_backend_dev_t *    output);

}  // namespace synth
