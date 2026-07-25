#pragma once

#include "arch/vits/vits.h"
#include "backend-device.h"

#include <cstring>
#include <memory>

inline synth_status_t load_vits_runner_model(const char *                          path,
                                             const char *                          backend,
                                             std::unique_ptr<synth::vits::Model> & output) {
    if (backend == nullptr || std::strcmp(backend, "cpu") == 0) {
        return synth::vits::Model::load_cpu(path, output);
    }
    if (std::strcmp(backend, "cuda") != 0) {
        output.reset();
        return SYNTH_ERR_INVALID_ARG;
    }
    ggml_backend_dev_t   device = nullptr;
    const synth_status_t status = synth::resolve_requested_device(SYNTH_BACKEND_CUDA, -1, &device);
    if (status != SYNTH_OK) {
        output.reset();
        return status;
    }
    return synth::vits::Model::load(path, device, true, output);
}
