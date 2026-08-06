// Task 8 (Plan 4): OmniVoice must not claim an Execution Backend it does not
// run. src/synthesize.cpp's synth_model_load used to compute
// `backend_supported` as one family-independent expression accepting
// AUTO/CPU/CPU_ACCEL/CUDA for every family, with a comment noting that when
// families diverge this becomes a per-family question -- they have: OmniVoice
// places every graph on `create_cpu_scheduler` over one CPU-resident weights
// buffer with no accelerator twin (src/arch/omnivoice/model.cpp's "Placement:
// everything is CPU ... One CPU buffer, no twin"), so an explicit CUDA
// request against it must be refused rather than silently loaded onto CPU
// while `synth_model_get_device` reports CUDA
// (docs/backends.md: "a backend that is present is not a backend that ran").
//
// This runs entirely against a synthetic package through the PUBLIC
// synth_model_load seam (the function that actually contained the defect),
// not the family's internal Model::load_cpu -- and it must hold regardless of
// whether the test machine has a CUDA device at all, because the refusal
// happens before device resolution is even attempted.

#include "arch/omnivoice/omnivoice.h"
#include "omnivoice_synthetic_package.h"
#include "synthesize.h"
#include "test-assert.h"

#include <cstdint>
#include <cstdio>
#include <string>

namespace {

struct SeenDiagnostic {
    synth_status_t status = SYNTH_OK;
    std::string    code;
    bool           seen = false;
};

void SYNTH_CALL record_diagnostic(void * user_data, const synth_diagnostic_t * diagnostic) {
    auto * seen  = static_cast<SeenDiagnostic *>(user_data);
    seen->status = diagnostic->status;
    seen->code.assign(diagnostic->code, static_cast<size_t>(diagnostic->code_size));
    seen->seen = true;
}

synth_diagnostic_sink_t make_sink(SeenDiagnostic & target) {
    synth_diagnostic_sink_t sink;
    synth_diagnostic_sink_init(&sink, sizeof(sink));
    sink.emit      = record_diagnostic;
    sink.user_data = &target;
    return sink;
}

// An explicit CUDA request is refused with SYNTH_ERR_BACKEND and a named
// diagnostic, never silently downgraded to CPU.
int check_explicit_cuda_refused(const std::string & package_path) {
    SeenDiagnostic                diagnostic;
    const synth_diagnostic_sink_t sink = make_sink(diagnostic);
    synth_model_load_params_t     params;
    synth_model_load_params_init(&params, sizeof(params));
    params.backend        = SYNTH_BACKEND_CUDA;
    params.diagnostics    = &sink;
    synth_model_t * model = reinterpret_cast<synth_model_t *>(uintptr_t(1));
    SYNTH_TEST_CHECK(synth_model_load(package_path.c_str(), &params, &model) == SYNTH_ERR_BACKEND);
    SYNTH_TEST_CHECK(model == nullptr);
    SYNTH_TEST_CHECK(diagnostic.seen);
    SYNTH_TEST_CHECK(diagnostic.status == SYNTH_ERR_BACKEND);
    SYNTH_TEST_CHECK(diagnostic.code == "backend.unavailable");
    return 0;
}

// AUTO means "pick something sensible, CPU is sensible" -- it must keep
// working for a CPU-only family rather than being caught by the same gate
// that refuses an EXPLICIT CUDA request.
int check_auto_still_loads(const std::string & package_path) {
    synth_model_load_params_t params;
    synth_model_load_params_init(&params, sizeof(params));
    params.backend        = SYNTH_BACKEND_AUTO;
    synth_model_t * model = nullptr;
    SYNTH_TEST_CHECK(synth_model_load(package_path.c_str(), &params, &model) == SYNTH_OK);
    SYNTH_TEST_CHECK(model != nullptr);
    synth_backend_device_t device;
    synth_backend_device_init(&device, sizeof(device));
    SYNTH_TEST_CHECK(synth_model_get_device(model, &device) == SYNTH_OK);
    SYNTH_TEST_CHECK(device.device_type == SYNTH_DEVICE_TYPE_CPU);
    synth_model_free(model);
    return 0;
}

// CPU_ACCEL keeps CPU as the primary backend and only ever adds optional
// host-memory accelerators (docs/backends.md), so it makes the same
// "everything on CPU" promise OmniVoice already keeps and must be accepted.
int check_cpu_accel_still_loads(const std::string & package_path) {
    synth_model_load_params_t params;
    synth_model_load_params_init(&params, sizeof(params));
    params.backend        = SYNTH_BACKEND_CPU_ACCEL;
    synth_model_t * model = nullptr;
    SYNTH_TEST_CHECK(synth_model_load(package_path.c_str(), &params, &model) == SYNTH_OK);
    SYNTH_TEST_CHECK(model != nullptr);
    synth_model_free(model);
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <scratch-dir>\n", argv[0]);
        return 2;
    }
    const std::string package_path = std::string(argv[1]) + "/synthetic-backend.gguf";
    SYNTH_TEST_CHECK(synth::omnivoice::testing::write_synthetic_package(package_path, {}));

    int failures = 0;
    failures += check_explicit_cuda_refused(package_path);
    failures += check_auto_still_loads(package_path);
    failures += check_cpu_accel_still_loads(package_path);
    return failures == 0 ? 0 : 1;
}
