// Task 8 (Plan 4): OmniVoice must not claim an Execution Backend it does not
// run. src/synthesize.cpp's synth_model_load used to compute
// `backend_supported` as one family-independent expression accepting
// AUTO/CPU/CPU_ACCEL/CUDA for every family, with a comment noting that when
// families diverge this becomes a per-family question -- they have.
//
// Task 11 flips the claim, after the evidence: `family_supports_explicit_
// backend`'s `Omnivoice` case (src/model-info.h) now returns true for
// `SYNTH_BACKEND_CUDA`, once the replay sweep proved every codec node reaches
// the device and every generator node stays on the CPU across all twenty
// golden cases. This test used to assert the OLD claim -- an explicit CUDA
// request refused unconditionally, before device resolution was even
// attempted, which is why it held regardless of whether the test machine had
// a CUDA device at all. That premise is gone. OmniVoice now claims CUDA
// exactly like VITS, Kokoro, and Qwen3-TTS, so an explicit CUDA request
// against a synthetic package reaches ordinary DEVICE resolution next --
// mirroring tests/vits_public_lifecycle_test.cpp's own CUDA branch: SYNTH_OK
// with a real device on a build that has one, SYNTH_ERR_BACKEND and
// "backend.device_unavailable" (not the old "backend.unavailable") on a
// build that does not. The synthetic package's every codec tensor already
// carries a real F32 payload (omnivoice_synthetic_package.h), so it needs no
// option flip to survive Model::load's twin-mirroring copy onto whatever
// device this build resolves.
//
// The CUDA-available branch is this suite's positive assertion (accumulated
// requirement 1) at the cheapest layer that can give it: a real load that
// asks for CUDA and gets back CUDA, checked directly via
// synth_model_get_device rather than inferred from SYNTH_OK alone -- a
// forgotten claim flip and a correct refusal both fail this same call, but
// only a correct flip also reports "cuda" back.
//
// This runs entirely against a synthetic package through the PUBLIC
// synth_model_load seam (the function the Task 8 defect actually lived in),
// not the family's internal Model::load_cpu.

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

// An explicit CUDA request now reaches device resolution rather than being
// refused at the family-claim gate (Task 11): SYNTH_OK and a real CUDA device
// on a build that has one, SYNTH_ERR_BACKEND / "backend.device_unavailable"
// on a build that does not -- never silently downgraded to CPU either way.
int check_explicit_cuda(const std::string & package_path) {
    SeenDiagnostic                diagnostic;
    const synth_diagnostic_sink_t sink = make_sink(diagnostic);
    synth_model_load_params_t     params;
    synth_model_load_params_init(&params, sizeof(params));
    params.backend              = SYNTH_BACKEND_CUDA;
    params.diagnostics          = &sink;
    synth_model_t *      model  = nullptr;
    const synth_status_t status = synth_model_load(package_path.c_str(), &params, &model);
    if (synth_backend_available(SYNTH_BACKEND_CUDA) == SYNTH_TRUE) {
        SYNTH_TEST_CHECK(status == SYNTH_OK);
        SYNTH_TEST_CHECK(model != nullptr);
        synth_backend_device_t device;
        synth_backend_device_init(&device, sizeof(device));
        SYNTH_TEST_CHECK(synth_model_get_device(model, &device) == SYNTH_OK);
        SYNTH_TEST_CHECK(device.kind != nullptr);
        SYNTH_TEST_CHECK(std::string(device.kind) == "cuda");
        synth_model_free(model);
    } else {
        SYNTH_TEST_CHECK(status == SYNTH_ERR_BACKEND);
        SYNTH_TEST_CHECK(model == nullptr);
        SYNTH_TEST_CHECK(diagnostic.seen);
        SYNTH_TEST_CHECK(diagnostic.status == SYNTH_ERR_BACKEND);
        SYNTH_TEST_CHECK(diagnostic.code == "backend.device_unavailable");
    }
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
    failures += check_explicit_cuda(package_path);
    failures += check_auto_still_loads(package_path);
    failures += check_cpu_accel_still_loads(package_path);
    return failures == 0 ? 0 : 1;
}
