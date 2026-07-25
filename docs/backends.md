# Execution Backend Policy

Status: Confirmed, last updated on 2026-07-26.

## Shared Inference Graph

Each model family has one backend-independent GGML inference implementation. CPU is the mandatory correctness baseline. CUDA, Metal, Vulkan, and future targets execute or offload the same semantic graph through GGML backend allocation, scheduling, and kernels; they do not receive separate model-family implementations.

Backend-specific kernels and tensor-placement policies belong to the backend layer. Hybrid CPU/device execution is permitted, but fallback and actual device placement must be observable and must not be presented as full GPU execution.

## v1 CUDA Baseline

The v1 CUDA Native Provider uses CUDA 13.3 Update 1 on Linux x86-64, Linux
AArch64, and Windows x86-64. On x86-64 its Runtime Support Floor is an NVIDIA GPU
with compute capability 7.5 or later and an R580-or-newer driver. This admits
Turing and newer architectures and excludes Maxwell, Pascal, and Volta from the
`cu13` Provider contract. On Linux AArch64, v1 supports DGX Spark/GB10 with
compute capability 12.1 and an R580-or-newer driver.

The x86-64 Native Cubin Target Set is `sm_75`, `sm_80`, `sm_86`, `sm_89`,
`sm_90`, `sm_100`, and `sm_120a`; the Linux AArch64 set is native `sm_121a` for
DGX Spark. Official builds use these explicit sets rather than the
compiler-dependent `all` or `all-major` meta-targets. A device is on the native
path when its driver selects a compatible cubin from its platform set. Optional
`compute_121a` PTX in the x86-64 Provider is experimental and does not establish
support. `sm_110` remains outside v1 because the AArch64 Provider does not target
Jetson.

Each release validates the R580 driver floor and the current release-driver branch
on physical hardware with complete model inference; compilation, device
enumeration, isolated operator success, or PTX JIT success is insufficient. Exact
GPU models, operating systems, driver builds, and target coverage are recorded in
the release's Validation Hardware Matrix rather than in the public C Interface.

CUDA builds default to strict FP32 cuBLAS math. `SYNTH_CUDA_TF32=ON` is an
explicit speed-over-precision experiment, not a release default. On the current
VITS F32 graph, ambient TF32 amplified the final PCM max-absolute difference from
`7.4365083e-4` to `1.1620114e-1`, while providing no material text-stage speedup
on GB10. The strict default is compile-time library policy so Rust, Python, and
other consumers do not need to mutate process-global CUDA environment variables.

The guarantee is scoped to cuBLAS, and the scope is a measured fact rather than
a drafting nicety. GGML's own tensor-core kernels (`mmf.cu`) take every F32
matrix multiply with 16 or fewer columns on Ampere-class or newer devices, and
they compute in TF32 (`mma.cuh`'s `mma...tf32` tiles) regardless of
`GGML_CUDA_DISABLE_TF32`, which reaches only the cuBLAS math mode. On the Kokoro
PL-BERT stage the cliff sits exactly at the kernel boundary — 2.1e-3 relative
CUDA-versus-CPU deviation at 16 input tokens, 1.9e-6 at 17 — and the vendored
GGML revision has no build option or environment variable that disables the
path. Consequences for short inputs are bounded by measurement, not assumption:
across every Kokoro profile and case the rounded durations stay exact and the
waveform correlation is unaffected, and each family's Golden suite is the
instrument that keeps that true. If a future upstream revision adds a gate for
these kernels, re-vendoring and enabling it restores the unscoped guarantee.

## CUDA Unified Memory Policy

The supported CUDA correctness path uses ordinary GGML CPU and device buffers and
the same scheduler semantics on every host. DGX Spark's physically unified memory
does not change tensor ownership, placement, copy, or lifetime rules and is not
observable through the public C Interface.

The standard `dev-dgx-spark` workflow and every Release preset leave GGML CUDA
unified-memory fallback disabled. A separate `dev-dgx-spark-uvm` workflow may run
dedicated processes with unified memory enabled before backend initialization for
large-model, memory-pressure, and performance research. Its results do not qualify
a model-family and backend combination as Supported, and it is not published as a
different Native Provider.

`GGML_CUDA_ENABLE_UNIFIED_MEMORY` remains an upstream implementation detail rather
than supported synthesize.cpp configuration. Because synthesize.cpp is primarily a
library, the Integration must not let an ambient setting silently change the
standard execution contract. v1 exposes no UVM switch, DGX-specific type, or
UVM-specific allocator. Any future backend-internal use requires DGX Spark
benchmarks, complete standard-CUDA correctness validation, and discrete-memory
regression testing on RTX 4070 SUPER.

Device discovery reports backend-visible total and free byte counts as advisory
snapshots with explicit validity, shared-memory, and approximate-value flags. A
shared physical memory flag describes hardware memory topology; it never implies
that CUDA unified-memory fallback is enabled. The Backend Integration owns this
translation and never uses `nvidia-smi`, operating-system command output, or the
public discovery snapshot as the sole model-admission decision.

## v1 Selection Policy

The public backend requests are `AUTO`, `CPU`, `CPU_ACCEL`, `CUDA`, `METAL`, and `VULKAN`. `AUTO` is the default. It considers only currently available Validated Backend Combinations, follows a library-defined device preference, and falls back to CPU; it is an automatic policy, not a promise to benchmark and select the fastest device.

`CPU` is the strict reference path without GPU or host-memory accelerator dispatch. `CPU_ACCEL` keeps CPU as the primary backend while allowing registered host-memory accelerators such as BLAS or AMX, and degrades to plain CPU when none is available. An explicit `CUDA`, `METAL`, or `VULKAN` request is strict: an unavailable or unvalidated combination returns an error rather than silently switching to CPU or another GPU backend.

v1 selects one primary device and may schedule unsupported or deliberately host-side work on CPU. The public Interface does not expose GGML device handles, layer-offload counts, tensor-placement overrides, multi-GPU split modes, or tensor-split ratios. These remain internal until a model-family-independent contract is justified and validated.

Device discovery and the actual device selected for a Loaded Model are observable through project-owned value types. Runtime device index `-1` means automatic selection; every non-negative value, including `0`, identifies an entry returned by the same runtime enumeration and is not a stable cross-host hardware identifier.

Implementation checkpoint (2026-07-22): the GGML device registry is behind the
public project-owned device structure and availability probe. VITS honors an
explicit global registry index for CPU and CUDA, and its Loaded Model query is
filled from the actual GGML backend device. `AUTO`, `CPU`, and `CPU_ACCEL` retain
a CPU primary device. A strict explicit CUDA request selects CUDA when the build
contains a matching runtime device; Metal and Vulkan remain unavailable for VITS.
Enumeration can still report a compiled device before its Model Family
combination is Supported, so the family loader and published support matrix remain
the support boundary.

The VITS model owns one internal Backend Plan rather than a raw backend handle.
The Plan owns the primary backend, optional host-memory accelerators, and the CPU
fallback; exposes the primary backend only for weight allocation; creates every
GGML scheduler in priority order with CPU last; and configures thread counts via
the backend registry capability. All VITS graph paths consume this Plan and no
longer construct backend-specific schedulers themselves. The CUDA 13.3 Update 1
builds on DGX Spark/GB10 Linux AArch64 and RTX 4070 SUPER Linux x86-64 place
every executable node in the five scheduled stages on CUDA: 893 duration-stage
nodes, 6 prior-expansion nodes, 4 latent-sampling nodes, 357 acoustic-flow nodes,
and 516 waveform-decoder nodes. View-only nodes are reported separately,
executable CPU fallback is zero, and each stage has one CUDA split. Both hosts
use strict FP32, native cubins (`sm_121a` and `sm_89`), and the R580 driver
branch.

This completes the release-toolkit physical-Linux implementation checkpoint,
not the full `cu13` Provider qualification. VITS/CUDA remains Experimental until
numerical acceptance thresholds, the R610 physical check, Windows x86-64, the
complete release cubin sets, and clean-runtime packaging gates are complete.
The combined evidence is recorded in
`reports/validate/vits/vits-ljspeech-cuda-13.3-linux.json`.

The General Model Capability query also reports Native Streaming Synthesis only when the loaded Model Package and selected Execution Backend combination has passed the separate streaming validation gates. Chunked Audio Delivery remains available independently and does not set that capability.

## Backend Module Loading

Both statically linked backends and optional GGML dynamic backend modules are supported. Static backends are registered by the build. Dynamic modules are loaded explicitly through `synth_backend_load_from_dir()` or `synth_backend_load_default()` before the first model load; in a static build these functions are successful no-ops.

`synth_backend_load_from_dir()` scans only the supplied artifact directory. `synth_backend_load_default()` resolves the loaded shared synthesize.cpp library and scans only its fixed relative `synthesize/backends/` directory, matching the native installation layout. Neither operation scans the directory containing the core library itself, the current working directory, a Model Package directory, `PATH`, or arbitrary system library paths. Repeating an operation for the same directory is idempotent.

v1 does not unload registered backend modules. Once model loading has begun, the process backend registry is fixed so a Loaded Model cannot retain a device or implementation from an unloaded module.

Implementation checkpoint (2026-07-23): the public module-loading Interface is
implemented with canonical-directory idempotency and serialization against the
first valid Model load. Dynamic builds install modules below the loaded core at
`synthesize/backends/`, select the highest-scoring compatible GGML module variant
for each backend kind, and do not honor GGML's ambient `GGML_BACKEND_PATH`
override. A registered install-and-consume test builds a dynamic CPU runtime,
checks empty-directory failure and package-local default discovery, and verifies
that model loading freezes further registry mutation. Static builds retain the
same validation and ordering Interface while module registration is a no-op.
Official Python Providers build with this dynamic-module mode: core GGML
libraries remain beside `libsynthesize`, while CPU and accelerator modules are
installed below the private `synthesize/backends/` directory with relocatable
loader paths. On x86-64, Provider builds carry the full GGML CPU variant set and
the loader registers only the highest-scoring compatible variant; on AArch64,
the same Interface loads the single architecture-specific CPU module. Variant
names remain private and the public backend kind is `cpu` in both cases.

## Initial Implementation Sequence

1. **CPU** establishes the complete correctness and performance baseline.
2. **CUDA** is the first GPU backend brought through the validation gates.
3. **Metal** follows using the same family graph and validation suite.
4. **Vulkan** follows as the first cross-vendor GPU backend in the initial plan.

This sequence is a delivery priority, not a permanent ranking of backends. Other GGML backends may be added later or developed opportunistically, but they are outside the initial support commitment and must pass the same validation gates.

## Support States

- **Unavailable**: the combination cannot complete inference.
- **Experimental**: it can complete some or all inference but has not passed every validation gate.
- **Supported**: the declared Model Variants have passed the applicable Port Validation Suite on the claimed backend combinations.

Compilation, model loading, isolated operator tests, or partial offload are not sufficient to declare support.

## Validation Gates

A model-family and backend combination must:

1. Use the same converted model and Port Validation Suite cases as the CPU baseline.
2. Match reference and CPU intermediate tensors within documented tolerances.
3. Produce finite, correctly shaped PCM that passes the declared deterministic
   waveform-regression tolerances.
4. Pass the normally 12--32 cases covering the applicable text, synthesis-control,
   stochastic, and Voice-conditioning paths.
5. Prove actual device placement or offload coverage and report latency,
   real-time factor, and peak memory as operational evidence.
6. Pass repeated-run and resource-cleanup checks.

Performance measurement is required for support, but no minimum speedup is. A
backend is described as accelerated only when the published comparison demonstrates
an improvement over the CPU baseline on the stated hardware and workload. Corpus
quality and perceptual evaluation are governed separately by the deferred Quality
Evaluation Suite and are not prerequisites for `port_validated` support.

The support matrix must identify the exact backend, hardware class, precision or quantization, model variants, and execution mode covered by each result.
