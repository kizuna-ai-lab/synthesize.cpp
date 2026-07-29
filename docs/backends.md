# Execution Backend Policy

Status: Confirmed, last updated on 2026-07-29.

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

CUDA F32 matrix multiplies compute at TF32 precision, and the project makes no
strict-FP32 promise on CUDA. There is no build option to change this: cuBLAS runs
in `CUBLAS_TF32_TENSOR_OP_MATH`, and GGML's own tensor-core kernels (`mmf.cu`)
take every F32 matrix multiply with 16 or fewer columns on Ampere-class or newer
devices through `mma.cuh`'s `mma...tf32` tiles. TF32 carries a 10-bit mantissa,
so the intermediate CUDA-versus-CPU deviation this produces is about 1e-3
relative rather than FP32's 1e-7.

This is a deliberate reversal, recorded on 2026-07-26. The project previously
carried a `SYNTH_CUDA_TF32` option and a local GGML patch set that gated both
paths, restoring strict FP32 at every width. Both were removed after a listening
test: two Kokoro renderings differing only in that gate were compared
sample-aligned, and the difference was inaudible. The gate was also free of any
speed cost, which cuts the other way too — it was buying nothing that could be
heard, and TTS graphs are dominated by wide decoder matmuls that never took the
`mmf` tile path in the first place. Measurements, method, and the reasoning are
in `reports/upstream/ggml-mmf-f32-tf32-gate.md`; the removed hunks are inventoried
in `ggml-patches/README.md`, which as of 2026-07-27 inventories the whole local
patch set, because none of it remains.

What replaces the guarantee is measurement. Consequences are bounded by each
family's Golden suite rather than by a precision flag: across every Kokoro profile
and case the rounded durations, frame count, and alignment stay exact — that is
the property required to hold across backends — while intermediate drift and
waveform correlation are recorded per stage in `tests/tolerances/`. Those
recorded CUDA numbers describe TF32 arithmetic, because that is what ships.

Should upstream's precision rework land an op-level strict-FP32 value, the family
graph builders can set it on individual sensitive matmuls without reintroducing a
backend-wide flag. Upstream declined the backend-flag shape
(llama.cpp#26112: precision belongs at the ggml level), and that remains the only
route back.

## Discrete Outputs Are Held On CPU

A stage whose output is a discrete value — a rounded frame count, an argmax, a
sampled token index — runs on CPU on every Execution Backend, and so does every
stage feeding it. Continuous stages stay on the primary backend.

The reason is that tolerance cannot absorb a discrete difference. TF32's roughly
1e-3 relative error is enough to move a value across a rounding boundary: on
Kokoro's 146-token case one token's duration rounded from 1 to 2 on CUDA, taking
the frame count from 376 to 377. The downstream stages then have different shapes
and there is no threshold that compares a 240,640-element tensor to a
241,280-element one. Nor can the stochastic-replay seam in
`port-validation.md` inject a noise tensor sized for a frame count that no longer
matches. Eighteen of the family's twenty-one CUDA validation runs failed that way,
and only three of those failures were numerical.

The mechanism is a CPU-only scheduler over CPU-resident weights, not forcing nodes
onto CPU inside a mixed graph. The distinction is measured: pinning nodes while
their operands stayed in the primary buffer produced 2,372 scheduler splits on the
duration graph and ran five times slower end to end than leaving the stage on the
GPU, whereas a single-backend scheduler over mirrored weights is one split.
`BackendPlan::create_cpu_scheduler` and `cpu_backend` exist for this, and
`tests/backend_plan_test.cpp` asserts the single split so a regression that
reintroduced cross-backend operands is caught.

Two consequences are worth stating plainly. The whole path to the rounding must be
held, not just the rounding stage: pinning Kokoro's duration predictor alone left
the logits 8.8e-2 away from the reference because PL-BERT still ran on CUDA and
fed it. And the cost is real — roughly a 95 percent increase in synthesis wall
time on Kokoro's longest case, 1.5 s to 2.9 s, still 3.2 times faster than real
time. The mirrored weights also occupy both buffers, 89 MB of 353 MB for this
family.

This rule scales with how much of a family sits upstream of a discrete decision,
so its cost is per family and has to be measured rather than assumed:

| Family | Held stages | Mirrored weights | Cost | Result |
| --- | --- | --- | --- | --- |
| Kokoro | PL-BERT, duration predictor | 89.2 of 352.8 MB | 1.5 s → 2.9 s, +95 % | 3.2× real time |
| VITS | text encoder, duration predictor | 27.5 of 113.2 MB | 1.09 s → 1.39 s, +29 % | 7.6× real time |

Kokoro is the more expensive of the two because its LSTMs are unrolled in the
graph, so the duration stage is 26,916 nodes and all of it moves. VITS is cheaper
because its discrete path is one graph and the HiFi-GAN decoder, which dominates,
never leaves the primary backend.

The mechanism generalizes and the second family needed no rework: identify the
weight groups the discrete path reads, mirror them, give any group a second view
when a continuous stage reads the same tensors from the primary buffer, and switch
that stage's scheduler. For an autoregressive codec language model, where every
sampled token feeds the next step, the held portion would be most of the model,
and that trade has to be measured before such a family is accepted.

## Operator Choice Is Part Of The Backend Contract

An operator that a backend declines does not fail: the scheduler places it on
the CPU, and the graph still produces a plausible answer. That makes operator
selection a backend-portability question, not only a numerical one.

VITS's transposed convolutions are the worked example. `ggml_conv_transpose_1d`
is accepted by CUDA and Vulkan only when both operands are F32, so with halved
weights those nodes silently moved to the CPU — the undeclared placement
`docs/port-validation.md` calls a hard failure. A local CUDA patch hid it on one
backend for a while and left Vulkan broken, because a patch covers the backend
it was written for and nothing else.

The durable fix was to stop using the fused operator. `ggml_mul_mat` followed by
`ggml_col2im_1d` computes the same thing, takes F16 weights on the ordinary
weight path, and has CPU, CUDA and Vulkan kernels upstream. Prefer an operator
every target backend accepts over one that needs a local patch to reach parity
on a single backend; `ggml-patches/README.md` records what that cost and bought.
Since 2026-07-27 the point is structural rather than advisory: `ggml/` is a
submodule and cannot carry a patch at all, so an operator no backend accepts has
to be replaced rather than worked around.

Where a decomposition changes accumulation — CUDA's F16 matrix multiply
accumulates in half precision — the tensor types feeding it are part of the
decision. VITS keeps transposed-convolution weights F32 in every profile for
that reason, at 5.32 MB per package.

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

## Placement And Cleanup Are Checked, Not Declared

Gate 6 above -- "Pass repeated-run and resource-cleanup checks" -- and the
placement half of gate 5 were, until 2026-07-29, satisfied by nothing. Every
Golden Manifest in this repository declares `backend_placement` and
`resource_cleanup` on every case, and outside the manifests and the schema enum
those strings appeared nowhere: no validator, runner or test read a case's
`checks` array at all. The evidence on record for VITS was twenty repeated
*processes*, which cannot observe an in-process leak because exit reclaims
everything.

**Placement is now counted, not asserted.** `BackendPlacement` gained
`off_cpu_node_count`, which classifies by device type independently of which
backend is primary -- the older counters classify against the primary, so a node
running on the primary accelerator was never reached by the type test and
`accelerator_node_count` stayed zero exactly when the work had moved. The family
reports per-stage counts and the replay validator decides the declared check
against them: on a CPU run every node of every stage must be on the CPU, and with
`--accelerate` the codec's must all have left it while the talker's and the code
predictor's must not have moved at all. That last clause is the discrete-output
rule made checkable rather than trusted.

A backend that is present is not a backend that ran. A graph placed on an
accelerator and silently fell back looks identical from outside the process,
which is why the check counts nodes rather than asking whether a device exists.

**Cleanup is measured across cycles.** `tests/public_cleanup_test.cpp` drives
whole load/context/synthesize/free cycles on every backend the build claims, for
all three families, and asserts two things one run cannot: that every cycle after
the first returns an identical frame count and PCM digest at a fixed seed, and
that the *floor* of post-free resident memory does not rise. The floor rather
than the difference between first and last: a leak raises the floor, arena churn
only raises peaks, and a draft written the other way failed on a run whose first
cycle happened to land in a trough.

## CPU Thread Count Is Below The CPU Count

A Synthesis Context defaults to **half the CPUs it may use**, not all of them,
and an embedder may override that through `synth_context_set_threads`.

Asking for exactly the CPU count is not a small pessimization. GGML's barrier
spins rather than yielding, and synthesis is batch-one autoregressive decoding
that crosses that barrier on the order of a hundred times per output frame -- for
Qwen3-TTS, 28 talker layers plus fifteen code-predictor steps of five layers.
Once the workers occupy every CPU, any thread the scheduler moves aside stalls
all the others. Measured on a twenty-CPU machine over one sentence, real-time
factor against thread count:

| threads | 1 | 2 | 4 | 8 | 10 | 12 | 14 | 16 | 18 | 19 | 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RTF | 3.31 | 2.17 | 1.50 | 1.25 | 1.18 | 1.13 | 1.17 | 1.36 | 1.99 | 3.91 | 10.2 |

The degradation begins well before the cliff, so leaving one CPU free is not
enough headroom. Half lands within five percent of the best count measured here
and cannot reach the cliff on any machine size.

This is a default, not a policy: the best count depends on the machine, and an
embedder that knows its deployment should say so rather than inherit a guess.
`available_cpu_parallelism()` continues to mean "CPUs this process may run on",
honouring the affinity mask and cgroup quota; the thread default is derived from
it rather than equal to it.

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
