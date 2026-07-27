# Local patches to the vendored ggml

**There are none.** `ggml/` is upstream's tracked tree at the SHA in
`ggml/UPSTREAM`, verbatim, minus `.github/`. This directory is kept for the
mechanism and for the record below.

The mechanism still stands: `scripts/sync-ggml.sh` regenerates `ggml/` from
upstream plus every `ggml-patches/*.patch` in filename order, and aborts loudly
if one stops applying. If a local change ever becomes necessary again, add a
patch file here and re-run the script -- never hand-edit `ggml/`.

## How this directory emptied

The tree once carried about 400 undocumented local lines, discovered on
2026-07-26. Nothing recorded them, and one `sync-ggml.sh` run would have
silently destroyed the packaging installs, an F16 inference path, a CI deadlock
fix, and the strict-FP32 precision policy at once. They were split into five
groups and each was measured against the reason given for it.

Four went that day. Three of those four because the justification did not
survive measurement, one because a better fix existed on this side of the seam.
The fifth -- the `conv_transpose_1d` CUDA kernel -- was measured, found to be
load-bearing, and kept, with a note saying its only exit was upstream fixing the
kernel rather than a replacement on this side. That note was wrong on both
counts, and the group went on 2026-07-27. Each section below records what was
tried and what the numbers were.

One consequence is worth stating. With no local modification left, `ggml/` could
become a git submodule pinned to the same SHA, removing 1,992 files and about
19.5 MB from this repository. A submodule is a pointer and cannot carry a local
change, so one hunk would have blocked it as surely as seventeen. That
conversion is deliberately not made here: it changes how the tree is cloned,
packaged into an sdist, and re-vendored, and it should be its own change.

## Removed on 2026-07-27: the templated conv_transpose_1d CUDA kernel

Three hunks in `src/ggml-cuda/conv-transpose-1d.cu` templated upstream's kernel
over the source type, and one in `src/ggml-cuda/ggml-cuda.cu` widened
`supports_op` to accept an F16 `src0`.

**What it was for.** VITS's F16 and Q8_MIXED profiles stored transposed-
convolution weights halved, and upstream's `supports_op` accepts
`CONV_TRANSPOSE_1D` only when both operands are F32, so those nodes fell back to
the CPU silently rather than erroring -- an undeclared CPU placement, which
`docs/port-validation.md` treats as a hard failure. Separately, and never
written down until the group was measured, upstream's kernel scales
quadratically in input length: with the group removed, `ljs-long` (315 tokens)
took over 240 s on CUDA against 1.48 s with it, and that was the **F32** profile,
which carries no halved weights and by the recorded description should have been
unaffected.

**What replaced it.** Both reasons are gone because the operator is gone.
`src/arch/vits/operations.cpp` now expresses a transposed convolution as a
column matrix multiply followed by `ggml_col2im_1d` -- what the fused op
decomposes into, and upstream already, with CPU, CUDA and Vulkan kernels.
`ggml_mul_mat` takes an F16 `src0` natively, that being the ordinary weight path,
so the `supports_op` widening is unnecessary; and the quadratic kernel is never
reached. Measured on GB10 across the four `upsample` stages at `ljs-long`
shapes, against the **patched** kernel rather than upstream's:

| | fused op, patched | mul_mat + col2im_1d |
| --- | --- | --- |
| CPU, four stages | 204 ms | 130 ms |
| CUDA, four stages | 33.8 ms | 4.9 ms |

Agreement with the fused op is exact on a small case (`0.000e+00`) and 5e-7
relative at real F32 shapes. The layout recipe -- merge the kernel's leading
`[kernel, out_channels]` pair so `k` varies faster than `oc`, then transpose so
the matrix multiply reduces over the input channels -- is
[qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp)'s, from its
`ARCHITECTURE.md`.

**What it cost.** VITS's transposed-convolution weights now stay F32 in every
profile (`resolve_vits_target_spec`), adding 5.32 MB per package: ljspeech F16
goes from 70.4 MB to 75.8 MB, Q8_MIXED from 52.9 MB to 58.2 MB. That is not
incidental. CUDA's F16 matrix multiply accumulates in half precision where the
templated kernel had upconverted and accumulated in F32, so leaving the weights
halved would have moved CUDA from bit-exact agreement with the F32 reference to
about 3e-3 relative. Keeping them whole leaves both backends near 1e-6 -- better
than the F16 CPU path managed before the change (1.6e-3, upstream's CPU kernel
losing precision there that `mul_mat` does not). Packages built earlier are
refused by tensor type in `src/arch/vits/weights.cpp` rather than run at reduced
accuracy.

Two things this also fixed, neither of them the point. Vulkan rejects
`CONV_TRANSPOSE_1D` with an F16 `src0` exactly as CUDA did, and the patch only
ever touched CUDA -- so VITS's F16 decoder had been falling back to the CPU on
Vulkan the whole time, undetected. And Qwen3-TTS's intake expected to need a 1-D
column scatter-add operator; it does not, because `ggml_col2im_1d` is already
upstream.

Kokoro is untouched. It never used the fused operator -- it decomposes
transposed convolution into a per-tap `mul_mat` loop of its own -- and it keeps
F16 weights with the tolerances measured for them.

## Removed on 2026-07-26: the barrier spin-then-yield and the MSVC pause hint

Three hunks in `src/ggml-cpu/ggml-cpu.c` added a bounded spin-then-yield to the
barrier wait, plus an MSVC branch for `ggml_thread_cpu_relax`. The barrier hunks
treated a symptom: GGML's barrier spins on a pause hint without yielding, so
*oversubscribed* threads let waiters starve the worker that has not arrived.

The cause was ours. `std::thread::hardware_concurrency` reports the host's CPU
count and ignores both the affinity mask and the cgroup quota, so inside a
two-vCPU container on a twenty-core host the library asked for twenty threads.
`synth::available_cpu_parallelism` in `src/cpu-parallelism.cpp` now clamps to the
affinity mask and the cgroup v1/v2 quota. Measured here: `hardware_concurrency`
reports 20 under `taskset -c 0,1` while the clamp reports 2, and 1 under a single
CPU. The public C Interface exposes no thread count, so no caller can route around
it.

Verified against the scenario the patch existed for: three real VITS syntheses
under `taskset -c 0,1` with the upstream barrier completed in 14.5 s each with
correct frame counts, no hang.

The MSVC pause hint went with them, deliberately. Upstream's
`ggml_thread_cpu_relax` covers `__aarch64__`, `__x86_64__` and `__riscv` and falls
back to an empty body otherwise, and MSVC defines `_M_X64` rather than
`__x86_64__`, so Windows builds spin without a pause instruction. With threads
clamped that is a power and latency cost under contention, not a hang, and
jiangzhuo accepted it rather than keep a patch for it.

## Removed on 2026-07-26: the explicit install destinations

Two hunks gave `ggml` and `ggml-base` explicit `RUNTIME`/`LIBRARY`/`ARCHIVE`/
`PUBLIC_HEADER` destinations, on the grounds that upstream's
`install(TARGETS ggml LIBRARY PUBLIC_HEADER)` lays out inconsistently across
platforms. The replacement lives in the project's own `CMakeLists.txt`.

Measurement narrowed what the patch was actually buying. On Linux upstream's
rules install the static archives and the versioned shared objects with their
symlinks correctly under both `BUILD_SHARED_LIBS` settings -- verified by removing
our rules entirely and comparing install trees, which came out identical. The gap
is Windows only: a shared library's artifacts there are RUNTIME (the DLL) and
ARCHIVE (the import library), and naming `LIBRARY` covers neither, so nothing
would be installed at all.

Our replacement is therefore scoped to `WIN32 AND SYNTH_BUILD_SHARED` and puts the
DLL in `BINDIR` next to the executables that load it, with the import library in
`LIBDIR`. That destination split is what the patched form got wrong: it sent both
to the directories CMake's artifact-kind names imply rather than the ones Windows
loads from. This has not been executed on Windows; `synthesize-installed-sdk-test`
is the gate that proves it there, and it passes on Linux for both static and
shared builds.

## Removed on 2026-07-26: the tiled 1-D im2col kernel

A hunk in `src/ggml-cuda/im2col.cu` carried a tiled 1-D im2col kernel, justified
on the premise that the im2col-plus-GEMM convolution decomposition dominates TTS
decoder graphs. Measurement does not support that premise, so the hunk is gone.

Nine-sample A/B on Kokoro's longest case, F32, GB10, the only difference being
this kernel: upstream 2.301 / 2.425 / 2.589 s (min / median / max), tiled
2.277 / 2.388 / 2.577 s. The 39 ms gap is inside a run-to-run spread of about
300 ms. VITS showed no difference at all, 1.393 s against 1.392 s median. The
decoder graphs are dominated by wide matrix multiplies rather than convolution:
Kokoro's decoder stage is 1,936 nodes.

A performance patch that cannot be shown to improve performance is maintenance
cost with no return, and this one sat in an actively developed CUDA file.

## Removed on 2026-07-26: the strict-FP32 CUDA gate

A fifth group once carried `GGML_CUDA_DISABLE_TF32` across `CMakeLists.txt`,
`src/ggml-cuda/CMakeLists.txt`, `common.cuh`, `solve_tri.cu`, and `mmf.cu`,
gating the cuBLAS math mode, `solve_tri`, and the F32 `mmf` tile path so CUDA F32
matrix multiplies computed in strict FP32 rather than TF32. It was dropped, along
with the `SYNTH_CUDA_TF32` option that drove it.

The reason was a listening test, not a maintenance preference. Two Kokoro
renderings differing only in that compile definition were compared
sample-aligned, and the difference was inaudible; the measured numbers are in
`reports/upstream/ggml-mmf-f32-tf32-gate.md`. TF32 also costs nothing in
precision terms that the project was spending: synthesis wall time was already
unchanged either way, because TTS graphs are dominated by wide decoder matmuls
that never take the `mmf` tile path.

Two consequences worth stating plainly. CUDA intermediate drift returns to the
TF32 level — PL-BERT about 2.2e-2 absolute at six tokens rather than 4.2e-5, F0
about 0.99 Hz rather than 0.012 Hz — so the CUDA measurements recorded in
`tests/tolerances/` had to be re-measured against the shipped configuration. And
`SYNTH_CUDA_TF32` had to be deleted from `CMakeLists.txt` and `CMakePresets.json`
in the same change: left behind, it would have set an unused cache variable that
nothing reads, configuring and building successfully while silently doing
nothing.

A third consequence surfaced only on 2026-07-27. `tests/kokoro_lstm_test.cpp`
carried a hardcoded `1e-4` cross-backend bound measured under the gate, and the
removal did not re-measure it -- a stale build directory, still holding
`GGML_CUDA_DISABLE_TF32=ON` in its cache for an option that no longer exists,
kept it passing. Built clean it reports 7.06e-4 on CUDA at length 37 against
1.79e-7 on the CPU, and the bound is now taken per device.

The upstream history is kept in that report: the gate was submitted as
[llama.cpp#26112](https://github.com/ggml-org/llama.cpp/pull/26112) and closed,
the maintainer holding that precision belongs at the ggml op level rather than in
a backend compile flag. Should a strict-FP32 op value ever land upstream, this
group does not need to come back — the family graph builders would set it on the
sensitive matmuls directly.
