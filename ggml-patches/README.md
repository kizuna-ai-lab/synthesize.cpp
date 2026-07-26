# Local patches to the vendored ggml

`ggml/` is upstream's tracked tree at the SHA in `ggml/UPSTREAM`, **plus the
patches in this directory**, applied in filename order by
`scripts/sync-ggml.sh`. The sync aborts loudly if a patch stops applying, so a
re-vendor can never silently drop a local change again.

This directory exists because the previous claim — that the snapshot was
verbatim — was discovered to be false on 2026-07-26: the tree carried about 400
undocumented local lines, and one `sync-ggml.sh` run would have silently
destroyed the packaging installs, an F16 inference path, a CI deadlock fix, and
the strict-FP32 precision policy all at once.

Do not edit `ggml/` directly. Change a patch file (or add one), then run
`scripts/sync-ggml.sh` with no arguments to regenerate `ggml/` from
upstream-plus-patches, and commit both together.

## 0001-synthesize-local.patch

One canonical patch, inventoried by hunk:

| Files | What and why |
| --- | --- |
| `src/ggml-cuda/conv-transpose-1d.cu`, `src/ggml-cuda/ggml-cuda.cu` | Templated `conv_transpose_1d` kernel plus `supports_op`, so F16-stored transposed-convolution weights execute on CUDA (the VITS F16/Q8 profiles store them halved). |

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

The upstream history is kept in that report: the gate was submitted as
[llama.cpp#26112](https://github.com/ggml-org/llama.cpp/pull/26112) and closed,
the maintainer holding that precision belongs at the ggml op level rather than in
a backend compile flag. Should a strict-FP32 op value ever land upstream, this
group does not need to come back — the family graph builders would set it on the
sensitive matmuls directly.
