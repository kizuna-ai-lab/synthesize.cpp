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
| `CMakeLists.txt`, `src/CMakeLists.txt` | Explicit install destinations (`RUNTIME`/`LIBRARY`/`ARCHIVE`/`PUBLIC_HEADER`) so the Provider wheels and the installed SDK lay out identically across platforms. |
| `src/ggml-cpu/ggml-cpu.c` | `ggml_thread_cpu_relax` for MSVC (`YieldProcessor`) and a bounded spin-then-yield in the barrier wait: a pure pause-spin never yields the core, so under CPU oversubscription (a 2-vCPU runner) waiters starve an un-arrived worker and `ggml_barrier` deadlocks. |
| `src/ggml-cuda/conv-transpose-1d.cu`, `src/ggml-cuda/ggml-cuda.cu` | Templated `conv_transpose_1d` kernel plus `supports_op`, so F16-stored transposed-convolution weights execute on CUDA (the VITS F16/Q8 profiles store them halved). |

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
