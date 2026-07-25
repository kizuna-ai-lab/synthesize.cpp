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
| `src/ggml-cuda/im2col.cu` | Tiled 1-D im2col kernel; the im2col-plus-GEMM convolution decomposition dominates TTS decoder graphs. |
| `CMakeLists.txt` (option), `src/ggml-cuda/CMakeLists.txt`, `src/ggml-cuda/common.cuh`, `src/ggml-cuda/solve_tri.cu`, `src/ggml-cuda/mmf.cu` | `GGML_CUDA_DISABLE_TF32`: strict FP32 on CUDA. Gates the cuBLAS math mode, `solve_tri`, and — added 2026-07-26 after the 16-column cliff was measured — the F32 `mmf` tile path, which computes via tf32 MMA. See `reports/upstream/ggml-mmf-f32-tf32-gate.md`. |

Upstreaming status: the `GGML_CUDA_DISABLE_TF32` set was submitted as
[llama.cpp#26112](https://github.com/ggml-org/llama.cpp/pull/26112) and closed —
the maintainer holds that precision belongs at the ggml (op) level, not as a
backend compile flag, and points to ongoing rework. The op-level mechanism does
not reach the tf32 tile path in the pinned revision, so this patch carries the
behavior until that rework lands; see
`reports/upstream/ggml-mmf-f32-tf32-gate.md` for the exit path.
