# Proposed upstream patch: honor GGML_CUDA_DISABLE_TF32 in the F32 mmf path

Status: measured on 2026-07-26 and applied locally the same day as part of
`ggml-patches/0001-synthesize-local.patch`. Submitted upstream as
[ggml-org/llama.cpp#26112](https://github.com/ggml-org/llama.cpp/pull/26112)
and **closed the same day** by the CUDA maintainer: "This is the wrong way to
do it. Precision should be defined at a ggml level, not at the level of a
specific ggml backend. There are ongoing efforts to fix this." The PR was also
flagged under the project's AI-usage policy, which prohibits AI-written pull
request descriptions and responses outright; any future interaction there must
be authored by the maintainer of this project personally.

## Problem

`GGML_CUDA_DISABLE_TF32` ("use strict FP32 instead of TF32 in cuBLAS") sets the
cuBLAS math mode and nothing else. `ggml_cuda_should_use_mmf` routes every F32
matrix multiply with `src1_ncols <= 16` on Ampere-or-newer devices to the mmf
tile kernels, which execute `mma.sync...f32.tf32.tf32.f32` — so a build that
requested strict FP32 still computes its skinny F32 matmuls at TF32 precision
(10-bit mantissa, ~1e-3 relative error).

## Patch

`ggml-mmf-f32-tf32-gate.patch` (against snapshot 707321c4): under
`GGML_CUDA_DISABLE_TF32`, `ggml_cuda_should_use_mmf` returns false for
`GGML_TYPE_F32`, falling back to mmvf or cuBLAS SGEMM. F16/BF16 are untouched:
a reduced-precision storage type is its own precision statement, while F32
storage is a strict-FP32 contract.

## Measurements (Kokoro PL-BERT stage, GB10/sm_121a, CUDA-vs-CPU relative)

| tokens | unpatched | patched |
| --- | --- | --- |
| 4 | 1.06e-3 | 2.16e-6 |
| 8 | 2.10e-3 | 2.62e-6 |
| 16 | 2.07e-3 | 2.13e-6 |
| 17 (cuBLAS either way) | 1.93e-6 | 1.93e-6 |

Downstream on the worst manifest case: F0 deviation 0.99 Hz → 0.003 Hz,
harmonic-source complex divergence 0.334 → 0.008. Synthesis wall time is
unchanged on both the longest and shortest cases (1.39 s → 1.38 s; 0.80–0.86 s
→ 0.82–0.83 s): TTS graphs are dominated by wide decoder matmuls that never
used mmf.

## Landing

The audit that preceded this fix found the "verbatim snapshot" premise was
already false — the tree carried ~400 undocumented local lines — so the
project adopted patch-on-sync (`ggml-patches/` + `scripts/sync-ggml.sh`), and
this gate is applied there. That is now the durable state, not an interim one:
upstream rejected the backend-level compile flag as a design, and the
op-level mechanism the maintainer points toward does not cover this case yet —
in the vendored revision `GGML_PREC_F32` reaches only the F16 cuBLAS
compute-type choices (`ggml-cuda.cu:1661,2278,2359`); `should_use_mmf` never
consults precision, so `ggml_mul_mat_set_prec` cannot disable the tf32 tile
path today.

The exit path is upstream's own precision rework. When a ggml-level mechanism
lands that lets an F32 matmul demand strict FP32 per op, the family graph
builders switch to it on the sensitive matmuls, and these hunks drop from the
local patch at that re-vendor.
