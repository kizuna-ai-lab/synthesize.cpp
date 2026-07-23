# VITS CUDA Performance Research Backlog

Status: Recorded on 2026-07-22; implementation decision deferred.

This file records possible follow-up work without committing the project to an
implementation. None of the candidates below is a release requirement, and none
should change production code until its dependency, numerical, portability, and
maintenance costs have been reviewed explicitly.

## Profile checkpoint

The evidence is the final strict-FP32 CUDA 13.3 Update 1 implementation after
the stride-aware transpose convolution, pointwise Conv1D graph change, tiled
Conv1D im2col, decoder-internal layout, and expanded 128-by-32 im2col tile. The
profile covers six fixed-seed `ljs-long` public syntheses on the Linux RTX 4070
SUPER. Total GPU kernel time is 860.390285 ms across 13,812 kernel instances.

| Kernel family | Instances | Time (ms) | GPU kernel time |
|---|---:|---:|---:|
| `ampere_sgemm_128x64_tn` | 438 | 276.310490 | 32.115% |
| `im2col_1d_tiled_f32_kernel` | 684 | 268.831098 | 31.245% |
| `conv_transpose_1d_kernel` | 24 | 99.399177 | 11.553% |
| `k_bin_bcast` | 4,518 | 70.821001 | 8.231% |
| `k_get_rows_float` | 180 | 42.009815 | 4.883% |
| `leaky_relu_kernel` | 462 | 33.991476 | 3.951% |
| `cpy_scalar_transpose` | 1,674 | 14.753186 | 1.715% |

Materialized im2col plus its main SGEMM consumes 545.141588 ms, or 63.360% of
profiled GPU kernel time. This is the only remaining candidate with a plausible
large end-to-end ceiling; it is also the candidate with the largest design and
numerical risk.

## Deferred candidates

### Implicit-GEMM or fused Conv1D

Potential: avoid writing and rereading the materialized im2col matrix, combining
part of the current 31.245% im2col and 32.115% SGEMM cost.

Research required:

- determine whether a project-owned CUDA kernel, a vendored compile-time
  implementation, or an NVIDIA library is acceptable without adding an
  incompatible runtime or packaging dependency;
- cover kernel widths 3, 7, and 11, dilation, padding, long decoder outputs, and
  the channel shapes actually shared by VITS variants;
- retain a generic fallback instead of making VITS shapes a GGML-wide contract;
- establish strict-FP32 accumulation behavior on `sm_89` and `sm_121a`.

Risk: High. The rejected direct pointwise GEMM experiment already showed that a
mathematically equivalent operand orientation can amplify final Golden drift.
An implicit convolution is expected to change accumulation order unless it is
designed specifically not to do so.

### SGEMM algorithm or cuBLASLt selection

Potential: improve the 438 large `TN` SGEMMs without replacing the surrounding
graph.

Research required:

- map the hot launches back to exact Conv1D shapes and test reproducible
  algorithm selection rather than relying on an opaque heuristic;
- compare CUDA architectures and driver branches because the selected algorithm
  and workspace behavior may differ;
- verify strict-FP32 output and package the required library/workspace policy on
  every release platform.

Risk: High. A faster algorithm may change reduction order, drift, workspace use,
or reproducibility across CUDA releases.

### Transpose-convolution fusion or kernel retuning

Potential: address the remaining 11.553%, possibly by fusing crop, bias, or the
following activation, or by retuning the current stride-aware kernel.

Research required: separate arithmetic time from adjacent crop/broadcast costs,
retain the established descending accumulation order, and verify that any
specialization generalizes beyond the current VITS kernel/stride pairs.

Risk: Medium to high. The current kernel is exact at established probes and
already removed the dominant wasted scan; the remaining ceiling is smaller.

### General elementwise graph fusion

Potential: reduce the 8.231% `k_bin_bcast` family and some of the 3.951%
leaky-ReLU cost.

Research required: correlate the 4,518 broadcast launches with bias, residual,
scale, and other graph operations before choosing a fusion. Any implementation
should be a reusable GGML operation or backend optimization, not a VITS-only
public Interface.

Risk: Medium. The individual kernels are small, the call family mixes several
semantics, and a new fusion seam may cost more maintenance than it saves.

### Embedding/get-rows investigation

Potential: address the 4.883% `k_get_rows_float` share.

Priority: Low. First determine whether the calls are bandwidth-limited, repeated
unnecessarily, or already close to the hardware limit. This is not a candidate
for implementation while the combined convolution path remains unresolved.

## Decision gates

Before promoting any candidate from research to implementation:

1. Make an explicit go/no-go decision and define its allowed dependency and
   maintenance budget.
2. Add exact boundary/shape unit tests before changing production code.
3. Run same-session A/B measurements on GB10 and RTX 4070 SUPER; a gain on one
   host does not justify a material regression on the other.
4. Preserve all seven 12-case Golden stages, strict-FP32 policy, fixed-seed
   repeatability, one-split all-CUDA placement, and zero executable CPU fallback.
5. Reject or isolate experiments that alter established drift, create a new
   runtime dependency without a packaging plan, or rely on an unstable
   architecture/driver heuristic.

Until these gates are deliberately opened, the current implementation is the
performance checkpoint and this document is informational backlog only.
