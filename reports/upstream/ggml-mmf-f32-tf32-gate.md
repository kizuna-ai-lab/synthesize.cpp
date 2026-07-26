# Proposed upstream patch: honor GGML_CUDA_DISABLE_TF32 in the F32 mmf path

Status: **superseded on 2026-07-26 — the gate was removed, not carried.** The
patch group described here, and the `SYNTH_CUDA_TF32` option that drove it, were
deleted the same day they were documented, after the end-to-end listening test
recorded at the bottom of this report found the difference inaudible on the
Kokoro variant. CUDA F32 matrix multiplies now compute at TF32 precision and the
project makes no strict-FP32 promise on CUDA; see `docs/backends.md` and the
"Removed on 2026-07-26" section of `ggml-patches/README.md`. Everything below is
retained because the measurements are the evidence for that decision and for any
future reversal — the analysis is still correct, only the conclusion changed.

Original status: measured on 2026-07-26 and applied locally the same day as part of
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

The exit path is upstream's own precision rework, which as of 2026-07-26 is at
the design stage, not implementation. Its clearest public articulation is the
CUDA maintainer's comment on
[llama.cpp#24364](https://github.com/ggml-org/llama.cpp/pull/24364)
(2026-07-16), an NVFP4 discussion where another per-backend precision control
got the same "solve it at the ggml level" answer:

- precision is a per-op property (`ggml_prec` in `op_params`), never a backend
  flag;
- `GGML_PREC_DEFAULT` means backends optimize freely — on Blackwell that
  includes TF32;
- the enum should grow minimum-precision values (`GGML_PREC_A8`,
  `GGML_PREC_A16` were sketched);
- and directly on point: "My opinion is that `GGML_PREC_F32` should require
  strict FP32 arithmetic, not just the numerical range of FP32. For that we
  should add a new value."

Today the enum in master still holds only `DEFAULT` and `F32`, no tracking
issue or implementation PR exists, and per-kernel attempts to honor
`GGML_PREC_F32` (ggml#1536, moved to llama.cpp#24984) were closed unmerged.
The things to watch are the `ggml_prec` enum in `ggml/include/ggml.h` and
`should_use_mmf` growing a precision parameter. When a strict-FP32 op value
lands, the family graph builders set it on the sensitive matmuls and these
hunks drop from the local patch at that re-vendor.

The measurements in this report — the 16/17-column cliff and the 0.99 Hz F0
consequence — are precisely the motivating evidence for that new enum value,
should the maintainer of this project choose to bring them to that
discussion personally.

## End-to-end A/B and listening result (2026-07-26)

The measurements above are stage probes. This section records the same gate
measured end to end on rendered audio, plus an informal listening result.

Method: one Kokoro synthesis rendered twice from `kokoro-v1-0-F32.gguf`, voice
`af_heart`, seed 42, `--backend cuda`, on GB10/sm_121a, from two builds of the
same tree differing only in `SYNTH_CUDA_TF32`. F32 weights are required for the
comparison to mean anything — the gate governs F32 matmuls, and F16 or quantized
profiles route elsewhere. Control: the same build run twice with the same seed
produced byte-identical output, so the noise floor is exactly zero and every
difference below is attributable to that one compile definition.

| Metric | Value |
| --- | --- |
| Samples, duration | 118,800 / 4.950 s, identical in both builds |
| Magnitude-only cosine | 0.999554 |
| STFT cosine | 0.999204 |
| Waveform cosine | 0.987444 |
| Residual RMS | −16.0 dB relative to signal |
| Peak absolute difference | 0.1032, 37 % of the 0.280 signal peak |
| Per-bin magnitude deviation | 2.58 % median, 5.16 % mean, 18.5 % p95 |
| Per-bin phase deviation | 2.96° median, 8.98° mean, 41.3° p95 |
| F0 deviation over voiced frames | 5.18 Hz mean, about 50 cents |

Spectrally the two are nearly the same signal while the waveforms are not, and
best-alignment lag is zero throughout, so the difference is a redistribution of
per-harmonic magnitude and phase rather than a spectral-envelope or timing
change. Divergence rises from 0.017 at t=0 to about 0.22 by t=1 s and then
saturates; it does not accumulate across the utterance despite this family's
accumulating excitation phase.

The end-to-end F0 deviation, about 5 Hz mean, is larger than the 0.99 Hz measured
at the PL-BERT stage probe. The measurement points differ and the end-to-end
estimator is a crude autocorrelation, so treat the magnitude as indicative.

**Listening result.** jiangzhuo compared the two renderings sample-aligned with
instant switching at a shared playback position, and reported no audible
difference at all. This is one sentence, one sample, sighted, n=1. It supports
"TF32 is perceptually indistinguishable here"; it does not rank the two, which
would need blind trials across many utterances — the deferred Quality Evaluation.

**What this settles and what it does not.** It settles the audio question for
this variant: the gate is not buying perceptual quality. It does not settle the
validation question, because the gate's function is to keep tolerances tight
enough to detect porting bugs, and that property is independent of audibility.
The gate is also CUDA-only: Port Validation phases 1 through 4 run on CPU against
the CPU oracle and are unaffected either way, so only the phase-5 backend rerun
is in scope.

## Decision (2026-07-26): remove the gate

jiangzhuo decided to remove the gate rather than keep it or make it a separately
declared configuration. What was done, in one change:

- The five group-5 hunks were dropped from
  `ggml-patches/0001-synthesize-local.patch`, taking it from 10 files and 17
  hunks to 6 files and 11 hunks. `ggml/` was regenerated with
  `scripts/sync-ggml.sh`; only those five files changed.
- `SYNTH_CUDA_TF32` was deleted from `CMakeLists.txt` and `CMakePresets.json`.
  This part was not optional. Left in place, it would have set a cache variable
  that nothing reads: configuration and build would still succeed while the
  documented default silently did nothing.
- `tests/backend_plan_test.cpp` asserted strict-FP32 accuracy on a 64-column
  matmul, which routes through cuBLAS and therefore now computes in TF32. The
  assertion became a TF32-scale *relative* bound of 5e-3 rather than being
  weakened to a finiteness check, so it still fails a misplaced or broken matmul.
- `tests/python/test_cmake_presets.py` asserted the option's value; it now
  asserts the option is absent, so no preset can reintroduce a knob the build no
  longer defines.
- `docs/backends.md`, `docs/testing.md`, `docs/porting/families/vits.md`, and
  `docs/models/kokoro-v1-0.md` had the strict-FP32 claims replaced. The VITS
  strict-FP32 measurement table is explicitly marked as measured under the
  removed gate and pending re-measurement rather than left to read as current.

Two consequences to keep in view. Any family whose validation depends on
per-stage forward fidelity rather than end-to-end comparison — an autoregressive
codec LM is the case in point — has a weaker handle under TF32 than under strict
FP32, because a 1e-3 logits perturbation flips sampled tokens far more readily
than 1e-7 does. This decision was taken on Kokoro evidence and should not be
extended to such a family without measuring it there. And the reversal route is
narrow: upstream declined the backend-flag shape, so the only way back is an
op-level strict-FP32 value in `ggml_prec`, set by family graph builders on
individual sensitive matmuls.
