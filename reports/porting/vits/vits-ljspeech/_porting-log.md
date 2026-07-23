# vits-ljspeech Porting Log

## 2026-07-22 — Intake and oracle smoke

- Pinned `jaywalnut310/vits` at commit
  `2e561ba58618d021b5b8323d3765880f7e0ecfdb`.
- Downloaded only the official `pretrained_ljs.pth` checkpoint into the ignored
  local model cache. No LJSpeech or VCTK dataset and no VCTK checkpoint was
  downloaded.
- Verified the 145,599,717-byte checkpoint at SHA-256
  `c94fb49d08ba90c598de16e7d5dec8d26bf225c1cf193a4fba05eb2dbda5a561`.
- Inspected the checkpoint with PyTorch `weights_only=True`; all 838 model tensor
  entries load strictly into the 36,321,072-parameter LJSpeech architecture.
- Locked the project-owned CPU reference environment in
  `scripts/envs/vits/uv.lock`. It uses Python 3.12.13 and Torch 2.7.1 on Linux
  AArch64 rather than the unavailable upstream Python 3.7 and Torch 1.6 stack.
- Built the upstream monotonic-alignment extension in the ignored source cache and
  verified the full source import.
- Resolved the upstream example to 33 blank-interspersed token IDs using
  `english_cleaners2`, phonemizer 3.3.0, and eSpeak 1.51.
- Ran `VITS is Awesome!` twice on the CPU oracle with seed 0,
  `noise_scale=0.667`, `noise_scale_w=0.8`, and `length_scale=1`. Both runs
  produced the same 25,344-frame F32 PCM hash
  `a99554e756576a1b9a42724d9b5280dea6c94eb7f8dde6cf6921e0b512bfa15e`.

## 2026-07-22 — Golden Manifest and reference dump

- Added the 12-case `vits-ljspeech` Golden Manifest and materialized all 204
  declared files under the ignored `build/goldens/` tree.
- Captured unscaled duration and acoustic-latent random tensors with a local
  PyTorch generator; the public seed API remains separate from the internal
  parity-replay seam.
- Reimplemented the pinned inference path only inside the dumper and compared
  every case against the original `model.infer`. Attention, expanded priors,
  latent tensors, flow output, and PCM match exactly.
- Preserved the non-contiguous expanded-prior strides when reproducing upstream
  `randn_like`; generating by shape alone assigns the same random sequence to
  different logical indices and fails the exact self-check.
- Verified every generated file path, byte count, and recorded SHA-256. The three
  seed cases have distinct PCM, and slow/default/fast rate cases contain 81,152,
  69,120, and 59,136 frames respectively.
- Kept the runtime package contract token-ID-only by declaring `frontend: null`;
  use of the pinned eSpeak cleaner by the offline oracle does not claim a runtime
  text frontend.

## Publication blocker

The pinned source code has an MIT license. The upstream README links the external
Google Drive checkpoint, but the README and Drive folder do not state a separate
license or redistribution permission for the checkpoint bytes. Local conversion
and technical validation may continue under project policy, but no converted model
is uploaded to the project Hugging Face organization until checkpoint
redistribution permission is established.

## 2026-07-22 — Source-F32 GGUF conversion

- Added the manifest-driven `scripts/convert-vits.py` checkpoint adapter and a
  small shared GGUF convention helper. The public converter interface has four
  required paths and intentionally has no quantization or tensor-selection
  switches.
- Pinned `gguf` 0.19.0 in the existing VITS `uv` environment and added hard
  package limits of 512 token IDs and 1,323,000 native output frames.
- Preserved the 178-row source vocabulary exactly, including duplicate ASCII
  apostrophe IDs 174 and 176 and the upstream last-index-wins lookup rule.
- Fused all 140 dimension-0 PyTorch weight-normalization pairs exactly. Emitted
  460 runtime F32 tensors and explicitly reported all 238 omitted logical tensors;
  all 838 original checkpoint entries are accounted for.
- Reopened the generated GGUF and verified authoritative metadata plus every
  tensor's canonical name, type, reversed GGML shape, and payload bytes.
- Ran the finalized conversion twice. Both runs produced the identical
  113,244,864-byte SHA-256
  `44d5fbfab510ad39b26a5224d77cf9e7d7cd8e2e520bdbdf76a42d62e4f929aa`.
- Kept the generated GGUF and detailed converter report ignored. The checkpoint
  license blocker still prevents project Hugging Face publication.

## 2026-07-22 — CPU C++ text-encoder slice

- Vendored the exact ggml snapshot used by pinned transcribe.cpp and added the
  initial offline CMake library build plus the confirmed C ABI/version bootstrap.
- Added a strict 111-tensor VITS text-weight catalog and a deep model module that
  owns GGUF, backend, weight-buffer, scheduler, and compute-context lifetimes.
- Implemented all six text blocks, including query-dependent relative-key and
  relative-value attention, with backend-independent GGML operations.
- Kept F32 convolution exactness by using `im2col + mul_mat`; the pinned generic
  Conv1d wrapper's forced-F16 path is not used for the F32 accuracy artifact.
- Normalized GGML's channel-fastest storage inside the module so validation and
  future adapters do not depend on backend tensor layout.
- Ran all 12 cases at one CPU thread. `text.mask` is exact; worst max-absolute
  drift is `1.2636185e-5` for `text.m_p` and `4.7087669e-6` for
  `text.logs_p`, both on `ljs-long`.
- Rejected a 513-token request and an out-of-range token ID before graph compute.
  The ABI smoke and upstream-default real-model case pass under ASan and UBSan.
- Left tolerances unsigned and Stage 4 incomplete. Duration, acoustic flow,
  decoder, full waveform parity, public model/context synthesis operations, and
  non-CPU backend validation remain outstanding.

## 2026-07-22 — Per-slice automated test gate

- Established the repository rule that every implementation slice must add and
  pass focused unit tests before the next slice begins; semantic graph changes
  additionally require the affected Golden integration suite and sanitizers.
- Added six model-independent C/C++ test executables covering all current C
  Interface statuses and structure-size behavior, VITS host request preparation,
  relative indices, output layout, GGUF metadata, the text tensor catalog, graph
  construction, and model-load error paths.
- Backfilled 18 locked Python unit cases for the completed oracle and converter
  stages: manifest validation, source selection, configuration derivation,
  dimension-0 weight normalization, namespace mappings, skip accounting, F32
  conversion, atomic writes, seed parsing, and artifact-path containment.
- Registered the 12-case real-model text comparison as an explicit CTest with
  `integration`, `golden`, and `vits` labels. The model-independent unit suite and
  the complete Golden suite both pass on the DGX Spark CPU path.
- Added `SYNTH_SANITIZE` so the same registered unit tests can run reproducibly
  under AddressSanitizer and UndefinedBehaviorSanitizer. All seven CTest unit
  entries and the full 12-case Golden integration entry pass in that build.

## 2026-07-22 — CPU stochastic-duration predictor slice

- Added the strict 114-tensor duration catalog and validated all stochastic
  duration metadata, including DDS depth, reverse-flow count, spline constants,
  and default `noise_scale_w`.
- Added a private replay seam that accepts the oracle's unscaled logical `[2, T]`
  duration noise, validates finite values, and converts it to GGML's physical
  channel-fastest layout. No public C API or library-owned RNG contract was added.
- Implemented the main condition network, three reverse ConvFlows, four flips,
  ten-bin rational-quadratic inverse splines, and inverse affine with portable
  GGML operations. F32 depthwise convolutions use explicit `im2col + mul_mat`
  with dilations 1, 3, and 9.
- Added host, tensor-catalog, graph-construction, invalid-input, and actual CPU
  graph-execution unit coverage. The execution test caught and fixed GGML CPU's
  requirement that scale inputs materialized from channel views be contiguous.
- Ran all 12 Golden cases at one CPU thread. Worst `duration.logw` drift is
  `1.7452985e-5` max absolute and `1.2574333e-6` mean absolute on `ljs-long`;
  every output has the expected element count and finite values.
- Registered duration replay as its own `integration;golden;vits` CTest. Release
  passes 10/10 unit entries and both 12-case Golden entries. The same complete
  suites pass under AddressSanitizer and UndefinedBehaviorSanitizer.
- Kept Stage 4 and numerical tolerances open. Duration rounding/path generation,
  expanded priors, latent sampling, acoustic flow, decoder, PCM parity, public
  synthesis operations, and non-CPU backend validation remain outstanding.

## 2026-07-22 — CPU duration-path slice

- Added strict GGUF loading and validation for `hop_length`, native maximum
  output frames, and minimum/maximum speaking rate. The default rate `1.0` must
  remain inside the declared range.
- Added a pure host Module at the dynamic-shape seam after `duration.logw`. Its
  small Interface accepts log durations and semantic speaking rate, while hiding
  inverse-rate mapping, F32 exp/ceil, length clamping, output-limit arithmetic,
  and dense monotonic-path generation.
- Enforced the native PCM limit as `Y * hop_length <= max_output_frames` before
  allocating attention. Hardened float-to-`uint64_t` conversion for malicious
  metadata near `2^64` and covered cumulative overflow independently.
- Added exact unit cases for ordinary paths, both rate boundaries, underflow to
  zero duration, minimum one-frame behavior, invalid rates and values, token and
  cumulative frame limits, physical attention layout, and malformed metadata.
- Added `Model::run_duration` and a thin real-model Adapter. Across all 12 Golden
  cases, every `duration.w_ceil`, `duration.y_length`, and
  `duration.attention` element is exactly equal to the pinned PyTorch oracle.
- Release passes 11/11 unit entries and all three 12-case Golden entries. The
  same complete suites pass under AddressSanitizer and UndefinedBehaviorSanitizer;
  the post-review overflow hardening was separately rerun through its affected
  sanitizer unit and 12-case Golden gate.
- Kept Stage 4 and numerical tolerances open. Expanded priors, latent sampling,
  acoustic flow, decoder, PCM parity, public synthesis operations, and non-CPU
  backend validation remain outstanding.

## 2026-07-22 — CPU expanded-prior slice

- Added host and GGML Modules for prior expansion. Their Interfaces hide physical
  channel-fastest layout, PyTorch output layout, dynamic `Y`, and the two
  attention projections from Model callers.
- Refactored duration execution so one text+duration graph computes `logw`,
  `text.m_p`, and `text.logs_p` together. Prior validation no longer reruns the
  text encoder before building the second graph.
- Added actual CPU graph tests with known two-channel values and a repeated-token
  attention path, plus host tests for domain-shape agreement, finite inputs,
  layout conversion, malformed attention, and output error mapping.
- Added `Model::run_prior_expansion` and a thin real-model Adapter. All 12 cases
  produce finite, correctly shaped expanded priors. Worst drift on `ljs-long` is
  `1.2636185e-5` max absolute and `1.0283067e-7` mean absolute for `m_p`, and
  `4.7087669e-6` max absolute and `4.5241178e-8` mean absolute for `logs_p`.
- The exact attention projection introduces no greater worst-case error than the
  corresponding text tensors. Tolerances remain unsigned until the complete F32
  graph has been measured.
- Release passes 13/13 unit entries and all four 12-case Golden entries. The same
  complete suites pass under AddressSanitizer and UndefinedBehaviorSanitizer.
- Kept Stage 4 open. Latent sampling, acoustic flow, decoder, PCM parity, public
  synthesis operations, and non-CPU backend validation remain outstanding.

## 2026-07-22 — CPU latent-sampling slice

- Added host and GGML Modules for latent sampling. The small Interface accepts
  expanded prior statistics, caller-replayed logical `[C,Y]` noise, and
  `noise_scale`; it hides validation and GGML's channel-fastest layout.
- Preserved the oracle's stride-sensitive `randn_like` semantics by consuming the
  already assigned logical values from `random.latent_noise`. The C++ library does
  not pretend to reproduce PyTorch's generator or expose a framework-specific RNG
  contract.
- Implemented `latent.z_p = m_p + latent_noise * exp(logs_p) * noise_scale` with
  portable GGML elementwise operations and added `Model::run_latent_sampling`
  without expanding the public C ABI.
- Added host tests for layout, shape, finite-value, scale, and output-error rules,
  plus an actual CPU GGML graph test with exact known values. The new tests first
  failed at the unimplemented seam before the implementation was added.
- Added a thin real-model Adapter and a fifth 12-case Golden comparison. Every
  output is finite and correctly shaped; worst `latent.z_p` drift is
  `1.5258789e-5` max absolute and `1.3051576e-7` mean absolute on
  `ljs-rate-slow`.
- Release passes 15/15 unit entries and all five 12-case Golden entries. The same
  suites pass under AddressSanitizer and UndefinedBehaviorSanitizer; the complete
  sanitizer Golden run took 546.64 seconds.
- Kept Stage 4 open. Acoustic flow, decoder, PCM parity, public synthesis
  operations, and non-CPU backend validation remain outstanding.

## 2026-07-22 — CPU reverse acoustic-flow slice

- Added strict metadata and an 80-tensor flow catalog for four mean-only
  residual coupling blocks. Each block loads pre/projection convolutions and
  four WN input plus residual/skip convolution pairs.
- Added a deep acoustic-flow Module whose small Interface accepts logical
  `latent.z_p` and returns logical `flow.z`, while its host Adapter hides GGML's
  channel-fastest layout and validates shapes and finite values.
- Implemented four reverse coupling blocks with kernel-5 F32 WN convolutions,
  tanh/sigmoid gates, residual/skip accumulation, and reverse mean subtraction.
  No backend-specific operation or public C ABI expansion was added.
- The first 2-channel graph test could not distinguish a full channel reversal
  from a half swap. Initial real-model comparison exposed max drift `7.69`; a
  new 4-channel block-specific-bias test failed first, then drove the fix to a
  shared I32 full-channel reversal. Final worst drift fell to `6.6339970e-5`
  max absolute and `1.1114858e-6` mean absolute on `ljs-rate-slow`.
- Added metadata overflow coverage so hostile dilation and padding cannot exceed
  the GGML integer interface. Added catalog, host, actual CPU graph, reverse
  order, invalid-input, and malformed-weight coverage.
- Added `Model::run_acoustic_flow`, a thin real-model Adapter, and the sixth
  12-case Golden comparison. Every final `flow.z` is finite and correctly shaped.
- Release passes 18/18 unit entries and all six Golden entries. The same suites
  pass under AddressSanitizer and UndefinedBehaviorSanitizer; the complete
  sanitizer Golden run took 730.30 seconds and acoustic flow took 180.96 seconds.
- Kept Stage 4 open. The waveform decoder, PCM parity, public synthesis
  operations, and non-CPU backend validation remain outstanding.

## 2026-07-22 — CPU waveform-decoder slice

- Added the complete 155-tensor decoder catalog, four transpose-convolution
  stages, all ResBlock1 branches, the final convolution, and tanh.
- Kept transpose-convolution padding inside a shared backend-portable GGML
  operation by computing the full result and taking a checked symmetric crop.
- Added host, weight, graph, transpose-convolution, malformed-input, and actual
  CPU execution tests before the implementation.
- All 12 Golden cases produce finite PCM with exactly `Y * 256` frames. Worst
  max-absolute drift is `2.8620008e-4`, and worst mean-absolute drift is
  `1.7319487e-6`, both measured on `ljs-long`.

## 2026-07-22 — Public C synthesis and reference CLI slice

- Expanded the versioned C Interface with opaque Model and Context lifecycle,
  strict capability/language/Preset Voice queries, a size-tagged synthesis
  request, Audio Sink, result metadata, and an owned complete-buffer Adapter.
- Added a backend-independent normal random stream. A concrete seed repeats the
  same logical duration and latent draws; `SYNTH_SEED_RANDOM` never returns its
  sentinel as the reported replay seed.
- Added request validation for token IDs, English regional fallback, the fixed
  default Voice, speaking rate, cancellation, and request/model output limits.
  The public output limit rejects after duration resolution and before waveform
  decoding.
- Added `synthesize-cli` as an optional Adapter that includes only the installed
  C header. It maps text, phoneme, and token-ID options to one request and writes
  native interleaved F32 audio as an IEEE-float WAV.
- The real public test proves bit-identical seed-42 PCM and a different seed-43
  result. The real CLI test proves the same relation at the WAV-file level and
  verifies unsupported text input leaves no file.
- Release passes 27/27 registered unit tests and 10/10 integration tests. The
  ASan/UBSan build passes all 27 unit tests plus the two public Interface tests
  and the real CLI test. A shared-library gate also rejects every non-`synth_*`
  dynamic export.
- Stage 4 is complete. Stage 5 numerical threshold acceptance, F16/quantization,
  non-CPU backend validation, and publication remain separate work; checkpoint
  redistribution permission continues to block Hugging Face upload.

## 2026-07-22 — Public device discovery and model placement query slice

- Added project-owned device types, memory flags, device enumeration, runtime
  backend-availability probes, and Loaded Model actual-device queries to the
  stable C Interface. No GGML type or handle crosses the public boundary.
- Added one internal GGML registry Adapter for CPU/GPU/IGPU/ACCEL classification,
  extensible backend kind names, live memory snapshots, and strict global device
  index resolution. The transcribe.cpp-derived classification design retains its
  MIT attribution in `THIRD_PARTY_NOTICES.md`.
- Changed VITS CPU loading to initialize the exact resolved GGML device. The
  public query reads the device attached to the model's owned backend, rather than
  echoing load parameters.
- Added a C-only public ABI test, pure classification/selection unit tests, and
  real-model lifecycle assertions. Explicit GPU requests still fail for VITS;
  runtime device availability is deliberately separate from Model Family support.
- Release passes 29/29 registered unit tests and 10/10 integration tests. The
  shared public ABI/export gate passes 4/4, and ASan/UBSan passes all 29 unit
  tests plus both real public Interface tests and the real CLI test.

## 2026-07-22 — Model-owned Backend Plan slice

- Added a model-level Backend Plan that owns the initialized primary backend,
  optional host-memory accelerator backends, and CPU fallback in one lifetime.
  The Plan keeps the primary backend separate from scheduler priority order and
  enforces GGML's CPU-last fallback invariant.
- Moved scheduler creation and registry-based thread configuration behind the
  Plan. All six GGML execution sites covering the seven VITS Golden stages now
  use the same model-owned plan; no family path calls
  `ggml_backend_cpu_set_n_threads()` or builds a one-element backend array.
- VITS weight allocation targets only `plan.primary()`. The existing actual
  device query therefore remains tied to the backend that owns model weights,
  even when later GPU plans add CPU fallback execution.
- Added a CPU Backend Plan lifecycle and graph-compute unit test, including
  strict CPU order and the `CPU_ACCEL` ordering invariant. This is CPU
  equivalence evidence only; the unvalidated GPU path is not exposed by the
  VITS public loader yet.
- Release passes 30/30 registered unit tests and 10/10 integration tests. The
  shared public ABI/export gate passes 4/4, and ASan/UBSan passes all 30 unit
  tests plus both real public Interface tests and the real CLI test.

## 2026-07-22 — Explicit CUDA experimental execution slice

- Added strict CUDA device resolution to the public VITS loader while keeping
  `AUTO` on the Supported CPU path. An explicit registry index must match CUDA;
  no request silently substitutes CPU or another backend. The Loaded Model query
  reports the actual CUDA device owned by the model.
- Extended Backend Plan tests for CUDA-primary/CPU-last ordering, strict FP32
  matrix behavior, primary assignment of weightless graphs, view-aware placement
  accounting, and CPU/CUDA transpose-convolution execution.
- Anchored prior expansion and latent sampling to the primary backend. On GB10,
  all executable nodes are now on CUDA: 893 duration-stage, 6 prior, 4 latent,
  357 acoustic-flow, and 516 decoder nodes. Executable CPU fallback is zero and
  every stage has one CUDA split; zero-compute views are counted separately.
- Added `SYNTH_DEBUG_BACKEND_PLACEMENT` as an internal diagnostic and extended all
  seven thin Golden runners and validators with an explicit CPU/CUDA selection.
  This does not add GGML handles or family-specific operations to the public ABI.
- Made strict FP32 the default CUDA math policy. `SYNTH_CUDA_TF32=ON` is an
  explicit lower-precision experiment. TF32 reached max-absolute drift
  `8.6592436e-3` at `text.m_p` and `1.1620114e-1` at final PCM; strict FP32
  reduces them to `1.1280179e-5` and `7.4365083e-4`.
- Replaced the GGML CUDA transpose-convolution kernel's full-input inner scan with
  an equivalent kernel-width traversal while preserving accumulation order. The
  12-case decoder validation fell from 221.61 seconds to 10.62 seconds without
  changing its measured output. `ljs-long` takes 1.30 seconds on CUDA versus
  14.16 seconds on one-thread CPU on the same DGX Spark.
- The development environment is GB10 compute capability 12.1, driver
  580.159.03, CUDA toolkit 13.0.88, CMake 3.28.3, and native `sm_121a`. All seven
  12-case CUDA reports complete; duration structure is exact, outputs are finite
  and correctly shaped, and public CUDA lifecycle plus synthesis pass.
- Release CPU regression passes 40/40 tests. The shared public ABI/export gate
  passes 4/4. ASan/UBSan passes 30/30 unit tests plus the three real public paths.
  The CUDA build passes 29/29 C/C++ unit tests on the physical GB10.
- This remains Experimental rather than Supported because the local system does
  not have the release-locked CUDA 13.3 Update 1 toolkit. The aggregate evidence
  is recorded in
  `reports/validate/vits/vits-ljspeech-cuda-experimental.json`.

## 2026-07-22 — CUDA 13.3 Update 1 physical Linux checkpoint

- Installed the official CUDA 13.3 Update 1 toolkit (compiler 13.3.73) into
  `/home/jiangzhuo/.local/cuda-13.3` on both hosts without replacing either
  system CUDA installation or R580 driver. The official runfiles matched
  NVIDIA's published byte sizes and MD5 checksums before installation.
- Built native `sm_121a` cubins on Ubuntu 24.04 AArch64 DGX Spark/GB10 with
  driver 580.159.03 and native `sm_89` cubins on Ubuntu 22.04 x86-64 RTX 4070
  SUPER with driver 580.173.02. Binary inspection and dynamic-library resolution
  confirm native cubins and CUDA 13.3 `cudart`/`cublas` on both hosts.
- Each host passes 30/30 C/C++ unit tests, public model lifecycle, public CUDA
  synthesis, reference CLI synthesis, and twenty repeated public synthesis
  processes with no residual synthesize process. The seven validators each complete
  all 12 Golden cases.
- Both hosts retain the established all-CUDA Backend Plan: 893 duration-stage,
  6 prior-expansion, 4 latent-sampling, 357 acoustic-flow, and 516 decoder
  executable nodes; executable CPU fallback is zero and each stage has one
  CUDA split.
- DGX Spark strict-FP32 worst PCM max-absolute drift is `7.4365083e-4`; RTX 4070
  SUPER is `9.7708963e-4`. Duration structure is exact on both. Numerical
  acceptance thresholds remain deliberately deferred rather than inferred from
  these measurements.
- For `ljs-long`, DGX Spark records 1.20 seconds CUDA versus 27.50 seconds
  one-thread CPU; RTX 4070 SUPER records 0.44 seconds versus 18.61 seconds after
  the stride-aware transpose-convolution optimization described below.
  Minimal public synthesis peaks at approximately 370 MiB and 326 MiB of GPU
  process memory respectively.
- The release-toolkit physical-Linux checkpoint is complete, but the `cu13`
  Provider remains Experimental pending numerical thresholds, physical R610 and
  Windows checks, complete release cubin builds, and clean-runtime packaging.
  Evidence is aggregated in
  `reports/validate/vits/vits-ljspeech-cuda-13.3-linux.json`.

## 2026-07-22 — CUDA transpose-convolution stride optimization

- Split the apparent 1.30-second DGX Spark result into model loading and repeated
  synthesis. CUDA 13.0 and 13.3 both recorded the same pre-change end-to-end
  result, so this was not a toolkit regression. Before optimization, median
  loaded-model `ljs-long` synthesis was 606.492 ms on GB10 and 324.454 ms on RTX
  4070 SUPER. Model loading alone was 318.459 ms and 110.297 ms respectively.
- Nsight Systems attributes approximately 89.1% of the profiled RTX GPU time to
  transpose convolution, im2col, and SGEMM. The pre-change steady-state host
  ratio of about 1.87 closely matches the theoretical memory-bandwidth ratio:
  approximately 504 GB/s for the RTX card versus 273 GB/s for DGX Spark. The
  strict-FP32 graph does not use DGX Spark's headline sparse-FP4 performance.
  Host-to-device transfers were faster on the unified-memory GB10 and were not
  the bottleneck. GB10 also spent substantially longer loading CUDA libraries in
  a fresh process, explaining part of the cold CLI difference.
- Added the exact VITS 16/8 kernel/stride boundary-and-overlap unit case before
  changing production code. The CUDA kernel now enumerates only weights whose
  indices are congruent to the output index modulo the stride, retaining the
  prior descending accumulation order. VITS 16/8 and 4/2 operations visit two
  weights per output instead of scanning 16 or 4.
- The optimized median loaded-model time is 403.555 ms on GB10 (33% lower) and
  187.437 ms on RTX 4070 SUPER (42% lower). End-to-end `ljs-long` is 1.20 and
  0.44 seconds; the 12-case decoder suites are 10.29 and 3.92 seconds. Repeated
  outputs are exact, and both hosts retain their established Golden drift.
- Release CPU passes 40/40, the public shared ABI/export gate passes 4/4,
  ASan/UBSan passes 30/30 unit tests plus all three real public paths, and both
  physical CUDA builds pass 29/29 unit tests. The support state remains
  Experimental; this optimization does not waive any release gate.

## 2026-07-22 — Pointwise Conv1D graph optimization

- Added a backend-executed Conv1D unit test before changing production code. It
  covers multi-input/multi-output kernel-one channel mixing with bias and a
  separate padded, dilated regular convolution on CPU and CUDA.
- The first direct pointwise GEMM experiment changed operand orientation. It
  passed the small exact unit case but caused accumulated real-model drift up to
  `3.0473e-3` at `text.m_p` and `3.4237e-2` at `flow.z`; Golden validation caught
  it and that implementation was rejected. The retained implementation removes
  only the redundant input transpose and kernel-one `im2col`, preserving the old
  GEMM operand order, bias layout, output transpose, and established drift.
- On the RTX 4070 SUPER six-synthesis profile, `im2col` launches fall from 1512
  to 828 and transpose-copy launches from 3246 to 2598. Same-session 20-sample
  A/B medians are 409.220 versus 409.597 ms on GB10 (+0.09%, noise-level) and
  182.937 versus 170.592 ms on RTX 4070 SUPER (-6.75%). All repeated PCM remains
  bitwise exact for a fixed request.
- The optimized graphs retain one CUDA split and zero executable CPU fallback.
  Duration-stage placement changes from 1434 total / 541 view / 893 executable
  nodes to 1254 / 451 / 803. Acoustic flow changes from 669 / 312 / 357 to
  573 / 264 / 309. Prior, latent, and decoder placement is unchanged.
- Final regression passes CPU Release 41/41, shared ABI/export 4/4, ASan/UBSan
  31/31 unit tests plus all three real public paths, and 30/30 unit tests on each
  physical CUDA host. Both hosts also pass public C API synthesis, reference CLI,
  and all seven 12-case Golden stages without changing established worst drift.
  VITS/CUDA remains Experimental under the same release gates.

## 2026-07-22 — Tiled F32 Conv1D im2col optimization

- Added the exact backend test before changing production code. Its N=1,
  IC=31, IW=OW=65, KW=7, padding=9, and dilation=3 shape crosses both dimensions
  of a 32-by-32 tile and checks every copied or zero-padded F32 value. The old
  generic kernel and the new kernel both pass it on CPU, GB10, and RTX 4070
  SUPER with zero executable CPU fallback.
- Added a shared-memory tiled CUDA path for large batch-one, unit-stride F32
  Conv1D. A 32-by-8 block loads a conceptual 32-by-32 tile using coalesced input
  reads, then transposes through padded shared memory for coalesced output
  writes. GGML's generic implementation remains the fallback for 2D, F16,
  batches, non-unit stride, small patches, and small output widths.
- In a six-synthesis RTX profile, 684 of the existing 828 `im2col` launches use
  the tiled path and 144 remain generic. Their combined time falls from
  333.062081 to 273.022617 ms (-18.03%); total launch count is unchanged. SGEMM
  becomes the largest single kernel family at 276.626099 ms.
- Same-session 20-sample A/B medians improve from 409.5965 to 398.155 ms on GB10
  (-2.79%) and from 170.5915 to 168.5085 ms on RTX 4070 SUPER (-1.22%). Every
  repeated fixed-seed PCM result remains bitwise exact.
- Final regression passes CPU Release 41/41, shared ABI/export 4/4, ASan/UBSan
  31/31 unit tests plus all three real public paths, and CUDA 30/30 on both
  physical hosts. Both hosts additionally pass public lifecycle, CLI, 20/20
  independent public synthesis processes, and all seven 12-case Golden stages;
  no residual synthesize process remains and all executable placement is CUDA.

## 2026-07-22 — Waveform-decoder internal layout optimization

- Added a backend-executed two-convolution equivalence test before changing the
  decoder. It compares the established `[C,T]` path with an internal `[T,C]`
  chain across leaky-ReLU, padding, dilation, non-tile-aligned 65-frame tensors,
  and 17/31/5 channels. CPU, GB10, and RTX 4070 SUPER are element-for-element
  exact and have zero executable CPU fallback.
- Kept the layout implementation private to the waveform decoder Module rather
  than adding layout selection to the operations Interface or public C ABI.
  Ordinary convolutions, activations, residuals, branch averaging, and transpose
  convolutions now remain `[T,C]`; only the decoder input and final PCM contract
  cross a transpose seam.
- Decoder placement falls from 979 total / 463 view / 516 executable nodes to
  671 / 309 / 362. It remains one CUDA split with zero executable CPU fallback.
  The six-synthesis RTX profile removes 924 transpose-copy launches, from 2598
  to 1674, and cuts transpose-copy time from 79.572389 to 13.812389 ms (-82.64%).
  Total profiled GPU-kernel time falls from 916.369242 to 869.357987 ms (-5.13%).
- Interleaved 20-sample A/B medians improve from 403.8775 to 367.101 ms on GB10
  (-9.11%) and from 167.8325 to 160.952 ms on RTX 4070 SUPER (-4.10%). Every
  fixed-seed benchmark output is bitwise exact and the real decoder retains its
  established 12-case Golden drift on both hosts.
- Final regression passes CPU Release 41/41, shared ABI/export 4/4, ASan/UBSan
  31/31 unit tests plus all three real public paths, CUDA 30/30 on both physical
  hosts, public lifecycle and CLI, 20/20 independent public synthesis processes,
  and all seven 12-case Golden stages. No synthesize process remains.

## 2026-07-22 — Expanded Conv1D im2col tile

- Extended the exact im2col test from 65 frames to also cover 257 frames before
  changing the kernel. The larger case spans two complete 128-wide output tiles,
  a one-element output tail, seven patch tiles, padding, and dilation;
  it passes on CPU and both CUDA hosts before and after the production change.
- Expanded the F32 Conv1D tile from 32 output positions by 32 patch elements to
  128 by 32. A linearized 256-thread block cooperatively loads and stores the
  4096-element tile through a padded 32-by-129 shared array, preserving coalesced
  global access while reducing block and synchronization count by four along the
  long decoder output dimension.
- Interleaved 20-sample A/B medians improve from 366.145 to 360.9795 ms on GB10
  (-1.41%) and from 155.754 to 154.918 ms on RTX 4070 SUPER (-0.54%). Every
  fixed-seed result is bitwise exact. The six-synthesis RTX profile reduces
  tiled im2col time from 279.465616 to 268.831098 ms (-3.81%) and total GPU
  kernel time from 869.357987 to 860.390285 ms (-1.03%).
- Final regression passes CPU Release 41/41, shared ABI/export 4/4, ASan/UBSan
  31/31 unit tests plus all three real public paths, CUDA 30/30 on both physical
  hosts, public lifecycle and CLI, 20/20 independent public synthesis processes,
  and all seven 12-case Golden stages. Placement remains one CUDA split per
  stage with zero executable CPU fallback, and no synthesize process remains.

## 2026-07-22 — Committed CUDA build presets and toolchain guard

- Added the five confirmed CUDA configure/build presets: DGX Spark development,
  isolated DGX UVM research, Linux AArch64 cu13 release, RTX Linux development,
  and Linux x86-64 cu13 release. Every preset has an isolated
  `build/<preset-name>` tree, strict FP32, an explicit native cubin target set,
  and matching build entry; development configurations also have test presets.
- Release configurations use exactly `121a-real` on Linux AArch64 and
  `75-real;80-real;86-real;89-real;90-real;100-real;120a-real` on Linux x86-64.
  The configure-time release policy rejects a CLI override, including `native`,
  `all`, `all-major`, or a target subset. UVM remains absent from build cache
  variables and is set only in the isolated UVM test process environment.
- Raised the implemented CMake floor from 3.16 to the already confirmed 3.24
  contract and added an exact CUDA 13.3 minor-version gate. A real first
  configure caught the 13.3 compiler being paired with `/usr/local/cuda` 13.0
  headers; after the gate, both CMake toolkit discovery passes and NVCC resolve
  to the same user-local CUDA 13.3.73 installation.
- The new preset/toolchain contract test passes 6/6, including valid and invalid
  release target sets and CUDA version cases. A clean `dev-dgx-spark` preset
  build completes and its complete CTest unit gate passes 31/31. A real Release
  configure rejects an `89-real` override for AArch64, then accepts the committed
  `121a-real` set with Release, shared-library, tests-off packaging posture.

## 2026-07-22 — Complete Linux CUDA Release cubin builds

- Built the committed CUDA 13.3 Release presets on both physical Linux hosts.
  AArch64 contains exactly native `sm_121a`; x86-64 contains exactly native
  `sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_90`, `sm_100`, and `sm_120a`.
- Added `synthesize-check-release-cubins`, which uses the selected toolkit's
  `cuobjdump` to reject a missing or extra cubin and all embedded PTX. Both
  complete Release binaries pass the automated check.
- Reconfigured and relinked both Release trees after the SDK/ABI work below;
  the exact cubin inventories remain unchanged. These are native source-tree
  Release builds, not evidence of manylinux policy or clean-runtime wheels.

## 2026-07-22 — Installed native C SDK and ABI packaging gate

- Added install rules for the canonical C header, synthesize and GGML native
  libraries, optional CLI, a relocatable `synthesize::synthesize` CMake package,
  and relocatable `synthesize.pc` metadata. Both static and shared SDK forms are
  consumed through the same public C header and target name.
- Corrected Linux shared-library compatibility naming to use
  `SYNTH_ABI_VERSION`: the installed SONAME is now `libsynthesize.so.1`, with
  the semantic version retained only in the fully versioned file. Installed
  shared libraries use `$ORIGIN` so the SDK tree remains relocatable.
- Added a registered packaging unit test that installs into a fresh prefix,
  builds and runs pure-C CMake and pkg-config consumers, and exercises
  `pkg-config --static` for archive builds. It caught and then locked down the
  CUDA runtime, cuBLAS, driver, C++ runtime, and platform dependencies missing
  from the original static package metadata.
- The RTX shared-SDK check also exposed that its user-local CUDA 13.3 runtime is
  not registered with the system loader, while DGX Spark could silently resolve
  the same SONAME from `/usr/local/cuda`. CUDA packaging tests now set the
  requested loader path and use `ldd` to require CUDA 13.3 `cudart` and `cublas`
  from that root rather than accepting another host toolkit.
- CPU static passes 32/32 tests, the public CPU shared gate passes 5/5, and both
  DGX Spark and RTX 4070 SUPER CUDA development builds pass 32/32. Both complete
  shared CUDA Release installations also pass the isolated CMake/pkg-config
  consumer test, SONAME check, `$ORIGIN` check, and exact-cubin/no-PTX gate.

## 2026-07-22 — Python Native Providers and manylinux checkpoint

- Implemented the `synthesize-cpp-native` default Provider and the
  `synthesize-cpp-native-cu13` CPU-plus-CUDA Provider as separate `py3-none`
  distributions. Their entry-point descriptors stamp Provider identity,
  release, public-header SHA-256, artifact directory, and ordered backend set.
  CUDA preparation resolves and preloads only absolute package paths, retains
  handles, is thread-safe and idempotent, never mutates `PATH` or
  `LD_LIBRARY_PATH`, and fails hard once selected.
- Added the component-filtered CMake wheel install. It carries one unversioned
  physical copy of synthesize and each required GGML shared library with
  `$ORIGIN`, plus Python/JSON contracts, while rejecting C SDK headers, CMake
  metadata, pkg-config files, and `.so.*` symlink duplicates. Provider and
  manylinux packaging tests now pass 11/11; the component stage is part of the
  registered unit gate. DGX Spark and RTX development builds both pass 33/33.
- Pinned scikit-build-core 1.0.3 and Ninja 1.13.0, assigned Provider-owned
  scikit-build directories, and stopped Python wheel builds from sharing a
  committed Release preset tree. The accidentally reused RTX tree was restored
  with CMake 3.31, rebuilt, and again passed exact seven-cubin/no-PTX and
  `libsynthesize.so.1` SONAME checks.
- Added the native-architecture manylinux builder with digest-pinned glibc 2.28
  images, read-only source/CUDA mounts, exact CUDA targets, `auditwheel repair`,
  wheel-contract/license inspection, and CUDA cubin verification. The final
  checkpoint artifacts and SHA-256 values are recorded in
  `reports/validate/python-native-providers-manylinux-2026-07-22.json`.
  Both architectures produced default and cu13 wheels; AArch64 carries only
  `sm_121a`, while x86-64 carries exactly `sm_75`, `sm_80`, `sm_86`, `sm_89`,
  `sm_90`, `sm_100`, and `sm_120a`, with no PTX.
- Installed every repaired wheel into a fresh Python 3.12 environment. The
  final default artifacts completed the public C VITS synthesis smoke on CPU;
  the final cu13 artifacts loaded the pinned NVIDIA packages and completed the
  same smoke on physical GB10 and RTX 4070 SUPER CUDA. Provider preparation did
  not alter process paths. The source and wheel tests also caught and repaired
  an abbreviated cu13 transcribe.cpp notice before accepting the final hashes.
- The official PyPI `nvidia-cublas==13.6.0.2` distribution currently declares
  `nvidia-cuda-nvrtc` transitively. Provider native objects neither link nor
  initialize NVRTC, but the installed package is now explicitly inventoried
  instead of being incorrectly described as absent.
- The x86-64 checkpoint exposed a separate publication gate: `GGML_NATIVE=OFF`
  still leaves SSE4.2, AVX, AVX2, F16C, FMA, and BMI2 enabled in the single CPU
  library. These x86 wheels are valid on the tested RTX host but remain withheld
  until runtime-selected fat CPU variants or an explicit narrower ISA floor is
  implemented and validated.

## Next

The remaining CUDA hotspots and possible implicit-GEMM, SGEMM-selection,
transpose-convolution, elementwise-fusion, and get-rows investigations are
recorded in `_performance-research.md`. They are informational backlog with an
explicitly deferred go/no-go decision, not committed implementation work.

Numerical acceptance policy remains deliberately deferred. The next publication
work is the thin CPython 3.11+ `abi3` API Adapter and the x86 runtime-selected
CPU-variant policy; the R610 driver branch, RTX 4070 SUPER Windows toolchain,
macOS Metal wheel, and Vulkan Provider also remain unexercised. VITS/CUDA stays
Experimental until the applicable release gates are complete.

## 2026-07-23 — F16/Q8_MIXED validation and Hugging Face publication

- Ran the existing quantizer tests before producing artifacts. Two independent
  complete quantizer runs are byte-identical for both profiles.
- F16 keeps 342 text, duration, normalization, bias, and scalar tensors in F32
  and stores 118 flow/decoder weights in F16. The 70,453,376-byte package has
  SHA-256
  `ac14fd237532c75e485649e965009ddbfedc20ddbc69c157329abc11a2940d31`.
- Q8_MIXED keeps those 342 tensors in F32, stores four native-layout
  transpose-convolution weights in F16, and stores 114 ordinary flow/decoder
  matrices in Q8_0. The 52,890,048-byte package has SHA-256
  `ac026b3731ac627c44043bfd542933a0d277c6a1ac72f9e473b9334208308302`.
- Both packages pass public synthesis and ASan/UBSan synthesis on CPU. All seven
  12-case stages pass on one-thread CPU, DGX Spark CUDA 13.3, and RTX 4070 SUPER
  CUDA 13.3. Duration structure is exact everywhere. Every inspected CUDA graph
  has one split and zero executable CPU fallback nodes.
- Added model-card tests before generalizing the VCTK-only template. The
  generator now distinguishes preset catalogs from an unnamed fixed
  package-default Voice, rejects contradictory usage metadata, and omits
  `--voice` for LJSpeech.
- Published the flat F32/F16/Q8_MIXED payload and model card to
  `jiangzhuo9357/vits-ljspeech-gguf`. Remote LFS sizes and SHA-256 values and the
  byte-identical README were verified at revision
  `95a4bacf90c24a6d368b86e3edd8b1a9ba76195e`.
- Quality evaluation and numerical tolerance acceptance remain deferred. This
  release claims functional port validation only.

## 2026-07-23 — Built-in phoneme frontend and package refresh

- Added focused tests before implementation for strict UTF-8 decoding, Unicode
  scalar lookup, duplicate-symbol `last_index_wins`, blank insertion, final
  sequence limits, unsupported raw text, malformed GGUF metadata, request
  routing, C API capability reporting, CLI input, and Python wheel synthesis.
- Added a private `TextFrontend` seam and the built-in
  `synthesize.symbol_map` contract version 1. VITS packages now advertise
  `PHONEMES_UTF8 | TOKEN_IDS`; raw-text G2P remains unsupported and no GPL
  frontend is linked or invoked.
- Regenerated F32, F16, and Q8_MIXED twice each. Every pair is byte-identical.
  The refreshed packages are 113,245,056, 70,453,568, and 52,890,240 bytes with
  SHA-256 values
  `bd17e44c7c2d761d33c1527059bd3921f9d3fa9d46bd73b3e020d746a8b7db7b`,
  `5fc428ba97416cc164055f509af1b9e120b089bd0bab792ad674c862411d7bca`,
  and
  `df95091f975e78088c4908c3f2adfc381cfa234c3f520ddd3ffe10160ff12ba1`.
- Compared every one of the 460 tensor names, types, shapes, and payload hashes
  with the previous published F32 package. There are zero tensor-payload
  differences; the 192-byte increase is frontend metadata only.
- Public phoneme input `ˈeɪ.` produces byte-identical PCM to its resolved token
  sequence for all three profiles. All 42 one-thread CPU and 42 DGX CUDA
  stage/profile checks pass with exact duration structure; DGX placement has one
  split and zero executable CPU fallback nodes.
- The refreshed gates pass Release 62/62, locked Python 97/97, ASan/UBSan unit
  44/44 plus four F16/Q8 real-model paths, and DGX CUDA 58/58. The RTX 4070
  SUPER host has now been revalidated, and the refreshed package was
  republished to `jiangzhuo9357/vits-ljspeech-gguf` at revision
  `53273eea758782f86ffd79a830ffb4ccde533acf`.

## 2026-07-23 — VoiceProfileAPI and LJSpeech integration gates

- Added boundary tests for the public VoiceProfile C API so undersized
  size-tagged structures are rejected before the implementation is entered.
- Added a Python lifecycle regression that keeps a voice profile usable after
  its parent model closes until the profile itself is explicitly closed.
- Rebuilt the DGX Spark unit gate. A missing executable registration for
  `synthesize-abi-initializer-bounds-test` was corrected by building the target
  directly; the full unit gate then passed 43/43.
- Configured a separate integration build with the local LJSpeech and VCTK
  GGUFs plus the materialized golden payloads. The integration gate passed
  18/18, covering the LJSpeech public lifecycle, public synthesis, CLI smoke,
  all seven LJSpeech golden stage checks, the optional VCTK public path, and
  all seven VCTK golden checks.

## 2026-07-23 — Unit gate dependency fix

- Added `synthesize-abi-initializer-bounds-test` and `synthesize-cli` to the
  `synthesize-check-unit` dependency set so the unit gate no longer relies on
  prebuilt artifacts.
- Re-ran `cmake --build build/dev-dgx-spark --target synthesize-check-unit`;
  the clean rebuild now compiles the missing targets itself and passes 43/43.
