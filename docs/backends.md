# Execution Backend Policy

Status: Confirmed, last updated on 2026-08-20.

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
| OmniVoice | none, as of Plan 5 Task 1 (see the exception below); previously the whole generator | 2,421.04 of 3042.2 MiB (84.24 codec + 2,336.80 generator) | 122.61 s → 5.481 s, −95.5 % (719-frame case) | 0.1906× real time (faster than real time) |

Kokoro is the more expensive of the two because its LSTMs are unrolled in the
graph, so the duration stage is 26,916 nodes and all of it moves. VITS is cheaper
because its discrete path is one graph and the HiFi-GAN decoder, which dominates,
never leaves the primary backend.

The mechanism generalizes, and the second family needed no rework: identify the
weight groups the discrete path reads, mirror them, give any group a second view
when a continuous stage reads the same tensors from the primary buffer, and switch
that stage's scheduler. OmniVoice (Plan 4 Task 11) is the case this section used to
describe only as a prediction -- "an autoregressive [decision loop] where every
sampled token feeds the next step, the held portion would be most of the model" --
and the table row above, as it stood through Plan 4, is what that measured to. That
row also ran the mechanism **inverted** from Kokoro and VITS, which is worth stating
plainly rather than letting the shared column headers imply a shared direction: for
Kokoro and VITS, primary is the accelerator and the discrete stage's weights get a
second, CPU-resident copy so the hold is enforceable without a five-times-slower
mixed scheduler. For OmniVoice, primary is CPU -- Plan 4 Task 11 gave only the
codec (152 tensors, 84.24 of the model's 3042.2 MiB) a second, accelerator-resident
copy, so the codec's own decode could leave the CPU while the generator's own RVQ
token selection, drawn once per step and fed back into the next one, stayed held:
twenty golden cases at `--accelerate` moved every one of 8,440 codec nodes off the
CPU and left every one of 880,032 generator nodes on it, with all seventeen greedy
cases' token grids byte-exact against the CPU baseline. "Cost" read as a saving in
that configuration, not a tax: moving the free 1.6 percent of wall time (the codec,
on the suite's longest case) off the CPU was a small net win precisely because the
held majority dominated -- the reverse of Kokoro and VITS's shape, where holding a
minority off the accelerator was the expensive part.

**This is now superseded.** Plan 5 Task 1 (2026-08-08) gave the generator its own
accelerator-resident twin (312 tensors, 2,336.80 of the model's 3042.2 MiB) under
the exception below, so `--accelerate` now moves the generator too: the table row
above reflects both twins and the honest end-to-end RTF this unlocks, and the
inverted-direction framing two paragraphs up no longer describes a held stage for
this family at all -- see the exception for what replaced it. Full per-case
figures, both the Plan 4 codec-only measurement and the Plan 5 whole-generator one:
`docs/porting/families/omnivoice.md`'s Execution Backends section.

## The Discrete-Outputs Rule Admits One Narrow Exception

Added 2026-08-08 (OmniVoice, Plan 5 Task 1), after jiangzhuo revised this
family's own bar from token identity to audible quality. The rule above is
**not repealed** and remains the default for every family, including future
ones: a discrete output is held on CPU, along with everything feeding it,
unless a family earns this specific, measured exception.

**The exception's condition:** a family's discrete decision may run on the
primary Execution Backend, instead of being held on CPU by the rule above,
only when both (1) the discrete output determines CONTENT within a canvas
whose SHAPE was fixed by an earlier, non-discrete stage before the first
forward that produces the discrete output ever runs, and (2) that shape
invariance has actually been measured, matching, across the family's whole
Golden suite -- not inferred from the architecture alone. A discrete output
that can itself change the shape of anything downstream fails clause (1)
regardless of how small or well-argued the rest of the case looks, and
stays held under the rule above.

**Kokoro is the counter-example this condition is written to exclude, and
still would be if measured today.** Its discrete output is a rounded frame
count: TF32's roughly 1e-3 relative error moved one token's duration from 1
to 2 on the family's own 146-token case, taking the total frame count from
376 to 377 and giving every downstream tensor a shape no threshold can
compare against the reference's. That is clause (1) failing outright -- the
discrete output IS the shape decision -- and no amount of measurement adds
a size invariance that does not exist. Eighteen of Kokoro's twenty-one CUDA
validation runs failed exactly this way. VITS's duration predictor is the
same shape, at smaller scale, for the same reason.

**OmniVoice's generator earns the exception because its discrete output
never touches downstream shape.** The canvas length is fixed by
`RuleDurationEstimator` -- ported as deterministic host arithmetic, not a
GGML forward at all -- before the generator's first step ever runs. Every
one of the generator's 32 mask-predict steps then commits a token INDEX
into an already-fixed-size canvas; TF32 can change which codebook entry
wins the per-step argmax, never how many frames the canvas holds. Clause
(2)'s measurement: Plan 5 Task 1/2 ran the full twenty-case Golden suite
with the generator's own accelerator twin bound, and grid SIZE matched the
CPU oracle's in all seventeen greedy cases, 17 of 17 -- the codec's own
cloning-path token grids (unaffected by this exception; see below) stayed
byte-exact too, 2 of 2. Content is a different story: only 3 of those 17
cases matched the oracle's grid byte-for-byte, and the rest ranged from
0.5% of positions flipped (`omni-rate-fast`, 1 of 200) to 98.3%
(`omni-digits`, 1038 of 1056) --
this family's generator does not survive the rule's own knife-edge
argument any better than Kokoro's duration predictor did. What is different
is that OmniVoice's content drift is not a structural failure: it is a
different, still-valid answer within a canvas whose size never moved.

**Clause (2)'s size invariance was re-measured under a much larger
perturbation, and held.** Added 2026-08-08. Halving the generator's denoising
step count from 32 to 16 -- an intervention far coarser than TF32, changing the
commit schedule itself and re-drawing 95.61% of the suite's committed token
positions -- left `grid.i32`'s byte size identical between arms, and left
`pcm_freerun.f32`'s sample count identical between arms, equal to the oracle's,
and equal to frames x 960, in **17 of 17 greedy cases**. Clause (1) is what
makes that possible: the canvas length is host arithmetic that takes no step
count and runs before the first forward. That step count was itself rejected on
listening grounds and is not shipped, but as evidence about clause (2) it is
the strongest data point this family has -- the size invariance is a property
of where the shape decision lives, not of how small the numeric perturbation
happens to be.

**"A different, still-valid answer" undersells the cost: the drift can change
WHO IS SPEAKING.** Added 2026-08-08, and this section should not be read
without it. OmniVoice's auto-voice mode supplies no speaker conditioning at all
-- no speaker embedding table exists in the family -- so the speaker is
emergent from which token grid the decode lands on, and a large enough content
drift can land on a different one. Sweeping median F0 over voiced frames for
all fourteen CPU-vs-CUDA generator pairs, thirteen sit within +/-3% and the
speaker survives. **`omni-short-en` does not: 118.2 Hz on the shipped CPU path
against 189.0 Hz on the shipped CUDA path** (normalised cross-correlation;
140.9 -> 199.0 Hz on an independent YIN tracker), with the fraction of voiced
frames below 165 Hz going 0.98 -> 0.00 -- complete separation on both trackers,
a male voice and a female voice for the same request. Its CPU arm is
byte-identical to the oracle, so this is a change the placement move introduced
against the reference, on a shipped backend. `omni-short-en` was not one of the
six pairs the audit below sampled, so it was **heard separately, and the
measurement was confirmed**: a two-pair blind audit on 2026-08-08 (order seed
2026080817, `omni-short-en` as pair 1 with CPU in slot A) returned **"different
people," with the two arms' audio quality judged indistinguishable**. Pair 2
was `omni-design-zh`, the sweep's one borderline case, which the trackers
declined to count and the listener also called the same person -- so the proxy
was confirmed against a human ear on a positive case and a negative one. The
effect is therefore identity, not degradation. It is recorded here at its true
cost rather than folded into "still-valid answer", and it is the reason this
family's model documentation no longer promises that the auto-voice speaker
follows the synthesis seed.

**Why content drift this large is an acceptable answer, and why that is
weaker evidence than the size measurement above.** jiangzhuo's own listening
verdict settled this, not a tolerance number: a blind A/B of six pairs
(generator-on-CPU vs. generator-on-CUDA, decoded through the identical
codec) reported no problem heard in any pair, including the pair built from
the 98.3%-flipped case above, whose two waveforms measure cosine 0.0515 --
essentially unrelated audio, both judged acceptable speech. This is one
listener, six pairs, on one day -- a Listening Audit in this project's own
vocabulary, explicitly not a statistical claim, and it is cited as exactly
that: the reason a human accepted this family's specific content drift, not
proof that content drift is inaudible in general or that a future family's
drift would pass the same way. **The sample had a known gap, since closed**:
the six pairs did not include `omni-short-en`, the one case where the same
drift moves the speaker, so this verdict covers the pairs heard and not that
case -- the two-pair audit above heard it separately and returned "different
people, quality indistinguishable," which is why the speaker change is stated
as an identity effect rather than left as an unheard measurement.
There is precedent for shipping a family
whose discrete decisions are not reproduced exactly: qwen3-tts ships
Q8_MIXED while stating plainly that "in normal operation it will select
different codes sometimes."

**This is a per-family judgment call, not a blanket loosening of the rule
above.** A future family whose discrete output gates downstream shape --
another duration predictor, another rounded frame count -- is still held on
CPU by the original rule, full stop, no matter how it is argued. A future
family whose discrete output is shape-safe by clause (1) still owes clause
(2)'s measurement across its own Golden suite before it can move, and even
then the acceptability of its own content drift is a separate question this
project answers by listening, case by case, not by inheriting OmniVoice's
verdict. What generalizes is the two-clause test above, not its outcome.

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

**That placement is superseded and must not be quoted as current.** Added
2026-08-12. The measurement above was taken on 2026-07-22, and on 2026-07-26
`df1351e` moved this family's text encoder and duration predictor onto CPU under
"Discrete Outputs Are Held On CPU" above — so the 893 CUDA duration-stage nodes
it counts are the very stage that no longer runs there, and the family table's
VITS row is the current statement. The counts remain accurate for the
configuration and date they name; what expired is "every executable node ... on
CUDA" as a description of what ships. This paragraph exists because four
published model cards asserted zero executable CPU fallback on the strength of
the sentence above, three of them for families that were never measured that way
at all.

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

### qwen3-tts-12hz-0-6b-base carries no CUDA sub-grid, on a measurement

Recorded 2026-08-17 by Stage 2 Plan 4. The Base variant's two new graphs -- the
ECAPA speaker encoder and the codec encoder -- **stay on the CPU**, and that is
a measured decision rather than an omission: a device twin would mirror 241 MB
for about 3.7 s of extra load, against the two graphs' entire CPU cost of about
1.69 s on an 8.08-second reference. The transfer is 2.2x the compute it would
accelerate, so no speedup makes it pay. They also run once per Voice Profile
rather than once per synthesis, so the amortization that justified Stage 1's
codec-decoder twin runs the other way here.

Consequently only ONE of this variant's three tolerance stages measures anything
different on CUDA. `public` does, through the Stage 1 codec-decoder twin, which
does move on a Base package. `replay` does not -- its probes are the x-vector,
whose driver calls `Model::load_cpu` and takes no backend argument, and the
host-side ICL prompt assembly. `codec_encoder` does not, by the decision above.

Since gate 2 above is per-stage and `tests/python/test_tolerance_coverage.py`
requires a `backends` sub-grid to carry the variant's full stage set, filling one
would mean recording CPU figures under a CUDA key for two of three stages. **The
sub-grid is therefore not committed**, which the coverage rule permits, and the
one real measurement is recorded in
`docs/porting/families/qwen3-tts.md` ("The Base variant carries no CUDA sub-grid,
and one real CUDA measurement") instead: 7 public-seam checks pass and 3 are
structurally skipped, on `rel-dgx-spark`, with the CUDA audio differing from the
CPU audio as the codec decoder's TF32 arithmetic predicts.

This is a placement-and-agreement result only. Gate 5's latency, real-time factor
and peak memory are not claimed for it.

### qwen3-tts-12hz-1-7b-voicedesign carries no CUDA sub-grid either, on the same measured shape

Recorded 2026-08-20 by Stage 3 Plan 3 Task 6. Unlike Base, this variant
introduces no new graphs at all -- no reference-audio path, so no speaker
encoder and no codec encoder for ICL enrollment. It is Stage 1's own graph
set (text frontend, talker, codec decoder) at the 1.7B talker's width, so
there was no placement decision to make here, only a measurement of the one
Stage 1 already made.

This variant's tolerance grid tracks two stages, and only one of them can
move under CUDA. `public` does, through the same Stage 1 codec-decoder twin
Base's own `public` cell exercises. `replay` does not: its only probe is
`prefill`, and `tests/qwen3_tts_voicedesign_prefill_real.cpp:289` calls
`ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)` directly, so today's
driver has no backend argument to pass -- but the deeper reason a backend
argument would not help is `src/arch/qwen3-tts/model.cpp:573-585`: the
device-mirroring loop only copies tensors whose name starts with
`codec.decoder.`, so the talker's own weights, which the prefill graph runs
entirely on, are never mirrored onto the device at all.

Since `tests/python/test_tolerance_coverage.py` requires a `backends`
sub-grid to carry a variant's full stage set, and `replay` cannot move, a
`backends.CUDA` sub-grid here would face the identical choice the Base
subsection above declined: fabricate a `replay` CUDA cell out of `replay`'s
own CPU numbers, or leave the sub-grid uncommitted. **It is left
uncommitted**, and the one real measurement is recorded in
`docs/porting/families/qwen3-tts.md` ("Stage 3: VoiceDesign Package, Plan 3
Task 6") instead: on the family's own fixed VoiceDesign measurement workload
(text, description, language, seed and thread count reused verbatim from
Plan 3 Task 5), BF16 synthesizes in 18.92 s on CPU against 17.76 s on CUDA
(RTF 4.64 -> 4.35), a **6.16%** end-to-end gain -- the quantity is the
reduction against the CPU baseline, `(CPU - CUDA) / CPU`, not the ratio of
the two: computed from the unrounded medians, `(18.9220 - 17.7564) /
18.9220` and `(4.6377 - 4.3520) / 4.6377` both give 6.16%, while dividing
CPU by CUDA gives 1.0656, a 6.56% throughput increase over the same pair.
(The display-rounded pairs shown here reduce to 6.13%/6.25% by the same
`(CPU - CUDA) / CPU`, reconciled in the family record's Plan 3 Task 6
section.) This is smaller than Base's own
~6.5%/6.3% (the "CUDA buys about 6 % and that is the expected amount"
paragraph in `docs/porting/families/qwen3-tts.md`; 6.47%/6.35% unrounded,
the same CPU-baseline reduction), which is the
direction Stage 3's design spec predicted for a talker whose per-layer
parameter count runs 3.2x larger than Base's (D7,
`docs/superpowers/specs/2026-08-18-qwen3-tts-stage-3-design.md:227-229`),
held on the CPU by the discrete-outputs rule the same way. 11 of 11
applicable `public`-seam checks pass under CUDA -- one of the eleven (the
empty-instruct-vs-oracle relation) passes through a figure the `replay`
stage's own CPU-only prefill probe already measured rather than recomputing
anything under CUDA, the same way it does for the CPU cell -- and the CUDA
waveform differs from the CPU one in bytes (cosine 0.9999988, max_abs
0.002414 against the same seed and workload), which is the codec decoder's
TF32 arithmetic showing up exactly as gate 2's own tolerance model predicts,
on a different package and case than Base's own comparison and not claimed
to reproduce it exactly.

No CUDA threshold was derived for this variant either. Unlike Base, there is
also no hypothetical future graph within it to pre-derive one for -- its
graph set is exhausted by the codec decoder and the prefill probe -- so the
measured cosine and max_abs figures above stand as agreement evidence only,
not as an input to any committed gate.
