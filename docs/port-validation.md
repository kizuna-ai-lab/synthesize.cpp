# Port Validation Contract

Status: Confirmed, last updated on 2026-07-30.

## Purpose and Boundary

The Port Validation Suite proves that one converted Model Variant runs the intended
inference graph, honors its public controls and Voice mapping, produces stable
finite PCM, and executes on every claimed backend. It is deliberately small and
is not a speech-quality dataset, a listening test, or a model-ranking benchmark.

Each variant owns one committed Golden Manifest conforming to
[`synthesize-golden-manifest-v1.schema.json`](schemas/synthesize-golden-manifest-v1.schema.json).
Large checkpoints, tensors, stochastic inputs, and PCM remain generated artifacts.
The manifest records their logical paths; generation and validation reports record
their SHA-256 values.

## Golden Manifest v1

The top-level fields have these responsibilities:

| Field | Contract |
| --- | --- |
| `schema` | Exact value `synthesize-golden-manifest-v1`. |
| `suite_version` | Monotonically increasing variant-suite revision. |
| `family`, `variant` | Stable project keys used by converters, tests, and reports. |
| `source` | Immutable source repository revision plus hashed config, checkpoint, source, frontend, and license artifacts. |
| `reference` | Pinned oracle implementation, runner, environment lock, dtype, and device. |
| `package_contract` | Input, audio, language, Voice, randomness, speaking-rate, and hard input/output safety limits being validated. `frontend` is explicitly `null` when the package claims resolved token IDs only. |
| `tolerance_file` | Version-controlled numerical tolerances selected by stage and probe. |
| `case_artifact_root` | Ignored build-tree root for generated oracle material. |
| `cases` | Twelve to thirty-two deterministic Port Validation Cases. |
| `relations` | Phase-qualified cross-case assertions such as seed-dependent change or speaking-rate frame ordering. |

JSON stores every `uint64_t` seed as a decimal string. This avoids precision loss
in JavaScript, Python bindings that pass through JSON numbers, and other manifest
consumers. A loader rejects values above `UINT64_MAX`, duplicate case IDs, unknown
relation targets, a minimum speaking rate greater than its maximum, and duplicate
source-artifact roles where the family contract permits only one.

## Case Execution

Every case carries its origin, coverage labels, resolved runtime input, Voice,
public request values, family-specific oracle parameters, required checks, and
expected generated artifacts. Upstream source text remains in the case for
provenance even when core graph parity uses oracle-resolved token IDs. Source text
alone does not establish runtime text-input support.

Each successful case produces at least:

- resolved input and request metadata as JSON;
- the family-defined intermediate probe tensors as little-endian scalar files;
- native mono or interleaved PCM as little-endian F32; and
- the structured operation result.

Discrete structural results such as token count, duration expansion, attention
path shape, channel count, sample rate, and output frame count must match exactly.
Floating-point probes and PCM use the stage-specific tolerance file. NaN, infinity,
missing probes, unexpected fallback, and undeclared CPU placement are always hard
failures regardless of numerical tolerance.

## Choosing the Oracle's dtype and Device

The oracle runs at the dtype the checkpoint stores, on the least-approximating
device that reproduces run to run. Until 2026-07-28 this section did not exist
and the phase list simply said "on CPU", which was never a decision -- it was
generalised from the only two families then ported.

`vits` ships 460 F32 tensors and `kokoro` 548, so for both of them CPU F32 *was*
the reference dtype and the question never arose. `qwen3-tts` is the first family
where it does: its talker checkpoint is 402 BF16 tensors. Running that oracle at
float32 upcasts every talker weight and produces a reference for a model that
does not exist, which is what the first qwen3-tts oracle dump did before this
rule was written down.

The rule, in order:

1. **dtype follows the checkpoint.** Stage 3's artifact is the
   "source/reference-dtype GGUF", so the oracle and that GGUF must agree. A
   family may be mixed: qwen3-tts is BF16 for the talker and F32 for the codec.
2. **Device follows reproducibility, then the model's own target.** A golden
   dump that cannot be regenerated is a weak artifact, so verify it: run one case
   twice under the same seed and require identical bytes. CPU is preferred when
   it satisfies (1) because it pins no hardware. Where the checkpoint's dtype is
   one the model targets on an accelerator, using that accelerator is correct
   rather than a compromise -- qwen3-tts's oracle is CUDA BF16, which regenerates
   byte-identically and runs at about real time where CPU F32 ran at 9.4 times
   slower.
3. **Record what was chosen and why**, in the manifest's `reference` block and
   the intake record. The schema has always permitted `bfloat16`/`float16` and
   `cuda`/`metal`/`vulkan`; only this document's prose said CPU.

**What does *not* change:** the C++ side's correctness path is still established
on CPU, and the oracle and the source GGUF must still agree on the dtype they
*store*. That is rule 1, and it is what "like with like" means here.

**What the arithmetic may differ in, and under what conditions.** ggml computes a
stored BF16 weight through F32 and rejects BF16 in elementwise operators, so for a
BF16-checkpoint family a port computing F32 against a reference at BF16 is the
only reachable comparison. Until 2026-07-30 this paragraph forbade exactly that,
which made it a rule no BF16 family could follow: rule 1 pins the oracle to the
checkpoint's dtype, phase 2 puts the replay on CPU, and those two together force
the comparison the sentence prohibited. It is permitted, on three conditions:

1. the comparison is named in the tolerance file's `reference_stage`, so no reader
   has to infer it;
2. probes whose deep layers carry outlier channels gate on cosine, not max-abs;
3. structural exactness is pinned by construction -- the replay seam supplies the
   oracle's codes -- and never by a tolerance.

The third condition is what the old wording was really reaching for. It feared a
port "failing" for being more accurate than its reference, and that cannot happen
while frame counts and shapes come from the replayed codes rather than from a
threshold.

An unequal-arithmetic comparison does make each threshold cover a dtype difference
as well as an implementation one. That is a reason to **measure the two components
separately and record the split**, not a reason to forbid the comparison. The
split is measurable without the port: the talker probes are captured on the
prefill, which consumes the whole prompt and draws nothing, so the same oracle run
at two dtypes can be compared directly. **No family has done this yet** -- it is
owed by qwen3-tts, whose thresholds currently state the combined figure.

One consequence to expect rather than discover. A lower-precision reference has a
heavier tail: under BF16, seed 0 on the two-character input "Hi." produced 121
codec frames where every other seed tried gave 12 or 13, and F32 never did it.
That is the shipped model's behaviour, so a faithful port reproduces it, but a
manifest case exists to cover something specific and should not be left sitting
on an outlier that defeats its own coverage.

## Stochastic Replay and Public Seeds

Stochastic model parity uses captured model inputs rather than relying on two
frameworks to implement the same pseudorandom generator. The oracle emits each
named random tensor, and an internal validation-only family seam replays those
exact tensors in the reference and C++ graphs. The seam is unavailable to the CLI,
public C Interface, Rust Adapter, Python Adapter, and ordinary model execution.

The public seed contract is tested separately. Cases marked
`request_repeatability` run twice through the ordinary public request path on the
same validated configuration and require identical structural results and PCM.
Cross-case `artifact_differs` relations prove that selected distinct seeds actually
affect a stochastic model. This separation identifies graph errors without forcing
synthesize.cpp to reproduce PyTorch's generator implementation.

## Validation Phases

1. The pinned oracle resolves text to token IDs, captures stochastic tensors, and
   writes reference probes and PCM in stable case order, at the **reference
   dtype** on a device that reproduces run to run. See "Choosing the oracle's
   dtype and device".
2. The source/reference-dtype GGUF replays the same resolved token IDs and random
   tensors on CPU and passes tensor, structure, and waveform comparison.
3. Public-request runs validate seed reporting, same-seed repeatability, controls,
   Voice selection, cleanup, and operation results without random-tensor
   injection.
4. F16 and every Quantization Profile rerun the suite against the accepted CPU
   reference under their declared tolerance stage.
5. Every claimed Execution Backend reruns the same cases, proves real placement,
   records operational measurements, and passes repeated-run cleanup.

Tolerance values are measured from the first working reference and implementation,
reviewed, and then committed before support is declared. A port must not weaken a
tolerance merely to accept its own failing output.
