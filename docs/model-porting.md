# Model Porting and Publication Workflow

Status: Confirmed, last updated on 2026-07-24.

This workflow follows transcribe.cpp's artifact discipline and publication layout.
TTS uses a compact Port Validation Suite to establish model support first; its
corpus-scale Quality Evaluation Suite is a separate, deferred evidence layer.

## Core Rule

```text
Commit golden contracts, not golden payloads.
```

A committed contract identifies the source, reference implementation, cases,
tolerances, and commands needed to reproduce validation. Large checkpoints,
reference tensors, synthesized waveforms, GGUFs, and downloaded Model Packages are
generated or restored into ignored work directories and caches.

## Porting Stages

Every new Model Family follows the same named sequence:

```text
1-intake -> 2-oracle -> 3-convert -> 4-cpp ->
5-port-validate -> 6-quants -> 7-backends -> 8-ship
```

1. **Intake** pins model identity, source artifacts, license, architecture,
   capabilities, Text Frontend, voices, languages, and reference implementation.
2. **Oracle** runs every manifest case in the pinned reference implementation and
   captures intermediate tensors, resolved metadata, and PCM output.
3. **Convert** produces only the source/reference-dtype GGUF and its converter
   report.
4. **C++** implements `src/arch/<family>/`, establishes tensor-by-tensor and
   waveform parity, and completes the CPU correctness path.
5. **Port Validate** runs the variant's 12 to 32 deterministic cases, beginning
   with exact upstream examples, and establishes source/reference-dtype CPU tensor
   and waveform parity plus end-to-end stability.
6. **Quants** produces F16 and each selected Quantization Profile and reruns the
   Port Validation Suite against the accepted CPU reference path. Passing this
   stage makes no perceptual-quality claim.
7. **Backends** runs the same cases on each claimed Execution Backend and records
   actual placement, latency, real-time factor, and peak memory as operational
   evidence rather than comparative quality rankings.
8. **Ship** prepares the family documentation, per-variant documentation, HF YAML,
   generated HF README, and flat GGUF publication directory with its Validation
   Level and explicit quality-evaluation status.

An accepted change at stages 5 through 7 reruns the affected Port Validation Suite
path. Passing it is necessary and sufficient for `port_validated` publication, but
it never establishes perceptual quality or a ranking against another model.

Every implementation slice inside these stages must pass the unit, sanitizer,
and applicable Golden integration gates in [`testing.md`](testing.md) before the
next slice begins. A manual smoke run is evidence during development, not a
replacement for a registered test.

## Stable Keys and Source Layout

Each family has one stable lowercase key, such as `vits`, used consistently in
paths, manifests, CTest names, report names, tolerance files, and tool environments.
Each Model Variant has one stable publication slug.

The target layout mirrors transcribe.cpp:

```text
src/arch/<family>/
scripts/envs/<family>/pyproject.toml
scripts/envs/<family>/uv.lock
scripts/convert-<family>.py
scripts/dump_reference_<family>_<framework>.py
scripts/validate-<family>-<stage>.py
scripts/bench/
scripts/hf_cards/generate.py
scripts/hf_cards/template.md.j2
scripts/hf_cards/<variant>.yaml
docs/port-validation.md
docs/schemas/synthesize-golden-manifest-v1.schema.json
docs/porting/families/<family>.md
docs/models/<variant>.md
tests/golden/<family>/<variant>.manifest.json
tests/tolerances/<family>.json
tests/fixtures/
tests/<family>_smoke.cpp
tests/<family>_real_smoke.cpp
tests/<family>_e2e_smoke.cpp
```

Family-specific converter and reference-dumper scripts remain readable top-to-bottom
programs. Shared source resolution, GGUF metadata, hashing, naming, dtype encoding,
and Quantization Profile policy belong in `scripts/lib/`; model architecture and
tensor mappings remain in the family script. There is no generic converter base
class or automatic arbitrary-model dispatch.

## Committed Contracts

The repository commits:

```text
tests/golden/<family>/<variant>.manifest.json
tests/tolerances/<family>.json
docs/porting/families/<family>.md
docs/models/<variant>.md
scripts/hf_cards/<variant>.yaml
reports/porting/<family>/<variant>/intake.json
reports/porting/<family>/<variant>/_porting-log.md
```

The Golden Manifest has schema `synthesize-golden-manifest-v1`, defined by
`docs/schemas/synthesize-golden-manifest-v1.schema.json` and
`docs/port-validation.md`. It pins the Model
Family and Model Variant, source artifact locators and revisions, source file
SHA-256 values, reference implementation and revision, reference dtype, Text
Frontend contract, native audio format, Voice and Language capabilities, tolerance
file, and validation cases. A case identifies its Linguistic Input, language,
Voice selection or conditioning, concrete random seed, and expected reference
artifacts. Cross-case relations capture behavior such as seed-dependent change and
speaking-rate frame ordering. Source and reference revisions are mandatory for new
ports; a mutable branch name is insufficient.

A manifest normally contains 12 to 32 cases. It begins with any usable examples
authored by the upstream model repository, preserving their exact inputs, Voice
selection, and synthesis controls, then adds only the cases needed to exercise
the remaining inference branches. Port Validation Suite inputs are committed
directly and do not require a training or evaluation corpus.

The Golden Manifest is validation provenance and is never rewritten by an HF upload.
The intake packet records the research that justified it. The family note records
architecture and porting decisions; the per-variant page records user-facing
downloads, Validation Level, Port Validation Suite evidence, quality-evaluation
status, operational measurements, and reproduction commands.

## Generated Payloads and Reports

Reference and synthesize.cpp payloads use transcribe.cpp's build-tree convention:

```text
build/goldens/<family>/<case>/
build/validate/<family>/<variant>/<case>/ref/
build/validate/<family>/<variant>/<case>/cpp/
models/<variant>/<variant>-<PROFILE>.gguf
reports/convert/<family>/<variant>-<dtype>.json
reports/validate/<family>/<case>.json
reports/reference/<family>/<case>.json
reports/bench/<machine>/<family>.<backend>.json
```

These numerical payloads and generated reports are ignored by Git. A converter
report records source locator and revision, source file hashes, converter command
and synthesize.cpp revision, output GGUF path and hash, family and variant identity,
dtype policy, tensor count, and every skipped, tied, or fused tensor.

Reference and C++ dump directories contain identically named little-endian F32
tensor payloads plus behavioral artifacts. TTS behavioral artifacts include native
PCM, resolved seed and Voice metadata, output format, frame count, and structured
operation result. For stochastic families, the reference runner also emits named
random input tensors that an internal validation-only seam replays in both graphs;
the ordinary public request path separately validates seed reporting and
repeatability. Per-stage `scripts/validate-<family>-<stage>.py` validators, as in
the seven `validate-vits-*.py` scripts, drive reference generation, C++
generation, tensor comparison, and deterministic audio regression from the same
manifest.

## Conversion and Quantization Split

The Python converter reads the upstream framework, applies layout transforms,
embeds authoritative metadata, and writes the source/reference-dtype GGUF. It has
no quantization option. The C++ `synthesize-quantize` tool consumes that GGUF and
writes F16 or another named Quantization Profile. This follows transcribe.cpp and
llama.cpp's converter-then-quantizer split.

The first GGUF is an accuracy artifact. It preserves reference dtype, tensor names,
layouts, Text Frontend metadata, Voice metadata, audio configuration, and inference
hyperparameters. Quantization begins only after that artifact passes the CPU
tensor-and-waveform parity gate.

## Deferred Quality Evaluation

The Quality Evaluation Suite is not implemented and is not a prerequisite for a
`port_validated` Published Model Package. Until it is commissioned, every Model
Page and generated HF README reports `quality_evaluation: not_run` and makes no
claim about perceptual quality, quantization transparency, or ranking against
another Model Variant. This deferred tooling does not block adding model families.

When activated, TTS corpus quality will retain transcribe.cpp's evidence flow while
remaining multidimensional rather than producing one universal TTS quality score.
Evaluator identities, versions, model revisions, normalization rules, metric
formulas, and thresholds must be locked before the first package declares the
`quality_evaluated` Validation Level. Thresholds are measured from real outputs
rather than guessed in the architecture.

For a Model Variant entering that later evaluation, the reference-dtype/F16
package and representative candidate Quantization Profile outputs run through the
full quality corpus before
numerical drift thresholds are accepted. The thresholds belong to that Model
Variant and Quality Evaluation Suite version, live in its source-controlled quality
manifest, and are frozen before the first non-reference-dtype package declares
`quality_evaluated`. They are not transferred across Model Variants or model
families.

The F16 Quality Baseline must independently pass hard correctness plus the pinned
upstream tensor-and-waveform parity gates; a candidate cannot make a broken baseline
acceptable. Once thresholds are frozen, candidate results cannot be used to loosen
them. Any threshold change creates a new Quality Evaluation Suite version, reruns
the F16 baseline, and prevents direct comparison with results from the old version.
Every candidate Quantization Profile is then evaluated against the matching Quality
Baseline under the frozen rules.

The suite has these independent axes:

1. **Hard correctness and stability** rejects NaN or infinite samples, empty or
   malformed output, incorrect sample rate or channel count, excessive clipping,
   excessive DC offset, unintended long silence, duration-limit violations, and
   failure to reproduce a deterministic case from its recorded seed. These are
   per-case gates rather than aggregate metrics.
2. **Intelligibility** transcribes synthesized cases through a pinned transcribe.cpp
   batch runner and the F16 file from `handy-computer/whisper-large-v3-gguf`. It
   reports WER for English and supported European languages and CER for Mandarin
   Chinese, Japanese, and Korean. The ASR repository revision, GGUF SHA-256,
   transcribe.cpp revision, decoding parameters, resampling recipe, and text
   normalizer versions are part of the report provenance. A language-specific
   validated evaluator may replace the default only when the manifest declares
   that choice explicitly.
3. **Naturalness** uses a pinned UTMOSv2 evaluator as a secondary regression signal,
   not as a standalone release gate. Acoustic diagnostics such as duration,
   clipping, silence, energy, and applicable pitch statistics remain separately
   visible. A candidate cannot compensate for a hard failure or intelligibility
   regression with a higher predicted naturalness score.
4. **Voice similarity** uses a pinned speaker-embedding evaluator and reference
   enrollment set when Voice identity applies. Results are grouped by Voice and
   conditioning mode; variants without an evidence-backed Voice reference report
   the axis as not applicable instead of inventing a score.
5. **Listening Audit** is optional, non-statistical release evidence. One maintainer
   may inspect a small automatically selected A/B set for obvious regressions; this
   is never reported as MOS, CMOS, a listening panel, or population-level evidence.

The automated axes are gates only for the `quality_evaluated` Validation Level;
they are not gates for `port_validated` publication. A quality-evaluated Model
Package may omit a Listening Audit, but its Model Page and generated HF README must
report exactly one of `no_obvious_regression`, `regression`, or `not_run`. The first
reference-dtype/F16 package for a Model Variant is compared with the pinned upstream
implementation. A candidate Quantization Profile is compared with the matching F16
Quality Baseline. Inputs, Voice conditioning, seed, and synthesis parameters are
held constant wherever the two implementations support equivalent controls.

The audit harness selects at most six A/B pairs: the worst intelligibility
regression, worst UTMOSv2 regression, worst applicable Voice-similarity regression,
longest-duration case, and two deterministic random cases from the remainder.
Duplicate selections reduce the set instead of adding more work. A/B order is
deterministically randomized and the report records the comparison identities,
case hashes, selection reason, and order seed.

For a language the maintainer understands, the audit may flag obvious content,
pronunciation, naturalness, Voice, truncation, noise, or timing problems. For other
languages it assesses only language-independent acoustic defects and records
language naturalness as `not_assessed`. `no_obvious_regression` means only that the
small audited set exposed no obvious problem. A `regression` result requires an
investigation or an explicit known-limitation note; it is not erased by a better
automated score. Community or laboratory listening studies may be attached later
as additional evidence without changing the runtime library or public C interface.

The default intelligibility recipe is:

```text
ASR repository: handy-computer/whisper-large-v3-gguf
ASR file:       whisper-large-v3-F16.gguf
task:           transcribe
language:       explicit language from the quality case
timestamps:     none
decode:         greedy plus transcribe.cpp's pinned default fallback recipe
previous text:  not conditioned
en:             EnglishTextNormalizer + WER
zh-CN/ja/ko:    BasicTextNormalizer + CER
other:          BasicTextNormalizer + WER
```

The harness uses transcribe.cpp's batch-report path and stores the complete recipe
with every result. It never uses automatic language detection for a scored case.
The executable revision, HF revision, F16 file hash, normalizer package versions,
and any deterministic resampling implementation are immutable within one Quality
Evaluation Suite version. Paths to the executable and ASR GGUF are supplied to the
quality tool; neither artifact is bundled into a synthesize.cpp Model Package.

The full large-v3 F16 evaluator is chosen over large-v3-turbo because this is an
offline release gate where evaluator accuracy has priority over evaluation speed.
Changing the default evaluator starts a new suite version and a new Quality
Baseline run; old and new intelligibility results are not compared directly.

The default naturalness recipe is:

```text
package:               UTMOSv2 v1.3.0
source commit:         14a281c6072d6fad2bf31f992ed2e60304400075
configuration:         fusion_stage3
folds:                 0, 1, 2, 3, 4
checkpoint pattern:    fold<FOLD>_s42_best_model.pth
repetitions per fold:  5
random seed:           42
data-loader workers:   0
input:                 in-memory PCM plus its sample rate
prediction domain:     sarulab
remove silent section: false
result:                mean of all 25 predictions per case
```

The harness downloads all five checkpoints from the official
`sarulab-speech/UTMOSv2` HF repository at one immutable revision, verifies every
SHA-256, and passes local checkpoint paths to UTMOSv2. It does not allow the
package's default download from moving `main`. The UTMOSv2 tag, source commit,
Python lockfile, checkpoint revision and hashes, execution device, and deterministic
framework settings are report provenance. Cases run in stable manifest order with
NumPy and PyTorch random state initialized from seed 42.

Silence removal is disabled so the evaluator cannot hide pauses or truncation
produced by the candidate. The raw predicted MOS is stored per case and summarized
by Model Variant, language, Voice, conditioning mode, and Quantization Profile. It
is compared only with the matching F16 Quality Baseline group; there is no absolute
or cross-language UTMOS threshold. A higher UTMOSv2 score cannot compensate for a
hard-correctness, intelligibility, or Voice-similarity failure.

UTMOSv2 is a secondary publication-tooling Adapter. Its package and official HF
model are MIT-licensed, but its checkpoints are cached only for evaluation and are
not redistributed with synthesize.cpp or a Model Package. Replacing UTMOSv2 or any
checkpoint starts a new Quality Evaluation Suite version and baseline run.

The default Voice-similarity recipe is:

```text
package:       SpeechBrain 1.1.0
model:         speechbrain/spkrec-ecapa-voxceleb
architecture:  ECAPA-TDNN
model license: Apache-2.0
input:         deterministic 16 kHz mono PCM
embedding:     L2-normalized speaker embedding
similarity:    cosine similarity
references:    at most 5 clips per Voice
```

A Preset Voice uses up to five manifest-pinned natural reference clips. A Voice
Profile uses the actual Voice Reference clips from which it was prepared, capped at
five in stable manifest order. Each reference embedding is L2-normalized; their
mean is normalized again to produce the evaluation-only Voice centroid. The
candidate embedding is normalized and compared with that centroid to produce
`target_similarity`.

When a Model Variant has two or more reference-backed Voices, the harness also
compares the candidate against every applicable non-target centroid and reports:

```text
target_margin = target_similarity - max(non_target_similarity)
```

This margin detects Voice-ID mapping errors and speaker leakage that a target score
alone can miss. A single-Voice variant reports only `target_similarity`. Same-
language and cross-language Voice Reference cases are separate report groups and
are never aggregated into one threshold.

The SpeechBrain package lock, HF model revision and file hashes, audio preprocessing
recipe, execution device, and reference-audio hashes are report provenance. The
published report contains reference count, total reference duration, grouped
similarities and margins, but never raw Voice Reference audio or speaker
embeddings. Evaluation embeddings are ephemeral and are not a Voice Profile,
Serialized Profile, Model Package resource, or public runtime representation.

Voice-similarity gates are relative to the matching F16 Quality Baseline Voice and
language group. Dataset-specific speaker-verification decision thresholds are not
reused as universal TTS thresholds. If a Voice has no legally usable and
reproducible natural reference, this axis is `not_applicable`; the harness never
manufactures a centroid from candidate output. Replacing SpeechBrain, ECAPA-TDNN,
or its checkpoint starts a new Quality Evaluation Suite version and baseline run.

Reference-dependent waveform metrics, including ViSQOL, are diagnostics only for
cases with a meaningful aligned reference waveform. Because valid TTS output is
one-to-many in timing and prosody, these metrics are not universal release gates.
New or experimental learned metrics may be reported, but do not become a gate until
their evaluator, robustness, license, and threshold have been explicitly locked.

The quality harness writes structured per-case results plus corpus distributions:
failure count, mean, median, fifth percentile, and ninety-fifth percentile where
the metric supports aggregation. It also reports results by language, Voice,
conditioning mode, and Quantization Profile. Release decisions consume these axes
independently and do not collapse them into a weighted aggregate.

When implemented, the quality harness and its Python evaluator environments will
live under `scripts/quality/`. They are publication tooling rather than runtime
dependencies and do not add functions or evaluator concepts to the public C
interface. Their model
artifacts remain outside synthesize.cpp Model Packages and are pinned only as
validation provenance.

Evaluator background and limitations are tracked from their primary sources:
[UTMOSv2](https://github.com/sarulab-speech/UTMOSv2),
[ViSQOL](https://github.com/google/visqol), the published
[speaker-similarity evaluation study](https://arxiv.org/abs/2207.00344), and the
[ECAPA-TDNN model card](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb).
The [documented adversarial failure modes](https://arxiv.org/abs/2606.31105) of
UTMOS-style predictors are one reason the suite does not use predicted MOS as a
sole release gate.

One YAML file under `scripts/hf_cards/` is the source for each variant's HF README.
The generator includes pinned upstream provenance, Validation Level, validation
revision and date, download table, quality status and any available results, usage,
license, and the pinned upstream model card.
The rendered README and GGUFs are staged together under `models/<variant>/` and
uploaded using the HF layout defined in `model-packages.md`.

As in transcribe.cpp, no separate HF package index or release-record bundle is
created. The validation sentence in the source-controlled model page and generated
HF README points to the synthesize.cpp revision that can regenerate the evidence.

## Test Tiers

Default unit and synthetic smoke tests require no downloaded model. Real-model and
end-to-end tests are opt-in and receive a local GGUF path through a family-specific
test environment variable. CI explicitly downloads and caches only selected small
canary GGUFs from `handy-computer`; absence of an HF token skips the private-canary
tier without weakening mandatory model-free tests.

Publication is different from ordinary CI: every Published Model Package must run
its complete Port Validation Suite from the pinned upstream checkpoint through the
produced GGUF. A load-only or audible-output smoke test is never sufficient.

CPU establishes correctness first. CUDA, Metal, Vulkan, and future Execution
Backends reuse the same manifest cases and must pass the backend validation gates
before their results appear in a model card.
