# Qwen3-TTS Family Selection and Port Plan

Status: Confirmed 2026-08-20. Stage 1 (`qwen3-tts-12hz-0.6b-customvoice`) is
complete and published. Intake, the oracle and conversion are done; stages 4
through 7 have their measured work done: oracle replay and the public seam
pass, and the codec runs on CUDA while the autoregressive half stays on the CPU
under the discrete-outputs rule. **BF16, F16 and Q8_MIXED are all built and
measured** (Q8_MIXED on 2026-07-29, see "Q8_MIXED, and the refusal that was
wrong"); **the public backend control reaches the split** (`synth_model_load`
with `SYNTH_BACKEND_CUDA`, see "It is reachable from the public seam"); **port
validation passed** at Validation Level `port_validated`, 18 Golden cases over
three stages, dated 2026-07-29 in `scripts/hf_cards/`; and **stage 8 published
the package on 2026-07-28** at
[`jiangzhuo9357/qwen3-tts-12hz-0-6b-customvoice-gguf`](https://huggingface.co/jiangzhuo9357/qwen3-tts-12hz-0-6b-customvoice-gguf),
last updated there 2026-07-29 carrying all three profiles. Selection was
accepted on 2026-07-26; the intake packet is
`reports/porting/qwen3-tts/qwen3-tts-12hz-0-6b-customvoice/`. **Stage 2
(`qwen3-tts-12hz-0.6b-base`) has Plans 1, 2 and 3 done** -- the package, the
x-vector clone path, and the transcript-assisted (ICL) clone path, all on CPU;
**Plan 4 executed 2026-08-17 and is complete, all sixteen tasks.**
It measured both Quantization Profiles (F16 clears every gate and does NOT pay
-- it is 184,448 bytes *larger* than its source; Q8_MIXED pays on both, 33.7 %
smaller at RTF 0.863 against BF16's 3.15, faster than real time), settled the
quantizer's blocker on the Base package's 237 new tensors, removed the semantic
gate's cliff, and **declined** a CUDA twin for the two new graphs on a
measurement -- the transfer costs 2.2× the compute it would accelerate. **The first ICL
Listening Audit ran on 2026-08-17 and recorded `no_obvious_regression`** across
five blind pairs and two labelled resemblance checks, meeting `spec:532`'s audit
gate; both clones were judged the same speaker as their source, and the two
quantization profiles were audible but not degraded.
See "Stage 2 Plan 4: what it measured, and what it refused to claim" below.
**The Base variant was published on 2026-08-17** to
`jiangzhuo9357/qwen3-tts-12hz-0-6b-base-gguf` (BF16, F16 and Q8_MIXED, 6.70 GB,
commit `d4df99e8`), on jiangzhuo's per-act confirmation naming that target --
which closes Stage 2 on the same terms Stage 1 closed on. **Stage 3
(`qwen3-tts-12hz-1.7b-voicedesign`) Plan 1 is done as of 2026-08-18**: Task 6
converts the checkpoint (659 tensors, 4.30 GB BF16) and, since a same-day fix
to the tensor catalog described below, loads it through `synth_model_load` --
and Task 7 closes the plan's completion gate: a prefill built at empty
instruct matches the oracle within tolerance (p95_relative 0.00269 against a
committed 0.01, 3.72x headroom), with a 365.7x fault-injection figure proving
the comparison can fail. See "Stage 3: VoiceDesign Package, Task 6" and
"...Task 7" below for the license check, the digests, a genuine architecture
gap Task 6 found and closed rather than merely converting around, and Task
7's oracle dumper, driver and fault injection. See the three Stage 2
paragraphs below for Stage 2.

**The capability snapshot Task 6 measured is corrected below, on jiangzhuo's
ruling after the final whole-branch review (2026-08-18), and no longer matches
the design's original prediction.** Task 6 measured exactly what the design
predicted -- zero Preset Voices, `SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT |
SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE`, Reference Audio absent, all six
reference limits at zero -- but `synth_voice_profile_create_from_description`
(`src/voice-profile.cpp`) still routed every family but OmniVoice to the
generic unsupported fallback at that point, so the RUNTIME's own capability
query was advertising a source it would then refuse: the same trap this
family's own transcript-assisted (ICL) mode was withheld from advertisement
for the whole of Stage 2 Plan 2 to avoid. Ruling: the RUNTIME withholds
`SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT`, and `SERIALIZED_PROFILE` does not
survive alone either (this package has no speaker encoder for
`load_qwen3_tts_profile_from_memory` to ever accept a Profile against), so the
capability snapshot as of this correction reports **zero Preset Voices and
zero Profile source flags** -- the same all-zero shape a CustomVoice package
reports, for a different reason. Only the RUNTIME half moved: the PACKAGE's
own declared `synthesize.voice.profile_sources` still names
`description-text`, and Task 3's loader and its cross-checks are untouched.
See "Stage 3: VoiceDesign Package, Task 6", "Load result" below for the
corrected measurement table.

**Second correction, 2026-08-18, Stage 3 Plan 2's Task 5: the withholding above
is no longer current.** It was a true account of a real interval, so it stands
rather than being rewritten, but a reader stopping at the paragraph above today
would be misled about what the runtime currently publishes. Both reasons for
the withholding are closed: Plan 2's Task 2 gave
`synth_voice_profile_create_from_description` a Qwen3-TTS arm, so the seam no
longer routes every request to the generic unsupported fallback, and Task 3
made `load_profile_from_memory` route on a design envelope's own declared kind
before the x-vector size check that used to refuse every serialized Profile
against this package. With both gaps closed, Task 5 republished the capability:
the VoiceDesign package now reports **zero Preset Voices and
`SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT | SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE`
with Reference Audio absent** -- exactly what Task 6 originally measured and
the design originally predicted, before the interval this section records.
See "Stage 3: VoiceDesign Package, Task 6", "Load result" below for Task 5's
own re-measurement against the real package (`0xa`, `qwen3-tts-voice-design`
version 1) -- the all-zero table the first correction above points to is
itself superseded there, in turn.

**Packages converted before `synthesize.voice.profile_sources` existed do not
load under this runtime any more.** Stage 3's loader change
(`src/arch/qwen3-tts/weights.cpp`) made a `profile-sources`-mode package
declare which Voice Profile sources it implements, rather than the loader
inferring "Base, therefore reference audio" from the mode alone -- inference
the mode stopped supporting once VoiceDesign started using the same mode for
Description Text. The three packages published above on 2026-08-17 (BF16, F16,
Q8_MIXED, commit `d4df99e8`) predate that key and are refused by
`synth_model_load`, so `tests/qwen3_tts_base_load_real.cpp` and the Base Golden
targets (`synthesize-qwen3-tts-base-xvector-golden`,
`synthesize-qwen3-tts-base-mel-shape`) will fail against them until the local
copy is re-converted. This is deliberate, not an oversight: the project is
pre-release with no users, so no compatibility shim was written to infer the
declaration from a package's other contents -- that is exactly the inference
this loader change exists to remove. Re-cut a local Base package with the
current `scripts/convert-qwen3-tts.py` before turning on
`-DSYNTH_BUILD_INTEGRATION_TESTS=ON` against it. **Done 2026-08-18**, as part of
Stage 3 Task 6: the local BF16 package at
`models/qwen3-tts-12hz-0-6b-base/qwen3-tts-12hz-0-6b-base-BF16.gguf` was
re-converted from the same source weights and manifest already on disk (no
re-download), producing a different sha256 (`76275beb...` against the
`993f4cd1...` the pre-`profile_sources` file carried) and a package that loads
with the capability snapshot Plan 2 recorded --
`SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO | SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE`,
six populated reference limits, schema `qwen3-tts-voice-clone`. The already-published Hugging Face
artifacts were NOT re-cut or re-uploaded -- their three digests are unchanged --
but the CARD beside them was, on 2026-08-18 (HF commit `9802704d`), on
jiangzhuo's per-act confirmation scoping it to the README alone. It now says
first thing in its publication note that those three files no longer load and
why, so the live artifact stops claiming a loadability it does not have. That
correction also carried a second one: the CUDA end-to-end figure read "about
8 %" against numbers that divide out to 6.5 %.

**Until 2026-08-12 this line read "Q8_MIXED, the public backend control and
stage 8 are not done. Port validation is not started."** All four clauses were
false, and each was contradicted by a later section of this same document —
the parenthetical cross-references above are those sections. The Hugging Face
API is the artifact that settles publication; the card specification and Open
Question 6 agree with it.
Stage 2 (`qwen3-tts-12hz-0.6b-base`) Plan 1 is done: the Base package is
pinned, converted (894 tensors), loads through `synth_model_load`, and its
capability snapshot -- zero Preset Voices and, in the Plan 1 state, zero Voice
Profile sources (because the runtime could not yet prepare or consume a Profile
for this family) -- is reported correctly and covered by an integration test
against the real package (see "Stage 2: Base Package, Plan 1" below). The
package's own Voice Profile contract is carried and validated at load time
regardless.

**Stage 2 Plan 2 is done: reference audio in, cloned audio out, on CPU, in
x-vector mode.** The mel front end and the ECAPA-TDNN speaker encoder graph
exist and run (473 nodes as the real BF16 package builds it, 435 with F32
weights -- see "ECAPA-TDNN graph and mel front end" below for what separates
them); `synth_voice_profile_create_from_reference` now prepares a real
Profile from the real package (measured against the oracle at cosine
0.99999536, a residual attributable to the oracle's own bfloat16 weights,
not the port); that Profile serializes, reloads, and substitutes into the
synthesis prompt in place of a Preset Voice; and
`tests/qwen3_tts_clone_real.cpp` proves, against the real package, that two
different reference inputs produce different cloned audio -- the assertion
that actually distinguishes "cloned" from merely "synthesized", though the
second input is a synthesized tone rather than a second recording, so it
says nothing about resemblance -- while a Serialized Profile round-trips to
the same audio and CustomVoice stays byte-identical throughout. The
capability snapshot now advertises
`SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO | SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE`
for Base. See "Stage 2: Base Package, Plan 2" below for the full record,
including what remains: ICL / transcript-assisted cloning (Plan 3),
Description Text (Stage 3), the CLI (a cross-family slice, deliberately not
this plan's), and quantization and CUDA for the new graphs (Plan 4).

**The Stage 2 Plan 2 Listening Audit ran on 2026-08-13 and recorded
`no_obvious_regression`.** One listener, four blind port-vs-oracle pairs over
three languages in x-vector mode; a labelled duration sweep that found the
shipped reference-duration bounds usable at 1 s, 3 s, 10 s and 30 s, with the
sub-minimum 0.5 s case refused by the library as designed; and a labelled
source-versus-clone pair the listener judged **the same speaker** -- the first
resemblance evidence in this repository, and one listener's judgement on one
source clip and one clone rather than a property of the port. The audit covers
neither ICL -- which this port did not implement when the audit ran, and does
implement since Plan 3 -- nor CUDA for the new graphs, which have only ever run
on CPU. Quality Evaluation stays `not_run` per ADR 0017 and the Validation
Level does not move. See "Listening Audits" below.

**Stage 2 Plan 3 is done: reference audio AND its transcript in, cloned audio
out, on CPU, in transcript-assisted (ICL) mode.** The codec encoder's graph
exists and runs, reproducing upstream's own float32 run on **100.000% of code
decisions** with the continuous chain at 9.303e-05 against a 1.0e-3 gate; the
two-track ICL prompt and its `min(T1, T2)` alignment are built and compared
against the oracle per track, both arms covered; a Serialized Profile carries a
second `kind` (`"icl"`) inside the unchanged v1 schema; `reference_transcript`
and `reference_language` moved to `SYNTH_REQUIREMENT_OPTIONAL` in the same
change that landed the synthesis path; and `tests/qwen3_tts_icl_real.cpp`
drives it end to end against the real package. **The plain-equality gate on
reference codes was dropped by ruling on 2026-08-13** -- upstream disagrees
with itself depending on a load-time `dtype` keyword -- and replaced with
stage-wise float artifacts at a bf16-derived tolerance. **A wrong two-track
alignment still returns `SYNTH_OK` with finite, non-silent audio**, so those
numerical gates are the whole defence; the frame count that betrays it is
observable on the public seam and nothing looks at it. (Whether that audio is
*fluent* is unknown — nobody listened to it, and this document says so wherever
the measurement is quoted.) The Validation Level does not move,
**no Listening Audit has run for ICL**, and no performance or quantization
number is claimed. See "Stage 2: Base Package, Plan 3" below, including what
remains: Description Text (Stage 3), the CLI, and quantization, CUDA and the
listening pass (Plan 4).

## Decision

The third Model Family is **Qwen3-TTS 12 Hz** (Qwen team, Alibaba). OmniVoice
(Xiaomi / k2-fsa) is recorded as the leading fourth-family candidate, deferred
for a licensing reason recorded below rather than a technical one.

## Reference Contract

The oracle is the pinned `QwenLM/Qwen3-TTS` PyTorch implementation at commit
`022e286b98fbec7e1e916cb940cdf532cd9f488e` (package version 0.1.1, not published
to PyPI, so the reference environment pins it by git revision), driving the
`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` weights at revision
`85e237c12c027371202489a0ec509ded67b5e4b5`. Both carry an explicit Apache-2.0
grant: the repository ships the Apache 2.0 text at the pinned commit, and the
model card declares `license: apache-2.0` in frontmatter — verified against the
card at the pinned revision, not only against `main`.

The checkpoint is 13 files and 2,498,388,392 bytes; per-file SHA-256 digests are
recorded in `intake.json` under `weights.files`. Upstream publishes no digest of
its own, so unlike Kokoro this provenance is self-measured rather than
cross-confirmed.

`Qwen3TTSModel.generate_custom_voice` is the pinned inference entry point, driven
at the checkpoint's bfloat16 on CUDA with `attn_implementation="eager"`. The first
intake drove it on CPU in F32, which upcast every talker weight and described a
model that does not exist. **It takes two independent
sampling switches.** `do_sample` governs the Talker and `subtalker_dosample` the
code predictor, the latter defaulting to `True`; a reproducible oracle requires
both set to `False`, and setting only one is silently non-deterministic.

Native output is 24,000 Hz mono F32 at a 12.5 Hz frame rate, so one codec frame
is exactly 1,920 samples. The model consumes raw text through a Qwen2 byte-level
BPE tokenizer whose vocabulary and merges ship in the variant repository.
`trust_remote_code` is not required, so `docs/scope.md`'s rule against executable
code in a Model Package is not engaged.

Nine preset speakers are selected by codec-vocabulary token id rather than by an
embedding table, which is why this variant carries no speaker encoder. Two of
them, `eric` and `dylan`, pin a dialect language token regardless of the
requested language.

## Why the Selection Criterion Changed

`docs/model-family-selection.md` selects families architecture-first. Kokoro was
additionally chosen for standing among CPU-practical open models. Applying that
same standing test in July 2026 does not produce a shippable answer.

Artificial Analysis Speech Arena lists 14 open-weight models out of 91. Only
four rank above the project's current Kokoro variant:

| Rank | Model | Elo | Weight license | Publishable as a Model Package |
| --- | --- | --- | --- | --- |
| 23 | Fish Audio S2 Pro | 1120 | Fish Audio Research License | No — commercial use requires a separate paid license |
| 24 | Step-Audio-EditX (3B) | 1113 | Apache-2.0 asserted | Doubtful — see below |
| 37 | Voxtral TTS (Mistral, 4B) | 1075 | CC BY-NC 4.0 | No |
| 46 | Magpie-Multilingual 357M (NVIDIA) | 1060 | NVIDIA Open Model License | Unverified; +3 Elo is inside noise |
| 47 | **Kokoro 82M v1.0** | **1057** | Apache-2.0 | Already published |

Every remaining open-weight entry — Maya1 (1046), Chatterbox (1013),
Zonos-v0.1 (1000), VibeVoice 7B (957), OpenVoice v2 (956), XTTS v2 (916),
StyleTTS 2 (890), Mist V2 (885) — ranks below the family already shipped.

Step-Audio-EditX is the only nominally Apache-2.0 model with a material lead,
and it fails two criteria at once. Its repository acknowledges that "part of the
code **and data** for this project comes from: CosyVoice", whose pretrained
weights carry no explicit license, and the maintainer's answer to a direct
question about this is non-responsive. It also requires 12 GB of GPU memory at
batch size 1, which cannot satisfy the mandatory CPU inference baseline in
`docs/scope.md`.

The selection criterion for this cycle is therefore **capability and language
coverage**, not arena standing. This is an explicit, recorded change, and it
carries an explicit consequence: the third family is not claimed to be better
sounding than the second. Its delivered Validation Level is `port_validated`.
Quality Evaluation remains deferred per ADR 0017.

## Why Qwen3-TTS

**License.** All four upstream repositories — `Qwen3-TTS-12Hz-0.6B-Base`,
`Qwen3-TTS-12Hz-0.6B-CustomVoice`, `Qwen3-TTS-12Hz-1.7B-VoiceDesign`, and the
shared `Qwen3-TTS-Tokenizer-12Hz` — carry `license: apache-2.0` in model-card
metadata with no additional restriction in prose. This was verified directly
against the raw model cards rather than taken from secondary summaries.

**Language coverage.** One family covers the entire project interest order in
`docs/languages.md`: English, Mandarin Chinese, the Core European Set
(fr, de, es, it, pt), Japanese, and Korean, plus Russian. Coverage is a claim
about the upstream model; each language becomes an advertised capability only
after that variant and language combination passes validation.

**Public interface debt.** `include/synthesize.h` declares four Voice Profile
sources and `src/voice-profile.cpp` currently fails all four with
`SYNTH_ERR_UNSUPPORTED_VOICE`. No shipped family advertises any
`SYNTH_PROFILE_SOURCE_*` flag. Qwen3-TTS is the first family whose upstream
variants map onto those flags directly rather than by analogy.

**Runtime surface.** The two shipped families are feed-forward. Qwen3-TTS
introduces KV-cached autoregressive decoding, a sampling chain, and a causal
codec decoder — the last of which is a candidate for Native Streaming Synthesis
rather than only Chunked Audio Delivery, subject to its own validation.

**Tractability.** Independent GGML ports already exist and are inspectable,
which answers criterion 4 (operator gaps measurable and tractable) with an
existence proof rather than an estimate. See Prior Art below.

## Architecture

Established from the upstream repositories and cross-read against existing GGML
ports. Items marked *(second-hand)* come from a port's documentation and must be
confirmed against upstream during intake.

```
Linguistic Input
  -> Qwen2 byte-level BPE text tokenizer
  -> Talker: Qwen3 decoder, 28 layers, GQA 16/8, head_dim 128, RoPE 1e6,
     per-head RMSNorm on Q and K, SwiGLU, KV cached
     hidden 1024 (0.6B) | 2048 (1.7B); codec_head -> 3072 logits
  -> Code Predictor: 5-layer Qwen3 MTP head, hidden 1024, own KV cache,
     expands one semantic code into 15 acoustic codes per frame
  -> Qwen3-TTS-Tokenizer-12Hz: RVQ 16 codebooks x 2048
     (1 semantic + 15 acoustic), Mimi-style SEANet + transformer,
     ConvNeXt upsample, DAC decoder with SnakeBeta   (second-hand)
  -> 24 kHz mono, 12.5 frames per second, hop 1920 samples
```

The Talker consumes two pad-aligned streams — a text stream and a codec stream —
summed into one embedding sequence. Upstream carries a multimodal RoPE with
three sections, but a synthesis timeline is single-axis, so the sections collapse
to plain 1D NEOX rope *(second-hand; confirm during intake)*.

Base variants additionally carry an ECAPA-TDNN speaker encoder producing a
fixed x-vector from Reference Audio. CustomVoice variants carry no speaker
encoder; their named speakers are precomputed codec-embedding rows. VoiceDesign
carries neither.

## Reference Model Variant Ladder

Upstream splits capability across separate checkpoints, so capability is staged
across three Reference Model Variants rather than compressed into one. Each
stage is independently shippable and each adds exactly one public capability.

**Stage 1 — `qwen3-tts-12hz-0.6b-customvoice`.**
Preset speakers only, resolved through the existing `voice_id` path and the
Preset Voice Catalog. Voice Profile sources stay unadvertised. This stage
isolates everything genuinely new in the runtime — BPE Text Frontend, KV-cached
autoregressive decoding, the sampling chain, the Code Predictor inner loop, and
the RVQ codec decoder — from all Voice Profile work. It is the first Reference
Model Variant and the one that makes Qwen3-TTS a Supported Model Family.

**Stage 2 — `qwen3-tts-12hz-0.6b-base`.**
Adds `SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO`. This is the stage that requires the
ECAPA-TDNN speaker encoder and the Audio Normalizer specified but not yet
implemented in `docs/voice-conditioning.md`, including vendored libsamplerate
0.2.2. Upstream supports both an x-vector-only mode and a transcript-assisted
in-context mode, which maps onto the optional transcript field already present
in the Reference Audio descriptor.

**Stage 3 — `qwen3-tts-12hz-1.7b-voicedesign`.**
Adds `SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT`. There is no 0.6B VoiceDesign
checkpoint upstream, so this stage necessarily moves to the 1.7B width and
inherits a larger CPU cost. Its natural-language schema stays Model
Variant-defined, as `docs/voice-conditioning.md` already requires.

Stage 1 is a completion gate for Stage 2, and Stage 2 for Stage 3. Random Seed
stays unadvertised across all three stages.

**What each stage advertises, stated once (corrected 2026-08-12, 2026-08-18).**
An earlier revision of this section said Serialized Profile stays unadvertised
at every stage, while the Stage 2 Plan 1 record below claimed the Base package
"honestly advertises Reference Audio and Serialized Profile support". Both
cannot be right, and neither described what shipped. The end position:

- **Plan 1 of Stage 2 advertised nothing.** `synth_model_get_voice_profile_capabilities`
  returned zero source flags for both variants of this family, and therefore
  zero in every field describing a source. The Base *package* carried a full
  Voice Profile contract (`synthesize.profile.*`, `synthesize.reference.*`),
  which the loader read and validated; the *runtime* could not prepare,
  consume or serialize a Profile for this family, because at that point
  `src/voice-profile.cpp` dispatched every source for OmniVoice alone.
  `docs/c-interface.md` decides which of those two facts the query reports:
  "A Model without runtime Voice Profile support reports zero flags." **Past
  tense as of Plan 2**, which is the next bullet: `voice-profile.cpp` now has
  a `ModelFamily::Qwen3Tts` arm on the create-from-reference, load-from-memory
  and serialize paths, and Base reports source flags 9.
- **Plan 2 advertises `SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO`** on the day it
  can actually prepare a Profile from a reference clip — and
  `SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE` with it, not as a separate
  decision: `docs/c-interface.md` requires that any Model which can create a
  v1 Profile also sets the Serialized Profile bit, because every successfully
  prepared v1 Profile can be serialized.
- **Stage 3 adds `SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT`** on the same terms
  as Plan 2 added Reference Audio -- on the day it can actually prepare a
  Profile from a description, not merely the day a package that could supply
  one loads. **Stage 3 Plan 1 is not that day.** Task 6 shipped this bullet's
  own rule violated -- the RUNTIME briefly advertised the bit while
  `create_from_description` still had no Qwen3-TTS arm -- and jiangzhuo's
  2026-08-18 ruling corrected it before Plan 1 closed: the VoiceDesign package
  reports zero source flags, same as CustomVoice, until the plan that wires
  the handler. See "Stage 3: VoiceDesign Package, Task 6" below for the
  corrected measurement.

So Serialized Profile is not a source this family pursues on its own; it
arrives as a consequence of being able to prepare one, and it is unadvertised
until then. The same is true of every OTHER source bit, Description Text
included, which the paragraph above already stated for Reference Audio and
Stage 3 Plan 1 is what made literally true of Description Text too.

## Port Validation Fit

An autoregressive sampler cannot be validated by end-to-end token or waveform
equality against a separate framework. A single argmax tie at floating-point
epsilon flips one code, every later frame conditions on it, and the two runs
walk different but equally valid trajectories. A published GGML port measures
per-stage forward cosine at or above 0.9994 while free-running code agreement
falls to 4.96 percent, which is the expected signature of this effect and not a
defect.

**This does not require a new validation contract.** `docs/port-validation.md`
already resolves exactly this problem generically:

> Stochastic model parity uses captured model inputs rather than relying on two
> frameworks to implement the same pseudorandom generator. The oracle emits each
> named random tensor, and an internal validation-only family seam replays those
> exact tensors in the reference and C++ graphs.

and Phase 2:

> The source/reference-dtype GGUF replays the same resolved token IDs and random
> tensors on CPU and passes tensor, structure, and waveform comparison. […] This
> separation identifies graph errors without forcing synthesize.cpp to reproduce
> PyTorch's generator implementation.

Applied to this family:

1. The oracle captures its sampled code sequence alongside its resolved token
   IDs, and the validation-only family seam replays that exact sequence into
   both graphs. The replayed artifact is a discrete `i32` code sequence rather
   than a float noise tensor. This is an application of the existing rule; it is
   recorded in the family contract and does not amend the Port Validation
   Contract or require a new ADR.
2. Per-stage probes and codebook-0 logits are compared under the tolerance file.
3. The codec decoder is deterministic given a fixed code sequence, so
   `codes -> PCM` keeps the ordinary end-to-end waveform comparison intact.
4. Phase 3 `request_repeatability` and `artifact_differs` cases validate the
   public seed contract without injection, as they already do for Kokoro.

## Text Frontend Consequence

Qwen3-TTS consumes raw text through a Qwen2 byte-level BPE tokenizer. A second
built-in Text Frontend Provider — vocabulary and merges read declaratively from
the Model Package, no executable mapping logic, no external process, no network —
would give the project **Input Level 1 raw text for the first time**, which
`docs/text-frontends.md` currently records as unimplemented. It reaches that
level without eSpeak NG and therefore without the GPL-3.0-or-later distribution
problem that keeps a G2P Provider optional and separately installed.

This is a genuine capability gain, and it is also a self-contained slice that
can be built and unit-tested before any inference code exists.

## Quantization Profile Shape

Selective quantization must not treat this family like a plain language model.
The constraint reported by an existing port, to be re-measured here rather than
assumed: RVQ codebooks and the projections wrapping them, plus the speaker
encoder, must stay at the reference dtype, because nearest-neighbour lookup is
sensitive to per-row quantization noise and even BF16 mantissa truncation drifts
codes enough to damage Voice fidelity. One-dimensional tensors stay F32.

This is the same shape as the Kokoro decision recorded in
`docs/porting/families/kokoro.md` — a family-measured reason for holding part of
the graph at the reference dtype — and it confirms that ADR 0006 generalizes.
Profiles are measured for this family, not inherited from either predecessor.

## GGML Operator Surface

Two operators beyond the vendored set were expected here: a Snake activation and
a 1-D column-to-image scatter-add for transposed convolution. **Both
expectations were wrong, and the incremental surface is zero.** This was
measured against `qwentts.cpp`, which maintains a ggml fork to supply both as
fused operators; this project needs neither, and therefore needs no fork.

| Expected operator | Measured status |
| --- | --- |
| 1-D column scatter-add | `ggml_col2im_1d` is upstream, with CPU, CUDA and Vulkan kernels. No Metal, which is outside the validation matrix anyway. VITS was moved onto it on 2026-07-27, so it is exercised rather than merely present. |
| SnakeBeta | `snake()` in `src/arch/kokoro/operations.cpp` already computes `x + sin²(αx)/α` from five ordinary operators. SnakeBeta is `x + sin²(αx)·(1/β)` with β independent — the divisor changes from α to a separate tensor. A parameter on the existing helper, not a new operator. |

A fused `GGML_OP_SNAKE` would be a performance optimization only. Do not add one
before a profile asks for it.

If a local change does become necessary, there is no longer a mechanism for
one: `ggml/` became a submodule on 2026-07-27 and a submodule cannot carry a
local modification. `ggml-patches/README.md` lists the options that remain --
avoid the operator, fork ggml under an organization this project controls, or
revert to vendoring -- and the first of those is what retired the last patch.
Given the outcome recorded in those notes, no upstream submission is planned or
implied by this design.

## Decomposition

This family is the largest single port the project has attempted. For scale:
`src/arch/vits` and `src/arch/kokoro` together are roughly 6,800 lines, while a
mature single-family GGML port of Qwen3-TTS is roughly 13,900 lines of
implementation excluding its vendored HTTP and JSON libraries, its command-line
tools, and GGML itself. Roughly two-to-one against two existing families is
enough to warrant decomposition and not enough to warrant alarm. This design
does not fit one implementation plan. It decomposes into independently plannable
sub-projects:

1. **Intake and reference environment** — pin upstream revisions and checkpoint
   hashes, lock a per-family uv environment, stand up the oracle runner, resolve
   every item in Open Questions, and write `docs/porting/families/qwen3-tts.md`.
2. **BPE Text Frontend Provider** — declarative vocabulary and merges, raw-text
   Input Level 1, independent of any family inference code.
3. **Converter and tensor catalog** — GGUF layout for the Talker, the Code
   Predictor, and the codec; conversion rules and their focused unit tests.
4. **Autoregressive runtime** — KV cache, the sampling chain, the Code Predictor
   inner loop, and the validation-only replay seam.
5. **RVQ codec decoder** — SEANet, encoder/decoder transformers, ConvNeXt
   upsample, DAC with SnakeBeta, and the causal frame-by-frame decode path.
6. **Stage 1 port validation** — Golden Manifest, cases, tolerances, CPU.
7. **Voice Profile and Audio Normalizer** — Reference Audio, ECAPA-TDNN,
   vendored libsamplerate; unblocks Stage 2.
8. **Quantization Profiles**, then **Execution Backends**, then publication.

Sub-projects 2 and 7 are not Qwen3-TTS-specific and pay down debt that any later
family would otherwise re-incur.

## Prior Art and Attribution

Three independent GGML ports were read during selection:
`ServeurpersoCom/qwentts.cpp`, `ServeurpersoCom/omnivoice.cpp`, and
`predict-woo/qwen3-tts.cpp`. All three are MIT.

Two consequences. First, they are legitimate cross-check oracles: a published
code round-trip tool and pre-converted GGUFs give the Port Validation Suite a
second reference alongside the PyTorch oracle. Second, MIT permits reuse but
requires preserving the copyright notice — any code adopted from them, as
opposed to understanding gained by reading them, must be recorded in
`THIRD_PARTY_NOTICES.md` at the time it is adopted, not retroactively.

An honest note on positioning. `qwentts.cpp` independently arrived at a
single-header C99 ABI with a bounded version range, a build-time C-consumability
test, hidden internal symbols, runtime-loadable multi-backend support, and a
codec-aware quantization policy. That is convergent with this project's
`SYNTH_ABI_VERSION`, `synthesize.map`, ADR 0010, and ADR 0014. This project's
distinction is not the individual port; it is one versioned ABI across multiple
validated families, with Port Validation Suites, Golden Manifests, Model
Packages, and recorded decisions. That distinction is real but narrower than it
looked before these repositories were read.

## Findings From Reading qwentts.cpp

Recorded 2026-07-26 from a source read of `ServeurpersoCom/qwentts.cpp`, before
any of this family's own code exists. Nothing here was copied; these are
constraints to verify during intake and conversion, not adopted implementation.
Anything later adopted as *code* triggers the `THIRD_PARTY_NOTICES.md`
obligation described above.

### Three conversion rules that fail silently

Each of these produces a converter that raises no error and output that is
wrong.

**The RVQ codebooks are not codebooks.** The checkpoint stores EMA
accumulators. The codebook has to be reconstructed at convert time:

```python
embedding = embedding_sum / clip(cluster_usage, 1e-5)[:, None]   # RVQ_EPS = 1e-5
```

`embed_sum` and `cluster_usage` are stored as pairs and must be matched by
`(origin, side, layer)`. Skip this and the nearest-neighbour lookup returns
garbage with no diagnostic. The reconstructed table must stay F32.

**Convolution kernels are forced to F16 at load**, independently of the GGUF
storage dtype, because ARM's im2col is strict about kernel dtype. This project's
primary development host is aarch64, so it is directly in the path rather than a
portability footnote.

**SnakeBeta's α and β pass through `exp()` on every forward** in the reference
implementation. They can be folded once on the CPU at load, leaving the graph to
multiply plain F32 buffers. Whether the folding is valid depends on the exact
forward, which is why Task 3 of the intake plan reads that source.

### Tensor topology the config does not reveal

The 15 acoustic codebooks each carry a **private embedding table and a private
linear head**. A converter tensor catalog derived from the configuration alone
will not have them.

### Quantization observations

`qwentts.cpp` keeps at source dtype: the RVQ codebooks and the projections
wrapping them, the speaker encoder's final fully-connected layer, every snake
α/β, and every 1-D tensor. Its stated reason for the codebooks is measured
rather than assumed — Q8_0 and K-quants corrupt the reference-audio encoding and
destroy voice cloning, and even BF16's mantissa loss is enough to make codes
drift.

That is the same shape as Kokoro's rule that the decoder's upstream stays at the
reference dtype, and it fits this project's existing Quantization Profile
framework without new mechanism. It is an input to profile design, not a
conclusion; this family's profiles are measured for this family.

### Validation: what transfers and what does not

**Transferable.** Their staged probe table is close to what this family's Golden
Manifest needs: `Embed`, `TrailingText`, `TTSPadEmbed`, `L0`, `L7`, `L14`,
`L21`, `L27`, `Final`, `Logits`, `NextEmbStep0`, `TalkerHiddenStep1`.

**Not transferable — their divergence percentages.** Their logs record the
PyTorch reference on CUDA against C++ on CPU, which compares two kernel stacks.
This project's oracle is CUDA bfloat16 -- see `docs/port-validation.md`,
"Choosing the Oracle's dtype and Device" -- so the same caution applies to our own
numbers and not only to theirs. Their
figures should not be carried into this family's expectations; measure ours.

**Not transferable — max-absolute tolerances on deep talker layers.** In one of
their runs the per-layer max-abs is 13.07 / 13.09 / 13.42 at L7 / L14 / L21 and
**66.8** at L27, while cosine similarity stays at or above 0.9998. Those are
outlier channels ahead of the final norm. `tests/tolerances/qwen3-tts.json` must
use a cosine threshold for these stages; a max-abs threshold would report
catastrophic failure on a correct port.

### Determinism: what "byte-exact sampling" there does not mean here

`src/philox.h` implements Philox4x32-10 and the header says it keeps the
multinomial sampler "byte for byte aligned with the upstream Python pipeline".
Read carefully before drawing the obvious conclusion.

Two things make it inapplicable as-is:

- Philox matches PyTorch's **CUDA** generator (cuRAND). This project's oracle
  runs on CUDA too, so that much would agree -- but the replay seam captures the
  codes rather than reproducing the stream, so no generator has to be matched.
- Their alignment is achieved by **replacing `torch.multinomial` in the
  reference** — `tests/cossim_common.py` defines `patched_multinomial`, which
  pulls the uniform draw from their own Philox stream and walks the F32
  cumulative sum the way `src/sampling.h` does. The oracle was changed to match
  the port, not the other way round.

That is a legitimate way to isolate the RNG, and it is the same *idea* as this
project's stochastic-replay seam: neutralize the generator so everything else
can be compared. The difference is where the neutralization happens. Theirs
modifies the reference implementation; ours captures the draws as model inputs
and replays them, leaving the oracle stock. Ours is the less invasive of the two
and stays valid when the reference changes.

**The conclusion to carry into stage 5: do not plan on end-to-end sampled parity
against an unmodified PyTorch oracle.** Neither port achieves it. Compare the
deterministic prefix, and replay the draws across the seam.

If a sampled path is ever compared, the operation order has to match
HuggingFace's `generate()` chain exactly, because each step changes the
distribution the next one sees:

```
repetition_penalty -> temperature -> top_k -> top_p -> softmax -> multinomial
```

Details in their implementation worth confirming against upstream rather than
assuming: `top_p` softmaxes the full vocabulary over the *sorted* tensor so the
cumulative-sum boundary matches `TopPLogitsWarper`; the Talker masks
`[vocab - 1024, vocab)` except `codec_eos` before sampling while the
CodePredictor needs no suppression; and the greedy path (`temperature <= 0`)
is argmax over the suppressed logits with no repetition penalty and no draw at
all.

### Native Streaming Synthesis looks reachable, and this is what it costs

Open question 5 asks whether the causal codec decoder qualifies as Native
Streaming Synthesis under `CONTEXT.md` or only Chunked Audio Delivery.
`qwentts.cpp` carries an existence proof for the stronger claim, which intake
should confirm against upstream rather than inherit.

They keep three pieces of decoder state resident between frames:

1. every causal convolution's left context,
2. every transposed convolution's overlap carry,
3. the transformer's sliding-window KV ring.

With those, their `pipeline-codec.h` states that a `T=1` frame decode
"reproduces the offline full decode exactly with zero re-decoded context". That
is incremental synthesis, not chunking.

The contrast is instructive. Their *other* path, `codec-chunked-decode.h`,
decodes a whole buffer in bounded-VRAM chunks and must prepend
`left_ctx_frames` of previously decoded frames and then strip the resulting
samples, because a chunk decoded in isolation has edge artefacts where the
causal kernel and the attention window have no left context. That path is a
memory-bounding device, and it is what this project would ship if the state
above were not carried — Chunked Audio Delivery, in `CONTEXT.md` terms.

So the distinction between the two claims for this family is concrete: it is
whether those three state objects are carried across frames. That is a design
decision for stage 4, not a property of the architecture, and it should be
recorded as such rather than discovered late.

### Text frontend: the pieces this project would need

Their byte-level BPE is 633 lines with no dependencies, reading vocabulary and
merges from the GGUF payload. The parts are: the GPT-2 byte-to-Unicode encoding
table, the GPT-2 regex pre-tokenizer, the BPE merge loop, and a registry of
verbatim special tokens loaded from caller-named GGUF keys — which is how the
TTS style markers and language tags are handled.

That matches the second built-in Text Frontend Provider sketched above, and
sizes it: a self-contained, unit-testable slice with no executable per-model
mapping logic. The one piece that is logic rather than data is the GPT-2
pre-tokenizer regex, which is fixed across models rather than per-package.

### Performance shape of the code predictor

Their fix resets the predictor KV cache each frame, prefills two positions
(`talker_hidden` and `embed(c0)`), then replays 14 single-token steps against a
graph built lazily at the first frame — uploading `N*4` bytes of code ids per
step. That takes the inner loop from `O(Σ(g+2)²)` to `O(16)`. Recorded as a
known-good shape to compare against, not as a design commitment.

## OmniVoice as Fourth-Family Candidate

OmniVoice (`k2-fsa/OmniVoice`) is technically the stronger fit on several axes:
a non-autoregressive mask-predict model whose refinement steps reabsorb
floating-point argmax flips, so a port can be checked for exact token agreement;
a single 0.6B checkpoint carrying voice cloning, voice design, and automatic
voice selection at once; and far broader claimed language coverage.

It is not the third family for one reason, verified verbatim from its official
model card:

> Our code is released under the Apache 2.0 License. The pre-trained model is
> licensed under the CC-BY-NC due to constraints from its training data
> (e.g., Emilia).

The repository `LICENSE` is Apache-2.0, but that governs the code. The model
card carries no `license:` field at all; the sentence above is the only license
statement for the weights. At least one existing GGML port states the upstream
model is Apache-2.0, which is incorrect — a reminder that license facts are
verified against the upstream model card, never against a downstream port.

The distinction that keeps OmniVoice alive is already in the project vocabulary:
a **Supported Model Family** requires a port that passes the Port Validation
Suite on CPU, while a **Published Model Package** additionally requires
releasing exact bytes and license through a project-owned repository.
CC-BY-NC weights can satisfy the first and cannot satisfy the second.

A fourth-family decision for OmniVoice must therefore resolve, before intake,
whether the project supports a family whose Reference Model Variant can be
converted by a user but never published by the project, and what that means for
a Port Validation Suite whose artifacts are already deliberately uncommitted.
That question is left open here rather than answered.

## Intake Measurements

Measured on 2026-07-27 from revision `85e237c1` on CPU, not read from a port or
a paper. These supersede the *(second-hand)* marks in the Architecture section.

### Talker

28 layers, hidden 1024, 16 attention heads over 8 KV heads, `head_dim` 128,
intermediate 3072, RMSNorm eps 1e-6, `rope_theta` 1e6, codec vocabulary 3072,
text vocabulary 151936 projected from `text_hidden_size` 2048. The code
predictor is 5 layers at the same width with vocabulary 2048 and
`num_code_groups` 16. `model.safetensors` holds 402 tensors and 905,788,672
parameters across `talker.model` (311), `talker.code_predictor` (86),
`talker.text_projection` (4) and `talker.codec_head` (1).

`trust_remote_code` is **not** required. The modelling code lives in the
installed `qwen_tts` package, not in the checkpoint, so `docs/scope.md`'s rule
against executable code in a Model Package is not engaged.

### The multimodal RoPE collapses exactly, and this is proven

Open question 2 asked whether a 1-D collapse is equivalent. **It is, exactly.**

`talker_config.rope_scaling` declares `mrope_section [24, 20, 20]` — summing to
64, which is `head_dim / 2` — with `interleaved: true`, so the machinery looks
real. It is never exercised. Every path that builds `position_ids` in
`modeling_qwen3_tts.py` produces three identical rows:
`cache_position.view(1,1,-1).expand(3, ...)`, `position_ids[None,...].expand(3,
...)`, `position_ids.unsqueeze(0).expand(3,-1,-1)`, and `get_rope_index`, whose
body is `attention_mask.cumsum(-1) - 1` followed by `.expand(3,-1,-1)`. That
method's docstring describes temporal/height/width video positions and is
inherited from Qwen2-VL; this model has no vision branch.

`apply_interleaved_rope` starts from `x[0].clone()` and overwrites strided
slices with rows 1 and 2. When the rows are equal each write stores the value it
replaces. Checked directly: with three identical rows the output is
`torch.equal` to plain 1-D RoPE; with three different rows it is not, differing
by 5.5 at these shapes. So the operator is genuine and the collapse is safe only
because of how the positions are built.

**Consequence for stage 4: implement ordinary RoPE.** No sectioning, no
interleaving. Record the reason, because the config will keep saying otherwise.

### The codec, measured against the tensors rather than its config

12.5 Hz frames at 24 kHz, `decode_upsample_rate` 1920 — which is exactly
24000/12.5, and decomposes as `upsampling_ratios [2, 2]` then `upsample_rates
[8, 5, 4, 3]`, four times four hundred and eighty. The two upsample blocks are
ConvNeXt: `pwconv1` widens 1024 to 4096 and `pwconv2` returns it, the usual
four-times expansion. `speech_tokenizer/model.safetensors` holds 496 tensors and
170,557,441 parameters.

The quantizer is 1 + 15, matching `num_code_groups` 16:
`decoder.quantizer.rvq_first` has one VQ layer and `rvq_rest` has fifteen, each
codebook `(2048, 256)`, wrapped by `input_proj` 512→256 and `output_proj`
256→512.

**`speech_tokenizer/config.json` disagrees with its own weights.** It declares
`codebook_dim: 512` and `semantic_codebook_size: 4096`; the tensors are 256-dim
and no codebook has 4096 entries anywhere in the file — the only 4096s are the
ConvNeXt pointwise widths. `codebook_size: 2048` is the one that matches. A
converter that sizes the codebooks from this config allocates the wrong tables.
**Size from the tensors.**

### The RVQ EMA rule, confirmed and with a trap the reference port does not hit

The codebooks really are EMA accumulators, as recorded under the findings above.
The checkpoint adds something that section does not mention: **the two sides use
different field names for the same pair.**

| Side | Fields | Count |
| --- | --- | --- |
| `decoder.quantizer.*.vq.layers.N._codebook` | `embedding_sum`, `cluster_usage` | 16 |
| `encoder.quantizer.*.layers.N.codebook` | `embed_sum`, `cluster_usage` | 32 |

A conversion rule keyed on `embedding_sum` silently skips every encoder
codebook, and one keyed on `embed_sum` skips every decoder codebook. Both
produce a converter that raises nothing. Encoder codebooks also carry an
`initialized` flag of shape `(1,)`, which is not a weight and must be dropped.

### Voices and languages: three different counts, all correct

`get_supported_speakers()` returns nine: `aiden`, `dylan`, `eric`, `ono_anna`,
`ryan`, `serena`, `sohee`, `uncle_fu`, `vivian`. They are not embeddings — each
is a **token id** in the codec vocabulary (2861 to 3066), which is why this
variant carries no speaker encoder. The tensor inventory confirms that: there
are no ECAPA-TDNN tensors in either file. The family plan's claim was right.

Two of the nine carry a dialect override in `spk_is_dialect`, which is not a
boolean but a language name: `eric` → `sichuan_dialect`, `dylan` →
`beijing_dialect`. Those two names appear in `codec_language_id` and nowhere in
the public language list, so a dialect is reachable only by selecting its
speaker, never by asking for it as a language.

That produces three counts which are easy to confuse and are all correct:

| Source | Count | Contents |
| --- | --- | --- |
| Model card frontmatter | 10 | the language codes |
| `get_supported_languages()` | 11 | those 10 plus `auto` |
| `codec_language_id` | 12 | those 10 plus the 2 dialects |

For the Preset Voice Catalog: nine Voice Profiles, two of which pin the language
token regardless of the request, and `auto` as a language meaning detect from
text.

### CPU oracle smoke, and what its speed says about the family

Greedy on CPU, F32, eager attention, `aiden` in `english`, text
"Qwen3-TTS is awesome!", on the aarch64 GB10 host with the GPU unused.

| | value |
| --- | --- |
| sample rate | 24000 |
| frames / duration | 97,920 / 4.08 s |
| wall time | 39.6 s and 38.4 s |
| **real-time factor** | **9.7 and 9.4** |
| finite, peak | yes, 0.574 |

**That number is about the family, not about the oracle.** Every sampled token
conditions the next, so `docs/backends.md`'s discrete-output rule holds the
autoregressive core on CPU on every Execution Backend — the CPU figure is
therefore **not** automatically the CUDA figure, and the earlier claim that it was
has been withdrawn. Two things make it a decision rather than an implication.

Upstream deploys this model on CUDA with bfloat16 and FlashAttention 2; that is
the only device guidance its model card gives. The figure above is CPU, F32 and
eager attention, which is what the first oracle dump used before the dtype
rule was written down. It is not a measurement of the model as its authors run
it, and it is no longer how this family's oracle runs.

Whether CUDA helps then depends on a policy question this family raises for the
first time. `docs/backends.md`'s discrete-output rule would hold the
autoregressive core on CPU on every backend, because each sampled token is a
discrete value conditioning the next. Applied as written, CUDA buys almost
nothing here -- Kokoro paid 95 % of synthesis time to hold two stages of seven
and VITS 29 % to hold one graph, while this family would hold the loop itself.

But the rule exists to keep *structural* results identical across backends, and
this family is stochastic by design. Port validation replays the captured codes,
so the sampler does not run in the graph being compared and its TF32 sensitivity
cannot affect stage 5 at all. The rule bites only on the public request path,
where the question becomes whether the same text and seed must yield the same
tokens on CPU and CUDA. That promise is cheap for VITS and Kokoro and very
expensive here.

Stage 7 has to take that decision deliberately and record it. Until it does, the
honest planning number for a Stage 1 **CPU** claim is roughly ten times slower
than real time, and nothing is claimed about CUDA.

The 1.7B ladder rung must be measured before it is promised anything.

### Greedy is reproducible, but only with both switches

The first attempt at this measurement produced two "greedy" runs that differed:
69,120 against 65,280 frames, diverging at sample 20 — inside the very first
frame. That looked like the material finding this plan warned about, that a
greedy oracle is not reproducible against itself and the replay seam needs more
than a captured code sequence.

It was not. `generate_custom_voice` takes **two** sampling switches, and
`do_sample` governs only the Talker. The sub-talker has its own
`subtalker_dosample`, which defaults to `True`, so the code predictor was
sampling under a nominally greedy call.

With `do_sample=False` **and** `subtalker_dosample=False`, two runs in separate
processes are bit-identical: 97,920 frames each, `max_abs_diff` exactly 0.0,
peak agreeing to the last digit at 0.5742930173873901.

So a reproducible oracle does exist, and the
stochastic-replay seam needs only the captured code sequence the family plan
assumed. **Any script that captures this oracle must set both switches**;
setting one is silently non-deterministic, which is a worse failure than an
error.

Unseeded sampling behaves as expected and sets the stochastic capability: two
runs gave 51,840 and 76,800 frames, differing by 0.70 over the common prefix.

### Stage 1 claims Chunked Audio Delivery

`CONTEXT.md` defines Native Streaming Synthesis as a *validated* capability to
produce usable audio incrementally, so architecture alone cannot earn the claim.
The call used here returns a complete waveform, and upstream is explicit that
its own flag does not change that: `non_streaming_mode` "currently only
simulates streaming text input when set to `false`, rather than enabling true
streaming input or streaming generation".

Stage 1 therefore claims **Chunked Audio Delivery**. The stronger claim remains
reachable — the findings above name its cost, three pieces of decoder state
carried across frames — and would need its own validated evidence at a later
stage. Recording it this way keeps the two claims from being conflated by
default, which `CONTEXT.md` warns against in both directions.

## What Three More Ports Say

Read on 2026-07-28: `HaujetZhao/Qwen3-TTS-GGUF`, `cgisky1980/Qwen3-TTS-Rust`,
`mzyfc/Qwen3-TTS-ncnn`. Nothing was copied.

**Licences differ and one is absent.** The Rust port declares
`MIT OR Apache-2.0` in `Cargo.toml`. The ncnn port states none of its own and
defers to its `THIRD_PARTY_NOTICES.md`. The GGUF port ships **no licence at
all** -- the Apache-2.0 headers inside it belong to a vendored copy of upstream
-- so it is readable but nothing in it may be reused. Treat all three as
read-only until a licence is confirmed, and remember the
`THIRD_PARTY_NOTICES.md` obligation applies to code adopted, not to
understanding gained.

### Every existing port splits the model; this project does not

All three separate the Talker from the Code Predictor and run the codec
elsewhere: the GGUF and Rust ports drive two llama.cpp GGUFs plus ONNX
Runtime for the codec, and the ncnn port exports the Talker and Code Predictor
as separate ncnn graphs. This project's single versioned C ABI over one GGML
runtime is the harder path, and it is a deliberate difference rather than an
oversight -- but it is worth knowing that no existing port attempts it.

### The Code Predictor is the bottleneck, not the Talker

The GGUF port states it directly, and the arithmetic is checkable: one second of
audio is 12.5 frames, and each frame needs 15 sequential Code Predictor steps,
so the predictor runs 187.5 steps per second against the Talker's 12.5. At the
0.6B rung the predictor is roughly 0.1B against the Talker's 0.6B, which puts
about two and a half times more work per second in the predictor. That is why
the same port reports the 0.6B and 1.7B rungs performing similarly: the rung
size changes only the Talker.

This is where stage 4's effort belongs, and it explains why `qwentts.cpp`'s
cache-reset rewrite of the predictor inner loop mattered as much as it did.

### Quantized CPU inference is far faster than the PyTorch reference

| Port | Backend | Quantization | Real-time factor |
| --- | --- | --- | ---: |
| Rust | CUDA | Q5_K_M | 0.553 |
| Rust | CUDA | Q8_0 | 0.640 |
| Rust | CPU | Q5_K_M | 1.677 |
| Rust | CPU | Q8_0 | 1.866 |
| GGUF, 1.7B | discrete GPU | Q5_K | 0.35 |
| GGUF, 1.7B | CPU | Q5_K | 1.3 |
| GGUF, 1.7B | integrated GPU | Q5_K | 1.3 |

Two independent implementations put quantized CPU inference between 1.3 and
1.9, on the same or a larger rung than this project's Stage 1 target. The 9.4
recorded from this project's own oracle is PyTorch, float32, eager attention --
an unoptimised reference, not a prediction of what a GGML port achieves.

**This changes the open question about the discrete-output rule.** The cost of
holding the autoregressive core on CPU is not the six-to-nine times implied by
comparing the PyTorch CPU reference against CUDA. Measured against these ports
it is closer to two or three times, from roughly 0.55 to roughly 1.7, and stays
near real time. That is a far cheaper price for cross-backend determinism than
the earlier framing suggested, and the decision should be taken against these
numbers rather than against the reference's.

### Smaller corroborations

The GGUF port exposes **independent seeds for the Talker and the Predictor**,
which is the same two-sampler structure this project found in
`do_sample`/`subtalker_dosample`. The ncnn port chose deterministic greedy
generation for its numerical acceptance work, the approach this project tried
and abandoned after finding greedy degenerates on some speaker-and-input
pairings -- worth watching whether they hit the same wall.

Their voice-clone path is in-context learning: reference text and target text
are concatenated, and the reference audio is injected as a speaker embedding
plus its codes so the model continues in that voice. That is the mechanism the
Reference Model Variant Ladder's later rungs will need, and it is not what this
Stage 1 CustomVoice variant does.

## Where the Time Actually Goes

Measured 2026-07-28 on the pinned reference, one sentence of English, with the
talker, the code predictor and the codec timed separately.

| | talker.model | code predictor | codec decode |
| --- | --- | --- | --- |
| CUDA BF16 | 38 calls, 1.00 s, 30.0 % | **555 calls, 1.45 s, 43.5 %** | 0.16 s, 4.9 % |
| CPU F32 | 42 calls, 8.97 s, 27.1 % | **615 calls, 22.46 s, 67.9 %** | 1.19 s, 3.6 % |

The call counts confirm the structure rather than assuming it: 555/37 and
615/41 are both exactly 15 predictor steps per frame.

### The codec is not worth outsourcing

The other three ports run the codec in a second runtime -- ONNX Runtime in two,
specialised ncnn graphs in the third. On these measurements that cannot be a
performance decision: the codec is **3.6 % to 4.9 % of synthesis**, at a
real-time factor of 0.053 on CUDA and 0.370 on CPU. Both are far under real
time, and a perfect codec would return at most four percent.

Against that, a second runtime costs a second dependency, a second backend
abstraction beside `BackendPlan`, and a Model Package that is no longer one
GGUF. The operator surface argument that might justify it does not apply here
either: intake measured the incremental surface at zero, because
`ggml_col2im_1d` is upstream and already exercised by VITS, and SnakeBeta is a
parameter on Kokoro's existing `snake()`.

**Decision: implement the codec in GGML like every other stage.** The reason
the other ports outsourced it is build effort, not speed, and this project has
already paid most of that cost in two earlier families.

### The code predictor is overhead-bound, which makes it an opportunity

It is the largest single cost, and it is *not* compute-bound. One step is
5 layers at hidden 1024 with intermediate 3072 and a 2048-entry head:

    per-token MACs   80.7 M      ->  161 MFLOP
    CPU F32   36.51 ms/step      ->   4.4 GFLOP/s
    CUDA BF16  2.61 ms/step      ->  61.9 GFLOP/s

A GB10 is a teraflop-class device, so the reference is running the predictor at
well under one percent of the hardware. The arithmetic is tiny; the cost is
what surrounds it.

The reason is visible in the reference. Every frame calls a full HuggingFace
`generate()` on the predictor for its fifteen steps -- generation-config
resolution, logits processors, stopping criteria and cache allocation, twelve
and a half times per second of audio.

The loop this project has to build is small and completely static:

1. reset the predictor's KV cache for the frame;
2. prefill exactly two positions, the talker hidden state and the embedding of
   the semantic code;
3. run `code_group_count - 1` single-token steps, where step *i* uses **its own**
   embedding table and **its own** output head -- the private tables per code
   group that the configuration does not reveal;
4. sum the sixteen code embeddings to form the next talker input, adding the
   trailing text hidden state while one remains.

Every step has identical shapes, so the graph can be built once and replayed
with a new code id, which is what `qwentts.cpp` reports taking its inner loop
from quadratic to constant. **This is the one place where this port should
expect to beat the reference substantially rather than merely match it**, and
it is where stage 4's effort belongs.

## The Code Predictor as Built

Built 2026-07-28 in `src/arch/qwen3-tts/code-predictor.{h,cpp}` and its `-host`
counterpart, over the shared decoder block in `operations.{h,cpp}`.

Three things about the block are specific enough to get wrong silently, and all
three are now pinned by a test against the reference's own
`Qwen3TTSDecoderLayer`:

- **Per-head QK-Norm.** Qwen3 normalizes each head of q and k at `head_dim`
  before rope, not the packed projection. Qwen2 has no such norm at all, so a
  block carried over from a Qwen2 port would simply lack it.
- **NEOX rope.** The halves of each head rotate against each other. The
  interleaved GPT-J layout applies the same angles to the wrong elements, which
  degrades quality without failing anything.
- **Grouped attention by broadcast.** `ggml_mul_mat` maps query head `i` to key
  head `i/(q_heads/kv_heads)`, which is exactly the blocked grouping the
  reference's `repeat_interleave` produces, so no materialized repeat is needed.

The talker and the predictor share this block exactly: 16 query heads over 8
key/value heads, `head_dim` 128, hidden 1024, intermediate 3072, `rope_theta`
1e6, `rms_norm_eps` 1e-6, every layer `full_attention` with no sliding window.
They differ only in layer count -- 28 against 5 -- and in rope type.

**The key/value cache is written in place through a view, never grown by
concatenation.** For the predictor's sixteen positions the two are equivalent.
For the talker they are not: concatenation copies the whole cache once per step,
which turns an utterance from linear into quadratic. The block therefore takes a
persistent cache and a graph, and expands its cache writes before the reads that
depend on them.

### The step schedule

Per frame the reference calls `generate()` once and takes fifteen steps inside
it. Written out, that is one prefill of two positions -- the talker's hidden
state and its embedding of the semantic code -- followed by fourteen
single-token steps, sixteen positions in total, producing fifteen acoustic
codes. The cache is per frame and starts empty each time.

The table and head indices are not aligned, which is the trap:

| call | positions | embedding table | output head | code produced |
| --- | --- | --- | --- | --- |
| prefill | 0, 1 | none -- the talker supplies both | 0 | group 1 |
| step g (1..14) | g+1 | g-1 | g | group g+1 |

A call embeds the *previous* code through the table of the group that code
belongs to, one behind its own head. A wrong table still yields plausible codes
and only subtly wrong audio, so the mapping lives in
`code_predictor_schedule()` as data rather than as arithmetic scattered through
a loop, and is unit-tested at the real group count of sixteen.

`small_to_mtp_projection` is an `Identity` for this variant because the talker
and predictor hidden sizes both equal 1024, so the package carries no tensor for
it. The 1.7B rung would need one, and the builder rejects a projection bound on
one side only.

### Selection is a host seam

Sampling turns a distribution into a value that indexes the next step's
embedding table, so a code differing by one between backends changes the entire
rest of the frame. That places it and its input path on the CPU under
`docs/backends.md`. The filter order is the reference's -- temperature, then
top-k, then top-p, then the draw -- and the package's generation defaults for
this head are temperature 0.9, top-k 50, and top-p 1.0, which keeps the whole
distribution and is therefore inert unless a request overrides it.

The draw itself is not comparable to the reference: PyTorch's generator and this
project's seeded stream are different streams by construction, which is why the
Port Validation Contract replays codes rather than reproducing them. What must
match is the distribution behind the draw, so the frame test decodes greedily
and compares every step's logits. They agree to 4.8e-07 on the CPU and 8.3e-07
on CUDA, and every code matches.

## The Package as Cut

Re-cut 2026-07-28: **657 tensors, 2274 MB**, down from 818 and 2493.

### The codec encoder is not carried

The speech tokenizer's encoder half turns audio into codes. Synthesis runs the
other way, so no graph in this package can reach it, and this checkpoint could
not use it regardless: `model.safetensors` is 402 tensors, all `talker.`, with
no speaker encoder at all, and the reference builds voice-clone prompts from the
**Base** variant. Sixteen of the encoder's codebooks and both of its quantizer
projections were bit-identical to the decoder's, so part of the 225 MB was
literally the same weights stored twice.

Dropping it is reversible for one conversion run, and leaves the catalog
covering exactly what a graph can reach -- no region that is present, unread and
therefore unvalidated. Reference Audio voice cloning is expected to need a
different checkpoint variant, which re-cuts the package anyway.

### Two defects the catalog surfaced

Both would have failed at load, and neither was visible before something tried
to resolve tensors by name.

**27 names exceeded `GGML_MAX_NAME`.** GGML stores a tensor name in a fixed
64-byte field and truncates past it without a word. A truncated name is not
findable by the name the catalog asks for, and two names differing only past the
cut become one tensor. Five path patterns overflowed, the longest at 70
characters. The components that made them long are shortened --
`post_attention_layernorm` to `post_attn_norm`, `self_attn_layer_scale` to
`self_attn_scale`, `mlp_layer_scale` to `mlp_scale`, `_codebook.codebook` to
`codebook` -- and the converter now refuses to emit any name at or over the
limit, which is the part that keeps it from recurring.

**The Voice catalog and the speaker table were ordered differently.** The
generic `synthesize.voice.N.id` list was alphabetical and
`synthesize.qwen3-tts.speakers.names` was by codec token id. They describe one
catalog and the loader reads them index by index, so every entry named one
speaker and would have selected another. The loader's own consistency check
refused the package outright, which is how this was found. Both now follow the
manifest's order.

### Codec geometry is in the package

The package carried only sample rate, hop and frame rate, which is not enough to
derive a single codec tensor's shape. It now carries the decoder's own numbers
under `synthesize.qwen3-tts.codec.decoder.*`, and the converter checks them
rather than copying them: the upsample factors must multiply to exactly one
frame of samples (8 x 5 x 4 x 3 x 2 x 2 = 1920), the residual stack must halve
once per stage, and the codec must take the number of code groups the talker
emits.

| | value |
| --- | --- |
| latent / decoder dim | 1024 / 1536 |
| codebook dim / size | 512 (quantizer runs at 256) / 2048 |
| quantizers | 16, of which 1 semantic |
| transformer | 8 layers, hidden 512, intermediate 1024, 16 heads x 64, sliding window 72 |
| residual stages | rates 8, 5, 4, 3 -- widths 1536 -> 768 -> 384 -> 192 -> 96 |
| ConvNeXt upsample | ratios 2, 2 at latent width |

## The Tensor Catalog

Built 2026-07-28 in `src/arch/qwen3-tts/catalog.{h,cpp}`.

Every entry is resolved by canonical name with its storage type and shape
checked, and the shape is **derived from the package's hyper-parameters** rather
than written out. A package whose metadata and tensors disagree is refused at
load; past that point the mistake stops being an error and becomes wrong audio.
Under the source profile the type follows the half -- the talker carries the
checkpoint's BF16, the codec the tokenizer's F32. No quantized package exists
for this family yet, so a package claiming one is refused rather than measured
against a rule nobody has written.

After resolution the package is swept: **a tensor the catalog never asked for is
an error.** A name nobody resolves is a name nobody checked.

Confidence comes from two independent statements of the same package agreeing.
The unit test writes the entry list out by hand from the checkpoint's layout
while the catalog derives it from hyper-parameters; and `expected_tensor_count()`
derives the total by arithmetic while the resolver derives it by enumeration. On
the real package the arithmetic gives 657 against 657 present, and every region
binds -- 28 talker layers, 15 predictor heads, 4 residual stages, 15 acoustic
codebooks.

One shape trap is worth naming because it survives an element-count check:
`Conv1d` stores `[out, in, kernel]` and `ConvTranspose1d` stores
`[in, out, kernel]`, so GGML reports `[kernel, in, out]` against
`[kernel, out, in]`. The two differ only in the order of the trailing pair. A
converter that confused them produces a package with exactly the right number of
elements in every tensor, so the catalog resolves them through separate helpers
and the test pins the swap as a rejection.

## The Talker as Built

Built 2026-07-28 in `src/arch/qwen3-tts/talker.{h,cpp}` and its `-host`
counterpart.

### The multimodal rope really does collapse

The talker's attention is the shared block's except that it goes through the
reference's `apply_multimodal_rotary_pos_emb` with `interleaved=True` and
`mrope_section = [24, 20, 20]`, where the code predictor uses the plain helper.
Intake argued from the shapes that this is exactly plain rope for this model
because every `position_ids` path yields three identical rows. That argument is
now checked: the test runs the real `Qwen3TTSTalkerModel` with a scaled-down but
genuinely interleaved section shape, and the port's hidden state agrees to
3.6e-07 on the CPU. Position ids are plain `0..n-1` --
`attention_mask.cumsum(-1) - 1` with no left padding at batch one, and
`position_id_per_seconds` never enters.

### Two towers meet at every position

The talker's input is a sum, not a concatenation. The text side is a token from
the 151936-entry text embedding at width 2048, brought to 1024 by
`text_projection`: two linears **with biases** and a SiLU between them. The codec
side is a row of the talker's own 3072-entry codec embedding. Both are pinned by
the test, because a mistake in either is a wrong voice rather than an error.

### The prompt layout

Every entry below is off-by-one bait, and none of it fails loudly:

| position | text side | codec side |
| --- | --- | --- |
| 0..2 | the role prefix's tokens | *none* |
| 3 | tts_pad | codec_think, or codec_nothink for auto |
| 4 | tts_pad | codec_think_bos |
| 5 | tts_pad | the language token, **absent** for auto |
| 6 | tts_pad | codec_think_eos |
| 7 | tts_pad | the speaker token, absent without a preset Voice |
| 8 | **tts_bos** | codec_pad |
| 9 | the **first** text token | codec_bos |

Ten positions with a language and a speaker, nine for either alone, eight for
neither. What makes it a trap:

- The codec stream is one longer than the text stream beside it, because its last
  token belongs to the position after it.
- The single tts_bos in the whole prompt sits one before the end.
- codec_bos pairs with the first text token, not with a pad -- which is why the
  schedule below starts at the *second* text token.
- Asking for auto is a shorter prompt, not the same prompt with a default
  language. nothink replaces think and no language token is emitted at all.

Decode step `k` then adds, on top of that frame's sixteen summed code
embeddings, `trailing[k]` -- the text tokens from the second onward, then one
tts_eos -- and the projected tts_pad embedding for every step past the end. That
last part is what lets the talker keep emitting frames after the text runs out.

A frame whose **semantic code** is `codec_eos` ends the utterance, and that frame
is not part of it.

### The layout is a host seam

It is entirely discrete -- token ids and which table each position reads -- so it
returns positions rather than tensors, and the graph only performs the two
lookups and the sum. The flattener then checks what the graph relies on instead
of assuming it: every position carries a text token, and the codec positions form
a contiguous tail. A gap would mean the codec stream is not a tail, and the graph
accumulates it into the text stream as one.

### A test hazard worth recording

Reading an intermediate back after `ggml_backend_graph_compute` is unsound
without `ggml_set_output`: the graph allocator is free to reuse its buffer for a
later node. This test reads four tensors including two intermediates, and
`ggml_acc` was handed the text tower's own buffer, so the prefill readback was
off by 2.05 while the final logits were correct to 3.9e-07. Marking the outputs
fixed it. A test that reads only the final tensor never meets this.

## The Codec Decoder as Built

Built 2026-07-28 in `src/arch/qwen3-tts/codec.{h,cpp}`. Codes in, waveform out,
agreeing with the reference's own `Qwen3TTSTokenizerV2Decoder` to **1.5e-06** on
the CPU across a 442-node graph.

### It is not the shared block

The codec's transformer looks like a Qwen3 block and is not one:

- **No per-head norms.** Adding them would renormalize vectors the reference
  leaves alone.
- **Per-branch layer scales.** Each residual branch is multiplied by a learned
  per-channel vector, initialised near 0.01, before being added back. Leaving one
  unbound runs that branch a hundredfold hot.
- **A sliding window of 72 frames** on top of causality. That is under six
  seconds at 12.5 Hz, and an utterance routinely exceeds it, so the window is
  part of the model rather than an optimisation to skip.

Residual quantization sums its levels rather than concatenating them -- each
level refines the one before it -- and the semantic/acoustic split is the same
one the talker and the predictor make.

### Layout and operators

The stack is channel-major, `ne = [channels, length]`, matching the rest of the
family; the reference transposes around its pre-transformer and this does not
have to, because the transformer already wants `[hidden, positions]`.

Two ggml facts forced the operator choice, both found by running rather than
reading:

- **`ggml_conv_1d`'s CPU path asserts an F16 kernel** and this family's are F32.
  Convolutions are built from `ggml_im2col` and a matrix multiply, which is what
  VITS already does for the same reason.
- **`nn.GELU()` is the exact erf form; `ggml_gelu` is the tanh approximation.**
  The difference is 2.4e-4 in a single ConvNeXt block -- small, but two thousand
  times what the rest of the stack disagrees by, and the real decoder has sixty
  of them. `ggml_gelu_erf` brings it to 1.8e-07.

Causality is the recurring hazard. Every convolution sees only the present and
the past, which is left-only padding; ggml pads symmetrically, so each pads wide
and keeps the prefix. The transposed form is causal by cropping instead, dropping
the trailing `kernel - stride` samples. Either mistake shifts the waveform in
time and fails nothing.

### Testing notes worth keeping

**Saturation hides errors.** At the same weight scale as the rest of the stack,
random weights drove forty percent of the reference waveform into the ±1 clamp,
where a wrong sample matches a right one. The final convolution is drawn small on
purpose so nothing saturates.

**The codebooks are EMA accumulators.** The reference divides `embedding_sum` by
`cluster_usage`; holding the usage at exactly one makes the accumulator the
table, so the test can fill the table directly. It must not draw the usage from
the shared weight stream -- doing so shifted every weight after the first
codebook and produced a waveform wrong by 0.98.

**`%.9g` renders 1.0 as `1`, and `1f` is not a C++ float literal.** This is the
first reference whose output reaches exactly one, because the clamp puts it
there. Every dump script now formats through one guard.

**The accelerator tolerance here is looser than elsewhere in the family**, at
2e-2 against 1e-4 on the CPU, and the test says why: 442 nodes with exponentials
in them, TF32 matmuls, and a diffuse error -- mean 7e-4 with no outlier, against
2.3e-07 on the CPU for the same graph. It still catches a wiring fault by a wide
margin; the one above showed up at 0.98.

### Unused in the decode path

The package binds `quantizer.*.input_proj` for both quantizers -- four tensors,
about 1 MB -- and nothing reads them. They are the encode direction of the
projection pair. The catalog resolves them so the package sweep stays absolute;
dropping them would mean filtering inside a module rather than at a prefix.

## The Text Frontend as Built

Built 2026-07-28 in `src/arch/qwen3-tts/bpe.{h,cpp}`. Byte-level byte-pair, which
no frontend in this project provided: the symbol-map frontend maps one symbol to
one id, and this maps a byte sequence to a sequence of merged pieces.

Against the real 151,643-entry vocabulary it reproduces the reference
tokenizer's ids **exactly on all thirteen cases**, including Chinese, Japanese,
Korean, an emoji outside the basic plane, contractions and the prompt template.

### The pre-tokenizer is the risky stage

The Qwen pattern is

    (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+

No C++ standard library implements `\p{L}`, so it is written out branch by
branch, first-match-wins, and the classes come from `src/unicode-ranges.h` --
806 ranges generated from Python's own Unicode data by
`scripts/generate-unicode-ranges.py`. Classifying by hand would be wrong exactly
at the edges, and the edges are where text-to-speech input lands.

Two rules in that pattern are easy to miss and change every downstream token:

- **A leading space belongs to the word after it**, so `" ab"` is one piece.
- **Whitespace gives up its last character** when a non-space follows, which is
  what makes that possible.

Merging is greedy by **rank**, not by position: the lowest-ranked pair anywhere
in the piece merges first. Taking pairs left to right produces a different and
equally plausible tokenization.

### Special tokens are added tokens

`<|im_start|>` is id 151644 against a vocabulary of 151643. They sit *past* the
vocabulary and cannot be looked up in it, so each is configured with its own id,
taken from the package's token metadata. This was found by running the frontend
against the real package, which refused it.

### The prompt template is a fixed string

The checkpoint carries no chat template. The reference wraps every request in

    <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n

and slices the result at 3 and -5 -- the role prefix and the closing markers --
which is exactly what the talker's prompt layout expects.

### Not registered yet

The comparison against the real vocabulary needs the package, so it belongs to
the integration tier this family does not have. The unit test pins the split
against the reference pattern's own output, which is the only stage checkable
without a 151k-entry vocabulary.

## Orchestration and the Public Seam

Built 2026-07-28 in `src/arch/qwen3-tts/model.cpp` and wired through
`src/synthesize.cpp`. `"Hello there."` with the `aiden` voice produces fifteen
frames, 28,800 samples at 24 kHz -- 1.20 seconds of audio in the range
[-0.42, 0.61].

### Placement: everything is on the CPU

The sampled code is a discrete output, and `docs/backends.md` holds a discrete
output and every stage feeding it on CPU. Here that is the talker *and* the code
predictor, which the intake measured at **73.5 %** of synthesis. The codec, the
only stage that could sit on an accelerator, is **4.9 %**.

So every graph runs on the CPU scheduler and the weights live in the CPU buffer.
No mirroring is needed, because nothing reads them from two places -- unlike
Kokoro and VITS, where the held stages were a minority and the rest stayed on the
primary backend. Whether the codec is worth a second buffer for a 4.9 % ceiling
is stage 7's decision, with the measurement in hand.

### Four seam findings

None of these could have been found by a unit test: all four live at the boundary
between the family and the core.

**`ModelInfo::vocab_size` means the vocabulary the frontend's ids index.** The
core range-checks every returned token against it. That is the text tower's
151,936, not the codec's 3,072; mapping it to the codec's rejected every request
with `SYNTH_ERR_INVALID_ARG` and no diagnostic.

**The core resolves an absent language tag to `"en"`.** The package names its
languages in full (`english`, `chinese`, …), so the family bridges BCP-47 to
those names on the primary subtag alone. The two dialect entries are deliberately
absent from that table: they are speaker overrides the reference reaches through
`spk_is_dialect`, not languages a request can ask for.

**The declared frame limit is not a cache size.** The talker attends over the
whole utterance at 229 kB a frame, and the package declares fifteen million
frames -- a generic "no limit". Sizing the cache from it asked for **three
terabytes**. It now comes from the request's effective limit, with a default of
2048 frames: about 164 seconds and 481 MB.

**ggml's elementwise operators reject BF16.** `ggml_mul_mat` reads a BF16 weight
directly; `ggml_add` and `ggml_mul` abort inside `binary_op`. Every norm gain and
bias in the talker half is BF16 because that is what the checkpoint stores, so
they are cast where they meet an elementwise operator.

### The language seam, and how it was found

The core used to be English-only in two places: `synth_model_get_language_count`
returned 1 with a hardcoded `"en"`, and the request validator ran a
`supported_english_tag` check before dispatch. This family is the first
multilingual one, so nine of its ten requestable languages were unreachable
through the public interface -- and therefore through the CLI and both bindings.

That was noted here as a known limit and shipped anyway, which is the part worth
recording. Every one of the eighteen Golden cases passes with Chinese, Japanese,
German and Korean text, because the replay harness drives the family directly and
never crosses the validator. Stage 5's public-request phase did cross it, but only
ever with `en`. A gate that no test approaches is a gate that stays shut, and it
took someone asking for a long Chinese sentence to find it.

The fix moves the declaration to where `docs/languages.md` already said it lived:
`ModelInfo::languages` carries BCP-47 tags with `SYNTH_LANGUAGE_*` flags, each
family fills it, the validator matches against it, and the enumeration reports it.
VITS and Kokoro declare `en` with regional fallback, which is exactly what the
hardcoded test did for them. This family maps its package's full names --
`chinese`, `japanese` -- to tags on the way out, dropping the two dialect entries,
which are speaker overrides rather than requestable languages.

One divergence to settle: `docs/languages.md` says a multilingual variant without
a declared default requires an explicit tag. This family declares no
`SYNTH_LANGUAGE_DEFAULT`, and omitting the tag gives the reference's no-think
prompt, which carries no language token at all rather than falling back to one.
That is the reference's behaviour and is not the same as "has no default".

The turn wrapper lives in the frontend rather than in the core. The core runs the
frontend, and for this family wrapping the request in an assistant turn is part
of turning text into the ids the model consumes -- so it belongs there rather
than as a family branch in the core.

### Not yet measured

Synthesis of 1.2 seconds took 30 seconds of wall clock on CPU. That is a
real-time factor of 25 against the reference's measured 9.4, and it is expected
to be dominated by rebuilding a graph per decode step -- 15 predictor steps and
one talker step per frame, each allocating a scheduler. `qwentts.cpp` reports
building the predictor's inner graph once and replaying it. That is stage 7's
work and the intake already named it as where this port should beat the
reference rather than match it.

## Stage 5: Oracle Replay

`scripts/validate-qwen3-tts-replay.py` drives `synthesize-qwen3-tts-replay-real`
over the Golden Manifest's cases and compares the port against the oracle's
artifacts.

### Why replay rather than reproduce

The oracle sampled its codes from PyTorch's generator, and this port draws from
its own seeded stream. Reproducing the same codes is not possible and is not the
claim; the contract's seam is to **replay** the oracle's codes and compare
everything downstream of the draw on identical inputs. So `run_synthesis` takes
an optional code stream, and with it set the loop never samples and never
consults the stop code -- the replay's length ends it.

### What is compared, and how

The oracle hooks the talker's **first forward alone** -- the prefill, the pass a
port reproduces without the sampled history behind it. So the probes are the
whole prompt's hidden states at layers 0, 7, 14, 21 and 27, the final norm, and
the codec head over every prompt position. The waveform is compared separately,
after the oracle's codes are fed back through the codec.

Deep talker layers are compared by **cosine similarity, not max-abs**. The
tolerance file recorded this before any measurement existed, from reading the
reference port: per-layer max-abs of 13.07 / 13.09 / 13.42 at L7 / L14 / L21 and
66.8 at L27 while cosine held at or above 0.9998. Those are outlier channels
ahead of the final norm, and a max-abs threshold would report catastrophic
failure on a correct port. The measurements below have the same shape.

### Four defects it found

None was reachable by a unit test: all four are agreements with the reference
that only the reference can settle.

**The prompt was the streaming layout.** `generate_custom_voice` defaults to
`non_streaming_mode=True`, which puts the whole text in the prompt and leaves the
decode loop nothing but padding to contribute. The streaming layout puts one
token in the prompt and feeds the rest a frame at a time. Both synthesize speech.

**The codec block was one position short.** `tts_bos` pairs with `codec_pad`, and
only `codec_bos` is held back for the position closing the prompt.

**The probes were captured per frame rather than over the prefill.** A per-frame
capture has exactly the same shape as a prefill capture whenever the frame count
happens to equal the prompt length -- which it did on the first case tried. That
is how a wrong capture survives a shape check.

**The replay adapter read the acoustic codes level-major.** The oracle splits
column 0 from columns 1..15 of a `[frames, 16]` block, so that stream is
frame-major. Transposed codes decode to audio rather than to an error.

Before these, layer 0 sat at a cosine of **-0.01**.

### Measured tolerances

All eighteen Golden cases pass. The oracle runs bfloat16 on CUDA and the port
runs F32 on CPU, so these cover a dtype and a device difference as well as an
implementation one. That comparison is permitted rather than merely tolerated --
see `docs/port-validation.md`, which was revised on 2026-07-30 after its earlier
wording turned out to forbid the only comparison a BF16-checkpoint family can
make -- and it comes with a condition this family has not met yet: the dtype and
implementation components of each threshold are not separated.

| probe | worst cosine | worst max-abs | gate |
| --- | --- | --- | --- |
| talker.hidden_l0 | 0.999992 | 0.084 | cosine 0.99996 |
| talker.hidden_l7 | 0.999986 | 4.53 | cosine 0.99993 |
| talker.hidden_l14 | 0.999971 | 7.63 | cosine 0.999855 |
| talker.hidden_l21 | 0.999942 | 12.05 | cosine 0.99971 |
| talker.hidden_l27 | 0.999915 | 17.39 | cosine 0.999575 |
| talker.final | 0.999452 | 3.84 | cosine 0.99726 |
| talker.logits | 0.999522 | 1.44 | cosine 0.99761 |
| audio.pcm | 0.999427 | 0.154 | cosine 0.997135, max-abs 0.5 |

Every threshold is **five times the measured deviation in (1 - cosine)**. That
headroom absorbs run-to-run variation and stays far tighter than any defect this
suite caught: a transposed code stream gave cosine 0.017 and a prompt one
position short gave -0.01.

The talker probes gate on cosine alone, and max-abs is recorded as observed. The
deep layers carry outlier channels ahead of the final norm -- 17.4 at L27 against
a cosine of 0.99992 -- so a max-abs gate would report catastrophic failure on a
correct port. The waveform gates on both, because a listener hears the waveform
and a cosine over it would hide a constant offset.

`--check` reads the file and refuses to run when a stage is not recorded in it,
so a tolerance stays an input to validation rather than something the suite
writes for itself.

### A fifth defect, in the harness

A dictionary fallback in the validator turned the manifest's `"auto"` language
into `"en"`, so the port built a prompt carrying a language token where the
oracle built one without. That is one position longer, and it surfaced as a shape
mismatch rather than a number -- the honest failure, but the cause was the
harness and not the port.

### Phase 3: the public seam

`scripts/validate-qwen3-tts-public.py` drives `synth_synthesize` the way a caller
does, injecting nothing. Eight checks, each of which can break while every tensor
comparison still passes:

| check | result |
| --- | --- |
| the seed is reported as requested | 7 in, 7 out |
| the same seed reproduces byte for byte | identical digests |
| a different seed changes the audio | seed 7 and seed 8 differ |
| a random seed is reported concretely | reported 11889680619108445593 |
| the reported random seed reproduces | identical digests |
| a different Voice changes the audio | aiden and vivian differ |
| the resolved Voice is reported | `vivian` |
| a dialect speaker still synthesizes | 21,120 frames |

Byte-identical rather than close, because the sampler is the only stochastic part
and it is seeded. Reporting a seed that does not reproduce would be worse than
reporting none, so the random-seed path is replayed rather than merely observed.

**This table is stale and pre-dates this document's own later corrections.**
The eight checks and their results above are a historical snapshot; the
CustomVoice BF16/CPU `public` cell in `tests/tolerances/qwen3-tts.json`
already records 10 checks (2026-08-17, a second kind of Voice) and then 11
(2026-08-19, Stage 3 Plan 2 Task 6's `create_from_description` refusal check).
Not rewritten here — the authoritative, dated count lives in that file, with a
note explaining each move; this table is left as the record of what phase 3
looked like when this section was first written, per this document's own
practice of superseding rather than silently editing history.

### Not done in this stage

Phases 4 and 5 of the contract -- the quantization profiles and the Execution
Backends -- are stages 6 and 7 and have not started. Neither validator is
registered with CTest yet: both need the package, so they belong to the
integration tier this family still does not have.

## Stage 6: Quantization Profiles

### The quantizer refused this family

Its source profile is mixed -- a bfloat16 talker and an F32 codec -- and the tool
accepted only F32 and F16 sources. `dequantize_to_f32` already handled BF16
through its type traits, so the guard was the whole of it.

### Where the policy lands

Classification is by name, as for the other two families, so a converter or
runtime change cannot have a storage type assigned to it by a catch-all suffix
rule. What differs is which tensors are held:

- **The quantizer's codebooks and both kernel-one projections stay at the
  reference dtype.** A residual codebook's later levels carry small magnitudes,
  so a relative error there is a large one against the residual it is meant to
  correct. The whole quantizer is under 35 MB of a 2 GB package.
- **Per-head norms, the codec's per-branch layer scales and the SnakeBeta curves
  stay exact.** Each multiplies or seeds a whole head or branch, the layer scales
  start near 0.01, and together they are a rounding error of the file.
- **The six transposed convolutions stay F32**, the override VITS makes for the
  same reason: they run as a column matrix multiply into `col2im_1d` and CUDA's
  F16 multiply accumulates in half precision.

The catalog carries a **role per tensor** rather than re-deriving one from the
name. The quantizer classifies by name and the runtime by role; having the
runtime repeat the name classification is how the two drift apart.

### F16 is a speed profile, not a size one

It is **the same 2168 MB as the source** and **4.5 times faster**. That is not
what a halving profile usually buys, and the reason is ggml rather than the
model: its CPU bfloat16 matrix multiply is far slower than its F16 one, so
halving weights that were already two bytes wide is nearly free accuracy and a
large amount of time.

| stage | BF16 source | F16 |
| --- | --- | --- |
| talker | 13.66 s | 1.74 s |
| code predictor | 25.11 s | 3.65 s |
| codec | 4.20 s | 4.19 s |
| **wall** | **43.04 s** | **9.64 s** |

A 37-frame case is 2.96 seconds of audio, so the real-time factor goes from 14.5
to **3.3** -- faster than the reference's own measured 9.4 on CPU.

Accuracy is a wash and on the deep talker layers marginally better, which follows
from where the bits are: F16 carries ten mantissa bits against bfloat16's seven.
The waveform agreement is *identical* to the source profile's, because the codec
that produced it is the same F32 graph.

All eighteen cases pass under the committed `f16-vs-oracle` stage.

### The codec is never halved

Halving it goes the wrong way. Its convolutions run through `ggml_im2col` into a
matrix multiply, and ggml's F16 path there is slower on CPU than its F32 one: a
uniform F16 profile took the codec from 4.2 seconds to **7.3**, and the whole
case from 9.6 to 12.8. So the codec half stays at the reference dtype under every
profile, in the quantizer's policy and the catalog's expectation both. It is 457
MB of space that costs more time than it returns.

### Q8_MIXED, and the refusal that was wrong

Built 2026-07-29. **1359 MiB against 2169, and a real-time factor of 0.85 against
F16's 1.24** on the same 9.6 second case: smaller *and* faster than real time.

This profile was refused for a month on a reason that did not survive being
checked. The refusal said packing a convolution kernel was a runtime change this
family had not made -- true in itself, and irrelevant, because
`classify_qwen3_codec` reports every codec tensor as sensitive under every
profile. No convolution kernel ever reaches a quantized type, so none is ever
packed. The belief was never tested, and nothing failed while it was wrong.

What actually blocked it was one gate in the quantizer. A two-dimensional weight
must not be packed -- a `[1024, 3072]` projection would become one row of
3145728 -- and the guard that says so was written for Kokoro and gated on its
name. This family's entire quantizable half is two-dimensional, so without that
guard every talker matrix was packed into nonsense.

The split, measured from the package:

| half | tensors | size | rows divisible by 32 |
| --- | ---: | ---: | ---: |
| talker | 316 | 1457.6 MiB | all |
| code predictor | 86 | 270.0 MiB | all |
| codec | 255 | 436.0 MiB | 212.0 MiB |

The autoregressive half is 80 % of the package and quantizes without argument:
rows of 1024, 2048 and 3072. The codec's convolutions carry the kernel in the
fastest dimension -- rows of 7, 16, 3 and 1 -- and stay at F32, which is the same
decision F16 makes and for the same measured reason.

Accuracy over the eighteen Golden cases: the waveform is unchanged at cosine
0.999427, because replay supplies the codes and the codec that renders them did
not move. The talker did: its final hidden state and logits fall from 0.9994 to
**0.9956**, about eight times the deviation.

**That number deserves more weight than the unchanged waveform.** Replay cannot
show what a logits difference of that size does to a *draw*, because it does not
draw. In normal operation this profile will sometimes select different codes,
exactly as a different backend does. Whether that is audible is a listening
question this profile has not been asked yet.

One trap worth keeping. `ggml_n_dims` collapses trailing unit dimensions, so
VITS's `decoder.post.weight` at `[7, 32, 1]` reports as two-dimensional. Writing
the guard as a general rule about shape rather than naming the families leaves it
a row of seven and breaks VITS -- which is what the quantizer's own fixture
caught when it was written that way.

### Q5_K_MIXED is buildable and is not recommended

Built 2026-07-29. **1035 MiB against 2169, a real-time factor of 0.82, and the
worst talker logits at cosine 0.9648.** The first two numbers are good and the
third is why this profile is not proposed for publication.

| profile | size | RTF | talker.logits | talker.final |
| --- | ---: | ---: | ---: | ---: |
| F16 | 2168.9 MiB | 1.21 | 0.999458 | 0.999430 |
| Q8_MIXED | 1359.2 MiB | 0.85 | 0.995731 | 0.995634 |
| Q5_K_MIXED | 1035.3 MiB | 0.82 | 0.964769 | 0.962882 |

Against Q8 it buys 324 MiB and **essentially no speed** -- 0.82 against 0.85,
inside run-to-run variation -- for eight times the deviation in the logits. The
logits are the distribution the draw is made from, and this family has already
demonstrated what a disturbed draw costs: a perturbation an order of magnitude
*smaller* than this gap is what the missing repetition penalty amounted to, and
it truncated sentences.

Where the error is, measured rather than assumed. Every hidden state stays above
0.9993; `talker.final`, one RMSNorm later, is 0.9629. The deep layers carry
outlier channels ahead of that norm -- max-abs reaches 28 at L27 -- so normalizing
amplifies the relative error in every other channel. The amplification of
(1 - cosine) from L27 to final is 7.7x under F16, 33x under Q8 and 55x under Q5_K.

Holding the output head at the reference dtype was tried and does not help:
`talker.codec_head.weight` is 6 MiB and sits *after* `talker.final`, which is
already degraded. Logits moved 0.9648 to 0.9691 and final did not move at all.
The error is 28 layers of accumulation, not one tensor, so the usual remedy of
keeping the output head at higher precision has nothing to fix here.

The profile is left in the quantizer because it is correct and cheap to keep --
one row in the table, one enum, one branch -- and because a future variant with a
shallower talker may want it. It carries no committed tolerances and no published
package, which is the accurate way to say "buildable, not validated".

### The sampler carried its own copy of the checkpoint's decisions

Found 2026-07-29, by listening. Long inputs stopped mid-sentence under every
profile -- English at 198 characters and Chinese at 58 both lost their tail, at a
point that moved with the seed.

The talker ends an utterance by drawing the codec end token, so everything in
front of that draw decides when it stops. This port reimplemented the filter
chain -- temperature, then top-k, then top-p -- and omitted `repetition_penalty`,
which the checkpoint ships at 1.05 and which the oracle samples with. The three
values it did implement were hardcoded at 0.9 / 50 / 1.0 and matched
`generation_config.json` exactly, which is why nothing looked wrong.

**The defect is the second copy, not the missing field.** A port that keeps its
own version of the checkpoint's decisions drifts from them silently, and the only
signal is a listener noticing a sentence end early. The converter now reads
`generation_config.json` and writes all seven values into the package -- the
talker's four and the code predictor's three, which upstream configures
separately and which ships no penalty. The loader requires them, so a package cut
before this refuses to load rather than sampling with something else, and the
generation config is digested beside `config.json`.

Verified by listening on the two reported lines, two seeds, both profiles: they
finish. Recorded for what it is -- one listener, informally.

### A retracted claim: the public CUDA request never moved the sampler

Recorded 2026-07-29, retracted the same day.

This family doc and the tolerance file both carried the claim that requesting
`SYNTH_BACKEND_CUDA` through the public seam placed the sampled path on the
accelerator, and that this was in tension with `docs/backends.md`'s
discrete-output rule. The evidence was that one case drew ten frames on CUDA
against eleven on CPU.

**That comparison used two different binaries** -- a Release CPU build against the
CUDA preset -- and optimization level alone changes CPU float contraction enough
to move a draw. The cleanup investigation found this incidentally while checking
something else: the same CPU code gives one PCM digest under `-O3` and another
under `-O2`.

Re-run properly, one binary, `--backend cpu` against `--backend cuda`, five cases
across English, Chinese and Japanese and three Voices: **frame counts are
identical in every one.** The sampled code sequence does not change. Only the
waveform bytes differ, which is the codec's tensor-core arithmetic and the same
signature the BF16-on-CUDA sweep shows.

So the placement was already what stage 7 measured and what the rule requires --
the codec moves, the talker and the code predictor do not. No change was needed,
and one was nearly made on the strength of a comparison that had two variables in
it.

### Stage 7's remaining gaps, closed 2026-07-29

Three things the family doc had been recording as owed.

**The source profile on CUDA.** Swept: eighteen cases, every talker probe
*bit-for-bit equal* to the CPU entry, which is the placement proving itself
rather than being asserted -- the talker feeds a sampled code and stays on the
CPU, so only the waveform can move, and it does, 0.999427 to 0.999404. Against
F16 on CUDA the waveform is identical to six digits under both profiles; the
codec is almost entirely F32 either way, which is read off the tensor types
rather than tested causally.

**Repeated-run and resource cleanup.** `docs/backends.md` gate 6 has always
required it and **no family had it in code.** Every case of all four manifests
declares `resource_cleanup` in its `checks` array, and outside the manifests and
the schema enum that string appears nowhere in the repository: nothing reads a
case's `checks` at all, and `backend_placement` is equally unenforced. VITS's
recorded evidence was twenty repeated *processes*, which cannot see an in-process
leak because exit reclaims everything -- and `docs/testing.md` already says a
manual run does not substitute for a registered test.

Measured before writing anything: thirty cycles on CPU leave the post-free
resident set oscillating in an 18 MB band with no trend, the minimum falling at
cycle 12; on CUDA it rises 7.6 MB over the first two cycles and then plateaus at
0-156 kB, which is context warm-up rather than a leak, since a leaked codec
buffer is tens of megabytes and a leaked model 1.4 GB. Three cycles under
LeakSanitizer are clean, with a deliberately leaking control program used to
confirm the sanitizer was armed.

`tests/qwen3_tts_public_cleanup_test.cpp` asserts the **floor** of post-free
resident memory rather than last-minus-first. That distinction is load-bearing: a
leak raises the floor, arena churn only raises peaks, and a first draft using
last-minus-first failed on a run whose first cycle happened to land in a trough.

**Twenty Golden cases.** Two utterances roughly twice the previous longest, 18.7
and 21.3 seconds against 9.3. Every committed threshold held and no worst-case
figure moved -- the new cases score *better* than the suite worst on every probe.
The port does not degrade with length. That is worth stating precisely because
the truncation defect appeared only past the old maximum: it was the sampler, and
these cases exist so the suite now reaches the lengths where a defect of that
shape lives.

### Closing the gap: the filter chain is compared, not the audio

Added 2026-07-29. `tests/qwen3_tts_sampling_test.cpp` feeds this port's filter
chain and Hugging Face's own logits processors **the same logits** and compares
the distributions they produce. `scripts/dump_reference_qwen3_tts_sampling.py`
captures the reference side, reading the parameters out of the checkpoint's
`generation_config.json` rather than repeating them -- a fixture built from
hardcoded numbers would reproduce this port's mistake instead of catching it.

**Verified by breaking it.** With the penalty forced back to 1.0 the test fails;
restored, it passes. A regression test that has never been seen to fail is a
guess.

Two designs were tried first and rejected, which is worth recording because both
look reasonable:

- **Duration proportional to text.** Correct behaviour varies by seed enough to
  overlap the truncated behaviour -- at 256 characters the fixed port produced
  16.0 to 22.2 seconds across three seeds -- so a threshold would either miss the
  defect or fire on healthy runs. It also would not have caught this one: the
  longest Golden case is 140 characters, and at that length the broken port
  emitted 117 frames against the oracle's 116.
- **Transcribing the output and comparing it to the input.** This decides the
  question outright, and it is the WER tooling ADR 0017 defers. Adding it here
  would be a large new dependency taken against a standing decision.

What the chosen check does *not* cover: it compares the filters, not the draw,
and not the prompt. A defect in how the prompt is laid out at length would still
reach a listener before it reached the suite. The Golden manifest's longest case
is still 140 characters, and extending it needs oracle runs.

### Why eighteen Golden cases could not see it

The Port Validation Contract's replay seam feeds the oracle's codes so that
comparison is deterministic. That is the right design and it has a consequence
worth naming: **`select_code` is the one stage the Golden suite never executes.**
Every probe in `tests/tolerances/qwen3-tts.json` was within tolerance before and
after this fix, to the digit, because none of them samples.

Stage 5's public-request phase does sample. It asserted relations between runs --
the same seed reproduces, a different seed differs, a different Voice moves the
audio -- and never that the speech was complete. The longest Golden case is 140
characters.

So the suite validated everything except what the model says. There is a unit
test for the penalty's arithmetic now, which is not the same thing: what is still
missing is a check that a long input terminates on the end token with a duration
proportional to its text, and that needs designing rather than asserting.

## Stage 7: Execution Backends

### What can move, and what cannot

The talker and the code predictor feed a sampled code. `docs/backends.md` holds a
discrete output and every stage upstream of it on the CPU, so neither can move --
together they are 59 percent of wall clock under the F16 profile. The codec is
the one stage that is free, and it is the other 41.

The intake estimated that share at 4.9 percent from the reference's own timings.
It is four times that here, because this port's autoregressive half is much
faster relative to its codec than PyTorch's was. Measuring rather than inheriting
the estimate is what made the placement worth doing.

### The measurement

A 37-frame case, same build, F16 profile:

| | codec | wall |
| --- | --- | --- |
| codec on CPU | 8.44 s | 20.69 s |
| codec on CUDA | **0.24 s** | **12.50 s** |

Thirty-five times on the stage and 1.66 end to end.

The codec half gets same-named twins on the primary backend and the catalog binds
it against those. The talker and the predictor get none: a second copy of 1.8 GB
that can never leave the CPU would be read by nothing. The package sweep still
runs over the package alone, because a twin is a placement detail rather than a
tensor the package carries.

### Placement is proven by the probes, not asserted

Running the whole suite with the codec on CUDA leaves **six of the eight probes
bit-identical** to the CPU stage, and `talker.final` agreeing to six decimals --
CPU reduction order is not bit-stable across runs at different thread
schedules. Only the waveform moved, and only in the sixth decimal of its cosine,
which is the codec's tensor-core arithmetic.

That is the shape a correct split has. Had the talker moved, its probes would
have shifted with it.

| probe | CPU F16 | codec on CUDA |
| --- | --- | --- |
| talker.hidden_l0 | 0.999989 | 0.999989 |
| talker.hidden_l27 | 0.999926 | 0.999926 |
| talker.logits | 0.999458 | 0.999458 |
| audio.pcm | 0.999427 | **0.999404** |

All eighteen cases pass under the committed `f16-cuda-codec-vs-oracle` stage.

### Hardware note

Asking for `GGML_BACKEND_DEVICE_TYPE_GPU` misses this machine entirely: the
GB10's CUDA device reports as **integrated**. The runner takes the first device
that is not the CPU rather than the first that calls itself a GPU, and anyone
testing on Spark hardware will need the same.

### It is reachable from the public seam

`synth_model_load` with `SYNTH_BACKEND_CUDA` gets the split: the seam passes the
selected device, `BackendPlan` makes it primary, and the model creates the codec
twins because primary is no longer the CPU. Both paths synthesize the same
utterance through `synth_synthesize` and return the same 28,800 samples.

### When it pays, and when it does not

Synthesis is 1.66 times faster with the split. Loading is about seven seconds
slower, because 457 MB of codec weights are copied into the accelerator's buffer
and the CUDA context is created.

For a one-shot 1.2-second utterance that is a net loss -- 17.9 seconds against
10.9 end to end including load. For a long utterance, or any process that loads
once and synthesizes repeatedly, it is a clear win. The split is therefore a
deployment choice rather than a default, which is what the backend request on
`synth_model_load` already expresses.

### Not done in this stage

The repeated-run cleanup the contract asks for is not written.

## Stage 8: Publication, Done

The model card is written and rendered from
`scripts/hf_cards/qwen3-tts-12hz-0-6b-customvoice.yaml`, with the digests checked
against the packages on disk by the generator. **The package was published on
2026-07-28** to
[`jiangzhuo9357/qwen3-tts-12hz-0-6b-customvoice-gguf`](https://huggingface.co/jiangzhuo9357/qwen3-tts-12hz-0-6b-customvoice-gguf),
on jiangzhuo's per-act confirmation, and last updated there on 2026-07-29
carrying BF16, F16 and Q8_MIXED. Every later edit to that repository is a fresh
outward act needing its own confirmation; see `OUTWARD_INTERACTION_POLICY.md`.

**This section read "Publication, Prepared" and "Nothing has been published"
until 2026-08-12**, two weeks after the upload, while Open Question 6 in this
same file already recorded "published 2026-07-28". The table and the list below
were written before the upload and are superseded by the card specification,
which carries the shipped digests and the third profile: they name two profiles
where three shipped, and their BF16 digest is from a pre-final cut.

| profile | size | sha256 (first 16) | CPU cosine | CUDA cosine |
| --- | --- | --- | --- | --- |
| BF16 (source) | 2168.8 MB | `aa96f152a113e199` | 0.999427 | not measured |
| F16 | 2168.9 MB | `db19d6317d156ad3` | 0.999427 | 0.999404 |

What the card declares and why:

- **`port_validated`, quality evaluation not run.** No arena or listening
  evidence supports a quality claim for this family and the project has no
  capability to produce one, which the intake accepted as a risk.
- **English only.** The checkpoint carries codec language tokens for nine more
  languages and two of its speakers pin a Chinese dialect; those paths load and
  run, but none has its own validation cases, so none is advertised.
- **Three profiles.** Q8_MIXED shipped 2026-07-29; see above.
- **The encoder half is absent**, and the card says so rather than leaving a
  reader to wonder why a speech tokenizer package cannot tokenize speech.

### What a publication would still need

- A decision on whether to publish at all, which is jiangzhuo's.
- The CUDA column for the source profile, which was never measured -- the
  accelerator work was done under F16.
- The repeated-run cleanup stage 7 owes.

## Stage 2: Base Package, Plan 1

The Reference Model Variant Ladder's second rung
(`qwen3-tts-12hz-0.6b-base`, `SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO`) has its
first plan done: intake, oracle, conversion, loading, and public-seam
validation. This section records what Stage 2 Plan 1 measured and shipped.
It does not add a rung to the ladder's completion gate on its own -- Plan 2
is what actually makes the family clone a voice.

### Pinned revision and digests

`Qwen/Qwen3-TTS-12Hz-0.6B-Base` at revision
`5d83992436eae1d760afd27aff78a71d676296fc`, the same Apache-2.0 basis as
Stage 1. 13 files, 2,516,106,051 bytes total, recorded in
`reports/porting/qwen3-tts/qwen3-tts-12hz-0-6b-base/intake.json`. The three
digests that matter downstream -- they are exactly what
`synthesize.profile.compatibility_id` hashes together, per Task 6 -- are:

| artifact | sha256 |
| --- | --- |
| `model.safetensors` (talker + speaker encoder) | `180b3b10eb1c9f1b4db7806d5475bae3071c0243c299d49926bab1da3b6946f6` |
| `speech_tokenizer/model.safetensors` (codec, both halves) | `836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258` |
| `config.json` | `2e714c787c8edb98b05432685cddb634add2de4d4e645f653d68251ef72ba011` |

`compatibility_id = sha256("qwen3-tts-voice-clone/1" + talker + codec +
config)` resolves to
`34d4de22a329b6bc8347cb952b6fa16513320012628598ab59743679cc16806e`, read
back from the shipped GGUF and independently recomputed from these three
digests by two separate reviewers (Tasks 6 and 9).

### Tensor census and the emitted count

Raw safetensors tensors: 402 talker + 76 speaker_encoder (`model.safetensors`)
and 225 encoder + 271 decoder (`speech_tokenizer/model.safetensors`) = 974.
The converter emits **894**:

```
402 (talker) + 76 (speaker_encoder) + 271 (codec decoder) + 225 (codec encoder)
  = 974 raw safetensors tensors
- 48 EMA-accumulator pairs collapsed into codebooks (16 decoder-layer +
    32 encoder-layer pairs, each 2 raw tensors -> 1 reconstructed codebook)
- 32 encoder `.initialized` shape-(1,) flags (not weights; the decoder half
    carries none of these)
= 894 emitted tensors
```

Split by GGUF type: BF16 478 = talker (402) + speaker_encoder (76), carried
unchanged. F32 416 = the codec's contribution: 271 + 225 − 48 = 448 tensors
survive reconstruction, 448 − 32 = 416 are emitted. 478 + 416 = 894, matching
the tensor count the loader's catalog independently derives from the
package's own hyperparameters (Task 8) and the count a reviewer got by
reimplementing the catalog in Python and diffing name-and-shape against all
894 tensors (0 missing, 0 stray, 0 shape mismatches).

**The codec encoder's own share, added 2026-08-14 because it appeared nowhere
in this file and a graph is built from it.** The census above is
whole-package and correct as it stands; what it never breaks out is the
encoder half on its own:

```
225 raw encoder safetensors tensors
- 32 encoder EMA-accumulator pairs collapsed into codebooks (2 raw -> 1 each)
- 32 encoder `.initialized` shape-(1,) flags
= 161 emitted encoder tensors
```

161 = downsample 1 + `encoder.layers` 28 (stem 2, four stages of 6, tail 2) +
transformer 96 (8 layers × 12) + semantic quantizer 3 + acoustic quantizer 33.
`expected_tensor_count` (`src/arch/qwen3-tts/catalog.cpp`) derives exactly that
arithmetic from the package's own hyperparameters, and
`tests/qwen3_tts_catalog_test.cpp` pins `base − custom_voice == 76 + 161` — the
speaker encoder and the codec encoder being the whole of what Base carries and
CustomVoice does not.

**The design's §1.1 row saying 225 is the RAW count, and the graph is built
from the 161** (`2026-08-11-qwen3-tts-stage-2-design.md:66`, "Codec encoder
half | 225 tensors"). That row reads off the safetensors header and is right
about the checkpoint; it is not the number a reader should expect to resolve
by name out of the GGUF, because the EMA pairs and the `.initialized` flags do
not survive conversion. Plan 3's `resolve_codec_encoder` resolves 161.

### The dedup measurement, and the decision not to act on it

`measure_shared_codebooks` (Task 5) checked every one of the encoder's
reconstructed RVQ codebooks against the decoder's own codebooks with a real
`torch.equal` over the actual checkpoint data, not a name-based shortcut.
Result: **16 of 16** index-aligned pairs that exist are bit-identical --
the encoder's `semantic_residual_vector_quantizer` layer 0 and
`acoustic_residual_vector_quantizer` layers 0-14 each match the decoder's
`rvq_first`/`rvq_rest` counterpart exactly. (An earlier version of this
measurement mis-described this as "16 of 32 diverge"; the encoder's
remaining 16 acoustic layers, 15-30, have no decoder-side counterpart at
all to compare against -- the decoder declares only 16 quantizer layers
total -- so there was nothing for them to diverge from.) Four additional
quantizer input/output projection tensors were separately confirmed
identical the same way.

The first implementation stored the 16 matched codebooks once and recorded
the alias in `Conversion.deduplicated` -- a field that lands only in a
gitignored convert report under `reports/convert/`, never in the package's
own metadata. A controller ruling reversed this: **carry both halves in
full, alias nothing.** The failure mode the reviewer named was "the alias
nobody can look up" -- a consumer walking the encoder's RVQ layers would
find some indices present and others silently absent, with no way to
discover, from the package alone, that the missing ones live under
`codec.decoder.quantizer.*`. The fix costs roughly 35 MB (~1.4% of the
2.52 GB package) in exchange for every tensor name being resolvable without
a side channel. The measurement itself -- 16 of 16 matched, not 16 of 32
diverged -- is kept as a recorded fact; only the storage decision changed.

### Measured reference-duration bounds

`scripts/dump_reference_qwen3_tts_base.py --trim-seconds N` against the
upstream `clone.wav` (8.08 s / 193,920 samples native; 10 s and 30 s loop the
clip rather than sourcing longer audio), same target text, language, and
seed 0 throughout. Full results, including per-duration `peak_abs`/`rms`/
distinct-code-fraction signals and an `intelligibility_basis` string for
each, are in `intake.json`'s `reference_bounds` key.

| requested | applied samples | looped | reference frames | generated frames | generated duration |
| ---: | ---: | --- | ---: | ---: | ---: |
| 0.5 s | 12,000 | no | 7 | 136 | 10.88 s |
| 1 s | 24,000 | no | 13 | 127 | 10.16 s |
| 3 s | 72,000 | no | 38 | 111 | 8.88 s |
| baseline (8.08 s, full clip) | 193,920 | no | 101 | 45 | 3.6 s |
| 10 s | 240,000 | 2x | 125 | 42 | 3.36 s |
| **30 s** | 720,000 | 4x | 375 | **9** | **0.72 s** |

The 0.5 s/1 s/3 s points each produced 2.5x-3.0x the 45-frame baseline for
the identical text and seed. **The 30 s point -- the plan's provisional
upper bound -- produced only 9 generated frames (0.72 s) for an 11-word
sentence that takes ~45 frames at the reference clip's native length**,
consistent with premature termination rather than with this family's
documented degeneracy signature (near-silence or one code dominating the
output; neither fires at either edge). The 10 s point is the best-behaved
of the five, landing within 3 frames of baseline.

#### The 9-frame anomaly, adjudicated 2026-08-14

Carried unresolved through Plans 1 and 2; Plan 2 could not settle it because
its regeneration ran in x-vector mode, which is not evidence about ICL. Plan 3
has ICL, so the case was driven again and the surrounding space measured with
`scripts/measure_qwen3_tts_icl_reference_length.py` (29 runs: 14 arm A, 5 arm B,
10 arm C, in two invocations and therefore two model loads -- the five seeds at
375 frames, which is the claim that matters, do share one load,
target text / clip / language / sampling / `max_new_tokens` all fixed; only
`trim_seconds`, the reference transcript's repeat count, and the seed vary).

**WHICH SIDE WAS MEASURED, stated first because everything below depends on it.**
Every number in this section comes from the **PyTorch reference implementation**,
driven through the committed oracle dumper. **This port was not compared on this
case.** That is a narrower claim than the plan's Step 3 asked for and a stronger
one than "the port has a bug": what is established is that the collapse and the
runaway are properties of the *reference implementation* under an incoherent
reference, so there is no port defect to hunt here. Of the plan's three outcomes,
only the third ("the oracle no longer reproduces it") is settled -- it is ruled
out. Outcomes 1 and 2, which ask whether the *port* reproduces 9 frames, remain
undistinguished; see "What Step 3 still needs" below.

**It reproduces exactly.** `base-ref-max` gives 9 generated frames, 0.72 s,
`peak_abs` 0.474609375, `rms` 0.0686, 8 distinct semantic codes, 375 reference
frames -- every figure Plan 1 recorded, to the digit. Every one of the six rows
in the table above reproduces to the frame. So the third outcome (the oracle no
longer reproduces it) is ruled out, and this is not measurement drift.

**But it is NOT the end of a monotone trend, and that hypothesis is refuted
rather than merely unsupported.** Three Golden points (13 -> 127, 101 -> 45,
375 -> 9) suggested output length is a decreasing function of reference length.
Densifying to eight points shows a genuinely monotone relation **only while the
clip is not looped**:

| reference frames | 7 | 13 | 25 | 38 | 50 | 75 | 100 | 101 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| generated frames | 136 | 127 | 122 | 111 | 90 | 67 | 51 | 45 |

That span (136 -> 45) is far larger than the seed spread at a fixed point
(45/53/48/57/55 over five seeds at 101 frames), so the trend is real. Past the
clip's own 8.08 s it collapses: 125 -> 42, ~150 -> ran away to the ceiling,
200 -> 43, 250 -> 100, 300 -> 43, 375 -> 9. Not a curve.

**Because past 8.08 s "a longer reference" is only reachable by LOOPING the
clip**, and the harness keeps the single-repetition transcript, so the reference
audio says the sentence up to four times while its transcript says it once.
That is a transcript-audio mismatch manufactured by the bound itself.

**At 375 frames the output is not a function of anything -- it is unstable.**
Five seeds at the identical input: 9, 8, 12, 4, and one run to the 2048-frame
ceiling.

**The matched-transcript control, published whole, including where it does
nothing.** Repeating the reference transcript to match the loop count, seed 0:

| reference frames | 125 | 200 | 250 | 300 | 375 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| transcript x1 | 42 | 43 | 100 | 43 | **9** |
| transcript matched | 112 | 44 | 102 | 40 | **66** |

It recovers strongly at two points (125 and 375) and changes **nothing** at the
other three. So "matching the transcript fixes it" is not what the data says,
and the honest reading is narrower: an incoherent reference is a condition under
which the stopping decision *may* go wrong, not a dial that sets output length.

**This control is n=1 per point, and that limitation bites hardest exactly where
it is quoted most.** The phenomenon being controlled for is instability, so a
single matched draw (66) against a single unmatched draw (9) rests entirely on
66 sitting outside the unmatched spread {4, 8, 9, 12} -- which it does, but one
draw cannot show the matched condition is *stable*. A seed scatter on the matched
arm is the missing measurement; it was not run.

**So the anomaly reproduces, with a mechanism that is not the trend**: an
incoherent reference, not reference length. The structural hypothesis offered
by the plan -- that the pad arm buries the target text -- is neither confirmed
nor needed here; the mismatch control moves the number without changing the arm,
so it is not recorded as the cause.

**Two symptoms CO-OCCUR under one input; the shared mechanism remains a
hypothesis.** Task 11's review found a real 8 s clip with a deliberately wrong
transcript running to the ceiling, and hypothesised it and this collapse were
one ICL length pathology. What the seed scatter above adds is co-occurrence, not
that mechanism: *both* symptoms arise from a *single* input configuration, four
collapses and one runaway across five seeds, differing only in seed. That is one
input, not a traced causal path -- nothing was instrumented, no intervention
isolated a cause -- so "one pathology" stays a hypothesis and is labelled as one
below. An incoherent reference makes the stopping decision unreliable in both
directions.

#### What Step 3 still needs, and what the port half now does say

The plan's Step 3 asks for a comparison **on the port**, and its outcomes 1 and 2
turn on whether the port also renders 9 frames. That comparison has not been
made, and the reason is worth stating precisely rather than as "not done":

**What was driven on the port.** The port's codec encoder now runs on
`base-ref-max`'s reference audio -- 720,000 samples, 375 frames -- and agrees
with upstream float32 on **100.000% of code decisions (0 of 6000)**, with the
continuous chain at `rel_absmax` 8.431e-05, *tighter* than base-icl-en's
9.303e-05. So the deterministic half of the port handles this input correctly,
and whatever the 9 frames are, they are not a codec-encoder defect. (That run
also surfaced a gate-scoping problem unrelated to the anomaly; see
`tests/tolerances/qwen3-tts.json`'s `gate_scope_warning`.)

**What cannot be answered this way.** The frame count is decided by the talker's
sampled EOS, and the port samples with its own RNG rather than PyTorch's, so
"does the port emit 9" has no single deterministic answer to compare -- the
oracle itself emits 9, 8, 12, 4 or 2047 depending only on the seed. A meaningful
port comparison is therefore *distributional*: drive the port's own synthesis at
several seeds on this input and ask whether its frame counts scatter the same
way. What that needs is a way to hand the port the looped 30 s reference --
either `trim_seconds` in the x-vector/synthesis drivers, or the looped waveform
materialized once as a WAV. **The same missing `trim_seconds` also blocks
widening the `replay` stage** (see that stage's `cases_not_extended`), so one
driver flag closes both holes; it belongs in the carryover, not only here.

Until then: the pathology is established **in the reference implementation**, and
the port is neither implicated nor cleared on it.

#### The alignment arms, measured across all ten ICL cases (2026-08-14)

`generate_icl_prompt` has two arms and the Golden suite covers both. The arm is
read out of the return statement that executed, never recomputed from `T1` and
`T2`: a block of `T2` positions is consistent with either arm, so inference from
lengths is not sound. `prompt/alignment.json` cross-checks each reading three
independent ways (return line, trailing-object identity, `text_embed` length)
and fails if they disagree.

| case | T1 | T2 | arm | ref frames | trailing |
| --- | ---: | ---: | --- | ---: | ---: |
| `base-upstream-clone-en` | 78 | 102 | pad | 101 | 1 |
| `base-icl-en` | 46 | 102 | pad | 101 | 1 |
| `base-icl-zh` | 45 | 102 | pad | 101 | 1 |
| `base-icl-ja` | 35 | 102 | pad | 101 | 1 |
| `base-ref-min` | 46 | 14 | **truncate** | 13 | 32 |
| `base-ref-max` | 46 | 376 | pad | 375 | 1 |
| `base-text-short` | 33 | 102 | pad | 101 | 1 |
| `base-text-long` | 980 | 102 | **truncate** | 101 | 878 |
| `base-seed-one` | 46 | 102 | pad | 101 | 1 |
| `base-seed-forty-two` | 46 | 102 | pad | 101 | 1 |

The two x-vector-only cases have no row: that mode never calls
`generate_icl_prompt`, so it has no arm rather than an unmeasured one.

**The truncate arm is reached two different ways and only one was anticipated.**
`base-ref-min` gets there by making `T2` small (a 1 s reference, 13 frames);
`base-text-long` gets there by making `T1` large (a 4,749-character target text,
`T1`=980). They exercise one branch from opposite sides, and `base-text-long` is
the only case whose trailing schedule carries substantial content -- 878
positions against every other case's 1 or 32.

**This is also what rules the arm out as the 9-frame cause.** `base-ref-max` is
in the pad arm, which is the shape the plan's structural hypothesis pointed at.
But the matched-transcript control holds the arm and `T2` fixed at pad/376 --
repeating the transcript lengthens `T1` to ~136, still well under 376 -- and the
output moves from 9 frames to 66. Same arm, same block length, different result,
so the arm is not what is driving it.

**Two runaways that are not the same thing**, distinguished by the code
diversity of the tail rather than by frame count. The pathological runaway holds
~6 distinct semantic codes across its last 400 frames -- a degenerate loop. The
`base-text-long` Golden case also reaches 2047 frames, but holds 233 distinct
codes there: it is still speaking, because its 4,749-character text needs
roughly 4,000 frames against a 2,048 budget. That one is arithmetic, and the
case as specified cannot terminate; it is recorded as a known non-terminating
input rather than treated as this defect.

**The manifest was corrected to match, on 2026-08-14.** `base-text-long`
declared `expected.status: "ok"` and required `audio.pcm`, `result.json` and
`metadata.json` -- artifacts it cannot produce, which is why it is the only case
directory without a `result.json`. A contract that describes a known-failing
case as passing is worse than no entry at all, so the declared status is now
`non_terminating_at_max_new_tokens` and the artifact list is the five it does
deterministically produce (the speaker and prompt halves and the drawn codes).
**The case stays deliberately**: it is one of only two that reach the truncate
alignment arm and the only one that reaches it by making `T1` large, so deleting
it would drop truncate coverage to a single case.

`base-ref-max` carries a `upstream-unstable-stopping` coverage tag for the same
reason -- a reader of the manifest alone would not otherwise expect 0.72 s of
audio from an 11-word sentence, and would have no way to know the 9 is a
property of the reference implementation under an incoherent reference rather
than a target the port must reproduce.

**No listening pass had happened when this table was measured.** `intake.json`
records `perceptual_evaluation_performed: false` for every point, and its
`recommendation` states plainly that the objective signals checked here
cannot substitute for one: they rule out the family's *known* hard-failure
pattern at both edges, but neither confirm nor deny that the resulting audio
is usable speech, particularly at 30 s. Task 6 shipped
`min_frames_per_clip=24000` / `max_frames_per_clip=720000` anyway, as safety
ceilings pulled from the plan rather than as perceptually validated bounds,
and every later task through Plan 2 carried that distinction forward rather
than quietly upgrading it.

**Both open points closed on 2026-08-13, and the table is left as measured.**
The Listening Audit's labelled duration sweep found 1 s, 3 s, 10 s and 30 s
references all **usable**, and the 0.5 s case **refused by the library**
(`voice_profile.reference_too_short`) -- so the shipped bounds produce usable
speech across their whole declared range, and the sub-minimum case fails
closed rather than degrading. The 30 s row above did **not** reproduce:
regenerating that case in x-vector mode gave 46 codec frames, in line with
every other duration in the audit's sweep, and the 9-frame figure came from a
transcript-assisted dump -- a mode this port does not implement, which is why
nothing in the shipped x-vector path could reach it. The anomaly is real and
belongs to Plan 3, where the ICL path is built and can be adjudicated against
its own renders. The table stays as the intake script measured it, because the
row is evidence about the dump, not about the shipped path. See "Listening
Audits" below.

**A SECOND ICL OUTPUT-LENGTH PATHOLOGY, 2026-08-14, AND IT IS NOT DIAGNOSED
EITHER.** Plan 3's Task 11 landed ICL synthesis, so the mode can now be driven
end to end; the first thing it produced was the opposite symptom. Through the
public seam, on the real 8.08 s `clone.wav` at its native length, with the
target text `"Hi."` and seed 7 on CPU against the BF16 Base package, at
`CMAKE_BUILD_TYPE=Release` -- **the build must be named, because Task 13 later
measured that the generated length depends on it** (Release 24,960 PCM frames
against RelWithDebInfo's 48,000 from identical input, seed and backend; see
"Generated length is build-dependent" under Plan 3 below):

| reference transcript | outcome |
| --- | --- |
| the clip's own (`"Okay. Yeah. I resent you. …"`) | `SYNTH_OK`, 13 codec frames = 24,960 PCM = **1.04 s of audio**, peak 0.681, 5.5 s elapsed |
| `"hello there"` — same audio | **never stops**: runs the full `kDefaultMaxFrames = 2048`, returns `SYNTH_ERR_OUTPUT_LIMIT`, zero audio, 475.9 s of CPU |

The variable is **transcript–audio mismatch**, not clip length and not
synthetic input — an earlier revision of Task 11's own test comment blamed a
synthetic tone, and the review falsified that by holding the real clip fixed
and changing only the transcript. It is an ordinary caller mistake: any
imperfect ASR transcript is one.

**Two symptoms, one suspected mechanism, neither diagnosed.** The 9-frame row
above under-produces (9 frames for an 11-word sentence, from a
transcript-assisted dump over a 4×-looped 30 s reference); this one never
produces a stop code at all. Both are ICL-mode output lengths uncorrelated
with the target text, and Stage 1 recorded a third member of the shape —
greedy decoding running away to 8191 frames, "the model never emitting its
stop code". Whether they share a cause is a HYPOTHESIS, not a finding: nothing
has been instrumented, and it may simply be upstream's behaviour under a
mismatched prompt. What is established is that the ICL path can burn the
default ceiling and return nothing on a realistic input.

**What exists today is mitigation, not a fix.** `src/synthesize.cpp`'s limit
stop now emits `synthesis.output_limit` with a message naming the reference
transcript when the request carried one, so a caller gets something to act on
instead of a bare status; and `tests/qwen3_tts_base_load_real.cpp` caps its
own ICL runs so the suite does not spend eight minutes reaching that state.
Neither shortens the run or explains it. Adjudicating this — together with the
9-frame row — is Plan 3's carry-over.

### What Plan 1 did not deliver

**Every bullet in this section is Plan 1's end state, in the past tense, and
Plan 2 closed all of it — see "Stage 2: Base Package, Plan 2" below.** The
section is kept because the capability correction it records is the reason
this file is trusted about the difference between a package's declaration and
a runtime's capability; it is not a description of the tree today.

- **No graphs.** The ECAPA-TDNN speaker encoder and the mel front end that
  `docs/voice-conditioning.md` specifies for this stage were not implemented,
  and this family reached none of the Audio Normalizer (including vendored
  libsamplerate 0.2.2), which already existed for OmniVoice. The
  speaker-encoder tensors and their metadata were catalogued (Task 8) and
  loaded, but nothing ran a forward pass over them. *Plan 2 wrote both graphs
  and routed this family through the existing Audio Normalizer.*
- **No Voice Profile preparation, and therefore no advertised source.**
  `synth_voice_profile_create_from_reference` returned
  `SYNTH_ERR_UNSUPPORTED_VOICE` for this family -- `src/voice-profile.cpp`
  dispatched that call for OmniVoice alone -- so
  `synth_model_get_voice_profile_capabilities` reported zero source flags for
  both variants, and zero in every field describing a source. *Plan 2 added
  the `ModelFamily::Qwen3Tts` arm and Base now reports source flags 9;
  CustomVoice still reports zero.*

  This is a correction, made 2026-08-12 on the branch's final review. Plan 1
  shipped the snapshot advertising `SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO |
  SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE`, on the argument that the gap was
  a dispatch gap rather than a capability lie. `docs/c-interface.md` does not
  leave that open: "A Model without runtime Voice Profile support reports zero
  flags", and at Plan 1 nothing in the runtime could create or consume such a
  Profile. The confirmed contract wins over the argument.

  What did NOT change: the Base package still declares its full Voice Profile
  contract, and `src/arch/qwen3-tts/weights.cpp` still reads and validates
  every `synthesize.profile.*` and `synthesize.reference.*` key at load time,
  refusing a package that declares them badly. The package declaring a
  contract and the runtime advertising a capability are different statements;
  only the second was false. See the ladder section above for what Plan 2
  advertises and why Serialized Profile arrives with Reference Audio rather
  than separately.
- **The reference-duration bounds were unaudited at the edges**, per the
  measurement above. *Closed by the 2026-08-13 Listening Audit, which found
  1 s, 3 s, 10 s and 30 s usable and 0.5 s refused; same section.*

### The real Base package through the public C interface

**This section records `tests/qwen3_tts_base_load_real.cpp` as Plan 1 left it,
in the past tense, and Plan 2 changed two of the assertions described below —
see "Stage 2: Base Package, Plan 2".** It is kept because the reasoning for
adding a model-guarded integration test at all, and the seam it was written to
cover, are still the reasons that file exists; the *current* assertions are in
the file itself, and the two that moved are marked inline. As with the section
above, this is not a description of the tree today.

Task 9's review found a real gap: deleting the one-line copy
`shared.voice_profile = info.voice_profile;` in
`src/synthesize.cpp::shared_info(qwen3tts::ModelInfo, ...)` left the entire
`synthesize-check-unit` gate green (89/91, unchanged), because nothing in
the unit tier loads a real package through the public C interface far
enough to notice. This task added `tests/qwen3_tts_base_load_real.cpp`, a
model-guarded integration test (`SYNTH_QWEN3_TTS_BASE_TEST_MODEL`,
registered only when `SYNTH_BUILD_INTEGRATION_TESTS=ON` and the package
exists on disk) that loads the real Base GGUF through
`synth_model_load` and asserts, through the public interface only:

- `synth_model_get_preset_voice_count` returns 0.
- `synth_model_get_voice_profile_capabilities` reported `source_flags`
  exactly zero, and with it zero (or `SYNTH_REQUIREMENT_UNSUPPORTED`) in
  every field describing a source, down to a null `profile_schema` and 32
  zero compatibility-id bytes. This assertion was inverted once already on
  2026-08-12, from the two-bit advertisement Plan 1 originally shipped; see
  "What Plan 1 did not deliver" above. *Plan 2 inverted it back, because the
  capability it describes now exists: `check_capabilities` asserts
  `SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO |
  SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE` exactly, the six real reference
  limits the package declares, `profile_schema` equal to
  `"qwen3-tts-voice-clone"` at version 1, and a nonzero compatibility id.
  `SYNTH_REQUIREMENT_UNSUPPORTED` survives on the three requirement fields
  alone.*
- A `synth_synthesize_to_buffer` request naming no Voice at all, and a
  second one naming a `voice_id` that could never exist in an empty
  catalog, both fail with `SYNTH_ERR_UNSUPPORTED_VOICE` -- the one voice
  error `docs/c-interface.md` defines, reached by two different code paths
  (the generic core's preset lookup for the named case, this family's own
  empty-catalog lookup in `Model::resolve_voice` for the unnamed one). The
  unnamed case additionally asserts the diagnostic code
  `synthesis.voice_unsupported`: the ABI has exactly one voice-error status,
  so the diagnostic is the only thing that tells a caller a Voice refusal
  from a codec that failed to run, and both used to arrive as
  `synthesis.graph_failed`.

**A cost of the 2026-08-12 correction, stated rather than left implicit —
and paid off by Plan 2.** While the capability snapshot was all-zero, this
test did not cover the seam line it was written for: an all-zero copy of an
all-zero struct is unobservable, so deleting
`shared.voice_profile = info.voice_profile;` left the integration test green
too. The line stayed because Plan 2 was expected to make it carry something,
and the coverage would return with the first nonzero field. What the test
still proved at the ABI meanwhile was the empty Preset Voice Catalog and both
refusal paths. *Plan 2 delivered the nonzero fields, so this cost is settled:
`tests/qwen3_tts_base_load_real.cpp`'s own header now records that deleting
that line fails this test rather than passing it silently.*

Building and running this test needs a build directory configured with
`-DSYNTH_BUILD_INTEGRATION_TESTS=ON` and the real Base GGUF on disk; it does
not run against the standard `build/` unit-gate configuration and is not part
of `synthesize-check-unit`.

## Stage 2: Base Package, Plan 2

Plan 2 is the ladder's second-rung completion gate: **reference audio in,
cloned audio out, on CPU, in x-vector mode — a usable clone capability
exists.** `tests/qwen3_tts_clone_real.cpp` closes it against the real Base
package and its pinned reference clip
(`models/qwen3-tts-reference-audio/clone.wav`, 8.08 s / 193,920 samples native
at 24 kHz mono), passing at commit `ecf8203` and this task's own commit on
top of it. Six assertions, the last of which is the one that actually
distinguishes "cloned" from "synthesized": a Profile prepared from the real
clip matches the oracle x-vector above the committed tolerance; synthesis
with it produces finite, non-silent 24 kHz PCM; the same Profile and seed
reproduce that PCM bit-for-bit; **two different reference inputs produce
different PCM** — a Profile the graph silently ignored would pass every
other assertion here too; a Serialized Profile round-trips (serialize, free
the original, load, synthesize) to the same PCM; and a Profile presented to
a second Loaded Model refuses with `SYNTH_ERR_UNSUPPORTED_VOICE` and writes
no audio.

The second "reference input" is **not** a second recording. Only one real
clip is materialized in this tree, so that assertion pairs `clone.wav` with
a synthesized 440 Hz sine at the package's minimum reference length
(`tests/qwen3_tts_clone_real.cpp`'s `make_tone`, which its own header states
and this paragraph did not until 2026-08-12). That is sufficient for what the
assertion decides — two different inputs must not produce identical output —
and it is **not** evidence that either clone resembles its source. The
resemblance evidence that exists is one listener's, from the 2026-08-13
Listening Audit recorded below, not this test's.

A deliberate break of the x-vector substitution (a fixed dummy vector in
place of the per-Profile one) was confirmed to fail exactly the "two
different inputs" assertion and nothing else, then restored; a deliberate
**tightening** of the oracle threshold (to 0.999999999999, above the
observed cosine) was confirmed to fail the x-vector assertion, then
restored. This paragraph called that a "loosening" until 2026-08-12, which
is backwards: loosening a `min_cosine` gate cannot make it fail.

### The mel front end's six conventions, read off upstream

`docs/voice-conditioning.md`'s mel front end needs six conventions beyond the
six numbers the package itself declares (`mel_bins` 128, `n_fft` 1024,
`hop_length` 256, `win_length` 1024, `fmin` 0, `fmax` 12000); each silently
changes the answer, so each is read off the pinned upstream source
(`QwenLM/Qwen3-TTS` at `022e286b98fbec7e1e916cb940cdf532cd9f488e`,
`qwen_tts/core/models/modeling_qwen3_tts.py`) rather than assumed, and
recorded verbatim in `conventions.json` (Task 1):

| convention | value | upstream |
| --- | --- | --- |
| mel scale | Slaney (not HTK) | `:435-437` — `librosa.filters.mel(...)` called with no `htk` kwarg; librosa 0.11.0 defaults `htk=False` |
| filterbank normalization | Slaney area (not peak, not unnormalized) | `:435-437`, same call, no `norm` kwarg; librosa 0.11.0 defaults `norm='slaney'` |
| spectrum magnitude | amplitude `\|X\|` with epsilon `1e-9` inside the square root (not power `\|X\|^2`) | `:459` |
| log compression | natural log, floor `1e-5`, `C=1` | `:396-397,462` |
| window | Hann, `periodic=True` (torch's default; never passed explicitly) | `:440` |
| padding / centering | **not** `center=True` STFT and **not** uncentered: a manual reflect pad of `(n_fft - hop_length) // 2` samples per side, then `torch.stft(center=False)` on the already-padded signal | `:442-445`, `:408`, `:453` |

The padding rule is the one a reader would most likely guess wrong: for this
package's `hop_length=256` the manual pad is 384 samples per side, not the
512 a naive `center=True` assumption would use. `conventions.json` also
records `min_pcm_samples = 385` (`(n_fft - hop_length) // 2 + 1`, the point
below which upstream's own reflect pad fails outright — observed directly by
padding a synthetic input, not derived from the formula alone) and the two
one-second frame counts (93 at `hop_length=256`, 46 at `hop_length=512`) used
to pin the port's own frame-count fixed points.

### ECAPA-TDNN graph and mel front end, measured against upstream

The GGML graph (`src/arch/qwen3-tts/speaker-encoder.h`/`.cpp`, Task 4) builds
**435 nodes with F32 weights and 473 with BF16**, and reproduces
`Qwen3TTSSpeakerEncoder`'s own forward pass to **7.15e-07** max-abs
difference, on synthetic weights at a small configuration — not against the
real 76-tensor checkpoint, which the residual below measures separately.

The node count is not a property of the topology alone, which is how this
document carried only 435 until 2026-08-12. The res2net scale and the block
count fix most of it, but the weights' own storage dtype fixes the rest:
`add_channel_bias` inserts a `ggml_cast` only when the bias is not already
F32, and the graph has 38 convolutions, so a BF16 package builds exactly 38
more nodes. **473 is the graph that actually runs** — every one of the real
Base package's 76 `speaker_encoder` tensors is BF16, measured with a GGUF
read. 435 is what `tests/qwen3_tts_speaker_encoder_test.cpp`'s own synthetic
F32 fixture builds; that test now pins both counts exactly and asserts the
38-node gap, and its ceiling was raised from 450 (which the real graph
exceeds) to 480.

The 7.15e-07 is against **upstream's own `Qwen3TTSSpeakerEncoder`**, run
directly — not against a re-transcription, which is the weaker claim this
paragraph made until 2026-08-12. A from-scratch numpy transcription of the
same topology was a separate and looser check, agreeing to 1.2e-6
(`src/arch/qwen3-tts/speaker-encoder.cpp`'s own header records it as such).

The topology facts a reader cannot get from the six declared
numbers alone (Res2Net dilations `[1,2,3,4,1]`, `reflect`-padded "same"
convolutions, the pass-through-first accumulation rule, the squeeze-excite
statistic and activations, the attention input's 4608-channel concatenation)
are in `conventions.json`'s `ecapa_topology` block, each with its own
upstream line citation.

The mel front end (`src/arch/qwen3-tts/mel.h`/`.cpp`, Task 2) is
bit-exact-able in principle — it runs entirely in float32 on the CPU upstream,
never touching bfloat16 — and measures **max_abs 3.31e-4, mean_abs 5.26e-7,
cosine 1.000000** against the oracle's own `mel.f32`. Recomputing the
reference FFT/filterbank independently in float64 puts the port's own mel
*closer* to that float64 truth (1.14e-4) than the oracle's is (2.21e-4): the
residual is ordinary cross-implementation floating-point rounding between two
different FFT/DFT implementations, not a defect in either the port or the
mel stage's contribution to the x-vector's own residual (see below).

### The x-vector's measured residual

`tests/tolerances/qwen3-tts.json`,
`variants.qwen3-tts-12hz-0-6b-base.profiles.BF16.stages.replay.probes.speaker.x_vector`
(Task 6, measured 2026-08-12 on CPU, `build/` at `CMAKE_BUILD_TYPE=Release`,
over cases `base-xvector-en` and `base-xvector-zh` — both the same reference
clip with no trim, byte-identical `speaker/*.f32` outputs between them, so
one measurement taken twice rather than two independent points):

```
min_cosine:          0.9999767764206815   (gate: 1 - 5 * (1 - observed))
observed_min_cosine: 0.9999953552841363
observed_max_abs:    0.021114349365234375
```

The residual is bfloat16 quantization of the **oracle's own** tensors, not a
port defect. Seven of the eight dumped `speaker/*.f32` artifacts (everything
but `mel.f32`) round-trip through bfloat16 exactly (max abs diff 0.0) — the
oracle's speaker encoder runs in `torch.bfloat16` throughout. At the
x-vector's own largest element (`|ref| = 7.28`), 0.02111 is 0.68 of one
bfloat16 ulp there, and **recomputing the oracle's own final ASP/FC layer
directly in float64 disagrees with the oracle's own recorded x-vector by
more** (0.022367) **than this port does.** `max_abs` is recorded but does not
gate: an x-vector is a direction consumed by a dot product against the
prompt, the same reasoning that leaves the talker probes' own `max_abs`
ungated elsewhere in this file.

### Profile Schema identity

Schema `qwen3-tts-voice-clone`, version 1 — the same string and version the
Base package's own `synthesize.profile.schema`/`schema_version` metadata
already declared and the loader already validated in Plan 1, now also the
identity a runtime-prepared Profile's v1 Serialized envelope carries
(`src/arch/qwen3-tts/profile.h`/`.cpp`, Tasks 7–8). One `kind` value ships
this plan: `"x-vector"`. Plan 3 adds `"icl"` for transcript-assisted cloning
**without a schema version bump** — the envelope discriminates on the
in-payload `kind` key rather than on `schema_version`, the same shape
`arch/omnivoice/profile.h`'s two kinds already use, so every Plan 2 profile
stays loadable under a Plan 3 build.

### The six family-conditioned points, and which four Plan 2 changed

Reference Audio support is decided at six places a family can hook into
across the shared core (`.superpowers/sdd/plan2-interfaces.md` §1) — five
dispatch arms in `src/voice-profile.cpp` plus one synthesis-time consumption
point in `src/synthesize.cpp`. Plan 2 changes four of the six:

| point | location | Plan 2 |
| --- | --- | --- |
| A — `synth_voice_profile_create_from_reference` | `voice-profile.cpp` | **Changed** (Task 9): a `Qwen3Tts` arm gated on `source_flags & REFERENCE_AUDIO`, refusing a transcript by name |
| B — `synth_voice_profile_create_from_description` | `voice-profile.cpp` | Untouched — Description Text is Stage 3's, not this rung's |
| C — `synth_voice_profile_create_random` | `voice-profile.cpp` | Untouched — no family, including OmniVoice, implements Random Seed |
| D — `synth_voice_profile_load_from_memory` | `voice-profile.cpp` | **Changed** (Task 9): a `Qwen3Tts` arm gated on `source_flags & SERIALIZED_PROFILE`, preserving the params-before-model validation order |
| E — `synth_voice_profile_serialize` | `voice-profile.cpp` | **Changed** (Task 9): branches on the **profile's** `family_tag == Qwen3TtsClone`, not the model's family |
| 6th — synthesis-time consumption | `synthesize.cpp`, the `ModelFamily::Qwen3Tts` arm inside `synth_synthesize` | **Changed** (Task 11): a blanket `SYNTH_ERR_UNSUPPORTED_VOICE` refusal replaced with a cross-model check, a `family_tag`/`CloneMode` check, and `family_request.x_vector = &clone.x_vector` |

Points A, D and E each additionally required `fill_voice_profile_capability`
(`src/arch/qwen3-tts/weights.cpp`, gated on `hparams.voice_mode ==
VoiceMode::ProfileSources`, not the inverse predicate `has_preset_voice_catalog`)
to publish real values, and Task 10 restructured `Model::resolve_voice` /
`Model::run_synthesis` to consult a Profile's x-vector instead of refusing
every Base request before one is reached — the family-internal half of the
same seam, not itself one of the six shared-core points.

### CustomVoice stayed byte-identical

None of Plan 2's changes touch the `PresetCatalog` (CustomVoice) path: every
dispatch point above additionally gates on this Loaded Model's own
`source_flags`/`family_tag`, not just `family == Qwen3Tts`, which is what a
CustomVoice Model shares with Base under one `ModelFamily` tag. Verified
directly for this task: `synthesize-cli --model
qwen3-tts-12hz-0-6b-customvoice-BF16.gguf --text "Hi." --voice aiden
--language en --seed 42` produces
`9ffbe5e5d41a7fc36cbb3f138660ee55cf13c4d3363e716ae944c0ab7cf1fddf` at this
task's own commit, identical to the digest value Tasks 10 and 11 each
recorded in their own reports for the same command.

### The Python binding needed no work — verified two ways, not assumed

The claim under test: the binding is entirely family-generic, so a second
family needs zero binding changes. Verified twice.

**Statically**:
`grep -rniE "omnivoice|qwen3|vits|kokoro|family" bindings/python/src/
bindings/python/CMakeLists.txt bindings/python/pyproject.toml` returns zero
matches.

**Dynamically**: built fresh Provider and API wheels (`uv build --wheel`,
the same two artifacts `tests/check-python-api-wheel.cmake` builds for its
own gate), installed both into a throwaway venv alongside `numpy`/`soundfile`,
and drove the real Base package through the installed extension —
`create_voice_profile_from_reference` from `clone.wav`, `synthesize_text`
with that Profile, `serialize()` — with **no change to any file under
`bindings/`**:

```
sources 9 transcript 0
frames 30720 finite True
serialized bytes 4832
```

`sources == 9` is `SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO (1) |
SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE (8)`; `transcript == 0` is
`SYNTH_REQUIREMENT_UNSUPPORTED` — both match the C capability snapshot
exactly, surfaced correctly through the extension. One correction to how
this check is written, for whoever runs it next: the binding's actual shape
is `model.voice_profile_capabilities` (a property, not a call),
`model.create_context().synthesize_text(text, voice_profile=..., language="en")`
(there is no top-level `model.synthesize(...)`, and the `language` keyword
takes a BCP-47 tag like `"en"`, not a spelled-out name like `"english"`) —
three shape details that do not match a natural first guess at the API, not
family-conditioning of any kind.

### The CLI's position, stated rather than left implied

`examples/cli/` is untouched by this plan, and stays untouched on purpose:
`grep -rn "voice_profile\|reference\|omnivoice\|family" examples/cli/`
returns zero matches, confirmed directly. The CLI has no audio reader and
never assigns `synth_request_t.voice_profile` for **any** Model Family,
including OmniVoice, which has had cloning capability for weeks. Adding one
is a cross-family slice — jiangzhuo's ruling, 2026-08-12 — not a qwen3-tts
increment; the spec's §8 row for this plan listing "CLI and binding Adapters"
overstated the CLI half of that clause, which does not survive contact with
the tree.

### What Plan 2 does not deliver

**This section is Plan 2's end state. The first bullet has since been closed by
Plan 3 and is kept in Plan 2's own tense; the rest still stand.** See "Stage 2:
Base Package, Plan 3" below.

- **ICL / transcript-assisted cloning.** Plan 3's, and **done there.** In the
  Plan 2 state `reference_transcript` and `reference_language` reported
  `SYNTH_REQUIREMENT_UNSUPPORTED`, a request carrying a transcript was refused
  by name (`voice_profile.transcript_unsupported`), the codec encoder graph was
  not written, `resolve_codec_encoder` discarded its pointers into a scratch
  struct deliberately, and ten of the Base manifest's twelve Golden cases were
  ICL and could not be driven end to end by anything in that plan. All of that
  changed in Plan 3: both fields are `SYNTH_REQUIREMENT_OPTIONAL`, the graph
  exists, the resolver keeps its 161 tensors, and the ICL path runs end to end.
- **Description Text.** Stage 3's, on `qwen3-tts-12hz-1.7b-voicedesign`. No
  `ModelFamily::Qwen3Tts` arm exists at dispatch point B above, and
  `SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT` stays unadvertised. Random Seed
  stays unadvertised too, by every family.
- **The CLI path.** See above.
- **Quantization or CUDA for the new graphs.** The speaker encoder runs on
  the CPU in F32/BF16 as the package ships it — Plan 1's own catalog note
  said the encoder half "stays on the CPU until a graph exists that reads
  it"; a graph now exists, and whether quantizing or accelerating it pays is
  a measured decision for Plan 4, not assumed from another family's
  precedent. No performance number is claimed by this plan.
- **The listening pass, which Plan 2 did not deliver and which has since
  run.** jiangzhuo scheduled it **last, after the plan work** (2026-08-12) —
  that is, in Plan 4's ship-prep phase, where the Stage 2 spec's own phase
  table (§8, phase 4) already places the listening audit. Not after Plan 2,
  which is what this branch's SDD ledger said until 2026-08-12; the ledger
  recorded "after Plan 2" because Plan 2 was the plan in flight when the
  decision was taken, not because Plan 2 was the boundary. It ran on
  2026-08-13 and recorded `no_obvious_regression`; see "Listening Audits"
  below for what it settled — the duration bounds and the 9-frame point —
  and for what it deliberately does not cover.
- **Publication and Quality Evaluation.** Unchanged: publication requires
  separate, per-submission confirmation; Quality Evaluation is deferred per
  ADR 0017.

## Stage 2: Base Package, Plan 3

Plan 3 is the ladder's third-rung completion gate: **reference audio AND its
transcript in, cloned audio out, on CPU, in transcript-assisted (ICL) mode.**
`tests/qwen3_tts_icl_real.cpp` closes it against the real Base package, its
pinned clip (`models/qwen3-tts-reference-audio/clone.wav`) and that clip's own
transcript, read out of the oracle's dump rather than retyped. Seven
assertions: an ICL Profile reports `CloneMode::Icl` and carries a `[16, T]`
reference code grid whose dequantized reconstruction meets the committed
codec gates; synthesis with it produces finite, non-silent 24 kHz PCM,
reproducibly; **the same clip with and without a transcript produces different
PCM** — the assertion that distinguishes "ICL" from merely "cloned"; the
Serialized Profile round-trips; a second Loaded Model refuses it; and a Plan 2
x-vector Profile still serializes to a Plan 2-shaped envelope and still
synthesizes. `reference_transcript` and `reference_language` moved from
`SYNTH_REQUIREMENT_UNSUPPORTED` to `SYNTH_REQUIREMENT_OPTIONAL` in the same
change that landed the synthesis path, so the capability snapshot never
advertised a mode the runtime refused.

**Read this section together with two others.** The alignment arms across all
ten ICL cases, and the 9-frame anomaly's adjudication, were measured by this
plan but are recorded above under "Measured reference-duration bounds",
because that is where the anomaly was raised. This section does not repeat
them; it says what they settled.

### A wrong alignment is fluent, and nothing in the end-to-end tier sees it

This is the finding that shapes everything below, and it is **measured on this
port rather than cited**. The two-track prompt aligns a text track against a
codec track; get the correspondence wrong by one frame and the model still
speaks. Injected into `talker-host.cpp`'s `append_icl_block` (reading frame
`index` where it must read `index - 1`) and run through the public C seam on
CPU at `CMAKE_BUILD_TYPE=Release`:

| observable | correct | one-frame codec rotation |
| --- | ---: | ---: |
| `synth_synthesize` status | `SYNTH_OK` | `SYNTH_OK` |
| PCM frames | 24,960 | 19,200 |
| audio duration | 1.0400 s | 0.8000 s |
| unit tests failing | — | **1**, measured at Task 11's commit over the 100 unit tests that existed then (`icl-prompt-test`) |
| `synthesize-qwen3-tts-icl-real` | passes | **passes** |
| `icl-prompt-real`, codec track | 2.232e-03 | **7.945e-01** (gate 2.0e-2) |

**What was measured is the table above and nothing more.** The misaligned run
returned `SYNTH_OK`, emitted 19,200 finite PCM frames at peak amplitude 0.7696,
and passed every differential the end-to-end test makes. **Nobody listened to
it.** No Listening Audit has run for ICL at all, so this document makes no claim
about whether the misaligned audio is fluent, intelligible, or in the reference
speaker's voice — earlier revisions of this paragraph said "plausible speech in
approximately the right voice", which no measurement in this branch supports.
The argument does not need it: a defect that every gate but the numerical ones
missed is the point, and "returned success with finite non-silent audio and a
23% shorter duration that nothing asserts" is what makes it. `frame_count`
is on the API surface, so a caller can see 0.80 s where 1.04 s is correct —
the accurate phrasing is **observable, never detected**: the evidence sits in
public view and nothing looks at it. That the end-to-end test passes is a
property of what an end-to-end differential can decide, not a defect in it:
its assertions 2–7 are differential (ICL against x-vector, a run against its
own repeat, a Profile against its own reload), and a rotation moves both arms
of every one of those comparisons equally. Its one oracle-anchored assertion
anchors on the Profile's code grid — waveform to codes — which is a different
stage. **Alignment is `tests/qwen3_tts_icl_prompt_real.cpp`'s, and only its.**
The numerical gates are the whole defence, which is why so much of this
section is about them.

**Generated length is build-dependent, and no frame count may be quoted
without naming its build.** Identical package, clip, transcript, target text,
seed and CPU backend, with the thread count swept over 1/2/4/8/20 and no
movement at all — but `Release` (-O3) gives 24,960 PCM frames where
`RelWithDebInfo` (-O2) gives 48,000, each reproducible within its own build.
The autoregressive stop decision is what moves; every deterministic quantity
here (the reconstruction percentiles, the code agreement rates) is
bit-identical across the two. So an exact frame count is a property of the
compiler's floating-point choices, and the test prints it rather than gating
on it. This is the record's performance-number rule applied to output
*content*.

### The codec encoder's third provenance, and the conventions it settles

The codec **encoder** — Plan 1 converted its 161 tensors, `resolve_codec_encoder`
discarded every pointer into a scratch struct until Plan 3's Task 3 made it keep
them, and Plan 3 then builds the graph
(`src/arch/qwen3-tts/codec-encoder.h`/`.cpp` and its `-host` counterpart) —
has a reference implementation no prior stage of this family had read:
`transformers`' `MimiModel`, at **`transformers==4.57.3`**. That is a **third
provenance** beside `QwenLM/Qwen3-TTS@022e286` and the checkpoint's own
`speech_tokenizer/config.json`, and reading a convention off the wrong one of
the three produces finite codes that decode to plausible audio. Qwen
contributes only a subclass that nulls the decoder halves
(`Qwen3TTSTokenizerV2Encoder(MimiModel)`, `modeling_qwen3_tts_tokenizer_v2.py:899-908`;
`upsample`/`decoder_transformer`/`decoder` all observed `None` on the loaded
model) plus two post-steps. Everything from the waveform to the codes is
unmodified `MimiModel`.

Task 1 transcribed the conventions into `codec_encoder/conventions.json`, each
with its own `file:line` and each with how it was checked. The ones that
change the answer silently:

| convention | value | source (`transformers/models/mimi/modeling_mimi.py`) |
| --- | --- | --- |
| convolution padding | causal, `constant`, value 0.0; **all** of `padding_total` on the LEFT, only `extra_padding` on the right | `:221-222`, `:237-246`, `:263-273`, `:331-333`. The symmetric split at `:249-250` is the NON-causal branch (`:336-338`) and this checkpoint never reaches it |
| activation | ELU, alpha 1.0, **before** each convolution, never after | `:418-419` (resnet block), `:463` (each downsampling conv), `:468` (the tail). No activation before the stem or after the tail |
| residual unit | kernels `(3, 1)`, **dilations `[1, 1]` at every stage**, bottleneck `dim // 2`, `nn.Identity` shortcut | `:409`, `:413`, `:415-419`, `:422-425`, `:441`, `:461` |
| split RVQ input | the acoustic branch is fed **the original 512-wide latent**, not the semantic branch's residual — two independent chains over one shared input | `:1318-1345`, specifically `:1340-1342` |
| split RVQ order | semantic stages first, then acoustic; residual in the `input_proj`'ed 256-wide space; euclidean, **not** squared; codebooks not L2-normalized | `:1269-1287`, `:1243-1246`. Order verified by shadowing each codebook module with its own `(branch, stage)` identity, not by reading |
| `trim_right_ratio` | 1.0, and it **never reaches the encode path** — `MimiConvTranspose1d` only, i.e. the decoder | `:359`, `:373-381`, `:393-399` |
| stride order | the encoder walks `upsampling_ratios` **reversed**: strides 4, 5, 6, 8 and kernels 8, 10, 12, 16 | `:456`, `:457`/`:466`, `:465` |
| frame downsampler padding | `replicate`, **not** `constant` — the one convolution on the encode path that does not zero-pad | `:1406-1415` overriding `:222` |
| transformer position | **after** the SEANet stack and **before** the frame downsampler | `:1456-1467` |

Three of those deserve naming as traps rather than as facts. `dilation_growth_rate`
is declared 2 and is **inert**: `num_residual_layers` is 1, so `j` is only ever
0 and `2**0 = 1` — a port reading "growth rate 2" as per-stage dilations
1/2/4/8 builds a different encoder from the same weights and every shape still
resolves. The checkpoint's `encoder_config.hidden_act` is `gelu` and belongs to
the **transformer's** MLP, not to the SEANet stack, which is ELU everywhere and
reads no config field to decide it. And the replicate-padded downsampler
corrupts exactly the first frame of every clip if padded with zeros like its
neighbours — one frame in 101, which no summary statistic shows.

Qwen's two post-steps read **top-level** config keys, and both have a decoy of
the same meaning inside `encoder_config`: the quantizer slice takes
`encoder_valid_num_quantizers` = 16 (`modeling_qwen3_tts_tokenizer_v2.py:983`,
key read at `:933`) where `encoder_config.num_quantizers` is 32, and the frame
trim divides by `encode_downsample_rate` = 1920 (`:984`, key read at `:939`)
where `encoder_config` carries no downsample rate at all. The package's own
`synthesize.qwen3-tts.codec.hop_length` is written from the checkpoint's
`decode_upsample_rate`; both are 1920 here, so it is the right number today
**by coincidence of this checkpoint**, and a future checkpoint where the two
differ would make the encoder's trim silently wrong.

### Codec-encoder geometry, reconciled before the graph was written

The package publishes **no** `codec.encoder.*` metadata namespace — the
catalog's encoder widths are compiled-in literals — so Task 1 reconciled the
checkpoint's declared values against what the port derives, as a table to be
checked rather than a specification to be implemented:

| quantity | checkpoint | port derives | agree |
| --- | --- | --- | --- |
| stage strides | `encoder_config.upsampling_ratios` `[8, 6, 5, 4]` | `[4, 5, 6, 8]` (`catalog.cpp:430-431`) | yes — the key is named for the DECODER's walk and the encoder reverses it |
| stage kernels / widths | — | `[8, 10, 12, 16]` / `128, 256, 512, 1024` from `num_filters` 64 | yes, observed on the instantiated modules |
| total downsampling | `frame_size` 1920 | 4·5·6·8 = 960, × the downsampler's stride 2 = 1920 | yes |
| frame downsampler | — | kernel 4, stride 2, 512→512, no bias | yes (`catalog.cpp:432`, `:546-547`); the catalog cannot record its `replicate` pad mode |
| transformer | `encoder_config` 8 layers / 512 / 8 heads / 2048 | same | yes |
| latent width | `encoder_config.hidden_size` 512 | 512 | yes |
| projected width | `encoder_config.codebook_dim` = `vector_quantization_hidden_dimension` = 256 | `codebook_dim / 2` from the **decoder's** 512 (`catalog.cpp:409`) | yes — two derivations, one answer, and this is where that is checked |
| codebook | `encoder_config.codebook_size` 2048, shape `[2048, 256]` | `{256, 2048}` (`catalog.cpp:413`) | yes |
| kept quantizers | **top-level** `encoder_valid_num_quantizers` 16 | 1 semantic + 31 acoustic resolved, 16 evaluated | yes; 16 of the 31 acoustic codebooks are never evaluated, intended, because the package carries what the checkpoint carries |

Two further findings from that reconciliation. The transformer's
`attn_implementation` is observed **`sdpa`**, not the talker's `eager`:
`modeling_qwen3_tts.py:1872` pops `attn_implementation` out of kwargs before
`:1915-1918` forwards the rest to the tokenizer, so any comparison against
these artifacts is comparing against SDPA's accumulation order.

And `encoder_config.sliding_window` declares 250 frames, which **the port
deliberately does not implement**. Upstream's own `create_causal_mask` uses the
plain causal mask unconditionally and never reads the field
(`modeling_mimi.py:1101` → `masking_utils.py:795`); the windowed variant is a
different function `modeling_mimi.py` never imports, and the only reader is
`MimiFlashAttention2`, which is not installed. The wrapper's sliding-window
code belongs to the codec **decoder**, a different class, which is the likely
source of confusion — the plan's own brief asserted the window applied here and
was **wrong**, and the implementer's contradiction was independently confirmed.
Measured on the real 96 tensors at `[1, 400, 512]`, the default is
bit-identical (0.000e+00) to an explicit plain-causal mask and 4.64x rms away
from an explicit sliding(250).

Two things make this a settled claim rather than a reading. First, the probe
that pins it had to be rebuilt: a single-layer sweep discriminates (**9.09e-4
at window ≥ 16, exactly 0 at ≤ 15**), where the eight-layer probe that first
tested it cannot see anything at all — with 8 layers the receptive field is
`1 + L(w - 1)`, so the last frame depends on frame 0 either way. That retained
eight-layer check is now labelled unreliable in the test file rather than
deleted, because it was the seventh check in this repository found unable to
fail. Second, the shipped clips do reach past the window: the three cases
`conventions.json` was written over are at most 101 frames and stay under it,
but **`base-ref-max` is 375 reference frames and the port agrees with
upstream-f32 there on 100.000% of code decisions (0 of 6000)** — direct
evidence, past 250, that the plain-causal choice is upstream's.

### The codebook dtype, and why the reference codes are not a gate

**The design prescribed plain equality on `codes/reference.i32`, and jiangzhuo
dropped that gate by ruling on 2026-08-13.** The ruling's home is the design
record's fourth erratum
(`2026-08-11-qwen3-tts-stage-2-design.md`, "Erratum, 2026-08-13"); it is
repeated here because a later variant's implementer will look for it in the
family record.

`MimiEuclideanCodebook.embed` is not stored — it is derived at runtime as
`embed_sum / cluster_usage.clamp(1e-5)` (`modeling_mimi.py:1186-1202`) at
whatever dtype the tokenizer was loaded with. The tokenizer inherits the
talker's dtype, and the model card's own `dtype=torch.bfloat16` makes the
division and the table **bfloat16**. The converter performs the same division
in float32 from the float32 safetensors and bakes an **f32** table
(`scripts/convert-qwen3-tts.py:376-388`) — deliberately, and it is the more
accurate of the two. They differ on all 2048 entries of every one of the 16
live codebooks.

What that costs, over 146 clips / 792,112 frame-stage decisions: substituting
the f32 table into the argmin alone changes **4.04%** of emitted codes;
carrying it through the whole RVQ changes **12.73%**; the per-decision flip
rate is **0.624%**. Not one of the 792,112 decisions was an exact tie, and the
rate is uniform across speech, tones, noise and 120 s clips, so it is not a
tail a lucky case avoids.

**And upstream does not agree with itself.** Loading with
`dtype=torch.bfloat16` — what the model card and the shipped demo both do —
reproduces the committed `codes/reference.i32` exactly. **Omitting `dtype`,
which `transformers` resolves to float32, disagrees with it on 760 of 1616
codes on `base-icl-en` — 47.03%.** (This read "794 of 1616 — 49%" until
2026-08-14, fifteen lines above a table giving 760 for the same comparison.
794 was a 2026-08-13 pre-execution estimate; 760 is what the committed
float32 re-run produces and is what the grid, the table below and
`scripts/dump_reference_qwen3_tts_codec_encoder_float32.py` all carry. The
same stale figure survives in the dated plan and spec documents, which record
what was believed when they were written and are not corrected here.) A gate
that one run of the reference
implementation passes and another fails is testing a load-time keyword
argument, not this port. Plan 1's "codes match byte-for-byte" observation was
never about the table either: both scripts load `bfloat16`, so it measured
determinism between two bf16 runs.

**What the gate became instead:** the codec encoder's stage-wise F32 artifacts
at a bf16-derived tolerance, plus the dequantized reconstruction. **The
agreement rates below are recorded as evidence and gate nothing.**

| comparison | `base-icl-en` | `base-ref-min` |
| --- | ---: | ---: |
| port vs upstream-f32 | **100.000%** (0 of 1616) | **100.000%** (0 of 208) |
| port vs the bf16 oracle | 52.970% (760 of 1616) | 53.846% (96 of 208) |
| upstream-f32 vs the bf16 oracle | 52.970% (760 of 1616) | 53.846% (96 of 208) |

> **THE THREE CODEC CASES ARE ONE RECORDING, AND THIS DOCUMENT DID NOT SAY SO
> UNTIL 2026-08-14.** `base-ref-min`'s `waveform.f32` is a **byte-exact
> prefix** of `base-icl-en`'s — the first 24,000 of its 193,920 samples,
> checked on the bytes — and `base-text-short` is that same clip again at full
> length, differing only in synthesis text the encoder never reads. So
> `base-icl-en` and `base-text-short` are the SAME 8.08-second clip: one
> speaker, one microphone, one sample rate, two lengths. Wherever this section
> says "three cases", or prints two columns side by side, or notes that "the
> three calibration cases sit at 0.00/3.96/3.96%" — the identical 3.96/3.96
> being that clip measured twice — **none of it is independent corroboration**.
> A second speaker would be, and **since 2026-08-17 there is one** — see "The
> second reference recording" below, which also records what it changed and
> what it did not. Everything above this sentence describes the four cases that
> existed before it. `tests/tolerances/qwen3-tts.json`,
> `scripts/dump_reference_qwen3_tts_codec_encoder.py` and
> `scripts/validate-qwen3-tts-codec_encoder.py` have all stated this from the
> start; this file — the one a later variant's implementer is directed to —
> did not, which is why the acoustic branch's 1.64x–2.0x headroom should be
> read as drawn from a single microphone.

The last two rows are equal because they differ on **exactly the same codes**:
the divergence is attributable to bf16, decision by decision, and the dropped
equality gate would have failed this correct port on roughly 47% of them. A
correction is folded in here because this record briefly stated otherwise: the
**4.04%** above is the codebook table's isolated contribution and a *floor* on
divergence, never the observed end-to-end rate.

**The residual risk this leaves, stated plainly.** A flipped code is not a
small error downstream — it selects a different embedding row in the ICL
prompt — and no continuous probe in this plan can watch that happen. What the
continuous gate buys is the ability to tell a near-tie from a structural
error. What it does not buy is a proof that this port's *selections* match
upstream's, and nothing here may be read as though it did. The 100.000% row
above is the strongest available substitute and it is a comparison against
upstream's own float32 run, not against the shipped bf16 oracle.

### CUDA placement for the speaker and codec encoders: measured, and declined

Stage 2 Plan 4 Task 12. §7 says the CUDA Execution Backend "reuses Stage 1's
wiring", and Stage 1's twin pass covers `codec.decoder.*` only -- the encoder is
deliberately outside it, by a decision recorded at `model.cpp:523-540` with its
byte cost. Reading §7 as commissioning a twin for the two new graphs is a
reading, not the text. So this task measured what a twin could possibly buy
before writing one, and the answer is that it cannot pay.

**The two graphs' CPU cost, measured on a `Release`-typed tree.** Each driver
loads the 2.4 GiB package before it runs its graph, so one timing is load plus
graph. Timing the same driver on a 1-second clip and a 30-second one separates
them, because the load is identical and the graph is not. Median of three, in
`build` at `CMAKE_BUILD_TYPE=Release`:

| driver | 1 s of audio | 30 s of audio | difference | per second of audio |
| --- | ---: | ---: | ---: | ---: |
| codec encoder | 1.943 s | 7.125 s | 5.182 s | **0.179 s** |
| speaker encoder | 1.800 s | 2.777 s | 0.977 s | **0.034 s** |

Both extrapolate to the same fixed cost -- 1.764 s and 1.766 s of model load --
which is the cross-check that the separation is real rather than a fit.

**A third point tests the extrapolation rather than assuming it.** The model
predicts 1.764 + 0.179 × 8.08 = **3.21 s** for the codec encoder on the
8.08-second reference; measured, median of three, it is **3.183 s** — within
0.9 %, and taken while a CUDA build was competing for cores, so if anything the
measurement is the pessimistic one. The graph's own share there is
3.183 − 1.764 = **1.42 s**.

So on that reference the two graphs together cost about **1.69 s**: 1.42 s in
the codec encoder and 0.27 s in the speaker encoder.

**What a twin would cost, against Stage 1's own measurement.** Mirroring the
codec encoder is 224,674,944 bytes (`model.cpp:523-540`); the speaker encoder
adds about 17 MiB. Stage 1 measured roughly **7 s of extra load for 457 MB**, so
about **3.7 s** for these 241 MB.

**So the transfer costs 2.2× the entire compute it would accelerate, before any
speedup is applied.** Even at Stage 1's 35× -- the figure its codec-decoder twin
actually achieved -- the saving is 1.69 × (1 − 1/35) ≈ 1.64 s against a 3.7 s
cost. There is no speedup at which this twin pays, because the ceiling is the
graphs' whole CPU time and the floor is a larger one-time transfer.

**And the amortization runs the wrong way.** Stage 1's codec-decoder split is a
deployment choice -- a net loss for a one-shot utterance and a clear win for a
load-once process -- because the decoder runs on every synthesis. These two run
**once per Voice Profile**, not once per synthesis: the Profile carries the
x-vector and the reference codes, and every later synthesis reuses them. A
load-once process therefore pays the mirror once and saves almost nothing,
which is the opposite of what made Stage 1's twin worth having.

**Outcome: the second. The twin is not worth writing, and it was not written.**
`model.cpp:523-540`'s comment said a later plan might measure this; it has been
measured and the comment is updated to say so. The discrete-outputs rule is not
even reached -- it would have held the RVQ argmin on the host regardless, so
only part of the codec encoder was ever eligible.

**What that leaves for the CUDA sub-grid, stated rather than skipped.** Under
this outcome the two new graphs stay on the CPU, so a `codec_encoder` CUDA cell
would describe a placement nothing runs on. But the Stage 1 codec-decoder twin
*does* move on a Base package, so `public` and `replay` CUDA figures are real
and are measured anyway -- `docs/backends.md` gate 5 applies to the placement
that demonstrably exists. Task 13 decides where they are recorded, and the
coverage rule's all-or-nothing sub-grid means they cannot become a `backends`
key without a third stage that would be a fabrication.

### The Base variant carries no CUDA sub-grid, and one real CUDA measurement

Stage 2 Plan 4 Task 13, conditional on Task 12's second outcome. The coverage
rule makes a `backends` sub-grid all-or-nothing -- every one must carry the
reference profile's full stage set, which for this variant is
{`public`, `replay`, `codec_encoder`} -- and an absent `backends` key is legal,
which is the permitted hole CustomVoice's `Q8_MIXED` already occupies.

**Only ONE of the three stages exercises CUDA on this variant, not the two Plan
4 expected.** Task 12's second outcome anticipated that `public` and `replay`
would both stay measurable. `replay` does not, and the reason is structural
rather than a matter of effort:

| stage | does a CUDA run measure anything different? | why |
| --- | --- | --- |
| `public` | **yes** | the Stage 1 codec-decoder twin does move on a Base package, so `synth_model_load` with `SYNTH_BACKEND_CUDA` puts a real graph on the device |
| `replay` | **no** | its two probes are `speaker.x_vector` and `prompt.icl_embed`. `tests/qwen3_tts_xvector_driver.cpp` calls `Model::load_cpu` and has no backend argument at all, and the ICL prompt comparison is host-side assembly. A "CUDA" run would return the identical numbers. |
| `codec_encoder` | **no** | Task 12 measured the twin and declined it, so this graph runs on the CPU under a CUDA request |

So two of the three cells a sub-grid needs could only be filled by recording CPU
figures under a CUDA key. **That is not done**, and the sub-grid is therefore not
committed. `tests/tolerances/qwen3-tts.json` gains no `backends` key for
`qwen3-tts-12hz-0-6b-base`.

**The measurement that IS real, recorded here because the cell cannot hold it.**
`scripts/validate-qwen3-tts-public.py` against the BF16 Base package with
`--backend cuda`, runner built from the `rel-dgx-spark` preset
(`CMAKE_BUILD_TYPE=Release`, sm_121a, CUDA 13.3), through a reference-audio
Voice Profile:

- **7 checks pass, 3 skipped**, the same seven and the same three structural
  skips as every CPU cell -- the Voice kind is a property of the package's
  catalogue, not of the backend. **Stale as of 2026-08-19, not re-measured
  here:** Stage 3 Plan 2 Task 6 added a `create_from_description` refusal
  check that runs for every non-`description_text` package, CPU or CUDA alike
  -- it probes `synth_voice_profile_create_from_description` directly and
  never reaches a backend-specific graph, so nothing about the reasoning above
  changes between backends. The CPU sibling of this exact cell moved 7 -> 8 in
  `tests/tolerances/qwen3-tts.json` the same day. This CUDA figure almost
  certainly moves the same way but was not re-run under CUDA to confirm it --
  this environment had no CUDA-preset build tree at the time -- so 7 is left
  standing here as what was actually measured rather than silently bumped to
  an unverified 8. Whoever next runs this cell under CUDA should expect 8, not
  be surprised by it.
- The audio differs from CPU, which is what says the request reached the
  device: seed 7 through reference A gives `ed32bb3b13800fe1` on CUDA against
  `9eef2beaf63cb60e` on CPU. That is the codec decoder's TF32 arithmetic, the
  same shape Stage 1 recorded when only `audio.pcm` moved between backends.
- Seed reporting, same-seed byte-identical repeatability, different-seed
  divergence and Voice divergence all hold **within** the CUDA backend.

**No threshold was derived, because nothing here needed one.** Plan 4's Task 13
Step 1 exists to stop the CPU `codec.chain` gate of 1.0e-3 being reused for a
backend whose own CUDA-versus-CPU disagreement is 5.20e-03 -- 5.2× that gate and
about 1.33 bf16 units. That derivation is only owed for a `codec_encoder` CUDA
cell, and there is none: the graph did not move. The 5.20e-03 figure stands
unused and unretracted, and a later rung that does place this graph must derive
from it rather than from the CPU gate.

**What this does not claim.** `docs/backends.md` gate 5 wants latency, RTF and
peak memory for a supported backend; those are Task 14's and are not asserted
here. This section records agreement and placement only.

### The first ICL Listening Audit, 2026-08-17: `no_obvious_regression`

Stage 2 Plan 4 Task 15. **No ICL listening audit had ever run for this family.**
The 2026-08-13 audit covered x-vector mode only and said so; ICL did not exist
when it ran.

One listener (jiangzhuo), five blind pairs and two labelled resemblance checks,
all on the same sentence at seed 7, generated from `build/rel-dgx-spark`
(`CMAKE_BUILD_TYPE=Release`). A/B positions were shuffled with a recorded seed
and the key was not on the page.

| # | comparison | verdict |
| --- | --- | --- |
| 1 | BF16 against **F16**, ICL, reference A | different, **neither degraded** |
| 2 | BF16 against **Q8_MIXED**, ICL, reference A | different, **neither degraded** |
| 3 | **CUDA** against **CPU**, BF16, ICL, reference A | indistinguishable |
| 4 | ICL against x-vector, BF16, reference A | indistinguishable |
| 5 | ICL against x-vector, BF16, reference B (second speaker) | indistinguishable |
| L1 | reference A source against its ICL clone | **same speaker** |
| L2 | reference B source against its ICL clone | **same speaker** |

**Result: `no_obvious_regression`.** No pair produced a named regression.

**Every pair differed in bytes**, checked before the verdict was recorded, so
each "indistinguishable" is a listening judgement rather than a trivial truth —
the two sides of pairs 3, 4 and 5 are genuinely different audio.

**What each row buys.**

- **Both quantization profiles are audible but not degraded.** That is exactly
  what a profile should produce, and for `Q8_MIXED` it is the audible evidence
  the measurements could not supply: 33.7 % smaller and RTF 0.863 is only worth
  having if it still sounds right, and a tolerance table cannot say whether it
  does.
- **CPU and CUDA are indistinguishable** despite differing bytes. That is the
  codec decoder's TF32 arithmetic being inaudible, and it is the same kind of
  evidence that removed this project's strict-FP32 gate on 2026-07-26 — two
  renderings differing only in that gate, compared sample-aligned, inaudible.
- **ICL and x-vector are indistinguishable on both references.** This is the
  first audible comparison of the two clone modes and it is **scoped, not a
  verdict on ICL**: two clips, one sentence, one listener. ICL's machinery may
  well matter on material these two references do not represent — prosody
  carried by the transcript, longer or harder utterances — and nothing here
  tests that. What it does say is that on these clips the extra path costs
  nothing audible and buys nothing audible either.
- **Both clones were judged the same speaker as their source, including the
  second speaker.** The 2026-08-13 audit produced that judgement on one source
  clip; this is the second, on a different speaker and a different recording
  chain.

**What it does not establish.** Two recordings, one sentence, one listener. The
Validation Level does not move and `quality_evaluation` stays deferred per ADR
0017 — this is evidence, not a level. `spec:532`'s "audit recorded" gate is now
**met**, which is a statement about this plan's deliverable and not about
publication.

The material is preserved at `build/listening-icl/` — `artifact.html` with the
audio embedded, the per-clip WAVs, and `manifest.json` carrying the shuffle seed
and the blind key.

### Stage 2 Plan 4: what it measured, and what it refused to claim

Executed 2026-08-17. Plan 4 was written as a **measurement** plan whose stated
correct answers include "this profile does not pay" and "this twin is not worth
writing", and both of those are among its results. The sections below this one
carry each measurement with the artifact that produced it; this is the summary
and the ledger of what remains.

**The blocker, first.** `synthesize-quantize` could not cut a Base package at
all: 76 `speaker_encoder.*` and 161 `codec.encoder.*` classified `Unknown` and
the tool stopped on the first one. Worse, the runtime disagreed with it -- the
`codec.` prefix put the encoder in the right half, but the speaker encoder's 38
convolution weights were `Role::Matrix`, so the runtime expected Q8_0 for a
package the tool refused to emit.

**The ruling settled it per region, and neither uniform answer was available.**
Holding those 38 at the profile's block type is not merely undesirable, it is
impossible: a block runs along `ne[0]`, which for a convolution kernel is the
kernel extent -- 1, 3 and 5 across all 38 -- against Q8_0's block of 32. The
packed alternative clears the block size but emits rank 2, and this family's
runtime implements neither half of consuming that. So they take a `ConvKernel`
role at the halved fallback, native three-axis. All three published CustomVoice
packages were re-cut and their digests proved unchanged.

**What each measurement answered**

| question | answer | where |
| --- | --- | --- |
| Does F16 pay? | **No.** 184,448 bytes *larger* than its source | "Stage 2 Plan 4" cells in `tests/tolerances/qwen3-tts.json` |
| Does Q8_MIXED pay? | **Yes, on both.** 33.7 % smaller, RTF 3.15 → 0.863 | "Base latency, RTF and peak memory" |
| Does quantizing the speaker encoder pay? | **No, and there is nothing to gain** | "Does quantizing the speaker encoder pay?" |
| Does the conv-exempt precedent transfer? | **No.** Drift is exactly zero | "Does the conv-exempt precedent transfer?" |
| Is a CUDA twin for the two new graphs worth writing? | **No.** Transfer is 2.2× the compute | "CUDA placement for the speaker and codec encoders" |
| Can the semantic gate's cliff be removed? | **Yes**, by changing which frames the percentile covers | `gate_scope_warning_retired` in the tolerance file |

**Five results contradicted what the plan expected, and each is recorded with
the contradiction rather than smoothed over.**

1. The flip rate is **not monotone in reference length**, so the length-scoping
   branch `gate_scope_warning` offered is refuted -- by the second speaker Task 1
   added precisely to test it.
2. The masked statistic the plan prescribed -- condition on non-flipped frames --
   has **~1× discrimination** and is undefined on several cases, because a mask
   that mentions the port selects away a fault in the port. The committed mask
   comes from `upstream-f32` against the oracle instead, and reaches 390–644×.
3. Both quantization routes the plan offered for the 38 convolutions are
   unavailable, for the block-arithmetic reason above.
4. The public validator's **entire model of a Voice** was inapplicable to a
   package with no Preset Voice catalogue, not merely a few of its checks.
5. Only **one** of three stages measures anything different on CUDA, not the two
   the plan assumed -- the x-vector driver has no backend argument at all.

**Two corrections to this plan's own reasoning, kept because they are the useful
part.** An inversion was written expecting a *cut* to fail on a row-size check;
the cut succeeded and it is the runtime that refuses, and in the normal
configuration the type check fires first so the rank disagreement is *masked*
until both sides are moved together. And a 25× tightening of the acoustic bound
passed every case in the validator and **failed the C++ arm on its first run**,
because the two consumers feed the port different inputs; it was withdrawn.

**The Listening Audit returned `no_obvious_regression` on 2026-08-17**, which is
recorded in its own section above and which met `spec:532`'s audit gate.

**What Plan 4 did NOT deliver, stated plainly.**

- **Publication.** Nothing was uploaded and nothing asked to be. The Base variant
  is not published.
- **Any movement of the Validation Level.** `quality_evaluation` stays deferred
  per ADR 0017.
- **A CUDA sub-grid**, by the measured decision above; the coverage rule permits
  the absent `backends` key and the one real CUDA measurement is recorded in
  prose instead of a fabricated cell.
- **Stage 3, the 1.7B variant, multi-clip enrollment, Native Streaming Synthesis,
  Voice Conversion, and the CLI** -- all §10 non-goals, untouched.

**Carried forward.** `prompt.icl_embed` is measured for BF16 only. The
autoregressive replay probes CustomVoice carries are still unmeasured here. The
`278` figure in the design's fifth erratum remains unverified -- its *companion*
was re-measured over five cases and reproduces, which narrows but does not close
what that erratum warns about.

### Base latency, RTF and peak memory, measured on Release 2026-08-17

Stage 2 Plan 4 Task 14, separate from Task 13 because it is a separate build
tree and that separation is the point: a `dev-*` preset proves correctness and
never wall-clock time.

**Every figure below comes from `build/rel-dgx-spark`** -- the committed
`rel-dgx-spark` preset, `CMAKE_BUILD_TYPE=Release`, sm_121a, CUDA Toolkit 13.3,
on a DGX Spark/GB10. Workload: "This is a test of Qwen three T T S base voice
cloning." synthesized through an ICL reference-audio Voice Profile prepared from
`base-icl-en`'s waveform, seed 7, `max_output_frames` 983040, 10 threads.

| profile | backend | frames | audio | synthesis | **RTF** | load | peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | CPU | 88,320 | 3.680 s | 11.60 s | **3.15** | 1.50 s | 3.29 GiB |
| BF16 | CUDA | 88,320 | 3.680 s | 10.85 s | **2.95** | 1.64 s | 3.29 GiB |
| Q8_MIXED | CPU | 78,720 | 3.280 s | 2.84 s | **0.863** | 1.13 s | 2.23 GiB |

Medians of three (BF16) and six (Q8_MIXED) runs. RTF is each row's own
synthesis time over its own audio length, never across rows -- the two profiles
stop at different frame counts, which is a property of their weights and not an
error.

**Q8_MIXED crosses real time and is the profile that pays.** RTF 3.15 → 0.863 is
a **3.65×** improvement, alongside a package 33.7 % smaller and 1.06 GiB less
peak RSS. That is the speed half Task 8 handed forward, and it turns Task 8's
"it pays on size" into "it pays on both". For orientation only, Stage 1 recorded
RTF 0.85 for CustomVoice's Q8_MIXED on its own workload -- a different variant
and a different case, but the closeness is a consistency signal rather than a
coincidence, since both quantize the same autoregressive half.

**CUDA buys about 6 % and that is the expected amount.** 11.60 s → 10.85 s on
the same tree, RTF 3.15 → 2.95 — 6.5 % and 6.3 % respectively, which is what
those two pairs divide out to. This paragraph read "about 8 %" until 2026-08-18,
against its own numbers on the same line. Only the Stage 1 codec-decoder twin moves; the
autoregressive half is held on the CPU by the discrete-outputs rule and
dominates, and the two new graphs stay on the CPU by Task 12's measured
decision. `docs/backends.md` requires performance measurement for support but no
minimum speedup, so this is recorded as measured and CUDA is **not** described
as accelerated for this variant beyond what the number says.

**Gate 6, repeated runs and cleanup.** Every repetition at a fixed seed produced
byte-identical frame counts -- 88,320 for BF16 and 78,720 for Q8_MIXED across
all runs -- and peak RSS varied by under 0.03 % between repetitions of the same
configuration, so nothing accumulates across runs.

**Two contaminants, named rather than discovered.** The second BPE frontend costs
about +45 MB of peak RSS and is inside every figure above. And the first
measurement pass produced BF16 and Q8_MIXED synthesis times spread 2.8–9.7 s
while a CUDA build was finishing on the same machine; re-measured on a quiet
machine, Q8_MIXED lands in 2.765–2.893 s across six runs. **The contended
figures are discarded, not averaged in.** One BF16 repetition still read 20.9 s
against its neighbours' 11.56 and 11.64; it is treated as an outlier and the
median of the stable runs is reported, which is why the run counts are stated
above.

**What is not claimed.** These are one machine, one workload and one utterance
length. This document's own build-dependent-length finding
(`qwen3-tts.md:2932-2942`) is why -- not `docs/testing.md`, cited here until
Task 5's fix round corrected it; that file documents a different claim, the
Release/RelWithDebInfo speed gap -- so an RTF computed from a frame count
taken on another tree would be wrong; every figure here is same-tree. No
listening judgement is implied -- Task 15 owns that, and a tolerance table is
not audible evidence.

### Does quantizing the speaker encoder pay? Measured 2026-08-17

Stage 2 Plan 4 Task 9. §7 says the speaker encoder "is small enough that
quantizing it is unlikely to pay" and then forbids assuming it. This is the
posterior.

**The artifact, as §7 requires.** Packages
`build/qwen3-tts-12hz-0-6b-base-{F16,Q8_MIXED}.gguf`, cut by
`build/bin/synthesize-quantize` from
`models/qwen3-tts-12hz-0-6b-base/qwen3-tts-12hz-0-6b-base-BF16.gguf`
(sha256 `993f4cd1…`); tensor census read with `gguf.GGUFReader`; x-vectors driven
by `synthesize-qwen3-tts-xvector-driver` through
`scripts/validate-qwen3-tts-replay.py --compare-x-vector`; trees `build` and
`build-integration`, both `CMAKE_BUILD_TYPE=Release`.

**Size — the subject is 38 convolution weights, 8,843,264 elements.**

| profile | storage | bytes | note |
| --- | --- | ---: | --- |
| BF16 (source) | BF16 | 16.87 MiB | 0.703 % of the package |
| F16 | F16 | **16.87 MiB** | **exactly zero saved** — both are two-byte types |
| Q8_MIXED | F16 | **16.87 MiB** | **exactly zero saved** — Task 6's ConvKernel ruling holds them at the halved fallback here too |
| *Q8_0, packed (hypothetical)* | *Q8_0* | *8.96 MiB* | *would save 7.91 MiB, 0.329 % of the package* |

The 38 biases are F32 under every non-source profile and are not the question.

**Accuracy — no measurable cost.** Worst x-vector cosine against the oracle,
over the four cases Task 4 made drivable: BF16 **0.99999467**, F16 and Q8_MIXED
both **0.99999501**. The two halved profiles are byte-identical to each other
and very slightly *better* than the source profile, which is run-to-run scale
rather than an improvement. The reason is recorded in the BF16 cell: the
residual is the **oracle's** bf16 storage, not the port's weights — recomputing
the oracle's own final ASP/FC layer in float64 disagrees with the oracle by more
than the port does — so an F16 port weight is still finer than what it is being
compared against.

**Wall time is not claimed.** No latency, RTF or peak-memory figure was taken;
`docs/backends.md` gate 5 wants all three and Task 14 owns them.

**The outcome is the second — it does not pay — and the remedy that outcome
prescribes would make things worse.** Plan 4's Task 9 says that on this outcome
"the 38 weights become `Sensitive`". For this family `Sensitive` means F32, four
bytes, so that change would make the package **16.87 MiB larger** than it is
now. Of the three classifications actually available, the committed one is the
best:

| classification | storage | bytes | reachable? |
| --- | --- | ---: | --- |
| `ConvKernel` (committed) | F16 | 16.87 MiB | yes |
| `Sensitive` | F32 | 33.75 MiB | yes, and strictly worse |
| `MatrixWeight` packed | Q8_0 | 8.96 MiB | **no** — needs a packed branch in `Resolver::find` and a packed path in `same_conv1d`, neither of which this family has (Task 6) |

So "it does not pay" here means **there is nothing available to gain**, not that
the current classification is wrong. §7's prior is confirmed, and for a stronger
reason than §7 gave: the speaker encoder does not shrink under any profile this
family has, because the only profile type narrower than its source is one the
runtime cannot consume.

### Does the conv-exempt precedent transfer? Measured 2026-08-17

Stage 2 Plan 4 Task 10. §7 says the codec encoder "is convolution-heavy, which
is the shape that produced the conv-exempt policy in another family" and then
says neither precedent is imported by analogy. This is the measurement that
replaces the analogy.

**The drift is zero.** The port's codec encoder was driven from all three
packages over all five cases and its output compared byte for byte:

| comparison | artifacts × cases | identical |
| --- | ---: | ---: |
| F16 against BF16 | 4 × 5 | **20 of 20** |
| Q8_MIXED against BF16 | 5 × 5 | **25 of 25** |

covering `latents.f32`, `rvq_reconstruction.f32`, `codes.i32`, `downsample.f32`
and `rvq_distance_margin.f32`. Not a small drift: **no drift**. Zero code flips,
on every case, at every profile.

**The first outcome is unreachable, and Plan 4 says so in advance.** It would
require some `codec.encoder.*` tensor to be in a quantized role at all.
`src/arch/qwen3-tts/catalog.cpp` splits halves on the `codec.` prefix, so the
whole encoder is F32 under every profile; Task 6's census confirmed all 161 of
its tensors are F32 in every package it cut. There is no role to add and nothing
for a `ConvKernel`-style exemption to protect.

**So the outcome is the second: the exemption does not transfer, because this
family's existing rule already covers the case.** Recorded with the drift
numbers that establish it rather than as an inheritance of that rule — which
would be the same analogy running the other way, and is the error Plan 4 names
explicitly.

**What the other family's evidence actually was, so the comparison is against
the real thing.** OmniVoice's `ConvKernel` role took clone-path RVQ drift from
1,023 of 2,808 positions to 98, and both its codec-half profiles remain
**blocked** because `ref.tokens` is 0/2 exact (`docs/quantization.md`). The
failure mode is real and this family's codec encoder has the same shape — a
nearest-neighbour RVQ argmin over a reference clip. It cannot arise here for a
reason that has nothing to do with convolutions: qwen3-tts never quantizes that
half at all, on an independent and separately measured ground recorded at
`classify_qwen3_codec` — halving it made the codec **1.75× slower on CPU**,
because its convolutions run through im2col into a matrix multiply where ggml's
F16 path is slower than its F32 one.

**What this does not say.** It does not say a quantized codec encoder would be
safe here; nothing measured one, because none exists. If a future profile ever
quantizes this half, the other family's evidence becomes live again and this
section is not a licence to skip it.

### The second reference recording

Added 2026-08-17 by Stage 2 Plan 4 Task 1, as `base-icl-en-second-speaker`.
Every flip-rate figure this family had came from one recording, and the task
that was about to derive a flip-rate threshold from them could not tell a
property of the port from a property of that clip.

**Provenance and licence, recorded here because the Golden Manifest cannot hold
them.** The plan directed that provenance and licence be written into "the
manifest case description"; the manifest schema
(`docs/schemas/synthesize-golden-manifest-v1.schema.json`) declares `case` with
`additionalProperties: false` and no `description` member, so there is nowhere
in the case object to put them. They live here instead, and the case's
`origin.locator` points at this section — which is what `origin` is for.

| | |
| --- | --- |
| locator | `https://zhu-han.github.io/omnivoice/audios/seedtts/prompt/seedtts_ref_en_1.wav` |
| sha256 | `57f25abc75c2e7cc4d3c9a6e45f54160034bc714b2ba952407dae7f91d6d31f3` |
| format | 24 kHz, mono, 16-bit PCM WAV — the package's own native rate, so nothing resamples |
| duration | 14.071916 s (337,726 samples), inside the declared 1 s–30 s bounds |
| reference code frames | 176 |
| transcript | "Some call me nature. Others call me Mother Nature. I've been here for over four point and five billion years, twenty-two thousand five hundred times longer than you." |
| licence | **Not stated upstream.** It is the Seed-TTS eval English reference set, served from the OmniVoice demo site; `BytedanceSpeech/seed-tts-eval` carries no `LICENSE` file at all. |

**Why this clip and not a cleaner-licensed one.** This repository already pins
and consumes these exact bytes: the same locator and the same digest are a
`reference-audio` artifact of `tests/golden/omnivoice/omnivoice-0-6b.manifest.json`,
and both committed OmniVoice clone goldens drive it. Reusing it adds a second
consumer of a dependency this tree already has, rather than a new one. The
alternative considered and rejected was upstream Qwen's own second demo asset,
`tokenizer_demo_1.wav` (24 kHz mono, 10.53 s), whose licence posture is
cleaner — same Apache-2.0 repository as `clone.wav` — but which publishes **no
transcript**, so an ICL case built on it would rest on a machine transcription,
and transcript mismatch is this family's one known catastrophic input (475.91 s
of CPU to `SYNTH_ERR_OUTPUT_LIMIT` with zero audio). The licence gap is recorded
above rather than resolved; it is the one thing this choice does not fix.

It is a genuinely different recording and not a transform of `clone.wav`: a
different speaker, a different corpus and a different recording chain, with
`ref_rms` 0.12291 against the port's own encoder. That distinction is the whole
point — the flip rate is a property of where a recording's latents fall
relative to the codebook, and a pitch-shifted or noise-added copy of one clip
moves along the same manifold and corroborates nothing.

**What it measured.** Dumped through the Base oracle, the ICL-prompt oracle,
the codec-encoder oracle and both float32 twins, all at the same
`dtype=torch.bfloat16` the other cases use (pinned at
`scripts/dump_reference_qwen3_tts_base.py:585`, not passed by the caller). It
takes the ICL alignment's **pad** arm (T1 50, T2 177), and all 25 of the
prompt oracle's checks pass, including the bitwise agreement between the
assembled block and the Base dumper's own `icl_embed.f32`.

Against the committed BF16/CPU `codec_encoder` gates, on the `Release`-typed
`build/` tree, it **passes every one**:

| quantity | measured | gate |
| --- | ---: | ---: |
| chain `rel_absmax` (worst stage `rvq_residual_s15`) | 2.483e-05 — 0.0064 of one bf16 unit | 1.0e-3 |
| semantic reconstruction p95 relative | 0.004152 | 2.0e-2 |
| acoustic reconstruction p95 relative | 0.2988 | 5.0e-1 |
| port vs upstream-f32 code agreement | **100.000%** (0 of 2816) | recorded, gates nothing |

The chain figure is **tighter than any of the other four cases**, including the
8.431e-05 that `base-ref-max` clears.

**The finding, which is not the one the plan expected.** Its semantic flip rate
is **1.70%** (3 of 176). Ordered by reference length the five rates now read:

| case | reference code frames | semantic flip rate | recording |
| --- | ---: | ---: | --- |
| `base-ref-min` | 13 | 0.00% | `clone.wav`, trimmed to 1.0 s |
| `base-icl-en` | 101 | 3.96% | `clone.wav`, full 8.08 s |
| `base-text-short` | 101 | 3.96% | `clone.wav`, full 8.08 s |
| **`base-icl-en-second-speaker`** | **176** | **1.70%** | **`seedtts_ref_en_1.wav`, full 14.07 s** |
| `base-ref-max` | 375 | 5.33% | `clone.wav`, **looped** to 30.0 s |

**That sequence is not monotone.** A longer reference from a second speaker
sits at less than half the rate of the shorter same-speaker cases. So reference
length does not order the flip rate, and "scope this probe by reference length"
— the first of the two branches `gate_scope_warning` offered the next plan — is
**refuted as a sufficient rule**. Plan 4 Task 1 Step 3 predicted three outcomes
and attached to its first one the inference that "the cliff is confirmed as a
function of reference length rather than of speaker": the measurement lands in
that outcome by its threshold (below ~5%) and **contradicts its inference**.
Both halves are recorded, because the plan's sentence is not evidence.

One confound is named rather than left implicit: `base-ref-max` is not a longer
recording. It is `clone.wav` **looped** to 30 s by `np.tile`
(`scripts/dump_reference_qwen3_tts_base.py:339-342`), roughly 3.7 repeats, so
its 5.33% may carry seam discontinuities rather than anything about length.
That makes the 375-frame point the weakest of the five for a length argument,
not the strongest.

**What this does not establish.** One additional speaker is two recordings, not
a corpus. Nobody has listened to this case's output — Task 15 owns that — and
the Validation Level does not move.

### Measured tolerances

Both stages below were **re-run for this record on 2026-08-14** in `build/` at
`CMAKE_BUILD_TYPE=Release` on the **CPU** backend against the real BF16 Base
package, and every committed figure reproduced to the digit. The thresholds
live in
`tests/tolerances/qwen3-tts.json`, `variants.qwen3-tts-12hz-0-6b-base.profiles.BF16.stages`.

**`codec_encoder`** — three cases, `base-icl-en`, `base-ref-min`,
`base-text-short`. Enforced by `synthesize-qwen3-tts-codec-encoder-golden`
(`tests/check-qwen3-tts-codec-encoder.cmake`), which re-runs the driver into
the build tree rather than reading a stale dump.

| gate | threshold | observed | headroom | injected fault reads |
| --- | ---: | ---: | ---: | ---: |
| `codec.chain` `rel_absmax`, per stage over 33 taps | 1.0e-3 | 9.303e-05 (`transformer_l7`) | 10.7x | 1.649 (1650x) |
| `codec.rvq_reconstruction.semantic`, p95 relative L2 | 2.0e-2 | 0.004103 | 4.9x | 1.601 (80x) |
| `codec.rvq_reconstruction.acoustic`, p95 relative L2 | 5.0e-1 | 0.2491 | 2.0x | 1.741 (3.5x) |

The chain gate is the primary one and the only one of the three independent of
the two f32 codebook tables being bit-identical by construction. Its worst tap
sits at **0.024 of one bfloat16 unit**, i.e. 42x below a single rounding. The
injected fault is upstream's own symmetric (non-causal) padding split — the
defect that builds the same shapes from the same weights and a different
encoder. The reconstruction gates are **percentiles, never maxima**, and that
is measured rather than chosen: one flipped code displaces the reconstruction
by about a full codebook-row separation, so the semantic branch's per-frame
maximum is 235.8 against a median of 0.55, and no bf16-scale maximum survives
that.

**Two qualifications on that headroom, neither of which was absorbed by
widening anything.**

- **The oracle's input is not the WAV.** `codec_encoder/waveform.f32` is
  exactly `bfloat16(clone.wav)`, verified element for element — the reference
  model loads in bfloat16. The stage-wise validator hands the port that
  already-rounded waveform, so its figures isolate the port's arithmetic with
  the input held identical. `tests/qwen3_tts_icl_real.cpp` hands the port the
  **WAV**, because that is what a caller hands the library, and pays one extra
  input rounding for it (`rel_absmax` 2.954e-03 on the waveform, 0.76 of one
  bf16 unit). Measured on `base-icl-en` alone, which is the only case that test
  drives: acoustic p95 **0.304719**, semantic 0.004033, code agreement
  **50.743%** where the bf16-waveform run of the same case gives 0.2491,
  **0.004103** and 52.970%. (That semantic figure was printed as 0.004033 on
  both sides until 2026-08-14 — the WAV-path number on both halves of a
  contrast whose whole point is that the two inputs differ, erasing the one
  move it was recording. The grid has it right in
  `observed_p95_relative_l2` against
  `observed_p95_relative_l2_from_wav_input`.) So **the acoustic branch's real headroom against a
  caller's own WAV is 1.64x, not the 2.0x the grid's own column records** —
  that column is the worst over three cases from the oracle's input, and the
  two are not the same measurement. Recorded under `registered_consumers`
  rather than accommodated; nothing was widened.
- **The semantic gate sits on a flip-rate cliff and does not generalize.** Run
  on `base-ref-max` (375 reference frames) it **fails**: p95 0.3478 against
  2.0e-2. That is **not a port defect** and three readings say so — codes agree
  with upstream-f32 100.000% (0 of 6000), and the chain gate passes *tighter*
  than the calibration cases at 8.431e-05. The p95 only enters the flipped tail
  above roughly a 5% flip rate; the three calibration cases sit at
  0.00/3.96/3.96% and `base-ref-max` at 5.33%, an 85x jump. **Widening was
  refused**: 0.35 would leave the injected fault 4.6x above the gate instead of
  80x, destroying its discrimination. It is recorded as `gate_scope_warning`,
  `cases` stays at 3, and the statistic needs redesigning in Plan 4 — gate the
  flip rate separately from the reconstruction error on non-flipped frames. The
  real evidence for that case is the 100% code agreement, not the p95.

**`replay.probes.prompt.icl_embed`** — the two-track prompt, same three case
ids, gated at **2.0e-2** relative on five comparisons each (text track, codec
track, their summed block, the block as it sits in the assembled prefill, and
the trailing schedule). Enforced by
`tests/qwen3_tts_icl_prompt_real.cpp`, which since 2026-08-14 reads the number
out of the tolerance file rather than carrying five hardcoded copies of it.

| case | arm | text | codec | block | prefill-tail | trailing |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `base-icl-en` | pad | 5.628e-03 | 2.232e-03 | 8.368e-03 | 8.368e-03 | 2.333e-03 |
| `base-ref-min` | truncate | 5.810e-03 | 2.338e-03 | **8.734e-03** | 8.734e-03 | 3.780e-03 |
| `base-text-short` | pad | 2.333e-03 | 2.232e-03 | 4.184e-03 | 4.184e-03 | 2.333e-03 |

Worst is 8.734e-03, so the gate carries 2.29x headroom. **It was deliberately
not recomputed by this file's own 5x widening rule**, which would have given
4.37e-02 and *loosened* a passing gate; an enforced bound is never relaxed to
match a formula. 2.0e-2 has its own justification — bfloat16's unit roundoff is
`2**-8` = 3.906e-03, so it is about five bf16 roundings.

**Decomposed against upstream's own module run in float32** (the method Task 4
established and every later graph task was required to use, because the bf16
dumps cannot settle correctness alone): the codec track is 0.0 … 4.2e-09,
float32-exact, being 16 lookups and 15 adds in the same order; the text track
is 1.28e-04 … 1.73e-04, **23x–30x below one bf16 unit roundoff**. The entire
residual against the oracle is the oracle's own arithmetic. The same holds
across the whole encoder chain: over the 33 taps the grid records, port vs
upstream-f32 is 8.021e-07 … 9.303e-05 (worst `transformer_l7`), where
upstream-f32 vs the bf16 oracle is 0.0 … **0.9896** (worst `rvq_residual_s14`,
with s08 at 0.9475, s15 at 0.9307 and s07 at 0.9278 behind it).

**Both figures in that last sentence were wrong until 2026-08-14 and the upper
one mattered.** It read "4.7e-07 … 9.3e-05, where upstream-f32 vs the bf16
oracle is 0.0 … 0.41". 0.41 is `transformer_l7`'s 0.4054 — the maximum over
the **non-RVQ** taps only — so it understated the real bf16-scale spread by
2.4x, and a later plan sizing a bf16-derived bound off this sentence would have
set it that much too tight. `4.7e-07` appears nowhere in the grid at all; the
smallest non-zero per-stage worst is `8.021e-07` (`transformer_l1`). Every
number here is now transcribed from `codec.chain.per_stage` in
`tests/tolerances/qwen3-tts.json`, which is the file that carries them.

**The five-way decomposition demonstrably localizes, which is its point.**
Shifting the codec track one frame moves `codec` 2.232e-03 → 7.945e-01 and
leaves `text` where it was; swapping the text track's two halves moves `text`
5.628e-03 → 5.159e-01 and leaves `codec` where it was. The summed block moves
in both cases and says nothing about which track was wrong.

**One consumer rule that is easy to get wrong from the far end:** the two f32
tracks must be summed and **rounded back through bfloat16** before comparison
against `icl_embed.f32`. Upstream computed the sum in bfloat16 and rounded
once; a raw float32 add is the exact real sum and deviates by construction, by
`2**-8` = 0.00390625 — bfloat16's *unit roundoff*, not the `2**-9` half-ulp a
first guess produces, and a tolerance set at `2**-9` fails. The figure is
published in a `numerics` block in both `alignment.json` and
`prompt_conventions.json`; take it from there.

### The ICL prompt's conventions, located by AST rather than cited

`prompt/prompt_conventions.json` (Task 2) records the prompt's construction
with every span **located by walking the AST of the installed `qwen_tts`
package at run time** and its text read straight out of that file — so a
citation there cannot go stale: if upstream moves or reshapes a construct, the
locator fails and no artifacts are written. Each claim also carries how it was
checked (`executed` = upstream's own code re-run and compared bitwise,
`observed` = read out of a running frame, `ast` = the structure asserted).

| element | rule | `modeling_qwen3_tts.py` |
| --- | --- | --- |
| text track | `text_projection(text_embeddings(cat([ref_id, text_id])))` then `cat([., tts_eos_embed])`; `T1 = len(ref_id) + len(text_id) + 1` | `:1978-1981` |
| codec track | per reference frame, the **sum** of 16 embeddings — group 0 from the talker's own codec table, groups 1..15 from the code predictor's; one `codec_bos_id` row prepended; `T2 = 1 + ref_frames` | `:1983-1998` |
| alignment | `if T1 > T2`: block is `text_embed[:, :T2] + codec_embed`, trailing is `text_embed[:, T2:]` (**truncate**); else the text track is padded with `tts_pad_embed` to `T2` and trailing is a bare `tts_pad_embed` (**pad**) | `:2015-2019` |
| placement | the block is concatenated **after** the existing prefix, replacing the single `tts_text_first_token` position the non-ICL branch appends | `:2197`, against `:2200-2202` |
| speaker slot | `speaker_embed` is inserted **regardless of mode** — ICL *adds* to the x-vector path and does not substitute for it | `:2166-2172` |
| slices | `text_id = input_id[:, 3:-5]`, `ref_id = ref_ids[index][:, 3:-2]` | `:2189-2196` |

The reference transcript gets **its own turn wrapper**, which is why the two
slices differ: `_build_ref_text` is
`<|im_start|>assistant\n{text}<|im_end|>\n` (`qwen3_tts_model.py:272-273`) with
no trailing `<|im_start|>assistant\n`, where the target text's wrapper carries
one (`:269-270`). **Exactly one input separates wrap-then-slice from bare
tokenization: a leading newline.** `"\nHello"` wrapped and sliced gives
`[9707]`, identical to `"Hello"`; bare gives `[198, 9707]`. A leading space,
tab, U+00A0, U+3000 and `é` all fail to differ. That matters because under bare
tokenization **all three oracle cases still pass** — the pinned transcripts
start with letters — so the comparison that looks strongest cannot discriminate
here, and the boundary assertion is the only check that can. It is stated in
the test so that nobody "simplifies" it to a space.

**Both arms are reached, and which one a case takes is read out of the
executed `return` rather than recomputed.** A block of `T2` positions is
consistent with either arm, so inference from lengths is not sound, and the
trailing schedule is the only discriminator. The ten-case table is under "The
alignment arms, measured across all ten ICL cases" above: 8 pad, 2 truncate,
with `base-ref-min` reaching truncate by making `T2` small and `base-text-long`
by making `T1` large. The trailing schedule is compared both as token ids and
as `trailing.f32`; perturbing it alone moves `trailing.f32` 2.333e-03 →
5.187e-02, past the gate, while every other artifact stays bit-identical.

### Profile Schema: a second kind, and no version bump

Schema `qwen3-tts-voice-clone`, still **version 1**. Two `kind` values now
ship: `"x-vector"` (Plan 2) and **`"icl"` (Plan 3), landed without a schema
version bump** — the envelope discriminates on the in-payload
`synthesize.voice_profile.kind` string rather than on `schema_version`, which
is the entire point, and what it buys is that every Plan 2 Profile stays
loadable under a Plan 3 build. Plan 2's own record predicted this shape and it
held.

An ICL envelope carries three tensors where an x-vector envelope carries one:
`profile.x_vector`, `profile.codes` (the `[16, T]` reference grid in
group-fastest order) and `profile.reference_text_ids`. **The third discloses the
reference transcript's content and the project says so** — byte-level BPE decodes
the ids back to text, so whoever holds the Profile can read what the transcript
said. Recovery is not byte-exact: the ids come from wrapping the transcript in
the reference turn, tokenizing, and slicing a fixed count off each end, and that
map is not injective — measured on the shipped Base package's vocabulary,
`"\nHello"` and `"Hello"` both reduce to the single id 9707
(`src/arch/qwen3-tts/bpe.h`). What a holder recovers is the content, up to
whitespace at its boundaries, and that is disclosure either way.
That declaration is the design's D5 and it now lives in
`docs/voice-conditioning.md`, which is the document a Profile's recipient
would read. Profiles still never carry enrollment audio. The divergence from
OmniVoice — which serializes the transcript *string* and re-tokenizes on load —
is deliberate and recorded at `src/arch/qwen3-tts/profile.h`, so that the next
reader does not "fix" the two families into consistency.

The clone mode is **fixed at preparation, not at synthesis** (D4), and
`create_qwen3_tts_profile_from_reference` is the one and only selector: the
transcript's *presence* chooses. Absent → the Plan 2 x-vector path unchanged;
present → tokenize through the reference turn and build an ICL Profile.

### The six family-conditioned points, and which four Plan 3 changed

The same six places Plan 2's own table enumerates — five dispatch arms in
`src/voice-profile.cpp` plus one synthesis-time consumption point in
`src/synthesize.cpp`. Plan 3 changes four, and they are not the same four in
the same way:

| point | Plan 2 | Plan 3 |
| --- | --- | --- |
| A — `synth_voice_profile_create_from_reference` | Changed: x-vector arm, refusing a transcript by name | **Changed** (Task 10): the transcript becomes the optional mode selector; `reference_language` validated by shape then against the package's declared languages |
| B — `synth_voice_profile_create_from_description` | Untouched | **Untouched** — Description Text is Stage 3's |
| C — `synth_voice_profile_create_random` | Untouched | **Untouched** — no family implements Random Seed |
| D — `synth_voice_profile_load_from_memory` | Changed | **Changed** (Task 9): loads either kind out of untrusted bytes, taking the whole `HParams` so it can bound codes and text ids per side |
| E — `synth_voice_profile_serialize` | Changed | **Changed** (Task 8/9): branches on the payload's own `CloneMode`; both writers refuse a payload naming the other kind, so a wrong branch is an error rather than a silent downgrade to a valid `kind="x-vector"` envelope |
| 6th — synthesis-time consumption | Changed: x-vector substitution | **Changed** (Task 11): the blanket ICL refusal at `synthesize.cpp` is gone and the reference grid, text ids and x-vector all reach the prompt |

### What the untrusted-envelope work found

Task 9's tamper matrix is worth one paragraph because two of its findings are
general. A **Critical** was fixed structurally rather than patched: the codes
and ids vectors were sized from untrusted counts *before* the truncation
guard, so a 1 KiB hostile Profile drove a **1.48 GiB** RSS delta.
`tensor_range_fits` now runs *inside* `i32_tensor_elements`, which refuses to
return a count until the bytes behind it exist — an unchecked count is
*unobtainable* rather than merely checked. The status assertion could not have
caught it: the status is `INVALID_ARG` both before and after, because the
allocation completes and then errors cleanly. The new arm asserts a peak-RSS
delta (+0 KiB after; +1,550,020 KiB with the guard removed).

And **three mutual-masking pairs** exist in the prescan/loader path where
deleting either check of a pair changes nothing and deleting both lets a sealed
forgery load with `status = 0` and a non-null payload. Single-deletion
inversion cannot see that class at all; each masked rule now names its partner
in the file where the next person will hit it.

### What Plan 3 does not deliver

- **Any movement of the Validation Level.** It stays `port_validated`, and
  `quality_evaluation` stays `not_run` per ADR 0017.
- **A Listening Audit for ICL.** None has run. The 2026-08-13 audit recorded
  under "Listening Audits" below was **x-vector-only** and says so; it covers
  neither this mode nor CUDA. One is scheduled for Plan 4's ship-prep phase.
  Note what follows from that, rather than what an earlier revision of this
  bullet claimed. It argued that a listener could not have caught the alignment
  defect because the misaligned audio "produces fluent speech in approximately
  the right voice" — a listener's judgement, asserted in the same bullet that
  says no listener ever heard it. **Nobody knows whether a listener would have
  caught it.** What is known is that the misaligned run returned `SYNTH_OK`
  with finite, non-silent audio and a 23% shorter duration, and that the
  numerical gates are the only thing in the tree that did catch it. Both facts
  stand on their own; neither needs an audible claim.
- **Quantization Profiles and the CUDA Execution Backend.** Plan 4's. No
  `F16`, `Q8_MIXED` or `CUDA` cell was added to the tolerance grid and **no
  performance number is claimed by this plan.** The codec encoder is
  convolution-heavy, which is the shape that produced another family's
  conv-exempt quantization policy; whether that transfers is a measurement for
  Plan 4, not an analogy to import here.
- **Description Text.** Stage 3's, on `qwen3-tts-12hz-1.7b-voicedesign`.
  `SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT` stays unadvertised, and so does
  Random Seed.
- **The CLI path.** `examples/cli/` is untouched, for the reasons under "The
  CLI's position" above: adding an audio reader is a cross-family slice, not a
  qwen3-tts increment.
- **Any diagnosis of the ICL output-length pathology, or a shorter failure.**
  Pairing a reference clip with a transcript that does not match it has been
  measured, on this port, running synthesis to the 2048-frame ceiling — about
  eight minutes, non-OK, zero audio. **Reaching the ceiling is not the only
  outcome a mismatch produces**: on the reference implementation the same kind
  of mismatch collapsed on four of five seeds — 9, 8, 12 and 4 frames — and
  reached the ceiling on the fifth, and a collapse hands back a short clip
  rather than a status a caller can key on. It is an ordinary caller mistake
  (any imperfect ASR transcript is one). What Plan 3 added is **mitigation, not
  a diagnosis and not a fix**: one static error string, so that the limit stop
  points a caller at the inputs to check instead of returning a bare status.
  **Shortening it needs a lower ICL ceiling or a run-away detector, and neither
  was built.**

  **The provenance, corrected on 2026-08-14 — an earlier revision of this
  bullet had it backwards and the correction matters for Plan 4.** It said the
  pathology "was measured on the PyTorch reference implementation; this port
  was not compared on it, so it is upstream's behaviour rather than a port
  defect." That is the 9-frame anomaly's caveat, borrowed and stapled to a
  different measurement. **The transcript-mismatch hang has only ever been
  measured on THIS PORT**, through the public C seam — 475.9 s of CPU,
  `SYNTH_ERR_OUTPUT_LIMIT` at `kDefaultMaxFrames = 2048`, zero audio, at
  `CMAKE_BUILD_TYPE=Release`. Upstream was never run on a mismatched
  transcript. So nothing establishes this as upstream's behaviour, and nothing
  clears the port of it either; the shared-cause link to the 9-frame case is
  recorded elsewhere in this document as **a hypothesis, not a finding**, and
  promoting it here would send Plan 4 hunting in the wrong implementation. The
  9-frame work *is* oracle-side (`scripts/measure_qwen3_tts_icl_reference_length.py`),
  and its section above is where the `trim_seconds` driver flag a port
  comparison would need is recorded.
- **A port-side answer on the 9-frame case.** The anomaly, open since Plan 1,
  **is adjudicated**: it reproduces exactly, and it **co-occurs** with the
  runaway under one input — five seeds, four collapses and one runaway. That
  they share a mechanism is a hypothesis, not something this measured. It is
  **not** a monotone function of reference length; that explanation was raised
  and refuted inside this plan. But the frame-count half of the port
  comparison is distributional and remains unrun.
- **Publication.** Separate, and requiring jiangzhuo's confirmation at the
  time.

## Listening Audits

Three have run for this family. Two on **2026-07-29** covered Stage 1's
CustomVoice variant and produced the `listening_audit: no_obvious_regression`
that `scripts/hf_cards/qwen3-tts-12hz-0-6b-customvoice.yaml` carries; the
**2026-08-13** audit is Stage 2's, on the Base variant's x-vector clone path,
and is the first in this repository to put a question about resemblance to a
listener. All three are one listener, non-statistical, and none moves
`quality_evaluation` off `not_run` or changes any Validation Level.

### Base, the x-vector clone path (Stage 2 Plan 2, 2026-08-13)

**Verdict, jiangzhuo, 2026-08-13: `no_obvious_regression`.** Recorded under
`order_seed 20260813`, listener count 1.

| question | cases | result |
| --- | --- | --- |
| Q2, port vs oracle, blind A/B | `agree-en-1`, `agree-en-2`, `agree-ja-1`, `agree-zh-1` | no obvious difference on all four |
| Q1, reference-duration bounds, labelled | 1 s, 3 s, 10 s, 30 s | usable at every one |
| Q1, sub-minimum, labelled | 0.5 s | refused by the library (`voice_profile.reference_too_short`) |
| Q3, resemblance to the source speaker, labelled | 1 source/clone pair | same speaker |

**Method.** A blind A/B web page. Each of the four Q2 pairs is the same
reference clip and sentence rendered twice: once by this port through the
public C interface on CPU, once by the pinned upstream PyTorch implementation
on CUDA at bfloat16. Both sides were driven in **x-vector-only mode**, which is
the only mode this port implements. A/B order was randomized per pair from
`order_seed 20260813`; two of the four were swapped. The duration sweep and the
source-versus-clone pair were **labelled rather than blind**, because in both
the label is the question: "is 30 s of reference usable" and "does this clone
sound like that source" cannot be asked of an unlabelled pair. Audio was 24 kHz
mono 16-bit, converted from each side's native 32-bit float. The generated
audio lives under a gitignored build directory and the page itself was not
committed, so this table and the family record are the audit's artifact.

**What it establishes.**

- `no_obvious_regression` against the reference implementation in x-vector
  mode, across four cases and three languages.
- The shipped reference-duration bounds produce usable speech at 1 s, 3 s,
  10 s and 30 s, and the sub-minimum case is refused as designed rather than
  synthesized badly. This closes the open item under "Measured
  reference-duration bounds" above.
- The 30 s / 9-frame anomaly recorded in that same section does not belong to
  the shipped path: regenerated in x-vector mode the case gave 46 codec
  frames, in line with the sweep's other durations. The 9-frame figure came
  from a transcript-assisted dump. The anomaly is real and was Plan 3's;
  **it has since been adjudicated** — see "The 9-frame anomaly, adjudicated
  2026-08-14" above.
- One listener's judgement that a clone is recognisably the same speaker as
  its source. This is the first such evidence in this repository.

**What it does not establish.** Any quality, naturalness or comparative-ranking
claim — ADR 0017 defers Quality Evaluation, and a Listening Audit does not move
the Validation Level. The audit *is* a comparison, and a bounded one: this port
against its own reference implementation, four cases, one listener. What stays
out of reach is ranking this model against any other, which is what
`comparative` means everywhere else in this project's documents. Any speaker-similarity metric; the resemblance finding is one
listener, one source clip, one clone, and is evidence rather than a property of
the port. Anything about transcript-assisted (ICL) mode: the port did not
implement it when this audit ran, and **it still has no Listening Audit** now
that Plan 3 has built it — one is scheduled for Plan 4's ship-prep phase.
Anything about CUDA for the new graphs, which have only ever run on CPU — the
port side of every pair here was CPU.

### CustomVoice, port vs oracle and Q8_MIXED vs F16 (Stage 1, 2026-07-29)

Two passes on the same day, both one listener, both `no_obvious_regression`.
The first offered eight replayed cases blind against the PyTorch reference plus
five natively-sampled clips on CPU and CUDA — the path replay does not cover.
The second compared six natively-sampled Q8_MIXED clips against F16, natively
sampled rather than A/B because on the replay path the two profiles' waveforms
are byte-identical and only a run that draws its own codes can show what
quantization costs. Full method, the two invalidated attempts that preceded the
second pass, and the duration observation the listener cleared are in
`reports/porting/qwen3-tts/qwen3-tts-12hz-0-6b-customvoice/_porting-log.md`
under its 2026-07-29 entries.

## Stage 3: VoiceDesign Package, Task 6

Executed 2026-08-18. Intake, conversion, and a package that loads -- the first
rung of Stage 3, and the first time this family's converter or loader ran
against a checkpoint whose talker and code predictor disagree in width. The
Reference Model Variant Ladder said VoiceDesign "necessarily moves to the 1.7B
width" (see above); this task is where that stopped being a sentence about
parameter count and started being a sentence about tensor shapes.

### License, verified from the upstream model card

```
$ curl -sL "https://huggingface.co/api/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['id'], (d.get('cardData') or {}).get('license'), d['sha'])"
Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign apache-2.0 5ecdb67327fd37bb2e042aab12ff7391903235d3
```

Matches the pinned revision exactly. The card's own prose (fetched at that
revision) carries no additional restriction beyond the `license: apache-2.0`
frontmatter -- confirming the "Why Qwen3-TTS" section's four-repository claim
for the fourth repository specifically, rather than inheriting it from the
other three.

### Pinned digests

| Artifact | sha256 |
| --- | --- |
| `config.json` | `aecd2cc4c1fe9edef1cb7ca7c401685a43879ad43f3f9e883f1c6760b61731e0` |
| `model.safetensors` | `391e8db219f292c515297cdceeb43e4eae67cdde35fa57e79a6a8a532fca0522` |

`speech_tokenizer/model.safetensors`, `vocab.json` and `merges.txt` are
byte-identical to the digests the Base and CustomVoice manifests already
record -- measured, not assumed, by hashing this checkpoint's own copies --
which is expected: `Qwen3-TTS-Tokenizer-12Hz` and the BPE vocabulary are shared
across the family's variants rather than re-shipped per rung. The Golden
Manifest, `tests/golden/qwen3-tts/qwen3-tts-12hz-1-7b-voicedesign.manifest.json`,
carries all five digests, none of `qwen3-tts-12hz-0-6b-base`'s two
`reference-audio` artifacts (this variant takes no recording), and zero cases
-- this task's own step removes every case the copied Base manifest carried.
The manifest stays at zero cases through Task 7 too: that task gates the
empty-instruct rung with an integration driver and a tolerance cell outside
the manifest rather than adding a case to it.

**Superseded, 2026-08-19, Stage 3 Plan 2 Task 7:** the manifest no longer
carries zero cases -- see "The manifest's thirteen cases, Plan 2 Task 7" far
below, where it landed thirteen and lost `suite_status: "incremental"`
entirely. Left standing rather than rewritten, per this file's own
convention: a reader relying on this paragraph during Stage 3 Plan 1 or Plan
2's first six tasks was correctly informed at the time.

### Measured package size against the design's estimate

**4,295,891,904 bytes (4.30 GB decimal, 4.00 GiB) against the design's ≈4.3 GB
estimate** -- 659 tensors, 404 BF16 + 255 F32, converted in one pass with no
manual intervention once the catalog gap below was closed. 659 is exactly
CustomVoice's tensor-shape count (657, this family's other no-speaker-encoder
variant) plus the two tensors the width mismatch below requires.

### A genuine architecture gap, found and closed, not converted around

The first conversion attempt succeeded -- the Python converter carries every
tensor under a known prefix with no name-based filtering of its own, so
`small_to_mtp_projection.weight`/`.bias` rode along without any converter
change being needed. **Loading the result refused**, and the refusal was not a
missing catalog entry for an unfamiliar name; it was a refusal
`src/arch/qwen3-tts/catalog.cpp` had carried on purpose since before any
package needed it:

```
qwen3-tts: the code predictor is 1024 wide against the talker's 2048, which
needs an input projection this package does not carry
```

`CodePredictorWeights` (`code-predictor.h`) already declared
`input_projection`/`input_projection_bias` pointers, and
`code-predictor.cpp`'s `build_code_predictor` already applied them whenever
non-null -- both written and comment-anticipated during Stage 1, before a
1.7B rung existed to exercise them. The graph side of this family's autoregressive
runtime needed **zero new code**. What was missing was entirely in the
metadata/catalog layer, and it was three assumptions deep, each invisible for
exactly the same reason: every package converted before this one (Base and
CustomVoice, both 0.6B) happened to declare the same value for the talker and
the code predictor on every axis that could have diverged, so a catalog that
silently reused the talker's number for the predictor's own geometry was
indistinguishable from a correct one.

1. **`intermediate_size`.** `CodePredictorParams` had no field of its own; the
   catalog sized the predictor's MLP from `hparams.talker.intermediate_size`
   (3072 == 3072 at 0.6B, 3072 != 6144 at 1.7B). Fixed by giving
   `CodePredictorParams` its own `intermediate_size`, read from a new
   `synthesize.qwen3-tts.code_predictor.intermediate_size` metadata key that
   `scripts/convert-qwen3-tts.py` now writes from the checkpoint's own
   `code_predictor_config.intermediate_size` rather than inheriting
   `talker_config`'s. **Optional at read time** (`GgufMetadata::has`), falling
   back to the talker's value when the key is absent, so every package
   converted before this key existed -- the published CustomVoice and Base
   artifacts included -- still loads unchanged.
2. **The width bridge.** `p.hidden_size != hparams.talker.hidden_size` used to
   be an unconditional refusal. Now it resolves
   `talker.code_predictor.small_to_mtp_projection.{weight,bias}` into the
   pointers `build_code_predictor` already knew how to use --
   `torch.nn.Linear(talker_config.hidden_size, config.hidden_size, bias=True)`
   in the reference (`Identity()` when the widths agree, which is why no
   0.6B package carries the tensor), confirmed by reading
   `Qwen3TTSTalkerCodePredictorModelForConditionalGeneration.__init__` and
   `.forward` in the pinned `qwen_tts` checkout rather than guessed from the
   tensor's name.
3. **`codec_embedding`'s width, which is not `lm_head`'s.** The two read like a
   matched embedding/head pair and are not: `lm_head` is
   `Linear(config.hidden_size, vocab_size)`, sized by the predictor's own
   width, but `codec_embedding` is built by
   `Qwen3TTSTalkerCodePredictorModel.__init__(self, config, embedding_dim)`
   with `embedding_dim=talker_config.hidden_size` passed in from the
   *outside* -- a second, separate constructor argument the predictor's own
   `config.hidden_size` never enters. `model.cpp`'s per-step wiring confirms
   why: a `codec_embedding` lookup is concatenated or summed with the
   talker's own hidden state (`t_hidden`, talker-width) *before* that
   combined value reaches `build_code_predictor`'s input-projection step, so
   it has to already be at the talker's width to be summable. The catalog
   sized it at `p.hidden_size` (predictor's own) and was wrong at 1.7B in the
   same invisible way as the first two.

Each of the three was measured against the pinned reference implementation,
not inferred from the shape error's message, and each has synthetic coverage
in `tests/qwen3_tts_catalog_test.cpp`
(`check_narrower_code_predictor_resolves_with_projection`, plus
`check_real_voicedesign_package_count` pinning the real package's 659 against
`expected_tensor_count`) alongside the negative case that already existed
(`check_rejections`' "no package carries one", which continues to refuse a
narrower predictor whose package omits the projection tensor). Verified
against three real local packages, not only against synthetic fixtures: the
two 0.6B packages this project has published or locally converted --
CustomVoice and Base (including its 2026-08-18 re-cut below) -- reload
unchanged after every change in this section, confirming no regression on the
coincident-width path both of them take; VoiceDesign itself loads for the
first time, confirming the fix. All three checked through the public C API
(`synth_model_load` and `synth_model_get_voice_profile_capabilities`), not
merely inferred from the catalog test passing.

### `generate_custom_voice` silently discards `instruct` for any 0.6B model

Worth its own paragraph, because it will mislead someone who reads only the
published CustomVoice package's card. `qwen_tts/inference/qwen3_tts_model.py:799-800`:

```python
if self.model.tts_model_size in "0b6": # for 0b6 model, instruct is not supported
    instruct = None
```

`tts_model_size` is `"0b6"` for every 0.6B checkpoint and `"1b7"` for
VoiceDesign (confirmed from both configs' own `tts_model_size` field), so this
line is a **membership test that happens to work for the one string it was
written against** rather than a documented size gate: `"0b6" in "0b6"` is
`True`, `"1b7" in "0b6"` is `False`. The consequence is asymmetric and
easy to miss: calling `generate_custom_voice` on the published
`qwen3-tts-12hz-0-6b-customvoice` package with a non-empty `instruct` argument
returns `SYNTH_OK` and audio, exactly as if the instruction had been honoured,
because nothing downstream reports that it was silently cleared first. The
call *looks* like Description Text working on a 0.6B package. It is not
running at all. Description Text is VoiceDesign's alone -- upstream's
`generate_voice_design` is the only entry point that reaches it, and
`SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT` is reserved for the one variant that
carries the tensors for it, once the RUNTIME half of that capability is wired
(2026-08-18 correction below: the bit itself is not yet published by any
variant, this package included) -- and this line is upstream's, not a defect
in this port -- but a caller who tests "does an instruction change the voice"
against CustomVoice and observes no audible change would reasonably conclude
the feature is broken rather than absent.

### The Golden Manifest schema did not anticipate a suite built across tasks

A second gap, in test infrastructure rather than the runtime: this plan's
manifest is deliberately committed with 0 cases now and stays there through
Task 7 -- Plan 2 is what gives it its first case and the rest -- but
`docs/schemas/synthesize-golden-manifest-v1.schema.json` required
`cases.minItems: 12` (every previously-committed manifest's own floor), and two
family-independent checks in `tests/python/test_golden_manifests.py` assumed
every manifest already had a first case (`test_upstream_examples_are_present`)
or a matching entry in its shared tolerance file's `variants` map
(`test_tolerance_case_count_matches_manifest`). All three would have failed
`synthesize-golden-manifest-contract` for this manifest alone, without
touching any other family's data.

**The first fix was too wide, and a Task 6 review round found it.**
`cases.minItems` was widened to 0 for every manifest the schema governs, not
only the one that needed it -- documented at the time in the schema's own new
`description` field, but a real regression rather than a wording nit:
`cases.minItems: 12` is `docs/port-validation.md`'s own Confirmed contract, and
the repo-wide zero let a truncated manifest through, demonstrated by
validating a 3-case copy of `vits-ljspeech`'s own committed manifest against
the widened schema, which the unwidened schema had rejected. Replaced with a
narrow opt-in instead: a top-level `if`/`then`/`else` keyed on a new optional
`suite_status` field relaxes `cases.minItems` to 0 only when a manifest
declares `suite_status: "incremental"` -- which only
`qwen3-tts-12hz-1-7b-voicedesign` did, at the time. Every other manifest omits
the field and stays at the 12-case floor, re-validated against the same
truncated VITS copy, which is rejected again. Both Python checks still
`continue` specifically for a manifest whose `cases` array is empty -- narrowly
enough that the moment this manifest gains its first case (Plan 2), both
checks re-engage on it exactly as they do on every other manifest.

**Superseded, 2026-08-19, Stage 3 Plan 2 Task 7.** Two things in the paragraph
above changed, both covered in full under "The manifest's thirteen cases, Plan
2 Task 7" far below: first, `qwen3-tts-12hz-1-7b-voicedesign` gained its
thirteen cases and lost `suite_status` entirely, so as of this task NO
manifest carries the flag -- the set of manifests permitted to is empty, named
explicitly as `SUITE_STATUS_ALLOWLIST` in
`tests/python/test_golden_manifests.py`. Second, "both Python checks still
`continue` ... for a manifest whose `cases` array is empty" was itself a
latent defect this task closed, not merely a description that stopped
applying: those two `continue` guards read `if not manifest["cases"]`, never
`suite_status` itself, so nothing before this task actually tied the schema's
own waiver to these two exemptions -- a manifest could have carried the flag
WITH cases, or had zero cases WITHOUT the flag, and neither shape would have
been caught. Found twice independently: while drafting this plan, and again by
an external review of the sibling Stage 3 Plan 2 PR that demonstrated it live
against a 3-case copy of `vits-ljspeech`'s own manifest carrying
`suite_status: "incremental"` and validating clean. Closed by a new guard,
`test_suite_status_matches_an_incrementally_built_manifest_exactly`, that
binds both directions at once (the carrying set equals the named allowlist;
a manifest carrying the flag has zero cases, one without it has at least
twelve), and by rewriting both `continue` guards to key on `suite_status`'s
own value instead of `cases` being empty as a proxy for it.

Task 6's own commit made no change to `tests/tolerances/qwen3-tts.json` or to
`tests/python/test_tolerance_coverage.py`; Task 7 is what later added this
variant's own tolerance entry. At Task 6's commit, though, the manifest's
`reference.runner` named the already-committed, family-generic
`scripts/dump_reference_qwen3_tts_pytorch.py` rather than a
VoiceDesign-specific script -- which let `test_runner_is_committed` pass
because *some* file existed at the named path, not because the named path was
right. It was not: this variant's real oracle runner is
`scripts/dump_reference_qwen3_tts_voicedesign.py`, written in Task 7 and
already what the tolerance grid names. The manifest is corrected to match.
Verified against the actual test suite, not merely reasoned about: both
`tests.python.test_golden_manifests` (24/24) and
`tests.python.test_tolerance_coverage` plus its inversion suite (8/8) pass
with the corrected manifest committed, and the two pre-existing
gitignored-VITS-artifact failures (`synthesize-python-api-wheel-test`,
`synthesize-vits-python-unit`) are unchanged.

### Load result

```
$ ./build/bin/synthesize-cli --model .../qwen3-tts-12hz-1-7b-voicedesign-BF16.gguf --help
```

carries no `--info` flag -- capability-snapshot printing is the CLI's own
cross-family slice, out of this plan's scope by the design's own File
Structure, not merely undiscovered here. Verified instead through a small,
uncommitted probe driving the public C API directly
(`synth_model_load` / `synth_model_get_preset_voice_count` /
`synth_model_get_voice_profile_capabilities`), matching the pattern
`tests/qwen3_tts_base_load_real.cpp` already exercises against the real Base
package:

| Field | VoiceDesign (as measured at Task 6) | Design's prediction |
| --- | --- | --- |
| `preset_voice_count` | 0 | zero Preset Voices |
| `profile_source_flags` | `0xa` = `SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT \| SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE` | Description Text + Serialized Profile |
| `SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO` | absent | absent |
| six `reference_*` limits | all 0 | left at zero |
| `profile_schema` | `qwen3-tts-voice-design`, version 1 | -- |

Every field matched the Interfaces section's prediction exactly.

**Erratum, 2026-08-18 -- jiangzhuo's ruling after the final whole-branch review
found the prediction this table matched was itself wrong.** The `profile_source_flags`
and `profile_schema` rows above are what Task 6 measured, and what the design
predicted, but not what the RUNTIME should have reported:
`synth_voice_profile_create_from_description` (`src/voice-profile.cpp`) had no
Qwen3-TTS arm at that point, so the capability query was advertising a source
the seam then refused -- the same trap this family's own transcript-assisted
(ICL) mode was withheld from advertisement for the whole of Stage 2 Plan 2 to
avoid, and `SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE` cannot survive alone
either, since this package has no speaker encoder for
`load_qwen3_tts_profile_from_memory` to ever accept a Profile against. The
corrected measurement, taken against the same probe after
`fill_voice_profile_capability` was fixed:

| Field | VoiceDesign (corrected) |
| --- | --- |
| `preset_voice_count` | 0 |
| `profile_source_flags` | `0x0` (none) |
| `SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO` | absent |
| six `reference_*` limits | all 0 |
| `profile_schema` | null, size 0, version 0 |

This is the same all-zero "no runtime Voice Profile support" shape a
CustomVoice package reports, per `docs/c-interface.md`'s rule that Serialized
Profile's schema fields are null/zero exactly when that bit is clear, for a
different reason than CustomVoice's: CustomVoice carries no ProfileContract at
all, while this package's `synthesize.profile.*` metadata is read and
validated in full at load time and the schema string above (`qwen3-tts-voice-design`,
version 1) is still there in `HParams` -- simply not surfaced through the
public capability query until a later plan wires `create_from_description`.
The PACKAGE's own declared `synthesize.voice.profile_sources` is unaffected by
this correction and still names `description-text`; Task 3's loader and its
cross-checks are untouched.

**Second correction, 2026-08-19, Stage 3 Plan 2's Task 5 -- the table
immediately above is itself superseded.** Round 1 of that task's own review
re-ran the same probe against the same real package after Task 5 landed and
measured the table this section originally reported, unchanged from Task 6:

| Field | VoiceDesign (Task 5 re-measurement) |
| --- | --- |
| `preset_voice_count` | 0 |
| `profile_source_flags` | `0xa` = `SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT \| SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE` |
| `SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO` | absent |
| six `reference_*` limits | all 0 |
| `profile_schema` | `qwen3-tts-voice-design`, version 1 |

Both reasons the first correction gave are closed: Task 2 gave
`create_from_description` a Qwen3-TTS arm, and Task 3 made a serialized design
Profile loadable. The all-zero "corrected" table above held only for the
interval between the final whole-branch review and Task 5's own close, and is
not rewritten; a reader today wants this table, not that one.

## Stage 3: VoiceDesign Package, Task 7

Executed 2026-08-18. Plan 1's completion gate -- a prefill built at empty
instruct compared against the oracle, with the fault-injection figure proving
that comparison can fail -- closing Plan 1 alongside Task 6's load result
above. No new graph code: the empty-instruct path needs no instruct block
(design decision D3), so this task's whole job is building the comparison
Task 6 deferred, not extending the runtime.

### The oracle dumper

`scripts/dump_reference_qwen3_tts_voicedesign.py`, modeled on
`scripts/dump_reference_qwen3_tts_base.py`, drives `generate_voice_design`
against the pinned checkpoint with `text="Qwen3-TTS is awesome!"`,
`instruct=""`, `language="English"`, pinned seed and greedy decoding matching
the Base dumper's own determinism settings. It dumps three raw float32
artifacts to `reports/porting/qwen3-tts/qwen3-tts-12hz-1-7b-voicedesign/oracle/`
-- **not** under `build/goldens/qwen3-tts/...`, the location every other
oracle dump in this family and VITS's uses (`docs/testing.md` states that
convention); this variant's own usage example and the task's brief put it
here instead, and `.gitignore` gained a dedicated `/reports/porting/**/oracle`
rule for it (Task 7 Fix Round 1) rather than relying on the already-ignored
`build/` tree:

- `prefill.f32` -- the assembled talker input embeddings, `[18, 2048]`
- `codes.i32` -- the talker's emitted codes
- `waveform.f32` -- the decoded audio

Only `prefill.f32` is compared against in this task. `codes.i32` and
`waveform.f32` are dumped for the record and read by nothing here -- a
synthesis-level comparison needs the instruct prompt block and the codec
decoder wiring Plan 2 builds. `tests/tolerances/qwen3-tts.json`'s
`qwen3-tts-12hz-1-7b-voicedesign` entry states this directly:
`"status": "partial-measurement-prefill-only"`.

### The driver, and the completion gate it did not enforce at first

`tests/qwen3_tts_voicedesign_prefill_real.cpp` loads the real package, builds
a prompt with `has_speaker = false` and no instruct tokens (Task 5's layout,
the branch section 5.3 calls "a branch that has never run"), flattens it, and
writes the prefill embeddings out for comparison against the oracle dump.

**Task 7 Fix Round 2 corrected a gap a review found in the completed task,
not in this record after the fact.** The driver's plan text said it "asserts
nothing," which was approved and, taken literally, meant the completion gate
enforced nothing: a fault-injected build printing a 365x tolerance breach and
a shape mismatch still exited 0. Put to jiangzhuo because a finding that
contradicts approved plan text is not a reviewer's or an implementer's to
overrule alone; ruling was that the finding governs, and the plan doc's Task
7 Step 3 carries the erratum rather than a silently rewritten sentence. The
driver now takes the committed `max_relative` bound from
`tests/tolerances/qwen3-tts.json` as an optional argument -- read at CMake
configure time and passed to `add_test`, the same shape
`synthesize-qwen3-tts-icl-prompt-real` already has -- and exits non-zero when
the measured p95 exceeds it or the shapes disagree. Registered as
`synthesize-qwen3-tts-voicedesign-prefill-real`, gated behind
`-DSYNTH_BUILD_INTEGRATION_TESTS=ON` and a real package at
`SYNTH_QWEN3_TTS_VOICEDESIGN_TEST_MODEL` plus the oracle dump above; it never
enters `synthesize-check-unit`.

### Measured, and the fault that proves the gate

Compared with `reconstruction_p95_relative` (`tests/qwen3_tts_percentile.h`),
this family's existing p95-over-positions statistic:

| | value |
| --- | --- |
| positions | 18 |
| hidden_size | 2048 |
| codec_offset | 3 |
| external_speaker_index | -1 |
| observed `p95_relative` | 0.00269 -- under half of one bf16 ulp |
| committed `max_relative` | 0.01 |
| headroom | 3.72x |

`0.01` is deliberately tighter than this family's own stated widening rule for
a probe of this shape (`prompt.icl_embed`'s "5x the measured deviation" would
give 0.01345): this is the probe's first and only measured case, with no
ICL-style two-track summing to accumulate a second bf16 rounding on top of the
weight tables' own quantization, and `0.01` sits close to 2.5 bfloat16
unit-roundoffs -- a consistent story from a second direction, not a competing
one.

**Fault injection, per Task 7 Step 4 and this task's own plan brief**:
`src/arch/qwen3-tts/talker-host.cpp:159`, `if (request.has_speaker)` changed
to `if (true)` -- Task 5's exact branch, inverted, retaining a
codec-vocabulary speaker slot even though `request.has_speaker` is false.

| | value |
| --- | --- |
| faulted `p95_relative` | 0.983659 |
| ratio | 365.7x |
| faulted position count | 19 (against the oracle's and the clean port's 18) |

The fault does not only perturb values at a fixed shape -- it changes the
position count. The extra codec-vocabulary slot pushes the codec-prefix loop
from 5 iterations to 6, shifting every position from index 7 onward relative
to the clean, correct layout; the driver reports `compared_positions: 18`
beside `shapes_match: false`, so the shape mismatch is at least as strong a
failure signal as the number itself. Reverted and verified by
`git status --porcelain` (clean) and `grep -n "if (request.has_speaker)"
src/arch/qwen3-tts/talker-host.cpp` (line 159, unchanged) after the
measurement.

### Plan 1's completion gate, met

All three of Plan 1's stated criteria hold as of this task: the BF16 package
loads (Task 6); a prefill built at empty instruct matches the oracle within
the recorded `replay` tolerance, with the fault-injection figure above proving
the comparison can fail (Task 7); and the capability snapshot reports zero
Preset Voices and, **on jiangzhuo's 2026-08-18 ruling after the final
whole-branch review, correcting what Task 6 originally measured and this
sentence originally claimed**, zero Profile source flags rather than
`DESCRIPTION_TEXT | SERIALIZED_PROFILE` -- `REFERENCE_AUDIO` was, and remains,
absent either way (Task 6, "Load result" above, which carries the correction
in full). Not delivered here, and not a gap against Plan 1's own scope:
`create_from_description`, the `DesignInstruct` payload, the instruct prompt
block, the public-seam validator, quantization, backends, and the listening
audit -- Plans 2 and 3.

**This is Plan 1's own gate as it stood at Plan 1's close, and is left as
history.** Plan 2's Task 5 republished the capability once Task 2 wired
`create_from_description` and Task 3 made a serialized design Profile
loadable -- see "Load result" above's own second correction for the
re-measurement (`0xa`, `DESCRIPTION_TEXT | SERIALIZED_PROFILE`, matching this
paragraph's original, pre-correction claim).

## Stage 3: VoiceDesign Package, Plan 2 Task 7

Executed 2026-08-19, the last task of Stage 3 Plan 2. Three things: the
manifest's cases, the schema's profile-contract widening, and a policy guard
binding `suite_status` to the two Python checks that used to waive themselves
independently of it. Plus two malformed-package holes an external review of
the sibling PR found in `src/arch/qwen3-tts/weights.cpp`, routed to this task
because they are the same class of defect as the guard above.

### The manifest's thirteen cases, Plan 2 Task 7

`tests/golden/qwen3-tts/qwen3-tts-12hz-1-7b-voicedesign.manifest.json` carried
`cases: []` and `suite_status: "incremental"` from Stage 3 Plan 1's Task 6
through Plan 2's Task 6 (six tasks). This task fills it to thirteen cases and
removes `suite_status` -- the twelve-case floor re-engages for this manifest
exactly as it always has for every other one.

Four of the thirteen are the measurements Tasks 4 and 6 already made, reused
verbatim rather than re-invented, with case ids chosen to match the ids those
tasks' own tolerance entries already use:

- `voicedesign-empty-instruct-en` and `voicedesign-nonempty-instruct-en` are
  Task 4's own prefill pair -- text "Qwen3-TTS is awesome!", instruct `""`
  and "A cheerful, bright female voice speaking with fast pacing and high
  energy.", both English, both seed 0. The ids are copied character for
  character from `tests/tolerances/qwen3-tts.json`'s
  `replay.probes.prefill.case_ids`, so a reader who already knows those two
  measurements recognizes them here directly.
- `voicedesign-description-cheerful-en` and `voicedesign-description-calm-en`
  reuse Task 6's own two `--description` instructs verbatim -- "A cheerful,
  bright female voice speaking with fast pacing and high energy." and "A
  deep, calm male voice speaking slowly and quietly." -- at seed 7, Task 6's
  own seed, over the same benchmark text. A `relations` entry
  (`artifact_differs`, `public_request`, `audio.pcm`) ties the pair together,
  the manifest-level expression of design section 6.3's relation 1 ("a
  different Voice changes the audio") that Task 6 already measured through
  the public seam.

Two more are genuine upstream examples, not project-authored coverage: the
GitHub README's own "Voice Design" section (`README.md#voice-design` at the
pinned revision `022e286b98fbec7e1e916cb940cdf532cd9f488e`) is the only place
in either the GitHub repository or the HuggingFace model card that actually
calls `generate_voice_design` with a real instruct -- the HuggingFace card's
own Quickstart section, checked directly, shows the *CustomVoice* example
instead (`generate_custom_voice` against `Qwen3-TTS-12Hz-1.7B-CustomVoice`),
copied across every variant's card rather than written per-variant.
`voicedesign-upstream-zh` and `voicedesign-upstream-en` are that README
section's own single-inference and first batch-inference example
respectively, text and instruct both copied verbatim (including the
batch-inference English text's own ellipsis and internal punctuation).
`test_upstream_examples_are_present` needs only one; this manifest carries
two, both real.

The remaining seven (`voicedesign-short`, `voicedesign-punctuation`,
`voicedesign-normalization`, `voicedesign-medium`, `voicedesign-lang-japanese`,
`voicedesign-seed-one`, `voicedesign-seed-forty-two`) extend coverage to the
floor the same way `qwen3-tts-12hz-0-6b-base` and
`qwen3-tts-12hz-0-6b-customvoice`'s own non-upstream cases were built:
short/medium/punctuation/normalization text shapes reused verbatim from those
two manifests' own cases, a non-Latin-script language (Japanese), and a
seed-variation trio (`voicedesign-nonempty-instruct-en` at seed 0, plus the
two new ones at seed 1 and 42, tied by a second `relations` entry) proving
design section 6.3's relation 2 in reverse -- a DIFFERENT seed, same
everything else, changes the audio. Their instruct strings are new,
project-authored voice descriptions (there is no third measured description
this task could have reused without duplicating one of the four above), and
their oracle data is COMMITTED CONTRACT, not yet committed MEASUREMENT --
consistent with every other qwen3-tts and VITS manifest's own cases at the
moment they first landed, and with CLAUDE.md's own rule that a Golden
Manifest pins provenance, cases, and tolerances while the payloads that would
prove each case (models, reference tensors, generated reports) stay
gitignored and are generated on demand. `tests/tolerances/qwen3-tts.json`'s
`variants.qwen3-tts-12hz-1-7b-voicedesign.case_count` moves from 2 to 13 in
the same task, this time meaning what that field means for every other
variant (the manifest's own case count) rather than the prefill probe's own
case count it had been coopted to describe since Task 4; `replay.probes.
prefill.cases` (still 2) and `public.checks` (still 11) are unchanged, still
naming exactly what Tasks 4 and 6 actually measured.

Each case's `expected.artifacts` and `oracle.stochastic_inputs` follow this
variant's own real oracle dumper
(`scripts/dump_reference_qwen3_tts_voicedesign.py`), not Base's or
CustomVoice's per-layer artifact shape: that dumper captures one combined
`prefill.f32` (not per-layer `talker.hidden_l*` tensors -- only the ONE place
upstream exposes the whole assembled prefill, its own module docstring says,
so there is no per-layer hook to capture), one combined `codes.i32` (not a
`codes.semantic`/`codes.acoustic` split -- `generate_voice_design`'s own
sixteen code groups come back as a single `[frames, 16]` array), and
`waveform.f32` -- because that is what genuinely exists today, driven by hand
per case, with no `--manifest` form yet (the dumper's own module docstring
says so explicitly, dated from Plan 1 when the manifest had no cases at all
to drive it from). Wiring a `--manifest` form for it, and a per-case
oracle-replay validator that actually executes `tensor_parity` and
`waveform_regression` against these thirteen cases, is infrastructure this
task's file list does not cover and is not claimed as delivered.

### The schema could not express this variant's own profile contract

`docs/schemas/synthesize-golden-manifest-v1.schema.json`'s
`$defs.profileContract` had `sources: {enum: [reference_audio,
serialized_profile]}` and unconditionally required `reference` -- a shape
that fit every profile-sources package that existed when it was written
(`qwen3-tts-12hz-0-6b-base`, the only manifest that had ever declared a
`profile` block), all of which happen to carry `reference_audio`. VoiceDesign
declares only `description_text`, so `package_contract.profile` was
inexpressible for it until now: this variant's declared sources lived only in
a doc table ("Load result", above, this same file) and an uncommitted local
probe, never in the manifest itself.

Widened: `description_text` added to the `sources` enum, and `reference` made
conditional (`if`/`then` on `sources` containing `reference_audio`) rather
than unconditionally required. `qwen3-tts-12hz-0-6b-base`'s own
`package_contract.profile` is untouched by this and still validates --
`reference_audio` is among its sources, so the `then` branch still requires
`reference` for it, exactly as the unconditional requirement used to. The
VoiceDesign manifest now carries its own `package_contract.profile`:
`{schema: "qwen3-tts-voice-design", schema_version: 1, sources:
["description_text"]}`, no `reference` block, matching
`fill_voice_profile_capability`'s own published shape for this variant (six
`reference_*` limits at zero, "Load result" above).

A second, smaller gap surfaced while writing the empty-instruct case:
`$defs.voice`'s `description` property carried `minLength: 1`, which made D3's
own legal empty instruct inexpressible as a `description_text` Voice.
Widened by dropping the floor (every previously-committed `description_text`
case -- omnivoice's own two -- carries a non-empty string regardless, so
nothing already committed is affected).

### `suite_status` had no policy guard, and closing that closed two independent findings

Verified while this plan was being written, and independently by an external
review of the sibling Stage 3 Plan 2 PR: `suite_status` appeared only in the
schema. `tests/python/test_golden_manifests.py` never read it -- its two
exemptions (`test_upstream_examples_are_present`,
`test_tolerance_case_count_matches_manifest`) were both keyed on `if not
manifest["cases"]`, never on the flag's own value. The schema waived the
twelve-case floor for whoever set the flag; the Python checks waived
themselves for whoever had zero cases; nothing tied the two together. A
manifest could carry the flag WITH cases (the schema's floor waived for a
manifest that did not need it waived), or have zero cases WITHOUT the flag
(the Python exemptions would silently skip it even though the schema itself
would reject it as too short) -- and either shape passed everything that
existed before this task. The external review demonstrated the first shape
live, against a 3-case copy of `vits-ljspeech`'s own committed manifest.

Closed with a new test,
`test_suite_status_matches_an_incrementally_built_manifest_exactly`, binding
both directions against a named, module-level `SUITE_STATUS_ALLOWLIST`: the
set of manifests carrying `suite_status` must equal the allowlist exactly,
and a manifest carrying it must have zero cases while a manifest without it
must have at least twelve. The allowlist is `frozenset()` -- empty, the
strongest form it can take, now that this task's own thirteen cases removed
the one manifest that ever needed the flag. The two existing exemptions were
also rewritten to key on `suite_status`'s own value (`manifest.get(
"suite_status") == "incremental"`) rather than on `cases` being empty as a
proxy for it, closing the same defect the new guard proves does not
otherwise recur.

Proved live, not merely by argument: adding `suite_status: "incremental"` to
a copy of `vits-ljspeech`'s own manifest (13 cases, well above the floor)
passes the schema unchanged (the schema alone does not enforce the
allowlist) and fails the new guard immediately, naming the offending
manifest. Truncating a copy of the VoiceDesign manifest to 3 cases, with no
`suite_status`, is rejected by the schema itself (`cases` too short) before
the guard is ever consulted.

### Two malformed-package holes in `weights.cpp`, found by the same external review

Unrelated to the manifest/schema/guard work above except in kind -- both are
the loader accepting a package shape no real converter would ever emit,
found the same way the `suite_status` gap above was: read closely against
what the code actually checks rather than what its comments claim.

**Mixed profile sources were accepted.** `read_profile_sources` ORs each
declared name's bit into `hparams.profile_sources` rather than refusing a
combination, so a package declaring BOTH `"reference-audio"` and
`"description-text"` -- with a speaker encoder attached, satisfying the
existing `wants_reference == carries_encoder` cross-check -- loaded clean.
No real converter emits this (`scripts/convert-qwen3-tts.py`'s
`profile_source_names` always returns exactly one name), but a loader that
exists to positively declare a package's shape must refuse the shapes it
cannot express, not merely the ones a well-behaved converter happens to
avoid. Closed with a check at the end of `read_profile_sources` itself:
`hparams.profile_sources` must equal exactly one of the two known bits, not
both (and, since the loop already refuses an empty or unrecognized name,
not neither either).

**A Description Text package could still carry surplus
`synthesize.reference.*` metadata.** `read_profile_contract`'s own
`has_speaker_encoder`-gated early return never inspects the six
`synthesize.reference.*` keys when the package carries no speaker encoder --
it just returns `true` before reaching them. A converter regression (or
hand-edited metadata) that left one or more of those keys on a converted
VoiceDesign-shaped package would load silently rather than being refused for
declaring a reference-audio contract it does not implement. Closed with an
explicit refusal in `read_profile_and_speaker_encoder`, checked one key at a
time (`GgufMetadata::has`) rather than by re-reading the whole block, for a
package that does not declare `reference-audio`.

Both proved by construction in `tests/qwen3_tts_metadata_test.cpp`
(`test_mixed_profile_sources_are_refused`,
`test_description_text_package_with_surplus_reference_keys_is_refused`), and
both proved by mutation: temporarily disabling each new check individually,
rebuilding, and confirming the corresponding new test is what catches the
regression (`check failed: synth::qwen3tts::read_hparams(...) != SYNTH_OK`),
then reverting and confirming clean (`git status --porcelain`, `git diff
--stat` showing only the intended lines) before committing.

### Gates

- `synthesize-qwen3-tts-metadata-test` (unit, the two new cases plus the
  rest of the file's existing package-metadata coverage): passes clean under
  the plain `build` tree and under `build-sanitize`
  (`-DSYNTH_SANITIZE=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo`).
- `synthesize-golden-manifest-contract` / `tests.python.test_golden_manifests`
  (25 tests, one new -- `test_suite_status_matches_an_incrementally_built_manifest_exactly`
  -- plus the manifest and schema widenings above exercised through the
  existing 24): all pass.
- `tests.python.test_tolerance_coverage`: unchanged, 3/3.
- Full unit gate, plain `build` tree: 104/106, the same two pre-existing,
  not-ours failures (`synthesize-python-api-wheel-test`,
  `synthesize-vits-python-unit`, both wanting gitignored VITS artifacts).
- Full unit gate, `build-sanitize`: 103/105, the identical two failures, no
  third.
- `scripts/ci/clang-format.sh --fix` after `git add`, then `--check-diff`:
  clean.

## Stage 3: VoiceDesign Package, Plan 3 Task 2

Cuts `qwen3-tts-12hz-1-7b-voicedesign-F16.gguf` from the 4,295,891,904-byte BF16
source and measures its size. **F16 is 283,136 bytes larger than the package it
was cut from** -- the same direction as Base's own F16, whose reproducible
current delta is **+184,288 bytes** (2,516,522,624 -> 2,516,706,912, both
measured directly off the packages on disk today), not the +184,448 this
document (`:22`, `:3356`) and `docs/quantization.md` still quote elsewhere.
Both are real: Base's BF16 was re-converted on 2026-08-18 after the
`code_predictor.intermediate_size` metadata key landed, moving the source by
+160 bytes, while Base's F16 was never re-cut against the new source -- so the
two historical figures describe two different BF16 files with the same name,
and 184,448 no longer reproduces against what is on disk now. 184,288 is what
a fresh `stat` gives and is the figure a later task should anchor against.
As the design anticipated, VoiceDesign's relative penalty is smaller than
Base's either way (0.0066% of the source here against Base's 0.0073%), because
the matrix half this variant halves is a larger share of a package whose
sensitive half does not also grow.

**This is a size result only, and it does not decide publication.** Whether
F16 ships for this family is a **speed** question, not a size one --
`docs/quantization.md:455-459` states the rule directly ("F16 is a speed
profile for this family rather than a size one"), and Base's own F16, also
larger than its source, **was published** on 2026-08-17 on exactly that
reasoning (see the Status paragraph above). The precedent this section
originally cited was wrong on two counts: `Q5_K_MIXED` is withheld for
CustomVoice, not Base ("Q5_K_MIXED is buildable and is not recommended",
above), and for an **accuracy** reason -- talker logits cosine 0.9648 -- not a
size one. Neither precedent supports a no-publish verdict from the size result
above. **Whether F16 ships for VoiceDesign is left open here**: this task is
forbidden from taking a timing figure (the global no-performance-figure rule),
so it cannot answer the question that actually decides publication for this
profile family-wide. Task 5 measures RTF on a `Release`-typed tree and settles
it.

**All figures below are from the `build` tree, `CMAKE_BUILD_TYPE=Release`,
reconfigured mid-task from CLAUDE.md's default unit-gate settings
(`-DSYNTH_BUILD_INTEGRATION_TESTS=OFF`) to `ON` -- the only way to obtain
`synthesize-qwen3-tts-voicedesign-prefill-real`, which is registered behind
that flag. No timing figure is taken from this build or claimed anywhere in
this section; the global no-performance-figure rule for this task is
unaffected by the reconfigure.**

### Cutting it required a quantizer fix, not just a run

The first `--quant F16` attempt refused outright:
`synthesize-quantize: unknown qwen3-tts tensor:
talker.code_predictor.small_to_mtp_projection.bias`. This is not a converted
oracle gap -- `src/arch/qwen3-tts/catalog.cpp` has resolved this exact tensor
pair since Stage 3 Task 6 ("A genuine architecture gap, found and closed, not
converted around", above) -- it is `tools/synthesize-quantize/policy.cpp`
never having been taught the name, because no prior task had ever run the
quantizer against a package that carries it: `small_to_mtp_projection` is the
width bridge a package needs only when the code predictor's hidden size
differs from the talker's, which is true of no 0.6B package (Base,
CustomVoice) and only of this 1.7B one (predictor 1024 against talker 2048).
Fixed by adding one classifier arm to `classify_qwen3_talker`'s
`code_predictor` branch, shaped identically to the already-existing
`text_projection.linear_fcN` arm 27 lines above it in the same function --
three other classifier arms apart (text_embedding/codec_embedding,
model.norm, the model.layers dispatch), not three lines apart:
the weight (a two-dimensional `Linear(talker.hidden_size, hidden_size)`,
`Role::Matrix` at the runtime catalog) classifies `MatrixWeight`, the bias
(one-dimensional) classifies `Sensitive`. Covered by two new cases in
`tests/qwen3_tts_quantization_policy_test.cpp`'s existing by-name loops (one
per role) and proved load-bearing by mutation: commenting out the new
classifier arm and rebuilding reproduced the exact pre-fix failure shape
(`check failed: resolve(*q8, name).type == GGML_TYPE_Q8_0` on the weight
case), reverted and confirmed clean before the F16 cut below ran. This fix
touches production code outside this task's own file list
(`tools/synthesize-quantize/policy.cpp`,
`tests/qwen3_tts_quantization_policy_test.cpp`) and is committed separately
from the tolerance/doc commit this task otherwise produces, for the same
reason Stage 3 Task 6's catalog fix stands on its own: cutting F16 at all had
no other path.

### Open item: the quantizer classifier and the runtime catalog have no shared pin

`tools/synthesize-quantize/policy.cpp`'s classifier (`classify_qwen3_tts_tensor`
and its family) and `src/arch/qwen3-tts/catalog.cpp`'s resolver are two
independent, hand-maintained descriptions of the same tensor set. Both carry a
comment saying the two "must not drift apart"
(`src/arch/qwen3-tts/catalog.cpp:35,179`, `tools/synthesize-quantize/policy.cpp:701-702`)
and neither has a mechanism enforcing it. This task's own
`small_to_mtp_projection` gap above is exactly what happens when they do: the
runtime has resolved that tensor pair since Stage 3 Task 6; the tool did not,
until this task, because no earlier task had ever run the quantizer against a
package that carries it. No CTest target loads a cut (non-BF16) package
through `synth_model_load` for this family, so a classifier gap or a
classifier/resolver shape disagreement is caught only by hand-running the
quantizer against a real, multi-GB package -- as this task did, by accident of
being the first to try it on VoiceDesign.

A cheap partial pin exists and is not implemented here, because it touches
`tests/qwen3_tts_catalog_test.cpp`, outside this task's file list: that file
already builds synthetic F16/Q8_MIXED fixtures and types each tensor with its
own local heuristic
(`tests/qwen3_tts_catalog_test.cpp:579-583`: `matrix = entry.ne.size() >= 2 &&
...`) rather than through the real classifier, and its own assertion is
deliberately loose --
`SYNTH_TEST_CHECK(status == SYNTH_OK || status == SYNTH_ERR_GGUF)`
(`:590`) -- specifically because "this synthetic package's role split is a
coarse approximation of the catalog's" (the file's own comment, `:586-588`).
Compiling `policy.cpp` into `synthesize-qwen3-tts-catalog-test` and typing
those same fixtures through the classifier instead of the local heuristic,
then tightening the assertion to `SYNTH_OK` alone, would catch a classifier
tensor the resolver accepts and the classifier doesn't (or the reverse) at
unit-test speed, with no real package on disk required. It would not replace
running the quantizer against a real package -- no synthetic fixture stands in
for a genuine checkpoint's actual tensor names -- but it would have caught
this exact class of gap earlier than this task did.

### Size and tensor census

| | BF16 (source) | F16 |
|---|---|---|
| bytes | 4,295,891,904 | 4,296,175,040 |
| tensor count | 659 | 659 |
| by type | 404 BF16 + 255 F32 | 267 F16 + 392 F32 |

Delta: **+283,136 bytes**, F16 larger than its source. 137 tensors (404 - 267)
moved from a two-byte type to F32 -- the Sensitive half widening from bf16 to
f32 storage, the same mechanism Base's own F16 cell already documents, applied
to a package whose matrix half (the talker + code predictor, halved) is a
correspondingly larger share of the 4.30 GB total than Base's 2.5 GB package's
own matrix half was.

### Load, confirmed before any cell was filled

The brief's own `synthesize-cli --list-voices` does not exist -- no such flag
is defined anywhere in `examples/cli/` (`--help` lists `--text`, `--phonemes`,
`--token-ids`, `--language`, `--voice`, `--seed`, `--rate`,
`--max-output-frames`, `--backend`, `--device`, nothing that lists Voices).
Confirmed instead with the already-built `synthesize-qwen3-tts-public-real`
runner in `probe:description` mode, which calls `synth_model_load` before
anything else and, since VoiceDesign supports `create_from_description`,
completed the probe cleanly: `{"probe": "description", "status": 0}`, exit 0.
The package loads.

### The `replay` cell: the ordinary case loop has no reading for this variant

`scripts/validate-qwen3-tts-replay.py --stage replay` (the brief's literal
Step 5) fails for this variant at **every** profile, BF16 included -- run
against the already-published, already-validated BF16 package it prints
"no case produced a comparison: 13 selected, 13 skipped for missing oracle
artifacts under build/goldens/qwen3-tts/qwen3-tts-12hz-1-7b-voicedesign" and
exits 1. This is not an F16 defect: `run_case`'s oracle-artifact check
(`oracle_root/<case-id>/codes/semantic.i32`) has nothing to find, because no
prior task ever dumped per-case codes/waveform oracle payloads for this
variant's thirteen manifest cases -- only the two explicit prefill cases
(design section 6.2's stage grid for this rung: no codes, no waveform measured
for this variant at any profile, stated already on this variant's BF16 `replay`
cell). The BF16 `replay.prefill` probe was filled by invoking
`tests/qwen3_tts_voicedesign_prefill_real.cpp` directly against
`reports/porting/qwen3-tts/qwen3-tts-12hz-1-7b-voicedesign/oracle{,-instruct}/prefill.f32`,
not through this script's ordinary case loop; F16's cell is filled the same
way, for the same reason, reusing BF16's 0.01 bound rather than re-deriving one
a profile change gave no reason to move:

| case | BF16 p95_relative | F16 p95_relative |
|---|---|---|
| voicedesign-empty-instruct-en | 0.00269 | 0.002689 |
| voicedesign-nonempty-instruct-en | 0.002903 | 0.002902 |

Both F16 figures are within half a bf16 unit-roundoff of BF16's own, headroom
3.72x and 3.45x against the unchanged 0.01 bound -- essentially identical to
BF16's 3.72x/3.44x. Same reasoning as Base's own F16 `speaker.x_vector` cell:
the residual is the *oracle's* bf16 storage, not the port's, so an F16 port
weight (finer than bf16) costs nothing measurable here either.

### The `public` cell: all eleven checks pass

`scripts/validate-qwen3-tts-public.py --profile F16 --backend cpu` (the
brief's Step 4 command, run as written -- both flag spellings checked out)
against the two description instructs, after the `replay` cell above was
filled first so relation 3's own citation ("an empty instruct reproduces the
oracle within the replay stage's recorded tolerance") had a real F16 record to
read rather than reporting unmeasured. All 11 checks passed, identical
verdicts to BF16, same 3 skips (no Preset Voice catalogue, no dialect speaker):
seed reporting and reproduction, a different seed and a different Voice both
change the audio, the resolved language is reported, the empty-instruct
public-seam smoke run (40320 frames), relation 3 above (p95_relative 0.002689
against 0.01), and both package-support refusals
(`SYNTH_ERR_UNSUPPORTED_VOICE` for `create_from_reference`,
`SYNTH_ERR_UNSUPPORTED_INPUT` for a supplied description language tag). Report:
`reports/validate/qwen3-tts/public-voicedesign-F16.json` (gitignored).

### Verification

- `synthesize-qwen3-tts-quantization-policy-test`: passes with the new
  classifier arm; mutation (comment out the arm, rebuild) reproduces the exact
  pre-fix refusal shape, reverted and confirmed clean before proceeding.
- `synthesize-golden-manifest-contract` / `synthesize-tolerance-coverage`: both
  pass -- the new F16 profile carries the same stage set as BF16
  (`replay`, `public`), and `case_count` (13) is untouched by this task.
- `synthesize-qwen3-tts-voicedesign-prefill-real` /
  `-prefill-instruct-real` (the BF16-driven CTest golden gates): still pass,
  unaffected by the quantizer-only fix.
- Full unit gate, `build` tree (`SYNTH_BUILD_INTEGRATION_TESTS=ON` per the
  reconfigure above): 104/106, the same two pre-existing, not-ours failures
  (`synthesize-python-api-wheel-test`, `synthesize-vits-python-unit`'s sole
  error `test_quantization_reports_match_current_artifacts`). No third.

### Fix round 1, review of this task

Code review found this section carrying a fabricated measurement and a wrong
publication conclusion; both are corrected above rather than left with a note
here, per this document's practice of superseding stale prose in place when
the stale text is short-lived and never described a real interval (contrast
the Status paragraph's own longer-lived corrections, which are kept as
history). For the record: `tests/tolerances/qwen3-tts.json`'s F16 `replay`
cell carried a `fault_injection_by_case` block copied verbatim from BF16's,
dated 2026-08-18 -- two days before the F16 package existed -- and has been
removed; no fault injection was run against F16, and BF16's own cell already
carries the record for the port source this variant shares. The F16 stage
`description` field, also BF16's verbatim and asserting "no instruct block
exists in the port yet," has been rewritten -- Plan 2 shipped the instruct
block before this task ran, and this cell's own `instruct_tokens: 19` already
contradicted the old text.

## Stage 3: VoiceDesign Package, Plan 3 Task 3

Cuts `qwen3-tts-12hz-1-7b-voicedesign-Q8_MIXED.gguf` from the same
4,295,891,904-byte BF16 source Task 2 cut F16 from, and measures its size and
the two tolerance cells the plan asks for. **This is a size result and a
tolerance-agreement result only -- it does not decide publication.** Whether
Q8_MIXED ships for this variant is settled in Task 4 (against `Q5_K_MIXED`)
and Task 5 (on speed, forbidden to this task by the global
no-performance-figure rule), not here -- the same boundary Task 2 drew for F16
after its own first attempt drew a conclusion this task's evidence could not
support.

**All figures below are from the `build` tree, `CMAKE_BUILD_TYPE=Release`,
left at `-DSYNTH_BUILD_INTEGRATION_TESTS=ON` by Task 2's own reconfigure (the
only way to obtain `synthesize-qwen3-tts-voicedesign-prefill-real`, used
below). No timing figure is taken from this build or claimed anywhere in this
section.**

### Cutting it required no fix this time

`--quant Q8_MIXED` ran and wrote the package on the first attempt --
`talker.code_predictor.small_to_mtp_projection`, the tensor pair Task 2 had to
teach the quantizer's classifier (`98d8b84`), is shared by both profiles'
runs through the same classifier code, so Task 2's fix already covers this
cut. No production code changed in this task.

### Size and tensor census

| | BF16 (source) | Q8_MIXED |
|---|---|---|
| bytes | 4,295,891,904 | 2,499,423,680 |
| tensor count | 659 | 659 |
| by type | 404 BF16 + 255 F32 | 267 Q8_0 + 392 F32 |

Delta: **-1,796,468,224 bytes**, a ratio of **0.5818** against BF16 (58.18 %
of the source's size, i.e. **41.82 % smaller**). This does **not** match
`qwen3-tts-12hz-0-6b-base`'s own Q8_MIXED ratio of 0.663 (33.7 % smaller,
2,516,522,624 -> 1,667,606,112) -- the brief warned against assuming it would,
and it does not: VoiceDesign shrinks proportionally *more*. The 267/392
tensor-type split is identical to F16's own 267/392 split (Task 2, above) --
same classifier boundary, same 267 tensors take the profile's matrix type
(F16 there, Q8_0 here) and the same 392 stay F32 -- which is the same
mechanism Task 2 named for why VoiceDesign's F16 penalty is smaller than
Base's: the matrix half this variant halves (or, here, quantizes to Q8_0) is
a larger share of the package than it is in the 2.5 GB Base package, so a
change to that half moves the total by proportionally more in either
direction, larger when two-byte types cannot shrink (F16) and smaller when an
8-bit block type can (Q8_MIXED).

### Load, confirmed before either cell was filled

The brief's own Step 3 literal is broken, independently of Task 2's
`--list-voices` finding for the same step in the F16 task: it invokes
`synthesize-qwen3-tts-public-real <model> probe:description en`, supplying
only two positional arguments after the model path where the driver requires
four (`<model.gguf> <out.pcm> <voice-id|...> <language-tag|-> <seed|random>
[max-frames] [cpu|cuda] [threads]`, confirmed against
`tests/qwen3_tts_public_real.c`'s own usage string). Running it as written
exits 2 on a usage error rather than probing anything. Confirmed instead with
the same substitute Task 2 used, with an explicit language tag and seed added
since this task's own out.pcm path makes the argument count unambiguous:

```
build/bin/synthesize-qwen3-tts-public-real \
  models/qwen3-tts-12hz-1-7b-voicedesign/qwen3-tts-12hz-1-7b-voicedesign-Q8_MIXED.gguf \
  /tmp/qwen3-tts-q8mixed-probe.pcm \
  probe:description - 0 8192 cpu
```

`{"probe": "description", "status": 0}`, exit 0. The package loads and
`create_from_description` works on the cut package.

### The `replay` cell: same prefill-driver method as F16, same reason

`scripts/validate-qwen3-tts-replay.py --stage replay` (the brief's own Step 5
text already says not to try this, and Task 2's report is the source of that
finding) is not re-verified here beyond re-reading Task 2's account -- it
fails identically for every profile of this variant because no per-case
oracle artifacts exist for it, a fact about the variant's Golden Manifest
plumbing, not about any one profile. `stages.replay` is filled the same way
Task 2 filled F16's: `tests/qwen3_tts_voicedesign_prefill_real.cpp`, built
already (Task 2's reconfigure), invoked directly against the same two
committed oracle dumps, reusing BF16's and F16's 0.01 bound:

| case | BF16 p95_relative | F16 p95_relative | Q8_MIXED p95_relative |
|---|---|---|---|
| voicedesign-empty-instruct-en | 0.00269 | 0.002689 | 0.006130 |
| voicedesign-nonempty-instruct-en | 0.002903 | 0.002902 | 0.006085 |

Both pass (`gate_passed: true`, `shapes_match: true`), but visibly further
from BF16/F16 than those two are from each other -- roughly 2.2-2.3x their
residual, rather than half a bf16 ulp. This is the expected direction and
rough size for the difference in kind, not just degree, between the two
profiles: F16's matrix half stays a two-byte float, finer than the oracle's
own bf16 storage, so its residual is attributable to the *oracle*. Q8_MIXED's
matrix half is an 8-bit block type, coarser than bf16, so some of this
residual is attributable to the *port* for the first time in this variant's
profile history. `observed_max_relative` is the larger of the two (0.00613),
giving **1.63x headroom** against the unchanged 0.01 bound -- positive and
passing, but the thinnest headroom this variant's `replay.prefill` probe has
recorded (BF16 **3.44x** -- its own committed two-case figure,
`tests/tolerances/qwen3-tts.json`'s BF16 `prefill` cell, not the
empty-instruct-only 3.72x Plan 1 recorded before the non-empty case existed;
F16 3.45x, nearly identical to BF16's; Q8_MIXED 1.63x). No fault injection was
run against Q8_MIXED; this cell carries no `fault_injection_by_case` block,
matching the standing instruction not to carry one forward from a profile it
was not measured on.

### The `public` cell: eleven checks pass, filled after `replay`

`scripts/validate-qwen3-tts-public.py --profile Q8_MIXED --backend cpu` (the
brief's Step 4 command, run as written) against the same two description
instructs Task 2 used. Run twice, in the same order Task 2 established for
F16: once before `tests/tolerances/qwen3-tts.json` carried a Q8_MIXED
`replay` cell (relation 3 -- "an empty instruct reproduces the oracle within
the replay stage's recorded tolerance" -- read "unmeasured" and was skipped,
a fourth skip beyond the usual three), and once after, so relation 3 had a
real Q8_MIXED record of its own to cite instead of falling back to BF16's or
F16's or reporting unmeasured. The second run is the one recorded: all 11
checks passed, identical verdicts to BF16 and F16, the same 3 skips (no
Preset Voice catalogue). Relation 3 read this same file's own Q8_MIXED
`replay.prefill` probe (p95_relative 0.00613 against max_relative 0.01) and
passed. Report: `reports/validate/qwen3-tts/public-voicedesign-Q8_MIXED.json`
(gitignored).

### Verification

- `synthesize-qwen3-tts-quantization-policy-test`: passes, unchanged by this
  task (no classifier edit was needed).
- `synthesize-golden-manifest-contract` / `synthesize-tolerance-coverage`:
  both pass -- the new Q8_MIXED profile carries the same stage set as BF16
  and F16 (`replay`, `public`), and `case_count` (13) is untouched.
- `synthesize-qwen3-tts-voicedesign-prefill-real` /
  `-prefill-instruct-real` (the BF16-driven CTest golden gates): still pass,
  unaffected -- this task changed no production code.
- `scripts/ci/clang-format.sh --check-diff`: clean (only `.json`/`.md`
  changed).
- Full unit gate, `build` tree (`SYNTH_BUILD_INTEGRATION_TESTS=ON` per Task
  2's still-standing reconfigure): 104/106, the same two pre-existing,
  not-ours failures (`synthesize-python-api-wheel-test`,
  `synthesize-vits-python-unit`'s sole error
  `test_quantization_reports_match_current_artifacts`). No third.
- `git status --porcelain` checked before staging; no `.gguf` or build
  artifact staged. `models/` is a symlink outside the worktree, gitignored
  via `/models`; `reports/validate/` is gitignored per the global
  constraints.

## Stage 3: VoiceDesign Package, Plan 3 Task 4

Cuts `qwen3-tts-12hz-1-7b-voicedesign-Q5_K_MIXED.gguf` from the same
4,295,891,904-byte BF16 source Tasks 2 and 3 cut F16 and Q8_MIXED from, and
measures its size and the two tolerance cells the plan asks for. **This is
this profile's first real evaluation for this variant -- the design spec's own
words were that the extra shrink `Q5_K_MIXED` buys over `Q8_MIXED` did not
change the recommendation at Base's/CustomVoice's ~2.4 GB source, but might at
this variant's ~4.3 GB one.** It does not, but the reason is not the one that
framing anticipated: the accuracy side moved further than the size side did.
`docs/quantization.md`'s new "VoiceDesign's Q5_K_MIXED" section carries the
full arithmetic and the recommendation; this section is the measurement
record it draws on.

**All figures below are from the `build` tree, `CMAKE_BUILD_TYPE=Release`,
left at `-DSYNTH_BUILD_INTEGRATION_TESTS=ON` by Task 2's own reconfigure (the
only way to obtain `synthesize-qwen3-tts-voicedesign-prefill-real`, used
below). No timing figure is taken from this build or claimed anywhere in this
section.**

### Cutting it required no fix, and the load check matches the brief this time

`--quant Q5_K_MIXED` ran and wrote the package on the first attempt, for the
same reason Task 3's Q8_MIXED cut needed none: `talker.code_predictor.
small_to_mtp_projection`, the tensor pair Task 2 taught the quantizer's
classifier (`98d8b84`), is shared by every profile's run through the same
classifier code. No production code changed in this task.

Unlike Tasks 2 and 3's own Step 2/3 literals (a missing `--list-voices` flag
and, separately, an under-supplied argument count), this task's brief gives
the load check with all five positional arguments the driver actually
requires:

```
build/bin/synthesize-qwen3-tts-public-real \
  models/qwen3-tts-12hz-1-7b-voicedesign/qwen3-tts-12hz-1-7b-voicedesign-Q5_K_MIXED.gguf \
  /dev/null probe:description en 0
```

Ran verbatim, no substitution needed: `{"probe": "description", "status": 0}`,
exit 0. The package loads and `create_from_description` works on the cut
package.

### Size and tensor census

| | BF16 (source) | Q8_MIXED | Q5_K_MIXED |
|---|---|---|---|
| bytes | 4,295,891,904 | 2,499,423,680 | 1,780,723,136 |
| tensor count | 659 | 659 | 659 |
| by type | 404 BF16 + 255 F32 | 267 Q8_0 + 392 F32 | 267 Q5_K + 392 F32 |

The 267/392 tensor-type split is identical to F16's and Q8_MIXED's own split
(Tasks 2 and 3) -- the same classifier boundary, only the matrix half's target
type changes (F16, then Q8_0, then Q5_K).

**Against BF16:** delta -2,515,168,768 bytes, ratio 0.4145 (41.45 % of the
source, 58.55 % smaller). **Against Q8_MIXED, the number this task exists to
answer:** delta -718,700,544 bytes, ratio 0.7125 (71.25 % of Q8_MIXED, i.e.
**28.75 % smaller**) -- close to but not exactly the design spec's own "~25 %"
estimate. 718.7 MB is a much larger absolute cut than the same percentage
would have bought at Base's/CustomVoice's ~2.4 GB source, which is exactly the
scale-up the brief flagged as worth re-checking rather than assuming answered.

### Load, confirmed before either cell was filled

Covered above -- the brief's own Step 2 literal ran clean, unlike Tasks 2's
and 3's own Step 2/3 literals.

### The `replay` cell: same prefill-driver method, and this time it fails

`scripts/validate-qwen3-tts-replay.py --stage replay` is not re-verified here
beyond re-reading Task 2's account (the brief's own Step 4 text already says
not to try it): it fails identically for every profile of this variant
because no per-case oracle artifacts exist for it, a fact about the variant's
Golden Manifest plumbing, not about any one profile. `stages.replay` is
filled the same way Tasks 2 and 3 filled F16's and Q8_MIXED's:
`tests/qwen3_tts_voicedesign_prefill_real.cpp`, already built (Task 2's
reconfigure), invoked directly against the same two committed oracle dumps,
against the same 0.01 bound:

| case | BF16 | F16 | Q8_MIXED | Q5_K_MIXED |
|---|---|---|---|---|
| voicedesign-empty-instruct-en | 0.00269 | 0.002689 | 0.006130 | **0.031029** |
| voicedesign-nonempty-instruct-en | 0.002903 | 0.002902 | 0.006085 | **0.030366** |

**Both cases FAIL** -- `gate_passed: false` in the driver's own output, on
both, reproduced byte-identically on a repeat run of the empty-instruct case.
This is a different finding in kind from every other profile this variant has
measured, not merely a further step down the same slope: BF16, F16 and
Q8_MIXED all cleared 0.01 with headroom (3.44x, 3.45x, 1.63x respectively);
Q5_K_MIXED's worst case (0.031029) is **3.10x OVER** the bound, giving a
"headroom" of 0.32x -- the first cell in this variant's history where that
number reads below 1. Scale check against the family's own progression:
Q8_MIXED's residual was already ~2.2-2.3x BF16's/F16's own (Task 3's finding,
attributed to Q8_0's 8-bit matrix half being coarser than the oracle's bf16
storage); Q5_K_MIXED's residual here is a further ~5.06x/4.99x on top of
Q8_MIXED's own (empty/nonempty respectively), and ~11.5x/10.5x BF16's --
consistent in DIRECTION with a coarser matrix quantization (5-bit K-quant
blocks against Q8_0's 8-bit ones) but a much larger jump than the F16 ->
Q8_MIXED step was, not a linear continuation of it. The bound is left at the
BF16-derived 0.01 rather than loosened to make this profile pass -- the bound
describes what this family's own probes have judged an acceptable prefill
deviation, and a probe failing it is the finding, not a reason to move the
goalpost. No fault injection was run against this profile; this cell carries
none, matching Task 3's own precedent of running fault injection only once,
on the BF16 cell that established the technique.

### The `public` cell: ten of eleven checks pass, filled after `replay`

`scripts/validate-qwen3-tts-public.py --profile Q5_K_MIXED --backend cpu`
(the brief's own Step 4 command block, run as written for the `public`
half -- the `replay` half described in the same step is the one Task 2
already established does not work for this variant, per the brief's own
text). Run twice, in the same order Tasks 2 and 3 established: once before
`tests/tolerances/qwen3-tts.json` carried a Q5_K_MIXED `replay` cell
(relation 3's citation half read "unmeasured" and was skipped, a fourth skip
beyond the usual three, 10 of 11 non-skip checks passing), and once after, so
relation 3 had a real Q5_K_MIXED record of its own to cite. **The second run
is the one recorded, and relation 3's citation half correctly FAILED**:
`check_empty_instruct_within_tolerance` (`scripts/validate-qwen3-tts-public.py`)
reads this same file's own Q5_K_MIXED `replay.prefill` probe (p95_relative
0.031029 against max_relative 0.01) and returns `passed=False` because the
committed observation genuinely exceeds the committed bound -- this is the
script working correctly, not a defect in it. All ten other checks passed,
including relation 3's OWN live smoke-run half ("an empty instruct
synthesizes through the public seam": 30720 frames, nonzero) -- a package
that loads and produces audio through the public seam is a different claim
from one whose prefill matches the oracle within tolerance, and this package
still does the former even though it fails the latter. Same 3 skips as every
other profile (no Preset Voice catalogue). Report:
`reports/validate/qwen3-tts/public-voicedesign-Q5_K_MIXED.json` (gitignored).

### Recommendation

**Do not publish `Q5_K_MIXED` for `qwen3-tts-12hz-1-7b-voicedesign`.** The
standing precedent is CustomVoice's own `Q5_K_MIXED` (talker-logits cosine
0.9648, deliberately unpublished on accuracy) -- this result matches that
precedent's direction but is the stronger of the two: CustomVoice's number was
a lower cosine on a probe with no committed pass/fail bound, while this is an
explicit breach of a committed `max_relative` gate, on both of this variant's
two measured `replay` cases. The size side of the question this task was
written to answer -- whether a further 28.75 % shrink over `Q8_MIXED` matters
more at 4.3 GB than an equivalent shrink did at Base's/CustomVoice's 2.4 GB --
is answered "yes, noticeably more" (718.7 MB against a package that is itself
larger to begin with), but that answer is moot once the accuracy side has
moved even further than the size side did. Task 8's card should record
`Q5_K_MIXED` as buildable and load-checked for this variant but not shipped,
on accuracy -- but the CustomVoice card Task 8 might otherwise use as a
template does not do this at all: `scripts/hf_cards/
qwen3-tts-12hz-0-6b-customvoice.yaml:84` reads `profiles: [F16, Q8_MIXED]`,
omitting `Q5_K_MIXED` entirely rather than naming it measured-and-not-shipped.
Task 8 has to establish this shape, not copy it. See "Open item: CustomVoice's
card omits Q5_K_MIXED" below for the tracked gap.

**Nothing mechanical currently enforces this recommendation.** No registered
test reads any of `tests/tolerances/qwen3-tts.json`'s `headroom`,
`observed_max_relative`, `all_passed`, `failed_checks` or `gate_passed`
fields for this or any profile -- `tests/CMakeLists.txt:1347-1361` hardcodes
`profiles BF16 stages replay probes "prefill" max_relative` for the prefill
CTest gate, and the public-request CTest gate (`tests/CMakeLists.txt:2070-
2081`) runs `--profile BF16 --backend cpu` against `SYNTH_QWEN3_TTS_TEST_MODEL`
(`CMakeLists.txt:128-130`, the CustomVoice package). F16's and Q8_MIXED's
cells for this variant are equally unread, so this is not a regression this
task introduced -- but it does mean a future cut-and-ship of `Q5_K_MIXED`
would not be caught by CTest; only this document, the porting record and the
tolerance file's own prose stand between the measurement and a mistaken
publication.

**Recording the failing cell at all is a departure from both of this
family's own precedents, made deliberately.** CustomVoice's `Q5_K_MIXED`
negative lives only in prose (`docs/quantization.md`, this document) with no
tolerance cell behind it; the Base variant's declined-CUDA negative (see "The
Base variant carries no CUDA sub-grid" above) is a deliberate *absence* of a
cell rather than a committed one that reads false. This task commits the
cell anyway -- `gate_passed: false`, `all_passed: false`, `failed_checks`
populated -- because an absent or prose-only record of a failure is strictly
weaker evidence than a reproducible, machine-readable one sitting in the
same grid a passing profile would occupy: a future reader, or a script
written later that DOES check these fields, can find this result by looking
at the grid rather than needing to already know to look for it in prose.

### Open item: CustomVoice's card omits Q5_K_MIXED instead of naming it measured-and-not-shipped

Found while writing this task's recommendation, not fixed here (out of this
task's file list). This plan's own Global Constraints state that a profile
which fails its own test belongs in the card "as measured-and-not-shipped
... rather than omitted." CustomVoice's published card,
`scripts/hf_cards/qwen3-tts-12hz-0-6b-customvoice.yaml:84`, reads `profiles:
[F16, Q8_MIXED]` -- `Q5_K_MIXED` (talker-logits cosine 0.9648, withheld on
accuracy, recorded in `docs/quantization.md`'s "What each profile is for")
is silently absent rather than named. A case-insensitive `grep -rn q5` across
all of `scripts/hf_cards/` and `docs/models/` confirms no card anywhere in
the tree names a withheld `Q5_K_MIXED`. This is a real, pre-existing gap
against the plan's own rule -- not introduced by this task, and not
something this task's file list (`tests/tolerances/qwen3-tts.json`,
`docs/quantization.md`, `docs/porting/families/qwen3-tts.md`) authorizes
fixing, since CustomVoice's card is already published and re-publishing it
is a separate outward act requiring its own confirmation. Left for whoever
next touches CustomVoice's card or Task 8's own card-writing work to close.

### Verification

- `synthesize-qwen3-tts-quantization-policy-test`: passes, unchanged by this
  task (no classifier edit was needed).
- `synthesize-golden-manifest-contract` / `synthesize-tolerance-coverage`:
  both pass -- the new Q5_K_MIXED profile carries the same stage set as BF16,
  F16 and Q8_MIXED (`replay`, `public`); neither test inspects `gate_passed`
  or `all_passed` values, only stage-set presence, so a profile that measures
  and FAILS is exactly as well-formed to these tests as one that measures and
  passes. `case_count` (13) is untouched.
- `synthesize-qwen3-tts-voicedesign-prefill-real` /
  `-prefill-instruct-real` (the BF16-driven CTest golden gates, which run
  against the pinned BF16 test model, not this cut): still pass, unaffected
  -- this task changed no production code.
- `scripts/ci/clang-format.sh --check-diff`: clean (only `.json`/`.md`
  changed).
- Full unit gate, `build` tree (`SYNTH_BUILD_INTEGRATION_TESTS=ON` per Task
  2's still-standing reconfigure): 104/106, the same two pre-existing,
  not-ours failures (`synthesize-python-api-wheel-test`,
  `synthesize-vits-python-unit`'s sole error
  `test_quantization_reports_match_current_artifacts`). No third.
- `git status --porcelain` checked before staging; no `.gguf` or build
  artifact staged. `models/` is a symlink outside the worktree, gitignored
  via `/models`; `reports/validate/` is gitignored per the global
  constraints.

## Stage 3: VoiceDesign Package, Plan 3 Task 5

Measures latency, RTF, load time and peak memory for the three profiles Tasks
2-4 cut (`BF16`, `F16`, `Q8_MIXED`), and decides the question Task 2 was
forbidden to answer: whether `F16`, which is *larger* than its `BF16` source
(+283,136 bytes, Task 2), is fast enough on this variant to justify shipping
it anyway. `docs/quantization.md:455-459` already records F16 as this
family's speed profile rather than its size one; this task supplies the speed
number. `Q5_K_MIXED` already failed its own accuracy gate in Task 4 (headroom
0.32x) and is measured here for context only, not as a candidate.

### The measurement tree

`build/rel-dgx-spark` predated this plan's HEAD by three days (last configured
2026-08-17, before Tasks 1-4 landed), so it was reconfigured and rebuilt
before anything was timed:

```
cmake --preset rel-dgx-spark
cmake --build --preset rel-dgx-spark -j 20
```

Confirmed before measuring anything:

```
$ grep CMAKE_BUILD_TYPE build/rel-dgx-spark/CMakeCache.txt
CMAKE_BUILD_TYPE:STRING=Release
```

`sm_121a`, CUDA Toolkit 13.3, aarch64/GB10 (`GGML_SYSTEM_ARCH: ARM`, ggml
commit `707321c4`) -- the same tree Base's own Task 14 used, rebuilt at this
task's own HEAD rather than reused stale.

### The workload, fixed and repeatable

VoiceDesign has no reference audio and no ICL path (Stage 3 Plan 1), so
Base's Task 14 workload -- a sentence through an ICL Voice Profile -- has no
counterpart here; the analogous fixed point for this variant is a `desc:`
Voice Profile built from a Description Text instruct, driven through
`synthesize-qwen3-tts-public-real`. Text and description, seed and thread
count are all fixed and repeated verbatim across every run in this table:

- **Text** (delivered on stdin): `"This is a test of Qwen three T T S voice
  design synthesis."` -- 13 words, chosen to mirror the length and cadence of
  Base's own Task 14 sentence without borrowing its "voice cloning" claim,
  which this variant cannot make.
- **Description** (the `desc:` voice): `"A cheerful, bright female voice
  speaking with fast pacing and high energy."` -- the same nonempty-instruct
  description Tasks 2-4 already used for this variant's `public`/`replay`
  checks, reused here rather than inventing a fourth string.
- **Language tag:** `en`. **Seed:** `7` -- the same seed
  `scripts/validate-qwen3-tts-public.py`'s own smoke run already uses for
  this variant. **Threads:** `10`, fixed with the driver's own
  `synth_context_set_threads`. **Backend:** `cpu`. **`max_output_frames`:**
  left at the driver's own default, `983040`.

```
printf '%s' "This is a test of Qwen three T T S voice design synthesis." | \
  /usr/bin/time -v build/rel-dgx-spark/bin/synthesize-qwen3-tts-public-real \
  models/qwen3-tts-12hz-1-7b-voicedesign/qwen3-tts-12hz-1-7b-voicedesign-<PROFILE>.gguf \
  <out.pcm> \
  "desc:A cheerful, bright female voice speaking with fast pacing and high energy." \
  en 7 983040 cpu 10
```

`frames`, `load_seconds` and `synthesis_seconds` are read from the driver's
own stdout JSON; peak RSS is `/usr/bin/time -v`'s `Maximum resident set
size`. Three repetitions per profile, run back to back with nothing else
compiling or synthesizing on the host; medians are reported.

### Latency, RTF and peak memory, measured on Release 2026-08-20

**Every figure below comes from `build/rel-dgx-spark`.**

| profile | backend | frames | audio | synthesis | **RTF** | load | peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | CPU | 97,920 | 4.080 s | 19.30 s | **4.73** | 2.18 s | 4.95 GiB |
| F16 | CPU | 103,680 | 4.320 s | 7.12 s | **1.65** | 2.13 s | 4.95 GiB |
| Q8_MIXED | CPU | 120,960 | 5.040 s | 5.31 s | **1.05** | 1.41 s | 3.16 GiB |

**The BF16 CPU row above was re-measured in Plan 3 Task 6** (n=8, median
18.92 s / RTF 4.64, against this table's own n=3 / 19.30 s / RTF 4.73) as
part of pairing it against a same-session CUDA run; see "Stage 3: VoiceDesign
Package, Plan 3 Task 6" below for which figure to use and why they differ.
Task 8's card should read from the later measurement.

Medians of three runs per profile. RTF is each row's own synthesis time over
its own audio length, never across rows -- the three profiles stop at
different frame counts (97,920 / 103,680 / 120,960), which is a property of
their weights, not an error: the autoregressive stop decision is what moves
here too, the same one-clause mechanism `qwen3-tts.md:2937` names for why it
moves between builds, just triggered by weight precision rather than a
compiler's floating-point choices. Matches Base's own precedent that BF16 and
Q8_MIXED need not agree on where they stop.

**Two known contaminants, named beside the table rather than in a footnote.**
The second BPE frontend costs about +45 MB of peak RSS and is inside every
figure above -- carried forward from Base's own Task 14 measurement
(carry-over §3.4: 5,594,676 to 5,639,716 KB), not independently
re-measured against this package in this task, since isolating it would need
a build variant this task does not have. And generated length is
build-dependent (`qwen3-tts.md:2932-2942`, not `docs/testing.md`, which
documents a different claim -- the Release/RelWithDebInfo speed gap): none of
the frame counts above may be read against any other tree, including
`build`'s own tolerance-cell runs, which use different requests and a
different tree entirely.

**Gate 6, repeated runs and cleanup.** Every repetition at the fixed seed
produced byte-identical frame counts within its own profile -- 97,920 (BF16),
103,680 (F16), 120,960 (Q8_MIXED) across all three reps each -- and peak RSS
varied by under 0.01% between repetitions of the same configuration (BF16
0.0006%, F16 0.009%, Q8_MIXED 0.0095%), all comfortably inside Base's own
<0.03% bound. Synthesis-time spread within a profile's three reps stayed
under 2.4% (tightest: F16 at 0.26%), nothing like the multi-times spread
Base's Task 14 had to discard as contention -- **no run in this table was
discarded**, unlike Base's, whose report is the reason this task checked for
contention at all.

**Sanity check against Base (brief Step 5).** Base measured BF16 RTF 3.15 on
CPU at 0.6B. VoiceDesign's talker is 3.2x larger, and its BF16 RTF came out
*worse*, not better: **4.73**, about 1.50x Base's own number
(4.7305 / 3.15 = 1.502) -- the expected direction for a larger model doing
more compute per generated frame, so the brief's re-measure trigger ("RTF
better than Base's is surprising") did not fire and nothing was re-measured
on that account. VoiceDesign's fastest profile, Q8_MIXED at RTF 1.05, is
likewise slower than Base's own Q8_MIXED at 0.863 -- same direction, same
reason.

### Does F16 pay here? Yes -- decided on speed, since size already said no

Task 2 measured F16 at +283,136 bytes against its BF16 source (0.0066%
larger) and declined to draw a publication conclusion, because
`docs/quantization.md:455-459` records F16 as this family's speed profile,
not its size one -- Base's own F16 published despite the same larger-than-
source shape. The question left for this task: **is F16 fast enough on this
variant to justify a profile that saves no disk?**

**Yes, decisively, on `build/rel-dgx-spark`.** F16 synthesizes 2.71x faster
than BF16 for the same request -- 7.12 s median against 19.30 s. Because the
two profiles stop at different frame counts (103,680 vs 97,920, the same
autoregressive-stop-decision mechanism named above), the fair,
audio-length-normalized comparison is RTF, not raw wall time: **4.73 -> 1.65,
a 2.87x improvement** (4.7305 / 1.6482 = 2.870). That crosses real time with
headroom to spare. Scored against this same table's own Q8_MIXED result (RTF
4.73 -> 1.05, 4.49x), F16 alone captures **87% of Q8_MIXED's wall-clock time
saved** (12.18 s of 13.99 s: `19.3006 - 7.1204` over `19.3006 - 5.3069`) and
**84% of its RTF-point improvement** (3.08 of 3.68 points:
`4.7305 - 1.6482` over `4.7305 - 1.0530`) -- F16 alone recovers most of what
quantizing all the way to Q8_0 buys, while changing nothing about the
package's on-disk footprint.

**Why, mechanically -- a source-confirmed explanation, not asserted.** ggml's
CPU backend (commit `707321c4`, `GGML_SYSTEM_ARCH: ARM` on this aarch64/GB10
host) has no ARM-vectorized GEMM path for BF16 weights, and does have one for
F16. `ggml/src/ggml-cpu/llamafile/sgemm.cpp`'s `case GGML_TYPE_BF16` (line
3781) branches only on `__AVX512BF16__`, `__AVX512F__`, `__AVX2__`, `__MMA__`
(POWER) and RISC-V's `__riscv_zvfbfwma` -- none of which this host defines --
and falls through to `return false` when none match, which sends BF16
matmuls to ggml's generic reference path instead of a tuned kernel.
`case GGML_TYPE_F16` (line 3845), by contrast, has an additional
`__ARM_NEON` branch that this host's build does take. The scalar dot product
tells the same story: `ggml_vec_dot_bf16` (`ggml/src/ggml-cpu/vec.cpp:139`)
vectorizes only under AVX512BF16/AVX512F/AVX2/AVX and RISC-V's
`zvfbfwma` -- there is no ARM branch at all, so it falls to the scalar tail
loop on this host, while F16's own dot product has broad ARM NEON support.
This is an architecture/build fact about this ggml commit's CPU backend on
this host, not a numerical-precision claim: Task 2 already showed F16 costs
nothing measurable on accuracy either (replay headroom 3.45x, essentially
identical to BF16's own 3.44x).

**Verdict: recommend ship, subject to two gates this task does not clear.**
This task's own measurement supports it on the evidence: **2.87x RTF
improvement on this variant, for a package 0.0066% larger than its source.**
That is the arithmetic that answers the question Task 2 could not -- F16 is
faster by more than enough to justify shipping a profile that saves no disk.
But two things stand between this recommendation and a shipped package, and
neither is this task's to close: **Task 7's blind A/B listening audit
explicitly covers BF16-vs-F16 and has not run yet**, and **publication itself
needs jiangzhuo's per-act confirmation naming the target repository** (this
plan's own Global Constraints). Task 8 should carry F16 in the card as a
speed profile on this evidence once both gates clear -- not before.

### Open item: Base's own F16 verdict rests on size alone, and was never speed-measured

`docs/quantization.md:447`'s table row for Base's F16 reads "**clears every
gate and does not pay**", RTF "not measured" -- and this file's own Status
paragraph (line 21) repeats it verbatim: "F16 clears every gate and does NOT
pay -- it is 184,448 bytes *larger* than its source." Only the speed-profile
*paragraph* just below that table (`docs/quantization.md:455-459`, "F16 is a
speed profile for this family rather than a size one") supports this task's
own framing that F16 should be judged on speed; Base's recorded *verdict* is
the opposite, and it was reached without an RTF number at all. This task does
**not** claim VoiceDesign's result matches how Base's F16 is characterized --
only the paragraph agrees; the verdict does not.

This task's own mechanism finding makes the gap bigger than a wording
mismatch. The ARM-vectorized-GEMM-for-F16-but-not-BF16 fact traced above
(`ggml/src/ggml-cpu/llamafile/sgemm.cpp`, `ggml/src/ggml-cpu/vec.cpp`) is
**host-wide**: it is a property of this ggml commit's CPU backend dispatch on
this aarch64/GB10 host, not of VoiceDesign's weights specifically, so it
would apply identically to Base's own BF16 and F16 packages on the same tree.
That makes it likely -- **not measured here, and out of this task's scope to
measure** -- that Base's committed "does not pay" verdict is itself
wrong-and-unmeasured, in the same class of gap as CustomVoice's card silently
omitting `Q5_K_MIXED` (Task 4's own "Open item," above): a real, pre-existing
inconsistency this task found but does not fix. Left for whoever next touches
Base's own quantization record or re-runs Task 14's measurement tree.

### Q5_K_MIXED, measured for context only -- not a shipping candidate

Q5_K_MIXED already failed its own accuracy gate in Task 4 (replay headroom
0.32x, both cases breaching the committed 0.01 bound by roughly 3x) and
Task 4's recommendation -- do not publish -- does not change on a speed
result, so what follows is orientation, not a fourth row in the table above.
Same workload, same tree:

| profile | backend | frames | audio | synthesis | RTF | load | peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q5_K_MIXED (context; fails Task 4's accuracy gate) | CPU | 96,000 | 4.000 s | 4.30 s | 1.08 | 1.07 s | 2.40 GiB |

Medians of three runs, byte-identical frame counts across all three, peak RSS
varying under 0.015% between repetitions. Q5_K_MIXED lands slightly *behind*
Q8_MIXED on RTF (1.08 vs 1.05) despite its smaller package -- both are well
inside real time and the difference is not the deciding evidence either way,
since Task 4's accuracy failure already settles this profile's publication
question on its own.

### What is not claimed

One machine, one workload, one utterance length, one Description Text
instruct. This document's own build-dependent-length finding
(`qwen3-tts.md:2932-2942`), not `docs/testing.md`, is why an RTF computed
from a frame count taken on another tree would be wrong; every figure above
is same-tree. This section does not re-open Task 4's
Q5_K_MIXED recommendation, does not claim CUDA placement (VoiceDesign has no
new graphs to place -- see this plan's own "What This Variant Does NOT Have"),
and implies no listening judgement -- a tolerance table and an RTF number are
not audible evidence, and no VoiceDesign listening audit has run as of this
task.

### Verification

- No production code, tolerance cell or tensor-classifier logic touched --
  this task's whole diff is prose in this file.
- `synthesize-golden-manifest-contract` / `synthesize-tolerance-coverage`:
  unaffected, unchanged by this task.
- Full unit gate, `build` tree (`SYNTH_BUILD_INTEGRATION_TESTS=ON` per Task
  2's still-standing reconfigure, `CMAKE_BUILD_TYPE=Release`): the same two
  pre-existing, not-ours failures (`synthesize-python-api-wheel-test`,
  `synthesize-vits-python-unit`'s sole error
  `test_quantization_reports_match_current_artifacts`). No third.
- `scripts/ci/clang-format.sh --check-diff`: clean (only `.md` changed).
- `git status --porcelain` checked before staging; no `.gguf`, `.pcm` or
  `/usr/bin/time` log staged. `models/` is a symlink outside the worktree,
  gitignored via `/models`.

### Fix round 1, review of this task

Spec passed; code quality review found four Important and four Minor
findings, all in prose -- the reviewer independently re-ran all three
profiles (largest delta -3.7% on BF16, well inside the review's own 15%
trigger), confirmed frame counts byte-identical, and verified the ggml
mechanism down to the preprocessor level (`__ARM_NEON` defined,
`__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` not, no `-march` flags -- the
`#elif defined(__ARM_NEON)` arm at `sgemm.cpp:3872` really is the one this
build takes). None of the four measured figures moved.

**Important -- the stale-citation narrative was itself wrong, in the
paragraph that claimed to have verified it.** The original text said the
plan's own `docs/quantization.md:449-453` citation "was correct before Task
4's edit inserted six lines above it." Re-verified with the reviewer's own
named command (`git show b5046c8:docs/quantization.md`): line 449 there is
**blank**, and the "F16 is a speed profile" paragraph has lived at **450-454**
since the section was introduced (`b5046c8`, "Qwen3-TTS Stage 2 Plan 4 --
quantization, backends and ship prep", PR #13) -- `:449-453` was never
correct on this branch. The paragraph did not move in Task 4's own main
commit (`6edf4b1`, confirmed still at 450-454 there); Task 4's *fix round*
(`a5f6be4`) is what moved it, five lines down to 455-459 -- +1 from turning a
two-sentence roster line into three, +4 from the four new VoiceDesign rows
that commit added to the "What each profile is for" table (verified via
`git diff 6edf4b1 a5f6be4 -- docs/quantization.md`). The false narrative is
removed rather than corrected in place, since -- unlike the Status
paragraph's own longer-lived corrections -- it never described a real
interval, only a wrong derivation. A second, independent citation to the same
paragraph, in this file's own Task 2 section, read `docs/quantization.md
:450-454` -- correct when Task 2 wrote it, stale for the same reason -- and
is now `:455-459` too, so this file no longer carries two different ranges
for one paragraph.

**Important -- `docs/testing.md` was cited for a claim it does not contain,
in three places, one of them pre-existing.** Grepped: `docs/testing.md` never
says "build-dependent" anywhere; its Release-preset section documents a
different claim, the -O2/-O3 *speed* gap. The real source is this same file,
`qwen3-tts.md:2932-2942` ("Generated length is build-dependent, and no frame
count may be quoted without naming its build"). Fixed at both of this task's
own citations and, since the reviewer traced the error to its origin, at
Base's own Task 14 text too (`qwen3-tts.md:3466` as originally written) --
that citation was already wrong before this task copied its shape into new
prose.

**Important -- the SHIP verdict read as settled; it is not, yet.** Two gates
were absent from the verdict paragraph itself: Task 7's blind A/B audit
explicitly covers BF16-vs-F16 and has not run, and publication needs
jiangzhuo's own per-act confirmation naming the target repository (this
plan's own Global Constraints). Task 3's and Task 4's own verdicts already
got this right ("no publication conclusion drawn"; "do not publish") --
Task 5's verdict now reads "recommend ship, subject to two gates this task
does not clear," both gates named in the same paragraph rather than in a
subsection thirty lines later.

**Important -- claimed agreement with Base's own F16 characterization where
none exists; recorded as a named tension instead.** The verdict originally
said VoiceDesign's F16 result "match[es] how Base's own F16 is already
characterized in `docs/quantization.md`." True only of the speed-profile
*paragraph*; false of the table's own *verdict cell* for Base's F16
("clears every gate and does not pay," RTF never measured) and of this
file's own Status paragraph, which repeats that verdict verbatim. Removed
the agreement claim and added "Open item: Base's own F16 verdict rests on
size alone, and was never speed-measured," which states the tension plainly
and goes further than a wording note: this task's own mechanism finding is
host-wide, not VoiceDesign-specific, so it would very likely apply to Base's
own BF16/F16 packages on the same tree -- making Base's committed "does not
pay" verdict probably wrong-and-unmeasured, a gap in the same class as
CustomVoice's card silently omitting `Q5_K_MIXED` (Task 4's own Open Item).
Base is explicitly **not** re-measured here -- out of this task's scope --
and the item is left for whoever next touches Base's own record.

**Minors, all fixed.** The frame-count-divergence sentence now cross-references
`qwen3-tts.md:2937`'s one-clause mechanism ("the autoregressive stop decision
is what moves") instead of asserting the divergence is non-alarming without
saying why. The "Does F16 pay here?" subsection now names its build
(`build/rel-dgx-spark`) in the sentence that first quotes frame counts,
matching the Q5_K_MIXED subsection's own "same workload, same tree" framing.
"2.71x shorter" (a duration) is now "2.71x faster" (a rate). And "F16 alone
recovers roughly two-thirds of what Q8_0 buys" -- a ratio of speedup
multipliers (2.870 / 4.4926 = 0.639) that understates F16's own contribution
-- is replaced with the two framings that measure it directly: **87%** of
Q8_MIXED's wall-clock time saved (12.18 s of 13.99 s) and **84%** of its
RTF-point improvement (3.08 of 3.68 points), both with the subtraction shown.

Re-ran `scripts/ci/clang-format.sh --check-diff` (clean) and confirmed
`git status --porcelain` clean after this round. No measurement, table
value, or arithmetic result changed -- every fix is in what the numbers were
cited from or said to mean, matching the class of finding Tasks 2 and 3 each
hit in their own fix rounds.

## Stage 3: VoiceDesign Package, Plan 3 Task 6

What CUDA buys this variant end to end, on Task 5's own fixed workload,
reused verbatim for comparability. The design spec's expectation, with its
reasoning: Stage 1's placement puts the codec decoder on the device and holds
the autoregressive half on the CPU by the discrete-outputs rule; this
variant's talker parameters run 3.2x larger than Base's per layer -- 15.73 M
-> 50.33 M, attention 6.29 M -> 12.58 M and MLP 9.44 M -> 37.75 M, at an
unchanged 28 layers (design spec D7,
`docs/superpowers/specs/2026-08-18-qwen3-tts-stage-3-design.md:227-229`) --
so the AR half's share of wall clock grows, and CUDA's end-to-end gain is
expected **smaller than Base's ~6%, not larger**. A smaller gain would
confirm the model, not undermine it.

### What CUDA reaches on this variant, before any number

Unlike Base, this variant introduces no new graphs at all -- no
reference-audio path, so no ECAPA speaker encoder and no codec encoder for
ICL enrollment. It is Stage 1's own graph set (text frontend, talker, codec
decoder) at the 1.7B talker's width. There is therefore no placement
*decision* to make here, only a measurement of the one Stage 1 already made.

This variant's tolerance grid tracks exactly two stages, and only one of them
can move:

| stage | does a CUDA run measure anything different? | why |
| --- | --- | --- |
| `public` | **yes** | the Stage 1 codec-decoder twin moves under `SYNTH_BACKEND_CUDA`, the same twin Base's own `public` cell exercises -- this variant's `synth_model_load` puts the same graph on the device |
| `replay` (`prefill` probe only) | **no** | `tests/qwen3_tts_voicedesign_prefill_real.cpp:289` calls `ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)` directly, so today's driver has no backend argument to pass -- but the deeper reason a backend argument would not help is `src/arch/qwen3-tts/model.cpp:573-585`: the device-mirroring loop only copies tensors whose name starts with `codec.decoder.` (14 characters, `strncmp` at `:584`), so the talker's own weights are never mirrored onto the device at all. The prefill graph runs entirely on the talker, so it has no device-side weights to run against regardless of what the driver requests |

One of this variant's two tolerance stages exercises CUDA, not Base's one of
three -- fewer stages exist here because there are fewer graphs to begin
with, not because more of them were declined.

### The measurement: CPU vs CUDA, Task 5's fixed workload, same tree

Every figure below comes from `build/rel-dgx-spark`, unchanged since Task 5's
own measurement: the runner binary predates Task 5's first commit
(`build/rel-dgx-spark/bin/synthesize-qwen3-tts-public-real` timestamped
05:51:30, Task 5's own commit `f849f40` at 06:06:37 the same morning) and
Task 5's fix round (`26609b2`) touched only prose, so the tree this task
measures against is the one Task 5 already measured against, not a
reconfigured one.

**CPU was re-measured rather than reused.** Task 5's own CPU BF16 median was
19.3006 s (n=3). This task's own CPU BF16 median, gathered fresh in the same
session as the CUDA runs below (n=8: five untimed plus three wrapped in
`/usr/bin/time -v` for peak RSS), is **18.9220 s** -- about 1.9% lower, inside
the run-to-run variance Task 5's own report already characterized (spread
under 2.4% within a profile) but large enough that pairing the CPU baseline
with the CUDA measurement from the same sitting removes any session-to-session
host drift as a possible confound in the backend comparison that follows.
Task 5's figure is not used below; this task's own paired measurement is.

```
printf '%s' "This is a test of Qwen three T T S voice design synthesis." | \
  build/rel-dgx-spark/bin/synthesize-qwen3-tts-public-real \
  models/qwen3-tts-12hz-1-7b-voicedesign/qwen3-tts-12hz-1-7b-voicedesign-BF16.gguf \
  <out.pcm> \
  "desc:A cheerful, bright female voice speaking with fast pacing and high energy." \
  en 7 983040 <cpu|cuda> 10
```

| profile | backend | frames | audio | synthesis | **RTF** | load | peak RSS | n |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | CPU | 97,920 | 4.080 s | 18.92 s | **4.64** | 2.29 s | 4.95 GiB | 8 |
| BF16 | CUDA | 97,920 | 4.080 s | 17.76 s | **4.35** | 2.37 s | 4.95 GiB | 8 |

Medians. Frame counts are byte-identical across all sixteen runs -- unlike
this plan's own named risk (the AR stop decision can move between CPU and
CUDA numerics), that risk did not materialize on this workload, so no
RTF-adjustment for differing audio length is needed here: both rows describe
the same 4.080 s of audio. Peak RSS medians of the three `/usr/bin/time -v`
runs per backend (CPU 5,185,860 KiB, CUDA 5,186,428 KiB -- a 0.011%
difference, not material) both round to the same 4.95 GiB Task 5 already
reported for this profile on CPU, a cross-check that this session's CPU
measurement is the same population as Task 5's. Synthesis-time spread stayed
under 2.4% for CPU (2.30%) and under 4.6% for CUDA (4.59%) -- wider than
CPU's own but nothing like a contention signature (no run doubled, unlike
Base's Task 14 discards), so no run was discarded.

### The gain, division shown

**18.92 s -> 17.76 s and RTF 4.64 -> 4.35**, which divide out to
(18.92 - 17.76) / 18.92 = **6.13%** and (4.64 - 4.35) / 4.64 = **6.25%** using
the table's own rounded figures -- the two percentages differ from each
other only because of that rounding. Base's own family-record text
(`qwen3-tts.md:3440-3441`) records its own pair the same way, at one decimal
place: "11.60 s -> 10.85 s ... RTF 3.15 -> 2.95 -- 6.5 % and 6.3 %
respectively". The design spec's own erratum commits the same pair at two
decimal places instead, verbatim:
"11.60 s -> 10.85 s and RTF 3.15 -> 2.95, which divide out to 6.47% and
6.35%" (`docs/superpowers/specs/2026-08-18-qwen3-tts-stage-3-design.md:637`)
-- both roundings of the same unrounded quotient are committed in this tree,
one decimal place in the family record and two in the design spec, and the
task brief's own "6.47% and 6.35%" citation is the spec's figure, not a
fabrication. The precise, unrounded medians for this task -- 18.9220 s and
17.7564 s (n=8 each) -- divide to **6.16%**, and RTF's own unrounded pair,
4.6377 -> 4.3520, divides to the identical **6.16%**: not a coincidence but
an algebraic identity, since both backends produced the same 97,920-frame,
4.080 s audio for this workload, so RTF's shared denominator cancels and the
two percentages restate one fraction rather than measuring two independent
things that happen to agree.

**Confirms the smaller-than-Base's-~6% prediction, narrowly.** 6.16% here
against Base's own 6.47%/6.35% unrounded (6.5%/6.3% as the family record
displays them, `qwen3-tts.md:3440-3441`) is smaller, in the direction the
design spec's reasoning predicted -- but by about 0.2-0.3 percentage points,
not by the large margin a literal reading of "the talker's per-layer
parameters run 3.2x larger, so its share of wall clock grows 3.2x" might
suggest. Stated plainly: the direction is right: **smaller, not larger**,
but the model's single stated mechanism (AR share growing) does not by
itself explain why the margin is this narrow.

**Why the margin is narrow, not large -- inferred from a source-read fact,
not independently re-derived.** The codec decoder -- the only graph CUDA
moves here -- is byte-identical between **Base** (the variant this task's
gain is measured against) and VoiceDesign: 255 tensors, 457,161,476 bytes,
same shapes, in both (`gguf.GGUFReader` against the `codec.decoder.*`
prefix -- one level up from an earlier draft of this paragraph, which named
`codec.decoder.decoder.*` and thereby described only 118 of these 255
tensors and 209,480,452 of these 457,161,476 bytes, a nested
weight-normalized-conv submodule rather than the whole decoder; the reported
counts were always the full-prefix ones, only the selector's own name was
wrong). `codec.decoder.` (14 characters) is also the exact prefix
`src/arch/qwen3-tts/model.cpp:585` mirrors onto the device, so this is the
same tensor set the runtime twin actually moves. Re-hashed for this fix
round rather than trusted from shape/size agreement alone: sha256 over
tensor name and data for every `codec.decoder.*` tensor is
`42772743b165...` identically across **all three** committed BF16
packages -- CustomVoice, Base and VoiceDesign -- so the codec decoder is one
shared, unmodified artifact across the whole family, not merely matched
between the two variants this paragraph compares.

The 3.2x figure this section opened with is a talker-parameter-count fact
(design spec D7, `docs/superpowers/specs/2026-08-18-qwen3-tts-stage-3-design.md:227-229`
-- talker parameters per layer, 15.73 M -> 50.33 M) and does not touch the
codec decoder at all. If the codec decoder's own per-frame cost -- and
CUDA's saving from it -- tracks frame count rather than talker size, and
this workload's frame counts are close between the two variants' own
fixed-workload measurements (97,920 here against Base's 88,320, +10.9%),
then the CUDA-accelerated numerator changes only a little between variants
while the AR-dominated wall-clock denominator grows -- which points toward a
small negative move in the percentage, not a large one. This is inferred
from the tensor-identity finding and the two variants' own frame counts;
this task did not isolate the codec decoder's own wall-clock cost in either
build the way Stage 2 Plan 4 Task 12 isolated the speaker and codec
*encoders*, so the size of the narrowing is not independently confirmed the
way the direction is.

### Structural agreement: CPU and CUDA disagree in bytes, as TF32 predicts

Two independent pieces of evidence, both measured this task, not asserted:

- `scripts/validate-qwen3-tts-public.py --backend cuda --profile BF16` against
  the same two descriptions the CPU `public` cell already uses ("A cheerful,
  bright female voice speaking with fast pacing and high energy." / "A deep,
  calm male voice speaking slowly and quietly.", text `"Hi."`, its own
  default): **all 11 checks pass, 3 skipped** for the same
  `description_text`-package reasons the CPU cell already records -- one of
  the eleven (the empty-instruct-vs-oracle relation) passes through the
  `replay` stage's own CPU-only prefill-probe figure rather than recomputing
  anything under CUDA, identically to how the CPU cell reports it. Seed 7's
  own digest differs from the CPU cell's -- `dcc9279c69679412` (CUDA) against
  `5e772fc5cc8b2a90` (CPU), reports at
  `reports/validate/qwen3-tts/public-voicedesign-BF16-cuda.json` and
  `reports/validate/qwen3-tts/public-voicedesign.json` -- which is what says
  the request actually reached the device, the same shape Base's own Task 13
  recorded.
- On Task 5's own fixed workload (a different case from the validator's
  `"Hi."` one, deliberately, so the agreement figure is not read off the same
  text the structural checks used): comparing this task's own CPU and CUDA
  seed-7 PCM output sample-for-sample (both 97,920 frames), **cosine
  0.9999988, max_abs 0.002414**. That is on the ~1e-3 TF32-deviation scale
  `docs/backends.md` predicts for a CUDA F32 matmul, and it is *not* the same
  number as Stage 1's own `audio.pcm` CUDA-vs-CPU figure (cosine 0.999404,
  max_abs 0.1602, a different package and a different case) -- cited here for
  the same shape of comparison, not as an equal one.

### The tolerance file: no CUDA sub-grid, for the same structural reason as Base

`tests/python/test_tolerance_coverage.py`'s per-backend check
(`:152-158`) requires any `backends.CUDA` sub-grid to carry the profile's
full stage set exactly -- for this variant, {`public`, `replay`}. `replay`'s
only probe is structurally CPU-only (previous section), so a `replay` cell
under a `CUDA` key could only ever hold `replay`'s own CPU numbers copied
over -- exactly the "recording CPU figures under a CUDA key" outcome Base's
own Task 13 declined for the same coverage rule.

**The sub-grid is therefore not committed.** `tests/tolerances/qwen3-tts.json`
gains no `backends` key on any `qwen3-tts-12hz-1-7b-voicedesign` profile; a
short paragraph was appended to the variant's own `note` field pointing here
and at `docs/backends.md`'s own recorded-absence subsection for this variant.

**No CUDA threshold was derived**, matching Base's own "nothing here needed
one" reasoning, and for a simpler reason than Base's own: Base still had to
say a threshold for a *hypothetical* future `codec_encoder` CUDA cell was
not owed here (its declined speaker/codec-encoder twin might someday be
revisited). This variant has no comparable third graph -- Step "What CUDA
reaches" above exhausts its graph set at `public`'s codec decoder and
`replay`'s prefill probe -- so there is no hypothetical cell to pre-derive
a threshold for either. The cosine and max_abs figures measured above are
recorded as evidence that the placement moves the audio, not as an input to
any committed gate.

### What is not claimed

One machine, one workload, one build tree, one profile (BF16). F16,
Q8_MIXED and Q5_K_MIXED were not measured under CUDA in this task -- Task 5
already measured their CPU speed, and this task's own brief pins its Step 2
command to `--profile BF16`, not a full profile x backend cross.
No listening judgement is implied: a numeric agreement figure and an RTF
number are not audible evidence, and no VoiceDesign CUDA listening comparison
has run. No publication conclusion is drawn, and no movement of the
Validation Level is claimed -- `quality_evaluation` stays deferred per
ADR 0017.

### Verification

- No production code touched -- this task's diff is prose in this file and
  in `docs/backends.md`, plus one appended paragraph in the
  `qwen3-tts-12hz-1-7b-voicedesign` variant's own `note` field in
  `tests/tolerances/qwen3-tts.json` (no tolerance cell, no `backends` key
  added; round-tripped through `json.load`/`json.dumps(indent=2)` before
  editing to confirm the file's existing formatting survives an
  edit-and-reserialize).
- Full unit gate, `build` tree (`SYNTH_BUILD_INTEGRATION_TESTS=ON`,
  `CMAKE_BUILD_TYPE=Release`): the same two pre-existing, not-ours failures
  (`synthesize-python-api-wheel-test`, `synthesize-vits-python-unit`'s sole
  error `test_quantization_reports_match_current_artifacts`). No third.
- `scripts/ci/clang-format.sh --check-diff`: clean (only `.md`/`.json`
  changed).
- `git status --porcelain` checked before staging: no `.gguf`, `.pcm` or
  `/usr/bin/time` log staged; all scratch files for this task live under
  `/tmp/qwen3-tts-task6/`, outside the worktree.

### Fix round 1 (review response)

Spec passed; code quality review found four Important and five Minor
findings, all in prose -- the reviewer mutation-tested the coverage rule
(confirmed a `public`-only `backends.CUDA` sub-grid fails
`test_measured_families_cover_every_profile_backend_and_stage`), reproduced
the CPU/CUDA runs within 1.15%/0.24% of this task's own medians, and
reproduced the PCM cosine to seven decimal places. No measurement, table
value, or arithmetic result changed in this round -- every fix is in what a
number was cited from or said to mean, or a fact was attributed to the wrong
selector or the wrong sibling package. Four Important findings, all
addressed:

1. **A fabricated quotation.** "this task's own brief scopes to 'at least
   BF16'" put quote marks around a phrase that appears nowhere in the brief.
   Replaced with what the brief actually does: its Step 2 command pins
   `--profile BF16`.
2. **The tensor-identity selector's own name didn't match its own numbers.**
   The reported 255 tensors / 457,161,476 bytes are correct for
   `codec.decoder.*` (the exact 14-character prefix
   `src/arch/qwen3-tts/model.cpp:585` mirrors) but the prose named the
   selector `codec.decoder.decoder.*`, a nested submodule that is only 118
   of those tensors and 209,480,452 of those bytes -- confirmed by re-running
   both selectors. The selector's name is fixed; the numbers were already
   right.
3. **Byte-identity proven against the wrong sibling.** The paragraph
   explaining why the CUDA gain is close to Base's own compared VoiceDesign
   against CustomVoice, not Base -- the variant this task's gain is actually
   measured against. Re-hashed `codec.decoder.*` (tensor name and data,
   sha256) across all three committed BF16 packages this round:
   `42772743b165...` identically in CustomVoice, Base and VoiceDesign, so
   the paragraph now states the identity against Base directly and cites the
   three-way match.
4. **"3.2x larger" cited to a line that records 2x.** Two citations pointed
   at `qwen3-tts.md:311` (hidden 1024 -> 2048, a 2x fact) for the 3.2x
   figure and called it "a talker-hidden-size fact". The real source is
   design spec D7
   (`docs/superpowers/specs/2026-08-18-qwen3-tts-stage-3-design.md:227-229`):
   talker parameters per layer, 15.73 M -> 50.33 M, a factor that comes from
   attention (2x) and MLP (4x) scaling together, not hidden size alone.
   Re-cited to D7 throughout; no longer called a hidden-size fact.

Five Minors, all addressed: the brief's "6.47%/6.35%" is not a fabrication --
the design spec's own erratum (`...design.md:637`) commits that exact pair
verbatim, two decimal places where the family record's own text
(`qwen3-tts.md:3440-3441`) carries one (6.5%/6.3%); both are real citations
to different committed documents, and the paragraph now says so instead of
calling the brief's figure something the tree doesn't contain. The tolerance
JSON's "as Base's own declined sub-grid above" pointed at this file, where
Base's own entry records no such decline -- repointed to
`docs/backends.md:506` and `qwen3-tts.md:3205`, where it actually lives.
Both docs' `replay`-immobility explanation now cites the deeper structural
fact, `model.cpp:573-585`'s device-mirroring loop copying only
`codec.decoder.`-prefixed tensors (so the talker has no device-side weights
regardless of what any driver requests), alongside the pre-existing
driver-hardcoding fact. `docs/backends.md`'s "11 of 11 applicable checks
pass" now notes that one of the eleven reuses a CPU-only prefill-probe
figure rather than recomputing anything under CUDA (this file's own parallel
sentence gained the same clause). Task 5's own BF16 CPU table row (19.30 s /
RTF 4.73) now carries a forward pointer to this task's later re-measurement
(18.92 s / RTF 4.64, same profile, later session), since Task 8 reads the
card off these tables and would otherwise see two disagreeing rows with no
signpost between them.

Re-ran `scripts/ci/clang-format.sh --check-diff` (clean) and the full unit
gate (98/106, same two pre-existing failures, no third) after this round.

## Open Questions for Intake

1. Confirm the codec decoder topology against upstream rather than against a
   port: SEANet stage ratios, ConvNeXt upsample factor, DAC strides, and whether
   SnakeBeta exponentials can be folded at load. **Mostly resolved 2026-07-27**
   from the checkpoint -- ratios, factors and quantizer geometry are under "The
   codec, measured against the tensors". What remains is the SnakeBeta folding,
   which needs the forward rather than the shapes.
2. ~~Confirm the RoPE section collapse is exactly equivalent for a
   text-plus-codec timeline.~~ **Resolved 2026-07-27**: exactly equivalent, and
   proven rather than argued. See "The multimodal RoPE collapses exactly".
3. ~~Measure real CPU speed for the 0.6B Stage 1 variant on project hardware.~~
   **Resolved 2026-07-27**: real-time factor 9.4 to 9.7 greedy on CPU, and that
   is a floor set by the first oracle configuration tried, not a measurement of
   the model as upstream deploys it (CUDA, bfloat16, FlashAttention 2) -- and not
   how this family's oracle ended up running either. Whether CUDA
   helps is a stage-7 policy decision, not an implication -- see "CPU oracle
   smoke". The cache-reset mitigation is still unmeasured here; it is a stage-4
   implementation concern.
4. ~~Enumerate the CustomVoice preset speakers and their dialect overrides for
   the Preset Voice Catalog.~~ **Resolved 2026-07-27**: nine speakers as codec
   token ids, two with dialect overrides. See "Voices and languages".
5. ~~Determine whether the causal codec decoder qualifies as Native Streaming
   Synthesis under `CONTEXT.md`.~~ **Resolved 2026-07-27**: Stage 1 claims
   Chunked Audio Delivery. See "Stage 1 claims Chunked Audio Delivery". The
   stronger claim stays reachable at a later stage with its own evidence, and
   its cost is named under the qwentts findings.
6. ~~Establish upstream provenance and redistribution permission for publishing
   converted Model Packages, as was done for Kokoro.~~ **Basis settled
   2026-07-27; published 2026-07-28.** Both the source at `022e286b` and the
   checkpoint at `85e237c1` carry an explicit Apache-2.0 grant, audited at the
   pinned revision rather than at `main`, with no restriction prose in either
   card. Alibaba does not disclose training corpora, so Apache-2.0 is the basis
   relied on, the same basis on which Kokoro was accepted. On that basis the
   package went live at `jiangzhuo9357/qwen3-tts-12hz-0-6b-customvoice-gguf`
   (created 2026-07-28, last updated 2026-07-29) carrying BF16, F16, and
   Q8_MIXED. Every later edit to that repository is a fresh outward act needing
   its own confirmation.

## Accepted Risks

- No arena or listening evidence supports a quality claim for this family, and
  the project has no capability to produce one. Delivery is `port_validated`.
- The 1.7B Stage 3 variant may not meet a CPU practicality bar that the 0.6B
  stages meet. Stage 3 may ship CPU-slow or not at all without invalidating
  Stages 1 and 2.
- Mature MIT prior art exists for this exact model. The project's value here is
  integration under one validated ABI, not novelty.
