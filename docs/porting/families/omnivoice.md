# OmniVoice Family Selection and Port Plan

Status: Confirmed 2026-08-07. Intake through the greedy synthesis core
(slices 4–6) done: single-forward parity, exact token grids 17/17, replay
waveform under committed tolerances (tests/tolerances/omnivoice.json). Plan 3
(the public sampled path, Reference Audio and Description Text Voice
Profiles, their Serialized Profile GGUF round-trip, the CLI and Python-wheel
Adapters, and this record's own close-out) is done as of Task 17. Plan 4
(Quantization Profile measurement, the validation-suite carry-over debt, the
CUDA Execution Backend, and ship) is done as of its own Task 17. Both
candidate Quantization Profiles were produced and both fail the clone path's
exact-token gate, so this family ships F32-only (see the quantization
section). The CUDA backend is claimed for the codec's decode path; the
generator stays on CPU by the discrete-outputs rule, and a separate
measurement confirmed it would not survive placement there anyway (see
Execution Backends and Open Questions below). The Listening Audit recorded
`no_obvious_regression` on all six audited pairs. Ship artifacts are prepared
as a Restricted Model Package (ADR 0018) under `models/publish/omnivoice-0-6b/`;
publication itself is a separate act awaiting jiangzhuo's per-act
confirmation. Plan 5's carry-over ledger is
`docs/superpowers/plans/2026-08-07-omnivoice-plan-5-carryover.md`.

## Decision

The fourth Model Family is **OmniVoice** (Xiaomi / Next-gen Kaldi, `k2-fsa`,
arXiv 2604.00688). It was recorded as the leading fourth-family candidate at
the third-family selection in `docs/model-family-selection.md` and confirmed on
2026-07-30; the design record is
`docs/superpowers/specs/2026-07-30-omnivoice-family-design.md`. The Reference
Model Variant is the flagship `k2-fsa/OmniVoice` checkpoint. The sibling
`k2-fsa/OmniVoice-Emilia` is **not** ported: it is a zh+en paper-reproduction
variant, it adds no capability this port needs, and its model card carries no
license text at all.

The publication ceiling for this family is a **Restricted Model Package**
(ADR 0018), not a Published Model Package. The deferral recorded at the
third-family selection — whether the project supports a family it can never
publish — is settled by that ADR: the restriction follows the weights whoever
runs the converter, so withholding publication would add friction for
legitimate non-commercial users without changing who may legally use the
weights.

Three separate grants apply, and the package carries all three verbatim.

**LM weights: CC-BY-NC, no version stated.** The upstream card has no
`license:` frontmatter key; the only statement is prose, quoted here exactly as
it appears:

> Our code is released under the Apache 2.0 License. The pre-trained model is
> licensed under the CC-BY-NC due to constraints from its training data (e.g.,
> Emilia).

Upstream names no CC-BY-NC version, so the model card says so rather than
inventing one.

**Codec weights: Boson Higgs Audio 2 Community License.** The weights
repository bundles `audio_tokenizer/LICENSE`, a Meta-Llama-3-derived agreement
that permits redistribution with the agreement text, attribution and naming
obligations, caps commercial use at 100k monthly active users, and forbids
using the outputs to train other models. The July 2026 deferral note recorded
only CC-BY-NC; this third license is a finding of the 2026-07-30 audit, and it
is what makes the codec text a declared Sidecar Resource rather than a
footnote.

**Code: Apache-2.0**, covering the GitHub source at `k2-fsa/OmniVoice` and
nothing else. No weight file, converted artifact, or GGUF produced by this
project may ever be labelled apache-2.0.

The card additionally carries a use disclaimer against unauthorized voice
cloning and impersonation; the rendered card must reproduce it.

Two rules follow from how these facts were obtained. License facts come from
the upstream model card and the files inside the weights repository at the
pinned revision — never from a port's README, a mirror, or a secondary
summary. And all three known community GGML ports misstate or omit these
terms; one Hugging Face GGUF repository labels the weights apache-2.0, which is
simply false. Nothing downstream is inherited.

## Reference Contract

The oracle is the pinned `k2-fsa/OmniVoice` Python package, installed by git
revision in `scripts/envs/omnivoice/pyproject.toml`, driving the pinned Hugging
Face weights of the same name.

| Pin | Value |
| --- | --- |
| Source revision | `468e927ba3716cd8dd86421148dfb3046e9f9d7b` (package 0.2.1) |
| Weights revision | `c5fdb5ccb189668d56333f77ba2629f4cd7535f4` |
| Checkpoint files | 13, totalling 3,267,470,260 bytes |
| Generator | `model.safetensors`, 2,450,344,112 bytes, 313 tensors |
| Codec | `audio_tokenizer/model.safetensors`, 805,665,628 bytes, 527 tensors |

The per-file SHA-256 digests, the full file list, and the tensor inventory are
recorded in `reports/porting/omnivoice/omnivoice-0-6b/intake.json` and
`tensor-inventory.json` beside it; this document does not restate them.

The oracle runs **F32 on CPU**. The checkpoint is F32 throughout, so
`docs/port-validation.md`'s two rules — dtype follows the checkpoint, and the
least approximating reproducible device wins — agree here instead of pulling
apart. This is the opposite of the qwen3-tts situation, where a bfloat16
checkpoint driven on CPU in F32 upcast every weight and described a model that
does not exist; there is no upcast to get wrong in this family.

`OmniVoice.generate` is the pinned entry point. Deterministic cases pin
`position_temperature=0.0, class_temperature=0.0`. Both being zero does more
than fix a seed: the decode loop takes neither Gumbel branch, so it makes **no
RNG call at all**. Greedy runs are therefore bit-reproducible across separate
processes, not merely repeatable inside one seeded process — the property that
made OmniVoice the fourth-family candidate, and the property the whole
validation strategy below is built on.

`class_temperature`'s upstream default is **0.0**; only
`position_temperature` defaults non-zero, at 5.0. So the public sampled path
**draws positions, not classes, unless a caller raises it**: at the defaults
the randomness decides which canvas positions a step commits, and the token
written at a committed position is still the argmax described under The decode
loop. A port that read both defaults as 5.0 would sample twice where upstream
samples once. Plan 3's sampler is where the class branch is implemented and
where a non-zero `class_temperature` first has a path to take.

Every oracle run, greedy or sampled, pins `postprocess_output=False,
pad_duration=0.0, fade_duration=0.0`. There is no pydub in the contract:
silence trimming, fades, and padding are upstream product behavior, not model
behavior. Clone cases additionally pin `preprocess_prompt=False` and pass an
explicit `ref_text`, so no silence stripping and no Whisper transcription enter
the comparison. Whatever scaling or normalization remains unconditionally
active inside the decode-and-post-process path with those switches off *is*
part of this family's contract; intake measures exactly what survives and the
port mirrors it.

Native output is 24 kHz mono F32 at a 25 Hz frame rate, so one codec frame is
exactly 960 samples. The model consumes raw text through a Qwen2 byte-level BPE
tokenizer that ships as a single `tokenizer.json` in the weights repository.

`trust_remote_code` is not required, so `docs/scope.md`'s rule against
executable code in a Model Package is not engaged. Verified at intake rather
than assumed, five ways: the weights repository contains no `.py` file at all at
the pinned revision; neither `config.json` nor `audio_tokenizer/config.json`
declares an `auto_map` key; the string `trust_remote_code` appears nowhere in
the pinned package source; the model is loaded by `OmniVoice.from_pretrained`
from the pinned pip package — Apache-2.0 package code — rather than by
`transformers.AutoModel` with Hub-hosted code; and the codec is a first-class
transformers architecture (`transformers/models/higgs_audio_v2_tokenizer/`),
not remote code either.

## Architecture

OmniVoice is a non-autoregressive mask-predict model — discrete masked
diffusion, not autoregression. A Qwen3-0.6B backbone is run with bidirectional
attention over a fixed-length canvas of 8 acoustic codebooks × 1025-entry
vocabulary at 25 Hz, refined over 32 parallel denoising steps (16 in fast mode)
with classifier-free guidance at scale 2, which costs two forwards per step.
The committed token grid is decoded to 24 kHz audio by the Higgs Audio V2
codec's DAC-style convolutional decoder at hop 960. One 0.6B + 0.2B checkpoint
carries reference-audio cloning, described-attribute voice design, and
automatic voice selection together, with 600+ claimed languages. The audio mask
id is 1024. Every backbone layer is full attention, so there is no KV cache
anywhere in this family and no causal masking to get wrong.

The Language Capability Catalog is `en`, `zh`, `ja` — validated languages only,
primary-subtag BCP-47 per the qwen3-tts precedent. Widening it later is
validation work, not code work.

Voice arrives through three modes that map onto public Voice Profile sources
already declared in `include/synthesize.h`: cloning is **Reference Audio** (a
transcript is required, language optional), voice design is **Description
Text**, and auto-voice is the package default with no profile selected. The
Preset Voice Catalog is empty and the package default is unnamed. OmniVoice is
the first family to exercise the Reference Audio and Description Text sources.
**Auto-voice means the voice follows the synthesis seed**: with no profile
selected, nothing but the sampled path's draws picks the speaker, so a caller
who wants the same speaker twice reuses the seed the request reports — there is
no named default Voice to ask for instead.

### The prompt layout

```text
[<|denoise|>?] <|lang_start|>{code-or-None}<|lang_end|> <|instruct_start|>{instruct-or-None}<|instruct_end|> <|text_start|>{ref_text + " " + text}<|text_end|> [ref audio tokens] [T × mask(1024)]
```

Absent fields are the literal four-character string `None`, not an omission —
a request that names no language emits `<|lang_start|>None<|lang_end|>`. The
`<|denoise|>` marker appears only in clone mode, that is, only when reference
audio tokens are present.

The whole prompt is a grid 8 rows deep. **Every row of the 8-codebook
dimension repeats the same text ids**; the rows differ only where audio tokens
live. The embedding merge is correspondingly asymmetric, and it **selects
rather than adds**: `_prepare_embed_inputs` ends in
`torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)`, so a text
position carries the text embedding of row 0 alone, and an audio position
carries the sum of the eight codebook embeddings alone — each read at its own
offset into the shared audio embedding table (`arange(8) * 1025`). The text
embedding is computed at every position, audio ones included, and then thrown
away wherever the audio mask is set; nothing is ever summed across the two
streams. The target region is `T` frames of mask id 1024 in all eight rows,
where `T` is the canvas length fixed by the duration estimator.

### The decode loop

Each step runs one batched forward holding a conditional and an unconditional
sequence. The unconditional branch carries the target region only — no style
markers, no text, no reference audio — so the two branches have different
lengths and the batch is padded accordingly.

The per-step commit budget is a schedule computed once from the canvas size.
Timesteps are `num_step + 1` points linearly spaced on `[0, 1]` and then
shifted by

```text
t' = t_shift·t / (1 + (t_shift − 1)·t)
```

with `t_shift = 0.1` by default, which concentrates the early steps at low
signal-to-noise. Step `s` commits `ceil(total_mask * Δt)` positions, where
`total_mask = T × 8` and `Δt` is that step's shifted interval, clamped to what
remains unmasked; the final step commits the entire remainder, so the canvas is
always fully committed after `num_step` steps regardless of rounding.

Within a step the two logit sets are combined in log-softmax space:

```text
log_softmax(log p_c + s·(log p_c − log p_u))
```

with guidance scale `s = 2.0`. The mask id is then banned by setting its
log-probability to `-inf`, so the model can never commit a mask; the predicted
token per position is the argmax over the remaining vocabulary and the
confidence is that position's maximum log-probability. Confidence is penalized
by `layer_index × 5.0` — the layer penalty factor, default 5.0 — which biases
commitment toward the coarse codebooks first. Positions already committed in an
earlier step are set to `-inf` so they cannot be revisited. The commit itself
is a **flat top-k over the flattened 8 × T score grid**: one step may commit
several codebooks at one frame or one codebook across many frames, whichever
the scores prefer. Committed tokens are written back into both the conditional
and the unconditional branch before the next forward.

One deliberate non-fact: `audio_codebook_weights = [8, 8, 6, 6, 4, 4, 2, 2]` in
the checkpoint configuration is a **training-loss weighting** and plays no part
in synthesis. A reference port documents it as the sampler's per-codebook
penalty; that is wrong, the penalty is the `layer_index × 5.0` term above, and
this project must not repeat the claim.

### The sampler: a port-defined draw-order contract (Plan 3, Tasks 3–4)

Upstream draws one dense `rand_like` tensor over the whole logits row at once;
this port draws one uniform per decision instead, so the ORDER those draws
happen in is this port's own contract, not a transcription of anything
upstream states explicitly. It is recorded once, at the single `NormalRandomStream`
construction site (`src/arch/omnivoice/model.cpp`, immediately above the step
loop), and restated here so it lives in the family record too:

- **Candidates are visited in codebook-major, frame-minor scan order** — the
  nested loop that builds them (`for codebook in 0..codebooks: for frame in
  0..frames`) is the same order the draws happen in; there is no separate
  ordering step.
- **For each candidate, the class draws happen first.** Inside
  `choose_token_sampled` (only reached when `class_temperature > 0`), as many
  uniforms are drawn as survive its own top-k filter (`ceil(0.1 ×
  vocab_size)`), one per surviving class in **ascending class-id order** —
  never in score order, since the survivors are re-sorted by id after the
  top-k selection specifically so the draw order does not depend on the
  scores themselves.
- **Then, iff `position_temperature > 0`, exactly one further draw** perturbs
  that same candidate's already-computed score (`gumbel_perturb`, below) —
  the position draw, consumed after any class draws for that candidate, never
  before.
- **A step whose budget is zero draws nothing at all.** The `if (budget == 0)
  { continue; }` guard sits above every draw site in the per-candidate loop,
  so a zero-budget step does not construct a candidate, run `choose_token` or
  `choose_token_sampled`, or consume a single uniform — matching upstream's
  own `if k <= 0: continue`, which likewise consumes no randomness for that
  step.
- **One stream serves the whole synthesis**, constructed once before the step
  loop, and only if either resolved temperature is positive. A fully greedy
  synthesis (both temperatures exactly 0) constructs no stream and therefore
  draws nothing — the zero-RNG property the Reference Contract above already
  states, preserved by construction rather than by a separate check.

**The Gumbel perturbation, float32 throughout.** `gumbel_perturb(logit,
temperature, uniform)` transcribes upstream's `_gumbel_sample`
(`omnivoice.py:1632-1636`) as one value:

```text
scaled = logit / temperature
noise  = -log(-log(uniform + 1e-10) + 1e-10)
result = scaled + noise
```

Every operation is `float` — there is no `double` intermediate anywhere in
this expression — and the grouping matches upstream's own line-for-line
rather than an algebraically equivalent rearrangement, because reordering a
floating-point expression changes its rounding, and therefore can change
which of several perturbed candidates an argmax picks.

**`log_prob` is the row's max guided log-probability, not the chosen
candidate's own value.** `choose_token_sampled` computes `log_prob` as the
maximum over the FULL, unfiltered `guided` array — the same value
`choose_token`'s own argmax would report for that row — regardless of which
survivor the Gumbel draw actually commits. This transcribes upstream's
`confidence_scores = log_probs.max(dim=-1)[0]` at **`omnivoice.py:1449`**
verbatim. The distinction is not cosmetic: Task 3's review proved that
reading `guided[chosen]` instead — the value the port originally computed —
diverges from upstream's definition on 9,026 of 20,000 seeds at `keep=2`, and
a 400-seed regression fixture (`tests/omnivoice_sampler_test.cpp`) now pins
`log_prob` against the plain-greedy value on every seed.

**Why `class_temperature`'s upstream default is 0.0, and what that means for
the public path.** The Reference Contract above already states it: only
`position_temperature` defaults non-zero (5.0), so at package defaults the
public sampled path draws POSITIONS, never CLASSES — `choose_token_sampled`
and its class-draw/top-k machinery are reached at all only when a caller
raises `class_temperature` above zero, which the public Interface allows but
the shipped package's own defaults do not exercise. A port that read both
defaults as 5.0 would sample twice where upstream samples once; this sampler
contract is what makes that distinction concrete rather than a note in the
Reference Contract alone.

## Port Validation Fit

Greedy decoding is fully deterministic and makes no RNG call, so this family
does not need the captured-inputs machinery that qwen3-tts needs. Golden cases
assert that the **exact 8 × T token grid equals the oracle's** — this family's
definition of the `structural_exactness` check — and then compare the waveform
through the codec under committed tolerances. Both sides run F32 on CPU, so
`tests/tolerances/omnivoice.json` records reference stage
`source-f32-oracle-vs-f32-cpu`: an equal-arithmetic comparison with tight
thresholds, measured once and committed before support is declared.

The public sampled path is a different claim and is validated differently. Its
default `position_temperature` is 5.0 and its Gumbel noise is drawn from the
global torch RNG: **upstream exposes no seed parameter of any kind**. There is
no upstream stream to reproduce, so the sampled path is validated by this
project's own seed contract exactly as for qwen3-tts — same seed gives
byte-identical PCM, different seeds give `artifact_differs`, and the concrete
seed is reported — never by reproducing PyTorch's generator.

The validation-only family seam that replays a Gumbel noise stream is a
fallback, not the mechanism. It gets implemented only if a sampled-path parity
case proves it necessary.

**Measured 2026-07-31:** all 17 greedy cases reproduce the oracle's grid
exactly on CPU F32 — 16 against the primary grid and `omni-fast-mode` against
its committed alternate — and the replay-stage thresholds are committed in
`tests/tolerances/omnivoice.json` with reference stage
`source-f32-oracle-vs-f32-cpu`. `scripts/validate-omnivoice-replay.py --check`
enforces them as the CTest gate `synthesize-omnivoice-replay-golden`, which
registers whenever the package and the oracle payload are both present.

### The knife edge: dual-admissible grids and margin screening

Ruling by jiangzhuo, 2026-07-31, on the slice-5 result. Greedy decoding is
deterministic but it is not *unconditioned*: the loop's every decision is an
argmax or a top-k over F32 scores, and slice 4 measured this port's step-0
logits against the oracle's at **6.1e-04 worst max_abs**. When two candidates
sit closer together than that, which one wins is not a fact about either
implementation being right. It is below the resolution of the arithmetic, and
both sides are equally entitled to their answer. Two mechanisms handle that,
and neither is a tolerance.

**Dual admissibility.** A case may pin more than one grid the reference itself
produced, via `oracle.alternate_grids` in the Golden Manifest: a committed
file, its sha256, and free-text provenance. The port passes by equalling ANY
admissible grid **byte for byte**; `scripts/validate-omnivoice-replay.py`
verifies each witness against its recorded digest before comparing, and names
the grid that matched in the per-case line and the report JSON. The target set
is widened only by *enumeration* — one more reference-produced output,
committed with its provenance — never by loosening the comparison. There is
still no tolerance anywhere on this path and there will not be one: a tolerance
on a token id is meaningless.

Currently one case uses it. `omni-fast-mode` has two admissible grids: the
pinned oracle (`70ebe309…`) and `omni-fast-mode.alternate-grid-1.i32`
(`ce41f5b2…`), which is the dump this case carried before Task 2 pinned the
dumper's thread pool. Pinning moved exactly one of the 400 slots — flat index
358, codebook 7, frame 8 — from `1004` to `237`, and the port's free run
reproduces the pre-pinning grid on all 400. The two tokens' guided
log-probabilities differ by **6.87e-05**. So the port did not miss the oracle;
it landed on the side of a boundary that one torch configuration takes while
the oracle records the side another takes. The witness is committed rather than
described because it is **not regenerable** — the ambient thread configuration
that produced it was never pinned — which is also why it is a contract witness
(1.6 KB, immutable, digest-pinned) and not a golden payload under the
commit-contracts-not-payloads rule.

**Margin screening for new cases.** A candidate golden case is screened before
it is adopted, using `--margin-report` on the replay runner (surfaced per case
by the validator). The runner tracks the narrowest decision of the whole run:
the *selection* margin, the guided-score gap between the last candidate a step
committed and the best one it rejected; and on steps that commit everything
still masked — which reject nothing — the *argmax* margin between a committed
position's token and its runner-up. **A minimum margin below 1e-4 disqualifies
the case.**

The derivation is the F32 logit divergence between the two implementations,
6.1e-04 max_abs at step 0, which CFG at `guidance_scale = 2.0` amplifies rather
than damps: a decision made by less than that is not one this arithmetic
resolves, and margins near 1e-5 are coin flips outright. What fixes the
constant at 1e-4 rather than anywhere else is where the measured suite splits.
Over the 17 greedy cases the two smallest margins are **9.5e-06**
(`omni-rate-slow`) and **6.9e-05** (`omni-fast-mode`) — precisely the two cases
this ruling had to be written for — and the third smallest is **1.16e-04**
(`omni-short-en`), with every remaining case above it. A screen at 1e-4
separates the cases that needed a ruling from the cases that did not, on the
evidence rather than on a round number.

**The split is 1.7× wide**, not the 12× the first reading of it reported. That
reading compared 1.16e-04 against `omni-rate-slow`'s 9.5e-06 across what looked
like an empty band, because the table it was drawn from listed selection
margins only and `omni-fast-mode`'s **argmax** margin — 6.9e-05, sitting inside
that band — had not been measured yet. Once both kinds are measured the nearest
neighbours across the screen are 6.9e-05 and 1.16e-04. The screen still lands
where the suite splits, but it splits narrowly: a future case between those two
figures is a judgement call, not an obvious one.

Two scope limits, stated so the screen is not read as stronger than it is.
Argmax margins are collected on full-commit steps only — in practice the final
step, the one the schedule gives the whole remainder — so a narrow argmax on a
partially committing step is not screened. And the screen measures how narrow a
decision was, not whether it was decided correctly; it predicts fragility, and
only the exact-token comparison establishes parity.

**Watch note.** Both sub-threshold cases are retained, because screening
governs the selection of NEW cases and not the retirement of ones already
demonstrating parity. `omni-fast-mode` is covered by dual admissibility above.
`omni-rate-slow`, at **9.5e-06**, is not: it has one admissible grid and
currently lands the oracle's answer exactly, on a decision sixty-four times
narrower than the measured logit divergence. That is luck rather than
correctness, and it is recorded as such rather than presented as a
demonstration. If a legitimate
torch-side re-dump ever flips it, the remedy is the dual-admissibility
mechanism — enumerate the second grid with its provenance — and never a
threshold.

**And the screen at 1e-4 is not the noise bound.** Six of the 17 greedy cases
decide something by less than the 6.1e-04 logit divergence, and all six are
retained. Two are the sub-threshold pair the ruling was written for —
`omni-rate-slow` (9.5e-06) and `omni-fast-mode` (6.9e-05), named above. The
other four clear the screen and are adopted without a ruling, and they are
named here so the list is not mistaken for two: **`omni-short-en` (1.16e-04),
`omni-long-boundary` (2.05e-04), `omni-rate-fast` (2.44e-04) and
`omni-lang-none` (6.03e-04)**. Each of those four still turns on a decision
this arithmetic does not resolve; they pass today because the port lands on the
oracle's side of it. The screen governs which cases are *adopted*, and it is
not a claim that every adopted case was decided by a comfortable margin. When a
quantized profile or a second backend perturbs the scores, these four are where
a flip is likeliest — ahead of the eleven above the bound — and the remedy is
the same enumeration, never a threshold on a token id.

**`omni-clone-zh` supersession.** The case's target text changed from
`克隆的声音读出这句话。` to `克隆的声音也要说中文的句子。` on 2026-07-31 under this
ruling, and the case was re-dumped (76 frames → 98). This is a deliberate case
re-pick, **not** oracle drift: the old text put the case on the same knife edge
without the evidence that admits `omni-fast-mode`'s. Its port grid matched no
torch run — the ambient and pinned dumps were the same file
(`64d2b011…`) and the port produced a third (`e61a8711…`) — so its whole case
rested on a 4.05e-06 selection margin at step 1 and on the rejected candidate
carrying the oracle's token, which is inference rather than byte identity. On
the new text the minimum margin is **1.28e-03**, 12.8× the screen, and the port
reproduces the oracle grid exactly. The reference audio, `ref_text`, coverage
tags, case id and every oracle parameter are unchanged; the zh design and clone
texts were never required to match, so no manifest relation moves.

### The public sampled path carries no margin screen

Margin screening above governs which GREEDY cases are adopted into the Golden
suite, and nothing about the public sampled path (Plan 3's `class_temperature`/
`position_temperature` draws, or cloning) is screened the same way. This is
recorded in `tests/tolerances/omnivoice.json`'s own `public` stage: "sampled
seed-contract cases carry no margin screen (screen rationale is oracle-parity
knife edges; this stage claims none)."

The reason is what each phase claims. The screen exists because a greedy
Golden case claims **oracle agreement** — that this port's committed grid
equals a specific PyTorch run's — and at 6.1e-04 max_abs step-0 logit
divergence, some decisions are narrower than the arithmetic resolves; the
screen keeps a case that would win or lose that agreement by luck out of the
suite. `synthesize-omnivoice-public-request` (`docs/testing.md`) claims
something categorically different: relations between runs of **this port
alone** — a seed reproduces its own prior output, a different seed produces
`artifact_differs`, a Voice Profile changes the digest from a same-seed
profile-less run — never agreement with a PyTorch run, because **upstream
exposes no seed parameter of any kind** for this path (Reference Contract,
above) and so has no run of its own to agree or disagree with. A screen
calibrated against the oracle's own logit divergence has nothing to measure
on a path making no oracle-parity claim; applying one anyway would be
screening a comparison this phase never makes. The 13 public-request checks
(Task 6's base 7: echo, same-seed identity, cross-seed distinctness,
random-seed concreteness, language echo, a no-language arm, and evidence
completeness; Task 14's clone checks 8–10; Task 15's voice-design checks
11–13) are this path's own gate instead.

### Clone-encode parity: probes, suite revision 3, and the exact-token gate (Tasks 9–13)

Cloning's own encode chain — resample, HuBERT semantic branch, DAC acoustic
branch plus fusion, RVQ encode — needed evidence of its own before Task 14
could wire it into `Model::synthesize`, distinct from the greedy decode
loop's token-grid parity above. Ruling by jiangzhuo, 2026-08-01: three oracle
probes were added (`ref/pcm_16k.f32`, `ref/semantic_mean.f32`,
`ref/fused_latent.f32`, captured from the same clone-case dumps already
pinned) and the Golden suite moved to **`suite_version` 3** — no case text
changed and every existing grid and waveform is unchanged; only new probes
were added to cases the suite already carried.

Measured 2026-08-01 against both clone cases (`omni-clone-en`,
`omni-clone-zh`), which share one reference clip and so produced **identical**
figures on every channel below:

| Probe | max_abs | Committed max_abs | Cosine deviation | Committed min_cosine |
| --- | --- | --- | --- | --- |
| `ref.pcm_16k` (resampler output) | 1.19209e-07 (≈ one float32 ULP) | 6e-07 | 5.56e-08 | 0.999999 |
| `ref.semantic_mean` (HuBERT, pre-downsample) | 6.09234e-05 | 4e-04 | 7.07e-08 (cosine 0.99999993) | 0.999999 |
| `ref.fused_latent` (DAC + fusion) | 9.32217e-05 | 5e-04 | cosine 1.00000011, capped at 1.0 | 0.999999 |

`ref.pcm_16k`'s and `ref.semantic_mean`'s cosine deviations are each within
that array's OWN self-cosine noise floor (measured by comparing the oracle's
own probe against itself through the identical float32 estimator), and
`ref.fused_latent`'s raw cosine sits fractionally above 1.0 — the same
float32 summation artifact the deep generator probes already show, not a
defect. Every committed threshold is 5× the observed figure, consistent with
this file's rule for every other probe.

**The exact-token gate.** The RVQ encode's own claim is not a tolerance at
all: this family's `structural_exactness` requirement now covers TWO grids,
not one — the greedy decode loop's 8×T token grid (17/17 exact, above) and,
as of Task 13, the cloning path's own reference-encode grid. Both clone
cases' encoded tokens matched `ref/tokens.i32` **byte for byte on the first
run**: 8 codebooks × 351 frames = **2,808 tokens per case, exact, 2/2 cases**.
The RVQ encode's own margin instrumentation — never gated, diagnostic only,
and answering a different question than the decode loop's selection/argmax
margins above (this is a nearest-neighbor codebook lookup, not a sampled
commit) — measured a narrowest best-vs-second-best distance of
**0.00239563** over the real reference clip: comfortably wide of a knife
edge, and recorded here as this family's own figure per
`tests/tolerances/omnivoice.json`'s own note that this task is where it
lands.

## Text Frontend

OmniVoice uses the same Qwen2 byte-level BPE as qwen3-tts, over the same GGUF
vocabulary layout, with the same pre-tokenizer regex — verified at intake
against the transcription committed in `src/arch/qwen3-tts/bpe.cpp` rather than
assumed from the shared lineage. Re-verification against both `tokenizer.json`
files is owed when the qwen3-tts weights are next materialized; Task 8 records
the same debt. The Text Frontend Provider id therefore stays `synthesize.qwen_bpe`,
and the implementation is hoisted out of `src/arch/qwen3-tts/` into a shared
internal module with its unit tests intact, rather than copied. No BOS token is
added.

The text vocabulary is 151,676 entries — 151,643 base plus 33 added tokens —
and that count is the contract the converter writes and the loader checks, not
a number rederived from the tokenizer file at load time.

Seven OmniVoice-specific markers occupy ids 151669–151675: `<|denoise|>`, the
`<|lang_start|>`/`<|lang_end|>` pair, the `<|instruct_start|>`/`<|instruct_end|>`
pair, and the `<|text_start|>`/`<|text_end|>` pair. Thirteen bracketed
nonverbal tags — `[laughter]`, `[sigh]`, `[confirmation-en]`, `[question-en]`,
`[question-ah]`, `[question-oh]`, `[question-ei]`, `[question-yi]`,
`[surprise-ah]`, `[surprise-oh]`, `[surprise-wa]`, `[surprise-yo]`,
`[dissatisfaction-hnn]` — are tokenized standalone, split out of the
surrounding text before it is tokenized, so that a tag receives the same ids
whatever language surrounds it.

Upstream's 646-entry language-name map is **not** ported. The public interface
speaks BCP-47 tags, the prompt consumes an ISO language code as plain text, and
for every language in the validated catalog the tag and the code are the same
string, so the request's tag is written into the prompt directly. The tag `auto`
is the one that is not written through: `auto` and an omitted language are
synonyms, both selecting the language-agnostic path whose `<|lang_start|>` slot
carries the literal string `None`, which is why the Golden suite exercises the
`auto` tag and the omitted language as two cases over one oracle parameter.
Porting a 646-entry table to serve three validated languages would be carrying
an untested mapping as if it were a contract. This reverses the design record,
which specified the table as a generated data table in three places; the
reversal is recorded there as the "Amended 2026-07-30" note under §3 of
`docs/superpowers/specs/2026-07-30-omnivoice-family-design.md`, and this
contract is the operative statement.

## Duration and the Canvas Length

The canvas length is not a model output; it is fixed before the first forward
by upstream's `RuleDurationEstimator`, which this project ports as deterministic
host code. The estimator scores each character by script and sums, so it must
be reproduced exactly or every downstream token index shifts. The phonetic
weight table, verbatim from `omnivoice/utils/duration.py` at the pinned
revision:

```python
"cjk": 3.0,             # Chinese, Japanese Kanji, etc.
"hangul": 2.5,          # Korean Hangul
"kana": 2.2,            # Japanese Hiragana/Katakana
"ethiopic": 3.0,        # Amharic/Ge'ez
"yi": 3.0,              # Yi script
"indic": 1.8,           # Hindi, Bengali, Tamil, etc.
"thai_lao": 1.5,        # Thai, Lao
"khmer_myanmar": 1.8,   # Khmer, Myanmar
"arabic": 1.5,          # Arabic, Persian, Urdu
"hebrew": 1.5,          # Hebrew
"latin": 1.0,           # English, Spanish, French, Vietnamese, etc. (Baseline)
"cyrillic": 1.0,        # Russian, Ukrainian
"greek": 1.0,           # Greek
"armenian": 1.0,        # Armenian
"georgian": 1.0,        # Georgian
"punctuation": 0.5,     # Pause capability
"space": 0.2,           # Word boundary/Breath
"digit": 3.5,           # Numbers
"mark": 0.0,            # Diacritics/Accents (Silent modifiers)
"default": 1.0,         # Fallback for unknown scripts
```

A character is classified in this order: ASCII `A–Z`/`a–z` is `latin`; U+0020 is
`space`; U+0640 (Arabic Tatweel) is `mark`; then the Unicode general category
decides — `M*` is `mark`, `P*` or `S*` is `punctuation`, `Z*` is `space`, `N*`
is `digit`; then a `bisect_left` over the 88-entry `(end_codepoint, script)`
range table in the same source file selects the script weight; a codepoint
above U+20000 that fell through is `cjk`; anything else is `default`. The
88-entry range table is transcribed from the pinned source when the estimator
is implemented, not paraphrased. Because that table and these weights are
upstream Apache-2.0 *code*, adopting them records the upstream notice in
`THIRD_PARTY_NOTICES.md` at the time the port lands.

The estimate itself is a proportion, verbatim in effect from the same file:

```python
speed_factor = total_weight(ref_text) / ref_duration
estimated    = total_weight(target_text) / speed_factor
if estimated < low_threshold:               # low_threshold = 50
    estimated = low_threshold * (estimated / low_threshold) ** (1 / boost_strength)
                                            # boost_strength = 3
```

`ref_duration` is measured in audio tokens, so the estimate comes out in
frames. When no reference audio is present — auto-voice and voice-design
requests — upstream substitutes the anchor pair `("Nice to meet you.", 25
tokens)`, and that anchor is part of this family's contract, not an
implementation detail. `speaking_rate` **divides** the estimate, and the result
is `max(1, int(est))`, a truncation toward zero that the port must match
exactly rather than rounding. The package declares a validated
`speaking_rate_range` of `[0.5, 2.0]`; that range is a validation claim backed
by golden cases at both ends, not a clamp inherited from upstream. Explicit
target duration is not exposed in v1: the request's frame limit remains a cap,
not a target.

**Recorded divergence: zero reference frames.** Handed a reference of *zero*
audio tokens, upstream's estimator divides by zero and its `max(1, int(...))`
returns a one-frame canvas. This port treats zero reference frames as **no
reference** and takes the anchor pair instead. The divergence is deliberate and
unreachable from a valid request: a Reference Audio profile with no audio
behind it is refused at load, so zero frames can only mean a caller bug, and
answering that bug with a one-frame canvas would turn it into a synthesis that
technically succeeded. `tests/omnivoice_frontend_test.cpp` pins both halves of
the rule — every anchored row reaches the same frame count from a request
carrying no reference at all — and cites this paragraph.

## Delivery and Limits

The family delivers complete audio only. There is no Chunked Audio Delivery
claim and no Native Streaming Synthesis claim in v1: the model paints a whole
canvas at once, so there is nothing partial to hand back that would be honest
to call either.

**The loader refuses a package without embedded generation defaults** — the
converter's "second copy" doctrine is enforced at load, not merely at
conversion. `num_step`, `guidance_scale`, `t_shift`, the layer penalty factor
and both temperatures are read from the pinned upstream configuration by the
converter and written into the GGUF; a package missing any of them is rejected
with `omnivoice: package declares no generation defaults; re-cut it` rather
than run against a hardcoded fallback. A fallback is exactly the second copy
the doctrine forbids: it would let a package cut before a defaults change
synthesise silently under the new code's numbers.

`max_output_frames` is 720000 native PCM frames (docs/c-interface.md's
contract unit) -- 750 of this family's own codec frames at 25 Hz, 30 seconds
at the 24 kHz output rate. (Until 2026-08-03 the package's metadata field
wrongly carried the codec-frame count, 750, directly; a caller honoring the
documented PCM-frame unit read that as a 31-millisecond ceiling. PR #6's
review caught it; `reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`'s
2026-08-03 entry has the fix and the re-cut digest.) That ceiling is a
statement about what this port validates, not about what the model can do:
upstream's path beyond 30 seconds is long-form text chunking with cross-fade
stitching, and that is out of scope for v1. Output post-processing — silence
removal, fades, padding — is upstream product behavior and is not part of this
package's contract, which is why every oracle case disables it.

### One scaling survives the switches, and it is part of the contract

Intake measured what remains active inside `_post_process_audio` under the
contract's `postprocess_output=False, pad_duration=0.0, fade_duration=0.0`.
`remove_silence` is skipped, and `fade_and_pad_audio` is reached but inert — at
zero durations both its sample counts are zero and every branch inside is
skipped. **The volume branch between them is gated on nothing.**

| Condition | Effect | Applies to |
| --- | --- | --- |
| `ref_rms is not None and ref_rms < 0.1` | `audio * ref_rms / 0.1` | clone with a quiet reference |
| `ref_rms is None` | peak-normalise to exactly 0.5 (when peak > 1e-6) | auto-voice and voice design |
| `ref_rms is not None and ref_rms >= 0.1` | nothing; the chain falls through | clone with a loud reference |

Measured, not only read: every intake smoke run — greedy and sampled alike —
reports a peak of exactly 0.5. The clone branch is the exact inverse of a
forward operation applied when the reference is encoded,
`if 0 < ref_rms < 0.1: ref_wav = ref_wav * 0.1 / ref_rms`. The two gates are
not symmetric: a digitally silent reference (`ref_rms == 0.0`) is not boosted on
input but is multiplied by zero on output, which the port reproduces or
deliberately rejects rather than discovers later.

Plan 2's port mirrors all three branches. A port that reproduced the codec
perfectly and omitted this would be wrong by a per-utterance gain factor on
every request, and the golden tolerances could not catch it because the oracle
dumps carry the scaling too. Where the scaling *lives* was decided on
2026-07-31 and is recorded under Open Questions: inside the family's synthesis
path, with no public normalisation control in v1.

Plan 3's Task 14 is what makes the quiet arm (row 1 of the table above) real:
until then, `run_synthesis` applied the no-reference peak-normalize branch
unconditionally, because nothing upstream of it could supply a reference RMS.
`Model::synthesize` now threads the profile's own `ref_rms` through
(`SynthesisRequest::reference_rms`, a negative sentinel meaning "no reference
known" that keeps the replay seam — which has no ref_rms of its own — on the
legacy branch), and `codec-host.h`'s `apply_reference_volume` implements
exactly the `rms >= 0.1` / `0 < rms < 0.1` split.

### Silent-Reference Rejection (jiangzhuo's ruling, 2026-08-01)

The table's middle row is not symmetric with the third: a digitally silent
reference (`ref_rms == 0.0`) is not boosted going in
(`clip_and_boost_reference`'s own `0 < ref_rms < 0.1` guard excludes exactly
zero), but upstream's `_post_process_audio` still takes the quiet arm's
formula on the way out — `generated_audio * 0.0 / 0.1` — and hands back
silence as if the request had succeeded.

This port refuses instead. `voice-profile.cpp`'s `create_from_reference`
handler rejects a Reference Audio clip whose measured `ref_rms` is exactly
`0.0f` with `SYNTH_ERR_INVALID_ARG` and diagnostic code
`voice_profile.reference_silent`, before a `ClonePrompt` (and so a Voice
Profile) can ever exist to carry one. The ruling: a caller who supplied
digital silence and calls the result a cloned voice has almost certainly
handed the wrong file to the wrong argument, and returning silence dressed as
success hides that mistake rather than surfacing it. This is a **deliberate,
recorded divergence** from upstream's own behavior, not an omission --
`apply_reference_volume` (codec-host.h) still documents `ref_rms == 0.0f` as
"unreachable" for exactly this reason, and the invariant depends on every
ClonePrompt-producing path (today, only this one) enforcing the rejection
before construction.

Coverage: no golden case exercises this path (a committed golden's reference
clip is, definitionally, real audio), so `tests/omnivoice_profile_test.cpp`
drives it directly against the real package with a synthesized all-zero
buffer at the package's own minimum clip length.

### Quiet-arm coverage disposition (carryover item 7, closed)

The quiet volume arm (`0 < ref_rms < 0.1`) itself has the same coverage gap
the silent-reference rejection does, for a different reason: this family's
only pinned Reference Audio clip
(`models/omnivoice-reference-audio/seedtts_ref_en_1.wav`) measures
`ref_rms ≈ 0.1229`, comfortably above the 0.1 gate, and both committed clone
goldens (`omni-clone-en`, `omni-clone-zh`) share it. No golden this family
has, or is likely to add without deliberately sourcing a second, quieter
reference clip, ever takes this branch. `scripts/validate-omnivoice-replay.py`'s
`VOLUME_BY_BRANCH` carries a `scale_by_ref_rms_over_0.1` entry for a future
case that might, but the replay runner has no corresponding argument to act
on it — recording the mapping, not building a path nothing can reach yet.

Carryover item 7 (docs/superpowers/plans/2026-07-30-omnivoice-plan-2-carryover.md)
is closed by a unit fixture instead:
`tests/omnivoice_codec_test.cpp:check_reference_volume` exercises
`apply_reference_volume` directly against hand-built audio and RMS values
spanning both sides of the 0.1 boundary, which is the only place this family
can prove the formula correct without a real, quieter reference recording.

### No pydub silence preprocessing (upstream divergence)

Upstream's `create_voice_clone_prompt` optionally runs `preprocess_prompt` --
pydub-based silence removal and long-audio trimming -- before measuring
`ref_rms` or encoding a reference. Both committed clone goldens pin
`preprocess_prompt=False` (see the Reference Contract above), and this port
carries **no** trim/silence-removal stage at all: `Model::encode_reference`
and `voice-profile.cpp`'s dispatcher work on exactly the clip a caller
supplies, RFE-prechecked and normalized to the package's target format, never
edited. The public contract this implies: callers hand clean 1-20 s clips (the
package's own `min_frames_per_clip`/`max_frames_per_clip`, 1 s and 20 s at
24 kHz); a reference with substantial leading/trailing silence, or one long
enough that upstream would have chunked it, is handed to the model as-is
rather than cleaned up first. This is a scope decision recorded here rather
than a defect: adding upstream's silence/chunking behavior is future work if
a caller ever needs it, not part of the v1 Reference Audio Profile contract.

### Serialized Profile: one schema, two kinds (Task 16)

The v1 Serialized Voice Profile envelope (ADR 0008, docs/c-interface.md's "v1
Serialized Profile GGUF Contract") declares exactly one
`synthesize.voice_profile.schema` per family: `"omnivoice-clone-prompt"`, the
same string every already-shipped package's own `synthesize.profile.schema`
metadata already carries (weights.cpp's `read_profile_contract`). Rather than
adding a second schema id for Description Text ("voice design") profiles,
this family's envelope carries one extra metadata string,
`synthesize.voice_profile.kind`, set to `"clone-prompt"` or
`"design-instruct"`, and dispatches its payload shape (a
`profile.reference_tokens` tensor plus transcript/ref_rms/language_tag
metadata, versus a bare `instruct` string) from that.

The alternative -- a second schema string, e.g.
`"omnivoice-design-instruct"` -- would be equally truthful, but every
already-shipped Model Package declares its OWN schema as a fixed, validated
metadata value (`weights.cpp`'s `read_profile_contract` refuses anything
other than `"omnivoice-clone-prompt"`); introducing a second schema the
loader must also accept would mean either re-cutting every shipped package to
declare a package-level schema list, or teaching the loader that a package
declaring schema A may still emit envelopes claiming schema B -- both a
larger, riskier change than one more metadata string this loader already
reads unconditionally for every kind. `kind` costs nothing to add without a
package re-cut and keeps the package's own single declared schema truthful
for every profile it can produce.

### The untrusted-bytes lesson: a whitelist, not a blacklist (Task 16)

A Serialized Profile arrives at `synth_voice_profile_load_from_memory` as
caller-supplied bytes with no prior validation, and `ggml`'s own GGUF parser
is not written defensively against them: certain reserved metadata keys are
read eagerly, during construction, before this project's own loader ever
runs a single check — and a wrong-typed value there hits a `GGML_ASSERT`,
which aborts the whole host process rather than returning an error. `ggml/`
is a submodule and cannot carry a local patch (`ggml-patches/README.md`), so
the only place to stop this is a pre-scan of the raw bytes in
`arch/omnivoice/profile.cpp`'s `prescan_buffer`, run before
`gguf_init_from_buffer` ever touches them.

**Two crash instances, found by two different methods, in one review
session.** The first: a buffer declaring `general.alignment` with the wrong
type (anything other than a single `UINT32`) reaches
`gguf_get_val_u32`'s own `GGML_ASSERT(get_ne() == 1)`
(`ggml/src/gguf.cpp:1102`) and `gguf_kv::get_val`'s
`GGML_ASSERT(type_to_gguf_type<T>::value == type)` (`ggml/src/gguf.cpp:194`)
during `gguf_init_from_buffer`'s own alignment resolution
(`ggml/src/gguf.cpp:610`) — proven reachable through the full public path,
SIGABRT before fix, `SYNTH_ERR_INVALID_ARG` after. The first fix round's
remedy was a **blacklist**: special-case `general.alignment` (require
exactly `UINT32`, count 1) plus generous structural ceilings (at most 8
tensors, 64 metadata entries, 256-byte keys, 256-element arrays, 1 MiB
strings). The reviewer's own ASan fuzz harness, run unmodified against that
fix within the same review round, found a **second, different** crash: a
40-byte buffer with a zero-length key sails through the ceiling checks
untouched (zero is not greater than the length ceiling) and an empty key is
never equal to `"general.alignment"`, so it reaches a DIFFERENT
`GGML_ASSERT(!key.empty())` inside all four of `gguf_kv`'s value-shape
constructors (`ggml/src/gguf.cpp:143,151,161,167`) — the same abort class,
a different trigger the blacklist's author had no way to have enumerated in
advance.

**The lesson, and why the remedy changed shape rather than grew a third
special case.** Enumerating another library's internal asserts is
necessarily a blacklist, and a blacklist of another library's invariants can
never be proven complete — it can only ever answer "is this the specific bad
thing I already know about", never "is this SOME bad thing". Fix round 2
replaced the blacklist with **positive validation against exactly the
format this project's own writer ever produces**: `kPrescanKnownKeys`, a
closed table of the 12 keys `set_common_metadata` /
`serialize_clone_prompt` / `serialize_design_instruct` can ever write, each
with its exact declared type, array-ness and count; unknown keys, duplicate
keys, an `n_kv` other than exactly 9 (DesignInstruct) or 11 (ClonePrompt) --
not a ceiling -- and a tensor section other than 0 tensors or exactly one
`profile.reference_tokens` tensor of this writer's own exact shape are all
rejected, independent of whether `gguf_init_from_buffer` would also abort on
the same bytes and independent of which specific asserts `ggml` happens to
carry today. This subsumes the whole class the blacklist could only chase
one instance at a time.

**Standing evidence.** The reviewer's own fuzz harness, unmodified, re-run
against the whitelist fix across three campaigns (single-byte mutation of a
valid envelope, random small buffers, and boundary/extreme
key-count/tensor-count/string-length/array-count/empty-key cases) at
**8,127 iterations total**: zero crashes, zero sanitizer reports, zero
hangs, reproduced twice. That number, not a proof of completeness a
whitelist cannot offer either, is the standing evidence this hardening
rests on, and it is what the next reviewer should re-run rather than trust
by citation.

## Quantization Profile Shape

F32 is the reference package, and this family has a measured reason to expect
that quantization is harder here than in any family shipped so far. A reference
port measured token agreement collapsing from 100% to roughly 7% with F16
generator weights: an argmax flip at one step is committed, fed back into both
branches, and conditions every later step, so error does not average out the
way it does in a feed-forward decoder. Every produced Quantization Profile must
re-pass the full suite including the exact-token gates; a profile that fails is
not shipped, and no perceptual claim substitutes for the gate. RVQ codebooks,
the `fc`/`fc2` projections, norms, and one-dimensional tensors stay at the
reference dtype — the consensus of the two reusable ports and of this project's
own qwen3-tts and Kokoro policy. Profiles are measured for this family, not
inherited.

### Q8_MIXED (Plan 4 Task 3): codec-only, and BLOCKED on the exact-token gate

By jiangzhuo's ruling of 2026-08-06, this family's Quantization Profile is
codec-only: every generator tensor (`llm.*`, `audio_embeddings.weight`,
`audio_heads.weight`) and the RVQ (`codec.quantizer.*`, `codec.fc`,
`codec.fc2`) stay at the reference dtype regardless of shape, for the reason
this section's opening paragraph already states. `src/arch/omnivoice/
quantization.h`'s `QuantRole` is the single classifier the offline quantizer
(`tools/synthesize-quantize`) and the runtime catalog (`catalog.cpp`) both
read, so an offline packing decision and a load-time expectation cannot
drift. The profile is named `Q8_MIXED`, not a family-specific name: the
quantizer's profile table (`tools/synthesize-quantize/policy.cpp:14-24`) is
shared across every family, and each family's own `resolve_<family>_target_spec`
function is what makes "Q8_MIXED" mean something different per family —
exactly as it already does for VITS, Kokoro and Qwen3-TTS. There is no
family-specific name to invent.

**Package.** Produced from the committed `omnivoice-0-6b-F32.gguf`
(sha256 `f6d504ff…f9fa3`) with `synthesize-quantize INPUT OUTPUT --quant
Q8_MIXED`: `omnivoice-0-6b-Q8_MIXED.gguf`, sha256
`b020933f…4b671e`, 2,703,016,576 bytes (2577.8 MiB) against the source's
3,189,953,504 bytes (3042.2 MiB) — a 15.3% reduction. 158 of 798 tensors
quantize to Q8_0; the tensor-data byte total (excluding GGUF header/padding
overhead) moves from 3,184,565,636 to 2,697,629,938 bytes, group by group:

| Group | F32 bytes | Q8_MIXED bytes | Tensors | Q8_0'd |
| --- | ---: | ---: | ---: | --- |
| generator (`llm.*`, audio tables) | 2,450,309,120 | 2,450,309,120 | 312 | none (Sensitive) |
| `codec.quantizer` + `fc`/`fc2` | 11,574,272 | 11,574,272 | 44 | none (Sensitive) |
| `codec.acoustic_decoder` | 80,952,452 | 50,943,986 | 110 | most matrix weights |
| `codec.acoustic_encoder` | 205,257,984 | 54,617,344 | 110 | most matrix weights |
| `codec.semantic_model` (HuBERT) | 377,483,264 | 114,511,872 | 209 | most Linears/convs |
| `codec.encoder_semantic` | 58,988,544 | 15,673,344 | 13 | most convs |

**A load-path bug found and fixed, orthogonal to the gate result below.**
`build_semantic_branch`'s own cross-check that a `feat_conv[index]` tensor's
`ne[0]` agrees with the package's declared `conv_kernel[index]`
(`src/arch/omnivoice/reference-encoder.cpp`) compared the tensor's raw
`ne[0]` against the bare kernel width unconditionally. That is only
`feat_conv[index]`'s real row when the tensor is unpacked; once Q8_MIXED
packs `feat_conv[1..6]` (every feat_conv but index 0, which reads the raw
single-channel waveform and stays Sensitive), `ne[0]` becomes
`kernel * in_channels` — 1536 for `feat_conv[1]` on this checkpoint
(kernel 3 × HuBERT's 512-wide `conv_dim[0]`), never 3 — so the unfixed check
refused the packed package outright: `Model::encode_reference` returned
`SYNTH_ERR_INTERNAL` for every request, silently, with exit code 0. Task 2's
own packed-conv work (`conv1d` in this same file) already carried the correct
packed dispatch for the *convolution itself*; this ONE cross-check, a
redundant defense against a mismatched package, was the one spot it missed —
caught only because a first pass at this task's own replay run reused the
F32 run's `--work` directory and returned suspiciously bit-identical
`ref.semantic_mean`/`ref.fused_latent` figures under a profile that should
have perturbed them, which is what prompted checking the runner's own stderr
directly. Fixed by deriving the expected shape from whether the tensor is
`ggml_is_quantized`, exactly `catalog.cpp`'s own packed-shape acceptance
logic; regression-tested by `omnivoice_reference_encoder_test.cpp`'s new
`check_packed_feat_conv1`, which builds `feat_conv[1]` both ways from
identical raw weights (mirroring `check_packed_encoder_semantic_conv`'s own
pattern for `encoder_semantic.conv`, the tensor that pattern already covered
and which is why this one slipped through) and separately asserts a
wrong-shaped packed tensor is still refused. **This fix is independent of
the profile decision below and should ship regardless**: a package that
legitimately packs `feat_conv[1..6]` could not load its reference-encode path
at all before it.

**THE GATE. FAILED.** Re-run with the fix in place,
`scripts/validate-omnivoice-replay.py --require all --margin-report --profile
Q8_MIXED --backend CPU --stage replay --model
omnivoice-0-6b-Q8_MIXED.gguf`, fresh `--work` directory:

```
token grids exact: 17/17
ref.tokens exact: 0/2
narrowest RVQ encode gap: 0.0102997
```

The 17 greedy grids are exact — expected, and for a structural reason rather
than luck: the entire generator and the RVQ stay F32/Sensitive under this
profile, so the decode loop's logits are bit-for-bit identical to the F32
package's, and every one of that run's own margins (below) reproduces the
F32 baseline to the last measured digit. **Both clone cases' RVQ encode
grids are NOT exact**: `omni-clone-en` and `omni-clone-zh` (which share one
reference clip and so produce identical figures) each mismatch at **1023 of
2808 positions (36.4%)**. This is not a knife-edge margin call the way the
greedy screen's sub-threshold pair are — the mismatch gaps at the diverging
positions run from 0.53 to 27.29 (RVQ nearest-neighbor distance units),
against an F32 baseline whose narrowest best-vs-second-best gap over the
whole clip was 0.00239563. First 20 of 1023 differing positions (both cases
identical; codebook and frame index the same layout `ref/tokens.i32` uses):

| codebook | frame | got | want | gap |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 3 | 554 | 26 | 1.5105 |
| 0 | 7 | 841 | 554 | 2.8479 |
| 0 | 9 | 423 | 26 | 0.7493 |
| 0 | 23 | 719 | 364 | 1.7428 |
| 0 | 38 | 532 | 708 | 4.1221 |
| 0 | 54 | 554 | 835 | 3.6883 |
| 0 | 55 | 207 | 26 | 1.1656 |
| 0 | 57 | 555 | 846 | 17.7489 |
| 0 | 58 | 653 | 515 | 2.4443 |
| 0 | 59 | 87 | 619 | 0.6864 |
| 0 | 60 | 730 | 923 | 27.2859 |
| 0 | 62 | 923 | 555 | 11.7191 |
| 0 | 67 | 923 | 423 | 10.9085 |
| 0 | 70 | 569 | 459 | 7.2939 |
| 0 | 83 | 950 | 675 | 0.5335 |
| 0 | 87 | 389 | 644 | 19.3942 |
| 0 | 91 | 97 | 243 | 0.6600 |
| 0 | 102 | 389 | 872 | 4.1656 |
| 0 | 117 | 11 | 916 | 5.2944 |
| 0 | 125 | 423 | 26 | 0.9558 |

The generator probes reproduce the F32 baseline exactly (structural, not
approximate); the codec-side probes that go through the quantized
`semantic_model`/`acoustic_encoder` move by orders of magnitude —
`ref.semantic_mean` max_abs 6.09e-05 → 0.342867, `ref.fused_latent` max_abs
9.32e-05 → 3.73227, `audio.pcm` min_cosine 0.99999986 → 0.99775438 — all in
the same direction, which is why this reads as accumulated quantization noise
through roughly 150 quantized HuBERT/DAC-encoder weights rather than a second
isolated bug: every probe touching the quantized encoder path degrades
together, by a similar order of magnitude, not just the one that happens to
flip a token.

**Margin protocol (carry-over item 8), old vs new.** Every greedy case's
margin is IDENTICAL between profiles, to the measured digit — not merely
close — because the entire generator is Sensitive/F32 under Q8_MIXED and the
margin is a property of the generator's own logits alone, never the codec's:

| case | F32 margin (baseline) | Q8_MIXED margin | kind |
| --- | ---: | ---: | --- |
| `omni-rate-slow` | 9.53674e-06 | 9.53674e-06 | selection |
| `omni-fast-mode` | 6.86646e-05 | 6.86646e-05 | argmax |
| `omni-short-en` | 1.15871e-04 | 1.15871e-04 | selection |
| `omni-long-boundary` | 2.04682e-04 | 2.04682e-04 | selection |
| `omni-rate-fast` | 2.44433e-04 | 2.44433e-04 | selection |
| `omni-lang-none` | 6.03199e-04 | 6.03199e-04 | selection |
| `omni-design-zh` | 6.40869e-04 | 6.40869e-04 | argmax |
| `omni-medium-en` | 7.17163e-04 | 7.17163e-04 | argmax |
| `omni-punctuation` | 8.39233e-04 | 8.39233e-04 | argmax |
| `omni-upstream-readme` | 1.14441e-03 | 1.14441e-03 | argmax |
| `omni-clone-en` | 1.19019e-03 | 1.19019e-03 | selection |
| `omni-clone-zh` | 1.28174e-03 | 1.28174e-03 | selection |
| `omni-digits` | 1.39546e-03 | 1.39546e-03 | argmax |
| `omni-design-en` | 1.41111e-03 | 1.41111e-03 | selection |
| `omni-nonverbal` | 1.89209e-03 | 1.89209e-03 | argmax |
| `omni-short-zh` | 2.46429e-03 | 2.46429e-03 | selection |
| `omni-short-ja` | 2.66457e-03 | 2.66457e-03 | selection |

None of the four in-band cases predicted as likely first flips
(`omni-short-en`, `omni-long-boundary`, `omni-rate-fast`, `omni-lang-none`)
or `omni-rate-slow` actually flipped — the prediction assumed the failure
mode would be a generator-side logit perturbation, and this profile cannot
produce one by construction. The real failure is in a subsystem the greedy
margin screen was never built to probe: the clone path's RVQ encode margin
(diagnostic only, never gated) was **0.0102997** narrowest over the whole
Q8_MIXED run — WIDER than the F32 baseline's 0.00239563 — yet 36.4% of
positions still flip, because the fused latent feeding the nearest-neighbor
search moved by orders of magnitude, not because any single decision was
narrow. The dual-admissibility mechanism (enumerate a second oracle-produced
grid with provenance) is built for a knife-edge single-decision flip; it has
no natural reading for 1023 arbitrary positions with no alternate oracle run
that would plausibly produce this exact grid.

**Status: BLOCKED, not shipped.** Per this section's own opening rule, a
profile that fails the exact-token gate is not shipped, and no perceptual
claim substitutes for it. No tolerance cell for Q8_MIXED is committed in
`tests/tolerances/omnivoice.json`: the exact-token check is unconditional and
independent of `--check`, so a tolerance cell would not make the grid pass
and would misrepresent a blocked profile as measured-and-ready. jiangzhuo's
ruling on what to try next is the following subsection.

### F16 (Plan 4 Task 3 continuation): also codec-only, also BLOCKED

jiangzhuo's ruling: measure F16 codec-only before concluding anything about
quantization for this family — a genuinely different measurement, not a
retry of Q8_MIXED. The tool's profile table gives F16
`TensorLayout::Native` (`tools/synthesize-quantize/policy.cpp:14-24`): it
does not pack, and `ggml_compute_forward_im2col` accepts an F16 destination
natively, so F16 needs none of Task 2's packed-branch machinery on the
convolution path — every MatrixWeight tensor keeps its native shape and only
halves its type, running through the existing (unmodified) builders. F16's
per-weight relative error is roughly an order of magnitude below Q8_0's
(≈2⁻¹⁰ against Q8_0's ≈1/127), which is the whole question for a
nearest-neighbour decision that Q8_MIXED missed by a distance gap of 1.51.
**Q8_MIXED itself stays blocked and unshipped; it was not revisited.**

**Package.** `omnivoice-0-6b-F16.gguf` from the same committed F32 source,
sha256 `530b2b85…0b955fa`, 2,858,422,240 bytes (2726.0 MiB) against the
source's 3,189,953,504 bytes — a 10.4% reduction, close to but not exactly
the ≈316 MiB naive estimate (158 tensors halving is 316,166,912 bytes of
tensor data by construction — F32 bytes minus half of F32 bytes for exactly
the same 158 tensors Q8_MIXED touched — the file-size delta differs slightly
because GGUF per-tensor alignment padding scales with tensor count and
boundary position, not with the bytes saved per tensor). Confirmed by
dumping tensor types rather than assumed: **exactly the same 158 of 798
tensors** change type as under Q8_MIXED (the classifier is shared), each
F32 → F16 in place, native shape unchanged — Native layout does not read the
classifier's roles any differently than PackedMatrix did, only what it does
with a MatrixWeight tensor once classified. Tensor-data bytes: 3,184,565,636
→ 2,853,034,948 (−331,530,688 bytes, −316.2 MiB — matching the naive
estimate almost exactly at the tensor-data level, confirming the file-size
gap above is padding-only).

**The load path.** `QuantizationProfile` gained an `F16` enumerator
alongside `F32`/`Q8Mixed` (`weights.h`, `weights.cpp`'s `read_quantization`,
`model.cpp`'s `get_info`); `catalog.cpp`'s `expected_type()` gained the
matching case (`MatrixWeight` → `GGML_TYPE_F16`, else F32) — the same
`classify_tensor` dispatch Q8Mixed already uses, so no second classifier
exists to drift from the first. **No third gap.** F16 never enters the
packed-shape branch in `catalog.cpp`'s `find()` (gated on `Q8Mixed`
specifically) or the packed arm of `reference-encoder.cpp`'s `feat_conv`
check (gated on `ggml_is_quantized`, which F16 is not) — a direct
single-case run of `Model::encode_reference` against the F16 package
succeeded on the first try, with none of Q8_MIXED's silent-failure history.
Regression coverage: `omnivoice_catalog_test.cpp`'s `check_f16_resolution`/
`check_f16_rejections` (a full synthetic package under `F16`, no
block-size-driven width changes needed since nothing packs) and
`omnivoice_metadata_test.cpp`'s `run_quantization_profile_acceptance`
(extended to cover `"F16"` alongside `"Q8_MIXED"`).

**THE GATE. FAILED, but far more narrowly than Q8_MIXED.** Same command,
`--profile F16`, fresh `--work` directory:

```
token grids exact: 17/17
ref.tokens exact: 0/2
narrowest RVQ encode gap: 0.00306702
```

17/17 greedy grids exact, for the identical structural reason as Q8_MIXED
(generator and RVQ are Sensitive/F32 under every profile). **Both clone
cases' RVQ encode grids are still NOT exact** — `omni-clone-en` and
`omni-clone-zh` each mismatch at **103 of 2808 positions (3.7%)**, roughly a
tenth of Q8_MIXED's 1023, consistent with F16's roughly-10x-smaller
per-weight error. First 20 of 103 (both cases identical):

| codebook | frame | got | want | gap |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 57 | 555 | 846 | 0.2024 |
| 1 | 25 | 442 | 278 | 0.4197 |
| 1 | 128 | 77 | 503 | 0.2124 |
| 1 | 155 | 658 | 991 | 0.1527 |
| 2 | 4 | 518 | 597 | 0.6510 |
| 2 | 57 | 613 | 597 | 4.8593 |
| 2 | 128 | 11 | 52 | 1.0621 |
| 2 | 155 | 39 | 130 | 27.4781 |
| 2 | 264 | 618 | 109 | 0.9983 |
| 2 | 348 | 228 | 508 | 0.6170 |
| 3 | 6 | 485 | 626 | 1.7481 |
| 3 | 7 | 202 | 975 | 0.0298 |
| 3 | 25 | 821 | 670 | 24.8352 |
| 3 | 27 | 706 | 395 | 0.3370 |
| 3 | 49 | 414 | 165 | 0.2210 |
| 3 | 54 | 101 | 596 | 0.0313 |
| 3 | 57 | 113 | 884 | 14.2502 |
| 3 | 102 | 559 | 685 | 0.1507 |
| 3 | 105 | 305 | 930 | 0.1803 |
| 3 | 128 | 926 | 91 | 2.4834 |

Unlike Q8_MIXED's mismatches, this set is a genuine MIX: several gaps
(0.0298, 0.0313, 0.1507, 0.1527, 0.1803, 0.2024, 0.2124, 0.2210) sit in the
same range as a legitimately narrow greedy-decode margin, consistent with
ordinary F16 rounding tipping a close nearest-neighbour call — but others
(27.4781, 24.8352, 14.2502, 4.8593) are not narrow by any reading, the same
signature of real accumulated error Q8_MIXED showed, just smaller. This is
not a single mechanism to explain away with one story; it is 103 positions
of varying character, and the gate does not average them.

**`audio.pcm` cosine, the second thing this measurement was for.** F32
baseline 0.99999986; **F16 0.99999743**; Q8_MIXED 0.99775438. F16's decode
side is close to two more nines than Q8_MIXED's — a real difference, and by
itself arguably an acceptable production number — but it does not rescue the
clone path, which is where both profiles actually fail. `ref.semantic_mean`
max_abs 6.09e-05 (F32) → 0.00795197 (F16) → 0.342867 (Q8_MIXED);
`ref.fused_latent` max_abs 9.32e-05 (F32) → 0.129286 (F16) → 3.73227
(Q8_MIXED) — F16 sits consistently between the other two, roughly an order
of magnitude closer to F32 than Q8_MIXED on every channel, exactly the
predicted relationship, and still not close enough to pass the encode grid.

**Margins.** Every greedy-case margin reproduces the F32 baseline exactly
under F16 too, for the identical reason as Q8_MIXED (the generator never
changes) — the same table in the Q8_MIXED section above applies unchanged
and is not repeated here.

**Status: BLOCKED, not shipped. This family ships F32-only.** Two codec-only
Quantization Profiles were measured against the exact-token gate and both
failed the cloning path's RVQ encode grid — not by the same mechanism or
magnitude, but neither is a knife-edge margin call eligible for dual
admissibility. No tolerance cell for F16 is committed, for the same reason as
Q8_MIXED. No per-profile golden gate is registered for either profile: a
permanently-red registered test is not an acceptable end state for the
branch, and the measured evidence for both is recorded here and in
`reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md` instead. If a
future task wants to revisit quantization for this family, narrowing scope
to exclude `codec.semantic_model`/`codec.acoustic_encoder` (which feed only
the clone-only encode path) while still quantizing `codec.acoustic_decoder`
(which the greedy/public synthesis path alone exercises) is the untried
option both measurements point toward — untried here because re-scoping the
profile to make a grid pass is exactly what this task's own gate discipline
prohibits doing unilaterally.

## GGML Operator Surface

**The incremental operator surface is zero.** Three independent ports and this
project's own VITS and Kokoro experience agree, and there is no path to a local
ggml change anyway: `ggml/` is a submodule and a submodule cannot carry one.

The Snake activation the codec needs is the alpha-only form,
`x + sin²(αx)/α`, which `snake()` in `src/arch/kokoro/operations.cpp` already
computes from ordinary primitives — this family needs no SnakeBeta parameter.
Convolution is `ggml_im2col` plus matmul. Transposed convolution is mul_mat
plus `ggml_col2im_1d`, the recipe VITS was moved onto on 2026-07-27, so it is
exercised rather than merely present. Argmax, confidence ranking, the layer
penalty and the top-k commit are host CPU work: they are discrete decisions,
and the discrete-outputs placement rule keeps them and their input path off the
accelerator.

## Execution Backends (stage 7)

Measured 2026-08-07 on the DGX Spark/GB10 development host (native `sm_121a`,
CUDA 13.3.73, the standard `dev-dgx-spark` preset -- GGML CUDA unified-memory
fallback stays off, per `docs/backends.md`'s CUDA Unified Memory Policy; the
research-only `dev-dgx-spark-uvm` preset was not used and would not qualify this
claim if it had been). The evidence and the exact commands are in
`reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`; this section
summarizes the outcome the family card and `family_supports_explicit_backend`
(`src/model-info.h`) now rely on.

### The codec moves, the generator does not

This resolves the "Generator on CUDA" Open Question below, and confirms the
qwen3-tts precedent this family always intended to follow: the generator's
own RVQ token selection is upstream of its own next step (every frame's
codes feed back into the canvas the next denoising step reads), so
`docs/backends.md`'s discrete-outputs rule holds the *generator* -- the whole
mask-predict denoising loop, all `num_step` iterations -- and its entire input
path on the CPU, unconditionally, exactly as it did before this task. Only the
codec's decode graph (the acoustic decoder, the RVQ dequantizer, and the final
projection -- Task 9's 152-tensor, 84.24 MiB "movable" partition) moves.

Swept all twenty golden cases with `--accelerate`
(`scripts/validate-omnivoice-replay.py --accelerate --profile F32 --backend
CUDA --stage replay`):

- **Every codec node left the CPU and every generator node did not, across the
  whole suite, not just on average.** Aggregated over all twenty cases: codec
  8,440 of 8,440 nodes off the CPU; generator 0 of 880,032. Docs/backends.md
  Validation Gate 5 (device placement) and the discrete-outputs rule are both
  satisfied by the same count.
- **All seventeen greedy cases' token grids stayed byte-exact against the CPU
  baseline (17/17), and both cloning cases' RVQ-encode token grids too (2/2,
  8×351 = 2,808 tokens each).** This is the gate the brief called
  non-negotiable: a flip here would mean the codec twin or its placement was
  wrong, not that sampling drifted, because the generator that draws the codes
  never left the CPU. None flipped.
- The waveform is the one artifact that does move, because it is what the
  accelerated codec's TF32 arithmetic actually touches: `audio.pcm`/
  `audio.pcm_freerun` worst cosine 0.9999963545175866 (deviation ≈3.65e-6,
  against CPU-only's ≈1.7e-7) and worst max_abs 0.007008261978626251 on a
  0.5-peak signal -- committed to `tests/tolerances/omnivoice.json`'s new
  `backends.CUDA.stages.replay` cell at min_cosine 0.999981 / max_abs 0.04
  (five times the measured deviation, this family's own established rounding
  convention). Every deep generator probe and every reference-encode probe
  (`ref.pcm_16k`/`ref.semantic_mean`/`ref.fused_latent`) stayed within the
  same noise floor the CPU-only file already documents (≈1e-7), because
  nothing upstream of the codec moved.
- On the suite's longest case (`omni-long-boundary`, 719 frames, same case
  `docs/testing.md`'s CPU golden-gate timeout budget is measured against): the
  codec itself is 9.68× faster (4.3069 s → 0.4451 s), but because the held
  generator is 98.9% of wall time on this case, the end-to-end effect is a
  3.4% reduction (267.30 s → 258.48 s; real-time factor 9.294 → 8.987). This
  is the opposite shape from Kokoro and VITS's own CUDA rows in
  `docs/backends.md`'s per-family cost table: there, holding a *minority*
  stage off an otherwise-GPU-primary graph is a tax; here, moving a free
  *minority* stage off an otherwise-CPU-primary graph is a small, bounded
  saving. `docs/backends.md`'s own per-family table carries the row and the
  reasoning for the inversion.

### Operational evidence (`docs/backends.md`'s Validation Gate 5)

Latency and real-time factor, measured through the public seam
(`tests/omnivoice_public_real.c`, now carrying the `[cpu|cuda]` backend
positional qwen3-tts's driver always had -- Plan 4's accumulated requirement 2;
before this task the flag was accepted by
`scripts/validate-omnivoice-public.py` and never reached the runner, which
would have made a CUDA sweep a false green): one seed-0 request of
"Sampling follows the seed." (45,120 PCM frames) loads in 2.0015 s / 2.0356 s
and synthesizes in 15.2393 s / 15.3927 s, CPU vs. CUDA -- the codec's share of
this particular request is too small to separate from run-to-run noise, which
is consistent with the longer case's own 1.6 percent codec share above.

**Peak memory is not separately measurable on this hardware the way it is on
discrete hardware, and reporting a bare number would claim more than it means
(Plan 4's accumulated requirement 4).** qwen3-tts's own CUDA evidence read
memory the same way discrete hardware is read -- an `nvidia-smi` resident
figure, treated as a second budget on top of host RAM. That recipe does not
transfer to DGX Spark/GB10: CPU and GPU physically share one DRAM pool here,
and this is not an inference, it is what this family's own public Interface
reports when asked. `synth_model_get_device()` on this host returns the CUDA
device with `SYNTH_DEVICE_MEMORY_SHARED` set and `memory_total` =
130,594,721,792 bytes -- bit-identical to what the CPU device entry reports as
system memory total, sourced from the CUDA runtime rather than from
`nvidia-smi` (`docs/backends.md`'s CUDA Unified Memory Policy already commits
the project to that source; `nvidia-smi -q -d MEMORY`'s own aggregate query
independently confirms the reason, returning "N/A" for this device's
Total/Reserved/Used/Free -- the plain `nvidia-smi` summary table separately
shows "Not Supported" in its Memory-Usage column, a different field of the
same underlying absence). A shared-memory flag on the standard, non-UVM
`dev-dgx-spark` preset is a hardware-topology fact, not a sign that GGML's UVM
fallback is enabled -- it is not, here.

So there is no second budget to report a peak against. What is real and
measured instead: `/usr/bin/time -v` on `omni-medium-en` (307 frames, same
binary, same host) reports Maximum resident set size 4,071,432 kB on CPU and
4,071,436 kB with `--accelerate` -- a 4 kB difference, i.e. no measurable host
RSS growth from moving the codec to CUDA, because host RSS accounting does not
count the device-mapped allocation at all. That allocation is real and
bounded, just invisible to RSS: `nvidia-smi --query-compute-apps` (which
returns real per-process numbers on this box even though the aggregate query
does not) shows this process holding 254 MiB right after load -- the CUDA
context plus the 84.24 MiB mirrored decode-path weights -- rising to a
transient 1,167 MiB while the codec's own compute buffers are live on the
suite's largest case, then falling back once that decode finishes.

**That 1,167 MiB is not a dedicated-VRAM requirement and must not be read as
one.** On discrete hardware a number like it would mean "this workload needs
1.2 GB of GPU memory the rest of the system cannot use," because discrete VRAM
is a separate physical pool. Here it is a transient share of the *same*
128 GB DRAM pool `SYNTH_DEVICE_MEMORY_SHARED` already names as identical to
system RAM (the 130,594,721,792-byte figure two paragraphs up) -- host RSS
already dominates this process's real footprint at ~3.9 GB, and this figure
rises and falls within that same one budget rather than adding a second,
GPU-exclusive one on top of it. Total physical commitment at any instant is
host RSS plus this per-process figure, both drawn from the identical pool;
there is no sense in which the GPU figure is "on top of" host RAM the way a
discrete card's VRAM would be, and no hardware on this machine is reserved
for it that a CPU-only run could not otherwise use.

### Repeated-run and resource cleanup (Validation Gate 6)

`tests/public_cleanup_test.cpp`'s CUDA arm is generic across families
(Task 8's fix round 1 made it assert the family's actual claim either way), so
it went live for OmniVoice the moment `family_supports_explicit_backend`
flipped -- no test code changed for this family. Twelve cycles each, a short
"Hi." request: post-free resident floor 367,268 → 367,268 KB on CPU (+0 KB)
and 563,632 → 563,632 KB on CUDA (+0 KB). No leak on either backend.

### The positive assertion (accumulated requirement 1)

The cleanup test above tolerates a forgotten backend-claim flip by design --
a correct refusal and a silently-wrong claim are observationally identical to
it. Two independent checks close that gap for this task specifically, both
querying `synth_model_get_device()` directly rather than trusting `SYNTH_OK`:
`tests/omnivoice_backend_test.cpp`'s `check_explicit_cuda` (a synthetic
package, unit-level, asserts `device.kind == "cuda"` after an explicit CUDA
load) and `scripts/validate-omnivoice-public.py`'s new check 14 (the real
0.6B package, integration-level, same assertion after a real
`synth_synthesize_to_buffer` call). Both passed.

### The claim

`family_supports_explicit_backend(ModelFamily::Omnivoice, SYNTH_BACKEND_CUDA)`
returns `true` as of this task, after all of the above rather than before it
(accumulated requirement 3): the replay runner calls `Model::load` directly and
bypasses the public seam Task 8's refusal lives behind, so the sweep measured
real placement on real hardware while the seam still said no. Every
`docs/backends.md` Validation Gate this family/backend combination owes is
now satisfied: same cases as the CPU baseline (1), tensor agreement within
committed tolerances (2), finite correctly-shaped PCM (3), twenty cases (4),
proven placement with latency/RTF/memory (5), and clean repeated-run cycles
(6). Publication and the support matrix are separate, later questions.

## Adapters (stage 7.5)

The CLI (`examples/cli/`) and the Python wheel are adapters over
`include/synthesize.h`, per this project's own rule that no synthesis
capability may exist only in an adapter (`CLAUDE.md`). Task 7 registered
both for this family: `synthesize-omnivoice-cli` (package-default voice, no
`--voice` flag — the package's catalog carries an unnamed default and no
preset ids; text input drives `--text`, and the unsupported-input arm is
pointed at `--phonemes`, the one kind this package's `input_flags` lacks)
and the Python wheel's family smoke (24 kHz, 960 samples per frame, seed 7/8
reproducibility, no preset Voice needed). **qwen3-tts never registered
either harness** — a stage-7.5 omission the design for this family named
explicitly and closed rather than repeated. Fixing the omission surfaced one
real cross-family bug rather than an omnivoice-specific one:
`synth_add_cleanup_test`'s unquoted `${ARGN}` silently dropped an empty
middle argument, which for omnivoice's empty-voice/named-language call shape
shifted the language string into the voice slot; the fix (quoted,
positional `foreach` over `ARGN`) was proven behaviorally identical for
VITS, Kokoro and qwen3-tts before being applied project-wide, not scoped to
this family alone.

## Decomposition

This family is large enough that it does not fit one implementation plan, and
it decomposes along the slice sequence in the design record rather than by
module. Plan 1 (foundations, slices 0–3) covers intake and ADR 0018, the locked
reference environment, the golden manifest and tolerance skeleton, the oracle
dumpers, the converter, and the C++ load path through metadata, tensor catalog
and text frontend — ending at a loadable `omnivoice` GGUF whose metadata,
tensors, and frontend are unit-tested. Plan 2 (slices 4–6) is the bidirectional
backbone graph, the greedy diffusion loop to exact token-grid parity, and codec
decode to an end-to-end greedy waveform. Plan 3 (slices 7–8) is the public
sampling path and cloning: internal 24→16 kHz resampler, HuBERT semantic
branch, acoustic encoder, RVQ encode, and the Reference Audio and Description
Text profiles with their Serialized Profile round-trip. Plan 4 (slices 9–13) is
the Port Validation Suite end-to-end with committed tolerances, Quantization
Profiles, Execution Backends, Adapter registration, and ship.

Plans 2 through 4 are written only after Plan 1's oracle artifacts exist,
because their test assertions consume oracle-measured values. Ship waits for
cloning: the package does not publish without the model's headline capability.

Two reuse decisions are settled in advance. The byte-level BPE is hoisted to a
shared internal module and qwen3-tts points at it. The Qwen3 block builder is
**not** shared: qwen3-tts's block is causal with a KV cache while this one is
bidirectional, cache-free, full-canvas and CFG-paired. That is
duplicate-with-adaptation, revisited only if a third Qwen-family port appears.

## Intake Measurements

Measured on 2026-07-30 against the pins in the Reference Contract above; the
evidence and the commands are in
`reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`.

### Greedy determinism holds across processes

Two separate Python processes, both temperatures zero, CPU F32, text
`"OmniVoice speaks with one voice."` in `en`, auto-voice: 48,000 samples each —
2.000 s at 24 kHz, 50 codec frames — **byte-identical** under `cmp(1)`, sha256
`2710bbd2…` for both, `max_abs_diff` exactly 0.0. Wall 22.6 s and 23.2 s, so a
CPU real-time factor of 11.3 to 11.6 on the 20-core aarch64 GB10 host with the
GPU unused. Upstream's README claims RTF as low as 0.025 on GPU; the two
figures measure different things and neither predicts this project's GGML speed.

The identity is stronger than qwen3-tts's, which came from switching sampling
off inside a seeded process. Here both temperatures at zero mean the decode loop
takes neither Gumbel branch, so it makes no RNG call at all — which is why the
result survives a process boundary.

Unseeded sampling at the package defaults differs at the first byte,
`max_abs_diff` 0.849, setting the stochastic capability. But **a duration
comparison would call this family deterministic when it is not**: all four runs
produced exactly 48,000 samples, because the canvas length is fixed by the
`RuleDurationEstimator` before the first forward. qwen3-tts's sampled runs
differed in length only because its length is an autoregressive stop decision.
Determinism checks for this family compare content.

### The checkpoint is F32, with one exception that is not a weight

| File | Tensors | Parameters | F32 | I64 |
| --- | --- | --- | --- | --- |
| `model.safetensors` | 313 | 612,577,288 | 312 | 1 |
| `audio_tokenizer/model.safetensors` | 527 | 201,400,553 | 527 | 0 |

Generator prefixes: `llm.` 310 (28 layers × 11, plus `embed_tokens` and `norm`),
`audio_embeddings` 1, `audio_heads` 1, and one unprefixed tensor. Codec
prefixes: `semantic_model.` 210, `acoustic_encoder.` 110, `acoustic_decoder.`
110, `quantizer.` 64, `decoder_semantic.` 14, `encoder_semantic.` 13, and six
`fc`/`fc1`/`fc2` tensors.

The single I64 tensor is `codebook_layer_offsets`, shape `[8]`, values
`[0, 1025, 2050, 3075, 4100, 5125, 6150, 7175]` — verified equal to
`arange(8) * 1025` by `torch.equal`. It is a derivable buffer and the converter
derives it. There is no `lm_head` tensor, which is `tie_word_embeddings: true`
visible in the file rather than only in the configuration.

Three intake findings change what the converter must do:

- **The RVQ codebooks are live tables, not EMA accumulators.** Higgs Audio V2
  decodes with `F.embedding(embed_ind, self.embed)`, so unlike qwen3-tts no
  reconstruction is needed; `embed_avg`, `cluster_size` and `inited` are
  training state, 24 tensors the converter drops.
- **`audio_tokenizer/config.json` disagrees with its weights three times** —
  `n_codebooks` 9 against 8 shipped, `acoustic_model_config.codebook_dim` 8
  against the measured 64, and a `sampling_rate` of 16000 that belongs to the
  semantic branch. Size from the tensors.
- **195 of the 527 codec names reach 58 characters** once a `codec.` prefix is
  added, against a `GGML_MAX_NAME` of 64, and the longest —
  `codec.semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original1`
  — overruns it by 18. The generator side is clear at 45 characters.

### The text frontend matches qwen3-tts, and the vocabulary counts agree

151,643 base entries plus 33 added tokens is 151,676, equal to
`llm_config.vocab_size`; 151,387 merges. The pre-tokenizer regex is
character-for-character the pattern already implemented in
`src/arch/qwen3-tts/bpe.cpp` — the comparison the Text Frontend section above
describes, and the weaker of the two available: the qwen3-tts weights are not
present on this host, so the two `tokenizer.json` files were never diffed
directly. The seven TTS markers occupy 151669–151675 in the documented order,
all `special: true`, and `tokenizer_config.json` lists exactly those seven under
`extra_special_tokens`. No BOS on either side.

### Upstream ships no pinnable reference audio

Neither the source repository at the pinned revision, nor the Hugging Face
Space, nor a revision of the authors' demo page offers a reference-audio
artifact addressable at a pinned revision; the README's own clone example passes
a `ref.wav` placeholder that upstream never ships. Clone cases therefore pin
their reference **by content digest** — which fails loudly on a byte change
rather than silently substituting. The candidates and the trade between them are
recorded in `intake.json` under `source.upstream_examples`; Task 5 owns the
manifest choice.

## Prior Art and Attribution

Four existing implementations were read at design time. None of their code is
in this project; anything later adopted as code from the two permissively
licensed ones records its notice in `THIRD_PARTY_NOTICES.md` at the time of
adoption.

| Repo | License | Use |
| --- | --- | --- |
| `ServeurpersoCom/omnivoice.cpp` | MIT | Best architecture document of the four, published GGUFs usable as cross-check oracles, and the col2im recipe. Its README misstates the upstream licenses. |
| `bluryar/omnivoice.cpp` | Apache-2.0 | Clearest MaskGIT-style decode loop in ggml, single-GGUF precedent, a ggml CUDA bug audit. Misstates the weight license and ships no converter. |
| `rockerritesh/omnivoice-tts.cpp` | **PolyForm Noncommercial** | **Read-only — no code reuse.** Source of the temperatures-zero bit-exactness finding and the F16 collapse measurement. |
| `0xShug0/audio.cpp` | unverified — treat as read-only | Active multi-model GGML audio engine that supports OmniVoice; a positioning mirror rather than a source. |

The license column is about each port's own code. It says nothing about the
weights, which every one of these repositories describes incorrectly or not at
all.

## Divergences Ledger (v1)

A consolidated list of this port's deliberate v1 divergences from upstream,
gathered here from where each was decided so ship-time review does not have
to re-derive them from the sections above:

1. **No pydub silence preprocessing.** Upstream's optional
   `preprocess_prompt` (silence trimming, long-audio chunking) is not
   ported; every clone case pins `preprocess_prompt=False` and this port
   carries no trim stage at all. See "No pydub silence preprocessing" above.
2. **No fade or pad on public output.** This is stronger than "the oracle
   comparison pinned zero durations": v1's public request surface (`struct
   synth_synthesis_params_t` and this family's `SynthesisRequest`) exposes
   no fade-duration or pad-duration control of any kind, and no code path in
   `Model::synthesize` applies one. Upstream's `fade_and_pad_audio` is
   product-layer behavior this port never implements, not a switch this
   port implements and leaves off.
3. **HuBERT LayerDrop draws but never skips — not reproduced.** Upstream's
   `HubertEncoder.forward` draws `torch.rand([])` every layer
   UNCONDITIONALLY, even in eval mode, but only acts on the draw when
   `self.training` (`skip_the_layer = self.training and dropout_probability
   < layerdrop`); eval mode never skips regardless of what the draw says.
   This port (`src/arch/omnivoice/reference-encoder.h`) draws nothing and
   skips nothing — a divergence of no consequence, since eval-mode
   upstream's own behavior is "never skip" too, and the semantic-branch
   parity figures above (max_abs 6.09234e-05, cosine 0.99999993) are
   measured against that same eval-mode upstream.
4. **Description-language is never sniffed from text.** A null
   `description_language` resolves to the fixed default `"en"`, never
   derived from the description's own bytes, per `docs/c-interface.md:568`'s
   explicit prohibition ("The implementation never detects the description
   language from its text"). See Open Questions, "Voice-design instruct
   passthrough", below.
5. **Instruct is unified at profile-creation time, using the resolved
   `description_language`, not upstream's per-call target-text baseline.**
   Upstream computes its unify-to-one-language decision from the TARGET TEXT
   being synthesized (`omnivoice.py:1068`), which does not exist yet at
   Voice Profile creation time; this port substitutes the request's own
   resolved `description_language` instead. See Open Questions, same
   paragraph, for the full rationale and the three places this is recorded.

## Listening Audit (Plan 4 Task 16)

**Verdict, jiangzhuo, 2026-08-07: `no_obvious_regression`, all six pairs.**
Reordered ahead of Tasks 13–15 in Slice D because a `regression` verdict is a
ship-blocker and every audio-producing slice (A/B/C) was already complete; the
carry-over ledger's open item ("The Listening Audit is still owed before
ship") is closed by this entry.

Six A/B pairs, `docs/model-porting.md:244-271`'s cap. None of the doc's three
named quality scores (worst intelligibility, worst UTMOSv2, worst
voice-similarity) exist for this family — **ADR 0017 defers that whole grid**
— so the selection substituted the per-case `audio.pcm` cosine this family's
own replay validator already reports: worst and second-worst cosine stand in
for intelligibility/UTMOSv2 (1 slot each), voice-similarity expanded to one
clone case plus one Description Text case rather than picking one
conditioning path arbitrarily (2 slots), longest-duration stood as specified
but had its comparison kind spent on backend coverage instead of a second
port-vs-oracle pair (1 slot, see below), and the two random slots reduced to
one to hold the total at six (1 slot). Five of the six pairs compare the
port's replayed codec output against the pinned PyTorch oracle; the sixth
compares the codec's CUDA decode against its CPU decode of the identical
committed token grid. Order and A/B side were independently randomized
(`order_seed 20260807`, `case_selection_seed 16`); 5 of 6 pairs were swapped
so neither the port nor the CUDA side was positionally guessable.

| Pair | Case | Comparison | Cosine |
| --- | --- | --- | ---: |
| 1 | `omni-medium-en` | port vs oracle (worst cosine) | 0.9999998558 |
| 2 | `omni-nonverbal` | port vs oracle (2nd-worst cosine) | 0.9999998990 |
| 3 | `omni-long-boundary` | **CUDA vs CPU** codec | 0.9999983311 |
| 4 | `omni-clone-en` | port vs oracle (Reference Audio) | 0.9999999031 |
| 5 | `omni-design-en` | port vs oracle (Description Text) | 0.9999999511 |
| 6 | `omni-punctuation` | port vs oracle (random pick) | 1.0000002084 |

**Pair 3 is the CUDA-vs-CPU comparison, and its inaudibility corroborates the
backend claim rather than merely accompanying it.** Among the six audited
pairs its max_abs (3.06e-03, ~450x pair 1's) is the largest, but that is a
property of comparison kind, not of the case: five pairs compare a port
against an oracle (1e-5 to 1e-6 range) and only this one compares two
backends. `omni-long-boundary` was picked for the audit's longest-duration
slot, the same case Task 11 already named as the codec's largest measured
CUDA *speedup* (9.68x at 719 frames) — not, as an earlier draft of this
section claimed, the case with the family's largest measured CUDA numeric
divergence: recomputing CPU-vs-CUDA `audio.pcm` cosine directly from the
replay artifacts for all twenty cases ranks `omni-long-boundary` 5th of 20 by
that measure (`reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`'s
2026-08-07 Task 16 entry has the full ranking). Task 11's byte-exact token
grids already established that nothing upstream of the codec's decode moves
between backends; a tolerance grid cannot say whether the codec's own
float32 arithmetic difference between backends is large enough to hear, and
pair 3 answers exactly that, on the case the longest-duration slot already
pointed to. The answer is no — which extends the Execution Backends
section's structural claim with the one kind of evidence a token-grid
comparison cannot provide, rather than standing beside it as an unrelated
data point.

**What this does not establish.** Per `CONTEXT.md`'s definition, a Listening
Audit "records obvious regressions without claiming population-level
subjective quality." This was one listener, six pairs, informally — no rated
comparison, no panel, no score. It says that on the six automatically
selected pairs above, at that hearing, no obvious problem was noticed. It
does not claim general perceptual equivalence between the port and the
oracle, or between the CUDA and CPU codecs, and it does not move
`quality_evaluation` off `not_run`: ADR 0017's automated grid has not run and
is not scheduled. Full method, the seeds, the swap pattern, and the values
Task 14's card needs are in
`reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`'s 2026-08-07 Task
16 entry.

## Open Questions

~~**Generator on CUDA.** The codec moves first, following the qwen3-tts
codec-on-CUDA precedent. Placing the generator on CUDA is claimed only if
placement evidence proves the committed token grids bit-identical to CPU; one
port measured CUDA-F32 token-exact and Metal-F32 at 83%, which is encouraging
and not evidence. Stage 7 decides.~~ **Answered 2026-08-07 by Stage 7 (see the
Execution Backends section above): the generator does not move, and is not
claimed to.** The generator's own RVQ token selection feeds its next step,
so `docs/backends.md`'s discrete-outputs rule holds the generator and its
whole input path on the CPU regardless of backend, the same way it always has
-- this was never a live candidate for the CUDA-F32-token-exact test the
question describes, because moving it would put a discrete decision's own
input path on the accelerator. Only the codec (152 tensors, downstream of
every sampled token rather than upstream of one) moved, and the twenty-case
sweep held all seventeen greedy grids and both cloning grids byte-exact
against the CPU baseline.

**The empirical test the question originally asked for was still run
(Plan 4 Task 12, 2026-08-07), separately from the policy answer above.**
Policy holding the generator on the CPU is not the same claim as measuring
what breaks if it does not, and this family doc's own standing rule --
"claimed only if placement evidence proves the committed token grids
bit-identical to CPU" -- is about the measurement. A one-line, hand-reverted
patch (`generator_branch_forward`'s `GraphRun::run(...)` call, `on_primary`
flipped to `true`; never registered behind any build flag or option, never
shipped, full method in the porting log) forced the generator's own graph
through the primary-backend scheduler, and the twenty-case sweep re-ran
against it: **3 of 17 greedy grids stayed byte-exact; 14 flipped, several
almost totally (`omni-digits` 98.30%, `omni-short-en` 94.00%). Aggregate
token agreement across all 17 cases: 45.34% (8,723 of 15,960 positions
differ).** This is markedly worse than the "Metal-F32 at 83%" reference
point above, which was already characterized as encouraging rather than
evidentiary.

The flip pattern does **not** match this doc's own margin-table prediction.
Three of the five cases the margin table names as likeliest to flip first do
flip (`omni-short-en`, `omni-long-boundary`, `omni-rate-slow`), one flips by
a single token (`omni-rate-fast`), and one does not flip at all
(`omni-lang-none`) -- but ten cases with comfortably safe CPU-vs-oracle
margins flip too, several worse than any of the predicted five
(`omni-digits`, margin 1.40e-03, 98.30% flipped). The mechanism why is in
the same sweep's own probe table: step-0 logits already diverge from the CPU
baseline by 0.10 max_abs -- about 90x the 6.1e-04 the margin screen is
calibrated against -- and TF32 error compounding through the generator's 28
transformer layers, 32 denoising steps and two CFG branches per step grows
that to 14.9 max_abs by the final layer, several orders of magnitude past
both the screen and the widest margin any of the 17 cases actually carries.
At that scale nearly every decision in the suite is exposed, not only the
already-narrow ones, which is what distinguishes this result from a
knife-edge story: it is a magnitude problem from depth times step count, not
a handful of coin-flip decisions. Full per-case numbers, per-position detail
for the four smallest flips, and the exact reproduction method are in
`reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`'s Task 12 entry.

This changes nothing about the shipped claim -- the discrete-outputs rule
already held this stage on the CPU by construction, independent of what this
measurement found -- but it closes the question the family doc originally
posed with a real number instead of an argument, so a future cycle does not
re-run the same experiment expecting a more favorable one.

**Quantized profiles against the argmax cascade.** ~~Whether any profile below
F32 survives the exact-token gates is an open measurement, not an
expectation.~~ **Answered 2026-08-06: no profile below F32 survives, and this
family ships F32-only.** Q8_MIXED and F16 were both produced and measured; both
keep the greedy grids exact (17/17 — the generator stays F32 and greedy decode
never reads the codec's encoder half) and both break the clone path's RVQ
grids, at 1023 and 103 of 2808 positions respectively. The measurement
narrative and its full tables are in this document's quantization section and
the porting log. The structural reason is recorded there too: 93.8% of the
quantizable weight is the clone-encode path feeding a discrete
nearest-neighbour decision, so the tensors worth quantizing are exactly the
ones that cannot be.

**The Gumbel replay seam.** Implemented only if a sampled-path parity case
proves it necessary; the seed contract covers the public sampled path
otherwise.

**Where the residual output scaling lives — decided 2026-07-31 (Plan 2).**
Inside the family's synthesis path: `Model::run_synthesis` applies the
no-reference peak-normalise-to-0.5 branch after codec decode, faithful to the
oracle, and no public normalisation control exists in v1. The replay seam's
`decode_codes` returns the raw waveform so validation can apply the oracle's
branch per case. The quiet-reference and zero-reference branches are decided
with cloning (Plan 3).

**Voice-design instruct passthrough — decided 2026-08-02 (Plan 3, Task 15).**
v1 DOES reimplement `_resolve_instruct`: `src/arch/omnivoice/profile.cpp`'s
`resolve_instruct()` transcribes the closed attribute/accent/dialect
vocabulary from `omnivoice/utils/voice_design.py:31-97` and the validation/
unification rules from `omnivoice/models/omnivoice.py:1492-1621`, both at the
pinned revision. `synth_voice_profile_create_from_description` (Task 14
staged the capability bit; Task 15 implements the dispatch) runs a raw
description through it before a Voice Profile is ever built, so a caller's
description is validated and canonicalized the same way upstream's own
`generate()` call validates an `instruct` string -- not passed through
unchanged. Two deliberate divergences, forced by the public Interface's own
shape rather than chosen for convenience:
  * upstream computes its unify-to-one-language baseline from the TARGET
    TEXT being synthesized (`omnivoice.py:1068`), which does not exist yet
    at Voice Profile creation time; this port substitutes the request's
    resolved `description_language` instead (English unless the caller
    names Chinese explicitly -- never detected from the description's own
    text, per docs/c-interface.md's explicit prohibition on that).
  * the closed-vocabulary rejection for an unsupported item is reported
    without upstream's `difflib` "did you mean" suggestion -- a diagnostic
    naming the bad item, not a search engine.
