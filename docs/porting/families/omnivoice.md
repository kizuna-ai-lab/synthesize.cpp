# OmniVoice Family Selection and Port Plan

Status: Confirmed 2026-07-31. Intake through the greedy synthesis core
(slices 4–6) done: single-forward parity, exact token grids 17/17, replay
waveform under committed tolerances (tests/tolerances/omnivoice.json). Public
sampling, cloning, and the full validation suite have not started.

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

`max_output_frames` is 750, which is 30 seconds at 25 Hz. That ceiling is a
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

## Open Questions

**Generator on CUDA.** The codec moves first, following the qwen3-tts
codec-on-CUDA precedent. Placing the generator on CUDA is claimed only if
placement evidence proves the committed token grids bit-identical to CPU; one
port measured CUDA-F32 token-exact and Metal-F32 at 83%, which is encouraging
and not evidence. Stage 7 decides.

**Quantized profiles against the argmax cascade.** Whether any profile below
F32 survives the exact-token gates is an open measurement, not an expectation.
Stage 6 decides, and a failing profile is simply not shipped.

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

**Voice-design instruct passthrough.** v1 does not reimplement upstream's
`_resolve_instruct` normalization. Oracle cases pin already-normalized instruct
strings, so the port passes Description Text through unchanged and the
normalization stays an upstream concern until something forces it inward.
