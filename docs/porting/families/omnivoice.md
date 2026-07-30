# OmniVoice Family Selection and Port Plan

Status: Confirmed 2026-07-30. Intake in progress; stages 2+ not started.

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
Face weights of the same name. Both revisions, the checkpoint file list, its
byte counts, and per-file SHA-256 digests are pinned at intake and recorded in
`reports/porting/omnivoice/omnivoice-0-6b/intake.json`; this document does not
restate them.

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
executable code in a Model Package is not engaged.

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

### The prompt layout

```
[<|denoise|>?] <|lang_start|>{code-or-None}<|lang_end|> <|instruct_start|>{instruct-or-None}<|instruct_end|> <|text_start|>{ref_text + " " + text}<|text_end|> [ref audio tokens] [T × mask(1024)]
```

Absent fields are the literal four-character string `None`, not an omission —
a request that names no language emits `<|lang_start|>None<|lang_end|>`. The
`<|denoise|>` marker appears only in clone mode, that is, only when reference
audio tokens are present.

The whole prompt is a grid 8 rows deep. **Every row of the 8-codebook
dimension repeats the same text ids**; the rows differ only where audio tokens
live. The embedding merge is correspondingly asymmetric: the sequence embedding
is the text embedding of row 0 plus the sum of the eight codebook embeddings,
each read at its own offset into the shared audio embedding table
(`arange(8) * 1025`). The target region is `T` frames of mask id 1024 in all
eight rows, where `T` is the canvas length fixed by the duration estimator.

### The decode loop

Each step runs one batched forward holding a conditional and an unconditional
sequence. The unconditional branch carries the target region only — no style
markers, no text, no reference audio — so the two branches have different
lengths and the batch is padded accordingly.

The per-step commit budget is a schedule computed once from the canvas size.
Timesteps are `num_step + 1` points linearly spaced on `[0, 1]` and then
shifted by

```
t' = t_shift·t / (1 + (t_shift − 1)·t)
```

with `t_shift = 0.1` by default, which concentrates the early steps at low
signal-to-noise. Step `s` commits `ceil(total_mask * Δt)` positions, where
`total_mask = T × 8` and `Δt` is that step's shifted interval, clamped to what
remains unmasked; the final step commits the entire remainder, so the canvas is
always fully committed after `num_step` steps regardless of rounding.

Within a step the two logit sets are combined in log-softmax space:

```
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

## Text Frontend

OmniVoice uses the same Qwen2 byte-level BPE as qwen3-tts, over the same GGUF
vocabulary layout, with the same pre-tokenizer regex — verified as identical
text in the two `tokenizer.json` files rather than assumed from the shared
lineage. The Text Frontend Provider id therefore stays `synthesize.qwen_bpe`,
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
string, so the request's tag is written into the prompt directly. Porting a
646-entry table to serve three validated languages would be carrying an
untested mapping as if it were a contract. This reverses the design record,
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

```
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

## Delivery and Limits

The family delivers complete audio only. There is no Chunked Audio Delivery
claim and no Native Streaming Synthesis claim in v1: the model paints a whole
canvas at once, so there is nothing partial to hand back that would be honest
to call either.

`max_output_frames` is 750, which is 30 seconds at 25 Hz. That ceiling is a
statement about what this port validates, not about what the model can do:
upstream's path beyond 30 seconds is long-form text chunking with cross-fade
stitching, and that is out of scope for v1. Output post-processing — silence
removal, fades, padding — is upstream product behavior and is not part of this
package's contract, which is why every oracle case disables it.

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

**Voice-design instruct passthrough.** v1 does not reimplement upstream's
`_resolve_instruct` normalization. Oracle cases pin already-normalized instruct
strings, so the port passes Description Text through unchanged and the
normalization stays an upstream concern until something forces it inward.
