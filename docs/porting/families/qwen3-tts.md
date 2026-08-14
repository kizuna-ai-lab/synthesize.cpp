# Qwen3-TTS Family Selection and Port Plan

Status: Confirmed 2026-08-14. Stage 1 (`qwen3-tts-12hz-0.6b-customvoice`) is
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
`reports/porting/qwen3-tts/qwen3-tts-12hz-0-6b-customvoice/`.

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
neither ICL, which this port does not implement, nor CUDA for the new graphs,
which have only ever run on CPU. Quality Evaluation stays `not_run` per
ADR 0017 and the Validation Level does not move. See "Listening Audits" below.

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

**What each stage advertises, stated once (corrected 2026-08-12).** An earlier
revision of this section said Serialized Profile stays unadvertised at every
stage, while the Stage 2 Plan 1 record below claimed the Base package
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
- **Stage 3 adds `SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT`** on the same terms.

So Serialized Profile is not a source this family pursues on its own; it
arrives as a consequence of being able to prepare one, and it is unadvertised
until then.

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

**One pathology, two symptoms -- the Task 11 hypothesis is now demonstrated,
not assumed.** Task 11's review found a real 8 s clip with a deliberately wrong
transcript running to the ceiling, and hypothesised it and this collapse were
one ICL length pathology. They are: the seed scatter above produces *both*
symptoms from a *single* input configuration, four collapses and one runaway,
differing only in seed. An incoherent reference makes the stopping decision
unreliable in both directions.

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
target text `"Hi."` and seed 7 on CPU against the BF16 Base package:

| reference transcript | outcome |
| --- | --- |
| the clip's own (`"Okay. Yeah. I resent you. …"`) | `SYNTH_OK`, 13 codec frames (24,960 PCM), peak 0.681, 5.5 s |
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

- **ICL / transcript-assisted cloning.** Plan 3's. `reference_transcript` and
  `reference_language` report `SYNTH_REQUIREMENT_UNSUPPORTED`, and a request
  carrying a transcript is refused by name
  (`voice_profile.transcript_unsupported`). The codec encoder graph is not
  written; `resolve_codec_encoder` still discards its pointers into a scratch
  struct, deliberately. Ten of the Base manifest's twelve Golden cases are
  ICL and cannot be driven end to end by anything in this plan.
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
  from a transcript-assisted dump. The anomaly is real and is Plan 3's, to be
  adjudicated when the ICL path exists.
- One listener's judgement that a clone is recognisably the same speaker as
  its source. This is the first such evidence in this repository.

**What it does not establish.** Any quality, naturalness or comparative-ranking
claim — ADR 0017 defers Quality Evaluation, and a Listening Audit does not move
the Validation Level. The audit *is* a comparison, and a bounded one: this port
against its own reference implementation, four cases, one listener. What stays
out of reach is ranking this model against any other, which is what
`comparative` means everywhere else in this project's documents. Any speaker-similarity metric; the resemblance finding is one
listener, one source clip, one clone, and is evidence rather than a property of
the port. Anything about transcript-assisted (ICL) mode, which this port does
not implement. Anything about CUDA for the new graphs, which have only ever run
on CPU — the port side of every pair here was CPU.

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
