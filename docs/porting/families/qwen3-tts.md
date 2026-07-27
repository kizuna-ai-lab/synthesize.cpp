# Qwen3-TTS Family Selection and Port Plan

Status: Selection accepted on 2026-07-26. Conversion, C++ implementation, and
port validation are not started; intake is in progress, Task 1 of five complete.
Two revisions on 2026-07-27: the operator surface dropped from two expected
additions to none, and a source read of `qwentts.cpp` is recorded under
"Findings From Reading qwentts.cpp" — three conversion rules that fail silently
and a tolerance shape that would misreport a correct port.

## Decision

The third Model Family is **Qwen3-TTS 12 Hz** (Qwen team, Alibaba). OmniVoice
(Xiaomi / k2-fsa) is recorded as the leading fourth-family candidate, deferred
for a licensing reason recorded below rather than a technical one.

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

Stage 1 is a completion gate for Stage 2, and Stage 2 for Stage 3. Serialized
Profile and Random Seed sources stay unadvertised across all three stages.

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
`docs/port-validation.md` Phase 1 requires the oracle on CPU, so this project
compares CPU PyTorch against CPU GGML — a strictly better condition. Their
figures should not be carried into this family's expectations; measure ours.

**Not transferable — max-absolute tolerances on deep talker layers.** In one of
their runs the per-layer max-abs is 13.07 / 13.09 / 13.42 at L7 / L14 / L21 and
**66.8** at L27, while cosine similarity stays at or above 0.9998. Those are
outlier channels ahead of the final norm. `tests/tolerances/qwen3-tts.json` must
use a cosine threshold for these stages; a max-abs threshold would report
catastrophic failure on a correct port.

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

## Open Questions for Intake

1. Confirm the codec decoder topology against upstream rather than against a
   port: SEANet stage ratios, ConvNeXt upsample factor, DAC strides, and whether
   SnakeBeta exponentials can be folded at load.
2. Confirm the RoPE section collapse is exactly equivalent for a
   text-plus-codec timeline.
3. Measure real CPU speed for the 0.6B Stage 1 variant on project hardware. The
   Code Predictor's per-frame sequential steps are reported to dominate
   generation time; confirm the cost and the documented cache-reset mitigation
   before committing to a CPU claim.
4. Enumerate the CustomVoice preset speakers and their dialect overrides for the
   Preset Voice Catalog.
5. Determine whether the causal codec decoder qualifies as Native Streaming
   Synthesis under `CONTEXT.md`, or whether Stage 1 claims only Chunked Audio
   Delivery.
6. Establish upstream provenance and redistribution permission for publishing
   converted Model Packages, as was done for Kokoro. Alibaba does not disclose
   training corpora; the Apache-2.0 grant is the basis relied on, and it is the
   same basis on which Kokoro was accepted.

## Accepted Risks

- No arena or listening evidence supports a quality claim for this family, and
  the project has no capability to produce one. Delivery is `port_validated`.
- The 1.7B Stage 3 variant may not meet a CPU practicality bar that the 0.6B
  stages meet. Stage 3 may ship CPU-slow or not at all without invalidating
  Stages 1 and 2.
- Mature MIT prior art exists for this exact model. The project's value here is
  integration under one validated ABI, not novelty.
