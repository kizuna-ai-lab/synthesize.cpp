# Qwen3-TTS Family Selection and Port Plan

Status: Confirmed 2026-07-28. Intake, the oracle and conversion are complete;
the C++ implementation is under way -- the shared decoder block, the code
predictor, the tensor catalog, the talker graph and the codec decoder are built
and tested; the text frontend and the public seam are not. Port validation is not started. Selection was accepted on
2026-07-26; the intake packet is
`reports/porting/qwen3-tts/qwen3-tts-12hz-0-6b-customvoice/`.

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
on CPU in F32 with `attn_implementation="eager"`. **It takes two independent
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

### Determinism: what "byte-exact sampling" there does not mean here

`src/philox.h` implements Philox4x32-10 and the header says it keeps the
multinomial sampler "byte for byte aligned with the upstream Python pipeline".
Read carefully before drawing the obvious conclusion.

Two things make it inapplicable as-is:

- Philox matches PyTorch's **CUDA** generator (cuRAND). PyTorch's **CPU**
  generator is MT19937. `docs/port-validation.md` Phase 1 puts this project's
  oracle on CPU, so their stream would not match ours even in principle.
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
eager attention because `docs/port-validation.md` Phase 1 requires a CPU oracle,
so it is a floor set by this project's validation rule, not a measurement of the
model as its authors run it.

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

So the reproducible CPU oracle Phase 1 requires does exist, and the
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
   is a floor set by this project's CPU-oracle rule, not a measurement of the
   model as upstream deploys it (CUDA, bfloat16, FlashAttention 2). Whether CUDA
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
6. Establish upstream provenance and redistribution permission for publishing
   converted Model Packages, as was done for Kokoro. **Basis settled
   2026-07-27**, decision deliberately not taken. Both the source at `022e286b`
   and the checkpoint at `85e237c1` carry an explicit Apache-2.0 grant, audited
   at the pinned revision rather than at `main`, with no restriction prose in
   either card. Alibaba does not disclose training corpora, so Apache-2.0 is the
   basis relied on, the same basis on which Kokoro was accepted. Publishing
   anything remains a separate act requiring its own confirmation.

## Accepted Risks

- No arena or listening evidence supports a quality claim for this family, and
  the project has no capability to produce one. Delivery is `port_validated`.
- The 1.7B Stage 3 variant may not meet a CPU practicality bar that the 0.6B
  stages meet. Stage 3 may ship CPU-slow or not at all without invalidating
  Stages 1 and 2.
- Mature MIT prior art exists for this exact model. The project's value here is
  integration under one validated ABI, not novelty.
