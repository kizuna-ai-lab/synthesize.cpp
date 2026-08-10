# OmniVoice 0.6B

Status: Confirmed 2026-08-10. F32, F16, and Q8 are `port_validated`, and all
three were published on 2026-08-09 -- **F16 is the default recommendation** and
Q8 the smaller option with a voice caveat; see "Package," below.
`Q4_K` and `BF16` were measured and are not published. **Quality evaluation has
not been run.** A
Listening Audit on 2026-08-07 found no obvious regression across six pairs
(`no_obvious_regression`) -- see "Listening Audit," below; neither claim moves
the Validation Level. Two corrections landed 2026-08-08 after a further audit:
see **"Short canvases produce unintelligible output"** and the auto-voice
speaker note under "Package" -- both describe the shipped configuration. The
quantization tables under "Package" were re-measured on 2026-08-09 under the
conv-exempt codec policy; `Q8_CODEC_MIXED` means something different for this family
since that date and still does not ship.
This is a **Restricted Model Package** (ADR 0018), not a
Published Model Package: the generator (LM) weights are CC-BY-NC (no version
stated by upstream) and the codec weights carry the Boson Higgs Audio 2
Community License. **It was published on 2026-08-09** to
`jiangzhuo9357/omnivoice-0-6b-gguf` on jiangzhuo's separate, per-act
confirmation (ADR 0018's last consequence), and corrected there on 2026-08-10
when a review found the package short of the Meta Llama 3 licence the Boson
agreement requires. Every later edit to this page's artifacts changes a live
repository and needs its own confirmation to re-upload. This document records
the artifacts and the commands that were run, without
running them.

## Package

The package is converted from the flagship `k2-fsa/OmniVoice` checkpoint at
pinned source revision `468e927ba3716cd8dd86421148dfb3046e9f9d7b` (package
0.2.1) and pinned weights revision `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`.
It is a non-autoregressive mask-predict diffusion language model: a
bidirectional Qwen3-0.6B backbone runs full attention (no KV cache) over a
fixed-length canvas of 8 acoustic codebooks x 1025-entry vocabulary at 25 Hz,
refined over 32 parallel denoising steps (16 in fast mode) with
classifier-free guidance at scale 2.0. The committed token grid is decoded to
24 kHz mono F32 PCM by the Higgs Audio V2 codec's DAC-style convolutional
decoder at hop 960.

Voice arrives through three modes mapped onto the public Voice Profile
sources: **Reference Audio** cloning (a transcript is required, language
optional), **Description Text** voice design, and an unnamed auto-voice
default with an empty Preset Voice Catalog.

**Auto-voice has no speaker conditioning, and the seed alone does not pin the
speaker.** Corrected 2026-08-08; this page previously said the speaker follows
the synthesis seed, full stop, which is not true. With no profile supplied
there is no speaker signal in the prompt at all and the model carries no
speaker embedding table, so which speaker you get is emergent from which token
grid the decode lands on. **Reproducing a speaker requires the same seed *and*
the same Execution Backend, package, and step count.** Switching between the
shipped CPU and CUDA backends can produce a different speaker for an otherwise
identical request: on one measured case the CPU path's median F0 is 118 Hz and
the CUDA path's is 189 Hz -- a male voice and a female voice for the same call.
Callers who need a stable identity should build a Voice Profile from Reference
Audio or Description Text; those paths are conditioned and are not subject to
this.

**How often, measured.** Both backends were rendered for all twenty Golden
cases at the shipped step count and compared on median F0 from two independent
pitch trackers: **one case of the seventeen greedy cases changes speaker**
(`omni-short-en`). Every other measurable case stays in the same register on
both trackers. `omni-rate-fast` is excluded rather than counted, because its
own reference audio is degenerate and neither arm carries a voice to compare.
A listener heard the changed pair blind, reported the two arms as different
people, and judged their audio quality indistinguishable -- so this is an
identity effect, not a quality one.

**Token drift does not predict it.** The case with the highest disagreement
between backends, 98.3% of committed tokens, keeps the same speaker
(178 Hz against 183 Hz). A caller cannot infer speaker stability from how much
the two backends' tokens differ, and neither can this project's tolerance
grid -- which is why the finding took a listening pass to surface.

The package declares a validated
Language Capability Catalog of `en`, `zh`, `ja`; the checkpoint claims 600+
languages through its training data and prompt format, but no language beyond
these three has its own validation case.

Input is **raw UTF-8 text only**. The package carries its own embedded
byte-level BPE frontend (`synthesize.qwen_bpe`, the same vocabulary layout
qwen3-tts uses, hoisted to a shared internal module) and declares
`input_flags = SYNTH_INPUT_SUPPORT_TEXT_UTF8` exactly -- unlike VITS, Kokoro,
and qwen3-tts, this package does **not** accept a raw token-ID bypass or
phoneme input: v1 assembles its own prompt from text, so a token-sequence
input has no consumer (`docs/porting/families/omnivoice.md`, "Amended
2026-07-31"), and the loader refuses to load any package declaring more than
the text flag.

Three profiles are published. Each quantizes the **generator**
half only and leaves all 486 `codec.*` tensors bit-identical to the F32
package, so each inherits this family's exact-token guarantee rather than
approximating it (see "Port validation," below):

| Profile | Bytes | Off F32 | Tensor storage | SHA-256 |
| --- | ---: | ---: | --- | --- |
| F32 | 3,189,953,504 | — | 798 F32 | `f6d504ffaddcbf32f80f1f6c847f075bbd5d2c7b50fe95a194ceb635772f9fa3` |
| F16 | 1,964,929,440 | 38.4% | 599 F32 + 199 F16 | `65c8cca59b350ccfdc6ad96c5a683b276f8c0fd5c8e96675d6dc110da3f52f70` |
| Q8 | 1,390,699,680 | 56.4% | 599 F32 + 199 Q8_0 | `61aec0de7cfa9246487e309c43508de9c95cf52fd225ce3fada3b4bb4982374e` |

What each one costs and buys, measured 2026-08-09. "LTAS" is long-term average
spectrum distance from F32's own CPU render, worst case over the seventeen
greedy Golden cases; roughly 3 dB is the line a listener has consistently
called "same person", calibrated on fourteen answers across two audits with no
exceptions:

| Profile | Worst LTAS vs F32/CPU | Accelerator resident | Load time | CUDA throughput vs F32 | CPU throughput vs F32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| F32 | — (reference) | 3506 MiB | 2.6 s | — | — |
| F16 | 2.87 dB | 2353 MiB | 1.6 s | **2.7% slower** | 1.2% quicker |
| Q8 | 12.22 dB | 1811 MiB | 1.1–1.4 s | 0.4% quicker | 1.5% slower |

**No profile is faster, and none may be described as faster.** The generator is
compute-bound at roughly 205 multiply-accumulates per weight byte, so a
narrower weight does not relieve the bottleneck; the honest claim for every
profile is size, memory and load time.

**F16 is the default recommendation.** Every one of the seventeen greedy cases
lands under the ~3 dB line, and it preserves both conditioned paths: Reference
Audio cloning at 0.22 / 0.12 dB and Description Text at 0.00 / 1.38 dB.

**Q8's caveat is a different voice, not a smaller feature set.** Nothing is
disabled -- all three voice modes work and the API is identical. It reproduces
F32's voice for Reference Audio cloning (2.02 / 2.70 dB). For auto-voice and
for Description Text it returns a *different* voice: LTAS 4.12–12.22 dB, and a
listener heard five of six sampled auto-voice pairs as different people **while
judging the quality of both indistinguishable on every one of them**. It still
honors a Description Text prompt, which the design cases make directly
checkable because they specify pitch: "female, young adult, high pitch" gives
F32 333.3 Hz and Q8 343.5 Hz; "男，老年，低音調" (male, elderly, low pitch)
gives 152.9 and 158.4 Hz, against ordinary voices in this suite at 118–130 Hz.
The 8.61 dB on `omni-design-en` means "a different voice within the same
description", **not** "the description was ignored" -- **never describe Q8 as
clone-only or as breaking voice design.** For auto-voice the caveat is barely
news: this page already states above that the speaker is unstable across
backends and seeds, because auto-voice carries no speaker conditioning at all.

The public C ABI is profile-independent. C++, Rust, and Python callers load a
local GGUF through the same model interface. The tensor catalog, RVQ dequant
path, and execution details remain private to the architecture module.

### Short canvases produce unintelligible output

Recorded 2026-08-08. **This is the one hazard on this page a caller can reach
without doing anything unusual, and nothing in the runtime warns about it.**

The model paints a fixed-length canvas whose length is estimated from the text
and then divided by the speaking rate. Below roughly **37-40 frames (~1.5-1.6
seconds)** the rollout has too few positions to place the text and the output
degenerates into a near-DC, sub-50 Hz rumble instead of speech: no amplitude
envelope, no silence, a large DC offset. Two ordinary requests reach it:

- **A high speaking rate on a short text.** The package declares a
  `speaking_rate_range` of `[0.5, 2.0]` and `synthesize-cli --rate 2.0` is
  accepted silently, but the pinned upstream oracle is already degenerate at
  rate 1.50 on a 32-character sentence, and upstream's own demo UI caps speed
  at 1.5. **The top of the declared range is not validated as speakable.**
- **A very short text at the default rate.** `--text "Hi."` resolves to a
  22-frame canvas and degenerates with no rate change at all. **No minimum
  input length is enforced anywhere.**

Rate 2.0 on a *long* text is fine, so the variable is the resolved canvas
length, not the rate as such.

**This is the upstream model's own behavior, faithfully reproduced, not a port
defect**: on the affected golden case this port matches the pinned PyTorch
oracle's waveform at Pearson r = 0.999670, the tightest of any case measured,
and the oracle's own reference audio for that case is equally unintelligible.
The golden suite does not catch it because port validation compares against the
oracle and makes no intelligibility claim (ADR 0017). Added 2026-08-09: the
generated model card now carries a short warning about this, so a downloader
who never reads this page still meets it before their first request — but that
is documentation, not a guard. Lowering the declared
range and emitting a diagnostic on a too-short canvas are both recommended and
neither has been done; see
`reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`'s 2026-08-08
step-count audit entry, Finding 1.

### What the profiles quantize, and what they never do

**A profile here quantizes one half of the model, never both.** This family has
two halves with opposite risk — the Qwen3-based generator that paints the token
canvas, and the Higgs Audio V2 codec that turns the canvas into audio — and
`classify_tensor_for_half` enforces the split, so there is no combined profile.
Which half a profile quantizes determines its name (jiangzhuo, 2026-08-09):
the generator half takes the plain name, the codec half takes a `_CODEC`
qualifier.

**The generator-half profiles ship; every codec-half profile is blocked.** That
is the reverse of what this page said through 2026-08-08, and the history is
worth keeping because the reasoning moved twice. By jiangzhuo's ruling of
2026-08-06 the scope was codec-only, on a measured prior finding rather than a
convention: a reference port measured greedy-decode token agreement collapsing
from 100% to roughly 7% under an F16 *generator*, because an argmax flip at one
committed step feeds back into every later step of the same synthesis. That
scope was lifted on 2026-08-09, because the generator is 76.9% of the tensor
bytes and no codec-only profile can go below about 2.62 GB however aggressive
it is. Four generator profiles were produced and measured. All four confirm the
prior finding about token agreement, and none is refuted by it — a quantized
generator emits a *different valid realization*, not a wrong one, so token flip
is recorded as data and is never a gate here. What decides a profile is whether
the voice changed, which only a listener can answer.

| Generator profile | Bytes | Reduction | Greedy token flip | Status |
| --- | ---: | ---: | ---: | --- |
| `F16` (halved) | 1,964,929,440 | 38.4% | — | **ships — default recommendation** |
| `Q8` (Q8_0) | 1,390,699,680 | 56.4% | 95.83% | **ships — the smaller option** |
| `Q4_K` (Q4_K + Q8_0 pin) | 1,166,300,320 | 63.4% | 98.99% | measured, **not published** |
| `BF16` (bfloat16) | 1,964,929,440 | 38.4% | — | measured, **not published** |

Every generator profile's codec half is bit-identical to the F32 package, so its
clone RVQ encode is byte-exact and it passes every hard gate — which is exactly
why this half is the shippable one. `Q4_K` is not
published because ten of seventeen renders go degenerate at 62× the probe drift
while buying no throughput at all; `BF16` is not published because eleven of
seventeen land over the quality line and it is 10.5× slower on CPU. Their code
and their rows stay; no package is cut. The full comparison, the packages'
digests and the reasoning are in
`reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md` (2026-08-09) and
`docs/quantization.md`.

The two shipped generator packages carry the same tensor bytes as the
`F16_GEN` / `Q8_GEN` artifacts every measurement on this page was taken from.
Verified 2026-08-09 rather than assumed: re-cutting under the new names changes
only the `synthesize.quantization.profile` string in the GGUF header, which
shifts the bytes behind it inside the header and nothing else. Both pairs
compare byte-for-byte identical from `tensor_data_start` (offset 5,387,264, the
same in old and new because the 32-byte alignment padding absorbs the shorter
string) to end of file; every difference `cmp -l` reports lies in the header,
below that offset. The measurements transfer unchanged because the weights are
the same weights.

Codec-only profiles were produced and measured against this family's own
exact-token gate (below), and **every one of them failed it**:

| Profile | Bytes | Reduction | Clone RVQ tokens mismatched | Greedy grids |
| --- | ---: | ---: | ---: | ---: |
| `Q8_CODEC_MIXED` (current, conv-exempt) | 2,778,427,360 | 12.9% | 98 of 2,808 (3.49%) | 17/17 exact |
| `F16_CODEC` | 2,858,422,240 | 10.4% | 103 of 2,808 (3.7%) | 17/17 exact |
| `Q8_CODEC_MIXED` (superseded, packed convs) | 2,703,016,576 | 15.3% | 1,023 of 2,808 (36.4%) | 17/17 exact |

Both codec profiles were called `Q8_MIXED` and `F16` when they were measured;
they took the `_CODEC` qualifier on 2026-08-09 when the plain names went to the
generator half. `Q8_CODEC_MIXED` means something different for this family since the conv-exempt
codec policy of 2026-08-09: it no longer block-quantizes any convolution
kernel, so 85 of the 158 codec matrix weights are held at F16 and only the 73
HuBERT Linears are Q8_0. The last row is what the same command line produced
before that change and is **no longer reproducible**; it is kept here because
the 36.4% figure it belongs to has been quoted. Exempting the convolutions cut
the clone drift by a factor of 10.4 and is what makes those first two rows
nearly equal -- the drift that remains is the precision of the convolutions,
not of the Linears (`docs/porting/families/omnivoice.md` has the four-cell
attribution).

Every *codec-half* profile reproduces the greedy decode loop's 8 x T token grid
exactly, for a structural reason rather than luck: the generator and the RVQ
are Sensitive/F32 under a codec-half profile, so the decode loop's logits are
bit-for-bit identical to the F32 package's. (The shipped generator-half
profiles are the mirror image of this: they re-draw the greedy grid, which is
recorded as data and never gated, and leave the codec bit-identical so the
cloning grid below stays exact.) What fails on the codec half is the
**Reference Audio cloning path's own RVQ encode**: quantizing the clone-encode
path moves the
fused latent that feeds the encode's nearest-neighbor codebook lookup
(`ref.fused_latent` max_abs 9.32e-05 at F32 versus 0.123921 at the current
Q8_CODEC_MIXED, 0.129286 at F16_CODEC and 3.73227 at the superseded one), which flips a
discrete nearest-neighbor decision at a large fraction of frames -- not a
knife-edge margin call eligible for the dual-admissibility mechanism, in any
case. Per this family's own gate discipline, a profile that fails the
exact-token gate is not shipped and no perceptual claim substitutes for it.
`tests/tolerances/omnivoice.json` still has exactly one measured cell, F32 on
CPU, for every profile including the two that now ship: a generator-half
profile re-draws the grid by design, so there is nothing for a per-profile
oracle-parity cell to hold it to, and the measurement stands as the record
instead
(`docs/porting/families/omnivoice.md`'s Quantization Profile Shape section;
`reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`'s Plan 4 Task 3 and
2026-08-09 entries).

## Port validation

Twenty deterministic Golden cases run against the pinned upstream PyTorch
oracle on DGX Spark CPU and NVIDIA GB10 CUDA 13.3: 17 greedy (both
temperatures pinned to zero, which makes the whole diffusion loop deterministic
and RNG-free) and 3 public-sampled-path cases. Per-stage oracle dumpers cover
the tokenizer, the prompt builder, a single conditional forward (probed at
five internal depths plus the final hidden state and the step-0 guided
logits), the codec decode, the HuBERT semantic branch, the RVQ encode, and the
end-to-end waveform -- eight stages in total.

**This family's real guarantee is exact token identity, not a tolerance, and
it is two separate claims:**

- The greedy decode loop's full 8 x T unmasking grid matches the oracle's
  byte-for-byte on **17 of 17** golden cases (one, `omni-fast-mode`, via a
  committed alternate grid the oracle itself also produces -- the
  dual-admissibility mechanism for a decision narrower than this arithmetic
  resolves, never a tolerance on a token id).
- The Reference Audio cloning path's own RVQ encode grid matches the oracle's
  byte-for-byte on **both** clone cases -- 8 codebooks x 351 frames, **2,808 of
  2,808 tokens exact**, on the first run.

Both grids stayed byte-exact when the codec's decode graph moved to CUDA too
(below): a flip in either grid, on any profile or backend, is a shipping
blocker by this project's own exact-token discipline, and none has occurred.

Everything else -- the deep generator probes and the decoded waveform -- is
measured for completeness rather than enforced as this family's real claim,
because pre-norm outlier channels make a sample-wise bound reject a provably
correct port: `generator.hidden_l27` reaches a max-abs of 0.08 while its
cosine still holds at eight nines. Worst figures, CPU F32 against the oracle:

| Probe | Worst cosine deviation (1 − cosine) | Worst max\_abs |
| --- | ---: | ---: |
| `generator.hidden_l0` | 1.30e-07 | 5.72e-05 |
| `generator.hidden_l7` | 1.97e-07 | 4.12e-04 |
| `generator.hidden_l14` | 9.46e-08 | 7.02e-04 |
| `generator.hidden_l21` | 1.37e-07 | 4.15e-03 |
| `generator.hidden_l27` | 2.14e-07 | 8.01e-02 |
| `generator.final` | 1.72e-07 | 1.04e-03 |
| `generator.logits_step0` | 9.06e-08 | 6.10e-04 |
| `audio.pcm` / `audio.pcm_freerun` | 1.44e-07 | 1.69e-05 |
| `ref.pcm_16k` (clone resampler) | 5.56e-08 | 1.19e-07 (≈1 float32 ULP) |
| `ref.semantic_mean` (clone HuBERT) | 7.07e-08 | 6.09e-05 |
| `ref.fused_latent` (clone DAC + fusion) | 0.0 (capped) | 9.32e-05 |

Every committed tolerance is 5x the measured deviation
(`tests/tolerances/omnivoice.json`, `reference_stage:
source-f32-oracle-vs-f32-cpu`); both sides run F32 on CPU for this
comparison, so the thresholds carry neither a dtype nor a device
difference, only an implementation one.

### Backends: CPU baseline, CUDA now the whole graph

CPU is the mandatory baseline; every measurement above is against it. CUDA
support **covered only the codec's decode graph through Plan 4** (the RVQ
dequantizer, the acoustic decoder, and the final projection -- 152 tensors),
with the generator held on CPU unconditionally under `docs/backends.md`'s
discrete-outputs rule. **Superseded 2026-08-08 (Plan 5 Task 1):** after
jiangzhuo revised this family's bar from token identity to audible quality
(a six-pair blind A/B heard no problem in generator-on-CUDA output), the
generator earned `docs/backends.md`'s new "one narrow exception" to that
rule -- its discrete token choice never changes the canvas shape, only its
content -- and now moves to CUDA too, mirrored through its own
accelerator-resident weight twin.

Measured 2026-08-07 across all twenty Golden cases with `--accelerate`
(Plan 4, codec-only, kept for the record): every codec node left the CPU and
every generator node did not -- aggregated over the whole suite, codec 8,440
of 8,440 nodes off the CPU, generator 0 of 880,032. All seventeen greedy
token grids and both cloning RVQ grids stayed byte-exact against the CPU
baseline. The one artifact that moved was the decoded waveform, because it
is what the accelerated codec's TF32 arithmetic actually touches:

| Backend | `audio.pcm` worst cosine | `audio.pcm` worst max\_abs |
| --- | ---: | ---: |
| CPU (F32 baseline) | 0.99999986 | 1.69e-05 |
| CUDA (codec only, Plan 4) | 0.99999635 | 7.01e-03 |

committed to `tests/tolerances/omnivoice.json`'s `backends.CUDA.stages.replay`
cell at `min_cosine 0.999981` / `max_abs 0.04` (five times the measured
deviation). Once the generator also moves (Plan 5), token *content* -- not
grid size -- diverges from the CPU baseline in most cases; this is a
deliberate, measured, and audited trade (see the family doc's Listening
Audit), not a regression.

**RTF, corrected.** The Plan 4 codec-only figure below (9.68x faster codec,
3.4% end-to-end reduction) was measured on the `dev-dgx-spark` preset, whose
`RelWithDebInfo` build compiles ggml-cpu at `-O2` -- 2.19x slower than the
shipped `Release`/`-O3` build -- so it is doubly stale: superseded by the
generator's own move, and pessimistic on top of that. The honest, corrected
pair, from a Release CUDA tree (`build/rel-dgx-spark`) with the generator
also on CUDA, same case (`omni-long-boundary`, 719 frames):

| Backend | Time | RTF |
| --- | ---: | ---: |
| CPU | 122.61 s | 4.263 |
| CUDA (generator + codec) | 5.481 s | 0.1906 |

**22.4x**, faster than real time. (For the record, the superseded Plan 4
figure: codec alone was 9.68x faster [4.3069 s to 0.4451 s] but the CPU-held
generator was 98.9% of wall time, so the end-to-end effect was a bounded
3.4% reduction, 267.30 s to 258.48 s, RTF 9.294 to 8.987 -- the opposite
shape from a GPU-primary family, where holding a minority stage on CPU is a
tax rather than a small saving.) Full method, the twenty-case sweep, the
corrected RTF measurement, and the operational-evidence tables (latency,
repeated-run cleanup, and why peak memory has no second budget on this UMA
host) are in `docs/porting/families/omnivoice.md`'s Execution Backends
section and `reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`'s
2026-08-07 Task 11 entry and 2026-08-08 erratum.

### Listening Audit

One maintainer, six A/B pairs, 2026-08-07: **`no_obvious_regression`**. Five
pairs compare the port's replayed codec against the pinned PyTorch oracle
(covering the worst and second-worst waveform cosine, one Reference Audio
clone case, one Description Text case, and one random pick); the sixth
compares the codec's CUDA decode against its CPU decode of the identical
byte-exact committed token grid, deliberately on the suite's longest-duration
case (also the codec's largest measured CUDA speedup, 9.68x at 719 frames) --
its inaudibility corroborates the backend claim rather than merely
accompanying it. This is one listener, six pairs,
non-statistical: it says no obvious problem was noticed on the pairs heard,
not that the port and the oracle are perceptually equivalent, and it does not
move `quality_evaluation` off `not_run` (ADR 0017's automated grid has not
run and is not scheduled). Full identity key, seeds, and method are in
`docs/porting/families/omnivoice.md`'s Listening Audits section and
`reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`'s 2026-08-07 Task
16 entry.

**A later audit rejected a candidate configuration, 2026-08-08.** Halving the
decode step count from the shipped 32 to 16 buys 44.6% of the wall clock; a
six-pair blind A/B returned four "no difference" and one clear preference for
the 32-step arm (`omni-short-ja`, whose 16-step audio is incomplete at the
end), so **16 steps is rejected and the shipped default stays at 32**. No code
changed, and `num_step` is not reachable through the public C interface in any
case. That audit is also where the short-canvas hazard above and the
auto-voice speaker correction in "Package" came from -- both concern the
**shipped** 32-step path, both were traced to the pinned oracle rather than to
this port, and both produced documentation corrections rather than a code
change. Neither alters the 2026-08-07 `no_obvious_regression` verdict, whose
subject is what ships.

## Reproduction

Materialize the reference artifacts, then run the registered validators:

```bash
uv run --project scripts/envs/omnivoice --locked \
  python scripts/dump_reference_omnivoice_pytorch.py \
    --manifest tests/golden/omnivoice/omnivoice-0-6b.manifest.json \
    --weights-dir models/omnivoice-0-6b

cmake -S . -B build -DSYNTH_BUILD_TESTS=ON -DSYNTH_BUILD_PYTHON_TESTS=ON \
  -DSYNTH_BUILD_INTEGRATION_TESTS=ON
cmake --build build --target synthesize-check-integration
```

The CUDA cell above is reproduced the same way on a `dev-dgx-spark`-configured
tree, adding `--accelerate` to `scripts/validate-omnivoice-replay.py`'s own
invocation (`docs/backends.md`;
`reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`'s 2026-08-07
entry has the exact command line).

The two shipped Quantization Profiles are cut from the F32 artifact with the
generic quantizer tool:

```bash
cmake -S . -B build -DSYNTH_BUILD_TOOLS=ON
cmake --build build --target synthesize-quantize
build/bin/synthesize-quantize \
  models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf \
  models/omnivoice-0-6b/omnivoice-0-6b-F16.gguf --quant F16
build/bin/synthesize-quantize \
  models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf \
  models/omnivoice-0-6b/omnivoice-0-6b-Q8.gguf --quant Q8
```

The measured-but-unpublished `Q4_K` and `BF16` are cut the same way. So are the
blocked codec-half profiles, which reproduce a negative measurement rather than
a package:

```bash
build/bin/synthesize-quantize \
  models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf \
  models/omnivoice-0-6b/omnivoice-0-6b-F16_CODEC.gguf --quant F16_CODEC
build/bin/synthesize-quantize \
  models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf \
  models/omnivoice-0-6b/omnivoice-0-6b-Q8_CODEC_MIXED.gguf --quant Q8_CODEC_MIXED
```

Neither codec-half output passes the exact-token gate (see Package).

Generate and verify the model card. Unlike the reproduction commands above,
this one is written as the plain sibling invocation on purpose (no
`--output` override): `generate.py`'s own `model_dir` resolution
(`REPO_ROOT / "models" / model_slug`) targets `models/omnivoice-0-6b/` by
default for every family, VITS and Kokoro included, and this family's card
must answer that same invocation the same way theirs do:

```bash
uv run scripts/hf_cards/generate.py scripts/hf_cards/omnivoice-0-6b.yaml
uv run scripts/hf_cards/generate.py scripts/hf_cards/omnivoice-0-6b.yaml --check
```

### The publication directory

`models/omnivoice-0-6b/` is the *working* directory: it holds the upstream
checkpoint (`model.safetensors`, `audio_tokenizer/`, `tokenizer.json`, the
upstream `config.json`, and, now, our own generated `README.md`) beside every
produced GGUF, because the converter, the oracle dumpers, and the (blocked)
quantizer all read and write there. `hf upload` pointed at it would publish
the raw upstream weights alongside the port.

**The upstream card was moved, not overwritten in place.** Before this task,
`models/omnivoice-0-6b/README.md` *was* the upstream checkpoint's own README
(no `synthesize.cpp` prose anywhere in it) — the one artifact that
`generate.py`'s default `--check` invocation actually verifies, and it was
never replaced. A copy was made to `models/upstream/omnivoice-0-6b/README.md`
first; the default location now holds our generated card, exactly like every
sibling family's `models/<slug>/README.md`, and `source.card_path` in the
YAML points at the moved copy so the "Original upstream project card"
section at the bottom of the generated README — and any future re-generation
— resolves against a copy that is never mixed with our own artifacts, the
same pattern Kokoro's `models/upstream/kokoro-v1_0/README.md` already uses.

`models/publish/omnivoice-0-6b/` is the clean, flat publication directory
this task built because, unlike Kokoro's or VITS's working directories, this
family's working directory is not itself flat — it still carries the
upstream checkpoint files above. It contains **exactly** the five artifacts
that would be uploaded, plus the generated card:

```text
models/publish/omnivoice-0-6b/
├── omnivoice-0-6b-F32.gguf       # 3,189,953,504 bytes, sha256 f6d504ff…772f9fa3
├── omnivoice-0-6b-F16.gguf       # 1,964,929,440 bytes, sha256 65c8cca5…a3f52f70 (default recommendation)
├── omnivoice-0-6b-Q8.gguf        # 1,390,699,680 bytes, sha256 61aec0de…4982374e
├── LICENSE-higgs-audio-2.txt     #         9,171 bytes, sha256 ac933dc0…df2fa049 — declared Sidecar Resource, from the weights
├── LICENSE-meta-llama-3.txt      #         7,801 bytes, sha256 47521163…1356819f — declared Sidecar Resource, from this repository
└── README.md                     # the freshly generated card
```

Every entry is a hard link to its working-directory original, sharing the same
filesystem and inode rather than duplicating 6.5 GB, which is safe because
`models/` is entirely git-ignored and this directory is a working artifact, not
a tracked one — and it means the publication directory can never drift from the
working copy that `--check` verifies. Verified 2026-08-09 by listing and
digesting it: five entries, every one at link count 2, digests as above; no
upstream checkpoint file, no `audio_tokenizer/`, no `tokenizer.json`, and no
`Q4_K`, `BF16`, `Q8_CODEC_MIXED` or `F16_CODEC` GGUF (none of those ships).
`LICENSE-meta-llama-3.txt` is the sixth entry, added 2026-08-10, and re-checked
the same way on that date.
`tests/python/test_hf_card_generator.py`'s
`test_omnivoice_publish_directory_is_flat_and_current` re-checks that set
against the card spec on every unit run, so a profile added to the spec and not
to this directory — or the reverse — fails the gate rather than being noticed
at upload time.

Seven stale-named packages — the four cut under the retired `_GEN` names, and
three codec-half packages including the one whose `…-F16.gguf` filename the
generator-half profile now claims — were moved to
`models/omnivoice-0-6b/retired-profile-names/` before the re-cut, rather than
deleted or overwritten. They sit outside the publication directory and cannot
be uploaded by accident.

### Publication commands (run 2026-08-09, and again 2026-08-10)

**Executed on jiangzhuo's per-act confirmation naming this exact target**
(`jiangzhuo9357/omnivoice-0-6b-gguf`, `main`) --- an approved plan is not that
confirmation, and each of the two uploads was confirmed on its own. The second
added `LICENSE-meta-llama-3.txt` and corrected the card's RTF and Q8 figures.
The commands, exactly as they were typed:

```bash
hf repos create jiangzhuo9357/omnivoice-0-6b-gguf \
  --type model --public --exist-ok
hf upload jiangzhuo9357/omnivoice-0-6b-gguf models/publish/omnivoice-0-6b . \
  --commit-message "Publish OmniVoice 0.6B F32/F16/Q8 (Restricted Model Package: CC-BY-NC generator + Boson Higgs Audio 2 Community License codec)"
```

That upload would carry 6.5 GB across three GGUFs, the sidecar license and the
card. Confirmation has to be for that set, not for "the package": F16 and Q8
were added to it on 2026-08-09 and the earlier recorded command named F32
alone.

## Licensing

Three separate upstream grants apply; only two touch the artifacts in
`models/publish/omnivoice-0-6b/`. All three GGUFs there carry both halves of
the model, so both of those grants apply to every one of them — quantizing the
generator half changes nothing about which license governs it.

**The generator (LM) weights are CC-BY-NC, with no version stated.** The
upstream model card has no `license:` frontmatter key; the only statement is
this prose, quoted verbatim from the pinned weights revision:

> Our code is released under the Apache 2.0 License. The pre-trained model is
> licensed under the CC-BY-NC due to constraints from its training data (e.g.,
> Emilia).

Upstream names no CC-BY-NC version anywhere on the card, so this project's
license metadata says exactly that instead of inventing one -- `license: other`
with `license_name:
omnivoice-cc-by-nc-unspecified-version-plus-boson-higgs-audio-2-community`,
never `cc-by-nc-4.0` (ADR 0018:29-31; the ruling of 2026-08-06). **These
weights are not licensed for commercial use.** The
restriction traces to the training data the statement itself names --
Emilia, a large-scale multilingual speech corpus distributed under its own
non-commercial terms.

**The codec (Higgs Audio V2) weights carry a second, separate license: the
Boson Higgs Audio 2 Community License**, bundled by the weights repository as
`audio_tokenizer/LICENSE` and carried with this package as a declared
**Sidecar Resource** (`LICENSE-higgs-audio-2.txt`), never a footnote. That
agreement is derived from the Meta Llama 3 Community License and requires
dual attribution on redistribution:

> "Meta Llama 3 is licensed under the Meta Llama 3 Community License,
> Copyright © Meta Platforms, Inc. All Rights Reserved."
> "Boson Higgs Audio 2 is licensed under the Boson Community License,
> Copyright © Boson AI USA, Inc. All Rights Reserved."

Two further obligations and one required notice, in the agreement's own terms.
**Corrected 2026-08-09**, where this page still carried a paraphrase the
generated card fixed on 2026-08-08: it said the agreement "caps commercial use
at 100,000 monthly active users", which was wrong twice over and wrong in the
reader's favour.

Above **100,000 annual active users in the preceding calendar year**, section 2
imposes no ceiling: it withdraws authorisation. You "must request an expanded
license from Boson AI, which Boson AI may grant to you in its sole discretion,
and you are not authorized to exercise any of the rights under this Agreement
unless or until Boson AI otherwise expressly grants you such rights."

Section 1.b.i(v) forbids using the Higgs Materials "or any output or results of
the Higgs Materials to improve any other large language model (excluding Boson
Higgs Audio 2 or derivative works thereof)" -- other *large language models*,
with Boson's own carved out, not all models as this page previously said.

That agreement also requires a redistributor to display the following notice,
which it specifies verbatim and which the generated card carries for that
purpose:

> "Built with Higgs Materials licensed from Boson AI USA, Inc., Copyright
> Boson AI USA, Inc., All Rights Reserved and Meta Llama 3 licensed under the
> Meta Llama 3 Community License, Copyright Meta Platforms, Inc., All Right
> Reserved"

All three are independent of, and in addition to, the generator's own CC-BY-NC
restriction above.

**That agreement also carries the Meta Llama 3 Community License into the
package as a second declared Sidecar Resource** (`LICENSE-meta-llama-3.txt`).
It is not an optional courtesy: the Boson agreement *defines its own name* to
include Meta's -- "'Agreement' means the terms and conditions … set forth
herein and the Meta License Agreement" -- and section 1.b.i(A) separately
requires "a copy of this Agreement and the … Meta License's Llama 3 agreement"
to accompany any redistribution of the Higgs Materials. Shipping only the Boson
text satisfied neither reading, which is the gap the 2026-08-10 correction
closed.

The two sidecars have different origins, and therefore different pins.
`LICENSE-higgs-audio-2.txt` is extracted from the weights
(`audio_tokenizer/LICENSE`) and is digest-checked against the manifest pin for
that input. Upstream bundles no copy of the Meta text at all, so this
repository commits one at `scripts/licenses/LICENSE-meta-llama-3.txt` (7,801
bytes, sha256 `47521163…1356819f`) — committing a third-party license text for
redistribution follows `bindings/python-native-cu13/LICENSE-ggml`. Since
nothing upstream can witness that copy, the digest above is what pins it, and
`scripts/convert-omnivoice.py` copies it beside the artifact on every
conversion and checks the copy against that digest.

The committed text is the April 18, 2024 version the Boson agreement cites,
taken from two independent Hugging Face mirrors that agree byte for byte (a
third agrees on wording after whitespace normalization), because the URL the
agreement names now redirects to a JavaScript page no fetch can extract text
from. Before this, the file was placed into the publication directory by hand
for the 2026-08-10 upload and nothing in the pipeline produced it: a fresh
clone plus a conversion yielded a package this page described but the tree
could not reproduce. It has an owner now, and
`tests/python/test_convert_omnivoice.py`'s `LicenseCarriageTests` fails if it
stops being carried or its text drifts from the pinned digest.

**Apache-2.0 covers only the upstream GitHub source code**
(`k2-fsa/OmniVoice`) and nothing produced by this project: no weight file,
converted artifact, or GGUF this project ships may ever be labelled
apache-2.0. All three community GGML ports read at design time misstate or
omit these terms (one Hugging Face GGUF repository labels the weights
apache-2.0, which is simply false); nothing downstream of upstream's own card
and license files is inherited.

The upstream card also carries a use disclaimer this package repeats
verbatim rather than paraphrases:

> Users are strictly prohibited from using this model for unauthorized voice
> cloning, voice impersonation, fraud, scams, or any other illegal or
> unethical activities. All users shall ensure full compliance with
> applicable local laws, regulations, and ethical standards. The developers
> assume no liability for any misuse of this model and advocate for
> responsible AI development and use, encouraging the community to uphold
> safety and ethical principles in AI research and applications.

Both upstream licenses permit non-commercial redistribution with attribution,
and upstream itself distributes the weights openly on Hugging Face -- the
basis for republishing this converted GGUF under exactly those terms rather
than a broadening of them (ADR 0018). This is a Restricted Model Package, not
a Published Model Package, and must never be described as one.
