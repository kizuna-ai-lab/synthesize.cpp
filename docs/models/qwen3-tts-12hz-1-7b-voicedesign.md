# Qwen3-TTS 12 Hz 1.7B VoiceDesign

Status: Confirmed 2026-08-20. **Prepared, not published.** Three GGUFs
(`BF16`, `F16`, `Q8_MIXED`) and this page's own card spec
(`scripts/hf_cards/qwen3-tts-12hz-1-7b-voicedesign.yaml`) are ready; nothing
has been pushed to `jiangzhuo9357/qwen3-tts-12hz-1-7b-voicedesign-gguf` or
anywhere else. Publication is a separate outward act requiring jiangzhuo's
explicit, per-act confirmation naming the target repository.

`Q8_MIXED` is the smallest and fastest of the three shipped profiles --
41.82 % smaller than `BF16` at RTF 1.05 -- with the thinnest replay headroom
of the three (1.63x) but still comfortably clearing the committed 0.01
bound. `F16` is **larger** than its `BF16` source (+283,136 bytes, 0.0066 %)
and ships anyway, on speed: roughly 2.8x faster on the host this was
measured on (RTF 4.64 -> 1.65), the same "speed profile, not a size one"
shape the Base variant's own `F16` already established for this family. A
fourth profile, `Q5_K_MIXED`, was cut and measured and is **not**
published: its `replay` prefill probe exceeds the committed 0.01 bound on
both measured cases, roughly 3x over (headroom 0.32x) -- see "Package,"
below, for why it is still named here rather than silently absent.

**The VoiceDesign Listening Audit ran 2026-08-20 and recorded
`no_obvious_regression`.** Four blind pairs -- `BF16` vs `F16`, `BF16` vs
`Q8_MIXED`, `BF16` vs `Q5_K_MIXED`, and `BF16` CPU vs CUDA -- were all
judged indistinguishable, one listener, one clip per pair. A separate,
labelled description-control pass returned "yes, in the described
direction" -- weak evidence by design, recorded as weak. See "Listening
Audit," below. Quality Evaluation stays deferred per ADR 0017, and neither
claim moves the Validation Level.

## What this variant is for

Stage 3 of the Reference Model Variant Ladder, and the family's third
distinct Voice Profile source. It adds
`SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT`: a natural-language description in
(pace, pitch, timbre, mood), synthesized audio out, in a voice built from
that description rather than cloned from a recording or picked from a
catalogue.

**It catalogues no Preset Voices and accepts no Reference Audio at all** --
`preset_ids` is an empty array and `SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO` is
absent from the capability snapshot. With no Voice Profile supplied, which
speaker a request produces is emergent and the synthesis seed alone does not
pin it -- the same `seed_default_with_profiles` shape the Base variant's own
page describes, but with Description Text as the only origin rather than
Reference Audio. This is the structural difference from both siblings
shipped so far: CustomVoice selects from nine preset voices and cannot
design one; Base clones from a reference recording and cannot design one
either.

Necessarily moves to the 1.7B talker width -- the Reference Model Variant
Ladder's own prediction, measured true by Stage 3 Task 6: the code
predictor's hidden size (1024) differs from the talker's (2048) for the
first time in this family, which needed a genuine architecture fix (a width
bridge, `small_to_mtp_projection`) rather than a conversion workaround. See
`docs/porting/families/qwen3-tts.md`'s "Stage 3: VoiceDesign Package, Task
6" for the full account.

## Package

659 tensors against CustomVoice's 657 -- two more, exactly the width bridge
(`talker.code_predictor.small_to_mtp_projection.{weight,bias}`) this
variant's 1.7B width needs and no 0.6B package carries.

| profile | bytes | tensor types | vs BF16 | shipped |
| --- | ---: | --- | ---: | :---: |
| BF16 | 4,295,891,904 | 404 BF16 + 255 F32 | -- (source) | yes |
| F16 | 4,296,175,040 | 267 F16 + 392 F32 | +283,136 B larger | yes |
| Q8_MIXED | 2,499,423,680 | 267 Q8_0 + 392 F32 | 58.18 % (41.82 % smaller) | yes |
| Q5_K_MIXED | 1,780,723,136 | 267 Q5_K + 392 F32 | 41.45 % (58.55 % smaller) | **no** |

Digests: BF16 `d0af7617…5524`, F16 `5a6f52d3…1465`, Q8_MIXED
`f8540471…f92a`, Q5_K_MIXED (measured, not shipped) `a47479d9…82bb`.

**F16 being larger than its source is not a defect.** BF16 and F16 are both
two-byte types, so the matrix weights do not shrink while the sensitive
tensors widen BF16 -> F32 -- the exact shape the Base variant's own F16
shows, applied to a package whose matrix half is a proportionally larger
share of the total, which is also why `Q8_MIXED` shrinks proportionally
*more* here than it does on Base (58.18 % of source against Base's 66.3 %).

### Q5_K_MIXED: measured, and named rather than omitted

`Q5_K_MIXED` saves a further 718,700,544 bytes over `Q8_MIXED` -- 28.75 %,
close to the design's own "~25 %" estimate -- but its `replay` prefill probe
(the same two-case, 0.01-bound comparison the table below uses) measures
p95_relative **0.031029** (empty-instruct) and **0.030366**
(non-empty-instruct), both roughly **3x over** the bound rather than under
it with room to spare. Headroom **0.32x** -- the first cell in this
variant's history that reads below 1. It loads and synthesizes through the
public seam (`synth_model_load` and
`synth_voice_profile_create_from_description` both succeed against it, 10
of its 11 non-skipped public-seam checks pass, the failing one being the
prefill-tolerance relation itself), and a 2026-08-20 blind listening pass on
one clip could not tell it apart from `BF16` -- a data point about the
gate's conservatism at this margin, not a recalibration.

**Recommendation: do not publish.** This matches the precedent CustomVoice's
own `Q5_K_MIXED` set (talker-logits cosine 0.9648, withheld on accuracy) but
is the stronger finding: that was a low cosine against no committed
pass/fail bound; this is an explicit breach of a committed gate, on both
measured cases, reproduced on a repeat run. **This is the first card in this
family to record a profile as measured-and-not-shipped rather than leaving
it out.** CustomVoice's own published card omits `Q5_K_MIXED` entirely
(`scripts/hf_cards/qwen3-tts-12hz-0-6b-customvoice.yaml` lists `profiles:
[F16, Q8_MIXED]`) -- a separate, pre-existing gap this page does not fix,
since that card is already published and re-publishing it is its own
outward act. See `docs/quantization.md`'s "VoiceDesign's Q5_K_MIXED" section
for the full arithmetic.

## Measured

All figures from `build/rel-dgx-spark` (`CMAKE_BUILD_TYPE=Release`, sm_121a,
CUDA 13.3, DGX Spark/GB10) unless stated. One sentence ("This is a test of
Qwen three T T S voice design synthesis.") through a Description Text Voice
Profile ("A cheerful, bright female voice speaking with fast pacing and high
energy."), language `en`, seed 7, 10 threads -- fixed and repeated verbatim
across every row.

| profile | frames | audio | synthesis | **RTF** | load | peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 97,920 | 4.080 s | 18.92 s | **4.64** | 2.29 s | 4.95 GiB |
| F16 | 103,680 | 4.320 s | 7.12 s | **1.65** | 2.13 s | 4.95 GiB |
| Q8_MIXED | 120,960 | 5.040 s | 5.31 s | **1.05** | 1.41 s | 3.16 GiB |
| Q5_K_MIXED (context; fails the gate below) | 96,000 | 4.000 s | 4.30 s | 1.08 | 1.07 s | 2.40 GiB |

BF16's row is a later, paired re-measurement (n=8, taken alongside the CUDA
run below) rather than the first pass this same workload originally
produced (n=3, RTF 4.73) -- both are real numbers from the same tree, and
this table carries the later one, per the family record's own forward
pointer between them. F16 and Q8_MIXED are medians of three runs; Q5_K_MIXED
is context only -- Task 4's accuracy failure already settles its publication
question regardless of speed.

RTF is each row's own synthesis time over its own audio length, never
compared across rows by dividing one wall clock by another's -- the four
profiles stop at different frame counts, a property of their weights, the
same autoregressive-stop-decision mechanism that moves this family's
generated length between builds. **F16 alone recovers 87 % of the
wall-clock time `Q8_MIXED` saves** (12.18 s of 13.99 s, Task 5's own n=3
pass -- 19.3006 s BF16 against this table's later n=8 18.92 s, the two
figures within the run-to-run variance both tasks already characterized)
while changing nothing about the package's on-disk footprint -- the
arithmetic that answers why `F16` ships despite saving no disk.

| case | BF16 | F16 | Q8_MIXED | Q5_K_MIXED |
| --- | ---: | ---: | ---: | ---: |
| `voicedesign-empty-instruct-en` | 0.00269 | 0.002689 | 0.006130 | **0.031029** |
| `voicedesign-nonempty-instruct-en` | 0.002903 | 0.002902 | 0.006085 | **0.030366** |
| headroom (0.01 bound / worse case) | 3.44x | 3.45x | 1.63x | **0.32x (FAILS)** |
| public-seam checks | 11 passed, 3 skipped | same | same | 10 passed, 1 failed, 3 skipped |

This is a **prefill-embedding** comparison against the pinned bfloat16
oracle, not a per-case codes-or-waveform replay: no per-case Golden oracle
artifacts exist for this variant yet, so the ordinary 13-case replay loop
(`scripts/validate-qwen3-tts-replay.py --stage replay`) reports "13
selected, 13 skipped for missing oracle artifacts" at **every** profile,
BF16 included -- not an F16 or Q8_MIXED defect. `replay.prefill`'s two cells
above are filled instead by invoking
`tests/qwen3_tts_voicedesign_prefill_real.cpp` directly against the
committed oracle dumps. The three structural skips in `public` are the same
everywhere: a Description Text Voice Profile has no id for the seam to
echo, and two further checks need a preset "dialect" Voice this package
does not catalogue.

## Backends

**CUDA moves exactly one graph here, the same one Stage 1 already placed.**
This variant introduces no reference-audio path, so there is no speaker or
codec encoder to place, unlike Base -- no new placement *decision*, only a
measurement of the one Stage 1 already made. The codec decoder runs on the
device; the autoregressive half (talker and code predictor) stays on the
CPU under the discrete-outputs rule, because a sampled codec token
conditions the next step and the sequence length itself.

Measured on this variant's own fixed workload, BF16, n=8 medians: **18.9220
s -> 17.7564 s and RTF 4.6377 -> 4.3520, both dividing to 6.16 % end to
end.** Narrower than the Base variant's own 6.47 %/6.35 % (~6.5 %/6.3 %), in
the direction the design predicted: this variant's talker runs 3.2x larger
per layer than Base's, so the CPU-bound autoregressive half's share of wall
clock grows and CUDA's relative gain shrinks -- though the margin (about
0.2-0.3 percentage points) is narrower than a literal 3.2x-larger reading
would suggest, because the codec decoder CUDA actually moves is
byte-identical (sha256 `42772743b165…`) across all three family packages,
so the accelerated numerator barely changes between variants while the
CPU-bound denominator grows.

CPU-vs-CUDA structural agreement, measured, not assumed: `public` cell
checks pass identically on both backends (11 of 11 applicable, 3 skipped);
seed-7 PCM output differs in bytes as TF32 predicts (cosine 0.9999988,
max_abs 0.002414, on the ~1e-3 scale `docs/backends.md` predicts for a CUDA
F32 matmul).

**No CUDA sub-grid is committed in the tolerance file**, for a structural
reason rather than an omission: `replay`'s only probe
(`tests/qwen3_tts_voicedesign_prefill_real.cpp`) hardcodes the CPU device,
and `src/arch/qwen3-tts/model.cpp:584`'s device-mirroring loop only copies
`codec.decoder.`-prefixed tensors -- the talker has no device-side weights
to run against regardless of what a driver requests. Matches the Base
variant's own declined CUDA sub-grid for the same structural reason.

## Listening Audit

**Ran 2026-08-20. Result: `no_obvious_regression`.** One listener
(jiangzhuo), delivered as two private claude.ai artifact pages so the
labelled half could not masquerade as blind.

| # | comparison | verdict |
| --- | --- | --- |
| 1 | BF16 vs F16, CPU | indistinguishable |
| 2 | BF16 vs Q8_MIXED, CPU | indistinguishable |
| 3 | BF16 vs Q5_K_MIXED, CPU | indistinguishable |
| 4 | BF16 CPU vs BF16 CUDA | indistinguishable |
| L1 | bright/fast description vs deep/calm/slow description | **yes, in the described direction** (labelled, weak) |

Four blind pairs on the family's fixed workload (text, description, seed 7,
threads 10), A/B positions shuffled from recorded seed 20260820, every pair
confirmed to differ in bytes before anyone listened. Pair 3 carries the
profile that **fails its numerical gate 3x over** -- included in the blind
half by jiangzhuo's 2026-08-20 ruling, unlabelled like every other pair,
precisely to ask the question the numbers cannot: does the breach audibly
manifest? On this clip it did not. **What that establishes, at its actual
strength: the 0.01 bound is conservative relative to audibility at this
margin, on one clip, one seed, one listener -- a data point about the
gate's calibration, not a recalibration.** Task 4's do-not-publish
recommendation for `Q5_K_MIXED` is unchanged by the audit itself.

The labelled half (row L1) used the same sentence and seed on BF16/CPU, two
descriptions differing in a stated direction ("bright, fast" against "deep,
calm, slow"); each clip was labelled with its own description, so the label
*is* the question and the evidence is recorded as weak by design. The
deep/calm/slow clip ran 51 % longer (6.160 s against 4.080 s) --
directionally consistent with "speaking slowly," one objective datum beside
the subjective verdict. The design's own standing rule -- an uncontrolled
description would be a recorded negative, not a published package -- does
not trigger.

One listener, one clip per pair, one seed. This is evidence, not a level.

## Coverage this variant does not have

- **No per-case Golden replay.** The variant's own 13-case manifest exists,
  but no per-case oracle codes or waveform artifacts have been dumped for
  it; only the two prefill-probe cases above are measured against the
  oracle.
- **No Reference Audio and no ICL path.** This package's only Voice Profile
  origin is Description Text; `SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO` is
  absent from its capability snapshot, unlike both siblings.
- **No multi-language validation.** Only `en` is exercised anywhere in this
  variant's record; the checkpoint carries the family's other codec
  language tokens, but none has a validation case here, so none is
  advertised.
- **No CLI coverage.** `examples/cli/` has no `--description` flag, so
  nothing in this repository's CLI can drive this variant's own Voice
  Profile source at all; every measurement above went through
  `synthesize-qwen3-tts-public-real` directly.
- **`token_ids` input is not declared.** This package's GGUF metadata
  records `synthesize.capabilities.input_flags = 1`
  (`SYNTH_INPUT_SUPPORT_TEXT_UTF8` only, confirmed by reading the raw
  metadata key off all three shipped files), so this card declares
  `input_kinds: [text_utf8]` alone. Base's and CustomVoice's own published
  cards additionally claim `token_ids`, but their own GGUFs carry the same
  `input_flags = 1` -- a discrepancy this task found while checking this
  variant's own capabilities, not something this page's file list
  authorizes correcting on those two already-published cards.

## Licensing

Same as the family: `QwenLM/Qwen3-TTS` at `022e286b98fbec7e1e916cb940cdf532cd9f488e`
and the `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` checkpoint at
`5ecdb67327fd37bb2e042aab12ff7391903235d3` both carry an explicit Apache-2.0
grant, confirmed 2026-08-18 directly from the Hugging Face API against the
pinned revision, with no additional restriction in the card's own prose at
that revision -- verified from the upstream model card rather than from any
port's README.

This variant carries no reference-recording licensing question at all: it
has no Reference Audio Voice Profile source, so none of Base's or the
Golden suite's reference-recording provenance applies here.

## Publication

**Not published.** The three GGUFs above and their card spec
(`scripts/hf_cards/qwen3-tts-12hz-1-7b-voicedesign.yaml`) are prepared;
nothing has been uploaded to `jiangzhuo9357/qwen3-tts-12hz-1-7b-voicedesign-gguf`
or any other repository. Publishing is a separate outward act that requires
jiangzhuo's explicit, per-act confirmation naming the target repository --
the same standing policy every other package in this family was published
under. `Q5_K_MIXED` is not a candidate for that confirmation on the evidence
above; the other three profiles are, pending that confirmation.
