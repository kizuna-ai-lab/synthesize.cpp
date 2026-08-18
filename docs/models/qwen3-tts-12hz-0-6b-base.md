# Qwen3-TTS 12 Hz 0.6B Base

Status: Confirmed 2026-08-18. **Published 2026-08-17** to
[`jiangzhuo9357/qwen3-tts-12hz-0-6b-base-gguf`](https://huggingface.co/jiangzhuo9357/qwen3-tts-12hz-0-6b-base-gguf)
carrying BF16, F16 and Q8_MIXED, on jiangzhuo's per-act confirmation naming that
exact target. Every later change to the packages or the card is its own act and
needs its own confirmation.

This is the family's **first** `docs/models/` page, added by Stage 2 Plan 4 Task
16. `qwen3-tts-12hz-0-6b-customvoice` is published and has none; that gap is
noted rather than filled here, because this page records what Plan 4 measured.

`Q8_MIXED` is the profile that pays: 33.7 % smaller than the source, RTF 0.863
against BF16's 3.15 -- **faster than real time** -- and 1.06 GiB less peak RSS,
with no measured accuracy cost on any gated quantity. `F16` clears every gate and
does **not** pay: it is 184,448 bytes *larger* than the package it was cut from.

**The first ICL Listening Audit ran on 2026-08-17 and recorded
`no_obvious_regression`** -- five blind pairs and two labelled resemblance
checks, one listener. Both quantization profiles were audible but **not
degraded**, CPU and CUDA were indistinguishable, and both clones were judged the
**same speaker** as their source. See "Listening Audit," below. Quality
Evaluation stays deferred per ADR 0017, and neither claim moves the Validation
Level.

## What this variant is for

Stage 2 of the Reference Model Variant Ladder. It adds
`SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO` and
`SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE`: reference audio in, cloned audio out,
in either x-vector mode or transcript-assisted (ICL) mode.

**It catalogues no Preset Voices** -- `preset_ids` is an empty array -- which is
the structural difference from CustomVoice and the reason a caller must supply a
Voice Profile. A tool written around a Preset Voice catalogue will not drive this
package at all; `scripts/validate-qwen3-tts-public.py` had to learn the other
kind of Voice before it could measure this variant's public seam.

## Package

894 tensors against CustomVoice's 657. The 237-tensor difference is exactly 76
`speaker_encoder.*` (an ECAPA-TDNN speaker encoder) and 161 `codec.encoder.*`
(the speech tokenizer's encoder half).

| profile | bytes | tensor types | RTF (CPU) | peak RSS | verdict |
| --- | ---: | --- | ---: | ---: | --- |
| BF16 | 2,516,522,464 | 478 BF16 + 416 F32 | 3.15 | 3.29 GiB | the source |
| F16 | 2,516,706,912 | 304 F16 + 590 F32 | not measured | — | clears the gates, does not pay |
| Q8_MIXED | 1,667,606,112 | 266 Q8_0 + 38 F16 + 590 F32 | **0.863** | 2.23 GiB | **pays on size and speed** |

Digests: BF16 `993f4cd1…999f`, F16 `e7484194…a046`, Q8_MIXED `808667ae…a789`.

The 38 F16 tensors in the `Q8_MIXED` census are exactly the speaker encoder's
convolution weights, held at the profile's halved fallback by the `ConvKernel`
role -- see `docs/quantization.md`'s Qwen3-TTS section for why a block cannot run
along a kernel axis of 1, 3 or 5.

**F16 being larger than its source is not a defect.** BF16 and F16 are both
two-byte types, so the matrix weights do not shrink while the sensitive tensors
widen BF16 → F32. Stage 1 measured the same shape on CustomVoice. F16 is a speed
profile for this family, not a size one.

## Measured

All figures from `build/rel-dgx-spark` (`CMAKE_BUILD_TYPE=Release`, sm_121a, CUDA
13.3, DGX Spark/GB10) unless stated. One sentence through an ICL reference-audio
Voice Profile, seed 7, 10 threads.

| stage | BF16/CPU | F16/CPU | Q8_MIXED/CPU |
| --- | --- | --- | --- |
| `public` | 7 checks pass, 3 skipped | same | same |
| `replay` (`speaker.x_vector`) | min cosine 0.99999467 | 0.99999501 | 0.99999501 |
| `codec_encoder` | gated | **byte-identical to BF16** | **byte-identical to BF16** |

The three structural skips in `public` are the same everywhere: a Voice Profile
has no id for the seam to echo, and the two dialect-speaker checks need a Preset
Voice this package does not have.

**The codec encoder is byte-identical across all three profiles** -- 20 of 20 and
25 of 25 sha256 comparisons over five cases. Its F16 and Q8_MIXED cells are
therefore not independent evidence, and they say so.

## Backends

**CUDA carries no sub-grid for this variant, on a measurement.** The two new
graphs stay on the CPU: a device twin would mirror 241 MB for about 3.7 s of
extra load against the graphs' entire CPU cost of about 1.69 s, so the transfer
is 2.2× the compute it would accelerate and no speedup makes it pay. They also
run once per Voice Profile rather than once per synthesis, so the amortization
that justified Stage 1's codec-decoder twin runs the other way.

Only `public` measures anything different on CUDA, through that Stage 1 twin,
which does move on a Base package: 7 checks pass, and the audio differs from CPU
as the codec decoder's TF32 arithmetic predicts. CUDA buys about 6 % end to end
(RTF 3.15 → 2.95, and 11.60 s → 10.85 s on the same tree). `docs/backends.md` requires performance measurement for
support but no minimum speedup, so CUDA is not described as accelerated here
beyond what that number says.

## Listening Audit

**Ran 2026-08-17. Result: `no_obvious_regression`.** The family's **first ICL
audit** -- the 2026-08-13 one covered x-vector mode only, and said so, because
ICL did not exist when it ran.

| # | comparison | verdict |
| --- | --- | --- |
| 1 | BF16 against F16 | different, **neither degraded** |
| 2 | BF16 against Q8_MIXED | different, **neither degraded** |
| 3 | CUDA against CPU | indistinguishable |
| 4 | ICL against x-vector, reference A | indistinguishable |
| 5 | ICL against x-vector, reference B | indistinguishable |
| L1 | reference A source against its clone | **same speaker** |
| L2 | reference B source against its clone | **same speaker** |

Five blind pairs with A/B shuffled on a recorded seed, then two labelled
resemblance checks. **Every pair was confirmed to differ in bytes before the
verdict was recorded**, so each "indistinguishable" is a listening judgement
rather than a trivial truth.

The row that matters for shipping is pair 2: `Q8_MIXED` being 33.7 % smaller at
RTF 0.863 is only worth having if it still sounds right, and no tolerance table
can say whether it does.

Pairs 4 and 5 are **scoped, not a verdict on ICL**. Two clips, one sentence, one
listener; ICL's machinery may matter on material these references do not
represent. What they say is that on these clips the extra path costs nothing
audible and buys nothing audible either.

One listener and two recordings. This is evidence, not a level.

## Coverage this variant does not have

- **One additional speaker, not a corpus.** Four of the five codec cases are the
  same recording at three lengths; the fifth is a second speaker added by Plan 4
  Task 1. The x-vector residual rests on three clip lengths of one recording.
- **No autoregressive replay probes.** `replay` carries `speaker.x_vector` and
  `prompt.icl_embed`; the deep talker and audio probes CustomVoice's `replay`
  carries are unmeasured here.
- **`prompt.icl_embed` is measured for BF16 only.** The F16 and Q8_MIXED cells
  record it as not measured rather than copying a figure.
- **No CLI coverage.** `examples/cli/` is untouched by design.

## Licensing

Same as the family: `QwenLM/Qwen3-TTS` and the `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
weights both carry an explicit Apache-2.0 grant, verified from the upstream model
card rather than from a port's README.

One reference recording used in the Golden suite has **no licence stated
upstream**: `seedtts_ref_en_1.wav`, from the Seed-TTS eval set, whose repository
carries no `LICENSE` file. This repository already pinned and consumed those
exact bytes for the OmniVoice family before Plan 4 reused them. The gap is
recorded rather than resolved -- see "The second reference recording" in
`docs/porting/families/qwen3-tts.md`.
