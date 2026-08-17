# Qwen3-TTS 12 Hz 0.6B Base

Status: Confirmed 2026-08-17. **Nothing here is published.** This variant's
packages exist locally and the ship artifacts are staged; publication is a
separate act requiring jiangzhuo's confirmation at the time.

This is the family's **first** `docs/models/` page, added by Stage 2 Plan 4 Task
16. `qwen3-tts-12hz-0-6b-customvoice` is published and has none; that gap is
noted rather than filled here, because this page records what Plan 4 measured.

`Q8_MIXED` is the profile that pays: 33.7 % smaller than the source, RTF 0.863
against BF16's 3.15 -- **faster than real time** -- and 1.06 GiB less peak RSS,
with no measured accuracy cost on any gated quantity. `F16` clears every gate and
does **not** pay: it is 184,448 bytes *larger* than the package it was cut from.

**The Listening Audit has not returned a verdict.** The material is built and
offered (see "Listening Audit," below); until a listener reports, `spec:532`'s
"audit recorded" gate is **not met** and this page does not claim it is. Quality
Evaluation stays deferred per ADR 0017. Neither claim would move the Validation
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
as the codec decoder's TF32 arithmetic predicts. CUDA buys about 8 % end to end
(RTF 3.15 → 2.95). `docs/backends.md` requires performance measurement for
support but no minimum speedup, so CUDA is not described as accelerated here
beyond what that number says.

## Listening Audit

**Offered 2026-08-17; no verdict yet.** `build/listening-icl/audit.html` carries
five blind pairs -- BF16 against F16, BF16 against Q8_MIXED, CPU against CUDA,
and ICL against x-vector on each of two reference recordings -- plus two labelled
source-against-clone resemblance checks. A/B positions are shuffled with a
recorded seed and the key is in `build/listening-icl/manifest.json`, not on the
page.

This is the family's **first ICL audit**. The 2026-08-13 audit covered x-vector
mode only and recorded `no_obvious_regression`; it explicitly did not cover ICL,
which did not exist when it ran.

Until a listener reports, no claim is made about how any of this sounds. A
recorded result is `no_obvious_regression` or a named regression -- never an
inference from a passing tolerance table.

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
