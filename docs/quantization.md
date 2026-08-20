# Quantization Policy

Status: Confirmed 2026-08-20.
VITS F16 and Q8_MIXED version 1 functionally validated on 2026-07-23;
both profiles re-cut on 2026-07-27 with transpose-convolution weights held at F32.
Kokoro F16 and Q8_MIXED version 1 functionally validated on 2026-07-26.
OmniVoice profiles measured on 2026-08-09 and renamed the same day by
jiangzhuo's naming ruling: `F16` is the default recommendation and `Q8` the
smaller option — **both were published on 2026-08-09** — `Q4_K` and `BF16`
are measured but not published, and the two
codec-half profiles `F16_CODEC` and `Q8_CODEC_MIXED` are blocked on the
exact-token gate. See "OmniVoice Profiles," below. The Q4/Q5 precondition under
"VITS Q8_MIXED Profile" was settled explicitly on the same date, in the
paragraph that follows it.
Qwen3-TTS `qwen3-tts-12hz-1-7b-voicedesign`'s three shipped profiles (`BF16`,
`F16`, `Q8_MIXED`) were measured 2026-08-20 by Stage 3 Plan 3 and are
**prepared, not published** — `F16` ships on speed despite costing 283,136
bytes over its source, `Q8_MIXED` pays on both size and speed at this
variant's thinnest replay headroom of the three. A fourth profile,
`Q5_K_MIXED`, was measured the same day and fails its own `replay`-stage
tolerance roughly 3× over the committed bound — and **is published anyway**,
on jiangzhuo's explicit 2026-08-20 ruling after the blind listening audit,
with the gate failure disclosed on the card itself. That makes four shipped
profiles, and makes this the project's first profile published over a
failing numerical gate. This paragraph described `Q5_K_MIXED` as
"measured-and-not-shipped" until that ruling. See "Qwen3-TTS Profiles," below, and
`docs/porting/families/qwen3-tts.md` for the full record.

## Validation Sequence

1. **F32 reference** preserves the converted source weights and establishes tensor-by-tensor and waveform parity with the upstream implementation.
2. **F16 production baseline** is validated against F32 before it becomes the normal unquantized distribution artifact.
3. **Selective mixed quantization** is introduced only after the F16 baseline passes, one named Quantization Profile at a time.

F32, F16, and every mixed profile are separate Model Packages with independent validation results.

## Profile Semantics

A Quantization Profile records both storage type and required compute or accumulation precision by tensor group. A nominal profile name does not imply that every tensor uses the same type.

Normalization parameters, biases, scalar controls, and small tensors remain F32 or F16 by default. Alignment, duration, and speaker-conditioning tensor groups are treated as numerically sensitive and remain at high precision until targeted Port Validation Suite cases demonstrate stable structure at a lower precision. This establishes functional support, not perceptual-quality equivalence. Large supported matrix weights are the first quantization candidates.

The GGUF metadata identifies the profile name and version, source-model checksum, conversion-tool revision, per-tensor storage types, and any required compute-precision rules. Quantization must be deterministic for the same inputs and profile version.

## VITS F16 Profile

VITS `F16` version 1 keeps the complete text encoder, stochastic duration
predictor, speaker embedding, every bias, normalization parameter, and scalar in
F32, along with the decoder's four transpose-convolution weights. The remaining
acoustic-flow and decoder weights use F16.

Two separate reasons put tensors in that F32 group. The duration-decision domain
is there because an all-F16 duration path changed `ceil` decisions; keeping it
whole restores exact duration structure in all 12 port-validation cases for both
current VITS variants. The transpose-convolution weights are there for a
different reason, given in the Q8_MIXED section below.

Generate it from the source-F32 artifact:

```bash
build/bin/synthesize-quantize \
  models/vits-vctk/vits-vctk-F32.gguf \
  models/vits-vctk/vits-vctk-F16.gguf \
  --quant F16
```

## VITS Q8_MIXED Profile

`Q8_MIXED` version 1 is the first accepted mixed Quantization Profile. It is a
conservative storage profile, not a claim that every weight or operation is Q8.
For VITS version 1, the complete text encoder, stochastic duration predictor,
speaker embedding, biases, normalization parameters, and scalars remain F32.
The 44 acoustic-flow matrix weights and 75 ordinary decoder-convolution matrix
weights use `GGML_TYPE_Q8_0`. The decoder's four transpose-convolution weights
are F32. They were F16 until 2026-07-27, when the transposed convolution became
a column matrix multiply plus `ggml_col2im_1d`: CUDA's F16 matrix multiply
accumulates in half precision, where the operator it replaced accumulated in
F32. A package that stores them halved is refused with `SYNTH_ERR_GGUF`.

Ordinary Q8 convolution weights are stored as `[kernel * input_channels,
output_channels]` packed matrices instead of native `[kernel, input_channels,
output_channels]` tensors. This makes every Q8 row a valid multiple of GGML's
32-element Q8_0 block and lets `mul_mat` consume it directly. `im2col` uses a
private shape-only tensor; neither CPU nor CUDA dequantizes the stored weight into
a persistent F32 copy. The profile does not quantize activations. Reductions,
normalization, softmax, duration alignment, flow integration, and waveform
accumulation retain their existing F32 compute path.

```bash
build/bin/synthesize-quantize \
  models/vits-vctk/vits-vctk-F32.gguf \
  models/vits-vctk/vits-vctk-Q8_MIXED.gguf \
  --quant Q8_MIXED
```

The VCTK package contains 119 Q8_0 tensors and 354 F32 tensors. It is
60,372,192 bytes, 49.86% smaller than F32 and 24.09% smaller than the VITS F16
profile. The LJSpeech package contains 114 Q8_0 tensors and 346 F32 tensors. It
is 58,215,040 bytes, 48.59% smaller than F32 and 23.18% smaller than its F16
package. The additional 192 bytes in every current package are frontend
capability metadata.

Both figures are 5,324,800 bytes larger than the ones first recorded here. On
2026-07-27 the four transpose-convolution weights per package stopped being
halved: they now feed a column matrix multiply, whose CUDA F16 path accumulates
in half precision where the operator they replaced accumulated in F32, so
halving them would cost about 3e-3 relative on the decoder's output
(`ggml-patches/README.md`). No other tensor's profile assignment changed.

CPU and GB10 CUDA 13.3 execute both public synthesis paths on the re-cut
packages, with zero executable CPU fallback and 12/12 duration structures
preserved. RTX 4070 SUPER CUDA 13.3 ran the same paths for the packages cut on
2026-07-23; that host was not available for the re-cut, so it is not claimed
here.
The larger waveform drift is recorded, but no tolerance or perceptual-quality
acceptance claim is made.

The stored tensor assignment is identical for every copy of a Model Package and
does not vary by Execution Backend. Each backend claim is validated independently;
a backend may use a native Q8 kernel or a documented correct dequantization path,
but does not rewrite the GGUF or create a backend-specific profile. Its Model Page
records actual placement and performance rather than inferring acceleration from
the `Q8_MIXED` name.

No Q4 or Q5 mixed profile is committed until `Q8_MIXED` has been calibrated on a
real Model Variant and the additional profile has independent port-validation,
memory, and backend evidence. A later Quality Evaluation Suite may add perceptual
and comparative evidence without changing the stored profile identity.

**Settled 2026-08-09, because OmniVoice's `Q4_K` reached the question.** The rule
above is a rule about *shipping* a Q4 or Q5 profile, not about building or
measuring one, and it is not repealed. OmniVoice cut a `Q4_K` generator-half
profile, ran it through port validation, placement and the speaker proxy, and
**did not publish it** — so the rule was honored rather than bent. Two things
follow, and both are now explicit rather than left to be inferred:

- **Building and measuring an unshipped Q4 or Q5 profile is always allowed**, and
  is in fact how the precondition gets satisfied. A profile that exists in the
  tool's table and in a documented measurement, with no package cut, is not a
  committed profile.
- **The calibration precondition is per family, not global.** `Q8_MIXED` being
  calibrated on VITS does not license a Q4 profile on OmniVoice. OmniVoice's own
  `Q4_K` would have needed OmniVoice's own `Q8` calibrated first — which it now
  is — *and* independent port-validation, memory and backend evidence in its own
  right. It has that evidence, and the evidence is what rejected it: ten of
  seventeen renders degenerate, 62x the probe drift, no throughput gain. See
  "OmniVoice Profiles," below.

The one thing the rule does not cover, and OmniVoice supplied, is a *quality*
reason to decline a profile that passes every hard gate. That is not an
exception to this rule; it is a second, independent bar this project applies
after it.

## Kokoro F16 and Q8_MIXED Profiles

Kokoro version 1 keeps PL-BERT, the prosody and duration path, the acoustic text
encoder, and the Voice tables at the reference dtype, and quantizes the decoder
and its iSTFTNet generator — 60% of the parameters. The Snake activations'
alphas are divided by, and the depthwise upsampling pools are read one tap at a
time as a per-channel scale, so both stay at the reference dtype whatever the
profile.

That split is the opposite of the general rule stated above, where large matrix
weights are the first candidates wherever they sit, and it is specific to this
family's structure rather than a revision of the policy. Kokoro's excitation is
a sine whose phase accumulates across the whole utterance, so a relative
difference in F0 of a few parts in ten thousand becomes radians of phase by the
last syllable. Quantizing the front end was measured: it changed six of one
case's 78 predicted durations and lengthened the utterance by 50 ms, which
removes the structural exactness the suite checks for.

Twelve decoder matrices carry F16 rather than Q8_0 in the mixed profile because
their packed row is not a whole number of blocks; the decoder concatenates the
two prosody curves and a narrow encoder residual onto its feature stream, which
lands those channel counts two short of a multiple of thirty-two. The predicate
that decides this, and the classifier that assigns every tensor its role, are
shared by the offline tool and the runtime so the two cannot disagree about a
package.

```bash
build/bin/synthesize-quantize \
  models/kokoro-v1-0/kokoro-v1-0-F32.gguf \
  models/kokoro-v1-0/kokoro-v1-0-Q8_MIXED.gguf \
  --quant Q8_MIXED
```

## OmniVoice Profiles

OmniVoice is the first family here whose profiles split by model half. Every
other family's `Q8_MIXED` means "this family's large matrix weights, halved or
blocked"; this family has two halves with opposite risk, so a profile quantizes
exactly one of them. `classify_tensor_for_half` enforces that invariant — there
is no combined profile.

**The two halves, and why the split is not a convention.** The *generator* is a
bidirectional Qwen3-0.6B backbone that paints a fixed-length canvas of acoustic
tokens over 32 mask-predict denoising steps; the *codec* is Higgs Audio V2,
which turns a finished canvas into audio and, on the Reference Audio cloning
path, also runs in reverse to encode a reference clip into RVQ tokens. Each
half fails differently when you quantize it, and only one of those failures is
a defect:

- Quantize the **generator** and the canvas it paints changes — `Q8` commits a
  different token at 95.83% of positions. That is a *different valid
  realization* of the same request, not a wrong answer, because nothing
  downstream depends on which token was chosen, only on how many there are, and
  the canvas length is fixed by deterministic host arithmetic before the first
  forward runs. Token flip is therefore recorded as data and is never a gate
  here. What decides a generator profile is whether the voice changed, which a
  pitch-and-spectrum proxy screens for and a listener settles.
- Quantize the **codec** and the cloning path's RVQ encode changes, which *is* a
  defect. That encode is a nearest-neighbor lookup against a fixed codebook, so
  a perturbation of the fused latent that feeds it flips a discrete choice: the
  port's own guarantee is that the encode reproduces the oracle's 2,808-token
  grid byte-for-byte, and a profile that breaks it produces a clone of a
  different clip than the caller supplied. No perceptual argument substitutes
  for that gate, and none of the margins involved is a knife edge eligible for
  the dual-admissibility mechanism.

**So the generator half is the half that ships.** A generator-half profile
leaves all 486 `codec.*` tensors bit-identical to F32, which means the cloning
grid stays byte-exact and the profile inherits the family's hard guarantee
rather than re-earning it. Every codec-half profile produced so far fails that
same guarantee — 98 of 2,808 tokens at best, after a change that already cut
the drift by a factor of 10.4 — and is blocked. That is the whole reason the
shipping ladder is generator-shaped, and it is also why the naming rule below
lands the shipping profiles on the plain names.

**The half a profile quantizes determines its name** (jiangzhuo, 2026-08-09).
The generator half takes the plain name and the codec half takes a `_CODEC`
qualifier, which lands this family's shipping profiles on the same names the
three siblings already publish. The rule is about the artifact rather than its
ship status, so a codec profile becoming shippable later would not force another
rename. The `_GEN`-suffixed names this family used while the profiles were being
measured are retired and no longer resolve.

| profile | quantizes | bytes | off F32 | status as of 2026-08-09 |
| --- | --- | ---: | ---: | --- |
| F32 | nothing | 3,189,953,504 | — | reference, shipped |
| `F16` | generator only, halved | 1,964,929,440 | 38.4% | **default recommendation** |
| `Q8` | generator only, Q8_0 | 1,390,699,680 | 56.4% | **the smaller option** |
| `Q4_K` | generator only, Q4_K + pin | 1,166,300,320 | 63.4% | measured, **not published** — rejected on quality |
| `BF16` | generator only, bfloat16 | 1,964,929,440 | 38.4% | measured, **not published** |
| `F16_CODEC` | codec only, convolutions exempt | 2,858,422,240 | 10.4% | **BLOCKED** on the exact-token gate |
| `Q8_CODEC_MIXED` | codec only, convolutions exempt | 2,778,427,360 | 12.9% | **BLOCKED** on the exact-token gate |

Cut from the F32 artifact:

```bash
build/bin/synthesize-quantize \
  models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf \
  models/omnivoice-0-6b/omnivoice-0-6b-F16.gguf \
  --quant F16
build/bin/synthesize-quantize \
  models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf \
  models/omnivoice-0-6b/omnivoice-0-6b-Q8.gguf \
  --quant Q8
```

`--quant F16` reaches this family's generator half rather than the shared `F16`
policy the other three families get, because the tool resolves the half from
`general.architecture` on the omnivoice dispatch path. The two shipped packages
are `sha256 65c8cca5…a3f52f70` (F16) and `61aec0de…4982374e` (Q8);
`docs/models/omnivoice-0-6b.md` carries the full digests and the publication
set.

**No profile claims a speed benefit.** The generator is compute-bound at roughly
205 MAC per weight byte, so a narrower weight does not relieve the bottleneck.
Measured against F32 on CUDA: `Q8` −0.4%, `F16` **+2.7% slower**. On CPU: `Q8`
+1.5% slower, `F16` −1.2%; `BF16` is 10.5× slower. The honest claim for every
profile is size, memory and load time. Accelerator resident memory: F32 3506
MiB, `F16` 2353 MiB, `Q8` 1811 MiB. Load time: F32 2.6 s, `F16` 1.6 s, `Q8`
1.1–1.4 s.

**The generator is 76.9% of the tensor bytes** (2,450,309,120 of 3,184,565,636),
so no codec-only profile can go below about 2.62 GB. That single fact is why
this family grew generator profiles at all.

### `F16` — generator only, the default recommendation

All 199 two-dimensional generator weights at F16 — the two `ggml_get_rows`
tables among them, which need no pin because CUDA's `GET_ROWS` takes F16
directly. The package reads `599 F32 + 199 F16` in its header; the 599 are all
486 `codec.*` tensors plus the 113 generator norms, and every one of them is
**bit-identical** to the F32 package. Long-term average spectrum distance from
F32/CPU across all 17 greedy cases reaches a maximum of **2.87 dB**, every case
under the ~3 dB line the listener has consistently called "same person". It
preserves Reference Audio cloning (0.22 / 0.12 dB) and Description Text (0.00 /
1.38 dB).

### `Q8` — generator only, the smaller option

All 199 two-dimensional generator weights at Q8_0 (`599 F32 + 199 Q8_0` in the
header), same codec treatment as `F16` above: bit-identical to F32, so RVQ
encode stays byte-exact and the codec half inherits its validation exactly
rather than within a tolerance.

Two tables are pinned to Q8_0 rather than following the profile: CUDA's
`GET_ROWS` supports no k-quant, and `llm.embed_tokens.weight` and
`audio_embeddings.weight` are read by `ggml_get_rows`. Under `Q8` the pin is
free; under `Q4_K` it costs 78.06 MiB and is what keeps the whole graph off the
CPU.

**Nothing is disabled under `Q8` — all three voice modes work and the API is
identical.** It reproduces F32's voice for Reference Audio cloning (2.02 / 2.70
dB). For auto-voice and Description Text it returns a *different voice*: LTAS
4.12–12.22 dB, and a listener heard five of six sampled auto-voice pairs as
different people **while judging quality indistinguishable on every one of
them**.

`Q8` still honors a Description Text prompt. The design cases specify pitch
explicitly and it is checkable: "female, young adult, high pitch" gives F32
333.3 Hz and `Q8` 343.5 Hz; "男，老年，低音調" (male, elderly, low pitch) gives
152.9 and 158.4 Hz, against ordinary voices in this suite at 118–130 Hz. The
8.61 dB on `omni-design-en` means "a different voice within the same
description", **not** "the description was ignored". `Q8` is neither clone-only
nor a profile that breaks voice design.

For auto-voice the caveat is barely news: this family's docs already state the
speaker is unstable across backends, seeds and step counts, because auto-voice
carries no speaker conditioning at all.

### `Q4_K` — measured, not published

197 generator weights at Q4_K, the two lookup tables pinned Q8_0. It passes every
hard gate, its placement proof is clean, and the pin is demonstrated necessary
rather than predicted. It is not published anyway: over `Q8` it buys 224 MB and
**zero throughput**, while ten of seventeen renders go degenerate, at 62× the
probe drift.

Recorded because it generalizes: **the speaker-identity proxy fails in the
dangerous direction here.** A render degraded until the pitch tracker finds no
voiced frame is scored *not comparable* rather than *changed*, so the proxy's
headline reads clean on a profile the degeneracy screen says is broken. A pitch
instrument answers "different speaker"; it cannot answer "not a voice". Both
must run.

### `BF16` — measured, not published

The same generator half as `F16` at bfloat16, and the same file size to the
byte. Not published: 11 of 17 cases land over the quality line, and it is 10.5×
slower on CPU. The row and the code stay — the format has native CUDA MUL_MAT
and GET_ROWS paths and the measurement is worth keeping — but no package ships.

Recorded because it generalizes: **the speaker-identity proxy fails in the
dangerous direction here.** A render degraded until the pitch tracker finds no
voiced frame is scored *not comparable* rather than *changed*, so the proxy's
headline reads clean on a profile the degeneracy screen says is broken. A pitch
instrument answers "different speaker"; it cannot answer "not a voice". Both
must run.

### `F16_CODEC` and `Q8_CODEC_MIXED` — codec only, convolutions exempt

Both carry the `_CODEC` qualifier because they quantize the codec half; before
the 2026-08-09 naming ruling they were spelled `F16` and `Q8_MIXED`, and the
plain `F16` now means the generator-half profile above. The codec-half
definition was itself redefined on 2026-08-09 rather than renamed, both profiles
being blocked and unpublished at the time.

This family **never block-quantizes a convolution kernel**: a fifth
`QuantRole`, `ConvKernel`, holds all 85 codec convolutions at the profile's
halved fallback at native three-axis shape, and the only codec tensors a profile
still quantizes are 73 HuBERT Linears whitelisted positively by module name. The
rule reads the name and never the rank — `codec.acoustic_decoder.conv2.weight`
is `[7, 32, 1]`, which `ggml_n_dims` collapses into something indistinguishable
from a Linear.

That took clone-path RVQ drift from 1,023 of 2,808 positions to 98, a factor of
10.4, and attribution showed all of the remainder is convolution precision:
quantizing the 73 Linears to Q8_0 costs nothing measurable (98 against
`F16_CODEC`'s 103) while saving 80 MB, whereas convolutions at F32 instead of
F16 reach 19 of 2,808 for 161 MB more. Both profiles are still **blocked** —
`ref.tokens` is 0/2 exact, the one gate with no "different valid realization"
defence.

Consequently this family deliberately does not use the packed-convolution branch
this project built for it. That code stays, shared with three other families and
covered by tests that build packed kernels directly; its presence is not
evidence that OmniVoice packs convolutions.

## Qwen3-TTS Profiles

Added 2026-08-17 by Stage 2 Plan 4. This family shipped `F16`, `Q8_MIXED` and a
buildable-but-unpublished `Q5_K_MIXED` from Stage 1 onward with **no section in
this document at all** — every profile fact lived only in
`docs/porting/families/qwen3-tts.md`. What follows covers all three Reference
Model Variants (Base, CustomVoice and, since 2026-08-20, VoiceDesign) and says
which claims are measured on which.

### The half that quantizes, and the half that never does

The package has two halves that behave completely differently, and the split is
by name in both the runtime and the quantizer — `src/arch/qwen3-tts/catalog.cpp`
tests a `codec.` prefix and `tools/synthesize-quantize/policy.cpp` classifies the
same way, deliberately, so the two cannot drift.

- **The autoregressive half** — talker and code predictor, 316 + 86 tensors,
  1727.6 MiB of 2164 — is entirely two-dimensional matrices with rows of 1024,
  2048 and 3072. It block-quantizes without argument, and it is where both the
  parameters and the win are.
- **The codec half never halves, under any profile**, and that is a measurement
  rather than caution: halving it made the codec **1.75× slower on CPU**, 4.2 s
  to 7.3 s on a 37-frame case, because its convolutions run through im2col into a
  matrix multiply where ggml's F16 path is slower than its F32 one.

`Q8_MIXED` here means what `docs/quantization.md`'s general rule says at the top:
a profile records storage type and required precision **by tensor group**, and a
nominal name never implies every tensor uses that type. Concretely the codebooks,
both kernel-1 projections, per-head norms, per-branch layer scales and SnakeBeta
curves stay exact, and six transposed convolutions stay F32 because CUDA's F16
matrix multiply accumulates in half precision.

### The Base variant's two extra regions

`qwen3-tts-12hz-0-6b-base` carries 894 tensors against CustomVoice's 657. The
237-tensor difference is 76 `speaker_encoder.*` and 161 `codec.encoder.*`, and
until Plan 4 Task 6 **the quantizer could not cut a Base package at all** — both
regions classified `Unknown` and the tool stopped on the first one it met.

- **`codec.encoder.*` (161)** are F32 under every profile, which is what the
  `codec.` prefix already said. The tool and the runtime were never in
  disagreement here; what was missing was recognition.
- **`speaker_encoder.*` convolution weights (38)** take a `ConvKernel` role: the
  profile's halved fallback (**F16** under `F16`, `Q8_MIXED` and `Q5_K_MIXED`) at
  native three-axis shape, never block-quantized and never packed. Their 38
  biases are `Sensitive`.

That last one is arithmetic, not caution. A block runs along `ne[0]`, which for a
convolution kernel is the **kernel extent** — 1, 3 and 5 across all 38 — against
Q8_0's block of 32. The packed `[kernel × in, out]` layout would clear the block
size, but it emits rank 2, and this family's runtime implements neither half of
consuming that: `Resolver::find` has no packed branch (Kokoro's does) and
`same_conv1d` reads `ne[0..2]` as `{kernel, in, out}`.

The role is shaped like OmniVoice's `ConvKernel` and reaches the same column for
a **different, family-local reason**. Neither is imported by the other.

### Does quantizing the speaker encoder pay? No, and there is nothing to gain

Measured 2026-08-17. The 38 weights are 8,843,264 elements, 0.703 % of the
package:

| profile | storage | bytes | saved |
| --- | --- | ---: | ---: |
| BF16 (source) | BF16 | 16.87 MiB | — |
| F16 | F16 | 16.87 MiB | **zero** — both are two-byte types |
| Q8_MIXED | F16 | 16.87 MiB | **zero** — held at the halved fallback |
| *Q8_0 packed (hypothetical)* | *Q8_0* | *8.96 MiB* | *7.91 MiB, and unreachable* |

Accuracy costs nothing: worst x-vector cosine against the oracle is 0.99999467 at
BF16 and **0.99999501** at both F16 and Q8_MIXED, which are byte-identical to each
other. So the design's "unlikely to pay" is confirmed, for a stronger reason than
it gave — the speaker encoder does not shrink under any profile this family has.

### What each profile is for

| profile | package | size | RTF (CPU) | peak RSS | verdict |
| --- | --- | ---: | ---: | ---: | --- |
| BF16 | Base | 2,516,522,464 B | 3.15 | 3.29 GiB | the source |
| F16 | Base | 2,516,706,912 B | not measured | — | **clears every gate and does not pay** |
| Q8_MIXED | Base | 1,667,606,112 B | **0.863** | 2.23 GiB | **pays on both size and speed** |
| Q5_K_MIXED | CustomVoice | 1035 MiB | 0.82 | — | buildable, deliberately unpublished |
| BF16 | VoiceDesign | 4,295,891,904 B | 4.64 | 4.95 GiB | the source, prepared, not published |
| F16 | VoiceDesign | 4,296,175,040 B | **1.65** | 4.95 GiB | **pays on speed, prepared, not published** — roughly 2.8x faster on the measured host, larger than source |
| Q8_MIXED | VoiceDesign | 2,499,423,680 B | **1.05** | 3.16 GiB | **pays on both size and speed, prepared, not published** — thinnest replay headroom of the three (1.63x) |
| Q5_K_MIXED | VoiceDesign | 1,780,723,136 B | 1.08 (context only) | 2.40 GiB | **fails replay tolerance (headroom 0.32x) — PUBLISHED ANYWAY on jiangzhuo's 2026-08-20 ruling, breach disclosed on the card, see below** |

**F16 is 184,448 bytes LARGER than the package it was cut from.** Both BF16 and
F16 are two-byte types, so the matrix weights do not shrink while the sensitive
tensors widen BF16 → F32. Stage 1 measured the same shape on CustomVoice
(2274.3 MB against a 2274.1 MB source), which is why F16 is a **speed** profile
for this family rather than a size one.

**`Q8_MIXED` crosses real time on the Base variant**: RTF 3.15 → 0.863 on
`rel-dgx-spark` at `CMAKE_BUILD_TYPE=Release`, a 3.65× improvement, with 33.7 %
less package and 1.06 GiB less peak RSS.

`Q5_K_MIXED` is the in-tree precedent for a profile that is buildable and
deliberately unpublished — 1035 MiB and RTF 0.82 on CustomVoice, but talker
logits cosine 0.9648.

**`F16` ships on speed for the VoiceDesign variant, not size.** Cut and
measured by Task 5 (2026-08-20, Stage 3 Plan 3): it is 283,136 bytes larger
than its BF16 source, but measures 2.87× faster on the aarch64/GB10 host it
was measured on, against Task 5's own BF16 pass (RTF 4.73 → 1.65,
`rel-dgx-spark`) — the pairing that produced Task 5's ship recommendation.
Task 6 later re-measured BF16 in the same session as its own CUDA run (n=8,
RTF 4.64, the figure this table's own row above carries); pairing that
later BF16 figure with F16's unchanged 1.65 instead gives roughly 2.8×, a
recomputation Task 8 made rather than either task's own reported ratio —
both figures are real, from different BF16 passes. Either way F16 recovers
most of what quantizing all the way to `Q8_MIXED` buys while changing
nothing about the package's on-disk footprint. The mechanism is the same
host fact Base's own F16 pattern already suggests — `ggml`'s CPU backend
has a NEON-vectorized GEMM path for F16 and none for BF16 on this host —
measured directly for the first time on this variant by Task 5, rather than
inferred. See `docs/porting/families/qwen3-tts.md`'s "Stage 3: VoiceDesign
Package, Plan 3 Task 5" for the full arithmetic and the mechanism trace, and
"...Task 6" for the later BF16 pass.

`Q8_MIXED` crosses real time on the VoiceDesign variant too: RTF 4.64 →
1.05, with 41.82 % less package — proportionally more shrink than Base's own
Q8_MIXED (58.18 % of source against Base's 66.3 %) — and the thinnest
replay-prefill headroom of this variant's three shipped profiles, 1.63×
against BF16's 3.44× and F16's 3.45×, still comfortably clearing the 0.01
bound.

### The codec encoder is byte-identical across all three profiles

Measured, not assumed: the port's codec-encoder output was compared byte for byte
across BF16, F16 and Q8_MIXED over five cases — 20 of 20 and 25 of 25 sha256
comparisons match, covering latents, the RVQ reconstruction, the codes, the
downsample tap and the tie margin. **Zero code flips at any profile.**

This is why OmniVoice's blocking reason does not transfer. Its codec-half
profiles are blocked because quantization flipped clone-path RVQ tokens, and this
family's codec encoder ends in the same nearest-neighbour argmin — but it never
quantizes that half, so the argmin sees the same F32 weights and returns the same
codes. **That is a measured answer, not an inheritance of the existing rule.** It
does not say a quantized codec encoder would be safe here; nothing measured one,
because none exists.

### VoiceDesign's Q5_K_MIXED — the first real evaluation this profile has had, and it fails

Measured 2026-08-20, Stage 3 Plan 3 Task 4. Until Stage 3 Plan 3 Task 8 filled
"What each profile is for," above, neither this section nor
`qwen3-tts-12hz-1-7b-voicedesign`'s F16/Q8_MIXED cuts were documented here at
all — those numbers lived only in `docs/porting/families/qwen3-tts.md` and
`tests/tolerances/qwen3-tts.json`. What follows is scoped to the question this
task exists to answer:
whether `Q5_K_MIXED`'s extra shrink over `Q8_MIXED` changes the
buildable-but-unpublished verdict `Q5_K_MIXED` already carries elsewhere in
this family, now that the source package is roughly twice CustomVoice's size.

| profile | size (bytes) | vs BF16 | vs Q8_MIXED |
| --- | ---: | ---: | ---: |
| BF16 (source) | 4,295,891,904 | — | — |
| Q8_MIXED | 2,499,423,680 | 58.18 % (41.82 % smaller) | — |
| Q5_K_MIXED | 1,780,723,136 | 41.45 % (58.55 % smaller) | **71.25 % (28.75 % smaller)** |

**The number to lead with is the last column.** `Q5_K_MIXED` saves a further
718,700,544 bytes over `Q8_MIXED` — 28.75 %, close to but not exactly the
design spec's own "~25 %" estimate. At Base's/CustomVoice's ~2.4 GB source this
same extra shrink did not change the recommendation to leave `Q5_K_MIXED`
unpublished; at this variant's 4.3 GB source, 718.7 MB is a much larger
absolute number, and the open question this task was written to close was
whether that changes the answer.

**It does not, but not for the reason the size column would suggest.** The
`replay` stage's talker-prefill probe (`tests/qwen3_tts_voicedesign_prefill_real.cpp`,
the same two cases and the same 0.01 `max_relative` bound every other profile
here reuses) measures p95_relative **0.031029** (empty-instruct) and
**0.030366** (non-empty-instruct) — both OVER the bound, by 3.10× and 3.04×
respectively, not under it with room to spare. This is a real gate failure, not
merely a thinner margin: the variant's own headroom progression across its
other three profiles is BF16 3.44×, F16 3.45×, Q8_MIXED 1.63× — all
comfortably above 1.0 (BF16 and F16 are within measurement noise of each
other, not a monotonic sequence; Q8_MIXED is the one visibly thinner of the
three, for the reason given above). `Q5_K_MIXED` breaks that progression
outright at **0.32×**
(0.01 / 0.031029), roughly a further 5.06× jump in residual over Q8_MIXED's
own 0.00613 (itself already the thinnest of the three), and about 11.5× BF16's
own 0.00269 — a much larger jump than the F16 → Q8_MIXED step was, not a
continuation of the same slope.

> **SUPERSEDED 2026-08-20 — this profile IS published.** jiangzhuo ruled the
> same day, after the blind listening audit below found `Q5_K_MIXED`
> indistinguishable from `BF16` (one clip, one sentence, one seed, one
> listener), that it ships. The recommendation immediately following is kept
> as history, not rewritten, under this document's own convention of
> superseding rather than deleting: it was the correct reading of the
> measurement, and the measurement has not changed. `gate_passed` is still
> `false`, the observed 0.031029/0.030366 still breach the committed 0.01
> bound, and headroom is still 0.32x. What changed is a publication
> decision taken with the breach in view, and it is disclosed on the
> published card itself — `scripts/hf_cards/qwen3-tts-12hz-1-7b-voicedesign.yaml`
> states the gate failure and the ruling together in the Downloads section.
> **This is the project's first profile published over a failing numerical
> gate**, and it does not establish that a 3x breach of this probe is
> generally acceptable.

**Recommendation (2026-08-20 Task 4, SUPERSEDED the same day by the ruling
above — retained as history): do not publish `Q5_K_MIXED` for
`qwen3-tts-12hz-1-7b-voicedesign`.**
The precedent this result matches is CustomVoice's own `Q5_K_MIXED` (talker-logits
cosine 0.9648, deliberately unpublished on accuracy, see "What each profile is
for" above) — but this is the stronger of the two findings: CustomVoice's number
was a lower cosine on a probe with no committed pass/fail bound; this is an
explicit breach of a committed `max_relative` gate, on both of this variant's two
measured `replay` cases, reproduced identically on a repeat run. Task 8's card
should record `Q5_K_MIXED` as buildable and load-checked (`synth_model_load` and
`synth_voice_profile_create_from_description` both succeed against the cut
package) but not shipped for this variant, on accuracy. **This is NOT
"exactly as CustomVoice's own card already does"** — `scripts/hf_cards/
qwen3-tts-12hz-0-6b-customvoice.yaml:176-209`'s `quants:` block ships `BF16`,
`F16` and `Q8_MIXED`; CustomVoice's `Q5_K_MIXED` is **omitted** from its card, not named as
measured-and-not-shipped, which is itself a departure from this plan's own
rule that a profile failing its own test "belongs in the card as
measured-and-not-shipped ... rather than omitted." Task 8 has no existing
template to copy for this shape and must establish it, for both packages;
CustomVoice's card gap is a separate, pre-existing item, not something this
task fixes (see `docs/porting/families/qwen3-tts.md`'s "Open item:
CustomVoice's card omits Q5_K_MIXED" for the tracked gap).

> **The card shape this paragraph asked for was never established, because
> the ruling overtook it (2026-08-20).** The VoiceDesign card does not record
> `Q5_K_MIXED` as measured-and-not-shipped; it SHIPS the profile with the
> gate failure disclosed. So this card establishes a
> published-despite-gate-failure shape, not the measured-and-not-shipped
> shape. CustomVoice's omission remains an open gap either way — but the
> open item can no longer point at this card for the shape it should copy,
> because this card no longer has an unshipped profile to model. Whoever
> closes that gap must decide the shape for CustomVoice on its own evidence.

The size question
this task was written to answer — whether a further 28.75 % shrink over
`Q8_MIXED` matters more at 4.3 GB than an equivalent shrink did at 2.4 GB —
is answered "yes, noticeably more," but that answer is moot once the
accuracy side has moved even further than the size side did.

**Nothing mechanical currently enforces this recommendation.** No registered
test reads `tests/tolerances/qwen3-tts.json`'s `headroom`,
`observed_max_relative`, `all_passed`, `failed_checks` or `gate_passed`
fields — `tests/CMakeLists.txt`'s prefill CTest gates hardcode `profiles
BF16` and the public-seam CTest gate runs `--profile BF16` against the
CustomVoice model, neither reads this cell at all. F16's and Q8_MIXED's
cells for this variant are equally unread, so this is not new to
`Q5_K_MIXED` — but it does mean a future cut-and-ship of this profile would
not be caught by CTest; only this document and the tolerance file's own
prose stand between the measurement and a mistaken publication.

**Recording the failing cell at all is a departure from both of this
family's own precedents, made deliberately.** CustomVoice's `Q5_K_MIXED`
negative lives only in prose (this document, `docs/porting/families/
qwen3-tts.md`) with no tolerance cell behind it at all; the Base variant's
declined-CUDA negative (see "The Base variant carries no CUDA sub-grid"
in `docs/porting/families/qwen3-tts.md`) is a deliberate *absence* of a
cell, not a committed one that reads false. This task commits the cell
anyway, `gate_passed: false` and all, because an absent or prose-only
record of a failure is strictly weaker evidence than a reproducible,
machine-readable one sitting in the same grid a passing profile would
occupy — a future reader (or a script written later) can find this result
by looking at the grid rather than needing to already know to look for it
in prose.

**Addendum, 2026-08-20, Stage 3 Plan 3 Task 7.** A blind listening pass
included this exact profile, unlabelled, beside `BF16` — and could not tell
them apart on one clip, one seed, one listener. That is a data point about
this 0.01 bound's conservatism at this margin, not a recalibration of it.
This addendum read "the do-not-publish recommendation above stands, and
whether the audit outcome changes it is jiangzhuo's call, not settled by the
audit itself" until later the same day, when **jiangzhuo made that call and
ruled the profile ships** — see the superseding note above the
recommendation. The audit remains what it was; the decision taken on it is
what is new, and the gate failure is disclosed on the published card. See
`docs/porting/families/qwen3-tts.md`'s "Stage 3: VoiceDesign Package, Plan 3
Task 7" for the full audit record, including the separate labelled
description-control half.

### Publication

CustomVoice's `BF16`, `F16` and `Q8_MIXED` are published. **The Base
variant's `BF16`, `F16` and `Q8_MIXED` were published 2026-08-17** to
`jiangzhuo9357/qwen3-tts-12hz-0-6b-base-gguf` — this line read "No Base
package is published" from Stage 2 Plan 4 until Stage 3 Plan 3 Task 8
corrected it, three days stale. **No `qwen3-tts-12hz-1-7b-voicedesign`
package is published, as of this writing.** Stage 3 Plan 3 measured and
prepared **four** shipped profiles (`BF16`, `F16`, `Q8_MIXED` and
`Q5_K_MIXED`; card spec at
`scripts/hf_cards/qwen3-tts-12hz-1-7b-voicedesign.yaml`, model page at
`docs/models/qwen3-tts-12hz-1-7b-voicedesign.md`). This paragraph said
`Q5_K_MIXED` "is not a publication candidate at all" until jiangzhuo's
2026-08-20 ruling put it on the roster despite its `replay`-stage gate
failure; that failure stands as measured and is disclosed on the card.
Publication of all four is a separate outward act requiring
jiangzhuo's explicit, per-act confirmation naming the target repository —
the ruling settled which profiles the card describes, not whether to upload,
and nothing described in this section has been uploaded.

## Validation

Each Quantization Profile must pass the Port Validation Suite independently for
every claimed Model Variant and Execution Backend. Validation covers intermediate-
tensor drift, deterministic waveform regression within declared tolerances, finite
and correctly shaped output, control and Voice-conditioning paths, repeatability,
resource cleanup, actual backend placement, latency, real-time factor, and peak
memory.

Port validation does not claim intelligibility, naturalness, Voice similarity, or
acceptable perceptual drift. Those claims require the later Quality Evaluation
Suite and the `quality_evaluated` Validation Level.

Support does not inherit from the F32 or F16 package, from another model variant, or from another backend. A runtime rejects tensor types or profile requirements that its selected backend cannot execute correctly.
