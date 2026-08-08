# Quantization Policy

Status: VITS F16 and Q8_MIXED version 1 functionally validated on 2026-07-23;
both profiles re-cut on 2026-07-27 with transpose-convolution weights held at F32.
Kokoro F16 and Q8_MIXED version 1 functionally validated on 2026-07-26.
OmniVoice profiles measured on 2026-08-09 and **none shipped**: `Q8_GEN` awaits a
Listening Audit, `Q4_K_GEN` is rejected on quality, and `Q8_MIXED` is blocked on
the exact-token gate. See "OmniVoice Profiles," below.

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

OmniVoice is the first family here to need **family-specific profile names**,
and the first whose profiles split by model half. Every other family's
`Q8_MIXED` means "this family's large matrix weights, halved or blocked"; this
family has two halves with opposite risk, so a profile quantizes exactly one of
them and the name says which. `classify_tensor_for_half` enforces that
invariant — there is no combined profile.

| profile | quantizes | bytes | off F32 | status as of 2026-08-09 |
| --- | --- | ---: | ---: | --- |
| F32 | nothing | 3,189,953,504 | — | reference, shipped |
| F16 | everything halved | 2,858,422,240 | 10.4% | measured |
| `Q8_MIXED` | codec only, convolutions exempt | 2,778,427,360 | 12.9% | **BLOCKED** on the exact-token gate |
| `Q8_GEN` | generator only, Q8_0 | 1,390,699,680 | 56.4% | **measured, not shipped** — awaits a Listening Audit |
| `Q4_K_GEN` | generator only, Q4_K + pin | 1,166,300,320 | 63.4% | **REJECTED on quality** |

**The generator is 76.9% of the tensor bytes** (2,450,309,120 of 3,184,565,636),
so no codec-only profile can go below about 2.62 GB. That single fact is why
this family grew generator profiles at all.

### `Q8_GEN` — generator only

199 two-dimensional generator weights at Q8_0; all 486 `codec.*` tensors and the
113 generator norms stay F32 and are **bit-identical** to the F32 package. The
codec half therefore inherits its validation exactly rather than within a
tolerance, and RVQ encode stays byte-exact — the argument for cutting this
profile before any codec profile.

Two tables are pinned to Q8_0 rather than following the profile: CUDA's
`GET_ROWS` supports no k-quant, and `llm.embed_tokens.weight` and
`audio_embeddings.weight` are read by `ggml_get_rows`. Under `Q8_GEN` the pin is
free; under `Q4_K_GEN` it costs 78.06 MiB and is what keeps the whole graph off
the CPU.

It buys size, memory and load time, and **no throughput**: 1.2% on CUDA and
nothing on CPU, both inside run-to-run spread. Accelerator twin 2,336.80 MiB →
620.90 MiB; peak CUDA RSS 4,019 MiB → 1,915 MiB, which is what puts a
deployment under 2 GiB. Load time 2.58 s → 1.16 s.

It is **not shipped.** Its speaker-identity F0 sweep moves 6 of 16 cases into a
different register where the accepted CPU→CUDA placement baseline moves 1 — and
that baseline case is one a listener called *different people*. Two independent
pitch instruments agree on the count. A quantized generator produces a different
valid realization rather than a wrong one, so the exact-token gate cannot
adjudicate this and the ear has to.

### `Q4_K_GEN` — rejected

197 generator weights at Q4_K, the two lookup tables pinned Q8_0. It passes every
hard gate, its placement proof is clean, and the pin is demonstrated necessary
rather than predicted. It is rejected anyway: over `Q8_GEN` it buys 224 MB and
**zero throughput** (0.3% slower on CUDA, 1.6% slower on CPU), while ten of
seventeen renders acquire a DC pedestal reaching −0.0924 and envelope ratios
collapse by two to three orders of magnitude.

Recorded because it generalizes: **the speaker-identity proxy fails in the
dangerous direction here.** A render degraded until the pitch tracker finds no
voiced frame is scored *not comparable* rather than *changed*, so the proxy's
headline reads clean on a profile the degeneracy screen says is broken. A pitch
instrument answers "different speaker"; it cannot answer "not a voice". Both
must run.

### `Q8_MIXED` — codec only, convolutions exempt

Redefined on 2026-08-09 rather than renamed, being blocked and unpublished at
the time. This family **never block-quantizes a convolution kernel**: a fifth
`QuantRole`, `ConvKernel`, holds all 85 codec convolutions at the profile's
halved fallback at native three-axis shape, and the only codec tensors a profile
still quantizes are 73 HuBERT Linears whitelisted positively by module name. The
rule reads the name and never the rank — `codec.acoustic_decoder.conv2.weight`
is `[7, 32, 1]`, which `ggml_n_dims` collapses into something indistinguishable
from a Linear.

That took clone-path RVQ drift from 1,023 of 2,808 positions to 98, a factor of
10.4, and attribution showed all of the remainder is convolution precision:
quantizing the 73 Linears to Q8_0 costs nothing measurable (98 against F16's
103) while saving 80 MB, whereas convolutions at F32 instead of F16 reach 19 of
2,808 for 161 MB more. The profile is still **blocked** — `ref.tokens` is 0/2
exact, the one gate with no "different valid realization" defence.

Consequently this family deliberately does not use the packed-convolution branch
this project built for it. That code stays, shared with three other families and
covered by tests that build packed kernels directly; its presence is not
evidence that OmniVoice packs convolutions.

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
