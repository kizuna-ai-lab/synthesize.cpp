# Quantization Policy

Status: VITS F16 and Q8_MIXED version 1 functionally validated on 2026-07-23;
both profiles re-cut on 2026-07-27 with transpose-convolution weights held at F32.
Kokoro F16 and Q8_MIXED version 1 functionally validated on 2026-07-26.

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
F32. Acoustic-flow and decoder weights use F16. This assignment was selected
after an all-F16 duration path changed `ceil` decisions; retaining the complete
duration-decision domain in F32 restores exact duration structure in all 12
port-validation cases for both current VITS variants.

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
remain F16 because that operator has a different native layout and kernel.

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
(`ggml-patches/README.md`). No other tensor's profile assignment changed. CPU, GB10 CUDA 13.3, and RTX 4070 SUPER
CUDA 13.3 execute both public synthesis paths; both CUDA hosts report zero
executable CPU fallback. All platforms preserve the 12/12 duration structures.
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
