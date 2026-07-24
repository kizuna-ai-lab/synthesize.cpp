# Kokoro Port Validation Plan

Status: Intake complete on 2026-07-25. Conversion, C++ implementation, and port
validation are not started.

## Reference Contract

The oracle is the pinned `hexgrad/kokoro` PyTorch implementation at commit
`dfb907a02bba8152ca444717ca5d78747ccb4bec` (package version 0.9.4), driving the
`hexgrad/Kokoro-82M` weights at revision
`f3ff3571791e39611d31c381e3a41a3af07b4987`. Both carry an explicit Apache-2.0
grant. The upstream model card declares the checkpoint SHA-256
`496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4`, which the
intake independently measured and matched.

The architecture is StyleTTS 2 decoder-only with an iSTFTNet generator. The
model is deliberately **language-blind**: it consumes IPA phoneme token IDs and
a 256-wide style vector and has no language input, so language is a property of
the caller's phonemes and the selected Voice rather than a graph conditioning
path.

`KModel.forward_with_tokens` is the pinned inference entry point. Public
`speaking_rate = r` maps to upstream `speed = r`, which divides the summed
duration. The initial validated speaking-rate range is `0.80` through `1.25`;
expanding it requires new cases and a suite-version change.

Native output is 24,000 Hz mono F32. One predicted duration step is exactly 600
samples through the fixed chain `x2` prosody upsample, `x10` and `x6`
transposed convolutions, and the `x5` iSTFT hop.

### Stochastic Contract

Kokoro synthesis consumes randomness. Two unseeded oracle runs of the same
request produce identical durations but PCM differing by up to `7.892871e-2`.
The randomness is confined to the harmonic-plus-noise source module:

- `SineGen._f02sine` draws `rand_ini`, a uniform initial phase of shape
  `[1, 9]` whose first element is forced to zero, so harmonics 1 through 8
  receive a random phase offset at the first time step;
- `SineGen.forward` adds Gaussian `noise` of the same shape as the sine waves;
- `SourceModuleHnNSF.forward` draws a third Gaussian tensor that `Generator`
  never consumes.

The package therefore sets the stochastic capability. Following
`docs/port-validation.md`, the oracle emits `source.rand_ini` and `source.noise`
and an internal validation-only family seam replays those exact tensors in both
the reference and C++ graphs. The seam is unavailable to the CLI, public C
Interface, and language Adapters. The public seed contract is validated
separately by `request_repeatability` cases and `artifact_differs` relations, and
synthesize.cpp's own random stream is not specified to reproduce PyTorch's
generator.

### Voice Contract

A Voice is one `[510, 1, 256]` voicepack. The 256 values split into
`ref_s[:, :128]` consumed by the decoder and `ref_s[:, 128:]` consumed by the
prosody predictor.

The row is selected by **input length**, not only by Voice identity. Upstream
`KPipeline` uses `pack[len(ps) - 1]`, where `ps` is the phoneme string before
vocabulary filtering. Because the project's Text Frontend rejects unmapped
scalars instead of silently dropping them, the mapped scalar count always equals
the unpadded token count, so the project's equivalent rule is:

```text
style_row = final_token_count - 3
```

where `final_token_count` includes the two padding tokens. Requests whose
`final_token_count` exceeds 512, or whose resolved `style_row` falls outside
`[0, 509]`, are rejected before graph execution.

Upstream also supports averaging several voicepacks into a blended Voice. That
is a Voice Profile fusion concern and is outside this first port; v1 exposes
only Preset Voices.

### Text Frontend Contract

The package declares `synthesize.symbol_map` with `unicode_scalar` mapping over
the 114-entry `config.json` vocabulary, whose IDs are sparse within the 178-row
embedding table. The padding rule differs from VITS: Kokoro wraps the mapped
sequence with token ID 0 at both ends and inserts no interleaved blanks.

Raw-text input is not claimed. misaki 0.9.4 and its espeak-ng fallback resolve
Golden source text offline; they are validation tooling and never runtime
dependencies. Exact token-ID input continues to bypass the frontend.

### Language Capability

The first port validates and declares `en` with narrow regional fallback. The
checkpoint additionally ships voices for British English, Japanese, Mandarin
Chinese, Spanish, French, Hindi, Italian, and Brazilian Portuguese. Because the
graph has no language conditioning, those capabilities are withheld until they
have their own validation cases rather than inferred from voice names.

## Required Probe Set

Every Kokoro parity case records these logical probes with stable project names.
Shapes are given for a case with `T` final tokens and `Y` summed duration steps.

```text
input.token_ids           [T]         exact
bert.hidden               [T, 768]
text.d_en                 [512, T]
duration.d                [T, 640]
duration.logits           [T, 50]
duration.pred_dur         [T]         exact
duration.y_length         scalar      exact
duration.alignment        [T, Y]      exact
prosody.en                [640, Y]
prosody.f0                [2Y]
prosody.n                 [2Y]
text.t_en                 [512, T]
text.asr                  [512, Y]
source.har                [22, 2Y*300/5 + 1]
audio.pcm                 [600Y]
```

`duration.d` and `prosody.en` carry 640 channels because the DurationEncoder
concatenates the 128-wide prosody style onto its 512-wide hidden state. F0 and
energy probes are at `2Y` because the second F0/N residual block upsamples.
`source.har` is the concatenated magnitude and phase of the harmonic source
STFT and is recorded to validate the source module rather than only its
downstream effect.

Integer token IDs, rounded durations, summed length, and the alignment path are
exact structural probes. Remaining F32 tensors and PCM use
`tests/tolerances/kokoro.json`.

The oracle additionally emits the replay inputs `source.rand_ini` `[1, 9]` and
`source.noise` `[600Y/300 * 300, 9]`, which are stochastic inputs rather than
compared probes.

## kokoro-v1-0 Variant

`kokoro-v1-0` exposes all 54 upstream voicepacks as stable Preset Voice
identifiers matching their upstream names, such as `af_heart` and `bm_george`.
No package default Voice is invented; every request selects one explicitly. All
cases use language `en`, oracle-resolved token IDs, and no caller output-frame
limit.

| Case ID | Source text | Voice | Seed | Rate | Coverage |
| --- | --- | --- | ---: | ---: | --- |
| `kokoro-upstream-default` | `The sky above the port was the color of television, tuned to a dead channel.` | `af_heart` | 0 | 1.00 | Exact upstream example, baseline, repeatability |
| `kokoro-minimal` | `Hi.` | `af_heart` | 0 | 1.00 | Minimal sequence, low style row, short-utterance branch |
| `kokoro-short` | `Hello, world!` | `af_heart` | 0 | 1.00 | Short sequence |
| `kokoro-punctuation` | `Wait... really? Yes!` | `af_heart` | 0 | 1.00 | Punctuation-shaped token sequence |
| `kokoro-normalization` | `Dr. Smith paid $12.50 on January 3rd.` | `af_heart` | 0 | 1.00 | Upstream normalization-shaped token sequence |
| `kokoro-medium` | `The quick brown fox jumps over the lazy dog.` | `af_heart` | 0 | 1.00 | Medium sequence and broad symbol use |
| `kokoro-repetition` | `No, no, no; yes, yes, yes.` | `af_heart` | 0 | 1.00 | Repeated tokens and durations |
| `kokoro-long` | `A small speech synthesis library should produce the same finite waveform whenever its model, input, controls, and random seed are unchanged.` | `af_heart` | 0 | 1.00 | Long sequence, high style row, allocation growth |
| `kokoro-voice-first` | `Hello, world!` | `af_alloy` | 0 | 1.00 | First catalog entry |
| `kokoro-voice-british` | `Hello, world!` | `bm_george` | 0 | 1.00 | Second English accent group |
| `kokoro-voice-last` | `Hello, world!` | `zm_yunyang` | 0 | 1.00 | Last catalog entry |
| `kokoro-seed-one` | `The sky above the port was the color of television, tuned to a dead channel.` | `af_heart` | 1 | 1.00 | Alternate stochastic path |
| `kokoro-seed-forty-two` | `The sky above the port was the color of television, tuned to a dead channel.` | `af_heart` | 42 | 1.00 | Second alternate stochastic path |
| `kokoro-rate-slow` | `The quick brown fox jumps over the lazy dog.` | `af_heart` | 0 | 0.80 | Slow-rate boundary |
| `kokoro-rate-fast` | `The quick brown fox jumps over the lazy dog.` | `af_heart` | 0 | 1.25 | Fast-rate boundary |

In the `public_request` phase, the manifest relates the three upstream-text seed
cases with `artifact_differs(audio.pcm)` and requires identical
`duration.pred_dur` across them, because only the source module consumes
randomness. It relates `kokoro-rate-slow`, `kokoro-medium`, and
`kokoro-rate-fast` with strictly decreasing output frame count. The
upstream-default case performs public same-seed repeatability in addition to
parity replay. The three voice cases share one text, so their `duration.pred_dur`
and `text.asr` differ only through style conditioning; the manifest relates their
`audio.pcm` with `artifact_differs`.

## Materialization Rule

The manifest is created only after the exact repository commit, checkpoint
bytes, config bytes, voicepack bytes, and reference environment have been
resolved and the license status audited. SHA-256 values are measured from
acquired files; placeholders or hashes copied from an unverified mirror are
forbidden. No training corpus is needed to materialize these cases.

The intake packet is recorded in
[`reports/porting/kokoro/kokoro-v1-0/intake.json`](../../../reports/porting/kokoro/kokoro-v1-0/intake.json).
Its source, weights, configuration, voicepack, and license evidence are pinned
and verified, and its measured stochastic behavior is the basis for the replay
contract above.

## New Operator Surface

Relative to the VITS port, Kokoro introduces operators that Stage 4 must supply
or decompose:

| Operator | Where | Note |
| --- | --- | --- |
| Bidirectional LSTM | 5 instances | No native GGML operator; the largest implementation item |
| `InstanceNorm1d` (affine) | every AdaIN1d | Normalizes over time per channel |
| Snake1D | AdaINResBlock1 | `x + (1/a) * sin(a*x)^2` |
| Forward STFT / inverse iSTFT | harmonic source, final output | `n_fft=20`, `hop=5`, periodic Hann, centered |
| Cumulative sum | sine phase accumulation | Over the downsampled rate grid |
| Linear / nearest interpolation | sine phase, prosody upsample | |
| Grouped `ConvTranspose1d` | upsampling AdainResBlk1d pool | Depthwise, stride 2 |
| Reflection padding | final generator upsample | |

The four shared ALBERT layers are one weight group replayed 12 times, so the
converter emits a single layer group rather than 12 copies.

The upstream `CustomSTFT` is an ONNX-export approximation that uses replicate
padding and omits DC/Nyquist and window-envelope normalization. It is not the
default path and is not the parity target; the port reproduces
`torch.stft`/`torch.istft` semantics.

Candidate host seams, mirroring the VITS duration seam, are the duration
rounding and alignment construction, and the harmonic source generation plus its
STFT. Both are host-side by nature: they consume a resolved length or produce a
fixed excitation, and neither needs to be a backend graph node for correctness.

### Recorded decision: LSTM placement

The five bidirectional LSTMs run on a **host seam** for the first port. GGML has
no LSTM operator, and the alternative of unrolling one graph node group per time
step does not scale on this architecture: `predictor.shared` runs over frames
rather than tokens, so the 376-step `kokoro-long` case would already build about
7,500 nodes and the package's 60-second output limit would allow roughly 48,000.

This follows the project's CPU-correctness-first sequence and keeps the graph
sizes bounded. It is explicitly a starting point, not the end state: the LSTM
stages stay on CPU, so the eventual CUDA claim will report real executable CPU
fallback for them until the placement is revisited. Moving them into the graph,
by unrolling the four token-length LSTMs or adding a dedicated operator, is a
separate slice with its own validation and measurements.
