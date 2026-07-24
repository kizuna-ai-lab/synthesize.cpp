# kokoro-v1_0 Porting Log

## 2026-07-25 — Intake and oracle smoke

- Pinned `hexgrad/kokoro` at commit `dfb907a02bba8152ca444717ca5d78747ccb4bec`
  (package version 0.9.4) and the weights repository `hexgrad/Kokoro-82M` at
  revision `f3ff3571791e39611d31c381e3a41a3af07b4987`.
- Downloaded only `config.json`, `kokoro-v1_0.pth`, and the 54 voicepacks into
  the ignored local model cache. No training corpus was downloaded.
- Verified the 327,212,226-byte checkpoint at SHA-256
  `496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4`. The
  upstream model card declares the same digest, so provenance is
  cross-confirmed rather than only self-measured.
- Inspected the checkpoint with PyTorch `weights_only=True`: five top-level
  namespaces (`bert`, `bert_encoder`, `predictor`, `decoder`, `text_encoder`)
  hold 548 entries and 81,763,410 parameters.
- Locked the project-owned CPU reference environment in
  `scripts/envs/kokoro/uv.lock` (Python 3.12.13, Torch 2.7.1+cpu,
  transformers 4.51.3, misaki 0.9.4, en_core_web_sm 3.8.0). `click` is pinned
  explicitly because spaCy's console entry points do not resolve it
  transitively in this resolution.
- Resolved eight candidate case texts through misaki with the espeak-ng
  fallback. All resolve with no characters dropped by the vocabulary filter;
  final token counts span 6 to 146.

### License audit — clear

Unlike the VITS port, both the source and the weights carry an explicit
Apache-2.0 grant: the pinned repository ships `LICENSE`, and the model card
metadata declares `license: apache-2.0` with the README stating "This is an
Apache-licensed model ... We welcome the deployment of the model in real use
cases." The voicepacks are distributed in the same repository at the same
revision. No publication blocker exists for this variant. The upstream card
lists CC BY training sources (Koniwa `tnc`, SIWIS) whose attribution must be
carried into the generated model card.

### Confirmed stochastic behavior

Two unseeded oracle runs of the same request produced identical durations but
different PCM, with a maximum absolute difference of `7.892871e-2`. Seeding
`torch.manual_seed(0)` makes runs bit-identical. The randomness lives entirely
in the harmonic-plus-noise source module: `SineGen._f02sine` draws a uniform
initial phase for harmonics 1 through 8, and `SineGen.forward` adds Gaussian
noise scaled by the voiced/unvoiced amplitude. `SourceModuleHnNSF` draws a
third Gaussian tensor that the Generator never consumes.

The package therefore declares the stochastic capability. Following
`docs/port-validation.md`, parity replay uses oracle-recorded random tensors
through an internal validation-only seam, while the public seed contract is
tested separately by repeatability and seed-difference relations.

### Architecture and operator surface

The model is StyleTTS 2 decoder-only with an iSTFTNet generator and is
deliberately language-blind: it consumes IPA phoneme token IDs plus a 256-wide
style vector, split as `[:128]` for the decoder and `[128:]` for the prosody
predictor. Native output is 24,000 Hz mono, and one predicted duration step is
exactly 600 samples (`x2` prosody upsample, `x10` and `x6` transposed
convolutions, `x5` iSTFT hop).

PL-BERT is an ALBERT whose single shared layer group is replayed for all 12
hidden layers, with a 128-wide embedding projected to 768. The style vector is
selected from a `[510, 1, 256]` voicepack by *input length*, which makes token
count part of Voice resolution rather than only a shape.

New operator surface relative to the VITS port, recorded for Stage 4 planning:

- bidirectional LSTM in five places, with no native GGML operator;
- `InstanceNorm1d` with affine parameters inside AdaIN1d;
- Snake1D activation `x + (1/a) * sin(a*x)^2`;
- forward STFT and inverse iSTFT at `n_fft=20`, `hop=5`, periodic Hann, centered;
- cumulative sum for sine phase accumulation;
- linear and nearest interpolation resampling;
- grouped (depthwise) `ConvTranspose1d` in the upsampling residual block;
- reflection padding.

The upstream `CustomSTFT` is an ONNX-export approximation (replicate padding,
no DC/Nyquist or window-envelope normalization) and is **not** the default
path. Parity targets `torch.stft`/`torch.istft` semantics.

### Recorded scope decisions

- **Language capabilities.** The first port validates and declares `en` with
  narrow regional fallback. The checkpoint also ships voices for British
  English, Japanese, Mandarin Chinese, Spanish, French, Hindi, Italian, and
  Brazilian Portuguese; those capabilities stay unadvertised until they have
  their own validation cases, per `docs/languages.md`.
- **Preset Voice Catalog.** All 54 voicepacks are embedded in the primary GGUF
  and exposed as stable identifiers. Voice conditioning is one shared
  table-lookup branch, so case coverage validates that branch and the
  length-indexed row selection rather than every individual voice.
- **Speaking rate.** Initial validated range 0.80 through 1.25, matching the
  VITS precedent. Upstream `speed` divides the summed duration, so the public
  `speaking_rate` maps directly.
- **Text input.** Not claimed. The package advertises phoneme and token-ID
  input only; misaki and espeak-ng remain offline validation tooling and are
  never runtime dependencies.
