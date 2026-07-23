# Reference CLI

Status: CPU reference Adapter implemented on 2026-07-22.

`synthesize-cli` is an optional command-line Adapter over the installed public C
Interface. It includes no Model Family or GGML headers and has no privileged
inference path. Disable it with `-DSYNTH_BUILD_CLI=OFF` when only the embeddable
library is needed.

The current VITS LJSpeech package accepts resolved token IDs:

```bash
build/bin/synthesize-cli \
  --model models/vits-ljspeech/vits-ljspeech-F32.gguf \
  --output speech.wav \
  --token-ids "0,156,0,47,0,102,0,4,0" \
  --backend cpu \
  --seed 42
```

The VCTK package uses the same generic request and requires a Preset Voice:

```bash
build/bin/synthesize-cli \
  --model models/vits-vctk/vits-vctk-F32.gguf \
  --output speech-vctk.wav \
  --token-ids "0,156,0,47,0,102,0,4,0" \
  --voice speaker-004 \
  --backend cuda \
  --seed 42
```

The CLI also exposes `--text` and `--phonemes` through the same generic request.
A Loaded Model that does not advertise those inputs returns its public Interface
error; the CLI does not add a private frontend. `--language`, `--voice`,
`--rate`, `--max-output-frames`, `--backend`, and `--device` map directly to
their public structures. `--seed random` maps to `SYNTH_SEED_RANDOM`, and the
CLI prints the concrete seed returned by synthesis.

Successful output is little-endian 32-bit IEEE-float WAV in the model's native
sample rate and channel count. Synthesis completes before the file is opened, so
model, request, cancellation, and output-limit failures do not leave an output
file. A later streaming-file Adapter may write Audio Sink chunks directly, but
it must still use `synth_synthesize()` rather than a Model Family seam.
