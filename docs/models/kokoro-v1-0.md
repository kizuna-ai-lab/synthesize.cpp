# Kokoro v1.0

Status: F32, F16, and Q8_MIXED are `port_validated`. Quality evaluation has not
been run. The packages are published in
[`jiangzhuo9357/kokoro-v1-0-gguf`](https://huggingface.co/jiangzhuo9357/kokoro-v1-0-gguf)
at revision `ec897a4c400e8e5a69eb9da6bd423c0152f9030c`.

## Package

The package is converted from the official Kokoro-82M v1.0 checkpoint at pinned
upstream revision `dfb907a02bba8152ca444717ca5d78747ccb4bec`, weights revision
`f3ff3571791e39611d31c381e3a41a3af07b4987`. It is a StyleTTS 2 decoder with an
iSTFTNet generator, produces 24,000 Hz mono F32 PCM, and carries all 54 preset
Voices with no package default, so every request must name one. It accepts UTF-8
phonemes through the built-in `synthesize.symbol_map` frontend or exact token
IDs. Raw-text G2P is not included.

The model is language-blind: it consumes IPA phoneme token IDs and a style
vector and has no language input of its own. A Kokoro Voice is a table indexed
by input length rather than a single vector, so the row is resolved per request
from the final token count.

| Profile | Bytes | Tensor storage | SHA-256 |
| --- | ---: | --- | --- |
| F32 | 352,813,952 | 511 F32 | `23cde0e3b2a3082fa97a84aed746d8c0cee79eca1ee9599c6e13b7e94ba0328b` |
| F16 | 246,451,648 | 384 F32 + 127 F16 | `951e4be979b8af5e72f2353de947c65372c84790b500dd0f53384c958ee2596e` |
| Q8_MIXED | 216,109,696 | 384 F32 + 14 F16 + 113 Q8_0 | `0d72f3778125a8f54c23468c6a2534f114d529121f9788e46ec45dc7d21b923c` |

The public C ABI is profile-independent. C++, Rust, and Python callers load a
local GGUF through the same model interface. The tensor policy, packed Q8 matrix
layout, and execution details remain private to the architecture module.

### What the profiles quantize, and what they never do

Everything that decides F0 and the durations stays at the reference dtype at
every profile: PL-BERT, the prosody and duration path, the acoustic text
encoder, and the Voice tables. Only the decoder and its generator are
quantized, and those are 60% of the parameters.

The reason is measured rather than conventional. Kokoro's excitation is a sine
whose phase accumulates across the whole utterance, so a relative difference in
F0 of a few parts in ten thousand becomes radians of phase by the last syllable.
A profile that quantized the front end changed six of one case's 78 predicted
durations and made the utterance 50 ms longer than the reference.

Twelve decoder matrices carry F16 rather than Q8_0 in the mixed profile. The
decoder concatenates the two prosody curves and a narrow encoder residual onto
its feature stream, which lands several channel counts two short of a multiple
of thirty-two, and a block-quantized row has to be a whole number of blocks.

## Port validation

Seven graph stages and 15 deterministic cases run on DGX Spark CPU and NVIDIA
GB10 CUDA 13.3, against the pinned upstream PyTorch implementation at
`suite_version` 2.

The predicted durations, the frame count, and the alignment are **exact in every
case, on every profile, and on both backends**. That is the structural
guarantee: every clip is the same length as the reference, to the sample.

| Probe | Worst absolute difference |
| --- | ---: |
| `text.t_en`, `text.asr` | 2.53e-6 |
| `bert.hidden` | 1.70e-5 |
| `prosody.n` | 2.17e-5 |
| `duration.d`, `prosody.en` | 3.45e-5 |
| `text.d_en` | 3.59e-5 |
| `duration.logits` | 2.86e-4 |
| `prosody.f0` | 2.58e-3 |

Two probes carry a comparison rule of their own rather than a wider threshold.
The harmonic source's spectrum is compared as a complex value, because a stored
phase is meaningless wherever its magnitude is near zero. The decoder's spectrum
leaves the graph as a logarithm and a pre-sine angle whose logarithms reach
about -69 — a magnitude of 1e-30 — so it is compared after both activations, as
the value the inverse transform consumes.

| Profile | CPU waveform correlation | GB10 CUDA |
| --- | ---: | ---: |
| F32 | 0.987437 | 0.989607 |
| F16 | 0.987445 | not measured |
| Q8_MIXED | 0.984790 | not measured |

Correlation is the honest measure here rather than a sample-wise tolerance, for
the same reason the profiles are split the way they are: the waveforms diverge
by construction while the speech does not. The figures are the worst case over
the suite; spectrogram correlation does not fall below 0.9978 on any profile.

These measurements prove functional execution and record numerical drift; they
are not perceptual-quality thresholds. Naturalness and intelligibility remain
unevaluated. An informal listening pass over eight cases found no audible
defect on any profile or backend, which is evidence that the measured agreement
is not hiding an audible disagreement — it is not a quality evaluation.

## Reproduction

Materialize the reference artifacts, then run the seven registered validators:

```bash
uv run --project scripts/envs/kokoro --locked \
  python scripts/dump_reference_kokoro_pytorch.py \
    --manifest tests/golden/kokoro/kokoro-v1-0.manifest.json \
    --source-dir models/upstream/kokoro-source \
    --weights-dir models/upstream/kokoro-v1_0

cmake -S . -B build -DSYNTH_BUILD_TESTS=ON -DSYNTH_BUILD_PYTHON_TESTS=ON \
  -DSYNTH_BUILD_INTEGRATION_TESTS=ON
cmake --build build --target synthesize-check-integration
```

Produce the quantized packages:

```bash
cmake -S . -B build -DSYNTH_BUILD_TOOLS=ON
cmake --build build --target synthesize-quantize
build/bin/synthesize-quantize \
  models/kokoro-v1-0/kokoro-v1-0-F32.gguf \
  models/kokoro-v1-0/kokoro-v1-0-F16.gguf --quant F16
build/bin/synthesize-quantize \
  models/kokoro-v1-0/kokoro-v1-0-F32.gguf \
  models/kokoro-v1-0/kokoro-v1-0-Q8_MIXED.gguf --quant Q8_MIXED
```

Generate and verify the model card:

```bash
uv run scripts/hf_cards/generate.py scripts/hf_cards/kokoro-v1-0.yaml
uv run scripts/hf_cards/generate.py scripts/hf_cards/kokoro-v1-0.yaml --check
```

The flat published directory was created with:

```bash
hf repos create jiangzhuo9357/kokoro-v1-0-gguf \
  --type model --public --exist-ok
hf upload jiangzhuo9357/kokoro-v1-0-gguf models/kokoro-v1-0 . \
  --commit-message "Publish Kokoro v1.0 F32, F16, and Q8_MIXED"
```

## Licensing

Both the source and the weights carry an explicit Apache-2.0 grant, so no
redistribution assumption is required for this variant. The upstream model card
lists CC BY training sources — Koniwa `tnc` and SIWIS — whose attribution is
carried into the generated model card.

Only English is declared. The checkpoint also ships Voices for British English,
Japanese, Mandarin Chinese, Spanish, French, Hindi, Italian, and Brazilian
Portuguese; those Voices load and run, but no language beyond `en` has its own
validation cases, so none is advertised. See `docs/languages.md`.
