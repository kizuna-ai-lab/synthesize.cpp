# kokoro-v1-0 Porting Log

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

## 2026-07-25 — Golden Manifest and reference dump

- Adopted the schema-legal variant slug `kokoro-v1-0`; the manifest `key`
  pattern forbids the underscore in the upstream checkpoint filename.
- Added the 15-case `kokoro-v1-0` Golden Manifest at `suite_version` 1. Its
  first case is the upstream `__main__` example; the remainder cover minimal,
  punctuation, normalization, repetition, and long sequences, three additional
  voices, two alternate seeds, and both speaking-rate boundaries.
- Resolved token IDs are committed **inline** rather than as generated
  artifacts. The manifest is therefore self-contained, and replay no longer
  depends on misaki, spaCy, or espeak-ng being installed.
- Declared the package contract: 24,000 Hz mono F32, phoneme and token-ID
  input, `synthesize.symbol_map` with `unicode_scalar` mapping and the
  `wrap_pad_token` padding rule, language `en`, all 54 preset voices with no
  package default, stochastic, speaking rate 0.80 to 1.25, 512 max input
  tokens, and 1,440,000 max output frames (60 seconds).
- Added `scripts/dump_reference_kokoro_pytorch.py`. It reimplements no Kokoro
  arithmetic: it drives the pinned upstream modules stage by stage to capture
  intermediates, and injects the three harmonic-source random draws so the same
  values can be replayed in C++. Every case is verified against an unmodified
  `KModel.forward_with_tokens` under the same injected randomness, and a
  mismatch aborts the dump.
- Dumped all 15 cases. Every case matched `forward_with_tokens` bit-exactly,
  produced exactly `y_length * 600` finite samples, and wrote its 16 probes plus
  2 replay inputs. Re-running a case reproduces byte-identical PCM.
- Verified the declared relations against the dumped payloads: the three
  upstream-text seed cases have distinct PCM, the four voice cases have distinct
  PCM, and the slow/default/fast rate cases emit 95,400, 78,000, and 64,200
  frames respectively. The three seed cases share `y_length` 198, confirming
  that randomness reaches only the source module.
- Recorded `tests/tolerances/kokoro.json` with status `not-yet-measured`. No
  placeholder threshold is claimed.

## 2026-07-25 — Source-F32 GGUF conversion

- Added the manifest-driven `scripts/convert-kokoro.py`. Its external interface
  is `--manifest`, `--config`, `--checkpoint`, `--voices-dir`, and `--outfile`;
  callers control no tensor mapping, layout, metadata, or quantization.
- The converter verifies the checkpoint, config, and all 54 voicepacks against
  the digests pinned in the manifest before reading them, which is a stronger
  provenance check than the VITS converter's Git-revision comparison because
  the Kokoro weights live in a separate repository from the source.
- Canonical GGUF tensor names are the upstream paths with the DataParallel
  `module.` prefix stripped, so every tensor stays traceable to its checkpoint
  entry without inventing a parallel naming scheme.
- Fused all 89 dimension-0 weight-normalization pairs with the upstream
  formula. Skipped exactly two tensors, `bert.pooler.weight` and
  `bert.pooler.bias`, because `CustomAlbert` returns the last hidden state and
  never calls the pooler. All 548 checkpoint entries are accounted for:
  178 fused sources plus 370 plain, minus 2 skipped, giving 457 model tensors.
- Embedded the 54 voicepacks as `voice.<id>` tensors, squeezing the singleton
  batch dimension to a `[510, 256]` style table. Total 511 GGUF tensors:
  bert 23, bert_encoder 2, predictor 106, decoder 305, text_encoder 21,
  voice 54.
- Recorded that `AdaIN1d`'s `InstanceNorm1d(affine=True)` parameters are absent
  from the checkpoint. `KModel` loads with `strict=False`, so they stay at their
  defaults of weight 1 and bias 0, making the operation a plain instance norm
  with eps 1e-5. The package declares
  `synthesize.kokoro.adain.instance_norm_affine = false` so the C++ module does
  not look for tensors that do not exist.
- The GGUF is reopened after writing and every metadata field, tensor name,
  type, reversed GGML shape, and payload byte is verified against the prepared
  source arrays.
- Two complete runs produced the identical 352,814,592-byte file with SHA-256
  `1fe650e99276466aca6cf3f71c30bd8df63878f40d32a18bb3a4ab655238e920`.
- Added 28 registered converter unit tests covering manifest identity and
  package-contract drift, config dimension drift, sparse-vocabulary
  densification, weight-norm fusion and its unpaired/collision/wrong-shape and
  wrong-count rejections, skip rules, emitted-count accounting, voicepack
  digest/shape/missing-file/incomplete-pin rejection, and size labels. The
  generated GGUF and converter report remain ignored artifacts.

## 2026-07-25 — Stage 4 slice 1: shared metadata reader and Kokoro hparams

- Extended the shared `synthesize.symbol_map` frontend rather than adding a
  Kokoro-specific one. The interleave-or-not boolean became a
  `SymbolPaddingRule` of `None`, `InterleavedBlank`, or `WrapPadToken`, and the
  dense symbol table now tolerates empty entries so a sparse vocabulary can
  reserve embedding rows. VITS keeps its behavior, derived from the metadata it
  already stores.
- Introduced `src/gguf-metadata.{h,cpp}` for typed, fail-closed GGUF reads and
  migrated VITS onto it by reducing its eight private helpers to one-line
  delegations, which keeps all 66 existing call sites untouched and deletes
  about 130 lines of logic that Kokoro would otherwise have duplicated.
- Added `src/arch/kokoro/weights.{h,cpp}` with `read_hparams`. Beyond typed
  reads it enforces the contract: identity and architecture version, a known
  quantization profile, phoneme-plus-token input only, mandatory speaking-rate
  and stochastic capabilities, mono audio, an ordered rate range containing 1.0,
  even hidden width for the bidirectional split, an odd text-encoder kernel, a
  single shared ALBERT group, a centered even-length iSTFT whose hop divides the
  transform, resblock kernels that are odd with in-range padding, and a source
  module whose sampling rate matches native output.
- Cross-checked derived quantities instead of trusting declarations: the
  `2 * prod(upsample_rates) * hop` chain must equal the declared 600 samples per
  duration step, and the source upsample scale must be exactly half of it.
- Enforced the Voice contract in metadata: the style table must hold a decoder
  and prosody half per row with contiguous offsets, must address the declared
  maximum input, and no catalog entry may claim a package default because the
  variant has none. `resolve_style_row` implements the
  `final_token_count - 3` rule with bounds checks at both ends.
- Rejected `instance_norm_affine = true`, which would reference AdaIN tensors
  the converter never emits.
- Added the registered `synthesize-kokoro-metadata-test`: it parses a small but
  structurally faithful synthetic package, checks the resolved hparams and every
  style-row boundary, and then asserts rejection for about sixty individual
  metadata mutations. A negative control confirmed the rejection helper
  discriminates rather than always reporting failure.
- Found that `synthesize-check-unit` carries an explicit DEPENDS list, so a new
  unit test silently never builds in a clean tree until it is registered there.
  The sanitizer tree caught this as a "Not Run" result; the new test is now
  listed. Both the ordinary gate (46/46) and a clean sanitizer gate (45/45)
  pass.

### Repository defect found and fixed

Authoring this manifest exposed that neither committed VITS manifest satisfied
the committed schema: `package_contract.frontend` set `additionalProperties` to
false while every manifest carried `phoneme_mapping`. Nothing detected it
because no test compared a manifest with the schema. The schema now accepts the
phoneme mapping plus an optional padding rule, and a registered
family-independent test `synthesize-golden-manifest-contract` validates every
manifest under `tests/golden/` together with the loader rules from
`docs/port-validation.md`. The test was confirmed to fail against the previous
schema.
