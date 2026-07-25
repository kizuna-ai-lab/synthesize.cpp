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

## 2026-07-26 — Stage 4 slice 9: harmonic source and its short-time spectrum

Added `source.{h,cpp}`, the second host seam. The module carries one learned
9-to-1 projection and is otherwise phase accumulation, resampling, and a
transform — work GGML expresses poorly — and it sits between two graph stages
that both need concrete shapes. Keeping it on the host also gives port
validation somewhere to inject the oracle's recorded randomness, which the
`SourceRandomInputs` struct exists for.

Three upstream behaviours were measured before being implemented rather than
read off the source, because each has a convention that is easy to assume
wrongly:

- `F.interpolate(mode="linear")` maps an output index with `align_corners=false`
  as `src = (o + 0.5) * in / out - 0.5`, clamped at zero. Confirmed exactly on
  the downsample and to 1e-4 on the upsample, the latter only because the phases
  being interpolated are already a few hundred radians.
- Phase accumulates on the *coarse* grid and is then multiplied by the upsample
  scale, since each coarse step stands for that many samples of advance. Doing
  the sum on the fine grid instead is off by exactly that factor.
- `torch.stft` with centre framing reflect-pads by `n_fft / 2` without repeating
  the edge sample, applies a periodic (not symmetric) Hann window, and yields
  `upsampled / hop + 1` frames. Matched to 4.8e-7 in magnitude.

The seam is validated end to end against upstream `SourceModuleHnNSF` plus
`TorchSTFT` at reduced dimensions, with the two random draws replayed from the
PyTorch run: 1e-5 on the excitation and on the magnitude bins, 1e-4 on the phase
bins compared as points on the circle, which is the right comparison since phase
wraps.

## 2026-07-26 — Stage 4 slice 8: acoustic text encoder

Added `text-encoder.{h,cpp}`: an embedding, then three convolution blocks with a
layer norm and leaky activation, then a bidirectional LSTM.

This is a second, separate encoder. PL-BERT conditions prosody; this one
produces the features the decoder consumes after alignment expansion. Its layer
norm normalizes over channels within a frame, the opposite axis from the
prosody stack's AdaIN, which is why it is compared against a host reference
rather than checked for shapes alone. Agreement is 1e-5.

The builder rejects an even convolution kernel, since only an odd kernel with
half-width padding preserves the sequence length the alignment expansion
assumes.

## 2026-07-26 — Stage 4 slice 7: prosody F0 and energy stack

- Added `prosody.{h,cpp}` with the AdaIN residual block and the F0/energy
  stacks. Both curves leave this stage at twice the frame rate because the
  middle block upsamples; the decoder halves them again with a strided
  convolution.
- AdaIN normalizes over **time within each channel**, a different axis from the
  adaptive layer norm in the duration path, which normalizes over channels
  within each frame. Getting these two the same way round is the kind of error
  a shape-only test would not catch, so the block is compared numerically.
- The block averages its residual and shortcut branches by scaling their sum by
  `1/sqrt(2)`. The shortcut must double the length by nearest-neighbour
  repetition exactly where the residual doubles it by the transposed pool.

### Two GGML constraints resolved

`ggml_conv_1d` asserts that its kernel is F16, because its im2col buffer is
half precision. Downcasting the weights of a source-F32 accuracy artifact is
not acceptable, so the family uses the same im2col-and-matrix-multiply
decomposition the VITS port established, keeping the kernel in F32.

That decomposition also settled a layout question. The first draft of the
convolution operators used `[time, channels]`, because that is what
`ggml_conv_1d` consumes directly, which would have left two layouts in the
codebase with transposes at the boundary. Since the F32 path transposes
internally anyway, every Kokoro stage now uses `[features, time]` like VITS,
and `operations.h` states that as the single convention.

The residual block is checked against a host reference across four shapes: the
three the prosody stack actually uses, plus a channel-changing block that does
not upsample, agreeing to 1e-4. A block that claims to upsample without a pool
is rejected as a catalog defect rather than silently skipping the upsample.

## 2026-07-26 — Stage 4 slice 6: depthwise transposed convolution

The prosody stack's upsampling block and the decoder both need a depthwise
transposed convolution, and GGML supplies no usable primitive:
`ggml_conv_transpose_1d` has no grouping and asserts that internal padding is
zero, while `ggml_conv_1d_dw` is a forward convolution.

Rather than reversing the stored kernel, which would either make the GGUF
disagree with the checkpoint or add weight preparation at load time, the
operation is assembled from the scatter definition. Inserting `stride - 1`
zeros between input samples turns the scatter into a plain correlation, and
each kernel tap then contributes one shifted copy scaled by a per-channel
value read as a strided view of the stored weight. The package therefore stays
byte-faithful and nothing is preprocessed.

The test compares against the scatter definition itself across seven shapes:
Kokoro's own kernel 3, stride 2, padding 1, output padding 1 pool, which
exactly doubles the length, plus unit stride, unit kernel, a wider kernel,
stride 3, a single-sample input, and the bias and bias-free forms. Agreement is
1e-5. It also pins that padding wider than the kernel is refused rather than
clamped, and that a kernel whose channel count disagrees with the input is
rejected as a catalog defect.

## 2026-07-26 — Stage 4 slice 5: duration path and its host seam

- Extracted `layer_norm`, `linear`, `ada_layer_norm` and a time broadcast into
  `operations`, so the PL-BERT and duration stages share one implementation.
- Verified `AdaLayerNorm` semantics against the pinned module rather than
  reading them off the source: its four transposes cancel, so the net operation
  is a channel layer norm **without learned affine**, followed by
  `(1 + gamma) * x + beta` where both come from one projection of the style
  vector split in half. Applying `1 + gamma` means a zero projection is the
  identity. Its epsilon is 1e-5, fixed by the module rather than declared by
  the package.
- Added `duration.{h,cpp}`. The encoder alternates a bidirectional LSTM with
  that adaptive norm and re-concatenates the style vector after each one, which
  is why `duration.d` is `hidden_dim + style_dim` wide rather than `hidden_dim`.
  The builder advertises its node count so callers can size graphs without
  guessing, and it refuses to build unless every LSTM has a persistent store.
- Added `duration-host.{h,cpp}`, the seam that turns a distribution into a
  concrete length the later graphs need as a static shape. It sums the sigmoid
  over the duration bins, divides by the speaking rate, rounds, and clamps to at
  least one step so no token is ever dropped.
- The output limit is checked **before** the alignment is allocated, because
  that allocation is the product of token count and frame count. A non-finite
  logit is reported rather than propagated into a length.
- The test compares the graph against a host reference of the whole encoder
  including the LSTM and adaptive norm, agreeing to 1e-5, and then checks the
  seam's structure directly: every frame is attributed to exactly one token, the
  tokens appear in ascending order, a faster rate never lengthens the output and
  a slower rate never shortens it, and the limit rejects before allocating.

## 2026-07-25 — Stage 4 slice 4: PL-BERT encoder

- Added `src/arch/kokoro/plbert.{h,cpp}`. Semantics were read from the pinned
  transformers implementation rather than assumed: embeddings sum the word,
  absolute-position and type-zero rows before a LayerNorm, the encoder projects
  the 128-wide embedding to 768, and each layer is post-norm, applying the norm
  after adding the residual for both attention and feed-forward.
- The stored layer group is replayed for all 12 hidden layers, matching the
  package's `shared_layer_groups = 1`.
- Confirmed from the resolved `AlbertConfig` that the activation is `gelu_new`,
  the layer-norm epsilon is 1e-12, and both dropout probabilities are zero.
  `ggml_gelu` implements exactly the `gelu_new` tanh form.
- No attention mask is built. Kokoro synthesizes one unpadded sequence at a
  time, so upstream's mask is all-visible; the intake measured `text_mask` as
  entirely false. The host seam owns never presenting a padded batch.

### GGML's GELU is a half-precision table

The graph initially disagreed with the reference by `3.6e-4`, which looked like
a defect. It is not: `GGML_GELU_FP16` is defined, so the CPU kernel evaluates
`f16(gelu(f16(x)))` through a lookup table rather than the exact activation.
Re-running the reference with that same table semantics shrinks the deviation to
`5.4e-5`, and matching GGML's association order changes nothing further, so the
remainder is ordinary accumulation-order noise carried through three layers.

The test asserts both: agreement with the exact activation within a threshold
that still catches real defects, and that modelling the table shrinks the
deviation by more than three times. The second assertion is what distinguishes
"the backend approximates" from "the graph is wrong".

This matters beyond this slice. Every Golden probe downstream of a GELU inherits
a deviation from PyTorch on the order of `1e-4` on CPU that no amount of graph
correctness can remove, so Stage 5 tolerances must be measured with this in mind
rather than set from an expectation of float-exactness.

## 2026-07-25 — Stage 4 slice 3: tensor catalog, and a converter defect

- Added `src/arch/kokoro/catalog.cpp`, resolving all 511 tensors across the
  six namespaces. Shapes are derived from the package hyper-parameters rather
  than hardcoded, so a package whose metadata and tensors disagree is rejected
  at load time. Optional members are contract-driven: an `AdainResBlk1d` has a
  learned shortcut only when it changes channel count, and a depthwise pool only
  when it upsamples.
- The catalog resolved the real converted GGUF on the first attempt once the
  naming defect below was fixed, so every derived width was correct, including
  the decoder's `dim_in` actually being `hidden_dim` rather than the config's
  `dim_in`.

### Converter defect: over-long tensor names

Loading the real GGUF failed immediately with `tensor name 5 is too long:
75 >= 64`. GGML stores names in a fixed 64-byte field, and the decision to keep
upstream paths verbatim produced 14 ALBERT names of up to 79 characters. The
converter had written the file without complaint because `gguf-py` does not
enforce the limit, so the package was unloadable by our own runtime and nothing
had caught it.

Two fixes. Upstream's `bert.encoder.albert_layer_groups.0.albert_layers.0.`
prefix is now shortened to `bert.layer.`; the group and layer indices carry no
information because the package declares a single shared layer group, and the
rename is recorded per tensor in the converter report as `+rename`. More
importantly the converter now refuses to write any name that reaches the limit,
so the class of defect cannot recur for a future family.

The regenerated artifact is 352,813,952 bytes with SHA-256
`23cde0e3b2a3082fa97a84aed746d8c0cee79eca1ee9599c6e13b7e94ba0328b`, and two
complete runs still produce it byte for byte.

Registered tests: `synthesize-kokoro-catalog-test` resolves a small but
structurally faithful synthetic package and asserts rejection for a missing
tensor, a wrong shape, a wrong dtype, and an unexpected trailing axis. The
converter suite gained coverage of the rename, its injectivity over the ALBERT
group, and the name-length gate at and past the limit.

## 2026-07-25 — LSTM spike, and a corrected decision

The earlier host-seam decision rested on an estimate that unrolling would
produce unmanageable graphs. A spike at the family's real shapes disproved it,
so the decision was reversed to in-graph unrolling. The measurements and the
accumulation comparison are recorded in the family note; the reasoning is here.

- Surveyed three GGML ports. encodec.cpp unrolls but issues a GEMV per step on
  the input side. TTS.cpp, which is itself a Kokoro port, batches the input
  projection over the whole sequence but splits the four gates into four
  per-step GEMVs and accumulates with a per-step `ggml_concat` chain that copies
  the whole accumulated tensor every step. parakeet.cpp runs one graph per step
  with state on the host, which is forced by autoregressive beam-search decoding
  and does not apply here: every Kokoro LSTM has its full input before it runs.
- The shipped design takes the batched input projection from TTS.cpp and the
  fused four-gate hidden GEMV from encodec.cpp.
- Graph size is not the constraint the earlier estimate assumed. At the
  60-second output limit the unrolled bidirectional chain is 110,404 nodes but
  builds in 20.7 ms and runs in 111 ms on CPU and 151 ms on CUDA. The longest
  committed case, `Y=376`, is 17,300 nodes and under 24 ms on either backend.
- A four-thread host implementation with the same batched input projection needs
  815 ms for the 60-second case, so in-graph is about seven times faster and
  additionally keeps the stage on the backend.

### The accumulation hazard the spike caught

Writing per-step outputs is where this goes wrong, and it goes wrong quietly.
`ggml_set_1d_inplace` into an allocator-managed tensor produced correct results
on CPU to 1e-6 and **wrong results on CUDA**, off by 1.8 in a tanh-bounded
signal. A balanced `ggml_concat` tree over per-step views was self-consistent
across both backends and wrong on both. Moving the seed and output tensors to
graph inputs broke CPU as well.

Only writing each step with `ggml_cpy` into disjoint views of a tensor held in a
persistent backend buffer, the llama.cpp KV-cache pattern, is correct on both:
1e-6 on CPU and 9e-6 on CUDA, the latter being ordinary cross-backend float
accumulation order. It also cuts the 60-second compute buffer from 26.9 MiB to
9.4 MiB, since the allocator is then left with only the per-step intermediates.

The lesson is recorded as a standing requirement: the LSTM builder's test must
run on every Execution Backend the package claims. A CPU-only gate would have
accepted the first strategy and shipped a CUDA package that produced wrong
audio.

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
