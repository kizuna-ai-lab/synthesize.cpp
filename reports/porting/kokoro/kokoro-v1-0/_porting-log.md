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

## 2026-07-26 — Completing CUDA coverage, and a measurement of the wrong binary

Every Quantization Profile now runs every stage on CUDA: 42 validator runs
across three profiles, seven stages, and 15 cases. The durations, frame count
and alignment are exact on all of them. Worst-case waveform correlation on CUDA
is 0.9896 at F32, 0.9898 at F16 and 0.9883 at Q8_MIXED — each slightly better
than the same profile on the CPU.

### The first sweep measured a binary that no longer existed

That sweep reported `decoder.spectrum` diverging by 1.2e14 on CUDA against 3.97
on the CPU. The number was real; what it described was not.

The tell was that it could not be true alongside the rest of the run: a spectrum
that wrong would reconstruct to an amplitude of 1e12, and the CUDA waveform
stage was producing a peak of 0.43 correlating at 0.9975. Two stages of the same
`compute_decoder` cannot disagree that way. Reading the dumped spectrum back as
graph layout instead of the transposed layout reconstructed to exactly the
waveform stage's output, which located the difference: the binary had not
transposed at all.

`build/dev-dgx-spark` was last built at the CUDA slice and the transpose was
added afterwards, with the decoder probe. The sweep had driven a stale runner.
Rebuilding and re-running put `decoder.spectrum` at 3.5 to 3.7, in line with the
CPU.

The lesson is procedural rather than technical: a validation run describes
whatever binary it drove, so the build has to be part of the run. The CPU tree
was rebuilt as a matter of course because it is what the unit gate uses; the
accelerator tree only gets rebuilt when someone remembers.

### Intermediate drift on CUDA is larger, and size-dependent

PL-BERT's hidden state differs by 1.9e-3 relative on CUDA against 7.5e-7 on the
CPU, with TF32 off and strict FP32 confirmed in the build. The deviation shrinks
as the sequence lengthens — 2.2e-2 absolute at six tokens, 7.7e-3 at sixteen,
2.5e-5 at twenty-three — and is spread evenly across token positions rather than
concentrated in one.

That shape argues against a defect: an uninitialized tail or a padding error
would sit in particular positions, and a real fault would usually worsen with
size rather than improve. Small matrix multiplies selecting different kernels
and reduction orders is the ordinary explanation. It does not reach the rounded
durations and does not degrade the waveform. It is recorded as measured; it is
not claimed to be fully explained.

## 2026-07-26 — Stage 8: publication artifacts

`docs/models/kokoro-v1-0.md`, `scripts/hf_cards/kokoro-v1-0.yaml`, the generated
`models/kokoro-v1-0/README.md`, and the flat publication directory are in place.
The package page states the Validation Level and the quality-evaluation status
explicitly, as the workflow requires.

### The card generator had VITS baked into it

Three things in the template were the first family's rather than every family's:
the metric columns (`cpu_pcm_max_abs` and two named CUDA machines), the
architecture name in prose, and the upstream repository label and licence
paragraph. Kokoro reports a correlation, not a sample-wise drift, and ran on two
platforms rather than three, so none of it fitted.

The spec now declares its own measured columns with a note saying what they
mean, its own architecture label, repository label, and licence paragraph. Prose
blocks in a spec are rendered through the template engine first, so a licence
note can cite `license_link` without repeating the URL — the previous shape
would have printed the placeholder literally.

Both VITS cards regenerate through the new path, and their tests pass unchanged
apart from the wording that now comes from the spec.

### Documents that claimed one family

`docs/model-family-selection.md` said the first supported family is VITS and
stopped there; it now records Kokoro as the second, why it was selected, and the
fact that a second family does not generalize the first one's decisions —
Kokoro's Quantization Profiles quantize the opposite part of the model, for a
reason measured in this family.

`docs/quantization.md` states the same thing at the policy level: the general
rule is that large matrix weights are the first candidates wherever they sit,
and Kokoro is a documented exception with the measurement behind it, not a
revision of the rule.

### Published

All three packages are published in `jiangzhuo9357/kokoro-v1-0-gguf`, public,
under Apache-2.0. The card was updated at revision
`020112ea74fbd5b4cf5d5f7886e9a009ee572f38` once every profile had CUDA
measurements to report.

The upload reported no bytes transferred, which is content-addressed
deduplication rather than a skipped upload. That is worth checking rather than
trusting, so the repository was read back: all three GGUFs are present at the
right sizes, their LFS SHA-256 digests match the card exactly, and the published
README is byte-identical to the generated one.

## 2026-07-26 — The decoder gap closed, and a listening pass

### Informal listening: no audible defect

The project owner listened to eight of the fifteen cases through a comparison
page that switches between the PyTorch reference and this port mid-playback at
the same offset, across F32, F16, Q8_MIXED, and CUDA, and reported no problem
with any of them.

That is worth recording precisely for what it is and is not. It is one listener,
informally, on eight cases, and it says no audible defect was found. It is not a
quality evaluation in the sense ADR 0017 defers: no rated comparison, no panel,
no score. The package's Validation Level stays `port_validated`. What the pass
does establish is that the measured agreement is not hiding an audible one — a
useful thing to know before shipping, and the reason the page was built.

### The decoder now has a probe of its own

The previous slice recorded that there was no `validate-kokoro-decoder.py`
because the oracle dumped no decoder probe, leaving the decoder covered only end
to end through the waveform. That gap is now closed.

The dumper hooks the generator's final convolution rather than reading the
inverse transform's inputs, because that is where the C++ graph stops: the
magnitude half is still a logarithm and the phase half a pre-sine angle, and
both activations belong with the transform. Probing after them would have
compared a quantity the port never produces.

Adding an artifact changes what validation proves, so `suite_version` goes to 2
and all fifteen cases were re-dumped. The five existing stages reproduce their
previous measurements digit for digit, which is what confirms the re-dump was
faithful rather than merely successful.

### A flat comparison would have measured nothing

The first run reported a maximum absolute difference of 0.61 on the raw rows.
That number is close to meaningless: the log-magnitude rows reach about -69,
which is a magnitude of 1e-30, and a difference of 0.61 between two numerically
silent bins is noise about noise. The probe is therefore compared as the complex
value the inverse transform consumes — exponential and sine applied — with the
raw halves still recorded so a change in the graph's own output stays visible.

Worst case over the suite is 3.97 against a reference peak magnitude of 24.19,
with a mean of 0.027. That is the harmonic source's inherited phase divergence
arriving where it was expected to.

Seven Golden validators are now registered, one per graph stage.

## 2026-07-26 — Stage 7: CUDA backend validation

The family runs on CUDA. All 14 Kokoro unit tests pass on a `dev-dgx-spark`
build, the whole pipeline reaches audio, and the Golden suite was re-run across
all 15 cases on the accelerator.

The guarantee that has to hold across backends does: `pred_dur`, `y_length`,
and `duration.alignment` are exact on CUDA as on the CPU. The duration logits
drift further there — 4.8e-2 worst against the CPU's 2.9e-4 — which is
cross-backend accumulation order, not a defect, and it does not reach the
rounded durations. Waveform correlation with the oracle is 0.98961 on CUDA
against 0.98744 on the CPU, and comparing the two backends directly gives
0.999893.

### The LSTM accumulation check finally ran on CUDA, and nearly did not

The LSTM test's own header has said since it was written that it must run on
every claimed backend, because the accumulation defect it guards against was
correct on the CPU and wrong on CUDA. It had never actually done so: it called
`ggml_backend_cpu_init` directly. It now enumerates the registry.

The first version of that enumeration allowed devices of type CPU or GPU. On
this project's DGX Spark the CUDA device reports as `IGPU` — correct for a
Grace Blackwell part with unified memory — so the allowlist skipped it and the
run still passed, reporting only the CPU. An allowlist of device types is the
wrong shape for this check: the test now excludes only the placeholder type,
and anything else that fails to initialize is a failure rather than a skip.

Had the accumulation strategy still been the broken one, that allowlist would
have hidden it on exactly the hardware it was written for.

The tolerance across backends is 1e-4 rather than the CPU's 1e-5, which is
accumulation order rather than drift.

## 2026-07-26 — Stage 6: Q8_MIXED, and what TTS.cpp does differently

### TTS.cpp splits Kokoro the opposite way

`TTS.cpp` also ports Kokoro and also quantizes it, and its split is the mirror
image of this one. `kokoro_is_quantizable` returns true only for `albert`,
`text_encoder.lstm`, and five named parts of the duration predictor; everything
under the decoder and generator falls through untouched. It excludes the same
small things this port does — voice tensors, biases, gammas, betas, alphas,
embeddings, norms — but quantizes the front end and leaves the audio path alone.

The reason is visible in their compatible-parts list: it contains no
convolution blocks. A convolution kernel's leading axis is the kernel width,
three or seven wide, so it cannot host a 32-wide quantization block without
being packed first. Their split is what GGML can quantize without that work.
Ours is what error propagation allows. Their README also records that Kokoro
quantization was never measured — the quality findings there are about Parler,
and the quantize documentation still says only Parler is supported.

So the disagreement was worth measuring rather than assuming. A spike
reproduced their split here: quantize the front end to Q8_0 and leave the
decoder at the reference dtype.

| probe | this port's split | TTS.cpp's split |
| --- | --- | --- |
| `bert.hidden` | 1.70e-5 | 1.37e-1 |
| `text.d_en` | 3.59e-5 | 2.64e-1 |
| `duration.logits` | 2.86e-4 | 2.72e0 |
| `pred_dur` | exact | 6 of 78 tokens differ |
| `y_length` | 198, exact | 200 |

The last row is the one that decides it. The utterance comes out 1,200 samples
— 50 ms — longer than the reference, which means every downstream probe has a
different shape and tensor parity stops being defined at all. The Golden
manifest declares the durations a structural check, so that split cannot pass
this suite. It is a reasonable choice for a project whose validation is
perceptual; it is not one for a project that claims tensor parity.

The spike also broke on `bert.layer.full_layer_layer_norm.weight`, which slipped
through a `token == "LayerNorm"` exclusion and was quantized into an elementwise
multiply. That is the failure mode substring rules have: TTS.cpp's are
suffix-based (`!name.ends_with("norm")`), and a name that spells the same thing
differently walks straight past. This port's classifier matches exhaustively on
token count and exact token names, so an unrecognised name is Unknown — an
error — rather than silently defaulted. The whole-package validator added in the
previous slice reported the mismatch precisely, where GGML had aborted.

### The profile itself

216.1 MB against 352.8 MB, a 39% reduction. Every stage through the harmonic
source is unchanged from F32, and the durations and alignment stay exact,
because everything quantized is downstream of them. Worst-case waveform
correlation over the suite is 0.98479 against F32's 0.98744, with spectrogram
correlation 0.99780 against 0.99814.

Three things had to be built:

- The convolution builder now consumes a packed kernel. A block-quantized
  kernel is stored flattened to `[kernel * in_channels, out_channels]`, so the
  input supplies the channel count and a separate shape tensor drives the column
  extraction, the same construction the VITS family uses.
- Twelve decoder matrices, 71.2 MB, cannot host blocks at all: the decoder
  concatenates the two prosody curves and the narrow encoder residual onto its
  feature stream, which lands channel counts like 1090, 514, and 1028 exactly
  two short of a multiple of thirty-two. They carry the halved type instead of
  being dropped from the profile. The predicate that decides this is shared
  between the tool and the runtime for the same reason the classifier is.
- Anything that needed a kernel's logical width had to stop reading `ne[0]`.
  A packed kernel's leading extent is `kernel * in_channels`, so the generator
  was computing a padding of 1792 and rejecting its own weights. `conv_kernel_size`
  now derives it from the channel count.

One near-miss worth recording: the two-dimensional rule and the block-alignment
fallback were first written to apply to every family, which changed what the
VITS quantizer produced and was caught by its own test. Both are now scoped to
Kokoro, and the VITS packages it produces are byte-identical to before.

## 2026-07-26 — Stage 6: the F16 Quantization Profile

The quantizer was VITS-only by construction: it refused any other architecture
and resolved storage per tensor from a VITS catalog classifier, with no
catch-all rule. That fail-closed shape is worth keeping, so Kokoro got its own
classifier rather than a relaxation.

The classifier lives in the family module, at `src/arch/kokoro/quantization.h`,
and the offline tool calls it. VITS keeps this knowledge in two places — the
tool's policy and the runtime's catalog — and for a 511-tensor family two
hand-maintained lists would drift silently, producing packages that simply fail
to load. One source of truth removes the possibility.

### Where the split falls, and why

Measured parameter mass: the decoder body and its generator are 60.4% of the
package, and everything upstream of them — PL-BERT, the prosody and duration
path, the acoustic text encoder, the Voice tables — is 39.6%.

That upstream part is exactly what decides F0 and the durations, and this
port has already measured what a small F0 difference costs: the harmonic
source accumulates phase across the whole utterance, so a few parts in ten
thousand becomes radians by the end. So none of it is quantized at any profile.
The split is the same shape as the VITS one, but for a reason this family
measured rather than inherited.

Three further cases needed their own rule:

- The Snake alphas are divided by, so they stay exact.
- The depthwise pools are transposed convolutions, but this runtime reads their
  taps one at a time as a per-channel scale, which only works at the reference
  dtype. They are Sensitive, not TransposeWeight, despite the name.
- `decoder.F0_conv` and `decoder.N_conv` halve the prosody curves before
  anything else reads them, so they sit on the F0 path rather than the audio
  path.

### Storage types are now checked over the whole package

The catalog used to check each tensor's type where it looked it up, and only
accepted F32. It now accepts what the profile implies, and the check moved to a
single pass over every tensor in the package. That is stronger than the
per-lookup form: it also rejects a tensor the catalog never looks up, and a name
outside the catalog, rather than ignoring either.

### Result

The F16 package is 246.5 MB against 352.8 MB, a 30% reduction, and costs
nothing measurable. Every stage up to and including the harmonic source
reproduces the F32 measurements **exactly**, digit for digit, because none of
those stages is quantized. Only the waveform differs, and by less than the
spread between cases: worst-case correlation with the oracle is 0.987445 at F16
against 0.987437 at F32, and comparing the two packages directly gives
correlation 0.999999 with a maximum absolute difference of 8.5e-4.

Q8_MIXED is not yet produced. It needs the block-packed matrix layout, which
this family's convolution builder does not yet consume — VITS handles it by
storing the kernel pre-packed as a 2-D tensor and driving `ggml_im2col` from a
separate shape tensor. That is the next slice.

## 2026-07-26 — Family dispatch in the public interface

Kokoro is now reachable through `synth_model_load`. Until this slice the public
entry point named `synth::vits::Model` directly, so the family that had passed
port validation could not actually be used through the product surface.

The core runtime now reads a family-independent `synth::ModelInfo`, declared in
`src/model-info.h`. Each family keeps its own info struct for the quantities
only its own graphs need — VITS's latent channel count and noise scales — while
everything the public interface, the request validator, and the audio delivery
path read has one shape. `prepare_synthesis_request` took `vits::ModelInfo` and
now takes the shared one; it read nine fields and every one of them was already
family-independent, which is what made the split cheap.

`synth_model_load` reads `general.architecture` before choosing a loader, so a
package that names no architecture, or one this build has no family for, is
refused at the seam rather than by whichever loader happened to be tried first.
That probe had to be placed after the file-existence check: it runs before any
family loader, so it now owns the distinction between a missing file and an
unreadable one, and the public API test caught it doing otherwise.

Kokoro's synthesis path needed a seeded entry point of its own. The harmonic
source's two random draws are sized from the resolved frame count, which is not
known until the duration stage has run, so the caller cannot size them. The
family draws them instead, from one seeded stream that supplies the uniform
initial phases and then the Gaussian noise — one stream, so a seed determines
the whole draw. `NormalRandomStream` gained a uniform read for this.

`synthesize-kokoro-public-lifecycle` checks what only the public seam can show:
the declared capabilities match the package contract, all 54 voices are exposed
with no default so a request must name one, a seed repeats exactly, a different
seed changes the audio but not its length, and the speaking rate orders the
output length. Those are the manifest's public-request relations. Measured
through the interface, the upstream case produces 118,800 samples at 24 kHz —
198 duration steps of 600 samples — in 2.8 seconds of wall time on CPU.

## 2026-07-26 — Stage 5: per-stage validators and measured tolerances

Six registered validators now drive the runner over all 15 manifest cases in
the locked environment, under the `integration`, `kokoro`, and `golden` labels.
They share `scripts/kokoro_validation_common.py`, because the stages take
identical arguments and differ only in which probes they compare; putting the
case resolution, runner invocation, and report writing in one place is what
keeps six scripts from drifting apart.

Worst case across the whole suite, source-F32 on CPU:

| probe | max abs | mean abs |
| --- | --- | --- |
| `text.t_en`, `text.asr` | 2.53e-6 | 9.9e-8 |
| `bert.hidden` | 1.70e-5 | 1.40e-6 |
| `prosody.n` | 2.17e-5 | 1.72e-6 |
| `duration.d`, `prosody.en` | 3.45e-5 | 4.5e-7 |
| `text.d_en` | 3.59e-5 | 3.74e-6 |
| `duration.logits` | 2.86e-4 | 1.13e-5 |
| `prosody.f0` | 2.58e-3 | 7.82e-5 |
| `source.har` (complex) | 9.50e-2 | 2.15e-3 |
| `audio.pcm` | correlation ≥ 0.9874, spectrogram correlation ≥ 0.9981 |

`duration.pred_dur`, `duration.y_length`, and `duration.alignment` are exact on
every case. They are declared structural, so the validator fails outright on a
difference rather than recording one: they decide the output length, and a
length that is merely close is a different utterance.

Two probes needed comparison rules of their own rather than a wider threshold.
`source.har` is compared as a complex value, since the stored phase is
meaningless wherever its magnitude is near zero. `audio.pcm` is reported as
correlation and spectral distance, because the excitation phase is a cumulative
sum over the utterance and therefore chaotic in F0 by construction — a
sample-wise threshold on it would be a number with no meaning attached.
`tests/tolerances/kokoro.json` now records all of this as measurements with
thresholds still deferred; no placeholder acceptance number is claimed.

### Gap recorded rather than papered over

There is no `validate-kokoro-decoder.py`, because the oracle dump contains no
decoder probe — the reference records the source spectrum and then the audio.
The decoder is covered end to end by the waveform stage, but not stage-locally.
Closing that needs a new probe in the dumper, which changes the artifact set and
so bumps `suite_version`; it is deliberately not folded into this slice.

## 2026-07-26 — First end-to-end run on the real package

Added `tests/kokoro_stages_real.cpp`, the runner the Stage 5 validators will
drive, and ran all seven stages against the converted GGUF for the
`kokoro-upstream-default` case. The pipeline reaches audio, and the structural
outputs are **bit-exact**: `pred_dur`, `y_length`, and the alignment all match
the oracle exactly, so the output length is not merely close but identical.

### Two defects the unit tests could not have caught

- The decoder built its feature stream 64 channels wide instead of 512.
  Upstream's `Decoder` names its own constructor argument `dim_in` and is
  passed 512, while the configuration carries a separate top-level `dim_in` of
  64. Reading the wrong one is a silent shape error, and the unit fixture had
  set `dim_in` for the decoder's width, so the two names agreed there by
  accident. The fixture now sets `hidden_dim` and deliberately leaves `dim_in`
  unset.
- The scheduler's hash set was sized from the graph's node capacity alone. It
  must also cover the leaves, which are every weight the graph reads — a few
  hundred for the decoder. The reduced-dimension unit test has few enough
  weights that it never crossed the limit.

### Measured agreement, and where it comes from

| probe | before | after |
| --- | --- | --- |
| `bert.hidden` | 1.72e-3 | 1.58e-5 |
| `text.d_en` | 2.84e-3 | 3.31e-5 |
| `duration.logits` | 1.39e-2 | 2.79e-4 |
| `prosody.f0` | 1.13e-1 | 6.54e-4 |
| `source.har` (complex) | 1.25e-1 | 3.04e-2 |
| waveform correlation | 0.99721 | 0.99821 |

`text.t_en` and `text.asr` agree to 2.3e-6 throughout, and `duration.alignment`
is exact.

The first column is what GGML's CPU GELU costs. Its half-precision lookup table
leaves about 1e-4 per activation, which compounds across PL-BERT's twelve
replayed layers and reaches F0. `GGML_GELU_FP16` is hard-coded in the vendored
snapshot and the snapshot is never hand-edited, so the fix is to evaluate the
`gelu_new` closed form from single-precision primitives instead — the same
reasoning that already keeps convolution off `ggml_conv_1d`. That is the second
column.

### The harmonic source is ill-conditioned in F0, and this is not a defect

Three controlled runs separate the source's own accuracy from what it inherits:

| comparison | complex absmax | mean |
| --- | --- | --- |
| upstream driven by the oracle's F0, against the oracle probe | 0 | 0 |
| upstream driven by **our** F0, against upstream driven by the oracle's | 1.23e-1 | 4.19e-3 |
| our C++ source driven by our F0, against upstream driven by the same F0 | 1.75e-2 | 9.4e-4 |

The first line confirms the replay harness reproduces the oracle exactly. The
second shows that feeding our F0 into the *unmodified upstream module* produces
almost the entire divergence, so it is inherited rather than introduced. The
reason is structural: the sine phase is a cumulative sum over the whole
utterance scaled by 300, so a 4e-4 relative difference in F0 becomes radians of
phase by the end. The third line is our own remaining float-ordering difference
in that same chaotic accumulation, an order of magnitude below the inherited
term.

Two consequences for Stage 5. `source.har`'s phase cannot carry a tight
tensor-parity tolerance, and must be compared as a complex value or weighted by
magnitude, since bins with near-zero magnitude have meaningless phase. And
waveform agreement has to be stated as a correlation and spectral distance
rather than sample-wise: the current run is 0.9982 correlated with the oracle
at 0.9992 spectrogram correlation, which is the right shape of claim for a
model whose excitation phase is chaotic by construction.

## 2026-07-26 — Stage 4 slice 11: family facade

Added `kokoro.h` and `model.cpp`: package loading and one entry point per
inference stage, mirroring what the VITS family already exposes. The staged
shape is what the Stage 5 validators drive — each stage recomputes its
predecessors so a validator can compare one stage against the oracle without
the family holding synthesis state between calls.

The seven stages are PL-BERT with its projection, the duration path and its
host seam, prosody, the acoustic text encoder, the harmonic source, the
decoder, and the inverse transform. Alignment expansion happens on the host
between stages, as the matrix product upstream writes as `d @ pred_aln_trg`.

Two layout rules had to be kept apart, and conflating them would have been a
silent wrong answer rather than a crash. Graph tensors run feature-fastest;
the reference probe artifacts run time-fastest. The staged outputs are
converted to the probe layout at the boundary, but the duration host seam
reads the graph's own layout, one token's bins at a time, so its logits are
passed through untransposed.

`stream_tensor_data` moved out of the VITS model into the shared GGUF helper.
Both families fill their tensors the same way and only differ in which tensors
they declare, so there was no reason for a second copy.

An end-to-end synthetic package was considered for the test and rejected: the
decoder's widths are fixed at 1024 upstream, so even a reduced-dimension
package is on the order of a hundred megabytes of weights. The unit test
therefore covers load error mapping, where the discrimination matters most —
an empty or foreign container reports an unsupported architecture rather than
a generic container error, which is what lets a dispatcher above the family
tell "not mine" from "corrupt". Numerical agreement is Stage 5's.

## 2026-07-26 — Stage 4 slice 10: decoder, iSTFTNet generator, inverse transform

Added `generator.{h,cpp}`, `decoder.{h,cpp}`, and `decoder-host.{h,cpp}`, which
completes the family's inference path from tokens to samples.

Three operations were missing and were added to `operations.{h,cpp}`: a stride
on the existing convolution, a dense transposed convolution, and the
single-frame reflection pad. The transposed convolution reuses the construction
the depthwise one documents — zero-interleave, widen, then one shifted matrix
multiply per tap — and gets the definition's kernel reversal by reading the taps
back to front, so the stored weights still match the checkpoint byte for byte.
Its reference was cross-checked against `F.conv_transpose1d` before being
trusted, because the weight axis order is the easy thing to get wrong: PyTorch
stores `[in, out, kernel]`, which GGML reports as `[kernel, out, in]`, the
reverse of an ordinary convolution's.

Two upstream details that a careless port would flatten:

- The generator's per-stage activation uses a slope of 0.1 and the one before
  the final projection uses PyTorch's default of 0.01. They are written
  differently upstream and are not interchangeable.
- `AdaINResBlock1` and `AdainResBlk1d` are different blocks despite the nearly
  identical names. The generator's is three Snake branches added straight back
  onto the running value; the decoder's is a two-convolution residual with a
  projection shortcut and a variance scale. Both now exist and are named apart.

The frame arithmetic is what ties the stage together, and the test pins it: the
upsamplers multiply the length by the product of the rates, the last stage's
reflected frame adds one, and that extra frame is exactly what makes the
features and the source spectrum the same length. The noise convolutions'
strides are chosen so each stage's resampled spectrum matches its own rate; the
test asserts the equality at every stage rather than only at the end.

The inverse transform is the third host seam, and it is checked against
`torch.istft` from the raw form the graph emits, so the exponential and the sine
that turn the output into a magnitude and a phase are part of what is verified.
Agreement is 1e-5.

The decoder's numerical agreement with the oracle is deliberately left to Stage
5, where the real checkpoint and the recorded reference tensors exist. What the
unit test covers is what the oracle cannot catch cheaply: the frame alignment,
the node budget, and the catalog shapes that must be rejected rather than
reinterpreted.

### A test that was passing because its exit code was discarded

Piping a test binary through `tail` to read its output made `$?` the pipe's
status, so a segmentation fault read as a pass. The fault was real: the host
reference indexed the second convolution's weights as `base + 2 + 8 * branches`,
which is out of range for every branch. Two further defects were behind it — the
Snake alphas are stored as `[1, channels, 1]` and the test built them flat, and
the upsampler weights were built with an ordinary convolution's axis order. All
three were found only once the exit code was read directly.

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
