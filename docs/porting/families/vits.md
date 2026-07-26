# VITS Port Validation Plan

Status: `vits-ljspeech` and `vits-vctk` F32/F16/Q8_MIXED functional ports
validated on 2026-07-23.

## Reference Contract

The oracle is the pinned `jaywalnut310/vits` PyTorch implementation. Its
[inference notebook](https://github.com/jaywalnut310/vits/blob/main/inference.ipynb)
defines the upstream LJSpeech and VCTK examples with `noise_scale=0.667`,
`noise_scale_w=0.8`, and `length_scale=1`. The public VITS inference function uses
separate stochastic duration and acoustic-latent draws, so the oracle exports them
as `duration_noise.f32` and `latent_noise.f32` for internal parity replay.

Both initial variants produce 22,050 Hz mono F32 PCM and use the upstream
`english_cleaners2` path to resolve Golden source text. The manifest stores the
resolved token IDs as the actual core-graph input. Published packages additionally
accept the corresponding UTF-8 phoneme sequence through the built-in
`synthesize.symbol_map` frontend without claiming that the core contains the
upstream eSpeak-based G2P path.

The package fixes `noise_scale=0.667` and `noise_scale_w=0.8` as Model
Variant inference hyperparameters rather than exposing VITS-specific fields in the
public C Interface. Public `speaking_rate = r` maps to upstream
`length_scale = 1 / r`. The initial validated speaking-rate range is `0.80` through
`1.25`; expanding it requires new cases and a suite-version change.

## Required Probe Set

Every VITS parity case records these logical probes with stable project names:

```text
input.token_ids
text.m_p
text.logs_p
text.mask
duration.logw
duration.w_ceil
duration.y_length
duration.attention
prior.m_p_expanded
prior.logs_p_expanded
latent.z_p
flow.z
audio.pcm
```

VCTK additionally records `voice.embedding`. Integer token IDs, masks, rounded
durations, lengths, and attention paths are exact structural probes. Remaining
F32 tensors and PCM use `tests/tolerances/vits.json`.

## LJSpeech Variant

`vits-ljspeech` has one unnamed fixed package-default Voice. All cases use language
`en`, oracle-resolved token IDs, `noise_scale=0.667`, `noise_scale_w=0.8`, and no
caller output-frame limit.

| Case ID | Source text | Seed | Rate | Coverage |
| --- | --- | ---: | ---: | --- |
| `ljs-upstream-default` | `VITS is Awesome!` | 0 | 1.00 | Exact upstream example, baseline, repeatability |
| `ljs-minimal` | `A.` | 0 | 1.00 | Minimal resolved sequence and padding |
| `ljs-short` | `Hello, world!` | 0 | 1.00 | Short sequence |
| `ljs-punctuation` | `Wait... really? Yes!` | 0 | 1.00 | Punctuation-shaped token sequence |
| `ljs-normalization` | `Dr. Smith paid $12.50 on January 3rd.` | 0 | 1.00 | Upstream normalization-shaped token sequence |
| `ljs-medium` | `The quick brown fox jumps over the lazy dog.` | 0 | 1.00 | Medium sequence and broad symbol use |
| `ljs-repetition` | `No, no, no; yes, yes, yes.` | 0 | 1.00 | Repeated tokens and durations |
| `ljs-long` | `A small speech synthesis library should produce the same finite waveform whenever its model, input, controls, and random seed are unchanged.` | 0 | 1.00 | Long sequence and allocation growth |
| `ljs-seed-one` | `VITS is Awesome!` | 1 | 1.00 | Alternate stochastic path |
| `ljs-seed-forty-two` | `VITS is Awesome!` | 42 | 1.00 | Second alternate stochastic path |
| `ljs-rate-slow` | `The quick brown fox jumps over the lazy dog.` | 0 | 0.80 | Slow-rate boundary; upstream length scale 1.25 |
| `ljs-rate-fast` | `The quick brown fox jumps over the lazy dog.` | 0 | 1.25 | Fast-rate boundary; upstream length scale 0.80 |

In the `public_request` phase, the manifest relates the three upstream-text seed
cases with `artifact_differs(audio.pcm)`. It relates `ljs-rate-slow`, `ljs-medium`,
and `ljs-rate-fast` with strictly decreasing output frame count. The
upstream-default case performs public same-seed repeatability in addition to parity
replay.

## VCTK Variant

The upstream configuration declares 109 speaker embeddings. synthesize.cpp exposes
the stable IDs `speaker-000` through `speaker-108`, mapped exactly to upstream
indices 0 through 108. No Preset Voice is invented as the package default; every
request selects one explicitly. All cases otherwise use the same language,
resolved-input, hyperparameter, and output-limit rules as LJSpeech.

| Case ID | Source text | Voice | Seed | Rate | Coverage |
| --- | --- | --- | ---: | ---: | --- |
| `vctk-upstream-speaker-004` | `VITS is Awesome!` | `speaker-004` | 0 | 1.00 | Exact upstream example, baseline, repeatability |
| `vctk-speaker-first` | `VITS is Awesome!` | `speaker-000` | 0 | 1.00 | First embedding-table entry |
| `vctk-speaker-middle` | `VITS is Awesome!` | `speaker-054` | 0 | 1.00 | Middle embedding-table entry |
| `vctk-speaker-last` | `VITS is Awesome!` | `speaker-108` | 0 | 1.00 | Last embedding-table entry |
| `vctk-seed-one` | `VITS is Awesome!` | `speaker-004` | 1 | 1.00 | Alternate stochastic path |
| `vctk-seed-forty-two` | `VITS is Awesome!` | `speaker-004` | 42 | 1.00 | Second alternate stochastic path |
| `vctk-rate-slow` | `The quick brown fox jumps over the lazy dog.` | `speaker-004` | 0 | 0.80 | Slow-rate boundary |
| `vctk-rate-fast` | `The quick brown fox jumps over the lazy dog.` | `speaker-004` | 0 | 1.25 | Fast-rate boundary |
| `vctk-minimal` | `A.` | `speaker-004` | 0 | 1.00 | Minimal sequence and padding |
| `vctk-punctuation` | `Wait... really? Yes!` | `speaker-004` | 0 | 1.00 | Punctuation-shaped token sequence |
| `vctk-normalization` | `Dr. Smith paid $12.50 on January 3rd.` | `speaker-004` | 0 | 1.00 | Upstream normalization-shaped token sequence |
| `vctk-long` | `A small speech synthesis library should produce the same finite waveform whenever its model, input, controls, and random seed are unchanged.` | `speaker-004` | 0 | 1.00 | Long sequence and allocation growth |

The manifest requires exact `voice.embedding` parity for every speaker case and,
in the `oracle_replay` phase, relates their embedding artifacts with
`artifact_differs`. Seed cases and rate cases use the same `public_request`
relations as LJSpeech. `vctk-upstream-speaker-004` names the upstream example, not
a package-default Voice.

## Materialization Rule

Each local JSON manifest is created only after the exact repository commit,
checkpoint bytes, config bytes, and reference environment have been resolved and
the license status has been audited. Missing separate checkpoint terms remain
visible in the Intake and model card. Under the current maintainer policy, their
absence is treated as no additional restriction rather than an automatic
publication blocker. SHA-256 values must be measured from acquired files;
placeholders or hashes copied from an unverified mirror are forbidden. No
LJSpeech or VCTK dataset archive is needed to materialize these cases.

The `vits-ljspeech` Intake is recorded in
[`reports/porting/vits/vits-ljspeech/intake.json`](../../../reports/porting/vits/vits-ljspeech/intake.json).
Its source, checkpoint, configuration, frontend, and CPU oracle smoke are pinned
and verified. The checkpoint has no explicit weight-license statement; this is
disclosed but is not a blocker under the current maintainer publication policy.

The corresponding multi-speaker Intake is
[`reports/porting/vits/vits-vctk/intake.json`](../../../reports/porting/vits/vits-vctk/intake.json).
It records the 109-speaker catalog, 256-channel conditioning topology, converter
accounting, CPU validation, and CUDA 13.3 validation on both physical Linux
hosts. Its checkpoint has the same missing separate terms. The generated
publication payload is available from
[`jiangzhuo9357/vits-vctk-gguf`](https://huggingface.co/jiangzhuo9357/vits-vctk-gguf)
at revision `eb27a41f7a01dbe3ec88dabcd834844e9383a5e7`.

## Source-F32 Conversion Contract

`scripts/convert-vits.py` is the VITS checkpoint adapter. Its complete external
interface is `--manifest`, `--source-dir`, `--checkpoint`, and `--outfile`. The
manifest selects the supported Model Variant and pins its artifacts; callers do
not control individual tensor mappings, layouts, metadata, or quantization. The
converter accepts the pinned `vits-ljspeech` and `vits-vctk` variants from that
source revision and produces only the `F32` profile. Variant selection comes
from the manifest; it is not a language switch. Later F16 and mixed-quantization
packages remain the responsibility of the C++ quantizer.

Run it through the locked family environment:

```bash
uv run --project scripts/envs/vits --locked python scripts/convert-vits.py \
  --manifest tests/golden/vits/vits-ljspeech.manifest.json \
  --source-dir models/upstream/vits-source \
  --checkpoint models/upstream/vits-ljspeech/pretrained_ljs.pth \
  --outfile models/vits-ljspeech/vits-ljspeech-F32.gguf
```

The VCTK conversion uses the same adapter and contract:

```bash
uv run --project scripts/envs/vits --locked python scripts/convert-vits.py \
  --manifest tests/golden/vits/vits-vctk.manifest.json \
  --source-dir models/upstream/vits-source \
  --checkpoint models/upstream/vits-vctk/pretrained_vctk.pth \
  --outfile models/vits-vctk/vits-vctk-F32.gguf
```

The package hard-limits the final post-frontend sequence to 512 token IDs and
native output to 1,323,000 frames, or 60 seconds at 22,050 Hz. The current suite
reaches 315 token IDs and 211,968 frames. Raising either package limit requires
expanding the cases and incrementing the suite version; these values are
allocation-safety contracts, not statements about the architecture's theoretical
context length.

The 838 checkpoint entries normalize to 698 logical tensors. PyTorch
weight-normalization pairs are fused with the upstream dimension-0 formula before
writing: 140 pairs are verified, of which 108 belong to inference tensors and 32
belong to skipped training-only paths. The primary GGUF contains exactly 460
runtime tensors under four canonical namespaces:

| Namespace | Tensors | Runtime role |
| --- | ---: | --- |
| `text_encoder` | 111 | token embedding, relative-attention blocks, FFN, prior projection |
| `duration_predictor` | 114 | stochastic-duration reverse inference |
| `flow` | 80 | reverse acoustic residual-coupling flow |
| `decoder` | 155 | transposed-convolution upsampler and HiFi-GAN residual blocks |

Exactly 238 logical tensors are omitted with an explicit report entry: 68 from
the training-only posterior encoder, 142 from the stochastic-duration posterior
path, and 28 from the first ConvFlow removed by upstream reverse inference. The
report accounts for all 838 original checkpoint entries, including both source
entries of every fused tensor; an unknown, missing, unpaired, non-finite, or
colliding tensor aborts conversion.

Tensor payloads need no explicit transpose. `gguf-py` reverses the declared NumPy
shape into GGML dimension order while preserving contiguous source bytes. Thus a
PyTorch Conv1d `[out, in, kernel]` is declared to GGML as
`[kernel, in, out]`, and a PyTorch ConvTranspose1d `[in, out, kernel]` as
`[kernel, out, in]`. The converter reopens the completed file and verifies every
tensor name, F32 type, GGML shape, and payload byte against the prepared source
array.

The GGUF embeds the pinned 178-entry upstream symbol array and declares
`synthesize.symbol_map` contract version 1 with `unicode_scalar` mapping. Upstream
contains ASCII apostrophe at both IDs 174 and 176; the array is preserved exactly
and declares the upstream `last_index_wins` lookup policy instead of renumbering
the embedding. The private Text Frontend validates UTF-8, maps phoneme scalars,
and applies the package blank-insertion rule. Raw-text G2P remains external.
LJSpeech capabilities are phoneme plus token input, fixed unnamed Voice, English
default with regional fallback, stochastic synthesis, speaking rate 0.80 through
1.25, and 22,050 Hz mono F32 output. VCTK keeps the same graph-level capabilities
but requires one stable Preset Voice ID from `speaker-000` through `speaker-108`
for every request.

On the DGX Spark, two independent finalized conversions produced the identical
113,245,056-byte GGUF SHA-256
`bd17e44c7c2d761d33c1527059bd3921f9d3fa9d46bd73b3e020d746a8b7db7b`.
The generated model and detailed 460-tensor converter report remain ignored under
`models/` and `reports/convert/`. This establishes conversion integrity only; it
does not establish C++ graph parity, perceptual quality, or publication rights.

The VCTK checkpoint normalizes from 858 entries to 713 logical tensors. Its GGUF
contains 473 runtime tensors: the same 111-tensor text encoder plus 116 duration,
88 flow, 157 decoder, and one speaker-embedding tensor. The additional tensors
are precisely the upstream global-conditioning projections. Two formal-manifest
conversions produced the identical 120,407,552-byte file with SHA-256
`b9e69b257cc600679a45e4197614f36ec678156180e2d366f7e1dcb6668b2be0`.

## CPU Library Path Through the Public C Interface

Stage 4 uses deliberately bounded CPU slices in `src/arch/vits/`. The
library module owns GGUF validation, the 111-tensor text catalog, the 114-tensor
duration catalog, the 80-tensor acoustic-flow catalog, the 155-tensor decoder
catalog, backend weight allocation, graph allocation, input upload, execution,
and output layout. Validation runners link this module; they do not contain a
second inference implementation. The CPU path now reaches final PCM through the
public `synth_synthesize()` operation and its `synth_synthesize_to_buffer()`
adapter. LJSpeech accepts its fixed unnamed default Voice. VCTK accepts an
explicit Preset Voice ID and reports that resolved ID in `synth_result_t`. Both
accept UTF-8 phonemes through the embedded symbol map or exact token IDs, `en` or
a permitted English regional fallback, a concrete or random seed, and the
validated speaking-rate range. Raw text is rejected until a compatible optional
G2P provider exists.

The root build vendors ggml commit
`707321c4cf6d21cb4bc831aa8b687dbf01a521ce`, matching the pinned transcribe.cpp
snapshot, and defaults to a static CPU build with the native GGML thread pool.
CUDA, Metal, and Vulkan remain build options over the same graph. The public
device registry wrapper enumerates those runtime devices independently. VITS now
resolves strict CPU and CUDA registry selections before allocation and reports
the device actually attached to the loaded model backend; `AUTO` remains on the
Supported CPU path. Metal and Vulkan have not entered family validation. A
model-owned Backend Plan
now provides weight placement, scheduler backend order, CPU fallback ownership,
and registry-based thread configuration to every graph path; no VITS execution
function constructs a CPU-only scheduler directly. The model-independent gate
covers the C Interface, request preparation, the public random stream, Preset
Voice selection, device classification and discovery, strict CPU selection,
Loaded Model device queries, Audio Sink delivery, relative indices, output layout,
metadata, all tensor catalogs, graph construction and CPU execution, model-load
errors, manifest contracts, weight normalization, tensor mapping, atomic output,
seed parsing, and artifact-path containment. Run the model-independent gate with:

```bash
cmake -S . -B build \
  -DSYNTH_METAL=OFF -DSYNTH_CUDA=OFF -DSYNTH_VULKAN=OFF \
  -DSYNTH_BUILD_TESTS=ON -DSYNTH_BUILD_PYTHON_TESTS=ON
cmake --build build --target synthesize-check-unit
```

The real model comparison is a separate, explicitly enabled CTest:

```bash
cmake -S . -B build \
  -DSYNTH_METAL=OFF -DSYNTH_CUDA=OFF -DSYNTH_VULKAN=OFF \
  -DSYNTH_BUILD_TESTS=ON -DSYNTH_BUILD_INTEGRATION_TESTS=ON \
  -DSYNTH_VITS_TEST_MODEL="$PWD/models/vits-ljspeech/vits-ljspeech-F32.gguf" \
  -DSYNTH_VITS_VCTK_TEST_MODEL="$PWD/models/vits-vctk/vits-vctk-F32.gguf"
cmake --build build --target synthesize-check-integration
```

The pinned GGML `ggml_conv_1d` wrapper forces an F16 im2col buffer and asserts on
F32 kernels. The VITS graph therefore uses transcribe.cpp's established
F32-preserving `im2col + mul_mat` decomposition; it does not downcast the accuracy
artifact. Relative key and value attention use a padded nine-row embedding table,
a checked distance-index tensor, and batched GGML matmuls. No family-specific CPU
operator was added, so the semantic graph remains backend-independent.

All 12 cases complete with finite, correctly shaped outputs. `text.mask` is exact
in every case. The worst observed F32 CPU drift is:

| Probe | Worst case | Max abs | Mean abs |
| --- | --- | ---: | ---: |
| `text.m_p` | `ljs-long` | 1.2636185e-5 | 1.0475525e-7 |
| `text.logs_p` | `ljs-long` | 4.7087669e-6 | 4.5068868e-8 |
| `text.mask` | all exact | 0 | 0 |
| `duration.logw` | `ljs-long` | 1.7452985e-5 | 1.2574333e-6 |
| `duration.w_ceil` | all exact | 0 | 0 |
| `duration.y_length` | all exact | 0 | 0 |
| `duration.attention` | all exact | 0 | 0 |
| `prior.m_p_expanded` | `ljs-long` | 1.2636185e-5 | 1.0283067e-7 |
| `prior.logs_p_expanded` | `ljs-long` | 4.7087669e-6 | 4.5241178e-8 |
| `latent.z_p` | `ljs-rate-slow` | 1.5258789e-5 | 1.3051576e-7 |
| `flow.z` | `ljs-rate-slow` | 6.6339970e-5 | 1.1114858e-6 |
| `audio.pcm` | `ljs-long` | 2.8620008e-4 | 1.7319487e-6 |

These are measurements, not accepted thresholds. `tests/tolerances/vits.json`
records that the complete source-F32 CPU measurements exist while threshold and
quality decisions remain deferred under ADR 0017. The current Release CPU gate
passes 43/43 tests, including the locked 93-case Python collection. All ten
LJSpeech integration tests pass. That integration set consists of
two public C Interface tests, one reference-CLI test, and seven full 12-case
Golden paths. The current 43/43 unit gate and real VCTK F32, F16, and Q8_MIXED
public synthesis tests pass under AddressSanitizer plus
UndefinedBehaviorSanitizer.
Inputs of 513 tokens and token ID 178 are rejected before graph execution,
exercising the package limit and vocabulary bound.

## VCTK Multi-Speaker Checkpoint

The private `voice-conditioning` module owns speaker lookup for every backend.
An I32 graph input selects one row from the 109-by-256 embedding table. The
duration predictor adds its projected condition after the duration
pre-convolution, each reverse acoustic-flow block consumes its own WN condition
slice, and the decoder adds its projected condition after its pre-convolution.
This follows the pinned upstream graph without exposing a VITS speaker index in
the public C Interface.

The registered `synthesize-vits-vctk-public` test verifies catalog endpoints,
missing and invalid Voice errors, resolved Voice metadata, same-seed
repeatability, and different-speaker output. Seven additional registered tests
replay all 12 cases through each graph stage. All 8/8 pass on CPU. The duration
path is structurally exact in every case; CPU final PCM worst max-absolute drift
is `1.7652847e-4`.

CUDA 13.3 strict-FP32 validation also passes all seven 12-case stages on DGX
Spark GB10 (`sm_121a`) and RTX 4070 SUPER (`sm_89`), with exact duration
structure and final PCM worst max-absolute drift of `4.3350458e-4` and
`4.7563761e-4` respectively. The aggregate record is
`reports/validate/vits/vits-vctk-cpu-cuda-13.3-linux.json`; per-stage reports use
the `-cuda-13.3-dgx-spark` and `-cuda-13.3-rtx4070super` suffixes.

Public seeded synthesis deliberately uses synthesize.cpp's stable random stream.
Parity runners instead replay PyTorch's recorded stochastic inputs, so equal
numeric seeds are repeatable within each runtime but are not specified to produce
PyTorch-identical waveforms. Perceptual quality evaluation remains deferred under
ADR 0017.

### LJSpeech F16 and Q8_MIXED packages

The C++ `synthesize-quantize` tool derives both packages directly from the
source-F32 GGUF. F16 version 1 stores 118 flow/decoder weights in F16 and keeps
342 duration-sensitive or scalar tensors in F32. Q8_MIXED version 1 stores 114
ordinary flow/decoder convolution matrices in Q8_0, four transpose-convolution
weights in F16, and the same 342 tensors in F32.

The packages are 70,453,568 and 52,890,240 bytes with SHA-256 values
`5fc428ba97416cc164055f509af1b9e120b089bd0bab792ad674c862411d7bca` and
`df95091f975e78088c4908c3f2adfc381cfa234c3f520ddd3ffe10160ff12ba1`.
Both are deterministic across two complete quantizer runs.

All seven 12-case stages pass on one-thread CPU, DGX Spark CUDA 13.3, and RTX
4070 SUPER CUDA 13.3. Duration structure is exact on every platform. F16 worst
final-PCM max-absolute drift is `0.01713`, `0.20996`, and `0.20820`; Q8_MIXED
records `0.36680`, `0.32943`, and `0.34679`. Both CUDA placement logs contain
zero executable CPU fallback. These are functional measurements; tolerance and
perceptual-quality acceptance remain deferred.

### VCTK F16 and Q8_MIXED packages

The C++ `synthesize-quantize` tool derives both packages directly from the
source-F32 GGUF. F16 version 1 stores 123 flow/decoder weights in F16 and keeps
350 duration-sensitive or scalar tensors in F32. Q8_MIXED version 1 stores 119
ordinary flow/decoder convolution matrices in Q8_0, four transpose-convolution
weights in F16, and the same 350 tensors in F32. The runtime validates the
profile name, version, `general.file_type`, exact tensor type, and native or
packed shape before allocating model buffers.

The packages are 74,208,224 and 55,047,392 bytes with SHA-256 values
`ff11efb1106834efb3609647e68642b48a58dbbdbabbc776d3afd83cf46085af` and
`149438d3a6c817ca6e4ab803a67207a41f62097cc8f586520355610801fb0542`.
Both are deterministic across two complete quantizer runs.

All seven 12-case stages run on one-thread CPU, DGX Spark CUDA 13.3, and RTX
4070 SUPER CUDA 13.3. Duration structure is exact on every platform and the
public multi-speaker API remains deterministic for a fixed request. Both CUDA
placement logs contain zero executable CPU fallback. F16 worst final-PCM
max-absolute drift is `0.06753` on CPU, `0.25651` on GB10, and `0.19204` on RTX.
Q8_MIXED records `0.77982`, `0.57463`, and `0.62478` respectively. These are
measurements with no finalized tolerance; perceptual and comparative quality
evaluation remains explicitly deferred.

## CUDA Experimental Checkpoint

The complete VITS graph also runs through the public explicit-CUDA path on the
DGX Spark development machine. This checkpoint uses CUDA toolkit 13.0.88, driver
580.159.03, GB10 compute capability 12.1, CMake 3.28.3, and native `sm_121a` code.
It is Experimental evidence only: the release Provider remains locked to CUDA
13.3 Update 1 and is not qualified by this local toolkit.

CUDA F32 matrix multiplies compute at TF32 precision; the strict-FP32 gate that
once made this configurable was removed on 2026-07-26 (`docs/backends.md`,
`ggml-patches/README.md`). In the 12-case suite, TF32 produced worst
max-absolute differences of `8.6592436e-3` at `text.m_p` and `1.1620114e-1` at
`audio.pcm`, against `1.1280179e-5` and `7.4365083e-4` under the gate. The TF32
figures are what current builds produce.

The table below was measured **under the removed gate** and therefore no longer
describes a shipped configuration. It is retained as the historical strict-FP32
reference until the CUDA grid is re-measured and these rows are replaced:

| Probe | Worst case | Max abs | Mean abs |
| --- | --- | ---: | ---: |
| `text.m_p` | `ljs-repetition` | 1.1280179e-5 | 1.7618415e-7 |
| `text.logs_p` | `ljs-long` | 4.6938658e-6 | 4.7407892e-8 |
| `text.mask` | all exact | 0 | 0 |
| `duration.logw` | `ljs-normalization` | 2.1278858e-5 | 1.1960203e-6 |
| duration path structure | all exact | 0 | 0 |
| `prior.m_p_expanded` | `ljs-repetition` | 1.1280179e-5 | 1.6939533e-7 |
| `prior.logs_p_expanded` | `ljs-long` | 4.6938658e-6 | 4.7073595e-8 |
| `latent.z_p` | `ljs-punctuation` | 1.5258789e-5 | 1.3995212e-7 |
| `flow.z` | `ljs-long` | 9.6797943e-5 | 2.5122832e-6 |
| `audio.pcm` | `ljs-long` | 7.4365083e-4 | 3.6846966e-6 |

Backend Plan inspection separates zero-compute views from executable nodes. The
duration, prior, latent, flow, and decoder graphs place 893, 6, 4, 357, and 516
executable nodes on CUDA respectively, with zero executable CPU fallback and one
CUDA split per stage. Weightless prior expansion and latent sampling are anchored
to the primary backend so their input flags do not pin the graph to CPU.

The initial GGML CUDA transpose-convolution kernel scanned the entire input for
every output element. Replacing that inner scan with the mathematically
equivalent kernel-width traversal, while retaining accumulation order, reduced
the 12-case waveform validation from 221.61 seconds to 10.62 seconds. On
`ljs-long`, strict CUDA takes 1.30 seconds versus 14.16 seconds for one-thread
CPU on the same DGX Spark. The full one-thread CPU waveform validator takes
46.48 seconds. These are development measurements, not a cross-platform release
performance guarantee. The aggregate environment, placement, accuracy, and
timing record is `reports/validate/vits/vits-ljspeech-cuda-experimental.json`.

## CUDA 13.3 Update 1 Linux Checkpoint

The release-locked CUDA 13.3 Update 1 toolkit has now completed the same
strict-FP32 physical validation on two Linux architectures. DGX Spark uses
Ubuntu 24.04 AArch64, GB10 compute capability 12.1, driver 580.159.03, and native
`sm_121a`. RTX 4070 SUPER uses Ubuntu 22.04 x86-64, compute capability 8.9,
driver 580.173.02, and native `sm_89`. Both builds use compiler 13.3.73 and load
CUDA 13.3 runtime libraries from an isolated user-local toolkit.

The DGX CUDA configuration passes 39/39 C/C++ unit tests. The RTX configuration
passes 40/40 because it also enables the host-only quantizer CLI contract test.
Both pass the public C lifecycle and synthesis paths, the reference CLI, twenty
repeated public synthesis processes, and all seven 12-case Golden stages.
Duration structure is exact. Worst final PCM
max-absolute drift is `7.4365083e-4` on GB10 and `9.7708963e-4` on RTX 4070
SUPER. The Backend Plan reports the same all-CUDA placement on both hosts, with
zero executable CPU fallback.

For the 9.61-second `ljs-long` waveform, the optimized end-to-end CLI path takes
1.20 seconds on GB10 versus 27.50 seconds on one-thread CPU; RTX 4070 SUPER takes
0.44 seconds versus 18.61 seconds. The first post-stride loaded-model checkpoint
recorded 403.555 ms on GB10 and 187.437 ms on RTX 4070 SUPER. Cold model loading
itself took 318.459 ms and 110.297 ms respectively.
The minimal public synthesis process peaks at approximately 370 MiB of GPU
memory on GB10 and 326 MiB on RTX 4070 SUPER. These are host-specific
operational measurements, not general performance guarantees.

The GB10 result is not a CUDA 13.3 regression: its earlier CUDA 13.0 and CUDA
13.3 checkpoints both recorded 1.30 seconds before the latest kernel change.
Profiling shows this strict-FP32 graph is dominated by transpose convolution,
im2col, and SGEMM, so the advertised sparse-FP4 Tensor performance is not the
relevant limit. DGX Spark has 273 GB/s unified-memory bandwidth, while the RTX
4070 SUPER's 21 Gbps GDDR6X and 192-bit interface provide about 504 GB/s of
theoretical bandwidth. The observed pre-optimization steady-state ratio of
about 1.87 closely tracks that bandwidth ratio. CUDA module loading also costs
more on GB10, which further penalizes a fresh CLI process.

The second transpose-convolution optimization starts at the largest kernel
weight congruent to the output index modulo the stride and advances by one
stride. VITS kernel/stride pairs 16/8 and 4/2 therefore visit two contributing
weights instead of scanning 16 or 4. A boundary-and-overlap unit case was added
before changing the kernel. CPU, both physical CUDA hosts, and all Golden checks
retain the same outputs; median loaded-model synthesis improved from 606.492 to
403.555 ms on GB10 and from 324.454 to 187.437 ms on RTX 4070 SUPER.

Pointwise Conv1D now bypasses the input transpose and `im2col` when the kernel
size is one, while deliberately retaining the old GEMM operand order and output
transpose. A fully transposed direct GEMM was rejected during validation: despite
being mathematically equivalent, its different cuBLAS path amplified accumulated
Golden drift. The retained path is bitwise-equivalent at all established probes.
A Conv1D unit test covering pointwise channel mixing plus padded, dilated regular
convolution was added before production code changed.

On a six-synthesis RTX profile, `im2col` launches fell from 1512 to 828 and
transpose-copy launches from 3246 to 2598. A same-session 20-sample A/B measured
409.220 ms before and 409.597 ms after on GB10 (+0.09%, noise-level), and 182.937
ms before versus 170.592 ms after on RTX 4070 SUPER (-6.75%). The optimization
therefore avoids a measurable GB10 regression while improving the discrete GPU.
The final placement remains all-CUDA with zero executable CPU fallback; duration
now has 803 executable nodes and acoustic flow 309 because the redundant graph
nodes no longer exist.

Regular F32 Conv1D initially used a 32-by-32 shared-memory tiled `im2col` path
for the large batch-one, unit-stride shapes used by VITS. Threads read adjacent output
positions and write adjacent patch elements; small tensors, batches, depthwise
convolutions, F16 output, non-unit stride, and 2D convolution retain GGML's
generic path. Before the production change, an exact backend test was added for
31 input channels, kernel width 7, 65 frames, padding 9, and dilation 3. It
exercises non-multiple tile edges and zero padding on CPU, GB10, and RTX 4070
SUPER.

The tiled path handles 684 of 828 regular `im2col` launches in the six-synthesis
RTX profile. Combined tiled and generic `im2col` time falls from 333.1 to 273.0
ms (-18.0%), without changing the launch count; SGEMM is now the largest single
profiled kernel family at 276.6 ms. Same-session 20-sample A/B medians improve
from 409.597 to 398.155 ms on GB10 (-2.79%) and from 170.592 to 168.509 ms on
RTX 4070 SUPER (-1.22%). Fixed-request PCM remains bitwise repeatable, all seven
Golden stages retain their established drift, and placement remains all-CUDA.

The waveform decoder now keeps its private intermediate tensors in `[T,C]`
layout across ordinary convolution, activation, residual addition, branch
averaging, and transpose convolution. Previously each ordinary convolution
materialized `[C,T]` output and the next convolution immediately copied it back
to `[T,C]`. The public graph Interface and logical `[C,T]` input/output contract
are unchanged; layout helpers remain private to the decoder Module. A two-conv
backend test with activation, padding, dilation, 17/31/5 channels, and 65 frames
proved exact equivalence on CPU and both CUDA hosts before the graph changed.

The decoder graph falls from 979 total / 463 view / 516 executable nodes to
671 / 309 / 362, with one CUDA split and zero executable CPU fallback. In the
six-synthesis RTX profile, transpose-copy launches fall from 2598 to 1674 and
their time from 79.57 to 13.81 ms (-82.6%); total GPU kernel time falls 5.13%.
Interleaved 20-sample A/B medians improve from 403.878 to 367.101 ms on GB10
(-9.11%) and from 167.833 to 160.952 ms on RTX 4070 SUPER (-4.10%). The larger
GB10 end-to-end improvement also includes reduced scheduling and unified-memory
copy overhead. The real decoder retains exactly the established Golden drift.

The final `im2col` tile expands only the output-position dimension from 32 to
128 while retaining 32 patch elements and 256 threads. Linearized cooperative
loads and stores keep both global-memory directions coalesced; the padded
32-by-129 shared array avoids bank conflicts. A second exact test at 257 frames
covers a complete 128-wide tile plus multiple tails before this production
change. Compared with the 32-by-32 layout build, interleaved 20-sample medians
improve from 366.145 to 360.980 ms on GB10 (-1.41%) and from 155.754 to 154.918
ms on RTX 4070 SUPER (-0.54%). The RTX profile measures tiled `im2col` at
279.47 versus 268.83 ms (-3.81%) and total GPU kernel time at 869.36 versus
860.39 ms (-1.03%). The launch count and numerical output are unchanged.

The final profile leaves SGEMM at 32.115%, tiled im2col at 31.245%, transpose
convolution at 11.553%, and broadcast/elementwise kernels at 8.231%. Possible
follow-up work is recorded in
`reports/porting/vits/vits-ljspeech/_performance-research.md`, including an
implicit-GEMM or fused Conv1D path, controlled SGEMM algorithm selection,
transpose-convolution fusion, general elementwise fusion, and get-rows analysis.
These entries are research candidates with a deferred go/no-go decision, not a
commitment to change the current implementation.

VITS/CUDA remains Experimental. This checkpoint validates the release toolkit
and the Linux native implementation, but does not define numerical acceptance
thresholds or satisfy the remaining R610, Windows, complete release cubin, and
clean-runtime packaging gates. The aggregate record is
`reports/validate/vits/vits-ljspeech-cuda-13.3-linux.json`.

The stochastic-duration graph consumes the text encoder's `[192, T]` hidden
state through a private module seam and replays the oracle's unscaled `[2, T]`
noise with explicit `noise_scale_w`. Its main and per-flow DDSConv blocks use F32
depthwise `im2col + mul_mat`, exact GELU, LayerNorm, and dilations 1, 3, and 9.
Three reverse ConvFlows apply the ten-bin rational-quadratic inverse spline with
linear tails, followed by inverse affine. The replay interface remains internal
C++. Public synthesis instead owns a backend-independent normal random stream:
one concrete seed deterministically produces the duration draw followed by the
acoustic-latent draw, while `SYNTH_SEED_RANDOM` selects and reports a replayable
concrete seed. The public sequence is intentionally not claimed to reproduce
PyTorch's RNG stream.

Duration rounding and monotonic path generation form the host seam between two
static-shape GGML graphs. The Module maps public `speaking_rate` to upstream
`length_scale = 1/rate`, performs F32 exponentiation and ceiling, clamps the
acoustic length to at least one, and generates logical `[Y,T]` attention with
token index contiguous. It reads `hop_length`, the native PCM output limit, and
the validated speaking-rate range from strict GGUF metadata. Requests whose
resolved `Y * hop_length` exceeds the package limit return
`SYNTH_ERR_OUTPUT_LIMIT` before attention allocation.

Model execution retains text statistics while computing `duration.logw`, avoiding
a second text-encoder pass. Once the host seam resolves `Y`, a second static GGML
graph applies exact attention to both `[C,T]` prior tensors and emits physical
channel-fastest `[C,Y]` results. The validation Adapter alone converts these to
logical channel-major files. This graph uses ordinary GGML matmul and transpose
operations and therefore does not add a CPU-specific operator or backend seam.

Latent sampling is a third static graph after `Y` is known. Its internal Interface
accepts the caller-replayed logical `[C,Y]` latent-noise values plus
`noise_scale`; the host Adapter owns finite-value, shape, and physical-layout
validation. The graph uses only elementwise multiply, exponential, scale, and
add operations. It does not reproduce PyTorch's RNG: the oracle already records
the stride-sensitive result of `randn_like`, keeping random-stream policy outside
this parity seam and allowing the same deterministic input from future C++, Rust,
or Python callers.

The reverse acoustic flow loads the package's complete 80-tensor `flow`
namespace into four mean-only residual coupling blocks. Each block contains a
pre-convolution, four gated WN layers, and a mean projection. Reverse execution
applies a full 192-channel flip before each block in order 3, 2, 1, 0; a shared
I32 row-index tensor expresses that reversal without a family-specific backend
operator. The WN path reuses the F32 `im2col + mul_mat` convolution module for
kernel 5, then combines tanh/sigmoid gates and residual/skip projections with
ordinary GGML operations. Flow metadata, including odd kernel, mean-only mode,
layer count, dilation, and padding arithmetic, is rejected before unsafe graph
construction.

The waveform decoder loads the complete 155-tensor `decoder` namespace. It
applies the 192-to-512 pre-convolution, four transpose-convolution stages with
rates 8, 8, 2, and 2, and three ResBlock1 branches per stage with kernels 3, 7,
and 11 and dilations 1, 3, and 5. GGML's transpose-convolution operation requires
zero internal padding, so the shared operation performs the full convolution and
then takes a checked symmetric view matching PyTorch's `(kernel - stride) / 2`
crop. ResBlock activations use the package slope 0.1; the upstream final
pre-output activation remains the separately pinned 0.01 slope before the
bias-free seven-tap output convolution and tanh.

All decoder operations are ordinary GGML graph nodes and reuse the same backend
seam as the earlier slices. The 12-case CPU report is
`reports/validate/vits/vits-ljspeech-waveform-decoder.json`; every output is
finite and has exactly `Y * 256` samples. The worst source-F32 difference is the
`ljs-long` measurement shown above. This establishes port parity and does not
claim a perceptual quality score.

The public C test loads the same GGUF through an opaque `synth_model_t`, creates a
reusable `synth_context_t`, and synthesizes the minimal Golden token sequence
through both Audio Sink and owned-buffer forms. Two seed-42 operations produce
bit-identical PCM on the same CPU configuration, while seed 43 produces a
different stochastic result. The result reports 22,050 Hz mono output, the
actual seed, and resolved `en` metadata. Request cancellation returns before
graph execution, and a one-frame request limit is rejected after duration
resolution but before latent allocation and waveform decoding. VITS and GGML
types remain private, so future Rust and Python bindings use this same C seam
rather than a family-specific inference path.
