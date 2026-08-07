# OmniVoice 0.6B

Status: F32 is `port_validated`. **Quality evaluation has not been run.** A
Listening Audit on 2026-08-07 found no obvious regression across six pairs
(`no_obvious_regression`) -- see "Listening Audit," below; neither claim moves
the Validation Level. This is a **Restricted Model Package** (ADR 0018), not a
Published Model Package: the generator (LM) weights are CC-BY-NC (no version
stated by upstream) and the codec weights carry the Boson Higgs Audio 2
Community License. It has **not been published**. Uploading it to
`jiangzhuo9357/omnivoice-0-6b-gguf` requires jiangzhuo's separate, per-act
confirmation (ADR 0018's last consequence); this document prepares the
artifacts and records the commands that confirmation would run, without
running them.

## Package

The package is converted from the flagship `k2-fsa/OmniVoice` checkpoint at
pinned source revision `468e927ba3716cd8dd86421148dfb3046e9f9d7b` (package
0.2.1) and pinned weights revision `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`.
It is a non-autoregressive mask-predict diffusion language model: a
bidirectional Qwen3-0.6B backbone runs full attention (no KV cache) over a
fixed-length canvas of 8 acoustic codebooks x 1025-entry vocabulary at 25 Hz,
refined over 32 parallel denoising steps (16 in fast mode) with
classifier-free guidance at scale 2.0. The committed token grid is decoded to
24 kHz mono F32 PCM by the Higgs Audio V2 codec's DAC-style convolutional
decoder at hop 960.

Voice arrives through three modes mapped onto the public Voice Profile
sources: **Reference Audio** cloning (a transcript is required, language
optional), **Description Text** voice design, and an unnamed auto-voice
default with an empty Preset Voice Catalog -- with no profile supplied, the
speaker follows the synthesis seed, so a caller who wants the same speaker
twice reuses the seed the request reports. The package declares a validated
Language Capability Catalog of `en`, `zh`, `ja`; the checkpoint claims 600+
languages through its training data and prompt format, but no language beyond
these three has its own validation case.

Input is **raw UTF-8 text only**. The package carries its own embedded
byte-level BPE frontend (`synthesize.qwen_bpe`, the same vocabulary layout
qwen3-tts uses, hoisted to a shared internal module) and declares
`input_flags = SYNTH_INPUT_SUPPORT_TEXT_UTF8` exactly -- unlike VITS, Kokoro,
and qwen3-tts, this package does **not** accept a raw token-ID bypass or
phoneme input: v1 assembles its own prompt from text, so a token-sequence
input has no consumer (`docs/porting/families/omnivoice.md`, "Amended
2026-07-31"), and the loader refuses to load any package declaring more than
the text flag.

| Profile | Bytes | Tensor storage | SHA-256 |
| --- | ---: | --- | --- |
| F32 | 3,189,953,504 | 798 F32 | `f6d504ffaddcbf32f80f1f6c847f075bbd5d2c7b50fe95a194ceb635772f9fa3` |

The public C ABI is profile-independent. C++, Rust, and Python callers load a
local GGUF through the same model interface. The tensor catalog, RVQ dequant
path, and execution details remain private to the architecture module.

### What the profiles quantize, and what they never do

**No Quantization Profile ships. This family is F32-only.** By jiangzhuo's
ruling of 2026-08-06, any produced profile is codec-only: the generator
(`llm.*`, `audio_embeddings.weight`, `audio_heads.weight`) and the RVQ
(`codec.quantizer.*`, `codec.fc`, `codec.fc2`) always stay at the reference
dtype, regardless of shape. The reason a codec-only scope was chosen at all is
a measured, prior finding rather than a convention: a reference port measured
greedy-decode token agreement collapsing from 100% to roughly 7% under an F16
*generator*, because an argmax flip at one committed step feeds back into
every later step of the same synthesis. Quantizing the generator was never
attempted here for that reason.

Two codec-only profiles were produced and measured against this family's own
exact-token gate (below), and **both failed it**:

| Profile | Bytes | Reduction | Clone RVQ tokens mismatched | Greedy grids |
| --- | ---: | ---: | ---: | ---: |
| Q8_MIXED | 2,703,016,576 | 15.3% | 1,023 of 2,808 (36.4%) | 17/17 exact |
| F16 | 2,858,422,240 | 10.4% | 103 of 2,808 (3.7%) | 17/17 exact |

Both profiles reproduce the greedy decode loop's 8 x T token grid exactly,
for a structural reason rather than luck: the generator and the RVQ are
Sensitive/F32 under every profile, so the decode loop's logits are bit-for-bit
identical to the F32 package's. What actually fails is the **Reference Audio
cloning path's own RVQ encode**: quantizing `codec.semantic_model` (HuBERT)
and `codec.acoustic_encoder` moves the fused latent that feeds the encode's
nearest-neighbor codebook lookup by orders of magnitude (`ref.fused_latent`
max_abs 9.32e-05 at F32 versus 0.129286 at F16 and 3.73227 at Q8_MIXED), which
flips a discrete nearest-neighbor decision at a large fraction of frames --
not a knife-edge margin call eligible for the dual-admissibility mechanism, in
either case. Per this family's own gate discipline, a profile that fails the
exact-token gate is not shipped and no perceptual claim substitutes for it, so
neither Q8_MIXED nor F16 has a registered golden gate or a committed tolerance
cell, and the measurement stands as the record instead
(`docs/porting/families/omnivoice.md`'s Quantization Profile Shape section;
`reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`'s Plan 4 Task 3
entries).

## Port validation

Twenty deterministic Golden cases run against the pinned upstream PyTorch
oracle on DGX Spark CPU and NVIDIA GB10 CUDA 13.3: 17 greedy (both
temperatures pinned to zero, which makes the whole diffusion loop deterministic
and RNG-free) and 3 public-sampled-path cases. Per-stage oracle dumpers cover
the tokenizer, the prompt builder, a single conditional forward (probed at
five internal depths plus the final hidden state and the step-0 guided
logits), the codec decode, the HuBERT semantic branch, the RVQ encode, and the
end-to-end waveform -- eight stages in total.

**This family's real guarantee is exact token identity, not a tolerance, and
it is two separate claims:**

- The greedy decode loop's full 8 x T unmasking grid matches the oracle's
  byte-for-byte on **17 of 17** golden cases (one, `omni-fast-mode`, via a
  committed alternate grid the oracle itself also produces -- the
  dual-admissibility mechanism for a decision narrower than this arithmetic
  resolves, never a tolerance on a token id).
- The Reference Audio cloning path's own RVQ encode grid matches the oracle's
  byte-for-byte on **both** clone cases -- 8 codebooks x 351 frames, **2,808 of
  2,808 tokens exact**, on the first run.

Both grids stayed byte-exact when the codec's decode graph moved to CUDA too
(below): a flip in either grid, on any profile or backend, is a shipping
blocker by this project's own exact-token discipline, and none has occurred.

Everything else -- the deep generator probes and the decoded waveform -- is
measured for completeness rather than enforced as this family's real claim,
because pre-norm outlier channels make a sample-wise bound reject a provably
correct port: `generator.hidden_l27` reaches a max-abs of 0.08 while its
cosine still holds at eight nines. Worst figures, CPU F32 against the oracle:

| Probe | Worst cosine deviation (1 − cosine) | Worst max\_abs |
| --- | ---: | ---: |
| `generator.hidden_l0` | 1.30e-07 | 5.72e-05 |
| `generator.hidden_l7` | 1.97e-07 | 4.12e-04 |
| `generator.hidden_l14` | 9.46e-08 | 7.02e-04 |
| `generator.hidden_l21` | 1.37e-07 | 4.15e-03 |
| `generator.hidden_l27` | 2.14e-07 | 8.01e-02 |
| `generator.final` | 1.72e-07 | 1.04e-03 |
| `generator.logits_step0` | 9.06e-08 | 6.10e-04 |
| `audio.pcm` / `audio.pcm_freerun` | 1.44e-07 | 1.69e-05 |
| `ref.pcm_16k` (clone resampler) | 5.56e-08 | 1.19e-07 (≈1 float32 ULP) |
| `ref.semantic_mean` (clone HuBERT) | 7.07e-08 | 6.09e-05 |
| `ref.fused_latent` (clone DAC + fusion) | 0.0 (capped) | 9.32e-05 |

Every committed tolerance is 5x the measured deviation
(`tests/tolerances/omnivoice.json`, `reference_stage:
source-f32-oracle-vs-f32-cpu`); both sides run F32 on CPU for this
comparison, so the thresholds carry neither a dtype nor a device
difference, only an implementation one.

### Backends: CPU baseline, CUDA partial

CPU is the mandatory baseline; every measurement above is against it. CUDA
support is **partial**, not full, and the split is a design rule rather than
an unmeasured gap: only the codec's decode graph (the RVQ dequantizer, the
acoustic decoder, and the final projection -- 152 tensors) moves to CUDA. The
generator -- the entire mask-predict denoising loop and its whole input path --
stays on the CPU **unconditionally**, on every case and every profile, because
the codec's own token selection feeds the generator's next denoising step and
`docs/backends.md`'s discrete-outputs rule holds a discrete decision and its
input path off the accelerator regardless of backend.

Measured 2026-08-07 across all twenty Golden cases with `--accelerate`: every
codec node left the CPU and every generator node did not, not just on
average -- aggregated over the whole suite, codec 8,440 of 8,440 nodes off the
CPU, generator 0 of 880,032. All seventeen greedy token grids and both
cloning RVQ grids stayed byte-exact against the CPU baseline. The one artifact
that does move is the decoded waveform, because it is what the accelerated
codec's TF32 arithmetic actually touches:

| Backend | `audio.pcm` worst cosine | `audio.pcm` worst max\_abs |
| --- | ---: | ---: |
| CPU (F32 baseline) | 0.99999986 | 1.69e-05 |
| CUDA (codec only) | 0.99999635 | 7.01e-03 |

committed to `tests/tolerances/omnivoice.json`'s `backends.CUDA.stages.replay`
cell at `min_cosine 0.999981` / `max_abs 0.04` (five times the measured
deviation). On the suite's longest case the codec itself is 9.68x faster
(4.3069 s to 0.4451 s), but because the CPU-held generator is 98.9% of wall
time there, the end-to-end effect is a bounded 3.4% reduction -- the opposite
shape from a GPU-primary family, where holding a minority stage on CPU is a
tax rather than a small saving. Full method, the twenty-case sweep, and the
operational-evidence tables (latency, repeated-run cleanup, and why peak
memory has no second budget on this UMA host) are in
`docs/porting/families/omnivoice.md`'s Execution Backends section and
`reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`'s 2026-08-07 Task
11 entry.

### Listening Audit

One maintainer, six A/B pairs, 2026-08-07: **`no_obvious_regression`**. Five
pairs compare the port's replayed codec against the pinned PyTorch oracle
(covering the worst and second-worst waveform cosine, one Reference Audio
clone case, one Description Text case, and one random pick); the sixth
compares the codec's CUDA decode against its CPU decode of the identical
byte-exact committed token grid, deliberately on the case with the largest
measured CUDA numeric divergence -- its inaudibility corroborates the backend
claim rather than merely accompanying it. This is one listener, six pairs,
non-statistical: it says no obvious problem was noticed on the pairs heard,
not that the port and the oracle are perceptually equivalent, and it does not
move `quality_evaluation` off `not_run` (ADR 0017's automated grid has not
run and is not scheduled). Full identity key, seeds, and method are in
`docs/porting/families/omnivoice.md`'s Listening Audit section and
`reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`'s 2026-08-07 Task
16 entry.

## Reproduction

Materialize the reference artifacts, then run the registered validators:

```bash
uv run --project scripts/envs/omnivoice --locked \
  python scripts/dump_reference_omnivoice_pytorch.py \
    --manifest tests/golden/omnivoice/omnivoice-0-6b.manifest.json \
    --weights-dir models/omnivoice-0-6b

cmake -S . -B build -DSYNTH_BUILD_TESTS=ON -DSYNTH_BUILD_PYTHON_TESTS=ON \
  -DSYNTH_BUILD_INTEGRATION_TESTS=ON
cmake --build build --target synthesize-check-integration
```

The CUDA cell above is reproduced the same way on a `dev-dgx-spark`-configured
tree, adding `--accelerate` to `scripts/validate-omnivoice-replay.py`'s own
invocation (`docs/backends.md`;
`reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`'s 2026-08-07
entry has the exact command line).

The blocked Quantization Profiles are reproduced, not shipped, with the
generic quantizer tool:

```bash
cmake -S . -B build -DSYNTH_BUILD_TOOLS=ON
cmake --build build --target synthesize-quantize
build/bin/synthesize-quantize \
  models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf \
  models/omnivoice-0-6b/omnivoice-0-6b-Q8_MIXED.gguf --quant Q8_MIXED
build/bin/synthesize-quantize \
  models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf \
  models/omnivoice-0-6b/omnivoice-0-6b-F16.gguf --quant F16
```

Neither output above passes the exact-token gate (see Package); running these
commands reproduces the negative measurement, not a package to ship.

Generate and verify the model card:

```bash
uv run scripts/hf_cards/generate.py scripts/hf_cards/omnivoice-0-6b.yaml \
  --output models/publish/omnivoice-0-6b/README.md
uv run scripts/hf_cards/generate.py scripts/hf_cards/omnivoice-0-6b.yaml \
  --output models/publish/omnivoice-0-6b/README.md --check
```

### The publication directory

`models/omnivoice-0-6b/` is the *working* directory: it holds the upstream
checkpoint (`model.safetensors`, `audio_tokenizer/`, `tokenizer.json`, the
upstream `config.json`, and the upstream README) beside every produced GGUF,
because the converter, the oracle dumpers, and the (blocked) quantizer all
read and write there. `hf upload` pointed at it would publish the raw upstream
weights alongside the port -- the same hazard Kokoro's ship task closed by
keeping its own working directory clean and its upstream card at
`models/upstream/<slug>/README.md`.

`models/publish/omnivoice-0-6b/` is the clean, flat publication directory
this task built for that reason, containing **exactly**:

```text
models/publish/omnivoice-0-6b/
├── omnivoice-0-6b-F32.gguf       # hard link to the working directory's F32 GGUF
├── LICENSE-higgs-audio-2.txt     # hard link; the declared Sidecar Resource
└── README.md                     # generated by scripts/hf_cards/generate.py
```

Verified by listing it: no upstream checkpoint file, no `audio_tokenizer/`, no
`tokenizer.json`, and no Q8_MIXED or F16 GGUF (neither ships) are present. The
hard links share the same filesystem and inode as the working directory's
copies rather than duplicating 3 GB, which is safe because `models/` is
entirely git-ignored and this directory is a working artifact, not a tracked
one. `models/upstream/omnivoice-0-6b/README.md` carries the same upstream
card copied out of the working directory, so
`scripts/hf_cards/omnivoice-0-6b.yaml`'s `source.card_path` -- and the
reproduced "Original upstream project card" section at the bottom of the
generated README --
resolve against a copy that is never mixed with our own artifacts, exactly as
Kokoro's `models/upstream/kokoro-v1_0/README.md` does.

### Publication commands (recorded, not yet run)

**Not executed. Publication requires jiangzhuo's separate, per-act
confirmation naming this exact target** (`jiangzhuo9357/omnivoice-0-6b-gguf`,
`main`) --- an approved plan is not that confirmation. The commands that
confirmation would run, exactly as they would be typed:

```bash
hf repos create jiangzhuo9357/omnivoice-0-6b-gguf \
  --type model --public --exist-ok
hf upload jiangzhuo9357/omnivoice-0-6b-gguf models/publish/omnivoice-0-6b . \
  --commit-message "Publish OmniVoice 0.6B F32 (Restricted Model Package: CC-BY-NC generator + Boson Higgs Audio 2 Community License codec)"
```

## Licensing

Three separate upstream grants apply; only two touch the artifact in
`models/publish/omnivoice-0-6b/`.

**The generator (LM) weights are CC-BY-NC, with no version stated.** The
upstream model card has no `license:` frontmatter key; the only statement is
this prose, quoted verbatim from the pinned weights revision:

> Our code is released under the Apache 2.0 License. The pre-trained model is
> licensed under the CC-BY-NC due to constraints from its training data (e.g.,
> Emilia).

Upstream names no CC-BY-NC version anywhere on the card, so this project's
license metadata says exactly that instead of inventing one -- `license: other`
with `license_name:
omnivoice-cc-by-nc-unspecified-version-plus-boson-higgs-audio-2-community`,
never `cc-by-nc-4.0` (ADR 0018:29-31; the ruling of 2026-08-06). **These
weights are not licensed for commercial use.** The
restriction traces to the training data the statement itself names --
Emilia, a large-scale multilingual speech corpus distributed under its own
non-commercial terms.

**The codec (Higgs Audio V2) weights carry a second, separate license: the
Boson Higgs Audio 2 Community License**, bundled by the weights repository as
`audio_tokenizer/LICENSE` and carried with this package as a declared
**Sidecar Resource** (`LICENSE-higgs-audio-2.txt`), never a footnote. That
agreement is derived from the Meta Llama 3 Community License and requires
dual attribution on redistribution:

> "Meta Llama 3 is licensed under the Meta Llama 3 Community License,
> Copyright © Meta Platforms, Inc. All Rights Reserved."
> "Boson Higgs Audio 2 is licensed under the Boson Community License,
> Copyright © Boson AI USA, Inc. All Rights Reserved."

It additionally caps commercial use at 100,000 monthly active users and
forbids using its outputs to train other models -- independent of, and in
addition to, the generator's own CC-BY-NC restriction above.

**Apache-2.0 covers only the upstream GitHub source code**
(`k2-fsa/OmniVoice`) and nothing produced by this project: no weight file,
converted artifact, or GGUF this project ships may ever be labelled
apache-2.0. All three community GGML ports read at design time misstate or
omit these terms (one Hugging Face GGUF repository labels the weights
apache-2.0, which is simply false); nothing downstream of upstream's own card
and license files is inherited.

The upstream card also carries a use disclaimer this package repeats
verbatim rather than paraphrases:

> Users are strictly prohibited from using this model for unauthorized voice
> cloning, voice impersonation, fraud, scams, or any other illegal or
> unethical activities. All users shall ensure full compliance with
> applicable local laws, regulations, and ethical standards. The developers
> assume no liability for any misuse of this model and advocate for
> responsible AI development and use, encouraging the community to uphold
> safety and ethical principles in AI research and applications.

Both upstream licenses permit non-commercial redistribution with attribution,
and upstream itself distributes the weights openly on Hugging Face -- the
basis for republishing this converted GGUF under exactly those terms rather
than a broadening of them (ADR 0018). This is a Restricted Model Package, not
a Published Model Package, and must never be described as one.
