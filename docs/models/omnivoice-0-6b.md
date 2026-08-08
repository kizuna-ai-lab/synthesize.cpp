# OmniVoice 0.6B

Status: F32 is `port_validated`. **Quality evaluation has not been run.** A
Listening Audit on 2026-08-07 found no obvious regression across six pairs
(`no_obvious_regression`) -- see "Listening Audit," below; neither claim moves
the Validation Level. Two corrections landed 2026-08-08 after a further audit:
see **"Short canvases produce unintelligible output"** and the auto-voice
speaker note under "Package" -- both describe the shipped configuration. The
quantization table under "Package" was re-measured on 2026-08-09 under the
conv-exempt codec policy; `Q8_MIXED` means something different for this family
since that date and still does not ship.
This is a **Restricted Model Package** (ADR 0018), not a
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
default with an empty Preset Voice Catalog.

**Auto-voice has no speaker conditioning, and the seed alone does not pin the
speaker.** Corrected 2026-08-08; this page previously said the speaker follows
the synthesis seed, full stop, which is not true. With no profile supplied
there is no speaker signal in the prompt at all and the model carries no
speaker embedding table, so which speaker you get is emergent from which token
grid the decode lands on. **Reproducing a speaker requires the same seed *and*
the same Execution Backend, package, and step count.** Switching between the
shipped CPU and CUDA backends can produce a different speaker for an otherwise
identical request: on one measured case the CPU path's median F0 is 118 Hz and
the CUDA path's is 189 Hz -- a male voice and a female voice for the same call.
Callers who need a stable identity should build a Voice Profile from Reference
Audio or Description Text; those paths are conditioned and are not subject to
this.

**How often, measured.** Both backends were rendered for all twenty Golden
cases at the shipped step count and compared on median F0 from two independent
pitch trackers: **one case of the seventeen greedy cases changes speaker**
(`omni-short-en`). Every other measurable case stays in the same register on
both trackers. `omni-rate-fast` is excluded rather than counted, because its
own reference audio is degenerate and neither arm carries a voice to compare.
A listener heard the changed pair blind, reported the two arms as different
people, and judged their audio quality indistinguishable -- so this is an
identity effect, not a quality one.

**Token drift does not predict it.** The case with the highest disagreement
between backends, 98.3% of committed tokens, keeps the same speaker
(178 Hz against 183 Hz). A caller cannot infer speaker stability from how much
the two backends' tokens differ, and neither can this project's tolerance
grid -- which is why the finding took a listening pass to surface.

The package declares a validated
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

### Short canvases produce unintelligible output

Recorded 2026-08-08. **This is the one hazard on this page a caller can reach
without doing anything unusual, and nothing in the runtime warns about it.**

The model paints a fixed-length canvas whose length is estimated from the text
and then divided by the speaking rate. Below roughly **37-40 frames (~1.5-1.6
seconds)** the rollout has too few positions to place the text and the output
degenerates into a near-DC, sub-50 Hz rumble instead of speech: no amplitude
envelope, no silence, a large DC offset. Two ordinary requests reach it:

- **A high speaking rate on a short text.** The package declares a
  `speaking_rate_range` of `[0.5, 2.0]` and `synthesize-cli --rate 2.0` is
  accepted silently, but the pinned upstream oracle is already degenerate at
  rate 1.50 on a 32-character sentence, and upstream's own demo UI caps speed
  at 1.5. **The top of the declared range is not validated as speakable.**
- **A very short text at the default rate.** `--text "Hi."` resolves to a
  22-frame canvas and degenerates with no rate change at all. **No minimum
  input length is enforced anywhere.**

Rate 2.0 on a *long* text is fine, so the variable is the resolved canvas
length, not the rate as such.

**This is the upstream model's own behavior, faithfully reproduced, not a port
defect**: on the affected golden case this port matches the pinned PyTorch
oracle's waveform at Pearson r = 0.999670, the tightest of any case measured,
and the oracle's own reference audio for that case is equally unintelligible.
The golden suite does not catch it because port validation compares against the
oracle and makes no intelligibility claim (ADR 0017). Lowering the declared
range and emitting a diagnostic on a too-short canvas are both recommended and
neither has been done; see
`reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`'s 2026-08-08
step-count audit entry, Finding 1.

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

Codec-only profiles were produced and measured against this family's own
exact-token gate (below), and **every one of them failed it**:

| Profile | Bytes | Reduction | Clone RVQ tokens mismatched | Greedy grids |
| --- | ---: | ---: | ---: | ---: |
| Q8_MIXED (current, conv-exempt) | 2,778,427,360 | 12.9% | 98 of 2,808 (3.49%) | 17/17 exact |
| F16 | 2,858,422,240 | 10.4% | 103 of 2,808 (3.7%) | 17/17 exact |
| Q8_MIXED (superseded, packed convs) | 2,703,016,576 | 15.3% | 1,023 of 2,808 (36.4%) | 17/17 exact |

`Q8_MIXED` means something different for this family since the conv-exempt
codec policy of 2026-08-09: it no longer block-quantizes any convolution
kernel, so 85 of the 158 codec matrix weights are held at F16 and only the 73
HuBERT Linears are Q8_0. The last row is what the same command line produced
before that change and is **no longer reproducible**; it is kept here because
the 36.4% figure it belongs to has been quoted. Exempting the convolutions cut
the clone drift by a factor of 10.4 and is what makes those first two rows
nearly equal -- the drift that remains is the precision of the convolutions,
not of the Linears (`docs/porting/families/omnivoice.md` has the four-cell
attribution).

Every profile reproduces the greedy decode loop's 8 x T token grid exactly,
for a structural reason rather than luck: the generator and the RVQ are
Sensitive/F32 under every profile, so the decode loop's logits are bit-for-bit
identical to the F32 package's. What actually fails is the **Reference Audio
cloning path's own RVQ encode**: quantizing the clone-encode path moves the
fused latent that feeds the encode's nearest-neighbor codebook lookup
(`ref.fused_latent` max_abs 9.32e-05 at F32 versus 0.123921 at the current
Q8_MIXED, 0.129286 at F16 and 3.73227 at the superseded one), which flips a
discrete nearest-neighbor decision at a large fraction of frames -- not a
knife-edge margin call eligible for the dual-admissibility mechanism, in any
case. Per this family's own gate discipline, a profile that fails the
exact-token gate is not shipped and no perceptual claim substitutes for it, so
no profile has a registered golden gate or a committed tolerance cell, and the
measurement stands as the record instead
(`docs/porting/families/omnivoice.md`'s Quantization Profile Shape section;
`reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`'s Plan 4 Task 3 and
2026-08-09 entries).

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

### Backends: CPU baseline, CUDA now the whole graph

CPU is the mandatory baseline; every measurement above is against it. CUDA
support **covered only the codec's decode graph through Plan 4** (the RVQ
dequantizer, the acoustic decoder, and the final projection -- 152 tensors),
with the generator held on CPU unconditionally under `docs/backends.md`'s
discrete-outputs rule. **Superseded 2026-08-08 (Plan 5 Task 1):** after
jiangzhuo revised this family's bar from token identity to audible quality
(a six-pair blind A/B heard no problem in generator-on-CUDA output), the
generator earned `docs/backends.md`'s new "one narrow exception" to that
rule -- its discrete token choice never changes the canvas shape, only its
content -- and now moves to CUDA too, mirrored through its own
accelerator-resident weight twin.

Measured 2026-08-07 across all twenty Golden cases with `--accelerate`
(Plan 4, codec-only, kept for the record): every codec node left the CPU and
every generator node did not -- aggregated over the whole suite, codec 8,440
of 8,440 nodes off the CPU, generator 0 of 880,032. All seventeen greedy
token grids and both cloning RVQ grids stayed byte-exact against the CPU
baseline. The one artifact that moved was the decoded waveform, because it
is what the accelerated codec's TF32 arithmetic actually touches:

| Backend | `audio.pcm` worst cosine | `audio.pcm` worst max\_abs |
| --- | ---: | ---: |
| CPU (F32 baseline) | 0.99999986 | 1.69e-05 |
| CUDA (codec only, Plan 4) | 0.99999635 | 7.01e-03 |

committed to `tests/tolerances/omnivoice.json`'s `backends.CUDA.stages.replay`
cell at `min_cosine 0.999981` / `max_abs 0.04` (five times the measured
deviation). Once the generator also moves (Plan 5), token *content* -- not
grid size -- diverges from the CPU baseline in most cases; this is a
deliberate, measured, and audited trade (see the family doc's Listening
Audit), not a regression.

**RTF, corrected.** The Plan 4 codec-only figure below (9.68x faster codec,
3.4% end-to-end reduction) was measured on the `dev-dgx-spark` preset, whose
`RelWithDebInfo` build compiles ggml-cpu at `-O2` -- 2.19x slower than the
shipped `Release`/`-O3` build -- so it is doubly stale: superseded by the
generator's own move, and pessimistic on top of that. The honest, corrected
pair, from a Release CUDA tree (`build/rel-dgx-spark`) with the generator
also on CUDA, same case (`omni-long-boundary`, 719 frames):

| Backend | Time | RTF |
| --- | ---: | ---: |
| CPU | 122.61 s | 4.263 |
| CUDA (generator + codec) | 5.481 s | 0.1906 |

**22.4x**, faster than real time. (For the record, the superseded Plan 4
figure: codec alone was 9.68x faster [4.3069 s to 0.4451 s] but the CPU-held
generator was 98.9% of wall time, so the end-to-end effect was a bounded
3.4% reduction, 267.30 s to 258.48 s, RTF 9.294 to 8.987 -- the opposite
shape from a GPU-primary family, where holding a minority stage on CPU is a
tax rather than a small saving.) Full method, the twenty-case sweep, the
corrected RTF measurement, and the operational-evidence tables (latency,
repeated-run cleanup, and why peak memory has no second budget on this UMA
host) are in `docs/porting/families/omnivoice.md`'s Execution Backends
section and `reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`'s
2026-08-07 Task 11 entry and 2026-08-08 erratum.

### Listening Audit

One maintainer, six A/B pairs, 2026-08-07: **`no_obvious_regression`**. Five
pairs compare the port's replayed codec against the pinned PyTorch oracle
(covering the worst and second-worst waveform cosine, one Reference Audio
clone case, one Description Text case, and one random pick); the sixth
compares the codec's CUDA decode against its CPU decode of the identical
byte-exact committed token grid, deliberately on the suite's longest-duration
case (also the codec's largest measured CUDA speedup, 9.68x at 719 frames) --
its inaudibility corroborates the backend claim rather than merely
accompanying it. This is one listener, six pairs,
non-statistical: it says no obvious problem was noticed on the pairs heard,
not that the port and the oracle are perceptually equivalent, and it does not
move `quality_evaluation` off `not_run` (ADR 0017's automated grid has not
run and is not scheduled). Full identity key, seeds, and method are in
`docs/porting/families/omnivoice.md`'s Listening Audits section and
`reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`'s 2026-08-07 Task
16 entry.

**A later audit rejected a candidate configuration, 2026-08-08.** Halving the
decode step count from the shipped 32 to 16 buys 44.6% of the wall clock; a
six-pair blind A/B returned four "no difference" and one clear preference for
the 32-step arm (`omni-short-ja`, whose 16-step audio is incomplete at the
end), so **16 steps is rejected and the shipped default stays at 32**. No code
changed, and `num_step` is not reachable through the public C interface in any
case. That audit is also where the short-canvas hazard above and the
auto-voice speaker correction in "Package" came from -- both concern the
**shipped** 32-step path, both were traced to the pinned oracle rather than to
this port, and both produced documentation corrections rather than a code
change. Neither alters the 2026-08-07 `no_obvious_regression` verdict, whose
subject is what ships.

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

Generate and verify the model card. Unlike the reproduction commands above,
this one is written as the plain sibling invocation on purpose (no
`--output` override): `generate.py`'s own `model_dir` resolution
(`REPO_ROOT / "models" / model_slug`) targets `models/omnivoice-0-6b/` by
default for every family, VITS and Kokoro included, and this family's card
must answer that same invocation the same way theirs do:

```bash
uv run scripts/hf_cards/generate.py scripts/hf_cards/omnivoice-0-6b.yaml
uv run scripts/hf_cards/generate.py scripts/hf_cards/omnivoice-0-6b.yaml --check
```

### The publication directory

`models/omnivoice-0-6b/` is the *working* directory: it holds the upstream
checkpoint (`model.safetensors`, `audio_tokenizer/`, `tokenizer.json`, the
upstream `config.json`, and, now, our own generated `README.md`) beside every
produced GGUF, because the converter, the oracle dumpers, and the (blocked)
quantizer all read and write there. `hf upload` pointed at it would publish
the raw upstream weights alongside the port.

**The upstream card was moved, not overwritten in place.** Before this task,
`models/omnivoice-0-6b/README.md` *was* the upstream checkpoint's own README
(no `synthesize.cpp` prose anywhere in it) — the one artifact that
`generate.py`'s default `--check` invocation actually verifies, and it was
never replaced. A copy was made to `models/upstream/omnivoice-0-6b/README.md`
first; the default location now holds our generated card, exactly like every
sibling family's `models/<slug>/README.md`, and `source.card_path` in the
YAML points at the moved copy so the "Original upstream project card"
section at the bottom of the generated README — and any future re-generation
— resolves against a copy that is never mixed with our own artifacts, the
same pattern Kokoro's `models/upstream/kokoro-v1_0/README.md` already uses.

`models/publish/omnivoice-0-6b/` is the clean, flat publication directory
this task built because, unlike Kokoro's or VITS's working directories, this
family's working directory is not itself flat — it still carries the
upstream checkpoint files above. It contains **exactly**:

```text
models/publish/omnivoice-0-6b/
├── omnivoice-0-6b-F32.gguf       # hard link to the working directory's F32 GGUF
├── LICENSE-higgs-audio-2.txt     # hard link; the declared Sidecar Resource
└── README.md                     # hard link to the freshly generated models/omnivoice-0-6b/README.md
```

Verified by listing it: no upstream checkpoint file, no `audio_tokenizer/`, no
`tokenizer.json`, and no Q8_MIXED or F16 GGUF (neither ships) are present. All
three entries are hard links sharing the same filesystem and inode as their
working-directory originals rather than duplicating 3 GB, which is safe
because `models/` is entirely git-ignored and this directory is a working
artifact, not a tracked one — and it means the publication directory can
never drift from the working copy that `--check` verifies.

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
