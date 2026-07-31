# omnivoice-0-6b Porting Log

## 2026-07-30 — Intake and oracle smoke

- Pinned `k2-fsa/OmniVoice` source at commit
  `468e927ba3716cd8dd86421148dfb3046e9f9d7b` (package version 0.2.1) and the
  weights repository of the same name at revision
  `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`. The weights revision was captured
  from `HfApi().model_info(...).sha` before downloading, and the download was
  made at that revision rather than at `main`.
- Downloaded 13 files totalling 3,267,470,260 bytes into the ignored local model
  cache: the 2,450,344,112-byte generator `model.safetensors`, the
  805,665,628-byte `audio_tokenizer/model.safetensors`, the 11,423,986-byte
  `tokenizer.json`, and the configuration and licence files. Every file's
  SHA-256 is recorded in `intake.json` under `weights.files`, and every LFS
  digest in the downloader's own tree manifest was re-verified against a local
  hash of the downloaded bytes.
- Inventoried both checkpoints into
  `reports/porting/omnivoice/omnivoice-0-6b/tensor-inventory.json`, which Tasks
  7 and 10 read: 313 tensors and 612,577,288 parameters in the generator, 527
  tensors and 201,400,553 in the codec.
- Ran the pinned package on CPU in F32. It produces a finite 24 kHz waveform and
  is byte-identical across separate processes under `position_temperature=0.0,
  class_temperature=0.0`.

### License audit — three grants, and only one of them is on the card

Audited against the downloaded snapshot at `c5fdb5cc` and the source `LICENSE`
at `468e927b`. Not the live site, and — the point that removed this family from
consideration once already — not any port's README.

**Code, Apache-2.0.** `LICENSE` at the pinned source revision, 11,342 bytes,
sha256 `c843a20e…`, opening with the Apache License Version 2.0 heading. It
covers the GitHub source and nothing else.

**Generator weights, CC-BY-NC with no version stated.** The card's frontmatter
spans lines 1–658 and carries exactly five keys — `base_model`, `language`,
`pipeline_tag`, `tags`, `library_name`. There is **no `license:` key**, so the
only licence statement anywhere on the card is this prose at line 783:

> Our code is released under the Apache 2.0 License. The pre-trained model is
> licensed under the CC-BY-NC due to constraints from its training data (e.g.,
> Emilia).

Upstream names no CC-BY-NC version. The model card must say so rather than
inventing one.

**Codec weights, Boson Higgs Audio 2 Community License.** `audio_tokenizer/`
bundles a 9,171-byte `LICENSE` (sha256 `ac933dc0…`) opening:

> BOSON HIGGS AUDIO 2 COMMUNITY LICENSE AGREEMENT
>
> Boson Higgs Audio 2 Version Release Date: June 20, 2025
>
> This License Agreement (the "Agreement") is entered into by and between
> Licensee (as defined below) and Boson AI USA, Inc. ("Boson") and is based upon
> the Meta Llama 3 Community License Agreement as of April 18, 2024 (the "Meta
> License Agreement")…

It incorporates the Meta Llama 3 terms by reference, requires the agreement text
to travel with any redistribution, imposes attribution and naming obligations,
caps commercial use at 100k monthly active users, and forbids using the outputs
to train other models.

The card-level restriction scan is the part worth recording as method. Searching
the downloaded card for *non-commercial*, *cc-by*, *restrict* or *licen[cs]e*
returns two hits: the `## License` heading and the CC-BY-NC sentence itself.
**A card-only scan would have found one grant and missed a second.** The Boson
agreement is not on the card at all; it is a file sitting beside the codec
weights. And `audio_tokenizer/README.md` is an unedited Hugging Face
auto-generated template whose License field reads `[More Information Needed]`,
so the bundled file is the codec's sole grant. That is exactly why it ships as a
declared Sidecar Resource rather than as a summarised footnote.

The card also carries a Disclaimer at line 787 prohibiting unauthorized voice
cloning, voice impersonation, fraud and scams. It is quoted verbatim in
`intake.json` and the rendered card must reproduce it at ship time.

Unlike qwen3-tts and Kokoro, there is no permissive grant to rely on here.
Upstream names Emilia as the constraining corpus and gives the non-commercial
term as the consequence.

### `trust_remote_code` is genuinely not required, and now there is evidence

The family document asserted this. It was unbacked, so it was checked five ways:

1. The weights repository contains no `.py` file at all at the pinned revision.
2. Neither `config.json` nor `audio_tokenizer/config.json` declares an
   `auto_map` key.
3. The string `trust_remote_code` appears nowhere in the pinned package source.
4. The model is loaded by `OmniVoice.from_pretrained` from the pinned pip
   package — a class defined in Apache-2.0 package code — not by
   `transformers.AutoModel` with Hub-hosted code.
5. The codec is a first-class transformers architecture
   (`transformers/models/higgs_audio_v2_tokenizer/`), not remote code either.

`docs/scope.md`'s rule against executable code in a Model Package is not
engaged.

### Tensor inventory: counts as expected, and one buffer that is not a weight

| File | Tensors | Parameters | Prefix histogram |
| --- | --- | --- | --- |
| `model.safetensors` | 313 | 612,577,288 | `llm.` 310, `audio_embeddings` 1, `audio_heads` 1, unprefixed 1 |
| `audio_tokenizer/model.safetensors` | 527 | 201,400,553 | `semantic_model.` 210, `acoustic_encoder.` 110, `acoustic_decoder.` 110, `quantizer.` 64, `decoder_semantic.` 14, `encoder_semantic.` 13, `fc1` 2, `fc2` 2, `fc` 2 |

Dtypes: **F32 everywhere except one I64 tensor.**

| File | F32 | I64 |
| --- | --- | --- |
| `model.safetensors` | 312 | 1 |
| `audio_tokenizer/model.safetensors` | 527 | 0 |

The single I64 tensor is `codebook_layer_offsets`, shape `[8]`, values
`[0, 1025, 2050, 3075, 4100, 5125, 6150, 7175]`. Checked rather than assumed:
`torch.equal` against `arange(8, dtype=int64) * 1025` returns `True`. It is a
derivable buffer, not a weight, and the converter derives it.

The generator's 310 `llm.` tensors are 28 layers × 11 plus `embed_tokens` and
`norm`. There is **no `lm_head` tensor**, which is `tie_word_embeddings: true`
showing up in the file rather than only in the configuration.
`audio_embeddings.weight` and `audio_heads.weight` are both `[8200, 1024]`, and
8200 is 8 × 1025 — one shared table read at a per-codebook offset, on both the
input and the output side.

### The codec's RVQ codebooks are live tables, not EMA accumulators

This is where qwen3-tts's shape does **not** repeat, and assuming it did would
have doubled the quantizer's size for nothing.

Each of the eight quantizers carries four codebook tensors: `embed [1024, 64]`,
`embed_avg [1024, 64]`, `cluster_size [1024]`, `inited [1]`. In qwen3-tts the
codebook had to be reconstructed as `embedding_sum / clip(cluster_usage, 1e-5)`.
Here it does not: `HiggsAudioV2TokenizerEuclideanCodebook.decode` is

```python
quantized = F.embedding(embed_ind.to(self.embed.device), self.embed)
```

`embed` is the live table. `embed_avg`, `cluster_size` and `inited` are training
state — 24 tensors the converter drops.

### `audio_tokenizer/config.json` disagrees with its own weights, three times

| Field | Config says | Tensors say |
| --- | --- | --- |
| `acoustic_model_config.n_codebooks` | 9 | 8 — `quantizer.quantizers.0`…`.7` exist, `.8` does not |
| `acoustic_model_config.codebook_dim` | 8 | 64 — every `codebook.embed` is `[1024, 64]`; the top-level `codebook_dim` of 64 is the right one |
| `acoustic_model_config.sampling_rate` | 16000 | 24000 output; 16000 is the semantic branch's rate |

Size the codebooks and the codebook count from the tensors. This is the same
failure mode qwen3-tts hit, in a different family, in the same file position.

### 195 codec tensor names crowd `GGML_MAX_NAME`

Measured as input to Task 7's `NAME_SHORTENINGS`: with a hypothetical `codec.`
prefix, **195 of the codec's 527 names reach 58 characters or more** against
`GGML_MAX_NAME` of 64. The longest is

```
codec.semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original1
```

at 82 characters — an 18-character overrun, not a near miss. The bulk come from
`semantic_model.encoder.layers.N.*` (HuBERT). The generator side has no problem
at all: its longest name is 45 characters
(`llm.layers.27.post_attention_layernorm.weight`).

The `parametrizations.weight.original0` / `original1` pair is also a note for
the converter: it is torch weight-norm state, needing reconstruction rather than
a straight copy.

### Greedy decoding is byte-identical across processes, and it makes no RNG call

Two separate Python processes, `position_temperature=0.0,
class_temperature=0.0`, `postprocess_output=False, pad_duration=0.0,
fade_duration=0.0`, text `"OmniVoice speaks with one voice."`, language `en`,
auto-voice:

- 48,000 samples each — 2.000 s at 24 kHz, 50 codec frames
- `cmp(1)`: **BYTE-IDENTICAL**
- sha256 `2710bbd2cffb9f4f7bce3ea3164ffd0908c6e6c496debd3fb9292a8337a6fbb1` for
  both dumps; `max_abs_diff` exactly 0.0
- wall 22.621 s and 23.164 s, real-time factor 11.31 and 11.58
- finite, peak 0.5, RMS 0.03857519955724174

This is the property the whole validation strategy rests on, and it is stronger
than qwen3-tts's. There, greedy identity came from setting two sampling switches
off inside a seeded process. Here both temperatures at zero mean the decode loop
takes neither Gumbel branch, so it makes **no RNG call at all** — the identity
is a property of the arithmetic, which is why it survives a process boundary.

Unseeded sampling at the package defaults behaves as expected and sets the
stochastic capability: two runs differ at byte 1, `max_abs_diff` 0.849.

**A duration comparison would report this family as deterministic when it is
not.** All four runs — greedy and sampled — produced exactly 48,000 samples,
because the canvas length is fixed by the deterministic `RuleDurationEstimator`
before the first forward. qwen3-tts's sampled runs differed in *length*
(51,840 against 76,800 frames) because its length is an autoregressive stop
decision. Any determinism check written for this family must compare content.

Real-time factor 11.3–11.6 is CPU F32 on the 20-core aarch64 GB10 host with the
GPU unused. Upstream's README claims RTF as low as 0.025 on GPU. The two
numbers measure different things, and neither predicts this project's GGML CPU
speed.

### Peak-normalise-to-0.5 survives every switch the contract turns off

The family document left this open, expecting intake to measure it. It is
active, and it is not small.

`OmniVoice._post_process_audio` has three stages. Only the first is gated:

```python
if gen_config.postprocess_output:
    generated_audio = remove_silence(...)

if ref_rms is not None and ref_rms < 0.1:
    generated_audio = generated_audio * ref_rms / 0.1
elif ref_rms is None:
    peak = np.abs(generated_audio).max()
    if peak > 1e-6:
        generated_audio = generated_audio / peak * 0.5

generated_audio = fade_and_pad_audio(
    generated_audio, pad_duration=..., fade_duration=..., sample_rate=...
)
```

`remove_silence` is skipped by `postprocess_output=False`. `fade_and_pad_audio`
is reached but inert: at `pad_duration=0` and `fade_duration=0` both
`fade_samples` and `pad_samples` are 0 and every branch inside is skipped, so it
returns a copy.

**The volume branch in the middle is gated on nothing.** Three cases:

| Condition | Effect | When |
| --- | --- | --- |
| `ref_rms is not None and ref_rms < 0.1` | `audio * ref_rms / 0.1` | clone with a quiet reference |
| `ref_rms is None` | peak-normalise to exactly 0.5 | auto-voice and voice-design |
| `ref_rms is not None and ref_rms >= 0.1` | nothing — the chain falls through | clone with a loud reference |

Confirmed by measurement, not only by reading: all four smoke runs report peak
**exactly 0.5**, greedy and sampled alike, despite entirely different content.
`ref_rms_list = [None] * batch_size` whenever no `voice_clone_prompt` is passed.

The clone branch is the inverse of a forward operation at reference-encode time,
`if 0 < ref_rms < 0.1: ref_wav = ref_wav * 0.1 / ref_rms`. Note the gate
asymmetry: the forward gate is `0 < ref_rms < 0.1`, the inverse gate is
`ref_rms < 0.1`. A digitally silent reference (`ref_rms == 0.0`) is therefore
not boosted on input but **is multiplied by zero on output** — an edge the port
reproduces or deliberately rejects, not one it discovers later.

This scaling is part of the family's contract. A port that reproduced the codec
perfectly and skipped this would be wrong by a per-utterance gain factor on
every request, and the golden tolerances would not catch it because the oracle
dumps carry the scaling too. Where it *lives* in the port — inside the synthesis
path, or hoisted into a documented normalisation control — is left open for
Plan 2.

### Upstream ships no pinnable reference audio anywhere

Step 5 wanted an upstream-hosted, revision-addressable audio artifact for clone
cases. There is none, and all three candidate locations were checked:

- the source repository at `468e927b` contains no `.wav`, `.flac`, `.mp3`,
  `.ogg` or `.m4a` file anywhere in the tree;
- the Hugging Face Space `k2-fsa/OmniVoice` at revision `64deef91` ships 50
  files, every one of them code;
- the authors' demo page at `zhu-han.github.io/omnivoice` does host audio, but
  GitHub Pages serves no revision-addressable URL and `zhu-han/omnivoice`
  exposes no `gh-pages` branch, so it cannot be pinned by revision.

The README's own clone example passes `ref_audio="ref.wav"` — a placeholder for
a file upstream never ships.

Recorded as a deviation and resolved by **pinning content instead of
revision**. The chosen artifact is
`https://zhu-han.github.io/omnivoice/audios/seedtts/prompt/seedtts_ref_en_1.wav`,
675,496 bytes, sha256 `57f25abc…`, RIFF PCM_16 mono at 24000 Hz, 337,726
frames, 14.0719 s. A digest pin is in one respect stronger than a revision pin:
if the bytes change, the manifest fails loudly rather than silently
substituting.

It was chosen over the alternative for two reasons. It is already at the model's
native 24 kHz as uncompressed PCM, so neither a resampler nor an audio codec
enters the comparison. And the demo page supplies its transcript, which a clone
case needs and which upstream supplies nowhere else:

> Some call me nature. Others call me Mother Nature. I've been here for over
> four point and five billion years, twenty-two thousand five hundred times
> longer than you.

Two caveats travel with it. At 14.07 s it exceeds upstream's own 3–10 s
reference recommendation, though it stays under the 20 s `trim_long_audio`
threshold. And it originates from the SeedTTS evaluation set, whose own licence
upstream does not state — the file stays git-ignored like every other model
input.

The cleaner-licensed alternative is recorded in `intake.json` rather than
discarded: `audios/minimax/prompt/common_voice_en_42357710.mp3` (74,731 bytes,
sha256 `adc1bb51…`) is Mozilla Common Voice, CC0. It loses on both other counts
— the demo page gives no transcript for it, and an MP3 puts a decoder in the
comparison path. Task 5 owns the final manifest choice and should not have to
rediscover the trade.

### The pre-tokenizer matches qwen3-tts's, by the weaker of the two comparisons

The seven alternatives in `tokenizer.json`'s pre-tokenizer regex are
character-for-character the pattern documented in `src/arch/qwen3-tts/bpe.cpp`
lines 108–117. Compared against the committed C++ implementation's own
transcription, because the qwen3-tts weights are not present on this host — a
weaker comparison than diffing the two `tokenizer.json` files, and recorded as
such. Task 8 should redo it against both files when it hoists the module.

The tokenizer ships as a **single `tokenizer.json`** — no `vocab.json`, no
`merges.txt`, unlike qwen3-tts. 151,643 base entries plus 33 added tokens is
151,676, which matches `llm_config.vocab_size` exactly; 151,387 merges. The
seven TTS markers occupy 151669–151675 in the documented order, all
`special: true`, and `tokenizer_config.json` lists exactly these seven under
`extra_special_tokens`. `bos_token` is null on both sides.

### Open decisions carried into stage 2

- Where the residual output scaling lives in the port — inside the family's
  synthesis path, or hoisted into a documented normalisation control. The
  golden tolerances depend on the answer.
- Generator on CUDA: claimed only if placement evidence proves the committed
  token grids bit-identical to CPU. Stage 7.
- Whether any quantization profile survives the exact-token gates against the
  argmax cascade. Stage 6, and a failing profile is simply not shipped.
- The Gumbel replay seam, implemented only if a sampled-path parity case proves
  it necessary.
- Voice-design instruct passthrough: v1 does not reimplement
  `_resolve_instruct`.
- Which clone reference artifact the manifest pins — Task 5.
- Whether the codec encoder half ships in the Model Package at all. Plans 1 and
  2 need only the decoder side; the split is roughly `semantic_model` (210) plus
  `acoustic_encoder` (110) plus `encoder_semantic` (13) against
  `acoustic_decoder` (110) plus `decoder_semantic` (14) plus `quantizer` (64).

## 2026-07-30 — Conversion (stage 3): 798 tensors emitted

`scripts/convert-omnivoice.py` produced
`models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf`, 3,189,953,536 bytes,
sha256 `b03fcdf81e7a4ef650f715cf13f29bef078d29ebf7c3c19fd7c39fb3b6f9a256`.

**Emitted tensor count: 798**, all F32. Task 10's `expected_tensor_count`
arithmetic must reproduce this:

| Half | Source | Skipped | Folded | Emitted |
| --- | ---: | ---: | ---: | ---: |
| Generator (`model.safetensors`) | 313 | 1 | 0 | **312** |
| Codec (`audio_tokenizer/model.safetensors`) | 527 | 40 | 2 → 1 | **486** |
| | | | | **798** |

The 41 skipped source tensors, itemized:

- 1 — `codebook_layer_offsets`, the checkpoint's only I64 tensor. Dropped only
  after the converter proves it still equals `arange(8) * 1025`; if the values
  ever move, the conversion stops rather than deleting information.
- 2 — `fc1.*` and 14 — `decoder_semantic.*`. Both feed the training-only
  semantic reconstruction loss. `HiggsAudioV2TokenizerModel.encode` (transformers
  5.14.1) uses `encoder_semantic`, `acoustic_encoder`, `fc`, `quantizer`; `.decode`
  uses `quantizer`, `fc2`, `acoustic_decoder`. Neither reaches `fc1` or
  `decoder_semantic`.
- 24 — the per-quantizer k-means state, 3 × 8: `codebook.embed_avg`,
  `codebook.cluster_size`, `codebook.inited`. The intake's finding holds against
  the weights: `codebook.embed` is a **live table** and nothing is reconstructed.

The weight-norm fold turns
`semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original0/1`
into one `…pos_conv_embed.conv.weight` (2 → 1). It goes through
`torch._weight_norm`, the operator the reference itself calls on every forward,
so the folded kernel is **bit-identical** — a hand-written `g·v/‖v‖` agreed only
to ~5e-9, which is enough to move an argmax this family commits and feeds back.

**The encode half ships.** Cloning is in this family's product surface, so
`semantic_model` (210), `acoustic_encoder` (110), `encoder_semantic` (13) and
`fc` (2) are all carried — the opposite of qwen3-tts's encoder drop. That
settles the last open decision from the intake list in favour of shipping it.

### `NAME_SHORTENINGS`: four rules, all load-bearing

59 emitted names overrun `GGML_MAX_NAME` before shortening; the longest is 77.
Rules are applied per path component so the catalog reads uniformly, which is
why 153 names are rewritten to fix those 59.

| Rule | Names rewritten | Overflows it alone fixes |
| --- | ---: | ---: |
| `.attention.` → `.attn.` | 96 | 2 |
| `.feed_forward.` → `.ff.` | 48 | 48 |
| `.intermediate_dense.` → `.inter_dense.` | 24 | 24 |
| `.feature_extractor.conv_layers.` → `.feat_conv.` | 9 | 9 |

Longest emitted name after shortening: 62 characters. No two names collide.
The brief's other two candidate rules — `.final_layer_norm.` and
`.pos_conv_embed.` — were **dropped**: neither name overflows once the fold has
collapsed the parametrization pair, and a rename nobody needs only makes the
catalog harder to read. `test_every_shortening_rule_is_load_bearing` re-derives
this from the committed inventory, so a fifth gratuitous rule cannot be added
silently.

### The three config disagreements were resolved from the tensors

Recorded in the conversion report under `config_disagreements` rather than
silently resolved: `acoustic_model_config.n_codebooks` 9 against 8 measured
tables, `acoustic_model_config.codebook_dim` 8 against measured 64,
`acoustic_model_config.sampling_rate` 16000 against the top-level 24000. The
package is sized from `quantizer.quantizers.N.codebook.embed`, all eight of them
`[1024, 64]`, and `add_metadata` refuses to write a geometry the tables
contradict.

### Licence carriage

`audio_tokenizer/LICENSE` was copied byte-identically to
`models/omnivoice-0-6b/LICENSE-higgs-audio-2.txt`, 9,171 bytes, sha256
`ac933dc084d119bd20401956b90d11ae87c248b2da62622cd580d82cdf2fa049` — the digest
the intake pinned. The report's `licenses` list carries all three grants, with
the CC-BY-NC sentence quoted verbatim from the model card.

## 2026-07-30 — C++ foundation (stage 4, slice 1): the package loads

The last task of Plan 1 closed the loop from converter to loaded model. Before
it, every C++ test in this family ran against a package the test itself built,
which can only prove the rules agree with each other. `synthesize-omnivoice-load-real`
runs them against the checkpoint.

**Emitted against expected: 798 = 798.** The load-real test opens the package,
reads its hyper-parameters and compares `gguf_get_n_tensors` with
`expected_tensor_count(hparams)` before loading anything; both are 798, the
count the stage-3 conversion emitted. The catalog then resolves every one of
them and sweeps the package for a name it never asked for. **The sweep passed
unchanged** — the catalog's shape derivations and the converter's rename rules
agreed on the real weights at the first attempt, so nothing here was fixed by
relaxing a check.

**Unit tests registered** under the `unit;omnivoice` label, four of them:
`synthesize-omnivoice-metadata-test`, `-catalog-test`, `-frontend-test`, and this
slice's `-model-errors-test` (empty path → `SYNTH_ERR_INVALID_ARG`, missing file
→ `SYNTH_ERR_FILE_NOT_FOUND`, a non-GGUF file and an empty GGUF container →
`SYNTH_ERR_GGUF`, with the handle left null after each). The whole gate is green:
72/72 in `build`, 72/72 in `build-sanitize`.

**Sanitizer clean on the real package too.** `synthesize-omnivoice-load-real`
was also built and run in the ASan/UBSan configuration — 52 s, no report — so the
3.2 GB weight buffer is allocated, streamed and freed without a leak or an
undefined operation, not merely without a crash. `build-sanitize` was
reconfigured for this run with `-DSYNTH_BUILD_INTEGRATION_TESTS=ON` against the
locally converted GGUF; the standing sanitize configuration keeps integration
tests off.

### What the smoke asserts, and why the public half is there

The family half reads the package's own numbers back: variant `omnivoice-0-6b`,
F32 profile, 24 kHz mono, 960 samples per frame, a text vocabulary of 151,676,
tags `en`/`zh`/`ja`, and a frontend built from the package's vocabulary and
merges rather than from anything compiled in.

The public half then loads the same file through `synth_model_load` and checks
what the C interface publishes: the three languages carry
`SYNTH_LANGUAGE_REGIONAL_FALLBACK` and **none carries** `SYNTH_LANGUAGE_DEFAULT`,
because a request naming no language gets the language-agnostic prompt — the
literal `None` slot — rather than falling back to one. The Preset Voice Catalog
is empty and `has_package_default` is true: identity arrives through Voice
Profiles and the package default is the unnamed auto-voice.

Finally it calls `synth_synthesize` and requires `SYNTH_ERR_INTERNAL` with the
diagnostic id `synthesis.not_implemented`. That assertion is not ceremony around
a stub. Removing the stub branch and re-running the smoke **segfaults**: the
request falls through to the VITS branch and dereferences a null model pointer.
A family that can be loaded but not synthesized has to refuse loudly, and this
is the test that says so. Plan 2's first task replaces the branch and the
assertion together.

### One gap the plan did not name: `synth_model_get_device`

The task's family header carried no `primary_device()`, on the reasoning that
Plan 1 runs no graph. But `synth_model_get_device` is a documented query —
docs/c-interface.md: "returns the Loaded Model's actual primary device" — and it
selects by walking the handle's family pointers. Without the accessor an
omnivoice model answered it with `SYNTH_ERR_BACKEND`: a successfully loaded model
that could not say where it lived. The accessor was added and the arm wired,
matching all three earlier families, and the smoke now asserts the reported kind
is `cpu`.

### Seams touched

`ModelFamily::Omnivoice` in `src/model-info.h`, the handle member and the
`"omnivoice"` architecture branch in `src/synthesize.cpp`, a fourth `shared_info`
overload, the load branch, and the synthesis stub. `include/synthesize.h` is
untouched: this family added no public ABI surface, which is the point of the
seam being where it is.

## 2026-07-30 — Ruling: `general.license` set to `other`

The final review's parked finding — the GGUF's machine-readable `general.license`
key carried `cc-by-nc-4.0`, a clean SPDX slug for a grant the upstream card
never version-states — was ruled on: `scripts/convert-omnivoice.py` now writes
`license_id="other"`, the same choice `convert-vits.py` makes for checkpoint
redistribution terms that are not a clean SPDX identifier. `license_name` and
`license_link` are unchanged; they already carry the full three-license story
("CC-BY-NC (version unstated upstream) + Boson Higgs Audio 2 Community License
(codec)" at `https://huggingface.co/k2-fsa/OmniVoice`). The ship-stage card
frontmatter is a separate, later decision.

Re-running the converter against the same locally held weights reproduced the
same shape — 798 tensors, tensor-for-tensor identical to the run above — with a
new file sha256, because the metadata changed:
`3ecaa5e2f6fbd735296ba1cd60680c90467be22d2140dc4f208fe80111ecb9e5`, superseding
`b03fcdf81e7a4ef650f715cf13f29bef078d29ebf7c3c19fd7c39fb3b6f9a256` recorded
above. `synthesize-omnivoice-load-real` was re-run against the re-cut package
and passed.

## 2026-07-31 — Oracle hardening: pinned threads/determinism moved `audio/pcm.f32`

Plan 2 Task 1 hardened `scripts/dump_reference_omnivoice_pytorch.py` two ways,
closing the two carry-over items from Plan 1's final review:

- **Execution is now pinned**, immediately after the runtime `import torch`:
  `torch.set_num_interop_threads(1)`, `torch.set_num_threads(1)`,
  `torch.use_deterministic_algorithms(True)`. Every dump now records the
  configuration it ran under — torch version, thread counts, the
  deterministic-algorithms flag, and `torch.backends.cpu.get_cpu_capability()`
  — in `metadata.json`'s new `"environment"` block (per case) and the dump
  report's top-level `"environment"` block. Before this, the twenty exact-token
  baselines were proven bit-reproducible only across two runs on this one
  machine with its ambient thread count; nothing pinned that configuration or
  recorded it.
- **Every weights-repository input the dump reads is now sha256-verified**
  against the manifest's `source.artifacts` before torch is even imported:
  both `model.safetensors` files, both `config.json` files, and
  `tokenizer.json`. Previously only the clone reference wav was digest-checked;
  the weights, configs and tokenizer were trusted. A deliberately corrupted
  `tokenizer.json` (RED, `/tmp/omni-fake`, made of symlinks to the real big
  files plus one corrupted small file) was rejected before any import or model
  load, in under three seconds wall time (dominated by `uv`'s environment
  sync, not the check): `tokenizer.json: sha256 … does not match the
  manifest's …; refusing to dump against unpinned inputs`, exit 1.

**GREEN — one case re-dumped under the pinned configuration, `omni-short-en`
(greedy).** `codes/grid.i32` is byte-identical to the pre-change dump
(`60473d82…`): the committed token grid does not move under thread pinning, as
expected for a family whose greedy decode makes no RNG draw. `audio/pcm.f32`
**did move** — `2710bbd2…` (unpinned, ambient configuration) to `7a063dac…`
(all three settings pinned together: one thread, one interop thread,
deterministic algorithms required). What was measured is exactly those two
digests with three settings changed at once, plus one negative result (the
grid did not move); that is not enough to say which of the three settings
moved the bytes, nor which pipeline stage. The unchanged grid only clears the
discrete argmax/Gumbel boundary — it says nothing about the continuous
backbone hidden states upstream of it. So the honest claim is: some float32
stage downstream of the stable discrete grid is sensitive to one or more of
the three pinned settings. The DAC decoder is the leading hypothesis, since
floating-point output first reaches the caller there, but it is not isolated
by this run. Task 2 should confirm by comparing the intermediate generator
artifacts (`generator/hidden_l*.f32`, `generator/logits_step0.f32`) before and
after in its full re-dump — a move there would localise the sensitivity to the
backbone rather than the codec. Per the task brief this move is not reverted
regardless of where it turns out to live: the pinned configuration supersedes
the unpinned one that produced every baseline dumped so far. The twenty
existing case directories under `build/goldens/omnivoice/` (git-ignored, not
committed) are stale for their audio artifact until Task 2's full re-dump
regenerates all of them consistently under this configuration; no comparison
or tolerance work has been built against them yet, so nothing downstream
depended on the old digests. `metadata.json`'s `environment` block for the
re-dumped case reads
`num_threads: 1, num_interop_threads: 1, deterministic_algorithms: True,
torch_version: 2.13.0+cu130, cpu_capability: SVE128`.

## 2026-07-31 — Oracle hardening and re-dump (Plan 2 Tasks 1–2)

The dumper now pins torch to one intra-op and one inter-op thread with
deterministic algorithms demanded, records that configuration in every
`metadata.json`, and sha256-verifies all six weights-repository inputs against
the manifest before loading anything. `audio_chunk_duration`/`audio_chunk_threshold`
are pinned at 15.0/30.0 in every case (upstream's own defaults at the pinned
revision); 30 s equals the 750-frame package ceiling, so no golden case can
take the chunked long-form path, and `load_manifest` now refuses a manifest
where that stops being true (`threshold_frames < max_output_frames`, checked
against `package_contract.max_output_frames` right after the two keys are
required). Pinning the two keys touched only `oracle.parameters` in all 20
cases — one added trailing-comma line plus two new keys per case, 60 insertions
/ 20 deletions, nothing else reformatted (`git diff --stat`).

**Full re-dump under the pinned configuration was NOT a clean no-op: 90 of 226
binary artifacts moved.** The literal Step 3 command hung a first background
run after ~81s (it wrote two of twenty cases and stopped silently, with the
tracked process gone and no exit code — the sandbox appears to reap a
long-running background job once the turn that started it stops actively
supervising it, not a bug in the dumper). Because `run_case` builds every
artifact in memory and only calls `write_case` once at the very end, the
half-finished third case's on-disk directory was untouched by the aborted
run — confirmed by its `metadata.json` still lacking the `environment` block
before the retry. The recovery was 7 supervised foreground invocations of the
same script, each with a `--case` subset small enough to finish inside one
tool call, sequentially, against the same weights and manifest:

    uv run --project scripts/envs/omnivoice --locked python scripts/dump_reference_omnivoice_pytorch.py \
      --manifest tests/golden/omnivoice/omnivoice-0-6b.manifest.json \
      --weights-dir models/omnivoice-0-6b \
      --report /tmp/omnivoice_batchN.json \
      --case <id> [--case <id> ...]

run for the batches `{omni-upstream-readme, omni-short-en, omni-short-zh,
omni-short-ja, omni-lang-none}`, `{omni-punctuation, omni-digits,
omni-nonverbal, omni-fast-mode}`, `{omni-medium-en, omni-rate-slow,
omni-rate-fast}`, `{omni-long-boundary}`, `{omni-design-en, omni-design-zh,
omni-clone-en, omni-clone-zh}` (this one hit a 10-minute tool timeout on the
fourth case, `omni-clone-zh`, after the first three wrote cleanly; the killed
attempt left no partial files for the same in-memory-then-write reason above),
`{omni-clone-zh}` alone, and `{omni-sampled-seed-zero, omni-sampled-seed-one,
omni-sampled-seed-forty-two}`. The seven per-batch `dump-report.json`s were
concatenated (in manifest case order) into
`build/goldens/omnivoice/dump-report.json`, the one used below and by every
later task; `total_wall_seconds` across the batches that actually finished
sums to 1811.6s (≈30.2 CPU-minutes), close to the brief's "tens of minutes"
estimate despite the reload overhead of splitting into seven processes. Every
greedy case's in-process double-run replay was identical and no greedy case's
RNG state moved (`double_run_identical: true`, `rng_untouched: true` for all
17), so within one fixed configuration the family is exactly as deterministic
as previously established; what follows is about drift *across* the
ambient-to-pinned configuration change, not within it.

The byte-identity check was the brief's literal Step 3, filter included, since
carry-over 10 requires the command actually run, not an idealized one:

    find build/goldens/omnivoice -name '*.i32' -o -name '*.f32' | grep -v '_smoke\|_pre_redump\|-replay' \
      | sort | xargs sha256sum > /tmp/omnivoice_pre_redump.sha256
    # ... seven batched invocations in place of the single one above ...
    find build/goldens/omnivoice -name '*.i32' -o -name '*.f32' | grep -v '_smoke\|_pre_redump\|-replay' \
      | sort | xargs sha256sum > /tmp/omnivoice_post_redump.sha256
    diff /tmp/omnivoice_pre_redump.sha256 /tmp/omnivoice_post_redump.sha256

226 files on both sides, 0 added, 0 removed, 90 with a changed digest — not
`BINARIES-UNCHANGED`. Per the supersession rule this pinned+chunked dump is
the new baseline regardless; nineteen of the twenty case directories had never
been re-dumped since Plan 1 (only `omni-short-en` was, as a Task 1 smoke test),
so this is the first time the thread/determinism pinning actually ran against
them. `omni-short-en` itself is fully unchanged top to bottom — its `before`
state in this diff was already Task 1's pinned dump, so this run only added
the (behaviourally inert) chunk keys to its `metadata.json`; both its
`codes/grid.i32` (`60473d82…`) and `audio/pcm.f32` (`7a063dac…`) digests
reproduce Task 1's recorded values exactly, an independent cross-check that
this run used the identical configuration.

**Divergence isolation, the open question Task 1 left (`generator/hidden_l*.f32`,
`generator/logits_step0.f32` old vs. new): the sensitivity is not confined to
the codec.** Two loci are now demonstrated, separately:

- **Codec-stage sensitivity, proven directly.** Eighteen of the nineteen
  cases whose audio changed kept `codes/grid.i32` — the full committed
  discrete token sequence across every frame, not just step 0 — byte-identical
  between the ambient and pinned dumps. A moved `audio/pcm.f32` fed by a
  byte-identical discrete input can only come from the codec/DAC-decode stage
  itself, so for these eighteen cases the DAC decoder's floating-point
  reduction order is confirmed sensitive to the pinned settings, exactly Task
  1's original hypothesis.
- **Backbone-stage sensitivity, also proven directly, and independent of the
  above.** Ten of the twenty cases (`omni-short-zh`, `omni-punctuation`,
  `omni-digits`, `omni-medium-en`, `omni-rate-slow`, `omni-design-en`,
  `omni-design-zh`, `omni-clone-en`, `omni-clone-zh`, `omni-fast-mode`) moved
  one or more of the seven step-0 generator probes
  (`generator/hidden_l{0,7,14,21,27}.f32`, `generator/final.f32`,
  `generator/logits_step0.f32`) — e.g. `omni-punctuation`'s
  `logits_step0.f32` went `b07c407b…` to `688e6365…`. Nine of those ten still
  left `codes/grid.i32` unmoved: the backbone's own reduction order drifted
  measurably at the very first forward pass, but not by enough to flip any
  argmax/Gumbel-boundary decision across the whole generation.
- **The two loci compound in the suite's thinnest margin.** `omni-fast-mode`
  (`num_step: 16`, the fewest diffusion refinement steps of any case — every
  other case runs 32) is the only case where `codes/grid.i32` itself moved
  (`ce41f5b2…` to `70ebe309…`), and its generator probes moved too (all but
  `hidden_l0.f32`, which — like `omni-punctuation`'s — stayed identical while
  every later layer diverged, for reasons this run does not investigate
  further). The move is exactly 1 of the grid's 400 `int32` slots — codebook 7
  of 8, frame 8 of 50 (0-indexed), `1004` to `237` — the other 399 slots and
  the grid's length both unchanged, no cascade. Fewer refinement steps leave
  the per-frame distribution less converged, i.e. closer to a tie, so the
  same drift that seventeen 32-step cases absorbed without a discrete change
  was enough here to cross the argmax boundary at that one frame. This is
  consistent with, not proof of, that causal story. An exact-token comparison
  for this case therefore has essentially no margin: whoever writes
  `omni-fast-mode`'s comparison logic must know this case sits on a knife's
  edge, where a single flipped code is expected from ordinary floating-point
  variation (a different compiler, a different BLAS/GGML kernel) and is not
  by itself evidence of a porting bug.

As before, this run changed three settings at once (one thread, one interop
thread, deterministic algorithms demanded) relative to the ambient dumps it is
compared against, so it still cannot attribute the drift to a single one of
the three; it answers *where* the sensitivity shows up, not *which* pinned
knob is responsible: codec-stage sensitivity is proven for 18 of the 20 cases
(everything except `omni-fast-mode`, whose own grid also moved, so its audio
change is not attributable to the codec in isolation, and `omni-short-en`,
where nothing moved at all); backbone-stage sensitivity is separately proven
for half the suite. `build/goldens/omnivoice/` (git-ignored) now holds this
pinned+chunked dump uniformly across all 20 cases; no comparison or tolerance
work has been built against any of the superseded digests, so nothing
downstream depended on them. The superseded ambient dump and the batch
records behind the numbers above live only under `/tmp/`
(`/tmp/omnivoice_pre_redump_backup/`, `/tmp/omnivoice_pre_redump.sha256`,
`/tmp/omnivoice_batch*.json`) and will not survive this machine; `/tmp` is
ephemeral, so the digests and counts recorded in this log, not those files,
are the durable record of this comparison.

## 2026-07-31 — slice 4 — single-forward parity

The port's step-0 **conditional** forward now runs end to end through the
replay harness (`tests/omnivoice_replay_real.cpp` +
`scripts/validate-omnivoice-replay.py`) and is compared against the oracle's
seven step-0 probes on **all 20 golden cases**. Every case ran; every case
passed. Measurement mode only — `tests/tolerances/omnivoice.json` still has no
`profiles` key, so `--check` refuses rather than passing vacuously, and Task 14
is what turns these numbers into reviewed thresholds.

Worst across the suite (`--require probes`, F32/CPU, 20/20 cases):

| probe | worst max_abs | worst mean_abs | min cosine |
|---|---:|---:|---:|
| `generator.hidden_l0` | 5.72205e-05 | 4.4685e-07 | 0.999999870 |
| `generator.hidden_l7` | 0.000411987 | 1.67257e-06 | 0.999999803 |
| `generator.hidden_l14` | 0.000701904 | 4.91952e-06 | 0.999999905 |
| `generator.hidden_l21` | 0.00415039 | 3.37815e-05 | 0.999999863 |
| `generator.hidden_l27` | 0.0800781 | 0.000195814 | 0.999999786 |
| `generator.final` | 0.0010376 | 5.28952e-06 | 0.999999828 |
| `generator.logits_step0` | 0.000610352 | 4.32935e-05 | 0.999999909 |

Three readings worth keeping:

- **The `hidden_l27` max_abs is not a defect, and this family shows why the
  cosine rule exists.** `l27` is the last block's output, read *before* the
  final RMSNorm, and it carries the same outlier channels the qwen3-tts port
  documented: its max_abs is two orders of magnitude larger than `l21`'s while
  its cosine is 0.9999998 and its mean_abs is 2e-04. The very next probe,
  `generator.final` — the same tensor after the norm — drops back to 1e-03
  max_abs. A max-abs threshold set from `l21` would report catastrophic failure
  on a correct port at `l27`.
- **The divergence is monotone in depth and flat in canvas length.** `l0` sits
  at 4e-05 on every one of the 20 cases regardless of whether the canvas is 41
  or 820 positions, and each later probe is roughly one accumulation step
  larger. That is the signature of ordinary F32 reduction-order difference
  compounding through 28 blocks, not of a structural error, which would show up
  as a step change at one depth or as growth with sequence length.
- **The prompt grid matched byte-for-byte on all 20 cases before any forward
  ran.** The runner assembles the grid with `build_prompt_grid` and `memcmp`s
  it against the oracle's `input/prompt_grid.i32`; that covers the two
  reference-audio clone cases (471 and 477 positions, 351 reference frames
  each) and the two voice-design cases as well as the no-reference ones. The
  check is not vacuous: feeding it a transposed 84×8 grid in place of the
  oracle's 8×84 is refused with exit 2.

Wall times, F32 CPU, `default_synthesis_threads()` on this 20-CPU machine: the
whole 20-case sweep took **48.6 s** including 20 separate loads of the 3.0 GiB
F32 package. Only **8.2 s** of that is the runners; the generator forward
itself is 0.10 s at 41 positions, 0.79 s at 357, 1.14 s at 471 and **2.06 s at
820** (`omni-long-boundary`, the longest canvas in the suite). Graph *setup*
— scheduler creation plus placement, which a per-step rebuild would pay 32
times over — is 0.30–0.40 ms per forward across all 20 cases, at most 0.31% of
the forward it precedes. That number is what stage 7's graph-reuse question
gets to argue against; on this evidence a fresh `GraphRun` per step is not
where the time goes.

Placement: 832 nodes per forward, **0 off the CPU** on every case, as
docs/backends.md's discrete-output rule requires. The validator fails the run
if any node leaves the CPU, so this is enforced rather than observed.

> Read the two paragraphs above per *graph*, not per *run*. Every number in
> them comes from a probe-only run, which makes exactly one forward, so
> `generator_seconds`, `generator_setup_seconds` and
> `generator_placement.nodes` each held one graph's value. Those three fields
> have always accumulated; from slice 5 on, a decode run makes two forwards per
> step and they report totals — 53 184 nodes for a 32-step run, not 832. See
> the slice 5 entry.

Not covered by this slice, and deliberately: the **unconditional** CFG branch
has no oracle artifact, so nothing here compares it — its tensors are
allocated in the persistent input buffer and Task 10 owns both its refill and
its forward. Likewise the greedy loop and the codec both still refuse loudly
(`SYNTH_ERR_INTERNAL`), which is what the runner's `run-greedy 1` and
`decode-replay 1` flags exercise today.

## 2026-07-31 — slice 5 — free-running greedy decode: 15/17 exact

The free-running mask-predict loop is in (`src/arch/omnivoice/model.cpp`,
`run_synthesis`) and reproduces the oracle's 8 × T token grid **exactly on 15
of the 17 greedy golden cases**. The two that differ do so for the same
reason, and it is not a rule this port got wrong: both are F32 near-ties that
the port and the oracle broke in opposite directions. For one of the two,
`omni-fast-mode`, that is close to demonstrated rather than argued — the port's
grid is **byte-identical to the ambient-torch dump T2 superseded**, so the
disputed slot is one torch itself has already been observed to move. The gate
was not relaxed, no fixture was touched, and the exact-token comparison stays
exact.

Wall clocks, F32 CPU, `default_synthesis_threads()` on this 20-CPU machine.
"forwards" is `2 × num_step` — the reference's one batched forward per step is
realized here as two, the conditional over the whole prompt and the
unconditional over the target region alone.

| case | frames | steps | exact | mismatches | runner wall (s) | generator (s) | forwards |
|---|---:|---:|:--:|---:|---:|---:|---:|
| `omni-upstream-readme` | 67 | 32 | yes | 0 | 10.9 | 10.8 | 64 |
| `omni-short-en` | 50 | 32 | yes | 0 | 8.1 | 8.0 | 64 |
| `omni-short-zh` | 54 | 32 | yes | 0 | 8.6 | 8.5 | 64 |
| `omni-short-ja` | 47 | 32 | yes | 0 | 7.5 | 7.4 | 64 |
| `omni-lang-none` | 49 | 32 | yes | 0 | 7.8 | 7.7 | 64 |
| `omni-punctuation` | 67 | 32 | yes | 0 | 10.5 | 10.4 | 64 |
| `omni-digits` | 132 | 32 | yes | 0 | 20.0 | 19.7 | 64 |
| `omni-nonverbal` | 61 | 32 | yes | 0 | 9.7 | 9.6 | 64 |
| `omni-medium-en` | 307 | 32 | yes | 0 | 44.2 | 43.7 | 64 |
| `omni-long-boundary` | 719 | 32 | yes | 0 | 122.2 | 120.4 | 64 |
| `omni-rate-slow` | 100 | 32 | yes | 0 | 14.3 | 14.2 | 64 |
| `omni-rate-fast` | 25 | 32 | yes | 0 | 5.0 | 4.9 | 64 |
| `omni-design-en` | 50 | 32 | yes | 0 | 8.5 | 8.4 | 64 |
| `omni-design-zh` | 49 | 32 | yes | 0 | 8.1 | 8.0 | 64 |
| `omni-clone-en` | 70 | 32 | yes | 0 | 37.2 | 37.0 | 64 |
| `omni-clone-zh` | 76 | 32 | **no** | 574 / 608 | 38.8 | 38.6 | 64 |
| `omni-fast-mode` | 50 | 16 | **no** | 1 / 400 | 4.0 | 4.0 | 32 |

The whole 20-case sweep (17 greedy free-runs plus the three sampled cases'
probe-only forwards) is **365 s of runner wall**, ~9 min end to end including
20 loads of the 3.0 GiB F32 package. The step-0 probe rows are unchanged from
slice 4 to the digit — the loop's first forward IS slice 4's forward — so the
`--require probes` gate still passes 20/20.

### The two mismatches are one phenomenon, measured

Both were instrumented directly: a temporary build printed, per step, every
committed position whose token differs from the oracle's final grid, that
position's guided log-probabilities for both candidate tokens, and the
**selection margin** — the score of the lowest-ranked position the step kept
against the highest-ranked one it rejected. The instrumentation is not
committed; its numbers are.

**`omni-fast-mode` — a token flip at 6.9e-05, no cascade.** The single
differing slot is codebook 7, frame 8, committed at **step 15 of 16** (the
last step). The port commits token 1004, the oracle 237, and the two tokens'
guided log-probabilities are `-2.65722704` and `-2.65729570` — a gap of
**6.87e-05**. The conditional branch alone prefers 237 by 0.79 nats; CFG at
`guidance_scale = 2.0` amplifies the unconditional branch's disagreement until
the two are level to five decimal places. Because the flip lands on the final
step, nothing downstream of it re-runs, which is why exactly 1 of 400 slots
moved.

**And this is not merely the same slot class the T2 re-dump caught — it is the
same slot, and the port lands byte-for-byte on the grid torch itself produced
before the pinning.** T2 recorded `omni-fast-mode`'s grid moving `ce41f5b2…` to
`70ebe309…` under the thread/determinism pinning, "exactly 1 of the grid's 400
`int32` slots — codebook 7 of 8, frame 8 of 50 (0-indexed), `1004` to `237`".
The superseded ambient dump was still on this machine under
`/tmp/omnivoice_pre_redump_backup/`, so the three digests could be compared
directly:

| grid | sha256 |
|---|---|
| this port's free run | `ce41f5b2c83b87a927c9a8609b38445ad51ee07470b448a5fd467e9605c9ebc4` |
| T2-superseded **ambient torch** dump | `ce41f5b2c83b87a927c9a8609b38445ad51ee07470b448a5fd467e9605c9ebc4` |
| current **pinned torch** oracle | `70ebe3093eb691c9409f0fdc8c525c875cd68781958045b4cf24313f8e8ff05a` |

The port's grid **is** the ambient-torch grid, all 400 slots, bit for bit — a
C++/GGML implementation independently reproducing a specific output of the
reference implementation. The one slot separating both of them from the pinned
oracle is flat index 358 = (codebook 7, frame 8), `1004` vs `237` — the same
slot, the same two token ids, the same direction that a torch thread-count
change alone produced. This is no longer "plausibly floating-point noise": the
port's arithmetic landed on the side of the boundary that one torch
configuration takes, and the oracle records the side another torch
configuration takes. `/tmp` is ephemeral and will not survive this machine, so
these digests — not that directory — are the durable record.

**`omni-clone-zh` — a *position* flip at 4.1e-06, cascading.** The 574/608
figure is not 574 independent errors; it is one coin flip at **step 1** and 30
steps of consequence. Step 0's two commits match the oracle's final grid.
Step 1 commits three positions: two match the oracle's final grid, and the
third does not — the port writes `1011` at (codebook 0, frame 0) where the
oracle's final grid holds `250`. What the instrumentation shows is that this
third commit won its slot by four parts in a million:

```
step=1 budget=3 committed=3 masked=606
  kept_low       = -0.872900724  [codebook 0, frame  0, token 1011]
  best_rejected  = -0.872904778  [codebook 0, frame 68, token  252]
  select_margin  =  4.05e-06
```

The verified fact, and the only one this evidence supports, is about the
**rejected** candidate: the oracle's final grid holds **`252` at (codebook 0,
frame 68)** — the exact token of the candidate this port rejected by 4.05e-06.
The oracle dumps no per-step grid, so which step it committed that slot at is
*not* observable here; what is observable is that the port and the oracle
disagree about a slot the port lost by four parts in a million, and that the
oracle's answer there is the one the port's runner-up carried. From step 2
onward the two canvases hold different committed sets, so every later forward
sees different context and the grids separate almost completely. The large
per-token margins reported at steps 2+ (2 to 22 nats) are *downstream of* that
divergence, not evidence of independent rule errors.

Two things this case does **not** have, stated so the two mismatches are not
read as equally well-evidenced:

- **No digest corroboration.** Unlike `omni-fast-mode`, `omni-clone-zh`'s grid
  did not move in the T2 re-dump: its ambient and pinned dumps are the same
  file (`64d2b0119f968b0428dcbaa7b9fc04e59d4603d6b453493e3a6f91556e6d0c16`),
  and the port's grid (`e61a871132d399ec535072192af79bafc0724bbcb31ecca88605e47a4104c05e`)
  matches neither. Its case rests on the 4.05e-06 margin and on the rejected
  candidate carrying the oracle's token — strong, but inference, not the byte
  identity `omni-fast-mode` has.
- **No per-step oracle.** Establishing the step-1 position swap beyond
  inference would need the oracle to dump its canvas per step, which the
  dumper does not do today. That is a cheap change if a ruling wants it.

`omni-clone-en` is the control: structurally identical to `omni-clone-zh` —
same 351-frame reference, same 50-token text region, only 70 frames instead of
76 — and it matches the oracle on all 560 slots.

### How close the whole suite runs to the boundary

Minimum selection margin per case, over every step that committed anything:

| case | min select margin | at step | diverged |
|---|---:|---:|:--:|
| `omni-clone-zh` | **4.05e-06** | 1 | yes |
| `omni-rate-slow` | **9.54e-06** | 28 | no |
| `omni-short-en` | 1.16e-04 | 7 | no |
| `omni-long-boundary` | 2.05e-04 | 11 | no |
| `omni-rate-fast` | 2.44e-04 | 10 | no |
| `omni-lang-none` | 6.03e-04 | 11 | no |
| the remaining 11 | 1.2e-03 … 5.0e-03 | — | no |

Two cases sit below 1e-05. One of them flipped and one did not; `omni-rate-slow`
kept a candidate that beat its rival by 9.5e-06 and happened to agree with the
oracle. That is luck, not correctness, and it is the honest reading of this
gate: at 15/17 the port is not *nearly* right on two cases, it is exactly right
on every case whose outcome F32 arithmetic can actually determine.

The scale that settles it: slice 4 measured this port's step-0 logits against
the oracle's at **6.1e-04 worst max_abs** (cosine 0.99999991), ordinary F32
reduction-order divergence compounding through 28 blocks. The CFG combination
`log_softmax(3·log p_c − 2·log p_u)` amplifies rather than damps that. A 4.1e-06
score gap is ~150× below the measured logit divergence and a 6.9e-05 gap ~9×
below it. Neither is a quantity F32 arithmetic on this graph can resolve, in
either direction. No threshold would help: a tolerance on a token id is
meaningless, and the only "fix" that would move these two cases is to make the
port's arithmetic bit-identical to torch's, which is not a thing this project
claims anywhere.

### What is deliberately NOT concluded here

That 15/17 is acceptable. It is a **finding for the plan owner to rule on**,
not a standard this task lowered. `docs/porting/families/omnivoice.md` defines
`structural_exactness` for this family as equality of the 8 × T grid, and
`scripts/validate-omnivoice-replay.py` still exits 1 on anything short of
17/17. Whatever the ruling, it has to be written into the family doc and the
Golden manifest rather than absorbed silently by the validator.

### Loop mechanics worth recording

- **Two forwards per step, not one.** The reference batches a conditional and
  an unconditional sequence into one padded forward; padding them to a common
  length here would buy nothing on the CPU and would need the padded-diagonal
  attention mask upstream carries for exactly that reason. Two separate
  forwards are the same arithmetic without the mask.
- **The unconditional canvas is the target region alone**, positions restarting
  at zero, refilled every step from the same committed canvas the conditional
  branch's target region carries. No oracle artifact compares it, so the
  exact-token gate is the only thing holding that definition honest — which,
  given 15/17 with both failures explained by near-ties, it now does.
- **Already-committed positions are skipped rather than scored at `-inf`.**
  Upstream masks them with `-inf` before `topk`; the budget is clamped to what
  remains masked, so a `-inf` candidate can never be selected and excluding it
  from the candidate list is the same selection with a smaller sort.
- **`generator_placement.nodes` is now a sum over forwards.** The conditional
  graph places 832 nodes and the unconditional one 830 (it has no text
  embedding to merge), so a 32-step run reports **53 184** and a 16-step run
  **26 592**, against **832** for a probe-only run. It was 832 flat in slice 4;
  the slice-4 entry's "832 nodes per forward" reading still holds per graph,
  but the field it was read from no longer reports one graph. **0 off the CPU**
  on every case, still enforced by the validator rather than observed.
- The `--require grid` path no longer has a not-built stage to name: the greedy
  loop's refusal string and its row in the validator's `NOT_BUILT_MARKERS` were
  deleted together, leaving only the codec's for Task 12.

## 2026-07-31 — Ruling implemented: dual-admissible grids, and the gate is 17/17

jiangzhuo's ruling on the slice-5 15/17 result, implemented. Neither of the two
mismatches was a rule this port got wrong and neither was absorbed by a
threshold: `omni-fast-mode` keeps its case under a **dual-admissible-grid
contract** — byte-exact against an enumerated set of torch-produced grids, zero
tolerance — and `omni-clone-zh` is **re-picked** under a documented
margin-screening standard. The exact-token comparison is unchanged; what
changed is how many enumerated answers count as exact, and which text one case
asks the model to say.

```
token grids exact: 17/17
narrowest margin: 9.53674e-06 (selection) in omni-rate-slow; 2 of 17 case(s) under the 0.0001 screen
EXIT: 0
```

### The witness was rescued from `/tmp` first

The slice-5 entry above records the decisive evidence for `omni-fast-mode`: the
port's grid is byte-identical to the **ambient-torch dump T2 superseded**, and
the one slot separating both from the pinned oracle is the same slot, the same
two ids and the same direction that a torch thread-count change alone produced.
That dump lived only in `/tmp/omnivoice_pre_redump_backup/`, and the slice-5
entry said so — "`/tmp` is ephemeral and will not survive this machine, so
these digests, not that directory, are the durable record."

The ruling makes that grid a contract, so the bytes had to become durable too.
Rescued and committed before anything else in this task, digest verified
against the table above:

| | sha256 |
|---|---|
| `/tmp/omnivoice_pre_redump_backup/omni-fast-mode/codes/grid.i32` | `ce41f5b2c83b87a927c9a8609b38445ad51ee07470b448a5fd467e9605c9ebc4` |
| `tests/golden/omnivoice/omni-fast-mode.alternate-grid-1.i32` | `ce41f5b2c83b87a927c9a8609b38445ad51ee07470b448a5fd467e9605c9ebc4` |

1600 bytes. This is a **contract witness**, not a golden payload, and the
distinction is not a euphemism: the payload rule exists so that regenerable
bulk artifacts stay out of git, and this grid is *not regenerable* — the
ambient thread configuration that produced it was never pinned, and Task 2
pinned the dumper precisely so that configuration can never recur. It is 1.6 KB,
immutable, pinned by digest in the manifest, and it carries contract force: the
validator will accept a port grid that equals it.

### Dual admissibility, and why it is not a tolerance

`oracle.alternate_grids` is now an optional per-case array in the Golden
Manifest schema (additive; every other manifest validates unchanged): `file`
manifest-relative and committed, `sha256`, and free-text `provenance`.
`scripts/validate-omnivoice-replay.py` reads each witness, **verifies its
digest against the manifest before comparing anything**, and passes the case if
the port grid elementwise-equals the primary oracle grid *or* any alternate. It
names the grid that matched, in the per-case line and in the report JSON:

```
omni-fast-mode: token grid exact against omni-fast-mode.alternate-grid-1.i32 of 2 admissible grids
```

No tolerance parameter exists anywhere on this path, and none was added. The
target *set* widened by one enumerated, committed, provenance-carrying
reference output; the *comparison* is still elementwise equality. The
difference matters: a tolerance would admit outputs nobody has ever seen, and
this admits exactly one output torch itself produced.

`tests/python/test_golden_manifests.py` gains a family-independent check that
every `alternate_grids` entry's file exists and still hashes to its recorded
digest — the one failure mode an admissible *set* has that a single answer does
not, and it runs in the unit gate because the witness is committed.

### Margin instrumentation, now committed

Task 10's forensics ran on a temporary instrumented build that was never
committed (its method is in `task-10-report.md` §3). That is replaced by
`--margin-report` on the replay runner, which discharges the corresponding part
of Task 13's tooling debt.

`SynthesisRequest::margin_report` asks the greedy loop to track its narrowest
decision and report it in `SynthesisOutput::margin`; the runner emits it as a
`"margin"` object in its JSON line (always present, `null` when unmeasured) and
the validator surfaces it per case. Two kinds, because the two slice-5
mismatches were one of each:

- **selection** — the guided-score gap between the last candidate a step
  committed and the best one it rejected. Close it and a different *position*
  is committed, which changes every later step's context. This is
  `omni-clone-zh`'s old failure.
- **argmax** — the gap between a committed position's token and its runner-up,
  from a new `runner_up_gap` out-parameter on `choose_token`. Close it and a
  different *token* lands in that slot. This is `omni-fast-mode`'s.

Argmax margins are collected on steps that commit everything still masked,
which reject nothing and would otherwise report no margin at all; in practice
that is the final step, the one the schedule hands the whole remainder. The
scope limit is real and is stated in the family doc: a narrow argmax on a
partially committing step is not screened.

One structural change came with it. `select_commits`' ordering lambda is now
the exported `commits_before`, because the margin code has to find the best
*rejected* candidate after `std::partial_sort` has left the tail unordered — a
second copy of that tie-break rule would eventually drift and report a margin
against a candidate the selection never considered.

### The whole suite, measured with the committed instrument

`--require grid --margin-report`, all 17 greedy cases:

| case | min margin | kind | at step | cb | frame |
|---|---:|---|---:|---:|---:|
| `omni-rate-slow` | **9.53674e-06** | selection | 28 | 4 | 80 |
| `omni-fast-mode` | **6.86646e-05** | argmax | 15 | 7 | 8 |
| `omni-short-en` | 1.15871e-04 | selection | 7 | 0 | 19 |
| `omni-long-boundary` | 2.04682e-04 | selection | 11 | 0 | 237 |
| `omni-rate-fast` | 2.44433e-04 | selection | 10 | 0 | 9 |
| `omni-lang-none` | 6.03199e-04 | selection | 11 | 0 | 13 |
| `omni-design-zh` | 6.40869e-04 | argmax | 31 | 7 | 46 |
| `omni-medium-en` | 7.17163e-04 | argmax | 31 | 7 | 297 |
| `omni-punctuation` | 8.39233e-04 | argmax | 31 | 7 | 32 |
| `omni-upstream-readme` | 1.14441e-03 | argmax | 31 | 6 | 54 |
| `omni-clone-en` | 1.19019e-03 | selection | 29 | 5 | 13 |
| `omni-clone-zh` | **1.28174e-03** | selection | 25 | 2 | 53 |
| `omni-digits` | 1.39546e-03 | argmax | 31 | 7 | 102 |
| `omni-design-en` | 1.41111e-03 | selection | 3 | 0 | 47 |
| `omni-nonverbal` | 1.89209e-03 | argmax | 31 | 7 | 28 |
| `omni-short-zh` | 2.46429e-03 | selection | 29 | 5 | 36 |
| `omni-short-ja` | 2.66457e-03 | selection | 25 | 2 | 33 |

**A correction to the slice-5 reading.** That entry reported an empty band
between 9.54e-06 and 1.16e-04 — a 12× gap — and the ruling was drafted against
it. That band is *not* empty once argmax margins are measured: `omni-fast-mode`
sits inside it at 6.87e-05, which is exactly the number the slice-5 forensics
had already published for that case's token flip. The slice-5 table listed
selection margins only, so `omni-fast-mode`'s ≥1.2e-03 selection margin put it
among "the remaining 11" and its real knife edge did not appear. Nothing about
the ruling changes; the *rationale* for the 1e-4 constant does, and the family
doc states the corrected one: the screen separates the two cases that required
a ruling (9.5e-06, 6.9e-05) from every case that did not (1.16e-04 and up).

### `omni-clone-zh` re-picked — a case change, not oracle drift

`omni-clone-zh` had the weaker of the two evidence tiers, and the slice-5 entry
said so explicitly: no digest corroboration (its ambient and pinned dumps were
the same file, and the port's grid matched neither) and no per-step oracle, so
its whole case rested on a 4.05e-06 selection margin plus the inference that
the rejected candidate carried the oracle's token. Rather than admit a grid on
inference, the ruling re-picks the case.

| | old | new |
|---|---|---|
| text | `克隆的声音读出这句话。` | `克隆的声音也要说中文的句子。` |
| frames | 76 | 98 |
| oracle grid sha256 | `64d2b0119f968b0428dcbaa7b9fc04e59d4603d6b453493e3a6f91556e6d0c16` | `44075f6cef0607f7dd1d257ae880cef10796d672c9332a7e3cc1989e007ce896` |
| port grid | `e61a871132d399ec535072192af79bafc0724bbcb31ecca88605e47a4104c05e` (matched nothing) | equals the oracle, 784/784 |
| min margin | 4.05e-06 (selection, step 1) | **1.28e-03** (selection, step 25) |

The old grid digest `64d2b011…` is **superseded by a text change under this
ruling — it is not oracle drift**, and it must not be read as one: the dumper,
its pinned inputs, the weights, the thread pinning and every oracle parameter
are byte-for-byte what Task 2 fixed. What changed is the sentence the case asks
the model to say. Everything else about the case is unchanged: same seedtts
reference audio and `ref_text`, same case id, same coverage tags
(`voice-clone`, `cross-lingual-clone`, `greedy-exact`), same 32 steps.

**Re-pick attempts.** The screen is part of the record, so every attempt is
listed. There was one:

| # | text | frames | min margin | verdict |
|---:|---|---:|---:|---|
| 1 | `克隆的声音也要说中文的句子。` | 98 | 1.28174e-03 | **accepted** — 12.8× the 1e-4 screen, and the grid is exact |

Re-dump: 151.8 s of model wall, 307 s end to end, `[1/1] omni-clone-zh`,
6 pinned inputs verified against the manifest before it ran.

**The tokenizer dump had to be refreshed with it, and this was nearly missed.**
`scripts/dump_reference_omnivoice_tokenizer.py` produces a second, separate
oracle — `build/goldens/omnivoice/tokenizer/cases.json` — whose duration rows
are transcribed into `tests/omnivoice_frontend_test.cpp` as the committed
statement of that file. A case's text is an input to *both* dumpers, so
re-picking the text staled the frontend fixtures while every replay gate stayed
green: the replay runner takes its frame count from the oracle artifacts and
never calls `DurationEstimator` at all, so the estimator's only clone-zh
coverage would have gone on testing a text no golden case uses. Re-run against
the updated manifest (`13 pre-tokenizer strings, 53 id rows, 20 prompts, 24
duration rows`), the case moves:

| | old | new |
|---|---:|---:|
| `total_weight` | 30.5 | **39.5** |
| `estimated_frames` | 76 | **98** |

with `ref_text_total_weight` 140.4 and `num_ref_audio_tokens` 351 unchanged —
the reference audio did not move. The estimator's 98 frames is the same 98 the
PyTorch oracle dumped, which is the cross-check that matters: two independent
reference paths agree on the new canvas length. `combined_text` for the case
was refreshed from the same dump. All 24 duration rows were then compared
against the fresh `cases.json` programmatically; only `omni-clone-zh` moved.

The general rule this earns: **a golden case's text feeds every oracle the
family has, and re-picking one means re-running all of them.** For omnivoice
that is two dumpers, and only one of them is guarded by a gate that would have
noticed.

Manifest relations were re-checked and none moves: `omni-clone-zh` belongs to
no `artifact_differs` relation. The same-text trio is
`omni-short-en`/`omni-design-en`/`omni-clone-en`, all English; the zh design and
clone texts were never required to match, which is why re-picking one of them
is a local change.

### Gates

| gate | result |
|---|---|
| `--require grid`, 17 greedy cases, `--margin-report` | **17/17 exact**, exit 0 |
| — `omni-fast-mode` | exact against `omni-fast-mode.alternate-grid-1.i32` (the alternate) |
| — the other 16 | exact against the primary oracle grid |
| `--require probes`, 20 cases | 20/20, exit 0; probe rows unchanged from slice 4 |
| `synthesize-golden-manifest-contract` | passes, including the new digest check |

## 2026-07-31 — slice 6 — end-to-end greedy waveform: 20/20 cases, 17/17 grids

`Model::decode_codes` is wired to Task 11's `build_codec_decoder`, and
`run_synthesis` now ends where a caller expects it to: committed grid → codec →
no-reference volume branch → `output.audio`. The slice-6 gate passes on the
whole suite.

```
uv run --project scripts/envs/omnivoice --locked python scripts/validate-omnivoice-replay.py \
  --model models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf \
  --runner build/bin/synthesize-omnivoice-replay-real \
  --require all --margin-report --report build/goldens/omnivoice-replay/full-report.json
```

**20/20 ok, 17/17 grids exact, no node off the CPU, no non-finite sample,
exit 0.** 7 min 33 s wall (411 s of it inside the runner), 901 % CPU on this
20-CPU machine.

| probe | max_abs | min_cosine |
|---|---:|---:|
| `audio.pcm` | 1.68812e-05 | 0.99999986 |
| `audio.pcm_freerun` | 1.68812e-05 | 0.99999986 |
| `generator.final` | 0.0010376 | 0.99999983 |
| `generator.hidden_l0` | 5.72205e-05 | 0.99999987 |
| `generator.hidden_l7` | 0.000411987 | 0.99999980 |
| `generator.hidden_l14` | 0.000701904 | 0.99999991 |
| `generator.hidden_l21` | 0.00415039 | 0.99999986 |
| `generator.hidden_l27` | 0.0800781 | 0.99999979 |
| `generator.logits_step0` | 0.000610352 | 0.99999991 |

The generator rows are slice 4's to the digit; the two new rows are the
waveform. `1.7e-05` on a signal whose peak is exactly `0.5` is 3.4e-5 relative —
F32 against F32, which is what the tolerance file's
`source-f32-oracle-vs-f32-cpu` reference stage was chosen to make possible.
Measured as an error SNR rather than as a bound, three cases: **102.2 dB**
(`omni-short-en`), **104.9 dB** (`omni-medium-en`), **118.8 dB**
(`omni-short-zh`).

### Two waveform channels, and they are not the same claim

`audio.pcm` replays the **oracle's** grid through the port's codec, which
isolates the codec from the decode loop. `audio.pcm_freerun` is what
`run_synthesis` returned for the grid **the port itself chose** — the only
artifact that can tell a wired-up codec from a codec that merely exists. Both
now ride the one runner (`pcm.f32`, `pcm_freerun.f32`).

The free-run channel needed two rulings the slice-6 brief predates:

1. **The clone cases.** Plan 2's `run_synthesis` applies the no-reference branch
   unconditionally — decision 3 of the plan, and the other two arms key off a
   reference RMS only Plan 3's cloning path can supply. The two clone cases'
   oracle waveforms carry **no** scaling (`ref_rms` 0.1229 > 0.1 → branch
   `none`), so the port's free-run waveform is the oracle's times `0.5/peak`.
   The validator therefore compares the free-run waveform against
   `peak_normalise(oracle pcm)`. For the 14 no-reference cases that transform is
   the identity to the bit — `x / 0.5 * 0.5` is exact in binary floating point —
   so one rule covers both. It is not a weakened comparison for the clones
   either: their `audio.pcm` row is the gain-sensitive one, compared raw against
   the unscaled oracle, and it passes at 3.5e-06 / 4.3e-06.
2. **`omni-fast-mode`, the dual-admissible case.** Its free-run grid is the
   committed *alternate*, and no oracle waveform exists for that grid — the
   oracle's `audio/pcm.f32` was produced from the primary. Comparing the two
   would report a difference the case does not claim. So this one case is
   **exempt from oracle parity on the free-run channel**, and gets a
   decode-determinism comparison instead: the runner's new `--alt-grid` decodes
   the same committed alternate through the same seam, and the two must be
   **byte-identical**. Measured: identical, `max_abs` exactly 0. That is a
   weaker claim, deliberately, and the validator prints it as such
   (`free-run waveforms: 16 against the oracle, 1 exempt (decode determinism)`)
   rather than folding it into the parity count. `omni-fast-mode`'s *replay*
   channel is unaffected and still carries full oracle parity at 2.87e-06.

### Per-case

| case | frames | `pcm` max_abs | `pcm` cosine | free-run max_abs / cosine | grid matched | runner wall (s) | codec (s) |
|---|---:|---:|---:|---|---|---:|---:|
| `omni-upstream-readme` | 67 | 4.96e-06 | 1.00000007 | 4.96e-06 / 1.00000007 | primary | 11.0 | 0.263 |
| `omni-short-en` | 50 | 1.17e-05 | 0.99999993 | 1.17e-05 / 0.99999993 | primary | 8.7 | 0.184 |
| `omni-short-zh` | 54 | 1.07e-06 | 0.99999993 | 1.07e-06 / 0.99999993 | primary | 9.3 | 0.202 |
| `omni-short-ja` | 47 | 2.81e-06 | 1.00000000 | 2.81e-06 / 1.00000000 | primary | 8.2 | 0.209 |
| `omni-lang-none` | 49 | 1.36e-05 | 1.00000006 | 1.36e-05 / 1.00000006 | primary | 8.5 | 0.209 |
| `omni-punctuation` | 67 | 7.00e-06 | 1.00000021 | 7.00e-06 / 1.00000021 | primary | 11.6 | 0.254 |
| `omni-digits` | 132 | 1.12e-05 | 0.99999995 | 1.12e-05 / 0.99999995 | primary | 20.8 | 0.563 |
| `omni-nonverbal` | 61 | **1.69e-05** | 0.99999990 | 1.69e-05 / 0.99999990 | primary | 20.4 | 0.249 |
| `omni-medium-en` | 307 | 6.68e-06 | **0.99999986** | 6.68e-06 / 0.99999986 | primary | 48.6 | 1.538 |
| `omni-long-boundary` | 719 | 8.38e-06 | 0.99999991 | 8.38e-06 / 0.99999991 | primary | 129.5 | 3.690 |
| `omni-rate-slow` | 100 | 2.83e-06 | 0.99999999 | 2.83e-06 / 0.99999999 | primary | 15.9 | 0.451 |
| `omni-rate-fast` | 25 | 3.25e-06 | 1.00000004 | 3.25e-06 / 1.00000004 | primary | 5.3 | 0.080 |
| `omni-design-en` | 50 | 2.06e-06 | 0.99999995 | 2.06e-06 / 0.99999995 | primary | 8.9 | 0.188 |
| `omni-design-zh` | 49 | 2.99e-06 | 1.00000001 | 2.99e-06 / 1.00000001 | primary | 8.7 | 0.193 |
| `omni-clone-en` | 70 | 3.54e-06 | 0.99999990 | 2.83e-06 / 0.99999994 | primary | 38.6 | 0.273 |
| `omni-clone-zh` | 98 | 4.34e-06 | 1.00000005 | 2.69e-06 / 1.00000004 | primary | 51.2 | 0.414 |
| `omni-fast-mode` | 50 | 2.87e-06 | 1.00000004 | exempt — identical (max_abs 0) | `omni-fast-mode.alternate-grid-1.i32` | 4.7 | 0.182 |
| `omni-sampled-seed-zero` | 46 | 1.30e-06 | 0.99999995 | — (sampled: nothing free-runs) | — | 0.3 | 0.000 |
| `omni-sampled-seed-one` | 46 | 2.64e-06 | 0.99999992 | — | — | 0.3 | 0.000 |
| `omni-sampled-seed-forty-two` | 46 | 1.88e-06 | 0.99999997 | — | — | 0.3 | 0.000 |

The clone rows are the only ones where the two waveform columns differ, and they
differ downward (3.54 → 2.83, 4.34 → 2.69): both cases' oracle peak exceeds 0.5,
so normalising shrinks the error along with the signal. Cosines above 1.0 are
the comparison's own float rounding on near-identical vectors, not a
measurement.

### The codec's cost, measured

`codec_seconds` is a flat **0.0027 s per frame** across the whole range —
0.080 s at 25 frames, 3.690 s at 719 — and the placed node count is **422 for
every case**, 25 frames and 719 alike. That is the qwen3-tts codec budget
holding: one pass over the whole stream, node count independent of length. The
codec is **2.9 % of a case's wall time** at 719 frames and under 4 % everywhere;
the generator's 64 forwards remain the whole cost of this family.

The three sampled cases report `codec_seconds` 0.000 and codec placement 0 while
their `audio.pcm` compares fine. That is not a hole in the decode — the replay
decode ran, it is what `pcm.f32` holds — it is that `decode_codes` reports its
timing through file-local scratch that only `run_synthesis` reads, and a
probe-only request returns before reaching it. The validator's CPU-only rule is
correspondingly vacuous for those three; the same codec graph reports CPU-only
on the other 17, so nothing is actually unwatched. Left for the T13 harness
batch.

### Listening

Not done by the implementer — an agent cannot listen, and
`docs/porting/families/omnivoice.md`'s standard is a human A/B, not a number.
What can be said numerically is that the port has nothing of its own to hear:
port and oracle agree to 102–119 dB SNR, and the waveform's shape statistics are
**identical to four decimals** — `omni-short-en` max sample step 0.3274 and DC
+5.91e-03 on both, `omni-medium-en` 0.5064 / +1.38e-04 on both, `omni-short-zh`
0.2309 / +2.14e-04 on both, with the largest step at the *same sample index* in
each pair. Any click or offset a listener finds is the model's, not this port's.
WAV pairs for the listening pass can be regenerated from
`build/goldens/omnivoice-replay/<case>/pcm.f32` against
`build/goldens/omnivoice/<case>/audio/pcm.f32`.

### The unbuilt-stage markers are retired

`scripts/validate-omnivoice-replay.py`'s `NOT_BUILT_MARKERS` table and
`model.cpp`'s `"codec decode is slice 6 and not built yet"` are deleted
together, as that table's own rule required. Every stage the three `--require`
levels name is now built, so a runner failure under any of them is a real
failure and there is no third answer. `git log -S "NOT_BUILT_MARKERS"` has the
mechanism if a later plan wants it back.

### What the unit gate now covers, and what it cannot

`tests/omnivoice_decode_loop_test.cpp` gains the codec: it is the only execution
of `Model::decode_codes` that ASan/UBSan ever sees, since the real package is
not committed. It asserts one hop of samples per frame, every sample finite, the
volume branch's fixed point (an exact peak of 0.5 — reached because the sample
that attained the peak becomes `peak / peak * 0.5`), codec placement on the CPU,
the refusals (mask id, negative, past-the-table, three wrong element counts,
zero frames), decode determinism across two calls, and the **wiring as an
identity**: `run_synthesis`'s audio IS its own grid put back through
`decode_codes` and the shared volume helper.

Two mutations, applied to `model.cpp`, built, run, reverted:

| mutation | caught by | signal |
|---|---|---|
| `apply_no_reference_volume` dropped from `run_synthesis` | `check_waveform` | peak 0.114649877, not 0.5 — on all three synthesis cases |
| `validate_code_grid` dropped from `decode_codes` | the refusal cases | `GGML_ASSERT(i01 >= 0 && i01 < ne01) failed` in `ggml-cpu/ops.cpp:4902` |

The second one is worth keeping: without the host guard an out-of-range code
does not return an error, it **aborts inside ggml**. That is the house rule
`validate_code_grid` exists to uphold, now demonstrated rather than asserted.

What this fixture cannot catch: its weights are one repeated constant, so the
committed grid is all zeros, and a mutation that decoded a *permuted* grid would
produce the same waveform. Only the real-package `audio.pcm_freerun` channel
catches that, and it does — the orientation trap would show as cosine ≈ 0.

### Gates

| gate | result |
|---|---|
| `--require all`, 20 cases, `--margin-report` | **20/20 ok, 17/17 grids exact**, exit 0, 7:33 wall |
| — free-run waveform | 16 against the oracle, 1 exempt (decode determinism), 3 sampled (no free-run) |
| — placement | every case CPU-only; codec 422 nodes, 0 accelerator |
| `cmake --build build --target synthesize-check-unit` | 78/78 passed |
| `cmake --build build-sanitize --target synthesize-check-unit` (ASan/UBSan) | 78/78 passed |
| real package (3.0 GiB F32 GGUF) | loaded and synthesised 20 times in the sweep |

### Self-review additions

Two guards on the new free-run channel. **Neither can happen through this
validator's own path** — it always passes `--alt-grid` when a case pins an
alternate, and the runner always writes `pcm_freerun.f32` when it is asked for
both a greedy run and a decode. The point is not that either was reachable; it
is that the channel **no longer depends on that** being true, which is what
made the first cut's correctness an argument rather than a check.

- **The work directory is reused across runs**, so a `pcm_freerun.f32` or
  `pcm_alt.f32` left by an earlier invocation, or missing entirely, would have
  been read as this run's output. The validator now cross-checks each against
  the sample count the runner reported in its own JSON (`freerun_samples`,
  `alternate_samples`) and returns `stale-freerun-waveform` rather than
  comparing — absence goes down the same road as a size disagreement, so a
  missing waveform is a reported result and not a `FileNotFoundError`
  traceback.
- **A case matching an alternate with no `--alt-grid` decode to compare
  against** now fails loudly instead of passing quietly. Demonstrated by a
  **hand invocation** with the `--alt-grid` argument patched out of the command
  — not something the validator can produce for itself:
  `omni-fast-mode: matched omni-fast-mode.alternate-grid-1.i32, but no decode of
  that grid was produced to compare its free-run waveform against`, **exit 1**.

`--require probes` and `--require grid` re-smoked after the change (exit 0
each): the free-run channel is inert at both levels, as intended.

**For Task 14:** the worst-table has a new row, `audio.pcm_freerun`, which
`--check` will demand a tolerance cell for alongside `audio.pcm`. Measured
values to write it from are in the table above.

### Fix round 1 (review): the `on_primary` comment was contradicting the header

`GraphRun::run`'s `on_primary` parameter comment still read *"Only the codec
ever asks for it"* — inherited from before this slice existed — while the same
commit's file header said the codec runs on the CPU scheduler like everything
else. Both cannot be true, and the header is the true one: **no Plan-2 caller
passes `on_primary` at all.** The generator may not (its output is a sampled
code, so docs/backends.md's discrete-outputs rule pins it and its whole input
path to the CPU), and the codec — the one stage that could — takes the false
default too, because Plan 2 has no measurement to move it on.

The parameter stays: it is the seam stage 7 needs to move a stage without
reworking the class, which is the qwen3-tts precedent. Its comment now says
exactly that, and names the discrete-outputs rule as what will decide which
stages may ever pass `true`. Comment-only; both gates re-run green afterwards.

Also folded: the `pcm_freerun.f32` read was unguarded where its `pcm_alt.f32`
sibling checks `.is_file()`, so a missing free-run waveform would have raised a
`FileNotFoundError` traceback instead of the `stale-freerun-waveform` result
the guard above it exists to produce. Absence now takes the same road as a size
disagreement. Negative control — the read pointed at a filename the runner
never writes — `omni-rate-fast: stale-freerun-waveform`, **exit 1**, no
traceback. The three-case real-package check re-run afterwards: 3/3 grids exact,
`audio.pcm` 3.54e-06, `audio.pcm_freerun` 3.25e-06, fast-mode still exempt and
identical, exit 0.

## 2026-07-31 — Plan 2 closeout — first tolerances committed, golden gate enforcing

The measurement that slices 4–6 produced is now a committed grid and a
registered CTest gate. `tests/tolerances/omnivoice.json` goes from
`pending-first-measurement`, which made `--check` refuse, to
`thresholds-committed-and-enforced`, which makes it gate.

### Provenance of the numbers

The thresholds are derived from **one** sweep — the first working reference and
implementation, per `docs/port-validation.md` — re-run at this commit before
being written down, and identical to slice 6's to every digit:

```bash
uv run --project scripts/envs/omnivoice --locked python scripts/validate-omnivoice-replay.py \
  --model models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf \
  --runner build/bin/synthesize-omnivoice-replay-real \
  --require all --margin-report --report build/goldens/omnivoice-replay/full-report.json
```

**20/20 ok, 17/17 grids exact, exit 0, 417.3 s wall.**

Every cosine threshold is five times the measured deviation in `(1 - cosine)`,
rounded DOWN to six decimals; `audio.pcm` and `audio.pcm_freerun` additionally
gate on max-abs at five times the measured value rounded UP to one significant
figure.

| probe | measured min_cosine | measured (1−cos) | committed min_cosine | actual headroom | measured max_abs | committed max_abs |
|---|---:|---:|---:|---:|---:|---:|
| `generator.hidden_l0` | 0.9999998701 | 1.299e-07 | 0.999999 | 7.7× | 5.72205e-05 | — |
| `generator.hidden_l7` | 0.9999998030 | 1.970e-07 | 0.999999 | 5.1× | 0.000411987 | — |
| `generator.hidden_l14` | 0.9999999054 | 9.458e-08 | 0.999999 | 10.6× | 0.000701904 | — |
| `generator.hidden_l21` | 0.9999998629 | 1.371e-07 | 0.999999 | 7.3× | 0.00415039 | — |
| `generator.hidden_l27` | 0.9999997860 | 2.140e-07 | **0.999998** | 9.3× | 0.0800781 | — |
| `generator.final` | 0.9999998281 | 1.719e-07 | 0.999999 | 5.8× | 0.0010376 | — |
| `generator.logits_step0` | 0.9999999094 | 9.057e-08 | 0.999999 | 11.0× | 0.000610352 | — |
| `audio.pcm` | 0.9999998558 | 1.442e-07 | 0.999999 | 6.9× | 1.68812e-05 | **9e-05** |
| `audio.pcm_freerun` | 0.9999998558 | 1.442e-07 | 0.999999 | 6.9× | 1.68812e-05 | **9e-05** |

**The headroom column is the reviewed part, and it is not 5× anywhere.** Six
decimals is the precision the qwen3-tts file uses, and there it is nearly
lossless because that family's deviations are 1e-04 to 5e-03. Here they are
1e-07, so the rounding dominates: eight probes collapse onto the same 1e-06
gate and `hidden_l27` onto 2e-06. Rounding the other way would have tightened
past the rule, so the coarser value is the one the rule permits — but writing
"5×" in the file and committing 11× would have been a false statement, which is
why the file states the ratio range.

There is a second and better reason not to cut finer, found while reviewing the
report rather than assumed: **the cosine estimator's own arithmetic is the
noise floor here.** `cosine()` accumulates in float32 over 24k–6.7M element
arrays, and on these same runs it returns values *above* 1.0 by up to
**2.08e-07** — the same scale as every deviation in the table above, and larger
than all but one of them (only `hidden_l27`'s 2.14e-07 exceeds it). A gate at 5×
(≈5e-07) would sit inside that noise and could fail on a thread-count change
with nothing wrong in the port. The waveform's max-abs is the channel that
keeps real resolution: it is a subtraction, not an accumulation, and 9e-05
against a measured 1.68812e-05 on a signal peaking at 0.5 is a threshold that
would actually catch a wrong gain.

One dependency recorded in the file itself: **`audio.pcm_freerun` is not
independent evidence on 14 of the 16 cases it covers.** For a no-reference case
that matched the primary grid, the oracle waveform is already peak-normalised
(so the port's volume branch is the identity to the bit) and the free-run grid
*is* the replayed grid, which makes both sides byte-identical to `audio.pcm`'s
comparison — the two probes' worst rows are the same numbers off the same case.
Only `omni-clone-en` and `omni-clone-zh` compare something new (2.83e-06 and
2.69e-06 against `audio.pcm`'s 3.54e-06 and 4.34e-06), and `omni-fast-mode` is
exempt on this channel. What the probe proves everywhere, and `audio.pcm`
cannot, is that the codec is wired into the synthesis path at all.

No threshold exists for the token grid and none ever will: `structural_exactness`
for this family is byte equality of the 8 × T grid against the oracle's or a
committed alternate's, and the validator fails an inexact grid before `--check`
is consulted.

### The gate

`synthesize-omnivoice-replay-golden`, registered beside the qwen3-tts golden
blocks and mirroring them: package + oracle-payload sentinel
(`build/goldens/omnivoice/omni-short-en/codes/grid.i32`, deliberately not
committed), the locked `scripts/envs/omnivoice` uv environment,
`--check --profile F32 --backend CPU --stage replay`, labels
`integration;omnivoice;golden`, `TIMEOUT 14400`, working directory at the
source root.

Run exactly as CI would, after a clean configure with
`-DSYNTH_BUILD_INTEGRATION_TESTS=ON`:

```
1/1 Test #87: synthesize-omnivoice-replay-golden ...   Passed  420.23 sec
100% tests passed, 0 tests failed out of 1
```

A `ctest -V` re-run (419.70 s, exit 0) captured what the validator actually
printed under the gate, since a passing CTest prints nothing: 17/17 grids
exact, `omni-fast-mode` exact against its alternate, 16 free-run waveforms
against the oracle and 1 exempt, and the closing line **`all probes within the
F32/CPU/replay tolerances`**. One wall-clock story, since three files now quote
it: about seven minutes for the twenty-case sweep on this 20-CPU host —
**417.3 s** for the measurement sweep above, **420.2 s** and **419.7 s** for the
two gate runs, and **453 s** at slice 6, which stands as the worst observed.
Four hours of timeout against that is headroom for a slower machine, not an
estimate.

**Negative controls, because a gate that cannot fail is not a gate.** Both ran
`--cases omni-short-en --check` against a doctored `--tolerances` file in
`/tmp`, leaving the committed one untouched:

- cell deleted (`profiles: {}`) → `tolerance cell F32/CPU/replay is not
  recorded in /tmp/t14-tol-nocell.json`, **exit 1**. This is the state the file
  was in before this commit, so it is also the proof that `--check` was
  refusing rather than passing vacuously all through Plan 2.
- `generator.hidden_l27` tightened to `0.99999999` →
  `generator.hidden_l27: cosine 0.99999998 breaches 0.99999999`, **exit 1**.

Both printed the full worst-table first: the measurement happens either way,
and only the verdict changes.

### What Plan 2 does NOT close

The public seam still answers `synthesis.not_implemented` for this family, and
that is deliberate: everything above is reached through the replay harness.
Stage 5 (Port Validate) is therefore partly done — the greedy replay half —
and the rest is owed:

- **Plan 3** — the public sampled path (Gumbel position draws, this project's
  seed contract, `class_temperature`'s class branch), Reference Audio cloning
  with the 24→16 kHz HuBERT resample and the quiet/zero-reference volume
  branches, Description Text voice design, Serialized Profiles, and the public
  request phase that opens the seam.
- **Plan 4** — stage 6 quants (expected to be the hard one: every profile
  re-passes the exact-token gate or is not shipped), stage 7 backends (the
  codec moves first; the generator only on bit-identical grid evidence), and
  stage 8 ship.

Also still open and now written into the family doc rather than carried in a
task list: the four golden cases that clear the 1e-4 margin screen while still
sitting under the 6.1e-04 logit-divergence bound (`omni-short-en`,
`omni-long-boundary`, `omni-rate-fast`, `omni-lang-none`) are where a quantized
profile or a second backend is likeliest to flip a token first.

### Gates

| gate | result |
|---|---|
| full sweep, `--require all`, 20 cases, `--margin-report` | **20/20 ok, 17/17 exact**, exit 0, 417.3 s |
| `ctest -R synthesize-omnivoice-replay-golden` (`--check` enforcing) | **Passed, 420.23 s**, exit 0; `-V` re-run 419.70 s, same verdict |
| — negative control: tolerance cell removed | refused, exit 1 |
| — negative control: `hidden_l27` tightened to 0.99999999 | breach reported, exit 1 |
| `ctest -R synthesize-golden-manifest-contract` | passed (`case_count` 20 == manifest 20) |
| `ctest -R synthesize-vits-python-unit` | passed (tolerance coverage: validators `{replay}` == measured `{replay}`) |
| `ctest -R synthesize-omnivoice-python-unit` | passed |
| `cmake --build build --target synthesize-check-unit` | 78/78 passed |
| `cmake --build build-sanitize --target synthesize-check-unit` (ASan/UBSan) | 78/78 passed |
| `ctest -L unit` | 78/78 passed |
| `scripts/ci/clang-format.sh --check-diff` | clean |
| package GGUF | `3ecaa5e2f6fbd735296ba1cd60680c90467be22d2140dc4f208fe80111ecb9e5`, unchanged |
| public seam | still `synthesis.not_implemented` for omnivoice |

## 2026-07-31 — Golden suite bumped to revision 2, correcting a missed bump

Final review of the whole branch caught a metadata gap: `docs/port-validation.md`
defines `suite_version` as a monotonically increasing variant-suite revision,
incremented by any manifest change that alters what validation proves, and the
"Ruling implemented" commit above did exactly that — it enumerated a second
admissible grid for `omni-fast-mode` and re-picked `omni-clone-zh`'s text and
oracle grid — while leaving `suite_version` at 1. The qwen3-tts precedent this
project already has is the manifest that took two new cases (eighteen to
twenty) and went from `suite_version` 1 to 2 for that alone; two revisions to
existing cases is the same kind of change, only smaller in scope, and it was
missed here. `tests/golden/omnivoice/omnivoice-0-6b.manifest.json` now carries
`suite_version: 2`. `tests/tolerances/omnivoice.json` carries its own
`suite_version` field, and its thresholds were measured after the ruling
commit landed, not before, so they were never stale — only mislabeled at 1; it
moves to 2 for that reason, not because any figure in it was re-measured.
Neither `scripts/validate-omnivoice-replay.py` nor either Python unit gate
compares the two files' `suite_version` values or asserts a literal one, so
this is a metadata correction with no behavior change.
