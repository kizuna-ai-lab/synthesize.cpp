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

## 2026-08-02 — Plan 3 closeout: the public sampled path, cloning, and untrusted-bytes hardening

Plan 3 (Tasks 1–17, branch `omnivoice-plan-3`) closes the family's public
sampled path, Reference Audio and Description Text Voice Profiles, their
Serialized Profile GGUF round-trip, and the CLI/Python-wheel Adapters. This
entry is the evidence record; `docs/porting/families/omnivoice.md` carries
the same facts as contract prose, and
`docs/superpowers/plans/2026-08-02-omnivoice-plan-4-carryover.md` carries
what remains open. Seventeen tasks landed as 21 commits total (base `0b794a0`
through `8e23f8c`); each subsection below cites the commits it reports on.

### Bookkeeping fixes carried in ahead of the family's own work (Tasks 1–2)

Two pre-existing gaps, unrelated to this family's own model semantics, were
closed first because later tasks depend on the tooling they touch. Task 1
(`a7e9169`, fix `75247ba`) corrected `tests/tolerances/qwen3-tts.json`'s
`suite_version` (1→2, no threshold changed) and its own porting-log
attribution, which had wrongly credited a case-count change to an
omnivoice-specific ruling instead of commit `0b80e534` (which actually added
`qwen3-longer-english`/`qwen3-longer-chinese`, 18→20 cases). Task 2
(`6395a1f`) replaced two independent, uncross-checked pinned-input lists —
`convert-omnivoice.py`'s own six-entry list and
`dump_reference_omnivoice_pytorch.py`'s private inference from `/resolve/`
locators — with one shared table, `scripts/omnivoice_pinned_inputs.py`; this
closed a real coverage gap (the dumper's old five-item `required` set
omitted `audio_tokenizer/LICENSE`) and is the digest byte-identical before
and after (`build/goldens/omnivoice/omni-short-en/codes/grid.i32` sha256
`60473d82…` both times). Both trees 78/78 and 79/79 respectively at each
task's own close; format 0.

### The sampler: draw order, log_prob, and the seed contract (Tasks 3–4)

Task 3 (`47a32e9`, fix `a49206f`) implemented `choose_token_sampled` and
`gumbel_perturb`. Review, run empirically against the installed upstream
package rather than by inspection alone, found `log_prob` computed as
`guided[chosen]` instead of upstream's own `confidence_scores =
log_probs.max(dim=-1)[0]` (`omnivoice.py:1449`) — the two diverge on **9,026
of 20,000 seeds** at `keep=2` on a fixture with a deliberate 0.1 guided-logit
gap between the top two classes (theoretical P(argmax wins) ≈ 52.5%, an
empirical ~45/55 split observed). The fix computes `log_prob` as the max
over the FULL unfiltered `guided` array, matching upstream exactly, and a
400-seed regression (`tests/omnivoice_sampler_test.cpp`) now asserts it
against the plain-greedy value on every seed, proven fail-before/pass-after
via a temporary revert. `gumbel_perturb` transcribes upstream's
`_gumbel_sample` (`omnivoice.py:1632-1636`) as `scaled = logit /
temperature; noise = -log(-log(uniform + 1e-10) + 1e-10); result = scaled +
noise`, entirely in `float` — no `double` intermediate — because a
different rounding could change which of several perturbed candidates an
argmax picks. Both trees 79/79 after the fix; format 0.

Task 4 (`fcae589`, fix `a6392ec`) wired `choose_token_sampled` into the
decode loop's per-candidate scan and added margin-report instrumentation.
Golden gate: **17/17 exact grids, `synthesize-omnivoice-replay-golden`
Passed 434.53 s**. Review found the seed-wiring untested — the synthetic
fixture's constant weights make the token grid seed-invariant by
construction (every class ties, so `choose_token`'s strict `>` tie-break
always lands on class 0 regardless of draw order), so a hardcoded seed
inside `run_synthesis` and a correctly-wired one would look identical on it.
Fix round 1 added a `varied_weights` synthetic package option (LCG-filled)
and `check_sampled_seed_actually_drives_the_grid`: fresh `SynthesisRequest`
per call (never reused), seed 7 reproduces itself (`7 ≡ 7`) and seed 7
differs from seed 8 (`7 ≠ 8`) on the varied-weight package's real token
grid. Mutation-proven at `model.cpp:518` (the stream-construction site):
hardcoding the seed to `42` there fails the new test, reverting passes it.
Both trees 79/79.

The draw-order contract this stream-construction site now documents in code
— codebook-major/frame-minor candidate scan; class draws (inside
`choose_token_sampled`, ascending class-id order over top-k survivors)
before that candidate's own position draw; a zero-budget step draws
nothing — is the same contract `docs/porting/families/omnivoice.md`'s new
"The sampler" section restates for the family record.

### First public synthesis of the family, and the public gate (Tasks 5–6)

Task 5 (`a57bff6`, fix `d670090`) opened the public seam for this family for
the first time (previously `synthesis.not_implemented` unconditionally).
Smoke evidence, `omnivoice-0-6b-F32.gguf`, text "This is the first real
public synthesis of the OmniVoice family.", 94,080 PCM samples (98 native
codec frames):

| run | seed | sha256 |
| --- | --- | --- |
| A | 7 | `0f4292aac1f7c6bfa8449a10a3e527ad40de99a677833625fe9d1d0dd43ea7e2` |
| B | 7 | `0f4292aac1f7c6bfa8449a10a3e527ad40de99a677833625fe9d1d0dd43ea7e2` |
| C | 8 | `7caeb17dbc1aa9671c822b6eca03c7a44990a7b10a71b337ea1efea80e2ae52c` |

Seed 7 reproduces itself exactly (A ≡ B); seed 8 differs (C ≠ A); peak
exactly 0.5 (no-reference volume branch). Review found `tests/omnivoice_
load_real.cpp` still asserted the removed `synthesis.not_implemented` stub
and had never actually been run under the `integration` label — a Critical
that would have shipped a red integration suite unnoticed. Fix round 1
routed whitespace-only text through the public seam (proven to reach the
FAMILY's own `SYNTH_ERR_INVALID_ARG`, not the core guard — genuinely-empty
text is rejected earlier, before family dispatch, with no diagnostic),
promoted `assemble_prompt_ids`'s text-end postcondition from an
NDEBUG-inert `assert` to a release-path check with an honest
crafted-mismatch unit case, and ran the omnivoice integration set to green:
`synthesize-omnivoice-load-real` 4.01–4.14 s,
`synthesize-omnivoice-replay-golden` 425.12–449.41 s. Both trees 79/79.

Task 6 (`040194e`, fix `4d072e3`) registered
`synthesize-omnivoice-public-request` — 7 checks (echo, same-seed identity,
cross-seed distinctness, random-seed concreteness, language echo, a
no-language arm, evidence completeness) over 7 runs. First pass: **Passed
66.43 s**. Review, run with fault injection (a missing/failing runner, a
stale PCM file), found check 4 (random-seed concreteness) did not exclude
`SYNTH_SEED_RANDOM`'s own sentinel value (`UINT64_MAX`) — a resolver that
returned the sentinel unchanged would still pass. Fix round 1 excluded it
with a one-line comment; re-run **Passed 65.78 s**, `tests/tolerances/
omnivoice.json`'s public stage recorded `"checks": 7, "all_passed": true`.

### Adapters: CLI and Python wheel (Task 7)

Task 7 (`d713ec7`) registered `synthesize-omnivoice-cli` (**Passed 30.64–
30.84 s**, package-default voice, `--phonemes` as the unsupported-input
arm — the one kind this package's `input_flags` lacks) and the Python wheel
family smoke (24 kHz, 960 samples/frame, seed 7/8 reproducibility,
`ctest -R '^synthesize-python-api-wheel-test$'` **Passed 29.02–29.36 s**),
closing the stage-7.5 gap qwen3-tts left open (neither harness was ever
registered for that family). `synthesize-omnivoice-public-cleanup`
(`synth_add_cleanup_test(omnivoice … 720000 "text:Hi." "" "en")`, 720,000
PCM samples = 750 codec frames, the package's own frame cap) **Passed
72.99–73.21 s**. Fixing omnivoice's registration surfaced a genuine
cross-family bug: `synth_add_cleanup_test`'s unquoted `${ARGN}` silently
dropped omnivoice's empty voice argument, shifting `"en"` into the voice
slot (`SYNTH_ERR_UNSUPPORTED_VOICE`). The fix (quoted, positional `foreach`
over `ARGN`) was verified behaviorally identical for VITS, Kokoro and
qwen3-tts by reproduction (`cmake -P`, before/after `CTestTestfile`
comparison) before landing project-wide. Both trees 79/79; format 0 (15
files checked).

### The Audio Normalizer: vendoring libsamplerate (Task 8, ADR 0009)

Task 8 (`68a0b7e` vendoring + `3eccf4c` module, fix `90d115b`) vendored
libsamplerate **0.2.2** (`third_party/libsamplerate/`), tarball sha256
`3258da280511d24b49d6b08615bbe824d0cacc9842b0e4caf11c52cf2b043893`,
re-fetched and byte-compared by the reviewer independently. Kept exactly
`src/*.c`, `src/*.h`, `include/samplerate.h`, `COPYING` (10 files); the
`clang-format` `EXCLUDE_RE` widened to `^ggml/|^third_party/` accordingly.
Review found the `src_simple` drain contract unchecked in code
(`input_frames_used` never compared against the frame count actually
passed) — a silent-crop hazard an empirical sweep does not guarantee against
production inputs. Fix round 1 added the drain check (`input_frames_used !=
frames → SYNTH_ERR_INTERNAL`), a symmetric `output_frames_gen` bounds guard,
an overflow test (`frames = UINT64_MAX`, asserts untouched output), and
three target-validation rejection tests (`target_rate=0`,
`target_channels∈{0,3}` — load-bearing because `convert_channels` only
implements the 2↔1 arms). Both trees 80/80 before and after; format 0 (18
files checked).

### Clone-encode oracle probes, and the Golden suite's suite_version 3 (Task 9)

Task 9 (`4f9486e`), ruling by jiangzhuo 2026-08-01: three new oracle probes —
`ref/pcm_16k.f32` ([224640] = ceil(336960 × 16000/24000), exact),
`ref/semantic_mean.f32` ([702, 768], pre-`[::2]`-downsample), and
`ref/fused_latent.f32` ([351, 1024], post-`fc`) — added to both clone-case
dumps. `tests/golden/omnivoice/omnivoice-0-6b.manifest.json` and
`tests/tolerances/omnivoice.json` both move `suite_version` 2→3; no case
text changed, and every one of the 16 pre-existing artifacts in each clone
case's dump is byte-identical before and after (verified by independent
sha256 recompute), as is the unrelated `omni-short-en` control case's
12-artifact set. Both trees 82/82; format 0.

### The resampler: transcription and a kernel-math correction (Task 10)

Task 10 (`7ef5728`, fix `a0814e7`) transcribed the 24→16 kHz sinc resampler
`torchaudio.functional.resample` uses at its defaults. The implementer's
own model — promote the float32 side to `double`, compute, round once back
to float32 — was proven WRONG by review, which compiled the shipped kernel
table standalone and diffed it against torchaudio's own real kernel tensor:
**38 of 46 tap coefficients mismatched.** The corrected model is that
PyTorch rounds the scalar constants to float32 FIRST and then computes
entirely in float32, with no double intermediate anywhere in the kernel
construction. Isolating one step of the construction (`t *= base_freq`)
confirmed the mechanism directly: round-scalar-first gave 0 mismatches (0.0
max diff) against torch's real tensor; promote-to-double gave 6 mismatches
(up to 4.77e-07). The fix declares the kernel's float32 constants
(`kBaseFreqF`, `kScaleF`, `kPiF`) once, rounded from their double
counterparts, and removes every `double(...)` promotion from the
kernel-construction expressions; the reconstructed kernel is then bit-exact
on all 46 tap coefficients.

Measured max_abs, before → after, by build config:

| fixture | Release (`build/`) | RelWithDebInfo (`build-sanitize/`) |
| --- | --- | --- |
| `dc96` | 1.19209e-07 → **0.0** | 0.0 |
| `impulse64` | 2.98023e-08 → **0.0** | 0.0 |
| `mixed48` | 5.96046e-08 → 5.96046e-08 | **0.0** |
| real signal (`ref/pcm_16k.f32`) | 1.78814e-07 → **1.19209e-07** | 0.0 |

`mixed48`'s and the real signal's residual figures in Release are
attributable to convolution accumulation ORDER, not the kernel table
itself — both are exactly 0.0 under RelWithDebInfo. **This is a
build-config-dependent noise floor, not a defect, and it matters for Plan
4's future tolerance work**: a gate this tight must account for it rather
than assume Release and sanitizer builds agree to the last bit. Gates
tightened accordingly: `dc96`/`impulse64` `== 0.0f` exactly, `mixed48` `<=
1e-7f`, the real-signal integration gate
(`synthesize-omnivoice-resampler-golden`) `<= 5e-7f`. `target_length`'s
`torch.ceil(torch.as_tensor(new_freq * length / orig_freq))` was confirmed
to downcast to float32 BEFORE the ceiling (`torch.as_tensor` on a bare
Python float takes torch's default dtype), transcribed as such; a sweep
over `length ∈ [1, 3,000,000]` found zero divergence from the naive
double-then-ceil ordering at this family's scale, so the distinction is
correctness-by-construction here rather than an observed difference. Both
trees 81/81 before and after; format 0 (21 files checked).

### HuBERT semantic branch (Task 11)

Task 11 (`b619ce6`, format fix `9b5dffe`) added the semantic branch
(resampled 16 kHz → HuBERT → `[::2]` downsample → `SemanticEncoder`).
Miniature fixture: 5.96e-07 max_abs (tolerance tightened to 3e-6, 5× the
measured figure). Real-scale `ref.semantic_mean` parity, both clone cases
identical: **max_abs 6.09234e-05, cosine 0.99999993**. Golden gate: **17/17
grids exact, `synthesize-omnivoice-replay-golden` Passed 423.90 s**, with
the new encode channel report-only (not yet gating — confirmed by reading
the validator's own code, not merely by the exit code). Review confirmed
GELU-erf (not the tanh approximation) both by config (`hidden_act:
"gelu"` resolving to `nn.functional.gelu`) and by an empirical fixture-
sensitivity check (227× against the wrong variant), traced the 13-hidden-
state capture point, and confirmed the `SamePad` trim and group-norm
variant. One Important: the task's own format-check claim was false — 572
violations across the three new files at HEAD, because the original
`--check-diff` run happened while those files were still untracked (`git
diff --name-only` is blind to untracked paths — the known git-selection
trap, in its untracked form this time rather than its unstaged form). Fix
round: format-only, no numeric change; re-verified identical (5.96e-07,
unchanged) after formatting. Both trees 82/82 (up from 81); format 0.

### DAC acoustic encoder and reference fusion (Task 12)

Task 12 (`f934612`) added the acoustic branch and the semantic/acoustic
fusion. `codec_conv1d` gained a `stride` parameter (default 1, three new
rejection tests for `stride ∈ {0, -1}` and one acceptance for `stride=2`).
Two things the checkpoint's own config would have gotten wrong were
resolved empirically rather than assumed: the downsampling/upsampling ratio
order is `[8, 5, 4, 2, 3]` in BOTH directions — `DacConfig.__post_init__`'s
own reversal logic (`upsampling_ratios = downsampling_ratios[::-1]`) is
overridden by the raw config on this checkpoint, confirmed by instantiating
the real checkpoint's config and reading `.encoder.block[i].conv1.stride`
directly rather than trusting the class's default logic; and the 480-sample
(`hop_length // 2`) pad upstream applies conditionally never fires for
hop-aligned input, confirmed both symbolically (the closed-form frame count
through ratios `[8,5,4,2,3]` reduces to exactly `T/960` for any `T` a
multiple of 960) and against the oracle's own numbers (336,960 samples =
351 × 960 exactly, matching `ref/fused_latent.f32`'s 351 rows unpadded).
Miniature fixtures: acoustic 8.94e-08, fused 5.96e-08 max_abs (tolerance
5e-7, 5× measured). Real-scale `ref.fused_latent` parity, both clone cases
identical: **max_abs 9.32217e-05, cosine 1.00000011** (fractionally above
1.0, the same float32 estimator artifact the deep generator probes already
show — not a defect). Golden gate: **17/17 grids exact, Passed 426.81 s**.
Independent review reproduction (ratio proof, fixture regeneration, binary
artifact diff) matched the implementer's figures exactly; zero fix rounds.
Both trees 82/82; format 0.

### RVQ encode and the exact-token gate (Task 13)

Task 13 (`dc42440`, fix `fc76b49`) added host-side RVQ encode.
**Exact-token gate passed on the first run: both clone cases' encoded
tokens matched `ref/tokens.i32` byte for byte — 8 codebooks × 351 frames =
2,808 tokens per case, 2,808/2,808 exact, 2/2 cases.** The RVQ encode's own
margin instrumentation (diagnostic only, never gated) measured a narrowest
best-vs-second-best nearest-neighbor distance of **0.00239563** over the
real reference clip. Golden gate: **Passed 433.51 s**. Review, independently
reproducing the validator, sha256-checking four token files, and
regenerating fixtures byte-identically, found `ref_rms` measured in the
WRONG order: on the already hop-clipped buffer (0.12305419892072678),
rather than upstream's own order — measure on the FULL, un-clipped buffer
FIRST (`omnivoice.py:774`), THEN boost (`:775-776`), THEN hop-clip
(`:816-818`). This is a systematic scale deviation on every boosted
(quiet) reference, unobservable on the committed goldens only because the
one pinned reference clip's true RMS (0.1229146420955658) sits above the
0.1 boost threshold either way. Fix round 1 reordered to measure-on-full-
buffer → boost-full-buffer → hop-clip, cited both omnivoice.py line ranges
in the code, and added six unit cases
(`tests/omnivoice_reference_encoder_test.cpp`) including a straddle-0.1
regression proven fail-before/pass-after via temporary revert (buggy order:
`ref_rms` 0.0500000007; correct order: 0.449444115, matching the expected
full-buffer value). Tokens remained 2,808/2,808 exact and the gap unchanged
after the fix; golden gate re-run at 682.70 s (elevated by concurrent
unit-gate builds on the same host — a concurrent wheel-test timeout in the
same window was contention-only, reconfirmed 82/82 in isolation). Both
trees 82/82; format 0.

Tolerance derivations committed in `tests/tolerances/omnivoice.json` for the
three Task 9 probes, now gated (5× the observed figure, per this file's
standing rule):

| probe | observed max_abs | committed max_abs | observed min_cosine | committed min_cosine |
| --- | --- | --- | --- | --- |
| `ref.pcm_16k` | 1.19209e-07 | 6e-07 | 0.9999999444538425 | 0.999999 |
| `ref.semantic_mean` | 6.09234e-05 | 4e-04 | 0.9999999293211036 | 0.999999 |
| `ref.fused_latent` | 9.32217e-05 | 5e-04 | 1.0 (capped) | 0.999999 |

### Reference Audio cloning, end to end (Task 14)

Task 14 (`fdb0b41` + `7613283` + `6c792c7` + `a41784d`) wired the whole
encode chain into `Model::synthesize` and `voice-profile.cpp`'s
`create_from_reference`. `Model::encode_reference`'s output token grid
equals `ref/tokens.i32` exactly; non-vacuity was proven three ways —
matching against the real file passes, against a deliberately wrong file
fails, and against a nonexistent path takes the documented sentinel-skip
path (still exit 0, but a different, named code path). Measured
`ref_rms`: 0.1229146…, matching the expected 0.1229146420955658 to within
1e-4. Silent-reference rejection (jiangzhuo's ruling, 2026-08-01):
`create_from_reference` refuses a `ref_rms == 0.0f` reference with
`SYNTH_ERR_INVALID_ARG` and diagnostic `voice_profile.reference_silent`,
proven against a real all-zero-sample fixture at the package's own minimum
clip length.

Public clone run, seed 0:

| run | frames | sha256 |
| --- | --- | --- |
| clone, run A | 67,200 | `6d1d61f414145542c88eb8131d79e030b8dfe53c4d70c71389d5a2dad30f702c` |
| clone, run B | 67,200 | `6d1d61f414145542c88eb8131d79e030b8dfe53c4d70c71389d5a2dad30f702c` |
| no profile | 48,000 | `1fa36c04b3da2647f33948a094167996fb9ff2d5d890a2b0c7e7b09ffbe38db1` |

Same-seed-with-profile reproduces exactly (A ≡ B); the profile run differs
from the no-profile run in both frame count and digest, confirming the
reference genuinely conditions output rather than being silently ignored.
`scripts/validate-omnivoice-public.py`'s checks grew 7→10 (all 10 passed);
integration total **36/36 passed** (1,040.89 s), with
`synthesize-omnivoice-replay-golden` at 429.72 s and
`synthesize-omnivoice-public-request` at 160.62 s named individually. Both
trees 83/83; format 0.

### Description Text voice design (Task 15)

Task 15 (`fc657e6`) implemented `synth_voice_profile_create_from_description`
via `resolve_instruct`, transcribing the closed attribute/accent/dialect
vocabulary from `omnivoice/utils/voice_design.py:31-97` (48 members: 23 EN —
13 dict entries across gender/age/pitch/style plus 10 accents — and 25 ZH —
13 dict entries plus 12 dialects) and the validation/unification rules from
`omnivoice/models/omnivoice.py:1492-1621`. Confirmed at source: an
unsupported item raises with no free-text fallback anywhere in the
function, ported as `SYNTH_ERR_INVALID_ARG` /
`voice_profile.instruct_unknown_item` with no `difflib`-style suggestion.
Two flagged, reviewer-adjudicated divergences: (1) a null
`description_language` resolves to the fixed default `"en"`, never
detected from the description's own text, per `docs/c-interface.md:568`'s
explicit prohibition — endorsed as the doc governing over a looser brief
paraphrase; (2) upstream's per-call `use_zh` baseline is computed from the
TARGET TEXT being synthesized (`omnivoice.py:1068`), which does not exist at
Voice Profile creation time, so this port substitutes the resolved
`description_language` instead — accepted because deferring unification
would break profile immutability. `synthesize-omnivoice-public-request`
grew 10→13 checks (voice-design synthesis succeeds, same-seed
reproducibility with a design profile, digest differs from a same-seed
no-profile run), **Passed 190.98 s, 13/13**; `synthesize-omnivoice-
profile-test` **Passed 22.91 s**. Both trees 84/84; format 0.

### Serialized Profiles, the GGUF round trip, and untrusted-bytes hardening (Task 16)

Task 16 (`e88ab6f`, fix rounds `dab8766` and `8e23f8c`) implemented the v1
Serialized Profile GGUF envelope end to end. Round-trip PCM is byte-
identical for both kinds (ClonePrompt and DesignInstruct) against the real
package at seed 7; an 8-arm tamper matrix (payload byte flip, wrong
compatibility id, wrong schema, wrong kind, truncation, out-of-range token,
oversized token count, cross-family) rejects every arm with the correct
status.

Review found a Critical: untrusted bytes reaching
`synth_voice_profile_load_from_memory` can abort the whole host process.
A buffer declaring `general.alignment` with the wrong type reaches
`GGML_ASSERT`s inside `gguf_init_from_reader`
(`ggml/src/gguf.cpp:194,610,1102`) BEFORE this project's own loader
validation runs — `ggml` is a submodule and cannot carry a local patch, so
the fix is a raw-byte pre-scan in this project's own loader. Fix round 1
added `prescan_buffer` as a BLACKLIST (special-case `general.alignment`,
generous structural ceilings); proven to turn the SIGABRT into
`SYNTH_ERR_INVALID_ARG`. Re-review, fuzzing the SAME harness (ASan,
8,127 iterations across three campaigns) found a SECOND, different crash
the blacklist could not have anticipated: a 40-byte buffer with a
zero-length key sails past the length ceiling (zero is not greater than
it) and an empty key is never `"general.alignment"`, reaching a DIFFERENT
`GGML_ASSERT(!key.empty())` inside all four of `gguf_kv`'s value-shape
constructors (`ggml/src/gguf.cpp:143,151,161,167`). **Lesson: blacklisting
another library's internal asserts can never be proven complete — round 1
closed one instance, and the fuzzer found the next one in the same
review session.** Fix round 2 replaced the blacklist with POSITIVE
validation against exactly the format this project's own writer ever
produces: a closed 12-key table (`kPrescanKnownKeys`) with exact
type/array-ness/count per key, duplicate-key rejection, an exact `n_kv`
(9 or 11, not a ceiling), and an exact tensor-section shape. Both crash
instances proven fixed with before/after evidence (disabled guard →
SIGABRT/exit 134 for both; whitelist alone → `SYNTH_ERR_INVALID_ARG`, no
ggml-side change). Standing evidence: the reviewer's own fuzz harness,
unmodified, re-run against the whitelist fix — **8,127 iterations across
three campaigns (single-byte mutation, random small buffers, boundary/
extreme cases), zero crashes, zero sanitizer reports, zero hangs,
reproduced twice.** Round-trip integration test **Passed 94.58 s**. Both
trees 86/86 (up from 84); format 0.

### Gates (Plan 3, cumulative at close)

| gate | result |
| --- | --- |
| `cmake --build build --target synthesize-check-unit` | 86/86 passed |
| `cmake --build build-sanitize --target synthesize-check-unit` (ASan/UBSan) | 86/86 passed |
| `synthesize-omnivoice-replay-golden` (17/17 grids + RVQ 2/2 exact) | Passed, most recent isolated run 433.51 s |
| `synthesize-omnivoice-resampler-golden` | Passed (`<= 5e-7f` real-signal gate) |
| `synthesize-omnivoice-public-request` | Passed 190.98 s, 13/13 checks |
| `synthesize-omnivoice-load-real` | Passed, ~4 s |
| `synthesize-omnivoice-profile-test` | Passed, 94.58 s (Serialized Profile round trip) |
| `synthesize-omnivoice-cli` | Passed 30.64–30.84 s |
| `synthesize-omnivoice-public-cleanup` | Passed 72.99–73.21 s |
| `synthesize-python-api-wheel-test` (omnivoice family smoke) | Passed 29.02–29.36 s |
| `scripts/ci/clang-format.sh --check-diff` | clean (exit 0) at every task's own close |
| untrusted-bytes fuzz (`repro_empty_key`-class harness) | 8,127 iterations, 0 crashes, 0 sanitizer reports, 0 hangs |

What Plan 3 does NOT close is recorded in
`docs/superpowers/plans/2026-08-02-omnivoice-plan-4-carryover.md`: the
remainder of the Port Validation Suite, Quantization Profiles, Execution
Backends, and ship.

## 2026-08-03 — `max_output_frames` unit fix: a 960x package-cap error, and a re-cut

PR #6's Codex review (kizuna-ai-lab/synthesize.cpp, branch `omnivoice-plan-3`)
found that `package_contract.max_output_frames` in
`tests/golden/omnivoice/omnivoice-0-6b.manifest.json` — and therefore in the
shipped `omnivoice-0-6b-F32.gguf` — carried **750**, this port's own
codec-frame ceiling (30 s at the codec's 25 Hz frame rate). `docs/c-interface.md`
documents the field as native PCM frames, matching every sibling package
(VITS 1,323,000; Kokoro 1,440,000; qwen3-tts 15,728,640). A caller honoring
the documented contract read 750 as **31 milliseconds**, and
`synth_model_get_capabilities` reported exactly that wrong number, because
`src/synthesize.cpp`'s `shared_info` passes `hparams.max_output_frames`
straight into the public capability struct with no conversion. Internal
synthesis only worked by coincidence: `src/arch/omnivoice/model.cpp` compared
its own codec-frame canvas estimate directly against the same
`hparams.max_output_frames` field, so as long as the field held a codec-frame
number the comparison was internally consistent — and would have broken (by
under-enforcing, not over-enforcing) the moment it held a real PCM value,
which is exactly what fixing only the manifest and nothing else would have
done. The reviewer's own suggested fix (divide the package cap by hop before
comparing, leaving the manifest at 750) was rejected for the same reason
stated in the brief that drove this fix: `750 / 960 = 0`, which refuses every
synthesis. The package value was what was wrong, not the comparison, or
rather: both were wrong, in a way that cancelled out.

**The corrected value is 720000** — `750 * hop_length (960)`, still 30 s at
24 kHz, now expressed in the field's documented unit. Six places needed the
unit made explicit, corrected, or both, beyond the manifest itself:

- `scripts/convert-omnivoice.py` gained `validate_output_frame_ceiling`: a
  package_contract whose `max_output_frames` is under one second of native
  PCM (< `SAMPLE_RATE`), or that does not land on a whole codec-frame
  boundary (`% HOP_LENGTH != 0`), now stops the conversion with a message
  that names the unit, so this class of error fails at conversion time
  rather than thirty layers away at a synthesis call. Five new unit tests in
  `tests/python/test_convert_omnivoice.py`'s `OutputFrameCeilingTests` pin
  it, including the literal defect (`max_output_frames: 750` refused,
  message contains `"750"` and `"PCM"`) and a check that the real committed
  manifest passes.
- `src/arch/omnivoice/model.cpp` — both places that compare a codec-frame
  quantity against `hparams.max_output_frames` (`run_synthesis`'s own ceiling
  check, and `Model::synthesize`'s `effective_limit`) now divide the package
  field by `hparams.codec.hop_length` first, with a comment at each site
  naming both units and citing PR #6.
- `src/synthesize.cpp` — the public dispatch's post-synthesis check compared
  `synthesis.frame_count` (codec frames) against `prepared.effective_frame_limit`
  (native PCM frames per docs/c-interface.md) directly; with the package cap
  now correctly PCM, that comparison would never trigger again (silent
  under-enforcement, the mirror image of the manifest's original defect). It
  now multiplies `synthesis.frame_count` by `samples_per_frame` first
  (overflow-guarded, matching the VITS branch's own pattern), and the stale
  comment ("The limit is in native frames, and this family's frame is the
  codec's hop rather than one sample") — which documented the very
  convention that was wrong — is replaced.
- `src/arch/omnivoice/omnivoice.h`'s `PublicSynthesisParams::max_output_frames`
  carried the comment `// native frames; 0 = package cap`, which is actually
  codec frames (the request-side conversion `src/synthesize.cpp` performs
  before calling in); reworded to say so explicitly, since the ambiguity
  between "this family's native frame" and "the contract's native PCM frame"
  is the exact shape of the bug.
- `tests/omnivoice_metadata_test.cpp`'s `valid_metadata()` fixture (a
  key-for-key transcription of the real package) and its `run_valid_package`
  assertion both moved 750 → 720000.
- `tests/omnivoice_synthetic_package.h`'s small-layout fixture now declares
  `16 * h.codec.hop_length` (96) rather than a bare `16`; the OUTPUT_LIMIT arm
  in `tests/omnivoice_decode_loop_test.cpp` (`kMaxFrames + 1` against the
  small package) is now commented to say explicitly that it exercises the
  PCM→codec conversion inside `run_synthesis`, not merely a same-unit
  ceiling — a dropped conversion there would have made the request pass
  instead of refuse, silently.
- `scripts/dump_reference_omnivoice_pytorch.py`'s `load_manifest` had an
  independent, latent instance of the identical defect: its
  `audio_chunk_threshold` gate compared a codec-frame quantity
  (`audio_chunk_threshold * FRAME_RATE_HZ`) against `contract["max_output_frames"]`
  raw. Not exercised by any committed test against the real manifest, but it
  would have raised `ManifestError` unconditionally the moment the manifest's
  value became a real PCM number (750 codec frames worth of threshold is
  always less than 720000 raw). Fixed the same way: divide the contract value
  by `SAMPLES_PER_FRAME` before comparing.
- `tests/omnivoice_load_real.cpp` had no assertion on
  `capabilities.max_output_frames` at all — the exact field this bug
  corrupted was unchecked by the one integration test that already loads the
  real package through the public C ABI. Added `kMaxOutputFrames = 750ULL *
  kSamplesPerFrame` and `SYNTH_TEST_CHECK(capabilities.max_output_frames ==
  kMaxOutputFrames)`, so this specific regression now has a standing gate.
- `tests/CMakeLists.txt`'s `synth_add_cleanup_test(omnivoice ...)` call
  already passed `720000` as its request-level `frames` argument — its own
  comment had already reasoned in PCM-frame terms and derived 720000 from
  "750 codec frames * 960" even while the manifest carried the wrong raw
  value, so this line's behavior needed no change, only its comment's framing
  (the package's declared field is no longer 750 needing conversion; it is
  720000 directly, and 750 is now what you get by converting the other way).
- `docs/porting/families/omnivoice.md`'s contract section is corrected
  in place (720000 native PCM frames, 750 codec frames, with an inline dated
  note); `docs/superpowers/plans/2026-07-30-omnivoice-plan-2-carryover.md`
  and `docs/superpowers/plans/2026-07-31-omnivoice-plan-2-synthesis-core.md`
  — historical, closed-out planning records that cite the pre-fix sha256 —
  are left as the historical record with a bracketed `[superseded 2026-08-03
  ...]` note at each of their four occurrences, rather than rewritten, for
  the same reason this porting log is append-only: those documents correctly
  recorded what was true when Plan 2 closed.

### Re-cut, verified metadata-only

Re-ran the converter exactly as its own docstring and the Plan 2 record
above specify:

```
uv run --project scripts/envs/omnivoice --locked python \
  scripts/convert-omnivoice.py \
  --manifest tests/golden/omnivoice/omnivoice-0-6b.manifest.json \
  --weights-dir models/omnivoice-0-6b \
  --output models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf
```

| | |
| --- | --- |
| old sha256 | `3ecaa5e2f6fbd735296ba1cd60680c90467be22d2140dc4f208fe80111ecb9e5` |
| new sha256 | `f6d504ffaddcbf32f80f1f6c847f075bbd5d2c7b50fe95a194ceb635772f9fa3` |
| file size | 3,189,953,504 bytes, **unchanged** |
| tensor count | 798, unchanged (generator 312, codec 486) |

**Verified metadata-only, not merely asserted.** `cmp -l` between the old and
new file found exactly **3 differing bytes**, at offsets 2712–2714 — the
three low-order bytes of the `synthesize.capabilities.max_output_frames`
little-endian u64 field (750 = `EE 02 00 …` vs 720000 = `80 FC 0A …`; bytes 3–7
are both `00`). Every other byte in the 3.19 GB file, including the entire
tensor payload, is bit-identical. This is a stronger claim than a tensor-by-
tensor comparison would give: it is a proof over the whole file, not a
sampled or shape-level check.

### Gates, against the re-cut package

| gate | result |
| --- | --- |
| `cmake --build build --target synthesize-check-unit` | 86/86 passed |
| `cmake --build build-sanitize --target synthesize-check-unit` (ASan/UBSan) | 86/86 passed |
| `ctest --test-dir build -L unit` | 100% tests passed, 0 tests failed out of 86 |
| `ctest --test-dir build-sanitize -L unit` | 100% tests passed, 0 tests failed out of 86 |
| `ctest --test-dir build -L integration` | 100% tests passed, 0 tests failed out of 36; 1150.19 s real |
| `synthesize-omnivoice-replay-golden` | Passed, 433.21 s; **token grids exact: 17/17**, all probes within tolerance |
| `synthesize-omnivoice-load-real` | Passed, 4.09 s — `capabilities.max_output_frames == 720000` via `synth_model_get_capabilities`, the new standing assertion |
| `synthesize-omnivoice-public-cleanup` | Passed, 73.85 s |
| `synthesize-omnivoice-cli` | Passed, 31.14 s |
| `synthesize-omnivoice-profile-test` | Passed, 103.94 s |
| `synthesize-omnivoice-resampler-golden` | Passed |
| `synthesize-omnivoice-public-request` | Passed, 192.85 s |
| `synthesize-python-api-wheel-test` (omnivoice family smoke) | Passed — this test failed against the pre-re-cut package with `synth_synthesize_to_buffer failed: output limit reached (15)` once the C++ side's PCM→codec conversion was in place but the package still declared 750, i.e. a 0-codec-frame ceiling (`750 / 960 = 0`); passing again after the re-cut is itself evidence the two halves of this fix are matched |
| `scripts/ci/clang-format.sh --check-diff` | exit 0 |

The 17/17 exact token grids and the probe table (worst `generator.hidden_l27`
max_abs 0.0800781, min_cosine 0.99999979 — unchanged from every prior run of
this suite) are the proof the re-cut changed nothing observable about
synthesis: every number here is either identical to or within the same
committed tolerance as the pre-fix package produced, because the only bytes
that moved are three bytes of one metadata field the golden replay path never
reads.

## Plan 4 Task 3: the Q8_MIXED codec profile — produced, and BLOCKED

Produced `omnivoice-0-6b-Q8_MIXED.gguf` from the committed F32 package with
`build/bin/synthesize-quantize models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf
models/omnivoice-0-6b/omnivoice-0-6b-Q8_MIXED.gguf --quant Q8_MIXED`. Named
`Q8_MIXED` rather than a family-specific name: `tools/synthesize-quantize/
policy.cpp:14-24`'s profile table is shared across every family (VITS,
Kokoro, Qwen3-TTS, OmniVoice), and each family's own
`resolve_<family>_target_spec` is what gives "Q8_MIXED" its family-specific
meaning — there is no per-family name in that table to begin with, so
inventing one (`Q8_CODEC`) would be inconsistent with how the tool and the
other three families already work, not more precise.

| | |
| --- | --- |
| source sha256 | `f6d504ffaddcbf32f80f1f6c847f075bbd5d2c7b50fe95a194ceb635772f9fa3` |
| output sha256 | `b020933facda39c875f276934d3a5611c7a8836d4f243efc3fb1403d109b671e` |
| source size | 3,189,953,504 bytes (3042.2 MiB) |
| output size | 2,703,016,576 bytes (2577.8 MiB), −15.3% |
| tensors quantized | 158 of 798 (the rest are Sensitive/TransposeWeight, unconditionally F32) |

Per-group tensor-data bytes (excludes GGUF header/alignment padding, which is
why these do not sum to the whole-file sizes above):

| group | F32 bytes | Q8_MIXED bytes | tensors |
| --- | ---: | ---: | ---: |
| generator (`llm.*` + audio tables) | 2,450,309,120 | 2,450,309,120 | 312 |
| `codec.quantizer`/`fc`/`fc2` | 11,574,272 | 11,574,272 | 44 |
| `codec.acoustic_decoder` | 80,952,452 | 50,943,986 | 110 |
| `codec.acoustic_encoder` | 205,257,984 | 54,617,344 | 110 |
| `codec.semantic_model` | 377,483,264 | 114,511,872 | 209 |
| `codec.encoder_semantic` | 58,988,544 | 15,673,344 | 13 |
| **total** | 3,184,565,636 | 2,697,629,938 | 798 |

### The load path: one bug found and fixed

`catalog.cpp`'s per-tensor `expected_type()` asserted F32 for every tensor
under every profile (Plan 1's placeholder, never widened). Fixed to dispatch
on `QuantizationProfile` and, for `Q8Mixed`, on `classify_tensor`'s own
`QuantRole` — the same classifier `tools/synthesize-quantize/policy.cpp`
already dispatches from, so the offline packing decision and this load-time
expectation read one source of truth rather than two hand-maintained lists.
`find()` gained the same packed-shape acceptance `kokoro`'s catalog already
carries: a `MatrixWeight` conv kernel's logical 3-axis shape
`[kernel, in, out]` is checked against the flattened `[kernel * in, out]` row
once quantized. `weights.h`/`weights.cpp`/`model.cpp` gained the
`Q8Mixed`/`"Q8_MIXED"` enumerator and string round-trip alongside `F32`.

A second, independent bug surfaced only once a REAL package was replayed
end to end: `reference-encoder.cpp`'s `build_semantic_branch` cross-checks
each `feat_conv[index]` tensor's `ne[0]` against the package's declared
`conv_kernel[index]`, to catch a converter that wrote the wrong kernel width.
That check compared the raw `ne[0]` against the bare kernel width
unconditionally — correct only when the tensor is unpacked. Once Q8_MIXED
packs `feat_conv[1..6]` (every feat_conv but index 0, which reads the raw
single-channel waveform and stays Sensitive), `ne[0]` becomes
`kernel * in_channels` (1536 for `feat_conv[1]` on this checkpoint: kernel 3
× HuBERT's 512-wide `conv_dim[0]`), so the unfixed check refused every
legitimately packed package: `Model::encode_reference` returned
`SYNTH_ERR_INTERNAL` (20) for both clone cases, silently, with the CLI
runner still exiting 0. **This was found, not assumed**: the first replay
run reused the F32 run's `--work` directory (the runner's own convention,
documented in `validate-omnivoice-replay.py`'s `run_case`) and reported
`ref.semantic_mean`/`ref.fused_latent`/`ref.tokens` as bit-identical to the
F32 run's own committed figures — implausible once codec weights genuinely
differ between the two packages, which is what prompted checking the
runner's own stderr (`encode_reference -> 20`) rather than trusting a report
built from stale files the encode step never overwrote. Fixed the same way
as `catalog.cpp`'s own shape check: derive the expected `ne[0]` from whether
`ggml_is_quantized(hubert.feat_conv[index].weight->type)`.
`omnivoice_reference_encoder_test.cpp`'s new `check_packed_feat_conv1`
regresses it directly — building `feat_conv[1]` both F32 and packed Q8_0
from identical raw weights (mirroring `check_packed_encoder_semantic_conv`'s
existing pattern for `encoder_semantic.conv`, the tensor that pattern
already covered and exactly why this one did not get the same coverage) and
separately asserting a wrong-shaped packed tensor is refused. **This fix
ships regardless of the profile decision below**: without it, no profile
that packs `feat_conv[1..6]` could run the reference-encode path at all.

### THE GATE: 17/17 greedy exact, 0/2 clone exact — FAILED

Re-run with the fix in place, fresh `--work` directory (eliminating any
possibility of the staleness above recurring):

```
scripts/envs/omnivoice/.venv/bin/python3 scripts/validate-omnivoice-replay.py \
  --require all --margin-report --profile Q8_MIXED --backend CPU --stage replay \
  --model models/omnivoice-0-6b/omnivoice-0-6b-Q8_MIXED.gguf \
  --work /tmp/q8_fresh_work --report /tmp/q8_fresh_report.json
```

```
token grids exact: 17/17
ref.tokens exact: 0/2
narrowest RVQ encode gap: 0.0102997
```

17/17 greedy grids exact is structural, not lucky: the whole generator and
the RVQ are Sensitive/F32 under this profile, so the decode loop's logits are
bit-for-bit identical to the F32 package's own, and every one of this run's
own margins reproduces the F32 baseline to the measured digit (table below).
**Both clone cases' RVQ encode grids are NOT exact** — `omni-clone-en` and
`omni-clone-zh` (one reference clip, identical figures on both) each
mismatch at **1023 of 2808 positions (36.4%)**. First 20 of 1023 (both cases
identical):

| codebook | frame | got | want | gap |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 3 | 554 | 26 | 1.5105 |
| 0 | 7 | 841 | 554 | 2.8479 |
| 0 | 9 | 423 | 26 | 0.7493 |
| 0 | 23 | 719 | 364 | 1.7428 |
| 0 | 38 | 532 | 708 | 4.1221 |
| 0 | 54 | 554 | 835 | 3.6883 |
| 0 | 55 | 207 | 26 | 1.1656 |
| 0 | 57 | 555 | 846 | 17.7489 |
| 0 | 58 | 653 | 515 | 2.4443 |
| 0 | 59 | 87 | 619 | 0.6864 |
| 0 | 60 | 730 | 923 | 27.2859 |
| 0 | 62 | 923 | 555 | 11.7191 |
| 0 | 67 | 923 | 423 | 10.9085 |
| 0 | 70 | 569 | 459 | 7.2939 |
| 0 | 83 | 950 | 675 | 0.5335 |
| 0 | 87 | 389 | 644 | 19.3942 |
| 0 | 91 | 97 | 243 | 0.6600 |
| 0 | 102 | 389 | 872 | 4.1656 |
| 0 | 117 | 11 | 916 | 5.2944 |
| 0 | 125 | 423 | 26 | 0.9558 |

These are not knife-edge gaps: they run 0.53 to 27.29 (RVQ nearest-neighbor
distance units) against an F32 baseline whose narrowest best-vs-second-best
gap over the whole clip was 0.00239563. The codec probes that traverse the
quantized `semantic_model`/`acoustic_encoder` move by orders of magnitude in
the same direction: `ref.semantic_mean` max_abs 6.09e-05 → 0.342867,
`ref.fused_latent` max_abs 9.32e-05 → 3.73227, `audio.pcm` min_cosine
0.99999986 → 0.99775438. Every probe touching the quantized encoder path
degrades together, by a similar order of magnitude — the signature of
accumulated quantization noise through roughly 150 quantized HuBERT/DAC-
encoder weights, not a second isolated bug (a due-diligence sweep of every
other `->ne[0]` shape comparison in `reference-encoder.cpp`/`codec.cpp`/
`generator.cpp` found no second instance of the packed-row class of bug the
fix above closes).

### Margin protocol (carry-over item 8): every greedy margin unchanged

| case | F32 margin (baseline) | Q8_MIXED margin | kind |
| --- | ---: | ---: | --- |
| `omni-rate-slow` | 9.53674e-06 | 9.53674e-06 | selection |
| `omni-fast-mode` | 6.86646e-05 | 6.86646e-05 | argmax |
| `omni-short-en` | 1.15871e-04 | 1.15871e-04 | selection |
| `omni-long-boundary` | 2.04682e-04 | 2.04682e-04 | selection |
| `omni-rate-fast` | 2.44433e-04 | 2.44433e-04 | selection |
| `omni-lang-none` | 6.03199e-04 | 6.03199e-04 | selection |
| `omni-design-zh` | 6.40869e-04 | 6.40869e-04 | argmax |
| `omni-medium-en` | 7.17163e-04 | 7.17163e-04 | argmax |
| `omni-punctuation` | 8.39233e-04 | 8.39233e-04 | argmax |
| `omni-upstream-readme` | 1.14441e-03 | 1.14441e-03 | argmax |
| `omni-clone-en` | 1.19019e-03 | 1.19019e-03 | selection |
| `omni-clone-zh` | 1.28174e-03 | 1.28174e-03 | selection |
| `omni-digits` | 1.39546e-03 | 1.39546e-03 | argmax |
| `omni-design-en` | 1.41111e-03 | 1.41111e-03 | selection |
| `omni-nonverbal` | 1.89209e-03 | 1.89209e-03 | argmax |
| `omni-short-zh` | 2.46429e-03 | 2.46429e-03 | selection |
| `omni-short-ja` | 2.66457e-03 | 2.66457e-03 | selection |

Identical to the measured digit, not merely close, because the margin is a
property of the generator's own logits alone and the generator never
changes under this profile. None of the four in-band cases predicted as
likely first flips (`omni-short-en`, `omni-long-boundary`, `omni-rate-fast`,
`omni-lang-none`) or `omni-rate-slow` actually flipped — the prediction
assumed a generator-side logit perturbation, which this codec-only profile
cannot produce by construction. The real failure is in a subsystem the
greedy margin screen was never built to probe.

### Gates

| gate | result |
| --- | --- |
| `cmake --build build --target synthesize-check-unit` | 88/88 passed |
| `cmake --build build-sanitize --target synthesize-check-unit` (ASan/UBSan) | 87/87 passed |
| `ctest --test-dir build -R omnivoice` (full family set: unit + integration) | 23/24 passed — the one failure is `synthesize-omnivoice-replay-golden-q8-mixed`, correctly reporting the gate result above; every other omnivoice test, including the F32 `synthesize-omnivoice-replay-golden` (439.26 s, 17/17 exact), passed |
| `scripts/ci/clang-format.sh --check-diff` | exit 0 |

### Status: BLOCKED, not shipped

Per this family's own Quantization Profile rule (`docs/porting/families/
omnivoice.md`'s "Quantization Profile Shape"): a profile that fails the
exact-token gate is not shipped, and no perceptual claim substitutes for it.
No tolerance cell is committed for Q8_MIXED in `tests/tolerances/
omnivoice.json` — the exact-token check is unconditional and independent of
`--check`, so a tolerance cell cannot make the grid pass and would
misrepresent a blocked profile as measured-and-ready. The per-profile golden
gate (`synthesize-omnivoice-replay-golden-q8-mixed` in `tests/CMakeLists.txt`,
guarded on `SYNTH_OMNIVOICE_Q8_MIXED_TEST_MODEL`'s existence the way the F32
gate is guarded on its own) is registered and left in the tree specifically
because it correctly fails today: it is the mechanism that would confirm any
future narrower profile. The choice between narrowing the profile's scope
(e.g. excluding `codec.semantic_model`/`codec.acoustic_encoder`, which feed
only the clone-only encode path, while still quantizing
`codec.acoustic_decoder`, which the greedy/public synthesis path alone
exercises) and dropping the profile entirely is jiangzhuo's and the
controller's to make, not this task's.

**Superseded by the continuation below**: jiangzhuo ruled to measure F16
before deciding anything further, F16 was measured and also failed, and the
per-profile golden gate this paragraph describes has since been removed
(de-registered, not left red) — see "Plan 4 Task 3 continuation" for the
final disposition. Left as the historical record of the state at the time,
per this log's own append-only convention.

## Plan 4 Task 3 continuation: F16 measured, also BLOCKED — F32-only ships

jiangzhuo's ruling: measure F16 codec-only before concluding anything about
quantization for this family — a genuinely different measurement, not a
retry. The tool's profile table gives F16 `TensorLayout::Native`
(`tools/synthesize-quantize/policy.cpp:14-24`): it does not pack, and
`ggml_compute_forward_im2col` accepts an F16 destination natively, so F16
needs none of Task 2's packed-branch machinery — every MatrixWeight tensor
keeps its native shape and only halves its type, through the unmodified
existing builders. F16's per-weight relative error (≈2⁻¹⁰) is roughly an
order of magnitude below Q8_0's (≈1/127), which is the whole question for a
nearest-neighbour decision Q8_MIXED missed by a distance gap of 1.51.
**Q8_MIXED itself was not revisited and stays blocked and unshipped.**

### Package

```
build/bin/synthesize-quantize models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf \
  models/omnivoice-0-6b/omnivoice-0-6b-F16.gguf --quant F16
```

| | |
| --- | --- |
| source sha256 | `f6d504ffaddcbf32f80f1f6c847f075bbd5d2c7b50fe95a194ceb635772f9fa3` |
| output sha256 | `530b2b85d7d1950f52b6b4374c5fa820740214358443003bb4c2cd2770b955fa` |
| source size | 3,189,953,504 bytes |
| output size | 2,858,422,240 bytes (2726.0 MiB), −10.4% |
| tensors halved | 158 of 798 — dumped and confirmed identical to the 158 Q8_MIXED quantizes, since both share `classify_tensor` |

Tensor-data bytes (excludes GGUF header/padding): 3,184,565,636 →
2,853,034,948, a reduction of 331,530,688 bytes (316.2 MiB) — matching the
≈316 MiB naive estimate (158 tensors' F32 bytes halved) almost exactly at
the tensor-data level; the file-size delta (331,531,264 bytes) differs
fractionally because GGUF per-tensor alignment padding scales with tensor
count and boundary position, not with bytes saved per tensor.

Per-group tensor-data bytes:

| group | F32 bytes | F16 bytes | tensors |
| --- | ---: | ---: | ---: |
| generator (`llm.*` + audio tables) | 2,450,309,120 | 2,450,309,120 | 312 |
| `codec.quantizer`/`fc`/`fc2` | 11,574,272 | 11,574,272 | 44 |
| `codec.acoustic_decoder` | 80,952,452 | 60,521,156 | 110 |
| `codec.acoustic_encoder` | 205,257,984 | 102,694,144 | 110 |
| `codec.semantic_model` | 377,483,264 | 198,438,912 | 209 |
| `codec.encoder_semantic` | 58,988,544 | 29,497,344 | 13 |

### Load path: no third gap

`QuantizationProfile` gained `F16` alongside `F32`/`Q8Mixed`
(`weights.h`/`weights.cpp`/`model.cpp`); `catalog.cpp`'s `expected_type()`
gained the matching `classify_tensor`-dispatched case (MatrixWeight →
`GGML_TYPE_F16`, else F32). F16 never reaches the packed-shape branch in
`catalog.cpp`'s `find()` (gated on `Q8Mixed` specifically) or the packed arm
of `reference-encoder.cpp`'s `feat_conv` check (gated on
`ggml_is_quantized`, which F16 is not): a direct single-case run of
`Model::encode_reference` against the F16 package succeeded on the first
try, no repeat of Q8_MIXED's silent-failure history. New regression
coverage: `omnivoice_catalog_test.cpp`'s `check_f16_resolution`/
`check_f16_rejections`, `omnivoice_metadata_test.cpp`'s
`run_quantization_profile_acceptance` extended to cover `"F16"`.

### THE GATE: 17/17 greedy exact, 0/2 clone exact — FAILED, more narrowly

Same command as the Q8_MIXED measurement, `--profile F16`, fresh `--work`
directory:

```
token grids exact: 17/17
ref.tokens exact: 0/2
narrowest RVQ encode gap: 0.00306702
```

17/17 greedy exact for the identical structural reason (generator/RVQ stay
F32 under every profile). **Both clone cases mismatch at 103 of 2808
positions (3.7%)** — about a tenth of Q8_MIXED's 1023, consistent with F16's
roughly-10x-smaller per-weight error. First 20 of 103 (both cases
identical):

| codebook | frame | got | want | gap |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 57 | 555 | 846 | 0.2024 |
| 1 | 25 | 442 | 278 | 0.4197 |
| 1 | 128 | 77 | 503 | 0.2124 |
| 1 | 155 | 658 | 991 | 0.1527 |
| 2 | 4 | 518 | 597 | 0.6510 |
| 2 | 57 | 613 | 597 | 4.8593 |
| 2 | 128 | 11 | 52 | 1.0621 |
| 2 | 155 | 39 | 130 | 27.4781 |
| 2 | 264 | 618 | 109 | 0.9983 |
| 2 | 348 | 228 | 508 | 0.6170 |
| 3 | 6 | 485 | 626 | 1.7481 |
| 3 | 7 | 202 | 975 | 0.0298 |
| 3 | 25 | 821 | 670 | 24.8352 |
| 3 | 27 | 706 | 395 | 0.3370 |
| 3 | 49 | 414 | 165 | 0.2210 |
| 3 | 54 | 101 | 596 | 0.0313 |
| 3 | 57 | 113 | 884 | 14.2502 |
| 3 | 102 | 559 | 685 | 0.1507 |
| 3 | 105 | 305 | 930 | 0.1803 |
| 3 | 128 | 926 | 91 | 2.4834 |

A genuine mix: several gaps (0.0298–0.2210) sit in narrow-margin range,
consistent with ordinary F16 rounding tipping a close call; others
(27.4781, 24.8352, 14.2502, 4.8593) are not narrow by any reading — the same
signature of real accumulated error Q8_MIXED showed, smaller in magnitude
but not a single mechanism.

### `audio.pcm` cosine (the second measurement this task was for)

| profile | `audio.pcm` min_cosine | `ref.semantic_mean` max_abs | `ref.fused_latent` max_abs |
| --- | ---: | ---: | ---: |
| F32 (baseline) | 0.99999986 | 6.09e-05 | 9.32e-05 |
| F16 | 0.99999743 | 0.00795197 | 0.129286 |
| Q8_MIXED | 0.99775438 | 0.342867 | 3.73227 |

F16's decode side sits close to two more nines than Q8_MIXED's, and every
channel places F16 consistently between F32 and Q8_MIXED, roughly an order
of magnitude closer to F32 than Q8_MIXED — the predicted relationship,
confirmed, and still not close enough to pass the clone encode grid.

### Margins: unchanged from the Q8_MIXED table

Every greedy-case margin reproduces the F32 baseline exactly under F16 too,
for the identical reason (the generator never changes under a codec-only
profile) — the table already recorded above applies unchanged.

### Gates

| gate | result |
| --- | --- |
| `cmake --build build --target synthesize-check-unit` | 88/88 passed |
| `cmake --build build-sanitize --target synthesize-check-unit` (ASan/UBSan) | 87/87 passed |
| `ctest --test-dir build -R omnivoice` (full family set) | all passed — no per-profile golden gate is registered for either measured profile (see disposition below), so there is no expected-red test in this run |
| `scripts/ci/clang-format.sh --check-diff` | exit 0 |

### Final disposition: F32-only

Two codec-only Quantization Profiles were measured against the exact-token
gate; both failed the cloning path's RVQ encode grid, by different
mechanisms and magnitudes, neither a knife-edge margin call eligible for
dual admissibility. **This family ships F32-only.** No tolerance cell is
committed for either profile in `tests/tolerances/omnivoice.json`. The
`synthesize-omnivoice-replay-golden-q8-mixed` gate registered during the
first half of this task is **removed** (`tests/CMakeLists.txt`,
`CMakeLists.txt`'s `SYNTH_OMNIVOICE_Q8_MIXED_TEST_MODEL` cache variable) —
a permanently-red registered test is not an acceptable branch state, and no
equivalent gate was registered for F16 either. The measured evidence for
both profiles is preserved here and in `docs/porting/families/
omnivoice.md`'s "Quantization Profile Shape" section, not as a failing
gate. The two load-path bugs this task found and fixed (`catalog.cpp` never
widened past F32-only checking; `reference-encoder.cpp`'s `feat_conv`
kernel-width cross-check not accounting for a packed tensor) ship regardless
of the profile outcome, with their own regression tests. If quantization is
revisited for this family, narrowing scope to exclude
`codec.semantic_model`/`codec.acoustic_encoder` (feeding only the
clone-only encode path) while still quantizing `codec.acoustic_decoder`
(which the greedy/public synthesis path alone exercises) is the untried
option both measurements point toward — untried here because re-scoping the
profile to force a grid to pass is exactly what this task's gate discipline
prohibits doing unilaterally.

## 2026-08-07 — Plan 4 Task 11: the CUDA sweep, and claiming the backend

Measured on the DGX Spark/GB10 development host: driver 580.159.03, CUDA
13.3.73, native `sm_121a`, the standard `dev-dgx-spark` preset (GGML CUDA
unified-memory fallback off — see the UVM note below).

```
export SYNTH_CUDA_ROOT=/usr/local/cuda-13.3
export PATH="$SYNTH_CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$SYNTH_CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
CUDAToolkit_ROOT="$SYNTH_CUDA_ROOT" CUDACXX="$SYNTH_CUDA_ROOT/bin/nvcc" \
  cmake --preset dev-dgx-spark -DSYNTH_BUILD_INTEGRATION_TESTS=ON
cmake --build --preset dev-dgx-spark -j 20
```

### The gate: byte-exact tokens, complete placement

```
uv run --project scripts/envs/omnivoice --locked python \
  scripts/validate-omnivoice-replay.py \
  --manifest tests/golden/omnivoice/omnivoice-0-6b.manifest.json \
  --model models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf \
  --runner build/dev-dgx-spark/bin/synthesize-omnivoice-replay-real \
  --accelerate --check --profile F32 --backend CUDA --stage replay
```

```
token grids exact: 17/17
ref.tokens exact: 2/2
```

Aggregated placement across all twenty cases (summed from each case's own
`"placement": {"generator": [nodes, off_cpu], "codec": [nodes, off_cpu]}`):

```
generator: 0 of 880,032 nodes off the CPU
codec:     8,440 of 8,440 nodes off the CPU
```

Both halves hold on every one of the twenty cases individually, not only in
aggregate. No greedy grid and no cloning-path `ref.tokens` grid flipped.
Re-run later as the registered gate
(`synthesize-omnivoice-replay-golden-cuda`, `tests/CMakeLists.txt`): **passed,
869.83s.**

A reviewer of this task independently re-derived the same aggregate from the
raw report JSON, re-ran six of the twenty cases live and confirmed each one's
own placement (`codec: [422, 422]`, `generator: [N, 0]`), and mutated the
backend-claim flip back to `false` to confirm the positive assertion below
actually fails without it — the check is not a tautology.

### Waveform drift, and the tolerance cell

Only the waveform channel shows the codec's CUDA (TF32) arithmetic:
`audio.pcm`/`audio.pcm_freerun` worst cosine 0.9999963545175866 (deviation
≈3.65e-6, against the CPU-only file's ≈1.7e-7) and worst max_abs
0.007008261978626251 on a 0.5-peak signal. Every deep generator probe and
every reference-encode probe (`ref.pcm_16k`/`ref.semantic_mean`/
`ref.fused_latent`) landed within the same ~1e-7 noise floor the CPU-only
file already documents — several probes came back bit-identical to the CPU
file's own committed `observed_*` values, because that half of the graph
never left the CPU. Committed to `tests/tolerances/omnivoice.json`'s new
`profiles.F32.backends.CUDA.stages.replay` cell using this family's own
established rule (5× the measured deviation, floored/ceiled the same way the
CPU cell's own note derives its numbers): min_cosine 0.999981, max_abs 0.04
for the two waveform probes; every other probe's committed min_cosine is
copied unchanged from the CPU cell.

### Codec timing: two independent measurements agree

On `omni-long-boundary` (719 frames, the suite's longest case), same binary,
`--backend CUDA` vs. the CPU-only run:

| run | codec_seconds (CPU) | codec_seconds (CUDA) | speedup | wall (CPU → CUDA) |
| --- | ---: | ---: | ---: | --- |
| mine | 4.3069 | 0.4451 | 9.68× | 267.30 s → 258.48 s (−3.4%, RTF 9.294 → 8.987) |
| reviewer's independent re-run | 4.2063 | 0.4382 | 9.6× | (not separately reported) |

Two separately-run measurements landing within 2% of each other on the
codec's own timing is itself evidence the CUDA path is doing real work: a
mislabelled or silently-CPU-fallback run could not reproduce a ~9.6× gap
twice from two different invocations. The held generator is 98.9% of wall
time on this case, so despite the codec's own large speedup the end-to-end
effect is a small, bounded saving rather than a dramatic one — the inverse
of Kokoro/VITS's own CUDA rows in `docs/backends.md`'s per-family cost table
(there, holding a minority stage off an otherwise-GPU-primary graph is a
tax; here, moving a free minority stage off an otherwise-CPU-primary graph
is a saving). Added OmniVoice's row there with that explanation.

### Operational evidence: latency, RTF, and why peak memory has no second budget here

Through the public seam (`tests/omnivoice_public_real.c`, now carrying the
`[cpu|cuda]` backend positional qwen3-tts's driver already had — see below),
one seed-0 request of "Sampling follows the seed." (45,120 PCM frames) loads
in 2.0015 s / 2.0356 s and synthesizes in 15.2393 s / 15.3927 s, CPU vs.
CUDA — too small a request to separate the codec's share from run-to-run
noise, consistent with its 1.6% share on the longer case above.

**Peak memory is not a second, GPU-exclusive budget on this hardware, and
this is measured rather than assumed.** `synth_model_get_device()` — this
family's own public Interface, sourced from the CUDA runtime rather than
`nvidia-smi` per `docs/backends.md`'s CUDA Unified Memory Policy — reports
the CUDA device with `SYNTH_DEVICE_MEMORY_SHARED` set and `memory_total` =
130,594,721,792 bytes, bit-identical to what the CPU device entry reports as
system memory total. `nvidia-smi -q -d MEMORY` independently corroborates:
its `FB Memory Usage` block reports `Total`/`Reserved`/`Used`/`Free` all as
literally `N/A` for this device (the plain `nvidia-smi` summary table
separately shows `Not Supported` in its Memory-Usage column — a different
field of the same underlying absence, not a second string for the same
one). The research-only `dev-dgx-spark-uvm` preset was read and deliberately
**not** used for this measurement or the claim below — `docs/backends.md`'s
own UVM policy says its results do not qualify a model-family/backend
combination as Supported, and the `SYNTH_DEVICE_MEMORY_SHARED` flag above is
a hardware-topology fact independent of whether that preset's UVM fallback
is enabled (it is not, under the standard preset this sweep used).

So there is no second budget to report a peak against. What is real and
measured instead: `/usr/bin/time -v` on `omni-medium-en` (307 frames, same
binary) reports Maximum resident set size 4,071,432 kB on CPU and
4,071,436 kB with `--accelerate` — a 4 kB difference, i.e. no measurable host
RSS growth from moving the codec to CUDA, because RSS accounting does not
see the device-mapped allocation at all. That allocation is real and
bounded, just invisible to RSS: `nvidia-smi --query-compute-apps` (which
returns real per-process numbers here even though the aggregate query does
not) showed this process holding 254 MiB right after load — the CUDA
context plus the 84.24 MiB mirrored decode-path weights — rising to a
**transient** 1,167 MiB while the codec's own compute buffers were live on
the 719-frame case, then falling back once that decode finished. That figure
is not a dedicated-VRAM requirement: it is a momentary share of the same
128 GB pool host RSS already draws from, not a second, GPU-exclusive
allocation the way a discrete card's VRAM reading would be — see
`docs/porting/families/omnivoice.md`'s Execution Backends section for the
fuller argument against reading it that way.

### Repeated-run cleanup

`tests/public_cleanup_test.cpp`'s CUDA arm is generic across families
(Task 8's fix round 1 made it assert the family's actual claim either way),
so it went live for OmniVoice the moment the claim flipped — no test code
changed for this family.

```
build/dev-dgx-spark/bin/synthesize-public-cleanup-test \
  models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf 12 720000 "text:Hi." "" "en"
```

```
cpu: post-free resident floor 367268 -> 367268 KB over 12 cycles (+0 KB)
cuda: post-free resident floor 563632 -> 563632 KB over 12 cycles (+0 KB)
```

No leak on either backend.

### Making `--backend cuda` real, and the positive assertion

`scripts/validate-omnivoice-public.py --backend cuda` used to be metadata
only (Task 10): the flag landed in the report dict and never reached the
runner. Fixed by giving `tests/omnivoice_public_real.c` the `[cpu|cuda]`
positional `tests/qwen3_tts_public_real.c` already has, and having
`validate-omnivoice-public.py`'s `synthesize()` forward `arguments.backend`.
Verified the pre-flip refusal was live before touching anything else: with
`family_supports_explicit_backend` still `false`, a `--backend cuda` run
failed every case with `load -> 17` (`SYNTH_ERR_BACKEND`).

The repeated-run cleanup test tolerates a forgotten claim flip by
construction — a correct refusal and a silently-wrong claim are
observationally identical to it. Two independent checks close that gap,
both querying `synth_model_get_device()` directly rather than trusting
`SYNTH_OK`: `tests/omnivoice_backend_test.cpp`'s `check_explicit_cuda`
(synthetic package, unit-level, asserts `device.kind == "cuda"`) and
`scripts/validate-omnivoice-public.py`'s new check 14 (real package,
integration-level, same assertion after a real `synth_synthesize_to_buffer`
call). Both passed; a reviewer of this task ran the full public validator
independently (`--backend cuda`, 388s, all 14 checks including check 14)
and separately proved check 14 is load-bearing by reverting the claim flip
and confirming it fails without it.

Only after all of the above did `family_supports_explicit_backend
(ModelFamily::Omnivoice, SYNTH_BACKEND_CUDA)` flip to `true`
(`src/model-info.h`) — the replay runner calls `Model::load` directly and
bypasses the public seam this refusal lives behind, so the sweep measured
real placement on real hardware while the seam still said no throughout.

### Gates

| gate | result |
| --- | --- |
| `cmake --build build --target synthesize-check-unit` (CPU-only, no CUDA compiled in) | 90/90 passed |
| `cmake --build build-sanitize --target synthesize-check-unit` (ASan/UBSan) | 89/89 passed |
| `ctest --test-dir build-integration -L integration -R omnivoice` | 7/7 passed (`synthesize-omnivoice-cli`, `-public-cleanup`, `-load-real`, `-profile-test`, `-resampler-golden`, `-replay-golden`, `-public-request`) |
| `ctest --test-dir build/dev-dgx-spark -R '^synthesize-omnivoice-(replay-golden-cuda|public-request-cuda)$'` | 2/2 passed (869.83s, 389.57s) |
| `python3 -m unittest tests.python.test_tolerance_coverage` | 3/3 passed |
| `scripts/ci/clang-format.sh --check-diff` | exit 0 |

Neither VITS/Kokoro nor Qwen3-TTS had a permanently-registered CUDA gate for
their golden/public validators before this task (their own CUDA evidence
lived in prose and manual runs); `synthesize-omnivoice-replay-golden-cuda`
and `synthesize-omnivoice-public-request-cuda` are new for the whole
project, not only for this family.

### The claim

Every `docs/backends.md` Validation Gate this family/backend combination
owes is now satisfied: same cases as the CPU baseline (1), tensor agreement
within committed tolerances (2), finite correctly-shaped PCM (3), twenty
cases (4), proven placement with latency/RTF/memory (5), and clean
repeated-run cycles (6). `docs/porting/families/omnivoice.md`'s "Generator
on CUDA" Open Question is resolved: the generator does not move, by the
discrete-outputs rule, and was never a live candidate — only the codec's 152
decode-path tensors do. Publication and the support matrix are separate,
later questions.

## 2026-08-07 — Plan 4 Task 12: generator-on-CUDA, measured as an experiment — flips, does not ship

Task 11 closed the "Generator on CUDA" Open Question by *policy*: the
discrete-outputs rule holds the generator on the CPU regardless of backend,
so it was never a live candidate for a claim. This task runs the *empirical*
test the Open Question originally asked for anyway — what actually happens
to the committed token grids if the rule is deliberately broken and the
generator is forced onto the primary backend — because a policy argument and
a measurement are different kinds of evidence, and the family doc's own
standing rule ("claimed only if placement evidence proves the committed
token grids bit-identical to CPU") is about the latter.

### Method: one line, reverted, never shipped

`src/arch/omnivoice/model.cpp`'s `generator_branch_forward` calls
`GraphRun::run(logits_tensor, "omnivoice.generator", threads)` at its
`on_primary=false` default — the one call site every generator forward (step
0, and both CFG branches of every denoising step) goes through. The
experiment is flipping that one argument to `true`:

```diff
-    const synth_status_t status  = run.run(logits_tensor, "omnivoice.generator", threads);
+    const synth_status_t status = run.run(logits_tensor, "omnivoice.generator", threads, true);
```

`on_primary=true` routes the graph through `BackendPlan::create_scheduler()`
(primary CUDA + CPU fallback, `op_offload=true`) instead of
`create_cpu_scheduler()`. No weight-mirroring twin was built for this
experiment — the generator's weights stay exactly where `Model::load`
already puts them, in the one CPU-resident `weights_context`/`weights_buffer`
Task 9's comment names as never retargetable. `ggml_backend_sched`'s
`op_offload` path is what makes this work at all: it offloads a
CPU-resident-weight op onto the primary backend anyway, copying whatever
operand it needs at each split boundary. That is the same mechanism
`docs/backends.md`'s discrete-outputs section measured at "2,372 scheduler
splits... five times slower" for Kokoro's duration stage and rejected for
the *shipped* path on performance grounds — irrelevant here, because this
run is not shipping.

Built once, into the existing `dev-dgx-spark` CUDA tree, on top of it:

```
export SYNTH_CUDA_ROOT=/usr/local/cuda-13.3
export PATH="$SYNTH_CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$SYNTH_CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cmake --build build/dev-dgx-spark --target synthesize-omnivoice-replay-real -j 20
```

Kept off the shipped path in the plainest way available: the patch lives
only in one hand-edited working-tree line, is never registered behind a
build flag or CLI option, and is reverted with `git checkout --
src/arch/omnivoice/model.cpp` (confirmed via `git diff`/`git status` — clean)
before any of the verification gates below ran or this commit was made. No
committed source carries a reachable path to `on_primary=true` for the
generator.

### Baseline: reproduce the family doc's own margin table first

Before touching the source, a plain CPU `--margin-report` run (unmodified
binary, no rebuild) reproduces the family doc's margin table exactly and
gives the exact position each case's narrowest decision sits at — useful
below when checking whether a flip lands where the margin table would
predict:

```
uv run --project scripts/envs/omnivoice --locked python3 \
  scripts/validate-omnivoice-replay.py \
  --model models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf \
  --require grid --margin-report \
  --work build/goldens/omnivoice-replay-task12-cpu-baseline
```

```
omni-short-en:      min selection margin 0.000115871 at step 7,  codebook 0, frame 19
omni-lang-none:     min selection margin 0.000603199 at step 11, codebook 0, frame 13
omni-long-boundary: min selection margin 0.000204682 at step 11, codebook 0, frame 237
omni-rate-slow:     min selection margin 9.53674e-06 at step 28, codebook 4, frame 80
omni-rate-fast:     min selection margin 0.000244433 at step 10, codebook 0, frame 9
token grids exact: 17/17
```

Matches the family doc's `1.16e-04` / `6.03e-04` / `2.05e-04` / `9.5e-06` /
`2.44e-04` to the measured digit. ~8 minutes wall for all 20 cases
(`--require grid`, no waveform decode).

### The gate: 3/17 byte-exact, not 17/17 — STOP

```
uv run --project scripts/envs/omnivoice --locked python3 \
  scripts/validate-omnivoice-replay.py \
  --model models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf \
  --runner build/dev-dgx-spark/bin/synthesize-omnivoice-replay-real \
  --accelerate --profile F32 --backend CUDA --stage replay \
  --require grid --margin-report \
  --work build/goldens/omnivoice-replay-task12-generator-cuda
```

2:12.77 wall for all 20 cases — *faster* than the CPU baseline despite
`op_offload`'s per-forward weight copies, confirming the generator's own
compute genuinely moved rather than merely being nominally rescheduled.
Placement, aggregated over all 20 cases: **849,315 of 880,032 generator
nodes (96.51%) left the CPU** — the same 880,032-node total Task 11's own
sweep reported (confirming this is the identical graph, only the scheduler
changed) — so the experiment is not a token gesture at the flag level: the
generator's compute substantially and verifiably ran on the primary device.

```
token grids exact: 3/17
```

| case | elements | mismatches | % flipped | GPU margin (kind@step,cb,frame=value) | CPU margin |
| --- | ---: | ---: | ---: | --- | --- |
| omni-upstream-readme | 536 | 23 | 4.29% | selection@19,1,5=6.47e-04 | argmax@31,6,54=1.14e-03 |
| omni-short-en | 400 | 376 | 94.00% | selection@7,0,47=2.23e-04 | selection@7,0,19=1.16e-04 |
| omni-short-zh | 432 | 208 | 48.15% | selection@25,2,47=1.69e-03 | selection@29,5,36=2.46e-03 |
| omni-short-ja | 376 | 142 | 37.77% | selection@26,3,24=3.32e-04 | selection@25,2,33=2.66e-03 |
| omni-lang-none | 392 | **0** | 0.00% | selection@11,0,13=8.88e-04 | selection@11,0,13=6.03e-04 |
| omni-punctuation | 536 | 4 | 0.75% | selection@1,0,19=4.26e-03 | argmax@31,7,32=8.39e-04 |
| omni-digits | 1056 | 1038 | 98.30% | argmax@31,6,64=5.26e-04 | argmax@31,7,102=1.40e-03 |
| omni-nonverbal | 488 | 10 | 2.05% | argmax@31,6,57=2.85e-03 | argmax@31,7,28=1.89e-03 |
| omni-medium-en | 2456 | 1651 | 67.22% | selection@27,3,115=1.77e-04 | argmax@31,7,297=7.17e-04 |
| omni-long-boundary | 5752 | 4190 | 72.84% | selection@27,3,664=3.32e-04 | selection@11,0,237=2.05e-04 |
| omni-rate-slow | 800 | 410 | 51.25% | selection@28,4,79=5.15e-05 | selection@28,4,80=9.54e-06 |
| omni-rate-fast | 200 | 1 | 0.50% | selection@10,0,9=2.43e-04 | selection@10,0,9=2.44e-04 |
| omni-design-en | 400 | **0** | 0.00% | selection@3,0,47=2.19e-03 | selection@3,0,47=1.41e-03 |
| omni-design-zh | 392 | 301 | 76.79% | argmax@31,7,16=2.59e-04 | argmax@31,7,46=6.41e-04 |
| omni-clone-en | 560 | 115 | 20.54% | selection@14,0,46=8.69e-04 | selection@29,5,13=1.19e-03 |
| omni-clone-zh | 784 | 254 | 32.40% | argmax@31,6,29=9.99e-04 | selection@25,2,53=1.28e-03 |
| omni-fast-mode | 400 | **0*** | 0.00% | selection@9,1,6=1.25e-03 | argmax@15,7,8=6.87e-05 |

\* `omni-fast-mode` matched its committed alternate grid
(`omni-fast-mode.alternate-grid-1.i32`), the same dual-admissible witness
Task 2/the 2026-07-31 ruling already covers — not a coincidental agreement
with the primary.

**Aggregate: 8,723 of 15,960 committed tokens (54.66%) differ from the CPU
baseline; 45.34% agree.** Per-case: **3/17 exact** (`omni-lang-none`,
`omni-design-en`, `omni-fast-mode`), 14/17 flip, several catastrophically
(`omni-digits` 98.30%, `omni-short-en` 94.00%). `ref.tokens` (the two
cloning cases' RVQ encode, unaffected by this patch since it never touches
`generator_branch_forward`) stayed 2/2 exact, as expected.

### Per-position detail for the four smallest flips

The full per-position `(codebook, frame, got, want)` quintuple for every
case with a tractable number of mismatches (the four largest flips run into
the thousands and are not reproduced element-by-element here; the counts
above and the raw `grid.i32` files under `build/goldens/omnivoice-replay-
task12-generator-cuda/` are the record):

```
omni-rate-fast (1 of 200):
  codebook 7 frame 13: want(cpu)=554  got(gpu)=984

omni-punctuation (4 of 536):
  codebook 7 frame 2:  want=658   got=597
  codebook 7 frame 7:  want=1018  got=761
  codebook 7 frame 23: want=14    got=481
  codebook 7 frame 32: want=217   got=1007

omni-nonverbal (10 of 488):
  codebook 6 frame 45: want=315  got=173
  codebook 6 frame 57: want=614  got=770   <- coincides with this run's
                                              own reported narrowest margin
                                              (argmax@31,6,57 = 2.85e-03)
  codebook 7 frame 3:  want=528  got=440
  codebook 7 frame 6:  want=863  got=865
  codebook 7 frame 7:  want=830  got=78
  codebook 7 frame 10: want=753  got=137
  codebook 7 frame 13: want=863  got=818
  codebook 7 frame 28: want=734  got=465
  codebook 7 frame 45: want=369  got=612
  codebook 7 frame 51: want=761  got=210

omni-upstream-readme (23 of 536): codebooks 4-7, various frames; none
  coincide with this run's own reported margin position (selection@19,1,5).
```

Only `omni-nonverbal`'s flip happens to land where the run's own margin
report points — and that position's margin (2.85e-03) is not itself
narrow; it is *not* in the sub-1e-4 screen band, nearly 3x the screen
threshold. The other three small-flip cases' actual flip positions are
elsewhere entirely. This is a limit of the instrumentation, not a surprise:
`MarginReport` (`generator-host.h`) keeps only the single narrowest
decision over the *whole run*, not a per-position ledger, so most flip
positions here have no recorded margin to quote — a run with 8 codebooks x
tens of frames x 32 steps makes far more decisions than the one the
instrument was built to surface.

### Interpretation: this does not match the margin-table prediction

The four in-band cases the family doc names as likeliest to flip first —
`omni-short-en` (1.16e-04), `omni-long-boundary` (2.05e-04), `omni-rate-fast`
(2.44e-04), `omni-lang-none` (6.03e-04) — plus `omni-rate-slow` (9.5e-06,
already below the screen) were the predicted leading indicators. Against
that prediction:

- Three of the five predicted cases flip (`omni-short-en`, `omni-long-
  boundary`, `omni-rate-slow`), one flips by a single token
  (`omni-rate-fast`), and one does **not** flip at all (`omni-lang-none`).
- Ten cases with comfortably "safe" CPU-vs-oracle margins — none within 6x
  of the screen — flip too, several worse than any of the predicted five:
  `omni-digits` (1.40e-03 margin, 98.30% flipped), `omni-short-zh` (2.46e-03
  margin, 48.15% flipped), `omni-short-ja` (2.66e-03 margin, 37.77%
  flipped).
- Only `omni-design-en` (1.41e-03) stays exact among the "safe" cases,
  alongside `omni-lang-none` and the dual-admissible `omni-fast-mode`.

So the answer is **no, the flip pattern does not match the margin table**,
and the reason is visible in the probe table the same run produced:

```
                        max_abs (GPU vs. CPU baseline)
generator.logits_step0  0.100906
generator.hidden_l0     0.00626373
generator.hidden_l7     0.0691681
generator.hidden_l14    0.0986023
generator.hidden_l21    0.726776
generator.hidden_l27    14.8662
```

Step 0's logit divergence alone (0.10 max_abs) is already ~90x the 6.1e-04
max_abs the margin screen was calibrated against (the CPU-port-vs-oracle
divergence the family doc's knife-edge ruling derives 1e-4 from), and by
the last of 28 layers it has compounded to 14.9 — five orders of magnitude
past the 1e-4 screen and roughly 4 orders past the widest of the 17 cases'
own recorded margins (2.66e-03). The margin table predicts which decisions
are *narrow relative to this port's own ~6e-4 CPU arithmetic difference
from the oracle*; TF32 compounding through a 28-layer transformer run 32
times, each forward doing both CFG branches, produces a perturbation two to
three orders of magnitude larger than that calibration basis. At that
scale nearly every decision in the suite is exposed, not only the
already-narrow ones — which is exactly what 14 of 17 cases flipping, with
several near-total, shows. `omni-lang-none` and `omni-design-en` surviving
are best read as luck at that scale, the same word the family doc already
uses for `omni-rate-slow`'s CPU-side margin — not as a demonstration that
either case's decisions are actually robust to it.

### Decision: the generator stays on CPU; no claim is made

Per the plan's own Step 2 rule ("Any flip → the generator stays on CPU, the
claim is not made, and the measurement is recorded so the next cycle does
not repeat it"): 14 of 17 greedy cases flip, most by a wide margin, so this
is not the ambiguous case the plan reserves for jiangzhuo — it is a clean
STOP. No change to `family_supports_explicit_backend` or any shipped
placement; `docs/backends.md`'s discrete-outputs rule already held this
stage on the CPU by construction (Task 11), and this measurement is
confirming evidence for keeping it there, not new grounds to reconsider it.
The Open Question in `docs/porting/families/omnivoice.md` gets this
measurement recorded alongside Task 11's policy answer, so a future cycle
does not re-run the same experiment expecting a different, more favorable
number.

### Revert, and re-verification that the shipped claim is unaffected

```
git checkout -- src/arch/omnivoice/model.cpp   # confirmed clean via git diff/status
cmake --build build/dev-dgx-spark --target synthesize-omnivoice-replay-real -j 20
uv run --project scripts/envs/omnivoice --locked python3 \
  scripts/validate-omnivoice-replay.py \
  --model models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf \
  --runner build/dev-dgx-spark/bin/synthesize-omnivoice-replay-real \
  --accelerate --profile F32 --backend CUDA --stage replay --require grid \
  --work build/goldens/omnivoice-replay-task12-revert-check
```

```
token grids exact: 17/17
ref.tokens exact: 2/2
```

Bit-for-bit the same result Task 11 committed — the shipped CUDA claim
(codec moves, generator does not) is unaffected by this task.

### Gates

| gate | result |
| --- | --- |
| `cmake --build build --target synthesize-check-unit` (CPU-only) | 90/90 passed |
| `cmake --build build-sanitize --target synthesize-check-unit` | 89/89 passed |
| `ctest --test-dir build-integration -L integration -R omnivoice` | 7/7 passed (902.49s: `cli` 34.28s, `public-cleanup` 96.01s, `load-real` 4.40s, `profile-test` 129.84s, `resampler-golden` 0.01s, `replay-golden` 441.74s, `public-request` 196.21s) |
| `scripts/ci/clang-format.sh --check-diff` | exit 0 (no C/C++ file carries a diff — the experimental patch was reverted before this ran) |

No source file differs from `HEAD` at the point this task committed; the
only committed changes are this entry, the family doc's Open Question
update, and the task report.

### Status

**DONE — STOP condition, as a complete and legitimate outcome.** The
generator is not claimed on CUDA. Commit follows this entry.

## 2026-08-07 — Plan 4 Task 16: The Listening Audit — `no_obvious_regression`, all six pairs

**Reordered to the front of Slice D at jiangzhuo's request** (ahead of Tasks
13–15): a `regression` verdict is a ship-blocker, and every audio-producing
slice (A/B/C) was already complete, so there was no reason to write the card
with `not_run` first and amend it. Preparation is `task-16-report.md`
(`.superpowers/sdd/2026-08-06-omnivoice-plan-4-quants-backends-ship/`); this
entry is the verdict record `docs/model-porting.md`'s Step 4 requires.

### Method: six pairs, `docs/model-porting.md:244-271`'s rule, with substitutions recorded

The doc's rule names five selection criteria — worst intelligibility, worst
UTMOSv2, worst voice-similarity, longest-duration, two deterministic random
cases — capped at six pairs. None of the three named quality scores exist for
this family: **ADR 0017 defers the whole intelligibility/UTMOSv2/
voice-similarity grid** (`quality_evaluated` is not a near-term goal), so
there is nothing to rank worst-of by those names. Substituted with what this
family *does* measure — the per-case `audio.pcm` cosine/max_abs
`scripts/validate-omnivoice-replay.py` already reports against the oracle for
all 20 golden cases (`build/goldens/omnivoice-replay/full-report.json`):

| Doc's slot | Substituted with | Slots |
| --- | --- | --- |
| worst intelligibility | worst cosine (no ASR evaluator exists) | 1 |
| worst UTMOSv2 | second-worst cosine (no naturalness evaluator exists) | 1 |
| worst voice-similarity | one clone case (`voice.kind=reference_audio`) **+** one Description Text case (`voice.kind=description_text`) — this family's two headline conditioning paths, rather than picking one arbitrarily | 2 |
| longest-duration | unchanged, but the comparison kind on this slot was spent on backend coverage (below) | 1 |
| two random | reduced to **one**, to hold the total at six after voice-similarity grew from 1 slot to 2 | 1 |

Total 1+1+2+1+1 = 6, at the doc's cap, nothing dropped.

**Backend coverage was folded into the longest-duration slot rather than
added as a seventh pair.** F32 is the only shipped profile (Q8_MIXED and F16
both failed the clone exact-token gate, Plan 4 Task 3/3-continuation), so
every pair trivially covers it. CPU and CUDA are both shipped backends (Task
11/12), but six pairs leaves no independent slot for that axis. Rather than
exceeding the cap or dropping a required criterion, the longest-duration
case's comparison kind was changed from port-vs-oracle to **CUDA-vs-CPU codec
decode of the same committed token grid** — deliberately `omni-long-boundary`,
because Task 11's own report already identified it as the case with the
codec's largest measured CUDA effect (9.68x speedup at 719 frames), the
single most informative case to spend that slot on.

Both clone/design picks went to the `-en` variant over `-zh`:
`docs/model-porting.md:265`'s "for a language the maintainer understands" —
English is that language here.

### The six pairs and the identity key

Two comparison kinds. **Port vs oracle** (5 pairs): the port's own codec
decoding the ORACLE's committed token grid
(`build/goldens/omnivoice-replay/<case>/pcm.f32`, isolating the codec from the
decode loop) against the oracle's own waveform
(`build/goldens/omnivoice/<case>/audio/pcm.f32`). **CUDA vs CPU** (pair 3):
the same case's codec decode of the *same* committed token grid, once per
backend — `grid.i32` verified byte-identical between the CPU and CUDA replay
directories before use, so the two waveforms differ only by codec backend.

| Pair | Case | Comparison | A | B | Duration | Cosine |
| --- | --- | --- | --- | --- | ---: | ---: |
| 1 | `omni-medium-en` | port vs oracle | oracle | port | 12.28s | 0.9999998558 |
| 2 | `omni-nonverbal` | port vs oracle | port | oracle | 2.44s | 0.9999998990 |
| 3 | `omni-long-boundary` | **CUDA vs CPU** | cuda | cpu | 28.76s | 0.9999983311 |
| 4 | `omni-clone-en` | port vs oracle | oracle | port | 2.80s | 0.9999999031 |
| 5 | `omni-design-en` | port vs oracle | oracle | port | 2.00s | 0.9999999511 |
| 6 | `omni-punctuation` | port vs oracle | oracle | port | 2.68s | 1.0000002084 |

Cosine values fractionally above 1.0 (pairs 1, 5, 6 read below 1.0, pair 6
above) are `compare()`'s own float32 rounding in the dot-product/norm
computation, not a bound violation — the same artifact the deep generator
probes already show elsewhere in this log.

Pair 3's figures are not in `full-report.json` (that report only ever ran the
CPU backend); computed once, directly, against the two already-produced
replay directories, with the project's own locked numpy env before the
manifest was written: max_abs 0.0030613476410508156, cosine
0.9999983310699463 — roughly three orders of magnitude larger than any
port-vs-oracle max_abs above (1e-5 to 1e-6 range), consistent with CUDA's
TF32 cuBLAS path (`docs/backends.md`) rather than a placement defect, since
the feeding token grid is byte-identical and the divergence is purely the
codec backend's arithmetic.

### Seeds and the A/B swap pattern

Two independent seeds, so re-deriving one never perturbs the other:

- **`case_selection_seed = 16`** (the task number) drives exactly one draw,
  `random.Random(16).choice(sorted(remainder))`, over the 15 cases left after
  the five fixed-criteria picks — landed on `omni-punctuation`.
- **`order_seed = 20260807`** (the audit date) drives one
  `random.Random(20260807).random() < 0.5` draw per pair, consumed in pair
  order 1→6, deciding whether that pair's first-defined identity (`port` /
  `cuda_codec`) is presented as A or as B.

**5 of the 6 pairs were A/B-swapped** from their first-defined identity (only
pair 2 was not), so the port (or the CUDA side) was not positionally
guessable across the set. The mapping lived only in `audit-manifest.json`
(sha256-pinned source paths for both sides of all six pairs), a sibling file
never embedded in the page; the page itself was grepped for `oracle`,
`port_vs_oracle`, `cuda_codec`, `cpu_codec`, every `omni-*` case id, and the
`build/goldens` path prefix, with zero matches outside the identity key.

### Verdict

**jiangzhuo, 2026-08-07: all six pairs "no obvious difference" ⇒
`listening_audit: no_obvious_regression`.**

### Pair 3 corroborates the backend claim, not merely accompanies it

Pair 3 is the one pair this audit spent on Execution Backends rather than
port-vs-oracle fidelity. Among the six audited pairs its max_abs (3.06e-03,
~450x pair 1's) is the largest, but that is an artifact of comparison kind,
not evidence about the case: five pairs compare a port against an oracle
(1e-5 to 1e-6 range) and only this one compares two backends, so it was never
going to land in the same range. `omni-long-boundary` was the case at hand
because it holds the audit's longest-duration slot, the same case Task 11
already named as the codec's largest measured CUDA *speedup* (9.68x at 719
frames) — a different claim from the largest measured CUDA numeric
divergence across the family, a distinction an earlier draft of this entry
blurred. **Fix (final branch review, 2026-08-07): recomputing
CPU-vs-CUDA `audio.pcm` cosine directly from `build/goldens/omnivoice-replay`
and `build/goldens/omnivoice-replay-cuda` for all twenty cases ranks
`omni-long-boundary` 5th of 20 by that measure** — behind `omni-upstream-
readme`, `omni-medium-en`, `omni-nonverbal` and `omni-short-en`, in that
order — so it is not, and was never claimed by Task 11 to be, the family's
largest CUDA divergence; the selection reason stands on the longest-duration
slot and the speedup finding alone. Task 11 already established the
*structural* half of the CUDA claim: byte-exact token grids on all 17 greedy
cases plus both clone cases, so nothing upstream of the codec's decode moved.
What a tolerance grid cannot show is whether the codec's own float32
arithmetic difference between backends is large enough to hear. Pair 3
answers exactly that question, on the case the longest-duration slot already
pointed to — and the answer is no. This is corroborating evidence for the
backend claim already made on structural grounds, not a second, independent
claim standing beside it: the byte-exact grids proved the CUDA path decodes
the same discrete decisions, and pair 3 shows that the floating-point
divergence the codec's CUDA path produces on a representative case is
inaudible to the one listener who checked. Had pair 3 come back "audible
difference," it would not have falsified Task 11's structural evidence, but
it would have been a reason to look harder at the codec's CUDA kernels before
shipping the backend regardless of what the token grids said — a tolerance
number is not a substitute for that check, which is the whole reason this
task exists.

### What this does, and does not, establish

Per `CONTEXT.md`'s definition, a Listening Audit "records obvious regressions
without claiming population-level subjective quality," and per
`docs/model-porting.md` it "is never reported as MOS, CMOS, a listening
panel, or population-level evidence." This was **one listener, six pairs,
informally, no rated comparison, no panel, no score.** It says: on the six
automatically selected pairs above, at that hearing, no obvious problem was
noticed. It does not say the port sounds identical to the oracle in general,
does not say the CUDA and CPU codecs are perceptually equivalent in general,
and does not move `quality_evaluation` off `not_run` — ADR 0017's automated
grid has not run and is not scheduled. `docs/porting/families/omnivoice.md`
records the outcome in the family record.

### Values for Task 14's card

Not written to any YAML here — Task 14 owns the card. For that task's
`listening_audit` / `listening_audit_detail` fields, following the shape
`kokoro-v1-0.yaml` and `qwen3-tts-12hz-0-6b-customvoice.yaml` already use:

```yaml
listening_audit: no_obvious_regression
listening_audit_detail:
  listeners: 1
  cases: 6 (5 port_vs_oracle, 1 cuda_vs_cpu)
  profiles: [F32]
  backends: [CPU, CUDA]
  date: 2026-08-07
  method: >-
    Blind A/B web page, five pairs replaying the port's codec against the
    pinned PyTorch oracle on the oracle's own committed token grid, one pair
    (omni-long-boundary, this family's largest measured CUDA effect)
    comparing the codec's CUDA decode against its CPU decode of the same
    byte-identical committed grid; A/B order independently randomized per
    pair (order_seed 20260807, 5 of 6 swapped) and case selection randomized
    for the one non-criteria slot (case_selection_seed 16); verified in a
    real headless browser for no trimming, offset-preserving side switch, and
    zero drift on a rapid double-switch.
```

### Status

Complete. Nothing under `build/`, `tests/golden/`, `ggml/`, or `third_party/`
was modified; the audit's page, manifest and scripts are working artifacts
under `$CLAUDE_JOB_DIR`, never the repo. Docs-only commit follows this entry.

## 2026-08-07 — Plan 4 closeout: the four slices tied together

Plan 4 took OmniVoice from "port-validated on CPU at F32" (Plan 3's close) to
a publication-ready Restricted Model Package, in four slices, each closed on
a measurement rather than an assumption. This section ties them together;
none of the individual entries above are restated, only cross-referenced.

**Slice A — Quantization (Tasks 1–4), negative and explained.** Task 1's
classifier and Task 2's packed-convolution branch are real, tested, shipped
infrastructure regardless of the outcome. Task 3 measured both codec-only
candidate profiles against the exact-token gate:

| profile | size | sha256 | greedy grids | clone RVQ grids | mismatch | narrowest gap |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| F32 (reference) | 3,189,953,504 B | `f6d504ff…f9fa3` | 17/17 | 2/2 | — | 0.00239563 |
| Q8_MIXED | 2,703,016,576 B (−15.3%) | `b020933f…4b671e` | 17/17 | 0/2 | 1023/2808 (36.4%) | 0.0102997 (wider than F32, mismatch anyway) |
| F16 | 2,858,422,240 B (−10.4%) | `530b2b85…0b955fa` | 17/17 | 0/2 | 103/2808 (3.7%) | 0.00306702 |

Both fail; both are BLOCKED, not shipped; no tolerance cell exists for
either. The greedy margin table is bit-identical across all three profiles
(the generator is Sensitive/F32 under every profile), which is why none of
the margin-table's four predicted-first-flip cases (`omni-short-en`,
`omni-long-boundary`, `omni-rate-fast`, `omni-lang-none`) or `omni-rate-slow`
actually flipped — the real failure lives entirely in the clone-encode RVQ
lookup, a subsystem the greedy margin screen was never built to probe. The
structural reason, computed once and not revisited: **93.8% of the
quantizable weight (593.3 of 632.3 MiB) is the clone-encode path**
(`semantic_model` + `acoustic_encoder` + `encoder_semantic`) feeding a
discrete RVQ nearest-neighbour decision, and the only 6.2% that is safe to
quantize (`acoustic_decoder`, 39.0 MiB) buys 0.64–0.94% of package size for a
measurable waveform change. The tensors worth quantizing are exactly the
ones that cannot be. **This family ships F32-only.**

**Slice B — Validation-suite debt (Tasks 5–7), all three carry-over items
closed.** Task 5 added
`synthesize-omnivoice-serialize-writer-agreement-test` (labels
`unit;omnivoice`), proven in both drift directions and, after fix round 1,
per-kind exact-equality plus the two `n_kv` count constants — closing the
dangerous direction (a key silently removed from a writer while the
whitelist stays permissive) that no prior fast test could see. Task 6 pinned
all 20 cases' primary-grid sha256 digests in the manifest (a new, additive
`stochasticInput` schema def, `suite_version` unchanged), extended
`scripts/validate-omnivoice-replay.py` to verify them before any comparison
runs (proven pre-spawn: a corrupted digest fails in well under a second, a
real run takes 400+), and added
`test_omnivoice_primary_grids_are_digest_pinned` to
`synthesize-golden-manifest-contract`. Task 7 found both of Plan 1's
carry-over converter defects (the `verify_gguf` shape check, the
license-copy ordering) were already fixed on `main` before Plan 4 began
(`b08757d`, `3b1d88f`, both 2026-07-31); the debt was purely missing unit
coverage, closed with four new tests plus a defense-in-depth reorder,
converter emission proven unchanged.

**Slice C — CUDA Execution Backend (Tasks 8–12), claimed for the codec,
refused for the generator, both on measurement.** Task 8 closed a live
honesty defect predating this plan (every family, OmniVoice included, could
accept `SYNTH_BACKEND_CUDA` and silently run entirely on CPU); after fix
round 1 the family-blind cleanup-test gap Task 8's own review found was
closed too. Task 9 built the codec's accelerator twin with a filter
narrower than qwen3-tts's blanket `codec.*` mirror (152 movable tensors,
84.24 MiB; a copied blanket filter would have mirrored all 486 `codec.*`
tensors, ≈700 MiB, of which the 334 the CPU-held clone-encode chain reads —
616.01 MiB, measured, not the ~593 MiB the pre-Task-9 survey estimated —
would have gone to the device for nothing), and fix round 1 caught and
fixed a second-consumer trap the first
draft missed (`codec.quantizer.*` is read by both the decode graph and
host-side `rvq_encode`). Task 10 made placement checkable. Task 11 ran the
sweep and claimed the backend:

- Placement, aggregated over all 20 cases: generator 0/880,032 nodes off
  CPU; codec 8,440/8,440 nodes off CPU — both unconditional.
- Token grids: 17/17 greedy + 2/2 clone RVQ byte-exact against the CPU
  baseline, no flip.
- Waveform: `audio.pcm` worst cosine 0.9999963545 (deviation ≈3.65e-6),
  committed at `backends.CUDA.stages.replay` (min_cosine 0.999981, max_abs
  0.04 — 5× measured, this family's standing rule).
- Cost is inverted from Kokoro/VITS: codec alone is 9.68× faster on the
  longest case, but the held generator is 98.9% of wall time, so end-to-end
  is a 3.4% saving (267.30 s → 258.48 s), not a tax.
- UMA peak memory has no second budget to report on this hardware
  (`SYNTH_DEVICE_MEMORY_SHARED`, `memory_total` bit-identical to the CPU
  device's); host RSS is flat within 4 KB, and the 1,167 MiB transient
  `nvidia-smi --query-compute-apps` figure is a momentary share of the one
  128 GB pool, not a dedicated-VRAM requirement.
- Two registered CUDA gates, neither VITS/Kokoro/qwen3-tts had before this:
  `synthesize-omnivoice-replay-golden-cuda` (869.83 s) and
  `synthesize-omnivoice-public-request-cuda` (389.57 s).

Task 12 then ran the empirical experiment the family doc's own Open Question
asked for — generator on CUDA, hand-reverted, never shipped — and it FLIPS:
3/17 greedy grids byte-exact, 14 flip (up to 98.30%), 54.66% of all 15,960
committed positions differ. **The valuable part is that the flip pattern
does NOT match the margin table**: three of the five margin-predicted
first-flip cases do flip, one flips by a single token, one does not flip at
all, while ten comfortably-safe-margin cases flip too, several worse than
any predicted case (`omni-digits`, margin 1.40e-03, 98.30% flipped). This is
a magnitude problem, not a knife-edge one: step-0 logit divergence from the
CPU baseline is already ~90× the margin screen's own calibration basis
(0.10 vs. 6.1e-04) and compounds through 28 layers, 32 steps and 2 CFG
branches per step to 14.9 max_abs by the final layer — several orders past
both the screen and the widest margin any of the 17 cases carries. **The
margin screen is calibrated for a different scale of perturbation entirely
and says nothing useful about TF32 at this depth**; it governs which GREEDY
cases are safe to adopt into the Golden suite on CPU-vs-oracle arithmetic,
and does not transfer to a backend-placement question at all. The decision
follows the plan's own rule without needing to reach jiangzhuo: the
generator stays on CPU, no claim is made, and the discrete-outputs rule
would have held it there regardless of this measurement's outcome.

**Slice D — Ship (Tasks 13–17).** Task 13 taught the shared HF card
generator OmniVoice's shape (Sidecar Resources, text input, `--text` usage,
the third `seed_default_with_profiles` voice mode, conditional CUDA-placement
prose) — 31/31 generator tests (10 pre-existing + 21 new), 267 python tests,
and found a real latent bug in the unreleased qwen3-tts card template along
the way. Task 14 built the clean, flat `models/publish/omnivoice-0-6b/`
directory (F32 GGUF, the Boson sidecar, the rendered README — no upstream
checkpoint, no non-shipping profile) and found a second false template claim
(unconditional "also accepts exact token IDs", false for a `TEXT_UTF8`-only
family) — 34/34 + 23/23 python tests. Task 16 (moved ahead of 13–15 at
jiangzhuo's request once every audio-producing slice was complete) recorded
the Listening Audit verdict: **`no_obvious_regression`, all six pairs**,
including the CUDA-vs-CPU codec pair (pair 3, the largest numeric divergence
in the set) corroborating Slice C's backend claim with the one kind of
evidence a tolerance grid cannot provide. Task 15 brought ADR 0018's
Restricted Model Package vocabulary into `docs/scope.md`,
`docs/model-packages.md`, and `CONTEXT.md`'s Validation Level entry. Task 17
(this entry, plus the family doc's Status line, `docs/testing.md`'s new
gates, the design spec's license-sentence amendment, and the Plan 5
carry-over ledger) is the close-out.

**What ships.** `omnivoice-0-6b-F32.gguf` only, as a Restricted Model Package
(ADR 0018) under `license: other` /
`license_name: omnivoice-cc-by-nc-unspecified-version-plus-boson-higgs-audio-2-community`,
CUDA claimed for the codec's decode path only, `listening_audit:
no_obvious_regression`. Publication itself — `hf repos create` / `hf
upload` — is a separate act awaiting jiangzhuo's explicit per-act
confirmation naming the target repository; nothing in this plan performed
it.

**What Plan 4 newly owes.** Recorded in full, each naming its file, in
`docs/superpowers/plans/2026-08-07-omnivoice-plan-5-carryover.md`: the margin
screen does not transfer to backend-placement questions (Task 12's finding,
above); `models/publish/` is a working, git-ignored directory the upload
command depends on and must be rebuilt if the shipped GGUF ever changes; and
the four project-wide questions this plan's own carry-over ledger declined
to settle unilaterally (`SYNTH_ASSERT` under `NDEBUG`, the
`synthesis.graph_failed` diagnostic name, the `std::optional` stream
micro-optimization, and the two-clips validation-order test), plus the
float64 cosine-estimator question, both still open from Plan 3's own ledger.

## 2026-08-08 — Plan 5 Tasks 1–2: the generator weight twin, the placement move, and an honest RTF

jiangzhuo revised the bar for this family from token identity to audible
quality on 2026-08-08, after a six-pair blind A/B (order seed 20260808, case
seed 12) comparing generator-on-CPU against generator-on-CUDA output heard
no problem in any pair, including `omni-digits` at waveform cosine 0.0515
with 98.3% of its tokens flipped between the two runs. That ruling reopens
the placement Plan 4 Task 11/12 closed by the discrete-outputs rule: the
generator can now move to the primary Execution Backend, which — per two
independent RTF investigations recorded before this plan — is the only
order-of-magnitude lever this family has (~1,722 token-forwards of a 0.44B
model per second of synthesized audio, mask-predict's 32 steps × no KV cache
× 2 CFG branches, against 25 for a cached autoregressive decoder).

### Task 1: the generator weight twin

Followed Task 9's codec-twin pattern exactly, generalized to a group with no
NOT-MOVABLE remainder: `bind_generator_weights` (`catalog.h`/`catalog.cpp`)
starts `generator_weights` as a whole-struct copy of `weights` and only
re-resolves `generator_weights.generator` against a twin context when one is
given, so `weights` itself — what any future second consumer would read — is
never mutated. **Before writing anything, every host-side reader of
`GeneratorWeights` was grepped across `src/` and `tests/`**: the only two
consumers are `build_canvas_embedding` and `build_generator_forward`
(`generator.cpp`), both reached exclusively through `model.cpp`'s file-local
`generator_branch_forward`, itself called only from `Model::run_synthesis`'s
three sites. Unlike `codec.quantizer.*` in Task 9's own twin, there is **no
second consumer analogous to `rvq_encode`** — no host-side reader of the
generator's weights outside the graph-building path — but the split is built
the same defensive way regardless, both because a future reader that
bypasses `generator_branch_forward` must keep seeing the CPU-resident
package by construction, and because `tests/omnivoice_catalog_test.cpp`'s
own unit tests call `build_model_weights` directly and must observe it
unaffected by whatever this function does elsewhere.

`Model::Impl` gained `generator_context`/`generator_buffer`/
`generator_weights`, built and streamed in `Model::load` the same way as the
codec's own `codec_context`/`codec_buffer`/`decode_weights`, sized for the
whole 312-tensor, 2,450,309,120-byte (2,336.80 MiB) generator group (computed
directly from `reports/convert/omnivoice/omnivoice-0-6b-F32.json`, summing
every tensor whose name starts `llm.` or equals
`audio_embeddings.weight`/`audio_heads.weight`). `generator_branch_forward`
gained an `on_primary` parameter forwarded to `GraphRun::run`;
`Model::run_synthesis` passes `impl.generator_weights` and
`impl.generator_context != nullptr` at all three call sites (the step-0
conditional forward, the per-step conditional refill, and the per-step
unconditional branch) — **both CFG branches move together**, since both read
the identical generator weights and there is no reason for one to run on the
primary backend while the other stays on the CPU. The `Persistent inputs`
buffer commits to the primary backend under the same condition, mirroring
`decode_codes`'s own rule for its input leaf and avoiding a cross-backend
copy the scheduler would otherwise insert.

**CPU bit-identity, proven not asserted.** With no accelerator,
`generator_context` stays null, `bind_generator_weights` returns a plain
copy, and `on_primary` is always `false` — byte-identical to before this
task. Verified:

```
uv run --project scripts/envs/omnivoice --locked python \
  scripts/validate-omnivoice-replay.py \
  --manifest tests/golden/omnivoice/omnivoice-0-6b.manifest.json \
  --model models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf \
  --runner build-integration/bin/synthesize-omnivoice-replay-real \
  --check --profile F32 --backend CPU --stage replay
```

```
token grids exact: 17/17
ref.tokens exact: 2/2
```

Both unit gates: `cmake --build build --target synthesize-check-unit`
(90/90 passed), `cmake --build build-sanitize --target synthesize-check-unit`
(89/89 passed, ASan/UBSan). The full omnivoice integration set on the CPU
tree (`ctest --test-dir build-integration -L integration -R omnivoice`):
7/7 passed, including `synthesize-omnivoice-replay-golden`'s own 17/17 +
2/2 exact-token confirmation above.

**A cross-check the implementation did not have to produce, but did.**
Running the new twin under `--accelerate` on `omni-long-boundary` alone
(`--cases omni-long-boundary --accelerate --backend CUDA --require grid`)
reports `generator nodes left the CPU: [53184, 53184]` — every one of this
case's 53,184 generator nodes, the identical total Plan 4's own CPU sweep
recorded for this case, confirming this is the same graph with only its
scheduler changed — and a token grid mismatch of **4,190 of 5,752
positions (72.84%)**, the exact count Plan 4 Task 12's hand-reverted,
one-line `on_primary=true` experiment measured for this same case one day
earlier. Two independently-built mechanisms (a real weight-mirroring twin
here; an `op_offload` scheduler patch there) landing on the identical flip
count is strong evidence both are exercising the same underlying CUDA
arithmetic rather than either one being a measurement artifact.

**This deliberately leaves one gate red.**
`tests/CMakeLists.txt`'s `synthesize-omnivoice-replay-golden-cuda` and
`tests/tolerances/omnivoice.json`'s `profiles.F32.backends.CUDA.stages.replay`
cell assert byte-exact tokens under `--backend CUDA` — true before this task,
false after it, by construction. Plan 5 Tasks 3–4 own the replacement
validation shape; neither the tolerance file nor the CUDA gate's assertion
was touched by this task, per the controlling instruction.

### Task 2: RTF, measured honestly, through the public seam

Built `synthesize-omnivoice-public-real` and `synthesize-omnivoice-replay-real`
on the `dev-dgx-spark` CUDA tree with Task 1's changes
(`SYNTH_CUDA_ROOT=/usr/local/cuda-13.3`, driver 580.159.03, CUDA 13.3.73,
native `sm_121a`). Measured `omni-long-boundary` — the same case Plan 4's own
codec-only CUDA claim measured at 267.30 s / RTF 9.294 — end to end through
`synth_synthesize_to_buffer` (`tests/omnivoice_public_real.c`, not the
internal replay runner, which bypasses the backend-capability gate), same
binary, `cpu` vs `cuda` backend positional, seed 0, language `en`,
`max-frames 0`, text from the case's own committed
`tests/golden/omnivoice/omnivoice-0-6b.manifest.json` entry. Two runs each
way:

| backend | run 1 (s) | run 2 (s) | mean (s) | RTF (mean) |
| --- | ---: | ---: | ---: | ---: |
| CPU  | 272.1212 | 267.3172 | 269.7192 | 9.365 |
| CUDA |   6.5184 |   6.3898 |   6.4541 | 0.224 |

(691,200 PCM frames at 24,000 Hz = 28.8 s of audio in every run; both
backends produced the identical sample count, confirming the canvas length
argument this task's own comments make — TF32 changes WHICH token is
committed, never HOW MANY frames exist.) `resolved_device` confirmed
`"cpu"`/`"cuda"` in every run (not merely requested), and
`device_memory_shared` flipped `false`→`true` on the CUDA runs as expected
for this UMA host.

**Speedup: 41.8× (269.7192 s / 6.4541 s), taking this case from RTF 9.365 to
RTF 0.224 — faster than real time.** This is the number the whole plan
exists for, and it is not a flattering rounding: even the SLOWER of the two
CUDA runs (6.5184 s) against the FASTER of the two CPU runs (267.3172 s)
still gives 41.0×. Plan 4's own codec-only figure recovered 3.4% of wall
time on this case because the generator, 98.9% of it, stayed held; moving
the generator recovers the other 96.6 percentage points. For comparison,
ServeurpersoCom's own port reports RTF 0.194 on an RTX 4060 Ti with the
whole generator on CUDA at Q8_0 (and states no fidelity claim beyond "smoke
surfaces" at any dtype) — this port's F32 measurement on DGX Spark/GB10
lands at RTF 0.224, the same order of magnitude, at full F32 precision and
without the step-count or quantization reductions Tasks 8–9 have not yet
revisited.

**What this measurement does not establish.** One case, one host, two runs
each way — not a claim of Support (Task 4 owns that once Task 3's
validation shape exists to back one), not a sweep across the other 19
golden cases (which would very likely show a smaller relative gain on
shorter cases, where fixed per-request overhead — load, resampling, WAV
framing — is a larger fraction of the total), and not a memory or
concurrency measurement. It is one honest, reproducible number: this is
what the generator's move buys on the case Plan 4 itself picked as the
suite's longest and most demanding.

**Implications for what this task deliberately left untouched (per the
controlling instruction):**

- **The model card and `docs/models/omnivoice-0-6b.md`** (Plan 4 Task 14's
  ship artifacts) state F32-only and CPU+partial-CUDA (codec only) with the
  267.30 s/RTF 9.294 figures. Both the backend scope and the RTF figures are
  now stale; Plan 5 Task 7 corrects them once Task 3/4 settle what the CUDA
  claim's own validation shape is, so the card is not corrected twice.
- **`tests/tolerances/omnivoice.json`'s CUDA backend cell and
  `tests/CMakeLists.txt`'s `synthesize-omnivoice-replay-golden-cuda`** now
  assert something false (byte-exact tokens under `--backend CUDA`) and will
  fail if run; Plan 5 Task 3 designs the replayed-grid/waveform-tolerance
  replacement described in this plan document, and Task 4 registers it.
  Left unregistered/unrun deliberately rather than silently loosened.
- **`docs/backends.md`'s discrete-outputs rule and its per-family cost
  table row** (currently: "the whole generator... held... 267.30 s → 258.48
  s, −3.4%... 8.99× real time") describe a configuration this task
  supersedes. Plan 5 Task 5 owns the principled-exception writeup (the
  fixed-canvas-shape argument, Kokoro's 376→377-frame counter-example, and
  this task's own 53,184/53,184-off-CPU + 691,200-frames-both-ways evidence)
  and the table row's replacement with this measurement's own numbers.

## 2026-08-08 — Erratum: the RTF measurement build defect, and the honest correction

This entry corrects, without rewriting, three prior measurements in this
file: the Plan 4 Task 11 codec-only pair above (267.30 s → 258.48 s, RTF
9.294 → 8.987), the Plan 4 closeout summary that repeats it, and the Plan 5
Task 2 entry immediately above this one (269.7192 s / RTF 9.365 CPU,
6.4541 s / RTF 0.224 CUDA, **41.8×**). All three stand as originally written;
this entry states what was wrong with the build each was measured on, and
gives the honest replacement.

### The defect

Every one of the RTF figures above — Plan 4's and Plan 5's alike — was
measured on `build/dev-dgx-spark`. That preset inherits `development-base`,
which sets `CMAKE_BUILD_TYPE: RelWithDebInfo` (`CMakePresets.json:32`), so
ggml-cpu compiled at **`-O2 -g`**. Verified directly from both trees'
`flags.make`:

| tree | ggml-cpu C_FLAGS |
| --- | --- |
| `build/dev-dgx-spark` | `-O2 -g -DNDEBUG` |
| `build/` (plain, defaults to Release) | `-O3 -DNDEBUG -mcpu=native` |

`-O2` costs **2.19×** on this workload. RTF 9.294 (Plan 4) and RTF 9.365
(Plan 5 Task 2's own CPU arm) are both a property of the measurement build,
not of the code. Shipped wheels were never affected: release presets use
`Release` (`CMakePresets.json:45`) and a bare `cmake -S . -B build` defaults
to `Release` (`CMakeLists.txt:39-40`) — only this project's own published
numbers were pessimistic, by roughly that factor.

### The honest pair

Measured on `build/rel-dgx-spark` — the `dev-dgx-spark` CUDA settings with
`CMAKE_BUILD_TYPE=Release`, now the committed `rel-dgx-spark` CMake preset
(`CMakePresets.json`). Same binary for both arms, `omni-long-boundary`
(719 frames = 28.76 s of audio), through `synth_synthesize_to_buffer`,
round-robin interleaved, N=3 per arm, every run under 1.0 other-cores of
contention (the host was quiet — load 1.41, versus load 50 during the
original Plan 5 Task 2 measurement).

| arm | best | mean | RTF (best) |
| --- | ---: | ---: | ---: |
| CPU | 122.61 s | 123.02 s | **4.263** |
| CUDA, generator on GPU | 5.481 s | 5.518 s | **0.1906** |

**Speedup 22.4×.** Not the 41.8× Plan 5 Task 2 reported. Both ends moved: the
CPU arm was 2.2× overstated, and the CUDA arm is itself ~1.17× faster at
`-O3` (its host-side sampling loop is CPU code too).

### A second correction to the same entry: the RTX 4060 Ti comparison

Plan 5 Task 2's own closing comparison ("For comparison, ServeurpersoCom's
own port reports RTF 0.194 on an RTX 4060 Ti... at Q8_0") misattributes the
benchmark. Verified directly against both repositories' own READMEs: the
`[rtf] total=0.194  seconds=14.932` figure on a local RTX 4060 Ti run is
`bluryar/omnivoice.cpp`'s own reported number (its README's Performance
section, `omnivoice-q8_0.gguf`), not `ServeurpersoCom/omnivoice.cpp`'s —
`ServeurpersoCom/omnivoice.cpp`'s own README carries no RTF figure at all.
With the correction above, this port's own honest RTF 0.1906 on DGX
Spark/GB10 (full F32 precision, no step-count or quantization reduction) is
in the same range as `bluryar`'s reported 0.194 on an RTX 4060 Ti at Q8_0.

### `-mcpu=native` is worth nothing

Two fresh Release CPU trees, N=3 each, interleaved, long case:

| tree | mean |
| --- | ---: |
| `GGML_NATIVE=ON` | 122.089 s |
| `GGML_NATIVE=OFF` | 122.003 s |

**0.07% apart — inside noise.** This settles a contradiction in the original
investigation (one agent's GEMM microbenchmark claimed 1.45× for native; an
end-to-end four-build A/B claimed nothing). End-to-end wins. It matters
because `pyproject.toml:42` sets `GGML_NATIVE=OFF` for wheels and aarch64 has
no runtime variant dispatch — so the entire `-O3` win is shippable as-is,
with no packaging change and no new CPU-variant machinery.

### Bit-identity holds

`-O2` output and all three `-O3` outputs hash identically
(`43db99bc…dca2d`). The Release CPU tree still passes the replay check at
**17/17 exact token grids, 2/2 exact clone-token grids**, and
`synthesize-omnivoice-replay-golden` passes (443.62 s).

### Open question, recorded not investigated: one CPU run in ten hashed differently

Of ten CPU runs collected during this re-measurement, **one**
(`rel-cpu-native` run 1) hashed differently from the canonical hash — 119 of
2,760,960 bytes, all in the last ~3% of the waveform. Runs 2 and 3 on that
same tree matched the canonical hash. This is run-to-run nondeterminism on
the CPU path, independent of `-O2`/`-O3` and of native/non-native (it
occurred on a native-CPU Release tree; nothing about this build's
optimization level or CPU-variant flag distinguishes it from the nine runs
that matched).

**This matters because this family's whole contract is exact tokens.**
Two things are known and two are not. Known: the replay gate (which checks
committed token grids, not the decoded waveform) passed on the tree this
divergent run came from, and the byte difference is confined to a small tail
of the waveform rather than spread throughout it. Not known: whether the
divergence originates in the generator's own token draws (with the codec
merely propagating an already-different grid) or downstream of the
generator, in the codec/vocoder's own arithmetic on an identical grid — a
119-byte PCM tail difference is consistent with either. The replay gate
passing weakly suggests the latter (downstream of the generator), but **that
is not established** — it has not been cross-checked against the actual
token grid the divergent run produced, and it is not this entry's job to
settle: **do not let it disappear into a footnote.** A future task needs a
deliberate repeat-run experiment (many more than ten runs, with the token
grid captured and diffed alongside the PCM on every run) before this can be
called understood.

### What this does and does not establish

One case, one host, N=3 per arm for the honest pair (N=3 per tree for the
`-mcpu=native` question, N=10 for the CPU determinism sweep that surfaced the
open question above). Not a sweep across the other 19 golden cases, not a
memory or concurrency measurement. It is what the corrected build changes,
measured honestly, plus one open question that measurement surfaced and does
not answer. Cross-references: `docs/backends.md`'s per-family cost table and
"Discrete-Outputs Rule Admits One Narrow Exception" section,
`docs/porting/families/omnivoice.md`'s Execution Backends section,
`docs/models/omnivoice-0-6b.md`'s Backends section, and `docs/testing.md`'s
"Performance Figures Require A Release Preset" — all four updated with these
numbers as part of this correction. `docs/port-validation.md` and the sibling
families' (VITS, Kokoro, qwen3-tts) own CPU figures were checked for the same
dev-preset provenance defect as part of this correction where they cited
OmniVoice; qwen3-tts's own unrelated oracle CPU-vs-CUDA figure
(`docs/port-validation.md:103`, "9.4 times") was inspected and found to be
about the PyTorch oracle's own device choice, not this project's C++ build
type, and is left untouched — checking whether *that* figure has its own
provenance problem is unstarted, sibling-family work this correction does
not do.
