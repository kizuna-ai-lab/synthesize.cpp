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
