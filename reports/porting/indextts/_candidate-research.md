# IndexTTS / IndexTTS-2.5 — Candidate-Family Research for synthesize.cpp

Date: 2026-08-21
Author: research pass (primary-source only)
Sources: cloned `index-tts/index-tts` @ `ee40fa7` (2026-08-18, `main`), Hugging Face
model repos and API metadata (`IndexTeam/IndexTTS-2.5`, `IndexTeam/IndexTTS-2`,
`IndexTeam/IndexTTS-1.5`, `facebook/w2v-bert-2.0`, `funasr/campplus`,
`nvidia/bigvgan_v2_22khz_80band_256x`, `Richasy/IndexTTS-2.5-GGUF`), arXiv 2601.03888,
the `0xShug0/audio.cpp` repository (docs, model spec, README, LICENSE) read through the
GitHub API, and one upstream issue. No outward interaction with any repository was
performed: every access was a read, a clone, or an unauthenticated API GET.

This is pre-intake candidate research, not an intake record; it precedes any
fifth-family decision in `docs/model-family-selection.md`. A separate VoxCPM
candidate-research note exists on the unmerged branch
`worktree-voxcpm-candidate-research`; the two are competing candidates, not a sequence.

---

## 1. TL;DR

IndexTTS-2.5 is bilibili/IndexTeam's zero-shot cloning TTS: a **0.8 B GPT-2-style
autoregressive Text-to-Semantic model** emitting 25 Hz semantic codes, a **residual-VQ
semantic codec** with Vocos/ConvNeXt encoder-decoder, a **13-layer DiT flow-matching
Semantic-to-Mel decoder** (25 Euler steps, CFG 0.7), and **BigVGAN v2** at 22.05 kHz —
plus, at inference time, a **580 M-parameter Wav2Vec2-BERT 2.0 semantic encoder**, a
**CAMPPlus** speaker embedder, and an optional **Qwen3-0.6B** emotion classifier. Five
languages (zh/en/ja/es/ar), emotion control disentangled from timbre, `duration_factor`
speed control, and inline `<字|reading>` pronunciation overrides.

Two facts dominate the assessment, and they point the same way:

1. **The license is not open-source.** IndexTTS-2 and 2.5 weights *and* "final code" are
   governed by the bespoke **bilibili Model Use License Agreement** — revocable on
   breach, with a downstream flow-down duty, a >100 M MAU / >RMB 100 M revenue carve-out
   requiring separate written authorization, a prohibition on using the model or its
   outputs to improve other AI models, PRC governing law, Shanghai arbitration, and a
   **controlling Chinese text that the shipped English translation gets wrong by 10×**
   (§2.1). IndexTTS-1.0/1.5 were Apache-2.0; the project *relicensed away from* Apache at
   v2. A GGUF conversion is squarely a "Derivative Work" under §1.5(iii).
2. **It is the slowest TTS in the one ggml engine that already ships it.** `audio.cpp`
   measures `index tts2` at **RTF 0.332 on an RTX 5090** long-form — 6.6× slower than
   OmniVoice (0.050) and 1.5× slower than Qwen3-TTS (0.222) in the *same* engine on the
   *same* GPU. No CPU figure is published for it by anyone. Inference needs **~7.1 GB of
   fp32 weights across six sub-models** before the optional Qwen emotion head.

Add a third, quieter one: the upstream pipeline is **numerically chaotic**. Upstream
issue #679 documents that a ~1e-4 difference in the mel filterbank front-end — caused
only by changing `OMP_NUM_THREADS` — flips the sign of Wav2Vec2-BERT `hidden_states[17]`
and produces an audibly different clone. That is a direct problem for
`docs/port-validation.md`'s stage-by-stage tensor comparison.

**Recommendation: do not adopt as the fifth family.** Keep it as a documented,
re-examinable candidate. If the project ever wants it, the honest framing is a
GPU-first, Restricted Model Package family under a *new* ADR — ADR 0018's category was
built for non-commercial grants and its required card language ("not usable in
commercial products") would be factually wrong here.

---

## 2. Variants, licenses, releases

| | IndexTTS 1.0 | IndexTTS-1.5 | IndexTTS-2 | **IndexTTS-2.5** |
|---|---|---|---|---|
| HF repo | `IndexTeam/Index-TTS` | `IndexTeam/IndexTTS-1.5` | `IndexTeam/IndexTTS-2` | `IndexTeam/IndexTTS-2.5` |
| Released | 2025-03-25 | 2025-05-14 | 2025-09-08 | **2026-08-10** |
| HF card license | — | `apache-2.0` | *(no license field)*, `LICENSE.txt` = bilibili | `license: other`, `license_name: bilibili-model-license` |
| Paper | arXiv 2502.05512 | 2502.05512 | 2506.21619 | **2601.03888** |
| Languages | zh, en | zh, en | zh, en | zh, en, ja, es, ar |
| Output SR | — | — | 22.05 kHz | 22.05 kHz |

Release dates from the GitHub README "News" section and HF API `createdAt`.
`IndexTeam/IndexTTS-2.5` `createdAt` = 2026-08-10T07:49:03Z, `lastModified` =
2026-08-12T07:23:44Z, `gated: false`.

### 2.1 The license, in detail

Verified from three places that agree: the repository `LICENSE` file at
`index-tts/index-tts@ee40fa7`, the HF card front-matter of `IndexTeam/IndexTTS-2.5`
(`license: other` / `license_name: bilibili-model-license` / `license_link: LICENSE`),
and `IndexTeam/IndexTTS-2`'s `LICENSE.txt` + `LICENSE_ZH.txt`. `pyproject.toml` declares
`license = "LicenseRef-Bilibili-IndexTTS"`.

Clauses that matter to synthesize.cpp, quoted or closely paraphrased:

- **§1.4 scope** — "Model" means "bilibili indextts2", "including but not limited to
  model weights **and final code**". The license covers the inference code too, not just
  the weights.
- **§1.5(iii) derivative** — includes any model created by "re-training, fine-tuning,
  **quantizing**, LoRA, parameter-efficient fine-tuning, or any other method involving
  incremental weights or merged checkpoints". **A GGUF conversion, quantized or not, is a
  Derivative Work.**
- **§2.2 scale carve-out — and a 10× translation error.** The English `LICENSE` says the
  carve-out triggers above 100 million MAU **or "RMB 1 billion" annual revenue**. The
  governing Chinese `LICENSE_ZH.txt` §2.2 says
  「月活跃用户数超过1亿，或者……上一自然年的年收入超过**1亿人民币**」 — 100 million MAU or
  **RMB 100 million** revenue. Under §9 the Chinese text prevails, so **the operative
  revenue threshold is RMB 100 million and the shipped English translation overstates it
  by a factor of ten.** Above it you "must request a separated license from us, which We
  may grant … in our sole discretion". `audio.cpp`'s documentation repeats the incorrect
  English figure; `Richasy/IndexTTS-2.5-GGUF`'s card carries the correct one. Verified
  against `IndexTeam/IndexTTS-2/LICENSE_ZH.txt` line 15 on 2026-08-21.
- **§3.4(a) flow-down** — you must impose contractual terms on downstream recipients and
  are "responsible for the consequences" if they breach.
- **§3.4(c) no-AI-improvement** — you may not use the model or derivative "to improve any
  AI model, except for the bilibili indextts2 itself, its Derivative Works, or
  non-commercial AI models".
- **§4.1(a) disclaimer duty** — a distributed Derivative Work must carry a specific
  verbatim disclaimer that modifications are not endorsed by the right-holder.
- **§4.2 high-risk prohibition** — medical, autonomous driving, military, critical
  infrastructure, large-scale biometric surveillance, automated credit/employment
  decisions.
- **§5.1 revocation** — the licensor may revoke on breach; you must then delete all
  copies.
- **§5.3 patent/IP retaliation** — any IP proceeding against bilibili terminates all
  grants.
- **§6 governing law** — PRC law; disputes to the Shanghai Arbitration Commission.
- **§8 version updates** — new license versions may be issued; they apply prospectively.
- **§9 language** — "the Chinese-language version shall prevail".

Separately, the repository `DISCLAIMER` (dated 2025-03-17, still on `main`) is an
acceptable-use document that, among other things, forbids synthesizing public figures'
voices and forbids unauthorized commercial use of synthesized voices. Note that its §3.1
still reads literally "本项目以[开源许可证类型]许可证开源" — an unfilled template
placeholder, left over from the Apache-2.0 era.

**Assessment against `docs/scope.md` and ADR 0018.** The weights cannot be a *Published
Model Package*: that term promises weights embeddable in other people's programs, and
these carry a flow-down duty, a revocation right, and a discretionary scale gate. They
*could* be a **Restricted Model Package** under ADR 0018 — the ADR's category covers
"non-commercial **or otherwise redistribution-restricted**" grants, and its deciding
argument (a restriction follows the weights regardless of who runs the converter) applies
unchanged. But ADR 0018's *consequences* section requires the model card to carry "a
prominent statement that the weights are not usable in commercial products", which would
be **false** here: ordinary commercial use is permitted below the thresholds. Adopting
this family therefore needs a new ADR that generalizes the category from "non-commercial"
to "restricted", or a second sub-category. That is a decision, not a detail.

### 2.2 Third-party weights in the inference path

The IndexTTS-2.5 HF repository does **not** contain everything inference needs.
`indextts/utils/model_download.py` pulls three more checkpoints on first run:

| Component | Repo | HF card license | Inference file |
|---|---|---|---|
| Wav2Vec2-BERT 2.0 | `facebook/w2v-bert-2.0` | `mit` | `model.safetensors`, 2,322 MB |
| CAMPPlus speaker embedder | `funasr/campplus` | `apache-2.0` | `campplus_cn_common.bin`, 28 MB |
| BigVGAN v2 22 kHz 80-band | `nvidia/bigvgan_v2_22khz_80band_256x` | `mit` | `bigvgan_generator.pt`, 449 MB |

The semantic codec (`codec.pth`) *is* in the 2.5 repository — unlike IndexTTS-2, which
downloaded MaskGCT's codec. The code file carries an Amphion MIT header
(`indextts/codec/models.py`). Those three aux licenses are all permissive; the
composition problem is the *number* of licenses that must travel with a package, not any
one of them. The existing third-party GGUF ships five separate
`THIRD_PARTY_LICENSES/*.txt` files plus a `NOTICE` to satisfy this.

### 2.3 Inference footprint

Measured from HF blob sizes (fp32 checkpoints):

| Component | Size |
|---|---:|
| `gpt.pth` (T2S) | 3,259.6 MB |
| `codec.pth` (semantic codec) | 607.3 MB |
| `s2mel.pth` (S2M + length regulator) | 414.9 MB |
| `w2v-bert-2.0/model.safetensors` | 2,322.1 MB |
| `bigvgan_generator.pt` | 449.2 MB |
| `campplus_cn_common.bin` | 28.0 MB |
| **Subtotal (no emotion-text head)** | **≈ 7.08 GB** |
| `qwen0.6bemo4-merge/model.safetensors` (optional) | 1,192.1 MB |
| **Total** | **≈ 8.27 GB** |

Cross-check: `Richasy/IndexTTS-2.5-GGUF` publishes a single original-dtype GGUF of
**7,885,093,568 bytes (7.89 GB) with 3,790 tensors** across ten namespaces, including the
Qwen emotion head. That is consistent with the sum above once container overhead and
dtype normalization are accounted for. The model card states "roughly 6 GB of VRAM for
inference" at bf16.

---

## 3. Architecture decomposition

Read from `indextts/infer_v2_5.py`, the modules it imports, and
`IndexTeam/IndexTTS-2.5/config.yaml`. Ten weight namespaces, matching the ten prefixes
`audio.cpp`'s `model_specs/index_tts2.json` declares (`gpt`, `s2mel`, `speaker_matrix`,
`emotion_matrix`, `wav2vec2bert_stats`, `wav2vec2bert`, `semantic_codec`, `campplus`,
`bigvgan`, `qwen_emotion`).

### 3.1 Text frontend and tokenizer

- `TextNormalizer` (`indextts/utils/front.py`): punctuation folding, then
  WeTextProcessing (Linux) / `wetext` (Win/Mac) OpenFST grammars for zh/en, plus `cn2an`,
  `jieba`, `g2p-en`.
- `nemo_tn.normalize_text` for `ja`/`es`.
- `JapaneseG2PProcessor` (`indextts/utils/ja_g2p.py`): `fugashi` + `unidic-lite`
  (MeCab) segmentation and kana readings; falls back to `mecab-python3`.
- Tokenizer: **tiktoken** BPE over `multilingual_zh_ja_yue_char_del.tiktoken`
  (0.91 MB, 60,509 text tokens), reached through `whisper.tokenizer.Tokenizer`.
  A language prefix token `<|zh|>` / `<|en|>` / … is prepended per segment.
- Inline pronunciation overrides: `<文字|发音>` is rewritten into
  `<|SPECIAL_TOKEN_1|>`-wrapped (English/CMU) or `<|SPECIAL_TOKEN_2|>`-wrapped
  (Chinese/pinyin) spans by `apply_pronunciation_annotations`.
- IndexTTS-2 (not 2.5) used a SentencePiece `bpe.model` instead.

### 3.2 Speaker and emotion conditioning

- Reference audio is loaded, **cut to 15 s max**, and resampled to both 22.05 kHz (for
  the reference mel) and 16 kHz (for w2v-BERT and CAMPPlus).
- `SeamlessM4TFeatureExtractor` → `Wav2Vec2BertModel` (24 layers, 1024-dim) produces
  `spk_cond_emb`; normalized by stored `mean`/`sqrt(var)` from
  `wav2vec2bert_stats.pt`.
- `torchaudio.compliance.kaldi.fbank` (80 mel bins, `dither=0`, mean-subtracted) →
  `CAMPPlus(feat_dim=80, embedding_size=192)` → a 192-d global style vector.
- Emotion has three mutually exclusive paths:
  1. **emotion reference audio** → same w2v-BERT path → `emo_conditioning_encoder`
     (Conformer, 4 blocks) → `emo_perceiver_encoder` (Perceiver resampler) →
     `emovec_layer` → `emo_layer`;
  2. **8-float emotion vector** `[happy, angry, sad, afraid, disgusted, melancholic,
     surprised, calm]`, mixed against `feat2.pt` (`emo_matrix`) rows selected per
     emotion group by cosine similarity to the speaker style against `feat1.pt`
     (`spk_matrix`); group sizes `emo_num: [3, 17, 2, 8, 4, 5, 10, 24]` (73 rows);
  3. **emotion text** → `QwenEmotion` (Qwen3-0.6B fine-tune) → 8-float vector.
- `merge_emovec` blends: `base + alpha * (emo - base)`.

### 3.3 T2S — `UnifiedVoice` (`indextts/gpt/model_v2.py`)

GPT-2 decoder: `model_dim 1280`, `layers 24`, `heads 20`, learned positional
embeddings, `number_text_tokens 60509`, `number_mel_codes 8194`
(`start_mel_token 8192`, `stop_mel_token 8193`), `max_mel_tokens 1815`,
`max_text_tokens 600`. Conditioning is `conformer_perceiver`: a `ConformerEncoder`
(`input_size 1024`, `output_size 512`, 6 blocks, `conv2d2` subsampling) feeding a
`PerceiverResampler`. In 2.5, `spk_cond_mode="campplus"`, so the *speaker* latent is the
192-d CAMPPlus vector through `spk_emb_proj` rather than the perceiver path; the
conformer/perceiver stack still runs for the emotion latent. `~0.8 B` parameters per the
model card.

Note the vendored HuggingFace code: `indextts/gpt/transformers_gpt2.py` (1,878 lines),
`transformers_modeling_utils.py` (5,525), `transformers_generation_utils.py` (4,747),
`transformers_beam_search.py` (1,013) — Apache-2.0 headers, copied rather than imported.
The reference decode path is therefore HF `generate()` semantics frozen at a specific
version.

`indextts/gpt/model_v2_5.py` (797 lines) exists but **nothing imports it** —
`infer_v2_5.py` imports `UnifiedVoice` from `model_v2`. Dead code; do not use it as a
spec.

### 3.4 Semantic codec — `EnhancedCodec` (`indextts/codec/models.py`)

Amphion/MaskGCT-derived (MIT header). `codebook_size 8192`, `codebook_dim 8`,
`hidden_size 1024`, single quantizer, `downsample_scale 2`. Encoder and decoder are each
a `VocosBackbone` (ConvNeXt, `dim 384`, `intermediate_dim 2048`, 12 layers) plus a linear
back to 1024. `decode()` is: `ResidualVQ.vq2emb(codes)` → decoder → `F.interpolate(…,
scale_factor=2, mode="nearest")` → `up` Conv1d. So the GPT emits **25 Hz** codes and the
codec restores the **50 Hz** w2v-BERT feature rate — exactly the "50 Hz → 25 Hz semantic
codec compression" the 2.5 paper claims as improvement (1).

### 3.5 Length regulator — `InterpolateRegulator`

`channels 512`, `in_channels 1024`, `is_discrete: false`, `sampling_ratios [1,1,1,1]` →
four `Conv1d(512,512,3) + GroupNorm + Mish` blocks then a `Conv1d(512,512,1)`. It
interpolates the 50 Hz semantic features to the target mel length
`round(S.shape[1] * 1.72 * duration_factor)`; 50 × 1.72 ≈ 86 ≈ 22050/256, the mel frame
rate. `duration_factor` (0.5–2.0) enters here and nowhere else — speed control is a pure
resampling of the semantic track, not a re-decode.

### 3.6 S2M — flow-matching DiT (`indextts/s2mel/modules/`)

`CFM.estimator = DiT(args)` — the config's `dit_type` is `"DiT"` and
`flow_matching.py` raises `NotImplementedError` for anything else. `hidden_dim 512`,
`depth 13`, `num_heads 8`, `in_channels 80`, `uvit_skip_connection: true`,
`long_skip_connection: true`, `style_condition: true`, `final_layer_type: 'wavenet'`,
`is_causal: false`. The transformer core is a `gpt_fast`-style stack (`setup_caches`,
`ModelArgs`, RMSNorm/RoPE lineage) with a `TimestepEmbedder`, weight-normed
`x_embedder`, and a WaveNet tail (`hidden_dim 512`, 8 layers, `kernel_size 5`,
`dilation_rate 1`) plus a `FinalLayer`.

`solve_euler` runs **`diffusion_steps = 25`** fixed Euler steps with
**`inference_cfg_rate = 0.7`**, and CFG is implemented by *stacking* conditional and null
inputs into one batch — so **50 DiT evaluations per segment**, each over the full
`prompt_mel + target_mel` length. The reference mel is written into the prompt region of
`x` each step. This is the dominant cost of the non-AR half of the pipeline.

**Discrepancy worth flagging:** arXiv 2601.03888's abstract claims improvement (2) is
replacing "the U-DiT-based backbone of the S2M module with a more efficient
Zipformer-based modeling architecture". The word "zipformer" does not appear anywhere in
the released code, and the released `config.yaml` selects `dit_type: "DiT"`. Either the
paper describes a configuration that was not open-released, or the naming diverged. Do
not plan against the paper's S2M description.

### 3.7 Vocoder — BigVGAN v2

`nvidia/bigvgan_v2_22khz_80band_256x`, loaded via
`indextts/s2mel/modules/bigvgan/bigvgan.py`, `remove_weight_norm()` applied at load.
80-band mel, 256× hop, 22.05 kHz. Optional fused anti-alias-activation CUDA kernel.

### 3.8 Emotion-text head — `QwenEmotion`

Qwen3-0.6B fine-tune (`qwen0.6bemo4-merge/`, 1.19 GB) loaded through
`modelscope.AutoModelForCausalLM`. Only constructed when `use_qwen_emo=True`; passing
`use_emo_text=True` without it raises at inference. It maps free text to the 8-float
emotion dictionary.

### 3.9 Glue

Reference-derived state (`spk_cond_emb`, style, `prompt_condition`, `ref_mel`,
`emo_cond_emb`) is cached keyed on the prompt path. Long text is split by tokenizer
budget (`max_text_tokens_per_segment=120`), each segment synthesized independently, and
the waveforms concatenated with a 200 ms silence — so prosody does not cross a segment
boundary (stated as a limitation on the model card). `remove_long_silence` post-processes
runs of `silent_token=52` beyond `max_consecutive=30`.

---

## 4. Inference loop (`infer_v2_5.py:infer_generator`)

1. Resolve emotion source (text → Qwen → vector; vector scaled by `emo_alpha`; else
   emotion reference audio; else the speaker clip itself with `emo_alpha` forced to 1.0).
2. Prompt encode (cached): load + cut to 15 s → resample 22.05 k and 16 k →
   `SeamlessM4TFeatureExtractor` → `Wav2Vec2BertModel` → normalize → `spk_cond_emb`;
   kaldi fbank → `CAMPPlus` → `style`; mel of the 22.05 k clip → `ref_mel`;
   `length_regulator(spk_cond_emb, ylens=len(ref_mel))` → `prompt_condition`.
3. Text: punctuation folding → WeText (zh/en) or NeMo (ja/es) normalization → case
   folding (lower for ja/zh/en, **upper** for es) → pronunciation annotations → fugashi
   for ja → segment by token budget → tiktoken encode with a `<|lang|>` prefix and a
   trailing `1` pad.
4. Per segment: `gpt.merge_emovec(...)` → `gpt.inference_speech(...)` →
   HF `generate()` with `do_sample=True, top_p=0.8, top_k=30, temperature=0.8,
   num_beams=3, repetition_penalty=10.0, length_penalty=0.0, max_mel_tokens=1500`.
   Codes truncated at `stop_mel_token`.
5. `semantic_codec.decode(codes)` → `length_regulator(..., ylens = S*1.72*duration_factor)`
   → concatenate `[prompt_condition, cond]` → `cfm.inference(..., 25 steps, cfg 0.7)`
   → strip the prompt-length prefix → BigVGAN → clamp to ±32767.
6. Concatenate segments with 200 ms silence; write 22.05 kHz PCM16.

The default **`num_beams=3` with `do_sample=True`** is HF "beam sample" — beam search
*and* stochastic sampling together, with a repetition penalty of 10.0. That is a
demanding decode contract to reproduce: `audio.cpp` exposes both `num_beams` and the
sampling knobs as request options, and its GGUF card records an unresolved Japanese
repetition-penalty parity question (PR 210).

---

## 5. Operator surface (and GGML mapping)

| Stage | Operators | Notes for a GGML port |
|---|---|---|
| Feature extraction | STFT, mel filterbank, kaldi fbank, CMVN | Already have STFT/mel from Kokoro; kaldi fbank is new. **This is the numerically fragile stage — see §8.** |
| Wav2Vec2-BERT 2.0 | conv2d subsampling, 24 conformer blocks: rel-pos MHA, depthwise conv module, SwiGLU/GLU, LayerNorm | Largest single new block; 580 M params. `ggml_conv_2d`, standard attention, depthwise via `ggml_conv_1d` with groups. |
| CAMPPlus | TDNN / dense-connected 1-D convs, BN, statistics pooling | Small, straightforward. |
| Conformer + Perceiver conditioner | conv2d2 subsampling, rel-pos MHA, cross-attention to learned latents | Cross-attention with fixed latents is new surface. |
| GPT-2 T2S | learned pos emb, 24× (LN, MHA with KV cache, GELU MLP), tied head | Well-trodden; KV-cache AR loop already exists for Qwen3-TTS. |
| Decode chain | top-k, top-p, temperature, repetition penalty, **beam search (3 beams) + sampling**, length penalty | Beam-sample is the one genuinely new runtime control. Beam reordering of the KV cache is required (`_reorder_cache`). |
| Semantic codec | ResidualVQ embedding lookup, ConvNeXt (dw conv7, LN, GELU, layer scale), `interpolate` nearest ×2, Conv1d | ConvNeXt/Vocos already exercised conceptually by OmniVoice-class work; nearest-neighbour upsample is trivial. |
| Length regulator | Conv1d, GroupNorm, **Mish**, linear interpolation to arbitrary length | GroupNorm and Mish (`x*tanh(softplus(x))`) both need checking against ggml's op set. |
| S2M DiT | 13× transformer with RoPE + KV cache, adaLN-style conditioning, U-ViT skip connections, weight-normed linear, timestep sinusoidal embedding | Weight-norm should be folded at conversion. U-ViT skips are graph plumbing, not new kernels. |
| WaveNet tail | dilated Conv1d, gated tanh⊙sigmoid, 1×1 skip/res convs | Familiar from VITS. |
| CFM sampler | 25 Euler steps, batched CFG (2× batch), prompt-region masking | Deterministic given the initial noise; noise must be injectable for validation. |
| BigVGAN v2 | ConvTranspose1d, AMP blocks, **Snake/SnakeBeta**, anti-aliased up/down sampling (`Activation1d`) | Snake already exists from Kokoro; the anti-alias low-pass resampling around each activation is the new part. |
| Qwen emotion (optional) | Qwen3-0.6B decode | Already a solved shape for this project. |

No operator here looks like a *missing kernel* problem. The problem is **breadth**: six
graphs and three distinct conditioning encoders, versus one or two for every family the
project has shipped.

---

## 6. Text frontend

This is the second-largest porting cost after sheer model count, and the one place where
an existing ggml port has already conceded defeat in writing.

Upstream requires, at minimum:

- **WeTextProcessing / wetext** — OpenFST-compiled zh/en normalization grammars.
- **`cn2an`**, **`jieba`**, **`g2p-en`** (NLTK + CMUdict) for Chinese numerals,
  segmentation, and English G2P.
- **NeMo text normalization** for `ja` and `es`.
- **`fugashi` + `unidic-lite`** (MeCab) for Japanese segmentation and kana readings.
- **tiktoken** BPE over a 60,509-entry vocabulary file.

`audio.cpp`'s own documentation says it "reimplements the official IndexTTS text
front-end … as lightweight C++ rules instead of running the official OpenFst grammars",
guards it with a 50+ case golden-corpus parity test, and states plainly that "a
rule-based port cannot be exhaustive: for unusual inputs (letter-attached digits, rare
date/unit formats, emails, URLs) the normalized text can still differ from the official
pipeline". For 2.5 it adds that **the Spanish NeMo normalizer is not ported**, and that
**Japanese fugashi segmentation is not ported**, so "ja text is tokenized unsegmented …
intelligible, but not token-identical to the official pipeline". The downstream GGUF card
goes further: its consumer "refuses numeric notation instead of producing a known-wrong
reading".

For synthesize.cpp this collides directly with the Text Frontend Provider model in
`docs/text-frontends.md`: five languages, four different normalization stacks, one of
which (Japanese) needs a dictionary-based morphological analyzer. The tiktoken BPE itself
is easy — the project already built a byte-level BPE for Qwen3-TTS. Everything in front
of it is not.

Licensing of the frontend dependencies was **not** verified in this pass (see §13); the
MeCab/UniDic chain in particular has historically been multi-licensed and needs checking
before anyone plans a bundled Japanese path.

---

## 7. Voice cloning / prompt conditioning / voice profiles

- **Reference audio is mandatory.** There is no preset-voice path and no `voice_id`
  equivalent; `audio.cpp` records "Built-in voices: Not exposed" for both variants. The
  family would land entirely on synthesize.cpp's Reference Audio branch.
- **No reference transcript is consumed** — cloning is from audio alone.
- The prompt is **hard-cut to 15 seconds** (`_load_and_cut_audio(..., 15, ...)`).
- A Voice Profile for this family would need to serialize: the w2v-BERT
  `spk_cond_emb` sequence, the 192-d CAMPPlus style vector, the `prompt_condition`
  (length-regulated to the reference mel length), and the reference mel itself — because
  `solve_euler` writes `ref_mel` into the prompt region at every step. That is a
  *sequence*-valued profile whose size scales with reference duration, not a fixed-size
  vector like Kokoro's style vector or Qwen3-TTS's x-vector. Worth costing against
  ADR 0008 before committing.
- Emotion is genuinely disentangled from timbre and reachable three ways (reference
  audio, 8-float vector, text) — this is the family's most distinctive capability and
  the strongest argument in its favour.

---

## 8. Randomness / determinism — the port-validation problem

Three separate sources of nondeterminism, in increasing order of severity:

1. **Sampling.** `do_sample=True` with `num_beams=3`, top-k 30, top-p 0.8, temperature
   0.8, repetition penalty 10.0. This is the same class of problem the project already
   solved for Qwen3-TTS: `docs/port-validation.md`'s stochastic-replay rule lets the
   oracle capture its sampled code sequence and replay it into both graphs. Beam search
   adds KV-cache reordering to what must be replayed, but the rule still applies.
2. **CFM noise.** 25 Euler steps from an initial Gaussian. Deterministic once the noise
   tensor is injected; the same injection technique applies.
3. **Chaotic amplification through Wav2Vec2-BERT — the real problem.** Upstream issue
   #679 (open, 2026-04-09) reports the same code, weights, and GPU producing "正常" vs
   "含糊不清" clones on two machines differing only in vCPU count. The reporter isolated
   it: `librosa.load` output was byte-identical; the `Wav2Vec2BertModel` forward on fixed
   random input was identical; the divergence was in
   `SeamlessM4TFeatureExtractor`, where `input_features` summed to `79.9994` vs
   `79.9993` — and after 24 conformer layers `hidden_states[17]` summed to **+5847 vs
   −3494**. Sign-inverted. Setting `OMP_NUM_THREADS=1` made the feature extractor
   reproducible; setting it to 8 broke the previously-good machine. The reporter
   attributes it to threaded floating-point reduction order in the FFT / mel filterbank.

Consequence for this project: a stage-by-stage tensor comparison against the PyTorch
oracle — the core of `docs/port-validation.md` — would be comparing against an oracle
that **does not reproduce itself across thread counts**. Any tolerance the project sets
at the `spk_cond_emb` boundary is meaningless unless the oracle is pinned to
single-threaded feature extraction, and even then the port's own mel filterbank must
match PyTorch's summation order to ~1e-5 relative, or every downstream stage tolerance is
unusable. This is a solvable engineering problem (pin `OMP_NUM_THREADS=1` in the oracle
runner, validate the feature extractor against a fixed-order reference implementation
first, and treat the w2v-BERT output as the first *contract* boundary rather than an
intermediate) — but it is real, undocumented upstream, and it should be priced into any
intake estimate rather than discovered in stage 3.

---

## 9. Reference implementation as validation oracle

**Good:** plain PyTorch, one repository, all stages reachable from `infer_v2_5.py`, a
`--device cpu` path, per-stage timing already instrumented (`gpt_gen_time`,
`s2mel_time`, `bigvgan_time`), `uv`-locked environment (`uv.lock` present),
Python 3.10–3.11.

**Bad:**
- No `seed` parameter anywhere in `infer()`/`infer_generator()`; `use_random` calls
  `random.randint` off the global RNG. Deterministic replay needs oracle-side patching.
- The `pyproject.toml` inference closure is heavy and pinned oddly: `keras==2.9.0`,
  `opencv-python==4.9.0.80`, `tensorboard==2.20.0`, `openai-whisper`, `modelscope`,
  `descript-audiotools`, plus `torch==2.8.*`. `modelscope` is imported *at module scope*
  in `infer_v2_5.py`.
- Vendored HF transformers (≈13 k lines across four files) means the decode contract is
  a frozen fork, not a released library version.
- `indextts/gpt/model_v2_5.py` is unreferenced dead code.
- The repo moves fast: `main` at `ee40fa7` is dated 2026-08-18, eight days after the 2.5
  release. Pin a commit, not a branch.
- Weights ship as `.pth` torch pickles (`gpt.pth`, `codec.pth`, `s2mel.pth`, `feat1.pt`,
  `feat2.pt`), not safetensors — relevant to the project's untrusted-parser stance. The
  `audio.cpp` converter unwraps container keys and restages to safetensors before GGUF,
  which is the pattern to copy.
- `config.yaml` in the official 2.5 snapshot ships `version: 2.0` in some revisions;
  `audio.cpp`'s converter normalizes it to `"2.5"` explicitly. The version field is how
  both engines select the variant.

---

## 10. Performance evidence (CPU feasibility)

**No published CPU measurement for IndexTTS-2/2.5 was found in any primary source.**
That absence is itself the finding: `audio.cpp` publishes a CPU row for `supertonic` in
its long-form table and for no other TTS model, `index_tts2` included.

What *is* published, all from `0xShug0/audio.cpp` (RTX 5090, Ubuntu, CUDA, long-form
6,026-character passage):

| model | audio len (s) | wall (s) | RTF |
|---|---:|---:|---:|
| supertonic | 379.32 | 2.02 | 0.005 |
| pocket tts | 353.12 | 7.30 | 0.021 |
| **omnivoice** | 357.00 | 17.77 | **0.050** |
| moss_tts_nano | 391.20 | 43.16 | 0.110 |
| vevo2 | 457.68 | 52.47 | 0.115 |
| chatterbox | 391.24 | 58.57 | 0.150 |
| miotts | 399.16 | 66.59 | 0.167 |
| moss_tts_local | 375.44 | 73.84 | 0.197 |
| **qwen3 tts** | 327.60 | 72.65 | **0.222** |
| voxcpm2 | 315.84 | 72.70 | 0.230 |
| **index tts2** | 422.12 | 139.95 | **0.332** |
| supertonic (CPU) | 379.40 | 61.40 | 0.162 |

`index tts2` is the **slowest** entry in that table. Against the two families
synthesize.cpp already ships that appear in it, on the same GPU and engine: **6.6× slower
than OmniVoice, 1.5× slower than Qwen3-TTS.**

The `Richasy/IndexTTS-2.5-GGUF` card adds independently-run CUDA numbers for 2.5 through
`audio.cpp` 0.6: 2.71 s of Chinese audio in 1.89 s (RTF 0.699) cold, RTF 0.452 warm, and
RTF 0.198 on a 6,300-character English passage — with independent Whisper large-v3-turbo
transcription giving exact zh/en/es and 0.947/0.909 normalized-character similarity for
ar/ja.

Extrapolating the shape rather than the number: the compute is roughly (a) a 0.8 B AR
decode with 3 beams, (b) 580 M of w2v-BERT over the prompt once, and (c) **50 DiT
evaluations of a 13-layer, 512-dim transformer over the full prompt+target mel length**
per segment. On the project's own precedent, an OmniVoice-class model at CUDA RTF 0.050
delivered CPU RTF 0.17 (memory record, corrected for the `-O2` dev-preset issue). A model
6.6× heavier on the same GPU would not plausibly land under RTF 1.0 on CPU at reference
settings. **Treat CPU real-time as unproven and unlikely; treat CPU offline synthesis as
the realistic claim**, and note that criterion 3 of `docs/model-family-selection.md`
("at least one model variant is practical for CPU inference") has no smaller variant to
fall back on — there is exactly one 2.5 checkpoint.

Reducible knobs, if anyone wants to test the ceiling: `diffusion_steps` 25 → 8–10,
`num_beams` 3 → 1, `inference_cfg_rate` 0.7 → 0 (halves DiT batch). None has a published
quality-vs-setting curve.

---

## 11. Prior art (C++/GGML/other ports)

| Project | Scope | Approach / status |
|---|---|---|
| **audio.cpp** (`0xShug0`) | `index_tts2` family covering both IndexTTS-2 and 2.5 | **The one that matters.** Apache-2.0 (ShugoAI LLC), C++/ggml, 1,888 stars, created 2026-06-23, release 0.6.1 on 2026-08-19 — IndexTTS-2.5 support shipped within ~3 days of the upstream release. CUDA/HIP/Vulkan/Metal/CPU. Ships `index_tts2_5_{q8_0,f16,orig}` GGUF packages, Q8_0 as default. Full option surface incl. `duration_factor`, `emotion_vector`, `num_beams`, per-graph arena sizing, speaker/emotion caches, and per-weight-type overrides. Converter: `tools/convert_index_tts2_5.py` + `audiocpp_gguf`. Documents its own frontend gaps honestly (§6). |
| **`Richasy/IndexTTS-2.5-GGUF`** | Original-dtype single-file GGUF | 7.89 GB, 3,790 tensors, 10 namespaces, embedded sidecars. Reproducible-conversion section pins model/converter/aux revisions and publishes the artifact SHA-256. Carries `NOTICE`, `THIRD_PARTY_NOTICES.md`, five `THIRD_PARTY_LICENSES/*`, and both `LICENSE`/`LICENSE_ZH.txt`. **A working example of what license-compliant redistribution of these weights looks like.** |
| **`vra/indextts-onnx`** | IndexTTS2 | ONNX Runtime, "no PyTorch at inference time"; 0 stars, last touched 2026-06-21. |
| **`ThreadAbort/IndexTTS-Rust`** | IndexTTS | ONNX-tagged HF repo, Rust; 7 likes. |
| **MLX** | IndexTTS, 1.5, 2.5 | `mlx-community/IndexTTS`, `mlx-community/IndexTTS-1.5`, `mlx-community/index-tts2-mlx`, `vanch007/mlx-indextts2-2.5-8bit`. Apple-only. |
| **vLLM** | IndexTTS-2 / 2.5 | Official recipe at `recipes.vllm.ai/IndexTeam/IndexTTS-2.5`, plus many community `*-vLLM` repos. The upstream-blessed production path is GPU serving. |

**Takeaway, and it cuts differently from VoxCPM's.** For VoxCPM the prior art was
"conversion recipes exist, none is a reason not to port". Here, one Apache-2.0 ggml
engine with 1,888 stars already ships this family with a *broader* option surface than
synthesize.cpp exposes for any family, on five backends, within days of upstream release,
and publishes GGUF packages for it. synthesize.cpp's differentiators (stable C ABI,
per-stage port validation, Voice Profiles, quantization matrix, published model cards)
remain real and orthogonal — but the marginal capability this family adds to *users* is
smaller than for a family nobody has done, and the marginal cost is the largest the
project has faced.

---

## 12. Assessment against `docs/model-family-selection.md`

1. **Representative architecture + new operator surface — yes, but mostly borrowed.**
   The genuinely new surface is: a flow-matching DiT with U-ViT skips and a WaveNet
   tail, CFG-batched Euler sampling, a Conformer+Perceiver conditioner, Wav2Vec2-BERT,
   CAMPPlus, and beam-sample decoding. Everything else (GPT-2 AR + KV cache, ConvNeXt,
   RVQ, Snake, dilated conv) the project has or nearly has. Note that a flow-matching
   DiT decoder is *also* what the VoxCPM candidate would bring, at a fraction of the
   model count — the two candidates overlap on the most valuable new surface.
2. **Stable, inspectable reference — qualified yes.** Plain PyTorch, all stages
   reachable, but fast-moving, vendored-HF, pickle weights, no seed control, and one
   dead module. Pin `ee40fa7` or later and patch for determinism.
3. **A variant practical for CPU inference — no evidence, and no fallback variant.**
   The weakest criterion by a distance. RTF 0.332 on an RTX 5090 in the best available
   ggml engine, ~7.1 GB of fp32 weights, six graphs, and a single released 2.5
   checkpoint with no smaller sibling. **This is the criterion that should decide the
   intake.**
4. **Measurable, tractable operator gaps — yes.** No missing kernels identified; the
   anti-aliased Snake resampling in BigVGAN v2 and GroupNorm/Mish in the length
   regulator are the items to confirm against ggml's op set first.
5. **Usable licenses — no, not for a Published Model Package; conditionally yes for a
   Restricted one.** The bilibili Model Use License is bespoke, revocable, flow-down,
   scale-gated, PRC-governed, and Chinese-controlling — and the English translation it
   ships understates the commercial trigger by 10×, which is exactly the kind of fact a
   redistributor inherits responsibility for. Three permissive third-party weight sets
   ride along. Requires a new or amended ADR (§2.1).
6. **Reproducible conversion + incremental quantization — yes, demonstrated.**
   `audio.cpp` publishes a converter, three precision tiers, and Q8_0 as its *default*;
   a third party reproduced the original-dtype conversion tensor-for-tensor (3,790
   equal, zero differences) and published the SHA-256. Quantization sensitivity of the
   DiT and BigVGAN is unmeasured.
7. **End-to-end testable pipeline — yes, with the §8 caveat.** Clean stage seams; the
   feature-extractor boundary is chaotic and needs to be handled as a contract, not an
   intermediate.

### Key risks, ranked

1. **License.** Not open-source, revocable, flow-down, discretionary scale gate,
   controlling text in a language most consumers of the package cannot read, and a
   verified 10× error in the English translation of the commercial trigger (§2.2).
   Publishing anything requires a new ADR and jiangzhuo's per-act confirmation.
2. **Cost vs. every prior family.** Ten weight namespaces, six inference graphs, three
   conditioning encoders, ~7.1 GB fp32. Qwen3-TTS — the largest family so far — took
   three staged variants and seven plans.
3. **CPU practicality unproven** and no smaller variant exists.
4. **Oracle chaos** through the feature extractor (§8) — new validation machinery, not
   just new tolerances.
5. **Text frontend breadth**: five languages, four normalization stacks, one requiring a
   Japanese morphological dictionary. The best existing ggml port openly ships an
   approximation.
6. **Paper/code divergence** (Zipformer vs. DiT) — the published description of the S2M
   module does not match the released weights.
7. **Quality regression reports.** The GGUF card notes "the official IndexTTS tracker
   contains reports of weaker timbre, emotion and prosody in 2.5 than in 2.0". Not
   independently verified here; would need a listening pass before any publication
   claim, per the project's standing practice.

### If it were adopted anyway

Reference variant: **IndexTTS-2.5** (`IndexTeam/IndexTTS-2.5`, `version: 2.5`), not
IndexTTS-2 — 2.5 is the multilingual, faster, actively-maintained one and shares the
same family code path. Stage it as: (1) reference-audio cloning, zh/en only, greedy
decode, CFG-batched CFM; (2) beam-sample decode parity and the remaining three
languages; (3) emotion vector and emotion reference audio; (4) emotion-text via the Qwen
head; (5) quantization and backends. Expect the Voice Profile design (§7) to need
revisiting because the profile is sequence-valued.

---

## 13. Unknowns / could not verify

- Whether the English translation is wrong anywhere *else*. §2.2's revenue threshold was
  checked line-by-line against `LICENSE_ZH.txt` and is wrong by 10×; the rest of the
  English text was read but **not** diffed against the Chinese. Since §9 makes the
  Chinese controlling, any decision resting on a specific English clause needs the
  Chinese text read first.
- Per-module parameter counts. Only the model card's "~0.8 B (GPT backbone)" is
  published; the table in §2.3 is file sizes, not parameter counts. `codec.pth` at 607 MB
  is larger than my rough estimate of the `EnhancedCodec` module and may carry extra
  container keys — not verified.
- Licenses of the text-frontend dependency closure (`WeTextProcessing`/`wetext`,
  `unidic-lite`, `fugashi`/MeCab, `g2p-en`/CMUdict, `cn2an`) — **not checked in this
  pass** and material to §6.
- Whether `audio.cpp`'s `index_tts2` path actually runs on its CPU backend at all, and
  at what RTF. Its README says CPU/Vulkan/Metal/HIP are "intended for portability and
  testing … performance and model coverage may be lower".
- arXiv 2601.03888 was read via the abstract page only. The Zipformer/DiT discrepancy in
  §3.6 is abstract-vs-code; the paper body may explain it.
- The 2.5 quality-regression reports are quoted from a third-party GGUF card, not read
  in the upstream tracker.
- ModelScope mirrors assumed identical to the HF repositories; not verified.
- I did not run the model. Every performance number here is someone else's measurement,
  and the CUDA figures come from a single engine on a single GPU.
- `IndexTeam/IndexTTS-2`'s HF card carries **no** license field even though the repo
  contains `LICENSE.txt`/`LICENSE_ZH.txt`. The 2.5 card does declare it. Per the
  project's standing rule, the 2.5 card is the license source used here.

---

## 14. Primary sources

- Cloned `github.com/index-tts/index-tts` @ `ee40fa7` (2026-08-18) — `LICENSE`,
  `DISCLAIMER`, `README.md`, `pyproject.toml`, `indextts/infer_v2_5.py`,
  `indextts/gpt/model_v2.py`, `indextts/codec/models.py`,
  `indextts/codec/kmeans/vocos.py`, `indextts/s2mel/modules/{commons,flow_matching,
  diffusion_transformer,length_regulator}.py`, `indextts/utils/{front,tokenizer,ja_g2p,
  model_download}.py`
- Hugging Face API + raw files: `IndexTeam/IndexTTS-2.5` (card front-matter,
  `config.yaml`, blob sizes), `IndexTeam/IndexTTS-2` (`LICENSE.txt`, file list),
  `IndexTeam/IndexTTS-1.5`, `facebook/w2v-bert-2.0`, `funasr/campplus`,
  `nvidia/bigvgan_v2_22khz_80band_256x`, `Richasy/IndexTTS-2.5-GGUF`
  (`README.md`, `NOTICE`, blob list), `huggingface.co/api/models?search=indextts`
- arXiv 2601.03888 "IndexTTS 2.5 Technical Report" (v1 2026-01-07, v5 2026-08-11) —
  abstract page; also referenced: 2506.21619 (IndexTTS2), 2502.05512 (IndexTTS 1.x)
- `github.com/0xShug0/audio.cpp` via GitHub API — `LICENSE`, `README.md`,
  `docs/models/index_tts.md`, `docs/tts.md`, `model_specs/index_tts2.json`, releases list
- `IndexTeam/IndexTTS-2/LICENSE_ZH.txt` — the governing Chinese licence text
- `github.com/index-tts/index-tts` issue #679 (open, 2026-04-09) — feature-extractor
  thread-count nondeterminism
- `github.com/vra/indextts-onnx` (repo metadata via GitHub search)
