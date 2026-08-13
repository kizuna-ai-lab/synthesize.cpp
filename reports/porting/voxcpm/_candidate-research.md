# VoxCPM / VoxCPM2 — Candidate-Family Research for synthesize.cpp

Date: 2026-08-13
Author: research agent (primary-source only)
Sources: cloned `OpenBMB/VoxCPM` @ `ee8161e` (2026-08-12, main), Hugging Face model repos (`openbmb/VoxCPM-0.5B`, `openbmb/VoxCPM1.5`, `openbmb/VoxCPM2`), arXiv 2509.24650 and 2606.06928, official readthedocs, plus GitHub repos of existing ports (read-only). No outward interaction with any repository was performed.

This is pre-intake candidate research, not an intake record; it precedes any
fifth-family decision in `docs/model-family-selection.md`.

---

## 1. TL;DR

VoxCPM is a **tokenizer-free, diffusion-autoregressive TTS family** (paradigm credited to DiTAR in the README acknowledgments): a MiniCPM-4-style decoder LM autoregresses over *continuous* audio-VAE latents in patches, and a small **flow-matching DiT ("LocDiT")** diffuses each next patch conditioned on the LM state (10 Euler steps, CFG 2.0). There are three variants, all **Apache-2.0 for both code and weights**: VoxCPM-0.5B (2025-09, 16 kHz, zh/en), VoxCPM1.5 (2025-12, 44.1 kHz, zh/en), and **VoxCPM2 (2026-04, 2B backbone, 30 languages, 16 kHz-in → 48 kHz-out asymmetric AudioVAE V2, voice design + controllable cloning)**. The repo `main` branch supports all three through one code base (`architecture` switch in `config.json`). Substantial GGML prior art exists (three independent GGUF conversions, one semi-official llama.cpp fork with VoxCPM2 support). The dominant risk for synthesize.cpp: **no existing port reaches real-time on CPU** — the per-patch diffusion inner loop (≈ 9 effective steps × 2-batch CFG) dominates compute; community CPU RTF is ~3.6 for the *smallest* variant.

---

## 2. Variants, licenses, releases

| | VoxCPM-0.5B | VoxCPM1.5 | VoxCPM2 |
|---|---|---|---|
| HF repo | openbmb/VoxCPM-0.5B | openbmb/VoxCPM1.5 | openbmb/VoxCPM2 |
| HF `createdAt` | 2025-09-16 | 2025-12-05 | 2026-04-03 |
| **Weight license (model card front-matter)** | `license: apache-2.0` | `license: apache-2.0` | `license: apache-2.0` |
| License wording on card | (front-matter only) | (front-matter only) | "📜 **Fully Open-Source & Commercial-Ready** — Apache-2.0 license, free for commercial use" |
| Main weights file | `pytorch_model.bin` 1304.7 MB (bf16 → ≈0.65 B params) | `model.safetensors` 1603.5 MB (bf16 → ≈0.80 B) | `model.safetensors` 4580.1 MB (bf16 → ≈2.29 B) |
| AudioVAE file | `audiovae.pth` 301.5 MB (fp32 → ≈75 M) | `audiovae.pth` 346.0 MB (≈86 M) | `audiovae.pth` 377.0 MB (≈94 M) |
| Output sample rate | 16 kHz | 44.1 kHz | **48 kHz** (encode side stays 16 kHz) |
| LM token rate | 12.5 Hz (patch 2 × 25 Hz latents) | 6.25 Hz (patch 4) | 6.25 Hz (patch 4) |
| Languages | zh, en | zh, en | 30 languages + 9 zh dialects |
| Cloning | continuation only | continuation only | isolated reference **and** continuation; voice design; style control |
| RTF RTX 4090 (README) | ~0.17 | ~0.15 | ~0.30 (~0.13 with Nano-vLLM) |
| VRAM (README) | ~5 GB | ~6 GB | ~8 GB |
| Tech report | arXiv 2509.24650 (ICLR 2026) | — | arXiv 2606.06928 (submitted 2026-06-05) |

- Code license: Apache-2.0 (`LICENSE`, `pyproject.toml: license = "Apache-2.0"`). GitHub README §License: "VoxCPM model weights and code are open-sourced under the Apache-2.0 license."
- Base model: the 0.5B/1.5 cards declare `base_model: openbmb/MiniCPM4-0.5B`. The VoxCPM2 card has no `base_model` tag; the GitHub README says "Built on a MiniCPM-4 backbone", and its 28-layer/2048-hidden config does not match any released MiniCPM4 size (see Unknowns).
- Training data notes: 0.5B/1.5 cards: "trained on a massive 1.8 million-hour bilingual corpus"; VoxCPM2 card: "over 2 million hours of multilingual speech data". No corpus list, no usage restriction beyond Apache-2.0; README "Risks and Limitations" forbids impersonation/fraud/disinformation (ethical statement, not a license term).
- Reported quality (GitHub README, Seed-TTS-eval): VoxCPM2 test-EN WER 1.84 / SIM 75.3, test-ZH CER 0.97 / SIM 79.5, test-Hard 8.13 / 75.3; VoxCPM-0.5B 1.85/72.9, 0.93/77.2, 8.87/73.0; VoxCPM1.5 2.12/71.4, 1.18/77.0, 7.74/73.1. CV3-eval multilingual and an internal 30-language ASR benchmark (avg 1.68% error, transcribed by Gemini 3.1 Flash Lite) are also published in the README, as are InstructTTSEval voice-design scores.

Repo/package state: PyPI package `voxcpm`, latest **2.0.3**; tags `1.0.x`, `1.5.0`, `2.0.0–2.0.3`; branches `main` (supports all variants), `dev_1.5`, `dev_2.0`. **VoxCPM2 support lives on `main`** — `src/voxcpm/core.py` dispatches on `config.json:"architecture"` (`"voxcpm"` → `VoxCPMModel`, `"voxcpm2"` → `VoxCPM2Model`).

---

## 3. Architecture decomposition

The synthesis path has five neural modules (names from code; the docs call the pipeline **LocEnc → TSLM → RALM → LocDiT**, then AudioVAE decode; readthedocs `models/architecture.html`):

### 3.1 Text-Semantic LM ("base_lm", TSLM) — MiniCPM-4-style decoder
`src/voxcpm/modules/minicpm4/model.py`. Standard pre-norm decoder: RMSNorm → GQA attention → residual → RMSNorm → SwiGLU MLP (gate/up/down, SiLU) → residual; final RMSNorm; **no biases** on any linear. Rotary embedding is **LongRoPE** (`MiniCPMLongRoPE`): per-dimension `short_factor`/`long_factor` rescaling of inverse frequencies plus a global `scaling_factor = sqrt(1 + log(scale)/log(orig_max_pos))` applied to cos/sin; at ≤32768 positions only `short_factor` is used. `use_mup: false` in every released config, so the µP hooks (`scale_emb`, `scale_depth`) are inert at inference.

Per-variant dims (`config.json` of each HF repo):

| | 0.5B | 1.5 | **2 (VoxCPM2)** |
|---|---|---|---|
| hidden | 1024 | 1024 | 2048 |
| intermediate | 4096 | 4096 | 6144 |
| layers | 24 | 24 | 28 |
| heads / kv-heads | 16 / 2 | 16 / 2 | 16 / 2 |
| head_dim | 64 | 64 | 128 (`kv_channels: 128`) |
| vocab | 73448 | 73448 | 73448 |
| derived params | ≈0.43 B | ≈0.43 B | ≈1.47 B (incl. 150 M embedding) |

### 3.2 Residual Acoustic LM ("residual_lm", RALM)
Same `MiniCPMModel` block, no embedding table, fewer layers: 6 (0.5B) / 8 (1.5) / 8 (VoxCPM2). VoxCPM2 sets `residual_lm_no_rope: true` — **the residual LM runs with no positional encoding at all** (`no_rope` → `rope_emb = None`, `model.py:348-351`). Derived params: ≈0.09 B / 0.12 B / **0.38 B**. Input fusion differs by generation: v1 adds (`enc_outputs + feat_embed`, `voxcpm.py:289`), v2 concatenates and projects (`fusion_concat_proj(cat(enc_outputs, feat_embed))`, `voxcpm2.py:339-341, 1073-1075`).

### 3.3 Local Encoder ("feat_encoder", LocEnc)
`modules/locenc/local_encoder.py` (30 lines): per patch of P latent frames (P=2/4/4), project 64-dim latents to hidden, prepend a learned CLS token, run a small **bidirectional** (`is_causal=False`) MiniCPM stack, take the CLS output → one embedding per patch. Dims: hidden 1024, ffn 4096, 16 heads; layers 4 / 8 / **12**; VoxCPM2 adds `kv_channels: 128`. Derived params ≈60 M / 120 M / **208 M**.

### 3.4 Local DiT + Unified CFM ("feat_decoder", LocDiT)
`modules/locdit/local_dit.py` (v1) / `local_dit_v2.py` (v2) + `unified_cfm.py`. The DiT estimator is again a bidirectional MiniCPM stack (hidden 1024, ffn 4096; layers 4 / 8 / **12**) over a tiny sequence: `[mu-token(s), (t-emb), cond-patch (P tokens), noisy-x patch (P tokens)]`, with sinusoidal timestep embedding (scale=1000) through a 2-layer SiLU MLP, plus a second "delta-t" embedding added in; output positions after the prefix are projected back to 64 channels. **v1 vs v2 difference**: v1 fuses `mu + t` into a single prefix token (`local_dit.py:109`); v2 reshapes `mu` (a concat of the two LM projections, 2×1024) into **two** prefix tokens plus a separate t token (`local_dit_v2.py:109-110`), and the two LM heads are combined by concat instead of add (`voxcpm2.py:1084-1086` vs `voxcpm.py:834-836`). Derived params ≈62 M / 122 M / **210 M**.

`UnifiedCFM.forward` (inference): draw `z ~ N(0,1)` of shape `(B, 64, P)`; timestep span = `linspace(1→0, n+1)` warped by **sway sampling** (`t + coef·(cos(π/2·t) − 1 + t)`, coef 1.0, `unified_cfm.py:66-67`); Euler ODE integration with **classifier-free guidance** where the unconditional branch zeroes only `mu` (the prefix `cond` patch is kept in both branches, `solve_euler:103-116`); **CFG-Zero\*** trick: the first `max(1, 4%·n)` steps output zero velocity, and the guided update uses an optimized projection scale `st* = ⟨pos,neg⟩/‖neg‖²` (`solve_euler:96-128`). Defaults: `inference_timesteps=10`, `cfg_value=2.0` (matches `cfm_config.inference_cfg_rate: 2.0` in all three config.json). `mean_mode`/`dt` (MeanFlow-style) is present but disabled in every released config.

### 3.5 FSQ bottleneck ("fsq_layer")
`modules/layers/scalar_quantization_layer.py` (26 lines): `Linear(hidden→latent) → tanh → round(x·9)/9 → Linear(latent→hidden)`; latent 256 (v1/1.5) / **512** (v2). Applied to base-LM outputs at audio positions only; this is the "semi-discrete residual representation" of the paper (arXiv 2509.24650 abstract) — it is a **hard rounding step at inference** (`self.training` branch is STE).

### 3.6 Stop predictor
`Linear → SiLU → Linear(hidden→2)`, argmax on the *previous* LM hidden decides stop (`voxcpm2.py:1113-1115`); generation stops when `i > min_len and stop_flag == 1`, plus a hard cap `max_len = min(6·len(target_tokens)+10, max_len)`.

### 3.7 AudioVAE (v1) / AudioVAE V2
`modules/audiovae/audio_vae.py` / `audio_vae_v2.py`. DAC-derived (README acknowledgment) **causal** conv VAE, weight-normalized:
- Encoder: `WNCausalConv1d(1→d,k7)` + 4–5 strided blocks, each `3×CausalResidualUnit (Snake + dilated causal conv k7 d∈{1,3,9} + Snake + 1×1) → Snake → strided causal conv (k=2s)`, then k3 convs for mu/logvar; inference uses `mu` only (`encode()` returns `["mu"]` — no reparameterization noise).
- Decoder: mirror with `WNCausalTransposeConv1d` upsampling (k=2s, right-trim `2p−op` for causality), Snake, dilated residual units, final `Snake → conv k7 → tanh`. `depthwise: true` (grouped convs) in all shipped configs.
- Latent: 64-dim, 25 Hz in all variants. Rates: 0.5B enc [2,5,8,8]@16k / dec [8,8,5,2] → 16 kHz out; 1.5 enc [2,3,6,7,7]@44.1k / dec mirror → 44.1 kHz out; **VoxCPM2 asymmetric**: enc [2,5,8,8]@16 kHz (hop 640), dec [8,6,5,2,2,2] (hop 1920) → **48 kHz out from 16 kHz in** (built-in 3× super-resolution).
- V2-only `SampleRateConditionLayer`: per-decoder-block scale/bias embeddings indexed by `bucketize(sr, [20000,30000,40000])`; at inference the default is the fixed `out_sample_rate=48000` (`audio_vae_v2.py:452-473`) — constant-foldable in a port.
- `NoiseBlock` (random noise injection in decoder) exists but `use_noise_block` is false/absent in every shipped config.
- V2 adds a **stateful streaming decoder** (`StreamingVAEDecoder`) that monkey-patches causal convs to carry left-context buffers so each latent patch decodes incrementally without overlap re-decode (`audio_vae_v2.py:504-579`).

### 3.8 Glue
Projections `enc_to_lm_proj`, `lm_to_dit_proj`, `res_to_dit_proj` (+ v2 `fusion_concat_proj`), all plain Linear. Optional LoRA wrappers on attention/projection linears (inference-relevant only if fine-tuned weights are loaded).

Derived totals (bf16 file size ÷ 2, cross-checked against per-module sums): **0.65 B / 0.80 B / 2.29 B** + VAE ≈ 75/86/94 M fp32. The marketing sizes (0.5B/0.6B/2B) refer to the backbone.

---

## 4. Inference loop (`voxcpm2.py:_inference`, mirrors v1)

1. **Sequence build** (§7 below): text tokens (+ optional ref/prompt segments) and aligned latent-patch slots; token id 101 `audio_start` appended after text; v2 adds ids 103/104 `ref_audio_start/end` around an isolated reference-audio segment.
2. **Prefill**: LocEnc embeds all prompt audio patches; text embeddings and patch embeddings are merged by masks into one sequence; one causal full-sequence pass of base_lm → fill a **preallocated StaticKVCache** (`minicpm4/cache.py`: `[2, layers, B, kv_heads, max_len, head_dim]`, max_len 4096 (v1) / 8192 (v2), batch 1); FSQ applied at audio positions; residual_lm prefilled the same way with fused inputs.
3. **AR loop** (one iteration = one patch = P latent frames = **160 ms** of audio for v2 (4×1920/48000), 80 ms for 0.5B):
   - `mu` = concat(v2)/add(v1) of the two projected LM states;
   - **LocDiT CFM sampling**: 10 Euler steps, first step zeroed (CFG-Zero*), remaining 9 run the estimator with **batch 2** (cond + uncond) over a ~7–11-token sequence;
   - predicted patch → LocEnc → next LM input embedding; patch also becomes the next step's `cond` prefix;
   - stop head argmax; then single-token `forward_step` of base_lm (KV-cache append) → FSQ → residual_lm `forward_step`.
4. **Decode**: latent sequence → AudioVAE decoder in fp32. Non-streaming: one decode of the whole sequence (with `context_len` prompt patches trimmed). **Streaming: yes, per-patch** — v2 yields one 160 ms chunk per AR step through the stateful VAE streaming decoder; v1 re-decodes a 3-patch sliding window and keeps the last patch.
5. **Bad-case retry** (default on in the `VoxCPM` wrapper): if generated length ≥ 6× text tokens, reseed (seed+1) and regenerate, ≤3 times.

KV-cache: static, filled once per utterance; single sequence; `kv_cache.step()` returns the position id. No MTP, no speculative decoding in the reference implementation (Nano-vLLM/vLLM-Omni add batching/paged attention externally).

---

## 5. Operator surface (and GGML mapping)

**Transformer side** (base_lm, residual_lm, LocEnc, LocDiT):
- RMSNorm (fp32 accumulation), SiLU, SwiGLU MLP, bias-free Linear — trivial.
- GQA attention 16 q-heads / 2 kv-heads via `scaled_dot_product_attention` (causal for LMs, **non-causal** for LocEnc/LocDiT, masked single-step for decode) — standard.
- **LongRoPE**: RoPE with per-dim frequency factors + global cos/sin scale — llama.cpp already implements this for Phi-3/MiniCPM4 (`ggml_rope_ext` with freq-factors tensor); needs the scaling_factor multiplication.
- **NoPE** residual LM (v2): attention with no positional encoding — trivial (skip rope).
- Learned CLS token concat (LocEnc); sinusoidal timestep embedding + 2 small MLPs (LocDiT) — timesteps are a fixed schedule given `n_timesteps`, so t/dt embeddings can be **precomputed host-side per step**.
- FSQ: `tanh → round(x·9)/9` — elementwise; **discrete output** (see §8; project rule: discrete outputs need CPU placement / careful backend handling).
- CFG batch-2 doubling; CFG-Zero* optimized scale (per-sample dot product and squared norm → scalar scale) — sum/mul ops or host-side.
- Euler ODE loop with sway-warped step sizes — host-side loop, per-step graph.
- Stop head argmax — host-side.

**AudioVAE side**:
- Causal Conv1d with dilation (1/3/9) and groups (depthwise) — `ggml_conv_1d` supports stride/pad/dilation; depthwise via `ggml_conv_1d_dw`; causality is left-padding (explicit pad).
- **CausalTransposeConv1d** (k=2s, right trim) — `ggml_conv_transpose_1d` (synthesize.cpp already exercises this op for VITS/Kokoro; the known ggml patch history around conv_transpose_1d applies).
- **weight_norm** on every conv — fold `g·v/‖v‖` into plain weights at conversion time.
- **Snake** activation `x + (1/α)sin²(αx)` — already implemented for Kokoro.
- Tanh output; embedding-lookup scale/bias (sample-rate conditioning, constant at 48 kHz → fold).
- **No STFT/FFT, no LSTM, no vocoder-style harmonic source** anywhere in the synthesis path.

**Not in the core path** (skippable for a port): ZipEnhancer denoiser (ModelScope, optional `load_denoiser`), stable-ts timestamps (optional extra), VAD trim (librosa, off by default), wetext normalization (off by default).

**New operator/paradigm surface relative to already-ported families** (VITS: flow+HiFi-GAN; Kokoro: iSTFTNet/LSTM/AdaIN/Snake/harmonic-plus-noise; Qwen3-TTS: AR decoder+MTP+RVQ codec; OmniVoice):
1. **Local flow-matching DiT decoder over continuous latents** — per-frame conditional ODE sampling inside an AR loop; CFG batching, CFG-Zero*, sway schedule. This is the headline new capability (diffusion/CFM inference), and it is a *small bidirectional transformer*, not a U-Net — no new conv ops needed for it.
2. **FSQ scalar quantization** as an inference-time rounding bottleneck (new; trivially implementable, discrete-output caveat).
3. **LongRoPE frequency-factor RoPE** + µP-config plumbing (new to synthesize.cpp, well-trodden in llama.cpp).
4. **Patch-level Local Encoder** (CLS-token pooling transformer).
5. **Asymmetric, sample-rate-conditioned, streaming causal VAE** with cross-call conv state (new streaming-decoder pattern; ops themselves overlap with existing families).
6. Continuous-latent AR loop itself (model feeds back *continuous* vectors, not token ids) — new inference-loop shape vs Qwen3-TTS's discrete AR.

Nothing in the list looks *missing* from GGML; the awkward parts are engineering (weight-norm folding, per-step graphs for the ODE loop or a step-count-templated graph, RNG, streaming conv state), not kernels.

---

## 6. Text frontend

- **Tokenizer**: `LlamaTokenizerFast` (SentencePiece-BPE, MiniCPM4 vocab, `vocab_size: 73448`, `tokenizer.json` 3.7 MB) wrapped by `mask_multichar_chinese_tokens` (`model/utils.py:40-138`) which **splits every multi-character pure-CJK token into single-character token ids** (the model was trained that way). VoxCPM2's HF repo ships this baked in as `tokenization_voxcpm2.py` (`VoxCPM2Tokenizer`, registered via `auto_map`) with a precomputed id→ids split map — exactly the artifact a converter can serialize into GGUF metadata. BOS is added (`add_bos_token: true`); special ids 101/102/103/104 (`audio_start/end`, `ref_audio_start/end`) are hardcoded in the model classes.
- **Normalization is optional and off by default** (`normalize: bool = False` in `core.py:_generate`; the 1.5 card notes enabling it "will disable native raw text support"). When enabled: `wetext` Normalizer (zh + en TN), `inflect` number spelling, regex/markdown/emoji cleanup (`utils/text_normalize.py`, functions copied from CosyVoice `frontend_utils.py` per the header comment). The only always-on preprocessing is whitespace collapsing (`core.py:246-247`).
- **No phonemization anywhere** — raw text (or normalized text) goes straight to BPE. 30-language coverage in VoxCPM2 needs no language tag (model card).
- **Voice design is purely textual**: a parenthesized natural-language description prepended to the text — `build_final_text` in `cli.py:68-70` does `f"({control}){text}"`. No extra module, no extra ops.

Embeddability verdict: excellent — the mandatory frontend is BPE + CJK char-split; wetext/inflect are optional and can stay out of a C library.

---

## 7. Voice cloning / prompt conditioning / voice profiles

Four sequence layouts in `voxcpm2.py:_generate` (v2; v1 supports only 2 and 1):
1. **Zero-shot / voice design**: `[text, audio_start]` (design = description inside the text).
2. **Continuation ("Ultimate Cloning")**: `[prompt_text + target_text, audio_start, prompt-audio latents]` — model continues the prompt audio; requires transcript.
3. **Reference-isolated (v2 only)**: `[ref_audio_start, ref latents, ref_audio_end]` prefix + `[text, audio_start]` — timbre cloning without transcript, style still steerable by text control.
4. **Combined** (3 + 2) for maximum similarity.

What gets encoded: the reference/prompt wav is loaded at 16 kHz (v2) via librosa, padded to patch alignment (left-pad for continuation, right-pad for reference), and passed through the **AudioVAE encoder → 64-dim, 25 Hz `mu` latents** — no speaker-embedding network exists.

**Precomputable voice profile: yes.** `build_prompt_cache` (`voxcpm2.py:694-754`) stores exactly `{mode, ref_audio_feat?, prompt_text?, audio_feat?}` — plain fp32 latent tensors (T×P×64) plus raw text. It is serializable, reusable across generations, and `merge_prompt_cache` supports appending generated audio to stabilize long-form synthesis. This maps cleanly onto synthesize.cpp's Voice Profile concept (the LM prefill KV-cache is *not* cached — it is recomputed per generation, so a profile is backend-independent).

---

## 8. Randomness / determinism

Sampling sites:
1. **One `torch.randn` per AR step** — the CFM initial noise `z (1×64×P)` (`unified_cfm.py:64`), temperature fixed 1.0. This is the only unavoidable noise in the synthesis path.
2. Seed plumbing: `materialize_generation_seed`/`apply_generation_seed` (`model/utils.py:27-37`) — explicit `seed` parameter, `torch.manual_seed` global; `last_successful_seed` is recorded. With a fixed seed the reference implementation is deterministic per platform.
3. **Bad-case retry** (default on in the high-level API) draws a fresh random seed when none is given and increments it on retry.
4. VAE: encoder uses `mu` only (no reparameterization); decoder `NoiseBlock` disabled in all shipped configs → **codec is deterministic**.
5. There is **no token sampling / temperature / top-k** — the "LM" never samples a discrete token; the only discrete decisions are the FSQ rounding and the stop-head argmax.

Implications for oracle (tensor-by-tensor) validation:
- Stage-wise validation is clean: every module is a deterministic function given its inputs, **provided the CFM noise `z` is injected** rather than drawn — the natural fit for synthesize.cpp's deterministic random streams is to capture/inject `z` per step (per-step noise is a single small tensor).
- **FSQ rounding boundaries** can flip on ULP-level differences (`round(tanh(x)·9)`), changing everything downstream — same class of hazard as the project's "discrete outputs need CPU placement" rule; tolerance design must treat FSQ outputs as discrete (compare pre-round values, or require exact index match with a boundary-distance guard).
- **Stop-head argmax** can diverge near the boundary → different output lengths; validation should compare per-step stop logits, not just final audio.
- AR feedback of continuous latents compounds small errors across steps × 9 CFM evaluations each; end-to-end golden tests need per-step comparison or generous late-step tolerances.
- Reference runs bf16 on CUDA but supports **CPU fp32** (`resolve_runtime_device` handles cpu/mps/cuda; MPS is forced to fp32) — use CPU/fp32 with `optimize=False` (disables torch.compile) as the oracle configuration.

---

## 9. Reference implementation as validation oracle

- **Install**: `pip install voxcpm` (PyPI 2.0.3), Python ≥3.10 <3.13, `torch>=2.5.0` (no upper pin). Heavy mandatory deps (`gradio`, `modelscope`, `funasr`, `datasets`, `librosa`…) even for core inference — workable in a locked per-family uv env like `scripts/envs/vits/`, but bloated; a slimmed oracle env may want to vendor just `src/voxcpm` + torch/transformers/librosa/safetensors.
- **Inspectability**: plain `nn.Module`s, no fused/custom kernels (SDPA only), `optimize=False` disables compile — hooks give every intermediate tensor. Generator-based `_inference` yields per-patch latents natively.
- **Stability**: tagged releases; `main` currently at 2.0.3 with all three variants behind one `architecture` switch; the project moves fast (3 generations in 8 months) so pinning tag `2.0.3` + HF revision hashes is advisable.
- Weight formats: v1.5/v2 main weights in safetensors; **AudioVAE is a pickle `.pth`** (loaded `weights_only=True`) — conversion tooling reads it in Python; positive-validation of our own GGUF output (whitelist-own-format rule) unaffected.

---

## 10. Performance evidence (CPU feasibility)

Official (GitHub README): RTF RTX 4090 — 0.5B ~0.17, 1.5 ~0.15, VoxCPM2 ~0.30; Nano-vLLM ~0.13; vLLM-Omni official support.

Ports (empirical, RTF = compute-time / audio-time, >1 is slower than real time):
- **llama.cpp-omni, VoxCPM2 Q8_0, Apple M4 Pro / Metal: RTF ~1.76** (OpenBMB README §On-Device Inference).
- **VoxCPM.cpp (bluryar), i5-12600K 8 threads, CPU**: VoxCPM-0.5B Q4_K full pipeline **RTF 3.609** (model-only 1.826); VoxCPM1.5 Q8_0 full **4.291** (OpenBMB/VoxCPM issue #194, author's benchmark). RTX 4060 Ti F16 0.5B full 0.567.
- **VoxCPMANE (CoreML/ANE), VoxCPM2: RTF ~1.02–1.03**, TTFB ~0.16 s.

Derived compute split (my estimate from configs, flagged as such): at 6.25 patches/s, VoxCPM2's LocDiT inner loop costs ≈ 9 steps × 2 batch × ~11 tokens × 2×0.21 GFLOP ≈ 80 GFLOP per patch ≈ **500 GFLOP per audio-second — ~25× the base-LM decode cost** (≈18 GFLOP/s) plus a ~94 M-param 48 kHz conv decoder. The diffusion loop, not the 2B LM, is the bottleneck; levers are `inference_timesteps` (CLI allows 4–30; quality trade-off unmeasured here), CFG batching, and the smaller variants (0.5B/1.5 have 4-/8-layer DiTs, ~4× cheaper).

**Conclusion: no existing implementation demonstrates real-time CPU synthesis for any variant.** VoxCPM is practical on CPU for *offline* synthesis (minutes-scale texts at 2–4× real time on desktop CPU for 0.5B/1.5), and real-time only with GPU.

---

## 11. Prior art (C++/GGML/other ports)

| Project | Scope | Approach / status |
|---|---|---|
| **llama.cpp-omni** (tc-mb) | VoxCPM2 (`voxcpm2-cli`), also MiniCPM-o | llama.cpp fork, MIT; GGUF split **BaseLM (F16/Q8_0) + Acoustic F16** ("ResidualLM + FSQ + LocEnc/LocDiT CFM + AudioVAE" per the GGUF repo card); CPU/Metal/CUDA/Vulkan; officially advertised in the OpenBMB README as the on-device path; weights at `DennisHuang648/VoxCPM2-GGUF` (BaseLM-Q8_0 1.6 GB, BaseLM-F16 3.0 GB, Acoustic-F16 1.7 GB). Note: acoustic stack kept at F16 — a signal about quantization sensitivity. |
| **VoxCPM.cpp** (bluryar) | 0.5B, 1.5, preliminary VoxCPM2 | Standalone ggml engine, Apache-2.0, ~89 stars; GGUF Q4_K/Q8_0/F16/F32, CPU/CUDA/Vulkan, OpenAI-compatible server, WASM playground; author states an ongoing runtime refactor and AI-assisted development; linked from the **official readthedocs deployment section**. GGUFs at `bluryar/VoxCPM-GGUF`; third-party v2 GGUFs at `cstr/voxcpm2-GGUF`. |
| **audio.cpp** (0xShug0) | VoxCPM2 among ~49 families | Unified ggml audio framework (TTS/ASR/music), CUDA/HIP/Vulkan/Metal/CPU, GGUF F16/Q8; release 0.5 (2026-07). |
| **VoxCPM-ONNX** (bluryar) | 0.5B | ONNX export + FastAPI; **archived**, redirects to a DakeQQ fork with 1.5 support; noted as fully AI-generated. |
| **VoxCPMANE** (0seba) | VoxCPM2 | CoreML/Apple Neural Engine + numpy runtime, RTF ~1.03, OpenAI-compatible server. |
| **voxcpm_rs** (madushan1000) | 0.5B | Rust / Burn framework; functional; **AGPL-3.0** (do not read as implementation reference for license hygiene). |
| MiniCPM-4 in llama.cpp | LM backbone | Official GGUF releases exist (`openbmb/MiniCPM4-0.5B-QAT-Int4-GGUF` 2025-06-09, `MiniCPM4-8B-GGUF`, `MiniCPM4.1-8B-GGUF`) → the MiniCPM4 architecture (GQA + LongRoPE) is supported upstream, derisking the TSLM. |

Takeaway: conversion recipes, GGUF layouts, and cross-check oracles exist in at least three independent ggml codebases (consistent with the "check reference ports before designing" practice); none is a reason not to port — synthesize.cpp's value (stable C ABI, per-stage validation, voice profiles, quant matrix) is orthogonal.

---

## 12. Assessment against synthesize.cpp family-selection criteria

1. **Representative architecture + useful new operator surface — strong yes.** Diffusion-autoregressive / local flow-matching decoding over continuous latents (DiTAR lineage) is the major open-TTS paradigm the library does not yet cover. New surface: CFM Euler sampling with CFG-Zero* and sway schedule, FSQ, LongRoPE, NoPE, patch CLS-encoder, asymmetric streaming causal VAE, continuous-vector AR feedback. Heavy reuse of existing surface: Snake, causal/transposed/dilated/depthwise conv (VITS/Kokoro), AR KV-cache loop (Qwen3-TTS).
2. **Stable, inspectable reference — yes.** PyPI 2.0.3, tagged, plain PyTorch, CPU/fp32 path, seed control, generator yields per-step tensors. Caveats: heavy dependency closure; fast-moving upstream → pin tag + HF revisions.
3. **A variant practical for CPU inference — the weak point.** Best evidence: 0.5B Q4_K full pipeline RTF ≈3.6 on an i5-12600K; VoxCPM2 RTF ≈1.8 even on M4 Pro Metal. Offline/batch CPU synthesis is practical; real-time CPU is not, at reference settings. Mitigations to evaluate during a port: fewer CFM steps (4–6), quantized DiT (unproven — llama.cpp-omni keeps acoustic at F16), smaller variants. This must be stated honestly in any intake decision; it is the same class of trade-off as other diffusion TTS, not an implementation deficiency.
4. **Measurable, tractable operator gaps — yes.** No missing GGML kernels identified; gaps are integration-level.
5. **Licenses — clean across the board.** Code Apache-2.0; all three weight sets Apache-2.0 by model card (quoted §2); tokenizer inherits MiniCPM4 (Apache-2.0); no bundled voices (voices come from user reference audio or textual design); mandatory frontend has no licensed data files. Optional extras (wetext, ZipEnhancer, SenseVoice, stable-ts) can be excluded. One hygiene note: `examples/*.wav` in the repo have no stated provenance — don't redistribute them.
6. **Reproducible conversion + incremental quantization — yes.** safetensors (+pickled VAE) → GGUF proven by three independent converters; weight-norm folding and the CJK split map are the notable conversion steps; quantization ladder F32→F16→Q8→Q4 demonstrated for the LM, with the open question of DiT/VAE sensitivity (prior art keeps acoustic F16 — plan quant experiments per module).
7. **End-to-end testable pipeline — yes.** Acoustic model + codec + frontend all in one repo/weights; stage boundaries (LocEnc / TSLM / RALM / LocDiT / VAE-enc / VAE-dec) are clean seams for per-stage golden tests; deterministic given injected noise.

### Key risks
- **CPU RTF** (above) — the diffusion inner loop is ~25× the LM decode cost; frame it as an offline-synthesis family on CPU, real-time on GPU.
- **Oracle comparison across an AR loop with per-step noise + two discrete operations** (FSQ round, stop argmax) — needs noise injection and boundary-aware tolerances; more validation machinery than VITS/Kokoro, similar in spirit to Qwen3-TTS sampling.
- **VoxCPM2 size** (2.29 B bf16 = 4.6 GB; Q8 ≈ 2.4 GB + F16 acoustic) — fine for desktop, heavy for embedded.
- Upstream velocity — three architectures in 8 months; the v1/v2 model classes already diverge (fusion, DiT prefix, ref tokens, VAE); a port should parameterize these five deltas rather than fork per variant.
- Repeated-generation UX for voice design ("try 1–3 times", README Risks) is a product-level nondeterminism, unaffected by porting.

### Recommended reference variant
**VoxCPM2** (`openbmb/VoxCPM2`, `architecture: "voxcpm2"`): it is the current flagship (30 languages, voice design, controllable + isolated-reference cloning, 48 kHz), has the technical report (arXiv 2606.06928), the most prior GGUF art to cross-check (llama.cpp-omni + DennisHuang648 GGUFs), ships the split-map tokenizer file, and its architecture is a superset of v1. Port the family against VoxCPM2, keeping the five v1/v2 deltas configurable so **VoxCPM1.5 (0.80 B, 44.1 kHz)** lands nearly free as the CPU-practical Model Variant; treat VoxCPM-0.5B as legacy (pickle-only weights, 16 kHz).

---

## 13. Unknowns / could not verify

- **Per-module parameter counts are my derivations** from config.json dims and file sizes; not officially published anywhere I found (readthedocs architecture page has no numbers).
- Whether the VoxCPM2 2B backbone was initialized from a released MiniCPM checkpoint (its 28L/2048h/6144ffn/head_dim-128 shape matches no published MiniCPM4 size; the model card omits `base_model`).
- Quantization sensitivity of LocDiT/AudioVAE — inferred only from llama.cpp-omni's choice to ship Acoustic at F16; no published ablation.
- Quality impact of reducing `inference_timesteps` below 10 — the CLI permits 4–30 but no official quality-vs-steps curve is published.
- Exact vocab entries for the hardcoded special ids 101–104 (not cross-checked against tokenizer.json contents).
- `wetext` package license (unverified; irrelevant if normalization stays out of the port).
- ModelScope mirrors assumed identical to HF (not verified).
- llama.cpp-omni's VoxCPM2 support is documented via the OpenBMB README and the GGUF weight card; its own README (as fetched) foregrounds MiniCPM-o and I did not read its voxcpm2 source.
- RTF convention in community benchmarks (issue #194) assumed compute-time/audio-duration; the issue reports the numbers without defining the ratio direction, though >1 values alongside "CPU" strongly imply it.
- arXiv 2606.06928 was read via abstract page only; section-level claims (e.g., unified sequence organization) are quoted from the abstract, not the PDF body.

## 14. Primary sources

- Cloned repo `github.com/OpenBMB/VoxCPM` @ ee8161e — files cited inline (`src/voxcpm/model/voxcpm.py`, `model/voxcpm2.py`, `model/utils.py`, `modules/minicpm4/{model,config,cache}.py`, `modules/locdit/{local_dit,local_dit_v2,unified_cfm}.py`, `modules/locenc/local_encoder.py`, `modules/audiovae/{audio_vae,audio_vae_v2}.py`, `modules/layers/scalar_quantization_layer.py`, `core.py`, `cli.py`, `utils/text_normalize.py`, `pyproject.toml`, `README.md`, `conf/voxcpm_v{1,1.5,2}/*.yaml`)
- HF: `huggingface.co/openbmb/VoxCPM-0.5B`, `/VoxCPM1.5`, `/VoxCPM2` (README.md front-matter + config.json + API metadata incl. file sizes and createdAt), `huggingface.co/DennisHuang648/VoxCPM2-GGUF`, `huggingface.co/api/models?author=openbmb&search=MiniCPM4`
- arXiv: 2509.24650 (VoxCPM, 2025-09-29), 2606.06928 (VoxCPM2 Technical Report, 2026-06-05)
- Docs: `voxcpm.readthedocs.io/en/latest/models/architecture.html`, `/deployment/voxcpm_cpp.html`
- Ports: `github.com/bluryar/VoxCPM.cpp`, `github.com/tc-mb/llama.cpp-omni`, `github.com/0xShug0/audio.cpp`, `github.com/0seba/VoxCPMANE`, `github.com/madushan1000/voxcpm_rs`, `github.com/bluryar/VoxCPM-ONNX`, `OpenBMB/VoxCPM` issue #194 (performance data), `huggingface.co/cstr/voxcpm2-GGUF`
- PyPI: `pypi.org/pypi/voxcpm/json`
