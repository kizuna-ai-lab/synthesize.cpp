# OmniVoice Model Family — Design

Status: Approved in discussion with jiangzhuo on 2026-07-30. This is the design
record for the fourth Model Family; the family contract itself will be
confirmed in `docs/porting/families/omnivoice.md` at intake, per
`docs/model-porting.md`.

## 1. Context and Selection

OmniVoice (`k2-fsa/OmniVoice`, Xiaomi / Next-gen Kaldi, arXiv 2604.00688) was
recorded as the leading fourth-family candidate at the third-family selection
(`docs/model-family-selection.md`, "Fourth Model Family Candidate") and is now
confirmed as the fourth Model Family. It is a non-autoregressive mask-predict
(discrete masked-diffusion) model: a Qwen3-0.6B backbone run with bidirectional
attention over a fixed-length canvas of 8 acoustic codebooks × 1025-entry
vocabulary at 25 Hz, refined over 32 (fast mode: 16) parallel denoising steps
with classifier-free guidance (scale 2, two forwards per step), decoded to
24 kHz audio by the Higgs Audio V2 codec's DAC-style convolutional decoder
(hop 960). One 0.6B + 0.2B checkpoint carries reference-audio cloning,
described-attribute voice design, and automatic voice selection together, with
600+ claimed languages.

The Reference Model Variant is the flagship `k2-fsa/OmniVoice` checkpoint
(fp32 safetensors, revision pinned at intake). `k2-fsa/OmniVoice-Emilia` is a
zh+en paper-reproduction variant whose model card carries no license text at
all; it is not ported.

## 2. Licensing and the Restricted Model Package (ADR 0018)

Verified against the upstream Hugging Face card on 2026-07-30 (no change since
2026-07-03):

- **LM weights: CC-BY-NC** (no version stated). The card has no `license:`
  frontmatter; the only statement is prose: "Our code is released under the
  Apache 2.0 License. The pre-trained model is licensed under the CC-BY-NC due
  to constraints from its training data (e.g., Emilia)."
- **Codec weights: Boson Higgs Audio 2 Community License** — the weights repo
  bundles `audio_tokenizer/LICENSE`, a Meta-Llama-3-derived agreement
  (redistribution permitted with the agreement text, attribution, and naming
  obligations; 100k-MAU commercial cap; no-training-other-models clause). The
  July intake note recorded only CC-BY-NC; this third license is a new finding.
- **Code: Apache-2.0** (GitHub `k2-fsa/OmniVoice`).
- The card also carries a use disclaimer against unauthorized voice cloning and
  impersonation; the rendered card must reproduce it.

**Decision (jiangzhuo, 2026-07-30, re-confirmed after the Boson finding):** the
family ships under a new publication category defined by **ADR 0018 —
Restricted Model Package**. Rationale: the NC restriction follows the weights
regardless of who runs the converter, so withholding publication would not
enlarge the legal audience — it would only add friction for legitimate
non-commercial users. Both licenses permit non-commercial redistribution with
attribution; upstream itself distributes the weights openly on HF.

A Restricted Model Package:

- passes the same Port Validation Suite as any package, and uses the same
  repository layout, model-card workflow, and acquisition flow as
  `docs/model-packages.md`;
- is **not** a Published Model Package and never carries that term's
  "embeddable in other people's programs" promise;
- declares `license: cc-by-nc-4.0`-class frontmatter (noting upstream states no
  version), ships the Boson agreement verbatim as a declared Sidecar Resource,
  carries dual attribution and a prominent no-commercial-use statement, and
  records the Emilia training-data provenance;
- is uploaded only with jiangzhuo's per-act confirmation, like every
  publication.

Cautionary context recorded for the ADR: all three community GGML ports
misstate or omit these terms (one HF GGUF repo labels the weights apache-2.0);
the project's card must not inherit any downstream claim.

## 3. Family Contract

- **Family** `omnivoice`; output 24 kHz mono; 25 Hz frame rate, hop 960;
  8 codebooks × 1025 vocabulary, `audio_mask_id` 1024.
- **Input forms:** text and token-sequence bypass. The Text Frontend is an
  embedded Qwen byte-level BPE (vocab 151,676) over raw text — no
  phonemization, **no external Text Frontend Provider dependency**. The prompt
  template's seven special markers (`<|denoise|>`, `<|lang_start|>`/`_end`,
  `<|instruct_start|>`/`_end`, `<|text_start|>`/`_end`) are pinned as data.
  The 646-entry language-name↔token mapping is ported as a generated data
  table regardless of how many languages the catalog declares.
- **Language Capability Catalog:** `en`, `zh`, `ja` — validated languages
  only, primary-subtag BCP-47 per the qwen3-tts bridge precedent
  (`src/arch/qwen3-tts/model.cpp` `kTags`). Widening the catalog later is
  validation work, not code work.
- **Voice — three modes mapped onto the existing public Profile sources**
  (`docs/voice-conditioning.md`; OmniVoice is the first family to exercise
  Reference Audio and Description Text):
  - *Cloning* → **Reference Audio**: transcript required, language optional.
    The public Audio Normalizer (ADR 0009) converts to the declared 24 kHz
    mono target; the family internally resamples 24→16 kHz (fixed-ratio 2/3
    polyphase, deterministic, validated against the oracle) for the HuBERT
    semantic branch.
  - *Voice design* → **Description Text**: schema is the upstream attribute /
    free-form instruct vocabulary, trained for zh/en.
  - *Auto-voice* → package-default behavior with no profile selected; the
    voice follows the synthesis seed. Preset Voice Catalog is empty with an
    unnamed package default.
  - Prepared prompts (RVQ tokens + transcript tokens) round-trip through
    **Serialized Profiles** (ADR 0008).
- **Controls:** `speaking_rate` scales the ported RuleDurationEstimator
  (deterministic host code) that fixes the canvas length; the package declares
  the validated range. Synthesis consumes randomness (Gumbel position
  sampling, CFG); generation defaults (num_steps 32, cfg_scale 2.0,
  temperatures, layer penalty 5.0, t_shift 0.1) are embedded by the converter
  from upstream configuration and the loader refuses a package without them
  (the qwen3-tts "second copy" lesson). Seed contract identical to qwen3-tts:
  same seed → byte-identical PCM, concrete seed reporting.
- **Delivery:** complete audio only (`deliver_complete_audio`). No Chunked
  Audio Delivery and no Native Streaming Synthesis claims in v1.

## 4. Package and Conversion

- **One primary GGUF**, `general.architecture = "omnivoice"`, holding LM and
  codec tensors (~800) — required by `docs/model-packages.md`; the bluryar
  port proves single-file feasibility. Catalog-canonical tensor names with
  terse prefixes (ggml's 64-char tensor-name limit is a real constraint for
  HuBERT paths); every tensor carries a role so the quantizer (by name) and
  runtime (by role) cannot drift.
- **Metadata:** `omnivoice.*` namespace (codebook counts, frame rate, special
  token IDs, Higgs codec hyperparameters), embedded tokenizer
  (`tokenizer.ggml.*`), generation defaults, Language Capability Catalog,
  Reference Audio declarations (24 kHz mono target, Reference Frame
  Equivalent limits, max reference count), Profile Schema + version + 32-byte
  Profile Compatibility ID, and source/license/provenance metadata including
  the verbatim CC-BY-NC statement.
- **Converter** `scripts/convert-omnivoice.py` in the locked
  `scripts/envs/omnivoice/` environment: fp32 passthrough (dtype follows the
  checkpoint); folds the HuBERT pos_conv weight-norm at conversion (the
  Serveurperso port validated the fold at ~4e-7 max abs diff); skips
  training-only tensors (fc1, decoder_semantic, RVQ EMA buffers); Higgs RVQ
  codebooks are stored directly as `codebook.embed` (no EMA reconstruction
  needed, unlike qwen3-tts); writes the Boson license file as a declared
  Sidecar Resource.

## 5. C++ Architecture

Module layout `src/arch/omnivoice/`, graph-builder / `-host` pairs per stage:

- `omnivoice.h` — family API. The validation-only seam carries an optional
  replayed Gumbel-noise stream; with validation strategy A it is a fallback,
  not the primary mechanism.
- `model.cpp` — orchestration: prompt build → diffusion loop → codec decode →
  delivery.
- `weights.{h,cpp}`, `catalog.{h,cpp}` — tensor catalog; shapes derived from
  package hyperparameters; a tensor the catalog never asked for is an error.
- `frontend-host.{h,cpp}` — prompt template, language table,
  RuleDurationEstimator.
- `generator.{h,cpp}` + `generator-host.{h,cpp}` — bidirectional Qwen3 block
  graph with the CFG cond/uncond pair built into one graph; host-side decode
  loop (t_shift schedule, confidence ranking, layer penalty, `nth_element`
  commit). **Discrete unmasking decisions always run on host CPU** (the
  discrete-outputs placement rule).
- `codec.{h,cpp}` + `codec-host.{h,cpp}` — RVQ dequant + fc2 + DAC decoder:
  Snake from stock primitives (Kokoro's helper family), conv via
  `ggml_im2col`+matmul, transposed conv via mul_mat + `ggml_col2im_1d`
  (the VITS recipe). **Zero new ggml operators** — three independent ports
  and our own qwen3-tts/VITS experience agree.
- Cloning slice adds `reference-encoder.{h,cpp}` + `-host`: internal 24→16 kHz
  resampler → HuBERT (mean of all 13 hidden states) → semantic + acoustic
  branches → RVQ encode; plus Voice Profile preparation and serialization.

Reuse decisions:

1. **Hoist the byte-level BPE to a shared internal module.** qwen3-tts's
   `bpe.cpp` is the same Qwen2 BPE over the same GGUF vocab layout; move it
   under `src/`, point qwen3-tts at the shared unit, keep its unit tests.
2. **Do not share the Qwen3 block builder.** qwen3-tts's block is causal with
   KV cache; OmniVoice's is bidirectional, cache-free, full-canvas with CFG
   pairing — duplicate-with-adaptation, revisit only if a third Qwen-family
   port appears.

Graphs and memory: no KV cache; two full-canvas forwards per step. Graphs are
built once per canvas shape and reused across all steps. Canvas length is
bounded by the request's effective frame limit and the duration estimate — the
package ceiling is never used as an allocation size (the qwen3-tts 3 TB
lesson).

Core seams touched: `ModelFamily::OmniVoice` + the four dispatch branches in
`src/synthesize.cpp`; first family consumers of the Reference Audio and
Description Text profile sources; `speaking_rate` capability declaration.
**No public ABI additions expected.**

## 6. Validation and Testing (Strategy A: exact-token goldens)

Greedy decoding (both temperatures 0) makes the entire diffusion loop
deterministic on CPU — the property that made OmniVoice the fourth-family
candidate. Port validation exploits it:

- **Oracle:** F32 CPU (checkpoint dtype is fp32; per
  `docs/port-validation.md`, dtype follows the checkpoint and the least
  approximating reproducible device wins), with a byte-identity self-check
  across two runs. Locked env `scripts/envs/omnivoice/`; per-stage dumpers
  (tokenizer, prompt, single forward, per-step unmask grid, codec decode,
  semantic branch, RVQ encode, end-to-end).
- **Golden manifest** (`tests/golden/omnivoice/<variant>.manifest.json`,
  ~20 cases): greedy cases across 3 languages × 3 voice modes assert the
  **exact 8×T token grid equals the oracle's**, with waveform compared through
  the codec under committed tolerances; seeded-sampling cases assert the
  public seed contract (same-seed byte-identical, different-seed
  `artifact_differs`); one fast-mode (16-step) case. Both port and oracle run
  F32 CPU, so `tests/tolerances/omnivoice.json` records reference_stage
  `source-f32-oracle-vs-f32-cpu` with equal-arithmetic (tight) thresholds,
  measured then committed before support is declared.
- **Unit tests per slice** (registered under the `unit` label at each slice,
  sanitizer gate for every inference-code slice): metadata, catalog,
  prompt-builder, duration-estimator, decoder-block, unmask-schedule (pure
  host logic), codec-ops, codec, gumbel-sampling (cross-checked against the
  reference logits processing like qwen3-tts's sampling test), internal
  resampler, rvq-encode, quantization-policy.
- The Gumbel replay seam is implemented only if a sampled-path parity case
  proves necessary; the public sampled path is otherwise covered by the seed
  contract, as for qwen3-tts.

## 7. Quantization and Backends

- **Quantization:** F32 is the reference package. The generator is known to be
  quantization-sensitive (a reference port measured token agreement collapsing
  from 100% to ~7% with f16 generator weights — argmax flips cascade across
  refinement steps). Every produced profile must re-pass the full suite
  including exact-token gates; a profile that fails is not shipped, and no
  perceptual claim substitutes. RVQ codebooks, fc/fc2, norms, and small
  tensors stay reference-dtype (consensus of both reusable ports and our own
  qwen3-tts policy).
- **Backends:** CPU is the mandatory baseline. CUDA begins with the codec
  (col2im path, following the qwen3-tts codec-on-CUDA precedent, with
  placement evidence). Generator-on-CUDA is claimed only if placement evidence
  proves the committed token grids bit-identical to CPU (one port measured
  CUDA-F32 token-exact and Metal-F32 at 83% — encouraging for CUDA,
  disqualifying for nothing until we measure); otherwise the generator stays
  on CPU and the claim is not made.

## 8. Slice Sequence

0. Intake: license audit (CC-BY-NC + Boson + Apache split, pinned revisions,
   SHA-256), `reports/porting/omnivoice/<variant>/intake.json`, ADR 0018,
   family contract doc.
1. Oracle: locked env, golden manifest first, per-stage dumpers.
2. Converter + metadata/catalog unit tests.
3. Load + catalog + frontend (BPE hoist, prompt builder, duration estimator,
   language table).
4. Single-forward backbone parity.
5. Greedy diffusion loop → exact token-grid parity.
6. Codec decode → end-to-end greedy waveform.
7. Public sampling path (Gumbel, CFG, embedded defaults, seed contract).
8. Cloning: internal resampler, HuBERT, acoustic encoder, RVQ encode,
   Reference Audio + Description Text profiles, Serialized Profile GGUF.
9. Port Validation Suite end-to-end; tolerances committed.
10. Quantization profiles (evidence-gated).
11. Execution Backends (CUDA; placement evidence).
12. Adapters: CLI run + Python binding smoke registered for omnivoice (stage
    7.5 — not repeating the qwen3-tts omission).
13. Ship: `docs/models/<variant>.md`, `scripts/hf_cards/<variant>.yaml`,
    rendered card; upload only on per-act confirmation.

Ship (stage 8) waits for cloning (slice 8): the package does not publish
without the model's headline capability.

## 9. Reference Implementations (read at design time)

| Repo | License | Use |
|---|---|---|
| `ServeurpersoCom/omnivoice.cpp` | MIT | Best architecture doc (900-line ARCHITECTURE.md), published GGUFs as cross-check oracles, col2im recipe. README misstates upstream licenses. |
| `bluryar/omnivoice.cpp` | Apache-2.0 | Clearest MaskGIT decode loop in ggml; single-GGUF precedent; ggml CUDA bug audit. Misstates weight license; no converter. |
| `rockerritesh/omnivoice-tts.cpp` | **PolyForm Noncommercial** | **Read-only — no code reuse.** MODEL_SPEC/STAGE0_DESIGN docs; the temps=0 bit-exactness finding; f16 collapse measurement. |
| `0xShug0/audio.cpp` | unverified — treat as read-only | Active multi-model GGML audio engine supporting OmniVoice; positioning mirror. |

Any adopted code (only from the MIT/Apache two) records its notice in
`THIRD_PARTY_NOTICES.md` at adoption time.

## 10. Non-Goals and Deferred

Recorded at intake as out of scope for v1: Whisper auto-transcription
(reference transcript is required), WeTextProcessing/num2words text
normalization, pinyin/CMUdict pronunciation overrides, long-form text chunking
and cross-fade synthesis, auto-voice chunk-to-chunk self-consistency, any
streaming claim, explicit target-duration exposure in the public ABI (the
frame limit remains a cap, not a target), and the OmniVoice-Emilia variant.
Deferred questions carry their owners: generator-on-CUDA (stage 7 evidence),
quantized profiles (stage 6 evidence), wider language catalog (validation
work, any later cycle).
