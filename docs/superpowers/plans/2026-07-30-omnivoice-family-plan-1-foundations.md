# OmniVoice Family — Plan 1: Foundations (slices 0–3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the OmniVoice family's foundations: intake + ADR 0018, the locked oracle environment, the golden manifest, the oracle dumpers, the converter, and the C++ load path through catalog and text frontend — ending with a loadable `omnivoice` GGUF whose metadata, tensors, and frontend are fully unit-tested.

**Architecture:** OmniVoice (`k2-fsa/OmniVoice`) is a non-autoregressive mask-predict TTS: a Qwen3-0.6B backbone run bidirectionally over an 8-codebook × 1025-vocab canvas at 25 Hz, decoded by the Higgs Audio V2 DAC-style codec (hop 960, 24 kHz). This plan follows the staged pipeline in `docs/model-porting.md` (stages 1–3 plus the first C++ slice of stage 4) and mirrors the qwen3-tts family everywhere a precedent exists. Design record: `docs/superpowers/specs/2026-07-30-omnivoice-family-design.md`.

**Tech Stack:** C++17 + GGML/GGUF (submodule, vanilla — zero new operators needed), Python via per-family locked uv env, CMake/CTest.

**Follow-on plans (not in this document):** Plan 2 = backbone graph + greedy diffusion loop + codec decode (slices 4–6); Plan 3 = public sampling + cloning/profiles (slices 7–8); Plan 4 = port-validation suite, quants, backends, adapters, ship (slices 9–13). They are written only after this plan's oracle artifacts exist, because their test assertions consume oracle-measured values.

## Global Constraints

- Weight licenses: LM = CC-BY-NC (no version stated upstream); codec = Boson Higgs Audio 2 Community License; code = Apache-2.0. Publication ceiling = Restricted Model Package (ADR 0018, Task 1). Never label any artifact apache-2.0.
- Reference port `rockerritesh/omnivoice-tts.cpp` is PolyForm Noncommercial: **read-only, zero code reuse**. `ServeurpersoCom/omnivoice.cpp` (MIT) and `bluryar/omnivoice.cpp` (Apache-2.0) may inform patterns; any adopted code adds its notice to `THIRD_PARTY_NOTICES.md` at adoption time.
- Family string everywhere: `omnivoice`. Variant slug: `omnivoice-0-6b`. GGUF arch: `general.architecture = "omnivoice"`; family metadata namespace `synthesize.omnivoice.*`. C++ namespace `synth::omnivoice`, directory `src/arch/omnivoice/`.
- Audio contract constants: sample rate 24000 Hz, mono, f32le; hop 960; frame rate 25 Hz; 8 codebooks; audio vocab 1025; mask id 1024; text vocab 151,676.
- Package contract numbers (used consistently in manifest, converter, readers, tests): `max_input_tokens = 2048`, `max_output_frames = 750` (30 s single-canvas ceiling — long-form chunking is out of scope v1), `speaking_rate_range = [0.5, 2.0]`, `stochastic = true`.
- Generation defaults are read from the pinned upstream package (`OmniVoiceGenerationConfig` dataclass) — never hardcoded a second time. Known values for cross-checks: num_step 32, guidance_scale 2.0, t_shift 0.1, layer_penalty_factor 5.0, position_temperature 5.0, class_temperature 0.0.
- Oracle: F32 on CPU (checkpoint dtype is F32 throughout), `position_temperature=0.0, class_temperature=0.0` for deterministic cases, `postprocess_output=False, pad_duration=0.0, fade_duration=0.0` for **all** cases, `preprocess_prompt=False` and explicit `ref_text` for clone cases (no pydub, no Whisper).
- Commit golden contracts, not payloads: manifests/tolerances/reports are committed; models, oracle tensors, GGUFs are not (`models/`, `build/goldens/` stay ignored).
- Every C++ slice (Tasks 8–12) ends with unit tests registered under the `unit` label AND a clean sanitizer run: `cmake --build build-sanitize --target synthesize-check-unit`.
- clang-format: `scripts/ci/clang-format.sh --fix` before every commit touching C/C++. Never format `ggml/`.
- All work happens on the current worktree branch `worktree-omnivoice`. Do not push or open a PR from inside a task — that decision is jiangzhuo's, per repo policy.

---

### Task 1: ADR 0018 (Restricted Model Package) + vocabulary + selection-doc update

**Files:**
- Create: `docs/adr/0018-publish-nc-families-as-restricted-model-packages.md`
- Modify: `CONTEXT.md` (add one term; read the file first to match its format)
- Modify: `docs/model-family-selection.md` (the "Fourth Model Family Candidate" section)

**Interfaces:**
- Produces: the term "Restricted Model Package" that Task 2's family doc, Task 5's manifest comments, and every later card/doc use verbatim.

- [ ] **Step 1: Write the ADR**

Create `docs/adr/0018-publish-nc-families-as-restricted-model-packages.md` (format mirrors `docs/adr/0017-*.md`: `status` frontmatter, statement, Consequences, Considered Options):

```markdown
---
status: accepted
---

# Publish NC-licensed families as Restricted Model Packages

A Model Family whose weights carry a non-commercial or otherwise
redistribution-restricted grant can pass the same Port Validation Suite as any
other family, but its packages must not be called Published Model Packages:
that term carries docs/scope.md's promise of weights embeddable in other
people's programs. synthesize.cpp therefore defines a second publication
category, the Restricted Model Package: identical repository layout, model-card
workflow, validation evidence, and acquisition flow, plus the complete upstream
terms carried with the artifact.

The deciding argument, recorded from the OmniVoice fourth-family intake
(2026-07-30): a license restriction follows the weights regardless of who runs
the converter. A user converting locally holds the same restricted derivative,
so withholding publication does not enlarge the legal audience — it only adds
friction for the users the license does permit. When the upstream terms allow
non-commercial redistribution with attribution, the project may republish under
exactly those terms.

## Consequences

- A Restricted Model Package passes the full Port Validation Suite before
  publication, like any package. Validation Levels are unchanged.
- Its model card declares the restrictive license in frontmatter, quotes the
  upstream statement verbatim, and carries a prominent statement that the
  weights are not usable in commercial products. When upstream omits a license
  version, the card says so rather than inventing one.
- Every license in the artifact travels with it: for OmniVoice that is the
  CC-BY-NC statement for the LM weights and the Boson Higgs Audio 2 Community
  License text as a declared Sidecar Resource for the codec weights, with dual
  attribution.
- The C interface, runtime, and Adapters are unaffected: restriction is a
  publication and documentation fact, not a runtime capability.
- docs/scope.md's "Publishing validated, directly loadable Model Packages"
  line continues to mean Published Model Packages; Restricted Model Packages
  are an addition, not a reinterpretation.
- Publication of any package, restricted or not, remains a separate act
  requiring jiangzhuo's explicit per-act confirmation.

## Considered Options

- Supported Model Family only (no published weights) was rejected because it
  does not change who may legally use the weights; it only pushes a Python
  conversion environment onto every legitimate non-commercial user.
- Publishing under the ordinary Published Model Package label with a license
  footnote was rejected because that term's embedding promise would become
  conditionally false, and a term that is sometimes false is not a term.
- Waiting for upstream relicensing was rejected as indefinite: the restriction
  derives from training data (Emilia), which retraining alone could lift.
```

- [ ] **Step 2: Add the term to CONTEXT.md**

Read `CONTEXT.md`, find the vocabulary entry for "Published Model Package", and add a sibling entry immediately after it, matching the file's exact entry format:

> **Restricted Model Package** — A Model Package that passed the Port Validation Suite and is released through a project-owned repository **with restrictive upstream weight terms carried in full** (e.g. CC-BY-NC). It is not a Published Model Package and never carries the embeddable-in-other-programs promise. Avoid: "published" for these packages; say "released as a Restricted Model Package". Decision: ADR 0018.

Adapt wording to CONTEXT.md's house style (it uses avoid-lists; keep those). Update the file's `Status: Confirmed …` date line to 2026-07-30 if the file carries one.

- [ ] **Step 3: Update the selection doc**

In `docs/model-family-selection.md`, the "Fourth Model Family Candidate" section ends with the open question ("Whether the project supports a family it can never publish is an open decision that a fourth-family intake must settle before conversion work begins."). Append a short dated resolution paragraph after it:

```markdown
Resolved 2026-07-30: OmniVoice is the fourth Model Family. The question is
settled by ADR 0018 — the family ships as a Restricted Model Package, carrying
the upstream CC-BY-NC statement for the LM weights and the Boson Higgs Audio 2
Community License for the bundled codec weights (a third license the original
deferral note did not record; it lives at `audio_tokenizer/LICENSE` inside the
weights repository). The porting record is `docs/porting/families/omnivoice.md`.
```

Update the doc's `Status: Confirmed …` date to 2026-07-30.

- [ ] **Step 4: Commit**

```bash
git add docs/adr/0018-publish-nc-families-as-restricted-model-packages.md CONTEXT.md docs/model-family-selection.md
git commit -m "Define the Restricted Model Package category for NC-licensed families (ADR 0018)"
```

---

### Task 2: Family contract doc `docs/porting/families/omnivoice.md`

**Files:**
- Create: `docs/porting/families/omnivoice.md`

**Interfaces:**
- Produces: the family contract every later task cites (`docs/porting/families/omnivoice.md#reference-contract` is the manifest cases' `origin.locator`). Task 4 appends measured intake sections to this same file.

- [ ] **Step 1: Write the doc**

Mirror the section order of `docs/porting/families/qwen3-tts.md` (Decision → Reference Contract → Architecture → Port Validation Fit → Text Frontend → Quantization shape → GGML Operator Surface → Decomposition → Prior Art → Open Questions), writing only what is design-known now; intake-measured sections are appended by Task 4. Required content (write as prose, not bullets-of-bullets):

- **Status line:** `Status: Confirmed 2026-07-30. Intake in progress; stages 2+ not started.`
- **Decision:** fourth family = OmniVoice (Xiaomi / k2-fsa), Reference Model Variant `k2-fsa/OmniVoice` (flagship). `OmniVoice-Emilia` not ported (zh+en research variant; its card carries no license text at all). Publication ceiling: Restricted Model Package per ADR 0018; three licenses quoted verbatim (CC-BY-NC prose sentence; Boson Higgs Audio 2 Community License at `audio_tokenizer/LICENSE`; Apache-2.0 code). Note that all three known community GGML ports misstate or omit these terms, and that license facts come from the upstream card only.
- **Reference Contract:** oracle = pinned `k2-fsa/OmniVoice` PyPI/git package driving the pinned HF weights, F32 on CPU (checkpoint dtype is F32 throughout — contrast with qwen3-tts's BF16 story), entry point `OmniVoice.generate` with `position_temperature=0.0, class_temperature=0.0` for deterministic cases; **with both temperatures zero the loop makes no RNG call at all**, so greedy runs are bit-reproducible across processes. All oracle runs pin `postprocess_output=False, pad_duration=0.0, fade_duration=0.0` (no pydub in the contract) and clone cases pin `preprocess_prompt=False` + explicit `ref_text` (no silence-stripping, no Whisper). Revisions/SHA-256 get filled by Task 4.
- **Architecture:** the §1 summary from the design spec, plus the exact prompt layout: `[<|denoise|>?] <|lang_start|>{code-or-None}<|lang_end|> <|instruct_start|>{instruct-or-None}<|instruct_end|> <|text_start|>{ref_text + " " + text}<|text_end|> [ref audio tokens] [T × mask(1024)]`, every row of the 8-codebook dim repeating the same text ids, embedding merge = text row 0 + sum of 8 offset codebook embeddings, `<|denoise|>` only in clone mode. Decode loop: cond + uncond (target-region-only) batch, per-step schedule `ceil(total_mask * Δt)` with shifted timesteps `t' = t_shift·t / (1+(t_shift−1)·t)`, CFG in log-softmax space `log_softmax(log p_c + s·(log p_c − log p_u))`, mask id banned before argmax, confidence = max log-prob, layer penalty `layer_index × 5.0`, flat top-k commit, committed tokens fed back into both branches. (`audio_codebook_weights` [8,8,6,6,4,4,2,2] is training-loss weighting only — one reference port misdocuments it as the sampler's penalty; do not repeat that.)
- **Port Validation Fit:** greedy decoding is fully deterministic ⇒ golden cases assert the exact 8×T token grid equals the oracle's (this family's `structural_exactness` check), plus waveform through the codec under committed tolerances; equal-arithmetic comparison (`source-f32-oracle-vs-f32-cpu`). Public sampled path (default `position_temperature=5.0`, Gumbel over global torch RNG upstream — **no seed parameter exists upstream**) is validated by the seed contract like qwen3-tts, never by RNG reproduction. Gumbel replay seam is a fallback, implemented only if needed.
- **Text Frontend:** the same Qwen2 byte-level BPE and the same pre-tokenizer pattern as qwen3-tts (verified: identical regex in `tokenizer.json`), provider id stays `synthesize.qwen_bpe`; the shared implementation is hoisted (Task 8). No BOS. Seven OmniVoice special tokens at ids 151669–151675. Nonverbal tags (13 bracketed tags) are tokenized standalone. The 646-entry language-name map is NOT ported: the public interface speaks BCP-47 tags, the prompt consumes ISO code text, and for the validated catalog (en, zh, ja) tag == code, so the request tag is written into the prompt directly.
- **Duration:** RuleDurationEstimator ported as deterministic host code (weights table + formula reproduced in this doc verbatim from `omnivoice/utils/duration.py`); auto/design anchor is `("Nice to meet you.", 25 tokens)`; `speaking_rate` divides the estimate; explicit duration is not exposed in v1.
- **Delivery and limits:** complete audio only; no chunking claims; `max_output_frames = 750` (30 s) because upstream's >30 s path is chunked long-form synthesis, out of scope v1; output post-processing (silence removal, fades, padding) is upstream UX, not part of this package's contract.
- **GGML Operator Surface:** zero new operators — Snake (alpha-only, `x + sin²(αx)/α`) from primitives, conv1d = im2col+matmul, transposed conv = mul_mat + `ggml_col2im_1d` (the VITS recipe), argmax/rank/commit on host CPU.
- **Prior Art:** the three ports with licenses and one-line verdicts (from the design spec §9), plus `0xShug0/audio.cpp` (license unverified, read-only).
- **Open Questions:** generator-on-CUDA (stage 7 evidence); quantized profiles vs argmax cascade (stage 6 evidence); Gumbel replay seam (only if a sampled-parity case proves necessary); voice-design instruct passthrough (v1 does not reimplement `_resolve_instruct` normalization — oracle cases pin already-normalized instruct strings).

- [ ] **Step 2: Commit**

```bash
git add docs/porting/families/omnivoice.md
git commit -m "Add the OmniVoice family contract doc (design-known content)"
```

---

### Task 3: Locked oracle environment `scripts/envs/omnivoice/`

**Files:**
- Create: `scripts/envs/omnivoice/pyproject.toml`
- Create: `scripts/envs/omnivoice/uv.lock` (generated)

**Interfaces:**
- Produces: the env every `uv run --project scripts/envs/omnivoice --locked` invocation in Tasks 4–7 uses; the pinned upstream revision `$OMNI_REV` recorded here is reused by the manifest, intake.json, and converter metadata.

- [ ] **Step 1: Pin the upstream revision**

```bash
git ls-remote https://github.com/k2-fsa/OmniVoice.git HEAD
```

Record the SHA as `OMNI_REV` (write it into `reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md` when Task 4 creates it; use the literal SHA in the files below, never a branch name).

- [ ] **Step 2: Write pyproject.toml**

Mirror `scripts/envs/qwen3-tts/pyproject.toml` exactly in shape:

```toml
[project]
name = "synthesize-omnivoice-reference"
version = "0.0.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    "gguf==0.19.0",
    "huggingface-hub==0.36.2",
    "numpy==1.26.4",
    "omnivoice @ git+https://github.com/k2-fsa/OmniVoice@<OMNI_REV>",
    "soundfile==0.13.1",
    "torch>=2.7,<3",
    "torchaudio",
]

[tool.uv]
package = false
```

Notes: substitute the literal `<OMNI_REV>` SHA. Start from this minimal set; if `uv lock` reports that `omnivoice` needs more (it depends on transformers etc. transitively — those resolve on their own), add explicit `==` pins only for packages the resolver leaves floating that appear in `omnivoice`'s own dependency list. Do NOT pin `huggingface-hub` lower than 0.34 (the qwen3-tts porting log records the transformers incompatibility).

- [ ] **Step 3: Lock and smoke it**

```bash
cd scripts/envs/omnivoice && uv lock && cd -
uv run --project scripts/envs/omnivoice --locked python -c "
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
import dataclasses
cfg = OmniVoiceGenerationConfig()
print(dataclasses.asdict(cfg))
assert cfg.num_step == 32 and cfg.guidance_scale == 2.0 and cfg.t_shift == 0.1
assert cfg.layer_penalty_factor == 5.0 and cfg.position_temperature == 5.0 and cfg.class_temperature == 0.0
print('generation defaults OK')
"
```

Expected: the dataclass dict prints and the asserts pass. If any assert fails, STOP — the pinned revision's defaults moved and the Global Constraints table plus family doc must be corrected first.

- [ ] **Step 4: Commit**

```bash
git add scripts/envs/omnivoice/pyproject.toml scripts/envs/omnivoice/uv.lock
git commit -m "Lock the OmniVoice reference environment"
```

---

### Task 4: Intake — weights, licenses, inventory, oracle smoke, intake.json

**Files:**
- Create: `reports/porting/omnivoice/omnivoice-0-6b/intake.json`
- Create: `reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`
- Modify: `docs/porting/families/omnivoice.md` (append measured sections; fill Reference Contract pins)

**Interfaces:**
- Consumes: `scripts/envs/omnivoice/` (Task 3).
- Produces: pinned `WEIGHTS_REV` + per-file SHA-256 set; the tensor-name inventory file `reports/porting/omnivoice/omnivoice-0-6b/tensor-inventory.json` that Tasks 7 and 10 read; measured greedy determinism evidence.

- [ ] **Step 1: Pin and download the weights**

```bash
uv run --project scripts/envs/omnivoice --locked python -c "
from huggingface_hub import HfApi
info = HfApi().model_info('k2-fsa/OmniVoice')
print(info.sha)
"
# record as WEIGHTS_REV, then:
uv run --project scripts/envs/omnivoice --locked \
  hf download k2-fsa/OmniVoice --revision <WEIGHTS_REV> --local-dir models/omnivoice-0-6b
sha256sum $(find models/omnivoice-0-6b -type f | sort)
```

`models/` is git-ignored; confirm with `git status` (nothing under models/ may appear).

- [ ] **Step 2: License audit at the pinned revision**

From the downloaded snapshot (not the live site): quote verbatim into `_porting-log.md` and `intake.json.license_audit`:
1. `models/omnivoice-0-6b/README.md` — the `## License` sentence ("Our code is released under the Apache 2.0 License. The pre-trained model is licensed under the CC-BY-NC due to constraints from its training data (e.g., Emilia)."), and confirm the frontmatter carries **no** `license:` key (`head -40 README.md`).
2. `models/omnivoice-0-6b/audio_tokenizer/LICENSE` — first line block ("BOSON HIGGS AUDIO 2 COMMUNITY LICENSE AGREEMENT…"), byte size, sha256.
3. The source repo LICENSE at `OMNI_REV`: `curl -sL https://raw.githubusercontent.com/k2-fsa/OmniVoice/<OMNI_REV>/LICENSE | head -5` (Apache-2.0).
4. The model-card Disclaimer section (unauthorized voice cloning prohibition) — quote it; the card must reproduce it at ship time.

- [ ] **Step 3: Tensor inventory**

Write a throwaway-free listing the later tasks consume:

```bash
uv run --project scripts/envs/omnivoice --locked python - <<'EOF'
import json
from safetensors import safe_open
inv = {}
for label, path in [("generator", "models/omnivoice-0-6b/model.safetensors"),
                    ("codec", "models/omnivoice-0-6b/audio_tokenizer/model.safetensors")]:
    with safe_open(path, framework="pt") as h:
        inv[label] = {k: {"shape": list(h.get_slice(k).get_shape()),
                          "dtype": str(h.get_slice(k).get_dtype())} for k in h.keys()}
with open("reports/porting/omnivoice/omnivoice-0-6b/tensor-inventory.json", "w") as f:
    json.dump(inv, f, indent=2, sort_keys=True)
print({k: len(v) for k, v in inv.items()})
EOF
```

Expected counts: generator 313 (312 weights + the I64 `codebook_layer_offsets` buffer), codec 527. Record prefix histograms (`llm.`, `audio_embeddings`, `audio_heads`; `semantic_model.`, `acoustic_encoder.`, `acoustic_decoder.`, `quantizer.`, `encoder_semantic.`, `decoder_semantic.`, `fc`, `fc1`, `fc2`) in `_porting-log.md`. Also record every dtype seen (expect F32 only, plus the one I64 buffer) and every tensor whose name length ≥ 58 after a hypothetical `codec.` prefix (input to Task 7's NAME_SHORTENINGS).

- [ ] **Step 4: Oracle smoke — greedy determinism across processes**

Run twice as separate processes; byte-compare:

```bash
mkdir -p build/goldens/omnivoice
for run in 1 2; do
uv run --project scripts/envs/omnivoice --locked python - <<EOF
import numpy as np, torch
from omnivoice import OmniVoice
model = OmniVoice.from_pretrained("models/omnivoice-0-6b", device_map="cpu", dtype=torch.float32)
audios = model.generate(
    text="OmniVoice speaks with one voice.", language="en",
    position_temperature=0.0, class_temperature=0.0,
    postprocess_output=False, pad_duration=0.0, fade_duration=0.0)
np.asarray(audios[0], dtype=np.float32).tofile("build/goldens/omnivoice/_smoke_run${run}.f32")
print(len(audios[0]))
EOF
done
cmp build/goldens/omnivoice/_smoke_run1.f32 build/goldens/omnivoice/_smoke_run2.f32 && echo BYTE-IDENTICAL
```

Expected: `BYTE-IDENTICAL`. Record frames, duration, wall seconds, RTF in `intake.json.oracle_smoke`. Also record one sampled run (defaults, no seeding) to confirm it differs — evidence for `stochastic: true`. While in this step, open the installed package's `_decode_and_post_process` source (`uv run --project scripts/envs/omnivoice --locked python -c "import inspect, omnivoice.models.omnivoice as m; print(inspect.getsource(m.OmniVoice._decode_and_post_process))"`) and record in `_porting-log.md` **exactly which scaling/normalization remains active when `postprocess_output=False, pad_duration=0, fade_duration=0`** (peak-normalize-to-0.5 and ref-RMS matching may be unconditional). Whatever remains active is part of this family's contract and Plan 2's port must mirror it; write the finding into the family doc's Delivery section.

- [ ] **Step 5: Pin the upstream example + a clone reference artifact**

The manifest (Task 5) needs at least one `upstream_example` case and one downloadable, pinned reference-audio artifact for clone cases. Enumerate candidates:

```bash
git clone --depth 1 https://github.com/k2-fsa/OmniVoice /tmp/omnivoice-src 2>/dev/null || true
git -C /tmp/omnivoice-src fetch --depth 1 origin <OMNI_REV> && git -C /tmp/omnivoice-src checkout <OMNI_REV>
find /tmp/omnivoice-src -name '*.wav' -o -name '*.flac' -o -name '*.mp3' | head
grep -rn "generate(" /tmp/omnivoice-src/README.md | head
```

Pin: (a) the README's first usage-example text as the upstream_example case input; (b) one upstream-hosted wav (repo example or HF Space asset — must be URL-addressable at a pinned revision) with sha256 as the clone reference artifact; record both in `intake.json` under `source.upstream_examples`. If the pinned revision truly ships no audio asset, use the HF Space `k2-fsa/OmniVoice` repo (`https://huggingface.co/spaces/k2-fsa/OmniVoice`) at a pinned revision, and record that provenance choice in `_porting-log.md`.

- [ ] **Step 6: Write intake.json**

Schema-by-example: `reports/porting/qwen3-tts/qwen3-tts-12hz-0-6b-customvoice/intake.json` (`"schema": "synthesize-port-intake-v1"`). Sections and their omnivoice content:

- `source`: repository `https://github.com/k2-fsa/OmniVoice`, revision `OMNI_REV`, package_version (from `pyproject.toml` at the pin), install "pinned by git revision in scripts/envs/omnivoice/pyproject.toml", license_file = Apache-2.0 LICENSE facts, papers: `arXiv:2604.00688`.
- `weights`: repository `https://huggingface.co/k2-fsa/OmniVoice`, revision `WEIGHTS_REV`, local_ignored_path `models/omnivoice-0-6b`, full per-file `{path, bytes, sha256}` list from Step 1.
- `architecture`: llm block (28 layers, hidden 1024, 16/8 heads, head_dim 128, ffn 3072, rms_eps 1e-6, rope_theta 1e6, vocab 151676, tie_word_embeddings true, **all 28 layer_types full_attention ⇒ bidirectional, no KV cache**); audio io (audio_vocab_size 1025, mask 1024, 8 codebooks, embeddings/heads [8200,1024], `codebook_layer_offsets` is a derivable buffer = `arange(8)*1025`, `audio_codebook_weights` [8,8,6,6,4,4,2,2] is training-loss-only); codec (Higgs Audio V2: sample_rate 24000, hop 960, frame_rate 25, semantic_sample_rate 16000, HuBERT 768d/12L/7 conv layers `conv_dim 512×7, kernel [10,3,3,3,3,2,2], stride [5,2,2,2,2,2,2]`, DAC ratios [8,5,4,2,3], decoder_hidden 1024, codebook_dim 64, codebook_size 1024 — and the config-vs-tensors caveat: `acoustic_model_config.n_codebooks` says 9, the shipped RVQ carries 8; **size from tensors**); `tensor_inventory` (counts + prefix histogram from Step 3); `trust_remote_code_required: false`; `checkpoint_dtypes`: all F32 (+1 I64 buffer).
- `stochastic_behavior`: sources = position Gumbel (default temp 5.0) and class Gumbel (default 0.0); measured greedy byte-identity (Step 4), sampled runs differ; **no upstream seed parameter — only the global torch RNG**; consequence: package is stochastic, public seeding is this port's own stream, exact-token goldens run greedy.
- `license_audit`: the three grants with verbatim evidence and `audited_at_pinned_revision: true`; `restriction_scan` noting the CC-BY-NC sentence and the Boson license; `project_policy`: local conversion allowed, any upload requires explicit per-act confirmation, publication category = Restricted Model Package (ADR 0018).
- `frontend_requirements`: Qwen2 byte-level BPE from `tokenizer.json` (single file — no vocab.json/merges.txt), 151,643 base + 33 added = 151,676, same pre-tokenizer pattern as qwen3-tts, provider `synthesize.qwen_bpe`, no BOS, the seven TTS markers 151669–151675.
- `reference_environment`: host facts, uv version, python/torch/transformers versions from `uv run … python -c "import torch, transformers; …"`, dtype float32, device cpu, pyproject/uv.lock sha256s.
- `oracle_smoke`: Step 4 numbers.
- `audio_delivery_claim`: stage-1 claim `complete_audio` (not even chunked delivery is claimed), reason: NAR whole-canvas synthesis; long-form is upstream-side text chunking, out of scope v1.
- `open_decisions`: generator-on-CUDA token identity; quantization vs argmax cascade; Gumbel replay seam; residual output scaling finding from Step 4; instruct passthrough (no `_resolve_instruct` port).

- [ ] **Step 7: Start `_porting-log.md`, append measured sections to the family doc**

`_porting-log.md` format: `# omnivoice-0-6b Porting Log`, then `## 2026-MM-DD — Intake and oracle smoke` with evidence bullets (mirror the qwen3-tts log's voice). Fill the family doc's Reference Contract pins (revisions, byte counts, digests pointer) and append an "Intake Measurements" section (greedy determinism, dtype table, inventory counts, the post-processing finding).

- [ ] **Step 8: Commit**

```bash
git add reports/porting/omnivoice docs/porting/families/omnivoice.md
git commit -m "Complete OmniVoice intake: pins, licenses, inventory, greedy-determinism evidence"
```

---

### Task 5: Golden manifest + tolerance skeleton (+ schema extension)

**Files:**
- Create: `tests/golden/omnivoice/omnivoice-0-6b.manifest.json`
- Create: `tests/tolerances/omnivoice.json`
- Modify (only if required): `docs/schemas/synthesize-golden-manifest-v1.schema.json`
- Test: existing `tests/python/test_golden_manifests.py` (self-registering via glob)

**Interfaces:**
- Consumes: pins and example/reference artifacts from Task 4.
- Produces: the case list Task 6's dumper iterates; `case_artifact_root: build/goldens/omnivoice`; artifact names/paths Plans 2–4 compare against.

- [ ] **Step 1: Read the schema and check three fields**

Read `docs/schemas/synthesize-golden-manifest-v1.schema.json`. Verify whether it constrains: (a) `package_contract.voices.mode` to an enum (qwen3-tts uses `preset-catalog`); (b) `case.voice.kind` to an enum (`preset_voice` seen); (c) any structure for reference-audio case inputs. Where enums exist, extend them additively: add `"profile-sources"` to voices.mode; add `"package_default"`, `"description_text"`, `"reference_audio"` to voice.kind; if reference audio needs a per-case artifact slot, model it as the existing optional `input.artifact` mechanics (`test_golden_manifests.py` already path-checks `input.artifact`). Keep the schema `v1` — these are additive enum widenings for the first profile-source family, and the contract test (`test_manifests_conform_to_schema`) is the gate. If nothing is enumerated, touch nothing.

- [ ] **Step 2: Write the manifest top-level**

```json
{
  "$schema": "../../../docs/schemas/synthesize-golden-manifest-v1.schema.json",
  "schema": "synthesize-golden-manifest-v1",
  "suite_version": 1,
  "family": "omnivoice",
  "variant": "omnivoice-0-6b",
  "source": {
    "repository": "https://github.com/k2-fsa/OmniVoice",
    "revision": "<OMNI_REV>",
    "artifacts": [
      { "role": "checkpoint", "locator": "https://huggingface.co/k2-fsa/OmniVoice/resolve/<WEIGHTS_REV>/model.safetensors", "sha256": "<from intake>" },
      { "role": "checkpoint", "locator": "https://huggingface.co/k2-fsa/OmniVoice/resolve/<WEIGHTS_REV>/audio_tokenizer/model.safetensors", "sha256": "<from intake>" },
      { "role": "config", "locator": "https://huggingface.co/k2-fsa/OmniVoice/resolve/<WEIGHTS_REV>/config.json", "sha256": "<from intake>" },
      { "role": "config", "locator": "https://huggingface.co/k2-fsa/OmniVoice/resolve/<WEIGHTS_REV>/audio_tokenizer/config.json", "sha256": "<from intake>" },
      { "role": "frontend-resource", "locator": "https://huggingface.co/k2-fsa/OmniVoice/resolve/<WEIGHTS_REV>/tokenizer.json", "sha256": "<from intake>" },
      { "role": "license", "locator": "https://raw.githubusercontent.com/k2-fsa/OmniVoice/<OMNI_REV>/LICENSE", "sha256": "<from intake>" },
      { "role": "license", "locator": "https://huggingface.co/k2-fsa/OmniVoice/resolve/<WEIGHTS_REV>/audio_tokenizer/LICENSE", "sha256": "<from intake>" },
      { "role": "reference-audio", "locator": "<pinned clone reference URL from intake>", "sha256": "<from intake>" }
    ]
  },
  "reference": {
    "implementation": "k2-fsa/OmniVoice@<OMNI_REV>",
    "framework": "pytorch-<torch version from intake>",
    "runner": "scripts/dump_reference_omnivoice_pytorch.py",
    "environment_lock": "scripts/envs/omnivoice/uv.lock",
    "dtype": "float32",
    "device": "cpu"
  },
  "package_contract": {
    "native_audio": { "sample_rate_hz": 24000, "channels": 1, "sample_format": "f32le" },
    "input_kinds": [ "text_utf8" ],
    "frontend": { "provider": "synthesize.qwen_bpe", "contract_version": 1 },
    "language_tags": [ "auto", "en", "zh", "ja" ],
    "voices": { "mode": "profile-sources", "default_id": null, "preset_ids": [] },
    "stochastic": true,
    "speaking_rate_range": [ 0.5, 2.0 ],
    "max_input_tokens": 2048,
    "max_output_frames": 750
  },
  "tolerance_file": "tests/tolerances/omnivoice.json",
  "case_artifact_root": "build/goldens/omnivoice",
  "cases": [ ],
  "relations": [ ]
}
```

(`role: "reference-audio"` — check the schema's role enum in Step 1 and widen additively if needed.)

- [ ] **Step 3: Write the 20 cases**

Three complete templates, then the case table. Every case carries the same 8 `checks` as qwen3-tts (`tensor_parity, structural_exactness, waveform_regression, finite_pcm, request_repeatability, result_metadata, backend_placement, resource_cleanup`). For this family `structural_exactness` is defined as **exact token-grid equality** (documented in the family doc). Every case declares `oracle.stochastic_inputs = [ { "name": "codes.grid", "path": "codes/grid.i32", "format": "i32le" } ]` — for greedy cases the grid is deterministic and doubles as the exact-equality target; for sampled cases it is the replay artifact feeding the codec comparison (this satisfies `test_stochastic_packages_declare_replay_inputs`).

**Template A — greedy auto-voice case:**

```json
{
  "id": "omni-short-en",
  "origin": { "kind": "project_coverage", "locator": "docs/porting/families/omnivoice.md#reference-contract" },
  "coverage": [ "short-sequence", "auto-voice", "greedy-exact" ],
  "input": { "kind": "text_utf8", "language_tag": "en", "text": "OmniVoice speaks with one voice." },
  "voice": { "kind": "package_default" },
  "request": { "seed_u64": "0", "speaking_rate": 1.0, "max_output_frames": 0 },
  "oracle": {
    "parameters": {
      "num_step": 32, "guidance_scale": 2.0, "t_shift": 0.1, "layer_penalty_factor": 5.0,
      "position_temperature": 0.0, "class_temperature": 0.0,
      "language": "en", "instruct": null,
      "postprocess_output": false, "pad_duration": 0.0, "fade_duration": 0.0
    },
    "stochastic_inputs": [ { "name": "codes.grid", "path": "codes/grid.i32", "format": "i32le" } ]
  },
  "checks": [ "tensor_parity", "structural_exactness", "waveform_regression", "finite_pcm",
              "request_repeatability", "result_metadata", "backend_placement", "resource_cleanup" ],
  "expected": {
    "status": "ok",
    "artifacts": [
      { "name": "input.token_ids", "path": "input/token_ids.i32", "format": "i32le" },
      { "name": "prompt.token_grid", "path": "input/prompt_grid.i32", "format": "i32le" },
      { "name": "generator.hidden_l0", "path": "generator/hidden_l0.f32", "format": "f32le" },
      { "name": "generator.hidden_l7", "path": "generator/hidden_l7.f32", "format": "f32le" },
      { "name": "generator.hidden_l14", "path": "generator/hidden_l14.f32", "format": "f32le" },
      { "name": "generator.hidden_l21", "path": "generator/hidden_l21.f32", "format": "f32le" },
      { "name": "generator.hidden_l27", "path": "generator/hidden_l27.f32", "format": "f32le" },
      { "name": "generator.final", "path": "generator/final.f32", "format": "f32le" },
      { "name": "generator.logits_step0", "path": "generator/logits_step0.f32", "format": "f32le" },
      { "name": "codes.grid", "path": "codes/grid.i32", "format": "i32le" },
      { "name": "audio.pcm", "path": "audio/pcm.f32", "format": "f32le" },
      { "name": "result", "path": "result.json", "format": "json" },
      { "name": "metadata", "path": "metadata.json", "format": "json" }
    ]
  }
}
```

(Hidden/final/logits probes are captured on the **step-0 conditional forward** — the one pass reproducible without any committed history; `prompt.token_grid` is the full 8×S conditional input at step 0.)

**Template B — greedy clone case (adds three artifacts and reference fields):** same as A plus `"input"` gains `"reference": { "artifact": "<manifest source artifact locator for the reference wav>", "transcript": "<its exact transcript>", "language_tag": "en" }` (field name per the Step 1 schema decision), `"voice": { "kind": "reference_audio" }`, oracle.parameters gains `"denoise": true, "preprocess_prompt": false, "ref_text": "<same transcript>"`, and `expected.artifacts` gains:

```json
      { "name": "ref.pcm_24k", "path": "ref/pcm_24k.f32", "format": "f32le" },
      { "name": "ref.tokens", "path": "ref/tokens.i32", "format": "i32le" },
      { "name": "ref.semantic_hidden", "path": "ref/semantic_hidden.f32", "format": "f32le" }
```

**Template C — sampled seed case:** same as A but `"coverage": ["sampled-path", "seed-contract"]`, `request.seed_u64` per the table, and `oracle.parameters` gains `"position_temperature": 5.0` and `"seeded_by": "request.seed_u64"`. Keep the same 8 checks and the same artifact list as A: tensor parity on a sampled case compares the codec path on the replayed grid, and the step-0 probes are still deterministic because step 0 sees only the prompt — the qwen3-tts model exactly.

**Case table (20 cases; all greedy/32-step unless stated):**

| id | lang | voice | text/notes | rate | seed |
|---|---|---|---|---|---|
| omni-upstream-readme | as in README | package_default | the pinned README example text; `origin.kind: "upstream_example"`, locator = README URL at WEIGHTS_REV | 1.0 | 0 |
| omni-short-en | en | package_default | Template A verbatim | 1.0 | 0 |
| omni-short-zh | zh | package_default | "欢迎使用语音合成引擎。" | 1.0 | 0 |
| omni-short-ja | ja | package_default | "音声合成へようこそ。" | 1.0 | 0 |
| omni-lang-none | (omit language → oracle language null, prompt "None") | package_default | "This request names no language." — coverage "language-agnostic" | 1.0 | 0 |
| omni-punctuation | en | package_default | "Wait — really?! Yes; truly: it works... (mostly)." | 1.0 | 0 |
| omni-digits | en | package_default | "In 2026 the model counted 1234567890 samples." | 1.0 | 0 |
| omni-nonverbal | en | package_default | "That is funny [laughter] but let me think." — coverage "nonverbal-tag" | 1.0 | 0 |
| omni-medium-en | en | package_default | 3–4 sentences ≈ 12 s | 1.0 | 0 |
| omni-long-boundary | en | package_default | text sized so the duration estimate lands in 650–740 frames (compute with the ported estimator once Task 11 exists; until then size by the weight table by hand) — coverage "near-frame-limit" | 1.0 | 0 |
| omni-rate-slow | en | package_default | same text as omni-short-en | 0.5 | 0 |
| omni-rate-fast | en | package_default | same text as omni-short-en | 2.0 | 0 |
| omni-design-en | en | description_text: "female, young, high pitch" | "Voice design shapes who is speaking." | 1.0 | 0 |
| omni-design-zh | zh | description_text: "男, 年长, 低沉" (pin the exact normalized form the oracle's `_resolve_instruct` emits — run it once and copy) | "语音设计决定说话人。" | 1.0 | 0 |
| omni-clone-en | en | reference_audio (Template B) | "A cloned voice reads this sentence." | 1.0 | 0 |
| omni-clone-zh | zh | reference_audio (same reference) | "克隆的声音读出这句话。" | 1.0 | 0 |
| omni-fast-mode | en | package_default | same text as omni-short-en, oracle num_step: 16 — coverage "fast-mode" | 1.0 | 0 |
| omni-sampled-seed-zero | en | package_default | Template C, "Sampling follows the seed." | 1.0 | 0 |
| omni-sampled-seed-one | en | package_default | Template C, same text | 1.0 | 1 |
| omni-sampled-seed-forty-two | en | package_default | Template C, same text | 1.0 | 42 |

**Relations:**

```json
"relations": [
  { "kind": "artifact_differs", "phase": "public_request",
    "cases": [ "omni-sampled-seed-zero", "omni-sampled-seed-one", "omni-sampled-seed-forty-two" ],
    "artifact": "audio.pcm" },
  { "kind": "artifact_differs", "phase": "oracle_replay",
    "cases": [ "omni-short-en", "omni-design-en", "omni-clone-en" ],
    "artifact": "audio.pcm" },
  { "kind": "artifact_differs", "phase": "oracle_replay",
    "cases": [ "omni-rate-slow", "omni-short-en", "omni-rate-fast" ],
    "artifact": "audio.pcm" }
]
```

- [ ] **Step 4: Write the tolerance skeleton**

`tests/tolerances/omnivoice.json` — mirror `tests/tolerances/qwen3-tts.json`'s structure but with an explicit not-yet-measured status so the validators' `--check` refuses to run (a tolerance is an input, not an output):

```json
{
  "schema": "synthesize-tolerances-v1",
  "family": "omnivoice",
  "variant": "omnivoice-0-6b",
  "status": "pending-first-measurement",
  "reference_stage": "source-f32-oracle-vs-f32-cpu",
  "profiles": {}
}
```

First read `tests/tolerances/qwen3-tts.json` and copy its actual top-level schema/keys exactly (the block above is the intent, not necessarily the literal key set — the qwen3-tts file is authoritative for shape; keep `status` and empty `profiles`).

- [ ] **Step 5: Run the contract test to verify it fails, then passes**

```bash
ctest --test-dir build --output-on-failure -R synthesize-golden-manifest-contract
```

Run once BEFORE committing the manifest with a deliberately missing `tolerance_file` path to see the failure mode (`test_tolerance_file_is_committed` fails), then fix and re-run. Expected final state: PASS. (Configure the build first if needed: the standard configure line from CLAUDE.md.)

- [ ] **Step 6: Commit**

```bash
git add tests/golden/omnivoice tests/tolerances/omnivoice.json
git add docs/schemas/synthesize-golden-manifest-v1.schema.json   # only if Step 1 modified it
git commit -m "Commit the omnivoice golden manifest, tolerance skeleton, and schema widening"
```

---

### Task 6: Oracle dumpers

**Files:**
- Create: `scripts/dump_reference_omnivoice_pytorch.py`
- Create: `scripts/dump_reference_omnivoice_tokenizer.py`

**Interfaces:**
- Consumes: manifest (Task 5), weights (Task 4), env (Task 3).
- Produces: per-case artifacts under `build/goldens/omnivoice/<case-id>/` with exactly the manifest's `expected.artifacts` paths; `build/goldens/omnivoice/tokenizer/cases.json` consumed by Task 11's unit tests.

- [ ] **Step 1: Write the tokenizer/duration dumper**

`scripts/dump_reference_omnivoice_tokenizer.py` (runs in the omnivoice env). It must emit, as one JSON file `build/goldens/omnivoice/tokenizer/cases.json`:

1. **Pre-tokenizer splits**: for a fixed list of ~12 strings (ASCII, contractions, digits, CJK, kana, mixed punctuation, leading/trailing spaces, newlines, the nonverbal-tag sentence), the exact piece list from the HF tokenizer's pre-tokenizer: `tok._tokenizer.pre_tokenizer.pre_tokenize_str(s)`.
2. **Token ids**: for the same strings plus every manifest case's full prompt strings: `style_text`, `wrapped_text` (built exactly as `_prepare_inference_inputs` does — import and call the private helpers where importable, else reproduce the two f-strings from the family doc), tokenized with `add_special_tokens=False`, and via `_tokenize_with_nonverbal_tags` for the wrapped text.
3. **Duration estimates**: for every manifest case: `calculate_total_weight(text)`, and `_estimate_target_tokens(text, ref_text, num_ref_audio_tokens, speed)` inputs/outputs — for no-ref cases the anchor is `("Nice to meet you.", 25)`; include speed 0.5/1.0/2.0 rows for the rate cases.
4. **Special token ids**: the 7 markers + eos/pad, read from the tokenizer, asserted equal to 151669–151675.

Structure the script like `scripts/dump_reference_qwen3_tts_pytorch.py` (argparse `--weights-dir`, `--output`; SystemExit on any mismatch with the pinned expectations). Run it; eyeball `cases.json`; the ids for `omni-design-zh`'s normalized instruct string get copied back into the manifest now (see the case table note).

- [ ] **Step 2: Write the main dumper**

`scripts/dump_reference_omnivoice_pytorch.py`, mirroring the structure of `scripts/dump_reference_qwen3_tts_pytorch.py` (same CLI: `--manifest --weights-dir --output-root --case --report`; same manifest guards with `family == "omnivoice"`; same `write_f32/write_i32/write_json` helpers and per-case `result.json`/`metadata.json`). Family-specific capture:

- Model load: `OmniVoice.from_pretrained(str(args.weights_dir), device_map="cpu", dtype=torch.float32)`.
- **Module-path discovery first** (one-time, printed): `print(model)` truncated — confirm the LLM attribute path (expected `model.llm` holding a `Qwen3ForCausalLM` or `Qwen3Model`; layers at `model.llm.model.layers[i]` or `model.llm.layers[i]`) and adjust the hook wiring to the real paths. Do not guess silently: assert the hooked module count == 5 probe layers + 1 final norm.
- **Step-0 probes**: forward hooks on llm layers {0,7,14,21,27} and the final norm, keeping only the FIRST forward call's **conditional row** (batch row 0) outputs, sliced to the conditional sequence; a hook on `audio_heads` (or capture of the returned logits) for `generator/logits_step0.f32` — save as float32 in `[C, S, V]` order flattened C-major, and record the shape in `metadata.json`.
- **Prompt grid**: after `_prepare_inference_inputs` equivalents run, save the 8×S conditional `input_ids` as `input/prompt_grid.i32` and row 0's text-region ids as `input/token_ids.i32`.
- **Generation**: call `model.generate(...)` with the case's `oracle.parameters` verbatim (language/instruct/ref handling per mode; for sampled cases `torch.manual_seed(int(case.request.seed_u64))` immediately before). Intercept the final token grid: wrap `model.audio_tokenizer.decode` (monkey-wrap like qwen3's `capture_codes`) to capture the `(1, 8, T)` codes → `codes/grid.i32` (int32, codebook-major `[8, T]`), and the decoded waveform reaching the caller → `audio/pcm.f32`.
- **Clone cases**: build the prompt via `model.create_voice_clone_prompt((wav_tensor, sr), ref_text=case_ref_text, preprocess_prompt=False)` where `wav_tensor` is the pinned reference file loaded with soundfile; save the post-resample 24 kHz mono float PCM (`ref/pcm_24k.f32` — capture the exact array passed to `audio_tokenizer.encode`, hop-clipped), the returned `ref_audio_tokens` (`ref/tokens.i32`), and a forward hook on the HuBERT model's final layer output during encode (`ref/semantic_hidden.f32`).
- **Guards**: PCM finite; `audio.shape[0] == grid.shape[1] * 960` (asserts codes↔audio tie); grid values in `[0, 1023]` (mask id must never survive); greedy cases run the generation **twice in-process** and assert the two grids equal (cheap determinism tripwire).

- [ ] **Step 3: Dump two cases and inspect**

```bash
uv run --project scripts/envs/omnivoice --locked python scripts/dump_reference_omnivoice_pytorch.py \
  --manifest tests/golden/omnivoice/omnivoice-0-6b.manifest.json \
  --weights-dir models/omnivoice-0-6b --case omni-short-en --case omni-clone-en
ls -laR build/goldens/omnivoice/omni-short-en build/goldens/omnivoice/omni-clone-en
```

Expected: every `expected.artifacts` path from the manifest exists, nothing else; `result.json` has finite peak and plausible duration. Listen to `audio/pcm.f32` once (`ffplay -f f32le -ar 24000 -ac 1 …` or convert to wav) — it must be intelligible speech.

- [ ] **Step 4: Dump the full suite**

Run without `--case` (all 20). Expect tens of minutes on CPU. Record wall time per case in the dumper report (`--report build/goldens/omnivoice/dump-report.json`). If `omni-long-boundary`'s estimate falls outside 650–740 frames, adjust that case's text now and re-dump only it.

- [ ] **Step 5: Commit (scripts only — artifacts stay out of git)**

```bash
git status   # verify build/ and models/ untracked
git add scripts/dump_reference_omnivoice_pytorch.py scripts/dump_reference_omnivoice_tokenizer.py \
        tests/golden/omnivoice/omnivoice-0-6b.manifest.json
git commit -m "Add the OmniVoice oracle dumpers and pin oracle-derived manifest values"
```

---

### Task 7: Converter + Python unit tests

**Files:**
- Create: `scripts/convert-omnivoice.py`
- Test: `tests/python/test_convert_omnivoice.py`

**Interfaces:**
- Consumes: manifest, weights, `tensor-inventory.json`, env; shared helpers `scripts/lib/gguf_common.py` (`add_general_identity, atomic_output_path, project_relative, sha256_file, write_json_atomic`).
- Produces: `models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf` + report `reports/convert/omnivoice/omnivoice-0-6b-F32.json`; the complete GGUF key set that Task 9's C++ reader and Task 10's catalog consume (listed exhaustively below — keep the three in lockstep).

- [ ] **Step 1: Write the failing tests first**

`tests/python/test_convert_omnivoice.py`, using the exact `importlib` preamble from `tests/python/test_convert_qwen3_tts.py` (hyphenated filename load + `sys.modules` registration). Test classes/functions (write them ALL now; they define the converter's pure-python rules):

```python
class SkipRuleTests(unittest.TestCase):
    # fc1.*, decoder_semantic.* are dropped with a recorded reason; a checkpoint
    # where the drop pattern matches nothing raises (layout-moved tripwire).
    def test_drops_fc1_and_decoder_semantic(self): ...
    def test_missing_drop_targets_is_an_error(self): ...
    def test_codebook_layer_offsets_buffer_is_skipped_not_emitted(self): ...
    def test_ema_buffers_are_skipped_when_embed_exists(self): ...

class WeightNormFoldTests(unittest.TestCase):
    # HuBERT pos_conv weight-norm: w = g * v / ||v|| folded at conversion,
    # recorded under conversion.transformed, g/v consumed.
    def test_folds_g_v_into_a_plain_weight(self): ...
    def test_fold_matches_torch_weight_norm(self): ...   # parametrize vs torch.nn.utils.weight_norm
    def test_orphan_g_without_v_is_an_error(self): ...

class NameLengthTests(unittest.TestCase):
    def test_every_inventory_name_fits_after_prefixing(self): ...
    # iterate reports/porting/omnivoice/omnivoice-0-6b/tensor-inventory.json codec names
    # with the "codec." prefix through shorten_name; every result < 64.
    def test_overflowing_name_without_a_rule_stops_conversion(self): ...

class DtypeTests(unittest.TestCase):
    def test_f32_stays_f32(self): ...
    def test_i64_and_other_dtypes_are_rejected(self): ...

class GenerationDefaultsTests(unittest.TestCase):
    # The defaults come from OmniVoiceGenerationConfig, not a literal table.
    def test_defaults_are_read_from_the_upstream_dataclass(self): ...
    def test_zero_num_step_is_refused(self): ...
```

Import of `omnivoice` inside the defaults tests must be lazy/skipped-if-absent (`unittest.skipUnless(importlib.util.find_spec("omnivoice"), ...)`) because this suite runs under the VITS env — same reason the converter itself imports `safetensors` and `omnivoice` lazily inside functions.

- [ ] **Step 2: Run to verify failure**

```bash
ctest --test-dir build --output-on-failure -R synthesize-vits-python-unit
```

Expected: FAIL — `convert-omnivoice.py` does not exist.

- [ ] **Step 3: Write the converter**

`scripts/convert-omnivoice.py`, modeled line-for-line on `scripts/convert-qwen3-tts.py`'s skeleton (same imports, `ConverterError`, `OutputTensor`/`Conversion` dataclasses, `shorten_name`, `numpy_of` (F32-only + reject others), `atomic_output_path` + `verify_gguf` + report write). Constants:

```python
ARCH_KEY = "omnivoice"
FORMAT_VERSION = 1
PROFILE_NAME = "F32"
PROFILE_VERSION = 1
ARCHITECTURE_VERSION = 1
GGML_MAX_NAME = 64
NAME_SHORTENINGS = (
    # Derived from tensor-inventory.json in Task 4 Step 3; the known overflow
    # class is codec.semantic_model.encoder.layers.N.* paths. Shortest edit that
    # fixes each, e.g.:
    (".attention.", ".attn."),
    (".feed_forward.", ".ff."),
    (".final_layer_norm.", ".final_ln."),
    (".pos_conv_embed.", ".pos_conv."),
)
SAMPLE_RATE = 24000
HOP_LENGTH = 960
FRAME_RATE_HZ = 25.0
NUM_CODEBOOKS = 8
AUDIO_VOCAB = 1025
AUDIO_MASK_ID = 1024
```

(Verify each NAME_SHORTENINGS entry against the actual inventory before finalizing; the `test_every_inventory_name_fits_after_prefixing` test is the gate. Add or drop rules until it passes; every rule must fire on ≥1 real name or be removed.)

Conversion flow in `main()`:
1. Load manifest; require `family == "omnivoice"`. Load `config.json`, `audio_tokenizer/config.json`, `tokenizer.json` from `--weights-dir`. Compute digests, verify against the manifest pins.
2. `convert_file(generator_path, "", conversion, drop=("codebook_layer_offsets",))` — passthrough names `llm.*`, `audio_embeddings.weight`, `audio_heads.weight`; skip the I64 offsets buffer with a recorded reason ("derivable: arange(8)*1025; I64 tensors are not GGUF-portable").
3. `convert_file(codec_path, "codec.", conversion, drop_prefixes=("fc1.", "decoder_semantic."), skip_ema=True, fold_weight_norm=True)` — keep `semantic_model.*`, `acoustic_encoder.*`, `acoustic_decoder.*`, `quantizer.*`, `encoder_semantic.*`, `fc.*`, `fc2.*` (cloning ships in this package, so the encode path is carried — the opposite of qwen3-tts's encoder drop; say so in the module docstring). EMA-skip rule: for any codebook that has both an `embed`-style table and `cluster_size`/`embed_avg`-style accumulators, emit the table and skip the accumulators with reasons (exact upstream field names read from tensor-inventory.json; write the rule against those names and cover it in the tests).
4. Duplicate-name collision check; GGUF write; verify; report to `reports/convert/omnivoice/omnivoice-0-6b-F32.json` (same report schema as qwen3, `dtype_policy: "F32 checkpoint carried through unchanged"`).
5. **License carriage:** copy `audio_tokenizer/LICENSE` from `--weights-dir` to `<output dir>/LICENSE-higgs-audio-2.txt` (byte-identical; sha256 recorded in the report under a `licenses` list alongside the CC-BY-NC statement quoted verbatim). The formal declarative Sidecar Resource descriptors of `docs/model-packages.md` are assembled at the ship stage (Plan 4) with the rest of the publication directory; what Plan 1 guarantees is that the license text never separates from the converted artifact and its hash is pinned from the first cut.

`add_metadata` — the complete key set (this is the contract Tasks 9/10 read; keep byte-identical spellings):

```python
def add_metadata(writer, manifest, config, codec_config, tokenizer_json, gen_defaults, digests):
    llm = config["llm_config"]
    add_general_identity(
        writer,
        name="OmniVoice 0.6B",
        basename=manifest["variant"],
        size_label="0.6B",
        languages=[t for t in manifest["package_contract"]["language_tags"] if t != "auto"],
        tags=["text-to-speech", "omnivoice", "synthesize.cpp"],
        author="k2-fsa (Xiaomi)",
        organization="k2-fsa",
        source_url=manifest["source"]["repository"],
        description=(
            "Source-dtype synthesize.cpp conversion of the pinned k2-fsa/OmniVoice "
            "checkpoint: F32 mask-predict generator plus F32 Higgs Audio V2 codec."
        ),
        license_id="cc-by-nc-4.0",
        license_name="CC-BY-NC (version unstated upstream) + Boson Higgs Audio 2 Community License (codec)",
        license_link="https://huggingface.co/k2-fsa/OmniVoice",
    )
    writer.add_repo_url(manifest["source"]["repository"])

    writer.add_uint32("synthesize.format_version", FORMAT_VERSION)
    writer.add_string("synthesize.model_family", ARCH_KEY)
    writer.add_string("synthesize.model_variant", manifest["variant"])
    writer.add_string("synthesize.quantization.profile", PROFILE_NAME)
    writer.add_uint32("synthesize.quantization.profile_version", PROFILE_VERSION)
    writer.add_string("synthesize.source.repository", manifest["source"]["repository"])
    writer.add_string("synthesize.source.revision", manifest["source"]["revision"])
    writer.add_string("synthesize.source.checkpoint.sha256", digests["generator"])
    writer.add_string("synthesize.source.codec.sha256", digests["codec"])
    writer.add_string("synthesize.source.config.sha256", digests["config"])
    writer.add_string("synthesize.source.checkpoint.license_status",
                      "cc-by-nc (LM, version unstated) + boson-higgs-audio-2-community (codec) + apache-2.0 (code)")
    writer.add_string("synthesize.converter", "scripts/convert-omnivoice.py")
    writer.add_uint32("synthesize.omnivoice.architecture_version", ARCHITECTURE_VERSION)

    # Generation defaults: read from the pinned package, never restated.
    writer.add_uint32("synthesize.omnivoice.generation.num_step", int(gen_defaults["num_step"]))
    writer.add_float32("synthesize.omnivoice.generation.guidance_scale", float(gen_defaults["guidance_scale"]))
    writer.add_float32("synthesize.omnivoice.generation.t_shift", float(gen_defaults["t_shift"]))
    writer.add_float32("synthesize.omnivoice.generation.layer_penalty_factor", float(gen_defaults["layer_penalty_factor"]))
    writer.add_float32("synthesize.omnivoice.generation.position_temperature", float(gen_defaults["position_temperature"]))
    writer.add_float32("synthesize.omnivoice.generation.class_temperature", float(gen_defaults["class_temperature"]))

    package = manifest["package_contract"]
    writer.add_uint32("synthesize.capabilities.input_flags", 1 << 0)          # INPUT_TEXT_UTF8
    writer.add_uint32("synthesize.capabilities.flags", (1 << 0) | (1 << 1))   # SPEAKING_RATE | STOCHASTIC
    writer.add_uint64("synthesize.capabilities.max_input_tokens", int(package["max_input_tokens"]))
    writer.add_uint64("synthesize.capabilities.max_output_frames", int(package["max_output_frames"]))
    low, high = package["speaking_rate_range"]
    writer.add_float32("synthesize.capabilities.min_speaking_rate", float(low))
    writer.add_float32("synthesize.capabilities.max_speaking_rate", float(high))

    audio = package["native_audio"]
    writer.add_uint32("synthesize.audio.sample_rate_hz", int(audio["sample_rate_hz"]))
    writer.add_uint32("synthesize.audio.channels", int(audio["channels"]))
    writer.add_string("synthesize.audio.sample_format", audio["sample_format"])

    # Voice: no preset catalog; the package default is auto-voice.
    writer.add_string("synthesize.voice.mode", "profile-sources")
    writer.add_bool("synthesize.voice.has_package_default", True)
    writer.add_uint32("synthesize.voice.preset_count", 0)

    # Generator geometry (checked against llm_config).
    for key, value in (
        ("layer_count", llm["num_hidden_layers"]),
        ("hidden_size", llm["hidden_size"]),
        ("attention_head_count", llm["num_attention_heads"]),
        ("key_value_head_count", llm["num_key_value_heads"]),
        ("head_dim", llm["head_dim"]),
        ("intermediate_size", llm["intermediate_size"]),
        ("text_vocab_size", llm["vocab_size"]),
    ):
        writer.add_uint32(f"synthesize.omnivoice.generator.{key}", int(value))
    writer.add_float32("synthesize.omnivoice.generator.rms_norm_eps", float(llm["rms_norm_eps"]))
    writer.add_float32("synthesize.omnivoice.generator.rope_theta", float(llm["rope_parameters"]["rope_theta"]))
    if any(t != "full_attention" for t in llm["layer_types"]):
        raise ConverterError("a layer_type is not full_attention; this port builds bidirectional graphs only")
    writer.add_string("synthesize.omnivoice.generator.attention", "bidirectional")
    if not bool(llm["tie_word_embeddings"]):
        raise ConverterError("tie_word_embeddings is false; the catalog expects no separate lm_head")

    # Audio canvas contract.
    writer.add_uint32("synthesize.omnivoice.audio.num_codebooks", int(config["num_audio_codebook"]))
    writer.add_uint32("synthesize.omnivoice.audio.vocab_size", int(config["audio_vocab_size"]))
    writer.add_uint32("synthesize.omnivoice.audio.mask_id", int(config["audio_mask_id"]))

    # Codec geometry, checked rather than copied (size codebooks from tensors, not config).
    writer.add_uint32("synthesize.omnivoice.codec.sample_rate", int(codec_config["sample_rate"]))
    writer.add_uint32("synthesize.omnivoice.codec.hop_length", HOP_LENGTH)
    writer.add_float32("synthesize.omnivoice.codec.frame_rate_hz", FRAME_RATE_HZ)
    if int(codec_config["sample_rate"]) != SAMPLE_RATE:
        raise ConverterError("codec sample rate moved")
    acoustic = codec_config["acoustic_model_config"]
    ratios = [int(r) for r in acoustic["upsampling_ratios"]]
    total = 1
    for r in ratios:
        total *= r
    if total != HOP_LENGTH:
        raise ConverterError(f"upsampling ratios {ratios} multiply to {total}, not hop {HOP_LENGTH}")
    writer.add_array("synthesize.omnivoice.codec.upsampling_ratios", ratios)
    writer.add_uint32("synthesize.omnivoice.codec.decoder_hidden_size", int(acoustic["decoder_hidden_size"]))
    writer.add_uint32("synthesize.omnivoice.codec.encoder_hidden_size", int(acoustic["encoder_hidden_size"]))
    writer.add_uint32("synthesize.omnivoice.codec.hidden_size", int(acoustic["hidden_size"]))
    writer.add_uint32("synthesize.omnivoice.codec.codebook_dim", int(codec_config["codebook_dim"]))
    writer.add_uint32("synthesize.omnivoice.codec.codebook_size", int(codec_config["codebook_size"]))
    writer.add_uint32("synthesize.omnivoice.codec.semantic_sample_rate", int(codec_config["semantic_sample_rate"]))
    semantic = codec_config["semantic_model_config"]
    writer.add_uint32("synthesize.omnivoice.semantic.hidden_size", int(semantic["hidden_size"]))
    writer.add_uint32("synthesize.omnivoice.semantic.layer_count", int(semantic["num_hidden_layers"]))
    writer.add_uint32("synthesize.omnivoice.semantic.attention_head_count", int(semantic["num_attention_heads"]))
    writer.add_uint32("synthesize.omnivoice.semantic.intermediate_size", int(semantic["intermediate_size"]))
    writer.add_array("synthesize.omnivoice.semantic.conv_dim", [int(x) for x in semantic["conv_dim"]])
    writer.add_array("synthesize.omnivoice.semantic.conv_kernel", [int(x) for x in semantic["conv_kernel"]])
    writer.add_array("synthesize.omnivoice.semantic.conv_stride", [int(x) for x in semantic["conv_stride"]])
    writer.add_float32("synthesize.omnivoice.semantic.layer_norm_eps", float(semantic["layer_norm_eps"]))

    # Special token ids the prompt builder needs by value.
    specials = {t["content"]: int(t["id"]) for t in tokenizer_json["added_tokens"]}
    for name, key in (("<|denoise|>", "denoise"), ("<|lang_start|>", "lang_start"), ("<|lang_end|>", "lang_end"),
                      ("<|instruct_start|>", "instruct_start"), ("<|instruct_end|>", "instruct_end"),
                      ("<|text_start|>", "text_start"), ("<|text_end|>", "text_end")):
        if name not in specials:
            raise ConverterError(f"tokenizer carries no {name}")
        writer.add_uint32(f"synthesize.omnivoice.token.{key}", specials[name])
    writer.add_uint32("synthesize.omnivoice.token.eos", int(config["eos_token_id"]))
    writer.add_uint32("synthesize.omnivoice.token.pad", int(config["pad_token_id"]))

    # Language Capability Catalog: validated tags only; prompt text == tag.
    writer.add_array("synthesize.omnivoice.languages.tags", ["en", "zh", "ja"])

    # Voice Profile contract (consumed from Plan 3 on; declared now so one
    # package serves the whole cycle).
    writer.add_string("synthesize.profile.schema", "omnivoice-clone-prompt")
    writer.add_uint32("synthesize.profile.schema_version", 1)
    writer.add_uint32("synthesize.reference.target_sample_rate", SAMPLE_RATE)
    writer.add_uint32("synthesize.reference.target_channels", 1)
    writer.add_uint64("synthesize.reference.min_frames_per_clip", 24000)      # 1 s
    writer.add_uint64("synthesize.reference.max_frames_per_clip", 480000)     # 20 s (upstream warns beyond)
    writer.add_uint64("synthesize.reference.max_total_frames", 480000)
    writer.add_uint64("synthesize.reference.max_reference_count", 1)

    # Profile Compatibility ID: sha256 over the family compatibility manifest.
    compat = hashlib.sha256()
    for piece in ("omnivoice-clone-prompt/1", digests["generator"], digests["codec"], digests["config"]):
        compat.update(piece.encode())
    writer.add_string("synthesize.profile.compatibility_id", compat.hexdigest())

    # Text Frontend payload from tokenizer.json.
    vocab = tokenizer_json["model"]["vocab"]
    ordered = sorted(vocab, key=lambda token: vocab[token])
    merges = [" ".join(m) if isinstance(m, list) else m for m in tokenizer_json["model"]["merges"]]
    writer.add_array("synthesize.omnivoice.frontend.vocab", ordered)
    writer.add_array("synthesize.omnivoice.frontend.merges", merges)
    writer.add_bool("synthesize.frontend.present", True)
    writer.add_string("synthesize.frontend.provider", "synthesize.qwen_bpe")
    writer.add_uint32("synthesize.frontend.contract_version", 1)
```

`gen_defaults` is produced by a small helper `read_generation_defaults()` that lazily does `from omnivoice import OmniVoiceGenerationConfig; return dataclasses.asdict(OmniVoiceGenerationConfig())` and raises `ConverterError` when the import is unavailable — the doctrine is "the defect is the second copy".

- [ ] **Step 4: Run the tests, then a real conversion**

```bash
ctest --test-dir build --output-on-failure -R synthesize-vits-python-unit
uv run --project scripts/envs/omnivoice --locked python scripts/convert-omnivoice.py \
  --manifest tests/golden/omnivoice/omnivoice-0-6b.manifest.json \
  --weights-dir models/omnivoice-0-6b \
  --output models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf
```

Expected: tests PASS; conversion writes the GGUF (~3.1 GB) and the report. Sanity-read the report: emitted tensor count = 312 (generator) + codec keeps (527 − fc1(count) − decoder_semantic(14) − EMA/flags), all F32; `renamed` lists only real overflows; `skipped` reasons all present. Record the exact emitted count in the report AND in `_porting-log.md` — Task 10's `expected_tensor_count` arithmetic must reproduce it.

- [ ] **Step 5: Commit**

```bash
git add scripts/convert-omnivoice.py tests/python/test_convert_omnivoice.py reports/convert/omnivoice/
git commit -m "Add the OmniVoice converter with its silent-failure-rule tests"
```

---

### Task 8: Hoist the byte-level BPE to a shared module

**Files:**
- Create: `src/bpe-frontend.h`, `src/bpe-frontend.cpp` (content moved from `src/arch/qwen3-tts/bpe.{h,cpp}`)
- Modify: `src/arch/qwen3-tts/bpe.h`, `src/arch/qwen3-tts/bpe.cpp` (shrink to family-specific remainder)
- Modify: `src/CMakeLists.txt` (add `bpe-frontend.cpp`)
- Test: existing `tests/qwen3_tts_bpe_test.cpp` must pass **unchanged**

**Interfaces:**
- Produces: `synth::BpeFrontendConfig`, `synth::make_bpe_frontend(const BpeFrontendConfig &, std::unique_ptr<TextFrontend> &)`, `synth::qwen_pretokenize(const std::string &)` — consumed by Task 12's omnivoice load path and by qwen3-tts via aliases.

This is a behavior-preserving refactor guarded by the existing qwen3-tts BPE tests. Ground truth for what is generic vs family-specific: the only qwen3-specific pieces of `bpe.cpp` are the namespace, `qwen_assistant_turn`, and the two `kAssistant*` constants; the pre-tokenizer pattern is the Qwen2 pattern that **omnivoice's tokenizer.json also uses verbatim** (verified at intake), so it hoists as-is.

- [ ] **Step 1: Run the guard tests before touching anything**

```bash
ctest --test-dir build --output-on-failure -R synthesize-qwen3-tts-bpe-test
```
Expected: PASS (baseline).

- [ ] **Step 2: Move the generic parts**

Create `src/bpe-frontend.h` containing, in `namespace synth`: `BpeFrontendConfig` (struct body verbatim from today's `src/arch/qwen3-tts/bpe.h`, comments included), `make_bpe_frontend`, and `qwen_pretokenize` (keep the name — it is the Qwen pattern, now shared by two Qwen-family models; say so in its comment). Create `src/bpe-frontend.cpp` by moving everything from `src/arch/qwen3-tts/bpe.cpp` except `qwen_assistant_turn`, renaming the namespace to `synth`. Includes stay: `"bpe-frontend.h"`, `"unicode-ranges.h"` (already shared), std headers.

Shrink `src/arch/qwen3-tts/bpe.h` to:

```cpp
#pragma once

#include "bpe-frontend.h"

#include <cstddef>
#include <string>

namespace synth::qwen3tts {

// The shared byte-level BPE, re-exported under this family's namespace so the
// family's own files and tests keep reading naturally.
using synth::BpeFrontendConfig;
using synth::make_bpe_frontend;
using synth::qwen_pretokenize;

// The assistant turn the reference wraps every request in, which is a fixed
// string rather than a template the package carries:
//
//     <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
std::string qwen_assistant_turn(const std::string & text);

constexpr size_t kAssistantRolePrefixTokens = 3;
constexpr size_t kAssistantSuffixTokens     = 5;

}  // namespace synth::qwen3tts
```

Shrink `src/arch/qwen3-tts/bpe.cpp` to the `qwen_assistant_turn` implementation alone (moved verbatim). Add `bpe-frontend.cpp` to `add_library(synthesize …)` in `src/CMakeLists.txt`, right after `text-frontend.cpp`.

- [ ] **Step 3: Build and re-run the guards**

```bash
cmake --build build --target synthesize-check-unit
```
Expected: everything builds; `synthesize-qwen3-tts-bpe-test` and the whole unit gate PASS with zero test edits.

- [ ] **Step 4: Sanitizer gate + format + commit**

```bash
cmake --build build-sanitize --target synthesize-check-unit
scripts/ci/clang-format.sh --fix
git add src/bpe-frontend.h src/bpe-frontend.cpp src/arch/qwen3-tts/bpe.h src/arch/qwen3-tts/bpe.cpp src/CMakeLists.txt
git commit -m "Hoist the byte-level BPE frontend to a shared module"
```

---

### Task 9: Package metadata reader (`read_hparams`) + metadata unit test

**Files:**
- Create: `src/arch/omnivoice/catalog.h` (HParams structs + `read_hparams` declaration — same file split as qwen3-tts: structs in catalog.h, reader implemented in weights.cpp)
- Create: `src/arch/omnivoice/weights.cpp` (the `read_hparams` implementation; weight-struct resolution arrives in Task 10)
- Modify: `src/CMakeLists.txt`
- Test: `tests/omnivoice_metadata_test.cpp`; Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: the exact GGUF key set written by Task 7 (`synthesize.omnivoice.*` etc.) and the shared `synth::GgufMetadata` reader (`src/gguf-metadata.h`).
- Produces: `namespace synth::omnivoice`: `struct HParams` and `synth_status_t read_hparams(const gguf_context *, HParams &)` — consumed by Tasks 10 and 12.

- [ ] **Step 1: Write the HParams structs**

`src/arch/omnivoice/catalog.h`:

```cpp
#pragma once

#include "synthesize.h"

#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;

namespace synth::omnivoice {

// Plan 1 carries the source profile only; quantization profiles are a stage-6
// decision and extend this enum then.
enum class QuantizationProfile : uint32_t {
    F32,
};

// The mask-predict generator: a Qwen3 block stack run bidirectionally over the
// whole canvas. No KV cache exists because no position is ever causal.
struct GeneratorParams {
    uint32_t layer_count          = 0;
    uint32_t hidden_size          = 0;
    uint32_t attention_head_count = 0;
    uint32_t key_value_head_count = 0;
    uint32_t head_dim             = 0;
    uint32_t intermediate_size    = 0;
    uint32_t text_vocab_size      = 0;
    float    rms_norm_eps         = 0.0f;
    float    rope_theta           = 0.0f;
};

// The 8-codebook token canvas the generator predicts into.
struct AudioCanvasParams {
    uint32_t num_codebooks = 0;
    uint32_t vocab_size    = 0;  // 1025: 1024 codes plus the mask id
    uint32_t mask_id       = 0;
};

// The Higgs Audio V2 codec halves this package carries in full: DAC decoder
// (the vocoder), and the encode path for cloning.
struct CodecParams {
    uint32_t              sample_rate          = 0;
    uint32_t              hop_length           = 0;
    float                 frame_rate_hz        = 0.0f;
    uint32_t              decoder_hidden_size  = 0;
    uint32_t              encoder_hidden_size  = 0;
    uint32_t              hidden_size          = 0;  // fc2's output width
    uint32_t              codebook_dim         = 0;
    uint32_t              codebook_size        = 0;
    uint32_t              semantic_sample_rate = 0;
    std::vector<uint32_t> upsampling_ratios;
};

// The HuBERT semantic branch, used only when preparing a cloning profile.
struct SemanticParams {
    uint32_t              hidden_size          = 0;
    uint32_t              layer_count          = 0;
    uint32_t              attention_head_count = 0;
    uint32_t              intermediate_size    = 0;
    float                 layer_norm_eps       = 0.0f;
    std::vector<uint32_t> conv_dim;
    std::vector<uint32_t> conv_kernel;
    std::vector<uint32_t> conv_stride;
};

// The seven prompt markers plus eos/pad, by value. All are added tokens past
// the base vocabulary.
struct SpecialTokens {
    uint32_t denoise        = 0;
    uint32_t lang_start     = 0;
    uint32_t lang_end       = 0;
    uint32_t instruct_start = 0;
    uint32_t instruct_end   = 0;
    uint32_t text_start     = 0;
    uint32_t text_end       = 0;
    uint32_t eos            = 0;
    uint32_t pad            = 0;
};

// The checkpoint's own decoding defaults, carried in the package rather than
// restated here. position_temperature 5.0 means the public path samples;
// both temperatures zero is the deterministic mode the goldens use.
struct GenerationDefaults {
    uint32_t num_step             = 0;
    float    guidance_scale       = 0.0f;
    float    t_shift              = 0.0f;
    float    layer_penalty_factor = 0.0f;
    float    position_temperature = 0.0f;
    float    class_temperature    = 0.0f;
};

// The Reference Audio limits and Serialized Profile identity the Voice Profile
// module enforces from Plan 3 on; read and validated from Plan 1 so a package
// is whole from its first cut.
struct ProfileContract {
    std::string schema;
    uint32_t    schema_version         = 0;
    std::string compatibility_id_hex;  // 64 hex chars = 32 bytes
    uint32_t    reference_sample_rate  = 0;
    uint32_t    reference_channels     = 0;
    uint64_t    min_frames_per_clip    = 0;
    uint64_t    max_frames_per_clip    = 0;
    uint64_t    max_total_frames       = 0;
    uint64_t    max_reference_count    = 0;
};

struct HParams {
    std::string         model_variant;
    QuantizationProfile quantization_profile         = QuantizationProfile::F32;
    uint32_t            quantization_profile_version = 1;
    uint32_t            architecture_version         = 1;

    uint32_t input_flags          = 0;
    uint32_t capability_flags     = 0;
    uint32_t output_sample_rate   = 0;
    uint32_t output_channel_count = 0;
    uint64_t max_input_tokens     = 0;
    uint64_t max_output_frames    = 0;
    float    min_speaking_rate    = 0.0f;
    float    max_speaking_rate    = 0.0f;

    GeneratorParams    generator;
    AudioCanvasParams  audio;
    CodecParams        codec;
    SemanticParams     semantic;
    SpecialTokens      tokens;
    GenerationDefaults generation;
    ProfileContract    profile;

    // Validated languages as BCP-47 tags. For this family the tag is also the
    // exact text the prompt's language slot carries, so no name bridge exists.
    std::vector<std::string> language_tags;

    bool        frontend_present          = false;
    std::string frontend_provider;
    uint32_t    frontend_contract_version = 0;
};

synth_status_t read_hparams(const gguf_context * gguf, HParams & hparams);

}  // namespace synth::omnivoice
```

- [ ] **Step 2: Write the failing metadata test**

`tests/omnivoice_metadata_test.cpp`, mirroring `tests/qwen3_tts_metadata_test.cpp` exactly in mechanism (`GgufDeleter`/`GgufContext`, `set_string_array`, `valid_metadata()` builder, `expect_rejected` mutation driver, chained `int check_*()` + `SYNTH_TEST_CHECK`). The `valid_metadata()` builder sets **every key from Task 7's `add_metadata`** with the real values (28 layers, hidden 1024, 16/8 heads, head_dim 128, ffn 3072, vocab 151676, eps 1e-6, theta 1e6, attention "bidirectional"; canvas 8/1025/1024; codec 24000/960/25.0, ratios {8,5,4,2,3}, decoder 1024, encoder 64, hidden 256, codebook 64/1024, semantic rate 16000; HuBERT 768/12/12/3072, conv arrays; tokens 151669–151675, eos 151645, pad 151643; generation 32/2.0/0.1/5.0/5.0/0.0; capabilities: input_flags 1, flags 3, max tokens 2048, max frames 750, rates 0.5/2.0; audio 24000/1/f32le; voice mode "profile-sources", has_package_default true, preset_count 0; languages tags {"en","zh","ja"}; profile schema "omnivoice-clone-prompt"/1, compat id = 64 hex chars, reference 24000/1, 24000/480000/480000/1; frontend present/"synthesize.qwen_bpe"/1; profile "F32"/1).

`run_valid_package()` asserts a representative spread of the parsed fields. `run_rejections()` — one `expect_rejected` per rule; the full list to implement:

1. `general.architecture` ≠ "omnivoice"
2. unknown `synthesize.quantization.profile` (e.g. "BF16" — this family's source is F32)
3. format_version ≠ 1; architecture_version ≠ 1
4. generator: zero layer_count; heads not divisible by kv heads; zero head_dim
5. `synthesize.omnivoice.generator.attention` ≠ "bidirectional" (e.g. "causal")
6. canvas: `mask_id != vocab_size - 1`; `num_codebooks == 0`
7. codec: ratios product ≠ hop; `sample_rate / hop != frame_rate_hz` (1e-6); codec rate ≠ declared output rate; empty ratios array
8. semantic: conv_dim/conv_kernel/conv_stride length mismatch; zero hidden
9. tokens: any marker id ≥ text_vocab_size; markers not strictly increasing from denoise..text_end is NOT required (do not over-constrain) — but duplicate ids are refused
10. generation: num_step == 0; guidance_scale < 0; t_shift ≤ 0; position_temperature < 0
11. capabilities: max_input_tokens 0; max_output_frames 0; speaking-rate range with min > 1.0 or max < 1.0 or min ≤ 0; SPEAKING_RATE capability flag missing while the range is non-degenerate; STOCHASTIC flag missing (this package samples)
12. audio: channels ≠ 1; sample_format ≠ "f32le"
13. voice: mode ≠ "profile-sources"; has_package_default false (auto-voice IS the default); preset_count ≠ 0
14. languages: empty tags array; a tag that is empty
15. profile: schema empty; compatibility id not 64 hex chars; reference sample rate 0; min_frames_per_clip 0 or > max_frames_per_clip; max_reference_count 0
16. frontend: present false; provider ≠ "synthesize.qwen_bpe"; contract_version ≠ 1

Register in `tests/CMakeLists.txt` next to the qwen3-tts block:

```cmake
synth_add_unit_test(synthesize-omnivoice-metadata-test omnivoice_metadata_test.cpp)
```

- [ ] **Step 3: Verify it fails to build (no reader yet)**

```bash
cmake --build build --target synthesize-omnivoice-metadata-test
```
Expected: FAIL — `read_hparams` undefined.

- [ ] **Step 4: Implement `read_hparams`**

`src/arch/omnivoice/weights.cpp`: mirror qwen3-tts's `weights.cpp` reader structure — `GgufMetadata meta(gguf, "omnivoice")`, one `read_*` helper per HParams group (`read_identity`, `read_quantization`, `read_capabilities`, `read_generation`, `read_generator`, `read_audio_canvas`, `read_codec`, `read_semantic`, `read_tokens`, `read_languages`, `read_profile_contract`, `read_frontend`), each `std::fprintf(stderr, "omnivoice: …\n"); return false;` on a semantic failure, top level mapping to `SYNTH_ERR_GGUF`, `nullptr → SYNTH_ERR_INVALID_ARG`. Every rejection rule from Step 2's list gets its check here. Read the u32 arrays with `positive_i32_array`. Add `src/arch/omnivoice/weights.cpp` to `src/CMakeLists.txt`.

- [ ] **Step 5: Run to green, sanitize, commit**

```bash
cmake --build build --target synthesize-check-unit
ctest --test-dir build --output-on-failure -R synthesize-omnivoice-metadata-test
cmake --build build-sanitize --target synthesize-check-unit
scripts/ci/clang-format.sh --fix
git add src/arch/omnivoice/catalog.h src/arch/omnivoice/weights.cpp src/CMakeLists.txt tests/omnivoice_metadata_test.cpp tests/CMakeLists.txt
git commit -m "Read and validate omnivoice package metadata"
```

---

### Task 10: Tensor catalog + weight structs + catalog unit test

**Files:**
- Create: `src/arch/omnivoice/weights.h` (weight structs + `build_model_weights` + `expected_tensor_count`)
- Create: `src/arch/omnivoice/catalog.cpp` (Resolver + per-group resolution + sweep)
- Modify: `src/CMakeLists.txt`
- Test: `tests/omnivoice_catalog_test.cpp`; Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `HParams` (Task 9); the authoritative name/shape list `reports/porting/omnivoice/omnivoice-0-6b/tensor-inventory.json` (Task 4) with Task 7's prefixing/renames applied (generator names bare, codec names `codec.`-prefixed and NAME_SHORTENINGS-shortened) — the converter report `reports/convert/omnivoice/omnivoice-0-6b-F32.json` lists the exact emitted names; **that report is the catalog's ground truth**.
- Produces: `ModelWeights`, `build_model_weights(ggml_context *, const HParams &, ModelWeights &)` (no twin-context parameter — Plan 1 is CPU-only; the accelerator-twin seam is added in the backends stage), `expected_tensor_count(const HParams &)`.

- [ ] **Step 1: Derive the layout from the converter report**

Read `reports/convert/omnivoice/omnivoice-0-6b-F32.json` `tensors[].name` + `ggml_shape` and write down the grouped inventory (generator: `llm.embed_tokens.weight`, `llm.layers.{0..27}.…` 11 per layer, `llm.norm.weight`, `audio_embeddings.weight` [8200,1024], `audio_heads.weight` [8200,1024]; codec groups: `codec.quantizer.…`, `codec.fc.…`/`codec.fc2.…`, `codec.acoustic_decoder.…`, `codec.acoustic_encoder.…`, `codec.semantic_model.…`, `codec.encoder_semantic.…`). Transcribe the per-group field pattern (e.g. per acoustic_decoder block: which snake/conv/conv_t tensors with which shapes) into the top-of-file comment of `catalog.cpp` — the catalog is written against these names, and the count arithmetic below must equal the report's `emitted_tensor_count` exactly.

- [ ] **Step 2: Write the weight structs**

`src/arch/omnivoice/weights.h`, following the qwen3-tts style (`LinearWeights`, `Conv1dWeights`, `LayerNormWeights`; plus this family's `SnakeWeights { ggml_tensor * alpha; }` — Higgs DAC uses plain Snake, alpha only, no beta). Groups:

```cpp
struct GeneratorLayerWeights {   // llm.layers.N.*
    ggml_tensor * input_layernorm = nullptr;
    ggml_tensor * q_proj = nullptr, * k_proj = nullptr, * v_proj = nullptr, * o_proj = nullptr;
    ggml_tensor * q_norm = nullptr, * k_norm = nullptr;   // per-head, width head_dim
    ggml_tensor * post_attention_layernorm = nullptr;
    ggml_tensor * gate_proj = nullptr, * up_proj = nullptr, * down_proj = nullptr;
};

struct GeneratorWeights {
    ggml_tensor * text_embedding = nullptr;   // llm.embed_tokens.weight; tied — no lm_head exists
    std::vector<GeneratorLayerWeights> layers;
    ggml_tensor * norm             = nullptr;
    ggml_tensor * audio_embeddings = nullptr;  // [8200, 1024] stacked codebooks
    ggml_tensor * audio_heads      = nullptr;  // [8200, 1024]
};
```

plus `RvqQuantizerWeights { input_proj, output_proj, codebook }` (×8), `fc`/`fc2` linears, `AcousticDecoderWeights`/`AcousticEncoderWeights`/`SemanticModelWeights`/`SemanticEncoderWeights` structured per Step 1's derived pattern (vectors sized from HParams: 28 generator layers, 8 quantizers, `upsampling_ratios.size()` decoder blocks, `semantic.layer_count` HuBERT layers, `conv_dim.size()` feature-extractor convs), and:

```cpp
struct ModelWeights {
    GeneratorWeights            generator;
    std::vector<RvqQuantizerWeights> quantizers;   // 8, order = codebook index
    LinearWeights               fc;    // encode-path concat projection
    LinearWeights               fc2;   // decode-path 1024 -> hidden_size
    AcousticDecoderWeights      acoustic_decoder;
    AcousticEncoderWeights      acoustic_encoder;
    SemanticModelWeights        semantic_model;
    SemanticEncoderWeights      encoder_semantic;
};

synth_status_t build_model_weights(ggml_context * context, const HParams & hparams, ModelWeights & weights);
uint64_t       expected_tensor_count(const HParams & hparams);
```

- [ ] **Step 3: Write the failing catalog test**

`tests/omnivoice_catalog_test.cpp`, the qwen3-tts catalog-test technique verbatim: an independent hand-written `std::vector<Entry>` layout, `make_context()` (no-alloc ggml context, `ggml_tensor_overhead() * 4096`), `populate()` with a `Mutation` hook, `small_hparams()` structurally faithful at reduced widths (2 generator layers, 2 codebooks, 2 decoder blocks with ratios {2,3} and hop 6 — keep every cross-check satisfiable: sample_rate 150, frame 25), types all F32. Tests:

- `check_resolution` — build succeeds; vector sizes match hparams; spot-check tensors non-null (q_norm, a codebook, a snake alpha, fc2.weight); `expected_tensor_count(small) == entries.size()`.
- `check_rejections` — drop one tensor; reshape one; retype one to F16; add one extra unresolved tensor (the sweep must refuse it).
- `check_real_package_count` — hparams with the real dimensions; `SYNTH_TEST_CHECK(expected_tensor_count(real) == <emitted_tensor_count from the converter report>);` (transcribe the literal number and cite the report path in a comment).

Register: `synth_add_unit_test(synthesize-omnivoice-catalog-test omnivoice_catalog_test.cpp)`. Build → expected FAIL (no catalog.cpp).

- [ ] **Step 4: Implement the catalog**

`src/arch/omnivoice/catalog.cpp`: copy the qwen3-tts `Resolver` mechanism (find with expected-shape/axis sweep, `fail()` printing once with `"omnivoice: "` prefix, `resolved_` set, `linear`/`conv`/`transpose_conv`/`snake` helpers). `expected_type()` is trivially F32 for every role in this profile. Resolution groups mirror Step 2's structs; every shape derives from HParams (e.g. q_proj `{hidden, head_count*head_dim}`, audio tables `{hidden, num_codebooks*vocab_size}`, codebook `{codebook_dim, codebook_size}`, HuBERT feature conv i `{conv_kernel[i], conv_dim[i-1] or 1, conv_dim[i]}`…). End with the post-resolution sweep ("a tensor the catalog never asked for is an error") and implement `expected_tensor_count` as independent arithmetic (constant per-group counts × HParams sizes — derive each constant from Step 1's pattern, comment the breakdown like qwen3's).

- [ ] **Step 5: Green, sanitize, commit**

```bash
cmake --build build --target synthesize-check-unit
ctest --test-dir build --output-on-failure -R synthesize-omnivoice-catalog-test
cmake --build build-sanitize --target synthesize-check-unit
scripts/ci/clang-format.sh --fix
git add src/arch/omnivoice/weights.h src/arch/omnivoice/catalog.cpp src/CMakeLists.txt tests/omnivoice_catalog_test.cpp tests/CMakeLists.txt
git commit -m "Resolve the omnivoice tensor catalog with shape and sweep checks"
```

---

### Task 11: Frontend host — prompt text construction + duration estimator

**Files:**
- Create: `src/arch/omnivoice/frontend-host.h`, `src/arch/omnivoice/frontend-host.cpp`
- Modify: `scripts/generate-unicode-ranges.py` + regenerate `src/unicode-ranges.h` (add mark and punctuation/symbol range tables)
- Modify: `src/CMakeLists.txt`
- Test: `tests/omnivoice_frontend_test.cpp`; Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `synth::TextFrontend` (shared), oracle values from `build/goldens/omnivoice/tokenizer/cases.json` (Task 6) transcribed as compile-time tables.
- Produces (namespace `synth::omnivoice`):
  - `std::string combine_text(const std::string & ref_text, const std::string & text);`
  - `std::string style_text(bool denoise, const std::string & language_tag, const std::string & instruct);`
  - `synth_status_t tokenize_wrapped_text(const TextFrontend & frontend, const std::string & wrapped, std::vector<int32_t> & ids);` (nonverbal-tag-aware)
  - `class DurationEstimator { double total_weight(const std::string & utf8) const; double estimate_duration(const std::string & target, const std::string & ref, double ref_duration) const; uint64_t estimate_target_frames(const std::string & text, const std::string & ref_text, uint64_t ref_frames, float speaking_rate) const; };`

- [ ] **Step 1: Extend the unicode range generator**

`scripts/generate-unicode-ranges.py` currently emits letter/number/whitespace ranges. Add two more tables to its output, generated the same way from `unicodedata.category`: `kMarkRanges` (categories `Mn, Mc, Me`) and `kPunctSymbolRanges` (categories starting `P` or `S`). Regenerate:

```bash
python3 scripts/generate-unicode-ranges.py > src/unicode-ranges.h   # use the script's own documented invocation if it differs
cmake --build build --target synthesize-check-unit                  # nothing may regress
```

- [ ] **Step 2: Write the failing tests**

`tests/omnivoice_frontend_test.cpp` (standalone main + `SYNTH_TEST_CHECK`):

- `check_combine_text`: table of `(ref, text) → expected` cases transcribed from the oracle's `_combine_text` behavior via `cases.json` — cover: plain join with one space; `\r\n` stripping; `（）→()`; tab/space collapsing; space adjacent to CJK removed ("你好 世界" → "你好世界"); empty ref.
- `check_style_text`: `style_text(false, "en", "") == "<|lang_start|>en<|lang_end|><|instruct_start|>None<|instruct_end|>"`; `style_text(true, "", "female, young") == "<|denoise|><|lang_start|>None<|lang_end|><|instruct_start|>female, young<|instruct_end|>"`.
- `check_nonverbal_split`: with a tiny fixture vocab (the qwen3 BPE-test technique: hand vocab containing the pieces plus `[laughter]` NOT as one token), assert `tokenize_wrapped_text` produces the tag's standalone tokenization between the neighbours' tokenizations, and that a text without tags equals plain `frontend.prepare`.
- `check_char_weights`: table asserting `_get_char_weight` ports: 'a'→1.0, ' '→0.2, '7'→3.5, '。'→0.5, '你'→3.0, 'あ'→2.2, '한'→2.5, combining acute (U+0301)→0.0, arabic tatweel (U+0640)→0.0, 'я'→1.0, 'ई'→1.8.
- `check_estimates_match_oracle`: transcribe from `cases.json` (comment: "values from scripts/dump_reference_omnivoice_tokenizer.py at <WEIGHTS_REV>") the `(text, ref_text, ref_frames, speed) → expected_frames` rows for every manifest case incl. the anchor fallback and the 0.5/2.0 speed rows; assert `estimate_target_frames` equality. Include one `< low_threshold` case asserting the boost curve `50 * (est/50)^(1/3)` to full double precision.

Register `synth_add_unit_test(synthesize-omnivoice-frontend-test omnivoice_frontend_test.cpp)`. Build → FAIL (no implementation).

- [ ] **Step 3: Implement**

`frontend-host.cpp` ports, faithfully and with the upstream formulas in comments:

- `combine_text`: `ref.strip + " " + text.strip` then the four cleanup rules in upstream order (strip `[\r\n]+`, `（）`→`()`, collapse `[ \t]+`, remove spaces adjacent to CJK codepoints — CJK test via the existing letter ranges' CJK blocks; iterate UTF-8 codepoint-wise).
- `style_text`: the two f-strings with `"None"` fallbacks; `<|denoise|>` only when `denoise`.
- `tokenize_wrapped_text`: split on the 13-tag table `{"[laughter]","[sigh]","[confirmation-en]","[question-en]","[question-ah]","[question-oh]","[question-ei]","[question-yi]","[surprise-ah]","[surprise-oh]","[surprise-wa]","[surprise-yo]","[dissatisfaction-hnn]"}` (longest-match scan), tokenize segments and tags independently, concatenate.
- `DurationEstimator`: the weight map (constants block, values verbatim from the family doc's table), the 87-entry `(end_codepoint, script)` range table transcribed from `omnivoice/utils/duration.py:73-162` (transcribe at implementation time from the pinned source: `uv run --project scripts/envs/omnivoice --locked python -c "import inspect, omnivoice.utils.duration as d; print(inspect.getsource(d))"`), classification order exactly upstream's (ASCII letter → latin; 0x20 → space; 0x0640 → mark; category M*→mark, P*/S*→punct, Z*→space, N*→digit — via the new shared range tables; then the script bisect; `>0x20000`→cjk; default). `estimate_duration` with `low_threshold=50, boost_strength=3`; `estimate_target_frames` applying the `("Nice to meet you.", 25)` anchor when refless and dividing by `speaking_rate` (`est/speed`, `max(1, int(...))` — match upstream's truncation exactly).

- [ ] **Step 4: Green, sanitize, commit**

```bash
cmake --build build --target synthesize-check-unit
ctest --test-dir build --output-on-failure -R synthesize-omnivoice-frontend-test
cmake --build build-sanitize --target synthesize-check-unit
scripts/ci/clang-format.sh --fix
git add src/arch/omnivoice/frontend-host.h src/arch/omnivoice/frontend-host.cpp \
        scripts/generate-unicode-ranges.py src/unicode-ranges.h src/CMakeLists.txt \
        tests/omnivoice_frontend_test.cpp tests/CMakeLists.txt
git commit -m "Port the omnivoice prompt text rules and duration estimator"
```

---

### Task 12: Model load path + core-runtime seams + real-package load smoke

**Files:**
- Create: `src/arch/omnivoice/omnivoice.h`, `src/arch/omnivoice/model.cpp`
- Modify: `src/model-info.h` (enum), `src/synthesize.cpp` (four seams + handle member), `src/CMakeLists.txt`
- Modify: root `CMakeLists.txt` (add `SYNTH_OMNIVOICE_TEST_MODEL`, default `models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf`)
- Test: `tests/omnivoice_model_errors_test.cpp` (unit), `tests/omnivoice_load_real.cpp` (integration, model-gated); Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: everything from Tasks 8–11.
- Produces: `synth::omnivoice::Model` with `load_cpu / load / get_info / text_frontend() / samples_per_frame() / text_vocab_size()`; `synth::ModelFamily::Omnivoice`; a public `synth_model_load` that loads the real GGUF end-to-end. Synthesis itself is Plan 2: the synthesis branch added here returns `SYNTH_ERR_INTERNAL` with diagnostic id `synthesis.not_implemented` and is replaced by Plan 2's first task.

- [ ] **Step 1: Write the family header**

`src/arch/omnivoice/omnivoice.h`:

```cpp
#pragma once

#include "synthesize.h"
#include "text-frontend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_backend_device;

namespace synth::omnivoice {

struct ModelInfo {
    std::string family = "omnivoice";
    std::string variant;
    std::string quantization_profile;
    uint32_t    architecture_version = 0;

    synth_input_flags_t input_flags          = 0;
    uint32_t            capability_flags     = 0;
    uint32_t            output_sample_rate   = 0;
    uint32_t            output_channel_count = 0;
    uint64_t            max_input_tokens     = 0;
    uint64_t            max_output_frames    = 0;
    float               min_speaking_rate    = 0.0f;
    float               max_speaking_rate    = 0.0f;

    // Auto-voice is the unnamed package default; the Preset Voice Catalog is
    // deliberately empty. Voice identity arrives through profiles.
    bool                     has_package_default = true;
    std::vector<std::string> language_tags;  // already BCP-47; no name bridge

    bool        frontend_present = false;
    std::string frontend_provider;
};

class Model {
  public:
    static synth_status_t load_cpu(const std::string & path, std::unique_ptr<Model> & output);
    static synth_status_t load(const std::string &      path,
                               ggml_backend_device *    primary_device,
                               bool                     include_accelerators,
                               std::unique_ptr<Model> & output);

    ~Model();
    Model(const Model &)             = delete;
    Model & operator=(const Model &) = delete;

    synth_status_t                      get_info(ModelInfo & output) const;
    std::shared_ptr<const TextFrontend> text_frontend() const;

    // The frame geometry the core reports, and the vocabulary the frontend's
    // ids index — the text tower's, which the core range-checks against.
    uint32_t samples_per_frame() const;  // the codec hop: 960
    uint32_t text_vocab_size() const;

  private:
    struct Impl;
    explicit Model(std::unique_ptr<Impl> implementation);
    std::unique_ptr<Impl> implementation_;
};

}  // namespace synth::omnivoice
```

- [ ] **Step 2: Write the failing tests**

(a) `tests/omnivoice_model_errors_test.cpp` — the vits/kokoro model-errors pattern: `argv[1]` fixture dir; empty path → `SYNTH_ERR_INVALID_ARG`; missing file → `SYNTH_ERR_FILE_NOT_FOUND`; a "not a GGUF" text file → `SYNTH_ERR_GGUF`; `gguf_init_empty()+gguf_write_to_file` empty GGUF → `SYNTH_ERR_GGUF`; each asserting `model == nullptr` after. Register with `synth_add_unit_test` + the fixture-dir argument convention used by `vits_model_errors_test` in `tests/CMakeLists.txt` (copy its `add_test` form).

(b) `tests/omnivoice_load_real.cpp` — integration, gated: takes the model path as `argv[1]`, `synth::omnivoice::Model::load_cpu`, asserts OK, prints `ModelInfo` fields as one JSON line, asserts `samples_per_frame()==960`, `text_vocab_size()==151676`, `language_tags == {en,zh,ja}`, frontend present, then loads through the PUBLIC seam too (`synth_model_load` + `synth_model_get_info` from `include/synthesize.h`) and asserts the shared info (sample rate 24000, languages count 3, empty preset voice list, speaking-rate range [0.5,2.0]). Register mirroring the qwen3-tts integration-target pattern, gated on `EXISTS ${SYNTH_OMNIVOICE_TEST_MODEL}`, labels `"integration;omnivoice;abi"`.

Build → both FAIL (no model.cpp, no enum).

- [ ] **Step 3: Implement the load path**

`src/arch/omnivoice/model.cpp`: mirror the qwen3-tts load flow with the Plan-1 simplifications (no codec twins, CPU buffer only): probe file → `BackendPlan::create` → `gguf_init_from_file(no_alloc, ctx)` → `read_hparams` → `build_model_weights(weights_context, hparams, weights)` → frontend construction:

```cpp
        if (implementation->hparams.frontend_present) {
            GgufMetadata      meta(implementation->gguf, "omnivoice");
            BpeFrontendConfig config;
            config.provider_id      = implementation->hparams.frontend_provider;
            config.contract_version = implementation->hparams.frontend_contract_version;
            if (!meta.string_array("synthesize.omnivoice.frontend.vocab", config.vocab) ||
                !meta.string_array("synthesize.omnivoice.frontend.merges", config.merges)) {
                return SYNTH_ERR_GGUF;
            }
            // The prompt markers are added tokens past the vocabulary; they
            // carry their ids. No prefix/suffix: this family's prompt is
            // assembled by the synthesis path, not wrapped at tokenize time.
            const SpecialTokens & t = implementation->hparams.tokens;
            config.special_tokens   = {
                { "<|denoise|>",        int32_t(t.denoise)        },
                { "<|lang_start|>",     int32_t(t.lang_start)     },
                { "<|lang_end|>",       int32_t(t.lang_end)       },
                { "<|instruct_start|>", int32_t(t.instruct_start) },
                { "<|instruct_end|>",   int32_t(t.instruct_end)   },
                { "<|text_start|>",     int32_t(t.text_start)     },
                { "<|text_end|>",       int32_t(t.text_end)       },
            };
            std::unique_ptr<TextFrontend> frontend;
            status = make_bpe_frontend(config, frontend);
            if (status != SYNTH_OK) {
                return status;
            }
            implementation->frontend = std::shared_ptr<const TextFrontend>(std::move(frontend));
        }
```

then CPU buffer alloc + `stream_tensor_data(path, gguf, weights_context, "omnivoice")`, the same exception envelope, and `get_info` populated from HParams (tags copied directly). `samples_per_frame()` returns `hparams.codec.hop_length`; `text_vocab_size()` returns `hparams.generator.text_vocab_size`.

- [ ] **Step 4: Wire the core seams**

All four seams, mirroring the precedent verbatim:
1. `src/model-info.h`: add `Omnivoice` to `enum class ModelFamily`.
2. `src/synthesize.cpp` `struct synth_model`: add `std::unique_ptr<synth::omnivoice::Model> omnivoice;`.
3. `read_model_family`: add `if (architecture == "omnivoice") { family = synth::ModelFamily::Omnivoice; return SYNTH_OK; }`.
4. New `shared_info` overload:

```cpp
synth::ModelInfo shared_info(const synth::omnivoice::ModelInfo &        info,
                             std::shared_ptr<const synth::TextFrontend> frontend,
                             uint32_t                                   samples_per_frame,
                             uint32_t                                   text_vocab_size) {
    synth::ModelInfo shared;
    shared.family              = synth::ModelFamily::Omnivoice;
    shared.has_package_default = info.has_package_default;
    // The Preset Voice Catalog is empty by design: identity arrives through
    // Voice Profiles, and the unnamed package default is auto-voice.
    for (const std::string & tag : info.language_tags) {
        shared.languages.push_back({ tag, SYNTH_LANGUAGE_REGIONAL_FALLBACK });
    }
    shared.text_frontend        = std::move(frontend);
    shared.input_flags          = info.input_flags;
    shared.capability_flags     = info.capability_flags;
    shared.output_sample_rate   = info.output_sample_rate;
    shared.output_channel_count = info.output_channel_count;
    shared.vocab_size           = text_vocab_size;
    shared.samples_per_frame    = samples_per_frame;
    shared.max_input_tokens     = info.max_input_tokens;
    shared.max_output_frames    = info.max_output_frames;
    shared.min_speaking_rate    = info.min_speaking_rate;
    shared.max_speaking_rate    = info.max_speaking_rate;
    return shared;
}
```

(No language is marked `SYNTH_LANGUAGE_DEFAULT`: a request naming no language gets the language-agnostic prompt — the literal `None` slot — same doctrine as qwen3-tts's no-think prompt.)
5. Load branch in `synth_model_load` (before the kokoro branch, same shape as the qwen3-tts branch, calling the four-argument `shared_info`).
6. Synthesis branch in `synth_synthesize`:

```cpp
    if (context->model->info.family == synth::ModelFamily::Omnivoice) {
        // Plan 2 lands the diffusion loop; a loadable-but-unsynthesizable
        // family must fail loudly rather than fall through to another
        // family's branch.
        emit_diagnostic(prepared.diagnostics, SYNTH_ERR_INTERNAL, "synthesis.not_implemented",
                        "omnivoice synthesis is not implemented yet");
        return SYNTH_ERR_INTERNAL;
    }
```

7. `src/CMakeLists.txt`: add `arch/omnivoice/model.cpp` (and the files from Tasks 9–11 if any registration was missed). Root `CMakeLists.txt`: add the `SYNTH_OMNIVOICE_TEST_MODEL` cache variable next to `SYNTH_QWEN3_TTS_TEST_MODEL`, defaulting to `${CMAKE_SOURCE_DIR}/models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf`.

- [ ] **Step 5: Green everything, then the real-package smoke**

```bash
cmake --build build --target synthesize-check-unit
ctest --test-dir build --output-on-failure -L unit
ctest --test-dir build --output-on-failure -R synthesize-omnivoice-load-real   # model-gated; must run on this machine since Task 7 produced the GGUF
cmake --build build-sanitize --target synthesize-check-unit
```

Expected: unit gate PASS; the load-real test PASSES against the converted GGUF — this closes the loop converter ↔ metadata reader ↔ catalog ↔ frontend on the real package. If the catalog sweep rejects a real tensor here, the converter report and Task 10's Step 1 derivation disagree — fix the catalog (or a wrong converter skip rule), never by loosening the sweep.

- [ ] **Step 6: Update the porting log + format + commit**

Append a dated section to `_porting-log.md` (stage-4 slice 1 evidence: unit tests registered, sanitizer clean, real-package load smoke result, emitted-vs-expected tensor count agreement). Then:

```bash
scripts/ci/clang-format.sh --fix
git add src/arch/omnivoice src/model-info.h src/synthesize.cpp src/CMakeLists.txt CMakeLists.txt \
        tests/omnivoice_model_errors_test.cpp tests/omnivoice_load_real.cpp tests/CMakeLists.txt \
        reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md
git commit -m "Load omnivoice packages end-to-end: family seams, load path, real-package smoke"
```

---

## Plan-Level Completion Checklist

- [ ] All 12 tasks committed on `worktree-omnivoice`; `git log --oneline` reads as the 12 task commits.
- [ ] `cmake --build build --target synthesize-check-unit` and the sanitizer build both green.
- [ ] `ctest -R synthesize-golden-manifest-contract`, `-R synthesize-vits-python-unit`, `-R synthesize-omnivoice-load-real` all green.
- [ ] `build/goldens/omnivoice/` holds all 20 dumped cases; nothing under `models/` or `build/` is tracked by git.
- [ ] `_porting-log.md` narrates intake + slice 1 with dates and evidence.
- [ ] Report to jiangzhuo with the evidence above; pushing the branch or opening any PR remains jiangzhuo's call.

