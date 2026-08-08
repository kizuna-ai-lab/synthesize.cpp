# OmniVoice Quantization: The Generator, Q4, and the Codec Policy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (this family's standing choice) to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quantize the part of OmniVoice that is actually large. The generator
is **2,450,309,120 of 3,184,565,636 tensor bytes — 76.94%** (verified below
from the convert report), and today every byte of it is pinned at F32 by a
ruling written when this family's headline claim was exact tokens. That claim
no longer governs the primary path. jiangzhuo has decided three things on
2026-08-09: ship a Q8_0-generator / F32-codec profile first; add a Q4 profile
and at least measure it for speed and quality; and adopt the reference port's
conv-exempt codec policy. This plan lands all three, in that order, with the
four-tier evidence shape each one actually needs — because the exact-token
gate that judged every previous OmniVoice profile **cannot judge these**.

**Architecture:** Five slices. **A (Tasks 1–2)** builds the two instruments
everything downstream needs: the generator's tensor→role classifier (without
which the quantizer refuses the package outright) and the speaker-identity F0
proxy, which is currently ad-hoc code that exists only as numbers in a log.
**B (Tasks 3–8)** is the first ship candidate — `Q8_GEN`, Q8_0 generator over
an untouched F32 codec — carried through all four evidence tiers, RTF, and a
ship decision. **C (Tasks 9–10)** is Q4, whose only real engineering problem
is that a k-quant embedding table is unrunnable on CUDA and the placement gate
will say so by name. **D (Tasks 11–12)** is the codec policy: the replayed-grid
free-run channel that is its hard prerequisite, then the conv-exempt change
itself — which this plan's own arithmetic says will make packages *bigger* and
still not pass the clone gate, a finding Task 12 must put in front of jiangzhuo
rather than discover halfway through. **E (Tasks 13–14)** settles
`docs/quantization.md`'s line 103 and closes out.

**Tech Stack:** C++17, GGML (pinned submodule), CMake ≥3.24, CUDA 13.3 exactly
(DGX Spark, `sm_121a`), uv-locked oracle env.

## Global Constraints

- Commit message footer, every commit:

  ```text
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01NFjueGrsKXs44rfZ7DunQK
  ```

- clang-format: `git add` (new files too) → `scripts/ci/clang-format.sh --fix`
  → `git add` → commit. FINAL command of every task:
  `scripts/ci/clang-format.sh --check-diff` with its literal exit code
  reported. Never format `ggml/` or `third_party/`.
- Every C++ task ends with BOTH unit gates green:
  `cmake --build build --target synthesize-check-unit` and the same on
  `build-sanitize`.
- Any task touching a validator, runner, tolerance file, or package artifact
  ends with the full omnivoice integration set on the CPU tree:
  `ctest -L integration -R omnivoice` on a tree configured with
  `SYNTH_BUILD_INTEGRATION_TESTS=ON`.
- **Timing is measured ONLY on `build/rel-dgx-spark`** (Release + CUDA +
  integration). Never `build/dev-dgx-spark`: it compiles ggml-cpu at `-O2 -g`,
  is 2.19× pessimistic, and already caused this project to publish a wrong
  speedup once (Plan 5 Task 2's retracted 41.8×). Check `/proc/loadavg` before
  every timed run and repeat anything taken above ~1.5 other-cores.
- **No publication.** `hf repos create` / `hf upload` are outside this plan
  entirely and require jiangzhuo's separate, per-act, named-target
  confirmation. Nothing in Slices B–D authorizes touching Hugging Face.
- **Which contracts are exact, and which are not.** State of play at plan time,
  verified in `scripts/validate-omnivoice-replay.py`:
  - The **F32/CPU reference profile keeps the exact-token contract
    unconditionally.** It is how this port's correctness is known at all.
    No task below touches it.
  - Greedy token CONTENT is already informational on the CUDA path
    (`--tokens-informational`, `tests/tolerances/omnivoice.json`'s
    `profiles.F32.backends.CUDA` cell). **Every profile in this plan that
    quantizes the generator inherits that shape**, on CPU as well as CUDA:
    a Q8_0 generator produces a different valid realization, not a wrong one.
  - Grid **SIZE** stays a hard, unconditional gate on every case
    (`validate-omnivoice-replay.py:554-563`) — it is the precondition
    `docs/backends.md`'s discrete-outputs exception is actually conditioned
    on, and a violation is real signal. It held 17/17 under the far coarser
    16-step perturbation.
  - **RVQ encode exactness (`ref.tokens`) stays an EXACT gate.** No tolerance,
    ever. It is 0 by construction whenever the codec is F32, which is the
    entire argument for shipping `Q8_GEN` first.
- **The four evidence tiers.** Every profile below is judged by these and only
  these. Do not substitute a token-flip percentage for any of them.
  - **Tier 1 — hard, dtype-independent, pass/fail.** Byte size, SHA-256,
    catalog acceptance, load, profile-string acceptance; **placement proof**
    (`validate-omnivoice-replay.py:624-633`); **structural invariance**
    (identical sample count vs the F32/CPU render, every case); **public seed
    contract** (same seed → byte-identical, `validate-omnivoice-public.py`).
  - **Tier 2 — codec half.** The `audio.pcm` probe: the oracle's committed
    grid replayed through the candidate's codec, gated on cosine AND max_abs
    at 5× measured. Trivially green — and required to reproduce the F32 cell
    *to the bit* — whenever the codec is F32.
  - **Tier 3 — generator half.** Token agreement is RECORDED AS DATA, NEVER AS
    A GATE. The instrument is the **speaker-identity F0 proxy** (Task 2):
    median F0 over voiced frames from YIN and an NCC tracker, plus the
    fraction of voiced frames below 165 Hz. "Changed" means complete register
    separation (below-165 fraction 1.00→0.00 on both) plus a >40% median shift
    on both. Report **N of 17 changed**; the accepted baseline is CPU→CUDA
    generator placement at **1 of 17**. Plus the degeneracy screen: DC offset,
    zero-crossing rate, 20 ms-frame envelope ratio, voiced-frame fraction —
    speech-shaped output sits at ZCR 3.4–4.2 kHz with envelope ratio 218–1988;
    the known failure mode is sub-50 Hz with DC≈0 and no envelope.
    **This proxy was confirmed against a human ear on 2026-08-08 in BOTH
    directions** (a positive case heard as two different speakers, a borderline
    negative heard as the same person; commit `4c215e5`), which is why it is
    citable rather than re-arguable.
  - **Tier 4 — cloning.** `ref.tokens` exact (above). Separately, clone
    fidelity is **target-relative**: compare the candidate's median F0 and
    register on `omni-clone-en` / `omni-clone-zh` against **the reference
    clip's own** F0, not against the F32 render. A profile may drift from F32
    and still be a faithful clone.
  - **Token drift does not predict speaker change.** `omni-digits` flips 98.3%
    and keeps its speaker; `omni-short-en` flips 94.0% and changes. Never
    reason from the flip percentage.
- **A permanently-red registered CTest target is not an acceptable end state.**
  Two consequences, both binding: a gate that needs a quantized package
  registers only when that package exists (the `SYNTH_*_TEST_MODEL` /
  sentinel-file pattern `tests/CMakeLists.txt` already uses); and a profile
  that fails an exact gate is recorded as BLOCKED in the family doc with its
  measurements, with **no** tolerance cell and **no** registered target — the
  precedent `Q8_MIXED` already set (`docs/porting/families/omnivoice.md`,
  "Status: BLOCKED, not shipped").
- Terms (CONTEXT.md): "Quantization Profile", "Execution Backend", "Voice
  Profile", "Validation Level", "Listening Audit" (never "MOS study" /
  "listening panel"), "Synthesis", "Model Package". This plan additionally
  uses "speaker-identity F0 proxy" for Tier 3's instrument; Task 2 adds it to
  CONTEXT.md rather than leaving it as ad-hoc prose.

## Verified Facts (re-derived for this plan — do not re-derive again, do cite)

Computed from `reports/convert/omnivoice/omnivoice-0-6b-F32.json` (798
tensors, all F32) and read out of the sources named. The arithmetic model is
validated by reproducing the committed `Q8_MIXED` package to within 1,248
bytes and both non-packing profiles exactly.

| Fact | Value | Source |
| --- | --- | --- |
| Total tensor bytes | 3,184,565,636 | convert report |
| Generator (`llm.*` + 2 audio tables) | **312 tensors, 2,450,309,120 B (76.94%)** | convert report |
| Codec | 486 tensors, 734,256,516 B | convert report |
| Generator 2-D weights | **199**, rows ∈ {1024, 2048, 3072}, all divisible by 32 **and** 256 | convert report |
| Generator 1-D tensors (norms) | 113, 262,144 B total | convert report |
| `llm.embed_tokens.weight` | [1024, 151676], 155,316,224 elements | convert report |
| `audio_embeddings.weight` | [1024, 8200], 8,396,800 elements | convert report |
| CUDA `GET_ROWS` supported types | F16, F32, BF16, I32, Q1_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 — **no k-quant** | `ggml/src/ggml-cuda/ggml-cuda.cu:5190-5207` |
| CPU `GET_ROWS` | has a quantized path (`ggml_compute_forward_get_rows_q`) | `ggml/src/ggml-cpu/ops.cpp:4743` |
| Both embedding tables are read by `ggml_get_rows` | `generator.cpp:123` (text), `:136` (audio) | source |
| `audio_heads.weight` is a plain `mul_mat` | `generator.cpp:193` — unconstrained | source |
| Generator emits no convolution; RoPE takes no weight operand | `ggml_rope_ext(… q/k, position_ids …)` at `generator.cpp:74-77` | source |
| Accelerator twin is dtype-agnostic | `ggml_new_tensor(ctx, tensor->type, …)` at `model.cpp:1424` | source |
| `matrix_family` already contains `"omnivoice"` | `quantize.cpp` | source |
| Deleting the generator pin ⇒ `Unknown` ⇒ hard fail | `quantization.cpp`'s `classify_tensor` falls through to `tokens[0] != "codec"`; `quantize.cpp` → `fail("unknown omnivoice tensor: …")` | source |

Predicted package sizes (32-byte tensor alignment + this package's 5,387,840 B
of non-data overhead):

| Profile | Predicted bytes | Q8_0 tensors | Confidence |
| --- | ---: | ---: | --- |
| F32 (shipped) | 3,189,953,504 | 0 | exact, measured |
| **Q8_0 generator / F32 codec (`Q8_GEN`)** | **1,390,700,256** | 199 | exact — no packing, model reproduces F32 and F16 to the byte |
| F16 generator / F32 codec | 1,964,930,016 | — | exact |
| Q4_K generator (plain) / F32 codec | 1,084,444,384 | 0 | exact |
| Q4_K generator, 2 tables pinned Q8_0 / F32 codec | 1,166,300,896 | 2 | exact |
| Q8_MIXED (today, codec-only) | 2,703,016,576 measured | 158 | measured |
| **F32 generator / conv-exempt codec** | **2,939,302,880** | 73 | exact — *larger than Q8_MIXED by 236 MB* |
| Q8_0 generator / conv-exempt codec | 1,140,049,632 | 272 | exact |

**Two corrections to the numbers this plan was briefed with, both material.**

1. The brief's Q4 figures (1,141,790,464 and 1,183,604,512, "Q4_K_M") do **not**
   reproduce from the convert report under any Q4_K/Q6_K mixture tried here.
   `Q4_K_M` is llama.cpp's mixture name; **this project has no `Q4_K_M`
   profile** — `policy.cpp`'s table holds exactly `F16`, `Q8_MIXED`,
   `Q5_K_MIXED`. Task 9 must therefore *define* the Q4 profile's composition
   before any size is claimed, and must not cite the brief's Q4 numbers.
2. The brief's "~41.8 MB" cost for pinning the two lookup tables to Q8_0
   assumes they were already at Q6_K. Against a **plain Q4_K** baseline the
   pin costs **81,856,512 B (78.1 MiB)** — 0.5 B/element over 163,713,024
   elements. Nearly double. Task 9 states the composition and the pin cost
   together, or neither.

**The finding Task 12 exists to surface, established here so it is not
discovered mid-flight.** All 158 codec `MatrixWeight` tensors split cleanly:
**85 are convolution kernels** (36 `acoustic_encoder`, 32 `acoustic_decoder`,
11 `encoder_semantic`, 6 `semantic_model.feat_conv`, plus the collapsed
`codec.acoustic_decoder.conv2.weight` at `[7,32,1]`), and **all 73 two-dimensional
ones live in `codec.semantic_model`** — the HuBERT stack. Therefore a
conv-exempt policy for *this* family:

- leaves the **entire acoustic decoder at F32** — the codec half every
  synthesis uses, cloning or not, gains nothing at all;
- quantizes **only** tensors on the clone-encode path — precisely the path
  whose `ref.tokens` gate is exact;
- produces a package **236 MB larger than today's `Q8_MIXED`**.

The reference port measures 0.7% RVQ-code drift under exactly this policy, so
the expected outcome is a profile that is bigger, still fails the exact clone
gate, and speeds nothing up. That does not make jiangzhuo's decision wrong —
the policy is right *as policy*, and 0.7% versus our 36.4% is the honest
comparison — but Task 12 must put this in front of him before cutting, not
after.

## File Structure

New:
- `scripts/omnivoice-speaker-proxy.py` (Task 2) — the Tier 3 instrument,
  currently ad-hoc code that exists only as numbers in the porting log.
- `tests/python/test_omnivoice_speaker_proxy.py` (Task 2).
- `tests/omnivoice_quantization_test.cpp` gains generator-role cases (Task 1);
  the file already exists.

Modified (principal, by task):
- Task 1: `src/arch/omnivoice/quantization.{h,cpp}`,
  `tools/synthesize-quantize/{policy.h,policy.cpp}`,
  `tests/omnivoice_quantization_test.cpp`.
- Task 2: `scripts/omnivoice-speaker-proxy.py`, `tests/CMakeLists.txt`,
  `CONTEXT.md`.
- Task 3: `tools/synthesize-quantize/policy.cpp`,
  `src/arch/omnivoice/{omnivoice.h,weights.cpp,catalog.cpp}`,
  `tests/omnivoice_{quantization,catalog,metadata}_test.cpp`.
- Tasks 4–7: no source changes expected; the porting log
  (`reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`) and
  `tests/tolerances/omnivoice.json`.
- Task 8: `tests/CMakeLists.txt`, `docs/porting/families/omnivoice.md`,
  `docs/testing.md`, `docs/models/omnivoice-0-6b.md`,
  `scripts/hf_cards/omnivoice-0-6b.yaml`.
- Task 9: `tools/synthesize-quantize/{policy.h,policy.cpp}`,
  `src/arch/omnivoice/{quantization.cpp,omnivoice.h,weights.cpp,catalog.cpp}`.
- Task 11: `scripts/validate-omnivoice-replay.py`, possibly
  `tests/omnivoice_replay_real.cpp`, `tests/tolerances/omnivoice.json`.
- Task 12: `src/arch/omnivoice/quantization.cpp`,
  `docs/porting/families/omnivoice.md`.
- Task 13: `docs/quantization.md`.
- Task 14: `docs/porting/families/omnivoice.md`, the porting log,
  `docs/testing.md`, a Plan 7 carry-over ledger.

---

## Slice A — The two instruments everything else needs

### Task 1: The generator classifier, and the fourth QuantRole

**Files:**
- Modify: `src/arch/omnivoice/quantization.h`, `src/arch/omnivoice/quantization.cpp`
- Modify: `tools/synthesize-quantize/policy.h`, `tools/synthesize-quantize/policy.cpp`
- Modify: `tests/omnivoice_quantization_test.cpp`

**Interfaces:**
- Consumes: `classify_tensor(name, ne)` — the single classifier both the
  offline quantizer and the runtime catalog read, so an offline packing
  decision and a load-time expectation cannot drift.
- Produces: generator tensors resolving to `MatrixWeight` (2-D weights) /
  `RowLookup` (the two tables) / `Sensitive` (norms), and a
  `Profile::row_lookup_type` the resolver can read.

**Do not delete the pin — replace it.** `is_generator_tensor` currently returns
`Sensitive` for the whole generator; removing it drops every `llm.*` name
through to `tokens[0] != "codec"` → `Unknown` → `fail("unknown omnivoice
tensor")`. The replacement is a classifier modelled on `policy.cpp:171-191`'s
`classify_qwen3_block`, which already handles the identical Qwen3 block shape
(`self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj` as matrices;
`input_layernorm`, `post_attn_norm`, `q_norm`, `k_norm` as sensitive).

- [ ] **Step 1: Failing tests first.** Extend
  `tests/omnivoice_quantization_test.cpp`: every one of the 199 generator 2-D
  weight name shapes classifies `MatrixWeight`; every generator norm
  classifies `Sensitive`; `llm.embed_tokens.weight` and
  `audio_embeddings.weight` classify `RowLookup`; `audio_heads.weight`
  classifies `MatrixWeight` (it is a plain `mul_mat`, `generator.cpp:193`);
  a bare `llm` and any stray name still classify `Unknown`. Run → FAIL.
- [ ] **Step 2: `quantization.h`.** Add `QuantRole::RowLookup` with a comment
  stating the *architectural* reason, not a preference: these two tensors are
  the only weights read by `ggml_get_rows` (`generator.cpp:123,136`), and
  CUDA's `GET_ROWS` supports F16/F32/BF16/I32/Q1_0/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0
  and **no k-quant** (`ggml/src/ggml-cuda/ggml-cuda.cu:5190-5207`), so a
  k-quant here drops nodes to the CPU and fails the placement proof by name.
  Rewrite the `Sensitive` doc comment, which currently states the 2026-08-06
  codec-only ruling as standing policy; replace it with the 2026-08-09 ruling
  and keep the old one as dated history.
- [ ] **Step 3: `quantization.cpp`.** Replace `is_generator_tensor` with
  `classify_generator_tensor`. Record in a comment that all 199 2-D rows are
  1024/2048/3072 — divisible by 32 and 256, so both Q8_0 and every k-quant
  clear the block-size check — and that the generator emits no convolution,
  so there is no `ggml_im2col` destination-type hazard here at all (the whole
  reason the codec's 85 conv kernels needed Plan 4 Task 2's packed branch).
- [ ] **Step 4: `policy.h`/`policy.cpp`.** Add `ggml_type row_lookup_type` to
  `Profile` and a value for all three existing rows (`F16`→F16,
  `Q8_MIXED`→Q8_0, `Q5_K_MIXED`→Q8_0). State in the header comment that the
  value is **inert** for every family whose classifier never returns
  `RowLookup` — which today is all of them but omnivoice — so this field
  cannot silently change VITS, Kokoro or Qwen3-TTS packages. Add the
  `RowLookup` arm to `resolve_omnivoice_target_spec` returning
  `{ profile.row_lookup_type, TensorLayout::Native }`.
- [ ] **Step 5: The 2-D demotion still applies and must be checked, not
  assumed.** `quantize.cpp` demotes `PackedMatrix` → `Native` for
  `matrix_family` tensors with `ggml_n_dims < 3`, which is every generator
  2-D weight — correct (packing a `[1024, 3072]` projection would flatten it
  to one meaningless row of 3,145,728). Add a quantizer-fixture case asserting
  a generator projection lands `Native`, not `PackedMatrix`, so a future edit
  to that predicate cannot silently start packing the generator.
- [ ] **Step 6:** Both unit gates. `scripts/ci/clang-format.sh --check-diff`,
  literal exit code. Commit. **No profile exists yet** — this task changes
  role assignment only, and no cut package changes, because no profile maps
  `MatrixWeight` onto the generator until Task 3.

### Task 2: The speaker-identity F0 proxy, committed as a tool

**Files:** `scripts/omnivoice-speaker-proxy.py`,
`tests/python/test_omnivoice_speaker_proxy.py`, `tests/CMakeLists.txt`,
`CONTEXT.md`

Tier 3's instrument is load-bearing for three profiles in this plan and it
**does not exist as code**. `git show --stat 4c215e5` and `9d89524` land only
docs, the porting log, and card changes; a repository-wide grep for
`CMND` / `zero-crossing` / `normalised-cross-correlation` in `scripts/`,
`tools/` and `tests/` returns nothing. Every published F0 number in
`_porting-log.md` came from code that was thrown away. Three profiles measured
by three ad-hoc reimplementations are not comparable to each other or to the
1-of-17 baseline they must be read against.

- [ ] **Step 1:** Write `scripts/omnivoice-speaker-proxy.py`: takes two WAV/f32
  renders (or a directory pair), emits per-case JSON with median F0 from YIN
  (CMND, threshold 0.15, 1024-sample frames, 10 ms hop — the parameters the
  2026-08-08 entry records) and from an NCC tracker, the fraction of voiced
  frames below 165 Hz for each, the ratio, and the verdict under the committed
  criterion (complete register separation on both **and** >40% median shift on
  both). Same tool emits the degeneracy screen: DC offset, zero-crossing rate,
  20 ms-frame envelope ratio, voiced-frame fraction.
- [ ] **Step 2: Pin it to the published sweep.** The tool's acceptance is that
  it reproduces the 2026-08-08 backend audit: 1 of 17 changed,
  `omni-short-en` at YIN 140.9→199.0 Hz and NCC 118.2→189.0 Hz with the
  below-165 fraction 1.00→0.00 on both, `omni-digits` unchanged at 98.3% token
  flip, `omni-rate-fast` reported as *excluded, reference degenerate* rather
  than counted. If a re-render is needed to check this, it runs on
  `build/rel-dgx-spark`. If the tool cannot reproduce those numbers, the tool
  is wrong — the published numbers were human-confirmed in both directions and
  are the fixed point.
- [ ] **Step 3:** Unit tests that do not need a model:
  synthesized sine/sawtooth at known F0 (100/150/200/300 Hz, plus one with
  additive noise) → both estimators within tolerance; a DC-offset buzz and a
  sub-50 Hz tone → the degeneracy screen fires; silence → "no voiced frames",
  reported as *not comparable*, never as "unchanged". Register under
  `LABELS "unit;python;omnivoice"` beside `synthesize-omnivoice-python-unit`.
- [ ] **Step 4:** Add "speaker-identity F0 proxy" to CONTEXT.md with its
  criterion and its avoid-pairing (never "pitch check", never "voice
  similarity score" — it measures register separation, not similarity), and
  cite the two-sided human confirmation of 2026-08-08 as the reason it is
  citable.
- [ ] **Step 5:** Python/unit gates; `--check-diff`; commit.

---

## Slice B — `Q8_GEN`: Q8_0 generator over an untouched F32 codec

The first ship candidate: maximum gain (76.94% of the model, 3,189,953,504 →
1,390,700,256 bytes, a 56.4% reduction), minimum risk (the codec is
byte-identical to the shipped F32 package, so Tier 2 is trivially green and
Tier 4's exact `ref.tokens` gate is 0 by construction), and the clone path is
untouched.

### Task 3: The `Q8_GEN` profile, end to end

**Files:** `tools/synthesize-quantize/policy.cpp`,
`src/arch/omnivoice/{omnivoice.h,weights.cpp,catalog.cpp}`,
`tests/omnivoice_{quantization,catalog,metadata}_test.cpp`

**Naming is a public contract and this plan makes a recommendation, not a
silent choice.** `docs/porting/families/omnivoice.md` argues there is "no
family-specific name to invent" because `policy.cpp`'s table is shared and
each family's resolver gives a shared name a family-specific meaning. That
argument holds when a family needs *one* split per type. This family now needs
*two* splits at Q8_0 — generator-only (this task) and generator-plus-HuBERT
(Task 12's conv-exempt policy) — so one of them needs a second name.
Recommendation: add `Q8_GEN` for this task, and let Task 12 **redefine what
`Q8_MIXED` means for omnivoice** (it is BLOCKED, unpublished, and carries no
committed tolerance cell, so redefining it costs only a log entry preserving
the old meaning's sha256). **Confirm the name with jiangzhuo before cutting a
package** — it lands in GGUF metadata and on the model page.

- [ ] **Step 1: Failing tests first.** `find_profile("Q8_GEN")` resolves;
  a generator projection resolves Q8_0/`Native`; both lookup tables resolve
  Q8_0; every generator norm and every codec tensor resolves F32; the runtime
  accepts the profile string and rejects a package whose generator tensor
  carries the wrong type with the profile named in the message.
- [ ] **Step 2:** `policy.cpp` — one table row:
  `{ "Q8_GEN", GGML_TYPE_Q8_0, TensorLayout::PackedMatrix, GGML_TYPE_F32, GGML_TYPE_F32, GGML_FTYPE_MOSTLY_Q8_0, 1, /*row_lookup*/ GGML_TYPE_Q8_0 }`.
  The codec stays F32 because this profile's *classifier arm* holds it there,
  not because the table says so — see Step 3.
- [ ] **Step 3:** The codec must be `Sensitive` under `Q8_GEN` and
  `MatrixWeight` under `Q8_MIXED`, from one classifier. `classify_tensor`
  takes no profile today. Add the profile to the omnivoice resolver's dispatch
  rather than to `classify_tensor`'s signature: `resolve_omnivoice_target_spec`
  already has `profile` in hand, so it can map `MatrixWeight` → `F32` for
  codec-prefixed names under `Q8_GEN`. Keep the split expressed **once**;
  `catalog.cpp`'s `want_type()` must read the same decision, or an offline
  packing choice and a load-time expectation will drift — which is the
  standing reason this family has a shared classifier at all.
- [ ] **Step 4:** `omnivoice.h` — `QuantizationProfile::Q8Gen`;
  `weights.cpp:65-73` — accept the string; `catalog.cpp:242-276` — the
  `want_type()` arm and `profile_name()`. Extend the packed-shape acceptance
  comment at `catalog.cpp:147-155`: under `Q8_GEN` nothing is packed, so that
  branch is inert, and saying so beats a reader inferring it.
- [ ] **Step 5:** Both unit gates, `build` and `build-sanitize`;
  `--check-diff`; commit.

### Task 4: Cut the package, and the Tier 1 hard gates

**Files:** the porting log (a dated entry); no source changes expected.

- [ ] **Step 1:** `synthesize-quantize models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf
  models/omnivoice-0-6b/omnivoice-0-6b-Q8_GEN.gguf --quant Q8_GEN`. Record
  byte size and sha256. **Predicted: 1,390,700,256 bytes, 199 Q8_0 tensors.**
  A disagreement is a finding, not a rounding difference — the same arithmetic
  reproduces F32 and F16 exactly and `Q8_MIXED` to 1,248 bytes.
- [ ] **Step 2:** Load it: `synthesize-omnivoice-load-real`, the ABI
  round-trip, `synthesize-omnivoice-profile-test`. Catalog acceptance and the
  profile string must pass unmodified.
- [ ] **Step 3: Placement proof on CUDA.** Run the replay validator with
  `--accelerate --profile Q8_GEN --backend CUDA --tokens-informational` on
  `build/rel-dgx-spark` and confirm `validate-omnivoice-replay.py:624-633`
  reports the generator fully off the CPU on every case. Q8_0 is in CUDA's
  `GET_ROWS` list, so this must pass; it is the control measurement that makes
  Task 10's Q4 result interpretable.
- [ ] **Step 4: Structural invariance.** Identical sample count against the
  F32/CPU render on all 17 greedy cases, both arms. Grid SIZE match is already
  gated unconditionally; assert the sample counts separately, since that is
  what a listener's file length is.
- [ ] **Step 5: Public seed contract.** `scripts/validate-omnivoice-public.py`
  against the new package: same seed → byte-identical, different seeds differ,
  `random` reports concrete. This does not depend on token identity and must
  keep holding.
- [ ] **Step 6:** Porting log entry with every Tier 1 number. `--check-diff`;
  commit.

### Task 5: Tier 2 — the codec half, and the tolerance cells

**Files:** `tests/tolerances/omnivoice.json`, the porting log

- [ ] **Step 1:** Full 20-case CPU replay:
  `scripts/validate-omnivoice-replay.py --require all --margin-report
  --profile Q8_GEN --backend CPU --stage replay --tokens-informational`,
  fresh `--work` directory. (The `--work` reuse trap that produced a false
  green during Plan 4 Task 3 is documented in the family doc; use a fresh
  directory and check the runner's stderr directly.)
- [ ] **Step 2: The strong assertion this profile is owed.** `audio.pcm`,
  `audio.pcm_freerun` (where the grid matches primary), `ref.pcm_16k`,
  `ref.semantic_mean`, `ref.fused_latent` and `ref.tokens` all run entirely
  through F32 codec weights, so they must reproduce the F32 cell's observed
  figures **to the bit** — not "within tolerance". Assert equality against
  `profiles.F32.stages.replay`'s committed `observed_*` values and treat any
  deviation as a bug in Task 3's classifier, not as drift.
- [ ] **Step 3:** `ref.tokens` exact: **2/2**. This is Tier 4's hard gate and
  it is 0 by construction here. If it is not 2/2, stop — something in the
  generator's role change reached the codec.
- [ ] **Step 4:** Commit a `profiles.Q8_GEN.stages.replay` cell. The generator
  probes (`hidden_l{0,7,14,21,27}`, `final`, `logits_step0`) WILL move
  substantially — qwen3-tts's Q8 talker sits at cosine 0.9957 for comparison —
  so commit them at 5× the measured `(1 - cosine)`, rounded down to six
  decimals, this family's established rule; the codec/reference probes carry
  the F32 numbers unchanged. Write the cell's `note` in the file's own voice,
  saying plainly that the codec figures are inherited-exact and the generator
  figures are the profile's actual cost.
- [ ] **Step 5:** Record the per-case token agreement **as data** in the
  porting log — a table, not a verdict. `--check-diff`; commit.

### Task 6: Tier 3 and Tier 4 — does it still sound like the same person

**Files:** the porting log

- [ ] **Step 1:** Render all 17 greedy cases at the shipped 32-step default,
  greedy free-run, one binary, `build/rel-dgx-spark`, F32/CPU arm vs
  Q8_GEN/CPU arm — one variable. Check `/proc/loadavg` first.
- [ ] **Step 2:** Task 2's tool over both arms. Report **N of 17 changed**
  against the accepted 1-of-17 backend baseline, plus the degeneracy screen on
  every render (ZCR in 3.4–4.2 kHz, envelope ratio 218–1988, DC, voiced
  fraction). Exclude `omni-rate-fast` explicitly with its reason, as the
  2026-08-08 sweep did, rather than silently.
- [ ] **Step 3:** Repeat the sweep for the Q8_GEN/**CUDA** arm against
  F32/CPU, so the profile's cost and the backend's cost are separable rather
  than compounded in one number.
- [ ] **Step 4: Tier 4, target-relative.** On `omni-clone-en` and
  `omni-clone-zh`, measure the candidate's median F0 and register against
  **the reference clip's own**, not against the F32 render. Both cases pin the
  *same* clip (`models/omnivoice-reference-audio/seedtts_ref_en_1.wav`) — which
  is why they produce identical `ref.*` figures — so the two cases test
  whether the target survives across two languages, not two targets. A profile
  may drift from F32 and still be a faithful clone; the question the card
  answers is "does it sound like the target".
- [ ] **Step 5:** Porting log, dated, with the full per-case table and the
  method by reference to Task 2's tool rather than re-described.
  `--check-diff`; commit.

### Task 7: RTF, on the Release tree only

**Files:** the porting log

- [ ] **Step 1:** `/proc/loadavg`; repeat anything above ~1.5 other-cores.
- [ ] **Step 2:** `synthesize-omnivoice-public-real` on `build/rel-dgx-spark`,
  `omni-long-boundary`'s committed text, seed 0, language `en`,
  `max-frames 0`, through the public seam (`synth_synthesize_to_buffer`, not
  the replay runner, which bypasses the backend-capability gate). Four runs:
  {F32, Q8_GEN} × {cpu, cuda}. Record `synthesis_seconds`, `frames`,
  `sample_rate` from each run's own JSON line.
- [ ] **Step 3:** RTF = `synthesis_seconds / (frames / sample_rate)`. Compare
  against Plan 5's honest anchor on the same case and host: **122.61 s /
  RTF 4.263 (CPU) vs 5.481 s / RTF 0.1906 (CUDA), 22.4×**. Report plainly,
  including if Q8_0 is *slower* than F32 on CUDA — a Q8_0 matmul that
  dequantizes per call can be, and this family has already published one
  wrong speedup. Note whether the 3.19 GB → 1.39 GB drop changes what fits on
  a device, which may matter more than the RTF.
- [ ] **Step 4:** Porting log; `--check-diff`; commit.

### Task 8: Register, document, decide

**Files:** `tests/CMakeLists.txt`, `docs/porting/families/omnivoice.md`,
`docs/testing.md`, `docs/models/omnivoice-0-6b.md`,
`scripts/hf_cards/omnivoice-0-6b.yaml`

- [ ] **Step 1: Offer the Listening Audit before claiming anything.** A
  tolerance grid is not audible evidence — this family's standing rule. Build
  the blind A/B pairs (F32/CPU vs Q8_GEN, order seed and case seed recorded)
  and ask jiangzhuo. Record the verdict dated, alongside the two existing
  audits; neither replaces the other.
- [ ] **Step 2:** Register `synthesize-omnivoice-replay-golden-q8gen`,
  **guarded on the package's existence** via a `SYNTH_OMNIVOICE_Q8GEN_TEST_MODEL`
  cache variable defaulting to `models/omnivoice-0-6b/omnivoice-0-6b-Q8_GEN.gguf`,
  registering only when the file is there — the pattern
  `tests/CMakeLists.txt` already uses for every model-guarded target. It runs
  with `--tokens-informational`, like the CUDA gate. Verify the target is
  non-vacuous: make it fail once on purpose (perturb a tolerance) and confirm
  CTest reports it red.
- [ ] **Step 3:** Family doc — a `Q8_GEN` section under "Quantization Profile
  Shape" with all four tiers' evidence, and a rewrite of that section's
  opening paragraph, which still asserts "Every produced Quantization Profile
  must re-pass the full suite including the exact-token gates". That sentence
  is superseded for the generator half and still binding for `ref.tokens`;
  say exactly that.
- [ ] **Step 4:** `docs/testing.md` — the new gate, its runtime, its guard.
  Status date. `docs/models/omnivoice-0-6b.md` + the card YAML: the profile
  table, the size, the RTF, the validation-level prose. Regenerate, `--check`.
  **Do not publish.**
- [ ] **Step 5:** Both unit gates, the full CPU integration set, the new gate,
  card `--check`, `--check-diff` — all with literal exit codes. Commit.

---

## Slice C — Q4, at least measured for speed and quality

### Task 9: Define the Q4 profile, and pin the lookup tables

**Files:** `tools/synthesize-quantize/{policy.h,policy.cpp}`,
`src/arch/omnivoice/{quantization.cpp,omnivoice.h,weights.cpp,catalog.cpp}`

`Q4_K_M` is llama.cpp's mixture name and does not exist in this project.
This task defines the profile before quoting any size.

- [ ] **Step 1: Define the composition and compute the size from it.**
  Recommendation: `Q4_K_GEN` — `Q4_K` for all 199 generator 2-D weights
  **except** the two `RowLookup` tables, which Task 1's role pins to `Q8_0`;
  codec F32; norms F32. Computed size **1,166,300,896 bytes**, and the pin
  costs **81,856,512 B (78.1 MiB)** against a plain-Q4_K baseline of
  1,084,444,384. If a Q6_K arm is wanted for `embed_tokens` (llama.cpp's
  `token_embd` convention), that is a *different* profile and gets its own
  row, its own name and its own recomputed size — do not blend the two and
  quote a number from neither.
- [ ] **Step 2:** The table row, with `row_lookup_type = GGML_TYPE_Q8_0` and
  a comment giving the architectural reason (Task 1 Step 2's), not a
  preference. Enum, `weights.cpp` string acceptance, `catalog.cpp`
  `want_type()`/`profile_name()`, all with failing tests first.
- [ ] **Step 3: Prove the pin is load-bearing, don't assert it.** Add a unit
  test asserting `classify_tensor("llm.embed_tokens.weight", …)` is
  `RowLookup` and that `resolve_omnivoice_target_spec` under `Q4_K_GEN`
  returns `Q8_0` for it and `Q4_K` for a projection. The runtime consequence
  is measured in Task 10 Step 2.
- [ ] **Step 4:** Both unit gates; `--check-diff`; commit.

### Task 10: Cut Q4, measure all four tiers and RTF, decide

**Files:** the porting log, `tests/tolerances/omnivoice.json` (only if the
profile survives), `docs/porting/families/omnivoice.md`

- [ ] **Step 1:** Cut. Record size and sha256 against Task 9's prediction.
- [ ] **Step 2: The placement proof is the point of this task.** Run the CUDA
  replay with `--accelerate`. With the pin: the generator must be fully off
  the CPU. **Then cut a second, throwaway package with the pin removed** (a
  local patch, not committed) and confirm `validate-omnivoice-replay.py:624-633`
  fails by name — that is the evidence the pin is necessary rather than
  cautious, and it costs one extra cut. Record both outcomes.
- [ ] **Step 3:** Tier 1 (size, sha256, load, structural invariance, seed
  contract), Tier 2 (codec figures must again be bit-inherited from F32),
  Tier 4 (`ref.tokens` 2/2 exact by construction).
- [ ] **Step 4:** Tier 3 — Task 2's tool, 17 cases, CPU and CUDA arms, plus
  the degeneracy screen. Q4 is where the screen earns its keep: a generator
  degraded past usefulness produces sub-50 Hz output with DC≈0 and no
  envelope, and that must be caught by measurement, not by someone happening
  to listen.
- [ ] **Step 5:** RTF, Task 7's exact method, `build/rel-dgx-spark`,
  `/proc/loadavg` checked. jiangzhuo asked for Q4 to be measured for **speed
  and quality**; report both, and report honestly if Q4 is no faster than
  Q8_0 — `Q5_K_MIXED` on qwen3-tts bought 324 MiB and "essentially no speed"
  (0.82 vs 0.85 RTF, inside run-to-run variation), which is the precedent to
  check against, not to assume.
- [ ] **Step 6:** Decide with jiangzhuo, offering a Listening Audit. If it
  ships: tolerance cell, guarded gate, docs, card, exactly as Task 8. If it
  does not: record BLOCKED-or-not-recommended in the family doc with all the
  numbers, keep the table row (the `Q5_K_MIXED` precedent: "correct and cheap
  to keep — one row, one enum, one branch"), commit **no** tolerance cell and
  register **no** target. `--check-diff`; commit.

---

## Slice D — The conv-exempt codec policy

### Task 11: The replayed-grid free-run channel (hard prerequisite for Task 12)

**Files:** `scripts/validate-omnivoice-replay.py`, possibly
`tests/omnivoice_replay_real.cpp`, `tests/tolerances/omnivoice.json`

**What is actually missing, verified — the brief's framing is close but not
exact, and the difference matters for what gets built.** The replayed-grid
*codec* comparison already exists and already gates per profile: `audio.pcm`
replays the **oracle's** grid through the candidate's codec
(`validate-omnivoice-replay.py:433`) and is the channel that measured
`Q8_MIXED`'s codec drift. What does **not** exist is any comparison at all on
the **free-run** waveform once the grid drifts: `audio.pcm_freerun` is added
to `measurements` only in the `grid["matched"] == "primary"` branch
(`:450-455`), and when the grid matches nothing the result is
`{"mode": "not-compared"}` (`:473-477`) — a mode neither reporting branch at
`:594-615` handles. It is dropped **silently**. So today, on every
generator-quantized or CUDA-generator run, the only artifact that "can tell a
wired-up codec from a codec that merely exists" (`omnivoice_replay_real.cpp`'s
own header comment) is not compared to anything.

This is pre-existing debt — the shipped CUDA-generator path already has it —
not something this plan creates. `Q8_GEN` and `Q4_K_GEN` inherit it but are
not blocked by it: their codec is byte-identical to the shipped F32 package,
so the free-run decode machinery is already evidenced by the F32 gate and a
dark channel hides no new risk. Task 12's codec **changes**, so a dark channel
there would hide a real regression. Hence: prerequisite for Task 12, not for
Slice B.

- [ ] **Step 1:** Handle `mode == "not-compared"` explicitly. At minimum it
  must print and, absent the new comparison, count as a failure rather than
  vanish — a silently-skipped channel is the failure mode this project has
  been bitten by before.
- [ ] **Step 2:** Add the comparison. `--alt-grid` is the existing seam
  (`omnivoice_replay_real.cpp:179-184,438-455`): pass the **F32/CPU
  reference's own committed primary grid**, have the candidate decode it
  through its own codec, and compare that decode against the F32/CPU
  reference's own committed free-run waveform under a committed
  cosine/max_abs cell. Note the runner currently accepts exactly one
  `--alt-grid` and the validator raises if a case pins more than one
  (`:395-397`); a case that pins a dual-admissible alternate **and** needs the
  reference grid needs the channel widened — decide and implement, do not
  leave it to raise at runtime.
- [ ] **Step 3:** Prove the new channel is non-vacuous: run it against
  `Q8_GEN` (codec F32) and require bit-identity, then perturb deliberately
  and confirm it fails.
- [ ] **Step 4:** Commit the cell for F32/CUDA too — that path has been running
  with this channel dark since Plan 5 Task 1 and this is the moment it stops.
- [ ] **Step 5:** Python/unit gates, the full CPU integration set, the CUDA
  gate; `--check-diff`; commit.

### Task 12: Adopt the conv-exempt codec policy — and put the arithmetic first

**Files:** `src/arch/omnivoice/quantization.cpp`,
`docs/porting/families/omnivoice.md`, the porting log

The policy: exempt 3-D `MatrixWeight` tensors from packing, i.e. never
quantize a convolution — the reference port's de-facto policy, and the reason
their codec drifts **0.7%** of RVQ codes where our `Q8_MIXED` drifts **36.4%**.
Our conv-packing *capability* is precisely why our codec profile is worse than
theirs.

- [ ] **Step 1: STOP and show jiangzhuo the arithmetic before cutting
  anything.** This plan's Verified Facts section establishes, from the convert
  report, that for **this** family the policy has an inverted cost/benefit:
  all 85 conv kernels include the **entire acoustic decoder** (32 tensors), so
  a conv-exempt codec leaves the decode path — the half every synthesis uses —
  100% F32 and gains nothing; and all 73 two-dimensional codec matrix weights
  live in `codec.semantic_model`, i.e. **only** on the clone-encode path,
  which is exactly the path whose gate is exact. Resulting package:
  **2,939,302,880 bytes, 236 MB LARGER than today's `Q8_MIXED`**. And the
  reference port's own 0.7% says the expected `ref.tokens` result is
  "much better, still not zero" — which under this family's unchanged exact
  gate is still BLOCKED. Present this; let jiangzhuo decide whether to adopt
  the policy as stated policy (defensible: it is the right general rule and
  0.7% ≫ 36.4% as engineering) or to record the finding and adopt something
  narrower. Do not decide it here.
- [ ] **Step 2 (if adopted):** In `classify_codec_matrix_region`, return
  `Sensitive` for any weight that is a convolution kernel. Name the
  `[7,32,1]` collapsed case (`codec.acoustic_decoder.conv2.weight`)
  explicitly — `ggml_n_dims` reports it as two-dimensional and a rule written
  as "`ggml_n_dims >= 3`" would quantize the one conv kernel the policy most
  obviously means to exempt. The three already-`Sensitive` named exceptions
  become redundant under the new rule but stay, with their comments amended to
  say so; deleting them would lose the reasons.
- [ ] **Step 3:** Decide and record: does `Q8_MIXED` change meaning for this
  family, or does the conv-exempt split get a new name? Recommendation:
  redefine `Q8_MIXED`, because it is BLOCKED, unpublished, and carries no
  committed tolerance cell — but the porting log's 36.4% measurement names a
  specific sha256, so the log entry must state that the old meaning is no
  longer reproducible from the same command line and pin what it was.
- [ ] **Step 4: Say plainly what is being given up.** Adopting this policy
  means this family deliberately stops using the packed-convolution branch
  this project built and ported for it: `codec_conv1d` in
  `src/arch/omnivoice/codec.cpp` and `conv1d` in
  `src/arch/omnivoice/reference-encoder.cpp` (Plan 4 Task 2), the packed
  shape acceptance at `catalog.cpp:147-155`, the `omnivoice_collapsed_conv_kernel`
  carve-out in `quantize.cpp`, and the `feat_conv[1..6]` load-path fix that
  cost a whole debugging session to find. Keep the code — it is shared with
  VITS, Kokoro and Qwen3-TTS, and `check_packed_feat_conv1` /
  `check_packed_encoder_semantic_conv` keep it covered cheaply — but state in
  the family doc that omnivoice no longer exercises it, so nobody reads its
  presence as evidence that this family packs convolutions.
- [ ] **Step 5:** Cut, then measure: `ref.tokens` exact count (the number that
  decides everything), the codec probes, Task 11's free-run channel, size, and
  RTF. If `ref.tokens` is not 2/2, the profile is BLOCKED: family-doc section
  with the measurements, **no** tolerance cell, **no** registered target,
  exactly as `Q8_MIXED` is recorded today.
- [ ] **Step 6:** Both unit gates, the full CPU integration set;
  `--check-diff`; commit.

---

## Slice E — Policy documentation and close-out

### Task 13: `docs/quantization.md` — line 103, and the per-family sections

**Files:** `docs/quantization.md`

The rule, verbatim at lines 103–106: *"No Q4 or Q5 mixed profile is committed
until `Q8_MIXED` has been calibrated on a real Model Variant and the additional
profile has independent port-validation, memory, and backend evidence."*

Note the brief paraphrased this as "calibrated on a **second family**"; the
text says "a real Model Variant". Both readings must be answered, and the
answer written down so it is not re-litigated.

- [ ] **Step 1: Establish whether the condition is met, and record the
  reading.** Project-wide reading (what the text says, and where the paragraph
  sits — in the shared policy section, before any family section): **met.**
  VITS `Q8_MIXED` v1 functionally validated 2026-07-23, Kokoro 2026-07-26
  (the file's own Status line), and qwen3-tts carries committed `Q8_MIXED`
  tolerances. Per-family reading: **not met** — omnivoice's `Q8_MIXED` is
  BLOCKED and was never calibrated. Record both, adopt the project-wide
  reading as the doc's stated meaning, **and** record explicitly that
  jiangzhuo's 2026-08-09 decision covers the per-family reading regardless, so
  the answer does not depend on which reading a future reader picks.
- [ ] **Step 2:** Record the precedent that resolves the rest: `Q5_K_MIXED`
  already exists in `policy.cpp` and is *not* a violation, because it "carries
  no committed tolerances and no published package, which is the accurate way
  to say buildable, not validated" (`docs/porting/families/qwen3-tts.md`).
  Write into `docs/quantization.md` that "committed" in line 103 means
  *validated and published*, not *present in the profile table* — which is
  exactly what makes Slice C legal before its evidence exists.
- [ ] **Step 3:** Add an OmniVoice section. This doc currently mentions
  neither omnivoice nor qwen3-tts, so it is describing a three-family project
  that has four families. At minimum: this family's split is
  **generator-first**, the inverse of Kokoro's and VITS's, because 76.94% of
  the bytes are in the generator and the codec's quantizable half sits
  entirely on the clone-encode path; the conv-exempt codec policy and its
  reason; and the four-tier evidence shape, since the exact-token gate that
  every other family's profiles are judged by cannot judge these.
- [ ] **Step 4:** Bump the Status line with today's date and what changed.
  `--check-diff`; commit.

### Task 14: Close-out

**Files:** `docs/porting/families/omnivoice.md`, the porting log,
`docs/testing.md`, a Plan 7 carry-over ledger

- [ ] **Step 1:** Family doc: the final profile table (what ships, what is
  blocked, what is buildable-and-not-recommended), every Listening Audit
  entry dated separately, and the Tier 3 N-of-17 figures side by side so a
  reader can compare profile cost against backend cost.
- [ ] **Step 2:** Porting log: one dated close-out section with every task's
  numbers gathered in one place, including the two corrections this plan made
  to the numbers it was briefed with.
- [ ] **Step 3:** `docs/testing.md`: the new gates, their guards, their
  measured runtimes; Status date.
- [ ] **Step 4:** Final gates, all quoted with literal exit codes: both unit
  trees; the full omnivoice integration set on the CPU tree; the CUDA gates;
  every newly registered guarded gate; card `--check`;
  `scripts/ci/clang-format.sh --check-diff`.
- [ ] **Step 5:** Carry-over ledger: whatever remains, at minimum the four
  project-wide questions and the float64 cosine-estimator question inherited
  from Plan 5 and not resolved here, plus anything Tasks 9–13 opened and
  deferred. Commit.

---

## Self-Review Notes

- **Sequencing, and why it is this order.** Task 1 is unconditional: the
  quantizer refuses the package outright without it, so nothing else can even
  be cut. Task 2 is placed before any measurement because three profiles
  measured by three throwaway reimplementations of the F0 proxy are not
  comparable to each other or to the 1-of-17 baseline — and the tool genuinely
  does not exist today, which a grep of `scripts/`, `tools/` and `tests/`
  confirms. Slice B before Slice C because `Q8_GEN` is the profile whose
  codec-half evidence is free. Task 11 before Task 12 and *not* before Slice B,
  for the reason Task 11 itself states.
- **Three STOP-shaped moments this plan is honest about.** Task 3 Step 1
  (profile naming is a public contract and this family's own doc argues
  against new names, so the recommendation needs confirmation before a package
  carries it). Task 10 Step 2 (deliberately cutting a throwaway unpinned Q4
  package to *prove* the placement gate fails, rather than asserting it from
  the ggml source). Task 12 Step 1 (the conv-exempt arithmetic, which says the
  package gets bigger, the decoder gains nothing, and the exact gate probably
  still fails — jiangzhuo decided the policy before this arithmetic existed,
  and he should see it before it is cut).
- **What this plan corrects in its own brief, and why that is in the document
  rather than only in a hand-off note.** Two of the briefed numbers do not
  reproduce (the Q4 sizes and the ~41.8 MB pin cost), and one framing is
  imprecise (the "`--alt-grid` is unbuilt" claim — the replayed-grid *codec*
  channel exists and gates; the *free-run* channel is the one that goes dark).
  This project has been burned by transcribing a prior finding into a confirmed
  doc without re-checking, which is why Plan 5 Task 6 Step 1 exists; the same
  discipline applies to a plan's own inputs.
- **Why Tier 2 gets a bit-identity assertion instead of a tolerance for
  `Q8_GEN` and `Q4_K_GEN`.** Their codec weights are the F32 package's
  weights. "Within tolerance" would pass a classifier bug that leaked a
  quantized type into a codec tensor; bit-identity would not. A tolerance cell
  is still committed for the generator probes, where the drift is the
  profile's actual, intended cost.
- **Why no task proposes relaxing `ref.tokens`.** It is the one gate whose
  input is a fixed clip and whose output is discrete, so it is the one place
  where "different valid realization" is not an available defence. Slice B
  ships *because* it keeps that gate at 0 by construction, and Slice D is
  expected to be blocked by it — which is information, not an obstacle to
  route around.
- **Spec coverage:** jiangzhuo's three decisions map 1 → Slice B (Tasks 3–8),
  2 → Slice C (Tasks 9–10), 3 → Slice D (Tasks 11–12). The three
  must-not-be-discovered-mid-flight items map: the `--alt-grid` prerequisite →
  Task 11 (with its framing corrected); `docs/quantization.md` line 103 →
  Task 13; tolerance cells, golden gates and no-permanently-red-tests → the
  Global Constraints plus Tasks 5, 8, 10 and 12. The packed-conv branch this
  family gives up → Task 12 Step 4, stated plainly rather than left to pass
  silently.
