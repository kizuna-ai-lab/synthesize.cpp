# OmniVoice Plan 3: Public Sampling + Cloning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (the standing choice for this family) to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Open the OmniVoice public synthesis seam — sampled decoding with the
project's own seed contract — and land the family's headline capability:
voice cloning from Reference Audio plus voice design from Description Text,
with Serialized Profile round-trip, validated to exact reference-token parity.

**Architecture:** Plan 2 left a greedy CPU synthesis core behind a
`synthesis.not_implemented` public stub. Plan 3 (design spec slices 7, 7.5, 8)
adds: (a) the Gumbel sampler in the existing host decode loop, seeded by this
port's own `NormalRandomStream` (upstream has no seed parameter); (b) the
public dispatch branch in `src/synthesize.cpp` on the qwen3-tts template, plus
the public-request validator that proves the seed contract; (c) the first real
implementation of the core Voice Profile subsystem (today an
`SYNTH_ERR_UNSUPPORTED_VOICE` stub for every family) and of the ADR 0009 Audio
Normalizer (today design-only); (d) the clone-encode chain — internal 24→16 kHz
resampler, HuBERT semantic branch, DAC acoustic encoder, RVQ encode — whose
output token grid is compared to the oracle's **exactly**, token for token.

**Tech Stack:** C11/C++17, GGML (submodule, untouchable), CMake ≥3.24, uv-locked
Python oracle env `scripts/envs/omnivoice/`, vendored libsamplerate 0.2.2 (new).

## Global Constraints

- Commit message footer, every commit in this plan:

  ```text
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01NFjueGrsKXs44rfZ7DunQK
  ```

- clang-format: run `scripts/ci/clang-format.sh --fix` **after `git add` of any
  new files and before the commit** in every task touching C/C++ (the script
  selects files via git — untracked files are invisible to it). The verified
  order per commit: `git add` → `--fix` → `git add` again → commit. The FINAL
  command of every task is `scripts/ci/clang-format.sh --check-diff` with its
  literal exit code reported. Never format `ggml/`.
- Every C++ slice ends with unit tests registered under the `unit` label AND a
  clean sanitizer run: `cmake --build build-sanitize --target synthesize-check-unit`
  (configure both trees per CLAUDE.md if fresh).
- Oracle invocation, always:
  `uv run --project scripts/envs/omnivoice --locked python …` (F32, CPU).
  Re-locking that env requires `uv lock --no-sources` (comment in its
  pyproject.toml; the upstream package routes torch to a CUDA wheel index
  otherwise).
- **Licensing:** never label any artifact apache-2.0. Weight licenses: LM
  CC-BY-NC (version unstated) + Boson Higgs Audio 2 Community (codec) +
  Apache-2.0 (code). `rockerritesh/omnivoice-tts.cpp` is PolyForm
  Noncommercial: read-only, zero code reuse. Code adopted from
  ServeurpersoCom (MIT) or bluryar (Apache-2.0) records its notice in
  `THIRD_PARTY_NOTICES.md` at adoption time. libsamplerate is BSD-2-Clause —
  notice required when vendored (Task 8).
- **Terms (CONTEXT.md):** "Synthesis" not "generation"; "Voice Profile" /
  "Voice Reference" / "Serialized Profile" / "Profile Compatibility ID" /
  "Reference Frame Equivalent" as defined; "Linguistic Input" not "prompt" in
  public docs; "Restricted Model Package" never "Published Model Package"; no
  Chunked Audio Delivery / Native Streaming Synthesis claims.
- **Exact-token discipline:** discrete comparisons (reference tokens, code
  grids) are exact, widened only by enumerated dual-admissibility — never a
  tolerance. Float probes get measured-then-committed tolerances
  (reference_stage `source-f32-oracle-vs-f32-cpu`), derivation margin ≤5×,
  committed before support is declared. The ≥1e-4 margin screen governs NEW
  greedy golden cases only.
- **Placement:** discrete decisions (argmax, nearest-neighbour token choice,
  sampling) always run on host CPU. Rounded/sampled values and their whole
  input path stay on CPU.
- **jiangzhuo's rulings for this plan (2026-08-01):**
  1. `ref_rms == 0` (digitally silent reference) → **rejected at profile
     creation** with `SYNTH_ERR_INVALID_ARG` and a named diagnostic; a
     deliberate divergence from upstream's silent all-zero output, recorded in
     the family doc.
  2. Clone-encode probes: **keep `ref.semantic_hidden` (final layer), add
     `ref.semantic_mean` (the consumed mean-of-13) and `ref.fused_latent`
     (fc fusion output, the RVQ nearest-neighbour input)**; plus `ref.pcm_16k`
     for the resampler boundary. Re-dump the two clone cases; manifest
     `suite_version` 2→3; new-probe tolerances measured then committed.
  3. Settled from the recorded rationale (not a new ruling): sampled
     seed-contract cases get **no margin screen** — the screen keeps
     oracle-parity claims off arithmetic knife edges, and the public sampled
     path claims relations between runs of this port, not oracle agreement.
     Record in the family doc.
- The three sampled manifest cases pin `class_temperature 0.0` and
  `position_temperature 5.0`; the public path applies **package defaults**
  (no temperature crosses the public ABI, per the qwen3-tts precedent).
- Upstream fidelity constants (transcribed from the pinned oracle,
  `omnivoice @ 468e927b`, transformers 5.14.1 — cited per task below):
  Gumbel: `g = -log(-log(u + 1e-10) + 1e-10)`, applied as `logits/T + g`;
  class branch top-k keeps `ceil(0.1 * 1025) = 103` classes; HuBERT semantic
  branch: resample 24→16 kHz, `pad (160, 160)`, mean of ALL 13 hidden states,
  then `[::2]` downsample; RVQ: plain L2 nearest neighbour on `project_in`
  outputs, no normalization; transcript punctuation rule appends `.` / `。`;
  reference wav tail-clipped to a whole number of 960-sample frames; input
  boost `* 0.1/ref_rms` iff `0 < ref_rms < 0.1` (stored ref_rms stays
  pre-boost); output arms: `ref_rms >= 0.1` → none, `0 < ref_rms < 0.1` →
  `* ref_rms/0.1`, no reference → peak-normalize to 0.5 iff peak > 1e-6.
- **Standing facts (do not rediscover):** greedy makes zero RNG calls; the
  commit schedule is float32 torch-elementwise (pinned by the
  `commit_schedule(1640, 32, 0.1)` fixture); embedding merge is a torch.where
  SELECT; any case-text change re-runs EVERY oracle dumper; golden suite
  enters this plan at revision 2 and leaves it at revision 3.

## File Structure

New files:
- `src/audio-normalizer.{h,cpp}` — core ADR 0009 module (validate → channel →
  resample; libsamplerate).
- `third_party/libsamplerate/` — vendored 0.2.2 (Task 8 pins the exact set).
- `src/arch/omnivoice/reference-encoder.{h,cpp}` — clone-encode graph builders
  (HuBERT, acoustic encoder, fusion) — the graph half.
- `src/arch/omnivoice/reference-encoder-host.{h,cpp}` — host half: 24→16 kHz
  resampler, ref_rms/boost, RVQ nearest-neighbour encode, hop clip.
- `src/arch/omnivoice/profile.{h,cpp}` — Voice Profile payloads (clone prompt,
  description), Serialized Profile GGUF writer/reader.
- `scripts/validate-omnivoice-public.py` + `tests/omnivoice_public_real.c` —
  public-seam validator pair (qwen3-tts template).
- Tests: `tests/omnivoice_sampler_test.cpp`, `tests/audio_normalizer_test.cpp`,
  `tests/omnivoice_resampler_test.cpp`, `tests/omnivoice_reference_encoder_test.cpp`,
  `tests/omnivoice_profile_test.cpp`, plus extensions of existing suites.

Modified (main): `src/arch/omnivoice/{omnivoice.h, model.cpp, generator-host.*,
frontend-host.*, codec-host.*}`, `src/{synthesize.cpp, voice-profile.cpp,
synthesis-request.{h,cpp}, model-info.h}`, `include/synthesize.h` (no ABI
additions expected — verify, do not add), `scripts/dump_reference_omnivoice_pytorch.py`,
`scripts/validate-omnivoice-replay.py`, `tests/omnivoice_replay_real.cpp`,
`tests/golden/omnivoice/omnivoice-0-6b.manifest.json`,
`tests/tolerances/omnivoice.json`, `tests/CMakeLists.txt`,
`docs/porting/families/omnivoice.md`.

---

### Task 1: First-touch debt, C++ batch

**Files:**
- Modify: `src/arch/omnivoice/model.cpp` (delete `GraphRun::nodes_`, ~lines 68, 147)
- Modify: `tests/tolerances/qwen3-tts.json` (line 5: `suite_version` 1 → 2)
- Modify: `reports/porting/qwen3-tts/qwen3-tts-12hz-0-6b-customvoice/_porting-log.md`
  (append one dated log note)

**Interfaces:** none produced; purely mechanical.

- [ ] **Step 1:** In `model.cpp`, remove the `nodes_` member from `GraphRun`
  and its single initialization site (grep `nodes_` — it is written once, read
  never; qwen3-tts's twin was removed the same way).
- [ ] **Step 2:** In `tests/tolerances/qwen3-tts.json` change `"suite_version": 1`
  to `"suite_version": 2` (its manifest has been at 2 since the clone-zh-era
  revision; same staleness class as omnivoice's 0887553 fix).
- [ ] **Step 3:** Append to the qwen3-tts porting log, dated 2026-08-01: one
  paragraph noting the tolerance file's suite_version was stale at 1 against
  the manifest's 2, corrected as a sibling of omnivoice commit 0887553; no
  thresholds changed.
- [ ] **Step 4:** Build + run: `cmake --build build --target synthesize-check-unit`
  (expect all pass; the omnivoice unit set was 78/78 at Plan 2 close, plus the
  three tests added in the PR #5 review round).
- [ ] **Step 5:** Sanitizer tree: `cmake --build build-sanitize --target synthesize-check-unit`.
- [ ] **Step 6:** Commit (`git add` → format-fix → `git add` → commit; footer).
  Final command: `scripts/ci/clang-format.sh --check-diff`, report exit code.

### Task 2: First-touch debt, Python batch — one pinned-input set, two consumers

**Files:**
- Modify: `scripts/convert-omnivoice.py` (its `PinnedInput` list)
- Modify: `scripts/dump_reference_omnivoice_pytorch.py` (its `/resolve/`-marker inference)
- Test: `tests/python/test_convert_omnivoice.py` (new cross-check test)

**Interfaces:**
- Produces: a single module-level pinned-input table importable by both
  consumers. Put it in `scripts/omnivoice_pinned_inputs.py` as
  `PINNED_INPUTS: list[PinnedInput]` where
  `PinnedInput = namedtuple("PinnedInput", "role relative_path sha256_key")`,
  derived from what the two scripts pin today (the converter's list is the
  richer one; the dumper infers paths from `/resolve/` markers in manifest
  locator URLs). The six pinned inputs the converter verifies at startup are
  the authoritative set.

- [ ] **Step 1:** Read both current implementations; extract the union of
  pinned inputs (they must agree — a disagreement is a finding to fix, not
  paper over).
- [ ] **Step 2:** Create `scripts/omnivoice_pinned_inputs.py` holding the
  table plus a `resolve_local(weights_dir, pin) -> Path` helper. Docstring
  states: "One pinned set, two consumers (converter, oracle dumper); Plan 3
  carryover item 4."
- [ ] **Step 3:** Point `convert-omnivoice.py` at it (delete its private
  list). Point `dump_reference_omnivoice_pytorch.py`'s digest-verification at
  it (keep the manifest-locator plumbing for artifact URLs; the shared table
  governs which local weight/config/tokenizer files get digest-checked).
- [ ] **Step 4:** Add to `tests/python/test_convert_omnivoice.py`: a test that
  imports both consumers and asserts each consumes `omnivoice_pinned_inputs.PINNED_INPUTS`
  (identity or content equality with the table — e.g. the converter's
  startup-verification list is built from it).
- [ ] **Step 5:** Run the omnivoice python unit target:
  `ctest --test-dir build --output-on-failure -R 'synthesize-omnivoice-python-unit'`.
- [ ] **Step 6:** Prove the dumper still gates: run it against one greedy case
  (`--case omni-short-en`) and confirm it verifies pinned inputs then produces
  byte-identical artifacts to the existing local dump (compare sha256 of
  `codes/grid.i32` before/after).
- [ ] **Step 7:** Commit (footer; format check final).

### Task 3: The sampler — Gumbel host logic, class branch, choose_token contract

**Files:**
- Modify: `src/arch/omnivoice/generator-host.h`, `generator-host.cpp`
- Test: `tests/omnivoice_sampler_test.cpp` (new), `tests/CMakeLists.txt`
- Modify: `tests/omnivoice_generator_host_test.cpp` (choose_token assert case)

**Interfaces:**
- Consumes: `synth::NormalRandomStream` (`src/random-stream.h`) —
  `next_uniform()` returns float64-in-float [0,1) from the seeded stream; one
  stream per synthesis, constructed by the caller (Task 4).
- Produces (all in namespace `synth::omnivoice`):
  ```cpp
  // generator-host.h
  // Upstream's _gumbel_sample (omnivoice.py:1632-1636), one value:
  //   scaled = logit / temperature; g = -log(-log(u + 1e-10) + 1e-10)
  //   result = scaled + g
  // float32 throughout; u comes from this port's own stream (upstream has no
  // seed parameter; the seed contract is ours).
  float gumbel_perturb(float logit, float temperature, float uniform);

  // Class-branch companion to choose_token. Applies upstream's top-k filter
  // (keep = ceil(0.1 * vocab_size_without_mask ... see step notes) then a
  // Gumbel-perturbed argmax at class_temperature. Draws exactly one uniform
  // per surviving class from `stream`, in ascending class-id order.
  // Preconditions identical to choose_token; asserts uncond!=nullptr when
  // guidance_scale != 0 (carryover item 1 — now enforced for BOTH paths).
  synth_status_t choose_token_sampled(const float * cond, const float * uncond,
                                      uint32_t vocab_size, uint32_t mask_id,
                                      float guidance_scale, float class_temperature,
                                      NormalRandomStream & stream,
                                      int32_t & token, float & log_prob);
  ```
  plus the existing `choose_token` gaining the assert (see Step 3).
- Downstream (Task 4) applies position noise by perturbing each
  `MaskedCandidate::score` with `gumbel_perturb(score, position_temperature, u)`
  before `select_commits`; that needs no new generator-host API.

**Semantics pinned here (the plan's contract, transcribed from upstream):**
- Upstream `_filter_top_k` keeps `ceil(0.1 * 1025) = 103` classes of the
  guided log-probs **including the mask class position** (the mask was already
  forced to −inf before the filter, so it never survives). Port equivalent:
  operate on the guided array after the mask ban; keep the 103 largest values,
  set the rest to −inf.
- Class draw: `argmax(guided[c]/T + g_c)` over surviving classes, `g_c` from
  one uniform per surviving class, consumed in ascending class-id order (a
  port-defined order — upstream draws a dense rand_like; our contract is our
  own, documented).
- `class_temperature == 0.0` short-circuits to the existing greedy
  `choose_token` (no top-k, no draws) — callers select the function, matching
  upstream's `if class_temperature > 0.0` split.

- [ ] **Step 1: Failing unit tests first.** Create
  `tests/omnivoice_sampler_test.cpp` with reference vectors:
  - `gumbel_perturb` exactness: for
    `(logit, T, u)` ∈ {(0.0f, 5.0f, 0.5f), (-3.25f, 5.0f, 0.0f), (2.5f, 1.0f, 0.9999999f), (-1e30f, 5.0f, 0.25f)}
    assert bit-equality with values computed by the same float32 expression
    written longhand in the test (document: the test pins the *expression
    shape* — `std::log(-std::log(u + 1e-10f) + 1e-10f)` negated — so a
    refactor to double or a reordering breaks it). Also: u=0 must be finite
    (the 1e-10 guards), u→1 must be finite.
  - Determinism: two `NormalRandomStream(7)` streams driving two
    `choose_token_sampled` calls over the same 8-class fixture produce the
    same token; seed 8 produces a different draw sequence (assert the raw
    uniforms differ, not the token — tokens may coincide).
  - Top-k boundary: a 10-class fixture with `keep = ceil(0.1*10) = 1` — only
    the argmax class can ever be drawn regardless of temperature.
  - Mask ban survives sampling: mask_id never returned across 64 draws at
    class_temperature 5.0 on a fixture where mask holds the largest raw logit.
  - Guidance consistency: with guidance 2.0 and a crafted cond/uncond pair,
    the surviving top-k set equals the top-k of the greedy guided array
    (reuse the guided-combination fixture values from
    `tests/omnivoice_generator_host_test.cpp`'s choose_token cases).
- [ ] **Step 2:** Register the test in `tests/CMakeLists.txt` (pattern:
  existing `omnivoice_generator_host_test` block; labels `unit;omnivoice`).
  Build; run; confirm it FAILS (functions absent).
- [ ] **Step 3:** Implement in `generator-host.{h,cpp}`:
  - `gumbel_perturb` as pinned above (float32; no double).
  - `choose_token_sampled` per the contract; share the guided-combination code
    with `choose_token` by extracting a private
    `build_guided(const float*, const float*, uint32_t, float, std::vector<float>&)`
    helper both call (log_softmax → combine → log_softmax → mask ban), so the
    two paths cannot drift.
  - Carryover item 1 in the SAME extraction: `build_guided` asserts
    `!(guidance_scale != 0.0f && uncond == nullptr)` (SYNTH_ASSERT or the
    project's assert macro — match existing generator-host style; the header
    contract line moves from prose to enforcement).
- [ ] **Step 4:** Run the new test — PASS. Run the full generator-host suite —
  the existing choose_token tests must still pass (the extraction is
  behavior-preserving; any diff is a bug).
- [ ] **Step 5:** Add to `tests/omnivoice_generator_host_test.cpp`: a
  `guidance == 0 && uncond == nullptr` legal-path case (uncovered today), and
  a comment on the choose_token contract block recording that the
  nullptr-with-guidance combination is now asserted rather than documented
  (the project's unit harness has no death-test mechanism; the assert's
  presence is reviewed, not executed).
- [ ] **Step 6:** Unit gate both trees; commit (footer; format check final).

### Task 4: The sampled decode path — seed in, grid out

**Files:**
- Modify: `src/arch/omnivoice/omnivoice.h` (SynthesisRequest fields)
- Modify: `src/arch/omnivoice/model.cpp` (run_synthesis step loop)
- Test: `tests/omnivoice_decode_loop_test.cpp` (extend)
- Modify: `tests/omnivoice_replay_real.cpp` (pin greedy explicitly)

**Interfaces:**
- Produces (family-internal `SynthesisRequest` additions):
  ```cpp
  // omnivoice.h — SynthesisRequest
  uint64_t seed = 0;                    // consumed only when a temperature > 0
  // Temperature overrides. Negative = package default (0.0 is a meaningful
  // value: greedy). The replay runner pins both to 0.0f; the public path
  // leaves both at -1.0f so the package's own defaults govern.
  float position_temperature = -1.0f;
  float class_temperature    = -1.0f;
  ```
- Consumes: Task 3's `gumbel_perturb` / `choose_token_sampled`.
- Downstream: Task 5's public branch sets only `seed` (defaults sample);
  the replay runner (this task) sets both temperatures to 0.0f.

**Loop semantics (the contract):** resolve
`pos_t = request.position_temperature < 0 ? hparams.generation.position_temperature : request.position_temperature`
(same for class). Construct `NormalRandomStream stream(request.seed)` once,
before the step loop, iff `pos_t > 0 || class_t > 0`. Per step, for each
still-masked candidate in the existing scan order (codebook-major — the order
`MaskedCandidate`s are built today): token choice via `choose_token`
(class_t == 0) or `choose_token_sampled` (class_t > 0, drawing from `stream`);
then `score = log_prob - codebook * layer_penalty_factor` as today; then iff
`pos_t > 0`, `score = gumbel_perturb(score, pos_t, stream.next_uniform())` —
one uniform per candidate, consumed in the same candidate order. Then
`select_commits` unchanged. Zero-budget steps consume NO draws (upstream's
`k <= 0 → continue`; our loop already skips selection on zero budget — extend
the skip to cover the draws). Margin machinery: `note_margin` only when both
temperatures are 0 (a Gumbel-perturbed margin is meaningless; `margin_report`
with a positive temperature is refused as INVALID_ARG).

- [ ] **Step 1: Failing tests.** Extend `tests/omnivoice_decode_loop_test.cpp`
  (synthetic package, existing harness):
  - Same seed → identical grid: two `run_synthesis` calls, seed 7,
    position_temperature 5.0f, class_temperature 0.0f → byte-equal
    `output.codes`.
  - Different seed → different grid: seed 7 vs seed 8 → grids differ (this
    synthetic canvas is 2 codebooks × ≤16 frames; if the fixture proves too
    small to differ, widen frames — assert on the drawn-uniform trace instead
    only as a last resort, and say so in a comment).
  - Greedy regression: temperatures 0.0f explicit → grid byte-equal to the
    same request at Plan 2 defaults (the request struct's new fields default
    to sampling! — this case pins that the replay path pins zeros).
  - `margin_report && pos_t > 0` → `SYNTH_ERR_INVALID_ARG`.
  - Sampled run makes the placement assertion window; tighten the decode-loop
    placement assertion from >4N to ~`2 * steps * N` and cover the
    `guidance == 0` loop branch (carryover item 2): add a synthetic-package
    case running with `guidance_scale` metadata 0 variant if the harness
    allows, else a `run_synthesis` on the existing package asserting the
    uncond graph is skipped (probe: placement counter or node count — use
    what the test already measures at >4N).
  - Run: FAIL (fields absent).
- [ ] **Step 2:** Implement per the contract above. The stream draw for
  `choose_token_sampled` and the position perturbation share ONE stream.
- [ ] **Step 3:** `tests/omnivoice_replay_real.cpp`: set
  `request.position_temperature = 0.0f; request.class_temperature = 0.0f;`
  with the comment "the replay contract is greedy; the package's own defaults
  sample" — this preserves every Plan 2 golden result.
- [ ] **Step 4:** Run decode-loop + generator-host + sampler tests → PASS.
- [ ] **Step 5:** Golden regression proof: run the replay golden gate
  (`ctest -R synthesize-omnivoice-replay-golden`, ~7 min) — 17/17 exact grids
  must survive the loop rework untouched.
- [ ] **Step 6:** Unit gate both trees; commit (footer; format check final).

### Task 5: The public synthesis branch — prompt assembly to delivered PCM

**Files:**
- Modify: `src/arch/omnivoice/frontend-host.{h,cpp}` (assemble_prompt)
- Modify: `src/arch/omnivoice/omnivoice.h`, `model.cpp` (public entry:
  `Model::synthesize_text` or extend run_synthesis callers — see contract)
- Modify: `src/synthesize.cpp` (replace the `synthesis.not_implemented` stub)
- Test: `tests/omnivoice_frontend_test.cpp` (assembly), new
  `tests/omnivoice_public_real.c` + CMake target (driver only; validation Task 6)

**Interfaces:**
- Produces (frontend-host):
  ```cpp
  // One call from raw request strings to the row-0 text-region ids.
  // Composes: style_text(denoise, language_tag_or_empty, instruct_or_empty)
  //   + "<|text_start|>" + combine_text(ref_text, text) + "<|text_end|>"
  // tokenized via tokenize_wrapped_text with the family's SpecialTokens.
  // `language_tag`: the core's resolved BCP-47 tag verbatim (en/zh/ja are the
  // ISO codes upstream expects; empty -> literal "None" inside style_text).
  // Returns false only on tokenizer failure (empty text was rejected upstream
  // of here).
  bool assemble_prompt_ids(const TextFrontend & frontend,
                           const SpecialTokens & tokens,
                           bool denoise,
                           const std::string & language_tag,
                           const std::string & instruct,
                           const std::string & ref_text,
                           const std::string & text,
                           std::vector<int32_t> & output);
  ```
- Produces (model.cpp / omnivoice.h): a public-facing family entry
  ```cpp
  struct PublicSynthesisParams {
      std::string           text;             // Linguistic Input, UTF-8
      std::string           language_tag;     // resolved by core; may be empty
      const ClonePrompt *   clone = nullptr;  // Task 14 threads this; null now
      const std::string *   instruct = nullptr; // Task 15; null now
      double                speaking_rate = 1.0;
      uint64_t              seed = 0;
      uint64_t              max_output_frames = 0; // native frames; 0 = package cap
      int32_t               threads = 0;
  };
  synth_status_t Model::synthesize(const PublicSynthesisParams &, SynthesisOutput &);
  ```
  which: assembles ids → estimates target frames
  (`DurationEstimator::estimate_target_frames(text, ref_text, ref_frames, rate)`) →
  clamps to the effective frame limit (request limit if nonzero, else package
  `max_output_frames`; estimate > limit is OUTPUT_LIMIT, mirroring upstream's
  estimator-fixes-canvas semantics — the limit is a cap, not a target) →
  `run_synthesis` (seed passed; temperatures left at −1 = package defaults) →
  `decode_codes` → volume arm (no reference → peak-normalize; clone arms in
  Task 14).
- Consumes in `src/synthesize.cpp`: the qwen3-tts branch (lines ~846-915) as
  the structural template — native-vs-PCM frame-limit conversion via
  `samples_per_frame`, `requested_frame_limit` raw value, sub-frame limit →
  `report_output_limit`, `family_request.seed = actual_seed`,
  `deliver_complete_audio`, resolved_language = request's tag.

- [ ] **Step 1: Failing assembly tests.** In `tests/omnivoice_frontend_test.cpp`:
  - `assemble_prompt_ids` with (denoise=false, "en", "", "", "Hi.") produces
    exactly `style_text ids + text_start + tokenize("Hi.") + text_end` — build
    the expectation from the same helpers the test file already exercises
    individually (composition test, not re-derivation).
  - Clone-shaped call (denoise=true, "en", "", "Some call me nature.", "Hi.")
    contains the denoise marker first and the combined text (space-joined).
  - Empty language and instruct → the "None" literals appear (assert via the
    tokens of `style_text("", "")` equality).
- [ ] **Step 2:** Implement `assemble_prompt_ids`; run → PASS.
- [ ] **Step 3:** Implement `Model::synthesize` (family side). Unit-test at
  family level against the synthetic package (extend
  `tests/omnivoice_decode_loop_test.cpp` or the load_synthetic suite): text in
  → finite PCM out, frame_count divisible relation, OUTPUT_LIMIT on a
  too-long estimate (craft text long enough for the synthetic 16-frame cap),
  empty text → INVALID_ARG.
- [ ] **Step 4:** Replace the stub in `src/synthesize.cpp`: mirror the qwen3
  branch: build `PublicSynthesisParams` from `prepared`, convert the frame
  limit (PCM → native via `samples_per_frame()`), map OUTPUT_LIMIT through
  `report_output_limit`, deliver via `deliver_complete_audio`. Voice profile
  pointer still rejected upstream (Task 14 relaxes) — assert `prepared`
  carries none here.
- [ ] **Step 5:** Write `tests/omnivoice_public_real.c` — transcribe
  `tests/qwen3_tts_public_real.c` (plain C, links `synthesize` only), CLI:
  `<model.gguf> <out.pcm> <language-tag|-> <seed|random> [max-frames] [threads]`
  (no voice-id positional: this family's catalog is empty and the package
  default is unnamed). JSON line out: status, frames, sample_rate,
  actual_seed, resolved_language, timing. Register the executable in
  `tests/CMakeLists.txt` (`synth_register_integration_target`, links
  `synthesize` ONLY — public seam).
- [ ] **Step 6:** Smoke by hand against the real package (model present in
  this workspace): seed 7 twice → identical PCM bytes; seed 8 differs; report
  the three digests in the task report. This is the first real end-to-end
  public synthesis of the family — listen-check NOT required here (Plan 4
  owns the listening pass), but the PCM must be finite and nonzero.
- [ ] **Step 7:** Unit gate both trees; golden gate re-run (assembly touched
  nothing on the replay path, prove it); commit (footer; format check final).

### Task 6: The public validator — seed contract as a registered gate

**Files:**
- Create: `scripts/validate-omnivoice-public.py`
- Modify: `tests/tolerances/omnivoice.json` (add `profiles.F32.stages.public`)
- Modify: `tests/CMakeLists.txt` (register `synthesize-omnivoice-public-request`)
- Test: `tests/python/test_tolerance_coverage.py` must pass unmodified (it
  FORCES the stage entry the moment the script exists — land script + stage
  entry in one commit).

**Interfaces:**
- Consumes: Task 5's `synthesize-omnivoice-public-real` driver.
- Template: `scripts/validate-qwen3-tts-public.py` (hardcoded cases + digest
  relations; the manifest's `relations` block is the declared contract, the
  script enforces it independently).

**Checks (mirroring the manifest's `public_request` relation over the three
sampled-seed cases, text "Sampling follows the seed."):**
1. seed 0, seed 1, seed 42: `actual_seed` echoes the request.
2. seed 0 run twice → identical PCM digest.
3. The three seeds → three pairwise-distinct digests (`artifact_differs`).
4. `random` → concrete nonzero seed reported; replaying it reproduces the digest.
5. resolved_language echoes the request tag; resolved_voice is null/absent
   (empty catalog, unnamed default).
6. A `language -` (none) run succeeds (the "None" slot is trained).
7. Auto-voice-follows-seed note: digests from check 3 double as voice
   divergence evidence — record in the report JSON, no extra run.

- [ ] **Step 1:** Write the script (argparse: `--model --runner --profile F32
  --backend cpu --report`; runs = the checks above; exit 1 on any failure;
  `phase: "public_request"` report JSON).
- [ ] **Step 2:** Tolerance stage entry, qwen3 pattern (behavioural, no
  probes): under `profiles.F32.stages`:
  ```json
  "public": {
    "backend": "CPU",
    "description": "The public C seam driven end to end at package defaults (sampled); gates on relations between runs of this port -- seed echo, same-seed byte identity, cross-seed artifact_differs -- which cannot be expressed as thresholds against the oracle.",
    "checks": 7,
    "all_passed": false,
    "note": "all_passed flips true when the registered gate first passes; sampled seed-contract cases carry no margin screen (screen rationale is oracle-parity knife edges; this stage claims none)."
  }
  ```
  Then run the gate once and flip `all_passed` to its measured value in the
  same task (never commit a claimed-true before the run).
- [ ] **Step 3:** Register CTest `synthesize-omnivoice-public-request`:
  model-guarded only (no golden sentinel — "asserts relations between runs of
  this port"), labels `integration;omnivoice;abi`, TIMEOUT 1800,
  WORKING_DIRECTORY source dir; command mirrors the qwen3 public gate with
  the omnivoice env.
- [ ] **Step 4:** Run: `ctest -R synthesize-omnivoice-public-request` → PASS;
  run `tests/python/test_tolerance_coverage.py` via the python unit target →
  the stage-set equality holds.
- [ ] **Step 5:** Unit gate; commit (footer; format check final).

### Task 7: Adapters (stage 7.5) — CLI + Python binding smoke

**Files:**
- Modify: `tests/CMakeLists.txt` (CLI harness registration for omnivoice)
- Modify: `tests/check-python-api-wheel.cmake` (`synth_family_smoke` for omnivoice)
- Modify: `tests/python/api_wheel_family_smoke.py` (only if a package-default
  arm is missing — Kokoro needed `-DSYNTH_VOICE`; omnivoice must pass with NO
  voice named)
- Modify: `tests/CMakeLists.txt` cleanup test (`synth_add_cleanup_test`)

**Interfaces:**
- Consumes: the public branch (Task 5). qwen3-tts never registered these two
  harnesses (the stage-7.5 omission this design explicitly does not repeat);
  VITS and Kokoro registrations are the working patterns.

- [ ] **Step 1:** Register `synthesize-omnivoice-cli` on the
  `tests/check-cli-real.cmake` harness (model-guarded; seeds 42/42/43; no
  `-DSYNTH_VOICE` — package default is unnamed; text input supported so the
  unsupported-input arm parameterizes accordingly — read the harness's
  parameter list and mirror the VITS block).
- [ ] **Step 2:** Register omnivoice in `synth_family_smoke` (parameters:
  sample rate 24000, samples-per-frame 960, no preset voice, seed 7/8
  reproducibility arms). If the smoke script's no-voice arm asserts a
  *refusal* (Kokoro behavior), add the package-default success arm guarded by
  a parameter — do not weaken Kokoro's.
- [ ] **Step 3:** `synth_add_cleanup_test(omnivoice … "text:Hi." "" "en")` —
  mirror the qwen3 entry's shape with an empty voice.
- [ ] **Step 4:** Run both registered tests against the real package + built
  wheel harness (`ctest -R 'omnivoice-cli|omnivoice.*smoke'` — exact names per
  registration). PASS required.
- [ ] **Step 5:** Commit (footer; format check final).

### Task 8: The Audio Normalizer (ADR 0009) — core module, first consumer pending

**Files:**
- Create: `third_party/libsamplerate/` (vendored 0.2.2: `src/*.c`, `include/`,
  `COPYING`; no autotools files, no examples/tests)
- Create: `src/audio-normalizer.{h,cpp}`
- Modify: root `CMakeLists.txt` (static object lib target `synth-libsamplerate`)
- Modify: `THIRD_PARTY_NOTICES.md` (libsamplerate BSD-2-Clause notice + version + origin)
- Test: `tests/audio_normalizer_test.cpp` (new), `tests/CMakeLists.txt`

**Interfaces:**
- Produces (namespace `synth`, core — family-independent):
  ```cpp
  // src/audio-normalizer.h
  // ADR 0009: fixed order validate -> channel-convert -> resample.
  // libsamplerate 0.2.2, SRC_SINC_BEST_QUALITY. Never pads, crops, or trims.
  struct NormalizedReference {
      std::vector<float> pcm;        // mono/stereo interleaved at target
      uint64_t           frames = 0; // frames at target rate
  };
  // Contract (docs/c-interface.md:486-496):
  //  - input rate outside [8000, 192000] or channels outside [1,2]
  //      -> SYNTH_ERR_UNSUPPORTED_INPUT
  //  - null/empty/NaN/Inf samples, zero-frame input, frame-count overflow
  //      -> SYNTH_ERR_INVALID_ARG
  //  - Reference Frame Equivalent = ceil(input_frames * target_rate / input_rate),
  //    computed in uint64 BEFORE allocation; caller checks clip limits against
  //    it (per-clip / total limits are the caller's, with
  //    SYNTH_ERR_INPUT_TOO_LONG mapped there).
  //  - stereo->mono: (l + r) * 0.5f;  mono->stereo: duplicate.
  //  - matching rate AND channels: borrowed copy (no resampler invocation).
  uint64_t reference_frame_equivalent(uint64_t input_frames,
                                      uint32_t input_rate, uint32_t target_rate);
  synth_status_t normalize_reference(const float * pcm, uint64_t frames,
                                     uint32_t input_rate, uint32_t input_channels,
                                     uint32_t target_rate, uint32_t target_channels,
                                     NormalizedReference & output);
  ```
- Consumes: nothing family-side. Task 14 is the first caller.

- [ ] **Step 1:** Vendor libsamplerate 0.2.2: fetch the release source from a
  local pip wheel or the GitHub release tarball for tag 0.2.2 (network fetch
  of the pinned tarball is fine — record the tarball sha256 in
  `third_party/libsamplerate/VENDORED.md` with tag, origin URL, and the file
  subset taken). Keep only `src/*.c`, `src/*.h`, `include/samplerate.h`,
  `COPYING`. Never edit vendored files; config via compile definitions
  (`ENABLE_SINC_BEST_CONVERTER` etc. — the 0.2.2 CMake defines; transcribe the
  minimal set into our CMake).
- [ ] **Step 2:** CMake: `add_library(synth-libsamplerate OBJECT ...)` with a
  narrow include dir; link into `synthesize`. It must build clean under both
  trees INCLUDING sanitizers; suppress third-party warnings locally
  (`target_compile_options` on that target only), never globally.
- [ ] **Step 3:** `THIRD_PARTY_NOTICES.md`: append the BSD-2-Clause notice
  block (name, version 0.2.2, copyright line from COPYING, origin URL).
- [ ] **Step 4: Failing tests.** `tests/audio_normalizer_test.cpp`:
  - RFE exactness: (24000 in, 24000 target, N frames) → N;
    (44100 → 24000, 44100 frames) → 24000; (8000 → 24000, 1) → 3;
    ceiling case (44100 → 24000, 147 frames) → ceil(147*24000/44100)=80.
  - Matching-format borrow: 24 kHz mono in → byte-identical pcm out, frames
    preserved, and (whitebox note) no resampler artifacts — assert
    bit-equality, which SRC would break.
  - Stereo→mono: two-channel fixture where L=0.5, R=-0.25 constant → all
    outputs 0.125f exactly.
  - Rate conversion sanity: a 1 kHz sine at 48 kHz → 24 kHz, assert output
    frame count == RFE and RMS within 5% of input RMS (SRC_SINC_BEST_QUALITY
    passband; this is a sanity gate, not a parity gate — the PARITY gate for
    clone audio is the oracle-pinned 24 kHz input, which arrives already at
    target rate and takes the borrow path).
  - Error mapping: 7999 Hz → UNSUPPORTED_INPUT; 192001 → UNSUPPORTED_INPUT;
    3 channels → UNSUPPORTED_INPUT; NaN sample → INVALID_ARG; zero frames →
    INVALID_ARG; null pcm → INVALID_ARG.
  - Register `unit` label; run → FAIL.
- [ ] **Step 5:** Implement `normalize_reference` (fixed order; SRC full-buffer
  `src_simple` with SRC_SINC_BEST_QUALITY when rates differ). Run → PASS.
- [ ] **Step 6:** Unit gate both trees (sanitizer build of vendored C matters
  here); commit (footer; format check final).

### Task 9: Oracle — clone-encode probes, suite revision 3

**Files:**
- Modify: `scripts/dump_reference_omnivoice_pytorch.py` (three new ref hooks)
- Modify: `tests/golden/omnivoice/omnivoice-0-6b.manifest.json`
  (suite_version 3; new artifacts on the two clone cases)
- Modify: `tests/tolerances/omnivoice.json` (suite_version 3; note update —
  thresholds for new probes land in Task 13 after measurement)
- Test: `tests/python/test_golden_manifests.py` must pass unmodified

**Interfaces:**
- Produces, per clone case (both `omni-clone-en` and `omni-clone-zh`), three
  new artifacts alongside the existing `ref/{pcm_24k.f32, tokens.i32, semantic_hidden.f32}`:
  - `ref.pcm_16k` (`ref/pcm_16k.f32`): the 16 kHz waveform handed to HuBERT —
    capture INSIDE `_extract_semantic_features` after the 24→16 resample and
    channel select, BEFORE the (160,160) pad. Hook: wrap
    `torchaudio.functional.resample` is fragile; instead hook the HuBERT
    module's pre-forward and strip the known 160-pad from both ends of
    `input_values` (the pad is a fixed constant; assert the padded length =
    16k-length + 320 so the strip is provably correct).
  - `ref.semantic_mean` (`ref/semantic_mean.f32`): the mean over all 13
    HuBERT hidden states, captured BEFORE the `[::2]` downsample — hook the
    HuBERT forward (output_hidden_states path) and compute
    `torch.stack(hidden_states).mean(dim=0)` exactly as
    `_extract_semantic_features` does (transcribe its expression; if the
    upstream call is `.mean(dim=1)` over a stacked dim-1, replicate THAT —
    fidelity to the pinned source, dumper comment cites
    modeling_higgs_audio_v2_tokenizer.py:501-505).
  - `ref.fused_latent` (`ref/fused_latent.f32`): the `self.fc` output — the
    1024-dim per-frame latent the RVQ encodes. Forward hook on
    `audio_tokenizer.fc` (Linear 1024→1024), first call.
  Shapes recorded in each case's metadata.json as today.
- The existing `semantic_module` metadata note is extended: semantic_hidden
  stays "a stage boundary, not the consumed feature"; semantic_mean IS the
  consumed feature pre-downsample; fused_latent is the quantiser input.

- [ ] **Step 1:** Implement the three hooks in the dumper's clone path
  (pattern: the existing `SemanticProbe` class; one probe class per capture,
  first-forward-only, assert-single-call).
- [ ] **Step 2:** Manifest: add the three artifact entries to both clone
  cases; bump `suite_version` 2→3 with a dated provenance comment in the
  manifest's provenance field naming the ruling ("clone-encode probes added
  by jiangzhuo's ruling of 2026-08-01; no case text changed; all grids and
  waveforms unchanged"). Bump tolerances `suite_version` to 3 in the same
  commit (its probe grid is untouched until Task 13's measurement; note says
  so).
- [ ] **Step 3:** Re-dump BOTH clone cases:
  `uv run --project scripts/envs/omnivoice --locked python scripts/dump_reference_omnivoice_pytorch.py --manifest tests/golden/omnivoice/omnivoice-0-6b.manifest.json --weights-dir models/omnivoice-0-6b --report /tmp/omnivoice_plan3_t9.json --case omni-clone-en --case omni-clone-zh`
- [ ] **Step 4:** Byte-identity proof for pre-existing artifacts: sha256 of
  all 16 prior artifacts per case, before vs after — identical (greedy is
  bit-reproducible; the new hooks must not perturb anything). Record the
  digest table in the task report. Any drift = STOP, investigate (the rule:
  a probe that changes the observed system is a broken probe).
- [ ] **Step 5:** Run `tests/python/test_golden_manifests.py` (schema,
  artifact URLs, case_count 20 unchanged) and the manifest structural suite →
  PASS. Run one greedy case dump (`--case omni-short-en`) → byte-identical
  (the shared dumper path is untouched for non-clone cases).
- [ ] **Step 6:** Commit (manifest + tolerances + dumper; footer; format
  check final — no C++ touched, the check is trivially clean).

### Task 10: The internal 24→16 kHz resampler — torchaudio transcription

**Files:**
- Create: `src/arch/omnivoice/reference-encoder-host.{h,cpp}` (started here:
  resampler only)
- Test: `tests/omnivoice_resampler_test.cpp` (new), `tests/CMakeLists.txt`

**Interfaces:**
- Produces:
  ```cpp
  // reference-encoder-host.h  (namespace synth::omnivoice)
  // torchaudio.functional.resample(24000 -> 16000) transcribed at the pinned
  // defaults: sinc_interp_hann, lowpass_filter_width=6, rolloff=0.99.
  // gcd(24000,16000)=8000 -> orig=3, new=2: a 2-phase polyphase over stride-3
  // input windows. float32 arithmetic; kernel built in float64 then cast once
  // (torchaudio builds the kernel in the input dtype via arange/clamp in
  // float32 — TRANSCRIBE ITS DTYPE CHOICES EXACTLY; the implementer reads
  // torchaudio/functional/functional.py:_get_sinc_resample_kernel at the
  // pinned 2.11.0 and records in a comment which steps are f32 vs f64).
  // Output length: ceil(input_len * 2 / 3) (torchaudio's target_length).
  bool resample_24k_to_16k(const std::vector<float> & input,
                           std::vector<float> & output);
  ```
- Parity target: `ref/pcm_16k.f32` from Task 9 (the pinned reference wav's
  16 kHz form). Tolerance: measured max_abs, committed in Task 13; the
  interim gate in THIS task's unit test is agreement with the oracle dump to
  max_abs ≤ 1e-6 relative to a ±1 signal (if measurement lands worse, STOP
  and compare kernel dtype choices before accepting anything looser — the
  chain downstream ends in a discrete NN decision).

- [ ] **Step 1:** Read torchaudio 2.11.0's `_get_sinc_resample_kernel` +
  `_apply_sinc_resample_kernel` in the locked venv; transcribe constants:
  width = 6 * 3 (lowpass_filter_width * orig per phase math), cutoff
  0.99 * 0.5 * 2/3-band — record the exact expressions in the header comment.
- [ ] **Step 2: Failing test** with three fixtures: (a) DC 1.0 input of 96
  samples → output ~0.99-cutoff DC (assert against oracle-computed vector —
  generate a small reference table by running torchaudio in the locked env
  and committing the expected floats as constants in the test, provenance
  comment with the generating one-liner); (b) impulse response — kernel taps
  visible, compare to committed torchaudio output; (c) length arithmetic:
  input 960 → output 640, input 961 → 641 (ceil).
- [ ] **Step 3:** Implement; run → PASS at the committed vectors.
- [ ] **Step 4:** Real-signal parity: a test-local runner (guarded by the
  model/goldens sentinel like other real tests — integration label) loads
  `ref/pcm_24k.f32` + `ref/pcm_16k.f32` from `build/goldens/omnivoice/omni-clone-en/`
  and asserts max_abs ≤ the interim 1e-6 gate. Register under
  `integration;omnivoice` (sentinel-guarded).
- [ ] **Step 5:** Unit gate both trees; commit (footer; format check final).

### Task 11: HuBERT semantic branch + SemanticEncoder — graph to semantic_mean parity

**Files:**
- Create: `src/arch/omnivoice/reference-encoder.{h,cpp}`
- Modify: `src/arch/omnivoice/reference-encoder-host.{h,cpp}` (orchestration)
- Test: `tests/omnivoice_reference_encoder_test.cpp` (new; synthetic-scale unit)
- Modify: `tests/omnivoice_replay_real.cpp` + `scripts/validate-omnivoice-replay.py`
  (encode channel, first probes)

**Interfaces:**
- Produces (graph half, `reference-encoder.h`):
  ```cpp
  // Builds the semantic branch on a 16 kHz mono input tensor [samples]:
  //   pad(160,160) -> feature extractor (7 conv layers: layer 0 stride 5
  //   kernel 10 + GroupNorm(512), layers 1-4 stride 2 kernel 3, layers 5-6
  //   stride 2 kernel 2; GELU) -> LayerNorm -> feature_projection ->
  //   + pos_conv (grouped conv k=128 g=16, weight-norm folded at convert
  //   time) -> 12 standard bidirectional attention layers (post-LN variant:
  //   do_stable_layer_norm=False -> attn, +res, LN, ffn, +res, LN) ->
  //   MEAN over the 13 hidden states (embedding output + each layer output,
  //   captured as graph nodes and averaged) -> [::2] stride-2 downsample ->
  //   SemanticEncoder conv stack (encoder_semantic.* tensors, 13 tensors).
  // Tensor names: the catalog's codec.semantic_model.* / codec.encoder_semantic.*
  // set (catalog.cpp:389-462) — the builder consumes SemanticModelWeights
  // resolved by the existing catalog.
  // Attention: full bidirectional, no cache, no rope (HuBERT uses learned
  // conv positional embedding, not rotary. NOTE: NOT the qwen3 layer).
  // LayerDrop: none — upstream draws RNG in eval but never skips (the port
  // neither draws nor skips; recorded divergence-of-no-consequence).
  ggml_tensor * build_semantic_branch(ggml_context *, const SemanticWeights &,
                                      ggml_tensor * pcm_16k, /*out*/ ...);
  ```
  (The task defines the exact weight-struct fields from catalog.cpp's tensor
  list; the builder mirrors the file-pair convention: graph in
  reference-encoder.cpp, orchestration/graph-run in the -host file.)
- Scale rule: unit tests run a SYNTHETIC miniature (2 layers, hidden 8,
  conv stack shrunk) via a small-layout header extension
  (`tests/omnivoice_small_layout.h` gains the semantic tensors), pinned
  against constants dumped by extending `scripts/dump_reference_omnivoice_generator.py`'s
  synthetic-fixture pattern — a new
  `scripts/dump_reference_omnivoice_semantic.py` (LcgStream-filled miniature
  HuBERT through the same transformers classes, printing C++ fixture
  constants; the Task-9-hardened parameter-coverage assert pattern is copied
  here from day one).
- Real-scale parity: the replay runner grows `--encode-reference <pcm_24k.f32>`
  which (when given) runs resample → semantic branch and writes
  `semantic_mean.f32`; the validator's clone cases gain an encode-probe
  comparison (max_abs + cosine, thresholds Task 13) against
  `ref/semantic_mean.f32`.

- [ ] **Step 1:** Write `scripts/dump_reference_omnivoice_semantic.py`
  (miniature; parameter-coverage assert; prints fixture arrays + expected
  semantic_mean). Run it; paste constants into
  `tests/omnivoice_reference_encoder_test.cpp` with provenance comments.
- [ ] **Step 2:** Failing unit test: miniature semantic branch forward vs
  fixture expectation (max_abs ≤ 1e-5 at miniature scale — measured, then
  tightened to observed×5).
- [ ] **Step 3:** Implement the builders. Conv feature extractor via
  `ggml_conv_1d` family ops already used by the codec (im2col+matmul pattern,
  codec.cpp precedent); GroupNorm/LayerNorm from stock GGML ops; attention
  transcribed from the omnivoice generator layer MINUS rope/q-k-norm plus
  post-LN ordering (cite generator.cpp's layer as the structural donor and
  enumerate the deltas in comments).
- [ ] **Step 4:** Unit PASS at miniature scale, both trees.
- [ ] **Step 5:** Real-scale: extend the replay runner with
  `--encode-reference`; run against `omni-clone-en`'s `ref/pcm_24k.f32`;
  compare to `ref/semantic_mean.f32`; record observed max_abs/cosine in the
  task report (expect f32-noise-floor agreement; tolerances committed in
  Task 13).
- [ ] **Step 6:** Unit gate; golden gate re-run (runner changed: prove the
  replay channels untouched); commit (footer; format check final).

### Task 12: DAC acoustic encoder + fusion — fused_latent parity

**Files:**
- Modify: `src/arch/omnivoice/reference-encoder.{h,cpp}` (acoustic branch + fusion)
- Modify: `scripts/dump_reference_omnivoice_codec.py` (encoder-side miniature fixture)
- Test: `tests/omnivoice_reference_encoder_test.cpp` (extend)

**Interfaces:**
- Produces: `build_acoustic_encoder` (24 kHz pcm → [256, T] via the DAC
  encoder mirror: conv → N blocks of (residual units ×3 + strided conv),
  Snake activations, ratios {8,5,4,2,3} DOWNsampling this direction — the
  catalog's `codec.acoustic_encoder.*` 110 tensors) and `build_reference_fusion`
  (concat [acoustic 256 ; semantic 768] → 1024, then `codec.fc` Linear →
  [1024, T]). Conditional 480-per-side pad: upstream pads the acoustic input
  only when conv output length would mismatch the semantic length — for
  hop-aligned inputs (ours always are: hop clip precedes encode) determine
  once whether the branch fires by measuring the pinned reference through
  the oracle (Task 9's dumps make the answer visible: acoustic length vs
  semantic length); transcribe the resolved constant behavior with a comment,
  and refuse (nullptr) on any input where the lengths would disagree — that
  is a wiring defect, not a runtime case.
- Consumes: Task 11's semantic output at 25 Hz.

- [ ] **Step 1:** Extend the codec dumper with an `--encoder` miniature mode
  (same LcgStream discipline, parameter-coverage assert): miniature DAC
  encoder + fc; print fixture constants + expected fused latent for a fixed
  code-free input.
- [ ] **Step 2:** Failing unit: miniature acoustic branch + fusion vs fixture.
- [ ] **Step 3:** Implement (Snake/conv builders REUSED from codec.cpp —
  export the shared helpers through codec.h rather than duplicating; the
  strided conv is `codec_conv1d` with stride support added if absent —
  check codec.cpp: decode used transpose convs; encode needs strided forward
  convs — extend `codec_conv1d` with a stride parameter defaulting 1, all
  existing callers unchanged, rejection tests extended for stride<=0).
- [ ] **Step 4:** Unit PASS both trees.
- [ ] **Step 5:** Real-scale: extend `--encode-reference` to emit
  `fused_latent.f32`; compare vs `ref/fused_latent.f32` (omni-clone-en);
  record observed figures for Task 13.
- [ ] **Step 6:** Unit gate; commit (footer; format check final).

### Task 13: RVQ encode + exact reference-token parity + tolerance commit

**Files:**
- Modify: `src/arch/omnivoice/reference-encoder-host.{h,cpp}` (RVQ encode, host)
- Modify: `tests/omnivoice_replay_real.cpp`, `scripts/validate-omnivoice-replay.py`
  (tokens-exact channel; encode probes enforced)
- Modify: `tests/tolerances/omnivoice.json` (measured thresholds for
  pcm_16k/semantic_mean/fused_latent probe comparisons + encode note)
- Test: `tests/omnivoice_reference_encoder_test.cpp` (extend: NN fixtures)

**Interfaces:**
- Produces:
  ```cpp
  // Host-side residual vector quantization, encode direction. DISCRETE
  // DECISION -> CPU host code by the placement rule; the projected inputs
  // (project_in outputs) are computed on host from the graph's fused latent
  // (pulled to CPU) so the whole decision path is CPU-deterministic.
  // Per level q in 0..7:
  //   z_q = project_in_q(residual)            // 1024 -> 64, host matmul
  //   code = argmin_e ||z_q - E_q[e]||^2      // plain L2, ties -> lowest id
  //   dequant = project_out_q(E_q[code])      // 64 -> 1024
  //   residual -= dequant
  // Margin instrumentation: per (level, frame), gap between best and
  // second-best squared distance; the narrowest gap over the clip is
  // reported (diagnostic only — the gate is exact tokens).
  bool rvq_encode(const RvqEncodeWeights &, const std::vector<float> & latent,
                  uint64_t frames, std::vector<int32_t> & tokens,
                  float * narrowest_gap = nullptr);
  ```
- Tie rule: upstream `argmax(-dist)` takes the FIRST maximal index — lowest
  code id. Transcribe as `<` strict comparison scanning ascending.
- Validator: clone cases under `--require all` now run the encode channel
  when `--encode-reference` artifacts exist: probes (pcm_16k, semantic_mean,
  fused_latent) under committed thresholds, then `tokens.i32` EXACT — a
  mismatched token is a failure with the (level, frame, got, want, gap)
  quintuple printed. `VOLUME_BY_BRANCH` untouched here (Task 14).

- [ ] **Step 1:** Unit fixtures: extend the codec dumper miniature to print
  RVQ-encode expected codes for the miniature latent (8 levels shrunk to 2,
  codebook 4×small-dim); include a crafted TIE fixture (two codebook rows
  equidistant — expected: lower id) and a near-tie margin fixture asserting
  `narrowest_gap` reporting.
- [ ] **Step 2:** Failing unit tests; implement; PASS.
- [ ] **Step 3:** Wire the full encode chain in reference-encoder-host:
  `encode_reference(pcm_24k) -> {tokens, ref_rms, frames}`: hop clip (tail,
  multiple of 960) → ref_rms (Neumaier-summed mean of squares, sqrt) →
  quiet boost (`0 < rms < 0.1` → scale to 0.1, STORED rms stays pre-boost) →
  resample 16k → graphs → rvq_encode.
- [ ] **Step 4:** Real parity: replay runner `--encode-reference` now also
  emits `tokens.i32`; run omni-clone-en AND omni-clone-zh (same wav — the
  encode is per-reference; run once, assert both cases' pinned `ref/tokens.i32`
  are identical first, then compare): **8×351 tokens EXACT**. Any mismatch:
  STOP; diagnose via the probe chain (that is what it is for); a genuine
  arithmetic knife-edge in NN distance follows the dual-admissibility
  process — enumerated torch-produced witness, jiangzhuo informed — never a
  tolerance.
- [ ] **Step 5:** Measure and commit tolerances: observed max_abs/cosine for
  the three encode probes (both clone cases), thresholds at ≤5× observed,
  `tests/tolerances/omnivoice.json` gains them under `stages.replay.probes`
  (they ride the replay stage — same validator, same phase); tolerance note
  updated; `--check` run green.
- [ ] **Step 6:** Full golden gate (`ctest -R synthesize-omnivoice-replay-golden`)
  → 20/20 with the encode channel active on the clone cases.
- [ ] **Step 7:** Unit gate both trees; commit (footer; format check final).

### Task 14: Reference Audio profiles end-to-end — the public cloning path

**Files:**
- Create: `src/arch/omnivoice/profile.{h,cpp}` (ClonePrompt payload)
- Modify: `src/voice-profile.cpp` (family dispatch), `src/model-info.h`
  (capabilities), `src/synthesize.cpp` (capabilities flow + synthesis-with-profile),
  `src/synthesis-request.{h,cpp}` (thread the profile), `src/arch/omnivoice/model.cpp`
  (volume arms; synthesize() clone params)
- Modify: `scripts/validate-omnivoice-replay.py` (`VOLUME_BY_BRANCH` +
  quiet-reference arm), `scripts/validate-omnivoice-public.py` (clone runs)
- Test: `tests/voice_profile_api_test.c` (matrix update),
  `tests/omnivoice_profile_test.cpp` (new), `tests/omnivoice_codec_test.cpp`
  (volume arms unit)

**Interfaces:**
- Produces (family payload):
  ```cpp
  // profile.h — the prepared clone prompt, the unit Serialized Profiles
  // round-trip (Task 16).
  struct ClonePrompt {
      std::vector<int32_t> reference_tokens;   // 8 x T_ref, codebook-major
      std::vector<int32_t> transcript_ids;     // tokenized ref_text AFTER the
                                               // punctuation rule (below)
      std::string          transcript_text;    // canonical post-punctuation text
                                               // (serialized: it is conditioning,
                                               // not source material — the prompt
                                               // template consumes it combined
                                               // with the target text at request
                                               // time, so the TEXT is canonical,
                                               // the ids are a cache)
      float                ref_rms = 0.0f;     // pre-boost, drives volume arms
      std::string          language_tag;       // optional, may be empty
  };
  ```
- Punctuation rule (upstream `add_punctuation`): if the stripped transcript's
  last character is not in the upstream END_PUNCTUATION set, append `。` when
  the transcript contains any CJK ideograph else `.`. The set and the CJK
  test are transcribed from `omnivoice/utils/text.py:38-61,213-225` into
  frontend-host with a provenance comment.
- Capabilities plumbing (core): `synth::ModelInfo` gains
  ```cpp
  struct VoiceProfileInfo {
      uint32_t source_flags = 0;             // bitmask of SYNTH_PROFILE_SOURCE_*
      synth_requirement_t reference_transcript = SYNTH_REQUIREMENT_UNSUPPORTED;
      synth_requirement_t reference_language   = SYNTH_REQUIREMENT_UNSUPPORTED;
      synth_requirement_t description_language = SYNTH_REQUIREMENT_UNSUPPORTED;
      uint32_t reference_target_sample_rate = 0, reference_target_channels = 0;
      uint64_t min_frames_per_clip = 0, max_frames_per_clip = 0,
               max_total_frames = 0;
      uint32_t max_reference_count = 0;
      uint8_t  compatibility_id[32] = {};     // decoded from the package hex
  } voice_profile;
  ```
  filled by the omnivoice `shared_info` from `ProfileContract` (hex→bytes
  decode helper; the package stores 64 lowercase hex chars);
  `synth_model_get_voice_profile_capabilities` copies it out. OmniVoice
  claims REFERENCE_AUDIO | DESCRIPTION_TEXT | SERIALIZED_PROFILE (serialized
  arrives Task 16 but the flag is claimed with it — set the two flags here,
  add SERIALIZED there); transcript REQUIRED, reference_language OPTIONAL,
  description_language OPTIONAL.
- `synth_voice_profile` grows a family payload variant
  (`std::variant<std::monostate, omnivoice::ClonePrompt, omnivoice::DesignInstruct>`
  or an opaque family blob — match the codebase's core/family firewall: core
  holds a type-erased `std::shared_ptr<void>` + family tag; the family
  interprets. The family firewall rule "family internals never cross the
  public seam" governs; pick the type-erased pointer).
- `create_from_reference` (core): struct_size/null checks (existing) → model
  family dispatch → omnivoice handler: exactly 1 clip (max_reference_count 1;
  more → INVALID_ARG naming the limit), transcript required (missing →
  INVALID_ARG named diagnostic), Audio Normalizer to 24 kHz mono (RFE
  pre-check against min/max frame limits → INPUT_TOO_LONG / INVALID_ARG) →
  family `encode_reference` → **ref_rms == 0.0 → SYNTH_ERR_INVALID_ARG,
  diagnostic `voice_profile.reference_silent` ("the reference audio is
  digitally silent; nothing can be cloned from it") — jiangzhuo's ruling
  2026-08-01, deliberate divergence from upstream's silent output** →
  ClonePrompt stored.
- Synthesis with profile: `prepare_synthesis_request` relaxation — profile +
  voice_id → INVALID_ARG (exists), profile from another model →
  UNSUPPORTED_VOICE, else thread `const synth_voice_profile_t *` through
  `PreparedSynthesisRequest`; the omnivoice branch passes ClonePrompt to
  `Model::synthesize` (params.clone): prompt assembly uses denoise=true,
  combined transcript+text, reference_tokens appended, duration estimator
  gets (text, transcript_text, T_ref, rate); volume arms after decode:
  `rms >= 0.1` → none; `0 < rms < 0.1` → `* rms / 0.1` (the quiet arm).
- Volume arms unit-tested at codec-host level:
  `apply_reference_volume(audio, ref_rms)` joins the existing
  `apply_no_reference_volume`; quiet-arm fixture: peak-1.0 signal, rms 0.05 →
  every sample × 0.5; boundary rms exactly 0.1 → untouched.

- [ ] **Step 1:** Volume arms first (smallest): failing codec-host unit tests
  (fixtures above incl. the 0.1 boundary and a mutation note mirroring the
  Plan 2 peak-normalize fixture); implement `apply_reference_volume`; PASS.
- [ ] **Step 2:** `VOLUME_BY_BRANCH` in the replay validator gains
  `scale_by_ref_rms_over_0.1`; a unit-level quiet-reference case lands in
  `tests/omnivoice_codec_test.cpp` (carryover item 7 closes: golden coverage
  is impossible — the pinned reference measures 0.123 — so the unit fixture
  IS the coverage, plus the disposition below).
- [ ] **Step 3:** Punctuation rule in frontend-host + unit tests (en no-punct
  → ".", zh → "。", already-terminated (each END_PUNCTUATION member) →
  unchanged, empty transcript → unchanged-empty is upstream's behavior —
  verify against text.py and pin what it does).
- [ ] **Step 4:** Core plumbing: ModelInfo.voice_profile + capabilities
  getter + hex→bytes decode (unit: `tests/voice_profile_api_test.c` matrix
  updates — a profile-capable model now returns real capabilities; VITS et al
  still zero + UNSUPPORTED_VOICE; keep both arms).
- [ ] **Step 5:** `create_from_reference` omnivoice handler, with
  `tests/omnivoice_profile_test.cpp` (real-package integration test,
  model-guarded): create from the PINNED reference wav (load
  `models/omnivoice-reference-audio/…` bytes; feed as PCM via a minimal WAV
  reader in the test or reuse the harness loader if one exists — check
  `tests/` for a wav helper first) + transcript → profile; assert token grid
  equals `ref/tokens.i32` (goldens sentinel-guarded), ref_rms ≈ 0.1229
  (|Δ| < 1e-4), silent-reference rejection (zero buffer → INVALID_ARG +
  diagnostic), two clips → INVALID_ARG, missing transcript → INVALID_ARG.
- [ ] **Step 6:** Request threading + synthesis-with-profile; extend
  `validate-omnivoice-public.py`: clone run (pinned wav + transcript via a
  small python-side WAV decode → the public driver grows
  `--reference <pcm.f32> --transcript <txt>` arguments), assert: synthesis
  succeeds, same-seed reproducibility holds WITH a profile, output differs
  from the no-profile run at the same seed (the reference conditions), and
  the greedy clone free-run grid claim stays where it lives (replay gate) —
  public phase claims relations only.
- [ ] **Step 7:** Family doc: record the silent-reference rejection ruling +
  the quiet-arm unit coverage disposition + upstream-divergence note (no
  pydub silence preprocessing: callers hand clean 1–20 s clips; the golden
  pins preprocess_prompt=false).
- [ ] **Step 8:** Unit gate both trees; public gate + replay gate; commit
  (footer; format check final).

### Task 15: Description Text profiles — the instruct vocabulary

**Files:**
- Modify: `src/arch/omnivoice/profile.{h,cpp}` (DesignInstruct payload +
  vocabulary), `src/voice-profile.cpp` (from_description dispatch),
  `src/arch/omnivoice/model.cpp` (instruct into synthesize())
- Test: `tests/omnivoice_profile_test.cpp` (extend)

**Interfaces:**
- Produces: `struct DesignInstruct { std::string instruct; }` — the validated,
  canonical (EN- or ZH-unified, `", "`/`"，"`-joined) instruct string.
- Vocabulary transcription from `omnivoice/utils/voice_design.py:31-97` at
  the pinned revision: the closed attribute sets (gender, age, pitch,
  whisper flag, EN accents, ZH dialects) as static tables with a provenance
  comment; rules: comma-split (both widths), per-item membership, per-category
  exclusivity, dialect+accent mix rejected, language unification (ZH iff any
  ZH-vocabulary item or CJK in the description; else EN — TRANSCRIBE the
  exact upstream condition from OMNI:1068 and voice_design.py, including
  what upstream does when the description mixes scripts), free-form items:
  upstream `_resolve_instruct` accepts only vocabulary items — a non-member
  raises. The port rejects with INVALID_ARG and a diagnostic naming the bad
  item (no difflib suggestions — a diagnostic string, not a search engine).
  NOTE the design spec says "attribute / free-form instruct vocabulary":
  verify at transcription time whether upstream accepts free text alongside
  attributes (OMNI:1492-1621); port EXACTLY what the pinned source accepts —
  the two design golden cases ("female, young adult, high pitch") prove the
  attribute path either way.
- `description_language` param: names the description's own language;
  omnivoice accepts en/zh (upstream's trained set), null → package default =
  "en" unless the description contains CJK (mirror the unification rule);
  the preparation `seed` field is accepted and unused (deterministic
  preparation; recorded in the header comment).
- Synthesis: instruct flows into `assemble_prompt_ids`'s instruct slot;
  denoise stays false; volume = no-reference arm.

- [ ] **Step 1:** Failing unit tests (vocabulary): each category accepted
  single; two of one category → INVALID_ARG; dialect+accent → INVALID_ARG;
  non-member ("sings well" if upstream rejects it — align with the
  transcription finding) → INVALID_ARG + diagnostic; zh dialect item forces
  ZH join "，"; the exact golden instruct "female, young adult, high pitch"
  canonicalizes to itself.
- [ ] **Step 2:** Transcribe + implement; PASS.
- [ ] **Step 3:** `create_from_description` dispatch + integration arm in
  `tests/omnivoice_profile_test.cpp`: golden-instruct profile → synthesize →
  finite PCM; and `validate-omnivoice-public.py` gains one design run
  (differs from same-seed no-profile run).
- [ ] **Step 4:** Unit gate; public gate; commit (footer; format check final).

### Task 16: Serialized Profiles — GGUF round-trip

**Files:**
- Modify: `src/arch/omnivoice/profile.{h,cpp}` (writer/reader),
  `src/voice-profile.cpp` (serialize / load_from_memory dispatch)
- Test: `tests/omnivoice_profile_test.cpp` (round-trip + tamper matrix)

**Interfaces:**
- Wire format (docs/c-interface.md:654-674, ADR 0008): little-endian GGUF v3,
  alignment 32, `general.architecture = "synthprofile"`,
  `synthesize.voice_profile.{format_version=1, model_family="omnivoice",
  schema="omnivoice-clone-prompt", schema_version=1,
  compatibility_id uint8[32], content_sha256 uint8[32]}`; content_sha256
  computed over the whole file with the hash field zeroed. Payload for a
  ClonePrompt: tensors `profile.reference_tokens` (i32 [T_ref, 8] — pick the
  GGUF-natural layout and document), metadata
  `synthesize.voice_profile.{transcript_text (string), ref_rms (f32),
  language_tag (string, may be empty)}`. transcript_ids are NOT serialized
  (a cache; re-tokenized on load — determinism guaranteed by the frozen BPE).
  DesignInstruct: schema is the SAME family schema with
  `synthesize.voice_profile.kind = "clone-prompt" | "design-instruct"` and
  the instruct string in metadata, no tensors — ONE schema, two kinds, so
  the package's single declared schema string stays truthful (record this
  choice in the family doc; the alternative — a second schema id — would
  need a package re-cut to declare).
- Loader hardening: bytes are untrusted — structural validation (magic,
  version, alignment), size arithmetic before allocation (the qwen3-3TB
  lesson applies to profiles too: token-count bounds from
  `max_total_frames * 8`), digest verify, schema + exact compatibility_id
  match → else UNSUPPORTED_VOICE; malformed → INVALID_ARG.
- Serialize uses ggml's `gguf_*` writer API into a memory buffer
  (`ByteBufferStorage` in voice-profile.cpp exists for ownership).

- [ ] **Step 1:** Failing round-trip tests: clone profile → serialize →
  load_from_memory → synthesize twice at one seed from original and loaded →
  byte-identical PCM (integration, model-guarded). Design profile round-trip
  likewise. Byte-level: serialize twice → identical bytes (deterministic
  writer; content_sha256 well-defined).
- [ ] **Step 2:** Tamper matrix (unit-scale, synthetic ClonePrompt): flip one
  payload byte → INVALID_ARG (digest); wrong compatibility_id →
  UNSUPPORTED_VOICE; wrong schema → UNSUPPORTED_VOICE; truncated buffer →
  INVALID_ARG; token id out of [0,1025) → INVALID_ARG; token count beyond
  max_total_frames*8 → INVALID_ARG (pre-allocation).
- [ ] **Step 3:** Implement writer/reader + dispatch; SERIALIZED_PROFILE
  capability flag set. PASS all arms.
- [ ] **Step 4:** Unit gate both trees; commit (footer; format check final).

### Task 17: Close-out — docs, margins, ledger, final gates

**Files:**
- Modify: `docs/porting/families/omnivoice.md` (Plan 3 record: sampler
  semantics + our draw-order contract; sampled-cases-no-screen rationale;
  silent-reference rejection; probe additions + suite rev 3; serialized
  schema kinds; adapters registered; divergences ledger: no pydub
  preprocessing, no fade/pad in v1 public output, LayerDrop non-consumption)
- Modify: `reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`
  (Plan 3 evidence: encode parity figures, tolerance derivations, public
  gate first-pass, digests)
- Create: `docs/superpowers/plans/2026-08-XX-omnivoice-plan-4-carryover.md`
  (dated at writing; consumes the Plan 3 ledger's deferred minors + the
  standing Plan 4 items: primary-grid digest pinning, converter harness
  items, quantization margin protocol, float64 cosine, listening pass)
- Modify: `docs/testing.md` (public gate + encode channel rows; Status line
  date bump)

**Interfaces:** none; this task turns the branch's evidence into confirmed
records and runs every gate one final time.

- [ ] **Step 1:** Family doc updates (each section carries its dated ruling
  attribution; CONTEXT.md terms throughout).
- [ ] **Step 2:** Porting log: one dated Plan 3 section — sampler contract,
  encode-parity table (probes observed vs committed, tokens exact 2/2),
  public-gate checks 7/7, adapter registrations, divergence ledger.
- [ ] **Step 3:** Plan 4 carryover ledger from the SDD ledger's deferred
  minors + standing items; every entry names its file.
- [ ] **Step 4:** docs/testing.md rows + `Status: Confirmed` date bumps on
  every doc touched this plan.
- [ ] **Step 5:** Final gates, in order, all green, exit codes reported:
  unit (both trees), `synthesize-omnivoice-replay-golden` (20/20 with encode
  channel), `synthesize-omnivoice-public-request`, the two adapter tests,
  python unit targets.
- [ ] **Step 6:** Commit (footer; format check final).

---

## Self-Review Notes (writing-plans checklist)

- **Spec coverage:** slice 7 → Tasks 3-6; slice 7.5 → Task 7; slice 8 →
  Tasks 8-16; slice 9's sampled/clone extension → Tasks 6, 13; carryover
  first-touch 1-5 → Tasks 1-4; carryover content 6-8 → Tasks 9/14/6 (ruled);
  ship-blocking items stay in Plan 4 (quants, backends, cards, listening pass).
- **Non-goals honored:** no Whisper, no pydub text/audio preprocessing, no
  long-form chunking, no streaming claims, no public ABI additions (Task 5
  verifies rather than adds; the profile API pre-exists in the header).
- **Type consistency spot-checks:** `NormalRandomStream` name per
  random-stream.h; `SYNTH_ERR_UNSUPPORTED_INPUT/INPUT_TOO_LONG` exist in the
  public header (used by kokoro reference path? verify at Task 8 — if the
  status codes differ in name, the c-interface.md names govern);
  `synth_requirement_t` values per include/synthesize.h:184-187.
- **Known open risk, named:** Task 12's conditional 480-pad resolution
  depends on measuring the pinned reference's branch behavior — the task
  resolves it empirically before transcribing; Task 15's free-form-instruct
  question is resolved at transcription time against the pinned source.
  Neither blocks planning; both are contained within their tasks.

