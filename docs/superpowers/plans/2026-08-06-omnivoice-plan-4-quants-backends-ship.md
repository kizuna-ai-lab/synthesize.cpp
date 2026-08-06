# OmniVoice Plan 4: Quantization, Backends, Ship — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (this family's standing choice) to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take OmniVoice from "port-validated on CPU at F32" to a publishable
Restricted Model Package: a codec-only Quantization Profile, an honest CUDA
Execution Backend claim, the Port Validation Suite's remaining debt closed, and
every ship artifact prepared — with publication itself left as jiangzhuo's
separate per-act decision.

**Architecture:** Four slices. **A (Tasks 1–4)** adds the family's first
Quantization Profile; quantization in this project is a C++ tool
(`tools/synthesize-quantize/`) driven by a per-family tensor classifier, not a
converter flag. **B (Tasks 5–7)** closes the validation-suite debt the Plan 3
carry-over ledger names. **C (Tasks 8–12)** makes the CUDA backend real —
starting with a live honesty defect on `main` — and is the only slice gated on
external setup. **D (Tasks 13–17)** prepares the ship artifacts, which requires
extending the shared HF-card generator, whose template cannot currently describe
this family correctly.

**Tech Stack:** C++17, GGML (pinned submodule), CMake ≥3.24, CUDA 13.3 exactly
(DGX Spark, `sm_121a`), uv-locked oracle env, Jinja2 card generator.

## Global Constraints

- Commit message footer, every commit:

  ```text
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01NFjueGrsKXs44rfZ7DunQK
  ```

- clang-format: `git add` (new files too — the untracked-file trap cost two
  fix rounds in Plan 3) → `scripts/ci/clang-format.sh --fix` → `git add` →
  commit. FINAL command of every task: `scripts/ci/clang-format.sh --check-diff`
  with its literal exit code reported. Never format `ggml/` or `third_party/`.
- Every C++ task ends with BOTH unit gates green:
  `cmake --build build --target synthesize-check-unit` and the same on
  `build-sanitize`.
- Any task touching a validator, runner, tolerance file, or package artifact
  ends with the full omnivoice integration set:
  `ctest --test-dir build --output-on-failure -L integration -R omnivoice`.
  Tasks that could affect another family run the whole set without `-R`.
- **Exact-token discipline is absolute.** Both grids — the greedy decode loop's
  8×T and the Reference Audio clone path's 2,808 RVQ tokens — compare byte-exact
  under every profile and every backend. A quantized profile or a backend that
  flips a token is not shipped; the remedy is the dual-admissibility mechanism
  (enumerate the second grid with its provenance) or dropping the profile, and
  **never** a tolerance on a token id.
- **jiangzhuo's rulings for this plan:**
  1. **Quantization is codec-only** (2026-08-06). The generator stays F32 in
     every shipped profile: a reference port measured token agreement collapsing
     from 100% to ~7% with F16 generator weights, and this family's headline
     claim is exact tokens.
  2. **The ship card's frontmatter is `license: other`** (2026-08-06) with a
     descriptive `license_name` slug, consistent with the 2026-07-30 GGUF ruling
     ("never invent a CC version") and with ADR 0018:29–31. The design spec's
     `cc-by-nc-4.0` wording is superseded — amend it with a dated note rather
     than silently rewriting it.
  3. **CUDA backends are in scope this cycle** (2026-08-06), gated on a CUDA
     13.3 toolkit jiangzhuo is installing. Slice C does not start until
     `cmake --preset dev-dgx-spark` configures.
- **Restricted Model Package (ADR 0018):** the card declares the restrictive
  license in frontmatter, quotes the upstream statement verbatim, carries a
  prominent no-commercial-use statement, says outright that upstream names no
  CC version, records the Emilia training-data provenance, and carries the
  anti-impersonation use disclaimer. The Boson Higgs Audio 2 Community License
  text travels with the artifact as a declared **Sidecar Resource** with dual
  attribution. Never label any weight artifact `apache-2.0`.
- **Publication is a separate act.** This plan prepares artifacts and stops.
  The `hf repos create` / `hf upload` pair runs only after jiangzhuo's explicit
  per-act confirmation, naming the target repository.
- Terms (CONTEXT.md): "Quantization Profile", "Execution Backend", "Restricted
  Model Package" (never "Published Model Package" for this family), "Listening
  Audit" (never "MOS study" / "listening panel"), "Validation Level",
  "Sidecar Resource", "Synthesis".
- **Standing facts (do not rediscover):** greedy makes zero RNG calls; the
  commit schedule and resampler kernel replicate torch's float32
  scalar-promotion order; discrete decisions run on host CPU
  (docs/backends.md:70–72); the pre-scan whitelist is a positive whitelist of
  our own writer's output; golden suite is at `suite_version` 3 with tolerances
  `thresholds-committed-and-enforced`; Release and sanitizer builds differ at
  the float32 ULP level in convolution accumulation order (Plan 3, Task 10).

## File Structure

New:
- `src/arch/omnivoice/quantization.{h,cpp}` — the family's tensor→role
  classifier, shared by the offline tool and the runtime (Kokoro precedent,
  `src/arch/kokoro/quantization.{h,cpp}`), so the two cannot drift.
- `tests/omnivoice_quantization_test.cpp` — classifier unit tests.
- `tests/omnivoice_serialize_writer_agreement_test.cpp` — the fast
  writer↔whitelist test the carry-over ledger's item 5 demands.
- `scripts/hf_cards/omnivoice-0-6b.yaml`, `docs/models/omnivoice-0-6b.md`.
- `models/publish/omnivoice-0-6b/` — the clean flat publication directory
  (git-ignored like `models/`).

Modified (principal): `tools/synthesize-quantize/{policy.cpp,quantize.cpp}`,
`src/arch/omnivoice/{catalog.{h,cpp},model.cpp,weights.cpp}`,
`src/synthesize.cpp`, `scripts/validate-omnivoice-{replay,public}.py`,
`tests/omnivoice_replay_real.cpp`, `tests/tolerances/omnivoice.json`,
`tests/golden/omnivoice/omnivoice-0-6b.manifest.json`, `tests/CMakeLists.txt`,
`scripts/hf_cards/{generate.py,template.md.j2}`,
`docs/{scope.md,model-packages.md,backends.md,testing.md}`, `CONTEXT.md`,
`docs/porting/families/omnivoice.md`,
`reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`.

---

## Slice A — The codec-only Quantization Profile

### Task 1: The family tensor classifier

**Files:**
- Create: `src/arch/omnivoice/quantization.{h,cpp}`
- Create: `tests/omnivoice_quantization_test.cpp`; register in `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `tools/synthesize-quantize/policy.h`'s role vocabulary
  (`MatrixWeight` / `TransposeWeight` / `Sensitive` / `Unknown`) — read it
  before designing, and mirror `src/arch/kokoro/quantization.h`'s shape so the
  tool's dispatch can call this the same way it calls Kokoro's.
- Produces:
  ```cpp
  // src/arch/omnivoice/quantization.h  (namespace synth::omnivoice)
  // The family's tensor->role classifier, shared by tools/synthesize-quantize
  // and the runtime so an offline decision and a load-time expectation cannot
  // drift. Codec-only by jiangzhuo's ruling of 2026-08-06: every generator
  // tensor is Sensitive, whatever its shape.
  QuantRole classify_tensor(const std::string & name, const int64_t ne[4]);
  ```

**The classification contract** (derive each line from
`src/arch/omnivoice/catalog.cpp`'s own registration — do not trust this list
blindly, verify every prefix exists and that no catalog tensor falls through
to `Unknown`):

- **Sensitive (stays reference dtype), all of:** every `llm.*`,
  `audio_embeddings.weight`, `audio_heads.weight` (the whole generator, 312
  tensors — the ruling); every `codec.quantizer.*` (`project_in`/`project_out`
  weights and biases, `codebook.embed` — 40 tensors, matching qwen3-tts's own
  RVQ rule at `policy.cpp:258-265`); `codec.fc.*` and `codec.fc2.*`; every
  Snake `*.alpha`; every bias; every 1-D norm.
- **TransposeWeight (F32 override):** `codec.acoustic_decoder.block.*.conv_t1.weight`
  — the standing transposed-convolution override (`policy.cpp:390-395`,
  `docs/quantization.md:52-56`).
- **Sensitive for a structural reason, documented individually:**
  `codec.acoustic_encoder.conv1.weight` (`ne0*ne1 = 7`) and
  `codec.semantic_model.feat_conv.0.conv.weight` (`= 10`) — packed rows not
  divisible by 32; and
  `codec.semantic_model.encoder.pos_conv_embed.conv.weight` — grouped, sliced
  per group by `ggml_view_3d` in `reference-encoder.cpp`'s `grouped_conv1d`,
  which a quantized tensor cannot serve. Each of these three gets its own
  comment naming the reason; a future reader must not "fix" them.
- **MatrixWeight (quantized):** everything else under
  `codec.acoustic_decoder.`, `codec.acoustic_encoder.`,
  `codec.semantic_model.` and `codec.encoder_semantic.` — expected 158 tensors.

- [ ] **Step 1: Failing tests first.** `tests/omnivoice_quantization_test.cpp`:
  one case per bullet above using real tensor names and shapes read out of
  `catalog.cpp`; a **completeness** case that walks the catalog's full
  registration and asserts NO tensor classifies as `Unknown` (this is the test
  that catches a future catalog addition silently escaping the policy); and a
  **count** case asserting exactly 158 MatrixWeight, with the count derived in
  the test from the catalog rather than hard-coded twice.
- [ ] **Step 2:** Run → FAIL (header absent). Register the test with labels
  `unit;omnivoice`.
- [ ] **Step 3:** Implement. Prefix matching, not substring matching — a
  `.contains()` classifier would misfile `codec.acoustic_encoder` tensors as
  decoder ones.
- [ ] **Step 4:** Run → PASS. Both unit gates.
- [ ] **Step 5:** Commit (footer; format check final with literal exit code).

### Task 2: Wire the classifier into the offline quantizer

**Files:**
- Modify: `tools/synthesize-quantize/{policy.cpp,quantize.cpp}`
- Modify: `tests/` — whatever suite covers the quantizer today (find it:
  grep for `synthesize-quantize` in `tests/CMakeLists.txt`)

**Interfaces:**
- Consumes: Task 1's `classify_tensor`.
- Produces: `resolve_omnivoice_target_spec` in the tool's dispatch, reached
  when `general.architecture == "omnivoice"`.

- [ ] **Step 1:** Read `quantize.cpp:118-134`'s dispatch and `policy.cpp`'s two
  inline resolvers plus Kokoro's out-of-line one. Follow the Kokoro shape
  (out-of-line, in the family directory) — it is the precedent that keeps tool
  and runtime honest.
- [ ] **Step 2: The packing whitelist.** `quantize.cpp:186-189` packs
  `[ne0*ne1, ne2]` for `kokoro || qwen3-tts` only, deliberately as a family
  whitelist rather than a shape rule, because `ggml_n_dims` collapses trailing
  unit dims and VITS's `[7,32,1]` tensor would be mistaken for a matrix.
  OmniVoice has exactly that hazard (`codec.acoustic_decoder.conv2.weight` is
  `[7,32,1]`) — **do not add omnivoice to the whitelist**, and add a comment
  at the whitelist naming omnivoice as the third family deliberately excluded
  and why.
- [ ] **Step 3:** Add the dispatch branch; verify the tool now refuses nothing
  it should accept and still aborts on an unrecognized tensor name (that abort
  is the policy's completeness guarantee — do not soften it).
- [ ] **Step 4:** Unit gates; commit.

### Task 3: Produce and validate the Q8_0 codec profile

**Files:**
- Modify: `tests/tolerances/omnivoice.json` (a new `Q8_CODEC` profile cell)
- Modify: `tests/CMakeLists.txt` (a per-profile golden gate)
- Modify: `docs/porting/families/omnivoice.md`, the porting log

**Naming:** the profile is `Q8_CODEC` unless the tool's existing profile table
(`policy.cpp:14-24`) makes a different name more consistent — read it and
justify the choice in the report. The filename is
`omnivoice-0-6b-<PROFILE>.gguf`, flat, per `docs/model-packages.md:66-73`.

- [ ] **Step 1:** Produce the package with the tool from the committed F32
  package. Record: old/new size, the per-group byte breakdown, and the new
  sha256.
- [ ] **Step 2: The load path.** Confirm the runtime loads it unchanged —
  `src/arch/omnivoice/weights.cpp`'s `read_quantization` must accept the new
  profile name, and `catalog.cpp`'s shape checks must not assert F32 anywhere
  for the quantized tensors (grep `GGML_TYPE_F32` in `src/arch/omnivoice/`
  first; if the catalog does assert, the fix belongs here and its test with it).
- [ ] **Step 3: THE GATE.** Re-run the full replay suite against the quantized
  package: `scripts/validate-omnivoice-replay.py --require all --check
  --profile <PROFILE> --backend CPU --stage replay`. **The 17 greedy grids and
  both clone token grids must be byte-exact.** Quote the result.
  - If a grid flips: STOP. Report the case, the flipped position, and the
    margin from `--margin-report`. Do not adopt a tolerance, do not adjust the
    manifest. The controller and jiangzhuo decide between the
    dual-admissibility path and dropping the profile.
- [ ] **Step 4: Margin protocol** (carry-over item 8): re-run `--margin-report`
  and compare against the F32 baseline table in the porting log. The four
  in-band cases (`omni-short-en` 1.16e-04, `omni-long-boundary` 2.05e-04,
  `omni-rate-fast` 2.44e-04, `omni-lang-none` 6.03e-04) and `omni-rate-slow`
  (9.5e-06) are the predicted first flips — report every case's new margin
  beside its old one.
- [ ] **Step 5:** Measure and commit the profile's tolerance cell: same stage
  set as F32 (`replay` and `public` — `test_tolerance_coverage.py` requires the
  stage sets match), thresholds at ≤5× observed, `all_passed` set from a real
  run and never before it.
- [ ] **Step 6:** Register the per-profile golden gate in `tests/CMakeLists.txt`,
  guarded on the quantized package's existence the way the F32 gate is guarded
  on its own.
- [ ] **Step 7:** Full omnivoice integration set; both unit gates; commit.

### Task 4: The public path under the quantized profile

**Files:** `scripts/validate-omnivoice-public.py`, `tests/CMakeLists.txt`,
`tests/tolerances/omnivoice.json`

- [ ] **Step 1:** Run the public gate against the quantized package
  (`--profile <PROFILE>`): the seed contract must hold (same seed → byte
  identical, different seeds differ, `random` reports concrete) and cloning
  must still work end to end.
- [ ] **Step 2:** Fill the profile's `public` stage cell with the measured
  outcome.
- [ ] **Step 3:** Register the gate; run it; quote the pass line. Commit.

---

## Slice B — Validation-suite debt

### Task 5: Pin the writer↔whitelist agreement with a fast test

**Files:** create `tests/omnivoice_serialize_writer_agreement_test.cpp`;
modify `tests/CMakeLists.txt`

This is carry-over item 5, and the failure mode it closes is the dangerous
direction: if a key is REMOVED from the writer without being removed from
`kPrescanKnownKeys`, the whitelist is silently too permissive and no fast test
notices. Today only the model-guarded round-trip integration test exercises the
real writer against the real whitelist.

- [ ] **Step 1:** Drive the REAL `serialize_clone_prompt` and
  `serialize_design_instruct` against a synthetic in-memory model (no GGUF
  package — the Task 14/16 tests show how to build one), parse the emitted key
  set with a raw GGUF reader, and assert it equals `kPrescanKnownKeys` **exactly
  in both directions** (no key emitted that the whitelist lacks; no whitelist
  entry the writer never emits for either kind — account for the two kinds'
  disjoint keys deliberately, e.g. by asserting the union).
- [ ] **Step 2:** Prove it catches both drift directions: temporarily add a key
  to the writer (test must fail), then temporarily remove one from the writer
  while leaving the whitelist (test must fail). Revert both; report the two
  observations.
- [ ] **Step 3:** Unit gates; commit.

### Task 6: Pin the primary grid digests

**Files:** `tests/golden/omnivoice/omnivoice-0-6b.manifest.json`,
`scripts/validate-omnivoice-replay.py`, `tests/python/test_golden_manifests.py`

Carry-over item 6: `omni-fast-mode` pins its alternate grid's digest, but every
case's PRIMARY grid is whatever the local oracle dump holds. The digest is
knowable and should be recorded.

- [ ] **Step 1:** Compute each case's `codes/grid.i32` sha256 from the committed
  oracle dumps; add them to the manifest beside the existing artifact entries
  (find the schema's legal place — `docs/schemas/synthesize-golden-manifest-v1.schema.json`
  has `additionalProperties: false`, so either an existing field accepts it or
  the schema needs a versioned addition; if the schema must change, say so and
  treat it as part of this task).
- [ ] **Step 2:** Make the validator verify the digest before comparing, the way
  it already does for alternates.
- [ ] **Step 3:** Add the structural test that every case carries one. Run the
  golden gate; commit.

### Task 7: Converter harness pass

**Files:** `scripts/convert-omnivoice.py`, `tests/python/test_convert_omnivoice.py`

Carry-over item 7 — Plan 1's Task 3 items, still end-to-end-only.

- [ ] **Step 1:** `verify_gguf` compares element counts, not shapes — make it
  compare shapes, and add the test that a transposed-but-same-element-count
  tensor is caught.
- [ ] **Step 2:** Move the license-copy existence check ahead of GGUF
  finalization so a missing license file fails before a 3 GB write.
- [ ] **Step 3:** Run the python unit targets; commit.

---

## Slice C — CUDA Execution Backend

**GATE: do not start Slice C until `cmake --preset dev-dgx-spark` configures
successfully.** The host had CUDA 13.0.88 at plan time and
`CMakeLists.txt:140` requires `13.3 EXACT`; jiangzhuo is installing it. If the
preset still fails, report BLOCKED with the configure output rather than
weakening the pin.

### Task 8: Stop claiming a backend this family does not run

**Files:** `src/synthesize.cpp`, `tests/` (a new arm)

**This is a live defect on `main`, independent of the rest of Slice C.**
`src/synthesize.cpp:574-577` accepts `SYNTH_BACKEND_CUDA` for every family with
the comment "Both families claim the same execution backends today; when they
diverge this becomes a per-family question" — and they have diverged. A CUDA
request for OmniVoice loads the model, `synth_model_get_device` reports CUDA,
and every graph runs on `create_cpu_scheduler` over CPU-resident weights
(`model.cpp:1027-1031`): exactly the "a backend that is present is not a
backend that ran" failure `docs/backends.md:242-244` names.

- [ ] **Step 1:** Make backend support a per-family question. Read how
  `ModelInfo` carries family capabilities today and add the backend set there
  (or the narrowest equivalent). OmniVoice declares CPU-only **until Task 11
  lands**; the other three families keep exactly what they claim now — verify
  by running their tests, and if any sibling test changes behavior, STOP and
  report rather than editing a sibling family's expectations.
- [ ] **Step 2:** A CUDA request against an OmniVoice package returns
  `SYNTH_ERR_BACKEND` with a named diagnostic rather than silently running on
  CPU. Add the test arm.
- [ ] **Step 3:** Both unit gates; the FULL integration set (this touches every
  family's load path); commit.

### Task 9: The codec twin, with a filter narrower than qwen3-tts's

**Files:** `src/arch/omnivoice/{catalog.{h,cpp},model.cpp}`

**The trap:** qwen3-tts mirrors every tensor whose name starts `codec.`
(`qwen3-tts/model.cpp:445`). For OmniVoice that prefix covers 486 tensors, but
only the DECODE path (`codec.acoustic_decoder`, `codec.quantizer`, `codec.fc2`)
is movable — `codec.acoustic_encoder`, `codec.semantic_model` and
`codec.encoder_semantic` (332 tensors, ~593 MiB) are read by the CPU-held
clone-encode chain. A copied filter would mirror ~593 MiB to the GPU that no
primary-side graph ever reads.

- [ ] **Step 1:** Add the twin context to `build_model_weights` (the seam
  `catalog.h:176-178` reserves), filtered to the decode-path prefixes only,
  with the group list and its byte cost in a comment.
- [ ] **Step 2:** Mirror only when `primary() != cpu_backend()`; copy after
  streaming, as qwen3-tts does.
- [ ] **Step 3:** `decode_codes` passes `on_primary = twin != nullptr`; every
  other graph keeps `create_cpu_scheduler` explicitly — the generator by the
  discrete-outputs rule, the clone-encode chain because its outputs feed a
  host-side RVQ argmax.
- [ ] **Step 4:** Unit gates (CPU behavior must be unchanged: with no
  accelerator the twin is absent and every path is byte-identical to today —
  prove it by running the full omnivoice integration set on the CPU tree).
  Commit.

### Task 10: Placement evidence, made checkable

**Files:** `tests/omnivoice_replay_real.cpp`,
`scripts/validate-omnivoice-replay.py`, `scripts/validate-omnivoice-public.py`

- [ ] **Step 1:** The runner reports per-stage placement as qwen3-tts's does
  (`{"generator": [nodes, off_cpu], "codec": [...]}`) and gains an accelerate
  mode selecting `SYNTH_BACKEND_CUDA`.
- [ ] **Step 2:** The validator's placement check becomes conditional, mirroring
  `validate-qwen3-tts-replay.py:143-161`: on CPU every node of every stage must
  be on CPU; with `--accelerate` the codec's must ALL have left it while the
  generator's must not have moved. Keep the CPU-only assertion as the default —
  "Plan 2 is CPU-only" becomes "the generator is CPU-only, always".
- [ ] **Step 3:** `validate-omnivoice-public.py`'s `--backend` gains `cuda`.
- [ ] **Step 4:** Commit (no CUDA run yet — that is Task 11).

### Task 11: Run the CUDA sweep and claim the backend

**Files:** `tests/tolerances/omnivoice.json`, `tests/CMakeLists.txt`,
`src/synthesize.cpp` (flip OmniVoice's declared backend set),
`docs/backends.md` (the per-family cost table), the porting log

- [ ] **Step 1:** Configure and build the CUDA tree per `docs/testing.md:173-190`
  (`SYNTH_CUDA_ROOT`, `cmake --preset dev-dgx-spark`).
- [ ] **Step 2:** Replay sweep with `--accelerate`: **every greedy token grid
  must remain byte-exact against the CPU baseline**, every codec node must be
  off CPU, every generator node on it. Quote the counts. A flipped token here
  is the same STOP as Task 3's.
- [ ] **Step 3:** Operational evidence `docs/backends.md:324-345` requires:
  latency, real-time factor, peak memory (`nvidia-smi` while the runner holds
  the device, as qwen3-tts recorded), and the repeated-run cleanup check.
- [ ] **Step 4:** Fill `profiles.<P>.backends.CUDA.stages.{replay,public}` for
  every profile that runs on CUDA — `test_tolerance_coverage.py:120-127`
  requires a backend cell to carry the same stage set as the reference profile,
  so both stages must be measured or the grid breaks.
- [ ] **Step 5:** Flip OmniVoice's declared backend set (Task 8's mechanism) to
  include CUDA, now that it is true. Add the family's row to
  `docs/backends.md`'s per-family cost table (it has only Kokoro and VITS
  today).
- [ ] **Step 6:** Register the CUDA gates; run everything; commit.

### Task 12: Generator-on-CUDA — measure, then decide

**Files:** the porting log; `docs/porting/families/omnivoice.md`

The family doc records the standing rule: "Placing the generator on CUDA is
claimed only if placement evidence proves the committed token grids bit-identical
to CPU; one port measured CUDA-F32 token-exact and Metal-F32 at 83%, which is
encouraging and not evidence."

- [ ] **Step 1:** Measure it as an experiment, off the shipped path: build a
  variant that places the generator on the primary backend and run the 17 greedy
  cases. Record the token agreement exactly (per-case, per-position if any case
  flips).
- [ ] **Step 2: The decision is evidence-driven and belongs to jiangzhuo if the
  answer is ambiguous.** 17/17 byte-exact → propose claiming it, with the TF32
  caveat stated (`docs/backends.md:35-41`: CUDA F32 matmuls compute at TF32 and
  there is no build option to change that). Any flip → the generator stays on
  CPU, the claim is not made, and the measurement is recorded so the next cycle
  does not repeat it.
- [ ] **Step 3:** Record the outcome either way. Commit.

---

## Slice D — Ship

### Task 13: Teach the card generator this family's shape

**Files:** `scripts/hf_cards/{generate.py,template.md.j2}`,
`tests/python/test_hf_card_generator.py`

The shared generator cannot currently describe OmniVoice correctly. Each gap
below is load-bearing; fix them as generator features (other families will need
them), not as OmniVoice special cases.

- [ ] **Step 1: Sidecar Resources.** There is no sidecar concept at all
  (`grep -rn sidecar scripts/hf_cards/` is empty), yet ADR 0018:32–35 requires
  the Boson license text to travel as a declared one. Add a `sidecars` spec key
  (filename, role, sha256, size), validate them the way `quants` are validated
  (`generate.py:137-151` opens and digest-checks every listed file), and render
  them in the card.
- [ ] **Step 2: Text input.** The "Voices and input" body is hardcoded to
  phonemes + `synthesize.symbol_map` and states "Callers starting from raw text
  must currently run a compatible G2P frontend externally" — false for this
  family. Branch on `capabilities.frontend_provider` / the declared input kinds.
- [ ] **Step 3: Usage.** The usage block hardcodes `--phonemes`; this family's
  CLI flag is `--text`.
- [ ] **Step 4: Voice mode.** `voice_mode ∈ {preset_catalog, fixed_default}`
  describes neither an empty catalog with an unnamed default whose voice follows
  the seed, nor Reference Audio / Description Text Profile sources. Add the
  third mode and its rendering.
- [ ] **Step 5: The CUDA sentence.** The validation prose asserts "CUDA
  placement contained zero executable CPU fallback nodes" unconditionally —
  make it conditional on a declared backend claim, since for this family the
  generator deliberately stays on CPU and the unconditional sentence would be
  false.
- [ ] **Step 6:** Extend `tests/python/test_hf_card_generator.py` for every new
  rule; run the python unit target; commit.

### Task 14: The publication directory and the ship artifacts

**Files:** `scripts/hf_cards/omnivoice-0-6b.yaml`,
`docs/models/omnivoice-0-6b.md`, `models/publish/omnivoice-0-6b/`

- [ ] **Step 1: The upload hazard.** `models/omnivoice-0-6b/` holds the upstream
  checkpoint (`model.safetensors` 2.45 GB, `audio_tokenizer/`, `tokenizer.json`,
  the upstream README) beside our GGUFs. `hf upload … models/omnivoice-0-6b .`
  would publish the raw upstream weights. Kokoro's directory was clean and its
  upstream card lived at `models/upstream/<slug>/README.md`. Build a clean flat
  publication directory holding only: the GGUFs, `README.md`, and the license
  sidecars. Record the layout in `docs/models/omnivoice-0-6b.md`'s reproduction
  section.
- [ ] **Step 2:** Write the YAML: `license: other` + the descriptive
  `license_name` slug (the ruling), `license_link` to the upstream model page,
  the verbatim upstream license quote, the no-commercial-use statement, the
  "upstream names no CC version" sentence, Emilia provenance, the
  anti-impersonation disclaimer, every profile's size/sha256/tensor types, the
  declared sidecars, and `listening_audit: not_run` until Task 16 changes it.
- [ ] **Step 3:** Write `docs/models/omnivoice-0-6b.md` on the
  `docs/models/kokoro-v1-0.md` skeleton: Status with Validation Level and the
  explicit "Quality evaluation has not been run", Package (with the profile
  table and what the profile quantizes and never does), Port validation (stage
  and case counts, the two exact-token claims, the probe table, backends),
  Reproduction (oracle dump, integration build, the quantize command, card
  generate + `--check`, and the publication commands as commands, not as an
  invitation to run them), Licensing.
- [ ] **Step 4:** Generate the card, run `--check`, commit the YAML and the
  page (the rendered README stays git-ignored). Commit.

### Task 15: Bring the ADR 0018 vocabulary into the confirmed docs

**Files:** `docs/scope.md`, `docs/model-packages.md`, `CONTEXT.md`

ADR 0018 is accepted, but the documents it re-reads have not been updated:
`docs/scope.md:21-22` still says only "Publishing validated, directly loadable
Model Packages"; `docs/model-packages.md` describes only the Published flow;
`CONTEXT.md`'s Validation Level entry still says "for a **Published** Model
Package".

- [ ] **Step 1:** Update each, minimally, to admit the restricted category
  exactly as ADR 0018:38–40 frames it — an addition, not a reinterpretation.
- [ ] **Step 2:** Bump each file's `Status: Confirmed` line date. Commit.

### Task 16: The Listening Audit

**Files:** the audit page and clips (git-ignored working artifacts), the YAML's
`listening_audit` + `listening_audit_detail`, the porting log

**This gate is jiangzhuo's and cannot be delegated.** A tolerance grid is not
audible evidence.

- [ ] **Step 1:** Prepare the pairs. The port and oracle agree at 102–119 dB SNR
  and the material is regenerable from
  `build/goldens/omnivoice-replay/<case>/pcm.f32` against
  `build/goldens/omnivoice/<case>/audio/pcm.f32`. Follow
  `docs/model-porting.md:244-271`'s selection rule: at most six A/B pairs, order
  deterministically randomized, and the report records the comparison
  identities, case hashes, selection reason, and order seed. Cover every shipped
  profile and backend, and include at least one clone case and one
  Description Text case — this family's two headline paths.
- [ ] **Step 2:** Build the comparison page. Two hard requirements learned from
  qwen3-tts, where the audit had to be re-run twice: the page must NOT trim or
  cap audio (a ten-second cap invalidated one round), and it must play the same
  offset when switching sides.
- [ ] **Step 3: Offer it to jiangzhuo** with the page ready and the pairs
  prepared — the carry-over ledger says to offer rather than wait to be asked.
  Then STOP until the verdict comes back.
- [ ] **Step 4:** Record the verdict as exactly one of
  `no_obvious_regression` / `regression` / `not_run` in the YAML plus the
  detail block (listeners, cases, profiles, backends, date, method), and the
  matching sentence on the Model Page. A `regression` verdict requires an
  investigation or an explicit known-limitation note — it does not silently
  ship. Commit.

### Task 17: Close-out

**Files:** `docs/porting/families/omnivoice.md`,
`reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`,
`docs/superpowers/specs/2026-07-30-omnivoice-family-design.md`,
`docs/testing.md`, a Plan 5 carry-over ledger if anything remains

- [ ] **Step 1:** Family doc: the Quantization Profile and exactly what it
  quantizes and never does; the backend claim with its placement evidence and
  the TF32 caveat; the generator-on-CUDA decision from Task 12; the Listening
  Audit outcome; the restricted-publication statement.
- [ ] **Step 2:** Porting log: one dated Plan 4 section carrying the numbers —
  profile sizes and digests, the exact-token results under every profile and
  backend, the margin table beside the F32 baseline, latency/RTF/peak memory,
  the audit's method and verdict.
- [ ] **Step 3:** Amend the design spec's `cc-by-nc-4.0` sentence with a dated
  note pointing at the 2026-08-06 ruling (its own amendment convention).
- [ ] **Step 4:** `docs/testing.md`: the new gates and their runtimes; Status
  date.
- [ ] **Step 5:** Final gates, all quoted with exit codes: both unit trees; the
  full integration set; the CUDA tree's set if Slice C landed; the card
  `--check`; `clang-format --check-diff`.
- [ ] **Step 6:** Write the carry-over ledger for whatever remains (the
  project-wide questions this plan does not own: the `SYNTH_ASSERT` policy,
  the `synthesis.graph_failed` diagnostic naming, the `std::optional` stream
  micro-optimization, the validation-order test, and the float64 cosine
  estimator if Task 3 did not force it). Commit.

---

## Self-Review Notes

- **Spec coverage:** design-spec slices 10 (quants) → Tasks 1–4; 11 (backends)
  → Tasks 8–12; 13 (ship) → Tasks 13–17. Carry-over items: 5 → Task 5, 6 →
  Task 6, 7 → Task 7, 8 → Task 3 Step 4, 10 → Task 16. Items 1–4 and 9 are
  project-wide questions this family should not settle unilaterally — they go
  to the Plan 5 ledger in Task 17 Step 6, stated as such rather than dropped.
- **The plan's two STOP conditions are deliberate:** a flipped token under a
  quantized profile (Task 3) or under CUDA (Task 11) halts that slice rather
  than adapting a threshold. Both are the family's headline claim.
- **Slice ordering is dependency-real:** A and B need no GPU and run first; C
  is gated on the toolkit; D's Task 14 needs A's digests and C's backend claim,
  so it comes last. Task 8 is in C only because it is about backends — it needs
  no GPU and can be pulled forward if the toolkit slips.
- **Known open risk, named:** Task 6 may require a golden-manifest schema
  change (`additionalProperties: false`); that is contained in the task and
  reported if it happens.
