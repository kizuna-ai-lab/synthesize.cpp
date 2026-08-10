# OmniVoice Plan 5: The Generator On GPU — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (this family's standing choice) to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move OmniVoice's mask-predict generator onto the primary Execution
Backend — the only order-of-magnitude lever this family has, per two
independent RTF investigations — now that jiangzhuo has revised the bar from
token identity to audible quality and a six-pair blind listening test heard
no problem in generator-on-CUDA output, including a pair whose waveforms
have cosine 0.0515 with 98.3% of tokens flipped. Land the weight twin and the
placement move first and measure the real end-to-end RTF honestly; the
validation-shape redesign, the documentation debt the decision creates, and
the further speed levers it reopens (quantization, step count) follow once
the number is on the table.

**Architecture:** Five slices. **A (Tasks 1–2)** is the decisive piece: the
generator weight twin (Task 9's codec-twin pattern, generalized) and the
placement move, plus an honest, un-gated RTF measurement through the public
seam — this is the only slice this plan executes now. **B (Tasks 3–4)**
redesigns the validation shape a backend that no longer preserves token
identity needs, following qwen3-tts's own precedent for its quantized
talker profiles. **C (Tasks 5–6)** writes the documentation debt the
decision opens: `docs/backends.md`'s principled exception to the
discrete-outputs rule, and the four-way correction to this family's
ServeurpersoCom prior-art entry the decision record also owes. **D (Task 7)**
corrects the ship artifacts Plan 4 prepared under the superseded claim.
**E (Tasks 8–9)** reconsiders the two speed levers Plan 4 closed on
token-identity grounds now that the bar has moved: generator quantization
and step-count reduction. **Task 10** closes out.

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
  ends with the full omnivoice integration set ON THE CPU TREE:
  `ctest --test-dir build-integration -L integration -R omnivoice` (or the
  equivalent tree with `SYNTH_BUILD_INTEGRATION_TESTS=ON`). The CUDA tree
  (`build/dev-dgx-spark`) is used for measurement, not as a required gate,
  until Task 4 redesigns what "passing" means for it — see the next bullet.
- **The exact-token discipline that governed Plans 2–4 is retired for the
  primary CUDA-generator path as of this plan, by jiangzhuo's 2026-08-08
  ruling, and this is a deliberate, known break of one existing gate.**
  `tests/CMakeLists.txt`'s `synthesize-omnivoice-replay-golden-cuda` target
  and `tests/tolerances/omnivoice.json`'s
  `profiles.F32.backends.CUDA.stages.replay` cell were written for
  Plan 4's codec-only CUDA claim and assert byte-exact token grids under
  `--backend CUDA`. Task 1 makes the generator move under that same flag,
  which the Plan 4 Task 12 experiment already measured collapses token
  agreement to 45.34% — so that gate WILL fail if run after Task 1, and nothing
  in Tasks 1–2 fixes it. This is intentional: Task 3 designs the replacement
  validation shape (waveform-under-tolerance on a replayed grid, not
  byte-exact tokens), and Task 4 registers it. Do not "fix" the old gate by
  loosening its exact-token assertion in Task 1 or Task 2 — that assertion is
  retired by a considered redesign, not patched in place. The **F32/CPU
  reference profile keeps the exact-token contract unconditionally**: it is
  how the port's correctness is known at all, and nothing in this plan touches
  it.
- **jiangzhuo's ruling for this plan (2026-08-08):** the bar for this family's
  synthesis output is audible quality, not token identity. Established by a
  six-pair blind A/B (order seed 20260808, case seed 12; all six
  "no problem heard"), including `omni-digits` at cosine 0.0515 with 98.3% of
  tokens flipped between the CPU and CUDA generator — two audibly-fine, mostly
  unrelated waveforms. This reopens quantization and step-count as live
  questions (Tasks 8–9) that Plan 4 closed only because it was measuring the
  wrong thing.
- **Standing facts from Plan 4 (`docs/superpowers/plans/2026-08-07-omnivoice-plan-5-carryover.md`),
  updated by this ruling where it supersedes them:**
  - This family ships F32-only for the CODEC's Quantization Profile (Q8_MIXED
    and F16 both failed the exact-token gate on the clone-encode RVQ path,
    which is unrelated to the generator and this ruling does not reopen it —
    only Task 8's GENERATOR quantization question is new).
  - The CUDA Execution Backend claim, as shipped by Plan 4, covers the codec's
    decode path only; this plan's Task 1 changes what CUDA does but the
    ALREADY-SHIPPED claim is not itself revoked by that — Task 4 is where the
    claim's scope gets corrected in the places that assert it.
  - The Listening Audit verdict from Plan 4 Task 16 (`no_obvious_regression`,
    six pairs, port-vs-oracle and one CUDA-vs-CPU CODEC pair) is a different,
    earlier audit from the one this plan's own ruling rests on (six pairs,
    generator-on-CPU vs generator-on-CUDA, 2026-08-08). Both stand; neither
    supersedes the other; Task 6 records the new one in the family doc
    alongside the old.
  - The margin screen does not transfer to backend-placement questions (Task
    12's finding): do not use it to reason about whether a future placement or
    precision change is "safe" — Tasks 8–9 must measure by ear or by
    replayed-grid waveform tolerance, not by the margin table.
  - Publication has not happened and this plan does not change that: `hf
    repos create` / `hf upload` require jiangzhuo's separate, per-act,
    named-target confirmation regardless of what this plan lands.
  - The four project-wide questions, the float64 cosine estimator question,
    and the two architectural notes (Task 2's by-name demotion exception,
    Task 9's blanket-codec-prefix warning) in the Plan 5 carryover ledger are
    still open and still not this plan's to settle unilaterally; Task 10
    carries them into the next ledger unchanged unless a task below happens
    to touch one directly.
- Terms (CONTEXT.md): "Quantization Profile", "Execution Backend", "Voice
  Profile", "Validation Level", "Listening Audit" (never "MOS study" /
  "listening panel"), "Synthesis". This plan additionally needs a term for
  "a backend whose validation does not require exact-token identity" —
  Task 3 proposes one rather than inventing prose ad hoc at each use site.

## File Structure

New:
- Nothing under `src/` — Task 1 extends `catalog.{h,cpp}` and `model.cpp` in
  place, following Task 9's own file list rather than adding new modules.
- `tests/omnivoice_catalog_test.cpp` gains `check_generator_twin_resolution`,
  analogous to `check_twin_resolution`.

Modified (principal, by task):
- Task 1: `src/arch/omnivoice/{catalog.h,catalog.cpp,model.cpp}`,
  `tests/omnivoice_catalog_test.cpp`.
- Task 2: the porting log only (a measurement entry; no source changes are
  anticipated, but if the public-seam driver needs a flag it does not already
  have, that change lands here, not silently inside Task 1).
- Task 3: `scripts/validate-omnivoice-replay.py`,
  `scripts/validate-omnivoice-public.py`, `tests/omnivoice_replay_real.cpp`
  (if the replay driver needs a decode-only/replayed-grid mode it does not
  already expose via `--alt-grid`), `docs/model-porting.md` or `CONTEXT.md`
  (the new validation-shape term).
- Task 4: `tests/tolerances/omnivoice.json`, `tests/CMakeLists.txt`,
  `src/synthesize.cpp` (if the backend claim's scope needs a new capability
  bit rather than a doc correction).
- Task 5: `docs/backends.md`.
- Task 6: `docs/porting/families/omnivoice.md`,
  `reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`.
- Task 7: `scripts/hf_cards/omnivoice-0-6b.yaml`, `docs/models/omnivoice-0-6b.md`.
- Task 8–9: `tools/synthesize-quantize/{policy.cpp,quantize.cpp}`,
  `src/arch/omnivoice/quantization.cpp`, `tests/tolerances/omnivoice.json`,
  generation defaults in `src/arch/omnivoice/weights.cpp` or the HParams
  default, the porting log.
- Task 10: `docs/superpowers/plans/2026-08-08-omnivoice-plan-6-carryover.md`
  (or whatever plan follows), `docs/testing.md`.

---

## Slice A — The generator on GPU (executed by this session)

### Task 1: The generator weight twin, and the placement move

**Files:**
- Modify: `src/arch/omnivoice/catalog.h`, `src/arch/omnivoice/catalog.cpp`,
  `src/arch/omnivoice/model.cpp`
- Modify: `tests/omnivoice_catalog_test.cpp`

**Interfaces:**
- Consumes: `catalog.h`'s existing `Resolver`/`resolve_generator` (unchanged),
  and Task 9's own `bind_decode_weights` as the pattern to generalize.
- Produces: `bind_generator_weights(ggml_context * generator_context, const
  HParams & hparams, const ModelWeights & weights, ModelWeights &
  generator_weights)`, and a second twin context/buffer on `Model::Impl`.

**Before writing anything:** grep every host-side reader of
`GeneratorWeights` fields (`weights.generator`, `llm.*`,
`audio_embeddings.weight`, `audio_heads.weight`) across `src/` and `tests/`.
Confirmed at plan-writing time: the ONLY consumers are
`build_canvas_embedding` and `build_generator_forward`
(`src/arch/omnivoice/generator.cpp`), both reached exclusively through
`model.cpp`'s file-local `generator_branch_forward`, itself called only from
`Model::run_synthesis`'s three call sites (the step-0 conditional forward,
the per-step conditional refill, and the per-step unconditional branch).
Unlike the codec's `codec.quantizer.*` group, there is **no second consumer**
of the generator's weights analogous to `rvq_encode`'s host-side read of the
quantizer tables — but the twin must still be built the second-consumer-safe
way (never mutate `weights` itself), both because a future reader that
bypasses `generator_branch_forward` must keep seeing the CPU-resident
package by construction rather than by continued vigilance, and because
`tests/omnivoice_catalog_test.cpp`'s unit tests exercise
`build_model_weights`'s output directly and must keep doing so unaffected by
whether a twin exists elsewhere.

- [ ] **Step 1: Failing tests first.** Add `check_generator_twin_resolution`
  to `tests/omnivoice_catalog_test.cpp`, mirroring `check_twin_resolution`'s
  shape: a synthetic package plus a synthetic generator-only twin context;
  assert `weights.generator.*` stays pointer-identical to the package with NO
  twin; assert `generator_weights.generator.*` points at the twin's own
  tensors when one is bound, while `weights.generator.*` is provably
  untouched (same pointers as before `bind_generator_weights` ran); assert
  every non-generator field of `generator_weights` still traces back to the
  package unchanged; assert a twin missing a tensor or disagreeing in shape
  is refused (`SYNTH_ERR_GGUF`) without mutating `weights`. Run → FAIL
  (`bind_generator_weights` does not exist yet).
- [ ] **Step 2: `catalog.h`/`catalog.cpp`.** Declare and implement
  `bind_generator_weights`, structured exactly like `bind_decode_weights`:
  `generator_weights = weights;` first (whole-struct copy), then — only when
  `generator_context != nullptr` — re-resolve `generator_weights.generator`
  against a fresh `Resolver` over `generator_context` using the existing
  `resolve_generator`. Document the byte cost in the header comment the way
  `bind_decode_weights` does, sourced from
  `reports/convert/omnivoice/omnivoice-0-6b-F32.json`: 312 tensors, verify
  the exact byte total from that file rather than trusting this plan's own
  arithmetic (compute it: sum of `bytes` for every tensor whose name starts
  `llm.` or equals `audio_embeddings.weight`/`audio_heads.weight`).
- [ ] **Step 3: `model.cpp` — the twin context.** Add `is_generator_tensor`
  (an `llm.` prefix plus the two exact names), a second twin context/buffer
  pair on `Model::Impl` (name them for what they hold — `generator_context`/
  `generator_buffer`/`generator_weights`, parallel to `codec_context`/
  `codec_buffer`/`decode_weights`), and build/allocate/copy them in
  `Model::load` the same way the codec twin is built: declared before
  binding, built only when `primary() != cpu_backend()`, allocated against
  `backend_plan->primary()`, copied from the streamed `weights_context` after
  streaming completes. Update every comment this touches — the file's
  top-of-file placement note, `GraphRun::run`'s `on_primary` comment ("only
  Model::decode_codes ever passes true... the generator never passes it" is
  no longer true), and `Model::Impl`'s field comments — to describe the new
  state rather than leaving a stale claim next to code that contradicts it.
- [ ] **Step 4: The placement move.** Give `generator_branch_forward` a `bool
  on_primary` parameter, forwarded to its `run.run(logits_tensor,
  "omnivoice.generator", threads, on_primary)` call. In
  `Model::run_synthesis`: commit the `Persistent inputs` buffer (`t_text`,
  `t_cond_audio`, `t_uncond_audio`, `t_cond_pos`, `t_uncond_pos`) to
  `backend_plan->primary()` rather than `cpu_backend()` when
  `impl.generator_context != nullptr` (decode_codes's own rule for its
  input leaf, avoiding a cross-backend copy the scheduler would otherwise
  insert); change all three `generator_branch_forward` call sites to pass
  `impl.generator_weights` instead of `impl.weights`, and
  `impl.generator_context != nullptr` as `on_primary`. **Both CFG branches
  move** — the conditional forward (step 0 and the per-step refill) and the
  unconditional forward both go through the same twin-aware weights and the
  same `on_primary` flag; there is no reason for one branch to move and not
  the other, since both read the identical weight tensors.
- [ ] **Step 5: CPU bit-identity, proven not asserted.** With no twin
  (`generator_context == nullptr`, i.e. every CPU-only configuration this
  project builds today), `bind_generator_weights` returns a plain copy and
  `on_primary` is always `false` — byte-identical code path to before this
  task. Prove it: both unit gates, then the full omnivoice integration set on
  the CPU tree (`ctest --test-dir build-integration -L integration -R
  omnivoice`, or the CPU tree's equivalent) must still show 17/17 exact
  greedy grids and 2/2 exact clone-token grids, unchanged from Plan 4's own
  numbers.
- [ ] **Step 6:** `scripts/ci/clang-format.sh --check-diff`, literal exit
  code. Commit.

### Task 2: RTF, measured honestly

**Amended 2026-08-08, after execution.** Step 1 below, as originally written,
directed this measurement onto the `dev-dgx-spark` CUDA tree — the same tree
every prior OmniVoice RTF figure used, and the root cause of what this
amendment corrects. That tree's `dev-dgx-spark` preset inherits
`RelWithDebInfo`, compiling ggml-cpu at `-O2` rather than the `-O3` a
`Release` build ships; measured 2.19× slower on this workload. Task 2's own
first execution reported a 41.8× speedup (269.7192 s / RTF 9.365 CPU,
6.4541 s / RTF 0.224 CUDA) from exactly that tree, and it is retracted for
the same reason. The honest pair, from a fresh Release CUDA tree
(`build/rel-dgx-spark`, now the committed `rel-dgx-spark` CMake preset):
**122.61 s / RTF 4.263 (CPU) vs. 5.481 s / RTF 0.1906 (CUDA), 22.4×** — full
method and the erratum are in the porting log's 2026-08-08 entry; the
project-wide rule this adds (a `dev-*` preset proves correctness, never
timing) is in `docs/testing.md`. Step 1 is left as originally written below,
for the historical record of what was actually run; a repeat of this
measurement should build on a `Release`-typed tree instead (`rel-dgx-spark`
for this same host).

**Files:** `reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md` (a new
dated entry); no source changes are expected, but if
`tests/omnivoice_public_real.c` needs a capability it does not already have
to complete this measurement, that change belongs here, named as such, not
folded silently into Task 1.

This is deliberately NOT a registered gate and NOT a claim: Task 4 is where
a backend/profile claim gets made, once Task 3 has designed what evidence a
non-exact-token backend owes. This task exists to put one honest number on
the table, because it is the number the whole plan exists for.

- [ ] **Step 1:** Build `synthesize-omnivoice-public-real` on the
  `dev-dgx-spark` CUDA tree with Task 1's changes
  (`SYNTH_CUDA_ROOT=/usr/local/cuda-13.3` etc., per `docs/testing.md`).
- [ ] **Step 2:** Run it twice against `omnivoice-0-6b-F32.gguf`, same binary,
  `omni-long-boundary`'s own committed text
  (`tests/golden/omnivoice/omnivoice-0-6b.manifest.json`'s `input.text`),
  seed 0, language `en`, `max-frames 0` — once with the `cpu` backend
  positional, once with `cuda` — through the public seam
  (`synth_synthesize_to_buffer`, not the internal replay runner that bypasses
  the backend-capability gate). Record `synthesis_seconds`, `frames`, and
  `sample_rate` from each run's own JSON line.
- [ ] **Step 3:** Compute RTF = `synthesis_seconds / (frames / sample_rate)`
  for each run and the speedup (CPU RTF / CUDA RTF). Report both numbers
  plainly, without editorializing toward a more flattering framing than the
  measurement supports — this plan's own goal statement already commits to
  that. Compare against Plan 4's own codec-only anchor
  (`docs/backends.md`'s table row: 267.30 s / RTF 9.294 on the same case) so
  the reader can see how much of the remaining wall time the generator's
  move actually recovers.
- [ ] **Step 4:** Record the method and the numbers in the porting log,
  dated. State plainly what this measurement does and does not establish:
  it is one case, one host, one run each way — not a claim of Support, which
  Task 4 owns once the validation shape exists to back one.
- [ ] **Step 5:** `scripts/ci/clang-format.sh --check-diff`, literal exit
  code (a no-op if no source changed). Commit.

---

## Slice B — The validation shape for a backend where tokens drift

### Task 3: Design the replayed-grid, waveform-tolerance validation path

**Files:** `scripts/validate-omnivoice-replay.py`,
`scripts/validate-omnivoice-public.py`, possibly `tests/omnivoice_replay_real.cpp`,
`CONTEXT.md` or `docs/model-porting.md` (the new term)

qwen3-tts's own precedent for its quantized talker profiles
(`tests/tolerances/qwen3-tts.json`'s F16/Q8_MIXED cells): probes that are
reachable before any token draws happen (prefill) gate on cosine against the
dtype-only oracle comparison; the waveform gates on both cosine and max_abs
under a tolerance five times the measured deviation; and "structural
exactness comes from the replayed codes rather than from any threshold
here" — i.e. the comparison that matters is decode of a KNOWN, committed code
grid through the candidate graph, not insistence that the candidate graph
regenerate an identical one. OmniVoice's own `--alt-grid` mechanism (already
used to decode Task 12's flipped CUDA grids for the 2026-08-08 listening
test) is the seam this reuses, not a new one.

- [ ] **Step 1:** Name the term this validation shape needs (Global
  Constraints above flags the gap) — something like "Replayed-Grid
  Validation" or "Audible-Parity Backend" — and add it to CONTEXT.md's
  vocabulary with an "avoid" pairing against whatever loose phrase this plan
  used ad hoc above.
- [ ] **Step 2:** Extend the replay validator with a mode that, for a
  declared non-exact-token backend/profile: (a) generates fresh, does NOT
  require the resulting greedy grid to match the CPU/oracle grid, and reports
  the token-agreement percentage as observed data rather than a pass/fail
  gate; (b) separately decodes a COMMITTED grid (the CPU F32 reference's own
  primary grid, via `--alt-grid` or equivalent) through the candidate
  backend's codec and compares the resulting waveform under a committed
  cosine/max_abs tolerance — this is the check that actually gates.
- [ ] **Step 3:** Decide, and record the reasoning: does this validation
  shape apply to `SYNTH_BACKEND_CUDA` as already claimed (meaning the
  existing claim's meaning changes under this plan) or does it need a
  distinct backend/profile identifier so a caller who wants the Plan-4-era
  exact-token CUDA codec-only behavior can still ask for it? The decision
  record's own framing ("the F32/CPU reference profile KEEPS the exact-token
  contract") suggests CPU stays the exact-token reference and CUDA becomes
  the audible-parity path outright, but this is exactly the kind of
  public-contract question that should not be settled by silent inference —
  if it is ambiguous, it goes to jiangzhuo rather than being decided here.
- [ ] **Step 4:** Extend `scripts/validate-omnivoice-public.py`'s existing
  seed-contract checks (same seed → byte identical, different seeds differ,
  `random` reports concrete) to run under the new shape too — the public
  seed contract does not depend on token identity and must keep holding.
- [ ] **Step 5:** Unit/python gates; commit.

### Task 4: Register the CUDA-generator gate, retire the stale one

**Files:** `tests/tolerances/omnivoice.json`, `tests/CMakeLists.txt`,
`src/synthesize.cpp` (only if Task 3 Step 3 concluded a new backend/profile
identifier is needed)

- [ ] **Step 1:** Run Task 3's validation shape end to end on the CUDA tree
  across the full case suite (not just `omni-long-boundary`); commit the
  measured waveform tolerance cell(s) the same way Task 3 of Plan 4 committed
  the codec-only CUDA cell (5× measured deviation, floored/ceiled per this
  family's established convention).
- [ ] **Step 2:** Replace `synthesize-omnivoice-replay-golden-cuda`'s
  exact-token assertion with the new shape's assertion (or add a
  differently-named gate if Task 3 concluded the old one should keep
  asserting exact tokens under a distinct, narrower request the codec-only
  claim still supports — follow whatever Task 3 decided, do not re-decide it
  here).
- [ ] **Step 3:** Update `family_supports_explicit_backend` / whatever
  capability surface Task 3 needs, if any.
- [ ] **Step 4:** Both unit gates; the full omnivoice integration set on the
  CPU tree; the new CUDA gate on `build/dev-dgx-spark`; commit.

---

## Slice C — Documentation debt this decision opens

### Task 5: The principled exception in `docs/backends.md`

**Files:** `docs/backends.md`

The decision record's own argument, not yet written into the confirmed
policy doc: `docs/backends.md:70–72`'s discrete-outputs rule holds a stage
with a discrete output on CPU because Kokoro's counter-example showed TF32
can move a DURATION across a rounding boundary (376 → 377 frames),
producing a STRUCTURAL failure (mismatched tensor shapes, not a numeric
tolerance question) in 18 of 21 CUDA runs. OmniVoice's generator also has a
discrete output — a committed token index — but the canvas LENGTH is fixed
by `RuleDurationEstimator` before the first generator forward runs, so TF32
can change WHICH codebook entry is committed but never HOW MANY frames
exist. Verified in Task 1/Task 12's own data: grid sizes were identical in
17/17 cases between CPU and CUDA-generator runs even as token CONTENT
diverged in 14 of them. The rule as written does not distinguish "a discrete
output that determines downstream SHAPE" (Kokoro's duration, VITS's
duration) from "a discrete output that determines CONTENT within an
already-fixed shape" (OmniVoice's token grid) — and only the first kind
produces the un-thresholdable structural mismatch the rule exists to
prevent.

- [ ] **Step 1:** Add a subsection (or amend the existing one, dated) stating
  this distinction explicitly, with Kokoro's 376→377 case as the
  counter-example that justifies keeping the rule for shape-determining
  discrete outputs, and OmniVoice's fixed-canvas argument plus the 17/17
  same-size measurement as the case that earns the exception.
- [ ] **Step 2:** Update the per-family cost table row (currently: "the whole
  generator... 84.24 of 3042.2 MiB... 267.30 s → 258.48 s, −3.4%... 8.99×
  real time") to reflect that the generator now also moves, using Task 2's
  measured numbers, and add the generator's own mirrored-weight byte cost
  (Task 1's ~2,336.8 MiB figure) beside the codec's.
- [ ] **Step 3:** State plainly that this is a per-family judgment call, not a
  blanket reversal of the discrete-outputs rule — a future family whose
  discrete output DOES gate downstream shape (or whose evidence does not
  include a listening test at this scale) is still held on CPU by the
  original rule.
- [ ] **Step 4:** Bump the file's `Status: Confirmed` date. Commit.

### Task 6: Fix the four-way prior-art error, and record the new listening verdict

**Files:** `docs/porting/families/omnivoice.md`,
`reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`

The decision record names four inaccuracies in this family doc's entry for
`ServeurpersoCom/omnivoice.cpp` (`docs/porting/families/omnivoice.md`'s Prior
Art table), found 2026-08-08:

- [ ] **Step 1:** Verify each of the four claims independently before editing
  (do not transcribe the decision record's findings uncritically into a
  confirmed doc without re-checking, since this doc feeds licensing and
  attribution decisions): (a) there is no `ARCHITECTURE.md` in that
  repository — the ~900-line document currently cited belongs to a different
  port; (b) the col2im_1d recipe this project used came from `qwentts.cpp`,
  not from ServeurpersoCom; (c) ServeurpersoCom publishes no GGUFs itself —
  its README points at a third-party community repository; (d) its license
  is Apache-2.0, not MIT.
- [ ] **Step 2:** Correct the Prior Art table row and any other place in this
  doc or `THIRD_PARTY_NOTICES.md` that repeats the wrong license or the wrong
  attribution. The license error is the serious one — check whether anything
  downstream (the model card, a sidecar) inherited it.
- [ ] **Step 3:** Rewrite the "Generator on CUDA" Open Question: it currently
  reads as settled ("the generator does not move, and is not claimed to").
  Replace with this plan's outcome, keeping the Plan 4 Task 12 measurement
  (45.34% token agreement, the per-case flip table) as historical record
  rather than deleting it — it is still the evidence that the naive
  per-token argument fails, which is exactly why the bar moved to audible
  quality instead.
- [ ] **Step 4:** Add the 2026-08-08 Listening Audit (six pairs,
  generator-on-CPU vs generator-on-CUDA, order seed 20260808, case seed 12,
  all six "no problem heard", `omni-digits` at cosine 0.0515/98.3% flipped as
  the load-bearing pair) to the family doc's Listening Audit section
  alongside Plan 4 Task 16's earlier, differently-scoped audit — both stand,
  dated separately, neither replaces the other.
- [ ] **Step 5:** Porting log entry, dated, with Task 1's twin implementation
  notes, Task 2's RTF numbers, and a pointer to the new listening audit's
  full method (order seed, case seed, per-pair cosines) if not already
  recorded verbatim somewhere this doc can cite instead of duplicate.
- [ ] **Step 6:** Commit.

---

## Slice D — Ship artifact correction

### Task 7: Correct the ship artifacts Plan 4 prepared under the superseded claim

**Files:** `scripts/hf_cards/omnivoice-0-6b.yaml`, `docs/models/omnivoice-0-6b.md`

PR #7's ship artifacts (Plan 4 Task 14) say F32-only and CPU+partial-CUDA
(codec only); this plan supersedes both the backend scope and, depending on
Task 8/9's outcome, possibly the profile claim too.

- [ ] **Step 1:** Update the backend/placement prose and the RTF figures with
  Task 2's (and, if it landed, Task 4's) measured numbers — do not leave the
  card citing Plan 4's 267.30 s/9.294 figures as if they were still the
  shipped configuration's own numbers.
- [ ] **Step 2:** If Task 3 concluded CUDA is no longer an exact-token path,
  update the card's validation-level prose accordingly — this family's
  headline claim changes from "exact tokens on every backend" to "exact
  tokens on the F32/CPU reference profile; audible parity, backed by a
  Listening Audit, on CUDA."
  Regenerate, `--check`, commit (rendered README stays git-ignored).
- [ ] **Step 3:** Note whether Task 8/9's outcome (if landed by this point)
  changes the profile table; if those tasks have not landed yet, say so
  explicitly rather than leaving a stale implication that F32-only is still
  the last word.

---

## Slice E — Reconsider the levers Plan 4 closed on the wrong grounds

### Task 8: Reconsider generator quantization (Q8_0), judged by ear

**Files:** `tools/synthesize-quantize/{policy.cpp,quantize.cpp}`,
`src/arch/omnivoice/quantization.cpp`, `tests/tolerances/omnivoice.json`,
the porting log

Plan 4's Slice A ruling ("Quantization is codec-only," 2026-08-06) rested
explicitly on token identity: "a reference port measured token agreement
collapsing from 100% to ~7% with F16 generator weights, and this family's
headline claim is exact tokens." That headline claim no longer governs the
CUDA path. The reference port ships Q8_0 for the generator and makes no
fidelity claim beyond "smoke surfaces" — this task is where that gets
measured on this port rather than assumed from theirs.

- [ ] **Step 1:** Quantize the generator to Q8_0 (the reference's own
  choice), following Task 1/2 of Plan 4's own classifier-and-runtime-branch
  pattern but pointed at the generator's tensors instead of the codec's.
- [ ] **Step 2:** Run Task 3's replayed-grid waveform comparison, not the
  retired exact-token gate. Measure token agreement as observed data (it
  will almost certainly not be 100%) and waveform cosine/max_abs against the
  F32 reference.
- [ ] **Step 3:** A Listening Audit pass, offered to jiangzhuo per the
  standing rule (a tolerance grid is not audible evidence) — this cannot be
  decided by the numbers alone, the same way Task 1's own placement decision
  could not.
- [ ] **Step 4:** Record the decision either way, with the evidence. Commit.

### Task 9: Reconsider step-count reduction, judged by ear

**Files:** `src/arch/omnivoice/weights.cpp` or wherever the generation
default lives, `tests/tolerances/omnivoice.json`, the porting log

The reference implementation's own CPU example silently runs 8 denoising
steps against this port's 32-step default and never justifies the
difference in its own documentation. A 4× step reduction is a direct,
linear RTF lever for the held-on-CPU generator's own compute (independent of
GPU placement) and was previously out of scope only because fewer steps
changes the committed token grid, which the old bar forbade.

- [ ] **Step 1:** Measure RTF at 8 steps (and any intermediate value worth
  recording, e.g. 16) on CPU and on the Task 1 GPU path, same case
  (`omni-long-boundary`) as Task 2's own baseline, for direct comparability.
- [ ] **Step 2:** Task 3's replayed-grid waveform comparison at each step
  count against the 32-step F32 reference.
- [ ] **Step 3:** A Listening Audit pass for step count specifically —
  fewer steps is a different kind of change from backend placement (it
  changes the ALGORITHM's own iteration count, not its arithmetic), and
  deserves its own audit rather than reuse of Task 1's or Task 8's.
- [ ] **Step 4:** Record the decision, with the evidence, including whether
  the default changes or only becomes a documented, non-default option.
  Commit.

---

## Task 10: Close-out

**Files:** `docs/porting/families/omnivoice.md`,
`reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`, `docs/testing.md`,
a Plan 6 carry-over ledger

- [ ] **Step 1:** Family doc: final backend claim (scope, RTF, the principled
  exception), the two Listening Audit entries, quantization/step-count
  outcomes from Tasks 8–9.
- [ ] **Step 2:** Porting log: one dated Plan 5 close-out section with every
  task's numbers gathered in one place.
- [ ] **Step 3:** `docs/testing.md`: the new/changed gates and their
  runtimes; Status date.
- [ ] **Step 4:** Final gates, all quoted with exit codes: both unit trees;
  the full integration set on the CPU tree; the CUDA tree's gates as
  redesigned by Task 4; card `--check`; `clang-format --check-diff`.
- [ ] **Step 5:** Carry-over ledger for whatever remains — at minimum the
  four project-wide questions and the float64 cosine estimator question this
  plan inherited and did not resolve, plus anything Tasks 3–9 opened and
  deferred. Commit.

---

## Self-Review Notes

- **What this session executes:** Tasks 1 and 2 only, per the controlling
  instruction. Tasks 3–10 are written to the same level of detail Plan 4
  used for slices it had not yet executed (its own Task 4 was written and
  then closed "N/A" after measurement showed the option didn't survive
  contact with evidence) — a real plan for later work, not a placeholder.
- **The two STOP-shaped moments this plan is honest about:** Task 1
  deliberately breaks `synthesize-omnivoice-replay-golden-cuda`'s exact-token
  assertion, and says so in Global Constraints rather than leaving a future
  reader to discover a red CI gate and wonder whether Task 1 introduced a
  bug. Task 3 Step 3 flags a real public-contract ambiguity (does
  `SYNTH_BACKEND_CUDA`'s meaning change, or does a new identifier appear)
  that this plan declines to resolve by silent inference.
- **Why Task 2 is not a gate:** measuring RTF honestly does not require
  deciding what "Support" means for the new shape first — that would invert
  the dependency (Task 3/4 need a real number to design tolerances around,
  and Task 2 needs no tolerance design to report a wall-clock time). Keeping
  it a plain, dated measurement in the porting log avoids the trap of
  smuggling a claim in ahead of its own evidence.
- **Why the generator twin duplicates Task 9's pattern instead of factoring
  out a shared helper:** Task 9's twin-construction code (the tensor-copy
  loop, the buffer allocation, the post-stream copy) is already unit- and
  integration-tested and currently shipping. A shared-helper refactor would
  touch that working code purely for DRY's sake while adding a second,
  larger (2,336.8 MiB vs 84.24 MiB) twin at the same time — two risks in one
  change. Task 1 duplicates the pattern; a future cleanup task can factor it
  once both twins have their own settled test coverage, with no schedule
  pressure from this plan's own goal.
- **Spec coverage:** the decision record's five "What Plan 5 must build"
  items map: 1 → Task 1; 2 → Task 5; 3 → Tasks 3–4; 4 → Tasks 2 and 7; 5 →
  Tasks 8–9. The prior-art correction it separately owes → Task 6.
