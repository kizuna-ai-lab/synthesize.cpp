# Qwen3-TTS Stage 3 Plan 3 — Quantization, Backends, Ship Prep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure what the 1.7B VoiceDesign package costs in every Quantization Profile, measure whether CUDA pays for it, audit it by ear in two halves labelled at different strengths, and prepare — not upload — the artifacts a publication would need.

**Architecture:** Almost nothing here is new code. The quantizer, the three validators, the tolerance schema, the card generator and the Release measurement preset all exist and were exercised on the Base variant by Stage 2 Plan 4. This plan cuts three profiles from a package that today has only its BF16 source, fills the tolerance cells each cut earns, measures latency and memory on a `Release`-typed tree, decides CUDA on a measurement rather than an analogy, and writes the card. One genuinely new piece of code: the quantizer cannot currently be *asked* what role a tensor resolves to, so the assertion section 7 requires cannot be written today.

**Tech Stack:** C++17, GGML/GGUF, `synthesize-quantize`, CMake presets, Python 3.12 in the locked `scripts/envs/qwen3-tts` uv environment, Jinja2 card generation.

## Global Constraints

- **Every performance figure comes from a `Release`-typed build.** Use `rel-dgx-spark` (`CMakePresets.json:73-81`, `CMAKE_BUILD_TYPE=Release`). Never `dev-dgx-spark` or any other `development-base` preset: they inherit `RelWithDebInfo`, which compiles ggml-cpu at `-O2` and measured **2.19× slower** than `-O3` on this workload. A figure whose build is not named is not a figure.
- **No frame count may be pinned or quoted without naming its build.** Release and RelWithDebInfo produce different generated-frame counts for the same run (24,960 vs 48,000 observed elsewhere in this family).
- **A profile that fails to pay is a result** — but *what counts as paying differs by profile, and this constraint got it wrong for F16 until 2026-08-20.* `docs/quantization.md:449-453` records that **F16 is a SPEED profile for this family, not a size one**: Base's F16 came out larger than its source and **was published anyway**. So a larger-than-source F16 settles nothing about publication, and Task 2 — which is forbidden to measure timing — cannot decide it. **Task 5 decides F16.** The genuine not-paying precedent is CustomVoice's Q5_K_MIXED: buildable, measured, deliberately unpublished, and withheld on **accuracy** (talker-logits cosine 0.9648), not size. Q5_K_MIXED is therefore judged on the shrink-versus-agreement trade in Task 4, and F16 on speed in Task 5. Where a profile does fail its own test, the deliverable is the recorded measurement and a profile absent from the card, named as measured-and-not-shipped rather than omitted.
- **If the listening audit finds that descriptions do not control the voice in any recognizable way, the honest output is a recorded negative, not a published package.** Design spec section 7.
- **Publication is out of scope.** This plan prepares artifacts. Uploading them is a separate act requiring jiangzhuo's confirmation at the time, naming the target repository. Do not upload, do not `hf upload`, do not open a PR against any model repo.
- **`quality_evaluation` stays `not_run`.** Deferred per ADR 0017. A listening audit does not move the Validation Level; the card carries the two as separate fields and says why.
- Focused unit tests are committed, passing, and registered with CTest under the `unit` label before a task is done.
- Sanitizer gate for any C/C++ change: `cmake -S . -B build-sanitize -DSYNTH_BUILD_TESTS=ON -DSYNTH_BUILD_PYTHON_TESTS=ON -DSYNTH_BUILD_INTEGRATION_TESTS=OFF -DSYNTH_SANITIZE=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo` then `cmake --build build-sanitize --target synthesize-check-unit`.
- `scripts/ci/clang-format.sh --fix` after `git add`, then `--check-diff` clean. Never format `ggml/`.
- Exactly two unit tests fail on a tree that never materialized a VITS model and belong to no task here: `synthesize-python-api-wheel-test`, and `synthesize-vits-python-unit` (sole error `test_quantization_reports_match_current_artifacts`). A third failure, or a different failure inside that target, belongs to the task that caused it.
- **Nothing under `models/` or `build/` is ever committed.** No `.gguf`, `.wav`, `.f32`, `.i32`, `.safetensors` or oracle dump enters a commit. Check `git status --porcelain` and stage by name — this tree has been swept by a stray `git add -A` before.
- `ggml/` is a submodule checkout and must never be edited.
- A changed `docs/` file carrying a `Status:` line must have that line updated in the same commit.
- The quantizer is **not built by default**: `SYNTH_BUILD_TOOLS` defaults to `OFF` (`CMakeLists.txt:103`). Configure with `-DSYNTH_BUILD_TOOLS=ON` or the binary will not exist.

## Starting State

- `models/qwen3-tts-12hz-1-7b-voicedesign/qwen3-tts-12hz-1-7b-voicedesign-BF16.gguf` — 4,295,891,904 bytes, the only file in that directory. No F16, Q8_MIXED or Q5_K_MIXED cut exists.
- `tests/tolerances/qwen3-tts.json`'s `qwen3-tts-12hz-1-7b-voicedesign` has one profile (`BF16`) with `stages: {replay, public}` and `backends: {}`.
- No `scripts/hf_cards/qwen3-tts-12hz-1-7b-voicedesign.yaml`. No `docs/models/qwen3-tts-12hz-1-7b-voicedesign.md`.
- `docs/quantization.md` has a "Qwen3-TTS Profiles" section written for Base and CustomVoice.

## What This Variant Does NOT Have

Established by Stage 3 Plan 1 and confirmed before this plan was written. Do not import Base's Plan 4 procedures by analogy:

- **No speaker encoder.** So no `ConvKernel` tensors, no ECAPA graph, no x-vector path. Task 1 asserts this positively instead of leaving it true by accident.
- **No codec encoder.** So **no `codec_encoder` stage** — only `public` and `replay` apply. Plan 4's percentile / semantic-flip-rate statistic redesign (its Tasks 2-3) gated the codec encoder and has nothing to gate here.
- **No new graphs.** Stage 3 Plan 1's architecture note: this is "Stage 1's graph set at doubled dimensions." Plan 4 had to decide CUDA placement for two graphs that did not previously exist; Task 6 here has no such decision, only a measurement of the existing Stage 1 placement at the larger width.

## File Structure

| File | Responsibility | Task |
| --- | --- | --- |
| `tools/synthesize-quantize/policy.h` | Gains one narrow role query so the ConvKernel proposition is testable | 1 |
| `tools/synthesize-quantize/policy.cpp` | Implements it over the existing classifier | 1 |
| `tests/qwen3_tts_quantization_policy_test.cpp` | Asserts a no-speaker-encoder package resolves no ConvKernel | 1 |
| `tests/tolerances/qwen3-tts.json` | Gains F16, Q8_MIXED, Q5_K_MIXED profiles and a CUDA sub-grid or a recorded absence | 2-4, 6 |
| `docs/porting/families/qwen3-tts.md` | The family record: every measurement, the audit table, the negative results | 2-8 |
| `docs/backends.md` | VoiceDesign's CUDA outcome, in the shape Plan 4's Task 12/13 established | 6 |
| `docs/quantization.md` | A VoiceDesign subsection; also one stale line to correct | 4, 8 |
| `scripts/hf_cards/qwen3-tts-12hz-1-7b-voicedesign.yaml` | The card spec | 8 |
| `docs/models/qwen3-tts-12hz-1-7b-voicedesign.md` | The model page | 8 |

---

### Task 1: Ask the quantizer a question it cannot currently answer

Design spec section 7 requires: *"Asserted positively in the policy test — a package with no speaker encoder resolves no `ConvKernel` — rather than left to be true by accident."* That assertion cannot be written today, and the reason matters.

`classify_qwen3_tts_tensor` (`tools/synthesize-quantize/policy.cpp:617`) lives in an anonymous namespace, and `policy.h` exposes only `resolve_qwen3_tts_target_spec`, which returns a `TargetSpec{type, layout}` — the *outcome*, not the role. Asserting on the outcome instead is not equivalent and is silently weaker: under `F16` the ConvKernel column (`profile.transpose_weight_type`) and the MatrixWeight column are **both** `GGML_TYPE_F16`, so a type-only check cannot tell them apart at all. It would pass on a package whose tensors really had been misclassified.

Expose exactly the proposition being asserted — not the whole taxonomy.

**Files:**
- Modify: `tools/synthesize-quantize/policy.h`
- Modify: `tools/synthesize-quantize/policy.cpp:617` (move the classifier's *use*, not the classifier)
- Test: `tests/qwen3_tts_quantization_policy_test.cpp`

**Interfaces:**
- Produces: `bool synth::quantize::qwen3_tts_tensor_is_conv_kernel(const std::string & name);` — true when the named tensor resolves to the ConvKernel role under the qwen3-tts classifier, for any profile. No later task consumes it; it exists for the test.

- [ ] **Step 1: Write the failing test**

Add to `tests/qwen3_tts_quantization_policy_test.cpp`, inside `main()` alongside the existing name-classification blocks:

```cpp
    // Design spec section 7: "a package with no speaker encoder resolves no
    // ConvKernel". Asserted on the ROLE, not on the resolved type: under F16
    // the ConvKernel column (profile.transpose_weight_type) and the
    // MatrixWeight column are both GGML_TYPE_F16, so a type-only check cannot
    // tell a misclassified tensor from a correct one and would pass on the
    // bug it exists to catch.
    //
    // The names below are the shape a VoiceDesign package actually carries --
    // talker, code predictor and codec decoder, no speaker_encoder.* prefix
    // anywhere -- which is also CustomVoice's shape. Both are real packages
    // with no speaker encoder; Base is the only variant that has one.
    for (const char * name : {
             "talker.blocks.0.attn.q.weight",
             "talker.blocks.0.mlp.gate.weight",
             "talker.output.weight",
             "code_predictor.blocks.0.attn.k.weight",
             "code_predictor.output.weight",
             "codec.decoder.blocks.0.conv.weight",
             "codec.decoder.head.weight",
         }) {
        if (synth::quantize::qwen3_tts_tensor_is_conv_kernel(name)) {
            std::fprintf(stderr, "no-speaker-encoder shape resolved ConvKernel for %s\n", name);
            return 1;
        }
    }

    // The control. Without it the loop above passes on a build where the
    // function always returns false, which is the same failure the loop is
    // written to catch, spelled the other way.
    if (!synth::quantize::qwen3_tts_tensor_is_conv_kernel("speaker_encoder.tdnn1.weight")) {
        std::fprintf(stderr, "a real speaker-encoder kernel did not resolve ConvKernel\n");
        return 1;
    }
```

- [ ] **Step 2: Run it to verify it fails to compile**

```bash
cmake -S . -B build -DSYNTH_BUILD_TESTS=ON -DSYNTH_BUILD_PYTHON_TESTS=ON \
  -DSYNTH_BUILD_INTEGRATION_TESTS=OFF -DSYNTH_BUILD_TOOLS=ON
cmake --build build --target synthesize-qwen3-tts-quantization-policy-test
```
Expected: compile error — `qwen3_tts_tensor_is_conv_kernel` is not declared.

- [ ] **Step 3: Declare it**

In `tools/synthesize-quantize/policy.h`, beside the existing `resolve_qwen3_tts_target_spec` declaration:

```cpp
// True when `name` resolves to the ConvKernel role under the qwen3-tts
// classifier. Exposed for one proposition the resolved TargetSpec cannot
// express: ConvKernel and MatrixWeight resolve to the SAME type under F16
// (both take profile.transpose_weight_type there), so a test asserting on
// type alone is blind exactly where a misclassification would hide. The role
// is the thing section 7 of the Stage 3 design asks to be asserted, so the
// role is what this returns.
bool qwen3_tts_tensor_is_conv_kernel(const std::string & name);
```

- [ ] **Step 4: Implement it**

In `tools/synthesize-quantize/policy.cpp`, immediately after the anonymous namespace closes at `:641`:

```cpp
bool qwen3_tts_tensor_is_conv_kernel(const std::string & name) {
    return classify_qwen3_tts_tensor(name) == CatalogRole::ConvKernel;
}
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
cmake --build build --target synthesize-qwen3-tts-quantization-policy-test
ctest --test-dir build --output-on-failure -R '^synthesize-qwen3-tts-quantization-policy-test$'
```
Expected: PASS.

- [ ] **Step 6: Prove the assertion can fail**

Temporarily change the implementation to `return false;`, rebuild, rerun. Expected: the control fails with `a real speaker-encoder kernel did not resolve ConvKernel`. Then change it to `return true;`, rebuild, rerun. Expected: the loop fails on `talker.blocks.0.attn.q.weight`. Revert to the real body, rebuild, confirm green, and confirm `git diff tools/synthesize-quantize/policy.cpp` is empty against what you wrote in Step 4.

- [ ] **Step 7: Sanitizer gate and format**

```bash
cmake --build build-sanitize --target synthesize-check-unit
git add tools/synthesize-quantize/policy.h tools/synthesize-quantize/policy.cpp tests/qwen3_tts_quantization_policy_test.cpp
scripts/ci/clang-format.sh --fix
scripts/ci/clang-format.sh --check-diff
```

- [ ] **Step 8: Commit**

```bash
git commit -m "qwen3-tts: ask the quantizer for a role, not a type, about ConvKernel"
```

---

### Task 2: Cut F16, and record whether it paid

The design spec expects F16 not to pay, *with a weaker argument than for Base*: on Base it came out 184,448 bytes **larger** than its source, because both are two-byte types while the sensitive tensors widen BF16 to F32. Here the matrix half is 3.2× larger against a sensitive half that is not, so the penalty is relatively smaller. **The direction is likely unchanged and the magnitude is not.** Measure it.

**Files:**
- Modify: `tests/tolerances/qwen3-tts.json` (add the `F16` profile under `qwen3-tts-12hz-1-7b-voicedesign`)
- Modify: `docs/porting/families/qwen3-tts.md`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `models/qwen3-tts-12hz-1-7b-voicedesign/qwen3-tts-12hz-1-7b-voicedesign-F16.gguf` on disk (uncommitted), and a `profiles.F16` object with `stages.public` and `stages.replay` cells that Tasks 5-7 read.

- [ ] **Step 1: Cut it**

```bash
build/bin/synthesize-quantize \
  models/qwen3-tts-12hz-1-7b-voicedesign/qwen3-tts-12hz-1-7b-voicedesign-BF16.gguf \
  models/qwen3-tts-12hz-1-7b-voicedesign/qwen3-tts-12hz-1-7b-voicedesign-F16.gguf \
  --quant F16
```

- [ ] **Step 2: Record the size delta, signed**

```bash
stat -c '%n %s' models/qwen3-tts-12hz-1-7b-voicedesign/qwen3-tts-12hz-1-7b-voicedesign-{BF16,F16}.gguf
```
BF16 is 4,295,891,904 bytes. Write the F16 figure and the **signed** difference into your report. A positive difference means F16 is larger than its source, which is a result, not a failure — Base's was +184,448.

- [ ] **Step 3: Confirm it loads at all**

```bash
build/bin/synthesize-qwen3-tts-public-real \
  models/qwen3-tts-12hz-1-7b-voicedesign/qwen3-tts-12hz-1-7b-voicedesign-F16.gguf \
  probe:description en
```
Expected: loads, reports zero Preset Voices. A load failure here is a Task 2 defect, not a later one — stop and report rather than proceeding to fill cells for a package that does not load.

- [ ] **Step 4: Fill the `public` cell**

```bash
uv run --project scripts/envs/qwen3-tts --locked python scripts/validate-qwen3-tts-public.py \
  --model models/qwen3-tts-12hz-1-7b-voicedesign/qwen3-tts-12hz-1-7b-voicedesign-F16.gguf \
  --runner build/bin/synthesize-qwen3-tts-public-real \
  --description "A cheerful, bright female voice speaking with fast pacing and high energy." \
  --description "A deep, calm male voice speaking slowly and quietly." \
  --profile F16 --backend cpu \
  --report reports/validate/qwen3-tts/public-voicedesign-F16.json
```
`--description` twice is required, not optional — relation 1 compares two different descriptions against each other, and one is a usage error rather than a shorter run.

- [ ] **Step 5: Fill the `replay` cell**

```bash
**This command does not work for this variant, and did not work for BF16 either.**
Task 2 established that `validate-qwen3-tts-replay.py --stage replay` fails against
every profile of `qwen3-tts-12hz-1-7b-voicedesign` because no oracle case artifacts
exist for it. Use the method Task 2 used and recorded — direct invocation of the
prefill driver, matching however the committed BF16 cell was produced. Read
`.superpowers/sdd/2026-08-20-qwen3-tts-stage-3-plan-3-quantization-backends-ship/task-2-report.md`
for the exact commands rather than reconstructing them, and record the cell as the
prefill probe it is rather than as a full replay.
```

- [ ] **Step 6: Verify the tolerance file still validates and its counts agree**

```bash
cmake --build build --target synthesize-check-unit
ctest --test-dir build --output-on-failure -R 'tolerance|golden-manifest'
```
Expected: pass. `tests/python/test_tolerance_coverage.py` asserts every `stages.*` key has a validator behind it, and `test_tolerance_case_count_matches_manifest` asserts the recorded `case_count` matches the manifest's 13 — if you added a cell with a stale count, this is where it fails.

- [ ] **Step 7: Write the family record entry**

Append to `docs/porting/families/qwen3-tts.md` a subsection under the Stage 3 heading, naming the build used for every figure, and stating plainly whether F16 paid. If it did not, say so in those words and say what that means for the card: it is measured and absent, not omitted.

Update that file's `Status:` line to today's date.

- [ ] **Step 8: Commit**

```bash
git add tests/tolerances/qwen3-tts.json docs/porting/families/qwen3-tts.md
git status --porcelain   # confirm no .gguf staged
git commit -m "qwen3-tts: cut voicedesign/F16 and record whether two-byte-to-two-byte paid at 1.7B"
```

---

### Task 3: Cut Q8_MIXED, and fill its cells

**Files:**
- Modify: `tests/tolerances/qwen3-tts.json`
- Modify: `docs/porting/families/qwen3-tts.md`

**Interfaces:**
- Consumes: nothing from Task 2 beyond the pattern.
- Produces: `profiles.Q8_MIXED` with `stages.public` and `stages.replay`, read by Tasks 5-7.

- [ ] **Step 1: Cut it**

```bash
build/bin/synthesize-quantize \
  models/qwen3-tts-12hz-1-7b-voicedesign/qwen3-tts-12hz-1-7b-voicedesign-BF16.gguf \
  models/qwen3-tts-12hz-1-7b-voicedesign/qwen3-tts-12hz-1-7b-voicedesign-Q8_MIXED.gguf \
  --quant Q8_MIXED
```

- [ ] **Step 2: Record the size**

```bash
stat -c '%n %s' models/qwen3-tts-12hz-1-7b-voicedesign/qwen3-tts-12hz-1-7b-voicedesign-Q8_MIXED.gguf
```
For scale: Base went 2,516,522,624 → 1,667,606,112, a ratio of 0.663. Record this variant's own ratio; do not assume it matches.

- [ ] **Step 3: Confirm it loads**

```bash
build/bin/synthesize-qwen3-tts-public-real \
  models/qwen3-tts-12hz-1-7b-voicedesign/qwen3-tts-12hz-1-7b-voicedesign-Q8_MIXED.gguf \
  probe:description en
```

- [ ] **Step 4: Fill the `public` cell**

```bash
uv run --project scripts/envs/qwen3-tts --locked python scripts/validate-qwen3-tts-public.py \
  --model models/qwen3-tts-12hz-1-7b-voicedesign/qwen3-tts-12hz-1-7b-voicedesign-Q8_MIXED.gguf \
  --runner build/bin/synthesize-qwen3-tts-public-real \
  --description "A cheerful, bright female voice speaking with fast pacing and high energy." \
  --description "A deep, calm male voice speaking slowly and quietly." \
  --profile Q8_MIXED --backend cpu \
  --report reports/validate/qwen3-tts/public-voicedesign-Q8_MIXED.json
```

- [ ] **Step 5: Fill the `replay` cell**

```bash
**This command does not work for this variant, and did not work for BF16 either.**
Task 2 established that `validate-qwen3-tts-replay.py --stage replay` fails against
every profile of `qwen3-tts-12hz-1-7b-voicedesign` because no oracle case artifacts
exist for it. Use the method Task 2 used and recorded — direct invocation of the
prefill driver, matching however the committed BF16 cell was produced. Read
`.superpowers/sdd/2026-08-20-qwen3-tts-stage-3-plan-3-quantization-backends-ship/task-2-report.md`
for the exact commands rather than reconstructing them, and record the cell as the
prefill probe it is rather than as a full replay.
```

- [ ] **Step 6: Verify**

```bash
cmake --build build --target synthesize-check-unit
```

- [ ] **Step 7: Record and commit**

```bash
git add tests/tolerances/qwen3-tts.json docs/porting/families/qwen3-tts.md
git status --porcelain
git commit -m "qwen3-tts: cut voicedesign/Q8_MIXED and fill its two CPU cells"
```

---

### Task 4: Q5_K_MIXED — the first real evaluation this profile has had

The design spec's own words: *"At Base's 2.4 GB the extra shrink over Q8_MIXED did not change the recommendation; at an estimated 4.3 GB it may."* Rows of 2048 and 6144 both divide by Q5_K's 256-element super-block, so there is no structural obstacle.

The standing precedent for the outcome is CustomVoice's: cut, measured, talker-logits cosine 0.9648, **deliberately unpublished**. That is a legitimate result of this task. So is publishing it. What is not legitimate is deciding before measuring.

**Files:**
- Modify: `tests/tolerances/qwen3-tts.json`
- Modify: `docs/quantization.md`
- Modify: `docs/porting/families/qwen3-tts.md`

**Interfaces:**
- Produces: `profiles.Q5_K_MIXED` with `stages.public` and `stages.replay`; and a recommendation, in words, that Task 8's card either honors or contradicts explicitly.

- [ ] **Step 1: Cut it**

```bash
build/bin/synthesize-quantize \
  models/qwen3-tts-12hz-1-7b-voicedesign/qwen3-tts-12hz-1-7b-voicedesign-BF16.gguf \
  models/qwen3-tts-12hz-1-7b-voicedesign/qwen3-tts-12hz-1-7b-voicedesign-Q5_K_MIXED.gguf \
  --quant Q5_K_MIXED
```

- [ ] **Step 2: Confirm the runtime accepts the profile string**

```bash
build/bin/synthesize-qwen3-tts-public-real \
  models/qwen3-tts-12hz-1-7b-voicedesign/qwen3-tts-12hz-1-7b-voicedesign-Q5_K_MIXED.gguf \
  probe:description en
```
`src/arch/qwen3-tts/weights.cpp`'s `read_quantization` accepts `Q5_K_MIXED` and `catalog.cpp`'s `expected_type` folds `Q5KMixed` into the shared F16/Q8Mixed/Q5KMixed branch, so this should load. If it does not, the failure is a real contradiction between the tool and the runtime and must be reported rather than worked around.

- [ ] **Step 3: Record size against Q8_MIXED, not only against BF16**

```bash
stat -c '%n %s' models/qwen3-tts-12hz-1-7b-voicedesign/qwen3-tts-12hz-1-7b-voicedesign-{BF16,Q8_MIXED,Q5_K_MIXED}.gguf
```
The question this task exists to answer is *the extra shrink over Q8_MIXED*, so that is the number to lead with.

- [ ] **Step 4: Fill both cells**

```bash
uv run --project scripts/envs/qwen3-tts --locked python scripts/validate-qwen3-tts-public.py \
  --model models/qwen3-tts-12hz-1-7b-voicedesign/qwen3-tts-12hz-1-7b-voicedesign-Q5_K_MIXED.gguf \
  --runner build/bin/synthesize-qwen3-tts-public-real \
  --description "A cheerful, bright female voice speaking with fast pacing and high energy." \
  --description "A deep, calm male voice speaking slowly and quietly." \
  --profile Q5_K_MIXED --backend cpu \
  --report reports/validate/qwen3-tts/public-voicedesign-Q5_K_MIXED.json

**This command does not work for this variant, and did not work for BF16 either.**
Task 2 established that `validate-qwen3-tts-replay.py --stage replay` fails against
every profile of `qwen3-tts-12hz-1-7b-voicedesign` because no oracle case artifacts
exist for it. Use the method Task 2 used and recorded — direct invocation of the
prefill driver, matching however the committed BF16 cell was produced. Read
`.superpowers/sdd/2026-08-20-qwen3-tts-stage-3-plan-3-quantization-backends-ship/task-2-report.md`
for the exact commands rather than reconstructing them, and record the cell as the
prefill probe it is rather than as a full replay.
```

- [ ] **Step 5: Write the recommendation, with its reasoning visible**

Add a `Q5_K_MIXED` paragraph to `docs/quantization.md`'s Qwen3-TTS section stating the size, the extra shrink over Q8_MIXED, the replay-stage agreement, and the recommendation. State the reasoning, not only the verdict: at 4 GB a further ~25% shrink means something different than it did at 2.4 GB, and the reader should be able to check the arithmetic.

Update `docs/quantization.md`'s `Status:` line.

- [ ] **Step 6: Commit**

```bash
git add tests/tolerances/qwen3-tts.json docs/quantization.md docs/porting/families/qwen3-tts.md
git status --porcelain
git commit -m "qwen3-tts: evaluate Q5_K_MIXED for real, at the size where it might matter"
```

---

### Task 5: Latency, RTF and peak memory, on a Release tree

**Files:**
- Modify: `docs/porting/families/qwen3-tts.md`

**Interfaces:**
- Consumes: the three cut packages from Tasks 2-4.
- Produces: a table Task 8's card copies figures from. Every row carries its build name.

- [ ] **Step 1: Configure and build the measurement tree**

```bash
cmake --preset rel-dgx-spark
cmake --build --preset rel-dgx-spark
```
Confirm before measuring anything:
```bash
grep CMAKE_BUILD_TYPE build/rel-dgx-spark/CMakeCache.txt
```
Expected: `CMAKE_BUILD_TYPE:STRING=Release`. If it says `RelWithDebInfo` you are on the wrong preset and every number you take is wrong by roughly 2.19×.

- [ ] **Step 2: Measure each profile on a fixed workload**

Fix the seed, fix the thread count at 10, take medians of at least 3 runs per profile, and record: frames, audio length, synthesis time, RTF, load time, peak RSS. Discard any run contended by a concurrent build — do not average it in. Plan 4 discarded exactly such runs and said so.

- [ ] **Step 3: Record the two known contaminants**

Both carry into every figure and both must appear beside the table, not in a footnote someone can miss: the second BPE frontend costs about +45 MB peak RSS, and generated frame counts are build-dependent, so no frame count may be quoted without naming its build.

- [ ] **Step 4: Decide F16, which Task 2 could not**

Task 2 measured F16 at +283,136 bytes — **larger** than its BF16 source — and correctly declined to draw a publication conclusion from that. `docs/quantization.md:449-453` records that **F16 is a speed profile for this family rather than a size one**, and Base's F16 was published despite also coming out larger than its source. So the question F16 actually has to answer is the one only this task can measure: **is it faster than BF16 on this variant, by enough to justify shipping a profile that saves no disk?**

Answer it in those terms. If F16 is not measurably faster here, it fails its own test and belongs in the card as measured-and-not-shipped — and that is a different verdict from Task 2's size figure, reached on different evidence. Say which evidence carried it.

- [ ] **Step 5: Sanity-check the shape against Base**

Base measured RTF 3.15 on CPU at 0.6B. This variant's talker is 3.2× larger. An RTF that comes out *better* than Base's is not automatically wrong, but it is surprising enough to re-measure before recording. Say in the report which way it went.

- [ ] **Step 6: Commit**

```bash
git add docs/porting/families/qwen3-tts.md
git commit -m "qwen3-tts: voicedesign latency, RTF and peak memory, measured on Release"
```

---

### Task 6: CUDA — a measurement, not an analogy

The design spec's expectation, and the reasoning behind it: Stage 1's placement puts the codec decoder on the device and keeps the autoregressive half on the CPU by the discrete-outputs rule. Because the talker is 3.2× larger, the AR half's share of wall clock **grows**, so **CUDA's end-to-end gain is expected to be smaller than Base's 6%, not larger.**

Note the direction of that claim: a *smaller* gain would confirm the model, not undermine it. Record whichever way it goes, and say which.

**Files:**
- Modify: `tests/tolerances/qwen3-tts.json` (a `backends.CUDA` sub-grid, or a recorded absence)
- Modify: `docs/backends.md`
- Modify: `docs/porting/families/qwen3-tts.md`

**Interfaces:**
- Consumes: the Release tree from Task 5.
- Produces: either CUDA cells or a documented, measured absence in the shape Plan 4's Task 12 established for Base.

- [ ] **Step 1: Confirm what CUDA even reaches here**

Unlike Base, this variant introduces no new graphs — it is Stage 1's graph set at doubled dimensions. So there is no placement *decision* to make, only a measurement of the existing one. Establish and state which stages are structurally measurable under CUDA and which are skipped, before taking any number.

- [ ] **Step 2: Measure end to end on the same fixed workload as Task 5**

```bash
uv run --project scripts/envs/qwen3-tts --locked python scripts/validate-qwen3-tts-public.py \
  --model models/qwen3-tts-12hz-1-7b-voicedesign/qwen3-tts-12hz-1-7b-voicedesign-BF16.gguf \
  --runner build/rel-dgx-spark/bin/synthesize-qwen3-tts-public-real \
  --description "A cheerful, bright female voice speaking with fast pacing and high energy." \
  --description "A deep, calm male voice speaking slowly and quietly." \
  --profile BF16 --backend cuda \
  --report reports/validate/qwen3-tts/public-voicedesign-BF16-cuda.json
```

- [ ] **Step 3: Compute the gain as a percentage, and show the division**

Base's record is `11.60 s → 10.85 s` and `RTF 3.15 → 2.95`, which divide out to 6.47% and 6.35% — recorded as "about 6%". That figure read "about 8%" until 2026-08-18 against its own numbers on the same line, and an external reviewer caught it. Write both the raw pair and the quotient so the same error cannot survive here.

- [ ] **Step 4: Derive any CUDA tolerance, do not scale the CPU one**

If a CUDA cell needs its own threshold, derive it from a CUDA-vs-CPU disagreement measurement at TF32 scale. Do **not** scale the CPU gate by a ratio — an earlier draft of Plan 4 made exactly that error and caught it in self-review. CUDA F32 matmuls run at TF32 and there is no build option to change it; expect ~1e-3 relative deviation rather than FP32's 1e-7.

- [ ] **Step 5: Record**

If CUDA does not pay for this variant, that is the Base outcome repeating and `docs/backends.md` already has the shape for saying so. Write it there with the measurement, and update that file's `Status:` line.

- [ ] **Step 6: Commit**

```bash
git add tests/tolerances/qwen3-tts.json docs/backends.md docs/porting/families/qwen3-tts.md
git commit -m "qwen3-tts: what CUDA buys the 1.7B variant, measured"
```

---

### Task 7: The listening audit, in two halves labelled at different strengths

This audit carries a question the family's previous three did not. Earlier audits asked whether a profile regressed, against a baseline. This one must also ask whether **the voice matches the description**, for which there is no ground truth. So it is two halves, and they are **not** recorded at the same strength:

- **Quantization regression** — blind A/B on natively-sampled clips, the established shape. Strong evidence.
- **Description control** — a *labelled* comparison of whether different descriptions produce recognizably different voices in the described direction. **Weak evidence, recorded as weak.**

Neither moves the Validation Level.

There is no tooling for this. All three prior audits were done by hand, with artifacts under `build/` (gitignored) and results recorded in two committed places. Follow that, and do not commit the artifacts.

**Files:**
- Create: `build/listening-voicedesign/` (uncommitted — artifacts only)
- Modify: `docs/porting/families/qwen3-tts.md`

**Interfaces:**
- Consumes: the three cut packages, and the Release tree from Task 5.
- Produces: a verdict recorded in the family record, which Task 8's card reads into `listening_audit` and `listening_audit_detail`.

- [ ] **Step 1: Generate the clips from the Release tree**

One sentence, one seed, generated from `build/rel-dgx-spark`. Cover BF16 against F16 and against Q8_MIXED (and Q5_K_MIXED if Task 4 recommended it), plus CPU against CUDA if Task 6 found CUDA reaches anything.

- [ ] **Step 2: Verify the pairs actually differ in bytes before anyone listens**

A blind pair whose two sides are byte-identical produces a confident "no difference" that means nothing. Check each pair and discard any that are identical, reporting which.

- [ ] **Step 3: Build the blind A/B page**

Shuffle A/B positions on a recorded seed and withhold the key from the page. Embed the audio so the page is self-contained. Write the shuffle seed and the key to a manifest beside it, not into the page.

- [ ] **Step 4: Build the labelled description-control comparison — separately**

This half is **not** blind and must not be presented as though it were. Same sentence, same seed, two or more descriptions differing in a stated direction (the two the tolerance cells already use are a reasonable pair: bright/fast/high-energy against deep/calm/slow). The question asked of the listener is whether the voices differ **in the described direction**, and the answer is weak evidence whichever way it goes.

- [ ] **Step 5: Ask jiangzhuo, and say what kind of evidence each half is**

Present both halves and ask for verdicts. Do not proceed to Task 8 on an assumed outcome, and do not describe the labelled half as though it carried the blind half's weight.

- [ ] **Step 6: Record the verdicts in the family record**

A table with `# | comparison | verdict`, the method line naming listeners, clips, seed and build, and the result as one of the established values. **If the description-control half finds that descriptions do not control the voice recognizably, record that as the negative it is** — and stop before Task 8's card claims otherwise. That outcome makes the honest deliverable a recorded negative rather than a published package, and it is jiangzhuo's call what happens next.

Update the `Status:` line.

- [ ] **Step 7: Commit**

```bash
git add docs/porting/families/qwen3-tts.md
git status --porcelain   # confirm no .wav and nothing from build/
git commit -m "qwen3-tts: the VoiceDesign listening audit, in two halves weighted differently"
```

---

### Task 8: Ship artifacts — prepared, not uploaded

**Files:**
- Create: `scripts/hf_cards/qwen3-tts-12hz-1-7b-voicedesign.yaml`
- Create: `docs/models/qwen3-tts-12hz-1-7b-voicedesign.md`
- Modify: `docs/quantization.md`
- Modify: `docs/porting/families/qwen3-tts.md`

**Interfaces:**
- Consumes: every measurement from Tasks 2-7.
- Produces: a rendered card and a model page. **No upload.**

- [ ] **Step 1: Write the card spec**

Model `scripts/hf_cards/qwen3-tts-12hz-0-6b-base.yaml`. Per profile: `filename`, `size`, `size_bytes`, `sha256`, `tensor_types`, and a `validation:` block. Carry the two audit fields separately and keep the comment explaining why they are separate:

```yaml
quality_evaluation: not_run
listening_audit: <the recorded value from Task 7>
listening_audit_detail:
  profiles: [...]
  listeners: 1
  method: >-
    blind A/B on natively-sampled clips for the quantization half, and a
    LABELLED comparison for the description-control half, which is weak
    evidence and is recorded as weak
  dates: [...]
```

Include only the profiles that earned a place. A profile measured and not recommended is **absent from `quants:` and named in the notes as measured-and-not-shipped** — silence would read as "not tried".

- [ ] **Step 2: Render it**

```bash
uv run --script scripts/hf_cards/generate.py scripts/hf_cards/qwen3-tts-12hz-1-7b-voicedesign.yaml --stdout
```

- [ ] **Step 3: Run the card test**

```bash
cmake --build build --target synthesize-check-unit
ctest --test-dir build --output-on-failure -R 'hf-card|hf_card'
```
`tests/python/test_hf_card_generator.py` validates card specs. A new card that the test does not cover is a gap — check whether it enumerates cards by glob or by name, and add the new one if by name.

- [ ] **Step 4: Write the model page**

`docs/models/qwen3-tts-12hz-1-7b-voicedesign.md`, modeled on `docs/models/qwen3-tts-12hz-0-6b-base.md`, with a `Status:` line dated today.

- [ ] **Step 5: Correct one stale line found while writing this plan**

`docs/quantization.md`'s Publication section reads *"No Base package is published"*. That is stale: the three Base GGUFs were published to `jiangzhuo9357/qwen3-tts-12hz-0-6b-base-gguf` on 2026-08-17, recorded in that card's own header and in `docs/models/qwen3-tts-12hz-0-6b-base.md`. Correct it and say when it became untrue.

- [ ] **Step 6: State what is prepared and what is not**

The final commit message and the family record must both say plainly: artifacts are prepared, nothing is uploaded, and publication is a separate act requiring jiangzhuo's confirmation naming the target repository.

- [ ] **Step 7: Commit**

```bash
git add scripts/hf_cards/qwen3-tts-12hz-1-7b-voicedesign.yaml \
        docs/models/qwen3-tts-12hz-1-7b-voicedesign.md \
        docs/quantization.md docs/porting/families/qwen3-tts.md
git status --porcelain
git commit -m "qwen3-tts: prepare the VoiceDesign card and model page, upload not included"
```

---

## Completion Gate

- The audit is recorded, both halves, at their stated strengths.
- The artifacts are prepared: three cuts measured, cells filled, card rendered, model page written.
- **Upload awaits separate confirmation** and has not happened.

Not delivered here, and not a gap: publication itself; any movement of `quality_evaluation`, which stays deferred per ADR 0017; multi-clip enrollment; Native Streaming Synthesis; Voice Conversion.

## Self-Review

**Spec coverage.** Section 7's five claims each have a task: ConvKernel unreachability → Task 1; Q5_K_MIXED's first real evaluation → Task 4; F16 expected not to pay, magnitude unknown → Task 2; CUDA gain expected smaller than 6% → Task 6; the two-halves audit → Task 7. The section 8 gate row → Task 8 plus the Completion Gate. The two standing rules are in Global Constraints and restated at the tasks where they bite (Tasks 4 and 7).

**Placeholder scan.** Every command is a real one against a real path. The one place I could not supply a literal is the fixed workload in Task 5 — Plan 4's own record does not pin the sentence it used, and inventing one here would be a fabricated citation. Task 5 specifies the properties instead (fixed seed, threads 10, medians of 3+, contended runs discarded) and requires the choice to be recorded.

**Type consistency.** `qwen3_tts_tensor_is_conv_kernel` is the only new symbol, declared in Task 1 Step 3 and used in Task 1 Step 1 and nowhere else.

**One thing worth flagging to the reader.** Task 6 may find CUDA reaches almost nothing on this variant, in which case its "measurement" is mostly a recorded absence — that is the Base outcome repeating, and `docs/backends.md` already has the shape for it. That is a legitimate task outcome, not a task failure.
