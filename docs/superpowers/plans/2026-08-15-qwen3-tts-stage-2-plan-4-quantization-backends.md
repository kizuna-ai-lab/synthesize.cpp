# Qwen3-TTS Stage 2, Plan 4: Quantization, Execution Backends and Ship Prep — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the Base package's Quantization Profiles and its Execution Backend placement, rather than inheriting either — fill the empty cells of `tests/tolerances/qwen3-tts.json`, decide the speaker encoder and the codec encoder on their own numbers, record every performance figure with the build that produced it, run the first ICL listening audit this family has ever had, and stage the ship artifacts. Publication is not part of it.

**This is a measurement plan, and "do not ship this profile" is one of its correct answers.** The design's §7 says so directly: *"the speaker encoder is small enough that quantizing it is unlikely to pay, and the codec encoder is convolution-heavy, which is the shape that produced the conv-exempt policy in another family. **Neither precedent is imported by analogy**; each is measured here and the result recorded with the artifact that produced it."* §7 names no target profile set, no size target and no speed target, and the project rule is that there is no minimum speedup (`docs/backends.md:499-501`). So a task that measures a profile into a hole and concludes it does not pay has completed, not failed. **Every measurement task below states the outcomes it can produce before it runs**, in the shape Plan 3's 9-frame adjudication used, and no task is written as though it knew which one it would get.

---

## What Plan 4 delivers

The design's plan-sequence row (`spec:532`): *"Quantization Profile measurement, CUDA backend, listening audit, model card and artifacts | audit recorded; artifacts prepared; upload awaits separate confirmation."*

Concretely:

- A quantizer that can cut a Base package at all — today it cannot (see "The blocker", below) — and a runtime whose expectations agree with it, with the disagreement resolved on evidence rather than by making one side match the other.
- Measured `F16` and `Q8_MIXED` cells for `qwen3-tts-12hz-0-6b-base` on CPU, or a recorded reason why a profile is not cut.
- The speaker encoder's and the codec encoder's own answers, each with the artifact that produced it.
- A redesigned semantic reconstruction statistic, because the one Plan 3 handed over fails a legitimate case and its failure is a property of the statistic (carry-over §1.3).
- CUDA placement for the two new graphs, or the measured decision not to place them, plus the tolerance cells and the performance and memory figures — each figure naming its build.
- The first ICL listening audit.
- Staged ship artifacts: a Hugging Face card specification, a `docs/models/` page, a `docs/quantization.md` section for this family, and the family-record and spec updates.

## What Plan 4 does NOT deliver

- **Publication.** `spec:532` — "upload awaits separate confirmation" — and `spec:561-562` lists "**Publication itself**, which is a separate act requiring confirmation at the time" among the non-goals. Nothing in this plan uploads anything or asks to. The artifacts are staged and stop.
- **Any movement of the Validation Level.** `port_validated` plus a listening audit is the delivery bar (`spec:563-564`); **`quality_evaluation` stays deferred per ADR 0017**. The audit Task 15 runs is evidence, not a level.
- **Stage 3 / Description Text**, the **1.7B Base variant**, **multi-clip enrollment**, **Native Streaming Synthesis** and **Voice Conversion** — all §10 non-goals (`spec:550-563`).
- **The CLI.** `examples/cli/` stays untouched, for the reasons recorded in the family document.
- **Any converter change or package re-cut**, unless a task proves one necessary and carries the byte-identity proof with it.

---

## The three things that shape this plan, stated before the tasks

### 1. A hard blocker sits in front of every measurement, and it was measured rather than read

`synthesize-quantize` **cannot cut a Base package at all**. `classify_qwen3_talker` requires `tokens[0] == "talker"` (`tools/synthesize-quantize/policy.cpp:305-307`) and `classify_qwen3_codec` requires `tokens[0] == "codec" && tokens[1] == "decoder"` (`:359-361`). The Base package's two new regions are `speaker_encoder.*` and `codec.encoder.*`. Neither matches; both classify `Unknown`; `Unknown` returns false from `resolve_qwen3_tts_target_spec` (`policy.cpp:511-512`) and `tools/synthesize-quantize/quantize.cpp:169-170` turns that into a hard failure.

Run on 2026-08-15 against the real 2,516,522,464-byte `models/qwen3-tts-12hz-0-6b-base/qwen3-tts-12hz-0-6b-base-BF16.gguf`:

```
$ build/bin/synthesize-quantize models/qwen3-tts-12hz-0-6b-base/qwen3-tts-12hz-0-6b-base-BF16.gguf \
    /tmp/plan4-probe-base-F16.gguf --quant F16
synthesize-quantize: unknown qwen3-tts tensor: speaker_encoder.asp.conv.bias
```

Exit 1, and the same for `--quant Q8_MIXED`. **No output file is produced.** (`build/` had `SYNTH_BUILD_TOOLS:BOOL=OFF`, so the tool had never been built in this worktree; reconfiguring with `-DSYNTH_BUILD_TOOLS=ON` is a prerequisite of Task 6 and not a repository change.)

**Worse: the runtime already has an opinion, and it contradicts the tool.** `src/arch/qwen3-tts/catalog.cpp:134` splits halves on `name.compare(0, 6, "codec.") == 0`. So `codec.encoder.*` lands in `Half::Codec` → F32 under every profile, which is consistent. But `speaker_encoder.*` lands in `Half::Talker`, and `Resolver::conv` passes `Role::Matrix` for the weight (`catalog.cpp:100-104`) — so under `F16` the runtime **expects 38 `speaker_encoder.*` convolution weights to be F16**, and under `Q8_MIXED` **Q8_0**, for a package the tool refuses to produce. That is precisely the drift the comment at `catalog.cpp:31-35` says the role/name split exists to prevent: *"the quantizer classifies by name and the two must not drift apart."*

**Task 6 has to close this, and it must say which of the two is right rather than making them agree by fiat.** Both directions are available and neither is obviously correct; the task states three outcomes and measures.

### 2. The gate Plan 4 inherits may not survive, and the plan says so up front

Plan 3 handed over a semantic p95 statistic sitting on a flip-rate cliff (carry-over §1.3, recorded in-tree at `tests/tolerances/qwen3-tts.json:670-685`):

- `base-ref-max` **fails** the semantic gate at **p95 0.3478 against the committed 2.0e-2**.
- **It is not a port defect**, on three independent readings: codes agree with upstream-f32 at **100.000% (0 of 6000)**; the chain gate passes *tighter* than the calibration case (**8.431e-05** against `base-icl-en`'s 9.303e-05); the acoustic branch passes at **0.2946**.
- The cliff: semantic flip rates are `base-ref-min` 0.00%, `base-icl-en` 3.96%, `base-text-short` 3.96%, `base-ref-max` **5.33%** (`:678`). A 95th percentile only enters the flipped tail above ~5% flips, so the statistic jumps **~85×**.
- **Widening was refused and the refusal is upheld**: a 0.35 threshold would leave the injected fault 4.6× above the gate instead of 80×, destroying discrimination. Recorded as `gate_scope_warning`; `cases` stays 3; the tree is green.

Plan 3's instruction: *"the statistic needs redesign — gate the flip rate separately from the reconstruction error on non-flipped frames."*

**The redesign has a hard constraint, and the plan states it as one.** Plan 3 measured the fault-injection discrimination of all three gates against upstream's own symmetric (non-causal) `MimiConv1d` padding split:

| gate | threshold | measured | headroom | injected fault | discrimination |
| --- | ---: | ---: | ---: | ---: | ---: |
| chain `rel_absmax` | 1.0e-3 | 9.303e-05 | 10.7× | 1.649 | **1650×** |
| semantic p95 | 2.0e-2 | 0.004103 | 4.9× | 1.601 | **80.05×** |
| acoustic p95 | 5.0e-1 | 0.2491 | 2.0× | 1.741 | **3.5×** |

(carry-over `:154-160`; the same figures are the `fault_injection_*` fields in the cell. The published "10.8×", "78×" and "87×" are arithmetic slips M1–M3 — the table above is correct.) A second, independent injection exists at `tests/qwen3_tts_codec_encoder_test.cpp:1438-1440`: *"a symmetric pad moves this to ~1.7 on both backends, 17,000x the CPU bound and 86x the accelerator one."*

**So a redesigned statistic must keep ≥80× discrimination on the semantic branch and must not lose the acoustic branch's already-narrow 3.5×** — and the acoustic branch's *real* headroom from WAV input is **1.64×**, not 2.0× (0.2491 → 0.304719, carry-over §3.3), which is the number the redesign actually has to live inside. **If the redesign cannot meet that, Task 3's third outcome applies and the plan stops to re-scope rather than shipping a weaker gate.**

**And the calibration data is thin.** Only four flip-rate points exist, and **all four come from one recording**: `base-ref-min` is a byte-exact 24,000-sample prefix of `base-icl-en`, and `base-text-short` is that same clip again at full length (`tests/tolerances/qwen3-tts.json:636`; carry-over §3.3, whose own conclusion is that *"a second reference clip and a second speaker are the highest-value coverage Plan 4 can add"*). That is why acquiring a second recording is **Task 1** and not a late addition: a flip-rate threshold calibrated on one speaker and then invalidated by Tasks 7, 8 or 13 would have to be re-derived after the cells it gates were already committed.

### 3. CUDA is expected to move the codes, which is why the statistic is redesigned first

`tests/qwen3_tts_codec_encoder_test.cpp:1416-1442` measured, on a GB10 (sm_121a), that **the same clip encoded on the two backends disagrees by 5.20e-03 relative** — larger than the 3.44e-03 between two CUDA runs, which is how the file establishes that the difference is arithmetic (reduced-mantissa tensor-core F32 with column-count tiling) rather than a broken prefix.

**5.20e-3 is ~1.33 bf16 units** (bf16 unit roundoff 2^-8 = 3.90625e-3, carry-over `:226-228`). Three consequences:

- It is **~56× the committed `codec.chain` gate of 1.0e-3**. A `codec_encoder` cell under `backends.CUDA` cannot reuse the CPU threshold — the chain gate would fail a correct CUDA run by construction.
- It is **bigger than the input rounding the WAV-input arm already spends** (2.954e-03 on the waveform, 0.76 of one bf16 unit, `tests/tolerances/qwen3-tts.json:668`), so it is not absorbable inside existing headroom.
- It is **directly a flip driver**: the RVQ argmin is a nearest-neighbour decision, and a 1.33-bf16-unit perturbation of the latent is larger than the codebook perturbation the spec's fourth erratum measured (mean relative 2.28e-3 to 2.52e-3, max 1.06e-2, `spec:395`), which alone flips 4.04–12.73% of codes. **A CUDA codec encoder should be expected to change codes**, and the 5% cliff would be crossed by the backend change alone on some cases.

**`SYNTH_CUDA_TF32` no longer exists.** `docs/backends.md:35-56`: CUDA F32 matrix multiplies compute at TF32 and *"There is no build option to change this"*; `tests/python/test_cmake_presets.py:102` asserts the variable does not appear in any preset. **`CLAUDE.md:43` is stale on this point** and still describes a strict-FP32 default — do not size a CUDA tolerance from that sentence.

---

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-08-11-qwen3-tts-stage-2-design.md`, §7 and §10, **including its five errata** (three added 2026-08-12, a fourth 2026-08-13, a fifth 2026-08-15 — `spec:3-18`). The fourth erratum's generalized rule governs everything this plan measures on the codec encoder (`spec:495-498`): *"where the port's arithmetic is deliberately more precise than the oracle's, the discrete output of a rounding-sensitive decision is not a gate — the continuous quantity it was rounded from is."* **The fifth erratum (`spec:443-456`) fires here**: the `278` table-alone divergence count *"was never reproduced, and no superseding value exists… Treat `278` as unverified rather than as a measurement, and re-measure the table-alone contribution before a later rung reasons from it."* Plan 4 is that later rung; Task 16 owns it, and no task may cite `278` as a measurement.
- Carry-over ledger: `docs/superpowers/plans/2026-08-14-qwen3-tts-stage-2-plan-3-carryover.md`. Structural template: `docs/superpowers/plans/2026-08-13-qwen3-tts-stage-2-plan-3-icl-path.md`.
- **Published packages stay byte-identical.** CustomVoice is live at `jiangzhuo9357/qwen3-tts-12hz-0-6b-customvoice-gguf` (created 2026-07-28, updated 2026-07-29) carrying BF16, F16 and Q8_MIXED, with digests pinned in `scripts/hf_cards/qwen3-tts-12hz-0-6b-customvoice.yaml:171-193` — BF16 `01dfad52dd507c26a14d101c4247d375257aa63b07e62706ec3daa0a33ea515d` (2,274,117,280 bytes, 657 tensors, 402 BF16 + 255 F32), F16 `f79ada91…` (266 F16 + 391 F32), Q8_MIXED `d1f9be7b…`. **The profile *names* `F16`, `Q8_MIXED` and `Q5_K_MIXED` are a public contract with already-shipped packages and cannot move** (`policy.cpp:29-35`). Any task that changes `policy.cpp` re-cuts all three CustomVoice packages and proves the digests unchanged; any task that changes `catalog.cpp:expected_type` proves the three published packages still load. The Base package's Profile Compatibility ID `34d4de22a329b6bc8347cb952b6fa16513320012628598ab59743679cc16806e` is unchanged or the change is that task's headline.
- **`ggml/` is a submodule and cannot carry a local change.** An edit vanishes at the next `git submodule update`. If a backend measurement wants a kernel ggml does not have, the answer is a different upstream operator, not a patched kernel — that is what emptied this project's patch set (`ggml-patches/README.md`), and other GGML TTS ports are worth reading first.
- **Every test this plan writes must be load-bearing, proved by inverting the rule under it — and the inversion must name the dimension it perturbs.** The standing procedure (green → delete or invert the rule → confirm failure → restore → record in the commit message) is necessary and **not sufficient**. Plan 3 found two classes it structurally cannot see (carry-over §2.3): **mutual masking**, where deleting either check of a pair changes nothing (five pairs found, four in `src/arch/qwen3-tts/profile.cpp` and one at the request seam), and **inversion sets that all perturb one dimension** — Task 6 of Plan 3 shipped eight inversions that *all* changed the token sequence's length and none of which exercised the element-wise off-by-one the task existed to prevent. **Ask which dimension each inversion perturbs, not how many there are**, and span at least four of kind/presence/value/order/length. `tests/qwen3_tts_profile_test.cpp:436-453` and `:2199-2275`, and `tests/qwen3_tts_codec_encoder_test.cpp:717-732`, are the three in-tree lists that do this correctly.
- **A standing hazard at one of those files:** the node-count assertion in `tests/qwen3_tts_codec_encoder_test.cpp` masks seven inversions' failure signals, all enumerated in the file with their isolated results (445 whole-frame and 449 ragged are both pinned, because the replicate pad grows 4 nodes when `extra_padding` ≠ 0). **Read that enumeration before adding an assertion there.**
- **The sanitizer gate runs for every task that touches C/C++ inference or tool code** — Tasks 2, 4, 6, 11, 12 at minimum, and any other task whose steps edit `src/`, `tools/` or `tests/*.cpp`.

  ```bash
  cmake --build build-sanitize --target synthesize-check-unit
  ```
- **`scripts/ci/clang-format.sh --check-diff origin/main` only checks TRACKED files.** Run it *after* `git add`. Plan 1's Task 10 shipped 8 violations by running it before.
- **Every performance number names its build, and so does every generated frame count.** The `dev-*` presets inherit `RelWithDebInfo`, so ggml-cpu compiles at `-O2 -g -DNDEBUG` against a plain `Release` tree's `-O3 -DNDEBUG -mcpu=native`; `-O2` measured **2.19× slower** (`docs/testing.md:438-443`). The rule (`:455-467`): *"a `dev-*` preset … proves correctness — functional behavior, placement, token/tensor agreement — never wall-clock time."* Performance figures come from a `Release`-typed tree — the committed `rel-dgx-spark` preset, or the plain `cmake -S . -B build` default on a CPU-only host. `rel-dgx-spark` is a measurement tree, **not** a registered CI gate (`:479-483`). And the same rule applies to output *content*: identical input, seed and backend, threads swept 1/2/4/8/20 with no movement, but **Release gives 24,960 PCM frames and RelWithDebInfo gives 48,000**, each reproducible (carry-over §4). **No frame count may be pinned or quoted without naming its build.** This plan will therefore be looking at two different outputs of the same code, by design.
- **`docs/backends.md`'s six gates apply to any backend claim** (`:486-497`): same cases as CPU; match reference and CPU tensors within tolerance; finite correctly shaped PCM; 12–32 cases; **prove actual device placement** and report latency, RTF and peak memory; repeated-run and resource-cleanup checks. And `:499-501`: *"Performance measurement is **required** for support, but no minimum speedup is."*
- **Gate baseline.** The carry-over records the branch-end state as unit and sanitizer both **98% of 101**, with exactly two known pre-existing failures — `synthesize-python-api-wheel-test` and `synthesize-vits-python-unit`, both on gitignored VITS artifacts — and qwen3-tts integration **14/14** (carry-over §4). **A further eight VITS integration failures are reported as pre-existing at this plan's commissioning and are recorded nowhere in the tree.** Do not assume the number. **Task 1 Step 0 establishes the actual baseline by running both gates and writing the observed failure list into the task's commit message**; after that, any failure outside the recorded list is yours.
- **A `unit`-labelled test may not depend on the real 2.5 GB package.** Family rules are unit-tested against synthetic `HParams` and in-memory LCG weights; anything needing the real GGUF is an integration test guarded on `SYNTH_QWEN3_TTS_BASE_TEST_MODEL`.
- **Nothing under `models/` or `build/` is ever committed.** GGUFs, oracle artifacts and reports live there and stay ignored; the Golden Manifest, the tolerance file and the card specification are the committed contract.
- **When you add a field to an envelope or a package, the question is not "is this value valid" but "is this count bounded by the same thing that bounds it at creation."** The recurring shape hit three times on Plan 3 (carry-over §3.1) — a count bounded where a value is created and unbounded where it is loaded, at 1.48 GiB, 23.3 GiB and a 34.4 s → 67.3 s wall-time amplification respectively. **The instrument that catches one is blind to another**: the third case reports **+0 KiB** peak RSS and moves only wall time, so copying MB1's peak-RSS arm there would have produced a guard that cannot fail. Expect a fourth.
- **The ICL transcript-mismatch hang costs 475.91 s of CPU at `CMAKE_BUILD_TYPE=Release`** — runs to `kDefaultMaxFrames = 2048`, returns `SYNTH_ERR_OUTPUT_LIMIT` with zero audio (carry-over §3.2). The current mitigation is one static error string; shortening it needs a lower ICL ceiling or a run-away detector, **not done**. Any timing run that trips this loses eight minutes per case — budget for it and do not read it as a regression.
- **The Base package is not published, and publication is a separate act requiring jiangzhuo's confirmation at the time.** Nothing in this plan uploads anything or asks to.

---

## File Structure

| File | Status | Responsibility |
| --- | --- | --- |
| `tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-base.manifest.json` | modify | The second reference recording's cases; `case_count` moves with them |
| `tests/qwen3_tts_percentile.h` | create | The C++ percentile lifted out of `icl_real.cpp` so a `unit` test can reach it |
| `tests/qwen3_tts_percentile_test.cpp` | create | The C++ half of the p95 cross-check, against a committed fixture |
| `tests/python/test_percentile_agreement.py` | create | The Python half, reading the same committed fixture |
| `tests/fixtures/qwen3-tts/percentile-cross-check.json` | create | One fixture, two consumers — the only thing that keeps the two implementations honest |
| `scripts/validate-qwen3-tts-codec_encoder.py` | modify | The redesigned statistic; a `--backend` coordinate; per-case flip rate as a gated quantity |
| `tests/qwen3_tts_icl_real.cpp` | modify | The C++ arm follows the redesigned statistic; consumes `tests/qwen3_tts_percentile.h` |
| `tests/qwen3_tts_xvector_driver.cpp` | modify | `--trim-seconds`, matching the oracle's own flag |
| `tests/qwen3_tts_codec_encoder_driver.cpp` | modify | `--trim-seconds`; a backend selector if Task 12 lands one |
| `scripts/validate-qwen3-tts-replay.py` | modify | Pass the trim through `run_compare_x_vector`; widen `replay` once it can |
| `tools/synthesize-quantize/policy.cpp` | modify | `classify_qwen3_speaker_encoder` and the `codec.encoder.*` arm; the resolver's new roles |
| `tests/qwen3_tts_quantization_policy_test.cpp` | modify | The 237 new names, by name, with role inversions |
| `src/arch/qwen3-tts/catalog.cpp` | modify | `expected_type` reconciled with the quantizer — **only if Task 6 rules that way** |
| `tests/tolerances/qwen3-tts.json` | modify | Five cells: base/F16/CPU, base/Q8_MIXED/CPU, base/BF16/CUDA, base/F16/CUDA, and base/BF16/CPU `public` |
| `src/arch/qwen3-tts/model.cpp` | modify | A twin pass for the new graphs, with its own prefix — **only if Task 12 rules that way** |
| `src/arch/qwen3-tts/speaker-encoder-host.cpp`, `codec-encoder-host.cpp` | modify | The hard-wired `create_cpu_scheduler` becomes a plan decision — same condition |
| `tests/CMakeLists.txt` | modify | `_synth_qwen3_tts_codec_upstream_root` gains an override; new tests registered |
| `scripts/hf_cards/qwen3-tts-12hz-0-6b-base.yaml` | create | Does not exist for this variant |
| `docs/quantization.md` | modify | This family has **no section at all** in it today |
| `docs/backends.md` | modify | The Base variant's placement result, whichever way it lands |
| `docs/porting/families/qwen3-tts.md` | modify | Record Stage 2 Plan 4 |
| `docs/superpowers/specs/2026-08-11-qwen3-tts-stage-2-design.md` | modify | The fifth erratum's re-measurement |

---

## The tolerance grid, and exactly which cells this plan fills

`tests/python/test_tolerance_coverage.py` reads `reference_profile` as `reference_stage.split("-")[1].upper()` (`:44-51`), which for `"source-bf16-oracle-vs-f32-cpu"` is **`BF16`**. The rule at `:145-158` is two assertions, not one:

- **every profile** under a variant must carry **exactly** the reference profile's stage set (`:146-151`);
- **every `backends.<name>` sub-grid** under any profile must carry **exactly that same set** (`:152-158`).

An absent `backends` key is permitted (`entry.get("backends", {})`), which is why CustomVoice's `Q8_MIXED` has no CUDA cell and the tree is green. That hole stays; this plan does not close another variant's.

Base's reference stage set is **{`public`, `replay`, `codec_encoder`}**, so **every new cell is a triple, not a number**:

| cell | stages required | today | owner |
| --- | --- | --- | --- |
| base / BF16 / CPU / `public` | — | **placeholder**: `checks: 0`, `all_passed: null` at `tests/tolerances/qwen3-tts.json:333-338` | Task 5 |
| base / **F16** / CPU | `public`, `replay`, `codec_encoder` | absent | Task 7 |
| base / **Q8_MIXED** / CPU | `public`, `replay`, `codec_encoder` | absent | Task 8 |
| base / BF16 / **backends.CUDA** | `public`, `replay`, `codec_encoder` | absent | Task 13 |
| base / F16 / **backends.CUDA** | `public`, `replay`, `codec_encoder` | absent | Task 13 |

The placeholder at `:333-338` says what it is, in its own words: *"PROVISIONAL placeholder, not yet measured… `checks: 0` and `all_passed: null` are deliberately not customvoice's `8`/`true` -- copying those would claim a result that has never happened."* Replacing it is a real run of `scripts/validate-qwen3-tts-public.py`, not a copy of CustomVoice's numbers.

**Validator support, as it stands.** `scripts/validate-qwen3-tts-replay.py` already takes `--profile` (default BF16), `--backend` (default CPU) and `--stage`, and resolves through `resolve_tolerance_stage(...)` (`:41-70`, `:99-101`), reading `backends` for anything that is not CPU (`:69`) — so `replay` cells are fillable today. `scripts/validate-qwen3-tts-codec_encoder.py` takes `--variant`, `--profile` and `--stage` (`:372-374`) but **no `--backend`**, and `load_gates` (`:315-334`) indexes straight through `profiles[profile]["stages"][stage]["probes"]` with no `backends` step. Task 11 adds that coordinate. The family's validator set is exactly {`codec_encoder`, `public`, `replay`}, so **adding any new stage name would also need a new validator** (`test_tolerance_coverage.py:160-213`) — this plan adds none.

---

### Task 1: A second reference recording, and the baseline this plan is measured against

**Files:**
- Modify: `tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-base.manifest.json`
- Modify: `tests/tolerances/qwen3-tts.json` (`case_count` only)

**Why this is first.** Every flip-rate number in the tree — 0.00 / 3.96 / 3.96 / 5.33 (`tests/tolerances/qwen3-tts.json:678`) — comes from **one recording, one speaker, one microphone, at two distinct lengths**. Three files say so (`:636`, `scripts/dump_reference_qwen3_tts_codec_encoder.py:1137-1143`, `scripts/validate-qwen3-tts-codec_encoder.py:137-142`), and the family record has said so since 2026-08-14 (`docs/porting/families/qwen3-tts.md:2980-2995`). The identical 3.96/3.96 pair is the same clip measured twice, so the four points are not four independent observations of anything. Task 3 is about to derive a **flip-rate threshold** from exactly this data, and Tasks 7, 8 and 13 are about to perturb the flip rate deliberately. **A threshold calibrated on one speaker and then invalidated by a later task would have to be re-derived after the cells it gates were committed** — which is the sequencing failure the plan is arranged to avoid.

- [ ] **Step 0: Establish the gate baseline by running it, and write down what you see**

```bash
cmake -S . -B build -DSYNTH_BUILD_TESTS=ON -DSYNTH_BUILD_PYTHON_TESTS=ON \
  -DSYNTH_BUILD_INTEGRATION_TESTS=OFF -DSYNTH_BUILD_TOOLS=ON
cmake --build build --target synthesize-check-unit
```

The carry-over records two pre-existing unit failures by name. A further eight VITS integration failures are reported and **are recorded nowhere in the tree**. Record the observed failure list — names, not a count — in this task's commit message. From here, any failure outside that list is yours.

- [ ] **Step 1: Source a second reference clip, from a second speaker**

Constraints the package declares and the manifest must respect: 24 kHz mono, `min_frames_per_clip` 24,000 samples (1 s), `max_frames_per_clip` 720,000 (30 s), `max_total_frames` 720,000, `max_reference_count` 1. It needs a transcript, because the case must be drivable in ICL mode as well as x-vector mode.

**It must be a genuinely different recording** — a different speaker and a different microphone, not a transform of `clone.wav`. A pitch-shifted or noise-added copy of the existing clip corroborates nothing, because the flip rate is a property of where the latents sit relative to the codebook, and a transform of one recording moves along the same manifold. Record provenance and licence in the manifest case description.

- [ ] **Step 2: Dump it through every existing oracle, at the recorded dtype**

The Base oracle, `scripts/dump_reference_qwen3_tts_codec_encoder.py`, `scripts/dump_reference_qwen3_tts_icl_prompt.py`, and the two `_float32` twins that Plan 3's decomposition method requires (carry-over §2.1: *"run upstream's own module in float32, and decompose the deviation into port vs upstream-f32 and upstream-f32 vs the bf16 oracle"*).

**Every case is dumped at the same oracle dtype and each one's `conventions.json` records which.** A case dumped under a load omitting `dtype=torch.bfloat16` sits on a float32 codebook that disagrees with the bf16 one on roughly half its codes, and would silently move the whole grid. A case whose recorded dtype differs from the rest is regenerated, not averaged in.

- [ ] **Step 3: Measure its semantic flip rate, and report which side of the cliff it lands on**

Run `scripts/validate-qwen3-tts-codec_encoder.py` over the new case at BF16/CPU and record the semantic flip rate beside the existing four. **State the outcome; this plan predicts none:**

- **Below ~5%.** Then the cliff is confirmed as a function of reference length rather than of speaker, and Task 3 has a fifth point on the safe side to calibrate with.
- **Above ~5% at a short reference length.** Then the cliff is *not* a length property, the `gate_scope_warning`'s own suggested branch ("scope this probe by reference length") is refuted, and Task 3 must take the flip-rate-conditioned branch rather than the scoping one.
- **The case cannot be dumped or driven at all** — a licence, a sample-rate, or an upstream-tokenizer obstacle. Then record what blocked it, and Task 3 proceeds on four points from one recording **with that stated as a limitation in the gate's own note**, not silently.

- [ ] **Step 4: Two edits, not one**

Adding a manifest case moves `variants["qwen3-tts-12hz-0-6b-base"].case_count` (today `12`) in the same change, because `test_tolerance_case_count_matches_manifest` (`tests/python/test_golden_manifests.py:165`, assertion `:207-209`) requires it to equal `len(manifest["cases"])`. The variant `note` at `:329` spells out "case_count (12) tracks this variant's manifest" and moves with it.

- [ ] **Step 5: Run the coverage tests and commit**

```bash
uv run --project scripts/envs/qwen3-tts --locked python -m unittest \
  tests.python.test_golden_manifests tests.python.test_tolerance_coverage -v
git add tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-base.manifest.json tests/tolerances/qwen3-tts.json
git commit -m "qwen3-tts: a second reference recording, so the flip-rate data is not one clip"
```

The commit message carries the observed gate baseline from Step 0, the new case's flip rate, and which of Step 3's three outcomes it produced.

---

### Task 2: One percentile, cross-checked — the C++ and Python implementations under a registered test

**Files:**
- Create: `tests/qwen3_tts_percentile.h`, `tests/qwen3_tts_percentile_test.cpp`, `tests/python/test_percentile_agreement.py`, `tests/fixtures/qwen3-tts/percentile-cross-check.json`
- Modify: `tests/qwen3_tts_icl_real.cpp`, `tests/CMakeLists.txt`

**This matters more here than it normally would, because this plan's conclusions are p95s.** The tolerance file discloses the hole itself (`tests/tolerances/qwen3-tts.json:668`): *"the end-to-end test transcribes the p95 statistic from the validator by hand (numpy-linear percentile of the per-frame L2 relative to the oracle frame's own norm), and **the two implementations have NO registered cross-check** — they were compared once, manually, on the same buffers, agreeing to every printed digit, and could drift apart silently afterwards."* The two sites are `scripts/validate-qwen3-tts-codec_encoder.py:253-265` (`np.percentile(relative, 95)`) and `tests/qwen3_tts_icl_real.cpp:363-400` (`percentile_linear`, which says in its own comment that it is deliberately reimplementing numpy's `"linear"` interpolation), called at `:817-828`.

**Task 3 is about to change the statistic in both places.** A redesign that lands in one and not the other, or lands in both with a differing interpolation, is exactly the drift `load_gates`'s fatal-disagreement discipline exists to prevent — and `load_gates` covers *thresholds*, not *implementations*, so it does not reach this.

- [ ] **Step 1: Lift the C++ percentile into a header a `unit` test can reach**

`percentile_linear` and `reconstruction_p95_relative` live inside an integration-tier file that `synthesize-check-unit` does not build. Move them to `tests/qwen3_tts_percentile.h` and have `icl_real.cpp` include it. **Nothing about the arithmetic changes in this step** — a behaviour change and a relocation in one commit makes the next bisect useless.

- [ ] **Step 2: Commit one fixture, consumed twice**

`tests/fixtures/qwen3-tts/percentile-cross-check.json` carries input vectors and the expected percentile for each, generated once by numpy and committed. It must include the cases where linear interpolation actually bites: n not congruent to 1 mod 20 (so the 95th percentile falls between two samples), n = 1, n = 2, ties, a vector whose values span several orders of magnitude, and one drawn from a real per-frame relative array so the fixture is not purely synthetic.

- [ ] **Step 3: Two tests, one fixture**

`tests/qwen3_tts_percentile_test.cpp` (registered under `unit`) and `tests/python/test_percentile_agreement.py` each read the same file and each assert against the same expected values. A test that recomputes the expectation with the implementation it is testing proves nothing; the fixture is the third party.

- [ ] **Step 4: Prove both are load-bearing, and name the dimension**

Change the C++ interpolation from linear to nearest-rank and confirm the C++ test fails (**value** dimension). Change the fixture's expected value for one entry and confirm **both** tests fail (**value**). Reverse the input ordering for a case where order should not matter and confirm neither fails, then reverse it where it should and confirm both do (**order**). Restore everything. Record all four in the commit message.

- [ ] **Step 5: Gates and commit**

```bash
cmake --build build --target synthesize-check-unit
cmake --build build-sanitize --target synthesize-check-unit
scripts/ci/clang-format.sh --fix
git add tests/qwen3_tts_percentile.h tests/qwen3_tts_percentile_test.cpp \
        tests/python/test_percentile_agreement.py \
        tests/fixtures/qwen3-tts/percentile-cross-check.json \
        tests/qwen3_tts_icl_real.cpp tests/CMakeLists.txt
scripts/ci/clang-format.sh --check-diff origin/main
git commit -m "qwen3-tts: the C++ and Python percentiles cross-check against one committed fixture"
```

---

### Task 3: Redesign the semantic statistic — flip rate gated separately from reconstruction on non-flipped frames

**Files:**
- Modify: `scripts/validate-qwen3-tts-codec_encoder.py`, `tests/qwen3_tts_icl_real.cpp`, `tests/tolerances/qwen3-tts.json`

**The ruling is Plan 3's and it is already made** (carry-over §1.3): *"the statistic needs redesign — gate the flip rate separately from the reconstruction error on non-flipped frames. The real evidence for `base-ref-max` is the 100% code agreement, not the p95."* The cell's own note (`tests/tolerances/qwen3-tts.json:684`) offers two branches and **deliberately declines to choose**: *"either scope this probe by reference length (and say so in the gate), or replace the percentile with a statistic that does not have a cliff in it (a flip-rate-conditioned measure, or the median, which sits at 0.0027 here and is stable across all four cases)."*

**Everything the redesign needs is already computed and printed.** `scripts/validate-qwen3-tts-codec_encoder.py:279-283` computes `differ = codes[left] != codes[right]` and `per_stage = differ.sum(axis=0)`, printing the semantic column's own count and the full per-column list for all three pairings. `:253-259` computes `per_frame` (L2), `norms` and `relative`, printing median / p95 / max per branch per pairing. **Conditioning on non-flipped frames is an index into arrays that already exist in the same scope.** What is missing is the mask, not the data.

**The constraint this redesign must meet, stated before it is written.** From the fault-injection table above: the semantic branch must keep **≥80× discrimination** against the symmetric-pad fault (which reads 1.601 on the semantic branch), and the acoustic branch must not lose its **3.5×** — inside a real headroom of **1.64×** from WAV input, not the 2.0× the `headroom` field records for the `waveform.f32` comparison. The two branches stay **separate keys and are never averaged**: rms 13.5 (semantic) against 3.11 (acoustic), and their dynamic ranges differ sharply — semantic moves 0.004103 → 1.601 under the fault (390×) while acoustic moves 0.2491 → 1.741 (7.0×) (carry-over §1.2).

- [ ] **Step 1: Build the masked statistic and measure it on every case available**

Two quantities where there was one:

- a **flip rate** per branch per case, which is `per_stage` already computed, expressed as a fraction of frame-stage decisions;
- a **reconstruction deviation conditioned on non-flipped frames**, which is the existing `relative` array indexed by `~differ`.

Measure both across all four existing cases (`base-icl-en`, `base-ref-min`, `base-text-short`, `base-ref-max`) **and Task 1's new case**, at BF16/CPU. Also measure the fault-injected values for both quantities, by re-injecting the symmetric `MimiConv1d` padding split into `src/arch/qwen3-tts/codec-encoder.cpp` and reverting — the same injection Plan 3 used, which the tolerance file records as reproducing 1.879 / 1.634 / 2.115 end to end through `synthesize-qwen3-tts-codec-encoder-golden` (`:667`).

**A flip rate needs a threshold, and nothing in the tree calibrates one.** The only data is the four (soon five) numbers above. Derive the threshold from the measurement and say what it is derived from, in the gate's own note. Do not carry a number over from another family or from the codebook-isolated 4.04% floor — the tolerance file already records one such conflation being corrected (`:652`).

- [ ] **Step 2: Report which of three outcomes the redesign produced, before committing anything**

- **The masked statistic keeps ≥80× and the flip rate has a defensible threshold.** Commit both, retire `gate_scope_warning`, add `base-ref-max` to `cases`, and record the discrimination figure beside each new gate.
- **The masked statistic works but the flip-rate threshold is not defensible on the available cases** — e.g. Task 1 landed on its third outcome and five points from one recording is all there is. Then commit the masked reconstruction gate (which does not need the threshold), leave the flip rate **recorded and ungated** with the reason stated, and keep `gate_scope_warning` with its scope narrowed to what is now actually unknown. **This is a legitimate landing, not a partial one.**
- **The masked statistic cannot hold ≥80×.** Then the redesign has failed its stated constraint and **the plan stops here to re-scope rather than shipping a weaker gate** — the same refusal Plan 3 made when it declined to widen 2.0e-2 to 0.35. Record the measured discrimination and what would be needed to recover it (more cases, a different branch statistic, a different comparison pairing). Tasks 7, 8 and 13 then run against the *old* gate with `gate_scope_warning` intact and their conclusions carry that caveat explicitly.

- [ ] **Step 3: Land it in both implementations, and prove they agree**

The Python validator and the C++ arm at `tests/qwen3_tts_icl_real.cpp:817-828` are **both consumers of the same two cells** (`registered_consumers`, `:654`). A redesign that lands in one is a silent divergence. Extend Task 2's fixture with the masked statistic, so the cross-check covers the new shape and not only the old percentile.

- [ ] **Step 4: Prove the new gates are load-bearing, naming dimensions**

Re-inject the symmetric pad and confirm **both** new gates fail (**value**). Invert the mask (condition on flipped frames instead of non-flipped) and confirm the statistic moves by the ratio the measurement predicts (**kind**). Drop the flip-rate gate entirely and confirm `base-ref-max`'s recorded rate stops being checked (**presence**). Restore.

- [ ] **Step 5: Gates and commit**

```bash
ctest --test-dir build -R '^synthesize-tolerance-coverage$' --output-on-failure
cmake --build build --target synthesize-check-unit
git commit -m "qwen3-tts: gate the semantic flip rate separately from reconstruction on non-flipped frames"
```

The commit message states which of Step 2's three outcomes ran, the measured discrimination for every gate it touched, and the number of independent recordings behind each threshold.

---

### Task 4: `trim_seconds` reaches the drivers

**Files:**
- Modify: `tests/qwen3_tts_xvector_driver.cpp`, `tests/qwen3_tts_codec_encoder_driver.cpp`, `scripts/validate-qwen3-tts-replay.py`

**One flag, two holes** (carry-over §3.4). The oracle side already has it — `scripts/dump_reference_qwen3_tts_base.py:187,224,275,323-343` takes `--trim-seconds`, and the manifest uses it: `trim_seconds: 1.0` for `base-ref-min` and `30.0` for `base-ref-max` (`tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-base.manifest.json:782,892`). **The gap is entirely on the C++ driver side.**

What it unblocks, in the tolerance file's own words (`:401`): *"THE ONLY TWO CASES THAT WOULD ADD A DATA POINT are base-ref-min and base-ref-max, whose x-vectors do differ… They are also the two this comparison cannot currently drive: `run_compare_x_vector` hands the driver the WAV path and **the driver has no trim**, so the port would encode the full clip while the oracle encoded a trimmed one, and the comparison would fail for a reason that is not the port's arithmetic."* And carry-over `:499-503`: the `replay` stage was **deliberately not widened** because 10 of its 12 x-vectors are byte-identical — *"widening would be theatre — the two that differ need `trim_seconds`."* It also unblocks the frame-count half of the 9-frame port comparison, which the carry-over records as still owed (`:464-466`).

- [ ] **Step 1: Match the oracle's flag exactly**

Same name, same units, same semantics as `dump_reference_qwen3_tts_base.py`'s. Read that script's implementation for how it handles a trim longer than the clip — `base-ref-max` at 30.0 s against an 8.08 s recording is a **loop**, not a truncation (`dump_reference_qwen3_tts_base.py:339-342`, `np.tile`), and a driver that truncates where the oracle loops produces a comparison failure that is not the port's arithmetic. **Transcribe the behaviour; do not infer it from the flag's name.**

- [ ] **Step 2: Widen `replay`, and report what widening bought**

With the flag in place, drive `base-ref-min` and `base-ref-max` through `run_compare_x_vector` and add them. Three outcomes:

- **Both agree within the existing `speaker.x_vector` gate.** Then `replay.cases` moves from 2 to 4, the `cases_not_extended` block is replaced by the measurement, and the x-vector residual finally rests on more than one reference clip.
- **One or both deviate beyond the gate.** Then it is found, not accommodated — the gate is not widened, and the deviation is attributed (trim arithmetic, loop handling, or the port's ECAPA) before this task closes.
- **The comparison still cannot be driven** for a reason the flag does not fix. Then record the actual blocker; `replay` stays at 2 with an updated note, and the "one recording" caveat stays on the x-vector residual.

- [ ] **Step 3: Prove, gate, commit**

Delete the trim application inside the driver (keep the flag parsed) and confirm the `base-ref-min` comparison fails (**value**). Set the trim to a different duration and confirm it fails differently (**value**, distinct). Restore.

```bash
cmake --build build --target synthesize-check-unit
cmake --build build-sanitize --target synthesize-check-unit
git commit -m "qwen3-tts: the x-vector and codec-encoder drivers take --trim-seconds"
```

---

### Task 5: Fill base/BF16/CPU `public` with a real run

**Files:**
- Modify: `tests/tolerances/qwen3-tts.json`

**The placeholder is explicit and self-disclosing** (`:333-338`): *"PROVISIONAL placeholder, not yet measured. No public-seam check has ever been run against qwen3-tts-12hz-0-6b-base… This key exists only so this variant's stage set satisfies tests/python/test_tolerance_coverage.py's family-wide validator check… `checks: 0` and `all_passed: null` are deliberately not customvoice's `8`/`true` -- copying those would claim a result that has never happened."*

It was honest when it was written because no port existed. Plan 3 finished the port. `scripts/validate-qwen3-tts-public.py` exists and is family-wide. **This is the reference cell every other cell in the grid is compared against**, so it is filled before any profile is cut, not after.

- [ ] **Step 1: Run the public-seam validator against the real BF16 package on CPU**

Record the actual `checks` count and `all_passed`, plus the build the run used. Do not copy CustomVoice's `8` — the Base variant's public seam differs (`reference_transcript` and `reference_language` are `OPTIONAL` here and `UNSUPPORTED` there, `source_flags` is exactly 9, description_language still `UNSUPPORTED`), so the check count is a measurement.

- [ ] **Step 2: Three outcomes**

- **Every check passes.** Replace the placeholder with the measured `checks` and `all_passed: true`, and delete the placeholder's `description` in favour of one describing the run.
- **A check fails.** It is a finding about the Base package's public seam, and it is fixed or recorded as a defect before this task closes. The cell is not committed with a failing check silently reclassified.
- **A check is inapplicable to this variant.** Then the count differs from CustomVoice's for a stated reason, which goes in the cell's `description` — a differing count with no explanation is the shape the placeholder was written to avoid.

- [ ] **Step 3: Commit**

```bash
uv run --project scripts/envs/qwen3-tts --locked python -m unittest tests.python.test_tolerance_coverage -v
git add tests/tolerances/qwen3-tts.json
git commit -m "qwen3-tts: base/BF16/CPU public is a measurement, not a placeholder"
```

---

### Task 6: Teach the quantizer the Base package's 237 tensors, and settle the runtime/tool contradiction

**Files:**
- Modify: `tools/synthesize-quantize/policy.cpp`, `tests/qwen3_tts_quantization_policy_test.cpp`
- Modify (conditionally, on Step 3's ruling): `src/arch/qwen3-tts/catalog.cpp`

**This is the blocker. Every quantization measurement in this plan is downstream of it.** 237 tensor names are unclassified — 76 `speaker_encoder.*` + 161 emitted `codec.encoder.*`, which is exactly the Base/CustomVoice difference (894 − 657 = 237; the arithmetic is pinned at `tests/python/test_tolerance_coverage.py:172-180` and in the family record's census).

`tests/qwen3_tts_quantization_policy_test.cpp` (128 lines) pins the split for CustomVoice-era names only — `talker.*`, `codec.decoder.*` — with **no** `speaker_encoder.*` or `codec.encoder.*` case. Its last block (`:120-126`) asserts unknown names are refused, so the current behaviour is deliberate for the names it knows and simply blind to the new ones.

- [ ] **Step 1: Write the failing test first, by name**

Extend `tests/qwen3_tts_quantization_policy_test.cpp` with the new names. **Assert every one of the 237 by name**, not by a loop that checks "at least one classifies". Plan 3's standing example is why: a resolver that aliased all 31 acoustic codebooks to slot 0 left every name resolved, every pointer non-null, the catalog sweep green and the by-name test passing — only `std::set::insert(...).second` caught it (carry-over §4).

Then run the tool against the real package and watch it fail exactly as measured:

```bash
build/bin/synthesize-quantize models/qwen3-tts-12hz-0-6b-base/qwen3-tts-12hz-0-6b-base-BF16.gguf \
  /tmp/plan4-probe-base-F16.gguf --quant F16
# synthesize-quantize: unknown qwen3-tts tensor: speaker_encoder.asp.conv.bias
```

- [ ] **Step 2: Establish the row arithmetic from the package, not from a claim**

Under `Q8_MIXED` the profile's `matrix_weight_layout` is `TensorLayout::PackedMatrix` (`policy.cpp:49-50`), and `quantize.cpp:212-215` demotes a `matrix_family` tensor to `Native` only when `ggml_n_dims(tensor) < 3`. A speaker-encoder convolution weight is three-dimensional `{kernel, in, out}`, so it stays packed and its row becomes `ne[0] * ne[1]` = `kernel * in` (`:228`), checked against `ggml_blck_size(target.type)` at `:231-233`. Under `F16` the layout is `Native` (`policy.cpp:42`) and F16's block size is 1, so no row constraint arises there.

**A correction the plan carries rather than repeating.** The research for this plan asserted the stem convolution `speaker_encoder.blocks.0.conv` (kernel 5, in = `mel_bins`, out 512 — `catalog.cpp:466`) has a packed row that is not a multiple of 32, offering 400 and 640 as the values for 80- and 128-bin front ends. **640 is 32 × 20**, and this package declares a 128-bin mel front end (`spec:65`; `mel_bins` is read from package metadata at `src/arch/qwen3-tts/weights.cpp:581`, not a literal). So the claim does not hold for this package as declared. It *would* hold at 80 bins (400 % 32 = 16), and **Q5_K's super-block of 256 is a separate and much tighter constraint** — 640 % 256 ≠ 0, 192 % 256 ≠ 0, 128 % 256 ≠ 0.

**Therefore: derive every row size from the actual package, print the table, and commit it.** Enumerate all 38 speaker-encoder convolutions with their `{kernel, in, out}` and their packed row, and state for each whether it clears 32 and whether it clears 256. Do not restate either the research's claim or this paragraph's arithmetic as a finding; the package is the source.

- [ ] **Step 3: Rule on the runtime/tool contradiction, and say which side is right**

`catalog.cpp:134` classifies `speaker_encoder.*` into `Half::Talker`, and `Resolver::conv` passes `Role::Matrix` for the weight, so **the runtime today expects 38 speaker-encoder convolution weights to be F16 under `F16` and Q8_0 under `Q8_MIXED`** — for a package `synthesize-quantize` refuses to emit. Note the biases are unaffected: `Resolver::conv` resolves `.bias` with the default `Role::Sensitive`, which is F32 under every profile.

`codec.encoder.*` is *not* contradictory: `"codec."` prefixes it, so it lands in `Half::Codec` → F32 under every profile, which agrees with the rule the file states at `:144-149` and which `classify_qwen3_codec`'s comment (`policy.cpp:350-357`) justifies with a measurement — halving the codec made it **1.75× slower on CPU**, 4.2 s to 7.3 s on a 37-frame case, because its convolutions run through im2col into a matrix multiply and ggml's F16 path there is slower than its F32 one.

**Three outcomes, and this task picks one on evidence:**

- **The runtime is right and the tool should follow.** Speaker-encoder convolution weights classify `MatrixWeight`, and the quantizer emits them at the profile type. This is the outcome that makes Task 9's question live — is quantizing 76 tensors worth anything? — and it requires Step 2's row table to clear the block size for every profile actually cut.
- **The tool's silence is right and the runtime should follow.** Speaker-encoder convolution weights classify `Sensitive` and stay F32, and `catalog.cpp:expected_type` changes so the runtime expects F32. This is the outcome §7 hints at without asserting ("small enough that quantizing it is unlikely to pay") — **but §7 forbids taking the hint as the answer**, so this outcome is only reachable from Task 9's measurement, not from the hint. If this task lands here provisionally, it says so and Task 9 confirms or overturns it.
- **Neither side is uniformly right** — e.g. the stem takes `Sensitive` on a row-size or a sensitivity ground while the rest take `MatrixWeight`. Then the split is per-tensor, and the reason for each is recorded at the classifier, in the shape OmniVoice's `ConvKernel` role uses (*"The rule reads the name and never the rank"*, `docs/quantization.md:339-366`).

**Whichever way it lands, `catalog.cpp:134`'s `"codec."` prefix test is a shared function and the published CustomVoice packages' load behaviour is a contract.** A change there re-cuts all three CustomVoice packages and proves the digests unchanged (`01dfad52…`, `f79ada91…`, `d1f9be7b…`) and that all three still load. If the reconciliation cannot be made without moving the published grid, **that is the finding**, and it is reported before the change is written rather than after.

- [ ] **Step 4: Cut a Base package and watch the tool succeed**

```bash
build/bin/synthesize-quantize models/qwen3-tts-12hz-0-6b-base/qwen3-tts-12hz-0-6b-base-BF16.gguf \
  build/qwen3-tts-12hz-0-6b-base-F16.gguf --quant F16
```

Record the tensor-type census of the output (the shape `scripts/hf_cards/qwen3-tts-12hz-0-6b-customvoice.yaml:171-193` uses: "402 BF16 + 255 F32" and so on) and its size. **The package must then load**, which is the real test of Step 3's ruling: `Resolver::find` (`catalog.cpp:61-88`) refuses a package whose tensor type disagrees with `expected_type`, so a package the tool emits and the runtime rejects means the reconciliation is wrong and Step 3 is reopened.

- [ ] **Step 5: Prove the classifier is load-bearing, naming dimensions**

Delete one speaker-encoder arm and confirm the by-name test fails for those tensors and only those (**presence**). Change one tensor's role from `MatrixWeight` to `Sensitive` and confirm the test fails on the expected type (**value**). Move a `codec.encoder.*` name into the talker arm and confirm the test catches the half (**kind**). Add a name that matches no arm and confirm it is still refused (**the `:120-126` block still bites**). Restore.

- [ ] **Step 6: Gates and commit**

```bash
ctest --test-dir build -R '^synthesize-qwen3-tts-quantization-policy-test$' --output-on-failure
cmake --build build --target synthesize-check-unit
cmake --build build-sanitize --target synthesize-check-unit
scripts/ci/clang-format.sh --fix && git add … && scripts/ci/clang-format.sh --check-diff origin/main
git commit -m "qwen3-tts: the quantizer classifies the Base package's 237 new tensors"
```

The commit message states which of Step 3's three outcomes ran and why, the packed-row table from Step 2, and the CustomVoice digest proof if `policy.cpp` or `catalog.cpp` moved.

---

### Task 7: Cut base/F16 and fill its three CPU cells

**Files:**
- Modify: `tests/tolerances/qwen3-tts.json`

**F16 is a speed profile for this family, not a size one.** Stage 1 measured it (`docs/porting/families/qwen3-tts.md:1684-1688`, `:1596-1619`): 2168.9 MiB against a 2168 MiB source, RTF 1.21, talker logits cosine 0.999458, final 0.999430 — and wall clock **43.04 s → 9.64 s**, 4.5×, because ggml's CPU bf16 matmul is far slower than its F16 one. **That is Stage 1's CustomVoice measurement and it is not evidence about Base**, whose two new graphs Stage 1 did not contain. §7's whole instruction is that Stage 1's profiles are measured here rather than assumed to carry over.

- [ ] **Step 1: Fill all three stages, because the coverage rule requires the set**

`public`, `replay` and `codec_encoder` — `test_tolerance_coverage.py:145-151` requires exactly the reference profile's stage set, so a `replay`-only F16 cell fails the tree. `scripts/validate-qwen3-tts-replay.py` already takes `--profile`; `scripts/validate-qwen3-tts-codec_encoder.py` already takes `--profile`; `scripts/validate-qwen3-tts-public.py` runs against whatever package it is given.

Every threshold is **measured in this task**; this plan prescribes none. Each measured deviation is checked against the bf16-derived expectation for its stage, and **a stage above that scale does not get a wider gate — it gets found** (the fourth erratum's rule, `spec:472-473`).

**Two figures to carry into the comparison** (carry-over §2.1): `transformer_l7` (135–265× the single-rounding floor) and `downsample` (53–60×) are the two chain stages to watch, being the two whose absmax collapses 35×; l0–l6 sit at 1.6–21×. And **the family doc's stated bf16 range "0.0 … 0.41" is understated 2.4×** — the grid's own maximum over the 33 taps is **0.9896** (`rvq_residual_s14`), with 0.41 being only the maximum over the *non-RVQ* taps. **A bound sized off the stated range is set 2.4× too tight.**

- [ ] **Step 2: Report which of three outcomes F16 produced**

- **F16 pays and clears every gate.** Commit the three cells with their measured values, the size, and the profile's tensor-type census. This is the outcome that makes an `F16` Base package a ship candidate.
- **F16 clears the gates but does not pay** — no meaningful size reduction and no meaningful speed gain on the Base package's graphs, or a gain confined to the talker half that Stage 1 already measured. Then commit the cells **and record "does not pay" as the result**, with the numbers. The profile being buildable is not a reason to publish it; `Q5_K_MIXED` is already buildable for this family and deliberately unpublished (1035 MiB, RTF 0.82, talker logits cosine 0.9648 — `docs/porting/families/qwen3-tts.md:1678-1712`).
- **F16 fails a gate.** Then attribute it before this task closes — decompose against upstream-f32 the way carry-over §2.1 mandates, so "the profile moved it" is separated from "the port moved it". A gate is not widened to admit a profile. **If the failure is the codec encoder's RVQ selections crossing the flip cliff, that is Task 10's subject and this task hands it over rather than adjudicating it.**

- [ ] **Step 3: Commit**

```bash
uv run --project scripts/envs/qwen3-tts --locked python -m unittest tests.python.test_tolerance_coverage -v
git add tests/tolerances/qwen3-tts.json
git commit -m "qwen3-tts: measured base/F16 on CPU across public, replay and codec_encoder"
```

The commit message states the build, the case ids, the package size and tensor census, the flip rates the profile produced, and which of Step 2's three outcomes ran.

---

### Task 8: Cut base/Q8_MIXED and fill its three CPU cells

**Files:**
- Modify: `tests/tolerances/qwen3-tts.json`

**What "MIXED" means here, so the result is read correctly.** `docs/quantization.md:56-57`: *"`Q8_MIXED` version 1 is the first accepted mixed Quantization Profile. It is a **conservative storage profile, not a claim that every weight or operation is Q8**."* And `:26`: *"A Quantization Profile records both storage type and required compute or accumulation precision by tensor group. A nominal profile name does not imply that every tensor uses the same type."* For qwen3-tts specifically the autoregressive half (talker + code predictor, 316 + 86 tensors, 1727.6 MiB of 2164) is entirely 2-D matrices with rows 1024 / 2048 / 3072 and quantizes; the codebooks, both kernel-1 projections, per-head norms, per-branch layer scales and SnakeBeta curves stay exact; six transposed convolutions stay F32 (`policy.cpp:502-507` — *"CUDA's F16 matrix multiply accumulates in half precision. There are six of them."*).

Stage 1's CustomVoice figures for orientation only: 1359.2 MiB, RTF 0.85, logits 0.995731, final 0.995634. **Base is not CustomVoice**, and whether the two new graphs are inside or outside the quantized set is Task 6's ruling, not an inheritance.

- [ ] **Step 1: Fill all three stages, same rule as Task 7**

- [ ] **Step 2: Expect the codes to move, and measure by how much**

Q8_MIXED moved the talker's logits eight-fold on Stage 1 (`docs/porting/families/qwen3-tts.md:1661-1664`). The codec encoder's RVQ argmin is a nearest-neighbour decision, and OmniVoice's codec-half profiles were **blocked** precisely because quantization flipped RVQ tokens on the clone path — *"the clone is a clone of a different clip than the caller supplied"* (`docs/quantization.md:198-205`). **Record the flip rate this profile produces, per case, against the redesigned statistic from Task 3.** If Task 3 landed on its second outcome (flip rate recorded but ungated), the number is still recorded — it is the input Task 10 reasons from.

- [ ] **Step 3: Report which of three outcomes Q8_MIXED produced**

- **Q8_MIXED pays and clears every gate**, including the flip rate. Commit the cells with size, speed and flip rates.
- **Q8_MIXED pays on size but crosses the flip threshold.** Then it is **blocked**, exactly as OmniVoice's codec profiles are, and the block is recorded with the number that produced it. This is a correct and complete result — not a failure of this task, and not a reason to widen anything.
- **Q8_MIXED does not pay on the Base package at all** — the new 237 tensors are a small enough share that the size difference from CustomVoice's ratio is uninteresting, or the wall clock does not move. Then record the numbers and the conclusion, and the profile is not published for this variant.

- [ ] **Step 4: Commit**

```bash
git add tests/tolerances/qwen3-tts.json
git commit -m "qwen3-tts: measured base/Q8_MIXED on CPU across public, replay and codec_encoder"
```

---

### Task 9: The speaker encoder's own answer — does quantizing 76 tensors pay?

**Files:**
- Modify: `docs/porting/families/qwen3-tts.md` (the measurement and its artifact)

**§7's exact words:** *"the speaker encoder is small enough that quantizing it is unlikely to pay."* **That is a hypothesis the design states and forbids assuming** — "Neither precedent is imported by analogy; each is measured here and the result recorded with the artifact that produced it." So this task measures it, and the honest outcome includes confirming the hypothesis with numbers, which is different from adopting it without them.

The subject is 76 tensors: 38 convolution weights and 38 biases (stem + 3 blocks × 11 + mfa + asp_tdnn + asp + fc), enumerated at `catalog.cpp:460-502`. The biases are `Sensitive` under every profile already; the question is only the 38 weights.

- [ ] **Step 1: Measure three quantities against the BF16 baseline, on a Release tree**

For each profile Task 6 made cuttable:

1. **Size** — bytes attributable to `speaker_encoder.*` in each package, not the package total. The whole point is whether 76 tensors of a 2.5 GB package are worth anything.
2. **Wall time** of the speaker-encoder stage alone, and end to end, **on a `Release`-typed tree** per the performance rule. A `dev-*` figure is not admissible here.
3. **x-vector cosine** against the oracle's `speaker/x_vector.f32`, through `scripts/validate-qwen3-tts-replay.py --compare-x-vector`, on every case Task 4's trim flag made drivable.

The existing BF16 baseline for the third: `min_cosine` 0.9999767764206815 with observed 0.9999953552841363 (`tests/tolerances/qwen3-tts.json:344-347`), and the note there establishes that the residual is the **oracle's** bf16 quantization rather than a port defect — at the x-vector's largest element (|ref| = 7.28) the 0.02111 max_abs is 0.68 of one bfloat16 ulp, and recomputing the oracle's own final ASP/FC layer in float64 disagrees with the oracle by *more* (0.022367) than the port does.

- [ ] **Step 2: Report which of three outcomes the measurement produced**

- **Quantizing the speaker encoder pays.** Then §7's expectation is overturned by measurement, the 38 weights stay `MatrixWeight`, and the size and speed figures are recorded with their build.
- **It does not pay** — the size saved is negligible against the package and the wall time does not move outside noise. Then the 38 weights become `Sensitive`, Task 6's ruling is revised if it landed the other way, and **the result is recorded as a measurement rather than as agreement with §7's guess.** State the actual bytes and milliseconds; "unlikely to pay" is the design's prior, and this task's output is the posterior.
- **It pays on size but costs accuracy** — the x-vector cosine moves enough to matter for a conditioning path whose whole job is voice identity. Then it does not ship, and the cosine figure is the reason. Note that a cosine change here is not absorbable downstream: the x-vector is substituted directly into the talker's prompt slot (`spec:76-82`).

- [ ] **Step 3: Record with the artifact that produced it**

§7 requires "the result recorded with the artifact that produced it" — the package digest, the case ids, the build type and preset, and the command line. A number without its artifact is not what §7 asked for.

```bash
git commit -m "qwen3-tts: the speaker encoder's quantization measured, not assumed"
```

---

### Task 10: The codec encoder's own answer — does the conv-exempt shape transfer?

**Files:**
- Modify: `docs/porting/families/qwen3-tts.md`, and `tools/synthesize-quantize/policy.cpp` if the answer changes a role

**§7's exact words:** *"the codec encoder is convolution-heavy, which is the shape that produced the conv-exempt policy in another family. Neither precedent is imported by analogy."* **This task is where the analogy is explicitly refused and replaced with a measurement.**

**The other family's policy and its evidence, so the comparison is against the real thing.** OmniVoice's `F16_CODEC` / `Q8_CODEC_MIXED` never block-quantize a convolution kernel: a fifth `QuantRole`, `ConvKernel`, holds all 85 codec convolutions at the profile's halved fallback at native three-axis shape, and the only codec tensors a profile still quantizes are 73 HuBERT Linears whitelisted positively by module name — *"The rule reads the name and never the rank"* (`docs/quantization.md:339-366`; adopted by jiangzhuo 2026-08-09, `src/arch/omnivoice/quantization.h:69`). The evidence (`docs/quantization.md:355-361`): the exemption took clone-path RVQ drift from **1,023 of 2,808 positions to 98**, a factor of 10.4, and attribution showed all of the remainder is convolution precision — quantizing the 73 Linears to Q8_0 costs nothing measurable (98 against `F16_CODEC`'s 103) while saving 80 MB, whereas convolutions at F32 instead of F16 reach **19 of 2,808** for 161 MB more. **Both profiles are still blocked** — `ref.tokens` is 0/2 exact.

**Why the transfer is genuinely open in both directions.** The failure mode that produced OmniVoice's policy — perturbing a fused latent that feeds a nearest-neighbour RVQ lookup, flipping a discrete choice — is *exactly* qwen3-tts Base's shape, the codec encoder's 16-stage RVQ argmin over the reference clip. **But qwen3-tts already holds its whole codec half at the reference dtype under every profile** (`catalog.cpp:144-149`, `policy.cpp:350-357`), for an independent and measured reason: halving it made the codec **1.75× slower on CPU** because its convolutions run through im2col into a matrix multiply. So it is equally arguable that qwen3-tts's existing rule already covers the case and no new role is needed, and that arguing so *from OmniVoice* is the analogy §7 forbids.

- [ ] **Step 1: Attribute, using the method that produced the other family's answer**

Per `docs/quantization.md:355-361`'s shape: measure the codec encoder's RVQ drift against the reference under each profile Task 6 made cuttable, and attribute the remainder — is it convolution precision, or the transformer, or the projections? The per-column flip counts the validator already computes (`scripts/validate-qwen3-tts-codec_encoder.py:279-283`) and the per-stage chain deviations are the instruments. Use Task 3's redesigned statistic, so the flip rate is a first-class quantity rather than something hiding inside a percentile.

- [ ] **Step 2: Report which of three outcomes the attribution produced**

- **The exemption transfers**: something in the codec encoder *is* being quantized under a qwen3-tts profile and its convolutions are the drift source. Then a role is added and the reason is recorded at the classifier. **Note this outcome is only reachable if Task 6 put some `codec.encoder.*` tensor into a quantized role at all** — under `catalog.cpp:134`'s existing `"codec."` prefix rule, nothing there is quantized, in which case this outcome is unreachable and Step 2 says so rather than inventing a role to exercise.
- **The exemption does not transfer, because qwen3-tts's existing rule already covers it.** The codec half stays F32 under every profile for the speed reason it already records, no convolution kernel is ever packed (`policy.cpp:362-367`, `tests/qwen3_tts_quantization_policy_test.cpp:15-19`), and there is nothing left for a `ConvKernel` role to protect. **Record that as the measured answer, with the drift numbers that establish it** — not as an inheritance of the existing rule, which would be the same analogy in the opposite direction.
- **The measurement is inconclusive at the available flip-rate resolution** — e.g. the drift is dominated by the oracle's own bf16 codebook rather than by anything the profile did, which is the regime `code_agreement` already documents (port vs upstream-f32 **100.000%**, port vs oracle **52.97%/53.846%**, `:645-652`). Then the honest result is "the profile's contribution is below the measurement floor set by the codebook dtype", stated with the floor, and no role changes.

- [ ] **Step 3: Record, with the artifact**

```bash
git commit -m "qwen3-tts: the conv-exempt precedent measured against this family, not imported"
```

---

### Task 11: A backend coordinate for the codec-encoder gate, and a build-tree override for its upstream root

**Files:**
- Modify: `scripts/validate-qwen3-tts-codec_encoder.py`, `tests/CMakeLists.txt`

Two tooling gaps stand between Task 13 and a CUDA `codec_encoder` cell, and both are recorded.

**The validator has no backend coordinate.** It takes `--variant`, `--profile`, `--stage` (`:372-374`) and `load_gates` (`:315-334`) indexes `document["variants"][variant]["profiles"][profile]["stages"][stage]["probes"]` with **no `backends` step**. `scripts/validate-qwen3-tts-replay.py` already has the shape to copy — `resolve_tolerance_stage(tolerances, variant, profile, backend, stage)` at `:41-70`, reading `backends` for anything that is not CPU at `:69`.

**The golden test's upstream root is hardcoded.** `_synth_qwen3_tts_codec_upstream_root` (`tests/CMakeLists.txt:1268-1269`) is `${CMAKE_SOURCE_DIR}/build/qwen3-tts-codec-encoder-f32` with no override variable, so **a second build tree — `rel-dgx-spark`, which this plan needs for every performance figure — cannot point the golden test at its dumps** (carry-over §3.4, `:563-565`).

- [ ] **Step 1: Add `--backend`, following the replay validator's resolution shape exactly**

Default CPU, so every existing invocation is unchanged. `load_gates`'s **fatal-disagreement discipline** must extend to the new coordinate: a JSON/module-constant disagreement is fatal, which is what keeps a transcription from drifting (carry-over §4).

- [ ] **Step 2: Give the golden root an override variable**

Cache variable with the current path as its default, so existing configurations are byte-identical. Then check that the registration guard still means what it says: it is guarded on three uncommitted artifacts and **deliberately does not register a weaker form when the upstream-f32 root is absent** — *"a missing dump is a missing test, not a passing one"* (carry-over §4). An override must not become a way to register the test against nothing.

- [ ] **Step 3: Prove both are load-bearing**

Point `--backend` at a name with no sub-grid and confirm it fails rather than silently falling back to CPU (**presence**). Point the override at an empty directory and confirm the test does not register rather than passing (**presence**). Point it at a *faulted* dump tree and confirm the gate fails (**value**) — the validator is already known to exit 1 on a faulted tree and 0 on all three real cases (carry-over §4).

- [ ] **Step 4: Consider the filename, and decide rather than drift**

`scripts/validate-qwen3-tts-codec_encoder.py` uses an **underscore** where every sibling uses a hyphen, which is what broke `test_every_registered_validator_has_a_measured_stage` for six tasks unnoticed (carry-over §4). Renaming it means renaming the stage key `codec_encoder` in the tolerance file too, because the coverage test derives the validator stem from the stage name — that is a coordinated change across a committed contract. **Either do it here as one atomic change with the coverage test run before and after, or record the decision not to and why.** Do not leave it as an undiscussed inconsistency for a third plan.

- [ ] **Step 5: Gates and commit**

```bash
uv run --project scripts/envs/qwen3-tts --locked python -m unittest tests.python.test_tolerance_coverage -v
git commit -m "qwen3-tts: the codec-encoder gate takes a backend coordinate and an overridable upstream root"
```

---

### Task 12: CUDA placement for the two new graphs — or the measured decision not to

**Files:**
- Modify: `src/arch/qwen3-tts/model.cpp`, `src/arch/qwen3-tts/speaker-encoder-host.cpp`, `src/arch/qwen3-tts/codec-encoder-host.cpp`

**§7 says "The CUDA Execution Backend reuses Stage 1's wiring" — and Stage 1's wiring covers `codec.decoder.*` only.** The twin pass is `std::strncmp(tensor->name, "codec.decoder.", 14) != 0 → continue` (`src/arch/qwen3-tts/model.cpp:552`), and the reasoning at `:523-540` is explicit that the encoder is deliberately outside it:

> *"A Base package also carries 161 `codec.encoder.*` tensors, and twinning those would mirror 224,674,944 bytes onto the device for a graph that is not bound against the twin. **There IS a codec encoder graph now… and it is deliberately still outside this pass.** It is bound against `weights_context`, so it reads the CPU-resident originals wherever its caller's BackendPlan puts the computation, exactly as speaker-encoder.cpp's ECAPA graph does. Moving it onto an accelerator is a twin pass of its own, with its own prefix; it is not this one widened by one strncmp."*

Both new host paths hard-wire a CPU scheduler: `speaker-encoder-host.cpp:189` and `codec-encoder-host.cpp:468`, both `scratch.plan->create_cpu_scheduler(hash_size)` — against `model.cpp:795`, where the codec decoder passes `on_primary = impl.codec_context != nullptr`. The family record states the position plainly (`docs/porting/families/qwen3-tts.md:3263-3268`, `:3376-3378`): the new graphs *"have only ever run on CPU"*.

**So §7 does not decide this either way**, and neither does this plan in advance. What Stage 1 achieved is the benchmark to beat or to decline: on a 37-frame case at F16, same build, the codec decoder went **8.44 s → 0.24 s** and wall clock **20.69 s → 12.50 s** — 35× on the stage, 1.66× end to end — with placement proven by probes rather than asserted: six of eight bit-identical, only `audio.pcm` moving (0.999427 → 0.999404). Loading is ~7 s slower (457 MB copied plus context creation), making the split a **deployment choice**: a net loss for a one-shot 1.2 s utterance (17.9 s vs 10.9 s end to end) and a clear win for a load-once process (`docs/porting/families/qwen3-tts.md:1857-1932`).

**Two other constraints bear on this.** The discrete-outputs rule (`docs/backends.md:68-72`) holds any stage whose output is discrete, and every stage feeding it, on CPU; the 2026-08-08 exception (`:147-164`) requires that (1) the discrete output determines **content within a canvas whose shape was fixed by an earlier non-discrete stage before the first forward runs**, and (2) that shape invariance is **measured** across the whole Golden suite. The codec encoder's frame count is fixed by the waveform length in host arithmetic, not by the argmin — so clause (1) is arguable — **but the exception is a measurement, not an analogy, and nothing has measured it here.** And the hardware note: the GB10's CUDA device reports as **integrated**, so the runner takes the first non-CPU device rather than the first that calls itself a GPU (`:1908-1913`).

- [ ] **Step 1: Decide with numbers what the twin would cost before writing it**

224,674,944 bytes for the codec encoder (`model.cpp:523-540`), plus the speaker encoder's own share. Against Stage 1's measured ~7 s load penalty for 457 MB. **Measure the two new graphs' CPU wall time first, on a `Release` tree**, so the ceiling on any speedup is known before the twin exists. If a graph is 2% of wall clock, a 35× speedup on it is 2%.

- [ ] **Step 2: Report which of three outcomes this produced**

- **The twin is worth writing.** Then it is a **new pass with its own prefix**, per `model.cpp:523-540` — not the existing one widened by one `strncmp` — and the hard-wired `create_cpu_scheduler` calls become plan decisions, the way `model.cpp:795` already does it. **Placement is proven by probe, not asserted**, exactly as Stage 1 proved it, and the discrete-outputs exception is measured across the whole Golden suite or the RVQ argmin stays on the host.
- **The twin is not worth writing.** Then the new graphs stay CPU-only, `model.cpp:523-540`'s comment is updated to say Plan 4 measured it rather than that a later plan might, **no Base CUDA cell is added to the grid**, and Task 13 records that instead of a measurement. This satisfies §7 in the sense §7 actually states — Stage 1's wiring is reused, and the codec decoder does move on a Base package too. **It is a complete outcome, not a shortfall**, and it is recorded with the numbers that produced it.
- **The twin is written and does not pay, or does not place.** Then the measurement is recorded and the change is reverted rather than shipped dark. `docs/backends.md:490-497` gate 5 requires proving actual device placement; a twin that exists and does not place is worse than none, because the grid would then describe a backend nothing runs on.

- [ ] **Step 3: Gates and commit (conditional on Step 2)**

```bash
cmake --build build --target synthesize-check-unit
cmake --build build-sanitize --target synthesize-check-unit
```

If the twin lands, the unit tier already sweeps every device of type CPU / GPU / IGPU and builds the encoder fixture on each (`tests/qwen3_tts_codec_encoder_test.cpp:1320-1337`), so the *graph* is already CUDA-capable with synthetic weights — what is new is placing it through the real model. **Remember the node-count hazard in that file** before adding an assertion there.

```bash
git commit -m "qwen3-tts: <the measured placement decision for the speaker and codec encoders>"
```

---

### Task 13: Fill base/BF16/CUDA and base/F16/CUDA — with thresholds derived, not copied

**Files:**
- Modify: `tests/tolerances/qwen3-tts.json`

**Conditional on Task 12's outcome.** If Task 12 landed on "not worth writing", this task records that the Base variant carries no CUDA sub-grid and why, and stops — which is permitted, because `test_tolerance_coverage.py:152` reads `entry.get("backends", {})` and an absent key is legal. That is the same permitted hole CustomVoice's `Q8_MIXED` already occupies.

If Task 12 placed the graphs, two cells go in, each carrying all three stages: `base / BF16 / backends.CUDA` and `base / F16 / backends.CUDA`. That mirrors CustomVoice's shape, where BF16 and F16 have CUDA sub-grids (`:74-137`, `:198-261`) and Q8_MIXED does not.

- [ ] **Step 1: Derive the CUDA thresholds from TF32 scale — do not copy the CPU numbers**

**The CPU `codec.chain` gate of 1.0e-3 cannot be reused.** The codec encoder's own CUDA-vs-CPU output disagreement on a real clip is **5.20e-03 relative** — **~56× that gate**, and **~1.33 bf16 units** — so the chain gate would fail a correct CUDA run by construction (`tests/qwen3_tts_codec_encoder_test.cpp:1416-1442`). The file's own tolerance split is the precedent to follow: `causal_tolerance = type == CPU ? 1e-4f : 2e-2f` (`:1441`), a 200× ratio, with the fault margin stated at `:1438-1440` — a symmetric pad reads ~1.7, which is 17,000× the CPU bound and **86× the accelerator one**. **The accelerator bound keeps 86× discrimination, and that is the property a derived threshold must preserve.**

Two options exist and this task picks one on the measurement: derive a CUDA-specific threshold at TF32 scale, or change the comparison so the CUDA cell compares **CUDA-against-CPU-port** rather than port-against-upstream-f32. The second is a different question than the CPU cell asks and must be labelled as such in the cell's `description` if chosen.

- [ ] **Step 2: Expect the codes to move, and say so in the cell**

A 1.33-bf16-unit perturbation of the latent is larger than the codebook perturbation that alone flips 4.04–12.73% of codes (`spec:395-399`). **The 5% flip cliff may be crossed by the backend change alone on some cases.** Record the per-case flip rate under CUDA against Task 3's redesigned statistic. If Task 3 landed on its third outcome — the redesign could not hold ≥80× — then this cell is measured against the *old* statistic with `gate_scope_warning` intact and the caveat is written into the cell, not omitted.

- [ ] **Step 3: Three outcomes**

- **Both CUDA cells clear derived thresholds that keep their discrimination.** Commit them with the derivation stated.
- **CUDA clears the continuous gates but moves the codes past the flip threshold.** Then it is recorded as a *correctness-relevant* placement result: the graph runs and produces different reference codes, which is a change in the Voice, not a rounding difference. `docs/backends.md:490-497` gate 2 requires matching CPU tensors within tolerance — if a derived tolerance that admits this cannot also catch the injected fault, the placement does not ship and Task 12's decision is revisited.
- **A cell cannot be derived without a threshold so wide it catches nothing.** Then the cell is not committed, the reason is recorded, and the Base variant carries no CUDA sub-grid for that profile — the permitted hole, taken deliberately.

- [ ] **Step 4: Commit**

```bash
uv run --project scripts/envs/qwen3-tts --locked python -m unittest tests.python.test_tolerance_coverage -v
git add tests/tolerances/qwen3-tts.json
git commit -m "qwen3-tts: the Base CUDA cells, with thresholds derived from TF32 scale"
```

---

### Task 14: Performance and memory on a `Release` tree

**Files:**
- Modify: `docs/backends.md`, `docs/porting/families/qwen3-tts.md`

**This is a separate task from Task 13 because it is a separate build tree, and that separation is the whole point of the rule.** `docs/testing.md:455-467`: a `dev-*` preset proves correctness and never wall-clock time; a performance figure is measured on a tree whose `CMAKE_BUILD_TYPE` is `Release` and nothing else — the committed `rel-dgx-spark` preset, or the plain `cmake -S . -B build` default for CPU-only hosts. `rel-dgx-spark` is a measurement tree, **not** a registered CI gate (`:479-483`). `GGML_NATIVE` is off by default on both presets and measured 0.07% apart end to end (`:484-488`).

`docs/backends.md:490-497` gate 5 requires **latency, RTF and peak memory**; gate 6 requires **repeated-run and resource-cleanup** checks.

- [ ] **Step 1: Measure, on `Release`, with every figure carrying its build**

Latency, RTF, peak RSS and load time, for each profile that survived Tasks 7–8 and each placement that survived Task 12. **Every figure names the preset and the `CMAKE_BUILD_TYPE`.**

Two known contaminants to account for rather than discover:

- **The second BPE frontend costs +45 MB peak RSS**, measured at 5,594,676 → 5,639,716 KB (carry-over §3.4). Any Base peak-memory figure carries it; say so.
- **Generated length is build-dependent** — Release 24,960 PCM frames against RelWithDebInfo's 48,000 (carry-over §4). An RTF computed from a frame count measured on the other tree is wrong by 2×. **Compute RTF from figures taken on the same tree.**

- [ ] **Step 2: Run gate 6 for the new graphs**

Repeated-run and resource-cleanup checks. Note the ICL transcript-mismatch hang costs **475.91 s of CPU at `Release`** and returns `SYNTH_ERR_OUTPUT_LIMIT` with zero audio (carry-over §3.2) — a repeated-run sweep that includes a mismatched transcript loses eight minutes per iteration. That is a known behaviour with a recorded mitigation, not a regression.

- [ ] **Step 3: Three outcomes**

- **The Base package meets the backend gates and the numbers are worth publishing.** Record them in `docs/backends.md` and the family record, each with its build.
- **The numbers are unremarkable** — no speedup, or a speedup confined to a stage that is a small share of wall clock. **`docs/backends.md:499-501` is explicit that no minimum speedup is required**, so this is a supportable result: record it, and describe the backend as accelerated only if the comparison demonstrates an improvement, which is the same sentence's other half.
- **A gate is not met** — placement cannot be proven, or cleanup leaks. Then it does not ship as a supported backend for this variant, and the failure is recorded rather than the claim softened.

- [ ] **Step 4: Commit**

```bash
git commit -m "qwen3-tts: Base latency, RTF and peak memory, measured on Release"
```

---

### Task 15: The ICL listening audit — the first one this family has ever had

**Files:**
- Modify: `docs/porting/families/qwen3-tts.md`

**No ICL listening audit has ever run.** The family record says so (`docs/porting/families/qwen3-tts.md:3251-3253`), the carry-over says so twice (§2.2: *"Nobody listened to that audio. No ICL listening audit was run at all."*), and Plan 3's own validation-strategy paragraph was **corrected after execution** because it had asserted an audible property nobody had tested. Eight sites in the tree once made such claims; all eight were rewritten, and a tree-wide grep now returns only quoted-and-refuted corrections.

The design's standard (`spec:509-510`): *"A listening audit precedes ship. Per the standard set on the fourth family, the acceptance bar is audible quality; a tolerance table is not audible evidence."*

- [ ] **Step 1: Build the blind A/B page and offer it before concluding anything**

The project's standing method: a blind A/B comparison, not a self-assessment. Cover, at minimum: BF16 against each surviving profile; CPU against CUDA if Task 12 placed the graphs; ICL against x-vector on the same reference clip; and **the second recording Task 1 added**, since one speaker is not an audit either.

- [ ] **Step 2: Do not describe audio nobody heard**

The rule the carry-over installed after eight violations: no claim that misaligned or quantized output is "fluent", "plausible" or "in approximately the right voice" unless a listener said so, in this audit, on this build. A recorded audit result is `no_obvious_regression` **or** a named regression — not an inference from a passing tolerance table.

- [ ] **Step 3: Three outcomes**

- **`no_obvious_regression` across the audited set.** Recorded as evidence, with the profiles, backends and cases it covered. **It does not move the Validation Level** — `quality_evaluation` stays deferred per ADR 0017.
- **A regression is audible on a profile.** Then that profile does not ship, regardless of what its tolerance cells say. This is the case the design's "a tolerance table is not audible evidence" sentence exists for.
- **The audit cannot be run** — no listener available, or the ship decision is deferred. Then it is recorded as not run, `spec:532`'s gate ("audit recorded") is **not** met, and this plan does not claim it is. Publication was already gated on separate confirmation; this makes the gap explicit rather than quiet.

- [ ] **Step 4: Commit**

```bash
git commit -m "qwen3-tts: the first ICL listening audit, and what it covered"
```

---

### Task 16: Ship artifacts, the documentation obligations, and the fifth erratum's re-measurement

**Files:**
- Create: `scripts/hf_cards/qwen3-tts-12hz-0-6b-base.yaml`, a `docs/models/` page for the variant
- Modify: `docs/quantization.md`, `docs/porting/families/qwen3-tts.md`, `docs/superpowers/specs/2026-08-11-qwen3-tts-stage-2-design.md`

**Three of these start from nothing.** `ls scripts/hf_cards/` returns only `kokoro-v1-0`, `omnivoice-0-6b`, `qwen3-tts-12hz-0-6b-customvoice`, `vits-ljspeech` and `vits-vctk` — **there is no Base card specification**. And **`docs/quantization.md` has no qwen3-tts section at all**: `grep -i qwen3` returns one incidental hit at `:183` describing OmniVoice's Qwen3-0.6B backbone. Every qwen3-tts profile fact lives only in the family record.

- [ ] **Step 1: The card specification**

Model it on `scripts/hf_cards/qwen3-tts-12hz-0-6b-customvoice.yaml` — its `:5` names the published profiles, `:166` the default, and `:171-193` carries a package row per profile with digest and tensor-type census. **Only profiles that survived Tasks 7–9 appear**, each with its measured size and its digest. The default profile is a decision this task records with its reason, not a copy of CustomVoice's `F16`.

**No upload. No `hf` command. Nothing that touches the Hub.** The artifact is staged and the plan stops there.

- [ ] **Step 2: `docs/quantization.md` grows a qwen3-tts section**

It carries a `Status: Confirmed 2026-08-10.` line at `:3` which moves with the change. The section states what this family's `Q8_MIXED` means concretely (`:176-178`: *"Every other family's `Q8_MIXED` means 'this family's large matrix weights, halved or blocked'"*), what Task 6 ruled about the two new regions, what Tasks 9 and 10 measured, and **which profiles are buildable and deliberately unpublished** — `Q5_K_MIXED` already is (1035 MiB, RTF 0.82, talker logits cosine 0.9648), and Tasks 7–8 may have added to that list.

- [ ] **Step 3: Re-measure the fifth erratum's unverified figure, or say it is still unverified**

`spec:443-456`: the `278` table-alone code-divergence count *"was never reproduced, and no superseding value exists… Treat `278` as unverified rather than as a measurement, and re-measure the table-alone contribution before a later rung reasons from it."* **Plan 4 is that rung** — Task 10's attribution reasons about where codec drift comes from, which is the same question. Two acceptable landings:

- **Re-measured.** Record the value, its method and its provenance in the erratum, the way the committed float32 re-run replaced other figures from that day.
- **Not re-measured**, because no Plan 4 conclusion turned on it. Then say so explicitly in the family record and leave the erratum standing — the erratum's own structural argument does not depend on the count, and inheriting an unverified number silently is what the erratum exists to prevent.

Also fix, while here, the stale items the carry-over lists that a Plan 4 measurement will read: `conventions.json`'s "101 frames at most" (`base-ref-max` is 375, and the generated artifact reproduces the stale line, carry-over §3.4), and `CLAUDE.md:43`'s claim that CUDA builds default to strict FP32 with `SYNTH_CUDA_TF32=OFF` — the option does not exist (`docs/backends.md:35-56`; `tests/python/test_cmake_presets.py:102` asserts its absence).

- [ ] **Step 4: The family record's Stage 2 Plan 4 section**

Carrying: which of Task 6's three outcomes settled the runtime/tool contradiction and why; the packed-row table; the measured profile cells with their case ids, backend and **build**; Task 9's speaker-encoder answer and Task 10's codec-encoder answer, each **with the artifact that produced it** as §7 requires; Task 12's placement decision; Task 14's performance figures with their preset; Task 15's audit result; and what remains — Stage 3, the 1.7B variant, the CLI, `quality_evaluation`, and publication.

Update the `Status:` line on every document touched, per `docs/` conventions.

- [ ] **Step 5: Commit**

```bash
git add scripts/hf_cards/qwen3-tts-12hz-0-6b-base.yaml docs/quantization.md \
        docs/porting/families/qwen3-tts.md docs/superpowers/specs/2026-08-11-qwen3-tts-stage-2-design.md
git commit -m "qwen3-tts: stage the Base ship artifacts and record Stage 2 Plan 4"
```

The commit message states plainly that **nothing was uploaded and that publication remains a separate act requiring jiangzhuo's confirmation at the time.**

---

## Risks

**1. The inherited gate may not survive its own redesign, and every conclusion in this plan is read through it.** This leads because it is the risk that can invalidate the rest. The committed semantic p95 already **fails** a legitimate case at 0.3478 against 2.0e-2, and the failure is a property of the statistic — a 95th percentile crossing into a flipped-frame tail at ~5% flips — not of the port, which agrees with upstream-f32 at 100.000% on that case. Widening was refused because 0.35 would leave the injected fault 4.6× above the gate instead of 80×. **The redesign must keep ≥80× on the semantic branch and 3.5× on the acoustic branch inside a real 1.64× headroom, and nothing guarantees a masked statistic can.** Worse, the flip-rate threshold the redesign needs has **no calibration data**: four numbers from **one recording**, one of which (3.96/3.96) is the same clip measured twice. And the perturbations this plan is about to introduce all push in the direction that breaks it — CUDA-vs-CPU on the codec encoder is **5.20e-3 ≈ 1.33 bf16 units**, larger than the codebook perturbation that alone flips 4.04–12.73% of codes, and OmniVoice's codec-half profiles were **blocked** precisely because quantization flipped RVQ tokens. **Mitigation:** Task 1 acquires a second recording *before* the statistic is committed, so the threshold is not calibrated on data Tasks 7, 8 and 13 then invalidate; Task 2 makes the two p95 implementations agree under a registered test before either is redesigned; Task 3 names its third outcome as "stop and re-scope" rather than "ship a weaker gate". **Residual risk:** if Task 3 lands on that third outcome, Tasks 7, 8 and 13 run against the old gate with `gate_scope_warning` intact, and every cell they commit carries that caveat explicitly — a smaller plan, honestly labelled, rather than a full one resting on a gate that does not discriminate.

**2. The blocker's resolution may not be reachable without touching a published grid.** `catalog.cpp:134`'s `"codec."` prefix test is a shared function, and the three published CustomVoice packages' load behaviour is a contract with digests already on the Hub. If the only way to make runtime and tool agree about `speaker_encoder.*` moves that function's behaviour for CustomVoice too, the reconciliation cannot be done quietly. **Mitigation:** Task 6 Step 3 names this as a possible finding and requires it be reported before the change is written; the byte-identity proof for all three digests is a Global Constraint, not a task step, so it cannot be skipped by a task that did not anticipate needing it. **Residual risk:** the honest resolution may be a per-tensor split with a recorded reason at each classifier arm, which is more code than either uniform answer and is the outcome Task 6 lists third for exactly that reason.

**3. §7's own hints are not answers, and the plan is arranged so they cannot become answers by default.** §7 says the speaker encoder is "small enough that quantizing it is unlikely to pay" and that the codec encoder "is convolution-heavy, which is the shape that produced the conv-exempt policy in another family" — and then forbids importing either precedent by analogy. **The failure mode is subtle: agreeing with §7's prior without measuring it looks identical, in the committed artifact, to measuring it and finding §7 right.** **Mitigation:** Tasks 9 and 10 each require the result be "recorded with the artifact that produced it" — package digest, case ids, build, command line — which is the discriminator. Task 10's second outcome is written specifically to name the opposite-direction analogy (inheriting qwen3-tts's own existing codec rule without measuring it) as equally forbidden.

**4. Measuring the two new graphs on CUDA is new wiring, not reuse, and §7 does not authorize it.** §7 says the backend "reuses Stage 1's wiring", and Stage 1's twin pass covers `codec.decoder.*` only, by a decision recorded at `model.cpp:523-540` with its byte cost. Reading §7 as commissioning a twin pass for the two new graphs is a reading, not the text. **Mitigation:** Task 12 measures the CPU cost of the new graphs *before* writing a twin, so the ceiling on any speedup is known; its second outcome — the graphs stay CPU-only, no Base CUDA cell exists, and Stage 1's wiring is reused in the sense §7 actually states — is a complete result and is written as one. **Residual risk:** taking that outcome leaves the Base variant with no CUDA sub-grid, which is permitted by the coverage rule but is less than a reader might expect from a plan whose title says "backends". The family record must say why, in numbers.

**5. Two build trees, two different outputs of the same code.** Correctness runs on `dev-*` (`RelWithDebInfo`, ggml at `-O2`) and performance on `Release` — and this project has already measured that the *same* input, seed and backend give **24,960 PCM frames on Release and 48,000 on RelWithDebInfo**, each reproducible, because the stop decision differs by build type. An RTF computed across the two trees is wrong by 2×; a frame count quoted without its build is meaningless. **Mitigation:** the constraint is stated globally rather than per-task, Task 14 computes RTF from same-tree figures only, and Task 11 gives the golden test's upstream root an override so a second build tree can drive it at all. **Residual risk:** `rel-dgx-spark` is not a registered CI gate, so nothing enforces that a future figure came from it — the discipline is the commit message and the card, not a test.

**6. Blockers that stop measurements before they start.** Three are known and carried: the missing `trim_seconds` driver flag, without which the only two cases that would add an x-vector data point cannot be driven at all and `replay` cannot honestly widen (Task 4); the two p95 implementations with no registered cross-check, which matters more than usual in a plan whose conclusions are p95s (Task 2); and the hardcoded `_synth_qwen3_tts_codec_upstream_root`, which blocks a Release tree from driving the golden gate (Task 11). **Mitigation:** all three are tasks, all three land before the measurements that depend on them. **Residual risk:** the carry-over lists further deferred minors that could surface mid-plan — `src/voice-profile.cpp`'s ICL branch unreachable from unit tests, no numerical gate over `run_synthesis`'s prefill assembly, I12's writer rule with no isolating arm, I10's `reference_text_tokens` process-abort path, and `quantizer.output_proj` resolved but never read on the encoder path (which Task 6 must assign a role to regardless). Each is recorded in this plan's carry-over rather than absorbed silently.

**7. Publication pressure at the end of a plan whose deliverable is "artifacts prepared".** The card, the digests and the docs will all exist, correct and complete, and the remaining step will look like one command. **It is not authorized by this plan, by an approved plan, by "go ahead", or by the fact that the work is finished.** Task 16 stages and stops; the commit message says so; and asking is a separate act, naming the repository, at the time.

---

## What Plan 4 does NOT deliver

Stated plainly, so nothing here is read as more than it is.

- **Publication.** `spec:532` and `spec:561-562`. Artifacts are staged; nothing is uploaded and nothing asks to be.
- **Any movement of the Validation Level.** `port_validated` plus a listening audit is the delivery bar; **`quality_evaluation` stays deferred per ADR 0017**. Task 15's audit is evidence, not a level.
- **A guaranteed profile set.** §7 assigns none, and `docs/backends.md:499-501` requires measurement without a minimum speedup. "This profile does not pay" and "this profile is blocked" are results this plan is written to be able to produce.
- **A guaranteed CUDA cell for the Base variant.** Task 12's second outcome leaves the two new graphs on CPU, which is what §7's sentence about reusing Stage 1's wiring actually describes.
- **Stage 3 / Description Text**, the **1.7B Base variant**, **multi-clip enrollment**, **Native Streaming Synthesis**, **Voice Conversion** — §10 non-goals.
- **The CLI path.** `examples/cli/` untouched.
- **Any converter change or package re-cut**, unless a task proves one necessary and carries the byte-identity proof for the three published CustomVoice packages with it.

---

## Self-review

**Spec coverage.** §7 is the whole plan. Its first sentence — measure rather than assume Stage 1's profiles carry over — is Tasks 6, 7 and 8, with Stage 1's own figures quoted as orientation and explicitly labelled as CustomVoice measurements rather than Base evidence. Its speaker-encoder clause is Task 9 and its codec-encoder clause is Task 10, each with three outcomes and each required to record the artifact, because "recorded with the artifact that produced it" is §7's own acceptance condition. **"Neither precedent is imported by analogy" is enforced in both directions**: Task 10's second outcome names inheriting *qwen3-tts's own* existing codec rule without measuring it as the same error in the opposite direction. §7's CUDA sentence is Task 12, which reads "reuses Stage 1's wiring" against `model.cpp:523-540` and declines to promote it into a commission for new wiring. Its build-naming sentence is a Global Constraint and the reason Task 14 is separate from Task 13. §8's plan-sequence row supplies the listening audit (Task 15) and the model card and artifacts (Task 16). §10's non-goals are the "does NOT deliver" section, quoted rather than paraphrased. The **fifth erratum** (`spec:443-456`) is Task 16 Step 3, with both landings — re-measured, or explicitly still unverified — written as acceptable, since the erratum's own argument does not turn on the count. The **fourth erratum's** generalized rule governs every codec-encoder threshold Tasks 7, 8, 10 and 13 touch.

**Carry-over coverage.** §1.3's ruling is Task 3, with the fault-injection margins (1650× / 80.05× / 3.5×) written as a constraint the redesign must meet rather than as background, and the corrected values used rather than the M1–M3 slips. §3.3's "three cases is one recording" and its own recommendation — *"a second reference clip and a second speaker are the highest-value coverage Plan 4 can add"* — is Task 1, promoted to first place because Task 3's threshold must not be calibrated on data Tasks 7, 8 and 13 invalidate. §3.4's deferred minors: `trim_seconds` is Task 4, the p95 cross-check is Task 2, the stale `conventions.json` line is Task 16, the `_synth_qwen3_tts_codec_upstream_root` hardcoding and the validator's underscore filename are Task 11, `quantizer.output_proj`'s unread role is Task 6, and the +45 MB BPE frontend is a named contaminant in Task 14. §3.1's recurring bounded-at-create/unbounded-at-load shape is a Global Constraint, with the third instance's lesson attached — the instrument that caught the first two is blind to it. §2.3's two classes of unfalsifiable check are a Global Constraint, with the "which dimension does this inversion perturb" question stated as the operative one. §4's baseline is carried, **and the eight VITS integration failures the commissioning names are carried as unverified with Task 1 Step 0 to establish the real list**, because they are recorded nowhere in the tree and a number I cannot source is not a specification.

**Where I deviated from the research's suggested seams, and why.** Three places. Its sixteen items become sixteen tasks with a different composition: I **added** a second-reference-recording task at position 1, which the research raised only inside its "riskiest unknown" as a sequencing argument (*"which argues for acquiring a second reference recording before, not after, the statistic is committed"*) — making it a task is the only way that sequencing is actually enforced. I **merged** its items 15 and 16 into one documentation-and-artifacts task, following Plan 3's shape, because the card, the `docs/quantization.md` section, the family record and the spec erratum are one editorial pass over the same findings. And I **kept** its items 12 and 13 separate — the CUDA cells and the performance figures — against the temptation to merge them, because they are measured on two different build trees and merging them is exactly how a `dev-*` figure ends up in a card. Net: 16 tasks, the top of the research's 13–16 estimate, and the CUDA half (Tasks 11–13) is the compressible part the research identified, which Task 12's second outcome compresses on measurement rather than by prior decision.

**One research claim I could not confirm and therefore did not carry as a specification.** The research states that the speaker encoder's stem convolution has a packed row that no Q8_0 block divides, offering 400 and 640 as the values for 80- and 128-bin front ends. **640 is 32 × 20**, and this package declares 128 mel bins (`spec:65`; `mel_bins` is read from package metadata at `weights.cpp:581`). The claim holds at 80 bins and not at 128, and Q5_K's 256-element super-block is a separate and much tighter constraint. Task 6 Step 2 therefore requires the row table be **derived from the actual package and committed**, and explicitly instructs the implementer not to restate either the research's arithmetic or mine as a finding.

**Placeholder scan.** This plan prescribes **no numeric tolerance**. Every threshold in Tasks 3, 5, 7, 8, 13 is measured by the task that commits it. Every figure quoted above is either a committed in-tree value with its `file:line`, a measurement recorded in the carry-over or family record with its provenance, or the one command run against the real tool on 2026-08-15 — whose output is quoted verbatim.

**Unknowns carried as unknowns**, each with the task that establishes it and a named branch for every answer rather than a preferred one:

1. **Whether the redesigned semantic statistic can hold ≥80× discrimination** (Task 3), with "stop and re-scope" as an explicit third outcome rather than an implied failure.
2. **Whether a flip-rate threshold is calibratable at all** on the available recordings (Tasks 1 and 3), with "recorded but ungated, and the reason stated in the gate's own note" as a legitimate landing.
3. **Which of the runtime and the tool is right about `speaker_encoder.*`** (Task 6), with three outcomes including a per-tensor split, and with "the reconciliation cannot be made without moving the published CustomVoice grid" named as a possible finding rather than an obstacle to route around.
4. **Whether either new profile pays on the Base package** (Tasks 7–9), where "does not pay" and "blocked" are results, not failures — `Q5_K_MIXED` is the in-tree precedent for a buildable, deliberately unpublished profile.
5. **Whether the conv-exempt precedent transfers** (Task 10), with a third outcome for the case where the profile's contribution sits below the measurement floor the oracle's own bf16 codebook sets.
6. **Whether a CUDA twin for the two new graphs is worth writing** (Task 12), measured against their CPU share of wall clock before any twin exists, with "no Base CUDA sub-grid" as a complete and permitted result.
7. **Whether the fifth erratum's `278` needs re-measuring at all** (Task 16), with "not re-measured, because no Plan 4 conclusion turned on it, and the erratum stands" as an acceptable landing.

None of the seven is written anywhere in this plan as a decided fact.
