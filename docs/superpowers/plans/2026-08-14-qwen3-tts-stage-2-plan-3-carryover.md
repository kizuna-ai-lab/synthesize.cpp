# Qwen3-TTS Stage 2 Plan 4 — Carry-Over Ledger from Plan 3

Written at Plan 3 close (2026-08-14), from the branch's SDD ledger and the
whole-branch review of `qwen3-tts-stage-2-plan-3` @ `e99205b` (39 commits on
merged main `5f3e700`, 58 files, +19785/−870). Like its predecessors
(`docs/superpowers/plans/2026-08-13-qwen3-tts-stage-2-plan-2-carryover.md`,
`…2026-08-12-qwen3-tts-stage-2-plan-1-carryover.md`), this file exists because
the ledger lives under `.superpowers/`, which is gitignored and does not survive
the merge. **This one was written before the workspace was deleted**, which is
the whole difference between it and Plan 2's — see §0.

Plan 3 delivered the transcript-assisted (ICL) clone path: the codec-encoder
graph over the package's 161 emitted `codec.encoder.*` tensors, the host-side
RVQ, the reference-transcript turn, the two-track ICL prompt and its
`min(T1,T2)` alignment, the `IclProfile` payload and its schema-v1 envelope
extension, the public seam flipped to `OPTIONAL`, synthesis dispatch, and the
tolerance grid that gates all of it. See `docs/porting/families/qwen3-tts.md`,
"Stage 2: Base Package, Plan 3", for the measured facts; this file is the
decisions and the debt.

**What Plan 3 explicitly did not deliver.** Quantization Profiles and the CUDA
Execution Backend are Plan 4's, by the design's §7 — Plan 3 measured no
Quantization Profile, claimed no performance number, and added no
`Q8_MIXED`/`F16`/`CUDA` cell to the grid. **Nothing in it moved the Validation
Level**: `quality_evaluation` stays deferred per ADR 0017, and **no listening
audit was run for ICL at all**. It published nothing.

---

## 0. Process failures, recorded so they are not repeated

**A completion report for Task 13 was fabricated.** It named commits `65da8b6`
and `dc9a2ba` — **not valid git objects** — a test `…clone-icl-real` absent from
the tree, a claim that the test *fails* under a one-frame rotation (it passes),
and a 40-byte `IclProfile` leak "found and fixed". The review disproved the leak
three ways: `synth_voice_profile_create_from_reference` never calls serialize;
the payload is `make_shared` into a `shared_ptr<const void>` with no raw `new`;
and `IclProfile` embeds a whole `XVectorProfile` plus two vectors, so it cannot
be 40 bytes. `git reflog --all` shows a linear history with no window for those
commits. Nothing from it reached the tree. **Treat a report as a lead, never as
evidence: verify every SHA exists before recording anything as done.**

**Several controller-supplied "facts" were wrong and were corrected mid-plan by
implementers who checked.** Three, each caught by measurement rather than
argument:

1. **The sliding window** (Task 4). The brief asserted the encoder transformer
   is windowed. `create_causal_mask` (`modeling_mimi.py:1101` →
   `masking_utils.py:795`) uses the plain causal mask unconditionally and never
   reads `config.sliding_window`; the windowed variant is a different function
   `modeling_mimi.py` never imports; only `MimiFlashAttention2` reads the window
   and flash-attn is not installed. The wrapper's sliding-window code belongs to
   the codec **decoder**, a different class — the likely source of the confusion.
   Measured on the real 96 tensors at `[1,400,512]`: the default is
   bit-identical (0.000e+00) to explicit plain-causal and 4.64x rms from
   explicit sliding(250).
2. **The language tag** (Tasks 6 and 10). The controller stated the accepted tag
   is `"english"` and that a comment calling `"en"` package-declared was false.
   **Both wrong.** `model.cpp:178-196` deliberately bridges the package's full
   names to BCP-47 (`{ "en", "english" }` ×10, with a comment saying so),
   `model.cpp:365` publishes the BCP-47 side, `declared_language` matches
   exactly, and `synth_model_get_language` returns `en de es zh ja fr ko ru it
   pt` measured. Three vocabularies had been conflated: package metadata, the
   public interface, and the Golden Manifest's description of the **oracle's**
   vocabulary. The comment at `tests/qwen3_tts_base_load_real.cpp:250-257` was
   right and was left alone.
3. **The hang's provenance** (whole-branch review) — see §3.2.

Two rules follow from (2). **A correction that overturns a specific existing
comment needs evidence too**; this error originated as a "correction" made
during pre-flight repair. And **a partial correction is worse than none**: the
first fix (`def1a7b`) repaired one paragraph while `plan:692` still said the
opposite in bold, and the document then looks reviewed. Completed in `e2b2e8a`.
The same shape recurs as MB3 in §3.1.

**The brief extraction dropped files the plan listed, in six tasks running**
(Tasks 5–10). Task 10's case was located: the plan lists `weights.h` and the
extraction dropped it. Stale line citations in briefs were endemic — Task 6
found 3 of 5 stale, Task 8 three more, Task 9 four more at ~160 lines off.
Task 7's fix round solved the class properly for its own file: every inversion
was re-run by a script that prints the binary's own `check failed` lines, and
line numbers are emitted via `__LINE__` so citation drift cannot recur.

**Task 13's shape, worth naming because every step looked done.** Task 12's
three committed tolerance thresholds were **inert** until Task 13.
`synthesize-tolerance-coverage` checks metadata consistency only — that
validators and stages line up — never that any threshold is met, and the
validator that checks them was invoked by nothing. Thresholds committed,
coverage test registered, and **nothing asked "if this broke, who would know?"**

Disclosed and checked: Task 8's implementer destroyed uncommitted `profile.cpp`
work with `git checkout --` mid-inversion and rebuilt it. The review audited the
reconstruction and found it complete.

## 1. Controller rulings

### 1.1 The plain-equality gate on reference codes is dropped

**Ruling, 2026-08-13, from a measurement.** The design's §6 prescribed plain
equality on the discrete reference codes. jiangzhuo chose **stage-wise float32
artifacts at a bf16-derived tolerance plus the dequantized reconstruction**,
over the two alternatives — baking bf16 codebooks into the converter, or
absorbing flips via an `alternate_grids` escape. Recorded as the spec's fourth
erratum.

**Evidence: 146 clips / 49,507 frames / 792,112 frame-stage decisions.** The
f32-vs-bf16 codebook differs on **4.04%** of codes for the argmin alone and
**12.73%** with the f32 table used throughout the RVQ, at a per-decision flip
rate of **0.624%**, on 136 of 146 clips and at every stage including the
semantic one. Upstream is not self-consistent with itself: loading with
`dtype=bfloat16` (model card and demo) versus omitting `dtype` (fp32 default)
disagrees on **760 of 1616 codes, 52.970%**.

Two things that cost something to establish:

- **Task 1's "codes match byte-for-byte" measured determinism, not table
  sensitivity.** Both scripts loaded bfloat16, so it was bf16-vs-bf16 and could
  never have seen this.
- **The "competing entries move together" hypothesis is refuted**:
  ‖δᵢ−δⱼ‖/‖δᵢ‖ ≈ √2.

**Task 5 proved the ruling was necessary, not merely prudent.** The port
reproduces upstream-f32 on the discrete path at **100.000% code agreement**
(1616 and 208 codes), while agreeing with the bf16 oracle on only 52.97%/53.85%
— and upstream-f32 disagrees with the oracle on *exactly those same codes*.
**The dropped equality gate would have failed a correct port on ~47% of codes.**

**`794 of 1616` is stale and is in five places.** It was a pre-execution
2026-08-13 throwaway measurement, superseded by 760/1616. It survives at
`docs/porting/families/qwen3-tts.md:2945`,
`docs/superpowers/specs/…design.md:413` and `:435`, and
`docs/superpowers/plans/…plan-3-icl-path.md:24` and `:452` — fifteen lines from
the correct figure in the first of those.

### 1.2 The reconstruction gate is a percentile, not a maximum — and the framing assumption behind it was refuted

The ruling was framed on the assumption that **a near-tie flip moves the
reconstruction by about the tie margin**. The measurement it asked for refuted
that. For the reconstruction vector, one flipped code displaces it by about a
**full codebook-row separation**: median |ΔRecon|/‖eᵢ−eⱼ‖ = **1.21**, which is
12.9% of the norm at the median and 32.8% at the max, and later RVQ stages do
not absorb it. The assumption holds only for the **residual norm** (2.4%), which
the gate does not compare. Deviation is bimodal: median 3.2e-3 (bf16 scale),
p95 1.9e-1 (the flip tail).

**Consequence:** the gate is a **p95 of per-frame relative L2**, per branch,
never averaged — never a maximum at bf16 scale, which no correct port could
satisfy. The choice of gate stands; only its shape changed.

**Two keys, not an average**, because the branches are an order of magnitude
apart: rms 13.5 (semantic) vs 3.11 (acoustic), explaining 43% vs 85% of target,
so the bf16 scale lands at ~0.80 vs ~0.18 absolute L2.

**The gates Task 12 committed**, from Task 5's measurement:

| gate | threshold | measured | headroom | injected fault |
| --- | --- | --- | --- | --- |
| chain `rel_absmax` | 1.0e-3 | 9.303e-05 | 10.7x | 1.649 (1650x) |
| semantic p95 | 2.0e-2 | 0.004103 | 4.9x | 1.601 (80.05x) |
| acoustic p95 | 5.0e-1 | 0.2491 | 2.0x | 1.741 (3.5x) |

The brief's predicted 0.18 absolute acoustic bound was **~39x too small**; 0.50
is where the acoustic branch clears 2x in both directions. Published headroom
figures of "10.8x", "78x" and "87x" are arithmetic slips (M1–M3) — the correct
values are above.

### 1.3 The semantic gate sits on a ~5% flip-rate cliff — Plan 4 redesigns the statistic

Running the port on `base-ref-max` fails the semantic gate: **p95 0.3478
against the committed 2.0e-2**. **This is not a port defect.** Codes agree with
upstream-f32 at **100.000% (0 of 6000 disagree)** and the chain gate passes
*tighter* than the calibration case (8.431e-05). The p95 only enters the flipped
tail above roughly 5% flips: the three calibration cases sit at 0.00/3.96/3.96%
and `base-ref-max` at **5.33%**, an 84.8x jump in the statistic.

**Widening was refused and the refusal is upheld**: a 0.35 threshold would leave
the injected fault 4.6x above the gate instead of 80x, destroying
discrimination. Recorded as a `gate_scope_warning` in
`tests/tolerances/qwen3-tts.json`; `cases` stays 3; tree green.

**For Plan 4: the statistic needs redesign — gate the flip rate separately from
the reconstruction error on non-flipped frames.** The real evidence for
`base-ref-max` is the 100% code agreement, not the p95.

Also corrected in-task, because it took two review rounds to establish in Tasks
4–5 and will be re-conflated otherwise: a published "95.96% code agreement"
where measured is ~53% — **4.04% was the codebook-isolated floor, never the
rate** — and 33 keys named `observed_port_vs_oracle_*` that actually held
upstream-f32-vs-oracle values, since renamed.

## 2. The method that earned its keep, and what the gates defend against

### 2.1 Decompose against upstream's own module run in float32

**The bf16 dumps cannot settle correctness alone.** Task 4 established the
method and it is mandatory for every later graph task: run upstream's own module
(`MimiEncoder`, `MimiTransformerModel`, the prompt builder) in **float32**, and
decompose the deviation into *port vs upstream-f32* and *upstream-f32 vs the
bf16 oracle*.

The numbers that made the case, all under one normalizer
(`max|Δ|/absmax(ref)`, published beside a single-rounding floor):

| comparison | port vs upstream-f32 | upstream-f32 vs bf16 oracle |
| --- | --- | --- |
| codec-encoder chain (33 taps) | 4.7e-07 … 9.3e-05 | 0.0 … **0.9896** |
| ICL prompt, codec track | 0.0 … 4.2e-09 | 2.232e-03 … 2.338e-03 |
| ICL prompt, text track | 1.28e-04 … 1.73e-04 | 2.307e-03 … 5.807e-03 |

Port-vs-upstream-f32 is **17.3x to 3953x below one bf16 rounding** — no defect;
the entire residual against the oracle is the oracle's own arithmetic. The ICL
text track sits 23–30x below one bf16 unit roundoff (3.906e-3). Worst chain stage
is `transformer_l7` at **0.024x** one bf16 unit, 42x below a single rounding.

Two things to carry into Plan 4's own comparisons:

- **`transformer_l7` (135–265x the floor) and `downsample` (53–60x)** are the
  two stages to watch — the two whose absmax collapses 35x. l0–l6 sit at 1.6–21x.
- **The family doc's stated bf16 range, "0.0 … 0.41", is understated 2.4x**
  (review I4). The grid's own maximum over the 33 taps is **0.9896**
  (`rvq_residual_s14`; then s08 0.9475, s15 0.9307, s07 0.9278). 0.41 is only
  the maximum over the *non-RVQ* taps. **A later plan sizing a bf16-scale bound
  off the stated range sets it 2.4x too tight.** The lower bound 4.7e-07 appears
  nowhere in the grid either; the smallest non-zero per-stage worst is 8.021e-07.

**bf16 numerics, settled and published rather than reasoned:** the relative
bound is **2^-8 = 0.00390625** (bfloat16's unit roundoff), *not* 2^-9 (its
half-ulp). Both numbers are real and a downstream tolerance set at 2^-9 fails.
It is published in `numerics` blocks in **both** `alignment.json` and
`prompt_conventions.json`, with the zero-block residue recorded as 0.0 so a
relative-only tolerance is safe. **Take it from the conventions file, not from
any report and not from here.** A related trap: consumers must round an f32
track sum back through bfloat16 before comparing to `icl_embed.f32` — a raw f32
add deviates by construction, and the resulting tolerance failure is very hard
to diagnose from the far end.

Stage comparisons feed `codec_encoder/waveform.f32`, never the WAV: the bf16
input cast alone is 2.6% of rms. Task 13's end-to-end tier deliberately feeds
the WAV instead, because that is what a caller hands the library — see §3.3.

### 2.2 What the numerical gates are defending against, stated to exactly what was measured

A **one-frame rotation of the codec track** in `append_icl_block`:

- the public seam returned **`SYNTH_OK`**;
- output moved **24,960 → 19,200 PCM frames** (1.0400 s → 0.8000 s), peak
  0.6808 → 0.7696;
- **exactly one new failure among 100 unit tests** (`icl-prompt-test`);
  `icl-prompt-real` failed on the codec track at **7.945e-01** against a 2.0e-2
  tolerance;
- the end-to-end tests **passed** — because they assert nothing about alignment.
  Their assertions are purely differential against the x-vector run, so a
  rotation moves both arms equally. That is a property of the tests, not of the
  audio.

The reviewer reproduced all of it in an isolated copy to four significant
figures, and Task 13 re-ran it independently.

**`frame_count` is on the public API surface**, so a caller sees 0.80 s where
1.04 s is correct. The accurate phrasing, settled after two rounds, is
**observable, never detectable**: the evidence is there and nothing looks at it.
`tests/qwen3_tts_icl_real.cpp:73-74` says outright that an exact frame-count
assertion was available and would have made the rotation detectable.

**Nobody listened to that audio.** No ICL listening audit was run at all. Do not
describe the misaligned output as fluent, plausible, or in approximately the
right voice — none of that was established. Seven sites in the tree used to make
such claims (review I1), plus an eighth the review did not list; **all eight were
rewritten in `5f8213b` and `ae63d12`**, and a tree-wide grep now returns
only quoted-and-refuted corrections. One of them,
`tests/qwen3_tts_icl_prompt_test.cpp:4-6`, additionally claimed the rotation
leaves the length right — which this branch measured false — and cited
`src/arch/qwen3-tts/talker-host.h:110-111` for it, where the text actually reads
*"in the wrong voice or the wrong language."* All three of that comment's
defects are corrected in place, with the old text quoted so the correction is
auditable. **The argument stands without any
audible claim:** a defect that the numerical gates caught and nothing else did
is the entire point.

### 2.3 Two classes of unfalsifiable check the standard discipline misses

The plan's standing rule — delete the rule under a green test, confirm the test
fails, restore — is necessary and **not sufficient**. Plan 3 found at least
seven more checks unable to fail (running total across this repository reached
fifteen by Task 10), in two classes single-deletion inversion structurally
cannot see.

**Mutual masking.** Deleting either check of a pair changes nothing; deleting
both lets a sealed forgery through. Task 9 found **three pairs** in
`src/arch/qwen3-tts/profile.cpp`: prescan/post-parse `kind` (`:853`/`:1151`),
prescan/loader tensor count (`:867`/`:1179` — **which had no test at all**), and
the SET/COUNT pair. `prescan_buffer` now carries a `MUTUAL MASKING` comment and
**each masked rule names its partner at the check** (`profile.cpp:800-1075`).
Task 10 found a fourth at the language seam — `valid_bcp47_shape` masked by
`declared_language`, one direction only — named at both production and test
sites; the implementer had run the single deletion and misread the status-only
movement as absence of a pair. A fifth lives at the request seam between
`validate_speaker_sources` and `reference_is_well_formed`, which is why
`tests/qwen3_tts_icl_request_test.cpp` exists. The whole-branch review searched
for an undisclosed pair and found none.

**Inversion sets that all perturb one dimension.** Task 6 shipped **eight**
inversions that *all* changed the token sequence's **length**; none exercised
the element-wise off-by-one the task existed to prevent. Inversion 9 (slice
shifted one left, length-preserving) does: `id 0 is 198, oracle has 32313`. The
audit that followed found two assertions **no inversion could fire at all**,
both removed. **Ask which dimension each inversion perturbs, not how many there
are.** The three lists that name their dimensions and each span four or more of
kind/presence/value/order/length are
`tests/qwen3_tts_profile_test.cpp:436-453` and `:2199-2275`, and
`tests/qwen3_tts_codec_encoder_test.cpp:717-732`;
`codec_encoder_test.cpp:101-129` and `tests/qwen3_tts_icl_real.cpp:29-43`
disclose their own collapse.

**A standing hazard at the same location:** the node-count assertion in
`tests/qwen3_tts_codec_encoder_test.cpp` masks seven inversions' failure
signals. All seven are enumerated in that file with their isolated results —
**read that before adding an assertion there.** (Node counts are not
length-independent: the replicate pad grows 4 nodes when `extra_padding` ≠ 0, so
two counts are pinned, 445 whole-frame and 449 ragged.)

The best single artifact of this discipline is the masking audit at
`tests/qwen3_tts_icl_real.cpp:115-167` — with one entry backwards, review I11 in
§3.1.

## 3. Carried items — things Plan 4 must revisit, not merely extend

### 3.1 The whole-branch review's must-fix set — CLOSED, with what it cost

**Status: all four blockers closed in `5f8213b`, re-reviewed and verified by
running.** This section is kept because the *shape* of MB1 recurred a third time
and Plan 4 should expect a fourth.

The ceiling now applied is the package's own: `ceil(min(max_frames_per_clip,
max_total_frames) / hop_length)` = **375**, both factors already present in the
`HParams` the loader takes — not a new constant, a constraint that was already
there and unused. Re-review confirmed the derivation against both creation
checks and the trim divisor, computed 375 at runtime, and measured the forged
envelope refused at **+0 KiB**. With the clause deleted in an isolated copy the
integration run reached **VmHWM 21,578,040 KiB and was still rising** when it was
killed. `base-ref-max` sits exactly at 375, so **no Golden Manifest case is
rejected** by the new bound.

The measurement that justified it: a **1,048,384-byte envelope built by this
project's own writer**, declaring 16,300 frames (43.5× the ceiling), drove peak
RSS to **24,447,664 KiB ≈ 23.3 GiB**, with load returning `SYNTH_OK` and the
process still running after 25 minutes. **A status assertion cannot see this** —
the arm has to measure peak RSS, as Task 9's did.

**The recurring shape, three times on this branch.** A count is bounded where a
value is created and unbounded where it is loaded:
1. Task 9 — codes and ids sized from unchecked counts before the truncation
   guard (1.48 GiB). Fixed structurally: the count is unobtainable until the
   bytes behind it exist.
2. MB1 — `declared_frames` bounded at creation, not at load (23.3 GiB,
   quadratic through `prefill`).
3. Found by the re-review looking for a third: `reference_text_ids`' count was
   unbounded at load while creation bounded it by `max_input_tokens`. Closed in
   `ae63d12` against `hparams.max_input_tokens`, with 0 refused rather than read
   as "no limit".

**And the third one taught something the first two did not: the instrument that
caught MB1 is blind to it.** A 1,059,520-byte envelope declaring 262,000 ids
(255.9× the bound) loads `SYNTH_OK`, **synthesis completes**, and peak RSS
reports **+0 KiB** — the amplification is linear and sits under the multi-GB
high-water mark the frames arm had already set. What actually moves is **wall
time: 34.4 s → 67.3 s**. So on this path the *status* assertion is the
load-bearing one and RSS is a bound that merely happens to hold, and the check
says so in the file. **Copying MB1's peak-RSS arm here would have produced a
guard that cannot fail** — the same defect class, arrived at by reusing the
right fix for the wrong reason.

**Plan 4: when you add a field to the envelope, the question is not "is this
value valid" but "is this count bounded by the same thing that bounds it at
creation".** Two of the three were found only because someone went looking for
the shape rather than for the instance.

- **MB1 CRITICAL — `declared_frames` is bounded at creation and not at load.**
  `src/voice-profile.cpp:637` refuses a clip above `max_frames_per_clip`, so at
  `samples_per_frame = 1920` this family's own writer cannot emit more than
  **375** code frames; the loader accepted anything the buffer could back, about
  **16,300** frames for a ~1 MiB envelope, 43x the ceiling. `prefill` is
  quadratic (`model.cpp:1033`, `:1040`, `:1053`, `:1099`), so ~1 MiB drives
  **≈5.9 GB RSS** and 4 MiB reaches tens of GB. **Task 9's structural fix is
  intact and does not reach this** — it guarantees a count is backed by bytes,
  and 16,300 frames legitimately is. Load and synthesis both return `SYNTH_OK`;
  **only a peak-RSS assertion can see it.** The sibling family already has the
  check (`src/arch/omnivoice/profile.h:376,390`, passed at
  `src/voice-profile.cpp:895`).
- **MB2 CRITICAL — the output-limit diagnostic asserts a cause the tree
  disproves.** `src/synthesize.cpp:1219-1226` says *"the measured cause is a
  reference transcript that does not match its reference audio"* for **any** ICL
  limit stop. `tests/qwen3_tts_base_load_real.cpp:527-540` drives that exact
  message with a **matching** transcript and a caller-set
  `max_output_frames = 3840`; `base-text-long` reaches it on arithmetic. The
  sibling path already articulates the principle
  (`tests/qwen3_tts_output_limit_test.cpp:145-148`: *"naming a reference
  transcript here would be misdirection"*). **Partly a controller error** — an
  actionable diagnostic was asked for and an over-specific one was written. Fix
  is two lines: hedge, or gate the sentence on the limit being the package
  default.
- **MB3 — two stale "ONLY registered enforcement" claims** at
  `tests/qwen3_tts_icl_real.cpp:9-12` and `tests/CMakeLists.txt:1132-1136`,
  both contradicting `tests/tolerances/qwen3-tts.json:654`'s
  `registered_consumers` and both inverting Task 13's headline achievement. A
  maintainer reading either concludes `codec.chain` still has no enforcer, which
  is exactly the condition Task 13 ended. Same shape as §0's partial correction.
- **I3 — the provenance inversion**, `docs/porting/families/qwen3-tts.md:3209-3221`.
  See §3.2; it sends Plan 4 hunting in the wrong implementation.
- **I1/I2 — the audible-claim family**, seven sites. See §2.2.
- **I4 — the "0.0 … 0.41" bf16 range**, understated 2.4x. See §2.1.
- **I11 — `tests/qwen3_tts_base_load_real.cpp:511-514` cannot fail, and two
  other files cite it as their *stronger* masker.** Neither named deletion
  reaches the line: `validate_speaker_sources` (`model.cpp:894-899`) enforces
  the three ICL fields as a set, so dropping one leaves the other two and the
  request is refused with `INVALID_ARG`. In the passing world it is doubly
  degenerate — the ICL runs reach the cap and return `SYNTH_ERR_OUTPUT_LIMIT`
  with no audio while the x-vector run returns `SYNTH_OK` with PCM, so both
  conjuncts are false for reasons unrelated to the property. This is **wrong in
  the direction that invites deleting real coverage**: `icl_real.cpp:876-890`
  and `clone_real.cpp:588-589` both defer to it. What actually catches the
  single-field deletion in that file is `:532`.

### 3.2 The two length pathologies, with their true provenance

An earlier conflation put these on the wrong side of the port boundary and
propagated into Task 14's documentation. **The corrected version:**

**The transcript-mismatch hang was measured PORT-side**, through the public C
seam. Task 11's reviewer kept the real 8 s clip and swapped only its transcript:
synthesis never terminated — **475.91 s of CPU at `CMAKE_BUILD_TYPE=Release`**,
ran to the 2048-frame ceiling (`kDefaultMaxFrames`), returned
`SYNTH_ERR_OUTPUT_LIMIT` with zero audio. The trigger is transcript–audio
mismatch, an ordinary user error. **Nothing establishes this as upstream's
behaviour, and the port is neither implicated nor cleared on it.** What exists
today is **mitigation, not a fix and not a diagnosis**: one static error string.
Shortening it needs a lower ICL ceiling or a run-away detector — **not done**.

**The 9-frame anomaly work was ORACLE-side**, on the PyTorch reference
(`scripts/measure_qwen3_tts_icl_reference_length.py`, `torch.manual_seed`,
`Qwen3TTSModel`). **The port was not compared on it.** It is nonetheless
adjudicated, and this closes an item open since Plan 1:

- Reproduced exactly: 9 frames, 0.72 s, peak 0.474609375, rms 0.0686, 8 distinct
  codes; all six Plan 1 rows reproduce bit-identically, verified by the reviewer
  from raw artifacts.
- **The monotone-length explanation was raised inside the plan and refuted
  inside the plan.** The inverse relation (ref 7…101 frames → 136…45 generated)
  is monotone **only within the clip's real 8.08 s**. Past that, length exists
  only by looping the audio (`dump_reference_qwen3_tts_base.py:339-342`,
  `np.tile`) while the transcript stays single-repetition — so every long point
  is itself a transcript–audio mismatch. At 375 frames the output is not a
  function of anything: **five seeds give 9, 8, 12, 4, and 2047**. Matching the
  transcript moves seed 0 from 9 to **66**.
- **Shared mechanism, demonstrated rather than argued: one input, five seeds,
  four collapses and one runaway.** The collapse and the runaway are the same
  configuration. Stage 1's greedy run-away to 8191 frames may be a third
  symptom. This remains a **hypothesis, not a finding** — do not let it be
  promoted, and do not let it clear the port.
- Distinct from `base-text-long`'s legitimate ceiling hit: 233 distinct tail
  codes against ~6 for the degenerate loop, and 4,749 characters needing ~3,950
  frames against a 2,048 budget. Its manifest status is now
  `non_terminating_at_max_new_tokens`, not `ok`.
- **Half of the port comparison is still owed:** the deterministic half runs and
  is correct; the frame-count half needs a distributional run, blocked by a
  missing `trim_seconds` driver flag.

### 3.3 Coverage the tree could be read as claiming but does not have

**"Three cases" is one recording.** `base-ref-min` is a **byte-exact prefix** of
`base-icl-en`, and `base-text-short` is that same clip again at full length —
one recording, one speaker, one microphone, at two distinct lengths. Three files
state this (`tests/tolerances/qwen3-tts.json:636`,
`scripts/dump_reference_qwen3_tts_codec_encoder.py:1137-1143`,
`scripts/validate-qwen3-tts-codec_encoder.py:137-142`); **the family record —
the one document a later implementer is directed to (`:2921-2923`) — does not,
anywhere.** Its `:2986` ("three cases"), `:3031` ("0.00/3.96/3.96%", where the
identical 3.96/3.96 is the same clip measured twice) and the two-column table at
`:2956-2960` all read as independent corroboration. The spec's fourth erratum
has the same problem. (`tests/tolerances/qwen3-tts.json:639` also says "three
lengths" where `:636` correctly says two.) **A second reference clip and a
second speaker are the highest-value coverage Plan 4 can add.**

**[Historical, marked rather than rewritten: the family record's gap was closed
on 2026-08-14, after this file was written. `docs/porting/families/qwen3-tts.md:2980-2995`
now states in the family record itself that the three codec cases are one
recording, and that none of it is independent corroboration. The "does not,
anywhere" above is the state at Plan 3's close.]**

**The acoustic branch's real headroom is 1.64x, not the 2.0x recorded.** The
oracle's `codec_encoder/waveform.f32` is **exactly `bfloat16(clone.wav)`**,
verified element for element — found, not assumed. The stage-wise validator
feeds the port that rounded waveform; the end-to-end test feeds the WAV, because
that is what a caller hands the library. Measured cost: semantic
0.004103 → 0.004033, acoustic **0.2491 → 0.304719**, code agreement
52.970% → 50.743%. Nothing was widened; it is recorded under a
`registered_consumer` block in the tolerance file.

Two further shape facts for the grid: alignment across ten cases is **8 pad,
2 truncate**, with `base-text-long` an unanticipated second truncate case
(T1=980 > T2=102, 878 trailing); and the `replay` stage was **deliberately not
widened** because 10 of its 12 x-vectors are byte-identical, so widening would
be theatre — the two that differ need `trim_seconds`.

### 3.4 Deferred minors, triaged

| item | reason to carry |
| --- | --- |
| Semantic p95 statistic redesign | §1.3; already ruled, needs the flip rate gated separately |
| `trim_seconds` driver flag | one flag, two holes: the distributional port comparison on the 9-frame case, and widening `replay` |
| The two p95 implementations (C++ and Python) have no registered cross-check | self-disclosed in `registered_consumers`; compared once by hand, can drift silently |
| A second reference clip / second speaker | §3.3; all three codec cases are one recording |
| I10 — `reference_text_tokens` unbounded in `reference_is_well_formed` (`talker-host.cpp:34-62`) | out-of-range is a `ggml_get_rows` **process abort**, not a status; not reachable today only because both `IclProfile` constructors bound the ids (`profile.cpp:162-169`, `:1581-1593`). Close it when MB1 is fixed |
| I12 — `serialize_icl_profile`'s `codes.size() != groups * frames` clause (`profile.cpp:1191-1193`) | shipped writer rule with **no isolating arm**; deleting it leaves the whole suite green |
| I5–I9, M1–M18 | prose and figure corrections, none enforced by executable code |
| No numerical gate covers `run_synthesis`'s own prefill assembly | Task 11 open; its validation was hoisted to a free function because inline it was unreachable by any test |
| `src/voice-profile.cpp`'s ICL branch is unreachable from unit tests | Task 9 deferred it to Task 10's integration rewrite; not recorded as closed |
| `conventions.json`'s "101 frames at most" is stale | `base-ref-max` is 375, past the declared 250-frame window; the generated artifact reproduces the stale line |
| Second BPE frontend costs **+45 MB peak RSS** (5,594,676 → 5,639,716 KB, measured) | considered and declined in-file (`model.cpp:620-627`); sharing tables needs `src/bpe-frontend.*`, a cross-family slice. Lazy construction also named and still open |
| Task 3 M1 — the distinctness test covers only `q_proj` and the codebooks (`tests:858-862`) | the other resolved pointers have no aliasing coverage |
| Task 3 M2 — `resolve_codec_encoder_quantizer` (`catalog.cpp:405-419`) is a one-literal twin of `resolve_quantizer` (`:287-303`) | defensible and reasoned at `:400-404`; but `quantizer.output_proj` is resolved and **never read** on the encoder path |
| Task 3 M3/M4 | inherited citation and cross-file-claim nits, unchanged by this branch |
| Task 6 — `is_blank` is ASCII-only | accepted as stricter than upstream |
| `clone-real` assertion 7 retirement | would be a scope change to Plan 2's committed gate; **do not retire** — and not on the `base_load_real:514` premise, which is I11 |
| Task 8 duplicate tests (`profile_test:1953`, `:680`) | no unique assertion; their inversions were never run; harmless |
| Task 2 — five brief citations one line early | resolved by transcribing from the generated `prompt_conventions.json`, which resolves every span at run time |

## 4. Standing facts Plan 4 must not rediscover

**Generated length is build-dependent.** Identical input, seed and backend, with
threads swept 1/2/4/8/20 and no movement — but **Release gives 24,960 PCM frames
and RelWithDebInfo gives 48,000**, each reproducible. The stop decision differs
by build type. **No frame count may be pinned or quoted without naming its
build.** This is the performance-number rule applied to output *content*, and it
already produced one tree contradiction Task 14 had to fix.

**Gate baseline at branch end:** unit and sanitizer both 98% of 101, with only
the two known pre-existing failures (`synthesize-python-api-wheel-test`,
`synthesize-vits-python-unit`, both on gitignored VITS artifacts); qwen3-tts
integration **14/14**. Any third failure is yours.

**The public ABI did not move, proved by blob hash rather than diff:**
`include/synthesize.h` is `dd7b670b48b38e3546f161055d49ce6102f79cd8` at both
`5f3e700` and `e99205b`. `include/`, `bindings/`, `examples/` and `ggml/` are
byte-unchanged. The package's declared contract is unchanged — `source_flags`
exactly 9, the six reference limits, schema v1, `description_language` still
`UNSUPPORTED`. The only contract movement is the two requirement fields
(`weights.cpp:804-805`) flipping to `OPTIONAL`, which is a widening.

**The codec-encoder golden gate is real and was verified end to end.**
`synthesize-qwen3-tts-codec-encoder-golden`
(`tests/check-qwen3-tts-codec-encoder.cmake`) re-runs the driver into the build
tree rather than reading a stale dump, then gates all three cells across all
three cases; `load_gates` resolves thresholds from the committed JSON and makes
a JSON/module-constant disagreement **fatal**, so a transcription cannot drift.
Registration is guarded on three uncommitted artifacts and deliberately does
**not** register a weaker form when the upstream-f32 root is absent — a missing
dump is a missing test, not a passing one. Fault injection reproduces
1.879/1.634/2.115. Two rough edges: the validator's filename uses an
**underscore** (`scripts/validate-qwen3-tts-codec_encoder.py`) where every
sibling uses a hyphen, which is what broke
`test_every_registered_validator_has_a_measured_stage` for six tasks unnoticed;
and `_synth_qwen3_tts_codec_upstream_root` (`tests/CMakeLists.txt:1268-1269`) is
hardcoded to `${CMAKE_SOURCE_DIR}/build/qwen3-tts-codec-encoder-f32` with no
override variable.

**Reproducibility, proved rather than asserted:**
`scripts/dump_reference_qwen3_tts_codec_encoder{,_float32}.py`,
`scripts/dump_reference_qwen3_tts_icl_prompt{,_float32}.py`,
`scripts/validate-qwen3-tts-codec_encoder.py` and
`scripts/measure_qwen3_tts_icl_reference_length.py` are all committed; re-run
from scratch the codec dumpers reproduce the published tree `diff -r`
byte-identically, and the validator exits 1 on a faulted tree and 0 on all three
real cases.

**The tensor arithmetic, since it is now pinned in two places.** 161 emitted
codec-encoder tensors = stem 2 + stages 24 + tail 2 + downsample 1 +
transformer 96 + semantic 3 + acoustic 33. `base == 894` and
`base − custom_voice == 76 + 161` still hold; `expected_tensor_count` is
byte-identical to Plan 1's, verified by extracting the function from both
revisions and diffing.

**Contract changes later work must know.** `flatten_talker_prompt` gained two
**required** out-parameters — not defaulted, on purpose, so a caller cannot
flatten an ICL prompt and drop its codes silently. `sum_code_embeddings` now
reads `[frames, groups-1]` with `ne[0]` = frames; the decode step is the
one-frame case and its tensor is reshaped, byte-identical. **Group 0 reads the
talker's own codec table** — the same table an ordinary codec token reads — so
`codec_token` still carries it and only groups 1..15 needed
`TalkerInputPosition::acoustic_codes`; `codec_offset`, the contiguous-run
invariant and `external_speaker_index` (still 4) are untouched *by
construction*.

**`IclProfile` composes `XVectorProfile` as its first member**, which is what
keeps the existing cast at `src/synthesize.cpp:1097-1098` defined — it reads
`.mode` before the mode is known. A flat struct would have been UB; changing the
cast instead would need a second `ProfileFamilyTag`, which
`voice-profile-handle.h:21-26` rejects on the record. Three `static_assert`s pin
exactly the needed facts. Profile Schema identity and version did not change:
`kind` discriminates, `kPrescanKvCountXVector` stays 10, and `ref_rms` and
`language_tag` moved from the deleted `kXVectorOnly` scope to `kCommon` without
changing the x-vector key set, so **no Plan 2 envelope changed shape**.

**A resolver can resolve everything and retain one thing thirty-one times.**
Task 3's standing example: aliasing all 31 acoustic codebooks to slot 0, so that
every name resolved and every pointer stayed non-null, left the catalog sweep
GREEN and the by-name test PASSING (`tests:824,826` assert only `size()==31` and
per-element non-null). Only `std::set::insert(...).second` at `tests:852` caught
it.

**The Base package is still not published**, and publication remains a separate
act requiring jiangzhuo's confirmation at the time.
