# OmniVoice Plan 5 — Carry-Over Ledger from Plan 4

Written at Plan 4 close (2026-08-07), from Task 17's whole-branch triage of
`.superpowers/sdd/2026-08-06-omnivoice-plan-4-quants-backends-ship/progress.md`
plus the standing items
`docs/superpowers/plans/2026-08-02-omnivoice-plan-4-carryover.md` left open.
Plan 4 landed the family's Quantization Profile measurement (negative,
F32-only ships), closed the Plan 3 validation-suite debt, claimed the CUDA
Execution Backend for the codec's decode path, ran the Listening Audit, and
prepared the ship artifacts — see
`reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`'s 2026-08-07 Plan 4
closeout entry for the evidence. There is no Plan 5 yet; this ledger exists so
whichever plan touches this family next (ship-adjacent work, or the next
family's own carry-over review) does not have to re-derive any of this from
the branch history.

## Plan 4 carryover items closed, not carried forward

For continuity with the previous ledger: items 5–8 and 10 of
`2026-08-02-omnivoice-plan-4-carryover.md` are resolved, not open.
Item 5 (the writer↔whitelist fast test) was closed by Task 5
(`tests/omnivoice_serialize_writer_agreement_test.cpp`, proven in both drift
directions, then extended in fix round 1 for per-kind exact equality and the
`n_kv` count constants). Item 6 (primary-grid digests) was closed by Task 6
(all 20 cases pinned, verified pre-comparison). Item 7 (the converter
harness items) was closed by Task 7, which additionally found both defects
were already fixed on `main` before Plan 4 began — the debt was missing unit
coverage only, now closed. Item 8 (the quantization margin protocol) was
closed by Task 3's measurement: the margin table is bit-identical across
every profile because the generator never changes, and the real failure is
in a subsystem (the clone-path RVQ lookup) the margin screen was never built
to probe — this is itself one of the open findings carried below, not a
reason to re-run the protocol. Item 10 (the Listening Audit) was closed by
Task 16: `no_obvious_regression`, all six pairs.

## Project-wide questions this plan did not own (still open)

These four were named in the Plan 3→4 ledger as needing a project-wide
decision rather than a per-family fix, and Plan 4's own plan document
(`docs/superpowers/plans/2026-08-06-omnivoice-plan-4-quants-backends-ship.md`,
Self-Review Notes) repeats that they belong in this ledger rather than being
settled unilaterally by an OmniVoice-only plan. None were touched by Plan 4.

1. **Should the project have a `SYNTH_ASSERT` that survives Release/
   RelWithDebInfo builds?** `build_guided`'s contract assert
   (`src/arch/omnivoice/generator-host.cpp:140` — `uncond` must be non-null
   when `guidance_scale != 0`) is inert under `NDEBUG` in every standard
   build configuration this project ships (both Release and RelWithDebInfo
   define it). This is a project-wide policy question: decide once, apply
   everywhere an assert currently carries a contract no release build
   checks.
2. **The family's `INVALID_ARG` diagnostic code is named
   `"synthesis.graph_failed"`** (`src/synthesize.cpp`, four call sites —
   currently lines 949/1044/1091/1164, mirroring qwen3-tts's own naming)
   even when the failure is input validation, not a graph failure.
   Consistent across both families that use it, but the name is misleading
   on its own terms — a project-wide diagnostic-naming question.
3. **`stream` could be `std::optional<NormalRandomStream>` instead of
   `std::unique_ptr`** (`src/arch/omnivoice/model.cpp:613`), avoiding one
   heap allocation per sampled synthesis. Not urgent — a micro-optimization,
   not a correctness issue — but real enough to measure rather than carry
   indefinitely.
4. **The two-clips rejection test doesn't pin validation ORDER**
   (`tests/omnivoice_profile_test.cpp:339-350`): it asserts
   `SYNTH_ERR_INVALID_ARG` fires when two reference clips are supplied
   against `max_reference_count == 1`, but nothing pins that the COUNT
   check specifically is what fired, ahead of any content check on either
   descriptor — a future reorder of the validation sequence that read an
   unvalidated second descriptor before rejecting the count could go
   unnoticed by this test alone. Needs a small design decision (should
   order be part of the contract, or is any rejection sufficient) before a
   fix, not a one-liner.

## The float64 cosine estimator question (still open)

Item 9 of `2026-08-02-omnivoice-plan-4-carryover.md`, unchanged: the deep
generator probes' cosine gates sit at 5.1×–11× the measured deviation
because `scripts/validate-omnivoice-replay.py:166`'s `cosine()` accumulates
in float32, whose own summation noise floor (as large as 2.14e-07,
comparable to the deviations being measured) dominates at this family's
precision. Plan 4 did not touch this — the Q8_MIXED/F16 measurements used
the same estimator and it was adequate for a 36.4%/3.7%-of-2808 mismatch,
several orders above its own noise floor, so there was no forcing function
to fix it this cycle. If a future cycle wants sharper deep-probe gates, make
`cosine()` accumulate in float64 and re-measure from scratch — the noise
floor moves, so the existing 5×-rule figures do not simply divide by
anything.

## Task 2's ARCHITECTURAL NOTE: the by-name demotion exception's silent-miscompute path

`tools/synthesize-quantize/quantize.cpp:186-207` (`matrix_family` /
`omnivoice_collapsed_conv_kernel`) names `codec.acoustic_decoder.conv2.weight`
out of the demotion-to-native fallback by exact tensor name, because packing
it succeeds (`7 * 32 = 224`, a whole number of Q8_0's 32-element blocks)
where the general demotion rule would otherwise misread its `[7, 32, 1]`
shape as a two-dimensional matrix (via `ggml_n_dims`'s trailing-unit-dimension
collapse) and demote it to a native row of 7, which no block divides. Task 2
verified this is the only such tensor in the family's current catalog. The
exception is correct today, but it is a by-name special case, not a general
one: a **future** collapsed conv kernel added to this family whose packed
width happens to be a whole multiple of 32 would pass the same row-size
check that legitimizes `conv2.weight`'s exception, and nothing distinguishes
"packed" from "quantized-but-should-have-been-demoted-native" for it — the
runtime has no way to tell those two cases apart, and the failure mode is a
silent miscompute, not a loud abort. If this family (or a future one sharing
this policy shape) ever adds a new conv kernel, re-run Task 2's survey before
assuming the demotion list is still exhaustive; do not add a second by-name
exception without also asking whether a general, checkable predicate is now
overdue.

## Task 9's note: qwen3-tts's blanket codec-prefix mirror is unsafe here, and may be unsafe elsewhere

`src/arch/qwen3-tts/model.cpp:445-457` mirrors every tensor whose name starts
`codec.` to the accelerator twin unconditionally. That is safe for qwen3-tts
only because qwen3-tts has **no host-side codec-weight consumer at all** — no
cloning/encode path reads `weights.codec` on the host. OmniVoice's Task 9
found this precedent does not generalize: `codec.quantizer.*` is read by
*both* the primary-resident decode graph (`build_codec_decoder`) and the
host-side `rvq_encode` readback through the same `ModelWeights::quantizers`
field, so a first-draft port of qwen3-tts's blanket "overwrite the shared
struct in place" pattern would have handed the twin's pointer to
`rvq_encode` — not a crash (`rvq_encode` reads exclusively through
`ggml_backend_tensor_get`, which is backend-aware and copies device-to-host
correctly), but 40 silent extra device→host PCIe round trips per cloning
request, contrary to `docs/backends.md`'s discrete-outputs intent for that
readback. Fixed here by keeping `impl.weights` always CPU-bound and giving
only `impl.decode_weights` the twin-aware binding
(`src/arch/omnivoice/model.cpp`, Task 9 fix round 1). **The standing
question for the next family (or the next accelerator twin any family
adds):** before copying ANY existing family's "mirror everything under this
prefix" pattern, grep for every consumer of the tensors that prefix covers,
not just the primary-graph builder the pattern was written for — a prefix
match is not evidence that every reader of it runs on the same schedule.

## What Plan 4 newly owes

Facts Plan 4's own work surfaced that are not bugs, but that whoever next
touches this family's backends, tolerances, or publication needs to know
rather than rediscover.

- **The margin screen does not transfer to backend-placement questions**
  (Task 12's finding, `docs/porting/families/omnivoice.md`'s Open Questions
  section and the porting log's 2026-08-07 Task 12 entry). The screen is
  calibrated against this port's own F32 arithmetic diverging from the
  oracle's F32 arithmetic at 6.1e-04 max_abs step-0 logits — a knife-edge
  detector for a specific, small scale of perturbation. Generator-on-CUDA's
  step-0 divergence from the CPU baseline was already ~90× that scale
  (0.10 max_abs) before any layer compounding, and the resulting flip
  pattern did not track the margin table at all: three of five
  margin-predicted first-flip cases flipped, one flipped by a single token,
  one did not flip, while ten "comfortably safe margin" cases flipped too,
  several worse than any predicted case. **Do not use the greedy margin
  table to predict, screen, or reason about the safety of ANY future
  backend-placement or precision-placement decision for this family** — it
  answers a different, narrower question (is this Golden case a
  knife-edge for CPU-vs-CPU-oracle comparison) than "will this
  perturbation's magnitude survive placement," and Task 12 measured that
  those two questions have no reliable relationship at this scale.
- **`models/publish/omnivoice-0-6b/` is a working, git-ignored directory
  that the eventual `hf upload` command depends on, not a byproduct
  directory that can be deleted casually.** Task 14 built it as a clean,
  flat publication directory (hard-linked F32 GGUF, the Boson license
  sidecar, the rendered `README.md`) specifically because
  `models/omnivoice-0-6b/` also holds the upstream checkpoint
  (`model.safetensors`, `audio_tokenizer/`, `tokenizer.json`, the upstream
  README) that must never be uploaded. If the shipped GGUF, its sidecars,
  or the rendered card ever change, `models/publish/omnivoice-0-6b/` must be
  rebuilt from Task 14's recorded recipe
  (`docs/models/omnivoice-0-6b.md`'s reproduction section) before any future
  publication attempt — it is not kept in sync automatically by anything.
- **A cross-family defect Task 7 found but correctly left unfixed**:
  `scripts/convert-qwen3-tts.py`'s own `verify_gguf` has the identical
  element-count-only shape-check defect that `scripts/convert-omnivoice.py`'s
  `verify_gguf` had before commit `b08757d` fixed it (comparing element
  counts rather than shapes, so a transposed-but-same-element-count tensor
  would pass). Task 7 did not touch it — out of file scope for an
  OmniVoice-only task — but it is a real, live gap in the qwen3-tts
  converter today and deserves its own matching fix, not rediscovery from
  scratch.

## Standing facts Plan 5 must not rediscover

- **This family ships F32-only.** Both codec-only Quantization Profiles
  (Q8_MIXED, F16) were measured against the exact-token gate and both
  failed the clone path's RVQ encode grid — not a knife-edge margin call
  either time, and not eligible for dual admissibility (there is no second
  oracle-produced grid to enumerate; this is the port's own arithmetic
  diverging). The structural reason: 93.8% of the quantizable weight feeds
  a discrete nearest-neighbour decision and cannot be quantized without
  flipping it; the 6.2% that is safe to quantize saves under 1% of package
  size. Re-litigating this without new evidence (a different profile shape,
  a different rounding scheme) would be repeating a closed measurement.
- **The CUDA Execution Backend is claimed for the codec's decode path only.**
  Placement is unconditional across all 20 cases: 0/880,032 generator nodes
  left the CPU, 8,440/8,440 codec nodes left it. The generator stays on CPU
  by the discrete-outputs rule (its whole mask-predict loop feeds back into
  itself every step), independent of the Task 12 experiment above, which
  only measured what would happen if that rule were ignored — it flips, so
  the rule and the measurement agree, but the rule would have governed the
  outcome either way.
- **The Listening Audit verdict is `no_obvious_regression`, all six pairs**,
  recorded 2026-08-07, including a CUDA-vs-CPU codec pair (the largest
  numeric divergence in the set) that corroborates the backend claim with
  audible evidence a tolerance grid cannot provide. This does not move
  `quality_evaluation` off `not_run` — ADR 0017's automated grid has not run
  and is not scheduled — and is not general perceptual-equivalence evidence
  beyond the six pairs actually heard.
- **The Restricted Model Package ceiling (ADR 0018) is settled, not
  provisional.** Card frontmatter is `license: other` +
  `license_name: omnivoice-cc-by-nc-unspecified-version-plus-boson-higgs-audio-2-community`
  (superseding the design spec's original `cc-by-nc-4.0`-class wording — see
  that document's 2026-08-06 amendment). Three licenses travel with the
  artifact (LM weights CC-BY-NC no stated version, codec weights the Boson
  Higgs Audio 2 Community License as a declared Sidecar Resource, code
  Apache-2.0 covering nothing shipped). No weight artifact from this family
  may ever be labelled `apache-2.0`.
- **Publication has not happened.** Plan 4 prepared every ship artifact and
  stopped, per its own standing constraint. `hf repos create` / `hf upload`
  for this family requires jiangzhuo's explicit per-act confirmation naming
  the target repository — an approved plan, a finished measurement, or the
  existence of `models/publish/omnivoice-0-6b/` do not themselves constitute
  that confirmation.
