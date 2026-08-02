# OmniVoice Plan 4 — Carry-Over Ledger from Plan 3

Written at Plan 3 close (2026-08-02), from Task 17's whole-branch triage of
`.superpowers/sdd/2026-08-01-omnivoice-plan-3-sampling-cloning/progress.md`'s
recorded deferred minors plus the standing items
`docs/superpowers/plans/2026-07-31-omnivoice-plan-3-carryover.md` left open.
Plan 3 landed the public sampled path, Reference Audio and Description Text
Voice Profiles, their Serialized Profile GGUF round-trip, and the CLI/
Python-wheel Adapters (see `reports/porting/omnivoice/omnivoice-0-6b/
_porting-log.md`'s 2026-08-02 entry for the evidence). Plan 4 (Port
Validation Suite completion, Quantization Profiles, Execution Backends, and
ship) consumes this register.

## Plan 3 carryover items closed, not carried forward

For continuity with the previous ledger: items 6–8 of
`2026-07-31-omnivoice-plan-3-carryover.md` are resolved, not open.
Item 6 (`ref/semantic_hidden.f32` vs the mean-of-13 the codec actually
consumes) was settled by Task 9's `ref/semantic_mean.f32` probe, which
captures exactly the value the codec's quantiser consumes and is
independently gated. Item 7 (the quiet-reference volume branch's coverage
gap) was settled by Task 14's Reference-Audio work: the branch is
structurally uncoverable by any golden case built from this family's one
pinned reference clip, and is closed via unit fixture instead — see "What
Plan 3 newly owes" below for the exact disposition, so it is not
re-litigated. Item 8 (whether the sampled path needs its own margin screen)
was settled by ruling: it does not — see the same section.

## Plan 4 — new debt from Plan 3's own minor-finding triage

Every "minor (deferred)" entry in the Plan 3 progress ledger was triaged at
Task 17. The ones judged cheap one-liners were fixed in that task (see its
report); the entries below are the ones that need real design or
test-authoring work and could not be bulk-fixed responsibly. Each names its
file.

1. **Should the project have a `SYNTH_ASSERT` that survives Release/
   RelWithDebInfo builds?** `build_guided`'s contract assert
   (`src/arch/omnivoice/generator-host.cpp:140-141` — `uncond` must be
   non-null when `guidance_scale != 0`) is inert under `NDEBUG` in every
   standard build configuration this project ships (both Release and
   RelWithDebInfo define it), so "now enforced" from Task 3's review was
   enforcement at review time only, not at runtime. This is a project-wide
   policy question, not an omnivoice-specific fix — decide once, apply
   everywhere an assert currently carries a contract no release build
   checks.
2. **`stream` could be `std::optional<NormalRandomStream>` instead of
   `std::unique_ptr`** (`src/arch/omnivoice/model.cpp:528`), avoiding one
   heap allocation per sampled synthesis. Not urgent — this is a
   micro-optimization, not a correctness issue — but real enough to measure
   before Plan 4's own performance work, rather than carry indefinitely.
3. **The family's `INVALID_ARG` diagnostic code is named
   `"synthesis.graph_failed"`** (`src/synthesize.cpp`, four call sites
   around lines 939/1019/1066/1139, mirroring qwen3-tts's own naming) even
   when the failure is input validation, not a graph failure. Consistent
   across both families that use it, but the name is misleading on its own
   terms — a project-wide diagnostic-naming question, not something this
   plan should have unilaterally renamed mid-family.
4. **The two-clips rejection test doesn't pin validation ORDER**
   (`tests/omnivoice_profile_test.cpp`): it currently asserts a *count*
   check fires before a content check, but nothing pins that a future
   reorder of the validation sequence would be caught — a future change
   that read an unvalidated second descriptor could go unnoticed by this
   test alone. Needs a small design decision (should order be part of the
   contract, or is any rejection sufficient) before a fix, not a one-liner.
5. **The Serialized Profile pre-scan whitelist's maintenance obligation has
   no fast, unconditional test pinning it.** `kPrescanKnownKeys`
   (`src/arch/omnivoice/profile.cpp`) is a closed table transcribed BY HAND
   from what `set_common_metadata` / `serialize_clone_prompt` /
   `serialize_design_instruct` write; `tests/omnivoice_serialize_test.cpp`'s
   `common_kv_bytes` helper is a SEPARATE hand-transcribed mirror of the
   same writer behavior, used to build synthetic whitelist-acceptance
   fixtures — neither is generated from, or mechanically checked against,
   the real writer functions. The only test that exercises the REAL writer
   against the REAL whitelist is the model-guarded round-trip integration
   test (`synthesize-omnivoice-profile-test`), which needs a real GGUF
   package and is not part of the fast unit loop. If the writer's key set
   ever changes without `kPrescanKnownKeys` changing to match, the failure
   mode is either a spurious rejection of the writer's own valid output
   (caught, loudly, but only by the model-guarded test) or — if a key is
   REMOVED from the writer without removing it from the whitelist — a
   silently-too-permissive whitelist that a fast unit test would never
   catch at all. **Action for Plan 4: add a fast unit test that pins
   writer↔whitelist agreement directly** — e.g. drive the real
   `serialize_clone_prompt`/`serialize_design_instruct` functions against a
   synthetic in-memory model (no real GGUF package required), parse the
   emitted key set with a raw GGUF reader, and assert it equals
   `kPrescanKnownKeys` exactly. This does not exist today.

## Plan 4 — standing items carried from Plan 3's own carryover (unchanged)

Items 9–13 of `2026-07-31-omnivoice-plan-3-carryover.md` were not addressed
by Plan 3's scope and remain open, renumbered here for this ledger's own
continuity:

6. **Pin primary-grid digests per case in the Golden Manifest**, symmetric
   with the alternate-grid entries `omni-fast-mode` already carries. Today
   the primary grid is whatever the local oracle dump holds; a divergence
   fails loud, but the digest itself is knowable and should be recorded
   rather than implicit.
7. **Converter harness items from Plan 1's Task 3** (`verify_gguf` shape
   check, license-copy reorder) are still end-to-end-only — close them in a
   dedicated converter harness pass rather than continuing to defer.
8. **Quantization margin protocol.** Before any quantized profile is
   judged, re-run `--margin-report` and compare against the F32 greedy
   baseline. The four in-band cases most likely to flip first are
   `omni-short-en` (1.16e-04), `omni-long-boundary` (2.05e-04),
   `omni-rate-fast` (2.44e-04) and `omni-lang-none` (6.03e-04) — all above
   the 1e-4 margin screen but still below the 6.1e-04 F32 logit-divergence
   bound. `omni-rate-slow`, at **9.5e-06**, is expected to flip almost
   surely under any perturbation; it is retained today only because
   screening governs new-case adoption, not the retirement of cases already
   demonstrating parity (`docs/porting/families/omnivoice.md`'s "Watch
   note"). The dual-admissibility mechanism (enumerate the second grid with
   its provenance) is the recorded remedy path for a flip — never a
   threshold on a token id.
9. **The deep-probe cosine gates sit at 5.1×–11× the measured deviation**
   because the validator's float32 cosine estimator's own summation noise
   floor (as large as 2.14e-07, comparable to the deviations being
   measured) dominates at this family's precision. If Plan 4 wants sharper
   gates, make `cosine()` accumulate in float64 and re-measure from
   scratch — the noise floor moves, so the existing 5×-rule figures do not
   simply divide by anything.
10. **The Listening Audit is still owed before ship.** Numerically the port
    adds nothing audible at any measured tolerance, but the blind A/B
    against the reference — jiangzhuo's own, at ship time, with prepared
    clip pairs — is the actual gate; a tolerance grid is not audible
    evidence and does not substitute for it. Offer it before shipping
    rather than waiting to be asked (per `CONTEXT.md`'s "Listening Audit"
    definition and this project's standing practice on prior families).

## What Plan 3 newly owes

Facts Plan 3's own work surfaced that are not bugs, but that Plan 4 (or
whoever next touches this family's tolerances or tests) needs to know
rather than rediscover:

- **Release-vs-sanitizer convolution accumulation is a measured, named
  noise source for tolerance work, not a hypothesis.** Task 10 found that
  `mixed48` and the real-signal resampler fixtures show a small residual
  (5.96e-08 and 1.19207e-07 respectively) in Release builds that is exactly
  0.0 under RelWithDebInfo (the sanitizer build's configuration) — the
  resampler's KERNEL TABLE is bit-exact in both; only the convolution's
  summation order differs by build configuration. Any future tolerance
  derived by comparing Release-build output against a sanitizer-build
  reference (or vice versa) must budget for this, and the two builds should
  not be assumed numerically identical at the float32 ULP level anywhere
  precision this tight is claimed.
- **The quiet volume arm (`0 < ref_rms < 0.1`) is closed by unit fixture,
  permanently, not provisionally.** This family's only pinned Reference
  Audio clip measures `ref_rms ≈ 0.1229`, above the 0.1 gate, and both
  committed clone goldens share it; no golden case this family is likely to
  add without deliberately sourcing a second, quieter reference clip will
  ever exercise this branch. `tests/omnivoice_reference_encoder_test.cpp`'s
  `check_quiet_boost_arithmetic` and `check_boost_boundaries` are the
  coverage, and `docs/porting/families/omnivoice.md` records this
  disposition — a future reviewer finding "no golden covers the quiet arm"
  is rediscovering a closed question, not opening a new one, unless a
  second reference clip is deliberately added.
- **Sampled-path margin screening is decided: not applicable, by design,
  not by oversight.** The greedy margin screen exists because greedy Golden
  cases claim oracle agreement at a resolution finer than this port's F32
  arithmetic and the oracle's F32 arithmetic can both legitimately
  resolve. The public sampled path (`synthesize-omnivoice-public-request`)
  claims relations between runs of this port alone — never agreement with
  an oracle run, since upstream exposes no seed of its own — so a screen
  calibrated against oracle logit divergence has nothing to measure there.
  `tests/tolerances/omnivoice.json`'s own `public` stage note states this
  outcome explicitly. Do not re-open this question without new evidence
  that the sampled path DOES need a screening mechanism of its own kind
  (e.g. against reproducibility failures, not oracle disagreement); nothing
  in Plan 3 found such a need.
- **The pre-scan whitelist's maintenance obligation is real and
  unenforced by a fast test today** — see item 5 above; it is listed there
  as the actionable item and repeated here only so it is not missed among
  the standing items.

## Standing facts Plan 4 must not rediscover

- **The sampler's draw-order contract is a port-defined choice, documented
  at its one construction site** (`src/arch/omnivoice/model.cpp`, where
  `NormalRandomStream` is built): candidates are visited codebook-major,
  frame-minor; class draws (inside `choose_token_sampled`, ascending
  class-id order over top-k survivors) precede that candidate's own
  position draw; a zero-budget step draws nothing. `gumbel_perturb` is
  `scaled = logit / temperature; noise = -log(-log(uniform + 1e-10) +
  1e-10); result = scaled + noise`, entirely float32, matching upstream's
  own line grouping rather than an equivalent rearrangement (reordering
  changes rounding, which can change an argmax's winner).
- **`log_prob` is the row's full max guided log-probability
  (`omnivoice.py:1449`), never the chosen survivor's own value** — reading
  `guided[chosen]` instead diverges from upstream on roughly 45% of seeds
  at `keep=2` on a fixture built to expose it. The 400-seed regression in
  `tests/omnivoice_sampler_test.cpp` is the standing proof; do not weaken
  or remove it.
- **The resampler's kernel constants must be rounded to float32 ONCE, then
  computed entirely in float32** — promoting to double for the
  construction and rounding once at the end reproduces PyTorch's OWN
  behavior incorrectly (38 of 46 tap coefficients wrong under that model).
  This is the same class of lesson as Plan 2's `commit_schedule` float32
  finding: torch's own default tensor dtype at each construction step is
  the thing to transcribe, not "whichever precision seems more careful."
- **The exact-token contract now covers two grids, not one**: the greedy
  decode loop's 8×T grid (17/17 exact) and, since Task 13, the Reference
  Audio cloning path's own RVQ encode grid (2,808/2,808 exact, both clone
  cases). Both are `structural_exactness` claims with no tolerance and
  never will be — a tolerance on a token id is meaningless, per the
  family doc's own standing rule.
- **`ggml` cannot be patched, and a hand-maintained blacklist of another
  library's internal asserts can never be proven complete.** The Serialized
  Profile pre-scan's history (a blacklist closed one crash, a fuzzer found
  a second, different one in the same review session; a positive whitelist
  against this project's own writer format subsumed the whole class)
  is the standing methodology lesson for any future untrusted-bytes
  surface this project adds: validate positively against what THIS
  project's own writer produces, never negatively against what the
  underlying library is known to reject today.
- **Golden suite is at `suite_version` 3** (Task 9's clone-encode probes);
  tolerances are measured and committed against revision 3, `status:
  thresholds-committed-and-enforced`.
