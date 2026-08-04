# OmniVoice Plan 2 — Carry-Over Ledger from Plan 1

Written at Plan 1 close (2026-07-30), from the final whole-branch review's
triage. Plan 1 landed foundations (intake → oracle → converter → load path);
this file is the debt register Plan 2's brainstorming/planning must consume.
Verdicts: every Plan-1 deferred finding was triaged MUST-FIX (done before
merge), FIX-IN-PLAN-2 (below), or ACCEPT-AS-IS (dropped, recorded in review
history only).

## Open-with-Plan-2 (oracle hardening — do these BEFORE trusting any parity gate)

1. Pin/record the dumper's thread count and torch deterministic-algorithms
   state in `scripts/dump_reference_omnivoice_pytorch.py` and its per-case
   metadata — exact-token baselines are currently proven reproducible on one
   machine only.
2. Digest-verify the dumper's weight/config/tokenizer inputs against the
   manifest pins (today only the clone reference audio is checked).
3. Pin `audio_chunk_duration`/`audio_chunk_threshold` in the golden manifest —
   `omni-long-boundary` (719 frames) sits just under upstream's default
   750-frame chunking threshold; a default change would silently alter case
   meaning.
4. Register a positive defaults-from-upstream converter test in the omnivoice
   env (the existing one skips permanently under the VITS-env discovery).

## Early-in-Plan-2 (blast radius grows with time)

5. Rename to uncross `src/arch/omnivoice/{catalog.h,weights.h}` naming versus
   the qwen3-tts convention (HParams currently in catalog.h, tensor structs in
   weights.h) — mechanical while only Plan-1 files reference them.
6. Bounds-check the metadata reader's head/canvas/ratio products
   (`src/arch/omnivoice/weights.cpp`, `catalog.cpp:276-277,301`) — one signed
   (UB) and one unsigned wrap reachable from adversarial metadata.
7. Synthetic on-disk package builder for load-path tests; first consumer: the
   untested frontend-arrays refusal branch (`model.cpp:151-153`).
8. Add a one-line re-lock comment above torch/torchaudio in
   `scripts/envs/omnivoice/pyproject.toml`: **re-lock with `uv lock
   --no-sources`** — the pinned upstream package's own pyproject routes torch
   to a CUDA wheel index that `uv lock` otherwise honors.

## With the runner/validator rewrite (stage 5 work)

9. Runner-skeleton debts: probe-layers-vs-manifest assertion; `error:`+exit-1
   on malformed manifests (`ManifestError` currently escapes mid-dump for
   unpinned clone digests — resolve digests in `load_manifest`); functional
   artifact `format`-vs-writer agreement; stale-artifact clearing at case
   entry; `preprocess_prompt=False` assertion in the tokenizer dumper;
   `urlopen` timeouts + bad-cache operator hint.
10. Commit the verification scripts behind the dumper report's evidence tables
    (or inline their commands) so the byte-identity evidence is re-runnable.
11. Tolerance-file note currently overclaims "every validator reads profiles";
    the first omnivoice validator must make it true or the note gets reworded.
    Add the tolerance `case_count`-vs-manifest cross-check to
    `test_golden_manifests.py` (also fixes the qwen3 file's inconsistency).
12. Pin which diagnostic fires in `expect_rejected` for the three
    order-dependent metadata rejections (sample-rate/hop/codebook mutations).

## With the synthesis/cloning slices (Plans 2–3 content decisions)

13. `ref/semantic_hidden.f32` probe captures HuBERT's final layer, but the
    codec consumes the mean of all 13 hidden states — decide the probe
    semantics before clone parity.
14. The quiet-reference volume branch (`0 < ref_rms < 0.1`) is untested by any
    golden case (pinned reference measures 0.123); needs a unit-level case.
    Fold the `ref_rms == 0.0` inverse-gate hazard (output multiplied by zero)
    into the same decision, and mirror whatever remains of upstream's
    unconditional peak-normalize-to-0.5 (measured at intake; all 18 no-ref
    golden cases peak at exactly 0.5).
15. Label the deliberate zero-ref-frames divergence in
    `tests/omnivoice_frontend_test.cpp` (upstream returns 1; this port treats
    zero frames as no reference). Record `class_temperature`'s upstream default
    and the auto-voice "voice follows the synthesis seed" clause in the family
    doc when next touched, plus the loader-refuses-missing-defaults obligation.
16. Re-verify the pre-tokenizer pattern against both `tokenizer.json` files
    when the qwen3-tts weights are next materialized (currently verified only
    against the committed transcription; recorded in three places).
17. Converter polish batch: `emitted_names()` test helper superset (838 vs
    798); config-structure `KeyError` guards; weight-norm fold clash assert;
    absent-package test discrimination; `verify_gguf` shape (not just element
    count) comparison; finiteness check in `numpy_of`; move `carry_licenses`
    existence check ahead of GGUF finalization (digest check already moved).
18. Restore the two rationale comments the BPE hoist dropped
    (`qwen_assistant_turn` cross-ref; `kAssistant*` 3/−5 provenance) and
    de-qwen3-ify the shared `bpe-frontend.h` prefix/suffix comment (the second
    consumer exists now).
19. Recheck the `> 0x20000` CJK Ext-B boundary (excludes U+20000 itself)
    against pinned upstream; one-line confirmation either way.
20. `tests/CMakeLists.txt` comment for `synthesize-vits-python-unit` predates
    OmniVoice; mention its converter tests ride the same discovery.

## Standing facts Plan 2 must not rediscover

- Greedy (both temperatures 0) is byte-identical across processes on CPU and
  makes zero RNG calls — the exact-token contract stands on real measurement.
- Duration weights must be summed with compensated (Neumaier) summation to
  match CPython 3.12 `sum()`; naive summation crosses `int()` truncation on
  8 of 15 oracle texts. Pinned by exact-equality tests.
- The 88-entry script range table is pinned by `static_assert`; the plan's "87"
  was wrong.
- Package GGUF (post-license-ruling) sha256:
  `3ecaa5e2f6fbd735296ba1cd60680c90467be22d2140dc4f208fe80111ecb9e5`
  [superseded 2026-08-03 by `f6d504ffaddcbf32f80f1f6c847f075bbd5d2c7b50fe95a194ceb635772f9fa3`,
  a metadata-only re-cut fixing a `max_output_frames` unit error (750 codec
  frames written where 720000 native PCM frames belonged); see the porting
  log's 2026-08-03 entry -- the digest above was correct as of this plan's own
  close and is left as the historical record];
  `general.license = "other"` per jiangzhuo's 2026-07-30 ruling (no invented
  CC version in machine-readable fields; ship-card frontmatter is a separate,
  later decision).
- Fresh worktrees need sibling-family artifacts (models/, build/goldens/,
  reports/convert/) symlinked or materialized before the two VITS-gated python
  tests pass; nothing omnivoice depends on them.
