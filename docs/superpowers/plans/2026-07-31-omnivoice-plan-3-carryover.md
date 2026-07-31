# OmniVoice Plan 3 — Carry-Over Ledger from Plan 2

Written at Plan 2 close (2026-07-31), from the final whole-branch review's
triage. Plan 2 landed the synthesis core (oracle hardening → generator →
greedy decode loop → codec → replay harness → first committed tolerances);
Plan 3 (public sampling + cloning) and Plan 4 (quants/backends/ship) consume
this register.

## Plan 3 — first touch of the named files

1. `choose_token`: assert instead of silently taking the unguided branch when
   `uncond == nullptr` with nonzero guidance (`generator-host.cpp`; the header
   documents the contract; the sampler rework is the natural moment).
2. Decode-loop placement assertion: tighten from >4N to ~2·steps·N; cover the
   `guidance == 0` loop branch (comes free with the sampler tests).
3. Delete the unread `GraphRun::nodes_` member (qwen3-symmetric one-liner).
4. Harmonize the two `verify_pinned_inputs` implementations (converter's
   PinnedInput list vs the dumper's /resolve/-marker inference) — one pinned
   set, two consumers.
5. `tests/tolerances/qwen3-tts.json` still carries `suite_version: 1` against
   its manifest's 2 — same-class staleness fixed for omnivoice at 0887553;
   one-line sibling correction with a log note.

## Plan 3 — content decisions carried from earlier reviews

6. `ref/semantic_hidden.f32` probe captures HuBERT's final layer but the codec
   consumes the mean of all 13 hidden states — settle the probe semantics
   before clone parity (T6-era note, still open).
7. The quiet-reference volume branch (`0 < ref_rms < 0.1`) has no golden case
   (pinned reference measures 0.123); needs a unit-level case, and the
   `ref_rms == 0.0` inverse-gate hazard (output multiplied to zero) needs its
   recorded disposition alongside.
8. Sampled-path margin screening: the greedy margin table in the porting log
   is the baseline; Gumbel sampling changes the landscape — decide whether the
   sampled seed-contract cases need their own screen.

## Plan 4 — hardening and ship

9. Pin primary-grid digests per case in the manifest (symmetry with the
   alternate-grid entries; today the primary is whatever the local oracle dump
   holds — divergence fails loud, but the digests are known and recordable).
10. Converter items from T3 (verify_gguf shape check, license-copy reorder)
    still end-to-end-only — close in the converter harness pass.
11. Quantization margin protocol: before any quantized profile is judged,
    re-run `--margin-report` and compare against the greedy baseline; the four
    in-band cases (`omni-short-en`, `omni-long-boundary`, `omni-rate-fast`,
    `omni-lang-none`, margins 1.16e-04–6.03e-04) are the predicted first
    flips, and `omni-rate-slow` (9.5e-06) flips almost surely — the
    dual-admissibility mechanism (enumerated digests) is the recorded remedy
    path, never a tolerance.
12. Deep-probe cosine gates sit at 5.1×–11× measured because the validator's
    float32 cosine estimator noise floor (2.08e-07) dominates; if sharper
    gates are wanted, make `cosine()` accumulate in float64 and re-measure.
13. The listening pass is still owed before ship: numerically the port adds
    nothing audible (102–119 dB error SNR), but the blind A/B against the
    reference is jiangzhuo's, at ship time, with prepared clip pairs.

## Standing facts Plan 3 must not rediscover

- The commit schedule MUST be computed in float32 replicating torch's
  elementwise ops (upstream linspace default dtype); double diverges on ~1.2%
  of canvas lengths. Pinned by the `commit_schedule(1640, 32, 0.1)` regression
  fixture; the 1/N-binary-exact caveat is in the code comment.
- The embedding merge is a torch.where SELECT (audio suffix carries summed
  offset codebook embeddings alone). The family doc is correct now; the
  design spec carries dated amendments.
- Greedy makes zero RNG calls; the exact-token contract holds at 17/17 under
  the dual-admissibility ruling (fast-mode: two enumerated torch-produced
  grids; margin screen ≥1e-4 for new case selection, standard in the family
  doc).
- Upstream has NO seed parameter — Plan 3's public sampling uses this port's
  own streams (random-stream.h has the distributions; Gumbel = −log(−log U)).
- Oracle re-dumps after any case-text change must re-run EVERY oracle the
  family has (PyTorch case dumper AND tokenizer dumper) — the rule is in the
  porting log.
- Golden suite is at revision 2 (clone-zh re-pick + alternate-grid entry);
  tolerances measured against revision 2.
