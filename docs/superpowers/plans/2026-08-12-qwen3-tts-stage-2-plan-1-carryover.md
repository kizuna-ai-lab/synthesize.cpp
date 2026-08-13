# Qwen3-TTS Stage 2 Plan 2 — Carry-Over Ledger from Plan 1

Written at Plan 1 close (2026-08-12), from the whole-branch review of
`docs/superpowers/plans/2026-08-11-qwen3-tts-stage-2-plan-1-base-package.md`
and the branch's SDD ledger. That ledger lives under `.superpowers/`, which is
gitignored and does not survive the merge, so every controller ruling, deferred
minor and carried item that a later plan would otherwise have to re-derive from
branch history is written down here instead.

Plan 1 delivered the Base Model Package and the runtime that validates it:
intake and census, the Base oracle, variant-aware conversion (894 tensors),
metadata and tensor cataloguing for the speaker encoder and both codec halves,
a Golden Manifest with provisional tolerance keys, and public-seam behaviour
for a Catalog-less package. It delivered **no graph and no Voice Profile
preparation** — that is Plan 2. See
`docs/porting/families/qwen3-tts.md`, "Stage 2: Base Package, Plan 1", for the
measured facts; this file is the decisions and the debt.

The governing design record is
`docs/superpowers/specs/2026-08-11-qwen3-tts-stage-2-design.md`. Plan 2 is
written from that spec, and two of the rulings below are recorded as errata in
the spec itself for exactly that reason.

## 1. Controller rulings

### 1.1 The oracle samples with a fixed seed; it does not run greedy

Plan 1's Task 2 followed the spec's section 6, which prescribed
`do_sample=False, subtalker_dosample=False` "wherever a reproducible trajectory
is needed". The first case ran away to 8191 frames — about 655 seconds of audio
for one short sentence (`semantic.i32` 32764 bytes = 8191 int32, `acoustic.i32`
8191 × 15). It was killed and its artifacts discarded.

This reproduced a Stage 1 finding already recorded in
`docs/porting/families/qwen3-tts.md`: greedy "degenerates on some
speaker-and-input pairings" and was abandoned in Stage 1 for that reason. Every
Stage 1 Golden case runs both switches true at `max_new_tokens 2048` with
`stochastic_inputs` recorded for replay.

**Ruling:** the oracle samples with both switches true and a fixed seed at
`max_new_tokens 2048`, and records the sampled sequence for the replay seam.
Reproducibility comes from replaying the recorded draw, never from switching
the draw off. The same case then drew 43 frames.

Recorded as an erratum in the spec, section 6. **Plan 2 inherits the rule for
every new oracle it adds** (mel/speaker-encoder, codec-encoder and clone-prompt
dumps are deterministic and unaffected; anything that runs the talker is not).

### 1.2 `profile-sources`, not the plan's invented `profile-only`

Plan 1's Task 3 was written against a Voice mode string `profile-only` that
exists nowhere. The Golden Manifest schema's enum and the string OmniVoice's
loader already requires (`src/arch/omnivoice/weights.cpp`) are both
`profile-sources`.

**Ruling:** `profile-sources` everywhere — manifest, converter, and the C++
enum (`VoiceMode::ProfileSources`). Left as written, Task 6 would have emitted a
mode string no loader accepts.

One family difference is deliberate and must be preserved: OmniVoice *requires*
a package default (auto-voice is its default Voice); this variant must have
**none**, and `read_voices` refuses a `profile-sources` package that names one.

### 1.3 Stop deduplicating the codec halves; carry both in full

Task 5 implemented the spec's store-once path: the 16 encoder codebooks with a
same-index decoder counterpart were stored once under the decoder's name. The
result was a package whose encoder names never appeared at all — a consumer
found encoder quantizer stages 15–30 present and 0–14 absent — with the mapping
living only in a conversion report under a gitignored build directory. The
spec's own instruction to "record the mapping in the tensor catalog" was never
made durable. That is the alias nobody can look up.

Two facts settled it. The measurement is smaller than it read: the decoder
declares `quantizer_count 16`, so encoder stages 15–30 have no counterpart at
all — the true figure is 16 of 16 *existing* pairs, not 16 of 32. And the
saving is 33.5 MB of codebooks plus 2 MB of quantizer projections that were
duplicated too, about 1.4% of a 2.48 GB package.

**Ruling:** carry both halves in full, always; keep the measurement as a
recorded fact. `measure_shared_codebooks` in `scripts/convert-qwen3-tts.py`
still compares every `.codebook` on both sides at every conversion and records
what it finds without acting on it. Expected Base tensor count: **894**
(402 talker + 76 speaker encoder + 255 codec decoder + 161 codec encoder).

Recorded as an erratum in the spec, section 4. It binds any later variant of
this family whose two codec halves overlap, not just Base.

Note the one thing `measure_shared_codebooks` does *not* measure: the four
input/output projections on the shared quantizer stages. They were compared by
hand at Stage 1 and again during Stage 2's review and were identical; nothing
in committed code re-measures them. The converter docstring now says so
explicitly rather than implying the scan covers them.

### 1.4 The family advertises ZERO Voice Profile sources until preparation exists

Plan 1 shipped `fill_voice_profile_capability` advertising
`SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO | SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE`
with the six reference limits, on the argument — mine — that the missing
dispatch in `src/voice-profile.cpp` was "a dispatch gap, not a capability lie".

`docs/c-interface.md` does not leave that open: *"A Model without runtime Voice
Profile support reports zero flags"*, and for an unsupported source *"all fields
that describe that source are `SYNTH_REQUIREMENT_UNSUPPORTED` or zero"* — down
to a null `profile_schema`, zero schema version, and 32 zero compatibility-id
bytes. Nothing in the runtime can create or consume a Qwen3-TTS Profile:
`src/voice-profile.cpp` guards every source on
`family != ModelFamily::Omnivoice`.

**Ruling (2026-08-12, whole-branch review):** the confirmed contract wins over
the argument. This family reports zero source flags, and zero in every field
that describes a source, until preparation actually exists.

What did **not** change: the Base package still declares its full Voice Profile
contract, and `read_profile_contract` / `read_speaker_encoder` still read and
validate every `synthesize.profile.*`, `synthesize.reference.*` and
`synthesize.qwen3-tts.speaker_encoder.*` key at load time, refusing a package
that declares them badly. The package declaring a contract and the runtime
advertising a capability are different statements; only the second was false.

**What Plan 2 must do**, precisely:

- Publish `SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO` on the day
  `synth_voice_profile_create_from_reference` can actually prepare a Profile
  for this family, from `hparams.profile`'s already-validated limits, gated on
  `hparams.voice_mode == VoiceMode::ProfileSources`.

  **Corrected 2026-08-12, before Plan 2 was written.** This line first said
  "gated on `has_preset_voice_catalog(hparams)`", which is inverted: that
  predicate returns true only for `VoiceMode::PresetCatalog`
  (`src/arch/qwen3-tts/weights.h:243-245`), i.e. for CustomVoice — the variant
  that has no speaker encoder and cannot prepare anything. Implemented
  literally, Plan 2 would have advertised Reference Audio on the wrong variant
  and left Base advertising nothing.
  `tests/qwen3_tts_voice_required_test.cpp:155` already asserts the predicate is
  **false** for Base, so the trap was sitting in front of a test that names it.
  The mistake came from reusing the nearest existing predicate rather than
  naming the condition; the condition is the voice mode.
- Publish `SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE` **with it**, not as a
  separate decision: `docs/c-interface.md` requires that any Model which can
  create a v1 Profile also sets that bit, because every successfully prepared
  v1 Profile can be serialized.
- Rewrite (not delete)
  `tests/qwen3_tts_voice_required_test.cpp`'s
  `test_base_capability_advertises_nothing_until_preparation_exists` and
  `tests/qwen3_tts_base_load_real.cpp`'s `check_capabilities`, which assert the
  zero shape today and name Plan 2 as what flips them.
- Random Seed and Description Text stay unadvertised at this rung; Description
  Text is Stage 3's.

## 2. Carried items — things Plan 2 must revisit, not merely extend

### 2.1 The new catalog resolvers discard their pointers

`resolve_speaker_encoder` and the codec-encoder resolvers in
`src/arch/qwen3-tts/catalog.cpp` deliberately throw away the `ggml_tensor *`
they resolve, into a scratch struct that nothing reads. Plan 1 catalogues and
shape-checks those two regions; it does not build a graph over them, and a
resolver that stored its pointers into `ModelWeights` would have been storing
them for nobody.

**Every call site must be revisited when the graphs are built, not merely
extended.** The shape is deliberately not the same as the decoder's resolvers,
which do store what they find; a Plan 2 task that copies the decoder pattern
onto these without noticing will silently resolve into the scratch struct
again. Expect to change the resolver signatures, not to add to them.

### 2.2 The reference-duration bounds are safety ceilings with no listening pass

**RESOLVED 2026-08-13 by the Stage 2 Listening Audit. Both halves of this item
closed, and the second one closed in an unexpected way. The original text is
kept below unchanged; the resolution follows it.**

Task 6 shipped `min_frames_per_clip 24000` (1 s) and
`max_frames_per_clip`/`max_total_frames 720000` (30 s) at 24 kHz, with
`max_reference_count 1`. These are **safety ceilings taken from the plan, not
perceptually validated bounds.** Nothing in the package, the family record or
the tests claims otherwise, and the integration test asserts only that the ABI
reports them faithfully.

The measurement that exists (Task 2, artifacts under
`build/qwen3-tts-reference-bounds/`): 0.5 s, 1 s and 3 s references each
produced 2.5–3× the 45-frame baseline output, while a **30 s reference produced
only 9 output frames for an 11-word sentence**. That 30 s undershoot is
**unadjudicated** — nobody has listened to it, and `base-ref-max` was never
executed end to end. `intelligible: null`,
`perceptual_evaluation_performed: false`.

A listening pass on the 0.5 s / 1 s / 30 s renders is owed before these bounds
can be described as validated, and it is owed **before Stage 2 ships**, not
before Plan 2 starts. See the user memory note "Offer the listening pass before
shipping": a tolerance grid is not audible evidence.

#### Resolution, 2026-08-13

**The bounds.** The audit's labelled duration sweep found 1 s, 3 s, 10 s and
30 s references all **usable**, and the 0.5 s case **refused by the library**
(`voice_profile.reference_too_short`) rather than synthesized badly. The
shipped ceilings therefore produce usable speech across their whole declared
range and fail closed below it. They are no longer "safety ceilings with no
listening pass"; they are safety ceilings that a listener has since heard the
ends of.

**The 30 s undershoot, and why it is the interesting half.** It did not
reproduce. Regenerating that case **in x-vector mode** — the only mode this
port implements — gave **46 codec frames**, in line with every other duration
in the audit's sweep. The 9-frame figure recorded above came from a
**transcript-assisted (ICL) dump**. The observation was real and correctly
recorded; what was wrong was the implicit assumption that it described the path
Plan 2 built. Nothing in the shipped x-vector path could have produced it.

**Consequence.** The anomaly is not closed, it is **reassigned to Plan 3**,
which builds the ICL path and is the first plan that can drive the case that
produced it end to end. Do not treat this as a defect of the shipped bounds,
and do not treat the 46-frame result as evidence about ICL: they are two
different modes and only one of them has been listened to.

Full record, including the audit's method and what it deliberately does not
cover, is in `docs/porting/families/qwen3-tts.md` under "Listening Audits".

## 3. Deferred minors, by task

Carried verbatim in substance from the branch ledger. None were judged worth a
fix round at the time; all are still open unless marked.

**Task 2.** (a) No runtime assertion that the speaker embedding is 1024
elements, unlike the `shape[1] == 16` checks on codes. (b)
`load_cases_from_manifest`'s field mapping was unexercised until Task 3 landed
a manifest — now exercised, effectively closed.

**Task 3.** (a) **Closed in `86d6ad1`, before this ledger was written** —
`profileContract.sources` in the Golden Manifest was an unconstrained string
array where sibling schemas use enums; it now carries
`"enum": ["reference_audio", "serialized_profile"]`. (b)
`resolve_reference_locator` hardcodes its cache directory where the OmniVoice
pattern exposes a flag. (c) **Closed in this commit (Plan 2 Task 6)** — the
provisional `speaker_encoder` stage that copied `talker.*` probe names the
Base oracle never dumps was retired from
`tests/tolerances/qwen3-tts.json`'s `provisional_variants`; a real
`speaker.x_vector` probe now lives in
`variants.qwen3-tts-12hz-0-6b-base.profiles.BF16.stages.replay` instead.
`provisional_variants`'s sibling `codec_encoder` stage still carries the same
copied `talker.*` names — Plan 3's to close the same way, once it gives the
codec encoder a real measurement to replace it with.

**Task 4.** The "CustomVoice with a `speaker_encoder_config`" rejection branch
in `variant_profile` has no covering test; `display_name` / `size_label` are
hardcoded literals rather than derived from `tts_model_size`. Both inherited
from the plan's own code block.

**Task 6.** `source_artifact` matches by substring, and the Base manifest has
two checkpoint-role artifacts containing `model.safetensors`; it is correct only
incidentally. (`talker_checkpoint_locator` was added for the license link and
*is* order-independent — this is about the other call site, the pinned-digest
check in `main()`.)

**Task 8.** Three constants are declared twice (in the resolver and in
`expected_tensor_count`). Drift is caught by the catalog test, so this is
readability only.

**Closed by the final review, listed so they are not re-reported as open:**
the tautological `base > custom_voice + 76` assertion (now
`base - custom_voice == 76 + 161`); the dedup leftovers in the catalog test and
the plan document; the `.mlp_layer_scale.` shortening rule's missing test; the
tolerance file's file-wide `suite_version`/`status`; the Base manifest's
`source.repository` (it pinned the checkpoint repository where all five
siblings pin the code repository — now `QwenLM/Qwen3-TTS@022e286`, which
changes the Base GGUF's `synthesize.source.*` metadata but not its
Compatibility ID).

## 4. Standing facts Plan 2 must not rediscover

1. **The published CustomVoice package must stay byte-identical.** sha256
   `01dfad52dd507c26a14d101c4247d375257aa63b07e62706ec3daa0a33ea515d`,
   2,274,117,280 bytes, 657 tensors. It was built by
   `scripts/convert-qwen3-tts.py` and is live on Hugging Face; any converter
   change that alters its output silently invalidates a shipped artifact. Every
   converter task on this branch re-converted it and compared. Do the same.
2. **Base is 894 tensors**, 2,516,522,464 bytes, BF16 478 / F32 416,
   Profile Compatibility ID
   `34d4de22a329b6bc8347cb952b6fa16513320012628598ab59743679cc16806e`.
3. **A `unit`-labelled test may not depend on the real package** (2.5 GB) —
   `docs/testing.md`. Family-layer rules are unit-tested against synthetic
   `HParams`; anything needing the real GGUF is an integration test guarded on
   `SYNTH_QWEN3_TTS_BASE_TEST_MODEL`.
4. **Each Reference Model Variant owns its own Golden artifact root.**
   `build/goldens/qwen3-tts/<variant>/`, as VITS already did. The two variants
   shared one root until the final review; one case id common to both would
   have overwritten published goldens in place.
5. **`scripts/ci/clang-format.sh --check-diff` only sees TRACKED files.** A new
   test file that has not been `git add`ed is silently excluded — it reports
   "checking N file(s)" and exits 0. Task 10 shipped 8 violations that way. Run
   it after `git add`.
6. **The dev CMake presets compile ggml at `-O2`.** Any performance number for
   this family must record the build that produced it (user memory: "Record the
   build with every perf number").
7. **Deleting `shared.voice_profile = info.voice_profile;` in
   `src/synthesize.cpp` breaks nothing today.** With the snapshot all-zero
   (§1.4), the copy is a no-op and both the unit gate and
   `tests/qwen3_tts_base_load_real.cpp` stay green without it. The line stays
   because Plan 2 makes it carry something; its coverage returns with the first
   nonzero field. This is a known, disclosed gap, not an oversight to
   re-discover.
8. **`has_preset_voice_catalog` currently has no production caller.** Both of
   its call sites were removed by the final review — the capability gate
   (§1.4) and a redundant guard in `Model::resolve_voice` that returned the
   same status the next line already returned. It is kept because the unit
   tests assert the discriminator directly and Plan 2's capability gate is
   specified against it. If Plan 2 does not consume it, delete it.
9. **A Voice refusal is `synthesis.voice_unsupported`, not
   `synthesis.graph_failed`.** The ABI defines exactly one voice-error status,
   so the diagnostic code is the only thing that distinguishes a Voice refusal
   from a codec that failed to run. Any new refusal path Plan 2 adds at the
   `src/synthesize.cpp` seam should follow it rather than falling into the
   generic branch.
