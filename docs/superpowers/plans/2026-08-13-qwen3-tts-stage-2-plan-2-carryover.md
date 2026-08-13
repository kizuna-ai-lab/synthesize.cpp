# Qwen3-TTS Stage 2 Plan 3 — Carry-Over Ledger from Plan 2

Written at Plan 2 close (2026-08-13), after the PR #10 review, the Listening
Audit, and the deferred-findings clearance that followed the merge. Like its
predecessor
(`docs/superpowers/plans/2026-08-12-qwen3-tts-stage-2-plan-1-carryover.md`),
this file exists because the SDD ledger lives under `.superpowers/`, which is
gitignored and does not survive the merge.

Plan 2 delivered the x-vector cloning path: the mel front end, the ECAPA-TDNN
speaker-encoder graph, Voice Profile preparation with the v1 GGUF envelope, the
public seam, and prompt-slot substitution. It delivered **no transcript-assisted
(ICL) cloning** — that is Plan 3. See `docs/porting/families/qwen3-tts.md`,
"Stage 2: Base Package, Plan 2" and "Listening Audits", for the measured facts;
this file is the decisions and the debt.

## 0. A process failure this file did not prevent

Plan 1's carryover was written *before* its workspace was deleted. Plan 2's was
not: the whole-branch review ran, the workspace was deleted as the skill's final
step prescribes, and only then was this file started. **Plan 2's per-task
deferred minors are therefore unrecoverable.** They are not absent because there
were none; they are absent because nobody copied them out in time.

What survived did so by accident of location — the PR-stage triage
(`pr10-triage.md`) was written outside the plan workspace, so the five findings
it deferred were still readable after the merge, and §1 below is the record of
clearing them.

Write the carry-over before deleting the workspace, not after.

## 1. Controller rulings, deferred-findings clearance (2026-08-13)

Five review findings from PR #10 were triaged as real-but-not-blocking and
cleared on branch `qwen3-tts-deferred-findings` after the merge. Three were
fixed, one was refused, and one reviewer claim was rejected outright.

### 1.1 The GGUF type tag is compared as an `int32_t`, never cast — and no test pins that

An out-of-range type tag read from a user-supplied file was cast to `gguf_type`
before any range check. `gguf_type` is an unscoped enum with no fixed underlying
type, so the conversion is undefined behaviour.

The reviewer proposed rejecting values `< 0 || >= GGUF_TYPE_COUNT`. **That was
refused.** It is a range check against ggml's enum — the blacklist shape this
project formally deleted after two crashes and 8,127 fuzz iterations
(`docs/porting/families/omnivoice.md`, and the binding rule in
`docs/superpowers/plans/2026-08-02-omnivoice-plan-4-carryover.md` §213-221). It
would also depend on a constant owned by a submodule that cannot carry a local
change.

**Ruling:** keep the value as `int32_t` and compare it against
`int32_t(spec.type)` — the pattern already present two functions further down in
both `src/arch/qwen3-tts/profile.cpp` and `src/arch/omnivoice/profile.cpp`. No
cast to the enum ever happens, and the check is positive: does this equal the
one type our own writer emits?

**The honest limit, which the tests state in their own comments:** reinstating
the cast leaves both the standard and the sanitizer gate green. It was measured,
not assumed. GCC 13.3's UBSan does not instrument this register-resident enum
conversion; no clang is available on this machine. The new test arms pin the
black-box rejection behaviour and nothing more. **A future simplifier who
reintroduces the cast will not be stopped by any test in this repository** —
which is exactly why the reasoning is written at the call site rather than only
here.

### 1.2 A package promising more Reference Audio clips than the runtime reads is refused at load

`max_reference_count` was published to callers straight from package metadata
while profile creation unconditionally rejected every `reference_count != 1`. A
caller could follow the advertised capability and still be refused. Separately,
the internal `VoiceProfileInfo` field was `uint32_t` against a `uint64_t`
contract, so a declared value of exactly 2^32 truncated to 0 — the value meaning
"no references allowed".

**Ruling:** both families require `max_reference_count == 1` at load, and the
internal field is widened to `uint64_t` so no narrowing remains. The
`reference_count != 1` rejection is **not** weakened — it is deliberate and
documented in two places.

**The published packages were verified, not assumed.** Read out of the real
files rather than inferred from the conversion scripts:

| Published package | `max_reference_count` | Under the new rule |
| --- | --- | --- |
| `jiangzhuo9357/omnivoice-0-6b-gguf` (F16) | 1 | accepted |
| `jiangzhuo9357/qwen3-tts-12hz-0-6b-customvoice-gguf` (BF16) | *no `synthesize.reference.*` keys at all* | rule never runs — it is a `preset-catalog` package |
| `qwen3-tts-12hz-0-6b-base` (BF16, local) | 1 | accepted |

The public ABI field in `include/synthesize.h` was **already** `uint64_t`; the
narrowing was internal, so no ABI or struct-size test needed updating. An
earlier reviewer claim that the public field was the narrow one is wrong.

`docs/schemas/synthesize-golden-manifest-v1.schema.json` was tightened from
`"minimum": 1` to `"const": 1` to match — it had drifted into validating a
manifest the loader now refuses.

### 1.3 WAV chunks are sized against the file, not against the header's claim

Four test readers trusted a 32-bit `chunk_size` before validating it, and two of
them compared the 12-byte RIFF header without checking the stream state first.

**Ruling:** fixed in place in all four. **No shared helper was extracted** — the
duplication between `qwen3_tts_mel_driver.cpp` and `qwen3_tts_xvector_driver.cpp`
is a recorded decision, and the two copies remain byte-identical (137 lines,
clean `diff`).

### 1.4 The mel filterbank gets no band guard — refused, with an entry condition for Plan 3

The reviewer asked for a guard in `build_mel_filterbank` against
`params.fmax <= params.fmin`, which divides by zero, at Major severity, arguing
that tests and Plan 3 call the function directly without the load-time gate that
`read_speaker_encoder` applies.

**Refused, on three grounds:**

1. `build_mel_filterbank` is in an anonymous namespace and **cannot be called
   directly from outside its translation unit**. The comments the reviewer cited
   as evidence of direct callers belong to a different function,
   `compute_log_mel`. The finding's central justification is factually wrong.
2. The division by zero occurs only at `fmax == fmin`. At `fmax < fmin` the
   result is finite garbage, not NaN — so the `<=` in the finding is imprecise.
3. A guard no caller can reach admits no test that fails when the guard is
   deleted. This repository has twice shipped a rule whose test stayed green
   after the rule was removed, and both times it was treated as a defect in its
   own right. Adding a third instance to satisfy a Major label would be the
   wrong trade.

**Entry condition for Plan 3:** if Plan 3 exposes this function — most likely by
giving Voice Profile preparation its own mel path — the guard goes in *together
with* a test written from the newly reachable caller, and that test must be
checked by deleting the guard and confirming it fails. Not before.

## 2. Carried items — things Plan 3 must revisit, not merely extend

### 2.1 ICL cloning needs the codec encoder, which does not exist yet

Plan 2 implements `x_vector_only_mode` alone. Transcript-assisted cloning needs
a codec encoder graph and the two-track prompt, neither of which Plan 2 built.

**On the tensor count, since two numbers are both correct and get confused:**
225 is the codec encoder's *raw* safetensors count; **161 is what the converter
emits** (225 − 32 collapsed EMA-accumulator pairs − 32 `.initialized` shape-(1,)
flags). A graph is built from the emitted set, so 161 is the number that matters
to Plan 3. The family record's 225 appears in the raw-tensor accounting and is
right there.

Plan 1 catalogued those tensors but deliberately went no further:
`resolve_codec_encoder` discards every pointer into a `Conv1dWeights` scratch,
there is no `CodecEncoderWeights` type, and no graph exists. Cataloguing is not
a head start on the graph — it only proves the tensors are present and named.

One more thing Plan 3 must not discover late: the codec encoder's reference
implementation is `transformers`' `MimiModel`, not a Qwen source file. That is a
**third** provenance, after the Qwen source and the checkpoint, and no prior
stage of this port has read it.

### 2.2 The 9-frame anomaly belongs to Plan 3

Plan 1 carried an unexplained 30 s reference producing 9 codec frames. It was
retired on 2026-08-13: regenerated in x-vector mode the same reference yields 46
frames, and the 9-frame figure came from an ICL-mode dump — a mode this port does
not implement. The regeneration is **not evidence about the ICL path**; it only
removes the anomaly from Plan 2's ledger. Plan 3 inherits it unresolved.

## 3. Standing facts Plan 3 must not rediscover

**Only one of the four WAV readers is built by a gate.** This cost two review
rounds to establish, so it is written here plainly:

| Reader | Built by `synthesize-check-unit`? |
| --- | --- |
| `tests/qwen3_tts_mel_driver.cpp` | yes, via `add_dependencies` |
| `tests/qwen3_tts_xvector_driver.cpp` | **no** — registers as an integration target only |
| `tests/qwen3_tts_clone_real.cpp` | **no** — `add_executable` guarded on model existence |
| `tests/omnivoice_profile_test.cpp` | **no** — same guard |

Established by deleting `build/bin/synthesize-qwen3-tts-xvector-driver`, running
the unit gate to completion, and observing that it was not recreated. Reading
`tests/CMakeLists.txt` alone produced the wrong answer twice.

**Two unit-gate tests fail in a fresh checkout** and are unrelated to any of this
work: `synthesize-python-api-wheel-test` and `synthesize-vits-python-unit`, both
on gitignored VITS artifacts that are not committed. They fail identically at
`322784d`. Do not spend a debugging session on them.

**The Listening Audit of 2026-08-13 is evidence, not a Validation Level.** One
listener, four blind port-vs-oracle pairs, four reference durations. It does not
move `quality_evaluation`, which ADR 0017 defers. See
`docs/porting/families/qwen3-tts.md`, "Listening Audits".

**The Base package is not published.** As of 2026-08-13 the account carries five
GGUF packages — two VITS, Kokoro, Qwen3-TTS CustomVoice, and OmniVoice. Stage 2's
publishable artifacts are built and audited but not uploaded, and uploading is a
separate act requiring its own confirmation.
