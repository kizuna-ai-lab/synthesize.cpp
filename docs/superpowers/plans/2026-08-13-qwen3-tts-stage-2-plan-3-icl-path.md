# Qwen3-TTS Stage 2, Plan 3: the ICL Path — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transcript-assisted (in-context-learning) voice cloning, end to end, on CPU — the codec encoder graph over the package's 161 emitted `codec.encoder.*` tensors turning a 24 kHz reference waveform into `[16, T]` reference codes, a reference-transcript tokenizer, the two-track ICL prompt block prepended to the talker's prefill, an ICL Voice Profile payload and its envelope extension, and the public seam flipped so `reference_transcript` is `SYNTH_REQUIREMENT_OPTIONAL` and its presence selects the mode.

**What Plan 3 delivers.** The second of the two clone modes D1 names. After it, both modes exist, `reference_transcript` reaches the `OPTIONAL` state section 3 of the design describes, and the Base Golden Manifest's ten ICL cases become drivable — today only two of its twelve are. The spec's §8 gate for this plan is: "ICL clone runs end to end; codes and prompt layout meet oracle tolerances; Golden Manifest covers both modes." **Read that clause through the spec's fourth erratum (2026-08-13):** "codes … meet oracle tolerances" is the codec encoder's stage-wise F32 artifacts and its dequantized reconstruction, at a bf16-derived tolerance. The discrete code indices are not a gate.

**What Plan 3 does NOT deliver.** Quantization Profiles and the CUDA Execution Backend are Plan 4's, by the design's §7 — this plan measures no Quantization Profile, claims no performance number, and adds no `Q8_MIXED`/`F16`/`CUDA` cell to the tolerance grid. It does not move the Validation Level: `port_validated` plus a listening audit is the delivery bar (design §10), `quality_evaluation` stays deferred per ADR 0017, and the Listening Audit of 2026-08-13 is evidence, not a level. It does not publish anything. It does not touch `examples/cli/`, and it adds no `SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT` (Stage 3).

**Architecture:** Plan 1 catalogued the codec encoder and deliberately kept nothing: `resolve_codec_encoder` (`src/arch/qwen3-tts/catalog.cpp:516-580`) writes every resolved pointer into one reused `Conv1dWeights scratch;` and its own header comment names Plan 3 as the owner of the conversion. Plan 2 did the identical conversion for the speaker encoder, so the shape of the change is known — but cataloguing gave the codec encoder **no head start** beyond proving the names and shapes are present (Plan 2 carryover §2.1). Four pieces are new. The graph is a Mimi-style SEANet stack (stem, four downsampling stages, tail conv), a frame downsampler, an eight-layer transformer, and a split residual vector quantizer whose argmin runs on the host. The prompt change is **structural, not additive**: `TalkerInputPosition` (`src/arch/qwen3-tts/talker-host.h:19-32`) carries exactly one `uint32_t codec_token` per position, and an ICL reference block carries sixteen codes per position, so the representation itself changes. The profile and envelope extensions land on seams that were pre-cut for them — `CloneMode`, `PrescanKeyScope`, `kPrescanKnownKeys`, `kPrescanKvCountXVector` each carry an in-file comment naming what Plan 3 adds. The public seam is a flip of two `SYNTH_REQUIREMENT_UNSUPPORTED` values and the conversion of two rejections into a mode selector.

**Tech Stack:** C++17 with GGML/GGUF; CTest under the `unit` and `integration` labels; Python 3.12 + torch + `transformers==4.57.3` in the locked `scripts/envs/qwen3-tts` environment for the oracle; `uv` for every Python entry point.

---

## The validation strategy, stated first because it shapes every task

**A Listening Audit cannot gate this plan.** Plan 2 could lean on one: an x-vector that is wrong in its last decimal place still produces speech, but an x-vector that is wrong in *structure* produces the wrong voice, and a listener hears that. ICL is not like that. The research established the property plainly, and this family's own source comments have been warning about it since Stage 1 (`src/arch/qwen3-tts/talker-host.h:76-80`: *"A prompt off by one still synthesizes speech, in the wrong voice or the wrong language"*). A two-track prompt whose alignment is wrong — text padded where it should have been truncated, the codec track offset by one, the sixteen groups summed in the wrong order — produces fluent, plausible, correctly-languaged speech in approximately the reference voice. No listener can distinguish that from a correct implementation, and neither can any end-to-end audio tolerance, because the talker is sampled and the audio was never going to match sample-for-sample anyway.

**So the alignment is pinned numerically, stage by stage, against the oracle, and nowhere else.** Concretely, and each of these is a task obligation rather than an aspiration:

1. Every stage inside the codec encoder gets its own oracle tap and its own committed comparison — SEANet stage outputs, the frame downsampler's output, per-transformer-layer hidden states, the RVQ's per-stage residual, its per-stage assignment distances, and the dequantized reconstruction the selected rows sum to. A wrong SEANet and a wrong transformer cancelling into plausible codes is the exact failure Plan 2's Task 1 was written to prevent for the speaker path, and it is a sharper risk here because the encoder's output is discrete and therefore *quantizes away* small errors until it does not.
2. **The reference codes are not a gate, and the stage-wise F32 comparison carries the whole load in their place.** The design's §6 prescribed plain equality; its **fourth erratum (2026-08-13)** drops that gate on a measurement, and jiangzhuo's ruling of the same date is what this plan implements. The short form: the oracle's RVQ codebook is bfloat16 (the tokenizer loads at the talker's dtype, so `embed_sum / cluster_usage` is computed in bf16), the converter bakes float32, and over 146 clips / 49,507 frames / 792,112 frame-stage decisions the two tables disagree on **4.04%** of codes for the argmin alone, **12.73%** with the f32 table throughout the RVQ, at a per-decision flip rate of **0.624%** — on 136 of 146 clips, every stage including the semantic one. Upstream is not even self-consistent: loading without `dtype=torch.bfloat16` gives an f32 table that disagrees with the committed `codes/reference.i32` on **794 of 1616 codes**, so equality tests a load-time keyword argument. The gate is therefore: **the stage-wise F32 artifacts compared at a bf16-derived tolerance, plus the dequantized reconstruction** — the sum of the selected codebook rows in the 256-wide projected space, compared as F32.
   **Two things this does not license.** It is not a claim that the port is correct because its codes differ by only a few percent — those percentages bound the *codebook's* contribution alone, the port's F32 SEANet and transformer activations are the larger contributor, and 0.624%/4.04% are a floor rather than a budget. And it is not a relaxation of anything: a stage whose measured deviation exceeds the bf16-derived expectation is a defect this plan must find, never a tolerance to widen. The code agreement rate is still measured and recorded per case; it decides nothing.
3. The two-track prompt block is compared against the oracle's `prompt/icl_embed.f32` as F32 tensors — the finished summed block — **and** against a new per-track dump, because the summed block alone cannot distinguish a text-track error that a codec-track error compensates.
4. The alignment *branch* is dumped explicitly and asserted, because it is not recoverable from any artifact that exists. See the next paragraph.

**The unmeasured branch.** Upstream's alignment rule (`qwen_tts/core/models/modeling_qwen3_tts.py:2015-2019`) has two arms: when `T1 > T2` the text track is truncated to `T2` and its tail becomes the trailing schedule; otherwise the text track is padded up to `T2` with `tts_pad_embed` and the trailing schedule is a bare `tts_pad_embed`.

**This plan asserts nothing about which arm any on-disk artifact takes, and no task may be written as though it knew.** The research offered a claim about arm coverage; it is unverifiable and is therefore not carried here. The reason is structural rather than a gap in diligence: the summed block is `T2` positions long in *both* arms, so `prompt/icl_embed.f32`'s size is silent about which one ran, and the only discriminator is the trailing schedule, which nothing on disk records. An implementer who reads an arm off the artifacts has read an inference. Two obligations follow, and neither is optional:

- **Task 2** dumps `T1`, `T2`, the branch taken, and the trailing schedule into its `alignment.json`, so the branch becomes a measurement rather than an inference. It is Task 2's and not Task 1's because the branch is a property of the two-track prompt construction: it is decided by `T1`, which is a function of the reference *and target text* token counts, and Task 1's codec-encoder oracle never sees any text at all.
- Task 12 requires at least one measured case in **each** arm, keyed to what Task 2 reported and to nothing else. Two candidates are worth checking first — `base-ref-min` (1 s reference, 13 reference frames, `T2 = 14`) for the truncating arm and `base-icl-en` (101 reference frames, `T2 = 102`) for the padding arm — but a candidate is a place to look, not a finding, and if Task 2's `alignment.json` contradicts either one, `alignment.json` is right.

---

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-11-qwen3-tts-stage-2-design.md`, sections 3–7 and 9, **including its four errata** — the fourth, of 2026-08-13, replaces §6's plain-equality gate on reference codes and governs Tasks 1, 5, 12 and 13. Carry-over ledgers: `docs/superpowers/plans/2026-08-13-qwen3-tts-stage-2-plan-2-carryover.md` and `docs/superpowers/plans/2026-08-12-qwen3-tts-stage-2-plan-1-carryover.md`. Structural template: `docs/superpowers/plans/2026-08-12-qwen3-tts-stage-2-plan-2-xvector-path.md`.
- **The codec encoder's reference implementation is `transformers`' `MimiModel`, not a Qwen source file.** `Qwen3TTSTokenizerV2Encoder` subclasses it with the decoder halves nulled (`qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py:26, 899-908`); the encode entry point is `Qwen3TTSTokenizerV2Model.encode` at `:961-988`. Every convention this plan must transcribe — the ELU activations (`transformers/models/mimi/modeling_mimi.py:302, 334, 338`), `MimiConv1d`'s causal padding (`:124`), the residual-unit dilations, `MimiSplitResidualVectorQuantizer.encode` (`:1357`) — lives in `modeling_mimi.py`. This is a **third provenance**, after `QwenLM/Qwen3-TTS@022e286` and the checkpoint itself, and no prior stage of this port has read it. A convention read from the wrong one of the three produces finite codes that decode to plausible audio. The oracle records `transformers==4.57.3` and a `file:line` for every convention it transcribes.
- **Published packages stay byte-identical.** This plan touches no converter code. If a task concludes that it must — the only live candidate is publishing a `synthesize.qwen3-tts.codec.encoder.*` metadata namespace, which does not exist today (research §1e) — then it is a package re-cut, and the task carries the proof: CustomVoice still converts to sha256 `01dfad52dd507c26a14d101c4247d375257aa63b07e62706ec3daa0a33ea515d` (2,274,117,280 bytes, 657 tensors), and the Base package's Profile Compatibility ID `34d4de22a329b6bc8347cb952b6fa16513320012628598ab59743679cc16806e` is unchanged or the change is the task's headline. **The default is a compiled-in literal beside the existing ones** (`catalog.cpp:429-434`), under the rule that file already states: a width that IS declared upstream is read from metadata; a width that is a property of the architecture is a literal.
- **`ggml/` is a submodule and cannot carry a local change.** It is a checkout of `ggml-org/ggml`; an edit there vanishes at the next `git submodule update`. If the codec encoder needs an operator ggml does not have, the answer is a different upstream operator, not a patched kernel — that is what emptied this project's patch set (`ggml-patches/README.md`), and other GGML TTS ports are worth reading first (`qwentts.cpp` handed this project the `col2im_1d` recipe).
- **Every test this plan writes must be load-bearing, and each is proved so by deleting the rule it claims to pin.** This is not boilerplate. Four tests in this repository have passed for the wrong reason: two shipped rules whose tests stayed green after the rule was removed (Plan 2 carryover §1.4); the GGUF type-tag arms, which the carryover discloses cannot catch a reinstated cast (§1.1); and the tautological `base > custom_voice + 76` tensor-count assertion, closed by a review (Plan 1 carryover §3). Plan 1's discard-into-scratch resolver is the same failure in a different costume — it resolves every name and keeps none, and every catalog sweep passes. **Procedure:** after a test goes green, delete or invert the rule under it, re-run, confirm the test fails, restore the rule. Record the confirmation in the task's commit message. A test that stays green is not a test.
- **The sanitizer gate runs for every task that touches inference code** — Tasks 2, 3, 4, 5, 6, 7, 8, 9, 10, 13.

  ```bash
  cmake --build build-sanitize --target synthesize-check-unit
  ```
- **`scripts/ci/clang-format.sh --check-diff origin/main` only checks TRACKED files.** Run it *after* `git add`. Plan 1's Task 10 shipped 8 violations by running it before.
- **Unit gate baseline: two known failures.** `synthesize-python-api-wheel-test` and `synthesize-vits-python-unit` fail in a fresh worktree on gitignored VITS artifacts. Any third failure is yours. Run `cmake --build build --target synthesize-check-unit` after every task.
- **A `unit`-labelled test may not depend on the real 2.5 GB package.** Family rules are unit-tested against synthetic `HParams` and in-memory LCG weights; anything needing the real GGUF is an integration test guarded on `SYNTH_QWEN3_TTS_BASE_TEST_MODEL`. Note that of the four WAV readers only `tests/qwen3_tts_mel_driver.cpp` is built by `synthesize-check-unit` (Plan 2 carryover §3) — reading `tests/CMakeLists.txt` produced the wrong answer twice.
- **The Profile Schema does not change identity or version.** `qwen3-tts-voice-clone`, version 1, in-envelope `synthesize.voice_profile.kind` gains the value `icl`. No version bump — discriminating on `kind` is the entire point (`src/arch/qwen3-tts/profile.h:131-145`). **Every Plan 2 `x-vector` profile stays loadable under a Plan 3 build, and Task 9's tamper matrix asserts it.**
- **The package's declared contract does not change.** The design is explicit (`spec:177-178`): *"This is a statement about the runtime, not the package: the Base package's declared contract is unchanged, and its `reference_*` limits stay as validated."* Concretely: `read_profile_contract` / `read_speaker_encoder` keep reading and validating every `synthesize.profile.*`, `synthesize.reference.*` and `synthesize.qwen3-tts.speaker_encoder.*` key at load; the six reference limits stay (`target_sample_rate` 24000, `target_channels` 1, `min_frames_per_clip` 24000, `max_frames_per_clip` 720000, `max_total_frames` 720000, `max_reference_count` 1); `source_flags` stays **exactly** `SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO | SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE` (declared at `spec:154-156`, its Plan 1 history at `:183-184`, its pairing rule at `:206-209`; the advertised source does not change when the second mode lands, which `spec:410-413` states from the plan-sequence side); Description Text and Random Seed stay unadvertised.
- **Do not weaken the `max_reference_count == 1` enforcement** at load for either family, nor `docs/schemas/synthesize-golden-manifest-v1.schema.json`'s `"const": 1` (Plan 2 carryover §1.2).
- **A Voice refusal carries `synthesis.voice_unsupported`, never `synthesis.graph_failed`** (Plan 1 carryover §4.9). The ABI defines exactly one voice-error status, so the diagnostic code is the only thing that distinguishes the two.
- **Nothing under `models/` or `build/` is ever committed.** Oracle artifacts, GGUFs, and reports live there and stay ignored; the Golden Manifest and the tolerance file are the committed contract.
- **Any performance number names its build.** The `dev-*` presets compile GGML at `-O2`. This plan claims none; Plan 4 does.
- **The CLI is out of scope.** `examples/cli/` has no Voice Profile support and no audio reader for any family; adding one is a cross-family slice (jiangzhuo's ruling 2026-08-12, `docs/porting/families/qwen3-tts.md`).
- **The Base package is not published, and publication is a separate act requiring jiangzhuo's confirmation at the time.** Nothing in this plan uploads anything or asks to.
- **Mel-guard entry condition, inherited from Plan 2 carryover §1.4.** If any task here exposes `build_mel_filterbank` beyond its anonymous namespace, the `fmax == fmin` divide-by-zero guard lands *together with* a test written from the newly reachable caller, and that test is checked by deleting the guard and confirming it fails — not before. **This most likely does not fire:** the codec encoder consumes the raw 24 kHz waveform, not mel (`spec:68`; `modeling_qwen3_tts_tokenizer_v2.py:961`). If a task finds itself near that function anyway, the condition is live and this is the rule.
- Base package facts that must hold: **894 tensors**, of which the codec encoder is **161 emitted** — `1 + 28 + 96 + 3 + 33`, arithmetic at `catalog.cpp:621-625`. The design's §1.1 row saying 225 is the *raw safetensors* count. The family record's census (`docs/porting/families/qwen3-tts.md:1983-1995`) is **whole-package** — `402 + 76 + 271 + 225 = 974`, less 48 collapsed EMA-accumulator pairs (16 decoder-layer + 32 encoder-layer) and 32 encoder `.initialized` flags, `= 894` — and **the number 161 appears nowhere in that file**. The encoder's own share, `225 − 32 encoder EMA pairs − 32 .initialized = 161`, is derivable from that census but is not recorded, which is why Task 14 Step 2 has to write it down rather than merely reword. **A graph is built from the emitted set, so 161 is the number that matters here.** `expected_tensor_count` must not move.
- **Plan 2's per-task deferred minors did not survive, and that qualifies the baseline above.** Plan 2 carryover §0 records that its workspace was deleted before the carryover was written, so only the five PR #10 triage findings came through; the per-task minor debt is unrecoverable and of unknown size. "Any third failure is yours" is therefore a working rule, not a proof: a unit-gate failure that traces to a Plan 2 minor is still a real finding, and it gets recorded in this plan's carryover rather than argued away.

---

## File Structure

| File | Status | Responsibility |
| --- | --- | --- |
| `scripts/dump_reference_qwen3_tts_codec_encoder.py` | create | Stage-wise oracle for the codec encoder: SEANet stages, downsampler, per-layer transformer states, RVQ residuals and distance margins |
| `scripts/dump_reference_qwen3_tts_icl_prompt.py` | create | Stage-wise oracle for the two-track prompt: text track, codec track, `T1`/`T2`, the branch taken, the trailing schedule |
| `src/arch/qwen3-tts/codec-encoder.h` / `.cpp` | create | GGML graph: waveform → `[codebook_dim, frames]` latents (SEANet + downsampler + transformer) |
| `src/arch/qwen3-tts/codec-encoder-host.h` / `.cpp` | create | Frame arithmetic against 1920, the host-side RVQ argmin, the 16-of-32 slice, code range checks, the dequantized reconstruction the gate compares, tie-margin instrumentation |
| `src/arch/qwen3-tts/catalog.h` / `.cpp` | modify | `CodecEncoderWeights` on `ModelWeights`; `resolve_codec_encoder` keeps its pointers; `expected_tensor_count` unchanged in value |
| `src/arch/qwen3-tts/bpe.h` / `.cpp` | modify | The reference-turn wrapper and its own slice constants |
| `src/arch/qwen3-tts/code-predictor.h` / `.cpp` | modify | `sum_code_embeddings` over `T` positions, not one |
| `src/arch/qwen3-tts/talker-host.h` / `.cpp` | modify | A per-position code *group* representation; the ICL block and the `min(T1,T2)` alignment |
| `src/arch/qwen3-tts/talker.h` / `.cpp` | modify | The ICL block concatenated into the prefill, replacing the first-text-token position |
| `src/arch/qwen3-tts/profile.h` / `.cpp` | modify | `CloneMode::Icl`, `IclProfile`, `create_icl_profile`, `serialize_icl_profile`, the `kIclOnly` scope, `kPrescanKvCountIcl` |
| `src/arch/qwen3-tts/qwen3-tts.h`, `model.cpp` | modify | `Model::prepare_codec_reference`, `codec_encoder_weights()`, the ICL fields on `SynthesisRequest`, the reference-transcript entry point |
| `src/arch/qwen3-tts/weights.h` / `.cpp` | modify | `reference_transcript` / `reference_language` → `SYNTH_REQUIREMENT_OPTIONAL` |
| `src/voice-profile.cpp` | modify | The two rejections become the mode selector; the language tag becomes validated rather than refused |
| `src/synthesize.cpp` | modify | The `CloneMode` refusal becomes a dispatch |
| `tests/qwen3_tts_codec_encoder_test.cpp` | create | Graph shapes, frame arithmetic, RVQ rules — synthetic weights, no GGUF |
| `tests/qwen3_tts_icl_prompt_test.cpp` | create | Two-track layout, both alignment arms, the trailing schedule |
| `tests/qwen3_tts_bpe_test.cpp` | modify | The reference turn is not the assistant turn |
| `tests/qwen3_tts_catalog_test.cpp` | modify | The codec-encoder resolver keeps every pointer it resolves |
| `tests/qwen3_tts_profile_test.cpp` | modify | ICL payload rules, the cross-kind tamper matrix, Plan-2-profile compatibility |
| `tests/qwen3_tts_voice_required_test.cpp` | modify | `OPTIONAL` ×2, rewritten not deleted |
| `tests/qwen3_tts_base_load_real.cpp` | modify | The two refusal checks become mode-selection checks |
| `tests/qwen3_tts_icl_real.cpp` | create | Integration: reference audio + transcript in, cloned audio out |
| `tests/CMakeLists.txt` | modify | Register the new tests |
| `tests/tolerances/qwen3-tts.json` | modify | Real codec/prompt probes in `replay`; retire `provisional_variants` entirely |
| `tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-base.manifest.json` | modify | Materialized ICL cases; the recorded per-case code agreement rate. **No `alternate_grids`** — the spec's fourth erratum rejects that mechanism here |
| `scripts/validate-qwen3-tts-replay.py` | modify | New comparison modes — deliberately NOT a new validator basename |
| `docs/porting/families/qwen3-tts.md` | modify | Record Stage 2 Plan 3 |
| `docs/voice-conditioning.md` | modify | The D5 recoverability declaration |

---

### Task 1: A stage-wise oracle for the codec encoder

**Files:**
- Create: `scripts/dump_reference_qwen3_tts_codec_encoder.py`

**Interfaces:**
- Consumes: `models/qwen3-tts-12hz-0-6b-base/` (including `speech_tokenizer/`), the pinned reference clip in `models/qwen3-tts-reference-audio/`.
- Produces: under `build/goldens/qwen3-tts/qwen3-tts-12hz-0-6b-base/<case-id>/codec_encoder/` — `waveform.f32`, `seanet_stage0..3.f32`, `seanet_tail.f32`, `downsample.f32`, `transformer_l0..l7.f32`, `latents.f32`, `rvq_residual_s00..s15.f32`, `rvq_reconstruction.f32`, `rvq_distance_margin.f32`, `codes.i32`, plus `conventions.json` at `<case-id>/codec_encoder/conventions.json`. Tasks 3, 4, 5 and 12 compare against these. **`rvq_reconstruction.f32` is half of the gate the spec's fourth erratum installs and is not optional**: the sum of the sixteen selected codebook rows in the 256-wide projected space, `[codebook_dim/2, frames]`, i.e. the continuous quantity the discrete indices were rounded from. **This task owns the filename `conventions.json`**; Task 2's parallel file is deliberately named differently so the two can never collide.
- Does **not** produce: `T1`, `T2`, the alignment branch, or the trailing schedule. Those are properties of the two-track prompt, they depend on text this script never reads, and Task 2 owns them.

**Why this is first, and why it is the largest oracle in the family so far.** The existing Base oracle (`scripts/dump_reference_qwen3_tts_base.py`) already runs ICL — `--ref-text` at `:101`, `--x-vector-only` at `:105`, `codes/reference.i32` asserted `[frames, 16]` at `:420-426`, `prompt/icl_embed.f32` captured by monkey-patching `generate_icl_prompt` at `:348-368`. **What it has is the finished output of a subsystem with eleven internal stages and nothing in between.** The design's §6 asks for a "codec encoder" dump script and it does not exist. Without it, a wrong SEANet stride and a wrong transformer scale can cancel into codes that are *nearly* right and then get rounded to *exactly* right by the quantizer — the same discreteness that hides an error until the clip changes. **Since the spec's fourth erratum, this script is not merely the sharper instrument, it is the only one**: the codes are no longer a gate at all, so every stage this script taps is load-bearing and a stage it omits is a stage nothing checks.

- [ ] **Step 1: Read the conventions off `transformers`, not off this plan**

The encoder is `MimiModel` with the decoder halves nulled. Open the pinned `transformers==4.57.3` wheel the locked `scripts/envs/qwen3-tts` environment resolves and record, **verbatim with its file and line number**, into `conventions.json`:

1. `MimiConv1d`'s padding mode — causal vs symmetric, and how the padding amount is derived from kernel, stride and dilation (`modeling_mimi.py:124`); the checkpoint declares `use_causal_conv True`, `pad_mode constant`.
2. The activation and where it sits relative to each convolution (`nn.ELU()` at `:302, 334, 338` — ELU, not GELU; the checkpoint's `hidden_act gelu` belongs to the *transformer*, not the SEANet stack).
3. The residual unit's dilations and how `dilation_growth_rate 2` / `num_residual_layers 1` expand into per-stage values.
4. Whether the residual unit's shortcut is a convolution (`use_conv_shortcut False`) and whether the residual is added before or after the stride.
5. `MimiSplitResidualVectorQuantizer.encode` (`:1357`) — the order of semantic and acoustic stages in the concatenated output, whether the residual is updated with the *quantized* or the *dequantized-and-projected* vector, and in which space the distance is computed.
6. Whether the RVQ codebooks are L2-normalized or compared by plain squared Euclidean distance, and whether any codebook re-initialization/EMA path is live at inference (it is not, but the reason must be recorded — the converter collapsed 32 EMA pairs and dropped 32 `.initialized` flags precisely because they are training state).
7. `trim_right_ratio 1.0` and where it applies.
8. `Qwen3TTSTokenizerV2Model.encode`'s own two post-steps (`:961-988`): the slice `audio_codes[:, :encoder_valid_num_quantizers]` (16 of 32) and the frame trim `-(-mask.sum() // self.encode_downsample_rate)`.

A convention that is not in `conventions.json` was guessed. This file is what Tasks 3 and 4 implement against.

- [ ] **Step 2: Record the geometry the port will hard-code, and reconcile it against the package**

The package publishes **no** `codec.encoder.*` metadata namespace at all (research §1e); the catalog's encoder widths are compiled-in literals at `catalog.cpp:429-434`. Read `models/qwen3-tts-12hz-0-6b-base/speech_tokenizer/config.json` and write the checkpoint values below into `conventions.json` beside the upstream lines.

**Two of the checkpoint values are NOT inside `encoder_config`, and reading them there picks up a decoy.** `encoder_valid_num_quantizers` (16) and `encode_downsample_rate` (1920) are **top-level** keys of `speech_tokenizer/config.json`. Inside `encoder_config` sit `num_quantizers` = **32** and `codebook_dim` = **256** — an implementer who reads "quantizers" and "downsample" out of `encoder_config` gets 32 quantizers and no downsample rate at all, and 32 is exactly the wrong answer the 16-of-32 slice exists to prevent. The `Checkpoint key` column below states where each value actually lives; the table is the reconciliation that must hold, checked here rather than discovered in Task 3:

| Quantity | Checkpoint key | What the port derives | Source of the port's value |
| --- | --- | --- | --- |
| transformer hidden | `encoder_config.hidden_size` 512 | `codec.decoder.codebook_dim` = **512** | GGUF metadata `synthesize.qwen3-tts.codec.decoder.codebook_dim` |
| transformer MLP | `encoder_config.intermediate_size` 2048 | `4 * hidden` = 2048 | `catalog.cpp:555` |
| transformer layers | `encoder_config.num_hidden_layers` 8 | `kCodecEncoderTransformerLayerCount` = 8 | `catalog.cpp:433` |
| heads / head_dim | `encoder_config.num_attention_heads` 8 / `head_dim` 64 | 8 × 64 = 512 = hidden | `encoder_config` |
| RVQ projected width | `encoder_config.codebook_dim` 256, `encoder_config.vector_quantization_hidden_dimension` 256 | `codebook_dim / 2` = **256**, from the *decoder's* 512 | `catalog.cpp:409` |
| codebook | `encoder_config.codebook_size` 2048 | `{codebook_dim/2, codebook_size}` = `{256, 2048}` | `catalog.cpp:413` |
| stage widths / kernels | `encoder_config.num_filters` 64, `encoder_config.compress` 2 | `{128, 256, 512, 1024}` / `{8, 10, 12, 16}` | `catalog.cpp:430-431` |
| frame downsampler | — | kernel 4, 512→512, **no bias** | `catalog.cpp:432, 546-547` |
| samples per frame | **top-level** `encode_downsample_rate` 1920 | `synthesize.qwen3-tts.codec.hop_length` = 1920 | GGUF metadata, written at `scripts/convert-qwen3-tts.py:643` and asserted against `SAMPLES_PER_FRAME` (`:131-132`) at `:636-638` |
| quantizers read | **top-level** `encoder_valid_num_quantizers` 16 (**not** `encoder_config.num_quantizers` 32) | `quantizer_count` 16 = 1 semantic + 15 acoustic | GGUF metadata |

Two things this table settles and one it does not. The encoder RVQ carries **31** acoustic stages (`kCodecEncoderAcousticQuantizerCount` at `catalog.cpp:434`) of which only 15 are ever read, so sixteen catalogued codebooks are resolved and never evaluated — record that as intended, not as a bug. The SEANet strides 4·5·6·8 = 960 times the downsampler's stride 2 gives 1920 samples per frame, which the artifacts already confirm (193,920 → 101 frames; 24,000 → 13; 720,000 → 375; the trim is a **ceiling** divide — 24,000/1920 = 12.5 → 13). What it does **not** settle is the stride *order*. The checkpoint states them under the decoder-oriented name `encoder_config.upsampling_ratios`, as `[8, 6, 5, 4]`, and this plan's own prose writes the kernels the other way round as `{8, 10, 12, 16}` against widths `{128, 256, 512, 1024}` — so the order the encoder actually walks is the one thing here that must be read off `modeling_mimi.py` rather than off either list. Record it with its line number. A reversed stride order produces the same 960× total and a different everything else, and the frame-count test in Task 4 cannot see it.

- [ ] **Step 3: Write the runner**

Model it on `scripts/dump_reference_qwen3_tts_speaker.py` (Plan 2's, the closest analogue) — same `--weights-dir` / `--ref-audio` / `--manifest` / `--case` / `--output-root` argument shape, the same raw-`tobytes()` `write_f32` with no `.npy` header, the same atomic writes. Tap the encoder's submodules with `register_forward_hook`, exactly as Plan 2 did; a hook is a read, not a change.

The RVQ needs more than a hook. For each of the 16 read stages, dump the residual *entering* the stage and, per frame, the two smallest squared distances over the 2048 codebook entries. Then dump the **reconstruction**: the sum of the sixteen selected codebook rows, in the 256-wide projected space the distances are computed in, as `rvq_reconstruction.f32` shaped `[codebook_dim/2, frames]`.

```python
# Two artifacts, two jobs. The reconstruction is HALF THE GATE (spec's fourth
# erratum, 2026-08-13): it is the continuous quantity the indices were rounded
# from, so a near-tie flip moves it by about the tie margin while a wrong
# stride, a wrong padding mode or a chained acoustic branch moves it by orders
# of magnitude more. The margin is now DIAGNOSTIC, not a gate selector: the
# question of whether flips occur is settled -- they do, at 0.624% per decision
# under a bf16-vs-f32 codebook alone -- so this records how close this case's
# decisions run, and nothing branches on it.
best_two = np.partition(distances, 1, axis=-1)[..., :2]
margin = best_two[..., 1] - best_two[..., 0]
```

Write `rvq_distance_margin.f32` as `[stages, frames]` and record its global minimum in `conventions.json` as `min_distance_margin`.

**Record the oracle's dtype in `conventions.json`, as its own key, and make it a fact rather than an assumption.** Read back `mimi.config.dtype` and the dtype of the derived `embed` table after loading, and write both. This is not bookkeeping: the erratum's whole finding is that the table's dtype decides the codes, that the checkpoint's own `"dtype": "float32"` declaration does not get a vote, and that a load omitting `dtype=torch.bfloat16` disagrees with the committed reference on 49% of `base-icl-en`'s codes. Every tolerance measured downstream is bf16-derived *because* this script loaded bf16; a script that silently loaded f32 would have moved the whole grid.

**No seed is needed.** The codec encoder is deterministic (an argmin over fixed codebooks) — the same note Plan 2's speaker script carries. The seeded-sampling rule (`spec:338-358`) applies only to anything that runs the talker, and this script does not.

- [ ] **Step 4: Run it on the three ICL cases that already have artifacts**

```bash
for case in base-icl-en base-ref-min base-text-short; do
  uv run --project scripts/envs/qwen3-tts --locked python \
    scripts/dump_reference_qwen3_tts_codec_encoder.py \
    --weights-dir models/qwen3-tts-12hz-0-6b-base \
    --manifest tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-base.manifest.json \
    --case "$case"
done
```

Expected, from the reference-code artifacts Plan 1 already left on disk: `base-icl-en` and `base-text-short` produce **101** frames (`codes/reference.i32` at 6,464 bytes = 101 × 16 × 4), `base-ref-min` produces **13** (832 bytes). The newly dumped `codes.i32` must be **byte-identical** to the `codes/reference.i32` already there. If it is not, the hooks changed the forward or the clip is being loaded differently — stop, because every tolerance measured downstream would inherit the difference.

**Read that check for exactly what it is, and no further.** Both scripts load `dtype=torch.bfloat16`, so this is an oracle-against-oracle comparison of two bf16 runs: it tests that the hooks did not perturb the forward and that the clip loads identically. **It is not evidence that the port's codes will match, and Plan 1's identical observation was misread as though it were** — that is the misreading the spec's fourth erratum documents. Nothing about this step licenses a code-equality gate anywhere downstream.

- [ ] **Step 5: Commit**

```bash
git add scripts/dump_reference_qwen3_tts_codec_encoder.py
git commit -m "qwen3-tts: dump every codec-encoder stage, its RVQ reconstruction and its tie margins, not just the codes"
```

Nothing under `build/` is added.

---

### Task 2: A stage-wise oracle for the two-track ICL prompt

**Files:**
- Create: `scripts/dump_reference_qwen3_tts_icl_prompt.py`

**Interfaces:**
- Consumes: the same weights and manifest.
- Produces: under `<case-id>/prompt/` beside the existing `icl_embed.f32` — `text_track.f32`, `codec_track.f32`, `trailing.f32`, `ref_text_ids.i32`, `target_text_ids.i32`, `prompt_conventions.json`, and an `alignment.json` carrying `T1`, `T2`, `branch` (`"truncate"` when `T1 > T2`, `"pad"` otherwise), `ref_frames`, and `trailing_positions`. Tasks 6, 7 and 12 compare against these.
- **The transcription file is `prompt_conventions.json`, not `conventions.json`.** Task 1 owns the name `conventions.json` and writes it under `<case-id>/codec_encoder/`; two scripts emitting a file of the same name is an overwrite waiting for the day someone flattens the directories. Task 14 Step 2 transcribes both, by their distinct names.

**This task, and not Task 1, owns the alignment dump.** `T1`, `T2`, the branch and the trailing schedule are decided by `generate_icl_prompt` from the reference *and target* text token counts against the reference frame count. Task 1's codec-encoder oracle produces the frame count and never reads a character of text, so it cannot compute `T1` and cannot know the branch. The plan's validation strategy leans on the branch being an artifact; the artifact is this script's `alignment.json`, and Step 3 below is where it is written.

**Why this is a second script and not a flag on the first.** `prompt/icl_embed.f32` is the *sum* of two tracks. A text track that is one position early and a codec track that is one position late produce a different sum, but so do a hundred other pairings, and the sum alone tells you only that something moved. The design's §6 lists "clone prompt layout" as one of its four dump scripts; this is that one. Adding it as a `dump_*` script does not touch the validator basename set that `tests/python/test_tolerance_coverage.py:129-148` globs — that test reads `scripts/validate-qwen3-tts-*.py` only.

- [ ] **Step 1: Transcribe the construction verbatim, with line numbers**

From `qwen_tts/core/models/modeling_qwen3_tts.py:1968-2019` (`generate_icl_prompt`), called at `:2189-2196`. Into `<case-id>/prompt/prompt_conventions.json`:

- **Text track** (`:1978-1981`): `text_projection(text_embeddings(cat([ref_id, text_id])))`, then `cat([·, tts_eos_embed])`. So `T1 = len(ref_id) + len(text_id) + 1`.
- **Codec track** (`:1983-1998`): per reference frame, the **sum of 16 embeddings** — group 0 from `talker.get_input_embeddings()`, groups 1..15 from `talker.code_predictor.get_input_embeddings()[i-1]` — with one `codec_bos_id` row prepended through the talker's own codec embedding. So `T2 = 1 + ref_frames`.
- **Alignment** (`:2015-2019`, `non_streaming_mode=False`, which is what `generate_voice_clone` defaults to — `qwen_tts/inference/qwen3_tts_model.py:478`): if `T1 > T2`, block = `text_embed[:, :T2] + codec_embed` and `text_embed[:, T2:]` becomes the trailing schedule; otherwise the text is padded with `tts_pad_embed` up to `T2` and the trailing schedule is a bare `tts_pad_embed`.
- **Placement** (`:2197`): the block is **concatenated after** the existing prefix, and it **replaces** the `tts_text_first_token` position the non-ICL branch builds (`:2199-2202`).
- **The x-vector is still there** (`:2166-2172`): `speaker_embed` is inserted into `codec_input_emebdding` regardless of mode. **ICL adds to the x-vector path; it does not substitute for it.** This matches D5's table (`spec:123-128`), where the speaker embedding is `yes` in both columns.
- **The reference transcript uses a different turn wrapper** (`qwen3_tts_model.py:271-272`): `f"<|im_start|>assistant\n{text}<|im_end|>\n"`, tokenized whole at `:598` and sliced `[3:-2]` at `modeling_qwen3_tts.py:2191`, against the target text's `[3:-5]` from `_build_assistant_text` (`:268-269`).

- [ ] **Step 2: Extend the existing capture rather than re-deriving it**

`dump_reference_qwen3_tts_base.py:348-368` already monkey-patches `generate_icl_prompt` because it is a plain method with nothing to hook. Reuse that mechanism: capture the two intermediate tensors and the branch inside the patched call, and write them plus `alignment.json`. Keep the existing assertion at `:453-457` that x-vector cases produce **no** ICL embed.

- [ ] **Step 3: Run it on the three materialized cases and report the branch**

```bash
for case in base-icl-en base-ref-min base-text-short; do
  uv run --project scripts/envs/qwen3-tts --locked python \
    scripts/dump_reference_qwen3_tts_icl_prompt.py \
    --weights-dir models/qwen3-tts-12hz-0-6b-base \
    --manifest tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-base.manifest.json \
    --case "$case"
done
```

Expected shapes, from the artifacts already on disk: `icl_embed.f32` is 417,792 bytes = 102 positions × 1024 for `base-icl-en` and `base-text-short` (`T2 = 1 + 101`), and 57,344 bytes = 14 positions × 1024 for `base-ref-min` (`T2 = 1 + 13`). `text_track.f32` and `codec_track.f32` must sum, elementwise, to `icl_embed.f32` — assert that in the script, because if they do not, the capture is not capturing what it claims.

**Then measure the branch for each case. This plan states no expectation, and neither should the implementer before the run.** The block is `T2` positions long in both arms, so its size proves nothing, and no other artifact on disk records the discriminator. Write the observed `T1`, `T2` and branch of each case into `alignment.json` from inside the patched call — read out of the arm that actually executed, not recomputed afterwards from a length comparison, since recomputing it would re-derive the plan's guess instead of observing upstream's choice.

One candidate is worth eyeballing afterwards as a sanity check on the capture rather than as a prediction of it: `base-ref-min` has `T2 = 14` against a reference transcript of "Okay. Yeah. I resent you. I love you. I…", which is visibly more than fourteen tokens, so a `"pad"` report for that case would mean either the capture is reading the wrong variable or `T1` is not what §Step 1 transcribed — investigate before continuing, in that order. A `"truncate"` report is not a confirmation of anything; it is just the absence of that particular alarm.

If all three cases report the same branch, Task 12 must construct a case for the other one. `alignment.json` — not this plan, and not the research — is what Task 7's tests and Task 12's arm coverage are written from.

- [ ] **Step 4: Commit**

```bash
git add scripts/dump_reference_qwen3_tts_icl_prompt.py scripts/dump_reference_qwen3_tts_base.py
git commit -m "qwen3-tts: dump the ICL prompt's two tracks, its alignment branch, and its trailing schedule"
```

The commit message states which branch each of the three cases took.

---

### Task 3: The codec-encoder resolver keeps its pointers

**Files:**
- Modify: `src/arch/qwen3-tts/catalog.h`, `src/arch/qwen3-tts/catalog.cpp`
- Modify: `tests/qwen3_tts_catalog_test.cpp`

**Interfaces:**
- Produces: `CodecEncoderWeights`, hung off `ModelWeights`. Task 4's graph reads it.
- `build_model_weights`'s signature does not change — `ModelWeights &` is already the sink.

**The trap, named in the source itself.** `catalog.cpp:510-515`:

> *"Still Plan 1's discard-into-scratch shape: every resolved pointer below is written into a local that nothing reads, on purpose, because no graph reaches this encoder yet. Do not read the sibling resolve_speaker_encoder above as evidence that this one was converted too — Plan 3 owns that, and its signature will need to change the same way this task changed resolve_speaker_encoder's."*

There is no `CodecEncoderWeights` type and `ModelWeights` (`catalog.h:160-166`) carries `SpeakerEncoderWeights speaker_encoder;` only. **Copying `resolve_codec`'s pattern onto these while leaving `Conv1dWeights scratch;` at `catalog.cpp:519` in place still compiles and still passes every catalog test**, because the sweep checks only that a name was *resolved*, never that a pointer was *kept*. Change the signature; do not add to it. Delete the scratch rather than leaving it unused — an unused local is what the next plan copies.

- [ ] **Step 1: Write the failing test**

In `tests/qwen3_tts_catalog_test.cpp`, on the shape Plan 2 used for the speaker encoder — `check_speaker_encoder_resolver_keeps_every_pointer()` at `:696`, wired into `main` at `:776`, and note the `check_` prefix this file uses throughout rather than `test_`. Assert every pointer **by name**; a loop that checks "at least one is non-null" passes with 160 dropped. Assert also that the 31 acoustic codebooks are 31 *distinct* tensors — a resolver that wrote them all into one slot would leave 30 aliases that the null checks cannot see. And assert that a CustomVoice package leaves the struct empty.

- [ ] **Step 2: Run it and watch it fail**

```bash
cmake --build build --target synthesize-qwen3-tts-catalog-test
```
Expected: FAIL — `'struct synth::qwen3tts::ModelWeights' has no member named 'codec_encoder'`.

- [ ] **Step 3: Add the types and convert the resolver**

In `catalog.h`, beside `SpeakerEncoderWeights`. The shape follows the resolver's own regions (`catalog.cpp:516-580`): a stem conv (kernel 7, 1→64), four downsampling stages each carrying a two-conv residual bottleneck (kernels 3 then 1, width→width/2→width) and a strided conv (kernels `{8,10,12,16}`, widths `{128,256,512,1024}`), a tail conv (`layers.14`, kernel 3, 1024→`codebook_dim`), the frame downsampler (`downsample.conv.weight`, kernel 4, hidden→hidden, **no bias**), eight transformer layers of 12 tensors each (two LayerNorms with weight and bias, q/k/v/o projections, `self_attn_scale.scale`, `mlp.fc1`/`fc2`, `mlp_scale.scale`), and two quantizer cascades (`semantic_rvq` with 1 stage, `acoustic_rvq` with 31), each with `input_proj.weight` / `output_proj.weight` and one `codebook` per stage.

Then change `resolve_codec_encoder`'s signature to take a `CodecEncoderWeights &` target, delete `Conv1dWeights scratch;` and `LayerNormWeights norm_scratch;`, and update the single call site — `!resolve_codec_encoder(resolver, hparams))` at **`catalog.cpp:659`**, inside `build_model_weights`'s `has_speaker_encoder` block at `:657-661`. (`:641-644` is the `resolve_talker` / `resolve_code_predictor` block and is not this call.) Rewrite the header comment at `:510-515` — it is now false, and leaving it would tell the next reader that Plan 3 has not run — and the block comment at `:651-656`, which still says the codec encoder "still has no graph (Plan 3 adds one)" and is resolved "only to bring its names into the sweep".

- [ ] **Step 4: Confirm `expected_tensor_count` is unchanged in value**

Its codec-encoder arithmetic (`catalog.cpp:621-625`) reuses the same constants and must still produce 894 for the Base hparams, and the existing assertion `base - custom_voice == 76 + 161` must still hold. Refactoring a resolver must not move a count; if it does, a name was dropped or added.

- [ ] **Step 5: Update the catalog header's stale claim**

`catalog.h:175-180` still states the codec encoder "has no graph". Plan 2 corrected the speaker-encoder half of the same sentence; this task corrects the other half.

- [ ] **Step 6: Prove the tests are load-bearing, then run the gates**

Delete one `resolver.conv(...)` target assignment — write it back into a local — and confirm the by-name assertion fails. Restore it. Then:

```bash
ctest --test-dir build -R '^synthesize-qwen3-tts-catalog-test$' --output-on-failure
cmake --build build --target synthesize-check-unit
cmake --build build-sanitize --target synthesize-check-unit
```

- [ ] **Step 7: Commit**

```bash
scripts/ci/clang-format.sh --fix
git add src/arch/qwen3-tts/catalog.h src/arch/qwen3-tts/catalog.cpp tests/qwen3_tts_catalog_test.cpp
scripts/ci/clang-format.sh --check-diff origin/main
git commit -m "qwen3-tts: the codec encoder resolver keeps the pointers it resolves"
```

---

### Task 4: The codec encoder graph

**Files:**
- Create: `src/arch/qwen3-tts/codec-encoder.h`, `src/arch/qwen3-tts/codec-encoder.cpp`
- Create: `tests/qwen3_tts_codec_encoder_test.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `CodecEncoderWeights` from Task 3, an F32 waveform tensor `[samples]`.
- Produces: `build_codec_encoder(...) -> ggml_tensor *` of shape `[codebook_dim, frames]` — the pre-quantization latents. Task 5's host wrapper runs it and quantizes.

**This is the largest single piece in the plan and it lands in two commits inside one task**, because the two halves are measured against different oracle artifacts and a combined commit would make a bisect useless. That is a deliberate deviation from the one-commit-per-task rhythm Plan 2 used, and it is the only one.

- [ ] **Step 1: Write the failing test**

`tests/qwen3_tts_codec_encoder_test.cpp`, on the fixture pattern of `tests/qwen3_tts_codec_test.cpp` — `make_context` with `no_alloc = true`, a `Fixture` owning backend/context/buffer/weights, a node budget, and a backend sweep in `main`. No GGUF; weights come from a single 64-bit LCG so both sides draw in the same order. The rules, each with an input that fails it:

1. **The frame count is `ceil(samples / 1920)`**, at the production geometry. Assert three points that the artifacts pin exactly: 193,920 → 101, 24,000 → 13, 720,000 → 375. A floor divide gives 12 for the second and is the single most likely off-by-one in this task.
2. **A different stride set produces a different frame count** — otherwise the strides are being ignored and 1920 is right by coincidence.
3. **The latents are `[codebook_dim, frames]` = `[512, frames]`** and every element is finite.
4. **A clip shorter than one frame is refused**, not silently padded to one.
5. **A non-finite sample is refused.**
6. **The causal padding is causal**: appending samples to the end of a clip must not change the latents of the frames that precede them. This is the assertion that catches a symmetric `MimiConv1d` pad, and nothing downstream can.

- [ ] **Step 2: Build the SEANet stack and the frame downsampler, and measure it**

Stem conv (kernel 7, 1→64), then four stages of `residual → strided conv` at widths `{128, 256, 512, 1024}` with kernels `{8, 10, 12, 16}`, the tail conv (kernel 3, 1024→512), then `downsample.conv` (kernel 4, 512→512, no bias, stride 2). ELU activations, causal padding, dilations — all from Task 1's `conventions.json`, none from this plan.

Then compare against the oracle, stage by stage, with a driver in the shape of `tests/qwen3_tts_mel_driver.cpp`:

```
seanet_stage0.f32 … seanet_stage3.f32, seanet_tail.f32, downsample.f32
```

Every stage must agree before the next is written. **A stage that is skipped here is a stage whose error is invisible for the rest of the plan.**

Commit this half:

```bash
git commit -m "qwen3-tts: the codec encoder's SEANet stack and frame downsampler"
```

- [ ] **Step 3: Build the encoder transformer, and measure it**

Eight layers at hidden 512, MLP intermediate 2048, 8 heads × 64. Standard LayerNorm (weight **and** bias) — not the decoder's RMSNorm — and a plain two-layer MLP, not the decoder's gated one. The two per-branch scale vectors (`self_attn_scale.scale`, `mlp_scale.scale`, `layer_scale_initial_scale 0.01` in the checkpoint) multiply their branch outputs before the residual add; confirm the placement from `modeling_mimi.py` rather than from the name. `sliding_window 250`, `rope_theta 10000.0`, `norm_eps 1e-05`, `attention_bias False` — the sliding window is a real constraint at 375 frames and is unexercised at 101, so a test must drive a length above 250.

Compare per layer against `transformer_l0..l7.f32`, then the final `latents.f32`.

- [ ] **Step 4: Prove the tests are load-bearing**

Delete the causal-padding offset and confirm test 6 fails. Change the ceiling divide to a floor and confirm test 1 fails. Restore both.

- [ ] **Step 5: Gates and commit**

```bash
cmake --build build --target synthesize-check-unit
cmake --build build-sanitize --target synthesize-check-unit
scripts/ci/clang-format.sh --fix && git add … && scripts/ci/clang-format.sh --check-diff origin/main
git commit -m "qwen3-tts: the codec encoder's transformer, measured per layer against the oracle"
```

---

### Task 5: RVQ encode and the host wrapper

**Files:**
- Create: `src/arch/qwen3-tts/codec-encoder-host.h`, `src/arch/qwen3-tts/codec-encoder-host.cpp`
- Modify: `src/arch/qwen3-tts/qwen3-tts.h`, `src/arch/qwen3-tts/model.cpp`
- Modify: `tests/qwen3_tts_codec_encoder_test.cpp`

**Interfaces:**
- Consumes: Task 4's latents, `CodecEncoderWeights`'s quantizer cascades.
- Produces: `CodecEncoding` and

  ```cpp
  synth_status_t encode_codec_reference(const HParams &, const CodecEncoderWeights &,
                                        const std::vector<float> & pcm, int threads,
                                        CodecEncoding & output,
                                        const char *& out_diagnostic_code,
                                        const char *& out_diagnostic_message);
  ```

  mirroring `encode_speaker_reference` (`src/arch/qwen3-tts/speaker-encoder-host.h:30-36`) field for field, and `Model::prepare_codec_reference` mirroring `Model::prepare_x_vector` (`qwen3-tts.h:199-203`). Add `Model::codec_encoder_weights()` beside `Model::speaker_encoder_weights()` (`qwen3-tts.h:215-216`, `model.cpp:393-395` — `:389-391` is `Model::hparams()`) so preparation is unit-testable without the 2.5 GB package.

**`create_icl_profile`'s signature is pinned here, three tasks before Task 8 writes it, because Task 6 has to know where reference-text ids come from.** `synth::qwen3tts::Model` can only be constructed through `Model::load`/`load_cpu`, both of which need a real GGUF on disk, and this family has no synthetic-package harness (`profile.h:74-79`) — which is why `create_x_vector_profile` takes `(const HParams &, const SpeakerEncoderWeights &, …)` and not a `Model &`. Its ICL sibling takes the same shape plus the two new inputs:

  ```cpp
  synth_status_t create_icl_profile(const HParams &                     hparams,
                                    const SpeakerEncoderWeights &       speaker_encoder,
                                    const CodecEncoderWeights &         codec_encoder,
                                    const std::vector<float> &          pcm_24k,
                                    const std::string &                 transcript,
                                    const std::vector<int32_t> &        reference_text_ids,
                                    const std::string &                 language_tag,
                                    int                                 threads,
                                    std::shared_ptr<const IclProfile> & output,
                                    const char *&                       out_diagnostic_code,
                                    const char *&                       out_diagnostic_message);
  ```

  **`reference_text_ids` arrives already tokenized, and that is the whole point.** The BPE tables live on `Model`; tokenizing inside this function would drag a `Model &` into it and turn every test of it into an integration test, against this plan's own Global Constraint on the `unit` label. Task 6's `Model::tokenize_reference_transcript` therefore runs at the dispatch site in `src/voice-profile.cpp`, which already holds a live `Model &` (Task 10 Step 3), and passes the ids down. `transcript` is still passed, for two reasons that need no tables: this function makes the empty/whitespace-only judgement itself, exactly where `create_x_vector_profile` makes it today; and it refuses the half-present state — a non-empty `transcript` with empty `reference_text_ids`, or the reverse — rather than trusting its caller, which is the same all-or-nothing rule Task 11 Step 1 enforces at the synthesis seam.

**The placement rule.** The RVQ assignment is a rounded, discrete output, and this project's standing rule puts those on the host — `omnivoice::rvq_encode` (`src/arch/omnivoice/reference-encoder-host.h:296-366`) states the reasoning at `:296-303` and is the template for everything here: the tie rule (ties resolve to the lowest id, strict `>` ascending, `:312-319`), the plain-float32 accumulation decision (`:323-330`), and the `narrowest_gap` / `out_gaps` margin instrumentation (`:340-349`) — which here is **diagnostic**, since the spec's fourth erratum settled the tie question with a measurement rather than leaving it for this task to decide. **One thing that template does not carry, and this task must add: the dequantized reconstruction.** The host wrapper emits the sum of the sixteen selected codebook rows in the 256-wide projected space alongside the codes, because that vector — not the indices — is what Step 2 compares against the oracle. It is a by-product of the cascade the argmin already computes, not a second pass. Qwen3-TTS's cascade differs — a split quantizer, a `codebook_dim/2` = 256-wide projected space, 32 stages of which 16 are read — but the shape of the solution already exists in-tree.

**The code-grid layout, settled here so no later task has to choose.** `CodecEncoding` stores the reference codes in **GGML index order `[16, T]`**: `ne[0] = 16` is the quantizer-group index and is the *fastest-varying* dimension, `ne[1] = T` is the frame index. Every structure downstream — `CodecEncoding`, the ICL payload, `TalkerInputPosition`'s widened code group, Task 7's per-position sum of sixteen embeddings — uses that and only that, which is why the design's D5 table and Tasks 7 and 8 already write `[16, T]`.

**There is no transpose anywhere, and the apparent conflict with the oracle is a notation clash rather than a layout clash.** The oracle asserts its array is numpy-shaped `[frames, 16]` (`dump_reference_qwen3_tts_base.py:420-426`) and writes it with `np.ascontiguousarray(...).tobytes()` (`:152-156`), i.e. C-order: sixteen codes contiguous per frame, frame index slowest. GGML `[16, T]` with `ne[0] = 16` describes *that identical byte sequence*; numpy names its slowest axis first and GGML names its fastest axis first. So `codes.i32` and `CodecEncoding`'s buffer compare as a flat `memcmp` of `16 * T` int32s, and any code that inserts a transpose to reconcile the two spellings has just broken the comparison. State this mapping in `codec-encoder-host.h` in exactly these terms, because the next reader will meet the same two spellings and reach for the same wrong fix.

The failure this actually guards against is not a spelling: it is **stage-major** storage — all `T` codes of quantizer stage 0, then all of stage 1 — which is what a reader who takes `[16, T]` for a numpy shape would build. That buffer has the right element count, the right value range, and the wrong order, and on the 16-frame case it even has the right *shape*.

- [ ] **Step 1: Write the failing test**

1. **Sixteen codes per frame, and the count is `quantizer_count` = 1 semantic + 15 acoustic**, not 32.
2. **Every code is in `[0, codebook_size)` = `[0, 2048)`.** A range check, not a spot check.
3. **The semantic stage leads.** Swapping semantic and acoustic order produces sixteen valid-looking codes and a different voice; assert the semantic stage's code equals a directly-computed argmin over `semantic_rvq`'s codebook.
4. **Ties resolve to the lowest id.** Build a synthetic codebook with two exactly-equal entries and assert the lower id wins.
5. **The layout is group-fastest, and a stage-major grid is not accepted as equal.** The settlement is above and is not this test's to re-decide: `CodecEncoding` is GGML `[16, T]`, `ne[0] = 16`, byte-identical to the oracle's C-order `[frames, 16]`. What this test pins is that the port *builds* it that way — assert that the flat buffer's first sixteen int32s are the sixteen quantizer codes of frame 0, not the frame-0 codes of sixteen frames at stage 0, and that a stage-major grid of the same dimensions compares unequal.

   **The comparison must be a production entry point, not a test-local helper**, or the assertion pins a property of the test. Put it on `CodecEncoding` — a `codes_equal(const CodecEncoding &, const int32_t * flat, int64_t frames)` (or an `operator==`) declared in `codec-encoder-host.h` — and drive that same function from this test, Task 9's serialize/load round trip, and Task 13's integration check. One comparison, three callers, and inverting it fails all three.

   **This function survives the fourth erratum and its callers change.** It is no longer the oracle gate — Step 2 is — but every remaining caller compares the port against *itself*, where exactness is free and mandatory: the layout assertion above, a serialize→load round trip that must return the identical grid, and Task 13's re-preparation identity. Do not repurpose it into an oracle comparison, and do not delete it because one caller went away.

   Build the stage-major counterexample at **16 frames**, since a 16×16 grid is the one size where the dimensions match under either reading and a shape check alone passes.
6. **The frame trim is the ceiling divide** and is applied after the graph, per `Qwen3TTSTokenizerV2Model.encode:961-988`.
7. **A silent reference is refused by name**, in the shape `encode_speaker_reference` already uses (`voice_profile.reference_silent`).

- [ ] **Step 2: Implement, then measure against the oracle — this step *is* the gate**

The design's §6 sent this step to compare `codes.i32` for exact equality. **Its fourth erratum (2026-08-13) drops that**, on jiangzhuo's ruling, because the oracle's codebook is bf16 and the converter's is f32: the two tables disagree on 4.04% of codes for the argmin alone and 12.73% through the full RVQ, 136 of 146 clips differ, and an upstream load that merely omits `dtype=torch.bfloat16` disagrees with the committed `codes/reference.i32` on 794 of its 1616 codes. Equality here would test a load-time keyword argument. **`base-icl-en` and `base-ref-min` are among the cases that differ — 59 of 1616 and 10 of 208 under the codebook change alone — so this is not a hypothetical this step might dodge.**

What this step compares, all of it, on all three materialized cases:

1. **Every continuous stage, in order**, against `waveform.f32`, `seanet_stage0..3.f32`, `seanet_tail.f32`, `downsample.f32`, `transformer_l0..l7.f32`, `latents.f32` and `rvq_residual_s00..s15.f32`. A stage that is not compared is a stage nothing checks — there is no longer a code-equality backstop behind it.
2. **The dequantized reconstruction** — the port's sum of the sixteen selected codebook rows in the 256-wide projected space — against `rvq_reconstruction.f32`, as F32. This is the discrete decision converted back into the continuous quantity it was rounded from: a near-tie flip moves it by about that decision's margin, a wrong stride or padding mode or a chained acoustic branch moves it by orders of magnitude more.
3. **The frame count and the code range**, exactly as before. Those are structural and stay strict.

**The tolerance is bf16-derived, and that cuts in both directions.** The oracle's activations and its codebook are bfloat16; the port's are F32, so the expected disagreement sits at the bf16 representation scale (relative 3.9e-3; the two codebooks' own mean relative difference is 2.28e-3 to 2.52e-3, max 1.06e-2) and *not* at F32 accumulation scale. Two obligations follow. A measured deviation at or below that scale is what this step is looking for, and Task 12 commits it under the tolerance file's existing widening rule. **A measured deviation above it is a defect to be found before this plan continues — never a gate to be widened to fit.** Record the observed per-stage maxima either way; they are what Task 12 has to work from.

**No claim about correctness is available from a small code difference, and none may be written.** The percentages above isolate the codebook; the port's F32 SEANet and F32 transformer feed the quantizer latents the bf16 oracle never produced, and their contribution is expected to be the larger one. Whether this port is right is what items 1 and 2 have to establish.

**A max-deviation gate at bf16 scale on the reconstruction is unsatisfiable by a correct port. Measured, not predicted — and it refutes the assumption the ruling was framed with.** When the ruling was chosen it was argued that a near-tie flip moves the reconstruction by roughly the tie margin while a real porting error moves it a lot. Task 1's addendum measured that and it is false for the reconstruction *vector*: a single flipped code displaces it by about one full codebook-row separation — median `|ΔRecon| / ‖eᵢ − eⱼ‖` of **1.21**, which is **12.9% of the norm at the median and 32.8% at the maximum** — and later stages do not absorb it. The "small displacement" claim holds only for the residual *norm* (2.4%), which is not what the gate compares. The consequence is a distribution, not a bound: deviation is bimodal, with a median at **3.2e-3** (the bf16 scale the paragraph above predicts) and a **p95 of 1.9e-1** (the flip tail). Task 12 therefore shapes this gate on a percentile or on a flip-aware comparison, and **may not** express it as a maximum at bf16 scale — that gate would fail a correct port on its tail. This is a change of shape, not a widening: the median must still sit at bf16 scale, and a median above it is the defect the paragraph above describes.

**One scalar tolerance cannot cover both branches.** Also measured in Task 1's addendum: the semantic branch has rms **13.5**, per-frame L2 median **205**, and explains only **43%** of its target — it is one stage — while the acoustic branch has rms **3.11**, L2 median **45.1**, and explains **85%**. An order of magnitude apart, so the bf16 scale lands at roughly **0.80** absolute L2 for the semantic branch and **0.18** for the acoustic. Task 12 sets them separately, from `conventions.json`, and does not average them into one key.

- [ ] **Step 3: Record the code agreement rate and the tie margin — and gate on neither**

Compare `codes.i32` anyway, per stage and per frame, and record: the agreement rate overall, the count and positions of the disagreements, and whether the semantic stage is among them. Read `min_distance_margin` from Task 1's `conventions.json` and record the port's own narrowest gap over the same cases, plus how many of the disagreements sit inside their frame's margin.

**These numbers are evidence, not a gate. Nothing branches on them, and no threshold is derived from them** — in particular not from the erratum's 0.624% and 4.04%, which bound the *codebook's* contribution alone and are therefore a floor on divergence, not a budget this port may spend. `oracle.alternate_grids` is **not** used: the erratum rejects it here on the ground that these flips are not a second configuration's enumerable output but ~4% of codes spread across 136 of 146 inputs, so every case would need its own committed grid generated by a table the converter does not bake.

They earn their place for two reasons. A rate far above the erratum's floor is a signal worth investigating even though it cannot fail the build by itself — and it is the honest disclosure the manifest and Task 12's commit message carry, so that no later reader mistakes "the gate passed" for "the codes match".

- [ ] **Step 4: Prove the tests are load-bearing, run the gates, commit**

Invert the tie rule to `>=` and confirm test 4 fails. Reverse the semantic/acoustic order and confirm test 3 fails.

**And prove the new gate is load-bearing, because it is now the only one.** With the codes no longer gating, a stage comparison that cannot fail leaves nothing behind it. Perturb the graph in one way the codes might have absorbed — reverse the SEANet stride order, or drop the causal padding to symmetric — and confirm Step 2's per-stage comparison **and** its reconstruction comparison both fail. Restore, and record the confirmation in the commit message under this plan's standing rule that a test that stays green is not a test.

```bash
git commit -m "qwen3-tts: reference waveform to [16, T] codes, gated stage-wise against the oracle"
```

The commit message records the observed per-stage deviations against the bf16-derived expectation, the reconstruction's deviation, and the code agreement rate as a disclosed observation that gated nothing.

---

### Task 6: The reference-transcript turn

**Files:**
- Modify: `src/arch/qwen3-tts/bpe.h`, `src/arch/qwen3-tts/bpe.cpp`
- Modify: `src/arch/qwen3-tts/model.cpp`, `src/arch/qwen3-tts/qwen3-tts.h`
- Modify: `tests/qwen3_tts_bpe_test.cpp`

**Interfaces:**
- Produces: `qwen_reference_turn(const std::string &)` and a `kReferenceSuffixTokens` constant beside the existing pair, plus a `Model::tokenize_reference_transcript` entry point returning `std::vector<int32_t>`.
- **Where its output goes, decided at Task 5 rather than here.** `Model::tokenize_reference_transcript` is called from the dispatch site in `src/voice-profile.cpp` (Task 10 Step 3), which holds a live `Model &`, and its ids are passed as `create_icl_profile`'s `reference_text_ids` parameter — the signature Task 5's Interfaces section pins. This task adds **no** accessor counterpart to `Model::codec_encoder_weights()`, because the BPE tables are never reached from a `unit` test: nothing below `create_icl_profile` tokenizes.

**This is not the existing wrapper, and tokenizing the bare transcript is not equivalent.** The port has exactly one turn wrapper — `qwen_assistant_turn` with `kAssistantRolePrefixTokens = 3` and `kAssistantSuffixTokens = 5` (`src/arch/qwen3-tts/bpe.h:16-26`), consumed at `model.cpp:820-823`. Upstream uses a **second, different** wrapper for the reference transcript (`qwen3_tts_model.py:271-272`):

```python
def _build_ref_text(self, text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n"
```

against the target text's `_build_assistant_text` (`:268-269`), which appends a further `<|im_start|>assistant\n`. It is tokenized whole (`:598`) and sliced `[3:-2]` at the call site (`modeling_qwen3_tts.py:2191`), against the target's `[3:-5]`. Tokenizing the bare transcript instead is not equivalent in general: the qwen pre-tokenizer's boundary behaviour after `assistant\n` is exactly what the wrap-then-slice idiom exists to preserve. `Model::tokenize_request` (`model.cpp:405-413`) applies the *assistant* wrapper unconditionally through the frontend, so the reference path needs its own entry point rather than a reuse.

- [ ] **Step 1: Write the failing test**

In `tests/qwen3_tts_bpe_test.cpp`:

1. **The two wrappers produce different strings** for the same input, and the reference wrapper ends at `<|im_end|>\n`.
2. **The slice constants differ**: 3 and **2** for the reference turn, 3 and 5 for the assistant turn.
3. **Round trip**: wrapping, tokenizing, and slicing `[3:-2]` recovers exactly the tokens the reference transcript contributes, and that token count is what Task 7's `T1` arithmetic uses.
4. **The boundary case that motivates the whole idiom**: wrapping changes the ids. Find a transcript for which `qwen_reference_turn` + tokenize + slice `[3:-2]` yields a **different** id sequence than tokenizing the bare transcript, and pin that transcript by value in the test. Start with a leading space, a leading non-ASCII character, and a leading newline, since the qwen pre-tokenizer's boundary behaviour after `assistant\n` is what the wrap-then-slice idiom exists to preserve.

   **There is no third option here.** Either the test names an input on which the two paths differ — and then deleting the wrapper makes it fail, which is what makes it a test — or, after the search, no such input exists, in which case **delete test 4 and write into the task's commit message and into `bpe.h` why the two paths cannot differ for this tokenizer**. A test whose assertion may hold vacuously, kept alive by a comment, is precisely the shape this plan's Global Constraints indict four times over; leaving one here would make the plan contradict itself. Note that deleting test 4 costs nothing that tests 1–3 do not already cover: the *claim* that the port uses the wrapper is pinned by test 1 (the two wrappers produce different strings), test 2 (the slice constants differ), and — decisively — by Step 2's exact comparison against the oracle's own `ref_text_ids.i32`.
5. **An empty or whitespace-only transcript is rejected**, per the design's §9 error table (`spec:419`) — `SYNTH_ERR_INVALID_ARG`.

- [ ] **Step 2: Implement, verify against the oracle's `ref_text_ids.i32`**

Task 2 dumps the reference ids upstream actually produced. The port's ids must match them exactly for all three materialized cases. This is a discrete comparison with no tolerance.

- [ ] **Step 3: Prove, gate, commit**

Point the reference path at `qwen_assistant_turn` instead and confirm the oracle comparison fails. Restore.

```bash
git commit -m "qwen3-tts: the reference transcript gets its own turn wrapper and slice"
```

---

### Task 7: The two-track ICL prompt

**Files:**
- Modify: `src/arch/qwen3-tts/talker-host.h`, `src/arch/qwen3-tts/talker-host.cpp`
- Modify: `src/arch/qwen3-tts/talker.h`, `src/arch/qwen3-tts/talker.cpp`
- Modify: `src/arch/qwen3-tts/code-predictor.h`, `src/arch/qwen3-tts/code-predictor.cpp`
- Create: `tests/qwen3_tts_icl_prompt_test.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: reference codes from Task 5 in the layout Task 5 settled — GGML `[16, T]`, `ne[0] = 16` group-fastest, byte-identical to the oracle's C-order `[frames, 16]`, no transpose at this boundary — reference text ids from Task 6, and target text ids.
- Produces: the ICL block, concatenated into `build_talker_prefill_input`'s output.

**This is the riskiest C++ piece in the plan, and it is a structural change rather than an extension.** `TalkerInputPosition` (`talker-host.h:19-32`) carries **one** `uint32_t codec_token` per position, and `flatten_talker_prompt` (`:93-97`) emits a single flat `codec_tokens` vector that `build_talker_prefill_input` (`talker.h:46-52`) adds as a contiguous tail. A reference block carries **sixteen** codes per position. That representation does not fit, and widening it touches every consumer: `flatten_talker_prompt`'s contiguous-run check, `codec_offset`, and the `external_speaker_index` bookkeeping — which is computed at `talker-host.cpp:57` (`speaker_codec_index = codec.size()`), assigned at `:100` (`out.external_speaker_index = speaker_codec_index`), and read at exactly one site, `model.cpp:982`. The two neighbouring lines are a different mechanism and must not be confused with it: `model.cpp:829` sets `prompt_request.speaker_is_external`, and `:889` is the `ggml_backend_tensor_set` that uploads the x-vector.

`sum_code_embeddings` (`code-predictor.h:65-69`, body `code-predictor.cpp:82-98`) already sums the 15 acoustic groups — but for **one frame at a time**, via `ggml_view_1d(context, codes, 1, table * codes->nb[0])` at `:90`. Extending it to `T` positions and adding the talker's own group-0 embedding is the work.

**What does not change:** the `external_speaker_index` / `speaker_is_external` mechanism (`talker-host.h:41-45, 66-71`) is untouched by ICL and stays. **The x-vector is still present in ICL mode** (`modeling_qwen3_tts.py:2166-2172`); ICL adds to that path, it does not replace it. A Plan 2 x-vector prompt must remain byte-identical, and a test asserts it.

- [ ] **Step 1: Write the failing test**

`tests/qwen3_tts_icl_prompt_test.cpp`, synthetic `HParams`, no GGUF. Every case is a rule the oracle's `alignment.json` supplies the target for:

1. **`T1 = len(ref_ids) + len(text_ids) + 1`** and **`T2 = 1 + ref_frames`**, computed from inputs the test controls.
2. **The truncating arm**: with `T1 > T2`, the block is `T2` positions, the text track contributes its first `T2` positions, and the trailing schedule is the remaining `T1 - T2` text positions. Drive it at `T2 = 14` — `base-ref-min`'s geometry.
3. **The padding arm**: with `T1 <= T2`, the block is `T2` positions, positions `T1..T2-1` of the text track are `tts_pad`, and the trailing schedule is a single `tts_pad`.

   **Both arms are tested here unconditionally, and neither is presumed covered elsewhere.** Tests 2 and 3 are written from `alignment.json`'s field names and geometry, not from which arm any case happened to take. Once Task 2 has run, read its report and add to **whichever** of the two tests corresponds to an arm no materialized case exercises: a comment saying that this unit test is the only thing exercising that arm until Task 12 lands a case. That comment is bookkeeping added after a measurement — it is not a licence to skip either test, and this plan does not know in advance which of the two it attaches to.
4. **The block is `T2` positions long in both arms.** Written as its own assertion because it is the property that makes `icl_embed.f32`'s size useless as a branch discriminator, and a future reader will otherwise try to use it as one.
5. **The codec track's first position is `codec_bos_id`** = 2149 (`synthesize.qwen3-tts.token.codec_bos_id`), embedded through the *talker's* codec embedding, not the code predictor's.
6. **Each reference position sums exactly 16 embeddings**, group 0 from the talker's table and groups 1..15 from the code predictor's, **in group order**. Permuting two groups must change the result — otherwise the tables are being summed as an unordered bag and a mis-ordered code grid is invisible.
7. **The block is appended after the existing prefix and replaces the first-text-token position** (`:2197`, `:2199-2202`), so an ICL prompt has the same total position count as a non-ICL one plus `T2 - 1`.
8. **A Plan 2 x-vector prompt is unchanged**: with no ICL inputs, `build_talker_prompt` produces exactly the positions it produced before this task, and the speaker slot is still substituted at the same index.

- [ ] **Step 2: Implement**

Widen the per-position codec representation, extend `sum_code_embeddings` to `T` positions, add the alignment, and concatenate. Keep `build_talker_prefill_input`'s new parameters defaulted so every Stage 1 call site compiles unchanged — the pattern Plan 2's Task 10 established for `speaker_embedding` / `speaker_index`.

- [ ] **Step 3: Measure against the oracle, per track and then summed**

Compare `text_track.f32` and `codec_track.f32` separately, then `icl_embed.f32`. **All three, in that order.** The summed comparison alone cannot separate a text-track error from a compensating codec-track error, which is the whole reason Task 2 exists.

- [ ] **Step 4: Prove the tests are load-bearing**

Swap the two alignment arms and confirm tests 2 and 3 fail. Reverse the group order in the sum and confirm test 6 fails. Shift the codec track by one position and confirm the per-track oracle comparison fails while the *end-to-end audio still synthesizes* — record that observation in the commit message, because it is the concrete evidence for this plan's whole validation strategy.

- [ ] **Step 5: Gates and commit**

```bash
git commit -m "qwen3-tts: the two-track ICL prompt block and its min(T1,T2) alignment"
```

---

### Task 8: The ICL Profile payload and its preparation

**Files:**
- Modify: `src/arch/qwen3-tts/profile.h`, `src/arch/qwen3-tts/profile.cpp`
- Modify: `tests/qwen3_tts_profile_test.cpp`

**Interfaces:**
- Consumes: the **free functions**, not the `Model` methods that wrap them — `encode_codec_reference(const HParams &, const CodecEncoderWeights &, …)` (Task 5) and `encode_speaker_reference(const HParams &, const SpeakerEncoderWeights &, …)` (Plan 2) — plus reference text ids the caller has already produced.
- Produces: `CloneMode::Icl`, the `IclProfile` payload, and `create_icl_profile(...)` beside `create_x_vector_profile`, at the signature **Task 5's Interfaces section pins**.

**Nothing in this task constructs a `Model`, and that is a constraint rather than a style.** `tests/qwen3_tts_profile_test.cpp` is registered as a **unit** test (`tests/CMakeLists.txt:178`), a `unit`-labelled test may not depend on the real 2.5 GB package, and `synth::qwen3tts::Model` has no constructor that avoids one — `Model::load`/`load_cpu` both require a real GGUF and this family has no synthetic-package harness (`profile.h:74-79`, which is the comment that put `create_x_vector_profile` on the two structs in the first place). So every test below runs against synthetic `HParams` and in-memory LCG weights, and where `create_icl_profile` needs reference text ids, the test supplies them directly — the same way the dispatch site will, via Task 6's `Model::tokenize_reference_transcript`.

**One assertion does not fit that and moves rather than shrinking.** The *selector* — the same public call producing an ICL Profile with a transcript and an x-vector Profile without one — lives in `src/voice-profile.cpp` and needs a live `Model`. It is asserted in `tests/qwen3_tts_base_load_real.cpp`, which is already an integration test guarded on `SYNTH_QWEN3_TTS_BASE_TEST_MODEL`, as part of **Task 10 Step 1**'s rewrite of `check_reference_transcript_refused` into a mode-selection check. It is not weakened by moving, and it needs no new registration. What stays here at the `unit` layer is the half that pins D4 at the family seam: each `create_*` function refuses the input naming the other mode.

**Every seam is pre-cut.** `CloneMode` (`profile.h:26-29`) declares `XVector = 0` with `// Icl -- Plan 3.` commented out. `XVectorProfile` (`:58-68`) already carries `language_tag`, added so "Plan 3's Icl payload does not need a shape change".

**The payload carries all three of D5's ICL rows**, including the speaker embedding — the design's table (`spec:123-128`) marks the `[1024]` embedding `yes` in **both** columns, and upstream inserts it regardless of mode. An ICL Profile is an x-vector Profile plus reference codes plus reference text token ids, not an alternative to it.

**A decision this task must make rather than inherit.** OmniVoice serializes `transcript_text` and **re-tokenizes on load** (`src/arch/omnivoice/profile.h:47-48, 64`; `profile.cpp:1464-1472`). D5 (`spec:114-128`) says the qwen3-tts ICL Profile carries **reference text token ids** and declares them recoverable. Follow D5: store ids, not the string. Record the divergence from OmniVoice in the header comment with the reason, so the next reader does not "fix" it into consistency.

- [ ] **Step 1: Write the failing test**

1. **Each `create_*` function refuses the input that names the other mode** (D4, mode fixed at preparation). `create_icl_profile` with an empty `transcript` — or with a non-empty transcript and empty `reference_text_ids`, or the reverse — is `SYNTH_ERR_INVALID_ARG` and builds nothing; `create_x_vector_profile` with a non-empty transcript still refuses rather than silently building the weaker Profile, which is the capability lie `profile.h:91-99` was written against. A successful `create_icl_profile` reports `CloneMode::Icl`; a successful `create_x_vector_profile` reports `CloneMode::XVector`. (The *selector* over the same public call is Task 10 Step 1's integration assertion, per this task's interface note.)
2. **An empty or whitespace-only transcript is `SYNTH_ERR_INVALID_ARG`** with a named diagnostic code, per `spec:419` — not "treated as absent". This is the distinction Plan 2's `test_a_whitespace_transcript_is_rejected_too` was written early precisely to preserve. Whitespace-only is a pure string judgement and needs no tokenizer, which is why it belongs in this function and in this test.
3. **The ICL payload carries all three**: a `[1024]` x-vector, codes in Task 5's settled GGML `[16, T]` group-fastest layout, and a non-empty id vector.
4. **The codes and the reference audio agree**: a Profile prepared from a 24,000-sample buffer carries 13 frames; from a 193,920-sample buffer, 101. Pure geometry, so LCG weights serve — derived from `hop_length` metadata, not a literal.
5. **A `language_tag` over `kMaxLanguageTagLength` is refused** by the ICL writer too — the tie `profile.h:31-49` documents applies to both kinds or it applies to neither.

- [ ] **Step 2: Implement, gate, commit**

Make `create_icl_profile` return `CloneMode::XVector` and confirm test 1 fails. Drop the empty-`reference_text_ids` guard and confirm test 1's half-present case fails.

```bash
git commit -m "qwen3-tts: CloneMode::Icl and the ICL Voice Profile payload"
```

---

### Task 9: The envelope extension

**Files:**
- Modify: `src/arch/qwen3-tts/profile.h`, `src/arch/qwen3-tts/profile.cpp`
- Modify: `tests/qwen3_tts_profile_test.cpp`

**Interfaces:**
- Produces: `serialize_icl_profile`, the `icl` value of `synthesize.voice_profile.kind`, `PrescanKeyScope::kIclOnly`, the ICL entries in `kPrescanKnownKeys`, and `kPrescanKvCountIcl`.

**The consequence Plan 3 must not be surprised by**, stated in-file at `profile.h:291-318`: a genuine ICL envelope presented to a **Plan 2 build** is rejected by the *prescan whitelist* before the `kind` branch is ever reached, and reports `SYNTH_ERR_INVALID_ARG` ("malformed") rather than `SYNTH_ERR_UNSUPPORTED_VOICE` ("newer than this build"). That is a disclosed, deliberate trade — it is what lets a Plan 2 profile stay loadable under a Plan 3 build with no schema bump — and this task does not try to undo it. **Widening the whitelist is a prerequisite for the `kind` branch becoming operative at all.**

**The scope trap, and the exact consequence of springing it.** `PrescanKeyScope`'s own header comment (`profile.h:148-159`) says the enum exists because a flat key-set check misses a key **moved between kinds**, or a per-kind count left stale. The ICL writer should emit `ref_rms` and `language_tag`, since the ICL payload carries both, so those two entries move from `kXVectorOnly` to `kCommon`. `kPrescanKnownKeys` (`profile.h:186-199`) holds exactly ten entries, of which **exactly those two** are `kXVectorOnly` — so the move leaves `kXVectorOnly` with **no members at all**, and this task must handle that rather than discover it:

- Either delete the `kXVectorOnly` enumerator and rewrite the enum's header comment to say the scopes are now `kCommon` and `kIclOnly`, or keep it with a comment stating it is deliberately empty and why. Do not leave an unexplained unused enumerator; that is what the next plan copies.
- `kPrescanKvCountXVector` **stays 10** (`profile.h:209`): the x-vector writer still emits the same ten keys, they are merely re-scoped. The "stale per-kind count" half of the trap does not fire for the x-vector kind, and the count that has to be right is the **new** `kPrescanKvCountIcl`. Say so in the change rather than dutifully "updating both counts" and moving a number that should not move.

- [ ] **Step 1: Write the failing test — the cross-kind tamper matrix**

1. **Round trip**: serialize an ICL Profile, load it, and get back identical codes, ids, x-vector, `ref_rms` and `language_tag`.
2. **Determinism**: the same payload serializes byte-identically twice.
3. **Every Plan 2 `x-vector` profile still loads** under the widened whitelist, byte-for-byte unchanged. Keep a committed Plan-2-shaped buffer in the test rather than regenerating one, so a writer change cannot quietly redefine what "a Plan 2 profile" means.
4. **An x-vector envelope with an ICL-only key added is refused.** The whitelist is per-kind, not a union.
5. **The per-kind required-key check works from the other side too, and from the moved keys.** The obvious dual — "an ICL envelope with an x-vector-only key missing" — is **unwritable after this task's own scope move**, because `kXVectorOnly` ends up empty and there is no such key left to omit. Write the two assertions that are satisfiable and pin the same machinery:
   - **An ICL envelope with an ICL-only key missing is refused** — the `kIclOnly` mirror of test 4, exercising the required-key path for the new kind.
   - **A `kCommon` key that moved is now required in BOTH kinds**: an ICL envelope missing `ref_rms` is refused, *and* an x-vector envelope missing `ref_rms` is refused. This is the "key moved between kinds" defect the enum was built for, asserted from both sides — and, unlike the original test 5, it is a statement that can actually fail.
6. **`n_kv` is exactly `kPrescanKvCountIcl`**, not a ceiling — and a hand-built envelope one key over or under is refused before `gguf_init_from_buffer` runs.
7. **An unrecognized `kind` is `SYNTH_ERR_INVALID_ARG`**, unchanged from Plan 2's mapping.
8. **The writer-agreement check extends to the ICL writer**: the real `serialize_icl_profile`'s key set is checked against `kPrescanKnownKeys` itself, not against a second hand-transcription.
9. **Payload-value parity**: a loaded ICL Profile satisfies the invariants a created one does — every code in `[0, 2048)`, a frame count consistent with the declared grid, finite non-all-zero x-vector, finite strictly-positive `ref_rms`. The x-vector half of this list is Plan 2's, measured after a reviewer found all four `ref_rms` states loading with `SYNTH_OK`; the codes half is new.

- [ ] **Step 2: Implement, prove, gate, commit**

Leave one ICL key out of `kPrescanKnownKeys` and confirm the round trip fails at the prescan. Set `kPrescanKvCountIcl` one too high and confirm test 6 fails.

```bash
git commit -m "qwen3-tts: the icl kind in the v1 envelope, with no schema version bump"
```

---

### Task 10: Open the public seam

**Files:**
- Modify: `src/arch/qwen3-tts/weights.cpp`
- Modify: `src/voice-profile.cpp`
- Modify: `tests/qwen3_tts_voice_required_test.cpp`, `tests/qwen3_tts_base_load_real.cpp`

**This is the change the design has been pointing at since it was approved.** `spec:171-175`:

> **"Plan 2 reports `reference_transcript` as `SYNTH_REQUIREMENT_UNSUPPORTED` and rejects a request that carries one.** Plan 3 flips it to `OPTIONAL` in the same change that lands ICL, and the rejection becomes the mode selector D4 describes. `reference_language` follows the transcript: it qualifies a transcript this rung cannot use."

And immediately after, `spec:177-178`:

> "This is a statement about the runtime, not the package: the Base package's declared contract is unchanged, and its `reference_*` limits stay as validated."

**Both halves are binding.** The flip and the mode selector land together — a build that reports `OPTIONAL` while `voice-profile.cpp` still refuses a transcript is the capability lie the design's erratum exists to prevent, in the opposite direction. And the package does not change: `source_flags` stays exactly `SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO | SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE`, the six reference limits stay, the schema stays `qwen3-tts-voice-clone` at version 1, and the Compatibility ID stays `34d4de22…806e`.

- [ ] **Step 1: Rewrite the assertions, do not delete them**

Four sites assert the current shape and every one is a rewrite:

- `tests/qwen3_tts_voice_required_test.cpp:192-194` — `UNSUPPORTED` becomes `OPTIONAL` for both fields. **`:121-123` is an adversarial non-Base case and must keep asserting `UNSUPPORTED`** — it is the assertion that would catch a capability gate keyed on the wrong predicate.
- `tests/qwen3_tts_base_load_real.cpp:91-92` — the same flip against the on-disk metadata.
- `tests/qwen3_tts_base_load_real.cpp:222-283` — `check_reference_transcript_refused` (asserting diagnostic code `voice_profile.transcript_unsupported` at `:243`) and `check_reference_language_refused` (`:280`), both wired at `:547-548`, become **mode-selection** checks: a transcript now yields a Profile whose `kind` is `icl`, and a valid language tag is accepted. `check_reference_transcript_refused`'s rewrite is also where **Task 8's selector assertion lands** — the same `synth_voice_profile_create_from_reference` call, once with a transcript and once without, producing `icl` and `x-vector` respectively. It belongs here rather than in Task 8's unit test because it needs a live `Model`.
- **The accepted tag is `"english"`, not `"en"`, and the existing comment at `:250-257` is wrong about this.** That comment calls `"en"` "a real, package-declared tag". The Base package declares its languages as full English names — `general.languages = ['chinese','english','french','german','italian','japanese','korean','portuguese','russian','spanish']`, and `synthesize.qwen3-tts.languages.names` likewise — and every Golden Manifest case carries `reference.language_tag` values from that set (`"english"`, `"chinese"`, `"japanese"`). Under Step 3's validation `"en"` passes `valid_bcp47_shape` and then fails `declared_language`, so it is refused with `SYNTH_ERR_UNSUPPORTED_LANGUAGE`. Rewrite the function to accept `"english"` and to keep refusing `"en"` and `"zz-ZZ"` — now by their correct, *different* statuses rather than identically — and correct the stale comment in the same change, since a false comment about which tags this package declares is exactly what would send the next reader to `"en"`.
- `info.description_language` stays `UNSUPPORTED` and `info.source_flags` stays exactly the two bits. Assert equality, not membership.

- [ ] **Step 2: Flip the capability**

`src/arch/qwen3-tts/weights.cpp:788-789` becomes `SYNTH_REQUIREMENT_OPTIONAL` for both, and the comment block at `:784-787` — which currently reads *"UNSUPPORTED, not OPTIONAL: this rung implements the x-vector mode only… Plan 3 flips both in the change that lands ICL"* — is rewritten to state that both modes now exist and that the transcript's presence selects between them per D4.

- [ ] **Step 3: The two rejections become the mode selector**

`src/voice-profile.cpp:499-504` and `:515-520`:

- The transcript rejection at `:499-504` becomes mode selection: present ⇒ tokenize through `Model::tokenize_reference_transcript` (Task 6) and call `create_icl_profile` with the resulting ids; absent ⇒ `create_x_vector_profile` exactly as today. **An empty or whitespace-only transcript stays `SYNTH_ERR_INVALID_ARG`** (`spec:419`). **The selection lives here and only here**, because this is the one site in the chain that holds a live `Model &` and can therefore reach the BPE tables — see Task 5's pinned signature for why that decides the whole shape.
- The language rejection becomes **validation**, not acceptance. The comment at `:508-514` records that a reviewer measured `language_tag="zz-ZZ"` returning `SYNTH_OK` before the blanket refusal went in, and names OmniVoice's `declared_language` as the shape that would have refused it. A tag that is not one of the package's declared languages is refused with `SYNTH_ERR_UNSUPPORTED_LANGUAGE`; a shape that is not BCP-47 is refused with `SYNTH_ERR_INVALID_ARG`. **Do not replace a blanket refusal with a blanket acceptance** — that would reintroduce the exact defect the refusal closed.

  This mapping is **not** a row in the design's §9 error table (`spec:417-424` has six rows and none is a reference-language row; `spec:419`, cited above for the empty transcript, is the empty-transcript row). It is not invented either: `src/voice-profile.cpp:308-314`, OmniVoice's own handler, already implements exactly this pair — `valid_bcp47_shape` → `SYNTH_ERR_INVALID_ARG`, then `declared_language` → `SYNTH_ERR_UNSUPPORTED_LANGUAGE`. Follow that in-tree precedent, and do not re-litigate it mid-plan on the grounds that §9 is silent.

  **CORRECTION, measured at Task 10 — the two paragraphs above are wrong, and the comment they told you to "fix" was right.** The accepted tag is `"en"`. Three different vocabularies were being conflated: the package's raw metadata names languages in full (`general.languages` holds `chinese english french …`), the **public interface speaks BCP-47**, and the Golden Manifest describes the *oracle's* vocabulary. `src/arch/qwen3-tts/model.cpp:178-196` bridges the first two on purpose, with a comment saying so — `{ "en", "english" }` and nine more — and `synth_model_get_language` on the real package returns `en de es zh ja fr ko ru it pt`, measured. So `"en"` is what a caller passes and `"english"` is what the package stores; the existing comment at `tests/qwen3_tts_base_load_real.cpp:250-257` was correct in its own context and must not be "corrected". Tests on this path assert the accepted tag **from the package** rather than hardcoding either spelling. The rule the paragraphs above state — shape first, then declared-vocabulary membership, with the two distinct statuses — is unaffected and stands.

`create_x_vector_profile`'s own defence-in-depth transcript refusal (`profile.h:91-99`, unreachable today per `voice-profile.cpp:559-570`) **stays a refusal and does not become the dispatch**, and its header comment — which currently reads "Plan 3 is the change that turns this rejection into the mode selector" — is rewritten to say where the selector actually went and why it could not go here: that function takes `(HParams, SpeakerEncoderWeights, …)` rather than a `Model &` precisely so it stays unit-testable, so it cannot tokenize and cannot produce `create_icl_profile`'s `reference_text_ids`. It keeps refusing a non-empty transcript, now unreachable for a second reason as well — the dispatch above routes one to `create_icl_profile` before it ever gets here — and Task 8 test 1 asserts that refusal directly at the `unit` layer.

- [ ] **Step 4: Prove, gate, commit**

Flip the capability back to `UNSUPPORTED` and confirm the rewritten tests fail. Drop the `declared_language` check so `"zz-ZZ"` and `"en"` are accepted, and confirm the language test fails on both. Drop the `valid_bcp47_shape` check and confirm the malformed-shape case stops returning `SYNTH_ERR_INVALID_ARG` — the two halves fail independently or one of them is not being tested.

```bash
git commit -m "qwen3-tts: reference_transcript becomes OPTIONAL and selects the clone mode"
```

The commit message states that the package's declared contract is unchanged and that `source_flags` is still exactly 9.

---

### Task 11: Dispatch at the synthesis seam

**Files:**
- Modify: `src/synthesize.cpp`
- Modify: `src/arch/qwen3-tts/qwen3-tts.h`, `src/arch/qwen3-tts/model.cpp`

**The guard was written for this day.** `src/synthesize.cpp:1082`:

```cpp
if (clone.mode != synth::qwen3tts::CloneMode::XVector) {
    emit_diagnostic(prepared.diagnostics, SYNTH_ERR_UNSUPPORTED_VOICE, "synthesis.voice_unsupported",
                    "this build supports x-vector Voice Profiles only");
    return SYNTH_ERR_UNSUPPORTED_VOICE;
}
```

with the comment at `:1075-1081` saying it was guarded now "so that landing Icl does not also have to remember to add this refusal". The refusal becomes a dispatch; `family_request.x_vector = &clone.x_vector` (`:1096`) stays for **both** modes and the ICL fields are added alongside it, borrowed from the Profile's `shared_ptr` payload whose lifetime the handle owns for the whole call — the same borrow shape `replay_codes` already uses.

- [ ] **Step 1: Add the ICL fields to `SynthesisRequest` and extend the mutual-exclusivity enforcement**

`SynthesisRequest::x_vector` (`qwen3-tts.h:129-138`) gains reference-code and reference-id siblings. `model.cpp:775-817` is the mutual-exclusivity and width enforcement; extend it so that the ICL fields are all-present or all-absent, never half. Half-present ICL inputs are the state that produces a plausible-sounding wrong prompt.

- [ ] **Step 2: Test the refusals by name**

A Profile from a different Loaded Model still refuses with `SYNTH_ERR_UNSUPPORTED_VOICE` and the diagnostic code `synthesis.voice_unsupported`, never `synthesis.graph_failed`. A request with no Voice Profile on this Catalog-less package still fails explicitly rather than selecting a Voice (D6).

- [ ] **Step 3: Prove, gate, commit**

Drop the ICL fields at the seam so an ICL Profile silently synthesizes as x-vector-only, and confirm a test fails. **If nothing fails, that is the finding** — write the test that does, because that silent downgrade is the failure mode this plan exists to make visible.

```bash
git commit -m "qwen3-tts: consume an ICL Voice Profile at the synthesis seam"
```

---

### Task 12: Close the provisional tolerance stage and the oracle sweep

**Files:**
- Modify: `tests/tolerances/qwen3-tts.json`
- Modify: `scripts/validate-qwen3-tts-replay.py`
- Modify: `tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-base.manifest.json`

**The constraint that fixes the shape of this task, so it is not rediscovered.** `tests/python/test_tolerance_coverage.py:129-148` asserts **exact set equality** between the validator basenames globbed from `scripts/validate-qwen3-tts-*.py` (stripped of the prefix) and each variant's reference-profile stage set. Both sides are `{public, replay}` today. **A `codec_encoder` stage therefore cannot be promoted into `variants` without a `scripts/validate-qwen3-tts-codec_encoder.py`** — and creating one would force a `codec_encoder` stage into CustomVoice's `BF16`/`F16`/`Q8_MIXED` grids and their CUDA sub-grids, five cells describing a subsystem that variant does not have. Plan 2 hit this exact wall and its Task 6 resolved it by landing the gate at **probe** granularity inside the existing `replay` stage, via a new comparison **mode** on `validate-qwen3-tts-replay.py` rather than a new basename (plan-2 file-structure row `:61`). Do the same.

**How Plan 2 closed the twin, quoted from the file's own `provisional_variants` note:**

> "`speaker_encoder` lived here beside `codec_encoder` until 2026-08-12, with the identical caveat and the identical copied numbers. It was retired the day Task 6 gave the speaker path a real cell … keeping both would have described the same subsystem twice, once as an honest placeholder and once as an actual measurement, and the placeholder is the one with nothing left to say."

`provisional_variants["qwen3-tts-12hz-0-6b-base"].profiles.BF16.stages` now holds **exactly one** entry, `codec_encoder`, `cases: 0`, whose eight probes (`talker.hidden_l0/l7/l14/l21/l27`, `talker.final`, `talker.logits`, `audio.pcm`) are copied verbatim from the **CustomVoice** BF16 replay stage and whose own `description` disclaims them: *"they describe a different model's talker prefill probes, not this stage's own tensors."* Plan 1 carryover §3 Task 3(c) assigns it: *"Plan 3's to close the same way, once it gives the codec encoder a real measurement to replace it with."*

- [ ] **Step 1: Materialize the seven missing ICL cases**

Ten of the manifest's twelve cases are ICL and three have artifacts (`base-icl-en`, `base-ref-min`, `base-text-short`). The seven missing: `base-upstream-clone-en`, `base-icl-zh`, `base-icl-ja`, `base-ref-max`, `base-text-long`, `base-seed-one`, `base-seed-forty-two`. Run the existing Base oracle plus Tasks 1 and 2's scripts over each.

The seeded-sampling rule applies to every one of them because they run the talker (`spec:338-358`): **both** switches true, fixed seed, `max_new_tokens 2048`, the sampled sequence recorded into `stochastic_inputs`. Setting only one is the Stage 1 trap that is silently non-deterministic.

**Every case is dumped at the same oracle dtype, and each one's `conventions.json` records which** (Task 1 Step 3). This is not housekeeping: a case dumped under a load that omitted `dtype=torch.bfloat16` sits on a float32 codebook that disagrees with the bf16 one on roughly half its codes, and its stage artifacts would carry an F32 stack the other cases do not — so it would silently move this task's whole tolerance grid. A case whose recorded dtype differs from the rest is regenerated, not averaged in.

- [ ] **Step 2: Cover both alignment arms, explicitly**

From Task 2's `alignment.json`, tabulate `T1`, `T2` and the branch for all ten ICL cases. **At least one case in each arm must be measured.** If the ten cases all land in the same arm, add a case to the manifest that does not — the shortest reference clip the package accepts is 24,000 samples = 13 frames = `T2` 14, and the target/reference text lengths are free, so both arms are reachable within the declared limits. Record which case covers which arm in the manifest's case description, not only in a commit message.

**Adding a case is two edits, not one.** `tests/tolerances/qwen3-tts.json` carries `variants["qwen3-tts-12hz-0-6b-base"].case_count`, today `12`, and `test_tolerance_case_count_matches_manifest` (`tests/python/test_golden_manifests.py:165`, assertion at `:207-209`) requires it to equal `len(manifest["cases"])`. Step 6 runs that module. So a new manifest case moves `case_count` to `13` in the same change, along with the `note` field that currently spells out "case_count (12) tracks this variant's manifest". If no case is added — because the ten already cover both arms — `case_count` stays `12` and the tolerance edits are Step 4's only.

- [ ] **Step 3: Adjudicate the 9-frame anomaly**

The inherited record, and it is inherited **unresolved**. Plan 2 carryover §2.2 and Plan 1 carryover §2.2 record the anomaly and that it stays open; the *numbers* are not in either carryover — they live in the family record's reference-bounds table, `docs/porting/families/qwen3-tts.md:2043-2055`, whose row `:2050` reads `| **30 s** | 720,000 | 4x | 375 | **9** | **0.72 s** |`, with the surrounding prose at `:2052-2058`. A 30 s reference (720,000 samples, looped 4×, 375 reference frames) produced **9 generated frames / 0.72 s** for an 11-word sentence that takes ~45 frames at the clip's native length. Regenerating that case *in x-vector mode* gave 46 frames — **which is not evidence about ICL**, and the carryover says so twice.

The artifacts survive at `build/qwen3-tts-reference-bounds/dur-30s.report.json`, but **not at the top level** — that file is manifest-shaped (`schema`, `family`, `variant`, `reference`, `case_count`, `cases`) and the values are nested one case deep. Under `cases[0].result`: `generated_code_frames 9`, `duration_seconds 0.72`, `peak_abs 0.474609375`, `rms 0.0686`, `distinct_semantic_codes 8`, `most_common_semantic_code_fraction 0.222`, `reference_code_frames 375`, and a `reference_audio` sub-object recording `source_samples 193920` / `loop_repeats 4` / `applied_samples 720000`. Under `cases[0].artifacts`: `prompt.icl_embed` shape `[1, 376, 1024]` and `codes.reference` shape `[375, 16]`. That case can be driven again without re-sourcing audio.

Plan 3 is the first plan that can render it through its own implementation. Compare the port's render against the oracle's for the same case and the same recorded draw, and report one of three outcomes:

- **The port reproduces 9 frames.** Then the behaviour is upstream's and belongs in the family record as a measured property of long references in ICL mode, with `base-ref-max` annotated accordingly.
- **The port does not reproduce it.** Then the port and the oracle disagree on a case that runs to completion, which is a defect in the port and must be found before this plan closes.
- **The oracle no longer reproduces it either.** Then record what changed since 2026-08-11 and treat the original observation as retired, the way the x-vector regeneration was.

One structural observation is available and is offered as a **hypothesis, not a finding**: at 375 reference frames `T2 = 376`, which would far exceed `T1` (reference transcript + a short target text + eos), and *if* the padding arm runs then the entire target text is consumed *inside* the prompt and the trailing schedule is a bare `tts_pad_embed` — a shape in which an early EOS would be unsurprising. **Which arm this case actually takes is a measurement this step must make, not an assertion this plan gets to hand it.** `dur-30s` is a reference-bounds case rather than a Golden Manifest case, so Task 2's script does not cover it by default: run Task 2's script over this case as part of driving it again, and read the branch out of the `alignment.json` that produces. The same `T2`-length-block argument that makes the branch unrecoverable everywhere else applies here too — `prompt.icl_embed`'s `[1, 376, 1024]` is consistent with both arms. Whether the arm, or the 4× loop of the same clip, or something else drives the early EOS is what this step adjudicates. Do not write the hypothesis into the family record as a cause, and do not key the three outcomes above to it.

- [ ] **Step 4: Add the real probes to the existing `replay` stage**

Add comparison modes to `scripts/validate-qwen3-tts-replay.py` — it already has `--compare-mel` (`:110-124`) and `--compare-x-vector` (`:126-153`) with `run_compare_mel` at `:161` and `run_compare_x_vector` at `:275`; add `--compare-codec-encoder` and `--compare-icl-prompt` in the same shape. **Do not add a new `validate-qwen3-tts-*.py` basename.**

Then fill `variants["qwen3-tts-12hz-0-6b-base"].profiles.BF16.stages.replay.probes` with measured cells beside the existing `speaker.x_vector`:

- **The codec encoder's continuous stages**, which are the gate the spec's fourth erratum installs in place of code equality. One probe per tapped stage, keyed to the artifact it compares: `codec.seanet_stage0`…`codec.seanet_stage3`, `codec.seanet_tail`, `codec.downsample`, `codec.transformer_l0`…`codec.transformer_l7`, `codec.latents`, `codec.rvq_residual_s00`…`codec.rvq_residual_s15`. Rolling them into one summary probe is not acceptable: a single number cannot say *which* stage drifted, and locating the stage is the entire reason the oracle taps eleven of them.
- `codec.rvq_reconstruction` — the dequantized reconstruction in the 256-wide projected space, the other half of that gate. **This key is mandatory**; the grid is not complete without it, because it is the only probe that says anything at all about the RVQ's selections.
- `prompt.icl_embed` — an F32 probe, gated on the rule this file already states for every other probe (`1 - 5*(1 - observed)` for cosine);
- a per-track probe if Task 7's measurement warrants one.

**No `codec.reference_codes` probe, and no `alternate_grids` anywhere in the manifest.** The erratum rejects both: a discrete-equality probe would test which `dtype` the oracle was loaded with, and enumerating admissible grids would mean committing a second oracle's output for every case. The code agreement rate Task 5 Step 3 measured is recorded instead — in the `note` of the codec probes and in this task's commit message, where it is legible as an observation. A `note` is not a gate, which is exactly why it is the right carrier.

**Every probe's gate is measured and then widened by this file's existing rule — with one added obligation.** Each measured deviation is also checked against the bf16-derived expectation for its stage, since the oracle's activations and codebook are bfloat16 and the port's are F32. A stage whose deviation lands above that scale does not get a wider gate; it gets found. Note it in the probe's own `note` when a stage sits materially below the scale too — that is information about where the port is tight, and the next plan will want it.

Move `variants[base].status` off `"partial-measurement-speaker-path-only"` and `replay.cases` off `2`. **And `case_count` off `12` if and only if Step 2 added a manifest case.** The complete list of tolerance-file edits this task makes is: `status`, `replay.cases`, the new probes, the variant `note` — which currently describes `replay` as "no longer entirely a placeholder" and names `provisional_variants` as still holding `codec_encoder`, both of which stop being true here — and, conditionally, `case_count`. Every number is measured in this task; **this plan prescribes none of them.**

- [ ] **Step 5: Delete the provisional stage, and the container with it**

Delete `provisional_variants["qwen3-tts-12hz-0-6b-base"].profiles.BF16.stages.codec_encoder`. It is the only entry, so `provisional_variants` disappears entirely, and the note explaining the retired `speaker_encoder` sibling goes with it — preserve that note's substance in the commit message, since the file will no longer carry it.

- [ ] **Step 6: Watch the coverage tests pass for the right reason**

```bash
uv run --project scripts/envs/qwen3-tts --locked python -m unittest \
  tests.python.test_tolerance_coverage tests.python.test_validate_qwen3_tts_replay \
  tests.python.test_golden_manifests -v
```

Then prove the constraint is real rather than assumed, exactly as Plan 2 did: create an empty `scripts/validate-qwen3-tts-codec_encoder.py`, re-run, and watch `test_every_registered_validator_has_a_measured_stage` fail for **both** variants. Delete it again.

- [ ] **Step 7: Commit**

```bash
git add tests/tolerances/qwen3-tts.json scripts/validate-qwen3-tts-replay.py \
        tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-base.manifest.json
git commit -m "qwen3-tts: commit the measured codec and ICL-prompt tolerances, retire the provisional stage"
```

The commit message states the observed values, the case ids, the backend, the build, which case covers which alignment arm, the 9-frame adjudication's outcome, and **the per-case code agreement rate with the note that it gated nothing** — the tolerance schema has no field for any of that, so the commit is the only carrier.

---

### Task 13: Reference audio and transcript in, cloned audio out

**Files:**
- Create: `tests/qwen3_tts_icl_real.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: everything above.
- Produces: the spec's §8 gate — "ICL clone runs end to end; codes and prompt layout meet oracle tolerances; Golden Manifest covers both modes" — where "codes … meet oracle tolerances" is read through the spec's fourth erratum: the stage-wise F32 artifacts and the dequantized reconstruction, at a bf16-derived tolerance. Task 5 Step 2 and Task 12 Step 4 own that half; this task owns end to end.

- [ ] **Step 1: Write the integration test**

On the pattern of `tests/qwen3_tts_clone_real.cpp` — `main` taking the model, a reference WAV, the oracle artifacts and a scratch directory — registered doubly guarded on the model and the reference clip, with `LABELS "integration;qwen3-tts;abi"` and a TIMEOUT of 3600. Note that this file is **not** built by `synthesize-check-unit` (Plan 2 carryover §3); the integration configuration is what builds it.

What it proves, each its own assertion:

1. A Profile prepared with a transcript reports `kind` `icl` and carries a `[16, T]` grid whose `T` matches the clip's frame arithmetic, with every code in `[0, 2048)`. **The oracle check on this Profile is its dequantized reconstruction against `rvq_reconstruction.f32`, inside the `codec.rvq_reconstruction` tolerance Task 12 committed — not equality against `codes/reference.i32`.** The spec's fourth erratum is why: the oracle's codebook is bf16, the package's is f32, and `base-icl-en` is one of the cases the two tables disagree on. The test still *records* the agreement rate, so a regression from a few percent to a few tens of percent is visible in the output of a failing-or-passing run — and it still asserts, through `codes_equal`, that preparing the same clip twice yields the identical grid, which is the port-against-itself claim that stays exact.
2. Synthesis with it produces finite, non-silent PCM at 24 kHz.
3. The same Profile and seed produce identical PCM twice.
4. **The same reference clip with and without a transcript produces different PCM.** This is the assertion that gates "ICL" as opposed to "cloned" — an ICL Profile whose extra fields are dropped at the seam produces perfectly good x-vector audio, and nothing else in this plan sees that from the outside.
5. serialize → free → load → synthesize produces the same PCM as the original Profile.
6. An ICL Profile presented to a second Loaded Model refuses with `SYNTH_ERR_UNSUPPORTED_VOICE` and writes no audio.
7. A Plan 2 x-vector Profile, serialized before this branch, still loads and still synthesizes.

- [ ] **Step 2: Verify the binding needs nothing, rather than assuming it**

Plan 2 Task 12 Step 3 established the binding is family-generic. Re-run its grep, then drive the real wheel with a transcript and confirm `reference_transcript` now reports optional through the extension — **with no change to any file under `bindings/`**. If any line has to change, the claim was wrong and the change belongs here with its own test.

- [ ] **Step 3: Run it, gate, commit**

```bash
cmake -S . -B build-integration -DSYNTH_BUILD_TESTS=ON -DSYNTH_BUILD_INTEGRATION_TESTS=ON \
  -DSYNTH_QWEN3_TTS_BASE_TEST_MODEL=models/qwen3-tts-12hz-0-6b-base/qwen3-tts-12hz-0-6b-base-BF16.gguf
cmake --build build-integration --target synthesize-qwen3-tts-icl-real
ctest --test-dir build-integration -R '^synthesize-qwen3-tts-icl-real$' --output-on-failure
```

```bash
git commit -m "qwen3-tts: reference audio and transcript in, cloned audio out -- the ICL path end to end"
```

---

### Task 14: The documentation obligations, including D5's

**Files:**
- Modify: `docs/voice-conditioning.md`
- Modify: `docs/porting/families/qwen3-tts.md`

**D5's declaration does not exist yet, and it is a confirmed-contract change.** `docs/voice-conditioning.md:63` currently states that Profiles "omit enrollment audio, transcript, and description source material by default" — and that sentence, unamended, is now incomplete for this family. D5 (`spec:112-121`):

> **"D5 — An ICL Profile carries recoverable transcript content, and says so.** The ICL graph consumes reference text token ids and reference codes at synthesis time, so a Serialized ICL Profile must carry both. … Token ids are derived conditioning rather than the source string, but byte-level BPE is invertible, so whoever holds the Profile can recover the transcript. The Profile Schema declares the field explicitly and the documentation states the recoverability, so that distributing a Profile is an informed act. Profiles never carry enrollment audio."

- [ ] **Step 1: Amend `docs/voice-conditioning.md`**

State, in the document's own register: a Model Family Profile Schema may declare derived conditioning that is invertible to its source, and where it does, the Schema declares the field explicitly and this document names the consequence — **a Serialized qwen3-tts ICL Profile carries reference text token ids, byte-level BPE is invertible, and whoever holds the Profile can recover the reference transcript**. Enrollment audio is still never carried. Amend the sentence at `:63` in the same edit, rather than adding a paragraph that contradicts it further down.

**This document has no `Status: Confirmed …` line, so decide rather than pattern-match.** `docs/voice-conditioning.md:3` reads *"Status: Research snapshot on 2026-07-21; the public source set, all four source-specific C representations, Reference Audio normalization, and the Serialized Profile format and compatibility rules are confirmed."* — a survey with a partial confirmation clause, not a confirmed contract with a date. **Do not overwrite `2026-07-21`**: that date is the snapshot's provenance, and moving it would falsely claim the whole survey was re-taken. Instead do two things to that line: extend the confirmation clause so the recoverability declaration is named among the confirmed items, and append `; amended 2026-08-13 for the qwen3-tts ICL Profile's recoverable transcript`. The result still reads as a snapshot, and it now says truthfully which parts of it the project is bound by.

The recoverability declaration itself is a promise, not a note — write it as one.

- [ ] **Step 2: Extend `docs/porting/families/qwen3-tts.md` with a "Stage 2, Plan 3" section**

Carrying: the codec-encoder conventions from Task 1's `codec_encoder/conventions.json` with their `transformers==4.57.3` line numbers, flagged as this family's **third provenance**; the ICL-prompt conventions from Task 2's `prompt/prompt_conventions.json`; the reconciliation table from Task 1 Step 2, including the two config keys that sit at the top level rather than under `encoder_config`; **the codebook-dtype finding and the gate that followed from it** — the oracle's bf16 table against the converter's f32 one, the measured divergence, jiangzhuo's ruling of 2026-08-13 dropping code equality, and the recorded agreement rate with the statement that it gates nothing (the spec's fourth erratum is the ruling's home; the family record is where a later variant's implementer will look for it); the measured codec and prompt tolerances with case ids, backend and build; the two alignment arms and which case covers each; the 9-frame adjudication's outcome; the Profile Schema's two `kind` values with the note that `icl` landed without a version bump; which of the six family-conditioned dispatch points Plan 3 changed; and what remains — Description Text (Stage 3), the CLI, quantization, CUDA, and the Plan 4 listening pass.

**The tensor-count fix is an addition, not a rewording.** The census at `:1983-1995` is whole-package and correct as it stands; what is missing is the encoder's own share, because **161 appears nowhere in this file**. Add the encoder line beside the package census — `225 raw − 32 encoder EMA-accumulator pairs − 32 encoder .initialized flags = 161 emitted` — and say in the same place that the design's §1.1 row saying 225 is the raw count while a graph is built from the 161. Then update the `Status:` line.

- [ ] **Step 3: Commit**

```bash
git add docs/voice-conditioning.md docs/porting/families/qwen3-tts.md
git commit -m "qwen3-tts: record Stage 2 Plan 3, and declare the ICL Profile's recoverable transcript"
```

---

## Risks

**1. The two-track alignment, and the 9-frame anomaly sitting inside it.** This is the risk that shapes the plan, and it leads because it is the only one a competent implementation can get wrong without anything noticing. The alignment rule branches on relative lengths (`modeling_qwen3_tts.py:2015-2019`); **which arm any artifact on disk exercises is unknown until Task 2 measures it, and unknowable from the artifacts themselves, because both arms produce a `T2`-length block and nothing on disk records the trailing schedule that discriminates them.** This plan therefore asserts no arm anywhere, and no task is keyed to an assumed one. Meanwhile the one anomaly Plan 3 inherits — a 30 s reference producing 9 generated frames — sits precisely in that region: a *behavioural* defect with no established mechanism, in a mode whose port does not exist yet, so it cannot be reproduced-then-fixed in the usual order. And the whole class is invisible to the acceptance method that carried Plan 2: a wrong alignment produces fluent speech in approximately the right voice. **Mitigation:** Task 2 makes the branch an artifact rather than an inference; Task 7 tests both arms directly at the unit layer and measures per track before summed; Task 12 requires a measured case in each arm and adjudicates the anomaly with three named outcomes rather than an expectation. **Residual risk:** if the anomaly turns out to be an upstream property of long references in ICL mode, `base-ref-max` documents a case whose output is 0.72 s of audio for an 11-word sentence, and the declared 30 s upper bound becomes a question for Plan 4's listening pass rather than a settled limit.

**2. A third provenance.** The codec encoder's reference is `transformers==4.57.3`'s `MimiModel` — a source no prior stage of this family has read, alongside `QwenLM/Qwen3-TTS@022e286` and the checkpoint's own config. A convention taken from the wrong one of the three (causal padding mode, ELU placement, the residual dilations, `trim_right_ratio`, the split RVQ's stage order) produces finite codes that decode to plausible audio. **Mitigation:** Task 1 Step 1 transcribes eight named conventions with `file:line`, and Task 1 Step 2 reconciles the checkpoint's `encoder_config` against the port's derived widths in a table before any graph is written.

**3. The prompt representation change is not local.** Widening `TalkerInputPosition` from one code to sixteen touches `flatten_talker_prompt`'s contiguous-run invariant, `codec_offset`, and the `external_speaker_index` mechanism — computed at `talker-host.cpp:57`, assigned at `:100`, read at the single site `model.cpp:982`. **Mitigation:** Task 7's test 8 asserts a Plan 2 x-vector prompt is unchanged, and `build_talker_prefill_input`'s new parameters default so every Stage 1 call site compiles untouched.

**4. Discreteness hides errors until it does not — and the codes cannot be the gate that catches it.** The encoder's output is quantized, so a latent that is slightly wrong produces codes that are exactly right until a margin is narrow, and then one code flips and the voice changes. The measurement behind the spec's fourth erratum sharpened this from a worry into a fact: the oracle's bf16 codebook and the converter's f32 one already disagree on 4.04% of codes with everything else held identical, so an equality gate would have failed on correct ports and passed on a load-time keyword argument. **Mitigation:** the gate moved to the continuous artifacts — every SEANet stage, the downsampler, all eight transformer layers, all sixteen RVQ residuals, and the dequantized reconstruction, each at a bf16-derived tolerance, with a stage above that scale treated as a defect rather than a tolerance to widen (Task 5 Step 2, Task 12 Step 4). **Residual risk, stated plainly: a flipped code is not a small error downstream.** It selects a different embedding row in the ICL prompt, and no continuous probe in this plan can see that happen. What the continuous gate buys is the ability to tell a near-tie from a structural error; what it does not buy is a proof that the port's selections match upstream's, and nothing in this plan may be written as though it did.

**5. The prescan whitelist is a prerequisite, not a detail.** A genuine ICL envelope is refused by the whitelist before the `kind` branch runs, and the per-kind scope enum exists specifically to catch a key moved between kinds with a stale count. **Mitigation:** Task 9's tamper matrix includes the moved-key and off-by-one-`n_kv` cases, and the writer-agreement test drives the real writer rather than a transcription.

**6. Scope creep into Plan 4.** The codec encoder is convolution-heavy, which is the shape that produced another family's conv-exempt quantization policy, and the temptation to measure a Quantization Profile while the graph is fresh will be real. **The design assigns it to Plan 4** — Quantization Profiles at `spec:390`, inside §7 (`:388-399`), and the CUDA Execution Backend in §8's plan-sequence row at `spec:408`, since §7's own CUDA paragraph (`:397-399`) names the wiring but not the plan — and Plan 4 measures rather than importing a precedent by analogy. This plan claims no quantization result and no performance number.

---

## What Plan 3 does NOT deliver

Stated plainly, so nothing here is read as more than it is.

- **Quantization Profiles and the CUDA Execution Backend.** Plan 4's, per the design's §7. No `F16`, `Q8_MIXED` or `CUDA` cell is added to the tolerance grid; no performance number is claimed; the conv-heavy-encoder question is measured there, not assumed here from another family.
- **Any movement of the Validation Level.** `port_validated` plus a listening audit is the delivery bar; `quality_evaluation` stays deferred per ADR 0017. The Listening Audit of 2026-08-13 is evidence, not a level — and, as this plan's validation strategy argues at length, an audit could not gate this work even if one ran.
- **The Plan 4 listening pass.** Not run here. If the 9-frame adjudication lands on "upstream property", the 30 s bound becomes a question that pass has to answer.
- **Description Text.** Stage 3, on `qwen3-tts-12hz-1.7b-voicedesign`. `SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT` stays unadvertised; Random Seed stays unadvertised.
- **The CLI path.** `examples/cli/` untouched, for the reasons recorded in the family document.
- **Any converter change or package re-cut**, unless a task proves one necessary and carries the byte-identity proof for the published CustomVoice package with it.
- **Publication.** Separate, and requiring jiangzhuo's confirmation at the time.

---

## Self-review

**Spec coverage.** D1's second mode is the whole plan. D4's "mode fixed at preparation" is Task 8 and Task 10. D5's three payload rows are Task 8 and its recoverability declaration is Task 14 — the one D5 obligation that was entirely absent from the tree. D6 is unchanged and re-asserted at Task 11 Step 2. D7's limits are unchanged and explicitly protected in Global Constraints. Section 3's `OPTIONAL` flip is Task 10, with the design's own wording quoted rather than paraphrased and its "statement about the runtime, not the package" half made a constraint. Section 5's two remaining module rows (`codec-encoder{,-host}`) are Tasks 4 and 5. Section 6's two missing dump scripts are Tasks 1 and 2; its seeded-sampling erratum governs Task 12 Step 1. **Its discrete-equality rule and its `alternate_grids` escape are gone, replaced by its fourth erratum (2026-08-13)** — the gate is the stage-wise F32 artifacts at a bf16-derived tolerance plus the dequantized reconstruction, which is Task 1's artifact list, Task 5 Steps 2 and 3, Task 12 Step 4 and Task 13's first assertion. The superseded §6 paragraph is left standing in the spec above its erratum, so a reader of either document meets both. Section 7 is explicitly refused as Plan 4's. Section 9's error table: the empty-transcript row (`spec:419`) is Tasks 6 and 8, the duration and clip-count rows are unchanged from Plan 2's Arm A, the input-limit row rides Stage 1's existing path, the compatibility row is Task 9, the no-Profile row is Task 11, the non-finite-PCM row is the Audio Normalizer's. **One mapping Task 10 lands is not in that six-row table at all** — the reference-language pair (`SYNTH_ERR_INVALID_ARG` for a non-BCP-47 shape, `SYNTH_ERR_UNSUPPORTED_LANGUAGE` for an undeclared tag). It is taken from the in-tree precedent at `src/voice-profile.cpp:308-314` rather than from §9, and Task 10 Step 3 says so, so that the gap is a recorded decision instead of something a reviewer rediscovers.

**Where I deviated from the research's suggested seams, and why.** Two places. Its item 3 offered the graph as "possibly two tasks"; I kept it as one task (Task 4) with **two commits**, because the two halves are measured against different artifacts and want separate bisect points but share one set of tests and one file pair. And I split its item 1 into two tasks — the codec-encoder oracle (Task 1) and the ICL-prompt oracle (Task 2) — because the design's §6 names them as two of its four dump scripts, and because the prompt oracle is the one that makes the alignment branch observable at all, which this plan's validation strategy depends on. The net is 14 tasks, the top of the research's own estimate.

**Placeholder scan.** This plan prescribes **no numeric tolerance**. Every threshold in the tolerance file is measured by Task 12 from artifacts Tasks 1, 2, 5 and 7 produce, and the task that measures each is named where it appears. The conventions the graph is built from — causal padding, ELU placement, dilations, RVQ residual space and stage order, stride order — are deliberately left to Task 1 Step 1 to read off `modeling_mimi.py`, because each is a value that changes the answer silently and a guess here would be worse than a blank. The one geometry table this plan does state (Task 1 Step 2) is read from the package's own GGUF metadata and the checkpoint's `config.json`, and is presented as a reconciliation to be checked rather than a specification to be implemented.

**Unknowns carried as unknowns.** Four, each with the task that establishes it and each with a named branch for both answers rather than a preferred one:

1. **Which alignment arm each case takes** (Task 2 Step 3). The research offered a claim; it is unverifiable from the artifacts and is therefore not carried. No task is keyed to an assumed arm: Task 7 tests both arms unconditionally, and Task 12 keys its coverage to `alignment.json`.
2. **How far the port's continuous stages sit from the bf16-derived expectation**, per stage (Task 5 Step 2, committed by Task 12 Step 4). Whether tie flips occur is no longer among these unknowns — the spec's fourth erratum measured that they do, at 0.624% per decision from the codebook's dtype alone, which is what removed the codes from the gate. What replaces it is genuinely open in both directions: at or below the bf16 scale, Task 12 commits the measurement; above it, the plan stops and finds the defect. The code agreement rate is recorded alongside and decides neither branch.
3. **What causes the 9-frame render** (Task 12 Step 3), adjudicated with three named outcomes and an explicitly-labelled hypothesis that must not be promoted to a cause.
4. **Whether the reference-turn wrapper changes any tokenization** (Task 6 Step 1 test 4). Either an input exists on which wrapped and bare ids differ — pin it — or none does, in which case the test is deleted and the reason recorded. There is no third branch in which a test survives without an assertion.

None of the four is written anywhere in this plan as a decided fact.
