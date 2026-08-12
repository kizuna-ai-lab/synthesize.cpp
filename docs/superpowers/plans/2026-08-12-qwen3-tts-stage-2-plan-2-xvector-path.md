# Qwen3-TTS Stage 2, Plan 2: the x-vector Path — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reference audio in, cloned audio out, on CPU, in x-vector mode — a 128-bin log-mel front end, the 76-tensor ECAPA-TDNN graph producing a `[1024]` x-vector, Voice Profile preparation and its GGUF serialization, that x-vector substituted into the synthesis prompt, and the public seam opened so `SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO` is true rather than declared.

**Architecture:** Plan 1 catalogued the speaker encoder's 76 tensors and validated its eight metadata fields; it built no graph and kept no pointer. Plan 2 makes both live. The mel front end is new code — the digest establishes that no mel filterbank and no FFT exist anywhere in C++ in this tree (`.superpowers/sdd/plan2-interfaces.md` §4) — and is host DSP per the spec's §5. The ECAPA forward is a GGML graph over weights the catalog now keeps. Preparation and the v1 envelope follow OmniVoice's shape file-for-file (digest §2), including the in-envelope `kind` discriminator that lets Plan 3 add ICL fields without a schema version bump. The prompt change is one row of one F32 tensor: `enc_dim == hidden_size`, so the x-vector displaces the Preset Voice token embedding at the position it already occupies, and nothing downstream of the prompt changes.

**Tech Stack:** C++17 with GGML/GGUF; CTest under the `unit` and `integration` labels; Python 3.12 + torch + librosa in the locked `scripts/envs/qwen3-tts` environment for the oracle; `uv` for every Python entry point.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-11-qwen3-tts-stage-2-design.md`, sections 3–6 and 8, **including its three errata**. Carry-over ledger: `docs/superpowers/plans/2026-08-12-qwen3-tts-stage-2-plan-1-carryover.md`. Interface digest: `.superpowers/sdd/plan2-interfaces.md` — cited by section throughout; use it instead of re-deriving a signature.
- **The capability gate is `hparams.voice_mode == VoiceMode::ProfileSources`.** Not `has_preset_voice_catalog(hparams)`, which is true only for `VoiceMode::PresetCatalog` — CustomVoice, the variant with no speaker encoder. The carryover §1.4 records the correction and `tests/qwen3_tts_voice_required_test.cpp:155` already asserts the predicate is **false** for Base. Two source comments still carry the inverted instruction (`src/arch/qwen3-tts/weights.h:271`, `src/arch/qwen3-tts/weights.cpp:720`); Task 9 corrects both.
- **`reference_transcript` and `reference_language` report `SYNTH_REQUIREMENT_UNSUPPORTED`, and a request carrying a transcript is REJECTED.** Plan 2 implements one of two modes; reporting the transcript optional would invite a caller to pass one and quietly receive the weaker clone it did not ask for. Plan 3 flips both to `OPTIONAL` in the same change that lands ICL. The spec's §3 says `OPTIONAL` in its original prose and is refined by its own 2026-08-12 note — the refinement governs.
- **Profile Schema identity is `qwen3-tts-voice-clone`, version 1, with an in-envelope `synthesize.voice_profile.kind` discriminator.** Plan 2 emits exactly one kind, `x-vector`. Plan 3 adds `icl` and its extra fields without a version bump, and every Plan 2 profile stays loadable. OmniVoice's envelope already has this exact shape (digest §2) — follow it, including the "one schema, internal kind tag" reasoning at `src/arch/omnivoice/profile.h:196-208`.
- **`SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO` and `SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE` publish together, in one commit, on the day preparation works.** Never separately. `docs/c-interface.md` requires the second bit of any Model that can create a v1 Profile.
- **The CLI is out of scope.** `examples/cli/` has no Voice Profile support and no audio *reader* for any family — `grep -rn "voice_profile\|reference\|omnivoice\|family" examples/cli/` returns zero matches (digest §10a). Adding it is a cross-family slice, not a qwen3-tts increment. The spec's §8 row for Plan 2 lists "CLI and binding Adapters"; the CLI half of that row does not survive contact with the tree.
- **The Python binding needs no work — Task 12 verifies that claim rather than assuming it.** The digest (§10b) reports zero family-conditioned lines across `native_loader.c`, `synthesize_cpp/*.py`, `CMakeLists.txt` and `pyproject.toml`.
- **Unit gate baseline is 89/91.** `synthesize-python-api-wheel-test` and `synthesize-vits-python-unit` fail in a fresh worktree on gitignored VITS artifacts it never materializes. Any third failure is yours. Run `cmake --build build --target synthesize-check-unit` after every task.
- **The sanitizer gate runs for every task that touches inference code** — Tasks 2, 3, 4, 5, 7, 8, 9, 10, 11.
- **`scripts/ci/clang-format.sh --check-diff origin/main` only checks TRACKED files.** Run it *after* `git add`. Plan 1's Task 10 shipped 8 violations by running it before.
- **Nothing under `models/` or `build/` is ever committed.** Oracle artifacts, GGUFs and reports live there and stay ignored; the manifest and the tolerance file are the committed contract.
- **This plan touches no converter code.** If a task ever needs to, the published CustomVoice package must still convert to sha256 `01dfad52dd507c26a14d101c4247d375257aa63b07e62706ec3daa0a33ea515d` (2,274,117,280 bytes, 657 tensors) and the re-conversion proof goes in that task.
- **Every numeric tolerance this plan writes is provisional until measured, and the task that measures it is named where it appears.** Task 6 is the measuring task for the x-vector; Task 2 Step 7 for the mel.
- Base package facts that must hold: 894 tensors, `enc_dim` 1024 = talker `hidden_size`, mel `n_fft` 1024 / `hop` 256 / `win` 1024 / `fmin` 0 / `fmax` 12000 at 24 kHz, Profile Compatibility ID `34d4de22a329b6bc8347cb952b6fa16513320012628598ab59743679cc16806e`.
- Reference limits already in the package: `target_sample_rate` 24000, `target_channels` 1, `min_frames_per_clip` 24000, `max_frames_per_clip` 720000, `max_total_frames` 720000, `max_reference_count` 1.
- A Voice refusal carries the diagnostic code `synthesis.voice_unsupported`, never `synthesis.graph_failed` (carryover §4.9). The ABI defines exactly one voice-error status.
- Any performance number recorded anywhere in this plan names the build that produced it. The `dev-*` presets compile GGML at `-O2`.

## File Structure

| File | Status | Responsibility |
| --- | --- | --- |
| `scripts/dump_reference_qwen3_tts_speaker.py` | create | Oracle for the mel and every ECAPA intermediate — the stage-wise reference that does not exist today |
| `src/arch/qwen3-tts/mel.h` / `.cpp` | create | Host DSP: 24 kHz F32 PCM to `[mel_bins, frames]` log-mel, radix-2 FFT |
| `src/arch/qwen3-tts/speaker-encoder.h` / `.cpp` | create | GGML graph: ECAPA-TDNN over mel to a `[enc_dim]` embedding |
| `src/arch/qwen3-tts/speaker-encoder-host.h` / `.cpp` | create | Shape and finiteness validation, single-shot execution, silent-reference refusal |
| `src/arch/qwen3-tts/profile.h` / `.cpp` | create | `XVectorProfile` payload, preparation, the v1 GGUF envelope and its prescan whitelist |
| `src/arch/qwen3-tts/catalog.h` | modify | `SpeakerEncoderWeights`, `Res2NetBlockWeights`, `SpeakerEncoderBlockWeights`; hang them off `ModelWeights` |
| `src/arch/qwen3-tts/catalog.cpp` | modify | `resolve_speaker_encoder` keeps its pointers; `expected_tensor_count` unchanged in value |
| `src/arch/qwen3-tts/weights.h` / `.cpp` | modify | Power-of-two `n_fft` rule; the capability flip; two stale comments |
| `src/arch/qwen3-tts/qwen3-tts.h` | modify | `Model::prepare_x_vector`, `SynthesisRequest::x_vector` |
| `src/arch/qwen3-tts/model.cpp` | modify | `prepare_x_vector`; `resolve_voice` conditional on a Profile; `run_synthesis` carries the x-vector |
| `src/arch/qwen3-tts/talker-host.h` / `.cpp` | modify | The speaker slot becomes externally sourced without moving |
| `src/arch/qwen3-tts/talker.h` / `.cpp` | modify | `build_talker_prefill_input` substitutes one row |
| `src/voice-profile-handle.h` | modify | `ProfileFamilyTag::Qwen3TtsClone` |
| `src/voice-profile.cpp` | modify | Three arms: create-from-reference, load-from-memory, serialize |
| `src/synthesize.cpp` | modify | Consume a qwen3-tts Profile instead of refusing it |
| `tests/qwen3_tts_mel_test.cpp` | create | Mel rules and limits, no GGUF |
| `tests/qwen3_tts_speaker_encoder_test.cpp` | create | ECAPA graph, synthetic weights, no GGUF |
| `tests/qwen3_tts_profile_test.cpp` | create | Preparation rules and the envelope tamper matrix, no GGUF |
| `tests/qwen3_tts_prompt_slot_test.cpp` | create | Prompt layout and the substitution index |
| `tests/qwen3_tts_catalog_test.cpp` | modify | The resolver keeps every pointer it resolves |
| `tests/qwen3_tts_metadata_test.cpp` | modify | Non-power-of-two `n_fft` is refused at load |
| `tests/qwen3_tts_voice_required_test.cpp` | modify | Rewrite the zero-capability assertions to the published shape |
| `tests/qwen3_tts_base_load_real.cpp` | modify | Rewrite `check_capabilities` to the published shape |
| `tests/qwen3_tts_clone_real.cpp` | create | Integration: reference audio in, cloned audio out, against the real package |
| `tests/CMakeLists.txt` | modify | Register five new tests |
| `tests/tolerances/qwen3-tts.json` | modify | Measured `speaker.x_vector` probe; retire the speaker-encoder placeholder |
| `scripts/validate-qwen3-tts-replay.py` | modify | Compare `speaker.x_vector` — deliberately NOT a new validator basename |
| `docs/porting/families/qwen3-tts.md` | modify | Record Stage 2 Plan 2 |

---

### Task 1: An oracle for the mel and the ECAPA intermediates

**Files:**
- Create: `scripts/dump_reference_qwen3_tts_speaker.py`

**Interfaces:**
- Consumes: `models/qwen3-tts-12hz-0-6b-base/`, the pinned reference clip in `models/qwen3-tts-reference-audio/`.
- Produces: under `build/goldens/qwen3-tts/qwen3-tts-12hz-0-6b-base/<case-id>/speaker/` — `mel.f32` `[mel_bins, frames]`, `blocks0.f32`, `block1.f32`, `block2.f32`, `block3.f32`, `mfa.f32`, `asp.f32`, `x_vector.f32`, plus `conventions.json`. Tasks 2, 4 and 5 compare against these.

**Why this is first.** The eight existing `scripts/dump_reference_qwen3_tts_*.py` scripts dump **only** the finished `[1024]` x-vector and nothing between the waveform and it — no mel, no ECAPA intermediate, no mel parameter, in any of them (digest §7a; `mel`, `ecapa`, `n_fft`, `hop`, `win`, `fbank`, `kaldi`, `spectrogram` all return zero grep hits). Until this script exists, a wrong mel and a wrong ECAPA can cancel to a plausible x-vector and nothing will say which stage moved. The spec's §6 already lists "mel plus speaker encoder" as one of four new dump scripts; this is that one.

- [ ] **Step 1: Read the mel conventions off upstream rather than transcribing them**

The five numbers in the package (`mel_bins` 128, `n_fft` 1024, `hop_length` 256, `win_length` 1024, `fmin` 0, `fmax` 12000) do not determine a mel spectrogram. Six more conventions do, and every one of them silently changes the answer. Open the pinned upstream source at `modeling_qwen3_tts.py:1941` (repository `QwenLM/Qwen3-TTS` at revision `022e286b98fbec7e1e916cb940cdf532cd9f488e`, the revision the Base manifest pins) and record, verbatim with its line number:

1. mel scale formula — Slaney (`librosa`'s `htk=False`) or HTK (`2595 log10(1 + f/700)`);
2. filterbank normalization — Slaney area normalization, `norm=None`, or per-filter peak;
3. spectrum magnitude or power (`|X|` or `|X|²`);
4. log base and floor — natural log with `clamp(min=1e-5)`, `log10`, dB with `amin`/`top_db`, or `log(x + eps)`;
5. window — Hann periodic or symmetric, and whether `win_length < n_fft` is zero-padded centred;
6. padding — `center=True` with reflect padding, or no centring; and therefore whether `frames == 1 + len(pcm) // hop`.

Write all six into `conventions.json` beside the artifacts, each with its upstream `file:line`. This file is what Task 2 implements against; a convention that is not in it was guessed.

- [ ] **Step 2: Write the runner**

Model it on `scripts/dump_reference_qwen3_tts_base.py` — same `--weights-dir` / `--ref-audio` / `--manifest` / `--case` / `--output-root` argument shape, the same raw-`tobytes()` `write_f32` (no `.npy` header, native little-endian), the same atomic writes. The Base-specific body hooks the encoder's submodules, which is the only way to see inside a forward that upstream never returns:

```python
SPEAKER_TAPS = {
    "speaker.blocks0": "speaker_encoder.blocks.0",
    "speaker.block1":  "speaker_encoder.blocks.1",
    "speaker.block2":  "speaker_encoder.blocks.2",
    "speaker.block3":  "speaker_encoder.blocks.3",
    "speaker.mfa":     "speaker_encoder.mfa",
    "speaker.asp":     "speaker_encoder.asp",
}


def install_taps(model, captured):
    """Capture each ECAPA stage's output on the way past.

    Upstream returns only the finished embedding, so a stage-wise comparison
    has no reference at all without these -- which is why the port's mel and
    its ECAPA forward could each be wrong and still agree at the output. A
    hook is a read, not a change: nothing here alters the forward.
    """
    handles = []
    modules = dict(model.named_modules())
    for name, path in SPEAKER_TAPS.items():
        module = modules.get(path)
        if module is None:
            raise SystemExit(f"upstream has no module {path!r}; the taps are stale")
        handles.append(module.register_forward_hook(
            lambda _m, _i, out, key=name: captured.__setitem__(key, to_numpy(out, torch.float32))))
    return handles
```

The mel itself is captured the same way, off whichever module or function `conventions.json` names as producing it; if upstream computes it inline rather than in a module, wrap that function instead and say so in `conventions.json`.

Dump the mel and the taps **before** the x-vector, so a failure in the pooling still leaves the front end's reference on disk.

- [ ] **Step 3: Run it on the one materialized x-vector case**

```bash
uv run --project scripts/envs/qwen3-tts --locked python scripts/dump_reference_qwen3_tts_speaker.py \
  --weights-dir models/qwen3-tts-12hz-0-6b-base \
  --manifest tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-base.manifest.json \
  --case base-xvector-en
```

Expected: `build/goldens/qwen3-tts/qwen3-tts-12hz-0-6b-base/base-xvector-en/speaker/x_vector.f32` is exactly 4096 bytes and **byte-identical to the one Plan 1 already dumped there**. Verify it:

```bash
sha256sum build/goldens/qwen3-tts/qwen3-tts-12hz-0-6b-base/base-xvector-en/speaker/x_vector.f32
```

If the digest differs from what Plan 1's run produced, the hooks changed the forward or the clip is being loaded differently — stop and find out which, because every tolerance measured downstream would inherit the difference.

- [ ] **Step 4: Materialize the second x-vector case**

`base-xvector-zh` is the only other Plan-2-runnable case in the manifest and is **not** on disk (digest §7h). Run the same command with `--case base-xvector-zh`. Expected: a second complete artifact set. Task 6 measures over both.

- [ ] **Step 5: Confirm the frame arithmetic against the clip you actually have**

```bash
uv run --project scripts/envs/qwen3-tts --locked python -c "
import numpy as np, json, pathlib
root = pathlib.Path('build/goldens/qwen3-tts/qwen3-tts-12hz-0-6b-base/base-xvector-en/speaker')
mel = np.fromfile(root / 'mel.f32', dtype=np.float32)
conv = json.loads((root / 'conventions.json').read_text())
print('mel elements', mel.size, 'bins', conv['mel_bins'], 'frames', mel.size // conv['mel_bins'])
print('predicted frames', 1 + 193920 // 256, 'centered' , conv['centered'])
"
```

Expected: `mel.size % mel_bins == 0`, and the frame count agrees with the centring convention recorded in Step 1 (`1 + 193920 // 256 == 758` if centred; `(193920 - 1024) // 256 + 1 == 754` if not). Whichever it is, that is the number Task 2 must reproduce — write the observed one into `conventions.json` as `frames_observed` so Task 2 has a target rather than a formula it chose.

- [ ] **Step 6: Commit**

```bash
git add scripts/dump_reference_qwen3_tts_speaker.py
git commit -m "qwen3-tts: dump the mel and every ECAPA stage, not just the x-vector"
```

Nothing under `build/` is added. The artifacts are the ignored payload; the script and the manifest are the committed contract.

---

### Task 2: The mel front end

**Files:**
- Create: `src/arch/qwen3-tts/mel.h`, `src/arch/qwen3-tts/mel.cpp`
- Create: `tests/qwen3_tts_mel_test.cpp`
- Modify: `src/arch/qwen3-tts/weights.cpp`, `tests/qwen3_tts_metadata_test.cpp`, `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `SpeakerEncoderParams` (`src/arch/qwen3-tts/weights.h:29-38`) — **all eight fields**, six of which have had no consumer at all since Plan 1 (digest Q1: "Nothing reads `n_fft`, `hop_length`, `win_length`, `fmin`, `fmax`, or `sample_rate` outside the validator that wrote them").
- Produces: `synth::qwen3tts::MelSpectrogram` and `compute_log_mel(...)`. Task 5 calls it.

This is new code. Kokoro's two transforms are naive O(n_fft²) DFTs, family-private, with no mel filterbank, no log, no `win_length` and no PCM entry point (digest §4a–4e); at `n_fft` 1024 the naive form is roughly 100× the work of a radix-2 butterfly per frame. The spec's §5 already ruled that Kokoro's code is not promoted for this.

- [ ] **Step 1: Write the failing test**

`tests/qwen3_tts_mel_test.cpp`, no GGUF and no Model — the governing rule is stated in-tree at `tests/qwen3_tts_voice_required_test.cpp:25-30`. Every case below is a rule with an input that fails it.

```cpp
#include "arch/qwen3-tts/mel.h"
#include "arch/qwen3-tts/weights.h"
#include "test-assert.h"

#include <cmath>
#include <vector>

namespace {

synth::qwen3tts::SpeakerEncoderParams production_params() {
    synth::qwen3tts::SpeakerEncoderParams p;
    p.enc_dim     = 1024;
    p.sample_rate = 24000;
    p.mel_bins    = 128;
    p.n_fft       = 1024;
    p.hop_length  = 256;
    p.win_length  = 1024;
    p.fmin        = 0.0f;
    p.fmax        = 12000.0f;
    return p;
}

// Every one of the eight fields reaches the front end. Six of them had no
// consumer at all after Plan 1 -- read_speaker_encoder validated them and
// nothing else looked -- so this is the test that makes them live.
int test_the_frame_count_follows_hop_length() {
    const synth::qwen3tts::SpeakerEncoderParams p = production_params();
    std::vector<float>                          pcm(24000, 0.5f);
    synth::qwen3tts::MelSpectrogram             mel;
    SYNTH_TEST_CHECK(synth::qwen3tts::compute_log_mel(p, pcm, mel) == SYNTH_OK);
    SYNTH_TEST_CHECK(mel.bins == 128);
    SYNTH_TEST_CHECK(mel.frames == kExpectedFramesForOneSecond);  // from Task 1's conventions.json
    SYNTH_TEST_CHECK(mel.values.size() == size_t(mel.bins) * size_t(mel.frames));
    return 0;
}

// A different hop must produce a different frame count, or hop_length is
// being ignored and the production value is right by coincidence.
int test_a_different_hop_produces_a_different_frame_count() {
    synth::qwen3tts::SpeakerEncoderParams p = production_params();
    p.hop_length                            = 512;
    std::vector<float>              pcm(24000, 0.5f);
    synth::qwen3tts::MelSpectrogram mel;
    SYNTH_TEST_CHECK(synth::qwen3tts::compute_log_mel(p, pcm, mel) == SYNTH_OK);
    SYNTH_TEST_CHECK(mel.frames == kExpectedFramesForOneSecondAtHop512);
    return 0;
}

// fmin/fmax select which filters exist at all. Narrowing the band must move
// the output, or the filterbank is being built from the sample rate alone.
int test_the_band_limits_change_the_filterbank() {
    const synth::qwen3tts::SpeakerEncoderParams wide = production_params();
    synth::qwen3tts::SpeakerEncoderParams       narrow = wide;
    narrow.fmin                                        = 300.0f;
    narrow.fmax                                        = 6000.0f;

    std::vector<float> pcm(24000);
    for (size_t i = 0; i < pcm.size(); ++i) {
        pcm[i] = std::sin(float(i) * 0.05f) * 0.4f;
    }
    synth::qwen3tts::MelSpectrogram a;
    synth::qwen3tts::MelSpectrogram b;
    SYNTH_TEST_CHECK(synth::qwen3tts::compute_log_mel(wide, pcm, a) == SYNTH_OK);
    SYNTH_TEST_CHECK(synth::qwen3tts::compute_log_mel(narrow, pcm, b) == SYNTH_OK);
    SYNTH_TEST_CHECK(a.values != b.values);
    return 0;
}

// win_length shorter than n_fft is a zero-padded window, not a shorter
// transform. Kokoro's transforms have no such concept, which is exactly why
// this one is new code.
int test_a_short_window_is_zero_padded_not_truncated() {
    synth::qwen3tts::SpeakerEncoderParams p = production_params();
    p.win_length                            = 512;
    std::vector<float>              pcm(24000, 0.25f);
    synth::qwen3tts::MelSpectrogram mel;
    SYNTH_TEST_CHECK(synth::qwen3tts::compute_log_mel(p, pcm, mel) == SYNTH_OK);
    SYNTH_TEST_CHECK(mel.frames == kExpectedFramesForOneSecond);  // unchanged: win_length is not the hop
    return 0;
}

// Digital silence must not produce -inf or NaN: the log floor is what stops
// it, and a floor nobody tested is a floor nobody has.
int test_digital_silence_stays_finite() {
    const synth::qwen3tts::SpeakerEncoderParams p = production_params();
    std::vector<float>                          pcm(24000, 0.0f);
    synth::qwen3tts::MelSpectrogram             mel;
    SYNTH_TEST_CHECK(synth::qwen3tts::compute_log_mel(p, pcm, mel) == SYNTH_OK);
    for (float value : mel.values) {
        SYNTH_TEST_CHECK(std::isfinite(value));
    }
    return 0;
}

int test_a_non_finite_sample_is_refused() {
    const synth::qwen3tts::SpeakerEncoderParams p = production_params();
    std::vector<float>                          pcm(24000, 0.1f);
    pcm[7]                                        = std::nanf("");
    synth::qwen3tts::MelSpectrogram mel;
    SYNTH_TEST_CHECK(synth::qwen3tts::compute_log_mel(p, pcm, mel) == SYNTH_ERR_INVALID_ARG);
    return 0;
}

int test_a_clip_shorter_than_one_frame_is_refused() {
    const synth::qwen3tts::SpeakerEncoderParams p = production_params();
    std::vector<float>                          pcm(16, 0.1f);
    synth::qwen3tts::MelSpectrogram             mel;
    SYNTH_TEST_CHECK(synth::qwen3tts::compute_log_mel(p, pcm, mel) == SYNTH_ERR_INVALID_ARG);
    return 0;
}

// The radix-2 transform is the whole reason n_fft must be a power of two.
// The load-time rule below is what keeps this unreachable in practice; this
// is the belt to that braces.
int test_a_non_power_of_two_n_fft_is_refused() {
    synth::qwen3tts::SpeakerEncoderParams p = production_params();
    p.n_fft                                 = 1000;
    std::vector<float>              pcm(24000, 0.1f);
    synth::qwen3tts::MelSpectrogram mel;
    SYNTH_TEST_CHECK(synth::qwen3tts::compute_log_mel(p, pcm, mel) == SYNTH_ERR_UNSUPPORTED_INPUT);
    return 0;
}

}  // namespace
```

`kExpectedFramesForOneSecond` and `kExpectedFramesForOneSecondAtHop512` are **not** in this plan: take them from Task 1's `conventions.json` (`frames_for_one_second` and `frames_for_one_second_at_hop_512`), so the port's frame arithmetic is pinned to what upstream actually did rather than to a formula this plan chose.

**Measured 2026-08-12 by Task 1, and this is why the plan refused to guess.**
The values are **93** and **46**, with **757** for the 193,920-sample reference
clip. Neither of the two formulas this plan floated produces them: upstream
neither centres nor leaves uncentred in the ordinary sense. It reflect-pads
`(n_fft - hop) // 2` samples per side by hand and then calls
`torch.stft(center=False)` — a third convention, giving
`frames = 1 + (len + 2*((n_fft - hop)//2) - n_fft) // hop`. A centred formula
predicts 758 and a plainly uncentred one 754. Task 1's reviewer re-derived all
three numbers from upstream's source and re-ran `mel_spectrogram` to observe
them. Use `conventions.json`; do not re-derive.

- [ ] **Step 2: Run it and watch it fail**

```bash
cmake --build build --target synthesize-qwen3-tts-mel-test
```
Expected: FAIL — `fatal error: arch/qwen3-tts/mel.h: No such file or directory`, and CMake does not know the target either. Add the registration in Step 3 and re-run to get the real compile error (`compute_log_mel` undeclared) before implementing.

- [ ] **Step 3: Register the test**

In `tests/CMakeLists.txt`, beside line 173:

```cmake
synth_add_unit_test(synthesize-qwen3-tts-mel-test qwen3_tts_mel_test.cpp)
```

`synth_add_unit_test` (`tests/CMakeLists.txt:26-33`) is the required helper and works because this test takes no argv.

- [ ] **Step 4: Implement the header**

```cpp
// src/arch/qwen3-tts/mel.h
#pragma once

#include "synthesize.h"

#include <cstdint>
#include <vector>

namespace synth::qwen3tts {

struct SpeakerEncoderParams;

// A log-mel spectrogram, mel-major: `values[frame * bins + bin]`. That is the
// layout the ECAPA graph reads directly as a ggml [bins, frames] F32 tensor,
// so nothing transposes between here and the graph.
struct MelSpectrogram {
    std::vector<float> values;
    uint32_t           bins   = 0;
    uint64_t           frames = 0;
};

// 24 kHz mono F32 PCM to log-mel at the package's own pinned parameters.
//
// Host DSP rather than a GGML graph, per the Stage 2 design's section 5: it
// runs once per enrollment, its cost is negligible beside the ECAPA forward,
// and keeping it on the host avoids a backend-placement question for a stage
// with no downstream shape dependence.
//
// Every conversion this performs -- mel scale, filterbank normalization,
// magnitude-vs-power, log floor, window shape, centring -- is transcribed
// from the pinned upstream source and recorded in the Task 1 oracle's
// conventions.json. None of the six is derivable from the package's eight
// metadata fields, and each one silently changes the answer rather than
// failing, which is why they are pinned by a dump rather than by a comment.
//
// Errors:
//   SYNTH_ERR_UNSUPPORTED_INPUT -- n_fft is not a power of two (the transform
//                                  is radix-2), or a parameter is zero.
//   SYNTH_ERR_INVALID_ARG       -- a non-finite sample, or a clip too short to
//                                  produce a single frame.
synth_status_t compute_log_mel(const SpeakerEncoderParams & params,
                               const std::vector<float> &   pcm,
                               MelSpectrogram &             output);

}  // namespace synth::qwen3tts
```

- [ ] **Step 5: Implement the transform**

In `mel.cpp`, in this order: parameter validation, an iterative radix-2 FFT, a Hann window built once, a mel filterbank built once, then the frame loop.

```cpp
namespace {

// Iterative radix-2 Cooley-Tukey, in place, on split real/imaginary buffers.
// `n` is a power of two, checked by the caller. Kokoro's two transforms are
// naive O(n^2) DFTs (src/arch/kokoro/source.cpp:163-179,
// src/arch/kokoro/decoder-host.cpp:77); at n_fft 1024 that is about 100x the
// work per frame, which is why this is written rather than reused -- along
// with the fact that neither of them has a mel filterbank, a log, a
// win_length, or a PCM entry point.
void fft_in_place(std::vector<float> & re, std::vector<float> & im) {
    const size_t n = re.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const double angle = -2.0 * M_PI / double(len);
        for (size_t start = 0; start < n; start += len) {
            for (size_t k = 0; k < len / 2; ++k) {
                const double theta = angle * double(k);
                const double wr    = std::cos(theta);
                const double wi    = std::sin(theta);
                const size_t a     = start + k;
                const size_t b     = a + len / 2;
                const double xr    = re[b] * wr - im[b] * wi;
                const double xi    = re[b] * wi + im[b] * wr;
                re[b]              = float(re[a] - xr);
                im[b]              = float(im[a] - xi);
                re[a]              = float(re[a] + xr);
                im[a]              = float(im[a] + xi);
            }
        }
    }
}

// Hz to mel and back. WHICH of the two formulas applies is Task 1's
// conventions.json entry `mel_scale`, read off the pinned upstream source --
// Slaney's piecewise-linear-then-log scale and HTK's single log differ by
// several bins at 12 kHz, and both produce a plausible-looking spectrogram.
double hz_to_mel(double hz);
double mel_to_hz(double mel);

// Triangular filters over the FFT bins. `normalization` is conventions.json's
// `filterbank_norm`: Slaney area normalization divides each filter by the
// width of its own band, which changes every value and no shape.
std::vector<float> build_mel_filterbank(const SpeakerEncoderParams & params);

}  // namespace
```

The frame loop: window the frame (zero-padding a `win_length < n_fft` window centred in the `n_fft` buffer), FFT, take magnitude or power per `conventions.json`, apply the filterbank, apply the log with its recorded floor. Reject a non-finite input sample before any of it — the Audio Normalizer already refuses non-finite PCM, but this function is also called directly by tests and by Plan 3.

- [ ] **Step 6: Add the load-time power-of-two rule**

The radix-2 transform makes `n_fft` a load-time contract, not a runtime surprise. In `read_speaker_encoder` (`src/arch/qwen3-tts/weights.cpp:531-`), beside the existing non-zero checks at line 549:

```cpp
    if ((params.n_fft & (params.n_fft - 1)) != 0) {
        std::fprintf(stderr, "qwen3-tts: speaker encoder n_fft %u is not a power of two\n", params.n_fft);
        return false;
    }
```

And the covering case in `tests/qwen3_tts_metadata_test.cpp`, using the existing `expect_base_rejected` helper (`:238`):

```cpp
    failures += expect_base_rejected(
        [](gguf_context * g) { gguf_set_val_u32(g, "synthesize.qwen3-tts.speaker_encoder.n_fft", 1000); },
        "non-power-of-two n_fft");
```

- [ ] **Step 7: Measure the mel against the oracle**

Extend `scripts/validate-qwen3-tts-replay.py` with a `--compare-mel` mode that runs `compute_log_mel` through a tiny driver and reports max-abs and cosine against `speaker/mel.f32`. Run it on `base-xvector-en`.

Expected: max-abs deviation on the log-mel below `1e-3`. **That number is a provisional placeholder invented for this plan — no mel measurement exists anywhere in this tree.** Record the observed deviation in the commit message; if it is worse than `1e-2`, a convention from Task 1 is wrong and the fix is in `conventions.json`, not in the tolerance. Task 6 is where the surviving number is committed.

- [ ] **Step 8: Run the unit gate and the sanitizer gate**

```bash
cmake --build build --target synthesize-check-unit
cmake --build build-sanitize --target synthesize-check-unit
```
Expected: 89/91 in both, with only `synthesize-python-api-wheel-test` and `synthesize-vits-python-unit` failing, and no sanitizer diagnostics.

- [ ] **Step 9: Commit**

```bash
scripts/ci/clang-format.sh --fix
git add src/arch/qwen3-tts/mel.h src/arch/qwen3-tts/mel.cpp tests/qwen3_tts_mel_test.cpp \
        tests/CMakeLists.txt src/arch/qwen3-tts/weights.cpp tests/qwen3_tts_metadata_test.cpp \
        scripts/validate-qwen3-tts-replay.py
scripts/ci/clang-format.sh --check-diff origin/main
git commit -m "qwen3-tts: a 128-bin log-mel front end, and the six metadata fields that now feed it"
```

---

### Task 3: The speaker-encoder resolver keeps its pointers

**Files:**
- Modify: `src/arch/qwen3-tts/catalog.h`, `src/arch/qwen3-tts/catalog.cpp`
- Modify: `tests/qwen3_tts_catalog_test.cpp`

**Interfaces:**
- Produces: `SpeakerEncoderWeights`, hung off `ModelWeights`. Task 4's graph reads it.
- `build_model_weights`'s signature does **not** change (`src/arch/qwen3-tts/catalog.h:155-158`) — `ModelWeights &` is already the sink.

**The trap this task exists to avoid**, carryover §2.1 and digest §5: `resolve_speaker_encoder` today has ten `resolver.conv(...)` calls (`catalog.cpp:464, 468, 474, 477, 478, 479, 482, 487, 488, 489`) that all write into one reused local `Conv1dWeights scratch;` at `catalog.cpp:463`, which nothing ever reads. There is no `SpeakerEncoderWeights` type and `ModelWeights` (`catalog.h:128-132`) has no slot. **Copying `resolve_codec`'s pattern onto these while leaving `scratch` in place still compiles and still passes the catalog tests**, because the sweep only checks that every name was *resolved*, never that a pointer was *kept*. Change the signature; do not add to it.

- [ ] **Step 1: Write the failing test**

In `tests/qwen3_tts_catalog_test.cpp`, beside the existing base cases. This is the test that no amount of "it resolved" can satisfy:

```cpp
// Plan 1 resolved these 76 names into a scratch struct nobody read -- there
// was no graph to hold them for. A resolver that resolves and discards passes
// every sweep check there is, so the only thing that can catch the regression
// is asserting the pointers arrived. Every one of them, by name: a loop that
// checked "at least one is non-null" would pass with 75 dropped.
int test_the_speaker_encoder_resolver_keeps_every_pointer() {
    synth::qwen3tts::HParams hparams = base_hparams();
    Context                  context = make_context();
    populate(context.get(), base_entries(hparams));

    synth::qwen3tts::ModelWeights weights;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_model_weights(context.get(), nullptr, hparams, weights) == SYNTH_OK);

    const synth::qwen3tts::SpeakerEncoderWeights & speaker = weights.speaker_encoder;
    SYNTH_TEST_CHECK(speaker.stem.weight != nullptr && speaker.stem.bias != nullptr);
    SYNTH_TEST_CHECK(speaker.blocks.size() == 3);
    for (const synth::qwen3tts::SpeakerEncoderBlockWeights & block : speaker.blocks) {
        SYNTH_TEST_CHECK(block.tdnn1.weight != nullptr && block.tdnn1.bias != nullptr);
        SYNTH_TEST_CHECK(block.res2net.size() == 7);
        for (const synth::qwen3tts::Conv1dWeights & conv : block.res2net) {
            SYNTH_TEST_CHECK(conv.weight != nullptr && conv.bias != nullptr);
        }
        SYNTH_TEST_CHECK(block.se1.weight != nullptr && block.se2.weight != nullptr);
        SYNTH_TEST_CHECK(block.tdnn2.weight != nullptr && block.tdnn2.bias != nullptr);
    }
    SYNTH_TEST_CHECK(speaker.mfa.weight != nullptr && speaker.asp_tdnn.weight != nullptr);
    SYNTH_TEST_CHECK(speaker.asp.weight != nullptr && speaker.fc.weight != nullptr);
    return 0;
}

// The res2net convolutions differ only by index. A resolver that wrote all
// seven into one slot would leave six aliases of the seventh, which the
// null checks above cannot see.
int test_the_res2net_convolutions_are_seven_distinct_tensors() {
    synth::qwen3tts::HParams hparams = base_hparams();
    Context                  context = make_context();
    populate(context.get(), base_entries(hparams));

    synth::qwen3tts::ModelWeights weights;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_model_weights(context.get(), nullptr, hparams, weights) == SYNTH_OK);

    std::set<const ggml_tensor *> seen;
    for (const synth::qwen3tts::SpeakerEncoderBlockWeights & block : weights.speaker_encoder.blocks) {
        for (const synth::qwen3tts::Conv1dWeights & conv : block.res2net) {
            SYNTH_TEST_CHECK(seen.insert(conv.weight).second);
        }
    }
    SYNTH_TEST_CHECK(seen.size() == 21);
    return 0;
}

// A CustomVoice package has no speaker encoder, and the struct must say so
// rather than carrying stale pointers from a previous resolve.
int test_customvoice_leaves_the_speaker_encoder_empty() {
    synth::qwen3tts::HParams hparams = custom_voice_hparams();
    Context                  context = make_context();
    populate(context.get(), custom_voice_entries(hparams));

    synth::qwen3tts::ModelWeights weights;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_model_weights(context.get(), nullptr, hparams, weights) == SYNTH_OK);
    SYNTH_TEST_CHECK(weights.speaker_encoder.blocks.empty());
    SYNTH_TEST_CHECK(weights.speaker_encoder.fc.weight == nullptr);
    return 0;
}
```

- [ ] **Step 2: Run it and watch it fail**

```bash
cmake --build build --target synthesize-qwen3-tts-catalog-test
```
Expected: FAIL — `'struct synth::qwen3tts::ModelWeights' has no member named 'speaker_encoder'`, and `SpeakerEncoderWeights` is not a member of the namespace.

- [ ] **Step 3: Add the types**

In `catalog.h`, beside `CodecDecoderWeights`:

```cpp
// One SE-Res2Net block: a TDNN 1x1 in, a scale-8 res2net body whose eighth
// split passes through unconvolved (hence seven convolutions, not eight), a
// squeeze-excite bottleneck pair, and a TDNN 1x1 out.
struct SpeakerEncoderBlockWeights {
    Conv1dWeights              tdnn1;
    std::vector<Conv1dWeights> res2net;
    Conv1dWeights              se1;
    Conv1dWeights              se2;
    Conv1dWeights              tdnn2;
};

// The ECAPA-TDNN speaker encoder Base variants carry. Plan 1 resolved these
// 76 names into a discarded scratch struct because no graph could reach them;
// Plan 2's speaker-encoder.cpp is that graph, so the pointers are kept.
struct SpeakerEncoderWeights {
    Conv1dWeights                           stem;      // blocks.0.conv, kernel 5
    std::vector<SpeakerEncoderBlockWeights> blocks;    // three
    Conv1dWeights                           mfa;       // multi-layer feature aggregation
    Conv1dWeights                           asp_tdnn;  // attention bottleneck
    Conv1dWeights                           asp;       // attention logits
    Conv1dWeights                           fc;        // pooled statistics -> enc_dim
};
```

**Erratum, 2026-08-12 — the comment on lines 604–605 says "eighth split passes
through unconvolved", but the first split is the one that passes through
unconvolved and leads the concatenation.** This is established by
`modeling_qwen3_tts.py:115-126` and already corrected in
`src/arch/qwen3-tts/catalog.h` and `catalog.cpp`.

and add `SpeakerEncoderWeights speaker_encoder;` to `ModelWeights`.

- [ ] **Step 4: Change the resolver's signature and delete the scratch**

```cpp
bool resolve_speaker_encoder(Resolver &                   resolver,
                             const SpeakerEncoderParams & params,
                             SpeakerEncoderWeights &      target) {
    constexpr int64_t kRes2NetWidth = kSpeakerEncoderChannels / kSpeakerEncoderRes2NetScale;
    const int64_t     mfa_channels  = 3 * kSpeakerEncoderChannels;

    resolver.conv("speaker_encoder.blocks.0.conv", 5, int64_t(params.mel_bins), kSpeakerEncoderChannels, target.stem);

    target.blocks.assign(kSpeakerEncoderBlockCount, SpeakerEncoderBlockWeights{});
    for (uint32_t block = 1; block <= kSpeakerEncoderBlockCount; ++block) {
        SpeakerEncoderBlockWeights & into = target.blocks[block - 1];
        const std::string            base = index_of("speaker_encoder.blocks.", block, ".");
        resolver.conv(base + "tdnn1.conv", 1, kSpeakerEncoderChannels, kSpeakerEncoderChannels, into.tdnn1);
        into.res2net.assign(kSpeakerEncoderRes2NetScale - 1, Conv1dWeights{});
        for (int64_t sub = 0; sub < kSpeakerEncoderRes2NetScale - 1; ++sub) {
            resolver.conv(index_of(base + "res2net_block.blocks.", sub, ".conv"), 3, kRes2NetWidth, kRes2NetWidth,
                          into.res2net[size_t(sub)]);
        }
        resolver.conv(base + "se_block.conv1", 1, kSpeakerEncoderChannels, kSpeakerEncoderSeChannels, into.se1);
        resolver.conv(base + "se_block.conv2", 1, kSpeakerEncoderSeChannels, kSpeakerEncoderChannels, into.se2);
        resolver.conv(base + "tdnn2.conv", 1, kSpeakerEncoderChannels, kSpeakerEncoderChannels, into.tdnn2);
    }

    resolver.conv("speaker_encoder.mfa.conv", 1, mfa_channels, mfa_channels, target.mfa);
    resolver.conv("speaker_encoder.asp.tdnn.conv", 1, 3 * mfa_channels, kSpeakerEncoderAttentionChannels,
                  target.asp_tdnn);
    resolver.conv("speaker_encoder.asp.conv", 1, kSpeakerEncoderAttentionChannels, mfa_channels, target.asp);
    resolver.conv("speaker_encoder.fc", 1, 2 * mfa_channels, int64_t(params.enc_dim), target.fc);
    return resolver.ok();
}
```

`Conv1dWeights scratch;` at the old line 463 is **deleted**, not left unused — an unused local is what the next plan copies. The single call site is `catalog.cpp:641-644`; it becomes:

```cpp
    if (hparams.has_speaker_encoder &&
        (!resolve_speaker_encoder(resolver, hparams.speaker_encoder, weights.speaker_encoder) ||
         !resolve_codec_encoder(resolver, hparams))) {
        return SYNTH_ERR_GGUF;
    }
```

`resolve_codec_encoder` keeps its discarding shape — Plan 3 owns it — but note in a comment beside it that it is still the Plan 1 pattern, so Plan 3 does not read this task as evidence that both were converted.

- [ ] **Step 5: Confirm `expected_tensor_count` is unchanged in value**

Its speaker-encoder arithmetic (`catalog.cpp:588-613`) reuses the same constants and must still produce 894 for the Base hparams. Add the assertion beside the existing count pins at `tests/qwen3_tts_catalog_test.cpp:680-686`:

```cpp
    SYNTH_TEST_CHECK(synth::qwen3tts::expected_tensor_count(base_hparams()) ==
                     synth::qwen3tts::expected_tensor_count(custom_voice_hparams()) + 76 + 161);
```

Refactoring a resolver must not move a count; if it does, a name was dropped or added.

- [ ] **Step 6: Update the catalog header's stale claim**

`catalog.h:142-147` states outright that the two regions have no graph builder and are "resolved here purely to bring their names into the sweep". Half of that is now false. Rewrite it to say: the speaker encoder is resolved into `ModelWeights::speaker_encoder` and read by `speaker-encoder.cpp`; the codec encoder is still sweep-only until Plan 3.

- [ ] **Step 7: Run the test, the unit gate and the sanitizer gate**

```bash
ctest --test-dir build -R '^synthesize-qwen3-tts-catalog-test$' --output-on-failure
cmake --build build --target synthesize-check-unit
cmake --build build-sanitize --target synthesize-check-unit
```
Expected: PASS, then 89/91 in both.

- [ ] **Step 8: Commit**

```bash
scripts/ci/clang-format.sh --fix
git add src/arch/qwen3-tts/catalog.h src/arch/qwen3-tts/catalog.cpp tests/qwen3_tts_catalog_test.cpp
scripts/ci/clang-format.sh --check-diff origin/main
git commit -m "qwen3-tts: the speaker encoder resolver keeps the pointers it resolves"
```

---

### Task 4: The ECAPA-TDNN graph

**Files:**
- Create: `src/arch/qwen3-tts/speaker-encoder.h`, `src/arch/qwen3-tts/speaker-encoder.cpp`
- Create: `tests/qwen3_tts_speaker_encoder_test.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `SpeakerEncoderWeights` from Task 3, a `[mel_bins, frames]` F32 tensor.
- Produces: `build_speaker_encoder(...) -> ggml_tensor *` of shape `[enc_dim]`. Task 5's host wrapper runs it.

- [ ] **Step 1: Read the topology off upstream, not off this plan**

The 76 tensors are convolutions and biases only — there are no normalization tensors in the checkpoint, which is a fact about the architecture, not an omission. Before writing the graph, read the pinned upstream speaker-encoder class and record, with `file:line`, into the same `conventions.json` Task 1 wrote:

1. the **dilations** on `blocks.1..3`'s res2net convolutions (SpeechBrain's reference ECAPA uses 2, 3, 4; the package declares none and the resolver does not see them, because a dilation is not a weight);
2. the activation after each convolution;
3. whether the res2net splits accumulate (`y_i = conv(x_i + y_{i-1})`) or are independent;
4. the SE block's two activations and how the excitation is applied;
5. the attentive-statistics-pooling detail: whether the attention input is `[x, mean, std]` broadcast over time (the `3 * mfa_channels` = 4608 input extent of `asp.tdnn.conv` says it is), the activation before `asp.conv`, and whether the softmax is over time;
6. whether `fc` sees `[mean, std]` concatenated (the `2 * mfa_channels` = 3072 input extent says it does).

A dilation guessed wrong produces a finite, plausible x-vector with a cosine around 0.9 — close enough to look like a tolerance problem and not like a bug.

- [ ] **Step 2: Write the failing test**

`tests/qwen3_tts_speaker_encoder_test.cpp`, on the fixture pattern of `tests/qwen3_tts_codec_test.cpp` — `make_context` with `no_alloc = true`, a `Fixture` that owns backend, context, buffer and weights and frees them, a node budget, and a backend sweep in `main`. No GGUF; weights are built in memory from a single 64-bit LCG so both sides draw in the same order, as `tests/omnivoice_reference_encoder_test.cpp:36-38` does.

```cpp
constexpr size_t kNodeBudget = 4096;

// A miniature encoder: the same topology at widths that fit a unit test.
// The production widths (512 channels, 128 attention, scale 8, 1536 mfa) are
// literals in catalog.cpp for the reasons recorded there; this fixture uses
// the same literals so the graph is exercised at its real shape, and only
// mel_bins/frames/enc_dim shrink.
int test_the_embedding_has_the_declared_width() {
    Fixture fixture = build_fixture(/*mel_bins=*/128, /*frames=*/40, /*enc_dim=*/1024);
    std::vector<float> output;
    SYNTH_TEST_CHECK(run_case(fixture, output) == SYNTH_OK);
    SYNTH_TEST_CHECK(output.size() == 1024);
    for (float value : output) {
        SYNTH_TEST_CHECK(std::isfinite(value));
    }
    return 0;
}

// Attentive statistics pooling is the whole point of the architecture: the
// embedding must not depend on how long the clip is in a trivial way, but it
// MUST depend on what is in it. Two different mels of the same length that
// produce the same embedding mean the pooling collapsed.
int test_different_mels_produce_different_embeddings() {
    Fixture fixture = build_fixture(128, 40, 1024);
    std::vector<float> a;
    std::vector<float> b;
    SYNTH_TEST_CHECK(run_case(fixture, a, /*mel_seed=*/1) == SYNTH_OK);
    SYNTH_TEST_CHECK(run_case(fixture, b, /*mel_seed=*/2) == SYNTH_OK);
    SYNTH_TEST_CHECK(a != b);
    return 0;
}

// A one-frame mel is the degenerate pooling case: the standard deviation over
// a single frame is zero, and a naive sqrt of it is a NaN that only shows up
// on the shortest clip anybody enrolls.
int test_a_single_frame_mel_stays_finite() {
    Fixture fixture = build_fixture(128, 1, 1024);
    std::vector<float> output;
    SYNTH_TEST_CHECK(run_case(fixture, output) == SYNTH_OK);
    for (float value : output) {
        SYNTH_TEST_CHECK(std::isfinite(value));
    }
    return 0;
}

int test_rejections() {
    Fixture fixture = build_fixture(128, 40, 1024);
    SYNTH_TEST_CHECK(synth::qwen3tts::build_speaker_encoder(nullptr, fixture.weights, fixture.mel) == nullptr);
    SYNTH_TEST_CHECK(synth::qwen3tts::build_speaker_encoder(fixture.graph, fixture.weights, nullptr) == nullptr);
    // A mel whose bin count disagrees with the stem's declared input extent
    // would convolve garbage rather than fail.
    SYNTH_TEST_CHECK(synth::qwen3tts::build_speaker_encoder(fixture.graph, fixture.weights,
                                                           wrong_bins(fixture, 64)) == nullptr);
    return 0;
}

int test_the_graph_stays_within_its_node_budget() {
    Fixture fixture = build_fixture(128, 40, 1024);
    SYNTH_TEST_CHECK(ggml_graph_n_nodes(fixture.built) < int(kNodeBudget));
    return 0;
}
```

- [ ] **Step 3: Run it and watch it fail**

```bash
cmake --build build --target synthesize-qwen3-tts-speaker-encoder-test
```
Expected: FAIL — `arch/qwen3-tts/speaker-encoder.h: No such file or directory`. Register the target first (`synth_add_unit_test(synthesize-qwen3-tts-speaker-encoder-test qwen3_tts_speaker_encoder_test.cpp)` beside `tests/CMakeLists.txt:173`), re-run, and confirm the failure is now `build_speaker_encoder` undeclared.

- [ ] **Step 4: Implement the graph**

```cpp
// src/arch/qwen3-tts/speaker-encoder.h
#pragma once

#include <cstdint>

struct ggml_context;
struct ggml_tensor;

namespace synth::qwen3tts {

struct SpeakerEncoderWeights;

// ECAPA-TDNN over a [mel_bins, frames] F32 mel, producing a [enc_dim] F32
// speaker embedding.
//
// Returns nullptr on a null argument, a mel whose bin count disagrees with the
// stem convolution's declared input extent, or a weights struct with a null
// pointer in it -- the last of which is only reachable if a resolver dropped
// something, which is what tests/qwen3_tts_catalog_test.cpp now pins.
//
// The dilations, activations and pooling details are transcribed from the
// pinned upstream source and recorded, with line numbers, in the Task 1
// oracle's conventions.json. None of them is declared in the package.
ggml_tensor * build_speaker_encoder(ggml_context *                context,
                                    const SpeakerEncoderWeights & weights,
                                    ggml_tensor *                 mel);

}  // namespace synth::qwen3tts
```

Structure of the body, in order: stem convolution (kernel 5, dilation 1, same padding) and activation; three SE-Res2Net blocks at their recorded dilations, each concatenating the eighth pass-through split with the seven convolved ones; concatenate the three block outputs to `3 * 512 = 1536` and apply `mfa`; attentive statistics pooling — mean and standard deviation over time, broadcast and concatenated to `[4608, frames]`, `asp_tdnn` to 128, the recorded activation, `asp` to 1536, softmax over time, weighted mean and weighted standard deviation to `[3072]`; `fc` to `[enc_dim]`.

**Erratum, 2026-08-12 — the paragraph says "each concatenating the eighth
pass-through split with the seven convolved ones", but the first split is the
one that passes through unconvolved and leads the concatenation.** This is
established by `modeling_qwen3_tts.py:115-126` and already corrected in
`src/arch/qwen3-tts/catalog.h` and `catalog.cpp`.

Use `ggml_conv_1d` with explicit dilation. The standard deviation needs a floor before the square root — a one-frame clip makes the variance exactly zero and `ggml_sqrt` of it is fine, but the *derivative-free* NaN risk is in `sqrt(variance)` where variance goes slightly negative from catastrophic cancellation; clamp with `ggml_clamp` at a small positive epsilon and say so in a comment, because `test_a_single_frame_mel_stays_finite` is the only thing that will ever notice.

- [ ] **Step 5: Run the test, the unit gate and the sanitizer gate**

```bash
ctest --test-dir build -R '^synthesize-qwen3-tts-speaker-encoder-test$' --output-on-failure
cmake --build build --target synthesize-check-unit
cmake --build build-sanitize --target synthesize-check-unit
```
Expected: PASS, then 89/91 in both.

- [ ] **Step 6: Commit**

```bash
scripts/ci/clang-format.sh --fix
git add src/arch/qwen3-tts/speaker-encoder.h src/arch/qwen3-tts/speaker-encoder.cpp \
        tests/qwen3_tts_speaker_encoder_test.cpp tests/CMakeLists.txt
scripts/ci/clang-format.sh --check-diff origin/main
git commit -m "qwen3-tts: the ECAPA-TDNN speaker encoder graph"
```

---

### Task 5: The host wrapper, and the x-vector against the oracle

**Files:**
- Create: `src/arch/qwen3-tts/speaker-encoder-host.h`, `src/arch/qwen3-tts/speaker-encoder-host.cpp`
- Modify: `src/arch/qwen3-tts/qwen3-tts.h`, `src/arch/qwen3-tts/model.cpp`

**Interfaces:**
- Produces: `Model::prepare_x_vector(const std::vector<float> & pcm_24k, int threads, XVectorEncoding & output)`, mirroring OmniVoice's `Model::encode_reference` (`src/arch/omnivoice/omnivoice.h:354`). Task 7 calls it.

- [ ] **Step 1: Declare the encoding and the entry point**

```cpp
// src/arch/qwen3-tts/speaker-encoder-host.h
#pragma once

#include "synthesize.h"

#include <cstdint>
#include <vector>

namespace synth::qwen3tts {

struct HParams;
struct SpeakerEncoderWeights;

// What one enrollment produced. `ref_rms` is instrumentation and a gate: a
// digitally silent clip produces an embedding that is finite, stable and
// meaningless, so it is refused by name rather than cloned.
struct XVectorEncoding {
    std::vector<float> x_vector;  // enc_dim floats
    float              ref_rms  = 0.0f;
    uint64_t           mel_frames = 0;
};

// PCM at the package's declared reference rate to a speaker embedding: mel
// front end, ECAPA graph, one shot on the CPU.
//
// The speaker encoder stays on the CPU in Plan 2. The catalog's twin-context
// note (src/arch/qwen3-tts/catalog.cpp:645-655) says the encoder half "is
// resolved against the package only, and stays on the CPU until a graph
// exists that reads it"; a graph now exists, and moving it to an accelerator
// is a measured decision Plan 4 makes, not a side effect of this one.
synth_status_t encode_speaker_reference(const HParams &               hparams,
                                        const SpeakerEncoderWeights & weights,
                                        const std::vector<float> &    pcm,
                                        int                           threads,
                                        XVectorEncoding &             output,
                                        const char *&                 out_diagnostic_code,
                                        const char *&                 out_diagnostic_message);

}  // namespace synth::qwen3tts
```

- [ ] **Step 2: Implement it**

Order: compute `ref_rms` over the PCM; refuse `ref_rms == 0.0f` with `"voice_profile.reference_silent"` (OmniVoice's own ruling, `src/arch/omnivoice/profile.cpp:51-88`, and the same diagnostic code so the two families answer alike); `compute_log_mel`; build a one-shot graph, allocate, compute, read back; assert the output is `hparams.speaker_encoder.enc_dim` elements and every value finite.

That length assertion closes carryover §3 Task 2(a): "No runtime assertion that the speaker embedding is 1024 elements, unlike the `shape[1] == 16` checks on codes."

- [ ] **Step 3: Add the Model method**

In `src/arch/qwen3-tts/qwen3-tts.h`, beside `decode_codes` (`:175`):

```cpp
    // Reference audio to a speaker embedding. Split out from Voice Profile
    // preparation the way decode_codes is split from run_synthesis: this half
    // is deterministic and is compared against the oracle on its own.
    synth_status_t prepare_x_vector(const std::vector<float> & pcm_24k,
                                    int                        threads,
                                    XVectorEncoding &          output,
                                    const char *&              out_diagnostic_code,
                                    const char *&              out_diagnostic_message) const;
```

In `model.cpp`, implement it as a thin forward to `encode_speaker_reference`, refusing with `SYNTH_ERR_UNSUPPORTED_VOICE` when `!hparams.has_speaker_encoder`.

- [ ] **Step 4: Drive the real package and compare against the oracle**

There is no unit-tier path to this: a `unit`-labelled test may not depend on the 2.5 GB package (`docs/testing.md`, carryover §4.3). Use the integration harness. Add a temporary driver arm to `tests/qwen3_tts_base_load_real.cpp` — or a scratch `main` under `build/` that is never committed — that loads the package, reads `models/qwen3-tts-reference-audio/clone.wav`, normalizes it, and calls `prepare_x_vector`. Then:

```bash
uv run --project scripts/envs/qwen3-tts --locked python scripts/validate-qwen3-tts-replay.py \
  --manifest tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-base.manifest.json \
  --case base-xvector-en --compare-x-vector
```

Expected: a cosine against `speaker/x_vector.f32` above `0.9999` and a max-abs below `0.05`. **Both numbers are provisional placeholders invented for this plan — no x-vector measurement exists in this tree, and the tolerance file's `speaker_encoder` placeholder is a verbatim copy of a different model's talker probes (`tests/tolerances/qwen3-tts.json:353-361`), not a measurement of anything.** Task 6 is what commits the real ones.

If the cosine is far below that, compare `speaker/mel.f32` first and then each of the four ECAPA taps in order — the first tap that diverges names the stage, which is the entire reason Task 1 exists.

- [ ] **Step 5: Run the unit gate and the sanitizer gate**

Expected: 89/91 in both, no sanitizer diagnostics.

- [ ] **Step 6: Commit**

```bash
scripts/ci/clang-format.sh --fix
git add src/arch/qwen3-tts/speaker-encoder-host.h src/arch/qwen3-tts/speaker-encoder-host.cpp \
        src/arch/qwen3-tts/qwen3-tts.h src/arch/qwen3-tts/model.cpp
scripts/ci/clang-format.sh --check-diff origin/main
git commit -m "qwen3-tts: reference audio to an x-vector, measured against the oracle"
```

Record the observed cosine and max-abs in the commit message, together with the build that produced them (`build/`, `RelWithDebInfo` via the dev preset) — the numbers are correctness, not timing, so the `-O2` caveat does not apply, but naming the build costs a line and settles the question.

---

### Task 6: Commit the measured tolerance without breaking the CustomVoice grid

**Files:**
- Modify: `tests/tolerances/qwen3-tts.json`
- Modify: `scripts/validate-qwen3-tts-replay.py`

**The constraint, stated so it is not rediscovered.** `tests/python/test_tolerance_coverage.py:129-149` asserts **exact set equality** between the family-wide validator basenames — globbed from `scripts/validate-qwen3-tts-*.py` and stripped of the `validate-qwen3-tts-` prefix — and **each** variant's reference-profile stage set. Today both sides are `{public, replay}`. `reference_stage` is `"source-bf16-oracle-vs-f32-cpu"`, so the reference profile is `BF16`. And `test_measured_families_cover_every_profile_backend_and_stage` (`:88-127`) requires every profile *and every backend* of a variant to carry that identical stage set.

So adding `scripts/validate-qwen3-tts-speaker_encoder.py` — which the spec's §6 implies and Plan 1's provisional keys anticipated — would force a `speaker_encoder` stage into `qwen3-tts-12hz-0-6b-customvoice`'s `BF16`, `F16` and `Q8_MIXED` grids *and* their `CUDA` sub-grids. CustomVoice has no speaker encoder. Those five cells would be a measurement that never happened, which is precisely the class of claim this project's tolerance file already refuses to make (`tests/tolerances/qwen3-tts.json:333-345`: copying customvoice's numbers "would claim a result that has never happened").

**The deliberate resolution: no new validator basename, and the gate lands at probe granularity inside the stage that already exists.** `speaker.x_vector` becomes a probe of the Base variant's `BF16/CPU/replay` stage, which currently carries `cases: 0` and `probes: {}`. Stage names stay `{public, replay}` on both variants, both coverage tests keep passing untouched, `test_the_committed_file_resolves_both_qwen3_tts_variants` keeps finding a `replay` cell with `probes`, and the gate is a real measured threshold rather than a stage nobody measured. The cost is that "speaker encoder" is not a stage name; the benefit is that no cell in the file describes a subsystem its variant does not have.

- [ ] **Step 1: Extend the existing replay validator**

In `scripts/validate-qwen3-tts-replay.py`, add the `speaker.x_vector` comparison to the replay path for any case whose `oracle.parameters.x_vector_only` is true, reading its threshold through the existing `resolve_tolerance_stage(document, variant, "BF16", "CPU", "replay")` (`:41`, body `:64-74`). Do **not** add a new script file — the basename is the thing the coverage test globs.

Add a `--compare-mel` and `--compare-x-vector` mode alongside, used by Tasks 2 and 5; those are flags on an existing script and do not change its basename either.

- [ ] **Step 2: Measure over both x-vector cases**

```bash
uv run --project scripts/envs/qwen3-tts --locked python scripts/validate-qwen3-tts-replay.py \
  --manifest tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-base.manifest.json \
  --case base-xvector-en --case base-xvector-zh --report-observed
```

Expected: two observed cosines and two observed max-abs values. Both cases must be materialized — `base-xvector-zh` by Task 1 Step 4. These are the only two of the manifest's 12 cases Plan 2 can drive end to end; the other ten are ICL and belong to Plan 3.

- [ ] **Step 3: Commit the measured cell**

Fill `variants.qwen3-tts-12hz-0-6b-base.profiles.BF16.stages.replay`:

```json
            "replay": {
              "phase": "oracle_replay",
              "description": "The Base variant's deterministic conditioning half on CPU: reference audio through the mel front end and the ECAPA-TDNN speaker encoder, compared against the oracle's own x-vector. The autoregressive half is not yet replayed here -- only the two x-vector-only cases can be driven end to end until Plan 3 lands ICL.",
              "cases": 2,
              "probes": {
                "speaker.x_vector": {
                  "min_cosine": 0.0,
                  "observed_min_cosine": 0.0,
                  "observed_max_abs": 0.0
                }
              },
              "backend": "CPU",
              "note": "Measured 2026-08-12 on CPU over base-xvector-en and base-xvector-zh. The gate is 5x the measured deviation in (1 - cosine), the rule this file's own top-level note states for every other probe here. max_abs is recorded but does not gate: an x-vector is a direction consumed by a dot product against the prompt, so cosine is the quantity that matters, the same reasoning that leaves max_abs ungated on the talker probes above and gating only on audio.pcm."
            },
```

Replace all three zeros with the measured values, and set `min_cosine` to `1 - 5 * (1 - observed_min_cosine)` per the file's own stated rule. **This plan does not prescribe the number; Step 2 produces it.**

- [ ] **Step 4: Retire the superseded placeholder**

Delete `provisional_variants.qwen3-tts-12hz-0-6b-base.profiles.BF16.stages.speaker_encoder` (`tests/tolerances/qwen3-tts.json:357-405`). It exists only because a real cell had nowhere to live, and its probe block is a verbatim copy of the CustomVoice talker probes that its own `description` disclaims. Leave `codec_encoder` (`:406-454`) exactly as it is — Plan 3 owns it.

Also close carryover §3 Task 3(c) while here: the Base variant's stages no longer copy `talker.*` probe names the Base oracle never dumps.

- [ ] **Step 5: Watch the coverage tests pass for the right reason**

```bash
uv run --project scripts/envs/qwen3-tts --locked python -m unittest \
  tests.python.test_tolerance_coverage tests.python.test_validate_qwen3_tts_replay \
  tests.python.test_golden_manifests -v
```
Expected: PASS. Then prove the constraint is real rather than assumed — create an empty `scripts/validate-qwen3-tts-speaker_encoder.py`, re-run, and watch `test_every_registered_validator_has_a_measured_stage` fail with `validators ['speaker_encoder'] have no recorded measurement` for **both** variants. Delete the file again. That failure is the evidence for this task's whole shape; a constraint nobody has seen bite is a constraint nobody has.

- [ ] **Step 6: Commit**

```bash
git add tests/tolerances/qwen3-tts.json scripts/validate-qwen3-tts-replay.py
git commit -m "qwen3-tts: commit the measured x-vector tolerance as a replay probe"
```

The commit message states the observed values, the case ids, the backend, and the build — the tolerance schema has no field for any of that (digest C9), so the commit is the only carrier.

---

### Task 7: Voice Profile preparation

**Files:**
- Create: `src/arch/qwen3-tts/profile.h`, `src/arch/qwen3-tts/profile.cpp`
- Create: `tests/qwen3_tts_profile_test.cpp`
- Modify: `src/voice-profile-handle.h`, `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `Model::prepare_x_vector` from Task 5.
- Produces: `synth::qwen3tts::XVectorProfile` and `create_x_vector_profile(...)`, on the shape of `synth::omnivoice::create_clone_prompt` (`src/arch/omnivoice/profile.h:117-124`). Task 9's dispatch arm calls it.

The public path stays refused through this task and Task 8 — the capability snapshot is still all-zero, so `synth_voice_profile_create_from_reference` still takes the generic fallback at `src/voice-profile.cpp:670`. That is deliberate: the ruling is that the advertisement appears the day preparation works end to end, and Task 9 is that day. Both tasks are unit-tested at the family layer, and the core dispatch arm's own coverage arrives in Task 9. That is a disclosed two-commit gap, in the same spirit as the `shared.voice_profile` gap Plan 1 disclosed (carryover §4.7).

- [ ] **Step 1: Write the failing test**

`tests/qwen3_tts_profile_test.cpp`, no GGUF, synthetic `HParams` and in-memory weights from a shared LCG — `tests/omnivoice_reference_encoder_test.cpp` is the closest existing analogue.

```cpp
// D4: the clone mode is fixed when the Profile is created. Plan 2 implements
// exactly one mode, so a transcript names a mode with no implementation
// behind it. Accepting it and silently building the x-vector Profile would
// hand back the weaker clone the caller did not ask for, which is the same
// capability lie the erratum removed from the source flags.
int test_a_transcript_is_rejected_not_ignored() {
    Fixture     fixture = build_fixture();
    const char * code    = nullptr;
    const char * message = nullptr;
    std::shared_ptr<const synth::qwen3tts::XVectorProfile> profile;
    const synth_status_t status = synth::qwen3tts::create_x_vector_profile(
        fixture.model, one_second_of_speech(), /*transcript=*/"hello there", /*language_tag=*/"",
        /*threads=*/1, profile, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(profile == nullptr);
    SYNTH_TEST_CHECK(code != nullptr && std::strcmp(code, "voice_profile.transcript_unsupported") == 0);
    return 0;
}

// The spec's section 9 error table: a transcript that is present but empty or
// whitespace-only is INVALID_ARG. In Plan 2 it takes the same rejection as
// any other transcript, and the distinction only matters from Plan 3 on --
// which is why this case is written now rather than discovered then.
int test_a_whitespace_transcript_is_rejected_too() {
    Fixture     fixture = build_fixture();
    const char * code    = nullptr;
    const char * message = nullptr;
    std::shared_ptr<const synth::qwen3tts::XVectorProfile> profile;
    SYNTH_TEST_CHECK(synth::qwen3tts::create_x_vector_profile(fixture.model, one_second_of_speech(), "   ", "", 1,
                                                              profile, code, message) == SYNTH_ERR_INVALID_ARG);
    return 0;
}

int test_a_silent_reference_is_refused_by_name() {
    Fixture     fixture = build_fixture();
    const char * code    = nullptr;
    const char * message = nullptr;
    std::shared_ptr<const synth::qwen3tts::XVectorProfile> profile;
    const std::vector<float> silence(24000, 0.0f);
    SYNTH_TEST_CHECK(synth::qwen3tts::create_x_vector_profile(fixture.model, silence, "", "", 1, profile, code,
                                                             message) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(code != nullptr && std::strcmp(code, "voice_profile.reference_silent") == 0);
    return 0;
}

int test_a_prepared_profile_carries_the_declared_width() {
    Fixture     fixture = build_fixture();
    const char * code    = nullptr;
    const char * message = nullptr;
    std::shared_ptr<const synth::qwen3tts::XVectorProfile> profile;
    SYNTH_TEST_CHECK(synth::qwen3tts::create_x_vector_profile(fixture.model, one_second_of_speech(), "", "", 1,
                                                             profile, code, message) == SYNTH_OK);
    SYNTH_TEST_CHECK(profile != nullptr);
    SYNTH_TEST_CHECK(profile->x_vector.size() == fixture.hparams.speaker_encoder.enc_dim);
    SYNTH_TEST_CHECK(profile->mode == synth::qwen3tts::CloneMode::XVector);
    return 0;
}
```

- [ ] **Step 2: Run it and watch it fail**

```bash
cmake --build build --target synthesize-qwen3-tts-profile-test
```
Expected: FAIL — `arch/qwen3-tts/profile.h: No such file or directory`. Register the target (`synth_add_unit_test(synthesize-qwen3-tts-profile-test qwen3_tts_profile_test.cpp)`), re-run, confirm `create_x_vector_profile` undeclared.

- [ ] **Step 3: Implement the payload and the entry point**

```cpp
// src/arch/qwen3-tts/profile.h (extract)

// Which of the two upstream clone modes this Profile fixes. D4: the mode is
// decided at preparation, not at synthesis, which is what gives a Serialized
// Profile one unambiguous meaning. Plan 2 produces only XVector; Plan 3 adds
// Icl and the fields it needs, without a schema version bump, because the
// envelope discriminates on a `kind` key rather than on its schema version.
enum class CloneMode : uint32_t {
    XVector = 0,
    // Icl -- Plan 3.
};

struct XVectorProfile {
    CloneMode          mode = CloneMode::XVector;
    std::vector<float> x_vector;    // enc_dim floats; substitutes for the prompt's speaker embedding
    float              ref_rms = 0.0f;
    std::string        language_tag;  // as declared by the caller, or empty
};

// Prepares a Voice Profile from one normalized reference clip.
//
// `transcript` non-empty is REJECTED in Plan 2 with
// "voice_profile.transcript_unsupported": this rung implements the x-vector
// mode only, and D4 fixes the mode at preparation, so accepting a transcript
// would return a Profile that means something other than what was asked for.
// Plan 3 is the change that turns this rejection into the mode selector.
synth_status_t create_x_vector_profile(const Model &                             model,
                                       const std::vector<float> &                pcm_24k,
                                       const std::string &                       transcript,
                                       const std::string &                       language_tag,
                                       int                                       threads,
                                       std::shared_ptr<const XVectorProfile> &   output,
                                       const char *&                             out_diagnostic_code,
                                       const char *&                             out_diagnostic_message);
```

Order inside: reject a non-empty transcript first (before any work); call `model.prepare_x_vector`; propagate its silent-reference refusal; build the payload.

- [ ] **Step 4: Add the profile tag**

In `src/voice-profile-handle.h:15-21`:

```cpp
enum class ProfileFamilyTag : uint32_t {
    None = 0,
    OmnivoiceClone,
    OmnivoiceDesign,
    // Wraps a synth::qwen3tts::XVectorProfile payload
    // (arch/qwen3-tts/profile.h). One tag covers both of this family's clone
    // modes: the payload's own CloneMode discriminates, so Plan 3's ICL
    // Profiles reuse this tag rather than adding a second one that every
    // switch would have to learn.
    Qwen3TtsClone,
};
```

- [ ] **Step 5: Run the test, the unit gate and the sanitizer gate**

Expected: PASS, then 89/91 in both.

- [ ] **Step 6: Commit**

```bash
scripts/ci/clang-format.sh --fix
git add src/arch/qwen3-tts/profile.h src/arch/qwen3-tts/profile.cpp \
        tests/qwen3_tts_profile_test.cpp tests/CMakeLists.txt src/voice-profile-handle.h
scripts/ci/clang-format.sh --check-diff origin/main
git commit -m "qwen3-tts: prepare an x-vector Voice Profile, and reject a transcript by name"
```

---

### Task 8: The v1 Serialized Profile envelope

**Files:**
- Modify: `src/arch/qwen3-tts/profile.h`, `src/arch/qwen3-tts/profile.cpp`
- Modify: `tests/qwen3_tts_profile_test.cpp`

**Interfaces:**
- Produces: `serialize_x_vector_profile(...)` and `load_profile_from_memory(...)`, on the shapes at `src/arch/omnivoice/profile.h:341-343` and `:386-394`. Task 9's dispatch arms call them.

Follow OmniVoice's envelope exactly (digest §2), because everything in its "repeated by any second family" column is unfactored and must be duplicated: `general.architecture = "synthprofile"`, `format_version` 1, the eight common keys, the zero-seeded `content_sha256` patched in after hashing the whole buffer, the hand-written framing (gguf has no public buffer writer outside `ggml-impl.h`), and the check order with its `UNSUPPORTED_VOICE`-vs-`INVALID_ARG` split. Family-specific: the schema string `qwen3-tts-voice-clone`, the `kind` values, and the tensor name and shape.

- [ ] **Step 1: Write the failing tamper matrix**

Model it on `tests/omnivoice_serialize_test.cpp` — hand-built payload, byte-level surgery helpers (`find_u8_32_value`, `find_string_value`), and one case per rejection. The status split is the load-bearing part and each arm gets its own failing input:

| Arm | Expected status | Why this status and not the other |
| --- | --- | --- |
| round trip of a well-formed profile | `SYNTH_OK` | — |
| `model_family` is `"omnivoice"` | `SYNTH_ERR_UNSUPPORTED_VOICE` | another family's envelope is "not mine", not "corrupt" |
| `schema` is `"qwen3-tts-voice-design"` | `SYNTH_ERR_UNSUPPORTED_VOICE` | same |
| `schema_version` is 2 | `SYNTH_ERR_UNSUPPORTED_VOICE` | a future version this build cannot read |
| `kind` is `"icl"` | `SYNTH_ERR_INVALID_ARG` | **deliberately not UNSUPPORTED_VOICE**: an unrecognized kind inside a schema this build owns is malformed, matching `src/arch/omnivoice/profile.cpp:1228-1240`. This is also the exact case Plan 3 turns into a success. |
| compatibility id one byte off | `SYNTH_ERR_UNSUPPORTED_VOICE` | prepared for a different package |
| `content_sha256` one byte off | `SYNTH_ERR_INVALID_ARG` | the bytes were altered |
| truncated buffer | `SYNTH_ERR_INVALID_ARG` | — |
| alignment not `GGUF_DEFAULT_ALIGNMENT` | `SYNTH_ERR_INVALID_ARG` | — |
| tensor count 0 or 2 | `SYNTH_ERR_INVALID_ARG` | exactly one x-vector tensor |
| x-vector length ≠ `enc_dim` | `SYNTH_ERR_INVALID_ARG` | a Profile of the wrong width builds a wrong-shaped prompt |
| a key outside the prescan whitelist | `SYNTH_ERR_INVALID_ARG` | refused before `gguf_init_from_buffer` runs |
| a whitelisted key with the wrong gguf type | `SYNTH_ERR_INVALID_ARG` | same |
| `n_kv` other than the pinned count | `SYNTH_ERR_INVALID_ARG` | same |

Plus a writer-agreement test on the shape of `tests/omnivoice_serialize_writer_agreement_test.cpp`: drive the **real** `serialize_x_vector_profile` and check its emitted key set and count against the same `kPrescanKnownKeys` table the reader uses. A key removed from the writer but left in the whitelist is a silently too-permissive whitelist that nothing else would notice.

- [ ] **Step 2: Run it and watch it fail**

```bash
ctest --test-dir build -R '^synthesize-qwen3-tts-profile-test$' --output-on-failure
```
Expected: FAIL — `serialize_x_vector_profile` undeclared.

- [ ] **Step 3: Implement the writer**

```cpp
// src/arch/qwen3-tts/profile.cpp (constants)
constexpr const char * kEnvelopeArchitecture  = "synthprofile";
constexpr uint32_t     kEnvelopeFormatVersion = 1;
constexpr const char * kEnvelopeModelFamily   = "qwen3-tts";
// ONE schema for both clone modes, exactly as omnivoice does it: the package's
// own ProfileContract already declares "qwen3-tts-voice-clone" and every
// already-converted Base package has it frozen, so a second schema id for ICL
// would need a package re-cut. `synthesize.voice_profile.kind` picks the
// payload shape instead, which is what lets Plan 3 add ICL fields with no
// schema version bump and leaves every Plan 2 Profile loadable afterwards.
constexpr const char * kEnvelopeSchema        = "qwen3-tts-voice-clone";
constexpr uint32_t     kEnvelopeSchemaVersion = 1;
constexpr const char * kKindXVector           = "x-vector";
constexpr const char * kTensorXVector         = "profile.x_vector";
```

Write the tensor as `n_dims = 1`, `ne = { enc_dim }`, `GGML_TYPE_F32`, offset 0. Reuse OmniVoice's `set_common_metadata` / `write_envelope` structure verbatim in shape, including seeding `content_sha256` with 32 zero bytes and patching the digest in after hashing (`src/arch/omnivoice/profile.cpp:638-642`).

Declare `kPrescanKnownKeys`, `PrescanKeySpec`, `PrescanKeyScope` and the pinned KV count **in the header**, `inline constexpr` at namespace scope, for the same reason OmniVoice does (`src/arch/omnivoice/profile.h:222-291`): the writer-agreement test drives the real writer against the same table rather than a second hand transcription of it.

- [ ] **Step 4: Implement the reader**

Signature on `src/arch/omnivoice/profile.h:386-394`'s shape, with the family loader setting `out_family_tag` itself (`= ProfileFamilyTag::Qwen3TtsClone`) — the core just copies it through, as `src/voice-profile.cpp:570` does.

Check order, verbatim from the OmniVoice reader's: alignment, architecture, format version, **model family (→ `UNSUPPORTED_VOICE`)**, schema and schema version (→ `UNSUPPORTED_VOICE`), `kind` (unrecognized → `INVALID_ARG`), compatibility id (→ `UNSUPPORTED_VOICE`), tensor count, then size arithmetic **before any allocation sized from untrusted bytes**.

- [ ] **Step 5: Run the tests, the unit gate and the sanitizer gate**

Expected: PASS, then 89/91 in both. The sanitizer gate matters more here than anywhere else in this plan: this is the only code in it that parses untrusted bytes.

- [ ] **Step 6: Commit**

```bash
scripts/ci/clang-format.sh --fix
git add src/arch/qwen3-tts/profile.h src/arch/qwen3-tts/profile.cpp tests/qwen3_tts_profile_test.cpp
scripts/ci/clang-format.sh --check-diff origin/main
git commit -m "qwen3-tts: the v1 Serialized Profile envelope, with a kind discriminator Plan 3 extends"
```

---

### Task 9: Open the public seam

**Files:**
- Modify: `src/arch/qwen3-tts/weights.h`, `src/arch/qwen3-tts/weights.cpp`
- Modify: `src/voice-profile.cpp`
- Modify: `tests/qwen3_tts_voice_required_test.cpp`, `tests/qwen3_tts_base_load_real.cpp`

**This is the day the capability becomes true.** Both source flags publish together, in this one commit, and the three `voice-profile.cpp` arms go live with them.

- [ ] **Step 1: Rewrite the two zero-shape assertions**

`tests/qwen3_tts_voice_required_test.cpp`'s `test_base_capability_advertises_nothing_until_preparation_exists` and `tests/qwen3_tts_base_load_real.cpp`'s `check_capabilities` both assert the all-zero shape today and both name Plan 2 as what flips them. Rewrite, do not delete — the carryover §1.4 says so, and the CMake comment at `tests/CMakeLists.txt:713-721` says why the second one cannot be folded into the first: a synthetic `HParams` fixture cannot prove the on-disk metadata reaches the public struct, only that the function would do the right thing if it did.

```cpp
// Plan 2 landed preparation, so the advertisement is now true. The gate is
// the voice mode, not has_preset_voice_catalog: that predicate is true only
// for PresetCatalog -- CustomVoice, the variant with no speaker encoder --
// and gating on it would advertise Reference Audio on the variant that cannot
// prepare anything. The carryover records the correction; the assertion below
// is what would have caught it.
int test_base_capability_publishes_both_sources_together() {
    const synth::qwen3tts::HParams h = base_hparams();
    synth::VoiceProfileInfo        info;
    synth::qwen3tts::fill_voice_profile_capability(h, info);

    SYNTH_TEST_CHECK((info.source_flags & SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO) != 0);
    SYNTH_TEST_CHECK((info.source_flags & SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE) != 0);
    // Description Text is Stage 3's and Random Seed is nobody's: exactly two
    // bits, not "at least these two".
    SYNTH_TEST_CHECK(info.source_flags ==
                     (SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO | SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE));

    // Plan 2 implements one of two modes, so a transcript names a mode with no
    // implementation behind it. Plan 3 flips both of these to OPTIONAL in the
    // same change that lands ICL.
    SYNTH_TEST_CHECK(info.reference_transcript == SYNTH_REQUIREMENT_UNSUPPORTED);
    SYNTH_TEST_CHECK(info.reference_language == SYNTH_REQUIREMENT_UNSUPPORTED);
    SYNTH_TEST_CHECK(info.description_language == SYNTH_REQUIREMENT_UNSUPPORTED);

    SYNTH_TEST_CHECK(info.reference_target_sample_rate == 24000);
    SYNTH_TEST_CHECK(info.reference_target_channels == 1);
    SYNTH_TEST_CHECK(info.min_frames_per_clip == 24000);
    SYNTH_TEST_CHECK(info.max_frames_per_clip == 720000);
    SYNTH_TEST_CHECK(info.max_total_frames == 720000);
    SYNTH_TEST_CHECK(info.max_reference_count == 1);
    SYNTH_TEST_CHECK(info.schema == "qwen3-tts-voice-clone");
    SYNTH_TEST_CHECK(info.schema_version == 1);
    bool any_nonzero = false;
    for (uint8_t byte : info.compatibility_id) {
        any_nonzero = any_nonzero || byte != 0;
    }
    SYNTH_TEST_CHECK(any_nonzero);
    return 0;
}
```

Keep `test_customvoice_capability_reports_no_voice_profile_support` and `test_speaker_encoder_without_profile_sources_advertises_nothing` **exactly as they are**. The second is the one that would catch a gate keyed on `has_speaker_encoder` alone: it builds `has_speaker_encoder = true` with `PresetCatalog` mode and requires the all-zero answer.

- [ ] **Step 2: Run them and watch them fail**

```bash
cmake --build build --target synthesize-qwen3-tts-voice-required-test && \
  ctest --test-dir build -R '^synthesize-qwen3-tts-voice-required-test$' --output-on-failure
```
Expected: FAIL on `info.source_flags == (...)` — the function still writes `info = VoiceProfileInfo{}`.

- [ ] **Step 3: Implement the capability**

Replace the body of `fill_voice_profile_capability` (`src/arch/qwen3-tts/weights.cpp:691-724`):

```cpp
void fill_voice_profile_capability(const HParams & hparams, VoiceProfileInfo & info) {
    info = VoiceProfileInfo{};
    // The gate is the voice mode. `has_speaker_encoder` alone is not enough:
    // an encoder flag says the package carries weights, not that the runtime
    // can prepare anything, and a PresetCatalog package that somehow carried
    // one must still advertise nothing (see
    // qwen3_tts_voice_required_test.cpp's adversarial case). And
    // `has_preset_voice_catalog` is the INVERSE of this condition -- it is
    // true only for CustomVoice -- which is the trap the Plan 1 carryover
    // corrected before Plan 2 was written.
    if (hparams.voice_mode != VoiceMode::ProfileSources) {
        return;
    }
    // Reference Audio and Serialized Profile publish together, never
    // separately: docs/c-interface.md requires the second bit of any Model
    // that can create a v1 Profile, because every successfully prepared v1
    // Profile can be serialized.
    info.source_flags = SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO | SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE;
    // UNSUPPORTED, not OPTIONAL: this rung implements the x-vector mode only.
    // D4 fixes the clone mode at preparation, so an optional transcript would
    // invite a caller to pass one and receive the weaker clone it did not ask
    // for. Plan 3 flips both in the change that lands ICL.
    info.reference_transcript = SYNTH_REQUIREMENT_UNSUPPORTED;
    info.reference_language   = SYNTH_REQUIREMENT_UNSUPPORTED;

    info.reference_target_sample_rate = hparams.profile.reference_sample_rate;
    info.reference_target_channels    = hparams.profile.reference_channels;
    info.min_frames_per_clip          = hparams.profile.min_frames_per_clip;
    info.max_frames_per_clip          = hparams.profile.max_frames_per_clip;
    info.max_total_frames             = hparams.profile.max_total_frames;
    info.max_reference_count          = uint32_t(hparams.profile.max_reference_count);
    info.schema                       = hparams.profile.schema;
    info.schema_version               = hparams.profile.schema_version;
    decode_profile_compatibility_id(hparams.profile.compatibility_id_hex, info.compatibility_id);
}
```

- [ ] **Step 4: Correct the two stale comments**

`src/arch/qwen3-tts/weights.h:271` and the block at `src/arch/qwen3-tts/weights.cpp:720` both still instruct a future reader that the gate is `has_preset_voice_catalog`. The commit that fixed the carryover and the spec (`91001fe`) touched documentation only; these two lines still carry the inverted instruction into the tree. Replace both with the voice-mode condition and a one-line note that the predicate is its inverse.

Then decide `has_preset_voice_catalog`'s fate, per carryover §4.8: it now has no production caller and Plan 2 does not consume it. Either keep it — the unit tests assert the discriminator directly, which is a real use — or delete it and rewrite `test_base_package_carries_no_preset_voice_catalog` against `voice_mode`. Make the choice explicitly in the commit message rather than leaving it drifting.

- [ ] **Step 5: Add the three dispatch arms**

**Arm A — `synth_voice_profile_create_from_reference`** (`src/voice-profile.cpp:656-685`). Add a `ModelFamily::Qwen3Tts` branch beside the OmniVoice one at line 669, calling a sibling `create_qwen3_tts_profile_from_reference` handler. Steps 1–9 of the OmniVoice handler (`src/voice-profile.cpp:203-372`) are family-independent and there is no shared helper for them — duplicate them in order, and keep two orderings that are load-bearing:

- **`synth::validate_reference_format(sample_rate, channel_count)` runs BEFORE any RFE arithmetic.** `reference_frame_equivalent` does not validate the rate (`src/audio-normalizer.h:32-40`), so a caller deriving a length decision first shadows an out-of-contract rate with a length-derived status. The OmniVoice handler calls it a second, earlier time for exactly this reason (`src/voice-profile.cpp:327-330`).
- The `reference_count != 1` refusal by name, not silent truncation.

The one family-specific difference: **a non-null transcript is refused here**, with `SYNTH_ERR_INVALID_ARG` and `"voice_profile.transcript_unsupported"`, before normalization. The capability says `UNSUPPORTED`; the runtime must agree.

**Arm B — `synth_voice_profile_load_from_memory`** (`:747-782`). Add the `Qwen3Tts` branch at line 769. **Preserve the ordering pinned by the comment at 763-768**: `params` is fully validated before `model` is dereferenced, because `tests/voice_profile_api_test.c` drives this function with a dummy `(synth_model_t *) 1`.

**Arm C — `synth_voice_profile_serialize`** (`:784-820`). This one branches on the **profile's** `family_tag`, not the model's family (`:802-803`). Add `ProfileFamilyTag::Qwen3TtsClone` to that condition and route it to `serialize_x_vector_profile`, reading the compatibility id off the model as OmniVoice does (`:507`).

**Not touched, deliberately:** `synth_voice_profile_create_from_description` (`:691-720`) — Description Text is Stage 3, and the carryover §1.4 says so explicitly; and `synth_voice_profile_create_random` (`:726-737`), which has no family branch at all and which no family implements.

- [ ] **Step 6: Prove the public path works, and that its refusals do**

Extend `tests/qwen3_tts_base_load_real.cpp` — the only tier that can prove the on-disk metadata reaches the public struct — with the published-shape assertions and a public round trip:

```c
// Create -> serialize -> load -> the loaded Profile is usable. Every one of
// these calls returned SYNTH_ERR_UNSUPPORTED_VOICE before this commit.
synth_voice_profile_t * profile = NULL;
SYNTH_TEST_CHECK(synth_voice_profile_create_from_reference(model, &params, &profile) == SYNTH_OK);
synth_byte_buffer_t * bytes = NULL;
SYNTH_TEST_CHECK(synth_voice_profile_serialize(profile, &serialize_params, &bytes) == SYNTH_OK);
synth_voice_profile_t * reloaded = NULL;
SYNTH_TEST_CHECK(synth_voice_profile_load_from_memory(model, &load_params, &reloaded) == SYNTH_OK);
```

plus the refusals: a transcript (`SYNTH_ERR_INVALID_ARG`, `"voice_profile.transcript_unsupported"`), two clips (`SYNTH_ERR_INVALID_ARG`), a clip below `min_frames_per_clip` and above `max_frames_per_clip`, and rate 0 / 7999 / 192001 and 3 channels at both the minimum and maximum length (`SYNTH_ERR_UNSUPPORTED_INPUT`) — the eight-assertion format-gate matrix `tests/omnivoice_profile_test.cpp:385-404` already established.

- [ ] **Step 7: Run the unit gate, the sanitizer gate and the integration gate**

```bash
cmake --build build --target synthesize-check-unit
cmake --build build-sanitize --target synthesize-check-unit
cmake --build build-integration --target synthesize-qwen3-tts-base-load-real && \
  ctest --test-dir build-integration -R '^synthesize-qwen3-tts-base-load-real$' --output-on-failure
```
Expected: 89/91, no sanitizer diagnostics, integration PASS.

- [ ] **Step 8: Commit**

```bash
scripts/ci/clang-format.sh --fix
git add src/arch/qwen3-tts/weights.h src/arch/qwen3-tts/weights.cpp src/voice-profile.cpp \
        tests/qwen3_tts_voice_required_test.cpp tests/qwen3_tts_base_load_real.cpp
scripts/ci/clang-format.sh --check-diff origin/main
git commit -m "qwen3-tts: publish Reference Audio and Serialized Profile, on the day preparation works"
```

This commit also returns coverage to `shared.voice_profile = info.voice_profile;` in `src/synthesize.cpp`, which has been a provably unobservable no-op since Plan 1 (carryover §4.7) precisely because the snapshot was all-zero. Deleting that line must now fail a test; check that it does before committing.

---

### Task 10: The x-vector into the prompt slot

**Files:**
- Modify: `src/arch/qwen3-tts/talker-host.h`, `src/arch/qwen3-tts/talker-host.cpp`
- Modify: `src/arch/qwen3-tts/talker.h`, `src/arch/qwen3-tts/talker.cpp`
- Modify: `src/arch/qwen3-tts/qwen3-tts.h`, `src/arch/qwen3-tts/model.cpp`
- Create: `tests/qwen3_tts_prompt_slot_test.cpp`
- Modify: `tests/CMakeLists.txt`

**The three places this touches, and why all three.** The speaker enters the prompt at four points (digest §6). The x-vector substitutes at the last: `src/arch/qwen3-tts/talker.cpp:49`, `ggml_get_rows(context, weights.codec_embedding, codec_tokens)`. But two upstream sites block it first:

- **`Model::resolve_voice` refuses every Base request before a profile is consulted.** `find_preset_voice` always fails on a Base package, so `src/arch/qwen3-tts/model.cpp:392-394` returns `SYNTH_ERR_UNSUPPORTED_VOICE` unconditionally, and it is called unconditionally at `:713-717`. Nothing in the spec or the carryover names this site (digest C8).
- **`prompt_request.has_speaker` is hardcoded `true` at `model.cpp:724`.**

**The substitution must not move the slot.** Upstream replaces the embedding at the position the speaker token occupies; it does not remove the position. Dropping the token would shorten the prompt by one and change every position after it, which produces speech rather than an error.

- [ ] **Step 1: Write the failing test**

`tests/qwen3_tts_prompt_slot_test.cpp`, synthetic `HParams`, no GGUF:

```cpp
// The externally-sourced speaker slot occupies the same POSITION the token
// would. A prompt one position shorter still synthesizes -- in the wrong
// voice at the wrong length -- which is why the length is asserted rather
// than the contents alone.
int test_an_external_speaker_slot_does_not_change_the_prompt_length() {
    const synth::qwen3tts::HParams h = base_hparams();

    synth::qwen3tts::TalkerPromptRequest with_token = request_with_language();
    with_token.has_speaker                          = true;
    with_token.speaker_token                        = 4242;

    synth::qwen3tts::TalkerPromptRequest with_x_vector = with_token;
    with_x_vector.speaker_is_external                  = true;

    synth::qwen3tts::TalkerPrompt a;
    synth::qwen3tts::TalkerPrompt b;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, with_token, a) == SYNTH_OK);
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, with_x_vector, b) == SYNTH_OK);
    SYNTH_TEST_CHECK(a.positions.size() == b.positions.size());
    SYNTH_TEST_CHECK(b.external_speaker_index >= 0);
    return 0;
}

// The index the graph substitutes at is an index into the flattened codec
// run, not into the prompt. Naming a language lengthens the codec stream by
// one, so a hardcoded index is right for exactly one of the two paths.
int test_the_substitution_index_follows_the_language_token() {
    const synth::qwen3tts::HParams h = base_hparams();

    synth::qwen3tts::TalkerPrompt named;
    synth::qwen3tts::TalkerPrompt automatic;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, external_request(/*has_language=*/true), named) == SYNTH_OK);
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, external_request(/*has_language=*/false), automatic) ==
                     SYNTH_OK);

    std::vector<int32_t> text;
    std::vector<int32_t> codec;
    int64_t              offset = 0;
    SYNTH_TEST_CHECK(synth::qwen3tts::flatten_talker_prompt(h, named, text, codec, offset) == SYNTH_OK);
    SYNTH_TEST_CHECK(named.external_speaker_index == 4);
    SYNTH_TEST_CHECK(synth::qwen3tts::flatten_talker_prompt(h, automatic, text, codec, offset) == SYNTH_OK);
    SYNTH_TEST_CHECK(automatic.external_speaker_index == 3);
    return 0;
}

// The Stage 1 path must be untouched: no x-vector, no index, and a graph
// identical to the one the talker test already pins.
int test_the_preset_voice_path_is_unchanged() {
    const synth::qwen3tts::HParams h = customvoice_hparams();
    synth::qwen3tts::TalkerPrompt  prompt;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, request_with_token(), prompt) == SYNTH_OK);
    SYNTH_TEST_CHECK(prompt.external_speaker_index == -1);
    return 0;
}

// The graph refuses an index it cannot honour rather than substituting into
// whatever row it lands on.
int test_the_graph_refuses_an_out_of_range_substitution() {
    Fixture fixture = build_talker_fixture();
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prefill_input(fixture.context, fixture.weights, fixture.text,
                                                                fixture.codec, /*codec_offset=*/0,
                                                                fixture.x_vector, /*speaker_index=*/999) == nullptr);
    // An x-vector whose width is not the talker's hidden size cannot be a row
    // of this tensor. enc_dim == hidden_size is enforced at load
    // (weights.cpp:553); this is what happens if a Profile from elsewhere
    // reaches here anyway.
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prefill_input(fixture.context, fixture.weights, fixture.text,
                                                                fixture.codec, 0, fixture.wrong_width_x_vector,
                                                                4) == nullptr);
    return 0;
}
```

- [ ] **Step 2: Run it and watch it fail**

```bash
cmake --build build --target synthesize-qwen3-tts-prompt-slot-test
```
Expected: FAIL — `'struct synth::qwen3tts::TalkerPromptRequest' has no member named 'speaker_is_external'`. Register the target beside `tests/CMakeLists.txt:173` first.

- [ ] **Step 3: Extend the prompt layout**

In `talker-host.h`, add to `TalkerPromptRequest`:

```cpp
    // The speaker slot's embedding comes from a prepared Voice Profile rather
    // than from the codec vocabulary. The POSITION is unchanged -- upstream
    // substitutes the embedding, it does not move or remove the slot -- so
    // `has_speaker` must also be set and `speaker_token` is written into the
    // stream as an inert placeholder whose row the graph never reads.
    bool                  speaker_is_external = false;
```

and to `TalkerPrompt`:

```cpp
    // Index into the flattened codec run whose embedding row the x-vector
    // replaces, or -1 when the speaker is an ordinary codec token. It is an
    // index into `codec_tokens`, not into `positions`: the codec run starts at
    // `codec_offset` and the graph adds it as a tail.
    int64_t external_speaker_index = -1;
```

In `build_talker_prompt`, when `speaker_is_external` is set, push `tokens.codec_pad` at the speaker slot instead of `request.speaker_token` and record the slot's index in the `codec` vector. That index is exactly the index into the flattened run, because `flatten_talker_prompt` (`talker-host.cpp:118-156`) already checks that the codec-carrying positions are a contiguous tail — the codec entries before the text pair with `tts_pad`, the text positions pair with `codec_pad`, so the run has no gap.

- [ ] **Step 4: Substitute in the graph**

Give `build_talker_prefill_input` two more parameters (`ggml_tensor * speaker_embedding`, `int64_t speaker_index`), defaulting to `nullptr` / `-1` so the Stage 1 path is byte-identical. When they are set, split the accumulation into three instead of one:

```cpp
    ggml_tensor * codec = ggml_get_rows(context, weights.codec_embedding, codec_tokens);
    if (speaker_embedding == nullptr) {
        return ggml_acc(context, text, codec, text->nb[1], text->nb[2], text->nb[3],
                        size_t(codec_offset) * text->nb[1]);
    }
    // The x-vector substitutes for one row of the codec embedding at the
    // position the speaker token occupies. enc_dim equals the talker's hidden
    // size -- weights.h:26-28: "there is no projection between them" -- so it
    // is literally a row of this [hidden_size, N] F32 tensor. Splitting the
    // accumulation around that index is what leaves the placeholder row
    // computed but never read; zeroing it in place would need a second pass
    // over a tensor ggml_acc has already consumed.
    ggml_tensor * head = ggml_view_2d(context, codec, codec->ne[0], speaker_index, codec->nb[1], 0);
    ggml_tensor * tail = ggml_view_2d(context, codec, codec->ne[0], codec->ne[1] - speaker_index - 1, codec->nb[1],
                                      size_t(speaker_index + 1) * codec->nb[1]);
    ggml_tensor * out  = text;
    if (speaker_index > 0) {
        out = ggml_acc(context, out, head, out->nb[1], out->nb[2], out->nb[3], size_t(codec_offset) * out->nb[1]);
    }
    out = ggml_acc(context, out, speaker_embedding, out->nb[1], out->nb[2], out->nb[3],
                   size_t(codec_offset + speaker_index) * out->nb[1]);
    if (tail->ne[1] > 0) {
        out = ggml_acc(context, out, tail, out->nb[1], out->nb[2], out->nb[3],
                       size_t(codec_offset + speaker_index + 1) * out->nb[1]);
    }
    return out;
```

Refuse (`nullptr`) when `speaker_index` is outside `[0, codec_tokens->ne[0])` or when `speaker_embedding->ne[0] != text->ne[0]`.

Nothing downstream changes: `build_talker_step_input` (`talker.cpp:56-67`) and the decode loop see the same `[hidden_size, N]` F32 tensor they always did.

- [ ] **Step 5: Restructure `resolve_voice` and `run_synthesis`**

Add `const std::vector<float> * x_vector = nullptr;` to `SynthesisRequest` (`src/arch/qwen3-tts/qwen3-tts.h:94`), which has no profile field today.

In `run_synthesis` (`model.cpp:698`), call `resolve_voice` only for the language when a Profile is present, and set `speaker_is_external`:

```cpp
    uint32_t       speaker_token  = 0;
    bool           has_language   = false;
    uint32_t       language_token = 0;
    const bool     external       = request.x_vector != nullptr;
    // A profile-sources package has an empty Preset Voice Catalog, so
    // resolve_voice refuses every request on it -- named or not. That refusal
    // is correct when no Profile was supplied and wrong when one was: the
    // Voice arrived as conditioning rather than as a name. Language
    // resolution still has to happen, so this splits the two rather than
    // skipping the call.
    synth_status_t status = external
                                ? resolve_language_only(request.language, has_language, language_token)
                                : resolve_voice(request.voice_id, request.language, speaker_token, has_language,
                                                language_token);
    if (status != SYNTH_OK) {
        return status;
    }
    ...
    prompt_request.has_speaker         = true;   // the slot exists either way
    prompt_request.speaker_is_external = external;
    prompt_request.speaker_token       = speaker_token;
```

The digest notes `has_speaker` was hardcoded `true` at line 724; it stays true, because the *position* is always occupied — what changes is where its embedding comes from. Add the assertion that `request.x_vector->size() == hparams.talker.hidden_size` before building the graph.

`resolve_language_only` is the language half of `resolve_voice` factored out, not a copy — a second copy of the dialect-override rule is how the two would drift.

- [ ] **Step 6: Run the tests, the unit gate and the sanitizer gate**

```bash
ctest --test-dir build -R '^synthesize-qwen3-tts-(prompt-slot|talker)-test$' --output-on-failure
cmake --build build --target synthesize-check-unit
cmake --build build-sanitize --target synthesize-check-unit
```
Expected: PASS — and `synthesize-qwen3-tts-talker-test` in particular, unchanged, because the Stage 1 path must be untouched.

- [ ] **Step 7: Commit**

```bash
scripts/ci/clang-format.sh --fix
git add src/arch/qwen3-tts/talker-host.h src/arch/qwen3-tts/talker-host.cpp src/arch/qwen3-tts/talker.h \
        src/arch/qwen3-tts/talker.cpp src/arch/qwen3-tts/qwen3-tts.h src/arch/qwen3-tts/model.cpp \
        tests/qwen3_tts_prompt_slot_test.cpp tests/CMakeLists.txt
scripts/ci/clang-format.sh --check-diff origin/main
git commit -m "qwen3-tts: substitute the x-vector at the prompt's speaker slot without moving it"
```

---

### Task 11: Consume the Profile at the synthesis seam

**Files:**
- Modify: `src/synthesize.cpp`

**Interfaces:**
- Consumes: `ProfileFamilyTag::Qwen3TtsClone` and `XVectorProfile` from Task 7, `SynthesisRequest::x_vector` from Task 10.

This is the sixth family-conditioned point, and the last one still refusing. `src/synthesize.cpp:1011-1013` returns `SYNTH_ERR_UNSUPPORTED_VOICE` for any profile reaching the qwen3-tts arm.

- [ ] **Step 1: Write the failing test**

Add to `tests/qwen3_tts_base_load_real.cpp` — this seam has no unit tier, for the same reason `tests/CMakeLists.txt:713-721` gives for the capability snapshot:

```c
// The whole point of Plan 2, at the public seam: a prepared Profile
// synthesizes rather than refusing.
SYNTH_TEST_CHECK(synth_synthesize_to_buffer(context, &request_with_profile, &result) == SYNTH_OK);
SYNTH_TEST_CHECK(result->frame_count > 0);

// A Profile prepared against a different Loaded Model is refused, and the
// refusal is a Voice refusal rather than a graph failure: the ABI has one
// voice-error status, so the diagnostic code is the only thing that tells
// them apart.
SYNTH_TEST_CHECK(synth_synthesize_to_buffer(other_context, &request_with_profile, &result) ==
                 SYNTH_ERR_UNSUPPORTED_VOICE);
SYNTH_TEST_CHECK(strcmp(seen.code, "synthesis.voice_unsupported") == 0);
SYNTH_TEST_CHECK(!wrote_audio);
```

- [ ] **Step 2: Run it and watch it fail**

```bash
cmake --build build-integration --target synthesize-qwen3-tts-base-load-real && \
  ctest --test-dir build-integration -R '^synthesize-qwen3-tts-base-load-real$' --output-on-failure
```
Expected: FAIL — `SYNTH_ERR_UNSUPPORTED_VOICE` from `src/synthesize.cpp:1012`, with no diagnostic emitted at all.

- [ ] **Step 3: Implement the arm**

Replace the blanket refusal with the OmniVoice-shaped recovery (`src/synthesize.cpp:905-923`): the cross-model check first, then the `family_tag` switch.

```cpp
        if (prepared.voice_profile != nullptr) {
            // Same cross-model rule as the omnivoice arm above: a Profile is
            // prepared against one Loaded Model's weights and its
            // Compatibility ID, and presenting it to another is a Voice
            // refusal, not a graph failure.
            if (prepared.voice_profile->model != context->model) {
                emit_diagnostic(prepared.diagnostics, SYNTH_ERR_UNSUPPORTED_VOICE, "synthesis.voice_unsupported",
                                "this Voice Profile was prepared for a different Loaded Model");
                return SYNTH_ERR_UNSUPPORTED_VOICE;
            }
            if (prepared.voice_profile->family_tag != synth::ProfileFamilyTag::Qwen3TtsClone) {
                emit_diagnostic(prepared.diagnostics, SYNTH_ERR_UNSUPPORTED_VOICE, "synthesis.voice_unsupported",
                                "this Voice Profile belongs to a different Model Family");
                return SYNTH_ERR_UNSUPPORTED_VOICE;
            }
            const auto & clone = *static_cast<const synth::qwen3tts::XVectorProfile *>(
                prepared.voice_profile->payload.get());
            // Plan 3 adds CloneMode::Icl; until then a Profile whose payload
            // says otherwise cannot be built into a prompt this rung knows how
            // to lay out, and a Profile is refused rather than downgraded.
            if (clone.mode != synth::qwen3tts::CloneMode::XVector) {
                emit_diagnostic(prepared.diagnostics, SYNTH_ERR_UNSUPPORTED_VOICE, "synthesis.voice_unsupported",
                                "this build supports x-vector Voice Profiles only");
                return SYNTH_ERR_UNSUPPORTED_VOICE;
            }
            family_request.x_vector = &clone.x_vector;
        }
```

Set `family_request.x_vector` inside the existing `try` block, beside the other `family_request` assignments, so `SynthesisRequest` is constructed once.

Rewrite the 15-line comment above the old refusal: it currently explains that this family cannot consume a Profile and that the capability snapshot advertises zero sources. Both statements are false as of Task 9.

- [ ] **Step 4: Confirm the Catalog-less refusal still holds**

A request with **no** Profile on a Base package must still fail explicitly (spec §9: "explicit error; never a silent Voice selection"). `Model::resolve_voice` is what refuses it, unchanged by Task 10 for that path. Re-run the two Plan 1 assertions and confirm both still pass.

- [ ] **Step 5: Run every gate**

```bash
cmake --build build --target synthesize-check-unit
cmake --build build-sanitize --target synthesize-check-unit
ctest --test-dir build-integration -L qwen3-tts --output-on-failure
```
Expected: 89/91, no sanitizer diagnostics, integration PASS.

- [ ] **Step 6: Commit**

```bash
scripts/ci/clang-format.sh --fix
git add src/synthesize.cpp tests/qwen3_tts_base_load_real.cpp
scripts/ci/clang-format.sh --check-diff origin/main
git commit -m "qwen3-tts: consume a Voice Profile at the synthesis seam instead of refusing it"
```

---

### Task 12: Reference audio in, cloned audio out

**Files:**
- Create: `tests/qwen3_tts_clone_real.cpp`
- Modify: `tests/CMakeLists.txt`, `docs/porting/families/qwen3-tts.md`

**Interfaces:**
- Consumes: everything above.
- Produces: the completion gate — "reference audio in, cloned audio out on CPU; a usable clone capability exists" (spec §8).

- [ ] **Step 1: Write the end-to-end integration test**

On the pattern of `tests/omnivoice_profile_test.cpp` — `main(argc == 5)` taking `<model.gguf> <reference.wav> <golden-x-vector.f32-or-missing> <scratch-dir>`, with its `read_wav_mono16` / `make_reference` / `make_params` / `synthesize_pcm` helpers. Register it doubly guarded, as OmniVoice's is at `tests/CMakeLists.txt:756-758`:

```cmake
set(_synth_qwen3_tts_reference_wav "${CMAKE_SOURCE_DIR}/models/qwen3-tts-reference-audio/clone.wav")
if(EXISTS "${SYNTH_QWEN3_TTS_BASE_TEST_MODEL}" AND EXISTS "${_synth_qwen3_tts_reference_wav}")
```

with `LABELS "integration;qwen3-tts;abi"` and a TIMEOUT of 3600.

What it proves, each as its own assertion:

1. a Profile prepared from the real clip has `enc_dim` floats and matches `speaker/x_vector.f32` above the tolerance Task 6 committed;
2. synthesis with that Profile produces finite, non-silent PCM at 24 kHz;
3. the same Profile and seed produce identical PCM twice (request repeatability);
4. two different reference clips produce different PCM — a Profile that is ignored would produce identical audio, and nothing else in this plan can see that;
5. serialize → free → load → synthesize produces the same PCM as the original Profile;
6. a Profile presented to a second Loaded Model refuses with `SYNTH_ERR_UNSUPPORTED_VOICE` and writes no audio.

Assertion 4 is the one that actually gates "cloned", as opposed to "synthesized". Assertion 5 is what makes the Serialized Profile a real artifact rather than a round-trippable blob.

- [ ] **Step 2: Run it**

```bash
cmake -S . -B build-integration -DSYNTH_BUILD_TESTS=ON -DSYNTH_BUILD_INTEGRATION_TESTS=ON \
  -DSYNTH_QWEN3_TTS_BASE_TEST_MODEL=models/qwen3-tts-12hz-0-6b-base/qwen3-tts-12hz-0-6b-base-BF16.gguf
cmake --build build-integration --target synthesize-qwen3-tts-clone-real
ctest --test-dir build-integration -R '^synthesize-qwen3-tts-clone-real$' --output-on-failure
```
Expected: PASS.

- [ ] **Step 3: Verify the Python binding claim rather than assuming it**

The digest reports the binding is fully family-generic — zero matches for `omnivoice|family` across `native_loader.c`, `synthesize_cpp/*.py`, `CMakeLists.txt` and `pyproject.toml` — and therefore that a second family needs nothing. Confirm it, twice: statically, and by running the real thing.

```bash
grep -rniE "omnivoice|qwen3|vits|kokoro|family" bindings/python/src/ bindings/python/CMakeLists.txt \
  bindings/python/pyproject.toml && echo "FAMILY-CONDITIONED — the claim is false" || echo "family-generic confirmed"
```
Expected: `family-generic confirmed`.

Then drive the real wheel against the real Base package, which is the only thing that proves the C status codes surface correctly through the extension:

```bash
uv run --project scripts/envs/qwen3-tts --locked python - <<'PY'
import numpy as np, soundfile as sf, synthesize_cpp
model = synthesize_cpp.Model("models/qwen3-tts-12hz-0-6b-base/qwen3-tts-12hz-0-6b-base-BF16.gguf")
caps = model.voice_profile_capabilities()
print("sources", caps.source_flags, "transcript", caps.reference_transcript)
pcm, sr = sf.read("models/qwen3-tts-reference-audio/clone.wav", dtype="float32")
profile = model.create_voice_profile_from_reference(
    [synthesize_cpp.VoiceReference(samples=pcm, sample_rate=sr, channel_count=1)])
audio = model.synthesize("This is a test.", voice_profile=profile, language="english")
print("frames", len(audio.samples), "finite", bool(np.isfinite(audio.samples).all()))
blob = profile.serialize()
print("serialized bytes", len(blob))
PY
```
Expected: two source flags, `reference_transcript` reported unsupported, a non-empty finite waveform, and a non-empty serialized blob — **with no change to any file under `bindings/`**. If any line of `bindings/` had to change, the digest's §10b claim was wrong and the change belongs in this task with its own test.

- [ ] **Step 4: State the CLI's position rather than leaving it implied**

Do not touch `examples/cli/`. Record in `docs/porting/families/qwen3-tts.md` that the CLI exposes no Voice Profile path for **any** family — it has no audio reader and never assigns `synth_request_t.voice_profile` — so reference-audio support there is a cross-family slice, not a qwen3-tts increment, and Plan 2's spec row overstated it.

- [ ] **Step 5: Record the outcome in the family document**

Extend `docs/porting/families/qwen3-tts.md` with a "Stage 2, Plan 2" section carrying: the mel conventions from Task 1's `conventions.json` with their upstream line numbers; the measured x-vector cosine and max-abs with the case ids, backend and build; the Profile Schema identity and its `kind` values, with the note that Plan 3 adds one without a version bump; the six family-conditioned points and which of them Plan 2 changed; and what remains — ICL, Description Text, the CLI, quantization, CUDA, and the listening pass. Update the document's `Status:` line.

- [ ] **Step 6: Commit**

```bash
scripts/ci/clang-format.sh --fix
git add tests/qwen3_tts_clone_real.cpp tests/CMakeLists.txt docs/porting/families/qwen3-tts.md
scripts/ci/clang-format.sh --check-diff origin/main
git commit -m "qwen3-tts: reference audio in, cloned audio out -- the x-vector path end to end"
```

---

## What Plan 2 does NOT deliver

Stated plainly, so nothing in this plan is read as more than it is.

- **ICL / transcript-assisted cloning.** Plan 3. `reference_transcript` and `reference_language` report `SYNTH_REQUIREMENT_UNSUPPORTED` here and a request carrying a transcript is rejected by name. The codec encoder graph is not written; `resolve_codec_encoder` still discards its pointers into a scratch struct, deliberately and with a comment saying so. Ten of the Base manifest's twelve Golden cases are ICL and cannot be driven end to end by anything in this plan.
- **Description Text.** Stage 3, on `qwen3-tts-12hz-1.7b-voicedesign`. No `ModelFamily::Qwen3Tts` arm is added to `synth_voice_profile_create_from_description`, and `SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT` stays unadvertised. Random Seed stays unadvertised too, by every family.
- **The CLI path.** `examples/cli/` is untouched. It has no Voice Profile support and no audio reader for any family; adding one is a separate cross-family slice.
- **Quantization or CUDA for the new graphs.** The speaker encoder runs on the CPU in F32/BF16 as the package ships it. Plan 4 measures whether quantizing a graph this small pays and whether it belongs on an accelerator; neither is assumed from another family's precedent, and the catalog's twin-context note is revisited there rather than here. No performance number is claimed by this plan.
- **The listening pass.** Carryover §2.2: the reference-duration bounds — 1 s minimum, 30 s maximum — are safety ceilings taken from the plan, not perceptually validated bounds. The one measurement that exists is unadjudicated: a 30 s reference produced only 9 output frames for an 11-word sentence, `base-ref-max` was never executed end to end, and `intelligible` is null. A listening pass on the 0.5 s / 1 s / 30 s renders is owed **before Stage 2 ships**, and jiangzhuo has scheduled it last. A tolerance grid is not audible evidence.
- **Publication.** Unchanged and separate, requiring confirmation at the time.
- **Quality Evaluation.** Deferred per ADR 0017.

---

## Self-review

**Spec coverage.** Section 3's family-contract additions land in Task 9, with the 2026-08-12 refinement (`UNSUPPORTED`, not `OPTIONAL`) and the erratum's pairing rule both implemented rather than paraphrased. Section 5's module table is delivered minus its two ICL rows (`codec-encoder{,-host}`), which are Plan 3's by the spec's own §8. Section 6's "mel plus speaker encoder" dump script is Task 1; its "committed tolerances" requirement is Task 6, resolved against the enforced set-equality test rather than around it. Section 9's error table is covered row by row — the transcript row by Task 7, the duration and clip-count rows by Task 9's Arm A, the compatibility-ID row by Task 8, the no-Profile row by Task 11 Step 4, the non-finite-PCM row by the Audio Normalizer path Task 9 Arm A preserves. Section 8's Plan 2 row is met except its "CLI Adapters" clause, which Task 12 Step 4 records as overstated rather than silently dropping.

Three spec/carryover conflicts the digest surfaced are resolved here rather than left for the implementer: the inverted capability gate (Global Constraints, Task 9, plus the two source comments `91001fe` did not reach); `reference_transcript` OPTIONAL-vs-UNSUPPORTED, where the spec's own refinement governs; and `Model::resolve_voice` refusing every Base request before a profile is consulted, which no source document names and which Task 10 restructures.

**Placeholder scan.** Four numbers in this plan are not measured. The mel max-abs `1e-3` (Task 2 Step 7) and the x-vector cosine `0.9999` / max-abs `0.05` (Task 5 Step 4) are invented placeholders, labelled as such in place, and all three are superseded by Task 6's measurement. `kExpectedFramesForOneSecond` and its hop-512 sibling are deliberately left unfilled, with Task 1's `conventions.json` named as the source, because a frame count this plan chose would pin the port to a formula rather than to upstream. The six mel conventions and the six ECAPA topology details are likewise left to Tasks 1 and 4 Step 1 to read off the pinned source; each is a value that changes the answer silently, so guessing one here would be worse than not naming it. No task ends with a tolerance that has never been observed to hold.

**Type consistency across tasks.** `SpeakerEncoderWeights` (Task 3) is what Task 4's `build_speaker_encoder` and Task 5's `encode_speaker_reference` both take. `MelSpectrogram` (Task 2) is `[mel_bins, frames]` mel-major in both its producer and the graph that reads it, so nothing transposes. `XVectorEncoding` (Task 5) is the host wrapper's output and `XVectorProfile` (Task 7) its serializable payload — separate types, because one carries instrumentation the envelope does not. `CloneMode` lives on the payload rather than on `ProfileFamilyTag`, so Plan 3's ICL Profiles reuse `Qwen3TtsClone` and no switch has to learn a second tag. `SynthesisRequest::x_vector` is a `const std::vector<float> *` borrowed from the Profile's `shared_ptr` payload, whose lifetime the handle owns for the whole call — the same borrow shape `replay_codes` already uses in that struct. `build_talker_prefill_input`'s two new parameters default to `nullptr` / `-1`, so every Stage 1 call site compiles unchanged and the preset-voice graph is identical, which Task 10's third test asserts.
