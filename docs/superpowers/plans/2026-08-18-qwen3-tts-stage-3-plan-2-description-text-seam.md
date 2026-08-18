# Qwen3-TTS Stage 3 Plan 2 — The Description Text Seam — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `synth_voice_profile_create_from_description` work on Qwen3-TTS — a
sentence of natural language in, a voice out — on the VoiceDesign package Plan 1
made loadable.

**Architecture:** No new graph. The Profile holds the instruct string; the text
frontend wraps and tokenizes it at synthesis; `build_talker_prompt` places the
result as text-only positions ahead of the assistant-role tokens, with the
speaker slot absent. Plan 1 built the package and proved the no-speaker prefill
against the oracle; this plan adds the seam that reaches it.

**Tech Stack:** C++17 + GGML/GGUF, CMake ≥3.24, CTest, Python via `uv` in
`scripts/envs/qwen3-tts/` (numpy 1.26.4, torch), pinned clang-format 22.1.5.

## Global Constraints

- Design of record: `docs/superpowers/specs/2026-08-18-qwen3-tts-stage-3-design.md`.
  Where execution contradicts it, add an erratum **to that document** in the
  section it corrects, leaving the original prescription standing above it. That
  document already carries errata from Plan 1; follow their shape.
- Plan 1's record: `docs/superpowers/plans/2026-08-18-qwen3-tts-stage-3-plan-1-voicedesign-package.md`,
  which also carries errata. Read them before assuming what Plan 1 delivered.
- Upstream checkpoint pinned at `5ecdb67327fd37bb2e042aab12ff7391903235d3`.
- Every slice is done only when its focused unit tests are committed, passing,
  and registered with CTest under the `unit` label (`docs/testing.md`).
- Commit golden contracts, not golden payloads. No `.gguf`, `.wav`, `.f32`,
  `.safetensors` or oracle dump is ever committed. `reports/porting/**/oracle`
  and `reports/porting/**/convert.json` are gitignored; check `git status` before
  every commit rather than trusting `git add -A`.
- Sanitizer gate for every slice touching C/C++:
  `cmake -S . -B build-sanitize -DSYNTH_BUILD_TESTS=ON -DSYNTH_BUILD_PYTHON_TESTS=ON -DSYNTH_BUILD_INTEGRATION_TESTS=OFF -DSYNTH_SANITIZE=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo`
  then `cmake --build build-sanitize --target synthesize-check-unit`.
- `scripts/ci/clang-format.sh --fix` runs **after** `git add`; then `--check-diff`
  must be clean.
- **`ctest -L qwen3-tts` does not run most of this family's tests** — the shared
  `synth_add_unit_test()` helper labels them `unit` only. Use `-R` with target
  names, or the whole gate. A green `-L qwen3-tts` proves almost nothing.
- Two unit tests fail on any tree that has never materialised a VITS model, and
  are **not yours**: `synthesize-python-api-wheel-test` (wants
  `build/goldens/vits/.../token_ids.i32`) and `synthesize-vits-python-unit`
  (wants `reports/convert/vits/vits-ljspeech-F16.json`). Any third is yours.
- Integration targets need `-DSYNTH_BUILD_INTEGRATION_TESTS=ON`; a
  `build-integration` tree exists, and the 4.3 GB VoiceDesign package and Plan 1's
  oracle dump are on disk.
- Never use bare `git stash` / `git stash pop`. Never edit anything under `ggml/`.
- Publication is out of scope for this plan.

---

## What Plan 1 left you

Read these before Task 1; each is a fact this plan depends on.

- The package **declares** `description-text` in `synthesize.voice.profile_sources`
  and the loader reads it into `HParams::profile_sources`. That is untouched.
- The **runtime withholds** the capability bit: `fill_voice_profile_capability`
  publishes `source_flags = 0` for this variant, deliberately, because
  advertising a source the seam refuses is what this family's own ICL precedent
  argues against. **Task 5 turns it back on**, and not before.
- `build_talker_prompt` already honours `has_speaker == false` by emitting a codec
  run one position shorter with `external_speaker_index == -1`. Proved against
  injected faults and against the oracle. **You do not change that layout.**
- `model.cpp:979` still sets `has_speaker = true` unconditionally. Task 4 makes it
  conditional — that is the only production change to the prompt path this plan
  makes.
- The integration driver `synthesize-qwen3-tts-voicedesign-prefill-real` compares
  an **empty-instruct** prefill against the oracle at p95 0.002690, bound 0.01,
  and exits non-zero on a breach or a shape mismatch. Task 4 extends it.
- The Golden Manifest is at `cases: []` under `suite_status: "incremental"`.
  **Task 7 fills it and removes the flag.**

---

## File Structure

**Modified:**

- `src/voice-profile-handle.h` — a second Qwen3-TTS profile tag.
- `src/arch/qwen3-tts/profile.h` / `.cpp` — the `DesignInstruct` payload, its
  constructor, its serialization, and its arm of the load dispatch.
- `src/voice-profile.cpp` — `create_from_description`'s Qwen3-TTS arm and the
  serialize/load dispatch arms.
- `src/synthesize.cpp` — the synthesis-time branch on the new tag.
- `src/arch/qwen3-tts/talker-host.h` / `.cpp` — `instruct_tokens` and its prefill
  placement.
- `src/arch/qwen3-tts/model.cpp` — `has_speaker` becomes conditional; the instruct
  tokens are threaded from the Profile.
- `src/arch/qwen3-tts/weights.cpp` — republish the capability bit.
- `docs/schemas/synthesize-golden-manifest-v1.schema.json` — `profileContract`
  learns `description_text`, and `reference` stops being unconditionally required.
- `tests/python/test_golden_manifests.py` — the `suite_status` allowlist guard.
- `scripts/validate-qwen3-tts-public.py` — `--description`.
- `tests/golden/qwen3-tts/qwen3-tts-12hz-1-7b-voicedesign.manifest.json` — cases.
- `tests/tolerances/qwen3-tts.json` — the instruct-prefill probe and the public cell.
- `tests/qwen3_tts_voicedesign_prefill_real.cpp` — a non-empty instruct.
- `scripts/dump_reference_qwen3_tts_voicedesign.py` — a non-empty instruct.

**Created:**

- `tests/qwen3_tts_design_profile_test.cpp` — the payload, its validation, its
  round-trip.

---

## Task 1: The DesignInstruct payload and its own tag

**Files:**
- Modify: `src/voice-profile-handle.h:15-27` (`ProfileFamilyTag`)
- Modify: `src/arch/qwen3-tts/profile.h`, `src/arch/qwen3-tts/profile.cpp`
- Create: `tests/qwen3_tts_design_profile_test.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `synth::ProfileFamilyTag::Qwen3TtsDesign`
  - `struct synth::qwen3tts::DesignInstruct { std::string instruct; };`
  - `synth_status_t synth::qwen3tts::create_design_profile(const HParams & hparams, const std::string & instruct, DesignInstruct & output);`
    — returns `SYNTH_OK`, or `SYNTH_ERR_INVALID_ARG` for invalid UTF-8 or for a
    string longer than `kMaxDesignInstructBytes`.
  - `constexpr size_t synth::qwen3tts::kMaxDesignInstructBytes = 4096;`

**Read first:** `src/arch/omnivoice/profile.h`'s own `DesignInstruct` — this
family's struct mirrors its shape and deliberately not its validation. OmniVoice
can reject an instruct because upstream defines a **closed attribute
vocabulary**; Qwen3-TTS's is free-form and upstream accepts any string
(`qwen_tts/inference/qwen3_tts_model.py:654`, "Empty string is allowed"). Design
decision D2 says inventing a vocabulary here would reject input upstream accepts,
which turns a port into a redesign. So this arm validates encoding and length and
nothing else, and the code must say why.

Also read `src/voice-profile-handle.h:21-26`: the existing `Qwen3TtsClone` tag
covers **both** clone modes because the payload's own `CloneMode` discriminates.
That reasoning does not extend here — a design payload has no x-vector and shares
no fields, so it takes its own tag rather than a third `CloneMode`.

- [ ] **Step 1: Write the failing tests**

Create `tests/qwen3_tts_design_profile_test.cpp`:

```cpp
// The Description Text Voice Profile payload.
//
// It holds the instruct string and nothing else, mirroring
// src/arch/omnivoice/profile.h's struct of the same name -- and deliberately
// NOT its validation. OmniVoice can reject an instruct because upstream defines
// a closed attribute vocabulary; this family's is free-form and upstream accepts
// any string, so a vocabulary invented here would reject input upstream accepts.
// Design D2.

#include "arch/qwen3-tts/profile.h"
#include "arch/qwen3-tts/weights.h"
#include "test-assert.h"

#include <string>

using synth::qwen3tts::create_design_profile;
using synth::qwen3tts::DesignInstruct;
using synth::qwen3tts::HParams;
using synth::qwen3tts::kMaxDesignInstructBytes;

namespace {

HParams voice_design_hparams() {
    HParams hparams;
    hparams.model_variant   = "qwen3-tts-12hz-1-7b-voicedesign";
    hparams.profile_sources = SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT;
    return hparams;
}

}  // namespace

int test_design_profile_holds_the_string_verbatim() {
    const HParams hparams = voice_design_hparams();
    DesignInstruct payload;
    const std::string instruct = "A warm, low voice, unhurried, with a slight rasp.";
    SYNTH_TEST_CHECK(create_design_profile(hparams, instruct, payload) == SYNTH_OK);
    // VERBATIM: no canonicalisation, no trimming, no reordering. Upstream
    // tokenizes whatever it is given, so anything done here would be a
    // difference from upstream that no tolerance would show.
    SYNTH_TEST_CHECK(payload.instruct == instruct);
    return 0;
}

int test_an_empty_instruct_is_accepted() {
    // Design D3: upstream's `instruct_ids.append(None)` path. An empty
    // description is a legal request for an unconditioned voice, not an error.
    const HParams hparams = voice_design_hparams();
    DesignInstruct payload;
    SYNTH_TEST_CHECK(create_design_profile(hparams, "", payload) == SYNTH_OK);
    SYNTH_TEST_CHECK(payload.instruct.empty());
    return 0;
}

int test_invalid_utf8_is_refused() {
    const HParams hparams = voice_design_hparams();
    DesignInstruct payload;
    // A lone continuation byte, and a truncated three-byte sequence.
    for (const std::string bad : { std::string("\x80"), std::string("\xE2\x82") }) {
        payload.instruct = "untouched";
        SYNTH_TEST_CHECK(create_design_profile(hparams, bad, payload) == SYNTH_ERR_INVALID_ARG);
        // On refusal the output is left alone rather than half-written.
        SYNTH_TEST_CHECK(payload.instruct == "untouched");
    }
    // And valid multi-byte UTF-8 is NOT refused -- without this the check could
    // pass by rejecting everything non-ASCII, which would reject the Chinese and
    // Japanese this package declares.
    payload.instruct.clear();
    SYNTH_TEST_CHECK(create_design_profile(hparams, "温かく低い声、少しかすれた", payload) == SYNTH_OK);
    SYNTH_TEST_CHECK(!payload.instruct.empty());
    return 0;
}

int test_an_over_long_instruct_is_refused_at_the_boundary() {
    const HParams hparams = voice_design_hparams();
    DesignInstruct payload;
    // Exactly at the bound is accepted; one byte over is refused. A test that
    // only checked a wildly long string would pass on an off-by-one.
    SYNTH_TEST_CHECK(create_design_profile(hparams, std::string(kMaxDesignInstructBytes, 'a'), payload) == SYNTH_OK);
    SYNTH_TEST_CHECK(create_design_profile(hparams, std::string(kMaxDesignInstructBytes + 1, 'a'), payload) ==
                     SYNTH_ERR_INVALID_ARG);
    return 0;
}

int main() {
    if (test_design_profile_holds_the_string_verbatim() != 0) {
        return 1;
    }
    if (test_an_empty_instruct_is_accepted() != 0) {
        return 1;
    }
    if (test_invalid_utf8_is_refused() != 0) {
        return 1;
    }
    if (test_an_over_long_instruct_is_refused_at_the_boundary() != 0) {
        return 1;
    }
    return 0;
}
```

**If `SYNTH_TEST_CHECK` or the `main()` shape differs** from what the other
qwen3-tts test files use, follow theirs — read `tests/qwen3_tts_profile_test.cpp`
first.

- [ ] **Step 2: Register the test and run it, watching it fail**

In `tests/CMakeLists.txt`, beside the other qwen3-tts unit tests:

```cmake
add_executable(synthesize-qwen3-tts-design-profile-test qwen3_tts_design_profile_test.cpp)
target_include_directories(synthesize-qwen3-tts-design-profile-test PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(synthesize-qwen3-tts-design-profile-test PRIVATE synthesize ggml)
add_test(NAME synthesize-qwen3-tts-design-profile-test COMMAND synthesize-qwen3-tts-design-profile-test)
set_tests_properties(synthesize-qwen3-tts-design-profile-test PROPERTIES LABELS "unit;qwen3-tts")
synth_register_unit_target(synthesize-qwen3-tts-design-profile-test)
```

```bash
cmake -S . -B build -DSYNTH_BUILD_TESTS=ON -DSYNTH_BUILD_PYTHON_TESTS=ON -DSYNTH_BUILD_INTEGRATION_TESTS=OFF
cmake --build build --target synthesize-qwen3-tts-design-profile-test -j 16
```

Expected: a compile error — `DesignInstruct` and `create_design_profile` do not
exist.

- [ ] **Step 3: Add the tag**

`src/voice-profile-handle.h`, inside `ProfileFamilyTag`, after `Qwen3TtsClone`:

```cpp
    // Description Text: wraps a synth::qwen3tts::DesignInstruct payload
    // (arch/qwen3-tts/profile.h). Its OWN tag rather than a third CloneMode
    // under Qwen3TtsClone, because the reasoning that let one tag cover both
    // clone modes does not reach here: those two share the x-vector and are
    // discriminated by a field inside a common struct, while a design payload
    // has no x-vector and no field in common with either.
    Qwen3TtsDesign,
```

- [ ] **Step 4: Add the payload and its constructor**

`src/arch/qwen3-tts/profile.h`:

```cpp
// The maximum instruct this family will accept, in bytes.
//
// A bound rather than a vocabulary. Upstream applies neither, so this is the
// project's own limit and exists only so a caller cannot hand the tokenizer an
// unbounded string; it is generous against the descriptions upstream's own
// examples use. NOT a semantic judgement about what makes a good description --
// design D2 rules that out.
constexpr size_t kMaxDesignInstructBytes = 4096;

// The prepared Description Text payload. One member, deliberately: upstream
// tokenizes the instruct at synthesis, so there is nothing to precompute, and
// storing token ids instead would bind the Profile to a package's frontend for
// no gain (design D1, D6).
struct DesignInstruct {
    std::string instruct;
};

// Validates a description and prepares its payload.
//
// Encoding and length, and NOTHING else. See design D2: OmniVoice's arm can
// reject on a closed attribute vocabulary because upstream defines one, and
// this family's upstream defines none, so any rule invented here would refuse
// input upstream accepts. An empty instruct is VALID (D3) -- it selects the
// unconditioned path, which is the same path Plan 1's completion gate measured.
//
// `output` is left untouched on any non-OK return.
synth_status_t create_design_profile(const HParams & hparams, const std::string & instruct, DesignInstruct & output);
```

`src/arch/qwen3-tts/profile.cpp` — place it beside `create_x_vector_profile`:

```cpp
namespace {

// Strict UTF-8, structurally. Rejects over-long encodings, surrogates and
// out-of-range code points as well as truncated sequences, because a validator
// that only checked continuation-byte counts would pass bytes the tokenizer
// then has to guess about.
bool is_well_formed_utf8(const std::string & text) {
    size_t index = 0;
    while (index < text.size()) {
        const unsigned char lead = static_cast<unsigned char>(text[index]);
        size_t              extra = 0;
        uint32_t            code  = 0;
        if (lead < 0x80) {
            index += 1;
            continue;
        } else if ((lead & 0xE0) == 0xC0) {
            extra = 1;
            code  = lead & 0x1Fu;
        } else if ((lead & 0xF0) == 0xE0) {
            extra = 2;
            code  = lead & 0x0Fu;
        } else if ((lead & 0xF8) == 0xF0) {
            extra = 3;
            code  = lead & 0x07u;
        } else {
            return false;
        }
        if (index + extra >= text.size()) {
            return false;
        }
        for (size_t step = 1; step <= extra; ++step) {
            const unsigned char continuation = static_cast<unsigned char>(text[index + step]);
            if ((continuation & 0xC0) != 0x80) {
                return false;
            }
            code = (code << 6) | (continuation & 0x3Fu);
        }
        const bool overlong = (extra == 1 && code < 0x80) || (extra == 2 && code < 0x800) ||
                              (extra == 3 && code < 0x10000);
        if (overlong || code > 0x10FFFF || (code >= 0xD800 && code <= 0xDFFF)) {
            return false;
        }
        index += extra + 1;
    }
    return true;
}

}  // namespace

synth_status_t create_design_profile(const HParams & hparams, const std::string & instruct, DesignInstruct & output) {
    (void) hparams;  // The payload does not depend on the package; the CALLER
                     // checks the variant, in voice-profile.cpp's dispatch,
                     // which is the one site holding the Model.
    if (instruct.size() > kMaxDesignInstructBytes) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (!is_well_formed_utf8(instruct)) {
        return SYNTH_ERR_INVALID_ARG;
    }
    output.instruct = instruct;
    return SYNTH_OK;
}
```

- [ ] **Step 5: Run the tests and watch them pass**

```bash
cmake --build build --target synthesize-qwen3-tts-design-profile-test -j 16
ctest --test-dir build --output-on-failure -R '^synthesize-qwen3-tts-design-profile-test$'
cmake --build build --target synthesize-check-unit -j 16
```

Expected: the new test passes and the gate shows only the two documented
baseline failures.

- [ ] **Step 6: Prove the UTF-8 check can fail**

Weaken `is_well_formed_utf8` to `return true;`, rebuild, run the test, confirm
`test_invalid_utf8_is_refused` fails, then restore and confirm `git status
--porcelain` is empty. Put the command and output in your report. A validator
nobody tried to break is a validator you have not tested.

- [ ] **Step 7: Sanitizer gate and commit**

```bash
cmake --build build-sanitize --target synthesize-check-unit -j 16
git add src/voice-profile-handle.h src/arch/qwen3-tts/profile.h src/arch/qwen3-tts/profile.cpp \
        tests/qwen3_tts_design_profile_test.cpp tests/CMakeLists.txt
scripts/ci/clang-format.sh --fix
git add -A
scripts/ci/clang-format.sh --check-diff
git status --porcelain
git commit -m "qwen3-tts: add the Description Text profile payload"
```

---

## Task 2: `create_from_description`'s Qwen3-TTS arm

**Files:**
- Modify: `src/voice-profile.cpp:1136-1165` (the dispatch)
- Modify: `tests/qwen3_tts_design_profile_test.cpp`

**Interfaces:**
- Consumes: Task 1's `DesignInstruct`, `create_design_profile`,
  `ProfileFamilyTag::Qwen3TtsDesign`.
- Produces: `synth_voice_profile_create_from_description` returning `SYNTH_OK`
  and a profile whose `family_tag` is `Qwen3TtsDesign` for a VoiceDesign model;
  `SYNTH_ERR_UNSUPPORTED_VOICE` for a CustomVoice or Base model.

**Read first:** `src/voice-profile.cpp:1146`, the line that currently routes
every non-OmniVoice family to `validate_unsupported_params`. Also read
`create_from_reference`'s dispatch above it: it gates on the **published
capability bit**, not on the family alone, and carries a comment explaining why
the family check is not enough. Your arm follows that shape — but note the
ordering problem it creates, below.

**The ordering problem, stated because it will bite you.** Plan 1 deliberately
set `source_flags = 0` for this variant, so a bit-gated dispatch would refuse
every request until Task 5 republishes it. Do **not** solve this by republishing
early — Task 5 owns that, and doing it here would advertise the capability
before serialization exists. Gate this arm on the package's **declared** sources
(`HParams::profile_sources`, which does contain `DESCRIPTION_TEXT`), and note in
a comment that the published bit follows in Task 5. The declared set is the right
gate anyway: it says what the package can implement, which is what a constructor
needs to know.

- [ ] **Step 1: Write the failing tests**

Append to `tests/qwen3_tts_design_profile_test.cpp`. These use the real public
entry point, so they need a Model; if the file has no Model fixture, follow
whatever `tests/qwen3_tts_voice_required_test.cpp` does to build one and say in
your report which approach you used.

```cpp
// The public seam. A Model whose package declares description-text gets a
// Profile; the other two variants get the unsupported-source refusal, which is
// the same answer they gave before this arm existed and must keep giving.
int test_create_from_description_accepts_a_voicedesign_model() {
    // Build a Model whose HParams declare DESCRIPTION_TEXT, call
    // synth_voice_profile_create_from_description with a valid description,
    // and assert: SYNTH_OK, non-null profile, and
    // profile->family_tag == synth::ProfileFamilyTag::Qwen3TtsDesign.
    // Then free it and assert no leak path was taken (synth_voice_profile_free
    // is safe on null).
}

int test_create_from_description_refuses_the_clone_variants() {
    // A Model declaring REFERENCE_AUDIO (Base) and a preset-catalog Model
    // (CustomVoice) must BOTH return the unsupported-source status, and must
    // leave *out_profile null.
}

int test_create_from_description_refuses_a_malformed_description() {
    // Invalid UTF-8 through the public entry point returns SYNTH_ERR_INVALID_ARG
    // rather than reaching the payload -- this pins that Task 1's validation is
    // actually wired into the dispatch and not bypassed.
}
```

**Write the bodies out in full** using the fixture style the neighbouring test
file uses. The comments above state exactly what each must assert; do not leave
them as comments.

- [ ] **Step 2: Run and watch them fail**

```bash
cmake --build build --target synthesize-qwen3-tts-design-profile-test -j 16
ctest --test-dir build --output-on-failure -R '^synthesize-qwen3-tts-design-profile-test$'
```

Expected: the accept case fails with the unsupported status, because the family
dispatch still sends every non-OmniVoice model to the fallback.

- [ ] **Step 3: Add the arm**

In `src/voice-profile.cpp`, replace the family check at line 1146 so OmniVoice
keeps its path and Qwen3-TTS gains one. Leave every other family on the fallback:

```cpp
    if (model->info.family == synth::ModelFamily::Qwen3Tts) {
        if (params == nullptr) {
            return SYNTH_ERR_INVALID_ARG;
        }
        if (params->struct_size < sizeof(uint64_t)) {
            return SYNTH_ERR_BAD_STRUCT_SIZE;
        }
        try {
            return create_qwen3_tts_profile_from_description(model, params, out_profile);
        } catch (const std::bad_alloc &) {
            return SYNTH_ERR_OUT_OF_MEMORY;
        }
    }
    if (model->info.family != synth::ModelFamily::Omnivoice) {
        return validate_unsupported_params(params, offsetof(synth_voice_description_params_t, diagnostics));
    }
```

and add the helper beside `create_omnivoice_profile_from_description`:

```cpp
// GATED ON THE PACKAGE'S DECLARED SOURCES, NOT THE PUBLISHED BIT, and that is
// temporary. Plan 1 deliberately set source_flags = 0 for this variant so the
// seam would not advertise a capability it refused; Plan 2 Task 5 republishes
// the bit once serialization exists. Until then a bit-gated dispatch would
// refuse every request. The declared set is the honest gate for a constructor
// either way: it says what the package can implement.
synth_status_t create_qwen3_tts_profile_from_description(const synth_model_t *                    model,
                                                         const synth_voice_description_params_t * params,
                                                         synth_voice_profile_t **                 out_profile) {
    const auto & hparams = model->qwen3_tts->hparams();
    if ((hparams.profile_sources & SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT) == 0) {
        return SYNTH_ERR_UNSUPPORTED_VOICE;
    }
    const std::string instruct = params->description == nullptr ?
                                     std::string() :
                                     std::string(params->description, params->description_size);
    auto payload = std::make_shared<synth::qwen3tts::DesignInstruct>();
    const synth_status_t status = synth::qwen3tts::create_design_profile(hparams, instruct, *payload);
    if (status != SYNTH_OK) {
        return status;
    }
    auto handle         = std::make_unique<synth_voice_profile>();
    handle->family_tag  = synth::ProfileFamilyTag::Qwen3TtsDesign;
    handle->payload     = std::move(payload);
    *out_profile        = handle.release();
    return SYNTH_OK;
}
```

**Check `synth_voice_profile`'s real member names and how
`create_from_reference` fills them** (`src/voice-profile.cpp:701` sets the clone
tag) before copying this — it may carry more fields, and a half-filled handle is
worse than a compile error.

- [ ] **Step 4: Run and watch them pass**

```bash
cmake --build build --target synthesize-qwen3-tts-design-profile-test -j 16
ctest --test-dir build --output-on-failure -R '^synthesize-qwen3-tts-design-profile-test$'
cmake --build build --target synthesize-check-unit -j 16
```

- [ ] **Step 5: Sanitizer gate and commit**

```bash
cmake --build build-sanitize --target synthesize-check-unit -j 16
git add src/voice-profile.cpp tests/qwen3_tts_design_profile_test.cpp
scripts/ci/clang-format.sh --fix
git add -A
scripts/ci/clang-format.sh --check-diff
git status --porcelain
git commit -m "qwen3-tts: wire create_from_description for the VoiceDesign variant"
```

---

## Task 3: Serialization, without loosening the x-vector check

**Files:**
- Modify: `src/arch/qwen3-tts/profile.h`, `src/arch/qwen3-tts/profile.cpp`
- Modify: `src/voice-profile.cpp` (the serialize and load dispatch arms)
- Modify: `tests/qwen3_tts_design_profile_test.cpp`

**Interfaces:**
- Consumes: Task 1's payload and tag.
- Produces:
  - `synth_status_t serialize_design_profile(const DesignInstruct & profile, const HParams & hparams, std::vector<uint8_t> & output);`
  - `load_profile_from_memory` accepting a design envelope and setting
    `out_family_tag = ProfileFamilyTag::Qwen3TtsDesign`.

**Read first, because this is the subtlety of the whole plan:**
`load_profile_from_memory` in `src/arch/qwen3-tts/profile.cpp` (around line 1392)
refuses `tensor_bytes == 0` and then requires `element_count == enc_dim`. On a
VoiceDesign package `enc_dim` is **0**, because no speaker encoder is read — so
those two branches are mutually exclusive and **no envelope of any kind can
currently load**. That is not an oversight: Plan 1's `source_flags = 0` decision
rests on exactly this, and a final reviewer verified it.

So this task must make a **design** envelope loadable **without** weakening the
x-vector size check that protects clone envelopes. Route on the envelope's own
declared kind before either size check runs, and keep the x-vector path's
`element_count == enc_dim` exactly as it is. If you find yourself relaxing that
comparison, stop and report — you are about to reopen the hole Plan 1 closed.

- [ ] **Step 1: Write the failing round-trip tests**

Append to `tests/qwen3_tts_design_profile_test.cpp`, with full bodies:

```cpp
// Round-trip: serialize a design Profile, load the bytes back, and get the same
// string with the same tag. Per design D6 the envelope stores the TEXT and the
// loader re-tokenizes -- there are no ids in it to drift.
int test_design_profile_round_trips();

// A clone envelope must not load as a design Profile and a design envelope must
// not load as a clone. The tag is what a consumer switches on, so a
// misidentified payload is a type confusion, not a wrong answer.
int test_the_two_envelope_kinds_are_not_confusable();

// The x-vector size check is UNCHANGED: a clone envelope whose element count
// disagrees with enc_dim is still refused. Pin it here because this task edits
// the function that performs it.
int test_a_clone_envelope_with_the_wrong_size_is_still_refused();

// An empty instruct round-trips as an empty instruct, not as an absent field.
int test_an_empty_instruct_round_trips();
```

- [ ] **Step 2: Run and watch them fail**

Expected: `serialize_design_profile` does not exist.

- [ ] **Step 3: Implement, following the existing envelope**

Read `serialize_x_vector_profile` and `serialize_icl_profile` and use the same
envelope framing, header fields and compatibility-id handling. The design
envelope's schema string is **`qwen3-tts-voice-design`**, which is what the
converter writes for this variant and what the loader validates — do not invent
a third name.

Then extend `load_profile_from_memory` to dispatch on the envelope's declared
kind **before** the tensor-size branches, so a design envelope never reaches the
`enc_dim` comparison and a clone envelope still does.

- [ ] **Step 4: Run and watch them pass, then prove two things can fail**

Both injections, with the command and output in your report:

1. Make the design load path accept a clone envelope. Confirm
   `test_the_two_envelope_kinds_are_not_confusable` fails. Restore.
2. Change the x-vector path's `element_count == enc_dim` to `>=`. Confirm
   `test_a_clone_envelope_with_the_wrong_size_is_still_refused` fails. Restore.

The second matters most: it is the check whose strength Plan 1's decision rests
on, and this is the task that could weaken it by accident.

- [ ] **Step 5: Wire the dispatch arms**

`src/voice-profile.cpp:1278`'s serialize dispatch and the load dispatch below it
both switch on the family tag. Add the design arm to each.

- [ ] **Step 6: Sanitizer gate and commit**

```bash
cmake --build build-sanitize --target synthesize-check-unit -j 16
git add src/arch/qwen3-tts/profile.h src/arch/qwen3-tts/profile.cpp src/voice-profile.cpp \
        tests/qwen3_tts_design_profile_test.cpp
scripts/ci/clang-format.sh --fix
git add -A
scripts/ci/clang-format.sh --check-diff
git commit -m "qwen3-tts: serialize a Description Text Profile without weakening the clone envelope's size check"
```

---

## Task 4: The instruct block reaches the prompt

**Files:**
- Modify: `src/arch/qwen3-tts/talker-host.h:63` (`TalkerPromptRequest`), `talker-host.cpp`
- Modify: `src/arch/qwen3-tts/model.cpp:979` and its prompt assembly
- Modify: `src/synthesize.cpp:1092` (the tag branch)
- Modify: `tests/qwen3_tts_prompt_slot_test.cpp`
- Modify: `scripts/dump_reference_qwen3_tts_voicedesign.py`, `tests/qwen3_tts_voicedesign_prefill_real.cpp`, `tests/tolerances/qwen3-tts.json`

**Interfaces:**
- Consumes: Task 2's Profile, Task 1's tag.
- Produces: `TalkerPromptRequest::instruct_tokens` (`std::vector<uint32_t>`),
  placed as text-only positions before `role_tokens`; `has_speaker == false`
  whenever the request carries a design Profile.

**Read first:** `src/synthesize.cpp:1085-1091`. Its comment predicted this task:

> a future `family_tag` added under a shared Model type must not fall through
> and have its payload misread as an `XVectorProfile`

Today that check refuses any tag that is not `Qwen3TtsClone`. Your branch must
send `Qwen3TtsDesign` down the instruct path instead of refusing it — and must
keep refusing genuinely foreign tags.

Also read `talker-host.h:62`: the chat template and byte-pair merges belong to the
**text frontend**, not the prompt builder. The wrapping
`<|im_start|>user\n{instruct}<|im_end|>\n` (`qwen3_tts_model.py:276`) is
tokenized on the frontend side; `instruct_tokens` arrives already wrapped.

- [ ] **Step 1: Write the failing prompt test**

Extend `tests/qwen3_tts_prompt_slot_test.cpp` with a case asserting that, given
`instruct_tokens`, the prefill's leading positions are exactly those tokens as
`Text::Token` with `has_codec == false`, that `role_tokens` follow them, that
`codec_offset` has shifted by the block's length, and that the graph invariant
`codec_offset + codec_tokens->ne[0] == text->ne[1]` still holds.

Pin the **values**, not only the counts — Plan 1's own review found a
count-only version of this test surviving a hardcoded-offset break.

- [ ] **Step 2: Run and watch it fail. Step 3: implement. Step 4: run and watch it pass.**

`TalkerPromptRequest` gains `std::vector<uint32_t> instruct_tokens;`.
`build_talker_prompt` prepends them; `TalkerInputPosition` needs no change.
`model.cpp` sets `has_speaker = false` when the request carries a design Profile
and threads the tokens in; `synthesize.cpp` branches the new tag to that path.

- [ ] **Step 5: Extend the oracle comparison to a non-empty instruct**

The dumper and the driver both take an empty instruct today. Give them a
non-empty one, re-dump, and add a second probe cell beside Plan 1's — keep the
empty-instruct cell, because it is the control that shows the instruct block is
what moved the number.

Record both, and a fault-injection figure for the new one: drop the instruct
block and confirm the comparison breaches. **If dropping it does not breach, say
so loudly** — that would mean the instruct is not reaching the prefill, which is
the silent failure design §6 is built around.

- [ ] **Step 6: Sanitizer gate and commit.**

---

## Task 5: Republish the capability bit

**Files:**
- Modify: `src/arch/qwen3-tts/weights.cpp` (`fill_voice_profile_capability`)
- Modify: `tests/qwen3_tts_voice_required_test.cpp`
- Modify: the plan and design errata that recorded the withholding

**Interfaces:**
- Consumes: Tasks 2 and 3 — the seam exists and Profiles serialize.
- Produces: `source_flags == SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT | SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE`
  for a VoiceDesign package.

**Read first:** Plan 1's errata (its Task 4, Task 6 Step 5 and Completion Gate)
and the design's D4/§3 errata. They record the bit being withheld and say Plan 2
turns it back on. **Amend those errata rather than deleting them** — the record
of why it was withheld is the reason the seam got built before the advertisement.

`SERIALIZED_PROFILE` becomes honest only because Task 3 made a design envelope
loadable. If Task 3 is not done, this task cannot be either.

- [ ] **Steps:** write the failing assertion; watch it fail; change the function;
  watch it pass; inject the old `source_flags = 0` and confirm the assertion
  fails; restore; amend the four errata; sanitizer gate; commit.

---

## Task 6: `--description` in the public-seam validator

**Files:**
- Modify: `scripts/validate-qwen3-tts-public.py`, `tests/qwen3_tts_public_real.c`
- Modify: `tests/tolerances/qwen3-tts.json` (the `public` cell for this variant)

**Interfaces:**
- Consumes: Tasks 2–5.
- Produces: `--description` accepted twice, mirroring `--reference`'s shape.

**Read first:** how `--reference` was added — `parse_args`, `resolve_max_frames`,
and the `voice_kind` branch. `--description` follows it, including that **two**
are required: the relation "two different descriptions give different audio"
needs two, and one would silently drop it.

Implement design §6.3's three relations and §6.4's four refusals:

1. Two different instructs, same seed, same text → the audio **differs**, and
   the byte difference is confirmed **before** the verdict is recorded.
2. The same instruct and seed → **byte-identical**. Exact; both sides are this
   port, so nothing licenses a tolerance.
3. An empty instruct reproduces upstream's `instruct=""` output within the
   `replay` stage's tolerance — a port-against-oracle comparison, stated
   separately from relation 2 because it is a different kind of claim.

Refusals: the VoiceDesign package refuses `create_from_reference`; CustomVoice and
Base refuse `create_from_description`; a package claiming `DESCRIPTION_TEXT`
without the variant behind it is refused at load — **note that design §6.4's
third refusal carries a Plan 1 erratum saying it is not implemented and arguably
cannot be, since Description Text has no distinguishing tensor. Read that erratum
and either implement what is implementable or extend it; do not silently drop the
line.** The fourth, retaining the speaker slot, is caught by Task 4's probe.

- [ ] Steps as in the other tasks: failing checks first, then the flag, then the
  relations, then the refusals, then fault injection on relation 1 (make both
  instructs produce the same audio and confirm the relation fails).

---

## Task 7: The manifest's cases, the schema, and the `suite_status` guard

**Files:**
- Modify: `tests/golden/qwen3-tts/qwen3-tts-12hz-1-7b-voicedesign.manifest.json`
- Modify: `docs/schemas/synthesize-golden-manifest-v1.schema.json`
- Modify: `tests/python/test_golden_manifests.py`
- Modify: `docs/port-validation.md`, `docs/porting/families/qwen3-tts.md`

**Interfaces:**
- Consumes: Tasks 4 and 6 — the cases are the measurements those produced.
- Produces: a manifest with at least 12 cases and **no** `suite_status`.

Three things, and the third is carried in from Plan 1's final review.

**The cases.** Fill the manifest from Tasks 4 and 6. When the count reaches 12,
**remove `suite_status: "incremental"`** — the flag exists only for the interval
this plan closes. Removing it re-engages the 12-case floor and the ~14 checks in
`test_golden_manifests.py` that iterate `cases` and are vacuous while it is empty.

**The schema cannot express this variant's contract.** `$defs.profileContract`
has `"sources": {"enum": ["reference_audio", "serialized_profile"]}` and
`"required": [..., "reference"]`. Add `description_text` to the enum and make
`reference` conditional on `reference_audio` being among `sources`, so the
manifest can finally carry `package_contract.profile` for this variant. Then add
it. Until now this variant's declared sources lived only in a doc table and an
uncommitted probe.

**`suite_status` has no policy guard, and the two mechanisms do not know about
each other.** Verified while writing this plan: `suite_status` appears **only in
the schema**. `tests/python/test_golden_manifests.py` never reads it — both of
its exemptions are keyed on `if not manifest["cases"]` (lines 203 and 309). So
the schema waives the 12-case floor for whoever sets the flag, while the Python
checks waive themselves for whoever has no cases, and nothing ties the two
together: a manifest can carry the flag WITH cases, or have no cases WITHOUT the
flag, and both pass. A reviewer also showed a 3-case VITS manifest carrying the
flag validates.

Add the assertion the ledger named, and make it bind both directions: the set of
manifests carrying `suite_status` equals a named allowlist, AND a manifest
carrying it has no cases while a manifest without it has at least twelve. After
this task the allowlist is **empty**, which is the strongest form the assertion
can take.

Prove all three: truncate a manifest and confirm rejection; validate all
manifests; and add `suite_status` to one that should not have it and confirm the
new guard fails.

---

## Completion Gate

1. A sentence of description in, audio out, through the public C interface.
2. The instruct prefill meets the oracle tolerance, with a recorded
   fault-injection figure showing the comparison fails when the instruct block is
   dropped.
3. Design §6.3's three relations and §6.4's implementable refusals hold in
   `validate-qwen3-tts-public.py`.
4. The manifest carries at least 12 cases and no `suite_status`; the schema
   expresses this variant's profile contract; the allowlist guard is in place and
   empty.
5. `source_flags` publishes `DESCRIPTION_TEXT | SERIALIZED_PROFILE`.

Not delivered here, and not a gap: quantization profiles, the CUDA placement
measurement, the listening audit and the model card — Plan 3.
