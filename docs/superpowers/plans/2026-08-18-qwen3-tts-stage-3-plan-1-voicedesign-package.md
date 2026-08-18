# Qwen3-TTS Stage 3 Plan 1 — The VoiceDesign Package — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` convert, load, and synthesize
at an empty instruct with output matching the PyTorch oracle — the first rung of
Stage 3, before any Description Text seam exists.

**Architecture:** No new graph is written. The checkpoint's talker, code
predictor and codec decoder are Stage 1's graph set at doubled dimensions, and
the port already reads every dimension from GGUF metadata. Three things change:
the converter learns a third variant, the loader learns that a package declares
*which* Voice Profile sources it implements (instead of inferring them from the
Voice Mode), and the talker prompt learns the case where the speaker slot is
absent rather than substituted.

**Tech Stack:** C++17 + GGML/GGUF, CMake ≥3.24, CTest, Python via `uv` in
`scripts/envs/qwen3-tts/` (numpy 1.26.4, torch), pinned clang-format 22.1.5.

## Global Constraints

- Design of record: `docs/superpowers/specs/2026-08-18-qwen3-tts-stage-3-design.md`.
  Where execution contradicts it, add an erratum **to that document** in the
  section it corrects, leaving the original prescription standing above it.
- Upstream checkpoint is pinned at revision
  `5ecdb67327fd37bb2e042aab12ff7391903235d3`, Apache-2.0. Verify the licence
  from the **upstream model card**, never from a port's README.
- Every slice is done only when its focused unit tests are committed, passing,
  and registered with CTest under the `unit` label (`docs/testing.md`).
- Commit golden contracts, not golden payloads. No `.gguf`, `.wav`, `.f32` or
  oracle dump is ever committed.
- Sanitizer gate for every slice touching C/C++:
  `cmake -S . -B build-sanitize -DSYNTH_BUILD_TESTS=ON -DSYNTH_BUILD_PYTHON_TESTS=ON -DSYNTH_BUILD_INTEGRATION_TESTS=OFF -DSYNTH_SANITIZE=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo`
  then `cmake --build build-sanitize --target synthesize-check-unit`.
- `scripts/ci/clang-format.sh --fix` must run **after** `git add` — it skips
  untracked files, a trap Plan 1 of Stage 2 and Plan 4 both fell into.
- Two unit tests fail at baseline on gitignored VITS artifacts and are **not**
  yours: `synthesize-python-api-wheel-test`, `synthesize-vits-python-unit`.
  Record their state before you start; any third failure is yours.
- A `dev-*` preset proves correctness, never timing. No performance figure from
  a `RelWithDebInfo` tree.
- Never use bare `git stash` / `git stash pop` — the stash stack is shared with
  other worktrees.
- Publication is out of scope for this plan and for Stage 3 Plans 1–2.

---

## File Structure

**Modified:**

- `scripts/convert-qwen3-tts.py` — `variant_profile()` gains a third arm;
  the profile-contract metadata block splits so a package without a speaker
  encoder still declares a Profile schema.
- `tests/python/test_convert_qwen3_tts.py` — `VariantDiscriminationTests` gains
  voice_design cases; one existing test **inverts**.
- `src/arch/qwen3-tts/weights.h` — `HParams` gains a declared profile-source set.
- `src/arch/qwen3-tts/weights.cpp` — reads that set; splits
  `read_profile_and_speaker_encoder`; `fill_voice_profile_capability` publishes
  from the declared set rather than from a hardcoded pair.
- `tests/qwen3_tts_metadata_test.cpp` — synthetic packages for the new shape.
- `tests/qwen3_tts_voice_required_test.cpp` — adversarial capability cases.
- `src/arch/qwen3-tts/talker-host.cpp` — the no-speaker prefill layout.
- `tests/qwen3_tts_prompt_slot_test.cpp` — the absent-slot prompt cases.
- `docs/porting/families/qwen3-tts.md` — Stage 3 opening, the silent
  instruct-drop hazard, and the intake record.

**Created:**

- `tests/golden/qwen3-tts/qwen3-tts-12hz-1-7b-voicedesign.manifest.json`
- `scripts/dump_reference_qwen3_tts_voicedesign.py`
- `tests/qwen3_tts_voicedesign_prefill_real.cpp` (integration driver)

---

## Task 1: The converter accepts a third variant

**Files:**
- Modify: `scripts/convert-qwen3-tts.py:212-238` (`variant_profile`)
- Test: `tests/python/test_convert_qwen3_tts.py:226-250` (`VariantDiscriminationTests`)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `convert.variant_profile(config) -> VariantProfile` accepting
  `tts_model_type == "voice_design"`, returning
  `VariantProfile(model_type="voice_design", carries_speaker_encoder=False,
  carries_codec_encoder=False, display_name="Qwen3-TTS 12Hz 1.7B VoiceDesign",
  size_label="1.7B")`.

**Read first:** `scripts/convert-qwen3-tts.py:212-238`. The function resolves
what a checkpoint carries from what it *declares*, then cross-checks the
declaration against the config, so a new variant cannot silently convert as
whichever branch it fell into.

- [ ] **Step 1: Invert the test that currently uses `voice_design` as its example of an unknown type**

`tests/python/test_convert_qwen3_tts.py` today contains this, and it is about to
become false. It is not a stale test — it is the guard that has been keeping an
unsupported variant out, and its *example* must move to a genuinely unknown type
rather than the test being deleted:

```python
    def test_an_unknown_model_type_is_refused(self) -> None:
        with self.assertRaises(convert.ConverterError):
            convert.variant_profile({"tts_model_type": "voice_design", "tts_model_size": "1b7"})
```

Replace it, and add the three voice_design cases, inside
`class VariantDiscriminationTests`:

```python
    def test_an_unknown_model_type_is_refused(self) -> None:
        # `voice_design` was this test's example until Stage 3 Plan 1 made it a
        # supported variant. The guard is what matters, not the example, so it
        # moved to a type upstream does not ship.
        with self.assertRaises(convert.ConverterError):
            convert.variant_profile({"tts_model_type": "dialogue", "tts_model_size": "1b7"})

    def test_voice_design_config_carries_neither_encoder(self) -> None:
        profile = convert.variant_profile({"tts_model_type": "voice_design", "tts_model_size": "1b7"})
        self.assertEqual(profile.model_type, "voice_design")
        self.assertFalse(profile.carries_speaker_encoder)
        self.assertFalse(profile.carries_codec_encoder)
        self.assertEqual(profile.size_label, "1.7B")

    def test_voice_design_config_with_a_speaker_encoder_is_refused(self) -> None:
        # The check runs in the direction CustomVoice's does: a voice_design
        # checkpoint that carried an encoder would be a different model than
        # the one this arm was written against.
        with self.assertRaises(convert.ConverterError) as caught:
            convert.variant_profile({
                "tts_model_type": "voice_design",
                "tts_model_size": "1b7",
                "speaker_encoder_config": {"enc_dim": 1024, "sample_rate": 24000},
            })
        self.assertIn("speaker_encoder_config", str(caught.exception))
```

- [ ] **Step 2: Run the tests and watch them fail**

```bash
ctest --test-dir build --output-on-failure -R '^synthesize-qwen3-tts-python-convert$' 2>&1 | tail -20
```

If that target name does not exist, find it with
`ctest --test-dir build -N | grep -i convert` and use the qwen3-tts one.

Expected: the two new voice_design tests fail with
`ConverterError: unsupported tts_model_type 'voice_design'`.

- [ ] **Step 3: Add the third arm**

In `scripts/convert-qwen3-tts.py`, inside `variant_profile()`, immediately
before the final `raise`:

```python
    if model_type == "voice_design":
        # No encoder of either kind: this checkpoint has no speaker_encoder_config
        # and there is no reference audio on its path, so the tensor set is
        # CustomVoice's shape rather than Base's. See the Stage 3 design, section 4.
        if has_encoder_config:
            raise ConverterError(
                "config declares tts_model_type=voice_design but carries a speaker_encoder_config"
            )
        return VariantProfile("voice_design", False, False,
                              "Qwen3-TTS 12Hz 1.7B VoiceDesign", "1.7B")
```

Also update the docstring of `variant_profile()`: it says "The two supported
variants", which is about to be three.

- [ ] **Step 4: Run the tests and watch them pass**

```bash
ctest --test-dir build --output-on-failure -R 'convert' 2>&1 | tail -20
```

Expected: PASS, including the two pre-existing base/custom_voice cases.

- [ ] **Step 5: Commit**

```bash
git add scripts/convert-qwen3-tts.py tests/python/test_convert_qwen3_tts.py
scripts/ci/clang-format.sh --fix
git add -A
git commit -m "qwen3-tts: teach the converter the voice_design variant"
```

---

## Task 2: A package without a speaker encoder still declares a Profile schema

**Files:**
- Modify: `scripts/convert-qwen3-tts.py:730-750` (the profile-contract block)
- Test: `tests/python/test_convert_qwen3_tts.py`

**Interfaces:**
- Consumes: `VariantProfile.model_type` from Task 1.
- Produces: GGUF metadata keys — `synthesize.voice.profile_sources` (string
  array), `synthesize.profile.schema` (`"qwen3-tts-voice-clone"` or
  `"qwen3-tts-voice-design"`), `synthesize.profile.schema_version` (1),
  `synthesize.profile.compatibility_id` (64 hex chars). The
  `synthesize.reference.*` and `synthesize.qwen3-tts.speaker_encoder.*` blocks
  are written **only** for a variant that carries a speaker encoder.

**Read first:** `scripts/convert-qwen3-tts.py:730-750`. The whole
profile-contract block sits under `if profile.carries_speaker_encoder:`. A
VoiceDesign package needs the schema and compatibility id (it can serialize a
Profile) but has no reference limits and no encoder metadata to write.

- [ ] **Step 1: Write the failing test**

Add to `tests/python/test_convert_qwen3_tts.py`. `profile_source_names` is a
new helper the converter exposes so this test does not have to reimplement the
mapping:

```python
class ProfileSourceDeclarationTests(unittest.TestCase):
    """A package declares which Profile sources it implements, positively.

    Before Stage 3 the runtime inferred this from the Voice Mode, which worked
    only while `profile-sources` meant exactly one variant. It now means two,
    whose sources differ, so the package has to say.
    """

    def test_base_declares_reference_audio(self) -> None:
        profile = convert.variant_profile({
            "tts_model_type": "base",
            "tts_model_size": "0b6",
            "speaker_encoder_config": {"enc_dim": 1024, "sample_rate": 24000},
        })
        self.assertEqual(convert.profile_source_names(profile), ["reference-audio"])
        self.assertEqual(convert.profile_schema_name(profile), "qwen3-tts-voice-clone")

    def test_voice_design_declares_description_text(self) -> None:
        profile = convert.variant_profile({"tts_model_type": "voice_design", "tts_model_size": "1b7"})
        self.assertEqual(convert.profile_source_names(profile), ["description-text"])
        self.assertEqual(convert.profile_schema_name(profile), "qwen3-tts-voice-design")

    def test_custom_voice_declares_none(self) -> None:
        # A preset-catalog package prepares nothing, so it carries no contract
        # at all -- not an empty one, which is a different claim.
        profile = convert.variant_profile({"tts_model_type": "custom_voice", "tts_model_size": "0b6"})
        self.assertEqual(convert.profile_source_names(profile), [])
        self.assertIsNone(convert.profile_schema_name(profile))
```

- [ ] **Step 2: Run it and watch it fail**

```bash
ctest --test-dir build --output-on-failure -R 'convert' 2>&1 | tail -20
```

Expected: `AttributeError: module has no attribute 'profile_source_names'`.

- [ ] **Step 3: Add the two helpers and split the metadata block**

In `scripts/convert-qwen3-tts.py`, after `variant_profile()`:

```python
def profile_source_names(profile: VariantProfile) -> list[str]:
    """Which Voice Profile sources this variant implements.

    Named rather than derived at the read side, because after Stage 3 the Voice
    Mode no longer determines this: `profile-sources` covers both Base, which
    clones from a recording, and VoiceDesign, which cannot clone at all. The
    runtime refuses a package whose declared sources do not match the blocks it
    carries, so this is a claim the package has to earn.
    """
    if profile.carries_speaker_encoder:
        return ["reference-audio"]
    if profile.model_type == "voice_design":
        return ["description-text"]
    return []


def profile_schema_name(profile: VariantProfile) -> str | None:
    """The Profile Schema this variant serializes under, or None if it prepares nothing."""
    if profile.carries_speaker_encoder:
        return "qwen3-tts-voice-clone"
    if profile.model_type == "voice_design":
        return "qwen3-tts-voice-design"
    return None
```

Then replace the block at line 730 (`if profile.carries_speaker_encoder:` down
to the `compatibility_id(...)` call) with:

```python
    sources = profile_source_names(profile)
    schema = profile_schema_name(profile)
    if sources:
        writer.add_array("synthesize.voice.profile_sources", sources)
        writer.add_string("synthesize.profile.schema", schema)
        writer.add_uint32("synthesize.profile.schema_version", 1)
        writer.add_string(
            "synthesize.profile.compatibility_id",
            compatibility_id(schema, 1, (digests["talker"], digests["codec"], digests["config"])),
        )
    # The reference limits and the encoder's own front-end contract describe
    # REFERENCE AUDIO. A package that cannot take a recording has nothing to
    # say here, and writing zeros would be a claim rather than a silence.
    if profile.carries_speaker_encoder:
        reference = package["profile"]["reference"]
        for key in ("target_sample_rate", "target_channels"):
            writer.add_uint32(f"synthesize.reference.{key}", int(reference[key]))
        for key in ("min_frames_per_clip", "max_frames_per_clip",
                    "max_total_frames", "max_reference_count"):
            writer.add_uint64(f"synthesize.reference.{key}", int(reference[key]))
        for key, value in speaker_encoder_metadata(config["speaker_encoder_config"]).items():
            if isinstance(value, float):
                writer.add_float32(f"synthesize.qwen3-tts.speaker_encoder.{key}", value)
            else:
                writer.add_uint32(f"synthesize.qwen3-tts.speaker_encoder.{key}", int(value))
```

Note the compatibility id now hashes the **schema name**, which differs between
the two variants — so a clone Profile can never be mistaken for a design one
even if the three digests collided.

- [ ] **Step 4: Run the tests and watch them pass**

```bash
ctest --test-dir build --output-on-failure -R 'convert' 2>&1 | tail -20
```

Expected: PASS. The Base path must be **unchanged** — its schema string, its
version, and its reference keys are all the same values in the same order.

- [ ] **Step 5: Commit**

```bash
git add scripts/convert-qwen3-tts.py tests/python/test_convert_qwen3_tts.py
git commit -m "qwen3-tts: declare a package's Profile sources positively"
```

---

## Task 3: The loader reads the declared source set

**Files:**
- Modify: `src/gguf-metadata.h:35-44`, `src/gguf-metadata.cpp` — add a `has()` presence check
- Modify: `src/arch/qwen3-tts/weights.h:170-207` (`HParams`)
- Modify: `src/arch/qwen3-tts/weights.cpp:475-500` (`read_profile_contract`), `:695-718` (`read_profile_and_speaker_encoder`)
- Test: `tests/qwen3_tts_metadata_test.cpp`

**Verified API surface** (do not substitute anything else): `GgufMetadata`
already has `string_array(key, std::vector<std::string> &)` and `u32`, `u64`,
`f32`, `boolean`, `string` — see `src/gguf-metadata.h:35-44`. It has **no**
presence check, so this task adds one. On the C side, `gguf_find_key`,
`gguf_remove_key`, `gguf_set_val_bool` and `gguf_set_arr_str` all exist
(`ggml/include/gguf.h:99,136,149,156`).

**Interfaces:**
- Consumes: the metadata keys Task 2 produces.
- Produces: `HParams::profile_sources` — a `uint32_t` bitmask using the public
  `SYNTH_PROFILE_SOURCE_*` values from `include/synthesize.h`. Task 4 reads it.

**Read first:** `src/arch/qwen3-tts/weights.cpp:695-718`. The comment there
explains why the pair is gated on the Voice Mode rather than on key presence,
and that reasoning survives Stage 3 — what changes is that the mode no longer
says *which* blocks to demand.

- [ ] **Step 1: Write the failing tests**

`tests/qwen3_tts_metadata_test.cpp` already builds synthetic packages in memory
via `valid_metadata()`; follow that shape. Add:

```cpp
// Stage 3: a package declares which Profile sources it implements, and the
// loader demands exactly the blocks that declaration implies. Before this,
// `profile-sources` meant Base and therefore meant reference audio; it now
// means two variants whose sources are disjoint.
int test_profile_sources_are_declared_not_inferred() {
    // A voice_design package: description-text, no encoder block, no reference
    // limits. This must LOAD -- before Stage 3 it failed on the missing
    // speaker-encoder keys.
    {
        GgufContext c = voice_design_metadata();
        HParams     hparams;
        SYNTH_TEST_CHECK(read_hparams(c.get(), hparams) == SYNTH_OK);
        SYNTH_TEST_CHECK(hparams.voice_mode == VoiceMode::ProfileSources);
        SYNTH_TEST_CHECK(!hparams.has_speaker_encoder);
        SYNTH_TEST_CHECK(hparams.profile_sources == SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT);
        SYNTH_TEST_CHECK(hparams.profile.schema == "qwen3-tts-voice-design");
    }
    // Declaring description-text while carrying a speaker encoder is a package
    // that disagrees with itself. Refused, not reconciled.
    {
        GgufContext c = voice_design_metadata();
        gguf_set_val_u32(c.get(), "synthesize.qwen3-tts.speaker_encoder.enc_dim", 2048);
        HParams hparams;
        SYNTH_TEST_CHECK(read_hparams(c.get(), hparams) != SYNTH_OK);
    }
    // Declaring reference-audio without the encoder block is the same fault
    // from the other side, and was already impossible; it must stay so.
    {
        GgufContext c = voice_design_metadata();
        set_string_array(c.get(), "synthesize.voice.profile_sources", { "reference-audio" });
        HParams hparams;
        SYNTH_TEST_CHECK(read_hparams(c.get(), hparams) != SYNTH_OK);
    }
    // An unknown source name is refused rather than ignored: a package from a
    // future converter must not load with a silently narrower capability.
    {
        GgufContext c = voice_design_metadata();
        set_string_array(c.get(), "synthesize.voice.profile_sources", { "telepathy" });
        HParams hparams;
        SYNTH_TEST_CHECK(read_hparams(c.get(), hparams) != SYNTH_OK);
    }
    // A truncated package: profile-sources mode with the declaration missing
    // entirely. It says every request must carry a Profile and offers no way
    // to make one. Tested by REMOVING the key rather than writing an empty
    // array, because a missing key is the shape a truncated package actually
    // takes and `gguf_set_arr_str` with n = 0 passes a null data pointer.
    {
        GgufContext c = voice_design_metadata();
        gguf_remove_key(c.get(), "synthesize.voice.profile_sources");
        HParams hparams;
        SYNTH_TEST_CHECK(read_hparams(c.get(), hparams) != SYNTH_OK);
    }
    return 0;
}
```

And the builder it uses, placed beside `valid_metadata()`. Start from the
existing Base synthetic package if the file has one; otherwise copy
`valid_metadata()` and change these fields:

```cpp
// A voice_design package: profile-sources mode, zero presets, doubled talker
// dimensions, no speaker encoder, and a design Profile schema.
GgufContext voice_design_metadata() {
    GgufContext    c = valid_metadata();
    gguf_context * g = c.get();
    gguf_set_val_str(g, "synthesize.model_variant", "qwen3-tts-12hz-1-7b-voicedesign");
    gguf_set_val_str(g, "synthesize.voice.mode", "profile-sources");
    gguf_set_val_bool(g, "synthesize.voice.has_package_default", false);
    gguf_set_val_u32(g, "synthesize.voice.preset_count", 0);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.talker.hidden_size", 2048);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.talker.intermediate_size", 6144);
    set_string_array(g, "synthesize.voice.profile_sources", { "description-text" });
    gguf_set_val_str(g, "synthesize.profile.schema", "qwen3-tts-voice-design");
    gguf_set_val_u32(g, "synthesize.profile.schema_version", 1);
    gguf_set_val_str(g, "synthesize.profile.compatibility_id",
                     "0000000000000000000000000000000000000000000000000000000000000001");
    return c;
}
```

**If `valid_metadata()` writes the speaker/preset arrays**, remove them here
with `gguf_remove_key`; check the exact key names by reading the function.
**If the talker metadata key names differ from the guesses above**, take them
from `read_talker` in `src/arch/qwen3-tts/weights.cpp` — do not invent them.

Register the new function in the file's `main()` alongside the existing tests.

- [ ] **Step 2: Run and watch it fail**

```bash
cmake --build build --target synthesize-qwen3-tts-metadata-test -j 16
ctest --test-dir build --output-on-failure -R '^synthesize-qwen3-tts-metadata-test$'
```

Expected: FAIL — `profile_sources` is not a member of `HParams`.

- [ ] **Step 3a: Give `GgufMetadata` a presence check**

It has none today. In `src/gguf-metadata.h`, beside the other accessors:

```cpp
    // Whether a key is present at all, without reading it and without the
    // error report the typed accessors emit on a miss. Used where a key's
    // ABSENCE is the meaningful answer -- a package that declares it cannot
    // clone must not carry a speaker encoder, and asking has to be quiet.
    bool has(const std::string & key) const;
```

and in `src/gguf-metadata.cpp`:

```cpp
bool GgufMetadata::has(const std::string & key) const {
    return gguf_find_key(gguf_, key.c_str()) >= 0;
}
```

- [ ] **Step 3b: Add the field and read it**

In `src/arch/qwen3-tts/weights.h`, inside `HParams`, replacing the comment on
`has_speaker_encoder`:

```cpp
    // Which Voice Profile sources this package DECLARES, as
    // SYNTH_PROFILE_SOURCE_* bits. Declared rather than inferred: before Stage
    // 3, `profile-sources` mode meant Base and therefore meant reference
    // audio, and that stopped being true when VoiceDesign arrived with a
    // disjoint source set and the same mode.
    uint32_t             profile_sources     = 0;
    // Only present for a variant with a speaker encoder (Base).
    bool                 has_speaker_encoder = false;
```

In `src/arch/qwen3-tts/weights.cpp`, replace
`read_profile_and_speaker_encoder` with:

```cpp
// The declared source names, mapped to the public bits. An unknown name is a
// refusal rather than a skip: a package written by a newer converter must not
// load with a capability quietly narrower than it claims.
bool read_profile_sources(const GgufMetadata & meta, HParams & hparams) {
    std::vector<std::string> names;
    if (!meta.string_array("synthesize.voice.profile_sources", names) || names.empty()) {
        std::fprintf(stderr, "qwen3-tts: profile-sources mode declares no profile sources\n");
        return false;
    }
    hparams.profile_sources = 0;
    for (const std::string & name : names) {
        if (name == "reference-audio") {
            hparams.profile_sources |= SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO;
        } else if (name == "description-text") {
            hparams.profile_sources |= SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT;
        } else {
            std::fprintf(stderr, "qwen3-tts: unknown profile source %s\n", name.c_str());
            return false;
        }
    }
    return true;
}

// The blocks a package must carry follow from what it declared, and the
// declaration must match what it actually has. Reference Audio needs the
// encoder and the reference limits; Description Text needs neither and must
// not carry them, because a package that ships an encoder while claiming it
// cannot clone disagrees with itself and one of the two statements is wrong.
bool read_profile_and_speaker_encoder(const GgufMetadata & meta, HParams & hparams) {
    if (!read_profile_sources(meta, hparams)) {
        return false;
    }
    const bool wants_reference = (hparams.profile_sources & SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO) != 0;
    const bool carries_encoder = meta.has("synthesize.qwen3-tts.speaker_encoder.enc_dim");
    if (wants_reference != carries_encoder) {
        std::fprintf(stderr,
                     "qwen3-tts: package declares reference-audio=%d but carries a speaker encoder=%d\n",
                     int(wants_reference), int(carries_encoder));
        return false;
    }
    if (wants_reference) {
        hparams.has_speaker_encoder = true;
        return read_speaker_encoder(meta, hparams) && read_profile_contract(meta, hparams);
    }
    return read_profile_contract(meta, hparams);
}
```

`read_profile_contract` currently hardcodes the schema string at line 493. Make
it accept either, keyed on the declared source:

```cpp
    const char * expected_schema = (hparams.profile_sources & SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO) != 0 ?
                                       "qwen3-tts-voice-clone" :
                                       "qwen3-tts-voice-design";
    if (profile.schema != expected_schema || profile.schema_version != kProfileSchemaVersion) {
        std::fprintf(stderr, "qwen3-tts: unsupported profile schema %s version %u\n", profile.schema.c_str(),
                     profile.schema_version);
        return false;
    }
```

and skip its reference-rate cross-check (line 526, which compares
`profile.reference_sample_rate` against the encoder's) when there is no
encoder — guard it with `if (hparams.has_speaker_encoder)`.

`weights.cpp` must include `synthesize.h` for the `SYNTH_PROFILE_SOURCE_*`
macros if it does not already — check its include block before adding one.

- [ ] **Step 4: Run and watch it pass**

`src/gguf-metadata.h` is shared with the other three families, so build
everything, not just this test:

```bash
cmake --build build --target synthesize-qwen3-tts-metadata-test -j 16
ctest --test-dir build --output-on-failure -R '^synthesize-qwen3-tts-metadata-test$'
cmake --build build --target synthesize-check-unit -j 16
```

Expected: PASS, and every pre-existing qwen3-tts test still passes — the Base
and CustomVoice paths must be untouched.

- [ ] **Step 5: Commit**

```bash
git add src/gguf-metadata.h src/gguf-metadata.cpp \
        src/arch/qwen3-tts/weights.h src/arch/qwen3-tts/weights.cpp \
        tests/qwen3_tts_metadata_test.cpp
scripts/ci/clang-format.sh --fix
git add -A
git commit -m "qwen3-tts: read a package's declared Profile sources"
```

---

## Task 4: Capability bits follow the declaration, not the Voice Mode

**Files:**
- Modify: `src/arch/qwen3-tts/weights.cpp:766-816` (`fill_voice_profile_capability`)
- Test: `tests/qwen3_tts_voice_required_test.cpp`

**Interfaces:**
- Consumes: `HParams::profile_sources` from Task 3.
- Produces: `VoiceProfileInfo::source_flags` carrying
  `SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT | SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE`
  for a VoiceDesign package, and the existing
  `REFERENCE_AUDIO | SERIALIZED_PROFILE` for Base.

**Read first:** `src/arch/qwen3-tts/weights.cpp:766-783`, whose comment records
a trap already caught once — `has_preset_voice_catalog` being the inverse of a
condition it looked like it matched. This task is the same trap in a new place.

- [ ] **Step 1: Write the failing tests**

Add to `tests/qwen3_tts_voice_required_test.cpp`:

```cpp
// A VoiceDesign package advertises Description Text and NOT Reference Audio.
// The bits follow the tensors, not the variant's name: with no speaker encoder
// and no codec encoder there is nothing to clone with, and advertising a source
// with no implementation behind it invites a caller to pass a recording and
// receive an error they were told would not happen.
int test_capability_follows_the_declared_sources() {
    {
        HParams hparams;
        hparams.voice_mode      = VoiceMode::ProfileSources;
        hparams.profile_sources = SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT;
        VoiceProfileInfo info{};
        fill_voice_profile_capability(hparams, info);
        SYNTH_TEST_CHECK((info.source_flags & SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT) != 0);
        SYNTH_TEST_CHECK((info.source_flags & SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE) != 0);
        SYNTH_TEST_CHECK((info.source_flags & SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO) == 0);
        // The reference limits describe a capability this package does not
        // have, so they stay zero rather than carrying Base's numbers.
        SYNTH_TEST_CHECK(info.max_reference_count == 0);
        SYNTH_TEST_CHECK(info.reference_target_sample_rate == 0);
    }
    // Base is unchanged.
    {
        HParams hparams;
        hparams.voice_mode          = VoiceMode::ProfileSources;
        hparams.profile_sources     = SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO;
        hparams.has_speaker_encoder = true;
        hparams.profile.reference_sample_rate = 24000;
        hparams.profile.max_reference_count   = 1;
        VoiceProfileInfo info{};
        fill_voice_profile_capability(hparams, info);
        SYNTH_TEST_CHECK((info.source_flags & SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO) != 0);
        SYNTH_TEST_CHECK((info.source_flags & SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT) == 0);
        SYNTH_TEST_CHECK(info.reference_target_sample_rate == 24000);
    }
    // A preset-catalog package advertises nothing, whatever it declares --
    // the adversarial case this file already makes for the catalog half.
    {
        HParams hparams;
        hparams.voice_mode      = VoiceMode::PresetCatalog;
        hparams.profile_sources = SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT;
        VoiceProfileInfo info{};
        fill_voice_profile_capability(hparams, info);
        SYNTH_TEST_CHECK(info.source_flags == 0);
    }
    return 0;
}
```

Register it in the file's `main()`.

- [ ] **Step 2: Run and watch it fail**

```bash
cmake --build build --target synthesize-qwen3-tts-voice-required-test -j 16
ctest --test-dir build --output-on-failure -R '^synthesize-qwen3-tts-voice-required-test$'
```

Expected: FAIL — the VoiceDesign case reports `REFERENCE_AUDIO`.

- [ ] **Step 3: Publish from the declaration**

Replace the hardcoded pair at `weights.cpp:783` and guard the reference limits:

```cpp
    // The bits follow what the package DECLARED and read_profile_sources
    // already checked against what it carries. Serialized Profile accompanies
    // either source, because every v1 Profile this family can create can be
    // serialized (docs/c-interface.md).
    info.source_flags = hparams.profile_sources | SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE;

    if ((hparams.profile_sources & SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO) == 0) {
        // No recording is taken on this path, so the six reference limits and
        // the transcript requirements describe nothing. Left at their zeroed
        // defaults: a zero here means "not applicable", and copying Base's
        // numbers would state a contract this package cannot honour.
        info.schema         = hparams.profile.schema;
        info.schema_version = hparams.profile.schema_version;
        decode_profile_compatibility_id(hparams.profile.compatibility_id_hex, info.compatibility_id);
        return;
    }
```

placed immediately before the existing
`info.reference_transcript = SYNTH_REQUIREMENT_OPTIONAL;` line, leaving
everything below it untouched.

**Erratum, 2026-08-18 — this task's own "Produces" line above and the code
snippet just above this note were superseded before Plan 1 closed.** The
final whole-branch review found that this task's rule ("bits follow what the
package declared") published `SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT` for a
VoiceDesign package while `create_from_description` (`src/voice-profile.cpp`)
had no Qwen3-TTS arm — the seam advertised a capability it then refused, per
jiangzhuo's ruling recorded in full at Task 6's and the Completion Gate's own
2026-08-18 errata below. `fill_voice_profile_capability` now masks
`SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT` out of the published `source_flags`
regardless of the package's declaration, so a VoiceDesign package (whose only
declared source is description-text) publishes `source_flags == 0`, not
`DESCRIPTION_TEXT | SERIALIZED_PROFILE`. The `Base` row of this task's rule is
unaffected. `tests/qwen3_tts_voice_required_test.cpp`'s
`test_capability_follows_the_declared_sources` (the test Step 1 adds above)
carries the corresponding update; its VoiceDesign case now asserts the same
all-zero shape `check_reports_no_voice_profile_support` already pins for
CustomVoice, rather than the three assertions shown in Step 1's snippet.

- [ ] **Step 4: Run and watch it pass**

```bash
cmake --build build --target synthesize-qwen3-tts-voice-required-test -j 16
ctest --test-dir build --output-on-failure -L qwen3-tts
```

Expected: PASS, all qwen3-tts tests.

- [ ] **Step 5: Commit**

```bash
git add src/arch/qwen3-tts/weights.cpp tests/qwen3_tts_voice_required_test.cpp
scripts/ci/clang-format.sh --fix
git add -A
git commit -m "qwen3-tts: publish Profile sources from the package's declaration"
```

---

## Task 5: The prompt with no speaker slot

**Files:**
- Modify: `src/arch/qwen3-tts/talker-host.cpp:150-175`
- Test: `tests/qwen3_tts_prompt_slot_test.cpp`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `build_talker_prompt(hparams, request, out)` honouring
  `request.has_speaker == false` by emitting a codec run one position shorter,
  with `out.external_speaker_index == -1`.

**Read first:** `src/arch/qwen3-tts/talker-host.cpp:159` — the branch exists.
`src/arch/qwen3-tts/model.cpp:979` sets `has_speaker = true` unconditionally, so
**nothing on the model path has ever reached the `else`.** Written is not run,
and run is not correct. Upstream's shape is at
`modeling_qwen3_tts.py:2166-2172`: with `speaker_embed` null the codec prefill
is `cat(prefill_0, prefill_1)` — no speaker element at all.

Do **not** change `model.cpp` in this task. Plan 2 wires it when a
`DesignInstruct` Profile exists to select it; this task makes the layout correct
and proves it.

- [ ] **Step 1: Write the failing test**

Add to `tests/qwen3_tts_prompt_slot_test.cpp`:

```cpp
// The third speaker case. CustomVoice fills the slot from the codec vocabulary
// and Base substitutes an embedding into it, but both KEEP the position;
// VoiceDesign has no slot at all, so its codec run is one shorter and the text
// stream shifts with it. modeling_qwen3_tts.py:2166-2172 -- with speaker_embed
// null the prefill is cat(prefill_0, prefill_1) and nothing else.
int test_prompt_without_a_speaker_slot() {
    HParams hparams = minimal_hparams();

    TalkerPromptRequest with_speaker;
    with_speaker.role_tokens   = { 1, 2, 3 };
    with_speaker.text_tokens   = { 10, 11, 12 };
    with_speaker.has_language  = true;
    with_speaker.language_token = 7;
    with_speaker.has_speaker   = true;
    with_speaker.speaker_token = 42;

    TalkerPromptRequest without_speaker = with_speaker;
    without_speaker.has_speaker         = false;
    without_speaker.speaker_token       = 0;

    TalkerPrompt kept;
    TalkerPrompt dropped;
    SYNTH_TEST_CHECK(build_talker_prompt(hparams, with_speaker, kept) == SYNTH_OK);
    SYNTH_TEST_CHECK(build_talker_prompt(hparams, without_speaker, dropped) == SYNTH_OK);

    // Exactly one position fewer, and it is a CODEC position that went.
    SYNTH_TEST_CHECK(dropped.positions.size() + 1 == kept.positions.size());
    size_t kept_codec = 0;
    size_t dropped_codec = 0;
    for (const TalkerInputPosition & p : kept.positions) {
        kept_codec += p.has_codec ? 1 : 0;
    }
    for (const TalkerInputPosition & p : dropped.positions) {
        dropped_codec += p.has_codec ? 1 : 0;
    }
    SYNTH_TEST_CHECK(dropped_codec + 1 == kept_codec);

    // The speaker token appears in one and not the other -- an implementation
    // that merely zeroed it would pass the count checks above.
    bool found = false;
    for (const TalkerInputPosition & p : dropped.positions) {
        found = found || (p.has_codec && p.codec_token == 42);
    }
    SYNTH_TEST_CHECK(!found);

    // No external substitution is pending: -1 means "the graph reads every
    // codec row", which is what talker.cpp's null-speaker branch expects.
    SYNTH_TEST_CHECK(dropped.external_speaker_index == -1);

    // The trailing schedule is unaffected: dropping the slot shortens the
    // prefill, not the text that follows it.
    SYNTH_TEST_CHECK(dropped.trailing.size() == kept.trailing.size());
    return 0;
}
```

**`minimal_hparams()`** — if the file has no such helper, build the smallest
`HParams` the existing tests in this file use; read the top of
`tests/qwen3_tts_prompt_slot_test.cpp` and reuse whatever it already does.

- [ ] **Step 2: Run and watch it fail**

```bash
cmake --build build --target synthesize-qwen3-tts-prompt-slot-test -j 16
ctest --test-dir build --output-on-failure -R '^synthesize-qwen3-tts-prompt-slot-test$'
```

Expected: FAIL. Record **how** it fails — if the counts already match, the
branch may be correct and the remaining work is only the assertions plus
Task 6's oracle comparison. Say so rather than editing until something changes.

- [ ] **Step 3: Make the layout right**

In `src/arch/qwen3-tts/talker-host.cpp`, the `if (request.has_speaker)` branch
at line 159 appends the speaker's codec position. Ensure the `else` path
appends nothing there and that every offset computed afterwards is derived from
the run's actual length rather than from a constant that assumed the slot.
Leave `external_speaker_index` at its `-1` default when `has_speaker` is false.

- [ ] **Step 4: Run and watch it pass, under the sanitizer too**

```bash
cmake --build build --target synthesize-qwen3-tts-prompt-slot-test -j 16
ctest --test-dir build --output-on-failure -L qwen3-tts
cmake --build build-sanitize --target synthesize-qwen3-tts-prompt-slot-test -j 16
ctest --test-dir build-sanitize --output-on-failure -R '^synthesize-qwen3-tts-prompt-slot-test$'
```

Expected: PASS in both trees. An off-by-one in prefill layout is exactly the
class ASan catches.

- [ ] **Step 5: Commit**

```bash
git add src/arch/qwen3-tts/talker-host.cpp tests/qwen3_tts_prompt_slot_test.cpp
scripts/ci/clang-format.sh --fix
git add -A
git commit -m "qwen3-tts: lay out the prompt that has no speaker slot"
```

---

## Task 6: Intake, conversion, and a package that loads

**Files:**
- Create: `tests/golden/qwen3-tts/qwen3-tts-12hz-1-7b-voicedesign.manifest.json`
- Modify: `docs/porting/families/qwen3-tts.md`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: `models/qwen3-tts-12hz-1-7b-voicedesign/qwen3-tts-12hz-1-7b-voicedesign-BF16.gguf`
  (gitignored) and its recorded digest in the manifest.

This task needs the ~4.5 GB download. Everything before it ran without one.

- [ ] **Step 1: Verify the licence from the upstream model card**

```bash
curl -sL "https://huggingface.co/api/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['id'], (d.get('cardData') or {}).get('license'), d['sha'])"
```

Expected: `apache-2.0` and sha `5ecdb67327fd37bb2e042aab12ff7391903235d3`.
Read the card's own text too — a code licence is not a weight licence, and a
port's README is never a licence source. If it does not say Apache-2.0, **stop
and report**; do not convert.

- [ ] **Step 2: Fetch the checkpoint at the pinned revision**

```bash
uv run --project scripts/envs/qwen3-tts python -c "
from huggingface_hub import snapshot_download
p = snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign',
                      revision='5ecdb67327fd37bb2e042aab12ff7391903235d3',
                      local_dir='models/qwen3-tts-12hz-1-7b-voicedesign-src')
print(p)
"
```

- [ ] **Step 3: Write the Golden Manifest**

Copy `tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-base.manifest.json` and change:
`variant` to `qwen3-tts-12hz-1-7b-voicedesign`; the `source` block's
`checkpoint_name`, `checkpoint_url` and revision to the pinned ones; and
**remove every case** — this plan adds one case in Task 7 and Plan 2 adds the
rest. Remove the `reference-audio` artifact entries: this variant takes no
recording. Fill `sha256` for each artifact from:

**Erratum, 2026-08-18 — a final whole-branch review found that "this plan adds
one case in Task 7" did not happen.** Task 7 gates the empty-instruct rung
with an integration driver and a `replay`-stage tolerance cell instead — a
comparison outside the manifest, not a Golden Manifest case — and the
manifest committed by Task 6 still carries `cases: []` after Task 7 lands.
This sentence is left as written above, as this file's own convention for a
plan text that later stopped matching what happened; the correction is: the
manifest's first case, and the rest, both arrive in Plan 2.

```bash
sha256sum models/qwen3-tts-12hz-1-7b-voicedesign-src/config.json \
          models/qwen3-tts-12hz-1-7b-voicedesign-src/model.safetensors
```

- [ ] **Step 4: Convert**

```bash
uv run --project scripts/envs/qwen3-tts python scripts/convert-qwen3-tts.py \
  --manifest tests/golden/qwen3-tts/qwen3-tts-12hz-1-7b-voicedesign.manifest.json \
  --weights-dir models/qwen3-tts-12hz-1-7b-voicedesign-src \
  --output models/qwen3-tts-12hz-1-7b-voicedesign/qwen3-tts-12hz-1-7b-voicedesign-BF16.gguf \
  --report reports/porting/qwen3-tts/qwen3-tts-12hz-1-7b-voicedesign/convert.json
```

Expected: it writes the file. If it refuses on a tensor name, the catalog needs
a new entry — read the error, add the name to `src/arch/qwen3-tts/catalog.cpp`,
and extend `tests/qwen3_tts_catalog_test.cpp` in the same commit.

- [ ] **Step 5: Load it**

```bash
./build/bin/synthesize-cli --model models/qwen3-tts-12hz-1-7b-voicedesign/qwen3-tts-12hz-1-7b-voicedesign-BF16.gguf --info
```

Expected: it loads; the capability snapshot reports **zero Preset Voices** and
Profile sources **Description Text + Serialized Profile**, with Reference Audio
**absent**. If `--info` is not the flag, find it with `--help`.

**Erratum, 2026-08-18 — jiangzhuo's ruling after the final whole-branch review
contradicts the capability snapshot stated above.** `fill_voice_profile_capability`
(`src/arch/qwen3-tts/weights.cpp`) advertised `SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT`
through the runtime's public capability query while
`synth_voice_profile_create_from_description` (`src/voice-profile.cpp`) still
routed every family but OmniVoice — Qwen3-TTS, VoiceDesign included — to the
generic unsupported fallback: the seam promised a capability it then refused,
exactly the mistake this family's own transcript-assisted (ICL) mode was
withheld from advertisement for the whole of Plan 2 to avoid. Ruling: withhold
`SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT` from the RUNTIME's published capability
until a later plan wires the handler. `SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE`
does not survive alone either, because this package has no speaker encoder and
`load_qwen3_tts_profile_from_memory` can therefore never accept a Profile
against it. The corrected expectation: `--info` reports **zero Preset Voices**
and **zero Profile source flags** — the same all-zero "no runtime Voice Profile
support" shape a CustomVoice package reports, for a different reason. This is a
statement about the RUNTIME only: the package's own declared
`synthesize.voice.profile_sources` still names `description-text`, and Task 3's
loader and its cross-checks are untouched.

- [ ] **Step 6: Record the intake in the family record**

Add to `docs/porting/families/qwen3-tts.md`: the Stage 3 opening, the pinned
revision and digests, the measured BF16 package size against the design's ≈4.3 GB
estimate, and — as its own paragraph, because it will mislead someone —
that `generate_custom_voice` silently discards `instruct` for any 0.6B model
(`qwen_tts/inference/qwen3_tts_model.py:799-800`), so Description Text appears
to work on the published CustomVoice package and does not.

- [ ] **Step 7: Commit**

```bash
git add tests/golden/qwen3-tts/qwen3-tts-12hz-1-7b-voicedesign.manifest.json \
        docs/porting/families/qwen3-tts.md
git commit -m "qwen3-tts: take in the VoiceDesign checkpoint and convert it"
```

---

## Task 7: The oracle at an empty instruct — the completion gate

**Files:**
- Create: `scripts/dump_reference_qwen3_tts_voicedesign.py`
- Create: `tests/qwen3_tts_voicedesign_prefill_real.cpp`
- Modify: `tests/CMakeLists.txt`, `tests/tolerances/qwen3-tts.json`

**Interfaces:**
- Consumes: the package from Task 6, the prompt layout from Task 5.
- Produces: a `replay`-stage tolerance cell for the new variant, and an
  integration driver registered with `synth_register_integration_target`.

Plan 1 has no `DesignInstruct` Profile — that is Plan 2 — and the public seam
refuses a synthesis with no Voice Profile on a Catalog-less package. So the gate
runs through an **integration driver** that builds the prompt directly, the same
adapter shape `synthesize-qwen3-tts-replay-real` already uses.

- [ ] **Step 1: Write the oracle dumper**

Model it on `scripts/dump_reference_qwen3_tts_base.py`. It must call
`generate_voice_design(text=..., instruct="", language=...)` and dump, as raw
float32:

- `prefill.f32` — the assembled talker input embeddings, shape `[T, hidden]`
- `codes.i32` — the talker's emitted codes
- `waveform.f32` — the decoded audio

Pin `torch.manual_seed` and use greedy decoding, matching the base dumper's own
determinism settings; read them from that file rather than choosing new ones.

- [ ] **Step 2: Run it**

```bash
uv run --project scripts/envs/qwen3-tts python scripts/dump_reference_qwen3_tts_voicedesign.py \
  --checkpoint models/qwen3-tts-12hz-1-7b-voicedesign-src \
  --text "Qwen3-TTS is awesome!" --instruct "" --language English \
  --out reports/porting/qwen3-tts/qwen3-tts-12hz-1-7b-voicedesign/oracle/
```

Expected: three files. Record their shapes — the prefill's `T` is what Task 5's
layout must reproduce, and a mismatch here is the whole point of the gate.

- [ ] **Step 3: Write the driver**

`tests/qwen3_tts_voicedesign_prefill_real.cpp` loads the package, builds a
prompt with `has_speaker = false` and no instruct tokens, flattens it, and
writes the prefill embeddings to a file for comparison. It asserts nothing — it
is an adapter, like the family's other `*-real` drivers. Register it in
`tests/CMakeLists.txt` with `synth_register_integration_target`.

**Erratum, 2026-08-18 — Task 7 Fix Round 2 — "it asserts nothing" is no
longer true.** A review of the completed task found that this sentence, taken
literally, left the completion gate enforcing nothing: the driver printed its
observations and exited 0 regardless of what they were, so a fault-injected
build reporting a 365× tolerance breach and a shape mismatch still passed
whatever ran it. Put to jiangzhuo, because a finding that contradicts approved
plan text is not a reviewer's or an implementer's to overrule alone. Ruling:
the finding governs, this sentence does not. The driver now takes the
committed `max_relative` bound from `tests/tolerances/qwen3-tts.json` as an
argument — read at CMake configure time and passed to `add_test`, the same
shape `prompt.icl_embed`'s own gate already has
(`tests/CMakeLists.txt`'s `synthesize-qwen3-tts-icl-prompt-real` block) — and
exits non-zero when the measured p95 exceeds it or when the shapes disagree.
Without a bound argument it still behaves exactly as this section describes,
which is what a manual, unregistered run still gets. **The tier did not
change**: it remains an integration target behind
`-DSYNTH_BUILD_INTEGRATION_TESTS=ON`, registered only with
`synth_register_integration_target` plus (now) one `add_test`, and it still
never enters `synthesize-check-unit`. Only the exit code changed.

- [ ] **Step 4: Compare, and record the tolerance**

Compare the driver's prefill against the oracle's with the p95 relative
statistic this family already uses — `reconstruction_p95_relative` from
`tests/qwen3_tts_percentile.h`. Add a `replay` cell for the new variant to
`tests/tolerances/qwen3-tts.json`, stating beside each figure the measurement
that set it.

Then **prove the gate can fail**: rebuild with the speaker slot retained (Task
5's branch inverted), re-run, and record the faulted figure beside the clean one.
A gate whose fault-injection figure is not recorded is not known to be a gate.

- [ ] **Step 5: Run the full gates and commit**

```bash
cmake --build build --target synthesize-check-unit -j 16
cmake --build build-sanitize --target synthesize-check-unit -j 16
git add -A
scripts/ci/clang-format.sh --fix
git add -A
scripts/ci/clang-format.sh --check-diff
git commit -m "qwen3-tts: match the oracle at an empty instruct"
```

Expected: only the two baseline VITS failures. Any third is yours.

---

## Completion Gate

Plan 1 is done when all three hold:

1. `models/.../qwen3-tts-12hz-1-7b-voicedesign-BF16.gguf` loads.
2. A prefill built at empty instruct matches the oracle within the recorded
   `replay` tolerance, and the fault-injection figure proving that comparison
   can fail is recorded beside it.
3. The capability snapshot reports zero Preset Voices and
   `DESCRIPTION_TEXT | SERIALIZED_PROFILE` with `REFERENCE_AUDIO` absent.

**Erratum, 2026-08-18 — jiangzhuo's ruling after the final whole-branch review
contradicts item 3 above.** `fill_voice_profile_capability` published
`DESCRIPTION_TEXT | SERIALIZED_PROFILE` while `create_from_description`
(`src/voice-profile.cpp`) still routed every family but OmniVoice to the
generic unsupported fallback — the seam advertised a capability it then
refused, which is the fault this family's own ICL precedent (Plan 2 →
Plan 3) was already built to avoid repeating. Ruling: the RUNTIME withholds
`SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT` until a later plan wires the handler,
and `SERIALIZED_PROFILE` does not survive alone either, since this package has
no speaker encoder for `load_qwen3_tts_profile_from_memory` to ever accept a
Profile against. Corrected item 3: **the capability snapshot reports zero
Preset Voices and zero Profile source flags** — the same shape a CustomVoice
package reports. The PACKAGE's own declared `profile_sources` is unaffected
and still names `description-text`; only the RUNTIME's published capability
changed, and Task 3's loader is untouched.

Explicitly **not** delivered here, and not a gap: `create_from_description`, the
`DesignInstruct` payload, the instruct prompt block, the public-seam validator,
quantization, backends, and the listening audit. Those are Plans 2 and 3.
