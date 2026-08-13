"""Every claimed profile and backend must have a measurement for every stage.

A validator added after a sweep leaves a hole that no failing test points at:
the runs that did happen all succeeded, and nothing compares what was run
against what is claimed. That is how the Kokoro decoder stage ended up
unmeasured on CPU for two Quantization Profiles while being measured on CUDA
for all three.

This reads the committed tolerance files and asserts the grid is full, so a new
stage or a new profile fails here until it has been measured everywhere the
package page claims support.

It covers only families that record measurements per stage. VITS did not, until
2026-07-27: its file carried an empty `stages` object with the measurements
described in prose, so there was nothing to cross-check. That gap had a
consequence rather than staying theoretical. Removing the CUDA strict-FP32 gate
changed `duration.w_ceil` on the 315-token case and moved the frame count by one
hop, and because this test skipped the family, nothing failed; it was found by
comparing CPU against CUDA by hand.

The grid is per Model Variant, not per family. Keying on family alone was the
second half of that same hole: vits-vctk carries its own twelve cases and its own
speaker-conditioned path, so a grid that is full for vits-ljspeech says nothing
about it. A family with one variant keeps `profiles` at the top level; a family
with several uses `variants`, and each is checked against its own reference
profile.

The legacy branch below is kept because a family may legitimately arrive before
its grid has been swept, but reaching it now means a family regressed to prose,
which is why it asserts the empty object rather than merely tolerating it.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
TOLERANCES = REPO_ROOT / "tests" / "tolerances"


def reference_profile(document: dict) -> str:
    """The profile a family measures everything else against.

    `reference_stage` reads like `source-f32-cpu`: what is being referenced,
    the dtype, and the backend. The dtype names the profile.
    """
    parts = document.get("reference_stage", "").split("-")
    return parts[1].upper() if len(parts) > 1 else "F32"


def load_tolerances() -> dict[str, dict]:
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(TOLERANCES.glob("*.json"))
    }


def measured_variants(document: dict) -> dict[str, dict]:
    """The profile grids this document records, keyed by Model Variant.

    A family with one Reference Model Variant puts `profiles` at the top level.
    A family with several uses `variants`, because keying the whole file on
    family is what left vits-vctk unchecked while vits-ljspeech was covered: a
    second variant has its own cases and its own conditioning path, and a grid
    that is full for one says nothing about the other.
    """
    if "variants" in document:
        return {name: entry["profiles"] for name, entry in document["variants"].items()}
    if "profiles" in document:
        return {document.get("variant", document["family"]): document["profiles"]}
    return {}


def validator_coverage_failures(
    validators: set[str], measured_by_variant: dict[str, set[str]]
) -> list[str]:
    """The validator/stage rule, as a pure function over already-loaded sets.

    Extracted from the test method so that
    tests/python/test_tolerance_coverage_inversion.py can drive the SHIPPED rule
    with constructed inputs rather than reimplementing it. An inversion test that
    restates the rule proves only that the restatement fails.

    Returns one message per violation, empty when the grid is sound.
    """
    failures: list[str] = []
    measured_anywhere: set[str] = set().union(*measured_by_variant.values()) if measured_by_variant else set()

    # A validator nobody measures has never been run -- the Kokoro-decoder hole.
    orphan_validators = sorted(validators - measured_anywhere)
    if orphan_validators:
        failures.append(
            f"validators {orphan_validators} have no recorded measurement in ANY variant"
        )

    # A measured stage with no validator is a number nothing can reproduce.
    for variant in sorted(measured_by_variant):
        orphan_stages = sorted(measured_by_variant[variant] - validators)
        if orphan_stages:
            failures.append(f"{variant}: stages {orphan_stages} have no validator")

    return failures


class ToleranceCoverageTests(unittest.TestCase):
    def test_every_tolerance_file_is_well_formed(self) -> None:
        files = load_tolerances()
        self.assertTrue(files, "no tolerance files are committed")
        for family, document in files.items():
            with self.subTest(family=family):
                self.assertEqual(document["schema"], "synthesize-tolerances-v1")
                self.assertEqual(document["family"], family)
                self.assertIn("status", document)
                self.assertIn("note", document)

    def test_measured_families_cover_every_profile_backend_and_stage(self) -> None:
        for family, document in load_tolerances().items():
            variants = measured_variants(document)
            if not variants:
                # Legacy shape: measurements are claimed in prose but not
                # recorded per stage, so there is no grid to check. See the
                # module docstring.
                self.assertEqual(document.get("stages"), {}, "a family with profiles must use them")
                continue

            for variant, profiles in variants.items():
                with self.subTest(family=family, variant=variant):
                    # The stage set is whatever the reference profile measured,
                    # and every other profile and backend has to match it
                    # exactly. Each variant is judged against its own reference
                    # rather than the first one seen, so a variant with a stage
                    # the others lack still has to fill its own grid.
                    reference = reference_profile(document)
                    self.assertIn(
                        reference,
                        set(profiles),
                        f"{family}/{variant}: no measurements for reference profile {reference}",
                    )
                    expected_stages = set(profiles[reference]["stages"])
                    self.assertTrue(expected_stages, "the reference profile measured no stages")

                    for profile, entry in profiles.items():
                        self.assertEqual(
                            set(entry["stages"]),
                            expected_stages,
                            f"{family}/{variant}/{profile} on CPU is missing "
                            f"{sorted(expected_stages - set(entry['stages']))}",
                        )
                        for backend, backend_entry in entry.get("backends", {}).items():
                            self.assertEqual(
                                set(backend_entry["stages"]),
                                expected_stages,
                                f"{family}/{variant}/{profile} on {backend} is missing "
                                f"{sorted(expected_stages - set(backend_entry['stages']))}",
                            )

    def test_every_registered_validator_has_a_measured_stage(self) -> None:
        """A validator with no measurement has never been run, and vice versa.

        Two directions, deliberately asymmetric in scope, because a capability
        can be specific to one Model Variant:

          - every validator is exercised by at least ONE variant (family-wide);
          - every measured stage has a validator (per variant).

        This was one assertion of exact per-variant set equality until
        2026-08-14, which is a stronger rule than the intent and became
        unsatisfiable the moment a variant-specific capability arrived.
        qwen3-tts-12hz-0-6b-base carries a codec encoder and an ECAPA speaker
        encoder; qwen3-tts-12hz-0-6b-customvoice carries neither -- 894 emitted
        tensors against 657, and the difference is exactly 76 speaker_encoder +
        161 codec encoder (docs/porting/families/qwen3-tts.md's tensor census
        and its "The codec encoder is not carried" section). Under exact
        equality, measuring the codec encoder on the variant that HAS one forced
        a codec_encoder cell into CustomVoice's BF16/F16/Q8_MIXED grids and
        their CUDA sub-grids too -- five cells describing a subsystem that
        package does not contain, which is precisely the fabricated-placeholder
        disease tests/tolerances/qwen3-tts.json spent two plans removing.

        For a single-variant family the two formulations are identical, so
        nothing is relaxed for the families this already covered. What is given
        up is only the claim that every variant measures every stage, which was
        never true of a family whose variants differ in capability.

        Both directions still bite; see
        tests/python/test_tolerance_coverage_inversion.py, which constructs a
        validator with no measurement anywhere and a stage with no validator and
        asserts this test rejects each.
        """
        for family, document in load_tolerances().items():
            variants = measured_variants(document)
            if not variants:
                continue
            validators = {
                path.stem[len(f"validate-{family}-") :]
                for path in (REPO_ROOT / "scripts").glob(f"validate-{family}-*.py")
            }
            if not validators:
                continue
            reference = reference_profile(document)
            measured_by_variant = {
                variant: set(profiles[reference]["stages"])
                for variant, profiles in variants.items()
            }
            with self.subTest(family=family):
                self.assertEqual(
                    validator_coverage_failures(validators, measured_by_variant),
                    [],
                    f"{family}: validator/stage coverage is incomplete",
                )


if __name__ == "__main__":
    unittest.main()
