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
        """A stage with a validator but no measurement has never been run."""
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
            for variant, profiles in variants.items():
                with self.subTest(family=family, variant=variant):
                    measured = set(profiles[reference_profile(document)]["stages"])
                    self.assertEqual(
                        validators,
                        measured,
                        f"{family}/{variant}: validators {sorted(validators - measured)} have no "
                        f"recorded measurement; stages {sorted(measured - validators)} have no validator",
                    )


if __name__ == "__main__":
    unittest.main()
