"""Every claimed profile and backend must have a measurement for every stage.

A validator added after a sweep leaves a hole that no failing test points at:
the runs that did happen all succeeded, and nothing compares what was run
against what is claimed. That is how the Kokoro decoder stage ended up
unmeasured on CPU for two Quantization Profiles while being measured on CUDA
for all three.

This reads the committed tolerance files and asserts the grid is full, so a new
stage or a new profile fails here until it has been measured everywhere the
package page claims support.

It covers only families that record measurements per stage. The VITS file
predates that format: it says `measurements-recorded-thresholds-deferred` while
its `stages` object is empty, so there is nothing to cross-check and this test
does not protect it. That is a real gap in the VITS port, recorded here rather
than papered over by loosening the check.
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
            profiles = document.get("profiles")
            if not profiles:
                # Legacy shape: measurements are claimed in prose but not
                # recorded per stage, so there is no grid to check. See the
                # module docstring.
                self.assertEqual(document.get("stages"), {}, "a family with profiles must use them")
                continue

            with self.subTest(family=family):
                # The stage set is whatever the reference profile measured, and
                # every other profile and backend has to match it exactly.
                reference = reference_profile(document)
                self.assertIn(
                    reference, set(profiles), f"{family}: no measurements for reference profile {reference}"
                )
                expected_stages = set(profiles[reference]["stages"])
                self.assertTrue(expected_stages, "the reference profile measured no stages")

                for profile, entry in profiles.items():
                    self.assertEqual(
                        set(entry["stages"]),
                        expected_stages,
                        f"{family}/{profile} on CPU is missing "
                        f"{sorted(expected_stages - set(entry['stages']))}",
                    )
                    for backend, backend_entry in entry.get("backends", {}).items():
                        self.assertEqual(
                            set(backend_entry["stages"]),
                            expected_stages,
                            f"{family}/{profile} on {backend} is missing "
                            f"{sorted(expected_stages - set(backend_entry['stages']))}",
                        )

    def test_every_registered_validator_has_a_measured_stage(self) -> None:
        """A stage with a validator but no measurement has never been run."""
        for family, document in load_tolerances().items():
            profiles = document.get("profiles")
            if not profiles:
                continue
            validators = {
                path.stem[len(f"validate-{family}-") :]
                for path in (REPO_ROOT / "scripts").glob(f"validate-{family}-*.py")
            }
            if not validators:
                continue
            with self.subTest(family=family):
                measured = set(profiles[reference_profile(document)]["stages"])
                self.assertEqual(
                    validators,
                    measured,
                    f"{family}: validators {sorted(validators - measured)} have no recorded "
                    f"measurement; stages {sorted(measured - validators)} have no validator",
                )


if __name__ == "__main__":
    unittest.main()
