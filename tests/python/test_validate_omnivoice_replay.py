"""Focused tests for the omnivoice replay validator's own comparison logic.

`compare_grid` runs over the raw i32 grids the runner and the oracle produce --
pure-function checks against synthetic arrays, so they need neither a runner
binary nor a model package. The validator script is hyphenated
(`validate-omnivoice-replay.py`), so it is loaded via `importlib`, the same
pattern `test_convert_omnivoice.py` uses for `convert-omnivoice.py`.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

_spec = importlib.util.spec_from_file_location(
    "validate_omnivoice_replay", SCRIPTS / "validate-omnivoice-replay.py"
)
validate = importlib.util.module_from_spec(_spec)
sys.modules["validate_omnivoice_replay"] = validate
_spec.loader.exec_module(validate)


class CompareGridTest(unittest.TestCase):
    def test_single_admissible_grid_reports_its_own_size(self):
        primary = np.array([1, 2, 3], dtype=np.int32)
        result = validate.compare_grid([("primary", primary)], primary.copy())
        self.assertTrue(result["exact"])
        self.assertEqual(result["matched"], "primary")
        self.assertEqual(result["elements"], 3)

    def test_elements_reports_the_matched_grids_size_not_always_primary(self):
        # A same-length alternate the port's output matches exactly, while the
        # primary is a different case's grid entirely.
        primary = np.array([1, 2, 3, 4], dtype=np.int32)
        alternate = np.array([9, 9, 9, 9], dtype=np.int32)
        actual = alternate.copy()
        result = validate.compare_grid([("primary", primary), ("alt", alternate)], actual)
        self.assertTrue(result["exact"])
        self.assertEqual(result["matched"], "alt")
        self.assertEqual(result["elements"], 4)

    def test_elements_reports_the_closest_grids_own_size_on_a_miss(self):
        # The primary has 6 elements, the alternate 4; the port's output is
        # closest to (one mismatch from) the 4-element alternate.
        primary = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
        alternate = np.array([9, 9, 9, 9], dtype=np.int32)
        actual = np.array([9, 9, 9, 0], dtype=np.int32)
        result = validate.compare_grid([("primary", primary), ("alt", alternate)], actual)
        self.assertFalse(result["exact"])
        self.assertEqual(result["closest"], "alt")
        # Before the fix this was the primary's size (6); the grid actually
        # compared against -- and the one `mismatches` is counted over -- is
        # the 4-element alternate.
        self.assertEqual(result["elements"], 4)
        self.assertEqual(result["mismatches"], 1)

    def test_elements_matches_primary_when_primary_is_closest(self):
        # The common case (no alternates, or the primary itself is closest)
        # must be unaffected: this is what every currently-committed case's
        # sweep already exercises.
        primary = np.array([1, 2, 3, 4], dtype=np.int32)
        actual = np.array([1, 2, 3, 0], dtype=np.int32)
        result = validate.compare_grid([("primary", primary)], actual)
        self.assertFalse(result["exact"])
        self.assertEqual(result["closest"], "primary")
        self.assertEqual(result["elements"], 4)
        self.assertEqual(result["mismatches"], 1)


if __name__ == "__main__":
    unittest.main()
