"""Focused tests for the qwen3-tts replay validator's tolerance lookup.

`resolve_tolerance_stage` is a pure function of an already-loaded tolerance
document: no model, no runner, no oracle payload. It exists because it broke
once. On 2026-08-11 `tests/tolerances/qwen3-tts.json` moved from one flat
`profiles` grid to a per-variant `variants.<name>.profiles` grid so the Base
manifest could share the file, and the lookup -- which had no variant
coordinate -- stopped resolving any cell at all, including for the PUBLISHED
CustomVoice variant whose gate then had no threshold behind it. The fix was
not covered by anything, which is what these tests are.

The last case runs against the real committed file rather than a fixture, so
a future restructuring of that file fails here rather than silently at the one
place a Golden gate reads it.

The validator script is hyphenated (`validate-qwen3-tts-replay.py`), so it is
loaded via `importlib`, the same pattern `test_validate_omnivoice_replay.py`
uses.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

_spec = importlib.util.spec_from_file_location(
    "validate_qwen3_tts_replay", SCRIPTS / "validate-qwen3-tts-replay.py"
)
validate = importlib.util.module_from_spec(_spec)
sys.modules["validate_qwen3_tts_replay"] = validate
_spec.loader.exec_module(validate)


def per_variant_document() -> dict:
    """The shape tests/tolerances/qwen3-tts.json uses: two variants, one file.

    Each variant's cell carries a distinguishable marker so a lookup that
    reaches the wrong variant is a wrong answer rather than a missing one.
    """
    return {
        "variants": {
            "alpha": {
                "case_count": 2,
                "profiles": {
                    "BF16": {
                        "stages": {"replay": {"marker": "alpha/BF16/CPU/replay"}},
                        "backends": {
                            "CUDA": {"stages": {"replay": {"marker": "alpha/BF16/CUDA/replay"}}},
                        },
                    },
                },
            },
            "beta": {
                "case_count": 3,
                "profiles": {
                    "BF16": {"stages": {"replay": {"marker": "beta/BF16/CPU/replay"}}},
                },
            },
        },
        # The sibling key that holds stages with no validator behind them.
        "provisional_variants": {
            "alpha": {"profiles": {"BF16": {"stages": {"speaker_encoder": {"marker": "provisional"}}}}},
        },
    }


class ResolveToleranceStageTests(unittest.TestCase):
    def test_per_variant_grid_resolves_the_named_variants_cell(self) -> None:
        document = per_variant_document()
        stage = validate.resolve_tolerance_stage(document, "alpha", "BF16", "CPU", "replay")
        self.assertEqual(stage["marker"], "alpha/BF16/CPU/replay")

    def test_a_second_variant_in_the_same_file_gets_its_own_cell(self) -> None:
        # The regression itself: one file, two variants. A lookup with no
        # variant coordinate cannot tell these apart -- it resolves the first,
        # or nothing at all.
        document = per_variant_document()
        stage = validate.resolve_tolerance_stage(document, "beta", "BF16", "CPU", "replay")
        self.assertEqual(stage["marker"], "beta/BF16/CPU/replay")

    def test_a_non_cpu_backend_reads_the_backends_subtree(self) -> None:
        document = per_variant_document()
        stage = validate.resolve_tolerance_stage(document, "alpha", "BF16", "cuda", "replay")
        self.assertEqual(stage["marker"], "alpha/BF16/CUDA/replay")

    def test_an_unknown_variant_resolves_nothing(self) -> None:
        # Not "falls back to some other variant's numbers": a gate with no
        # committed threshold must refuse to run, which is what None makes
        # the caller do.
        document = per_variant_document()
        self.assertIsNone(validate.resolve_tolerance_stage(document, "gamma", "BF16", "CPU", "replay"))

    def test_an_unknown_profile_backend_or_stage_resolves_nothing(self) -> None:
        document = per_variant_document()
        self.assertIsNone(validate.resolve_tolerance_stage(document, "alpha", "Q8_MIXED", "CPU", "replay"))
        self.assertIsNone(validate.resolve_tolerance_stage(document, "alpha", "BF16", "METAL", "replay"))
        self.assertIsNone(validate.resolve_tolerance_stage(document, "alpha", "BF16", "CPU", "public_request"))

    def test_provisional_variants_are_not_reachable(self) -> None:
        # `provisional_variants` holds stages with no validator and no
        # measurement behind them; resolving one would let a gate pass against
        # a number nobody measured.
        document = per_variant_document()
        self.assertIsNone(validate.resolve_tolerance_stage(document, "alpha", "BF16", "CPU", "speaker_encoder"))

    def test_a_flat_single_variant_file_still_resolves(self) -> None:
        # kokoro.json and omnivoice.json still use this shape, and a future
        # --tolerances could point at one.
        document = {"profiles": {"F32": {"stages": {"replay": {"marker": "flat"}}}}}
        stage = validate.resolve_tolerance_stage(document, "whatever", "F32", "CPU", "replay")
        self.assertEqual(stage["marker"], "flat")

    def test_the_committed_file_resolves_both_qwen3_tts_variants(self) -> None:
        path = REPO_ROOT / "tests" / "tolerances" / "qwen3-tts.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        for variant in ("qwen3-tts-12hz-0-6b-customvoice", "qwen3-tts-12hz-0-6b-base"):
            with self.subTest(variant=variant):
                stage = validate.resolve_tolerance_stage(document, variant, "BF16", "CPU", "replay")
                self.assertIsNotNone(stage, f"{path.name}: no BF16/CPU/replay cell for {variant}")
                self.assertIn("probes", stage)


if __name__ == "__main__":
    unittest.main()
