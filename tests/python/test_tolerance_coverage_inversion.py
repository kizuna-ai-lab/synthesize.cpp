"""The validator/stage coverage rule must still reject what it exists to reject.

test_tolerance_coverage.py's rule was weakened on 2026-08-14 -- from per-variant
exact set equality to "every validator measured in at least one variant, every
measured stage has a validator" -- because exact equality became unsatisfiable
once a capability existed in one Model Variant and not another. A weakened rule
is exactly the kind that quietly stops biting, and this repository has already
shipped fifteen checks that could not fail. So the weakening is pinned here by
construction rather than by inspection.

This drives the SHIPPED function, `validator_coverage_failures`, with
constructed inputs. It does not restate the rule: an inversion test that
reimplements what it is checking proves only that the reimplementation fails.

Each test names the dimension it inverts.
"""

from __future__ import annotations

import unittest

from tests.python.test_tolerance_coverage import validator_coverage_failures


class ValidatorCoverageInversionTests(unittest.TestCase):
    def test_the_real_shape_passes(self) -> None:
        """Control. The qwen3-tts shape as committed must be accepted.

        Without this, every assertion below could pass on a rule that rejects
        everything, which would be just as broken and much harder to see.
        """
        self.assertEqual(
            validator_coverage_failures(
                {"codec_encoder", "public", "replay"},
                {
                    "qwen3-tts-12hz-0-6b-base": {"codec_encoder", "public", "replay"},
                    "qwen3-tts-12hz-0-6b-customvoice": {"public", "replay"},
                },
            ),
            [],
        )

    def test_a_validator_measured_nowhere_is_rejected(self) -> None:
        """Dimension: validator with no measurement in ANY variant.

        This is the regression that actually happened -- Task 6 committed
        scripts/validate-qwen3-tts-codec-encoder.py while no variant measured a
        `codec-encoder` stage -- and the case the relaxed rule must still catch,
        because it is the Kokoro-decoder hole the module exists for.
        """
        failures = validator_coverage_failures(
            {"codec-encoder", "public", "replay"},
            {
                "qwen3-tts-12hz-0-6b-base": {"public", "replay"},
                "qwen3-tts-12hz-0-6b-customvoice": {"public", "replay"},
            },
        )
        self.assertTrue(failures, "a validator measured nowhere was accepted")
        self.assertIn("codec-encoder", failures[0])

    def test_a_measured_stage_with_no_validator_is_rejected(self) -> None:
        """Dimension: measured stage with no validator, in one variant.

        The direction that stayed per-variant. A committed number no script can
        reproduce is unfalsifiable, which is what `vocoder_polish` demonstrated
        when a relaxed form of this check let it through.
        """
        failures = validator_coverage_failures(
            {"public", "replay"},
            {
                "qwen3-tts-12hz-0-6b-base": {"public", "replay"},
                "qwen3-tts-12hz-0-6b-customvoice": {"public", "replay", "vocoder_polish"},
            },
        )
        self.assertTrue(failures, "a measured stage with no validator was accepted")
        self.assertIn("vocoder_polish", failures[0])

    def test_a_stage_measured_in_only_one_variant_is_still_covered(self) -> None:
        """Dimension: the specific relaxation, asserted rather than assumed.

        A capability present in one variant and absent from another must be
        accepted -- this is the whole reason the rule changed. Stated as its own
        test so that tightening the rule back to exact equality fails HERE, with
        this explanation attached, instead of failing on the real grid where the
        reason would have to be rediscovered.
        """
        self.assertEqual(
            validator_coverage_failures(
                {"codec_encoder", "public"},
                {"has-it": {"codec_encoder", "public"}, "lacks-it": {"public"}},
            ),
            [],
        )

    def test_single_variant_family_is_unchanged_by_the_relaxation(self) -> None:
        """Dimension: equivalence claim for one-variant families.

        For a single variant, "measured somewhere" and "measured here" are the
        same set, so the relaxation must be a no-op. Both orphan directions still
        fail.
        """
        self.assertEqual(validator_coverage_failures({"replay"}, {"only": {"replay"}}), [])
        self.assertTrue(validator_coverage_failures({"replay", "extra"}, {"only": {"replay"}}))
        self.assertTrue(validator_coverage_failures({"replay"}, {"only": {"replay", "extra"}}))


if __name__ == "__main__":
    unittest.main()
