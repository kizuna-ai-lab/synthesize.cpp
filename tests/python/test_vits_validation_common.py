from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import vits_validation_common as validation  # noqa: E402


class VitsValidationCommonTest(unittest.TestCase):
    def test_accepts_supported_reference_variants(self) -> None:
        self.assertEqual(
            validation.manifest_variant({"family": "vits", "variant": "vits-ljspeech"}),
            "vits-ljspeech",
        )
        self.assertEqual(
            validation.manifest_variant({"family": "vits", "variant": "vits-vctk"}),
            "vits-vctk",
        )

    def test_rejects_unknown_family_or_variant(self) -> None:
        for manifest in (
            {"family": "other", "variant": "vits-vctk"},
            {"family": "vits", "variant": "future"},
        ):
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                validation.manifest_variant(manifest)

    def test_resolves_only_vctk_preset_oracle_id(self) -> None:
        case = {"id": "speaker", "voice": {"kind": "preset_voice", "oracle_id": "54"}}
        self.assertIsNone(validation.speaker_index_for_case(case, "vits-ljspeech"))
        self.assertEqual(validation.speaker_index_for_case(case, "vits-vctk"), 54)

        for voice in (
            {},
            {"kind": "default", "oracle_id": "4"},
            {"kind": "preset_voice", "oracle_id": "-1"},
            {"kind": "preset_voice", "oracle_id": "speaker-004"},
        ):
            with self.subTest(voice=voice), self.assertRaises(ValueError):
                validation.speaker_index_for_case({"id": "bad", "voice": voice}, "vits-vctk")

    def test_appends_speaker_after_backend_only_when_conditioned(self) -> None:
        base = ["runner", "model", "cpu"]
        case = {"id": "speaker", "voice": {"kind": "preset_voice", "oracle_id": "4"}}
        self.assertEqual(
            validation.with_speaker_argument(base, case, "vits-vctk"),
            ["runner", "model", "cpu", "4"],
        )
        self.assertEqual(
            validation.with_speaker_argument(base, case, "vits-ljspeech"), base
        )
        self.assertEqual(base, ["runner", "model", "cpu"])

    def test_profile_aware_report_names_preserve_f32_compatibility(self) -> None:
        self.assertEqual(
            validation.validation_report_name(
                "vits-vctk", "text-encoder", "cpu", "F32"
            ),
            "vits-vctk-text-encoder.json",
        )
        self.assertEqual(
            validation.validation_report_name(
                "vits-vctk", "text-encoder", "cuda", "F32"
            ),
            "vits-vctk-text-encoder-cuda.json",
        )
        self.assertEqual(
            validation.validation_report_name(
                "vits-vctk", "text-encoder", "cpu", "F16"
            ),
            "vits-vctk-F16-text-encoder.json",
        )
        self.assertEqual(
            validation.validation_report_name(
                "vits-vctk", "text-encoder", "cuda", "F16"
            ),
            "vits-vctk-F16-text-encoder-cuda.json",
        )

    def test_reads_and_validates_model_quantization_identity(self) -> None:
        profile = mock.Mock()
        profile.contents.return_value = "F16"
        version = mock.Mock()
        version.contents.return_value = 1
        reader = mock.Mock()
        reader.fields = {
            "synthesize.quantization.profile": profile,
            "synthesize.quantization.profile_version": version,
        }
        with mock.patch.object(validation, "GGUFReader", return_value=reader):
            self.assertEqual(
                validation.model_quantization_identity(Path("model.gguf")),
                ("F16", 1),
            )

        profile.contents.return_value = "f16"
        with mock.patch.object(validation, "GGUFReader", return_value=reader):
            with self.assertRaises(ValueError):
                validation.model_quantization_identity(Path("model.gguf"))


if __name__ == "__main__":
    unittest.main()
