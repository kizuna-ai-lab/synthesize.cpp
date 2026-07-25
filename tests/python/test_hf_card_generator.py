from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = ROOT / "scripts" / "hf_cards" / "generate.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("synthesize_hf_card_generator", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HuggingFaceCardGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generator = load_generator()

    def test_artifact_validation_checks_size_and_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            payload = b"deterministic-gguf-fixture"
            (model_dir / "fixture-F16.gguf").write_bytes(payload)
            spec = {
                "model_slug": "fixture",
                "target_repo": "handy-computer/fixture-gguf",
                "quants": [
                    {
                        "name": "F16",
                        "filename": "fixture-F16.gguf",
                        "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }

            self.generator.validate_artifacts(spec, model_dir)
            spec["quants"][0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                self.generator.validate_artifacts(spec, model_dir)

    def test_render_exposes_validation_and_quality_status(self) -> None:
        spec = {
            "model_slug": "fixture",
            "display_name": "Fixture TTS",
            "model_family": "fixture",
            "target_repo": "handy-computer/fixture-gguf",
            "source": {
                "repository_label": "example/source",
                "repository": "https://example.com/source",
                "revision": "0123456789abcdef",
                "card_url": "https://example.com/source/card",
                "checkpoint_url": "https://example.com/source/checkpoint",
            },
            "license": "other",
            "license_name": "Upstream terms",
            "license_link": "https://example.com/source/license",
            "library_name": "synthesize.cpp",
            "pipeline_tag": "text-to-speech",
            "languages": ["en"],
            "tags": ["gguf", "synthesize.cpp", "text-to-speech"],
            "summary": "A deterministic fixture.",
            "capabilities": {
                "sample_rate_hz": 22050,
                "voice_count": 2,
                "voice_ids": "speaker-000..speaker-001",
                "input_kinds": ["phonemes_utf8", "token_ids"],
                "frontend_provider": "synthesize.symbol_map",
            },
            "validation": {
                "level": "port_validated",
                "quality_evaluation": "not_run",
                "date": "2026-07-23",
                "reference": "fixture reference",
                "cases_per_stage": 12,
                "stages": 7,
                "columns": [{"key": "cpu_metric", "title": "CPU metric"}],
                "metric_note": "The metric is a fixture.",
                "platforms": ["CPU", "CUDA"],
            },
            "architecture_label": "Fixture",
            "license_note": "Upstream terms apply: [licence]({{ license_link }}).",
            "publication_note": "Terms were reviewed by the maintainer.",
            "usage": {
                "profile": "F16",
                "voice": "speaker-000",
                "phonemes": "ˈeɪ.",
            },
            "quants": [
                {
                    "name": "F16",
                    "filename": "fixture-F16.gguf",
                    "size": "1 byte",
                    "size_bytes": 1,
                    "sha256": "a" * 64,
                    "tensor_types": "1 F16",
                    "validation": {
                        "cpu_metric": 0,
                        "dgx_pcm_max_abs": 0,
                        "rtx_pcm_max_abs": 0,
                    },
                }
            ],
        }

        card = self.generator.render(spec, "# Original model card")

        self.assertIn("library_name: synthesize.cpp", card)
        self.assertIn("pipeline_tag: text-to-speech", card)
        self.assertIn("validation_level: port_validated", card)
        self.assertIn("quality_evaluation: not_run", card)
        self.assertIn("handy-computer/fixture-gguf/resolve/main/fixture-F16.gguf", card)
        self.assertIn("Quality evaluation has not been run", card)
        self.assertIn('--phonemes "ˈeɪ."', card)
        self.assertIn("`synthesize.symbol_map` frontend", card)
        self.assertIn("does not perform grapheme-to-phoneme conversion", card)
        self.assertIn("# Original model card", card)

    def test_vctk_spec_matches_local_artifacts(self) -> None:
        spec = self.generator.load_spec(ROOT / "scripts" / "hf_cards" / "vits-vctk.yaml")
        self.generator.validate_spec(spec)
        self.generator.validate_artifacts(spec, ROOT / "models" / "vits-vctk")
        self.assertEqual(spec["target_repo"], "jiangzhuo9357/vits-vctk-gguf")
        self.assertEqual([quant["name"] for quant in spec["quants"]], ["F32", "F16", "Q8_MIXED"])

    def test_quantization_reports_match_current_artifacts(self) -> None:
        for model_slug in ("vits-ljspeech", "vits-vctk"):
            model_dir = ROOT / "models" / model_slug
            report_dir = ROOT / "reports" / "convert" / "vits"
            for profile in ("F16", "Q8_MIXED"):
                report = json.loads(
                    (report_dir / f"{model_slug}-{profile}.json").read_text(encoding="utf-8")
                )
                for section in ("source", "output"):
                    artifact = ROOT / report[section]["path"]
                    self.assertEqual(artifact.parent, model_dir)
                    self.assertEqual(report[section]["bytes"], artifact.stat().st_size)
                    self.assertEqual(
                        report[section]["sha256"],
                        self.generator.file_sha256(artifact),
                    )

    def test_license_name_must_be_a_hugging_face_slug(self) -> None:
        spec = self.generator.load_spec(ROOT / "scripts" / "hf_cards" / "vits-vctk.yaml")
        spec["license_name"] = "Invalid License Name"
        with self.assertRaisesRegex(ValueError, "license_name"):
            self.generator.validate_spec(spec)

    def test_fixed_default_voice_rejects_an_explicit_usage_voice(self) -> None:
        spec = self.generator.load_spec(ROOT / "scripts" / "hf_cards" / "vits-vctk.yaml")
        spec["capabilities"]["voice_mode"] = "fixed_default"
        with self.assertRaisesRegex(ValueError, "fixed-default"):
            self.generator.validate_spec(spec)

    def test_usage_requires_a_phoneme_example(self) -> None:
        spec = self.generator.load_spec(ROOT / "scripts" / "hf_cards" / "vits-vctk.yaml")
        del spec["usage"]["phonemes"]
        with self.assertRaisesRegex(ValueError, "phoneme"):
            self.generator.validate_spec(spec)

    def test_generated_vctk_payload_is_current_and_flat(self) -> None:
        spec = self.generator.load_spec(ROOT / "scripts" / "hf_cards" / "vits-vctk.yaml")
        model_dir = ROOT / "models" / "vits-vctk"
        card_path = model_dir / "README.md"
        expected_card = self.generator.render(spec, self.generator.load_upstream_card(spec))
        self.assertEqual(card_path.read_text(encoding="utf-8"), expected_card)

        _, frontmatter, _ = expected_card.split("---", 2)
        metadata = yaml.safe_load(frontmatter)
        self.assertEqual(metadata["library_name"], "synthesize.cpp")
        self.assertEqual(metadata["pipeline_tag"], "text-to-speech")
        self.assertEqual(metadata["synthesize_cpp"]["validation_level"], "port_validated")
        self.assertEqual(metadata["synthesize_cpp"]["quality_evaluation"], "not_run")
        self.assertEqual(metadata["synthesize_cpp"]["input_kinds"], ["phonemes_utf8", "token_ids"])
        self.assertEqual(metadata["synthesize_cpp"]["frontend_provider"], "synthesize.symbol_map")

        expected_files = {"README.md", *(quant["filename"] for quant in spec["quants"])}
        actual_files = {path.name for path in model_dir.iterdir() if path.is_file()}
        self.assertEqual(actual_files, expected_files)

    def test_generated_kokoro_payload_requires_a_voice_and_is_flat(self) -> None:
        spec = self.generator.load_spec(ROOT / "scripts" / "hf_cards" / "kokoro-v1-0.yaml")
        self.generator.validate_spec(spec)
        model_dir = ROOT / "models" / "kokoro-v1-0"
        if not model_dir.is_dir():
            self.skipTest("the Kokoro packages have not been materialized locally")
        self.generator.validate_artifacts(spec, model_dir)
        self.assertEqual(spec["capabilities"]["voice_mode"], "preset_catalog")
        self.assertEqual(spec["capabilities"]["voice_count"], 54)
        self.assertEqual(spec["capabilities"]["sample_rate_hz"], 24000)

        expected_card = self.generator.render(spec, self.generator.load_upstream_card(spec))
        self.assertEqual((model_dir / "README.md").read_text(encoding="utf-8"), expected_card)

        # The upstream card names CC BY training corpora, so the generated card
        # has to carry that attribution forward rather than only the licence.
        self.assertIn("Koniwa", expected_card)
        self.assertIn("SIWIS", expected_card)

        expected_files = {"README.md", *(quant["filename"] for quant in spec["quants"])}
        actual_files = {path.name for path in model_dir.iterdir() if path.is_file()}
        self.assertEqual(actual_files, expected_files)

    def test_generated_ljspeech_payload_uses_fixed_default_voice_and_is_flat(self) -> None:
        spec = self.generator.load_spec(ROOT / "scripts" / "hf_cards" / "vits-ljspeech.yaml")
        self.generator.validate_spec(spec)
        self.generator.validate_artifacts(spec, ROOT / "models" / "vits-ljspeech")
        self.assertEqual(spec["target_repo"], "jiangzhuo9357/vits-ljspeech-gguf")
        self.assertEqual(spec["capabilities"]["voice_mode"], "fixed_default")
        self.assertEqual(spec["usage"]["phonemes"], "ˈeɪ.")

        model_dir = ROOT / "models" / "vits-ljspeech"
        card_path = model_dir / "README.md"
        expected_card = self.generator.render(spec, self.generator.load_upstream_card(spec))
        self.assertEqual(card_path.read_text(encoding="utf-8"), expected_card)
        self.assertIn("one fixed package-default Voice", expected_card)
        self.assertNotIn("--voice ", expected_card)
        self.assertIn('--phonemes "ˈeɪ." \\', expected_card)
        self.assertNotIn("--token-ids", expected_card)
        self.assertIn("  --language en \\\n  --seed 0", expected_card)
        self.assertNotIn("Voice selection, deterministic request path", expected_card)

        expected_files = {"README.md", *(quant["filename"] for quant in spec["quants"])}
        actual_files = {path.name for path in model_dir.iterdir() if path.is_file()}
        self.assertEqual(actual_files, expected_files)


if __name__ == "__main__":
    unittest.main()
