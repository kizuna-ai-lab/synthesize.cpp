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


def base_fixture_spec() -> dict:
    """A minimal, valid spec other tests customize with `copy.deepcopy`.

    Shape mirrors `test_render_exposes_validation_and_quality_status`'s own
    fixture so the two stay interchangeable; kept as a function rather than a
    shared constant so each test mutates its own copy.
    """
    return {
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
        "license_name": "upstream-terms",
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
            "voice_mode": "preset_catalog",
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
                "validation": {"cpu_metric": 0},
            }
        ],
    }


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

    # -- Task 13, feature 1: Sidecar Resources -------------------------------

    def test_sidecar_entries_are_validated_and_rendered(self) -> None:
        spec = base_fixture_spec()
        spec["sidecars"] = [
            {
                "filename": "LICENSE-example.txt",
                "role": "Example Community License",
                "size": "9.0 KB",
                "size_bytes": 9171,
                "sha256": "b" * 64,
            }
        ]
        self.generator.validate_spec(spec)
        card = self.generator.render(spec, "# stub")
        self.assertIn("## Sidecar files", card)
        self.assertIn("Example Community License", card)
        self.assertIn(
            "handy-computer/fixture-gguf/resolve/main/LICENSE-example.txt", card
        )
        self.assertIn("`" + "b" * 64 + "`", card)

    def test_sidecars_are_absent_from_a_spec_without_them(self) -> None:
        spec = base_fixture_spec()
        card = self.generator.render(spec, "# stub")
        self.assertNotIn("## Sidecar files", card)

    def test_sidecar_entry_missing_a_key_is_rejected(self) -> None:
        spec = base_fixture_spec()
        spec["sidecars"] = [{"filename": "LICENSE.txt", "role": "License", "size": "1 KB"}]
        with self.assertRaisesRegex(ValueError, "sidecar entry is missing"):
            self.generator.validate_spec(spec)

    def test_sidecar_filename_must_be_flat(self) -> None:
        spec = base_fixture_spec()
        spec["sidecars"] = [
            {
                "filename": "licenses/LICENSE.txt",
                "role": "License",
                "size": "1 KB",
                "size_bytes": 10,
                "sha256": "c" * 64,
            }
        ]
        with self.assertRaisesRegex(ValueError, "flat repository path"):
            self.generator.validate_spec(spec)

    def test_sidecar_sha256_must_be_hexadecimal(self) -> None:
        spec = base_fixture_spec()
        spec["sidecars"] = [
            {
                "filename": "LICENSE.txt",
                "role": "License",
                "size": "1 KB",
                "size_bytes": 10,
                "sha256": "not-hex-" + "0" * 56,
            }
        ]
        with self.assertRaisesRegex(ValueError, "not hexadecimal"):
            self.generator.validate_spec(spec)

    def test_sidecar_filename_must_not_collide_with_a_quant_filename(self) -> None:
        spec = base_fixture_spec()
        spec["sidecars"] = [
            {
                "filename": "fixture-F16.gguf",
                "role": "License",
                "size": "1 byte",
                "size_bytes": 1,
                "sha256": "d" * 64,
            }
        ]
        with self.assertRaisesRegex(ValueError, "distinct from quant filenames"):
            self.generator.validate_spec(spec)

    def test_sidecar_artifact_validation_checks_size_and_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            payload = b"a license text sidecar"
            (model_dir / "LICENSE-example.txt").write_bytes(payload)
            spec = {
                "model_slug": "fixture",
                "target_repo": "handy-computer/fixture-gguf",
                "quants": [],
                "sidecars": [
                    {
                        "filename": "LICENSE-example.txt",
                        "role": "Example License",
                        "size": "23 bytes",
                        "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }
            self.generator.validate_artifacts(spec, model_dir)
            spec["sidecars"][0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                self.generator.validate_artifacts(spec, model_dir)

    # -- Task 13, feature 2: text input (no hardcoded phoneme/G2P prose) ----

    def test_text_utf8_frontend_renders_without_the_g2p_disclaimer(self) -> None:
        spec = base_fixture_spec()
        spec["capabilities"]["input_kinds"] = ["text_utf8", "token_ids"]
        spec["capabilities"]["frontend_provider"] = "synthesize.qwen_bpe"
        spec["usage"] = {"profile": "F16", "voice": "speaker-000", "text": "Hello there."}

        card = self.generator.render(spec, "# stub")
        self.assertIn("accepts raw UTF-8 text", card)
        self.assertIn("`synthesize.qwen_bpe` frontend", card)
        self.assertNotIn("does not perform grapheme-to-phoneme conversion", card)
        self.assertNotIn("must currently run a compatible", card)

    def test_phonemes_utf8_frontend_still_renders_the_g2p_disclaimer(self) -> None:
        spec = base_fixture_spec()
        card = self.generator.render(spec, "# stub")
        self.assertIn("accepts UTF-8 phoneme strings", card)
        self.assertIn("does not perform grapheme-to-phoneme conversion", card)

    # -- Task 14 fix: the "also accepts exact token IDs" clause is only true
    # for a package that actually declares that input kind (OmniVoice's
    # loader refuses to load any package whose input_flags is not EXACTLY
    # SYNTH_INPUT_SUPPORT_TEXT_UTF8, so this family has no token-ID bypass
    # to claim) -----------------------------------------------------------

    def test_text_utf8_frontend_omits_the_token_id_clause_when_undeclared(self) -> None:
        spec = base_fixture_spec()
        spec["capabilities"]["input_kinds"] = ["text_utf8"]
        spec["capabilities"]["frontend_provider"] = "synthesize.qwen_bpe"
        spec["usage"] = {"profile": "F16", "voice": "speaker-000", "text": "Hello there."}

        card = self.generator.render(spec, "# stub")
        self.assertIn("accepts raw UTF-8 text", card)
        self.assertNotIn("also accepts exact token", card)

    def test_phonemes_utf8_frontend_omits_the_token_id_clause_when_undeclared(self) -> None:
        spec = base_fixture_spec()
        spec["capabilities"]["input_kinds"] = ["phonemes_utf8"]

        card = self.generator.render(spec, "# stub")
        self.assertIn("accepts UTF-8 phoneme strings", card)
        self.assertNotIn("also accepts exact token", card)

    def test_omnivoice_spec_declares_text_only_input_with_no_token_id_claim(self) -> None:
        # Regression for a real (non-fixture) spec: OmniVoice's public seam
        # dispatches straight from the raw request bytes to its own text
        # frontend (src/synthesize.cpp's Omnivoice branch) and its loader
        # refuses any package declaring more than SYNTH_INPUT_SUPPORT_TEXT_UTF8,
        # so the card must not claim a token-ID bypass this family cannot
        # accept.
        spec = self.generator.load_spec(ROOT / "scripts" / "hf_cards" / "omnivoice-0-6b.yaml")
        self.generator.validate_spec(spec)
        self.assertEqual(spec["capabilities"]["input_kinds"], ["text_utf8"])
        card = self.generator.render(spec, "# stub upstream card")
        self.assertIn("accepts raw UTF-8 text", card)
        self.assertNotIn("also accepts exact token", card)

    # -- 2026-08-09: per-profile notes in the download table -----------------

    def test_download_table_has_no_notes_column_when_no_quant_declares_one(
        self,
    ) -> None:
        # The column is opt-in per spec. Every family that shipped before
        # 2026-08-09 declares no `note`, and their cards must render exactly as
        # they did -- `--check` compares byte-for-byte against a committed
        # README, so a stray column or a stray separator cell breaks all of
        # them at once.
        spec = base_fixture_spec()
        card = self.generator.render(spec, "# stub")
        self.assertIn("| Profile | Download | Size | Tensor storage | SHA-256 |\n", card)
        self.assertIn("| --- | --- | ---: | --- | --- |\n", card)
        self.assertNotIn("What to know before choosing it", card)

    def test_a_quant_note_renders_in_that_quant_s_own_table_row(self) -> None:
        # The point of the field, and the reason it is not a paragraph under
        # the table: a reader picking a file out of the download table has to
        # meet the caveat in the row they are picking. Assert the note is on
        # the same LINE as the profile it belongs to, and that a profile
        # without one gets a placeholder rather than shifting the columns.
        spec = base_fixture_spec()
        spec["quants"].append(
            {
                "name": "Q8",
                "filename": "fixture-Q8.gguf",
                "size": "1 byte",
                "size_bytes": 1,
                "sha256": "b" * 64,
                "tensor_types": "1 Q8_0",
                "note": "Renders a different voice than F16 does.",
                "validation": {"cpu_metric": 0},
            }
        )
        card = self.generator.render(spec, "# stub")
        self.assertIn(
            "| Profile | Download | Size | Tensor storage | SHA-256 |"
            " What to know before choosing it |\n",
            card,
        )
        self.assertIn("| --- | --- | ---: | --- | --- | --- |\n", card)

        # The validation table further down repeats every profile name in the
        # same leading-cell position, so match on the download link too.
        rows = {
            line.split("|")[1].strip(): line
            for line in card.splitlines()
            if line.startswith(("| F16 |", "| Q8 |")) and "/resolve/main/" in line
        }
        self.assertEqual(set(rows), {"F16", "Q8"})
        self.assertTrue(rows["Q8"].endswith("Renders a different voice than F16 does. |"))
        # The un-noted profile keeps the cell, so the table stays rectangular.
        self.assertTrue(rows["F16"].endswith(" | — |"))
        self.assertNotIn("Renders a different voice", rows["F16"])

    def test_downloads_note_is_optional_and_renders_under_the_table(self) -> None:
        # Same optional-field shape as speaker_backend_variation, and added for
        # the same reason: the generator runs Jinja under StrictUndefined, so a
        # template reading an absent key raises for every family that does not
        # set it. Both branches need a test.
        spec = base_fixture_spec()
        without = self.generator.render(spec, "# stub")
        self.assertIn("profile name describes a versioned storage policy", without)

        spec["downloads_note"] = "**No profile here is faster than F32.**"
        with_note = self.generator.render(spec, "# stub")
        self.assertIn(
            "not the language or Execution\nBackend.\n\n"
            "**No profile here is faster than F32.**\n",
            with_note,
        )

    def test_omnivoice_ships_three_profiles_with_q8_s_caveat_in_its_own_row(
        self,
    ) -> None:
        # Regression for the real spec. Three things this family's card must
        # not lose, each of which a well-meaning edit has already got wrong at
        # least once on this branch:
        #
        #   1. F16 and Q8 are listed at all, under their post-rename names.
        #   2. Q8's caveat is IN the Q8 download row, not only in prose below.
        #   3. No profile is described as faster, in any spelling. Measured
        #      against F32: on CUDA Q8 -0.4% and F16 +2.7%; on CPU Q8 +1.5% and
        #      F16 -1.2%, positive being the slower one. The generator is
        #      compute-bound at ~205 MAC per weight byte, so a narrower weight
        #      buys size, memory and load time, never speed.
        spec = self.generator.load_spec(ROOT / "scripts" / "hf_cards" / "omnivoice-0-6b.yaml")
        self.generator.validate_spec(spec)
        self.assertEqual([quant["name"] for quant in spec["quants"]], ["F32", "F16", "Q8"])

        card = self.generator.render(spec, "# stub upstream card")
        q8_row = next(
            line
            for line in card.splitlines()
            if line.startswith("| Q8 |") and "/resolve/main/" in line
        )
        self.assertIn("different voice", q8_row)
        self.assertIn("Reference Audio cloning is not affected", q8_row)
        self.assertIn("Nothing is disabled", q8_row)
        # The two framings jiangzhuo corrected, which the row must never
        # reintroduce: Q8 is not clone-only, and it does honor a Description
        # Text prompt (it renders a different voice WITHIN the description).
        self.assertNotIn("clone-only", card)
        self.assertIn("not an ignored one", q8_row)

        f16_row = next(
            line
            for line in card.splitlines()
            if line.startswith("| F16 |") and "/resolve/main/" in line
        )
        self.assertIn("Recommended default", f16_row)

        # Every use of "faster" on the card must be a denial. Checked by
        # enumerating the occurrences rather than by a substring blacklist, so
        # a newly invented speed claim cannot slip past a phrase this test did
        # not think to forbid.
        lowered = card.lower()
        allowed = {
            "no profile here is faster than f32, and none claims to be.",
            "not faster than f32.",
        }
        for index, _ in enumerate(lowered):
            if lowered.startswith("faster", index):
                context = lowered[max(0, index - 80) : index + 80]
                self.assertTrue(
                    any(phrase in context for phrase in allowed),
                    f"the card must make no speed claim; found ...{context}...",
                )
        # Enumerating "faster" is not enough, and this test used to claim it
        # was. A speed benefit has more than one spelling: the paragraph three
        # lines below the denial above once read "Q8 is 0.4% quicker ... F16
        # 1.2% quicker", which asserts exactly what the denial denies and which
        # a search for "faster" waves straight through. These spellings have no
        # legitimate use on this card, so they are forbidden outright rather
        # than allowed in a denial context.
        for forbidden in (
            "quicker",
            "speedup",
            "speed-up",
            "speeds up",
            "speed boost",
            "x faster",
            "% faster",
        ):
            self.assertNotIn(forbidden, lowered)

    def test_omnivoice_scopes_its_exact_token_claim_to_the_grid_that_gates_it(
        self,
    ) -> None:
        # Regression for a claim that shipped wrong in the published artifact.
        # The card asserted that "a profile or a backend that flips even one of
        # those tokens is not shipped" -- while the same card lists Q8, which
        # commits a different token at 95.83% of greedy positions, and
        # describes a CUDA backend that re-draws the grid by design. This
        # family has TWO token grids and only one of them is a gate:
        #
        #   * the greedy decode grid is certified against the oracle in the
        #     reference configuration only, F32 on CPU. A generator-half
        #     profile re-draws it because its weights are different, and the
        #     CUDA backend re-draws it because TF32 moves the per-step argmax.
        #     Both were accepted by listening, and both flip counts are data.
        #   * the cloning RVQ-encode grid IS gated unconditionally, and holds
        #     on all three shipped profiles because their codec halves are
        #     bit-identical to F32's. It is what blocks every codec-half
        #     profile.
        #
        # Checked by enumerating every oracle-parity claim rather than by
        # blacklisting the sentence that was wrong: the defect is an unscoped
        # claim, and a blacklist cannot name the spellings nobody has written
        # yet.
        spec = self.generator.load_spec(ROOT / "scripts" / "hf_cards" / "omnivoice-0-6b.yaml")
        self.generator.validate_spec(spec)
        card = self.generator.render(spec, "# stub upstream card")
        lowered = card.lower()

        for index, _ in enumerate(lowered):
            if lowered.startswith("byte-for-byte", index):
                context = lowered[index : index + 160]
                self.assertTrue(
                    "f32 on cpu" in context or "clone" in context,
                    "every oracle-parity claim must name the configuration or the"
                    f" grid it holds for; found ...{context}...",
                )

        # The gated claim, stated as the gate, with the count that makes it
        # checkable.
        self.assertIn(
            "A profile that flips even one of those 2,808 tokens is not shipped",
            card,
        )
        # And the re-draws stated as re-draws, not hidden behind the gate.
        self.assertIn("re-draws that grid by design", card)
        self.assertIn("95.83%", card)

    def test_omnivoice_default_readme_matches_the_generated_card(self) -> None:
        # This family's working directory (models/omnivoice-0-6b) is not flat
        # -- it holds the raw upstream checkpoint beside our GGUFs, unlike
        # every sibling family -- but generate.py's own model_dir resolution
        # (REPO_ROOT / "models" / model_slug) still targets it by default,
        # exactly as it does for VITS and Kokoro. A defect caught after this
        # task's first pass: the card had only ever been written to a
        # --output override (models/publish/omnivoice-0-6b/README.md), which
        # left the plain `generate.py omnivoice-0-6b.yaml --check` invocation
        # -- the one every sibling family answers with exit 0 -- checking a
        # stale copy of the *upstream* README instead, silently, because
        # nothing exercised the no-argument path. This pins the default path
        # too, the same way the sibling `test_generated_*_is_flat` tests do.
        spec = self.generator.load_spec(ROOT / "scripts" / "hf_cards" / "omnivoice-0-6b.yaml")
        self.generator.validate_spec(spec)
        model_dir = ROOT / "models" / "omnivoice-0-6b"
        if not model_dir.is_dir():
            self.skipTest("the OmniVoice packages have not been materialized locally")
        self.generator.validate_artifacts(spec, model_dir)
        expected_card = self.generator.render(spec, self.generator.load_upstream_card(spec))
        self.assertEqual((model_dir / "README.md").read_text(encoding="utf-8"), expected_card)
        # The upstream card was moved to models/upstream/omnivoice-0-6b/, not
        # overwritten in place, so source.card_path has a stable home that is
        # never mixed with our own artifacts (the Kokoro precedent). Confirm
        # it is still genuinely the upstream card, not a copy of ours.
        self.assertNotIn("synthesize.cpp", self.generator.load_upstream_card(spec))

    def test_omnivoice_publish_directory_is_flat_and_current(self) -> None:
        # models/publish/omnivoice-0-6b/ is the clean, flat publication
        # directory this family needs precisely because its working
        # directory is not flat (see the test above). It must hold the GGUF,
        # the declared sidecar, and the generated README -- and nothing else.
        spec = self.generator.load_spec(ROOT / "scripts" / "hf_cards" / "omnivoice-0-6b.yaml")
        self.generator.validate_spec(spec)
        publish_dir = ROOT / "models" / "publish" / "omnivoice-0-6b"
        if not publish_dir.is_dir():
            self.skipTest("the OmniVoice publication directory has not been built locally")
        self.generator.validate_artifacts(spec, publish_dir)
        expected_card = self.generator.render(spec, self.generator.load_upstream_card(spec))
        self.assertEqual((publish_dir / "README.md").read_text(encoding="utf-8"), expected_card)

        expected_files = {
            "README.md",
            *(quant["filename"] for quant in spec["quants"]),
            *(sidecar["filename"] for sidecar in spec.get("sidecars", [])),
        }
        actual_files = {path.name for path in publish_dir.iterdir() if path.is_file()}
        self.assertEqual(actual_files, expected_files)

    # -- Task 13, feature 3: usage (--text vs --phonemes) --------------------

    def test_usage_renders_text_flag_for_a_text_utf8_family(self) -> None:
        spec = base_fixture_spec()
        spec["capabilities"]["input_kinds"] = ["text_utf8", "token_ids"]
        spec["usage"] = {"profile": "F16", "voice": "speaker-000", "text": "Hello there."}

        card = self.generator.render(spec, "# stub")
        self.assertIn('--text "Hello there." \\', card)
        self.assertNotIn("--phonemes", card)

    def test_usage_requires_a_text_example_for_a_text_utf8_frontend(self) -> None:
        spec = base_fixture_spec()
        spec["capabilities"]["input_kinds"] = ["text_utf8", "token_ids"]
        del spec["usage"]["phonemes"]
        with self.assertRaisesRegex(ValueError, "text example"):
            self.generator.validate_spec(spec)

    def test_qwen3_tts_spec_declares_text_input_and_renders_the_text_flag(self) -> None:
        # Regression for a real (non-fixture) spec: this family's frontend is
        # `synthesize.qwen_bpe` over raw text, and its usage example was
        # mislabeled `phonemes` before Task 13 -- the old generator never
        # noticed because it hardcoded `--phonemes` for every family.
        spec = self.generator.load_spec(
            ROOT / "scripts" / "hf_cards" / "qwen3-tts-12hz-0-6b-customvoice.yaml"
        )
        self.generator.validate_spec(spec)
        self.assertIn("text_utf8", spec["capabilities"]["input_kinds"])
        self.assertIn("text", spec["usage"])
        card = self.generator.render(spec, "# stub upstream card")
        self.assertIn('--text "Qwen3-TTS is awesome!" \\', card)
        self.assertNotIn("--phonemes", card)
        self.assertIn("accepts raw UTF-8 text", card)

    # -- Task 13, feature 4: the seed_default_with_profiles voice mode ------

    def test_seed_default_with_profiles_renders_both_sources(self) -> None:
        spec = base_fixture_spec()
        spec["capabilities"]["voice_mode"] = "seed_default_with_profiles"
        spec["capabilities"]["voice_profile_sources"] = ["reference_audio", "description_text"]
        del spec["usage"]["voice"]

        self.generator.validate_spec(spec)
        card = self.generator.render(spec, "# stub")
        self.assertIn("no preset speaker catalog", card)
        # Amended 2026-08-08. This asserted "follows the synthesis seed" until
        # the OmniVoice step-count listening audit measured the claim false:
        # auto-voice supplies no speaker conditioning at all, and the same
        # seed on the CPU and CUDA backends can produce different speakers
        # (omni-short-en, 118 Hz vs 189 Hz median F0). The card must state the
        # seed is necessary and not sufficient, and must name the backend as
        # part of what has to be held fixed -- a caller reading the old
        # sentence would have believed in a reproducibility this family does
        # not provide. See reports/porting/omnivoice/omnivoice-0-6b/
        # _porting-log.md, the 2026-08-08 step-count audit entry, Finding 2.
        self.assertIn("no speaker conditioning", card)
        self.assertIn("the synthesis seed alone does\nnot pin it", card)
        self.assertIn("Execution Backend", card)
        self.assertNotIn("follows the synthesis seed", card)
        self.assertIn("Reference Audio (voice cloning)", card)
        self.assertIn("Description Text (voice design)", card)
        self.assertNotIn("--voice ", card)
        # The "quality evaluation has not been run" paragraph names the
        # request path too; an unconditioned default is not "fixed", so it
        # must not fall into the fixed_default wording by default-branch
        # accident.
        self.assertIn("the unconditioned default-Voice request path", card)
        self.assertNotIn("the fixed-Voice request path", card)

    def test_seed_default_with_profiles_renders_a_single_declared_source(self) -> None:
        spec = base_fixture_spec()
        spec["capabilities"]["voice_mode"] = "seed_default_with_profiles"
        spec["capabilities"]["voice_profile_sources"] = ["reference_audio"]
        del spec["usage"]["voice"]

        self.generator.validate_spec(spec)
        card = self.generator.render(spec, "# stub")
        self.assertIn("Reference Audio (voice cloning)", card)
        self.assertNotIn("Description Text", card)

    def test_speaker_backend_variation_is_optional_and_rendered_when_present(
        self,
    ) -> None:
        # Added 2026-08-08 with the field itself. The two tests above cover
        # only its ABSENCE, which is how the field first shipped broken: the
        # template read `capabilities.speaker_backend_variation` directly and
        # the generator runs Jinja under StrictUndefined, so every family that
        # does not set it raised UndefinedError. The field is optional by
        # design -- it carries a per-family measurement and must not become a
        # required key in the shared template -- so both branches need a test.
        # The measurement it carries for OmniVoice: one of seventeen greedy
        # Golden cases changes speaker between the CPU and CUDA backends, with
        # a listener confirming that pair as two different speakers of equal
        # quality. See the 2026-08-08 backend speaker audit in
        # reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md.
        spec = base_fixture_spec()
        spec["capabilities"]["voice_mode"] = "seed_default_with_profiles"
        spec["capabilities"]["voice_profile_sources"] = ["reference_audio"]
        del spec["usage"]["voice"]

        # Absent: renders, and says nothing about a measured rate.
        self.generator.validate_spec(spec)
        without = self.generator.render(spec, "# stub")
        self.assertNotIn("Measured on this", without)

        # Present: the sentence appears, joined to the preceding one rather
        # than glued to it, and the family's own wording is passed through.
        spec["capabilities"]["speaker_backend_variation"] = (
            "one case in seventeen changes speaker between the CPU and CUDA "
            "backends"
        )
        self.generator.validate_spec(spec)
        with_note = self.generator.render(spec, "# stub")
        self.assertIn(
            "Measured on this\npackage's own validation cases, one case in "
            "seventeen changes speaker",
            with_note,
        )
        self.assertIn("backends. Callers who want a stable identity", with_note)

    def test_seed_default_with_profiles_requires_a_declared_source(self) -> None:
        spec = base_fixture_spec()
        spec["capabilities"]["voice_mode"] = "seed_default_with_profiles"
        del spec["usage"]["voice"]
        with self.assertRaisesRegex(ValueError, "voice_profile_sources"):
            self.generator.validate_spec(spec)

    def test_seed_default_with_profiles_rejects_an_explicit_usage_voice(self) -> None:
        spec = base_fixture_spec()
        spec["capabilities"]["voice_mode"] = "seed_default_with_profiles"
        spec["capabilities"]["voice_profile_sources"] = ["reference_audio"]
        with self.assertRaisesRegex(ValueError, "seed-default"):
            self.generator.validate_spec(spec)

    # -- Task 13, feature 5: the conditional CUDA placement sentence --------

    def test_cuda_placement_defaults_to_the_unconditional_full_sentence(self) -> None:
        spec = base_fixture_spec()
        card = self.generator.render(spec, "# stub")
        self.assertIn(
            "case. CUDA placement contained zero executable CPU fallback nodes.", card
        )

    def test_cuda_placement_none_omits_the_cuda_sentence(self) -> None:
        spec = base_fixture_spec()
        spec["validation"]["cuda_placement"] = "none"
        self.generator.validate_spec(spec)
        card = self.generator.render(spec, "# stub")
        self.assertNotIn("CUDA placement contained zero", card)
        self.assertIn("Duration structure was exact in every\ncase.", card)

    def test_cuda_placement_partial_renders_the_declared_note_instead(self) -> None:
        spec = base_fixture_spec()
        spec["validation"]["cuda_placement"] = "partial"
        spec["validation"]["cuda_placement_note"] = (
            "Only the codec moved to CUDA; the generator stays on CPU by design."
        )
        self.generator.validate_spec(spec)
        card = self.generator.render(spec, "# stub")
        self.assertNotIn("CUDA placement contained zero executable CPU fallback nodes.", card)
        self.assertIn("Only the codec moved to CUDA; the generator stays on CPU by design.", card)

    def test_cuda_placement_partial_requires_a_note(self) -> None:
        spec = base_fixture_spec()
        spec["validation"]["cuda_placement"] = "partial"
        with self.assertRaisesRegex(ValueError, "cuda_placement_note"):
            self.generator.validate_spec(spec)

    def test_cuda_placement_rejects_an_unknown_value(self) -> None:
        spec = base_fixture_spec()
        spec["validation"]["cuda_placement"] = "everywhere"
        with self.assertRaisesRegex(ValueError, "cuda_placement"):
            self.generator.validate_spec(spec)


if __name__ == "__main__":
    unittest.main()
