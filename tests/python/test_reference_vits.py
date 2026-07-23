from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_dumper():
    path = PROJECT_ROOT / "scripts" / "dump_reference_vits_pytorch.py"
    spec = importlib.util.spec_from_file_location("synthesize_reference_vits", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load reference dumper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dumper = load_dumper()


class ReferenceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(
            (PROJECT_ROOT / "tests/golden/vits/vits-ljspeech.manifest.json").read_text(
                encoding="utf-8"
            )
        )
        cls.vctk_manifest = json.loads(
            (PROJECT_ROOT / "tests/golden/vits/vits-vctk.manifest.json").read_text(
                encoding="utf-8"
            )
        )

    def test_seed_is_strict_unsigned_decimal(self) -> None:
        self.assertEqual(dumper.parse_seed("0", "case"), 0)
        self.assertEqual(
            dumper.parse_seed(str(dumper.UINT64_MAX), "case"), dumper.UINT64_MAX
        )
        for value in (0, -1, "-1", "1.0", str(dumper.UINT64_MAX + 1)):
            with self.subTest(value=value), self.assertRaises(dumper.DumperError):
                dumper.parse_seed(value, "case")

    def test_manifest_selection_preserves_manifest_order(self) -> None:
        cases = dumper.validate_manifest(copy.deepcopy(self.manifest), None)
        self.assertEqual(len(cases), 12)
        selected = dumper.validate_manifest(
            copy.deepcopy(self.manifest), [cases[2]["id"], cases[0]["id"]]
        )
        self.assertEqual(
            [case["id"] for case in selected], [cases[0]["id"], cases[2]["id"]]
        )
        with self.assertRaises(dumper.DumperError):
            dumper.validate_manifest(copy.deepcopy(self.manifest), [cases[0]["id"]] * 2)
        with self.assertRaises(dumper.DumperError):
            dumper.validate_manifest(copy.deepcopy(self.manifest), ["unknown-case"])

    def test_vctk_manifest_and_speaker_selection(self) -> None:
        cases = dumper.validate_manifest(copy.deepcopy(self.vctk_manifest), None)
        self.assertEqual(len(cases), 12)
        self.assertEqual(cases[0]["voice"]["id"], "speaker-004")
        self.assertIn(
            "voice.embedding",
            [artifact["name"] for artifact in cases[0]["expected"]["artifacts"]],
        )

        model = type("Model", (), {"n_speakers": 109})()
        selected = dumper.resolve_voice(cases[0], model)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.dtype, torch.int64)
        self.assertEqual(selected.tolist(), [4])

        invalid = copy.deepcopy(cases[0])
        invalid["voice"]["oracle_id"] = "109"
        with self.assertRaises(dumper.DumperError):
            dumper.resolve_voice(invalid, model)

    def test_manifest_rejects_bad_seed_rate_and_artifact_format(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["cases"][0]["request"]["seed_u64"] = 0
        with self.assertRaises(dumper.DumperError):
            dumper.validate_manifest(manifest, None)
        manifest = copy.deepcopy(self.manifest)
        manifest["cases"][0]["request"]["speaking_rate"] = True
        with self.assertRaises(dumper.DumperError):
            dumper.validate_manifest(manifest, None)
        manifest = copy.deepcopy(self.manifest)
        manifest["cases"][0]["input"]["artifact"]["format"] = "f32le"
        with self.assertRaises(dumper.DumperError):
            dumper.validate_manifest(manifest, None)

    def test_artifact_paths_cannot_escape_case_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "case"
            root.mkdir()
            self.assertEqual(
                dumper.safe_artifact_path(root, "text/m_p.f32"),
                root / "text/m_p.f32",
            )
            for path in (".", "../escape", str(Path(directory).resolve())):
                with self.subTest(path=path), self.assertRaises(dumper.DumperError):
                    dumper.safe_artifact_path(root, path)

    def test_tensor_payload_is_little_endian_and_shape_preserving(self) -> None:
        tensor = torch.tensor([[1, 2], [3, 4]])
        payload, shape = dumper.tensor_payload(tensor, "i32le")
        self.assertEqual(shape, [2, 2])
        np.testing.assert_array_equal(np.frombuffer(payload, dtype="<i4"), [1, 2, 3, 4])
        with self.assertRaises(dumper.DumperError):
            dumper.tensor_payload(tensor, "unknown")


if __name__ == "__main__":
    unittest.main()
