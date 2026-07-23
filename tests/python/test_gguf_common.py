from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import gguf_common  # noqa: E402


class GgufCommonTest(unittest.TestCase):
    def test_hash_and_project_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "payload.bin"
            payload.write_bytes(b"synthesize")
            self.assertEqual(
                gguf_common.sha256_file(payload),
                hashlib.sha256(b"synthesize").hexdigest(),
            )
            self.assertEqual(gguf_common.project_relative(payload, root), "payload.bin")
            self.assertEqual(
                gguf_common.project_relative(PROJECT_ROOT, root), str(PROJECT_ROOT)
            )
            self.assertIsNone(gguf_common.git_revision(root))

    def test_f32_numpy_converts_and_rejects_invalid_values(self) -> None:
        source = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64).t()
        output = gguf_common.f32_numpy(source)
        self.assertEqual(output.dtype, np.float32)
        self.assertTrue(output.flags.c_contiguous)
        np.testing.assert_array_equal(output, np.array([[1, 3], [2, 4]], np.float32))
        with self.assertRaises(TypeError):
            gguf_common.f32_numpy(torch.tensor([1], dtype=torch.int64))
        for invalid in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                gguf_common.f32_numpy(torch.tensor([invalid]))

    def test_atomic_output_replaces_only_on_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "result.json"
            destination.write_text("old", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                with gguf_common.atomic_output_path(destination) as temporary:
                    temporary.write_text("incomplete", encoding="utf-8")
                    raise RuntimeError("stop")
            self.assertEqual(destination.read_text(encoding="utf-8"), "old")
            self.assertEqual(list(root.glob(".result.json.*.tmp")), [])

            with gguf_common.atomic_output_path(destination) as temporary:
                temporary.write_text("complete", encoding="utf-8")
            self.assertEqual(destination.read_text(encoding="utf-8"), "complete")

    def test_atomic_json_is_sorted_utf8_with_newline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "record.json"
            gguf_common.write_json_atomic(destination, {"z": 1, "name": "音声"})
            payload = destination.read_bytes()
            self.assertTrue(payload.endswith(b"\n"))
            self.assertEqual(json.loads(payload), {"name": "音声", "z": 1})
            self.assertLess(payload.index(b'"name"'), payload.index(b'"z"'))


if __name__ == "__main__":
    unittest.main()
