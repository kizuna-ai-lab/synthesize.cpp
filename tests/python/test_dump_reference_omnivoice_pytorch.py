"""Focused test for the omnivoice pytorch dumper's writer/format agreement.

`write_case`'s format check is a pure function over a `case` dict and a
`produced` dict of (writer, payload) pairs -- it never calls the writer
functions themselves (they are only used as dict keys), so it needs neither a
manifest that satisfies schema validation nor real oracle data. Unlike
`validate-omnivoice-replay.py` (hyphenated, so its test must go through
`importlib`), this script's name is a legal module name and its heavy imports
(torch, transformers, omnivoice) are all function-local, so a plain import
after the `sys.path` insertion is all the loading it needs.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import dump_reference_omnivoice_pytorch as dump  # noqa: E402


class WriteCaseFormatAgreementTest(unittest.TestCase):
    def test_a_writer_that_disagrees_with_the_declared_format_is_refused(self):
        case = {
            "id": "format-mismatch-demo",
            "expected": {"artifacts": [{"name": "audio.pcm", "path": "audio/pcm.f32", "format": "i32le"}]},
        }
        # write_f32 would write f32le bytes; the manifest above declares i32le.
        produced = {"audio.pcm": (dump.write_f32, [0.0, 1.0])}
        with tempfile.TemporaryDirectory() as case_dir:
            with self.assertRaises(SystemExit) as raised:
                dump.write_case(case, Path(case_dir), produced)
        message = str(raised.exception)
        self.assertIn("audio.pcm", message)
        self.assertIn("declared 'i32le'", message)
        self.assertIn("'f32le'", message)

    def test_agreeing_writer_and_format_are_not_refused(self):
        # The negative control: the same shapes, but the writer the manifest's
        # format string actually names -- must not raise, and must produce the
        # artifact record write_case returns on success.
        case = {
            "id": "format-agreement-demo",
            "expected": {"artifacts": [{"name": "audio.pcm", "path": "audio/pcm.f32", "format": "f32le"}]},
        }
        produced = {"audio.pcm": (dump.write_f32, [0.0, 1.0])}
        with tempfile.TemporaryDirectory() as case_dir:
            artifacts = dump.write_case(case, Path(case_dir), produced)
        self.assertIn("audio.pcm", artifacts)
        self.assertEqual(artifacts["audio.pcm"]["path"], "audio/pcm.f32")


if __name__ == "__main__":
    unittest.main()
