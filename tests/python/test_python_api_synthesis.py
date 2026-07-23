import array
import importlib
import math
import sys
import types
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PACKAGE_SOURCE = REPO / "bindings/python/src/synthesize_cpp"


class FakeNative(types.ModuleType):
    def __init__(self):
        super().__init__("synthesize_cpp._native")
        self.calls = []

    def synthesize(self, *arguments):
        self.calls.append(arguments)
        return {
            "audio_owner": array.array("f", [0.25, -0.5, 0.75, -1.0]),
            "frame_count": 2,
            "sample_rate": 24000,
            "channel_count": 2,
            "frames_emitted": 2,
            "actual_seed": 42,
            "result_flags": 1,
            "resolved_language_tag": "en-US",
            "resolved_voice_id": None,
        }


class PythonApiSynthesisTests(unittest.TestCase):
    def tearDown(self):
        for name in tuple(sys.modules):
            if name == "synthesize_cpp" or name.startswith("synthesize_cpp."):
                sys.modules.pop(name, None)

    def load_audio_module(self, native=None):
        package = types.ModuleType("synthesize_cpp")
        package.__path__ = [str(PACKAGE_SOURCE)]
        package.__package__ = "synthesize_cpp"
        sys.modules["synthesize_cpp"] = package
        native = native or FakeNative()
        sys.modules[native.__name__] = native
        return importlib.import_module("synthesize_cpp.audio"), native

    def test_owned_audio_is_exposed_as_read_only_zero_copy_float_view(self):
        module, native = self.load_audio_module()
        token_buffer = array.array("i", [1, 2, 3])

        result = module._synthesize_tokens(
            object(), token_buffer, language="en-US", voice=None, seed=42,
            speaking_rate=1.0, max_output_frames=None, diagnostics=None,
            should_cancel=None,
        )

        self.assertIs(native.calls[0][2], token_buffer)
        self.assertEqual(native.calls[0][1], 2)
        self.assertEqual(
            native.calls[0][3:8], ("en-US", None, 42, 1.0, 0)
        )
        self.assertIsNone(native.calls[0][8])
        self.assertIsNone(native.calls[0][9])
        self.assertIsNone(native.calls[0][10])
        self.assertTrue(result.audio.samples.readonly)
        self.assertEqual(result.audio.samples.format, "f")
        self.assertEqual(result.audio.samples.tolist(), [0.25, -0.5, 0.75, -1.0])
        self.assertEqual(result.audio.frame_count, 2)
        self.assertEqual(result.audio.sample_rate, 24000)
        self.assertEqual(result.audio.channel_count, 2)
        self.assertEqual(result.actual_seed, 42)
        self.assertEqual(result.resolved_language_tag, "en-US")
        self.assertIsNone(result.resolved_voice_id)
        self.assertTrue(result.flags & module.SynthesisResultFlags.SEED_USED)

    def test_sequence_conversion_random_seed_and_argument_validation(self):
        module, native = self.load_audio_module()
        module._synthesize_tokens(
            object(), [1, 2], language=None, voice="alice", seed=None,
            speaking_rate=0.8, max_output_frames=100, diagnostics=None,
            should_cancel=None,
        )
        arguments = native.calls[0]
        self.assertIsInstance(arguments[2], array.array)
        self.assertEqual(arguments[2].typecode, "i")
        self.assertEqual(arguments[4], "alice")
        self.assertEqual(arguments[5], 2**64 - 1)

        invalid = (
            ({"seed": -1}, ValueError),
            ({"seed": 2**64}, OverflowError),
            ({"speaking_rate": math.inf}, ValueError),
            ({"speaking_rate": 0.0}, ValueError),
            ({"max_output_frames": -1}, ValueError),
        )
        defaults = {
            "language": None, "voice": None, "seed": 0,
            "speaking_rate": 1.0, "max_output_frames": None,
            "diagnostics": None, "should_cancel": None,
        }
        for update, error in invalid:
            with self.subTest(update=update), self.assertRaises(error):
                module._synthesize_tokens(object(), [1], **(defaults | update))

    def test_native_synthesis_error_is_an_adapter_error(self):
        module, native = self.load_audio_module()

        def fail(*arguments):
            raise RuntimeError("output limit (15)")

        native.synthesize = fail
        with self.assertRaisesRegex(module.SynthesizeError, "synthesize.*output limit"):
            module._synthesize_tokens(
                object(), [1], language=None, voice=None, seed=0,
                speaking_rate=1.0, max_output_frames=None, diagnostics=None,
                should_cancel=None,
            )

    def test_text_phonemes_diagnostics_and_cancellation_use_one_native_seam(self):
        module, native = self.load_audio_module()
        captured = []

        result = module._synthesize_text(
            object(), "Hello, 世界", language="en", voice=None, seed=7,
            speaking_rate=1.0, max_output_frames=None,
            diagnostics=captured.append, should_cancel=lambda: False,
        )
        self.assertEqual(result.audio.frame_count, 2)
        arguments = native.calls[0]
        self.assertEqual(arguments[1], 0)
        self.assertEqual(arguments[2], "Hello, 世界")
        self.assertTrue(callable(arguments[8]))
        self.assertTrue(callable(arguments[9]))
        arguments[8]({
            "level": 1,
            "status": 13,
            "code": "frontend.failed",
            "message": "frontend unavailable",
        })
        self.assertEqual(captured[0].code, "frontend.failed")
        self.assertEqual(captured[0].status, 13)

        module._synthesize_phonemes(
            object(), "həˈloʊ", language="en", voice=None, seed=0,
            speaking_rate=1.0, max_output_frames=None,
            diagnostics=None, should_cancel=None,
        )
        self.assertEqual(native.calls[1][1], 1)
        self.assertEqual(native.calls[1][2], "həˈloʊ")

    def test_callback_argument_validation(self):
        module, _ = self.load_audio_module()
        defaults = {
            "language": None,
            "voice": None,
            "seed": 0,
            "speaking_rate": 1.0,
            "max_output_frames": None,
            "diagnostics": None,
            "should_cancel": None,
        }
        with self.assertRaisesRegex(TypeError, "diagnostics"):
            module._synthesize_tokens(
                object(), [1], **(defaults | {"diagnostics": 1})
            )
        with self.assertRaisesRegex(TypeError, "should_cancel"):
            module._synthesize_tokens(
                object(), [1], **(defaults | {"should_cancel": False})
            )
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            module._synthesize_text(object(), "", **defaults)
        with self.assertRaisesRegex(TypeError, "text must be str"):
            module._synthesize_text(object(), b"text", **defaults)


if __name__ == "__main__":
    unittest.main()
