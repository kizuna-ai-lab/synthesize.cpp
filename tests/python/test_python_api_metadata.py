import importlib
import sys
import types
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PACKAGE_SOURCE = REPO / "bindings/python/src/synthesize_cpp"


def sample_metadata():
    return {
        "device": {
            "index": None,
            "name": "CUDA0",
            "description": "Discrete GPU",
            "kind": "cuda",
            "device_id": "GPU-1",
            "memory_total": 24_000,
            "memory_free": 20_000,
            "device_type": 1,
            "flags": 1,
        },
        "capabilities": {
            "input_flags": 1 | 4,
            "capability_flags": 1 | 2 | 0x80,
            "output_sample_rate": 24_000,
            "output_channel_count": 1,
            "min_speaking_rate": 0.5,
            "max_speaking_rate": 2.0,
            "max_input_tokens": 1024,
            "max_output_frames": 48_000,
        },
        "preset_voices": (
            {
                "id": "alice",
                "display_name": "Alice",
                "flags": 1 | 0x20,
            },
            {"id": "bob", "display_name": None, "flags": 0},
        ),
        "languages": (
            {"tag": "en-US", "flags": 1 | 2},
            {"tag": "ja", "flags": 0},
        ),
        "voice_profile_capabilities": {
            "source_flags": 1 | 8,
            "reference_transcript": 2,
            "reference_language": 1,
            "description_language": 0,
            "max_reference_count": 4,
            "profile_schema": "speaker-embedding-v1",
            "profile_schema_version": 1,
            "profile_compatibility_id": bytes(range(32)),
            "reference_target_sample_rate": 16000,
            "reference_target_channel_count": 1,
            "min_reference_frames_per_clip": 8000,
            "max_reference_frames_per_clip": 160000,
            "max_reference_total_frames": 320000,
        },
    }


class FakeNative(types.ModuleType):
    def __init__(self):
        super().__init__("synthesize_cpp._native")
        self.calls = []

    def model_load(self, path, backend, device_index, diagnostics):
        self.calls.append(("model_load", path, backend, device_index))
        return object()

    def model_metadata(self, handle):
        self.calls.append(("model_metadata", handle))
        return sample_metadata()

    def model_close(self, handle):
        self.calls.append(("model_close", handle))


class PythonApiMetadataTests(unittest.TestCase):
    def tearDown(self):
        for name in tuple(sys.modules):
            if name == "synthesize_cpp" or name.startswith("synthesize_cpp."):
                sys.modules.pop(name, None)

    def load_models_module(self, native=None):
        package = types.ModuleType("synthesize_cpp")
        package.__path__ = [str(PACKAGE_SOURCE)]
        package.__package__ = "synthesize_cpp"
        sys.modules["synthesize_cpp"] = package
        native = native or FakeNative()
        sys.modules[native.__name__] = native
        return importlib.import_module("synthesize_cpp.models"), native

    def test_model_copies_typed_immutable_metadata(self):
        module, native = self.load_models_module()
        devices = importlib.import_module("synthesize_cpp.devices")
        metadata = importlib.import_module("synthesize_cpp.metadata")
        model = module.Model("voice.gguf", backend=module.Backend.CUDA)

        self.assertIsNone(model.device.index)
        self.assertEqual(model.device.kind, "cuda")
        self.assertEqual(model.device.memory_total, 24_000)
        self.assertEqual(model.device.device_type, devices.DeviceType.GPU)
        self.assertEqual(
            model.capabilities.input_flags,
            metadata.InputFlags.TEXT_UTF8 | metadata.InputFlags.TOKEN_IDS,
        )
        self.assertTrue(
            model.capabilities.capability_flags
            & metadata.ModelCapabilityFlags.STOCHASTIC
        )
        self.assertEqual(
            int(model.capabilities.capability_flags) & 0x80, 0x80
        )
        self.assertEqual(model.capabilities.output_sample_rate, 24_000)
        self.assertEqual(model.preset_voices[0].id, "alice")
        self.assertEqual(model.preset_voices[0].display_name, "Alice")
        self.assertTrue(
            model.preset_voices[0].flags & metadata.PresetVoiceFlags.DEFAULT
        )
        self.assertIsNone(model.preset_voices[1].display_name)
        self.assertEqual(model.languages[0].tag, "en-US")
        self.assertEqual(
            model.languages[0].flags,
            metadata.LanguageFlags.DEFAULT
            | metadata.LanguageFlags.REGIONAL_FALLBACK,
        )
        self.assertIsInstance(model.preset_voices, tuple)
        self.assertIsInstance(model.languages, tuple)
        self.assertEqual(model.voice_profile_capabilities.max_reference_count, 4)
        self.assertEqual(
            model.voice_profile_capabilities.profile_compatibility_id,
            bytes(range(32)),
        )
        self.assertEqual(
            int(model.voice_profile_capabilities.source_flags), 1 | 8
        )
        with self.assertRaises(FrozenInstanceError):
            model.capabilities.output_sample_rate = 1

        model.close()
        self.assertEqual(model.languages[1].tag, "ja")

    def test_metadata_failure_closes_just_loaded_native_model(self):
        module, native = self.load_models_module()
        handle = object()
        native.model_load = lambda *arguments: handle

        def fail_metadata(value):
            raise RuntimeError("bad metadata (5)")

        native.model_metadata = fail_metadata
        with self.assertRaisesRegex(module.SynthesizeError, "metadata.*bad metadata"):
            module.Model("voice.gguf")
        self.assertEqual(native.calls, [("model_close", handle)])


if __name__ == "__main__":
    unittest.main()
