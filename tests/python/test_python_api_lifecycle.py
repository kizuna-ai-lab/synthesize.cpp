import importlib
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
        self.next_handle = 0

    def model_load(self, path, backend, device_index, diagnostics):
        self.calls.append(
            ("model_load", path, backend, device_index, diagnostics)
        )
        self.next_handle += 1
        return ("model", self.next_handle)

    def model_close(self, handle):
        self.calls.append(("model_close", handle))

    def model_metadata(self, handle):
        self.calls.append(("model_metadata", handle))
        return {
            "device": {
                "index": None,
                "name": "CPU",
                "description": "CPU",
                "kind": "cpu",
                "device_id": None,
                "memory_total": 1,
                "memory_free": 1,
                "device_type": 0,
                "flags": 1,
            },
            "capabilities": {
                "input_flags": 4,
                "capability_flags": 0,
                "output_sample_rate": 22050,
                "output_channel_count": 1,
                "min_speaking_rate": 1.0,
                "max_speaking_rate": 1.0,
                "max_input_tokens": 512,
                "max_output_frames": 1000,
            },
            "preset_voices": (),
            "languages": (),
            "voice_profile_capabilities": {
                "source_flags": 0,
                "reference_transcript": 0,
                "reference_language": 0,
                "description_language": 0,
                "max_reference_count": 0,
                "profile_schema": None,
                "profile_schema_version": 0,
                "profile_compatibility_id": bytes(32),
                "reference_target_sample_rate": 0,
                "reference_target_channel_count": 0,
                "min_reference_frames_per_clip": 0,
                "max_reference_frames_per_clip": 0,
                "max_reference_total_frames": 0,
            },
        }

    def context_create(self, model_handle):
        self.calls.append(("context_create", model_handle))
        self.next_handle += 1
        return ("context", self.next_handle)

    def context_close(self, handle):
        self.calls.append(("context_close", handle))


class PythonApiLifecycleTests(unittest.TestCase):
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

    def test_model_and_context_close_exactly_once(self):
        module, native = self.load_models_module()
        path = Path("voice.gguf")

        model = module.Model(
            path,
            backend=module.Backend.CPU,
            device_index=2,
        )
        context = model.create_context()
        self.assertFalse(model.closed)
        self.assertFalse(context.closed)
        self.assertIs(context.model, model)

        context.close()
        context.close()
        model.close()
        model.close()

        self.assertTrue(context.closed)
        self.assertTrue(model.closed)
        self.assertEqual(
            native.calls,
            [
                ("model_load", "voice.gguf", 1, 2, None),
                ("model_metadata", ("model", 1)),
                ("context_create", ("model", 1)),
                ("context_close", ("context", 2)),
                ("model_close", ("model", 1)),
            ],
        )

    def test_context_retains_model_and_model_close_invalidates_new_work(self):
        module, native = self.load_models_module()
        model = module.Model("voice.gguf")
        context = model.create_context()

        model.close()
        self.assertIs(context.model, model)
        with self.assertRaisesRegex(module.SynthesizeError, "Model is closed"):
            model.create_context()
        context.close()

        self.assertEqual(
            [call[0] for call in native.calls],
            [
                "model_load",
                "model_metadata",
                "context_create",
                "model_close",
                "context_close",
            ],
        )

    def test_both_objects_are_context_managers(self):
        module, native = self.load_models_module()
        with module.Model("voice.gguf") as model:
            with model.create_context() as context:
                self.assertFalse(context.closed)
            self.assertTrue(context.closed)
        self.assertTrue(model.closed)
        self.assertEqual(
            [call[0] for call in native.calls],
            [
                "model_load",
                "model_metadata",
                "context_create",
                "context_close",
                "model_close",
            ],
        )

    def test_argument_and_native_failures_are_adapter_errors(self):
        module, native = self.load_models_module()
        with self.assertRaises(TypeError):
            module.Model("voice.gguf", backend=1)
        with self.assertRaises(ValueError):
            module.Model("voice.gguf", device_index=-2)
        with self.assertRaises(OverflowError):
            module.Model("voice.gguf", device_index=2**31)

        def fail(*arguments):
            raise RuntimeError("file not found (3)")

        native.model_load = fail
        with self.assertRaisesRegex(module.SynthesizeError, "load.*file not found"):
            module.Model("missing.gguf")

    def test_model_load_diagnostic_callback_is_adapted(self):
        module, native = self.load_models_module()
        captured = []
        model = module.Model("voice.gguf", diagnostics=captured.append)
        native_callback = native.calls[0][4]
        self.assertTrue(callable(native_callback))
        native_callback({
            "level": 1,
            "status": 17,
            "code": "backend.unavailable",
            "message": "no matching backend",
        })
        self.assertEqual(captured[0].code, "backend.unavailable")
        self.assertEqual(captured[0].status, 17)
        model.close()

        with self.assertRaisesRegex(TypeError, "diagnostics"):
            module.Model("voice.gguf", diagnostics=object())


if __name__ == "__main__":
    unittest.main()
