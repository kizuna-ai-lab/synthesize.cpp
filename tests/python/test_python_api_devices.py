import importlib
import sys
import types
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PACKAGE_SOURCE = REPO / "bindings/python/src/synthesize_cpp"


class PythonApiDeviceTests(unittest.TestCase):
    def tearDown(self):
        for name in tuple(sys.modules):
            if name == "synthesize_cpp" or name.startswith("synthesize_cpp."):
                sys.modules.pop(name, None)

    def load_devices_module(self, native):
        package = types.ModuleType("synthesize_cpp")
        package.__path__ = [str(PACKAGE_SOURCE)]
        package.__package__ = "synthesize_cpp"
        sys.modules["synthesize_cpp"] = package
        sys.modules["synthesize_cpp._native"] = native
        return importlib.import_module("synthesize_cpp.devices")

    def test_backend_devices_are_immutable_copied_value_objects(self):
        native = types.ModuleType("synthesize_cpp._native")
        native.backend_devices = lambda: (
            {
                "index": 0,
                "name": "CPU",
                "description": "Host CPU",
                "kind": "cpu",
                "device_id": None,
                "memory_total": 1024,
                "memory_free": 512,
                "device_type": 0,
                "flags": 1 | 4 | 0x80,
            },
            {
                "index": 1,
                "name": "opaque",
                "description": "No memory query",
                "kind": "unknown",
                "device_id": "future:1",
                "memory_total": 0,
                "memory_free": 0,
                "device_type": 3,
                "flags": 0,
            },
        )
        module = self.load_devices_module(native)

        result = module.backend_devices()

        self.assertIsInstance(result, tuple)
        self.assertEqual(result[0].index, 0)
        self.assertEqual(result[0].device_type, module.DeviceType.CPU)
        self.assertEqual(result[0].memory_total, 1024)
        self.assertEqual(result[0].memory_free, 512)
        self.assertTrue(result[0].flags & module.DeviceFlags.APPROXIMATE)
        self.assertEqual(int(result[0].flags) & 0x80, 0x80)
        self.assertIsNone(result[1].memory_total)
        self.assertIsNone(result[1].memory_free)
        self.assertEqual(result[1].device_id, "future:1")
        with self.assertRaises(FrozenInstanceError):
            result[0].name = "changed"

    def test_backend_available_accepts_only_public_backend_values(self):
        calls = []
        native = types.ModuleType("synthesize_cpp._native")
        native.backend_available = lambda value: calls.append(value) or value == 3
        module = self.load_devices_module(native)

        self.assertTrue(module.backend_available(module.Backend.CUDA))
        self.assertFalse(module.backend_available(module.Backend.CPU))
        self.assertEqual(calls, [3, 1])
        with self.assertRaises(TypeError):
            module.backend_available(3)

    def test_native_device_failure_uses_adapter_exception(self):
        native = types.ModuleType("synthesize_cpp._native")

        def fail():
            raise RuntimeError("invalid argument (1)")

        native.backend_devices = fail
        module = self.load_devices_module(native)
        with self.assertRaisesRegex(module.SynthesizeError, "enumerate.*invalid"):
            module.backend_devices()


if __name__ == "__main__":
    unittest.main()
