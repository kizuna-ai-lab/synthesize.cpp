import importlib
import sys
import types
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PACKAGE_SOURCE = REPO / "bindings/python/src/synthesize_cpp"


class PythonApiDiagnosticTests(unittest.TestCase):
    def tearDown(self):
        for name in tuple(sys.modules):
            if name == "synthesize_cpp" or name.startswith("synthesize_cpp."):
                sys.modules.pop(name, None)

    def load_module(self):
        package = types.ModuleType("synthesize_cpp")
        package.__path__ = [str(PACKAGE_SOURCE)]
        package.__package__ = "synthesize_cpp"
        sys.modules["synthesize_cpp"] = package
        return importlib.import_module("synthesize_cpp.diagnostics")

    def test_native_diagnostic_becomes_an_immutable_public_value(self):
        module = self.load_module()
        captured = []
        callback = module._diagnostic_adapter(captured.append)
        callback({
            "level": 0,
            "status": 12,
            "code": "resource.optional_missing",
            "message": "optional dictionary is missing",
        })

        diagnostic = captured[0]
        self.assertEqual(diagnostic.level, module.DiagnosticLevel.WARNING)
        self.assertEqual(diagnostic.status, module.Status.MISSING_RESOURCE)
        self.assertEqual(diagnostic.code, "resource.optional_missing")
        self.assertEqual(diagnostic.message, "optional dictionary is missing")
        with self.assertRaises(AttributeError):
            diagnostic.code = "changed"

    def test_unknown_future_values_are_preserved_as_integers(self):
        module = self.load_module()
        captured = []
        module._diagnostic_adapter(captured.append)({
            "level": 99,
            "status": 999,
            "code": "future.value",
            "message": "future",
        })
        self.assertEqual(captured[0].level, 99)
        self.assertEqual(captured[0].status, 999)

    def test_none_disables_sink_and_non_callable_is_rejected(self):
        module = self.load_module()
        self.assertIsNone(module._diagnostic_adapter(None))
        with self.assertRaisesRegex(TypeError, "diagnostics"):
            module._diagnostic_adapter(object())


if __name__ == "__main__":
    unittest.main()
