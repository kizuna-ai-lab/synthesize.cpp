import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PACKAGE_SOURCE = REPO / "bindings/python/src/synthesize_cpp"


class PythonApiRuntimeTests(unittest.TestCase):
    def tearDown(self):
        for name in tuple(sys.modules):
            if name == "synthesize_cpp" or name.startswith("synthesize_cpp."):
                sys.modules.pop(name, None)

    def install_source_package_shell(self):
        package = types.ModuleType("synthesize_cpp")
        package.__path__ = [str(PACKAGE_SOURCE)]
        package.__package__ = "synthesize_cpp"
        sys.modules["synthesize_cpp"] = package
        return importlib.import_module("synthesize_cpp._providers")

    def install_build_contract(self):
        contract = types.ModuleType("synthesize_cpp._build_contract")
        contract.VERSION = "0.1.0"
        contract.PUBLIC_HEADER_HASH = "a" * 64
        sys.modules[contract.__name__] = contract

    def test_import_prepares_selected_provider_before_native_load(self):
        providers = self.install_source_package_shell()
        self.install_build_contract()
        calls = []

        with tempfile.TemporaryDirectory() as directory:
            library_path = Path(directory).resolve() / "libsynthesize.so"
            library_path.touch()
            provider = types.SimpleNamespace(
                provider_id="default",
                library_path=library_path,
                backends=("cpu",),
            )

            class FakeRegistry:
                def __init__(self, **arguments):
                    calls.append(("registry", arguments))

                def prepare(self):
                    calls.append(("prepare",))
                    return provider

            native = types.ModuleType("synthesize_cpp._native")

            def load(path):
                calls.append(("load", path))
                return {
                    "path": path,
                    "abi_version": 1,
                    "version": (0, 1, 0),
                }

            native.load = load
            native.backend_available = lambda request: calls.append(
                ("backend_available", request)
            ) or request == 1
            sys.modules[native.__name__] = native
            providers.ProviderRegistry = FakeRegistry

            runtime = importlib.import_module("synthesize_cpp._runtime")

        self.assertEqual(calls[0][0], "registry")
        self.assertEqual(
            calls[0][1],
            {
                "expected_base_release": "0.1.0",
                "expected_header_hash": "a" * 64,
            },
        )
        self.assertEqual(calls[1], ("prepare",))
        self.assertEqual(calls[2], ("load", str(library_path)))
        self.assertEqual(calls[3], ("backend_available", 1))
        self.assertIs(runtime._provider, provider)
        self.assertEqual(runtime._native_info["abi_version"], 1)

    def test_native_load_failure_is_a_provider_error_naming_selection(self):
        providers = self.install_source_package_shell()
        self.install_build_contract()
        provider = types.SimpleNamespace(
            provider_id="cu13",
            library_path=Path("/native/libsynthesize.so"),
            backends=("cuda", "cpu"),
        )

        class FakeRegistry:
            def __init__(self, **arguments):
                pass

            def prepare(self):
                return provider

        native = types.ModuleType("synthesize_cpp._native")

        def fail_load(path):
            raise RuntimeError("missing synth_get_version")

        native.load = fail_load
        sys.modules[native.__name__] = native
        providers.ProviderRegistry = FakeRegistry

        with self.assertRaisesRegex(
            providers.ProviderError,
            "cu13.*load.*missing synth_get_version",
        ):
            importlib.import_module("synthesize_cpp._runtime")

    def test_import_rejects_an_advertised_backend_that_did_not_register(self):
        providers = self.install_source_package_shell()
        self.install_build_contract()
        provider = types.SimpleNamespace(
            provider_id="cu13",
            library_path=Path("/native/libsynthesize.so"),
            backends=("cuda", "cpu"),
        )

        class FakeRegistry:
            def __init__(self, **arguments):
                pass

            def prepare(self):
                return provider

        native = types.ModuleType("synthesize_cpp._native")
        native.load = lambda path: {
            "path": path,
            "abi_version": 1,
            "version": (0, 1, 0),
        }
        native.backend_available = lambda request: request == 1
        sys.modules[native.__name__] = native
        providers.ProviderRegistry = FakeRegistry

        with self.assertRaisesRegex(
            providers.ProviderError,
            "cu13.*advertised backend 'cuda'.*unavailable",
        ):
            importlib.import_module("synthesize_cpp._runtime")


if __name__ == "__main__":
    unittest.main()
