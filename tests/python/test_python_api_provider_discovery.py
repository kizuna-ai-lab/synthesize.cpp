import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PYTHON_SOURCE = REPO / "bindings/python/src"
GOOD_HASH = "a" * 64


class FakeEntryPoint:
    def __init__(self, name, value=None, error=None):
        self.name = name
        self._value = value
        self._error = error

    def load(self):
        if self._error is not None:
            raise self._error
        return self._value


class PythonApiProviderDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(PYTHON_SOURCE))

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(PYTHON_SOURCE))

    def setUp(self):
        for name in tuple(sys.modules):
            if name == "synthesize_cpp" or name.startswith("synthesize_cpp."):
                sys.modules.pop(name, None)
        package = types.ModuleType("synthesize_cpp")
        package.__path__ = [str(PYTHON_SOURCE / "synthesize_cpp")]
        package.__package__ = "synthesize_cpp"
        sys.modules["synthesize_cpp"] = package
        self.providers = importlib.import_module("synthesize_cpp._providers")

    def descriptor(
        self,
        root,
        provider_id="default",
        distribution="synthesize-cpp-native",
        backends=("cpu",),
        base_release="0.1.0",
        header_hash=GOOD_HASH,
        prepare=None,
    ):
        artifact_dir = Path(root) / provider_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / self.providers.library_filename()).touch()
        value = {
            "provider_id": provider_id,
            "distribution": distribution,
            "artifact_dir": str(artifact_dir.resolve()),
            "base_release": base_release,
            "header_hash": header_hash,
            "backends": list(backends),
        }
        if prepare is not None:
            value["prepare"] = prepare
        return value

    def discover(self, entry_points):
        return self.providers.discover_providers(
            entry_points,
            expected_base_release="0.1.0",
            expected_header_hash=GOOD_HASH,
        )

    def test_descriptor_is_complete_and_resolves_platform_library(self):
        with tempfile.TemporaryDirectory() as directory:
            [provider] = self.discover(
                [
                    FakeEntryPoint(
                        "default",
                        lambda: self.descriptor(directory),
                    )
                ]
            )

        self.assertEqual(provider.provider_id, "default")
        self.assertEqual(provider.distribution, "synthesize-cpp-native")
        self.assertEqual(provider.backends, ("cpu",))
        self.assertEqual(provider.rank, 1)
        self.assertEqual(
            provider.library_path.name, self.providers.library_filename()
        )

    def test_malformed_or_broken_discovered_provider_is_a_hard_error(self):
        with tempfile.TemporaryDirectory() as directory:
            malformed = self.descriptor(directory)
            malformed.pop("header_hash")
            with self.assertRaisesRegex(
                self.providers.ProviderError, "header_hash"
            ):
                self.discover([FakeEntryPoint("default", lambda: malformed)])

            with self.assertRaisesRegex(
                self.providers.ProviderError, "broken.*import exploded"
            ):
                self.discover(
                    [
                        FakeEntryPoint(
                            "broken", error=RuntimeError("import exploded")
                        )
                    ]
                )

    def test_every_provider_is_validated_before_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            cuda = self.descriptor(
                directory,
                provider_id="cu13",
                distribution="synthesize-cpp-native-cu13",
                backends=("cuda", "cpu"),
            )
            stale = self.descriptor(
                directory,
                provider_id="default",
                header_hash="b" * 64,
            )
            with self.assertRaisesRegex(
                self.providers.ProviderError, "public-header hash"
            ):
                self.discover(
                    [
                        FakeEntryPoint("cu13", lambda: cuda),
                        FakeEntryPoint("default", lambda: stale),
                    ]
                )

    def test_backend_rank_and_auto_selection_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            cpu = self.descriptor(directory)
            vulkan = self.descriptor(
                directory,
                provider_id="vulkan",
                distribution="synthesize-cpp-native-vulkan",
                backends=("vulkan", "cpu"),
            )
            cuda = self.descriptor(
                directory,
                provider_id="cu13",
                distribution="synthesize-cpp-native-cu13",
                backends=("cuda", "cpu"),
            )
            discovered = self.discover(
                [
                    FakeEntryPoint("default", lambda: cpu),
                    FakeEntryPoint("vulkan", lambda: vulkan),
                    FakeEntryPoint("cu13", lambda: cuda),
                ]
            )

        selected = self.providers.select_provider(discovered)
        self.assertEqual(selected.provider_id, "cu13")
        self.assertEqual(selected.rank, 5)

    def test_equal_highest_rank_is_an_ambiguity_error(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.descriptor(
                directory,
                provider_id="cu13",
                distribution="synthesize-cpp-native-cu13",
                backends=("cuda", "cpu"),
            )
            second = self.descriptor(
                directory,
                provider_id="cuda-private",
                distribution="private-cuda-provider",
                backends=("cuda", "cpu"),
            )
            discovered = self.discover(
                [
                    FakeEntryPoint("cu13", lambda: first),
                    FakeEntryPoint("cuda-private", lambda: second),
                ]
            )
            with self.assertRaisesRegex(
                self.providers.ProviderError, "ambiguous.*cu13.*cuda-private"
            ):
                self.providers.select_provider(discovered)

    def test_override_matches_only_exact_stable_provider_id(self):
        with tempfile.TemporaryDirectory() as directory:
            cpu = self.descriptor(directory)
            cuda = self.descriptor(
                directory,
                provider_id="cu13",
                distribution="synthesize-cpp-native-cu13",
                backends=("cuda", "cpu"),
            )
            discovered = self.discover(
                [
                    FakeEntryPoint("default", lambda: cpu),
                    FakeEntryPoint("cu13", lambda: cuda),
                ]
            )

        self.assertEqual(
            self.providers.select_provider(discovered, override="default").provider_id,
            "default",
        )
        for invalid in ("cuda", "synthesize-cpp-native-cu13", "/tmp/core.so"):
            with self.assertRaisesRegex(
                self.providers.ProviderError, "not installed"
            ):
                self.providers.select_provider(discovered, override=invalid)

    def test_missing_provider_is_an_actionable_error(self):
        with self.assertRaisesRegex(
            self.providers.ProviderError, "no native Provider is installed"
        ):
            self.providers.select_provider([])

    def test_registry_selects_once_and_prepares_only_the_selected_provider(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            cpu = self.descriptor(
                directory, prepare=lambda: calls.append("cpu")
            )
            cuda = self.descriptor(
                directory,
                provider_id="cu13",
                distribution="synthesize-cpp-native-cu13",
                backends=("cuda", "cpu"),
                prepare=lambda: calls.append("cu13"),
            )
            entry_points = [
                FakeEntryPoint("default", lambda: cpu),
                FakeEntryPoint("cu13", lambda: cuda),
            ]
            registry = self.providers.ProviderRegistry(
                lambda: entry_points,
                expected_base_release="0.1.0",
                expected_header_hash=GOOD_HASH,
                environ={},
            )

            first = registry.select()
            second = registry.select()
            self.assertIs(first, second)
            self.assertEqual(calls, [])
            registry.prepare()
            registry.prepare()

        self.assertEqual(calls, ["cu13"])

    def test_registry_environment_override_is_frozen_on_first_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            cpu = self.descriptor(directory)
            cuda = self.descriptor(
                directory,
                provider_id="cu13",
                distribution="synthesize-cpp-native-cu13",
                backends=("cuda", "cpu"),
            )
            environ = {"SYNTHESIZE_NATIVE_PROVIDER": "default"}
            registry = self.providers.ProviderRegistry(
                lambda: [
                    FakeEntryPoint("default", lambda: cpu),
                    FakeEntryPoint("cu13", lambda: cuda),
                ],
                expected_base_release="0.1.0",
                expected_header_hash=GOOD_HASH,
                environ=environ,
            )
            self.assertEqual(registry.select().provider_id, "default")
            environ["SYNTHESIZE_NATIVE_PROVIDER"] = "cu13"
            self.assertEqual(registry.select().provider_id, "default")

    def test_prepare_failure_is_final_and_names_selected_provider(self):
        def fail():
            raise RuntimeError("runtime missing")

        with tempfile.TemporaryDirectory() as directory:
            cuda = self.descriptor(
                directory,
                provider_id="cu13",
                distribution="synthesize-cpp-native-cu13",
                backends=("cuda", "cpu"),
                prepare=fail,
            )
            registry = self.providers.ProviderRegistry(
                lambda: [FakeEntryPoint("cu13", lambda: cuda)],
                expected_base_release="0.1.0",
                expected_header_hash=GOOD_HASH,
                environ={},
            )
            with self.assertRaisesRegex(
                self.providers.ProviderError, "cu13.*prepare.*runtime missing"
            ):
                registry.prepare()


if __name__ == "__main__":
    unittest.main()
