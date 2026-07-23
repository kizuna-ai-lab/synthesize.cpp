import ctypes
import importlib
import sys
import tempfile
import tomllib
import types
import unittest
from pathlib import Path
from unittest import mock


REPO = Path(__file__).resolve().parents[2]


class PythonProviderPackagingTests(unittest.TestCase):
    def tearDown(self):
        for name in tuple(sys.modules):
            if name == "synthesize_cpp_native" or name.startswith(
                "synthesize_cpp_native."
            ):
                sys.modules.pop(name, None)
            if name == "synthesize_cpp_native_cu13" or name.startswith(
                "synthesize_cpp_native_cu13."
            ):
                sys.modules.pop(name, None)

    @staticmethod
    def _metadata(path):
        with path.open("rb") as source:
            return tomllib.load(source)

    def test_provider_projects_declare_the_confirmed_distribution_contract(self):
        default = self._metadata(REPO / "pyproject.toml")
        cu13 = self._metadata(
            REPO / "bindings/python-native-cu13/pyproject.toml"
        )

        self.assertEqual(default["project"]["name"], "synthesize-cpp-native")
        self.assertEqual(
            default["project"]["entry-points"]["synthesize_cpp.native"],
            {"default": "synthesize_cpp_native:descriptor"},
        )
        self.assertEqual(
            cu13["project"]["name"], "synthesize-cpp-native-cu13"
        )
        self.assertEqual(
            cu13["project"]["entry-points"]["synthesize_cpp.native"],
            {"cu13": "synthesize_cpp_native_cu13:descriptor"},
        )
        self.assertEqual(
            cu13["project"]["dependencies"],
            [
                "nvidia-cuda-runtime==13.3.29",
                "nvidia-cublas==13.6.0.2",
            ],
        )
        for metadata in (default, cu13):
            build = metadata["build-system"]
            self.assertEqual(build["build-backend"], "scikit_build_core.build")
            self.assertEqual(
                build["requires"],
                ["scikit-build-core==1.0.3"],
            )
            wheel = metadata["tool"]["scikit-build"]
            self.assertEqual(wheel["ninja"]["version"], "==1.13.0")
            self.assertEqual(wheel["wheel"]["py-api"], "py3")
            self.assertEqual(wheel["install"]["components"], ["wheel"])
            self.assertIn(
                "-DSYNTH_GGML_BACKEND_DL=ON",
                wheel["cmake"]["args"],
            )

        default_build_dir = default["tool"]["scikit-build"]["build-dir"]
        cu13_build_dir = cu13["tool"]["scikit-build"]["build-dir"]
        self.assertEqual(
            default_build_dir,
            "build/python-provider/default/{wheel_tag}",
        )
        self.assertEqual(
            cu13_build_dir,
            "../../build/python-provider/cu13/{wheel_tag}",
        )
        for build_dir in (default_build_dir, cu13_build_dir):
            self.assertNotIn("release-", build_dir)

    def test_provider_wheels_carry_complete_adapted_code_notice(self):
        notices = (
            REPO / "THIRD_PARTY_NOTICES.md",
            REPO / "bindings/python-native-cu13/THIRD_PARTY_NOTICES.md",
        )
        for notice in notices:
            contents = notice.read_text(encoding="utf-8")
            self.assertIn("transcribe.cpp", contents)
            self.assertIn("Permission is hereby granted", contents)
            self.assertIn("THE SOFTWARE IS PROVIDED \"AS IS\"", contents)

    def test_default_descriptor_is_a_complete_provider_identity(self):
        package_root = REPO / "bindings/python-native"
        with mock.patch.object(sys, "path", [str(package_root), *sys.path]):
            provider = importlib.import_module("synthesize_cpp_native")
        contract = types.ModuleType("synthesize_cpp_native._contract")
        contract.VERSION = "0.1.0"
        contract.PUBLIC_HEADER_HASH = "1" * 64
        contract.PROVIDER_ID = "default"
        contract.DISTRIBUTION = "synthesize-cpp-native"
        contract.BACKENDS = ("cpu",)
        sys.modules[contract.__name__] = contract

        descriptor = provider.descriptor()

        self.assertEqual(descriptor["provider_id"], "default")
        self.assertEqual(descriptor["distribution"], "synthesize-cpp-native")
        self.assertEqual(descriptor["base_release"], "0.1.0")
        self.assertEqual(descriptor["header_hash"], "1" * 64)
        self.assertEqual(descriptor["backends"], ["cpu"])
        self.assertEqual(
            Path(descriptor["artifact_dir"]),
            package_root / "synthesize_cpp_native/_native",
        )
        self.assertNotIn("prepare", descriptor)

    @unittest.skipIf(sys.platform == "win32", "Linux provider layout test")
    def test_cu13_prepare_preloads_pinned_runtime_libraries_in_order(self):
        package_root = REPO / "bindings/python-native-cu13"
        with mock.patch.object(sys, "path", [str(package_root), *sys.path]):
            provider = importlib.import_module("synthesize_cpp_native_cu13")

        with tempfile.TemporaryDirectory() as directory:
            nvidia_root = Path(directory)
            relative_libraries = (
                "cu13/lib/libcudart.so.13",
                "cu13/lib/libcublasLt.so.13",
                "cu13/lib/libcublas.so.13",
            )
            for relative in relative_libraries:
                path = nvidia_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            nvidia = types.ModuleType("nvidia")
            nvidia.__path__ = [str(nvidia_root)]
            loaded = []

            def record(path, mode):
                loaded.append((Path(path), mode))
                return object()

            with mock.patch.dict(sys.modules, {"nvidia": nvidia}), mock.patch.object(
                ctypes, "CDLL", side_effect=record
            ):
                provider.prepare()

        self.assertEqual(
            loaded,
            [
                (nvidia_root / relative, ctypes.RTLD_GLOBAL)
                for relative in relative_libraries
            ],
        )

    def test_cu13_descriptor_requires_preparation(self):
        package_root = REPO / "bindings/python-native-cu13"
        with mock.patch.object(sys, "path", [str(package_root), *sys.path]):
            provider = importlib.import_module("synthesize_cpp_native_cu13")
        contract = types.ModuleType("synthesize_cpp_native_cu13._contract")
        contract.VERSION = "0.1.0"
        contract.PUBLIC_HEADER_HASH = "2" * 64
        contract.PROVIDER_ID = "cu13"
        contract.DISTRIBUTION = "synthesize-cpp-native-cu13"
        contract.BACKENDS = ("cuda", "cpu")
        sys.modules[contract.__name__] = contract

        descriptor = provider.descriptor()

        self.assertEqual(descriptor["provider_id"], "cu13")
        self.assertEqual(
            descriptor["distribution"], "synthesize-cpp-native-cu13"
        )
        self.assertEqual(descriptor["backends"], ["cuda", "cpu"])
        self.assertIs(descriptor["prepare"], provider.prepare)

    @unittest.skipIf(sys.platform == "win32", "Linux provider layout test")
    def test_cu13_prepare_rejects_missing_runtime_package_or_library(self):
        package_root = REPO / "bindings/python-native-cu13"
        with mock.patch.object(sys, "path", [str(package_root), *sys.path]):
            provider = importlib.import_module("synthesize_cpp_native_cu13")

        with mock.patch.dict(sys.modules, {"nvidia": None}):
            with self.assertRaisesRegex(RuntimeError, "NVIDIA runtime packages"):
                provider.prepare()

        with tempfile.TemporaryDirectory() as directory:
            nvidia = types.ModuleType("nvidia")
            nvidia.__path__ = [directory]
            with mock.patch.dict(sys.modules, {"nvidia": nvidia}):
                with self.assertRaisesRegex(RuntimeError, "libcudart.so.13"):
                    provider.prepare()

    @unittest.skipIf(sys.platform == "win32", "Linux provider layout test")
    def test_cu13_prepare_maps_loader_failure_to_a_hard_error(self):
        package_root = REPO / "bindings/python-native-cu13"
        with mock.patch.object(sys, "path", [str(package_root), *sys.path]):
            provider = importlib.import_module("synthesize_cpp_native_cu13")

        with tempfile.TemporaryDirectory() as directory:
            nvidia_root = Path(directory)
            for relative in (
                "cu13/lib/libcudart.so.13",
                "cu13/lib/libcublasLt.so.13",
                "cu13/lib/libcublas.so.13",
            ):
                path = nvidia_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            nvidia = types.ModuleType("nvidia")
            nvidia.__path__ = [str(nvidia_root)]
            with mock.patch.dict(sys.modules, {"nvidia": nvidia}), mock.patch.object(
                ctypes, "CDLL", side_effect=OSError("loader rejected library")
            ):
                with self.assertRaisesRegex(RuntimeError, "could not preload"):
                    provider.prepare()


if __name__ == "__main__":
    unittest.main()
