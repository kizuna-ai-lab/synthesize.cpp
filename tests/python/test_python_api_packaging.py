import tomllib
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


class PythonApiPackagingTests(unittest.TestCase):
    def test_api_distribution_and_abi3_build_contract(self):
        pyproject_path = REPO / "bindings/python/pyproject.toml"
        with pyproject_path.open("rb") as source:
            metadata = tomllib.load(source)

        self.assertEqual(metadata["project"]["name"], "synthesize-cpp")
        self.assertEqual(metadata["project"]["requires-python"], ">=3.11")
        self.assertEqual(
            metadata["project"]["dependencies"],
            ["synthesize-cpp-native==0.1.0.*"],
        )
        self.assertEqual(
            metadata["project"]["optional-dependencies"]["cu13"],
            ["synthesize-cpp-native-cu13==0.1.0.*"],
        )
        self.assertEqual(
            metadata["project"]["optional-dependencies"]["vulkan"],
            ["synthesize-cpp-native-vulkan==0.1.0.*"],
        )

        self.assertEqual(
            metadata["build-system"]["requires"],
            ["scikit-build-core==1.0.3"],
        )
        self.assertEqual(
            metadata["build-system"]["build-backend"],
            "scikit_build_core.build",
        )
        build = metadata["tool"]["scikit-build"]
        self.assertEqual(build["build-dir"], "../../build/python-api/{wheel_tag}")
        self.assertEqual(build["cmake"]["source-dir"], ".")
        self.assertEqual(build["wheel"]["py-api"], "cp311")
        self.assertEqual(build["wheel"]["packages"], ["src/synthesize_cpp"])
        self.assertEqual(build["install"]["components"], ["wheel"])

    def test_api_distribution_has_no_arbitrary_native_library_override(self):
        package = REPO / "bindings/python/src/synthesize_cpp"
        combined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(package.glob("*.py"))
        )
        self.assertNotIn("SYNTHESIZE_LIBRARY", combined)
        self.assertIn("SYNTHESIZE_NATIVE_PROVIDER", combined)

    def test_wheel_generates_adapter_contract_and_initializes_runtime(self):
        cmake = (REPO / "bindings/python/CMakeLists.txt").read_text(
            encoding="utf-8"
        )
        package_init = (
            REPO / "bindings/python/src/synthesize_cpp/__init__.py"
        ).read_text(encoding="utf-8")

        self.assertIn("file(SHA256", cmake)
        self.assertIn("_build_contract.py.in", cmake)
        self.assertIn("DESTINATION synthesize_cpp COMPONENT wheel", cmake)
        self.assertTrue(
            (
                REPO
                / "bindings/python/_build_contract.py.in"
            ).is_file()
        )
        self.assertIn("from . import _runtime as _runtime", package_init)


if __name__ == "__main__":
    unittest.main()
