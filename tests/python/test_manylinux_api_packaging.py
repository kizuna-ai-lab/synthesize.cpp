import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/ci/build_manylinux_api.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_manylinux_api", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ManylinuxApiPackagingTests(unittest.TestCase):
    def test_container_is_digest_locked_read_only_and_builds_cp311_abi3(self):
        builder = load_builder()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("source", "work", "build"):
                (root / name).mkdir()
            command = builder.container_command(
                arch="aarch64",
                source_dir=root / "source",
                work_dir=root / "work",
                build_dir=root / "build",
                jobs=3,
            )

        rendered = "\n".join(command)
        self.assertEqual(
            builder.MANYLINUX_IMAGES["aarch64"],
            "quay.io/pypa/manylinux_2_28_aarch64@"
            "sha256:162c81dfd3efc710732a571717d3c916a6945ebf279e879ddee3243af96fe46f",
        )
        self.assertIn(
            f"type=bind,src={(root / 'source').resolve()},dst=/project,readonly",
            command,
        )
        self.assertIn("CMAKE_BUILD_PARALLEL_LEVEL=3", command)
        self.assertIn("/project/bindings/python", rendered)
        self.assertIn("/opt/python/cp311-cp311/bin/python", rendered)
        self.assertIn("auditwheel repair --plat manylinux_2_28_aarch64", rendered)
        self.assertNotIn("/opt/cuda", rendered)

    def test_repaired_wheel_requires_adapter_contract_and_no_provider_payload(self):
        builder = load_builder()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / (
                "synthesize_cpp-0.1.0-cp311-abi3-"
                "manylinux_2_28_aarch64.whl"
            )
            self.write_wheel(wheel, "manylinux_2_28_aarch64")
            contract = builder.validate_repaired_wheel(
                wheel,
                arch="aarch64",
                expected_version="0.1.0",
                expected_header_hash="1" * 64,
            )
            self.assertEqual(contract["version"], "0.1.0")

            leaked = root / (
                "leaked-0.1.0-cp311-abi3-manylinux_2_28_aarch64.whl"
            )
            self.write_wheel(
                leaked,
                "manylinux_2_28_aarch64",
                extra={"synthesize_cpp/_native/libsynthesize.so": b""},
            )
            with self.assertRaisesRegex(ValueError, "Provider payload"):
                builder.validate_repaired_wheel(
                    leaked,
                    arch="aarch64",
                    expected_version="0.1.0",
                    expected_header_hash="1" * 64,
                )

    @staticmethod
    def write_wheel(path, platform_tag, extra=None):
        files = {
            "synthesize_cpp/__init__.py": b"",
            "synthesize_cpp/_build_contract.py": (
                'VERSION = "0.1.0"\nPUBLIC_HEADER_HASH = "' + "1" * 64 + '"\n'
            ).encode(),
            "synthesize_cpp/_native.abi3.so": b"",
            "synthesize_cpp/_providers.py": b"",
            "synthesize_cpp/_runtime.py": b"",
            "synthesize_cpp/audio.py": b"",
            "synthesize_cpp/devices.py": b"",
            "synthesize_cpp/diagnostics.py": b"",
            "synthesize_cpp/errors.py": b"",
            "synthesize_cpp/metadata.py": b"",
            "synthesize_cpp/models.py": b"",
            "synthesize_cpp/voice_profiles.py": b"",
            "synthesize_cpp-0.1.0.dist-info/WHEEL": (
                "Wheel-Version: 1.0\nRoot-Is-Purelib: false\n"
                f"Tag: cp311-abi3-{platform_tag}\n"
            ).encode(),
        }
        files.update(extra or {})
        with zipfile.ZipFile(path, "w") as archive:
            for name, data in files.items():
                archive.writestr(name, data)


if __name__ == "__main__":
    unittest.main()
