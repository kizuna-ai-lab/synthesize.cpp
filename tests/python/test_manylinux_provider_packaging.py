import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/ci/build_manylinux_provider.py"


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_manylinux_provider", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ManylinuxProviderPackagingTests(unittest.TestCase):
    def test_release_inputs_are_digest_and_architecture_locked(self):
        builder = load_builder()

        self.assertEqual(
            builder.MANYLINUX_IMAGES,
            {
                "aarch64": (
                    "quay.io/pypa/manylinux_2_28_aarch64@"
                    "sha256:162c81dfd3efc710732a571717d3c916a6945ebf279e879ddee3243af96fe46f"
                ),
                "x86_64": (
                    "quay.io/pypa/manylinux_2_28_x86_64@"
                    "sha256:a61875a2f84cab7df8de222ff12cabc08ff86eb4ad402ac90ba7bdaed9600cca"
                ),
            },
        )
        self.assertEqual(builder.CUDA_ARCHITECTURES["aarch64"], "121a-real")
        self.assertEqual(
            builder.CUDA_ARCHITECTURES["x86_64"],
            "75-real;80-real;86-real;89-real;90-real;100-real;120a-real",
        )
        self.assertEqual(
            builder.EXPECTED_CUBINS["aarch64"], "sm_121a"
        )
        self.assertEqual(
            builder.EXPECTED_CUBINS["x86_64"],
            "sm_75,sm_80,sm_86,sm_89,sm_90,sm_100,sm_120a",
        )
        self.assertEqual(
            builder.CPU_MODULES["aarch64"],
            ("synthesize/backends/libggml-cpu.so",),
        )
        self.assertEqual(
            builder.CPU_MODULES["x86_64"],
            tuple(
                f"synthesize/backends/libggml-cpu-{variant}.so"
                for variant in (
                    "x64",
                    "sse42",
                    "sandybridge",
                    "ivybridge",
                    "piledriver",
                    "haswell",
                    "skylakex",
                    "cannonlake",
                    "cascadelake",
                    "icelake",
                    "cooperlake",
                    "zen4",
                    "alderlake",
                    "sapphirerapids",
                )
            ),
        )

    def test_cu13_container_command_uses_read_only_inputs_and_exact_targets(self):
        builder = load_builder()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            work = root / "work"
            build = root / "build"
            cuda = root / "cuda"
            for path in (source, work, build, cuda):
                path.mkdir()

            command = builder.container_command(
                provider="cu13",
                arch="aarch64",
                source_dir=source,
                work_dir=work,
                build_dir=build,
                cuda_root=cuda,
                jobs=7,
            )

        rendered = "\n".join(command)
        self.assertIn(builder.MANYLINUX_IMAGES["aarch64"], command)
        self.assertIn(
            f"type=bind,src={source.resolve()},dst=/project,readonly",
            command,
        )
        self.assertIn(
            f"type=bind,src={cuda.resolve()},dst=/opt/cuda-13.3,readonly",
            command,
        )
        self.assertIn("CMAKE_BUILD_PARALLEL_LEVEL=7", command)
        self.assertIn(
            "CMAKE_ARGS=-DCMAKE_CUDA_ARCHITECTURES=121a-real", command
        )
        self.assertFalse(
            any(value.startswith("PATH=") for value in command), command
        )
        self.assertIn(
            'export PATH=/opt/cuda-13.3/bin:"$PATH"', rendered
        )
        self.assertIn("--plat manylinux_2_28_aarch64", rendered)
        for external in (
            "libcuda.so.1",
            "libcudart.so*",
            "libcublas.so*",
            "libcublasLt.so*",
        ):
            self.assertIn(f"--exclude {external}", rendered)
        self.assertIn("VerifyCudaCubins.cmake", rendered)
        self.assertIn("sm_121a", rendered)

    def test_default_container_command_has_no_cuda_input_or_exclusion(self):
        builder = load_builder()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("source", "work", "build"):
                (root / name).mkdir()
            command = builder.container_command(
                provider="default",
                arch="aarch64",
                source_dir=root / "source",
                work_dir=root / "work",
                build_dir=root / "build",
                cuda_root=None,
                jobs=2,
            )

        rendered = "\n".join(command)
        self.assertNotIn("/opt/cuda-13.3", rendered)
        self.assertNotIn("--exclude", rendered)
        self.assertNotIn("VerifyCudaCubins.cmake", rendered)
        self.assertIn("--plat manylinux_2_28_aarch64", rendered)

    def test_repaired_wheel_contract_rejects_host_linux_tag_and_duplicates(self):
        builder = load_builder()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / (
                "synthesize_cpp_native-0.1.0-py3-none-"
                "manylinux_2_28_aarch64.whl"
            )
            self._write_default_wheel(wheel, "manylinux_2_28_aarch64")

            contract = builder.validate_repaired_wheel(
                wheel, provider="default", arch="aarch64"
            )
            self.assertEqual(contract["backends"], ["cpu"])

            host_wheel = root / (
                "synthesize_cpp_native-0.1.0-py3-none-linux_aarch64.whl"
            )
            self._write_default_wheel(host_wheel, "linux_aarch64")
            with self.assertRaisesRegex(ValueError, "manylinux_2_28"):
                builder.validate_repaired_wheel(
                    host_wheel, provider="default", arch="aarch64"
                )

            duplicate = root / (
                "duplicate-0.1.0-py3-none-manylinux_2_28_aarch64.whl"
            )
            self._write_default_wheel(
                duplicate,
                "manylinux_2_28_aarch64",
                extra={"synthesize_cpp_native/_native/libggml.so.1": b""},
            )
            with self.assertRaisesRegex(ValueError, "versioned shared library"):
                builder.validate_repaired_wheel(
                    duplicate, provider="default", arch="aarch64"
                )

            flat_module = root / (
                "flat-module-0.1.0-py3-none-manylinux_2_28_aarch64.whl"
            )
            self._write_default_wheel(
                flat_module,
                "manylinux_2_28_aarch64",
                extra={
                    "synthesize_cpp_native/_native/libggml-cpu.so": b""
                },
            )
            with self.assertRaisesRegex(ValueError, "flat backend module"):
                builder.validate_repaired_wheel(
                    flat_module, provider="default", arch="aarch64"
                )

            unlicensed = root / (
                "unlicensed-0.1.0-py3-none-manylinux_2_28_aarch64.whl"
            )
            self._write_default_wheel(
                unlicensed,
                "manylinux_2_28_aarch64",
                include_licenses=False,
            )
            with self.assertRaisesRegex(ValueError, "license"):
                builder.validate_repaired_wheel(
                    unlicensed, provider="default", arch="aarch64"
                )

    def test_x86_wheel_requires_every_runtime_dispatched_cpu_module(self):
        builder = load_builder()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / (
                "synthesize_cpp_native-0.1.0-py3-none-"
                "manylinux_2_28_x86_64.whl"
            )
            self._write_default_wheel(
                wheel, "manylinux_2_28_x86_64", arch="x86_64"
            )
            contract = builder.validate_repaired_wheel(
                wheel, provider="default", arch="x86_64"
            )
            self.assertEqual(contract["backends"], ["cpu"])

            incomplete = root / (
                "incomplete-0.1.0-py3-none-manylinux_2_28_x86_64.whl"
            )
            self._write_default_wheel(
                incomplete,
                "manylinux_2_28_x86_64",
                arch="x86_64",
                omitted={"synthesize/backends/libggml-cpu-x64.so"},
            )
            with self.assertRaisesRegex(ValueError, "libggml-cpu-x64.so"):
                builder.validate_repaired_wheel(
                    incomplete, provider="default", arch="x86_64"
                )

    @staticmethod
    def _write_default_wheel(
        path,
        platform_tag,
        extra=None,
        include_licenses=True,
        arch="aarch64",
        omitted=None,
    ):
        builder = load_builder()
        contract = {
            "provider_id": "default",
            "distribution": "synthesize-cpp-native",
            "version": "0.1.0",
            "base_release": "0.1.0",
            "header_hash": "1" * 64,
            "backends": ["cpu"],
        }
        files = {
            "synthesize_cpp_native/_native/contract.json": json.dumps(
                contract
            ).encode(),
            "synthesize_cpp_native/_native/libsynthesize.so": b"",
            "synthesize_cpp_native/_native/libggml.so": b"",
            "synthesize_cpp_native/_native/libggml-base.so": b"",
            "synthesize_cpp_native-0.1.0.dist-info/WHEEL": (
                "Wheel-Version: 1.0\n"
                "Generator: test\n"
                "Root-Is-Purelib: false\n"
                f"Tag: py3-none-{platform_tag}\n"
            ).encode(),
        }
        omitted = omitted or set()
        for module in builder.CPU_MODULES[arch]:
            if module not in omitted:
                files[f"synthesize_cpp_native/_native/{module}"] = b""
        if include_licenses:
            files.update(
                {
                    "synthesize_cpp_native-0.1.0.dist-info/licenses/THIRD_PARTY_NOTICES.md": b"notice",
                    "synthesize_cpp_native-0.1.0.dist-info/licenses/ggml/LICENSE": b"license",
                }
            )
        files.update(extra or {})
        with zipfile.ZipFile(path, "w") as archive:
            for name, data in files.items():
                archive.writestr(name, data)


if __name__ == "__main__":
    unittest.main()
