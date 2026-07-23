from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRESETS_PATH = PROJECT_ROOT / "CMakePresets.json"
POLICY_PATH = PROJECT_ROOT / "cmake" / "SynthesizeReleaseCudaPolicy.cmake"
TOOLCHAIN_POLICY_PATH = (
    PROJECT_ROOT / "cmake" / "SynthesizeCudaToolchainPolicy.cmake"
)
CUBIN_POLICY_PATH = PROJECT_ROOT / "cmake" / "VerifyCudaCubins.cmake"


class CMakePresetContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = json.loads(PRESETS_PATH.read_text(encoding="utf-8"))
        cls.configure = {
            preset["name"]: preset
            for preset in cls.document.get("configurePresets", [])
        }

    def resolved_configure(self, name: str) -> dict[str, object]:
        def merge(target: dict[str, object], source: dict[str, object]) -> None:
            for key, value in source.items():
                if key in {"name", "inherits", "hidden", "displayName", "description"}:
                    continue
                if key == "cacheVariables":
                    variables = target.setdefault(key, {})
                    assert isinstance(variables, dict)
                    assert isinstance(value, dict)
                    for variable, setting in value.items():
                        variables.setdefault(variable, setting)
                else:
                    target.setdefault(key, value)

        def visit(preset_name: str, stack: tuple[str, ...]) -> dict[str, object]:
            self.assertNotIn(preset_name, stack, "configure-preset inheritance cycle")
            preset = self.configure[preset_name]
            resolved: dict[str, object] = {}
            parents = preset.get("inherits", [])
            if isinstance(parents, str):
                parents = [parents]
            for parent in parents:
                merge(resolved, visit(parent, stack + (preset_name,)))
            # A derived preset overrides inherited fields.
            for key, value in preset.items():
                if key == "cacheVariables":
                    variables = resolved.setdefault(key, {})
                    assert isinstance(variables, dict)
                    variables.update(value)
                elif key not in {"inherits"}:
                    resolved[key] = value
            return resolved

        return visit(name, ())

    def test_schema_and_minimum_cmake_match_the_build_contract(self) -> None:
        self.assertEqual(self.document["version"], 5)
        self.assertEqual(
            self.document["cmakeMinimumRequired"],
            {"major": 3, "minor": 24, "patch": 0},
        )
        first_line = (PROJECT_ROOT / "CMakeLists.txt").read_text(
            encoding="utf-8"
        ).splitlines()[0]
        self.assertEqual(first_line, "cmake_minimum_required(VERSION 3.24)")

    def test_initial_cuda_configure_presets_are_complete_and_isolated(self) -> None:
        expected = {
            "dev-dgx-spark",
            "dev-dgx-spark-uvm",
            "release-linux-aarch64-cu13",
            "dev-linux-x86_64-cuda",
            "release-linux-x86_64-cu13",
        }
        visible = {
            name
            for name, preset in self.configure.items()
            if not preset.get("hidden", False)
        }
        self.assertEqual(visible, expected)
        for name in expected:
            with self.subTest(name=name):
                resolved = self.resolved_configure(name)
                self.assertEqual(resolved["binaryDir"], "${sourceDir}/build/${presetName}")
                variables = resolved["cacheVariables"]
                self.assertEqual(variables["SYNTH_CUDA"], "ON")
                self.assertEqual(variables["SYNTH_CUDA_TF32"], "OFF")
                self.assertEqual(variables["SYNTH_METAL"], "OFF")
                self.assertEqual(variables["SYNTH_VULKAN"], "OFF")
                self.assertEqual(variables["GGML_NATIVE"], "OFF")
                self.assertNotIn("GGML_CUDA_ENABLE_UNIFIED_MEMORY", variables)

    def test_development_and_release_cuda_target_sets_are_explicit(self) -> None:
        expected = {
            "dev-dgx-spark": "121a-real",
            "dev-dgx-spark-uvm": "121a-real",
            "release-linux-aarch64-cu13": "121a-real",
            "dev-linux-x86_64-cuda": "89-real",
            "release-linux-x86_64-cu13": (
                "75-real;80-real;86-real;89-real;90-real;100-real;120a-real"
            ),
        }
        for name, architectures in expected.items():
            with self.subTest(name=name):
                variables = self.resolved_configure(name)["cacheVariables"]
                self.assertEqual(
                    variables["CMAKE_CUDA_ARCHITECTURES"], architectures
                )
                lowered = architectures.lower().split(";")
                self.assertFalse({"native", "all", "all-major"} & set(lowered))

        aarch64 = self.resolved_configure("release-linux-aarch64-cu13")
        x86_64 = self.resolved_configure("release-linux-x86_64-cu13")
        self.assertEqual(
            aarch64["cacheVariables"]["SYNTH_INTERNAL_RELEASE_CUDA_PLATFORM"],
            "linux-aarch64",
        )
        self.assertEqual(
            x86_64["cacheVariables"]["SYNTH_INTERNAL_RELEASE_CUDA_PLATFORM"],
            "linux-x86_64",
        )

    def test_build_and_test_presets_cover_each_configure_entry(self) -> None:
        configure_names = {
            name
            for name, preset in self.configure.items()
            if not preset.get("hidden", False)
        }
        builds = {
            preset["configurePreset"] for preset in self.document["buildPresets"]
        }
        tests = {
            preset["configurePreset"] for preset in self.document["testPresets"]
        }
        self.assertEqual(builds, configure_names)
        self.assertEqual(
            tests,
            {"dev-dgx-spark", "dev-dgx-spark-uvm", "dev-linux-x86_64-cuda"},
        )
        uvm = next(
            preset
            for preset in self.document["testPresets"]
            if preset["configurePreset"] == "dev-dgx-spark-uvm"
        )
        self.assertEqual(
            uvm["environment"], {"GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1"}
        )
        release_builds = {
            preset["name"]: preset
            for preset in self.document["buildPresets"]
            if preset["name"].startswith("release-")
        }
        for name, preset in release_builds.items():
            with self.subTest(name=name):
                self.assertEqual(
                    preset["targets"],
                    ["synthesize", "synthesize-check-release-cubins"],
                )

    def run_policy(
        self,
        platform: str,
        architectures: str,
        *,
        synth_cuda: str = "ON",
        ggml_native: str = "OFF",
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "cmake",
                f"-DSYNTH_INTERNAL_RELEASE_CUDA_PLATFORM={platform}",
                f"-DCMAKE_CUDA_ARCHITECTURES={architectures}",
                f"-DSYNTH_CUDA={synth_cuda}",
                f"-DGGML_NATIVE={ggml_native}",
                "-P",
                str(POLICY_PATH),
            ],
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    def test_release_policy_accepts_only_exact_target_sets(self) -> None:
        valid = {
            "linux-aarch64": "121a-real",
            "linux-x86_64": (
                "75-real;80-real;86-real;89-real;90-real;100-real;120a-real"
            ),
        }
        for platform, architectures in valid.items():
            with self.subTest(platform=platform):
                self.assertEqual(self.run_policy(platform, architectures).returncode, 0)

        invalid = (
            ("linux-aarch64", "native", "exact native cubin target set"),
            ("linux-aarch64", "121a", "exact native cubin target set"),
            ("linux-x86_64", "all", "exact native cubin target set"),
            ("linux-x86_64", "89-real", "exact native cubin target set"),
            ("unknown", "121a-real", "unknown release CUDA platform"),
        )
        for platform, architectures, message in invalid:
            with self.subTest(platform=platform, architectures=architectures):
                result = self.run_policy(platform, architectures)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stdout)

        self.assertNotEqual(
            self.run_policy("linux-aarch64", "121a-real", synth_cuda="OFF").returncode,
            0,
        )
        self.assertNotEqual(
            self.run_policy("linux-aarch64", "121a-real", ggml_native="ON").returncode,
            0,
        )

    def test_cuda_toolchain_policy_accepts_only_13_3(self) -> None:
        def run(version: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [
                    "cmake",
                    f"-DCUDAToolkit_VERSION={version}",
                    "-P",
                    str(TOOLCHAIN_POLICY_PATH),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

        for version in ("13.3", "13.3.73"):
            with self.subTest(version=version):
                self.assertEqual(run(version).returncode, 0)
        for version in ("", "13.0.88", "13.2.9", "13.4.0", "14.0"):
            with self.subTest(version=version):
                result = run(version)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("CUDA Toolkit 13.3", result.stdout)

    def test_cubin_verifier_requires_exact_native_set_and_no_ptx(self) -> None:
        def run(
            actual_cubins: tuple[str, ...],
            *,
            expected: tuple[str, ...] = ("sm_75", "sm_89", "sm_120a"),
            ptx: bool = False,
        ) -> subprocess.CompletedProcess[str]:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                tool = root / "fake-cuobjdump"
                library = root / "libggml-cuda.so"
                elf_lines = "\\n".join(
                    f"ELF file {index}: fake.{cubin}.cubin"
                    for index, cubin in enumerate(actual_cubins, start=1)
                )
                ptx_line = "PTX file 1: fake.compute_121.ptx" if ptx else (
                    "cuobjdump info: No PTX file found to extract"
                )
                tool.write_text(
                    "#!/bin/sh\n"
                    "case \"$1\" in\n"
                    f"  --list-elf) printf '%s\\n' '{elf_lines}' ;;\n"
                    f"  --list-ptx) printf '%s\\n' '{ptx_line}' ;;\n"
                    "  *) exit 2 ;;\n"
                    "esac\n",
                    encoding="utf-8",
                )
                tool.chmod(tool.stat().st_mode | 0o100)
                library.write_bytes(b"fixture")
                environment = os.environ.copy()
                return subprocess.run(
                    [
                        "cmake",
                        f"-DSYNTH_CUOBJDUMP={tool}",
                        f"-DSYNTH_CUDA_LIBRARY={library}",
                        f"-DSYNTH_EXPECTED_CUBINS={','.join(expected)}",
                        "-P",
                        str(CUBIN_POLICY_PATH),
                    ],
                    cwd=PROJECT_ROOT,
                    env=environment,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )

        self.assertEqual(run(("sm_120a", "sm_75", "sm_89", "sm_75")).returncode, 0)
        for actual in (
            ("sm_75", "sm_89"),
            ("sm_75", "sm_89", "sm_100", "sm_120a"),
        ):
            with self.subTest(actual=actual):
                result = run(actual)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("exact native cubin set", result.stdout)
        result = run(("sm_75", "sm_89", "sm_120a"), ptx=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must not contain PTX", result.stdout)


if __name__ == "__main__":
    unittest.main()
