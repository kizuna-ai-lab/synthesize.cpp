#!/usr/bin/env python3
"""Build and audit one native Provider wheel in a pinned manylinux_2_28 image.

The source and CUDA toolkit are mounted read-only. CMake state is kept in a
Provider-owned build directory and never shares a committed preset tree. The
produced artifact is accepted only after auditwheel repair, contract inspection,
and (for CUDA) the existing exact-cubin/no-PTX verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


MANYLINUX_IMAGES = {
    "aarch64": (
        "quay.io/pypa/manylinux_2_28_aarch64@"
        "sha256:162c81dfd3efc710732a571717d3c916a6945ebf279e879ddee3243af96fe46f"
    ),
    "x86_64": (
        "quay.io/pypa/manylinux_2_28_x86_64@"
        "sha256:a61875a2f84cab7df8de222ff12cabc08ff86eb4ad402ac90ba7bdaed9600cca"
    ),
}

CUDA_ARCHITECTURES = {
    "aarch64": "121a-real",
    "x86_64": "75-real;80-real;86-real;89-real;90-real;100-real;120a-real",
}

EXPECTED_CUBINS = {
    "aarch64": "sm_121a",
    "x86_64": "sm_75,sm_80,sm_86,sm_89,sm_90,sm_100,sm_120a",
}

_X86_64_CPU_VARIANTS = (
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

CPU_MODULES = {
    "aarch64": ("synthesize/backends/libggml-cpu.so",),
    "x86_64": tuple(
        f"synthesize/backends/libggml-cpu-{variant}.so"
        for variant in _X86_64_CPU_VARIANTS
    ),
}

_PROVIDERS = {
    "default": {
        "project": "/project",
        "package": "synthesize_cpp_native",
        "distribution": "synthesize-cpp-native",
        "backends": ["cpu"],
        "libraries": (
            "libsynthesize.so",
            "libggml.so",
            "libggml-base.so",
        ),
    },
    "cu13": {
        "project": "/project/bindings/python-native-cu13",
        "package": "synthesize_cpp_native_cu13",
        "distribution": "synthesize-cpp-native-cu13",
        "backends": ["cuda", "cpu"],
        "libraries": (
            "libsynthesize.so",
            "libggml.so",
            "libggml-base.so",
            "synthesize/backends/libggml-cuda.so",
        ),
    },
}


def normalize_arch(value: str) -> str:
    normalized = value.lower().replace("-", "_")
    aliases = {
        "amd64": "x86_64",
        "arm64": "aarch64",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in MANYLINUX_IMAGES:
        raise ValueError(f"unsupported manylinux architecture: {value}")
    return normalized


def _provider_libraries(provider: str, arch: str) -> tuple[str, ...]:
    return (*_PROVIDERS[provider]["libraries"], *CPU_MODULES[arch])


def _container_script(provider: str, arch: str) -> str:
    config = _PROVIDERS[provider]
    platform_tag = f"manylinux_2_28_{arch}"
    excludes = ""
    cuda_verification = ""
    path_setup = ""
    if provider == "cu13":
        path_setup = 'export PATH=/opt/cuda-13.3/bin:"$PATH"'
        excludes = " ".join(
            f"--exclude {library}"
            for library in (
                "libcuda.so.1",
                "libcudart.so*",
                "libcublas.so*",
                "libcublasLt.so*",
            )
        )
        cuda_verification = f"""
\"$PYTHON\" - \"${{REPAIRED[0]}}\" <<'PY'
import sys
import zipfile
from pathlib import Path

wheel = Path(sys.argv[1])
with zipfile.ZipFile(wheel) as archive:
    names = [name for name in archive.namelist()
             if name.endswith('/_native/synthesize/backends/libggml-cuda.so')]
    if len(names) != 1:
        raise SystemExit(f"expected one libggml-cuda.so, found {{names}}")
    Path('/work/libggml-cuda.so').write_bytes(archive.read(names[0]))
PY
cmake \
  -DSYNTH_CUOBJDUMP=/opt/cuda-13.3/bin/cuobjdump \
  -DSYNTH_CUDA_LIBRARY=/work/libggml-cuda.so \
  -DSYNTH_EXPECTED_CUBINS={EXPECTED_CUBINS[arch]} \
  -P /project/cmake/VerifyCudaCubins.cmake
"""

    return f"""set -euo pipefail
{path_setup}
PYTHON=/opt/python/cp311-cp311/bin/python
rm -rf /work/raw /work/repaired /work/libggml-cuda.so
mkdir -p /work/raw /work/repaired
\"$PYTHON\" -m build --wheel --outdir /work/raw {config['project']}
RAW=(/work/raw/*.whl)
if [[ ${{#RAW[@]}} -ne 1 ]]; then
  echo \"expected exactly one raw wheel, found ${{#RAW[@]}}\" >&2
  exit 1
fi
auditwheel repair --plat {platform_tag} {excludes} \
  --wheel-dir /work/repaired \"${{RAW[0]}}\"
REPAIRED=(/work/repaired/*.whl)
if [[ ${{#REPAIRED[@]}} -ne 1 ]]; then
  echo \"expected exactly one repaired wheel, found ${{#REPAIRED[@]}}\" >&2
  exit 1
fi
auditwheel show \"${{REPAIRED[0]}}\"
{cuda_verification}"""


def container_command(
    *,
    provider: str,
    arch: str,
    source_dir: Path,
    work_dir: Path,
    build_dir: Path,
    cuda_root: Path | None,
    jobs: int,
) -> list[str]:
    if provider not in _PROVIDERS:
        raise ValueError(f"unsupported Provider: {provider}")
    arch = normalize_arch(arch)
    if jobs < 1:
        raise ValueError("jobs must be positive")
    if provider == "cu13" and cuda_root is None:
        raise ValueError("cu13 requires an explicit CUDA 13.3 toolkit root")
    if provider == "default" and cuda_root is not None:
        raise ValueError("the default Provider does not accept a CUDA toolkit")

    command = [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--mount",
        f"type=bind,src={source_dir.resolve()},dst=/project,readonly",
        "--mount",
        f"type=bind,src={work_dir.resolve()},dst=/work",
        "--mount",
        f"type=bind,src={build_dir.resolve()},dst=/build",
        "--env",
        "PIP_DISABLE_PIP_VERSION_CHECK=1",
        "--env",
        "PIP_CACHE_DIR=/tmp/synthesize-pip-cache",
        "--env",
        f"CMAKE_BUILD_PARALLEL_LEVEL={jobs}",
        "--env",
        "SKBUILD_BUILD_DIR=/build",
    ]
    if provider == "cu13":
        command.extend(
            [
                "--mount",
                f"type=bind,src={cuda_root.resolve()},dst=/opt/cuda-13.3,readonly",
                "--env",
                "CUDAToolkit_ROOT=/opt/cuda-13.3",
                "--env",
                "CUDACXX=/opt/cuda-13.3/bin/nvcc",
                "--env",
                "LD_LIBRARY_PATH=/opt/cuda-13.3/lib64",
                "--env",
                f"CMAKE_ARGS=-DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCHITECTURES[arch]}",
            ]
        )
    command.extend(
        [
            MANYLINUX_IMAGES[arch],
            "bash",
            "-lc",
            _container_script(provider, arch),
        ]
    )
    return command


def validate_repaired_wheel(
    wheel: Path,
    *,
    provider: str,
    arch: str,
    expected_header_hash: str | None = None,
) -> dict:
    if provider not in _PROVIDERS:
        raise ValueError(f"unsupported Provider: {provider}")
    arch = normalize_arch(arch)
    required_platform = f"manylinux_2_28_{arch}"
    if required_platform not in wheel.name:
        raise ValueError(
            f"wheel is not tagged for {required_platform}: {wheel.name}"
        )

    config = _PROVIDERS[provider]
    package = config["package"]
    native_root = f"{package}/_native/"
    provider_libraries = _provider_libraries(provider, arch)
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        required = {
            native_root + "contract.json",
            *(native_root + library for library in provider_libraries),
        }
        missing = sorted(required.difference(names))
        if missing:
            raise ValueError(f"wheel is missing Provider artifacts: {missing}")

        flat_backend_modules = sorted(
            native_root + Path(library).name
            for library in provider_libraries
            if library.startswith("synthesize/backends/")
            and native_root + Path(library).name in names
        )
        if flat_backend_modules:
            raise ValueError(
                "wheel contains a flat backend module outside "
                f"synthesize/backends: {flat_backend_modules}"
            )

        versioned = sorted(
            name for name in names if name.startswith(native_root) and ".so." in name
        )
        if versioned:
            raise ValueError(
                f"wheel contains a versioned shared library duplicate: {versioned}"
            )

        leaked = sorted(
            name
            for name in names
            if "/include/" in name
            or "/cmake/" in name
            or "/pkgconfig/" in name
        )
        if leaked:
            raise ValueError(f"wheel leaked native SDK files: {leaked}")

        wheel_metadata = [name for name in names if name.endswith(".dist-info/WHEEL")]
        if len(wheel_metadata) != 1:
            raise ValueError(
                f"wheel must contain one WHEEL metadata file: {wheel_metadata}"
            )
        dist_info = wheel_metadata[0].rsplit("/", 1)[0]
        license_files = [
            name for name in names if name.startswith(dist_info + "/licenses/")
        ]
        if not any(
            name.endswith("/THIRD_PARTY_NOTICES.md") for name in license_files
        ) or not any("license" in Path(name).name.lower() for name in license_files):
            raise ValueError(
                "wheel license payload must include third-party notices and the GGML license"
            )
        metadata = archive.read(wheel_metadata[0]).decode("utf-8")
        if f"Tag: py3-none-{required_platform}" not in metadata:
            raise ValueError(
                f"WHEEL metadata does not declare py3-none-{required_platform}"
            )
        contract = json.loads(
            archive.read(native_root + "contract.json").decode("utf-8")
        )

    expected_contract = {
        "provider_id": provider,
        "distribution": config["distribution"],
        "backends": config["backends"],
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise ValueError(
                f"Provider contract {key} must be {expected!r}, got {contract.get(key)!r}"
            )
    if contract.get("version") != contract.get("base_release"):
        raise ValueError("Provider contract version/base_release mismatch")
    header_hash = contract.get("header_hash", "")
    if len(header_hash) != 64 or any(
        character not in "0123456789abcdef" for character in header_hash
    ):
        raise ValueError("Provider contract header_hash is not lowercase SHA256")
    if expected_header_hash is not None and header_hash != expected_header_hash:
        raise ValueError("Provider contract header_hash does not match the source")
    return contract


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=sorted(_PROVIDERS), required=True)
    parser.add_argument("--arch", choices=sorted(MANYLINUX_IMAGES))
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--cuda-root", type=Path)
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source_dir = (args.source_dir or Path(__file__).resolve().parents[2]).resolve()
    arch = normalize_arch(args.arch or platform.machine())
    host_arch = normalize_arch(platform.machine())
    if arch != host_arch:
        raise ValueError(
            f"release wheels require native execution: host={host_arch}, requested={arch}"
        )
    if not (source_dir / "include/synthesize.h").is_file():
        raise ValueError(f"not a synthesize.cpp source tree: {source_dir}")

    output_dir = (
        args.output_dir
        or source_dir / "wheelhouse" / args.provider / arch
    ).resolve()
    build_dir = (
        args.build_dir
        or source_dir / "build/manylinux" / args.provider / arch
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    cuda_root = args.cuda_root.resolve() if args.cuda_root else None
    if args.provider == "cu13":
        if cuda_root is None or not (cuda_root / "bin/nvcc").is_file():
            raise ValueError("--cuda-root must name a CUDA 13.3 toolkit")
        nvcc = subprocess.run(
            [str(cuda_root / "bin/nvcc"), "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if "release 13.3" not in nvcc:
            raise ValueError("--cuda-root is not CUDA 13.3")
    elif cuda_root is not None:
        raise ValueError("--cuda-root is valid only for the cu13 Provider")

    subprocess.run(
        ["docker", "image", "inspect", MANYLINUX_IMAGES[arch]],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    header_hash = _sha256(source_dir / "include/synthesize.h")
    with tempfile.TemporaryDirectory(
        prefix=".synthesize-manylinux-", dir=output_dir
    ) as directory:
        work_dir = Path(directory)
        command = container_command(
            provider=args.provider,
            arch=arch,
            source_dir=source_dir,
            work_dir=work_dir,
            build_dir=build_dir,
            cuda_root=cuda_root,
            jobs=args.jobs,
        )
        print("+", " ".join(command[:-1]), "<container script>", flush=True)
        subprocess.run(command, check=True)
        wheels = sorted((work_dir / "repaired").glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError(f"expected one repaired wheel, found {wheels}")
        wheel = wheels[0]
        contract = validate_repaired_wheel(
            wheel,
            provider=args.provider,
            arch=arch,
            expected_header_hash=header_hash,
        )
        destination = output_dir / wheel.name
        if destination.exists():
            if _sha256(destination) != _sha256(wheel):
                raise FileExistsError(
                    f"refusing to overwrite different artifact: {destination}"
                )
        else:
            shutil.copy2(wheel, destination)

    print(
        f"{destination}: sha256={_sha256(destination)} "
        f"provider={contract['provider_id']} backends={contract['backends']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileExistsError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"error: {error}") from error
