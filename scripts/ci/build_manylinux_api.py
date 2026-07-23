#!/usr/bin/env python3
"""Build and validate the CPython 3.11+ abi3 API wheel in manylinux_2_28."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
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

_REQUIRED_PACKAGE_FILES = {
    "synthesize_cpp/__init__.py",
    "synthesize_cpp/_build_contract.py",
    "synthesize_cpp/_providers.py",
    "synthesize_cpp/_runtime.py",
    "synthesize_cpp/audio.py",
    "synthesize_cpp/devices.py",
    "synthesize_cpp/diagnostics.py",
    "synthesize_cpp/errors.py",
    "synthesize_cpp/metadata.py",
    "synthesize_cpp/models.py",
    "synthesize_cpp/voice_profiles.py",
}


def normalize_arch(value: str) -> str:
    normalized = value.lower().replace("-", "_")
    normalized = {"amd64": "x86_64", "arm64": "aarch64"}.get(
        normalized, normalized
    )
    if normalized not in MANYLINUX_IMAGES:
        raise ValueError(f"unsupported manylinux architecture: {value}")
    return normalized


def _container_script(arch: str) -> str:
    platform_tag = f"manylinux_2_28_{arch}"
    return f"""set -euo pipefail
PYTHON=/opt/python/cp311-cp311/bin/python
rm -rf /work/raw /work/repaired
mkdir -p /work/raw /work/repaired
"$PYTHON" -m build --wheel --outdir /work/raw /project/bindings/python
RAW=(/work/raw/*.whl)
if [[ ${{#RAW[@]}} -ne 1 ]]; then
  echo "expected exactly one raw API wheel, found ${{#RAW[@]}}" >&2
  exit 1
fi
auditwheel repair --plat {platform_tag} \
  --wheel-dir /work/repaired "${{RAW[0]}}"
REPAIRED=(/work/repaired/*.whl)
if [[ ${{#REPAIRED[@]}} -ne 1 ]]; then
  echo "expected exactly one repaired API wheel, found ${{#REPAIRED[@]}}" >&2
  exit 1
fi
auditwheel show "${{REPAIRED[0]}}"
"""


def container_command(
    *,
    arch: str,
    source_dir: Path,
    work_dir: Path,
    build_dir: Path,
    jobs: int,
) -> list[str]:
    arch = normalize_arch(arch)
    if jobs < 1:
        raise ValueError("jobs must be positive")
    return [
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
        MANYLINUX_IMAGES[arch],
        "bash",
        "-lc",
        _container_script(arch),
    ]


def _contract_value(source: str, name: str) -> str:
    match = re.search(rf'^{name}\s*=\s*"([^"]+)"\s*$', source, re.MULTILINE)
    if match is None:
        raise ValueError(f"generated Adapter contract is missing {name}")
    return match.group(1)


def validate_repaired_wheel(
    wheel: Path,
    *,
    arch: str,
    expected_version: str,
    expected_header_hash: str,
) -> dict[str, str]:
    arch = normalize_arch(arch)
    required_platform = f"manylinux_2_28_{arch}"
    if required_platform not in wheel.name:
        raise ValueError(
            f"API wheel is not tagged for {required_platform}: {wheel.name}"
        )
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        missing = sorted(_REQUIRED_PACKAGE_FILES.difference(names))
        if missing:
            raise ValueError(f"API wheel is missing package files: {missing}")
        extensions = sorted(
            name
            for name in names
            if name.startswith("synthesize_cpp/_native.") and name.endswith(".so")
        )
        if extensions != ["synthesize_cpp/_native.abi3.so"]:
            raise ValueError(f"API wheel must contain one abi3 extension: {extensions}")
        if any(name.endswith(".in") for name in names):
            raise ValueError("API wheel leaked a build template")
        if any("libsynthesize" in name or "libggml" in name for name in names):
            raise ValueError("API wheel leaked a Native Provider payload")
        wheel_files = [name for name in names if name.endswith(".dist-info/WHEEL")]
        if len(wheel_files) != 1:
            raise ValueError(f"API wheel must contain one WHEEL file: {wheel_files}")
        wheel_metadata = archive.read(wheel_files[0]).decode("utf-8")
        if f"Tag: cp311-abi3-{required_platform}" not in wheel_metadata:
            raise ValueError(
                f"WHEEL metadata does not declare cp311-abi3-{required_platform}"
            )
        contract_source = archive.read(
            "synthesize_cpp/_build_contract.py"
        ).decode("utf-8")

    contract = {
        "version": _contract_value(contract_source, "VERSION"),
        "header_hash": _contract_value(contract_source, "PUBLIC_HEADER_HASH"),
    }
    if contract["version"] != expected_version:
        raise ValueError("Adapter contract version does not match the source")
    if contract["header_hash"] != expected_header_hash:
        raise ValueError("Adapter contract header hash does not match the source")
    return contract


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_contract(source_dir: Path) -> tuple[str, str]:
    header = source_dir / "include/synthesize.h"
    source = header.read_text(encoding="utf-8")
    values = []
    for component in ("MAJOR", "MINOR", "PATCH"):
        match = re.search(rf"SYNTH_VERSION_{component}\s+(\d+)", source)
        if match is None:
            raise ValueError(f"could not parse SYNTH_VERSION_{component}")
        values.append(match.group(1))
    return ".".join(values), _sha256(header)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=sorted(MANYLINUX_IMAGES))
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source_dir = (args.source_dir or Path(__file__).resolve().parents[2]).resolve()
    arch = normalize_arch(args.arch or platform.machine())
    if arch != normalize_arch(platform.machine()):
        raise ValueError("release API wheels require native-architecture execution")
    version, header_hash = _source_contract(source_dir)
    output_dir = (args.output_dir or source_dir / "wheelhouse/api" / arch).resolve()
    build_dir = (args.build_dir or source_dir / "build/manylinux/api" / arch).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["docker", "image", "inspect", MANYLINUX_IMAGES[arch]],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    with tempfile.TemporaryDirectory(
        prefix=".synthesize-manylinux-api-", dir=output_dir
    ) as directory:
        work_dir = Path(directory)
        command = container_command(
            arch=arch,
            source_dir=source_dir,
            work_dir=work_dir,
            build_dir=build_dir,
            jobs=args.jobs,
        )
        print("+", " ".join(command[:-1]), "<container script>", flush=True)
        subprocess.run(command, check=True)
        wheels = sorted((work_dir / "repaired").glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError(f"expected one repaired API wheel, found {wheels}")
        wheel = wheels[0]
        validate_repaired_wheel(
            wheel,
            arch=arch,
            expected_version=version,
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
    print(f"{destination}: sha256={_sha256(destination)} cp311-abi3")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileExistsError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"error: {error}") from error
