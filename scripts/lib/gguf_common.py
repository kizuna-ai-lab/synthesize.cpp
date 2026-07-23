"""Small shared Module for source-F32 GGUF converters.

Family tensor catalogs and architecture metadata deliberately stay in each
convert-<family>.py script. This Module owns only conventions that every
converter must reproduce: hashing, conventional general.* identity fields,
contiguous F32 encoding, and atomic generated-file replacement.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterator

from gguf import GGUFWriter, LlamaFileType
import numpy as np
import torch


SOURCE_F32_FILE_TYPE = LlamaFileType.ALL_F32


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_relative(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(project_root.resolve()))
    except ValueError:
        return str(resolved)


def git_revision(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    revision = result.stdout.strip()
    return revision if len(revision) == 40 else None


def add_general_identity(
    writer: GGUFWriter,
    *,
    name: str,
    basename: str,
    size_label: str,
    languages: list[str],
    tags: list[str],
    author: str,
    organization: str,
    source_url: str,
    description: str,
    license_id: str,
    license_name: str,
    license_link: str | None = None,
) -> None:
    """Emit conventional GGUF identity fields understood by ggml tooling."""
    writer.add_name(name)
    writer.add_basename(basename)
    writer.add_size_label(size_label)
    writer.add_file_type(int(SOURCE_F32_FILE_TYPE))
    writer.add_languages(languages)
    writer.add_tags(tags)
    writer.add_author(author)
    writer.add_organization(organization)
    writer.add_source_url(source_url)
    writer.add_description(description)
    writer.add_license(license_id)
    writer.add_license_name(license_name)
    if license_link is not None:
        writer.add_license_link(license_link)


def f32_numpy(tensor: torch.Tensor) -> np.ndarray:
    if not torch.is_floating_point(tensor):
        raise TypeError(f"expected a floating-point tensor, got {tensor.dtype}")
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("tensor contains NaN or infinity")
    return np.ascontiguousarray(value.numpy(), dtype=np.float32)


@contextmanager
def atomic_output_path(destination: Path) -> Iterator[Path]:
    """Yield a sibling temporary path and replace destination on success."""
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        yield temporary
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    with atomic_output_path(path) as temporary:
        temporary.write_bytes(payload)
