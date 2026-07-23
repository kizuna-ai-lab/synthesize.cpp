"""CUDA 13 platform-native runtime Provider for :mod:`synthesize_cpp`."""

from __future__ import annotations

import sys
import threading
from pathlib import Path


if sys.platform == "win32":
    _NVIDIA_LIBRARIES = (
        "cu13/bin/cudart64_13.dll",
        "cu13/bin/cublasLt64_13.dll",
        "cu13/bin/cublas64_13.dll",
    )
else:
    _NVIDIA_LIBRARIES = (
        "cu13/lib/libcudart.so.13",
        "cu13/lib/libcublasLt.so.13",
        "cu13/lib/libcublas.so.13",
    )

_PREPARE_LOCK = threading.Lock()
_RUNTIME_HANDLES: list[object] = []


def prepare() -> None:
    """Preload the release-pinned NVIDIA runtime without changing process paths.

    The Adapter calls this only after selecting ``cu13``. A missing package,
    library, or loader dependency is therefore a hard Provider error; selection
    never falls through silently to another runtime.
    """
    import ctypes
    import os

    with _PREPARE_LOCK:
        if _RUNTIME_HANDLES:
            return
        try:
            import nvidia
        except ImportError as error:
            raise RuntimeError(
                "cu13 Provider requires its pinned NVIDIA runtime packages"
            ) from error

        roots = tuple(Path(root) for root in nvidia.__path__)
        resolved = []
        for relative in _NVIDIA_LIBRARIES:
            library = next(
                (root / relative for root in roots if (root / relative).is_file()),
                None,
            )
            if library is None:
                raise RuntimeError(
                    f"cu13 Provider runtime library is missing: {relative}"
                )
            resolved.append(library)

        handles = []
        for library in resolved:
            try:
                if sys.platform == "win32":
                    os.add_dll_directory(str(library.parent))
                    handle = ctypes.CDLL(str(library))
                else:
                    handle = ctypes.CDLL(
                        str(library), mode=ctypes.RTLD_GLOBAL
                    )
            except OSError as error:
                raise RuntimeError(
                    f"cu13 Provider could not preload {library}: {error}"
                ) from error
            handles.append(handle)
        _RUNTIME_HANDLES.extend(handles)


def descriptor() -> dict:
    """Return the internal ``synthesize_cpp.native`` Provider contract."""
    from . import _contract

    return {
        "provider_id": _contract.PROVIDER_ID,
        "distribution": _contract.DISTRIBUTION,
        "artifact_dir": str(Path(__file__).resolve().parent / "_native"),
        "base_release": _contract.VERSION,
        "header_hash": _contract.PUBLIC_HEADER_HASH,
        "backends": list(_contract.BACKENDS),
        "prepare": prepare,
    }
