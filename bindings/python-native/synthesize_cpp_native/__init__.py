"""Default platform-native runtime Provider for :mod:`synthesize_cpp`."""

from pathlib import Path


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
    }
