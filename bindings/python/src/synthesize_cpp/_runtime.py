"""Internal process-global Native Provider bootstrap."""

from __future__ import annotations

from . import _build_contract, _native
from ._providers import ProviderRegistry
from .errors import ProviderError


_BACKEND_REQUESTS = {
    "cpu": 1,
    "cpu_accel": 2,
    "cuda": 3,
    "metal": 4,
    "vulkan": 5,
}


_registry = ProviderRegistry(
    expected_base_release=_build_contract.VERSION,
    expected_header_hash=_build_contract.PUBLIC_HEADER_HASH,
)
_provider = _registry.prepare()
try:
    _native_info = _native.load(str(_provider.library_path))
    for _backend in _provider.backends:
        if not _native.backend_available(_BACKEND_REQUESTS[_backend]):
            raise RuntimeError(
                f"advertised backend {_backend!r} is unavailable after "
                "module registration"
            )
except Exception as error:
    raise ProviderError(
        f"native Provider {_provider.provider_id!r} failed to load or "
        f"initialize: {error}"
    ) from error
