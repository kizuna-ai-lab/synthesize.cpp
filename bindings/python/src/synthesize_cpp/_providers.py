"""Strict discovery and process-global selection of one Native Provider."""

from __future__ import annotations

import importlib.metadata
import os
import re
import sys
import threading
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from .errors import ProviderError


ENTRY_POINT_GROUP = "synthesize_cpp.native"
OVERRIDE_ENVIRONMENT_VARIABLE = "SYNTHESIZE_NATIVE_PROVIDER"

_BACKEND_RANK = {
    "cuda": 5,
    "metal": 4,
    "vulkan": 3,
    "cpu_accel": 2,
    "cpu": 1,
}
_PROVIDER_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_FIELDS = (
    "provider_id",
    "distribution",
    "artifact_dir",
    "base_release",
    "header_hash",
    "backends",
)


def library_filename() -> str:
    if sys.platform == "darwin":
        return "libsynthesize.dylib"
    if sys.platform == "win32":
        return "synthesize-1.dll"
    return "libsynthesize.so"


class Provider:
    """Validated, side-effect-free Native Provider descriptor."""

    def __init__(
        self,
        *,
        provider_id: str,
        distribution: str,
        artifact_dir: Path,
        base_release: str,
        header_hash: str,
        backends: tuple[str, ...],
        prepare: Callable[[], None] | None,
    ) -> None:
        self.provider_id = provider_id
        self.distribution = distribution
        self.artifact_dir = artifact_dir
        self.base_release = base_release
        self.header_hash = header_hash
        self.backends = backends
        self._prepare = prepare

    @property
    def library_path(self) -> Path:
        return self.artifact_dir / library_filename()

    @property
    def rank(self) -> int:
        return max(_BACKEND_RANK[backend] for backend in self.backends)


def _provider_from_descriptor(
    entry_point_name: str,
    descriptor: object,
    *,
    expected_base_release: str,
    expected_header_hash: str,
) -> Provider:
    if not isinstance(descriptor, Mapping):
        raise ProviderError(
            f"native Provider {entry_point_name!r} descriptor is not a mapping"
        )
    missing = [field for field in _REQUIRED_FIELDS if field not in descriptor]
    if missing:
        raise ProviderError(
            f"native Provider {entry_point_name!r} is missing descriptor field "
            f"{missing[0]!r}"
        )

    provider_id = descriptor["provider_id"]
    distribution = descriptor["distribution"]
    artifact_value = descriptor["artifact_dir"]
    base_release = descriptor["base_release"]
    header_hash = descriptor["header_hash"]
    backend_value = descriptor["backends"]
    prepare = descriptor.get("prepare")

    if not isinstance(provider_id, str) or not _PROVIDER_ID.fullmatch(provider_id):
        raise ProviderError(
            f"native Provider {entry_point_name!r} has invalid provider_id"
        )
    if provider_id != entry_point_name:
        raise ProviderError(
            f"native Provider entry point {entry_point_name!r} reports "
            f"different provider_id {provider_id!r}"
        )
    if not isinstance(distribution, str) or not distribution:
        raise ProviderError(
            f"native Provider {provider_id!r} has invalid distribution"
        )
    if not isinstance(artifact_value, str) or not artifact_value:
        raise ProviderError(
            f"native Provider {provider_id!r} has invalid artifact_dir"
        )
    artifact_dir = Path(artifact_value)
    if not artifact_dir.is_absolute():
        raise ProviderError(
            f"native Provider {provider_id!r} artifact_dir must be absolute"
        )
    if not artifact_dir.is_dir():
        raise ProviderError(
            f"native Provider {provider_id!r} artifact directory is missing: "
            f"{artifact_dir}"
        )

    if base_release != expected_base_release:
        raise ProviderError(
            f"native Provider {provider_id!r} base release {base_release!r} "
            f"does not match synthesize-cpp {expected_base_release!r}"
        )
    if not isinstance(header_hash, str) or not _LOWER_SHA256.fullmatch(
        header_hash
    ):
        raise ProviderError(
            f"native Provider {provider_id!r} header_hash is not lowercase SHA-256"
        )
    if header_hash != expected_header_hash:
        raise ProviderError(
            f"native Provider {provider_id!r} public-header hash does not match "
            "this synthesize-cpp Adapter"
        )

    if not isinstance(backend_value, (list, tuple)) or not backend_value:
        raise ProviderError(
            f"native Provider {provider_id!r} backends must be a non-empty list"
        )
    backends = tuple(backend_value)
    if any(
        not isinstance(backend, str) or backend not in _BACKEND_RANK
        for backend in backends
    ):
        raise ProviderError(
            f"native Provider {provider_id!r} advertises an unknown backend"
        )
    if len(set(backends)) != len(backends):
        raise ProviderError(
            f"native Provider {provider_id!r} advertises duplicate backends"
        )
    if "cpu" not in backends:
        raise ProviderError(
            f"native Provider {provider_id!r} must contain the CPU backend"
        )
    if prepare is not None and not callable(prepare):
        raise ProviderError(
            f"native Provider {provider_id!r} prepare field is not callable"
        )

    provider = Provider(
        provider_id=provider_id,
        distribution=distribution,
        artifact_dir=artifact_dir,
        base_release=base_release,
        header_hash=header_hash,
        backends=backends,
        prepare=prepare,
    )
    if not provider.library_path.is_file():
        raise ProviderError(
            f"native Provider {provider_id!r} library is missing: "
            f"{provider.library_path}"
        )
    return provider


def discover_providers(
    entry_points: Iterable[object],
    *,
    expected_base_release: str,
    expected_header_hash: str,
) -> tuple[Provider, ...]:
    """Load and validate every descriptor without running preparation hooks."""
    providers = []
    seen_ids = set()
    for entry_point in entry_points:
        name = getattr(entry_point, "name", "<unnamed>")
        try:
            loaded = entry_point.load()
            descriptor = loaded() if callable(loaded) else loaded
        except Exception as error:
            raise ProviderError(
                f"native Provider {name!r} is broken: {error}"
            ) from error
        provider = _provider_from_descriptor(
            name,
            descriptor,
            expected_base_release=expected_base_release,
            expected_header_hash=expected_header_hash,
        )
        if provider.provider_id in seen_ids:
            raise ProviderError(
                f"native Provider ID {provider.provider_id!r} is registered twice"
            )
        seen_ids.add(provider.provider_id)
        providers.append(provider)
    return tuple(providers)


def select_provider(
    providers: Iterable[Provider], override: str | None = None
) -> Provider:
    candidates = tuple(providers)
    if override is not None:
        for provider in candidates:
            if provider.provider_id == override:
                return provider
        installed = ", ".join(
            sorted(provider.provider_id for provider in candidates)
        ) or "(none)"
        raise ProviderError(
            f"requested native Provider {override!r} is not installed; "
            f"installed Provider IDs: {installed}"
        )
    if not candidates:
        raise ProviderError(
            "no native Provider is installed; install synthesize-cpp-native "
            "or an accelerator extra matching synthesize-cpp"
        )

    highest = max(provider.rank for provider in candidates)
    finalists = tuple(
        provider for provider in candidates if provider.rank == highest
    )
    if len(finalists) != 1:
        names = ", ".join(sorted(provider.provider_id for provider in finalists))
        raise ProviderError(
            f"ambiguous highest-ranked native Providers: {names}; set "
            f"{OVERRIDE_ENVIRONMENT_VARIABLE} to one exact Provider ID"
        )
    return finalists[0]


def installed_entry_points() -> list[object]:
    entry_points = importlib.metadata.entry_points()
    if hasattr(entry_points, "select"):
        return list(entry_points.select(group=ENTRY_POINT_GROUP))
    return list(entry_points.get(ENTRY_POINT_GROUP, []))


class ProviderRegistry:
    """Select and prepare one Provider exactly once for this Adapter instance."""

    def __init__(
        self,
        entry_point_loader: Callable[[], Iterable[object]] = installed_entry_points,
        *,
        expected_base_release: str,
        expected_header_hash: str,
        environ: Mapping[str, str] = os.environ,
    ) -> None:
        self._entry_point_loader = entry_point_loader
        self._expected_base_release = expected_base_release
        self._expected_header_hash = expected_header_hash
        self._environ = environ
        self._lock = threading.RLock()
        self._selected: Provider | None = None
        self._prepared = False
        self._prepare_error: ProviderError | None = None

    def select(self) -> Provider:
        with self._lock:
            if self._selected is None:
                providers = discover_providers(
                    self._entry_point_loader(),
                    expected_base_release=self._expected_base_release,
                    expected_header_hash=self._expected_header_hash,
                )
                self._selected = select_provider(
                    providers,
                    override=self._environ.get(
                        OVERRIDE_ENVIRONMENT_VARIABLE
                    ),
                )
            return self._selected

    def prepare(self) -> Provider:
        with self._lock:
            provider = self.select()
            if self._prepare_error is not None:
                raise self._prepare_error
            if self._prepared:
                return provider
            try:
                if provider._prepare is not None:
                    provider._prepare()
            except Exception as error:
                self._prepare_error = ProviderError(
                    f"native Provider {provider.provider_id!r} failed in its "
                    f"prepare hook: {error}"
                )
                raise self._prepare_error from error
            self._prepared = True
            return provider
