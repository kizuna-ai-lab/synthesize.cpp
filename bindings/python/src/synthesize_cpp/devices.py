"""Execution Backend and runtime-device value types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag

from . import _native
from .errors import SynthesizeError


class Backend(IntEnum):
    """Execution Backend request passed to model loading and availability checks."""

    AUTO = 0
    CPU = 1
    CPU_ACCEL = 2
    CUDA = 3
    METAL = 4
    VULKAN = 5


class DeviceType(IntEnum):
    """Broad type reported for a runtime device."""

    CPU = 0
    GPU = 1
    IGPU = 2
    ACCEL = 3


class DeviceFlags(IntFlag):
    """Device-memory metadata flags, preserving unknown future bits."""

    MEMORY_INFO_VALID = 1 << 0
    SHARED = 1 << 1
    APPROXIMATE = 1 << 2


@dataclass(frozen=True, slots=True)
class BackendDevice:
    """One copied snapshot from the synthesize.cpp runtime device registry."""

    index: int | None
    name: str
    description: str
    kind: str
    device_id: str | None
    memory_total: int | None
    memory_free: int | None
    device_type: DeviceType
    flags: DeviceFlags


def backend_available(backend: Backend) -> bool:
    """Return whether the selected Provider currently exposes this Backend."""
    if not isinstance(backend, Backend):
        raise TypeError("backend must be a synthesize_cpp.Backend value")
    return bool(_native.backend_available(int(backend)))


def backend_devices() -> tuple[BackendDevice, ...]:
    """Copy the current native runtime-device snapshot into immutable values."""
    try:
        native_devices = _native.backend_devices()
    except RuntimeError as error:
        raise SynthesizeError(
            f"could not enumerate native backend devices: {error}"
        ) from error

    return tuple(_device_from_native(native) for native in native_devices)


def _device_from_native(native: dict) -> BackendDevice:
    flags = DeviceFlags(native["flags"])
    memory_valid = bool(flags & DeviceFlags.MEMORY_INFO_VALID)
    return BackendDevice(
        index=native["index"],
        name=native["name"],
        description=native["description"],
        kind=native["kind"],
        device_id=native["device_id"],
        memory_total=native["memory_total"] if memory_valid else None,
        memory_free=native["memory_free"] if memory_valid else None,
        device_type=DeviceType(native["device_type"]),
        flags=flags,
    )
