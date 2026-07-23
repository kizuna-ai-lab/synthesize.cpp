"""Runtime Voice Profile capabilities, sources, and owned handles."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import IntEnum
from enum import IntFlag
from typing import TYPE_CHECKING
from typing import Iterable
from typing import Self

from . import _native
from .diagnostics import DiagnosticCallback
from .diagnostics import _diagnostic_adapter
from .errors import SynthesizeError

if TYPE_CHECKING:
    from .models import Model


class VoiceProfileSourceFlags(IntFlag):
    REFERENCE_AUDIO = 1 << 0
    DESCRIPTION_TEXT = 1 << 1
    RANDOM_SEED = 1 << 2
    SERIALIZED_PROFILE = 1 << 3


class Requirement(IntEnum):
    UNSUPPORTED = 0
    OPTIONAL = 1
    REQUIRED = 2


@dataclass(frozen=True, slots=True)
class VoiceProfileCapabilities:
    source_flags: VoiceProfileSourceFlags
    reference_transcript: Requirement
    reference_language: Requirement
    description_language: Requirement
    max_reference_count: int
    profile_schema: str | None
    profile_schema_version: int
    profile_compatibility_id: bytes
    reference_target_sample_rate: int
    reference_target_channel_count: int
    min_reference_frames_per_clip: int
    max_reference_frames_per_clip: int
    max_reference_total_frames: int


@dataclass(frozen=True, slots=True)
class VoiceReference:
    samples: object
    sample_rate: int
    channel_count: int
    transcript: str | None = None
    language: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.sample_rate, bool) or not isinstance(
            self.sample_rate, int
        ):
            raise TypeError("sample_rate must be an integer")
        if not 8000 <= self.sample_rate <= 192000:
            raise ValueError("sample_rate must be between 8000 and 192000")
        if isinstance(self.channel_count, bool) or not isinstance(
            self.channel_count, int
        ):
            raise TypeError("channel_count must be an integer")
        if self.channel_count not in (1, 2):
            raise ValueError("channel_count must be 1 or 2")
        if self.transcript is not None and not isinstance(self.transcript, str):
            raise TypeError("transcript must be str or None")
        if self.language is not None and not isinstance(self.language, str):
            raise TypeError("language must be str or None")


def _voice_profile_capabilities_from_native(
    native: dict,
) -> VoiceProfileCapabilities:
    compatibility_id = bytes(native["profile_compatibility_id"])
    if len(compatibility_id) != 32:
        raise ValueError("Profile Compatibility ID must contain 32 bytes")
    return VoiceProfileCapabilities(
        source_flags=VoiceProfileSourceFlags(native["source_flags"]),
        reference_transcript=Requirement(native["reference_transcript"]),
        reference_language=Requirement(native["reference_language"]),
        description_language=Requirement(native["description_language"]),
        max_reference_count=native["max_reference_count"],
        profile_schema=native["profile_schema"],
        profile_schema_version=native["profile_schema_version"],
        profile_compatibility_id=compatibility_id,
        reference_target_sample_rate=native["reference_target_sample_rate"],
        reference_target_channel_count=native[
            "reference_target_channel_count"
        ],
        min_reference_frames_per_clip=native[
            "min_reference_frames_per_clip"
        ],
        max_reference_frames_per_clip=native[
            "max_reference_frames_per_clip"
        ],
        max_reference_total_frames=native["max_reference_total_frames"],
    )


def _concrete_profile_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if seed >= 2**64 - 1:
        if seed == 2**64 - 1:
            raise ValueError("Voice Profile seed must be concrete")
        raise OverflowError("seed does not fit uint64_t")
    return seed


class VoiceProfile:
    """Immutable runtime Voice conditioning bound to one Loaded Model."""

    __slots__ = ("_handle", "_lock", "_model")

    def __init__(self) -> None:
        raise TypeError("VoiceProfile objects are created by Model methods")

    @classmethod
    def _from_native(cls, model: Model, handle: object) -> VoiceProfile:
        instance = cls.__new__(cls)
        instance._handle = handle
        instance._lock = threading.RLock()
        instance._model = model
        return instance

    @property
    def model(self) -> Model:
        return self._model

    @property
    def closed(self) -> bool:
        return self._handle is None

    def _handle_for_call(self) -> object:
        with self._lock:
            if self._handle is None:
                raise SynthesizeError("Voice Profile is closed")
            return self._handle

    def serialize(
        self, *, diagnostics: DiagnosticCallback | None = None
    ) -> memoryview:
        """Serialize to a native-owned, read-only GGUF byte view."""
        with self._lock:
            if self._handle is None:
                raise SynthesizeError("Voice Profile is closed")
            native_diagnostics = _diagnostic_adapter(diagnostics)
            try:
                owner = _native.voice_profile_serialize(
                    self._handle, native_diagnostics
                )
            except RuntimeError as error:
                raise SynthesizeError(
                    f"could not serialize Voice Profile: {error}"
                ) from error
            return memoryview(owner).toreadonly()

    def close(self) -> None:
        with self._lock:
            handle = self._handle
            if handle is None:
                return
            self._handle = None
            _native.voice_profile_close(handle)

    def __enter__(self) -> Self:
        if self.closed:
            raise SynthesizeError("Voice Profile is closed")
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _references_for_native(
    references: Iterable[VoiceReference],
) -> tuple[tuple[object, int, int, str | None, str | None], ...]:
    result = tuple(references)
    if not result:
        raise ValueError("at least one VoiceReference is required")
    if any(not isinstance(reference, VoiceReference) for reference in result):
        raise TypeError("references must contain only VoiceReference values")
    return tuple(
        (
            reference.samples,
            reference.sample_rate,
            reference.channel_count,
            reference.transcript,
            reference.language,
        )
        for reference in result
    )
