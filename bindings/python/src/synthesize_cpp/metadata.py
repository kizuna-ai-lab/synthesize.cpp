"""Immutable Loaded Model capability and catalog values."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag


class InputFlags(IntFlag):
    TEXT_UTF8 = 1 << 0
    PHONEMES_UTF8 = 1 << 1
    TOKEN_IDS = 1 << 2


class ModelCapabilityFlags(IntFlag):
    SPEAKING_RATE = 1 << 0
    STOCHASTIC = 1 << 1
    NATIVE_STREAMING = 1 << 2


class PresetVoiceFlags(IntFlag):
    DEFAULT = 1 << 0


class LanguageFlags(IntFlag):
    DEFAULT = 1 << 0
    REGIONAL_FALLBACK = 1 << 1


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    input_flags: InputFlags
    capability_flags: ModelCapabilityFlags
    output_sample_rate: int
    output_channel_count: int
    min_speaking_rate: float
    max_speaking_rate: float
    max_input_tokens: int
    max_output_frames: int


@dataclass(frozen=True, slots=True)
class PresetVoice:
    id: str
    display_name: str | None
    flags: PresetVoiceFlags


@dataclass(frozen=True, slots=True)
class LanguageCapability:
    tag: str
    flags: LanguageFlags


def _capabilities_from_native(native: dict) -> ModelCapabilities:
    return ModelCapabilities(
        input_flags=InputFlags(native["input_flags"]),
        capability_flags=ModelCapabilityFlags(native["capability_flags"]),
        output_sample_rate=native["output_sample_rate"],
        output_channel_count=native["output_channel_count"],
        min_speaking_rate=native["min_speaking_rate"],
        max_speaking_rate=native["max_speaking_rate"],
        max_input_tokens=native["max_input_tokens"],
        max_output_frames=native["max_output_frames"],
    )


def _preset_voices_from_native(native: tuple[dict, ...]) -> tuple[PresetVoice, ...]:
    return tuple(
        PresetVoice(
            id=voice["id"],
            display_name=voice["display_name"],
            flags=PresetVoiceFlags(voice["flags"]),
        )
        for voice in native
    )


def _languages_from_native(
    native: tuple[dict, ...],
) -> tuple[LanguageCapability, ...]:
    return tuple(
        LanguageCapability(
            tag=language["tag"],
            flags=LanguageFlags(language["flags"]),
        )
        for language in native
    )
