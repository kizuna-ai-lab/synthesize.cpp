"""Owned zero-copy audio values and synchronous token synthesis adapter."""

from __future__ import annotations

import array
import math
from dataclasses import dataclass
from enum import IntFlag
from typing import Callable

from . import _native
from .errors import SynthesizeError
from .diagnostics import DiagnosticCallback
from .diagnostics import _diagnostic_adapter


class SynthesisResultFlags(IntFlag):
    SEED_USED = 1 << 0
    NATIVE_STREAMING_USED = 1 << 1


@dataclass(frozen=True, slots=True)
class Audio:
    samples: memoryview
    frame_count: int
    sample_rate: int
    channel_count: int


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    audio: Audio
    frames_emitted: int
    actual_seed: int | None
    flags: SynthesisResultFlags
    resolved_language_tag: str | None
    resolved_voice_id: str | None


def _token_buffer(tokens: object) -> object:
    try:
        view = memoryview(tokens)
    except TypeError:
        return array.array("i", tokens)  # type: ignore[arg-type]
    if (
        view.ndim == 1
        and view.c_contiguous
        and view.itemsize == 4
        and view.format in ("i", "@i", "=i")
    ):
        return tokens
    return array.array("i", view.tolist())


def _synthesize_tokens(
    context_handle: object,
    tokens: object,
    *,
    language: str | None,
    voice: str | None,
    seed: int | None,
    speaking_rate: float,
    max_output_frames: int | None,
    diagnostics: DiagnosticCallback | None,
    should_cancel: Callable[[], bool] | None,
    voice_profile: object | None = None,
) -> SynthesisResult:
    return _synthesize(
        context_handle,
        2,
        _token_buffer(tokens),
        "token IDs",
        language=language,
        voice=voice,
        seed=seed,
        speaking_rate=speaking_rate,
        max_output_frames=max_output_frames,
        diagnostics=diagnostics,
        should_cancel=should_cancel,
        voice_profile=voice_profile,
    )


def _synthesize_text(
    context_handle: object,
    text: str,
    *,
    language: str | None,
    voice: str | None,
    seed: int | None,
    speaking_rate: float,
    max_output_frames: int | None,
    diagnostics: DiagnosticCallback | None,
    should_cancel: Callable[[], bool] | None,
    voice_profile: object | None = None,
) -> SynthesisResult:
    return _synthesize_string(
        context_handle,
        0,
        text,
        "text",
        language=language,
        voice=voice,
        seed=seed,
        speaking_rate=speaking_rate,
        max_output_frames=max_output_frames,
        diagnostics=diagnostics,
        should_cancel=should_cancel,
        voice_profile=voice_profile,
    )


def _synthesize_phonemes(
    context_handle: object,
    phonemes: str,
    *,
    language: str | None,
    voice: str | None,
    seed: int | None,
    speaking_rate: float,
    max_output_frames: int | None,
    diagnostics: DiagnosticCallback | None,
    should_cancel: Callable[[], bool] | None,
    voice_profile: object | None = None,
) -> SynthesisResult:
    return _synthesize_string(
        context_handle,
        1,
        phonemes,
        "phonemes",
        language=language,
        voice=voice,
        seed=seed,
        speaking_rate=speaking_rate,
        max_output_frames=max_output_frames,
        diagnostics=diagnostics,
        should_cancel=should_cancel,
        voice_profile=voice_profile,
    )


def _synthesize_string(
    context_handle: object,
    input_kind: int,
    value: str,
    input_name: str,
    **options: object,
) -> SynthesisResult:
    if not isinstance(value, str):
        raise TypeError(f"{input_name} must be str")
    if not value:
        raise ValueError(f"{input_name} must not be empty")
    return _synthesize(
        context_handle, input_kind, value, input_name, **options
    )


def _synthesize(
    context_handle: object,
    input_kind: int,
    native_input: object,
    input_name: str,
    *,
    language: str | None,
    voice: str | None,
    seed: int | None,
    speaking_rate: float,
    max_output_frames: int | None,
    diagnostics: DiagnosticCallback | None,
    should_cancel: Callable[[], bool] | None,
    voice_profile: object | None,
) -> SynthesisResult:
    if language is not None and not isinstance(language, str):
        raise TypeError("language must be str or None")
    if voice is not None and not isinstance(voice, str):
        raise TypeError("voice must be str or None")
    if seed is None:
        native_seed = 2**64 - 1
    else:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer or None")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        if seed > 2**64 - 1:
            raise OverflowError("seed does not fit uint64_t")
        native_seed = seed
    if not isinstance(speaking_rate, (int, float)) or isinstance(
        speaking_rate, bool
    ):
        raise TypeError("speaking_rate must be a real number")
    speaking_rate = float(speaking_rate)
    if not math.isfinite(speaking_rate) or speaking_rate <= 0:
        raise ValueError("speaking_rate must be finite and positive")
    if max_output_frames is None:
        native_max_output_frames = 0
    else:
        if isinstance(max_output_frames, bool) or not isinstance(
            max_output_frames, int
        ):
            raise TypeError("max_output_frames must be an integer or None")
        if max_output_frames < 0:
            raise ValueError("max_output_frames must be non-negative")
        if max_output_frames > 2**64 - 1:
            raise OverflowError("max_output_frames does not fit uint64_t")
        native_max_output_frames = max_output_frames

    native_diagnostics = _diagnostic_adapter(diagnostics)
    if should_cancel is not None and not callable(should_cancel):
        raise TypeError("should_cancel must be callable or None")
    try:
        native = _native.synthesize(
            context_handle,
            input_kind,
            native_input,
            language,
            voice,
            native_seed,
            speaking_rate,
            native_max_output_frames,
            native_diagnostics,
            should_cancel,
            voice_profile,
        )
    except RuntimeError as error:
        raise SynthesizeError(f"could not synthesize {input_name}: {error}") from error

    flags = SynthesisResultFlags(native["result_flags"])
    samples = memoryview(native["audio_owner"]).toreadonly()
    audio = Audio(
        samples=samples,
        frame_count=native["frame_count"],
        sample_rate=native["sample_rate"],
        channel_count=native["channel_count"],
    )
    return SynthesisResult(
        audio=audio,
        frames_emitted=native["frames_emitted"],
        actual_seed=(
            native["actual_seed"]
            if flags & SynthesisResultFlags.SEED_USED
            else None
        ),
        flags=flags,
        resolved_language_tag=native["resolved_language_tag"],
        resolved_voice_id=native["resolved_voice_id"],
    )
