"""Structured diagnostics emitted by synchronous native operations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable


class DiagnosticLevel(IntEnum):
    WARNING = 0
    ERROR = 1


class Status(IntEnum):
    OK = 0
    INVALID_ARGUMENT = 1
    BAD_STRUCT_SIZE = 2
    FILE_NOT_FOUND = 3
    IO = 4
    GGUF = 5
    UNSUPPORTED_ARCH = 6
    UNSUPPORTED_VARIANT = 7
    UNSUPPORTED_INPUT = 8
    UNSUPPORTED_LANGUAGE = 9
    UNSUPPORTED_VOICE = 10
    UNSUPPORTED_CONTROL = 11
    MISSING_RESOURCE = 12
    TEXT_FRONTEND = 13
    INPUT_TOO_LONG = 14
    OUTPUT_LIMIT = 15
    OUT_OF_MEMORY = 16
    BACKEND = 17
    CANCELLED = 18
    SINK = 19
    INTERNAL = 20


@dataclass(frozen=True, slots=True)
class Diagnostic:
    level: DiagnosticLevel | int
    status: Status | int
    code: str
    message: str


DiagnosticCallback = Callable[[Diagnostic], object]


def _enum_or_int(enum_type: type[IntEnum], value: int) -> IntEnum | int:
    try:
        return enum_type(value)
    except ValueError:
        return value


def _diagnostic_adapter(
    callback: DiagnosticCallback | None,
) -> Callable[[dict], object] | None:
    if callback is None:
        return None
    if not callable(callback):
        raise TypeError("diagnostics must be callable or None")

    def emit(native: dict) -> object:
        diagnostic = Diagnostic(
            level=_enum_or_int(DiagnosticLevel, native["level"]),
            status=_enum_or_int(Status, native["status"]),
            code=native["code"],
            message=native["message"],
        )
        return callback(diagnostic)

    return emit
