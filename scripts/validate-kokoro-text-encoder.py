#!/usr/bin/env python3
"""Measure the C++ Kokoro acoustic text encoder against its oracle."""

from __future__ import annotations

from kokoro_validation_common import main_wrapper


PROBES = (
    ('text.t_en', 't_en.f32', 'f32', False),
    ('text.asr', 'asr.f32', 'f32', False),
)


if __name__ == "__main__":
    main_wrapper('text-encoder', 'text-encoder', PROBES, __doc__ or "")
