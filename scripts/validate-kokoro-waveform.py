#!/usr/bin/env python3
"""Measure the C++ Kokoro waveform against its oracle."""

from __future__ import annotations

from kokoro_validation_common import main_wrapper


PROBES = (
    ('audio.pcm', 'pcm.f32', 'waveform', False),
)


if __name__ == "__main__":
    main_wrapper('waveform', 'waveform', PROBES, __doc__ or "")
