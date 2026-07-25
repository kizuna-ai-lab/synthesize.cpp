#!/usr/bin/env python3
"""Measure the C++ Kokoro decoder and generator against their oracle."""

from __future__ import annotations

from kokoro_validation_common import main_wrapper


PROBES = (
    ('decoder.spectrum', 'spectrum.f32', 'log_spectrum', False),
)


if __name__ == "__main__":
    main_wrapper('decoder', 'decoder', PROBES, __doc__ or "")
