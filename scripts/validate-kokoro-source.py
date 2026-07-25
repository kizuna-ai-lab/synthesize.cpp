#!/usr/bin/env python3
"""Measure the C++ Kokoro harmonic source against its oracle."""

from __future__ import annotations

from kokoro_validation_common import main_wrapper


PROBES = (
    ('source.har', 'har.f32', 'spectrum', False),
)


if __name__ == "__main__":
    main_wrapper('source', 'source', PROBES, __doc__ or "")
