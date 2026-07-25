#!/usr/bin/env python3
"""Measure the C++ Kokoro PL-BERT stage against its oracle."""

from __future__ import annotations

from kokoro_validation_common import main_wrapper


PROBES = (
    ('bert.hidden', 'hidden.f32', 'f32', False),
    ('text.d_en', 'd_en.f32', 'f32', False),
)


if __name__ == "__main__":
    main_wrapper('plbert', 'plbert', PROBES, __doc__ or "")
