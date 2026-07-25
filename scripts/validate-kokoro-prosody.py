#!/usr/bin/env python3
"""Measure the C++ Kokoro prosody stage against its oracle."""

from __future__ import annotations

from kokoro_validation_common import main_wrapper


PROBES = (
    ('prosody.en', 'en.f32', 'f32', False),
    ('prosody.f0', 'f0.f32', 'f32', False),
    ('prosody.n', 'n.f32', 'f32', False),
)


if __name__ == "__main__":
    main_wrapper('prosody', 'prosody', PROBES, __doc__ or "")
