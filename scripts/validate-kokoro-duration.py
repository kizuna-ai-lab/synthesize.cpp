#!/usr/bin/env python3
"""Measure the C++ Kokoro duration stage against its oracle."""

from __future__ import annotations

from kokoro_validation_common import main_wrapper


PROBES = (
    ('duration.logits', 'logits.f32', 'f32', False),
    ('duration.d', 'd.f32', 'f32', False),
    ('duration.pred_dur', 'pred_dur.i64', 'i64', True),
    ('duration.y_length', 'y_length.i64', 'i64', True),
    ('duration.alignment', 'alignment.f32', 'f32', True),
)


if __name__ == "__main__":
    main_wrapper('duration', 'duration', PROBES, __doc__ or "")
