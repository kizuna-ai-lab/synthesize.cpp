---
status: accepted
---

# Pin Reference Audio normalization behind a private Module

synthesize.cpp centralizes Reference Audio validation, channel conversion, and sample-rate conversion in a private CPU Audio Normalizer Module owned by the Voice Profile Module. Validated builds vendor and statically compile libsamplerate 0.2.2 with `SRC_SINC_BEST_QUALITY`, use its stateful full Interface, drain finite input, and never expose dependency types or quality choices through the public C Interface. Pinning the source, configuration, and processing order avoids system-library and caller-selected behavior changing the validated Voice Profile preparation path; matching target-format PCM can still be borrowed without an extra PCM copy, while conversions use internal storage. Numerical equivalence is validated across supported hosts, but floating-point normalization is not promised to be bit-identical across architectures.

Reference Audio limits use checked Reference Frame Equivalents equal to `ceil(input_frames * target_rate / input_rate)`. This value measures input duration at the target rate for preflight and allocation safety; it does not require the resampler to emit exactly that many frames. The normalizer retains the actual drained output rather than padding or cropping it to force the equivalent length.

## Considered Options

- A system libsamplerate was rejected because an embedding application could silently select a different implementation version than the one validated with the Model Package.
- A public resampler or quality selector was rejected because it would create multiple Voice Profile preparation semantics behind one capability declaration.
- miniaudio's built-in linear resampler was rejected for this quality-sensitive enrollment path, and importing its wider playback stack would not deepen the Audio Normalizer Module.
- SpeexDSP remains a reasonable speech-oriented alternative, but exposing or retaining a second Adapter would create an unused seam in v1.
- A custom sinc implementation was rejected initially because its DSP maintenance and validation cost would not add public leverage.

Vendoring must preserve libsamplerate's BSD-2-Clause notice in the project's third-party license material.
