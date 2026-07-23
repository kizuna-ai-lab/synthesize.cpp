---
status: accepted
---

# Share model graphs across execution backends

synthesize.cpp uses the CPU execution path as its correctness baseline and runs the same model-family GGML graph through CPU, CUDA, Metal, Vulkan, and future backends. Backend-specific kernels and placement remain below the family layer rather than creating separate VITS or other model implementations. A model-family and backend combination is supported only after it passes the applicable Port Validation Suite correctness, deterministic waveform, stability, placement, and operational-measurement gates; compilation or partial operator coverage is insufficient. Corpus-scale and perceptual quality evidence is a separate Validation Level under ADR 0017.
