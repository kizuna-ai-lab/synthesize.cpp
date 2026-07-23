---
status: accepted
---

# Use a stable C interface with opaque handles

synthesize.cpp exposes a versioned C ABI with an immutable, shareable Loaded Model handle, a mutable Synthesis Context handle used by one concurrent synthesis at a time, and an immutable, model-bound Voice Profile handle for reusing prepared runtime Voice conditioning. Preset Voices remain immutable Model Package metadata rather than runtime handles. The CLI and future Rust and Python bindings are adapters over the same exported C Interface and receive no alternate inference path. ABI-visible status, enum-like, flag, count, and boolean values use fixed-width integer representations rather than compiler-sized enums or `bool`. Audio is emitted through one Audio Sink seam with callback and complete-buffer adapters, while GGML and model-family types remain private. Chunked delivery is deliberately distinct from validated Native Streaming Synthesis, keeping a small stable interface without making false latency or universal zero-copy guarantees.
