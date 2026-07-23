---
status: accepted
---

# Package models as one GGUF plus declarative sidecars

Each synthesize.cpp model variant uses one primary GGUF containing all synthesis-graph tensors and immutable inference metadata. Large or separately licensed frontend and attribution resources may remain as checksum-pinned, non-executable sidecars, while Text Frontend Provider code is installed separately. Model loading never executes package content or implicitly accesses the network, trading single-file purity for explicit licensing, inspectability, and a safer loading boundary.
