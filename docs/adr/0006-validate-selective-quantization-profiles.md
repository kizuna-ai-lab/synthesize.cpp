---
status: accepted
---

# Validate selective quantization profiles

synthesize.cpp establishes F32 reference parity, validates an F16 production baseline, and then introduces named mixed Quantization Profiles that assign storage and compute precision by tensor group. Numerically sensitive and small tensors remain F32 or F16 unless targeted tests justify lower precision, and each profile is distributed and passes the Port Validation Suite as an independent Model Package for every claimed backend. This trades maximum compression for reproducible functional behavior and truthful backend support. Perceptual quality and quantization transparency require the separate Quality Evaluation Suite and `quality_evaluated` Validation Level defined by ADR 0017.
