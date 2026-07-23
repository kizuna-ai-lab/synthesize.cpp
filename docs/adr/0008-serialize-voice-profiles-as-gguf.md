---
status: accepted
---

# Serialize Voice Profiles as GGUF with explicit compatibility IDs

synthesize.cpp serializes Voice Profiles as self-contained little-endian GGUF v3 rather than introducing another binary container. A project format version, Model Family Profile Schema, exact 32-byte Profile Compatibility ID, and whole-file SHA-256 integrity value make family dispatch, evolution, corruption detection, and safe loading explicit. The compatibility ID is derived from a converter-defined canonical manifest of the profile representation and compatibility-critical source checkpoint material, while quantization and Execution Backend are excluded so validated packages of the same logical model can exchange canonical backend-independent profiles. Loaders never infer compatibility from filenames, architecture names, dimensions, `general.uuid`, or an entire quantized-file hash, and they validate untrusted structure and size arithmetic before allocating tensors or invoking a family implementation. This reuses GGUF's metadata and tensor machinery at the cost of defining a strict project namespace and requiring every profile-capable converter and Model Family to maintain its Profile Schema.
