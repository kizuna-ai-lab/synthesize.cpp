# Use a native GGML/GGUF runtime

synthesize.cpp uses native GGML graphs and GGUF model files as its only official inference runtime. PyTorch and ONNX may be used by conversion and reference-validation tooling, but the core library does not link ONNX Runtime, LibTorch, or llama.cpp as required runtime dependencies; optional third-party runtime adapters remain outside the core. This choice trades automatic model compatibility for a smaller dependency surface, controlled CPU/GPU execution, unified quantization, and explicit per-family correctness validation.
