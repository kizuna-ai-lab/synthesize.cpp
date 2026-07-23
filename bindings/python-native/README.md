# synthesize-cpp-native

Platform-native runtime Provider for the `synthesize-cpp` Python Adapter. This
distribution contains no public synthesis API; it carries one matching shared
`synthesize.cpp` runtime, its private GGML libraries, and a build-stamped
Provider contract. Applications install and import `synthesize-cpp`, which
discovers this package through the internal `synthesize_cpp.native` entry point.

The default Linux and Windows artifact is CPU-only. The default macOS ARM64
artifact contains CPU and Metal. CUDA is supplied by the separate
`synthesize-cpp-native-cu13` Provider.
