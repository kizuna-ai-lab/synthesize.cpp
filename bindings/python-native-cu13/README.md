# synthesize-cpp-native-cu13

Opt-in CUDA 13 runtime Provider for the `synthesize-cpp` Python Adapter. It
contains a matching synthesize.cpp core with CPU and CUDA execution and no
public Python synthesis API.

CUDA runtime and cuBLAS are supplied by the exact NVIDIA Python dependencies
declared by this distribution. Its preparation hook loads those libraries by
absolute package path before the Provider core is opened. The host NVIDIA
driver is required and is never bundled.
