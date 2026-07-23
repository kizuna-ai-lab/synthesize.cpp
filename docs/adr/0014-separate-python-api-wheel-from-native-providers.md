---
status: accepted
---

# Separate the Python API wheel from Native Provider distributions

synthesize.cpp publishes `synthesize-cpp` as its only public Python API
distribution and `synthesize_cpp` as its only import package. The API
distribution depends by default on the implementation distribution
`synthesize-cpp-native`. Accelerator variants are requested as extras on the API
distribution; `synthesize-cpp[cu13]` installs
`synthesize-cpp-native-cu13` alongside the default provider. The CUDA version
choice and its runtime floor are recorded separately in ADR 0016.

A Native Provider is a complete packaging and deployment unit containing one
compatible native core, CPU execution, its advertised optional accelerator
backends, required runtime libraries, and provider metadata. An Execution Backend
is a runtime compute implementation such as CPU, CUDA, Metal, or Vulkan. Because
one provider may carry more than one backend, the two terms are deliberately
distinct. `backend` remains part of runtime C and
binding APIs, but it is not used in public provider distribution names.

For every official release `X.Y.Z`, the API distribution requires
`synthesize-cpp-native==X.Y.Z.*` and each accelerator extra requires its
corresponding provider with the same wildcard pin, such as
`synthesize-cpp-native-cu13==X.Y.Z.*`. This admits packaging-only `.postN`
repairs but no different base release. Before `dlopen`, the selected provider's
declared base release and generated public-header hash must match the binding.
Mismatch is a hard error. C ABI, minimum-version, and required-symbol checks still
run after loading.

Providers register descriptors through the internal `synthesize_cpp.native`
entry-point group. The Python Adapter validates every discovered descriptor and
selects exactly one Provider for the process. A constrained
`SYNTHESIZE_NATIVE_PROVIDER` value may name an installed stable Provider ID but
may not specify a path. Otherwise, the Adapter ranks advertised backends as CUDA,
Metal, Vulkan, CPU_ACCEL, then CPU. A tie at the highest rank is a configuration
error rather than an order-dependent choice.

A stale, malformed, wrong-release, or wrong-header Provider is a hard
configuration error. Failure while preparing, loading, validating, or registering
the selected Provider is also final and does not fall through to a lower-ranked
Provider. After it loads, per-model `backend=AUTO` remains a separate native
decision and may choose the Provider's mandatory CPU path. No accelerator-specific
import or `activate()` operation is exposed.

The default Linux and Windows Provider contains CPU only. The default macOS
Provider also contains Metal with its Metal library embedded, and relies on
operating-system frameworks rather than copying them. The optional
`synthesize-cpp[vulkan]` extra installs a complete CPU-plus-Vulkan Provider with
embedded SPIR-V; it does not distribute the host Vulkan loader, ICD, or driver.

The optional `synthesize-cpp[cu13]` Provider contains CPU plus CUDA. It depends
on `nvidia-cuda-runtime==13.3.29` and `nvidia-cublas==13.6.0.2`, and its
preparation hook preloads those exact release-tested libraries from
their installed package paths without changing `PATH` or `LD_LIBRARY_PATH`.
The host NVIDIA driver is never distributed. The Provider's native artifacts do
not link or initialize NCCL, cuDNN, NVRTC, or CUPTI. The locked
`nvidia-cublas==13.6.0.2` Python distribution nevertheless declares
`nvidia-cuda-nvrtc` as an upstream transitive dependency; its installed presence
is inventoried and license-reviewed but does not make NVRTC a Provider feature.

Every Provider carries its matching synthesize.cpp core, CPU execution, private
GGML runtime artifacts, backend modules in the fixed relative directory,
descriptor metadata, licenses, and notices or SBOM. Providers contain no Model
Package, Voice data, or Text Frontend resources. Project-owned and required
non-system shared libraries may be carried after license review, but operating
system frameworks and GPU drivers remain host dependencies.

Provider packages do not introduce another synthesis API or require users to
import an accelerator-specific module. The API binding loads native artifacts
only through a controlled provider contract and performs the same ABI,
minimum-version, and required-symbol validation before using them.

## Considered Options

- Bundling the CPU core directly inside the API wheel was rejected because it
  couples the stable Python interface distribution to one native artifact variant
  and makes additive accelerator providers harder to model. This supersedes ADR
  0013 while preserving its Limited-API extension and controlled-loading rules.
- Public distributions named `synthesize-cpp-backend-cuda13` were rejected
  because they expose an internal execution term at the distribution seam and
  incorrectly imply that one package always equals one backend.
- Accelerator-specific public imports were rejected because all providers must
  serve the same `synthesize_cpp` API and public C Interface.
- Combining a core from one Provider with backend modules from another was
  rejected for official Python wheels because it makes cross-distribution loader
  state, runtime-library ownership, and atomic compatibility validation part of
  the Adapter Interface.
- Entry-point or package-name ordering for equal-ranked Providers was rejected
  because installation order is not a meaningful execution policy.
- Falling through to CPU or another Provider after a stale or broken accelerator
  Provider was rejected because it hides packaging failures and can silently
  violate an application's acceleration expectation.
- Including Vulkan in the default Linux and Windows Provider was rejected because
  the already-confirmed no-fallback policy would make a CPU installation fail to
  load on a host without a usable Vulkan loader.
- Copying NVIDIA or Vulkan drivers into a wheel was rejected because drivers are
  host-managed, hardware-specific dependencies rather than Provider artifacts.
- Linking or initializing optional CUDA libraries such as NCCL, cuDNN, NVRTC, or
  CUPTI in v1 was rejected because the current runtime does not require them and
  the public Interface exposes no multi-GPU policy that would justify NCCL. An
  upstream Python package's transitive installation dependency is tracked
  separately from this native runtime decision.

No public C Interface change results from this distribution-layer distinction.
