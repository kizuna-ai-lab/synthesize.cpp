---
status: accepted
---

# Use CUDA 13 for the v1 CUDA Native Provider

v1 publishes one CUDA Native Provider named `cu13` for Linux x86-64, Linux
AArch64, and Windows x86-64. Its Release Toolchain Lock uses CUDA 13.3 Update 1 and
pins the runtime components distributed through Python as
`nvidia-cuda-runtime==13.3.29` and `nvidia-cublas==13.6.0.2`. The Provider package
is `synthesize-cpp-native-cu13`, selected through `synthesize-cpp[cu13]`.

On x86-64, the Provider's Runtime Support Floor is compute capability 7.5 or
later and an R580-or-newer NVIDIA driver. It therefore supports Turing and newer
architecture families, while Maxwell, Pascal, and Volta are outside the v1 CUDA
contract. On Linux AArch64, v1 supports DGX Spark/GB10 with compute capability
12.1 and an R580-or-newer driver; it does not claim Jetson support. The
common Windows build-toolchain decision remains Visual Studio 2022 17.14 / MSVC
19.44 and must be exercised with this exact CUDA release in CI. The existing
`manylinux_2_28` packaging floor remains unchanged for both Linux architectures.

The v1 Native Cubin Target Sets are explicit, release-pinned, and platform-specific:

```text
Linux x86-64 / Windows x86-64: sm_75;sm_80;sm_86;sm_89;sm_90;sm_100;sm_120a
Linux AArch64:                 sm_121a
```

The build must not derive official targets from `all` or `all-major`, whose
expansion can change with the compiler. A device is on the native path when the
driver selects a compatible cubin from its platform set; support never depends
on PTX JIT. The x86-64 Provider may additionally carry `compute_121a` PTX as an
experimental forward-compatibility path, but that path does not expand the
support claim. `sm_110` is intentionally absent because the AArch64 Provider
targets DGX Spark rather than Jetson; adding Jetson requires its own runtime and
physical-validation contract.

Release qualification runs complete TTS inference on physical hardware using
both the R580 driver floor and the current release-driver branch. At the time of
this decision that current branch is R610; each Release Toolchain Lock records
the exact driver builds used. Exact device models and target coverage are recorded
separately in the release's Validation Hardware Matrix.

## Consequences

- Package identity follows the CUDA ABI generation (`cu13`), while exact toolkit
  and component versions remain in the Release Toolchain Lock.
- The project gives up CUDA support for pre-Turing GPUs in the v1 Provider.
- CUDA 13.3, the pinned GGML revision, MSVC 19.44, every TTS graph, and every
  required operator must pass together before a CUDA artifact is published.
- If physical testing finds that a required feature cannot run on R580, the
  documented driver floor is raised; the implementation does not hide this with
  fallback.
- Supporting another native CUDA architecture requires an explicit target-set
  change, a new official build, and corresponding validation evidence.
- The Linux AArch64 wheel is built under the `manylinux_2_28` policy and must pass
  complete native CPU and CUDA inference on DGX Spark/GB10 before publication.
- Linux CUDA development and release builds use separate digest-pinned containers;
  the Ubuntu development image cannot serve as the manylinux release builder, and
  release validation installs the produced wheel in a clean runtime environment.
- Committed development presets compile only their named host architecture, while
  committed Release presets carry the complete platform target set and prohibit
  hardware-derived CUDA architecture selection.
- Standard development and Release presets use ordinary CUDA buffers. DGX Spark
  UVM experiments run only in an isolated research workflow, create no Provider or
  public C Interface, and cannot supply release-qualification evidence.

## Considered Options

- CUDA 12 was rejected for v1 because its broader legacy-GPU coverage would make
  the initial Provider start on an older toolkit generation. A future CUDA 12
  build, if justified, would be a separate Provider and support contract.
- Publishing both CUDA 12 and CUDA 13 Providers in v1 was rejected because it
  doubles packaging and physical-validation work and creates an equal-rank
  Provider-selection ambiguity.
- Using the current release-driver branch as the runtime floor was rejected because
  it would unnecessarily shorten the supported deployment window. The project
  instead validates the CUDA 13 compatibility floor and the current branch.
- `-arch=all` and `-arch=all-major` were rejected for official releases because
  their expansion is compiler-dependent and can silently change artifact size and
  support claims when the CUDA toolchain changes.
- Shipping only `sm_75` plus PTX was rejected because first launch would depend on
  driver JIT for newer major architectures and could not establish native support.
