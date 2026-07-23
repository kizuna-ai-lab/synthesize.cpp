# Native Library Packaging

Status: Confirmed, last updated on 2026-07-22.

## Build Contract

The native C library is the mandatory build product. The command-line reference
Adapter is an optional target and links only through the same installed C
Interface; it is never required to build, install, or embed the library and
receives no private inference path.

`SYNTH_BUILD_CLI` controls that Adapter and defaults to `ON` for source-tree
development. SDK, Rust, and Python native builds set it to `OFF` when they need
only the library. The executable target is `synthesize-cli`.

`SYNTH_BUILD_SHARED` selects one linkage form per build tree and drives the
matching `BUILD_SHARED_LIBS` value for the vendored GGML build.
Static and shared distributions are produced by separate builds so both publish
the same logical `synthesize::synthesize` target without platform-specific target
selection or a Windows static/import-library name collision. A shared build
propagates `SYNTH_SHARED` to consumers and privately defines
`SYNTH_BUILDING_LIBRARY`; a static build defines neither.

## Installed C SDK

An installation contains:

- `include/synthesize.h` as the canonical C declaration source;
- the selected static or shared native library;
- a relocatable CMake package exporting `synthesize::synthesize`;
- a generated `synthesize.pc` for non-CMake consumers; and
- optional dynamic backend modules under
  `${CMAKE_INSTALL_LIBDIR}/synthesize/backends/`.

The CMake package and pkg-config metadata carry required include paths, link
flags, and shared-build definitions without embedding build-tree paths. The
CLI is a separate optional installed program, not a dependency of any SDK artifact.

Implementation checkpoint (2026-07-22): static and shared installations now
provide this complete SDK surface. The installed-package test stages the current
build under an isolated prefix, then builds and runs two pure-C consumers: one
through `find_package(synthesize CONFIG)` and one through `pkg-config`. Static
CUDA metadata supplies the GGML, CUDA runtime, cuBLAS, driver, C++ runtime, and
platform libraries needed to consume the archives. Shared Linux installations
use the ABI SONAME and `$ORIGIN` RUNPATH described below, while CUDA runtime
libraries remain external runtime dependencies rather than copied into the SDK.
The CUDA packaging test supplies the selected runtime directory to the loader
and verifies that CUDA 13.3 `cudart` and `cublas` are actually resolved from it.

## Toolchain, Runtime, and Validation Contracts

Build and support requirements are recorded as four separate contracts:

1. The **Minimum Build Toolchain** is the source-build floor enforced and
   documented by the project.
2. The **Release Toolchain Lock** records the exact tools, dependency versions,
   upstream GGML revision, and base-image or SDK identities used to construct one
   official release. It may be newer than the Minimum Build Toolchain.
3. The **Runtime Support Floor** states what an installed Native Provider needs
   from its host, such as an operating-system version, driver, GPU API, and device
   capability. It never requires a compiler or build SDK merely to run a wheel or
   installed native package.
4. The **Validation Hardware Matrix** records the physical devices, operating
   systems, drivers, model variants, and execution modes that supplied a release's
   end-to-end support evidence. It is not a device allowlist: an unlisted device
   that satisfies the Runtime Support Floor may run, but remains unvalidated.

Release artifacts retain or reference a machine-readable build manifest containing
their Release Toolchain Lock and upstream provenance. The public C Interface does
not acquire compiler-, SDK-, or packaging-specific configuration as a result;
callers continue to select Execution Backends and receive project-owned runtime
diagnostics through the established seam.

## Common Minimum Build Toolchain

All supported v1 source-build configurations share this floor before applying
backend-specific SDK requirements:

| Build dimension | Minimum supported value |
| --- | --- |
| Build system | CMake 3.24 |
| C language | C11 with compiler extensions disabled |
| C++ language | C++17 with compiler extensions disabled |
| Linux x86-64 and AArch64 | GCC 11 or Clang 15 |
| Windows x64 | Visual Studio 2022 17.14 / MSVC 19.44 |
| macOS ARM64 | Xcode 15.4 / AppleClang 15 |

Ninja is optional for ordinary source builds; supported native CMake generators
remain usable. Official release builds use Ninja where applicable and record its
exact version in the Release Toolchain Lock. v1 does not claim MinGW as a
supported Windows build toolchain. Older tools may happen to compile a subset of
the project, but they are outside the support contract.

CUDA, Metal, and Vulkan add their own compiler, SDK, shader-language, and host
requirements without weakening this common floor. The project's minimum-version
CI lanes exercise the floor, while official release lanes may use newer tools
whose exact identities are locked in the release manifest.

## Primary Development Hosts

DGX Spark is the primary development host. It exercises Linux AArch64, CPU,
CUDA, native `sm_121a`, and unified CPU/GPU memory during ordinary development.
Its preinstalled compiler and CUDA versions do not define official artifacts;
release builds still use the Release Toolchain Lock and build Linux wheels under
the `manylinux_2_28` policy.

The Linux RTX 4070 SUPER host is the secondary x86-64 and discrete-memory CUDA
path and may boot Windows 11 for `win_amd64` Provider validation. The Mac mini M4
is the native macOS ARM64 CPU and Metal validation host. Keeping these paths in
the validation workflow prevents the Implementation from accidentally depending
on DGX Spark's ARM CPU or unified-memory behavior; none of these host identities
crosses the public C Interface.

## v1 Metal Toolchain, Runtime, and Validation Contract

v1 publishes Metal only inside the `default` Native Provider for
`macosx_12_0_arm64`. The Provider contains both CPU and Metal Execution Backends;
there is no separate Metal distribution, macOS x86-64 build, or `universal2`
build. The compiled Metal library is embedded in the Provider artifact, while
Foundation, Metal, MetalKit, and other operating-system frameworks remain host
dependencies.

The Metal-specific Minimum Build Toolchain is Xcode 15.4, AppleClang 15, the
macOS 14.5 SDK bundled with that Xcode release, and its `metal` and `metallib`
tools. Official builds set the macOS and Metal deployment target to 12.0. Xcode's
build-host operating-system requirement is not the Runtime Support Floor, and an
installed Provider never requires Xcode, an SDK, or a shader compiler. The
Release Toolchain Lock records the exact Xcode build, SDK build, upstream GGML
revision, CMake inputs, deployment target, and Metal compilation options; a
marketing-level label such as "Metal 3" is not a separate compatibility contract.

The Metal Runtime Support Floor is macOS 12.0 on Apple silicon with an available
Metal device. Backend initialization still performs runtime capability checks and
returns project-owned diagnostics when the host cannot execute the required
kernels. A device that satisfies this floor may run even when it is not in the
Validation Hardware Matrix, but remains unvalidated.

Every published macOS Provider artifact is installed into a clean runtime
environment on the Mac mini M4 and runs the required CPU and Metal end-to-end
release suite. The release manifest records the exact Mac model, SoC, operating-
system build, locked toolchain, model variants, Quantization Profiles, and
execution modes. The M4 is validation evidence rather than a device allowlist;
earlier Apple-silicon generations are not implicitly represented by that run.
If the M4 host is unavailable or the suite fails, the macOS Provider artifact is
withheld rather than published with an untested Metal claim. v1 does not require
another Mac solely to enlarge the physical matrix.

## v1 CUDA Validation Hardware Matrix

v1 requires no CUDA device beyond the two physical NVIDIA systems already
available. The RTX 4070 SUPER contributes two host combinations by booting two
operating systems; it is not counted as two devices.

| Published Provider artifact | Physical validation host | Native path | Cadence |
| --- | --- | --- | --- |
| Linux AArch64 `cu13` | DGX Spark/GB10, Linux AArch64 | `sm_121a` | Every artifact publication |
| Linux x86-64 `cu13` | RTX 4070 SUPER, Linux x86-64 | `sm_89` | Every artifact publication |
| Windows `win_amd64` `cu13` | The same RTX 4070 SUPER, Windows 11 x64 | `sm_89` | Every artifact publication |

Each run installs the produced artifact into a clean runtime environment and runs
the required CPU and CUDA end-to-end release suite. The release manifest records
the exact device, operating-system build, NVIDIA driver, model variants,
Quantization Profiles, and execution modes. Driver-floor and current-driver runs
may reuse these physical systems after an explicit driver change; they do not imply
additional hardware.

The wider x86-64 Native Cubin Target Set and Runtime Support Floor do not expand
this physical matrix. An unlisted device or architecture may run when it satisfies
the Runtime Support Floor, but remains explicitly unvalidated even when its cubin
is present. Compilation, emulation, device enumeration, or a community success
report alone is not official validation evidence.

If a required physical host is unavailable or fails the release suite, the
corresponding CUDA Provider artifact is withheld rather than published with an
untested claim; other platform artifacts may continue independently. Community
hardware may extend a future matrix only after the project can reproduce the full
release procedure and retain its provenance. v1 does not require purchasing or
borrowing another CUDA device.

## v1 Vulkan Toolchain, Runtime, and Validation Contract

v1 publishes the `vulkan` Native Provider only for
`manylinux_2_28_x86_64` and `win_amd64`. Each Provider contains CPU and Vulkan
Execution Backends with its compiled SPIR-V embedded. Linux AArch64 and macOS do
not receive a v1 Vulkan Provider. The artifact does not distribute a Vulkan
loader, ICD, or GPU driver.

The Vulkan-specific Minimum Build Toolchain is LunarG Vulkan SDK 1.3.283.0, or
on Linux an equivalent set of versioned Vulkan headers and loader development
files, `glslc`, and SPIRV-Headers. Shader compilation targets the Vulkan 1.3
environment. These are source-build requirements rather than host runtime
dependencies: an installed Provider does not invoke `glslc` or compile shaders.
The Release Toolchain Lock records the exact SDK or component versions, shader
compiler identity, options, embedded SPIR-V inventory, and upstream GGML revision.

The Vulkan Runtime Support Floor is a Vulkan 1.2 loader and compute-capable
physical device that exposes Vulkan 1.2 plus the baseline features accepted by
the runtime capability probe, including 16-bit storage. Shader float16 and int8,
integer dot product, cooperative matrix, bfloat16, and later vendor or Khronos
extensions are optional acceleration capabilities rather than Provider-wide
requirements. Missing optional features disable their optimized paths unless a
future Model Variant explicitly declares and validates such a feature as required.
The Vulkan 1.3 shader compilation environment is therefore not the Vulkan Runtime
Support Floor.

The v1 Vulkan Validation Hardware Matrix reuses the one available discrete GPU:

| Published Provider artifact | Physical validation host | Native path | Cadence |
| --- | --- | --- | --- |
| Linux x86-64 `vulkan` | RTX 4070 SUPER, Linux x86-64 | NVIDIA Vulkan driver | Every artifact publication |
| Windows `win_amd64` `vulkan` | The same RTX 4070 SUPER, Windows 11 x64 | NVIDIA Vulkan driver | Every artifact publication |

Each run installs the produced artifact into a clean runtime environment and runs
the required CPU and Vulkan end-to-end release suite. The release manifest records
the exact GPU, operating-system build, Vulkan loader, ICD and driver, reported API
and feature set, model variants, Quantization Profiles, and execution modes.

This matrix validates v1 Vulkan execution on NVIDIA only. AMD and Intel devices
that satisfy the Runtime Support Floor may run but remain explicitly unvalidated;
the project makes no correctness or performance claim for them until the complete
release procedure is reproduced on physical hardware. If either required host
combination is unavailable or fails, only that platform's Vulkan Provider artifact
is withheld. CPU, CUDA, Metal, and other platform artifacts remain independent,
and v1 does not require purchasing another GPU to enlarge the matrix.

## Containerized Linux CUDA Workflow

Linux CUDA development runs in pinned containers instead of replacing or
reusing a host CUDA Toolkit. On DGX Spark, the development image is the ARM64
variant of `nvidia/cuda:13.3.0-devel-ubuntu24.04`, pinned by immutable digest in
the Release Toolchain Lock. NVIDIA Container Runtime supplies only the host
driver and GPU devices required at runtime; the container owns the compiler,
headers, user-space CUDA libraries, and other build tools.

The source tree, model cache, and compiler or build caches are mounted from
explicit host directories. Models, credentials, and generated release artifacts
are not baked into the development image. The image tag is a human-readable
label; reproducible automation resolves and records its platform-specific digest.

Official Linux wheels use separate, digest-pinned `manylinux_2_28` build images
for x86-64 and AArch64 with the Release Toolchain Lock's exact CUDA 13.3
toolchain. The Ubuntu development image is never used as evidence that a wheel
satisfies the manylinux policy. After building and policy repair, each wheel is
installed from the produced artifact into a clean runtime environment and runs
the complete native validation suite on physical target hardware. Tests do not
load libraries or Python packages from the build tree.

The implemented release builder is
`scripts/ci/build_manylinux_provider.py`. Its pinned image digests, native host
architecture check, Provider-owned `build/manylinux/<provider>/<arch>` CMake
trees, read-only source/toolkit mounts, and wheel contract inspection are unit
tested. It uses the image's glibc 2.28 and GCC toolchain with the mounted exact
CUDA 13.3.73 toolkit; scikit-build-core 1.0.3 and Ninja 1.13.0 are exact direct
build requirements. A raw host-tagged wheel is never the publication artifact:
only the repaired output under `wheelhouse/<provider>/<arch>` proceeds to clean
runtime validation.

The x86-64 Provider build uses GGML's runtime-selected CPU variants. When
`SYNTH_GGML_BACKEND_DL=ON` on x86-64, `SYNTH_CPU_RUNTIME_DISPATCH` defaults to
`ON` and produces modules from the compiler's x86-64 baseline through SSE4.2,
Sandy Bridge, Haswell, AVX-512, Zen 4, Alder Lake, and Sapphire Rapids variants.
The package-local loader probes their compatibility scores and registers only
the highest-scoring compatible module. The baseline `libggml-cpu-x64.so` is
compiled without a host-native `-march`, so the core Provider is no longer
accidentally pinned to the release machine's AVX2 feature set. AArch64 remains a
single architecture-specific CPU module. `SYNTH_CPU_RUNTIME_DISPATCH=OFF`
retains the single-module x86 build for explicit non-release experiments. This
CPU gate remains independent of the CUDA cubin set.

This container contract applies to Linux CUDA development and publication.
Windows CUDA and macOS Metal artifacts continue to use their confirmed native
platform toolchains and validation hosts.

## CMake Preset Contract

The repository commits `CMakePresets.json` as the supported build entry point.
Presets own platform, build type, backend set,
Native Cubin Target Set, tests, optional CLI, and isolated binary directory so a
developer or release job does not reconstruct those combinations from ad hoc
command-line flags.

The initial CUDA presets are:

| Preset | Host and purpose | CUDA targets | Build mode |
| --- | --- | --- | --- |
| `dev-dgx-spark` | Linux AArch64 daily CPU/CUDA development | `sm_121a` | `RelWithDebInfo`, tests and CLI enabled |
| `dev-dgx-spark-uvm` | Linux AArch64 isolated UVM research | `sm_121a` | `RelWithDebInfo`, dedicated tests only |
| `release-linux-aarch64-cu13` | Linux AArch64 Provider release | `sm_121a` | `Release`, locked packaging inputs |
| `dev-linux-x86_64-cuda` | Linux x86-64 RTX 4070 SUPER development | `sm_89` | `RelWithDebInfo`, tests and CLI enabled |
| `release-linux-x86_64-cu13` | Linux x86-64 Provider release | `sm_75;sm_80;sm_86;sm_89;sm_90;sm_100;sm_120a` | `Release`, locked packaging inputs |

CPU-only, macOS Metal, and Vulkan configurations have separate development and
release presets rather than optional flags layered onto a CUDA preset. Developer
presets compile only the architecture needed by their named host to shorten the
edit-build-test loop. Release presets use the complete confirmed platform set
and reject architecture auto-detection, `native`, `all`, and `all-major`.

`CMakeUserPresets.json` may contain uncommitted local inheritance and cache
overrides, including compiler-cache paths, but official automation never reads it.
Every committed preset uses its own `build/<preset-name>` binary directory.
Official release jobs name a committed Release preset and may not reproduce it
with an equivalent-looking list of `-D` arguments.

`dev-dgx-spark` uses the ordinary CUDA allocation path. The
`dev-dgx-spark-uvm` workflow inherits its architecture and toolchain choices but
is experimental, launches only isolated test processes with UVM enabled before
backend initialization, and is never used for release qualification. Release
automation rejects UVM-specific configuration. No committed preset turns UVM
into a public build option or application-facing runtime contract.

The implemented Release build target
`synthesize-check-release-cubins` inspects the built GGML CUDA binary with
the selected CUDA toolkit's `cuobjdump`. It requires the exact platform cubin
set and rejects embedded PTX. The physical Linux AArch64 and x86-64 builds pass
this gate with `sm_121a` and
`sm_75;sm_80;sm_86;sm_89;sm_90;sm_100;sm_120a`, respectively. These are native
source-tree Release artifacts; manylinux policy and clean-runtime wheel
qualification remain separate publication gates.

## Binary Names and ABI Versioning

Static builds install the platform's ordinary unversioned archive. Shared builds
use the independent `SYNTH_ABI_VERSION` for loader compatibility names: Linux
installs the `libsynthesize.so.1` SONAME family, macOS installs the
`libsynthesize.1.dylib` compatibility family, and Windows installs
`synthesize-1.dll` plus its import library. Semantic release versions may appear
in fully versioned files and package metadata, but they do not replace the ABI
value in the SONAME, install name, or DLL name.

Changing the semantic major version without changing the C ABI retains the
binary compatibility name. Changing `SYNTH_ABI_VERSION` creates a new
compatibility name that may coexist with the previous ABI where the platform
permits it.

## Backend Module Discovery

Dynamic GGML backend modules are private runtime artifacts, not additional public
libraries. `synth_backend_load_default()` resolves the loaded shared core
library's location and scans only its fixed relative `synthesize/backends/`
directory. This makes a complete installation relocatable as one tree. It does
not scan the process working directory, a Model Package directory, `PATH`, or
arbitrary system library paths.

`synth_backend_load_from_dir()` remains the explicit application-controlled
override for another artifact directory. In a static build, the default loader
is a successful no-op and statically linked backends are registered by the build.
