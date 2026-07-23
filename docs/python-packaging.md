# Python Binding Packaging

Status: Python API and Native Provider packaging confirmed, last updated on 2026-07-23.

## Distribution and Interpreter Contract

The distribution name is `synthesize-cpp` and its import package is
`synthesize_cpp`. It supports CPython 3.11 and later through one thin C extension
compiled with `Py_LIMITED_API=0x030B0000` and published with the `abi3` tag. The
extension uses only the CPython Limited API and the public synthesize.cpp C
Interface; it does not bind C++ implementation types or use pybind11.

Implementation checkpoint (2026-07-23): the Adapter discovers and prepares one
Provider at import, checks ABI 1, release 0.1.0, and all 46 exported C symbols,
and exposes device discovery; owned Model, Context, and Voice Profile lifecycles;
immutable model, Voice Profile, Preset Voice, and language capability snapshots;
and synchronous text, phoneme, and token-ID synthesis. Model loading,
synthesis, and Voice Profile preparation release the GIL. Diagnostic and
cancellation callbacks reacquire it and defer Python exceptions until the native
call returns. Contiguous F32 Voice Reference buffers cross the seam without an
Adapter PCM copy, while Audio and Serialized Profile outputs use read-only
`memoryview` exporters that release their native buffers exactly once. Closing a
Model defers native release while any Context or Voice Profile retains it.

The package's generated low-level declarations, minimum native version, and
required-symbol manifest come from the same synthesize.cpp release and are checked
against the selected Native Provider. Wheel CI tests every advertised CPython and
platform tag rather than treating compilation with `Py_LIMITED_API` alone as proof of compatibility.

## v1 Official Wheel Matrix

The v1 binary support contract is:

| Platform tag | API wheel | `default` Provider | `cu13` Provider | `vulkan` Provider |
| --- | --- | --- | --- | --- |
| `manylinux_2_28_x86_64` | yes | CPU | CPU + CUDA | CPU + Vulkan |
| `manylinux_2_28_aarch64` | yes | CPU | CPU + CUDA | no |
| `win_amd64` | yes | CPU | CPU + CUDA | CPU + Vulkan |
| `macosx_12_0_arm64` | yes | CPU + Metal | no | no |

The API distribution contains a CPython extension and uses
`cp311-abi3-<platform>`. Native Provider distributions contain platform-native
libraries but no CPython extension and use `py3-none-<platform>`. Stable `abi3`
compatibility removes per-minor Python builds; it does not make a binary portable
between operating systems or CPU architectures.

`manylinux_2_28` promises compatibility with glibc 2.28 or later. Linux uses the
standard architecture spelling `aarch64` for 64-bit Arm, while macOS uses
`arm64`; documentation may say "Linux ARM64", but wheel filenames retain the
standard platform tag. Windows runtime support is Windows 11 x64 even though the
wheel tag is the version-independent `win_amd64`.

v1 publishes no official binary wheel for:

- musllinux or Alpine Linux;
- Linux i686, ppc64le, s390x, riscv64, or other architectures;
- Windows x86 or ARM64;
- macOS x86-64 or universal2; or
- Android, iOS, WebAssembly, or Pyodide.

These targets may be attempted as source builds but receive no binary-support
claim. Adding a platform requires the same packaging, ABI, ownership, and
end-to-end inference validation as an initial target; compilation alone is not
sufficient.

API wheels are installed and tested on every supported stable CPython minor from
3.11 through the latest stable release at publication time. Linux AArch64 runtime
tests execute on a physical DGX Spark/GB10; emulation may build an artifact but is
not support evidence. Every published CUDA, Metal, or Vulkan Provider completes
the applicable end-to-end model and backend validation on physical target
hardware for that operating system. A wheel is not published merely because it
imports, loads a driver, or enumerates a device.

Linux CUDA wheels are constructed in digest-pinned `manylinux_2_28` release
containers rather than the Ubuntu CUDA development image. Validation installs the
produced wheel into a clean runtime environment on physical hardware and never
imports the binding or native libraries from the build tree.

The API wheel uses the same pinned native-architecture containers through
`scripts/ci/build_manylinux_api.py`. Its source mount is read-only, its CMake
tree is Adapter-owned, and auditwheel verifies that the thin extension requires
no non-system shared libraries. On Linux the current extension is compatible
with glibc 2.17, so repaired filenames also carry manylinux2014/2_17 in addition
to the required 2_28 tag; this does not broaden the separately versioned Native
Provider's support claim.

## API Wheel and Native Providers

The public distribution remains `synthesize-cpp`. Its default dependency installs
the implementation distribution `synthesize-cpp-native`, which supplies the
ordinary native provider. Accelerator variants are selected through extras on the
public distribution; for example:

```sh
pip install synthesize-cpp
pip install "synthesize-cpp[cu13]"
pip install "synthesize-cpp[vulkan]"
```

The `cu13` extra installs `synthesize-cpp-native-cu13`. Future accelerator
extras follow the same separation when a distinct provider is justified. Users do
not install a public package named `synthesize-cpp-backend-*` and do not import a
second synthesis API.

## Release-Cohort Version Binding

For an API distribution release `X.Y.Z`, official dependencies use:

```toml
dependencies = [
  "synthesize-cpp-native==X.Y.Z.*"
]

[project.optional-dependencies]
cu13 = [
  "synthesize-cpp-native-cu13==X.Y.Z.*"
]

vulkan = [
  "synthesize-cpp-native-vulkan==X.Y.Z.*"
]
```

Every official Native Provider must therefore share the API distribution's exact
`X.Y.Z` base release. A `.postN` provider is permitted so packaging-only repairs
do not require a new API distribution release. Before `dlopen`, the selected
provider's descriptor must report that base release and the exact generated public
header hash. A mismatch is a hard error and never falls back to another release or
silently selects CPU. After loading, the Adapter still performs the C ABI,
minimum-version, and required-symbol checks. Distributor builds targeting an
installed native C SDK use those runtime compatibility checks rather than the
official-provider distribution pin.

## Provider Discovery and Selection

Each official Native Provider is a complete native-runtime candidate containing
one matching core, the CPU execution path, its advertised accelerator backends,
and the runtime artifacts needed to load them. An accelerator Provider is not a
plugin-only distribution to be combined with the core from another Provider. The
Python Adapter selects exactly one Provider for the process.

Providers register an internal entry point in the `synthesize_cpp.native` group.
Its descriptor supplies a stable Provider ID such as `default` or `cu13`, the
distribution identity, native library and artifact directory, base release,
generated public-header hash, advertised backend kinds, and any provider-owned
preparation hook. This is an internal packaging seam and does not extend the
public C Interface.

The optional `SYNTHESIZE_NATIVE_PROVIDER` environment variable selects one
installed Provider by its stable ID before import. It cannot contain a filesystem
path or cause arbitrary host-library discovery. Without an override, the Adapter
ranks compatible installed Providers by their highest advertised backend:

```text
CUDA > Metal > Vulkan > CPU_ACCEL > CPU
```

Two different Providers tied at the highest rank are an ambiguity error; the
Adapter never breaks a tie by entry-point order or distribution name. Every
discovered Provider must belong to the binding's release cohort and report its
exact public-header hash. A stale, malformed, or mismatched Provider is an
environment-configuration error rather than an ignored candidate.

Selection is process-global and happens once. Preparation, `dlopen`, ABI or
symbol validation, or backend registration failure in the selected Provider is a
hard error and does not try the next Provider. Once the native runtime is loaded,
a model's `backend=AUTO` request remains a separate C Interface decision and may
select that Provider's CPU path. Users neither import accelerator-specific modules
nor call an `activate()` operation.

After resolving the complete C symbol manifest, the thin extension calls
`synth_backend_load_default()` before exposing the native API. The Adapter then
requires every backend advertised by the selected Provider descriptor to be
available. A missing CPU or accelerator module is therefore a Provider
initialization error; it cannot silently turn a `cu13`, Metal, or Vulkan Provider
into a CPU-only runtime.

## Provider Runtime Contents

Every Provider carries its own matching synthesize.cpp core, CPU execution path,
private GGML runtime libraries, backend modules under the fixed
`synthesize/backends/` tree, descriptor metadata, and required notices. It does
not contain models, Voice data, or Text Frontend resources.

Linux x86-64 Providers package the complete runtime-dispatched GGML CPU module
set, including the baseline `libggml-cpu-x64.so`; the loader maps only the best
compatible module for the current processor. Linux AArch64 Providers package
one `libggml-cpu.so`. This physical module difference is private packaging
detail: both Provider descriptors advertise one logical `cpu` backend, and
Python, Rust, and C callers do not select microarchitecture names.

The `default` Provider is CPU-only on Linux and Windows. On supported macOS
targets it contains CPU and Metal; the Metal library is embedded in the backend
binary, while Foundation, Metal, MetalKit, and other operating-system frameworks
remain host dependencies.

The `cu13` Provider is published for Linux x86-64, Linux AArch64, and Windows
x86-64 and contains CPU plus CUDA. v1 builds it with CUDA 13.3 Update 1 and pins the
tested `nvidia-cuda-runtime==13.3.29` and `nvidia-cublas==13.6.0.2` packages.
The preparation hook preloads those installed runtime libraries from their
resolved package paths without changing process `PATH` or `LD_LIBRARY_PATH`. The NVIDIA driver
`libcuda.so` or `nvcuda.dll` is a host prerequisite and is never distributed.
The Provider's native artifacts do not link or initialize NCCL, cuDNN, NVRTC, or
CUPTI. The locked `nvidia-cublas==13.6.0.2` Python distribution currently
declares `nvidia-cuda-nvrtc` as an upstream transitive dependency, so a clean pip
installation contains that package even though it is not a Provider feature or a
native `DT_NEEDED` dependency. Release inventory and license review include this
transitive package.

Wheel builds use Provider-owned scikit-build directories under
`build/python-provider/<provider>/<wheel-tag>`. They never reuse a committed
development or Release preset tree; CMake generator and cache state in those
trees are part of the native validation evidence and must remain isolated from
Python build frontends.

On x86-64, the Runtime Support Floor is a Turing-or-newer GPU with compute
capability 7.5 or later and an R580-or-newer NVIDIA driver. Maxwell, Pascal, and
Volta are therefore outside the `cu13` contract. The x86-64 Native Cubin Target
Set is `sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_90`, `sm_100`, and `sm_120a`.
On Linux AArch64, v1 targets DGX Spark/GB10 and carries native `sm_121a`; it does
not claim Jetson support. Official builds do not use `all` or `all-major`.
Optional `compute_121a` PTX in the x86-64 Provider is experimental and does not
establish support. Release validation exercises the driver floor on physical
R580 hardware and also exercises the current
release-driver branch. Exact devices and target coverage belong to each release's
Validation Hardware Matrix.

The `vulkan` Provider is installed through `synthesize-cpp[vulkan]` as
`synthesize-cpp-native-vulkan` on supported Linux and Windows targets. It
contains CPU, the Vulkan backend, and compiled SPIR-V embedded in its runtime
artifacts. It does not distribute a Vulkan loader, ICD, or GPU driver; a compatible
host Vulkan loader and driver are prerequisites. Keeping Vulkan out of the default
Linux and Windows Provider ensures an ordinary CPU installation does not become
unloadable merely because the host lacks Vulkan.

Provider wheels may carry project-owned and non-system shared libraries required
by their native tree, but never copy operating-system frameworks or GPU drivers.
All carried third-party binaries and Python runtime dependencies are release-pinned,
license-audited, inventoried in the wheel's notices or SBOM, and tested from a
clean environment on every published platform tag.

A **Native Provider** is a distribution artifact containing one compatible native
runtime: a matching core, CPU execution, its optional accelerator backends,
required runtime libraries, and provider metadata. An **Execution Backend** is the
runtime compute implementation selected through the C Interface, such as CPU,
CUDA, Metal, or Vulkan. A provider may carry one or more backends, so provider
package identity and backend identity are not a one-to-one mapping.

The thin extension is not linked against all native symbols. During module
initialization it opens only the library selected through the provider contract
and performs the confirmed ABI, minimum-version, and required-symbol checks before
exposing higher-level Python objects. It does not search the current working
directory, `PATH`, system library directories, or an unrelated synthesize.cpp
installation.

Provider-owned native trees retain `synthesize/backends/` for compatible dynamic
backend artifacts. The default provider must work without an optional accelerator
provider. No official provider contains a Model Package, Voice data, Sidecar
Resource, URL downloader, or network installation path.

## Python Ownership and Buffers

Python Model, Synthesis Context, and Voice Profile objects own exactly one
corresponding native handle and provide idempotent `close()` plus context-manager
cleanup. Native Audio and Byte Buffers are held by Python owner objects whose
finalizers call the matching native release function once. A read-only standard
`memoryview` keeps that owner alive, so consumers including optional NumPy
`frombuffer` can view contiguous F32 PCM without another PCM copy.

NumPy is not a required dependency and the baseline return type is a standard
Python buffer view. Borrowed input buffers remain pinned only for the synchronous
native operation. File and encoded-audio conveniences decode before crossing the C
seam.

Long native operations release the GIL. Python Audio Sink, cancellation, and
diagnostic callbacks reacquire it, capture exceptions, return a native error or
cancellation result, and re-raise only after the C call has returned. No Python
exception unwinds through the native callback.

## Distributor System-Library Build

The source build supports an explicit distributor option that omits the default
Native Provider dependency and targets one configured installed native C SDK.
That build validates the same ABI, minimum version, and symbol manifest and is
never published as an official PyPI wheel. Failure does not download or silently
select another library.
Official wheels expose no unrestricted runtime environment-variable override that
can select an arbitrary host library.

Wheel tests repeatedly create and destroy native owners, retain buffer views past
their producing call, and exercise callback exceptions and GIL transitions on all
supported CPython minors and platform tags.
