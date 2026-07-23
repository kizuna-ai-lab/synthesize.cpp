---
status: accepted
---

# Limit the v1 Python wheel platform matrix

v1 publishes the `synthesize-cpp` CPython 3.11+ Limited-API Adapter and its
Native Providers only for this binary matrix:

| Platform tag | API | default | cu13 | vulkan |
| --- | --- | --- | --- | --- |
| `manylinux_2_28_x86_64` | `cp311-abi3` | CPU | CPU + CUDA | CPU + Vulkan |
| `manylinux_2_28_aarch64` | `cp311-abi3` | CPU | CPU + CUDA | none |
| `win_amd64` | `cp311-abi3` | CPU | CPU + CUDA | CPU + Vulkan |
| `macosx_12_0_arm64` | `cp311-abi3` | CPU + Metal | none | none |

Provider wheels use `py3-none-<platform>` because they contain native runtime
artifacts but no CPython extension. The Adapter's `abi3` tag removes
per-CPython-minor builds only; operating-system and CPU-architecture tags remain
independent compatibility dimensions.

Linux wheels target glibc 2.28 or later through the maintained
`manylinux_2_28` policy. Linux uses `aarch64` for 64-bit Arm, macOS uses
`arm64`, and Windows uses `win_amd64` for x86-64 according to their standard
wheel-platform spellings. Runtime support for the Windows wheels is Windows 11
x64.

v1 publishes no musllinux, 32-bit, Windows ARM64, macOS x86-64 or universal2,
mobile, WebAssembly, or other architecture wheel. A source build may work on an
unlisted target but is not an official binary-support claim.

Every API wheel is installed and tested on each supported stable CPython minor
from 3.11 through the latest stable release available at publication time.
AArch64 runtime tests run on a physical DGX Spark/GB10. Each GPU Provider passes
end-to-end model and backend validation on physical target hardware for every
published operating-system tag; cross-compilation, emulation, successful import,
driver loading, or device enumeration alone is insufficient.

## Considered Options

- `manylinux2014` was rejected as the primary build policy because its CentOS 7
  base is end-of-life and its older toolchain provides no required compatibility
  benefit for the initial project target.
- A newer glibc floor was rejected because `manylinux_2_28` already provides a
  maintained modern toolchain while preserving compatibility with glibc 2.28
  distributions.
- musllinux was deferred because it creates another C-runtime packaging and test
  matrix, while the selected CUDA runtime dependencies are distributed for
  glibc-based Linux.
- The Linux AArch64 CUDA Provider is included because the pinned NVIDIA runtime
  dependencies publish AArch64 wheels and DGX Spark/GB10 supplies native
  end-to-end validation hardware.
- The Vulkan AArch64 Provider remains deferred until stable native GPU CI and the
  full validation suite exist for that target.
- macOS x86-64, universal2, and Windows ARM64 wheels were deferred because each
  adds another native validation target rather than following automatically from
  source portability.
- Treating QEMU, cross-compilation, import success, or device enumeration as
  support evidence was rejected because none exercises full model inference on
  the published target.

No public C Interface change results from this Python distribution decision.
The CUDA Provider identity and runtime floor are defined by ADR 0016.
