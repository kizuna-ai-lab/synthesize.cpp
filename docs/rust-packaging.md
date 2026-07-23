# Rust Binding Packaging

Status: Confirmed on 2026-07-21.

## Crate Split

The Rust distribution has two crates. Package `synthesize-cpp-sys` exposes the
generated low-level C declarations, the release's required-symbol manifest and
minimum-version metadata, and native build or discovery logic. Package
`synthesize-cpp` depends on that crate and provides the safe Model, Synthesis
Context, Voice Profile, request, sink, and owned-buffer Adapter. Its Cargo crate
names use underscores as `synthesize_cpp_sys` and `synthesize_cpp`.

`synthesize-cpp-sys` declares `links = "synthesize"`, so Cargo permits only one
native synthesize.cpp runtime in a final dependency graph. Neither crate
reimplements Model Family dispatch, backend selection, Voice Profile preparation,
synthesis, or native ownership policy.

## Default Bundled Build

The default build compiles the exact native source bundled in the
`synthesize-cpp-sys` release as a static CPU library through CMake with
`BUILD_SHARED_LIBS=OFF`. It does not build the CLI and does not download source,
models, Voice data, Sidecar Resources, or backend modules. The checked-in
generated Rust declarations and bundled native source come from the same release.

The additive `cuda`, `metal`, and `vulkan` Cargo features apply only to the
bundled build and statically compile and register the selected native backend.
CPU remains present in every combination. Requesting a backend on a target or
toolchain that cannot build it is a build error rather than a silent CPU-only
result. Bundled static builds do not package dynamic backend modules, and
`synth_backend_load_default()` retains its documented no-op behavior.

## System SDK Build

The additive `system` feature switches the build script from bundled compilation
to the installed native C SDK. An explicit installation prefix takes precedence;
otherwise supported Unix-like hosts use the confirmed `synthesize.pc` metadata,
with platform-specific installed-SDK discovery supplied where pkg-config is
unavailable. Discovery does not download or build a replacement library after
failure.

The sys crate continues to use its checked-in declarations in system mode; it
does not run bindgen against the installed header, so users do not need Clang.

An explicit prefix names one SDK root; mixing a header from one installation
with a library from another installation is rejected.

`system` combined with `cuda`, `metal`, or `vulkan` is a compile-time error.
Those features cannot determine the composition of a preinstalled SDK; the
application or distributor selects an appropriately built native package instead.
The installed package determines static or shared linkage.

System discovery checks its advertised release against the crate's minimum native
version. Normal linking validates the required symbol set. Before constructing
the first safe Rust object, `synthesize-cpp` also checks `synth_abi_version()` and
the actual `synth_get_version()` result once and caches only a successful
compatibility result.

## Safe Adapter Rules

The safe crate owns native handles and buffers with `Drop`, preserves Model
lifetimes behind Context and Voice Profile values, and requires exclusive mutable
Context access for synthesis. Borrowed audio, text, token, diagnostic,
cancellation, and Audio Sink views remain scoped to the synchronous native call.
Callback trampolines catch Rust panic and translate it without unwinding through C.

Cargo features select only packaging and compiled backend availability. They do
not create family-specific Rust entry points, download Model Packages, or bypass
native capability checks. CI tests default CPU, each supported backend, installed static/shared SDKs, ABI layouts, ownership, and callback round trips.
