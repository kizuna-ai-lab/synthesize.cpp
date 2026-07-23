# Language Binding Contract

Status: Cross-language support, ABI encoding, Rust packaging, and Python API and
Native Provider packaging are confirmed Interface constraints, last updated on 2026-07-22.

## One External Seam

```text
                       +-> CLI reference adapter
C++ implementation -> C ABI -> Rust safe adapter
                       +-> Python abi3 C adapter
```

Every consumer crosses the same public C Interface. Rust and Python wrappers may improve ergonomics but cannot reproduce model-family dispatch, backend selection, profile preparation, synthesis, buffering, or error policy.

## Common Contract

- Export only C symbols, opaque handles, fixed-layout structures, fixed-width scalars, callbacks, and explicit pointer-plus-count views.
- Keep all calls synchronous in v1; borrowed input memory remains valid for the call and is never retained silently.
- Pair every library-owned allocation or handle with exactly one library release function.
- Keep UTF-8 strings length-delimited and do not require null termination.
- Return stable status values and structured diagnostics; binding-specific exceptions are adapter behavior.
- Never allow a C++ exception, Rust panic, or Python exception to unwind across the C seam.
- Test behavior through the C Interface; binding tests add ownership and translation checks rather than duplicating synthesis correctness suites.

## ABI Bootstrap and Release Version

Dynamic Rust and Python loaders resolve `synth_abi_version()` before declaring or calling the rest of the native Interface and require an exact match with the ABI version compiled into the binding. A mismatch is a load-time error with both expected and actual ABI values; the Adapter must not attempt best-effort ABI adaptation. Static bindings may retain the same check as a startup or debug assertion.

Only after the ABI matches may a binding resolve `synth_version_init()` and `synth_get_version()`, initialize `synth_version_t`, and obtain the independent semantic release version. A different major, minor, or patch value is not itself an ABI failure, but the runtime must be at least the binding's generated minimum native version. Neither loader depends on Git metadata, build timestamps, compiler strings, or filename parsing for compatibility.

## Dynamic Loader Compatibility

Each binding release contains a minimum native semantic version and a
required-symbol manifest generated from the same `synthesize.h` release as its low-level
declarations. The minimum advances when the binding consumes a newly introduced
symbol, appended structure field, or associated semantic contract; it is not
inferred from the binding package filename at load time.

After the ABI and minimum-version checks, the loader eagerly resolves every
manifest entry before publishing any safe Rust or Python object. A missing
version-bootstrap or manifest symbol is a load-time error that reports the missing
name and the expected and actual version information available at that point. The
Adapter neither exposes a partially usable Interface nor silently ignores a newer
field to run against an older library.

The version check protects additions that do not create new symbols, while the
manifest detects incomplete or incorrectly packaged libraries. Official Adapters
use ordinary exported C symbols and do not introduce a duplicate `synth_get_api()`
function table. A direct C dynamic loader should apply the same sequence; an
ordinarily linked C program receives the corresponding symbol check from its
linker.

## Linkage, Visibility, and Calling Convention

Bindings consume the installed header's C linkage and `SYNTH_CALL` contract rather than inferring a convention from the host language. Rust declares exported functions and callbacks as `extern "C"`; Python uses cdecl loading and callback forms (`CDLL` and `CFUNCTYPE` semantics), including on Windows. No Adapter uses stdcall or binds C++ symbols.

Shared-library packaging propagates `SYNTH_SHARED`; only the native library build defines `SYNTH_BUILDING_LIBRARY`. Static packages define neither. Windows packages include the checked undecorated `.def` export set, and Linux/macOS packages expose only default-visibility `synth_*` symbols. Binding CI compares its expected symbol manifest with the actual library export table before running ownership and synthesis tests.

## ABI Layout Conformance

The installed `synthesize.h` header remains the canonical declaration source. Low-level Rust declarations and Python FFI declarations are generated during the project's release process, checked into or bundled with their respective packages, and tested against the same release header. End users do not need Clang, a header parser, or a binding generator to build or import a supported package.

On every tested target, a test-only C ABI probe supplies native `sizeof`, alignment, and field-offset values. Rust and Python calculate the same values from their generated declarations and compare them item by item; they do not reuse an x86-64 golden table on other data models. The bindings additionally exercise diagnostic, cancellation, and Audio Sink callbacks through real native round trips.

Size-tagged compatibility tests pass every retained older v1 structure prefix to the current library with surrounding guard bytes. They verify that initializers write only within the caller-declared size and that the library accepts and reads only fields present in that prefix. The C probe and these fixtures remain test implementation and never become another public query Interface.

## Fixed-width ABI Values

All status and enum-like public types are `int32_t` or `uint32_t` aliases with explicitly cast constants, not C enums. The only public boolean representation is `synth_bool_t`, a `uint32_t` for which the library writes exactly zero or one and callbacks may use any nonzero value as true. Flags use explicitly sized unsigned integers, while counts and structure sizes use `uint64_t`.

The low-level Rust crate maps these types to `i32`, `u32`, and `u64`; its safe Adapter converts them into Rust enums and `bool` only after validation. Python sees ordinary integers at the native seam and converts them to Python enums and booleans in the Adapter. Neither binding depends on compiler enum size, C `_Bool`, C++ `bool`, or `-fshort-enums` behavior.

## Rust Adapter

The low-level crate mirrors the C header. A safe crate wraps Model, Synthesis Context, Voice Profile, and Audio Buffer with `Drop`; retains the Loaded Model behind Context and Profile wrappers; requires mutable access to a Context for synthesis; and exposes immutable Model and Voice Profile sharing only where the C Interface guarantees it. Callback trampolines catch panic before returning to C.

Borrowed PCM and token inputs map to slices for the synchronous call. An owned audio result may expose a slice tied to an owning Rust value without copying.

The published packages are `synthesize-cpp-sys` and `synthesize-cpp`.
The sys crate declares `links = "synthesize"`, builds the matching static CPU
runtime from bundled source by default, and can instead select the installed SDK
with its `system` feature. Bundled `cuda`, `metal`, and `vulkan` features compile
those backends statically; combining any of them with `system` is a build error
because an installed SDK owns its backend composition. Neither packaging mode
performs network downloads.

## Python Adapter

Python classes own the three opaque handles and offer explicit `close()` plus context-manager cleanup. Long-running native work may release the GIL, but a Python Audio Sink, cancellation callback, or diagnostic callback must reacquire it. Callback exceptions are captured and re-raised only after the C call has returned an error.

The implemented Adapter owns Model, Context, and Voice Profile handles; copies
device, general capability, Voice Profile capability, Preset Voice, and language
values into immutable Python objects; and performs synchronous text, phoneme,
and token-ID synthesis with diagnostic and cancellation callbacks. Its
`Audio.samples` is a read-only one-dimensional F32 `memoryview`; the exporter
owns the native Audio Buffer, so the view remains valid after closing Context
and Model without a PCM copy. The actual seed and resolved language/Voice are
returned in an immutable `SynthesisResult`.

Contiguous F32 objects supporting the Python buffer protocol provide borrowed
Voice Reference or token input views. Profile preparation builds only the small
descriptor array and pins each PCM exporter for the synchronous native call.
Serialized Voice Profile output likewise uses a read-only byte `memoryview`
whose exporter calls `synth_byte_buffer_free()` when the last view dies.

The published API distribution is `synthesize-cpp` and imports as
`synthesize_cpp`. Official CPython 3.11+ `abi3` API wheels contain a thin
Limited-API C extension. Their default dependency is the implementation
distribution `synthesize-cpp-native`; `synthesize-cpp[cu13]` additionally
installs `synthesize-cpp-native-cu13`. Native providers are distribution
artifacts, while CPU, CUDA, Metal, and Vulkan remain runtime Execution Backends.
Provider packages do not create additional public Python imports and are not
publicly named `synthesize-cpp-backend-*`.

v1 publishes `cp311-abi3` API wheels and `py3-none` Native Provider wheels for
`manylinux_2_28_x86_64`, `manylinux_2_28_aarch64`, `win_amd64`, and
`macosx_12_0_arm64`. Linux AArch64, Windows, and macOS receive only the Provider
variants declared in the Python packaging matrix; v1 does not publish musllinux,
32-bit, Windows ARM64, macOS x86-64 or universal2, mobile, or WebAssembly wheels.
Stable ABI compatibility spans supported CPython minors but never spans platform
tags. GPU Provider publication requires end-to-end validation on physical target
hardware.

Official Native Providers register through the internal
`synthesize_cpp.native` entry-point group and each contains a complete core,
mandatory CPU execution, and its advertised accelerators. The Adapter selects one
Provider per process. `SYNTHESIZE_NATIVE_PROVIDER` may name an installed stable
Provider ID but cannot specify a path; otherwise precedence is CUDA, Metal,
Vulkan, CPU_ACCEL, then CPU. Equal highest-ranked Providers are an error.

All discovered Provider descriptors must match the binding release cohort and
public-header hash. Selection or loading failures do not fall through to another
Provider. Provider selection remains distinct from a later per-model
`backend=AUTO` decision, and users do not import or activate an
accelerator-specific package.

The default Provider is CPU-only on Linux and Windows and CPU-plus-Metal on
supported macOS targets, with the Metal library embedded. The `cu13` Provider
contains CPU plus CUDA and pins the tested `nvidia-cuda-runtime==13.3.29` and
`nvidia-cublas==13.6.0.2` runtime packages; its preparation hook resolves them without
mutating `PATH` or `LD_LIBRARY_PATH`. The NVIDIA driver remains a host
prerequisite. Provider native artifacts do not link or initialize NCCL, cuDNN,
NVRTC, or CUPTI; the locked cuBLAS Python distribution's upstream transitive
`nvidia-cuda-nvrtc` dependency is still inventoried and license-reviewed.

Linux AArch64 uses the same Provider ID and distribution name as x86-64 but its
platform wheel carries native `sm_121a` code for DGX Spark/GB10. Platform wheel
selection does not add another public Python import or Provider-selection rule.

The `vulkan` Provider contains CPU, Vulkan, and embedded SPIR-V, while the Vulkan
loader, ICD, and driver remain host prerequisites. Providers carry the matching
core, private GGML runtime artifacts, backend modules, descriptors, licenses, and
notices or SBOM, but no models, Voice data, Text Frontend resources, operating
system frameworks, or GPU drivers.

Module initialization loads only a library selected through the controlled native
provider contract and validates ABI, minimum version, and every required symbol.
Standard read-only `memoryview` output is the dependency-free zero-copy view;
NumPy remains optional. Official wheels never search for or fall back to an
unrelated system library, while distributor-built local wheels may explicitly
target one configured installed SDK.

## Preset Voice Catalog

Both bindings enumerate the immutable Catalog through the C count and indexed-get operations; neither receives an internal array, creates a Catalog handle, or uses a callback. They expose `flags` as validated bitflags while retaining unknown future bits. Applications select and persist the stable `id`, never the display name or enumeration index.

A safe Rust Adapter may expose each entry as a read-only view whose string lifetimes are tied to the Loaded Model. The Python Adapter copies the small `id` and optional `display_name` into Python strings so a result does not retain borrowed native pointers. An empty Catalog remains distinguishable from failure and may coexist with an unnamed package-default Voice. Neither Adapter invents language, gender, age, or accent fields or inserts runtime Voice Profiles into the Catalog.

## Language Capability Catalog

Rust and Python enumerate language capabilities through the C count and indexed-get operations. The Rust Adapter may expose canonical tags as read-only strings tied to the Loaded Model; the Python Adapter copies each small tag into a Python string. Both expose known flags ergonomically while retaining unknown future bits, and neither localizes names or reconstructs capabilities from Model Family or Preset Voice metadata.

Adapters may validate basic BCP 47 syntax for early feedback but the core repeats validation, case-insensitive matching, exact-match precedence, default selection, and permitted regional fallback. They return the core's canonical resolved tag and do not broaden fallback to scripts, variants, private-use tags, or different primary languages. v1 exposes no separate Voice-and-language preflight query; synthesis performs the authoritative combination validation.

## General Model Capabilities

Rust and Python map the two fixed-width flag sets to ergonomic bitflags while retaining unknown future bits. They expose the native output format, speaking-rate range, and hard input-token and output-frame limits as immutable properties of the Model wrapper. Available input flags come from the Loaded Model snapshot and are not reconstructed from package names or binding-side frontend discovery.

Bindings may use these values to select input methods, convert frame limits to durations, and decide whether to expect Native Streaming Synthesis, but the core repeats every request check. Operation result flags remain authoritative for actual seed and streaming-path use.

## Device Discovery

Rust and Python preserve `memory_total` and `memory_free` as unsigned byte counts
and map `synth_device_flags_t` to ergonomic bitflags while retaining unknown future
bits. When `SYNTH_DEVICE_MEMORY_INFO_VALID` is absent, both Adapters expose memory
information as unavailable rather than interpreting the native zero values as a
zero-capacity device. They present `SHARED` and `APPROXIMATE` independently and do
not rename shared physical memory to CUDA unified memory.

Bindings may display the snapshot or use it for advisory planning, but cannot turn
it into a model-load precondition. They do not run `nvidia-smi`, inspect operating-
system memory counters, or synthesize different values from platform-specific
sources; all device discovery crosses the same C Interface.

## Voice Profile Sources

Reference Audio, Description Text, Random Seed, and Serialized Profile are language-neutral source semantics. Each has a separate C preparation function with a size-tagged source structure and returns the same opaque Voice Profile type. There is no public union, generic untyped source pointer, or Model Family-specific preparation function. Rust and Python adapters can therefore expose overloads, constructors, or class methods without reproducing native dispatch logic or declaring union layouts.

Reference Audio uses a contiguous array of size-tagged descriptors with an explicit byte stride. Rust supplies `size_of::<synth_voice_reference_t>()`; a Python Adapter builds only the small descriptor array while borrowing each contiguous `float32` buffer, so constructing the descriptors does not copy PCM. File and encoded-audio conveniences decode in an Adapter before crossing the C seam. The descriptor stride concerns structure layout only and is not a PCM sample stride.

Description Text maps directly from a borrowed Rust UTF-8 string or the UTF-8 bytes of a Python string for the synchronous call. Its language tag describes the prompt rather than later synthesis output. Both Adapters require or generate a concrete preparation seed before crossing the C seam; they do not pass `SYNTH_SEED_RANDOM` to profile preparation.

Random Seed maps to one concrete `u64` or non-negative Python integer within the accepted range. Safe Adapters reject `SYNTH_SEED_RANDOM` before the call and may offer a convenience that obtains a random concrete seed from the host, returns or stores that value, and then calls the same C function. They do not reproduce the Model Family's Voice sampling logic.

Serialized Profile import borrows a Rust byte slice or a Python buffer only for the synchronous call. Export returns a library-owned Byte Buffer whose Rust slice or Python `memoryview` retains the owning wrapper and calls `synth_byte_buffer_free()` exactly once. Path-based convenience methods perform ordinary file I/O in the Adapter and still call the same C memory operations.

Bindings treat Serialized Profile GGUF bytes as opaque and do not parse, rewrite, or relax their Profile Schema, integrity, or Profile Compatibility ID checks. A profile exported on one validated Execution Backend can be passed unchanged to a compatible Loaded Model on another backend; any backend preparation occurs inside the C Interface after validation.

## Voice Profile Capability Query

The low-level bindings preserve `source_flags` as a fixed-width integer and the three requirement fields as `uint32_t`. A safe Rust Adapter exposes bitflags plus a validated requirement enum and copies the compatibility ID into `[u8; 32]`. A Python Adapter exposes an `IntFlag`, requirement enum values, an immutable schema string, and a 32-byte `bytes` value. Neither Adapter reconstructs capabilities from Model Family names or Preset Voice entries.

The same capability value exposes the Model's Reference Audio target format and Reference Frame Equivalent limits. Rust and Python may preflight the checked `ceil(input_frames * target_rate / input_rate)` duration measure for user feedback, but the core repeats every check and owns resampling and channel conversion. Adapters do not expose resampler algorithms or quality settings. Zero-copy borrowing applies only to contiguous F32 PCM already matching the target format; an Adapter must not claim zero-copy when it converts dtype, layout, sample rate, or channels.
