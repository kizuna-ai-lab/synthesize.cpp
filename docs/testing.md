# Testing Policy

Status: Confirmed, last updated on 2026-07-30.

Testing is a per-slice completion gate. A new converter rule, graph stage,
runtime control, backend path, or public Interface is not complete merely because
its happy path ran once. Its focused tests must be committed and passing before
implementation begins on the next slice.

## Required Gate for Every Slice

1. Add model-independent unit tests for the slice's observable behavior,
   including its valid path, limits, malformed input, and error mapping.
2. Register those tests with CTest under the `unit` label. Python porting tools
   use the family's locked `uv` environment and the same label.
3. If the slice changes model semantics, add or extend the family Golden
   integration test and register it under `integration`, `golden`, and the
   family label.
4. Run all unit tests in a sanitizer build. Run at least one real-model case
   under the same build whenever the slice executes C or C++ inference code.
5. Run the complete affected Golden suite in the reference-dtype CPU build and
   record the evidence in the family porting log.

A skipped test requires an explicit blocker in the porting log. Manual commands,
an unregistered runner, or a generated comparison report do not substitute for a
registered test.

Tests cross the smallest stable module Interface. Family implementation details
remain private: tests may use internal seams from `src/arch/<family>/`, but those
seams are not added to the public C Interface merely for testing. Real model files,
Golden tensor payloads, and generated reports remain ignored artifacts.

## Test Classes

| Label | Dependencies | Purpose |
| --- | --- | --- |
| `unit` | Source tree and locked family environment | Fast behavior, malformed input, metadata, mapping, shape, and lifecycle checks |
| `integration` | Local converted model and materialized Golden payloads | Exercise the real library module through a thin runner |
| `golden` | Pinned oracle payloads | Compare named tensors and behavior against the reference implementation |

The `unit` suite is enabled by `SYNTH_BUILD_TESTS=ON`. Locked Python tests are
controlled by `SYNTH_BUILD_PYTHON_TESTS`; it defaults to `ON` whenever project
tests are built. Model-dependent tests remain explicit because their artifacts
are intentionally not committed.

## Commands

`ggml/` is a git submodule. Every command below assumes it is checked out:

```bash
git submodule update --init --recursive
```

A clone made without `--recurse-submodules` leaves it empty, and the configure
step fails naming this command rather than reporting a missing `CMakeLists.txt`.

Run the ordinary unit gate:

```bash
cmake -S . -B build \
  -DSYNTH_BUILD_TESTS=ON \
  -DSYNTH_BUILD_PYTHON_TESTS=ON \
  -DSYNTH_BUILD_INTEGRATION_TESTS=OFF
cmake --build build --target synthesize-check-unit
```

Enable the VITS real-model Golden gate after materializing its ignored model and
reference payloads:

```bash
cmake -S . -B build \
  -DSYNTH_BUILD_TESTS=ON \
  -DSYNTH_BUILD_INTEGRATION_TESTS=ON \
  -DSYNTH_VITS_TEST_MODEL="$PWD/models/vits-ljspeech/vits-ljspeech-F32.gguf" \
  -DSYNTH_VITS_VCTK_TEST_MODEL="$PWD/models/vits-vctk/vits-vctk-F32.gguf"
cmake --build build --target synthesize-check-integration
```

When the VCTK model or materialized VCTK Golden sentinel is absent, its optional
eight-test integration group is not registered; the required LJSpeech integration
gate remains unchanged.

Qwen3-TTS registers two integration tests from `SYNTH_QWEN3_TTS_TEST_MODEL`,
which defaults to the family's **BF16** package rather than an F32 one: its
talker checkpoint stores bfloat16, and widening it would describe a different
model.

```bash
cmake -S . -B build \
  -DSYNTH_BUILD_TESTS=ON \
  -DSYNTH_BUILD_INTEGRATION_TESTS=ON \
  -DSYNTH_QWEN3_TTS_TEST_MODEL="$PWD/models/qwen3-tts-12hz-0-6b-customvoice/qwen3-tts-12hz-0-6b-customvoice-BF16.gguf"
cmake --build build --target synthesize-check-integration
```

`synthesize-qwen3-tts-replay-golden` runs the replay validator with `--check`, so
it is a gate against the committed tolerances rather than a measurement -- without
it `tests/tolerances/qwen3-tts.json` is a record nothing enforces. It needs the
oracle payload under `build/goldens/qwen3-tts/` and is not registered without it.

`synthesize-qwen3-tts-public-request` needs only the package. It asserts relations
between runs of this port -- a seed reproduces, a different seed does not, a Voice
change moves the audio -- rather than agreement with the reference, so it has no
Golden sentinel.

OmniVoice registers one integration test from `SYNTH_OMNIVOICE_TEST_MODEL`,
which defaults to `models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf`. Unlike
Qwen3-TTS, this family's source checkpoint stores F32 in both the generator
and the codec halves, so F32 is the profile a port is checked against rather
than a widened dtype.

```bash
cmake -S . -B build \
  -DSYNTH_BUILD_TESTS=ON \
  -DSYNTH_BUILD_INTEGRATION_TESTS=ON \
  -DSYNTH_OMNIVOICE_TEST_MODEL="$PWD/models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf"
cmake --build build --target synthesize-check-integration
```

`synthesize-omnivoice-load-real` needs only the package: it opens it through
the family loader and then through the public seam, and is registered under
`integration`, `omnivoice`, and `abi`.

Run the DGX Spark CUDA 13.3 Update 1 gate in a separate build tree. CUDA F32
matrix multiplies compute at TF32 precision and there is no build option to
change that; see `docs/backends.md`. The committed development preset owns the
backend, build mode, tests, and native `sm_121a` target. Point both CMake toolkit
discovery and its CUDA compiler selection at one CUDA 13.3 root; configuration
rejects another CUDA minor version so host 13.0 headers cannot be mixed with the
13.3 compiler:

```bash
export SYNTH_CUDA_ROOT=/path/to/cuda-13.3
export PATH="$SYNTH_CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$SYNTH_CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
CUDAToolkit_ROOT="$SYNTH_CUDA_ROOT" \
  CUDACXX="$SYNTH_CUDA_ROOT/bin/nvcc" \
  cmake --preset dev-dgx-spark
cmake --build --preset dev-dgx-spark -j
ctest --preset dev-dgx-spark
```

The equivalent RTX 4070 SUPER command uses the
`dev-linux-x86_64-cuda` preset and native `sm_89`. Release jobs use the
committed `release-linux-aarch64-cu13` and `release-linux-x86_64-cu13` presets;
their exact cubin target sets cannot be replaced with `native`, `all`,
`all-major`, or a command-line subset. The `a-real` CMake spelling for 12.x
architectures selects the architecture-specific native cubin form required by
GGML and CUDA 13.3.

Every test-enabled build also registers `synthesize-installed-sdk-test`. It
installs the current build under a fresh isolated prefix and builds and runs a
pure-C consumer once through the installed CMake package and once through the
installed `synthesize.pc`. Static builds use `pkg-config --static`. CUDA runs
also verify on Linux that `libcudart.so.13` and `libcublas.so.13` resolve from
the CUDA 13.3 root selected by the build, rather than from another host toolkit.
Run only this packaging gate with:

```bash
ctest --test-dir build/dev-dgx-spark \
  --output-on-failure -R '^synthesize-installed-sdk-test$'
```

Release presets build the library and run the registered cubin verifier even
though source-tree tests are disabled:

```bash
cmake --build --preset release-linux-aarch64-cu13 -j
cmake --build --preset release-linux-x86_64-cu13 -j
```

Build Linux Provider wheels only in the digest-pinned native-architecture
manylinux container. The builder mounts the source and CUDA toolkit read-only,
uses a Provider-owned CMake tree, runs `auditwheel repair`, checks the wheel
contract, and applies the exact-cubin/no-PTX verifier to CUDA artifacts:

```bash
# DGX Spark / Linux AArch64
docker pull quay.io/pypa/manylinux_2_28_aarch64@sha256:162c81dfd3efc710732a571717d3c916a6945ebf279e879ddee3243af96fe46f
python3 scripts/ci/build_manylinux_provider.py \
  --provider default --arch aarch64 --jobs 12
python3 scripts/ci/build_manylinux_provider.py \
  --provider cu13 --arch aarch64 --cuda-root /path/to/cuda-13.3 --jobs 12

# Native Linux x86-64 host
docker pull quay.io/pypa/manylinux_2_28_x86_64@sha256:a61875a2f84cab7df8de222ff12cabc08ff86eb4ad402ac90ba7bdaed9600cca
python3 scripts/ci/build_manylinux_provider.py \
  --provider default --arch x86_64 --jobs 12
python3 scripts/ci/build_manylinux_provider.py \
  --provider cu13 --arch x86_64 --cuda-root /path/to/cuda-13.3 --jobs 12

# Build the cp311-abi3 API Adapter on each native host.
python3 scripts/ci/build_manylinux_api.py --arch aarch64 --jobs 12
python3 scripts/ci/build_manylinux_api.py --arch x86_64 --jobs 12
```

The script refuses emulated cross-architecture release builds and refuses to
overwrite a different wheel with the same filename. CUDA repair deliberately
leaves `libcuda.so.1`, `libcudart.so.13`, `libcublas.so.13`, and
`libcublasLt.so.13` external: the driver is host-owned and the other libraries
come from the Provider's pinned NVIDIA Python dependencies. Therefore a clean
runtime smoke installs dependencies normally, calls the Provider preparation
hook, verifies it does not mutate `PATH` or `LD_LIBRARY_PATH`, and runs at least
one complete public-C-Interface synthesis from the repaired wheel on physical
hardware. An `auditwheel show` classification of plain `linux_<arch>` for a
CUDA wheel is expected from those intentional non-policy externals and does not
replace this clean-runtime dependency and hardware gate.

The API wheel gate installs the same repaired `cp311-abi3` artifact on every
supported CPython minor, combines it separately with the default and accelerator
Providers, and performs actual model loading and synthesis. The 0.1.0 AArch64
artifact passed CPython 3.11.15, 3.12.13, 3.13.14, and 3.14.6 on DGX Spark; the
x86-64 artifact passed CPython 3.12.13 on the RTX 4070 SUPER host. Both physical
hosts passed default CPU and cu13 CUDA Provider combinations.

The public shared-library development gate consists of the ABI smoke, public
API, backend-device, installed-SDK, and export-list tests. Private VITS unit
executables intentionally link internal symbols and are therefore not part of
the hidden-symbol shared-library gate.

The seven VITS validators accept `--backend cuda`; qualification archives their
reports with toolkit and host tags so the CUDA 13.0 checkpoint, CUDA 13.3 hosts,
and CPU evidence remain distinct.

Run the unit gate under AddressSanitizer and UndefinedBehaviorSanitizer:

```bash
cmake -S . -B build-sanitize \
  -DSYNTH_BUILD_TESTS=ON \
  -DSYNTH_BUILD_PYTHON_TESTS=ON \
  -DSYNTH_BUILD_INTEGRATION_TESTS=OFF \
  -DSYNTH_SANITIZE=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-sanitize --target synthesize-check-unit
```

Direct label selection remains available for CI and diagnosis:

```bash
ctest --test-dir build --output-on-failure -L unit
ctest --test-dir build --output-on-failure -L integration
ctest --test-dir build --output-on-failure -L vits
```

## Check targets build what they run

`synthesize-check-unit` and `synthesize-check-integration` derive their
dependencies from global properties that are appended to where each executable
is declared — `synth_add_unit_test` does it automatically, and
`synth_register_integration_target` does it for Golden runners and the CLI,
which are not tests and so are not picked up by a test helper.

They used to carry hand-written dependency lists. A test could then be correct,
registered, and never built by the gate, in which case it ran against whatever
binary was already in the tree or did not run at all. That happened twice: once
for a unit test and once for a Golden runner, and a stale local build hid both.
Deriving the list from declaration removes the second place where the two could
disagree.

A new test or runner therefore needs no change here. What still does is anything
the gate drives that is neither: register it explicitly.
