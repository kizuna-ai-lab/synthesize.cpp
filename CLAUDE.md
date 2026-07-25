# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

synthesize.cpp is an embeddable, local, offline C/C++ text-to-speech inference library built on GGML/GGUF. The stable C interface in `include/synthesize.h` is the primary product surface; the CLI (`examples/cli/`) and the Python/Rust bindings are only adapters over that same C interface — no synthesis capability may exist only in the CLI or a binding. See `docs/scope.md` for confirmed in/out-of-scope items.

## Commands

Requires CMake ≥3.24 and `uv` (Python tests and the pinned clang-format run through it).

```bash
# Configure + run the standard unit gate (the default development loop)
cmake -S . -B build \
  -DSYNTH_BUILD_TESTS=ON \
  -DSYNTH_BUILD_PYTHON_TESTS=ON \
  -DSYNTH_BUILD_INTEGRATION_TESTS=OFF
cmake --build build --target synthesize-check-unit

# Run a single test / a labeled subset
ctest --test-dir build --output-on-failure -R '^synthesize-public-api-test$'
ctest --test-dir build --output-on-failure -L unit        # labels: unit, integration, golden, vits, abi, backend
```

Sanitizer gate (required for every slice that touches C/C++ inference code):

```bash
cmake -S . -B build-sanitize -DSYNTH_BUILD_TESTS=ON -DSYNTH_BUILD_PYTHON_TESTS=ON \
  -DSYNTH_BUILD_INTEGRATION_TESTS=OFF -DSYNTH_SANITIZE=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-sanitize --target synthesize-check-unit
```

Formatting (pinned clang-format 22.1.5 via uvx; never format `ggml/`):

```bash
scripts/ci/clang-format.sh --fix          # format the tree
scripts/ci/clang-format.sh --check-diff   # what CI checks for changed files vs origin/main
```

Model-dependent gates: real model GGUFs and Golden tensor payloads are deliberately not committed. Integration/golden tests only register after materializing them locally, then configuring with `-DSYNTH_BUILD_INTEGRATION_TESTS=ON` (model paths default to `models/vits-ljspeech/…` and `models/vits-vctk/…`, overridable via `SYNTH_VITS_TEST_MODEL` / `SYNTH_VITS_VCTK_TEST_MODEL`) and building `synthesize-check-integration`.

CUDA development uses the committed presets (`cmake --preset dev-linux-x86_64-cuda` for RTX/sm_89, `dev-dgx-spark` for DGX Spark/sm_121a), each with matching `--build` and `ctest` presets. CUDA builds require CUDA Toolkit 13.3 *exactly* and default to strict FP32 cuBLAS math (`SYNTH_CUDA_TF32=OFF`; GGML's own tensor-core kernels still use TF32 for F32 matmuls with ≤16 columns — see `docs/backends.md`). Release presets pin exact cubin sets and run a no-PTX verifier — never replace them with `native`/`all`. Full details and the manylinux wheel-build commands: `docs/testing.md`.

## Project Language

`CONTEXT.md` defines the project's canonical vocabulary with explicit "avoid" terms. Use it in code, docs, commits, and discussion — e.g. "Synthesis" not "generation", "Chunked Audio Delivery" vs "Native Streaming Synthesis" (they are different claims), "Model Family" vs "Model Variant", "Execution Backend" vs "Native Provider", "Voice Profile" vs "Preset Voice". `docs/adr/` records the architectural decisions behind these distinctions.

## Architecture

- **Public seam**: `include/synthesize.h` is a versioned C ABI with opaque handles (`synth_model_t`, `synth_context_t`, `synth_voice_profile_t`) and `struct_size`-tagged structs. C++ types, GGML tensors, and family internals never cross it. `SYNTH_VERSION_*`/`SYNTH_ABI_VERSION` in this header are the single source of truth — CMake and `pyproject.toml` both parse the version out of it.
- **Core runtime** (`src/`): family-independent modules — synthesis request validation, audio delivery/sinks, backend device/module/plan management, text frontend dispatch, voice profiles, deterministic random streams.
- **Model families** (`src/arch/<family>/`, currently `vits/` and `kokoro/`): each inference stage is a pair of files — the GGML graph builder (e.g. `text-encoder.cpp`) and a `-host` counterpart for CPU-side orchestration. A Loaded Model is immutable and shareable; a Synthesis Context holds the mutable per-synthesis state. Family internals are private; tests may reach them, but they are never exposed through the C interface.
- **Vendored ggml** (`ggml/`): a verbatim snapshot pinned by `ggml/UPSTREAM`, not a submodule. Never hand-edit or reformat it; re-vendor with `scripts/sync-ggml.sh <ref>`.
- **Python packaging** is split (ADR 0014): `bindings/python` is the single abi3 API wheel (`synthesize_cpp`); `bindings/python-native*` are Native Provider wheels (default CPU, cu13 CUDA) discovered via the `synthesize_cpp.native` entry point. Providers ship the shared library plus dynamically loadable GGML backend modules (`SYNTH_GGML_BACKEND_DL`).
- **Porting pipeline**: new models follow the staged workflow in `docs/model-porting.md` / `docs/model-import-workflow.md` (intake → oracle → convert → cpp → port-validate → quants → backends → ship). Python conversion/validation tooling in `scripts/` runs in the locked per-family uv environment (`scripts/envs/vits/`), and the seven `validate-vits-*.py` scripts compare stage-by-stage against the PyTorch oracle.

## Testing Policy (enforced, see docs/testing.md)

Testing is a per-slice completion gate: a converter rule, graph stage, runtime control, or public-interface change is not done until its focused unit tests (valid path, limits, malformed input, error mapping) are committed and passing, registered with CTest under the `unit` label. Changes to model semantics also extend the family Golden integration test. The core rule for artifacts: **commit golden contracts, not golden payloads** — `tests/golden/vits/*.manifest.json` pin provenance, cases, and tolerances; models, reference tensors, and generated reports stay ignored.

## Documentation Conventions

Docs in `docs/` are confirmed contracts, each carrying a `Status: Confirmed …` line with a date; update that line when changing one. ADRs in `docs/adr/` are numbered and append-only in spirit — add a new ADR rather than rewriting a decision.
