# Model Import Workflow

Status: Confirmed, last updated on 2026-07-24.

This document is an operational guide distilled from the VITS port. It walks
through importing either:

- a new Model Variant within an already supported Model Family; or
- a new Model Family / framework with its first supported Variant.

[`model-porting.md`](model-porting.md) remains the canonical workflow and
publication contract. This guide reuses its stage names and records the
practical order of operations observed while porting VITS; where the two
documents disagree, `model-porting.md` wins.

The workflow follows the repository's existing Port Validation and publication
discipline:

- commit contracts, not large generated payloads;
- validate CPU correctness before quantization or backend expansion;
- keep quality evaluation separate from functional port validation; and
- require tests before advancing to the next implementation slice.

## Scope

This workflow applies to TTS model import, including any required text frontend,
voice conditioning, tokenization, checkpoint conversion, quantization, and
backend validation.

It does not define perceptual quality evaluation. Quality evaluation is a
separate, deferred process and is not required for `port_validated`
publication.

## Two Import Modes

### 1. Importing a New Model Variant

A new Model Variant reuses an already supported family architecture. Typical
examples are a new checkpoint, a new speaker catalog, or a new language
coverage set within the same family.

A new Quantization Profile is deliberately not a new Model Variant: it is an
independent Model Package of the same Variant. Adding one reruns only the
`6-quants` and `7-backends` stages plus publication for the affected packages.

The family contract is already known. The work is to pin the variant-specific
artifacts, prove parity, and publish the new package.

### 2. Importing a New Model Family

A new Model Family introduces a new architecture or a materially different
inference graph. It requires a new family contract before the first variant can
be ported.

The family contract defines the public capabilities, supported conditioning
sources, input forms, output format, and validation boundaries for that
architecture.

## Guiding Principles

- Architecture first, language second.
- Model variants are selected by capability, not by language category alone.
- Voice support is a property of the model variant.
- The public C Interface exposes stable contracts, not family internals.
- Family-specific tensor mapping and conversion logic remain in the family
  script, not in a generic converter framework.
- Testing is a per-slice gate inside every stage, not a separate stage: a slice
  is complete only when its focused tests are committed, registered with CTest,
  and passing (see [`testing.md`](testing.md)).
- Quantization, backend validation, and quality evaluation are separate stages.

## Workflow for a New Model Variant

The stages reuse the canonical [`model-porting.md`](model-porting.md) sequence:

```text
1-intake -> 2-oracle -> 3-convert -> 4-cpp ->
5-port-validate -> 6-quants -> 7-backends -> 8-ship
```

### Stage 1: Intake

Record the immutable identity of the variant:

- upstream repository and revision;
- checkpoint location and checksum;
- config location and checksum;
- license status and publication constraints;
- target family;
- supported languages;
- voice model type;
- text frontend requirements;
- output format and sample rate;
- any variant-specific execution constraints.

Write the result to:

- `reports/porting/<family>/<variant>/intake.json`

Also create the porting log now and maintain it through every following stage:

- `reports/porting/<family>/<variant>/_porting-log.md`

`testing.md` requires every skipped test to record an explicit blocker in this
log, and Golden evidence runs are recorded here as they happen. The log is a
working document, not a Ship-stage artifact.

If the variant requires a Text Frontend Provider that is not implemented,
record that in intake and continue. Following the VITS precedent, the port
proceeds with the input forms that are actually usable — at minimum exact
token IDs, plus phonemes when the built-in `synthesize.symbol_map` contract
applies — and the published package advertises only those forms. Implementing
a new Text Frontend Provider is its own separately validated slice and never
blocks conversion, parity, or a phoneme/token-only publication.

### Stage 2: Oracle and Golden Manifest

Materialize the committed Golden Manifest first; the oracle runs the
manifest's cases, so the cases must exist before the oracle:

- `tests/golden/<family>/<variant>.manifest.json`

The manifest conforms to
`docs/schemas/synthesize-golden-manifest-v1.schema.json` and must pin:

- source identity;
- reference implementation identity;
- validation cases;
- random-seed behavior;
- voice and language selections;
- speaking-rate rules;
- output limits;
- tolerance file.

Begin with representative upstream examples, preserving their exact inputs,
then add the smallest number of cases needed to cover the remaining inference
branches; a manifest normally contains 12 to 32 cases.

`suite_version` starts at 1 for a new variant. Any later manifest change that
alters what validation proves — cases, relations, the tolerance-file
reference, package-contract limits, or source/reference pins — increments it,
and validation reports record the `suite_version` they ran against.

Then run the pinned reference implementation for those cases and capture the
reference evidence needed for parity replay:

- resolved input;
- intermediate tensors;
- stochastic replay tensors when applicable;
- voice and language metadata;
- audio output;
- structured result metadata.

The oracle is validation evidence, not a product artifact.

### Stage 3: Convert

Produce the source/reference-dtype GGUF only.

The converter must:

- preserve authoritative metadata;
- map tensors deterministically;
- record skipped, fused, or transformed tensors;
- produce a reproducible report; and
- avoid embedding quantization policy.

Write the report to:

- `reports/convert/<family>/<variant>-<dtype>.json`

where `<dtype>` is the source/reference dtype — `F32` for both current VITS
variants, but not assumed by the convention.

### Stage 4: C++ Implementation

Implement or extend the family module in `src/arch/<family>/`, in bounded
slices.

This stage owns:

- GGUF loading and validation;
- tensor catalog checks;
- host-side shape and metadata preparation;
- the family inference graph;
- voice conditioning seams;
- backend-plan integration; and
- CPU correctness.

Every slice ends by committing its focused unit tests, registered with CTest
under the `unit` label, before the next slice begins. Required coverage
typically includes:

- valid paths;
- boundary checks;
- malformed input;
- error mapping;
- lifecycle rules;
- public capability reporting;
- voice or language routing;
- metadata and shape validation.

Run the unit gate in a sanitizer build for every slice that executes C or C++
inference code.

### Stage 5: Port Validation

Run the manifest cases end to end on CPU at the source/reference dtype and
establish tensor-by-tensor and waveform parity within the committed
tolerances.

Validation is driven the way the VITS port does it: one registered per-stage
validator `scripts/validate-<family>-<stage>.py` per graph stage, running in
the locked `scripts/envs/<family>/` environment, plus C++ Golden runners
registered with CTest under the `integration`, `golden`, and family labels.
Manual commands and generated comparison reports do not substitute for
registered tests.

Tolerances are measured from the first working reference and implementation,
reviewed, and committed to `tests/tolerances/<family>.json` before support is
declared. A port must not weaken a tolerance merely to accept its own failing
output.

### Stage 6: Quantization

After the source/reference-dtype path passes port validation, generate the
selected Quantization Profiles with the C++ `synthesize-quantize` tool; the
Python converter has no quantization option.

Quantization must be deterministic and must rerun the same Port Validation
Suite against the accepted reference path for every produced package.

### Stage 7: Backend Validation

Run the same validated cases on each claimed Execution Backend and record
actual placement and operational evidence. Only claimed backends are
validated: a CPU-only variant validates the CPU backend and may ship without
any GPU claim.

This stage validates support, not comparative quality.

### Stage 8: Ship

Prepare the publication artifacts:

- `docs/models/<variant>.md`
- `scripts/hf_cards/<variant>.yaml`
- generated HF README
- flat GGUF publication directory
- the completed `reports/porting/<family>/<variant>/_porting-log.md`

The generated package page must report the current validation level and the
quality-evaluation status explicitly.

## Workflow for a New Model Family

### Step 1: Select the Architecture

Choose a family based on architecture and validation feasibility, not on
language breadth.

The family should have:

- a stable and inspectable reference implementation;
- a representative end-to-end TTS graph;
- a practical CPU correctness baseline;
- tractable backend operator coverage;
- usable licensing for source, weights, and required frontend resources;
- reproducible conversion and quantization paths.

### Step 2: Define the Family Contract

Before implementing the first model, define the family-level contract:

- supported conditioning sources;
- supported voice types;
- language-handling rules;
- text frontend requirements;
- sample rate and output format;
- public capability queries;
- error and ownership rules;
- backend behavior boundaries.

This contract is what the public C Interface exposes indirectly through
versioned requests and capability queries. Family internals must remain private.

### Step 3: Choose the First Reference Variant

Start with the simplest useful Variant that represents the family well.

Preferred starting properties:

- stable upstream checkpoint;
- inspectable reference code;
- CPU-practical inference;
- minimal voice complexity;
- clear text-path coverage.

If the family is multilingual or multi-voice, do not attempt to validate every
language or voice at once. Validate the baseline variant first, then extend
coverage in later variants.

### Step 4: Build the Oracle and Manifest

Materialize the Golden Manifest for the first variant, then run the reference
implementation for its cases, exactly as in variant Stage 2.

The manifest must pin all inputs and all reference outputs needed for parity
replay.

### Step 5: Implement the Family Module

Create the family implementation under `src/arch/<family>/` and keep all
family-specific tensor mappings there.

The family module should own:

- weights;
- graph construction;
- host preparation;
- voice conditioning;
- backend-plan use;
- output layout;
- validation of family metadata.

### Step 6: Add Tests Before Advancing

Add tests at the smallest stable interface boundary:

- C/C++ unit tests for the family module;
- Python binding tests when bindings exist;
- golden integration tests for end-to-end parity;
- sanitizer runs for memory safety and undefined behavior;
- backend tests for each claimed backend.

### Step 7: Add Quantization and Backend Support

Only after the CPU reference path and validation suite are stable should the
family gain:

- F16 or mixed quantization profiles;
- CUDA support;
- Metal, Vulkan, or other backend support;
- performance measurements.

### Step 8: Publish the Family and Variants

Once the family has at least one validated variant, publish the family
documentation, variant documentation, and release artifacts.

Later variants then reuse the same family contract and extend it only where the
architecture requires.

## Required Deliverables

For a new Model Family, the minimum committed set is:

- `docs/porting/families/<family>.md`
- `docs/models/<variant>.md`
- `tests/golden/<family>/<variant>.manifest.json`
- `tests/tolerances/<family>.json`
- `reports/porting/<family>/<variant>/intake.json`
- `reports/porting/<family>/<variant>/_porting-log.md`
- `scripts/envs/<family>/pyproject.toml` and `scripts/envs/<family>/uv.lock`
- `scripts/convert-<family>.py`
- `scripts/dump_reference_<family>_<framework>.py`
- `scripts/validate-<family>-<stage>.py` for each validated graph stage
- `scripts/hf_cards/<variant>.yaml`

For a new Model Variant, the work usually reuses the existing family documents
and scripts and adds only the variant-specific entries and reports.

## Practical Rule of Thumb

If the work changes architecture, define or revise the family contract first.
If the work stays inside an established architecture, extend the existing family
contract and variant manifest.

In both cases, do not proceed to the next slice until the current slice's tests
are committed and passing.
