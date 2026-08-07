# Project Scope

Status: Confirmed, last updated on 2026-08-07.

## Purpose

synthesize.cpp is primarily an embeddable, local, offline, native C/C++ text-to-speech inference library for other programs. Its official runtime uses GGML and GGUF and supports multiple explicitly ported and validated model families. The command-line program is a reference client of the library rather than a separate or privileged product surface.

## In Scope

- A mandatory CPU inference baseline.
- GPU acceleration enabled only for model and backend combinations that pass complete inference validation.
- Model conversion and profile-based selective inference quantization tooling.
- A stable, embeddable C interface as the primary integration surface.
- A command-line reference client implemented exclusively through that C interface.
- Text and phoneme synthesis inputs.
- Model-selected text frontends with direct phoneme and token-sequence bypasses.
- Declarative Model Packages containing one primary GGUF and optional non-executable Sidecar Resources.
- Compact, deterministic Port Validation Suites that establish conversion and
  end-to-end inference support without claiming corpus-scale perceptual quality.
- Publishing validated, directly loadable Model Packages through project-owned
  Hugging Face model repositories.
- Publishing a Restricted Model Package through the same repository and
  validation flow for a Model Family whose upstream weight terms are
  redistribution-restricted, carrying those terms in full (ADR 0018). This is
  an addition to the line above, not a reinterpretation of it: a Restricted
  Model Package is not a Published Model Package and never carries its
  embeddable-in-other-programs promise.

## Out of Scope

- Model training or fine-tuning.
- Automatically running arbitrary Hugging Face TTS models.
- Requiring a network or hosted inference service at runtime.
- Loading executable code or implicitly downloading resources from a Model Package.
- Claiming backend support based only on compilation or partial operator coverage.

## Deferred

- Corpus-scale Quality Evaluation Suites and cross-model quality comparison.
- Downloading and hashing LJSpeech, VCTK, or other quality-reference corpora.
- Pinning ASR, UTMOSv2, ECAPA-TDNN, resampling, and other evaluator artifacts.
- Measuring and freezing perceptual or quantization-quality thresholds.

## Unresolved

- Remaining synthesis controls, remaining Model capability queries, and ABI compatibility test fixtures.
