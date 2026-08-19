# Model Package Format

Status: Confirmed, last updated on 2026-08-20.

## Package Boundary

Each model variant has one primary GGUF containing every tensor required by its synthesis inference graph and the immutable metadata needed to interpret those tensors. Optional Sidecar Resources are permitted for frontend dictionaries, normalization data, attribution, license text, and other declarative resources that are unsuitable for embedding in GGUF.

A Text Frontend Provider is installed code, not a model resource. The package declares the provider identity and required resources but cannot supply or load provider binaries.

## Primary GGUF

The primary GGUF is authoritative for:

- architecture and format versions;
- tensor names, shapes, types, and quantization information;
- inference hyperparameters and audio configuration;
- symbol or token tables required by the synthesis graph;
- speaker and voice metadata required for inference;
- required Text Frontend Provider identity and Sidecar Resource descriptors;
- source, license, conversion, and provenance metadata.

Splitting one synthesis graph's tensors across multiple GGUF shards is outside the initial format.

## General Synthesis Capabilities

The primary GGUF declares supported input forms, native output sample rate and channel count, the positive maximum final input-token count, the positive maximum native output-frame count, whether synthesis may consume randomness, and any validated non-default speaking-rate range. These values are semantic Model Variant metadata shared by its validated precision and quantization packages rather than inferred from tensor shapes at runtime.

Text input also depends on the required Text Frontend Provider being available when the model is loaded. The Loaded Model capability snapshot therefore reports currently usable input forms, while package-declared but unavailable text input fails with an explicit frontend error. Native Streaming Synthesis is additionally conditioned on the selected Execution Backend and is exposed only for a validated package-backend combination.

## Preset Voice Catalog

The primary GGUF declares the immutable Preset Voice Catalog as an ordered set of entries. Each entry has a non-empty UTF-8 identifier unique within the Model Variant, an optional human-readable display name, and append-only flags. The converter preserves identifiers across F32, F16, and quantized Model Packages representing the same logical Model Variant; a display name and array index are not stable selection identities.

At most one entry is marked as the Preset Voice default. A Model Variant may instead have an unnamed package default that is not selectable by identifier, including a fixed single-speaker Voice, so a usable default does not imply a non-empty Catalog. The package does not place runtime Voice Profiles in this metadata and does not infer language, gender, age, or accent attributes for Catalog entries.

## Language Capability Catalog

The primary GGUF declares at least one validated language capability as a canonical, unique BCP 47 tag with append-only flags. At most one entry is the package default. The converter preserves the exact capability set across quantized Model Packages that represent the same validated Model Variant and never derives entries from an architecture name, tokenizer vocabulary, training dataset label, or Preset Voice metadata.

A base tag may declare narrow same-language regional fallback. The flag is invalid on a locale-specific tag and does not authorize script, variant, extension, private-use, or cross-language fallback. Catalog order is immutable within a Loaded Model but is not a persistent identity; callers persist canonical tags.

## Voice Profile Compatibility

A Model Package that accepts Serialized Profiles declares its Profile Schema, Profile Schema version, and 32-byte Profile Compatibility ID in the primary GGUF. The compatibility ID represents the meaning and consumer contract of prepared Voice conditioning, not the bytes of this quantized Model Package. Validated F32, F16, and quantized packages derived from the same compatibility-critical source may therefore share the identifier across Execution Backends.

The converter computes the identifier as SHA-256 over the Model Family's canonical compatibility manifest, including upstream checkpoint provenance and fingerprints of Voice-conditioning configuration and source weights. If compatibility cannot be demonstrated, the converter emits a distinct identifier. Model loading never infers profile compatibility from filenames, architecture labels, tensor dimensions, `general.uuid`, or approximate metadata.

A Model Package that can prepare a Voice Profile at all declares which sources it implements positively, in `synthesize.voice.profile_sources`. This is a declared list, not an inference: a Model Family that offers more than one incompatible source under the same Voice Mode -- Reference Audio for one Model Variant, Description Text for another, both reported under `profile-sources` -- cannot be told apart by Voice Mode alone, so the declaration exists to say which one a given package means.

Loading validates this declaration only as far as the package's structure lets it. A source with a distinguishing tensor block is cross-checked against that block's presence: Reference Audio requires the Voice encoder, so a package declaring it without carrying the encoder is refused, and a package carrying the encoder without declaring it is refused the same way. A source with no distinguishing tensor block cannot be corroborated and is taken at its word -- Description Text has no tensors of its own, since the instruct text it conditions on is tokenized and fed to the Talker the same way ordinary request text is, so nothing in the package's shape can confirm or contradict the claim. A Model Variant's identity string (`synthesize.model_variant`) is surfaced to callers and is also a validated input, cross-checked at load against the Voice Mode and the declared sources. Only the string's final segment -- the Model Variant kind -- is read; a string with no separator is its own final segment and is read the same way, so a Model Variant named exactly `base` claims the base kind. A recognized kind fixes the Voice Mode the package must declare and, under `profile-sources`, the single source it must name; a package that disagrees is refused. Reading the kind rather than the whole string is what keeps this from being an enumeration lock: a Model Variant differing from a published one only in size or frame rate (`qwen3-tts-24hz-3b-voicedesign` against `qwen3-tts-12hz-1-7b-voicedesign`) is governed by the kind the two share, while a kind the build does not recognize is governed by nothing and loads untouched. That last property collides with the published-filename convention below, and the collision is deliberate rather than an oversight: a published file is named `<variant-slug>-<QUANT>.gguf`, so its FILENAME's final segment is a Quantization Profile, not a kind. The check never reads a filename -- it reads the `synthesize.model_variant` metadata string, which carries the unsuffixed slug in every published package including the quantized ones. A variant string that did end in a profile name would simply present an unrecognized kind and load ungoverned, which is the safe direction. No fixed table of whole Model Variant strings is maintained, which is deliberate -- one would refuse a legitimate future variant purely for not having been updated to name it. What the check establishes is that two of the package's own declarations agree, which makes it a refusal of a *mislabelled* package rather than of an unimplemented one: a Model Variant kind is a string, not a tensor, so it is not evidence that an implementation is present. Description Text still has no tensor footprint to corroborate, and a package declaring it under an agreeing kind is still taken at its word for the reason given above. The converter refuses the same disagreement before writing it, so a package produced by this project cannot carry one. A package with a Preset Voice Catalog and no Voice Profile contract at all has nothing to declare a source for; loading refuses `synthesize.voice.profile_sources` outright if such a package carries the key, rather than reading a declaration that has no contract behind it and silently ignoring it.

A package that declares Reference Audio also declares the Voice encoder's target sample rate and channel count, minimum and maximum Reference Frame Equivalents per clip, maximum total Reference Frame Equivalents, and maximum reference count. These are mandatory nonzero safety and capability values rather than advisory UI metadata. The runtime derives each equivalent from input duration with checked arithmetic before conversion; it does not treat the value as the exact number of frames that the private resampler must emit. Validation is identical for every Execution Backend.

## Sidecar Rules

Every Sidecar Resource is identified declaratively with a package-relative path, content type, size, checksum, purpose, and applicable license metadata. The primary GGUF's parent directory is the Package Root. Package loading resolves declared Sidecar Resources relative to that directory and rejects missing or mismatched resources and paths that escape it; it does not discover resources by scanning the directory.

Sidecars cannot contain executable code, shared libraries, scripts that the runtime executes, or instructions to fetch resources from a network. The runtime performs no implicit network access; resource acquisition is an explicit operation outside model loading.

## Publication and Acquisition

Official runnable models follow transcribe.cpp's Hugging Face layout while using
the maintainer's `jiangzhuo9357` personal namespace. The released artifact is the
converted primary GGUF plus every declared Sidecar Resource needed by
synthesize.cpp; an upstream framework checkpoint is conversion input and is not
presented as a runnable synthesize.cpp model.

### Repository Layout

Each Model Variant has one repository named
`jiangzhuo9357/<variant-slug>-gguf`. All published precision and Quantization
Profiles are flat files at its root using the llama.cpp-style
`<variant-slug>-<QUANT>.gguf` convention, such as `F32`, `F16`, or `Q8_MIXED`.
The filename carries the stable profile name while GGUF metadata carries its
version and exact tensor assignment. A family whose profiles do not all
quantize the same half of the model names the half in the profile: OmniVoice's
generator-half profiles take the plain name and its codec-half profiles take a
`_CODEC` qualifier (`F16_CODEC`, `Q8_CODEC_MIXED`). See
`docs/quantization.md`. Variants, quantizations, and Execution
Backends do not receive nested repositories or backend-specific copies.

The root also contains `README.md` and any license or attribution material required
by the source model. When a TTS package needs Sidecar Resources, they are flat,
stable-named root files shared by the applicable GGUFs; each primary GGUF remains
authoritative for their paths, sizes, and checksums. This is the only TTS-specific
extension to transcribe.cpp's otherwise self-contained GGUF repository layout.

### Model Cards and Evidence

The source tree keeps one YAML model-card specification per variant and renders the
Hugging Face `README.md` beside the generated GGUFs before upload, following
transcribe.cpp's `scripts/hf_cards/` workflow. Its metadata declares the upstream
base model, license, `library_name: synthesize.cpp`, text-to-speech pipeline,
languages, and relevant tags. A project-specific `synthesize_cpp` metadata block
declares the Validation Level and quality-evaluation status, and may carry
available quality, performance, and capability summaries.

The committed contracts, generated payloads, report paths, and eight porting stages
are defined in `model-porting.md`.

The rendered card records the pinned upstream revision and date, the synthesize.cpp
revision and reference implementation used for validation, its Validation Level,
an explicit quality-evaluation status, a direct-download table for every published
Quantization Profile, usage, license, and the pinned upstream model card for
offline reference. A corresponding source-controlled
`docs/models/<variant-slug>.md` contains the Port Validation Suite evidence,
operational measurements, reproduction instructions, and any available quality
results. Direct links use
`jiangzhuo9357/<variant-slug>-gguf/resolve/main/<variant-slug>-<QUANT>.gguf`, as
in transcribe.cpp.

Committed golden manifests pin source and reference revisions and Port Validation
Suite cases; converter reports record input and output SHA-256 values. Generated
tensors, GGUFs, validation payloads, and benchmark reports stay outside Git. As in
transcribe.cpp,
the Hugging Face repository does not add a separate package index, `SHA256SUMS`, or
release-record bundle: the model-card validation sentence and Hugging Face revision
history identify what was published.

Publication uses the same prepared-directory operation as transcribe.cpp:
`hf upload jiangzhuo9357/<variant-slug>-gguf models/<variant-slug> .`. License
metadata in the card does not create or alter redistribution rights. The project
records the available upstream evidence verbatim. When an official checkpoint has
no separately stated weight terms, the maintainer's default publication policy is
to treat that absence as imposing no additional restriction beyond the published
source-model terms. The generated card must disclose both the missing separate
statement and this project-policy assumption. Contrary evidence still blocks or
removes the affected artifact.

### Restricted Model Packages

A Model Family whose upstream weights carry a non-commercial or otherwise
redistribution-restricted grant may still be published, as a Restricted Model
Package rather than a Published Model Package (ADR 0018) — an addition to the
publication flow above, not a reinterpretation of it. A Restricted Model
Package passes the same Port Validation Suite, uses the same repository
layout and model-card workflow, and reaches the same Validation Levels as any
other package, but its card declares the restrictive license in frontmatter,
quotes the upstream statement verbatim, and states plainly that the weights
are not usable in commercial products. It is never called a Published Model
Package and never carries this document's embeddable-in-other-programs
promise. Publication of either kind remains a separate act requiring
jiangzhuo's explicit per-act confirmation.

Acquisition is explicit and occurs before model loading. Documentation and CI use
the Hugging Face CLI to download selected files and cache canary files where
appropriate:

```bash
hf download jiangzhuo9357/<variant-slug>-gguf <filename> \
  --local-dir models/<variant-slug>
```

A caller downloads its chosen GGUF and all declared Sidecars into one local Package
Root.
The core library, CLI inference path, Rust Adapter, and Python Adapter neither
contact Hugging Face nor accept a repository identifier in place of a local
primary-GGUF path. No workflow uses remote code or `trust_remote_code` to load a
Published Model Package.
