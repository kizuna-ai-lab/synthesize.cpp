# Speech Synthesis

This context defines the language used by synthesize.cpp when turning linguistic input into speech audio. It distinguishes deliberately supported synthesis models from models that merely share a file format or upstream framework.

## Language

**Synthesis**:
The production of speech audio from linguistic input using a loaded model.
_Avoid_: Generation, TTS operation

**Audio Sink**:
A caller-selected destination for synthesized PCM frames, such as a complete buffer or callback adapter.
_Avoid_: Native streaming

**Chunked Audio Delivery**:
Delivery of synthesized audio to an Audio Sink in multiple blocks, without promising that generation itself is incremental.
_Avoid_: Native Streaming Synthesis

**Native Streaming Synthesis**:
A validated capability of a model variant to produce usable audio incrementally before its complete utterance waveform has been generated.
_Avoid_: Chunked Audio Delivery

**Linguistic Input**:
The text or phoneme sequence supplied to a synthesis request.
_Avoid_: Prompt

**Text Frontend**:
The model-selected transformation from text or phonemes into the exact symbol or token sequence expected by a model variant, including applicable normalization, grapheme-to-phoneme conversion, and symbol mapping.
_Avoid_: Model-family language logic, universal tokenizer

**Text Frontend Provider**:
A named built-in or optional component that supplies the Text Frontend required by one or more model variants.
_Avoid_: Mandatory G2P engine, model family

**Model Family**:
A group of synthesis models that share an inference architecture and can use the same family implementation.
_Avoid_: Model type, backend

**Supported Model Family**:
A model family that has been explicitly ported and passed the Port Validation Suite for at least one Reference Model Variant on CPU.
_Avoid_: Compatible model, GGUF model

**Model Variant**:
A specific checkpoint and configuration within a model family, with its own voices and language capabilities.
_Avoid_: Model family, architecture

**Reference Model Variant**:
A model variant designated as the validation oracle for a family implementation. Its selection reduces validation complexity and does not establish a product or language priority.
_Avoid_: First supported language, benchmark model

**Model Package**:
The complete local artifact set for one model variant, consisting of one primary GGUF and any declared Sidecar Resources.
_Avoid_: Arbitrary model directory, executable bundle

**Published Model Package**:
A Model Package that has at least passed the Port Validation Suite and whose exact bytes, provenance, license, and Validation Level are released through a project-owned model repository.
_Avoid_: Raw checkpoint, unvalidated conversion, Native Provider

**Loaded Model**:
An immutable, shareable runtime instance of a Model Package on one Execution Backend.
_Avoid_: Model Package, Synthesis Context

**Synthesis Context**:
The mutable runtime state used for one concurrent synthesis at a time with a Loaded Model.
_Avoid_: Loaded Model, global session

**Sidecar Resource**:
A non-executable local artifact declared by a Model Package but stored outside its primary GGUF, such as a frontend dictionary or license notice.
_Avoid_: Plugin binary, implicit download

**Quantization Profile**:
A named and versioned assignment of storage types and required compute precision to tensor groups of a model variant, validated as an independent Model Package.
_Avoid_: File suffix, quantize everything

**Port Validation Suite**:
A compact, versioned set of deterministic upstream and project-authored inference cases that proves model conversion, family implementation, synthesis controls, and Execution Backend behavior against a pinned reference within declared tolerances.
_Avoid_: Quality Evaluation Suite, corpus benchmark, load-only smoke test

**Port Validation Case**:
One deterministic synthesis scenario in a Port Validation Suite, with pinned input, Voice, controls, randomness, expected evidence, and checks.
_Avoid_: Quality sample, benchmark corpus item

**Golden Manifest**:
The committed, versioned contract that pins one Model Variant's source provenance, reference implementation, Port Validation Cases, expected evidence, and tolerance policy without containing the generated model, tensor, or audio payloads.
_Avoid_: Golden payload, Model Package, benchmark report

**Validation Level**:
The strongest evidence completed for a Published Model Package: `port_validated` after the Port Validation Suite, or `quality_evaluated` after a later versioned Quality Evaluation Suite.
_Avoid_: Quality score, model ranking

**Quality Evaluation Suite**:
A later, locked collection of corpus-scale evidence axes used to judge a port-validated Model Package, including intelligibility, naturalness, applicable Voice similarity, and comparative quality drift. It may include optional Listening Audit evidence, and its results are not collapsed into one universal score.
_Avoid_: TTS score, universal quality metric

**Listening Audit**:
An optional, non-statistical maintainer review of a small automatically selected A/B sample set that records obvious regressions without claiming population-level subjective quality.
_Avoid_: MOS study, listening panel, subjective release gate

**Quality Baseline**:
The reference-dtype Model Package for the same Model Variant against which a candidate Quantization Profile's quality drift is evaluated.
_Avoid_: Cross-family quality target, universal baseline

**Execution Backend**:
A GGML execution target, such as CPU, CUDA, Metal, or Vulkan, used to run a model family's shared inference graph.
_Avoid_: Model family, GPU implementation

**Native Provider**:
An installed distribution artifact set that supplies a compatible native runtime to a language Adapter; one Native Provider may expose multiple Execution Backends.
_Avoid_: Backend package, Model Package

**Minimum Build Toolchain**:
The oldest documented tool set for which building synthesize.cpp from source is supported.
_Avoid_: Release Toolchain Lock, Runtime Support Floor

**Release Toolchain Lock**:
The exact tool and dependency identities used to construct one official release.
_Avoid_: Minimum Build Toolchain, Runtime Support Floor

**Native Cubin Target Set**:
The release-pinned real CUDA architectures for which an official CUDA Native Provider embeds compiled device code; PTX-only JIT compatibility is not part of this set.
_Avoid_: Runtime Support Floor, Validation Hardware Matrix, PTX fallback

**Runtime Support Floor**:
The minimum host software and device capabilities required to use a published Native Provider without building it.
_Avoid_: Minimum Build Toolchain, Validation Hardware Matrix

**Validation Hardware Matrix**:
The physical hardware and host combinations on which a release has passed its required validation; it is evidence for support, not a device allowlist.
_Avoid_: Supported-device list, Runtime Support Floor

**Validated Backend Combination**:
A model-family and Execution Backend pairing that has passed the applicable Port Validation Suite correctness, stability, and backend-execution gates for its declared model variants.
_Avoid_: Compiled backend, operator coverage

**Voice**:
The perceived speaker or timbre identity used for Synthesis, whether supplied by a Model Package or prepared at runtime.
_Avoid_: Character, persona, style

**Preset Voice**:
A named Voice declared by a Model Package and selected by its stable identifier.
_Avoid_: Voice Profile, custom voice

**Preset Voice Catalog**:
The immutable set of Preset Voices declared by one Model Variant.
_Avoid_: All voices, runtime voice registry

**Voice Reference**:
Caller-supplied enrollment material, such as one or more speech clips and any required transcripts, used to prepare a Voice Profile.
_Avoid_: Preset Voice, Linguistic Input

**Reference Frame Equivalent**:
The duration of a Voice Reference expressed as a frame count at the Model Variant's target sample rate for limit checking; it is not the number of frames an internal resampler must emit.
_Avoid_: Resampled frame count, normalized frame count

**Voice Profile**:
An immutable, reusable, model-bound runtime representation of Voice conditioning prepared from a supported source.
_Avoid_: Preset Voice Catalog, raw speaker embedding, Synthesis Context

**Serialized Profile**:
A self-contained persistence representation of a Voice Profile that a compatible Loaded Model can import.
_Avoid_: Model Package, raw speaker tensor, backend snapshot

**Profile Compatibility ID**:
An identity shared only by Model Packages that can safely consume the same Serialized Profile representation.
_Avoid_: Model UUID, model filename, architecture name, whole-file checksum

**Mandarin Chinese**:
The project's canonical meaning of Chinese language support, targeting Standard Mandarin (`zh-CN`). It does not include Cantonese or other Sinitic varieties.
_Avoid_: Chinese dialects, all Chinese

**Language**:
A linguistic system a model can synthesize, identified at the broadest level supported by evidence, such as `en` for unspecified English.
_Avoid_: Locale, accent

**Language Capability**:
A language that a particular model variant has declared and passed project validation for.
_Avoid_: Family language, project language

**Locale**:
An optional, evidence-backed regional language variant such as `en-US` or `en-GB`.
_Avoid_: Language, accent

**Accent**:
A voice's pronunciation character as declared by reliable model or training documentation; it remains unknown when not documented.
_Avoid_: Locale, inferred nationality

**Core European Set**:
The first non-English European language priority group: French (`fr`), German (`de`), Spanish (`es`), Italian (`it`), and Portuguese (`pt`).
_Avoid_: Other European languages, all European languages
