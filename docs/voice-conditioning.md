# Voice Conditioning Survey

Status: Research snapshot on 2026-07-21; the public source set, all four source-specific C representations, Reference Audio normalization, the Serialized Profile format and compatibility rules, and the declaration rule for invertible derived conditioning stated below are confirmed; amended 2026-08-14 for the qwen3-tts ICL Profile's recoverable transcript.

## Purpose

This survey classifies how current TTS Model Families obtain Voice identity. It separates external Voice sources from family-specific internal representations and from controls such as emotion, prosody, speed, and recording environment.

## Observed Voice Sources

1. **No selectable Voice.** A single-speaker Model Variant has one fixed Voice. It is represented as the package default rather than a special runtime source.
2. **Preset speaker identity.** The Model Package exposes a speaker ID, name, or learned table entry. Piper is a representative speaker-ID implementation.
3. **Reference speech enrollment.** One or more clips, sometimes with transcript and language, condition or clone an unseen speaker. XTTS supports one or multiple clips; Qwen3-TTS and GPT-SoVITS demonstrate transcript-assisted reference conditioning.
4. **Direct speaker representation.** A caller or speaker encoder supplies an x-vector, d-vector, speaker embedding, or other prepared conditioning. The representation is family-specific and should remain hidden by a Voice Profile rather than cross the public Interface as an arbitrary tensor.
5. **Acoustic-token or in-context prompt.** Codec-language models condition on discrete acoustic tokens or a multimodal speech prefix. VALL-E is the canonical example. This differs internally from a fixed speaker embedding but can still be prepared and retained inside a Voice Profile.
6. **Natural-language Voice design.** A model synthesizes an identity from a textual description rather than an enrolled speaker. Qwen3-TTS VoiceDesign and Parler-TTS expose this source directly.
7. **Stochastic Voice generation.** A model samples an identity or style latent from a seed without reference speech. Tortoise exposes a random Voice mode, while StyleTTS 2 models style as a latent random variable.
8. **Multiple-reference fusion or Voice mixing.** A profile may be prepared from several clips or multiple speaker sources. GPT-SoVITS exposes auxiliary references for tone fusion. This is profile construction, not a new request-time Voice identity type.
9. **Speaker-specific weights, fine-tuning, or adapters.** Some systems adapt weights for a speaker rather than supplying inference-time conditioning. In synthesize.cpp v1, such weights form another Model Variant or Model Package; they are not a Voice Profile unless a future validated adapter format is introduced.

## Not Voice Identity

Emotion, accent, rhythm, pauses, intonation, speaking rate, and recording environment may be entangled with a reference in some models, but they are not automatically part of Voice identity. OpenVoice explicitly separates tone-color cloning from style controls, while VALL-E can preserve emotion and acoustic environment from its prompt. Instruction-following and inline control tags therefore belong to synthesis controls or a future expressive-conditioning interface, not the Preset Voice Catalog.

Voice conversion also consumes source speech, but changes existing speech rather than synthesizing linguistic input. It remains outside the current Synthesis scope.

## Interface Consequences

- The **Preset Voice Catalog** enumerates only named Model Package entries. It excludes runtime Voice Profiles and may be empty when a fixed single-speaker Model Variant still has an unnamed package-default Voice.
- A **Voice Profile** is the common immutable runtime result of supported preparation sources, including reference speech, direct family-owned conditioning, text design, stochastic generation, or fusion.
- The Loaded Model advertises which profile source kinds it supports. Unsupported sources fail explicitly.
- Raw family tensors, codec tokens, and GGML objects do not cross the public C seam.
- Speaker-specific model weights remain Model Packages in v1.
- The public Interface guarantees reusable prepared conditioning, not universal physical zero-copy across formats and memory domains.
- One Loaded Model-level capability query reports supported public Profile sources, per-field transcript and language requirements, maximum reference count, Profile Schema, and Profile Compatibility ID; it does not enumerate Voices.

## Confirmed Public Profile Sources

The common Interface exposes four source semantics: **Reference Audio**, **Description Text**, **Random Seed**, and **Serialized Profile**. Loaded Model capabilities determine which sources are accepted. Preset Voice selection remains a separate `voice_id` path; multiple references and fusion belong inside Reference Audio; family-specific embeddings and acoustic tokens remain private representations; and speaker-specific weights remain Model Packages.

The C representation must remain friendly to direct C use, generated Rust FFI, and Python extension or CFFI adapters. In particular, source data is expressed as explicit borrowed byte, text, PCM, or array views for a synchronous preparation call, while the returned Voice Profile owns or retains the prepared representation.

Each confirmed source uses a separate public function. The Interface does not expose a common tagged union or untyped source descriptor; the functions converge inside the Voice Profile Module and always return the same opaque Voice Profile type.

Reference Audio is represented by one or more size-tagged descriptors over borrowed contiguous interleaved F32 PCM. Each descriptor may also carry a borrowed length-delimited transcript and language tag. A caller-supplied descriptor stride preserves append-only ABI evolution for arrays. The core Interface does not decode paths, URLs, or encoded containers, and the prepared Voice Profile does not retain the source views after the synchronous preparation call.

The Voice Profile Module accepts a common 8–192 kHz mono-or-stereo F32 input contract and owns conversion to the Model Variant's declared target sample rate and channel count. Capability fields expose per-clip and total Reference Frame Equivalent limits. Limits are preflighted with checked arithmetic; references are never silently trimmed or discarded.

## Confirmed Reference Audio Normalization

One private Audio Normalizer Module centralizes validation, channel conversion, sample-rate conversion, buffer ownership, and resampler error translation. Its Interface is internal to the Voice Profile Module: no libsamplerate type, algorithm selector, quality level, or streaming state crosses the public C seam or appears in Rust and Python bindings.

Validated builds vendor and statically compile pinned libsamplerate 0.2.2 source and use `SRC_SINC_BEST_QUALITY`; they do not resolve a system libsamplerate. The normalizer runs on CPU before any family Voice encoder or Execution Backend transfer. It processes in the fixed order validate, channel-convert, then resample; stereo-to-mono uses `(left + right) * 0.5`, mono-to-stereo duplicates the sample, and PCM already matching the target format remains eligible for direct borrowing without an extra PCM copy.

The normalizer uses the stateful full API, marks finite input with `end_of_input`, and drains the conversion while honoring the reported input-used and output-generated counts. Reference Frame Equivalents are computed as `ceil(input_frames * target_rate / input_rate)` for checked limit preflight and do not force the resampler's output length. The normalizer neither pads nor crops output merely to reach that equivalent. Pinning source, configuration, and processing order provides one validated numerical path, but cross-architecture behavior is judged by numerical tolerance rather than a bit-identity promise.

Description Text is one borrowed, non-empty, length-delimited UTF-8 prompt with an optional explicit description-language tag and a concrete preparation seed. Its natural-language schema stays Model Variant-defined. The description language is separate from synthesis output language, no automatic detection occurs, and the returned profile no longer depends on the prompt bytes.

Random Seed has no borrowed identity payload: one concrete `uint64_t` selects from the Loaded Model's own Voice distribution and produces a model-bound profile. The seed has no cross-model meaning and is separate from later synthesis randomness. Profile preparation does not accept `SYNTH_SEED_RANDOM`; Adapters generate and retain a concrete seed when nondeterministic selection is desired.

Serialized Profile uses paired memory import and export operations. Import borrows a non-empty opaque self-contained byte view for one synchronous call; export returns a library-owned read-only Byte Buffer. The representation may encapsulate family-specific conditioning but exposes no public tensor types, external resource references, or alternate Model Family dispatch path.

The wire representation is little-endian GGUF v3 with a project format version, versioned Model Family Profile Schema, Profile Compatibility ID, and whole-file SHA-256 integrity value. Profiles contain canonical backend-independent conditioning. **Profiles never carry enrollment audio.** They omit transcript and description source material by default, and where a Model Family Profile Schema instead declares derived conditioning that is invertible to that source material, the rule in the next paragraph governs it. Compatibility is exact and converter-declared; architecture names, shapes, filenames, and whole Model Package hashes are not substitutes.

**Invertible derived conditioning is declared, never implied.** A Model Family Profile Schema may carry conditioning that is derived from source material rather than being that source material, and from which the source is nonetheless reconstructible. Where it does, two things are required of it: the Schema declares that field explicitly, and this document names what a holder of the Profile can recover from it. One such field exists today. **A Serialized qwen3-tts ICL Profile carries reference text token ids, byte-level BPE is invertible, and whoever holds the Profile can recover the reference transcript.** The field is `profile.reference_text_ids`, declared by the `icl` kind of Profile Schema `qwen3-tts-voice-clone`; the x-vector kind of the same Schema carries no such field. Distributing an ICL Profile therefore distributes its reference transcript, and the point of stating it here is that doing so is an informed act. Enrollment audio remains excluded in both kinds and under every Schema.

## Representative Primary Sources

- [Piper training documentation](https://github.com/rhasspy/piper/blob/master/TRAINING.md)
- [Qwen3-TTS official repository](https://github.com/QwenLM/Qwen3-TTS)
- [CosyVoice official repository](https://github.com/FunAudioLLM/CosyVoice)
- [Parler-TTS official repository](https://github.com/huggingface/parler-tts)
- [VALL-E project page](https://www.microsoft.com/en-us/research/project/vall-e-x/vall-e/)
- [StyleTTS 2 official repository](https://github.com/yl4579/StyleTTS2)
- [Tortoise TTS official repository](https://github.com/neonbjb/tortoise-tts)
- [XTTS official documentation](https://github.com/coqui-ai/TTS/blob/dev/docs/source/models/xtts.md)
- [OpenVoice official repository](https://github.com/myshell-ai/OpenVoice)
- [GPT-SoVITS official inference interface](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/api_v2.py)
- [libsamplerate official repository](https://github.com/libsndfile/libsamplerate)
- [libsamplerate full API and finite-input contract](https://libsndfile.github.io/libsamplerate/api_full.html)
- [libsamplerate transport-delay FAQ](https://libsndfile.github.io/libsamplerate/faq.html)
