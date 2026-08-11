# Qwen3-TTS Stage 2 — Reference Audio Voice Cloning — Design

Status: Approved in discussion with jiangzhuo on 2026-08-11. This is the design
record for the second rung of the Qwen3-TTS Reference Model Variant Ladder
(`docs/porting/families/qwen3-tts.md`, "Reference Model Variant Ladder"). The
family record itself is extended at intake, per `docs/model-porting.md`.

## 1. Context

Stage 1 (`qwen3-tts-12hz-0.6b-customvoice`) is delivered and published: Preset
Voices through the existing `voice_id` path, F16 and Q8_MIXED Quantization
Profiles, a listening audit recorded as `no_obvious_regression`, and a live
package at `jiangzhuo9357/qwen3-tts-12hz-0-6b-customvoice-gguf` (created
2026-07-28, last updated 2026-07-29, carrying BF16, F16, and Q8_MIXED).

That status was itself established from the Hugging Face API rather than from
the repository, because the card specification's header claimed "Prepared, not
published" two weeks after the upload; both it and the family record's open
question 6 are corrected in the same commit as this spec. Stage 2 delivers its
artifacts to the same prepared-and-awaiting-confirmation point, and the upload
is a separate act as always.

Stages 2 and 3 were never started, which was confirmed by inspection rather
than by reading a status line:

- `src/voice-profile.cpp:669` guards every Voice Profile source on
  `model->info.family != synth::ModelFamily::Omnivoice`, so a Qwen3-TTS model
  takes the generic `SYNTH_ERR_UNSUPPORTED_VOICE` fallback;
- `src/arch/qwen3-tts/` contains no speaker encoder and no
  `SYNTH_PROFILE_SOURCE_*` reference at all;
- `scripts/convert-qwen3-tts.py` converts the CustomVoice checkpoint only;
- exactly one Golden Manifest, one intake report directory, and one Hugging Face
  card exist, all `…-customvoice`.

Two Stage 2 prerequisites recorded in the family plan's decomposition (item 7)
were paid down by the fourth Model Family rather than by this one:
`src/audio-normalizer.cpp` exists and `third_party/libsamplerate` is vendored.
What remains is this family's own conditioning path.

### 1.1 Upstream facts established at design time

Every fact below was read from the pinned upstream source or measured against
the actual checkpoint header. None is carried over from a port's documentation.

| Fact | Value | Source |
| --- | --- | --- |
| Reference Model Variant | `Qwen/Qwen3-TTS-12Hz-0.6B-Base`, `license: apache-2.0` | model card metadata |
| Revision | `5d83992436eae1d760afd27aff78a71d676296fc` | HF model API |
| Tensor inventory | 478 = 402 `talker.*` + 76 `speaker_encoder.*` | `model.safetensors` header |
| Speaker encoder shape | `blocks` / `res2net_block` / `mfa` / `asp` / `fc` — ECAPA-TDNN | tensor names |
| Speaker embedding width | `enc_dim: 1024`, `sample_rate: 24000` | `config.json` |
| Talker width | `hidden_size: 1024` | `config.json` |
| Speaker encoder input | 128-bin mel: `n_fft=1024`, `hop=256`, `win=1024`, `fmin=0`, `fmax=12000` at 24 kHz | `modeling_qwen3_tts.py:1941` |
| Clone modes | mutually exclusive: x-vector-only, or ICL requiring `ref_text` | `qwen3_tts_model.py:356` |
| ICL inputs | `speech_tokenizer.encode(...)` codes **and** reference transcript | `qwen3_tts_model.py:385` |
| Codec encoder half | 225 tensors: `downsample` 1, `encoder` 28, `encoder_transformer` 96, `quantizer` 100 | `Qwen3-TTS-Tokenizer-12Hz` header |
| Code group agreement | `encoder_valid_num_quantizers: 16` = `num_code_groups: 16` | tokenizer / talker configs |
| Frame geometry | `encode_downsample_rate` = `decode_upsample_rate` = 1920 | tokenizer config |
| Codec encoder input | raw 24 kHz waveform, not mel | `modeling_qwen3_tts_tokenizer_v2.py:961` |
| Preset Voices on Base | `spk_id = {}`, `spk_is_dialect = {}` — none | `config.json` |
| Reference clips per Voice | one; upstream's list argument is a batch of separate prompts | `qwen3_tts_model.py:356` |

Two consequences shape everything below.

**The x-vector is a drop-in.** `enc_dim` equals the Talker's `hidden_size`, and
upstream substitutes the speaker embedding into exactly the prompt position
where Stage 1 places the Preset Voice token embedding
(`modeling_qwen3_tts.py:2087-2105`). The rest of the prompt — think/nothink
prefix, language id, bos/eos/pad — is unchanged.

**Stage 2 extends the prompt prefix; it does not change the inference loop.**
The sampling chain, KV cache, Code Predictor inner loop, and RVQ codec decoder
validated in Stage 1 are untouched. This is the property that keeps the slice
tractable and it is the first thing to re-verify if the port drifts.

## 2. Decisions

**D1 — Both clone modes, x-vector first.** Stage 2 delivers x-vector-only and
ICL. Upstream states cloning quality is reduced in x-vector-only mode, so
shipping it alone would advertise `SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO` at the
weaker of two available qualities. Delivering it first, as its own completion
gate, exercises the mel front end, the ECAPA graph, and the whole Voice Profile
pipeline against a small graph before the codec encoder is written.

**D2 — Finish line is publishable artifacts, not publication.** The work runs
through Quantization Profiles, the CUDA Execution Backend, a listening audit,
and prepared ship artifacts. Uploading anything remains a separate act
requiring jiangzhuo's confirmation at the time.

**D3 — One primary GGUF.** `docs/model-packages.md` requires one primary GGUF
per Model Variant carrying every tensor its inference graphs need, and forbids
Sidecar Resources from carrying tensors. The speaker encoder and the codec
encoder therefore ride inside the Base package. Stage 1 dropped the codec
encoder because "no graph in this package can reach it"; in Stage 2 a graph
reaches it, so the same rule now requires its presence.

**D4 — The clone mode is fixed when the Voice Profile is created.** A
transcript makes the Profile an ICL Profile; its absence makes it
x-vector-only. Upstream decides this when it builds the prompt; moving the
decision to preparation is what gives a Serialized Profile one unambiguous
meaning.

**D5 — An ICL Profile carries recoverable transcript content, and says so.**
The ICL graph consumes reference text token ids and reference codes at
synthesis time, so a Serialized ICL Profile must carry both.
`docs/voice-conditioning.md` states that Profiles omit enrollment audio,
transcript, and description source material by default. Token ids are derived
conditioning rather than the source string, but byte-level BPE is invertible,
so whoever holds the Profile can recover the transcript. The Profile Schema
declares the field explicitly and the documentation states the recoverability,
so that distributing a Profile is an informed act. Profiles never carry
enrollment audio.

| Content | x-vector-only | ICL |
| --- | --- | --- |
| Speaker embedding `[1024]` | yes | yes |
| Reference codes `[16, T]` | no | yes |
| Reference text token ids | no | yes |
| Enrollment audio, transcript string | no | no |

**D6 — The Base package has an empty Preset Voice Catalog and no fallback.**
Upstream ships no speakers for this variant. The package declares zero Preset
Voices and no unnamed package default, so `voice_id` resolves nothing and
auto-voice does not exist here. A Synthesis request without a Voice Profile
fails with an explicit error rather than silently selecting a Voice. This is
the project's first package that cannot synthesize without prepared
conditioning, and the Loaded Model capability snapshot must report that
honestly.

**D7 — Reference Audio limits.** One clip per Voice Profile
(`max_reference_count = 1`, an upstream fact, not a conservative choice);
1 s minimum and 30 s maximum per clip; 30 s maximum total. Expressed in the
package as Reference Frame Equivalents at 1920 samples per frame. The two
duration bounds are provisional: intake measures where quality collapses below
the short bound and whether the long bound only wastes prompt length, and the
measured basis is recorded with the numbers. Values that survive that
measurement become the package's mandatory safety values.

## 3. Family Contract Additions

The Base variant declares `SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO` and, because
every successfully prepared v1 Profile can be serialized,
`SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE`. `SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT`
stays unadvertised; it belongs to Stage 3. Random Seed stays unadvertised.

`reference_transcript` is `SYNTH_REQUIREMENT_OPTIONAL` — the shape the family
plan predicted before the checkpoint was read — and its presence selects the
mode per D4. `reference_language` is optional. The declared Voice encoder
target is 24 kHz mono.

The Language Capability Catalog is unchanged from Stage 1's language handling
except that the two dialect entries, which exist only as Preset Voice overrides,
have no carrier on a variant with no Preset Voices.

## 4. Package and Conversion

The variant slug is `qwen3-tts-12hz-0-6b-base`. Conversion extends
`scripts/convert-qwen3-tts.py` rather than forking it: the CustomVoice path
stays byte-identical in output, and the Base path adds the `speaker_encoder.*`
namespace and the codec encoder half.

Both halves of the speech tokenizer are now carried. Stage 1 measured that 16 of
the encoder's codebooks and both quantizer projections were bit-identical to the
decoder's. Conversion re-measures that at this revision instead of assuming it,
then takes one of two explicit paths: if the tensors are still identical, store
each once under a canonical name that both graphs resolve and record the
mapping in the tensor catalog; if any pair has diverged, store both and say so.
What is ruled out is the third outcome — duplicated bytes nobody decided on.
The `GGML_MAX_NAME`
discipline established in Stage 1 — refuse to emit any name at or over the
limit — applies to every new name.

The Profile Compatibility ID is computed per `docs/model-packages.md` as
SHA-256 over the family's canonical compatibility manifest, including the
upstream checkpoint provenance and fingerprints of the Voice-conditioning
configuration and source weights. F32, F16, and quantized packages derived from
the same compatibility-critical source share it.

## 5. C++ Architecture

New modules in `src/arch/qwen3-tts/`, following the family's existing
graph-plus-`-host` pairing:

| Module | Kind | Responsibility |
| --- | --- | --- |
| `mel.{h,cpp}` | host DSP | 24 kHz F32 PCM to `[128, T]` log-mel at the pinned STFT parameters |
| `speaker-encoder.{h,cpp}` | GGML graph | ECAPA-TDNN over mel to a `[1024]` embedding |
| `speaker-encoder-host.{h,cpp}` | host | shape and finiteness validation, single-shot execution |
| `codec-encoder.{h,cpp}` | GGML graph | waveform to `[16, T]` codes: downsample, encoder, encoder transformer, RVQ |
| `codec-encoder-host.{h,cpp}` | host | frame arithmetic against 1920, code range checks |
| `profile.{h,cpp}` | host | Profile preparation, GGUF envelope, compatibility checks |

The mel front end is host DSP rather than a GGML graph because it runs once per
enrollment, its cost is negligible beside the ECAPA forward, and keeping it on
the host avoids a backend-placement question for a stage with no downstream
shape dependence. Kokoro's transform code is family-private and is not promoted
to shared code for this: one measured, family-local implementation is cheaper
than a shared abstraction serving two families with different parameters.

Preparation flow:

```
caller PCM ─► Audio Normalizer (existing) ─► 24 kHz mono F32
                                              ├─► mel ─► speaker-encoder ─► x-vector [1024]
                                              └─► (ICL only) codec-encoder ─► ref codes [16, T]
              transcript ─► BPE (Stage 1) ─► ref text token ids
                                              ▼
                                    Voice Profile (in-memory, serializable)
```

Synthesis flow: the x-vector replaces the Preset Voice token embedding at the
existing prompt position. In ICL mode a prefix block is prepended — a text track
of `[ref ids, text ids] + eos` and a codec track of `[codec_bos, sum of the 16
reference code group embeddings]`, added position-wise, which is the same
two-track structure Stage 1 already implements. Nothing downstream of the prompt
changes.

## 6. Validation and Testing

Per `docs/testing.md`, each slice lands with its own focused unit tests
registered under the `unit` label: converter rules, mel parameters, graph
stages, prompt layout, Profile round-trip, and every error mapping in section 9.
The sanitizer gate runs for every slice touching inference code.

Four new oracle dump scripts join the locked `scripts/envs/qwen3-tts/`
environment: mel plus speaker encoder, codec encoder, clone prompt layout, and
Base end-to-end. The oracle runs at the checkpoint's dtype, and both sampling
switches (`do_sample`, `subtalker_dosample`) are set false wherever a
reproducible trajectory is needed — the Stage 1 trap that is silently
non-deterministic if only one is set.

Mel to x-vector and waveform to codes are deterministic given fixed input, so
both compare tensor-by-tensor under committed tolerances. Reference codes are
discrete and are compared for equality, but equality is not assumed to be
free: RVQ assignment is an argmax over codebook distances, and a near-tie can
flip under the F32 divergence that separates two implementations — the case
`docs/port-validation.md` already handles with `oracle.alternate_grids`, where
the manifest enumerates further grids *the reference itself produced* under a
different configuration, each pinned by `sha256`. Intake measures the actual
tie margin on the reference clips before the gate's final form is committed:
if flips do not occur, the gate is plain equality; if they do, admissible
grids are enumerated by that mechanism and never by relaxing the comparison.

The autoregressive half keeps Stage 1's replay rule: the oracle records its
sampled sequence and a validation-only seam replays it into both graphs.

A new Golden Manifest, `tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-base.manifest.json`,
pins provenance, cases, and tolerances. Cases cover both clone modes, several
languages, short and long reference clips at the declared bounds, transcript
present and absent, deterministic seeds, and every rejection path. Tolerances
extend Stage 1's existing profile-by-backend-by-stage keying.

A listening audit precedes ship. Per the standard set on the fourth family, the
acceptance bar is audible quality; a tolerance table is not audible evidence.

## 7. Quantization and Execution Backends

Plan 4 measures Quantization Profiles rather than assuming Stage 1's carry over.
Two regions are new and are judged on measurement: the speaker encoder is small
enough that quantizing it is unlikely to pay, and the codec encoder is
convolution-heavy, which is the shape that produced the conv-exempt policy in
another family. Neither precedent is imported by analogy; each is measured here
and the result recorded with the artifact that produced it.

The CUDA Execution Backend reuses Stage 1's wiring. Any performance number is
recorded together with the build that produced it — the development presets
compile GGML at `-O2` and have produced misleading figures before.

## 8. Plan Sequence

| Plan | Delivers | Completion gate |
| --- | --- | --- |
| 1. Base package | intake and pinned digests, converter extension, tensor catalog, empty-Catalog load semantics, Compatibility ID | package loads; capability snapshot reports zero Preset Voices and required Profile; converter unit tests pass |
| 2. x-vector path | mel front end, ECAPA graph, Profile preparation and serialization, prompt-slot wiring, public seam, CLI and binding Adapters | reference audio in, cloned audio out on CPU; mel and x-vector stages meet oracle tolerances; a usable clone capability exists |
| 3. ICL path | codec encoder graph, two-track prompt, transcript semantics, Profile Schema extension and its recoverability declaration | ICL clone runs end to end; codes and prompt layout meet oracle tolerances; Golden Manifest covers both modes |
| 4. Quantization, backends, ship prep | Quantization Profile measurement, CUDA backend, listening audit, model card and artifacts | audit recorded; artifacts prepared; upload awaits separate confirmation |

Each plan is a completion gate for the next. Plan 2 is the point at which the
public interface is honest on its own: `SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO` is
real, with one of its two modes available, and a later plan adds the second
without changing the advertised source.

## 9. Error Mapping

| Condition | Status |
| --- | --- |
| Transcript present but empty or whitespace-only | `SYNTH_ERR_INVALID_ARG` |
| Reference clip outside the declared duration bounds, or more than one clip | the existing Reference Audio limit path |
| ICL prefix plus target text exceeding the Talker's input limit | Stage 1's existing input-limit path |
| Serialized Profile whose Compatibility ID does not match the package | the existing compatibility rejection |
| Synthesis request with no Voice Profile on a Catalog-less package | explicit error; never a silent Voice selection |
| Non-finite or malformed PCM after normalization | the Audio Normalizer's existing validation |

## 10. Non-Goals and Deferred

- **Stage 3** — `SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT` on
  `qwen3-tts-12hz-1.7b-voicedesign`. Stage 2 is its completion gate.
- **The 1.7B Base variant.** The ladder puts 0.6B at this rung; 1.7B Base
  exists upstream and is not part of this work.
- **Multi-clip enrollment.** Upstream has no path that merges several clips
  into one Voice; adding one would be a project invention presented as a port.
- **Native Streaming Synthesis.** Unchanged from Stage 1's Chunked Audio
  Delivery claim.
- **Voice Conversion**, which is out of scope project-wide.
- **Publication itself**, which is a separate act requiring confirmation at the
  time.
- **Quality Evaluation**, deferred per ADR 0017. Delivery is `port_validated`
  plus a listening audit.
