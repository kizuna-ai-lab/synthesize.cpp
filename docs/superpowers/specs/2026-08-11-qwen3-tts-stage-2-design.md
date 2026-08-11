# Qwen3-TTS Stage 2 — Reference Audio Voice Cloning — Design

Status: Approved in discussion with jiangzhuo on 2026-08-11; three errata added
2026-08-12 after Plan 1 (the Base package) was executed. This is the design
record for the second rung of the Qwen3-TTS Reference Model Variant Ladder
(`docs/porting/families/qwen3-tts.md`, "Reference Model Variant Ladder"). The
family record itself is extended at intake, per `docs/model-porting.md`.

Plans 2–4 are written from this document, so where execution contradicted it
the correction lives here rather than only in the plan that found it. All three
errata are marked in bold in the section they correct: section 3 on the Voice
Profile capability advertisement, section 4 on codec deduplication, section 6 on
greedy oracle decoding. Nothing else in this document has been rewritten — the
original prescription is left standing above each erratum so the change is
legible.

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
package as Reference Frame Equivalents, which this project already measures in
samples at the target rate rather than in codec frames — `convert-omnivoice.py`
writes 24000 for one second at 24 kHz, so this package writes 24000 and 720000.
Stating it in codec frames of 1920 samples would be the same unit confusion
that produced the three mis-stated frame limits corrected in `985941a`. The two
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

**Refined 2026-08-12, when Plan 2 was scoped.** `OPTIONAL` is the Stage 2 end
state, reached when both modes exist. It is not what Plan 2 reports. Plan 2
implements the x-vector path only, so a transcript names a mode that has no
implementation behind it, and reporting it optional would invite a caller to
pass one and quietly receive the weaker clone it did not ask for — the same
capability lie the erratum below removes from the source flags.

So: **Plan 2 reports `reference_transcript` as `SYNTH_REQUIREMENT_UNSUPPORTED`
and rejects a request that carries one.** Plan 3 flips it to `OPTIONAL` in the
same change that lands ICL, and the rejection becomes the mode selector D4
describes. `reference_language` follows the transcript: it qualifies a
transcript this rung cannot use.

This is a statement about the runtime, not the package: the Base package's
declared contract is unchanged, and its `reference_*` limits stay as validated.

**Erratum, 2026-08-12 — the runtime advertises no Voice Profile source at all
until preparation exists. The two paragraphs above describe what the package
declares, not what the capability snapshot may report.** They were implemented
as written: Plan 1's Task 9 published `SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO |
SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE` with the optional transcript and
language requirements and the reference limits, and the branch's final review
reversed it. `docs/c-interface.md` does not leave the question open: *"A Model
without runtime Voice Profile support reports zero flags"*, and for an
unsupported source *"all fields that describe that source are
`SYNTH_REQUIREMENT_UNSUPPORTED` or zero"* — down to a null `profile_schema`,
zero schema version, and 32 zero compatibility-id bytes. Nothing in
`src/voice-profile.cpp` can create or consume a Profile for this family; every
source there is guarded on `family != ModelFamily::Omnivoice`, so a caller
acting on the advertisement would have been refused by its very next call. "A
dispatch gap, not a capability lie" was the argument for shipping it, and the
confirmed contract overrides it. The ruling is
`docs/superpowers/plans/2026-08-12-qwen3-tts-stage-2-plan-1-carryover.md` §1.4.

What governs instead: **for the whole of Plan 1 this family reports zero source
flags, and zero in every field that describes a source.**
`fill_voice_profile_capability` leaves `VoiceProfileInfo` at its all-zero
default for every variant of the family, and
`tests/qwen3_tts_voice_required_test.cpp` and
`tests/qwen3_tts_base_load_real.cpp` assert that zero shape.

**Plan 2 is what makes the advertisement above true.** It publishes
`SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO` on the day
`synth_voice_profile_create_from_reference` can actually prepare a Profile for
this family, and `SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE` with it under
exactly the pairing rule stated above rather than as a separate decision. The
pairing rule and the 24 kHz mono encoder target are correct and survive
unchanged; they apply then rather than now, and the two tests named above are
rewritten, not deleted. `reference_transcript` and `reference_language` reach
`OPTIONAL` only at Plan 3, per the refinement recorded above section 3's
requirement paragraph — Plan 2 reports them `UNSUPPORTED` because it implements
no mode that can use them.

What did **not** change is the package: the Base variant still declares its
full Voice Profile contract, and `read_profile_contract` / `read_speaker_encoder`
still read and validate every `synthesize.profile.*`, `synthesize.reference.*`
and `synthesize.qwen3-tts.speaker_encoder.*` key at load time, refusing a
package that declares them badly. **A package carrying a Voice Profile contract
and a runtime advertising a capability are different statements**; only the
second was false, so only the second is withheld.

The rule generalizes beyond this rung, as the other two errata do: no variant
of this family advertises a Voice Profile source before the code that serves
that source exists.

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

**Erratum, 2026-08-12 — the store-once path was built, then withdrawn. Both
halves are carried in full, always.** The paragraph above prescribes two
explicit paths and Plan 1 implemented the identical-tensors one: 16 encoder
codebooks stored once under the decoder's name. What that produced was a
package in which the encoder's own names never appeared at all — a consumer
reading the GGUF found encoder quantizer stages 15–30 present and 0–14 simply
absent, with no clue that they had been folded into the decoder's. The other
half of the prescription, "record the mapping in the tensor catalog", was never
made durable: the mapping existed only in a conversion report under an ignored
build directory, so it did not survive the package. An alias no consumer can
resolve from the package alone is worse than the duplicated bytes it saves.

Two facts settled it. The measurement is smaller than it looked — the decoder
declares 16 quantizers, so encoder stages 15–30 have no counterpart to share
with at all; the real figure is 16 of 16 *existing* pairs, not 16 of 32. And
the saving is small: 33.5 MB of codebooks plus 2 MB of quantizer projections
that were duplicated too, about 1.4% of a 2.48 GB package.

What governs instead: **carry both halves in full and keep the measurement as
a recorded fact.** Every tensor name resolves from the package alone, and the
catalog, the loader and any future quantizer have no cross-half aliasing
contract to honour. `measure_shared_codebooks` in `scripts/convert-qwen3-tts.py`
still runs the comparison at every conversion and records what it finds without
acting on it. The expected Base tensor count is 894.

This erratum is not limited to Stage 2's Base package: the same rule applies to
any later variant of this family whose two codec halves overlap.

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

**Erratum, 2026-08-12 — greedy decoding does not produce a usable oracle for
this family, and was replaced by seeded sampling.** The paragraph above
prescribes `do_sample=False, subtalker_dosample=False` for reproducibility,
and Plan 1's Task 2 ran the Base oracle exactly that way. Its first case ran
away to 8191 frames — roughly 655 seconds of audio for one short sentence
(`semantic.i32` at 32764 bytes = 8191 int32, `acoustic.i32` at 8191 × 15) —
and the run was killed. This is not new: `docs/porting/families/qwen3-tts.md`
already records that greedy "degenerates on some speaker-and-input pairings"
and that Stage 1 abandoned it for that reason, and every Stage 1 Golden case
runs both switches true at `max_new_tokens 2048`. The prescription above was
written without carrying that finding forward.

Reproducibility here does not come from removing the draw. It comes from
recording it: **the oracle samples with both switches true and a fixed seed at
`max_new_tokens 2048`, and writes the sampled sequence into the case's
artifacts** (`stochastic_inputs`, the shape Stage 1 already uses). The
validation-only replay seam then feeds that exact sequence into both graphs,
which is what the last paragraph of this section already describes for the
autoregressive half — the erratum makes the oracle's own configuration agree
with it. The degenerate run's artifacts were discarded and regenerated; the
same case then drew 43 frames.

The rule generalizes to every later stage of this family: an oracle for a
sampled model is made reproducible by replaying its recorded draw, never by
switching the draw off.

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
