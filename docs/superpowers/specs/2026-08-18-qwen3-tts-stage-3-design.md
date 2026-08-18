# Qwen3-TTS Stage 3 — Description Text Voice Design — Design

Status: Approved in discussion with jiangzhuo on 2026-08-18. This is the design
record for the third and final rung of the Qwen3-TTS Reference Model Variant
Ladder (`docs/porting/families/qwen3-tts.md`, "Reference Model Variant Ladder").
The family record itself is extended at intake, per `docs/model-porting.md`.

Plans 1–3 are written from this document, so where execution contradicts it the
correction belongs here rather than only in the plan that found it, marked as an
erratum in the section it corrects, with the original prescription left standing
above it — the convention Stage 2's design established across its five errata.

## 1. Context

Stage 2 is delivered and published. `qwen3-tts-12hz-0-6b-base` clones a voice
from a reference recording in both upstream modes — x-vector and
transcript-assisted (ICL) — and is live at
`jiangzhuo9357/qwen3-tts-12hz-0-6b-base-gguf` carrying BF16, F16 and Q8_MIXED
(published 2026-08-17, 6.70 GB, commit `d4df99e8`). Stage 1
(`qwen3-tts-12hz-0-6b-customvoice`) remains published with nine Preset Voices.

Stage 3 is `SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT` on
`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` — a sentence of natural language in, a
voice out, with no recording and no preset catalogue. The checkpoint is
Apache-2.0 at revision `5ecdb67327fd37bb2e042aab12ff7391903235d3`, verified from
the upstream model card rather than from a port's README; `model.safetensors` is
3.83 GB and `speech_tokenizer/model.safetensors` is 0.68 GB.

**Stage 3 is narrower than Stage 2, not wider.** The two configs were compared
field by field: 128 keys are identical and six differ.

| | Base 0.6B | VoiceDesign 1.7B |
| --- | --- | --- |
| `talker_config.hidden_size` | 1024 | 2048 |
| `talker_config.intermediate_size` | 3072 | 6144 |
| `tts_model_size` | `0b6` | `1b7` |
| `tts_model_type` | `base` | `voice_design` |
| `speaker_encoder_config.enc_dim` | 1024 | **absent** |
| `speaker_encoder_config.sample_rate` | 24000 | **absent** |

**Erratum, 2026-08-18 — the two rows above name `talker_config.hidden_size`
and `talker_config.intermediate_size` correctly, but the table gives no reason
to expect the code predictor's OWN widths to move with them, and Task 6 found
that the assumption they do was already load-bearing.** `talker_config`
nests a `code_predictor_config`, and that nested config's `hidden_size` and
`intermediate_size` do **not** appear in the 128-identical/6-differ count
above because they genuinely do not differ: they stay 1024/3072 at both
rungs, while the talker widens to 2048/6144 at VoiceDesign alone. Stage 1's
tensor catalog (`src/arch/qwen3-tts/catalog.cpp`) was written against the
0.6B rung, where talker and code-predictor widths happen to coincide on every
axis, and it read the code predictor's feed-forward width off the talker's
own field rather than a field of its own — invisible until a rung existed
whose two configs disagreed. VoiceDesign's checkpoint carries the bridge the
reference applies between them, `small_to_mtp_projection.{weight,bias}`
(`Qwen3TTSTalkerCodePredictorModelForConditionalGeneration.__init__`,
`torch.nn.Linear(talker_config.hidden_size, config.hidden_size, bias=True)`,
`Identity()` when the two agree) — and `code-predictor.h`/`.cpp` had already
declared and wired the pointers for it during Stage 1, before any package
needed them, so closing this gap needed zero new graph code, only a
metadata/catalog fix. Full account: `docs/porting/families/qwen3-tts.md`,
"Stage 3: VoiceDesign Package, Task 6". **"Stage 3 needs no new graph," a few
lines below, held up exactly as written** — the design was right about the
graph and wrong only by omission about what the catalog still assumed.

Everything a port reads is otherwise unchanged: 28 layers, 16 attention heads,
8 key/value heads, `head_dim` 128, codec vocabulary 3072, `text_hidden_size`
2048, `text_vocab_size` 151936, all ten language ids, every `codec_*` id, and
`spk_id` empty in both. `speech_tokenizer/config.json` is **byte-identical**
between the two checkpoints, so the codec half is the same model.

The consequence is that **Stage 3 needs no new graph.** The ECAPA-TDNN speaker
encoder Stage 2 Plan 2 built and the codec encoder Plan 3 built are both absent
from this checkpoint and both unnecessary: there is no reference audio on this
path. What remains — talker, code predictor, codec decoder — is Stage 1's graph
set at doubled dimensions, and the port reads every dimension from GGUF metadata
rather than hardcoding it.

Upstream's own VoiceDesign path (`qwen_tts/inference/qwen3_tts_model.py`'s
`generate_voice_design`, gated at line 686 on `tts_model_type == "voice_design"`)
runs the same prompt assembly as the other two modes and differs in exactly two
places, both in `qwen_tts/core/models/modeling_qwen3_tts.py`:

* Lines 2076–2080 prepend the instruct text's projected embeddings to
  `talker_input_embeds` before the tts text prompt is built.
* Lines 2088–2089 leave `speaker_embed` as `None`, so the codec prefill at
  lines 2166–2172 omits the speaker slot entirely rather than substituting into
  it.

### 1.1 A silent failure mode, recorded because it will mislead someone

`generate_custom_voice` accepts an `instruct` argument and then, at
`qwen3_tts_model.py:799-800`, discards it for any 0.6B model:

```python
if self.model.tts_model_size in "0b6":  # for 0b6 model, instruct is not supported
    instruct = None
```

Anyone who tries Description Text against the published CustomVoice package gets
a successful synthesis that silently ignored the instruction. This belongs in
the family record, not only here: it is a result that looks like a working
feature.

## 2. Decisions

**D1. A Description Text Voice Profile holds the string.** The payload is
`DesignInstruct { std::string instruct; }`, mirroring
`src/arch/omnivoice/profile.h`'s struct of the same name. Tokenization happens
at synthesis, not at creation.

**D2. Creation validates UTF-8 and length, and nothing else, deliberately.**
OmniVoice's arm can reject an instruct because upstream defines a closed
attribute vocabulary. Qwen3-TTS's is free-form — the docstring at
`qwen3_tts_model.py:654` says "Empty string is allowed" and upstream applies no
vocabulary at all. Inventing one here would reject input upstream accepts, which
turns a port into a redesign. The consequence is that `create_from_description`
does very little on this family, and the model card must say so rather than
imply a validation that does not happen.

**D3. An empty instruct is a legal input, not an error.** It maps to upstream's
`instruct_ids.append(None)` — no instruct block — which with no speaker embedding
is an unconditioned generation whose voice is the seed's. This is the path Plan 1
uses as its completion gate.

**D4. Capability bits follow the tensors present, not the variant name.** This
package advertises `SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT |
SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE` and **not**
`SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO`: with no speaker encoder and no codec
encoder there is nothing to clone with. See section 3.

**D5. Language is a per-synthesis field and does not enter the Profile.**
Upstream takes `text`, `instruct` and `language` as three separate arguments to
`generate_voice_design`; the request already carries language.

**D6. Serialization stores the text and re-tokenizes on load**, the policy
OmniVoice's `ClonePrompt` already applies to `transcript_ids`: determinism comes
from the frozen BPE vocabulary, so storing ids costs nothing and risks
everything across a package upgrade.

**D7. Speed is not this rung's claim.** Talker parameters per layer go from
15.73 M to 50.33 M — attention 6.29 M → 12.58 M, MLP 9.44 M → 37.75 M — a factor
of **3.2×** at an unchanged 28 layers. The autoregressive half dominates CPU wall
clock (Stage 2 measured CUDA buying only 8% end to end for that reason), and
Base's Q8_MIXED measured RTF 0.863, so Stage 3's Q8_MIXED is **estimated at
2.5–2.8** and is not expected to cross real time. This is arithmetic available
before any download; it is recorded as an estimate and Plan 3 replaces it with a
measurement.

## 3. Family Contract Additions

`src/arch/qwen3-tts/weights.cpp:783` publishes the source flags as a hardcoded
pair, guarded only by the Voice Mode:

```cpp
if (hparams.voice_mode != VoiceMode::ProfileSources) return;
info.source_flags = SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO | SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE;
```

After Stage 3, `VoiceMode::ProfileSources` is shared by two variants whose
sources differ, so "Voice Mode determines the flags" stops being true. That is
the same class of trap the comment at lines 770–783 already records for
`has_preset_voice_catalog`, caught once and worth not repeating.

The rule becomes: **a source bit is published only when the tensors that
implement it are present.** `REFERENCE_AUDIO` requires the speaker encoder;
`DESCRIPTION_TEXT` requires the voice_design variant; `SERIALIZED_PROFILE`
accompanies either, because every v1 Profile this family can create can be
serialized (`docs/c-interface.md`). A package whose metadata claims a source it
cannot implement is refused at load, not tolerated.

| | CustomVoice | Base | VoiceDesign |
| --- | --- | --- | --- |
| `synthesize.model_variant` | `custom_voice` | `base` | `voice_design` |
| Speaker encoder / codec encoder | no / no | yes / yes | **no / no** |
| Voice Mode | PresetCatalog | ProfileSources | ProfileSources |
| Source flags | (none) | REF \| SER | **DESC \| SER** |
| `hidden_size` / `intermediate_size` | 1024 / 3072 | 1024 / 3072 | **2048 / 6144** |
| Preset Voices | 9 | 0 | 0 |

**This row is the talker's `hidden_size`/`intermediate_size` only** — see the
2026-08-18 erratum in section 1. The code predictor's own widths stay
1024/3072 in every column, VoiceDesign included; a reader of this table alone
could not tell the two apart, and Task 6 found that a catalog which could not
either refused to load the package.

## 4. Package and Conversion

`scripts/convert-qwen3-tts.py:212`'s `variant_profile()` already resolves what a
checkpoint carries from what it declares, and cross-checks the declaration
against the config so a new variant cannot silently convert as whichever branch
it fell into. It gains a third arm with the check in the same direction
CustomVoice's runs:

```python
if model_type == "voice_design":
    if has_encoder_config:
        raise ConverterError(
            "config declares tts_model_type=voice_design but carries a speaker_encoder_config"
        )
    return VariantProfile("voice_design", False, False,
                          "Qwen3-TTS 12Hz 1.7B VoiceDesign", "1.7B")
```

Both booleans are `False`: **the tensor set is CustomVoice's shape, not Base's.**
No converter arithmetic changes — `hidden_size` and `intermediate_size` already
flow from the config into metadata (lines 601–608), and
`synthesize.model_variant` is already written from the manifest (line 529).

`text_hidden_size` stays 2048 while `hidden_size` doubles, so the talker's
`text_projection` becomes square (2048 → 2048) where Base's is 2048 → 1024. The
shape comes from the checkpoint; nothing in the port asserts it is rectangular.

Package naming follows the family: `qwen3-tts-12hz-1-7b-voicedesign`. A new
Golden Manifest file is added for the variant; the pinned digests are recorded at
intake.

The BF16 package is **estimated at ≈4.3 GB** — 3.83 GB of source-dtype talker
and text table, plus the codec decoder widened to F32 as every package in this
family does. That second term is expected to be the same ≈457 MB Base's package
carries, because the speech tokenizer is the same model at the same config, not
merely a similar one. Three profiles would be roughly 10 GB against Base's
6.70 GB. Plan 3 replaces every figure here with a measurement.

`--quant BF16` is refused for this family by the quantizer
(`tools/synthesize-quantize/policy.cpp`'s `profile_applies_to_architecture`), so
the BF16 package comes from the converter here exactly as it does for Stage 2.

## 5. C++ Architecture

### 5.1 The Profile payload

`src/arch/qwen3-tts/profile.h` gains `DesignInstruct` behind its own
`ProfileFamilyTag`, alongside the existing `XVectorProfile` and `IclProfile`.
`src/voice-profile.cpp`'s `create_from_description` gains a Qwen3-TTS arm beside
the OmniVoice one, refusing on a package whose variant is not `voice_design`.

### 5.2 The instruct block

`TalkerPromptRequest` (`src/arch/qwen3-tts/talker-host.h:63`) gains one field,
`instruct_tokens`. `build_talker_prompt` places it as **text-only positions**
(`TalkerInputPosition::Text::Token`, `has_codec = false`) **before**
`role_tokens`, which is where upstream appends it — into `talker_input_embeds`
prior to building the tts text prompt.

`TalkerInputPosition` needs no change. `codec_offset` shifts by the block's
length and the graph's invariant at `src/arch/qwen3-tts/talker.cpp:69`
(`codec_offset + codec_tokens->ne[0] == text->ne[1]`) continues to hold.

The chat-template wrapping `<|im_start|>user\n{instruct}<|im_end|>\n`
(`qwen3_tts_model.py:276`) belongs to the **text frontend**, not to the prompt
builder — `talker-host.h:62` already states that the chat template and the
byte-pair merges are the frontend's business.

Because the instruct goes through the same `text_projection(text_embeddings(·))`
path the assistant text does, no new graph code is written for it.

### 5.3 The third speaker case, and a branch that has never run

Today `src/arch/qwen3-tts/model.cpp:979` sets `prompt_request.has_speaker = true`
**unconditionally**, with the comment "The slot exists either way -- upstream
substitutes the embedding, it does not remove the position". That is true of both
existing cases and false of this one.

| Case | `has_speaker` | Slot |
| --- | --- | --- |
| Preset Voice (CustomVoice) | true | a codec-vocabulary token |
| x-vector / ICL (Base) | true | placeholder token, embedding substituted |
| **Description Text (VoiceDesign)** | **false** | **absent** |

`src/arch/qwen3-tts/talker-host.cpp:159` already branches on `has_speaker`, so
the code for the third row exists — but nothing on the model path has ever
reached its `else`. **Written is not run and run is not correct.** Plan 1's
completion gate exercises exactly this branch, at empty instruct, before any
instruct work exists, so that "the slot disappeared" is checked on its own rather
than entangled with "the instruction took effect".

## 6. Validation and Testing

**The dangerous failure on this rung is silent.** An empty instruct is legal and
produces ordinary speech, so **a port that ignored the instruction entirely would
pass every smoke test** — it would speak, and it would reproduce on a seed. Only
a comparison that moves when the instruction moves can catch it. Every gate below
is chosen against that standard.

### 6.1 Oracle

A new dumper drives `generate_voice_design` at the pinned revision and records
four things: the instruct block's projected embeddings, the **assembled talker
prefill**, the talker codes, and the decoded waveform.

The second is the load-bearing one. Comparing only
`text_projection(text_embeddings(instruct_ids))` would largely re-test
`build_text_projection`, which the assistant text already exercises. What the
instruct block contributes independently is the tokenization and wrapping, and
the block's **position** in the prefill — so the probe is on the assembled
prefill, not on the projection alone.

### 6.2 Stage grid

`replay` and `public`, with **no `codec_encoder` stage** — this variant has no
encoder — plus the instruct-prefill probe. Tolerances are recorded in
`tests/tolerances/qwen3-tts.json` under a new variant key, each figure beside the
measurement that set it and the injected fault it still catches, as Stage 2's
cells do.

### 6.3 Behavioural relations at the public seam

`scripts/validate-qwen3-tts-public.py` gains `--description`, in the shape Plan 4
gave `--reference`:

1. Two different instructs, same seed, same text → the audio **differs**.
2. The same instruct and seed → **byte-identical** audio. This one is exact:
   both sides are this port, so nothing licenses a tolerance.
3. An empty instruct reproduces upstream's `instruct=""` output **within the
   `replay` stage's recorded tolerance** — a port-against-oracle comparison, not
   a byte comparison, and stated separately from relation 2 because the two are
   different kinds of claim.

Relation 1 follows Plan 4's listening-audit discipline: **confirm the two outputs
differ in bytes before recording the verdict**, so that "different" is a
judgement rather than a trivial truth.

### 6.4 Refusals, which are gates too

* The VoiceDesign package refuses `create_from_reference` — no speaker encoder.
* The CustomVoice and Base packages refuse `create_from_description`.
* A package whose metadata claims `DESCRIPTION_TEXT` without the variant behind
  it is refused at load.
* Retaining the speaker slot — section 5.3's new branch written the wrong way —
  is caught by the prefill probe.

### 6.5 Fault injection

Each gate is shown to fail, per Plan 4's practice. Three faults, chosen because
each is a plausible mistake rather than a synthetic one:

* **Drop the instruct block.** The port degrades to empty-instruct behaviour. If
  the prefill probe does not catch this, that probe is decoration.
* **Place the block after `role_tokens`** instead of before it.
* **Retain the speaker slot** when it must be absent.

Unit-tier coverage accompanies each slice, registered with CTest under the `unit`
label, per `docs/testing.md` — including the converter's third arm and the
capability-bit rule of section 3, whose adversarial case follows
`tests/qwen3_tts_voice_required_test.cpp`.

## 7. Quantization and Execution Backends

**The `ConvKernel` role is unreachable on this variant.** Those 38 tensors are
the speaker encoder's convolution weights and this checkpoint has no speaker
encoder, so neither the quantizer's `ConvKernel` arm nor the runtime's
`Role::ConvKernel` is exercised here. Asserted positively in the policy test — a
package with no speaker encoder resolves no `ConvKernel` — rather than left to be
true by accident.

**Q5_K_MIXED deserves real evaluation for the first time.** At Base's 2.4 GB the
extra shrink over Q8_MIXED did not change the recommendation; at an estimated
4.3 GB it may. There is no structural obstacle: rows of 2048 and 6144 both divide
by Q5_K's 256-element super-block.

**F16 is expected not to pay, with a weaker argument than before.** On Base it
came out 184,448 bytes *larger* than its source, because both are two-byte types
while the sensitive tensors widen BF16 to F32. Here the matrix half is 3.2×
larger against a sensitive half that is not, so the penalty is relatively
smaller. The direction is likely unchanged and the magnitude is not — measured,
not assumed.

**The backend arithmetic runs the other way from Plan 4's.** Plan 4 declined a
CUDA twin for the speaker encoder and codec encoder because the transfer cost
2.2× the compute it would accelerate; neither graph exists here. What remains is
Stage 1's placement — codec decoder on the device, the autoregressive half on the
CPU by the discrete-outputs rule — and because the talker is 3.2× larger, the AR
half's share of wall clock grows, so **CUDA's end-to-end gain is expected to be
smaller than Base's 8%**, not larger.

**The listening audit carries a question the previous three did not.** Earlier
audits asked whether a profile regressed, against a baseline. This one must also
ask whether **the voice matches the description**, for which there is no ground
truth. The audit is therefore two halves, labelled at different strengths:

* Quantization regression — blind A/B on natively-sampled clips, the established
  shape, strong evidence.
* Description control — a **labelled** comparison of whether different
  descriptions produce recognizably different voices in the described direction.
  This is weak evidence and is recorded as weak.

Neither moves the Validation Level. `quality_evaluation` stays deferred per
ADR 0017.

**"Not shipping" has a stronger form on this rung.** Plan 4 established that a
profile failing to pay is a result. Here: **if the audit finds that descriptions
do not control the voice in any recognizable way, the honest output is a recorded
negative, not a published package.** Publication remains a separate act requiring
jiangzhuo's confirmation at the time, naming the target repository.

## 8. Plan Sequence

| Plan | Delivers | Completion gate |
| --- | --- | --- |
| 1. VoiceDesign package | intake and pinned digests, converter's third arm, catalog and metadata for the doubled dimensions and the absent speaker encoder, capability-bit rule, the no-speaker prompt case | package loads; a synthesis at empty instruct matches the oracle; capability snapshot reports zero Preset Voices and `DESCRIPTION_TEXT` without `REFERENCE_AUDIO` |
| 2. Description Text seam | `DesignInstruct` payload, `create_from_description` arm, serialization round-trip, `instruct_tokens` and the prefill placement, public-seam validator and Golden Manifest cases | a sentence of description in, audio out; instruct prefill meets oracle tolerances; the three behavioural relations and four refusals hold |
| 3. Quantization, backends, ship prep | Quantization Profile measurement including Q5_K_MIXED, CUDA placement measurement, listening audit in two labelled halves, model card and artifacts | audit recorded; artifacts prepared; upload awaits separate confirmation |

Each plan is a completion gate for the next. Plan 1 answers "does the 1.7B load
and compute correctly" before any instruct work is written, which is why the
empty-instruct path — needing no new prompt code — is its gate.

## 9. Error Mapping

| Condition | Status |
| --- | --- |
| Description text that is not valid UTF-8 | `SYNTH_ERR_INVALID_ARG` |
| Description text over the declared byte bound | `SYNTH_ERR_INVALID_ARG` |
| Empty description text | **accepted** — D3, the unconditioned path |
| `create_from_description` on a CustomVoice or Base package | the existing unsupported-source path |
| `create_from_reference` on a VoiceDesign package | the existing unsupported-source path |
| Serialized Profile whose Compatibility ID does not match the package | the existing compatibility rejection |
| Synthesis request with no Voice Profile on a Catalog-less package | Stage 2's existing explicit error; never a silent Voice selection |
| Instruct block plus target text exceeding the Talker's input limit | Stage 1's existing input-limit path |

## 10. Non-Goals and Deferred

- **One package that both clones and designs.** Three variants, three packages,
  as Stages 1 and 2 established. The checkpoints differ; merging them would be a
  project invention presented as a port.
- **New graphs.** Talker, code predictor and codec decoder are reused unchanged.
- **Real-time synthesis on CPU.** D7 states the estimate; the card will state the
  measurement.
- **The 1.7B Base variant**, which exists upstream and is not on this ladder.
- **Multi-clip enrollment**, **Native Streaming Synthesis**, **Voice Conversion**
  and **CLI-only capability** — unchanged from Stage 2's non-goals.
- **Automated quality evaluation** — deferred per ADR 0017; the listening audit
  is not a substitute and does not move the Validation Level.
