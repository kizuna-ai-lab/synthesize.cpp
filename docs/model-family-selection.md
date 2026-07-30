# Model Family Selection

Status: Confirmed on 2026-07-30.

Model families are selected architecture-first. Language breadth is a capability of individual model variants and is not an admission requirement for a family implementation.

The first family should be chosen using these criteria:

1. It exercises a representative TTS architecture and a useful set of GGML operators.
2. It has a stable, inspectable reference implementation suitable for tensor-by-tensor validation.
3. At least one model variant is practical for CPU inference.
4. Its operator gaps across CPU and GPU backends are measurable and tractable.
5. Its code, weights, voices, and required frontend resources have usable licenses.
6. Its weights can be converted reproducibly and quantized incrementally.
7. Its acoustic model, vocoder or codec decoder, and required frontend can be tested as an end-to-end pipeline.

Language priorities guide which available variants are validated first. They do not require one family or one variant to support every priority language.

## First Model Family

The first supported model family is VITS. The initial implementation targets the native inference architecture shared by baseline VITS models: text encoding, duration prediction, latent expansion, inverse flow, and waveform generation.

Piper, MMS-TTS, MeloTTS, YourTTS, and Bert-VITS2 are follow-on adaptation candidates because they descend from or extend VITS. They are not automatically considered supported: each requires its own converter or configuration mapping and complete end-to-end validation.

The choice is architecture-based. The languages, voices, and locales validated first will be decided separately when selecting model variants.

## Second Model Family

The second supported model family is Kokoro: a StyleTTS 2 decoder with an
iSTFTNet generator. It was selected under the same criteria and adds operator
surface the first family does not exercise — a bidirectional LSTM, adaptive
instance normalization, a Snake activation, a harmonic-plus-noise source with
phase accumulation, and a forward and inverse short-time transform.

It is also the first family that is architecturally language-blind. A Kokoro
model consumes IPA phoneme token IDs and a style vector and has no language
input of its own, which keeps language breadth where this document already
places it: a property of variants and their validation cases, not of the family.

Its reference variant is `kokoro-v1-0`, the official Kokoro-82M v1.0
checkpoint. Both the source and the weights carry an explicit Apache-2.0 grant.

A second family does not generalize the first family's decisions. Kokoro's
Quantization Profiles quantize a different part of the model than VITS's, for a
reason measured in that family and recorded in
`docs/porting/families/kokoro.md`: its excitation phase accumulates across a
whole utterance, so everything upstream of the decoder stays at the reference
dtype.

## Third Model Family

The third supported model family is Qwen3-TTS 12 Hz: a Qwen3 decoder that emits
one semantic code per frame, a multi-token-prediction head that expands each
frame into fifteen acoustic codes, and a residual vector-quantized codec that
reconstructs 24 kHz audio. Its selection rationale, staged variants, and
decomposition are recorded in
[`porting/families/qwen3-tts.md`](porting/families/qwen3-tts.md).

Kokoro was chosen partly on standing among CPU-practical open models. That test
was applied again and deliberately abandoned for this cycle. Of the four
open-weight models ranking above the shipped Kokoro variant, two are restricted
to non-commercial or separately licensed use, one asserts Apache-2.0 while
acknowledging that part of its code and data derives from a checkpoint carrying
no explicit license and requires 12 GB of GPU memory at batch size 1, and the
fourth leads by an amount inside measurement noise. This family is therefore
selected on capability and language coverage. It is not claimed to sound better
than the second family, and its delivered Validation Level is `port_validated`.

It adds operator and runtime surface neither predecessor exercises: KV-cached
autoregressive decoding, a sampling chain, a multi-token-prediction head, and a
causal residual vector-quantized codec decoder. It is also the first family
whose upstream variants map directly onto declared Voice Profile sources rather
than onto the `voice_id` path alone, and the first to require a byte-level BPE
Text Frontend Provider, which is what would give the project raw-text input
without depending on a GPL-licensed grapheme-to-phoneme engine.

Capability is split across upstream checkpoints, so it is staged across three
Reference Model Variants: `qwen3-tts-12hz-0.6b-customvoice` establishes the
family on the existing Preset Voice path, `qwen3-tts-12hz-0.6b-base` adds
Reference Audio, and `qwen3-tts-12hz-1.7b-voicedesign` adds Description Text.
Each stage is a completion gate for the next. All four upstream repositories,
including the shared `Qwen3-TTS-Tokenizer-12Hz`, carry an explicit Apache-2.0
grant verified against their model-card metadata.

Autoregressive sampling does not amend the Port Validation Contract. Two
frameworks are not required to agree on a sampling trajectory; following the
stochastic-replay rule already in `port-validation.md`, the oracle captures its
sampled code sequence and a validation-only family seam replays it into both
graphs. The codec decoder remains deterministic given a fixed code sequence and
keeps ordinary end-to-end waveform comparison.

## Fourth Model Family Candidate

OmniVoice is recorded as the leading fourth-family candidate. It is a
non-autoregressive mask-predict model whose refinement steps reabsorb
floating-point argmax flips, making exact token agreement checkable, and a
single checkpoint carries reference-audio cloning, described-attribute voice
design, and automatic voice selection together.

Its pre-trained weights are licensed CC-BY-NC because of training-data
constraints, while only its code is Apache-2.0. A family in that position can
become a Supported Model Family, because that status requires a port passing the
Port Validation Suite on CPU. It cannot become a Published Model Package,
because that status additionally requires releasing exact bytes and license
through a project-owned model repository. Whether the project supports a family
it can never publish is an open decision that a fourth-family intake must settle
before conversion work begins.

License facts for this candidate were established from its upstream model card.
A downstream community port describes these weights as Apache-2.0, which is
incorrect; port documentation is not a license source.

Resolved 2026-07-30: OmniVoice is the fourth Model Family. The question is
settled by ADR 0018 — the family ships as a Restricted Model Package, carrying
the upstream CC-BY-NC statement for the LM weights and the Boson Higgs Audio 2
Community License for the bundled codec weights (a third license the original
deferral note did not record; it lives at `audio_tokenizer/LICENSE` inside the
weights repository). The porting record is `docs/porting/families/omnivoice.md`.

## Reference Validation Variants

The upstream reference implementation is `jaywalnut310/vits`.

The first reference model variant is its single-speaker LJSpeech configuration and checkpoint. It is used to validate the baseline VITS inference path without multi-speaker conditioning.

The second reference model variant is the upstream multi-speaker VCTK configuration and checkpoint. It extends validation to speaker embeddings and speaker-conditioned inference after the baseline path agrees with the reference implementation.

These choices minimize validation complexity. They do not make English a language priority, and matching them does not automatically validate other VITS-derived model variants.

## Upstream Port Validation Cases

The initial VITS Port Validation Suite does not require either training corpus.
It pins the upstream repository revision and imports the two ordinary TTS examples
from its [inference notebook](https://github.com/jaywalnut310/vits/blob/main/inference.ipynb):

- the LJSpeech checkpoint synthesizes `VITS is Awesome!` with the upstream
  `ljs_base.json` configuration; and
- the VCTK checkpoint synthesizes the same text with the upstream `vctk_base.json`
  configuration and speaker ID 4.

The reference runner preserves the upstream `noise_scale=0.667`,
`noise_scale_w=0.8`, and `length_scale=1` controls and fixes `seed=0` so the
stochastic inference path is reproducible. These upstream-authored cases
remain in every compatible manifest. Each Model Variant normally has 12 to 32
Port Validation Suite cases in total; additional project-authored cases cover
short and long inputs, punctuation and normalization, deterministic seeds,
synthesis controls, and every applicable Voice-conditioning branch. Case count is
driven by inference-path coverage rather than corpus representativeness.

The exact first-version case matrices, VITS probe names, stochastic replay inputs,
speaker-ID mapping, and speaking-rate boundaries are defined in
[`porting/families/vits.md`](porting/families/vits.md).

Ordinary upstream TTS inference needs the pinned source, configuration, checkpoint,
Text Frontend, and Linguistic Input, not the LJSpeech or VCTK audio archives. The
upstream notebook loads VCTK audio only for its separate Voice Conversion example;
Voice Conversion is not part of this TTS port. Initial port validation therefore
downloads neither full dataset and commits no dataset hash.

## Deferred Quality Reference Data

LJSpeech 1.1 and CSTR VCTK Corpus 0.92 remain candidate reference sources for a
future corpus-scale Quality Evaluation Suite. If that work is commissioned, its
own manifest will pin official archive locators, releases, licenses, citations,
archive SHA-256 values, selected relative paths, and selected-file SHA-256 values
after an intentional download. Runtime synthesis, Model Packages, HF repositories,
and mandatory model-free CI will not redistribute or implicitly download either
dataset.

The upstream checkpoints are reference and conversion inputs rather than user-facing
synthesize.cpp artifacts. After provenance and redistribution permission are
established, the project converts them and applies the Port Validation Suite before
publishing directly loadable Model Packages through its own Hugging Face model
repositories. The project does not redirect users to a mutable upstream checkpoint
or treat a third-party framework conversion as its released model.
