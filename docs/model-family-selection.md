# Model Family Selection

Status: Confirmed on 2026-07-26.

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
