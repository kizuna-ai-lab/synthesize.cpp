# synthesize.cpp

An embeddable, local, offline text-to-speech inference library in C/C++, built on
GGML and GGUF. It runs explicitly ported and validated speech synthesis models on
your own machine, with no network at runtime and no hosted service.

The product is the stable C interface in [`include/synthesize.h`](include/synthesize.h).
The command-line program and the language Adapters are clients of that same
interface — no synthesis capability exists only in one of them.

- Version `0.1.0`, ABI version `1` (`SYNTH_VERSION_*` and `SYNTH_ABI_VERSION` in
  the header are the single source of truth; CMake and `pyproject.toml` both
  parse it from there).
- Four Model Families, six Reference Model Variants, five published model
  repositories.
- Delivered Validation Level is `port_validated` throughout. This project has
  **not** run a corpus-scale Quality Evaluation and makes no quality,
  naturalness, or comparative-ranking claim ([ADR 0017](docs/adr/0017-separate-port-validation-from-quality-evaluation.md)).

## What it can do today

- Synthesize speech from raw UTF-8 text, from phonemes, or from exact token IDs,
  into a caller-selected Audio Sink (a complete buffer or a callback).
- Load Model Packages that are a single primary GGUF plus optional declarative
  Sidecar Resources. Nothing in a Model Package is executable and nothing is
  downloaded implicitly.
- Select a Preset Voice from a package's catalog, set a speaking rate and a
  synthesis seed, and bound the output length.
- Prepare a Voice Profile from Reference Audio or from Description Text —
  **for the OmniVoice family only**. `src/voice-profile.cpp` dispatches every
  Voice Profile source for OmniVoice and refuses the other three families with
  `SYNTH_ERR_UNSUPPORTED_VOICE`.
- Run on CUDA as a second Execution Backend, with per-family placement rules
  (see [Execution Backends](#execution-backends) — the status differs per family
  and is not summarizable as "GPU support").

Two Text Frontend Providers are built in: `synthesize.symbol_map` for phoneme
input (VITS, Kokoro) and `synthesize.qwen_bpe`, a declarative byte-level BPE that
reads its vocabulary and merges from the Model Package, for raw-text input
(Qwen3-TTS, OmniVoice). There is no grapheme-to-phoneme engine: raw text reaches
the model by tokenization, not by G2P. See [docs/text-frontends.md](docs/text-frontends.md).

## Model Families and variants

| Family | Reference Model Variant | Rate | Voices | Input | Profiles | Level |
| --- | --- | ---: | --- | --- | --- | --- |
| VITS | `vits-ljspeech` | 22.05 kHz | 1, fixed default | phonemes, token IDs | F32, F16, Q8_MIXED | `port_validated` |
| VITS | `vits-vctk` | 22.05 kHz | 109 preset | phonemes, token IDs | F32, F16, Q8_MIXED | `port_validated` |
| Kokoro | `kokoro-v1-0` | 24 kHz | 54 preset | phonemes, token IDs | F32, F16, Q8_MIXED | `port_validated` |
| Qwen3-TTS | `qwen3-tts-12hz-0.6b-customvoice` | 24 kHz | 9 preset | text, token IDs | BF16, F16, Q8_MIXED | `port_validated` |
| Qwen3-TTS | `qwen3-tts-12hz-0.6b-base` | 24 kHz | none | text | source dtype only | converted and loads; not validated, not published |
| OmniVoice | `omnivoice-0-6b` | 24 kHz | no named Voices; an unnamed auto-voice default, plus Reference Audio and Description Text Voice Profiles | text | F32, F16, Q8 | `port_validated` |

Declared Language Capability differs by family and is read out of the package,
not inferred from the architecture. VITS and Kokoro declare `en`. OmniVoice
declares `en`, `zh`, `ja`. Both Qwen3-TTS packages declare ten languages —
`en`, `de`, `es`, `zh`, `ja`, `fr`, `ko`, `ru`, `it`, `pt` — published as BCP 47
tags by the runtime, which is a claim about the upstream model rather than a
statement that each has its own validation cases.

The two families' catalogs differ in one instructive way. The CustomVoice
package's family-level language table carries twelve entries: the ten above plus
`sichuan_dialect` and `beijing_dialect`. Those two are dropped before the public
Language Capability Catalog is built, because a dialect here is reachable only by
selecting the Preset Voice that pins it, never by asking for it as a language.
The Base package's table carries ten and no dialects at all — it ships no Preset
Voices, so nothing can pin one.

Per-variant records are in [docs/models/](docs/models/) and the porting contracts
in [docs/porting/families/](docs/porting/families/). Golden Manifests — the
committed contracts that pin provenance, cases and tolerances without committing
the payloads — are in [tests/golden/](tests/golden/).

## Published packages

Four **Published Model Packages** and one **Restricted Model Package**, all under
the `jiangzhuo9357` Hugging Face account. Repository names end in `-gguf`; other
repositories on that account belong to unrelated work.

| Repository | Files | Last modified | Category | License |
| --- | --- | --- | --- | --- |
| [`vits-ljspeech-gguf`](https://huggingface.co/jiangzhuo9357/vits-ljspeech-gguf) | 3 GGUF | 2026-07-28 | Published | MIT source; checkpoint terms unspecified |
| [`vits-vctk-gguf`](https://huggingface.co/jiangzhuo9357/vits-vctk-gguf) | 3 GGUF | 2026-07-28 | Published | MIT source; checkpoint terms unspecified |
| [`kokoro-v1-0-gguf`](https://huggingface.co/jiangzhuo9357/kokoro-v1-0-gguf) | 3 GGUF | 2026-07-29 | Published | Apache-2.0 |
| [`qwen3-tts-12hz-0-6b-customvoice-gguf`](https://huggingface.co/jiangzhuo9357/qwen3-tts-12hz-0-6b-customvoice-gguf) | 3 GGUF | 2026-07-29 | Published | Apache-2.0 |
| [`omnivoice-0-6b-gguf`](https://huggingface.co/jiangzhuo9357/omnivoice-0-6b-gguf) | 3 GGUF + 2 licence files | 2026-08-10 | **Restricted** | see below |

**OmniVoice is a Restricted Model Package, not a Published Model Package**
([ADR 0018](docs/adr/0018-publish-nc-families-as-restricted-model-packages.md)).
Its generator weights are **CC-BY-NC**, with no version stated by upstream, and
its codec weights carry the **Boson Higgs Audio 2 Community License**, which is
derived from and defines itself to include the Meta Llama 3 Community License —
so two licence files travel with the weights and must be downloaded with them.
These weights are **not licensed for commercial use**, and this package never
carries the embeddable-in-other-programs promise that a Published Model Package
does. It must not be described as Apache-2.0 or as unrestricted.

The VITS packages are published on an explicit project-policy assumption: the
upstream source is MIT, but the official checkpoint carries no separate
checkpoint-license statement. Anyone needing a different legal reading should
review the upstream provenance before redistributing.

Model files are not committed to this repository. Download one package into
`models/<variant-slug>/` and point the library at the local GGUF path; the
library never accepts a repository identifier in place of a path.

## Execution Backends

CPU is the mandatory correctness baseline for every family. Each family has one
backend-independent GGML graph; backends execute the same graph rather than
receiving their own implementation.

Per-family status differs and is deliberately not generalized:

| Family | CUDA | Stages held on CPU |
| --- | --- | --- |
| VITS | runs; recorded as **Experimental** until the remaining Provider gates close | text encoder, duration predictor |
| Kokoro | runs; no Support State word recorded | PL-BERT, duration predictor |
| Qwen3-TTS | runs; no Support State word recorded | talker, code predictor (the codec is the half that moves) |
| OmniVoice | runs; both halves move, since 2026-08-08 | none, since 2026-08-08 |

Every published model card states its own placement in these terms: as of
2026-08-12 all five declare `cuda_placement` explicitly, and none can inherit a
default. Four of them previously asserted "CUDA placement contained zero
executable CPU fallback nodes" because the card generator defaulted to that
sentence; the "Stages held on CPU" column above is what those cards now say.

A stage whose output is a discrete value — a rounded frame count, an argmax, a
sampled token index — runs on CPU on every backend, and so does every stage
feeding it, because tolerance cannot absorb a discrete difference. OmniVoice's
generator is the one measured exception, admitted because its discrete output
picks content inside a canvas whose shape was fixed before the first forward ran.
**Metal and Vulkan are unavailable for every family.** CUDA F32 matrix multiplies
compute at TF32 precision and the project makes no strict-FP32 promise on CUDA.
All of this, with the measurements behind it, is in [docs/backends.md](docs/backends.md).

Only VITS/CUDA carries an explicit Support State anywhere in this repository. For
the other three, CUDA demonstrably runs graph work and no document declares it
Supported; this README does not invent the word in either direction.

## In progress

**Qwen3-TTS Stage 2 — the Base variant, toward Reference Audio voice cloning.**
Plan 1 is done: the Base package is pinned, converted (894 tensors), loads through
`synth_model_load`, and its declared Voice Profile contract
(`synthesize.profile.*`, `synthesize.reference.*`) is read and validated at load
time, refusing a package that declares it badly.

**Voice cloning is not delivered for this family.** The runtime reports **zero
Voice Profile source flags** for both Qwen3-TTS variants, and zero in every field
describing a source, because no preparation path exists:
`synth_voice_profile_create_from_reference` still returns
`SYNTH_ERR_UNSUPPORTED_VOICE` here. The speaker encoder's tensors are catalogued
and shape-checked but no graph runs over them; the ECAPA-TDNN encoder, the Audio
Normalizer and the mel front end are Plan 2's scope. The package declaring a
contract and the runtime advertising a capability are different statements, and
`docs/c-interface.md` decides which one the query reports. The reference-duration
bounds that shipped are safety ceilings, not perceptually validated ones; no
listening pass has happened. Details:
[docs/superpowers/plans/2026-08-12-qwen3-tts-stage-2-plan-1-carryover.md](docs/superpowers/plans/2026-08-12-qwen3-tts-stage-2-plan-1-carryover.md).

Stage 3 (`qwen3-tts-12hz-1.7b-voicedesign`, Description Text) is planned behind
Stage 2. Metal and Vulkan follow CUDA in the backend sequence and have not begun.

## Adapters

| Adapter | State |
| --- | --- |
| `synthesize-cli` | Implemented. A reference client over the installed C interface, with no Model Family or GGML headers of its own. Disable with `-DSYNTH_BUILD_CLI=OFF`. |
| Python | Implemented. One abi3 API wheel (`synthesize_cpp`) plus Native Provider wheels (default CPU, `cu13` CUDA) discovered through an entry point ([ADR 0014](docs/adr/0014-separate-python-api-wheel-from-native-providers.md)). |
| Rust | **Specified, not present in this repository.** [docs/language-bindings.md](docs/language-bindings.md) and [docs/rust-packaging.md](docs/rust-packaging.md) define the crate contract; no crate source or `Cargo.toml` exists in the tree yet. |

```
                       +-> CLI reference adapter
C++ implementation -> C ABI -> Rust safe adapter (specified)
                       +-> Python abi3 adapter
```

## Building and the unit gate

Requires CMake >= 3.24 and [`uv`](https://docs.astral.sh/uv/) (the Python tests
and the pinned clang-format run through it). `ggml/` is a git submodule pinned to
a commit of `ggml-org/ggml`; an uninitialized submodule fails the build rather
than degrading quietly.

```bash
git submodule update --init --recursive

cmake -S . -B build \
  -DSYNTH_BUILD_TESTS=ON \
  -DSYNTH_BUILD_PYTHON_TESTS=ON \
  -DSYNTH_BUILD_INTEGRATION_TESTS=OFF
cmake --build build --target synthesize-check-unit
```

Run a subset by label or by name:

```bash
ctest --test-dir build --output-on-failure -L unit
ctest --test-dir build --output-on-failure -R '^synthesize-public-api-test$'
```

Labels in use include `unit`, `integration`, `golden`, `abi`, `backend`, `cli`,
`cuda`, `packaging`, `python`, `quantization`, `voice`, `voice-profile`, and one
per family (`vits`, `kokoro`, `qwen3-tts`, `omnivoice`).

Integration and Golden tests register only after you materialize real model GGUFs
and reference payloads locally, then configure with
`-DSYNTH_BUILD_INTEGRATION_TESTS=ON` and build `synthesize-check-integration`.
Neither models nor generated payloads are committed — the rule is *commit golden
contracts, not golden payloads*. Full instructions, the sanitizer gate, and the
CUDA presets: [docs/testing.md](docs/testing.md).

Synthesize one utterance once you have a package on disk:

```bash
hf download jiangzhuo9357/kokoro-v1-0-gguf kokoro-v1-0-F16.gguf \
  --local-dir models/kokoro-v1-0

build/bin/synthesize-cli \
  --model models/kokoro-v1-0/kokoro-v1-0-F16.gguf \
  --output output.wav \
  --phonemes "ðə skˈI əbˈʌv ðə pˈɔɹt wʌz ðə kˈʌləɹ ʌv tˈɛləvˌɪʒən, tˈund tə ɐ dˈɛd ʧˈænᵊl." \
  --language en \
  --voice af_heart \
  --seed 0
```

Kokoro consumes phonemes; Qwen3-TTS and OmniVoice take `--text` instead, since
they carry their own byte-level BPE frontend.

Output is little-endian 32-bit float WAV at the model's native sample rate.
Synthesis completes before the file is opened, so a failure leaves no output file.

## Explicitly out of scope

From [docs/scope.md](docs/scope.md):

- Model training or fine-tuning.
- Automatically running arbitrary Hugging Face TTS models.
- Requiring a network or hosted inference service at runtime.
- Loading executable code or implicitly downloading resources from a Model Package.
- Claiming backend support based only on compilation or partial operator coverage.

Deferred rather than rejected: corpus-scale Quality Evaluation Suites and
cross-model quality comparison, the reference corpora and evaluator artifacts
they would need, and frozen perceptual thresholds. Until that exists, delivery is
`port_validated` plus, for three variants, an optional non-statistical
**Listening Audit** by one maintainer recording `no_obvious_regression`. A
Listening Audit is not a quality claim and does not move the Validation Level.

## The records drift, and that is checkable

This project's documents are meant to be verified against artifacts rather than
believed, and the reason is not theoretical. In the week to 2026-08-12, **five
status lines in this repository were found to be stale**, each written accurately
and then never revisited while the work moved past it:

| Document | Said | Actually |
| --- | --- | --- |
| `scripts/hf_cards/qwen3-tts-12hz-0-6b-customvoice.yaml` | "Prepared, not published" | Published two weeks earlier; corrected 2026-08-11 |
| `docs/porting/families/kokoro.md` | "Conversion, C++ implementation, and port validation are not started" | Converted, implemented, validated, quantized, audited and published |
| `docs/porting/families/qwen3-tts.md` (Status) | "Q8_MIXED, the public backend control and stage 8 are not done. Port validation is not started" | All four done; each contradicted by a later section of the same file |
| `docs/porting/families/qwen3-tts.md` (Stage 8) | "Nothing has been published" | Published 2026-07-28, as that file's own Open Question 6 already recorded |
| `docs/text-frontends.md` | "Raw-text/G2P providers are not implemented yet" | Raw text implemented via `synthesize.qwen_bpe`; G2P genuinely still absent |

The first was corrected on 2026-08-11 after being read as fact during design
work; the other four on 2026-08-12 while this README was written. Each corrected
line now records what it previously claimed, so the correction is auditable too.

The lesson is recorded rather than resolved: a status line is a hypothesis about
the past, and the artifact — the Hugging Face API, the card specification, the
Golden Manifest, the code — is the evidence. If something here disagrees with an
artifact, the artifact is right and this document is stale. Please say so.

## Documentation

Documents under [`docs/`](docs/) are confirmed contracts and carry a
`Status: Confirmed …` line with a date. [`docs/adr/`](docs/adr/) holds numbered
architecture decisions, added rather than rewritten.
[`CONTEXT.md`](CONTEXT.md) defines the project's vocabulary with explicit terms
to avoid, and is the reason this README says "Synthesis" rather than
"generation", "Model Family" rather than "model type", and distinguishes a
Published Model Package from a Restricted one.

Start with [scope](docs/scope.md), the [C interface](docs/c-interface.md),
[model packages](docs/model-packages.md), [backends](docs/backends.md),
[port validation](docs/port-validation.md), and [testing](docs/testing.md).

## Third-party code

`ggml/` is a submodule of [`ggml-org/ggml`](https://github.com/ggml-org/ggml) and
carries no local modification; [`ggml-patches/README.md`](ggml-patches/README.md)
records the five patches that once existed and why each one went away. Upstream
model and code attributions are in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
