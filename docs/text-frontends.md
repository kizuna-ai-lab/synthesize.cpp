# Text Frontend Boundary

Status: `synthesize.symbol_map` phoneme input implemented and validated on
2026-07-23. Raw-text/G2P providers are not implemented yet.

## Decision

Text processing is separate from model-family inference. A Model Variant declares
the Text Frontend and resources it requires. The core library owns frontend
selection and orchestration, while the architecture implementation consumes only
the resulting token sequence and does not hardcode language-specific rules.

## Provider Model

The core now has a private `TextFrontend` interface used by model loading and
request preparation. It has no mandatory G2P engine. A future public provider
registry may allow an application or optional component to supply G2P, but that
registry is deliberately outside the first implementation slice.

The Loaded Model capability snapshot is immutable. The runtime must not silently
substitute a frontend, guess a phoneme mapping, invoke an external executable, or
download resources. A package exposes only input kinds that are usable with its
loaded frontend.

## First Built-in Provider: `symbol_map`

The first bundled Text Frontend has canonical identity `synthesize.symbol_map`
and contract version 1. Its code is compiled into the core library. It consumes
only declarative vocabulary and blank-insertion metadata from the Model Package;
the package cannot add executable normalization or mapping logic.

The implemented contract accepts `SYNTH_INPUT_PHONEMES_UTF8` with
`unicode_scalar` mapping. It strictly validates UTF-8, maps every Unicode scalar
through the GGUF symbol table using the pinned `last_index_wins` rule, and applies
the VITS blank-insertion rule. Invalid UTF-8, malformed symbol metadata, and
unknown scalars return `SYNTH_ERR_TEXT_FRONTEND`. The final token limit is checked
after blank insertion. Exact token-ID input continues to bypass the frontend.

`synthesize.symbol_map` performs no grapheme-to-phoneme conversion, language or
accent detection, dictionary fallback, implicit Provider substitution, or network
access. It therefore advertises phoneme and token input, not raw-text input.
`delimited_symbol`, declarative text normalization, and an optional G2P provider
remain later slices.

eSpeak NG is not built, linked, or redistributed by the core project because its
official distribution is GPL-3.0-or-later. A future eSpeak-compatible Provider is
therefore optional and separately installed, licensed, and identified; it never
changes the behavior of `synthesize.symbol_map`. See the
[official eSpeak NG repository](https://github.com/espeak-ng/espeak-ng).

During VITS inference validation, pinned upstream tooling generates exact token
and phoneme sequences outside the runtime. This is validation tooling, not a
runtime dependency.

## Input Levels

1. Text input will run normalization, G2P when required, and model-specific symbol
   mapping. This level is not yet exposed by the VITS packages.
2. Phoneme input bypasses normalization and G2P, then uses the embedded
   model-specific symbol mapping. This level is implemented.
3. Token-sequence input bypasses the Text Frontend. This level remains available
   for reference validation and advanced callers.

All paths must produce or provide the exact token IDs expected by the selected
Model Variant. The General Model Capability query reports the positive maximum
length of the final sequence consumed by the synthesis graph.

## Validation Order

VITS validation first compares native inference with the upstream implementation
using identical token-ID sequences. Public phoneme input for `ˈeɪ.` is then
required to produce byte-identical PCM to the corresponding exact token sequence.
Both checks now pass for LJSpeech and VCTK across F32, F16, and Q8_MIXED.
End-to-end raw-text parity remains pending until a G2P provider is implemented.
