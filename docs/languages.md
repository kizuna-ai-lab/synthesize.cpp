# Language Capabilities

Status: Confirmed on 2026-07-21.

Language support is declared and validated per model variant. The project interest order is English, Mandarin Chinese, the Core European Set, Japanese, and Korean, but this order does not govern model-family selection or require one variant to cover every language.

## Confirmed

- Model families are selected for their architecture rather than their language breadth.
- A supported family may contain variants with different language capabilities.
- A variant's language is advertised only after that variant and language combination passes project validation.
- Mandarin Chinese means Standard Mandarin (`zh-CN`).
- Cantonese and other Sinitic varieties require separate future support and are not implied by Mandarin support.
- The English baseline is generic English (`en`).
- Locale and accent are optional capabilities and are recorded only when supported by reliable model or training documentation.
- A request for a specific English locale may fall back to `en`, but the resolved language and fallback must be reported to the caller.
- An undocumented accent remains unknown and is never inferred from listening alone.
- The Core European Set is French (`fr`), German (`de`), Spanish (`es`), Italian (`it`), and Portuguese (`pt`).
- Dutch, Polish, Russian, and Nordic languages belong to later expansion rather than the Core European Set.

## Request Selection

- `language_tag` selects an exact advertised language or locale such as `en`, `zh-CN`, or `ja`, or a regional tag explicitly permitted to fall back; it is never guessed from the input text.
- Omitting `language_tag` selects the Model Package default. A multilingual variant without a declared default requires an explicit tag.
- Language selection remains meaningful for token-ID input because it may select independent language conditioning.
- `voice_id` selects an identifier from the Loaded Model's Voice metadata. It may be omitted only when the package declares a default Voice.
- Accent is metadata of a Voice rather than an independent request field. Unknown accents are not converted into inferred controls.

## Capability Catalog

Every Supported Model Variant exposes at least one canonical BCP 47 language tag through an immutable Loaded Model-level Catalog. The entries are the Model Variant's validated language capabilities rather than languages inferred from its Model Family, tokenizer, training data name, or Preset Voices. At most one entry is the package default; a multilingual Model Variant may deliberately have no default and require explicit selection.

Exact advertised tags take precedence. A base-language entry may explicitly allow the narrow regional fallback from the same primary language plus one region subtag, such as `en-US` or `en-GB` to `en`. This permission does not cover scripts, variants, extensions, private-use subtags, or a different primary language, and a locale-specific capability such as `zh-CN` cannot serve as a fallback for `zh-HK` or `zh-TW`. Requests match BCP 47 tags case-insensitively and results report the package's canonical resolved tag.

The Catalog is Model Variant-level only. Whether a particular Preset Voice can be combined with a particular advertised language is not encoded by adding language fields to Preset Voice entries. v1 exposes no separate combination preflight query; the caller uses known Model Package semantics and synthesis performs authoritative validation.
