---
status: accepted
---

# Separate text frontends from model families

synthesize.cpp keeps text normalization, grapheme-to-phoneme conversion, and symbol mapping outside model-family inference implementations. The selected model variant declares its required text frontend, while the VITS or other family implementation consumes token IDs; direct phoneme and token-sequence inputs provide controlled bypasses. The core defines a provider interface and registry but requires no particular G2P engine, so providers may be built in or supplied optionally. This prevents language rules from becoming architecture rules and allows new frontends to be added without modifying a validated synthesis graph.
