---
status: accepted
---

# Require version and symbol checks after an ABI match

synthesize.cpp treats an exact `SYNTH_ABI_VERSION` match as necessary but
not sufficient for dynamic loading because compatible additions can add ordinary
symbols or size-tagged fields without changing binary layout. Official Rust and
Python Adapters therefore check the runtime's semantic version against a generated
minimum native version and eagerly resolve a header-generated required-symbol
manifest before exposing any higher-level object. This guarantees an older caller
can use a newer library while a newer caller uses an older library only when both
checks succeed.

## Considered Options

- ABI equality alone was rejected because an older library can share the ABI
  counter while lacking the behavior of a newer appended field.
- A symbol manifest without a minimum version was rejected because structure-field
  additions do not necessarily add a symbol.
- A `synth_get_api()` function table was rejected because it would duplicate the
  ordinary exported C Interface and create a second version-negotiation seam.
