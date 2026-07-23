---
status: superseded
superseded-by: 0014-separate-python-api-wheel-from-native-providers
---

# Bundle a matching CPU library in Python abi3 wheels

This decision was superseded on 2026-07-21 by ADR 0014. The stable-ABI extension,
ownership rules, and controlled native loading remain, but the native artifacts
are supplied by separate Native Provider distributions rather than stored in the
API wheel itself.

synthesize.cpp publishes `synthesize-cpp` as a CPython 3.11+ `abi3` wheel with
one thin Limited-API C Adapter and its exact matching shared CPU library. The
extension opens only the package-relative library and validates ABI, minimum
version, and required symbols before exposing Python objects. Stable-ABI
ownership and buffer mechanisms retain native resources. This gives ordinary
Python installation deterministic compatibility without bypassing the public C
seam or requiring NumPy, a compiler, or a system synthesize.cpp installation.

## Considered Options

- Per-Python-minor extension wheels were rejected because the required capsule
  and buffer facilities are available from the selected Python 3.11 Limited API.
- A pure ctypes or CFFI-only binding was rejected because a thin extension gives
  one controlled place for GIL transitions, callback exception capture,
  native-owner finalization, and buffer exporting.
- System-library discovery in official PyPI wheels was rejected because host
  search state could load a native release different from the wheel's generated
  declarations.
- Statically linking the native core into the extension was rejected because a
  separately bundled shared core preserves explicit native ABI validation and the
  established dynamic backend-module layout.

No public C Interface changes result from this packaging choice.
