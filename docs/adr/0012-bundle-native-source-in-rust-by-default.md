---
status: accepted
---

# Bundle matching native source in Rust by default

synthesize.cpp publishes a low-level `synthesize-cpp-sys` crate and a safe
`synthesize-cpp` Adapter crate. The sys crate builds matching bundled native
source as a static CPU library by default, with explicit bundled GPU features,
while its `system` feature selects a separately installed SDK and cannot be
combined with backend-build features. This makes the ordinary Rust build
reproducible and version-matched without hiding the native C seam or preventing
system integration.

## Considered Options

- A system-library-only crate was rejected because a normal Cargo build would
  depend on an independently installed and potentially incompatible SDK.
- One crate containing both generated unsafe declarations and the safe Adapter
  was rejected because native build/link concerns and Rust ownership ergonomics
  have different consumers.
- Dynamic backend modules in the bundled default were rejected because the
  bundled core is static and explicit Cargo backend features can register the
  same implementations without runtime artifact discovery.
- Applying backend-build features to `system` mode was rejected because Cargo
  cannot change which backends a preinstalled native package contains.
