---
status: accepted
---

# Version native artifacts by ABI and discover backends relatively

synthesize.cpp installs one static or shared C SDK per build, exports one
`synthesize::synthesize` CMake target plus pkg-config metadata, and
versions shared-library compatibility names with `SYNTH_ABI_VERSION` rather than
the semantic major version. Optional dynamic backend modules live in a fixed
`synthesize/backends/` directory relative to the loaded core library, so default
discovery is relocatable and never depends on the working directory, `PATH`, a
Model Package, or a CLI-specific path.

## Considered Options

- Semantic-major SONAMEs were rejected because the release version advances
  independently of binary compatibility.
- Building and exporting static and shared variants from one build was rejected
  because it complicates one stable consumer target and creates archive/import
  naming conflicts on Windows.
- Working-directory and system-path backend scans were rejected because host
  process state could silently change which executable module is loaded.
