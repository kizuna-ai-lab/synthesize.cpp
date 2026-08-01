# Vendored: libsamplerate

- **Tag:** `0.2.2`
- **Origin URL:** https://github.com/libsndfile/libsamplerate/releases/download/0.2.2/libsamplerate-0.2.2.tar.xz
- **Tarball sha256:** `3258da280511d24b49d6b08615bbe824d0cacc9842b0e4caf11c52cf2b043893`
- **License:** BSD-2-Clause, verified from the tarball's own `COPYING` (not from
  memory); see `THIRD_PARTY_NOTICES.md` for the transcribed notice.

## Why

ADR 0009 (`docs/adr/0009-pin-reference-audio-normalization.md`) pins Reference
Audio normalization to a statically-compiled, vendored libsamplerate 0.2.2
using `SRC_SINC_BEST_QUALITY`, rather than a system library an embedding
application could silently swap for a different implementation or version.

## File subset kept

Only:

- `src/*.c`
- `src/*.h`
- `include/samplerate.h`
- `COPYING`

Dropped from the release tarball: autotools/libtool build files (`configure`,
`configure.ac`, `Makefile.am`, `Makefile.in`, `aclocal.m4`, `build-aux/`,
`m4/`), the upstream CMake build (`CMakeLists.txt` at every level,
`cmake/`, `config.h.in`, `src/config.h.in`, `samplerate.pc.in`,
`libsamplerate.spec.in`, `src/Version_script.in`, `src/check_asm.sh`),
`examples/`, `tests/`, `docs/`, `Octave/`, `Win32/`, and the repository
metadata files (`AUTHORS`, `ChangeLog`, `NEWS`, `README.md`, `INSTALL`,
`autogen.sh`).

None of the kept files have been edited. Configuration that upstream would
normally supply via a generated `config.h` is instead supplied as compile
definitions on the `synth-libsamplerate` CMake target (see root
`CMakeLists.txt`); `HAVE_CONFIG_H` is never defined, so the vendored sources'
own `#ifdef HAVE_CONFIG_H` / `#include "config.h"` guards never fire and no
`config.h` is needed at all.

## Re-syncing to a newer release

1. Download the new release tarball from the same origin (a different tag)
   and record its sha256 here.
2. Replace the four kept locations with the new tarball's copies, verbatim.
3. Re-check this file's compile-definition assumptions still hold: search the
   new `src/*.c` for `HAVE_CONFIG_H`, `ENABLE_SINC_*`, `PACKAGE`, `VERSION`,
   `HAVE_STDBOOL_H`, and `CPU_CLIPS_*` before assuming the CMake target in the
   root `CMakeLists.txt` needs no changes.
