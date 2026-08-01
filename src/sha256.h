#pragma once

#include <cstddef>
#include <cstdint>

namespace synth {

// Minimal, dependency-free SHA-256 (FIPS 180-4) over one contiguous byte
// buffer. Used only for the v1 Serialized Voice Profile's `content_sha256`
// integrity field (ADR 0008, docs/c-interface.md's "v1 Serialized Profile
// GGUF Contract"): a corruption check, never a signature, so this
// implementation makes no attempt at side-channel resistance and offers no
// incremental/streaming interface -- a Serialized Profile is small enough
// (bounded by the family's own `max_total_frames`) to hash in one call.
//
// No third-party dependency exists in this codebase for this (the Python
// conversion tooling uses `hashlib.sha256`, which has no C++ counterpart
// here; ggml itself carries no hashing of any kind -- see
// docs/porting/families/omnivoice.md's Serialized Profile section for the
// resolution this vendors instead of pulling in a dependency for 32 bytes of
// digest).
//
// `out_digest` receives the 32-byte digest, big-endian per word as FIPS
// 180-4 defines the output (not related to the little-endian GGUF container
// this digest is stored inside).
void sha256(const void * data, size_t size, uint8_t out_digest[32]);

}  // namespace synth
