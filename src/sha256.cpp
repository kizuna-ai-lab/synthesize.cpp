#include "sha256.h"

#include <cstring>

namespace synth {

namespace {

constexpr uint32_t kInitialHash[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                       0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };

// FIPS 180-4 section 4.2.2: the first 32 bits of the fractional parts of the
// cube roots of the first 64 primes.
constexpr uint32_t kRoundConstants[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

inline uint32_t rotr(uint32_t value, uint32_t bits) {
    return (value >> bits) | (value << (32u - bits));
}

// Absorbs exactly one 64-byte block into `state`, per FIPS 180-4 section
// 6.2.2.
void process_block(uint32_t state[8], const uint8_t block[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (uint32_t(block[i * 4]) << 24) | (uint32_t(block[i * 4 + 1]) << 16) | (uint32_t(block[i * 4 + 2]) << 8) |
               uint32_t(block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i]              = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    for (int i = 0; i < 64; ++i) {
        const uint32_t s1    = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const uint32_t ch    = (e & f) ^ ((~e) & g);
        const uint32_t temp1 = h + s1 + ch + kRoundConstants[i] + w[i];
        const uint32_t s0    = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const uint32_t maj   = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temp2 = s0 + maj;
        h                    = g;
        g                    = f;
        f                    = e;
        e                    = d + temp1;
        d                    = c;
        c                    = b;
        b                    = a;
        a                    = temp1 + temp2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

}  // namespace

void sha256(const void * data, size_t size, uint8_t out_digest[32]) {
    uint32_t state[8];
    std::memcpy(state, kInitialHash, sizeof(state));

    const auto * bytes = static_cast<const uint8_t *>(data);
    if (bytes == nullptr) {
        // `bytes + offset` and the tail memcpy below are UB even at a
        // zero-length offset when the base pointer is null. Every current
        // caller passes a real buffer or std::string::data() (guaranteed
        // non-null even for ""), so this is not reachable today -- it is
        // defensive hardening for the mandatory UBSan gate.
        size = 0;
    }
    size_t offset = 0;
    while (offset + 64 <= size) {
        process_block(state, bytes + offset);
        offset += 64;
    }

    // Final padding (FIPS 180-4 section 5.1.1): a single 0x80 byte, zeros,
    // then the original bit length as a big-endian uint64 -- all packed into
    // one block if it fits after the 0x80 marker, otherwise spilling into a
    // second. `remaining` is at most 63, so 128 bytes always suffices for
    // either case.
    uint8_t      tail[128] = {};
    const size_t remaining = size - offset;
    // `bytes` is null only when `size` was forced to 0 above, which makes
    // `remaining` 0 too -- skip the copy rather than forming `bytes + offset`
    // (pointer arithmetic on null) and calling memcpy with a null source,
    // both UB even at a zero count.
    if (remaining > 0) {
        std::memcpy(tail, bytes + offset, remaining);
    }
    tail[remaining]           = 0x80;
    const uint64_t bit_length = uint64_t(size) * 8;
    const size_t   total_len  = (remaining + 1 + 8 <= 64) ? 64 : 128;
    for (int i = 0; i < 8; ++i) {
        tail[total_len - 1 - i] = uint8_t(bit_length >> (8 * i));
    }

    process_block(state, tail);
    if (total_len == 128) {
        process_block(state, tail + 64);
    }

    for (int i = 0; i < 8; ++i) {
        out_digest[i * 4 + 0] = uint8_t(state[i] >> 24);
        out_digest[i * 4 + 1] = uint8_t(state[i] >> 16);
        out_digest[i * 4 + 2] = uint8_t(state[i] >> 8);
        out_digest[i * 4 + 3] = uint8_t(state[i]);
    }
}

}  // namespace synth
