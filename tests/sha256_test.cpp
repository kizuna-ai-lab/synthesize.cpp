// Known-answer tests for src/sha256.{h,cpp}, the vendor-free SHA-256 the
// Serialized Voice Profile envelope (ADR 0008) hashes its `content_sha256`
// field with. Three NIST/FIPS 180-4 vectors, chosen to exercise the three
// distinct shapes the finalization padding can take:
//   * the empty message (0 bytes: padding alone fills the one block)
//   * "abc" (3 bytes: the common one-block smoke test)
//   * the 56-byte two-block vector (56 + 1 marker + 8 length = 65 > 64,
//     forcing the padding to spill into a second block -- the one edge case
//     a short input can't reach)
// Expected digests are the standard, widely published SHA-256 test vectors
// for these exact inputs.

#include "sha256.h"

#include "test-assert.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace {

bool parse_hex32(const char * hex, uint8_t (&out)[32]) {
    if (std::strlen(hex) != 64) {
        return false;
    }
    for (size_t i = 0; i < 32; ++i) {
        unsigned int byte = 0;
        if (std::sscanf(hex + i * 2, "%2x", &byte) != 1) {
            return false;
        }
        out[i] = uint8_t(byte);
    }
    return true;
}

// Returns 0 on a match, or (via SYNTH_TEST_CHECK's own `return 1`) a nonzero
// failure code -- SYNTH_TEST_CHECK expands to a bare `return`, so every
// function using it has to return `int` the way main() does.
int check_vector(const std::string & message, const char * expected_hex) {
    uint8_t expected[32];
    SYNTH_TEST_CHECK(parse_hex32(expected_hex, expected));

    uint8_t actual[32];
    synth::sha256(message.data(), message.size(), actual);
    SYNTH_TEST_CHECK(std::memcmp(actual, expected, sizeof(actual)) == 0);
    return 0;
}

}  // namespace

int main() {
    // FIPS 180-2 change notice 1 / widely reproduced empty-string vector.
    if (int rc = check_vector("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")) {
        return rc;
    }
    // FIPS 180-2 Appendix B.1's one-block message.
    if (int rc = check_vector("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")) {
        return rc;
    }
    // FIPS 180-2 Appendix B.2's two-block message.
    if (int rc = check_vector("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1")) {
        return rc;
    }
    return 0;
}
