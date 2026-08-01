// model-info.h's family-independent helper: the Profile Compatibility ID
// hex-to-bytes decode Task 14 needs to fill VoiceProfileInfo from a package's
// GGUF metadata (ProfileContract::compatibility_id_hex is validated as 64
// lowercase hex characters at load time, weights.cpp's own is_sha256_hex;
// this is the inverse of writing that same shape out).

#include "model-info.h"
#include "test-assert.h"

#include <cstring>
#include <string>

namespace {

int check_decode_profile_compatibility_id() {
    // A concrete, arbitrary 32-byte value with every nibble represented, both
    // cases of hex digit exercised as SOURCE text (lowercase only is valid
    // input, but the decode itself must read '0'-'9' and 'a'-'f' correctly
    // across the whole range) and a leading/trailing byte that is not
    // symmetric, so a byte-order bug would not cancel itself out.
    const std::string hex = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    SYNTH_TEST_CHECK(hex.size() == 64);
    const uint8_t expected[32] = { 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45,
                                   0x67, 0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab,
                                   0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef };
    uint8_t       decoded[32];
    std::memset(decoded, 0xa5, sizeof(decoded));
    SYNTH_TEST_CHECK(synth::decode_profile_compatibility_id(hex, decoded));
    SYNTH_TEST_CHECK(std::memcmp(decoded, expected, sizeof(expected)) == 0);

    // All-zero and all-'f' bound the nibble range on both sides.
    uint8_t zero_bytes[32];
    SYNTH_TEST_CHECK(synth::decode_profile_compatibility_id(std::string(64, '0'), zero_bytes));
    for (uint8_t byte : zero_bytes) {
        SYNTH_TEST_CHECK(byte == 0x00);
    }
    uint8_t max_bytes[32];
    SYNTH_TEST_CHECK(synth::decode_profile_compatibility_id(std::string(64, 'f'), max_bytes));
    for (uint8_t byte : max_bytes) {
        SYNTH_TEST_CHECK(byte == 0xff);
    }

    // Wrong length: one character short and one character long.
    uint8_t untouched[32];
    std::memset(untouched, 0xa5, sizeof(untouched));
    SYNTH_TEST_CHECK(!synth::decode_profile_compatibility_id(std::string(63, '0'), untouched));
    SYNTH_TEST_CHECK(!synth::decode_profile_compatibility_id(std::string(65, '0'), untouched));
    SYNTH_TEST_CHECK(!synth::decode_profile_compatibility_id("", untouched));
    // Left untouched on refusal -- the guard against reading a partial decode.
    for (uint8_t byte : untouched) {
        SYNTH_TEST_CHECK(byte == 0xa5);
    }

    // Uppercase hex is not the shape the package stores (lowercase only,
    // weights.cpp's is_sha256_hex) and is refused rather than silently
    // accepted as an alternate spelling.
    std::string uppercase = hex;
    for (char & character : uppercase) {
        if (character >= 'a' && character <= 'f') {
            character = static_cast<char>(character - 'a' + 'A');
        }
    }
    SYNTH_TEST_CHECK(!synth::decode_profile_compatibility_id(uppercase, untouched));

    // A non-hex character anywhere in the string is refused.
    std::string bad_character = hex;
    bad_character[10]         = 'g';
    SYNTH_TEST_CHECK(!synth::decode_profile_compatibility_id(bad_character, untouched));
    std::string bad_symbol = hex;
    bad_symbol[63]         = '-';
    SYNTH_TEST_CHECK(!synth::decode_profile_compatibility_id(bad_symbol, untouched));

    return 0;
}

}  // namespace

int main() {
    SYNTH_TEST_CHECK(check_decode_profile_compatibility_id() == 0);
    return 0;
}
