#pragma once

#include "unicode-ranges.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace synth {

// Decodes one UTF-8 sequence, reporting how many bytes it consumed. A malformed
// byte is taken as itself rather than rejected: the caller is tokenizing text
// that has already been accepted at the public seam.
inline uint32_t decode_utf8(const std::string & text, size_t offset, size_t & length) {
    const unsigned char lead      = static_cast<unsigned char>(text[offset]);
    size_t              want      = 1;
    uint32_t            codepoint = lead;
    if ((lead & 0xE0) == 0xC0) {
        want      = 2;
        codepoint = lead & 0x1Fu;
    } else if ((lead & 0xF0) == 0xE0) {
        want      = 3;
        codepoint = lead & 0x0Fu;
    } else if ((lead & 0xF8) == 0xF0) {
        want      = 4;
        codepoint = lead & 0x07u;
    }
    if (offset + want > text.size()) {
        length = 1;
        return lead;
    }
    for (size_t index = 1; index < want; ++index) {
        const unsigned char next = static_cast<unsigned char>(text[offset + index]);
        if ((next & 0xC0) != 0x80) {
            length = 1;
            return lead;
        }
        codepoint = (codepoint << 6) | (next & 0x3Fu);
    }
    length = want;
    return codepoint;
}

inline bool in_ranges(uint32_t codepoint, const CodepointRange * ranges, size_t count) {
    size_t low  = 0;
    size_t high = count;
    while (low < high) {
        const size_t middle = low + (high - low) / 2;
        if (codepoint < ranges[middle].low) {
            high = middle;
        } else if (codepoint > ranges[middle].high) {
            low = middle + 1;
        } else {
            return true;
        }
    }
    return false;
}

}  // namespace synth
