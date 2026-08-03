#pragma once

// The BCP-47 REGION subtag shape, shared by every "does this tag fall back to
// a package's own primary-language entry" check in this project.
//
// docs/c-interface.md:341, SYNTH_LANGUAGE_REGIONAL_FALLBACK's contract,
// verbatim: the flag "permits a request consisting of that same primary
// language plus ONE REGION SUBTAG, such as en-US or en-GB ... It does not
// accept script, variant, extension, or private-use additions." A BCP-47
// region subtag (RFC 5646 2.2.4) is exactly one of: two ASCII letters
// (ISO 3166-1 alpha-2, ci) or three ASCII digits (UN M.49) -- nothing else,
// and nothing more than that one subtag.
//
// Before this header existed, three independent call sites each implemented
// "primary subtag matches, and the tag contains a '-'" without checking what
// followed that '-' at all -- accepting a script ("en-Latn"), a three-letter
// region-shaped string that isn't one ("en-USA"), or a full script+region tag
// ("en-Latn-US") wherever a plain two-letter/three-digit region belongs.
// PR #6 triage FIX 5 hoists the one correct check the three copies were each
// missing, rather than fixing each copy in place a fourth time this project
// would then owe itself a comment about (see profile.cpp's own former header
// comment on is_cjk_ideograph for this project's general tolerance for ONE
// small duplicate before hoisting -- three was past that point).
//
// Call sites: src/synthesis-request.cpp's `declared_language` (the core
// path, shared by all four families), src/voice-profile.cpp's
// `declared_language` (the Reference Audio/Description Text `language_tag`
// step), and src/arch/omnivoice/profile.cpp's `declared_language_tag` (the
// Serialized Voice Profile loader's parity copy).

#include <cctype>
#include <cstddef>

namespace synth {

// True iff `[value, value + size)` is exactly one BCP-47 region subtag: two
// ASCII letters or three ASCII digits, case-insensitive on the letters
// (matching every `equals_ascii_case` comparison at each call site). False
// for anything shorter, longer, or containing a byte outside that shape --
// including a further '-', which correctly refuses a caller that passed a
// full multi-subtag suffix instead of the one subtag this contract allows.
inline bool is_bcp47_region_subtag(const char * value, size_t size) {
    if (value == nullptr) {
        return false;
    }
    if (size == 2) {
        for (size_t index = 0; index < 2; ++index) {
            const unsigned char ch = static_cast<unsigned char>(value[index]);
            if (ch >= 128 || std::isalpha(ch) == 0) {
                return false;
            }
        }
        return true;
    }
    if (size == 3) {
        for (size_t index = 0; index < 3; ++index) {
            const unsigned char ch = static_cast<unsigned char>(value[index]);
            if (ch >= 128 || std::isdigit(ch) == 0) {
                return false;
            }
        }
        return true;
    }
    return false;
}

}  // namespace synth
