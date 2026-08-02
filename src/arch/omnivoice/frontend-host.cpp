#include "arch/omnivoice/frontend-host.h"

#include "codepoint-scan.h"
#include "unicode-ranges.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>

namespace synth::omnivoice {

namespace {

// --------------------------------------------------------------------------
// UTF-8 and character classes
// --------------------------------------------------------------------------

// What Python's `str.isspace` reports, which is both what `str.strip` removes
// and what `\s` matches in the reference's cleanup pattern -- the two agree
// across the whole code space, so one table serves both.
bool is_whitespace(uint32_t codepoint) {
    return in_ranges(codepoint, kUnicodeWhitespaceRanges, std::size(kUnicodeWhitespaceRanges));
}

// --------------------------------------------------------------------------
// combine_text
// --------------------------------------------------------------------------

// The reference's `chinese_range = r"[一-鿿]"`. Deliberately narrower
// than "the CJK blocks": kana, Hangul and the extension planes are outside it,
// so the spaces around them survive.
bool is_cjk_ideograph(uint32_t codepoint) {
    return codepoint >= 0x4E00 && codepoint <= 0x9FFF;
}

// Python's `str.strip()` with no argument.
std::string strip(const std::string & text) {
    size_t begin = 0;
    while (begin < text.size()) {
        size_t         length    = 0;
        const uint32_t codepoint = decode_utf8(text, begin, length);
        if (!is_whitespace(codepoint)) {
            break;
        }
        begin += length;
    }
    size_t end = text.size();
    while (end > begin) {
        // Walk back over continuation bytes to find the last character's start.
        size_t start = end - 1;
        while (start > begin && (static_cast<unsigned char>(text[start]) & 0xC0) == 0x80) {
            --start;
        }
        size_t         length    = 0;
        const uint32_t codepoint = decode_utf8(text, start, length);
        if (start + length != end || !is_whitespace(codepoint)) {
            break;
        }
        end = start;
    }
    return text.substr(begin, end - begin);
}

}  // namespace

std::string combine_text(const std::string & ref_text, const std::string & text) {
    // `if ref_text:` -- the raw string's truthiness, before stripping, so a
    // reference of only spaces still joins and contributes its separator.
    std::string full = ref_text.empty() ? strip(text) : strip(ref_text) + " " + strip(text);

    // 1. Delete carriage returns and newlines. They are removed, not replaced
    //    by a space, so words on adjacent lines run together.
    std::string without_breaks;
    without_breaks.reserve(full.size());
    for (char raw : full) {
        if (raw != '\r' && raw != '\n') {
            without_breaks.push_back(raw);
        }
    }

    // 2. Full-width parentheses become ASCII ones.
    std::string unparenthesized;
    unparenthesized.reserve(without_breaks.size());
    for (size_t offset = 0; offset < without_breaks.size();) {
        size_t         length    = 0;
        const uint32_t codepoint = decode_utf8(without_breaks, offset, length);
        if (codepoint == 0xFF08) {
            unparenthesized.push_back('(');
        } else if (codepoint == 0xFF09) {
            unparenthesized.push_back(')');
        } else {
            unparenthesized.append(without_breaks, offset, length);
        }
        offset += length;
    }

    // 3. Collapse runs of spaces and tabs -- only those two -- into one space.
    std::string collapsed;
    collapsed.reserve(unparenthesized.size());
    for (size_t offset = 0; offset < unparenthesized.size(); ++offset) {
        const char raw = unparenthesized[offset];
        if (raw != ' ' && raw != '\t') {
            collapsed.push_back(raw);
            continue;
        }
        collapsed.push_back(' ');
        while (offset + 1 < unparenthesized.size() &&
               (unparenthesized[offset + 1] == ' ' || unparenthesized[offset + 1] == '\t')) {
            ++offset;
        }
    }

    // 4. Remove whitespace that touches a CJK ideograph on either side. The
    //    reference writes this as `(?<=CJK)\s+|\s+(?=CJK)`, whose leftmost
    //    alternation over a greedy run comes to the same thing as dropping a
    //    maximal run whose neighbour on either side is an ideograph.
    std::string result;
    result.reserve(collapsed.size());
    uint32_t previous = 0;
    for (size_t offset = 0; offset < collapsed.size();) {
        size_t         length    = 0;
        const uint32_t codepoint = decode_utf8(collapsed, offset, length);
        if (!is_whitespace(codepoint)) {
            result.append(collapsed, offset, length);
            previous = codepoint;
            offset += length;
            continue;
        }

        // Take the whole run, then look at what follows it.
        const size_t run_begin = offset;
        while (offset < collapsed.size()) {
            size_t         step = 0;
            const uint32_t next = decode_utf8(collapsed, offset, step);
            if (!is_whitespace(next)) {
                break;
            }
            offset += step;
        }
        uint32_t following = 0;
        if (offset < collapsed.size()) {
            size_t step = 0;
            following   = decode_utf8(collapsed, offset, step);
        }
        if (!is_cjk_ideograph(previous) && !is_cjk_ideograph(following)) {
            result.append(collapsed, run_begin, offset - run_begin);
            previous = 0;
        }
    }
    return result;
}

namespace {

// Upstream's END_PUNCTUATION (omnivoice/utils/text.py:38-65), as the set of
// codepoints `text[-1] not in END_PUNCTUATION` (text.py:220) can actually
// test against -- see add_punctuation's own header comment for why the
// set's one two-character member, "……", needs no entry of its own.
constexpr uint32_t kEndPunctuation[] = {
    0x003B,  // ;
    0x003A,  // :
    0x002C,  // ,
    0x002E,  // .
    0x0021,  // !
    0x003F,  // ?
    0x2026,  // … (U+2026; also covers a trailing "……")
    0x0029,  // )
    0x005D,  // ]
    0x007D,  // }
    0x0022,  // "
    0x0027,  // '
    0x201C,  // “
    0x201D,  // ”
    0x2018,  // ‘
    0x2019,  // ’
    0xFF1B,  // ；
    0xFF1A,  // ：
    0xFF0C,  // ，
    0x3002,  // 。
    0xFF01,  // ！
    0xFF1F,  // ？
    0x3001,  // 、
    0xFF09,  // ）
    0x3011,  // 】
};

bool is_end_punctuation(uint32_t codepoint) {
    for (uint32_t candidate : kEndPunctuation) {
        if (candidate == codepoint) {
            return true;
        }
    }
    return false;
}

// The last codepoint of a UTF-8 string, or 0 for an empty one. Walks back
// over continuation bytes the same way `strip()` above does to find its own
// last character's start.
uint32_t last_codepoint(const std::string & text) {
    if (text.empty()) {
        return 0;
    }
    size_t start = text.size() - 1;
    while (start > 0 && (static_cast<unsigned char>(text[start]) & 0xC0) == 0x80) {
        --start;
    }
    size_t length = 0;
    return decode_utf8(text, start, length);
}

}  // namespace

std::string add_punctuation(const std::string & transcript) {
    std::string text = strip(transcript);
    if (text.empty()) {
        return text;
    }
    if (is_end_punctuation(last_codepoint(text))) {
        return text;
    }
    bool is_chinese = false;
    for (size_t offset = 0; offset < text.size();) {
        size_t         length    = 0;
        const uint32_t codepoint = decode_utf8(text, offset, length);
        if (is_cjk_ideograph(codepoint)) {
            is_chinese = true;
            break;
        }
        offset += length;
    }
    text += is_chinese ? "\xE3\x80\x82" /* 。 U+3002 */ : ".";
    return text;
}

std::string style_text(bool denoise, const std::string & language_tag, const std::string & instruct) {
    // `<|denoise|>` + `<|lang_start|>{lang}<|lang_end|>` +
    // `<|instruct_start|>{instruct}<|instruct_end|>`, with "None" for an empty
    // slot -- a string the model was trained on, not a placeholder.
    std::string out;
    if (denoise) {
        out += "<|denoise|>";
    }
    out += "<|lang_start|>";
    out += language_tag.empty() ? "None" : language_tag;
    out += "<|lang_end|><|instruct_start|>";
    out += instruct.empty() ? "None" : instruct;
    out += "<|instruct_end|>";
    return out;
}

namespace {

// The thirteen tags the reference's `_NONVERBAL_PATTERN` alternates over, in
// its order. No tag is a prefix of another, so matching the longest at a given
// offset and taking the alternation's leftmost branch are the same thing.
constexpr const char * kNonverbalTags[kNonverbalTagCount] = {
    "[laughter]",    "[sigh]",        "[confirmation-en]",     "[question-en]", "[question-ah]",
    "[question-oh]", "[question-ei]", "[question-yi]",         "[surprise-ah]", "[surprise-oh]",
    "[surprise-wa]", "[surprise-yo]", "[dissatisfaction-hnn]",
};

// The longest tag starting at `offset`, or nullptr.
const char * tag_at(const std::string & text, size_t offset) {
    const char * best = nullptr;
    size_t       size = 0;
    for (const char * tag : kNonverbalTags) {
        const std::string candidate(tag);
        if (candidate.size() > size && text.compare(offset, candidate.size(), candidate) == 0) {
            best = tag;
            size = candidate.size();
        }
    }
    return best;
}

}  // namespace

synth_status_t tokenize_wrapped_text(const TextFrontend &   frontend,
                                     const std::string &    wrapped,
                                     std::vector<int32_t> & ids) {
    ids.clear();

    std::vector<int32_t> piece;
    const auto           append = [&](const std::string & span) {
        // No output limit here: the request's frame limit is a separate cap,
        // applied against the canvas rather than the prompt.
        const synth_status_t status = frontend.prepare(SYNTH_INPUT_TEXT_UTF8, span.data(), span.size(), 0, piece);
        if (status == SYNTH_OK) {
            ids.insert(ids.end(), piece.begin(), piece.end());
        }
        return status;
    };

    size_t last  = 0;
    size_t found = 0;
    for (size_t cursor = 0; cursor < wrapped.size();) {
        const char * tag = wrapped[cursor] == '[' ? tag_at(wrapped, cursor) : nullptr;
        if (tag == nullptr) {
            ++cursor;
            continue;
        }
        if (cursor > last) {
            const synth_status_t status = append(wrapped.substr(last, cursor - last));
            if (status != SYNTH_OK) {
                ids.clear();
                return status;
            }
        }
        const std::string    text(tag);
        const synth_status_t status = append(text);
        if (status != SYNTH_OK) {
            ids.clear();
            return status;
        }
        cursor += text.size();
        last = cursor;
        ++found;
    }

    // No tag: one call over the whole string, which is what the reference falls
    // back to and what keeps this path identical to a plain prepare.
    //
    // The clear is not redundant. A frontend that fails partway through a span
    // returns the error with what it had already emitted still in the vector --
    // the byte-pair frontend does exactly that -- so passing `ids` straight out
    // would break this function's postcondition on the path most requests take.
    if (found == 0) {
        const synth_status_t status = frontend.prepare(SYNTH_INPUT_TEXT_UTF8, wrapped.data(), wrapped.size(), 0, ids);
        if (status != SYNTH_OK) {
            ids.clear();
        }
        return status;
    }

    if (last < wrapped.size()) {
        const synth_status_t status = append(wrapped.substr(last));
        if (status != SYNTH_OK) {
            ids.clear();
            return status;
        }
    }
    return SYNTH_OK;
}

// --------------------------------------------------------------------------
// DurationEstimator
// --------------------------------------------------------------------------

namespace {

enum class Script : uint8_t {
    Cjk,
    Hangul,
    Kana,
    Ethiopic,
    Yi,
    Indic,
    ThaiLao,
    KhmerMyanmar,
    Arabic,
    Hebrew,
    Latin,
    Cyrillic,
    Greek,
    Armenian,
    Georgian,
    Punctuation,
    Space,
    Digit,
    Mark,
    Fallback,
};

// The phonetic weight table, verbatim from `omnivoice/utils/duration.py` at the
// pinned source revision. 1.0 is one Latin letter, about 40-50 ms.
double weight_of(Script script) {
    switch (script) {
        case Script::Cjk:
            return 3.0;  // Chinese, Japanese Kanji, etc.
        case Script::Hangul:
            return 2.5;  // Korean Hangul
        case Script::Kana:
            return 2.2;  // Japanese Hiragana/Katakana
        case Script::Ethiopic:
            return 3.0;  // Amharic/Ge'ez
        case Script::Yi:
            return 3.0;  // Yi script
        case Script::Indic:
            return 1.8;  // Hindi, Bengali, Tamil, etc.
        case Script::ThaiLao:
            return 1.5;  // Thai, Lao
        case Script::KhmerMyanmar:
            return 1.8;  // Khmer, Myanmar
        case Script::Arabic:
            return 1.5;  // Arabic, Persian, Urdu
        case Script::Hebrew:
            return 1.5;  // Hebrew
        case Script::Latin:
            return 1.0;  // English, Spanish, French, Vietnamese (baseline)
        case Script::Cyrillic:
            return 1.0;  // Russian, Ukrainian
        case Script::Greek:
            return 1.0;  // Greek
        case Script::Armenian:
            return 1.0;  // Armenian
        case Script::Georgian:
            return 1.0;  // Georgian
        case Script::Punctuation:
            return 0.5;  // Pause capability
        case Script::Space:
            return 0.2;  // Word boundary/breath
        case Script::Digit:
            return 3.5;  // Numbers
        case Script::Mark:
            return 0.0;  // Diacritics/accents (silent modifiers)
        case Script::Fallback:
            return 1.0;  // Unknown scripts
    }
    return 1.0;
}

struct ScriptRange {
    uint32_t end;
    Script   script;
};

// The 88-entry `(End_Codepoint, Type_Key)` table, transcribed in order from
// `omnivoice/utils/duration.py` at the pinned source revision. The comments are
// upstream's. It is searched with a `bisect_left` over the end points, so an
// entry claims everything from the previous entry's end + 1 through its own --
// including the gaps between blocks, which is why "Cherokee" covers more than
// Cherokee.
constexpr ScriptRange kScriptRanges[] = {
    { 0x02AF, Script::Latin        }, // Latin (Basic, Supplement, Ext, IPA)
    { 0x03FF, Script::Greek        }, // Greek & Coptic
    { 0x052F, Script::Cyrillic     }, // Cyrillic
    { 0x058F, Script::Armenian     }, // Armenian
    { 0x05FF, Script::Hebrew       }, // Hebrew
    { 0x077F, Script::Arabic       }, // Arabic, Syriac, Arabic Supplement
    { 0x089F, Script::Arabic       }, // Arabic Extended-B (+ Syriac Supp)
    { 0x08FF, Script::Arabic       }, // Arabic Extended-A
    { 0x097F, Script::Indic        }, // Devanagari
    { 0x09FF, Script::Indic        }, // Bengali
    { 0x0A7F, Script::Indic        }, // Gurmukhi
    { 0x0AFF, Script::Indic        }, // Gujarati
    { 0x0B7F, Script::Indic        }, // Oriya
    { 0x0BFF, Script::Indic        }, // Tamil
    { 0x0C7F, Script::Indic        }, // Telugu
    { 0x0CFF, Script::Indic        }, // Kannada
    { 0x0D7F, Script::Indic        }, // Malayalam
    { 0x0DFF, Script::Indic        }, // Sinhala
    { 0x0EFF, Script::ThaiLao      }, // Thai & Lao
    { 0x0FFF, Script::Indic        }, // Tibetan (Abugida)
    { 0x109F, Script::KhmerMyanmar }, // Myanmar
    { 0x10FF, Script::Georgian     }, // Georgian
    { 0x11FF, Script::Hangul       }, // Hangul Jamo
    { 0x137F, Script::Ethiopic     }, // Ethiopic
    { 0x139F, Script::Ethiopic     }, // Ethiopic Supplement
    { 0x13FF, Script::Fallback     }, // Cherokee
    { 0x167F, Script::Fallback     }, // Canadian Aboriginal Syllabics
    { 0x169F, Script::Fallback     }, // Ogham
    { 0x16FF, Script::Fallback     }, // Runic
    { 0x171F, Script::Fallback     }, // Tagalog (Baybayin)
    { 0x173F, Script::Fallback     }, // Hanunoo
    { 0x175F, Script::Fallback     }, // Buhid
    { 0x177F, Script::Fallback     }, // Tagbanwa
    { 0x17FF, Script::KhmerMyanmar }, // Khmer
    { 0x18AF, Script::Fallback     }, // Mongolian
    { 0x18FF, Script::Fallback     }, // Canadian Aboriginal Syllabics Ext
    { 0x194F, Script::Indic        }, // Limbu
    { 0x19DF, Script::Indic        }, // Tai Le & New Tai Lue
    { 0x19FF, Script::KhmerMyanmar }, // Khmer Symbols
    { 0x1A1F, Script::Indic        }, // Buginese
    { 0x1AAF, Script::Indic        }, // Tai Tham
    { 0x1B7F, Script::Indic        }, // Balinese
    { 0x1BBF, Script::Indic        }, // Sundanese
    { 0x1BFF, Script::Indic        }, // Batak
    { 0x1C4F, Script::Indic        }, // Lepcha
    { 0x1C7F, Script::Indic        }, // Ol Chiki (Santali)
    { 0x1C8F, Script::Cyrillic     }, // Cyrillic Extended-C
    { 0x1CBF, Script::Georgian     }, // Georgian Extended
    { 0x1CCF, Script::Indic        }, // Sundanese Supplement
    { 0x1CFF, Script::Indic        }, // Vedic Extensions
    { 0x1D7F, Script::Latin        }, // Phonetic Extensions
    { 0x1DBF, Script::Latin        }, // Phonetic Extensions Supplement
    { 0x1DFF, Script::Fallback     }, // Combining Diacritical Marks Supplement
    { 0x1EFF, Script::Latin        }, // Latin Extended Additional (Vietnamese)
    { 0x309F, Script::Kana         }, // Hiragana
    { 0x30FF, Script::Kana         }, // Katakana
    { 0x312F, Script::Cjk          }, // Bopomofo (Pinyin)
    { 0x318F, Script::Hangul       }, // Hangul Compatibility Jamo
    { 0x9FFF, Script::Cjk          }, // CJK Unified Ideographs (Main)
    { 0xA4CF, Script::Yi           }, // Yi Syllables
    { 0xA4FF, Script::Fallback     }, // Lisu
    { 0xA63F, Script::Fallback     }, // Vai
    { 0xA69F, Script::Cyrillic     }, // Cyrillic Extended-B
    { 0xA6FF, Script::Fallback     }, // Bamum
    { 0xA7FF, Script::Latin        }, // Latin Extended-D
    { 0xA82F, Script::Indic        }, // Syloti Nagri
    { 0xA87F, Script::Fallback     }, // Phags-pa
    { 0xA8DF, Script::Indic        }, // Saurashtra
    { 0xA8FF, Script::Indic        }, // Devanagari Extended
    { 0xA92F, Script::Indic        }, // Kayah Li
    { 0xA95F, Script::Indic        }, // Rejang
    { 0xA97F, Script::Hangul       }, // Hangul Jamo Extended-A
    { 0xA9DF, Script::Indic        }, // Javanese
    { 0xA9FF, Script::KhmerMyanmar }, // Myanmar Extended-B
    { 0xAA5F, Script::Indic        }, // Cham
    { 0xAA7F, Script::KhmerMyanmar }, // Myanmar Extended-A
    { 0xAADF, Script::Indic        }, // Tai Viet
    { 0xAAFF, Script::Indic        }, // Meetei Mayek Extensions
    { 0xAB2F, Script::Ethiopic     }, // Ethiopic Extended-A
    { 0xAB6F, Script::Latin        }, // Latin Extended-E
    { 0xABBF, Script::Fallback     }, // Cherokee Supplement
    { 0xABFF, Script::Indic        }, // Meetei Mayek
    { 0xD7AF, Script::Hangul       }, // Hangul Syllables
    { 0xFAFF, Script::Cjk          }, // CJK Compatibility
    { 0xFDFF, Script::Arabic       }, // Arabic Presentation Forms-A
    { 0xFE6F, Script::Fallback     }, // Variation Selectors
    { 0xFEFF, Script::Arabic       }, // Arabic Presentation Forms-B
    { 0xFFEF, Script::Latin        }, // Fullwidth Latin
};

static_assert(std::size(kScriptRanges) == 88, "the transcribed table is upstream's 88 entries");

// `_get_char_weight`'s order, which is the whole of its behaviour: the general
// category is consulted before the script table, so punctuation and digits
// never reach the bisect and the bisect never has to exclude them.
Script classify(uint32_t codepoint) {
    if ((codepoint >= 'A' && codepoint <= 'Z') || (codepoint >= 'a' && codepoint <= 'z')) {
        return Script::Latin;
    }
    if (codepoint == 0x20) {
        return Script::Space;
    }
    // Arabic tatweel is a written elongation with no sound.
    if (codepoint == 0x0640) {
        return Script::Mark;
    }
    if (in_ranges(codepoint, kUnicodeMarkRanges, std::size(kUnicodeMarkRanges))) {
        return Script::Mark;
    }
    if (in_ranges(codepoint, kUnicodePunctuationSymbolRanges, std::size(kUnicodePunctuationSymbolRanges))) {
        return Script::Punctuation;
    }
    // Category Z only. A tab or a newline is category Cc and reaches the table
    // below instead, where it weighs as a Latin letter -- surprising, and
    // upstream's behaviour.
    if (in_ranges(codepoint, kUnicodeSeparatorRanges, std::size(kUnicodeSeparatorRanges))) {
        return Script::Space;
    }
    if (in_ranges(codepoint, kUnicodeNumberRanges, std::size(kUnicodeNumberRanges))) {
        return Script::Digit;
    }

    const ScriptRange * found =
        std::lower_bound(std::begin(kScriptRanges), std::end(kScriptRanges), codepoint,
                         [](const ScriptRange & range, uint32_t value) { return range.end < value; });
    if (found != std::end(kScriptRanges)) {
        return found->script;
    }
    // Upstream tests `code > 0x20000` -- strictly greater, so U+20000 itself is
    // `default` (1.0) and only U+20001 onward is `cjk`. Verified against
    // omnivoice/utils/duration.py at 468e927b; the asymmetry is upstream's, and
    // matching it is the contract.
    if (codepoint > 0x20000) {
        return Script::Cjk;
    }
    return Script::Fallback;
}

// The anchor pair the reference substitutes when a request carries no
// reference. Contract, not a default.
constexpr const char * kAnchorText   = "Nice to meet you.";
constexpr uint64_t     kAnchorFrames = 25;

constexpr double kLowThreshold  = 50.0;
constexpr double kBoostStrength = 3.0;

}  // namespace

double DurationEstimator::total_weight(const std::string & utf8) const {
    // Neumaier compensated summation, because the reference sums with Python's
    // `sum()` and CPython 3.12 -- the pinned oracle environment -- compensates
    // float sequences there. A plain running total agrees to about 1e-14, which
    // is enough to move a frame count across a truncation boundary.
    double total      = 0.0;
    double correction = 0.0;
    for (size_t offset = 0; offset < utf8.size();) {
        size_t         length    = 0;
        const uint32_t codepoint = decode_utf8(utf8, offset, length);
        offset += length;

        const double value = weight_of(classify(codepoint));
        const double sum   = total + value;
        if (std::fabs(total) >= std::fabs(value)) {
            correction += (total - sum) + value;
        } else {
            correction += (value - sum) + total;
        }
        total = sum;
    }
    return total + correction;
}

double DurationEstimator::estimate_duration(const std::string & target,
                                            const std::string & ref,
                                            double              ref_duration) const {
    if (ref_duration <= 0.0 || ref.empty()) {
        return 0.0;
    }
    const double ref_weight = total_weight(ref);
    if (ref_weight == 0.0) {
        return 0.0;
    }

    //     speed_factor = total_weight(ref) / ref_duration
    //     estimated    = total_weight(target) / speed_factor
    const double speed_factor = ref_weight / ref_duration;
    const double estimated    = total_weight(target) / speed_factor;

    //     if estimated < 50:
    //         estimated = 50 * (estimated / 50) ** (1 / 3)
    //
    // Short utterances are read slower than the proportion predicts, so the
    // curve pulls them back toward the threshold rather than clamping at it.
    if (estimated < kLowThreshold) {
        return kLowThreshold * std::pow(estimated / kLowThreshold, 1.0 / kBoostStrength);
    }
    return estimated;
}

uint64_t DurationEstimator::estimate_target_frames(const std::string & text,
                                                   const std::string & ref_text,
                                                   uint64_t            ref_frames,
                                                   float               speaking_rate) const {
    std::string reference = ref_text;
    uint64_t    frames    = ref_frames;
    if (reference.empty() || frames == 0) {
        reference = kAnchorText;
        frames    = kAnchorFrames;
    }

    double estimated = estimate_duration(text, reference, static_cast<double>(frames));
    // `if speed > 0 and speed != 1.0` -- a rate of zero is left alone rather
    // than treated as an error, which is upstream's guard verbatim.
    if (speaking_rate > 0.0f && speaking_rate != 1.0f) {
        estimated /= static_cast<double>(speaking_rate);
    }

    // `max(1, int(est))`: truncation toward zero, then a floor of one frame. A
    // request always gets a canvas.
    const double truncated = std::trunc(estimated);
    if (!(truncated >= 1.0)) {
        return 1;
    }
    // Every double past 2^53 is already an integer, and converting a larger one
    // to uint64_t is only defined below its maximum. The request's frame limit
    // is the real ceiling; this only keeps the conversion in range.
    constexpr double kConvertible = 9007199254740992.0;  // 2^53
    if (truncated >= kConvertible) {
        return static_cast<uint64_t>(kConvertible);
    }
    return static_cast<uint64_t>(truncated);
}

// --------------------------------------------------------------------------
// assemble_prompt_ids
// --------------------------------------------------------------------------

bool assemble_prompt_ids(const TextFrontend &   frontend,
                         const SpecialTokens &  tokens,
                         bool                   denoise,
                         const std::string &    language_tag,
                         const std::string &    instruct,
                         const std::string &    combined_text,
                         std::vector<int32_t> & output) {
    std::string wrapped = style_text(denoise, language_tag, instruct);
    wrapped += "<|text_start|>";
    wrapped += combined_text;
    wrapped += "<|text_end|>";

    if (tokenize_wrapped_text(frontend, wrapped, output) != SYNTH_OK) {
        output.clear();
        return false;
    }
    // The composed string always closes on the text-end marker, and nothing
    // in `wrapped` follows it, so tokenize_wrapped_text -- which matches it
    // literally -- always emits its id last. build_prompt_grid (model.cpp)
    // finds the target region by LENGTH alone; a frontend that silently
    // dropped or reordered the closing marker would corrupt every row
    // without tripping a status, which is exactly the failure mode this
    // family's docs warn produces different, still-plausible speech rather
    // than an error -- so this is enforced on the RELEASE path, not left to
    // an assert: both trees this project builds (Release and
    // RelWithDebInfo) define NDEBUG, which would make a plain assert() here
    // inert in every standard build configuration, and a silently corrupted
    // prompt is worse than a synthesis refused before it starts.
    if (output.empty() || output.back() != int32_t(tokens.text_end)) {
        output.clear();
        return false;
    }
    return true;
}

}  // namespace synth::omnivoice
