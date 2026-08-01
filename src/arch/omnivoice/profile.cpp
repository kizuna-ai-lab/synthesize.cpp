#include "arch/omnivoice/profile.h"

#include "arch/omnivoice/frontend-host.h"
#include "arch/omnivoice/omnivoice.h"
#include "codepoint-scan.h"
#include "text-frontend.h"

#include <cctype>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace synth::omnivoice {

synth_status_t create_clone_prompt(Model &                              model,
                                   const std::vector<float> &           pcm_24k,
                                   const std::string &                  transcript,
                                   const std::string &                  language_tag,
                                   int                                  threads,
                                   std::shared_ptr<const ClonePrompt> & output,
                                   const char *&                        out_diagnostic_code,
                                   const char *&                        out_diagnostic_message) {
    output.reset();
    out_diagnostic_code    = nullptr;
    out_diagnostic_message = nullptr;

    ReferenceEncoding    encoding;
    const synth_status_t encode_status = model.encode_reference(pcm_24k, threads, encoding);
    if (encode_status != SYNTH_OK) {
        return encode_status;
    }

    // jiangzhuo's ruling, 2026-08-01 (docs/porting/families/omnivoice.md's
    // "Silent-Reference Rejection" note): upstream's own `_post_process_audio`
    // (omnivoice/models/omnivoice.py:898-899) multiplies a digitally silent
    // reference's synthesized output by zero rather than refusing the
    // request, so a silent clip still "succeeds" and hands back silence. This
    // port refuses instead -- silence is not a cloned voice, and shipping one
    // as if it were would be indistinguishable, from the caller's side, from
    // a real defect swallowing the whole reference. Deliberate divergence,
    // not an oversight.
    if (encoding.ref_rms == 0.0f) {
        out_diagnostic_code    = "voice_profile.reference_silent";
        out_diagnostic_message = "the reference audio is digitally silent; nothing can be cloned from it";
        return SYNTH_ERR_INVALID_ARG;
    }

    const std::string                         canonical_transcript = add_punctuation(transcript);
    std::vector<int32_t>                      transcript_ids;
    const std::shared_ptr<const TextFrontend> frontend = model.text_frontend();
    if (frontend == nullptr || frontend->prepare(SYNTH_INPUT_TEXT_UTF8, canonical_transcript.data(),
                                                 canonical_transcript.size(), 0, transcript_ids) != SYNTH_OK) {
        out_diagnostic_code    = "voice_profile.transcript_unencodable";
        out_diagnostic_message = "the reference transcript contains a byte this package's frontend has no token for";
        return SYNTH_ERR_INVALID_ARG;
    }

    auto prompt              = std::make_shared<ClonePrompt>();
    prompt->reference_tokens = std::move(encoding.tokens);
    prompt->transcript_ids   = std::move(transcript_ids);
    prompt->transcript_text  = canonical_transcript;
    prompt->ref_rms          = encoding.ref_rms;
    prompt->language_tag     = language_tag;
    output                   = std::move(prompt);
    return SYNTH_OK;
}

namespace {

// ---------------------------------------------------------------------------
// The voice-design instruct vocabulary, transcribed verbatim from
// omnivoice/utils/voice_design.py:31-97 at the pinned revision (see
// scripts/envs/omnivoice/.venv/lib/python3.12/site-packages/omnivoice/utils/
// voice_design.py, read end to end for this port). Six mutually-exclusive
// categories: four EN<->ZH attribute dicts (gender, age, pitch, style),
// then two flat, single-language sets (accent EN-only, dialect ZH-only) --
// this order matches `_INSTRUCT_CATEGORIES` exactly, which is what keeps a
// future source diff checkable against this file line for line.
// ---------------------------------------------------------------------------

struct AttributeItem {
    const char * en;
    const char * zh;
};

// voice_design.py:32-47 (the four dict categories).
const std::vector<std::vector<AttributeItem>> & dict_categories() {
    static const std::vector<std::vector<AttributeItem>> categories = {
        // gender
        { { "male", "男" }, { "female", "女" } },
        // age
        {
         { "child", "儿童" },
         { "teenager", "少年" },
         { "young adult", "青年" },
         { "middle-aged", "中年" },
         { "elderly", "老年" },
         },
        // pitch
        {
         { "very low pitch", "极低音调" },
         { "low pitch", "低音调" },
         { "moderate pitch", "中音调" },
         { "high pitch", "高音调" },
         { "very high pitch", "极高音调" },
         },
        // style
        { { "whisper", "耳语" } },
    };
    return categories;
}

// voice_design.py:49-60: English-only, no Chinese counterpart.
const std::vector<std::string> & accent_category() {
    static const std::vector<std::string> accents = {
        "american accent", "british accent", "australian accent", "chinese accent", "canadian accent",
        "indian accent",   "korean accent",  "portuguese accent", "russian accent", "japanese accent",
    };
    return accents;
}

// voice_design.py:61-75: Chinese-only, no English counterpart.
const std::vector<std::string> & dialect_category() {
    static const std::vector<std::string> dialects = {
        "河南话", "陕西话",   "四川话", "贵州话", "云南话", "桂林话",
        "济南话", "石家庄话", "甘肃话", "宁夏话", "青岛话", "东北话",
    };
    return dialects;
}

const std::unordered_map<std::string, std::string> & en_to_zh() {
    static const std::unordered_map<std::string, std::string> table = [] {
        std::unordered_map<std::string, std::string> map;
        for (const std::vector<AttributeItem> & category : dict_categories()) {
            for (const AttributeItem & item : category) {
                map.emplace(item.en, item.zh);
            }
        }
        return map;
    }();
    return table;
}

const std::unordered_map<std::string, std::string> & zh_to_en() {
    static const std::unordered_map<std::string, std::string> table = [] {
        std::unordered_map<std::string, std::string> map;
        for (const std::vector<AttributeItem> & category : dict_categories()) {
            for (const AttributeItem & item : category) {
                map.emplace(item.zh, item.en);
            }
        }
        return map;
    }();
    return table;
}

// _INSTRUCT_MUTUALLY_EXCLUSIVE (voice_design.py:80-87): one EN+ZH union set
// per dict category, then the flat accent set, then the flat dialect set --
// six entries, in this exact order.
const std::vector<std::unordered_set<std::string>> & mutually_exclusive_categories() {
    static const std::vector<std::unordered_set<std::string>> categories = [] {
        std::vector<std::unordered_set<std::string>> result;
        for (const std::vector<AttributeItem> & category : dict_categories()) {
            std::unordered_set<std::string> set;
            for (const AttributeItem & item : category) {
                set.insert(item.en);
                set.insert(item.zh);
            }
            result.push_back(std::move(set));
        }
        result.emplace_back(accent_category().begin(), accent_category().end());
        result.emplace_back(dialect_category().begin(), dialect_category().end());
        return result;
    }();
    return categories;
}

// _INSTRUCT_ALL_VALID (voice_design.py:89-94).
const std::unordered_set<std::string> & all_valid_items() {
    static const std::unordered_set<std::string> items = [] {
        std::unordered_set<std::string> set;
        for (const std::pair<const std::string, std::string> & entry : en_to_zh()) {
            set.insert(entry.first);
        }
        for (const std::pair<const std::string, std::string> & entry : zh_to_en()) {
            set.insert(entry.first);
        }
        set.insert(accent_category().begin(), accent_category().end());
        set.insert(dialect_category().begin(), dialect_category().end());
        return set;
    }();
    return items;
}

// ---------------------------------------------------------------------------
// String helpers, all ASCII-only by design: `std::tolower`/`std::isspace`
// are only ever applied to bytes cast through `unsigned char`, so a UTF-8
// continuation byte or a CJK lead byte (always >= 0x80) is never
// reinterpreted -- the same care voice-profile.cpp's own `ascii_alpha`/
// `ascii_digit` take for the same reason.
// ---------------------------------------------------------------------------

bool ascii_is_space(char value) {
    return std::isspace(static_cast<unsigned char>(value)) != 0;
}

std::string ascii_lowercase(const std::string & value) {
    std::string result = value;
    for (char & character : result) {
        character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    }
    return result;
}

std::string trim_ascii(const std::string & value) {
    size_t start = 0;
    while (start < value.size() && ascii_is_space(value[start])) {
        ++start;
    }
    size_t end = value.size();
    while (end > start && ascii_is_space(value[end - 1])) {
        --end;
    }
    return value.substr(start, end - start);
}

// Splits on both widths of comma -- half-width ',' (0x2C) and full-width
// '，' (U+FF0C, UTF-8 bytes EF BC 8C) -- trims ASCII whitespace from each
// piece, and drops empty pieces: `re.split(r"\s*[,，]\s*", instruct_str)`
// followed by `[x for x in raw_items if x]` (omnivoice.py:1544-1545). A
// regex is not needed: the two separators are fixed byte sequences and the
// surrounding whitespace this family's instructs use is always ASCII.
std::vector<std::string> split_instruct_items(const std::string & text) {
    std::vector<std::string> items;
    std::string              current;
    auto                     flush = [&]() {
        std::string trimmed = trim_ascii(current);
        if (!trimmed.empty()) {
            items.push_back(std::move(trimmed));
        }
        current.clear();
    };
    for (size_t index = 0; index < text.size();) {
        if (text[index] == ',') {
            flush();
            ++index;
            continue;
        }
        if (index + 2 < text.size() && static_cast<unsigned char>(text[index]) == 0xEF &&
            static_cast<unsigned char>(text[index + 1]) == 0xBC &&
            static_cast<unsigned char>(text[index + 2]) == 0x8C) {
            flush();
            index += 3;
            continue;
        }
        current.push_back(text[index]);
        ++index;
    }
    flush();
    return items;
}

// Duplicated from frontend-host.cpp's own private `is_cjk_ideograph`
// (0x4E00-0x9FFF, the exact range upstream's `_ZH_RE = re.compile(r"[一-鿿]")`
// tests) rather than shared: this project tolerates two independent copies
// of a helper this small before hoisting one out (voice-profile.cpp's own
// BCP-47 helpers carry the same rationale in their header comment), and the
// two call sites answer different questions -- combine_text's cleanup rule
// vs. this function's separator decision below.
bool is_cjk_ideograph(uint32_t codepoint) {
    return codepoint >= 0x4E00 && codepoint <= 0x9FFF;
}

bool contains_cjk(const std::string & text) {
    for (size_t offset = 0; offset < text.size();) {
        size_t         length    = 0;
        const uint32_t codepoint = synth::decode_utf8(text, offset, length);
        if (is_cjk_ideograph(codepoint)) {
            return true;
        }
        offset += length;
    }
    return false;
}

// Every dialect item ends with "话" (U+8BDD, UTF-8 E8 AF 9D); no non-dialect
// vocabulary item does (checked by inspection against the four dict
// categories' Chinese values: 男,女,儿童,少年,青年,中年,老年,极低音调,低音调,
// 中音调,高音调,极高音调,耳语 -- none end in 话), so this reproduces
// `n.endswith("话")` (omnivoice.py:1580) as a plain byte-suffix check.
bool ends_with_dialect_marker(const std::string & item) {
    static constexpr char kMarker[]    = "\xE8\xAF\x9D";  // "话"
    constexpr size_t      kMarkerBytes = sizeof(kMarker) - 1;
    return item.size() >= kMarkerBytes && item.compare(item.size() - kMarkerBytes, kMarkerBytes, kMarker) == 0;
}

// `" accent" in n` (omnivoice.py:1581) -- every accent item's own name ends
// in " accent" and no other vocabulary item contains that substring.
bool contains_accent_marker(const std::string & item) {
    return item.find(" accent") != std::string::npos;
}

}  // namespace

synth_status_t resolve_instruct(const std::string &                     description,
                                bool                                    use_zh,
                                std::shared_ptr<const DesignInstruct> & output,
                                const char *&                           out_diagnostic_code,
                                std::string &                           out_diagnostic_message) {
    output.reset();
    out_diagnostic_code = nullptr;
    out_diagnostic_message.clear();

    // omnivoice.py:1544-1556: split, validate membership per item. Fails on
    // the FIRST unknown item rather than collecting every one the way
    // upstream's own aggregate ValueError does -- this port's other
    // diagnostics (create_clone_prompt above) all report the first thing
    // wrong, and a caller fixes one item at a time regardless.
    const std::vector<std::string> raw_items = split_instruct_items(description);
    std::vector<std::string>       normalized;
    normalized.reserve(raw_items.size());
    for (const std::string & raw : raw_items) {
        const std::string lowered = ascii_lowercase(raw);
        if (all_valid_items().count(lowered) == 0) {
            out_diagnostic_code    = "voice_profile.instruct_unknown_item";
            out_diagnostic_message = "unsupported voice-design instruct item: '" + raw + "'";
            return SYNTH_ERR_INVALID_ARG;
        }
        normalized.push_back(lowered);
    }

    // omnivoice.py:1579-1592: a dialect item forces Chinese, an accent item
    // forces English (checked against the PRE-unification items, which may
    // legally mix scripts -- e.g. "male, 河南话" validates item by item
    // regardless of which script each one was written in); the two
    // together are rejected outright.
    bool has_dialect = false;
    bool has_accent  = false;
    for (const std::string & item : normalized) {
        has_dialect = has_dialect || ends_with_dialect_marker(item);
        has_accent  = has_accent || contains_accent_marker(item);
    }
    if (has_dialect && has_accent) {
        out_diagnostic_code = "voice_profile.instruct_dialect_accent_mix";
        out_diagnostic_message =
            "cannot mix a Chinese dialect and an English accent in one voice-design instruct; "
            "dialects are for Chinese speech, accents for English speech";
        return SYNTH_ERR_INVALID_ARG;
    }
    if (has_dialect) {
        use_zh = true;
    } else if (has_accent) {
        use_zh = false;
    }

    // omnivoice.py:1594-1598: unify to one language. An item with no
    // counterpart in the target language (an accent staying English, a
    // dialect staying Chinese) passes through unchanged via the map's
    // find-or-keep below, exactly like Python's `dict.get(n, n)`.
    const std::unordered_map<std::string, std::string> & translation = use_zh ? en_to_zh() : zh_to_en();
    for (std::string & item : normalized) {
        const auto found = translation.find(item);
        if (found != translation.end()) {
            item = found->second;
        }
    }

    // omnivoice.py:1600-1615: at most one item per mutually-exclusive
    // category, checked AFTER unification (exactly as upstream orders it).
    for (const std::unordered_set<std::string> & category : mutually_exclusive_categories()) {
        std::vector<std::string> hits;
        for (const std::string & item : normalized) {
            if (category.count(item) != 0) {
                hits.push_back(item);
            }
        }
        if (hits.size() > 1) {
            std::string joined;
            for (size_t index = 0; index < hits.size(); ++index) {
                if (index != 0) {
                    joined += "' vs '";
                }
                joined += hits[index];
            }
            out_diagnostic_code    = "voice_profile.instruct_category_conflict";
            out_diagnostic_message = "conflicting voice-design instruct items in the same category: '" + joined + "'";
            return SYNTH_ERR_INVALID_ARG;
        }
    }

    // omnivoice.py:1617-1621: the separator follows the FINAL unified
    // language, scanned rather than inferred from `use_zh` directly -- by
    // construction every item is either all-Chinese or all-English once
    // unification above has run (an accent item forces English throughout,
    // since it can only survive alongside `use_zh == false`; symmetric for
    // dialect and Chinese), so this scan and reading `use_zh` back would
    // agree on every non-empty `normalized` -- but the scan is what upstream
    // actually does, and this port matches it rather than its own proof.
    bool has_zh_output = false;
    for (const std::string & item : normalized) {
        has_zh_output = has_zh_output || contains_cjk(item);
    }
    const std::string separator = has_zh_output ? "\xEF\xBC\x8C" /* "，" */ : ", ";

    std::string joined;
    for (size_t index = 0; index < normalized.size(); ++index) {
        if (index != 0) {
            joined += separator;
        }
        joined += normalized[index];
    }

    auto design      = std::make_shared<DesignInstruct>();
    design->instruct = std::move(joined);
    output           = std::move(design);
    return SYNTH_OK;
}

}  // namespace synth::omnivoice
