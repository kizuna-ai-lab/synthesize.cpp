#include "arch/omnivoice/profile.h"

#include "arch/omnivoice/frontend-host.h"
#include "arch/omnivoice/omnivoice.h"
#include "codepoint-scan.h"
#include "ggml.h"
#include "gguf-metadata.h"
#include "gguf.h"
#include "sha256.h"
#include "text-frontend.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
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

// ---------------------------------------------------------------------------
// Task 16: the v1 Serialized Voice Profile envelope. See profile.h's own
// header comment on this section for the schema/kind split and the status
// mapping load_profile_from_memory below implements.
// ---------------------------------------------------------------------------

namespace {

constexpr const char * kEnvelopeArchitecture    = "synthprofile";
constexpr uint32_t     kEnvelopeFormatVersion   = 1;
constexpr const char * kEnvelopeModelFamily     = "omnivoice";
// The SAME string the package's own ProfileContract declares
// (weights.cpp's read_profile_contract, "omnivoice-clone-prompt") -- see
// profile.h's header comment on why this is one schema for two kinds rather
// than two schemas.
constexpr const char * kEnvelopeSchema          = "omnivoice-clone-prompt";
constexpr uint32_t     kEnvelopeSchemaVersion   = 1;
constexpr const char * kKindClonePrompt         = "clone-prompt";
constexpr const char * kKindDesignInstruct      = "design-instruct";
constexpr const char * kKeyCompatibilityId      = "synthesize.voice_profile.compatibility_id";
constexpr const char * kKeyContentSha256        = "synthesize.voice_profile.content_sha256";
constexpr const char * kTensorReferenceTokens   = "profile.reference_tokens";
constexpr uint32_t     kReferenceTokenCodebooks = 8;

struct GgufContextDeleter {
    void operator()(gguf_context * context) const {
        if (context != nullptr) {
            gguf_free(context);
        }
    }
};

using OwnedGgufContext = std::unique_ptr<gguf_context, GgufContextDeleter>;

// ---------------------------------------------------------------------------
// Little-endian byte-buffer writers -- GGUF's own encoding (gguf.h's header
// comment), and this project's blanket little-endian-host assumption
// (gguf-metadata.cpp's own raw memcpy reads make the same assumption
// elsewhere in this codebase; ggml's own writer/reader never byte-swaps
// either).
// ---------------------------------------------------------------------------

void put_bytes(std::vector<uint8_t> & out, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    out.insert(out.end(), bytes, bytes + size);
}

template <typename T> void put(std::vector<uint8_t> & out, T value) {
    put_bytes(out, &value, sizeof(value));
}

void put_gguf_string(std::vector<uint8_t> & out, const std::string & value) {
    put<uint64_t>(out, uint64_t(value.size()));
    put_bytes(out, value.data(), value.size());
}

void pad_to_alignment(std::vector<uint8_t> & out, size_t alignment) {
    while (out.size() % alignment != 0) {
        out.push_back(0);
    }
}

// Encodes every KV pair `ctx` holds, in insertion order, exactly mirroring
// gguf.cpp's own gguf_write_out for the metadata portion of a GGUF file --
// see write_envelope below for why the header and tensor sections are
// hand-written instead of going through gguf's own (file-only) writer.
// Handles only the value shapes this schema ever produces (uint32, uint64,
// float32, and string scalars; one 32-element uint8 array shape, used for
// both 32-byte ID fields) -- returning false for anything else, since
// nothing here ever encodes untrusted input and reaching that arm is this
// writer's own defect.
//
// Records the byte offset of `kKeyContentSha256`'s 32-byte VALUE into
// `out_sha_offset` as it is written, so write_envelope can patch the real
// digest in afterward without a second pass over the buffer.
bool encode_metadata_kv(const gguf_context * ctx, std::vector<uint8_t> & out, size_t & out_sha_offset) {
    bool          found_sha = false;
    const int64_t n_kv      = gguf_get_n_kv(ctx);
    for (int64_t index = 0; index < n_kv; ++index) {
        const std::string key  = gguf_get_key(ctx, index);
        const gguf_type   type = gguf_get_kv_type(ctx, index);
        put_gguf_string(out, key);
        if (type == GGUF_TYPE_ARRAY) {
            const gguf_type element_type = gguf_get_arr_type(ctx, index);
            const size_t    count        = gguf_get_arr_n(ctx, index);
            put<int32_t>(out, int32_t(GGUF_TYPE_ARRAY));
            put<int32_t>(out, int32_t(element_type));
            put<uint64_t>(out, uint64_t(count));
            if (element_type != GGUF_TYPE_UINT8) {
                return false;
            }
            const auto * data = static_cast<const uint8_t *>(gguf_get_arr_data(ctx, index));
            if (data == nullptr && count > 0) {
                return false;
            }
            if (key == kKeyContentSha256) {
                out_sha_offset = out.size();
                found_sha      = true;
            }
            put_bytes(out, data, count);
            continue;
        }
        put<int32_t>(out, int32_t(type));
        switch (type) {
            case GGUF_TYPE_UINT32:
                put<uint32_t>(out, gguf_get_val_u32(ctx, index));
                break;
            case GGUF_TYPE_UINT64:
                put<uint64_t>(out, gguf_get_val_u64(ctx, index));
                break;
            case GGUF_TYPE_FLOAT32:
                put<float>(out, gguf_get_val_f32(ctx, index));
                break;
            case GGUF_TYPE_STRING:
                put_gguf_string(out, gguf_get_val_str(ctx, index));
                break;
            default:
                return false;
        }
    }
    return found_sha;
}

// Appends the common v1 envelope header metadata every kind shares.
// `compatibility_id` is copied verbatim. `content_sha256` is seeded at
// zero -- docs/c-interface.md's exact rule: write_envelope hashes the
// assembled buffer with this placeholder still in place, then patches the
// real digest into the same 32 bytes afterward.
void set_common_metadata(gguf_context * ctx, const char * kind, const uint8_t (&compatibility_id)[32]) {
    gguf_set_val_str(ctx, "general.architecture", kEnvelopeArchitecture);
    gguf_set_val_u32(ctx, "synthesize.voice_profile.format_version", kEnvelopeFormatVersion);
    gguf_set_val_str(ctx, "synthesize.voice_profile.model_family", kEnvelopeModelFamily);
    gguf_set_val_str(ctx, "synthesize.voice_profile.schema", kEnvelopeSchema);
    gguf_set_val_u32(ctx, "synthesize.voice_profile.schema_version", kEnvelopeSchemaVersion);
    gguf_set_arr_data(ctx, kKeyCompatibilityId, GGUF_TYPE_UINT8, compatibility_id, 32);
    static constexpr uint8_t kZeroDigest[32] = {};
    gguf_set_arr_data(ctx, kKeyContentSha256, GGUF_TYPE_UINT8, kZeroDigest, 32);
    gguf_set_val_str(ctx, "synthesize.voice_profile.kind", kind);
}

// Hand-writes the header, tensor-info section (0 or 1 entries), alignment
// padding, and tensor data (if any) that gguf.h's own writer API cannot
// produce into a memory buffer: the function gguf_get_meta_size/
// gguf_get_meta_data themselves call to do this internally
// (gguf_write_to_buf) is declared only in ggml-impl.h, an internal header
// this project's ggml submodule boundary puts out of reach (CLAUDE.md:
// "never edit ggml/ in place ... it is a checkout of another repository";
// depending on its private headers carries the same fragility -- an
// upstream refactor could move or remove it without notice). The metadata
// KV section above genuinely goes through gguf's own setter/getter API
// (gguf_set_val_*/gguf_get_*); only the header/tensor-info/tensor-data
// framing below is this project's own, and it is deliberately narrow: this
// schema never has more than one tensor, with a shape this function already
// knows locally (it is never read back from `ctx`, which has no
// per-dimension shape getter to read it back FROM -- gguf_get_tensor_size
// only ever returns the flattened byte count).
//
// `ctx` must already hold `content_sha256` seeded at 32 zero bytes
// (set_common_metadata's job).
synth_status_t write_envelope(const gguf_context *         ctx,
                              const std::vector<int32_t> * tensor_tokens,
                              uint64_t                     tensor_frames,
                              std::vector<uint8_t> &       out_bytes) {
    std::vector<uint8_t> bytes;
    const int64_t        n_tensors = (tensor_tokens != nullptr) ? 1 : 0;

    put_bytes(bytes, GGUF_MAGIC, 4);
    put<uint32_t>(bytes, uint32_t(GGUF_VERSION));
    put<int64_t>(bytes, n_tensors);
    put<int64_t>(bytes, gguf_get_n_kv(ctx));

    size_t sha_offset = 0;
    if (!encode_metadata_kv(ctx, bytes, sha_offset)) {
        return SYNTH_ERR_INTERNAL;
    }

    if (tensor_tokens != nullptr) {
        put_gguf_string(bytes, kTensorReferenceTokens);
        put<uint32_t>(bytes, uint32_t(2));                       // n_dims
        put<int64_t>(bytes, int64_t(tensor_frames));             // ne[0] = T_ref
        put<int64_t>(bytes, int64_t(kReferenceTokenCodebooks));  // ne[1] = 8
        put<int32_t>(bytes, int32_t(GGML_TYPE_I32));
        put<uint64_t>(bytes, uint64_t(0));                       // the only tensor starts at offset 0
    }
    pad_to_alignment(bytes, GGUF_DEFAULT_ALIGNMENT);

    if (tensor_tokens != nullptr) {
        put_bytes(bytes, tensor_tokens->data(), tensor_tokens->size() * sizeof(int32_t));
        pad_to_alignment(bytes, GGUF_DEFAULT_ALIGNMENT);
    }

    // The digest step docs/c-interface.md prescribes: hash the buffer with
    // content_sha256 already zero (set_common_metadata seeded it that way,
    // untouched above), then overwrite just those 32 bytes with the result.
    uint8_t digest[32];
    synth::sha256(bytes.data(), bytes.size(), digest);
    std::memcpy(bytes.data() + sha_offset, digest, sizeof(digest));

    out_bytes = std::move(bytes);
    return SYNTH_OK;
}

// Reads a required, exactly-32-element uint8 array metadata value. `false`
// covers "missing", "wrong GGUF type", and "wrong element count" alike --
// the caller maps all three to the same malformed outcome.
bool read_u8_32_array(const gguf_context * ctx, const char * key, uint8_t (&out)[32]) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx, id) != GGUF_TYPE_UINT8 ||
        gguf_get_arr_n(ctx, id) != 32) {
        return false;
    }
    const void * data = gguf_get_arr_data(ctx, id);
    if (data == nullptr) {
        return false;
    }
    std::memcpy(out, data, 32);
    return true;
}

// Locates the byte OFFSET of `key`'s 32-element uint8 array VALUE within a
// raw, untrusted GGUF byte buffer, by searching for that KV entry's own
// on-disk encoding PREFIX (length-prefixed key text, then the array marker,
// element-type tag, and element count -- see gguf.cpp's gguf_kv::write for
// the exact byte layout this mirrors). This is independent of the parsed
// gguf_context's own bookkeeping: gguf_get_arr_data only ever hands back a
// pointer into the PARSED context's own internal copy, never an offset into
// the caller's original bytes, and the digest check below needs the latter
// to hash "the complete input with that one value treated as zero"
// (docs/c-interface.md) rather than a reconstruction built from parsed
// fields.
bool find_u8_32_value_offset(const uint8_t * data, size_t data_size, const std::string & key, size_t & out_offset) {
    std::vector<uint8_t> needle;
    put<uint64_t>(needle, uint64_t(key.size()));
    put_bytes(needle, key.data(), key.size());
    put<int32_t>(needle, int32_t(GGUF_TYPE_ARRAY));
    put<int32_t>(needle, int32_t(GGUF_TYPE_UINT8));
    put<uint64_t>(needle, uint64_t(32));

    const uint8_t * begin = data;
    const uint8_t * end   = data + data_size;
    const uint8_t * found = std::search(begin, end, needle.begin(), needle.end());
    if (found == end) {
        return false;
    }
    const size_t value_offset = size_t(found - begin) + needle.size();
    if (value_offset + 32 > data_size) {
        return false;
    }
    out_offset = value_offset;
    return true;
}

// ---------------------------------------------------------------------------
// Untrusted-buffer pre-scan: a defect in ggml's own parser that this loader
// cannot fix downstream of calling it, only guard against beforehand.
//
// Fix round 1 closed ONE instance: ggml/src/gguf.cpp's gguf_init_from_reader
// reads "general.alignment" eagerly during construction
// (gguf_get_val_u32(ctx, alignment_idx) at gguf.cpp:610), and that getter's
// GGML_ASSERT(type_to_gguf_type<T>::value == type) (gguf.cpp:194) aborts the
// whole process on a type mismatch rather than returning an error. Round 1's
// guard special-cased that one key. Fix round 2 exists because fuzzing found
// a SECOND, different instance in minutes: a KV entry with key_length == 0
// passed round 1's guard untouched (0 is not > any length ceiling, and an
// empty key never equals "general.alignment"), reached
// gguf_kv's constructors, and hit GGML_ASSERT(!key.empty()) -- all four
// overloads, gguf.cpp:143/151/161/167 -- for ANY value type, reserved or
// not. Same SIGABRT, same public path, different key.
//
// THE LESSON: enumerating ggml's internal asserts is a blacklist, and a
// blacklist of another library's invariants can never be proven complete --
// closing one instance told us nothing about the next one, and fuzzing
// found the next one in the same session. This guard is POSITIVE validation
// instead: a v1 Serialized Profile is narrow and fully known to US -- we
// write it (write_envelope, set_common_metadata, serialize_clone_prompt,
// serialize_design_instruct, all in this file) -- so the pre-scan accepts
// ONLY what our own writer ever emits and refuses everything else, before
// gguf_init_from_buffer ever runs. This subsumes the empty-key hole (empty
// is not a member of the known key set below), the alignment hole round 1
// closed (this writer never emits "general.alignment" at all, so its mere
// PRESENCE is now an unknown-key rejection, no special case needed), and
// every OTHER ggml invariant about a key or value shape this project has
// not yet found by name: an unknown key, an unexpected type for a known
// key, a duplicate key, or a tensor section that isn't exactly "zero
// tensors" or "one profile.reference_tokens tensor of our exact shape" is
// refused on the same "not our format" basis, with no dependency on which
// ggml assert it might otherwise have hit.
//
// The bounds-safe byte-walking machinery below (prescan_has_remaining/
// prescan_skip/prescan_read_bytes/prescan_read/prescan_fixed_type_size/
// prescan_skip_value) is unchanged from round 1 -- an 8000+ iteration fuzz
// campaign against it (see task-16-report.md's fix-round-2 section) found
// zero sanitizer reports and zero hangs, so only the ACCEPTANCE PREDICATE
// changes here: what used to ask "is this obviously hostile?" now asks "is
// this exactly our format?".

// The exact, closed set of metadata keys this family's writer ever emits --
// transcribed from set_common_metadata (the 8 common keys, including
// "general.alignment"'s absence: this writer never sets that key, so it is
// simply not a member here, and its mere presence in an inbound buffer is
// therefore an unknown-key rejection with no special case), plus
// serialize_clone_prompt's 3 ClonePrompt-only keys and
// serialize_design_instruct's 1 DesignInstruct-only key. `is_array` and
// `count` are only meaningful together; a scalar entry's `count` is
// ignored.
struct PrescanKeySpec {
    const char * key;
    gguf_type    type;
    bool         is_array;
    uint64_t     count;
};

constexpr PrescanKeySpec kPrescanKnownKeys[] = {
    // set_common_metadata (8 keys, every kind).
    { "general.architecture",                     GGUF_TYPE_STRING,  false, 0  },
    { "synthesize.voice_profile.format_version",  GGUF_TYPE_UINT32,  false, 0  },
    { "synthesize.voice_profile.model_family",    GGUF_TYPE_STRING,  false, 0  },
    { "synthesize.voice_profile.schema",          GGUF_TYPE_STRING,  false, 0  },
    { "synthesize.voice_profile.schema_version",  GGUF_TYPE_UINT32,  false, 0  },
    { kKeyCompatibilityId,                        GGUF_TYPE_UINT8,   true,  32 },
    { kKeyContentSha256,                          GGUF_TYPE_UINT8,   true,  32 },
    { "synthesize.voice_profile.kind",            GGUF_TYPE_STRING,  false, 0  },
    // serialize_clone_prompt (3 more keys, "clone-prompt" only).
    { "synthesize.voice_profile.transcript_text", GGUF_TYPE_STRING,  false, 0  },
    { "synthesize.voice_profile.ref_rms",         GGUF_TYPE_FLOAT32, false, 0  },
    { "synthesize.voice_profile.language_tag",    GGUF_TYPE_STRING,  false, 0  },
    // serialize_design_instruct (1 more key, "design-instruct" only).
    { "synthesize.voice_profile.instruct",        GGUF_TYPE_STRING,  false, 0  },
};
constexpr size_t kPrescanKnownKeyCount = sizeof(kPrescanKnownKeys) / sizeof(kPrescanKnownKeys[0]);

// n_kv is exactly one of these two values: 8 common + 1 ("instruct") for
// DesignInstruct, 8 common + 3 (transcript_text/ref_rms/language_tag) for
// ClonePrompt -- not a generous ceiling, an exact enumeration, since this
// writer never produces anything else. Which SPECIFIC keys are required for
// a given `kind` is still load_profile_from_memory's own job afterward
// (GgufMetadata's per-field reads already fail closed on a missing field);
// this pre-scan only bounds the total count and rejects any key outside the
// union above.
constexpr int64_t kPrescanKvCountDesign = 9;
constexpr int64_t kPrescanKvCountClone  = 11;

constexpr uint64_t kPrescanMaxKeyLength    = 256;
constexpr uint64_t kPrescanMaxStringLength = 1u << 20;  // 1 MiB: far past any reasonable transcript/instruct text

bool prescan_has_remaining(size_t offset, size_t size, size_t need) {
    return offset <= size && need <= size - offset;
}

bool prescan_skip(size_t & offset, size_t size, size_t amount) {
    if (!prescan_has_remaining(offset, size, amount)) {
        return false;
    }
    offset += amount;
    return true;
}

bool prescan_read_bytes(const uint8_t * data, size_t size, size_t & offset, void * out, size_t amount) {
    if (!prescan_has_remaining(offset, size, amount)) {
        return false;
    }
    std::memcpy(out, data + offset, amount);
    offset += amount;
    return true;
}

template <typename T> bool prescan_read(const uint8_t * data, size_t size, size_t & offset, T & out) {
    return prescan_read_bytes(data, size, offset, &out, sizeof(out));
}

// The fixed byte width of every scalar GGUF type this walk can skip without
// decoding it (everything except STRING, which is itself length-prefixed
// and handled separately in prescan_skip_value).
bool prescan_fixed_type_size(gguf_type type, size_t & out_size) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            out_size = 1;
            return true;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            out_size = 2;
            return true;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            out_size = 4;
            return true;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            out_size = 8;
            return true;
        default:
            return false;
    }
}

// Skips over one value's bytes (a single scalar, or `count` elements of an
// array) without decoding it. `count` is 1 for a scalar (the caller's own
// convention, matching gguf_init_from_reader's own `uint64_t n = 1` default).
bool prescan_skip_value(const uint8_t * data, size_t size, size_t & offset, gguf_type type, uint64_t count) {
    if (type == GGUF_TYPE_STRING) {
        for (uint64_t index = 0; index < count; ++index) {
            uint64_t length = 0;
            if (!prescan_read(data, size, offset, length) || length > kPrescanMaxStringLength) {
                return false;
            }
            if (!prescan_skip(offset, size, size_t(length))) {
                return false;
            }
        }
        return true;
    }
    size_t element_size = 0;
    if (!prescan_fixed_type_size(type, element_size)) {
        return false;  // GGUF_TYPE_ARRAY-of-ARRAY or any other unrecognized type tag
    }
    if (count > SIZE_MAX / element_size) {
        return false;  // overflow guard on count * element_size
    }
    return prescan_skip(offset, size, size_t(count) * element_size);
}

// The full walk: positive validation against the exact format this family's
// own writer produces (this section's own header comment explains why, and
// what it replaced). Returns false for anything outside that format --
// whether or not gguf_init_from_buffer would also refuse it, and whether or
// not any KNOWN ggml assert applies -- and the caller maps `false` to
// SYNTH_ERR_INVALID_ARG without ever calling gguf_init_from_buffer on these
// bytes.
bool prescan_buffer(const uint8_t * data, size_t size) {
    size_t offset = 0;

    char magic[4];
    if (!prescan_read_bytes(data, size, offset, magic, sizeof(magic)) ||
        std::memcmp(magic, GGUF_MAGIC, sizeof(magic)) != 0) {
        return false;
    }

    uint32_t version = 0;
    if (!prescan_read(data, size, offset, version) || version != GGUF_VERSION) {
        // This project only ever WRITES GGUF_VERSION (3); a different
        // version is either an old writer this loader never claimed to
        // support or corrupt data -- both malformed for a v1 Serialized
        // Profile's own contract.
        return false;
    }

    int64_t n_tensors = 0;
    int64_t n_kv      = 0;
    // Exactly 0 (DesignInstruct) or 1 (ClonePrompt) tensor -- this writer
    // never produces any other count.
    if (!prescan_read(data, size, offset, n_tensors) || (n_tensors != 0 && n_tensors != 1)) {
        return false;
    }
    // Exactly kPrescanKvCountDesign or kPrescanKvCountClone metadata
    // entries -- see those constants' own comment.
    if (!prescan_read(data, size, offset, n_kv) || (n_kv != kPrescanKvCountDesign && n_kv != kPrescanKvCountClone)) {
        return false;
    }

    bool seen[kPrescanKnownKeyCount] = {};

    for (int64_t index = 0; index < n_kv; ++index) {
        uint64_t key_length = 0;
        if (!prescan_read(data, size, offset, key_length) || key_length > kPrescanMaxKeyLength) {
            return false;
        }
        if (!prescan_has_remaining(offset, size, size_t(key_length))) {
            return false;
        }
        const std::string key(reinterpret_cast<const char *>(data + offset), size_t(key_length));
        offset += size_t(key_length);

        int32_t type_raw = 0;
        if (!prescan_read(data, size, offset, type_raw)) {
            return false;
        }
        gguf_type type     = gguf_type(type_raw);
        bool      is_array = false;
        uint64_t  count    = 1;
        if (type == GGUF_TYPE_ARRAY) {
            is_array                 = true;
            int32_t element_type_raw = 0;
            if (!prescan_read(data, size, offset, element_type_raw)) {
                return false;
            }
            type = gguf_type(element_type_raw);
            if (!prescan_read(data, size, offset, count)) {
                return false;
            }
        }

        // Positive validation (this section's own header comment): `key`
        // must be one of the exact keys this writer ever emits, it must
        // not repeat, and its declared shape must match that key's own
        // exact type/array-ness/count -- not merely "a type ggml would not
        // abort on". An empty key (fix round 2's own crash) is rejected
        // here because "" is not a member of kPrescanKnownKeys, the same
        // way "general.alignment" is rejected because this writer never
        // emits it at all -- no per-key special case for either.
        size_t spec_index = kPrescanKnownKeyCount;
        for (size_t candidate = 0; candidate < kPrescanKnownKeyCount; ++candidate) {
            if (key == kPrescanKnownKeys[candidate].key) {
                spec_index = candidate;
                break;
            }
        }
        if (spec_index == kPrescanKnownKeyCount) {
            return false;  // unknown key
        }
        if (seen[spec_index]) {
            return false;  // duplicate key
        }
        seen[spec_index] = true;

        const PrescanKeySpec & spec = kPrescanKnownKeys[spec_index];
        if (type != spec.type || is_array != spec.is_array || (is_array && count != spec.count)) {
            return false;
        }

        if (!prescan_skip_value(data, size, offset, type, count)) {
            return false;
        }
    }

    // The tensor section: 0 entries, or exactly one entry named
    // "profile.reference_tokens" with this writer's own exact shape
    // (write_envelope: n_dims=2, ne[1]=kReferenceTokenCodebooks, type
    // GGML_TYPE_I32, offset 0 since it is the only tensor). ne[0] (T_ref)
    // is the one field that legitimately varies with the reference clip's
    // own length, so it is only checked for positivity here; the
    // max_total_frames budget check still happens later, against the real
    // parsed tensor size, before any allocation sized from it.
    if (n_tensors == 1) {
        uint64_t name_length = 0;
        if (!prescan_read(data, size, offset, name_length) || name_length > kPrescanMaxKeyLength) {
            return false;
        }
        if (!prescan_has_remaining(offset, size, size_t(name_length))) {
            return false;
        }
        const std::string tensor_name(reinterpret_cast<const char *>(data + offset), size_t(name_length));
        offset += size_t(name_length);
        if (tensor_name != kTensorReferenceTokens) {
            return false;
        }

        uint32_t n_dims = 0;
        if (!prescan_read(data, size, offset, n_dims) || n_dims != 2) {
            return false;
        }

        int64_t ne0 = 0;
        int64_t ne1 = 0;
        if (!prescan_read(data, size, offset, ne0) || ne0 <= 0) {
            return false;
        }
        if (!prescan_read(data, size, offset, ne1) || ne1 != int64_t(kReferenceTokenCodebooks)) {
            return false;
        }

        int32_t tensor_type_raw = 0;
        if (!prescan_read(data, size, offset, tensor_type_raw) || tensor_type_raw != int32_t(GGML_TYPE_I32)) {
            return false;
        }

        uint64_t tensor_offset = 0;
        if (!prescan_read(data, size, offset, tensor_offset) || tensor_offset != 0) {
            return false;
        }
    }

    return true;
}

}  // namespace

synth_status_t serialize_clone_prompt(const ClonePrompt & prompt,
                                      const uint8_t (&compatibility_id)[32],
                                      std::vector<uint8_t> & out_bytes) {
    if (prompt.reference_tokens.empty() || prompt.reference_tokens.size() % kReferenceTokenCodebooks != 0) {
        return SYNTH_ERR_INVALID_ARG;
    }
    OwnedGgufContext ctx(gguf_init_empty());
    if (ctx == nullptr) {
        return SYNTH_ERR_OOM;
    }
    set_common_metadata(ctx.get(), kKindClonePrompt, compatibility_id);
    gguf_set_val_str(ctx.get(), "synthesize.voice_profile.transcript_text", prompt.transcript_text.c_str());
    gguf_set_val_f32(ctx.get(), "synthesize.voice_profile.ref_rms", prompt.ref_rms);
    gguf_set_val_str(ctx.get(), "synthesize.voice_profile.language_tag", prompt.language_tag.c_str());

    const uint64_t frames = uint64_t(prompt.reference_tokens.size() / kReferenceTokenCodebooks);
    return write_envelope(ctx.get(), &prompt.reference_tokens, frames, out_bytes);
}

synth_status_t serialize_design_instruct(const DesignInstruct & instruct,
                                         const uint8_t (&compatibility_id)[32],
                                         std::vector<uint8_t> & out_bytes) {
    OwnedGgufContext ctx(gguf_init_empty());
    if (ctx == nullptr) {
        return SYNTH_ERR_OOM;
    }
    set_common_metadata(ctx.get(), kKindDesignInstruct, compatibility_id);
    gguf_set_val_str(ctx.get(), "synthesize.voice_profile.instruct", instruct.instruct.c_str());
    return write_envelope(ctx.get(), nullptr, 0, out_bytes);
}

synth_status_t load_profile_from_memory(Model &         model,
                                        const uint8_t * data,
                                        size_t          data_size,
                                        const uint8_t (&compatibility_id)[32],
                                        uint64_t                      max_total_frames,
                                        synth::ProfileFamilyTag &     out_family_tag,
                                        std::shared_ptr<const void> & out_payload,
                                        const char *&                 out_diagnostic_code,
                                        const char *&                 out_diagnostic_message) {
    out_payload.reset();
    out_family_tag         = synth::ProfileFamilyTag::None;
    out_diagnostic_code    = nullptr;
    out_diagnostic_message = nullptr;

    if (data == nullptr || data_size == 0) {
        return SYNTH_ERR_INVALID_ARG;
    }

    // Independent raw-byte hardening BEFORE gguf_init_from_buffer ever
    // touches these bytes -- see prescan_buffer's own header comment for
    // why: ggml's parser aborts the process on a reserved key (at minimum
    // "general.alignment") declared with a type its own eager read does not
    // accept, and that call happens inside gguf_init_from_buffer itself,
    // before any line below this one runs.
    if (!prescan_buffer(data, data_size)) {
        return SYNTH_ERR_INVALID_ARG;
    }

    // Structural parse only: `ctx == nullptr` means gguf_init_from_buffer
    // never reads (or allocates for) the tensor DATA blob, no matter how
    // large the file's own tensor-info section claims it is -- only the
    // bounded tensor-info section (name/shape/type/offset) is parsed here.
    // The declared token count is checked against `max_total_frames` below,
    // using only that bounded info, before this function ever allocates
    // anything sized from it.
    gguf_init_params init_params{};
    init_params.no_alloc = true;
    init_params.ctx      = nullptr;
    OwnedGgufContext ctx(gguf_init_from_buffer(data, data_size, init_params));
    if (ctx == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    gguf_context * g = ctx.get();

    if (gguf_get_alignment(g) != GGUF_DEFAULT_ALIGNMENT) {
        return SYNTH_ERR_INVALID_ARG;
    }

    GgufMetadata meta(g, "omnivoice");
    if (!meta.require_string("general.architecture", kEnvelopeArchitecture)) {
        return SYNTH_ERR_INVALID_ARG;
    }
    uint32_t format_version = 0;
    if (!meta.u32("synthesize.voice_profile.format_version", format_version) ||
        format_version != kEnvelopeFormatVersion) {
        return SYNTH_ERR_INVALID_ARG;
    }
    std::string model_family;
    if (!meta.string("synthesize.voice_profile.model_family", model_family)) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (model_family != kEnvelopeModelFamily) {
        // A structurally sound envelope for a DIFFERENT family, not corrupt
        // data -- this loader understands the shape, just not for itself.
        return SYNTH_ERR_UNSUPPORTED_VOICE;
    }
    std::string schema;
    uint32_t    schema_version = 0;
    if (!meta.string("synthesize.voice_profile.schema", schema) ||
        !meta.u32("synthesize.voice_profile.schema_version", schema_version)) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (schema != kEnvelopeSchema || schema_version != kEnvelopeSchemaVersion) {
        return SYNTH_ERR_UNSUPPORTED_VOICE;
    }
    std::string kind;
    if (!meta.string("synthesize.voice_profile.kind", kind)) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (kind != kKindClonePrompt && kind != kKindDesignInstruct) {
        // Unlike model_family/schema/schema_version above, `kind` is this
        // ONE schema's own internal tag, not a different schema, version, or
        // family: a value outside its two-member enum means the payload
        // structure the rest of the bytes describe cannot be interpreted at
        // all, which this project treats as malformed rather than merely
        // unsupported (profile.h's own header comment on this function).
        return SYNTH_ERR_INVALID_ARG;
    }
    uint8_t file_compatibility_id[32];
    if (!read_u8_32_array(g, kKeyCompatibilityId, file_compatibility_id)) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (std::memcmp(file_compatibility_id, compatibility_id, sizeof(file_compatibility_id)) != 0) {
        return SYNTH_ERR_UNSUPPORTED_VOICE;
    }
    uint8_t stored_digest[32];
    if (!read_u8_32_array(g, kKeyContentSha256, stored_digest)) {
        return SYNTH_ERR_INVALID_ARG;
    }

    const bool is_clone = (kind == kKindClonePrompt);
    if (gguf_get_n_tensors(g) != (is_clone ? 1 : 0)) {
        return SYNTH_ERR_INVALID_ARG;
    }

    // Size arithmetic BEFORE allocation: the declared token count is bounded
    // by max_total_frames using only cheap tensor-info getters (no tensor
    // payload byte has been read yet) before anything sized from it is ever
    // allocated below -- neither the digest scratch copy (sized from
    // `data_size`, which the caller already materialized, not from anything
    // this file's own bytes claim) nor, later, the token vector itself.
    int64_t  tensor_id     = -1;
    uint64_t element_count = 0;
    uint64_t frames        = 0;
    if (is_clone) {
        tensor_id = gguf_find_tensor(g, kTensorReferenceTokens);
        if (tensor_id < 0 || gguf_get_tensor_type(g, tensor_id) != GGML_TYPE_I32) {
            return SYNTH_ERR_INVALID_ARG;
        }
        const size_t tensor_bytes = gguf_get_tensor_size(g, tensor_id);
        if (tensor_bytes == 0 || tensor_bytes % (sizeof(int32_t) * kReferenceTokenCodebooks) != 0) {
            return SYNTH_ERR_INVALID_ARG;
        }
        element_count = tensor_bytes / sizeof(int32_t);
        frames        = element_count / kReferenceTokenCodebooks;
        if (frames == 0 || frames > max_total_frames) {
            return SYNTH_ERR_INVALID_ARG;
        }
    }

    // Digest verification (docs/c-interface.md's exact rule): hash the
    // complete input with content_sha256's own 32 bytes treated as zero, and
    // compare against the value stored there.
    size_t sha_offset = 0;
    if (!find_u8_32_value_offset(data, data_size, kKeyContentSha256, sha_offset)) {
        // The parse above already confirmed this key exists as a 32-element
        // uint8 array; failing to relocate it by its own on-disk encoding is
        // this loader's own defect, not a caller mistake.
        return SYNTH_ERR_INTERNAL;
    }
    std::vector<uint8_t> scratch(data, data + data_size);
    std::memset(scratch.data() + sha_offset, 0, 32);
    uint8_t computed_digest[32];
    synth::sha256(scratch.data(), scratch.size(), computed_digest);
    if (std::memcmp(computed_digest, stored_digest, sizeof(computed_digest)) != 0) {
        return SYNTH_ERR_INVALID_ARG;
    }

    if (!is_clone) {
        std::string instruct;
        if (!meta.string("synthesize.voice_profile.instruct", instruct)) {
            return SYNTH_ERR_INVALID_ARG;
        }
        auto design      = std::make_shared<DesignInstruct>();
        design->instruct = std::move(instruct);
        out_payload      = std::move(design);
        out_family_tag   = synth::ProfileFamilyTag::OmnivoiceDesign;
        return SYNTH_OK;
    }

    std::string transcript_text;
    float       ref_rms = 0.0f;
    std::string language_tag;
    if (!meta.string("synthesize.voice_profile.transcript_text", transcript_text) ||
        !meta.f32("synthesize.voice_profile.ref_rms", ref_rms) ||
        !meta.string("synthesize.voice_profile.language_tag", language_tag)) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (!(ref_rms > 0.0f)) {
        // ClonePrompt::ref_rms's own contract (profile.h): "always strictly
        // positive for a ClonePrompt that exists at all" -- a serialized
        // envelope claiming otherwise is malformed, the same rejection
        // create_clone_prompt itself applies before a ClonePrompt is ever
        // built.
        return SYNTH_ERR_INVALID_ARG;
    }

    // Truncation guard: the DECLARED tensor byte range must actually fit
    // inside the SUPPLIED buffer. gguf_init_from_buffer was called with
    // `ctx == nullptr` specifically so it never performed this read (or the
    // allocation it implies) itself -- see this function's own comment
    // above the size-arithmetic block.
    const size_t data_offset   = gguf_get_data_offset(g);
    const size_t tensor_offset = gguf_get_tensor_offset(g, tensor_id);
    const size_t tensor_bytes  = size_t(element_count) * sizeof(int32_t);
    if (data_offset > data_size) {
        return SYNTH_ERR_INVALID_ARG;
    }
    size_t remaining = data_size - data_offset;
    if (tensor_offset > remaining) {
        return SYNTH_ERR_INVALID_ARG;
    }
    remaining -= tensor_offset;
    if (tensor_bytes > remaining) {
        return SYNTH_ERR_INVALID_ARG;
    }

    std::vector<int32_t> tokens(element_count);
    std::memcpy(tokens.data(), data + data_offset + tensor_offset, tensor_bytes);

    const uint32_t vocab_size = model.audio_vocab_size();
    const uint32_t mask_id    = model.audio_mask_id();
    for (int32_t token : tokens) {
        if (token < 0 || uint32_t(token) >= vocab_size || uint32_t(token) == mask_id) {
            return SYNTH_ERR_INVALID_ARG;
        }
    }

    auto prompt              = std::make_shared<ClonePrompt>();
    prompt->reference_tokens = std::move(tokens);
    prompt->transcript_text  = transcript_text;
    prompt->ref_rms          = ref_rms;
    prompt->language_tag     = language_tag;

    // transcript_ids is NOT serialized (profile.h's own ClonePrompt
    // comment): re-tokenize the already-canonical transcript_text through
    // this Loaded Model's own frontend, the same call create_clone_prompt
    // itself makes on the punctuated transcript -- determinism guaranteed by
    // the frozen BPE vocabulary, so this reproduces the exact ids the
    // original profile's own creation call produced.
    const std::shared_ptr<const TextFrontend> frontend = model.text_frontend();
    if (frontend == nullptr || frontend->prepare(SYNTH_INPUT_TEXT_UTF8, transcript_text.data(), transcript_text.size(),
                                                 0, prompt->transcript_ids) != SYNTH_OK) {
        out_diagnostic_code    = "voice_profile.transcript_unencodable";
        out_diagnostic_message = "the reference transcript contains a byte this package's frontend has no token for";
        return SYNTH_ERR_INVALID_ARG;
    }

    out_payload    = std::move(prompt);
    out_family_tag = synth::ProfileFamilyTag::OmnivoiceClone;
    return SYNTH_OK;
}

}  // namespace synth::omnivoice
