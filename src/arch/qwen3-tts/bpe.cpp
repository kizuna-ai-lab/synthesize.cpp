// The Qwen byte-level byte-pair tokenizer.
//
// Three stages, and each of them changes the token ids if it is wrong -- which
// produces different, still-plausible speech rather than an error:
//
// 1. Pre-tokenization splits the text on a pattern with Unicode character
//    classes. The classes come from a generated table because classifying them
//    by hand is wrong exactly at the edges.
// 2. Byte-level encoding remaps every byte into a printable code point, so the
//    vocabulary can be text while covering all 256 byte values.
// 3. Merging applies the ranked pairs greedily, lowest rank first, until no pair
//    in the piece is mergeable.

#include "bpe.h"

#include "unicode-ranges.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <unordered_map>

namespace synth::qwen3tts {

namespace {

bool in_ranges(uint32_t codepoint, const CodepointRange * ranges, size_t count) {
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

bool is_letter(uint32_t codepoint) {
    return in_ranges(codepoint, kUnicodeLetterRanges, std::size(kUnicodeLetterRanges));
}

bool is_number(uint32_t codepoint) {
    return in_ranges(codepoint, kUnicodeNumberRanges, std::size(kUnicodeNumberRanges));
}

bool is_whitespace(uint32_t codepoint) {
    return in_ranges(codepoint, kUnicodeWhitespaceRanges, std::size(kUnicodeWhitespaceRanges));
}

// Decodes one UTF-8 sequence, reporting how many bytes it consumed. A malformed
// byte is taken as itself rather than rejected: the caller is tokenizing text
// that has already been accepted at the public seam.
uint32_t decode_utf8(const std::string & text, size_t offset, size_t & length) {
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

struct Codepoint {
    uint32_t value  = 0;
    size_t   offset = 0;
    size_t   length = 0;
};

std::vector<Codepoint> decode_all(const std::string & text) {
    std::vector<Codepoint> out;
    size_t                 offset = 0;
    while (offset < text.size()) {
        size_t         length = 0;
        const uint32_t value  = decode_utf8(text, offset, length);
        out.push_back(Codepoint{ value, offset, length });
        offset += length;
    }
    return out;
}

// The Qwen pre-tokenizer pattern, in order:
//
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   [^\r\n\p{L}\p{N}]?\p{L}+
//   \p{N}
//    ?[^\s\p{L}\p{N}]+[\r\n]*
//   \s*[\r\n]+
//   \s+(?!\S)
//   \s+
//
// Written out rather than handed to a regex engine because no standard library
// implements \p{L}. Alternation is first-match-wins, so the order below is the
// pattern's order and not an arrangement.
class PreTokenizer {
  public:
    explicit PreTokenizer(const std::string & text) : text_(text), points_(decode_all(text)) {}

    std::vector<std::string> split() {
        std::vector<std::string> pieces;
        size_t                   index = 0;
        while (index < points_.size()) {
            const size_t next  = match(index);
            // Every branch of the pattern consumes at least one code point, and
            // the last is `\s+`, so a zero-width match would be a defect here.
            const size_t taken = next > index ? next : index + 1;
            pieces.push_back(slice(index, taken));
            index = taken;
        }
        return pieces;
    }

  private:
    uint32_t at(size_t index) const { return index < points_.size() ? points_[index].value : 0; }

    bool has(size_t index) const { return index < points_.size(); }

    std::string slice(size_t from, size_t to) const {
        const size_t begin = points_[from].offset;
        const size_t end   = to < points_.size() ? points_[to].offset : text_.size();
        return text_.substr(begin, end - begin);
    }

    // Case-insensitive English contraction suffixes.
    size_t match_contraction(size_t index) const {
        if (at(index) != '\'') {
            return index;
        }
        static const char * const kSuffixes[] = { "s", "t", "re", "ve", "m", "ll", "d" };
        for (const char * suffix : kSuffixes) {
            const size_t length = std::strlen(suffix);
            size_t       cursor = index + 1;
            bool         all    = true;
            for (size_t position = 0; position < length; ++position, ++cursor) {
                const uint32_t lowered = at(cursor) | 0x20u;
                if (!has(cursor) || lowered != static_cast<uint32_t>(suffix[position])) {
                    all = false;
                    break;
                }
            }
            if (all) {
                return cursor;
            }
        }
        return index;
    }

    size_t match(size_t index) const {
        size_t cursor = match_contraction(index);
        if (cursor > index) {
            return cursor;
        }

        // An optional leading character that is neither a newline nor a letter
        // nor a number, then one or more letters. This is what keeps a space
        // attached to the word that follows it.
        {
            size_t start = index;
            if (has(start) && at(start) != '\r' && at(start) != '\n' && !is_letter(at(start)) &&
                !is_number(at(start))) {
                ++start;
            }
            if (has(start) && is_letter(at(start))) {
                cursor = start;
                while (has(cursor) && is_letter(at(cursor))) {
                    ++cursor;
                }
                return cursor;
            }
        }

        // One number, alone: digits do not group.
        if (has(index) && is_number(at(index))) {
            return index + 1;
        }

        // An optional space, then punctuation, then any newlines trailing it.
        {
            size_t start = index;
            if (has(start) && at(start) == ' ') {
                ++start;
            }
            if (has(start) && !is_whitespace(at(start)) && !is_letter(at(start)) && !is_number(at(start))) {
                cursor = start;
                while (has(cursor) && !is_whitespace(at(cursor)) && !is_letter(at(cursor)) && !is_number(at(cursor))) {
                    ++cursor;
                }
                while (has(cursor) && (at(cursor) == '\r' || at(cursor) == '\n')) {
                    ++cursor;
                }
                return cursor;
            }
        }

        // Whitespace then newlines.
        {
            size_t cursor_space = index;
            while (has(cursor_space) && is_whitespace(at(cursor_space)) && at(cursor_space) != '\r' &&
                   at(cursor_space) != '\n') {
                ++cursor_space;
            }
            if (has(cursor_space) && (at(cursor_space) == '\r' || at(cursor_space) == '\n')) {
                cursor = cursor_space;
                while (has(cursor) && (at(cursor) == '\r' || at(cursor) == '\n')) {
                    ++cursor;
                }
                return cursor;
            }
        }

        // Whitespace, which gives up its last character when a non-space follows:
        // that trailing space belongs to the next piece, and this is the rule
        // that makes " a" one token rather than " " and "a".
        if (has(index) && is_whitespace(at(index))) {
            cursor = index;
            while (has(cursor) && is_whitespace(at(cursor))) {
                ++cursor;
            }
            if (has(cursor) && cursor > index + 1) {
                return cursor - 1;
            }
            return cursor;
        }
        return index;
    }

    const std::string &    text_;
    std::vector<Codepoint> points_;
};

// The byte-level alphabet: every byte becomes a printable code point, so a
// vocabulary of text can still name any byte. Printable ASCII and Latin-1 map to
// themselves; everything else is lifted above U+0100.
std::array<uint32_t, 256> byte_to_codepoint() {
    std::array<uint32_t, 256> table{};
    std::array<bool, 256>     direct{};
    for (uint32_t value = '!'; value <= '~'; ++value) {
        direct[value] = true;
    }
    for (uint32_t value = 0xA1; value <= 0xAC; ++value) {
        direct[value] = true;
    }
    for (uint32_t value = 0xAE; value <= 0xFF; ++value) {
        direct[value] = true;
    }
    uint32_t lifted = 0;
    for (uint32_t value = 0; value < 256; ++value) {
        table[value] = direct[value] ? value : 256 + lifted++;
    }
    return table;
}

void append_utf8(std::string & out, uint32_t codepoint) {
    if (codepoint < 0x80) {
        out.push_back(static_cast<char>(codepoint));
    } else if (codepoint < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    }
}

class BpeFrontend : public TextFrontend {
  public:
    BpeFrontend(std::unordered_map<std::string, int32_t>     vocab,
                std::unordered_map<std::string, int32_t>     ranks,
                std::vector<std::pair<std::string, int32_t>> specials,
                std::string                                  prefix,
                std::string                                  suffix) :
        vocab_(std::move(vocab)),
        ranks_(std::move(ranks)),
        specials_(std::move(specials)),
        prefix_(std::move(prefix)),
        suffix_(std::move(suffix)),
        byte_table_(byte_to_codepoint()) {
        // Longest first, so a marker that is a prefix of another cannot win.
        std::sort(specials_.begin(), specials_.end(),
                  [](const auto & left, const auto & right) { return left.first.size() > right.first.size(); });
    }

    synth_input_flags_t input_flags() const noexcept override { return SYNTH_INPUT_SUPPORT_TEXT_UTF8; }

    synth_status_t prepare(synth_input_kind_t     input_kind,
                           const void *           input_data,
                           uint64_t               input_size,
                           uint64_t               max_output_tokens,
                           std::vector<int32_t> & output) const override {
        output.clear();
        if (input_kind != SYNTH_INPUT_TEXT_UTF8 || (input_data == nullptr && input_size != 0)) {
            return SYNTH_ERR_INVALID_ARG;
        }
        // The turn wrapper goes on before tokenizing, so its markers are
        // matched as the special tokens they are rather than as text.
        const std::string text =
            prefix_ + std::string(static_cast<const char *>(input_data), static_cast<size_t>(input_size)) + suffix_;

        // Special markers are matched literally and never split, which is what
        // lets the prompt template's own markers survive tokenization.
        size_t cursor = 0;
        while (cursor < text.size()) {
            const size_t marker = next_special(text, cursor);
            if (marker > cursor) {
                if (!encode_span(text.substr(cursor, marker - cursor), output)) {
                    return SYNTH_ERR_INVALID_ARG;
                }
            }
            if (marker >= text.size()) {
                break;
            }
            const std::pair<std::string, int32_t> * hit = special_at(text, marker);
            if (hit == nullptr) {
                return SYNTH_ERR_INVALID_ARG;
            }
            output.push_back(hit->second);
            cursor = marker + hit->first.size();
        }

        if (max_output_tokens != 0 && output.size() > max_output_tokens) {
            output.clear();
            return SYNTH_ERR_INPUT_TOO_LONG;
        }
        return SYNTH_OK;
    }

  private:
    const std::pair<std::string, int32_t> * special_at(const std::string & text, size_t offset) const {
        for (const std::pair<std::string, int32_t> & special : specials_) {
            if (text.compare(offset, special.first.size(), special.first) == 0) {
                return &special;
            }
        }
        return nullptr;
    }

    size_t next_special(const std::string & text, size_t from) const {
        for (size_t offset = from; offset < text.size(); ++offset) {
            if (special_at(text, offset) != nullptr) {
                return offset;
            }
        }
        return text.size();
    }

    bool encode_span(const std::string & span, std::vector<int32_t> & output) const {
        PreTokenizer pre(span);
        for (const std::string & piece : pre.split()) {
            // Byte level: the piece's bytes, each remapped so the vocabulary can
            // name it.
            std::vector<std::string> symbols;
            symbols.reserve(piece.size());
            for (char raw : piece) {
                std::string symbol;
                append_utf8(symbol, byte_table_[static_cast<unsigned char>(raw)]);
                symbols.push_back(std::move(symbol));
            }
            merge(symbols);
            for (const std::string & symbol : symbols) {
                const auto found = vocab_.find(symbol);
                if (found == vocab_.end()) {
                    return false;
                }
                output.push_back(found->second);
            }
        }
        return true;
    }

    // Greedy by rank: repeatedly merge the best-ranked adjacent pair until none
    // is mergeable. Taking pairs left to right instead would produce a different
    // and equally plausible tokenization.
    void merge(std::vector<std::string> & symbols) const {
        while (symbols.size() > 1) {
            int32_t best_rank  = std::numeric_limits<int32_t>::max();
            size_t  best_index = symbols.size();
            for (size_t index = 0; index + 1 < symbols.size(); ++index) {
                const auto found = ranks_.find(symbols[index] + " " + symbols[index + 1]);
                if (found != ranks_.end() && found->second < best_rank) {
                    best_rank  = found->second;
                    best_index = index;
                }
            }
            if (best_index == symbols.size()) {
                return;
            }
            symbols[best_index] += symbols[best_index + 1];
            symbols.erase(symbols.begin() + static_cast<long>(best_index) + 1);
        }
    }

    std::unordered_map<std::string, int32_t>     vocab_;
    std::unordered_map<std::string, int32_t>     ranks_;
    std::vector<std::pair<std::string, int32_t>> specials_;
    std::string                                  prefix_;
    std::string                                  suffix_;
    std::array<uint32_t, 256>                    byte_table_;
};

}  // namespace

std::vector<std::string> qwen_pretokenize(const std::string & text) {
    PreTokenizer pre(text);
    return pre.split();
}

synth_status_t make_bpe_frontend(const BpeFrontendConfig & config, std::unique_ptr<TextFrontend> & output) {
    output.reset();
    if (config.vocab.empty() || config.contract_version != 1) {
        return SYNTH_ERR_GGUF;
    }

    std::unordered_map<std::string, int32_t> vocab;
    vocab.reserve(config.vocab.size());
    for (size_t index = 0; index < config.vocab.size(); ++index) {
        if (config.vocab[index].empty()) {
            continue;
        }
        vocab.emplace(config.vocab[index], static_cast<int32_t>(index));
    }

    std::unordered_map<std::string, int32_t> ranks;
    ranks.reserve(config.merges.size());
    for (size_t index = 0; index < config.merges.size(); ++index) {
        // A merge is a pair separated by one space; anything else would silently
        // rank a pair that can never occur.
        if (config.merges[index].find(' ') == std::string::npos) {
            return SYNTH_ERR_GGUF;
        }
        ranks.emplace(config.merges[index], static_cast<int32_t>(index));
    }

    std::vector<std::pair<std::string, int32_t>> specials;
    specials.reserve(config.special_tokens.size());
    for (const std::pair<std::string, int32_t> & marker : config.special_tokens) {
        // An added token sits past the vocabulary, so its id is taken on trust;
        // what is checked is that it is one, and that it names something.
        if (marker.first.empty() || marker.second < 0) {
            return SYNTH_ERR_GGUF;
        }
        specials.push_back(marker);
    }

    output = std::make_unique<BpeFrontend>(std::move(vocab), std::move(ranks), std::move(specials), config.prefix,
                                           config.suffix);
    return SYNTH_OK;
}

std::string qwen_assistant_turn(const std::string & text) {
    return "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
}

}  // namespace synth::qwen3tts
