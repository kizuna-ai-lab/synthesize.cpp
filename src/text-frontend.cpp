#include "text-frontend.h"

#include <cstddef>
#include <limits>
#include <new>
#include <unordered_map>

namespace synth {

namespace {

bool continuation(unsigned char byte) {
    return (byte & 0xc0u) == 0x80u;
}

bool decode_scalar(const unsigned char * bytes, size_t size, size_t & offset, uint32_t & scalar) {
    if (offset >= size) {
        return false;
    }
    const unsigned char first = bytes[offset];
    if (first <= 0x7fu) {
        scalar = first;
        ++offset;
        return true;
    }
    if (first >= 0xc2u && first <= 0xdfu) {
        if (size - offset < 2 || !continuation(bytes[offset + 1])) {
            return false;
        }
        scalar = (static_cast<uint32_t>(first & 0x1fu) << 6u) | static_cast<uint32_t>(bytes[offset + 1] & 0x3fu);
        offset += 2;
        return true;
    }
    if (first >= 0xe0u && first <= 0xefu) {
        if (size - offset < 3 || !continuation(bytes[offset + 1]) || !continuation(bytes[offset + 2])) {
            return false;
        }
        const unsigned char second = bytes[offset + 1];
        if ((first == 0xe0u && second < 0xa0u) || (first == 0xedu && second > 0x9fu)) {
            return false;
        }
        scalar = (static_cast<uint32_t>(first & 0x0fu) << 12u) | (static_cast<uint32_t>(second & 0x3fu) << 6u) |
                 static_cast<uint32_t>(bytes[offset + 2] & 0x3fu);
        offset += 3;
        return true;
    }
    if (first >= 0xf0u && first <= 0xf4u) {
        if (size - offset < 4 || !continuation(bytes[offset + 1]) || !continuation(bytes[offset + 2]) ||
            !continuation(bytes[offset + 3])) {
            return false;
        }
        const unsigned char second = bytes[offset + 1];
        if ((first == 0xf0u && second < 0x90u) || (first == 0xf4u && second > 0x8fu)) {
            return false;
        }
        scalar = (static_cast<uint32_t>(first & 0x07u) << 18u) | (static_cast<uint32_t>(second & 0x3fu) << 12u) |
                 (static_cast<uint32_t>(bytes[offset + 2] & 0x3fu) << 6u) |
                 static_cast<uint32_t>(bytes[offset + 3] & 0x3fu);
        offset += 4;
        return true;
    }
    return false;
}

bool one_scalar(const std::string & value, uint32_t & scalar) {
    if (value.empty()) {
        return false;
    }
    size_t offset = 0;
    if (!decode_scalar(reinterpret_cast<const unsigned char *>(value.data()), value.size(), offset, scalar)) {
        return false;
    }
    return offset == value.size();
}

class SymbolMapFrontend final : public TextFrontend {
  public:
    SymbolMapFrontend(std::unordered_map<uint32_t, int32_t> symbols, int32_t blank_id, SymbolPaddingRule padding_rule) :
        symbols_(std::move(symbols)),
        blank_id_(blank_id),
        padding_rule_(padding_rule) {}

    synth_input_flags_t input_flags() const noexcept override { return SYNTH_INPUT_SUPPORT_PHONEMES_UTF8; }

    synth_status_t prepare(synth_input_kind_t     input_kind,
                           const void *           input_data,
                           uint64_t               input_size,
                           uint64_t               max_output_tokens,
                           std::vector<int32_t> & output) const override {
        output.clear();
        if (input_kind != SYNTH_INPUT_PHONEMES_UTF8) {
            return SYNTH_ERR_UNSUPPORTED_INPUT;
        }
        if (input_data == nullptr || input_size == 0 || max_output_tokens == 0 ||
            input_size > std::numeric_limits<size_t>::max()) {
            return SYNTH_ERR_INVALID_ARG;
        }

        const auto *   bytes      = static_cast<const unsigned char *>(input_data);
        const size_t   size       = static_cast<size_t>(input_size);
        const bool     interleave = padding_rule_ == SymbolPaddingRule::InterleavedBlank;
        const bool     wrap       = padding_rule_ == SymbolPaddingRule::WrapPadToken;
        // A wrapped sequence reserves its trailing pad up front so the limit is
        // enforced against the final length rather than an intermediate one.
        const uint64_t reserved   = wrap ? 1 : 0;
        try {
            if (interleave || wrap) {
                if (max_output_tokens < 1 + reserved) {
                    return SYNTH_ERR_INPUT_TOO_LONG;
                }
                output.push_back(blank_id_);
            }
            size_t offset = 0;
            while (offset < size) {
                uint32_t scalar = 0;
                if (!decode_scalar(bytes, size, offset, scalar)) {
                    output.clear();
                    return SYNTH_ERR_TEXT_FRONTEND;
                }
                const auto symbol = symbols_.find(scalar);
                if (symbol == symbols_.end()) {
                    output.clear();
                    return SYNTH_ERR_TEXT_FRONTEND;
                }
                const uint64_t additional = interleave ? 2 : 1;
                const uint64_t budget     = max_output_tokens - reserved;
                if (output.size() > budget || additional > budget - output.size()) {
                    output.clear();
                    return SYNTH_ERR_INPUT_TOO_LONG;
                }
                output.push_back(symbol->second);
                if (interleave) {
                    output.push_back(blank_id_);
                }
            }
            if (wrap) {
                output.push_back(blank_id_);
            }
        } catch (const std::bad_alloc &) {
            output.clear();
            return SYNTH_ERR_OOM;
        }
        return SYNTH_OK;
    }

  private:
    std::unordered_map<uint32_t, int32_t> symbols_;
    int32_t                               blank_id_     = 0;
    SymbolPaddingRule                     padding_rule_ = SymbolPaddingRule::None;
};

}  // namespace

synth_status_t make_symbol_map_frontend(const SymbolMapFrontendConfig & config,
                                        std::unique_ptr<TextFrontend> & output) {
    output.reset();
    if (config.provider_id != "synthesize.symbol_map" || config.contract_version != 1 ||
        config.mapping_mode != SymbolMappingMode::UnicodeScalar || config.symbols.empty() ||
        config.symbols.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
        config.blank_id >= config.symbols.size()) {
        return SYNTH_ERR_TEXT_FRONTEND;
    }
    switch (config.padding_rule) {
        case SymbolPaddingRule::None:
        case SymbolPaddingRule::InterleavedBlank:
        case SymbolPaddingRule::WrapPadToken:
            break;
        default:
            return SYNTH_ERR_TEXT_FRONTEND;
    }
    try {
        std::unordered_map<uint32_t, int32_t> symbols;
        symbols.reserve(config.symbols.size());
        for (size_t index = 0; index < config.symbols.size(); ++index) {
            // A sparse vocabulary leaves unmapped ids empty; they are reserved
            // slots in the embedding table rather than mappable symbols.
            if (config.symbols[index].empty()) {
                continue;
            }
            uint32_t scalar = 0;
            if (!one_scalar(config.symbols[index], scalar)) {
                return SYNTH_ERR_TEXT_FRONTEND;
            }
            symbols[scalar] = static_cast<int32_t>(index);
        }
        if (symbols.empty()) {
            return SYNTH_ERR_TEXT_FRONTEND;
        }
        output = std::make_unique<SymbolMapFrontend>(std::move(symbols), static_cast<int32_t>(config.blank_id),
                                                     config.padding_rule);
        return SYNTH_OK;
    } catch (const std::bad_alloc &) {
        output.reset();
        return SYNTH_ERR_OOM;
    }
}

}  // namespace synth
