#pragma once

#include "synthesize.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace synth {

enum class SymbolMappingMode : uint32_t {
    UnicodeScalar = 1,
};

// How a Model Variant pads the mapped symbol sequence. The rule is package
// metadata rather than a frontend-wide constant: VITS interleaves a blank
// around every symbol, while Kokoro wraps the sequence with one pad token.
enum class SymbolPaddingRule : uint32_t {
    None             = 0,
    InterleavedBlank = 1,
    WrapPadToken     = 2,
};

struct SymbolMapFrontendConfig {
    std::string              provider_id;
    uint32_t                 contract_version = 0;
    SymbolMappingMode        mapping_mode     = SymbolMappingMode::UnicodeScalar;
    // Dense table indexed by token id. An empty entry marks an id the Model
    // Variant does not map, which a sparse vocabulary leaves unused.
    std::vector<std::string> symbols;
    uint32_t                 blank_id     = 0;
    SymbolPaddingRule        padding_rule = SymbolPaddingRule::None;
};

class TextFrontend {
  public:
    virtual ~TextFrontend() = default;

    virtual synth_input_flags_t input_flags() const noexcept                 = 0;
    virtual synth_status_t      prepare(synth_input_kind_t     input_kind,
                                        const void *           input_data,
                                        uint64_t               input_size,
                                        uint64_t               max_output_tokens,
                                        std::vector<int32_t> & output) const = 0;
};

synth_status_t make_symbol_map_frontend(const SymbolMapFrontendConfig & config, std::unique_ptr<TextFrontend> & output);

}  // namespace synth
