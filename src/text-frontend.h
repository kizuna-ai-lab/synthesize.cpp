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

struct SymbolMapFrontendConfig {
    std::string              provider_id;
    uint32_t                 contract_version = 0;
    SymbolMappingMode        mapping_mode     = SymbolMappingMode::UnicodeScalar;
    std::vector<std::string> symbols;
    uint32_t                 blank_id  = 0;
    bool                     add_blank = false;
};

class TextFrontend {
  public:
    virtual ~TextFrontend() = default;

    virtual synth_input_flags_t input_flags() const noexcept = 0;
    virtual synth_status_t      prepare(synth_input_kind_t   input_kind,
                                        const void *         input_data,
                                        uint64_t             input_size,
                                        uint64_t             max_output_tokens,
                                        std::vector<int32_t> & output) const = 0;
};

synth_status_t make_symbol_map_frontend(const SymbolMapFrontendConfig & config,
                                        std::unique_ptr<TextFrontend> & output);

}  // namespace synth
