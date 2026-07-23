#pragma once

#include <string>

namespace synth::quantize {

// Converts a source GGUF to a named derived storage profile. The destination
// must not already exist. Data is written to a sibling temporary file and
// renamed only after the complete GGUF has been serialized.
bool quantize_file(const std::string & input_path,
                   const std::string & output_path,
                   const char *        profile_name,
                   std::string &       error_out);

}  // namespace synth::quantize
