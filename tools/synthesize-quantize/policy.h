#pragma once

#include "ggml.h"

#include <cstdint>
#include <string>

namespace synth::quantize {

enum class TensorLayout {
    Native,
    PackedMatrix,
};

struct Profile {
    const char * name;
    ggml_type    matrix_weight_type;
    TensorLayout matrix_weight_layout;
    ggml_type    transpose_weight_type;
    ggml_type    sensitive_type;
    uint32_t     file_type;
    uint32_t     version;
};

struct TargetSpec {
    ggml_type    type;
    TensorLayout layout;
};

// Profile lookup is case-insensitive. Only profiles implemented and validated
// by this tool are exposed.
const Profile * find_profile(const char * name);

// Resolves one canonical VITS tensor name. Returns false for every name outside
// the architecture catalog so a converter/runtime change cannot be silently
// assigned a storage type by a catch-all suffix rule.
bool resolve_vits_target_type(const Profile & profile, const std::string & name, ggml_type & type_out);
bool resolve_vits_target_spec(const Profile & profile, const std::string & name, TargetSpec & spec_out);

}  // namespace synth::quantize
