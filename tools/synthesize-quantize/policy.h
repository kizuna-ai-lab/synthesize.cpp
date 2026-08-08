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
    // The storage type for a weight read by `ggml_get_rows` rather than by a
    // matrix multiply, which CUDA supports for a strictly narrower set of
    // types (no k-quant at all; see src/arch/omnivoice/quantization.h's
    // RowLookup). This field is **inert** for every family whose classifier
    // never reports that role -- today VITS, Kokoro and Qwen3-TTS, all three
    // of which have no resolver arm reading it -- so setting it on a shared
    // profile row cannot change what those families' packages contain.
    ggml_type    row_lookup_type;
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

// The same for Kokoro. Its split is not the same as VITS's by coincidence:
// everything upstream of the decoder decides F0 and the durations, and the
// harmonic source turns a small relative F0 difference into radians of phase
// over an utterance, so PL-BERT, the prosody and duration path, the acoustic
// text encoder, and the Voice tables all stay at the reference dtype. The
// decoder and its generator are where the parameters are and where quantizing
// pays. See reports/porting/kokoro/kokoro-v1-0/_porting-log.md.
bool resolve_kokoro_target_type(const Profile & profile, const std::string & name, ggml_type & type_out);
bool resolve_kokoro_target_spec(const Profile & profile, const std::string & name, TargetSpec & spec_out);

// And for Qwen3-TTS. Its split follows the same principle as the other two and
// lands differently: the quantizer's codebooks and both of its kernel-one
// projections stay at the reference dtype, because a residual codebook's later
// levels carry small magnitudes and a relative error there is a large one
// against the residual it corrects. Per-head norms, layer scales and the
// SnakeBeta curves stay exact for the same reason -- each multiplies a whole
// branch or head, and together they are a rounding error of the file.
bool resolve_qwen3_tts_target_type(const Profile & profile, const std::string & name, ggml_type & type_out);
bool resolve_qwen3_tts_target_spec(const Profile & profile, const std::string & name, TargetSpec & spec_out);

// And for OmniVoice, which is the one family whose profiles do not all
// quantize the same half of the package. `Q8_MIXED` and `F16` quantize the
// codec and hold the generator at F32; `Q8_GEN` does the reverse. The split
// is expressed once, in the family module's classify_tensor_for_half
// (src/arch/omnivoice/quantization.h), which this dispatch and the runtime's
// catalog both read so they cannot disagree about a tensor. See
// omnivoice_quantized_half in policy.cpp for which profile means which half.
bool resolve_omnivoice_target_type(const Profile & profile, const std::string & name, ggml_type & type_out);
bool resolve_omnivoice_target_spec(const Profile & profile, const std::string & name, TargetSpec & spec_out);

}  // namespace synth::quantize
