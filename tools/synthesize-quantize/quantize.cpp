#include "quantize.h"

#include "arch/kokoro/quantization.h"
#include "ggml.h"
#include "gguf.h"
#include "policy.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

namespace synth::quantize {

namespace {

struct InputContexts {
    gguf_context * gguf = nullptr;
    ggml_context * ggml = nullptr;

    ~InputContexts() {
        if (gguf != nullptr) {
            gguf_free(gguf);
        }
        if (ggml != nullptr) {
            ggml_free(ggml);
        }
    }
};

struct OutputContexts {
    gguf_context * gguf = nullptr;
    ggml_context * ggml = nullptr;

    ~OutputContexts() {
        if (gguf != nullptr) {
            gguf_free(gguf);
        }
        if (ggml != nullptr) {
            ggml_free(ggml);
        }
    }
};

bool fail(std::string & error, const std::string & message) {
    error = message;
    return false;
}

bool read_string(const gguf_context * gguf, const char * key, std::string & value) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_STRING) {
        return false;
    }
    const char * stored = gguf_get_val_str(gguf, id);
    if (stored == nullptr) {
        return false;
    }
    value = stored;
    return true;
}

bool dequantize_to_f32(const ggml_tensor * tensor, std::vector<float> & output) {
    const int64_t count = ggml_nelements(tensor);
    if (count <= 0) {
        return false;
    }
    output.resize(static_cast<size_t>(count));
    if (tensor->type == GGML_TYPE_F32) {
        std::memcpy(output.data(), tensor->data, static_cast<size_t>(count) * sizeof(float));
        return true;
    }
    const ggml_type_traits * traits = ggml_get_type_traits(tensor->type);
    if (traits == nullptr || traits->to_float == nullptr) {
        return false;
    }
    traits->to_float(tensor->data, output.data(), count);
    return true;
}

}  // namespace

bool quantize_file(const std::string & input_path,
                   const std::string & output_path,
                   const char *        profile_name,
                   std::string &       error_out) {
    error_out.clear();
    const Profile * profile = find_profile(profile_name);
    if (profile == nullptr) {
        return fail(error_out,
                    "unknown quantization profile: " + std::string(profile_name == nullptr ? "(null)" : profile_name));
    }
    if (input_path.empty() || output_path.empty() || input_path == output_path) {
        return fail(error_out, "input and output paths must be distinct and non-empty");
    }

    std::error_code filesystem_error;
    if (std::filesystem::exists(output_path, filesystem_error)) {
        return fail(error_out, "output already exists: " + output_path);
    }
    if (filesystem_error) {
        return fail(error_out, "could not inspect output path: " + filesystem_error.message());
    }

    InputContexts    input;
    gguf_init_params input_params{};
    input_params.no_alloc = false;
    input_params.ctx      = &input.ggml;
    input.gguf            = gguf_init_from_file(input_path.c_str(), input_params);
    if (input.gguf == nullptr || input.ggml == nullptr) {
        return fail(error_out, "could not read input GGUF: " + input_path);
    }

    // Each family gets its own catalog resolver. There is deliberately no
    // catch-all: a tensor the resolver does not recognise stops the run, so a
    // converter or runtime change cannot quietly acquire a storage type.
    std::string architecture;
    if (!read_string(input.gguf, "general.architecture", architecture)) {
        return fail(error_out, "input GGUF declares no general.architecture");
    }
    bool (*resolve_target_spec)(const Profile &, const std::string &, TargetSpec &) = nullptr;
    if (architecture == "vits") {
        resolve_target_spec = resolve_vits_target_spec;
    } else if (architecture == "kokoro") {
        resolve_target_spec = resolve_kokoro_target_spec;
    } else if (architecture == "qwen3-tts") {
        resolve_target_spec = resolve_qwen3_tts_target_spec;
    } else if (architecture == "omnivoice") {
        resolve_target_spec = resolve_omnivoice_target_spec;
    } else {
        return fail(error_out, "unsupported input GGUF general.architecture: " + architecture);
    }

    struct PlanEntry {
        ggml_tensor *          source;
        ggml_type              target_type;
        TensorLayout           target_layout;
        std::array<int64_t, 4> target_ne;
        int                    target_n_dims;
        int64_t                target_rows;
        size_t                 target_bytes;
    };

    const int64_t n_tensors = gguf_get_n_tensors(input.gguf);
    if (n_tensors <= 0) {
        return fail(error_out, "input GGUF contains no tensors");
    }

    std::vector<PlanEntry> plan;
    plan.reserve(static_cast<size_t>(n_tensors));
    size_t total_data_bytes = 0;
    for (int64_t index = 0; index < n_tensors; ++index) {
        const char *  name   = gguf_get_tensor_name(input.gguf, index);
        ggml_tensor * tensor = name == nullptr ? nullptr : ggml_get_tensor(input.ggml, name);
        if (tensor == nullptr) {
            return fail(error_out, "input tensor catalog is inconsistent");
        }
        // BF16 joins the list because Qwen3-TTS's source profile is mixed: its
        // talker half is the checkpoint's bfloat16 and its codec half is F32.
        // dequantize_to_f32 already handles any type carrying a to_float trait.
        if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_BF16) {
            return fail(error_out, "unsupported source tensor type for " + std::string(name));
        }
        TargetSpec target{};
        if (!resolve_target_spec(*profile, name, target)) {
            return fail(error_out, "unknown " + architecture + " tensor: " + std::string(name));
        }
        // A matrix weight whose packed row is not a whole number of blocks
        // falls back to the halved type rather than failing the run. The
        // predicate is shared with the runtime so both agree which ones those
        // are; see src/arch/kokoro/quantization.h.
        // A two-dimensional weight is already the matrix the multiply wants, and
        // packing one would flatten it into a single meaningless row: a
        // [1024, 3072] projection would become one row of 3145728. Packing
        // exists for convolution kernels, whose fastest dimension is the kernel
        // and therefore too short to hold a block. Kokoro's projections, the
        // whole of Qwen3-TTS's quantizable half, and omnivoice's HuBERT
        // attention/feed-forward Linears (feature_projection.projection plus
        // every layer's q/k/v/out_proj and inter_dense/output_dense) are the
        // former.
        //
        // Naming the families is deliberate. `ggml_n_dims` collapses trailing
        // unit dimensions, so VITS's `decoder.post.weight` at [7, 32, 1] reports
        // as two-dimensional and would be read as a matrix -- leaving a row of
        // seven, which no block divides. The test fixture catches exactly that,
        // and it is why this cannot be written as a general rule about shape.
        // omnivoice has the identical hazard shape at
        // codec.acoustic_decoder.conv2.weight, also [7, 32, 1] -- but unlike
        // VITS it also has genuine Linears to protect (above), so it cannot
        // simply sit outside matrix_family the way VITS does. It is named out
        // of the demotion below instead: packing it succeeds (7 * 32 = 224, a
        // whole number of Q8_0's 32-element blocks) where demoting it to a
        // native row of 7 would not, and there is exactly one such tensor in
        // this family's whole catalog (see Task 2's report for the survey
        // that confirmed it).
        const bool matrix_family =
            architecture == "kokoro" || architecture == "qwen3-tts" || architecture == "omnivoice";
        const bool omnivoice_collapsed_conv_kernel =
            architecture == "omnivoice" && std::string(name) == "codec.acoustic_decoder.conv2.weight";
        if (matrix_family && !omnivoice_collapsed_conv_kernel && target.layout == TensorLayout::PackedMatrix &&
            ggml_n_dims(tensor) < 3) {
            target.layout = TensorLayout::Native;
        }
        if (architecture == "kokoro" && ggml_is_quantized(target.type) &&
            !synth::kokoro::matrix_is_block_quantizable(tensor->ne, ggml_n_dims(tensor), false,
                                                        ggml_blck_size(target.type))) {
            target = { profile->transpose_weight_type, TensorLayout::Native };
        }
        std::array<int64_t, 4> target_ne     = { tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3] };
        int                    target_n_dims = ggml_n_dims(tensor);
        if (target.layout == TensorLayout::PackedMatrix) {
            if (tensor->ne[0] <= 0 || tensor->ne[1] <= 0 || tensor->ne[2] <= 0 || tensor->ne[3] != 1 ||
                tensor->ne[0] > std::numeric_limits<int64_t>::max() / tensor->ne[1]) {
                return fail(error_out, "cannot pack tensor shape for " + std::string(name));
            }
            target_ne     = { tensor->ne[0] * tensor->ne[1], tensor->ne[2], 1, 1 };
            target_n_dims = 2;
        }
        const int64_t block_size = ggml_blck_size(target.type);
        if (block_size <= 0 || target_ne[0] % block_size != 0) {
            return fail(error_out, "tensor row size is incompatible with target type for " + std::string(name));
        }
        int64_t rows = 1;
        for (int dimension = 1; dimension < GGML_MAX_DIMS; ++dimension) {
            if (target_ne[dimension] <= 0 || rows > std::numeric_limits<int64_t>::max() / target_ne[dimension]) {
                return fail(error_out, "invalid tensor shape for " + std::string(name));
            }
            rows *= target_ne[dimension];
        }
        const size_t row_size = ggml_row_size(target.type, target_ne[0]);
        if (rows <= 0 || row_size > std::numeric_limits<size_t>::max() / static_cast<size_t>(rows)) {
            return fail(error_out, "tensor size overflow for " + std::string(name));
        }
        const size_t bytes = row_size * static_cast<size_t>(rows);
        if (total_data_bytes > std::numeric_limits<size_t>::max() - bytes - 31) {
            return fail(error_out, "output tensor data size overflow");
        }
        total_data_bytes = (total_data_bytes + bytes + 31) & ~size_t{ 31 };
        plan.push_back({ tensor, target.type, target.layout, target_ne, target_n_dims, rows, bytes });
    }

    const size_t     overhead = static_cast<size_t>(n_tensors) * ggml_tensor_overhead();
    constexpr size_t slack    = 16 * 1024 * 1024;
    if (total_data_bytes > std::numeric_limits<size_t>::max() - overhead - slack) {
        return fail(error_out, "output context size overflow");
    }
    ggml_init_params output_params{};
    output_params.mem_size = total_data_bytes + overhead + slack;
    output_params.no_alloc = false;
    OutputContexts output;
    output.ggml = ggml_init(output_params);
    output.gguf = gguf_init_empty();
    if (output.ggml == nullptr || output.gguf == nullptr) {
        return fail(error_out, "could not allocate output GGUF");
    }

    gguf_set_kv(output.gguf, input.gguf);
    gguf_set_val_u32(output.gguf, "general.file_type", profile->file_type);
    gguf_set_val_str(output.gguf, "synthesize.quantization.profile", profile->name);
    gguf_set_val_u32(output.gguf, "synthesize.quantization.profile_version", profile->version);

    std::vector<float> scratch;
    for (const PlanEntry & entry : plan) {
        ggml_tensor * source = entry.source;
        ggml_tensor * target =
            ggml_new_tensor(output.ggml, entry.target_type, entry.target_n_dims, entry.target_ne.data());
        if (target == nullptr || target->data == nullptr) {
            return fail(error_out, "could not allocate output tensor " + std::string(source->name));
        }
        ggml_set_name(target, source->name);
        if (source->type == target->type && entry.target_layout == TensorLayout::Native) {
            if (ggml_nbytes(source) != entry.target_bytes) {
                return fail(error_out, "source tensor byte size mismatch for " + std::string(source->name));
            }
            std::memcpy(target->data, source->data, entry.target_bytes);
        } else {
            if (!dequantize_to_f32(source, scratch)) {
                return fail(error_out, "could not convert source tensor " + std::string(source->name));
            }
            const size_t written = ggml_quantize_chunk(target->type, scratch.data(), target->data, 0, entry.target_rows,
                                                       entry.target_ne[0], nullptr);
            if (written != entry.target_bytes) {
                return fail(error_out, "output tensor byte size mismatch for " + std::string(source->name));
            }
        }
        gguf_add_tensor(output.gguf, target);
    }

    const std::string temporary_path = output_path + ".tmp";
    std::filesystem::remove(temporary_path, filesystem_error);
    filesystem_error.clear();
    if (!gguf_write_to_file(output.gguf, temporary_path.c_str(), false)) {
        std::filesystem::remove(temporary_path, filesystem_error);
        return fail(error_out, "could not write output GGUF: " + temporary_path);
    }
    std::filesystem::rename(temporary_path, output_path, filesystem_error);
    if (filesystem_error) {
        std::error_code remove_error;
        std::filesystem::remove(temporary_path, remove_error);
        return fail(error_out, "could not commit output GGUF: " + filesystem_error.message());
    }
    ggml_quantize_free();
    return true;
}

}  // namespace synth::quantize
