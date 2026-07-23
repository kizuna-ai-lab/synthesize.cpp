#include "synthesize.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>

struct synth_voice_profile {
    const synth_model_t * model = nullptr;
};

namespace {

struct ByteBufferStorage {
    synth_byte_buffer_t        public_value{};
    std::unique_ptr<uint8_t[]> data;
};

template <typename T> void initialize_struct(T * output, uint64_t struct_size) {
    if (output == nullptr || struct_size == 0) {
        return;
    }
    std::memset(output, 0, std::min(static_cast<size_t>(struct_size), sizeof(*output)));
    if (struct_size >= sizeof(output->struct_size)) {
        output->struct_size = struct_size;
    }
}

template <typename T, typename V> V read_visible(const T * input, size_t offset, V default_value) {
    if (input == nullptr || input->struct_size < offset + sizeof(V)) {
        return default_value;
    }
    V value;
    std::memcpy(&value, reinterpret_cast<const unsigned char *>(input) + offset, sizeof(value));
    return value;
}

bool valid_diagnostic_sink(const synth_diagnostic_sink_t * sink) {
    return sink == nullptr ||
           (sink->struct_size >= offsetof(synth_diagnostic_sink_t, emit) + sizeof(sink->emit) && sink->emit != nullptr);
}

template <typename T> synth_status_t validate_unsupported_params(const T * params, size_t diagnostics_offset) {
    if (params == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (params->struct_size < sizeof(uint64_t)) {
        return SYNTH_ERR_BAD_STRUCT_SIZE;
    }
    const synth_diagnostic_sink_t * diagnostics =
        read_visible(params, diagnostics_offset, static_cast<const synth_diagnostic_sink_t *>(nullptr));
    return valid_diagnostic_sink(diagnostics) ? SYNTH_ERR_UNSUPPORTED_VOICE : SYNTH_ERR_INVALID_ARG;
}

}  // namespace

void synth_voice_profile_capabilities_init(synth_voice_profile_capabilities_t * capabilities, uint64_t struct_size) {
    initialize_struct(capabilities, struct_size);
}

synth_status_t synth_model_get_voice_profile_capabilities(const synth_model_t *                model,
                                                          synth_voice_profile_capabilities_t * out_capabilities) {
    if (model == nullptr || out_capabilities == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (out_capabilities->struct_size < sizeof(uint64_t)) {
        return SYNTH_ERR_BAD_STRUCT_SIZE;
    }
    const uint64_t struct_size = out_capabilities->struct_size;
    synth_voice_profile_capabilities_init(out_capabilities, struct_size);
    return SYNTH_OK;
}

void synth_voice_reference_init(synth_voice_reference_t * reference, uint64_t struct_size) {
    initialize_struct(reference, struct_size);
}

void synth_voice_reference_params_init(synth_voice_reference_params_t * params, uint64_t struct_size) {
    initialize_struct(params, struct_size);
}

synth_status_t synth_voice_profile_create_from_reference(const synth_model_t *                  model,
                                                         const synth_voice_reference_params_t * params,
                                                         synth_voice_profile_t **               out_profile) {
    if (out_profile == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    *out_profile = nullptr;
    if (model == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    return validate_unsupported_params(params, offsetof(synth_voice_reference_params_t, diagnostics));
}

void synth_voice_description_params_init(synth_voice_description_params_t * params, uint64_t struct_size) {
    initialize_struct(params, struct_size);
}

synth_status_t synth_voice_profile_create_from_description(const synth_model_t *                    model,
                                                           const synth_voice_description_params_t * params,
                                                           synth_voice_profile_t **                 out_profile) {
    if (out_profile == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    *out_profile = nullptr;
    if (model == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    return validate_unsupported_params(params, offsetof(synth_voice_description_params_t, diagnostics));
}

void synth_voice_random_params_init(synth_voice_random_params_t * params, uint64_t struct_size) {
    initialize_struct(params, struct_size);
}

synth_status_t synth_voice_profile_create_random(const synth_model_t *               model,
                                                 const synth_voice_random_params_t * params,
                                                 synth_voice_profile_t **            out_profile) {
    if (out_profile == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    *out_profile = nullptr;
    if (model == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    return validate_unsupported_params(params, offsetof(synth_voice_random_params_t, diagnostics));
}

void synth_voice_profile_load_params_init(synth_voice_profile_load_params_t * params, uint64_t struct_size) {
    initialize_struct(params, struct_size);
}

void synth_voice_profile_serialize_params_init(synth_voice_profile_serialize_params_t * params, uint64_t struct_size) {
    initialize_struct(params, struct_size);
}

synth_status_t synth_voice_profile_load_from_memory(const synth_model_t *                     model,
                                                    const synth_voice_profile_load_params_t * params,
                                                    synth_voice_profile_t **                  out_profile) {
    if (out_profile == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    *out_profile = nullptr;
    if (model == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    return validate_unsupported_params(params, offsetof(synth_voice_profile_load_params_t, diagnostics));
}

synth_status_t synth_voice_profile_serialize(const synth_voice_profile_t *                  profile,
                                             const synth_voice_profile_serialize_params_t * params,
                                             synth_byte_buffer_t **                         out_data) {
    if (out_data == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    *out_data = nullptr;
    if (profile == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (params != nullptr && params->struct_size < sizeof(uint64_t)) {
        return SYNTH_ERR_BAD_STRUCT_SIZE;
    }
    const synth_diagnostic_sink_t * diagnostics =
        read_visible(params, offsetof(synth_voice_profile_serialize_params_t, diagnostics),
                     static_cast<const synth_diagnostic_sink_t *>(nullptr));
    return valid_diagnostic_sink(diagnostics) ? SYNTH_ERR_UNSUPPORTED_VOICE : SYNTH_ERR_INVALID_ARG;
}

void synth_byte_buffer_free(synth_byte_buffer_t * buffer) {
    if (buffer == nullptr) {
        return;
    }
    auto * storage = reinterpret_cast<ByteBufferStorage *>(buffer);
    delete storage;
}

void synth_voice_profile_free(synth_voice_profile_t * profile) {
    delete profile;
}
