#include "gguf-metadata.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <fstream>

namespace synth {

synth_status_t stream_tensor_data(const std::string &  path,
                                  const gguf_context * gguf,
                                  ggml_context *       weights_context,
                                  const char *         family) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return SYNTH_ERR_IO;
    }
    const size_t               data_offset = gguf_get_data_offset(gguf);
    std::vector<unsigned char> staging;
    for (ggml_tensor * tensor = ggml_get_first_tensor(weights_context); tensor != nullptr;
         tensor               = ggml_get_next_tensor(weights_context, tensor)) {
        const int64_t index = gguf_find_tensor(gguf, tensor->name);
        if (index < 0) {
            std::fprintf(stderr, "%s: tensor %s is absent from GGUF data\n", family, tensor->name);
            return SYNTH_ERR_GGUF;
        }
        const size_t         bytes = ggml_nbytes(tensor);
        const std::streamoff absolute =
            static_cast<std::streamoff>(data_offset) + static_cast<std::streamoff>(gguf_get_tensor_offset(gguf, index));
        input.seekg(absolute);
        if (!input) {
            return SYNTH_ERR_IO;
        }
        if (staging.size() < bytes) {
            staging.resize(bytes);
        }
        input.read(reinterpret_cast<char *>(staging.data()), static_cast<std::streamsize>(bytes));
        if (!input || static_cast<size_t>(input.gcount()) != bytes) {
            return SYNTH_ERR_IO;
        }
        ggml_backend_tensor_set(tensor, staging.data(), 0, bytes);
    }
    return SYNTH_OK;
}

void GgufMetadata::report(const std::string & key, const char * detail) const {
    std::fprintf(stderr, "%s: %s metadata %s\n", family_.c_str(), detail, key.c_str());
}

bool GgufMetadata::u32(const std::string & key, uint32_t & value) const {
    const int64_t id = gguf_find_key(gguf_, key.c_str());
    if (id < 0 || gguf_get_kv_type(gguf_, id) != GGUF_TYPE_UINT32) {
        report(key, "missing or invalid uint32");
        return false;
    }
    value = gguf_get_val_u32(gguf_, id);
    return true;
}

bool GgufMetadata::u64(const std::string & key, uint64_t & value) const {
    const int64_t id = gguf_find_key(gguf_, key.c_str());
    if (id < 0 || gguf_get_kv_type(gguf_, id) != GGUF_TYPE_UINT64) {
        report(key, "missing or invalid uint64");
        return false;
    }
    value = gguf_get_val_u64(gguf_, id);
    return true;
}

bool GgufMetadata::f32(const std::string & key, float & value) const {
    const int64_t id = gguf_find_key(gguf_, key.c_str());
    if (id < 0 || gguf_get_kv_type(gguf_, id) != GGUF_TYPE_FLOAT32) {
        report(key, "missing or invalid float32");
        return false;
    }
    value = gguf_get_val_f32(gguf_, id);
    return true;
}

bool GgufMetadata::boolean(const std::string & key, bool & value) const {
    const int64_t id = gguf_find_key(gguf_, key.c_str());
    if (id < 0 || gguf_get_kv_type(gguf_, id) != GGUF_TYPE_BOOL) {
        report(key, "missing or invalid bool");
        return false;
    }
    value = gguf_get_val_bool(gguf_, id);
    return true;
}

bool GgufMetadata::string(const std::string & key, std::string & value) const {
    const int64_t id = gguf_find_key(gguf_, key.c_str());
    if (id < 0 || gguf_get_kv_type(gguf_, id) != GGUF_TYPE_STRING) {
        report(key, "missing or invalid string");
        return false;
    }
    const char * stored = gguf_get_val_str(gguf_, id);
    if (stored == nullptr) {
        report(key, "invalid string");
        return false;
    }
    value = stored;
    return true;
}

bool GgufMetadata::positive_i32_array(const std::string & key, std::vector<uint32_t> & value) const {
    const int64_t id = gguf_find_key(gguf_, key.c_str());
    if (id < 0 || gguf_get_kv_type(gguf_, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(gguf_, id) != GGUF_TYPE_INT32) {
        report(key, "missing or invalid int32 array");
        return false;
    }
    const size_t n    = gguf_get_arr_n(gguf_, id);
    const void * data = gguf_get_arr_data(gguf_, id);
    if (n > 0 && data == nullptr) {
        report(key, "invalid int32 array");
        return false;
    }
    if (n == 0) {
        value.clear();
        return true;
    }
    const auto * first = static_cast<const int32_t *>(data);
    value.resize(n);
    for (size_t index = 0; index < n; ++index) {
        if (first[index] <= 0) {
            report(key, "non-positive value in int32 array");
            return false;
        }
        value[index] = static_cast<uint32_t>(first[index]);
    }
    return true;
}

bool GgufMetadata::string_array(const std::string & key, std::vector<std::string> & value) const {
    const int64_t id = gguf_find_key(gguf_, key.c_str());
    if (id < 0 || gguf_get_kv_type(gguf_, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(gguf_, id) != GGUF_TYPE_STRING) {
        report(key, "missing or invalid string array");
        return false;
    }
    const size_t count = gguf_get_arr_n(gguf_, id);
    value.clear();
    value.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        const char * item = gguf_get_arr_str(gguf_, id, index);
        if (item == nullptr) {
            report(key, "invalid string array");
            return false;
        }
        value.emplace_back(item);
    }
    return true;
}

bool GgufMetadata::has(const std::string & key) const {
    return gguf_find_key(gguf_, key.c_str()) >= 0;
}

bool GgufMetadata::require_string(const std::string & key, const char * expected) const {
    std::string value;
    if (!string(key, value)) {
        return false;
    }
    if (value != expected) {
        std::fprintf(stderr, "%s: metadata %s must be %s\n", family_.c_str(), key.c_str(), expected);
        return false;
    }
    return true;
}

}  // namespace synth
