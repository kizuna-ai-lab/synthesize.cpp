#include "synthesis-request.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <new>

namespace synth {

namespace {

template <typename V> V read_field(const synth_request_t * request, size_t offset, V default_value) {
    if (request->struct_size < offset + sizeof(V)) {
        return default_value;
    }
    V value;
    std::memcpy(&value, reinterpret_cast<const unsigned char *>(request) + offset, sizeof(value));
    return value;
}

bool ascii_alpha(char value) {
    const unsigned char character = static_cast<unsigned char>(value);
    return std::isalpha(character) != 0 && character < 128;
}

bool ascii_digit(char value) {
    const unsigned char character = static_cast<unsigned char>(value);
    return std::isdigit(character) != 0 && character < 128;
}

bool ascii_alnum(char value) {
    return ascii_alpha(value) || ascii_digit(value);
}

bool equals_ascii_case(const char * value, size_t size, const char * expected) {
    if (std::strlen(expected) != size) {
        return false;
    }
    for (size_t index = 0; index < size; ++index) {
        const unsigned char left  = static_cast<unsigned char>(value[index]);
        const unsigned char right = static_cast<unsigned char>(expected[index]);
        if (std::tolower(left) != std::tolower(right)) {
            return false;
        }
    }
    return true;
}

bool valid_bcp47_shape(const char * value, size_t size) {
    if (value == nullptr || size < 2 || size > 63) {
        return false;
    }
    size_t subtag_start = 0;
    size_t subtag_count = 0;
    for (size_t index = 0; index <= size; ++index) {
        if (index != size && value[index] != '-') {
            if (!ascii_alnum(value[index])) {
                return false;
            }
            continue;
        }
        const size_t subtag_size = index - subtag_start;
        if (subtag_size == 0 || subtag_size > 8) {
            return false;
        }
        if (subtag_count == 0) {
            if (subtag_size < 2) {
                return false;
            }
            for (size_t item = subtag_start; item < index; ++item) {
                if (!ascii_alpha(value[item])) {
                    return false;
                }
            }
        }
        ++subtag_count;
        subtag_start = index + 1;
    }
    return true;
}

bool supported_english_tag(const char * value, size_t size) {
    if (equals_ascii_case(value, size, "en")) {
        return true;
    }
    if ((size != 5 && size != 6) || !equals_ascii_case(value, 2, "en") || value[2] != '-') {
        return false;
    }
    if (size == 5) {
        return ascii_alpha(value[3]) && ascii_alpha(value[4]);
    }
    return ascii_digit(value[3]) && ascii_digit(value[4]) && ascii_digit(value[5]);
}

}  // namespace

synth_status_t prepare_synthesis_request(const ModelInfo &          info,
                                         const synth_request_t *    request,
                                         PreparedSynthesisRequest & output) {
    output = {};
    if (request == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (request->struct_size < offsetof(synth_request_t, input_count) + sizeof(request->input_count)) {
        return SYNTH_ERR_BAD_STRUCT_SIZE;
    }
    const synth_input_kind_t input_kind = request->input_kind;
    if (input_kind != SYNTH_INPUT_TEXT_UTF8 && input_kind != SYNTH_INPUT_PHONEMES_UTF8 &&
        input_kind != SYNTH_INPUT_TOKEN_IDS) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (request->input_data == nullptr || request->input_count == 0 ||
        request->input_count > std::numeric_limits<size_t>::max()) {
        return SYNTH_ERR_INVALID_ARG;
    }
    synth_input_flags_t required_input_flag = SYNTH_INPUT_SUPPORT_TOKEN_IDS;
    if (input_kind == SYNTH_INPUT_TEXT_UTF8) {
        required_input_flag = SYNTH_INPUT_SUPPORT_TEXT_UTF8;
    } else if (input_kind == SYNTH_INPUT_PHONEMES_UTF8) {
        required_input_flag = SYNTH_INPUT_SUPPORT_PHONEMES_UTF8;
    }
    if ((info.input_flags & required_input_flag) == 0) {
        return SYNTH_ERR_UNSUPPORTED_INPUT;
    }
    if (input_kind == SYNTH_INPUT_TOKEN_IDS && request->input_count > info.max_input_tokens) {
        return SYNTH_ERR_INPUT_TOO_LONG;
    }
    if (input_kind != SYNTH_INPUT_TOKEN_IDS && info.text_frontend == nullptr) {
        return SYNTH_ERR_TEXT_FRONTEND;
    }

    const char * language_tag =
        read_field(request, offsetof(synth_request_t, language_tag), static_cast<const char *>(nullptr));
    const uint64_t language_size = read_field(request, offsetof(synth_request_t, language_tag_size), uint64_t(0));
    if ((language_tag == nullptr) != (language_size == 0)) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (language_tag != nullptr) {
        if (language_size > std::numeric_limits<size_t>::max() ||
            !valid_bcp47_shape(language_tag, static_cast<size_t>(language_size))) {
            return SYNTH_ERR_INVALID_ARG;
        }
        if (!supported_english_tag(language_tag, static_cast<size_t>(language_size))) {
            return SYNTH_ERR_UNSUPPORTED_LANGUAGE;
        }
    }

    const char * voice_id =
        read_field(request, offsetof(synth_request_t, voice_id), static_cast<const char *>(nullptr));
    const uint64_t voice_size             = read_field(request, offsetof(synth_request_t, voice_id_size), uint64_t(0));
    const synth_voice_profile_t * profile = read_field(request, offsetof(synth_request_t, voice_profile),
                                                       static_cast<const synth_voice_profile_t *>(nullptr));
    if ((voice_id == nullptr) != (voice_size == 0) || (voice_id != nullptr && profile != nullptr)) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (profile != nullptr) {
        return SYNTH_ERR_UNSUPPORTED_VOICE;
    }
    uint32_t speaker_index = UINT32_MAX;
    if (voice_id != nullptr) {
        if (voice_size > std::numeric_limits<size_t>::max()) {
            return SYNTH_ERR_INVALID_ARG;
        }
        for (size_t index = 0; index < info.preset_voice_ids.size(); ++index) {
            const std::string & candidate = info.preset_voice_ids[index];
            if (candidate.size() == voice_size && std::memcmp(candidate.data(), voice_id, candidate.size()) == 0) {
                speaker_index = static_cast<uint32_t>(index);
                break;
            }
        }
        if (speaker_index == UINT32_MAX) {
            return SYNTH_ERR_UNSUPPORTED_VOICE;
        }
    } else if (!info.preset_voice_ids.empty() && !info.has_package_default) {
        return SYNTH_ERR_UNSUPPORTED_VOICE;
    }

    const float speaking_rate = read_field(request, offsetof(synth_request_t, speaking_rate), 1.0f);
    if (!std::isfinite(speaking_rate) || speaking_rate < info.min_speaking_rate ||
        speaking_rate > info.max_speaking_rate) {
        return SYNTH_ERR_INVALID_ARG;
    }

    try {
        if (input_kind == SYNTH_INPUT_TOKEN_IDS) {
            const auto * tokens = static_cast<const int32_t *>(request->input_data);
            output.token_ids.assign(tokens, tokens + static_cast<size_t>(request->input_count));
        } else {
            const synth_status_t frontend_status = info.text_frontend->prepare(
                input_kind, request->input_data, request->input_count, info.max_input_tokens, output.token_ids);
            if (frontend_status != SYNTH_OK) {
                output = {};
                return frontend_status;
            }
        }
        for (int32_t token : output.token_ids) {
            if (token < 0 || static_cast<uint32_t>(token) >= info.vocab_size) {
                output = {};
                return SYNTH_ERR_INVALID_ARG;
            }
        }
    } catch (const std::bad_alloc &) {
        output = {};
        return SYNTH_ERR_OOM;
    }

    output.seed                  = read_field(request, offsetof(synth_request_t, seed), uint64_t(0));
    output.speaking_rate         = speaking_rate;
    const uint64_t request_limit = read_field(request, offsetof(synth_request_t, max_output_frames), uint64_t(0));
    output.effective_frame_limit =
        request_limit == 0 ? info.max_output_frames : std::min(request_limit, info.max_output_frames);
    output.should_cancel =
        read_field(request, offsetof(synth_request_t, should_cancel), static_cast<synth_cancel_callback_t>(nullptr));
    output.cancel_user_data =
        read_field(request, offsetof(synth_request_t, cancel_user_data), static_cast<void *>(nullptr));
    output.diagnostics            = read_field(request, offsetof(synth_request_t, diagnostics),
                                               static_cast<const synth_diagnostic_sink_t *>(nullptr));
    output.resolved_language_tag  = "en";
    output.resolved_language_size = 2;
    output.speaker_index          = speaker_index;
    if (speaker_index != UINT32_MAX) {
        const std::string & resolved = info.preset_voice_ids[speaker_index];
        output.resolved_voice_id     = resolved.c_str();
        output.resolved_voice_size   = resolved.size();
    }
    return SYNTH_OK;
}

}  // namespace synth
