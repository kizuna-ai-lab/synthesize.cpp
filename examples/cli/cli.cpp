#include "cli.h"

#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>

namespace synth_cli {
namespace {

bool is_space(char value) {
    return value == ' ' || value == '\t' || value == '\n' || value == '\r';
}

template <typename Integer> bool parse_integer(const std::string & text, Integer & output) {
    if (text.empty()) {
        return false;
    }
    const char * begin = text.data();
    const char * end   = begin + text.size();
    const auto   result = std::from_chars(begin, end, output, 10);
    return result.ec == std::errc{} && result.ptr == end;
}

bool parse_rate(const std::string & text, float & output) {
    if (text.empty()) {
        return false;
    }
    char * end = nullptr;
    errno      = 0;
    const float value = std::strtof(text.c_str(), &end);
    if (errno == ERANGE || end != text.c_str() + text.size() || !std::isfinite(value) || value <= 0.0f) {
        return false;
    }
    output = value;
    return true;
}

bool parse_token_ids(const std::string & text, std::vector<int32_t> & output) {
    output.clear();
    size_t position = 0;
    while (position < text.size()) {
        while (position < text.size() && is_space(text[position])) {
            ++position;
        }
        if (position == text.size()) {
            break;
        }
        const size_t start = position;
        while (position < text.size() && text[position] >= '0' && text[position] <= '9') {
            ++position;
        }
        if (position == start) {
            return false;
        }
        int32_t token = 0;
        const auto result = std::from_chars(text.data() + start, text.data() + position, token, 10);
        if (result.ec != std::errc{} || result.ptr != text.data() + position) {
            return false;
        }
        output.push_back(token);

        const size_t number_end = position;
        while (position < text.size() && is_space(text[position])) {
            ++position;
        }
        const bool separated_by_space = position != number_end;
        if (position == text.size()) {
            break;
        }
        if (text[position] == ',') {
            ++position;
            while (position < text.size() && is_space(text[position])) {
                ++position;
            }
            if (position == text.size() || text[position] == ',') {
                return false;
            }
        } else if (!separated_by_space) {
            return false;
        }
    }
    return !output.empty();
}

bool parse_backend(const std::string & text, synth_backend_request_t & output) {
    if (text == "auto") {
        output = SYNTH_BACKEND_AUTO;
    } else if (text == "cpu") {
        output = SYNTH_BACKEND_CPU;
    } else if (text == "cpu-accel") {
        output = SYNTH_BACKEND_CPU_ACCEL;
    } else if (text == "cuda") {
        output = SYNTH_BACKEND_CUDA;
    } else if (text == "metal") {
        output = SYNTH_BACKEND_METAL;
    } else if (text == "vulkan") {
        output = SYNTH_BACKEND_VULKAN;
    } else {
        return false;
    }
    return true;
}

void write_u16(std::ostream & output, uint16_t value) {
    const unsigned char bytes[] = { static_cast<unsigned char>(value),
                                    static_cast<unsigned char>(value >> 8) };
    output.write(reinterpret_cast<const char *>(bytes), sizeof(bytes));
}

void write_u32(std::ostream & output, uint32_t value) {
    const unsigned char bytes[] = { static_cast<unsigned char>(value),
                                    static_cast<unsigned char>(value >> 8),
                                    static_cast<unsigned char>(value >> 16),
                                    static_cast<unsigned char>(value >> 24) };
    output.write(reinterpret_cast<const char *>(bytes), sizeof(bytes));
}

}  // namespace

bool parse_arguments(int argc, const char * const * argv, Options & output, std::string & error) {
    output = {};
    error.clear();
    if (argc <= 0 || argv == nullptr) {
        error = "missing command line";
        return false;
    }

    int input_count = 0;
    for (int index = 1; index < argc; ++index) {
        if (argv[index] == nullptr) {
            error = "null command-line argument";
            return false;
        }
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            output.show_help = true;
            return true;
        }
        if (index + 1 >= argc || argv[index + 1] == nullptr) {
            error = "missing value for " + argument;
            return false;
        }
        const std::string value = argv[++index];
        if (argument == "--model") {
            output.model_path = value;
        } else if (argument == "--output") {
            output.output_path = value;
        } else if (argument == "--text" || argument == "--phonemes" || argument == "--token-ids") {
            ++input_count;
            if (argument == "--text") {
                output.input_kind       = SYNTH_INPUT_TEXT_UTF8;
                output.linguistic_input = value;
            } else if (argument == "--phonemes") {
                output.input_kind       = SYNTH_INPUT_PHONEMES_UTF8;
                output.linguistic_input = value;
            } else if (!parse_token_ids(value, output.token_ids)) {
                error = "--token-ids must be a non-empty comma- or whitespace-separated list of int32 values";
                return false;
            } else {
                output.input_kind = SYNTH_INPUT_TOKEN_IDS;
            }
        } else if (argument == "--language") {
            output.language_tag = value;
        } else if (argument == "--voice") {
            output.voice_id = value;
        } else if (argument == "--seed") {
            if (value == "random") {
                output.seed = SYNTH_SEED_RANDOM;
            } else if (!parse_integer(value, output.seed) || output.seed == SYNTH_SEED_RANDOM) {
                error = "--seed must be an unsigned 64-bit integer below UINT64_MAX, or random";
                return false;
            }
        } else if (argument == "--rate") {
            if (!parse_rate(value, output.speaking_rate)) {
                error = "--rate must be a finite positive number";
                return false;
            }
        } else if (argument == "--max-output-frames") {
            if (!parse_integer(value, output.max_output_frames)) {
                error = "--max-output-frames must be an unsigned 64-bit integer";
                return false;
            }
        } else if (argument == "--backend") {
            if (!parse_backend(value, output.backend)) {
                error = "--backend must be auto, cpu, cpu-accel, cuda, metal, or vulkan";
                return false;
            }
        } else if (argument == "--device") {
            int64_t device = 0;
            if (!parse_integer(value, device) || device < -1 || device > std::numeric_limits<int32_t>::max()) {
                error = "--device must be -1 or a non-negative int32 value";
                return false;
            }
            output.device_index = static_cast<int32_t>(device);
        } else {
            error = "unknown option: " + argument;
            return false;
        }
    }

    if (output.model_path.empty()) {
        error = "--model is required";
        return false;
    }
    if (output.output_path.empty()) {
        error = "--output is required";
        return false;
    }
    if (input_count != 1 ||
        (output.input_kind != SYNTH_INPUT_TOKEN_IDS && output.linguistic_input.empty())) {
        error = "exactly one non-empty --text, --phonemes, or --token-ids input is required";
        return false;
    }
    return true;
}

const char * usage_text() {
    return
        "Usage: synthesize-cli --model MODEL.gguf --output AUDIO.wav INPUT [OPTIONS]\n"
        "\n"
        "Input (exactly one):\n"
        "  --text UTF8                 Text input\n"
        "  --phonemes UTF8             Phoneme input\n"
        "  --token-ids IDS             Comma- or whitespace-separated int32 token IDs\n"
        "\n"
        "Options:\n"
        "  --language TAG              BCP 47 language tag\n"
        "  --voice ID                  Preset Voice identifier\n"
        "  --seed N|random             Synthesis seed (default: 0)\n"
        "  --rate F                    Speaking-rate multiplier (default: 1.0)\n"
        "  --max-output-frames N       Request output limit (default: model limit)\n"
        "  --backend NAME              auto, cpu, cpu-accel, cuda, metal, or vulkan\n"
        "  --device N                  Device index (default: -1, automatic)\n"
        "  -h, --help                  Show this help\n";
}

bool write_f32_wav(const std::string & path,
                   const float *       samples,
                   uint64_t            frame_count,
                   uint32_t            sample_rate,
                   uint32_t            channel_count,
                   std::string &       error) {
    static_assert(sizeof(float) == 4, "the CLI requires 32-bit float");
    static_assert(std::numeric_limits<float>::is_iec559, "the CLI requires IEEE-754 float");
    error.clear();
    if (path.empty() || sample_rate == 0 || channel_count == 0 ||
        (samples == nullptr && frame_count != 0) || channel_count > UINT16_MAX / sizeof(float) ||
        frame_count > UINT32_MAX || frame_count > UINT32_MAX / channel_count / sizeof(float)) {
        error = "invalid or RIFF-incompatible audio shape";
        return false;
    }
    const uint32_t block_align = channel_count * sizeof(float);
    if (sample_rate > UINT32_MAX / block_align) {
        error = "WAV byte rate exceeds RIFF limits";
        return false;
    }
    const uint32_t data_bytes = static_cast<uint32_t>(frame_count * block_align);
    if (data_bytes > UINT32_MAX - 48) {
        error = "WAV data exceeds RIFF limits";
        return false;
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        error = "could not open output WAV: " + path;
        return false;
    }
    output.write("RIFF", 4);
    write_u32(output, 48 + data_bytes);
    output.write("WAVEfmt ", 8);
    write_u32(output, 16);
    write_u16(output, 3);
    write_u16(output, static_cast<uint16_t>(channel_count));
    write_u32(output, sample_rate);
    write_u32(output, sample_rate * block_align);
    write_u16(output, static_cast<uint16_t>(block_align));
    write_u16(output, 32);
    output.write("fact", 4);
    write_u32(output, 4);
    write_u32(output, static_cast<uint32_t>(frame_count));
    output.write("data", 4);
    write_u32(output, data_bytes);

    const uint64_t sample_count = frame_count * channel_count;
    for (uint64_t index = 0; index < sample_count; ++index) {
        uint32_t bits = 0;
        std::memcpy(&bits, samples + index, sizeof(bits));
        write_u32(output, bits);
    }
    if (!output) {
        error = "failed while writing output WAV: " + path;
        return false;
    }
    return true;
}

}  // namespace synth_cli
