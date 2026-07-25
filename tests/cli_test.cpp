#include "cli.h"

#include "test-assert.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

bool parse(std::initializer_list<const char *> arguments, synth_cli::Options & options, std::string & error) {
    std::vector<const char *> argv(arguments);
    return synth_cli::parse_arguments(static_cast<int>(argv.size()), argv.data(), options, error);
}

uint16_t read_u16(const std::vector<unsigned char> & bytes, size_t offset) {
    return static_cast<uint16_t>(bytes[offset]) | static_cast<uint16_t>(static_cast<uint16_t>(bytes[offset + 1]) << 8);
}

uint32_t read_u32(const std::vector<unsigned char> & bytes, size_t offset) {
    return static_cast<uint32_t>(bytes[offset]) | (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
           (static_cast<uint32_t>(bytes[offset + 2]) << 16) | (static_cast<uint32_t>(bytes[offset + 3]) << 24);
}

}  // namespace

int main(int argc, char ** argv) {
    SYNTH_TEST_CHECK(argc == 2);

    synth_cli::Options options;
    std::string        error;
    SYNTH_TEST_CHECK(parse({ "synthesize-cli",
                             "--model",
                             "model.gguf",
                             "--output",
                             "speech.wav",
                             "--token-ids",
                             "0, 156 0,47",
                             "--language",
                             "en-US",
                             "--voice",
                             "speaker-a",
                             "--seed",
                             "42",
                             "--rate",
                             "1.25",
                             "--max-output-frames",
                             "100",
                             "--backend",
                             "cpu",
                             "--device",
                             "0" },
                           options, error));
    SYNTH_TEST_CHECK(error.empty());
    SYNTH_TEST_CHECK(options.model_path == "model.gguf" && options.output_path == "speech.wav");
    SYNTH_TEST_CHECK(options.input_kind == SYNTH_INPUT_TOKEN_IDS);
    SYNTH_TEST_CHECK(options.token_ids == std::vector<int32_t>({ 0, 156, 0, 47 }));
    SYNTH_TEST_CHECK(options.language_tag == "en-US" && options.voice_id == "speaker-a");
    SYNTH_TEST_CHECK(options.seed == 42 && options.speaking_rate == 1.25f);
    SYNTH_TEST_CHECK(options.max_output_frames == 100);
    SYNTH_TEST_CHECK(options.backend == SYNTH_BACKEND_CPU && options.device_index == 0);

    SYNTH_TEST_CHECK(parse({ "synthesize-cli", "--model", "m", "--output", "o", "--text", "Hello", "--seed", "random",
                             "--backend", "cuda" },
                           options, error));
    SYNTH_TEST_CHECK(options.input_kind == SYNTH_INPUT_TEXT_UTF8 && options.linguistic_input == "Hello");
    SYNTH_TEST_CHECK(options.seed == SYNTH_SEED_RANDOM && options.backend == SYNTH_BACKEND_CUDA);

    SYNTH_TEST_CHECK(parse({ "synthesize-cli", "--help" }, options, error) && options.show_help);
    SYNTH_TEST_CHECK(std::strstr(synth_cli::usage_text(), "--token-ids") != nullptr);

    SYNTH_TEST_CHECK(!parse({ "synthesize-cli", "--model", "m", "--output", "o" }, options, error));
    SYNTH_TEST_CHECK(!error.empty());
    SYNTH_TEST_CHECK(!parse({ "synthesize-cli", "--model", "m", "--output", "o", "--text", "a", "--phonemes", "b" },
                            options, error));
    SYNTH_TEST_CHECK(
        !parse({ "synthesize-cli", "--model", "m", "--output", "o", "--token-ids", "1,,2" }, options, error));
    SYNTH_TEST_CHECK(
        !parse({ "synthesize-cli", "--model", "m", "--output", "o", "--token-ids", "-1" }, options, error));
    SYNTH_TEST_CHECK(
        !parse({ "synthesize-cli", "--model", "m", "--output", "o", "--token-ids", "2147483648" }, options, error));
    SYNTH_TEST_CHECK(
        !parse({ "synthesize-cli", "--model", "m", "--output", "o", "--text", "a", "--seed", "18446744073709551616" },
               options, error));
    SYNTH_TEST_CHECK(
        !parse({ "synthesize-cli", "--model", "m", "--output", "o", "--text", "a", "--rate", "nan" }, options, error));
    SYNTH_TEST_CHECK(
        !parse({ "synthesize-cli", "--model", "m", "--output", "o", "--text", "a", "--device", "-2" }, options, error));
    SYNTH_TEST_CHECK(!parse({ "synthesize-cli", "--model", "m", "--output", "o", "--text", "a", "--backend", "other" },
                            options, error));
    SYNTH_TEST_CHECK(
        !parse({ "synthesize-cli", "--model", "m", "--output", "o", "--text", "a", "--unknown" }, options, error));

    const std::filesystem::path fixture_root = argv[1];
    std::error_code             filesystem_error;
    std::filesystem::create_directories(fixture_root, filesystem_error);
    SYNTH_TEST_CHECK(!filesystem_error);
    const std::filesystem::path wav_path = fixture_root / "cli-f32.wav";
    std::filesystem::remove(wav_path, filesystem_error);

    const std::array<float, 2> pcm = { -0.5f, 0.25f };
    SYNTH_TEST_CHECK(synth_cli::write_f32_wav(wav_path.string(), pcm.data(), 2, 22050, 1, error));
    std::ifstream                    input(wav_path, std::ios::binary);
    const std::vector<unsigned char> bytes{ std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>() };
    SYNTH_TEST_CHECK(bytes.size() == 64);
    SYNTH_TEST_CHECK(std::memcmp(bytes.data(), "RIFF", 4) == 0 && read_u32(bytes, 4) == 56);
    SYNTH_TEST_CHECK(std::memcmp(bytes.data() + 8, "WAVEfmt ", 8) == 0 && read_u32(bytes, 16) == 16);
    SYNTH_TEST_CHECK(read_u16(bytes, 20) == 3 && read_u16(bytes, 22) == 1);
    SYNTH_TEST_CHECK(read_u32(bytes, 24) == 22050 && read_u32(bytes, 28) == 88200);
    SYNTH_TEST_CHECK(read_u16(bytes, 32) == 4 && read_u16(bytes, 34) == 32);
    SYNTH_TEST_CHECK(std::memcmp(bytes.data() + 36, "fact", 4) == 0 && read_u32(bytes, 40) == 4);
    SYNTH_TEST_CHECK(read_u32(bytes, 44) == 2);
    SYNTH_TEST_CHECK(std::memcmp(bytes.data() + 48, "data", 4) == 0 && read_u32(bytes, 52) == 8);
    float decoded[2]{};
    std::memcpy(decoded, bytes.data() + 56, sizeof(decoded));
    SYNTH_TEST_CHECK(decoded[0] == pcm[0] && decoded[1] == pcm[1]);

    SYNTH_TEST_CHECK(!synth_cli::write_f32_wav(wav_path.string(), nullptr, 2, 22050, 1, error));
    SYNTH_TEST_CHECK(!synth_cli::write_f32_wav(wav_path.string(), pcm.data(), 2, 0, 1, error));
    SYNTH_TEST_CHECK(!synth_cli::write_f32_wav(wav_path.string(), reinterpret_cast<const float *>(uintptr_t(1)),
                                               UINT64_MAX, 22050, 1, error));
    std::filesystem::remove(wav_path, filesystem_error);
    return 0;
}
