#include "quantize.h"

#include <cstdio>
#include <string>

namespace {

void print_usage(const char * executable) {
    std::fprintf(stderr,
                 "usage: %s INPUT.gguf OUTPUT.gguf --quant F16|Q8_MIXED|Q5_K_MIXED|Q8_GEN\n"
                 "\n"
                 "The output path must not already exist.\n",
                 executable);
}

}  // namespace

int main(int argc, char ** argv) {
    const char * input   = nullptr;
    const char * output  = nullptr;
    const char * profile = nullptr;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        if (argument == "--quant" || argument == "-q") {
            if (++index >= argc) {
                print_usage(argv[0]);
                return 2;
            }
            profile = argv[index];
        } else if (input == nullptr) {
            input = argv[index];
        } else if (output == nullptr) {
            output = argv[index];
        } else {
            std::fprintf(stderr, "synthesize-quantize: unexpected argument: %s\n", argv[index]);
            print_usage(argv[0]);
            return 2;
        }
    }
    if (input == nullptr || output == nullptr || profile == nullptr) {
        print_usage(argv[0]);
        return 2;
    }

    std::string error;
    if (!synth::quantize::quantize_file(input, output, profile, error)) {
        std::fprintf(stderr, "synthesize-quantize: %s\n", error.c_str());
        return 1;
    }
    std::printf("synthesize-quantize: wrote %s (%s)\n", output, profile);
    return 0;
}
