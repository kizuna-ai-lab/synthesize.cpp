// Load error mapping for the Kokoro family facade.
//
// This is the first thing a caller meets, so each way a package can be wrong
// has to reach a distinct status rather than a generic failure. The staged
// numerical agreement is a Stage 5 concern and needs the real checkpoint; what
// is checkable without one is the rejection behaviour.

#include "arch/kokoro/kokoro.h"
#include "gguf.h"
#include "test-assert.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

namespace {

struct FixtureFiles {
    std::filesystem::path malformed;
    std::filesystem::path empty_gguf;
    std::filesystem::path wrong_family;

    ~FixtureFiles() {
        std::error_code error;
        std::filesystem::remove(malformed, error);
        error.clear();
        std::filesystem::remove(empty_gguf, error);
        error.clear();
        std::filesystem::remove(wrong_family, error);
    }
};

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        return 2;
    }
    const std::filesystem::path fixture_root = argv[1];
    std::error_code             error;
    std::filesystem::create_directories(fixture_root, error);
    SYNTH_TEST_CHECK(!error);

    FixtureFiles fixtures{ fixture_root / "malformed.gguf", fixture_root / "empty.gguf",
                           fixture_root / "wrong-family.gguf" };

    std::unique_ptr<synth::kokoro::Model> model;
    SYNTH_TEST_CHECK(synth::kokoro::Model::load_cpu("", model) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(model == nullptr);
    SYNTH_TEST_CHECK(synth::kokoro::Model::load_cpu((fixture_root / "absent.gguf").string(), model) ==
                     SYNTH_ERR_FILE_NOT_FOUND);
    SYNTH_TEST_CHECK(model == nullptr);

    {
        std::ofstream output(fixtures.malformed, std::ios::binary | std::ios::trunc);
        SYNTH_TEST_CHECK(output.good());
        output << "not a GGUF";
    }
    SYNTH_TEST_CHECK(synth::kokoro::Model::load_cpu(fixtures.malformed.string(), model) == SYNTH_ERR_GGUF);
    SYNTH_TEST_CHECK(model == nullptr);

    // A well-formed container with no metadata names no architecture, which is
    // a different failure from a corrupt container and gets its own status.
    gguf_context * empty = gguf_init_empty();
    SYNTH_TEST_CHECK(empty != nullptr);
    const bool written = gguf_write_to_file(empty, fixtures.empty_gguf.string().c_str(), false);
    gguf_free(empty);
    SYNTH_TEST_CHECK(written);
    SYNTH_TEST_CHECK(synth::kokoro::Model::load_cpu(fixtures.empty_gguf.string(), model) == SYNTH_ERR_UNSUPPORTED_ARCH);
    SYNTH_TEST_CHECK(model == nullptr);

    // Another family's package must be refused by this facade even though the
    // container itself is valid; the dispatch above it is what routes families,
    // and it needs the architecture status to tell that apart from corruption.
    gguf_context * foreign = gguf_init_empty();
    SYNTH_TEST_CHECK(foreign != nullptr);
    gguf_set_val_str(foreign, "general.architecture", "vits");
    const bool foreign_written = gguf_write_to_file(foreign, fixtures.wrong_family.string().c_str(), false);
    gguf_free(foreign);
    SYNTH_TEST_CHECK(foreign_written);
    SYNTH_TEST_CHECK(synth::kokoro::Model::load_cpu(fixtures.wrong_family.string(), model) ==
                     SYNTH_ERR_UNSUPPORTED_ARCH);
    SYNTH_TEST_CHECK(model == nullptr);
    return 0;
}
