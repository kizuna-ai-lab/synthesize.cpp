// What the load path reports for a package it cannot use.
//
// Every case here is reachable without the real package, which is the point: the
// real-package smoke proves a good package loads, and this proves a bad one is
// refused with the status the public interface promises rather than with
// whatever the first failing read happened to return.

#include "arch/omnivoice/omnivoice.h"
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

    ~FixtureFiles() {
        std::error_code error;
        std::filesystem::remove(malformed, error);
        error.clear();
        std::filesystem::remove(empty_gguf, error);
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

    FixtureFiles fixtures{ fixture_root / "omnivoice-malformed.gguf", fixture_root / "omnivoice-empty.gguf" };
    std::filesystem::remove(fixtures.malformed, error);
    error.clear();
    std::filesystem::remove(fixtures.empty_gguf, error);

    std::unique_ptr<synth::omnivoice::Model> model;
    SYNTH_TEST_CHECK(synth::omnivoice::Model::load_cpu("", model) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(model == nullptr);
    SYNTH_TEST_CHECK(synth::omnivoice::Model::load_cpu((fixture_root / "omnivoice-absent.gguf").string(), model) ==
                     SYNTH_ERR_FILE_NOT_FOUND);
    SYNTH_TEST_CHECK(model == nullptr);

    {
        std::ofstream output(fixtures.malformed, std::ios::binary | std::ios::trunc);
        SYNTH_TEST_CHECK(output.good());
        output << "not a GGUF";
    }
    SYNTH_TEST_CHECK(synth::omnivoice::Model::load_cpu(fixtures.malformed.string(), model) == SYNTH_ERR_GGUF);
    SYNTH_TEST_CHECK(model == nullptr);

    // A well-formed container that declares nothing. The metadata reader is
    // fail-closed, so this is a contract violation rather than a package with
    // defaults.
    gguf_context * empty = gguf_init_empty();
    SYNTH_TEST_CHECK(empty != nullptr);
    const bool written = gguf_write_to_file(empty, fixtures.empty_gguf.string().c_str(), false);
    gguf_free(empty);
    SYNTH_TEST_CHECK(written);
    SYNTH_TEST_CHECK(synth::omnivoice::Model::load_cpu(fixtures.empty_gguf.string(), model) == SYNTH_ERR_GGUF);
    SYNTH_TEST_CHECK(model == nullptr);
    return 0;
}
