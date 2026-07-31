// Loads a synthetic on-disk package end to end, then knocks out exactly the
// frontend arrays to prove the refusal branch nothing has ever executed:
// frontend declared present, vocab/merges unreadable -> SYNTH_ERR_GGUF.

#include "arch/omnivoice/omnivoice.h"
#include "omnivoice_synthetic_package.h"
#include "test-assert.h"

#include <cstdio>
#include <memory>
#include <string>

namespace {

std::string package_path(const char * directory, const char * name) {
    return std::string(directory) + "/" + name;
}

int check_valid_package_loads(const char * directory) {
    const std::string path = package_path(directory, "synthetic-valid.gguf");
    SYNTH_TEST_CHECK(synth::omnivoice::testing::write_synthetic_package(path, {}));
    std::unique_ptr<synth::omnivoice::Model> model;
    SYNTH_TEST_CHECK(synth::omnivoice::Model::load_cpu(path, model) == SYNTH_OK);
    SYNTH_TEST_CHECK(model != nullptr);
    synth::omnivoice::ModelInfo info;
    SYNTH_TEST_CHECK(model->get_info(info) == SYNTH_OK);
    SYNTH_TEST_CHECK(info.variant == "synthetic");
    SYNTH_TEST_CHECK(model->samples_per_frame() == 6);
    SYNTH_TEST_CHECK(model->text_frontend() != nullptr);
    return 0;
}

int check_missing_frontend_arrays_are_refused(const char * directory) {
    for (int variant = 0; variant < 2; ++variant) {
        synth::omnivoice::testing::SyntheticPackageOptions options;
        options.omit_frontend_vocab  = variant == 0;
        options.omit_frontend_merges = variant == 1;
        const std::string path = package_path(directory, variant == 0 ? "synthetic-no-vocab.gguf"
                                                                      : "synthetic-no-merges.gguf");
        SYNTH_TEST_CHECK(synth::omnivoice::testing::write_synthetic_package(path, options));
        std::unique_ptr<synth::omnivoice::Model> model;
        // The package declares a frontend it does not carry: the load must fail
        // at the arrays, not fall through to a frontend-less model.
        SYNTH_TEST_CHECK(synth::omnivoice::Model::load_cpu(path, model) == SYNTH_ERR_GGUF);
        SYNTH_TEST_CHECK(model == nullptr);
    }
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <scratch-dir>\n", argv[0]);
        return 2;
    }
    int failures = 0;
    failures += check_valid_package_loads(argv[1]);
    failures += check_missing_frontend_arrays_are_refused(argv[1]);
    return failures == 0 ? 0 : 1;
}
