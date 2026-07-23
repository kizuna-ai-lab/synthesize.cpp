#include "synthesize.h"

#include <stddef.h>
#include <stdint.h>

#define CHECK(condition)     \
    do {                     \
        if (!(condition)) {  \
            return __LINE__; \
        }                    \
    } while (0)

int main(int argc, char ** argv) {
    CHECK(argc == 2);

    CHECK(synth_backend_load_from_dir(NULL) == SYNTH_ERR_INVALID_ARG);
    CHECK(synth_backend_load_from_dir("") == SYNTH_ERR_INVALID_ARG);
    CHECK(synth_backend_load_from_dir("/nonexistent-synthesize-backend-directory") == SYNTH_ERR_FILE_NOT_FOUND);

    CHECK(synth_backend_load_default() == SYNTH_OK);
    CHECK(synth_backend_load_from_dir(argv[1]) == SYNTH_OK);
    CHECK(synth_backend_load_from_dir(argv[1]) == SYNTH_OK);

    synth_model_t * model = (synth_model_t *) (uintptr_t) 1;
    CHECK(synth_model_load("/nonexistent-synthesize-model.gguf", NULL, &model) == SYNTH_ERR_FILE_NOT_FOUND);
    CHECK(model == NULL);

    CHECK(synth_backend_load_from_dir(argv[1]) == SYNTH_ERR_INVALID_ARG);
    CHECK(synth_backend_load_default() == SYNTH_ERR_INVALID_ARG);
    return 0;
}
