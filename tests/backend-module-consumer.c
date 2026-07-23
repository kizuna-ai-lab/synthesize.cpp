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
    CHECK(argc == 3);

    CHECK(synth_backend_load_from_dir(argv[1]) == SYNTH_ERR_BACKEND);
    CHECK(synth_backend_load_from_dir(argv[1]) == SYNTH_ERR_BACKEND);
    CHECK(synth_backend_device_count() == 0u);

    CHECK(synth_backend_load_default() == SYNTH_OK);
    CHECK(synth_backend_load_from_dir(argv[2]) == SYNTH_OK);
    CHECK(synth_backend_device_count() > 0u);
    CHECK(synth_backend_available(SYNTH_BACKEND_CPU) == SYNTH_TRUE);

    synth_model_t * model = (synth_model_t *) (uintptr_t) 1;
    CHECK(synth_model_load("/nonexistent-synthesize-model.gguf", NULL, &model) == SYNTH_ERR_FILE_NOT_FOUND);
    CHECK(model == NULL);
    CHECK(synth_backend_load_default() == SYNTH_ERR_INVALID_ARG);
    CHECK(synth_backend_load_from_dir(argv[2]) == SYNTH_ERR_INVALID_ARG);
    return 0;
}
