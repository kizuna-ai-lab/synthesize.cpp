#include "synthesize.h"

#include <inttypes.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#define GUARD_BYTE 0xa5u
#define GUARD_SIZE 16u

#define DEFINE_INITIALIZER_BOUNDS_TEST(test_name, type, initializer)                                                 \
    static int test_name(void) {                                                                                     \
        struct guarded_value {                                                                                       \
            type          value;                                                                                     \
            unsigned char guard[GUARD_SIZE];                                                                         \
        } guarded;                                                                                                   \
                                                                                                                     \
        initializer(NULL, sizeof(type));                                                                             \
        for (uint64_t requested = 0; requested <= (uint64_t) sizeof(type) + GUARD_SIZE; ++requested) {               \
            memset(&guarded, GUARD_BYTE, sizeof(guarded));                                                           \
            initializer(&guarded.value, requested);                                                                  \
                                                                                                                     \
            const size_t          visible = requested < sizeof(type) ? (size_t) requested : sizeof(type);            \
            const unsigned char * bytes   = (const unsigned char *) &guarded.value;                                  \
            for (size_t i = visible; i < sizeof(type) + GUARD_SIZE; ++i) {                                           \
                if (bytes[i] != GUARD_BYTE) {                                                                        \
                    fprintf(stderr,                                                                                  \
                            #initializer                                                                             \
                            " wrote byte %zu past its declared/known prefix "                                        \
                            "(requested=%" PRIu64 ", sizeof=%zu)\n",                                                 \
                            i, requested, sizeof(type));                                                             \
                    return 1;                                                                                        \
                }                                                                                                    \
            }                                                                                                        \
            if (requested >= sizeof(uint64_t) && guarded.value.struct_size != requested) {                           \
                fprintf(stderr, #initializer " recorded struct_size=%" PRIu64 " instead of requested=%" PRIu64 "\n", \
                        guarded.value.struct_size, requested);                                                       \
                return 1;                                                                                            \
            }                                                                                                        \
        }                                                                                                            \
        return 0;                                                                                                    \
    }

DEFINE_INITIALIZER_BOUNDS_TEST(test_version, synth_version_t, synth_version_init)
DEFINE_INITIALIZER_BOUNDS_TEST(test_diagnostic_sink, synth_diagnostic_sink_t, synth_diagnostic_sink_init)
DEFINE_INITIALIZER_BOUNDS_TEST(test_backend_device, synth_backend_device_t, synth_backend_device_init)
DEFINE_INITIALIZER_BOUNDS_TEST(test_model_load_params, synth_model_load_params_t, synth_model_load_params_init)
DEFINE_INITIALIZER_BOUNDS_TEST(test_model_capabilities, synth_model_capabilities_t, synth_model_capabilities_init)
DEFINE_INITIALIZER_BOUNDS_TEST(test_preset_voice, synth_preset_voice_t, synth_preset_voice_init)
DEFINE_INITIALIZER_BOUNDS_TEST(test_language_capability, synth_language_capability_t, synth_language_capability_init)
DEFINE_INITIALIZER_BOUNDS_TEST(test_voice_profile_capabilities,
                               synth_voice_profile_capabilities_t,
                               synth_voice_profile_capabilities_init)
DEFINE_INITIALIZER_BOUNDS_TEST(test_voice_reference, synth_voice_reference_t, synth_voice_reference_init)
DEFINE_INITIALIZER_BOUNDS_TEST(test_voice_reference_params,
                               synth_voice_reference_params_t,
                               synth_voice_reference_params_init)
DEFINE_INITIALIZER_BOUNDS_TEST(test_voice_description_params,
                               synth_voice_description_params_t,
                               synth_voice_description_params_init)
DEFINE_INITIALIZER_BOUNDS_TEST(test_voice_random_params, synth_voice_random_params_t, synth_voice_random_params_init)
DEFINE_INITIALIZER_BOUNDS_TEST(test_voice_profile_load_params,
                               synth_voice_profile_load_params_t,
                               synth_voice_profile_load_params_init)
DEFINE_INITIALIZER_BOUNDS_TEST(test_voice_profile_serialize_params,
                               synth_voice_profile_serialize_params_t,
                               synth_voice_profile_serialize_params_init)
DEFINE_INITIALIZER_BOUNDS_TEST(test_request, synth_request_t, synth_request_init)
DEFINE_INITIALIZER_BOUNDS_TEST(test_audio_sink, synth_audio_sink_t, synth_audio_sink_init)
DEFINE_INITIALIZER_BOUNDS_TEST(test_result, synth_result_t, synth_result_init)

int main(void) {
    return test_version() || test_diagnostic_sink() || test_backend_device() || test_model_load_params() ||
           test_model_capabilities() || test_preset_voice() || test_language_capability() ||
           test_voice_profile_capabilities() || test_voice_reference() || test_voice_reference_params() ||
           test_voice_description_params() || test_voice_random_params() || test_voice_profile_load_params() ||
           test_voice_profile_serialize_params() || test_request() || test_audio_sink() || test_result();
}
