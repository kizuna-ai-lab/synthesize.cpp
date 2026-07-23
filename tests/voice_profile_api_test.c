#include "synthesize.h"

#include <stdint.h>
#include <string.h>

#define CHECK(condition)     \
    do {                     \
        if (!(condition)) {  \
            return __LINE__; \
        }                    \
    } while (0)

int main(void) {
    CHECK(SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO == 1u);
    CHECK(SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT == 2u);
    CHECK(SYNTH_PROFILE_SOURCE_RANDOM_SEED == 4u);
    CHECK(SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE == 8u);
    CHECK(SYNTH_REQUIREMENT_UNSUPPORTED == 0u);
    CHECK(SYNTH_REQUIREMENT_OPTIONAL == 1u);
    CHECK(SYNTH_REQUIREMENT_REQUIRED == 2u);
    CHECK(SYNTH_REFERENCE_SAMPLE_RATE_MIN == 8000u);
    CHECK(SYNTH_REFERENCE_SAMPLE_RATE_MAX == 192000u);
    CHECK(SYNTH_REFERENCE_CHANNELS_MAX == 2u);

    synth_voice_profile_capabilities_t capabilities;
    memset(&capabilities, 0xa5, sizeof(capabilities));
    synth_voice_profile_capabilities_init(&capabilities, sizeof(capabilities));
    CHECK(capabilities.struct_size == sizeof(capabilities));
    CHECK(capabilities.source_flags == 0u);
    CHECK(capabilities.reference_transcript == SYNTH_REQUIREMENT_UNSUPPORTED);
    CHECK(capabilities.reference_language == SYNTH_REQUIREMENT_UNSUPPORTED);
    CHECK(capabilities.description_language == SYNTH_REQUIREMENT_UNSUPPORTED);
    CHECK(capabilities.max_reference_count == 0u);
    CHECK(capabilities.profile_schema == NULL && capabilities.profile_schema_size == 0u);
    CHECK(capabilities.profile_schema_version == 0u);
    for (size_t index = 0; index < sizeof(capabilities.profile_compatibility_id); ++index) {
        CHECK(capabilities.profile_compatibility_id[index] == 0u);
    }
    CHECK(capabilities.reference_target_sample_rate == 0u);
    CHECK(capabilities.reference_target_channel_count == 0u);
    CHECK(capabilities.min_reference_frames_per_clip == 0u);
    CHECK(capabilities.max_reference_frames_per_clip == 0u);
    CHECK(capabilities.max_reference_total_frames == 0u);
    CHECK(synth_model_get_voice_profile_capabilities(NULL, &capabilities) == SYNTH_ERR_INVALID_ARG);

    struct guarded_capabilities {
        synth_voice_profile_capabilities_t value;
        unsigned char                      guard[16];
    } guarded;

    memset(&guarded, 0xa5, sizeof(guarded));
    synth_voice_profile_capabilities_init(&guarded.value, sizeof(guarded));
    CHECK(guarded.value.struct_size == sizeof(guarded));
    for (size_t index = 0; index < sizeof(guarded.guard); ++index) {
        CHECK(guarded.guard[index] == 0xa5);
    }

    memset(&capabilities, 0xa5, sizeof(capabilities));
    synth_voice_profile_capabilities_init(&capabilities, sizeof(uint32_t));
    for (size_t index = 0; index < sizeof(capabilities); ++index) {
        CHECK(((const unsigned char *) &capabilities)[index] == (index < sizeof(uint32_t) ? 0x00 : 0xa5));
    }

    synth_voice_reference_t reference;
    memset(&reference, 0xa5, sizeof(reference));
    synth_voice_reference_init(&reference, sizeof(reference));
    CHECK(reference.struct_size == sizeof(reference));
    CHECK(reference.samples == NULL && reference.frame_count == 0u);
    CHECK(reference.sample_rate == 0u && reference.channel_count == 0u);
    CHECK(reference.transcript == NULL && reference.transcript_size == 0u);
    CHECK(reference.language_tag == NULL && reference.language_tag_size == 0u);

    synth_voice_reference_params_t reference_params;
    synth_voice_reference_params_init(&reference_params, sizeof(reference_params));
    CHECK(reference_params.struct_size == sizeof(reference_params));
    CHECK(reference_params.references == NULL && reference_params.reference_count == 0u);
    CHECK(reference_params.reference_stride == 0u && reference_params.diagnostics == NULL);

    synth_voice_description_params_t description;
    synth_voice_description_params_init(&description, sizeof(description));
    CHECK(description.struct_size == sizeof(description));
    CHECK(description.description == NULL && description.description_size == 0u);
    CHECK(description.language_tag == NULL && description.language_tag_size == 0u);
    CHECK(description.seed == 0u && description.diagnostics == NULL);

    synth_voice_random_params_t random;
    synth_voice_random_params_init(&random, sizeof(random));
    CHECK(random.struct_size == sizeof(random));
    CHECK(random.seed == 0u && random.diagnostics == NULL);

    synth_voice_profile_load_params_t load;
    synth_voice_profile_load_params_init(&load, sizeof(load));
    CHECK(load.struct_size == sizeof(load));
    CHECK(load.data == NULL && load.data_size == 0u && load.diagnostics == NULL);

    synth_voice_profile_serialize_params_t serialize;
    synth_voice_profile_serialize_params_init(&serialize, sizeof(serialize));
    CHECK(serialize.struct_size == sizeof(serialize) && serialize.diagnostics == NULL);

    synth_voice_profile_t * profile = (synth_voice_profile_t *) (uintptr_t) 1;
    synth_byte_buffer_t *   buffer  = (synth_byte_buffer_t *) (uintptr_t) 1;

    synth_voice_profile_load_params_init(&load, sizeof(uint32_t));
    profile = (synth_voice_profile_t *) (uintptr_t) 1;
    CHECK(synth_voice_profile_load_from_memory((const synth_model_t *) 1, &load, &profile) ==
          SYNTH_ERR_BAD_STRUCT_SIZE);
    CHECK(profile == NULL);

    synth_voice_profile_serialize_params_init(&serialize, sizeof(uint32_t));
    profile = (synth_voice_profile_t *) (uintptr_t) 1;
    CHECK(synth_voice_profile_serialize(profile, &serialize, &buffer) == SYNTH_ERR_BAD_STRUCT_SIZE);
    CHECK(buffer == NULL);

    CHECK(synth_voice_profile_create_from_reference(NULL, &reference_params, &profile) == SYNTH_ERR_INVALID_ARG);
    CHECK(profile == NULL);
    profile = (synth_voice_profile_t *) (uintptr_t) 1;
    CHECK(synth_voice_profile_create_from_description(NULL, &description, &profile) == SYNTH_ERR_INVALID_ARG);
    CHECK(profile == NULL);
    profile = (synth_voice_profile_t *) (uintptr_t) 1;
    CHECK(synth_voice_profile_create_random(NULL, &random, &profile) == SYNTH_ERR_INVALID_ARG);
    CHECK(profile == NULL);
    profile = (synth_voice_profile_t *) (uintptr_t) 1;
    CHECK(synth_voice_profile_load_from_memory(NULL, &load, &profile) == SYNTH_ERR_INVALID_ARG);
    CHECK(profile == NULL);

    CHECK(synth_voice_profile_serialize(NULL, NULL, &buffer) == SYNTH_ERR_INVALID_ARG);
    CHECK(buffer == NULL);
    synth_byte_buffer_free(NULL);
    synth_voice_profile_free(NULL);
    return 0;
}
