#include "synthesize.h"
#include "test-assert.h"

#include <cstring>

namespace {

struct DiagnosticCapture {
    int            count  = 0;
    synth_status_t status = SYNTH_OK;
};

void SYNTH_CALL capture_diagnostic(void * user_data, const synth_diagnostic_t * diagnostic) {
    auto * capture = static_cast<DiagnosticCapture *>(user_data);
    ++capture->count;
    capture->status = diagnostic->status;
}

void SYNTH_CALL throw_diagnostic(void *, const synth_diagnostic_t *) {
    throw 1;
}

}  // namespace

int main(int argc, char ** argv) {
    SYNTH_TEST_CHECK(argc == 2);

    DiagnosticCapture       capture;
    synth_diagnostic_sink_t diagnostic;
    synth_diagnostic_sink_init(&diagnostic, sizeof(diagnostic));
    diagnostic.emit      = capture_diagnostic;
    diagnostic.user_data = &capture;

    synth_model_load_params_t params;
    synth_model_load_params_init(&params, sizeof(params));
    params.backend     = SYNTH_BACKEND_CPU;
    params.diagnostics = &diagnostic;

    uint32_t               cpu_device_index = UINT32_MAX;
    synth_backend_device_t enumerated_device;
    for (uint32_t i = 0; i < synth_backend_device_count(); ++i) {
        synth_backend_device_init(&enumerated_device, sizeof(enumerated_device));
        SYNTH_TEST_CHECK(synth_backend_device_get(i, &enumerated_device) == SYNTH_OK);
        if (enumerated_device.device_type == SYNTH_DEVICE_TYPE_CPU) {
            cpu_device_index = i;
            break;
        }
    }
    SYNTH_TEST_CHECK(cpu_device_index != UINT32_MAX);
    params.device_index = static_cast<int32_t>(cpu_device_index);

    synth_model_t * model = nullptr;
    SYNTH_TEST_CHECK(synth_model_load(argv[1], &params, &model) == SYNTH_OK);
    SYNTH_TEST_CHECK(model != nullptr && capture.count == 0);

    synth_backend_device_t actual_device;
    synth_backend_device_init(&actual_device, sizeof(actual_device));
    SYNTH_TEST_CHECK(synth_model_get_device(model, &actual_device) == SYNTH_OK);
    SYNTH_TEST_CHECK(std::strcmp(actual_device.name, enumerated_device.name) == 0);
    SYNTH_TEST_CHECK(std::strcmp(actual_device.kind, enumerated_device.kind) == 0);
    SYNTH_TEST_CHECK(actual_device.device_type == SYNTH_DEVICE_TYPE_CPU);
    SYNTH_TEST_CHECK(actual_device.flags == enumerated_device.flags);
    SYNTH_TEST_CHECK((actual_device.device_id == nullptr && enumerated_device.device_id == nullptr) ||
                     (actual_device.device_id != nullptr && enumerated_device.device_id != nullptr &&
                      std::strcmp(actual_device.device_id, enumerated_device.device_id) == 0));

    synth_backend_device_init(&actual_device, sizeof(uint32_t));
    SYNTH_TEST_CHECK(synth_model_get_device(model, &actual_device) == SYNTH_ERR_BAD_STRUCT_SIZE);
    SYNTH_TEST_CHECK(synth_model_get_device(model, nullptr) == SYNTH_ERR_INVALID_ARG);

    synth_model_capabilities_t capabilities;
    synth_model_capabilities_init(&capabilities, sizeof(capabilities));
    SYNTH_TEST_CHECK(synth_model_get_capabilities(model, &capabilities) == SYNTH_OK);
    SYNTH_TEST_CHECK(capabilities.input_flags == (SYNTH_INPUT_SUPPORT_PHONEMES_UTF8 | SYNTH_INPUT_SUPPORT_TOKEN_IDS));
    SYNTH_TEST_CHECK(capabilities.capability_flags ==
                     (SYNTH_MODEL_CAPABILITY_SPEAKING_RATE | SYNTH_MODEL_CAPABILITY_STOCHASTIC));
    SYNTH_TEST_CHECK(capabilities.output_sample_rate == 22050 && capabilities.output_channel_count == 1);
    SYNTH_TEST_CHECK(capabilities.min_speaking_rate == 0.8f && capabilities.max_speaking_rate == 1.25f);
    SYNTH_TEST_CHECK(capabilities.max_input_tokens == 512 && capabilities.max_output_frames == 1323000);

    uint64_t count = 99;
    SYNTH_TEST_CHECK(synth_model_get_preset_voice_count(model, &count) == SYNTH_OK && count == 0);
    SYNTH_TEST_CHECK(synth_model_get_language_count(model, &count) == SYNTH_OK && count == 1);
    synth_language_capability_t language;
    synth_language_capability_init(&language, sizeof(language));
    SYNTH_TEST_CHECK(synth_model_get_language(model, 0, &language) == SYNTH_OK);
    SYNTH_TEST_CHECK(language.tag_size == 2 && std::memcmp(language.tag, "en", 2) == 0);
    SYNTH_TEST_CHECK(language.flags == (SYNTH_LANGUAGE_DEFAULT | SYNTH_LANGUAGE_REGIONAL_FALLBACK));
    SYNTH_TEST_CHECK(synth_model_get_language(model, 1, &language) == SYNTH_ERR_INVALID_ARG);

    synth_voice_profile_capabilities_t profile_capabilities;
    synth_voice_profile_capabilities_init(&profile_capabilities, sizeof(profile_capabilities));
    SYNTH_TEST_CHECK(synth_model_get_voice_profile_capabilities(model, &profile_capabilities) == SYNTH_OK);
    SYNTH_TEST_CHECK(profile_capabilities.source_flags == 0);
    SYNTH_TEST_CHECK(profile_capabilities.reference_transcript == SYNTH_REQUIREMENT_UNSUPPORTED);
    SYNTH_TEST_CHECK(profile_capabilities.reference_language == SYNTH_REQUIREMENT_UNSUPPORTED);
    SYNTH_TEST_CHECK(profile_capabilities.description_language == SYNTH_REQUIREMENT_UNSUPPORTED);
    SYNTH_TEST_CHECK(profile_capabilities.max_reference_count == 0);
    SYNTH_TEST_CHECK(profile_capabilities.profile_schema == nullptr && profile_capabilities.profile_schema_size == 0 &&
                     profile_capabilities.profile_schema_version == 0);
    synth_voice_profile_capabilities_init(&profile_capabilities, sizeof(uint32_t));
    SYNTH_TEST_CHECK(synth_model_get_voice_profile_capabilities(model, &profile_capabilities) ==
                     SYNTH_ERR_BAD_STRUCT_SIZE);

    synth_voice_profile_t * profile          = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
    float                   reference_sample = 0.0f;
    synth_voice_reference_t reference;
    synth_voice_reference_init(&reference, sizeof(reference));
    reference.samples       = &reference_sample;
    reference.frame_count   = 1;
    reference.sample_rate   = 22050;
    reference.channel_count = 1;
    synth_voice_reference_params_t reference_params;
    synth_voice_reference_params_init(&reference_params, sizeof(reference_params));
    reference_params.references       = &reference;
    reference_params.reference_count  = 1;
    reference_params.reference_stride = sizeof(reference);
    SYNTH_TEST_CHECK(synth_voice_profile_create_from_reference(model, &reference_params, &profile) ==
                         SYNTH_ERR_UNSUPPORTED_VOICE &&
                     profile == nullptr);
    reference_params.struct_size = sizeof(uint32_t);
    profile                      = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
    SYNTH_TEST_CHECK(synth_voice_profile_create_from_reference(model, &reference_params, &profile) ==
                         SYNTH_ERR_BAD_STRUCT_SIZE &&
                     profile == nullptr);

    synth_voice_description_params_t description_params;
    synth_voice_description_params_init(&description_params, sizeof(description_params));
    description_params.description      = "warm voice";
    description_params.description_size = 10;
    profile                             = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
    SYNTH_TEST_CHECK(synth_voice_profile_create_from_description(model, &description_params, &profile) ==
                         SYNTH_ERR_UNSUPPORTED_VOICE &&
                     profile == nullptr);

    synth_voice_random_params_t random_params;
    synth_voice_random_params_init(&random_params, sizeof(random_params));
    random_params.seed = 42;
    profile            = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
    SYNTH_TEST_CHECK(synth_voice_profile_create_random(model, &random_params, &profile) ==
                         SYNTH_ERR_UNSUPPORTED_VOICE &&
                     profile == nullptr);

    const uint8_t                     serialized_data[] = { 0x47, 0x47, 0x55, 0x46 };
    synth_voice_profile_load_params_t profile_load_params;
    synth_voice_profile_load_params_init(&profile_load_params, sizeof(profile_load_params));
    profile_load_params.data      = serialized_data;
    profile_load_params.data_size = sizeof(serialized_data);
    profile                       = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
    SYNTH_TEST_CHECK(synth_voice_profile_load_from_memory(model, &profile_load_params, &profile) ==
                         SYNTH_ERR_UNSUPPORTED_VOICE &&
                     profile == nullptr);

    synth_context_t * first  = nullptr;
    synth_context_t * second = nullptr;
    SYNTH_TEST_CHECK(synth_context_create(model, &first) == SYNTH_OK && first != nullptr);
    SYNTH_TEST_CHECK(synth_context_create(model, &second) == SYNTH_OK && second != nullptr && second != first);
    synth_context_free(second);
    synth_context_free(first);
    synth_model_free(model);

    params.backend      = SYNTH_BACKEND_CUDA;
    params.device_index = -1;
    if (synth_backend_available(SYNTH_BACKEND_CUDA) == SYNTH_TRUE) {
        uint32_t               cuda_device_index = UINT32_MAX;
        synth_backend_device_t cuda_device;
        for (uint32_t i = 0; i < synth_backend_device_count(); ++i) {
            synth_backend_device_init(&cuda_device, sizeof(cuda_device));
            SYNTH_TEST_CHECK(synth_backend_device_get(i, &cuda_device) == SYNTH_OK);
            if (std::strcmp(cuda_device.kind, "cuda") == 0) {
                cuda_device_index = i;
                break;
            }
        }
        SYNTH_TEST_CHECK(cuda_device_index != UINT32_MAX);
        params.device_index = static_cast<int32_t>(cuda_device_index);
        SYNTH_TEST_CHECK(synth_model_load(argv[1], &params, &model) == SYNTH_OK && model != nullptr);
        synth_backend_device_init(&actual_device, sizeof(actual_device));
        SYNTH_TEST_CHECK(synth_model_get_device(model, &actual_device) == SYNTH_OK);
        SYNTH_TEST_CHECK(std::strcmp(actual_device.kind, "cuda") == 0);
        SYNTH_TEST_CHECK(std::strcmp(actual_device.name, cuda_device.name) == 0);
        synth_model_free(model);
    } else {
        SYNTH_TEST_CHECK(synth_model_load(argv[1], &params, &model) == SYNTH_ERR_BACKEND && model == nullptr);
        SYNTH_TEST_CHECK(capture.count == 1 && capture.status == SYNTH_ERR_BACKEND);
    }

    diagnostic.emit     = throw_diagnostic;
    params.backend      = SYNTH_BACKEND_CPU;
    params.device_index = static_cast<int32_t>(synth_backend_device_count());
    SYNTH_TEST_CHECK(synth_model_load(argv[1], &params, &model) == SYNTH_ERR_BACKEND && model == nullptr);
    return 0;
}
