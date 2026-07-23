#ifndef SYNTHESIZE_H
#define SYNTHESIZE_H

#include <stdint.h>

#define SYNTH_VERSION_MAJOR 0
#define SYNTH_VERSION_MINOR 1
#define SYNTH_VERSION_PATCH 0
#define SYNTH_ABI_VERSION   ((uint32_t) 1u)

#if defined(_WIN32)
#    define SYNTH_CALL __cdecl
#    if defined(SYNTH_SHARED)
#        if defined(SYNTH_BUILDING_LIBRARY)
#            define SYNTH_API __declspec(dllexport)
#        else
#            define SYNTH_API __declspec(dllimport)
#        endif
#    else
#        define SYNTH_API
#    endif
#elif defined(__GNUC__) && defined(SYNTH_SHARED) && defined(SYNTH_BUILDING_LIBRARY)
#    define SYNTH_API __attribute__((visibility("default")))
#    define SYNTH_CALL
#else
#    define SYNTH_API
#    define SYNTH_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t synth_status_t;
#define SYNTH_OK                       ((synth_status_t) 0)
#define SYNTH_ERR_INVALID_ARG          ((synth_status_t) 1)
#define SYNTH_ERR_BAD_STRUCT_SIZE      ((synth_status_t) 2)
#define SYNTH_ERR_FILE_NOT_FOUND       ((synth_status_t) 3)
#define SYNTH_ERR_IO                   ((synth_status_t) 4)
#define SYNTH_ERR_GGUF                 ((synth_status_t) 5)
#define SYNTH_ERR_UNSUPPORTED_ARCH     ((synth_status_t) 6)
#define SYNTH_ERR_UNSUPPORTED_VARIANT  ((synth_status_t) 7)
#define SYNTH_ERR_UNSUPPORTED_INPUT    ((synth_status_t) 8)
#define SYNTH_ERR_UNSUPPORTED_LANGUAGE ((synth_status_t) 9)
#define SYNTH_ERR_UNSUPPORTED_VOICE    ((synth_status_t) 10)
#define SYNTH_ERR_UNSUPPORTED_CONTROL  ((synth_status_t) 11)
#define SYNTH_ERR_MISSING_RESOURCE     ((synth_status_t) 12)
#define SYNTH_ERR_TEXT_FRONTEND        ((synth_status_t) 13)
#define SYNTH_ERR_INPUT_TOO_LONG       ((synth_status_t) 14)
#define SYNTH_ERR_OUTPUT_LIMIT         ((synth_status_t) 15)
#define SYNTH_ERR_OOM                  ((synth_status_t) 16)
#define SYNTH_ERR_BACKEND              ((synth_status_t) 17)
#define SYNTH_ERR_CANCELLED            ((synth_status_t) 18)
#define SYNTH_ERR_SINK                 ((synth_status_t) 19)
#define SYNTH_ERR_INTERNAL             ((synth_status_t) 20)

typedef uint32_t synth_bool_t;
#define SYNTH_FALSE ((synth_bool_t) 0u)
#define SYNTH_TRUE  ((synth_bool_t) 1u)

typedef struct synth_model         synth_model_t;
typedef struct synth_context       synth_context_t;
typedef struct synth_voice_profile synth_voice_profile_t;

typedef struct synth_version {
    uint64_t struct_size;
    uint32_t abi_version;
    uint32_t version_major;
    uint32_t version_minor;
    uint32_t version_patch;
} synth_version_t;

typedef uint32_t synth_diagnostic_level_t;
#define SYNTH_DIAGNOSTIC_WARNING ((synth_diagnostic_level_t) 0u)
#define SYNTH_DIAGNOSTIC_ERROR   ((synth_diagnostic_level_t) 1u)

typedef struct synth_diagnostic {
    uint64_t                 struct_size;
    synth_diagnostic_level_t level;
    synth_status_t           status;
    const char *             code;
    uint64_t                 code_size;
    const char *             message;
    uint64_t                 message_size;
} synth_diagnostic_t;

typedef void(SYNTH_CALL * synth_diagnostic_callback_t)(void * user_data, const synth_diagnostic_t * diagnostic);

typedef struct synth_diagnostic_sink {
    uint64_t                    struct_size;
    synth_diagnostic_callback_t emit;
    void *                      user_data;
} synth_diagnostic_sink_t;

typedef uint32_t synth_backend_request_t;
#define SYNTH_BACKEND_AUTO      ((synth_backend_request_t) 0u)
#define SYNTH_BACKEND_CPU       ((synth_backend_request_t) 1u)
#define SYNTH_BACKEND_CPU_ACCEL ((synth_backend_request_t) 2u)
#define SYNTH_BACKEND_CUDA      ((synth_backend_request_t) 3u)
#define SYNTH_BACKEND_METAL     ((synth_backend_request_t) 4u)
#define SYNTH_BACKEND_VULKAN    ((synth_backend_request_t) 5u)

typedef uint32_t synth_device_type_t;
#define SYNTH_DEVICE_TYPE_CPU   ((synth_device_type_t) 0u)
#define SYNTH_DEVICE_TYPE_GPU   ((synth_device_type_t) 1u)
#define SYNTH_DEVICE_TYPE_IGPU  ((synth_device_type_t) 2u)
#define SYNTH_DEVICE_TYPE_ACCEL ((synth_device_type_t) 3u)

typedef uint32_t synth_device_flags_t;
#define SYNTH_DEVICE_MEMORY_INFO_VALID       ((synth_device_flags_t) (1u << 0))
#define SYNTH_DEVICE_MEMORY_SHARED           ((synth_device_flags_t) (1u << 1))
#define SYNTH_DEVICE_MEMORY_INFO_APPROXIMATE ((synth_device_flags_t) (1u << 2))

typedef struct synth_backend_device {
    uint64_t             struct_size;
    const char *         name;
    const char *         description;
    const char *         kind;
    const char *         device_id;
    uint64_t             memory_total;
    uint64_t             memory_free;
    synth_device_type_t  device_type;
    synth_device_flags_t flags;
} synth_backend_device_t;

typedef struct synth_model_load_params {
    uint64_t                        struct_size;
    synth_backend_request_t         backend;
    int32_t                         device_index;
    const synth_diagnostic_sink_t * diagnostics;
} synth_model_load_params_t;

typedef uint32_t synth_input_flags_t;
#define SYNTH_INPUT_SUPPORT_TEXT_UTF8     ((synth_input_flags_t) (1u << 0))
#define SYNTH_INPUT_SUPPORT_PHONEMES_UTF8 ((synth_input_flags_t) (1u << 1))
#define SYNTH_INPUT_SUPPORT_TOKEN_IDS     ((synth_input_flags_t) (1u << 2))

typedef uint32_t synth_model_capability_flags_t;
#define SYNTH_MODEL_CAPABILITY_SPEAKING_RATE    ((synth_model_capability_flags_t) (1u << 0))
#define SYNTH_MODEL_CAPABILITY_STOCHASTIC       ((synth_model_capability_flags_t) (1u << 1))
#define SYNTH_MODEL_CAPABILITY_NATIVE_STREAMING ((synth_model_capability_flags_t) (1u << 2))

typedef struct synth_model_capabilities {
    uint64_t                       struct_size;
    synth_input_flags_t            input_flags;
    synth_model_capability_flags_t capability_flags;
    uint32_t                       output_sample_rate;
    uint32_t                       output_channel_count;
    float                          min_speaking_rate;
    float                          max_speaking_rate;
    uint64_t                       max_input_tokens;
    uint64_t                       max_output_frames;
} synth_model_capabilities_t;

typedef uint32_t synth_preset_voice_flags_t;
#define SYNTH_PRESET_VOICE_DEFAULT ((synth_preset_voice_flags_t) (1u << 0))

typedef struct synth_preset_voice {
    uint64_t                   struct_size;
    const char *               id;
    uint64_t                   id_size;
    const char *               display_name;
    uint64_t                   display_name_size;
    synth_preset_voice_flags_t flags;
} synth_preset_voice_t;

typedef uint32_t synth_language_flags_t;
#define SYNTH_LANGUAGE_DEFAULT           ((synth_language_flags_t) (1u << 0))
#define SYNTH_LANGUAGE_REGIONAL_FALLBACK ((synth_language_flags_t) (1u << 1))

typedef struct synth_language_capability {
    uint64_t               struct_size;
    const char *           tag;
    uint64_t               tag_size;
    synth_language_flags_t flags;
} synth_language_capability_t;

typedef uint32_t synth_voice_profile_source_flags_t;
#define SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO    ((synth_voice_profile_source_flags_t) (1u << 0))
#define SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT   ((synth_voice_profile_source_flags_t) (1u << 1))
#define SYNTH_PROFILE_SOURCE_RANDOM_SEED        ((synth_voice_profile_source_flags_t) (1u << 2))
#define SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE ((synth_voice_profile_source_flags_t) (1u << 3))

typedef uint32_t synth_requirement_t;
#define SYNTH_REQUIREMENT_UNSUPPORTED ((synth_requirement_t) 0u)
#define SYNTH_REQUIREMENT_OPTIONAL    ((synth_requirement_t) 1u)
#define SYNTH_REQUIREMENT_REQUIRED    ((synth_requirement_t) 2u)

#define SYNTH_REFERENCE_SAMPLE_RATE_MIN ((uint32_t) 8000u)
#define SYNTH_REFERENCE_SAMPLE_RATE_MAX ((uint32_t) 192000u)
#define SYNTH_REFERENCE_CHANNELS_MAX    ((uint32_t) 2u)

typedef struct synth_voice_profile_capabilities {
    uint64_t                           struct_size;
    synth_voice_profile_source_flags_t source_flags;
    synth_requirement_t                reference_transcript;
    synth_requirement_t                reference_language;
    synth_requirement_t                description_language;
    uint64_t                           max_reference_count;
    const char *                       profile_schema;
    uint64_t                           profile_schema_size;
    uint32_t                           profile_schema_version;
    uint8_t                            profile_compatibility_id[32];
    uint32_t                           reference_target_sample_rate;
    uint32_t                           reference_target_channel_count;
    uint64_t                           min_reference_frames_per_clip;
    uint64_t                           max_reference_frames_per_clip;
    uint64_t                           max_reference_total_frames;
} synth_voice_profile_capabilities_t;

typedef struct synth_voice_reference {
    uint64_t      struct_size;
    const float * samples;
    uint64_t      frame_count;
    uint32_t      sample_rate;
    uint32_t      channel_count;
    const char *  transcript;
    uint64_t      transcript_size;
    const char *  language_tag;
    uint64_t      language_tag_size;
} synth_voice_reference_t;

typedef struct synth_voice_reference_params {
    uint64_t                        struct_size;
    const synth_voice_reference_t * references;
    uint64_t                        reference_count;
    uint64_t                        reference_stride;
    const synth_diagnostic_sink_t * diagnostics;
} synth_voice_reference_params_t;

typedef struct synth_voice_description_params {
    uint64_t                        struct_size;
    const char *                    description;
    uint64_t                        description_size;
    const char *                    language_tag;
    uint64_t                        language_tag_size;
    uint64_t                        seed;
    const synth_diagnostic_sink_t * diagnostics;
} synth_voice_description_params_t;

typedef struct synth_voice_random_params {
    uint64_t                        struct_size;
    uint64_t                        seed;
    const synth_diagnostic_sink_t * diagnostics;
} synth_voice_random_params_t;

typedef struct synth_voice_profile_load_params {
    uint64_t                        struct_size;
    const uint8_t *                 data;
    uint64_t                        data_size;
    const synth_diagnostic_sink_t * diagnostics;
} synth_voice_profile_load_params_t;

typedef struct synth_voice_profile_serialize_params {
    uint64_t                        struct_size;
    const synth_diagnostic_sink_t * diagnostics;
} synth_voice_profile_serialize_params_t;

typedef struct synth_byte_buffer {
    uint64_t        struct_size;
    const uint8_t * data;
    uint64_t        data_size;
} synth_byte_buffer_t;

typedef uint32_t synth_input_kind_t;
#define SYNTH_INPUT_TEXT_UTF8     ((synth_input_kind_t) 0u)
#define SYNTH_INPUT_PHONEMES_UTF8 ((synth_input_kind_t) 1u)
#define SYNTH_INPUT_TOKEN_IDS     ((synth_input_kind_t) 2u)

#define SYNTH_SEED_RANDOM UINT64_MAX

typedef synth_bool_t(SYNTH_CALL * synth_cancel_callback_t)(void * user_data);

typedef struct synth_request {
    uint64_t                        struct_size;
    synth_input_kind_t              input_kind;
    const void *                    input_data;
    uint64_t                        input_count;
    const char *                    language_tag;
    uint64_t                        language_tag_size;
    const char *                    voice_id;
    uint64_t                        voice_id_size;
    uint64_t                        seed;
    float                           speaking_rate;
    synth_cancel_callback_t         should_cancel;
    void *                          cancel_user_data;
    const synth_diagnostic_sink_t * diagnostics;
    uint64_t                        max_output_frames;
    const synth_voice_profile_t *   voice_profile;
} synth_request_t;

typedef uint32_t synth_sink_result_t;
#define SYNTH_SINK_CONTINUE ((synth_sink_result_t) 0u)
#define SYNTH_SINK_CANCEL   ((synth_sink_result_t) 1u)
#define SYNTH_SINK_ERROR    ((synth_sink_result_t) 2u)

typedef struct synth_audio_chunk {
    uint64_t      struct_size;
    const float * samples;
    uint64_t      frame_count;
    uint64_t      frame_offset;
    uint32_t      sample_rate;
    uint32_t      channel_count;
} synth_audio_chunk_t;

typedef synth_sink_result_t(SYNTH_CALL * synth_audio_write_callback_t)(void *                      user_data,
                                                                       const synth_audio_chunk_t * chunk);

typedef struct synth_audio_sink {
    uint64_t                     struct_size;
    synth_audio_write_callback_t write;
    void *                       user_data;
} synth_audio_sink_t;

typedef uint32_t synth_result_flags_t;
#define SYNTH_RESULT_SEED_USED             ((synth_result_flags_t) (1u << 0))
#define SYNTH_RESULT_NATIVE_STREAMING_USED ((synth_result_flags_t) (1u << 1))

typedef struct synth_result {
    uint64_t             struct_size;
    uint64_t             frames_emitted;
    uint64_t             actual_seed;
    uint32_t             sample_rate;
    uint32_t             channel_count;
    synth_result_flags_t flags;
    const char *         resolved_language_tag;
    uint64_t             resolved_language_tag_size;
    const char *         resolved_voice_id;
    uint64_t             resolved_voice_id_size;
} synth_result_t;

typedef struct synth_audio_buffer {
    uint64_t      struct_size;
    const float * samples;
    uint64_t      frame_count;
    uint32_t      sample_rate;
    uint32_t      channel_count;
} synth_audio_buffer_t;

SYNTH_API uint32_t SYNTH_CALL       synth_abi_version(void);
SYNTH_API const char * SYNTH_CALL   synth_status_string(synth_status_t status);
SYNTH_API void SYNTH_CALL           synth_version_init(synth_version_t * version, uint64_t struct_size);
SYNTH_API synth_status_t SYNTH_CALL synth_get_version(synth_version_t * out_version);
SYNTH_API void SYNTH_CALL           synth_diagnostic_sink_init(synth_diagnostic_sink_t * sink, uint64_t struct_size);
SYNTH_API synth_status_t SYNTH_CALL synth_backend_load_from_dir(const char * artifact_dir);
SYNTH_API synth_status_t SYNTH_CALL synth_backend_load_default(void);
SYNTH_API uint32_t SYNTH_CALL       synth_backend_device_count(void);
SYNTH_API void SYNTH_CALL           synth_backend_device_init(synth_backend_device_t * device, uint64_t struct_size);
SYNTH_API synth_status_t SYNTH_CALL synth_backend_device_get(uint32_t index, synth_backend_device_t * out_device);
SYNTH_API synth_bool_t SYNTH_CALL   synth_backend_available(synth_backend_request_t request);
SYNTH_API synth_status_t SYNTH_CALL synth_model_get_device(const synth_model_t *    model,
                                                           synth_backend_device_t * out_device);
SYNTH_API void SYNTH_CALL synth_model_load_params_init(synth_model_load_params_t * params, uint64_t struct_size);
SYNTH_API synth_status_t SYNTH_CALL synth_model_load(const char *                      model_path,
                                                     const synth_model_load_params_t * params,
                                                     synth_model_t **                  out_model);
SYNTH_API void SYNTH_CALL           synth_model_free(synth_model_t * model);
SYNTH_API synth_status_t SYNTH_CALL synth_context_create(const synth_model_t * model, synth_context_t ** out_context);
SYNTH_API void SYNTH_CALL           synth_context_free(synth_context_t * context);

SYNTH_API void SYNTH_CALL           synth_model_capabilities_init(synth_model_capabilities_t * capabilities,
                                                                  uint64_t                     struct_size);
SYNTH_API synth_status_t SYNTH_CALL synth_model_get_capabilities(const synth_model_t *        model,
                                                                 synth_model_capabilities_t * out_capabilities);
SYNTH_API void SYNTH_CALL           synth_preset_voice_init(synth_preset_voice_t * voice, uint64_t struct_size);
SYNTH_API synth_status_t SYNTH_CALL synth_model_get_preset_voice_count(const synth_model_t * model,
                                                                       uint64_t *            out_count);
SYNTH_API synth_status_t SYNTH_CALL synth_model_get_preset_voice(const synth_model_t *  model,
                                                                 uint64_t               index,
                                                                 synth_preset_voice_t * out_voice);
SYNTH_API void SYNTH_CALL synth_language_capability_init(synth_language_capability_t * language, uint64_t struct_size);
SYNTH_API synth_status_t SYNTH_CALL synth_model_get_language_count(const synth_model_t * model, uint64_t * out_count);
SYNTH_API synth_status_t SYNTH_CALL synth_model_get_language(const synth_model_t *         model,
                                                             uint64_t                      index,
                                                             synth_language_capability_t * out_language);

SYNTH_API void SYNTH_CALL synth_voice_profile_capabilities_init(synth_voice_profile_capabilities_t * capabilities,
                                                                uint64_t                             struct_size);
SYNTH_API synth_status_t SYNTH_CALL
synth_model_get_voice_profile_capabilities(const synth_model_t *                model,
                                           synth_voice_profile_capabilities_t * out_capabilities);
SYNTH_API void SYNTH_CALL synth_voice_reference_init(synth_voice_reference_t * reference, uint64_t struct_size);
SYNTH_API void SYNTH_CALL synth_voice_reference_params_init(synth_voice_reference_params_t * params,
                                                            uint64_t                         struct_size);
SYNTH_API synth_status_t SYNTH_CALL
synth_voice_profile_create_from_reference(const synth_model_t *                  model,
                                          const synth_voice_reference_params_t * params,
                                          synth_voice_profile_t **               out_profile);
SYNTH_API void SYNTH_CALL synth_voice_description_params_init(synth_voice_description_params_t * params,
                                                              uint64_t                           struct_size);
SYNTH_API synth_status_t SYNTH_CALL
synth_voice_profile_create_from_description(const synth_model_t *                    model,
                                            const synth_voice_description_params_t * params,
                                            synth_voice_profile_t **                 out_profile);
SYNTH_API void SYNTH_CALL synth_voice_random_params_init(synth_voice_random_params_t * params, uint64_t struct_size);
SYNTH_API synth_status_t SYNTH_CALL synth_voice_profile_create_random(const synth_model_t *               model,
                                                                      const synth_voice_random_params_t * params,
                                                                      synth_voice_profile_t **            out_profile);
SYNTH_API void SYNTH_CALL           synth_voice_profile_load_params_init(synth_voice_profile_load_params_t * params,
                                                                         uint64_t                            struct_size);
SYNTH_API void SYNTH_CALL synth_voice_profile_serialize_params_init(synth_voice_profile_serialize_params_t * params,
                                                                    uint64_t struct_size);
SYNTH_API synth_status_t SYNTH_CALL
synth_voice_profile_load_from_memory(const synth_model_t *                     model,
                                     const synth_voice_profile_load_params_t * params,
                                     synth_voice_profile_t **                  out_profile);
SYNTH_API synth_status_t SYNTH_CALL synth_voice_profile_serialize(const synth_voice_profile_t * profile,
                                                                  const synth_voice_profile_serialize_params_t * params,
                                                                  synth_byte_buffer_t ** out_data);
SYNTH_API void SYNTH_CALL           synth_byte_buffer_free(synth_byte_buffer_t * buffer);
SYNTH_API void SYNTH_CALL           synth_voice_profile_free(synth_voice_profile_t * profile);

SYNTH_API void SYNTH_CALL           synth_request_init(synth_request_t * request, uint64_t struct_size);
SYNTH_API void SYNTH_CALL           synth_audio_sink_init(synth_audio_sink_t * sink, uint64_t struct_size);
SYNTH_API void SYNTH_CALL           synth_result_init(synth_result_t * result, uint64_t struct_size);
SYNTH_API synth_status_t SYNTH_CALL synth_synthesize(synth_context_t *          context,
                                                     const synth_request_t *    request,
                                                     const synth_audio_sink_t * sink,
                                                     synth_result_t *           out_result);
SYNTH_API synth_status_t SYNTH_CALL synth_synthesize_to_buffer(synth_context_t *       context,
                                                               const synth_request_t * request,
                                                               synth_audio_buffer_t ** out_audio,
                                                               synth_result_t *        out_result);
SYNTH_API void SYNTH_CALL           synth_audio_buffer_free(synth_audio_buffer_t * audio);

#ifdef __cplusplus
}
#endif

#endif
