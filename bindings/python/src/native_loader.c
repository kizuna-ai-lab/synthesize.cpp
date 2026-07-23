#define PY_SSIZE_T_CLEAN
#include "synthesize.h"

#include <Python.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    include <windows.h>
typedef HMODULE synth_library_handle_t;
#else
#    include <dlfcn.h>
typedef void * synth_library_handle_t;
#endif

typedef uint32_t(SYNTH_CALL * synth_abi_version_fn)(void);
typedef const char *(SYNTH_CALL * synth_status_string_fn)(synth_status_t);
typedef void(SYNTH_CALL * synth_version_init_fn)(synth_version_t *, uint64_t);
typedef synth_status_t(SYNTH_CALL * synth_get_version_fn)(synth_version_t *);
typedef void(SYNTH_CALL * synth_diagnostic_sink_init_fn)(synth_diagnostic_sink_t *, uint64_t);
typedef uint32_t(SYNTH_CALL * synth_backend_device_count_fn)(void);
typedef void(SYNTH_CALL * synth_backend_device_init_fn)(synth_backend_device_t *, uint64_t);
typedef synth_status_t(SYNTH_CALL * synth_backend_device_get_fn)(uint32_t, synth_backend_device_t *);
typedef synth_bool_t(SYNTH_CALL * synth_backend_available_fn)(synth_backend_request_t);
typedef synth_status_t(SYNTH_CALL * synth_backend_load_default_fn)(void);
typedef synth_status_t(SYNTH_CALL * synth_model_get_device_fn)(const synth_model_t *, synth_backend_device_t *);
typedef void(SYNTH_CALL * synth_model_load_params_init_fn)(synth_model_load_params_t *, uint64_t);
typedef synth_status_t(SYNTH_CALL * synth_model_load_fn)(const char *,
                                                         const synth_model_load_params_t *,
                                                         synth_model_t **);
typedef void(SYNTH_CALL * synth_model_free_fn)(synth_model_t *);
typedef synth_status_t(SYNTH_CALL * synth_context_create_fn)(const synth_model_t *, synth_context_t **);
typedef void(SYNTH_CALL * synth_context_free_fn)(synth_context_t *);
typedef void(SYNTH_CALL * synth_model_capabilities_init_fn)(synth_model_capabilities_t *, uint64_t);
typedef synth_status_t(SYNTH_CALL * synth_model_get_capabilities_fn)(const synth_model_t *,
                                                                     synth_model_capabilities_t *);
typedef void(SYNTH_CALL * synth_preset_voice_init_fn)(synth_preset_voice_t *, uint64_t);
typedef synth_status_t(SYNTH_CALL * synth_model_get_preset_voice_count_fn)(const synth_model_t *, uint64_t *);
typedef synth_status_t(SYNTH_CALL * synth_model_get_preset_voice_fn)(const synth_model_t *,
                                                                     uint64_t,
                                                                     synth_preset_voice_t *);
typedef void(SYNTH_CALL * synth_language_capability_init_fn)(synth_language_capability_t *, uint64_t);
typedef synth_status_t(SYNTH_CALL * synth_model_get_language_count_fn)(const synth_model_t *, uint64_t *);
typedef synth_status_t(SYNTH_CALL * synth_model_get_language_fn)(const synth_model_t *,
                                                                 uint64_t,
                                                                 synth_language_capability_t *);
typedef void(SYNTH_CALL * synth_voice_profile_capabilities_init_fn)(synth_voice_profile_capabilities_t *, uint64_t);
typedef synth_status_t(SYNTH_CALL * synth_model_get_voice_profile_capabilities_fn)(
    const synth_model_t *,
    synth_voice_profile_capabilities_t *);
typedef void(SYNTH_CALL * synth_voice_reference_init_fn)(synth_voice_reference_t *, uint64_t);
typedef void(SYNTH_CALL * synth_voice_reference_params_init_fn)(synth_voice_reference_params_t *, uint64_t);
typedef synth_status_t(SYNTH_CALL * synth_voice_profile_create_from_reference_fn)(
    const synth_model_t *,
    const synth_voice_reference_params_t *,
    synth_voice_profile_t **);
typedef void(SYNTH_CALL * synth_voice_description_params_init_fn)(synth_voice_description_params_t *, uint64_t);
typedef synth_status_t(SYNTH_CALL * synth_voice_profile_create_from_description_fn)(
    const synth_model_t *,
    const synth_voice_description_params_t *,
    synth_voice_profile_t **);
typedef void(SYNTH_CALL * synth_voice_random_params_init_fn)(synth_voice_random_params_t *, uint64_t);
typedef synth_status_t(SYNTH_CALL * synth_voice_profile_create_random_fn)(const synth_model_t *,
                                                                          const synth_voice_random_params_t *,
                                                                          synth_voice_profile_t **);
typedef void(SYNTH_CALL * synth_voice_profile_load_params_init_fn)(synth_voice_profile_load_params_t *, uint64_t);
typedef void(SYNTH_CALL * synth_voice_profile_serialize_params_init_fn)(synth_voice_profile_serialize_params_t *,
                                                                        uint64_t);
typedef synth_status_t(SYNTH_CALL * synth_voice_profile_load_from_memory_fn)(const synth_model_t *,
                                                                             const synth_voice_profile_load_params_t *,
                                                                             synth_voice_profile_t **);
typedef synth_status_t(SYNTH_CALL * synth_voice_profile_serialize_fn)(const synth_voice_profile_t *,
                                                                      const synth_voice_profile_serialize_params_t *,
                                                                      synth_byte_buffer_t **);
typedef void(SYNTH_CALL * synth_byte_buffer_free_fn)(synth_byte_buffer_t *);
typedef void(SYNTH_CALL * synth_voice_profile_free_fn)(synth_voice_profile_t *);
typedef void(SYNTH_CALL * synth_request_init_fn)(synth_request_t *, uint64_t);
typedef void(SYNTH_CALL * synth_result_init_fn)(synth_result_t *, uint64_t);
typedef synth_status_t(SYNTH_CALL * synth_synthesize_to_buffer_fn)(synth_context_t *,
                                                                   const synth_request_t *,
                                                                   synth_audio_buffer_t **,
                                                                   synth_result_t *);
typedef void(SYNTH_CALL * synth_audio_buffer_free_fn)(synth_audio_buffer_t *);

typedef struct synth_loaded_api {
    synth_library_handle_t                         library;
    char *                                         path;
    uint32_t                                       abi_version;
    uint32_t                                       version_major;
    uint32_t                                       version_minor;
    uint32_t                                       version_patch;
    synth_status_string_fn                         status_string;
    synth_diagnostic_sink_init_fn                  diagnostic_sink_init;
    synth_backend_device_count_fn                  backend_device_count;
    synth_backend_device_init_fn                   backend_device_init;
    synth_backend_device_get_fn                    backend_device_get;
    synth_backend_available_fn                     backend_available;
    synth_model_get_device_fn                      model_get_device;
    synth_model_load_params_init_fn                model_load_params_init;
    synth_model_load_fn                            model_load;
    synth_model_free_fn                            model_free;
    synth_context_create_fn                        context_create;
    synth_context_free_fn                          context_free;
    synth_model_capabilities_init_fn               model_capabilities_init;
    synth_model_get_capabilities_fn                model_get_capabilities;
    synth_preset_voice_init_fn                     preset_voice_init;
    synth_model_get_preset_voice_count_fn          model_get_preset_voice_count;
    synth_model_get_preset_voice_fn                model_get_preset_voice;
    synth_language_capability_init_fn              language_capability_init;
    synth_model_get_language_count_fn              model_get_language_count;
    synth_model_get_language_fn                    model_get_language;
    synth_voice_profile_capabilities_init_fn       voice_profile_capabilities_init;
    synth_model_get_voice_profile_capabilities_fn  model_get_voice_profile_capabilities;
    synth_voice_reference_init_fn                  voice_reference_init;
    synth_voice_reference_params_init_fn           voice_reference_params_init;
    synth_voice_profile_create_from_reference_fn   voice_profile_create_from_reference;
    synth_voice_description_params_init_fn         voice_description_params_init;
    synth_voice_profile_create_from_description_fn voice_profile_create_from_description;
    synth_voice_random_params_init_fn              voice_random_params_init;
    synth_voice_profile_create_random_fn           voice_profile_create_random;
    synth_voice_profile_load_params_init_fn        voice_profile_load_params_init;
    synth_voice_profile_serialize_params_init_fn   voice_profile_serialize_params_init;
    synth_voice_profile_load_from_memory_fn        voice_profile_load_from_memory;
    synth_voice_profile_serialize_fn               voice_profile_serialize;
    synth_byte_buffer_free_fn                      byte_buffer_free;
    synth_voice_profile_free_fn                    voice_profile_free;
    synth_request_init_fn                          request_init;
    synth_result_init_fn                           result_init;
    synth_synthesize_to_buffer_fn                  synthesize_to_buffer;
    synth_audio_buffer_free_fn                     audio_buffer_free;
} synth_loaded_api_t;

static synth_loaded_api_t synth_api = { 0 };

#define SYNTH_MODEL_CAPSULE                "synthesize_cpp.Model"
#define SYNTH_CLOSED_MODEL_CAPSULE         "synthesize_cpp.Model.closed"
#define SYNTH_CONTEXT_CAPSULE              "synthesize_cpp.Context"
#define SYNTH_CLOSED_CONTEXT_CAPSULE       "synthesize_cpp.Context.closed"
#define SYNTH_VOICE_PROFILE_CAPSULE        "synthesize_cpp.VoiceProfile"
#define SYNTH_CLOSED_VOICE_PROFILE_CAPSULE "synthesize_cpp.VoiceProfile.closed"

typedef struct synth_model_owner {
    synth_model_t * model;
    uint64_t        context_count;
    uint64_t        profile_count;
    int             close_requested;
} synth_model_owner_t;

typedef struct synth_context_owner {
    synth_context_t *     context;
    synth_model_owner_t * model_owner;
    PyObject *            model_capsule;
    uint64_t              active_count;
    int                   close_requested;
} synth_context_owner_t;

typedef struct synth_voice_profile_owner {
    synth_voice_profile_t * profile;
    synth_model_owner_t *   model_owner;
    PyObject *              model_capsule;
    uint64_t                active_count;
    int                     close_requested;
} synth_voice_profile_owner_t;

typedef struct synth_audio_owner {
    PyObject_HEAD synth_audio_buffer_t * audio;
    Py_ssize_t                           shape;
    Py_ssize_t                           stride;
} synth_audio_owner_t;

typedef struct synth_byte_buffer_owner {
    PyObject_HEAD synth_byte_buffer_t * buffer;
    Py_ssize_t                          shape;
    Py_ssize_t                          stride;
} synth_byte_buffer_owner_t;

static PyObject * synth_audio_owner_type       = NULL;
static PyObject * synth_byte_buffer_owner_type = NULL;

typedef struct synth_python_callback {
    PyObject * callback;
    PyObject * exception_type;
    PyObject * exception_value;
    PyObject * exception_traceback;
} synth_python_callback_t;

static void synth_capture_callback_exception(synth_python_callback_t * state) {
    if (state->exception_type == NULL) {
        PyErr_Fetch(&state->exception_type, &state->exception_value, &state->exception_traceback);
    } else {
        PyErr_Clear();
    }
}

static void synth_restore_callback_exception(synth_python_callback_t * state) {
    if (state->exception_type != NULL) {
        PyErr_Restore(state->exception_type, state->exception_value, state->exception_traceback);
        state->exception_type      = NULL;
        state->exception_value     = NULL;
        state->exception_traceback = NULL;
    }
}

static PyObject * synth_decode_diagnostic_string(const char * data, uint64_t size, const char * field_name) {
    if ((data == NULL) != (size == 0)) {
        PyErr_Format(PyExc_RuntimeError, "native diagnostic %s has inconsistent storage", field_name);
        return NULL;
    }
    if (size > (uint64_t) PY_SSIZE_T_MAX) {
        PyErr_Format(PyExc_OverflowError, "native diagnostic %s is too large", field_name);
        return NULL;
    }
    return PyUnicode_DecodeUTF8(data != NULL ? data : "", (Py_ssize_t) size, "strict");
}

static void SYNTH_CALL synth_python_emit_diagnostic(void * user_data, const synth_diagnostic_t * diagnostic) {
    synth_python_callback_t * state = (synth_python_callback_t *) user_data;
    PyGILState_STATE          gil_state;
    PyObject *                code            = NULL;
    PyObject *                message         = NULL;
    PyObject *                value           = NULL;
    PyObject *                callback_result = NULL;
    if (state == NULL || state->callback == NULL || state->exception_type != NULL) {
        return;
    }

    gil_state = PyGILState_Ensure();
    if (diagnostic == NULL ||
        diagnostic->struct_size < offsetof(synth_diagnostic_t, message_size) + sizeof(diagnostic->message_size)) {
        PyErr_SetString(PyExc_RuntimeError, "native diagnostic is malformed");
        synth_capture_callback_exception(state);
        PyGILState_Release(gil_state);
        return;
    }
    code = synth_decode_diagnostic_string(diagnostic->code, diagnostic->code_size, "code");
    if (code != NULL) {
        message = synth_decode_diagnostic_string(diagnostic->message, diagnostic->message_size, "message");
    }
    if (message != NULL) {
        value = Py_BuildValue("{s:I,s:i,s:O,s:O}", "level", (unsigned int) diagnostic->level, "status",
                              (int) diagnostic->status, "code", code, "message", message);
    }
    if (value != NULL) {
        callback_result = PyObject_CallFunctionObjArgs(state->callback, value, NULL);
    }
    Py_XDECREF(callback_result);
    Py_XDECREF(value);
    Py_XDECREF(message);
    Py_XDECREF(code);
    if (PyErr_Occurred()) {
        synth_capture_callback_exception(state);
    }
    PyGILState_Release(gil_state);
}

static synth_bool_t SYNTH_CALL synth_python_should_cancel(void * user_data) {
    synth_python_callback_t * state = (synth_python_callback_t *) user_data;
    PyGILState_STATE          gil_state;
    PyObject *                result;
    int                       truth;
    if (state == NULL || state->callback == NULL) {
        return SYNTH_FALSE;
    }
    if (state->exception_type != NULL) {
        return SYNTH_TRUE;
    }

    gil_state = PyGILState_Ensure();
    result    = PyObject_CallNoArgs(state->callback);
    if (result == NULL) {
        synth_capture_callback_exception(state);
        PyGILState_Release(gil_state);
        return SYNTH_TRUE;
    }
    truth = PyObject_IsTrue(result);
    Py_DECREF(result);
    if (truth < 0) {
        synth_capture_callback_exception(state);
        PyGILState_Release(gil_state);
        return SYNTH_TRUE;
    }
    PyGILState_Release(gil_state);
    return truth != 0 ? SYNTH_TRUE : SYNTH_FALSE;
}

static void synth_close_library(synth_library_handle_t library) {
#if defined(_WIN32)
    if (library != NULL) {
        FreeLibrary(library);
    }
#else
    if (library != NULL) {
        dlclose(library);
    }
#endif
}

static synth_library_handle_t synth_open_library(const char * path) {
#if defined(_WIN32)
    return LoadLibraryExA(path, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

static int synth_set_open_error(const char * path) {
#if defined(_WIN32)
    PyErr_Format(PyExc_RuntimeError, "could not load native library '%s' (Windows error %lu)", path,
                 (unsigned long) GetLastError());
#else
    const char * message = dlerror();
    PyErr_Format(PyExc_RuntimeError, "could not load native library '%s': %s", path,
                 message != NULL ? message : "unknown dynamic-loader error");
#endif
    return -1;
}

static int synth_resolve_symbol(synth_library_handle_t library,
                                const char *           path,
                                const char *           name,
                                void *                 out_function,
                                size_t                 out_function_size) {
#if defined(_WIN32)
    FARPROC address = GetProcAddress(library, name);
    if (address == NULL) {
        PyErr_Format(PyExc_RuntimeError, "native library '%s' is missing required symbol '%s'", path, name);
        return -1;
    }
#else
    void *       address;
    const char * error;
    dlerror();
    address = dlsym(library, name);
    error   = dlerror();
    if (error != NULL) {
        PyErr_Format(PyExc_RuntimeError, "native library '%s' is missing required symbol '%s': %s", path, name, error);
        return -1;
    }
#endif

    if (out_function != NULL) {
        if (out_function_size != sizeof(address)) {
            PyErr_Format(PyExc_RuntimeError, "cannot represent native symbol '%s' on this platform", name);
            return -1;
        }
        memcpy(out_function, &address, sizeof(address));
    }
    return 0;
}

static void synth_set_status_error(const char * operation, synth_status_t status) {
    const char * message = synth_api.status_string != NULL ? synth_api.status_string(status) : NULL;
    PyErr_Format(PyExc_RuntimeError, "%s failed: %s (%d)", operation, message != NULL ? message : "unknown status",
                 (int) status);
}

static synth_model_owner_t * synth_model_owner_from_capsule(PyObject * capsule, int require_open) {
    const char *          name = NULL;
    synth_model_owner_t * owner;
    if (PyCapsule_IsValid(capsule, SYNTH_MODEL_CAPSULE)) {
        name = SYNTH_MODEL_CAPSULE;
    } else if (!require_open && PyCapsule_IsValid(capsule, SYNTH_CLOSED_MODEL_CAPSULE)) {
        name = SYNTH_CLOSED_MODEL_CAPSULE;
    }
    if (name == NULL) {
        PyErr_SetString(PyExc_TypeError,
                        require_open ? "expected an open internal Model handle" : "expected an internal Model handle");
        return NULL;
    }
    owner = (synth_model_owner_t *) PyCapsule_GetPointer(capsule, name);
    if (owner == NULL) {
        return NULL;
    }
    if (require_open && (owner->model == NULL || owner->close_requested)) {
        PyErr_SetString(PyExc_RuntimeError, "native Model handle is closed");
        return NULL;
    }
    return owner;
}

static synth_context_owner_t * synth_context_owner_from_capsule(PyObject * capsule, int require_open) {
    const char *            name = NULL;
    synth_context_owner_t * owner;
    if (PyCapsule_IsValid(capsule, SYNTH_CONTEXT_CAPSULE)) {
        name = SYNTH_CONTEXT_CAPSULE;
    } else if (!require_open && PyCapsule_IsValid(capsule, SYNTH_CLOSED_CONTEXT_CAPSULE)) {
        name = SYNTH_CLOSED_CONTEXT_CAPSULE;
    }
    if (name == NULL) {
        PyErr_SetString(PyExc_TypeError, require_open ? "expected an open internal Context handle" :
                                                        "expected an internal Context handle");
        return NULL;
    }
    owner = (synth_context_owner_t *) PyCapsule_GetPointer(capsule, name);
    if (owner == NULL) {
        return NULL;
    }
    if (require_open && owner->context == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "native Context handle is closed");
        return NULL;
    }
    return owner;
}

static synth_voice_profile_owner_t * synth_voice_profile_owner_from_capsule(PyObject * capsule, int require_open) {
    const char *                  name = NULL;
    synth_voice_profile_owner_t * owner;
    if (PyCapsule_IsValid(capsule, SYNTH_VOICE_PROFILE_CAPSULE)) {
        name = SYNTH_VOICE_PROFILE_CAPSULE;
    } else if (!require_open && PyCapsule_IsValid(capsule, SYNTH_CLOSED_VOICE_PROFILE_CAPSULE)) {
        name = SYNTH_CLOSED_VOICE_PROFILE_CAPSULE;
    }
    if (name == NULL) {
        PyErr_SetString(PyExc_TypeError, require_open ? "expected an open internal Voice Profile handle" :
                                                        "expected an internal Voice Profile handle");
        return NULL;
    }
    owner = (synth_voice_profile_owner_t *) PyCapsule_GetPointer(capsule, name);
    if (owner == NULL) {
        return NULL;
    }
    if (require_open && owner->profile == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "native Voice Profile handle is closed");
        return NULL;
    }
    return owner;
}

static void synth_release_context_owner(synth_context_owner_t * owner) {
    synth_context_t * context;
    synth_model_t *   deferred_model = NULL;
    PyObject *        model_capsule;
    if (owner == NULL || owner->context == NULL || owner->active_count != 0) {
        return;
    }

    context              = owner->context;
    owner->context       = NULL;
    model_capsule        = owner->model_capsule;
    owner->model_capsule = NULL;
    if (owner->model_owner != NULL) {
        if (owner->model_owner->context_count > 0) {
            --owner->model_owner->context_count;
        }
        if (owner->model_owner->context_count == 0 && owner->model_owner->profile_count == 0 &&
            owner->model_owner->close_requested && owner->model_owner->model != NULL) {
            deferred_model            = owner->model_owner->model;
            owner->model_owner->model = NULL;
        }
    }
    owner->model_owner = NULL;

    synth_api.context_free(context);
    if (deferred_model != NULL) {
        synth_api.model_free(deferred_model);
    }
    Py_XDECREF(model_capsule);
}

static void synth_release_voice_profile_owner(synth_voice_profile_owner_t * owner) {
    synth_voice_profile_t * profile;
    synth_model_t *         deferred_model = NULL;
    PyObject *              model_capsule;
    if (owner == NULL || owner->profile == NULL || owner->active_count != 0) {
        return;
    }

    profile              = owner->profile;
    owner->profile       = NULL;
    model_capsule        = owner->model_capsule;
    owner->model_capsule = NULL;
    if (owner->model_owner != NULL) {
        if (owner->model_owner->profile_count > 0) {
            --owner->model_owner->profile_count;
        }
        if (owner->model_owner->context_count == 0 && owner->model_owner->profile_count == 0 &&
            owner->model_owner->close_requested && owner->model_owner->model != NULL) {
            deferred_model            = owner->model_owner->model;
            owner->model_owner->model = NULL;
        }
    }
    owner->model_owner = NULL;

    synth_api.voice_profile_free(profile);
    if (deferred_model != NULL) {
        synth_api.model_free(deferred_model);
    }
    Py_XDECREF(model_capsule);
}

static void synth_model_capsule_destructor(PyObject * capsule) {
    synth_model_owner_t * owner = synth_model_owner_from_capsule(capsule, 0);
    if (owner == NULL) {
        PyErr_Clear();
        return;
    }
    if (owner->model != NULL) {
        synth_api.model_free(owner->model);
        owner->model = NULL;
    }
    free(owner);
}

static void synth_context_capsule_destructor(PyObject * capsule) {
    synth_context_owner_t * owner = synth_context_owner_from_capsule(capsule, 0);
    if (owner == NULL) {
        PyErr_Clear();
        return;
    }
    synth_release_context_owner(owner);
    free(owner);
}

static void synth_voice_profile_capsule_destructor(PyObject * capsule) {
    synth_voice_profile_owner_t * owner = synth_voice_profile_owner_from_capsule(capsule, 0);
    if (owner == NULL) {
        PyErr_Clear();
        return;
    }
    synth_release_voice_profile_owner(owner);
    free(owner);
}

static int synth_audio_owner_getbuffer(PyObject * exporter, Py_buffer * view, int flags) {
    synth_audio_owner_t * owner = (synth_audio_owner_t *) exporter;
    uint64_t              sample_count;
    if ((flags & PyBUF_WRITABLE) != 0) {
        PyErr_SetString(PyExc_BufferError, "synthesized audio is read-only");
        return -1;
    }
    if (owner->audio == NULL) {
        PyErr_SetString(PyExc_BufferError, "synthesized audio buffer is closed");
        return -1;
    }
    if (owner->audio->channel_count != 0 && owner->audio->frame_count > UINT64_MAX / owner->audio->channel_count) {
        PyErr_SetString(PyExc_OverflowError, "audio sample count overflow");
        return -1;
    }
    sample_count = owner->audio->frame_count * owner->audio->channel_count;
    if (sample_count > (uint64_t) PY_SSIZE_T_MAX / sizeof(float)) {
        PyErr_SetString(PyExc_OverflowError, "audio buffer is too large for Python");
        return -1;
    }

    owner->shape  = (Py_ssize_t) sample_count;
    owner->stride = (Py_ssize_t) sizeof(float);
    view->buf     = (void *) owner->audio->samples;
    view->obj     = exporter;
    Py_INCREF(exporter);
    view->len        = owner->shape * owner->stride;
    view->readonly   = 1;
    view->itemsize   = owner->stride;
    view->format     = (flags & PyBUF_FORMAT) != 0 ? "f" : NULL;
    view->ndim       = 1;
    view->shape      = (flags & PyBUF_ND) != 0 ? &owner->shape : NULL;
    view->strides    = (flags & PyBUF_STRIDES) != 0 ? &owner->stride : NULL;
    view->suboffsets = NULL;
    view->internal   = NULL;
    return 0;
}

static void synth_audio_owner_releasebuffer(PyObject * exporter, Py_buffer * view) {
    (void) exporter;
    (void) view;
}

static void synth_audio_owner_dealloc(PyObject * object) {
    synth_audio_owner_t * owner = (synth_audio_owner_t *) object;
    PyTypeObject *        type  = Py_TYPE(object);
    if (owner->audio != NULL) {
        synth_api.audio_buffer_free(owner->audio);
        owner->audio = NULL;
    }
    PyObject_Free(object);
    Py_DECREF((PyObject *) type);
}

static PyType_Slot synth_audio_owner_slots[] = {
    { Py_tp_new,           PyType_GenericNew               },
    { Py_tp_dealloc,       synth_audio_owner_dealloc       },
    { Py_bf_getbuffer,     synth_audio_owner_getbuffer     },
    { Py_bf_releasebuffer, synth_audio_owner_releasebuffer },
    { 0,                   NULL                            },
};

static PyType_Spec synth_audio_owner_spec = {
    "synthesize_cpp._native._AudioBuffer", sizeof(synth_audio_owner_t), 0, Py_TPFLAGS_DEFAULT, synth_audio_owner_slots,
};

static int synth_byte_buffer_owner_getbuffer(PyObject * exporter, Py_buffer * view, int flags) {
    synth_byte_buffer_owner_t * owner = (synth_byte_buffer_owner_t *) exporter;
    if ((flags & PyBUF_WRITABLE) != 0) {
        PyErr_SetString(PyExc_BufferError, "serialized Voice Profile is read-only");
        return -1;
    }
    if (owner->buffer == NULL) {
        PyErr_SetString(PyExc_BufferError, "serialized Voice Profile buffer is closed");
        return -1;
    }
    if (owner->buffer->data_size > (uint64_t) PY_SSIZE_T_MAX) {
        PyErr_SetString(PyExc_OverflowError, "serialized Voice Profile is too large for Python");
        return -1;
    }
    owner->shape  = (Py_ssize_t) owner->buffer->data_size;
    owner->stride = 1;
    view->buf     = (void *) owner->buffer->data;
    view->obj     = exporter;
    Py_INCREF(exporter);
    view->len        = owner->shape;
    view->readonly   = 1;
    view->itemsize   = 1;
    view->format     = (flags & PyBUF_FORMAT) != 0 ? "B" : NULL;
    view->ndim       = 1;
    view->shape      = (flags & PyBUF_ND) != 0 ? &owner->shape : NULL;
    view->strides    = (flags & PyBUF_STRIDES) != 0 ? &owner->stride : NULL;
    view->suboffsets = NULL;
    view->internal   = NULL;
    return 0;
}

static void synth_byte_buffer_owner_releasebuffer(PyObject * exporter, Py_buffer * view) {
    (void) exporter;
    (void) view;
}

static void synth_byte_buffer_owner_dealloc(PyObject * object) {
    synth_byte_buffer_owner_t * owner = (synth_byte_buffer_owner_t *) object;
    PyTypeObject *              type  = Py_TYPE(object);
    if (owner->buffer != NULL) {
        synth_api.byte_buffer_free(owner->buffer);
        owner->buffer = NULL;
    }
    PyObject_Free(object);
    Py_DECREF((PyObject *) type);
}

static PyType_Slot synth_byte_buffer_owner_slots[] = {
    { Py_tp_new,           PyType_GenericNew                     },
    { Py_tp_dealloc,       synth_byte_buffer_owner_dealloc       },
    { Py_bf_getbuffer,     synth_byte_buffer_owner_getbuffer     },
    { Py_bf_releasebuffer, synth_byte_buffer_owner_releasebuffer },
    { 0,                   NULL                                  },
};

static PyType_Spec synth_byte_buffer_owner_spec = {
    "synthesize_cpp._native._ByteBuffer", sizeof(synth_byte_buffer_owner_t), 0, Py_TPFLAGS_DEFAULT,
    synth_byte_buffer_owner_slots,
};

static PyObject * synth_build_load_info(PyObject * path_object) {
    return Py_BuildValue("{s:O,s:I,s:(III)}", "path", path_object, "abi_version", synth_api.abi_version, "version",
                         synth_api.version_major, synth_api.version_minor, synth_api.version_patch);
}

static int synth_path_is_absolute(const char * path) {
#if defined(_WIN32)
    const unsigned char first = (unsigned char) path[0];
    return (first != '\0' && path[1] == ':' && (path[2] == '\\' || path[2] == '/')) ||
           (path[0] == '\\' && path[1] == '\\');
#else
    return path[0] == '/';
#endif
}

static PyObject * synth_python_limited_api_hex(PyObject * self, PyObject * ignored) {
    (void) self;
    (void) ignored;
    return PyLong_FromUnsignedLong((unsigned long) Py_LIMITED_API);
}

static PyObject * synth_python_load(PyObject * self, PyObject * args) {
    PyObject *                                     path_object;
    PyObject *                                     path_bytes = NULL;
    const char *                                   path;
    size_t                                         path_size;
    synth_library_handle_t                         candidate                             = NULL;
    synth_abi_version_fn                           abi_version                           = NULL;
    synth_status_string_fn                         status_string                         = NULL;
    synth_diagnostic_sink_init_fn                  diagnostic_sink_init                  = NULL;
    synth_version_init_fn                          version_init                          = NULL;
    synth_get_version_fn                           get_version                           = NULL;
    synth_backend_device_count_fn                  backend_device_count                  = NULL;
    synth_backend_device_init_fn                   backend_device_init                   = NULL;
    synth_backend_device_get_fn                    backend_device_get                    = NULL;
    synth_backend_available_fn                     backend_available                     = NULL;
    synth_backend_load_default_fn                  backend_load_default                  = NULL;
    synth_model_get_device_fn                      model_get_device                      = NULL;
    synth_model_load_params_init_fn                model_load_params_init                = NULL;
    synth_model_load_fn                            model_load                            = NULL;
    synth_model_free_fn                            model_free                            = NULL;
    synth_context_create_fn                        context_create                        = NULL;
    synth_context_free_fn                          context_free                          = NULL;
    synth_model_capabilities_init_fn               model_capabilities_init               = NULL;
    synth_model_get_capabilities_fn                model_get_capabilities                = NULL;
    synth_preset_voice_init_fn                     preset_voice_init                     = NULL;
    synth_model_get_preset_voice_count_fn          model_get_preset_voice_count          = NULL;
    synth_model_get_preset_voice_fn                model_get_preset_voice                = NULL;
    synth_language_capability_init_fn              language_capability_init              = NULL;
    synth_model_get_language_count_fn              model_get_language_count              = NULL;
    synth_model_get_language_fn                    model_get_language                    = NULL;
    synth_voice_profile_capabilities_init_fn       voice_profile_capabilities_init       = NULL;
    synth_model_get_voice_profile_capabilities_fn  model_get_voice_profile_capabilities  = NULL;
    synth_voice_reference_init_fn                  voice_reference_init                  = NULL;
    synth_voice_reference_params_init_fn           voice_reference_params_init           = NULL;
    synth_voice_profile_create_from_reference_fn   voice_profile_create_from_reference   = NULL;
    synth_voice_description_params_init_fn         voice_description_params_init         = NULL;
    synth_voice_profile_create_from_description_fn voice_profile_create_from_description = NULL;
    synth_voice_random_params_init_fn              voice_random_params_init              = NULL;
    synth_voice_profile_create_random_fn           voice_profile_create_random           = NULL;
    synth_voice_profile_load_params_init_fn        voice_profile_load_params_init        = NULL;
    synth_voice_profile_serialize_params_init_fn   voice_profile_serialize_params_init   = NULL;
    synth_voice_profile_load_from_memory_fn        voice_profile_load_from_memory        = NULL;
    synth_voice_profile_serialize_fn               voice_profile_serialize               = NULL;
    synth_byte_buffer_free_fn                      byte_buffer_free                      = NULL;
    synth_voice_profile_free_fn                    voice_profile_free                    = NULL;
    synth_request_init_fn                          request_init                          = NULL;
    synth_result_init_fn                           result_init                           = NULL;
    synth_synthesize_to_buffer_fn                  synthesize_to_buffer                  = NULL;
    synth_audio_buffer_free_fn                     audio_buffer_free                     = NULL;
    synth_version_t                                version;
    char *                                         stored_path         = NULL;
    static const char * const                      remaining_symbols[] = {
        "synth_audio_sink_init",
        "synth_backend_load_from_dir",
        "synth_synthesize",
    };
    size_t index;

    (void) self;
    if (!PyArg_ParseTuple(args, "U:load", &path_object)) {
        return NULL;
    }
    path_bytes = PyUnicode_AsUTF8String(path_object);
    if (path_bytes == NULL) {
        return NULL;
    }
    path = PyBytes_AsString(path_bytes);
    if (path == NULL) {
        Py_DECREF(path_bytes);
        return NULL;
    }
    if (!synth_path_is_absolute(path)) {
        PyErr_SetString(PyExc_RuntimeError, "native library path must be absolute");
        Py_DECREF(path_bytes);
        return NULL;
    }

    if (synth_api.library != NULL) {
        if (strcmp(synth_api.path, path) == 0) {
            Py_DECREF(path_bytes);
            return synth_build_load_info(path_object);
        }
        PyErr_Format(PyExc_RuntimeError, "native library '%s' is already loaded; cannot load '%s'", synth_api.path,
                     path);
        Py_DECREF(path_bytes);
        return NULL;
    }

    candidate = synth_open_library(path);
    if (candidate == NULL) {
        synth_set_open_error(path);
        Py_DECREF(path_bytes);
        return NULL;
    }

    if (synth_resolve_symbol(candidate, path, "synth_abi_version", &abi_version, sizeof(abi_version)) < 0) {
        goto fail;
    }
    if (abi_version() != SYNTH_ABI_VERSION) {
        PyErr_Format(PyExc_RuntimeError, "native library '%s' has ABI %u; expected ABI %u", path,
                     (unsigned int) abi_version(), (unsigned int) SYNTH_ABI_VERSION);
        goto fail;
    }

    if (synth_resolve_symbol(candidate, path, "synth_status_string", &status_string, sizeof(status_string)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_version_init", &version_init, sizeof(version_init)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_get_version", &get_version, sizeof(get_version)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_diagnostic_sink_init", &diagnostic_sink_init,
                             sizeof(diagnostic_sink_init)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_backend_device_count", &backend_device_count,
                             sizeof(backend_device_count)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_backend_device_init", &backend_device_init,
                             sizeof(backend_device_init)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_backend_device_get", &backend_device_get,
                             sizeof(backend_device_get)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_backend_available", &backend_available,
                             sizeof(backend_available)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_backend_load_default", &backend_load_default,
                             sizeof(backend_load_default)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_model_get_device", &model_get_device, sizeof(model_get_device)) <
            0 ||
        synth_resolve_symbol(candidate, path, "synth_model_load_params_init", &model_load_params_init,
                             sizeof(model_load_params_init)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_model_load", &model_load, sizeof(model_load)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_model_free", &model_free, sizeof(model_free)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_context_create", &context_create, sizeof(context_create)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_context_free", &context_free, sizeof(context_free)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_model_capabilities_init", &model_capabilities_init,
                             sizeof(model_capabilities_init)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_model_get_capabilities", &model_get_capabilities,
                             sizeof(model_get_capabilities)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_preset_voice_init", &preset_voice_init,
                             sizeof(preset_voice_init)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_model_get_preset_voice_count", &model_get_preset_voice_count,
                             sizeof(model_get_preset_voice_count)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_model_get_preset_voice", &model_get_preset_voice,
                             sizeof(model_get_preset_voice)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_language_capability_init", &language_capability_init,
                             sizeof(language_capability_init)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_model_get_language_count", &model_get_language_count,
                             sizeof(model_get_language_count)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_model_get_language", &model_get_language,
                             sizeof(model_get_language)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_voice_profile_capabilities_init", &voice_profile_capabilities_init,
                             sizeof(voice_profile_capabilities_init)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_model_get_voice_profile_capabilities",
                             &model_get_voice_profile_capabilities, sizeof(model_get_voice_profile_capabilities)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_voice_reference_init", &voice_reference_init,
                             sizeof(voice_reference_init)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_voice_reference_params_init", &voice_reference_params_init,
                             sizeof(voice_reference_params_init)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_voice_profile_create_from_reference",
                             &voice_profile_create_from_reference, sizeof(voice_profile_create_from_reference)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_voice_description_params_init", &voice_description_params_init,
                             sizeof(voice_description_params_init)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_voice_profile_create_from_description",
                             &voice_profile_create_from_description,
                             sizeof(voice_profile_create_from_description)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_voice_random_params_init", &voice_random_params_init,
                             sizeof(voice_random_params_init)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_voice_profile_create_random", &voice_profile_create_random,
                             sizeof(voice_profile_create_random)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_voice_profile_load_params_init", &voice_profile_load_params_init,
                             sizeof(voice_profile_load_params_init)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_voice_profile_serialize_params_init",
                             &voice_profile_serialize_params_init, sizeof(voice_profile_serialize_params_init)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_voice_profile_load_from_memory", &voice_profile_load_from_memory,
                             sizeof(voice_profile_load_from_memory)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_voice_profile_serialize", &voice_profile_serialize,
                             sizeof(voice_profile_serialize)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_byte_buffer_free", &byte_buffer_free, sizeof(byte_buffer_free)) <
            0 ||
        synth_resolve_symbol(candidate, path, "synth_voice_profile_free", &voice_profile_free,
                             sizeof(voice_profile_free)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_request_init", &request_init, sizeof(request_init)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_result_init", &result_init, sizeof(result_init)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_synthesize_to_buffer", &synthesize_to_buffer,
                             sizeof(synthesize_to_buffer)) < 0 ||
        synth_resolve_symbol(candidate, path, "synth_audio_buffer_free", &audio_buffer_free,
                             sizeof(audio_buffer_free)) < 0) {
        goto fail;
    }
    for (index = 0; index < sizeof(remaining_symbols) / sizeof(remaining_symbols[0]); ++index) {
        if (synth_resolve_symbol(candidate, path, remaining_symbols[index], NULL, 0) < 0) {
            goto fail;
        }
    }

    memset(&version, 0, sizeof(version));
    version_init(&version, sizeof(version));
    if (get_version(&version) != SYNTH_OK) {
        PyErr_Format(PyExc_RuntimeError, "native library '%s' could not report its version", path);
        goto fail;
    }
    if (version.struct_size != sizeof(version) || version.abi_version != SYNTH_ABI_VERSION ||
        version.version_major != SYNTH_VERSION_MAJOR || version.version_minor != SYNTH_VERSION_MINOR ||
        version.version_patch != SYNTH_VERSION_PATCH) {
        PyErr_Format(PyExc_RuntimeError, "native library '%s' reports incompatible version %u.%u.%u (ABI %u)", path,
                     (unsigned int) version.version_major, (unsigned int) version.version_minor,
                     (unsigned int) version.version_patch, (unsigned int) version.abi_version);
        goto fail;
    }

    synth_status_t         backend_status;
    Py_BEGIN_ALLOW_THREADS backend_status = backend_load_default();
    Py_END_ALLOW_THREADS if (backend_status != SYNTH_OK) {
        const char * message = status_string(backend_status);
        PyErr_Format(PyExc_RuntimeError, "native library '%s' synth_backend_load_default failed: %s (%d)", path,
                     message != NULL ? message : "unknown status", (int) backend_status);
        goto fail;
    }

    path_size   = strlen(path);
    stored_path = (char *) malloc(path_size + 1);
    if (stored_path == NULL) {
        PyErr_NoMemory();
        goto fail;
    }
    memcpy(stored_path, path, path_size + 1);

    synth_api.library                               = candidate;
    synth_api.path                                  = stored_path;
    synth_api.abi_version                           = version.abi_version;
    synth_api.version_major                         = version.version_major;
    synth_api.version_minor                         = version.version_minor;
    synth_api.version_patch                         = version.version_patch;
    synth_api.status_string                         = status_string;
    synth_api.diagnostic_sink_init                  = diagnostic_sink_init;
    synth_api.backend_device_count                  = backend_device_count;
    synth_api.backend_device_init                   = backend_device_init;
    synth_api.backend_device_get                    = backend_device_get;
    synth_api.backend_available                     = backend_available;
    synth_api.model_get_device                      = model_get_device;
    synth_api.model_load_params_init                = model_load_params_init;
    synth_api.model_load                            = model_load;
    synth_api.model_free                            = model_free;
    synth_api.context_create                        = context_create;
    synth_api.context_free                          = context_free;
    synth_api.model_capabilities_init               = model_capabilities_init;
    synth_api.model_get_capabilities                = model_get_capabilities;
    synth_api.preset_voice_init                     = preset_voice_init;
    synth_api.model_get_preset_voice_count          = model_get_preset_voice_count;
    synth_api.model_get_preset_voice                = model_get_preset_voice;
    synth_api.language_capability_init              = language_capability_init;
    synth_api.model_get_language_count              = model_get_language_count;
    synth_api.model_get_language                    = model_get_language;
    synth_api.voice_profile_capabilities_init       = voice_profile_capabilities_init;
    synth_api.model_get_voice_profile_capabilities  = model_get_voice_profile_capabilities;
    synth_api.voice_reference_init                  = voice_reference_init;
    synth_api.voice_reference_params_init           = voice_reference_params_init;
    synth_api.voice_profile_create_from_reference   = voice_profile_create_from_reference;
    synth_api.voice_description_params_init         = voice_description_params_init;
    synth_api.voice_profile_create_from_description = voice_profile_create_from_description;
    synth_api.voice_random_params_init              = voice_random_params_init;
    synth_api.voice_profile_create_random           = voice_profile_create_random;
    synth_api.voice_profile_load_params_init        = voice_profile_load_params_init;
    synth_api.voice_profile_serialize_params_init   = voice_profile_serialize_params_init;
    synth_api.voice_profile_load_from_memory        = voice_profile_load_from_memory;
    synth_api.voice_profile_serialize               = voice_profile_serialize;
    synth_api.byte_buffer_free                      = byte_buffer_free;
    synth_api.voice_profile_free                    = voice_profile_free;
    synth_api.request_init                          = request_init;
    synth_api.result_init                           = result_init;
    synth_api.synthesize_to_buffer                  = synthesize_to_buffer;
    synth_api.audio_buffer_free                     = audio_buffer_free;

    Py_DECREF(path_bytes);
    return synth_build_load_info(path_object);

fail:
    synth_close_library(candidate);
    free(stored_path);
    Py_DECREF(path_bytes);
    return NULL;
}

static PyObject * synth_python_backend_available(PyObject * self, PyObject * args) {
    unsigned long request;
    (void) self;

    if (!PyArg_ParseTuple(args, "k:backend_available", &request)) {
        return NULL;
    }
    if (request > UINT32_MAX) {
        PyErr_SetString(PyExc_OverflowError, "backend request does not fit uint32_t");
        return NULL;
    }
    if (synth_api.library == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "native library has not been loaded");
        return NULL;
    }
    return PyBool_FromLong(synth_api.backend_available((synth_backend_request_t) request) != SYNTH_FALSE);
}

static PyObject * synth_python_backend_devices(PyObject * self, PyObject * ignored) {
    uint32_t   count;
    uint32_t   index;
    PyObject * result;
    (void) self;
    (void) ignored;

    if (synth_api.library == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "native library has not been loaded");
        return NULL;
    }
    count = synth_api.backend_device_count();
    if ((uint64_t) count > (uint64_t) PY_SSIZE_T_MAX) {
        PyErr_SetString(PyExc_OverflowError, "native device count is too large");
        return NULL;
    }
    result = PyTuple_New((Py_ssize_t) count);
    if (result == NULL) {
        return NULL;
    }

    for (index = 0; index < count; ++index) {
        synth_backend_device_t device;
        synth_status_t         status;
        PyObject *             value;

        synth_api.backend_device_init(&device, sizeof(device));
        status = synth_api.backend_device_get(index, &device);
        if (status != SYNTH_OK) {
            const char * message = synth_api.status_string(status);
            PyErr_Format(PyExc_RuntimeError, "synth_backend_device_get(%u) failed: %s (%d)", (unsigned int) index,
                         message != NULL ? message : "unknown status", (int) status);
            Py_DECREF(result);
            return NULL;
        }
        if (device.name == NULL || device.description == NULL || device.kind == NULL) {
            PyErr_Format(PyExc_RuntimeError, "synth_backend_device_get(%u) returned a null required string",
                         (unsigned int) index);
            Py_DECREF(result);
            return NULL;
        }
        value = Py_BuildValue("{s:I,s:s,s:s,s:s,s:z,s:K,s:K,s:I,s:I}", "index", index, "name", device.name,
                              "description", device.description, "kind", device.kind, "device_id", device.device_id,
                              "memory_total", (unsigned long long) device.memory_total, "memory_free",
                              (unsigned long long) device.memory_free, "device_type", (unsigned int) device.device_type,
                              "flags", (unsigned int) device.flags);
        if (value == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        if (PyTuple_SetItem(result, (Py_ssize_t) index, value) < 0) {
            Py_DECREF(value);
            Py_DECREF(result);
            return NULL;
        }
    }
    return result;
}

static PyObject * synth_python_model_load(PyObject * self, PyObject * args) {
    PyObject *                path_object;
    PyObject *                path_bytes;
    const char *              path;
    unsigned long             backend;
    int                       device_index;
    PyObject *                diagnostics;
    synth_model_load_params_t params;
    synth_diagnostic_sink_t   diagnostic_sink;
    synth_python_callback_t   diagnostic_state = { 0 };
    synth_model_t *           model            = NULL;
    synth_status_t            status;
    synth_model_owner_t *     owner;
    PyObject *                capsule;
    (void) self;

    if (!PyArg_ParseTuple(args, "UkiO:model_load", &path_object, &backend, &device_index, &diagnostics)) {
        return NULL;
    }
    if (synth_api.library == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "native library has not been loaded");
        return NULL;
    }
    if (backend > UINT32_MAX) {
        PyErr_SetString(PyExc_OverflowError, "backend request does not fit uint32_t");
        return NULL;
    }
    if (diagnostics != Py_None && !PyCallable_Check(diagnostics)) {
        PyErr_SetString(PyExc_TypeError, "diagnostics must be callable or None");
        return NULL;
    }
    path_bytes = PyUnicode_AsUTF8String(path_object);
    if (path_bytes == NULL) {
        return NULL;
    }
    path = PyBytes_AsString(path_bytes);
    if (path == NULL) {
        Py_DECREF(path_bytes);
        return NULL;
    }

    synth_api.model_load_params_init(&params, sizeof(params));
    params.backend      = (synth_backend_request_t) backend;
    params.device_index = (int32_t) device_index;
    if (diagnostics != Py_None) {
        synth_api.diagnostic_sink_init(&diagnostic_sink, sizeof(diagnostic_sink));
        diagnostic_state.callback = diagnostics;
        diagnostic_sink.emit      = synth_python_emit_diagnostic;
        diagnostic_sink.user_data = &diagnostic_state;
        params.diagnostics        = &diagnostic_sink;
    }
    Py_BEGIN_ALLOW_THREADS status = synth_api.model_load(path, &params, &model);
    Py_END_ALLOW_THREADS   Py_DECREF(path_bytes);
    if (diagnostic_state.exception_type != NULL) {
        if (model != NULL) {
            synth_api.model_free(model);
        }
        synth_restore_callback_exception(&diagnostic_state);
        return NULL;
    }
    if (status != SYNTH_OK) {
        synth_set_status_error("synth_model_load", status);
        return NULL;
    }
    if (model == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "synth_model_load succeeded without returning a Model");
        return NULL;
    }

    owner = (synth_model_owner_t *) calloc(1, sizeof(*owner));
    if (owner == NULL) {
        synth_api.model_free(model);
        return PyErr_NoMemory();
    }
    owner->model = model;
    capsule      = PyCapsule_New(owner, SYNTH_MODEL_CAPSULE, synth_model_capsule_destructor);
    if (capsule == NULL) {
        synth_api.model_free(model);
        free(owner);
        return NULL;
    }
    return capsule;
}

static PyObject * synth_build_model_device(synth_model_t * model) {
    synth_backend_device_t device;
    synth_status_t         status;
    synth_api.backend_device_init(&device, sizeof(device));
    status = synth_api.model_get_device(model, &device);
    if (status != SYNTH_OK) {
        synth_set_status_error("synth_model_get_device", status);
        return NULL;
    }
    if (device.name == NULL || device.description == NULL || device.kind == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "synth_model_get_device returned a null required string");
        return NULL;
    }
    return Py_BuildValue("{s:O,s:s,s:s,s:s,s:z,s:K,s:K,s:I,s:I}", "index", Py_None, "name", device.name, "description",
                         device.description, "kind", device.kind, "device_id", device.device_id, "memory_total",
                         (unsigned long long) device.memory_total, "memory_free",
                         (unsigned long long) device.memory_free, "device_type", (unsigned int) device.device_type,
                         "flags", (unsigned int) device.flags);
}

static PyObject * synth_build_model_capabilities(synth_model_t * model) {
    synth_model_capabilities_t capabilities;
    synth_status_t             status;
    synth_api.model_capabilities_init(&capabilities, sizeof(capabilities));
    status = synth_api.model_get_capabilities(model, &capabilities);
    if (status != SYNTH_OK) {
        synth_set_status_error("synth_model_get_capabilities", status);
        return NULL;
    }
    return Py_BuildValue("{s:I,s:I,s:I,s:I,s:d,s:d,s:K,s:K}", "input_flags", (unsigned int) capabilities.input_flags,
                         "capability_flags", (unsigned int) capabilities.capability_flags, "output_sample_rate",
                         (unsigned int) capabilities.output_sample_rate, "output_channel_count",
                         (unsigned int) capabilities.output_channel_count, "min_speaking_rate",
                         (double) capabilities.min_speaking_rate, "max_speaking_rate",
                         (double) capabilities.max_speaking_rate, "max_input_tokens",
                         (unsigned long long) capabilities.max_input_tokens, "max_output_frames",
                         (unsigned long long) capabilities.max_output_frames);
}

static PyObject * synth_build_preset_voices(synth_model_t * model) {
    uint64_t       count = 0;
    uint64_t       index;
    synth_status_t status = synth_api.model_get_preset_voice_count(model, &count);
    PyObject *     result;
    if (status != SYNTH_OK) {
        synth_set_status_error("synth_model_get_preset_voice_count", status);
        return NULL;
    }
    if (count > (uint64_t) PY_SSIZE_T_MAX) {
        PyErr_SetString(PyExc_OverflowError, "Preset Voice count is too large");
        return NULL;
    }
    result = PyTuple_New((Py_ssize_t) count);
    if (result == NULL) {
        return NULL;
    }
    for (index = 0; index < count; ++index) {
        synth_preset_voice_t voice;
        PyObject *           id;
        PyObject *           display_name;
        PyObject *           value;
        synth_api.preset_voice_init(&voice, sizeof(voice));
        status = synth_api.model_get_preset_voice(model, index, &voice);
        if (status != SYNTH_OK) {
            synth_set_status_error("synth_model_get_preset_voice", status);
            Py_DECREF(result);
            return NULL;
        }
        if (voice.id == NULL || voice.id_size == 0 || voice.id_size > (uint64_t) PY_SSIZE_T_MAX ||
            (voice.display_name == NULL && voice.display_name_size != 0) ||
            voice.display_name_size > (uint64_t) PY_SSIZE_T_MAX) {
            PyErr_SetString(PyExc_RuntimeError, "synth_model_get_preset_voice returned invalid string storage");
            Py_DECREF(result);
            return NULL;
        }
        id = PyUnicode_DecodeUTF8(voice.id, (Py_ssize_t) voice.id_size, "strict");
        if (id == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        if (voice.display_name == NULL) {
            display_name = Py_None;
            Py_INCREF(display_name);
        } else {
            display_name = PyUnicode_DecodeUTF8(voice.display_name, (Py_ssize_t) voice.display_name_size, "strict");
        }
        if (display_name == NULL) {
            Py_DECREF(id);
            Py_DECREF(result);
            return NULL;
        }
        value =
            Py_BuildValue("{s:O,s:O,s:I}", "id", id, "display_name", display_name, "flags", (unsigned int) voice.flags);
        Py_DECREF(id);
        Py_DECREF(display_name);
        if (value == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        if (PyTuple_SetItem(result, (Py_ssize_t) index, value) < 0) {
            Py_DECREF(value);
            Py_DECREF(result);
            return NULL;
        }
    }
    return result;
}

static PyObject * synth_build_languages(synth_model_t * model) {
    uint64_t       count = 0;
    uint64_t       index;
    synth_status_t status = synth_api.model_get_language_count(model, &count);
    PyObject *     result;
    if (status != SYNTH_OK) {
        synth_set_status_error("synth_model_get_language_count", status);
        return NULL;
    }
    if (count > (uint64_t) PY_SSIZE_T_MAX) {
        PyErr_SetString(PyExc_OverflowError, "Language Capability count is too large");
        return NULL;
    }
    result = PyTuple_New((Py_ssize_t) count);
    if (result == NULL) {
        return NULL;
    }
    for (index = 0; index < count; ++index) {
        synth_language_capability_t language;
        PyObject *                  tag;
        PyObject *                  value;
        synth_api.language_capability_init(&language, sizeof(language));
        status = synth_api.model_get_language(model, index, &language);
        if (status != SYNTH_OK) {
            synth_set_status_error("synth_model_get_language", status);
            Py_DECREF(result);
            return NULL;
        }
        if (language.tag == NULL || language.tag_size == 0 || language.tag_size > (uint64_t) PY_SSIZE_T_MAX) {
            PyErr_SetString(PyExc_RuntimeError, "synth_model_get_language returned invalid string storage");
            Py_DECREF(result);
            return NULL;
        }
        tag = PyUnicode_DecodeUTF8(language.tag, (Py_ssize_t) language.tag_size, "strict");
        if (tag == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        value = Py_BuildValue("{s:O,s:I}", "tag", tag, "flags", (unsigned int) language.flags);
        Py_DECREF(tag);
        if (value == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        if (PyTuple_SetItem(result, (Py_ssize_t) index, value) < 0) {
            Py_DECREF(value);
            Py_DECREF(result);
            return NULL;
        }
    }
    return result;
}

static PyObject * synth_build_voice_profile_capabilities(synth_model_t * model) {
    synth_voice_profile_capabilities_t capabilities;
    synth_status_t                     status;
    PyObject *                         schema           = NULL;
    PyObject *                         compatibility_id = NULL;
    PyObject *                         result;
    synth_api.voice_profile_capabilities_init(&capabilities, sizeof(capabilities));
    status = synth_api.model_get_voice_profile_capabilities(model, &capabilities);
    if (status != SYNTH_OK) {
        synth_set_status_error("synth_model_get_voice_profile_capabilities", status);
        return NULL;
    }
    if ((capabilities.profile_schema == NULL) != (capabilities.profile_schema_size == 0) ||
        capabilities.profile_schema_size > (uint64_t) PY_SSIZE_T_MAX) {
        PyErr_SetString(PyExc_RuntimeError, "Voice Profile capabilities contain invalid schema storage");
        return NULL;
    }
    if (capabilities.profile_schema == NULL) {
        schema = Py_None;
        Py_INCREF(schema);
    } else {
        schema =
            PyUnicode_DecodeUTF8(capabilities.profile_schema, (Py_ssize_t) capabilities.profile_schema_size, "strict");
    }
    if (schema == NULL) {
        return NULL;
    }
    compatibility_id = PyBytes_FromStringAndSize((const char *) capabilities.profile_compatibility_id, 32);
    if (compatibility_id == NULL) {
        Py_DECREF(schema);
        return NULL;
    }
    result = Py_BuildValue(
        "{s:I,s:I,s:I,s:I,s:K,s:O,s:I,s:O,s:I,s:I,s:K,s:K,s:K}", "source_flags",
        (unsigned int) capabilities.source_flags, "reference_transcript",
        (unsigned int) capabilities.reference_transcript, "reference_language",
        (unsigned int) capabilities.reference_language, "description_language",
        (unsigned int) capabilities.description_language, "max_reference_count",
        (unsigned long long) capabilities.max_reference_count, "profile_schema", schema, "profile_schema_version",
        (unsigned int) capabilities.profile_schema_version, "profile_compatibility_id", compatibility_id,
        "reference_target_sample_rate", (unsigned int) capabilities.reference_target_sample_rate,
        "reference_target_channel_count", (unsigned int) capabilities.reference_target_channel_count,
        "min_reference_frames_per_clip", (unsigned long long) capabilities.min_reference_frames_per_clip,
        "max_reference_frames_per_clip", (unsigned long long) capabilities.max_reference_frames_per_clip,
        "max_reference_total_frames", (unsigned long long) capabilities.max_reference_total_frames);
    Py_DECREF(compatibility_id);
    Py_DECREF(schema);
    return result;
}

static PyObject * synth_python_model_metadata(PyObject * self, PyObject * args) {
    PyObject *            model_capsule;
    synth_model_owner_t * owner;
    PyObject *            device;
    PyObject *            capabilities;
    PyObject *            voices;
    PyObject *            languages;
    PyObject *            voice_profile_capabilities;
    PyObject *            result;
    (void) self;

    if (!PyArg_ParseTuple(args, "O:model_metadata", &model_capsule)) {
        return NULL;
    }
    owner = synth_model_owner_from_capsule(model_capsule, 1);
    if (owner == NULL) {
        return NULL;
    }
    device = synth_build_model_device(owner->model);
    if (device == NULL) {
        return NULL;
    }
    capabilities = synth_build_model_capabilities(owner->model);
    if (capabilities == NULL) {
        Py_DECREF(device);
        return NULL;
    }
    voices = synth_build_preset_voices(owner->model);
    if (voices == NULL) {
        Py_DECREF(capabilities);
        Py_DECREF(device);
        return NULL;
    }
    languages = synth_build_languages(owner->model);
    if (languages == NULL) {
        Py_DECREF(voices);
        Py_DECREF(capabilities);
        Py_DECREF(device);
        return NULL;
    }
    voice_profile_capabilities = synth_build_voice_profile_capabilities(owner->model);
    if (voice_profile_capabilities == NULL) {
        Py_DECREF(languages);
        Py_DECREF(voices);
        Py_DECREF(capabilities);
        Py_DECREF(device);
        return NULL;
    }
    result = Py_BuildValue("{s:O,s:O,s:O,s:O,s:O}", "device", device, "capabilities", capabilities, "preset_voices",
                           voices, "languages", languages, "voice_profile_capabilities", voice_profile_capabilities);
    Py_DECREF(voice_profile_capabilities);
    Py_DECREF(languages);
    Py_DECREF(voices);
    Py_DECREF(capabilities);
    Py_DECREF(device);
    return result;
}

static PyObject * synth_python_model_close(PyObject * self, PyObject * args) {
    PyObject *            capsule;
    synth_model_owner_t * owner;
    synth_model_t *       model = NULL;
    (void) self;

    if (!PyArg_ParseTuple(args, "O:model_close", &capsule)) {
        return NULL;
    }
    owner = synth_model_owner_from_capsule(capsule, 0);
    if (owner == NULL) {
        return NULL;
    }
    if (!owner->close_requested) {
        if (PyCapsule_SetName(capsule, SYNTH_CLOSED_MODEL_CAPSULE) < 0) {
            return NULL;
        }
        owner->close_requested = 1;
    }
    if (owner->context_count == 0 && owner->profile_count == 0 && owner->model != NULL) {
        model        = owner->model;
        owner->model = NULL;
    }
    if (model != NULL) {
        Py_BEGIN_ALLOW_THREADS synth_api.model_free(model);
        Py_END_ALLOW_THREADS
    }
    Py_RETURN_NONE;
}

static PyObject * synth_python_context_create(PyObject * self, PyObject * args) {
    PyObject *              model_capsule;
    synth_model_owner_t *   model_owner;
    synth_context_owner_t * owner;
    synth_context_t *       context        = NULL;
    synth_model_t *         deferred_model = NULL;
    synth_status_t          status;
    PyObject *              capsule;
    (void) self;

    if (!PyArg_ParseTuple(args, "O:context_create", &model_capsule)) {
        return NULL;
    }
    model_owner = synth_model_owner_from_capsule(model_capsule, 1);
    if (model_owner == NULL) {
        return NULL;
    }
    owner = (synth_context_owner_t *) calloc(1, sizeof(*owner));
    if (owner == NULL) {
        return PyErr_NoMemory();
    }

    owner->model_owner   = model_owner;
    owner->model_capsule = model_capsule;
    ++model_owner->context_count;
    Py_INCREF(model_capsule);
    Py_BEGIN_ALLOW_THREADS status = synth_api.context_create(model_owner->model, &context);
    Py_END_ALLOW_THREADS if (status != SYNTH_OK || context == NULL) {
        --model_owner->context_count;
        if (model_owner->context_count == 0 && model_owner->profile_count == 0 && model_owner->close_requested &&
            model_owner->model != NULL) {
            deferred_model     = model_owner->model;
            model_owner->model = NULL;
        }
        Py_DECREF(model_capsule);
        free(owner);
        if (deferred_model != NULL) {
            synth_api.model_free(deferred_model);
        }
        if (status != SYNTH_OK) {
            synth_set_status_error("synth_context_create", status);
        } else {
            PyErr_SetString(PyExc_RuntimeError, "synth_context_create succeeded without returning a Context");
        }
        return NULL;
    }

    owner->context = context;
    capsule        = PyCapsule_New(owner, SYNTH_CONTEXT_CAPSULE, synth_context_capsule_destructor);
    if (capsule == NULL) {
        synth_release_context_owner(owner);
        free(owner);
        return NULL;
    }
    return capsule;
}

static PyObject * synth_python_context_close(PyObject * self, PyObject * args) {
    PyObject *              capsule;
    synth_context_owner_t * owner;
    (void) self;

    if (!PyArg_ParseTuple(args, "O:context_close", &capsule)) {
        return NULL;
    }
    owner = synth_context_owner_from_capsule(capsule, 0);
    if (owner == NULL) {
        return NULL;
    }
    if (PyCapsule_IsValid(capsule, SYNTH_CONTEXT_CAPSULE)) {
        if (PyCapsule_SetName(capsule, SYNTH_CLOSED_CONTEXT_CAPSULE) < 0) {
            return NULL;
        }
        owner->close_requested = 1;
    }
    if (owner->context != NULL && owner->active_count == 0) {
        synth_release_context_owner(owner);
    }
    Py_RETURN_NONE;
}

static int synth_prepare_python_diagnostics(PyObject *                       callback,
                                            synth_diagnostic_sink_t *        sink,
                                            synth_python_callback_t *        state,
                                            const synth_diagnostic_sink_t ** out_sink) {
    *out_sink = NULL;
    if (callback == Py_None) {
        return 0;
    }
    if (!PyCallable_Check(callback)) {
        PyErr_SetString(PyExc_TypeError, "diagnostics must be callable or None");
        return -1;
    }
    synth_api.diagnostic_sink_init(sink, sizeof(*sink));
    state->callback = callback;
    sink->emit      = synth_python_emit_diagnostic;
    sink->user_data = state;
    *out_sink       = sink;
    return 0;
}

static void synth_cancel_profile_reservation(synth_model_owner_t * model_owner) {
    synth_model_t * deferred_model = NULL;
    if (model_owner->profile_count > 0) {
        --model_owner->profile_count;
    }
    if (model_owner->context_count == 0 && model_owner->profile_count == 0 && model_owner->close_requested &&
        model_owner->model != NULL) {
        deferred_model     = model_owner->model;
        model_owner->model = NULL;
    }
    if (deferred_model != NULL) {
        synth_api.model_free(deferred_model);
    }
}

static PyObject * synth_wrap_voice_profile(PyObject *              model_capsule,
                                           synth_model_owner_t *   model_owner,
                                           synth_voice_profile_t * profile) {
    synth_voice_profile_owner_t * owner;
    PyObject *                    capsule;
    owner = (synth_voice_profile_owner_t *) calloc(1, sizeof(*owner));
    if (owner == NULL) {
        synth_api.voice_profile_free(profile);
        synth_cancel_profile_reservation(model_owner);
        return PyErr_NoMemory();
    }
    owner->profile       = profile;
    owner->model_owner   = model_owner;
    owner->model_capsule = model_capsule;
    Py_INCREF(model_capsule);
    capsule = PyCapsule_New(owner, SYNTH_VOICE_PROFILE_CAPSULE, synth_voice_profile_capsule_destructor);
    if (capsule == NULL) {
        synth_release_voice_profile_owner(owner);
        free(owner);
        return NULL;
    }
    return capsule;
}

static PyObject * synth_finish_profile_preparation(const char *              operation,
                                                   PyObject *                model_capsule,
                                                   synth_model_owner_t *     model_owner,
                                                   synth_voice_profile_t *   profile,
                                                   synth_status_t            status,
                                                   synth_python_callback_t * diagnostic_state) {
    if (diagnostic_state->exception_type != NULL) {
        if (profile != NULL) {
            synth_api.voice_profile_free(profile);
        }
        synth_cancel_profile_reservation(model_owner);
        synth_restore_callback_exception(diagnostic_state);
        return NULL;
    }
    if (status != SYNTH_OK) {
        if (profile != NULL) {
            synth_api.voice_profile_free(profile);
        }
        synth_cancel_profile_reservation(model_owner);
        synth_set_status_error(operation, status);
        return NULL;
    }
    if (profile == NULL) {
        synth_cancel_profile_reservation(model_owner);
        PyErr_Format(PyExc_RuntimeError, "%s succeeded without returning a Voice Profile", operation);
        return NULL;
    }
    return synth_wrap_voice_profile(model_capsule, model_owner, profile);
}

static PyObject * synth_python_voice_profile_close(PyObject * self, PyObject * args) {
    PyObject *                    capsule;
    synth_voice_profile_owner_t * owner;
    (void) self;
    if (!PyArg_ParseTuple(args, "O:voice_profile_close", &capsule)) {
        return NULL;
    }
    owner = synth_voice_profile_owner_from_capsule(capsule, 0);
    if (owner == NULL) {
        return NULL;
    }
    if (PyCapsule_IsValid(capsule, SYNTH_VOICE_PROFILE_CAPSULE)) {
        if (PyCapsule_SetName(capsule, SYNTH_CLOSED_VOICE_PROFILE_CAPSULE) < 0) {
            return NULL;
        }
        owner->close_requested = 1;
    }
    if (owner->profile != NULL && owner->active_count == 0) {
        synth_release_voice_profile_owner(owner);
    }
    Py_RETURN_NONE;
}

static PyObject * synth_python_voice_profile_create_from_description(PyObject * self, PyObject * args) {
    PyObject *                       model_capsule;
    PyObject *                       description;
    PyObject *                       language;
    unsigned long long               seed;
    PyObject *                       diagnostics;
    synth_model_owner_t *            model_owner;
    PyObject *                       description_bytes = NULL;
    PyObject *                       language_bytes    = NULL;
    synth_voice_description_params_t params;
    synth_diagnostic_sink_t          diagnostic_sink;
    const synth_diagnostic_sink_t *  diagnostic_sink_pointer;
    synth_python_callback_t          diagnostic_state = { 0 };
    synth_voice_profile_t *          profile          = NULL;
    synth_status_t                   status;
    (void) self;

    if (!PyArg_ParseTuple(args, "OOOKO:voice_profile_create_from_description", &model_capsule, &description, &language,
                          &seed, &diagnostics)) {
        return NULL;
    }
    model_owner = synth_model_owner_from_capsule(model_capsule, 1);
    if (model_owner == NULL) {
        return NULL;
    }
    if (!PyUnicode_Check(description) || (language != Py_None && !PyUnicode_Check(language))) {
        PyErr_SetString(PyExc_TypeError, "description must be str and language must be str or None");
        return NULL;
    }
    if (synth_prepare_python_diagnostics(diagnostics, &diagnostic_sink, &diagnostic_state, &diagnostic_sink_pointer) <
        0) {
        return NULL;
    }
    description_bytes = PyUnicode_AsUTF8String(description);
    if (description_bytes == NULL) {
        return NULL;
    }
    if (language != Py_None) {
        language_bytes = PyUnicode_AsUTF8String(language);
        if (language_bytes == NULL) {
            Py_DECREF(description_bytes);
            return NULL;
        }
    }
    synth_api.voice_description_params_init(&params, sizeof(params));
    params.description       = PyBytes_AsString(description_bytes);
    params.description_size  = (uint64_t) PyBytes_Size(description_bytes);
    params.language_tag      = language_bytes != NULL ? PyBytes_AsString(language_bytes) : NULL;
    params.language_tag_size = language_bytes != NULL ? (uint64_t) PyBytes_Size(language_bytes) : 0;
    params.seed              = (uint64_t) seed;
    params.diagnostics       = diagnostic_sink_pointer;
    ++model_owner->profile_count;
    Py_BEGIN_ALLOW_THREADS status =
        synth_api.voice_profile_create_from_description(model_owner->model, &params, &profile);
    Py_END_ALLOW_THREADS Py_XDECREF(language_bytes);
    Py_DECREF(description_bytes);
    return synth_finish_profile_preparation("synth_voice_profile_create_from_description", model_capsule, model_owner,
                                            profile, status, &diagnostic_state);
}

static PyObject * synth_python_voice_profile_create_from_reference(PyObject * self, PyObject * args) {
    PyObject *                      model_capsule;
    PyObject *                      references;
    PyObject *                      diagnostics;
    synth_model_owner_t *           model_owner;
    Py_ssize_t                      reference_count;
    synth_voice_reference_t *       descriptors      = NULL;
    Py_buffer *                     sample_views     = NULL;
    PyObject **                     transcript_bytes = NULL;
    PyObject **                     language_bytes   = NULL;
    Py_ssize_t                      index;
    synth_voice_reference_params_t  params;
    synth_diagnostic_sink_t         diagnostic_sink;
    const synth_diagnostic_sink_t * diagnostic_sink_pointer;
    synth_python_callback_t         diagnostic_state = { 0 };
    synth_voice_profile_t *         profile          = NULL;
    synth_status_t                  status;
    PyObject *                      result = NULL;
    (void) self;

    if (!PyArg_ParseTuple(args, "OOO:voice_profile_create_from_reference", &model_capsule, &references, &diagnostics)) {
        return NULL;
    }
    model_owner = synth_model_owner_from_capsule(model_capsule, 1);
    if (model_owner == NULL) {
        return NULL;
    }
    if (!PyTuple_Check(references)) {
        PyErr_SetString(PyExc_TypeError, "references must be a tuple");
        return NULL;
    }
    reference_count = PyTuple_Size(references);
    if (reference_count <= 0) {
        PyErr_SetString(PyExc_ValueError, "at least one Voice Reference is required");
        return NULL;
    }
    if (synth_prepare_python_diagnostics(diagnostics, &diagnostic_sink, &diagnostic_state, &diagnostic_sink_pointer) <
        0) {
        return NULL;
    }
    if ((size_t) reference_count > SIZE_MAX / sizeof(*descriptors) ||
        (size_t) reference_count > SIZE_MAX / sizeof(*sample_views) ||
        (size_t) reference_count > SIZE_MAX / sizeof(*transcript_bytes)) {
        return PyErr_NoMemory();
    }
    descriptors      = (synth_voice_reference_t *) calloc((size_t) reference_count, sizeof(*descriptors));
    sample_views     = (Py_buffer *) calloc((size_t) reference_count, sizeof(*sample_views));
    transcript_bytes = (PyObject **) calloc((size_t) reference_count, sizeof(*transcript_bytes));
    language_bytes   = (PyObject **) calloc((size_t) reference_count, sizeof(*language_bytes));
    if (descriptors == NULL || sample_views == NULL || transcript_bytes == NULL || language_bytes == NULL) {
        PyErr_NoMemory();
        goto cleanup;
    }

    for (index = 0; index < reference_count; ++index) {
        PyObject *    item = PyTuple_GetItem(references, index);
        PyObject *    samples;
        unsigned long sample_rate;
        unsigned long channel_count;
        PyObject *    transcript;
        PyObject *    language;
        uint64_t      scalar_count;
        if (item == NULL || !PyTuple_Check(item) ||
            !PyArg_ParseTuple(item, "OkkOO:VoiceReference", &samples, &sample_rate, &channel_count, &transcript,
                              &language)) {
            goto cleanup;
        }
        if (sample_rate > UINT32_MAX || channel_count > UINT32_MAX || channel_count == 0) {
            PyErr_SetString(PyExc_OverflowError, "Voice Reference format does not fit the C Interface");
            goto cleanup;
        }
        if ((transcript != Py_None && !PyUnicode_Check(transcript)) ||
            (language != Py_None && !PyUnicode_Check(language))) {
            PyErr_SetString(PyExc_TypeError, "Voice Reference transcript and language must be str or None");
            goto cleanup;
        }
        if (PyObject_GetBuffer(samples, &sample_views[index], PyBUF_CONTIG_RO | PyBUF_FORMAT) < 0) {
            goto cleanup;
        }
        if (sample_views[index].ndim != 1 || sample_views[index].itemsize != (Py_ssize_t) sizeof(float) ||
            sample_views[index].len < 0 || sample_views[index].len % (Py_ssize_t) sizeof(float) != 0 ||
            sample_views[index].format == NULL ||
            (strcmp(sample_views[index].format, "f") != 0 && strcmp(sample_views[index].format, "@f") != 0 &&
             strcmp(sample_views[index].format, "=f") != 0) ||
            ((uintptr_t) sample_views[index].buf % _Alignof(float)) != 0) {
            PyErr_SetString(PyExc_TypeError,
                            "Voice Reference samples must be a contiguous, aligned, native float32 buffer");
            goto cleanup;
        }
        scalar_count = (uint64_t) sample_views[index].len / sizeof(float);
        if (scalar_count % channel_count != 0) {
            PyErr_SetString(PyExc_ValueError, "Voice Reference sample count is not divisible by channel_count");
            goto cleanup;
        }
        if (transcript != Py_None) {
            transcript_bytes[index] = PyUnicode_AsUTF8String(transcript);
            if (transcript_bytes[index] == NULL) {
                goto cleanup;
            }
        }
        if (language != Py_None) {
            language_bytes[index] = PyUnicode_AsUTF8String(language);
            if (language_bytes[index] == NULL) {
                goto cleanup;
            }
        }
        synth_api.voice_reference_init(&descriptors[index], sizeof(descriptors[index]));
        descriptors[index].samples       = (const float *) sample_views[index].buf;
        descriptors[index].frame_count   = scalar_count / channel_count;
        descriptors[index].sample_rate   = (uint32_t) sample_rate;
        descriptors[index].channel_count = (uint32_t) channel_count;
        descriptors[index].transcript =
            transcript_bytes[index] != NULL ? PyBytes_AsString(transcript_bytes[index]) : NULL;
        descriptors[index].transcript_size =
            transcript_bytes[index] != NULL ? (uint64_t) PyBytes_Size(transcript_bytes[index]) : 0;
        descriptors[index].language_tag =
            language_bytes[index] != NULL ? PyBytes_AsString(language_bytes[index]) : NULL;
        descriptors[index].language_tag_size =
            language_bytes[index] != NULL ? (uint64_t) PyBytes_Size(language_bytes[index]) : 0;
    }

    synth_api.voice_reference_params_init(&params, sizeof(params));
    params.references       = descriptors;
    params.reference_count  = (uint64_t) reference_count;
    params.reference_stride = sizeof(*descriptors);
    params.diagnostics      = diagnostic_sink_pointer;
    ++model_owner->profile_count;
    Py_BEGIN_ALLOW_THREADS status =
        synth_api.voice_profile_create_from_reference(model_owner->model, &params, &profile);
    Py_END_ALLOW_THREADS result = synth_finish_profile_preparation(
        "synth_voice_profile_create_from_reference", model_capsule, model_owner, profile, status, &diagnostic_state);

cleanup:
    if (sample_views != NULL) {
        for (index = 0; index < reference_count; ++index) {
            if (sample_views[index].obj != NULL) {
                PyBuffer_Release(&sample_views[index]);
            }
        }
    }
    if (transcript_bytes != NULL) {
        for (index = 0; index < reference_count; ++index) {
            Py_XDECREF(transcript_bytes[index]);
        }
    }
    if (language_bytes != NULL) {
        for (index = 0; index < reference_count; ++index) {
            Py_XDECREF(language_bytes[index]);
        }
    }
    free(language_bytes);
    free(transcript_bytes);
    free(sample_views);
    free(descriptors);
    return result;
}

static PyObject * synth_python_voice_profile_create_random(PyObject * self, PyObject * args) {
    PyObject *                      model_capsule;
    unsigned long long              seed;
    PyObject *                      diagnostics;
    synth_model_owner_t *           model_owner;
    synth_voice_random_params_t     params;
    synth_diagnostic_sink_t         diagnostic_sink;
    const synth_diagnostic_sink_t * diagnostic_sink_pointer;
    synth_python_callback_t         diagnostic_state = { 0 };
    synth_voice_profile_t *         profile          = NULL;
    synth_status_t                  status;
    (void) self;

    if (!PyArg_ParseTuple(args, "OKO:voice_profile_create_random", &model_capsule, &seed, &diagnostics)) {
        return NULL;
    }
    model_owner = synth_model_owner_from_capsule(model_capsule, 1);
    if (model_owner == NULL) {
        return NULL;
    }
    if (synth_prepare_python_diagnostics(diagnostics, &diagnostic_sink, &diagnostic_state, &diagnostic_sink_pointer) <
        0) {
        return NULL;
    }
    synth_api.voice_random_params_init(&params, sizeof(params));
    params.seed        = (uint64_t) seed;
    params.diagnostics = diagnostic_sink_pointer;
    ++model_owner->profile_count;
    Py_BEGIN_ALLOW_THREADS status = synth_api.voice_profile_create_random(model_owner->model, &params, &profile);
    Py_END_ALLOW_THREADS return synth_finish_profile_preparation("synth_voice_profile_create_random", model_capsule,
                                                                 model_owner, profile, status, &diagnostic_state);
}

static PyObject * synth_python_voice_profile_load(PyObject * self, PyObject * args) {
    PyObject *                        model_capsule;
    PyObject *                        data;
    PyObject *                        diagnostics;
    synth_model_owner_t *             model_owner;
    Py_buffer                         data_view;
    synth_voice_profile_load_params_t params;
    synth_diagnostic_sink_t           diagnostic_sink;
    const synth_diagnostic_sink_t *   diagnostic_sink_pointer;
    synth_python_callback_t           diagnostic_state = { 0 };
    synth_voice_profile_t *           profile          = NULL;
    synth_status_t                    status;
    (void) self;

    memset(&data_view, 0, sizeof(data_view));
    if (!PyArg_ParseTuple(args, "OOO:voice_profile_load", &model_capsule, &data, &diagnostics)) {
        return NULL;
    }
    model_owner = synth_model_owner_from_capsule(model_capsule, 1);
    if (model_owner == NULL) {
        return NULL;
    }
    if (synth_prepare_python_diagnostics(diagnostics, &diagnostic_sink, &diagnostic_state, &diagnostic_sink_pointer) <
        0) {
        return NULL;
    }
    if (PyObject_GetBuffer(data, &data_view, PyBUF_CONTIG_RO) < 0) {
        return NULL;
    }
    if (data_view.len <= 0) {
        PyBuffer_Release(&data_view);
        PyErr_SetString(PyExc_ValueError, "serialized Voice Profile must not be empty");
        return NULL;
    }
    synth_api.voice_profile_load_params_init(&params, sizeof(params));
    params.data        = (const uint8_t *) data_view.buf;
    params.data_size   = (uint64_t) data_view.len;
    params.diagnostics = diagnostic_sink_pointer;
    ++model_owner->profile_count;
    Py_BEGIN_ALLOW_THREADS status = synth_api.voice_profile_load_from_memory(model_owner->model, &params, &profile);
    Py_END_ALLOW_THREADS   PyBuffer_Release(&data_view);
    return synth_finish_profile_preparation("synth_voice_profile_load_from_memory", model_capsule, model_owner, profile,
                                            status, &diagnostic_state);
}

static PyObject * synth_python_voice_profile_serialize(PyObject * self, PyObject * args) {
    PyObject *                             profile_capsule;
    PyObject *                             diagnostics;
    synth_voice_profile_owner_t *          profile_owner;
    synth_voice_profile_serialize_params_t params;
    synth_diagnostic_sink_t                diagnostic_sink;
    const synth_diagnostic_sink_t *        diagnostic_sink_pointer;
    synth_python_callback_t                diagnostic_state = { 0 };
    synth_byte_buffer_t *                  buffer           = NULL;
    synth_status_t                         status;
    PyObject *                             owner_object;
    synth_byte_buffer_owner_t *            owner;
    (void) self;

    if (!PyArg_ParseTuple(args, "OO:voice_profile_serialize", &profile_capsule, &diagnostics)) {
        return NULL;
    }
    profile_owner = synth_voice_profile_owner_from_capsule(profile_capsule, 1);
    if (profile_owner == NULL) {
        return NULL;
    }
    if (synth_prepare_python_diagnostics(diagnostics, &diagnostic_sink, &diagnostic_state, &diagnostic_sink_pointer) <
        0) {
        return NULL;
    }
    synth_api.voice_profile_serialize_params_init(&params, sizeof(params));
    params.diagnostics = diagnostic_sink_pointer;
    ++profile_owner->active_count;
    Py_BEGIN_ALLOW_THREADS status = synth_api.voice_profile_serialize(profile_owner->profile, &params, &buffer);
    Py_END_ALLOW_THREADS-- profile_owner->active_count;
    if (profile_owner->close_requested && profile_owner->active_count == 0) {
        synth_release_voice_profile_owner(profile_owner);
    }
    if (diagnostic_state.exception_type != NULL) {
        if (buffer != NULL) {
            synth_api.byte_buffer_free(buffer);
        }
        synth_restore_callback_exception(&diagnostic_state);
        return NULL;
    }
    if (status != SYNTH_OK) {
        if (buffer != NULL) {
            synth_api.byte_buffer_free(buffer);
        }
        synth_set_status_error("synth_voice_profile_serialize", status);
        return NULL;
    }
    if (buffer == NULL || buffer->struct_size < sizeof(*buffer) || buffer->data_size == 0 || buffer->data == NULL) {
        if (buffer != NULL) {
            synth_api.byte_buffer_free(buffer);
        }
        PyErr_SetString(PyExc_RuntimeError, "synth_voice_profile_serialize returned invalid owned bytes");
        return NULL;
    }
    owner_object = PyObject_CallNoArgs(synth_byte_buffer_owner_type);
    if (owner_object == NULL) {
        synth_api.byte_buffer_free(buffer);
        return NULL;
    }
    owner         = (synth_byte_buffer_owner_t *) owner_object;
    owner->buffer = buffer;
    return owner_object;
}

static PyObject * synth_decode_result_string(const char * data, uint64_t size, const char * field_name) {
    if (data == NULL) {
        if (size != 0) {
            PyErr_Format(PyExc_RuntimeError, "synthesis result %s has null storage with nonzero size", field_name);
            return NULL;
        }
        Py_RETURN_NONE;
    }
    if (size > (uint64_t) PY_SSIZE_T_MAX) {
        PyErr_Format(PyExc_OverflowError, "synthesis result %s is too large", field_name);
        return NULL;
    }
    return PyUnicode_DecodeUTF8(data, (Py_ssize_t) size, "strict");
}

static PyObject * synth_python_synthesize(PyObject * self, PyObject * args) {
    PyObject *                    context_capsule;
    unsigned long                 input_kind;
    PyObject *                    input;
    PyObject *                    language;
    PyObject *                    voice;
    unsigned long long            seed;
    double                        speaking_rate;
    unsigned long long            max_output_frames;
    PyObject *                    diagnostics;
    PyObject *                    should_cancel;
    PyObject *                    voice_profile;
    synth_context_owner_t *       context_owner;
    synth_voice_profile_owner_t * profile_owner = NULL;
    Py_buffer                     input_view;
    PyObject *                    input_bytes    = NULL;
    PyObject *                    language_bytes = NULL;
    PyObject *                    voice_bytes    = NULL;
    synth_request_t               request;
    synth_diagnostic_sink_t       diagnostic_sink;
    synth_python_callback_t       diagnostic_state   = { 0 };
    synth_python_callback_t       cancellation_state = { 0 };
    synth_result_t                synthesis_result;
    synth_audio_buffer_t *        audio = NULL;
    synth_status_t                status;
    PyObject *                    audio_owner_object = NULL;
    synth_audio_owner_t *         audio_owner;
    PyObject *                    resolved_language = NULL;
    PyObject *                    resolved_voice    = NULL;
    PyObject *                    result            = NULL;
    (void) self;

    memset(&input_view, 0, sizeof(input_view));
    if (!PyArg_ParseTuple(args, "OkOOOKdKOOO:synthesize", &context_capsule, &input_kind, &input, &language, &voice,
                          &seed, &speaking_rate, &max_output_frames, &diagnostics, &should_cancel, &voice_profile)) {
        return NULL;
    }
    context_owner = synth_context_owner_from_capsule(context_capsule, 1);
    if (context_owner == NULL) {
        return NULL;
    }
    if (input_kind > UINT32_MAX) {
        PyErr_SetString(PyExc_OverflowError, "input kind does not fit uint32_t");
        return NULL;
    }
    if ((language != Py_None && !PyUnicode_Check(language)) || (voice != Py_None && !PyUnicode_Check(voice))) {
        PyErr_SetString(PyExc_TypeError, "language and voice must be str or None");
        return NULL;
    }
    if (diagnostics != Py_None && !PyCallable_Check(diagnostics)) {
        PyErr_SetString(PyExc_TypeError, "diagnostics must be callable or None");
        return NULL;
    }
    if (should_cancel != Py_None && !PyCallable_Check(should_cancel)) {
        PyErr_SetString(PyExc_TypeError, "should_cancel must be callable or None");
        return NULL;
    }
    if (voice_profile != Py_None) {
        profile_owner = synth_voice_profile_owner_from_capsule(voice_profile, 1);
        if (profile_owner == NULL) {
            return NULL;
        }
        if (profile_owner->model_owner != context_owner->model_owner) {
            PyErr_SetString(PyExc_ValueError, "Voice Profile belongs to a different Loaded Model");
            return NULL;
        }
    }
    if (input_kind == SYNTH_INPUT_TOKEN_IDS &&
        PyObject_GetBuffer(input, &input_view, PyBUF_CONTIG_RO | PyBUF_FORMAT) < 0) {
        return NULL;
    }
    if (input_kind == SYNTH_INPUT_TOKEN_IDS &&
        (input_view.ndim != 1 || input_view.itemsize != (Py_ssize_t) sizeof(int32_t) || input_view.len < 0 ||
         input_view.len % (Py_ssize_t) sizeof(int32_t) != 0 || input_view.format == NULL ||
         (strcmp(input_view.format, "i") != 0 && strcmp(input_view.format, "@i") != 0 &&
          strcmp(input_view.format, "=i") != 0) ||
         ((uintptr_t) input_view.buf % _Alignof(int32_t)) != 0)) {
        PyErr_SetString(PyExc_TypeError, "tokens must be a contiguous, aligned, native int32 buffer");
        goto fail;
    }
    if (input_kind == SYNTH_INPUT_TEXT_UTF8 || input_kind == SYNTH_INPUT_PHONEMES_UTF8) {
        if (!PyUnicode_Check(input)) {
            PyErr_SetString(PyExc_TypeError, "text and phoneme input must be str");
            goto fail;
        }
        input_bytes = PyUnicode_AsUTF8String(input);
        if (input_bytes == NULL) {
            goto fail;
        }
    } else if (input_kind != SYNTH_INPUT_TOKEN_IDS) {
        PyErr_SetString(PyExc_ValueError, "unknown synthesis input kind");
        goto fail;
    }
    if (language != Py_None) {
        language_bytes = PyUnicode_AsUTF8String(language);
        if (language_bytes == NULL) {
            goto fail;
        }
    }
    if (voice != Py_None) {
        voice_bytes = PyUnicode_AsUTF8String(voice);
        if (voice_bytes == NULL) {
            goto fail;
        }
    }

    synth_api.request_init(&request, sizeof(request));
    request.input_kind = (synth_input_kind_t) input_kind;
    if (input_kind == SYNTH_INPUT_TOKEN_IDS) {
        request.input_data  = input_view.buf;
        request.input_count = (uint64_t) input_view.len / sizeof(int32_t);
    } else {
        request.input_data  = PyBytes_AsString(input_bytes);
        request.input_count = (uint64_t) PyBytes_Size(input_bytes);
    }
    request.language_tag      = language_bytes != NULL ? PyBytes_AsString(language_bytes) : NULL;
    request.language_tag_size = language_bytes != NULL ? (uint64_t) PyBytes_Size(language_bytes) : 0;
    request.voice_id          = voice_bytes != NULL ? PyBytes_AsString(voice_bytes) : NULL;
    request.voice_id_size     = voice_bytes != NULL ? (uint64_t) PyBytes_Size(voice_bytes) : 0;
    request.seed              = (uint64_t) seed;
    request.speaking_rate     = (float) speaking_rate;
    request.max_output_frames = (uint64_t) max_output_frames;
    request.voice_profile     = profile_owner != NULL ? profile_owner->profile : NULL;
    if (diagnostics != Py_None) {
        synth_api.diagnostic_sink_init(&diagnostic_sink, sizeof(diagnostic_sink));
        diagnostic_state.callback = diagnostics;
        diagnostic_sink.emit      = synth_python_emit_diagnostic;
        diagnostic_sink.user_data = &diagnostic_state;
        request.diagnostics       = &diagnostic_sink;
    }
    if (should_cancel != Py_None) {
        cancellation_state.callback = should_cancel;
        request.should_cancel       = synth_python_should_cancel;
        request.cancel_user_data    = &cancellation_state;
    }
    synth_api.result_init(&synthesis_result, sizeof(synthesis_result));

    ++context_owner->active_count;
    if (profile_owner != NULL) {
        ++profile_owner->active_count;
    }
    Py_BEGIN_ALLOW_THREADS status =
        synth_api.synthesize_to_buffer(context_owner->context, &request, &audio, &synthesis_result);
    Py_END_ALLOW_THREADS-- context_owner->active_count;
    if (profile_owner != NULL) {
        --profile_owner->active_count;
    }

    if (input_view.obj != NULL) {
        PyBuffer_Release(&input_view);
        memset(&input_view, 0, sizeof(input_view));
    }
    Py_CLEAR(input_bytes);
    Py_CLEAR(language_bytes);
    Py_CLEAR(voice_bytes);
    if (context_owner->close_requested && context_owner->active_count == 0) {
        synth_release_context_owner(context_owner);
    }
    if (profile_owner != NULL && profile_owner->close_requested && profile_owner->active_count == 0) {
        synth_release_voice_profile_owner(profile_owner);
    }

    if (diagnostic_state.exception_type != NULL || cancellation_state.exception_type != NULL) {
        if (audio != NULL) {
            synth_api.audio_buffer_free(audio);
        }
        if (diagnostic_state.exception_type != NULL) {
            synth_restore_callback_exception(&diagnostic_state);
        } else {
            synth_restore_callback_exception(&cancellation_state);
        }
        return NULL;
    }
    if (status != SYNTH_OK) {
        if (audio != NULL) {
            synth_api.audio_buffer_free(audio);
        }
        synth_set_status_error("synth_synthesize_to_buffer", status);
        return NULL;
    }
    if (audio == NULL || audio->struct_size < sizeof(*audio) || audio->channel_count == 0 ||
        (audio->frame_count != 0 && audio->samples == NULL)) {
        if (audio != NULL) {
            synth_api.audio_buffer_free(audio);
        }
        PyErr_SetString(PyExc_RuntimeError, "synth_synthesize_to_buffer returned invalid owned audio");
        return NULL;
    }

    resolved_language = synth_decode_result_string(
        synthesis_result.resolved_language_tag, synthesis_result.resolved_language_tag_size, "resolved_language_tag");
    if (resolved_language == NULL) {
        synth_api.audio_buffer_free(audio);
        return NULL;
    }
    resolved_voice = synth_decode_result_string(synthesis_result.resolved_voice_id,
                                                synthesis_result.resolved_voice_id_size, "resolved_voice_id");
    if (resolved_voice == NULL) {
        Py_DECREF(resolved_language);
        synth_api.audio_buffer_free(audio);
        return NULL;
    }
    audio_owner_object = PyObject_CallNoArgs(synth_audio_owner_type);
    if (audio_owner_object == NULL) {
        Py_DECREF(resolved_voice);
        Py_DECREF(resolved_language);
        synth_api.audio_buffer_free(audio);
        return NULL;
    }
    audio_owner        = (synth_audio_owner_t *) audio_owner_object;
    audio_owner->audio = audio;
    result = Py_BuildValue("{s:O,s:K,s:I,s:I,s:K,s:K,s:I,s:O,s:O}", "audio_owner", audio_owner_object, "frame_count",
                           (unsigned long long) audio->frame_count, "sample_rate", (unsigned int) audio->sample_rate,
                           "channel_count", (unsigned int) audio->channel_count, "frames_emitted",
                           (unsigned long long) synthesis_result.frames_emitted, "actual_seed",
                           (unsigned long long) synthesis_result.actual_seed, "result_flags",
                           (unsigned int) synthesis_result.flags, "resolved_language_tag", resolved_language,
                           "resolved_voice_id", resolved_voice);
    Py_DECREF(audio_owner_object);
    Py_DECREF(resolved_voice);
    Py_DECREF(resolved_language);
    return result;

fail:
    if (input_view.obj != NULL) {
        PyBuffer_Release(&input_view);
    }
    Py_XDECREF(input_bytes);
    Py_XDECREF(language_bytes);
    Py_XDECREF(voice_bytes);
    return NULL;
}

static PyMethodDef synth_native_methods[] = {
    { "limited_api_hex",                       synth_python_limited_api_hex,                       METH_NOARGS,  NULL },
    { "load",                                  synth_python_load,                                  METH_VARARGS, NULL },
    { "backend_available",                     synth_python_backend_available,                     METH_VARARGS, NULL },
    { "backend_devices",                       synth_python_backend_devices,                       METH_NOARGS,  NULL },
    { "model_load",                            synth_python_model_load,                            METH_VARARGS, NULL },
    { "model_metadata",                        synth_python_model_metadata,                        METH_VARARGS, NULL },
    { "model_close",                           synth_python_model_close,                           METH_VARARGS, NULL },
    { "context_create",                        synth_python_context_create,                        METH_VARARGS, NULL },
    { "context_close",                         synth_python_context_close,                         METH_VARARGS, NULL },
    { "voice_profile_create_from_reference",   synth_python_voice_profile_create_from_reference,   METH_VARARGS, NULL },
    { "voice_profile_create_from_description", synth_python_voice_profile_create_from_description, METH_VARARGS, NULL },
    { "voice_profile_create_random",           synth_python_voice_profile_create_random,           METH_VARARGS, NULL },
    { "voice_profile_load",                    synth_python_voice_profile_load,                    METH_VARARGS, NULL },
    { "voice_profile_serialize",               synth_python_voice_profile_serialize,               METH_VARARGS, NULL },
    { "voice_profile_close",                   synth_python_voice_profile_close,                   METH_VARARGS, NULL },
    { "synthesize",                            synth_python_synthesize,                            METH_VARARGS, NULL },
    { NULL,                                    NULL,                                               0,            NULL },
};

static struct PyModuleDef synth_native_module = {
    PyModuleDef_HEAD_INIT, "_native", NULL, -1, synth_native_methods, NULL, NULL, NULL, NULL,
};

PyMODINIT_FUNC PyInit__native(void) {
    PyObject * module = PyModule_Create(&synth_native_module);
    PyObject * audio_type;
    PyObject * byte_buffer_type;
    if (module == NULL) {
        return NULL;
    }
    audio_type = PyType_FromSpec(&synth_audio_owner_spec);
    if (audio_type == NULL) {
        Py_DECREF(module);
        return NULL;
    }
    synth_audio_owner_type = audio_type;
    if (PyModule_AddObject(module, "_AudioBuffer", audio_type) < 0) {
        synth_audio_owner_type = NULL;
        Py_DECREF(audio_type);
        Py_DECREF(module);
        return NULL;
    }
    byte_buffer_type = PyType_FromSpec(&synth_byte_buffer_owner_spec);
    if (byte_buffer_type == NULL) {
        Py_DECREF(module);
        return NULL;
    }
    synth_byte_buffer_owner_type = byte_buffer_type;
    if (PyModule_AddObject(module, "_ByteBuffer", byte_buffer_type) < 0) {
        synth_byte_buffer_owner_type = NULL;
        Py_DECREF(byte_buffer_type);
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
