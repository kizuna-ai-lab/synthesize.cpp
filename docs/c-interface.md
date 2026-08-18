# Public C Interface

Status: Confirmed on 2026-07-21; amended 2026-08-19 for the v1 Description Text section, where a non-empty description is now the DEFAULT requirement rather than an absolute one and a Model Variant may accept an empty description as a distinct legal input.

Implementation checkpoint (2026-07-22): the installed header and CPU runtime now
implement the version/status bootstrap, opaque Model and Context lifecycle,
model loading, general capability and language/Preset Voice queries, token-ID
synthesis, cooperative cancellation, Audio Sink delivery, synthesis results, and
the owned-buffer adapter. Device discovery, runtime backend-availability probes,
strict CPU registry-index selection, and Loaded Model actual-device queries are
also implemented. Model execution beyond CPU, Text Frontend input, and Voice
Profile operations described below remain confirmed v1 contracts for later
implementation rather than alternate inference paths.

## External Seam

The public library seam is a versioned C ABI expressed through a small C Interface. It exposes opaque handles and project-owned value types; C++ implementation types, GGML contexts, tensors, backend objects, and model-family internals never cross the seam. Public option and request structures are size- or version-tagged so compatible fields can be added without silently changing layout expectations.

The primary consumer of the Interface is an embedding program that needs a common inference backend across supported Model Families and Execution Backends. The command-line program is only a reference adapter over this same C Interface: it receives no privileged access to internal modules, and no synthesis capability may exist only in the CLI.

Rust and Python bindings are also adapters over this exact C Interface. They may provide language-native ownership wrappers, slices, buffer views, iterators, exceptions, and context managers, but they cannot call C++ or model-family internals or introduce another inference path.

### v1 Export, Linkage, and Calling Convention

The installed public header owns the build-mode macros and the one calling convention used by both exported functions and callbacks:

```c
#if defined(_WIN32)
#  define SYNTH_CALL __cdecl
#  if defined(SYNTH_SHARED)
#    if defined(SYNTH_BUILDING_LIBRARY)
#      define SYNTH_API __declspec(dllexport)
#    else
#      define SYNTH_API __declspec(dllimport)
#    endif
#  else
#    define SYNTH_API
#  endif
#elif defined(__GNUC__) && defined(SYNTH_SHARED) && \
      defined(SYNTH_BUILDING_LIBRARY)
#  define SYNTH_API __attribute__((visibility("default")))
#  define SYNTH_CALL
#else
#  define SYNTH_API
#  define SYNTH_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* All public declarations live here. */

#ifdef __cplusplus
}
#endif
```

`SYNTH_SHARED` is propagated by the shared-library build target to both the library and its consumers. `SYNTH_BUILDING_LIBRARY` is private to compilation of synthesize.cpp itself. Static-library targets define neither macro, leaving `SYNTH_API` empty. Consumers do not redefine `SYNTH_API` or `SYNTH_CALL`; manual builds reproduce the target-provided definitions exactly.

Every exported function is declared as `SYNTH_API <return-type> SYNTH_CALL synth_...(...)`, and every callback pointer places `SYNTH_CALL` on its function type. Later Interface fragments may omit `SYNTH_API` and `SYNTH_CALL` for readability, but the installed header and generated binding declarations may not. Windows always uses `__cdecl`; Rust uses `extern "C"` function and callback types, while Python cdecl adapters use `CDLL`/`CFUNCTYPE` semantics rather than stdcall forms.

Linux and macOS shared builds compile internal code with hidden visibility and export only `SYNTH_API` declarations. Windows shared builds combine `__declspec(dllexport)` with a generated and checked `.def` file so even 32-bit builds expose undecorated `synth_*` names. CI snapshots the actual dynamic export table on every supported platform and rejects missing public symbols or accidental GGML, C++, backend, and Model Family exports. The bootstrap symbol is always named exactly `synth_abi_version`.

### v1 Status Codes

```c
typedef uint32_t synth_bool_t;
#define SYNTH_FALSE ((synth_bool_t) 0u)
#define SYNTH_TRUE  ((synth_bool_t) 1u)

typedef int32_t synth_status_t;
#define SYNTH_OK                       ((synth_status_t)  0)
#define SYNTH_ERR_INVALID_ARG          ((synth_status_t)  1)
#define SYNTH_ERR_BAD_STRUCT_SIZE      ((synth_status_t)  2)
#define SYNTH_ERR_FILE_NOT_FOUND       ((synth_status_t)  3)
#define SYNTH_ERR_IO                   ((synth_status_t)  4)
#define SYNTH_ERR_GGUF                 ((synth_status_t)  5)
#define SYNTH_ERR_UNSUPPORTED_ARCH     ((synth_status_t)  6)
#define SYNTH_ERR_UNSUPPORTED_VARIANT  ((synth_status_t)  7)
#define SYNTH_ERR_UNSUPPORTED_INPUT    ((synth_status_t)  8)
#define SYNTH_ERR_UNSUPPORTED_LANGUAGE ((synth_status_t)  9)
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

const char *synth_status_string(synth_status_t status);
```

Published numeric values are stable and the status constant set is append-only. `synth_status_string()` accepts the fixed-width status type so callers may safely pass unknown future or invalid values; it returns `"unknown status"` for an unrecognized value. Returned descriptions are static, read-only, and valid for the process lifetime.

`synth_bool_t` is the only ABI-visible boolean representation. Library functions return exactly `SYNTH_FALSE` or `SYNTH_TRUE`; callback results treat zero as false and any nonzero value as true. C `bool`, C++ `bool`, and language-runtime boolean layouts never cross the public seam.

Status codes identify an actionable class rather than encoding model-, file-, field-, or device-specific detail. Structured diagnostics refine failures separately. The Interface has no process-global last-error state, and log text is never a machine-readable substitute for either status or diagnostics.

### v1 ABI and Library Version Bootstrap

```c
#define SYNTH_ABI_VERSION ((uint32_t) 1u)

SYNTH_API uint32_t SYNTH_CALL synth_abi_version(void);

typedef struct synth_version {
    uint64_t struct_size;
    uint32_t abi_version;
    uint32_t version_major;
    uint32_t version_minor;
    uint32_t version_patch;
} synth_version_t;

SYNTH_API void SYNTH_CALL synth_version_init(
    synth_version_t *version,
    uint64_t struct_size);

SYNTH_API synth_status_t SYNTH_CALL synth_get_version(
    synth_version_t *out_version);
```

`synth_abi_version()` is the permanent bootstrap symbol for a dynamically loaded synthesize.cpp library. Its symbol name, parameter-free form, fixed-width return type, and calling convention never change, and it requires no library initialization, structure layout, allocator, model, backend, or thread-local state. A loader resolves and calls only this symbol first, compares the result exactly with the header's `SYNTH_ABI_VERSION`, and invokes no other library symbol when they differ.

After an ABI match, the caller initializes `synth_version_t` and calls `synth_get_version()`. The output pointer is required, the function writes only fields present in its declared `struct_size`, and `abi_version` equals `synth_abi_version()`. Major, minor, and patch are the library release version generated from the release configuration; they do not encode a Git commit, build timestamp, compiler, or other unreproducible build identity.

The ABI counter changes only for a binary-incompatible revision, including a changed published field layout or type, numeric constant, ownership rule, symbol contract, or calling convention. Compatible new symbols, appended size-tagged structure fields, and append-only flags or status values retain the current ABI counter. Library release versions advance independently and may change without changing the ABI.

### v1 Dynamic Loading Compatibility

An exact ABI version match is necessary but not sufficient for a newer caller to
use an older shared library. The ABI counter describes binary representation and
incompatible contract changes; compatible additions may still introduce a symbol
or appended field that an older release does not implement. Compatibility is
guaranteed from an older caller to a newer library with the same ABI. A newer
caller may use an older library only when that library satisfies the caller's
minimum native release and complete required-symbol manifest.

A dynamic loader follows one fail-closed sequence. It resolves and calls only
`synth_abi_version()` first. After an exact match, it resolves
`synth_version_init()` and `synth_get_version()`, obtains the semantic library
version, and requires it to be greater than or equal to the caller's declared
minimum native version. It then resolves every entry in the required-symbol
manifest generated from the caller's release header. No Model, backend, profile,
or synthesis symbol is called and no higher-level Adapter is exposed until every
check succeeds.

The minimum native version prevents an older implementation from silently
ignoring the meaning of a newer appended structure field; the symbol manifest
also detects incomplete, stripped, or incorrectly packaged libraries. Failure
identifies the expected ABI, minimum and actual library versions when available,
and the first missing symbol. v1 does not expose a `synth_get_api()` function
table or a second version-negotiated Interface: dynamic callers continue to
resolve ordinary `synth_*` C symbols.

### v1 ABI Layout Conformance

The installed `synthesize.h` header is the sole authority for public C layout. The project builds a test-only C ABI probe for each tested target; it reports `sizeof`, alignment, and every field's `offsetof` for every public non-opaque structure, together with the sizes of ABI-visible scalar and callback-pointer types. The probe is neither installed nor exported and adds no public runtime query Interface.

Rust and Python layout tests compare their low-level declarations with the target-native C probe. They do not compare against one architecture-independent golden size table, because pointer size and alignment may vary by target. Every supported target runs its own comparison.

Compatibility fixtures retain each previously published v1 structure prefix. Guard bytes surround caller storage while initializers and consuming operations verify that the library neither writes beyond the supplied `struct_size` nor reads absent appended fields. Diagnostic, cancellation, and Audio Sink callbacks also make real C-to-binding-to-C round trips so a calling-convention mismatch fails the ABI suite.

### v1 Structured Diagnostics

Status codes remain the sole operation result. An optional diagnostic sink adds
machine-readable and human-readable context without introducing global
`last_error` state or requiring applications to parse logs:

```c
typedef uint32_t synth_diagnostic_level_t;
#define SYNTH_DIAGNOSTIC_WARNING ((synth_diagnostic_level_t) 0u)
#define SYNTH_DIAGNOSTIC_ERROR   ((synth_diagnostic_level_t) 1u)

typedef struct synth_diagnostic {
    uint64_t struct_size;
    synth_diagnostic_level_t level;
    synth_status_t status;
    const char *code;
    uint64_t code_size;
    const char *message;
    uint64_t message_size;
} synth_diagnostic_t;

typedef void (SYNTH_CALL *synth_diagnostic_callback_t)(
    void *user_data,
    const synth_diagnostic_t *diagnostic);

typedef struct synth_diagnostic_sink {
    uint64_t struct_size;
    synth_diagnostic_callback_t emit;
    void *user_data;
} synth_diagnostic_sink_t;

void synth_diagnostic_sink_init(synth_diagnostic_sink_t *sink,
                                uint64_t struct_size);
```

`code` is a stable, machine-readable ASCII identifier such as
`gguf.missing_tensor`. `message` is human-readable UTF-8 and must not be parsed
for program behavior. The diagnostic structure and both strings are borrowed
and valid only for the duration of the callback.

The callback runs synchronously on the thread executing the enclosing library
call. An operation may emit zero, one, or multiple diagnostics, but its returned
`synth_status_t` alone determines the final outcome. A null sink disables
diagnostics and does not alter that outcome. Diagnostic callbacks are for
warnings and failure context, not ordinary logging, performance telemetry,
audio progress, or control flow. They must be lightweight and non-blocking, and
must not re-enter operations involving the related Model or Synthesis Context.

Model loading and synthesis each accept an optional diagnostic sink. The sink
and its user data are borrowed only for the enclosing synchronous call; their
parameter initializers set the sink pointer to null.

### v1 Parameter Structure Convention

Every public parameter structure begins with `uint64_t struct_size`. A caller that customizes parameters initializes them with the corresponding function and passes its compile-time size:

```c
synth_model_load_params_t params;
synth_model_load_params_init(&params, sizeof(params));
```

Passing a null parameter pointer selects the documented defaults. A non-null structure with `struct_size == 0`, including an ordinary `{0}` initialization, is rejected with `SYNTH_ERR_BAD_STRUCT_SIZE`. The initializer writes defaults only within the caller-supplied size, and a consuming function reads only the known fields present within that size.

Compatible revisions may append fields only at the end of a structure; existing field order and meaning cannot change. Parameter defaults are not returned by value, avoiding a public convention analogous to `llama_model_default_params()`.

### Foreign-language Binding Contract

The installed C header and exported C symbols are the canonical foreign-function seam. Public structures use fixed-width scalar fields and explicit pointer-plus-count views rather than C++ containers, C enums, C or C++ `bool`, null-terminated-string requirements, or caller-visible implementation objects. Rust and Python adapters must preserve the documented borrowed and owned lifetimes.

No foreign exception or panic may unwind through a C callback. A Rust adapter catches unwinding before returning a sink result; a Python adapter reacquires the GIL for Python callbacks, captures any exception, returns the corresponding C cancellation or sink-error value, and raises only after the synchronous C call returns. The core library never knows about the GIL or a language runtime.

An owned `synth_audio_buffer_t` may back a Rust slice or Python buffer/NumPy view without another PCM copy. The binding must keep the owning buffer alive until the final view is released and then call `synth_audio_buffer_free()` exactly once. Borrowed request and Voice Reference buffers remain pinned only for the enclosing synchronous call or profile-preparation operation.

## Handle Ownership

The public Interface uses three opaque runtime handle types:

```c
typedef struct synth_model synth_model_t;
typedef struct synth_context synth_context_t;
typedef struct synth_voice_profile synth_voice_profile_t;
```

The **Loaded Model** handle owns immutable model metadata, weights, and backend allocations. After successful loading it may be shared by multiple threads and must outlive every Synthesis Context created from it.

The **Synthesis Context** handle owns mutable scratch memory, random state, and any incremental model state. A context permits one active synthesis at a time and may be reused sequentially after completion, cancellation, or a documented reset. Concurrent synthesis uses one context per concurrent operation while sharing the Loaded Model.

The **Voice Profile** handle owns or retains an immutable, prepared form of runtime Voice conditioning. It is bound to the Loaded Model that prepared it, may be shared by multiple Synthesis Contexts and concurrent synthesis calls, and must outlive every call that selects it. The Loaded Model must outlive all of its Voice Profiles. A profile prevents repeated reference-audio encoding; it does not promise that every Model Family and Execution Backend can avoid all physical memory copies.

### v1 Object Lifecycle

v1 exposes exactly three runtime object types: Loaded Model, Synthesis Context, and Voice Profile. Their lifecycle operations include `synth_model_load()`, `synth_model_free()`, `synth_context_create()`, `synth_context_free()`, Voice Profile preparation from supported sources, and `synth_voice_profile_free()`.

The public Interface has no global initialization object, model-manager handle, Voice registry, or backend registry handle. Backend and model-family discovery remain implementation concerns of model loading. Creating a context or preparing a Voice Profile accepts an existing Loaded Model; destroying the model while any of its contexts or profiles remain alive is invalid.

### v1 Preset Voice Catalog

The Preset Voice Catalog is immutable metadata owned by the Loaded Model rather than a fourth runtime handle:

```c
typedef uint32_t synth_preset_voice_flags_t;

#define SYNTH_PRESET_VOICE_DEFAULT \
    ((synth_preset_voice_flags_t) (1u << 0))

typedef struct synth_preset_voice {
    uint64_t struct_size;
    const char *id;
    uint64_t id_size;
    const char *display_name;
    uint64_t display_name_size;
    synth_preset_voice_flags_t flags;
} synth_preset_voice_t;

void synth_preset_voice_init(
    synth_preset_voice_t *voice,
    uint64_t struct_size);

synth_status_t synth_model_get_preset_voice_count(
    const synth_model_t *model,
    uint64_t *out_count);

synth_status_t synth_model_get_preset_voice(
    const synth_model_t *model,
    uint64_t index,
    synth_preset_voice_t *out_voice);
```

Every Catalog entry has a non-empty, length-delimited UTF-8 `id` that is unique within the Model Variant. The identifier is the stable machine value used by synthesis requests and is preserved across validated Model Packages representing the same logical Model Variant. `display_name` is optional human-readable UTF-8, need not be unique, and is never accepted as a selection identifier. Neither string requires null termination.

Indices provide enumeration only. Their order is deterministic and immutable for the lifetime of the Loaded Model, but callers persist `id`, not an index, across package revisions. At most one entry has `SYNTH_PRESET_VOICE_DEFAULT`; a package default need not be a Catalog entry. A fixed single-speaker Model Variant may therefore have a usable unnamed default Voice while reporting a count of zero.

Both queries are read-only and may run concurrently on a shared Loaded Model. The count operation requires `out_count`, sets it to zero before validation, and returns zero successfully for an empty Catalog. The indexed operation requires an initialized output structure, writes only fields present in its declared `struct_size`, and returns `SYNTH_ERR_INVALID_ARG` for an out-of-range index. Its initializer clears every visible field and records the caller-provided size.

Returned strings are borrowed from immutable Loaded Model storage and remain valid until `synth_model_free()`. Flags are append-only and callers ignore unknown future bits. v1 Catalog entries do not contain language, gender, age, or inferred accent fields; those values are not universal Voice identity metadata, and language capability is queried separately. Runtime Voice Profiles are never inserted into the Catalog.

### v1 Language Capability Catalog

Language capabilities are immutable Model Variant metadata owned by the Loaded Model and enumerated independently of Preset Voices:

```c
typedef uint32_t synth_language_flags_t;

#define SYNTH_LANGUAGE_DEFAULT \
    ((synth_language_flags_t) (1u << 0))
#define SYNTH_LANGUAGE_REGIONAL_FALLBACK \
    ((synth_language_flags_t) (1u << 1))

typedef struct synth_language_capability {
    uint64_t struct_size;
    const char *tag;
    uint64_t tag_size;
    synth_language_flags_t flags;
} synth_language_capability_t;

void synth_language_capability_init(
    synth_language_capability_t *language,
    uint64_t struct_size);

synth_status_t synth_model_get_language_count(
    const synth_model_t *model,
    uint64_t *out_count);

synth_status_t synth_model_get_language(
    const synth_model_t *model,
    uint64_t index,
    synth_language_capability_t *out_language);
```

Every Supported Model Variant exposes at least one entry. `tag` is a non-empty, unique, canonical BCP 47 language tag stored as length-delimited ASCII, such as `en`, `zh-CN`, or `ja`; it need not be null-terminated. Requests validate BCP 47 syntax, reject underscore-separated forms, and match tags without regard to ASCII case. The query always returns the Model Package's canonical spelling. A malformed tag returns `SYNTH_ERR_INVALID_ARG`, while a well-formed but unsupported tag returns `SYNTH_ERR_UNSUPPORTED_LANGUAGE`.

At most one entry has `SYNTH_LANGUAGE_DEFAULT`. Omitting `language_tag` selects that entry; if no default is declared, omission returns `SYNTH_ERR_UNSUPPORTED_LANGUAGE`. An exact advertised tag always wins before fallback handling.

`SYNTH_LANGUAGE_REGIONAL_FALLBACK` is valid only on a base entry consisting of one primary-language subtag, such as `en`. It permits a request consisting of that same primary language plus one region subtag, such as `en-US` or `en-GB`, to resolve to the base entry when no exact advertised entry exists. It does not accept script, variant, extension, or private-use additions and never changes the primary language; in particular it cannot map `zh-HK` or `zh-TW` to `zh-CN`. The canonical resolved tag is returned through `synth_result_t.resolved_language_tag`.

Indices are for enumeration only. The count operation requires `out_count` and sets it to zero before validation. The indexed operation requires an initialized output structure, writes only fields present in its declared `struct_size`, and returns `SYNTH_ERR_INVALID_ARG` for an out-of-range index. Both operations are read-only and may run concurrently on a shared Loaded Model. Returned tags are borrowed until `synth_model_free()`, flags are append-only, and callers ignore unknown future bits.

### v1 General Model Capability Query

One Loaded Model-level summary exposes the synthesis invariants a caller needs to construct requests and consume native output:

```c
typedef uint32_t synth_input_flags_t;

#define SYNTH_INPUT_SUPPORT_TEXT_UTF8 \
    ((synth_input_flags_t) (1u << 0))
#define SYNTH_INPUT_SUPPORT_PHONEMES_UTF8 \
    ((synth_input_flags_t) (1u << 1))
#define SYNTH_INPUT_SUPPORT_TOKEN_IDS \
    ((synth_input_flags_t) (1u << 2))

typedef uint32_t synth_model_capability_flags_t;

#define SYNTH_MODEL_CAPABILITY_SPEAKING_RATE \
    ((synth_model_capability_flags_t) (1u << 0))
#define SYNTH_MODEL_CAPABILITY_STOCHASTIC \
    ((synth_model_capability_flags_t) (1u << 1))
#define SYNTH_MODEL_CAPABILITY_NATIVE_STREAMING \
    ((synth_model_capability_flags_t) (1u << 2))

typedef struct synth_model_capabilities {
    uint64_t struct_size;
    synth_input_flags_t input_flags;
    synth_model_capability_flags_t capability_flags;
    uint32_t output_sample_rate;
    uint32_t output_channel_count;
    float min_speaking_rate;
    float max_speaking_rate;
    uint64_t max_input_tokens;
    uint64_t max_output_frames;
} synth_model_capabilities_t;

void synth_model_capabilities_init(
    synth_model_capabilities_t *capabilities,
    uint64_t struct_size);

synth_status_t synth_model_get_capabilities(
    const synth_model_t *model,
    synth_model_capabilities_t *out_capabilities);
```

`input_flags` reports input forms usable by this Loaded Model in the current runtime, not merely forms understood by its Model Family. If a required Text Frontend Provider was unavailable when the model was loaded, text input is not set even when the Model Package declares it; registering a provider later does not mutate an existing Loaded Model, so the caller reloads the model to obtain a different immutable capability snapshot. A fundamentally unsupported input kind returns `SYNTH_ERR_UNSUPPORTED_INPUT`, while a package-declared text path unavailable because of its provider returns `SYNTH_ERR_TEXT_FRONTEND` with diagnostics.

`output_sample_rate` and `output_channel_count` are positive and describe the fixed native interleaved F32 format delivered by every operation and Audio Sink chunk. `max_input_tokens` is the positive hard limit on the final token sequence consumed by the synthesis graph, after any Text Frontend processing and model-owned special-token insertion. Token-ID input can be checked directly; text and phoneme input are checked after conversion. Exceeding it returns `SYNTH_ERR_INPUT_TOO_LONG`.

`max_output_frames` is the Model Package's positive hard safety limit in native PCM frames. A request-level nonzero output limit combines with it by taking the smaller value; a request value of zero leaves the model limit in force. Reaching either limit follows the common `SYNTH_ERR_OUTPUT_LIMIT` and partial-output rules.

Every Model accepts speaking rate `1.0f`. Without `SYNTH_MODEL_CAPABILITY_SPEAKING_RATE`, both range fields are exactly `1.0f`; with the flag, they are finite, positive, ordered, and contain `1.0f`. `SYNTH_MODEL_CAPABILITY_STOCHASTIC` means at least one synthesis path consumes request randomness, while `SYNTH_RESULT_SEED_USED` reports whether a particular operation did. `SYNTH_MODEL_CAPABILITY_NATIVE_STREAMING` is set only for the currently loaded Model Package and Execution Backend combination after full streaming validation; the operation result still reports whether that path was actually used.

The caller initializes the output structure; the query writes only fields present in its declared `struct_size`. The summary is immutable, read-only, and safe to query concurrently. Both flag sets are append-only and callers ignore unknown future bits. Language, Preset Voice, Voice Profile, and device metadata remain in their dedicated queries.

### v1 Voice Profile Source Semantics

The public Interface recognizes four Voice Profile preparation sources:

- **Reference Audio**: one or more borrowed PCM views plus any model-required transcript and language metadata;
- **Description Text**: a length-delimited UTF-8 description for a model that can design Voice identity from natural language;
- **Random Seed**: deterministic sampling of a reusable Voice identity when the Model Variant supports it; and
- **Serialized Profile**: a length-delimited, versioned profile representation whose model compatibility is validated during import.

Each source has its own public preparation operation:

```c
synth_voice_profile_create_from_reference(...);
synth_voice_profile_create_from_description(...);
synth_voice_profile_create_random(...);
synth_voice_profile_load_from_memory(...);
synth_voice_profile_free(...);
```

The source-specific parameter structures remain size-tagged and binding-friendly, while all four preparation operations delegate validation, backend preparation, ownership, and diagnostics to the same internal Voice Profile Module. v1 has no generic `synth_voice_profile_create()`, public union, untyped source pointer, or Model Family-specific profile entry point. Adding a future source adds a new function and source structure without changing existing layouts.

Preset Voice selection continues to use `voice_id` and is not a profile-preparation source. Multi-reference enrollment and fusion are represented inside Reference Audio rather than as additional top-level kinds. Raw speaker tensors, x-vectors, and acoustic tokens never cross the public seam; converters or the family implementation encapsulate them in a Serialized Profile or prepared handle.

### v1 Voice Profile Capability Query

```c
typedef uint32_t synth_voice_profile_source_flags_t;

#define SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO \
    ((synth_voice_profile_source_flags_t) (1u << 0))
#define SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT \
    ((synth_voice_profile_source_flags_t) (1u << 1))
#define SYNTH_PROFILE_SOURCE_RANDOM_SEED \
    ((synth_voice_profile_source_flags_t) (1u << 2))
#define SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE \
    ((synth_voice_profile_source_flags_t) (1u << 3))

typedef uint32_t synth_requirement_t;

#define SYNTH_REQUIREMENT_UNSUPPORTED \
    ((synth_requirement_t) 0u)
#define SYNTH_REQUIREMENT_OPTIONAL \
    ((synth_requirement_t) 1u)
#define SYNTH_REQUIREMENT_REQUIRED \
    ((synth_requirement_t) 2u)

#define SYNTH_REFERENCE_SAMPLE_RATE_MIN ((uint32_t) 8000u)
#define SYNTH_REFERENCE_SAMPLE_RATE_MAX ((uint32_t) 192000u)
#define SYNTH_REFERENCE_CHANNELS_MAX    ((uint32_t) 2u)

typedef struct synth_voice_profile_capabilities {
    uint64_t struct_size;
    synth_voice_profile_source_flags_t source_flags;
    synth_requirement_t reference_transcript;
    synth_requirement_t reference_language;
    synth_requirement_t description_language;
    uint64_t max_reference_count;
    const char *profile_schema;
    uint64_t profile_schema_size;
    uint32_t profile_schema_version;
    uint8_t profile_compatibility_id[32];
    uint32_t reference_target_sample_rate;
    uint32_t reference_target_channel_count;
    uint64_t min_reference_frames_per_clip;
    uint64_t max_reference_frames_per_clip;
    uint64_t max_reference_total_frames;
} synth_voice_profile_capabilities_t;

void synth_voice_profile_capabilities_init(
    synth_voice_profile_capabilities_t *capabilities,
    uint64_t struct_size);

synth_status_t synth_model_get_voice_profile_capabilities(
    const synth_model_t *model,
    synth_voice_profile_capabilities_t *out_capabilities);
```

This is one Loaded Model-level capability summary, not a Voice list or a family-specific query. `source_flags` is append-only and callers ignore unknown future bits. A Model without runtime Voice Profile support reports zero flags; Preset Voices remain discoverable only through the separate Preset Voice Catalog.

For an unsupported source, all fields that describe that source are `SYNTH_REQUIREMENT_UNSUPPORTED` or zero. When Reference Audio is supported, `max_reference_count` is at least one and its transcript and language fields independently report unsupported, optional, or required. When Description Text is supported, `description_language` uses the same policy. Supplying a field declared unsupported or omitting one declared required fails explicitly rather than being silently ignored or inferred.

Any Loaded Model that creates a Profile from Reference Audio, Description Text, or Random Seed also sets `SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE`, because every successfully prepared v1 Profile can be serialized. A Model may set only the Serialized Profile bit when it can consume prebuilt profiles but cannot prepare them from another public source.

When Serialized Profile is unsupported, `profile_schema` is null with zero size, its version is zero, and the 32 ID bytes are zero. When supported, the length-delimited schema string, nonzero schema version, and exact Profile Compatibility ID match the primary GGUF metadata. The schema string is borrowed from immutable Loaded Model storage and remains valid until `synth_model_free()`; the fixed ID is copied into the caller's output structure.

The caller initializes `out_capabilities`; the query writes only fields present in its declared `struct_size`. A null or undersized output pointer is invalid. The initializer clears every visible field and writes the caller-provided structure size.

#### Reference Audio Normalization and Limits

All v1 Models that support Reference Audio accept contiguous interleaved F32 PCM at an integer sample rate from 8000 through 192000 Hz and with one or two channels. Every sample must be finite. Inputs outside that common format contract return `SYNTH_ERR_UNSUPPORTED_INPUT`; null or empty sample storage, checked-arithmetic overflow, and NaN or infinity return `SYNTH_ERR_INVALID_ARG`.

The Loaded Model declares the Voice encoder's positive target sample rate and one- or two-channel target format. The Voice Profile Module owns one private Audio Normalizer Module and applies a fixed order: validate, convert channels, then resample. Matching input can be read directly, stereo-to-mono uses an equal-weight channel mean, mono-to-stereo duplicates each sample, and any supported conversion uses internal storage. CLI, Rust, Python, and Model Family implementations do not define alternate normalization policies or expose resampler choices.

The minimum and maximum per-clip fields and maximum total field are Reference Frame Equivalent limits. Before resampling or large allocation, the implementation computes each equivalent with checked integer arithmetic equal to `ceil(input_frames * target_rate / input_rate)` and checks the per-clip and checked-sum total limits. This is a target-rate duration measure, not a requirement that the resampler emit exactly that many frames.

After the limits pass, the Audio Normalizer drains the complete finite-input conversion and retains the actual frames reported by the resampler. It does not pad or crop the converted signal merely to force its length to the Reference Frame Equivalent, and it never trims, truncates, or discards caller-supplied references to satisfy a limit. The fixed validated implementation aims for numerical equivalence across supported hosts; this Interface does not promise bit-identical normalization across architectures or compiler floating-point implementations.

When Reference Audio is unsupported, all five appended normalization and limit fields are zero. When supported, the target fields, per-clip minimum and maximum, and total maximum are nonzero; the minimum does not exceed the per-clip maximum, and the per-clip maximum does not exceed the total maximum. Exceeding either maximum returns `SYNTH_ERR_INPUT_TOO_LONG`; falling below the minimum is `SYNTH_ERR_INVALID_ARG`.

### v1 Reference Audio Profile Preparation

```c
typedef struct synth_voice_reference {
    uint64_t struct_size;
    const float *samples;
    uint64_t frame_count;
    uint32_t sample_rate;
    uint32_t channel_count;
    const char *transcript;
    uint64_t transcript_size;
    const char *language_tag;
    uint64_t language_tag_size;
} synth_voice_reference_t;

typedef struct synth_voice_reference_params {
    uint64_t struct_size;
    const synth_voice_reference_t *references;
    uint64_t reference_count;
    uint64_t reference_stride;
    const synth_diagnostic_sink_t *diagnostics;
} synth_voice_reference_params_t;

void synth_voice_reference_init(synth_voice_reference_t *reference,
                                uint64_t struct_size);
void synth_voice_reference_params_init(
    synth_voice_reference_params_t *params,
    uint64_t struct_size);

synth_status_t synth_voice_profile_create_from_reference(
    const synth_model_t *model,
    const synth_voice_reference_params_t *params,
    synth_voice_profile_t **out_profile);
```

The core Interface accepts decoded, contiguous, interleaved F32 PCM rather than file paths, URLs, or encoded audio containers. `frame_count` counts sample frames rather than scalar samples, so each descriptor exposes `frame_count * channel_count` floats subject to checked size arithmetic. Every descriptor follows the common sample-rate, channel-count, finite-sample, normalization, and Reference Frame Equivalent rules reported by the Loaded Model capabilities. A transcript is length-delimited UTF-8; `language_tag` is length-delimited ASCII BCP 47. Neither requires null termination, and either may be absent with a null pointer and zero size when Loaded Model capabilities do not require it.

`references` names a contiguous array containing at least one descriptor. `reference_stride` is the byte distance between consecutive descriptors and is supplied explicitly as the caller's compile-time `sizeof(synth_voice_reference_t)`. It must cover every descriptor's declared `struct_size`. This prevents a newer library that knows appended descriptor fields from assuming its own larger stride over an array compiled by an older C, Rust, or Python binding. Future fields are appended to the descriptor; the audio data is kept flat rather than embedding another extensible structure by value.

All descriptor arrays, PCM samples, strings, and diagnostics are borrowed only until this synchronous operation returns. The implementation may read contiguous Rust slices or Python `float32` buffers without an additional ABI-staging copy. It may still allocate or transform data for resampling, channel conversion, feature extraction, backend transfer, or family-specific preparation, so the Interface does not promise physical zero-copy processing.

Multiple descriptors remain one Reference Audio source. Their validated fusion is selected by the Model Family implementation; no implementation may silently discard extra references. On success the returned immutable Voice Profile owns or retains all prepared conditioning and no longer depends on the borrowed descriptors or PCM. `out_profile` is required and is set to null before validation; `synth_voice_profile_free()` accepts null.

The reference initializer writes the caller-provided structure size and clears every field present in that size. The parameter initializer likewise clears its visible fields; no default reference array or descriptor stride exists, so the caller fills `references`, `reference_count`, and `reference_stride` explicitly.

### v1 Description Text Profile Preparation

```c
typedef struct synth_voice_description_params {
    uint64_t struct_size;
    const char *description;
    uint64_t description_size;
    const char *language_tag;
    uint64_t language_tag_size;
    uint64_t seed;
    const synth_diagnostic_sink_t *diagnostics;
} synth_voice_description_params_t;

void synth_voice_description_params_init(
    synth_voice_description_params_t *params,
    uint64_t struct_size);

synth_status_t synth_voice_profile_create_from_description(
    const synth_model_t *model,
    const synth_voice_description_params_t *params,
    synth_voice_profile_t **out_profile);
```

`description` is length-delimited UTF-8 and need not be null-terminated. **A non-empty description is the default requirement, not an absolute one: a Model Variant may accept an empty description as a distinct legal input,** with a meaning that variant defines. A variant that does not accept one returns `SYNTH_ERR_INVALID_ARG`, whichever way the caller spells the emptiness. Qwen3-TTS VoiceDesign is a variant that does accept one, where it requests an unconditioned Voice -- its only path to a Voice Profile without a description, since it catalogues no Preset Voices and supports neither Reference Audio nor Random Seed. The description's vocabulary, schema, and expressive range are entirely Model Variant-defined; the common Interface does not standardize fields such as gender, age, accent, emotion, or recording style and does not reinterpret the description as ordinary synthesis text.

**A caller discovers which of the two a variant is by attempting the call.** v1 declares this nowhere a program can read it: `synth_voice_profile_capabilities_t` has no field for it, no Model Package metadata key carries it, and `SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT` says only that the source exists, not what an empty description does within it. The only statement that a given variant accepts one is prose: for Qwen3-TTS VoiceDesign it is the paragraph above and `docs/voice-conditioning.md`'s Description Text entry, both written for a human. A machine-readable declaration -- a capability field, or a Model Package key -- is not built here and should be proposed as its own change rather than assumed to exist. An Adapter must not pre-empt the attempt: refusing an empty description of its own accord makes the Interface's own answer unreachable in both directions, for the variant that accepts one and for the variant whose refusal it hides. This paragraph amends a line that read "`description` is required, non-empty" until 2026-08-19; `bindings/python` enforced exactly that refusal until the same date.

`language_tag` is a length-delimited ASCII BCP 47 tag identifying the language of the description itself, not the target language of a later synthesis request. It follows the common syntax and case-matching rules but is validated against the Model Variant's description-language support rather than its synthesis Language Capability Catalog. A null pointer with zero size selects the Model Package's declared description-language default. The implementation never detects the description language from its text, and a variant with neither an explicit tag nor a declared usable default fails explicitly.

Description-driven preparation may be stochastic, so `seed` controls this preparation operation independently of synthesis request randomness. The initializer selects concrete seed zero. v1 profile preparation rejects `SYNTH_SEED_RANDOM`; an application or Adapter that wants a random Voice generates and records a concrete `uint64_t` before calling the Interface. This avoids constructing a reusable profile from an unrecoverable random seed. Random Seed remains a separate source whose identity input is only a concrete seed, while Description Text uses a seed as a preparation control.

The parameter structure, its strings, and diagnostics are borrowed only for the synchronous operation. On success the returned immutable Voice Profile owns or retains the encoded description conditioning and no longer depends on them. A Loaded Model that does not support Description Text returns `SYNTH_ERR_UNSUPPORTED_VOICE`. `out_profile` is required, is set to null before validation, and is released only with `synth_voice_profile_free()`.

The initializer writes the caller-provided structure size, clears all visible fields, and sets `seed` to zero when that field is present. There is still no valid null-parameter default, and the amendment above does not create one: the parameter structure itself is required and a null `params` returns `SYNTH_ERR_INVALID_ARG`, on every variant. An absent parameter structure is not a spelling of an empty description -- it also withholds the language tag, the seed, and the diagnostics sink, which the Interface has no basis for inventing. A `struct_size` too small to expose `description` at all is a different case: the visible-field rules already give the missing fields their cleared defaults, so such a call presents an empty description and resolves exactly as an explicitly empty one does on that variant.

### v1 Random Seed Profile Preparation

```c
typedef struct synth_voice_random_params {
    uint64_t struct_size;
    uint64_t seed;
    const synth_diagnostic_sink_t *diagnostics;
} synth_voice_random_params_t;

void synth_voice_random_params_init(
    synth_voice_random_params_t *params,
    uint64_t struct_size);

synth_status_t synth_voice_profile_create_random(
    const synth_model_t *model,
    const synth_voice_random_params_t *params,
    synth_voice_profile_t **out_profile);
```

Random Seed samples a reusable Voice identity from the Loaded Model's own Voice distribution using only one concrete seed. It accepts no description, language, reference audio, caller-visible latent, or architecture-specific sampling field. The same numeric seed has no cross-model semantic meaning, and a resulting profile remains bound to the Loaded Model that created it.

The initializer clears every visible field and selects seed zero. As with Description Text preparation, `SYNTH_SEED_RANDOM` is invalid for v1 profile creation; an application or Adapter requests a nondeterministically chosen identity by generating and retaining a concrete seed before the call. The seed belongs to profile preparation and is independent of any later synthesis request seed.

For the same Model Package, Model Variant, and validated execution configuration, a concrete seed selects the same logical random sequence. This does not promise compatible identities across package or model revisions, or bit-identical prepared data and PCM across CPU, CUDA, Metal, and Vulkan floating-point implementations.

A Loaded Model that does not support Random Seed returns `SYNTH_ERR_UNSUPPORTED_VOICE`; the value is never interpreted as a Preset Voice ID or synthesis seed. The parameter structure and diagnostics are borrowed only for the synchronous call. `out_profile` is required, is set to null before validation, and a successful immutable profile is released only with `synth_voice_profile_free()`.

### v1 Serialized Profile Memory Interface

```c
typedef struct synth_voice_profile_load_params {
    uint64_t struct_size;
    const uint8_t *data;
    uint64_t data_size;
    const synth_diagnostic_sink_t *diagnostics;
} synth_voice_profile_load_params_t;

typedef struct synth_voice_profile_serialize_params {
    uint64_t struct_size;
    const synth_diagnostic_sink_t *diagnostics;
} synth_voice_profile_serialize_params_t;

typedef struct synth_byte_buffer {
    uint64_t struct_size;
    const uint8_t *data;
    uint64_t data_size;
} synth_byte_buffer_t;

void synth_voice_profile_load_params_init(
    synth_voice_profile_load_params_t *params,
    uint64_t struct_size);
void synth_voice_profile_serialize_params_init(
    synth_voice_profile_serialize_params_t *params,
    uint64_t struct_size);

synth_status_t synth_voice_profile_load_from_memory(
    const synth_model_t *model,
    const synth_voice_profile_load_params_t *params,
    synth_voice_profile_t **out_profile);

synth_status_t synth_voice_profile_serialize(
    const synth_voice_profile_t *profile,
    const synth_voice_profile_serialize_params_t *params,
    synth_byte_buffer_t **out_data);

void synth_byte_buffer_free(synth_byte_buffer_t *buffer);
```

Serialized Profile is a library-defined, opaque, self-contained little-endian GGUF v3 representation. It carries a project format version and Profile Compatibility ID in addition to the GGUF container version. It cannot depend on file paths, URLs, dynamic code, implicit network access, or undeclared external resources. Family-specific embeddings, latents, and acoustic tokens may occur inside the representation but never become typed public tensors.

Loading requires non-null, non-empty bytes and borrows them only until the synchronous call returns. The loader validates size arithmetic, structural bounds, format version, integrity data, resource limits, and Model compatibility before large allocation or backend upload. A successful Voice Profile owns or retains its prepared representation and no longer depends on the input bytes. Incompatible Model data returns `SYNTH_ERR_UNSUPPORTED_VOICE`; malformed, truncated, or corrupt data returns `SYNTH_ERR_INVALID_ARG`, with stable diagnostic codes providing detail.

Serialization accepts a valid immutable Voice Profile and allocates a read-only `synth_byte_buffer_t` plus its data. `out_data` is required and is set to null before validation. The caller releases both only with `synth_byte_buffer_free()`, which accepts null. A null serialization-parameter pointer selects defaults; load parameters cannot be null because no input bytes have a default.

Rust and Python Adapters may expose the owned bytes as a slice or `memoryview` tied to the owning buffer without another byte copy. Path-based `load` and `save` conveniences belong to CLI, Rust, or Python Adapters, which read or write bytes before or after crossing the same memory Interface. The core library exposes no separate file-loading path.

### v1 Serialized Profile GGUF Contract

Every v1 serialization writes little-endian GGUF v3 with 32-byte alignment and these required metadata entries:

```text
general.architecture                              = "synthprofile"
synthesize.voice_profile.format_version           = 1
synthesize.voice_profile.model_family             = <string>
synthesize.voice_profile.schema                   = <string>
synthesize.voice_profile.schema_version           = <uint32>
synthesize.voice_profile.compatibility_id         = <uint8[32]>
synthesize.voice_profile.content_sha256            = <uint8[32]>
```

`format_version` versions the project contract independently of GGUF's structural version. `model_family`, `schema`, and `schema_version` select the private family deserializer and canonical payload layout. Unknown optional namespaced metadata may be ignored, but missing, duplicated, wrongly typed, or unsupported required entries are invalid. Tensor and array names, shapes, element types, and required/optional status are specified by each supported Model Family's versioned Profile Schema.

The Profile Compatibility ID is the SHA-256 digest of a converter-defined canonical compatibility manifest. That manifest includes the Model Family, Profile Schema and version, upstream checkpoint identity and revision, and fingerprints of configuration and source weights that affect the meaning or consumption of Voice conditioning. The same 32 bytes are stored in every compatible Model Package's primary GGUF. Equality is exact: loading never falls back to matching filenames, `general.uuid`, architecture names, tensor dimensions, or similar-looking metadata.

Quantization and Execution Backend are deliberately excluded from the compatibility manifest. F32, F16, and validated quantized Model Packages derived from the same compatibility-critical source may share an ID, and a canonical profile serialized after CPU or GPU preparation remains portable across validated backends. If a converter cannot prove compatibility, it emits a distinct ID. The Serialized Profile stores canonical host-side conditioning rather than device pointers, backend buffers, GGML graphs, or backend-specific packed layouts.

To compute `content_sha256`, the serializer first emits the complete GGUF with the required 32-byte hash value set to zero, hashes that entire byte sequence, and then replaces the zero value with the digest. Verification hashes the complete input while treating that one value as zero. The digest detects accidental corruption but is neither a signature nor proof of origin; v1 provides no profile signing or encryption.

The loader treats the bytes as untrusted. Before family deserialization, large allocation, or backend upload, it validates the top-level GGUF counts, every length and offset, checked size arithmetic, declared tensor byte totals, alignment, required metadata, the content digest, configured resource limits, Profile Schema, and exact Profile Compatibility ID. Source reference audio, transcript, and Description Text are not serialized by default; the payload contains prepared canonical conditioning only.

`synth_model_load()` accepts the path of the Model Package's primary GGUF, not a directory. The parent directory of that file is the Package Root, and every declared Sidecar Resource is resolved relative to it. Loading does not scan the directory for undeclared resources, execute package content, or perform implicit network access.

### v1 Backend Module Loading

```c
synth_status_t synth_backend_load_from_dir(const char *artifact_dir);
synth_status_t synth_backend_load_default(void);
```

`synth_backend_load_from_dir()` and `synth_backend_load_default()` register optional GGML dynamic backend modules before the first call to `synth_model_load()`. They are process-level functions, not another public object or registry handle. In statically linked builds they return success without changing state.

The directory-taking operation scans only its explicit artifact directory; the
default operation resolves the loaded shared core library and scans only its fixed
relative `synthesize/backends/` directory. Neither scans the core-library
directory itself, working directory, Package Root, model directory, `PATH`, or
arbitrary system library paths. In particular, the project loader does not honor
GGML's `GGML_BACKEND_PATH` escape hatch. Repeating the same canonical directory
returns the first scan's status without registering duplicate devices. Null or
empty explicit paths return `SYNTH_ERR_INVALID_ARG`, a path that is not an
existing directory returns `SYNTH_ERR_FILE_NOT_FOUND`, and a dynamic directory
that registers no compute device returns `SYNTH_ERR_BACKEND`. The first valid
`synth_model_load()` call freezes the process registry; either loading function
then returns `SYNTH_ERR_INVALID_ARG`. v1 exposes no backend-unload operation.

### v1 Backend Selection

Model loading uses project-owned types equivalent to the following stable fields:

```c
typedef uint32_t synth_backend_request_t;
#define SYNTH_BACKEND_AUTO      ((synth_backend_request_t) 0u)
#define SYNTH_BACKEND_CPU       ((synth_backend_request_t) 1u)
#define SYNTH_BACKEND_CPU_ACCEL ((synth_backend_request_t) 2u)
#define SYNTH_BACKEND_CUDA      ((synth_backend_request_t) 3u)
#define SYNTH_BACKEND_METAL     ((synth_backend_request_t) 4u)
#define SYNTH_BACKEND_VULKAN    ((synth_backend_request_t) 5u)

typedef struct synth_model_load_params {
    uint64_t struct_size;
    synth_backend_request_t backend;
    int32_t device_index; /* -1 selects automatically */
    const synth_diagnostic_sink_t *diagnostics;
} synth_model_load_params_t;
```

`AUTO` is the default and currently selects the Supported CPU path. Explicit GPU
backend requests are strict: they never silently substitute CPU, another GPU
backend, or a different explicitly indexed device. A development build may expose
an Experimental combination before it becomes a release-qualified Supported
combination; successful loading is therefore execution evidence, not a support
claim. `CPU` is the strict reference path, while `CPU_ACCEL` permits host-memory
accelerators and may degrade to plain CPU.

The Interface provides project-owned device enumeration, backend-availability, and Loaded Model device queries. It never exposes `ggml_backend_dev_t` or v1 controls analogous to `n_gpu_layers`, multi-GPU split modes, tensor splits, or tensor buffer overrides. A Loaded Model has one observable primary device even when its graph also schedules work on CPU.

### v1 Device Discovery

The device discovery Interface mirrors transcribe.cpp's GGML registry wrapper under project-owned names:

```c
typedef uint32_t synth_device_type_t;
#define SYNTH_DEVICE_TYPE_CPU   ((synth_device_type_t) 0u)
#define SYNTH_DEVICE_TYPE_GPU   ((synth_device_type_t) 1u)
#define SYNTH_DEVICE_TYPE_IGPU  ((synth_device_type_t) 2u)
#define SYNTH_DEVICE_TYPE_ACCEL ((synth_device_type_t) 3u)

typedef uint32_t synth_device_flags_t;
#define SYNTH_DEVICE_MEMORY_INFO_VALID \
    ((synth_device_flags_t) (1u << 0))
#define SYNTH_DEVICE_MEMORY_SHARED \
    ((synth_device_flags_t) (1u << 1))
#define SYNTH_DEVICE_MEMORY_INFO_APPROXIMATE \
    ((synth_device_flags_t) (1u << 2))

typedef struct synth_backend_device {
    uint64_t struct_size;
    const char *name;
    const char *description;
    const char *kind;
    const char *device_id;
    uint64_t memory_total;
    uint64_t memory_free;
    synth_device_type_t device_type;
    synth_device_flags_t flags;
} synth_backend_device_t;

uint32_t synth_backend_device_count(void);
void synth_backend_device_init(synth_backend_device_t *device,
                               uint64_t struct_size);
synth_status_t synth_backend_device_get(uint32_t index,
                                        synth_backend_device_t *out_device);
synth_bool_t synth_backend_available(synth_backend_request_t request);
synth_status_t synth_model_get_device(const synth_model_t *model,
                                      synth_backend_device_t *out_device);
```

`kind` is an extensible backend name such as `"cpu"`, `"accel"`, `"cuda"`, `"metal"`, `"vulkan"`, `"sycl"`, or `"unknown"`; there is no separate closed enum for the resolved backend. The enumeration index is supplied to `synth_backend_device_get()` and is not duplicated in the returned structure. `device_index == -1` remains the automatic-selection sentinel, so registry index `0` is explicitly selectable.

Device strings are borrowed from library-owned storage and remain valid for the
process lifetime because v1 never unloads backends. Device flags are append-only,
and callers ignore unknown future bits.

`memory_total` and `memory_free` are byte counts from the Execution Backend's view
of the device. They are populated only when `SYNTH_DEVICE_MEMORY_INFO_VALID` is
set; otherwise both are zero. `memory_free` is a snapshot refreshed by each query,
and both values are informational rather than an allocation or model-loading
guarantee. The implementation does not use these discovery values alone to reject
a model load; the actual backend allocation result is authoritative.

`SYNTH_DEVICE_MEMORY_SHARED` means that host and device draw from the same physical
memory pool, as on DGX Spark or an applicable integrated GPU. It does not mean that
GGML CUDA unified-memory fallback is enabled. `SYNTH_DEVICE_MEMORY_INFO_APPROXIMATE`
marks values whose useful capacity can be affected by operating-system reclamation,
caches, or another mechanism not represented by a dedicated device-memory counter.
DGX Spark therefore reports `VALID | SHARED | APPROXIMATE`; an ordinary discrete
CUDA device reports the CUDA backend's device-memory view without `SHARED`.

Memory data comes through the backend Integration. The CUDA Integration uses the
CUDA runtime or GGML CUDA query visible to the process; it does not parse
`nvidia-smi`, `/proc/meminfo`, or platform command output. Other backends may clear
`VALID` when they cannot supply meaningful values.

`synth_backend_available()` reports only whether a registered runtime device can
satisfy the request; Model Variant validation is checked separately during model
loading. `synth_model_get_device()` returns the Loaded Model's actual primary
device.

Implementation checkpoint (2026-07-22): the static CPU provider is wrapped by
this exact Interface. Each query refreshes GGML device properties, preserves a
backend-reported hardware ID or `NULL`, and marks CPU system-memory information
`VALID | APPROXIMATE`. Integrated GPU and accelerator memory is marked
`VALID | SHARED | APPROXIMATE`; Metal receives the same unified-memory flags even
though GGML classifies its device on the GPU axis. A zero reported total clears
all memory flags and both byte counts. The VITS loader resolves an explicit
global registry index before allocation and `synth_model_get_device()` then
queries the device attached to the backend actually owned by the model.

Device enumeration or `synth_backend_available(CUDA)` does not by itself make a
VITS/CUDA combination Supported. The VITS family now accepts a strict explicit
CUDA request in CUDA-enabled builds and reports the actual CUDA primary device;
`AUTO` remains on CPU, and explicit Metal or Vulkan requests remain unavailable.
The VITS/CUDA path is Experimental until the declared CUDA 13.3 Update 1 Provider
and release hardware matrix complete the same placement, numerical, performance,
and packaging gates. Native CUDA 13.3 Update 1 builds on DGX Spark/GB10 Linux
AArch64 and RTX 4070 SUPER Linux x86-64 have completed all seven 12-case Golden
stages with zero executable-node CPU fallback. That physical-Linux checkpoint
does not yet qualify the release Provider: numerical thresholds, R610, Windows,
the complete release cubin sets, and clean-runtime packages remain open.

Any implementation code adapted directly from transcribe.cpp retains its MIT license attribution. Model-family placement and scheduling remain synthesize.cpp implementations rather than copied STT graph policy.

## Synthesis Interface

A synthesis request identifies its input level, input data, selected voice or speaker conditioning, deterministic seed where applicable, and supported generation controls. One synthesis operation sends PCM frames and final metadata to an Audio Sink.

The callback sink and complete-buffer adapter share the same Audio Sink seam. They do not create separate family implementations or synthesis paths. Callback buffers are borrowed only for the callback duration; owned audio buffers have an explicit matching release operation.

The Interface returns project status codes and structured diagnostics rather than throwing exceptions across the C ABI or relying on process-global last-error state. Sink cancellation or failure is returned as an operation status, while an optional synthesis result carries resolved metadata and partial progress without duplicating that status.

### v1 Entry Points

v1 exposes `synth_synthesize()` as the only core synthesis operation. It synchronously accepts one Synthesis Context, request, and Audio Sink and does not return until synthesis completes, fails, or is cancelled.

`synth_synthesize_to_buffer()` is a complete-buffer adapter over the same operation, and its owned audio buffer is released with `synth_audio_buffer_free()`. Neither the adapter nor the command-line program receives a separate family or inference path.

Text, phoneme, and token-ID inputs are selected inside the request rather than through separate public functions. Model families, including VITS, cannot add family-specific synthesis entry points; their variation remains behind capability queries and versioned request fields.

### v1 Request Input Prefix

The synthesis request begins with a flat, binding-friendly input representation:

```c
typedef uint32_t synth_input_kind_t;
#define SYNTH_INPUT_TEXT_UTF8     ((synth_input_kind_t) 0u)
#define SYNTH_INPUT_PHONEMES_UTF8 ((synth_input_kind_t) 1u)
#define SYNTH_INPUT_TOKEN_IDS     ((synth_input_kind_t) 2u)

#define SYNTH_SEED_RANDOM UINT64_MAX

typedef synth_bool_t
(SYNTH_CALL *synth_cancel_callback_t)(void *user_data);

typedef struct synth_request {
    uint64_t struct_size;
    synth_input_kind_t input_kind;
    const void *input_data;
    uint64_t input_count;
    const char *language_tag;
    uint64_t language_tag_size;
    const char *voice_id;
    uint64_t voice_id_size;
    uint64_t seed;
    float speaking_rate;
    synth_cancel_callback_t should_cancel;
    void *cancel_user_data;
    const synth_diagnostic_sink_t *diagnostics;
    uint64_t max_output_frames;
    const synth_voice_profile_t *voice_profile;
    /* append-only v1 fields follow */
} synth_request_t;
```

For `TEXT_UTF8` and `PHONEMES_UTF8`, `input_data` points to UTF-8 bytes and `input_count` is the byte count. Phoneme text uses the representation required by the Model Variant's Text Frontend. For `TOKEN_IDS`, the pointer names a contiguous `int32_t` array and the count is the number of token IDs.

String input is length-delimited and need not be null-terminated. All input memory is borrowed and must remain valid only until the synchronous synthesis call returns. There is no separate `synth_input_t` or additional input initializer.

`language_tag` is length-delimited ASCII BCP 47, while `voice_id` is length-delimited UTF-8; neither requires null termination. `NULL` with a zero language size requests the Model Package default. A non-null `voice_id` must name a Preset Voice in the Loaded Model's immutable Preset Voice Catalog rather than a family-specific implementation identifier.

Language selection is explicit. The implementation never guesses a language from input text. A multilingual Model Variant without a declared default requires `language_tag`; this remains true for token-ID input because language conditioning is independent of the input representation.

Voice selection has three valid forms: `voice_id` selects a Preset Voice, `voice_profile` selects a prepared runtime Voice Profile, and leaving both absent requests the Model Package default. Setting both is `SYNTH_ERR_INVALID_ARG`; leaving both absent when no default exists is `SYNTH_ERR_UNSUPPORTED_VOICE`. A profile prepared by another Loaded Model is `SYNTH_ERR_UNSUPPORTED_VOICE`. The profile pointer is borrowed for the synchronous synthesis call.

Accent is Voice metadata, not a synthesis request field. v1 does not infer an accent or expose an undocumented accent control.

The initialized default `seed` is zero, making stochastic synthesis deterministic and reproducible by default. Every value except `SYNTH_SEED_RANDOM` is a deterministic seed; `SYNTH_SEED_RANDOM` asks the runtime to obtain a nondeterministic seed. The seed belongs to one synthesis operation rather than the Loaded Model, Synthesis Context, or Execution Backend. When a Model Variant consumes randomness, the synthesis result reports the actual seed used so the operation can be replayed. A deterministic variant accepts the field but does not use it or turn it into an error.

A fixed seed guarantees a repeatable logical random sequence for the same validated execution configuration. It does not promise bit-identical PCM across CPU, CUDA, Metal, or Vulkan because backend floating-point evaluation may differ within the project's validation tolerances. The library initializer keeps deterministic seed zero as its default; a CLI may offer an explicit `--random-seed` convenience that maps to `SYNTH_SEED_RANDOM`.

The request initializer sets `speaking_rate` to `1.0f`, meaning the Model Package's default speaking rate. Values greater than `1.0f` request faster speech and values below `1.0f` request slower speech; this is a relative multiplier, not an absolute words-, characters-, or syllables-per-second promise. The value must be finite and greater than zero.

The General Model Capability query advertises whether the Loaded Model supports non-default speaking rate and its validated range. Every variant accepts `1.0f`; a non-default value on an unsupported variant returns `SYNTH_ERR_UNSUPPORTED_CONTROL`, and a value outside the advertised range returns `SYNTH_ERR_INVALID_ARG`. The library does not silently ignore or clamp either case. Family implementations translate this semantic control internally—for example, VITS may map it inversely to `length_scale`—without exposing architecture-specific parameters to embedding programs or the CLI.

### v1 Output Frame Limit

`max_output_frames` is a request-level hard limit measured in PCM frames at the
Model Variant's native output sample rate. The request initializer sets it to
zero, meaning the caller imposes no additional limit. The General Model
Capability query's positive model limit always applies; a nonzero request limit
combines with it by taking the smaller value.

A nonzero value guarantees that the operation never exposes more than that many
frames. Natural completion at or below the limit returns `SYNTH_OK`. If the
implementation can determine before audio generation that completion would
exceed the effective limit, it returns `SYNTH_ERR_OUTPUT_LIMIT` without emitting
audio. If the limit is reached only during generation, the operation may emit
up to the limit and then returns `SYNTH_ERR_OUTPUT_LIMIT`. Limit termination is
never reported as successful truncation.

The rule lives in the core synthesis operation and is identical for a caller's
Audio Sink and the complete-buffer adapter. `synth_result_t.frames_emitted`
always reports the number of frames actually delivered, including zero for a
preflight rejection.

### v1 Cooperative Cancellation

`should_cancel == NULL` disables request-side cancellation. Otherwise the implementation polls the callback at model-stage, audio-block, and other available computation checkpoints on the same thread that called `synth_synthesize()`. The callback and `cancel_user_data` are borrowed only for that synchronous call; they are not installed as persistent Synthesis Context state. Another thread may request cancellation by updating thread-safe state, typically an atomic flag, that the callback reads.

The callback must be lightweight and non-blocking and must not re-enter the active Synthesis Context. Returning any nonzero value, conventionally `SYNTH_TRUE`, ends the operation with `SYNTH_ERR_CANCELLED` after temporary internal resources are released. Cancellation is cooperative and does not promise interruption inside one indivisible CPU or GPU backend operation or a fixed maximum latency.

An Audio Sink may also request cancellation through its own return value. Both mechanisms converge on the same cancellation status and cleanup path; v1 exposes no persistent Context setter or separate cancel function.

### v1 Audio Sink

```c
typedef uint32_t synth_sink_result_t;
#define SYNTH_SINK_CONTINUE ((synth_sink_result_t) 0u)
#define SYNTH_SINK_CANCEL   ((synth_sink_result_t) 1u)
#define SYNTH_SINK_ERROR    ((synth_sink_result_t) 2u)

typedef struct synth_audio_chunk {
    uint64_t struct_size;
    const float *samples;
    uint64_t frame_count;
    uint64_t frame_offset;
    uint32_t sample_rate;
    uint32_t channel_count;
} synth_audio_chunk_t;

typedef synth_sink_result_t
(SYNTH_CALL *synth_audio_write_callback_t)(
    void *user_data,
    const synth_audio_chunk_t *chunk);

typedef struct synth_audio_sink {
    uint64_t struct_size;
    synth_audio_write_callback_t write;
    void *user_data;
} synth_audio_sink_t;

void synth_audio_sink_init(synth_audio_sink_t *sink,
                           uint64_t struct_size);
```

The caller initializes the sink and supplies a non-null `write` callback; a null, undersized, or callback-less sink is invalid. The implementation invokes `write` synchronously on the thread calling `synth_synthesize()`. The chunk view and its interleaved F32 `samples` are borrowed and valid only until that callback returns.

Every delivered chunk has a positive `frame_count`. `frame_offset` begins at zero and equals the total frames delivered before that chunk. Sample rate and channel count remain constant throughout one operation. Returning from `synth_synthesize()` marks the end, so the library does not send a zero-length final chunk.

`SYNTH_SINK_CONTINUE` accepts the chunk. `SYNTH_SINK_CANCEL` follows the common `SYNTH_ERR_CANCELLED` cleanup path, while `SYNTH_SINK_ERROR` returns `SYNTH_ERR_SINK`; an unknown callback value is treated as a sink error. A model may deliver one complete chunk or many chunks, and chunk count alone makes no Native Streaming Synthesis guarantee.

`synth_synthesize_to_buffer()` is an in-library collecting adapter over this exact seam. It receives no alternate family path and releases its owned audio with `synth_audio_buffer_free()`.

### v1 Synthesis Result

```c
#define SYNTH_RESULT_SEED_USED             (1u << 0)
#define SYNTH_RESULT_NATIVE_STREAMING_USED (1u << 1)

typedef struct synth_result {
    uint64_t struct_size;
    uint64_t frames_emitted;
    uint64_t actual_seed;
    uint32_t sample_rate;
    uint32_t channel_count;
    uint32_t flags;
    const char *resolved_language_tag;
    uint64_t resolved_language_tag_size;
    const char *resolved_voice_id;
    uint64_t resolved_voice_id_size;
} synth_result_t;

void synth_result_init(synth_result_t *result,
                       uint64_t struct_size);

synth_status_t synth_synthesize(
    synth_context_t *context,
    const synth_request_t *request,
    const synth_audio_sink_t *sink,
    synth_result_t *out_result);
```

`out_result == NULL` is valid. Otherwise the caller initializes its storage, and the library writes only fields present in the supplied `struct_size`; the result itself requires no release operation. Status remains the function return value and is not duplicated inside the structure. Callers ignore unknown future flag bits.

`frames_emitted` counts every frame made visible to the Audio Sink, including the final chunk whose callback returns cancel or error and every frame delivered before an output limit terminates the operation. `actual_seed` is meaningful only with `SYNTH_RESULT_SEED_USED`; deterministic Model Variants leave that flag clear. `SYNTH_RESULT_NATIVE_STREAMING_USED` reports the path actually used by this operation rather than merely repeating a model capability or inferring streaming from chunk count.

The resolved language and Preset Voice identify package defaults and any permitted regional fallback. `resolved_language_tag` is length-delimited ASCII BCP 47 and `resolved_voice_id` is length-delimited UTF-8; both are borrowed from immutable Loaded Model metadata and remain valid until `synth_model_free()`. When a Voice Profile is selected, `resolved_voice_id` is null with size zero because a runtime profile is not inserted into the Preset Voice Catalog. On success, cancellation, output-limit termination, or sink error, the implementation fills every result field it has resolved and every frame it has emitted. Validation and other preflight failures leave all fields after `struct_size` at their initializer-provided zero values; an output-limit preflight rejection always reports zero emitted frames.

### v1 Complete-Buffer Adapter

```c
typedef struct synth_audio_buffer {
    uint64_t struct_size;
    const float *samples;
    uint64_t frame_count;
    uint32_t sample_rate;
    uint32_t channel_count;
} synth_audio_buffer_t;

synth_status_t synth_synthesize_to_buffer(
    synth_context_t *context,
    const synth_request_t *request,
    synth_audio_buffer_t **out_audio,
    synth_result_t *out_result);

void synth_audio_buffer_free(synth_audio_buffer_t *audio);
```

`out_audio` is required, and the library sets `*out_audio` to `NULL` before validation or synthesis. The library allocates both the public buffer structure and its read-only interleaved F32 samples; callers release the pair only with `synth_audio_buffer_free()`, which accepts `NULL`. The library-written `struct_size` describes the returned structure version.

Success returns the complete buffer. `SYNTH_ERR_CANCELLED` and `SYNTH_ERR_OUTPUT_LIMIT` return a partial buffer when at least one frame was collected and otherwise leave `*out_audio == NULL`. Invalid arguments, backend failures, sink failures internal to the collector, and `SYNTH_ERR_OOM` release all partial storage and return no buffer. When `out_result` is non-null, `frames_emitted` equals the returned buffer's `frame_count`, or zero when no buffer is returned.

This adapter is implemented by collecting the same Audio Sink callbacks used by `synth_synthesize()`. v1 has no caller-preallocated output form because final length is not generally known before synthesis; hosts that need bounded buffering, zero-copy handoff, or real-time consumption use the Audio Sink directly.

## v1 Conventions

All public symbols use the `synth_` prefix. Synthesis calls are synchronous: the library does not create background worker threads, and a call returns only after synthesis completes, fails, or is cancelled.

The canonical audio representation is interleaved F32 PCM with explicit sample rate, channel count, and frame count. A complete audio buffer is allocated by the library and released by the caller with `synth_audio_buffer_free()`.

v1 does not expose a custom allocator. This keeps allocation policy out of the initial public Interface while leaving room to add allocator callbacks later through versioned structures if concrete embedding requirements justify another adapter.

## Streaming Semantics

Chunked Audio Delivery is always an output-delivery option and may internally buffer an entire utterance. It provides no time-to-first-audio or bounded-memory guarantee by itself.

Native Streaming Synthesis is a separate Loaded Model capability. `SYNTH_MODEL_CAPABILITY_NATIVE_STREAMING` is advertised only after the specific Model Package and Execution Backend pass incremental-output, latency, continuity, cancellation, and long-running-state validation. Callers query it without knowing the model family, while `SYNTH_RESULT_NATIVE_STREAMING_USED` reports the path actually used by an operation.

## Internal Modules

Model-family implementations and Text Frontend Providers are selected behind internal seams. They are not public handles and cannot add family-specific public entry points; model-specific variation is expressed through capability and metadata queries plus versioned request fields.
