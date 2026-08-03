#include "synthesize.h"

#include "arch/kokoro/kokoro.h"
#include "arch/omnivoice/omnivoice.h"
#include "arch/omnivoice/profile.h"
#include "arch/qwen3-tts/qwen3-tts.h"
#include "arch/vits/vits.h"
#include "audio-delivery.h"
#include "backend-device.h"
#include "backend-module.h"
#include "cpu-parallelism.h"
#include "gguf-metadata.h"
#include "gguf.h"
#include "model-handle.h"
#include "model-info.h"
#include "random-stream.h"
#include "synthesis-request.h"
#include "voice-profile-handle.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <thread>
#include <vector>

struct synth_context {
    const synth_model_t * model   = nullptr;
    int                   threads = 1;
    std::atomic_flag      active  = ATOMIC_FLAG_INIT;
};

namespace {

template <typename T> void write_visible(T * output, size_t offset, const void * value, size_t value_size) {
    if (output == nullptr || output->struct_size < offset + value_size) {
        return;
    }
    std::memcpy(reinterpret_cast<unsigned char *>(output) + offset, value, value_size);
}

template <typename T>
void write_initializer_default(T * output, uint64_t struct_size, size_t offset, const void * value, size_t value_size) {
    if (output == nullptr || struct_size < offset + value_size) {
        return;
    }
    std::memcpy(reinterpret_cast<unsigned char *>(output) + offset, value, value_size);
}

template <typename T> void initialize_struct(T * output, uint64_t struct_size) {
    if (output == nullptr || struct_size == 0) {
        return;
    }
    std::memset(output, 0, std::min(static_cast<size_t>(struct_size), sizeof(*output)));
    if (struct_size >= sizeof(output->struct_size)) {
        output->struct_size = struct_size;
    }
}

template <typename T, typename V> V read_visible(const T * input, size_t offset, V default_value) {
    if (input == nullptr || input->struct_size < offset + sizeof(V)) {
        return default_value;
    }
    V value;
    std::memcpy(&value, reinterpret_cast<const unsigned char *>(input) + offset, sizeof(value));
    return value;
}

bool valid_diagnostic_sink(const synth_diagnostic_sink_t * sink) {
    return sink == nullptr ||
           (sink->struct_size >= offsetof(synth_diagnostic_sink_t, emit) + sizeof(sink->emit) && sink->emit != nullptr);
}

void emit_diagnostic(const synth_diagnostic_sink_t * sink,
                     synth_status_t                  status,
                     const char *                    code,
                     const char *                    message) {
    if (sink == nullptr || !valid_diagnostic_sink(sink)) {
        return;
    }
    synth_diagnostic_t diagnostic{};
    diagnostic.struct_size  = sizeof(diagnostic);
    diagnostic.level        = SYNTH_DIAGNOSTIC_ERROR;
    diagnostic.status       = status;
    diagnostic.code         = code;
    diagnostic.code_size    = std::strlen(code);
    diagnostic.message      = message;
    diagnostic.message_size = std::strlen(message);
    try {
        sink->emit(sink->user_data, &diagnostic);
    } catch (...) {
        // Foreign callbacks may not unwind through the public C ABI.
    }
}

class ContextLease {
  public:
    explicit ContextLease(synth_context_t * context) : context_(context) {
        acquired_ = context_ != nullptr && !context_->active.test_and_set(std::memory_order_acquire);
    }

    ~ContextLease() {
        if (acquired_) {
            context_->active.clear(std::memory_order_release);
        }
    }

    bool acquired() const { return acquired_; }

  private:
    synth_context_t * context_  = nullptr;
    bool              acquired_ = false;
};

bool cancellation_requested(const synth::PreparedSynthesisRequest & request) {
    if (request.should_cancel == nullptr) {
        return false;
    }
    try {
        return request.should_cancel(request.cancel_user_data) != SYNTH_FALSE;
    } catch (...) {
        return true;
    }
}

synth::AudioDeliveryInfo make_delivery_info(const synth_context_t *                 context,
                                            const synth::PreparedSynthesisRequest & request,
                                            uint64_t                                actual_seed) {
    synth::AudioDeliveryInfo info;
    info.actual_seed            = actual_seed;
    info.sample_rate            = context->model->info.output_sample_rate;
    info.channel_count          = context->model->info.output_channel_count;
    info.result_flags           = SYNTH_RESULT_SEED_USED;
    info.resolved_language_tag  = request.resolved_language_tag;
    info.resolved_language_size = request.resolved_language_size;
    info.resolved_voice_id      = request.resolved_voice_id;
    info.resolved_voice_size    = request.resolved_voice_size;
    return info;
}

struct BufferCollector {
    std::vector<float> samples;
    uint32_t           sample_rate       = 0;
    uint32_t           channel_count     = 0;
    bool               allocation_failed = false;
};

synth_sink_result_t SYNTH_CALL collect_audio(void * user_data, const synth_audio_chunk_t * chunk) {
    auto * collector = static_cast<BufferCollector *>(user_data);
    if (collector == nullptr || chunk == nullptr || chunk->struct_size < sizeof(synth_audio_chunk_t) ||
        chunk->samples == nullptr || chunk->frame_count == 0 || chunk->channel_count == 0 ||
        chunk->frame_count > std::numeric_limits<size_t>::max() / chunk->channel_count) {
        return SYNTH_SINK_ERROR;
    }
    const size_t sample_count = static_cast<size_t>(chunk->frame_count) * chunk->channel_count;
    if ((!collector->samples.empty() &&
         (collector->sample_rate != chunk->sample_rate || collector->channel_count != chunk->channel_count)) ||
        (collector->channel_count != 0 &&
         chunk->frame_offset != collector->samples.size() / collector->channel_count)) {
        return SYNTH_SINK_ERROR;
    }
    collector->sample_rate   = chunk->sample_rate;
    collector->channel_count = chunk->channel_count;
    try {
        collector->samples.insert(collector->samples.end(), chunk->samples, chunk->samples + sample_count);
    } catch (const std::bad_alloc &) {
        collector->allocation_failed = true;
        return SYNTH_SINK_ERROR;
    } catch (...) {
        return SYNTH_SINK_ERROR;
    }
    return SYNTH_SINK_CONTINUE;
}

// An output limit fills the result metadata it resolved, per docs/c-interface.md:
// "On success, cancellation, output-limit termination, or sink error, the
// implementation fills every result field it has resolved."
//
// It is not a sink completion. deliver_complete_audio returns before touching the
// sink when the frame count is zero, and the ABI has no completion callback --
// the contract forbids a zero-length final chunk. What was missing was the
// metadata: a limit stop reported actual_seed 0, sample_rate 0 and an empty
// resolved Voice, while the two sibling families filled all of it.
//
// Nor is it a graph failure. Routing it through the generic branch also emitted a
// `synthesis.graph_failed` diagnostic, which the sibling paths do not.
synth_status_t report_output_limit(const synth::AudioDeliveryInfo & info,
                                   const synth_audio_sink_t *       sink,
                                   synth_result_t *                 out_result) {
    (void) synth::deliver_complete_audio(nullptr, 0, info, sink, out_result);
    return SYNTH_ERR_OUTPUT_LIMIT;
}

void reset_result(synth_result_t * result) {
    if (result != nullptr) {
        const uint64_t struct_size = result->struct_size;
        synth_result_init(result, struct_size);
    }
}

// Reads the package's declared architecture so the right family loader is
// called. A container that names no architecture, or names one this build has
// no family for, is refused here rather than by whichever loader happened to be
// tried first.
synth_status_t read_model_family(const char * model_path, synth::ModelFamily & family) {
    // A missing file has to keep reporting as missing. The probe runs before
    // any family loader, so it owns that distinction now.
    std::ifstream probe(model_path, std::ios::binary);
    if (!probe) {
        return SYNTH_ERR_FILE_NOT_FOUND;
    }
    probe.close();

    gguf_init_params parameters{};
    parameters.no_alloc = true;
    parameters.ctx      = nullptr;
    gguf_context * gguf = gguf_init_from_file(model_path, parameters);
    if (gguf == nullptr) {
        return SYNTH_ERR_GGUF;
    }
    std::string architecture;
    const bool  present = synth::GgufMetadata(gguf, "synthesize").string("general.architecture", architecture);
    gguf_free(gguf);
    if (!present) {
        return SYNTH_ERR_GGUF;
    }
    if (architecture == "vits") {
        family = synth::ModelFamily::Vits;
        return SYNTH_OK;
    }
    if (architecture == "kokoro") {
        family = synth::ModelFamily::Kokoro;
        return SYNTH_OK;
    }
    if (architecture == "qwen3-tts") {
        family = synth::ModelFamily::Qwen3Tts;
        return SYNTH_OK;
    }
    if (architecture == "omnivoice") {
        family = synth::ModelFamily::Omnivoice;
        return SYNTH_OK;
    }
    return SYNTH_ERR_UNSUPPORTED_ARCH;
}

synth::ModelInfo shared_info(const synth::vits::ModelInfo & info) {
    synth::ModelInfo shared;
    shared.family              = synth::ModelFamily::Vits;
    shared.has_package_default = info.has_package_default;
    shared.preset_voice_ids    = info.preset_voice_ids;
    shared.preset_voice_flags  = info.preset_voice_flags;
    // English-only, and now says so rather than relying on the validator having
    // assumed it. Regional fallback keeps "en-GB" and "en-029" accepted.
    shared.languages           = {
        { "en", SYNTH_LANGUAGE_DEFAULT | SYNTH_LANGUAGE_REGIONAL_FALLBACK }
    };
    shared.text_frontend        = info.text_frontend;
    shared.input_flags          = info.input_flags;
    shared.capability_flags     = info.capability_flags;
    shared.output_sample_rate   = info.output_sample_rate;
    shared.output_channel_count = info.output_channel_count;
    shared.vocab_size           = info.vocab_size;
    shared.samples_per_frame    = info.hop_length;
    shared.max_input_tokens     = info.max_input_tokens;
    shared.max_output_frames    = info.max_output_frames;
    shared.min_speaking_rate    = info.min_speaking_rate;
    shared.max_speaking_rate    = info.max_speaking_rate;
    return shared;
}

synth::ModelInfo shared_info(const synth::kokoro::ModelInfo & info) {
    synth::ModelInfo shared;
    shared.family              = synth::ModelFamily::Kokoro;
    shared.has_package_default = info.has_package_default;
    shared.preset_voice_ids    = info.preset_voice_ids;
    shared.preset_voice_flags  = info.preset_voice_flags;
    shared.languages           = {
        { "en", SYNTH_LANGUAGE_DEFAULT | SYNTH_LANGUAGE_REGIONAL_FALLBACK }
    };
    shared.text_frontend        = info.text_frontend;
    shared.input_flags          = info.input_flags;
    shared.capability_flags     = info.capability_flags;
    shared.output_sample_rate   = info.output_sample_rate;
    shared.output_channel_count = info.output_channel_count;
    shared.vocab_size           = info.vocab_size;
    shared.samples_per_frame    = info.samples_per_frame;
    shared.max_input_tokens     = info.max_input_tokens;
    shared.max_output_frames    = info.max_output_frames;
    shared.min_speaking_rate    = info.min_speaking_rate;
    shared.max_speaking_rate    = info.max_speaking_rate;
    return shared;
}

synth::ModelInfo shared_info(const synth::qwen3tts::ModelInfo &         info,
                             std::shared_ptr<const synth::TextFrontend> frontend,
                             uint32_t                                   samples_per_frame,
                             uint32_t                                   text_vocab_size) {
    synth::ModelInfo shared;
    shared.family              = synth::ModelFamily::Qwen3Tts;
    shared.has_package_default = info.has_package_default;
    shared.preset_voice_ids    = info.preset_voice_ids;
    // This family declares no per-Voice flags, so every entry is zero rather
    // than absent: the two lists are read in step.
    shared.preset_voice_flags.assign(info.preset_voice_ids.size(), 0);
    // Every language the package declares, not just the one the validator used
    // to allow. No entry is marked DEFAULT: a request that names no language
    // gets this family's no-think prompt, which carries no language token at
    // all, so there is no language it silently falls back to.
    for (const std::string & tag : info.language_tags) {
        shared.languages.push_back({ tag, SYNTH_LANGUAGE_REGIONAL_FALLBACK });
    }
    shared.text_frontend        = std::move(frontend);
    shared.input_flags          = info.input_flags;
    shared.capability_flags     = info.capability_flags;
    shared.output_sample_rate   = info.output_sample_rate;
    shared.output_channel_count = info.output_channel_count;
    // The ids the core range-checks come from the text frontend, so this is
    // the text tower's vocabulary rather than the codec's.
    shared.vocab_size           = text_vocab_size;
    shared.samples_per_frame    = samples_per_frame;
    shared.max_input_tokens     = info.max_input_tokens;
    shared.max_output_frames    = info.max_output_frames;
    shared.min_speaking_rate    = info.min_speaking_rate;
    shared.max_speaking_rate    = info.max_speaking_rate;
    return shared;
}

synth::ModelInfo shared_info(const synth::omnivoice::ModelInfo &        info,
                             std::shared_ptr<const synth::TextFrontend> frontend,
                             uint32_t                                   samples_per_frame,
                             uint32_t                                   text_vocab_size) {
    synth::ModelInfo shared;
    shared.family              = synth::ModelFamily::Omnivoice;
    shared.has_package_default = info.has_package_default;
    // The Preset Voice Catalog is empty by design: identity arrives through
    // Voice Profiles, and the unnamed package default is auto-voice.
    for (const std::string & tag : info.language_tags) {
        shared.languages.push_back({ tag, SYNTH_LANGUAGE_REGIONAL_FALLBACK });
    }
    shared.text_frontend        = std::move(frontend);
    shared.input_flags          = info.input_flags;
    shared.capability_flags     = info.capability_flags;
    shared.output_sample_rate   = info.output_sample_rate;
    shared.output_channel_count = info.output_channel_count;
    shared.vocab_size           = text_vocab_size;
    shared.samples_per_frame    = samples_per_frame;
    shared.max_input_tokens     = info.max_input_tokens;
    shared.max_output_frames    = info.max_output_frames;
    shared.min_speaking_rate    = info.min_speaking_rate;
    shared.max_speaking_rate    = info.max_speaking_rate;

    // Voice Profile capabilities: REFERENCE_AUDIO (Task 14) and
    // DESCRIPTION_TEXT (Task 15) dispatch for real; SERIALIZED_PROFILE
    // (Task 16) now does too -- serialize/load_from_memory dispatch on
    // family in voice-profile.cpp, backed by arch/omnivoice/profile.cpp's
    // GGUF envelope writer/reader. docs/c-interface.md: "Any Loaded Model
    // that creates a Profile from Reference Audio, Description Text, or
    // Random Seed also sets SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE, because
    // every successfully prepared v1 Profile can be serialized" -- this
    // family creates from the first two, so it claims the third
    // unconditionally alongside them.
    synth::VoiceProfileInfo & profile = shared.voice_profile;
    profile.source_flags              = SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO | SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT |
                                        SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE;
    profile.reference_transcript      = SYNTH_REQUIREMENT_REQUIRED;
    profile.reference_language        = SYNTH_REQUIREMENT_OPTIONAL;
    profile.description_language      = SYNTH_REQUIREMENT_OPTIONAL;
    profile.reference_target_sample_rate = info.profile.reference_sample_rate;
    profile.reference_target_channels    = info.profile.reference_channels;
    profile.min_frames_per_clip          = info.profile.min_frames_per_clip;
    profile.max_frames_per_clip          = info.profile.max_frames_per_clip;
    profile.max_total_frames             = info.profile.max_total_frames;
    profile.max_reference_count          = static_cast<uint32_t>(info.profile.max_reference_count);
    // Best-effort: HParams::profile is already validated at load time
    // (weights.cpp's read_profile_contract, is_sha256_hex), so this should
    // never fail for a package that made it this far; a defect that slipped
    // through leaves the bytes at their all-zero default rather than
    // propagating a load failure this deep into shared_info.
    (void) synth::decode_profile_compatibility_id(info.profile.compatibility_id_hex, profile.compatibility_id);
    // The Serialized Profile schema identity the same v1 envelope declares
    // (arch/omnivoice/profile.cpp's kEnvelopeSchema/kEnvelopeSchemaVersion
    // are this exact pair, by construction -- weights.cpp's
    // read_profile_contract already refused any package that disagrees).
    profile.schema         = info.profile.schema;
    profile.schema_version = info.profile.schema_version;

    return shared;
}

}  // namespace

uint32_t synth_abi_version(void) {
    return SYNTH_ABI_VERSION;
}

const char * synth_status_string(synth_status_t status) {
    switch (status) {
        case SYNTH_OK:
            return "ok";
        case SYNTH_ERR_INVALID_ARG:
            return "invalid argument";
        case SYNTH_ERR_BAD_STRUCT_SIZE:
            return "bad structure size";
        case SYNTH_ERR_FILE_NOT_FOUND:
            return "file not found";
        case SYNTH_ERR_IO:
            return "I/O error";
        case SYNTH_ERR_GGUF:
            return "invalid GGUF";
        case SYNTH_ERR_UNSUPPORTED_ARCH:
            return "unsupported architecture";
        case SYNTH_ERR_UNSUPPORTED_VARIANT:
            return "unsupported model variant";
        case SYNTH_ERR_UNSUPPORTED_INPUT:
            return "unsupported input";
        case SYNTH_ERR_UNSUPPORTED_LANGUAGE:
            return "unsupported language";
        case SYNTH_ERR_UNSUPPORTED_VOICE:
            return "unsupported voice";
        case SYNTH_ERR_UNSUPPORTED_CONTROL:
            return "unsupported control";
        case SYNTH_ERR_MISSING_RESOURCE:
            return "missing resource";
        case SYNTH_ERR_TEXT_FRONTEND:
            return "text frontend error";
        case SYNTH_ERR_INPUT_TOO_LONG:
            return "input too long";
        case SYNTH_ERR_OUTPUT_LIMIT:
            return "output limit reached";
        case SYNTH_ERR_OOM:
            return "out of memory";
        case SYNTH_ERR_BACKEND:
            return "backend error";
        case SYNTH_ERR_CANCELLED:
            return "cancelled";
        case SYNTH_ERR_SINK:
            return "audio sink error";
        case SYNTH_ERR_INTERNAL:
            return "internal error";
        default:
            return "unknown status";
    }
}

void synth_version_init(synth_version_t * version, uint64_t struct_size) {
    if (version == nullptr || struct_size == 0) {
        return;
    }
    std::memset(version, 0, std::min(static_cast<size_t>(struct_size), sizeof(*version)));
    if (struct_size >= sizeof(version->struct_size)) {
        version->struct_size = struct_size;
    }
}

synth_status_t synth_get_version(synth_version_t * out_version) {
    if (out_version == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (out_version->struct_size < sizeof(uint64_t)) {
        return SYNTH_ERR_BAD_STRUCT_SIZE;
    }
    const uint32_t abi   = SYNTH_ABI_VERSION;
    const uint32_t major = SYNTH_VERSION_MAJOR;
    const uint32_t minor = SYNTH_VERSION_MINOR;
    const uint32_t patch = SYNTH_VERSION_PATCH;
    write_visible(out_version, offsetof(synth_version_t, abi_version), &abi, sizeof(abi));
    write_visible(out_version, offsetof(synth_version_t, version_major), &major, sizeof(major));
    write_visible(out_version, offsetof(synth_version_t, version_minor), &minor, sizeof(minor));
    write_visible(out_version, offsetof(synth_version_t, version_patch), &patch, sizeof(patch));
    return SYNTH_OK;
}

void synth_diagnostic_sink_init(synth_diagnostic_sink_t * sink, uint64_t struct_size) {
    initialize_struct(sink, struct_size);
}

uint32_t synth_backend_device_count(void) {
    try {
        return synth::backend_device_count();
    } catch (...) {
        return 0;
    }
}

void synth_backend_device_init(synth_backend_device_t * device, uint64_t struct_size) {
    initialize_struct(device, struct_size);
}

synth_status_t synth_backend_device_get(uint32_t index, synth_backend_device_t * out_device) {
    if (out_device == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    try {
        return synth::get_backend_device(index, out_device->struct_size, out_device);
    } catch (const std::bad_alloc &) {
        return SYNTH_ERR_OOM;
    } catch (...) {
        return SYNTH_ERR_INTERNAL;
    }
}

synth_bool_t synth_backend_available(synth_backend_request_t request) {
    try {
        return synth::backend_available(request);
    } catch (...) {
        return SYNTH_FALSE;
    }
}

synth_status_t synth_model_get_device(const synth_model_t * model, synth_backend_device_t * out_device) {
    if (model == nullptr || out_device == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    try {
        ggml_backend_dev_t device = model->qwen3_tts != nullptr ? model->qwen3_tts->primary_device() :
                                    model->omnivoice != nullptr ? model->omnivoice->primary_device() :
                                    model->kokoro != nullptr    ? model->kokoro->primary_device() :
                                    model->vits != nullptr      ? model->vits->primary_device() :
                                                                  nullptr;
        if (device == nullptr) {
            return SYNTH_ERR_BACKEND;
        }
        return synth::get_backend_device(device, out_device->struct_size, out_device);
    } catch (const std::bad_alloc &) {
        return SYNTH_ERR_OOM;
    } catch (...) {
        return SYNTH_ERR_INTERNAL;
    }
}

void synth_model_load_params_init(synth_model_load_params_t * params, uint64_t struct_size) {
    initialize_struct(params, struct_size);
    const int32_t automatic_device = -1;
    write_initializer_default(params, struct_size, offsetof(synth_model_load_params_t, device_index), &automatic_device,
                              sizeof(automatic_device));
}

synth_status_t synth_model_load(const char *                      model_path,
                                const synth_model_load_params_t * params,
                                synth_model_t **                  out_model) {
    if (out_model == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    *out_model = nullptr;
    if (model_path == nullptr || model_path[0] == '\0') {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (params != nullptr && params->struct_size < sizeof(uint64_t)) {
        return SYNTH_ERR_BAD_STRUCT_SIZE;
    }
    const synth_backend_request_t backend =
        read_visible(params, offsetof(synth_model_load_params_t, backend), SYNTH_BACKEND_AUTO);
    const int32_t device_index = read_visible(params, offsetof(synth_model_load_params_t, device_index), int32_t(-1));
    const synth_diagnostic_sink_t * diagnostics = read_visible(params, offsetof(synth_model_load_params_t, diagnostics),
                                                               static_cast<const synth_diagnostic_sink_t *>(nullptr));
    if (!valid_diagnostic_sink(diagnostics) || device_index < -1 || backend > SYNTH_BACKEND_VULKAN) {
        return SYNTH_ERR_INVALID_ARG;
    }
    try {
        synth::freeze_backend_modules();
        synth::ModelFamily   family        = synth::ModelFamily::Vits;
        const synth_status_t family_status = read_model_family(model_path, family);
        if (family_status != SYNTH_OK) {
            emit_diagnostic(diagnostics, family_status, "model.unsupported_architecture",
                            "the package does not declare an architecture this build supports");
            return family_status;
        }
        ggml_backend_dev_t   selected_device   = nullptr;
        // Both families claim the same execution backends today; when they
        // diverge this becomes a per-family question.
        const bool           backend_supported = backend == SYNTH_BACKEND_AUTO || backend == SYNTH_BACKEND_CPU ||
                                                 backend == SYNTH_BACKEND_CPU_ACCEL || backend == SYNTH_BACKEND_CUDA;
        const synth_status_t selection_status =
            backend_supported ? synth::resolve_requested_device(backend, device_index, &selected_device) :
                                SYNTH_ERR_BACKEND;
        if (selection_status != SYNTH_OK) {
            emit_diagnostic(
                diagnostics, selection_status, backend_supported ? "backend.device_unavailable" : "backend.unavailable",
                backend_supported ?
                    "the requested execution device is unavailable or does not match the requested backend" :
                    "the requested execution backend is unavailable for this model family");
            return selection_status;
        }
        const bool     include_accelerators = backend != SYNTH_BACKEND_CPU;
        auto           model                = std::make_unique<synth_model>();
        synth_status_t status               = SYNTH_OK;
        if (family == synth::ModelFamily::Qwen3Tts) {
            status = synth::qwen3tts::Model::load(model_path, selected_device, include_accelerators, model->qwen3_tts);
            if (status == SYNTH_OK) {
                synth::qwen3tts::ModelInfo info;
                status = model->qwen3_tts->get_info(info);
                if (status == SYNTH_OK) {
                    model->info =
                        shared_info(info, model->qwen3_tts->text_frontend(), model->qwen3_tts->samples_per_frame(),
                                    model->qwen3_tts->text_vocab_size());
                }
            }
        } else if (family == synth::ModelFamily::Omnivoice) {
            status = synth::omnivoice::Model::load(model_path, selected_device, include_accelerators, model->omnivoice);
            if (status == SYNTH_OK) {
                synth::omnivoice::ModelInfo info;
                status = model->omnivoice->get_info(info);
                if (status == SYNTH_OK) {
                    model->info =
                        shared_info(info, model->omnivoice->text_frontend(), model->omnivoice->samples_per_frame(),
                                    model->omnivoice->text_vocab_size());
                }
            }
        } else if (family == synth::ModelFamily::Kokoro) {
            status = synth::kokoro::Model::load(model_path, selected_device, include_accelerators, model->kokoro);
            if (status == SYNTH_OK) {
                synth::kokoro::ModelInfo info;
                status = model->kokoro->get_info(info);
                if (status == SYNTH_OK) {
                    model->info = shared_info(info);
                }
            }
        } else {
            status = synth::vits::Model::load(model_path, selected_device, include_accelerators, model->vits);
            if (status == SYNTH_OK) {
                status = model->vits->get_info(model->vits_extras);
                if (status == SYNTH_OK) {
                    model->info = shared_info(model->vits_extras);
                }
            }
        }
        if (status != SYNTH_OK) {
            emit_diagnostic(diagnostics, status, "model.load_failed", synth_status_string(status));
            return status;
        }
        *out_model = model.release();
        return SYNTH_OK;
    } catch (const std::bad_alloc &) {
        emit_diagnostic(diagnostics, SYNTH_ERR_OOM, "allocation.failed", "model handle allocation failed");
        return SYNTH_ERR_OOM;
    } catch (...) {
        emit_diagnostic(diagnostics, SYNTH_ERR_INTERNAL, "internal.exception",
                        "unexpected exception while loading the model");
        return SYNTH_ERR_INTERNAL;
    }
}

void synth_model_free(synth_model_t * model) {
    delete model;
}

synth_status_t synth_context_create(const synth_model_t * model, synth_context_t ** out_context) {
    if (out_context == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    *out_context = nullptr;
    if (model == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    try {
        auto context     = std::make_unique<synth_context>();
        context->model   = model;
        context->threads = synth::default_synthesis_threads();
        context->active.clear();
        *out_context = context.release();
        return SYNTH_OK;
    } catch (const std::bad_alloc &) {
        return SYNTH_ERR_OOM;
    } catch (...) {
        return SYNTH_ERR_INTERNAL;
    }
}

void synth_context_free(synth_context_t * context) {
    delete context;
}

synth_status_t synth_context_set_threads(synth_context_t * context, int32_t threads) {
    if (context == nullptr || threads < 0) {
        return SYNTH_ERR_INVALID_ARG;
    }
    // Taking the lease refuses the call while a synthesis is running on this
    // context, rather than letting the count change under a graph mid-flight.
    ContextLease lease(context);
    if (!lease.acquired()) {
        return SYNTH_ERR_INVALID_ARG;
    }
    context->threads = threads == 0 ? synth::default_synthesis_threads() : threads;
    return SYNTH_OK;
}

synth_status_t synth_context_get_threads(const synth_context_t * context, int32_t * out_threads) {
    if (context == nullptr || out_threads == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    *out_threads = context->threads;
    return SYNTH_OK;
}

void synth_model_capabilities_init(synth_model_capabilities_t * capabilities, uint64_t struct_size) {
    initialize_struct(capabilities, struct_size);
}

synth_status_t synth_model_get_capabilities(const synth_model_t *        model,
                                            synth_model_capabilities_t * out_capabilities) {
    if (model == nullptr || out_capabilities == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (out_capabilities->struct_size < sizeof(uint64_t)) {
        return SYNTH_ERR_BAD_STRUCT_SIZE;
    }
    const uint64_t struct_size = out_capabilities->struct_size;
    synth_model_capabilities_init(out_capabilities, struct_size);
    write_visible(out_capabilities, offsetof(synth_model_capabilities_t, input_flags), &model->info.input_flags,
                  sizeof(model->info.input_flags));
    write_visible(out_capabilities, offsetof(synth_model_capabilities_t, capability_flags),
                  &model->info.capability_flags, sizeof(model->info.capability_flags));
    write_visible(out_capabilities, offsetof(synth_model_capabilities_t, output_sample_rate),
                  &model->info.output_sample_rate, sizeof(model->info.output_sample_rate));
    write_visible(out_capabilities, offsetof(synth_model_capabilities_t, output_channel_count),
                  &model->info.output_channel_count, sizeof(model->info.output_channel_count));
    write_visible(out_capabilities, offsetof(synth_model_capabilities_t, min_speaking_rate),
                  &model->info.min_speaking_rate, sizeof(model->info.min_speaking_rate));
    write_visible(out_capabilities, offsetof(synth_model_capabilities_t, max_speaking_rate),
                  &model->info.max_speaking_rate, sizeof(model->info.max_speaking_rate));
    write_visible(out_capabilities, offsetof(synth_model_capabilities_t, max_input_tokens),
                  &model->info.max_input_tokens, sizeof(model->info.max_input_tokens));
    write_visible(out_capabilities, offsetof(synth_model_capabilities_t, max_output_frames),
                  &model->info.max_output_frames, sizeof(model->info.max_output_frames));
    return SYNTH_OK;
}

void synth_preset_voice_init(synth_preset_voice_t * voice, uint64_t struct_size) {
    initialize_struct(voice, struct_size);
}

synth_status_t synth_model_get_preset_voice_count(const synth_model_t * model, uint64_t * out_count) {
    if (out_count == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    *out_count = 0;
    if (model == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    *out_count = model->info.preset_voice_ids.size();
    return SYNTH_OK;
}

synth_status_t synth_model_get_preset_voice(const synth_model_t *  model,
                                            uint64_t               index,
                                            synth_preset_voice_t * out_voice) {
    if (model == nullptr || out_voice == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (out_voice->struct_size < sizeof(uint64_t)) {
        return SYNTH_ERR_BAD_STRUCT_SIZE;
    }
    const uint64_t struct_size = out_voice->struct_size;
    synth_preset_voice_init(out_voice, struct_size);
    if (index >= model->info.preset_voice_ids.size()) {
        return SYNTH_ERR_INVALID_ARG;
    }
    const std::string &              id      = model->info.preset_voice_ids[index];
    const char *                     id_data = id.c_str();
    const uint64_t                   id_size = id.size();
    const synth_preset_voice_flags_t flags   = model->info.preset_voice_flags[index];
    write_visible(out_voice, offsetof(synth_preset_voice_t, id), &id_data, sizeof(id_data));
    write_visible(out_voice, offsetof(synth_preset_voice_t, id_size), &id_size, sizeof(id_size));
    write_visible(out_voice, offsetof(synth_preset_voice_t, flags), &flags, sizeof(flags));
    return SYNTH_OK;
}

void synth_language_capability_init(synth_language_capability_t * language, uint64_t struct_size) {
    initialize_struct(language, struct_size);
}

synth_status_t synth_model_get_language_count(const synth_model_t * model, uint64_t * out_count) {
    if (out_count == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    *out_count = 0;
    if (model == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    *out_count = model->info.languages.size();
    return SYNTH_OK;
}

synth_status_t synth_model_get_language(const synth_model_t *         model,
                                        uint64_t                      index,
                                        synth_language_capability_t * out_language) {
    if (model == nullptr || out_language == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (out_language->struct_size < sizeof(uint64_t)) {
        return SYNTH_ERR_BAD_STRUCT_SIZE;
    }
    const uint64_t struct_size = out_language->struct_size;
    synth_language_capability_init(out_language, struct_size);
    if (index >= model->info.languages.size()) {
        return SYNTH_ERR_INVALID_ARG;
    }
    // Borrowed from the Loaded Model, which outlives the capability the caller
    // is handed, exactly as the preset Voice ids are.
    const synth::LanguageCapability & entry    = model->info.languages[static_cast<size_t>(index)];
    const char *                      tag      = entry.tag.c_str();
    const uint64_t                    tag_size = entry.tag.size();
    const synth_language_flags_t      flags    = static_cast<synth_language_flags_t>(entry.flags);
    write_visible(out_language, offsetof(synth_language_capability_t, tag), &tag, sizeof(tag));
    write_visible(out_language, offsetof(synth_language_capability_t, tag_size), &tag_size, sizeof(tag_size));
    write_visible(out_language, offsetof(synth_language_capability_t, flags), &flags, sizeof(flags));
    return SYNTH_OK;
}

void synth_request_init(synth_request_t * request, uint64_t struct_size) {
    initialize_struct(request, struct_size);
    const float speaking_rate = 1.0f;
    write_initializer_default(request, struct_size, offsetof(synth_request_t, speaking_rate), &speaking_rate,
                              sizeof(speaking_rate));
}

void synth_audio_sink_init(synth_audio_sink_t * sink, uint64_t struct_size) {
    initialize_struct(sink, struct_size);
}

void synth_result_init(synth_result_t * result, uint64_t struct_size) {
    initialize_struct(result, struct_size);
}

synth_status_t synth_synthesize(synth_context_t *          context,
                                const synth_request_t *    request,
                                const synth_audio_sink_t * sink,
                                synth_result_t *           out_result) {
    if (context == nullptr || context->model == nullptr || !synth::valid_audio_sink(sink)) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (out_result != nullptr && out_result->struct_size < sizeof(uint64_t)) {
        return SYNTH_ERR_BAD_STRUCT_SIZE;
    }
    reset_result(out_result);

    ContextLease lease(context);
    if (!lease.acquired()) {
        return SYNTH_ERR_INVALID_ARG;
    }

    synth::PreparedSynthesisRequest prepared;
    synth_status_t                  status = synth::prepare_synthesis_request(context->model->info, request, prepared);
    if (status != SYNTH_OK) {
        return status;
    }
    if (!valid_diagnostic_sink(prepared.diagnostics)) {
        return SYNTH_ERR_INVALID_ARG;
    }

    const uint64_t actual_seed = prepared.seed == SYNTH_SEED_RANDOM ? synth::nondeterministic_seed() : prepared.seed;
    const synth::AudioDeliveryInfo delivery_info = make_delivery_info(context, prepared, actual_seed);
    if (cancellation_requested(prepared)) {
        (void) synth::deliver_complete_audio(nullptr, 0, delivery_info, sink, out_result);
        return SYNTH_ERR_CANCELLED;
    }

    if (context->model->info.family == synth::ModelFamily::Omnivoice) {
        try {
            synth::omnivoice::PublicSynthesisParams family_request;
            // input_kind is guaranteed SYNTH_INPUT_TEXT_UTF8 by this point:
            // arch/omnivoice/weights.cpp's read_capabilities refuses to LOAD
            // any package whose input_flags is not EXACTLY
            // SYNTH_INPUT_SUPPORT_TEXT_UTF8 (reviewer FINDING 4 -- until that
            // fix, this loader only checked the TEXT bit was PRESENT, so a
            // package additionally declaring SYNTH_INPUT_SUPPORT_TOKEN_IDS
            // would have reached this dispatch with an input_kind of
            // SYNTH_INPUT_TOKEN_IDS and had its int32 token array read as
            // raw UTF-8 text bytes below), and prepare_synthesis_request
            // above already refused any request kind the package does not
            // declare -- so the raw bytes behind `prepared.token_ids` (this
            // family's frontend has no prefix/suffix, so that generic
            // tokenization is unused here; see model.cpp's registration
            // comment) are read straight from the request instead.
            family_request.text.assign(static_cast<const char *>(request->input_data),
                                       static_cast<size_t>(request->input_count));
            family_request.language_tag.assign(
                prepared.resolved_language_tag != nullptr ? prepared.resolved_language_tag : "",
                size_t(prepared.resolved_language_size));
            // `clone` (Task 14) and `instruct` (Task 15): a profile from a
            // different model, or one that is neither of this family's own
            // payload shapes, is refused here rather than silently ignored
            // -- prepare_synthesis_request threads any non-null profile
            // through without checking either property, because it has no
            // access to synth_voice_profile's full definition
            // (voice-profile-handle.h) to do so. The two payload shapes are
            // mutually exclusive by construction (one synth_voice_profile_t
            // carries exactly one family_tag), so at most one of `clone`/
            // `instruct` is ever set below.
            if (prepared.voice_profile != nullptr) {
                if (prepared.voice_profile->model != context->model) {
                    return SYNTH_ERR_UNSUPPORTED_VOICE;
                }
                if (prepared.voice_profile->family_tag == synth::ProfileFamilyTag::OmnivoiceClone) {
                    family_request.clone =
                        static_cast<const synth::omnivoice::ClonePrompt *>(prepared.voice_profile->payload.get());
                } else if (prepared.voice_profile->family_tag == synth::ProfileFamilyTag::OmnivoiceDesign) {
                    // Points at the DesignInstruct's own already-canonical
                    // `instruct` member rather than copying it: the profile
                    // (and so its payload) outlives this synchronous
                    // synthesis call, which is all PublicSynthesisParams'
                    // borrowed pointer needs.
                    family_request.instruct =
                        &static_cast<const synth::omnivoice::DesignInstruct *>(prepared.voice_profile->payload.get())
                             ->instruct;
                } else {
                    return SYNTH_ERR_UNSUPPORTED_VOICE;
                }
            }
            family_request.speaking_rate = double(prepared.speaking_rate);
            family_request.seed          = actual_seed;
            family_request.threads       = context->threads;

            // Same PCM-vs-native conversion qwen3-tts needs and for the same
            // reason: the core's limit counts output PCM frames, and this
            // family's own limit counts codec frames of `samples_per_frame`
            // each.
            const uint32_t samples_per_frame = context->model->info.samples_per_frame;
            if (prepared.requested_frame_limit != 0 && samples_per_frame != 0) {
                family_request.max_output_frames = prepared.requested_frame_limit / samples_per_frame;
                if (family_request.max_output_frames == 0) {
                    // A limit smaller than one frame cannot be met by
                    // emitting anything, and asking for zero frames would be
                    // read as unset.
                    return report_output_limit(delivery_info, sink, out_result);
                }
            } else {
                family_request.max_output_frames = 0;  // the family applies its own package cap
            }

            synth::omnivoice::SynthesisOutput synthesis;
            status = context->model->omnivoice->synthesize(family_request, synthesis);
            // This family estimates and clamps its own canvas length before
            // ever reaching run_synthesis, so -- like qwen3-tts -- its limit
            // stop arrives here rather than at the delivering check further
            // down.
            if (status == SYNTH_ERR_OUTPUT_LIMIT) {
                return report_output_limit(delivery_info, sink, out_result);
            }
            if (status != SYNTH_OK) {
                emit_diagnostic(prepared.diagnostics, status, "synthesis.graph_failed", synth_status_string(status));
                return status;
            }
            // `prepared.effective_frame_limit` is native PCM frames
            // (docs/c-interface.md), and `synthesis.frame_count` is this
            // family's own codec/decoder-frame count (samples_per_frame PCM
            // samples each) -- converted here rather than compared directly.
            // PR #6's finding: while the package's own max_output_frames
            // metadata was (wrongly) written in codec frames, this comparison
            // "worked" only because prepared.effective_frame_limit inherited
            // that same wrong codec-frame magnitude; now that the package
            // field is correctly PCM frames, a codec-frame count would never
            // exceed it and this check would silently stop enforcing the
            // limit at all.
            if (samples_per_frame == 0 ||
                synthesis.frame_count > std::numeric_limits<uint64_t>::max() / samples_per_frame) {
                (void) synth::deliver_complete_audio(nullptr, 0, delivery_info, sink, out_result);
                return SYNTH_ERR_OUTPUT_LIMIT;
            }
            const uint64_t synthesis_pcm_frame_count = synthesis.frame_count * samples_per_frame;
            if (synthesis_pcm_frame_count > prepared.effective_frame_limit) {
                (void) synth::deliver_complete_audio(nullptr, 0, delivery_info, sink, out_result);
                return SYNTH_ERR_OUTPUT_LIMIT;
            }
            if (cancellation_requested(prepared)) {
                (void) synth::deliver_complete_audio(nullptr, 0, delivery_info, sink, out_result);
                return SYNTH_ERR_CANCELLED;
            }
            return synth::deliver_complete_audio(synthesis.audio.data(), synthesis.audio.size(), delivery_info, sink,
                                                 out_result);
        } catch (const std::bad_alloc &) {
            emit_diagnostic(prepared.diagnostics, SYNTH_ERR_OOM, "allocation.failed",
                            "synthesis temporary allocation failed");
            return SYNTH_ERR_OOM;
        } catch (...) {
            emit_diagnostic(prepared.diagnostics, SYNTH_ERR_INTERNAL, "internal.exception",
                            "unexpected exception during synthesis");
            return SYNTH_ERR_INTERNAL;
        }
    }

    if (context->model->info.family == synth::ModelFamily::Qwen3Tts) {
        // This family has no Voice Profile support of its own: a profile
        // reaching here would either be a defect in the cross-model check
        // above (see the omnivoice branch) or a future family's own profile
        // presented to the wrong one. Either way, silently ignoring
        // `prepared.voice_profile` and synthesizing anyway would answer with
        // the wrong Voice rather than the refusal the caller asked for.
        if (prepared.voice_profile != nullptr) {
            return SYNTH_ERR_UNSUPPORTED_VOICE;
        }
        try {
            synth::qwen3tts::SynthesisRequest family_request;
            family_request.token_ids = prepared.token_ids;
            family_request.voice_id.assign(prepared.resolved_voice_id != nullptr ? prepared.resolved_voice_id : "",
                                           size_t(prepared.resolved_voice_size));
            family_request.language.assign(
                prepared.resolved_language_tag != nullptr ? prepared.resolved_language_tag : "",
                size_t(prepared.resolved_language_size));
            family_request.seed              = actual_seed;
            family_request.threads           = context->threads;
            // The core's limit is in native frames, which for this family is the
            // codec's hop, so it doubles as the cache's size.
            // The public field counts output PCM frames; this family's limit
            // counts codec frames of `samples_per_frame` each. Passing one as the
            // other let a request for 600 frames emit 21120 of them, against the
            // contract's guarantee that a nonzero value is never exceeded.
            //
            // Zero still means unset here, which is why the raw request value is
            // used rather than the normalised one: the normalised value is the
            // package's fifteen-million "no limit", and the family sizes its
            // talker cache from what it is given.
            const uint32_t samples_per_frame = context->model->info.samples_per_frame;
            if (prepared.requested_frame_limit != 0 && samples_per_frame != 0) {
                family_request.max_frames = prepared.requested_frame_limit / samples_per_frame;
                if (family_request.max_frames == 0) {
                    // A limit smaller than one frame cannot be met by emitting
                    // anything, and asking for zero frames would be read as unset.
                    return report_output_limit(delivery_info, sink, out_result);
                }
            } else {
                family_request.max_frames = 0;  // the family applies kDefaultMaxFrames
            }

            synth::qwen3tts::SynthesisOutput synthesis;
            status = context->model->qwen3_tts->run_synthesis(family_request, synthesis);
            // This family is the only one that carries the request's limit into
            // its own loop, so it is the only one whose limit stop arrives here
            // rather than at a delivering check further down. Taking the generic
            // branch left every resolved result field zeroed and called it a
            // graph failure.
            if (status == SYNTH_ERR_OUTPUT_LIMIT) {
                return report_output_limit(delivery_info, sink, out_result);
            }
            if (status != SYNTH_OK) {
                emit_diagnostic(prepared.diagnostics, status, "synthesis.graph_failed", synth_status_string(status));
                return status;
            }
            // The limit is in native frames, and this family's frame is the
            // codec's hop rather than one sample.
            if (synthesis.frame_count > prepared.effective_frame_limit) {
                (void) synth::deliver_complete_audio(nullptr, 0, delivery_info, sink, out_result);
                return SYNTH_ERR_OUTPUT_LIMIT;
            }
            if (cancellation_requested(prepared)) {
                (void) synth::deliver_complete_audio(nullptr, 0, delivery_info, sink, out_result);
                return SYNTH_ERR_CANCELLED;
            }
            return synth::deliver_complete_audio(synthesis.audio.data(), synthesis.audio.size(), delivery_info, sink,
                                                 out_result);
        } catch (const std::bad_alloc &) {
            emit_diagnostic(prepared.diagnostics, SYNTH_ERR_OOM, "allocation.failed",
                            "synthesis temporary allocation failed");
            return SYNTH_ERR_OOM;
        } catch (...) {
            emit_diagnostic(prepared.diagnostics, SYNTH_ERR_INTERNAL, "internal.exception",
                            "unexpected exception during synthesis");
            return SYNTH_ERR_INTERNAL;
        }
    }

    if (context->model->info.family == synth::ModelFamily::Kokoro) {
        // This family has no Voice Profile support of its own; see the
        // matching guard in the Qwen3-TTS branch above for why this refuses
        // rather than silently ignoring the profile.
        if (prepared.voice_profile != nullptr) {
            return SYNTH_ERR_UNSUPPORTED_VOICE;
        }
        try {
            // A Kokoro Voice is a table row chosen by input length, so the row
            // is resolved from the final token count rather than at load time.
            uint32_t voice_row = 0;
            if (!context->model->kokoro->resolve_voice_row(prepared.token_ids.size(), voice_row)) {
                emit_diagnostic(prepared.diagnostics, SYNTH_ERR_INPUT_TOO_LONG, "synthesis.voice_row",
                                "the input length has no style row in this voice table");
                return SYNTH_ERR_INPUT_TOO_LONG;
            }
            synth::kokoro::WaveformOutput waveform;
            status =
                context->model->kokoro->run_synthesis(prepared.token_ids, prepared.speaker_index, voice_row,
                                                      prepared.speaking_rate, actual_seed, context->threads, waveform);
            if (status != SYNTH_OK) {
                emit_diagnostic(prepared.diagnostics, status, "synthesis.graph_failed", synth_status_string(status));
                return status;
            }
            if (waveform.sample_count > prepared.effective_frame_limit) {
                (void) synth::deliver_complete_audio(nullptr, 0, delivery_info, sink, out_result);
                return SYNTH_ERR_OUTPUT_LIMIT;
            }
            if (cancellation_requested(prepared)) {
                (void) synth::deliver_complete_audio(nullptr, 0, delivery_info, sink, out_result);
                return SYNTH_ERR_CANCELLED;
            }
            return synth::deliver_complete_audio(waveform.pcm.data(), waveform.sample_count, delivery_info, sink,
                                                 out_result);
        } catch (const std::bad_alloc &) {
            emit_diagnostic(prepared.diagnostics, SYNTH_ERR_OOM, "allocation.failed",
                            "synthesis temporary allocation failed");
            return SYNTH_ERR_OOM;
        } catch (...) {
            emit_diagnostic(prepared.diagnostics, SYNTH_ERR_INTERNAL, "internal.exception",
                            "unexpected exception during synthesis");
            return SYNTH_ERR_INTERNAL;
        }
    }

    // The remaining branch is VITS, which -- like Kokoro and Qwen3-TTS above
    // -- has no Voice Profile support of its own.
    if (prepared.voice_profile != nullptr) {
        return SYNTH_ERR_UNSUPPORTED_VOICE;
    }

    try {
        synth::NormalRandomStream random(actual_seed);
        if (prepared.token_ids.size() > std::numeric_limits<size_t>::max() / 2) {
            return SYNTH_ERR_INPUT_TOO_LONG;
        }
        std::vector<float> duration_noise(prepared.token_ids.size() * 2);
        random.fill(duration_noise.data(), duration_noise.size());

        synth::vits::DurationOutput duration;
        status = context->model->vits->run_duration(
            prepared.token_ids, duration_noise, context->model->vits_extras.duration_noise_scale,
            prepared.speaking_rate, context->threads, duration, prepared.speaker_index);
        if (status != SYNTH_OK) {
            emit_diagnostic(prepared.diagnostics, status, "synthesis.duration_failed", synth_status_string(status));
            return status;
        }
        if (cancellation_requested(prepared)) {
            (void) synth::deliver_complete_audio(nullptr, 0, delivery_info, sink, out_result);
            return SYNTH_ERR_CANCELLED;
        }
        if (context->model->vits_extras.hop_length == 0 ||
            duration.frame_count > std::numeric_limits<uint64_t>::max() / context->model->vits_extras.hop_length) {
            return SYNTH_ERR_OUTPUT_LIMIT;
        }
        const uint64_t pcm_frame_count = duration.frame_count * context->model->vits_extras.hop_length;
        if (pcm_frame_count > prepared.effective_frame_limit) {
            (void) synth::deliver_complete_audio(nullptr, 0, delivery_info, sink, out_result);
            return SYNTH_ERR_OUTPUT_LIMIT;
        }
        if (context->model->vits_extras.inter_channels == 0 ||
            duration.frame_count > std::numeric_limits<size_t>::max() / context->model->vits_extras.inter_channels) {
            return SYNTH_ERR_OUTPUT_LIMIT;
        }
        std::vector<float> latent_noise(static_cast<size_t>(duration.frame_count) *
                                        context->model->vits_extras.inter_channels);
        random.fill(latent_noise.data(), latent_noise.size());

        synth::vits::WaveformDecoderOutput waveform;
        status = context->model->vits->run_waveform_decoder(
            prepared.token_ids, duration_noise, latent_noise, context->model->vits_extras.latent_noise_scale,
            context->model->vits_extras.duration_noise_scale, prepared.speaking_rate, context->threads, waveform,
            prepared.speaker_index);
        if (status != SYNTH_OK) {
            emit_diagnostic(prepared.diagnostics, status, "synthesis.graph_failed", synth_status_string(status));
            return status;
        }
        if (waveform.sample_count != waveform.pcm.size() || waveform.sample_count != pcm_frame_count) {
            emit_diagnostic(prepared.diagnostics, SYNTH_ERR_INTERNAL, "synthesis.output_shape",
                            "model output shape does not match the resolved duration");
            return SYNTH_ERR_INTERNAL;
        }
        if (cancellation_requested(prepared)) {
            (void) synth::deliver_complete_audio(nullptr, 0, delivery_info, sink, out_result);
            return SYNTH_ERR_CANCELLED;
        }
        return synth::deliver_complete_audio(waveform.pcm.data(), waveform.sample_count, delivery_info, sink,
                                             out_result);
    } catch (const std::bad_alloc &) {
        emit_diagnostic(prepared.diagnostics, SYNTH_ERR_OOM, "allocation.failed",
                        "synthesis temporary allocation failed");
        return SYNTH_ERR_OOM;
    } catch (...) {
        emit_diagnostic(prepared.diagnostics, SYNTH_ERR_INTERNAL, "internal.exception",
                        "unexpected exception during synthesis");
        return SYNTH_ERR_INTERNAL;
    }
}

synth_status_t synth_synthesize_to_buffer(synth_context_t *       context,
                                          const synth_request_t * request,
                                          synth_audio_buffer_t ** out_audio,
                                          synth_result_t *        out_result) {
    if (out_audio == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    *out_audio = nullptr;
    BufferCollector    collector;
    synth_audio_sink_t sink;
    synth_audio_sink_init(&sink, sizeof(sink));
    sink.write     = collect_audio;
    sink.user_data = &collector;

    const synth_status_t status = synth_synthesize(context, request, &sink, out_result);
    if (collector.allocation_failed) {
        reset_result(out_result);
        return SYNTH_ERR_OOM;
    }
    if (status != SYNTH_OK && status != SYNTH_ERR_CANCELLED && status != SYNTH_ERR_OUTPUT_LIMIT) {
        return status;
    }
    if (collector.samples.empty()) {
        return status;
    }
    try {
        auto audio         = std::make_unique<synth_audio_buffer_t>();
        auto owned_samples = std::make_unique<float[]>(collector.samples.size());
        std::copy(collector.samples.begin(), collector.samples.end(), owned_samples.get());
        audio->struct_size   = sizeof(*audio);
        audio->samples       = owned_samples.release();
        audio->frame_count   = collector.samples.size() / collector.channel_count;
        audio->sample_rate   = collector.sample_rate;
        audio->channel_count = collector.channel_count;
        *out_audio           = audio.release();
        return status;
    } catch (const std::bad_alloc &) {
        reset_result(out_result);
        return SYNTH_ERR_OOM;
    } catch (...) {
        reset_result(out_result);
        return SYNTH_ERR_INTERNAL;
    }
}

void synth_audio_buffer_free(synth_audio_buffer_t * audio) {
    if (audio == nullptr) {
        return;
    }
    delete[] const_cast<float *>(audio->samples);
    delete audio;
}
