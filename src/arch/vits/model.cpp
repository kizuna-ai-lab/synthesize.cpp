#include "acoustic-flow-host.h"
#include "acoustic-flow.h"
#include "backend-plan.h"
#include "duration-path.h"
#include "duration-predictor-host.h"
#include "duration-predictor.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "latent-sampling-host.h"
#include "latent-sampling.h"
#include "prior-expansion-host.h"
#include "prior-expansion.h"
#include "text-encoder-host.h"
#include "text-encoder.h"
#include "vits.h"
#include "waveform-decoder-host.h"
#include "waveform-decoder.h"
#include "weights.h"

#include <cstdio>
#include <fstream>
#include <limits>
#include <new>

namespace synth::vits {

namespace {

struct DurationStageOutput {
    TextEncoderOutput       text;
    DurationPredictorOutput predictor;
};

synth_status_t stream_tensor_data(const std::string & path, const gguf_context * gguf, ggml_context * weights_context) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return SYNTH_ERR_IO;
    }
    const size_t               data_offset = gguf_get_data_offset(gguf);
    std::vector<unsigned char> staging;
    for (ggml_tensor * tensor = ggml_get_first_tensor(weights_context); tensor != nullptr;
         tensor               = ggml_get_next_tensor(weights_context, tensor)) {
        const int64_t index = gguf_find_tensor(gguf, tensor->name);
        if (index < 0) {
            std::fprintf(stderr, "vits: tensor %s is absent from GGUF data\n", tensor->name);
            return SYNTH_ERR_GGUF;
        }
        const size_t         bytes = ggml_nbytes(tensor);
        const std::streamoff absolute =
            static_cast<std::streamoff>(data_offset) + static_cast<std::streamoff>(gguf_get_tensor_offset(gguf, index));
        input.seekg(absolute);
        if (!input) {
            return SYNTH_ERR_IO;
        }
        if (staging.size() < bytes) {
            staging.resize(bytes);
        }
        input.read(reinterpret_cast<char *>(staging.data()), static_cast<std::streamsize>(bytes));
        if (!input || static_cast<size_t>(input.gcount()) != bytes) {
            return SYNTH_ERR_IO;
        }
        ggml_backend_tensor_set(tensor, staging.data(), 0, bytes);
    }
    return SYNTH_OK;
}

}  // namespace

struct Model::Impl {
    gguf_context *                      gguf            = nullptr;
    ggml_context *                      weights_context = nullptr;
    std::unique_ptr<BackendPlan>        backend_plan;
    ggml_backend_buffer_t               weights_buffer = nullptr;
    HParams                             hparams;
    std::shared_ptr<const TextFrontend> text_frontend;
    VoiceWeights                        voice_weights;
    TextWeights                         text_weights;
    DurationWeights                     duration_weights;
    FlowWeights                         flow_weights;
    DecoderWeights                      decoder_weights;

    synth_status_t compute_duration_stage(const std::vector<int32_t> & token_ids,
                                          const std::vector<float> &   duration_noise,
                                          float                        noise_scale_w,
                                          int                          threads,
                                          uint32_t                     speaker_index,
                                          DurationStageOutput &        output) const;

    ~Impl() {
        if (weights_buffer != nullptr) {
            ggml_backend_buffer_free(weights_buffer);
        }
        if (weights_context != nullptr) {
            ggml_free(weights_context);
        }
        if (gguf != nullptr) {
            gguf_free(gguf);
        }
    }
};

Model::Model(std::unique_ptr<Impl> implementation) : implementation_(std::move(implementation)) {}

Model::~Model() = default;

synth_status_t Model::get_info(ModelInfo & output) const {
    output = {};
    if (implementation_ == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    const HParams & hparams     = implementation_->hparams;
    output.has_package_default  = hparams.has_package_default;
    output.preset_voice_ids     = hparams.preset_voice_ids;
    output.preset_voice_flags   = hparams.preset_voice_flags;
    output.text_frontend        = implementation_->text_frontend;
    output.input_flags          = hparams.input_flags;
    output.capability_flags     = hparams.capability_flags;
    output.output_sample_rate   = hparams.output_sample_rate;
    output.output_channel_count = hparams.output_channel_count;
    output.inter_channels       = hparams.inter_channels;
    output.vocab_size           = hparams.vocab_size;
    output.hop_length           = hparams.hop_length;
    output.max_input_tokens     = hparams.max_input_tokens;
    output.max_output_frames    = hparams.max_output_frames;
    output.min_speaking_rate    = hparams.min_speaking_rate;
    output.max_speaking_rate    = hparams.max_speaking_rate;
    output.latent_noise_scale   = hparams.latent_noise_scale;
    output.duration_noise_scale = hparams.duration_noise_scale_w;
    return SYNTH_OK;
}

synth_status_t Model::load_cpu(const std::string & path, std::unique_ptr<Model> & output) {
    return load_cpu(path, ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), output);
}

ggml_backend_dev_t Model::primary_device() const {
    if (implementation_ == nullptr || implementation_->backend_plan == nullptr) {
        return nullptr;
    }
    return implementation_->backend_plan->primary_device();
}

synth_status_t Model::load_cpu(const std::string & path, ggml_backend_dev_t device, std::unique_ptr<Model> & output) {
    if (device == nullptr || ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        output.reset();
        return SYNTH_ERR_BACKEND;
    }
    return load(path, device, false, output);
}

synth_status_t Model::load(const std::string &      path,
                           ggml_backend_dev_t       primary_device,
                           bool                     include_accelerators,
                           std::unique_ptr<Model> & output) {
    output.reset();
    if (path.empty()) {
        return SYNTH_ERR_INVALID_ARG;
    }
    std::ifstream probe(path, std::ios::binary);
    if (!probe) {
        return SYNTH_ERR_FILE_NOT_FOUND;
    }
    probe.close();

    try {
        auto           implementation = std::make_unique<Impl>();
        synth_status_t status = BackendPlan::create(primary_device, include_accelerators, implementation->backend_plan);
        if (status != SYNTH_OK) {
            return status;
        }

        gguf_init_params parameters{};
        parameters.no_alloc  = true;
        parameters.ctx       = &implementation->weights_context;
        implementation->gguf = gguf_init_from_file(path.c_str(), parameters);
        if (implementation->gguf == nullptr || implementation->weights_context == nullptr) {
            return SYNTH_ERR_GGUF;
        }
        status = read_hparams(implementation->gguf, implementation->hparams);
        if (status != SYNTH_OK) {
            return status;
        }
        if (implementation->hparams.frontend_present) {
            std::unique_ptr<TextFrontend> frontend;
            status = make_symbol_map_frontend(implementation->hparams.frontend_config, frontend);
            if (status != SYNTH_OK) {
                return status;
            }
            implementation->text_frontend = std::shared_ptr<const TextFrontend>(std::move(frontend));
        }
        const int64_t expected_tensor_count = implementation->hparams.speaker_count == 0 ? 460 : 473;
        if (gguf_get_n_tensors(implementation->gguf) != expected_tensor_count) {
            std::fprintf(stderr, "vits: %s package must contain exactly %lld tensors\n",
                         implementation->hparams.model_variant.c_str(), static_cast<long long>(expected_tensor_count));
            return SYNTH_ERR_GGUF;
        }
        status = build_voice_weights(implementation->weights_context, implementation->hparams,
                                     implementation->voice_weights);
        if (status != SYNTH_OK) {
            return status;
        }
        status =
            build_text_weights(implementation->weights_context, implementation->hparams, implementation->text_weights);
        if (status != SYNTH_OK) {
            return status;
        }
        status = build_duration_weights(implementation->weights_context, implementation->hparams,
                                        implementation->duration_weights);
        if (status != SYNTH_OK) {
            return status;
        }
        status =
            build_flow_weights(implementation->weights_context, implementation->hparams, implementation->flow_weights);
        if (status != SYNTH_OK) {
            return status;
        }
        status = build_decoder_weights(implementation->weights_context, implementation->hparams,
                                       implementation->decoder_weights);
        if (status != SYNTH_OK) {
            return status;
        }

        implementation->weights_buffer =
            ggml_backend_alloc_ctx_tensors(implementation->weights_context, implementation->backend_plan->primary());
        if (implementation->weights_buffer == nullptr) {
            return SYNTH_ERR_OOM;
        }
        ggml_backend_buffer_set_usage(implementation->weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        status = stream_tensor_data(path, implementation->gguf, implementation->weights_context);
        if (status != SYNTH_OK) {
            return status;
        }

        output = std::unique_ptr<Model>(new Model(std::move(implementation)));
        return SYNTH_OK;
    } catch (const std::bad_alloc &) {
        return SYNTH_ERR_OOM;
    } catch (...) {
        return SYNTH_ERR_INTERNAL;
    }
}

synth_status_t Model::run_text_encoder(const std::vector<int32_t> & token_ids,
                                       int                          threads,
                                       TextEncoderOutput &          output) const {
    output = {};
    if (implementation_ == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    const HParams &          hparams = implementation_->hparams;
    PreparedTextEncoderInput prepared;
    synth_status_t           status = prepare_text_encoder_input(hparams, token_ids, threads, prepared);
    if (status != SYNTH_OK) {
        return status;
    }

    ggml_context *       compute_context = nullptr;
    ggml_backend_sched_t scheduler       = nullptr;
    try {
        ggml_init_params parameters{};
        parameters.mem_size   = 32 * 1024 * 1024;
        parameters.mem_buffer = nullptr;
        parameters.no_alloc   = true;
        compute_context       = ggml_init(parameters);
        if (compute_context == nullptr) {
            return SYNTH_ERR_OOM;
        }

        const int64_t    token_count = prepared.token_count;
        TextEncoderGraph graph =
            build_text_encoder_graph(compute_context, implementation_->text_weights, hparams, token_count);
        if (graph.graph == nullptr || graph.token_ids == nullptr || graph.relative_indices == nullptr ||
            graph.encoded == nullptr || graph.m_p == nullptr || graph.logs_p == nullptr) {
            ggml_free(compute_context);
            return SYNTH_ERR_INTERNAL;
        }

        scheduler = implementation_->backend_plan->create_scheduler(8192);
        if (scheduler == nullptr) {
            ggml_free(compute_context);
            return SYNTH_ERR_BACKEND;
        }
        if (!ggml_backend_sched_alloc_graph(scheduler, graph.graph)) {
            ggml_backend_sched_free(scheduler);
            ggml_free(compute_context);
            return SYNTH_ERR_OOM;
        }
        implementation_->backend_plan->log_placement_if_enabled("text_encoder", scheduler, graph.graph);

        ggml_backend_tensor_set(graph.token_ids, token_ids.data(), 0, token_ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(graph.relative_indices, prepared.relative_indices.data(), 0,
                                prepared.relative_indices.size() * sizeof(int32_t));
        implementation_->backend_plan->set_threads(threads);
        if (ggml_backend_sched_graph_compute(scheduler, graph.graph) != GGML_STATUS_SUCCESS) {
            ggml_backend_sched_free(scheduler);
            ggml_free(compute_context);
            return SYNTH_ERR_BACKEND;
        }

        const size_t       elements = static_cast<size_t>(hparams.inter_channels) * token_ids.size();
        std::vector<float> m_p_ggml(elements);
        std::vector<float> logs_p_ggml(elements);
        ggml_backend_tensor_get(graph.m_p, m_p_ggml.data(), 0, elements * sizeof(float));
        ggml_backend_tensor_get(graph.logs_p, logs_p_ggml.data(), 0, elements * sizeof(float));
        status = finalize_text_encoder_output(hparams, token_ids.size(), m_p_ggml, logs_p_ggml, output);

        ggml_backend_sched_free(scheduler);
        ggml_free(compute_context);
        return status;
    } catch (const std::bad_alloc &) {
        if (scheduler != nullptr) {
            ggml_backend_sched_free(scheduler);
        }
        if (compute_context != nullptr) {
            ggml_free(compute_context);
        }
        return SYNTH_ERR_OOM;
    } catch (...) {
        if (scheduler != nullptr) {
            ggml_backend_sched_free(scheduler);
        }
        if (compute_context != nullptr) {
            ggml_free(compute_context);
        }
        return SYNTH_ERR_INTERNAL;
    }
}

synth_status_t Model::Impl::compute_duration_stage(const std::vector<int32_t> & token_ids,
                                                   const std::vector<float> &   duration_noise,
                                                   float                        noise_scale_w,
                                                   int                          threads,
                                                   uint32_t                     speaker_index,
                                                   DurationStageOutput &        output) const {
    output                                 = {};
    const HParams &          model_hparams = hparams;
    PreparedTextEncoderInput text_input;
    synth_status_t           status = prepare_text_encoder_input(model_hparams, token_ids, threads, text_input);
    if (status != SYNTH_OK) {
        return status;
    }
    PreparedDurationInput duration_input;
    status = prepare_duration_input(token_ids.size(), duration_noise, noise_scale_w, duration_input);
    if (status != SYNTH_OK) {
        return status;
    }

    ggml_context *       compute_context = nullptr;
    ggml_backend_sched_t scheduler       = nullptr;
    try {
        ggml_init_params context_parameters{};
        context_parameters.mem_size   = 64 * 1024 * 1024;
        context_parameters.mem_buffer = nullptr;
        context_parameters.no_alloc   = true;
        compute_context               = ggml_init(context_parameters);
        if (compute_context == nullptr) {
            return SYNTH_ERR_OOM;
        }

        const int64_t    token_count = text_input.token_count;
        TextEncoderGraph text_graph =
            build_text_encoder_graph(compute_context, text_weights, model_hparams, token_count);
        if (text_graph.token_ids == nullptr || text_graph.relative_indices == nullptr ||
            text_graph.encoded == nullptr) {
            ggml_free(compute_context);
            return SYNTH_ERR_INTERNAL;
        }
        DurationPredictorGraph duration_graph =
            build_duration_predictor_graph(compute_context, text_graph.encoded, duration_weights, voice_weights,
                                           model_hparams, speaker_index, token_count, noise_scale_w);
        if (duration_graph.graph == nullptr || duration_graph.duration_noise == nullptr ||
            duration_graph.logw == nullptr) {
            ggml_free(compute_context);
            return SYNTH_ERR_INTERNAL;
        }
        ggml_build_forward_expand(duration_graph.graph, text_graph.m_p);
        ggml_build_forward_expand(duration_graph.graph, text_graph.logs_p);

        scheduler = backend_plan->create_scheduler(16384);
        if (scheduler == nullptr) {
            ggml_free(compute_context);
            return SYNTH_ERR_BACKEND;
        }
        if (!ggml_backend_sched_alloc_graph(scheduler, duration_graph.graph)) {
            ggml_backend_sched_free(scheduler);
            ggml_free(compute_context);
            return SYNTH_ERR_OOM;
        }
        backend_plan->log_placement_if_enabled("duration_stage", scheduler, duration_graph.graph);

        ggml_backend_tensor_set(text_graph.token_ids, token_ids.data(), 0, token_ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(text_graph.relative_indices, text_input.relative_indices.data(), 0,
                                text_input.relative_indices.size() * sizeof(int32_t));
        ggml_backend_tensor_set(duration_graph.duration_noise, duration_input.noise_channel_fastest.data(), 0,
                                duration_input.noise_channel_fastest.size() * sizeof(float));
        if (duration_graph.speaker_index != nullptr) {
            const int32_t selected_speaker = static_cast<int32_t>(speaker_index);
            ggml_backend_tensor_set(duration_graph.speaker_index, &selected_speaker, 0, sizeof(selected_speaker));
        }
        backend_plan->set_threads(threads);
        if (ggml_backend_sched_graph_compute(scheduler, duration_graph.graph) != GGML_STATUS_SUCCESS) {
            ggml_backend_sched_free(scheduler);
            ggml_free(compute_context);
            return SYNTH_ERR_BACKEND;
        }

        const size_t       elements = static_cast<size_t>(model_hparams.inter_channels) * token_ids.size();
        std::vector<float> m_p(elements);
        std::vector<float> logs_p(elements);
        std::vector<float> logw(token_ids.size());
        ggml_backend_tensor_get(text_graph.m_p, m_p.data(), 0, m_p.size() * sizeof(float));
        ggml_backend_tensor_get(text_graph.logs_p, logs_p.data(), 0, logs_p.size() * sizeof(float));
        ggml_backend_tensor_get(duration_graph.logw, logw.data(), 0, logw.size() * sizeof(float));
        status = finalize_text_encoder_output(model_hparams, token_ids.size(), m_p, logs_p, output.text);
        if (status == SYNTH_OK) {
            status = finalize_duration_output(token_ids.size(), logw, output.predictor);
        }

        ggml_backend_sched_free(scheduler);
        ggml_free(compute_context);
        return status;
    } catch (const std::bad_alloc &) {
        if (scheduler != nullptr) {
            ggml_backend_sched_free(scheduler);
        }
        if (compute_context != nullptr) {
            ggml_free(compute_context);
        }
        return SYNTH_ERR_OOM;
    } catch (...) {
        if (scheduler != nullptr) {
            ggml_backend_sched_free(scheduler);
        }
        if (compute_context != nullptr) {
            ggml_free(compute_context);
        }
        return SYNTH_ERR_INTERNAL;
    }
}

synth_status_t Model::run_duration_predictor(const std::vector<int32_t> & token_ids,
                                             const std::vector<float> &   duration_noise,
                                             float                        noise_scale_w,
                                             int                          threads,
                                             DurationPredictorOutput &    output,
                                             uint32_t                     speaker_index) const {
    output = {};
    if (implementation_ == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    DurationStageOutput  stage;
    const synth_status_t status = implementation_->compute_duration_stage(token_ids, duration_noise, noise_scale_w,
                                                                          threads, speaker_index, stage);
    if (status == SYNTH_OK) {
        output = std::move(stage.predictor);
    }
    return status;
}

synth_status_t Model::run_duration(const std::vector<int32_t> & token_ids,
                                   const std::vector<float> &   duration_noise,
                                   float                        noise_scale_w,
                                   float                        speaking_rate,
                                   int                          threads,
                                   DurationOutput &             output,
                                   uint32_t                     speaker_index) const {
    output = {};
    DurationPredictorOutput predictor;
    synth_status_t          status =
        run_duration_predictor(token_ids, duration_noise, noise_scale_w, threads, predictor, speaker_index);
    if (status != SYNTH_OK) {
        return status;
    }
    try {
        ResolvedDurationPath path;
        status = resolve_duration_path(implementation_->hparams, predictor.logw, speaking_rate, path);
        if (status != SYNTH_OK) {
            return status;
        }
        output.token_count = predictor.token_count;
        output.frame_count = path.frame_count;
        output.logw        = std::move(predictor.logw);
        output.w_ceil      = std::move(path.w_ceil);
        output.attention   = std::move(path.attention);
        return SYNTH_OK;
    } catch (const std::bad_alloc &) {
        output = {};
        return SYNTH_ERR_OOM;
    } catch (...) {
        output = {};
        return SYNTH_ERR_INTERNAL;
    }
}

synth_status_t Model::run_prior_expansion(const std::vector<int32_t> & token_ids,
                                          const std::vector<float> &   duration_noise,
                                          float                        noise_scale_w,
                                          float                        speaking_rate,
                                          int                          threads,
                                          PriorExpansionOutput &       output,
                                          uint32_t                     speaker_index) const {
    output = {};
    if (implementation_ == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }

    DurationStageOutput stage;
    synth_status_t status = implementation_->compute_duration_stage(token_ids, duration_noise, noise_scale_w, threads,
                                                                    speaker_index, stage);
    if (status != SYNTH_OK) {
        return status;
    }

    ggml_context *       compute_context = nullptr;
    ggml_backend_sched_t scheduler       = nullptr;
    try {
        ResolvedDurationPath path;
        status = resolve_duration_path(implementation_->hparams, stage.predictor.logw, speaking_rate, path);
        if (status != SYNTH_OK) {
            return status;
        }
        DurationOutput duration;
        duration.token_count = stage.predictor.token_count;
        duration.frame_count = path.frame_count;
        duration.logw        = std::move(stage.predictor.logw);
        duration.w_ceil      = std::move(path.w_ceil);
        duration.attention   = std::move(path.attention);

        PreparedPriorExpansionInput prepared;
        status = prepare_prior_expansion_input(stage.text, duration, prepared);
        if (status != SYNTH_OK) {
            return status;
        }
        if (prepared.channels > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            prepared.token_count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            prepared.frame_count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            return SYNTH_ERR_OUTPUT_LIMIT;
        }

        ggml_init_params parameters{};
        parameters.mem_size   = 4 * 1024 * 1024;
        parameters.mem_buffer = nullptr;
        parameters.no_alloc   = true;
        compute_context       = ggml_init(parameters);
        if (compute_context == nullptr) {
            return SYNTH_ERR_OOM;
        }
        PriorExpansionGraph graph =
            build_prior_expansion_graph(compute_context, prepared.channels, static_cast<int64_t>(prepared.token_count),
                                        static_cast<int64_t>(prepared.frame_count));
        if (graph.graph == nullptr || graph.m_p == nullptr || graph.logs_p == nullptr || graph.attention == nullptr ||
            graph.m_p_expanded == nullptr || graph.logs_p_expanded == nullptr) {
            ggml_free(compute_context);
            return SYNTH_ERR_INTERNAL;
        }

        scheduler = implementation_->backend_plan->create_scheduler(512);
        if (scheduler == nullptr) {
            ggml_free(compute_context);
            return SYNTH_ERR_BACKEND;
        }
        if (!implementation_->backend_plan->assign_to_primary(scheduler, graph.m_p_expanded) ||
            !implementation_->backend_plan->assign_to_primary(scheduler, graph.logs_p_expanded)) {
            ggml_backend_sched_free(scheduler);
            ggml_free(compute_context);
            return SYNTH_ERR_BACKEND;
        }
        if (!ggml_backend_sched_alloc_graph(scheduler, graph.graph)) {
            ggml_backend_sched_free(scheduler);
            ggml_free(compute_context);
            return SYNTH_ERR_OOM;
        }
        implementation_->backend_plan->log_placement_if_enabled("prior_expansion", scheduler, graph.graph);

        ggml_backend_tensor_set(graph.m_p, prepared.m_p_channel_fastest.data(), 0,
                                prepared.m_p_channel_fastest.size() * sizeof(float));
        ggml_backend_tensor_set(graph.logs_p, prepared.logs_p_channel_fastest.data(), 0,
                                prepared.logs_p_channel_fastest.size() * sizeof(float));
        ggml_backend_tensor_set(graph.attention, duration.attention.data(), 0,
                                duration.attention.size() * sizeof(float));
        implementation_->backend_plan->set_threads(threads);
        if (ggml_backend_sched_graph_compute(scheduler, graph.graph) != GGML_STATUS_SUCCESS) {
            ggml_backend_sched_free(scheduler);
            ggml_free(compute_context);
            return SYNTH_ERR_BACKEND;
        }

        const size_t elements = static_cast<size_t>(prepared.channels) * static_cast<size_t>(prepared.frame_count);
        std::vector<float> m_p(elements);
        std::vector<float> logs_p(elements);
        ggml_backend_tensor_get(graph.m_p_expanded, m_p.data(), 0, m_p.size() * sizeof(float));
        ggml_backend_tensor_get(graph.logs_p_expanded, logs_p.data(), 0, logs_p.size() * sizeof(float));
        status = finalize_prior_expansion_output(prepared.channels, prepared.frame_count, m_p, logs_p, output);

        ggml_backend_sched_free(scheduler);
        ggml_free(compute_context);
        return status;
    } catch (const std::bad_alloc &) {
        if (scheduler != nullptr) {
            ggml_backend_sched_free(scheduler);
        }
        if (compute_context != nullptr) {
            ggml_free(compute_context);
        }
        return SYNTH_ERR_OOM;
    } catch (...) {
        if (scheduler != nullptr) {
            ggml_backend_sched_free(scheduler);
        }
        if (compute_context != nullptr) {
            ggml_free(compute_context);
        }
        return SYNTH_ERR_INTERNAL;
    }
}

synth_status_t Model::run_latent_sampling(const std::vector<int32_t> & token_ids,
                                          const std::vector<float> &   duration_noise,
                                          const std::vector<float> &   latent_noise,
                                          float                        noise_scale,
                                          float                        noise_scale_w,
                                          float                        speaking_rate,
                                          int                          threads,
                                          LatentSamplingOutput &       output,
                                          uint32_t                     speaker_index) const {
    output = {};
    PriorExpansionOutput prior;
    synth_status_t       status =
        run_prior_expansion(token_ids, duration_noise, noise_scale_w, speaking_rate, threads, prior, speaker_index);
    if (status != SYNTH_OK) {
        return status;
    }

    ggml_context *       compute_context = nullptr;
    ggml_backend_sched_t scheduler       = nullptr;
    try {
        PreparedLatentSamplingInput prepared;
        status = prepare_latent_sampling_input(prior, latent_noise, noise_scale, prepared);
        if (status != SYNTH_OK) {
            return status;
        }
        if (prepared.channels > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            prepared.frame_count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            return SYNTH_ERR_OUTPUT_LIMIT;
        }

        ggml_init_params parameters{};
        parameters.mem_size   = 2 * 1024 * 1024;
        parameters.mem_buffer = nullptr;
        parameters.no_alloc   = true;
        compute_context       = ggml_init(parameters);
        if (compute_context == nullptr) {
            return SYNTH_ERR_OOM;
        }
        LatentSamplingGraph graph = build_latent_sampling_graph(
            compute_context, prepared.channels, static_cast<int64_t>(prepared.frame_count), prepared.noise_scale);
        if (graph.graph == nullptr || graph.m_p == nullptr || graph.logs_p == nullptr ||
            graph.latent_noise == nullptr || graph.z_p == nullptr) {
            ggml_free(compute_context);
            return SYNTH_ERR_INTERNAL;
        }

        scheduler = implementation_->backend_plan->create_scheduler(64);
        if (scheduler == nullptr) {
            ggml_free(compute_context);
            return SYNTH_ERR_BACKEND;
        }
        if (!implementation_->backend_plan->assign_to_primary(scheduler, graph.z_p)) {
            ggml_backend_sched_free(scheduler);
            ggml_free(compute_context);
            return SYNTH_ERR_BACKEND;
        }
        if (!ggml_backend_sched_alloc_graph(scheduler, graph.graph)) {
            ggml_backend_sched_free(scheduler);
            ggml_free(compute_context);
            return SYNTH_ERR_OOM;
        }
        implementation_->backend_plan->log_placement_if_enabled("latent_sampling", scheduler, graph.graph);

        ggml_backend_tensor_set(graph.m_p, prepared.m_p_channel_fastest.data(), 0,
                                prepared.m_p_channel_fastest.size() * sizeof(float));
        ggml_backend_tensor_set(graph.logs_p, prepared.logs_p_channel_fastest.data(), 0,
                                prepared.logs_p_channel_fastest.size() * sizeof(float));
        ggml_backend_tensor_set(graph.latent_noise, prepared.noise_channel_fastest.data(), 0,
                                prepared.noise_channel_fastest.size() * sizeof(float));
        implementation_->backend_plan->set_threads(threads);
        if (ggml_backend_sched_graph_compute(scheduler, graph.graph) != GGML_STATUS_SUCCESS) {
            ggml_backend_sched_free(scheduler);
            ggml_free(compute_context);
            return SYNTH_ERR_BACKEND;
        }

        const size_t elements = static_cast<size_t>(prepared.channels) * static_cast<size_t>(prepared.frame_count);
        std::vector<float> z_p(elements);
        ggml_backend_tensor_get(graph.z_p, z_p.data(), 0, z_p.size() * sizeof(float));
        status = finalize_latent_sampling_output(prepared.channels, prepared.frame_count, z_p, output);

        ggml_backend_sched_free(scheduler);
        ggml_free(compute_context);
        return status;
    } catch (const std::bad_alloc &) {
        if (scheduler != nullptr) {
            ggml_backend_sched_free(scheduler);
        }
        if (compute_context != nullptr) {
            ggml_free(compute_context);
        }
        return SYNTH_ERR_OOM;
    } catch (...) {
        if (scheduler != nullptr) {
            ggml_backend_sched_free(scheduler);
        }
        if (compute_context != nullptr) {
            ggml_free(compute_context);
        }
        return SYNTH_ERR_INTERNAL;
    }
}

synth_status_t Model::run_acoustic_flow(const std::vector<int32_t> & token_ids,
                                        const std::vector<float> &   duration_noise,
                                        const std::vector<float> &   latent_noise,
                                        float                        noise_scale,
                                        float                        noise_scale_w,
                                        float                        speaking_rate,
                                        int                          threads,
                                        AcousticFlowOutput &         output,
                                        uint32_t                     speaker_index) const {
    output = {};
    LatentSamplingOutput latent;
    synth_status_t status = run_latent_sampling(token_ids, duration_noise, latent_noise, noise_scale, noise_scale_w,
                                                speaking_rate, threads, latent, speaker_index);
    if (status != SYNTH_OK) {
        return status;
    }

    ggml_context *       compute_context = nullptr;
    ggml_backend_sched_t scheduler       = nullptr;
    try {
        PreparedAcousticFlowInput prepared;
        status = prepare_acoustic_flow_input(latent, prepared);
        if (status != SYNTH_OK) {
            return status;
        }
        if (prepared.channels != implementation_->hparams.inter_channels ||
            prepared.frame_count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            return SYNTH_ERR_OUTPUT_LIMIT;
        }

        ggml_init_params parameters{};
        parameters.mem_size   = 64 * 1024 * 1024;
        parameters.mem_buffer = nullptr;
        parameters.no_alloc   = true;
        compute_context       = ggml_init(parameters);
        if (compute_context == nullptr) {
            return SYNTH_ERR_OOM;
        }
        AcousticFlowGraph graph = build_acoustic_flow_graph(compute_context, implementation_->flow_weights,
                                                            implementation_->voice_weights, implementation_->hparams,
                                                            speaker_index, static_cast<int64_t>(prepared.frame_count));
        if (graph.graph == nullptr || graph.z_p == nullptr || graph.channel_indices == nullptr || graph.z == nullptr) {
            ggml_free(compute_context);
            return SYNTH_ERR_INTERNAL;
        }

        scheduler = implementation_->backend_plan->create_scheduler(8192);
        if (scheduler == nullptr) {
            ggml_free(compute_context);
            return SYNTH_ERR_BACKEND;
        }
        if (!ggml_backend_sched_alloc_graph(scheduler, graph.graph)) {
            ggml_backend_sched_free(scheduler);
            ggml_free(compute_context);
            return SYNTH_ERR_OOM;
        }
        implementation_->backend_plan->log_placement_if_enabled("acoustic_flow", scheduler, graph.graph);

        ggml_backend_tensor_set(graph.z_p, prepared.z_p_channel_fastest.data(), 0,
                                prepared.z_p_channel_fastest.size() * sizeof(float));
        std::vector<int32_t> channel_indices(prepared.channels);
        for (uint32_t channel = 0; channel < prepared.channels; ++channel) {
            channel_indices[channel] = static_cast<int32_t>(prepared.channels - 1 - channel);
        }
        ggml_backend_tensor_set(graph.channel_indices, channel_indices.data(), 0,
                                channel_indices.size() * sizeof(int32_t));
        if (graph.speaker_index != nullptr) {
            const int32_t selected_speaker = static_cast<int32_t>(speaker_index);
            ggml_backend_tensor_set(graph.speaker_index, &selected_speaker, 0, sizeof(selected_speaker));
        }
        implementation_->backend_plan->set_threads(threads);
        if (ggml_backend_sched_graph_compute(scheduler, graph.graph) != GGML_STATUS_SUCCESS) {
            ggml_backend_sched_free(scheduler);
            ggml_free(compute_context);
            return SYNTH_ERR_BACKEND;
        }

        const size_t elements = static_cast<size_t>(prepared.channels) * static_cast<size_t>(prepared.frame_count);
        std::vector<float> z(elements);
        ggml_backend_tensor_get(graph.z, z.data(), 0, z.size() * sizeof(float));
        status = finalize_acoustic_flow_output(prepared.channels, prepared.frame_count, z, output);

        ggml_backend_sched_free(scheduler);
        ggml_free(compute_context);
        return status;
    } catch (const std::bad_alloc &) {
        if (scheduler != nullptr) {
            ggml_backend_sched_free(scheduler);
        }
        if (compute_context != nullptr) {
            ggml_free(compute_context);
        }
        return SYNTH_ERR_OOM;
    } catch (...) {
        if (scheduler != nullptr) {
            ggml_backend_sched_free(scheduler);
        }
        if (compute_context != nullptr) {
            ggml_free(compute_context);
        }
        return SYNTH_ERR_INTERNAL;
    }
}

synth_status_t Model::run_waveform_decoder(const std::vector<int32_t> & token_ids,
                                           const std::vector<float> &   duration_noise,
                                           const std::vector<float> &   latent_noise,
                                           float                        noise_scale,
                                           float                        noise_scale_w,
                                           float                        speaking_rate,
                                           int                          threads,
                                           WaveformDecoderOutput &      output,
                                           uint32_t                     speaker_index) const {
    output = {};
    AcousticFlowOutput flow;
    synth_status_t     status = run_acoustic_flow(token_ids, duration_noise, latent_noise, noise_scale, noise_scale_w,
                                                  speaking_rate, threads, flow, speaker_index);
    if (status != SYNTH_OK) {
        return status;
    }

    ggml_context *       compute_context = nullptr;
    ggml_backend_sched_t scheduler       = nullptr;
    try {
        PreparedWaveformDecoderInput prepared;
        status = prepare_waveform_decoder_input(flow, prepared);
        if (status != SYNTH_OK) {
            return status;
        }
        const HParams & hparams = implementation_->hparams;
        if (prepared.channels != hparams.inter_channels ||
            prepared.frame_count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            prepared.frame_count > std::numeric_limits<size_t>::max() / hparams.hop_length) {
            return SYNTH_ERR_OUTPUT_LIMIT;
        }
        const size_t sample_count = static_cast<size_t>(prepared.frame_count) * hparams.hop_length;
        if (sample_count > hparams.max_output_frames) {
            return SYNTH_ERR_OUTPUT_LIMIT;
        }

        ggml_init_params parameters{};
        parameters.mem_size   = 128 * 1024 * 1024;
        parameters.mem_buffer = nullptr;
        parameters.no_alloc   = true;
        compute_context       = ggml_init(parameters);
        if (compute_context == nullptr) {
            return SYNTH_ERR_OOM;
        }
        WaveformDecoderGraph graph = build_waveform_decoder_graph(
            compute_context, implementation_->decoder_weights, implementation_->voice_weights, hparams, speaker_index,
            static_cast<int64_t>(prepared.frame_count));
        if (graph.graph == nullptr || graph.z == nullptr || graph.pcm == nullptr) {
            ggml_free(compute_context);
            return SYNTH_ERR_INTERNAL;
        }

        scheduler = implementation_->backend_plan->create_scheduler(32768);
        if (scheduler == nullptr) {
            ggml_free(compute_context);
            return SYNTH_ERR_BACKEND;
        }
        if (!ggml_backend_sched_alloc_graph(scheduler, graph.graph)) {
            ggml_backend_sched_free(scheduler);
            ggml_free(compute_context);
            return SYNTH_ERR_OOM;
        }
        implementation_->backend_plan->log_placement_if_enabled("waveform_decoder", scheduler, graph.graph);

        ggml_backend_tensor_set(graph.z, prepared.z_channel_fastest.data(), 0,
                                prepared.z_channel_fastest.size() * sizeof(float));
        if (graph.speaker_index != nullptr) {
            const int32_t selected_speaker = static_cast<int32_t>(speaker_index);
            ggml_backend_tensor_set(graph.speaker_index, &selected_speaker, 0, sizeof(selected_speaker));
        }
        implementation_->backend_plan->set_threads(threads);
        if (ggml_backend_sched_graph_compute(scheduler, graph.graph) != GGML_STATUS_SUCCESS) {
            ggml_backend_sched_free(scheduler);
            ggml_free(compute_context);
            return SYNTH_ERR_BACKEND;
        }

        std::vector<float> pcm(sample_count);
        ggml_backend_tensor_get(graph.pcm, pcm.data(), 0, pcm.size() * sizeof(float));
        status = finalize_waveform_decoder_output(sample_count, pcm, output);

        ggml_backend_sched_free(scheduler);
        ggml_free(compute_context);
        return status;
    } catch (const std::bad_alloc &) {
        if (scheduler != nullptr) {
            ggml_backend_sched_free(scheduler);
        }
        if (compute_context != nullptr) {
            ggml_free(compute_context);
        }
        return SYNTH_ERR_OOM;
    } catch (...) {
        if (scheduler != nullptr) {
            ggml_backend_sched_free(scheduler);
        }
        if (compute_context != nullptr) {
            ggml_free(compute_context);
        }
        return SYNTH_ERR_INTERNAL;
    }
}

}  // namespace synth::vits
