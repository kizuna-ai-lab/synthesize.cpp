#include "backend-plan.h"
#include "decoder-host.h"
#include "decoder.h"
#include "duration-host.h"
#include "duration.h"
#include "generator.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-metadata.h"
#include "gguf.h"
#include "kokoro.h"
#include "plbert.h"
#include "prosody.h"
#include "random-stream.h"
#include "source.h"
#include "text-encoder.h"
#include "weights.h"

#include <cstdio>
#include <fstream>
#include <limits>
#include <new>

namespace synth::kokoro {

namespace {

// A context plus the backend buffer that backs it, freed together.
//
// The LSTM stores live here rather than in the graph arena: the recurrent chain
// writes each step into a view of a persistent tensor, which the scheduler must
// not recycle. See docs/porting/families/kokoro.md.
class Arena {
  public:
    ~Arena() { reset(); }

    Arena()                          = default;
    Arena(const Arena &)             = delete;
    Arena & operator=(const Arena &) = delete;

    bool open(size_t tensors) {
        reset();
        ggml_init_params parameters{};
        parameters.mem_size = ggml_tensor_overhead() * tensors;
        parameters.no_alloc = true;
        context_            = ggml_init(parameters);
        return context_ != nullptr;
    }

    ggml_context * context() const { return context_; }

    // Allocates every declared tensor and clears it, so the recurrence starts
    // from a known state rather than whatever the buffer happened to hold.
    bool commit(ggml_backend_t backend) {
        buffer_ = ggml_backend_alloc_ctx_tensors(context_, backend);
        if (buffer_ == nullptr) {
            return false;
        }
        for (ggml_tensor * tensor = ggml_get_first_tensor(context_); tensor != nullptr;
             tensor               = ggml_get_next_tensor(context_, tensor)) {
            const std::vector<unsigned char> zeros(ggml_nbytes(tensor), 0);
            ggml_backend_tensor_set(tensor, zeros.data(), 0, zeros.size());
        }
        return true;
    }

    void reset() {
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
            buffer_ = nullptr;
        }
        if (context_ != nullptr) {
            ggml_free(context_);
            context_ = nullptr;
        }
    }

  private:
    ggml_context *        context_ = nullptr;
    ggml_backend_buffer_t buffer_  = nullptr;
};

constexpr size_t kGraphArenaBytes = 64ull * 1024 * 1024;

// Headroom in the scheduler's hash set for the weight tensors a graph reads,
// which enter as leaves rather than nodes. The decoder touches the most, at a
// few hundred.
constexpr size_t kSchedulerLeafAllowance = 2048;

// One graph run: a header-only context and the scheduler that places it.
//
// Tensors already sitting in a backend buffer, such as the LSTM stores, are
// left alone by the scheduler's allocator, which is what keeps the recurrent
// chain off the reused compute arena.
class GraphRun {
  public:
    GraphRun(const BackendPlan & plan, size_t bytes) : plan_(plan) {
        ggml_init_params parameters{};
        parameters.mem_size = bytes;
        parameters.no_alloc = true;
        context_            = ggml_init(parameters);
    }

    ~GraphRun() {
        if (scheduler_ != nullptr) {
            ggml_backend_sched_free(scheduler_);
        }
        if (context_ != nullptr) {
            ggml_free(context_);
        }
    }

    GraphRun(const GraphRun &)             = delete;
    GraphRun & operator=(const GraphRun &) = delete;

    ggml_context * context() const { return context_; }

    // Places and allocates the graph. Inputs must be written afterwards, since
    // that is when they acquire memory.
    synth_status_t allocate(ggml_cgraph * graph, const char * stage) {
        if (context_ == nullptr || graph == nullptr) {
            return SYNTH_ERR_OOM;
        }
        // The scheduler's hash set has to cover the leaves as well as the
        // nodes, and the leaves are every weight the graph touches, so sizing it
        // from the node capacity alone is not enough.
        scheduler_ = plan_.create_scheduler(size_t(ggml_graph_size(graph)) + kSchedulerLeafAllowance);
        if (scheduler_ == nullptr) {
            return SYNTH_ERR_BACKEND;
        }
        if (!ggml_backend_sched_alloc_graph(scheduler_, graph)) {
            return SYNTH_ERR_OOM;
        }
        plan_.log_placement_if_enabled(stage, scheduler_, graph);
        return SYNTH_OK;
    }

    synth_status_t dispatch(ggml_cgraph * graph, int threads) {
        plan_.set_threads(threads);
        return ggml_backend_sched_graph_compute(scheduler_, graph) == GGML_STATUS_SUCCESS ? SYNTH_OK :
                                                                                            SYNTH_ERR_BACKEND;
    }

  private:
    const BackendPlan &  plan_;
    ggml_context *       context_   = nullptr;
    ggml_backend_sched_t scheduler_ = nullptr;
};

void read_tensor(const ggml_tensor * tensor, std::vector<float> & output) {
    output.resize(size_t(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, output.data(), 0, ggml_nbytes(tensor));
}

// Tensors leave the graph as [features, time] with features contiguous; the
// staged outputs are [features, time] with time contiguous, matching the
// reference probes.
void transpose_to_feature_major(const std::vector<float> & input,
                                uint64_t                   features,
                                uint64_t                   length,
                                std::vector<float> &       output) {
    output.assign(size_t(features) * size_t(length), 0.0f);
    for (uint64_t time = 0; time < length; ++time) {
        for (uint64_t feature = 0; feature < features; ++feature) {
            output[size_t(feature) * size_t(length) + size_t(time)] =
                input[size_t(time) * size_t(features) + size_t(feature)];
        }
    }
}

// Expands a state by the one-hot alignment, which is the matrix product
// upstream writes as `d @ pred_aln_trg`. Both the source and the result are in
// graph layout, where the feature index runs fastest.
void expand_by_alignment(const std::vector<float> & source,
                         const std::vector<float> & alignment,
                         uint64_t                   features,
                         uint64_t                   token_count,
                         uint64_t                   frame_count,
                         std::vector<float> &       output) {
    output.assign(size_t(features) * size_t(frame_count), 0.0f);
    for (uint64_t token = 0; token < token_count; ++token) {
        for (uint64_t frame = 0; frame < frame_count; ++frame) {
            const float weight = alignment[size_t(token) * size_t(frame_count) + size_t(frame)];
            if (weight == 0.0f) {
                continue;
            }
            for (uint64_t feature = 0; feature < features; ++feature) {
                output[size_t(frame) * size_t(features) + size_t(feature)] +=
                    weight * source[size_t(token) * size_t(features) + size_t(feature)];
            }
        }
    }
}

}  // namespace

struct Model::Impl {
    gguf_context *                      gguf            = nullptr;
    ggml_context *                      weights_context = nullptr;
    std::unique_ptr<BackendPlan>        backend_plan;
    ggml_backend_buffer_t               weights_buffer = nullptr;
    HParams                             hparams;
    std::shared_ptr<const TextFrontend> text_frontend;
    ModelWeights                        weights;

    // The style halves of one Voice row. A Kokoro Voice is a table indexed by
    // input length, so the row is a synthesis-time choice, not a load-time one.
    synth_status_t read_style(uint32_t             voice_index,
                              uint32_t             row,
                              std::vector<float> & decoder_style,
                              std::vector<float> & prosody_style) const;

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

synth_status_t Model::Impl::read_style(uint32_t             voice_index,
                                       uint32_t             row,
                                       std::vector<float> & decoder_style,
                                       std::vector<float> & prosody_style) const {
    if (voice_index >= weights.voices.packs.size()) {
        return SYNTH_ERR_INVALID_ARG;
    }
    ggml_tensor * pack = weights.voices.packs[voice_index];
    if (pack == nullptr || row >= uint32_t(pack->ne[1])) {
        return SYNTH_ERR_INVALID_ARG;
    }
    std::vector<float> full(size_t(hparams.voice_dim));
    ggml_backend_tensor_get(pack, full.data(), size_t(row) * size_t(hparams.voice_dim) * sizeof(float),
                            full.size() * sizeof(float));
    decoder_style.assign(full.begin() + hparams.voice_decoder_offset,
                         full.begin() + hparams.voice_decoder_offset + hparams.style_dim);
    prosody_style.assign(full.begin() + hparams.voice_prosody_offset,
                         full.begin() + hparams.voice_prosody_offset + hparams.style_dim);
    return SYNTH_OK;
}

Model::Model(std::unique_ptr<Impl> implementation) : implementation_(std::move(implementation)) {}

Model::~Model() = default;

synth_status_t Model::get_info(ModelInfo & output) const {
    if (implementation_ == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    const HParams & hparams     = implementation_->hparams;
    output                      = ModelInfo{};
    output.has_package_default  = hparams.has_package_default;
    output.preset_voice_ids     = hparams.preset_voice_ids;
    output.preset_voice_flags   = hparams.preset_voice_flags;
    output.text_frontend        = implementation_->text_frontend;
    output.input_flags          = hparams.input_flags;
    output.capability_flags     = hparams.capability_flags;
    output.output_sample_rate   = hparams.output_sample_rate;
    output.output_channel_count = hparams.output_channel_count;
    output.vocab_size           = hparams.n_token;
    output.samples_per_frame    = hparams.samples_per_frame;
    output.max_input_tokens     = hparams.max_input_tokens;
    output.max_output_frames    = hparams.max_output_frames;
    output.min_speaking_rate    = hparams.min_speaking_rate;
    output.max_speaking_rate    = hparams.max_speaking_rate;
    return SYNTH_OK;
}

ggml_backend_device * Model::primary_device() const {
    if (implementation_ == nullptr || implementation_->backend_plan == nullptr) {
        return nullptr;
    }
    return implementation_->backend_plan->primary_device();
}

bool Model::resolve_voice_row(uint64_t final_token_count, uint32_t & row) const {
    if (implementation_ == nullptr) {
        return false;
    }
    return resolve_style_row(implementation_->hparams, final_token_count, row);
}

uint32_t Model::source_harmonics() const {
    return implementation_ == nullptr ? 0 : source_harmonic_count(implementation_->hparams);
}

uint64_t Model::source_sample_count(uint64_t frame_count) const {
    if (implementation_ == nullptr) {
        return 0;
    }
    // The prosody curves run at twice the frame rate, and the source module
    // upsamples them again by its own scale.
    return source_upsampled_length(implementation_->hparams, frame_count * 2);
}

synth_status_t Model::load_cpu(const std::string & path, std::unique_ptr<Model> & output) {
    return load_cpu(path, ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), output);
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
        status = build_model_weights(implementation->weights_context, implementation->hparams, implementation->weights);
        if (status != SYNTH_OK) {
            return status;
        }

        implementation->weights_buffer =
            ggml_backend_alloc_ctx_tensors(implementation->weights_context, implementation->backend_plan->primary());
        if (implementation->weights_buffer == nullptr) {
            return SYNTH_ERR_OOM;
        }
        ggml_backend_buffer_set_usage(implementation->weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        status = stream_tensor_data(path, implementation->gguf, implementation->weights_context, "kokoro");
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

namespace {

synth_status_t check_tokens(const HParams & hparams, const std::vector<int32_t> & token_ids) {
    if (token_ids.empty()) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (hparams.max_input_tokens != 0 && token_ids.size() > hparams.max_input_tokens) {
        return SYNTH_ERR_INPUT_TOO_LONG;
    }
    for (int32_t id : token_ids) {
        if (id < 0 || uint32_t(id) >= hparams.n_token) {
            return SYNTH_ERR_INVALID_ARG;
        }
    }
    return SYNTH_OK;
}

}  // namespace

// Everything the staged entry points share. Each stage recomputes its
// predecessors, which is what lets a validator drive one stage at a time
// against the oracle without the family holding synthesis state.
struct Model::Pipeline {
    uint32_t           token_count = 0;
    uint64_t           frame_count = 0;
    std::vector<float> decoder_style;
    std::vector<float> prosody_style;
    DurationResult     duration;
    // Graph layout, feature index fastest.
    std::vector<float> plbert_hidden;
    std::vector<float> plbert_projected;
    std::vector<float> duration_logits;
    std::vector<float> duration_encoded;
    std::vector<float> f0;
    std::vector<float> energy;
    std::vector<float> expanded;
    std::vector<float> text_encoded;
    std::vector<float> aligned;
    SourceResult       source;
};

synth_status_t Model::run_plbert(const std::vector<int32_t> & token_ids, int threads, PLBertOutput & output) const {
    output = PLBertOutput{};
    Pipeline       state;
    synth_status_t status = compute_plbert(token_ids, threads, state);
    if (status != SYNTH_OK) {
        return status;
    }
    const HParams & hparams = implementation_->hparams;
    output.hidden_size      = hparams.plbert.hidden_size;
    output.token_count      = state.token_count;
    // PL-BERT's own probe keeps the encoder's [tokens, features] order, while
    // the projection is transposed the moment it leaves the stage.
    output.hidden           = state.plbert_hidden;
    transpose_to_feature_major(state.plbert_projected, hparams.hidden_dim, state.token_count, output.projected);
    return SYNTH_OK;
}

synth_status_t Model::compute_plbert(const std::vector<int32_t> & token_ids, int threads, Pipeline & state) const {
    if (implementation_ == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    const HParams & hparams = implementation_->hparams;
    synth_status_t  status  = check_tokens(hparams, token_ids);
    if (status != SYNTH_OK) {
        return status;
    }
    state.token_count = uint32_t(token_ids.size());

    try {
        GraphRun run(*implementation_->backend_plan, kGraphArenaBytes);
        if (run.context() == nullptr) {
            return SYNTH_ERR_OOM;
        }
        PLBertGraph built =
            build_plbert_graph(run.context(), implementation_->weights.bert, hparams, state.token_count);
        if (built.graph == nullptr) {
            return SYNTH_ERR_INTERNAL;
        }
        // The projection into the duration path's width belongs to this stage,
        // not to the encoder, which is why it is expanded onto the same graph.
        ggml_tensor * projected = linear(run.context(), built.hidden, implementation_->weights.bert_encoder);
        ggml_set_output(projected);
        ggml_build_forward_expand(built.graph, projected);

        status = run.allocate(built.graph, "plbert");
        if (status != SYNTH_OK) {
            return status;
        }
        ggml_backend_tensor_set(built.token_ids, token_ids.data(), 0, ggml_nbytes(built.token_ids));
        status = run.dispatch(built.graph, threads);
        if (status != SYNTH_OK) {
            return status;
        }
        read_tensor(built.hidden, state.plbert_hidden);
        read_tensor(projected, state.plbert_projected);
        return SYNTH_OK;
    } catch (const std::bad_alloc &) {
        return SYNTH_ERR_OOM;
    }
}

synth_status_t Model::compute_duration(const std::vector<int32_t> & token_ids,
                                       uint32_t                     voice_index,
                                       uint32_t                     voice_row,
                                       float                        speaking_rate,
                                       int                          threads,
                                       Pipeline &                   state) const {
    synth_status_t status = compute_plbert(token_ids, threads, state);
    if (status != SYNTH_OK) {
        return status;
    }
    const HParams & hparams = implementation_->hparams;
    if (speaking_rate < hparams.min_speaking_rate || speaking_rate > hparams.max_speaking_rate) {
        return SYNTH_ERR_INVALID_ARG;
    }
    status = implementation_->read_style(voice_index, voice_row, state.decoder_style, state.prosody_style);
    if (status != SYNTH_OK) {
        return status;
    }

    try {
        const uint32_t half   = hparams.hidden_dim / 2;
        const uint32_t layers = uint32_t(implementation_->weights.predictor.text_encoder.lstms.size());

        Arena arena;
        if (!arena.open(size_t(layers) * 2 + 8)) {
            return SYNTH_ERR_OOM;
        }
        DurationScratch scratch;
        scratch.zero_state = ggml_new_tensor_1d(arena.context(), GGML_TYPE_F32, half);
        scratch.encoder.resize(layers);
        for (uint32_t layer = 0; layer < layers; ++layer) {
            scratch.encoder[layer].zero_state = scratch.zero_state;
            scratch.encoder[layer].forward_store =
                ggml_new_tensor_2d(arena.context(), GGML_TYPE_F32, half, state.token_count);
            scratch.encoder[layer].reverse_store =
                ggml_new_tensor_2d(arena.context(), GGML_TYPE_F32, half, state.token_count);
        }
        scratch.predictor.zero_state    = scratch.zero_state;
        scratch.predictor.forward_store = ggml_new_tensor_2d(arena.context(), GGML_TYPE_F32, half, state.token_count);
        scratch.predictor.reverse_store = ggml_new_tensor_2d(arena.context(), GGML_TYPE_F32, half, state.token_count);
        if (!arena.commit(implementation_->backend_plan->primary())) {
            return SYNTH_ERR_OOM;
        }

        GraphRun run(*implementation_->backend_plan, kGraphArenaBytes);
        if (run.context() == nullptr) {
            return SYNTH_ERR_OOM;
        }
        DurationGraph built = build_duration_graph(run.context(), implementation_->weights.predictor, hparams, scratch,
                                                   state.token_count);
        if (built.graph == nullptr) {
            return SYNTH_ERR_INTERNAL;
        }
        ggml_build_forward_expand(built.graph, built.hidden);
        status = run.allocate(built.graph, "duration");
        if (status != SYNTH_OK) {
            return status;
        }
        ggml_backend_tensor_set(built.input, state.plbert_projected.data(), 0, ggml_nbytes(built.input));
        ggml_backend_tensor_set(built.style, state.prosody_style.data(), 0, ggml_nbytes(built.style));
        status = run.dispatch(built.graph, threads);
        if (status != SYNTH_OK) {
            return status;
        }
        read_tensor(built.logits, state.duration_logits);
        read_tensor(built.hidden, state.duration_encoded);

        // The host seam reads the graph's own layout, one token's bins at a
        // time, so the logits are passed through untransposed.
        status = resolve_durations(state.duration_logits, hparams, state.token_count, speaking_rate,
                                   hparams.max_output_frames, state.duration);
        if (status != SYNTH_OK) {
            return status;
        }
        state.frame_count = state.duration.y_length;
        return SYNTH_OK;
    } catch (const std::bad_alloc &) {
        return SYNTH_ERR_OOM;
    }
}

synth_status_t Model::run_duration(const std::vector<int32_t> & token_ids,
                                   uint32_t                     voice_index,
                                   uint32_t                     voice_row,
                                   float                        speaking_rate,
                                   int                          threads,
                                   DurationOutput &             output) const {
    output = DurationOutput{};
    Pipeline       state;
    synth_status_t status = compute_duration(token_ids, voice_index, voice_row, speaking_rate, threads, state);
    if (status != SYNTH_OK) {
        return status;
    }
    const HParams & hparams = implementation_->hparams;
    output.token_count      = state.token_count;
    output.frame_count      = state.frame_count;
    output.durations        = state.duration.pred_dur;
    output.alignment        = state.duration.alignment;
    // Both of these keep the graph's order, which is also the probes'.
    output.logits           = state.duration_logits;
    output.encoded          = state.duration_encoded;
    (void) hparams;
    return SYNTH_OK;
}

synth_status_t Model::compute_prosody(const std::vector<int32_t> & token_ids,
                                      uint32_t                     voice_index,
                                      uint32_t                     voice_row,
                                      float                        speaking_rate,
                                      int                          threads,
                                      Pipeline &                   state) const {
    synth_status_t status = compute_duration(token_ids, voice_index, voice_row, speaking_rate, threads, state);
    if (status != SYNTH_OK) {
        return status;
    }
    const HParams & hparams  = implementation_->hparams;
    const uint32_t  features = hparams.hidden_dim + hparams.style_dim;
    const uint32_t  frames   = uint32_t(state.frame_count);

    try {
        expand_by_alignment(state.duration_encoded, state.duration.alignment, features, state.token_count, frames,
                            state.expanded);

        const uint32_t half = hparams.hidden_dim / 2;
        Arena          arena;
        if (!arena.open(8)) {
            return SYNTH_ERR_OOM;
        }
        ProsodyScratch scratch;
        scratch.shared.zero_state    = ggml_new_tensor_1d(arena.context(), GGML_TYPE_F32, half);
        scratch.shared.forward_store = ggml_new_tensor_2d(arena.context(), GGML_TYPE_F32, half, frames);
        scratch.shared.reverse_store = ggml_new_tensor_2d(arena.context(), GGML_TYPE_F32, half, frames);
        if (!arena.commit(implementation_->backend_plan->primary())) {
            return SYNTH_ERR_OOM;
        }

        GraphRun run(*implementation_->backend_plan, kGraphArenaBytes);
        if (run.context() == nullptr) {
            return SYNTH_ERR_OOM;
        }
        ProsodyGraph built =
            build_prosody_graph(run.context(), implementation_->weights.predictor, hparams, scratch, frames);
        if (built.graph == nullptr) {
            return SYNTH_ERR_INTERNAL;
        }
        status = run.allocate(built.graph, "prosody");
        if (status != SYNTH_OK) {
            return status;
        }
        ggml_backend_tensor_set(built.expanded, state.expanded.data(), 0, ggml_nbytes(built.expanded));
        ggml_backend_tensor_set(built.style, state.prosody_style.data(), 0, ggml_nbytes(built.style));
        status = run.dispatch(built.graph, threads);
        if (status != SYNTH_OK) {
            return status;
        }
        read_tensor(built.f0, state.f0);
        read_tensor(built.energy, state.energy);
        return SYNTH_OK;
    } catch (const std::bad_alloc &) {
        return SYNTH_ERR_OOM;
    }
}

synth_status_t Model::run_prosody(const std::vector<int32_t> & token_ids,
                                  uint32_t                     voice_index,
                                  uint32_t                     voice_row,
                                  float                        speaking_rate,
                                  int                          threads,
                                  ProsodyOutput &              output) const {
    output = ProsodyOutput{};
    Pipeline       state;
    synth_status_t status = compute_prosody(token_ids, voice_index, voice_row, speaking_rate, threads, state);
    if (status != SYNTH_OK) {
        return status;
    }
    const HParams & hparams = implementation_->hparams;
    output.frame_count      = state.frame_count;
    transpose_to_feature_major(state.expanded, hparams.hidden_dim + hparams.style_dim, state.frame_count,
                               output.expanded);
    output.f0     = state.f0;
    output.energy = state.energy;
    return SYNTH_OK;
}

synth_status_t Model::compute_text_encoder(const std::vector<int32_t> & token_ids,
                                           uint32_t                     voice_index,
                                           uint32_t                     voice_row,
                                           float                        speaking_rate,
                                           int                          threads,
                                           Pipeline &                   state) const {
    synth_status_t status = compute_prosody(token_ids, voice_index, voice_row, speaking_rate, threads, state);
    if (status != SYNTH_OK) {
        return status;
    }
    const HParams & hparams = implementation_->hparams;

    try {
        const uint32_t half = hparams.hidden_dim / 2;
        Arena          arena;
        if (!arena.open(8)) {
            return SYNTH_ERR_OOM;
        }
        TextEncoderScratch scratch;
        scratch.lstm.zero_state    = ggml_new_tensor_1d(arena.context(), GGML_TYPE_F32, half);
        scratch.lstm.forward_store = ggml_new_tensor_2d(arena.context(), GGML_TYPE_F32, half, state.token_count);
        scratch.lstm.reverse_store = ggml_new_tensor_2d(arena.context(), GGML_TYPE_F32, half, state.token_count);
        if (!arena.commit(implementation_->backend_plan->primary())) {
            return SYNTH_ERR_OOM;
        }

        GraphRun run(*implementation_->backend_plan, kGraphArenaBytes);
        if (run.context() == nullptr) {
            return SYNTH_ERR_OOM;
        }
        TextEncoderGraph built = build_text_encoder_graph(run.context(), implementation_->weights.text_encoder, hparams,
                                                          scratch, state.token_count);
        if (built.graph == nullptr) {
            return SYNTH_ERR_INTERNAL;
        }
        status = run.allocate(built.graph, "text_encoder");
        if (status != SYNTH_OK) {
            return status;
        }
        ggml_backend_tensor_set(built.token_ids, token_ids.data(), 0, ggml_nbytes(built.token_ids));
        status = run.dispatch(built.graph, threads);
        if (status != SYNTH_OK) {
            return status;
        }
        read_tensor(built.encoded, state.text_encoded);
        expand_by_alignment(state.text_encoded, state.duration.alignment, hparams.hidden_dim, state.token_count,
                            state.frame_count, state.aligned);
        return SYNTH_OK;
    } catch (const std::bad_alloc &) {
        return SYNTH_ERR_OOM;
    }
}

synth_status_t Model::run_text_encoder(const std::vector<int32_t> & token_ids,
                                       uint32_t                     voice_index,
                                       uint32_t                     voice_row,
                                       float                        speaking_rate,
                                       int                          threads,
                                       TextEncoderOutput &          output) const {
    output = TextEncoderOutput{};
    Pipeline       state;
    synth_status_t status = compute_text_encoder(token_ids, voice_index, voice_row, speaking_rate, threads, state);
    if (status != SYNTH_OK) {
        return status;
    }
    const HParams & hparams = implementation_->hparams;
    output.channels         = hparams.hidden_dim;
    output.token_count      = state.token_count;
    output.frame_count      = state.frame_count;
    transpose_to_feature_major(state.text_encoded, hparams.hidden_dim, state.token_count, output.encoded);
    transpose_to_feature_major(state.aligned, hparams.hidden_dim, state.frame_count, output.aligned);
    return SYNTH_OK;
}

synth_status_t Model::compute_source(const std::vector<int32_t> & token_ids,
                                     uint32_t                     voice_index,
                                     uint32_t                     voice_row,
                                     float                        speaking_rate,
                                     const SourceRandom &         random,
                                     int                          threads,
                                     Pipeline &                   state) const {
    synth_status_t status = compute_text_encoder(token_ids, voice_index, voice_row, speaking_rate, threads, state);
    if (status != SYNTH_OK) {
        return status;
    }
    const HParams & hparams = implementation_->hparams;

    // The merge projection is one row; reading it here keeps the seam free of
    // GGML types.
    ggml_tensor * weight = implementation_->weights.decoder.generator.source_linear.weight;
    ggml_tensor * bias   = implementation_->weights.decoder.generator.source_linear.bias;
    if (weight == nullptr || bias == nullptr) {
        return SYNTH_ERR_INTERNAL;
    }
    std::vector<float> merge_weight(size_t(ggml_nelements(weight)));
    ggml_backend_tensor_get(weight, merge_weight.data(), 0, ggml_nbytes(weight));
    float merge_bias = 0.0f;
    ggml_backend_tensor_get(bias, &merge_bias, 0, sizeof(float));

    SourceRandomInputs inputs;
    inputs.rand_ini = random.rand_ini;
    inputs.noise    = random.noise;
    return build_harmonic_source(hparams, state.f0, merge_weight, merge_bias, inputs, state.source);
}

synth_status_t Model::run_source(const std::vector<int32_t> & token_ids,
                                 uint32_t                     voice_index,
                                 uint32_t                     voice_row,
                                 float                        speaking_rate,
                                 const SourceRandom &         random,
                                 int                          threads,
                                 SourceOutput &               output) const {
    output = SourceOutput{};
    Pipeline       state;
    synth_status_t status = compute_source(token_ids, voice_index, voice_row, speaking_rate, random, threads, state);
    if (status != SYNTH_OK) {
        return status;
    }
    output.frames     = state.source.frames;
    output.bins       = state.source.bins;
    output.excitation = state.source.har_source;
    // The seam builds the spectrum in the decoder's order; the probe keeps the
    // frame index contiguous.
    transpose_to_feature_major(state.source.har, 2 * state.source.bins, state.source.frames, output.spectrum);
    return SYNTH_OK;
}

synth_status_t Model::compute_decoder(const std::vector<int32_t> & token_ids,
                                      uint32_t                     voice_index,
                                      uint32_t                     voice_row,
                                      float                        speaking_rate,
                                      const SourceRandom &         random,
                                      int                          threads,
                                      Pipeline &                   state,
                                      std::vector<float> &         spectrum) const {
    synth_status_t status = compute_source(token_ids, voice_index, voice_row, speaking_rate, random, threads, state);
    if (status != SYNTH_OK) {
        return status;
    }
    const HParams & hparams = implementation_->hparams;

    try {
        GraphRun run(*implementation_->backend_plan, kGraphArenaBytes);
        if (run.context() == nullptr) {
            return SYNTH_ERR_OOM;
        }
        DecoderGraph built =
            build_decoder_graph(run.context(), implementation_->weights.decoder, hparams, uint32_t(state.frame_count));
        if (built.graph == nullptr) {
            return SYNTH_ERR_INTERNAL;
        }
        if (uint64_t(built.har->ne[1]) != state.source.frames) {
            std::fprintf(stderr, "kokoro: the harmonic source and the decoder disagree on the frame count\n");
            return SYNTH_ERR_INTERNAL;
        }
        status = run.allocate(built.graph, "decoder");
        if (status != SYNTH_OK) {
            return status;
        }
        ggml_backend_tensor_set(built.asr, state.aligned.data(), 0, ggml_nbytes(built.asr));
        ggml_backend_tensor_set(built.f0, state.f0.data(), 0, ggml_nbytes(built.f0));
        ggml_backend_tensor_set(built.energy, state.energy.data(), 0, ggml_nbytes(built.energy));
        ggml_backend_tensor_set(built.har, state.source.har.data(), 0, ggml_nbytes(built.har));
        ggml_backend_tensor_set(built.style, state.decoder_style.data(), 0, ggml_nbytes(built.style));
        status = run.dispatch(built.graph, threads);
        if (status != SYNTH_OK) {
            return status;
        }
        read_tensor(built.spectrum, spectrum);
        return SYNTH_OK;
    } catch (const std::bad_alloc &) {
        return SYNTH_ERR_OOM;
    }
}

synth_status_t Model::run_decoder(const std::vector<int32_t> & token_ids,
                                  uint32_t                     voice_index,
                                  uint32_t                     voice_row,
                                  float                        speaking_rate,
                                  const SourceRandom &         random,
                                  int                          threads,
                                  DecoderOutput &              output) const {
    output = DecoderOutput{};
    Pipeline           state;
    std::vector<float> spectrum;
    synth_status_t     status =
        compute_decoder(token_ids, voice_index, voice_row, speaking_rate, random, threads, state, spectrum);
    if (status != SYNTH_OK) {
        return status;
    }
    output.frames   = state.source.frames;
    output.spectrum = spectrum;
    return SYNTH_OK;
}

synth_status_t Model::run_synthesis(const std::vector<int32_t> & token_ids,
                                    uint32_t                     voice_index,
                                    uint32_t                     voice_row,
                                    float                        speaking_rate,
                                    uint64_t                     seed,
                                    int                          threads,
                                    WaveformOutput &             output) const {
    output = WaveformOutput{};

    // The frame count decides how much noise the source consumes, so the
    // duration stage runs first and the rest of the pipeline continues from the
    // state it leaves behind.
    Pipeline       state;
    synth_status_t status = compute_prosody(token_ids, voice_index, voice_row, speaking_rate, threads, state);
    if (status != SYNTH_OK) {
        return status;
    }

    const uint32_t harmonics = source_harmonics();
    const uint64_t samples   = source_sample_count(state.frame_count);
    if (harmonics == 0 || samples == 0) {
        return SYNTH_ERR_INTERNAL;
    }
    if (samples > std::numeric_limits<size_t>::max() / harmonics) {
        return SYNTH_ERR_OUTPUT_LIMIT;
    }

    try {
        SourceRandom       random;
        NormalRandomStream stream(seed);
        random.rand_ini.resize(harmonics);
        stream.fill_uniform(random.rand_ini.data(), random.rand_ini.size());
        // Upstream forces the fundamental's initial phase to zero; only the
        // overtones carry one.
        random.rand_ini[0] = 0.0f;
        random.noise.resize(size_t(samples) * harmonics);
        stream.fill(random.noise.data(), random.noise.size());
        return run_waveform(token_ids, voice_index, voice_row, speaking_rate, random, threads, output);
    } catch (const std::bad_alloc &) {
        return SYNTH_ERR_OOM;
    }
}

synth_status_t Model::run_waveform(const std::vector<int32_t> & token_ids,
                                   uint32_t                     voice_index,
                                   uint32_t                     voice_row,
                                   float                        speaking_rate,
                                   const SourceRandom &         random,
                                   int                          threads,
                                   WaveformOutput &             output) const {
    output = WaveformOutput{};
    Pipeline           state;
    std::vector<float> spectrum;
    synth_status_t     status =
        compute_decoder(token_ids, voice_index, voice_row, speaking_rate, random, threads, state, spectrum);
    if (status != SYNTH_OK) {
        return status;
    }
    status = inverse_stft(implementation_->hparams, spectrum, state.source.frames, output.pcm);
    if (status != SYNTH_OK) {
        return status;
    }
    output.sample_count = output.pcm.size();
    return SYNTH_OK;
}

}  // namespace synth::kokoro
