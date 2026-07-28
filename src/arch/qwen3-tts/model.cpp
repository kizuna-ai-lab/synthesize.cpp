// The Qwen3-TTS model object and its decode loop.
//
// Placement: every graph runs on the CPU scheduler. This family's sampled code
// is a discrete output, and docs/backends.md holds a discrete output and every
// stage feeding it on CPU -- which here is the talker and the code predictor,
// measured at 73.5 % of synthesis. The codec is the only stage that could sit on
// an accelerator, and it is 4.9 %. Deciding whether that is worth a second
// buffer is stage 7's job, with the measurement in hand rather than assumed.
//
// The loop itself is per frame: the talker emits one semantic code, the code
// predictor expands it into the frame's acoustic codes, their embeddings are
// summed into the talker's next input, and the codec turns the finished code
// stream into audio in one pass at the end.

#include "backend-plan.h"
#include "bpe.h"
#include "catalog.h"
#include "code-predictor-host.h"
#include "code-predictor.h"
#include "codec.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-metadata.h"
#include "gguf.h"
#include "qwen3-tts.h"
#include "random-stream.h"
#include "talker-host.h"
#include "talker.h"
#include "weights.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <fstream>
#include <new>

namespace synth::qwen3tts {

namespace {

// Headroom in the scheduler's hash set for the weights a graph reads, which
// enter as leaves rather than nodes.
constexpr size_t kSchedulerLeafAllowance = 4096;

// A wall clock for the placement measurement. Monotonic, because the question is
// how long a stage took and not what time it was.
double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// A context sized for a graph's headers plus the graph itself.
class GraphRun {
  public:
    GraphRun(const BackendPlan & plan, size_t nodes) : plan_(plan), nodes_(nodes) {
        ggml_init_params parameters{};
        parameters.mem_size = ggml_tensor_overhead() * (nodes + 256) + ggml_graph_overhead_custom(nodes, false);
        parameters.no_alloc = true;
        context_            = ggml_init(parameters);
        if (context_ != nullptr) {
            graph_ = ggml_new_graph_custom(context_, nodes, false);
        }
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

    ggml_cgraph * graph() const { return graph_; }

    bool ok() const { return context_ != nullptr && graph_ != nullptr; }

    // How long the scheduler took to be created and to place the graph, as
    // opposed to computing it. A per-step rebuild pays this every step, and
    // whether that dominates is the question stage 7 has to answer with a
    // number rather than an argument.
    double setup_seconds = 0.0;

    // `on_primary` places the graph on the primary backend rather than the CPU.
    // Only the codec ever asks for it: everything else feeds a sampled code.
    synth_status_t run(ggml_tensor * output, const char * stage, int threads, bool on_primary = false) {
        if (!ok() || output == nullptr) {
            return SYNTH_ERR_INTERNAL;
        }
        const double setup_started = now_seconds();
        ggml_build_forward_expand(graph_, output);
        // A CPU-only scheduler over CPU-resident weights is one split; forcing
        // nodes onto CPU inside a mixed graph is not the same thing and was
        // measured five times slower. See docs/backends.md.
        const size_t hash_size = size_t(ggml_graph_size(graph_)) + kSchedulerLeafAllowance;
        scheduler_             = on_primary ? plan_.create_scheduler(hash_size) : plan_.create_cpu_scheduler(hash_size);
        if (scheduler_ == nullptr) {
            return SYNTH_ERR_BACKEND;
        }
        if (!ggml_backend_sched_alloc_graph(scheduler_, graph_)) {
            return SYNTH_ERR_OOM;
        }
        plan_.log_placement_if_enabled(stage, scheduler_, graph_);
        plan_.set_threads(threads);
        setup_seconds = now_seconds() - setup_started;
        return ggml_backend_sched_graph_compute(scheduler_, graph_) == GGML_STATUS_SUCCESS ? SYNTH_OK :
                                                                                             SYNTH_ERR_BACKEND;
    }

  private:
    const BackendPlan &  plan_;
    size_t               nodes_;
    ggml_context *       context_   = nullptr;
    ggml_cgraph *        graph_     = nullptr;
    ggml_backend_sched_t scheduler_ = nullptr;
};

// A buffer of tensors the graphs read and write across calls: the key/value
// caches, which the graph allocator must not own because a decode step reads
// what earlier steps wrote.
class Persistent {
  public:
    ~Persistent() { reset(); }

    Persistent()                               = default;
    Persistent(const Persistent &)             = delete;
    Persistent & operator=(const Persistent &) = delete;

    bool open(size_t tensors) {
        reset();
        ggml_init_params parameters{};
        parameters.mem_size = ggml_tensor_overhead() * tensors;
        parameters.no_alloc = true;
        context_            = ggml_init(parameters);
        return context_ != nullptr;
    }

    ggml_context * context() const { return context_; }

    bool commit(ggml_backend_t backend) {
        buffer_ = ggml_backend_alloc_ctx_tensors(context_, backend);
        return buffer_ != nullptr;
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

// The public interface speaks BCP-47 and the package names its languages in
// full, so the two are bridged here rather than in either of them. Only the
// primary subtag is used: a region says nothing this model can act on.
//
// The package's two dialect entries are deliberately absent. They are speaker
// overrides the reference reaches through spk_is_dialect, not languages a
// request can ask for.
constexpr std::pair<const char *, const char *> kTags[] = {
    { "en", "english"    },
    { "de", "german"     },
    { "es", "spanish"    },
    { "zh", "chinese"    },
    { "ja", "japanese"   },
    { "fr", "french"     },
    { "ko", "korean"     },
    { "ru", "russian"    },
    { "it", "italian"    },
    { "pt", "portuguese" },
};

std::string language_name_for_tag(const std::string & tag) {
    const size_t      cut     = tag.find('-');
    const std::string primary = cut == std::string::npos ? tag : tag.substr(0, cut);
    for (const std::pair<const char *, const char *> & entry : kTags) {
        if (primary.size() == std::strlen(entry.first) &&
            std::equal(primary.begin(), primary.end(), entry.first,
                       [](char left, char right) { return std::tolower(left) == right; })) {
            return entry.second;
        }
    }
    return std::string();
}

// The reverse: the package's own name for a language back to the tag a request
// carries. Empty for a name this bridge does not cover, which is how the two
// dialect entries stay out of the published list.
std::string tag_for_language_name(const std::string & name) {
    for (const std::pair<const char *, const char *> & entry : kTags) {
        if (name == entry.second) {
            return entry.first;
        }
    }
    return std::string();
}

void read_floats(const ggml_tensor * tensor, std::vector<float> & output) {
    output.resize(size_t(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, output.data(), 0, ggml_nbytes(tensor));
}

}  // namespace

struct Model::Impl {
    gguf_context *                      gguf            = nullptr;
    ggml_context *                      weights_context = nullptr;
    std::unique_ptr<BackendPlan>        backend_plan;
    ggml_backend_buffer_t               weights_buffer = nullptr;
    // Twins of the codec half on the primary backend, present only when that is
    // not the CPU. The talker and the code predictor have no twins: their output
    // feeds a sampled code and docs/backends.md holds them on the CPU, so a
    // second copy would never be read.
    ggml_context *                      codec_context  = nullptr;
    ggml_backend_buffer_t               codec_buffer   = nullptr;
    HParams                             hparams;
    std::shared_ptr<const TextFrontend> frontend;
    ModelWeights                        weights;

    ~Impl() {
        if (weights_buffer != nullptr) {
            ggml_backend_buffer_free(weights_buffer);
        }
        if (codec_buffer != nullptr) {
            ggml_backend_buffer_free(codec_buffer);
        }
        if (codec_context != nullptr) {
            ggml_free(codec_context);
        }
        if (weights_context != nullptr) {
            ggml_free(weights_context);
        }
        if (gguf != nullptr) {
            gguf_free(gguf);
        }
    }

    AttentionShape talker_shape() const {
        AttentionShape shape;
        shape.hidden_size          = hparams.talker.hidden_size;
        shape.attention_head_count = hparams.talker.attention_head_count;
        shape.key_value_head_count = hparams.talker.key_value_head_count;
        shape.head_dim             = hparams.talker.head_dim;
        shape.rms_norm_eps         = hparams.talker.rms_norm_eps;
        shape.rope_theta           = hparams.talker.rope_theta;
        return shape;
    }

    AttentionShape predictor_shape() const {
        AttentionShape shape       = talker_shape();
        shape.attention_head_count = hparams.code_predictor.attention_head_count;
        shape.key_value_head_count = hparams.code_predictor.key_value_head_count;
        shape.head_dim             = hparams.code_predictor.head_dim;
        shape.hidden_size          = hparams.code_predictor.hidden_size;
        return shape;
    }
};

Model::Model(std::unique_ptr<Impl> implementation) : implementation_(std::move(implementation)) {}

Model::~Model() = default;

synth_status_t Model::get_info(ModelInfo & output) const {
    output                  = ModelInfo{};
    const HParams & hparams = implementation_->hparams;

    output.variant              = hparams.model_variant;
    output.architecture_version = hparams.architecture_version;
    switch (hparams.quantization_profile) {
        case QuantizationProfile::BF16:
            output.quantization_profile = "BF16";
            break;
        case QuantizationProfile::F16:
            output.quantization_profile = "F16";
            break;
        case QuantizationProfile::Q8Mixed:
            output.quantization_profile = "Q8_MIXED";
            break;
    }
    output.input_flags          = hparams.input_flags;
    output.capability_flags     = hparams.capability_flags;
    output.output_sample_rate   = hparams.output_sample_rate;
    output.output_channel_count = hparams.output_channel_count;
    output.max_input_tokens     = hparams.max_input_tokens;
    output.max_output_frames    = hparams.max_output_frames;
    output.min_speaking_rate    = hparams.min_speaking_rate;
    output.max_speaking_rate    = hparams.max_speaking_rate;
    output.has_package_default  = hparams.has_package_default;
    for (const PresetVoice & voice : hparams.preset_voices) {
        output.preset_voice_ids.push_back(voice.id);
    }
    output.language_names = hparams.language_names;
    // The same bridge as language_name_for_tag, walked the other way, so the
    // core can publish what a caller may ask for without knowing this package
    // names its languages in full. A name with no tag -- the two dialect
    // entries -- is dropped: they are speaker overrides, not languages a
    // request can name.
    for (const std::string & name : hparams.language_names) {
        const std::string tag = tag_for_language_name(name);
        if (!tag.empty()) {
            output.language_tags.push_back(tag);
        }
    }
    output.frontend_present  = hparams.frontend_present;
    output.frontend_provider = hparams.frontend_provider;
    return SYNTH_OK;
}

ggml_backend_device * Model::primary_device() const {
    return implementation_->backend_plan->primary_device();
}

std::shared_ptr<const TextFrontend> Model::text_frontend() const {
    return implementation_->frontend;
}

uint32_t Model::samples_per_frame() const {
    return implementation_->hparams.codec.hop_length;
}

uint32_t Model::text_vocab_size() const {
    return implementation_->hparams.talker.text_vocab_size;
}

synth_status_t Model::tokenize_request(const std::string & text, std::vector<int32_t> & token_ids) const {
    token_ids.clear();
    if (implementation_->frontend == nullptr) {
        return SYNTH_ERR_TEXT_FRONTEND;
    }
    // The frontend applies the turn wrapper itself, so this hands it the text.
    return implementation_->frontend->prepare(SYNTH_INPUT_TEXT_UTF8, text.data(), text.size(),
                                              implementation_->hparams.max_input_tokens, token_ids);
}

synth_status_t Model::resolve_voice(const std::string & voice_id,
                                    const std::string & language,
                                    uint32_t &          speaker_token,
                                    bool &              has_language,
                                    uint32_t &          language_token) const {
    speaker_token  = 0;
    has_language   = false;
    language_token = 0;

    PresetVoice voice;
    if (!find_preset_voice(implementation_->hparams, voice_id, voice)) {
        return SYNTH_ERR_UNSUPPORTED_VOICE;
    }
    speaker_token = voice.token_id;

    // "auto" is the no-think prompt, which carries no language token at all
    // rather than a default one.
    const std::string requested = language == "auto" ? std::string() : language_name_for_tag(language);
    if (language.empty() || language == "auto") {
        std::string resolved;
        // A speaker pinning a dialect still contributes its token under auto,
        // because the reference resolves the override before the request.
        if (!voice.dialect_override.empty() &&
            resolve_language_token(implementation_->hparams, requested, voice, language_token, resolved)) {
            has_language = true;
        }
        return SYNTH_OK;
    }
    if (requested.empty()) {
        return SYNTH_ERR_UNSUPPORTED_LANGUAGE;
    }
    std::string resolved;
    if (!resolve_language_token(implementation_->hparams, requested, voice, language_token, resolved)) {
        return SYNTH_ERR_UNSUPPORTED_LANGUAGE;
    }
    has_language = true;
    return SYNTH_OK;
}

synth_status_t Model::load_cpu(const std::string & path, std::unique_ptr<Model> & output) {
    return load(path, ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), false, output);
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
        // Twins of the codec half, so it can run on the primary backend while
        // the autoregressive half stays on the CPU. Declared before binding,
        // because the catalog binds the codec against them.
        const bool split = implementation->backend_plan->primary() != implementation->backend_plan->cpu_backend();
        if (split) {
            ggml_init_params twin_params{};
            twin_params.mem_size          = ggml_tensor_overhead() * 1024;
            twin_params.no_alloc          = true;
            implementation->codec_context = ggml_init(twin_params);
            if (implementation->codec_context == nullptr) {
                return SYNTH_ERR_OOM;
            }
            for (ggml_tensor * tensor = ggml_get_first_tensor(implementation->weights_context); tensor != nullptr;
                 tensor               = ggml_get_next_tensor(implementation->weights_context, tensor)) {
                if (std::strncmp(tensor->name, "codec.", 6) != 0) {
                    continue;
                }
                ggml_tensor * twin =
                    ggml_new_tensor(implementation->codec_context, tensor->type, ggml_n_dims(tensor), tensor->ne);
                if (twin == nullptr) {
                    return SYNTH_ERR_OOM;
                }
                ggml_set_name(twin, tensor->name);
            }
        }
        status = build_model_weights(implementation->weights_context, implementation->codec_context,
                                     implementation->hparams, implementation->weights);
        if (status != SYNTH_OK) {
            return status;
        }

        if (implementation->hparams.frontend_present) {
            GgufMetadata      meta(implementation->gguf, "qwen3-tts");
            BpeFrontendConfig config;
            config.provider_id      = implementation->hparams.frontend_provider;
            config.contract_version = implementation->hparams.frontend_contract_version;
            if (!meta.string_array("synthesize.qwen3-tts.frontend.vocab", config.vocab) ||
                !meta.string_array("synthesize.qwen3-tts.frontend.merges", config.merges)) {
                return SYNTH_ERR_GGUF;
            }
            // The chat markers are added tokens sitting past the vocabulary, so
            // they carry their ids rather than being looked up in it.
            config.special_tokens = {
                { "<|im_start|>", int32_t(implementation->hparams.tokens.im_start) },
                { "<|im_end|>",   int32_t(implementation->hparams.tokens.im_end)   },
            };
            // The reference wraps every request in an assistant turn, and the
            // talker's layout slices the tokenized result at its two fixed
            // counts, so the wrapping belongs here rather than in the caller.
            const std::string turn   = qwen_assistant_turn("");
            const size_t      middle = turn.find("<|im_end|>");
            config.prefix            = turn.substr(0, middle);
            config.suffix            = turn.substr(middle);
            std::unique_ptr<TextFrontend> frontend;
            status = make_bpe_frontend(config, frontend);
            if (status != SYNTH_OK) {
                return status;
            }
            implementation->frontend = std::shared_ptr<const TextFrontend>(std::move(frontend));
        }

        // Every graph runs on the CPU scheduler, so the weights live in the CPU
        // backend's buffer; see the note at the top of this file.
        implementation->weights_buffer = ggml_backend_alloc_ctx_tensors(implementation->weights_context,
                                                                        implementation->backend_plan->cpu_backend());
        if (implementation->weights_buffer == nullptr) {
            return SYNTH_ERR_OOM;
        }
        ggml_backend_buffer_set_usage(implementation->weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        status = stream_tensor_data(path, implementation->gguf, implementation->weights_context, "qwen3-tts");
        if (status != SYNTH_OK) {
            return status;
        }
        if (implementation->codec_context != nullptr) {
            implementation->codec_buffer =
                ggml_backend_alloc_ctx_tensors(implementation->codec_context, implementation->backend_plan->primary());
            if (implementation->codec_buffer == nullptr) {
                return SYNTH_ERR_OOM;
            }
            ggml_backend_buffer_set_usage(implementation->codec_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            // Copy after streaming, so the twins carry what the package holds.
            std::vector<unsigned char> scratch;
            for (ggml_tensor * twin = ggml_get_first_tensor(implementation->codec_context); twin != nullptr;
                 twin               = ggml_get_next_tensor(implementation->codec_context, twin)) {
                const ggml_tensor * source = ggml_get_tensor(implementation->weights_context, twin->name);
                if (source == nullptr || ggml_nbytes(source) != ggml_nbytes(twin)) {
                    return SYNTH_ERR_GGUF;
                }
                scratch.resize(ggml_nbytes(source));
                ggml_backend_tensor_get(source, scratch.data(), 0, scratch.size());
                ggml_backend_tensor_set(twin, scratch.data(), 0, scratch.size());
            }
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

// The talker's cache spans the utterance; the predictor's covers one frame and
// is reset before each. Both are written in place by the shared block, so both
// live outside the graph allocator.
struct Caches {
    Persistent         storage;
    TalkerCache        talker;
    CodePredictorCache predictor;
};

synth_status_t open_caches(const HParams & hparams, ggml_backend_t backend, int64_t talker_capacity, Caches & caches) {
    const uint32_t layers = hparams.talker.layer_count + hparams.code_predictor.layer_count;
    if (!caches.storage.open(size_t(layers) * 2 + 8)) {
        return SYNTH_ERR_OOM;
    }
    ggml_context * context = caches.storage.context();

    caches.talker.layers.resize(hparams.talker.layer_count);
    for (KvCache & layer : caches.talker.layers) {
        layer.k = ggml_new_tensor_3d(context, GGML_TYPE_F32, hparams.talker.head_dim,
                                     hparams.talker.key_value_head_count, talker_capacity);
        layer.v = ggml_new_tensor_3d(context, GGML_TYPE_F32, hparams.talker.head_dim,
                                     hparams.talker.key_value_head_count, talker_capacity);
    }

    // A frame is code_group_count positions in the predictor and never more.
    const int64_t predictor_capacity = int64_t(hparams.code_predictor.code_group_count);
    caches.predictor.layers.resize(hparams.code_predictor.layer_count);
    for (KvCache & layer : caches.predictor.layers) {
        layer.k = ggml_new_tensor_3d(context, GGML_TYPE_F32, hparams.code_predictor.head_dim,
                                     hparams.code_predictor.key_value_head_count, predictor_capacity);
        layer.v = ggml_new_tensor_3d(context, GGML_TYPE_F32, hparams.code_predictor.head_dim,
                                     hparams.code_predictor.key_value_head_count, predictor_capacity);
    }
    return caches.storage.commit(backend) ? SYNTH_OK : SYNTH_ERR_OOM;
}

template <typename Cache> void set_filled(Cache & cache, int64_t filled) {
    for (KvCache & layer : cache.layers) {
        layer.filled = filled;
    }
}

void set_i32(ggml_tensor * tensor, int32_t value) {
    ggml_backend_tensor_set(tensor, &value, 0, sizeof(value));
}

}  // namespace

namespace {
// Scratch for the codec's own wall clock, which decode_codes measures and
// run_synthesis reports. Both are const, and this is a measurement rather than
// state the model carries between calls.
thread_local double codec_seconds_ = 0.0;
}  // namespace

synth_status_t Model::decode_codes(const std::vector<int32_t> & codes,
                                   uint64_t                     frame_count,
                                   int                          threads,
                                   std::vector<float> &         audio) const {
    audio.clear();
    const Impl &    impl    = *implementation_;
    const HParams & hparams = impl.hparams;
    const uint32_t  groups  = hparams.talker.code_group_count;
    if (frame_count == 0 || codes.size() != size_t(frame_count) * groups) {
        return SYNTH_ERR_INVALID_ARG;
    }

    Persistent inputs;
    if (!inputs.open(8)) {
        return SYNTH_ERR_OOM;
    }
    ggml_context * ictx    = inputs.context();
    // The codec reads one level per row, so the frame-major stream the loop
    // accumulates is transposed on the way in.
    ggml_tensor *  t_codes = ggml_new_tensor_2d(ictx, GGML_TYPE_I32, int64_t(frame_count), int64_t(groups));
    ggml_tensor *  t_pos   = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, int64_t(frame_count));
    ggml_tensor *  t_mask  = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, int64_t(frame_count), int64_t(frame_count));
    if (!inputs.commit(impl.codec_context != nullptr ? impl.backend_plan->primary() :
                                                       impl.backend_plan->cpu_backend())) {
        return SYNTH_ERR_OOM;
    }

    std::vector<int32_t> level_major(codes.size());
    for (uint64_t frame = 0; frame < frame_count; ++frame) {
        for (uint32_t level = 0; level < groups; ++level) {
            level_major[size_t(level) * frame_count + frame] = codes[size_t(frame) * groups + level];
        }
    }
    ggml_backend_tensor_set(t_codes, level_major.data(), 0, ggml_nbytes(t_codes));

    std::vector<int32_t> sequential(size_t(frame_count), 0);
    for (uint64_t frame = 0; frame < frame_count; ++frame) {
        sequential[size_t(frame)] = int32_t(frame);
    }
    ggml_backend_tensor_set(t_pos, sequential.data(), 0, ggml_nbytes(t_pos));

    std::vector<float> mask(size_t(frame_count) * size_t(frame_count));
    codec_fill_sliding_window_mask(mask.data(), int64_t(frame_count), int64_t(hparams.codec.decoder.sliding_window));
    ggml_backend_tensor_set(t_mask, mask.data(), 0, ggml_nbytes(t_mask));

    // The codec is one pass over the whole stream, so its node count does not
    // grow with the frame count.
    GraphRun run(*impl.backend_plan, 8192);
    if (!run.ok()) {
        return SYNTH_ERR_OOM;
    }
    ggml_tensor * wav = build_codec_decoder(run.context(), t_codes, t_pos, t_mask, impl.weights.codec, hparams);
    if (wav == nullptr) {
        return SYNTH_ERR_INTERNAL;
    }
    const double         started = now_seconds();
    const synth_status_t status  = run.run(wav, "qwen3-tts.codec", threads, impl.codec_context != nullptr);
    if (status != SYNTH_OK) {
        return status;
    }
    read_floats(wav, audio);
    codec_seconds_ = now_seconds() - started;
    return SYNTH_OK;
}

synth_status_t Model::run_synthesis(const SynthesisRequest & request, SynthesisOutput & output) const {
    output                  = SynthesisOutput{};
    const Impl &    impl    = *implementation_;
    const HParams & hparams = impl.hparams;
    const uint32_t  groups  = hparams.talker.code_group_count;

    // The turn has to carry text between its markers, or the prompt's final
    // position has no token to sit on.
    if (request.token_ids.size() <= kAssistantRolePrefixTokens + kAssistantSuffixTokens) {
        return SYNTH_ERR_INVALID_ARG;
    }

    uint32_t       speaker_token  = 0;
    bool           has_language   = false;
    uint32_t       language_token = 0;
    synth_status_t status =
        resolve_voice(request.voice_id, request.language, speaker_token, has_language, language_token);
    if (status != SYNTH_OK) {
        return status;
    }

    TalkerPromptRequest prompt_request;
    prompt_request.role_tokens.assign(request.token_ids.begin(),
                                      request.token_ids.begin() + kAssistantRolePrefixTokens);
    prompt_request.text_tokens.assign(request.token_ids.begin() + kAssistantRolePrefixTokens,
                                      request.token_ids.end() - kAssistantSuffixTokens);
    prompt_request.has_speaker    = true;
    prompt_request.speaker_token  = speaker_token;
    prompt_request.has_language   = has_language;
    prompt_request.language_token = language_token;

    TalkerPrompt prompt;
    status = build_talker_prompt(hparams, prompt_request, prompt);
    if (status != SYNTH_OK) {
        return status;
    }

    std::vector<int32_t> prompt_text;
    std::vector<int32_t> prompt_codec;
    int64_t              codec_offset = 0;
    status                            = flatten_talker_prompt(hparams, prompt, prompt_text, prompt_codec, codec_offset);
    if (status != SYNTH_OK) {
        return status;
    }

    const int64_t prefill    = int64_t(prompt.positions.size());
    // The cache is sized from what this request may actually emit, not from the
    // package's declared ceiling; see kDefaultMaxFrames.
    uint64_t      max_frames = request.max_frames != 0 ? request.max_frames : kDefaultMaxFrames;
    max_frames               = std::min(max_frames, hparams.max_output_frames);
    // The talker attends over the prompt plus one position per frame it emits.
    Caches caches;
    status = open_caches(hparams, impl.backend_plan->cpu_backend(), prefill + int64_t(max_frames) + 1, caches);
    if (status != SYNTH_OK) {
        return status;
    }

    Persistent inputs;
    if (!inputs.open(24)) {
        return SYNTH_ERR_OOM;
    }
    ggml_context * ictx           = inputs.context();
    ggml_tensor *  t_prompt_text  = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, prefill);
    ggml_tensor *  t_prompt_codec = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, int64_t(prompt_codec.size()));
    ggml_tensor *  t_prompt_pos   = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, prefill);
    ggml_tensor *  t_prompt_mask  = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, prefill, prefill);
    ggml_tensor *  t_step_text    = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, 1);
    ggml_tensor *  t_step_pos     = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, 1);
    ggml_tensor *  t_acoustic     = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, int64_t(groups) - 1);
    ggml_tensor *  t_semantic     = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, 1);
    ggml_tensor *  t_previous     = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, 1);
    ggml_tensor *  t_hidden       = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, hparams.code_predictor.hidden_size, 1);
    ggml_tensor *  t_pair_pos     = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, 2);
    ggml_tensor *  t_pair_mask    = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, 2, 2);
    ggml_tensor *  t_one_pos      = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, 1);
    if (!inputs.commit(impl.backend_plan->cpu_backend())) {
        return SYNTH_ERR_OOM;
    }

    ggml_backend_tensor_set(t_prompt_text, prompt_text.data(), 0, ggml_nbytes(t_prompt_text));
    ggml_backend_tensor_set(t_prompt_codec, prompt_codec.data(), 0, ggml_nbytes(t_prompt_codec));
    {
        std::vector<int32_t> sequential(size_t(prefill), 0);
        for (int64_t index = 0; index < prefill; ++index) {
            sequential[size_t(index)] = int32_t(index);
        }
        ggml_backend_tensor_set(t_prompt_pos, sequential.data(), 0, ggml_nbytes(t_prompt_pos));
        // The prompt is one causal block; the talker has no sliding window.
        std::vector<float> mask(size_t(prefill) * size_t(prefill));
        codec_fill_sliding_window_mask(mask.data(), prefill, 0);
        ggml_backend_tensor_set(t_prompt_mask, mask.data(), 0, ggml_nbytes(t_prompt_mask));
    }
    {
        const int32_t pair[2] = { 0, 1 };
        ggml_backend_tensor_set(t_pair_pos, pair, 0, ggml_nbytes(t_pair_pos));
        std::vector<float> mask(4);
        codec_fill_sliding_window_mask(mask.data(), 2, 0);
        ggml_backend_tensor_set(t_pair_mask, mask.data(), 0, ggml_nbytes(t_pair_mask));
    }

    const AttentionShape                 talker_shape    = impl.talker_shape();
    const AttentionShape                 predictor_shape = impl.predictor_shape();
    const std::vector<CodePredictorStep> schedule        = code_predictor_schedule(groups);
    if (schedule.size() + 1 != groups) {
        return SYNTH_ERR_INTERNAL;
    }

    SamplingParams sampling;
    sampling.enabled     = request.sample;
    sampling.temperature = request.temperature;
    sampling.top_k       = request.top_k;
    sampling.top_p       = request.top_p;
    NormalRandomStream stream(request.seed);

    // Sized from the layer count rather than guessed: each block is under fifty
    // nodes and the heads and projections add a fixed tail.
    const size_t talker_nodes    = size_t(hparams.talker.layer_count) * 64 + 512;
    const size_t predictor_nodes = size_t(hparams.code_predictor.layer_count) * 64 + 512;

    // Replay means the loop takes each frame's codes from the oracle instead of
    // drawing them, so everything downstream of the draw is compared on
    // identical inputs. It also bounds the loop: the replay ends when its frames
    // do, with no stop code involved.
    const bool     replaying   = request.replay_codes != nullptr && request.replay_frames != 0;
    const uint64_t frame_limit = replaying ? std::min<uint64_t>(request.replay_frames, max_frames) : max_frames;
    if (replaying && request.replay_codes->size() < size_t(frame_limit) * groups) {
        return SYNTH_ERR_INVALID_ARG;
    }
    for (uint32_t layer : request.probe_layers) {
        if (layer >= hparams.talker.layer_count) {
            return SYNTH_ERR_INVALID_ARG;
        }
    }
    output.talker_layers.resize(request.probe_layers.size());

    std::vector<int32_t> frame_codes(groups, 0);
    std::vector<float>   logits;
    std::vector<float>   hidden_state;
    std::vector<float>   probe;
    int64_t              filled = 0;

    for (uint64_t frame = 0; frame <= frame_limit; ++frame) {
        GraphRun talker(*impl.backend_plan, talker_nodes);
        if (!talker.ok()) {
            return SYNTH_ERR_OOM;
        }
        ggml_context * tctx = talker.context();

        ggml_tensor * input     = nullptr;
        ggml_tensor * positions = nullptr;
        ggml_tensor * mask      = nullptr;
        if (frame == 0) {
            input = build_talker_prefill_input(tctx, impl.weights.talker, t_prompt_text, t_prompt_codec, codec_offset);
            positions = t_prompt_pos;
            mask      = t_prompt_mask;
        } else {
            // A later step is the frame's summed code embeddings plus the text it
            // contributes, which is tts_pad once the text has run out.
            set_i32(t_step_text, int32_t(talker_step_text_token(hparams, prompt, frame - 1)));
            set_i32(t_step_pos, int32_t(filled));

            ggml_tensor * acoustic = sum_code_embeddings(tctx, impl.weights.code_predictor, t_acoustic);
            // Group 0 is embedded through the talker's own table, not the
            // predictor's, which is the one asymmetry in the sum.
            ggml_tensor * semantic = ggml_get_rows(tctx, impl.weights.talker.codec_embedding, t_semantic);
            if (acoustic == nullptr) {
                return SYNTH_ERR_INTERNAL;
            }
            input = build_talker_step_input(tctx, impl.weights.talker, ggml_add(tctx, acoustic, semantic), t_step_text);
            positions = t_step_pos;
        }
        if (input == nullptr) {
            return SYNTH_ERR_INTERNAL;
        }

        set_filled(caches.talker, filled);
        // The oracle probes the prefill pass alone, because that is the pass a
        // port reproduces without the sampled history behind it. So the probes
        // are captured at frame zero over every prompt position, not one row per
        // frame -- the two have the same shape whenever the frame count happens
        // to equal the prompt length, which is how a wrong capture survives.
        const bool                 probing    = !request.probe_layers.empty() && frame == 0;
        ggml_tensor *              hidden     = nullptr;
        ggml_tensor *              all_hidden = nullptr;
        std::vector<ggml_tensor *> layers;
        ggml_tensor *              talker_logits =
            build_talker_step(tctx, talker.graph(), input, positions, mask, impl.weights.talker, talker_shape,
                              caches.talker, &hidden, probing ? &layers : nullptr, probing ? &all_hidden : nullptr);
        if (talker_logits == nullptr || hidden == nullptr) {
            return SYNTH_ERR_INTERNAL;
        }
        // Everything read back has to be marked, or the allocator reuses one
        // tensor's buffer for another; see docs/porting/families/qwen3-tts.md.
        ggml_set_output(hidden);
        ggml_build_forward_expand(talker.graph(), hidden);
        ggml_tensor * prefill_logits = nullptr;
        if (probing) {
            for (uint32_t layer : request.probe_layers) {
                ggml_set_output(layers[layer]);
                ggml_build_forward_expand(talker.graph(), layers[layer]);
            }
            ggml_set_output(all_hidden);
            ggml_build_forward_expand(talker.graph(), all_hidden);
            // The oracle's logits probe covers the whole prefill, so the head is
            // applied to every position rather than only the one that predicts.
            prefill_logits = ggml_mul_mat(tctx, impl.weights.talker.codec_head, all_hidden);
            ggml_set_output(prefill_logits);
            ggml_build_forward_expand(talker.graph(), prefill_logits);
        }
        const double talker_started = now_seconds();
        status                      = talker.run(talker_logits, "qwen3-tts.talker", request.threads);
        if (status != SYNTH_OK) {
            return status;
        }
        output.talker_seconds += now_seconds() - talker_started;
        filled += input->ne[1];

        read_floats(talker_logits, logits);
        if (replaying) {
            if (frame == frame_limit) {
                break;
            }
            frame_codes[0] = (*request.replay_codes)[size_t(frame) * groups];
        } else {
            frame_codes[0] = int32_t(select_code(logits, sampling, stream));
            if (talker_frame_ends_utterance(hparams, uint32_t(frame_codes[0]))) {
                break;
            }
            if (frame == frame_limit) {
                return SYNTH_ERR_OUTPUT_LIMIT;
            }
        }
        read_floats(hidden, hidden_state);

        if (probing) {
            read_floats(prefill_logits, output.talker_logits);
            read_floats(all_hidden, output.talker_final);
            for (size_t index = 0; index < request.probe_layers.size(); ++index) {
                read_floats(layers[request.probe_layers[index]], probe);
                output.talker_layers[index] = probe;
            }
        }
        ggml_backend_tensor_set(t_hidden, hidden_state.data(), 0, ggml_nbytes(t_hidden));
        set_i32(t_semantic, frame_codes[0]);

        // The code predictor expands that semantic code into the frame's
        // acoustic ones, its cache starting empty every frame.
        int64_t predictor_filled = 0;
        for (size_t step = 0; step < schedule.size(); ++step) {
            const CodePredictorStep & plan = schedule[step];
            GraphRun                  run(*impl.backend_plan, predictor_nodes);
            if (!run.ok()) {
                return SYNTH_ERR_OOM;
            }
            ggml_context * pctx = run.context();

            ggml_tensor * step_input = nullptr;
            ggml_tensor * step_pos   = nullptr;
            ggml_tensor * step_mask  = nullptr;
            if (plan.embedding_table == kPrefillStep) {
                // The talker hands over its hidden state and its own embedding of
                // the semantic code, in that order and as one two-position block.
                ggml_tensor * embedded = ggml_get_rows(pctx, impl.weights.talker.codec_embedding, t_semantic);
                step_input             = ggml_concat(pctx, t_hidden, embedded, 1);
                step_pos               = t_pair_pos;
                step_mask              = t_pair_mask;
            } else {
                set_i32(t_previous, frame_codes[plan.embedding_table + 1]);
                set_i32(t_one_pos, int32_t(plan.first_position));
                step_input = code_predictor_embed(pctx, impl.weights.code_predictor, plan.embedding_table, t_previous);
                step_pos   = t_one_pos;
            }
            if (step_input == nullptr) {
                return SYNTH_ERR_INTERNAL;
            }

            set_filled(caches.predictor, predictor_filled);
            ggml_tensor * step_logits =
                build_code_predictor(pctx, run.graph(), step_input, step_pos, step_mask, impl.weights.code_predictor,
                                     predictor_shape, plan.lm_head, caches.predictor);
            if (step_logits == nullptr) {
                return SYNTH_ERR_INTERNAL;
            }
            const double step_started = now_seconds();
            status                    = run.run(step_logits, "qwen3-tts.code-predictor", request.threads);
            if (status != SYNTH_OK) {
                return status;
            }
            output.predictor_seconds += now_seconds() - step_started;
            output.predictor_setup_seconds += run.setup_seconds;
            predictor_filled += plan.position_count;

            read_floats(step_logits, logits);
            frame_codes[step + 1] = replaying ? (*request.replay_codes)[size_t(frame) * groups + step + 1] :
                                                int32_t(select_code(logits, sampling, stream));
        }

        ggml_backend_tensor_set(t_acoustic, frame_codes.data() + 1, 0, ggml_nbytes(t_acoustic));
        output.codes.insert(output.codes.end(), frame_codes.begin(), frame_codes.end());
        ++output.frame_count;
    }

    if (output.frame_count == 0) {
        // Stopping on the first frame is an empty utterance, not a failure.
        return SYNTH_OK;
    }
    status               = decode_codes(output.codes, output.frame_count, request.threads, output.audio);
    output.codec_seconds = codec_seconds_;
    return status;
}

}  // namespace synth::qwen3tts
