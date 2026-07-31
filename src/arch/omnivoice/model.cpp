// The OmniVoice model object: what a package becomes once it is loaded.
//
// Plan 1 loads and reports; the diffusion loop, the codec and the cloning path
// are Plan 2's. That split is deliberate rather than partial work — the load
// path is the only place where the converter, the metadata reader, the tensor
// catalog and the frontend meet, and it is worth closing on the real package
// before any graph exists to blame a wrong number on.
//
// Placement: everything is CPU. This family's canvas is a table of sampled
// codes, so docs/backends.md's discrete-output rule holds the generator and its
// whole input path on the CPU; the codec reads a committed grid and could move,
// but Plan 2 has no measurement to move it on, so it runs on the CPU scheduler
// like everything else. The weights therefore live in one CPU buffer with no
// accelerator twin — the seam qwen3-tts carries for its codec half arrives with
// the stage that can use it.

#include "arch/omnivoice/catalog.h"
#include "arch/omnivoice/codec-host.h"
#include "arch/omnivoice/codec.h"
#include "arch/omnivoice/generator-host.h"
#include "arch/omnivoice/generator.h"
#include "arch/omnivoice/omnivoice.h"
#include "arch/omnivoice/weights.h"
#include "backend-plan.h"
#include "bpe-frontend.h"
#include "cpu-parallelism.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-metadata.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <new>
#include <utility>
#include <vector>

namespace synth::omnivoice {

namespace {

// Headroom in the scheduler's hash set for the weights a graph reads, which
// enter as leaves rather than nodes.
constexpr size_t kSchedulerLeafAllowance = 4096;

// The replay seam calls decode_codes free-standing, so there is no output
// object in scope to write its timing and placement into; run_synthesis copies
// them out of this scratch straight after its own call. The qwen3-tts pattern
// (src/arch/qwen3-tts/model.cpp), sited above run_synthesis here because this
// file defines the two members the other way round -- a measurement, not state
// the model carries between calls.
thread_local double   codec_seconds_           = 0.0;
thread_local uint64_t codec_placed_nodes_      = 0;
thread_local uint64_t codec_accelerator_nodes_ = 0;

// A wall clock for the placement measurement. Monotonic, because the question is
// how long a stage took and not what time it was.
double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// Keeps the narrowest margin seen so far. A NaN margin means the scoring broke,
// so it wins the comparison rather than losing to every real number and leaving
// the report reading like a healthy run; once recorded it is not displaced.
void note_margin(MarginReport &     report,
                 MarginReport::Kind kind,
                 float              value,
                 uint32_t           step,
                 uint32_t           codebook,
                 uint64_t           frame) {
    if (report.measured) {
        const bool already_broken = std::isnan(report.value);
        const bool narrower       = std::isnan(value) || value < report.value;
        if (already_broken || !narrower) {
            return;
        }
    }
    report.measured = true;
    report.kind     = kind;
    report.value    = value;
    report.step     = step;
    report.codebook = codebook;
    report.frame    = frame;
}

// Duplicated from qwen3-tts rather than shared: the third copy is the signal
// to hoist (the BPE rule), and stage 7's graph-reuse question may reshape this
// family's copy anyway. A fresh GraphRun per forward is the measured, known
// pattern; its cost is what setup_seconds exists to expose.

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

    // Where this graph's nodes were actually placed. Read from the scheduler
    // after allocation, which is the only moment the answer exists: before it
    // there is no assignment, and after compute the scheduler has been freed.
    uint64_t placed_nodes      = 0;
    uint64_t accelerator_nodes = 0;

    // `on_primary` places the graph on the primary backend rather than the CPU.
    // NO Plan-2 caller passes it: the generator may not (its output is a
    // sampled code, and docs/backends.md's discrete-outputs rule holds it and
    // its whole input path on the CPU), and the codec -- the one stage that
    // could -- takes the false default here too, because Plan 2 has no
    // measurement to move it on. The parameter is kept as the seam stage 7
    // needs to move a stage without reworking this class, matching the
    // qwen3-tts precedent; that rule is what will decide which stages may ever
    // pass true. See the placement note at the top of this file, which says the
    // same thing about today's state.
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
        const BackendPlacement placement = plan_.inspect_placement(scheduler_, graph_);
        placed_nodes                     = placement.node_count - placement.view_node_count;
        accelerator_nodes                = placement.off_cpu_node_count;
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

// A buffer of tensors the graphs read and write across calls: here the canvas
// id and position inputs, which the graph allocator must not own because a
// decode step refills what the previous step's graph read.
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

void read_floats(const ggml_tensor * tensor, std::vector<float> & output) {
    output.resize(size_t(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, output.data(), 0, ggml_nbytes(tensor));
}

// Which probe buffers a forward should fill; empty = no probes.
struct ForwardProbeSinks {
    const std::vector<uint32_t> *     layer_indices = nullptr;
    std::vector<float> *              logits_full   = nullptr;
    std::vector<float> *              final_hidden  = nullptr;
    std::vector<std::vector<float>> * layer_hidden  = nullptr;
};

// One full-canvas forward of one CFG branch on the CPU scheduler. Reads back
// the FULL logits [vocab, codebooks, positions] into `logits`; the caller
// slices the target region (the trailing target_frames positions). text_ids
// is null for the unconditional branch, whose every position is an audio slot.
//
// Takes the plan/weights/hparams pieces rather than Model::Impl: a file-local
// function cannot name a private nested type, and passing the pieces keeps it
// callable from every member without a friend declaration.
synth_status_t generator_branch_forward(const BackendPlan &       plan,
                                        const ModelWeights &      weights,
                                        const HParams &           hparams,
                                        ggml_tensor *             text_ids,
                                        ggml_tensor *             audio_ids,
                                        ggml_tensor *             positions,
                                        int                       threads,
                                        const ForwardProbeSinks & probes,
                                        std::vector<float> &      logits,
                                        SynthesisOutput &         output) {
    const AttentionShape shape{ hparams.generator.hidden_size,          hparams.generator.attention_head_count,
                                hparams.generator.key_value_head_count, hparams.generator.head_dim,
                                hparams.generator.rms_norm_eps,         hparams.generator.rope_theta };
    // Each block is under fifty nodes; the embedding merge and the head add a
    // fixed tail (the qwen3-tts budget formula).
    const size_t         nodes = size_t(hparams.generator.layer_count) * 64 + 512;
    GraphRun             run(plan, nodes);
    if (!run.ok()) {
        return SYNTH_ERR_OOM;
    }
    ggml_tensor * embeddings =
        build_canvas_embedding(run.context(), weights.generator, text_ids, audio_ids, hparams.audio.num_codebooks);
    std::vector<ggml_tensor *> layer_tensors;
    ggml_tensor *              final_tensor = nullptr;
    const bool                 probing      = probes.logits_full != nullptr;
    ggml_tensor *              logits_tensor =
        build_generator_forward(run.context(), embeddings, positions, nullptr, weights.generator, shape, hparams.audio,
                                probing ? &layer_tensors : nullptr, probing ? &final_tensor : nullptr);
    if (logits_tensor == nullptr) {
        return SYNTH_ERR_INTERNAL;
    }
    if (probing) {
        // Read-back tensors that are not the graph output must be marked and
        // expanded before allocation, or the allocator reuses their buffers.
        for (ggml_tensor * tensor : layer_tensors) {
            ggml_set_output(tensor);
            ggml_build_forward_expand(run.graph(), tensor);
        }
        ggml_set_output(final_tensor);
        ggml_build_forward_expand(run.graph(), final_tensor);
    }
    const double         started = now_seconds();
    const synth_status_t status  = run.run(logits_tensor, "omnivoice.generator", threads);
    if (status != SYNTH_OK) {
        return status;
    }
    // All four accumulate ACROSS forwards, and a decode run makes two per step:
    // `generator_placement.nodes` is a running total, not a graph's node count.
    // Only the probe-only path (one forward) leaves it equal to one graph's.
    output.generator_seconds += now_seconds() - started;
    output.generator_setup_seconds += run.setup_seconds;
    output.generator_placement.nodes += run.placed_nodes;
    output.generator_placement.accelerator_nodes += run.accelerator_nodes;

    read_floats(logits_tensor, logits);
    if (probing) {
        *probes.logits_full = logits;
        read_floats(final_tensor, *probes.final_hidden);
        probes.layer_hidden->clear();
        for (uint32_t wanted : *probes.layer_indices) {
            if (wanted >= layer_tensors.size()) {
                return SYNTH_ERR_INVALID_ARG;
            }
            std::vector<float> values;
            read_floats(layer_tensors[wanted], values);
            probes.layer_hidden->push_back(std::move(values));
        }
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
    std::shared_ptr<const TextFrontend> frontend;
    ModelWeights                        weights;

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
    output                  = ModelInfo{};
    const HParams & hparams = implementation_->hparams;

    output.variant              = hparams.model_variant;
    output.architecture_version = hparams.architecture_version;
    switch (hparams.quantization_profile) {
        case QuantizationProfile::F32:
            output.quantization_profile = "F32";
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
    // Auto-voice is the package default and the Preset Voice Catalog is empty;
    // the metadata reader refuses a package that says otherwise, so there is no
    // per-package value to carry here.
    output.has_package_default  = true;
    // No name bridge: this family's tags are already BCP-47, and the same string
    // is what the prompt's language slot carries.
    output.language_tags        = hparams.language_tags;
    output.frontend_present     = hparams.frontend_present;
    output.frontend_provider    = hparams.frontend_provider;
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
    return implementation_->hparams.generator.text_vocab_size;
}

synth_status_t Model::run_synthesis(const SynthesisRequest & request, SynthesisOutput & output) {
    output                  = SynthesisOutput{};
    Impl &          impl    = *implementation_;
    const HParams & hparams = impl.hparams;

    if (request.prompt_text_ids.empty() || request.target_frames == 0) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (request.target_frames > hparams.max_output_frames) {
        return SYNTH_ERR_OUTPUT_LIMIT;
    }
    const uint32_t num_step = request.num_step != 0 ? request.num_step : hparams.generation.num_step;
    const float    guidance = hparams.generation.guidance_scale;
    const int      threads  = request.threads > 0 ? request.threads : default_synthesis_threads();

    PromptLayout   prompt;
    synth_status_t status = build_prompt_grid(request.prompt_text_ids, request.reference_tokens, request.target_frames,
                                              hparams.audio.num_codebooks, hparams.audio.mask_id, prompt);
    if (status != SYNTH_OK) {
        return status;
    }
    const uint64_t total     = prompt.total();
    const uint64_t frames    = prompt.target_frames;
    const uint32_t codebooks = hparams.audio.num_codebooks;
    const uint32_t vocab     = hparams.audio.vocab_size;
    // Set before the probe-only return rather than beside `codes` at the end:
    // the canvas length is settled here, and a probe-only run that reported
    // zero frames while holding a full canvas of probes was reading as an empty
    // synthesis.
    output.frame_count       = frames;

    // Every reusable input lives in one persistent buffer; the per-step
    // refills touch only the audio-id tensors.
    Persistent inputs;
    if (!inputs.open(8)) {
        return SYNTH_ERR_OOM;
    }
    ggml_context * ictx   = inputs.context();
    ggml_tensor *  t_text = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, int64_t(prompt.text_length));
    ggml_tensor *  t_cond_audio =
        ggml_new_tensor_2d(ictx, GGML_TYPE_I32, int64_t(prompt.audio_length()), int64_t(codebooks));
    ggml_tensor * t_uncond_audio = ggml_new_tensor_2d(ictx, GGML_TYPE_I32, int64_t(frames), int64_t(codebooks));
    ggml_tensor * t_cond_pos     = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, int64_t(total));
    ggml_tensor * t_uncond_pos   = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, int64_t(frames));
    if (!inputs.commit(impl.backend_plan->cpu_backend())) {
        return SYNTH_ERR_OOM;
    }
    // The unconditional branch's canvas is the TARGET REGION ALONE -- no style
    // markers, no text, no reference audio -- refilled from the same committed
    // canvas the conditional branch's target region carries, and read at
    // positions that restart at zero. That is the reference's second batch row:
    // dropping the condition means dropping every conditioning token, not
    // blanking them in place, so its sequence is `target_frames` long where the
    // conditional one is `total`. Nothing compares it directly -- the oracle
    // dumps no unconditional probe -- so the exact-token gate is the only thing
    // that holds this definition honest.

    ggml_backend_tensor_set(t_text, request.prompt_text_ids.data(), 0, ggml_nbytes(t_text));
    // static_cast rather than a functional cast: `size_t(total)` here parses as
    // a parameter declaration, which makes the whole line a function
    // declaration rather than a vector.
    std::vector<int32_t> sequential(static_cast<size_t>(total));
    for (uint64_t index = 0; index < total; ++index) {
        sequential[size_t(index)] = int32_t(index);
    }
    ggml_backend_tensor_set(t_cond_pos, sequential.data(), 0, ggml_nbytes(t_cond_pos));
    // The unconditional branch is its own sequence: positions restart at zero,
    // exactly as the reference's padded batch gives its second row.
    ggml_backend_tensor_set(t_uncond_pos, sequential.data(), 0, ggml_nbytes(t_uncond_pos));

    std::vector<int32_t> shifted;
    fill_shifted_audio_ids(prompt.grid.data(), total, prompt.audio_start(), prompt.audio_length(), codebooks, vocab,
                           shifted);
    ggml_backend_tensor_set(t_cond_audio, shifted.data(), 0, ggml_nbytes(t_cond_audio));

    // --- Step-0 conditional forward, with probes when asked. Its logits are
    // also step 0's conditional half once the loop runs.
    ForwardProbeSinks sinks;
    const bool        probing = !request.probe_layers.empty();
    if (probing) {
        sinks.layer_indices = &request.probe_layers;
        sinks.logits_full   = &output.logits_step0;
        sinks.final_hidden  = &output.final_hidden;
        sinks.layer_hidden  = &output.layer_hidden;
    }
    std::vector<float> cond_logits;
    status = generator_branch_forward(*impl.backend_plan, impl.weights, hparams, t_text, t_cond_audio, t_cond_pos,
                                      threads, sinks, cond_logits, output);
    if (status != SYNTH_OK) {
        return status;
    }
    if (request.probe_only) {
        return SYNTH_OK;
    }

    // The canonical canvas: the target region's committed state, codebook-major
    // [codebooks * frames], all mask at step 0. The reference writes committed
    // tokens back into both the conditional and unconditional rows; here both
    // branches' id tensors are refilled from this one grid, so there is a
    // single source of truth instead of two copies to keep in step.
    std::vector<int32_t> canvas(size_t(codebooks) * frames, int32_t(hparams.audio.mask_id));

    const std::vector<uint64_t> schedule =
        commit_schedule(uint64_t(codebooks) * frames, num_step, double(hparams.generation.t_shift));

    std::vector<float>           uncond_logits;
    std::vector<int32_t>         uncond_shifted;
    std::vector<MaskedCandidate> candidates;
    const bool                   track_margin = request.margin_report;
    // One position's logits in the read-back buffer: codebooks * vocab floats,
    // position-major (position s starts at s * row).
    const uint64_t               row          = uint64_t(codebooks) * vocab;
    const ForwardProbeSinks      no_probes;

    for (uint32_t step = 0; step < num_step; ++step) {
        if (step > 0) {
            // The reference tokens never move; only the target region follows
            // the canvas. Refill and rerun the conditional branch (step 0's ran
            // above, with the probes).
            for (uint32_t codebook = 0; codebook < codebooks; ++codebook) {
                int32_t * target_row =
                    prompt.grid.data() + size_t(codebook) * total + prompt.audio_start() + prompt.reference_frames;
                std::memcpy(target_row, canvas.data() + size_t(codebook) * frames, size_t(frames) * sizeof(int32_t));
            }
            fill_shifted_audio_ids(prompt.grid.data(), total, prompt.audio_start(), prompt.audio_length(), codebooks,
                                   vocab, shifted);
            ggml_backend_tensor_set(t_cond_audio, shifted.data(), 0, ggml_nbytes(t_cond_audio));
            status = generator_branch_forward(*impl.backend_plan, impl.weights, hparams, t_text, t_cond_audio,
                                              t_cond_pos, threads, no_probes, cond_logits, output);
            if (status != SYNTH_OK) {
                return status;
            }
        }
        if (guidance != 0.0f) {
            // The unconditional branch carries the target region only -- no
            // style markers, no text, no reference audio.
            fill_shifted_audio_ids(canvas.data(), frames, 0, frames, codebooks, vocab, uncond_shifted);
            ggml_backend_tensor_set(t_uncond_audio, uncond_shifted.data(), 0, ggml_nbytes(t_uncond_audio));
            status = generator_branch_forward(*impl.backend_plan, impl.weights, hparams, nullptr, t_uncond_audio,
                                              t_uncond_pos, threads, no_probes, uncond_logits, output);
            if (status != SYNTH_OK) {
                return status;
            }
        }

        const uint64_t budget = schedule[step];
        if (budget == 0) {
            // The forwards above have already run, because this loop keeps the
            // reference's shape: upstream computes the whole batch and only
            // then does its per-item `if k <= 0: continue`. Nothing consumes
            // those logits here, so hoisting the test above the forwards would
            // produce the same grid, only faster -- shape fidelity is the
            // reason to run them, not necessity.
            continue;
        }
        candidates.clear();
        for (uint32_t codebook = 0; codebook < codebooks; ++codebook) {
            for (uint64_t frame = 0; frame < frames; ++frame) {
                if (canvas[size_t(codebook) * frames + frame] != int32_t(hparams.audio.mask_id)) {
                    continue;  // committed in an earlier step; cannot be revisited
                }
                const size_t    cond_offset   = size_t(total - frames + frame) * row + size_t(codebook) * vocab;
                const size_t    uncond_offset = size_t(frame) * row + size_t(codebook) * vocab;
                MaskedCandidate candidate;
                candidate.codebook = codebook;
                candidate.frame    = frame;
                float log_prob     = 0.0f;
                choose_token(cond_logits.data() + cond_offset,
                             guidance != 0.0f ? uncond_logits.data() + uncond_offset : nullptr, vocab,
                             hparams.audio.mask_id, guidance, candidate.token, log_prob,
                             track_margin ? &candidate.argmax_gap : nullptr);
                // The layer penalty biases commitment toward the coarse
                // codebooks first. (audio_codebook_weights is training-loss
                // weighting and plays NO part here -- see the family doc.)
                candidate.score = log_prob - float(codebook) * hparams.generation.layer_penalty_factor;
                candidates.push_back(candidate);
            }
        }
        const size_t committed = select_commits(candidates, budget);
        if (track_margin) {
            if (committed < candidates.size()) {
                // A partial step. What could have gone the other way is the
                // boundary between the last candidate kept and the best one
                // rejected; partial_sort leaves the tail unordered, so the best
                // rejected is whichever of it the commit order puts first.
                const MaskedCandidate & kept = candidates[committed - 1];
                const auto              rival =
                    std::min_element(candidates.begin() + ptrdiff_t(committed), candidates.end(), commits_before);
                note_margin(output.margin, MarginReport::Kind::selection, kept.score - rival->score, step,
                            kept.codebook, kept.frame);
            } else {
                // Nothing was rejected, so no position was in contest and the
                // only decision left to be narrow is each committed token
                // against its own runner-up.
                for (size_t index = 0; index < committed; ++index) {
                    const MaskedCandidate & choice = candidates[index];
                    note_margin(output.margin, MarginReport::Kind::argmax, choice.argmax_gap, step, choice.codebook,
                                choice.frame);
                }
            }
        }
        for (size_t index = 0; index < committed; ++index) {
            const MaskedCandidate & choice                          = candidates[index];
            canvas[size_t(choice.codebook) * frames + choice.frame] = choice.token;
        }
    }

    for (int32_t token : canvas) {
        if (token == int32_t(hparams.audio.mask_id)) {
            std::fprintf(stderr, "omnivoice: a mask survived the schedule; the loop is wrong\n");
            return SYNTH_ERR_INTERNAL;
        }
    }
    output.codes = std::move(canvas);

    status = decode_codes(output.codes, frames, threads, output.audio);
    if (status != SYNTH_OK) {
        return status;
    }
    output.codec_seconds                     = codec_seconds_;
    output.codec_placement.nodes             = codec_placed_nodes_;
    output.codec_placement.accelerator_nodes = codec_accelerator_nodes_;
    // Decision 3 of the plan: the residual output scaling lives HERE, inside
    // the family's synthesis path, faithful to the oracle. Auto-voice and
    // voice-design requests take the no-reference branch; the clone branches
    // arrive with Plan 3's reference handling, which is the only thing that can
    // supply the reference RMS the other two arms key off -- so a Plan 2
    // request carrying reference TOKENS still takes this branch, and the replay
    // seam applies the oracle's own branch separately rather than reading this
    // one.
    apply_no_reference_volume(output.audio);
    return SYNTH_OK;
}

synth_status_t Model::decode_codes(const std::vector<int32_t> & codes,
                                   uint64_t                     frame_count,
                                   int                          threads,
                                   std::vector<float> &         audio) {
    audio.clear();
    Impl &          impl    = *implementation_;
    const HParams & hparams = impl.hparams;
    const uint32_t  groups  = hparams.audio.num_codebooks;

    synth_status_t status = validate_code_grid(codes, frame_count, groups, hparams.codec.codebook_size);
    if (status != SYNTH_OK) {
        return status;
    }

    Persistent inputs;
    if (!inputs.open(2)) {
        return SYNTH_ERR_OOM;
    }
    ggml_tensor * t_codes = ggml_new_tensor_2d(inputs.context(), GGML_TYPE_I32, int64_t(frame_count), int64_t(groups));
    if (!inputs.commit(impl.backend_plan->cpu_backend())) {
        return SYNTH_ERR_OOM;
    }
    // The committed grid is codebook-major [c * frames + t], which IS the
    // level-major row layout the quantizer's per-level views read -- no
    // transpose, unlike qwen3-tts's frame-major stream.
    ggml_backend_tensor_set(t_codes, codes.data(), 0, ggml_nbytes(t_codes));

    // One pass over the whole stream; the node count does not grow with the
    // frame count (the qwen3-tts codec budget).
    GraphRun run(*impl.backend_plan, 8192);
    if (!run.ok()) {
        return SYNTH_ERR_OOM;
    }
    ggml_tensor * wave = build_codec_decoder(run.context(), t_codes, impl.weights, hparams);
    if (wave == nullptr) {
        return SYNTH_ERR_INTERNAL;
    }
    const double started = now_seconds();
    status               = run.run(wave, "omnivoice.codec", threads > 0 ? threads : default_synthesis_threads());
    if (status != SYNTH_OK) {
        return status;
    }
    codec_seconds_           = now_seconds() - started;
    codec_placed_nodes_      = run.placed_nodes;
    codec_accelerator_nodes_ = run.accelerator_nodes;
    read_floats(wave, audio);
    if (audio.size() != size_t(frame_count) * hparams.codec.hop_length) {
        std::fprintf(stderr, "omnivoice: the codec produced %zu samples for %llu frames\n", audio.size(),
                     (unsigned long long) frame_count);
        return SYNTH_ERR_INTERNAL;
    }
    return SYNTH_OK;
}

synth_status_t Model::load_cpu(const std::string & path, std::unique_ptr<Model> & output) {
    return load(path, ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), false, output);
}

synth_status_t Model::load(const std::string &      path,
                           ggml_backend_device *    primary_device,
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
        status = build_model_weights(implementation->weights_context, implementation->hparams, implementation->weights);
        if (status != SYNTH_OK) {
            return status;
        }

        if (implementation->hparams.frontend_present) {
            GgufMetadata      meta(implementation->gguf, "omnivoice");
            BpeFrontendConfig config;
            config.provider_id      = implementation->hparams.frontend_provider;
            config.contract_version = implementation->hparams.frontend_contract_version;
            if (!meta.string_array("synthesize.omnivoice.frontend.vocab", config.vocab) ||
                !meta.string_array("synthesize.omnivoice.frontend.merges", config.merges)) {
                return SYNTH_ERR_GGUF;
            }
            // The prompt markers are added tokens past the vocabulary; they
            // carry their ids. No prefix/suffix: this family's prompt is
            // assembled by the synthesis path, not wrapped at tokenize time.
            const SpecialTokens & t = implementation->hparams.tokens;
            config.special_tokens   = {
                { "<|denoise|>",        int32_t(t.denoise)        },
                { "<|lang_start|>",     int32_t(t.lang_start)     },
                { "<|lang_end|>",       int32_t(t.lang_end)       },
                { "<|instruct_start|>", int32_t(t.instruct_start) },
                { "<|instruct_end|>",   int32_t(t.instruct_end)   },
                { "<|text_start|>",     int32_t(t.text_start)     },
                { "<|text_end|>",       int32_t(t.text_end)       },
            };
            std::unique_ptr<TextFrontend> frontend;
            status = make_bpe_frontend(config, frontend);
            if (status != SYNTH_OK) {
                return status;
            }
            implementation->frontend = std::shared_ptr<const TextFrontend>(std::move(frontend));
        }

        // One CPU buffer, no twin; see the placement note at the top of this
        // file.
        implementation->weights_buffer = ggml_backend_alloc_ctx_tensors(implementation->weights_context,
                                                                        implementation->backend_plan->cpu_backend());
        if (implementation->weights_buffer == nullptr) {
            return SYNTH_ERR_OOM;
        }
        ggml_backend_buffer_set_usage(implementation->weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        status = stream_tensor_data(path, implementation->gguf, implementation->weights_context, "omnivoice");
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

}  // namespace synth::omnivoice
