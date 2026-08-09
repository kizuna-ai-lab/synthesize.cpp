// The OmniVoice model object: what a package becomes once it is loaded.
//
// Plan 1 loads and reports; the diffusion loop, the codec and the cloning path
// are Plan 2's. That split is deliberate rather than partial work — the load
// path is the only place where the converter, the metadata reader, the tensor
// catalog and the frontend meet, and it is worth closing on the real package
// before any graph exists to blame a wrong number on.
//
// Placement, as of Plan 5 Task 1: the generator can now run on an accelerator
// too, following jiangzhuo's 2026-08-08 ruling that this family's bar is
// audible quality rather than token identity (a six-pair blind listening test
// heard no problem in generator-on-CPU vs generator-on-CUDA output, including
// a pair whose token grids agree at only 1.7%). Before this task,
// docs/backends.md's discrete-outputs rule held the generator and everything
// feeding it on the CPU on every Execution Backend, because its own committed
// token grid is a discrete decision; that rule's text is not yet amended
// (Plan 5 Task 5 owns that), but this family's own measurement now qualifies
// for the exception the rule's rationale never actually forbade: the canvas
// LENGTH is fixed by RuleDurationEstimator before the first generator forward
// runs, so a discrete decision here can change WHICH token is committed but
// never the downstream tensor SHAPE the way Kokoro's duration-to-frame-count
// rounding could. The clone-encode chain (reference-encoder-host.cpp) stays on
// the CPU unconditionally regardless: its own output is continuous, but what
// reads it -- rvq_encode's host-side nearest-neighbour argmax -- is a discrete
// decision feeding no downstream forward at all, so there is no fixed-shape
// argument to make for it the way there is for the generator's own canvas.
//
// Both the codec's DECODE half and the generator can therefore run on an
// accelerator: decode_codes's committed code grid is the last discrete value
// on the codec's own path, so build_codec_decoder's RVQ-sum/fc2/DAC-decoder
// graph is free to move (Task 9, Plan 4); the generator's own canvas-length
// argument above is why generator_branch_forward's graph is free to move too
// (Task 1, Plan 5). Two independent twins make this possible --
// Model::Impl::codec_context/codec_buffer and
// Model::Impl::generator_context/generator_buffer, each present only when the
// primary backend is not the CPU. See the comments above their construction in
// Model::load for the exact tensor groups and byte cost, and catalog.h's
// bind_decode_weights/bind_generator_weights for how the catalog binds against
// each.
//
// codec.quantizer is read by BOTH directions -- the decode graph's dequantize
// sum and encode_reference's own host-side RVQ nearest-neighbour argmax
// (rvq_encode) -- which a first draft of Task 9 missed: it let the twin
// re-resolve overwrite the one ModelWeights every reader shared, so
// rvq_encode would have started reading a CUDA-resident tensor through
// ggml_backend_tensor_get the moment Task 11 gave this family a CUDA primary.
// Fixed by never mutating `weights` for either twin at all: `Model::Impl`
// carries `weights` (bound to the package, read by every host-side caller
// including rvq_encode), a `decode_weights` (bind_decode_weights's own output,
// read only by decode_codes), and a `generator_weights` (bind_generator_weights's
// own output, read only by generator_branch_forward) side by side. The
// generator has no second host-side consumer analogous to rvq_encode -- Task
// 1's own pre-flight grep found none -- but the split is built the same
// defensive way regardless: a future reader that bypasses
// generator_branch_forward must keep seeing the CPU-resident package by
// construction, not by continued vigilance. With no accelerator both twins are
// null, `decode_weights`/`generator_weights` are exact copies of `weights`, and
// every path here runs exactly as it did before either twin existed.

#include "arch/omnivoice/catalog.h"
#include "arch/omnivoice/codec-host.h"
#include "arch/omnivoice/codec.h"
#include "arch/omnivoice/frontend-host.h"
#include "arch/omnivoice/generator-host.h"
#include "arch/omnivoice/generator.h"
#include "arch/omnivoice/omnivoice.h"
#include "arch/omnivoice/profile.h"
#include "arch/omnivoice/reference-encoder-host.h"
#include "arch/omnivoice/weights.h"
#include "backend-plan.h"
#include "bpe-frontend.h"
#include "cpu-parallelism.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-metadata.h"
#include "gguf.h"
#include "random-stream.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace synth::omnivoice {

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

// note_margin lives in generator-host.{h,cpp} now, where its NaN-wins update
// rule is unit-testable without a Model.

// Duplicated from qwen3-tts rather than shared: the third copy is the signal
// to hoist (the BPE rule). This family's copy has since diverged -- it is
// reusable across forwards where qwen3-tts's is not -- which is the second
// reason not to have shared it.

// A context sized for a graph's headers plus the graph itself.
//
// Setup is idempotent. The first run() creates the scheduler and allocates the
// graph; every later run() on the same object recomputes that SAME allocated
// graph with whatever its input leaves now hold. This is ggml's documented
// "single-use in terms of allocation, multi-use in terms of computation"
// contract (ggml/include/ggml-backend.h), and the allocate-once half is
// load-bearing rather than a mere optimization: ggml_gallocr_init_tensor
// (ggml/src/ggml-alloc.c) leaves a tensor's `data` pointer alone once it is
// non-null, so re-allocating an already-allocated graph would silently keep
// addresses that a differently-shaped intervening allocation has re-planned
// over -- wrong numbers, and no assert fires. The shape below makes that
// unreachable by construction: allocation happens only on the
// `scheduler_ == nullptr` path, and scheduler_ becomes non-null there and
// stays non-null until destruction. Do NOT add a ggml_backend_sched_reset
// call; it clears the scheduler's `is_alloc` and re-opens exactly that door.
//
// Reuse is therefore per SHAPE, not per object lifetime: one GraphRun may only
// ever be handed one canvas geometry. Model::run_synthesis keeps one per CFG
// branch, on its own stack frame, because the two branches' canvases differ in
// length and every canvas is fixed for the life of one synthesis call.
class GraphRun {
  public:
    GraphRun(const BackendPlan & plan, size_t nodes) : plan_(plan) {
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

    // The compute interval of the LAST run(): ggml_backend_sched_graph_compute
    // and nothing else. Callers derive their non-compute overhead by
    // subtracting this from the whole of the work they did, which is what makes
    // that overhead complete rather than a hand-picked subset -- see
    // SynthesisOutput::generator_setup_seconds in omnivoice.h.
    double compute_seconds = 0.0;

    // Where this graph's nodes were actually placed. Read from the scheduler
    // once, at allocation, which is the only moment the answer exists: before
    // it there is no assignment, and after the scheduler is freed there is
    // nothing to ask. Retained across reuse so that a caller accumulating per
    // forward still sees one graph's worth per forward.
    uint64_t placed_nodes      = 0;
    uint64_t accelerator_nodes = 0;

    // `on_primary` places the graph on the primary backend rather than the CPU.
    // Model::decode_codes passes true when a codec twin exists
    // (Model::Impl::codec_context != nullptr): its output is the last discrete
    // value on the decode path, so the DAC decoder graph downstream of it is
    // free to move. Since Plan 5 Task 1, generator_branch_forward passes true
    // too when a generator twin exists (Model::Impl::generator_context !=
    // nullptr): the canvas length is fixed before the first forward runs, so a
    // discrete decision here changes token CONTENT, never downstream tensor
    // SHAPE -- see the placement note at the top of this file for the full
    // argument and its Kokoro counter-example. Neither twin exists with no
    // accelerator, so `on_primary` is always false in that configuration and
    // every call site's behavior is unchanged from before either twin existed.
    synth_status_t run(ggml_tensor * output, const char * stage, int threads, bool on_primary = false) {
        if (!ok() || output == nullptr) {
            return SYNTH_ERR_INTERNAL;
        }
        if (scheduler_ == nullptr) {
            ggml_build_forward_expand(graph_, output);
            // A CPU-only scheduler over CPU-resident weights is one split;
            // forcing nodes onto CPU inside a mixed graph is not the same thing
            // and was measured five times slower. See docs/backends.md.
            const size_t hash_size = size_t(ggml_graph_size(graph_)) + kSchedulerLeafAllowance;
            scheduler_ = on_primary ? plan_.create_scheduler(hash_size) : plan_.create_cpu_scheduler(hash_size);
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
        }
        // Every forward, not only the first. set_threads pokes every backend
        // globally rather than this scheduler, so making it conditional would
        // be a behavior change for anything else that reads a thread count
        // mid-loop. It costs microseconds; the reuse win is not in it.
        plan_.set_threads(threads);
        const double      compute_started = now_seconds();
        const ggml_status status          = ggml_backend_sched_graph_compute(scheduler_, graph_);
        compute_seconds                   = now_seconds() - compute_started;
        return status == GGML_STATUS_SUCCESS ? SYNTH_OK : SYNTH_ERR_BACKEND;
    }

    // Whether this object has been set up, i.e. whether run() would take the
    // allocation path. Callers use it to decide whether the graph still needs
    // building; see GeneratorBranch.
    bool prepared() const { return scheduler_ != nullptr; }

    // The compute buffers this graph's allocation owns, summed over the
    // scheduler's backends. Only meaningful once prepared(); reuse holds this
    // live for the whole synthesis where a per-forward rebuild held it for one
    // forward, which is the peak-memory cost of reuse and is reported by
    // log_placement_if_enabled's own line.
    size_t buffer_bytes() const {
        if (scheduler_ == nullptr) {
            return 0;
        }
        size_t total = 0;
        for (int index = 0; index < ggml_backend_sched_get_n_backends(scheduler_); ++index) {
            total += ggml_backend_sched_get_buffer_size(scheduler_, ggml_backend_sched_get_backend(scheduler_, index));
        }
        return total;
    }

  private:
    const BackendPlan &  plan_;
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

// The three name prefixes build_codec_decoder actually reads: the RVQ
// dequantization sum, fc2, and the DAC decoder. This is deliberately narrower
// than a blanket `codec.` prefix (qwen3-tts's own twin filter, where it is
// correct because nothing else under `codec.` there is CPU-held) -- see
// catalog.h's bind_decode_weights for the full group/byte-cost accounting and
// why the rest of `codec.` must NOT be twinned. Note that `codec.quantizer.`
// being movable here is about which CONTEXT its twin lives in, not which
// ModelWeights reads it -- rvq_encode reads the same group through a
// different, never-mutated binding; see that same comment.
constexpr const char * kDecodePathPrefixes[] = {
    "codec.acoustic_decoder.",
    "codec.quantizer.",
    "codec.fc2.",
};

bool is_decode_path_tensor(const char * name) {
    for (const char * prefix : kDecodePathPrefixes) {
        if (std::strncmp(name, prefix, std::strlen(prefix)) == 0) {
            return true;
        }
    }
    return false;
}

// The whole generator group: `llm.*` (the Qwen3 backbone, one prefix covers
// the embedding, every layer and the final norm) plus the two audio tables,
// which sit outside the `llm.` prefix in the catalog's own naming (catalog.cpp's
// layout comment). Unlike kDecodePathPrefixes above, this is not a narrowing
// of a wider group -- see catalog.h's bind_generator_weights for why the whole
// generator is movable and has no NOT-MOVABLE remainder to exclude.
constexpr const char * kGeneratorPrefixes[] = {
    "llm.",
};
constexpr const char * kGeneratorExactNames[] = {
    "audio_embeddings.weight",
    "audio_heads.weight",
};

bool is_generator_tensor(const char * name) {
    for (const char * prefix : kGeneratorPrefixes) {
        if (std::strncmp(name, prefix, std::strlen(prefix)) == 0) {
            return true;
        }
    }
    for (const char * exact : kGeneratorExactNames) {
        if (std::strcmp(name, exact) == 0) {
            return true;
        }
    }
    return false;
}

// One CFG branch's graph, kept alive across the forwards that share its canvas
// geometry. Built on the first forward and recomputed by every later one; see
// GraphRun's own comment for why the pairing is per shape and why the two
// branches may not share one.
//
// `logits` is the built graph's output tensor, which the caller reads back
// after each compute. It is owned by the GraphRun's arena, so it is only valid
// while `run` is.
struct GeneratorBranch {
    std::unique_ptr<GraphRun> run;
    ggml_tensor *             logits = nullptr;
};

// Which probe buffers a forward should fill; empty = no probes.
struct ForwardProbeSinks {
    const std::vector<uint32_t> *     layer_indices = nullptr;
    std::vector<float> *              logits_full   = nullptr;
    std::vector<float> *              final_hidden  = nullptr;
    std::vector<std::vector<float>> * layer_hidden  = nullptr;
};

// One full-canvas forward of one CFG branch, on the primary backend when
// `on_primary` is true (a generator twin exists -- Plan 5 Task 1) and on the
// CPU scheduler otherwise. Reads back the FULL logits [vocab, codebooks,
// positions] into `logits`; the caller slices the target region (the trailing
// target_frames positions). text_ids is null for the unconditional branch,
// whose every position is an audio slot.
//
// Takes the plan/weights/hparams pieces rather than Model::Impl: a file-local
// function cannot name a private nested type, and passing the pieces keeps it
// callable from every member without a friend declaration. `weights` is
// `Model::Impl::generator_weights` at every call site (Model::run_synthesis),
// never `Model::Impl::weights` directly -- see this file's top-of-file
// placement note and catalog.h's bind_generator_weights for why the two must
// not be conflated.
//
// `branch` carries the graph between calls: the first forward of a branch
// builds it, every later forward with the same canvas geometry recomputes it.
// The caller owns the pairing of a branch to a shape -- see GeneratorBranch.
synth_status_t generator_branch_forward(const BackendPlan &       plan,
                                        const ModelWeights &      weights,
                                        const HParams &           hparams,
                                        GeneratorBranch &         branch,
                                        ggml_tensor *             text_ids,
                                        ggml_tensor *             audio_ids,
                                        ggml_tensor *             positions,
                                        int                       threads,
                                        bool                      on_primary,
                                        const ForwardProbeSinks & probes,
                                        std::vector<float> &      logits,
                                        SynthesisOutput &         output) {
    // Everything this function does that is NOT the compute is overhead the
    // reuse work exists to delete, so the bracket opens here -- ahead of the
    // arena and the node build, which the pre-reuse instrumentation omitted
    // entirely. See SynthesisOutput::generator_setup_seconds in omnivoice.h.
    const double               entered = now_seconds();
    const bool                 probing = probes.logits_full != nullptr;
    std::vector<ggml_tensor *> layer_tensors;
    ggml_tensor *              final_tensor = nullptr;
    if (branch.run == nullptr) {
        const AttentionShape shape{ hparams.generator.hidden_size,          hparams.generator.attention_head_count,
                                    hparams.generator.key_value_head_count, hparams.generator.head_dim,
                                    hparams.generator.rms_norm_eps,         hparams.generator.rope_theta };
        // Each block is under fifty nodes; the embedding merge and the head add
        // a fixed tail (the qwen3-tts budget formula).
        const size_t         nodes = size_t(hparams.generator.layer_count) * 64 + 512;
        // Built into a local and only handed to `branch` once it is whole, so a
        // failed build leaves the branch empty rather than poisoned with a
        // half-built graph a later forward would happily reuse.
        auto                 run   = std::make_unique<GraphRun>(plan, nodes);
        if (!run->ok()) {
            return SYNTH_ERR_OOM;
        }
        ggml_tensor * embeddings =
            build_canvas_embedding(run->context(), weights.generator, text_ids, audio_ids, hparams.audio.num_codebooks);
        ggml_tensor * logits_tensor = build_generator_forward(
            run->context(), embeddings, positions, nullptr, weights.generator, shape, hparams.audio,
            probing ? &layer_tensors : nullptr, probing ? &final_tensor : nullptr);
        if (logits_tensor == nullptr) {
            return SYNTH_ERR_INTERNAL;
        }
        if (probing) {
            // Read-back tensors that are not the graph output must be marked and
            // expanded before allocation, or the allocator reuses their buffers.
            for (ggml_tensor * tensor : layer_tensors) {
                ggml_set_output(tensor);
                ggml_build_forward_expand(run->graph(), tensor);
            }
            ggml_set_output(final_tensor);
            ggml_build_forward_expand(run->graph(), final_tensor);
        }
        branch.run    = std::move(run);
        branch.logits = logits_tensor;
    } else if (probing) {
        // Unreachable from Model::run_synthesis, which gives the probing
        // forward a branch of its own: those ggml_set_output marks make the
        // probe graph a third shape, and the layer/final handles the read-back
        // needs exist only on the build path above. Refused rather than
        // silently returning a forward with no probes filled in.
        return SYNTH_ERR_INTERNAL;
    }
    const synth_status_t status = branch.run->run(branch.logits, "omnivoice.generator", threads, on_primary);
    if (status != SYNTH_OK) {
        return status;
    }
    // All four accumulate ACROSS forwards, and a decode run makes two per step:
    // `generator_placement.nodes` is a running total, not a graph's node count.
    // Only the probe-only path (one forward) leaves it equal to one graph's.
    // The placement pair is the GraphRun's retained answer from its single
    // allocation, so a reused branch still contributes one graph's worth per
    // forward and the running total is what it was before reuse.
    const double elapsed = now_seconds() - entered;
    output.generator_seconds += elapsed;
    output.generator_setup_seconds += elapsed - branch.run->compute_seconds;
    output.generator_placement.nodes += branch.run->placed_nodes;
    output.generator_placement.accelerator_nodes += branch.run->accelerator_nodes;

    read_floats(branch.logits, logits);
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

// The compute-buffer cost of holding both branches at once, behind the same
// switch the placement dump uses. Reuse raises peak device memory: a
// per-forward rebuild held one branch's buffer at a time, and this holds the
// conditional and unconditional buffers together for the whole decode loop.
// That is irrelevant on a large unified-memory device and is exactly the number
// that decides it on a small discrete one, so it is reportable rather than
// argued about. Called while both branches are still live.
void log_branch_buffers_if_enabled(const GeneratorBranch & conditional, const GeneratorBranch & unconditional) {
    const char * enabled = std::getenv("SYNTH_DEBUG_BACKEND_PLACEMENT");
    if (enabled == nullptr || enabled[0] == '\0') {
        return;
    }
    const size_t cond_bytes   = conditional.run == nullptr ? 0 : conditional.run->buffer_bytes();
    const size_t uncond_bytes = unconditional.run == nullptr ? 0 : unconditional.run->buffer_bytes();
    std::fprintf(stderr, "omnivoice_generator_buffers: conditional=%llu unconditional=%llu both_live=%llu\n",
                 static_cast<unsigned long long>(cond_bytes), static_cast<unsigned long long>(uncond_bytes),
                 static_cast<unsigned long long>(cond_bytes + uncond_bytes));
}

// Frees a branch's scheduler and arena, charging the cost to the same field the
// rest of the branch's non-compute time goes to. Teardown is not a rounding
// error -- ggml_backend_sched_free is where the compute buffer's device free
// happens -- and before reuse it was paid 64 times per synthesis while being
// counted by no field at all. Called explicitly rather than left to scope exit
// so it lands inside the measurement; the destructor still runs on the error
// paths, which report no timings anyway.
void release_branch(GeneratorBranch & branch, SynthesisOutput & output) {
    if (branch.run == nullptr) {
        return;
    }
    const double started = now_seconds();
    branch.run.reset();
    branch.logits        = nullptr;
    const double elapsed = now_seconds() - started;
    output.generator_seconds += elapsed;
    output.generator_setup_seconds += elapsed;
}

}  // namespace

struct Model::Impl {
    gguf_context *                      gguf            = nullptr;
    ggml_context *                      weights_context = nullptr;
    std::unique_ptr<BackendPlan>        backend_plan;
    ggml_backend_buffer_t               weights_buffer    = nullptr;
    // Twin of the codec's decode path on the primary backend, present only
    // when that is not the CPU. See catalog.h's build_model_weights/
    // bind_decode_weights and this file's placement comment for the tensor
    // group and byte cost.
    ggml_context *                      codec_context     = nullptr;
    ggml_backend_buffer_t               codec_buffer      = nullptr;
    // Twin of the generator (Plan 5 Task 1), present under the same condition
    // as codec_context above. The clone-encode chain is the one stage that
    // still has no twin and stays on the CPU unconditionally regardless of the
    // primary backend: its own output is continuous, but what reads it --
    // rvq_encode's host-side nearest-neighbour argmax -- is a discrete
    // decision feeding no downstream forward, so there is no fixed-canvas-shape
    // argument to move it the way this file's placement note makes for the
    // generator. See catalog.h's bind_generator_weights for the tensor group
    // and byte cost.
    ggml_context *                      generator_context = nullptr;
    ggml_backend_buffer_t               generator_buffer  = nullptr;
    HParams                             hparams;
    std::shared_ptr<const TextFrontend> frontend;
    // `weights` is bound against `weights_context` alone, ALWAYS -- every
    // host-side reader (rvq_encode via encode_reference) reads this and only
    // this, so it can never be made to read a twin. `decode_weights` is
    // bind_decode_weights's own output (read only by Model::decode_codes) and
    // `generator_weights` is bind_generator_weights's own output (read only by
    // generator_branch_forward via Model::run_synthesis); each is identical to
    // `weights` in every field when its own twin context is null (no
    // accelerator; see catalog.h's header comments on both bind_* functions
    // for the second-consumer trap this split closes), diverging only in the
    // fields its own twin re-resolves.
    ModelWeights                        weights;
    ModelWeights                        decode_weights;
    ModelWeights                        generator_weights;

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
        if (generator_buffer != nullptr) {
            ggml_backend_buffer_free(generator_buffer);
        }
        if (generator_context != nullptr) {
            ggml_free(generator_context);
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
        case QuantizationProfile::Q8Mixed:
            output.quantization_profile = "Q8_MIXED";
            break;
        case QuantizationProfile::F16:
            output.quantization_profile = "F16";
            break;
        case QuantizationProfile::Q8Gen:
            output.quantization_profile = "Q8_GEN";
            break;
        case QuantizationProfile::Q4KGen:
            output.quantization_profile = "Q4_K_GEN";
            break;
        case QuantizationProfile::F16Gen:
            output.quantization_profile = "F16_GEN";
            break;
        case QuantizationProfile::BF16Gen:
            output.quantization_profile = "BF16_GEN";
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
    output.profile              = hparams.profile;
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

uint32_t Model::audio_vocab_size() const {
    return implementation_->hparams.audio.vocab_size;
}

uint32_t Model::audio_mask_id() const {
    return implementation_->hparams.audio.mask_id;
}

synth_status_t Model::run_synthesis(const SynthesisRequest & request, SynthesisOutput & output) {
    output                  = SynthesisOutput{};
    Impl &          impl    = *implementation_;
    const HParams & hparams = impl.hparams;

    if (request.prompt_text_ids.empty() || request.target_frames == 0) {
        return SYNTH_ERR_INVALID_ARG;
    }
    // The ids index the text-embedding table through ggml_get_rows, which has
    // no bounds of its own -- an out-of-range row is an abort inside ggml, not
    // a status. The frontend only emits in-vocabulary ids; this holds the
    // request seam itself to the same range.
    for (int32_t id : request.prompt_text_ids) {
        if (id < 0 || uint32_t(id) >= hparams.generator.text_vocab_size) {
            return SYNTH_ERR_INVALID_ARG;
        }
    }
    // hparams.max_output_frames is the package's declared ceiling in native
    // PCM frames (docs/c-interface.md); request.target_frames is a codec/
    // decoder-frame count (one committed grid column, hop_length PCM samples
    // each). Converting the ceiling into the same codec-frame unit before
    // comparing is the fix for PR #6's finding: the package once carried a
    // codec-frame count directly in this PCM-frame field, and comparing the
    // two without converting made that 960x error invisible. hop_length is
    // validated non-zero at load (weights.cpp).
    const uint64_t max_output_frames_codec = hparams.max_output_frames / hparams.codec.hop_length;
    if (request.target_frames > max_output_frames_codec) {
        return SYNTH_ERR_OUTPUT_LIMIT;
    }
    const uint32_t num_step = request.num_step != 0 ? request.num_step : hparams.generation.num_step;
    const float    guidance = hparams.generation.guidance_scale;
    const int      threads  = request.threads > 0 ? request.threads : default_synthesis_threads();
    // Negative means "the package's own default governs"; 0.0f is a
    // meaningful value (greedy), not an unset one. See SynthesisRequest's own
    // comment in omnivoice.h.
    const float    pos_t =
        request.position_temperature < 0.0f ? hparams.generation.position_temperature : request.position_temperature;
    const float class_t =
        request.class_temperature < 0.0f ? hparams.generation.class_temperature : request.class_temperature;
    // A Gumbel-perturbed margin measures nothing meaningful: the report exists
    // to screen the greedy loop's narrowest decisions, and sampling replaces
    // "narrow" with "randomly won or lost" -- a different axis entirely.
    if (request.margin_report && (pos_t > 0.0f || class_t > 0.0f)) {
        return SYNTH_ERR_INVALID_ARG;
    }

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
    // On the primary backend whenever the generator's graph will be, so the
    // graph reads these leaves without a cross-backend copy -- decode_codes's
    // own rule for its input leaf, applied here since Plan 5 Task 1 gives the
    // generator the same option.
    if (!inputs.commit(impl.generator_context != nullptr ? impl.backend_plan->primary() :
                                                           impl.backend_plan->cpu_backend())) {
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
    // The two graphs this synthesis reuses, one per CFG branch. They are scoped
    // to THIS CALL and must stay that way: the canvas geometry each one is
    // allocated for is fixed for the life of the call (prompt.total() and
    // prompt.target_frames are settled above, before the first forward, and
    // nothing in the loop moves them) but differs from call to call, so there
    // is no invalidation check to get wrong here -- and hoisting them to
    // Model::Impl would need a shape key, a mutex, and would break the
    // Loaded Model's immutable-and-shareable contract. Declared AFTER `inputs`
    // so they destruct BEFORE it: the built graphs hold pointers into that
    // buffer's tensors.
    GeneratorBranch    cond_branch;
    GeneratorBranch    uncond_branch;
    {
        // Probing expands extra ggml_set_output tensors into the graph, making
        // it a third shape that no later forward wants; it happens once, so it
        // gets a branch of its own that is released as soon as it is read.
        GeneratorBranch probe_branch;
        status = generator_branch_forward(*impl.backend_plan, impl.generator_weights, hparams,
                                          probing ? probe_branch : cond_branch, t_text, t_cond_audio, t_cond_pos,
                                          threads, impl.generator_context != nullptr, sinks, cond_logits, output);
        release_branch(probe_branch, output);
        if (status != SYNTH_OK) {
            return status;
        }
    }
    if (request.probe_only) {
        release_branch(cond_branch, output);
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

    // ONE stream drives every draw this synthesis makes, constructed HERE --
    // before the step loop -- iff either resolved temperature above is
    // positive. A fully greedy synthesis (both temperatures 0) constructs no
    // stream at all and therefore draws nothing: the family's recorded
    // zero-RNG property.
    //
    // Draw-order contract: per step, still-masked candidates are enumerated in
    // codebook-major, frame-minor scan order, and every uniform that step will
    // consume is drawn UP FRONT in that order, by the serial pre-pass below --
    // then handed to each candidate by index. Per candidate the order within
    // its own slice is unchanged: the class draws come FIRST (only when
    // class_t > 0 -- exactly topk_keep(vocab) of them, consumed in ascending
    // class-id order by choose_token_sampled), then, iff pos_t > 0, exactly
    // ONE further draw perturbs that same candidate's score. The resulting
    // sequence of next_uniform() calls is identical, value for value, to the
    // one an inline per-candidate draw produced -- which is what makes the
    // scoring safe to spread across threads and the output bit-identical
    // either way. It is identical only because the count per candidate is
    // fixed (topk_keep does not depend on the logits, and the survivor loop
    // has no early exit): a data-dependent draw count would break the
    // correspondence, so anything that made one must move the draws back
    // inside and give up the parallel scan.
    //
    // A step whose budget is zero skips this whole per-candidate phase,
    // INCLUDING every draw it would have made: upstream's `k <= 0: continue`
    // consumes no randomness for that step either, and the `continue` below
    // already sits above the pre-pass.
    std::unique_ptr<NormalRandomStream> stream;
    if (pos_t > 0.0f || class_t > 0.0f) {
        stream = std::make_unique<NormalRandomStream>(request.seed);
    }
    // How many uniforms one candidate consumes, and where inside its slice the
    // position draw sits. Both are constants of the request, not of the data.
    const uint32_t     class_draws    = class_t > 0.0f ? topk_keep(vocab) : 0u;
    const size_t       draws_per_slot = size_t(class_draws) + (pos_t > 0.0f ? 1u : 0u);
    std::vector<float> uniforms;

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
            status = generator_branch_forward(*impl.backend_plan, impl.generator_weights, hparams, cond_branch, t_text,
                                              t_cond_audio, t_cond_pos, threads, impl.generator_context != nullptr,
                                              no_probes, cond_logits, output);
            if (status != SYNTH_OK) {
                return status;
            }
        }
        if (guidance != 0.0f) {
            // The unconditional branch carries the target region only -- no
            // style markers, no text, no reference audio. Both CFG branches
            // move together: there is no reason for one to run on the primary
            // backend while the other stays on the CPU, since both read the
            // identical generator weights.
            fill_shifted_audio_ids(canvas.data(), frames, 0, frames, codebooks, vocab, uncond_shifted);
            ggml_backend_tensor_set(t_uncond_audio, uncond_shifted.data(), 0, ggml_nbytes(t_uncond_audio));
            status = generator_branch_forward(*impl.backend_plan, impl.generator_weights, hparams, uncond_branch,
                                              nullptr, t_uncond_audio, t_uncond_pos, threads,
                                              impl.generator_context != nullptr, no_probes, uncond_logits, output);
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
        // --- Phase A (serial): enumerate this step's still-masked positions,
        // and draw everything the step will consume from the stream.
        //
        // The enumeration is the same codebook-major, frame-minor walk it has
        // always been, and it is what fixes both the RNG order and the
        // candidate order; only the SCORING of each enumerated position moves
        // off this thread. `candidates` keeps its capacity across steps, so
        // after step 0 this allocates nothing.
        candidates.clear();
        for (uint32_t codebook = 0; codebook < codebooks; ++codebook) {
            for (uint64_t frame = 0; frame < frames; ++frame) {
                if (canvas[size_t(codebook) * frames + frame] != int32_t(hparams.audio.mask_id)) {
                    continue;  // committed in an earlier step; cannot be revisited
                }
                MaskedCandidate candidate;
                candidate.codebook = codebook;
                candidate.frame    = frame;
                candidates.push_back(candidate);
            }
        }
        if (draws_per_slot != 0) {
            // Candidate i owns [i * draws_per_slot, (i + 1) * draws_per_slot):
            // its class draws first, then its position draw. Filling the whole
            // block in one sequential pass IS the old inline draw order (see
            // the contract above the stream's construction).
            uniforms.resize(candidates.size() * draws_per_slot);
            stream->fill_uniform(uniforms.data(), uniforms.size());
        }

        // --- Phase B (parallel): score each enumerated position. Every
        // iteration reads immutable state (two read-only logit buffers, the
        // hparams, its own pre-drawn uniforms) and writes only its own
        // candidate, so the split cannot reach the result: scoring is a pure
        // function of one position's two logit rows, every reduction inside it
        // is confined to a single 1025-entry row, and no tie-break consults
        // another candidate. Same inputs, same bits, whichever thread runs it.
        std::atomic<int> scoring_status{ int(SYNTH_OK) };
        parallel_for(candidates.size(), threads, [&](size_t begin, size_t end) {
            for (size_t index = begin; index < end; ++index) {
                MaskedCandidate & candidate = candidates[index];
                const size_t      cond_offset =
                    size_t(total - frames + candidate.frame) * row + size_t(candidate.codebook) * vocab;
                const size_t  uncond_offset = size_t(candidate.frame) * row + size_t(candidate.codebook) * vocab;
                const float * uncond_row    = guidance != 0.0f ? uncond_logits.data() + uncond_offset : nullptr;
                float         log_prob      = 0.0f;
                if (class_t > 0.0f) {
                    const synth_status_t chosen = choose_token_sampled(
                        cond_logits.data() + cond_offset, uncond_row, vocab, hparams.audio.mask_id, guidance, class_t,
                        uniforms.data() + index * draws_per_slot, candidate.token, log_prob);
                    if (chosen != SYNTH_OK) {
                        // Unreachable today -- this overload has no failing
                        // path -- so the point is only that a future one would
                        // surface rather than be swallowed by a worker thread.
                        // First writer wins; the rest of the scan still runs
                        // and is then discarded whole by the check below.
                        int unset = int(SYNTH_OK);
                        scoring_status.compare_exchange_strong(unset, int(chosen));
                    }
                } else {
                    choose_token(cond_logits.data() + cond_offset, uncond_row, vocab, hparams.audio.mask_id, guidance,
                                 candidate.token, log_prob, track_margin ? &candidate.argmax_gap : nullptr);
                }
                // The layer penalty biases commitment toward the coarse
                // codebooks first. (audio_codebook_weights is training-loss
                // weighting and plays NO part here -- see the family doc.)
                candidate.score = log_prob - float(candidate.codebook) * hparams.generation.layer_penalty_factor;
                if (pos_t > 0.0f) {
                    // This candidate's own position draw, the last slot of its
                    // own slice -- after its class draws, exactly where the
                    // inline `stream->next_uniform()` used to sit.
                    candidate.score =
                        gumbel_perturb(candidate.score, pos_t, uniforms[index * draws_per_slot + class_draws]);
                }
            }
        });
        if (scoring_status.load() != int(SYNTH_OK)) {
            return synth_status_t(scoring_status.load());
        }

        // --- Phase C (serial): selection and commit, unchanged.
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

    // The generator is finished with, so give its two compute buffers back
    // before the codec asks for its own. Reuse holds both branches live for the
    // whole decode loop where a per-forward rebuild held one at a time; freeing
    // here keeps that raised peak from also overlapping the codec's.
    log_branch_buffers_if_enabled(cond_branch, uncond_branch);
    release_branch(cond_branch, output);
    release_branch(uncond_branch, output);

    status = decode_codes(output.codes, frames, threads, output.audio, &output.codec_seconds, &output.codec_placement);
    if (status != SYNTH_OK) {
        return status;
    }
    // Decision 3 of the plan: the residual output scaling lives HERE, inside
    // the family's synthesis path, faithful to the oracle's own
    // `_post_process_audio` three-arm branch (docs/porting/families/omnivoice.md's
    // "One scaling survives the switches" table). Task 14 makes the choice
    // real: `request.reference_rms` (see its own comment in omnivoice.h) is
    // negative for auto-voice, voice-design, and every caller that predates
    // Plan 3's cloning path -- including the replay seam, which applies the
    // oracle's own branch itself rather than reading this one -- and
    // non-negative only when the public seam (Model::synthesize) built this
    // request from a Reference Audio profile, in which case
    // apply_reference_volume's own `rms >= 0.1 -> none` / `0 < rms < 0.1 ->
    // scale` split governs instead.
    if (request.reference_rms >= 0.0f) {
        apply_reference_volume(output.audio, request.reference_rms);
    } else {
        apply_no_reference_volume(output.audio);
    }
    return SYNTH_OK;
}

synth_status_t Model::synthesize(const PublicSynthesisParams & params, SynthesisOutput & output) {
    output                  = SynthesisOutput{};
    Impl &          impl    = *implementation_;
    const HParams & hparams = impl.hparams;

    // Task 15 threads the free-text instruction through `params.instruct`;
    // this task (14) threads Reference Audio through `params.clone`.
    // `ref_text`/`denoise`/`ref_frames` are the clone-shaped inputs
    // assemble_prompt_ids and DurationEstimator both expect: empty/false/0
    // for auto-voice and voice-design, which is what makes them read as "no
    // reference" and take their own no-reference branches (the anchor pair
    // and the no-reference volume arm respectively) exactly as before this
    // task.
    const bool        has_clone  = params.clone != nullptr;
    const std::string ref_text   = has_clone ? params.clone->transcript_text : std::string();
    const std::string instruct   = params.instruct != nullptr ? *params.instruct : std::string();
    const bool        denoise    = has_clone;  // the clone-only marker
    uint64_t          ref_frames = 0;          // the reference's own frame count, T_ref
    if (has_clone) {
        const uint32_t codebooks = hparams.audio.num_codebooks;
        // A ClonePrompt's reference_tokens is always an exact codebook-major
        // multiple of num_codebooks (Model::encode_reference's own
        // postcondition, ReferenceEncoding::tokens); a profile that
        // disagrees is a wiring defect upstream of this call, not a runtime
        // case a caller can trigger through the public seam (voice-profile.cpp
        // never stores a ClonePrompt that failed encode_reference).
        if (codebooks == 0 || params.clone->reference_tokens.size() % codebooks != 0) {
            return SYNTH_ERR_INTERNAL;
        }
        ref_frames = params.clone->reference_tokens.size() / codebooks;
    }

    // Empty or whitespace-only text is refused before any estimate is made:
    // upstream never runs a synthesis for a request that names no Linguistic
    // Input, and DurationEstimator's own floor (`max(1, int(...))`) would
    // otherwise hand back a one-frame canvas instead of a refusal. Computed
    // once, here, and handed to assemble_prompt_ids below rather than
    // recomputed inside it: reusing combine_text's own stripping keeps this
    // check's idea of "empty" the exact one assemble_prompt_ids applies,
    // rather than a second, possibly divergent, whitespace classifier, and
    // a single join is all either call needs.
    const std::string combined_text = combine_text(ref_text, params.text);
    if (combined_text.empty()) {
        return SYNTH_ERR_INVALID_ARG;
    }

    std::vector<int32_t> prompt_ids;
    const synth_status_t assemble_status = assemble_prompt_ids(
        *impl.frontend, hparams.tokens, denoise, params.language_tag, instruct, combined_text, prompt_ids);
    if (assemble_status != SYNTH_OK) {
        // Propagated verbatim from assemble_prompt_ids: the tokenizer's own
        // status (e.g. SYNTH_ERR_TEXT_FRONTEND for an input byte the
        // package's frontend has no id for), or SYNTH_ERR_INVALID_ARG for the
        // text-end postcondition failure -- see that function's doc comment.
        return assemble_status;
    }
    // docs/c-interface.md: "max_input_tokens is the positive hard limit on
    // the final token sequence consumed by the synthesis graph, after any
    // Text Frontend processing and model-owned special-token insertion ...
    // Exceeding it returns SYNTH_ERR_INPUT_TOO_LONG." This is NOT redundant
    // with the core's own check (synthesis-request.cpp's
    // prepare_synthesis_request, via `info.text_frontend->prepare()`): that
    // check tokenizes and bounds `params.text` ALONE, with none of what this
    // family's own prompt wraps around it -- the clone transcript
    // combine_text folded in above, and assemble_prompt_ids's own style/
    // lang/instruct markers -- so a request could pass the core's generic
    // pre-check yet still assemble a `prompt_ids` well past this package's
    // real ceiling (a long `params.instruct` is the clearest way: it never
    // reaches the core's own tokenization at all, see synthesize.cpp's own
    // comment on why `family_request.text` is read from the raw request
    // bytes directly). `prompt_ids` is what `run_synthesis` actually feeds
    // the graph, so it is the one this family must bound itself --
    // hparams.max_input_tokens is refused at load time when zero
    // (weights.cpp), so this is always a real, positive ceiling here.
    if (prompt_ids.size() > hparams.max_input_tokens) {
        return SYNTH_ERR_INPUT_TOO_LONG;
    }

    const DurationEstimator estimator;
    const uint64_t          estimated =
        estimator.estimate_target_frames(params.text, ref_text, ref_frames, float(params.speaking_rate));

    // The request's own limit if it named one, else the package's declared
    // ceiling -- the same "request limit if nonzero, else package cap" rule
    // the core applies to `effective_frame_limit`, restated here because this
    // family settles its canvas length itself rather than being handed one.
    // `params.max_output_frames` already arrived in codec frames -- converted
    // from the public request's native-PCM-frame limit by src/synthesize.cpp
    // before this call -- while `hparams.max_output_frames` is the package's
    // own ceiling in native PCM frames (docs/c-interface.md). Converting the
    // latter into codec frames before the two are compared is the fix for PR
    // #6's finding: the package once carried a codec-frame count directly in
    // this PCM-frame field, so the unconverted min() silently compared a
    // codec-frame request limit against a value that LOOKED like PCM frames
    // but was secretly already codec frames too -- correct only by that
    // coincidence, and wrong the moment the field held a real PCM value.
    const uint64_t max_output_frames_codec = hparams.max_output_frames / hparams.codec.hop_length;
    const uint64_t effective_limit         = params.max_output_frames != 0 ?
                                                 std::min(params.max_output_frames, max_output_frames_codec) :
                                                 max_output_frames_codec;
    if (estimated > effective_limit) {
        // A cap on the estimate, not a target for it: mirrors upstream's
        // estimator-fixes-canvas semantics rather than silently truncating a
        // request to whatever the limit allows.
        return SYNTH_ERR_OUTPUT_LIMIT;
    }

    SynthesisRequest request;
    request.prompt_text_ids = std::move(prompt_ids);
    request.target_frames   = estimated;
    request.seed            = params.seed;
    request.threads         = params.threads;
    if (has_clone) {
        request.reference_tokens = params.clone->reference_tokens;
        request.reference_rms    = params.clone->ref_rms;
    }
    // position_temperature and class_temperature stay at SynthesisRequest's
    // own -1.0f default: the package's own defaults govern, which for the
    // public path is what makes it sample rather than decode greedily.
    return run_synthesis(request, output);
}

synth_status_t Model::decode_codes(const std::vector<int32_t> &      codes,
                                   uint64_t                          frame_count,
                                   int                               threads,
                                   std::vector<float> &              audio,
                                   double *                          out_seconds,
                                   SynthesisOutput::StagePlacement * out_placement) {
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
    // On the primary backend whenever the decode graph will be, so the graph
    // reads this leaf without a cross-backend copy -- qwen3-tts's own rule for
    // its codec's inputs (Model::decode_codes there).
    if (!inputs.commit(impl.codec_context != nullptr ? impl.backend_plan->primary() :
                                                       impl.backend_plan->cpu_backend())) {
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
    // decode_weights, not weights: the only reader of the twin-bound
    // quantizer/fc2/acoustic_decoder pointers when one exists. impl.weights
    // stays CPU-resident for encode_reference's own rvq_encode call, which
    // must never read through this same graph's binding -- see catalog.h's
    // bind_decode_weights for why the two cannot share one ModelWeights.
    ggml_tensor * wave = build_codec_decoder(run.context(), t_codes, impl.decode_weights, hparams);
    if (wave == nullptr) {
        return SYNTH_ERR_INTERNAL;
    }
    const double started = now_seconds();
    status               = run.run(wave, "omnivoice.codec", threads > 0 ? threads : default_synthesis_threads(),
                                   impl.codec_context != nullptr);
    if (status != SYNTH_OK) {
        return status;
    }
    // Reported directly to the caller rather than through a scratch variable:
    // run_synthesis passes its own output fields, and the replay seam -- the
    // only free-standing caller -- passes its own locals. Before this, a
    // free-standing call's timing and placement were measured and then
    // discarded, which is why the three sampled cases (probe-only, so
    // run_synthesis's own decode_codes call is never reached) reported a
    // fictional zero for both.
    if (out_seconds != nullptr) {
        *out_seconds = now_seconds() - started;
    }
    if (out_placement != nullptr) {
        out_placement->nodes             = run.placed_nodes;
        out_placement->accelerator_nodes = run.accelerator_nodes;
    }
    read_floats(wave, audio);
    if (audio.size() != size_t(frame_count) * hparams.codec.hop_length) {
        std::fprintf(stderr, "omnivoice: the codec produced %zu samples for %llu frames\n", audio.size(),
                     (unsigned long long) frame_count);
        // The owed-length check runs after read_floats already populated
        // `audio` with the wrong-length buffer; every other refusal above
        // returns before audio is ever written, and this path must leave the
        // same cleared-on-error contract rather than hand back stale bytes.
        audio.clear();
        return SYNTH_ERR_INTERNAL;
    }
    return SYNTH_OK;
}

synth_status_t Model::encode_reference(const std::vector<float> & pcm_24k, int threads, ReferenceEncoding & output) {
    output                  = ReferenceEncoding{};
    Impl &          impl    = *implementation_;
    const HParams & hparams = impl.hparams;

    if (pcm_24k.empty()) {
        return SYNTH_ERR_INVALID_ARG;
    }

    // Step 1: ref_rms (measured on the FULL, un-clipped `pcm_24k`), the
    // quiet-reference boost, THEN the hop-clip (tail-clip to a whole number
    // of hop_length-sample frames) -- all in place, upstream's own order;
    // see clip_and_boost_reference's own header comment for the line-by-line
    // citation.
    std::vector<float> clipped = pcm_24k;
    clip_and_boost_reference(clipped, hparams.codec.hop_length, output.ref_rms);
    if (clipped.empty()) {
        return SYNTH_ERR_INVALID_ARG;
    }

    // Step 2: resample to 16 kHz for the semantic branch.
    std::vector<float> pcm_16k;
    if (!resample_24k_to_16k(clipped, pcm_16k)) {
        return SYNTH_ERR_INVALID_ARG;
    }
    const int resolved_threads = threads > 0 ? threads : default_synthesis_threads();

    // Step 3: the HuBERT semantic branch. The fusion's own semantic-side
    // input is build_semantic_branch's PRIMARY return value
    // (semantic_encoder_output), not semantic_mean, so this always asks for
    // it regardless of whether a caller reads output.semantic_encoder_output
    // back.
    std::vector<float> semantic_encoder_output;
    synth_status_t status = run_semantic_branch(*impl.backend_plan, impl.weights, hparams, pcm_16k, resolved_threads,
                                                output.semantic_mean, &semantic_encoder_output);
    if (status != SYNTH_OK) {
        return status;
    }

    // Step 4: the DAC acoustic branch plus reference fusion, over the
    // clipped+boosted 24 kHz segment -- NOT the original `pcm_24k` -- so the
    // acoustic and semantic branches see the identical waveform samples the
    // RVQ encode below is a function of.
    status = run_acoustic_and_fuse(*impl.backend_plan, impl.weights, hparams, clipped, semantic_encoder_output,
                                   resolved_threads, output.fused_latent);
    if (status != SYNTH_OK) {
        return status;
    }
    output.pcm_16k                 = std::move(pcm_16k);
    output.semantic_encoder_output = std::move(semantic_encoder_output);

    // Step 5: RVQ nearest-neighbour encode. `concat` is the RVQ's own input
    // width (codec.hidden_size + semantic.hidden_size, catalog.cpp's own
    // concat_width -- duplicated here as a one-line formula rather than
    // exposed from that file's anonymous namespace) and divides
    // fused_latent's own size exactly for any hop-aligned input, matching
    // build_reference_fusion's own contract.
    const uint64_t concat = uint64_t(hparams.codec.hidden_size) + hparams.semantic.hidden_size;
    if (concat == 0 || output.fused_latent.size() % concat != 0) {
        // Same cleared-on-error contract as decode_codes: an early return must
        // not hand back the ref_rms/pcm_16k/semantic_mean/semantic_encoder_output/
        // fused_latent fields Steps 1-4 already populated.
        output = ReferenceEncoding{};
        return SYNTH_ERR_INTERNAL;
    }
    output.frames = output.fused_latent.size() / concat;

    if (!rvq_encode(impl.weights.quantizers, output.fused_latent, output.frames, output.tokens, &output.narrowest_gap,
                    &output.gaps)) {
        output = ReferenceEncoding{};
        return SYNTH_ERR_INTERNAL;
    }
    output.margin_measured = true;
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
        // Twins of the codec's decode path and of the generator (Plan 5 Task
        // 1), so both can run on the primary backend while the clone-encode
        // chain stays on the CPU. Declared before binding, because
        // bind_decode_weights/bind_generator_weights below bind each path
        // against its own twin when present. The codec twin is filtered to
        // is_decode_path_tensor's three prefixes rather than every `codec.`
        // tensor -- see this file's placement comment and catalog.h's
        // bind_decode_weights for why the rest of the codec (334 tensors,
        // 616.01 MiB in this checkpoint's F32 GGUF) must stay off it. The
        // generator twin has no such narrowing: is_generator_tensor covers the
        // whole group, because catalog.h's bind_generator_weights has no
        // NOT-MOVABLE remainder to carve out.
        const bool split = implementation->backend_plan->primary() != implementation->backend_plan->cpu_backend();
        if (split) {
            ggml_init_params twin_params{};
            twin_params.mem_size          = ggml_tensor_overhead() * 256;
            twin_params.no_alloc          = true;
            implementation->codec_context = ggml_init(twin_params);
            if (implementation->codec_context == nullptr) {
                return SYNTH_ERR_OOM;
            }
            for (ggml_tensor * tensor = ggml_get_first_tensor(implementation->weights_context); tensor != nullptr;
                 tensor               = ggml_get_next_tensor(implementation->weights_context, tensor)) {
                if (!is_decode_path_tensor(tensor->name)) {
                    continue;
                }
                ggml_tensor * twin =
                    ggml_new_tensor(implementation->codec_context, tensor->type, ggml_n_dims(tensor), tensor->ne);
                if (twin == nullptr) {
                    return SYNTH_ERR_OOM;
                }
                ggml_set_name(twin, tensor->name);
            }
            // 312 generator tensors (catalog.h's bind_generator_weights own
            // count); 384 leaves the same proportional headroom the codec
            // twin's 256-for-152 sizing above does.
            ggml_init_params generator_twin_params{};
            generator_twin_params.mem_size    = ggml_tensor_overhead() * 384;
            generator_twin_params.no_alloc    = true;
            implementation->generator_context = ggml_init(generator_twin_params);
            if (implementation->generator_context == nullptr) {
                return SYNTH_ERR_OOM;
            }
            for (ggml_tensor * tensor = ggml_get_first_tensor(implementation->weights_context); tensor != nullptr;
                 tensor               = ggml_get_next_tensor(implementation->weights_context, tensor)) {
                if (!is_generator_tensor(tensor->name)) {
                    continue;
                }
                ggml_tensor * twin =
                    ggml_new_tensor(implementation->generator_context, tensor->type, ggml_n_dims(tensor), tensor->ne);
                if (twin == nullptr) {
                    return SYNTH_ERR_OOM;
                }
                ggml_set_name(twin, tensor->name);
            }
        }
        // `weights` is bound against the package alone -- ALWAYS, whether or
        // not either twin exists -- so the host clone-encode chain
        // (rvq_encode, reached through encode_reference) can never be handed a
        // twin pointer. `decode_weights` and `generator_weights` are the
        // separate, twin-aware bindings only Model::decode_codes and
        // generator_branch_forward (via Model::run_synthesis) read
        // respectively; see catalog.h's own header comments on
        // bind_decode_weights/bind_generator_weights for the second-consumer
        // trap this split exists to close.
        status = build_model_weights(implementation->weights_context, implementation->hparams, implementation->weights);
        if (status != SYNTH_OK) {
            return status;
        }
        status = bind_decode_weights(implementation->codec_context, implementation->hparams, implementation->weights,
                                     implementation->decode_weights);
        if (status != SYNTH_OK) {
            return status;
        }
        status = bind_generator_weights(implementation->generator_context, implementation->hparams,
                                        implementation->weights, implementation->generator_weights);
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

        // Every graph reads the package's own tensors when there is no twin,
        // so the weights live in the CPU backend's buffer regardless of the
        // primary; see the placement note at the top of this file.
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
        if (implementation->generator_context != nullptr) {
            implementation->generator_buffer = ggml_backend_alloc_ctx_tensors(implementation->generator_context,
                                                                              implementation->backend_plan->primary());
            if (implementation->generator_buffer == nullptr) {
                return SYNTH_ERR_OOM;
            }
            ggml_backend_buffer_set_usage(implementation->generator_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            // Copy after streaming, same as the codec twin above.
            std::vector<unsigned char> scratch;
            for (ggml_tensor * twin = ggml_get_first_tensor(implementation->generator_context); twin != nullptr;
                 twin               = ggml_get_next_tensor(implementation->generator_context, twin)) {
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

}  // namespace synth::omnivoice
