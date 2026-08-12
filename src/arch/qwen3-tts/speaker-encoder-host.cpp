// The host wrapper around the ECAPA-TDNN speaker encoder graph: reference PCM
// in (already at the package's own declared reference rate), an enc_dim-wide
// x-vector out, one shot on the CPU.
//
// Structurally this is OmniVoice's own run_semantic_branch/run_acoustic_and_fuse
// (src/arch/omnivoice/reference-encoder-host.cpp) with one deliberate
// difference: this function owns its BackendPlan rather than receiving one.
// Those two are called from inside Model::encode_reference, which already
// holds impl.backend_plan; this one's signature (HParams, weights, pcm,
// threads -- no BackendPlan) is what Task 7's Voice Profile preparation and
// this file's own header comment settle on, and a CPU-only plan built here is
// exactly the plan Model::load would have handed over: primary device CPU, no
// accelerators, matching the catalog's own "stays on the CPU" ruling.

#include "speaker-encoder-host.h"

#include "backend-plan.h"
#include "cpu-parallelism.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "mel.h"
#include "speaker-encoder.h"
#include "weights.h"

#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>

namespace synth::qwen3tts {

namespace {

// The encoder builds 435 nodes with F32 weights and 473 with BF16. Three
// things fix that count and none of them is the mel: the res2net scale, the
// block count, and the weights' own storage dtype. The 38 extra nodes are
// add_channel_bias's dtype-conditional ggml_cast (speaker-encoder.cpp), one
// per convolution, inserted only when the bias is not already F32.
//
// 473 is the number that matters here: every one of the real Base package's
// 76 speaker_encoder tensors is BF16, so 473 is what this host actually
// builds, and 435 is what tests/qwen3_tts_speaker_encoder_test.cpp's own
// synthetic F32 fixture builds. Both are pinned exactly by that test. This
// budget is a constant for a given package, and 4096 keeps the same
// order-of-magnitude headroom over the larger of the two.
constexpr size_t kSpeakerGraphNodeBudget = 4096;

// Not upstream's own quantity: there is nothing named "ref_rms" in the
// reference speaker path, unlike OmniVoice's clip_and_boost_reference, which
// has to reproduce omnivoice.py's `sqrt(mean(ref_wav**2))` bit for bit
// (reference-encoder-host.cpp's own reference_rms). This is purely this
// port's own silent-clip gate -- see XVectorEncoding::ref_rms's doc comment
// and the reference_silent refusal below -- so a plain double accumulation is
// enough: the only decision it drives is ref_rms == 0.0f, and no summation
// order changes whether every sample was exactly zero to begin with.
float reference_rms(const std::vector<float> & pcm) {
    if (pcm.empty()) {
        return 0.0f;
    }
    double sum = 0.0;
    for (float sample : pcm) {
        sum += double(sample) * double(sample);
    }
    return float(std::sqrt(sum / double(pcm.size())));
}

// One-shot scratch for the graph run. RAII because there are several
// early-return points below and every one of them must free whatever was
// already opened -- the same shape as model.cpp's own Persistent/GraphRun,
// duplicated here rather than shared because those two are private to
// model.cpp and this function does not have a BackendPlan handed to it the
// way every caller of those does.
struct GraphScratch {
    std::unique_ptr<BackendPlan> plan;
    ggml_context *               input_context = nullptr;
    ggml_backend_buffer_t        input_buffer  = nullptr;
    ggml_context *               graph_context = nullptr;
    ggml_backend_sched_t         scheduler     = nullptr;

    GraphScratch()                                 = default;
    GraphScratch(const GraphScratch &)             = delete;
    GraphScratch & operator=(const GraphScratch &) = delete;

    ~GraphScratch() {
        if (scheduler != nullptr) {
            ggml_backend_sched_free(scheduler);
        }
        if (graph_context != nullptr) {
            ggml_free(graph_context);
        }
        if (input_buffer != nullptr) {
            ggml_backend_buffer_free(input_buffer);
        }
        if (input_context != nullptr) {
            ggml_free(input_context);
        }
    }
};

}  // namespace

synth_status_t encode_speaker_reference(const HParams &               hparams,
                                        const SpeakerEncoderWeights & weights,
                                        const std::vector<float> &    pcm,
                                        int                           threads,
                                        XVectorEncoding &             output,
                                        const char *&                 out_diagnostic_code,
                                        const char *&                 out_diagnostic_message) {
    output                 = XVectorEncoding{};
    out_diagnostic_code    = nullptr;
    out_diagnostic_message = nullptr;

    if (pcm.empty()) {
        return SYNTH_ERR_INVALID_ARG;
    }

    // jiangzhuo's ruling for this family, 2026-08-12 (mirroring OmniVoice's
    // own silent-reference ruling, src/arch/omnivoice/profile.cpp:42-55, and
    // reusing its diagnostic code so the two families answer alike): a
    // digitally silent reference still produces a finite, stable x-vector --
    // attentive statistics pooling over an all-zero mel does not diverge --
    // and that embedding is meaningless, not a cloned voice. Checked here,
    // inside the encode call itself, rather than one layer up the way
    // OmniVoice's create_clone_prompt does it: this function is the only
    // speaker-path seam Plan 2 has, and there is no separate
    // profile-creation call above it yet to do the check instead.
    output.ref_rms = reference_rms(pcm);
    if (output.ref_rms == 0.0f) {
        out_diagnostic_code    = "voice_profile.reference_silent";
        out_diagnostic_message = "the reference audio is digitally silent; nothing can be cloned from it";
        return SYNTH_ERR_INVALID_ARG;
    }

    MelSpectrogram mel;
    synth_status_t status = compute_log_mel(hparams.speaker_encoder, pcm, mel);
    if (status != SYNTH_OK) {
        return status;
    }

    GraphScratch scratch;
    // CPU only, no accelerators: the header comment above records why this
    // stays here rather than moving with the codec.
    status = BackendPlan::create(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), false, scratch.plan);
    if (status != SYNTH_OK) {
        return status;
    }

    // The mel input tensor lives in its own tiny persistent buffer, the same
    // shape reference-encoder-host.cpp's own input tensors do: the graph
    // allocator below must not own it.
    ggml_init_params input_params{};
    input_params.mem_size = ggml_tensor_overhead() * 4;
    input_params.no_alloc = true;
    scratch.input_context = ggml_init(input_params);
    if (scratch.input_context == nullptr) {
        return SYNTH_ERR_OOM;
    }
    // [bins, frames]: ggml lists the fastest-varying axis first, and mel.h's
    // own layout comment records that MelSpectrogram::values is exactly this
    // order already -- no transpose needed between here and the graph.
    ggml_tensor * mel_tensor =
        ggml_new_tensor_2d(scratch.input_context, GGML_TYPE_F32, int64_t(mel.bins), int64_t(mel.frames));
    scratch.input_buffer = ggml_backend_alloc_ctx_tensors(scratch.input_context, scratch.plan->cpu_backend());
    if (scratch.input_buffer == nullptr) {
        return SYNTH_ERR_OOM;
    }
    ggml_backend_tensor_set(mel_tensor, mel.values.data(), 0, ggml_nbytes(mel_tensor));

    ggml_init_params graph_params{};
    graph_params.mem_size = ggml_tensor_overhead() * (kSpeakerGraphNodeBudget + 256) +
                            ggml_graph_overhead_custom(kSpeakerGraphNodeBudget, false);
    graph_params.no_alloc = true;
    scratch.graph_context = ggml_init(graph_params);
    if (scratch.graph_context == nullptr) {
        return SYNTH_ERR_OOM;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(scratch.graph_context, kSpeakerGraphNodeBudget, false);

    ggml_tensor * embedding = build_speaker_encoder(scratch.graph_context, weights, mel_tensor);
    if (embedding == nullptr) {
        return SYNTH_ERR_INTERNAL;
    }
    ggml_build_forward_expand(graph, embedding);

    // A single-backend scheduler over CPU-resident weights is one split --
    // BackendPlan::create_cpu_scheduler's own reasoning, which applies here
    // even though this plan never carries a second backend at all.
    const size_t hash_size = size_t(ggml_graph_size(graph)) + 4096;
    scratch.scheduler      = scratch.plan->create_cpu_scheduler(hash_size);
    if (scratch.scheduler == nullptr) {
        return SYNTH_ERR_BACKEND;
    }
    if (!ggml_backend_sched_alloc_graph(scratch.scheduler, graph)) {
        return SYNTH_ERR_OOM;
    }
    scratch.plan->set_threads(threads > 0 ? threads : default_synthesis_threads());
    if (ggml_backend_sched_graph_compute(scratch.scheduler, graph) != GGML_STATUS_SUCCESS) {
        return SYNTH_ERR_BACKEND;
    }

    std::vector<float> embedding_values(size_t(ggml_nelements(embedding)));
    ggml_backend_tensor_get(embedding, embedding_values.data(), 0, ggml_nbytes(embedding));

    // Carryover Section 3 Task 2(a): unlike the shape[1] == 16 checks on
    // codes, nothing runtime-asserted that the speaker embedding is exactly
    // enc_dim elements. Checked here rather than trusted from the graph,
    // because a package whose metadata and weights silently disagree on
    // enc_dim would otherwise hand back a wrong-width "embedding" that still
    // looks like a plain vector of floats to every caller downstream.
    if (embedding_values.size() != size_t(hparams.speaker_encoder.enc_dim)) {
        return SYNTH_ERR_INTERNAL;
    }
    for (float value : embedding_values) {
        if (!std::isfinite(value)) {
            return SYNTH_ERR_INTERNAL;
        }
    }

    output.x_vector   = std::move(embedding_values);
    output.mel_frames = mel.frames;
    return SYNTH_OK;
}

}  // namespace synth::qwen3tts
