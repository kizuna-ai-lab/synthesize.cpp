// The host wrapper around the codec encoder: reference PCM in (already at the
// package's own declared reference rate), the [16, T] reference code grid out,
// one shot on the CPU.
//
// Structurally this is speaker-encoder-host.cpp with a second half. The graph
// part is the same shape -- own the BackendPlan rather than receive one, CPU
// only, one scheduler, one compute -- for the same reason that file records:
// this encoder's tensors are bound against the package context and never a
// twin, so they are CPU-resident, and Voice Profile preparation has no
// BackendPlan to hand down.
//
// The second half is the split residual vector quantizer, and it is HOST code
// because its output is discrete. See codec-encoder-host.h for the placement
// rule, the tie rule and the accumulation decision; what follows here is the
// transcription, with the upstream line each step came from.

#include "codec-encoder-host.h"

#include "backend-plan.h"
#include "catalog.h"
#include "codec-encoder.h"
#include "cpu-parallelism.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "weights.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>

namespace synth::qwen3tts {

namespace {

// The real Base package builds 445 nodes on a whole-frame clip and 449 on a
// ragged one (tests/qwen3_tts_codec_encoder_test.cpp pins both, and records why
// the graph is not length-independent: the frame downsampler's replicate
// right-pad costs four nodes exactly when the clip is not a whole number of
// frames). 8192 is the same headroom the driver already uses over that.
constexpr size_t kCodecEncoderGraphNodeBudget = 8192;

// Not upstream's own quantity -- there is nothing named "ref_rms" on the
// reference codec path. This is purely this port's silent-clip gate, so a plain
// double accumulation is enough: the only decision it drives is
// `ref_rms == 0.0f`, and no summation order changes whether every sample was
// exactly zero to begin with. Same function, same reasoning, as
// speaker-encoder-host.cpp's own.
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
// already opened -- speaker-encoder-host.cpp's own GraphScratch, duplicated
// there from model.cpp for the same reason it is duplicated here: those types
// are private to their own translation units.
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

// A weight tensor's values as F32, whatever it is stored as.
//
// Not a raw ggml_backend_tensor_get into a float buffer, which is what
// omnivoice::rvq_encode does under an explicit F32-only refusal: this family's
// graph half already widens with `as_f32` (codec-encoder.cpp), so a package
// storing this half at BF16 runs the convolutions fine and would then have its
// codebook bytes reinterpreted as float here. All 161 `codec.encoder.*` tensors
// in the real Base package are F32 today, which is exactly the state
// speaker-encoder.cpp's weights were assumed to be in before the package proved
// otherwise and aborted.
bool pull_f32(const ggml_tensor * tensor, std::vector<float> & values) {
    if (tensor == nullptr || !ggml_is_contiguous(tensor)) {
        return false;
    }
    const int64_t count = ggml_nelements(tensor);
    if (count <= 0) {
        return false;
    }
    values.assign(size_t(count), 0.0f);
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
        return true;
    }
    const ggml_type_traits * traits = ggml_get_type_traits(tensor->type);
    if (traits == nullptr || traits->to_float == nullptr) {
        return false;
    }
    std::vector<uint8_t> raw(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size());
    traits->to_float(raw.data(), values.data(), count);
    return true;
}

// Every shape one branch's cascade needs, checked before a single tensor is
// pulled to host -- omnivoice::rvq_encode's own "check every level before
// touching data" rule, which is what keeps a half-resolved package from
// producing a wild read rather than a refusal.
bool branch_shapes_agree(const CodecQuantizerWeights & quantizer,
                         int64_t                       stages,
                         int64_t                       latent_width,
                         int64_t                       projected,
                         int64_t                       codebook_size) {
    if (stages <= 0 || int64_t(quantizer.codebooks.size()) < stages) {
        return false;
    }
    const ggml_tensor * input_proj = quantizer.input_proj;
    // A kernel-one convolution, no bias: ggml reports [kernel, in, out], so the
    // three extents are exactly 1, the latent width, and the projected width.
    if (input_proj == nullptr || !ggml_is_contiguous(input_proj) || input_proj->ne[0] != 1 ||
        input_proj->ne[1] != latent_width || input_proj->ne[2] != projected) {
        return false;
    }
    for (int64_t stage = 0; stage < stages; ++stage) {
        const ggml_tensor * codebook = quantizer.codebooks[size_t(stage)];
        if (codebook == nullptr || !ggml_is_contiguous(codebook) || codebook->ne[0] != projected ||
            codebook->ne[1] != codebook_size || codebook_size < 2) {
            return false;
        }
    }
    return true;
}

// One branch of the split quantizer, start to finish.
//
// `latents` is [latent_width, frames] channel-fastest; `first_group` is where
// this branch's stages land in the [16, T] grid (0 for semantic, 1 for
// acoustic). The residual runs in the PROJECTED space and the branch's own
// `input_proj` is applied ONCE, before the cascade -- upstream's
// MimiResidualVectorQuantizer.encode projects and then loops, rather than
// projecting per stage.
//
// `output_proj` is never touched: it is decode-only
// (MimiResidualVectorQuantizer.decode applies it after summing), which is why
// the reconstruction this leaves behind is in the projected space and not the
// latent one.
bool quantize_branch(const CodecQuantizerWeights & quantizer,
                     const std::vector<float> &    latents,
                     int64_t                       latent_width,
                     int64_t                       projected,
                     int64_t                       frames,
                     int64_t                       stages,
                     int64_t                       first_group,
                     int64_t                       groups,
                     float *                       branch_reconstruction,
                     CodecEncoding &               output,
                     float &                       narrowest) {
    std::vector<float> input_proj;
    if (!pull_f32(quantizer.input_proj, input_proj)) {
        return false;
    }

    // residual = input_proj(latents), [frames][projected]. The weight is
    // [1, latent_width, projected], so output channel `o`'s row of
    // `latent_width` values starts at `o * latent_width`. No bias: upstream
    // constructs both projections with `bias=False`.
    std::vector<float> residual(size_t(frames) * size_t(projected), 0.0f);
    for (int64_t frame = 0; frame < frames; ++frame) {
        const float * latent = latents.data() + size_t(frame) * size_t(latent_width);
        float *       target = residual.data() + size_t(frame) * size_t(projected);
        for (int64_t out = 0; out < projected; ++out) {
            const float * row         = input_proj.data() + size_t(out) * size_t(latent_width);
            float         accumulator = 0.0f;
            for (int64_t in = 0; in < latent_width; ++in) {
                accumulator += row[in] * latent[in];
            }
            target[size_t(out)] = accumulator;
        }
    }

    std::vector<float> codebook;
    std::vector<float> embed_sq;
    for (int64_t stage = 0; stage < stages; ++stage) {
        const int64_t group = first_group + stage;
        if (!pull_f32(quantizer.codebooks[size_t(stage)], codebook)) {
            return false;
        }
        const int64_t codebook_size = quantizer.codebooks[size_t(stage)]->ne[1];

        // The residual ENTERING this stage, before anything is subtracted --
        // the oracle captures `hidden_states` at the same point, on the way
        // into `quantize`.
        std::copy(residual.begin(), residual.end(),
                  output.residuals.begin() + std::ptrdiff_t(size_t(group) * residual.size()));

        // ||E[e]||^2 depends only on the codebook, so it is computed once per
        // stage rather than once per (stage, frame) -- upstream likewise
        // computes `y_norm` once and broadcasts it over every row.
        embed_sq.assign(size_t(codebook_size), 0.0f);
        for (int64_t code = 0; code < codebook_size; ++code) {
            const float * row = codebook.data() + size_t(code) * size_t(projected);
            float         sum = 0.0f;
            for (int64_t index = 0; index < projected; ++index) {
                sum += row[index] * row[index];
            }
            embed_sq[size_t(code)] = sum;
        }

        for (int64_t frame = 0; frame < frames; ++frame) {
            float * z = residual.data() + size_t(frame) * size_t(projected);

            float scaled = 0.0f;
            for (int64_t index = 0; index < projected; ++index) {
                scaled += z[index] * z[index];
            }

            // Squared Euclidean nearest neighbour, in the SAME expanded form
            // and the SAME grouping torch's own `cdist` uses above 25 rows:
            // `x_norm + y_norm - 2 * x @ y.T`. Deliberately not
            // omnivoice::rvq_encode's `scaled - 2*dot + embed_sq` grouping,
            // which is the same value in exact arithmetic and a different one
            // in float32; matching upstream's grouping is the point.
            //
            // Not clamped at zero the way torch clamps before its square root.
            // The clamp is monotone, so it cannot change an argmin unless two
            // candidates are both negative, and the smallest distance anywhere
            // in the real cases is around 2.6e4.
            //
            // Strict `<` scanning `code` ascending IS the tie rule: a later
            // equal candidate never displaces an earlier one, so the lowest id
            // wins, matching `dists.argmin(dim=-1)`.
            int64_t best_code   = -1;
            float   best_dist   = std::numeric_limits<float>::infinity();
            float   second_dist = std::numeric_limits<float>::infinity();
            for (int64_t code = 0; code < codebook_size; ++code) {
                const float * row = codebook.data() + size_t(code) * size_t(projected);
                float         dot = 0.0f;
                for (int64_t index = 0; index < projected; ++index) {
                    dot += z[index] * row[index];
                }
                const float distance = (scaled + embed_sq[size_t(code)]) - 2.0f * dot;
                if (distance < best_dist) {
                    second_dist = best_dist;
                    best_dist   = distance;
                    best_code   = code;
                } else if (distance < second_dist) {
                    second_dist = distance;
                }
            }

            // `best_code` stays -1 only when every comparison above was false,
            // which a NaN residual makes possible: a NaN loses every `<`,
            // including against another NaN. `size_t(best_code)` would then
            // wrap and the codebook read below would be a wild pointer, so this
            // is refused as a failure rather than asserted. The latents are
            // already checked finite before this runs; this is the backstop for
            // a NaN that appears in the projection itself.
            if (best_code < 0) {
                return false;
            }

            const size_t slot  = size_t(frame) * size_t(groups) + size_t(group);
            output.codes[slot] = int32_t(best_code);
            const float gap    = second_dist - best_dist;
            output.gaps[slot]  = gap;
            narrowest          = std::fmin(narrowest, gap);

            // The dequantized row, in the projected space: added to this
            // branch's running reconstruction and subtracted from the residual
            // the next stage reads. Upstream's `residual = residual -
            // quantized`, where `quantized` is the raw codebook row.
            const float * best_row      = codebook.data() + size_t(best_code) * size_t(projected);
            float *       reconstructed = branch_reconstruction + size_t(frame) * size_t(projected);
            for (int64_t index = 0; index < projected; ++index) {
                reconstructed[index] += best_row[index];
                z[index] -= best_row[index];
            }
        }
    }
    return true;
}

}  // namespace

bool codes_equal(const CodecEncoding & encoding, const int32_t * flat, int64_t frames) {
    if (flat == nullptr || encoding.groups == 0 || frames < 0 || uint64_t(frames) != encoding.frames) {
        return false;
    }
    const size_t expected = size_t(encoding.groups) * size_t(encoding.frames);
    if (encoding.codes.size() != expected) {
        return false;
    }
    for (size_t index = 0; index < expected; ++index) {
        if (encoding.codes[index] != flat[index]) {
            return false;
        }
    }
    return true;
}

synth_status_t encode_codec_reference(const HParams &             hparams,
                                      const CodecEncoderWeights & weights,
                                      const std::vector<float> &  pcm,
                                      int                         threads,
                                      CodecEncoding &             output,
                                      const char *&               out_diagnostic_code,
                                      const char *&               out_diagnostic_message) {
    output                 = CodecEncoding{};
    out_diagnostic_code    = nullptr;
    out_diagnostic_message = nullptr;

    if (pcm.empty()) {
        return SYNTH_ERR_INVALID_ARG;
    }

    // The same ruling, the same diagnostic code, as the speaker path
    // (speaker-encoder-host.cpp): a digitally silent reference still produces
    // finite, stable codes, and they are meaningless rather than a cloned
    // voice. Checked before the graph rather than after, because there is
    // nothing downstream that would notice.
    output.ref_rms = reference_rms(pcm);
    if (output.ref_rms == 0.0f) {
        out_diagnostic_code    = "voice_profile.reference_silent";
        out_diagnostic_message = "the reference audio is digitally silent; nothing can be cloned from it";
        return SYNTH_ERR_INVALID_ARG;
    }

    // Refuses a sub-frame clip and a non-finite sample, which a graph builder
    // cannot see: it reads shapes, never values.
    CodecEncoderGeometry geometry;
    synth_status_t       status = codec_encoder_check_waveform(weights, pcm.data(), pcm.size(), geometry);
    if (status != SYNTH_OK) {
        return status;
    }

    const CodecDecoderParams & params        = hparams.codec.decoder;
    const int64_t              groups        = int64_t(params.quantizer_count);
    const int64_t              semantic      = int64_t(params.semantic_quantizer_count);
    const int64_t              latent_width  = int64_t(params.codebook_dim);
    const int64_t              projected     = latent_width / 2;
    const int64_t              codebook_size = int64_t(params.codebook_size);
    if (groups <= 0 || semantic <= 0 || semantic >= groups || latent_width <= 0 || latent_width % 2 != 0 ||
        codebook_size < 2) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (!branch_shapes_agree(weights.semantic, semantic, latent_width, projected, codebook_size) ||
        !branch_shapes_agree(weights.acoustic, groups - semantic, latent_width, projected, codebook_size)) {
        return SYNTH_ERR_INVALID_ARG;
    }

    GraphScratch scratch;
    status = BackendPlan::create(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), false, scratch.plan);
    if (status != SYNTH_OK) {
        return status;
    }

    ggml_init_params input_params{};
    input_params.mem_size = ggml_tensor_overhead() * 8;
    input_params.no_alloc = true;
    scratch.input_context = ggml_init(input_params);
    if (scratch.input_context == nullptr) {
        return SYNTH_ERR_OOM;
    }
    // [1, samples]: this family's layout is channel-major, so a mono clip is
    // one channel of `samples` -- build_codec_encoder_seanet refuses a bare 1-D
    // [samples] tensor rather than transposing on the caller's behalf.
    ggml_tensor * waveform  = ggml_new_tensor_2d(scratch.input_context, GGML_TYPE_F32, 1, int64_t(pcm.size()));
    // The transformer runs BEFORE the frame downsampler, so its positions are
    // the SEANet stack's output length and not the frame count. Taken from the
    // geometry rather than recomputed.
    ggml_tensor * positions = ggml_new_tensor_1d(scratch.input_context, GGML_TYPE_I32, geometry.transformer_positions);
    scratch.input_buffer    = ggml_backend_alloc_ctx_tensors(scratch.input_context, scratch.plan->cpu_backend());
    if (scratch.input_buffer == nullptr) {
        return SYNTH_ERR_OOM;
    }
    ggml_backend_tensor_set(waveform, pcm.data(), 0, ggml_nbytes(waveform));
    std::vector<int32_t> sequential(size_t(geometry.transformer_positions), 0);
    for (int64_t index = 0; index < geometry.transformer_positions; ++index) {
        sequential[size_t(index)] = int32_t(index);
    }
    ggml_backend_tensor_set(positions, sequential.data(), 0, ggml_nbytes(positions));

    ggml_init_params graph_params{};
    graph_params.mem_size = ggml_tensor_overhead() * (kCodecEncoderGraphNodeBudget + 256) +
                            ggml_graph_overhead_custom(kCodecEncoderGraphNodeBudget, false);
    graph_params.no_alloc = true;
    scratch.graph_context = ggml_init(graph_params);
    if (scratch.graph_context == nullptr) {
        return SYNTH_ERR_OOM;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(scratch.graph_context, kCodecEncoderGraphNodeBudget, false);

    ggml_tensor * latents = build_codec_encoder(scratch.graph_context, waveform, positions, weights);
    if (latents == nullptr) {
        return SYNTH_ERR_INTERNAL;
    }
    if (latents->ne[0] != latent_width) {
        return SYNTH_ERR_INTERNAL;
    }
    ggml_build_forward_expand(graph, latents);

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

    // THE FRAME TRIM, applied AFTER the graph -- Qwen3TTSTokenizerV2Model.encode
    // (modeling_qwen3_tts_tokenizer_v2.py:961-988) slices the encoder's output
    // to `-(-mask.sum() // encode_downsample_rate)`, a ceiling divide by the
    // product of the strides. Composing the per-convolution ceilings gives the
    // same number, so on every clip this port has met the trim removes nothing
    // -- which is a property to CHECK rather than one to assume, since a graph
    // that ever produced more frames must be cut here and not downstream.
    const int64_t graph_frames = latents->ne[1];
    const int64_t trim_target  = (geometry.samples + geometry.samples_per_frame - 1) / geometry.samples_per_frame;
    const int64_t frames       = std::min(graph_frames, trim_target);
    if (frames <= 0 || graph_frames < frames) {
        return SYNTH_ERR_INTERNAL;
    }

    std::vector<float> latent_values(size_t(ggml_nelements(latents)));
    ggml_backend_tensor_get(latents, latent_values.data(), 0, ggml_nbytes(latents));
    latent_values.resize(size_t(frames) * size_t(latent_width));
    for (float value : latent_values) {
        if (!std::isfinite(value)) {
            return SYNTH_ERR_INTERNAL;
        }
    }

    output.groups       = uint64_t(groups);
    output.frames       = uint64_t(frames);
    output.latent_width = uint64_t(latent_width);
    output.projected    = uint64_t(projected);
    output.codes.assign(size_t(groups) * size_t(frames), 0);
    output.gaps.assign(size_t(groups) * size_t(frames), 0.0f);
    output.residuals.assign(size_t(groups) * size_t(frames) * size_t(projected), 0.0f);
    output.reconstruction.assign(2u * size_t(frames) * size_t(projected), 0.0f);
    output.latents = std::move(latent_values);

    // Both branches are fed the SAME latents. The acoustic cascade does NOT
    // start from the semantic branch's leftover -- MimiSplitResidualVectorQuantizer
    // .encode passes `embeddings` to each of them -- and a port that chains them
    // produces sixteen valid-looking codes and a different voice.
    float narrowest = std::numeric_limits<float>::infinity();
    if (!quantize_branch(weights.semantic, output.latents, latent_width, projected, frames, semantic, 0, groups,
                         output.reconstruction.data(), output, narrowest) ||
        !quantize_branch(weights.acoustic, output.latents, latent_width, projected, frames, groups - semantic, semantic,
                         groups, output.reconstruction.data() + size_t(frames) * size_t(projected), output,
                         narrowest)) {
        output = CodecEncoding{};
        return SYNTH_ERR_INTERNAL;
    }
    output.narrowest_gap = std::isfinite(narrowest) ? narrowest : 0.0f;

    // Structural, and strict: every code must index a real codebook row. This
    // is one of the two things the fourth erratum left gating -- the codes'
    // VALUES are no longer compared against the oracle, but their range and
    // their count still are.
    for (int32_t code : output.codes) {
        if (code < 0 || code >= int32_t(codebook_size)) {
            output = CodecEncoding{};
            return SYNTH_ERR_INTERNAL;
        }
    }
    return SYNTH_OK;
}

}  // namespace synth::qwen3tts
