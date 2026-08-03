#include "arch/omnivoice/reference-encoder-host.h"

#include "arch/omnivoice/reference-encoder.h"
#include "backend-plan.h"
#include "cpu-parallelism.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace synth::omnivoice {

namespace {

// The fixed 24000 -> 16000 pair, reduced by gcd(24000, 16000) = 8000. See the
// header for the citations behind every constant and formula here.
constexpr int    kOrigFreq           = 3;  // 24000 / 8000
constexpr int    kNewFreq            = 2;  // 16000 / 8000
constexpr int    kLowpassFilterWidth = 6;
constexpr double kRolloffD           = 0.99;

// base_freq = min(orig_freq, new_freq) * rolloff = min(3, 2) * 0.99 = 1.98.
// This is plain Python arithmetic in torchaudio (functional.py:1345-1350) --
// no tensor exists yet at this point in _get_sinc_resample_kernel -- so it
// stays double-precision here too, exactly like width below.
constexpr double kBaseFreqD = double(kOrigFreq < kNewFreq ? kOrigFreq : kNewFreq) * kRolloffD;

// width = ceil(lowpass_filter_width * orig_freq / base_freq)
//       = ceil(6 * 3 / 1.98) = ceil(9.0909...) = 10 (measured against the
// pinned torchaudio in the locked venv -- see the header comment). Also
// plain double-precision Python arithmetic (functional.py:1369).
int sinc_kernel_width() {
    return static_cast<int>(std::ceil(double(kLowpassFilterWidth) * double(kOrigFreq) / kBaseFreqD));
}

// From here down, every one of these constants is combined with a float32
// "tensor" value inside _get_sinc_resample_kernel. PyTorch's tensor-scalar
// op for float32_tensor op python_number rounds the scalar down to float32
// FIRST, then computes entirely in float32 -- there is no double
// intermediate anywhere in the kernel-construction loop below (see the
// header for how this was verified and why an earlier revision of this file
// had it backwards). kBaseFreqF and kScaleF are exactly kBaseFreqD and
// scale rounded to float32 once, at the point they stop being pure-Python
// scalars and start being multiplied into tensor-shaped values.
constexpr float kBaseFreqF = float(kBaseFreqD);
// scale = base_freq / orig_freq = 1.98 / 3 = 0.66, likewise computed in
// double (pure Python, functional.py:1395) then rounded to float32 once.
constexpr float kScaleF    = float(kBaseFreqD / double(kOrigFreq));
constexpr float kPiF       = 3.14159265358979323846f;

// The lazily-built polyphase kernel table: kNewFreq phases, each
// 2*width + kOrigFreq taps (23 with the pinned constants above, so 46
// coefficients total). Row-major: taps[phase * taps_per_phase + k].
struct SincKernel {
    int                width          = 0;
    int                taps_per_phase = 0;
    std::vector<float> taps;
};

const SincKernel & sinc_kernel() {
    static const SincKernel kernel = [] {
        SincKernel built;
        built.width          = sinc_kernel_width();
        built.taps_per_phase = 2 * built.width + kOrigFreq;
        built.taps.resize(size_t(kNewFreq) * size_t(built.taps_per_phase));

        for (int phase = 0; phase < kNewFreq; ++phase) {
            for (int k = 0; k < built.taps_per_phase; ++k) {
                // idx[k] = float32((k - width) / orig_freq) (functional.py:1376).
                // orig_freq (a Python int) is exactly representable in
                // float32, so casting it to float here IS "round to float32
                // first" -- no separate rounding step is needed.
                const float idx = float(k - built.width) / float(kOrigFreq);
                // t = float32(-phase / new_freq) + idx[k] (functional.py:1378).
                float       t   = float(-phase) / float(kNewFreq) + idx;
                // t *= base_freq (functional.py:1379) -- float32 * float32.
                t               = t * kBaseFreqF;
                // t = clamp(t, -lowpass_filter_width, lowpass_filter_width) (functional.py:1380).
                t               = std::clamp(t, -float(kLowpassFilterWidth), float(kLowpassFilterWidth));

                // window = cos(((t * pi) / lowpass_filter_width) / 2) ^ 2 (functional.py:1385).
                const float t_pi   = t * kPiF;
                const float w_arg1 = t_pi / float(kLowpassFilterWidth);
                const float w_arg2 = w_arg1 / 2.0f;
                const float cosine = std::cos(w_arg2);
                const float window = cosine * cosine;

                // t_rad = t * pi (functional.py:1393 reassigns t -- the same
                // expression as t_pi above, so reused rather than recomputed).
                const float t_rad = t_pi;
                // sinc = t_rad == 0 ? 1.0 : sin(t_rad) / t_rad (functional.py:1396).
                const float sinc  = (t_rad == 0.0f) ? 1.0f : std::sin(t_rad) / t_rad;

                // kernel = sinc * (window * scale) (functional.py:1397) -- both
                // multiplies are float32 * float32.
                const float scaled_window                                            = window * kScaleF;
                built.taps[size_t(phase) * size_t(built.taps_per_phase) + size_t(k)] = sinc * scaled_window;
            }
        }
        return built;
    }();
    return kernel;
}

}  // namespace

bool resample_24k_to_16k(const std::vector<float> & input, std::vector<float> & output) {
    output.clear();
    const size_t length = input.size();
    if (length == 0) {
        return false;
    }

    const SincKernel & kernel = sinc_kernel();
    const int          width  = kernel.width;
    const int          taps   = kernel.taps_per_phase;

    // Pad `width` zeros on the left and `width + orig_freq` on the right
    // (functional.py:1424).
    const size_t       left_pad      = size_t(width);
    const size_t       right_pad     = size_t(width + kOrigFreq);
    const size_t       padded_length = length + left_pad + right_pad;
    std::vector<float> padded(padded_length, 0.0f);
    std::copy(input.begin(), input.end(), padded.begin() + std::ptrdiff_t(left_pad));

    // target_length = torch.ceil(torch.as_tensor(new_freq * length /
    // orig_freq)).long() (functional.py:1427). `new_freq * length /
    // orig_freq` is plain Python int/true-division arithmetic (double
    // precision) -- but torch.as_tensor's default dtype is float32, so the
    // pre-ceil value is downcast to float32 BEFORE ceiling, not ceiled
    // directly in double. Checked against a plain double-then-ceil model
    // for every length in [1, 200000] and a sparse sample up to 3,000,000:
    // zero divergence (this module's own reference clips cap well under
    // 500,000 samples), but the downcast is transcribed anyway so this file
    // carries no unverified promotion gap.
    const double   pre_ceil_d    = double(kNewFreq) * double(length) / double(kOrigFreq);
    const float    pre_ceil_f    = float(pre_ceil_d);
    const uint64_t target_length = uint64_t(std::ceil(pre_ceil_f));

    output.assign(size_t(target_length), 0.0f);
    for (uint64_t j = 0; j < target_length; ++j) {
        const size_t block = size_t(j / uint64_t(kNewFreq));
        const size_t phase = size_t(j % uint64_t(kNewFreq));
        const size_t base  = block * size_t(kOrigFreq);
        if (base + size_t(taps) > padded_length) {
            // Cannot happen for the formulas above (conv1d's own output
            // length identity guarantees enough padded samples remain for
            // every j < target_length); guarded rather than assumed.
            output.clear();
            return false;
        }
        const float * tap_row = kernel.taps.data() + phase * size_t(taps);
        float         acc     = 0.0f;
        for (int k = 0; k < taps; ++k) {
            acc += tap_row[k] * padded[base + size_t(k)];
        }
        output[size_t(j)] = acc;
    }
    return true;
}

// A generous, single-shot budget: this graph is built once per cloning
// request, not once per denoising step the way the generator's are, so it is
// not worth sizing precisely the way model.cpp's per-layer formula does for a
// graph that gets rebuilt dozens of times in one synthesis.
constexpr size_t kSemanticGraphNodeBudget = 8192;

synth_status_t run_semantic_branch(const BackendPlan &        plan,
                                   const ModelWeights &       weights,
                                   const HParams &            hparams,
                                   const std::vector<float> & pcm_16k,
                                   int                        threads,
                                   std::vector<float> &       semantic_mean,
                                   std::vector<float> *       out_semantic_encoder) {
    semantic_mean.clear();
    if (out_semantic_encoder != nullptr) {
        out_semantic_encoder->clear();
    }
    if (pcm_16k.empty()) {
        return SYNTH_ERR_INVALID_ARG;
    }

    // The input tensor lives in its own tiny persistent buffer, the same
    // shape model.cpp's Persistent gives every graph-run input in this
    // family: the graph allocator below must not own it.
    ggml_init_params input_params{};
    input_params.mem_size        = ggml_tensor_overhead() * 4;
    input_params.no_alloc        = true;
    ggml_context * input_context = ggml_init(input_params);
    if (input_context == nullptr) {
        return SYNTH_ERR_OOM;
    }
    ggml_tensor *         pcm_tensor   = ggml_new_tensor_1d(input_context, GGML_TYPE_F32, int64_t(pcm_16k.size()));
    ggml_backend_buffer_t input_buffer = ggml_backend_alloc_ctx_tensors(input_context, plan.cpu_backend());
    if (input_buffer == nullptr) {
        ggml_free(input_context);
        return SYNTH_ERR_OOM;
    }
    ggml_backend_tensor_set(pcm_tensor, pcm_16k.data(), 0, ggml_nbytes(pcm_tensor));

    ggml_init_params graph_params{};
    graph_params.mem_size        = ggml_tensor_overhead() * (kSemanticGraphNodeBudget + 256) +
                                   ggml_graph_overhead_custom(kSemanticGraphNodeBudget, false);
    graph_params.no_alloc        = true;
    ggml_context * graph_context = ggml_init(graph_params);
    if (graph_context == nullptr) {
        ggml_backend_buffer_free(input_buffer);
        ggml_free(input_context);
        return SYNTH_ERR_OOM;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(graph_context, kSemanticGraphNodeBudget, false);

    ggml_tensor * mean_tensor = nullptr;
    ggml_tensor * final_tensor =
        build_semantic_branch(graph_context, pcm_tensor, weights, hparams, nullptr, &mean_tensor, nullptr);
    synth_status_t status = SYNTH_OK;
    if (final_tensor == nullptr || mean_tensor == nullptr) {
        status = SYNTH_ERR_INTERNAL;
    } else {
        // `mean_tensor` is a side tap, not the graph's own output, so it must
        // be marked and expanded before allocation or the allocator is free
        // to reuse its buffer for a later node -- generator_branch_forward's
        // own rule for its probe tensors.
        ggml_set_output(mean_tensor);
        ggml_build_forward_expand(graph, mean_tensor);
        ggml_build_forward_expand(graph, final_tensor);

        const size_t         hash_size = size_t(ggml_graph_size(graph)) + 4096;
        ggml_backend_sched_t scheduler = plan.create_cpu_scheduler(hash_size);
        if (scheduler == nullptr) {
            status = SYNTH_ERR_BACKEND;
        } else {
            if (!ggml_backend_sched_alloc_graph(scheduler, graph)) {
                status = SYNTH_ERR_OOM;
            } else {
                plan.set_threads(threads > 0 ? threads : default_synthesis_threads());
                status = ggml_backend_sched_graph_compute(scheduler, graph) == GGML_STATUS_SUCCESS ? SYNTH_OK :
                                                                                                     SYNTH_ERR_BACKEND;
                if (status == SYNTH_OK) {
                    semantic_mean.resize(size_t(ggml_nelements(mean_tensor)));
                    ggml_backend_tensor_get(mean_tensor, semantic_mean.data(), 0, ggml_nbytes(mean_tensor));
                    if (out_semantic_encoder != nullptr) {
                        out_semantic_encoder->resize(size_t(ggml_nelements(final_tensor)));
                        ggml_backend_tensor_get(final_tensor, out_semantic_encoder->data(), 0,
                                                ggml_nbytes(final_tensor));
                    }
                }
            }
            ggml_backend_sched_free(scheduler);
        }
    }
    ggml_free(graph_context);
    ggml_backend_buffer_free(input_buffer);
    ggml_free(input_context);
    return status;
}

// This graph (the acoustic branch plus the fusion Linear) is built once per
// cloning request too, matching kSemanticGraphNodeBudget's own reasoning;
// smaller because there is no attention stack here.
constexpr size_t kAcousticGraphNodeBudget = 4096;

synth_status_t run_acoustic_and_fuse(const BackendPlan &        plan,
                                     const ModelWeights &       weights,
                                     const HParams &            hparams,
                                     const std::vector<float> & pcm_24k,
                                     const std::vector<float> & semantic_encoder_output,
                                     int                        threads,
                                     std::vector<float> &       fused_latent,
                                     std::vector<float> *       out_acoustic) {
    fused_latent.clear();
    if (out_acoustic != nullptr) {
        out_acoustic->clear();
    }
    const uint32_t semantic_hidden = hparams.semantic.hidden_size;
    if (pcm_24k.empty() || semantic_hidden == 0 || semantic_encoder_output.empty() ||
        semantic_encoder_output.size() % semantic_hidden != 0) {
        return SYNTH_ERR_INVALID_ARG;
    }
    const int64_t semantic_frames = int64_t(semantic_encoder_output.size() / semantic_hidden);

    // Two persistent inputs this time -- the raw waveform and the
    // already-computed semantic tensor -- sharing the same "graph allocator
    // must not own it" rule run_semantic_branch's own single input follows.
    ggml_init_params input_params{};
    input_params.mem_size        = ggml_tensor_overhead() * 4;
    input_params.no_alloc        = true;
    ggml_context * input_context = ggml_init(input_params);
    if (input_context == nullptr) {
        return SYNTH_ERR_OOM;
    }
    ggml_tensor * pcm_tensor = ggml_new_tensor_1d(input_context, GGML_TYPE_F32, int64_t(pcm_24k.size()));
    ggml_tensor * semantic_tensor =
        ggml_new_tensor_2d(input_context, GGML_TYPE_F32, int64_t(semantic_hidden), semantic_frames);
    ggml_backend_buffer_t input_buffer = ggml_backend_alloc_ctx_tensors(input_context, plan.cpu_backend());
    if (input_buffer == nullptr) {
        ggml_free(input_context);
        return SYNTH_ERR_OOM;
    }
    ggml_backend_tensor_set(pcm_tensor, pcm_24k.data(), 0, ggml_nbytes(pcm_tensor));
    ggml_backend_tensor_set(semantic_tensor, semantic_encoder_output.data(), 0, ggml_nbytes(semantic_tensor));

    ggml_init_params graph_params{};
    graph_params.mem_size        = ggml_tensor_overhead() * (kAcousticGraphNodeBudget + 256) +
                                   ggml_graph_overhead_custom(kAcousticGraphNodeBudget, false);
    graph_params.no_alloc        = true;
    ggml_context * graph_context = ggml_init(graph_params);
    if (graph_context == nullptr) {
        ggml_backend_buffer_free(input_buffer);
        ggml_free(input_context);
        return SYNTH_ERR_OOM;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(graph_context, kAcousticGraphNodeBudget, false);

    ggml_tensor *  acoustic_tensor = build_acoustic_encoder(graph_context, pcm_tensor, weights, hparams);
    ggml_tensor *  fused_tensor = acoustic_tensor != nullptr ?
                                      build_reference_fusion(graph_context, acoustic_tensor, semantic_tensor, weights) :
                                      nullptr;
    synth_status_t status       = SYNTH_OK;
    if (fused_tensor == nullptr) {
        // Either builder refusing -- including build_reference_fusion's own
        // frame-count check -- means the acoustic branch's length disagreed
        // with the semantic branch's, which build_acoustic_encoder's own
        // header comment names as a wiring defect upstream of this file.
        status = SYNTH_ERR_INTERNAL;
    } else {
        if (out_acoustic != nullptr) {
            // A side tap, not the graph's own output: marked and expanded
            // before allocation, run_semantic_branch's own rule for
            // `mean_tensor`.
            ggml_set_output(acoustic_tensor);
            ggml_build_forward_expand(graph, acoustic_tensor);
        }
        ggml_build_forward_expand(graph, fused_tensor);

        const size_t         hash_size = size_t(ggml_graph_size(graph)) + 4096;
        ggml_backend_sched_t scheduler = plan.create_cpu_scheduler(hash_size);
        if (scheduler == nullptr) {
            status = SYNTH_ERR_BACKEND;
        } else {
            if (!ggml_backend_sched_alloc_graph(scheduler, graph)) {
                status = SYNTH_ERR_OOM;
            } else {
                plan.set_threads(threads > 0 ? threads : default_synthesis_threads());
                status = ggml_backend_sched_graph_compute(scheduler, graph) == GGML_STATUS_SUCCESS ? SYNTH_OK :
                                                                                                     SYNTH_ERR_BACKEND;
                if (status == SYNTH_OK) {
                    fused_latent.resize(size_t(ggml_nelements(fused_tensor)));
                    ggml_backend_tensor_get(fused_tensor, fused_latent.data(), 0, ggml_nbytes(fused_tensor));
                    if (out_acoustic != nullptr) {
                        out_acoustic->resize(size_t(ggml_nelements(acoustic_tensor)));
                        ggml_backend_tensor_get(acoustic_tensor, out_acoustic->data(), 0, ggml_nbytes(acoustic_tensor));
                    }
                }
            }
            ggml_backend_sched_free(scheduler);
        }
    }
    ggml_free(graph_context);
    ggml_backend_buffer_free(input_buffer);
    ggml_free(input_context);
    return status;
}

float reference_rms(const std::vector<float> & pcm) {
    if (pcm.empty()) {
        return 0.0f;
    }
    // Neumaier compensated summation over pcm's squared float32 values,
    // widened to double for the running sum -- DurationEstimator::total_weight's
    // own technique (frontend-host.cpp), cited in this function's header
    // comment for why it is the closest a plain host loop gets to numpy's own
    // float32 pairwise `np.mean` without reimplementing that algorithm.
    double total      = 0.0;
    double correction = 0.0;
    for (float sample : pcm) {
        const double value = double(sample) * double(sample);
        const double sum   = total + value;
        if (std::fabs(total) >= std::fabs(value)) {
            correction += (total - sum) + value;
        } else {
            correction += (value - sum) + total;
        }
        total = sum;
    }
    const double mean = (total + correction) / double(pcm.size());
    return float(std::sqrt(mean));
}

void clip_and_boost_reference(std::vector<float> & pcm, uint32_t hop_length, float & ref_rms) {
    // Step 1: ref_rms, measured on the FULL input, before anything else
    // touches it -- omnivoice.py:774, `ref_rms = sqrt(mean(ref_wav**2))`.
    // This is the value the caller keeps (ReferenceEncoding::ref_rms):
    // PRE-boost AND pre-clip, matching upstream's own measurement point
    // exactly, not merely "pre-boost" -- see this function's own header
    // comment for why an earlier revision that clipped first was wrong at
    // more than a boundary.
    ref_rms = reference_rms(pcm);

    // Step 2: the quiet-reference boost, applied to that SAME full buffer,
    // still before any clipping -- omnivoice.py:775-776, `if 0 < ref_rms <
    // 0.1: ref_wav = ref_wav * 0.1 / ref_rms`. `ref_rms == 0` (a digitally
    // silent reference) takes neither this arm nor any other special
    // handling here -- Task 14's profile-creation seam is what rejects that
    // case, not this function.
    if (ref_rms > 0.0f && ref_rms < 0.1f) {
        const float scale = 0.1f / ref_rms;
        for (float & sample : pcm) {
            sample *= scale;
        }
    }

    // Step 3: hop-clip, LAST, on the (possibly boosted) buffer -- upstream's
    // own `clip_size = ref_wav.shape[-1] % chunk_size; ref_wav[:,
    // :-clip_size]` (omnivoice.py:816-818), transcribed as a resize rather
    // than a negative-index slice. `ref_rms` is left as the real measurement
    // over the original input even when the buffer ends up empty here (too
    // short for one hop, or hop_length == 0) -- it was a genuine measurement
    // of what the caller passed in, not something this branch invalidates.
    if (hop_length == 0 || pcm.size() < size_t(hop_length)) {
        pcm.clear();
        return;
    }
    const size_t clipped_length = pcm.size() - (pcm.size() % size_t(hop_length));
    pcm.resize(clipped_length);
}

bool rvq_encode(const std::vector<RvqQuantizerWeights> & quantizers,
                const std::vector<float> &               latent,
                uint64_t                                 frames,
                std::vector<int32_t> &                   tokens,
                float *                                  narrowest_gap,
                std::vector<float> *                     out_gaps) {
    tokens.clear();
    if (out_gaps != nullptr) {
        out_gaps->clear();
    }
    if (narrowest_gap != nullptr) {
        *narrowest_gap = 0.0f;
    }
    if (quantizers.empty() || frames == 0) {
        return false;
    }

    // Every level must resolve its full triple and agree on `concat`
    // (input_proj's own in-width) before a single tensor is pulled to host --
    // codec_rvq_decode's own "check every level before touching data" rule.
    int64_t concat = -1;
    for (const RvqQuantizerWeights & quantizer : quantizers) {
        if (quantizer.input_proj.weight == nullptr || quantizer.input_proj.bias == nullptr ||
            quantizer.output_proj.weight == nullptr || quantizer.output_proj.bias == nullptr ||
            quantizer.codebook == nullptr || !ggml_is_contiguous(quantizer.input_proj.weight) ||
            !ggml_is_contiguous(quantizer.input_proj.bias) || !ggml_is_contiguous(quantizer.output_proj.weight) ||
            !ggml_is_contiguous(quantizer.output_proj.bias) || !ggml_is_contiguous(quantizer.codebook)) {
            return false;
        }
        // The host copy below (ggml_backend_tensor_get) reads each tensor's
        // raw bytes into a float-typed destination vector, verbatim and
        // unconverted -- a non-F32 tensor (e.g. a future quantized profile)
        // would have its bytes silently misinterpreted as float rather than
        // dequantized. Not reachable today (only F32 profiles ship), but
        // quantized profiles are named Plan 4 work.
        if (quantizer.input_proj.weight->type != GGML_TYPE_F32 || quantizer.input_proj.bias->type != GGML_TYPE_F32 ||
            quantizer.output_proj.weight->type != GGML_TYPE_F32 || quantizer.output_proj.bias->type != GGML_TYPE_F32 ||
            quantizer.codebook->type != GGML_TYPE_F32) {
            return false;
        }
        const int64_t level_concat = quantizer.input_proj.weight->ne[0];
        const int64_t level_dim    = quantizer.input_proj.weight->ne[1];
        if (level_concat <= 0 || level_dim <= 0 || quantizer.input_proj.bias->ne[0] != level_dim ||
            quantizer.output_proj.weight->ne[0] != level_dim || quantizer.output_proj.weight->ne[1] != level_concat ||
            quantizer.output_proj.bias->ne[0] != level_concat || quantizer.codebook->ne[0] != level_dim ||
            quantizer.codebook->ne[1] < 2) {
            return false;
        }
        if (concat < 0) {
            concat = level_concat;
        } else if (concat != level_concat) {
            return false;
        }
    }
    if (concat <= 0 || latent.size() != size_t(concat) * size_t(frames)) {
        return false;
    }

    const size_t levels = quantizers.size();
    tokens.assign(levels * size_t(frames), 0);
    if (out_gaps != nullptr) {
        out_gaps->assign(levels * size_t(frames), 0.0f);
    }

    // Frame-major, channel-fastest -- run_acoustic_and_fuse's own
    // `fused_latent` layout exactly -- mutated in place as each level's
    // dequant is subtracted, so the next level's project_in reads the
    // residual rather than the original latent.
    std::vector<float> residual(latent);
    float              global_narrowest = std::numeric_limits<float>::infinity();
    std::vector<float> z;
    std::vector<float> embed_sq;

    for (size_t level = 0; level < levels; ++level) {
        const RvqQuantizerWeights & quantizer      = quantizers[level];
        const int64_t               dim            = quantizer.input_proj.weight->ne[1];
        const int64_t               codebook_size  = quantizer.codebook->ne[1];
        const size_t                dim_count      = size_t(dim);
        const size_t                concat_count   = size_t(concat);
        const size_t                codebook_count = size_t(codebook_size);

        // Named size locals rather than `Type(expr)` directly in the
        // constructor call: `std::vector<float> in_bias(size_t(dim))` is the
        // classic most-vexing-parse trap -- a single parenthesized
        // type-cast-looking argument reads as a function declaration, not a
        // call to the vector's size constructor.
        std::vector<float> in_weight(concat_count * dim_count);
        std::vector<float> in_bias(dim_count);
        std::vector<float> out_weight(dim_count * concat_count);
        std::vector<float> out_bias(concat_count);
        std::vector<float> codebook(dim_count * codebook_count);
        ggml_backend_tensor_get(quantizer.input_proj.weight, in_weight.data(), 0,
                                ggml_nbytes(quantizer.input_proj.weight));
        ggml_backend_tensor_get(quantizer.input_proj.bias, in_bias.data(), 0, ggml_nbytes(quantizer.input_proj.bias));
        ggml_backend_tensor_get(quantizer.output_proj.weight, out_weight.data(), 0,
                                ggml_nbytes(quantizer.output_proj.weight));
        ggml_backend_tensor_get(quantizer.output_proj.bias, out_bias.data(), 0,
                                ggml_nbytes(quantizer.output_proj.bias));
        ggml_backend_tensor_get(quantizer.codebook, codebook.data(), 0, ggml_nbytes(quantizer.codebook));

        // sum_d E[e][d]^2 depends only on the codebook, so it is computed once
        // per level rather than once per (level, frame) -- upstream's own
        // `embed.pow(2).sum(0, keepdim=True)` is likewise computed once and
        // broadcast over every frame.
        embed_sq.assign(size_t(codebook_size), 0.0f);
        for (int64_t code = 0; code < codebook_size; ++code) {
            const float * row = codebook.data() + size_t(code) * size_t(dim);
            float         sum = 0.0f;
            for (int64_t d = 0; d < dim; ++d) {
                sum += row[d] * row[d];
            }
            embed_sq[size_t(code)] = sum;
        }

        z.assign(size_t(dim), 0.0f);
        for (uint64_t frame = 0; frame < frames; ++frame) {
            float * frame_residual = residual.data() + size_t(frame) * size_t(concat);

            // z = project_in(residual): concat -> dim, biased Linear. Plain
            // left-to-right float32 accumulation -- see this function's own
            // header comment for the accumulation-order choice and why the
            // real-scale exact-token gate, not this comment, is what settles
            // it.
            for (int64_t d = 0; d < dim; ++d) {
                const float * weight_row  = in_weight.data() + size_t(d) * size_t(concat);
                float         accumulator = in_bias[size_t(d)];
                for (int64_t index = 0; index < concat; ++index) {
                    accumulator += weight_row[index] * frame_residual[index];
                }
                z[size_t(d)] = accumulator;
            }
            float scaled = 0.0f;
            for (int64_t d = 0; d < dim; ++d) {
                scaled += z[size_t(d)] * z[size_t(d)];
            }

            // Plain L2 nearest neighbour: dist[e] = -(scaled - 2*z.E[e] +
            // ||E[e]||^2); argmax over e, ties -> LOWEST id. The strict '>'
            // below is the tie rule itself: a later e with an EQUAL dist never
            // replaces an earlier one, so the first (lowest) maximal index
            // wins, matching torch's own dist.max(dim=-1).indices.
            int64_t best_code   = -1;
            float   best_dist   = -std::numeric_limits<float>::infinity();
            float   second_dist = -std::numeric_limits<float>::infinity();
            for (int64_t code = 0; code < codebook_size; ++code) {
                const float * row = codebook.data() + size_t(code) * size_t(dim);
                float         dot = 0.0f;
                for (int64_t d = 0; d < dim; ++d) {
                    dot += z[size_t(d)] * row[d];
                }
                const float inner = scaled - 2.0f * dot + embed_sq[size_t(code)];
                const float dist  = -inner;
                if (dist > best_dist) {
                    second_dist = best_dist;
                    best_dist   = dist;
                    best_code   = code;
                } else if (dist > second_dist) {
                    second_dist = dist;
                }
            }

            // `best_code` stays -1 only when every `dist` compared false
            // against `best_dist` -- which a real, if unlikely, NaN latent
            // value (this function's input comes from a graph, not from a
            // programmer's own literal) makes possible: a NaN loses every
            // `>` comparison, including against another NaN, so the loop
            // above never advances off its `-1` seed. `size_t(best_code)`
            // below would then wrap to SIZE_MAX, turning
            // `codebook.data() + size_t(best_code) * size_t(dim)` into a
            // wild pointer -- a real out-of-bounds read, not a merely bad
            // token. Refused as a caller-visible failure (matching this
            // function's own "Returns false" contract in the header) rather
            // than asserted: a NaN reaching here is an internal-error-shaped
            // runtime state, not a can't-happen a debug build alone should
            // catch.
            if (best_code < 0) {
                tokens.clear();
                if (out_gaps != nullptr) {
                    out_gaps->clear();
                }
                return false;
            }

            const size_t slot = level * size_t(frames) + size_t(frame);
            tokens[slot]      = int32_t(best_code);
            const float gap   = best_dist - second_dist;
            if (out_gaps != nullptr) {
                (*out_gaps)[slot] = gap;
            }
            global_narrowest = std::fmin(global_narrowest, gap);

            // dequant = project_out(E[best]): dim -> concat, biased Linear;
            // residual -= dequant, in the concat space -- upstream's own
            // `residual = residual - quantized` (the RVQ loop's own line, not
            // a per-channel or per-level rescale).
            const float * best_row = codebook.data() + size_t(best_code) * size_t(dim);
            for (int64_t out_index = 0; out_index < concat; ++out_index) {
                const float * weight_row  = out_weight.data() + size_t(out_index) * size_t(dim);
                float         accumulator = out_bias[size_t(out_index)];
                for (int64_t d = 0; d < dim; ++d) {
                    accumulator += weight_row[d] * best_row[d];
                }
                frame_residual[out_index] -= accumulator;
            }
        }
    }

    if (narrowest_gap != nullptr) {
        *narrowest_gap = std::isfinite(global_narrowest) ? global_narrowest : 0.0f;
    }
    return true;
}

}  // namespace synth::omnivoice
