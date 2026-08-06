#pragma once

#include "arch/omnivoice/catalog.h"
#include "arch/omnivoice/weights.h"
#include "synthesize.h"

#include <cstdint>
#include <vector>

namespace synth {
class BackendPlan;
}

namespace synth::omnivoice {

// The 24 kHz -> 16 kHz resampler HuBERT's semantic branch reads Reference
// Audio through (Task 11 onward: src/arch/omnivoice/reference-encoder.{h,cpp}
// builds the graph that consumes this function's output). This is a line-by-
// line transcription of torchaudio 2.11.0's resample() at the pinned defaults
// -- lowpass_filter_width=6, rolloff=0.99, resampling_method=
// "sinc_interp_hann", beta=None -- for exactly the orig_freq=24000,
// new_freq=16000 pair. Source (pinned at
// scripts/envs/omnivoice/.venv/lib/python3.12/site-packages/torchaudio/functional/functional.py,
// torchaudio 2.11.0):
//   - resample():                          functional.py:1435-1490
//   - _get_sinc_resample_kernel():         functional.py:1305-1402
//   - _apply_sinc_resample_kernel():       functional.py:1405-1432
//
// gcd(24000, 16000) = 8000 -> after reduction orig_freq = 24000/8000 = 3,
// new_freq = 16000/8000 = 2 (functional.py:1340-1341): a 2-phase polyphase
// filter over stride-3 input windows.
//
// DTYPE (confirmed by instrumenting the pinned torchaudio in the locked venv,
// not assumed): resample() calls
// `_get_sinc_resample_kernel(..., waveform.device, waveform.dtype)`
// (functional.py:1486-1487) -- for the float32 waveform this port's caller
// always passes, that `dtype` argument is torch.float32, NOT None. Inside
// _get_sinc_resample_kernel, `idx_dtype = dtype if dtype is not None else
// torch.float64` (functional.py:1374) therefore evaluates to float32, and
// `idx`, `t`, `window` and `kernels` (functional.py:1376-1397) all inherit
// float32 from that same non-None dtype -- there is no float64 stage at all
// on this call path. The `if dtype is None: kernels =
// kernels.to(dtype=torch.float32)` cast at functional.py:1399-1400 is
// correspondingly never taken. An earlier sketch of this task assumed
// "float64 then cast once"; that is the *other* call path (dtype=None, used
// when no waveform dtype is threaded through) and is NOT what runs here --
// instrumenting `_get_sinc_resample_kernel` inside `resample()` for a
// float32 waveform shows `kernel_dtype == torch.float32` and confirms the
// two paths differ (float32-throughout vs float64-then-cast kernels differ
// by up to ~6e-8 per tap on this exact orig/new pair), so getting this wrong
// is observable, not academic.
//
// Kernel construction (functional.py:1345-1397). `base_freq`, `width` and
// `scale` are plain Python doubles, computed with no tensor involved yet
// (functional.py:1345-1350, 1369, 1395):
//   base_freq  = min(orig_freq, new_freq) * rolloff        = min(3,2)*0.99 = 1.98
//   width      = ceil(lowpass_filter_width * orig_freq / base_freq)
//              = ceil(6*3/1.98) = ceil(9.0909...) = 10          (measured)
//   taps       = 2*width + orig_freq = 23 per phase; 2 phases (new_freq)
//              = 46 coefficients total                          (measured)
//   scale      = base_freq / orig_freq = 1.98/3 = 0.66
//
// SCALAR PROMOTION -- the second fact this task got wrong before it was
// caught by code review, corrected here with the evidence that settled it.
// Once `base_freq`, `scale` and `math.pi` (all Python doubles) are combined
// with a float32 "tensor" value (`t *= base_freq`, `window * scale`, `t *
// math.pi`, ...), PyTorch's tensor-scalar op for `float32_tensor op
// python_number` **rounds the scalar down to float32 FIRST, then computes
// entirely in float32 -- there is no double intermediate.** A first draft of
// this file modeled it the other way (promote the float32 side to double,
// multiply, round once back to float32) on the strength of a 5-value manual
// spot check that happened not to expose the difference. Code review
// rebuilt the kernel-construction code standalone and diffed it against
// `_get_sinc_resample_kernel`'s real output tensor directly: the
// double-intermediate model was bit-exact on only 8 of 46 taps (up to
// 4.77e-7 off elsewhere); rounding each scalar to float32 once and then
// computing float32-op-float32 throughout is bit-exact on all 46 -- 0.0
// diff, confirmed independently by isolating the very first scalar multiply
// (`t *= base_freq`) against a from-scratch torch tensor of the same
// pre-multiply values, and again end-to-end with a full from-scratch
// reconstruction of the kernel. `kBaseFreqF` and `kScaleF` in the .cpp are
// exactly `kBaseFreqD` and `scale` rounded to float32 once, at the point
// they stop being pure-Python scalars; every downstream expression is
// `float op float`, never `float op double`.
//
// This is the same class of mistake as Plan 2's `shifted_timesteps`
// (src/arch/omnivoice/generator-host.cpp:54-82): torch's reference
// computation is float32-elementwise, not double, and the plan's own
// standing facts record it as such ("the commit schedule is float32
// torch-elementwise", docs/superpowers/plans/2026-08-01-omnivoice-plan-3-sampling-cloning.md:101).
// There too, a double-precision port of a nominally-double formula silently
// disagreed with the float32 reference at a small fraction of input
// lengths. Getting the *rounding order* right, not just the *final dtype*,
// is the recurring lesson.
//
// Per phase p in [0, new_freq), tap k in [0, 2*width+orig_freq), every step
// float32 throughout (orig_freq, new_freq, lowpass_filter_width are Python
// ints, exactly representable in float32, so casting them to float IS
// "round to float32 first"):
//   idx[k]   = float32(k - width) / float32(orig_freq)            (functional.py:1376)
//   t        = float32(-p) / float32(new_freq) + idx[k]           (functional.py:1378)
//   t       *= kBaseFreqF                                         (functional.py:1379)
//   t        = clamp(t, -lowpass_filter_width, lowpass_filter_width)  (functional.py:1380)
//   window   = cos(((t * pi) / lowpass_filter_width) / 2) ^ 2    (functional.py:1385, Hann)
//   t_rad    = t * pi                                            (functional.py:1393, reassigns t)
//   sinc     = t_rad == 0 ? 1.0 : sin(t_rad) / t_rad              (functional.py:1396)
//   kernel   = sinc * (window * kScaleF)                          (functional.py:1397)
//
// Convolution (_apply_sinc_resample_kernel, functional.py:1420-1428): pad the
// input with `width` zeros on the left and `width + orig_freq` zeros on the
// right (functional.py:1424), then for output index j (0-based over the
// interleaved [block, phase] stream conv1d's transpose(1,2).reshape produces,
// functional.py:1425-1426), letting block = j / new_freq, phase = j %
// new_freq:
//   y[j] = sum_{k=0}^{taps-1} kernel[phase][k] * padded[block*orig_freq + k]
//
// target_length = torch.ceil(torch.as_tensor(new_freq * length /
// orig_freq)).long() (functional.py:1427) -- a THIRD promotion point:
// `new_freq * length / orig_freq` is plain Python int/true-division
// arithmetic (double precision), but `torch.as_tensor` on a plain Python
// float defaults to torch's default dtype, float32 (confirmed directly:
// `torch.as_tensor(2/3).dtype == torch.float32`), so the pre-ceil value is
// downcast to float32 BEFORE ceiling, not ceiled directly in double. Output
// is the first target_length samples of the interleaved stream
// (functional.py:1428). Checked (both by code review and independently
// reproduced here) against a plain double-then-ceil model for every length
// in [1, 200000] plus a sparse sample up to 3,000,000: zero divergence --
// this module's reference clips cap well under 500,000 samples -- but the
// downcast is transcribed anyway so this file carries no unverified
// promotion gap. Confirmed against the oracle's own measured pair: 336960
// input samples -> 224640 output samples (336960 divides evenly by 3, so
// this particular case has no rounding ambiguity to probe on its own; the
// arithmetic above is exercised at a non-exact ratio by this module's own
// length-arithmetic tests, e.g. 961 -> 641).
//
// Returns false (leaving `output` empty) only when `input` is empty; every
// other length, including 1-3 samples, has a well-defined torchaudio output
// (measured against the pinned venv) and succeeds here too.
bool resample_24k_to_16k(const std::vector<float> & input, std::vector<float> & output);

// Runs reference-encoder.h's build_semantic_branch once over a 16 kHz mono
// PCM buffer and reads the results back: the graph-construction and
// GraphRun-style allocate/compute/read-back split this family keeps
// throughout (model.cpp's GraphRun, duplicated in miniature here rather than
// exposed from that file, which is anonymous-namespace-local to model.cpp).
//
// Placement is CPU-only, unconditionally: `plan.create_cpu_scheduler` is the
// scheduler this calls, and always will be, regardless of what Model::load's
// codec twin (Task 9, Plan 4) does for the decode path. This branch's own
// output is continuous, but what reads it -- rvq_encode's host-side
// nearest-neighbour argmax (this header, below) -- is a discrete decision, so
// docs/backends.md's discrete-outputs rule holds the whole chain on the CPU
// the same way it holds the generator there. catalog.h's build_model_weights
// documents the tensor groups this reasoning keeps off the accelerator twin.
//
// `pcm_16k` is PRE-pad (see reference-encoder.h). `semantic_mean` receives
// the mean over all hidden states BEFORE the stride-2 downsample -- the
// oracle's own `ref/semantic_mean.f32` probe. `out_semantic_encoder`, when
// non-null, additionally receives the SemanticEncoder's own output (the
// builder's primary return value): no committed oracle probe exists for it
// yet (only `semantic_mean` and the downstream `fused_latent`, which also
// depends on the acoustic branch Task 12 adds, are captured), so it is
// reported for debugging rather than compared against anything today.
//
// Returns SYNTH_ERR_INVALID_ARG for an empty `pcm_16k` (resample_24k_to_16k's
// own empty-input refusal propagates here the same way), SYNTH_ERR_OOM/
// SYNTH_ERR_BACKEND on allocation or scheduling failure, and
// SYNTH_ERR_INTERNAL when build_semantic_branch itself refuses the package's
// shapes. `semantic_mean` (and `out_semantic_encoder`, if requested) are
// cleared up front and left empty on any non-OK return.
synth_status_t run_semantic_branch(const BackendPlan &        plan,
                                   const ModelWeights &       weights,
                                   const HParams &            hparams,
                                   const std::vector<float> & pcm_16k,
                                   int                        threads,
                                   std::vector<float> &       semantic_mean,
                                   std::vector<float> *       out_semantic_encoder = nullptr);

// Runs reference-encoder.h's build_acoustic_encoder once over a 24 kHz mono
// PCM buffer, then build_reference_fusion against an ALREADY-COMPUTED
// semantic branch output -- `semantic_encoder_output`, the flat
// [semantic.hidden_size, T] buffer run_semantic_branch's own
// `out_semantic_encoder` tap produced (build_semantic_branch's PRIMARY
// return value, NOT semantic_mean) -- one graph, one scheduler pass, the
// same allocate/compute/read-back shape run_semantic_branch itself uses.
// Re-injecting an already-computed host buffer as a second graph input
// rather than rebuilding the semantic branch here keeps this function
// independent of run_semantic_branch's own tested contract (Task 11's), at
// the cost of one host-side round trip for a tensor this family only builds
// once per cloning request.
//
// Placement is CPU-only, unconditionally, matching run_semantic_branch's own
// reasoning above: its weights (codec.acoustic_encoder, codec.fc) have no
// accelerator twin either, for the same discrete-argmax reason.
//
// `semantic_encoder_output.size()` must be an exact multiple of
// `hparams.semantic.hidden_size` (the frame count is inferred from that
// division); a mismatch, or a frame count that disagrees with the acoustic
// branch's own output length, is SYNTH_ERR_INVALID_ARG or
// SYNTH_ERR_INTERNAL respectively -- see build_acoustic_encoder's own
// comment for why a disagreeing pair of lengths is a wiring defect this
// refuses rather than a runtime case to pad around.
//
// `fused_latent` receives build_reference_fusion's own output, [
// hparams.codec.hidden_size + hparams.semantic.hidden_size, T] flattened
// frame-major. `out_acoustic`, when non-null, additionally receives
// build_acoustic_encoder's own output (a debugging tap, mirroring
// run_semantic_branch's `out_semantic_encoder`: no committed oracle probe
// isolates the acoustic-only stage today).
//
// Returns SYNTH_ERR_INVALID_ARG for an empty `pcm_24k`/`semantic_encoder_output`
// or a `semantic_encoder_output` size not divisible by the semantic hidden
// width, SYNTH_ERR_OOM/SYNTH_ERR_BACKEND on allocation or scheduling
// failure, and SYNTH_ERR_INTERNAL when either builder refuses the package's
// shapes (including a length disagreement between the two branches).
// `fused_latent` (and `out_acoustic`, if requested) are cleared up front and
// left empty on any non-OK return.
synth_status_t run_acoustic_and_fuse(const BackendPlan &        plan,
                                     const ModelWeights &       weights,
                                     const HParams &            hparams,
                                     const std::vector<float> & pcm_24k,
                                     const std::vector<float> & semantic_encoder_output,
                                     int                        threads,
                                     std::vector<float> &       fused_latent,
                                     std::vector<float> *       out_acoustic = nullptr);

// The reference's loudness: sqrt(mean(pcm^2)), Neumaier-compensated (the same
// technique frontend-host.h's DurationEstimator::total_weight uses for its
// own CPython-`sum()`-matching reduction) over pcm's squared float32 values
// widened to double for the running sum. This is the closest a plain host
// loop gets to numpy's own float32 `np.mean` (which pairwise-sums in float32,
// not double, and not sequentially) without literally reimplementing that
// block-recursive algorithm; VoiceClonePrompt.ref_rms's own value is float32
// pairwise-summed and only needs matching where it crosses the quiet-boost
// threshold below, which this reduction's few-ULP-level agreement with numpy
// does not put in doubt for any reference this family's own limits admit.
// Returns 0.0 for an empty `pcm` (nothing to sum), which is also what keeps
// the quiet-boost gate below from ever firing on a genuinely empty buffer.
float reference_rms(const std::vector<float> & pcm);

// Measures `ref_rms`, applies the quiet-reference boost, THEN hop-clips
// `pcm` to a whole number of `hop_length`-sample frames (the tail remainder,
// if any, is dropped) -- all in place. This is upstream's own order,
// transcribed line for line from `create_voice_clone_prompt`
// (omnivoice/models/omnivoice.py):
//   774:     ref_rms = sqrt(mean(ref_wav**2))            -- on the FULL,
//            un-clipped, un-boosted buffer, before anything else runs.
//   775-776: if 0 < ref_rms < 0.1: ref_wav *= 0.1 / ref_rms
//            -- the boost, applied to that SAME full buffer.
//   778-797: preprocess_prompt (silence removal / long-audio trimming) --
//            SKIPPED here: Plan 3's own two committed clone goldens both run
//            with preprocess_prompt=False, and this port carries no
//            trim/silence-removal stage at all (see the family doc).
//   816-818: chunk_size = hop_length; clip_size = len % chunk_size;
//            ref_wav = ref_wav[:, :-clip_size] if clip_size > 0 else ref_wav
//            -- the hop-clip, LAST, on the (possibly boosted) buffer.
// `ref_rms >= 0.1` or `ref_rms == 0` (a digitally silent reference) takes
// neither the boost nor any other special handling here -- rejecting a
// silent reference is Task 14's profile-creation seam, not this function's
// (see ReferenceEncoding::ref_rms's own comment, omnivoice.h, on the split).
// `ref_rms` receives the PRE-boost value, matching upstream
// VoiceClonePrompt.ref_rms's own contract exactly, including WHICH samples
// it is measured over.
//
// An earlier revision of this function clipped BEFORE measuring `ref_rms`,
// which is wrong at more than a boundary: the boost scale (`0.1/ref_rms`)
// differs systematically for every reference that actually crosses the
// threshold, since the pre-clip and post-clip rms values are not merely
// rounding-apart (measured against the real reference wav this family's
// golden cases use, `models/omnivoice-reference-audio/seedtts_ref_en_1.wav`:
// 0.1229146420955658 on the full 337726-sample file vs. a measurably
// different 0.12305419892072678 on the 336960-sample hop-clipped segment).
// Caught by review before Task 14 could inherit the bug silently: neither
// committed clone case ever crosses the 0.1 threshold (their rms is ~0.123
// either way), so the wrong order never actually moved either golden's
// tokens -- see tests/omnivoice_reference_encoder_test.cpp's own
// measurement-point regression case for the input that DOES distinguish the
// two orders.
//
// `pcm` is left at whatever length the clip produced (possibly empty, if
// `pcm.size() < hop_length` or `hop_length == 0`); `ref_rms` is still the
// real measurement over the ORIGINAL input in that case, not reset to 0.
void clip_and_boost_reference(std::vector<float> & pcm, uint32_t hop_length, float & ref_rms);

// Host-side residual vector quantization, encode direction. DISCRETE
// DECISION -> CPU host code, by the placement rule docs/backends.md and this
// family's own model.cpp header comment both state for the decode
// direction's codec: the eight per-level weight tensors are pulled to host
// with ggml_backend_tensor_get (never read through a stray tensor->data --
// this family carries no rule that weights are always CPU-resident forever,
// only that Plan 1/2 have not yet needed to move them), and every arithmetic
// step from there is a plain C++ loop.
//
// Per level q in 0..levels-1, in ascending order (a later level always reads
// the PREVIOUS level's residual, transcribing
// HiggsAudioV2TokenizerResidualVectorQuantization.encode,
// modeling_higgs_audio_v2_tokenizer.py:427-441):
//   z_q      = project_in_q(residual)         // concat -> dim, biased Linear
//   scaled   = sum_d z_q[d]^2                 // ||z_q||^2, once per frame
//   dist[e]  = -(scaled - 2*(z_q . E_q[e]) + sum_d E_q[e][d]^2)   // per row
//   code     = argmax_e dist[e]               // plain L2 nearest neighbour;
//                                              // ties -> LOWEST id (strict
//                                              // '>' scanning e ascending --
//                                              // upstream's own
//                                              // dist.max(dim=-1).indices
//                                              // takes the first maximal
//                                              // index, HiggsAudioV2Tokenizer
//                                              // EuclideanCodebook.quantize)
//   dequant  = project_out_q(E_q[code])       // dim -> concat, biased Linear
//   residual = residual - dequant             // IN the concat (1024) space
//
// Every dot product and accumulation above is a plain left-to-right float32
// loop (`float acc = 0.0f; for (...) acc += a[i] * b[i];`) -- NOT Neumaier,
// NOT a double-precision intermediate: upstream's own computation is a
// float32 torch matmul (`hidden_states @ embed`, `nn.Linear.forward`), and
// this task's report documents the choice and the real-scale exact-token
// gate (the RVQ token grid has no tolerance, ever) is what actually settles
// whether it agrees with upstream's own BLAS reduction order closely enough,
// at this family's real codebook_dim (64) and codebook_size (1024).
//
// `latent` is frame-major, channel-fastest -- run_acoustic_and_fuse's own
// `fused_latent` layout exactly, `frames` frames of `concat` values each,
// where `concat` is read from the FIRST level's `input_proj.weight` shape and
// every level must agree with it. `tokens` receives `levels * frames`
// codebook-major values (level l, frame t at `l * frames + t`) -- the
// oracle's own `ref/tokens.i32` layout, and SynthesisRequest::reference_tokens'
// own layout.
//
// Margin instrumentation (diagnostic only -- the gate is exact tokens):
// `narrowest_gap`, when non-null, receives the smallest best-vs-second-best
// `dist` gap over every (level, frame) decision made -- 0.0 for a clip with
// no admissible second candidate anywhere is impossible once this returns
// true (codebook_size < 2 is refused below), so this is always a real
// measurement, never a fallback value dressed as one. `out_gaps`, when
// non-null, receives the FULL per-(level, frame) gap grid in the same
// codebook-major layout as `tokens` -- what a caller diagnosing a mismatch
// against the exact-token gate reads to print the (level, frame, got, want,
// gap) quintuple the validator's own contract requires.
//
// Returns false (leaving `tokens`/`out_gaps` cleared) for an empty
// `quantizers`, `frames == 0`, an unresolved or inconsistently-shaped level
// (an input/output projection whose widths disagree, a codebook narrower
// than 2 rows, or a level whose own `concat` disagrees with the first
// level's), a `latent` size that is not exactly `concat * frames`, or a
// (level, frame) whose nearest-neighbour `dist` comes out NaN for every
// codebook row (a NaN latent value propagating through the dot products --
// every `>` comparison against it is false, so no code is ever selected).
// The last case is caught before the resulting `-1` code ever reaches the
// dequantization pointer arithmetic below it.
bool rvq_encode(const std::vector<RvqQuantizerWeights> & quantizers,
                const std::vector<float> &               latent,
                uint64_t                                 frames,
                std::vector<int32_t> &                   tokens,
                float *                                  narrowest_gap = nullptr,
                std::vector<float> *                     out_gaps      = nullptr);

}  // namespace synth::omnivoice
