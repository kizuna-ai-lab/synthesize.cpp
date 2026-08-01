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
// scheduler this calls, matching Model::decode_codes's own "no measurement to
// move it" note -- cloning preparation is a once-per-request cost, not a
// per-step one, so there is nothing here Plan 2's placement work needs to
// revisit.
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

}  // namespace synth::omnivoice
