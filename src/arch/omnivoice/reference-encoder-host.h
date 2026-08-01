#pragma once

#include <cstdint>
#include <vector>

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
// Kernel construction (functional.py:1345-1397), all scalar constants below
// are Python doubles exactly as computed there; each combination of a
// float32 "tensor" value with one of these plain-number scalars is a
// PyTorch "wrapped number" op, which computes as if promoting the float32
// operand to double, multiplying/dividing in double, and rounding once back
// to float32 (verified empirically against the pinned torchaudio for
// representative operands) -- precisely what C++'s ordinary float/double
// mixed-arithmetic promotion already does, so the port below leans on plain
// `float op double` expressions rather than hand-rolled rounding:
//   base_freq  = min(orig_freq, new_freq) * rolloff        = min(3,2)*0.99 = 1.98
//   width      = ceil(lowpass_filter_width * orig_freq / base_freq)
//              = ceil(6*3/1.98) = ceil(9.0909...) = 10          (measured)
//   taps       = 2*width + orig_freq = 23 per phase; 2 phases (new_freq)
//              = 46 coefficients total                          (measured)
//   scale      = base_freq / orig_freq = 1.98/3 = 0.66
//
// Per phase p in [0, new_freq), tap k in [0, 2*width+orig_freq):
//   idx[k]   = float32((k - width) / orig_freq)                  (functional.py:1376)
//   t        = float32(-p / new_freq) + idx[k]                   (functional.py:1378)
//   t       *= base_freq                                         (functional.py:1379)
//   t        = clamp(t, -lowpass_filter_width, lowpass_filter_width)  (functional.py:1380)
//   window   = cos(((t * pi) / lowpass_filter_width) / 2) ^ 2    (functional.py:1385, Hann)
//   t_rad    = t * pi                                            (functional.py:1393, reassigns t)
//   sinc     = t_rad == 0 ? 1.0 : sin(t_rad) / t_rad              (functional.py:1396)
//   kernel   = sinc * (window * scale)                           (functional.py:1397)
//
// Convolution (_apply_sinc_resample_kernel, functional.py:1420-1428): pad the
// input with `width` zeros on the left and `width + orig_freq` zeros on the
// right (functional.py:1424), then for output index j (0-based over the
// interleaved [block, phase] stream conv1d's transpose(1,2).reshape produces,
// functional.py:1425-1426), letting block = j / new_freq, phase = j %
// new_freq:
//   y[j] = sum_{k=0}^{taps-1} kernel[phase][k] * padded[block*orig_freq + k]
// target_length = ceil(new_freq * input_length / orig_freq)      (functional.py:1427)
// output is the first target_length samples of that interleaved stream
// (functional.py:1428) -- confirmed against the oracle's own measured pair:
// 336960 input samples -> 224640 output samples (336960 divides evenly by 3,
// so this particular case has no rounding ambiguity to probe, but the
// arithmetic above is exercised at a non-exact ratio by this module's own
// length-arithmetic tests, e.g. 961 -> 641).
//
// Returns false (leaving `output` empty) only when `input` is empty; every
// other length, including 1-3 samples, has a well-defined torchaudio output
// (measured against the pinned venv) and succeeds here too.
bool resample_24k_to_16k(const std::vector<float> & input, std::vector<float> & output);

}  // namespace synth::omnivoice
