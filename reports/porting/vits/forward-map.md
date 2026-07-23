# VITS Forward Map

Status: LJSpeech and VCTK Stage 4 C++ implementations validated on CPU and CUDA;
Stage 5 numerical acceptance remains open on 2026-07-23.

Canonical reference: `jaywalnut310/vits` commit
`2e561ba58618d021b5b8323d3765880f7e0ecfdb`. In-tree GGML patterns come
from pinned transcribe.cpp revision
`8c7ae674ea7b26b0c9074529da99f938553db32f`; they are structural examples,
not semantic authorities.

## Inference Sequence

| Stage | Canonical reference | Planned C++ / GGML expression | Gate |
| --- | --- | --- | --- |
| Token embedding and mask | `models.py:TextEncoder.forward`, `commons.py:sequence_mask` | Implemented: `ggml_get_rows`, F32 scale by `sqrt(hidden)`, checked single-sequence mask | `text.mask` exact in 12/12 cases |
| Text attention stack | `attentions.py:Encoder.forward`, `MultiHeadAttention` | Implemented: six post-norm residual blocks; 1x1 Conv1d projections; ordinary and relative key/value matmuls | `text.m_p` / `text.logs_p` measured in 12/12 cases |
| Text FFN | `attentions.py:FFN.forward` | Implemented: same-padded F32 Conv1d, ReLU, same-padded F32 Conv1d | `text.m_p` / `text.logs_p` measured in 12/12 cases |
| Prior projection | `models.py:TextEncoder.forward` | Implemented: 1x1 F32 Conv1d and channel split | `text.m_p`, `text.logs_p` measured in 12/12 cases |
| Stochastic duration | `models.py:StochasticDurationPredictor.forward(reverse=True)` | Implemented: detached text hidden state, pre/DDS/projection conditioning, caller-supplied replay noise, four flips, three inverse ConvFlows, inverse affine | `duration.logw` measured in 12/12 cases |
| Duration path | `models.py:SynthesizerTrn.infer`, `commons.py:generate_path` | Implemented at the dynamic host seam: semantic speaking-rate transform, F32 exp/ceil, output-limit check, monotonic path generation | `duration.w_ceil`, `duration.y_length`, `duration.attention` exact in 12/12 cases |
| Expanded prior | `models.py:SynthesizerTrn.infer` | Implemented: attention matmuls in a second static-shape GGML graph after host resolves `Y` | `prior.m_p_expanded`, `prior.logs_p_expanded` measured in 12/12 cases |
| Latent sampling | `models.py:SynthesizerTrn.infer` | Implemented: caller-replayed logical latent noise, layout Adapter, portable GGML `m + noise * exp(logs) * noise_scale` graph | `latent.z_p` measured in 12/12 cases |
| Acoustic flow | `models.py:ResidualCouplingBlock.forward(reverse=True)` | Implemented: four full-channel flips and four reverse mean-only residual coupling blocks with gated WN | `flow.z` measured in 12/12 cases |
| Waveform decoder | `models.py:Generator.forward` | Implemented: four cropped transpose-convolution stages, three ResBlock1 branches per stage, final convolution and tanh | `audio.pcm` measured in 12/12 cases |

## Text Encoder Detail

| Operation | Shape in C++ (`ne[0]` first) | Important invariant |
| --- | --- | --- |
| Token IDs | `[T]` I32 | `1 <= T <= 512`; every ID is in `[0, 178)` |
| Embedding output | `[192, T]` F32 | multiply checkpoint rows by `sqrt(192)` |
| Q/K/V projections | `[192, T]` | Conv1d kernel 1, including bias |
| Head view | `[96, T, 2]` | scaling is `1/sqrt(96)` before both key dot products |
| Relative lookup | `[96, T_key, T_query]` | indices outside relative window 4 select an appended zero row |
| Attention scores | `[T_key, T_query, 2]` | ordinary plus relative-key scores; softmax over key dimension |
| Relative value term | `[96, T_query, 2]` | query-dependent embedding lookup and weighted key reduction |
| LayerNorm | `[192, T]` | post-residual norm over channel dimension, epsilon `1e-5` |
| FFN hidden | `[768, T]` | Conv1d kernel 3, symmetric one-frame padding, ReLU |
| Encoder output | `[192, T]` | mask after final block |
| Prior stats | `[384, T]` | split into `m_p` and `logs_p`, each `[192, T]` |

The relative embedding checkpoint has nine learned rows. For an input of length
`T`, host code supplies a checked I32 lookup matrix for relative distance
`key - query + 4`; distances outside `[-4, 4]` select a graph-local zero row.
The learned rows and zero row then feed batched GGML matrix multiplications for
both the relative-key logits and relative-value output. This preserves the
canonical query-dependent math without a family-specific backend kernel.

## Variant Notes

The `vits-ljspeech` variant is batch-one, single-speaker, has no global
conditioning, uses stochastic duration prediction, six text blocks, two heads,
relative window four, decoder resblock type 1, and token-ID-only core input.
`vits-vctk` keeps that architecture and adds a required selected row from its
109-by-256 speaker table. Stable public IDs map directly to upstream indices.
The selected vector is projected into duration conditioning after `dp.pre`, into
each acoustic-flow WN block, and into the decoder after `dec.conv_pre`. Selection
is a graph input owned by the private `voice-conditioning` module, so CPU and CUDA
share one semantic graph and no VITS-specific index enters the public Interface.

## Stochastic-Duration Detail

The parity seam accepts the unscaled recorded `random.duration_noise` tensor as
logical channel-major `[2, T]` F32 plus `noise_scale_w`. Host preparation checks
that it contains exactly `2*T` finite values and converts it to GGML's physical
channel-fastest `[2, T]` layout. This is an internal validation seam, not the
future public seed interface.

The canonical upstream flow list contains an affine followed by four ConvFlow and
flip pairs. Reverse inference reverses that list, removes the first/unused forward
ConvFlow, and therefore executes exactly:

```text
flip -> ConvFlow 2 -> flip -> ConvFlow 1 ->
flip -> ConvFlow 0 -> flip -> inverse affine
```

Each ConvFlow splits `[2, T]` into one conditioning channel and one transformed
channel. Its pre-convolution, three-layer dilated depth-separable DDSConv, and
29-channel projection produce ten widths, ten heights, and nine internal
derivatives for a ten-bin rational-quadratic spline. Linear tails outside
`[-5, 5]`, endpoint derivative one, and minimum width, height, and derivative
`1e-3` match the pinned reference.

The inverse spline is expressed with backend-portable GGML operations: softmax,
cumulative sum, softplus, step-derived bin selection, reductions, and the
quadratic inverse. F32 depthwise convolution uses an explicit F32
`im2col + mul_mat` decomposition with dilation 1, 3, and 9. No host-side spline
loop or backend-specific operator is allowed.

## Duration-Path Detail

`duration.logw` has static shape `[1,T]`, but its rounded sum determines the
runtime acoustic length `Y` and therefore the shape of every later graph. The
library resolves this at the host seam between GGML graphs instead of exposing
dynamic-shape mechanics to callers or adding a backend-specific path operator.
The Module maps semantic `speaking_rate` to upstream `length_scale = 1/rate`,
computes F32 `ceil(exp(logw) * length_scale)`, and clamps the summed duration to
at least one frame.

The GGUF's output limit is in native PCM frames. Duration resolution therefore
requires `Y * hop_length <= max_output_frames` before allocating the dense path.
The returned attention uses logical `[Y,T]` layout with token index contiguous;
each nonzero-duration token owns one consecutive frame interval. Zero total
duration retains upstream's one-frame mask with an all-zero attention path.

## Expanded-Prior Detail

The duration graph now retains `text.m_p` and `text.logs_p` as additional outputs,
so Model execution computes the text encoder and duration predictor only once.
After the host seam resolves `Y`, a second backend-independent GGML graph accepts
physical `[C,T]` text statistics and `[T,Y]` attention. It evaluates the upstream
`attention[Y,T] @ prior[T,C]` projection for both tensors and retains channel-
fastest `[C,Y]` storage for later inference stages.

The validation Adapter converts only the observable output to PyTorch's logical
channel-major layout. Because the exact monotonic attention merely selects and
repeats token statistics, the projection introduces no additional worst-case
drift beyond the text encoder.

## Latent-Sampling Detail

The oracle records `random.latent_noise` after upstream allocates it with
`empty_like(expanded_m_p)`. Although that tensor inherits the non-contiguous
expanded-prior strides during random assignment, the artifact stores its final
logical `[C,Y]` values contiguously. The parity interface therefore accepts the
recorded logical values directly instead of attempting to reproduce PyTorch's
generator or stride-sensitive assignment inside the library.

The host Adapter validates the prior statistics, noise, shape, finite values, and
non-negative `noise_scale`, then converts all three inputs to GGML's physical
channel-fastest layout. A static-shape, backend-independent graph evaluates
`m_p + latent_noise * exp(logs_p) * noise_scale`; only the observable result is
converted back to logical channel-major storage. The supplied noise makes this
internal C++ interface deterministic for C++, Python, and Rust adapters while the
future public seed-to-random-stream contract remains independent.

## Reverse Acoustic-Flow Detail

The strict flow catalog loads all 80 `flow.*` tensors: four coupling blocks,
each with pre/projection convolutions and four WN layers. Every WN layer applies
a same-padded kernel-5 F32 convolution, tanh/sigmoid gating, and a residual/skip
projection. Layers zero through two split their projection between the next
hidden state and accumulated skip output; the final layer contributes only skip.
The LJSpeech Model Variant is explicitly validated as mean-only and unconditioned.

Reverse inference walks the upstream alternating flow list in reverse:
`flip -> block 3 -> flip -> block 2 -> flip -> block 1 -> flip -> block 0`.
Each flip reverses all 192 channels using a shared I32 row index tensor; it does
not merely exchange the two 96-channel halves. Each coupling splits after that
flip and computes `second = second - mean`. Since the runtime sequence is batch
one and its resolved acoustic length is exactly `Y`, the upstream `y_mask` is all
ones and need not be exposed at this internal seam.

The graph uses only backend-portable row lookup, transpose, F32
`im2col + mul_mat`, tanh, sigmoid, addition, subtraction, concatenation, and
contiguous-layout operations. Metadata validation proves every dilation and
padding fits the GGML integer interface before graph construction.

## Current Gate

The complete source-F32 CPU path now runs across all 12 manifest cases.
Worst `duration.logw` F32 CPU drift is
`1.7452985e-5` max absolute and `1.2574333e-6` mean absolute on `ljs-long`; all
rounded durations, output lengths, and attention elements are exact. Expanded
prior drift remains bounded by the selected text values: `1.2636185e-5` for
`m_p` and `4.7087669e-6` for `logs_p`. Worst `latent.z_p` drift is
`1.5258789e-5` max absolute and `1.3051576e-7` mean absolute on
`ljs-rate-slow`. Reverse flow amplifies accumulated and convolution drift;
worst `flow.z` drift is `6.6339970e-5` max absolute and `1.1114858e-6` mean
absolute on `ljs-rate-slow`. The final waveform has worst max-absolute drift
`2.8620008e-4` and mean-absolute drift `1.7319487e-6` on `ljs-long`; all output
lengths equal `Y * 256` and every sample is finite.

Stage 4 is complete: no inference row remains unimplemented, all seven 12-case
tensor/waveform paths pass, and the public C Interface plus reference CLI reach
the same CPU implementation. Stage 5 remains open because these measurements
have deliberately not been converted into accepted numerical thresholds; this
is separate from the deferred perceptual Quality Evaluation Suite.

Twenty-nine C/C++ unit executables and 18 Python unit cases cover the completed
behavior. All 30 registered unit tests and ten integration tests pass in Release.
The same unit suite, two public C Interface tests, and reference-CLI test pass in
ASan/UBSan builds; all seven Golden paths have also completed their documented
sanitizer gates.
