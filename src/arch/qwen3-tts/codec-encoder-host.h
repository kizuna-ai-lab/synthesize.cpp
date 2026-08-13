#pragma once

#include "synthesize.h"

#include <cstdint>
#include <vector>

namespace synth::qwen3tts {

struct HParams;
struct CodecEncoderWeights;

// THE CODE GRID'S LAYOUT, AND THE TRANSPOSE THAT MUST NEVER BE ADDED.
//
// `CodecEncoding::codes` is GGML index order [16, T]: `ne[0] = 16` is the
// quantizer-group index and is the FASTEST-varying dimension, `ne[1] = T` is
// the frame index. Every structure downstream uses that and only that.
//
// The oracle spells the same bytes differently, and the two spellings look like
// a conflict without being one. Its dumper asserts its array is numpy-shaped
// `[frames, 16]` (dump_reference_qwen3_tts_base.py:420-426) and writes it with
// `np.ascontiguousarray(...).tobytes()` (:152-156), i.e. C-order: sixteen codes
// contiguous per frame, frame index slowest. GGML `[16, T]` with `ne[0] = 16`
// describes THAT IDENTICAL BYTE SEQUENCE; numpy names its slowest axis first
// and GGML names its fastest axis first. So `codes.i32` and this buffer compare
// as a flat memcmp of `16 * T` int32s, and any code that inserts a transpose to
// reconcile the two spellings has just broken the comparison. THERE IS NO
// TRANSPOSE ANYWHERE, and the next reader who meets the two spellings will
// reach for the same wrong fix.
//
// The failure this actually guards against is not a spelling: it is
// STAGE-MAJOR storage -- all `T` codes of quantizer stage 0, then all of stage
// 1 -- which is what a reader who takes `[16, T]` for a numpy shape would
// build. That buffer has the right element count, the right value range and the
// wrong order, and on a 16-frame clip it even has the right shape.
//
// WHAT ONE REFERENCE CLIP TURNS INTO.
//
// Everything past `frames` is a by-product of the same single pass, kept
// because the comparison against the oracle is stage-wise now that the
// plain-equality gate on codes is gone (the design's fourth erratum,
// 2026-08-13): the codebook the converter bakes is float32 and the oracle's is
// bfloat16, they disagree on 4.04% of frame-stage decisions, and upstream
// disagrees with ITSELF depending on whether `dtype` is passed at load. So the
// codes cannot be the gate, and every continuous quantity behind them has to be
// comparable instead. None of it costs a second pass: the residuals and the
// reconstruction are what the argmin already walks over.
struct CodecEncoding {
    // `groups * frames` values, GROUP-FASTEST -- the layout described above.
    std::vector<int32_t> codes;
    // `quantizer_count`: 1 semantic + 15 acoustic. NOT the 32 stages the
    // checkpoint carries; the encoder's acoustic cascade runs deeper than
    // anything reads back (catalog.cpp's own resolve_codec_encoder comment).
    uint64_t             groups = 0;
    // `ceil(samples / samples_per_frame)`, after the trim.
    uint64_t             frames = 0;

    // The pre-quantization latents, `latent_width * frames`, CHANNEL-FASTEST
    // (frame-major) -- the graph's own [latent_width, frames] order, untouched.
    // The oracle's `latents.f32` is the transpose of this, being C-order
    // `[channels, frames]`; a comparison transposes one of them.
    std::vector<float> latents;
    uint64_t           latent_width = 0;

    // The width of the space the quantizer decides in: `codebook_dim / 2`, 256
    // at the production geometry. Both branches project the SAME latents into
    // their own copy of it, and the residual, the codebook rows and the
    // reconstruction all live there. `output_proj` is a decode-time tensor and
    // is never read on this path.
    uint64_t projected = 0;

    // The residual ENTERING each of the `groups` kept stages, laid out
    // `[groups][frames][projected]` with `projected` fastest -- byte-identical,
    // slice for slice, to the oracle's `rvq_residual_s00..s15.f32`, each of
    // which is C-order `[frames, projected]`.
    //
    // Slice 0 is the semantic branch's `input_proj`-ed latent; slice 1 is the
    // ACOUSTIC branch's `input_proj`-ed latent and not the semantic leftover,
    // because the two branches are fed the same latents (modeling_mimi.py's
    // MimiSplitResidualVectorQuantizer.encode passes `embeddings` to both).
    // Slices 2..15 are the acoustic cascade's own successive residuals.
    std::vector<float> residuals;

    // The dequantized reconstruction: the sum of the selected codebook rows, in
    // the projected space, PER BRANCH -- `[2][frames][projected]`, semantic
    // first, matching the oracle's `rvq_reconstruction.f32` element for
    // element.
    //
    // Per branch and not summed into one, because the two are an order of
    // magnitude apart (measured: rms 13.5 against 3.11, explaining 43% and 85%
    // of their targets) and one scalar tolerance does not fit both. This is the
    // quantity the gate compares: it is the discrete decision converted back
    // into the continuous thing it was rounded from, so a near-tie flip moves
    // it by about that decision's margin while a wrong stride, a wrong padding
    // mode, or an acoustic branch chained onto the semantic residual moves it
    // by orders of magnitude more.
    std::vector<float> reconstruction;

    // This port's own silent-clip gate, not an upstream quantity -- the same
    // role XVectorEncoding::ref_rms plays for the speaker path.
    float ref_rms = 0.0f;

    // Margin instrumentation, DIAGNOSTIC ONLY. `narrowest_gap` is the smallest
    // best-vs-second-best squared-distance gap over every (group, frame)
    // decision; `gaps` is the full grid.
    //
    // `gaps` shares `codes`' GROUP-FASTEST layout, so the two index alike. The
    // oracle's `rvq_distance_margin.f32` is the other way round -- C-order
    // `[16, frames]`, i.e. stage-major -- so a comparison transposes THAT one.
    // Stated because the symmetry a reader would assume from the residual and
    // reconstruction dumps, which do match slice for slice, is not there.
    float              narrowest_gap = 0.0f;
    std::vector<float> gaps;
};

// Whether `encoding` holds exactly the `groups * frames` codes at `flat`, in
// `encoding`'s own group-fastest order.
//
// A PRODUCTION ENTRY POINT WITH THREE CALLERS, deliberately, rather than a
// helper local to any one of them: the layout assertion in
// tests/qwen3_tts_codec_encoder_test.cpp, the serialize/load round trip, and
// the re-preparation identity check all drive this same function, so inverting
// it fails all three. A test-local copy would pin a property of the test.
//
// This survived the fourth erratum and its callers changed. It is no longer the
// oracle gate -- the stage-wise comparison is -- but every remaining caller
// compares the port against ITSELF, where exactness is free and mandatory. Do
// not repurpose it into an oracle comparison, and do not delete it because one
// caller went away.
//
// False for a null `flat`, a `frames` that disagrees with `encoding.frames`, an
// unset `groups`, or a `codes` buffer whose size is not `groups * frames`.
bool codes_equal(const CodecEncoding & encoding, const int32_t * flat, int64_t frames);

// PCM at the package's declared reference rate to the [16, T] reference code
// grid: the codec encoder graph, then the split residual vector quantizer on
// the host, one shot on the CPU.
//
// THE QUANTIZER IS ON THE HOST BECAUSE ITS OUTPUT IS DISCRETE. That is this
// project's standing placement rule (docs/backends.md), and
// `omnivoice::rvq_encode` (src/arch/omnivoice/reference-encoder-host.h:296-366)
// states the reasoning and is the template for the tie rule and the
// accumulation decision below. The graph half stays on the CPU for the separate
// reason encode_speaker_reference records: this encoder is resolved against the
// package context and never a twin, so its weights are CPU-resident.
//
// THE TIE RULE: ties resolve to the LOWEST id. Upstream's own
// `dists.argmin(dim=-1)` (modeling_mimi.py's MimiEuclideanCodebook.quantize)
// returns the first minimal index, so the scan below is strictly `<` over `e`
// ascending and a later equal candidate never displaces an earlier one.
//
// THE ACCUMULATION: plain left-to-right float32, no Neumaier and no double
// intermediate -- and here that is a match rather than a concession. Upstream
// computes `torch.cdist(hidden[None].float(), embed[None].float(), p=2)`, whose
// default compute mode uses the matmul-expanded form `||z||^2 - 2 z.E +
// ||E||^2` above 25 rows, which is what every clip here has. The port computes
// that same expansion, in float32, as upstream does.
//
// Distances are compared SQUARED. `argmin` is unchanged by the monotone square
// root, and the reported gap `d2^2 - d1^2` is exactly the oracle's own
// `(d2 - d1) * (d2 + d1)` (dump_reference_qwen3_tts_codec_encoder.py:1942-1956)
// rather than a different quantity that happens to order the same way.
//
// `threads` follows encode_speaker_reference's convention: 0 selects
// default_synthesis_threads(). On any non-OK return `output` is left cleared.
// A digitally silent clip is refused by name with
// "voice_profile.reference_silent", the same code and the same reasoning the
// speaker path already uses.
synth_status_t encode_codec_reference(const HParams &             hparams,
                                      const CodecEncoderWeights & weights,
                                      const std::vector<float> &  pcm,
                                      int                         threads,
                                      CodecEncoding &             output,
                                      const char *&               out_diagnostic_code,
                                      const char *&               out_diagnostic_message);

}  // namespace synth::qwen3tts
