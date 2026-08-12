#pragma once

#include "catalog.h"

#include <cstdint>

struct ggml_context;
struct ggml_tensor;

namespace synth::qwen3tts {

// The deepest reflection any convolution in this graph performs, in frames.
//
// Every convolution is PyTorch's `padding="same", padding_mode="reflect"`, so a
// kernel `k` at dilation `d` reflects `d * (k - 1) / 2` frames off each edge,
// and a reflection cannot reach past the signal it reflects. The widest is the
// third SE-Res2Net block's 3-wide kernel at dilation 4, which reflects 4.
//
// This is upstream's own domain limit, not a limit this port adds:
// `F.pad(..., mode="reflect")` raises on a shorter input, so the reference
// implementation cannot encode such a clip either. At the package's own
// hop_length of 256 it excludes reference audio under roughly 53 ms.
constexpr int64_t kSpeakerEncoderDeepestReflection = 4;

// ECAPA-TDNN over a [mel_bins, frames] F32 mel, producing a [enc_dim] F32
// speaker embedding.
//
// `mel` is channel-major -- ne[0] is the bin count, ne[1] the frame count --
// which is both the layout MelSpectrogram hands over (see mel.h) and the layout
// the rest of this family's convolution stacks use.
//
// Returns nullptr rather than aborting on: a null argument; a mel that is not a
// 2-D F32 tensor; a mel whose bin count disagrees with the stem convolution's
// declared input extent; a mel with `kSpeakerEncoderDeepestReflection` frames or
// fewer; a width that disagrees between two adjacent stages; or a weights struct
// with a null pointer in it -- the last of which is only reachable if a resolver
// dropped something, which tests/qwen3_tts_catalog_test.cpp pins.
//
// Every width is read off the weights themselves rather than named here, so a
// package whose metadata and tensors disagree has already been refused by
// catalog.cpp's resolver and never reaches this graph with a shape to guess at.
//
// The dilations, activations and pooling details are transcribed from the pinned
// upstream source and recorded, with line numbers, in the Task 1 oracle's
// conventions.json (`ecapa_topology`) and in speaker-encoder.cpp's own header
// comment. None of them is declared in the package.
ggml_tensor * build_speaker_encoder(ggml_context * context, const SpeakerEncoderWeights & weights, ggml_tensor * mel);

}  // namespace synth::qwen3tts
