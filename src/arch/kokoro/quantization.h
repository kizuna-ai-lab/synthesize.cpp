#pragma once

#include <string>

namespace synth::kokoro {

// How one catalog tensor is stored once a Quantization Profile is applied.
//
// This is the single source of truth for the split, shared by the runtime's
// tensor catalog and by the offline quantizer. Keeping one classifier is not a
// convenience: a package is only loadable when the tool and the runtime agree
// tensor for tensor, and two hand-maintained lists would drift silently.
enum class TensorRole {
    Unknown,
    // Read through a matrix multiply, so it can carry a block-quantized type.
    MatrixWeight,
    // A transposed convolution kernel. This runtime slices and permutes it, so
    // it can be halved but not block-quantized.
    TransposeWeight,
    // Stays at the reference dtype. Two reasons appear here: the value is read
    // elementwise rather than through a matrix multiply, or it sits on the path
    // that decides F0 and the durations.
    //
    // The second reason is specific to this family. The harmonic source
    // accumulates phase across the whole utterance, so a relative F0 difference
    // of a few parts in ten thousand becomes radians by the end. PL-BERT, the
    // prosody and duration path, the acoustic text encoder, and the Voice
    // tables therefore stay exact at every profile; the decoder and its
    // generator, which are most of the parameters, are what gets quantized.
    Sensitive,
};

// Returns Unknown for any name outside the catalog. Callers must treat that as
// an error rather than assigning a fallback type, so that a converter or
// runtime change cannot quietly acquire one.
TensorRole tensor_role(const std::string & name);

}  // namespace synth::kokoro
