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

// The row length a matrix weight would have once flattened for the multiply.
//
// A convolution kernel is packed to [kernel * in_channels, out_channels]; a
// linear weight is already in that form. `quantized` says which of the two the
// caller is holding, since a packed tensor has lost its logical rank.
int64_t packed_row_length(const int64_t * ne, int dimensions, bool quantized);

// Whether a matrix weight can host block-quantized rows at all.
//
// Kokoro's decoder concatenates the two prosody curves and the narrow encoder
// residual onto its feature stream, which lands several channel counts two
// short of a multiple of thirty-two. Those weights fall back to the halved type
// rather than being excluded from the profile, and this predicate is shared so
// the offline tool and the runtime cannot disagree about which ones they are.
bool matrix_is_block_quantizable(const int64_t * ne, int dimensions, bool quantized, int64_t block_size);

}  // namespace synth::kokoro
