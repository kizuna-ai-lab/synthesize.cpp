#pragma once

#include <cstdint>
#include <string>

namespace synth::omnivoice {

// How one catalog tensor is stored once a Quantization Profile is applied.
//
// This is the single source of truth for the split, shared by the offline
// quantizer (tools/synthesize-quantize) and this family's runtime catalog, so
// an offline packing decision and a load-time expectation cannot drift the way
// two hand-maintained lists could.
enum class QuantRole {
    Unknown,
    // Read through a matrix multiply, so it can carry a block-quantized type.
    MatrixWeight,
    // A transposed convolution kernel. This runtime runs it as a column
    // matrix multiply into col2im_1d, and CUDA's F16 matrix multiply
    // accumulates in half precision, so it can be halved but not
    // block-quantized -- the same override VITS and Qwen3-TTS's decoder make
    // (policy.cpp:390-395, docs/quantization.md:52-56).
    TransposeWeight,
    // Stays at the reference dtype. jiangzhuo's ruling of 2026-08-06 makes
    // this the whole generator -- every `llm.*` tensor,
    // `audio_embeddings.weight`, `audio_heads.weight` -- regardless of shape:
    // a reference port measured exact-token agreement collapsing from 100% to
    // roughly 7% with an F16 generator, and this family's headline claim is
    // exact tokens. The RVQ (`codec.quantizer.*`) joins it for the same
    // reason qwen3-tts's own RVQ stays exact (policy.cpp:258-265): a residual
    // codebook's later levels carry small magnitudes, so a relative error
    // there is large against the residual it corrects. The two concatenation
    // projections around the RVQ (`codec.fc`, `codec.fc2`) stay with it.
    // Every bias, every Snake alpha and every 1-D norm are read elementwise
    // rather than through a matrix multiply, so packing buys nothing there
    // either. Three further tensors are Sensitive for reasons specific to
    // their shape or to how this runtime reads them; see the comments beside
    // them in quantization.cpp -- a future reader must not "fix" those three.
    Sensitive,
};

// The family's tensor->role classifier, shared by tools/synthesize-quantize
// and the runtime so an offline decision and a load-time expectation cannot
// drift. Codec-only by jiangzhuo's ruling of 2026-08-06: every generator
// tensor is Sensitive, whatever its shape.
//
// `ne` is accepted for parity with the offline tool's per-tensor loop, which
// always has a ggml_tensor's shape on hand, and so that a future catalog
// tensor whose role genuinely depends on shape does not force an ABI change
// here. Every tensor this catalog resolves today classifies from its name
// alone: the two narrow-input convolutions and the grouped positional
// convolution are Sensitive because of what and how they are, an
// architectural fact rather than a hyperparameter that could vary at another
// scale, so the classifier names them rather than re-deriving the reason from
// `ne` at every call.
//
// Returns Unknown for any name outside the catalog. Callers must treat that
// as an error rather than assigning a fallback role, so a converter or
// runtime change cannot quietly acquire one.
QuantRole classify_tensor(const std::string & name, const int64_t ne[4]);

}  // namespace synth::omnivoice
