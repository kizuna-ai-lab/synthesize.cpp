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
    // Read through a matrix multiply, so it *may* carry a block-quantized
    // type -- this role is about how a tensor is read, not a promise that
    // every consumer already handles a packed one. 85 of these tensors reach
    // a matrix multiply through ggml_im2col, whose CPU implementation
    // aborts on any destination type besides F16/F32
    // (ggml/src/ggml-cpu/ops.cpp's ggml_compute_forward_im2col); this
    // family's conv1d builders (codec.cpp's codec_conv1d, reference-
    // encoder.cpp's conv1d) carry the same packed-safe F32 destination VITS
    // and Kokoro use (src/arch/vits/operations.cpp:37-43,
    // src/arch/kokoro/operations.cpp:46-79), ported by Plan 4 Task 2, so
    // ggml_im2col never sees a quantized destination type. See
    // quantization.cpp's classify_codec_matrix_region for which tensors this
    // covers. All 197 of the generator's own MatrixWeight tensors are plain
    // `ggml_mul_mat` operands with no convolution anywhere in that half, so
    // none of the above applies to them; see classify_generator_tensor.
    MatrixWeight,
    // Read by `ggml_get_rows` rather than by a matrix multiply. Exactly two
    // tensors in this family are: `llm.embed_tokens.weight` and
    // `audio_embeddings.weight` (generator.cpp:123, 136). The role exists
    // because CUDA's GET_ROWS accepts a strictly narrower set of types than
    // its matrix multiply does -- F16/F32/BF16/I32/Q1_0/Q4_0/Q4_1/Q5_0/Q5_1/
    // Q8_0 and no k-quant at all (ggml/src/ggml-cuda/ggml-cuda.cu:5190-5207)
    // -- so a profile that puts a k-quant here does not fail loudly. It
    // silently drops those nodes to the CPU, which
    // scripts/validate-omnivoice-replay.py's placement proof then reports as
    // a CUDA gate failure naming the tensor. A profile therefore gives this
    // role its own type (Profile::row_lookup_type) instead of the matrix
    // weight type, and the tables are never packed: `ggml_get_rows` indexes
    // whole rows of the tensor's declared shape.
    //
    // Under Q8_GEN the row-lookup type and the matrix weight type are both
    // Q8_0 and this role changes nothing. Under Q4_K_GEN they diverge --
    // Q4_K for the matrices, Q8_0 for these two tables -- which is the case
    // the role exists for and the one that measures the difference: the
    // pin costs 81,856,512 bytes and is what keeps the whole generator on the
    // accelerator.
    RowLookup,
    // A transposed convolution kernel. This runtime runs it as a column
    // matrix multiply into col2im_1d, and CUDA's F16 matrix multiply
    // accumulates in half precision, so it can be halved but not
    // block-quantized -- the same override VITS and Qwen3-TTS's decoder make
    // (policy.cpp:390-395, docs/quantization.md:52-56).
    TransposeWeight,
    // Stays at the reference dtype. The RVQ (`codec.quantizer.*`) is here for
    // the same reason qwen3-tts's own RVQ stays exact (policy.cpp:258-265): a
    // residual codebook's later levels carry small magnitudes, so a relative
    // error there is large against the residual it corrects. The two
    // concatenation projections around the RVQ (`codec.fc`, `codec.fc2`) stay
    // with it. Every bias, every Snake alpha and every 1-D norm are read
    // elementwise rather than through a matrix multiply, so packing buys
    // nothing there either -- which covers the generator's RMS norms and its
    // per-head q/k norms as well as the codec's. Three further tensors are
    // Sensitive for reasons specific to their shape or to how this runtime
    // reads them; see the comments beside them in quantization.cpp -- a
    // future reader must not "fix" those three.
    //
    // History, because the role assignment above changed and a reader will
    // otherwise find the old rule quoted in shipped packages' documentation:
    // jiangzhuo's ruling of 2026-08-06 made the *whole* generator Sensitive
    // whatever its shape, on the grounds that a reference port measured
    // exact-token agreement collapsing from 100% to roughly 7% with an F16
    // generator and that this family's headline claim was exact tokens. That
    // ruling was revised on 2026-08-09: a quantized generator produces a
    // different valid realization rather than a wrong one -- this family
    // already ships a CUDA generator that flips 94-98% of greedy tokens and
    // was accepted by ear -- so exact-token agreement is recorded as data and
    // the ship decision rests on the speaker-identity F0 proxy instead. The
    // generator is 76.9% of the package's tensor bytes, so no codec-only
    // profile can reach the sizes this family needs.
    Sensitive,
};

// Which half of the package a tensor belongs to. Every Quantization Profile
// this family cuts quantizes exactly one half and holds the other at the
// reference dtype, because the two halves fail differently: the codec's
// error shows up as measurable drift against a fixed reference grid, while
// the generator's shows up as a different realization that only a listening
// instrument can judge. Keeping one half exact per profile is what makes each
// package's evidence attributable to one cause.
enum class ModelHalf {
    // The mask-predict generator: `llm.*` plus the two canvas tables and the
    // audio heads. 2,450,309,120 of the F32 package's 3,184,565,636 tensor
    // bytes.
    Generator,
    // Everything under `codec.` -- the Higgs Audio V2 acoustic decoder, the
    // acoustic encoder, HuBERT and the RVQ.
    Codec,
};

// The half `name` belongs to, from the name alone. Note this answers a
// name-space question and nothing else: it is meaningful only for a name
// classify_tensor already recognises, which is why classify_tensor_for_half
// below resolves the role first.
ModelHalf tensor_half(const std::string & name);

// The family's tensor->role classifier, shared by tools/synthesize-quantize
// and the runtime so an offline decision and a load-time expectation cannot
// drift. This reports the *architectural* role -- how the graph reads the
// tensor -- for both halves of the package, with no profile in view. What a
// given profile then does with that role is
// classify_tensor_for_half's question, not this one's.
//
// `ne` is accepted for parity with the offline tool's per-tensor loop, which
// always has a ggml_tensor's shape on hand, and so that a future catalog
// tensor whose role genuinely depends on shape does not force an ABI change
// here. Every tensor this catalog resolves today classifies from its name
// alone: the grouped positional convolution is Sensitive because of how this
// runtime reads it (see quantization.cpp), an architectural fact that cannot
// vary at another scale, and the two narrow-input convolutions are Sensitive
// because reading one input channel is likewise architectural -- though for
// one of the two, the kernel width that makes its packed row fail a
// block-size check is a checkpoint hyperparameter, not itself architectural
// (quantization.cpp says which is which). Either way the classifier names
// these tensors rather than re-deriving the reason from `ne` at every call.
//
// Returns Unknown for any name outside the catalog. Callers must treat that
// as an error rather than assigning a fallback role, so a converter or
// runtime change cannot quietly acquire one.
QuantRole classify_tensor(const std::string & name, const int64_t ne[4]);

// The role `name` takes under a profile that quantizes `quantized_half`:
// classify_tensor's architectural role for a tensor in that half, and
// Sensitive for every tensor in the other one. This is the single place the
// half split is expressed, read by both the offline quantizer's dispatch
// (tools/synthesize-quantize/policy.cpp) and the runtime catalog's load-time
// expectation (catalog.cpp's expected_type), so an offline packing choice and
// a load-time check cannot drift.
//
// Unknown propagates ahead of the half test rather than being flattened into
// Sensitive. A stray name is fatal under every profile, including one that
// leaves the half it would land in untouched.
QuantRole classify_tensor_for_half(const std::string & name, const int64_t ne[4], ModelHalf quantized_half);

}  // namespace synth::omnivoice
