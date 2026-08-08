#include "quantization.h"

#include <initializer_list>
#include <string_view>
#include <vector>

namespace synth::omnivoice {

namespace {

std::vector<std::string_view> split_name(const std::string & name) {
    std::vector<std::string_view> tokens;
    size_t                        begin = 0;
    while (begin <= name.size()) {
        const size_t end = name.find('.', begin);
        tokens.emplace_back(name.data() + begin, (end == std::string::npos ? name.size() : end) - begin);
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return tokens;
}

bool one_of(std::string_view value, std::initializer_list<std::string_view> choices) {
    for (std::string_view choice : choices) {
        if (value == choice) {
            return true;
        }
    }
    return false;
}

bool is_index(std::string_view value) {
    if (value.empty() || (value.size() > 1 && value.front() == '0')) {
        return false;
    }
    for (char character : value) {
        if (character < '0' || character > '9') {
            return false;
        }
    }
    return true;
}

// One Qwen3 decoder block. The projections are the parameters; the norms are
// four narrow vectors a profile gains nothing by quantizing and can lose
// accuracy to. Modelled on qwen3-tts's own classify_qwen3_block
// (tools/synthesize-quantize/policy.cpp:171-191), which handles the identical
// block shape -- with one name difference this family must not paper over:
// omnivoice's second norm is `post_attention_layernorm`, the Hugging Face
// name, where qwen3-tts's converter renamed it `post_attn_norm`. Each
// classifier matches what its own converter actually emits.
QuantRole classify_generator_block(const std::vector<std::string_view> & tokens, size_t offset) {
    if (tokens.size() == offset + 2 && tokens[offset + 1] == "weight" &&
        one_of(tokens[offset], { "input_layernorm", "post_attention_layernorm" })) {
        return QuantRole::Sensitive;
    }
    if (tokens.size() == offset + 3 && tokens[offset] == "self_attn" && tokens[offset + 2] == "weight") {
        if (one_of(tokens[offset + 1], { "q_proj", "k_proj", "v_proj", "o_proj" })) {
            return QuantRole::MatrixWeight;
        }
        // Qwen3 normalizes each head at head_dim, not the packed projection:
        // 128 values against a projection's two million, applied to every
        // head before rope.
        if (one_of(tokens[offset + 1], { "q_norm", "k_norm" })) {
            return QuantRole::Sensitive;
        }
        return QuantRole::Unknown;
    }
    if (tokens.size() == offset + 3 && tokens[offset] == "mlp" && tokens[offset + 2] == "weight" &&
        one_of(tokens[offset + 1], { "gate_proj", "up_proj", "down_proj" })) {
        return QuantRole::MatrixWeight;
    }
    return QuantRole::Unknown;
}

// The mask-predict generator: the Qwen3 layer stack (`llm.*`) plus its two
// canvas tables, which are a second table each rather than a tied pair, plus
// the audio heads.
//
// Two facts hold for every MatrixWeight tensor this function returns, and
// neither holds for the codec's:
//
//   * All 199 two-dimensional generator weights have rows (ne[0]) of 1024,
//     2048 or 3072 -- divisible by 32 and by 256 -- so Q8_0 and every k-quant
//     clear the block-size check in
//     tools/synthesize-quantize/quantize.cpp without any per-tensor
//     exception. The codec needed three such exceptions; this half needs
//     none.
//   * The generator graph (generator.cpp) emits no convolution at all, so
//     there is no ggml_im2col destination-type hazard here -- the whole
//     reason the codec's 85 convolution kernels needed Plan 4 Task 2's packed
//     branch. Every generator matrix is a plain `ggml_mul_mat` operand, and
//     rope takes no weight operand.
//
// Where they are read is what splits the two tables from the rest:
// `llm.embed_tokens.weight` and `audio_embeddings.weight` reach
// `ggml_get_rows` (generator.cpp:123, 136) and are RowLookup, while
// `audio_heads.weight` -- the same shape, and easy to assume is the tied
// transpose of the embeddings -- is a plain `ggml_mul_mat`
// (generator.cpp:193) and is an ordinary MatrixWeight.
QuantRole classify_generator_tensor(const std::vector<std::string_view> & tokens, const std::string & name) {
    if (name == "audio_embeddings.weight") {
        return QuantRole::RowLookup;
    }
    if (name == "audio_heads.weight") {
        return QuantRole::MatrixWeight;
    }
    // "llm.*" requires at least one child segment; a bare "llm" names nothing
    // in the catalog and falls through to Unknown like any other stray name.
    if (tokens.size() < 2 || tokens[0] != "llm") {
        return QuantRole::Unknown;
    }
    if (name == "llm.embed_tokens.weight") {
        // Tied to the text head, so no `lm_head` exists to classify beside it
        // (catalog.cpp:321-322).
        return QuantRole::RowLookup;
    }
    if (tokens.size() == 3 && tokens[1] == "norm" && tokens[2] == "weight") {
        return QuantRole::Sensitive;
    }
    if (tokens.size() > 3 && tokens[1] == "layers" && is_index(tokens[2])) {
        return classify_generator_block(tokens, 3);
    }
    return QuantRole::Unknown;
}

// codec.quantizer.*, codec.fc.* and codec.fc2.*: the RVQ and the two
// projections that concatenate the acoustic and semantic latents around it.
// The RVQ rule matches qwen3-tts's own (policy.cpp:258-265): a residual
// codebook's later levels carry small magnitudes, so a relative error there
// is large against the residual it corrects, and the whole quantizer here is
// smaller still (40 tensors against qwen3-tts's own codebook set).
bool is_blanket_sensitive_codec_prefix(const std::vector<std::string_view> & tokens) {
    return tokens.size() >= 2 && one_of(tokens[1], { "quantizer", "fc", "fc2" });
}

// The HuBERT stack's Linear modules, and the whole of what this family
// block-quantizes in the codec since the conv-exempt policy of 2026-08-09.
// Whitelisted positively -- these seven module names, under
// `codec.semantic_model` and nowhere else -- rather than by excluding the
// convolutions, so that a catalog tensor nobody anticipated lands on the
// unquantized side. Together they are exactly 73 tensors: four attention
// projections and two feed-forward matrices across each of the twelve
// encoder layers, plus `feature_projection.projection`.
//
// A shape test cannot do this job and must not be substituted for it. The
// convolution kernel `codec.acoustic_decoder.conv2.weight` is [7, 32, 1], so
// its ne is indistinguishable from a Linear's -- `ggml_n_dims` reports it as
// two-dimensional and its ne[2] is 1 either way.
bool is_semantic_model_linear(const std::vector<std::string_view> & tokens) {
    return tokens.size() >= 3 && tokens[1] == "semantic_model" &&
           one_of(tokens[tokens.size() - 2],
                  { "q_proj", "k_proj", "v_proj", "out_proj", "inter_dense", "output_dense", "projection" });
}

// codec.acoustic_decoder., codec.acoustic_encoder., codec.semantic_model. and
// codec.encoder_semantic.: the four codec modules whose parameters are read
// through a matrix multiply. `tokens[0]` is "codec" and `tokens[1]` is the
// module name; every decision below reads only the tokens past that, so it
// applies identically to all four.
QuantRole classify_codec_matrix_region(const std::vector<std::string_view> & tokens, const std::string & name) {
    const std::string_view suffix = tokens.back();
    // Every bias is added elementwise, and every Snake curve is divided by
    // elementwise; neither is ever read through a matrix multiply.
    if (suffix == "bias" || suffix == "alpha") {
        return QuantRole::Sensitive;
    }
    if (suffix != "weight") {
        return QuantRole::Unknown;
    }

    // A LayerNorm's weight is one curve per channel, read elementwise rather
    // than through a matrix multiply. This one token check covers all five
    // norm sites in this region -- the HuBERT feature group norm, the
    // feature projection's norm, each encoder layer's two norms, and the
    // encoder's own exit norm -- instead of five separate name patterns.
    if (tokens.size() >= 2) {
        const std::string_view parent = tokens[tokens.size() - 2];
        if (parent == "layer_norm" || parent == "final_layer_norm") {
            return QuantRole::Sensitive;
        }
    }

    // Three tensors are Sensitive for reasons specific to their shape or to
    // how this runtime reads them, named individually rather than folded
    // into a generic rule -- a future reader must not "fix" any of them.
    // Since the conv-exempt policy they are all three also convolution
    // kernels that the ConvKernel fall-through below would have kept out of a
    // block-quantized type anyway, so the reasons no longer *decide* anything
    // on their own; they stay because Sensitive is F32 where ConvKernel is
    // halved, because the already-cut F16 and Q8_MIXED packages hold these
    // three at F32, and because deleting them would lose the reasons.
    //
    // A note that applies to the *other* convolution kernels returned
    // below, not just these three: ggml_compute_forward_im2col
    // (ggml/src/ggml-cpu/ops.cpp, ~6369-6386) aborts on any destination type
    // besides F16/F32, so all 85 conv1d-consumed MatrixWeight tensors here
    // (the 32 acoustic_decoder + 36 acoustic_encoder + 6 feat_conv.1-6 + 11
    // encoder_semantic convolutions; the semantic_model attention/feed-
    // forward Linears are mul_mat directly and unaffected) would abort at
    // the first synthesis under a profile that packs them, if their
    // conv1d builders still passed the weight's own type as that
    // destination. Plan 4's Task 2 ported VITS's and Kokoro's own remedy
    // into this family's two conv1d builders (codec.cpp's codec_conv1d,
    // reference-encoder.cpp's conv1d): a packed branch that passes
    // GGML_TYPE_F32 as im2col's destination and feeds the quantized kernel
    // straight into mul_mat as its already-2-D operand
    // (src/arch/vits/operations.cpp:37-43,
    // src/arch/kokoro/operations.cpp:46-79). Under the conv-exempt policy no
    // profile this family cuts packs any of them any more, so that branch is
    // no longer reached by an omnivoice package -- it stays live for VITS,
    // Kokoro and Qwen3-TTS and stays covered here by
    // omnivoice_reference_encoder_test.cpp's check_packed_feat_conv1 and
    // check_packed_encoder_semantic_conv, which build packed kernels
    // directly rather than through a profile.
    if (name == "codec.acoustic_encoder.conv1.weight") {
        // Reads the raw mono reference waveform, so its packed row is
        // kernel * in_channels = 7 * 1 = 7: seven elements, never a multiple
        // of the 32-element block. The input channel count is architectural
        // (this convolution's whole purpose is to read one channel), not a
        // hyperparameter that could grow at another scale.
        return QuantRole::Sensitive;
    }
    if (name == "codec.semantic_model.feat_conv.0.conv.weight") {
        // The HuBERT feature extractor's first convolution also reads a raw
        // single-channel waveform, so in_channels=1 is architectural here
        // too. Its kernel width, though, is hparams.semantic.conv_kernel[0]
        // (10 for this checkpoint) -- a hyperparameter, not an architectural
        // constant like the encoder case above. The packed row (kernel * 1)
        // happens not to divide 32 for this checkpoint's width; the
        // exception is named by tensor rather than derived from that
        // arithmetic, so a differently-configured checkpoint would not
        // silently reclassify it. Conservative either way: packing a single
        // narrow-input convolution buys little regardless of its kernel
        // width.
        return QuantRole::Sensitive;
    }
    if (name == "codec.semantic_model.encoder.pos_conv_embed.conv.weight") {
        // Grouped by sixteen. reference-encoder.cpp's grouped_conv1d slices
        // this kernel per group with ggml_view_3d over the tensor's native
        // [kernel, in, out] layout -- a view that is well-formed on a
        // quantized tensor (ggml_view_impl has no type restriction here; each
        // slice is a whole output-channel plane at an exact nb[2] multiple).
        // The real blocker is what a block-quantized *matrix* weight is
        // stored as once packed: a flattened 2-D [kernel * in, out] matrix
        // with no separate in-axis left to slice per group, so
        // grouped_conv1d has nothing 3-D to view once this tensor is packed.
        // Its packed row (128 * 48 = 6144) is itself divisible by 32, so
        // nothing about its size would otherwise stop a future reader from
        // "optimizing" it away -- the grouped view is the whole reason.
        return QuantRole::Sensitive;
    }

    // codec.acoustic_decoder.block.<i>.conv_t1.weight: the transposed
    // convolution that upsamples each decoder block by its ratio. The same
    // override VITS and Qwen3-TTS's decoder make, and for the same reason
    // (policy.cpp:390-395, docs/quantization.md:52-56).
    if (tokens.size() == 6 && tokens[1] == "acoustic_decoder" && tokens[2] == "block" && is_index(tokens[3]) &&
        tokens[4] == "conv_t1") {
        return QuantRole::TransposeWeight;
    }

    // The conv-exempt split. Everything reaching here is read through a matrix
    // multiply; only the HuBERT Linears are block-quantized, and every
    // remaining weight in these four modules is an ordinary convolution kernel
    // held at the profile's halved type. See quantization.h's ConvKernel for
    // the measurement behind the policy, and note what it costs: the 85
    // convolutions include all 32 of the acoustic decoder's, so the decode
    // path every synthesis runs is left entirely unquantized and the 73
    // tensors that remain sit only on the clone-encode path.
    return is_semantic_model_linear(tokens) ? QuantRole::MatrixWeight : QuantRole::ConvKernel;
}

}  // namespace

ModelHalf tensor_half(const std::string & name) {
    const std::vector<std::string_view> tokens = split_name(name);
    return tokens.size() >= 2 && tokens[0] == "codec" ? ModelHalf::Codec : ModelHalf::Generator;
}

QuantRole classify_tensor(const std::string & name, const int64_t ne[4]) {
    // Every role this catalog resolves today follows from the name alone; see
    // the header comment for why the parameter still exists.
    (void) ne;

    const std::vector<std::string_view> tokens = split_name(name);
    if (tokens.size() < 2 || tokens[0] != "codec") {
        return classify_generator_tensor(tokens, name);
    }
    if (is_blanket_sensitive_codec_prefix(tokens)) {
        return QuantRole::Sensitive;
    }
    if (one_of(tokens[1], { "acoustic_decoder", "acoustic_encoder", "semantic_model", "encoder_semantic" })) {
        return classify_codec_matrix_region(tokens, name);
    }
    return QuantRole::Unknown;
}

QuantRole classify_tensor_for_half(const std::string & name, const int64_t ne[4], ModelHalf quantized_half) {
    const QuantRole role = classify_tensor(name, ne);
    if (role == QuantRole::Unknown) {
        return QuantRole::Unknown;
    }
    return tensor_half(name) == quantized_half ? role : QuantRole::Sensitive;
}

}  // namespace synth::omnivoice
