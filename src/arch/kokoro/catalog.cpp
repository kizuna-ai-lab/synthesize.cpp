// Resolution of the Kokoro tensor catalog.
//
// Every entry is looked up by its canonical name and checked for type and
// shape. Shapes are derived from the package hyper-parameters rather than
// hardcoded, so a package whose metadata and tensors disagree is rejected at
// load time instead of producing wrong audio.
//
// GGML reports dimensions in the reverse of PyTorch's order: a Linear weight
// stored as [out, in] has ne = [in, out], and a Conv1d weight stored as
// [out, in, kernel] has ne = [kernel, in, out].

#include "ggml.h"
#include "quantization.h"
#include "weights.h"

#include <cstdio>
#include <initializer_list>
#include <string>
#include <vector>

namespace synth::kokoro {

namespace {

// Channel widths the upstream architecture fixes rather than declaring.
constexpr int64_t kDecoderEncodeOut = 1024;
constexpr int64_t kAsrResChannels   = 64;
// The decoder concatenates the F0 and energy curves onto its feature stream.
constexpr int64_t kCurveChannels    = 2;

// The storage type every tensor must carry under this package's profile.
ggml_type expected_storage_type(TensorRole role, QuantizationProfile profile, const ggml_tensor * tensor) {
    switch (profile) {
        case QuantizationProfile::F32:
            return GGML_TYPE_F32;
        case QuantizationProfile::F16:
            return role == TensorRole::Sensitive ? GGML_TYPE_F32 : GGML_TYPE_F16;
        case QuantizationProfile::Q8Mixed:
            if (role == TensorRole::MatrixWeight) {
                // A matrix whose packed row is not a whole number of blocks
                // carries the halved type instead.
                return matrix_is_block_quantizable(tensor->ne, ggml_n_dims(tensor), ggml_is_quantized(tensor->type),
                                                   ggml_blck_size(GGML_TYPE_Q8_0)) ?
                           GGML_TYPE_Q8_0 :
                           GGML_TYPE_F16;
            }
            return role == TensorRole::TransposeWeight ? GGML_TYPE_F16 : GGML_TYPE_F32;
    }
    return GGML_TYPE_F32;
}

// Walks the whole package rather than only what the catalog resolves, so a
// tensor nobody looks up cannot smuggle in an unexpected type, and a name
// outside the catalog is rejected instead of being ignored.
bool validate_storage_types(ggml_context * context, const HParams & hparams) {
    for (ggml_tensor * tensor = ggml_get_first_tensor(context); tensor != nullptr;
         tensor               = ggml_get_next_tensor(context, tensor)) {
        const std::string name = tensor->name;
        const TensorRole  role = tensor_role(name);
        if (role == TensorRole::Unknown) {
            std::fprintf(stderr, "kokoro: tensor %s is outside the catalog\n", name.c_str());
            return false;
        }
        const ggml_type expected = expected_storage_type(role, hparams.quantization_profile, tensor);
        if (tensor->type != expected) {
            std::fprintf(stderr, "kokoro: tensor %s has type %s, expected %s for this quantization profile\n",
                         name.c_str(), ggml_type_name(tensor->type), ggml_type_name(expected));
            return false;
        }
    }
    return true;
}

// `expected` is always the tensor's logical shape. Under a profile that packs
// matrix weights, a convolution kernel is stored as the two-dimensional
// [kernel * in_channels, out_channels] that a matrix multiply can consume, so
// the check compares against that instead.
ggml_tensor * find(ggml_context *                 context,
                   const HParams &                hparams,
                   const std::string &            name,
                   std::initializer_list<int64_t> expected) {
    ggml_tensor * tensor = ggml_get_tensor(context, name.c_str());
    if (tensor == nullptr) {
        std::fprintf(stderr, "kokoro: missing tensor %s\n", name.c_str());
        return nullptr;
    }
    // Storage type is not checked here. It is a function of the tensor's name
    // and the package's Quantization Profile alone, so it is checked once over
    // the whole package by validate_storage_types below, which also catches a
    // tensor the catalog never looks up.
    if (hparams.quantization_profile == QuantizationProfile::Q8Mixed && ggml_is_quantized(tensor->type) &&
        expected.size() == 3) {
        const auto *  want   = expected.begin();
        const int64_t packed = want[0] * want[1];
        if (want[0] <= 0 || want[1] <= 0 || tensor->ne[0] != packed || tensor->ne[1] != want[2] || tensor->ne[2] != 1 ||
            tensor->ne[3] != 1) {
            std::fprintf(stderr, "kokoro: tensor %s does not have the packed shape [%lld, %lld]\n", name.c_str(),
                         (long long) packed, (long long) want[2]);
            return nullptr;
        }
        return tensor;
    }
    size_t axis = 0;
    for (int64_t want : expected) {
        if (axis >= GGML_MAX_DIMS || tensor->ne[axis] != want) {
            std::fprintf(stderr, "kokoro: tensor %s has shape [%lld, %lld, %lld, %lld], expected axis %zu to be %lld\n",
                         name.c_str(), (long long) tensor->ne[0], (long long) tensor->ne[1], (long long) tensor->ne[2],
                         (long long) tensor->ne[3], axis, (long long) want);
            return nullptr;
        }
        ++axis;
    }
    for (; axis < GGML_MAX_DIMS; ++axis) {
        if (tensor->ne[axis] != 1) {
            std::fprintf(stderr, "kokoro: tensor %s has unexpected extent on axis %zu\n", name.c_str(), axis);
            return nullptr;
        }
    }
    return tensor;
}

bool load_linear(ggml_context *      context,
                 const HParams &     hparams,
                 const std::string & prefix,
                 int64_t             in,
                 int64_t             out,
                 LinearWeights &     out_w) {
    out_w.weight = find(context, hparams, prefix + ".weight", { in, out });
    out_w.bias   = find(context, hparams, prefix + ".bias", { out });
    return out_w.weight != nullptr && out_w.bias != nullptr;
}

bool load_norm(ggml_context *      context,
               const HParams &     hparams,
               const std::string & prefix,
               int64_t             channels,
               NormWeights &       out_w) {
    out_w.weight = find(context, hparams, prefix + ".weight", { channels });
    out_w.bias   = find(context, hparams, prefix + ".bias", { channels });
    return out_w.weight != nullptr && out_w.bias != nullptr;
}

// The text encoder's LayerNorm stores its parameters as gamma and beta.
bool load_gamma_beta(ggml_context *      context,
                     const HParams &     hparams,
                     const std::string & prefix,
                     int64_t             channels,
                     NormWeights &       out_w) {
    out_w.weight = find(context, hparams, prefix + ".gamma", { channels });
    out_w.bias   = find(context, hparams, prefix + ".beta", { channels });
    return out_w.weight != nullptr && out_w.bias != nullptr;
}

bool load_conv(ggml_context *      context,
               const HParams &     hparams,
               const std::string & prefix,
               int64_t             kernel,
               int64_t             in,
               int64_t             out,
               Conv1dWeights &     out_w) {
    out_w.weight = find(context, hparams, prefix + ".weight", { kernel, in, out });
    out_w.bias   = find(context, hparams, prefix + ".bias", { out });
    return out_w.weight != nullptr && out_w.bias != nullptr;
}

// ConvTranspose1d stores [in, out, kernel], so GGML reports [kernel, out, in].
bool load_transpose_conv(ggml_context *      context,
                         const HParams &     hparams,
                         const std::string & prefix,
                         int64_t             kernel,
                         int64_t             in,
                         int64_t             out,
                         Conv1dWeights &     out_w) {
    out_w.weight = find(context, hparams, prefix + ".weight", { kernel, out, in });
    out_w.bias   = find(context, hparams, prefix + ".bias", { out });
    return out_w.weight != nullptr && out_w.bias != nullptr;
}

// A depthwise transposed convolution keeps one filter per channel.
bool load_depthwise_transpose_conv(ggml_context *      context,
                                   const HParams &     hparams,
                                   const std::string & prefix,
                                   int64_t             kernel,
                                   int64_t             channels,
                                   Conv1dWeights &     out_w) {
    out_w.weight = find(context, hparams, prefix + ".weight", { kernel, 1, channels });
    out_w.bias   = find(context, hparams, prefix + ".bias", { channels });
    return out_w.weight != nullptr && out_w.bias != nullptr;
}

bool load_adain(ggml_context *      context,
                const HParams &     hparams,
                const std::string & prefix,
                int64_t             style_dim,
                int64_t             channels,
                AdaINWeights &      out_w) {
    return load_linear(context, hparams, prefix + ".fc", style_dim, 2 * channels, out_w.fc);
}

bool load_lstm_direction(ggml_context *         context,
                         const HParams &        hparams,
                         const std::string &    prefix,
                         const std::string &    suffix,
                         int64_t                input_dim,
                         int64_t                hidden,
                         LstmDirectionTensors & out_w) {
    out_w.weight_ih = find(context, hparams, prefix + ".weight_ih_l0" + suffix, { input_dim, 4 * hidden });
    out_w.weight_hh = find(context, hparams, prefix + ".weight_hh_l0" + suffix, { hidden, 4 * hidden });
    out_w.bias_ih   = find(context, hparams, prefix + ".bias_ih_l0" + suffix, { 4 * hidden });
    out_w.bias_hh   = find(context, hparams, prefix + ".bias_hh_l0" + suffix, { 4 * hidden });
    return out_w.weight_ih != nullptr && out_w.weight_hh != nullptr && out_w.bias_ih != nullptr &&
           out_w.bias_hh != nullptr;
}

bool load_lstm(ggml_context *      context,
               const HParams &     hparams,
               const std::string & prefix,
               int64_t             input_dim,
               int64_t             hidden,
               LstmTensors &       out_w) {
    return load_lstm_direction(context, hparams, prefix, "", input_dim, hidden, out_w.forward) &&
           load_lstm_direction(context, hparams, prefix, "_reverse", input_dim, hidden, out_w.reverse);
}

// AdainResBlk1d. `upsample` blocks carry a depthwise transposed convolution,
// and blocks that change the channel count carry a learned shortcut.
bool load_adain_resblock(ggml_context *         context,
                         const HParams &        hparams,
                         const std::string &    prefix,
                         int64_t                style_dim,
                         int64_t                in,
                         int64_t                out,
                         bool                   upsample,
                         AdainResBlockWeights & out_w) {
    if (!load_conv(context, hparams, prefix + ".conv1", 3, in, out, out_w.conv1) ||
        !load_conv(context, hparams, prefix + ".conv2", 3, out, out, out_w.conv2) ||
        !load_adain(context, hparams, prefix + ".norm1", style_dim, in, out_w.norm1) ||
        !load_adain(context, hparams, prefix + ".norm2", style_dim, out, out_w.norm2)) {
        return false;
    }
    if (in != out) {
        out_w.conv1x1 = find(context, hparams, prefix + ".conv1x1.weight", { 1, in, out });
        if (out_w.conv1x1 == nullptr) {
            return false;
        }
    }
    if (upsample && !load_depthwise_transpose_conv(context, hparams, prefix + ".pool", 3, in, out_w.pool)) {
        return false;
    }
    return true;
}

// AdaINResBlock1: three dilated convolution pairs with Snake activations.
bool load_adain_resblock1(ggml_context *                context,
                          const HParams &               hparams,
                          const std::string &           prefix,
                          int64_t                       style_dim,
                          int64_t                       channels,
                          int64_t                       kernel,
                          const std::vector<uint32_t> & dilations,
                          AdaINResBlock1Weights &       out_w) {
    const size_t layers = dilations.size();
    out_w.convs1.resize(layers);
    out_w.convs2.resize(layers);
    out_w.adain1.resize(layers);
    out_w.adain2.resize(layers);
    out_w.alpha1.resize(layers);
    out_w.alpha2.resize(layers);
    for (size_t layer = 0; layer < layers; ++layer) {
        const std::string index = std::to_string(layer);
        if (!load_conv(context, hparams, prefix + ".convs1." + index, kernel, channels, channels,
                       out_w.convs1[layer]) ||
            !load_conv(context, hparams, prefix + ".convs2." + index, kernel, channels, channels,
                       out_w.convs2[layer]) ||
            !load_adain(context, hparams, prefix + ".adain1." + index, style_dim, channels, out_w.adain1[layer]) ||
            !load_adain(context, hparams, prefix + ".adain2." + index, style_dim, channels, out_w.adain2[layer])) {
            return false;
        }
        // Snake alphas are stored as [1, channels, 1].
        out_w.alpha1[layer] = find(context, hparams, prefix + ".alpha1." + index, { 1, channels, 1 });
        out_w.alpha2[layer] = find(context, hparams, prefix + ".alpha2." + index, { 1, channels, 1 });
        if (out_w.alpha1[layer] == nullptr || out_w.alpha2[layer] == nullptr) {
            return false;
        }
    }
    return true;
}

bool build_bert(ggml_context * context, const HParams & hparams, PLBertWeights & out_w) {
    const int64_t hidden    = hparams.plbert.hidden_size;
    const int64_t embedding = hparams.plbert.hidden_size;  // replaced below
    (void) embedding;
    // The embedding width is whatever the stored table declares; it is smaller
    // than the hidden size in an ALBERT and is projected up.
    ggml_tensor * words = ggml_get_tensor(context, "bert.embeddings.word_embeddings.weight");
    if (words == nullptr) {
        std::fprintf(stderr, "kokoro: missing tensor bert.embeddings.word_embeddings.weight\n");
        return false;
    }
    const int64_t embed_dim = words->ne[0];
    if (embed_dim <= 0 || words->ne[1] != hparams.n_token) {
        std::fprintf(stderr, "kokoro: word embedding must be [embed_dim, n_token]\n");
        return false;
    }

    out_w.word_embeddings =
        find(context, hparams, "bert.embeddings.word_embeddings.weight", { embed_dim, hparams.n_token });
    out_w.position_embeddings = find(context, hparams, "bert.embeddings.position_embeddings.weight",
                                     { embed_dim, hparams.plbert.max_position_embeddings });
    out_w.token_type_embeddings =
        find(context, hparams, "bert.embeddings.token_type_embeddings.weight", { embed_dim, 2 });
    if (out_w.word_embeddings == nullptr || out_w.position_embeddings == nullptr ||
        out_w.token_type_embeddings == nullptr) {
        return false;
    }
    if (!load_norm(context, hparams, "bert.embeddings.LayerNorm", embed_dim, out_w.embedding_norm) ||
        !load_linear(context, hparams, "bert.encoder.embedding_hidden_mapping_in", embed_dim, hidden,
                     out_w.hidden_mapping)) {
        return false;
    }

    // The converter shortens upstream's ALBERT path, which overruns GGML's
    // 64-byte name field; the group and layer indices carry no information
    // because the package declares a single shared layer group.
    const std::string layer = "bert.layer";
    return load_linear(context, hparams, layer + ".attention.query", hidden, hidden, out_w.layer.query) &&
           load_linear(context, hparams, layer + ".attention.key", hidden, hidden, out_w.layer.key) &&
           load_linear(context, hparams, layer + ".attention.value", hidden, hidden, out_w.layer.value) &&
           load_linear(context, hparams, layer + ".attention.dense", hidden, hidden, out_w.layer.dense) &&
           load_norm(context, hparams, layer + ".attention.LayerNorm", hidden, out_w.layer.attention_norm) &&
           load_linear(context, hparams, layer + ".ffn", hidden, hparams.plbert.intermediate_size, out_w.layer.ffn) &&
           load_linear(context, hparams, layer + ".ffn_output", hparams.plbert.intermediate_size, hidden,
                       out_w.layer.ffn_output) &&
           load_norm(context, hparams, layer + ".full_layer_layer_norm", hidden, out_w.layer.output_norm);
}

bool build_text_encoder(ggml_context * context, const HParams & hparams, TextEncoderWeights & out_w) {
    const int64_t channels = hparams.hidden_dim;
    const int64_t kernel   = hparams.text_encoder_kernel_size;
    out_w.embedding        = find(context, hparams, "text_encoder.embedding.weight", { channels, hparams.n_token });
    if (out_w.embedding == nullptr) {
        return false;
    }
    out_w.cnn.resize(hparams.n_layer);
    out_w.cnn_norm.resize(hparams.n_layer);
    for (uint32_t block = 0; block < hparams.n_layer; ++block) {
        const std::string prefix = "text_encoder.cnn." + std::to_string(block);
        if (!load_conv(context, hparams, prefix + ".0", kernel, channels, channels, out_w.cnn[block]) ||
            !load_gamma_beta(context, hparams, prefix + ".1", channels, out_w.cnn_norm[block])) {
            return false;
        }
    }
    return load_lstm(context, hparams, "text_encoder.lstm", channels, channels / 2, out_w.lstm);
}

bool build_predictor(ggml_context * context, const HParams & hparams, ProsodyPredictorWeights & out_w) {
    const int64_t hidden    = hparams.hidden_dim;
    const int64_t style     = hparams.style_dim;
    const int64_t lstm_in   = hidden + style;
    const int64_t half      = hidden / 2;
    const int64_t f0_hidden = hidden / 2;

    out_w.text_encoder.lstms.resize(hparams.n_layer);
    out_w.text_encoder.ada_norm.resize(hparams.n_layer);
    for (uint32_t block = 0; block < hparams.n_layer; ++block) {
        // The module list alternates LSTM and adaptive norm entries.
        const std::string lstm_prefix = "predictor.text_encoder.lstms." + std::to_string(block * 2);
        const std::string norm_prefix = "predictor.text_encoder.lstms." + std::to_string(block * 2 + 1);
        if (!load_lstm(context, hparams, lstm_prefix, lstm_in, half, out_w.text_encoder.lstms[block]) ||
            !load_linear(context, hparams, norm_prefix + ".fc", style, 2 * hidden,
                         out_w.text_encoder.ada_norm[block])) {
            return false;
        }
    }

    if (!load_lstm(context, hparams, "predictor.lstm", lstm_in, half, out_w.lstm) ||
        !load_linear(context, hparams, "predictor.duration_proj.linear_layer", hidden, hparams.max_dur,
                     out_w.duration_proj) ||
        !load_lstm(context, hparams, "predictor.shared", lstm_in, half, out_w.shared)) {
        return false;
    }

    // F0 and energy share the same three-block shape: keep, upsample, keep.
    struct BlockShape {
        int64_t in;
        int64_t out;
        bool    upsample;
    };

    const BlockShape shapes[] = {
        { hidden,    hidden,    false },
        { hidden,    f0_hidden, true  },
        { f0_hidden, f0_hidden, false },
    };
    out_w.f0.resize(3);
    out_w.n.resize(3);
    for (size_t block = 0; block < 3; ++block) {
        const BlockShape & shape = shapes[block];
        if (!load_adain_resblock(context, hparams, "predictor.F0." + std::to_string(block), style, shape.in, shape.out,
                                 shape.upsample, out_w.f0[block]) ||
            !load_adain_resblock(context, hparams, "predictor.N." + std::to_string(block), style, shape.in, shape.out,
                                 shape.upsample, out_w.n[block])) {
            return false;
        }
    }
    return load_conv(context, hparams, "predictor.F0_proj", 1, f0_hidden, 1, out_w.f0_proj) &&
           load_conv(context, hparams, "predictor.N_proj", 1, f0_hidden, 1, out_w.n_proj);
}

bool build_generator(ggml_context * context, const HParams & hparams, GeneratorWeights & out_w) {
    const IStftNetParams & net    = hparams.istftnet;
    const int64_t          style  = hparams.style_dim;
    const int64_t          bins   = net.gen_istft_n_fft + 2;
    const size_t           stages = net.upsample_rates.size();

    // The harmonic source merges its overtones through a single projection.
    if (!load_linear(context, hparams, "decoder.generator.m_source.l_linear", hparams.source.harmonic_num + 1, 1,
                     out_w.source_linear)) {
        return false;
    }

    out_w.ups.resize(stages);
    out_w.noise_convs.resize(stages);
    out_w.noise_res.resize(stages);
    out_w.resblocks.resize(stages * net.resblock_kernel_sizes.size());

    int64_t channels = net.upsample_initial_channel;
    for (size_t stage = 0; stage < stages; ++stage) {
        const int64_t next = channels / 2;
        if (!load_transpose_conv(context, hparams, "decoder.generator.ups." + std::to_string(stage),
                                 net.upsample_kernel_sizes[stage], channels, next, out_w.ups[stage])) {
            return false;
        }
        for (size_t branch = 0; branch < net.resblock_kernel_sizes.size(); ++branch) {
            const size_t index = stage * net.resblock_kernel_sizes.size() + branch;
            if (!load_adain_resblock1(context, hparams, "decoder.generator.resblocks." + std::to_string(index), style,
                                      next, net.resblock_kernel_sizes[branch], net.resblock_dilations[branch],
                                      out_w.resblocks[index])) {
                return false;
            }
        }
        // Every stage but the last downsamples the source spectrum to its rate.
        int64_t noise_kernel = 1;
        if (stage + 1 < stages) {
            int64_t stride = 1;
            for (size_t rest = stage + 1; rest < stages; ++rest) {
                stride *= net.upsample_rates[rest];
            }
            noise_kernel = stride * 2;
        }
        if (!load_conv(context, hparams, "decoder.generator.noise_convs." + std::to_string(stage), noise_kernel, bins,
                       next, out_w.noise_convs[stage])) {
            return false;
        }
        // The final stage widens its residual kernel from 7 to 11.
        const int64_t               noise_res_kernel = (stage + 1 < stages) ? 7 : 11;
        const std::vector<uint32_t> noise_dilations  = { 1, 3, 5 };
        if (!load_adain_resblock1(context, hparams, "decoder.generator.noise_res." + std::to_string(stage), style, next,
                                  noise_res_kernel, noise_dilations, out_w.noise_res[stage])) {
            return false;
        }
        channels = next;
    }
    return load_conv(context, hparams, "decoder.generator.conv_post", 7, channels, bins, out_w.conv_post);
}

bool build_decoder(ggml_context * context, const HParams & hparams, DecoderWeights & out_w) {
    const int64_t hidden = hparams.hidden_dim;
    const int64_t style  = hparams.style_dim;

    if (!load_conv(context, hparams, "decoder.F0_conv", 3, 1, 1, out_w.f0_conv) ||
        !load_conv(context, hparams, "decoder.N_conv", 3, 1, 1, out_w.n_conv) ||
        !load_conv(context, hparams, "decoder.asr_res.0", 1, hidden, kAsrResChannels, out_w.asr_res)) {
        return false;
    }
    if (!load_adain_resblock(context, hparams, "decoder.encode", style, hidden + kCurveChannels, kDecoderEncodeOut,
                             false, out_w.encode)) {
        return false;
    }

    // Three blocks keep the width, then one upsamples down to the generator's
    // initial channel count. Each takes the encode output re-concatenated with
    // the ASR residual and both curves.
    const int64_t decode_in   = kDecoderEncodeOut + kCurveChannels + kAsrResChannels;
    const int64_t final_out   = hparams.istftnet.upsample_initial_channel;
    const size_t  block_count = 4;
    out_w.decode.resize(block_count);
    for (size_t block = 0; block < block_count; ++block) {
        const bool    last = block + 1 == block_count;
        const int64_t out  = last ? final_out : kDecoderEncodeOut;
        if (!load_adain_resblock(context, hparams, "decoder.decode." + std::to_string(block), style, decode_in, out,
                                 last, out_w.decode[block])) {
            return false;
        }
    }
    return build_generator(context, hparams, out_w.generator);
}

bool build_voices(ggml_context * context, const HParams & hparams, VoiceWeights & out_w) {
    out_w.packs.clear();
    out_w.packs.reserve(hparams.preset_voice_ids.size());
    for (const std::string & id : hparams.preset_voice_ids) {
        ggml_tensor * pack = find(context, hparams, "voice." + id, { hparams.voice_dim, hparams.voice_rows });
        if (pack == nullptr) {
            return false;
        }
        out_w.packs.push_back(pack);
    }
    return true;
}

}  // namespace

synth_status_t build_model_weights(ggml_context * context, const HParams & hparams, ModelWeights & weights) {
    if (context == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    weights = ModelWeights{};
    if (!validate_storage_types(context, hparams)) {
        return SYNTH_ERR_GGUF;
    }
    if (!build_bert(context, hparams, weights.bert) ||
        !load_linear(context, hparams, "bert_encoder", hparams.plbert.hidden_size, hparams.hidden_dim,
                     weights.bert_encoder) ||
        !build_text_encoder(context, hparams, weights.text_encoder) ||
        !build_predictor(context, hparams, weights.predictor) || !build_decoder(context, hparams, weights.decoder) ||
        !build_voices(context, hparams, weights.voices)) {
        return SYNTH_ERR_GGUF;
    }
    return SYNTH_OK;
}

}  // namespace synth::kokoro
