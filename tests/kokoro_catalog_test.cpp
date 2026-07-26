// The Kokoro tensor catalog is resolved from a small synthetic package.
//
// The catalog is also exercised against the real 511-tensor GGUF by the
// integration tier; this test keeps the shape contract covered without a model
// and pins the rejection behavior for a package whose tensors and metadata
// disagree.

#include "arch/kokoro/weights.h"
#include "ggml.h"
#include "test-assert.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace {

struct ContextDeleter {
    void operator()(ggml_context * context) const {
        if (context != nullptr) {
            ggml_free(context);
        }
    }
};

using Context = std::unique_ptr<ggml_context, ContextDeleter>;

Context make_context() {
    ggml_init_params parameters{};
    parameters.mem_size = ggml_tensor_overhead() * 4096;
    parameters.no_alloc = true;
    return Context(ggml_init(parameters));
}

// A structurally faithful package at reduced widths: two upsample stages, two
// resblock branches, three encoder layers, and two voices.
synth::kokoro::HParams small_hparams() {
    synth::kokoro::HParams h;
    h.n_token                  = 16;
    h.hidden_dim               = 8;
    h.style_dim                = 4;
    h.n_layer                  = 3;
    h.n_mels                   = 6;
    h.max_dur                  = 5;
    h.dim_in                   = 4;
    h.text_encoder_kernel_size = 5;
    h.samples_per_frame        = 600;

    h.plbert.hidden_size             = 12;
    h.plbert.num_attention_heads     = 3;
    h.plbert.intermediate_size       = 20;
    h.plbert.num_hidden_layers       = 4;
    h.plbert.max_position_embeddings = 32;
    h.plbert.shared_layer_groups     = 1;
    h.plbert.layer_norm_eps          = 1e-12f;

    h.istftnet.upsample_rates        = { 10, 6 };
    h.istftnet.upsample_kernel_sizes = { 20, 12 };
    h.istftnet.resblock_kernel_sizes = { 3, 7 };
    h.istftnet.resblock_dilations    = {
        { 1, 3, 5 },
        { 1, 3, 5 }
    };
    h.istftnet.upsample_initial_channel = 16;
    h.istftnet.gen_istft_n_fft          = 20;
    h.istftnet.gen_istft_hop_size       = 5;
    h.istftnet.center                   = true;

    h.source.sampling_rate  = 24000;
    h.source.harmonic_num   = 8;
    h.source.upsample_scale = 300;

    h.voice_rows           = 12;
    h.voice_dim            = 8;
    h.voice_prosody_offset = 4;
    h.preset_voice_ids     = { "af_one", "bm_two" };
    return h;
}

// Records every tensor the catalog is expected to find, so a single builder can
// both populate a valid package and drive the mutation cases.
struct Entry {
    std::string          name;
    std::vector<int64_t> ne;
};

void add(std::vector<Entry> & out, const std::string & name, std::vector<int64_t> ne) {
    out.push_back({ name, std::move(ne) });
}

void add_linear(std::vector<Entry> & out, const std::string & prefix, int64_t in, int64_t o) {
    add(out, prefix + ".weight", { in, o });
    add(out, prefix + ".bias", { o });
}

void add_norm(std::vector<Entry> & out, const std::string & prefix, int64_t c) {
    add(out, prefix + ".weight", { c });
    add(out, prefix + ".bias", { c });
}

void add_conv(std::vector<Entry> & out, const std::string & prefix, int64_t k, int64_t in, int64_t o) {
    add(out, prefix + ".weight", { k, in, o });
    add(out, prefix + ".bias", { o });
}

void add_lstm(std::vector<Entry> & out, const std::string & prefix, int64_t in, int64_t hidden) {
    for (const std::string suffix : { std::string(), std::string("_reverse") }) {
        add(out, prefix + ".weight_ih_l0" + suffix, { in, 4 * hidden });
        add(out, prefix + ".weight_hh_l0" + suffix, { hidden, 4 * hidden });
        add(out, prefix + ".bias_ih_l0" + suffix, { 4 * hidden });
        add(out, prefix + ".bias_hh_l0" + suffix, { 4 * hidden });
    }
}

void add_adain_resblock(std::vector<Entry> & out,
                        const std::string &  prefix,
                        int64_t              style,
                        int64_t              in,
                        int64_t              o,
                        bool                 upsample) {
    add_conv(out, prefix + ".conv1", 3, in, o);
    add_conv(out, prefix + ".conv2", 3, o, o);
    add_linear(out, prefix + ".norm1.fc", style, 2 * in);
    add_linear(out, prefix + ".norm2.fc", style, 2 * o);
    if (in != o) {
        add(out, prefix + ".conv1x1.weight", { 1, in, o });
    }
    if (upsample) {
        add(out, prefix + ".pool.weight", { 3, 1, in });
        add(out, prefix + ".pool.bias", { in });
    }
}

void add_adain_resblock1(std::vector<Entry> & out,
                         const std::string &  prefix,
                         int64_t              style,
                         int64_t              c,
                         int64_t              k,
                         size_t               layers) {
    for (size_t layer = 0; layer < layers; ++layer) {
        const std::string i = std::to_string(layer);
        add_conv(out, prefix + ".convs1." + i, k, c, c);
        add_conv(out, prefix + ".convs2." + i, k, c, c);
        add_linear(out, prefix + ".adain1." + i + ".fc", style, 2 * c);
        add_linear(out, prefix + ".adain2." + i + ".fc", style, 2 * c);
        add(out, prefix + ".alpha1." + i, { 1, c, 1 });
        add(out, prefix + ".alpha2." + i, { 1, c, 1 });
    }
}

std::vector<Entry> expected_entries(const synth::kokoro::HParams & h) {
    const int64_t hidden = h.hidden_dim;
    const int64_t style  = h.style_dim;
    const int64_t embed  = 6;  // ALBERT embedding width, narrower than the hidden size
    const int64_t bert   = h.plbert.hidden_size;

    std::vector<Entry> out;
    add(out, "bert.embeddings.word_embeddings.weight", { embed, h.n_token });
    add(out, "bert.embeddings.position_embeddings.weight", { embed, h.plbert.max_position_embeddings });
    add(out, "bert.embeddings.token_type_embeddings.weight", { embed, 2 });
    add_norm(out, "bert.embeddings.LayerNorm", embed);
    add_linear(out, "bert.encoder.embedding_hidden_mapping_in", embed, bert);
    add_linear(out, "bert.layer.attention.query", bert, bert);
    add_linear(out, "bert.layer.attention.key", bert, bert);
    add_linear(out, "bert.layer.attention.value", bert, bert);
    add_linear(out, "bert.layer.attention.dense", bert, bert);
    add_norm(out, "bert.layer.attention.LayerNorm", bert);
    add_linear(out, "bert.layer.ffn", bert, h.plbert.intermediate_size);
    add_linear(out, "bert.layer.ffn_output", h.plbert.intermediate_size, bert);
    add_norm(out, "bert.layer.full_layer_layer_norm", bert);
    add_linear(out, "bert_encoder", bert, hidden);

    add(out, "text_encoder.embedding.weight", { hidden, h.n_token });
    for (uint32_t block = 0; block < h.n_layer; ++block) {
        const std::string prefix = "text_encoder.cnn." + std::to_string(block);
        add_conv(out, prefix + ".0", h.text_encoder_kernel_size, hidden, hidden);
        add(out, prefix + ".1.gamma", { hidden });
        add(out, prefix + ".1.beta", { hidden });
    }
    add_lstm(out, "text_encoder.lstm", hidden, hidden / 2);

    const int64_t lstm_in = hidden + style;
    for (uint32_t block = 0; block < h.n_layer; ++block) {
        add_lstm(out, "predictor.text_encoder.lstms." + std::to_string(block * 2), lstm_in, hidden / 2);
        add_linear(out, "predictor.text_encoder.lstms." + std::to_string(block * 2 + 1) + ".fc", style, 2 * hidden);
    }
    add_lstm(out, "predictor.lstm", lstm_in, hidden / 2);
    add_linear(out, "predictor.duration_proj.linear_layer", hidden, h.max_dur);
    add_lstm(out, "predictor.shared", lstm_in, hidden / 2);
    const int64_t f0_hidden = hidden / 2;
    for (const std::string branch : { std::string("predictor.F0"), std::string("predictor.N") }) {
        add_adain_resblock(out, branch + ".0", style, hidden, hidden, false);
        add_adain_resblock(out, branch + ".1", style, hidden, f0_hidden, true);
        add_adain_resblock(out, branch + ".2", style, f0_hidden, f0_hidden, false);
    }
    add_conv(out, "predictor.F0_proj", 1, f0_hidden, 1);
    add_conv(out, "predictor.N_proj", 1, f0_hidden, 1);

    add_conv(out, "decoder.F0_conv", 3, 1, 1);
    add_conv(out, "decoder.N_conv", 3, 1, 1);
    add_conv(out, "decoder.asr_res.0", 1, hidden, 64);
    add_adain_resblock(out, "decoder.encode", style, hidden + 2, 1024, false);
    const int64_t decode_in = 1024 + 2 + 64;
    for (size_t block = 0; block < 4; ++block) {
        const bool    last = block == 3;
        const int64_t o    = last ? h.istftnet.upsample_initial_channel : 1024;
        add_adain_resblock(out, "decoder.decode." + std::to_string(block), style, decode_in, o, last);
    }

    const int64_t bins = h.istftnet.gen_istft_n_fft + 2;
    add_linear(out, "decoder.generator.m_source.l_linear", h.source.harmonic_num + 1, 1);
    int64_t      channels = h.istftnet.upsample_initial_channel;
    const size_t stages   = h.istftnet.upsample_rates.size();
    const size_t branches = h.istftnet.resblock_kernel_sizes.size();
    for (size_t stage = 0; stage < stages; ++stage) {
        const int64_t next = channels / 2;
        add(out, "decoder.generator.ups." + std::to_string(stage) + ".weight",
            { int64_t(h.istftnet.upsample_kernel_sizes[stage]), next, channels });
        add(out, "decoder.generator.ups." + std::to_string(stage) + ".bias", { next });
        for (size_t branch = 0; branch < branches; ++branch) {
            add_adain_resblock1(out, "decoder.generator.resblocks." + std::to_string(stage * branches + branch), style,
                                next, h.istftnet.resblock_kernel_sizes[branch],
                                h.istftnet.resblock_dilations[branch].size());
        }
        int64_t noise_kernel = 1;
        if (stage + 1 < stages) {
            int64_t stride = 1;
            for (size_t rest = stage + 1; rest < stages; ++rest) {
                stride *= h.istftnet.upsample_rates[rest];
            }
            noise_kernel = stride * 2;
        }
        add_conv(out, "decoder.generator.noise_convs." + std::to_string(stage), noise_kernel, bins, next);
        add_adain_resblock1(out, "decoder.generator.noise_res." + std::to_string(stage), style, next,
                            (stage + 1 < stages) ? 7 : 11, 3);
        channels = next;
    }
    add_conv(out, "decoder.generator.conv_post", 7, channels, bins);

    for (const std::string & voice : h.preset_voice_ids) {
        add(out, "voice." + voice, { h.voice_dim, h.voice_rows });
    }
    return out;
}

// Creates every entry, optionally transforming one of them to model a defect.
void populate(ggml_context *                                                   context,
              const std::vector<Entry> &                                       entries,
              const std::function<bool(const Entry &, Entry &, ggml_type &)> & mutate = nullptr) {
    for (const Entry & entry : entries) {
        Entry     effective = entry;
        ggml_type type      = GGML_TYPE_F32;
        if (mutate != nullptr && mutate(entry, effective, type)) {
            if (effective.ne.empty()) {
                continue;  // the mutation drops the tensor entirely
            }
        }
        std::vector<int64_t> ne = effective.ne;
        ne.resize(GGML_MAX_DIMS, 1);
        ggml_tensor * tensor = ggml_new_tensor(context, type, GGML_MAX_DIMS, ne.data());
        ggml_set_name(tensor, effective.name.c_str());
    }
}

}  // namespace

int main() {
    const synth::kokoro::HParams h       = small_hparams();
    const std::vector<Entry>     entries = expected_entries(h);

    Context                     context = make_context();
    synth::kokoro::ModelWeights weights;
    populate(context.get(), entries);
    SYNTH_TEST_CHECK(synth::kokoro::build_model_weights(context.get(), nullptr, h, weights) == SYNTH_OK);

    SYNTH_TEST_CHECK(weights.bert.word_embeddings != nullptr);
    SYNTH_TEST_CHECK(weights.bert.layer.ffn.weight != nullptr);
    SYNTH_TEST_CHECK(weights.text_encoder.cnn.size() == h.n_layer);
    SYNTH_TEST_CHECK(weights.text_encoder.cnn_norm.size() == h.n_layer);
    SYNTH_TEST_CHECK(weights.predictor.text_encoder.lstms.size() == h.n_layer);
    SYNTH_TEST_CHECK(weights.predictor.text_encoder.ada_norm.size() == h.n_layer);
    SYNTH_TEST_CHECK(weights.predictor.f0.size() == 3 && weights.predictor.n.size() == 3);
    // Only the upsampling block carries a pool, and only shape-changing blocks
    // carry a learned shortcut.
    SYNTH_TEST_CHECK(weights.predictor.f0[0].pool.weight == nullptr);
    SYNTH_TEST_CHECK(weights.predictor.f0[1].pool.weight != nullptr);
    SYNTH_TEST_CHECK(weights.predictor.f0[0].conv1x1 == nullptr);
    SYNTH_TEST_CHECK(weights.predictor.f0[1].conv1x1 != nullptr);
    SYNTH_TEST_CHECK(weights.decoder.decode.size() == 4);
    SYNTH_TEST_CHECK(weights.decoder.decode[3].pool.weight != nullptr);
    SYNTH_TEST_CHECK(weights.decoder.generator.ups.size() == 2);
    SYNTH_TEST_CHECK(weights.decoder.generator.noise_res.size() == 2);
    SYNTH_TEST_CHECK(weights.decoder.generator.resblocks.size() == 4);
    SYNTH_TEST_CHECK(weights.decoder.generator.resblocks[0].alpha1.size() == 3);
    SYNTH_TEST_CHECK(weights.voices.packs.size() == h.preset_voice_ids.size());

    SYNTH_TEST_CHECK(synth::kokoro::build_model_weights(nullptr, nullptr, h, weights) == SYNTH_ERR_INVALID_ARG);

    // A package missing any single catalog entry is rejected.
    for (const std::string & missing :
         { std::string("bert.layer.attention.query.weight"), std::string("predictor.shared.bias_hh_l0_reverse"),
           std::string("decoder.generator.resblocks.3.alpha2.2"), std::string("decoder.decode.3.pool.weight"),
           std::string("voice.bm_two") }) {
        Context                     ctx = make_context();
        synth::kokoro::ModelWeights parsed;
        populate(ctx.get(), entries, [&](const Entry & entry, Entry & effective, ggml_type &) {
            if (entry.name == missing) {
                effective.ne.clear();
                return true;
            }
            return false;
        });
        SYNTH_TEST_CHECK(synth::kokoro::build_model_weights(ctx.get(), nullptr, h, parsed) == SYNTH_ERR_GGUF);
    }

    // A tensor whose shape disagrees with the declared hyper-parameters is a
    // package defect, not something to accept and mis-execute.
    for (const std::string & wrong :
         { std::string("text_encoder.embedding.weight"), std::string("predictor.duration_proj.linear_layer.weight"),
           std::string("decoder.generator.conv_post.weight"), std::string("voice.af_one") }) {
        Context                     ctx = make_context();
        synth::kokoro::ModelWeights parsed;
        populate(ctx.get(), entries, [&](const Entry & entry, Entry & effective, ggml_type &) {
            if (entry.name == wrong) {
                effective.ne[0] += 1;
                return true;
            }
            return false;
        });
        SYNTH_TEST_CHECK(synth::kokoro::build_model_weights(ctx.get(), nullptr, h, parsed) == SYNTH_ERR_GGUF);
    }

    // The source-F32 catalog accepts only F32 tensors.
    {
        Context                     ctx = make_context();
        synth::kokoro::ModelWeights parsed;
        populate(ctx.get(), entries, [](const Entry & entry, Entry &, ggml_type & type) {
            if (entry.name == "bert_encoder.weight") {
                type = GGML_TYPE_F16;
                return true;
            }
            return false;
        });
        SYNTH_TEST_CHECK(synth::kokoro::build_model_weights(ctx.get(), nullptr, h, parsed) == SYNTH_ERR_GGUF);
    }

    // A trailing axis the catalog does not expect is also rejected, so a
    // silently reshaped tensor cannot slip through.
    {
        Context                     ctx = make_context();
        synth::kokoro::ModelWeights parsed;
        populate(ctx.get(), entries, [](const Entry & entry, Entry & effective, ggml_type &) {
            if (entry.name == "predictor.F0_proj.bias") {
                effective.ne.push_back(2);
                return true;
            }
            return false;
        });
        SYNTH_TEST_CHECK(synth::kokoro::build_model_weights(ctx.get(), nullptr, h, parsed) == SYNTH_ERR_GGUF);
    }

    return 0;
}
