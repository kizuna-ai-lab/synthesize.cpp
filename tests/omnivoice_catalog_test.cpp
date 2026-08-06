// The OmniVoice tensor catalog is resolved from a small synthetic package.
//
// The entry list below is a second, independent statement of the contract: it is
// written out by hand from the checkpoint's layout, while the catalog derives
// the same names and shapes from the hyper-parameters. Agreeing is the point.
// The real 798-tensor package is exercised by the integration tier.
//
// Most of the value here is in the rejections. A package whose metadata and
// tensors disagree must be refused at load: past that point the mistake becomes
// wrong audio rather than an error.

#include "arch/omnivoice/catalog.h"
#include "arch/omnivoice/quantization.h"
#include "arch/omnivoice/weights.h"
#include "ggml.h"
#include "omnivoice_small_layout.h"
#include "test-assert.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace synth::omnivoice::testing;

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

// `mutate` may rewrite one entry's shape or type, or drop it by clearing `ne`.
using Mutation = std::function<bool(const Entry &, Entry &, ggml_type &)>;

void populate(ggml_context * context, const std::vector<Entry> & entries, const Mutation & mutate) {
    for (const Entry & entry : entries) {
        Entry     effective = entry;
        ggml_type type      = GGML_TYPE_F32;
        if (mutate && mutate(entry, effective, type)) {
            if (effective.ne.empty()) {
                continue;
            }
        }
        int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
        for (size_t axis = 0; axis < effective.ne.size() && axis < GGML_MAX_DIMS; ++axis) {
            ne[axis] = effective.ne[axis];
        }
        ggml_tensor * tensor =
            ggml_new_tensor(context, type, int(std::min<size_t>(effective.ne.size(), GGML_MAX_DIMS)), ne);
        ggml_set_name(tensor, effective.name.c_str());
    }
}

int check_resolution(const synth::omnivoice::HParams & h, const std::vector<Entry> & entries) {
    Context                        context = make_context();
    synth::omnivoice::ModelWeights weights;
    populate(context.get(), entries, nullptr);
    SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), nullptr, h, weights) == SYNTH_OK);

    // The count the catalog resolves and the count derived by arithmetic are two
    // independent statements of the same package, so they must agree.
    SYNTH_TEST_CHECK(synth::omnivoice::expected_tensor_count(h) == entries.size());

    SYNTH_TEST_CHECK(weights.generator.layers.size() == h.generator.layer_count);
    SYNTH_TEST_CHECK(weights.generator.text_embedding != nullptr);
    SYNTH_TEST_CHECK(weights.generator.norm != nullptr);
    // Qwen3 normalizes each head at head_dim, not the packed projection.
    SYNTH_TEST_CHECK(weights.generator.layers[0].q_norm != nullptr);
    SYNTH_TEST_CHECK(weights.generator.layers[0].q_norm->ne[0] == h.generator.head_dim);
    // The canvas tables stack every codebook into one table.
    SYNTH_TEST_CHECK(weights.generator.audio_embeddings != nullptr);
    SYNTH_TEST_CHECK(weights.generator.audio_heads->ne[1] ==
                     int64_t(h.audio.num_codebooks) * int64_t(h.audio.vocab_size));

    SYNTH_TEST_CHECK(weights.quantizers.size() == h.audio.num_codebooks);
    SYNTH_TEST_CHECK(weights.quantizers[1].codebook != nullptr);
    SYNTH_TEST_CHECK(weights.quantizers[1].codebook->ne[0] == h.codec.codebook_dim);
    // fc2 brings the concatenated latent back down to the acoustic width.
    SYNTH_TEST_CHECK(weights.fc2.weight != nullptr);
    SYNTH_TEST_CHECK(weights.fc2.weight->ne[1] == h.codec.hidden_size);
    SYNTH_TEST_CHECK(weights.fc.weight->ne[0] == int64_t(h.codec.hidden_size) + h.semantic.hidden_size);

    SYNTH_TEST_CHECK(weights.acoustic_decoder.blocks.size() == h.codec.upsampling_ratios.size());
    SYNTH_TEST_CHECK(weights.acoustic_decoder.blocks[0].res_units.size() == 3);
    // Plain Snake, so there is an alpha and nothing beside it.
    SYNTH_TEST_CHECK(weights.acoustic_decoder.blocks[0].res_units[2].snake2.alpha != nullptr);
    // The decoder halves once per block, so the last width is the narrowest and
    // the exit convolution emits one channel.
    SYNTH_TEST_CHECK(weights.acoustic_decoder.snake1.alpha->ne[1] ==
                     int64_t(h.codec.decoder_hidden_size) >> h.codec.upsampling_ratios.size());
    SYNTH_TEST_CHECK(weights.acoustic_decoder.conv2.weight->ne[2] == 1);

    SYNTH_TEST_CHECK(weights.acoustic_encoder.blocks.size() == h.codec.upsampling_ratios.size());
    SYNTH_TEST_CHECK(weights.acoustic_encoder.conv1.weight->ne[1] == 1);
    SYNTH_TEST_CHECK(weights.acoustic_encoder.conv2.weight->ne[2] == h.codec.hidden_size);

    SYNTH_TEST_CHECK(weights.semantic_model.feat_conv.size() == h.semantic.conv_dim.size());
    // The feature extractor's convolutions carry no bias, and only the first is
    // normalized.
    SYNTH_TEST_CHECK(weights.semantic_model.feat_conv[0].bias == nullptr);
    SYNTH_TEST_CHECK(weights.semantic_model.feat_conv_norm.weight != nullptr);
    SYNTH_TEST_CHECK(weights.semantic_model.layers.size() == h.semantic.layer_count);
    SYNTH_TEST_CHECK(weights.semantic_model.layers[1].out_proj.bias != nullptr);
    // Grouped by sixteen, so each filter sees a sixteenth of the width.
    SYNTH_TEST_CHECK(weights.semantic_model.pos_conv.weight->ne[1] == int64_t(h.semantic.hidden_size) / 16);

    SYNTH_TEST_CHECK(weights.encoder_semantic.blocks.size() == 2);
    SYNTH_TEST_CHECK(weights.encoder_semantic.blocks[1].res_units.size() == 2);
    SYNTH_TEST_CHECK(weights.encoder_semantic.blocks[1].res_units[0].conv1.bias == nullptr);
    SYNTH_TEST_CHECK(weights.encoder_semantic.conv.bias == nullptr);
    return 0;
}

int check_rejections(const synth::omnivoice::HParams & h, const std::vector<Entry> & entries) {
    synth::omnivoice::ModelWeights weights;
    SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(nullptr, nullptr, h, weights) == SYNTH_ERR_INVALID_ARG);

    // Any single missing entry is a package defect. One is taken from each
    // region so a whole region cannot go unresolved unnoticed.
    for (const std::string & missing :
         { std::string("llm.layers.1.self_attn.k_norm.weight"), std::string("audio_heads.weight"),
           std::string("codec.quantizer.quantizers.1.codebook.embed"), std::string("codec.fc2.bias"),
           std::string("codec.acoustic_decoder.block.1.res_unit3.conv2.bias"),
           std::string("codec.acoustic_encoder.block.0.snake1.alpha"),
           std::string("codec.semantic_model.feat_conv.0.layer_norm.bias"),
           std::string("codec.semantic_model.encoder.layers.1.ff.output_dense.weight"),
           std::string("codec.semantic_model.encoder.pos_conv_embed.conv.weight"),
           std::string("codec.encoder_semantic.conv_blocks.1.res_units.0.conv2.weight") }) {
        Context                        context = make_context();
        synth::omnivoice::ModelWeights parsed;
        populate(context.get(), entries, [&](const Entry & entry, Entry & effective, ggml_type &) {
            if (entry.name == missing) {
                effective.ne.clear();
                return true;
            }
            return false;
        });
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), nullptr, h, parsed) == SYNTH_ERR_GGUF);
    }

    // A shape that disagrees with the declared hyper-parameters.
    for (const std::string & wrong :
         { std::string("llm.embed_tokens.weight"), std::string("audio_embeddings.weight"),
           std::string("codec.quantizer.quantizers.0.project_in.weight"), std::string("codec.fc.weight"),
           std::string("codec.acoustic_decoder.conv1.weight"),
           std::string("codec.acoustic_encoder.block.1.conv1.weight"),
           std::string("codec.semantic_model.feat_conv.1.conv.weight"),
           std::string("codec.encoder_semantic.conv.weight") }) {
        Context                        context = make_context();
        synth::omnivoice::ModelWeights parsed;
        populate(context.get(), entries, [&](const Entry & entry, Entry & effective, ggml_type &) {
            if (entry.name == wrong) {
                effective.ne[0] += 1;
                return true;
            }
            return false;
        });
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), nullptr, h, parsed) == SYNTH_ERR_GGUF);
    }

    // A transposed convolution stored in a plain convolution's order. The two
    // differ only in the order of the trailing pair, so a package built by a
    // converter that confused them still has the right element count.
    {
        Context                        context = make_context();
        synth::omnivoice::ModelWeights parsed;
        populate(context.get(), entries, [](const Entry & entry, Entry & effective, ggml_type &) {
            if (entry.name == "codec.acoustic_decoder.block.0.conv_t1.weight") {
                std::swap(effective.ne[1], effective.ne[2]);
                return true;
            }
            return false;
        });
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), nullptr, h, parsed) == SYNTH_ERR_GGUF);
    }

    // This profile carries the checkpoint through unchanged, so every tensor is
    // F32 and anything halved is a package defect.
    for (const std::string & recast :
         { std::string("llm.norm.weight"), std::string("codec.acoustic_decoder.block.0.res_unit1.conv1.weight") }) {
        Context                        context = make_context();
        synth::omnivoice::ModelWeights parsed;
        populate(context.get(), entries, [&](const Entry & entry, Entry &, ggml_type & type) {
            if (entry.name == recast) {
                type = GGML_TYPE_F16;
                return true;
            }
            return false;
        });
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), nullptr, h, parsed) == SYNTH_ERR_GGUF);
    }

    // A trailing axis the catalog does not expect, so a silently reshaped tensor
    // cannot slip through with the right element count.
    {
        Context                        context = make_context();
        synth::omnivoice::ModelWeights parsed;
        populate(context.get(), entries, [](const Entry & entry, Entry & effective, ggml_type &) {
            if (entry.name == "llm.norm.weight") {
                effective.ne.push_back(2);
                return true;
            }
            return false;
        });
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), nullptr, h, parsed) == SYNTH_ERR_GGUF);
    }

    // A tensor nobody looks up is a tensor nobody checks. This is what would
    // otherwise let a skipped training buffer -- an EMA table, the semantic
    // decoder -- ride along unread.
    {
        Context                        context = make_context();
        synth::omnivoice::ModelWeights parsed;
        populate(context.get(), entries, nullptr);
        ggml_tensor * stray = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 4);
        ggml_set_name(stray, "codec.decoder_semantic.conv1.weight");
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), nullptr, h, parsed) == SYNTH_ERR_GGUF);
    }

    // The quantizer's width is the acoustic and semantic widths added together
    // rather than a number the package states, so a semantic width that moved
    // is refused even though every name is still present.
    {
        Context                        context = make_context();
        synth::omnivoice::HParams      other   = h;
        synth::omnivoice::ModelWeights parsed;
        other.semantic.hidden_size = h.semantic.hidden_size + 16;
        populate(context.get(), entries, nullptr);
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), nullptr, other, parsed) ==
                         SYNTH_ERR_GGUF);
    }

    // Hyper-parameters the resolver would have to index past: refused rather
    // than read out of bounds.
    {
        Context                        context = make_context();
        synth::omnivoice::HParams      other   = h;
        synth::omnivoice::ModelWeights parsed;
        other.semantic.conv_kernel.pop_back();
        populate(context.get(), entries, nullptr);
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), nullptr, other, parsed) ==
                         SYNTH_ERR_GGUF);
    }
    return 0;
}

// Task 9's own filter, restated independently here rather than reused from
// model.cpp: the same "second, independent statement of the contract"
// argument this file's header comment already makes about `expected_entries`
// applies just as much to the twin's own prefix list.
bool is_decode_path_entry(const Entry & entry) {
    for (const char * prefix : { "codec.acoustic_decoder.", "codec.quantizer.", "codec.fc2." }) {
        if (entry.name.rfind(prefix, 0) == 0) {
            return true;
        }
    }
    return false;
}

// The twin context: build_model_weights must bind the three movable groups
// against it when present, and leave every other codec field bound to the
// package -- the trap this whole task exists to avoid mirroring past.
int check_twin_resolution(const synth::omnivoice::HParams & h, const std::vector<Entry> & entries) {
    Context context = make_context();
    populate(context.get(), entries, nullptr);

    std::vector<Entry> twin_entries;
    for (const Entry & entry : entries) {
        if (is_decode_path_entry(entry)) {
            twin_entries.push_back(entry);
        }
    }
    // 59 at this synthetic package's reduced widths: 47 acoustic_decoder (2
    // upsampling blocks, kPerDacBlock's own 21-per-block plus the 2+1+2
    // entry/exit tensors) + 10 quantizer (2 codebooks * 5) + 2 fc2. The real
    // package's numbers -- 110/40/2, catalog.h's own MOVABLE accounting -- are
    // this same arithmetic at h.codec.upsampling_ratios.size() == 5 and
    // h.audio.num_codebooks == 8.
    SYNTH_TEST_CHECK(twin_entries.size() == 59);

    Context twin = make_context();
    populate(twin.get(), twin_entries, nullptr);

    synth::omnivoice::ModelWeights weights;
    SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), twin.get(), h, weights) == SYNTH_OK);

    // The three movable groups are bound to the twin, not the package.
    SYNTH_TEST_CHECK(weights.quantizers[0].codebook ==
                     ggml_get_tensor(twin.get(), "codec.quantizer.quantizers.0.codebook.embed"));
    SYNTH_TEST_CHECK(weights.quantizers[0].codebook !=
                     ggml_get_tensor(context.get(), "codec.quantizer.quantizers.0.codebook.embed"));
    SYNTH_TEST_CHECK(weights.fc2.weight == ggml_get_tensor(twin.get(), "codec.fc2.weight"));
    SYNTH_TEST_CHECK(weights.fc2.weight != ggml_get_tensor(context.get(), "codec.fc2.weight"));
    SYNTH_TEST_CHECK(weights.acoustic_decoder.conv1.weight ==
                     ggml_get_tensor(twin.get(), "codec.acoustic_decoder.conv1.weight"));
    SYNTH_TEST_CHECK(weights.acoustic_decoder.conv1.weight !=
                     ggml_get_tensor(context.get(), "codec.acoustic_decoder.conv1.weight"));

    // Every other codec field stays bound to the package: the twin never
    // carries them, and a filter that mirrored them too -- qwen3-tts's own
    // blanket `codec.` prefix, copied without narrowing -- would move weight
    // the CPU-held clone-encode chain reads to an accelerator no primary-side
    // graph shares with it.
    SYNTH_TEST_CHECK(weights.fc.weight == ggml_get_tensor(context.get(), "codec.fc.weight"));
    SYNTH_TEST_CHECK(weights.acoustic_encoder.conv1.weight ==
                     ggml_get_tensor(context.get(), "codec.acoustic_encoder.conv1.weight"));
    SYNTH_TEST_CHECK(weights.semantic_model.feat_conv[0].weight ==
                     ggml_get_tensor(context.get(), "codec.semantic_model.feat_conv.0.conv.weight"));
    SYNTH_TEST_CHECK(weights.encoder_semantic.conv.weight ==
                     ggml_get_tensor(context.get(), "codec.encoder_semantic.conv.weight"));

    // A twin missing one of the movable tensors is refused, the same as a
    // missing package tensor.
    {
        std::vector<Entry> incomplete;
        for (const Entry & entry : twin_entries) {
            if (entry.name != "codec.fc2.bias") {
                incomplete.push_back(entry);
            }
        }
        Context                        broken_twin = make_context();
        synth::omnivoice::ModelWeights parsed;
        populate(broken_twin.get(), incomplete, nullptr);
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), broken_twin.get(), h, parsed) ==
                         SYNTH_ERR_GGUF);
    }

    // A twin whose movable tensor disagrees in shape is refused too.
    {
        Context                        broken_twin = make_context();
        synth::omnivoice::ModelWeights parsed;
        populate(broken_twin.get(), twin_entries, [](const Entry & entry, Entry & effective, ggml_type &) {
            if (entry.name == "codec.acoustic_decoder.conv1.weight") {
                effective.ne[0] += 1;
                return true;
            }
            return false;
        });
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), broken_twin.get(), h, parsed) ==
                         SYNTH_ERR_GGUF);
    }
    return 0;
}

// The real package's shape, to catch an arithmetic change that the synthetic
// package is too small to notice. 798 is the emitted tensor count in
// reports/convert/omnivoice/omnivoice-0-6b-F32.json: 312 generator plus 486
// codec, after the converter's declared skips.
int check_real_package_count() {
    synth::omnivoice::HParams h;
    h.generator.layer_count   = 28;
    h.audio.num_codebooks     = 8;
    h.codec.upsampling_ratios = { 8, 5, 4, 2, 3 };
    h.semantic.layer_count    = 12;
    h.semantic.conv_dim       = { 512, 512, 512, 512, 512, 512, 512 };
    SYNTH_TEST_CHECK(synth::omnivoice::expected_tensor_count(h) == 798);
    return 0;
}

// Widths for the Q8_MIXED variant of the small package. Every "in" dimension
// a MatrixWeight tensor is ever block-quantized against -- packed (kernel *
// in_channels) or native (a Linear's own row) -- has to be a multiple of
// Q8_0's 32-element block, or ggml_new_tensor's own row-size assert fires
// while building the fixture below, not this catalog's code. The whole
// generator and RVQ are Sensitive under every profile (quantization.h), so
// only the codec's widths matter here: the decoder halves twice (128, 64, 32)
// and the encoder doubles twice (32, 64, 128) over this package's two
// upsampling ratios, `codec.hidden_size` feeds the decoder's entry
// convolution's row, and `semantic.intermediate_size` and every
// `semantic.conv_dim` entry feed a HuBERT row the same way. Kernel widths and
// `semantic.hidden_size` (already 32) are untouched.
synth::omnivoice::HParams q8_mixed_hparams() {
    synth::omnivoice::HParams h  = small_hparams();
    h.quantization_profile       = synth::omnivoice::QuantizationProfile::Q8Mixed;
    h.codec.decoder_hidden_size  = 128;
    h.codec.encoder_hidden_size  = 32;
    h.codec.hidden_size          = 32;
    h.semantic.intermediate_size = 32;
    h.semantic.conv_dim          = { 32, 32, 32 };
    return h;
}

// Recasts one F32 entry list to what the offline quantizer would have written
// for it under Q8_MIXED: a MatrixWeight tensor packs to Q8_0 (a three-axis
// entry -- a convolution kernel -- flattens to [kernel * in, out]; a two-axis
// one -- a Linear -- keeps its shape), and every other role stays F32,
// exactly classify_tensor's own split.
std::vector<std::pair<Entry, ggml_type>> to_q8_mixed(const std::vector<Entry> & entries) {
    std::vector<std::pair<Entry, ggml_type>> out;
    out.reserve(entries.size());
    for (const Entry & entry : entries) {
        int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
        for (size_t axis = 0; axis < entry.ne.size() && axis < GGML_MAX_DIMS; ++axis) {
            ne[axis] = entry.ne[axis];
        }
        if (synth::omnivoice::classify_tensor(entry.name, ne) == synth::omnivoice::QuantRole::MatrixWeight) {
            if (entry.ne.size() == 3) {
                out.emplace_back(
                    Entry{
                        entry.name, { entry.ne[0] * entry.ne[1], entry.ne[2] }
                },
                    GGML_TYPE_Q8_0);
            } else {
                out.emplace_back(entry, GGML_TYPE_Q8_0);
            }
        } else {
            out.emplace_back(entry, GGML_TYPE_F32);
        }
    }
    return out;
}

void populate_typed(ggml_context * context, const std::vector<std::pair<Entry, ggml_type>> & entries) {
    for (const auto & [entry, type] : entries) {
        int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
        for (size_t axis = 0; axis < entry.ne.size() && axis < GGML_MAX_DIMS; ++axis) {
            ne[axis] = entry.ne[axis];
        }
        ggml_tensor * tensor =
            ggml_new_tensor(context, type, int(std::min<size_t>(entry.ne.size(), GGML_MAX_DIMS)), ne);
        ggml_set_name(tensor, entry.name.c_str());
    }
}

int check_q8_mixed_resolution() {
    const synth::omnivoice::HParams                h           = q8_mixed_hparams();
    const std::vector<Entry>                       f32_entries = expected_entries(h);
    const std::vector<std::pair<Entry, ggml_type>> entries     = to_q8_mixed(f32_entries);

    Context context = make_context();
    populate_typed(context.get(), entries);

    synth::omnivoice::ModelWeights weights;
    SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), nullptr, h, weights) == SYNTH_OK);

    // A packed convolution kernel: its row is kernel * in_channels, flattened
    // out of the logical three-axis shape.
    SYNTH_TEST_CHECK(weights.acoustic_decoder.conv1.weight->type == GGML_TYPE_Q8_0);
    SYNTH_TEST_CHECK(weights.acoustic_decoder.conv1.weight->ne[0] == 7 * int64_t(h.codec.hidden_size));
    SYNTH_TEST_CHECK(weights.acoustic_decoder.conv1.weight->ne[1] == h.codec.decoder_hidden_size);
    // The one collapsed conv kernel (mono exit, out=1): packing still applies.
    SYNTH_TEST_CHECK(weights.acoustic_decoder.conv2.weight->type == GGML_TYPE_Q8_0);
    // A native Linear inside the codec's HuBERT half: quantized in place,
    // shape unchanged.
    SYNTH_TEST_CHECK(weights.semantic_model.layers[0].q_proj.weight->type == GGML_TYPE_Q8_0);
    SYNTH_TEST_CHECK(weights.semantic_model.layers[0].q_proj.weight->ne[0] == int64_t(h.semantic.hidden_size));
    // A transposed convolution: never packed, whatever the profile.
    SYNTH_TEST_CHECK(weights.acoustic_decoder.blocks[0].conv_t1.weight->type == GGML_TYPE_F32);
    // The three named-Sensitive exceptions and a norm/bias/alpha stay F32.
    SYNTH_TEST_CHECK(weights.acoustic_encoder.conv1.weight->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.semantic_model.feat_conv[0].weight->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.semantic_model.pos_conv.weight->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.acoustic_decoder.blocks[0].snake1.alpha->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.acoustic_decoder.conv1.bias->type == GGML_TYPE_F32);
    // The whole generator and the RVQ stay exact under every profile.
    SYNTH_TEST_CHECK(weights.generator.text_embedding->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.generator.layers[0].q_proj->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.generator.audio_embeddings->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.quantizers[0].codebook->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.fc.weight->type == GGML_TYPE_F32);
    return 0;
}

int check_q8_mixed_rejections() {
    const synth::omnivoice::HParams h           = q8_mixed_hparams();
    const std::vector<Entry>        f32_entries = expected_entries(h);

    // A MatrixWeight tensor the offline quantizer never touched: still F32
    // (at its unpacked shape) under a profile that packs it.
    {
        std::vector<std::pair<Entry, ggml_type>> entries = to_q8_mixed(f32_entries);
        for (auto & [entry, type] : entries) {
            if (entry.name == "codec.acoustic_decoder.conv1.weight") {
                entry = Entry{
                    entry.name, { 7, int64_t(h.codec.hidden_size), int64_t(h.codec.decoder_hidden_size) }
                };
                type = GGML_TYPE_F32;
            }
        }
        Context                        context = make_context();
        synth::omnivoice::ModelWeights weights;
        populate_typed(context.get(), entries);
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), nullptr, h, weights) == SYNTH_ERR_GGUF);
    }

    // A packed convolution whose row does not match kernel * in_channels.
    {
        std::vector<std::pair<Entry, ggml_type>> entries = to_q8_mixed(f32_entries);
        for (auto & [entry, type] : entries) {
            if (entry.name == "codec.acoustic_decoder.conv1.weight") {
                entry.ne[0] += 32;
            }
        }
        Context                        context = make_context();
        synth::omnivoice::ModelWeights weights;
        populate_typed(context.get(), entries);
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), nullptr, h, weights) == SYNTH_ERR_GGUF);
    }

    // The generator stays exact under every profile; a halved generator
    // tensor is a package defect even where a codec MatrixWeight tensor at
    // the same position would legitimately pack.
    {
        std::vector<std::pair<Entry, ggml_type>> entries = to_q8_mixed(f32_entries);
        for (auto & [entry, type] : entries) {
            if (entry.name == "llm.norm.weight") {
                type = GGML_TYPE_F16;
            }
        }
        Context                        context = make_context();
        synth::omnivoice::ModelWeights weights;
        populate_typed(context.get(), entries);
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), nullptr, h, weights) == SYNTH_ERR_GGUF);
    }
    return 0;
}

// F16 (Plan 4 Task 3) never packs: the tool's profile table gives it
// TensorLayout::Native (tools/synthesize-quantize/policy.cpp:14-24), so a
// MatrixWeight tensor keeps its native shape and only its type changes --
// unlike to_q8_mixed above, no width needs choosing for block-size
// divisibility, so small_hparams()'s own widths are unmodified.
std::vector<std::pair<Entry, ggml_type>> to_f16(const std::vector<Entry> & entries) {
    std::vector<std::pair<Entry, ggml_type>> out;
    out.reserve(entries.size());
    for (const Entry & entry : entries) {
        int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
        for (size_t axis = 0; axis < entry.ne.size() && axis < GGML_MAX_DIMS; ++axis) {
            ne[axis] = entry.ne[axis];
        }
        const bool matrix_weight =
            synth::omnivoice::classify_tensor(entry.name, ne) == synth::omnivoice::QuantRole::MatrixWeight;
        out.emplace_back(entry, matrix_weight ? GGML_TYPE_F16 : GGML_TYPE_F32);
    }
    return out;
}

int check_f16_resolution() {
    synth::omnivoice::HParams h                                = small_hparams();
    h.quantization_profile                                     = synth::omnivoice::QuantizationProfile::F16;
    const std::vector<Entry>                       f32_entries = expected_entries(h);
    const std::vector<std::pair<Entry, ggml_type>> entries     = to_f16(f32_entries);

    Context context = make_context();
    populate_typed(context.get(), entries);

    synth::omnivoice::ModelWeights weights;
    SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), nullptr, h, weights) == SYNTH_OK);

    // A MatrixWeight conv kernel: F16, native (unpacked) three-axis shape.
    SYNTH_TEST_CHECK(weights.acoustic_decoder.conv1.weight->type == GGML_TYPE_F16);
    SYNTH_TEST_CHECK(weights.acoustic_decoder.conv1.weight->ne[0] == 7);
    SYNTH_TEST_CHECK(weights.acoustic_decoder.conv1.weight->ne[1] == int64_t(h.codec.hidden_size));
    SYNTH_TEST_CHECK(weights.acoustic_decoder.conv1.weight->ne[2] == int64_t(h.codec.decoder_hidden_size));
    // A native MatrixWeight Linear: F16, shape unchanged.
    SYNTH_TEST_CHECK(weights.semantic_model.layers[0].q_proj.weight->type == GGML_TYPE_F16);
    SYNTH_TEST_CHECK(weights.semantic_model.layers[0].q_proj.weight->ne[0] == int64_t(h.semantic.hidden_size));
    // A transposed convolution: never halved, whatever the profile.
    SYNTH_TEST_CHECK(weights.acoustic_decoder.blocks[0].conv_t1.weight->type == GGML_TYPE_F32);
    // The three named-Sensitive exceptions and a norm/bias/alpha stay F32.
    SYNTH_TEST_CHECK(weights.acoustic_encoder.conv1.weight->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.semantic_model.feat_conv[0].weight->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.semantic_model.pos_conv.weight->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.acoustic_decoder.blocks[0].snake1.alpha->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.acoustic_decoder.conv1.bias->type == GGML_TYPE_F32);
    // The whole generator and the RVQ stay exact under every profile.
    SYNTH_TEST_CHECK(weights.generator.text_embedding->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.generator.layers[0].q_proj->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.generator.audio_embeddings->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.quantizers[0].codebook->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.fc.weight->type == GGML_TYPE_F32);
    return 0;
}

int check_f16_rejections() {
    const synth::omnivoice::HParams h = [] {
        synth::omnivoice::HParams hp = small_hparams();
        hp.quantization_profile      = synth::omnivoice::QuantizationProfile::F16;
        return hp;
    }();
    const std::vector<Entry> f32_entries = expected_entries(h);

    // A MatrixWeight tensor the offline quantizer never touched: still F32
    // under a profile that halves it.
    {
        std::vector<std::pair<Entry, ggml_type>> entries = to_f16(f32_entries);
        for (auto & [entry, type] : entries) {
            if (entry.name == "codec.acoustic_decoder.conv1.weight") {
                type = GGML_TYPE_F32;
            }
        }
        Context                        context = make_context();
        synth::omnivoice::ModelWeights weights;
        populate_typed(context.get(), entries);
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), nullptr, h, weights) == SYNTH_ERR_GGUF);
    }

    // The generator stays exact under every profile; a halved generator
    // tensor is a package defect even under a profile that halves the codec.
    {
        std::vector<std::pair<Entry, ggml_type>> entries = to_f16(f32_entries);
        for (auto & [entry, type] : entries) {
            if (entry.name == "llm.norm.weight") {
                type = GGML_TYPE_F16;
            }
        }
        Context                        context = make_context();
        synth::omnivoice::ModelWeights weights;
        populate_typed(context.get(), entries);
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), nullptr, h, weights) == SYNTH_ERR_GGUF);
    }
    return 0;
}

}  // namespace

int main() {
    const synth::omnivoice::HParams h       = small_hparams();
    const std::vector<Entry>        entries = expected_entries(h);

    SYNTH_TEST_CHECK(check_resolution(h, entries) == 0);
    SYNTH_TEST_CHECK(check_rejections(h, entries) == 0);
    SYNTH_TEST_CHECK(check_twin_resolution(h, entries) == 0);
    SYNTH_TEST_CHECK(check_real_package_count() == 0);
    SYNTH_TEST_CHECK(check_q8_mixed_resolution() == 0);
    SYNTH_TEST_CHECK(check_q8_mixed_rejections() == 0);
    SYNTH_TEST_CHECK(check_f16_resolution() == 0);
    SYNTH_TEST_CHECK(check_f16_rejections() == 0);
    return 0;
}
