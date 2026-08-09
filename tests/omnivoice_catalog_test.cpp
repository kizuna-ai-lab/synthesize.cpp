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
    SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, weights) == SYNTH_OK);

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
    SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(nullptr, h, weights) == SYNTH_ERR_INVALID_ARG);

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
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, parsed) == SYNTH_ERR_GGUF);
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
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, parsed) == SYNTH_ERR_GGUF);
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
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, parsed) == SYNTH_ERR_GGUF);
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
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, parsed) == SYNTH_ERR_GGUF);
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
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, parsed) == SYNTH_ERR_GGUF);
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
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, parsed) == SYNTH_ERR_GGUF);
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
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), other, parsed) == SYNTH_ERR_GGUF);
    }

    // Hyper-parameters the resolver would have to index past: refused rather
    // than read out of bounds.
    {
        Context                        context = make_context();
        synth::omnivoice::HParams      other   = h;
        synth::omnivoice::ModelWeights parsed;
        other.semantic.conv_kernel.pop_back();
        populate(context.get(), entries, nullptr);
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), other, parsed) == SYNTH_ERR_GGUF);
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

// The twin context: bind_decode_weights must bind the three movable groups
// against it when present, WITHOUT EVER MUTATING `weights` itself -- the
// second-consumer trap fix-round-1 exists to close. `codec.quantizer.*` is
// read by two independent consumers (the decode graph AND rvq_encode's
// host-side argmax, through `weights.quantizers`), so the invariant this test
// pins is directional: `weights` must always stay bound to the package
// (`context`), and only `decode_weights` -- a distinct object, read only by
// Model::decode_codes -- may ever point at the twin.
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
    SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, weights) == SYNTH_OK);

    // `weights` itself is bound to the package alone, exactly as if
    // bind_decode_weights had never been called -- this is the fact rvq_encode
    // (reached through Model::encode_reference) depends on to never read a
    // twin.
    ggml_tensor * package_codebook = ggml_get_tensor(context.get(), "codec.quantizer.quantizers.0.codebook.embed");
    ggml_tensor * package_fc2      = ggml_get_tensor(context.get(), "codec.fc2.weight");
    ggml_tensor * package_decoder  = ggml_get_tensor(context.get(), "codec.acoustic_decoder.conv1.weight");
    SYNTH_TEST_CHECK(weights.quantizers[0].codebook == package_codebook);
    SYNTH_TEST_CHECK(weights.fc2.weight == package_fc2);
    SYNTH_TEST_CHECK(weights.acoustic_decoder.conv1.weight == package_decoder);

    // bind_decode_weights with NO twin: decode_weights is a plain copy, so
    // Model::decode_codes reads exactly what it always has when there is no
    // accelerator -- the CPU-identity property Task 9's own gate depends on.
    synth::omnivoice::ModelWeights no_twin_decode;
    SYNTH_TEST_CHECK(synth::omnivoice::bind_decode_weights(nullptr, h, weights, no_twin_decode) == SYNTH_OK);
    SYNTH_TEST_CHECK(no_twin_decode.quantizers[0].codebook == package_codebook);
    SYNTH_TEST_CHECK(no_twin_decode.fc2.weight == package_fc2);
    SYNTH_TEST_CHECK(no_twin_decode.acoustic_decoder.conv1.weight == package_decoder);
    // Every non-movable field is carried over unchanged too -- the copy is a
    // whole-struct one, not a field-by-field reconstruction that could drift.
    SYNTH_TEST_CHECK(no_twin_decode.fc.weight == weights.fc.weight);
    SYNTH_TEST_CHECK(no_twin_decode.acoustic_encoder.conv1.weight == weights.acoustic_encoder.conv1.weight);
    SYNTH_TEST_CHECK(no_twin_decode.semantic_model.feat_conv[0].weight == weights.semantic_model.feat_conv[0].weight);
    SYNTH_TEST_CHECK(no_twin_decode.encoder_semantic.conv.weight == weights.encoder_semantic.conv.weight);

    // bind_decode_weights WITH a twin: decode_weights's three movable fields
    // move to the twin; `weights` itself must not have changed AT ALL, in
    // either direction -- neither consumer may silently start reading the
    // other's copy.
    synth::omnivoice::ModelWeights decode_weights;
    SYNTH_TEST_CHECK(synth::omnivoice::bind_decode_weights(twin.get(), h, weights, decode_weights) == SYNTH_OK);

    SYNTH_TEST_CHECK(decode_weights.quantizers[0].codebook ==
                     ggml_get_tensor(twin.get(), "codec.quantizer.quantizers.0.codebook.embed"));
    SYNTH_TEST_CHECK(decode_weights.quantizers[0].codebook != package_codebook);
    SYNTH_TEST_CHECK(decode_weights.fc2.weight == ggml_get_tensor(twin.get(), "codec.fc2.weight"));
    SYNTH_TEST_CHECK(decode_weights.fc2.weight != package_fc2);
    SYNTH_TEST_CHECK(decode_weights.acoustic_decoder.conv1.weight ==
                     ggml_get_tensor(twin.get(), "codec.acoustic_decoder.conv1.weight"));
    SYNTH_TEST_CHECK(decode_weights.acoustic_decoder.conv1.weight != package_decoder);

    // `weights` -- the one rvq_encode reads -- is untouched: still bound to
    // the package, identical to what it was before bind_decode_weights ran at
    // all. This is the exact assertion the second-consumer trap's first draft
    // would have failed: that draft overwrote these same three pointers in
    // place.
    SYNTH_TEST_CHECK(weights.quantizers[0].codebook == package_codebook);
    SYNTH_TEST_CHECK(weights.fc2.weight == package_fc2);
    SYNTH_TEST_CHECK(weights.acoustic_decoder.conv1.weight == package_decoder);

    // Every non-movable field of decode_weights still traces back to the
    // package (through the initial whole-struct copy) even when a twin
    // exists: the twin never carries them, and a filter that mirrored them
    // too -- qwen3-tts's own blanket `codec.` prefix, copied without
    // narrowing -- would move weight the CPU-held clone-encode chain reads to
    // an accelerator no primary-side graph shares with it.
    SYNTH_TEST_CHECK(decode_weights.fc.weight == weights.fc.weight);
    SYNTH_TEST_CHECK(decode_weights.acoustic_encoder.conv1.weight == weights.acoustic_encoder.conv1.weight);
    SYNTH_TEST_CHECK(decode_weights.semantic_model.feat_conv[0].weight == weights.semantic_model.feat_conv[0].weight);
    SYNTH_TEST_CHECK(decode_weights.encoder_semantic.conv.weight == weights.encoder_semantic.conv.weight);

    // A twin missing one of the movable tensors is refused, the same as a
    // missing package tensor -- and `weights` still must not move.
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
        SYNTH_TEST_CHECK(synth::omnivoice::bind_decode_weights(broken_twin.get(), h, weights, parsed) ==
                         SYNTH_ERR_GGUF);
        SYNTH_TEST_CHECK(weights.quantizers[0].codebook == package_codebook);
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
        SYNTH_TEST_CHECK(synth::omnivoice::bind_decode_weights(broken_twin.get(), h, weights, parsed) ==
                         SYNTH_ERR_GGUF);
        SYNTH_TEST_CHECK(weights.acoustic_decoder.conv1.weight == package_decoder);
    }
    return 0;
}

// Plan 5 Task 1's own filter, restated independently here for the same reason
// is_decode_path_entry above is: a second, independent statement of
// model.cpp's is_generator_tensor, so the two cannot silently drift apart.
bool is_generator_entry(const Entry & entry) {
    if (entry.name.rfind("llm.", 0) == 0) {
        return true;
    }
    return entry.name == "audio_embeddings.weight" || entry.name == "audio_heads.weight";
}

// The generator twin (Plan 5 Task 1): bind_generator_weights must bind the
// WHOLE generator group against it when present, WITHOUT EVER MUTATING
// `weights` itself -- the same second-consumer-safe discipline
// check_twin_resolution above pins for the codec's narrower twin, applied
// here even though Task 1's own pre-flight grep found no second host-side
// consumer of GeneratorWeights: a future reader that bypasses
// generator_branch_forward must keep seeing the CPU-resident package by
// construction, not by continued vigilance.
int check_generator_twin_resolution(const synth::omnivoice::HParams & h, const std::vector<Entry> & entries) {
    Context context = make_context();
    populate(context.get(), entries, nullptr);

    std::vector<Entry> generator_entries;
    for (const Entry & entry : entries) {
        if (is_generator_entry(entry)) {
            generator_entries.push_back(entry);
        }
    }
    // 26 at this synthetic package's reduced widths: embed_tokens (1) + 2
    // layers * 11 tensors each (22) + norm (1) + the two audio tables (2).
    // The real package's numbers -- 312 total, catalog.h's own
    // bind_generator_weights accounting -- are this same arithmetic at
    // h.generator.layer_count == 28.
    SYNTH_TEST_CHECK(generator_entries.size() == 26);

    Context twin = make_context();
    populate(twin.get(), generator_entries, nullptr);

    synth::omnivoice::ModelWeights weights;
    SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, weights) == SYNTH_OK);

    // `weights` itself is bound to the package alone -- the fact a future
    // second consumer would depend on, the same as the codec twin's own
    // invariant above.
    ggml_tensor * package_embed = ggml_get_tensor(context.get(), "llm.embed_tokens.weight");
    ggml_tensor * package_norm  = ggml_get_tensor(context.get(), "llm.norm.weight");
    ggml_tensor * package_heads = ggml_get_tensor(context.get(), "audio_heads.weight");
    SYNTH_TEST_CHECK(weights.generator.text_embedding == package_embed);
    SYNTH_TEST_CHECK(weights.generator.norm == package_norm);
    SYNTH_TEST_CHECK(weights.generator.audio_heads == package_heads);

    // bind_generator_weights with NO twin: generator_weights is a plain copy,
    // so generator_branch_forward reads exactly what it always has when there
    // is no accelerator -- the CPU-identity property Task 1's own gate
    // depends on.
    synth::omnivoice::ModelWeights no_twin_generator;
    SYNTH_TEST_CHECK(synth::omnivoice::bind_generator_weights(nullptr, h, weights, no_twin_generator) == SYNTH_OK);
    SYNTH_TEST_CHECK(no_twin_generator.generator.text_embedding == package_embed);
    SYNTH_TEST_CHECK(no_twin_generator.generator.norm == package_norm);
    SYNTH_TEST_CHECK(no_twin_generator.generator.audio_heads == package_heads);
    // Every non-generator field is carried over unchanged too -- the copy is a
    // whole-struct one, not a field-by-field reconstruction that could drift.
    SYNTH_TEST_CHECK(no_twin_generator.fc2.weight == weights.fc2.weight);
    SYNTH_TEST_CHECK(no_twin_generator.acoustic_decoder.conv1.weight == weights.acoustic_decoder.conv1.weight);
    SYNTH_TEST_CHECK(no_twin_generator.quantizers[0].codebook == weights.quantizers[0].codebook);

    // bind_generator_weights WITH a twin: generator_weights's whole generator
    // group moves to the twin; `weights` itself must not have changed AT ALL.
    synth::omnivoice::ModelWeights generator_weights;
    SYNTH_TEST_CHECK(synth::omnivoice::bind_generator_weights(twin.get(), h, weights, generator_weights) == SYNTH_OK);

    SYNTH_TEST_CHECK(generator_weights.generator.text_embedding ==
                     ggml_get_tensor(twin.get(), "llm.embed_tokens.weight"));
    SYNTH_TEST_CHECK(generator_weights.generator.text_embedding != package_embed);
    SYNTH_TEST_CHECK(generator_weights.generator.norm == ggml_get_tensor(twin.get(), "llm.norm.weight"));
    SYNTH_TEST_CHECK(generator_weights.generator.norm != package_norm);
    SYNTH_TEST_CHECK(generator_weights.generator.audio_heads == ggml_get_tensor(twin.get(), "audio_heads.weight"));
    SYNTH_TEST_CHECK(generator_weights.generator.audio_heads != package_heads);
    SYNTH_TEST_CHECK(generator_weights.generator.layers[0].q_proj ==
                     ggml_get_tensor(twin.get(), "llm.layers.0.self_attn.q_proj.weight"));

    // `weights` -- what a future second consumer would depend on -- is
    // untouched: still bound to the package, identical to what it was before
    // bind_generator_weights ran at all.
    SYNTH_TEST_CHECK(weights.generator.text_embedding == package_embed);
    SYNTH_TEST_CHECK(weights.generator.norm == package_norm);
    SYNTH_TEST_CHECK(weights.generator.audio_heads == package_heads);

    // Every non-generator field of generator_weights still traces back to the
    // package even when a twin exists: the twin never carries them.
    SYNTH_TEST_CHECK(generator_weights.fc2.weight == weights.fc2.weight);
    SYNTH_TEST_CHECK(generator_weights.acoustic_decoder.conv1.weight == weights.acoustic_decoder.conv1.weight);
    SYNTH_TEST_CHECK(generator_weights.quantizers[0].codebook == weights.quantizers[0].codebook);

    // A twin missing one of the generator's tensors is refused, the same as a
    // missing package tensor -- and `weights` still must not move.
    {
        std::vector<Entry> incomplete;
        for (const Entry & entry : generator_entries) {
            if (entry.name != "llm.norm.weight") {
                incomplete.push_back(entry);
            }
        }
        Context                        broken_twin = make_context();
        synth::omnivoice::ModelWeights parsed;
        populate(broken_twin.get(), incomplete, nullptr);
        SYNTH_TEST_CHECK(synth::omnivoice::bind_generator_weights(broken_twin.get(), h, weights, parsed) ==
                         SYNTH_ERR_GGUF);
        SYNTH_TEST_CHECK(weights.generator.text_embedding == package_embed);
    }

    // A twin whose tensor disagrees in shape is refused too.
    {
        Context                        broken_twin = make_context();
        synth::omnivoice::ModelWeights parsed;
        populate(broken_twin.get(), generator_entries, [](const Entry & entry, Entry & effective, ggml_type &) {
            if (entry.name == "llm.embed_tokens.weight") {
                effective.ne[0] += 1;
                return true;
            }
            return false;
        });
        SYNTH_TEST_CHECK(synth::omnivoice::bind_generator_weights(broken_twin.get(), h, weights, parsed) ==
                         SYNTH_ERR_GGUF);
        SYNTH_TEST_CHECK(weights.generator.text_embedding == package_embed);
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
// for it under Q8_MIXED. Since the conv-exempt policy of 2026-08-09 that is:
// a MatrixWeight tensor -- necessarily a two-axis HuBERT Linear -- becomes
// Q8_0 at its own shape, a ConvKernel is halved to F16 at its native
// three-axis shape rather than flattened to [kernel * in, out], and every
// other role stays F32. Nothing is packed, which is why no entry's shape is
// rewritten here any more.
std::vector<std::pair<Entry, ggml_type>> to_q8_mixed(const std::vector<Entry> & entries) {
    std::vector<std::pair<Entry, ggml_type>> out;
    out.reserve(entries.size());
    for (const Entry & entry : entries) {
        int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
        for (size_t axis = 0; axis < entry.ne.size() && axis < GGML_MAX_DIMS; ++axis) {
            ne[axis] = entry.ne[axis];
        }
        ggml_type type = GGML_TYPE_F32;
        switch (synth::omnivoice::classify_tensor_for_half(entry.name, ne, synth::omnivoice::ModelHalf::Codec)) {
            case synth::omnivoice::QuantRole::MatrixWeight:
                type = GGML_TYPE_Q8_0;
                break;
            case synth::omnivoice::QuantRole::ConvKernel:
                type = GGML_TYPE_F16;
                break;
            default:
                break;
        }
        out.emplace_back(entry, type);
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
    SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, weights) == SYNTH_OK);

    // A convolution kernel, since the conv-exempt policy of 2026-08-09:
    // halved, never packed, and keeping its logical three-axis shape. Before
    // the policy this was Q8_0 with ne[0] == 7 * in_channels.
    SYNTH_TEST_CHECK(weights.acoustic_decoder.conv1.weight->type == GGML_TYPE_F16);
    SYNTH_TEST_CHECK(weights.acoustic_decoder.conv1.weight->ne[0] == 7);
    SYNTH_TEST_CHECK(weights.acoustic_decoder.conv1.weight->ne[1] == int64_t(h.codec.hidden_size));
    SYNTH_TEST_CHECK(weights.acoustic_decoder.conv1.weight->ne[2] == h.codec.decoder_hidden_size);
    // The one collapsed conv kernel (mono exit, out=1) -- the tensor a
    // shape-based conv-exempt rule would misfile as a Linear.
    SYNTH_TEST_CHECK(weights.acoustic_decoder.conv2.weight->type == GGML_TYPE_F16);
    // A native Linear inside the codec's HuBERT half, and the only thing this
    // profile still block-quantizes: quantized in place, shape unchanged.
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
    // The whole generator stays exact under a codec-half profile, and the RVQ
    // under every profile there is.
    SYNTH_TEST_CHECK(weights.generator.text_embedding->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.generator.layers[0].q_proj->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.generator.audio_embeddings->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.generator.audio_heads->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.quantizers[0].codebook->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.fc.weight->type == GGML_TYPE_F32);
    return 0;
}

int check_q8_mixed_rejections() {
    const synth::omnivoice::HParams h           = q8_mixed_hparams();
    const std::vector<Entry>        f32_entries = expected_entries(h);

    // A convolution kernel the offline quantizer never halved: still F32
    // where the conv-exempt policy says F16.
    {
        std::vector<std::pair<Entry, ggml_type>> entries = to_q8_mixed(f32_entries);
        for (auto & [entry, type] : entries) {
            if (entry.name == "codec.acoustic_decoder.conv1.weight") {
                type = GGML_TYPE_F32;
            }
        }
        Context                        context = make_context();
        synth::omnivoice::ModelWeights weights;
        populate_typed(context.get(), entries);
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, weights) == SYNTH_ERR_GGUF);
    }

    // A PACKED Q8_0 convolution kernel -- exactly what a Q8_MIXED package cut
    // before 2026-08-09 contains. It must now be refused rather than
    // silently accepted by the packed-shape branch that is still in find():
    // the type check runs first, and the conv-exempt policy expects F16
    // there. This is the assertion that stops a pre-policy package loading
    // under the profile name it still carries.
    {
        std::vector<std::pair<Entry, ggml_type>> entries = to_q8_mixed(f32_entries);
        for (auto & [entry, type] : entries) {
            if (entry.name == "codec.acoustic_decoder.conv1.weight") {
                entry = Entry{
                    entry.name, { 7 * int64_t(h.codec.hidden_size), int64_t(h.codec.decoder_hidden_size) }
                };
                type = GGML_TYPE_Q8_0;
            }
        }
        Context                        context = make_context();
        synth::omnivoice::ModelWeights weights;
        populate_typed(context.get(), entries);
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, weights) == SYNTH_ERR_GGUF);
    }

    // A halved convolution whose kernel extent does not match the catalog's.
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
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, weights) == SYNTH_ERR_GGUF);
    }

    // And a Linear the quantizer skipped: F32 where the profile says Q8_0.
    // Paired with the convolution case above, this is what proves the split
    // is enforced in both directions rather than the loader simply accepting
    // anything narrower than F32.
    {
        std::vector<std::pair<Entry, ggml_type>> entries = to_q8_mixed(f32_entries);
        for (auto & [entry, type] : entries) {
            if (entry.name == "codec.semantic_model.encoder.layers.0.attn.q_proj.weight") {
                type = GGML_TYPE_F32;
            }
        }
        Context                        context = make_context();
        synth::omnivoice::ModelWeights weights;
        populate_typed(context.get(), entries);
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, weights) == SYNTH_ERR_GGUF);
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
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, weights) == SYNTH_ERR_GGUF);
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
        // A Linear and a convolution kernel land on the same F16 under this
        // profile -- its matrix weight type and its halved fallback column
        // are both F16 -- which is why the conv-exempt policy leaves an F16
        // package byte-identical to the one cut before it.
        const synth::omnivoice::QuantRole role =
            synth::omnivoice::classify_tensor_for_half(entry.name, ne, synth::omnivoice::ModelHalf::Codec);
        const bool halved =
            role == synth::omnivoice::QuantRole::MatrixWeight || role == synth::omnivoice::QuantRole::ConvKernel;
        out.emplace_back(entry, halved ? GGML_TYPE_F16 : GGML_TYPE_F32);
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
    SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, weights) == SYNTH_OK);

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
    // The whole generator stays exact under a codec-half profile, and the RVQ
    // under every profile there is.
    SYNTH_TEST_CHECK(weights.generator.text_embedding->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.generator.layers[0].q_proj->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.generator.audio_embeddings->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.generator.audio_heads->type == GGML_TYPE_F32);
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
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, weights) == SYNTH_ERR_GGUF);
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
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, weights) == SYNTH_ERR_GGUF);
    }
    return 0;
}

// Widths for the Q8_GEN variant of the small package. This profile quantizes
// the generator half instead of the codec, so it is the generator's widths
// that have to be multiples of Q8_0's 32-element block -- specifically every
// width that lands on a quantized tensor's *row* (ne[0]): `hidden_size`
// (which is the row of the embedding table, q/k/v_proj, gate/up_proj and both
// canvas tables), `intermediate_size` (down_proj's row) and
// `attention_head_count * head_dim` (o_proj's row). The key/value width is
// only ever a column, so `key_value_head_count` is untouched, and so is the
// entire codec half. The real checkpoint's rows are 1024, 2048 and 3072 and
// need no such adjustment at all.
synth::omnivoice::HParams q8_gen_hparams() {
    synth::omnivoice::HParams h      = small_hparams();
    h.quantization_profile           = synth::omnivoice::QuantizationProfile::Q8Gen;
    h.generator.hidden_size          = 32;
    h.generator.head_dim             = 8;
    h.generator.attention_head_count = 4;
    h.generator.key_value_head_count = 2;
    h.generator.intermediate_size    = 32;
    return h;
}

// Recasts one F32 entry list to what the offline quantizer would have written
// for it under Q8_GEN: every generator MatrixWeight and RowLookup tensor
// becomes Q8_0 at its declared shape (nothing is packed -- the generator half
// is two-dimensional throughout), and every other tensor, the whole codec
// included, stays F32.
std::vector<std::pair<Entry, ggml_type>> to_q8_gen(const std::vector<Entry> & entries) {
    std::vector<std::pair<Entry, ggml_type>> out;
    out.reserve(entries.size());
    for (const Entry & entry : entries) {
        int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
        for (size_t axis = 0; axis < entry.ne.size() && axis < GGML_MAX_DIMS; ++axis) {
            ne[axis] = entry.ne[axis];
        }
        const synth::omnivoice::QuantRole role =
            synth::omnivoice::classify_tensor_for_half(entry.name, ne, synth::omnivoice::ModelHalf::Generator);
        const bool quantized =
            role == synth::omnivoice::QuantRole::MatrixWeight || role == synth::omnivoice::QuantRole::RowLookup;
        out.emplace_back(entry, quantized ? GGML_TYPE_Q8_0 : GGML_TYPE_F32);
    }
    return out;
}

int check_q8_gen_resolution() {
    const synth::omnivoice::HParams                h           = q8_gen_hparams();
    const std::vector<Entry>                       f32_entries = expected_entries(h);
    const std::vector<std::pair<Entry, ggml_type>> entries     = to_q8_gen(f32_entries);

    Context context = make_context();
    populate_typed(context.get(), entries);

    synth::omnivoice::ModelWeights weights;
    SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, weights) == SYNTH_OK);

    // Both `ggml_get_rows` tables, and both at their declared shape: a packed
    // lookup table would be unindexable.
    SYNTH_TEST_CHECK(weights.generator.text_embedding->type == GGML_TYPE_Q8_0);
    SYNTH_TEST_CHECK(weights.generator.text_embedding->ne[0] == int64_t(h.generator.hidden_size));
    SYNTH_TEST_CHECK(weights.generator.text_embedding->ne[1] == int64_t(h.generator.text_vocab_size));
    SYNTH_TEST_CHECK(weights.generator.audio_embeddings->type == GGML_TYPE_Q8_0);
    SYNTH_TEST_CHECK(weights.generator.audio_embeddings->ne[0] == int64_t(h.generator.hidden_size));
    // The heads are a plain matrix multiply, not the embeddings' transpose.
    SYNTH_TEST_CHECK(weights.generator.audio_heads->type == GGML_TYPE_Q8_0);

    // The seven projections per layer.
    SYNTH_TEST_CHECK(weights.generator.layers[0].q_proj->type == GGML_TYPE_Q8_0);
    SYNTH_TEST_CHECK(weights.generator.layers[0].k_proj->type == GGML_TYPE_Q8_0);
    SYNTH_TEST_CHECK(weights.generator.layers[0].v_proj->type == GGML_TYPE_Q8_0);
    SYNTH_TEST_CHECK(weights.generator.layers[0].o_proj->type == GGML_TYPE_Q8_0);
    SYNTH_TEST_CHECK(weights.generator.layers[0].gate_proj->type == GGML_TYPE_Q8_0);
    SYNTH_TEST_CHECK(weights.generator.layers[0].up_proj->type == GGML_TYPE_Q8_0);
    SYNTH_TEST_CHECK(weights.generator.layers[0].down_proj->type == GGML_TYPE_Q8_0);
    SYNTH_TEST_CHECK(weights.generator.layers[0].q_proj->ne[0] == int64_t(h.generator.hidden_size));

    // Every norm in the half stays exact.
    SYNTH_TEST_CHECK(weights.generator.norm->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.generator.layers[0].input_layernorm->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.generator.layers[0].post_attention_layernorm->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.generator.layers[0].q_norm->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.generator.layers[0].k_norm->type == GGML_TYPE_F32);

    // And the whole codec half, at its own unpacked shape: the tensors a
    // codec-half profile would pack, halve or hold native are all untouched
    // here, which is what makes this package's codec byte-identical to F32.
    SYNTH_TEST_CHECK(weights.acoustic_decoder.conv1.weight->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.acoustic_decoder.conv1.weight->ne[0] == 7);
    SYNTH_TEST_CHECK(weights.acoustic_decoder.conv1.weight->ne[1] == int64_t(h.codec.hidden_size));
    SYNTH_TEST_CHECK(weights.acoustic_decoder.blocks[0].conv_t1.weight->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.semantic_model.layers[0].q_proj.weight->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.quantizers[0].codebook->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.fc.weight->type == GGML_TYPE_F32);
    return 0;
}

int check_q8_gen_rejections() {
    const synth::omnivoice::HParams h           = q8_gen_hparams();
    const std::vector<Entry>        f32_entries = expected_entries(h);

    // A generator matrix the offline quantizer never touched.
    {
        std::vector<std::pair<Entry, ggml_type>> entries = to_q8_gen(f32_entries);
        for (auto & [entry, type] : entries) {
            if (entry.name == "llm.layers.0.self_attn.q_proj.weight") {
                type = GGML_TYPE_F32;
            }
        }
        Context                        context = make_context();
        synth::omnivoice::ModelWeights weights;
        populate_typed(context.get(), entries);
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, weights) == SYNTH_ERR_GGUF);
    }

    // A generator norm that was quantized anyway.
    {
        std::vector<std::pair<Entry, ggml_type>> entries = to_q8_gen(f32_entries);
        for (auto & [entry, type] : entries) {
            if (entry.name == "llm.norm.weight") {
                type = GGML_TYPE_F16;
            }
        }
        Context                        context = make_context();
        synth::omnivoice::ModelWeights weights;
        populate_typed(context.get(), entries);
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, weights) == SYNTH_ERR_GGUF);
    }

    // A codec tensor this profile must not have touched. This is the check
    // that would fire if the half split leaked -- the failure mode that makes
    // Q8_GEN's "codec byte-identical to F32" claim worth asserting at load
    // time and not only at cut time.
    {
        std::vector<std::pair<Entry, ggml_type>> entries = to_q8_gen(f32_entries);
        for (auto & [entry, type] : entries) {
            if (entry.name == "codec.semantic_model.encoder.layers.0.attn.q_proj.weight") {
                type = GGML_TYPE_Q8_0;
            }
        }
        Context                        context = make_context();
        synth::omnivoice::ModelWeights weights;
        populate_typed(context.get(), entries);
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, weights) == SYNTH_ERR_GGUF);
    }
    return 0;
}

// Widths for the Q4_K_GEN variant. Q4_K's super-block is 256 against Q8_0's
// 32, so every width that lands on a quantized tensor's row has to be a
// multiple of 256 rather than 32 -- the same three widths q8_gen_hparams()
// names, raised. `head_dim` moves with them because `attention_head_count *
// head_dim` is o_proj's row; the key/value width is still only ever a column.
// The real checkpoint's rows are 1024, 2048 and 3072, all multiples of 256, so
// this is the small package catching up with the real one rather than a
// concession the real one also needs.
synth::omnivoice::HParams q4_k_gen_hparams() {
    synth::omnivoice::HParams h      = small_hparams();
    h.quantization_profile           = synth::omnivoice::QuantizationProfile::Q4KGen;
    h.generator.hidden_size          = 256;
    h.generator.head_dim             = 64;
    h.generator.attention_head_count = 4;
    h.generator.key_value_head_count = 2;
    h.generator.intermediate_size    = 256;
    return h;
}

// The recast, and the one place in this file where two roles diverge: a
// MatrixWeight goes to Q4_K, a RowLookup holds at Q8_0. Under Q8_GEN both are
// Q8_0 and to_q8_gen can treat them as one case; here it cannot, because
// CUDA's GET_ROWS accepts no k-quant.
std::vector<std::pair<Entry, ggml_type>> to_q4_k_gen(const std::vector<Entry> & entries) {
    std::vector<std::pair<Entry, ggml_type>> out;
    out.reserve(entries.size());
    for (const Entry & entry : entries) {
        int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
        for (size_t axis = 0; axis < entry.ne.size() && axis < GGML_MAX_DIMS; ++axis) {
            ne[axis] = entry.ne[axis];
        }
        ggml_type type = GGML_TYPE_F32;
        switch (synth::omnivoice::classify_tensor_for_half(entry.name, ne, synth::omnivoice::ModelHalf::Generator)) {
            case synth::omnivoice::QuantRole::MatrixWeight:
                type = GGML_TYPE_Q4_K;
                break;
            case synth::omnivoice::QuantRole::RowLookup:
                type = GGML_TYPE_Q8_0;
                break;
            default:
                break;
        }
        out.emplace_back(entry, type);
    }
    return out;
}

int check_q4_k_gen_resolution() {
    const synth::omnivoice::HParams                h           = q4_k_gen_hparams();
    const std::vector<Entry>                       f32_entries = expected_entries(h);
    const std::vector<std::pair<Entry, ggml_type>> entries     = to_q4_k_gen(f32_entries);

    Context context = make_context();
    populate_typed(context.get(), entries);

    synth::omnivoice::ModelWeights weights;
    SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, weights) == SYNTH_OK);

    // The pin, at load time. These two are Q8_0 in a file whose every other
    // quantized generator tensor is Q4_K.
    SYNTH_TEST_CHECK(weights.generator.text_embedding->type == GGML_TYPE_Q8_0);
    SYNTH_TEST_CHECK(weights.generator.text_embedding->ne[0] == int64_t(h.generator.hidden_size));
    SYNTH_TEST_CHECK(weights.generator.text_embedding->ne[1] == int64_t(h.generator.text_vocab_size));
    SYNTH_TEST_CHECK(weights.generator.audio_embeddings->type == GGML_TYPE_Q8_0);
    SYNTH_TEST_CHECK(weights.generator.audio_embeddings->ne[0] == int64_t(h.generator.hidden_size));

    // `audio_heads` shares the embedding tables' shape and is not one of them:
    // a plain matrix multiply, so it takes the matrix type.
    SYNTH_TEST_CHECK(weights.generator.audio_heads->type == GGML_TYPE_Q4_K);

    // The seven projections per layer.
    SYNTH_TEST_CHECK(weights.generator.layers[0].q_proj->type == GGML_TYPE_Q4_K);
    SYNTH_TEST_CHECK(weights.generator.layers[0].k_proj->type == GGML_TYPE_Q4_K);
    SYNTH_TEST_CHECK(weights.generator.layers[0].v_proj->type == GGML_TYPE_Q4_K);
    SYNTH_TEST_CHECK(weights.generator.layers[0].o_proj->type == GGML_TYPE_Q4_K);
    SYNTH_TEST_CHECK(weights.generator.layers[0].gate_proj->type == GGML_TYPE_Q4_K);
    SYNTH_TEST_CHECK(weights.generator.layers[0].up_proj->type == GGML_TYPE_Q4_K);
    SYNTH_TEST_CHECK(weights.generator.layers[0].down_proj->type == GGML_TYPE_Q4_K);
    SYNTH_TEST_CHECK(weights.generator.layers[0].q_proj->ne[0] == int64_t(h.generator.hidden_size));

    // Norms exact, and the whole codec half untouched.
    SYNTH_TEST_CHECK(weights.generator.norm->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.generator.layers[0].q_norm->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.acoustic_decoder.conv1.weight->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.acoustic_decoder.blocks[0].conv_t1.weight->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.semantic_model.layers[0].q_proj.weight->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.quantizers[0].codebook->type == GGML_TYPE_F32);
    return 0;
}

int check_q4_k_gen_rejections() {
    const synth::omnivoice::HParams h           = q4_k_gen_hparams();
    const std::vector<Entry>        f32_entries = expected_entries(h);

    // The rejection this profile exists to make, and the only one in this file
    // that a wrong-but-plausible package would actually hit: a lookup table
    // k-quantized along with everything else. Without the RowLookup arm in
    // catalog.cpp's expected_type, the load succeeds and the accelerator
    // quietly runs both embedding lookups on the CPU.
    for (const char * table : { "llm.embed_tokens.weight", "audio_embeddings.weight" }) {
        std::vector<std::pair<Entry, ggml_type>> entries = to_q4_k_gen(f32_entries);
        for (auto & [entry, type] : entries) {
            if (entry.name == table) {
                type = GGML_TYPE_Q4_K;
            }
        }
        Context                        context = make_context();
        synth::omnivoice::ModelWeights weights;
        populate_typed(context.get(), entries);
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, weights) == SYNTH_ERR_GGUF);
    }

    // The mirror: a matrix left at the sibling profile's Q8_0. This is what a
    // Q4_K_GEN package cut before the profile row existed would look like.
    {
        std::vector<std::pair<Entry, ggml_type>> entries = to_q4_k_gen(f32_entries);
        for (auto & [entry, type] : entries) {
            if (entry.name == "llm.layers.0.self_attn.q_proj.weight") {
                type = GGML_TYPE_Q8_0;
            }
        }
        Context                        context = make_context();
        synth::omnivoice::ModelWeights weights;
        populate_typed(context.get(), entries);
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, weights) == SYNTH_ERR_GGUF);
    }

    // And a codec tensor this profile must not have touched.
    {
        std::vector<std::pair<Entry, ggml_type>> entries = to_q4_k_gen(f32_entries);
        for (auto & [entry, type] : entries) {
            if (entry.name == "codec.semantic_model.encoder.layers.0.attn.q_proj.weight") {
                type = GGML_TYPE_Q8_0;
            }
        }
        Context                        context = make_context();
        synth::omnivoice::ModelWeights weights;
        populate_typed(context.get(), entries);
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, weights) == SYNTH_ERR_GGUF);
    }
    return 0;
}

// F16_GEN and BF16_GEN (provisional names, see weights.h) quantize the same
// generator half as Q8_GEN, narrowed rather than block-quantized. Neither
// format is `ggml_is_quantized`, so -- like the codec-half F16 profile above,
// and unlike Q8_GEN/Q4_K_GEN -- there is no block-size divisibility to
// satisfy and small_hparams()'s own widths are unmodified.
std::vector<std::pair<Entry, ggml_type>> to_generator_narrowed(const std::vector<Entry> & entries, ggml_type narrow) {
    std::vector<std::pair<Entry, ggml_type>> out;
    out.reserve(entries.size());
    for (const Entry & entry : entries) {
        int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
        for (size_t axis = 0; axis < entry.ne.size() && axis < GGML_MAX_DIMS; ++axis) {
            ne[axis] = entry.ne[axis];
        }
        const synth::omnivoice::QuantRole role =
            synth::omnivoice::classify_tensor_for_half(entry.name, ne, synth::omnivoice::ModelHalf::Generator);
        const bool narrowed =
            role == synth::omnivoice::QuantRole::MatrixWeight || role == synth::omnivoice::QuantRole::RowLookup;
        out.emplace_back(entry, narrowed ? narrow : GGML_TYPE_F32);
    }
    return out;
}

int check_generator_narrowed_resolution(synth::omnivoice::QuantizationProfile profile, ggml_type narrow) {
    synth::omnivoice::HParams h                                = small_hparams();
    h.quantization_profile                                     = profile;
    const std::vector<Entry>                       f32_entries = expected_entries(h);
    const std::vector<std::pair<Entry, ggml_type>> entries     = to_generator_narrowed(f32_entries, narrow);

    Context context = make_context();
    populate_typed(context.get(), entries);

    synth::omnivoice::ModelWeights weights;
    SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, weights) == SYNTH_OK);

    // Both `ggml_get_rows` tables, at their declared shape: row_lookup_type
    // equals matrix_weight_type for both these profiles, unlike Q4_K_GEN.
    SYNTH_TEST_CHECK(weights.generator.text_embedding->type == narrow);
    SYNTH_TEST_CHECK(weights.generator.text_embedding->ne[0] == int64_t(h.generator.hidden_size));
    SYNTH_TEST_CHECK(weights.generator.text_embedding->ne[1] == int64_t(h.generator.text_vocab_size));
    SYNTH_TEST_CHECK(weights.generator.audio_embeddings->type == narrow);
    SYNTH_TEST_CHECK(weights.generator.audio_embeddings->ne[0] == int64_t(h.generator.hidden_size));
    // The heads are a plain matrix multiply, not the embeddings' transpose.
    SYNTH_TEST_CHECK(weights.generator.audio_heads->type == narrow);

    // The seven projections per layer.
    SYNTH_TEST_CHECK(weights.generator.layers[0].q_proj->type == narrow);
    SYNTH_TEST_CHECK(weights.generator.layers[0].k_proj->type == narrow);
    SYNTH_TEST_CHECK(weights.generator.layers[0].v_proj->type == narrow);
    SYNTH_TEST_CHECK(weights.generator.layers[0].o_proj->type == narrow);
    SYNTH_TEST_CHECK(weights.generator.layers[0].gate_proj->type == narrow);
    SYNTH_TEST_CHECK(weights.generator.layers[0].up_proj->type == narrow);
    SYNTH_TEST_CHECK(weights.generator.layers[0].down_proj->type == narrow);
    SYNTH_TEST_CHECK(weights.generator.layers[0].q_proj->ne[0] == int64_t(h.generator.hidden_size));

    // Every norm in the half stays exact.
    SYNTH_TEST_CHECK(weights.generator.norm->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.generator.layers[0].input_layernorm->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.generator.layers[0].post_attention_layernorm->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.generator.layers[0].q_norm->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.generator.layers[0].k_norm->type == GGML_TYPE_F32);

    // And the whole codec half, at its own unpacked shape -- byte-identical to
    // F32, the same claim Q8_GEN makes.
    SYNTH_TEST_CHECK(weights.acoustic_decoder.conv1.weight->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.acoustic_decoder.conv1.weight->ne[0] == 7);
    SYNTH_TEST_CHECK(weights.acoustic_decoder.conv1.weight->ne[1] == int64_t(h.codec.hidden_size));
    SYNTH_TEST_CHECK(weights.acoustic_decoder.blocks[0].conv_t1.weight->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.semantic_model.layers[0].q_proj.weight->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.quantizers[0].codebook->type == GGML_TYPE_F32);
    SYNTH_TEST_CHECK(weights.fc.weight->type == GGML_TYPE_F32);
    return 0;
}

int check_generator_narrowed_rejections(synth::omnivoice::QuantizationProfile profile, ggml_type narrow) {
    synth::omnivoice::HParams h          = small_hparams();
    h.quantization_profile               = profile;
    const std::vector<Entry> f32_entries = expected_entries(h);

    // A generator matrix the offline quantizer never touched.
    {
        std::vector<std::pair<Entry, ggml_type>> entries = to_generator_narrowed(f32_entries, narrow);
        for (auto & [entry, type] : entries) {
            if (entry.name == "llm.layers.0.self_attn.q_proj.weight") {
                type = GGML_TYPE_F32;
            }
        }
        Context                        context = make_context();
        synth::omnivoice::ModelWeights weights;
        populate_typed(context.get(), entries);
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, weights) == SYNTH_ERR_GGUF);
    }

    // A generator norm that was narrowed anyway.
    {
        std::vector<std::pair<Entry, ggml_type>> entries = to_generator_narrowed(f32_entries, narrow);
        for (auto & [entry, type] : entries) {
            if (entry.name == "llm.norm.weight") {
                type = narrow;
            }
        }
        Context                        context = make_context();
        synth::omnivoice::ModelWeights weights;
        populate_typed(context.get(), entries);
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, weights) == SYNTH_ERR_GGUF);
    }

    // A codec tensor this profile must not have touched -- the failure mode
    // that would fire if the half split leaked.
    {
        std::vector<std::pair<Entry, ggml_type>> entries = to_generator_narrowed(f32_entries, narrow);
        for (auto & [entry, type] : entries) {
            if (entry.name == "codec.semantic_model.encoder.layers.0.attn.q_proj.weight") {
                type = narrow;
            }
        }
        Context                        context = make_context();
        synth::omnivoice::ModelWeights weights;
        populate_typed(context.get(), entries);
        SYNTH_TEST_CHECK(synth::omnivoice::build_model_weights(context.get(), h, weights) == SYNTH_ERR_GGUF);
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
    SYNTH_TEST_CHECK(check_generator_twin_resolution(h, entries) == 0);
    SYNTH_TEST_CHECK(check_real_package_count() == 0);
    SYNTH_TEST_CHECK(check_q8_mixed_resolution() == 0);
    SYNTH_TEST_CHECK(check_q8_mixed_rejections() == 0);
    SYNTH_TEST_CHECK(check_f16_resolution() == 0);
    SYNTH_TEST_CHECK(check_f16_rejections() == 0);
    SYNTH_TEST_CHECK(check_q8_gen_resolution() == 0);
    SYNTH_TEST_CHECK(check_q8_gen_rejections() == 0);
    SYNTH_TEST_CHECK(check_q4_k_gen_resolution() == 0);
    SYNTH_TEST_CHECK(check_q4_k_gen_rejections() == 0);
    SYNTH_TEST_CHECK(
        check_generator_narrowed_resolution(synth::omnivoice::QuantizationProfile::F16Gen, GGML_TYPE_F16) == 0);
    SYNTH_TEST_CHECK(
        check_generator_narrowed_rejections(synth::omnivoice::QuantizationProfile::F16Gen, GGML_TYPE_F16) == 0);
    SYNTH_TEST_CHECK(
        check_generator_narrowed_resolution(synth::omnivoice::QuantizationProfile::BF16Gen, GGML_TYPE_BF16) == 0);
    SYNTH_TEST_CHECK(
        check_generator_narrowed_rejections(synth::omnivoice::QuantizationProfile::BF16Gen, GGML_TYPE_BF16) == 0);
    return 0;
}
