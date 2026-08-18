// The x-vector's substitution into the talker prompt's speaker slot (Stage 2
// Plan 2 Task 10).
//
// Upstream puts the speaker embedding in exactly the slot a preset Voice's
// token embedding goes -- modeling_qwen3_tts.py:2087-2105 selects one or the
// other into the same position. This is a drop-in: the POSITION never moves,
// only where its embedding comes from does. Four rules are pinned here:
//
//   1. Host layer (talker-host.h/.cpp): an external speaker slot occupies the
//      same position a token speaker would, so the prompt is the same length
//      either way.
//   2. Host layer: the index the graph substitutes at follows the codec run,
//      which is one token longer when a language is named -- a hardcoded
//      index is right for exactly one of the two cases.
//   3. Host layer: the Stage 1 (preset-Voice) path is untouched -- no x-vector,
//      no index recorded.
//   4. Graph layer (talker.h/.cpp): build_talker_prefill_input refuses a
//      speaker_index it cannot honour, and a speaker_embedding whose width is
//      not the talker's hidden size, rather than substituting into whatever
//      row it lands on.
//
// A fifth check goes further than the brief's four: that a valid substitution
// actually lands the embedding at its index and leaves every other position
// exactly as the ordinary codec-accumulation path would. The nullptr checks
// above prove the graph refuses a malformed request; nothing else here proves
// build_talker_prefill_input's three-way ggml_acc split (talker.cpp) is
// wired correctly instead of merely shaped correctly -- an off-by-one there
// would still return a non-null tensor of the right shape.
//
// No GGUF and no loaded Model: synthetic HParams and hand-built TalkerWeights
// throughout, the way qwen3_tts_talker_test.cpp and
// qwen3_tts_voice_required_test.cpp already do.

#include "arch/qwen3-tts/talker-host.h"
#include "arch/qwen3-tts/talker.h"
#include "arch/qwen3-tts/weights.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "test-assert.h"

#include <cmath>
#include <cstdint>
#include <memory>
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

Context make_context(size_t bytes) {
    ggml_init_params parameters{};
    parameters.mem_size = bytes;
    parameters.no_alloc = true;
    return Context(ggml_init(parameters));
}

// A Base-shaped package's talker geometry and special tokens. Mirrors
// qwen3_tts_talker_test.cpp's prompt_hparams() and
// qwen3_tts_voice_required_test.cpp's base_hparams() -- the fields relevant to
// each are the ones build_talker_prompt/flatten_talker_prompt actually read.
synth::qwen3tts::HParams base_hparams() {
    synth::qwen3tts::HParams h;
    h.voice_mode              = synth::qwen3tts::VoiceMode::ProfileSources;
    h.has_speaker_encoder     = true;
    h.talker.code_group_count = 4;
    h.tokens.tts_bos          = 100;
    h.tokens.tts_eos          = 101;
    h.tokens.tts_pad          = 102;
    h.tokens.codec_bos        = 200;
    h.tokens.codec_eos        = 201;
    h.tokens.codec_pad        = 202;
    h.tokens.codec_think      = 203;
    h.tokens.codec_nothink    = 204;
    h.tokens.codec_think_bos  = 205;
    h.tokens.codec_think_eos  = 206;
    return h;
}

// A CustomVoice-shaped package: same tokens, no speaker encoder, a Preset
// Voice Catalog instead of Profile Sources. The Stage 1 path this exercises
// does not consult voice_mode either, but the label documents which real
// package shape a request like this comes from.
synth::qwen3tts::HParams customvoice_hparams() {
    synth::qwen3tts::HParams h = base_hparams();
    h.voice_mode               = synth::qwen3tts::VoiceMode::PresetCatalog;
    h.has_speaker_encoder      = false;
    return h;
}

synth::qwen3tts::TalkerPromptRequest request_with_language() {
    synth::qwen3tts::TalkerPromptRequest request;
    request.role_tokens    = { 10, 11, 12 };
    request.text_tokens    = { 20, 21, 22 };
    request.has_language   = true;
    request.language_token = 300;
    return request;
}

// An ordinary preset-Voice request: a codec-vocabulary speaker token, not an
// embedding.
synth::qwen3tts::TalkerPromptRequest request_with_token() {
    synth::qwen3tts::TalkerPromptRequest request = request_with_language();
    request.has_speaker                          = true;
    request.speaker_token                        = 4242;
    return request;
}

// An external-speaker request: has_speaker is still set (the slot is always
// occupied), but speaker_token is an inert placeholder -- the graph never
// reads its row -- and speaker_is_external picks the x-vector path.
// `has_language` is a parameter because it is what changes the codec run's
// length, and with it the index the graph substitutes at.
synth::qwen3tts::TalkerPromptRequest external_request(bool has_language) {
    synth::qwen3tts::TalkerPromptRequest request = request_with_token();
    request.has_language                         = has_language;
    request.speaker_is_external                  = true;
    return request;
}

// Rule 1: an external speaker slot occupies the same POSITION a token speaker
// would. A prompt one position shorter still synthesizes -- in the wrong
// voice at the wrong length -- which is why the length is asserted rather
// than the contents alone.
int test_an_external_speaker_slot_does_not_change_the_prompt_length() {
    const synth::qwen3tts::HParams h = base_hparams();

    synth::qwen3tts::TalkerPromptRequest with_token = request_with_token();

    synth::qwen3tts::TalkerPromptRequest with_x_vector = with_token;
    with_x_vector.speaker_is_external                  = true;

    synth::qwen3tts::TalkerPrompt a;
    synth::qwen3tts::TalkerPrompt b;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, with_token, a) == SYNTH_OK);
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, with_x_vector, b) == SYNTH_OK);
    SYNTH_TEST_CHECK(a.positions.size() == b.positions.size());
    SYNTH_TEST_CHECK(a.external_speaker_index == -1);
    SYNTH_TEST_CHECK(b.external_speaker_index >= 0);
    return 0;
}

// Rule 2: the index the graph substitutes at is an index into the flattened
// codec run, not into the prompt. Naming a language lengthens the codec
// stream by one, so a hardcoded index is right for exactly one of the two
// paths.
int test_the_substitution_index_follows_the_language_token() {
    const synth::qwen3tts::HParams h = base_hparams();

    synth::qwen3tts::TalkerPrompt named;
    synth::qwen3tts::TalkerPrompt automatic;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, external_request(/*has_language=*/true), named) ==
                     SYNTH_OK);
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, external_request(/*has_language=*/false), automatic) ==
                     SYNTH_OK);

    std::vector<int32_t> text;
    std::vector<int32_t> codec;
    std::vector<int32_t> acoustic;
    int64_t              offset          = 0;
    int64_t              acoustic_offset = 0;
    SYNTH_TEST_CHECK(synth::qwen3tts::flatten_talker_prompt(h, named, text, codec, offset, acoustic, acoustic_offset) ==
                     SYNTH_OK);
    SYNTH_TEST_CHECK(named.external_speaker_index == 4);
    SYNTH_TEST_CHECK(synth::qwen3tts::flatten_talker_prompt(h, automatic, text, codec, offset, acoustic,
                                                            acoustic_offset) == SYNTH_OK);
    SYNTH_TEST_CHECK(automatic.external_speaker_index == 3);
    return 0;
}

// Rule 3: the Stage 1 path must be untouched -- no x-vector, no index.
int test_the_preset_voice_path_is_unchanged() {
    const synth::qwen3tts::HParams h = customvoice_hparams();
    synth::qwen3tts::TalkerPrompt  prompt;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, request_with_token(), prompt) == SYNTH_OK);
    SYNTH_TEST_CHECK(prompt.external_speaker_index == -1);
    return 0;
}

constexpr uint32_t kFixtureHidden    = 4;
constexpr uint32_t kFixtureTextWidth = 6;
constexpr uint32_t kFixtureTextVocab = 10;
constexpr uint32_t kFixtureCodecSize = 8;
// Text length equals codec length so codec_offset == 0 covers the whole
// prompt -- the simplest layout that still lets every valid speaker_index
// (0..kFixturePositions) be exercised.
constexpr uint32_t kFixturePositions = 5;

struct Fixture {
    Context                        storage;
    ggml_context *                 context = nullptr;
    synth::qwen3tts::TalkerWeights weights;
    ggml_tensor *                  text                 = nullptr;
    ggml_tensor *                  codec                = nullptr;
    ggml_tensor *                  x_vector             = nullptr;
    ggml_tensor *                  wrong_width_x_vector = nullptr;
};

// No backend and no allocated buffers: every check below is refused (or not)
// from tensor shape alone, at graph-construction time, and none of these
// tensors is ever read for data.
Fixture build_talker_fixture() {
    Fixture fixture;
    fixture.storage    = make_context(ggml_tensor_overhead() * 32);
    fixture.context    = fixture.storage.get();
    ggml_context * ctx = fixture.context;

    synth::qwen3tts::TalkerWeights & w = fixture.weights;
    w.text_embedding                   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kFixtureTextWidth, kFixtureTextVocab);
    w.text_projection_1.weight         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kFixtureTextWidth, kFixtureTextWidth);
    w.text_projection_1.bias           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFixtureTextWidth);
    w.text_projection_2.weight         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kFixtureTextWidth, kFixtureHidden);
    w.text_projection_2.bias           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFixtureHidden);
    w.codec_embedding                  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kFixtureHidden, kFixtureCodecSize);

    fixture.text                 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kFixturePositions);
    fixture.codec                = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kFixturePositions);
    // A real row of this tensor's width, and one that could never be: enc_dim
    // == hidden_size is enforced at load (weights.cpp:553), so only a Profile
    // from somewhere else could hand the graph a mismatched width.
    fixture.x_vector             = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kFixtureHidden, 1);
    fixture.wrong_width_x_vector = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kFixtureHidden - 1, 1);
    return fixture;
}

// Rule 4: the graph refuses an index it cannot honour, and an embedding whose
// width is not the talker's hidden size, rather than substituting into
// whatever row it lands on. codec_offset is 0 and the codec run is
// kFixturePositions long, so index 4 is in range for the second call --
// isolating the width check from the range check, which the first call
// isolates the other way (a correctly-shaped x_vector, wildly out of range).
int test_the_graph_refuses_an_out_of_range_substitution() {
    Fixture fixture = build_talker_fixture();
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prefill_input(fixture.context, fixture.weights, fixture.text,
                                                                 fixture.codec, /*codec_offset=*/0, fixture.x_vector,
                                                                 /*speaker_index=*/999) == nullptr);
    // An x-vector whose width is not the talker's hidden size cannot be a row
    // of this tensor. enc_dim == hidden_size is enforced at load
    // (weights.cpp:553); this is what happens if a Profile from elsewhere
    // reaches here anyway.
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prefill_input(fixture.context, fixture.weights, fixture.text,
                                                                 fixture.codec, 0, fixture.wrong_width_x_vector,
                                                                 4) == nullptr);
    // The defaulted call -- no speaker embedding, no index -- is the Stage 1
    // path and must still build.
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prefill_input(fixture.context, fixture.weights, fixture.text,
                                                                 fixture.codec, 0) != nullptr);
    return 0;
}

class LcgStream {
  public:
    explicit LcgStream(uint64_t seed) : state_(seed) {}

    float next() {
        state_ = state_ * 6364136223846793005ull + 1442695040888963407ull;
        return float(state_ >> 40) / 8388608.0f - 1.0f;
    }

    std::vector<float> fill(size_t count, float scale, float offset) {
        std::vector<float> values(count);
        for (float & value : values) {
            value = next() * scale + offset;
        }
        return values;
    }

  private:
    uint64_t state_;
};

constexpr uint32_t kSubHidden    = 3;
constexpr uint32_t kSubTextWidth = 4;
constexpr uint32_t kSubTextVocab = 6;
constexpr uint32_t kSubCodecSize = 5;
constexpr uint32_t kSubPositions = 4;
constexpr uint64_t kSubSeed      = 20260812u;

constexpr int32_t kSubTextIds[kSubPositions]  = { 1, 3, 2, 4 };
constexpr int32_t kSubCodecIds[kSubPositions] = { 0, 2, 4, 1 };

struct SubstitutionFixture {
    ggml_backend_t                 backend = nullptr;
    Context                        persistent;
    ggml_backend_buffer_t          buffer = nullptr;
    synth::qwen3tts::TalkerWeights weights;
    ggml_tensor *                  text_tokens  = nullptr;
    ggml_tensor *                  codec_tokens = nullptr;
    ggml_tensor *                  speaker      = nullptr;

    ~SubstitutionFixture() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
};

bool build_substitution_fixture(SubstitutionFixture & fixture) {
    ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (device == nullptr) {
        return false;
    }
    fixture.backend = ggml_backend_dev_init(device, nullptr);
    if (fixture.backend == nullptr) {
        return false;
    }
    fixture.persistent  = make_context(ggml_tensor_overhead() * 32);
    ggml_context * pctx = fixture.persistent.get();

    std::vector<ggml_tensor *> ordered;
    std::vector<float>         scales;
    std::vector<float>         offsets;
    auto                       add = [&](ggml_tensor * tensor, float scale, float offset) {
        ordered.push_back(tensor);
        scales.push_back(scale);
        offsets.push_back(offset);
        return tensor;
    };

    synth::qwen3tts::TalkerWeights & w = fixture.weights;
    w.text_embedding           = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kSubTextWidth, kSubTextVocab), 0.5f, 0.0f);
    w.text_projection_1.weight = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kSubTextWidth, kSubTextWidth), 0.5f, 0.0f);
    w.text_projection_1.bias   = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kSubTextWidth), 0.25f, 0.0f);
    w.text_projection_2.weight = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kSubTextWidth, kSubHidden), 0.5f, 0.0f);
    w.text_projection_2.bias   = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kSubHidden), 0.25f, 0.0f);
    w.codec_embedding          = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kSubHidden, kSubCodecSize), 0.5f, 0.0f);

    fixture.text_tokens  = ggml_new_tensor_1d(pctx, GGML_TYPE_I32, kSubPositions);
    fixture.codec_tokens = ggml_new_tensor_1d(pctx, GGML_TYPE_I32, kSubPositions);
    fixture.speaker      = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kSubHidden, 1), 0.5f, 0.0f);

    fixture.buffer = ggml_backend_alloc_ctx_tensors(pctx, fixture.backend);
    if (fixture.buffer == nullptr) {
        return false;
    }

    LcgStream stream(kSubSeed);
    for (size_t index = 0; index < ordered.size(); ++index) {
        const std::vector<float> values =
            stream.fill(size_t(ggml_nelements(ordered[index])), scales[index], offsets[index]);
        ggml_backend_tensor_set(ordered[index], values.data(), 0, ggml_nbytes(ordered[index]));
    }
    ggml_backend_tensor_set(fixture.text_tokens, kSubTextIds, 0, ggml_nbytes(fixture.text_tokens));
    ggml_backend_tensor_set(fixture.codec_tokens, kSubCodecIds, 0, ggml_nbytes(fixture.codec_tokens));
    return true;
}

// Runs the substitution at one index and checks it against the ordinary
// (non-external) accumulation computed alongside it in the same graph: every
// position but `speaker_index` must match the plain text-plus-codec sum
// exactly, and `speaker_index` must match text-plus-speaker instead. This is
// what an off-by-one in the three-way ggml_acc split (talker.cpp) cannot
// survive: a shifted head/tail view still returns a non-null tensor of the
// right shape, so only a value comparison catches it.
bool run_substitution_case(SubstitutionFixture & fixture, int64_t speaker_index) {
    constexpr size_t kNodeBudget = 256;
    Context          graph_ctx =
        make_context(ggml_tensor_overhead() * (kNodeBudget + 32) + ggml_graph_overhead_custom(kNodeBudget, false));
    ggml_context * gctx  = graph_ctx.get();
    ggml_cgraph *  graph = ggml_new_graph_custom(gctx, kNodeBudget, false);

    ggml_tensor * text  = synth::qwen3tts::build_text_projection(gctx, fixture.weights, fixture.text_tokens);
    ggml_tensor * codec = ggml_get_rows(gctx, fixture.weights.codec_embedding, fixture.codec_tokens);
    ggml_tensor * actual =
        synth::qwen3tts::build_talker_prefill_input(gctx, fixture.weights, fixture.text_tokens, fixture.codec_tokens,
                                                    /*codec_offset=*/0, fixture.speaker, speaker_index);
    if (text == nullptr || codec == nullptr || actual == nullptr) {
        return false;
    }

    for (ggml_tensor * output : { text, codec, actual }) {
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
    }

    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(fixture.backend));
    bool           ok        = allocator != nullptr && ggml_gallocr_alloc_graph(allocator, graph);
    if (ok) {
        ok = ggml_backend_graph_compute(fixture.backend, graph) == GGML_STATUS_SUCCESS;
    }
    std::vector<float> text_values;
    std::vector<float> codec_values;
    std::vector<float> actual_values;
    std::vector<float> speaker_values;
    if (ok) {
        text_values.resize(size_t(ggml_nelements(text)));
        ggml_backend_tensor_get(text, text_values.data(), 0, ggml_nbytes(text));
        codec_values.resize(size_t(ggml_nelements(codec)));
        ggml_backend_tensor_get(codec, codec_values.data(), 0, ggml_nbytes(codec));
        actual_values.resize(size_t(ggml_nelements(actual)));
        ggml_backend_tensor_get(actual, actual_values.data(), 0, ggml_nbytes(actual));
        speaker_values.resize(kSubHidden);
        ggml_backend_tensor_get(fixture.speaker, speaker_values.data(), 0, ggml_nbytes(fixture.speaker));
    }
    if (allocator != nullptr) {
        ggml_gallocr_free(allocator);
    }
    if (!ok) {
        return false;
    }

    for (int64_t position = 0; position < int64_t(kSubPositions); ++position) {
        for (uint32_t row = 0; row < kSubHidden; ++row) {
            const size_t index = size_t(position) * kSubHidden + row;
            const float  expected =
                text_values[index] + (position == speaker_index ? speaker_values[row] : codec_values[index]);
            if (std::fabs(actual_values[index] - expected) > 1e-5f) {
                return false;
            }
        }
    }
    return true;
}

// Not one of the brief's four, but the same testing policy applies: this is a
// rule (a valid substitution lands the embedding at its index and nowhere
// else) with no test above that would fail without it. speaker_index 0 skips
// the head accumulation, kSubPositions-1 skips the tail, and the middle value
// exercises both -- talker.cpp's three branches, each hit once.
int test_the_graph_places_the_embedding_at_its_slot_and_nowhere_else() {
    SubstitutionFixture fixture;
    SYNTH_TEST_CHECK(build_substitution_fixture(fixture));
    const int64_t cases[] = { 0, 2, int64_t(kSubPositions) - 1 };
    for (int64_t speaker_index : cases) {
        SYNTH_TEST_CHECK(run_substitution_case(fixture, speaker_index));
    }
    return 0;
}

// Task 10 review, IMPORTANT 2: the case above only ever runs at
// codec_offset == 0, which is degenerate for this axis -- `size_t(0) *
// out->nb[1]` is indistinguishable from omitting the offset term entirely,
// so a mutant that dropped codec_offset from all three of talker.cpp's
// external ggml_acc calls passed the case above unnoticed. Every real
// request has codec_offset == role_tokens.size() (3, under the reference
// template -- bpe.h's kAssistantRolePrefixTokens), so this fixture mirrors
// that shape: three text-only positions (no codec row exists for them at
// all) ahead of a four-long codec run.
constexpr uint32_t kOffsetTextPositions = 7;
constexpr uint32_t kOffsetCodecCount    = 4;
constexpr int64_t  kOffsetCodecOffset   = int64_t(kOffsetTextPositions - kOffsetCodecCount);  // 3, matching the
                                                                                              // reference template.

constexpr int32_t kOffsetTextIds[kOffsetTextPositions] = { 5, 1, 3, 2, 4, 0, 3 };
constexpr int32_t kOffsetCodecIds[kOffsetCodecCount]   = { 0, 2, 4, 1 };

struct OffsetSubstitutionFixture {
    ggml_backend_t                 backend = nullptr;
    Context                        persistent;
    ggml_backend_buffer_t          buffer = nullptr;
    synth::qwen3tts::TalkerWeights weights;
    ggml_tensor *                  text_tokens  = nullptr;
    ggml_tensor *                  codec_tokens = nullptr;
    ggml_tensor *                  speaker      = nullptr;

    ~OffsetSubstitutionFixture() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
};

bool build_offset_substitution_fixture(OffsetSubstitutionFixture & fixture) {
    ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (device == nullptr) {
        return false;
    }
    fixture.backend = ggml_backend_dev_init(device, nullptr);
    if (fixture.backend == nullptr) {
        return false;
    }
    fixture.persistent  = make_context(ggml_tensor_overhead() * 32);
    ggml_context * pctx = fixture.persistent.get();

    std::vector<ggml_tensor *> ordered;
    std::vector<float>         scales;
    std::vector<float>         offsets;
    auto                       add = [&](ggml_tensor * tensor, float scale, float offset) {
        ordered.push_back(tensor);
        scales.push_back(scale);
        offsets.push_back(offset);
        return tensor;
    };

    synth::qwen3tts::TalkerWeights & w = fixture.weights;
    w.text_embedding           = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kSubTextWidth, kSubTextVocab), 0.5f, 0.0f);
    w.text_projection_1.weight = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kSubTextWidth, kSubTextWidth), 0.5f, 0.0f);
    w.text_projection_1.bias   = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kSubTextWidth), 0.25f, 0.0f);
    w.text_projection_2.weight = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kSubTextWidth, kSubHidden), 0.5f, 0.0f);
    w.text_projection_2.bias   = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kSubHidden), 0.25f, 0.0f);
    w.codec_embedding          = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kSubHidden, kSubCodecSize), 0.5f, 0.0f);

    fixture.text_tokens  = ggml_new_tensor_1d(pctx, GGML_TYPE_I32, kOffsetTextPositions);
    fixture.codec_tokens = ggml_new_tensor_1d(pctx, GGML_TYPE_I32, kOffsetCodecCount);
    fixture.speaker      = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kSubHidden, 1), 0.5f, 0.0f);

    fixture.buffer = ggml_backend_alloc_ctx_tensors(pctx, fixture.backend);
    if (fixture.buffer == nullptr) {
        return false;
    }

    // A different seed from build_substitution_fixture's so the two fixtures'
    // weights don't coincide.
    LcgStream stream(kSubSeed + 1);
    for (size_t index = 0; index < ordered.size(); ++index) {
        const std::vector<float> values =
            stream.fill(size_t(ggml_nelements(ordered[index])), scales[index], offsets[index]);
        ggml_backend_tensor_set(ordered[index], values.data(), 0, ggml_nbytes(ordered[index]));
    }
    ggml_backend_tensor_set(fixture.text_tokens, kOffsetTextIds, 0, ggml_nbytes(fixture.text_tokens));
    ggml_backend_tensor_set(fixture.codec_tokens, kOffsetCodecIds, 0, ggml_nbytes(fixture.codec_tokens));
    return true;
}

// The same check as run_substitution_case, generalized for a nonzero
// codec_offset: a position before the offset has no codec row at all, so it
// must come out as bare text -- neither the ordinary accumulation nor the
// external one may add anything there. A position from the offset on must
// match the plain accumulation except at the substituted (relative) row. A
// mutant that dropped codec_offset from talker.cpp's three external
// ggml_acc calls would add codec/speaker starting at position 0 instead of
// kOffsetCodecOffset, corrupting the pre-offset positions -- exactly what
// this catches and run_substitution_case's codec_offset == 0 case cannot.
bool run_offset_substitution_case(OffsetSubstitutionFixture & fixture, int64_t speaker_index) {
    constexpr size_t kNodeBudget = 256;
    Context          graph_ctx =
        make_context(ggml_tensor_overhead() * (kNodeBudget + 32) + ggml_graph_overhead_custom(kNodeBudget, false));
    ggml_context * gctx  = graph_ctx.get();
    ggml_cgraph *  graph = ggml_new_graph_custom(gctx, kNodeBudget, false);

    ggml_tensor * text  = synth::qwen3tts::build_text_projection(gctx, fixture.weights, fixture.text_tokens);
    ggml_tensor * codec = ggml_get_rows(gctx, fixture.weights.codec_embedding, fixture.codec_tokens);
    ggml_tensor * actual =
        synth::qwen3tts::build_talker_prefill_input(gctx, fixture.weights, fixture.text_tokens, fixture.codec_tokens,
                                                    kOffsetCodecOffset, fixture.speaker, speaker_index);
    if (text == nullptr || codec == nullptr || actual == nullptr) {
        return false;
    }

    for (ggml_tensor * output : { text, codec, actual }) {
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
    }

    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(fixture.backend));
    bool           ok        = allocator != nullptr && ggml_gallocr_alloc_graph(allocator, graph);
    if (ok) {
        ok = ggml_backend_graph_compute(fixture.backend, graph) == GGML_STATUS_SUCCESS;
    }
    std::vector<float> text_values;
    std::vector<float> codec_values;
    std::vector<float> actual_values;
    std::vector<float> speaker_values;
    if (ok) {
        text_values.resize(size_t(ggml_nelements(text)));
        ggml_backend_tensor_get(text, text_values.data(), 0, ggml_nbytes(text));
        codec_values.resize(size_t(ggml_nelements(codec)));
        ggml_backend_tensor_get(codec, codec_values.data(), 0, ggml_nbytes(codec));
        actual_values.resize(size_t(ggml_nelements(actual)));
        ggml_backend_tensor_get(actual, actual_values.data(), 0, ggml_nbytes(actual));
        speaker_values.resize(kSubHidden);
        ggml_backend_tensor_get(fixture.speaker, speaker_values.data(), 0, ggml_nbytes(fixture.speaker));
    }
    if (allocator != nullptr) {
        ggml_gallocr_free(allocator);
    }
    if (!ok) {
        return false;
    }

    for (int64_t position = 0; position < int64_t(kOffsetTextPositions); ++position) {
        for (uint32_t row = 0; row < kSubHidden; ++row) {
            const size_t index = size_t(position) * kSubHidden + row;
            float        expected;
            if (position < kOffsetCodecOffset) {
                expected = text_values[index];
            } else {
                const int64_t relative     = position - kOffsetCodecOffset;
                const size_t  relative_idx = size_t(relative) * kSubHidden + row;
                expected =
                    text_values[index] + (relative == speaker_index ? speaker_values[row] : codec_values[relative_idx]);
            }
            if (std::fabs(actual_values[index] - expected) > 1e-5f) {
                return false;
            }
        }
    }
    return true;
}

int test_the_substitution_respects_a_nonzero_codec_offset() {
    OffsetSubstitutionFixture fixture;
    SYNTH_TEST_CHECK(build_offset_substitution_fixture(fixture));
    const int64_t cases[] = { 0, 2, int64_t(kOffsetCodecCount) - 1 };
    for (int64_t speaker_index : cases) {
        SYNTH_TEST_CHECK(run_offset_substitution_case(fixture, speaker_index));
    }
    return 0;
}

// The third speaker case. CustomVoice fills the slot from the codec vocabulary
// and Base substitutes an embedding into it, but both KEEP the position;
// VoiceDesign has no slot at all, so its codec run is one shorter and the text
// stream shifts with it. modeling_qwen3_tts.py:2166-2172 -- with speaker_embed
// null the prefill is cat(prefill_0, prefill_1) and nothing else.
int test_prompt_without_a_speaker_slot() {
    synth::qwen3tts::HParams hparams = base_hparams();

    synth::qwen3tts::TalkerPromptRequest with_speaker;
    with_speaker.role_tokens    = { 1, 2, 3 };
    with_speaker.text_tokens    = { 10, 11, 12 };
    with_speaker.has_language   = true;
    with_speaker.language_token = 7;
    with_speaker.has_speaker    = true;
    with_speaker.speaker_token  = 42;

    synth::qwen3tts::TalkerPromptRequest without_speaker = with_speaker;
    without_speaker.has_speaker                          = false;
    without_speaker.speaker_token                        = 0;

    synth::qwen3tts::TalkerPrompt kept;
    synth::qwen3tts::TalkerPrompt dropped;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(hparams, with_speaker, kept) == SYNTH_OK);
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(hparams, without_speaker, dropped) == SYNTH_OK);

    // Exactly one position fewer, and it is a CODEC position that went.
    SYNTH_TEST_CHECK(dropped.positions.size() + 1 == kept.positions.size());
    size_t kept_codec    = 0;
    size_t dropped_codec = 0;
    for (const synth::qwen3tts::TalkerInputPosition & p : kept.positions) {
        kept_codec += p.has_codec ? 1 : 0;
    }
    for (const synth::qwen3tts::TalkerInputPosition & p : dropped.positions) {
        dropped_codec += p.has_codec ? 1 : 0;
    }
    SYNTH_TEST_CHECK(dropped_codec + 1 == kept_codec);

    // The speaker token appears in one and not the other -- an implementation
    // that merely zeroed it would pass the count checks above.
    bool found = false;
    for (const synth::qwen3tts::TalkerInputPosition & p : dropped.positions) {
        found = found || (p.has_codec && p.codec_token == 42);
    }
    SYNTH_TEST_CHECK(!found);

    // No external substitution is pending: -1 means "the graph reads every
    // codec row", which is what talker.cpp's null-speaker branch expects.
    SYNTH_TEST_CHECK(dropped.external_speaker_index == -1);

    // The trailing schedule is unaffected: dropping the slot shortens the
    // prefill, not the text that follows it.
    SYNTH_TEST_CHECK(dropped.trailing.size() == kept.trailing.size());
    return 0;
}

}  // namespace

int main() {
    SYNTH_TEST_CHECK(test_an_external_speaker_slot_does_not_change_the_prompt_length() == 0);
    SYNTH_TEST_CHECK(test_the_substitution_index_follows_the_language_token() == 0);
    SYNTH_TEST_CHECK(test_the_preset_voice_path_is_unchanged() == 0);
    SYNTH_TEST_CHECK(test_the_graph_refuses_an_out_of_range_substitution() == 0);
    SYNTH_TEST_CHECK(test_the_graph_places_the_embedding_at_its_slot_and_nowhere_else() == 0);
    SYNTH_TEST_CHECK(test_the_substitution_respects_a_nonzero_codec_offset() == 0);
    SYNTH_TEST_CHECK(test_prompt_without_a_speaker_slot() == 0);
    return 0;
}
