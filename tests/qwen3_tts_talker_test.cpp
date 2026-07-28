// The talker, compared against the reference implementation's own classes.
//
// Two things are pinned here that the shared decoder block's test cannot reach.
//
// The **text tower**: the wide text embedding, then a two-layer projection with
// biases and a SiLU between them, which is what brings a text token down to the
// talker's width. Both towers meet at every input position and are summed, so a
// mistake in either shows up as the wrong voice rather than as an error.
//
// The **step**: layers, final norm, then codec_head. The talker's attention goes
// through the reference's *multimodal* rope helper rather than the plain one the
// code predictor uses. All three of its position rows are always identical for
// this model, so it collapses to plain rope -- running the real class with the
// real interleaved section shape is how that gets checked rather than argued.
//
// Reference values come from scripts/dump_reference_qwen3_tts_talker.py.

#include "arch/qwen3-tts/catalog.h"
#include "arch/qwen3-tts/talker-host.h"
#include "arch/qwen3-tts/talker.h"
#include "arch/qwen3-tts/weights.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "test-assert.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

namespace {

constexpr uint32_t kHidden       = 8;
constexpr uint32_t kTextHidden   = 12;
constexpr uint32_t kHeads        = 4;
constexpr uint32_t kKvHeads      = 2;
constexpr uint32_t kHeadDim      = 8;
constexpr uint32_t kIntermediate = 16;
constexpr uint32_t kLayers       = 2;
constexpr uint32_t kTextVocab    = 20;
constexpr uint32_t kCodecVocab   = 14;
constexpr uint32_t kPositions    = 5;
constexpr uint32_t kCodecCount   = 3;
constexpr float    kRmsNormEps   = 1e-6f;
constexpr float    kRopeTheta    = 1000000.0f;
constexpr uint64_t kSeed         = 20260728u;

constexpr int32_t kTextIds[kPositions]  = { 3, 11, 4, 0, 7 };
constexpr int32_t kCodecIds[kCodecCount] = { 2, 9, 5 };

// 5 positions x 8 hidden: the text tower alone.
constexpr float kExpectedTextProjection[] = {
    -0.321217328f, -0.0568927117f, -0.241882145f, -0.329237521f,
    -0.203585491f, -0.550951362f, -0.037478745f, -0.00465635024f,
    -0.241684139f, -0.0340938382f, -0.128059134f, -0.312719822f,
    -0.184373066f, -0.117215395f, -0.0755563304f, -0.0471986346f,
    -0.275102645f, 0.0206815228f, -0.138152763f, -0.236012101f,
    -0.194840252f, -0.344171733f, -0.173974916f, -0.149927258f,
    0.14595297f, 0.104290701f, -0.251262516f, -0.447087288f,
    -0.438497752f, -0.260504335f, 0.214015529f, 0.015037423f,
    -0.265943319f, 0.173397273f, -0.18261075f, -0.233840168f,
    -0.33851552f, -0.684959054f, -0.138663232f, -0.281280667f,
};

// The same, with the codec stream summed into its trailing positions.
constexpr float kExpectedPrefillInput[] = {
    -0.321217328f, -0.0568927117f, -0.241882145f, -0.329237521f,
    -0.203585491f, -0.550951362f, -0.037478745f, -0.00465635024f,
    -0.241684139f, -0.0340938382f, -0.128059134f, -0.312719822f,
    -0.184373066f, -0.117215395f, -0.0755563304f, -0.0471986346f,
    -0.0214849412f, -0.141487092f, 0.229130939f, 0.0199397802f,
    -0.0414662957f, -0.69021821f, -0.662320435f, -0.529608786f,
    0.151630193f, 0.24861756f, 0.108713716f, -0.903019547f,
    -0.718533397f, -0.206514448f, 0.287897229f, 0.140239596f,
    -0.315214008f, 0.64344871f, -0.44977361f, -0.524056435f,
    -0.595454037f, -1.04675293f, -0.628060758f, 0.0776343644f,
};

// The last position's hidden state after the final norm.
constexpr float kExpectedLastHidden[] = {
    -1.32450056f, -1.0357573f, 0.510571361f, 1.10772491f,
    -0.213531166f, -0.578867435f, -1.65141177f, -1.01671517f,
};

// codec_head over that: 14 logits.
constexpr float kExpectedLogits[] = {
    0.0105163399f, -0.335359484f, -1.7436831f, -0.13869749f,
    0.856686056f, 0.316256732f, 1.40711629f, 0.512437046f,
    -1.40626621f, 0.820484281f, -1.13014758f, 0.632562459f,
    -0.51303792f, -0.683827579f,
};

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

struct ContextDeleter {
    void operator()(ggml_context * context) const {
        if (context != nullptr) {
            ggml_free(context);
        }
    }
};

using Context = std::unique_ptr<ggml_context, ContextDeleter>;

constexpr size_t kNodeBudget = 1024;

Context make_context(size_t bytes) {
    ggml_init_params parameters{};
    parameters.mem_size = bytes;
    parameters.no_alloc = true;
    return Context(ggml_init(parameters));
}

Context make_graph_context() {
    return make_context(ggml_tensor_overhead() * (kNodeBudget + 64) + ggml_graph_overhead_custom(kNodeBudget, false));
}

synth::qwen3tts::AttentionShape make_shape() {
    synth::qwen3tts::AttentionShape shape;
    shape.hidden_size          = kHidden;
    shape.attention_head_count = kHeads;
    shape.key_value_head_count = kKvHeads;
    shape.head_dim             = kHeadDim;
    shape.rms_norm_eps         = kRmsNormEps;
    shape.rope_theta           = kRopeTheta;
    return shape;
}

struct Fixture {
    ggml_backend_t                 backend = nullptr;
    Context                        persistent;
    ggml_backend_buffer_t          buffer = nullptr;
    synth::qwen3tts::TalkerWeights weights;
    synth::qwen3tts::TalkerCache   cache;
    ggml_tensor *                  text_tokens  = nullptr;
    ggml_tensor *                  codec_tokens = nullptr;
    ggml_tensor *                  positions    = nullptr;
    ggml_tensor *                  mask         = nullptr;

    ~Fixture() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
};

bool build_fixture(ggml_backend_dev_t device, Fixture & fixture) {
    fixture.backend = ggml_backend_dev_init(device, nullptr);
    if (fixture.backend == nullptr) {
        return false;
    }
    fixture.persistent  = make_context(ggml_tensor_overhead() * 128);
    ggml_context * pctx = fixture.persistent.get();

    std::vector<ggml_tensor *> ordered;
    std::vector<float>         scales;
    std::vector<float>         offsets;
    auto add = [&](ggml_tensor * tensor, float scale, float offset) {
        ordered.push_back(tensor);
        scales.push_back(scale);
        offsets.push_back(offset);
        return tensor;
    };

    // The fill order is the reference script's.
    synth::qwen3tts::TalkerWeights & w = fixture.weights;
    w.text_embedding = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kTextHidden, kTextVocab), 0.5f, 0.0f);
    w.text_projection_1.weight = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kTextHidden, kTextHidden), 0.5f, 0.0f);
    w.text_projection_1.bias   = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kTextHidden), 0.25f, 0.0f);
    w.text_projection_2.weight = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kTextHidden, kHidden), 0.5f, 0.0f);
    w.text_projection_2.bias   = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kHidden), 0.25f, 0.0f);
    w.codec_embedding = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kCodecVocab), 0.5f, 0.0f);
    w.codec_head      = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kCodecVocab), 0.5f, 0.0f);

    w.layers.resize(kLayers);
    for (uint32_t index = 0; index < kLayers; ++index) {
        synth::qwen3tts::DecoderLayerWeights & layer = w.layers[index];
        layer.input_layernorm = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kHidden), 0.25f, 1.0f);
        layer.q_proj    = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kHeads * kHeadDim), 0.5f, 0.0f);
        layer.k_proj    = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kKvHeads * kHeadDim), 0.5f, 0.0f);
        layer.v_proj    = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kKvHeads * kHeadDim), 0.5f, 0.0f);
        layer.o_proj    = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHeads * kHeadDim, kHidden), 0.5f, 0.0f);
        layer.q_norm    = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kHeadDim), 0.25f, 1.0f);
        layer.k_norm    = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kHeadDim), 0.25f, 1.0f);
        layer.post_attention_layernorm = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kHidden), 0.25f, 1.0f);
        layer.gate_proj = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kIntermediate), 0.5f, 0.0f);
        layer.up_proj   = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kIntermediate), 0.5f, 0.0f);
        layer.down_proj = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kIntermediate, kHidden), 0.5f, 0.0f);
    }
    w.norm = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kHidden), 0.25f, 1.0f);

    fixture.cache.layers.resize(kLayers);
    for (uint32_t index = 0; index < kLayers; ++index) {
        fixture.cache.layers[index].k = ggml_new_tensor_3d(pctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kPositions);
        fixture.cache.layers[index].v = ggml_new_tensor_3d(pctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kPositions);
    }

    fixture.text_tokens  = ggml_new_tensor_1d(pctx, GGML_TYPE_I32, kPositions);
    fixture.codec_tokens = ggml_new_tensor_1d(pctx, GGML_TYPE_I32, kCodecCount);
    fixture.positions    = ggml_new_tensor_1d(pctx, GGML_TYPE_I32, kPositions);
    fixture.mask         = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kPositions, kPositions);

    fixture.buffer = ggml_backend_alloc_ctx_tensors(pctx, fixture.backend);
    if (fixture.buffer == nullptr) {
        return false;
    }

    LcgStream stream(kSeed);
    for (size_t index = 0; index < ordered.size(); ++index) {
        const std::vector<float> values =
            stream.fill(size_t(ggml_nelements(ordered[index])), scales[index], offsets[index]);
        ggml_backend_tensor_set(ordered[index], values.data(), 0, ggml_nbytes(ordered[index]));
    }
    ggml_backend_tensor_set(fixture.text_tokens, kTextIds, 0, ggml_nbytes(fixture.text_tokens));
    ggml_backend_tensor_set(fixture.codec_tokens, kCodecIds, 0, ggml_nbytes(fixture.codec_tokens));

    int32_t sequential[kPositions];
    for (uint32_t index = 0; index < kPositions; ++index) {
        sequential[index] = int32_t(index);
    }
    ggml_backend_tensor_set(fixture.positions, sequential, 0, ggml_nbytes(fixture.positions));

    const float        infinity = std::numeric_limits<float>::infinity();
    std::vector<float> mask(size_t(kPositions) * kPositions, 0.0f);
    for (uint32_t query = 0; query < kPositions; ++query) {
        for (uint32_t key = 0; key < kPositions; ++key) {
            mask[size_t(query) * kPositions + key] = key <= query ? 0.0f : -infinity;
        }
    }
    ggml_backend_tensor_set(fixture.mask, mask.data(), 0, ggml_nbytes(fixture.mask));
    return true;
}

bool compute(Fixture & fixture, ggml_cgraph * graph, const std::vector<ggml_tensor *> & outputs,
             std::vector<std::vector<float>> & values) {
    for (ggml_tensor * output : outputs) {
        // Intermediates are read back here, and the graph allocator would
        // otherwise be free to reuse their buffers for later nodes.
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
    }
    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(fixture.backend));
    bool           ok        = allocator != nullptr && ggml_gallocr_alloc_graph(allocator, graph);
    if (ok) {
        ok = ggml_backend_graph_compute(fixture.backend, graph) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        values.clear();
        for (ggml_tensor * output : outputs) {
            std::vector<float> one(size_t(ggml_nelements(output)));
            ggml_backend_tensor_get(output, one.data(), 0, ggml_nbytes(output));
            values.push_back(std::move(one));
        }
    }
    if (allocator != nullptr) {
        ggml_gallocr_free(allocator);
    }
    return ok;
}

float deviation(const std::vector<float> & got, const float * expected, size_t count) {
    if (got.size() != count) {
        return std::numeric_limits<float>::infinity();
    }
    float worst = 0.0f;
    for (size_t index = 0; index < count; ++index) {
        worst = std::fmax(worst, std::fabs(got[index] - expected[index]));
    }
    return worst;
}

bool run_case(ggml_backend_dev_t device, float & max_diff) {
    Fixture fixture;
    if (!build_fixture(device, fixture)) {
        return false;
    }

    Context       graph_ctx = make_graph_context();
    ggml_cgraph * graph     = ggml_new_graph_custom(graph_ctx.get(), kNodeBudget, false);

    ggml_tensor * text = synth::qwen3tts::build_text_projection(graph_ctx.get(), fixture.weights,
                                                                fixture.text_tokens);
    ggml_tensor * input = synth::qwen3tts::build_talker_prefill_input(
        graph_ctx.get(), fixture.weights, fixture.text_tokens, fixture.codec_tokens, kPositions - kCodecCount);
    if (text == nullptr || input == nullptr || input->ne[0] != int64_t(kHidden) ||
        input->ne[1] != int64_t(kPositions)) {
        return false;
    }

    ggml_tensor * hidden = nullptr;
    ggml_tensor * logits =
        synth::qwen3tts::build_talker_step(graph_ctx.get(), graph, input, fixture.positions, fixture.mask,
                                          fixture.weights, make_shape(), fixture.cache, &hidden);
    if (logits == nullptr || hidden == nullptr || logits->ne[0] != int64_t(kCodecVocab) || logits->ne[1] != 1) {
        return false;
    }

    std::vector<std::vector<float>> values;
    if (!compute(fixture, graph, { text, input, hidden, logits }, values)) {
        return false;
    }
    const float d0 = deviation(values[0], kExpectedTextProjection, std::size(kExpectedTextProjection));
    const float d1 = deviation(values[1], kExpectedPrefillInput, std::size(kExpectedPrefillInput));
    const float d2 = deviation(values[2], kExpectedLastHidden, std::size(kExpectedLastHidden));
    const float d3 = deviation(values[3], kExpectedLogits, std::size(kExpectedLogits));
    std::printf("  text %.3g  prefill %.3g  hidden %.3g  logits %.3g\n", double(d0), double(d1), double(d2),
                double(d3));
    max_diff = std::fmax(std::fmax(d0, d1), std::fmax(d2, d3));
    return true;
}

// The prompt layout, which is where an off-by-one changes the voice or the
// language without changing anything observable.
synth::qwen3tts::HParams prompt_hparams() {
    synth::qwen3tts::HParams h;
    h.talker.code_group_count   = 4;
    h.tokens.tts_bos            = 100;
    h.tokens.tts_eos            = 101;
    h.tokens.tts_pad            = 102;
    h.tokens.codec_bos          = 200;
    h.tokens.codec_eos          = 201;
    h.tokens.codec_pad          = 202;
    h.tokens.codec_think        = 203;
    h.tokens.codec_nothink      = 204;
    h.tokens.codec_think_bos    = 205;
    h.tokens.codec_think_eos    = 206;
    return h;
}

using Position = synth::qwen3tts::TalkerInputPosition;

int expect_position(const Position & got, Position::Text text, uint32_t text_token, bool has_codec,
                    uint32_t codec_token) {
    SYNTH_TEST_CHECK(got.text == text);
    if (text == Position::Text::Token) {
        SYNTH_TEST_CHECK(got.text_token == text_token);
    }
    SYNTH_TEST_CHECK(got.has_codec == has_codec);
    if (has_codec) {
        SYNTH_TEST_CHECK(got.codec_token == codec_token);
    }
    return 0;
}

int check_prompt_layout() {
    const synth::qwen3tts::HParams h = prompt_hparams();

    synth::qwen3tts::TalkerPromptRequest request;
    request.role_tokens    = { 10, 11, 12 };
    request.text_tokens    = { 20, 21, 22 };
    request.has_language   = true;
    request.language_token = 300;
    request.has_speaker    = true;
    request.speaker_token  = 400;

    synth::qwen3tts::TalkerPrompt prompt;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, request, prompt) == SYNTH_OK);

    // Three role positions, then the codec stream of seven -- think, think_bos,
    // language, think_eos, speaker, codec_pad, codec_bos -- laid against a text
    // stream of pads that ends in tts_bos, with codec_bos landing on the first
    // text token instead of on a pad. Ten positions in all.
    SYNTH_TEST_CHECK(prompt.positions.size() == 10);
    for (size_t index = 0; index < 3; ++index) {
        SYNTH_TEST_CHECK(expect_position(prompt.positions[index], Position::Text::Token,
                                         request.role_tokens[index], false, 0) == 0);
    }
    SYNTH_TEST_CHECK(expect_position(prompt.positions[3], Position::Text::TtsPad, 0, true,
                                     h.tokens.codec_think) == 0);
    SYNTH_TEST_CHECK(expect_position(prompt.positions[4], Position::Text::TtsPad, 0, true,
                                     h.tokens.codec_think_bos) == 0);
    SYNTH_TEST_CHECK(expect_position(prompt.positions[5], Position::Text::TtsPad, 0, true, 300) == 0);
    SYNTH_TEST_CHECK(expect_position(prompt.positions[6], Position::Text::TtsPad, 0, true,
                                     h.tokens.codec_think_eos) == 0);
    SYNTH_TEST_CHECK(expect_position(prompt.positions[7], Position::Text::TtsPad, 0, true, 400) == 0);
    // The one tts_bos in the whole prompt sits on the codec pad, one before the end.
    SYNTH_TEST_CHECK(expect_position(prompt.positions[8], Position::Text::TtsBos, 0, true,
                                     h.tokens.codec_pad) == 0);
    SYNTH_TEST_CHECK(expect_position(prompt.positions[9], Position::Text::Token, 20, true,
                                     h.tokens.codec_bos) == 0);

    // The first text token went into the prefill, so the schedule starts at the
    // second and ends with one tts_eos.
    SYNTH_TEST_CHECK(prompt.trailing.size() == 3);
    SYNTH_TEST_CHECK(expect_position(prompt.trailing[0], Position::Text::Token, 21, false, 0) == 0);
    SYNTH_TEST_CHECK(expect_position(prompt.trailing[1], Position::Text::Token, 22, false, 0) == 0);
    SYNTH_TEST_CHECK(expect_position(prompt.trailing[2], Position::Text::TtsEos, 0, false, 0) == 0);

    // Steps past the schedule pad forever rather than running off the end.
    SYNTH_TEST_CHECK(synth::qwen3tts::talker_step_text_token(h, prompt, 0) == 21);
    SYNTH_TEST_CHECK(synth::qwen3tts::talker_step_text_token(h, prompt, 2) == h.tokens.tts_eos);
    SYNTH_TEST_CHECK(synth::qwen3tts::talker_step_text_token(h, prompt, 3) == h.tokens.tts_pad);
    SYNTH_TEST_CHECK(synth::qwen3tts::talker_step_text_token(h, prompt, 4000) == h.tokens.tts_pad);

    // Asking for auto is a shorter prompt, not the same prompt with a default
    // language: nothink replaces think and no language token is emitted at all.
    synth::qwen3tts::TalkerPromptRequest automatic = request;
    automatic.has_language                         = false;
    synth::qwen3tts::TalkerPrompt auto_prompt;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, automatic, auto_prompt) == SYNTH_OK);
    SYNTH_TEST_CHECK(auto_prompt.positions.size() == 9);
    SYNTH_TEST_CHECK(expect_position(auto_prompt.positions[3], Position::Text::TtsPad, 0, true,
                                     h.tokens.codec_nothink) == 0);
    SYNTH_TEST_CHECK(expect_position(auto_prompt.positions[5], Position::Text::TtsPad, 0, true,
                                     h.tokens.codec_think_eos) == 0);

    // No preset Voice: one position fewer again.
    synth::qwen3tts::TalkerPromptRequest anonymous = request;
    anonymous.has_speaker                          = false;
    synth::qwen3tts::TalkerPrompt anonymous_prompt;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, anonymous, anonymous_prompt) == SYNTH_OK);
    SYNTH_TEST_CHECK(anonymous_prompt.positions.size() == 9);
    SYNTH_TEST_CHECK(expect_position(anonymous_prompt.positions[7], Position::Text::TtsBos, 0, true,
                                     h.tokens.codec_pad) == 0);

    // A single text token still leaves a schedule, holding only the eos.
    synth::qwen3tts::TalkerPromptRequest terse = request;
    terse.text_tokens                          = { 20 };
    synth::qwen3tts::TalkerPrompt terse_prompt;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, terse, terse_prompt) == SYNTH_OK);
    SYNTH_TEST_CHECK(terse_prompt.trailing.size() == 1);
    SYNTH_TEST_CHECK(terse_prompt.trailing[0].text == Position::Text::TtsEos);

    // Nothing to say is a caller defect, not an empty utterance.
    synth::qwen3tts::TalkerPromptRequest silent = request;
    silent.text_tokens.clear();
    synth::qwen3tts::TalkerPrompt silent_prompt;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, silent, silent_prompt) == SYNTH_ERR_INVALID_ARG);

    SYNTH_TEST_CHECK(synth::qwen3tts::talker_frame_ends_utterance(h, h.tokens.codec_eos));
    SYNTH_TEST_CHECK(!synth::qwen3tts::talker_frame_ends_utterance(h, h.tokens.codec_bos));
    return 0;
}

int check_flatten() {
    const synth::qwen3tts::HParams h = prompt_hparams();

    synth::qwen3tts::TalkerPromptRequest request;
    request.role_tokens    = { 10, 11, 12 };
    request.text_tokens    = { 20, 21 };
    request.has_language   = true;
    request.language_token = 300;
    request.has_speaker    = true;
    request.speaker_token  = 400;

    synth::qwen3tts::TalkerPrompt prompt;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, request, prompt) == SYNTH_OK);

    std::vector<int32_t> text;
    std::vector<int32_t> codec;
    int64_t              offset = -1;
    SYNTH_TEST_CHECK(synth::qwen3tts::flatten_talker_prompt(h, prompt, text, codec, offset) == SYNTH_OK);

    // Every position carries a text token, the specials resolved to their
    // text-vocabulary ids, and the codec stream is the trailing seven.
    const std::vector<int32_t> expected_text = { 10, 11, 12, 102, 102, 102, 102, 102, 100, 20 };
    const std::vector<int32_t> expected_codec = { 203, 205, 300, 206, 400, 202, 200 };
    SYNTH_TEST_CHECK(text == expected_text);
    SYNTH_TEST_CHECK(codec == expected_codec);
    SYNTH_TEST_CHECK(offset == 3);

    // A codec token before a position without one would mean the stream is not a
    // tail, and the graph adds it as one.
    synth::qwen3tts::TalkerPrompt gapped = prompt;
    gapped.positions[5].has_codec        = false;
    SYNTH_TEST_CHECK(synth::qwen3tts::flatten_talker_prompt(h, gapped, text, codec, offset) ==
                     SYNTH_ERR_INVALID_ARG);

    // A position with no text side at all is a layout defect: the graph reads one
    // text token per position.
    synth::qwen3tts::TalkerPrompt textless = prompt;
    textless.positions[4].text             = Position::Text::None;
    SYNTH_TEST_CHECK(synth::qwen3tts::flatten_talker_prompt(h, textless, text, codec, offset) ==
                     SYNTH_ERR_INVALID_ARG);

    synth::qwen3tts::TalkerPrompt empty;
    SYNTH_TEST_CHECK(synth::qwen3tts::flatten_talker_prompt(h, empty, text, codec, offset) ==
                     SYNTH_ERR_INVALID_ARG);
    return 0;
}

int check_rejections() {
    Context        context = make_graph_context();
    ggml_context * ctx     = context.get();
    ggml_cgraph *  graph   = ggml_new_graph_custom(ctx, kNodeBudget, false);

    synth::qwen3tts::TalkerWeights weights;
    weights.text_embedding           = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kTextHidden, kTextVocab);
    weights.text_projection_1.weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kTextHidden, kTextHidden);
    weights.text_projection_1.bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kTextHidden);
    weights.text_projection_2.weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kTextHidden, kHidden);
    weights.text_projection_2.bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHidden);
    weights.codec_embedding          = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, kCodecVocab);
    weights.codec_head               = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, kCodecVocab);
    weights.norm                     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHidden);
    weights.layers.resize(1);

    ggml_tensor * text_tokens  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kPositions);
    ggml_tensor * codec_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kCodecCount);

    // A projection whose bias is missing would silently drop it.
    synth::qwen3tts::TalkerWeights unbiased = weights;
    unbiased.text_projection_2.bias         = nullptr;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_text_projection(ctx, unbiased, text_tokens) == nullptr);

    // Token ids must be I32; an F32 vector would index by reinterpreted bits.
    ggml_tensor * float_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kPositions);
    SYNTH_TEST_CHECK(synth::qwen3tts::build_text_projection(ctx, weights, float_tokens) == nullptr);

    // The codec stream must end exactly at the last position; an offset that
    // leaves it short or hanging over is a layout defect.
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prefill_input(ctx, weights, text_tokens, codec_tokens, 0) ==
                     nullptr);
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prefill_input(ctx, weights, text_tokens, codec_tokens, -1) ==
                     nullptr);

    // A step whose cache does not cover its layers.
    synth::qwen3tts::TalkerCache cache;
    ggml_tensor *                input     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, 1);
    ggml_tensor *                positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor *                hidden    = nullptr;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_step(ctx, graph, input, positions, nullptr, weights, make_shape(),
                                                       cache, &hidden) == nullptr);

    // A step with no head to read logits from.
    synth::qwen3tts::TalkerWeights headless = weights;
    headless.codec_head                     = nullptr;
    cache.layers.resize(1);
    cache.layers[0].k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kPositions);
    cache.layers[0].v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kPositions);
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_step(ctx, graph, input, positions, nullptr, headless,
                                                       make_shape(), cache, &hidden) == nullptr);

    SYNTH_TEST_CHECK(ggml_graph_n_nodes(graph) == 0);
    return 0;
}

}  // namespace

int main() {
    SYNTH_TEST_CHECK(check_prompt_layout() == 0);
    SYNTH_TEST_CHECK(check_flatten() == 0);
    SYNTH_TEST_CHECK(check_rejections() == 0);

    const size_t device_count = ggml_backend_dev_count();
    SYNTH_TEST_CHECK(device_count > 0);

    size_t exercised = 0;
    for (size_t index = 0; index < device_count; ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (device == nullptr) {
            continue;
        }
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(device);
        if (type != GGML_BACKEND_DEVICE_TYPE_CPU && type != GGML_BACKEND_DEVICE_TYPE_GPU &&
            type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }
        float max_diff = 0.0f;
        SYNTH_TEST_CHECK(run_case(device, max_diff));
        std::printf("qwen3-tts-talker: %s (device type %d) max_diff %.3g\n", ggml_backend_dev_name(device), int(type),
                    double(max_diff));
        const float tolerance = type == GGML_BACKEND_DEVICE_TYPE_CPU ? 1e-4f : 5e-3f;
        SYNTH_TEST_CHECK(max_diff < tolerance);
        ++exercised;
    }
    SYNTH_TEST_CHECK(exercised > 0);
    return 0;
}
