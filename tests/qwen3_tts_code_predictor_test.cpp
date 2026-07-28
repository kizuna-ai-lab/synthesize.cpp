// One frame of the code predictor, compared against the reference
// implementation's own generate() loop.
//
// The predictor is the family's hot loop: it runs code_group_count - 1 times per
// frame, each call reading a different embedding table and a different output
// head. That mapping is off-by-one bait -- a wrong table still produces
// plausible codes -- so the test drives the real schedule rather than a
// hand-written one, and checks every step's logits, not just the codes they
// select.
//
// Reference values come from scripts/dump_reference_qwen3_tts_code_predictor.py.
// It decodes greedily on purpose: the reference draws from PyTorch's generator
// and this port from its own seeded stream, so only the distribution behind the
// draw is comparable. Sampling itself is checked separately below.

#include "arch/qwen3-tts/code-predictor-host.h"
#include "arch/qwen3-tts/code-predictor.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "random-stream.h"
#include "test-assert.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

namespace {

constexpr uint32_t kHidden       = 8;
constexpr uint32_t kHeads        = 4;
constexpr uint32_t kKvHeads      = 2;
constexpr uint32_t kHeadDim      = 4;
constexpr uint32_t kIntermediate = 16;
constexpr uint32_t kLayers       = 2;
constexpr uint32_t kVocab        = 12;
constexpr uint32_t kCodeGroups   = 5;
constexpr float    kRmsNormEps   = 1e-6f;
constexpr float    kRopeTheta    = 1000000.0f;
constexpr uint64_t kSeed         = 20260728u;

// 4 steps x 12 vocabulary, step-major.
constexpr float kExpectedLogits[] = {
    1.9114958f,    -0.212811843f,  1.81460583f,    2.15212417f,   -1.72130978f,  -0.198428839f, -1.54300416f,
    0.720535517f,  0.50348711f,    0.191554591f,   0.970674276f,  -0.16088739f,  -0.593399167f, -0.888796091f,
    -0.427311599f, -0.94523102f,   0.212974012f,   -1.77432179f,  -0.49431631f,  -0.437291354f, 0.21668376f,
    0.218171254f,  0.404909164f,   -0.116344802f,  -0.667533398f, 0.424502909f,  -0.613975763f, -0.831218541f,
    -0.259975433f, -1.6156106f,    -0.0992554128f, 1.09658647f,   0.340317011f,  -0.661373615f, 1.68910265f,
    1.56436217f,   -0.615264773f,  0.371960461f,   1.15123785f,   -1.37388122f,  -0.60074091f,  0.562539399f,
    1.37328827f,   -0.0802822039f, 1.40103304f,    0.397551984f,  -0.820950806f, -0.809743226f,
};

constexpr int32_t kExpectedCodes[] = { 3, 10, 10, 8 };

// Sum of each acoustic code's embedding from its own group's table.
constexpr float kExpectedSummedEmbedding[] = {
    0.253510892f, -0.330078065f, 0.119186223f, -0.536568522f, 0.450968087f, 0.44172591f, 0.703374445f, 0.344540179f,
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
    void operator()(ggml_context * context) const { ggml_free(context); }
};

using Context = std::unique_ptr<ggml_context, ContextDeleter>;

Context make_context(size_t bytes) {
    ggml_init_params parameters = {};
    parameters.mem_size         = bytes;
    parameters.no_alloc         = true;
    return Context(ggml_init(parameters));
}

constexpr size_t kNodeBudget = 1024;

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

// Everything the frame needs on one device, so the persistent tensors and the
// backend stay together for the whole loop.
struct Fixture {
    ggml_backend_t                        backend = nullptr;
    Context                               persistent;
    ggml_backend_buffer_t                 buffer = nullptr;
    synth::qwen3tts::CodePredictorWeights weights;
    synth::qwen3tts::CodePredictorCache   cache;
    ggml_tensor *                         prefill      = nullptr;
    ggml_tensor *                         token        = nullptr;
    ggml_tensor *                         prefill_pos  = nullptr;
    ggml_tensor *                         step_pos     = nullptr;
    ggml_tensor *                         prefill_mask = nullptr;
    ggml_tensor *                         codes        = nullptr;

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

    // The fill order below is the reference script's; a tensor added out of
    // order shifts every weight after it.
    std::vector<ggml_tensor *> ordered;
    std::vector<float>         scales;
    std::vector<float>         offsets;
    auto                       add = [&](ggml_tensor * tensor, float scale, float offset) {
        ordered.push_back(tensor);
        scales.push_back(scale);
        offsets.push_back(offset);
        return tensor;
    };

    fixture.weights.layers.resize(kLayers);
    for (uint32_t index = 0; index < kLayers; ++index) {
        synth::qwen3tts::DecoderLayerWeights & layer = fixture.weights.layers[index];
        layer.input_layernorm = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kHidden), 0.25f, 1.0f);
        layer.q_proj          = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kHeads * kHeadDim), 0.5f, 0.0f);
        layer.k_proj          = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kKvHeads * kHeadDim), 0.5f, 0.0f);
        layer.v_proj          = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kKvHeads * kHeadDim), 0.5f, 0.0f);
        layer.o_proj          = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHeads * kHeadDim, kHidden), 0.5f, 0.0f);
        layer.q_norm          = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kHeadDim), 0.25f, 1.0f);
        layer.k_norm          = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kHeadDim), 0.25f, 1.0f);
        layer.post_attention_layernorm = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kHidden), 0.25f, 1.0f);
        layer.gate_proj = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kIntermediate), 0.5f, 0.0f);
        layer.up_proj   = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kIntermediate), 0.5f, 0.0f);
        layer.down_proj = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kIntermediate, kHidden), 0.5f, 0.0f);
    }
    fixture.weights.norm = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kHidden), 0.25f, 1.0f);
    for (uint32_t group = 0; group + 1 < kCodeGroups; ++group) {
        fixture.weights.codec_embedding.push_back(
            add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kVocab), 0.5f, 0.0f));
    }
    for (uint32_t group = 0; group + 1 < kCodeGroups; ++group) {
        fixture.weights.lm_head.push_back(add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kVocab), 0.5f, 0.0f));
    }

    fixture.cache.layers.resize(kLayers);
    for (uint32_t index = 0; index < kLayers; ++index) {
        fixture.cache.layers[index].k = ggml_new_tensor_3d(pctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCodeGroups);
        fixture.cache.layers[index].v = ggml_new_tensor_3d(pctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCodeGroups);
    }

    fixture.prefill      = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, 2);
    fixture.token        = ggml_new_tensor_1d(pctx, GGML_TYPE_I32, 1);
    fixture.prefill_pos  = ggml_new_tensor_1d(pctx, GGML_TYPE_I32, 2);
    fixture.step_pos     = ggml_new_tensor_1d(pctx, GGML_TYPE_I32, 1);
    fixture.prefill_mask = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, 2, 2);
    fixture.codes        = ggml_new_tensor_1d(pctx, GGML_TYPE_I32, kCodeGroups - 1);

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

    const std::vector<float> prefill = stream.fill(2 * size_t(kHidden), 0.5f, 0.0f);
    ggml_backend_tensor_set(fixture.prefill, prefill.data(), 0, ggml_nbytes(fixture.prefill));

    const int32_t prefill_positions[2] = { 0, 1 };
    ggml_backend_tensor_set(fixture.prefill_pos, prefill_positions, 0, ggml_nbytes(fixture.prefill_pos));

    const float infinity = std::numeric_limits<float>::infinity();
    const float mask[4]  = { 0.0f, -infinity, 0.0f, 0.0f };
    ggml_backend_tensor_set(fixture.prefill_mask, mask, 0, ggml_nbytes(fixture.prefill_mask));
    return true;
}

// Runs one graph and copies `output` back to the host.
bool compute(Fixture & fixture, ggml_cgraph * graph, ggml_tensor * output, std::vector<float> & values) {
    ggml_build_forward_expand(graph, output);
    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(fixture.backend));
    bool           ok        = allocator != nullptr && ggml_gallocr_alloc_graph(allocator, graph);
    if (ok) {
        ok = ggml_backend_graph_compute(fixture.backend, graph) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        values.resize(size_t(ggml_nelements(output)));
        ggml_backend_tensor_get(output, values.data(), 0, ggml_nbytes(output));
    }
    if (allocator != nullptr) {
        ggml_gallocr_free(allocator);
    }
    return ok;
}

// Decodes one frame greedily and returns the largest deviation from the
// reference logits and the summed embedding.
bool run_frame(ggml_backend_dev_t device, float & max_diff, std::vector<int32_t> & codes) {
    Fixture fixture;
    if (!build_fixture(device, fixture)) {
        return false;
    }

    const synth::qwen3tts::AttentionShape                 shape = make_shape();
    const std::vector<synth::qwen3tts::CodePredictorStep> schedule =
        synth::qwen3tts::code_predictor_schedule(kCodeGroups);
    if (schedule.size() != kCodeGroups - 1) {
        return false;
    }

    synth::NormalRandomStream       stream(1u);
    synth::qwen3tts::SamplingParams greedy;
    greedy.enabled = false;

    max_diff = 0.0f;
    codes.clear();
    int64_t filled = 0;
    for (size_t index = 0; index < schedule.size(); ++index) {
        const synth::qwen3tts::CodePredictorStep & step = schedule[index];
        if (step.first_position != filled) {
            return false;
        }

        Context       graph_ctx = make_graph_context();
        ggml_cgraph * graph     = ggml_new_graph_custom(graph_ctx.get(), kNodeBudget, false);

        ggml_tensor * input     = nullptr;
        ggml_tensor * positions = nullptr;
        ggml_tensor * mask      = nullptr;
        if (step.embedding_table == synth::qwen3tts::kPrefillStep) {
            input     = fixture.prefill;
            positions = fixture.prefill_pos;
            mask      = fixture.prefill_mask;
        } else {
            const int32_t previous = codes.back();
            ggml_backend_tensor_set(fixture.token, &previous, 0, ggml_nbytes(fixture.token));
            const int32_t position = int32_t(step.first_position);
            ggml_backend_tensor_set(fixture.step_pos, &position, 0, ggml_nbytes(fixture.step_pos));
            input = synth::qwen3tts::code_predictor_embed(graph_ctx.get(), fixture.weights, step.embedding_table,
                                                          fixture.token);
            if (input == nullptr) {
                return false;
            }
            positions = fixture.step_pos;
        }

        for (uint32_t layer = 0; layer < kLayers; ++layer) {
            fixture.cache.layers[layer].filled = filled;
        }
        ggml_tensor * logits = synth::qwen3tts::build_code_predictor(
            graph_ctx.get(), graph, input, positions, mask, fixture.weights, shape, step.lm_head, fixture.cache);
        if (logits == nullptr || logits->ne[0] != int64_t(kVocab) || logits->ne[1] != 1) {
            return false;
        }

        std::vector<float> values;
        if (!compute(fixture, graph, logits, values)) {
            return false;
        }
        for (size_t slot = 0; slot < values.size(); ++slot) {
            max_diff = std::fmax(max_diff, std::fabs(values[slot] - kExpectedLogits[index * kVocab + slot]));
        }
        codes.push_back(int32_t(synth::qwen3tts::select_code(values, greedy, stream)));
        filled += step.position_count;
    }

    if (filled != int64_t(kCodeGroups)) {
        return false;
    }

    // The summed embedding the talker consumes as its next input.
    ggml_backend_tensor_set(fixture.codes, codes.data(), 0, ggml_nbytes(fixture.codes));
    Context       sum_ctx = make_graph_context();
    ggml_cgraph * graph   = ggml_new_graph_custom(sum_ctx.get(), kNodeBudget, false);
    ggml_tensor * summed  = synth::qwen3tts::sum_code_embeddings(sum_ctx.get(), fixture.weights, fixture.codes);
    if (summed == nullptr || summed->ne[0] != int64_t(kHidden)) {
        return false;
    }
    std::vector<float> values;
    if (!compute(fixture, graph, summed, values)) {
        return false;
    }
    for (size_t slot = 0; slot < values.size(); ++slot) {
        max_diff = std::fmax(max_diff, std::fabs(values[slot] - kExpectedSummedEmbedding[slot]));
    }
    return true;
}

int check_schedule() {
    // Sixteen code groups is the real variant: one prefill of two positions plus
    // fourteen single-token steps, producing fifteen acoustic codes over sixteen
    // positions.
    const std::vector<synth::qwen3tts::CodePredictorStep> schedule = synth::qwen3tts::code_predictor_schedule(16);
    SYNTH_TEST_CHECK(schedule.size() == 15);
    SYNTH_TEST_CHECK(schedule.front().embedding_table == synth::qwen3tts::kPrefillStep);
    SYNTH_TEST_CHECK(schedule.front().lm_head == 0);
    SYNTH_TEST_CHECK(schedule.front().first_position == 0);
    SYNTH_TEST_CHECK(schedule.front().position_count == 2);

    int64_t filled = 0;
    for (size_t index = 0; index < schedule.size(); ++index) {
        const synth::qwen3tts::CodePredictorStep & step = schedule[index];
        SYNTH_TEST_CHECK(step.first_position == filled);
        SYNTH_TEST_CHECK(step.lm_head == index);
        if (index > 0) {
            // Each step embeds the previous step's code through the table of the
            // group that code belongs to, which is one behind this step's head.
            SYNTH_TEST_CHECK(step.embedding_table == step.lm_head - 1);
            SYNTH_TEST_CHECK(step.position_count == 1);
        }
        filled += step.position_count;
    }
    SYNTH_TEST_CHECK(filled == 16);

    // A package with nothing to predict yields no steps rather than a loop that
    // reads head -1.
    SYNTH_TEST_CHECK(synth::qwen3tts::code_predictor_schedule(1).empty());
    SYNTH_TEST_CHECK(synth::qwen3tts::code_predictor_schedule(0).empty());
    SYNTH_TEST_CHECK(synth::qwen3tts::code_predictor_schedule(2).size() == 1);
    return 0;
}

int check_sampling() {
    synth::NormalRandomStream stream(7u);

    // Greedy ignores the stream entirely.
    synth::qwen3tts::SamplingParams greedy;
    greedy.enabled = false;
    SYNTH_TEST_CHECK(synth::qwen3tts::select_code({ 0.1f, 0.9f, 0.3f }, greedy, stream) == 1);

    // A temperature low enough collapses the distribution onto the largest
    // logit, so sampling and greedy agree.
    synth::qwen3tts::SamplingParams cold;
    cold.temperature = 1e-3f;
    for (int trial = 0; trial < 16; ++trial) {
        SYNTH_TEST_CHECK(synth::qwen3tts::select_code({ 0.1f, 0.9f, 0.3f }, cold, stream) == 1);
    }

    // top-k is a hard filter: with k = 1 nothing but the largest can be drawn,
    // however flat the temperature makes the distribution.
    synth::qwen3tts::SamplingParams single;
    single.top_k       = 1;
    single.temperature = 100.0f;
    for (int trial = 0; trial < 16; ++trial) {
        SYNTH_TEST_CHECK(synth::qwen3tts::select_code({ 0.1f, 0.9f, 0.3f, 0.5f }, single, stream) == 1);
    }

    // So is top-p at its lower edge, which keeps only the most likely code.
    synth::qwen3tts::SamplingParams narrow;
    narrow.top_p       = 0.0f;
    narrow.temperature = 100.0f;
    for (int trial = 0; trial < 16; ++trial) {
        SYNTH_TEST_CHECK(synth::qwen3tts::select_code({ 0.1f, 0.9f, 0.3f, 0.5f }, narrow, stream) == 1);
    }

    // The package's own defaults must reach beyond the largest logit, or the
    // head would be greedy in all but name.
    synth::qwen3tts::SamplingParams defaults;
    SYNTH_TEST_CHECK(defaults.enabled);
    SYNTH_TEST_CHECK(std::fabs(defaults.temperature - 0.9f) < 1e-6f);
    SYNTH_TEST_CHECK(defaults.top_k == 50);
    SYNTH_TEST_CHECK(std::fabs(defaults.top_p - 1.0f) < 1e-6f);

    const std::vector<float> flat = { 0.0f, 0.0f, 0.0f, 0.0f };
    std::vector<int>         hits(flat.size(), 0);
    for (int trial = 0; trial < 400; ++trial) {
        hits[synth::qwen3tts::select_code(flat, defaults, stream)] += 1;
    }
    for (size_t index = 0; index < hits.size(); ++index) {
        SYNTH_TEST_CHECK(hits[index] > 0);
    }

    // An empty distribution is a wiring defect upstream, not something to draw
    // from; it yields code zero rather than reading past the end.
    SYNTH_TEST_CHECK(synth::qwen3tts::select_code({}, defaults, stream) == 0);
    return 0;
}

int check_rejections() {
    Context        context = make_graph_context();
    ggml_context * ctx     = context.get();
    ggml_cgraph *  graph   = ggml_new_graph_custom(ctx, kNodeBudget, false);

    synth::qwen3tts::CodePredictorWeights weights;
    weights.layers.resize(1);
    weights.norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHidden);
    weights.codec_embedding.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, kVocab));
    weights.lm_head.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, kVocab));

    synth::qwen3tts::CodePredictorCache cache;
    cache.layers.resize(1);
    cache.layers[0].k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCodeGroups);
    cache.layers[0].v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCodeGroups);

    ggml_tensor * input     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, 1);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * ids       = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    const synth::qwen3tts::AttentionShape shape = make_shape();

    // An output head outside the catalog. The loop indexes heads by group, so a
    // package whose group count and head count disagree must not silently read
    // the last head for every group past it.
    SYNTH_TEST_CHECK(synth::qwen3tts::build_code_predictor(ctx, graph, input, positions, nullptr, weights, shape, 1,
                                                           cache) == nullptr);

    // A cache with the wrong number of layers.
    synth::qwen3tts::CodePredictorCache short_cache;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_code_predictor(ctx, graph, input, positions, nullptr, weights, shape, 0,
                                                           short_cache) == nullptr);

    // A projection bound on one side only would drop its bias without a word.
    synth::qwen3tts::CodePredictorWeights half_projected = weights;
    half_projected.input_projection                      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, kHidden);
    SYNTH_TEST_CHECK(synth::qwen3tts::build_code_predictor(ctx, graph, input, positions, nullptr, half_projected, shape,
                                                           0, cache) == nullptr);

    // Embedding tables are addressed by group; one outside the catalog is a
    // defect, not a lookup to clamp.
    SYNTH_TEST_CHECK(synth::qwen3tts::code_predictor_embed(ctx, weights, 1, ids) == nullptr);
    SYNTH_TEST_CHECK(synth::qwen3tts::code_predictor_embed(ctx, weights, 0, input) == nullptr);

    // The summed embedding needs exactly one code per acoustic group.
    ggml_tensor * two_codes = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
    SYNTH_TEST_CHECK(synth::qwen3tts::sum_code_embeddings(ctx, weights, two_codes) == nullptr);
    SYNTH_TEST_CHECK(synth::qwen3tts::sum_code_embeddings(ctx, weights, input) == nullptr);

    SYNTH_TEST_CHECK(ggml_graph_n_nodes(graph) == 0);
    return 0;
}

}  // namespace

int main() {
    SYNTH_TEST_CHECK(check_schedule() == 0);
    SYNTH_TEST_CHECK(check_sampling() == 0);
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

        float                max_diff = 0.0f;
        std::vector<int32_t> codes;
        SYNTH_TEST_CHECK(run_frame(device, max_diff, codes));
        std::printf("qwen3-tts-code-predictor: %s (device type %d) max_diff %.3g\n", ggml_backend_dev_name(device),
                    int(type), double(max_diff));

        const float tolerance = type == GGML_BACKEND_DEVICE_TYPE_CPU ? 1e-4f : 5e-3f;
        SYNTH_TEST_CHECK(max_diff < tolerance);
        SYNTH_TEST_CHECK(codes.size() == std::size(kExpectedCodes));
        for (size_t slot = 0; slot < codes.size(); ++slot) {
            SYNTH_TEST_CHECK(codes[slot] == kExpectedCodes[slot]);
        }
        ++exercised;
    }
    SYNTH_TEST_CHECK(exercised > 0);
    return 0;
}
