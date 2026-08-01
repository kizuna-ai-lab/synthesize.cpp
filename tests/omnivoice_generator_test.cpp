// The mask-predict generator's graph, compared against the real Qwen3 class.
//
// Three things are pinned here.
//
// The **embedding merge**: upstream `_prepare_embed_inputs` ends in
// `torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)` -- a
// SELECT, not a sum. An audio position carries the summed offset codebook
// embeddings alone. A port that added the two streams would still produce
// audio, just the wrong audio, so the merge is checked against the reference
// arithmetic rather than argued from the source.
//
// The **block**: the ordinary Qwen3 one -- per-head q/k norms before rope, no
// layer scales -- run bidirectionally. The reference side is
// `transformers.Qwen3Model` itself driven with the 4-D all-True boolean mask
// OmniVoice's `_prepare_batch` builds, so "bidirectional" is measured and not
// assumed; the reference script asserts that perturbing the last position moves
// the first one before it prints a single value.
//
// The **heads**: bias-free, over EVERY position (there is no last-position
// slice in a mask-predict model), with the stacked row order c * vocab + v that
// the decode loop will index by.
//
// Weights are not carried in either file. Both sides draw them from the same
// 64-bit LCG in the same order, so only the outputs are pinned.

#include "arch/omnivoice/catalog.h"
#include "arch/omnivoice/generator.h"
#include "arch/omnivoice/weights.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "test-assert.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

namespace {

constexpr uint32_t kHidden         = 8;
constexpr uint32_t kHeads          = 4;
constexpr uint32_t kKvHeads        = 2;
constexpr uint32_t kHeadDim        = 8;
constexpr uint32_t kIntermediate   = 16;
// Two layers, not one. At depth 1 a stack that fed the embeddings to every layer
// instead of the running hidden state would be indistinguishable from a correct
// one, and so would a probe list that pushed the wrong tensor.
constexpr uint32_t kLayers         = 2;
constexpr uint32_t kTextVocab      = 20;
constexpr uint32_t kAudioVocab     = 6;  // toy canvas vocabulary; mask id 5
constexpr uint32_t kCodebooks      = 3;
constexpr uint32_t kTextPositions  = 2;
constexpr uint32_t kAudioPositions = 3;
constexpr uint32_t kPositions      = kTextPositions + kAudioPositions;
constexpr uint32_t kAudioRows      = kCodebooks * kAudioVocab;
constexpr float    kRmsNormEps     = 1e-6f;
constexpr float    kRopeTheta      = 1000000.0f;
constexpr uint64_t kSeed           = 20260731u;

constexpr int32_t kTextIds[kTextPositions] = { 3, 7 };

// The reference's AUDIO_GRID with the host shift `id + codebook * kAudioVocab`
// already applied, codebook-major. The shift is the host's job: the eight
// codebooks share one stacked table, and the graph only ever sees rows.
constexpr int32_t kAudioIds[kCodebooks * kAudioPositions] = {
    5,  0,  2,   // codebook 0: {5, 0, 2} + 0
    11, 7,  11,  // codebook 1: {5, 1, 5} + 6
    16, 17, 15,  // codebook 2: {4, 5, 3} + 12
};

// values from scripts/dump_reference_omnivoice_generator.py, pasted verbatim, at
// transformers 5.14.1, torch 2.13.0+cu130. That run's two self-checks: the
// bidirectionality probe moved position 0 by 0.4756, and the closest two layer
// probes differ by 1.962.

// 5 positions x 8 hidden, position-major: text rows 0-1, audio rows 2-4.
constexpr float kExpectedMerged[] = {
    -0.350337803f, 0.341513932f,  -0.264598131f, -0.145631552f,  -0.49714303f,  0.0172014832f, -0.151831567f,
    -0.459532678f, -0.416963518f, 0.353890836f,  -0.0641898513f, -0.166313589f, 0.0815419555f, 0.477606475f,
    0.0727072954f, 0.150041044f,  0.15332222f,   0.211242199f,   -0.425216079f, 0.121826231f,  -0.298108459f,
    0.43455857f,   -0.212404191f, 0.0686778426f, 0.0763853788f,  0.377587497f,  0.693324327f,  0.785543025f,
    0.379410684f,  -0.454412282f, -0.212570429f, -0.700475931f,  -0.282034814f, 0.110867023f,  0.321638882f,
    0.0584337711f, 0.337857962f,  0.396249712f,  -0.17385006f,   0.646888793f,
};

// After each block, before the final norm. Captured off per-layer forward
// hooks: transformers 5.x records hidden states through a generic decorator
// whose last entry is already post-norm. The two layers' weights are drawn
// from different stretches of the stream, so these arrays differ by 1.962 at
// their closest -- which is what makes a stack that skipped the chaining
// (feeding the embeddings to both layers) visible here.
constexpr float kExpectedLayer0Output[] = {
    -0.106362939f, 0.977852643f,   0.277025819f,  -0.475410461f, 0.655190587f, -0.216368973f, -0.118029535f,
    -1.42813838f,  0.277654916f,   0.642834425f,  -0.498886913f, 0.449305534f, 0.208927676f,  1.10728037f,
    -0.785657644f, -0.555486739f,  1.52539599f,   1.0017277f,    -0.27838552f, 1.10587621f,   0.171068281f,
    0.846385717f,  -0.614286304f,  -0.899877667f, 1.59533381f,   0.966762424f, 0.866229653f,  1.61224735f,
    0.477152407f,  -0.0247476697f, -0.893645644f, -2.54164004f,  1.16578507f,  0.634394646f,  0.433816791f,
    0.581606388f,  0.157043219f,   0.928154171f,  -0.948468864f, -1.46784914f,
};

constexpr float kExpectedLayer1Output[] = {
    -1.60530567f,  1.12391877f, 2.13784313f, -0.45530507f,   2.37038589f, 0.193897069f, -0.643466175f, -1.45853758f,
    -1.62266505f,  1.09111381f, 1.22014666f, 0.00430706143f, 2.02704525f, 1.37434733f,  -1.68405962f,  -0.729590416f,
    -0.373707175f, 1.48569536f, 1.34590876f, 0.767348051f,   1.88991702f, 1.24342012f,  -1.71673071f,  -0.602498293f,
    -0.219197989f, 1.60885024f, 2.18288803f, 1.19218206f,    2.18411565f, 0.153822243f, -2.06274557f,  -2.27419066f,
    -0.491003901f, 1.13955045f, 1.93190134f, 0.151296973f,   2.11946511f, 1.05000317f,  -2.03405929f,  -1.14486718f,
};

// After the final norm.
constexpr float kExpectedFinal[] = {
    -1.20396352f,  0.718925655f, 1.45343661f,  -0.322750002f,  1.58722103f, 0.147072986f,  -0.535383403f, -0.840886116f,
    -1.30256772f,  0.747024715f, 0.887867451f, 0.00326783909f, 1.45277262f, 1.11576831f,   -1.49972808f,  -0.450209379f,
    -0.316637844f, 1.07362998f,  1.03374088f,  0.614514232f,   1.42967367f, 1.06550467f,   -1.61367917f,  -0.392420083f,
    -0.140434146f, 0.879114032f, 1.26774609f,  0.721916318f,   1.2493223f,  0.0996692851f, -1.46610618f,  -1.12002313f,
    -0.373596668f, 0.739511728f, 1.33250141f,  0.108806908f,   1.43981636f, 0.808006287f,  -1.71698022f,  -0.669633448f,
};

// The heads over every position, flat order s * 18 + c * 6 + v -- which is what
// a contiguous ggml [vocab, codebooks, positions] tensor reads back as.
constexpr float kExpectedLogits[] = {
    0.144517273f,  -0.0795090348f, 1.31958592f,   -0.545370758f, 0.787238657f,  -0.931361079f,  -0.494994193f,
    -1.00499439f,  -0.253023237f,  -0.964516103f, 0.0650293306f, -1.51765871f,  0.0506015122f,  0.0337602608f,
    -0.530017138f, -0.688528299f,  1.25869465f,   -0.387598664f, 0.08630009f,   -0.459124386f,  1.64854658f,
    -0.586081624f, 1.04914212f,    -0.692183673f, -0.334894449f, -1.09573174f,  -0.0799395591f, -1.23264885f,
    -0.397595704f, -1.66165411f,   0.165087417f,  0.810793281f,  -0.271938533f, -0.627276897f,  1.46544588f,
    0.339684904f,  0.348380446f,   -0.393690765f, 1.33714986f,   -0.27128312f,  1.22666299f,    -0.4371517f,
    -1.11150241f,  -0.673130512f,  -0.085050486f, -1.07363641f,  -0.762144029f, -1.49048626f,   0.319258332f,
    0.931585371f,  -0.164946675f,  -0.911064327f, 1.64098787f,   0.809039652f,  0.805777311f,   -0.242239952f,
    0.902696013f,  -0.176884651f,  1.01119781f,   0.139548406f,  -1.68398726f,  -0.15999043f,   0.119909354f,
    -0.9156968f,   -0.907952964f,  -1.32181001f,  -0.271737576f, 0.711509109f,  -0.361992627f,  -0.890426397f,
    1.86110198f,   0.463737905f,   0.543822229f,  -0.626145899f, 1.20666742f,   -0.366357803f,  1.31072283f,
    -0.400758982f, -0.99860549f,   -0.366008431f, -0.2035117f,   -1.353912f,    -0.849338531f,  -1.53872657f,
    -0.13943027f,  0.781410336f,   -0.470202208f, -0.847447217f, 1.57156777f,   0.52707094f,
};

// The twin of LcgStream in the reference script. Every value is a 24-bit
// numerator over 2^23, so both sides start from bit-identical inputs.
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

synth::omnivoice::AttentionShape make_shape() {
    synth::omnivoice::AttentionShape shape;
    shape.hidden_size          = kHidden;
    shape.attention_head_count = kHeads;
    shape.key_value_head_count = kKvHeads;
    shape.head_dim             = kHeadDim;
    shape.rms_norm_eps         = kRmsNormEps;
    shape.rope_theta           = kRopeTheta;
    return shape;
}

synth::omnivoice::AudioCanvasParams make_canvas() {
    synth::omnivoice::AudioCanvasParams canvas;
    canvas.num_codebooks = kCodebooks;
    canvas.vocab_size    = kAudioVocab;
    canvas.mask_id       = kAudioVocab - 1;
    return canvas;
}

struct Fixture {
    ggml_backend_t                     backend = nullptr;
    Context                            persistent;
    ggml_backend_buffer_t              buffer = nullptr;
    synth::omnivoice::GeneratorWeights weights;
    ggml_tensor *                      text_ids   = nullptr;
    ggml_tensor *                      audio_ids  = nullptr;
    ggml_tensor *                      positions  = nullptr;
    // The mask-gating check feeds the merged canvas back in as a plain input so
    // it can also run a shortened prefix of it.
    ggml_tensor *                      canvas     = nullptr;
    ggml_tensor *                      prefix     = nullptr;
    ggml_tensor *                      prefix_pos = nullptr;
    ggml_tensor *                      gate_mask  = nullptr;

    Fixture()                            = default;
    Fixture(const Fixture &)             = delete;
    Fixture & operator=(const Fixture &) = delete;

    ~Fixture() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
};

// How many positions the mask-gating check keeps when it drops the last one.
constexpr uint32_t kGatedPrefix = kPositions - 1;

bool build_fixture(ggml_backend_dev_t device, Fixture & fixture) {
    fixture.backend = ggml_backend_dev_init(device, nullptr);
    if (fixture.backend == nullptr) {
        return false;
    }
    fixture.persistent  = make_context(ggml_tensor_overhead() * 64);
    ggml_context * pctx = fixture.persistent.get();
    if (pctx == nullptr) {
        return false;
    }

    std::vector<ggml_tensor *> ordered;
    std::vector<float>         scales;
    std::vector<float>         offsets;
    auto                       add = [&](ggml_tensor * tensor, float scale, float offset) {
        ordered.push_back(tensor);
        scales.push_back(scale);
        offsets.push_back(offset);
        return tensor;
    };

    // The fill order is the reference script's, and the ggml shapes are the
    // torch shapes reversed. Reordering any line silently changes every weight.
    synth::omnivoice::GeneratorWeights & w = fixture.weights;
    w.text_embedding   = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kTextVocab), 0.5f, 0.0f);
    w.audio_embeddings = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kAudioRows), 0.5f, 0.0f);
    w.audio_heads      = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kAudioRows), 0.5f, 0.0f);
    w.norm             = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kHidden), 0.25f, 1.0f);

    // Every layer's eleven, in the script's order, so layer 1 draws a different
    // stretch of the stream than layer 0 and the two blocks cannot be confused.
    w.layers.resize(kLayers);
    for (uint32_t index = 0; index < kLayers; ++index) {
        synth::omnivoice::GeneratorLayerWeights & layer = w.layers[index];
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

    fixture.text_ids   = ggml_new_tensor_1d(pctx, GGML_TYPE_I32, kTextPositions);
    fixture.audio_ids  = ggml_new_tensor_2d(pctx, GGML_TYPE_I32, kAudioPositions, kCodebooks);
    fixture.positions  = ggml_new_tensor_1d(pctx, GGML_TYPE_I32, kPositions);
    fixture.canvas     = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kPositions);
    fixture.prefix     = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kGatedPrefix);
    fixture.prefix_pos = ggml_new_tensor_1d(pctx, GGML_TYPE_I32, kGatedPrefix);
    fixture.gate_mask  = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kPositions, kPositions);

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

    ggml_backend_tensor_set(fixture.text_ids, kTextIds, 0, ggml_nbytes(fixture.text_ids));
    ggml_backend_tensor_set(fixture.audio_ids, kAudioIds, 0, ggml_nbytes(fixture.audio_ids));

    int32_t sequential[kPositions];
    for (uint32_t index = 0; index < kPositions; ++index) {
        sequential[index] = int32_t(index);
    }
    ggml_backend_tensor_set(fixture.positions, sequential, 0, ggml_nbytes(fixture.positions));
    ggml_backend_tensor_set(fixture.prefix_pos, sequential, 0, ggml_nbytes(fixture.prefix_pos));

    // The merged canvas as a plain input, and its first four positions.
    ggml_backend_tensor_set(fixture.canvas, kExpectedMerged, 0, ggml_nbytes(fixture.canvas));
    ggml_backend_tensor_set(fixture.prefix, kExpectedMerged, 0, ggml_nbytes(fixture.prefix));

    // Every query may read every key except the last, which only reads itself.
    // Against a mask-free run of the first four positions this is the same
    // problem, so any difference in those four is the mask failing to gate.
    const float        infinity = std::numeric_limits<float>::infinity();
    std::vector<float> mask(size_t(kPositions) * kPositions, 0.0f);
    for (uint32_t query = 0; query < kPositions; ++query) {
        for (uint32_t key = 0; key < kPositions; ++key) {
            const bool last_query                  = query == kPositions - 1;
            const bool last_key                    = key == kPositions - 1;
            const bool visible                     = last_query ? last_key : !last_key;
            mask[size_t(query) * kPositions + key] = visible ? 0.0f : -infinity;
        }
    }
    ggml_backend_tensor_set(fixture.gate_mask, mask.data(), 0, ggml_nbytes(fixture.gate_mask));
    return true;
}

bool compute(Fixture &                          fixture,
             ggml_cgraph *                      graph,
             const std::vector<ggml_tensor *> & outputs,
             std::vector<std::vector<float>> &  values) {
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

// The merge and the forward share one graph: the forward runs on the merged
// tensor the merge produced, so the two are checked as the chain they are.
bool check_merge_and_forward(Fixture & fixture, float & worst) {
    Context       graph_ctx = make_graph_context();
    ggml_cgraph * graph     = ggml_new_graph_custom(graph_ctx.get(), kNodeBudget, false);

    ggml_tensor * merged = synth::omnivoice::build_canvas_embedding(graph_ctx.get(), fixture.weights, fixture.text_ids,
                                                                    fixture.audio_ids, kCodebooks);
    if (merged == nullptr || merged->ne[0] != int64_t(kHidden) || merged->ne[1] != int64_t(kPositions)) {
        return false;
    }

    std::vector<ggml_tensor *> layers;
    ggml_tensor *              final = nullptr;
    // No mask: full bidirectional attention is the only mode synthesis uses.
    ggml_tensor *              logits =
        synth::omnivoice::build_generator_forward(graph_ctx.get(), merged, fixture.positions, nullptr, fixture.weights,
                                                  make_shape(), make_canvas(), &layers, &final);
    if (logits == nullptr || final == nullptr || layers.size() != kLayers || logits->ne[0] != int64_t(kAudioVocab) ||
        logits->ne[1] != int64_t(kCodebooks) || logits->ne[2] != int64_t(kPositions)) {
        return false;
    }

    std::vector<std::vector<float>> values;
    if (!compute(fixture, graph, { merged, layers[0], layers[1], final, logits }, values)) {
        return false;
    }
    const float d0 = deviation(values[0], kExpectedMerged, std::size(kExpectedMerged));
    // Both layers, separately. Layer 1's expected values are what a stack that
    // fed `embeddings` to every block instead of the running hidden state would
    // miss -- at depth 1 that defect has nothing to disagree with.
    const float d1 = deviation(values[1], kExpectedLayer0Output, std::size(kExpectedLayer0Output));
    const float d2 = deviation(values[2], kExpectedLayer1Output, std::size(kExpectedLayer1Output));
    const float d3 = deviation(values[3], kExpectedFinal, std::size(kExpectedFinal));
    const float d4 = deviation(values[4], kExpectedLogits, std::size(kExpectedLogits));
    std::printf("  merged %.3g  layer0 %.3g  layer1 %.3g  final %.3g  logits %.3g\n", double(d0), double(d1),
                double(d2), double(d3), double(d4));
    worst = std::fmax(std::fmax(std::fmax(d0, d1), std::fmax(d2, d3)), d4);
    return true;
}

// The unconditional branch has no text region at all, so the merge must build a
// canvas out of the audio stream alone -- the same three columns the merged
// canvas ends with, not a shifted or padded variant of them.
bool check_uncond_embedding(Fixture & fixture, float & worst) {
    Context       graph_ctx = make_graph_context();
    ggml_cgraph * graph     = ggml_new_graph_custom(graph_ctx.get(), kNodeBudget, false);

    ggml_tensor * audio_only = synth::omnivoice::build_canvas_embedding(graph_ctx.get(), fixture.weights, nullptr,
                                                                        fixture.audio_ids, kCodebooks);
    if (audio_only == nullptr || audio_only->ne[0] != int64_t(kHidden) ||
        audio_only->ne[1] != int64_t(kAudioPositions)) {
        return false;
    }

    std::vector<std::vector<float>> values;
    if (!compute(fixture, graph, { audio_only }, values)) {
        return false;
    }
    worst = deviation(values[0], kExpectedMerged + size_t(kTextPositions) * kHidden, size_t(kAudioPositions) * kHidden);
    std::printf("  uncond %.3g\n", double(worst));
    return true;
}

// Self-consistency, no Python: a mask that hides the last key from every other
// query must reproduce, in those queries, a mask-free run over the prefix alone.
// If the mask tensor were ignored, the hidden position would leak in and the two
// would disagree.
bool check_mask_gates(Fixture & fixture, float & worst) {
    Context       graph_ctx = make_graph_context();
    ggml_cgraph * graph     = ggml_new_graph_custom(graph_ctx.get(), kNodeBudget, false);

    const synth::omnivoice::GeneratorLayerWeights & layer = fixture.weights.layers[0];
    const synth::omnivoice::AttentionShape          shape = make_shape();

    ggml_tensor * masked = synth::omnivoice::generator_layer(graph_ctx.get(), fixture.canvas, fixture.positions,
                                                             fixture.gate_mask, layer, shape);
    ggml_tensor * open =
        synth::omnivoice::generator_layer(graph_ctx.get(), fixture.prefix, fixture.prefix_pos, nullptr, layer, shape);
    if (masked == nullptr || open == nullptr || masked->ne[1] != int64_t(kPositions) ||
        open->ne[1] != int64_t(kGatedPrefix)) {
        return false;
    }

    std::vector<std::vector<float>> values;
    if (!compute(fixture, graph, { masked, open }, values)) {
        return false;
    }
    worst = deviation(std::vector<float>(values[0].begin(), values[0].begin() + size_t(kGatedPrefix) * kHidden),
                      values[1].data(), size_t(kGatedPrefix) * kHidden);
    std::printf("  mask-gating %.3g\n", double(worst));
    return true;
}

bool run_case(ggml_backend_dev_t device, float & max_diff) {
    Fixture fixture;
    if (!build_fixture(device, fixture)) {
        return false;
    }
    float forward = 0.0f;
    float uncond  = 0.0f;
    float gating  = 0.0f;
    if (!check_merge_and_forward(fixture, forward) || !check_uncond_embedding(fixture, uncond) ||
        !check_mask_gates(fixture, gating)) {
        return false;
    }
    // Gating is compared against this port's own output rather than against the
    // reference, so it is held to a much tighter bound and kept out of the
    // reference deviation the caller reports.
    if (!(gating < 1e-5f)) {
        std::printf("  mask-gating deviation %.3g exceeds 1e-5\n", double(gating));
        return false;
    }
    max_diff = std::fmax(forward, uncond);
    return true;
}

// A shape the builders cannot serve is a wiring defect, so they return nullptr
// rather than aborting inside ggml on an assertion the caller cannot catch.
int check_rejections() {
    // Roomy: the two rejected forwards each build most of a block's tensors
    // before the check that turns them down.
    Context        context = make_context(ggml_tensor_overhead() * 512 + 4096);
    ggml_context * ctx     = context.get();
    SYNTH_TEST_CHECK(ctx != nullptr);

    synth::omnivoice::GeneratorWeights weights;
    weights.text_embedding   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, kTextVocab);
    weights.audio_embeddings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, kAudioRows);
    weights.audio_heads      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, kAudioRows);
    weights.norm             = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHidden);
    synth::omnivoice::GeneratorLayerWeights layer;
    layer.input_layernorm          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHidden);
    layer.q_proj                   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, kHeads * kHeadDim);
    layer.k_proj                   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, kKvHeads * kHeadDim);
    layer.v_proj                   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, kKvHeads * kHeadDim);
    layer.o_proj                   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHeads * kHeadDim, kHidden);
    layer.q_norm                   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHeadDim);
    layer.k_norm                   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHeadDim);
    layer.post_attention_layernorm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHidden);
    layer.gate_proj                = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, kIntermediate);
    layer.up_proj                  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, kIntermediate);
    layer.down_proj                = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kIntermediate, kHidden);
    // The rejected forwards run the same depth the numeric checks do; sharing one
    // block's tensors across both slots is enough, since nothing here computes.
    weights.layers.assign(kLayers, layer);

    ggml_tensor * input     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, kPositions);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kPositions);
    ggml_tensor * text_ids  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kTextPositions);
    ggml_tensor * audio_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, kAudioPositions, kCodebooks);

    const synth::omnivoice::AttentionShape shape = make_shape();

    // The defect this family invites: Qwen2 has no q_norm, so a block copied
    // from a Qwen2 port would leave it null and rotate an unnormalized vector.
    synth::omnivoice::GeneratorLayerWeights no_q_norm = layer;
    no_q_norm.q_norm                                  = nullptr;
    SYNTH_TEST_CHECK(synth::omnivoice::generator_layer(ctx, input, positions, nullptr, no_q_norm, shape) == nullptr);

    // One position id per token, no more and no fewer.
    ggml_tensor * short_positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kPositions - 1);
    SYNTH_TEST_CHECK(synth::omnivoice::generator_layer(ctx, input, short_positions, nullptr, layer, shape) == nullptr);

    // Positions must be I32; an F32 vector would be read as garbage indices.
    ggml_tensor * float_positions = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kPositions);
    SYNTH_TEST_CHECK(synth::omnivoice::generator_layer(ctx, input, float_positions, nullptr, layer, shape) == nullptr);

    // A mask narrower than the key count would be applied to the wrong keys.
    ggml_tensor * short_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kPositions - 1, kPositions);
    SYNTH_TEST_CHECK(synth::omnivoice::generator_layer(ctx, input, positions, short_mask, layer, shape) == nullptr);

    // Grouped attention needs the head counts to divide.
    synth::omnivoice::AttentionShape ungrouped = shape;
    ungrouped.key_value_head_count             = 3;
    SYNTH_TEST_CHECK(synth::omnivoice::generator_layer(ctx, input, positions, nullptr, layer, ungrouped) == nullptr);

    // An audio grid whose row count is not the codebook count would have its
    // rows summed against the wrong table offsets.
    SYNTH_TEST_CHECK(synth::omnivoice::build_canvas_embedding(ctx, weights, text_ids, audio_ids, kCodebooks + 1) ==
                     nullptr);

    // Neither stream: there is no canvas to build.
    SYNTH_TEST_CHECK(synth::omnivoice::build_canvas_embedding(ctx, weights, nullptr, nullptr, kCodebooks) == nullptr);

    // Text ids must be one row; a grid here would index the table off its end.
    ggml_tensor * text_grid = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, kTextPositions, kCodebooks);
    SYNTH_TEST_CHECK(synth::omnivoice::build_canvas_embedding(ctx, weights, text_grid, nullptr, kCodebooks) == nullptr);

    // A head table whose width is not codebooks * vocab is a package/metadata
    // disagreement, and the reshape that names the row order would lie.
    synth::omnivoice::GeneratorWeights narrow_heads = weights;
    narrow_heads.audio_heads                        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, kAudioRows - 1);
    SYNTH_TEST_CHECK(synth::omnivoice::build_generator_forward(ctx, input, positions, nullptr, narrow_heads, shape,
                                                               make_canvas(), nullptr, nullptr) == nullptr);

    // No layers at all: a package that resolved zero blocks.
    synth::omnivoice::GeneratorWeights no_layers = weights;
    no_layers.layers.clear();
    SYNTH_TEST_CHECK(synth::omnivoice::build_generator_forward(ctx, input, positions, nullptr, no_layers, shape,
                                                               make_canvas(), nullptr, nullptr) == nullptr);
    return 0;
}

// The layer probes only have power if they disagree with each other. Were the
// two reference arrays ever re-dumped from a one-layer model, or the second one
// pasted over the first, the depth-2 chaining check would silently go back to
// proving nothing -- so the fixture asserts its own spread before using it.
int check_probes_are_distinct() {
    SYNTH_TEST_CHECK(std::size(kExpectedLayer0Output) == std::size(kExpectedLayer1Output));
    const float spread =
        deviation(std::vector<float>(std::begin(kExpectedLayer0Output), std::end(kExpectedLayer0Output)),
                  kExpectedLayer1Output, std::size(kExpectedLayer1Output));
    std::printf("omnivoice-generator: layer probes differ by %.3g\n", double(spread));
    SYNTH_TEST_CHECK(spread > 1e-2f);
    return 0;
}

}  // namespace

int main() {
    SYNTH_TEST_CHECK(check_rejections() == 0);
    SYNTH_TEST_CHECK(check_probes_are_distinct() == 0);

    // Every registered device. This block is the family's whole hot path, so a
    // backend that disagrees is worth catching here rather than in a golden.
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

        std::printf("omnivoice-generator: %s (device type %d)\n", ggml_backend_dev_name(device), int(type));
        float max_diff = 0.0f;
        SYNTH_TEST_CHECK(run_case(device, max_diff));

        // Accelerators run F32 matmuls through tensor cores at reduced mantissa
        // width, so they are held to a looser bound than the CPU.
        const float tolerance = type == GGML_BACKEND_DEVICE_TYPE_CPU ? 1e-4f : 5e-3f;
        std::printf("  max_diff %.3g (tolerance %.3g)\n", double(max_diff), double(tolerance));
        SYNTH_TEST_CHECK(max_diff < tolerance);
        ++exercised;
    }
    SYNTH_TEST_CHECK(exercised > 0);
    return 0;
}
