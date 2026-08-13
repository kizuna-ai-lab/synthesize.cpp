// The ECAPA-TDNN speaker encoder graph, mel in and x-vector out, against the
// reference implementation's own Qwen3TTSSpeakerEncoder.
//
// End to end on purpose. Almost nothing in this architecture is declared by the
// package: the three blocks' dilations, the order the res2net splits accumulate
// in, the residual around each block, the two activations in the pooling
// attention branch, the axis its softmax runs over, and the reflect padding
// every convolution uses are all read off the pinned upstream source. Each of
// them, guessed wrong, still produces a finite embedding of exactly the right
// width -- so a shape test cannot tell them apart and only a value can.
//
// Widths here are all distinct and all smaller than the checkpoint's, which
// makes a swapped extent visible rather than coincidentally right; the one
// production number kept is the res2net scale of eight, because the split
// arithmetic is what it governs. Nothing in the graph names a width, so a
// fixture at the checkpoint's own widths would test strictly less than this one.
//
// Reference values come from scripts/dump_reference_qwen3_tts_speaker_encoder.py.

#include "arch/qwen3-tts/speaker-encoder.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "test-assert.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <limits>
#include <memory>
#include <vector>

namespace {

constexpr int64_t  kMelBins           = 6;
constexpr int64_t  kChannels          = 16;
constexpr int64_t  kAggregated        = 3 * kChannels;
constexpr int64_t  kRes2NetScale      = 8;
constexpr int64_t  kSeChannels        = 5;
constexpr int64_t  kAttentionChannels = 7;
constexpr int64_t  kEncDim            = 11;
constexpr int64_t  kFrames            = 13;
constexpr uint64_t kSeed              = 20260812u;

constexpr float kWeightScale = 0.5f;
constexpr float kBiasScale   = 0.25f;

// 11 values, seed 20260812.
constexpr float kExpectedEmbedding[] = {
    0.638147116f, -1.46581924f, 3.81697369f,  0.879564404f,  -1.43383455f,  -3.29678035f,
    0.418415725f, 1.78037608f,  -2.17367673f, -0.987673998f, -0.543753147f,
};

// This topology builds TWO node counts, and which one you get is not a
// property of the topology: 435 with F32 weights, 473 with BF16. The 38-node
// difference is add_channel_bias's dtype-conditional ggml_cast
// (speaker-encoder.cpp), one per convolution, inserted only when the bias is
// not already F32. Nothing else about the graph moves -- not the mel bins,
// not the frame count, not any width.
//
// Both are pinned exactly, because a ceiling loose enough to hold both at
// once detects nothing: one redundant ggml_cont per convolution is itself 38
// nodes, exactly the gap between them. Task 4 pinned only the F32 number and
// called 450 "deliberately thin slack" -- but the graph that actually runs is
// the BF16 one (every one of the real Base package's 76 speaker_encoder
// tensors is BF16), and it exceeds 450. That was harmless only because
// speaker-encoder-host.cpp's own budget is 4096.
//
// The ceiling survives as the coarse guard the graph's capacity argument
// needs, now set seven above the LARGER count rather than fifteen above the
// smaller one. The exact pins below are what actually detect a change.
constexpr int    kNodesWithF32Weights  = 435;
constexpr int    kNodesWithBf16Weights = 473;
constexpr int    kNodeCeiling          = 480;
constexpr size_t kGraphCapacity        = 4096;

class LcgStream {
  public:
    explicit LcgStream(uint64_t seed) : state_(seed) {}

    float next() {
        state_ = state_ * 6364136223846793005ull + 1442695040888963407ull;
        return float(state_ >> 40) / 8388608.0f - 1.0f;
    }

    std::vector<float> fill(size_t count, float scale) {
        std::vector<float> values(count);
        for (float & value : values) {
            value = next() * scale;
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

Context make_context(size_t bytes) {
    ggml_init_params parameters{};
    parameters.mem_size = bytes;
    parameters.no_alloc = true;
    return Context(ggml_init(parameters));
}

// How the mel is filled, which is the only thing that varies between cases.
enum class Mel {
    Stream,     // drawn from the weight stream, as the reference script does
    Alternate,  // drawn from a second stream, for a different speaker
    Constant,   // every frame identical, which drives the pooled variance to zero
};

struct Shape {
    int64_t mel_bins = kMelBins;
    int64_t frames   = kFrames;
    int64_t scale    = kRes2NetScale;
    Mel     mel      = Mel::Stream;
};

// Creates every tensor the encoder resolves, in the order the reference script
// draws them. `ordered`/`scales`, when non-null, collect them for filling; a
// caller that only needs shapes passes null and skips the backend entirely.
// `weight_type` is F32 for every case that computes -- the reference
// embedding is an F32 measurement -- and BF16 only for the node count, where
// nothing is filled or run. It is the package's own storage dtype, and the
// real Base package stores all 76 of these tensors in BF16.
void add_weights(ggml_context *                           context,
                 const Shape &                            shape,
                 synth::qwen3tts::SpeakerEncoderWeights & weights,
                 std::vector<ggml_tensor *> *             ordered,
                 std::vector<float> *                     scales,
                 ggml_type                                weight_type = GGML_TYPE_F32) {
    auto add = [&](ggml_tensor * tensor, float scale) {
        if (ordered != nullptr) {
            ordered->push_back(tensor);
            scales->push_back(scale);
        }
        return tensor;
    };
    auto add_conv = [&](synth::qwen3tts::Conv1dWeights & target, int64_t kernel, int64_t in, int64_t out) {
        // ggml reports a Conv1d kernel stored as [out, in, kernel] in reverse.
        target.weight = add(ggml_new_tensor_3d(context, weight_type, kernel, in, out), kWeightScale);
        target.bias   = add(ggml_new_tensor_1d(context, weight_type, out), kBiasScale);
    };

    add_conv(weights.stem, 5, shape.mel_bins, kChannels);
    weights.blocks.assign(3, synth::qwen3tts::SpeakerEncoderBlockWeights{});
    for (synth::qwen3tts::SpeakerEncoderBlockWeights & block : weights.blocks) {
        const int64_t width = kChannels / shape.scale;
        add_conv(block.tdnn1, 1, kChannels, kChannels);
        block.res2net.assign(size_t(shape.scale - 1), synth::qwen3tts::Conv1dWeights{});
        for (synth::qwen3tts::Conv1dWeights & split : block.res2net) {
            add_conv(split, 3, width, width);
        }
        add_conv(block.se1, 1, kChannels, kSeChannels);
        add_conv(block.se2, 1, kSeChannels, kChannels);
        add_conv(block.tdnn2, 1, kChannels, kChannels);
    }
    add_conv(weights.mfa, 1, kAggregated, kAggregated);
    add_conv(weights.asp_tdnn, 1, 3 * kAggregated, kAttentionChannels);
    add_conv(weights.asp, 1, kAttentionChannels, kAggregated);
    add_conv(weights.fc, 1, 2 * kAggregated, kEncDim);
}

struct Fixture {
    ggml_backend_t                         backend = nullptr;
    Context                                persistent;
    ggml_backend_buffer_t                  buffer = nullptr;
    synth::qwen3tts::SpeakerEncoderWeights weights;
    ggml_tensor *                          mel = nullptr;

    ~Fixture() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
};

bool build_fixture(ggml_backend_dev_t device, const Shape & shape, Fixture & fixture) {
    fixture.backend = ggml_backend_dev_init(device, nullptr);
    if (fixture.backend == nullptr) {
        return false;
    }
    fixture.persistent  = make_context(ggml_tensor_overhead() * 256);
    ggml_context * pctx = fixture.persistent.get();

    std::vector<ggml_tensor *> ordered;
    std::vector<float>         scales;
    add_weights(pctx, shape, fixture.weights, &ordered, &scales);

    fixture.mel    = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, shape.mel_bins, shape.frames);
    fixture.buffer = ggml_backend_alloc_ctx_tensors(pctx, fixture.backend);
    if (fixture.buffer == nullptr) {
        return false;
    }

    LcgStream stream(kSeed);
    for (size_t index = 0; index < ordered.size(); ++index) {
        const std::vector<float> values = stream.fill(size_t(ggml_nelements(ordered[index])), scales[index]);
        ggml_backend_tensor_set(ordered[index], values.data(), 0, ggml_nbytes(ordered[index]));
    }

    const size_t       elements = size_t(shape.mel_bins) * size_t(shape.frames);
    std::vector<float> mel;
    switch (shape.mel) {
        case Mel::Stream:
            mel = stream.fill(elements, 1.0f);
            break;
        case Mel::Alternate:
            mel = LcgStream(kSeed + 1).fill(elements, 1.0f);
            break;
        case Mel::Constant:
            {
                // One frame's worth, repeated. ggml's [bins, frames] puts the bins of
                // one frame together, so a frame is a contiguous run.
                LcgStream                frame_stream(kSeed + 2);
                const std::vector<float> frame = frame_stream.fill(size_t(shape.mel_bins), 1.0f);
                for (int64_t index = 0; index < shape.frames; ++index) {
                    mel.insert(mel.end(), frame.begin(), frame.end());
                }
                break;
            }
    }
    ggml_backend_tensor_set(fixture.mel, mel.data(), 0, ggml_nbytes(fixture.mel));
    return true;
}

// Builds and runs the graph, returning the embedding. `nodes` reports the graph
// size so a caller can pin it.
bool run_case(ggml_backend_dev_t device, const Shape & shape, std::vector<float> & output, int & nodes) {
    Fixture fixture;
    if (!build_fixture(device, shape, fixture)) {
        return false;
    }

    Context       graph_ctx = make_context(ggml_tensor_overhead() * (kGraphCapacity + 64) +
                                           ggml_graph_overhead_custom(kGraphCapacity, false));
    ggml_cgraph * graph     = ggml_new_graph_custom(graph_ctx.get(), kGraphCapacity, false);

    ggml_tensor * embedding = synth::qwen3tts::build_speaker_encoder(graph_ctx.get(), fixture.weights, fixture.mel);
    if (embedding == nullptr) {
        std::printf("    build_speaker_encoder returned nullptr\n");
        return false;
    }
    // The embedding is one vector, not a one-column matrix: the prompt slot the
    // talker fills reads it as a plain run of enc_dim floats.
    if (ggml_n_dims(embedding) != 1 || embedding->ne[0] != kEncDim) {
        std::printf("    unexpected embedding shape [%lld, %lld]\n", (long long) embedding->ne[0],
                    (long long) embedding->ne[1]);
        return false;
    }

    ggml_build_forward_expand(graph, embedding);
    nodes                    = ggml_graph_n_nodes(graph);
    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(fixture.backend));
    bool           ok        = allocator != nullptr && ggml_gallocr_alloc_graph(allocator, graph);
    if (ok) {
        ok = ggml_backend_graph_compute(fixture.backend, graph) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        output.assign(size_t(ggml_nelements(embedding)), 0.0f);
        ggml_backend_tensor_get(embedding, output.data(), 0, ggml_nbytes(embedding));
    }
    if (allocator != nullptr) {
        ggml_gallocr_free(allocator);
    }
    return ok;
}

bool all_finite(const std::vector<float> & values) {
    for (float value : values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return !values.empty();
}

float max_difference(const std::vector<float> & left, const std::vector<float> & right) {
    if (left.size() != right.size()) {
        return std::numeric_limits<float>::infinity();
    }
    float worst = 0.0f;
    for (size_t index = 0; index < left.size(); ++index) {
        worst = std::fmax(worst, std::fabs(left[index] - right[index]));
    }
    return worst;
}

// The reference values, and with them every convention the package does not
// declare.
int check_against_reference(ggml_backend_dev_t device, float tolerance, int & nodes) {
    std::vector<float> embedding;
    SYNTH_TEST_CHECK(run_case(device, Shape{}, embedding, nodes));
    SYNTH_TEST_CHECK(embedding.size() == std::size(kExpectedEmbedding));
    SYNTH_TEST_CHECK(all_finite(embedding));

    const std::vector<float> expected(std::begin(kExpectedEmbedding), std::end(kExpectedEmbedding));
    const float              worst = max_difference(embedding, expected);
    std::printf("    reference max_diff %.3g, nodes %d\n", double(worst), nodes);
    SYNTH_TEST_CHECK(worst < tolerance);
    return 0;
}

// Attentive statistics pooling is the whole point of the architecture: the
// embedding must depend on what is in the clip, not merely on its shape.
int check_different_mels_differ(ggml_backend_dev_t device) {
    int                nodes = 0;
    std::vector<float> first;
    std::vector<float> second;
    Shape              alternate;
    alternate.mel = Mel::Alternate;
    SYNTH_TEST_CHECK(run_case(device, Shape{}, first, nodes));
    SYNTH_TEST_CHECK(run_case(device, alternate, second, nodes));
    SYNTH_TEST_CHECK(first.size() == second.size());
    SYNTH_TEST_CHECK(max_difference(first, second) > 1e-3f);
    return 0;
}

// A mel that never changes over time pins two things at once that no other case
// here can reach.
//
// The pooled variance over such a clip is exactly zero, which is the input that
// turns an unfloored sqrt into a NaN -- and into a NaN only on this input, which
// is why nothing else notices.
//
// And every stage stays constant over time only because the padding is
// reflective: zero padding would make the edge frames differ from the interior,
// so a longer clip would carry a different proportion of edges and pool to a
// different embedding. Doubling the length and demanding the same answer is
// therefore a test of the padding mode as much as of the pooling.
int check_constant_mel_is_finite_and_length_invariant(ggml_backend_dev_t device, float tolerance) {
    int   nodes = 0;
    Shape shorter;
    shorter.mel   = Mel::Constant;
    Shape longer  = shorter;
    longer.frames = 2 * kFrames;

    std::vector<float> first;
    std::vector<float> second;
    SYNTH_TEST_CHECK(run_case(device, shorter, first, nodes));
    SYNTH_TEST_CHECK(run_case(device, longer, second, nodes));
    SYNTH_TEST_CHECK(all_finite(first));
    SYNTH_TEST_CHECK(all_finite(second));
    const float worst = max_difference(first, second);
    std::printf("    constant-mel length invariance max_diff %.3g\n", double(worst));
    SYNTH_TEST_CHECK(worst < tolerance);
    return 0;
}

// The res2net scale is `res2net.size() + 1`, read off the weights rather than
// assumed, so a package with a different split count builds a different graph
// instead of a wrong one.
int check_scale_comes_from_the_weights(ggml_backend_dev_t device) {
    int                nodes = 0;
    std::vector<float> narrow;
    Shape              four;
    four.scale = 4;
    SYNTH_TEST_CHECK(run_case(device, four, narrow, nodes));
    SYNTH_TEST_CHECK(all_finite(narrow));

    std::vector<float> eight;
    int                eight_nodes = 0;
    SYNTH_TEST_CHECK(run_case(device, Shape{}, eight, eight_nodes));
    // Four splits is a genuinely different graph, not the same one relabelled.
    SYNTH_TEST_CHECK(nodes < eight_nodes);
    SYNTH_TEST_CHECK(max_difference(narrow, eight) > 1e-3f);
    return 0;
}

int check_node_ceiling(int nodes) {
    SYNTH_TEST_CHECK(nodes > 0);
    SYNTH_TEST_CHECK(nodes < kNodeCeiling);
    return 0;
}

// Builds the graph and counts its nodes without a backend, an allocation or a
// forward pass: the count is a property of construction, and the BF16 case
// below has no values to compute with anyway.
int count_nodes(ggml_type weight_type) {
    Context context = make_context(ggml_tensor_overhead() * (kGraphCapacity + 512) +
                                   ggml_graph_overhead_custom(kGraphCapacity, false));
    if (context == nullptr) {
        return -1;
    }
    ggml_context * ctx = context.get();

    synth::qwen3tts::SpeakerEncoderWeights weights;
    add_weights(ctx, Shape{}, weights, nullptr, nullptr, weight_type);
    ggml_tensor * mel       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kMelBins, kFrames);
    ggml_tensor * embedding = synth::qwen3tts::build_speaker_encoder(ctx, weights, mel);
    if (embedding == nullptr) {
        return -1;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, kGraphCapacity, false);
    ggml_build_forward_expand(graph, embedding);
    return ggml_graph_n_nodes(graph);
}

// Both counts, exactly, and the reason they differ measured rather than
// asserted: swapping only the weights' storage dtype must move the count by
// exactly one node per convolution and by nothing else.
int check_both_node_counts() {
    const int f32_nodes  = count_nodes(GGML_TYPE_F32);
    const int bf16_nodes = count_nodes(GGML_TYPE_BF16);
    std::printf("    nodes: F32 weights %d, BF16 weights %d\n", f32_nodes, bf16_nodes);
    SYNTH_TEST_CHECK(f32_nodes == kNodesWithF32Weights);
    SYNTH_TEST_CHECK(bf16_nodes == kNodesWithBf16Weights);
    // One ggml_cast per convolution, and this graph has 38 of them -- the same
    // 16 kernel-1 plus 22 im2col split speaker-encoder.cpp's header counts.
    SYNTH_TEST_CHECK(bf16_nodes - f32_nodes == 38);
    SYNTH_TEST_CHECK(bf16_nodes < kNodeCeiling);
    return 0;
}

int check_rejections() {
    // Room for several whole graphs: a rejection is only interesting beside the
    // accepted case it is compared against, and each accepted build is a graph.
    Context        context = make_context(ggml_tensor_overhead() * 4096);
    ggml_context * ctx     = context.get();

    synth::qwen3tts::SpeakerEncoderWeights weights;
    add_weights(ctx, Shape{}, weights, nullptr, nullptr);
    ggml_tensor * mel = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kMelBins, kFrames);

    SYNTH_TEST_CHECK(synth::qwen3tts::build_speaker_encoder(nullptr, weights, mel) == nullptr);
    SYNTH_TEST_CHECK(synth::qwen3tts::build_speaker_encoder(ctx, weights, nullptr) == nullptr);
    // The happy path has to build, or every rejection below proves nothing.
    SYNTH_TEST_CHECK(synth::qwen3tts::build_speaker_encoder(ctx, weights, mel) != nullptr);

    // A mel whose bin count disagrees with the stem's declared input extent
    // would convolve garbage rather than fail.
    ggml_tensor * wide_mel = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kMelBins + 1, kFrames);
    SYNTH_TEST_CHECK(synth::qwen3tts::build_speaker_encoder(ctx, weights, wide_mel) == nullptr);

    // Reflect padding cannot reach past the signal, and ggml asserts rather than
    // returning on it. The boundary is exact: the deepest reflection is four
    // frames, so four is refused and five is not.
    ggml_tensor * at_limit =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kMelBins, synth::qwen3tts::kSpeakerEncoderDeepestReflection);
    ggml_tensor * past_limit =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kMelBins, synth::qwen3tts::kSpeakerEncoderDeepestReflection + 1);
    SYNTH_TEST_CHECK(synth::qwen3tts::build_speaker_encoder(ctx, weights, at_limit) == nullptr);
    SYNTH_TEST_CHECK(synth::qwen3tts::build_speaker_encoder(ctx, weights, past_limit) != nullptr);

    // An integer mel reaches ggml_pad_reflect_1d's F32 assertion, and a batched
    // one would be convolved as though the batch were more frames.
    ggml_tensor * integer_mel = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, kMelBins, kFrames);
    ggml_tensor * batched_mel = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kMelBins, kFrames, 2);
    SYNTH_TEST_CHECK(synth::qwen3tts::build_speaker_encoder(ctx, weights, integer_mel) == nullptr);
    SYNTH_TEST_CHECK(synth::qwen3tts::build_speaker_encoder(ctx, weights, batched_mel) == nullptr);

    // A working width that the split count does not divide has no res2net
    // splitting; dropping one convolution turns scale 8 into scale 7, which 16
    // channels do not divide by.
    synth::qwen3tts::SpeakerEncoderWeights indivisible = weights;
    indivisible.blocks[1].res2net.pop_back();
    SYNTH_TEST_CHECK(synth::qwen3tts::build_speaker_encoder(ctx, indivisible, mel) == nullptr);

    // A resolver that dropped one tensor of the seventy-six.
    synth::qwen3tts::SpeakerEncoderWeights incomplete = weights;
    incomplete.blocks[1].res2net[3].weight            = nullptr;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_speaker_encoder(ctx, incomplete, mel) == nullptr);

    synth::qwen3tts::SpeakerEncoderWeights biasless = weights;
    biasless.asp_tdnn.bias                          = nullptr;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_speaker_encoder(ctx, biasless, mel) == nullptr);

    // A default-constructed struct, which is what a CustomVoice package leaves
    // behind: no blocks at all rather than empty ones.
    SYNTH_TEST_CHECK(synth::qwen3tts::build_speaker_encoder(ctx, synth::qwen3tts::SpeakerEncoderWeights{}, mel) ==
                     nullptr);
    return 0;
}

}  // namespace

int main() {
    SYNTH_TEST_CHECK(check_rejections() == 0);
    // Construction only, so it runs once rather than per device.
    SYNTH_TEST_CHECK(check_both_node_counts() == 0);

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
        std::printf("qwen3-tts-speaker-encoder: %s (device type %d)\n", ggml_backend_dev_name(device), int(type));

        // The accelerator bound is looser on purpose, for the reason the codec
        // test records: CUDA runs F32 matmuls through tensor cores at reduced
        // mantissa width, and this graph stacks thirty-eight convolutions, an
        // exponential and two square roots. A wiring fault does not hide under
        // it -- the reference values are not nearly this close to each other.
        const float tolerance = type == GGML_BACKEND_DEVICE_TYPE_CPU ? 1e-4f : 2e-2f;

        int nodes = 0;
        SYNTH_TEST_CHECK(check_against_reference(device, tolerance, nodes) == 0);
        SYNTH_TEST_CHECK(check_node_ceiling(nodes) == 0);
        SYNTH_TEST_CHECK(check_different_mels_differ(device) == 0);
        SYNTH_TEST_CHECK(check_constant_mel_is_finite_and_length_invariant(device, tolerance) == 0);
        SYNTH_TEST_CHECK(check_scale_comes_from_the_weights(device) == 0);
        ++exercised;
    }
    SYNTH_TEST_CHECK(exercised > 0);
    return 0;
}
