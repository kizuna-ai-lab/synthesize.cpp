// The codec encoder's graph: waveform in, pre-quantization latents out.
//
// Six rules, each with an input that fails it. None of them is a reference
// value: the encoder's numerical agreement with upstream is measured against
// Task 1's stage-wise oracle dump, which needs the real 2.5 GB Base package and
// so cannot live at the `unit` tier. What lives here is everything that can be
// wrong about the graph's SHAPE and its CAUSALITY -- which is most of what a
// port of this gets wrong, and all of what a tolerance grid cannot see.
//
// The two production-geometry checks (frame count, and that the strides are
// what produce it) run on weights carrying only kernel extents, because a
// geometry walk reads ne[0] and nothing else. The graph checks run on a small
// synthetic stack drawn from one 64-bit LCG, which is what keeps them cheap
// enough to be a unit test at all: the real transformer is 8 x 512 x 2048.
//
// The causality check is the one nothing downstream can replace. A symmetric
// MimiConv1d pad -- upstream's own non-causal branch, one `if` away from the
// one this checkpoint takes -- builds the same shapes from the same weights and
// a different encoder, and produces finite latents that decode to plausible
// audio.

#include "arch/qwen3-tts/codec-encoder.h"
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

// The real Base checkpoint's encode path. `upsampling_ratios` is [8, 6, 5, 4]
// and the encoder walks it REVERSED (modeling_mimi.py:456), so its strides are
// 4, 5, 6, 8 and its kernels are twice those. The frame downsampler adds one
// more stride of 2, for 4 * 5 * 6 * 8 * 2 = 1920 samples per frame.
constexpr int64_t kProductionStrideKernels[] = { 8, 10, 12, 16 };
constexpr int64_t kProductionSamplesPerFrame = 1920;

// A stack with the same structure at a fraction of the width, so the graph can
// actually be run. Strides 2, 3, 2, 2 and a downsampler stride of 2: 48 samples
// per frame.
constexpr int64_t  kStem            = 4;
constexpr int64_t  kStageWidths[]   = { 8, 16, 32, 64 };
constexpr int64_t  kStageKernels[]  = { 4, 6, 4, 4 };
constexpr int64_t  kHidden          = 32;
constexpr int64_t  kIntermediate    = 128;
constexpr int64_t  kLayers          = 8;
constexpr int64_t  kSamplesPerFrame = 48;
constexpr size_t   kNodeBudget      = 8192;
constexpr uint64_t kSeed            = 20260813u;

// The graph this fixture builds, exactly. Pinned because a budget of 8192
// against 445 catches nothing: a stray ggml_cont per layer, a convolution that
// grew a copy, or a mask that stopped being inplace all fit inside it
// unnoticed. tests/qwen3_tts_speaker_encoder_test.cpp:416-426 pins its own
// counts the same way.
//
// TWO counts, because the graph is NOT length-independent, which is worth
// stating precisely since an earlier revision of this comment claimed it was.
// Thirteen convolutions and eight transformer layers do not change shape with
// the clip -- but the frame downsampler REPLICATE-pads, and its right-hand
// `extra_padding` is zero exactly when the clip is a whole number of frames.
// When it is not, that pad is built out of a view, a cont, a repeat and a
// concat: four more nodes. Measured, not derived. The real package shows the
// same split: 445 nodes on base-icl-en's 193,920 samples (101 whole frames) and
// 449 on base-ref-min's 24,000 (12.5 frames).
//
// Both are F32 counts. `as_f32` inserts a ggml_cast per elementwise weight that
// is not already F32, so a package storing this half at lower precision would
// build a larger graph -- the F32/BF16 split the speaker encoder's test pins.
// All 161 `codec.encoder.*` tensors in the real Base package are F32 (measured
// with a GGUF read), so these are the real package's counts too.
constexpr int kNodesWholeFrames = 445;
constexpr int kNodesRaggedTail  = 449;

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

Context make_context(size_t bytes) {
    ggml_init_params parameters{};
    parameters.mem_size = bytes;
    parameters.no_alloc = true;
    return Context(ggml_init(parameters));
}

// Weights carrying kernel extents and nothing else. codec_encoder_geometry
// reads ne[0] of each convolution and the channel widths never enter the
// arithmetic, so a one-channel stack drives the production geometry exactly.
void build_geometry_weights(ggml_context *                         context,
                            const int64_t                          stride_kernels[4],
                            synth::qwen3tts::CodecEncoderWeights & weights) {
    auto conv = [context](synth::qwen3tts::Conv1dWeights & target, int64_t kernel) {
        target.weight = ggml_new_tensor_3d(context, GGML_TYPE_F32, kernel, 1, 1);
        target.bias   = ggml_new_tensor_1d(context, GGML_TYPE_F32, 1);
    };
    conv(weights.stem, 7);
    weights.stages.assign(4, synth::qwen3tts::CodecEncoderStage{});
    for (size_t stage = 0; stage < weights.stages.size(); ++stage) {
        conv(weights.stages[stage].bottleneck_in, 3);
        conv(weights.stages[stage].bottleneck_out, 1);
        conv(weights.stages[stage].stride_conv, stride_kernels[stage]);
    }
    conv(weights.tail, 3);
    weights.downsample = ggml_new_tensor_3d(context, GGML_TYPE_F32, 4, 1, 1);
}

struct Fixture {
    ggml_backend_t                       backend = nullptr;
    Context                              persistent;
    ggml_backend_buffer_t                buffer = nullptr;
    synth::qwen3tts::CodecEncoderWeights weights;

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
    fixture.persistent  = make_context(ggml_tensor_overhead() * 512);
    ggml_context * pctx = fixture.persistent.get();

    std::vector<ggml_tensor *> ordered;
    std::vector<float>         scales;
    auto                       add = [&](ggml_tensor * tensor, float scale) {
        ordered.push_back(tensor);
        scales.push_back(scale);
        return tensor;
    };
    auto add_conv = [&](synth::qwen3tts::Conv1dWeights & target, int64_t kernel, int64_t in, int64_t out) {
        // Scaled down by the fan-in so an eleven-convolution stack neither
        // saturates nor decays to nothing before the transformer sees it.
        const float scale = 2.0f / float(std::sqrt(double(kernel * in)));
        target.weight     = add(ggml_new_tensor_3d(pctx, GGML_TYPE_F32, kernel, in, out), scale);
        target.bias       = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, out), 0.05f);
    };

    add_conv(fixture.weights.stem, 7, 1, kStem);
    fixture.weights.stages.assign(4, synth::qwen3tts::CodecEncoderStage{});
    int64_t width = kStem;
    for (size_t stage = 0; stage < fixture.weights.stages.size(); ++stage) {
        synth::qwen3tts::CodecEncoderStage & into = fixture.weights.stages[stage];
        add_conv(into.bottleneck_in, 3, width, width / 2);
        add_conv(into.bottleneck_out, 1, width / 2, width);
        add_conv(into.stride_conv, kStageKernels[stage], width, kStageWidths[stage]);
        width = kStageWidths[stage];
    }
    add_conv(fixture.weights.tail, 3, width, kHidden);
    // No bias: the frame downsampler is the one convolution here built with
    // `bias=False`, which is why the catalog holds it as a bare tensor.
    fixture.weights.downsample = add(ggml_new_tensor_3d(pctx, GGML_TYPE_F32, 4, kHidden, kHidden), 0.15f);

    fixture.weights.layers.assign(kLayers, synth::qwen3tts::CodecEncoderTransformerLayerWeights{});
    for (synth::qwen3tts::CodecEncoderTransformerLayerWeights & layer : fixture.weights.layers) {
        // LayerNorm carries a weight AND a bias here, unlike the decoder's
        // RMSNorm. The weight is drawn about 1 and the bias about 0, which is
        // what a trained LayerNorm looks like.
        layer.input_layernorm.weight          = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kHidden), 0.0f);
        layer.input_layernorm.bias            = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kHidden), 0.05f);
        layer.q_proj                          = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kHidden), 0.2f);
        layer.k_proj                          = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kHidden), 0.2f);
        layer.v_proj                          = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kHidden), 0.2f);
        layer.o_proj                          = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kHidden), 0.2f);
        // MimiLayerScale initialises at 0.01 and stays small after training.
        layer.self_attn_layer_scale           = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kHidden), 0.02f);
        layer.post_attention_layernorm.weight = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kHidden), 0.0f);
        layer.post_attention_layernorm.bias   = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kHidden), 0.05f);
        layer.fc1             = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kIntermediate), 0.2f);
        layer.fc2             = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kIntermediate, kHidden), 0.1f);
        layer.mlp_layer_scale = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kHidden), 0.02f);
    }

    fixture.buffer = ggml_backend_alloc_ctx_tensors(pctx, fixture.backend);
    if (fixture.buffer == nullptr) {
        return false;
    }

    LcgStream stream(kSeed);
    for (size_t index = 0; index < ordered.size(); ++index) {
        // A LayerNorm gain is drawn about one; everything else about zero. The
        // two norm gains are the only weights with a non-zero offset, and they
        // are the ones whose scale entry is 0.
        const float              scale  = scales[index] == 0.0f ? 0.1f : scales[index];
        const float              offset = scales[index] == 0.0f ? 1.0f : 0.0f;
        const std::vector<float> values = stream.fill(size_t(ggml_nelements(ordered[index])), scale, offset);
        ggml_backend_tensor_set(ordered[index], values.data(), 0, ggml_nbytes(ordered[index]));
    }
    return true;
}

// The clip a run_encoder call encodes: `samples` LCG samples, with `bump` added
// to sample 0. Same stream at every length, so a longer clip is the shorter one
// with samples appended and nothing else changed.
std::vector<float> lcg_clip(int64_t samples, float bump) {
    LcgStream          stream(kSeed ^ 0x9e3779b97f4a7c15ull);
    std::vector<float> pcm = stream.fill(size_t(samples), 0.5f, 0.0f);
    pcm[0] += bump;
    return pcm;
}

// Runs the encoder over `samples` LCG samples and returns the latents.
//
// `seanet_tail_out`, when non-null, additionally reads back the seanet_tail tap
// -- which is what proves the tap is readable at all, since it is an
// intermediate the graph allocator would otherwise be free to recycle.
bool run_encoder(const Fixture &      fixture,
                 int64_t              samples,
                 float                first_sample_bump,
                 std::vector<float> & latents,
                 int64_t &            frames,
                 int &                nodes,
                 std::vector<float> * seanet_tail_out = nullptr) {
    synth::qwen3tts::CodecEncoderGeometry geometry;
    if (!synth::qwen3tts::codec_encoder_geometry(fixture.weights, samples, geometry)) {
        return false;
    }

    Context       input_ctx = make_context(ggml_tensor_overhead() * 8);
    // Channel-major: one channel of `samples`, the same shape the decoder hands
    // back at the other end of the codec.
    ggml_tensor * waveform  = ggml_new_tensor_2d(input_ctx.get(), GGML_TYPE_F32, 1, samples);
    ggml_tensor * positions = ggml_new_tensor_1d(input_ctx.get(), GGML_TYPE_I32, geometry.transformer_positions);
    ggml_backend_buffer_t input_buffer = ggml_backend_alloc_ctx_tensors(input_ctx.get(), fixture.backend);
    if (input_buffer == nullptr) {
        return false;
    }

    const std::vector<float> pcm = lcg_clip(samples, first_sample_bump);
    ggml_backend_tensor_set(waveform, pcm.data(), 0, ggml_nbytes(waveform));

    std::vector<int32_t> sequential(size_t(geometry.transformer_positions));
    for (int64_t index = 0; index < geometry.transformer_positions; ++index) {
        sequential[size_t(index)] = int32_t(index);
    }
    ggml_backend_tensor_set(positions, sequential.data(), 0, ggml_nbytes(positions));

    Context graph_ctx =
        make_context(ggml_tensor_overhead() * (kNodeBudget + 64) + ggml_graph_overhead_custom(kNodeBudget, false));
    ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx.get(), kNodeBudget, false);

    synth::qwen3tts::CodecEncoderTaps taps;
    // Deliberately pre-loaded with junk: the builder must CLEAR its own lists
    // rather than append to them, or a reused taps struct silently carries the
    // previous graph's pointers.
    taps.seanet_stages.assign(3, nullptr);
    taps.transformer_layers.assign(5, nullptr);
    taps.downsample      = waveform;
    taps.seanet_tail     = waveform;
    ggml_tensor * result = synth::qwen3tts::build_codec_encoder(graph_ctx.get(), waveform, positions, fixture.weights,
                                                                seanet_tail_out == nullptr ? nullptr : &taps);
    bool          ok     = result != nullptr;
    if (ok) {
        // Rule 3: the latents are as wide as the tail convolution's output,
        // which catalog.cpp binds to the package's `codebook_dim` (512 in the
        // real Base package, checked by the catalog test).
        ok = result->ne[0] == kHidden && result->ne[1] == geometry.frames;
        if (!ok) {
            std::printf("    unexpected latent shape [%lld, %lld], wanted [%lld, %lld]\n", (long long) result->ne[0],
                        (long long) result->ne[1], (long long) kHidden, (long long) geometry.frames);
        }
    }
    if (ok && seanet_tail_out != nullptr) {
        ok = taps.seanet_stages.size() == std::size(kStageWidths) &&
             taps.transformer_layers.size() == size_t(kLayers) && taps.seanet_tail != nullptr &&
             taps.downsample == result;
        if (!ok) {
            std::printf("    taps not filled as declared: %zu stages, %zu layers\n", taps.seanet_stages.size(),
                        taps.transformer_layers.size());
        }
    }
    ggml_gallocr_t allocator = nullptr;
    if (ok) {
        ggml_build_forward_expand(graph, result);
        nodes     = ggml_graph_n_nodes(graph);
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(fixture.backend));
        ok        = allocator != nullptr && ggml_gallocr_alloc_graph(allocator, graph);
    }
    if (ok) {
        ok = ggml_backend_graph_compute(fixture.backend, graph) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        latents.assign(size_t(ggml_nelements(result)), 0.0f);
        ggml_backend_tensor_get(result, latents.data(), 0, ggml_nbytes(result));
        frames = geometry.frames;
        if (seanet_tail_out != nullptr) {
            seanet_tail_out->assign(size_t(ggml_nelements(taps.seanet_tail)), 0.0f);
            ggml_backend_tensor_get(taps.seanet_tail, seanet_tail_out->data(), 0, ggml_nbytes(taps.seanet_tail));
        }
    }
    if (allocator != nullptr) {
        ggml_gallocr_free(allocator);
    }
    ggml_backend_buffer_free(input_buffer);
    return ok;
}

// Rule 1: the frame count is ceil(samples / 1920) at the production geometry.
// Rule 4: a clip that does not fill one frame is refused.
int check_production_geometry() {
    Context                              context = make_context(ggml_tensor_overhead() * 64);
    synth::qwen3tts::CodecEncoderWeights weights;
    build_geometry_weights(context.get(), kProductionStrideKernels, weights);

    struct Case {
        int64_t samples;
        int64_t frames;
    };

    // 24000 is the one that matters: a floor divide gives 12 where the oracle
    // measured 13, and it is the single most likely off-by-one on this path.
    // The other two divide exactly and cannot tell the two apart.
    const Case cases[] = {
        { 193920, 101 },
        { 24000,  13  },
        { 720000, 375 }
    };
    for (const Case & one : cases) {
        synth::qwen3tts::CodecEncoderGeometry geometry;
        SYNTH_TEST_CHECK(synth::qwen3tts::codec_encoder_geometry(weights, one.samples, geometry));
        SYNTH_TEST_CHECK(geometry.samples_per_frame == kProductionSamplesPerFrame);
        SYNTH_TEST_CHECK(geometry.frames == one.frames);
        // The transformer runs before the frame downsampler, so it is twice as
        // long as the frame count -- give or take the downsampler's own
        // ceiling. A port that sized it from `frames` would rope every position
        // wrong.
        SYNTH_TEST_CHECK(geometry.transformer_positions == 2 * one.frames ||
                         geometry.transformer_positions == 2 * one.frames - 1);
    }

    // Rule 4. 1920 samples is exactly one frame and is accepted; one sample
    // fewer is refused rather than padded up to a fabricated frame.
    synth::qwen3tts::CodecEncoderGeometry geometry;
    SYNTH_TEST_CHECK(synth::qwen3tts::codec_encoder_geometry(weights, 1920, geometry));
    SYNTH_TEST_CHECK(geometry.frames == 1);
    SYNTH_TEST_CHECK(!synth::qwen3tts::codec_encoder_geometry(weights, 1919, geometry));
    SYNTH_TEST_CHECK(!synth::qwen3tts::codec_encoder_geometry(weights, 1, geometry));
    SYNTH_TEST_CHECK(!synth::qwen3tts::codec_encoder_geometry(weights, 0, geometry));

    // Unbound weights are refused rather than walked.
    synth::qwen3tts::CodecEncoderWeights empty;
    SYNTH_TEST_CHECK(!synth::qwen3tts::codec_encoder_geometry(empty, 193920, geometry));
    return 0;
}

// Rule 2: a different stride set produces a different frame count. Without
// this, 1920 could be right by coincidence -- a walk that ignored the strides
// entirely and divided by a constant would pass rule 1 and fail nothing.
int check_strides_are_read() {
    Context context = make_context(ggml_tensor_overhead() * 128);

    // Strides 4, 5, 6, 7 rather than 4, 5, 6, 8.
    const int64_t                        altered[4] = { 8, 10, 12, 14 };
    synth::qwen3tts::CodecEncoderWeights weights;
    build_geometry_weights(context.get(), altered, weights);

    synth::qwen3tts::CodecEncoderGeometry geometry;
    SYNTH_TEST_CHECK(synth::qwen3tts::codec_encoder_geometry(weights, 193920, geometry));
    SYNTH_TEST_CHECK(geometry.samples_per_frame == 4 * 5 * 6 * 7 * 2);
    SYNTH_TEST_CHECK(geometry.frames != 101);

    // And reversing the order changes it too, which is the mistake the
    // `reversed(upsampling_ratios)` reading exists to prevent: the total
    // downsampling is the same 1920 either way, so only a stage-wise walk can
    // tell them apart.
    const int64_t                        reversed[4] = { 16, 12, 10, 8 };
    synth::qwen3tts::CodecEncoderWeights backwards;
    build_geometry_weights(context.get(), reversed, backwards);
    synth::qwen3tts::CodecEncoderGeometry other;
    SYNTH_TEST_CHECK(synth::qwen3tts::codec_encoder_geometry(backwards, 193920, other));
    SYNTH_TEST_CHECK(other.samples_per_frame == kProductionSamplesPerFrame);
    // Same samples-per-frame, same frame count -- and a different stack. This
    // is the assertion recording that the frame count CANNOT catch a reversed
    // walk, so nobody later mistakes rule 1 for coverage of the stride order.
    SYNTH_TEST_CHECK(other.frames == 101);
    return 0;
}

// Rule 5: a non-finite sample is refused. The graph cannot see it -- a graph
// builder has shapes and no values -- so this is the seam that must.
int check_waveform_values() {
    Context                              context = make_context(ggml_tensor_overhead() * 64);
    synth::qwen3tts::CodecEncoderWeights weights;
    build_geometry_weights(context.get(), kProductionStrideKernels, weights);

    synth::qwen3tts::CodecEncoderGeometry geometry;
    std::vector<float>                    pcm(size_t(kProductionSamplesPerFrame) * 3, 0.25f);
    SYNTH_TEST_CHECK(synth::qwen3tts::codec_encoder_check_waveform(weights, pcm.data(), pcm.size(), geometry) ==
                     SYNTH_OK);
    SYNTH_TEST_CHECK(geometry.frames == 3);

    for (float poison : { std::numeric_limits<float>::quiet_NaN(), INFINITY, -INFINITY }) {
        std::vector<float> spoiled  = pcm;
        // Late in the clip, not first: a check that only looks at the head
        // would pass a clip whose tail is NaN.
        spoiled[spoiled.size() - 2] = poison;
        SYNTH_TEST_CHECK(synth::qwen3tts::codec_encoder_check_waveform(weights, spoiled.data(), spoiled.size(),
                                                                       geometry) != SYNTH_OK);
        // A refused clip reports no geometry, so a caller that ignores the
        // status cannot go on to size a buffer from it.
        SYNTH_TEST_CHECK(geometry.frames == 0);
    }

    // Rule 4 again, at this seam rather than the geometry one.
    SYNTH_TEST_CHECK(synth::qwen3tts::codec_encoder_check_waveform(weights, pcm.data(), 1919, geometry) != SYNTH_OK);
    SYNTH_TEST_CHECK(synth::qwen3tts::codec_encoder_check_waveform(weights, nullptr, 193920, geometry) != SYNTH_OK);
    return 0;
}

// Rule 6: the causal padding is causal. Appending samples to the end of a clip
// must not change the latents of the frames that precede them.
//
// Both lengths are whole multiples of the 48 samples a frame costs, so every
// convolution's `extra_padding` is zero on both runs and the shared prefix is
// exact rather than approximate. That is deliberate: at a length that is NOT a
// multiple, the frame downsampler's replicate padding legitimately changes the
// final frame, and a test that did not control for it would either fail or have
// to carry a fudge that hides a real bug.
bool check_causality(const Fixture & fixture, float & max_diff) {
    std::vector<float> shorter;
    std::vector<float> longer;
    int64_t            short_frames = 0;
    int64_t            long_frames  = 0;
    int                nodes        = 0;
    int                second_nodes = 0;
    if (!run_encoder(fixture, kSamplesPerFrame * 8, 0.0f, shorter, short_frames, nodes)) {
        return false;
    }
    if (!run_encoder(fixture, kSamplesPerFrame * 12, 0.0f, longer, long_frames, second_nodes)) {
        return false;
    }
    if (short_frames != 8 || long_frames != 12) {
        return false;
    }
    // Both lengths are whole frames, so both build the same graph.
    if (nodes != kNodesWholeFrames || second_nodes != kNodesWholeFrames) {
        std::printf("    causality runs built %d and %d nodes, expected %d\n", nodes, second_nodes, kNodesWholeFrames);
        return false;
    }

    max_diff = 0.0f;
    for (size_t index = 0; index < shorter.size(); ++index) {
        max_diff = std::fmax(max_diff, std::fabs(shorter[index] - longer[index]));
    }
    return true;
}

// The seanet_tail tap must hold the SEANet stack's own output after the graph
// has run.
//
// Checked against a second graph that ends at the tail, where it is the final
// node and so cannot be recycled. Without the builder's ggml_set_output the
// first graph's tap is whatever later node the allocator put on top of it --
// plausible floats of exactly the right shape, which is how this went unnoticed
// through a whole stage-wise measurement run against the real package.
bool check_taps_survive_allocation(const Fixture & fixture, float & max_diff) {
    std::vector<float> latents;
    std::vector<float> tapped;
    int64_t            frames = 0;
    int                nodes  = 0;
    if (!run_encoder(fixture, kSamplesPerFrame * 8, 0.0f, latents, frames, nodes, &tapped)) {
        return false;
    }

    // The same stack, built so that the tail IS the graph's output.
    synth::qwen3tts::CodecEncoderGeometry geometry;
    if (!synth::qwen3tts::codec_encoder_geometry(fixture.weights, kSamplesPerFrame * 8, geometry)) {
        return false;
    }
    Context               input_ctx    = make_context(ggml_tensor_overhead() * 8);
    ggml_tensor *         waveform     = ggml_new_tensor_2d(input_ctx.get(), GGML_TYPE_F32, 1, kSamplesPerFrame * 8);
    ggml_backend_buffer_t input_buffer = ggml_backend_alloc_ctx_tensors(input_ctx.get(), fixture.backend);
    if (input_buffer == nullptr) {
        return false;
    }
    const std::vector<float> pcm = lcg_clip(kSamplesPerFrame * 8, 0.0f);
    ggml_backend_tensor_set(waveform, pcm.data(), 0, ggml_nbytes(waveform));

    Context graph_ctx =
        make_context(ggml_tensor_overhead() * (kNodeBudget + 64) + ggml_graph_overhead_custom(kNodeBudget, false));
    ggml_cgraph *  graph     = ggml_new_graph_custom(graph_ctx.get(), kNodeBudget, false);
    ggml_tensor *  tail      = synth::qwen3tts::build_codec_encoder_seanet(graph_ctx.get(), waveform, fixture.weights);
    bool           ok        = tail != nullptr && size_t(ggml_nelements(tail)) == tapped.size();
    ggml_gallocr_t allocator = nullptr;
    if (ok) {
        ggml_build_forward_expand(graph, tail);
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(fixture.backend));
        ok        = allocator != nullptr && ggml_gallocr_alloc_graph(allocator, graph);
    }
    if (ok) {
        ok = ggml_backend_graph_compute(fixture.backend, graph) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        std::vector<float> direct(size_t(ggml_nelements(tail)), 0.0f);
        ggml_backend_tensor_get(tail, direct.data(), 0, ggml_nbytes(tail));
        max_diff = 0.0f;
        for (size_t index = 0; index < direct.size(); ++index) {
            max_diff = std::fmax(max_diff, std::fabs(direct[index] - tapped[index]));
        }
    }
    if (allocator != nullptr) {
        ggml_gallocr_free(allocator);
    }
    ggml_backend_buffer_free(input_buffer);
    return ok;
}

// The attention is causal and NOT windowed, on a probe that can tell the
// difference.
//
// ONE layer, sixteen positions. That shape is the whole point. With L stacked
// layers a window of w gives a receptive field of 1 + L*(w - 1) positions, so at
// eight layers the last position depends on the first for any window down to 2
// -- an end-to-end check over the real stack cannot see a window at all below
// about 2400 frames, and the 30-frame one in main() certainly cannot. At L = 1
// the receptive field IS the window: position 15 sees positions 15-w+1..15 and
// nothing earlier, so a windowed mask of any width under 16 severs its
// dependence on position 0 and a plain causal mask does not.
//
// This replaces an assertion that could not have failed. See codec-encoder.h
// for the source trace and the measurement on the real weights.
bool check_attention_is_unwindowed(const Fixture & fixture, double & change) {
    constexpr int64_t kProbePositions = 16;

    synth::qwen3tts::CodecEncoderWeights one_layer = fixture.weights;
    one_layer.layers.resize(1);

    Context               input_ctx    = make_context(ggml_tensor_overhead() * 8);
    ggml_tensor *         hidden       = ggml_new_tensor_2d(input_ctx.get(), GGML_TYPE_F32, kHidden, kProbePositions);
    ggml_tensor *         positions    = ggml_new_tensor_1d(input_ctx.get(), GGML_TYPE_I32, kProbePositions);
    ggml_backend_buffer_t input_buffer = ggml_backend_alloc_ctx_tensors(input_ctx.get(), fixture.backend);
    if (input_buffer == nullptr) {
        return false;
    }
    // Braced, not parenthesised: `size_t(kProbePositions)` on a constexpr name
    // parses as a parameter declaration, not a size.
    std::vector<int32_t> sequential(static_cast<size_t>(kProbePositions), 0);
    for (int64_t index = 0; index < kProbePositions; ++index) {
        sequential[size_t(index)] = int32_t(index);
    }
    ggml_backend_tensor_set(positions, sequential.data(), 0, ggml_nbytes(positions));

    LcgStream          stream(kSeed ^ 0xd1b54a32d192ed03ull);
    std::vector<float> base = stream.fill(size_t(kHidden * kProbePositions), 1.0f, 0.0f);

    std::vector<float> outputs[2];
    for (int run = 0; run < 2; ++run) {
        std::vector<float> values = base;
        if (run == 1) {
            // Position 0 only, and one channel of it: a LayerNorm subtracts the
            // mean, so shifting every channel equally would be erased before
            // the attention ever saw it.
            values[0] += 8.0f;
        }
        ggml_backend_tensor_set(hidden, values.data(), 0, ggml_nbytes(hidden));

        Context graph_ctx =
            make_context(ggml_tensor_overhead() * (kNodeBudget + 64) + ggml_graph_overhead_custom(kNodeBudget, false));
        ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx.get(), kNodeBudget, false);
        ggml_tensor * result =
            synth::qwen3tts::build_codec_encoder_transformer(graph_ctx.get(), hidden, positions, one_layer);
        if (result == nullptr) {
            ggml_backend_buffer_free(input_buffer);
            return false;
        }
        ggml_build_forward_expand(graph, result);
        ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(fixture.backend));
        bool           ok        = allocator != nullptr && ggml_gallocr_alloc_graph(allocator, graph);
        if (ok) {
            ok = ggml_backend_graph_compute(fixture.backend, graph) == GGML_STATUS_SUCCESS;
        }
        if (ok) {
            outputs[run].assign(size_t(ggml_nelements(result)), 0.0f);
            ggml_backend_tensor_get(result, outputs[run].data(), 0, ggml_nbytes(result));
        }
        if (allocator != nullptr) {
            ggml_gallocr_free(allocator);
        }
        if (!ok) {
            ggml_backend_buffer_free(input_buffer);
            return false;
        }
    }
    ggml_backend_buffer_free(input_buffer);

    change = 0.0;
    for (int64_t channel = 0; channel < kHidden; ++channel) {
        const size_t at = size_t((kProbePositions - 1) * kHidden + channel);
        change          = std::fmax(change, double(std::fabs(outputs[1][at] - outputs[0][at])));
    }
    return true;
}

}  // namespace

int main() {
    SYNTH_TEST_CHECK(check_production_geometry() == 0);
    SYNTH_TEST_CHECK(check_strides_are_read() == 0);
    SYNTH_TEST_CHECK(check_waveform_values() == 0);

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
        std::printf("qwen3-tts-codec-encoder: %s (device type %d)\n", ggml_backend_dev_name(device), int(type));

        Fixture fixture;
        SYNTH_TEST_CHECK(build_fixture(device, fixture));

        // Rule 3: the latents are [codebook_dim, frames] and every element is
        // finite. run_encoder asserts the shape; the finiteness is here.
        std::vector<float> latents;
        int64_t            frames = 0;
        int                nodes  = 0;
        SYNTH_TEST_CHECK(run_encoder(fixture, kSamplesPerFrame * 30, 0.0f, latents, frames, nodes));
        SYNTH_TEST_CHECK(nodes == kNodesWholeFrames);
        SYNTH_TEST_CHECK(frames == 30);
        SYNTH_TEST_CHECK(latents.size() == size_t(kHidden * 30));
        double magnitude = 0.0;
        for (float value : latents) {
            SYNTH_TEST_CHECK(std::isfinite(value));
            magnitude = std::fmax(magnitude, double(std::fabs(value)));
        }
        // A stack that decayed to zero would satisfy every other rule here.
        SYNTH_TEST_CHECK(magnitude > 1e-6);
        std::printf("    frames %lld, |latents|max %.4g, %d nodes\n", (long long) frames, magnitude, nodes);

        // A clip that does NOT end on a frame boundary. This is the only place
        // the frame downsampler's right-hand extra_padding is exercised at all
        // -- every other length here is a whole number of frames, where it is
        // zero -- and the replicate pad it needs on that side is four more
        // graph nodes. 401 samples is 8.35 frames, so the encoder rounds up to
        // 9 and the downsampler pads its input by one position.
        std::vector<float> ragged;
        int64_t            ragged_frames = 0;
        int                ragged_nodes  = 0;
        SYNTH_TEST_CHECK(run_encoder(fixture, kSamplesPerFrame * 8 + 17, 0.0f, ragged, ragged_frames, ragged_nodes));
        SYNTH_TEST_CHECK(ragged_frames == 9);
        SYNTH_TEST_CHECK(ragged_nodes == kNodesRaggedTail);
        for (float value : ragged) {
            SYNTH_TEST_CHECK(std::isfinite(value));
        }
        std::printf("    ragged clip: %lld frames, %d nodes\n", (long long) ragged_frames, ragged_nodes);

        // End-to-end liveness: sample 0 must reach the last frame at all. This
        // is NOT a check on the sliding window -- at eight layers the last
        // position depends on the first under a window as well as without one,
        // which is exactly the mistake check_attention_is_unwindowed exists to
        // correct. What it does catch is a stack that has stopped propagating:
        // a severed residual, a mask that blocks everything, an attention that
        // reads only the diagonal.
        std::vector<float> bumped;
        int64_t            bumped_frames = 0;
        SYNTH_TEST_CHECK(run_encoder(fixture, kSamplesPerFrame * 30, 4.0f, bumped, bumped_frames, nodes));
        double last_frame_change = 0.0;
        for (int64_t channel = 0; channel < kHidden; ++channel) {
            const size_t at   = size_t((30 - 1) * kHidden + channel);
            last_frame_change = std::fmax(last_frame_change, double(std::fabs(bumped[at] - latents[at])));
        }
        std::printf("    last frame moves %.4g when sample 0 moves\n", last_frame_change);
        SYNTH_TEST_CHECK(last_frame_change > 1e-6);

        // The window probe that can actually fail.
        double window_change = 0.0;
        SYNTH_TEST_CHECK(check_attention_is_unwindowed(fixture, window_change));
        std::printf("    single-layer probe: position 15 moves %.4g when position 0 moves\n", window_change);
        SYNTH_TEST_CHECK(window_change > 1e-6);

        // The taps a stage-wise comparison reads are readable.
        float tap_diff = 0.0f;
        SYNTH_TEST_CHECK(check_taps_survive_allocation(fixture, tap_diff));
        std::printf("    seanet_tail tap vs direct build: max_diff %.4g\n", double(tap_diff));
        SYNTH_TEST_CHECK(tap_diff == 0.0f);

        // Rule 6.
        float causal_diff = 0.0f;
        SYNTH_TEST_CHECK(check_causality(fixture, causal_diff));
        std::printf("    causal prefix max_diff %.4g\n", double(causal_diff));
        // Exactly 0 on the CPU -- the shared prefix is bit-identical, because
        // every output element of every convolution reduces over the same
        // inputs in the same order whatever follows it.
        //
        // NOT exactly 0 on an accelerator, and that is arithmetic rather than a
        // broken prefix: CUDA tiles a matrix multiply by its column count, so
        // the 8-frame and 12-frame runs accumulate the shared columns in
        // different orders. Measured on a GB10 (sm_121a): 0.003435, stable to
        // every digit across repeated runs, against a signal whose peak is
        // 3.001 -- one part in 900. The same split the sibling codec test makes
        // for the same reason (qwen3_tts_codec_test.cpp's own 1e-4 / 2e-2).
        //
        // Both bounds stay far under the fault they exist to catch: a symmetric
        // pad moves this to 1.729, which is 17,000x the CPU bound and 86x the
        // accelerator one.
        const float causal_tolerance = type == GGML_BACKEND_DEVICE_TYPE_CPU ? 1e-4f : 2e-2f;
        SYNTH_TEST_CHECK(causal_diff < causal_tolerance);

        ++exercised;
    }
    SYNTH_TEST_CHECK(exercised > 0);
    return 0;
}
