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

#include "arch/qwen3-tts/codec-encoder-host.h"
#include "arch/qwen3-tts/codec-encoder.h"
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
#include <string>
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

// The quantizer half's widths. `kProjected` is `codebook_dim / 2` exactly as
// catalog.cpp derives it, and every number here is small and mutually prime
// with the others so a swapped extent cannot line up by coincidence.
constexpr int64_t kProjected     = kHidden / 2;  // 16
constexpr int64_t kCodebookSize  = 8;
constexpr int64_t kGroups        = 16;
constexpr int64_t kSemanticGroup = 1;

// The fixture carries all THIRTY-ONE acoustic codebooks, not the fifteen it
// reads. That is what the real checkpoint carries (catalog.cpp's
// kCodecEncoderAcousticQuantizerCount) and it is load-bearing here rather than
// decorative: an implementation that took the code-group count from the
// resolved codebook list instead of from `quantizer_count` would emit 32 groups
// on this fixture, and a fixture holding exactly fifteen could not tell the two
// apart.
constexpr int64_t kAcousticAvailable = 31;

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

// READ THIS BEFORE INJECTING A FAULT INTO THIS FILE'S SUBJECT, and before
// adding an assertion above the two node checks.
//
// These two are the strongest assertions here, and they fire FIRST for any
// fault that changes the graph's size -- which swallows the failure signal of
// every weaker assertion below them. That is correct behaviour for a tighter
// check and it is not silent (the suite still fails), but it means a
// rule-deletion run reports the node count rather than the rule it was aimed
// at. Enumerated, from the runs that produced them:
//
//   MASKED by the node pin, and what each reports once it is relaxed:
//     symmetric (non-causal) convolution padding  -> `causal_diff`, at ~1.7
//     latents scaled to zero                      -> `magnitude > 1e-6`
//     ANY injected sliding window                 -> the liveness check, then
//                                                    `window_change > 1e-6`
//   NOT masked, because they change no shape:
//     record_tap without ggml_set_output          -> `tap_diff == 0`, at ~3.79
//     a tap list not cleared on entry             -> the taps-as-declared check
//     ceil -> floor divide                        -> the frame count
//     latents returned transposed                 -> rule 3's shape check
//
// To isolate one of the masked three, comment out the three
// `nodes == kNodes*` assertions and the pair inside check_causality, then
// re-run. Do not weaken the pin to make an inversion legible.
//
// A consequence worth knowing rather than fixing: a real windowing regression
// would trip the node pin and the liveness check before reaching
// check_attention_is_unwindowed, so that probe is insurance, not the assertion
// that would report the fault.

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

    // The two quantizer cascades. `input_proj` is a kernel-one convolution with
    // no bias, so ggml reports it [1, in, out]; `output_proj` is deliberately
    // left null, because it is a DECODE-time tensor and the encode path must
    // never read it. A wrapper that did would fail here rather than quietly
    // reconstructing in the wrong space.
    auto add_quantizer = [&](synth::qwen3tts::CodecQuantizerWeights & target, int64_t stages) {
        target.input_proj = add(ggml_new_tensor_3d(pctx, GGML_TYPE_F32, 1, kHidden, kProjected), 0.25f);
        target.codebooks.assign(size_t(stages), nullptr);
        for (int64_t stage = 0; stage < stages; ++stage) {
            target.codebooks[size_t(stage)] =
                add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kProjected, kCodebookSize), 0.6f);
        }
    };
    add_quantizer(fixture.weights.semantic, kSemanticGroup);
    add_quantizer(fixture.weights.acoustic, kAcousticAvailable);

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
    // A whole number of frames, so the same graph as everywhere else. Asserted
    // here as well as in main(): asking for taps is the one thing that changes
    // what the builder does, and this is the only run that asks.
    if (nodes != kNodesWholeFrames) {
        std::printf("    taps run built %d nodes, expected %d\n", nodes, kNodesWholeFrames);
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

// ---------------------------------------------------------------------------
// The host wrapper and the split residual vector quantizer.
//
// These run on the CPU only, and once rather than once per device, because
// encode_codec_reference builds its OWN CPU-only BackendPlan -- that is its
// documented placement, not an accident of this test -- so a fixture whose
// weights live on an accelerator has nothing for it to read.
//
// None of them is masked by the node pin above: the quantizer is host code and
// changes no graph. The three faults the pin swallows are all graph faults.
//
// Every rule below was inverted and re-run, and each reported at the assertion
// it was aimed at:
//
//   `<` -> `<=` in the argmin scan              -> `code == 1` (the tie check)
//   acoustic codes leading the grid             -> `codes[0] == semantic_frame0`
//   the grid stored stage-major                 -> the per-frame semantic check
//   codes_equal blind to element order          -> the stage-major inequality
//   groups from the codebook list (32, not 16)  -> `groups == kGroups`
//   a stage-local code written as a global one  -> the range check
//   the frame trim as a floor divide            -> `ragged.frames == 9`
//   the silent-reference gate removed           -> the INVALID_ARG refusal
//
// ONE MASKING, recorded rather than fixed, in the spirit of the node-pin
// comment above. Storing the grid stage-major trips the per-frame semantic
// check first, so the `codes_equal` stage-major assertion below it is never
// reached on that fault -- which is why `codes_equal` is inverted separately,
// on its own, rather than being assumed to be covered by the layout fault.
// ---------------------------------------------------------------------------

synth::qwen3tts::HParams make_hparams() {
    synth::qwen3tts::HParams hparams;
    // `codebook_dim` is this port's name for the LATENT width (512 in the real
    // package); the projected space the quantizer decides in is half of it.
    // catalog.cpp derives both the same way, so the fixture must too.
    hparams.codec.decoder.codebook_dim             = uint32_t(kHidden);
    hparams.codec.decoder.codebook_size            = uint32_t(kCodebookSize);
    hparams.codec.decoder.quantizer_count          = uint32_t(kGroups);
    hparams.codec.decoder.semantic_quantizer_count = uint32_t(kSemanticGroup);
    return hparams;
}

synth_status_t run_encode(const Fixture &                  fixture,
                          const synth::qwen3tts::HParams & hparams,
                          int64_t                          samples,
                          synth::qwen3tts::CodecEncoding & encoding,
                          const char *&                    diagnostic_code) {
    const char *             message = nullptr;
    const std::vector<float> pcm     = lcg_clip(samples, 0.0f);
    return synth::qwen3tts::encode_codec_reference(hparams, fixture.weights, pcm, 1, encoding, diagnostic_code,
                                                   message);
}

// The stage-0 code one branch WOULD choose, recomputed here from the fixture's
// own tensors and the encoding's latents.
//
// Deliberately not read out of `encoding.residuals`: that buffer is produced by
// the code under test, so an argmin over it would agree with a wrapper that had
// swapped the two branches. This projects the latents with the branch's own
// `input_proj` and scans the branch's own codebook, which is the "directly
// computed argmin" the rule is about.
//
// In double, and it also reports the winning margin, so the caller can assert
// the decision is not close enough for float32 ordering to be what separated
// the two answers.
bool independent_stage0_code(const synth::qwen3tts::CodecQuantizerWeights & quantizer,
                             const std::vector<float> &                     latents,
                             int64_t                                        frame,
                             int32_t &                                      code,
                             double &                                       margin) {
    if (quantizer.input_proj == nullptr || quantizer.codebooks.empty() ||
        latents.size() < size_t((frame + 1) * kHidden)) {
        return false;
    }
    std::vector<float> projection(size_t(ggml_nelements(quantizer.input_proj)));
    std::vector<float> codebook(size_t(ggml_nelements(quantizer.codebooks[0])));
    ggml_backend_tensor_get(quantizer.input_proj, projection.data(), 0, ggml_nbytes(quantizer.input_proj));
    ggml_backend_tensor_get(quantizer.codebooks[0], codebook.data(), 0, ggml_nbytes(quantizer.codebooks[0]));

    const float *       latent = latents.data() + size_t(frame * kHidden);
    std::vector<double> z(size_t(kProjected), 0.0);
    for (int64_t out = 0; out < kProjected; ++out) {
        double accumulator = 0.0;
        for (int64_t in = 0; in < kHidden; ++in) {
            accumulator += double(projection[size_t(out * kHidden + in)]) * double(latent[in]);
        }
        z[size_t(out)] = accumulator;
    }

    double best   = std::numeric_limits<double>::infinity();
    double second = std::numeric_limits<double>::infinity();
    code          = -1;
    for (int64_t row = 0; row < kCodebookSize; ++row) {
        double distance = 0.0;
        for (int64_t index = 0; index < kProjected; ++index) {
            const double difference = z[size_t(index)] - double(codebook[size_t(row * kProjected + index)]);
            distance += difference * difference;
        }
        if (distance < best) {
            second = best;
            best   = distance;
            code   = int32_t(row);
        } else if (distance < second) {
            second = distance;
        }
    }
    margin = second - best;
    return code >= 0;
}

// Rules 1, 2, 3, 5 and 6: the grid's width, its value range, which branch leads
// it, how it is laid out, and the frame trim.
int check_reference_encoding(const Fixture & fixture) {
    const synth::qwen3tts::HParams hparams = make_hparams();

    // Sixteen frames exactly, because the layout check below needs the one
    // size where a 16x16 grid has the SAME dimensions under either reading and
    // a shape check alone would pass a stage-major buffer.
    synth::qwen3tts::CodecEncoding encoding;
    const char *                   diagnostic = nullptr;
    SYNTH_TEST_CHECK(run_encode(fixture, hparams, kSamplesPerFrame * 16, encoding, diagnostic) == SYNTH_OK);
    SYNTH_TEST_CHECK(diagnostic == nullptr);

    // Rule 1: sixteen groups, from `quantizer_count`. NOT the 32 the fixture's
    // codebook lists could supply, and not the 31 the acoustic cascade carries.
    SYNTH_TEST_CHECK(encoding.groups == uint64_t(kGroups));
    SYNTH_TEST_CHECK(encoding.frames == 16);
    SYNTH_TEST_CHECK(encoding.codes.size() == size_t(kGroups * 16));
    SYNTH_TEST_CHECK(encoding.projected == uint64_t(kProjected));
    SYNTH_TEST_CHECK(encoding.latent_width == uint64_t(kHidden));
    SYNTH_TEST_CHECK(encoding.residuals.size() == size_t(kGroups * 16 * kProjected));
    SYNTH_TEST_CHECK(encoding.reconstruction.size() == size_t(2 * 16 * kProjected));

    // Rule 2: a RANGE check over every code, not a spot check.
    for (int32_t code : encoding.codes) {
        SYNTH_TEST_CHECK(code >= 0 && code < int32_t(kCodebookSize));
    }

    // The fixture has to be able to discriminate at all: a codebook that every
    // frame resolves to the same row would make the layout check below vacuous.
    bool spread = false;
    for (size_t index = 1; index < encoding.codes.size(); ++index) {
        spread = spread || encoding.codes[index] != encoding.codes[0];
    }
    SYNTH_TEST_CHECK(spread);

    // Rule 3: THE SEMANTIC STAGE LEADS. Swapping the two branches produces
    // sixteen valid-looking codes and a different voice, and nothing about the
    // grid's shape or range would notice.
    int32_t semantic_frame0 = -1;
    int32_t acoustic_frame0 = -1;
    int32_t semantic_frame1 = -1;
    double  margin          = 0.0;
    SYNTH_TEST_CHECK(independent_stage0_code(fixture.weights.semantic, encoding.latents, 0, semantic_frame0, margin));
    SYNTH_TEST_CHECK(margin > 1e-6);
    SYNTH_TEST_CHECK(independent_stage0_code(fixture.weights.acoustic, encoding.latents, 0, acoustic_frame0, margin));
    SYNTH_TEST_CHECK(margin > 1e-6);
    // The precondition that makes the NEXT line able to fail under a swapped
    // pair of branches. Without it, `codes[0] == semantic_frame0` would hold
    // under both orders whenever the two branches happen to agree at frame 0,
    // and the assertion the branch-swap inversion is aimed at would be passing
    // on fixture luck rather than on the rule.
    SYNTH_TEST_CHECK(acoustic_frame0 != semantic_frame0);
    SYNTH_TEST_CHECK(encoding.codes[0] == semantic_frame0);
    for (int64_t frame = 0; frame < 16; ++frame) {
        int32_t expected = -1;
        SYNTH_TEST_CHECK(independent_stage0_code(fixture.weights.semantic, encoding.latents, frame, expected, margin));
        SYNTH_TEST_CHECK(margin > 1e-6);
        SYNTH_TEST_CHECK(encoding.codes[size_t(frame * kGroups)] == expected);
    }

    // Rule 5: THE LAYOUT IS GROUP-FASTEST. Element 1 of the flat buffer is
    // frame 0's group 1 -- the acoustic branch's first stage -- and NOT frame
    // 1's group 0, which is what a stage-major buffer would put there.
    SYNTH_TEST_CHECK(independent_stage0_code(fixture.weights.semantic, encoding.latents, 1, semantic_frame1, margin));
    SYNTH_TEST_CHECK(margin > 1e-6);
    // Without this the previous two cannot tell the layouts apart, and the
    // assertion below would hold under both of them.
    SYNTH_TEST_CHECK(acoustic_frame0 != semantic_frame1);
    SYNTH_TEST_CHECK(encoding.codes[1] == acoustic_frame0);

    // ... and the same rule stated through the production comparison, which is
    // what Task 9's round trip and Task 13's re-preparation identity also
    // drive. A stage-major grid of the same dimensions is NOT accepted as
    // equal.
    SYNTH_TEST_CHECK(synth::qwen3tts::codes_equal(encoding, encoding.codes.data(), 16));
    std::vector<int32_t> stage_major(encoding.codes.size(), 0);
    for (int64_t frame = 0; frame < 16; ++frame) {
        for (int64_t group = 0; group < kGroups; ++group) {
            stage_major[size_t(group * 16 + frame)] = encoding.codes[size_t(frame * kGroups + group)];
        }
    }
    SYNTH_TEST_CHECK(stage_major != encoding.codes);
    SYNTH_TEST_CHECK(!synth::qwen3tts::codes_equal(encoding, stage_major.data(), 16));
    // The frame count is part of the comparison, not just the buffer.
    SYNTH_TEST_CHECK(!synth::qwen3tts::codes_equal(encoding, encoding.codes.data(), 15));
    SYNTH_TEST_CHECK(!synth::qwen3tts::codes_equal(encoding, nullptr, 16));

    // The gap grid: same size and same GROUP-FASTEST layout as the codes, so
    // the two index alike. Pinned by value at group 0, against the margin
    // `independent_stage0_code` computes in double from the direct (z-e)^2
    // form -- which is both a value check and a layout check, since a
    // stage-major gap grid would put frame 1's group 0 at index 1.
    SYNTH_TEST_CHECK(encoding.gaps.size() == encoding.codes.size());
    float smallest = std::numeric_limits<float>::infinity();
    for (float gap : encoding.gaps) {
        // Non-negative by construction: the second best is never nearer than
        // the best. Strictly positive here because the drawn fixture produces
        // no exact ties -- which is what makes the tie fixture's all-zero grid
        // a distinguishable state rather than the default one.
        SYNTH_TEST_CHECK(std::isfinite(gap) && gap > 0.0f);
        smallest = std::fmin(smallest, gap);
    }
    SYNTH_TEST_CHECK(encoding.narrowest_gap == smallest);
    for (int64_t frame = 0; frame < 16; ++frame) {
        int32_t ignored  = -1;
        double  expected = 0.0;
        SYNTH_TEST_CHECK(independent_stage0_code(fixture.weights.semantic, encoding.latents, frame, ignored, expected));
        const double observed = double(encoding.gaps[size_t(frame * kGroups)]);
        SYNTH_TEST_CHECK(std::fabs(observed - expected) <= 1e-3 * expected);
    }

    // Rule 6: the frame trim is the CEILING divide, applied after the graph. A
    // clip of 8 whole frames plus 17 samples is nine frames, not eight.
    synth::qwen3tts::CodecEncoding ragged;
    SYNTH_TEST_CHECK(run_encode(fixture, hparams, kSamplesPerFrame * 8 + 17, ragged, diagnostic) == SYNTH_OK);
    SYNTH_TEST_CHECK(ragged.frames == 9);
    SYNTH_TEST_CHECK(ragged.codes.size() == size_t(kGroups * 9));
    SYNTH_TEST_CHECK(ragged.latents.size() == size_t(kHidden * 9));

    std::printf("    encoding: %llu frames x %llu groups, ref_rms %.4g, narrowest gap %.4g\n",
                (unsigned long long) encoding.frames, (unsigned long long) encoding.groups, double(encoding.ref_rms),
                double(encoding.narrowest_gap));
    return 0;
}

// Rule 4: TIES RESOLVE TO THE LOWEST ID.
//
// The fixture's codebooks are overwritten so that rows 1 and 2 are both
// all-zero -- byte-identical, so their squared distances are equal bit for bit,
// not merely close -- while every other row sits a million away in each of the
// sixteen projected dimensions and can never win. Whatever the residual, the
// argmin is a genuine exact tie between 1 and 2, and every one of the sixteen
// stages must choose 1.
//
// Subtracting an all-zero row leaves the residual untouched, so the tie recurs
// at every stage rather than only the first, and the expected grid is every
// code equal to 1 -- which an inverted rule turns into every code equal to 2.
int check_tie_resolves_to_lowest_id(Fixture & fixture) {
    const int64_t      rows = kCodebookSize;
    std::vector<float> pattern(size_t(rows * kProjected), 0.0f);
    for (int64_t row = 0; row < rows; ++row) {
        if (row == 1 || row == 2) {
            continue;
        }
        const float far = (row % 2 == 0 ? 1.0f : -1.0f) * 1.0e6f;
        for (int64_t index = 0; index < kProjected; ++index) {
            pattern[size_t(row * kProjected + index)] = far;
        }
    }
    auto overwrite = [&](synth::qwen3tts::CodecQuantizerWeights & quantizer) {
        for (ggml_tensor * codebook : quantizer.codebooks) {
            ggml_backend_tensor_set(codebook, pattern.data(), 0, ggml_nbytes(codebook));
        }
    };
    overwrite(fixture.weights.semantic);
    overwrite(fixture.weights.acoustic);

    const synth::qwen3tts::HParams hparams = make_hparams();
    synth::qwen3tts::CodecEncoding encoding;
    const char *                   diagnostic = nullptr;
    SYNTH_TEST_CHECK(run_encode(fixture, hparams, kSamplesPerFrame * 4, encoding, diagnostic) == SYNTH_OK);
    SYNTH_TEST_CHECK(encoding.codes.size() == size_t(kGroups * 4));
    for (int32_t code : encoding.codes) {
        SYNTH_TEST_CHECK(code == 1);
    }
    // The tie is exact, so the reported gap between best and second best is
    // exactly zero -- which is also what proves the two candidates really were
    // equal rather than merely close. EVERY entry, not just the minimum: the
    // tie recurs at all sixteen stages, so a single zero somewhere would be a
    // much weaker statement than the whole grid being zero.
    SYNTH_TEST_CHECK(encoding.narrowest_gap == 0.0f);
    SYNTH_TEST_CHECK(encoding.gaps.size() == encoding.codes.size());
    for (float gap : encoding.gaps) {
        SYNTH_TEST_CHECK(gap == 0.0f);
    }
    return 0;
}

// The refusal surface: every shape and hyper-parameter this wrapper checks
// before it touches a tensor.
//
// CLAUDE.md's testing policy asks for the malformed-input and error-mapping
// half of each slice, and a refusal nothing drives is a refusal nobody checked
// -- the same reasoning the catalog's own resolver tests use. Each case mutates
// exactly one thing away from a pair that is asserted to be ACCEPTED first, so
// a refusal can only be attributed to the mutation.
int check_refusals(const Fixture & fixture, ggml_backend_t backend) {
    using Weights                       = synth::qwen3tts::CodecEncoderWeights;
    const synth::qwen3tts::HParams base = make_hparams();
    const std::vector<float>       pcm  = lcg_clip(kSamplesPerFrame * 3, 0.0f);
    synth::qwen3tts::CodecEncoding encoding;
    const char *                   code    = nullptr;
    const char *                   message = nullptr;

    // Without this every refusal below is unattributable.
    SYNTH_TEST_CHECK(synth::qwen3tts::encode_codec_reference(base, fixture.weights, pcm, 1, encoding, code, message) ==
                     SYNTH_OK);

    auto refused = [&](const synth::qwen3tts::HParams & hparams, const Weights & weights) {
        const synth_status_t status =
            synth::qwen3tts::encode_codec_reference(hparams, weights, pcm, 1, encoding, code, message);
        // INVALID_ARG and not INTERNAL: a package whose metadata and tensors
        // disagree is a bad argument to this function, not a bug inside it.
        // Nothing is left behind, and no diagnostic name is claimed -- only the
        // silent-reference refusal names itself.
        return status == SYNTH_ERR_INVALID_ARG && encoding.codes.empty() && code == nullptr;
    };

    // The hyper-parameter surface.
    //
    // MEASURED, NOT ASSUMED: with the whole `groups <= 0 || semantic <= 0 ||
    // semantic >= groups || ...` gate deleted, every case in this block below
    // the last one STILL refuses, because `branch_shapes_agree` catches the
    // same configurations one step later -- a zero or negative stage count, an
    // `input_proj` whose middle extent is not the declared latent width, a
    // codebook whose row count is not the declared size. That gate is
    // defence-in-depth: it refuses earlier and more cheaply, and it is not
    // independently observable through this seam.
    //
    // So what this block pins is THE CONTRACT -- every one of these
    // configurations is refused, as INVALID_ARG, leaving nothing behind and
    // claiming no diagnostic name -- rather than any single line of it. Said
    // here because a rule-deletion run on that gate alone comes back green, and
    // a reader who did not know why would take the tests for decoration.
    //
    // The LAST case is the exception and is the one that isolates the gate: it
    // supplies tensors that agree with each other at an ODD latent width, which
    // `branch_shapes_agree` accepts and only the `latent_width % 2` rule
    // rejects. With the gate gone it reaches the graph and comes back
    // SYNTH_ERR_INTERNAL instead, so `refused` fails.
    {
        synth::qwen3tts::HParams h      = base;
        h.codec.decoder.quantizer_count = 0;
        SYNTH_TEST_CHECK(refused(h, fixture.weights));
        h                                        = base;
        h.codec.decoder.semantic_quantizer_count = 0;
        SYNTH_TEST_CHECK(refused(h, fixture.weights));
        // Semantic must be a strict prefix: equal leaves the acoustic branch
        // nothing to do and would silently emit a semantic-only grid.
        h                                        = base;
        h.codec.decoder.semantic_quantizer_count = uint32_t(kGroups);
        SYNTH_TEST_CHECK(refused(h, fixture.weights));
        h                            = base;
        h.codec.decoder.codebook_dim = 0;
        SYNTH_TEST_CHECK(refused(h, fixture.weights));
        // Odd: the projected width is codebook_dim / 2, and a half that is not
        // exact means the two are not the pair this cascade was built from.
        h                            = base;
        h.codec.decoder.codebook_dim = uint32_t(kHidden) + 1;
        SYNTH_TEST_CHECK(refused(h, fixture.weights));
        h                             = base;
        h.codec.decoder.codebook_size = 1;
        SYNTH_TEST_CHECK(refused(h, fixture.weights));
        // Declared wider than the tensors: caught by the shape agreement, not
        // by the range sweep at the end, and so before anything is computed.
        h                             = base;
        h.codec.decoder.codebook_size = uint32_t(kCodebookSize) + 1;
        SYNTH_TEST_CHECK(refused(h, fixture.weights));
        // More acoustic stages than the cascade carries.
        h                               = base;
        h.codec.decoder.quantizer_count = uint32_t(kAcousticAvailable) + 2;
        SYNTH_TEST_CHECK(refused(h, fixture.weights));
    }

    // An ODD latent width whose tensors agree with it, which is the one
    // configuration `branch_shapes_agree` cannot see -- see the note above.
    {
        constexpr int64_t     kOdd    = kHidden + 1;  // 33
        constexpr int64_t     kHalf   = kOdd / 2;     // 16, the same projected width
        Context               scratch = make_context(ggml_tensor_overhead() * 8);
        ggml_context *        sctx    = scratch.get();
        ggml_tensor *         odd     = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, 1, kOdd, kHalf);
        ggml_backend_buffer_t buffer  = ggml_backend_alloc_ctx_tensors(sctx, backend);
        SYNTH_TEST_CHECK(buffer != nullptr);
        SYNTH_TEST_CHECK(kHalf == kProjected);  // or the codebooks would disagree too

        synth::qwen3tts::HParams h   = base;
        h.codec.decoder.codebook_dim = uint32_t(kOdd);
        Weights weights              = fixture.weights;
        weights.semantic.input_proj  = odd;
        weights.acoustic.input_proj  = odd;
        SYNTH_TEST_CHECK(refused(h, weights));

        ggml_backend_buffer_free(buffer);
    }

    // The resolved-pointer surface.
    {
        Weights weights             = fixture.weights;
        weights.semantic.input_proj = nullptr;
        SYNTH_TEST_CHECK(refused(base, weights));
        weights                     = fixture.weights;
        weights.acoustic.input_proj = nullptr;
        SYNTH_TEST_CHECK(refused(base, weights));
        weights                       = fixture.weights;
        weights.semantic.codebooks[0] = nullptr;
        SYNTH_TEST_CHECK(refused(base, weights));
        weights                        = fixture.weights;
        weights.acoustic.codebooks[14] = nullptr;  // the last stage that is READ
        SYNTH_TEST_CHECK(refused(base, weights));
        weights = fixture.weights;
        weights.semantic.codebooks.clear();
        SYNTH_TEST_CHECK(refused(base, weights));
        // A CustomVoice package resolves nothing at all.
        SYNTH_TEST_CHECK(refused(base, Weights{}));
    }

    // The shape surface. Wrong extents in each of the three positions, and a
    // NON-CONTIGUOUS tensor with otherwise correct extents -- a view with a row
    // stride twice its own width, which is what a future twin or a sliced
    // package would hand over and which the host copy below cannot read.
    {
        Context               scratch    = make_context(ggml_tensor_overhead() * 16);
        ggml_context *        sctx       = scratch.get();
        ggml_tensor *         narrow     = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, 1, kHidden - 2, kProjected);
        ggml_tensor *         thick      = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, 2, kHidden, kProjected);
        ggml_tensor *         shallow    = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, 1, kHidden, kProjected - 1);
        ggml_tensor *         short_book = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, kProjected - 1, kCodebookSize);
        ggml_tensor *         tall_book  = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, kProjected, kCodebookSize + 1);
        ggml_backend_buffer_t buffer     = ggml_backend_alloc_ctx_tensors(sctx, backend);
        SYNTH_TEST_CHECK(buffer != nullptr);
        // Created after the allocation: a view is not allocated, it borrows.
        ggml_tensor * strided = ggml_view_3d(sctx, thick, 1, kHidden, kProjected, thick->nb[1], thick->nb[2], 0);
        SYNTH_TEST_CHECK(strided->ne[0] == 1 && strided->ne[1] == kHidden && strided->ne[2] == kProjected);
        SYNTH_TEST_CHECK(!ggml_is_contiguous(strided));

        Weights weights             = fixture.weights;
        weights.semantic.input_proj = narrow;
        SYNTH_TEST_CHECK(refused(base, weights));
        weights.semantic.input_proj = thick;
        SYNTH_TEST_CHECK(refused(base, weights));
        weights.semantic.input_proj = shallow;
        SYNTH_TEST_CHECK(refused(base, weights));
        weights.semantic.input_proj = strided;
        SYNTH_TEST_CHECK(refused(base, weights));

        weights                       = fixture.weights;
        weights.acoustic.codebooks[3] = short_book;
        SYNTH_TEST_CHECK(refused(base, weights));
        weights.acoustic.codebooks[3] = tall_book;
        SYNTH_TEST_CHECK(refused(base, weights));

        ggml_backend_buffer_free(buffer);
    }
    return 0;
}

// A BF16 codebook is CONVERTED, not reinterpreted.
//
// The header argues at length that this path exists because the graph half
// already widens with `as_f32`, so a package storing this half at BF16 would
// run the convolutions fine and then have its codebook bytes read as float. The
// argument is worth nothing until something drives it, and the fixture is
// all-F32.
//
// The comparison is against an F32 run whose codebooks hold the SAME values
// after a bf16 round trip, so the two differ only in storage. Reinterpreting
// the bytes instead of converting them cannot land on the same codes: a bf16
// bit pattern read as the top half of a float is a different number entirely.
int check_bf16_codebook_is_converted(ggml_backend_t backend) {
    const synth::qwen3tts::HParams hparams = make_hparams();
    const std::vector<float>       pcm     = lcg_clip(kSamplesPerFrame * 5, 0.0f);

    Fixture fixture;
    SYNTH_TEST_CHECK(build_fixture(backend != nullptr ? ggml_backend_get_device(backend) : nullptr, fixture));

    // Every codebook the two branches carry, in one list, so the F32 fixture
    // and the BF16 twin are filled from the identical values.
    std::vector<ggml_tensor *> books;
    for (ggml_tensor * book : fixture.weights.semantic.codebooks) {
        books.push_back(book);
    }
    for (ggml_tensor * book : fixture.weights.acoustic.codebooks) {
        books.push_back(book);
    }

    Context                    scratch = make_context(ggml_tensor_overhead() * (books.size() + 8));
    ggml_context *             sctx    = scratch.get();
    std::vector<ggml_tensor *> twins;
    for (size_t index = 0; index < books.size(); ++index) {
        twins.push_back(ggml_new_tensor_2d(sctx, GGML_TYPE_BF16, kProjected, kCodebookSize));
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(sctx, fixture.backend);
    SYNTH_TEST_CHECK(buffer != nullptr);

    const size_t             count = size_t(kProjected * kCodebookSize);
    std::vector<float>       values(count);
    std::vector<ggml_bf16_t> packed(count);
    std::vector<float>       rounded(count);
    for (size_t index = 0; index < books.size(); ++index) {
        ggml_backend_tensor_get(books[index], values.data(), 0, ggml_nbytes(books[index]));
        ggml_fp32_to_bf16_row(values.data(), packed.data(), int64_t(count));
        ggml_bf16_to_fp32_row(packed.data(), rounded.data(), int64_t(count));
        // The F32 fixture now holds bf16-representable values ...
        ggml_backend_tensor_set(books[index], rounded.data(), 0, ggml_nbytes(books[index]));
        // ... and the twin holds the same values in bf16 storage.
        ggml_backend_tensor_set(twins[index], packed.data(), 0, ggml_nbytes(twins[index]));
    }

    synth::qwen3tts::CodecEncoding f32_encoding;
    const char *                   code    = nullptr;
    const char *                   message = nullptr;
    SYNTH_TEST_CHECK(synth::qwen3tts::encode_codec_reference(hparams, fixture.weights, pcm, 1, f32_encoding, code,
                                                             message) == SYNTH_OK);

    synth::qwen3tts::CodecEncoderWeights bf16 = fixture.weights;
    size_t                               next = 0;
    for (ggml_tensor *& book : bf16.semantic.codebooks) {
        book = twins[next++];
    }
    for (ggml_tensor *& book : bf16.acoustic.codebooks) {
        book = twins[next++];
    }
    synth::qwen3tts::CodecEncoding bf16_encoding;
    SYNTH_TEST_CHECK(synth::qwen3tts::encode_codec_reference(hparams, bf16, pcm, 1, bf16_encoding, code, message) ==
                     SYNTH_OK);

    SYNTH_TEST_CHECK(bf16_encoding.frames == f32_encoding.frames);
    SYNTH_TEST_CHECK(
        synth::qwen3tts::codes_equal(bf16_encoding, f32_encoding.codes.data(), int64_t(f32_encoding.frames)));
    // The reconstruction too, exactly: the selected rows are the same values,
    // so their sums are the same floats. This is what separates "converted" from
    // "converted to something plausible".
    SYNTH_TEST_CHECK(bf16_encoding.reconstruction == f32_encoding.reconstruction);

    ggml_backend_buffer_free(buffer);
    return 0;
}

// Rule 7: a digitally silent reference is refused BY NAME, in the shape
// encode_speaker_reference already uses.
int check_silent_reference_is_refused(const Fixture & fixture) {
    const synth::qwen3tts::HParams hparams = make_hparams();
    const std::vector<float>       silent(size_t(kSamplesPerFrame * 4), 0.0f);
    synth::qwen3tts::CodecEncoding encoding;
    const char *                   diagnostic = nullptr;
    const char *                   message    = nullptr;
    SYNTH_TEST_CHECK(synth::qwen3tts::encode_codec_reference(hparams, fixture.weights, silent, 1, encoding, diagnostic,
                                                             message) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(diagnostic != nullptr);
    SYNTH_TEST_CHECK(std::string(diagnostic) == "voice_profile.reference_silent");
    SYNTH_TEST_CHECK(message != nullptr);
    SYNTH_TEST_CHECK(encoding.codes.empty());

    // An empty clip and a sub-frame clip are refused too, and NOT as silence:
    // they are a different mistake and get a different answer.
    const std::vector<float> empty;
    SYNTH_TEST_CHECK(synth::qwen3tts::encode_codec_reference(hparams, fixture.weights, empty, 1, encoding, diagnostic,
                                                             message) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(diagnostic == nullptr);
    const std::vector<float> stub = lcg_clip(kSamplesPerFrame - 1, 0.0f);
    SYNTH_TEST_CHECK(synth::qwen3tts::encode_codec_reference(hparams, fixture.weights, stub, 1, encoding, diagnostic,
                                                             message) != SYNTH_OK);
    SYNTH_TEST_CHECK(diagnostic == nullptr);
    return 0;
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

        // End-to-end liveness: sample 0 must reach the last frame at all.
        //
        // What it detects is a stack that has stopped propagating -- a severed
        // residual, a mask that blocks everything, an attention that reads only
        // the diagonal. What it is NOT is a reliable window detector, and it is
        // stated that way because both stronger readings have been wrong here.
        // It is not blind to a window: injected windows at this geometry read
        // 1.19e-07 (w=16) and 5.96e-08 (w=15), both of which fail the bound
        // below. But it has no threshold -- it decays smoothly into float32
        // noise, non-monotonically (5.3e-06 at w=32, which passes; 0 at w=16),
        // so what it reports is attenuation and not structure. At the real
        // geometry, where the declared window is 250 over 202 positions, it
        // would detect nothing at all. check_attention_is_unwindowed is the
        // check with an exact threshold; this one is liveness.
        std::vector<float> bumped;
        int64_t            bumped_frames = 0;
        SYNTH_TEST_CHECK(run_encoder(fixture, kSamplesPerFrame * 30, 4.0f, bumped, bumped_frames, nodes));
        SYNTH_TEST_CHECK(nodes == kNodesWholeFrames);
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
        // broken prefix. The dominant term is reduced-mantissa tensor-core F32
        // -- CUDA runs an F32 matmul through TF32, which this project has
        // already recorded as unconditional (docs/backends.md) -- with tiling
        // by column count on top of it, so the 8-frame and 12-frame runs do not
        // reduce the shared columns identically. Accumulation order alone does
        // not produce ~1e-3 relative on a graph this shallow. Same mechanism,
        // same split, as qwen3_tts_codec_test.cpp's own 1e-4 / 2e-2.
        //
        // Measured on a GB10 (sm_121a): 0.003435, stable to every digit across
        // repeated runs, against a signal whose peak is 3.001. That it is
        // arithmetic and not a broken prefix is measurable rather than
        // asserted: the SAME clip encoded on the two backends disagrees by
        // 5.20e-03, MORE than the 3.44e-03 this check sees between two CUDA
        // runs -- so there is no residue left over for a real prefix fault to
        // hide in. A widened tolerance covering a genuine difference would look
        // the opposite way round.
        //
        // Both bounds stay far under the fault they exist to catch: a symmetric
        // pad moves this to ~1.7 on both backends, 17,000x the CPU bound and
        // 86x the accelerator one.
        const float causal_tolerance = type == GGML_BACKEND_DEVICE_TYPE_CPU ? 1e-4f : 2e-2f;
        SYNTH_TEST_CHECK(causal_diff < causal_tolerance);

        ++exercised;
    }
    SYNTH_TEST_CHECK(exercised > 0);

    // The host wrapper and the quantizer, once and on the CPU -- see their own
    // section comment for why they are not inside the device loop.
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    SYNTH_TEST_CHECK(cpu != nullptr);
    std::printf("qwen3-tts-codec-encoder-host: %s\n", ggml_backend_dev_name(cpu));
    {
        Fixture fixture;
        SYNTH_TEST_CHECK(build_fixture(cpu, fixture));
        SYNTH_TEST_CHECK(check_reference_encoding(fixture) == 0);
        SYNTH_TEST_CHECK(check_silent_reference_is_refused(fixture) == 0);
        SYNTH_TEST_CHECK(check_refusals(fixture, fixture.backend) == 0);
    }
    {
        // Its own fixture: the bf16 check rewrites every codebook in place.
        Fixture host;
        SYNTH_TEST_CHECK(build_fixture(cpu, host));
        SYNTH_TEST_CHECK(check_bf16_codebook_is_converted(host.backend) == 0);
    }
    {
        // Its own fixture: check_tie_resolves_to_lowest_id overwrites every
        // codebook, and a later check reading those would be looking at the tie
        // pattern rather than at the drawn one.
        Fixture fixture;
        SYNTH_TEST_CHECK(build_fixture(cpu, fixture));
        SYNTH_TEST_CHECK(check_tie_resolves_to_lowest_id(fixture) == 0);
    }
    return 0;
}
