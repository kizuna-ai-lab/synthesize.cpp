// The whole codec decoder, codes in and waveform out, against the reference
// implementation's own Qwen3TTSTokenizerV2Decoder.
//
// End to end on purpose. Every stage feeds the next, so a length that drifts by
// one, a channel width that halves at the wrong place, or a sliding window that
// reaches one frame too far shows up in the waveform and nowhere earlier. The
// four operators underneath are pinned separately in the codec-ops test.
//
// The decoder is not the shared Qwen3 block: its transformer has no per-head
// norms, scales each residual branch by a learned per-channel vector, and
// attends under a sliding window as well as causally.
//
// Reference values come from scripts/dump_reference_qwen3_tts_codec.py.

#include "arch/qwen3-tts/codec.h"
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

constexpr int64_t  kFrames       = 6;
constexpr int64_t  kQuantizers   = 3;
constexpr int64_t  kSemantic     = 1;
constexpr int64_t  kCodebookSize = 5;
constexpr int64_t  kCodebookDim  = 8;
constexpr int64_t  kLatentDim    = 6;
constexpr int64_t  kDecoderDim   = 8;
constexpr int64_t  kHidden       = 4;
constexpr int64_t  kIntermediate = 6;
constexpr int64_t  kLayers       = 2;
constexpr int64_t  kHeads        = 2;
constexpr int64_t  kHeadDim      = 4;
constexpr int64_t  kWindow       = 3;
constexpr float    kRmsNormEps   = 1e-5f;
constexpr float    kRopeTheta    = 10000.0f;
constexpr uint64_t kSeed         = 20260728u;

// The quantizer works at half the codebook width, which is where the level
// tables live.
constexpr int64_t kInner = kCodebookDim / 2;

// 6 frames x 3 levels, level-major as the graph reads it.
constexpr int32_t kCodes[] = { 0, 1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3, 4 };

// 72 samples: 6 frames at a hop of 12.
constexpr float kExpectedWaveform[] = {
    0.0430629253f,  0.0496943593f,  0.0436964817f,  0.0537883528f, 0.0343213938f, 0.0550069213f,  -0.082467854f,
    -0.0596274622f, 0.0598963052f,  0.0130123179f,  0.337962568f,  0.0279859975f, 0.0683623478f,  -0.0621552467f,
    0.0229360592f,  -0.0729538947f, 0.00223509921f, 0.0241887886f, -0.218194619f, -0.330903053f,  -0.076504536f,
    -0.285010159f,  0.263241231f,   -0.0534071475f, 0.359312177f,  0.325519681f,  -0.206268966f,  -0.0114302905f,
    0.0267978404f,  -0.22471419f,   -0.353043377f,  -0.384356141f, -0.138252422f, -0.276507944f,  0.489850342f,
    -0.0675853714f, 0.42872259f,    0.0550262257f,  -0.103630424f, 0.0320492275f, 0.187197894f,   -0.182592958f,
    0.233706757f,   -0.11331404f,   -0.0645799935f, -0.16113542f,  0.347144365f,  -0.0391040556f, 0.415682077f,
    0.3803415f,     -0.0819592997f, -0.533792496f,  0.0512245595f, -0.682382584f, -0.273802608f,  -0.296162218f,
    -0.220653296f,  0.568420529f,   0.728501797f,   0.134239987f,  1.0f,          0.530099988f,   0.176891223f,
    -0.101559699f,  0.0180694181f,  -0.350834429f,  -0.413589388f, -0.636791766f, -0.157987878f,  0.28005743f,
    0.289192557f,   0.20970346f,
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

constexpr size_t kNodeBudget = 4096;

Context make_context(size_t bytes) {
    ggml_init_params parameters{};
    parameters.mem_size = bytes;
    parameters.no_alloc = true;
    return Context(ggml_init(parameters));
}

synth::qwen3tts::HParams make_hparams() {
    synth::qwen3tts::HParams h;
    h.talker.code_group_count               = uint32_t(kQuantizers);
    h.codec.hop_length                      = 12;
    synth::qwen3tts::CodecDecoderParams & p = h.codec.decoder;
    p.latent_dim                            = uint32_t(kLatentDim);
    p.dim                                   = uint32_t(kDecoderDim);
    p.codebook_dim                          = uint32_t(kCodebookDim);
    p.codebook_size                         = uint32_t(kCodebookSize);
    p.quantizer_count                       = uint32_t(kQuantizers);
    p.semantic_quantizer_count              = uint32_t(kSemantic);
    p.hidden_size                           = uint32_t(kHidden);
    p.intermediate_size                     = uint32_t(kIntermediate);
    p.layer_count                           = uint32_t(kLayers);
    p.attention_head_count                  = uint32_t(kHeads);
    p.key_value_head_count                  = uint32_t(kHeads);
    p.head_dim                              = uint32_t(kHeadDim);
    p.sliding_window                        = uint32_t(kWindow);
    p.rms_norm_eps                          = kRmsNormEps;
    p.rope_theta                            = kRopeTheta;
    p.upsample_rates                        = { 2, 3 };
    p.upsampling_ratios                     = { 2 };
    return h;
}

struct Fixture {
    ggml_backend_t                       backend = nullptr;
    Context                              persistent;
    ggml_backend_buffer_t                buffer = nullptr;
    synth::qwen3tts::CodecDecoderWeights weights;
    ggml_tensor *                        codes     = nullptr;
    ggml_tensor *                        positions = nullptr;
    ggml_tensor *                        mask      = nullptr;

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
    std::vector<float>         offsets;
    auto                       add = [&](ggml_tensor * tensor, float scale, float offset) {
        ordered.push_back(tensor);
        scales.push_back(scale);
        offsets.push_back(offset);
        return tensor;
    };
    auto add_conv = [&](synth::qwen3tts::Conv1dWeights & target, int64_t kernel, int64_t in, int64_t out,
                        float weight_scale = 0.5f, float bias_scale = 0.25f) {
        target.weight = add(ggml_new_tensor_3d(pctx, GGML_TYPE_F32, kernel, in, out), weight_scale, 0.0f);
        target.bias   = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, out), bias_scale, 0.0f);
    };
    auto add_transpose = [&](synth::qwen3tts::Conv1dWeights & target, int64_t kernel, int64_t in, int64_t out) {
        // ConvTranspose1d stores [in, out, kernel], so ggml reports [kernel, out, in].
        target.weight = add(ggml_new_tensor_3d(pctx, GGML_TYPE_F32, kernel, out, in), 0.5f, 0.0f);
        target.bias   = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, out), 0.25f, 0.0f);
    };
    auto add_snake = [&](synth::qwen3tts::SnakeBetaWeights & target, int64_t channels) {
        target.alpha = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, channels), 0.25f, 0.0f);
        target.beta  = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, channels), 0.25f, 0.0f);
    };
    auto add_linear = [&](synth::qwen3tts::LinearWeights & target, int64_t in, int64_t out) {
        target.weight = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, in, out), 0.5f, 0.0f);
        target.bias   = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, out), 0.25f, 0.0f);
    };

    // The fill order is the reference script's. Its codebooks are reconstructed
    // from an accumulator divided by a usage count of one, so the table the test
    // fills is exactly the accumulator the script draws.
    auto add_quantizer = [&](synth::qwen3tts::CodecQuantizerWeights & target, int64_t levels) {
        target.codebooks.assign(size_t(levels), nullptr);
        for (int64_t level = 0; level < levels; ++level) {
            target.codebooks[size_t(level)] =
                add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kInner, kCodebookSize), 0.5f, 0.0f);
        }
        target.output_proj = add(ggml_new_tensor_3d(pctx, GGML_TYPE_F32, 1, kInner, kCodebookDim), 0.5f, 0.0f);
    };
    add_quantizer(fixture.weights.semantic, kSemantic);
    add_quantizer(fixture.weights.acoustic, kQuantizers - kSemantic);

    add_conv(fixture.weights.pre_conv, 3, kCodebookDim, kLatentDim);

    synth::qwen3tts::CodecTransformerWeights & transformer = fixture.weights.pre_transformer;
    add_linear(transformer.input_proj, kLatentDim, kHidden);
    add_linear(transformer.output_proj, kHidden, kLatentDim);
    transformer.layers.resize(size_t(kLayers));
    for (synth::qwen3tts::CodecTransformerLayerWeights & layer : transformer.layers) {
        const int64_t inner            = kHeads * kHeadDim;
        layer.input_layernorm          = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kHidden), 0.25f, 1.0f);
        layer.q_proj                   = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, inner), 0.5f, 0.0f);
        layer.k_proj                   = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, inner), 0.5f, 0.0f);
        layer.v_proj                   = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, inner), 0.5f, 0.0f);
        layer.o_proj                   = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, inner, kHidden), 0.5f, 0.0f);
        layer.self_attn_layer_scale    = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kHidden), 0.1f, 0.2f);
        layer.post_attention_layernorm = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kHidden), 0.25f, 1.0f);
        layer.gate_proj       = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kIntermediate), 0.5f, 0.0f);
        layer.up_proj         = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kIntermediate), 0.5f, 0.0f);
        layer.down_proj       = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kIntermediate, kHidden), 0.5f, 0.0f);
        layer.mlp_layer_scale = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kHidden), 0.1f, 0.2f);
    }
    transformer.norm = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kHidden), 0.25f, 1.0f);

    fixture.weights.upsample.resize(1);
    for (synth::qwen3tts::CodecUpsampleStage & stage : fixture.weights.upsample) {
        add_transpose(stage.transpose_conv, 2, kLatentDim, kLatentDim);
        stage.convnext.dwconv.weight = add(ggml_new_tensor_3d(pctx, GGML_TYPE_F32, 7, 1, kLatentDim), 0.5f, 0.0f);
        stage.convnext.dwconv.bias   = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kLatentDim), 0.25f, 0.0f);
        stage.convnext.norm.weight   = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kLatentDim), 0.25f, 1.0f);
        stage.convnext.norm.bias     = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kLatentDim), 0.25f, 0.0f);
        add_linear(stage.convnext.pwconv1, kLatentDim, 4 * kLatentDim);
        add_linear(stage.convnext.pwconv2, 4 * kLatentDim, kLatentDim);
        stage.convnext.gamma = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kLatentDim), 0.5f, 1.0f);
    }

    add_conv(fixture.weights.input_conv, 7, kLatentDim, kDecoderDim);
    const int64_t rates[] = { 2, 3 };
    fixture.weights.stages.resize(2);
    int64_t width = kDecoderDim;
    for (size_t stage = 0; stage < fixture.weights.stages.size(); ++stage) {
        synth::qwen3tts::CodecResidualStage & residual = fixture.weights.stages[stage];
        const int64_t                         narrower = width / 2;
        add_snake(residual.act, width);
        add_transpose(residual.transpose_conv, 2 * rates[stage], width, narrower);
        residual.units.resize(3);
        for (synth::qwen3tts::CodecResidualUnit & unit : residual.units) {
            add_snake(unit.act1, narrower);
            add_conv(unit.conv1, 7, narrower, narrower);
            add_snake(unit.act2, narrower);
            add_conv(unit.conv2, 1, narrower, narrower);
        }
        width = narrower;
    }
    add_snake(fixture.weights.output_act, width);
    // Drawn small so the waveform stays inside the clamp; a wrong sample clamped
    // to -1 would match a right one clamped to -1.
    add_conv(fixture.weights.output_conv, 7, width, 1, 0.06f, 0.05f);

    // The graph reads one level per row, so the codes are level-major.
    fixture.codes     = ggml_new_tensor_2d(pctx, GGML_TYPE_I32, kFrames, kQuantizers);
    fixture.positions = ggml_new_tensor_1d(pctx, GGML_TYPE_I32, kFrames);
    fixture.mask      = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kFrames, kFrames);

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

    ggml_backend_tensor_set(fixture.codes, kCodes, 0, ggml_nbytes(fixture.codes));
    std::vector<int32_t> sequential(size_t(kFrames), 0);
    for (int64_t index = 0; index < kFrames; ++index) {
        sequential[size_t(index)] = int32_t(index);
    }
    ggml_backend_tensor_set(fixture.positions, sequential.data(), 0, ggml_nbytes(fixture.positions));

    std::vector<float> mask(size_t(kFrames) * kFrames);
    synth::qwen3tts::codec_fill_sliding_window_mask(mask.data(), kFrames, kWindow);
    ggml_backend_tensor_set(fixture.mask, mask.data(), 0, ggml_nbytes(fixture.mask));
    return true;
}

bool run_case(ggml_backend_dev_t device, float & max_diff) {
    Fixture fixture;
    if (!build_fixture(device, fixture)) {
        return false;
    }

    Context graph_ctx =
        make_context(ggml_tensor_overhead() * (kNodeBudget + 64) + ggml_graph_overhead_custom(kNodeBudget, false));
    ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx.get(), kNodeBudget, false);

    ggml_tensor * wav = synth::qwen3tts::build_codec_decoder(graph_ctx.get(), fixture.codes, fixture.positions,
                                                             fixture.mask, fixture.weights, make_hparams());
    if (wav == nullptr) {
        return false;
    }
    // One channel, and exactly the hop's worth of samples per frame.
    if (wav->ne[0] != 1 || wav->ne[1] != kFrames * 12) {
        std::printf("    unexpected waveform shape [%lld, %lld]\n", (long long) wav->ne[0], (long long) wav->ne[1]);
        return false;
    }

    ggml_build_forward_expand(graph, wav);
    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(fixture.backend));
    bool           ok        = allocator != nullptr && ggml_gallocr_alloc_graph(allocator, graph);
    if (ok) {
        ok = ggml_backend_graph_compute(fixture.backend, graph) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        std::vector<float> values(size_t(ggml_nelements(wav)));
        ggml_backend_tensor_get(wav, values.data(), 0, ggml_nbytes(wav));
        max_diff = 0.0f;
        if (values.size() != std::size(kExpectedWaveform)) {
            return false;
        }
        double sum  = 0.0;
        size_t over = 0;
        for (size_t index = 0; index < values.size(); ++index) {
            const float one = std::fabs(values[index] - kExpectedWaveform[index]);
            max_diff        = std::fmax(max_diff, one);
            sum += double(one);
            over += one > 1e-3f ? 1 : 0;
        }
        std::printf("    mean %.3g, over 1e-3: %zu of %zu\n", sum / double(values.size()), over, values.size());
        std::printf("    nodes %d, samples %zu\n", ggml_graph_n_nodes(graph), values.size());
    }
    if (allocator != nullptr) {
        ggml_gallocr_free(allocator);
    }
    return ok;
}

int check_mask() {
    // Causal and no further back than the window: a query at i reads keys in
    // (i - window, i]. Reaching one frame further is inaudible on a short case
    // and wrong on a long one.
    std::vector<float> mask(5 * 5);
    synth::qwen3tts::codec_fill_sliding_window_mask(mask.data(), 5, 3);
    for (int64_t query = 0; query < 5; ++query) {
        for (int64_t key = 0; key < 5; ++key) {
            const bool  visible  = key <= query && key > query - 3;
            const float observed = mask[size_t(query) * 5 + key];
            SYNTH_TEST_CHECK(visible ? observed == 0.0f : std::isinf(observed) && observed < 0.0f);
        }
    }

    // A window of zero is no window at all rather than a query that sees nothing.
    std::vector<float> unbounded(3 * 3);
    synth::qwen3tts::codec_fill_sliding_window_mask(unbounded.data(), 3, 0);
    SYNTH_TEST_CHECK(unbounded[2 * 3 + 0] == 0.0f);
    SYNTH_TEST_CHECK(std::isinf(unbounded[0 * 3 + 2]));
    return 0;
}

int check_rejections() {
    Context        context = make_context(ggml_tensor_overhead() * 64 + 4096);
    ggml_context * ctx     = context.get();

    synth::qwen3tts::CodecQuantizerWeights quantizer;
    quantizer.output_proj = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, kInner, kCodebookDim);
    quantizer.codebooks.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kInner, kCodebookSize));

    // One row of codes per level: a package and a code stream that disagree about
    // the level count would decode a different number of levels than were emitted.
    ggml_tensor * two_levels = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, kFrames, 2);
    SYNTH_TEST_CHECK(synth::qwen3tts::codec_quantizer_decode(ctx, quantizer, two_levels) == nullptr);

    ggml_tensor * float_codes = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kFrames, 1);
    SYNTH_TEST_CHECK(synth::qwen3tts::codec_quantizer_decode(ctx, quantizer, float_codes) == nullptr);

    // A layer missing its scale would run its residual branch at unit gain, which
    // is a hundredfold at this initialisation.
    synth::qwen3tts::CodecTransformerLayerWeights layer;
    ggml_tensor *                                 input     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, kFrames);
    ggml_tensor *                                 positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kFrames);
    ggml_tensor *                                 mask      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kFrames, kFrames);
    synth::qwen3tts::AttentionShape               shape;
    shape.hidden_size          = uint32_t(kHidden);
    shape.attention_head_count = uint32_t(kHeads);
    shape.key_value_head_count = uint32_t(kHeads);
    shape.head_dim             = uint32_t(kHeadDim);
    shape.rms_norm_eps         = kRmsNormEps;
    shape.rope_theta           = kRopeTheta;
    SYNTH_TEST_CHECK(synth::qwen3tts::codec_transformer_layer(ctx, input, positions, mask, layer, shape) == nullptr);

    // The decoder needs a mask; attending over the whole utterance would be a
    // different model, not a slower one.
    synth::qwen3tts::CodecDecoderWeights weights;
    ggml_tensor *                        codes = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, kFrames, kQuantizers);
    SYNTH_TEST_CHECK(synth::qwen3tts::build_codec_decoder(ctx, codes, positions, mask, weights, make_hparams()) ==
                     nullptr);
    return 0;
}

}  // namespace

int main() {
    SYNTH_TEST_CHECK(check_mask() == 0);
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
        std::printf("qwen3-tts-codec: %s (device type %d)\n", ggml_backend_dev_name(device), int(type));
        float max_diff = 0.0f;
        SYNTH_TEST_CHECK(run_case(device, max_diff));
        std::printf("    max_diff %.3g\n", double(max_diff));
        // The CPU bound is the precision check and is tight. The accelerator
        // bound is looser than elsewhere in this family on purpose: this graph is
        // 442 nodes deep with exponentials in it, CUDA runs F32 matmuls through
        // tensor cores at reduced mantissa width, and the resulting error is
        // diffuse -- mean 7e-4 over the waveform with no outlier, against 2e-6 on
        // the CPU for the same graph. A wiring fault is not what this hides: one
        // showed up at 0.98 while this was being written.
        const float tolerance = type == GGML_BACKEND_DEVICE_TYPE_CPU ? 1e-4f : 2e-2f;
        SYNTH_TEST_CHECK(max_diff < tolerance);
        ++exercised;
    }
    SYNTH_TEST_CHECK(exercised > 0);
    return 0;
}
