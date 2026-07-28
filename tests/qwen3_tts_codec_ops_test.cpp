// The codec decoder's convolution stack, compared against the reference
// implementation's own operators.
//
// Every one of these has a way to be wrong that produces audio rather than an
// error, which is why they are pinned individually before the decoder is
// assembled from them:
//
// - the causal convolution pads on the **left only**, and ggml pads
//   symmetrically, so the port convolves wide and keeps the prefix
// - the depthwise form has a kernel whose input extent is one rather than the
//   channel count, and ggml's own depthwise convolution carries an upstream
//   "very likely wrong for some cases" warning
// - the transposed form drops the trailing `kernel - stride` samples; cropping
//   the other end is still causal-looking
// - SnakeBeta stores both curves as logarithms, so neither is near one
//
// Reference values come from scripts/dump_reference_qwen3_tts_codec_ops.py.

#include "arch/qwen3-tts/codec.h"
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

constexpr int64_t  kLength      = 7;
constexpr int64_t  kChannels    = 4;
constexpr int64_t  kOutChannels = 6;
constexpr int64_t  kKernel      = 3;
constexpr int      kDilation    = 2;
constexpr int      kStride      = 2;
constexpr uint64_t kSeed        = 20260728u;

// ne = [6 channels, 7 length].
constexpr float kExpectedCausalConv[] = {
    -0.044349093f, -0.113175407f, -0.0532087423f, 0.357307076f,
    0.225883022f, 0.10605213f, -0.0804323256f, -0.0914198756f,
    -0.330780745f, 0.121067591f, -0.293832153f, 0.0818357095f,
    0.319016904f, -0.0427661352f, -0.0826792866f, 0.330877572f,
    -0.00236040354f, -0.104884334f, -0.187254056f, 0.0129179731f,
    -0.440484613f, 0.196778163f, 0.068801932f, 0.0170272384f,
    0.231575072f, -0.315061659f, -0.109611161f, 0.429358721f,
    0.0634099245f, -0.272652298f, -0.371566862f, 0.312565029f,
    -0.0644516051f, -0.253964871f, 0.721820414f, 0.0443798751f,
    0.24401167f, 0.00472286716f, -0.123672083f, 0.431849808f,
    0.283916533f, -0.358228385f,
};

// ne = [4 channels, 7 length].
constexpr float kExpectedDepthwise[] = {
    -0.110034592f, -0.164307237f, -0.164613813f, -0.0801490396f,
    -0.0955354795f, -0.205003217f, -0.190139636f, -0.266549885f,
    -0.0687134713f, -0.178187236f, -0.339178383f, -0.054378029f,
    -0.189022914f, -0.0767542943f, -0.0484885909f, 0.043512255f,
    0.0629738122f, -0.161004007f, -0.311598599f, 0.0872185677f,
    -0.439835906f, -0.118405335f, -0.130255714f, -0.0524706952f,
    0.0108251609f, -0.136212245f, -0.120674387f, -0.36198777f,
};

// ne = [6 channels, 14 length].
constexpr float kExpectedTransposed[] = {
    -0.185159475f, 0.294198006f, -0.3488608f, 0.399559528f,
    -0.115084536f, -0.155954868f, 0.210257739f, 0.243169487f,
    -0.231953323f, 0.286713481f, -0.303467155f, -0.151800901f,
    0.468250245f, 0.397378683f, -0.363762021f, -0.243587315f,
    -0.452624351f, -0.0850169212f, 0.0366950631f, 0.312010765f,
    -0.326329172f, 0.283921391f, -0.325410426f, -0.229189426f,
    -0.0458339676f, 0.0441707373f, -0.120846435f, 0.26471436f,
    0.105467021f, -0.161327124f, -0.0450076759f, -0.264992744f,
    -0.465083718f, 0.556341648f, -0.178614214f, 0.00243459642f,
    0.0298268571f, -0.204512745f, -0.0544818193f, 0.0391616374f,
    -0.240094557f, -0.464788556f, -0.292717189f, 0.0151424259f,
    -0.163343504f, -0.00114241242f, -0.00104613602f, 0.0131613612f,
    0.0433805846f, 0.374323159f, -0.468574762f, 0.124293894f,
    -0.163627997f, 0.0110455155f, 0.329000503f, 0.35879451f,
    -0.380473644f, 0.0751181692f, -0.276529878f, -0.27585572f,
    -0.086698547f, -0.305549145f, -0.210307121f, 0.905746937f,
    0.00916711986f, 0.119378388f, -0.109759197f, -0.0641605258f,
    0.0637532473f, 0.243605196f, -0.193075776f, -0.0104810148f,
    0.0492940322f, 0.757307947f, -0.384620816f, 0.193260342f,
    -0.532383502f, -0.362530947f, 0.358118802f, 0.800749063f,
    0.15528664f, 0.115284026f, -0.353857428f, -0.635562778f,
};

// ne = [4 channels, 7 length].
constexpr float kExpectedSnakeBeta[] = {
    0.211418718f, -0.211093172f, 0.340142608f, 0.0482888743f,
    0.374764264f, 0.557753801f, -0.124428213f, -0.318217635f,
    0.270131081f, -0.0398793221f, 0.302310228f, -0.267833978f,
    -0.354219884f, -0.0402596407f, -0.100752652f, -0.0567697249f,
    0.549094796f, 0.490957767f, 0.0119910855f, 0.370509863f,
    -0.354136705f, -0.213347778f, 0.149043277f, 0.568386793f,
    -0.0294143446f, 0.192955092f, 0.547572315f, -0.00632938929f,
};

// ne = [4 channels, 7 length].
constexpr float kExpectedConvNeXt[] = {
    0.348275661f, -0.42144388f, -0.763700426f, 0.145622522f,
    0.879683495f, 0.751017928f, 0.273373425f, -0.420851588f,
    0.645706713f, -0.0742798895f, -0.749853969f, -0.0716668367f,
    -0.308953762f, -0.130112231f, -1.11264944f, 0.0203875005f,
    0.60067457f, 0.509364724f, -0.230222121f, 0.251232028f,
    -0.501762629f, -0.498339742f, -0.78526181f, 0.478380471f,
    0.787209988f, 0.248576686f, -0.0628299415f, 0.135890499f,
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

constexpr size_t kNodeBudget = 512;

Context make_context(size_t bytes) {
    ggml_init_params parameters{};
    parameters.mem_size = bytes;
    parameters.no_alloc = true;
    return Context(ggml_init(parameters));
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

struct Fixture {
    ggml_backend_t        backend = nullptr;
    Context               persistent;
    ggml_backend_buffer_t buffer = nullptr;

    ggml_tensor * signal = nullptr;

    synth::qwen3tts::Conv1dWeights    conv;
    synth::qwen3tts::Conv1dWeights    depthwise;
    synth::qwen3tts::Conv1dWeights    transposed;
    synth::qwen3tts::SnakeBetaWeights snake;
    synth::qwen3tts::ConvNeXtWeights  convnext;

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
    fixture.persistent  = make_context(ggml_tensor_overhead() * 64);
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

    // The fill order is the reference script's, and the signal comes first.
    fixture.signal = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kChannels, kLength), 0.5f, 0.0f);

    fixture.conv.weight = add(ggml_new_tensor_3d(pctx, GGML_TYPE_F32, kKernel, kChannels, kOutChannels), 0.5f, 0.0f);
    fixture.conv.bias   = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kOutChannels), 0.25f, 0.0f);

    fixture.depthwise.weight = add(ggml_new_tensor_3d(pctx, GGML_TYPE_F32, kKernel, 1, kChannels), 0.5f, 0.0f);
    fixture.depthwise.bias   = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kChannels), 0.25f, 0.0f);

    // ConvTranspose1d stores [in, out, kernel], so ggml reports [kernel, out, in].
    fixture.transposed.weight =
        add(ggml_new_tensor_3d(pctx, GGML_TYPE_F32, 2 * kStride, kOutChannels, kChannels), 0.5f, 0.0f);
    fixture.transposed.bias = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kOutChannels), 0.25f, 0.0f);

    fixture.snake.alpha = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kChannels), 0.5f, 0.0f);
    fixture.snake.beta  = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kChannels), 0.5f, 0.0f);

    fixture.convnext.dwconv.weight = add(ggml_new_tensor_3d(pctx, GGML_TYPE_F32, 7, 1, kChannels), 0.5f, 0.0f);
    fixture.convnext.dwconv.bias   = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kChannels), 0.25f, 0.0f);
    fixture.convnext.norm.weight   = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kChannels), 0.25f, 1.0f);
    fixture.convnext.norm.bias     = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kChannels), 0.25f, 0.0f);
    fixture.convnext.pwconv1.weight =
        add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kChannels, 4 * kChannels), 0.5f, 0.0f);
    fixture.convnext.pwconv1.bias = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, 4 * kChannels), 0.25f, 0.0f);
    fixture.convnext.pwconv2.weight =
        add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, 4 * kChannels, kChannels), 0.5f, 0.0f);
    fixture.convnext.pwconv2.bias = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kChannels), 0.25f, 0.0f);
    fixture.convnext.gamma        = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kChannels), 0.5f, 1.0f);

    fixture.buffer = ggml_backend_alloc_ctx_tensors(pctx, fixture.backend);
    if (fixture.buffer == nullptr) {
        return false;
    }

    LcgStream stream(kSeed);
    for (size_t index = 0; index < ordered.size(); ++index) {
        std::vector<float> values =
            stream.fill(size_t(ggml_nelements(ordered[index])), scales[index], offsets[index]);
        if (ordered[index] == fixture.signal) {
            // The reference draws the signal channel-major in its own sense --
            // [channels, length] with time contiguous. This layout has channels
            // contiguous instead, so the draw is transposed on the way in.
            std::vector<float> laid_out(values.size());
            for (int64_t channel = 0; channel < kChannels; ++channel) {
                for (int64_t time = 0; time < kLength; ++time) {
                    laid_out[size_t(time) * kChannels + channel] = values[size_t(channel) * kLength + time];
                }
            }
            values = laid_out;
        }
        ggml_backend_tensor_set(ordered[index], values.data(), 0, ggml_nbytes(ordered[index]));
    }
    return true;
}

bool run_case(ggml_backend_dev_t device, float & max_diff) {
    Fixture fixture;
    if (!build_fixture(device, fixture)) {
        return false;
    }

    Context graph_ctx = make_context(ggml_tensor_overhead() * (kNodeBudget + 64) +
                                     ggml_graph_overhead_custom(kNodeBudget, false));
    ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx.get(), kNodeBudget, false);
    ggml_context * ctx  = graph_ctx.get();

    ggml_tensor * conv = synth::qwen3tts::codec_causal_conv1d(ctx, fixture.signal, fixture.conv, kDilation);
    ggml_tensor * depthwise = synth::qwen3tts::codec_causal_depthwise_conv1d(ctx, fixture.signal, fixture.depthwise);
    ggml_tensor * transposed =
        synth::qwen3tts::codec_causal_transpose_conv1d(ctx, fixture.signal, fixture.transposed, kStride);
    ggml_tensor * snake    = synth::qwen3tts::codec_snake_beta(ctx, fixture.signal, fixture.snake);
    ggml_tensor * convnext = synth::qwen3tts::codec_convnext_block(ctx, fixture.signal, fixture.convnext);
    if (conv == nullptr || depthwise == nullptr || transposed == nullptr || snake == nullptr ||
        convnext == nullptr) {
        return false;
    }

    // A causal convolution keeps the input's length; a transposed one at stride s
    // multiplies it.
    if (conv->ne[1] != kLength || conv->ne[0] != kOutChannels || depthwise->ne[1] != kLength ||
        depthwise->ne[0] != kChannels || transposed->ne[1] != kLength * kStride ||
        transposed->ne[0] != kOutChannels || snake->ne[1] != kLength || convnext->ne[1] != kLength) {
        return false;
    }

    const std::vector<ggml_tensor *> outputs = { conv, depthwise, transposed, snake, convnext };
    for (ggml_tensor * output : outputs) {
        // Intermediates are read back, so the allocator must not reuse them.
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
    }

    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(fixture.backend));
    bool           ok        = allocator != nullptr && ggml_gallocr_alloc_graph(allocator, graph);
    if (ok) {
        ok = ggml_backend_graph_compute(fixture.backend, graph) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        const float * expected[] = { kExpectedCausalConv, kExpectedDepthwise, kExpectedTransposed,
                                     kExpectedSnakeBeta, kExpectedConvNeXt };
        const size_t  counts[]   = { std::size(kExpectedCausalConv), std::size(kExpectedDepthwise),
                                     std::size(kExpectedTransposed), std::size(kExpectedSnakeBeta),
                                     std::size(kExpectedConvNeXt) };
        const char *  labels[]   = { "conv", "depthwise", "transposed", "snake", "convnext" };
        max_diff                 = 0.0f;
        for (size_t index = 0; index < outputs.size(); ++index) {
            std::vector<float> values(size_t(ggml_nelements(outputs[index])));
            ggml_backend_tensor_get(outputs[index], values.data(), 0, ggml_nbytes(outputs[index]));
            const float one = deviation(values, expected[index], counts[index]);
            std::printf("    %-11s %.3g\n", labels[index], double(one));
            max_diff = std::fmax(max_diff, one);
        }
    }
    if (allocator != nullptr) {
        ggml_gallocr_free(allocator);
    }
    return ok;
}

int check_rejections() {
    Context        context = make_context(ggml_tensor_overhead() * 64 + 4096);
    ggml_context * ctx     = context.get();

    ggml_tensor * signal = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kLength, kChannels);

    synth::qwen3tts::Conv1dWeights conv;
    conv.weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kKernel, kChannels, kOutChannels);
    conv.bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kOutChannels);

    // A convolution whose kernel expects a different channel count than the
    // signal carries is a wiring defect, not something to broadcast.
    synth::qwen3tts::Conv1dWeights mismatched = conv;
    mismatched.weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kKernel, kChannels + 1, kOutChannels);
    SYNTH_TEST_CHECK(synth::qwen3tts::codec_causal_conv1d(ctx, signal, mismatched, 1) == nullptr);

    // A dropped bias would be silently omitted.
    synth::qwen3tts::Conv1dWeights unbiased = conv;
    unbiased.bias                           = nullptr;
    SYNTH_TEST_CHECK(synth::qwen3tts::codec_causal_conv1d(ctx, signal, unbiased, 1) == nullptr);
    SYNTH_TEST_CHECK(synth::qwen3tts::codec_causal_conv1d(ctx, signal, conv, 0) == nullptr);

    // The depthwise form's kernel must have an input extent of one; passing the
    // dense kernel would convolve every channel against every other.
    SYNTH_TEST_CHECK(synth::qwen3tts::codec_causal_depthwise_conv1d(ctx, signal, conv) == nullptr);

    // A transposed kernel narrower than its stride cannot be cropped causally.
    synth::qwen3tts::Conv1dWeights narrow;
    narrow.weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, kOutChannels, kChannels);
    narrow.bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kOutChannels);
    SYNTH_TEST_CHECK(synth::qwen3tts::codec_causal_transpose_conv1d(ctx, signal, narrow, kStride) == nullptr);

    // SnakeBeta's curves are per channel, so a curve of the wrong width would
    // broadcast into the time axis instead.
    synth::qwen3tts::SnakeBetaWeights snake;
    snake.alpha = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kChannels + 1);
    snake.beta  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kChannels + 1);
    SYNTH_TEST_CHECK(synth::qwen3tts::codec_snake_beta(ctx, signal, snake) == nullptr);

    synth::qwen3tts::ConvNeXtWeights convnext;
    SYNTH_TEST_CHECK(synth::qwen3tts::codec_convnext_block(ctx, signal, convnext) == nullptr);
    return 0;
}

}  // namespace

int main() {
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
        std::printf("qwen3-tts-codec-ops: %s (device type %d)\n", ggml_backend_dev_name(device), int(type));
        float max_diff = 0.0f;
        SYNTH_TEST_CHECK(run_case(device, max_diff));
        const float tolerance = type == GGML_BACKEND_DEVICE_TYPE_CPU ? 1e-4f : 5e-3f;
        SYNTH_TEST_CHECK(max_diff < tolerance);
        ++exercised;
    }
    SYNTH_TEST_CHECK(exercised > 0);
    return 0;
}
