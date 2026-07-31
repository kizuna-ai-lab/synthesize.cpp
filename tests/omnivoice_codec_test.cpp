// The Higgs Audio V2 decode graph, compared against the real transformers
// DacDecoder with the real Higgs adjustment applied.
//
// Three things are pinned here.
//
// The **two Higgs adjustments**. `_adjust_dac_decoder` sets
// `output_padding = stride % 2` on every ConvTranspose1d and replaces the final
// `Tanh` with `Identity`. The first is load-bearing for the shape: with an odd
// ratio, plain DAC padding produces one sample too few, so the stride-3 block
// would leave 23 samples where 24 are owed and the builder's own length check
// would refuse. The second is load-bearing for the values: a port that kept the
// tanh would still produce audio, just quieter and squashed, which no shape
// catches. Both are applied by the upstream staticmethod in the reference
// script rather than restated there, so the numbers move if upstream moves.
//
// The **absence of causality**. qwen3-tts's codec is causal and every one of
// its convolutions pads wide and keeps the prefix. This one is not: DAC pads
// symmetrically, so a crop copied over from that family would shift the
// waveform in time -- audible, and invisible to every shape check. The wave
// array below is the only thing that catches it.
//
// The **RVQ order**. Levels are summed 0..n in ascending order, since float
// addition is not associative and the reference sums in that order; and the
// projection is a biased Linear rather than the kernel-1 convolution most RVQ
// ports use.
//
// Weights are not carried in either file. Both sides draw them from the same
// 64-bit LCG in the same order, so only the outputs are pinned.

#include "arch/omnivoice/catalog.h"
#include "arch/omnivoice/codec-host.h"
#include "arch/omnivoice/codec.h"
#include "arch/omnivoice/weights.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "test-assert.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <limits>
#include <memory>
#include <vector>

namespace {

constexpr uint32_t kQuantizers    = 2;
constexpr uint32_t kCodebookSize  = 5;
constexpr uint32_t kCodebookDim   = 3;
constexpr uint32_t kConcat        = 4;  // RVQ width = fc2 input
constexpr uint32_t kAcoustic      = 4;  // fc2 output = decoder input channels
constexpr uint32_t kDecoderHidden = 8;
constexpr uint32_t kFrames        = 4;
// One even ratio and one odd one. The odd one is the point: output_padding is
// stride % 2, so a block that ignored it would only misbehave here.
constexpr uint32_t kRatios[2]     = { 2, 3 };
constexpr uint32_t kHop           = kRatios[0] * kRatios[1];
constexpr uint32_t kSamples       = kFrames * kHop;
constexpr uint32_t kDacUnits      = 3;
constexpr uint64_t kSeed          = 20260732u;

// Accelerators run F32 matmuls through tensor cores at reduced mantissa width,
// so they are held to a looser bound than the CPU.
constexpr float kCpuTolerance         = 1e-4f;
constexpr float kAcceleratorTolerance = 5e-3f;

// Level-major, the reference script's CODES flattened.
constexpr int32_t kCodes[kQuantizers * kFrames] = { 0, 3, 1, 4, 2, 2, 0, 1 };

// values from scripts/dump_reference_omnivoice_codec.py, pasted verbatim, at
// transformers 5.14.1, torch 2.13.0+cu130. That run's two self-checks: a
// surviving tanh would move the wave by 0.873 and a one-sample time shift by
// 2.94, both far outside even the accelerator tolerance below.

// 4 frames x 4 concat channels, frame-major -- what a contiguous ggml
// [concat, frames] tensor reads back as.
constexpr float kExpectedLatent[] = {
    0.193872139f,   0.405090392f, 0.289982051f,   0.49864012f,    -0.0147755742f, 0.375171125f,
    0.275515437f,   0.119079873f, 0.302843034f,   -0.0857183561f, 0.376214862f,   -0.312757999f,
    -0.0189586915f, 0.413028091f, -0.0458257645f, 0.317769408f,
};

// After fc2: 4 frames x 4 acoustic channels, same order.
constexpr float kExpectedAcoustic[] = {
    0.20535697f,   0.307556748f,  0.14337343f,   0.158771276f,   0.242736846f,  0.185659707f,
    0.0645200908f, 0.0863374546f, 0.0916352943f, -0.0468924195f, -0.146364376f, -0.0484940186f,
    0.227833569f,  0.121916659f,  0.107762441f,  0.00949309394f,
};

// The raw decoder output: 24 samples, no tanh, no clamp, no normalization.
// Peak-normalizing to 0.5 is the caller's separate branch, not the graph's --
// which is also why samples past +-1 here are not a fixture defect: with the
// tanh gone the reference's own output is unbounded.
constexpr float kExpectedWave[] = {
    0.984655082f,  1.63933909f,   -1.30004716f,  -0.831128359f,  1.82161975f,   -1.08492458f,
    -1.04619563f,  -0.143402547f, -0.236692294f, 0.476798683f,   0.0315622427f, 0.708551824f,
    -0.119919963f, -0.392588794f, -0.542761266f, -0.433344603f,  0.134225845f,  -0.155875832f,
    -0.983499646f, -0.118334264f, 1.73286819f,   -0.0957114175f, -1.35010052f,  -0.710008681f,
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

synth::omnivoice::HParams make_hparams() {
    synth::omnivoice::HParams hparams;
    hparams.audio.num_codebooks       = kQuantizers;
    hparams.audio.vocab_size          = kCodebookSize + 1;  // codes plus the mask id
    hparams.audio.mask_id             = kCodebookSize;
    hparams.codec.hidden_size         = kAcoustic;
    hparams.codec.decoder_hidden_size = kDecoderHidden;
    hparams.codec.hop_length          = kHop;
    hparams.codec.codebook_dim        = kCodebookDim;
    hparams.codec.codebook_size       = kCodebookSize;
    hparams.codec.upsampling_ratios.assign(std::begin(kRatios), std::end(kRatios));
    // The concat width the RVQ works at is codec.hidden_size + semantic.hidden_size,
    // and the toy fixture puts the whole width on the acoustic side.
    hparams.semantic.hidden_size = 0;
    return hparams;
}

struct Fixture {
    ggml_backend_t                 backend = nullptr;
    Context                        persistent;
    ggml_backend_buffer_t          buffer = nullptr;
    synth::omnivoice::ModelWeights weights;
    ggml_tensor *                  codes = nullptr;

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

bool build_fixture(ggml_backend_dev_t device, Fixture & fixture) {
    fixture.backend = ggml_backend_dev_init(device, nullptr);
    if (fixture.backend == nullptr) {
        return false;
    }
    fixture.persistent  = make_context(ggml_tensor_overhead() * 128);
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
    // The three roles the reference script draws with (its WEIGHT_SCALE,
    // BIAS_SCALE and ALPHA_SCALE/ALPHA_OFFSET), so a line here reads as the
    // role rather than as two magic numbers.
    auto weight = [&](ggml_tensor * tensor) {
        return add(tensor, 0.5f, 0.0f);
    };
    auto bias = [&](ggml_tensor * tensor) {
        return add(tensor, 0.1f, 0.0f);
    };
    auto alpha = [&](ggml_tensor * tensor) {
        return add(tensor, 0.25f, 1.0f);
    };

    // ggml shapes are the torch shapes reversed: a Conv1d weight [out, in, k]
    // becomes [k, in, out], a ConvTranspose1d weight [in, out, k] becomes
    // [k, out, in], a Linear weight [out, in] becomes [in, out], and Snake's
    // alpha keeps its [1, channels, 1] rank.
    auto conv = [&](synth::omnivoice::Conv1dWeights & target, int64_t kernel, int64_t in, int64_t out) {
        target.weight = weight(ggml_new_tensor_3d(pctx, GGML_TYPE_F32, kernel, in, out));
        target.bias   = bias(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, out));
    };
    auto transpose_conv = [&](synth::omnivoice::Conv1dWeights & target, int64_t kernel, int64_t in, int64_t out) {
        target.weight = weight(ggml_new_tensor_3d(pctx, GGML_TYPE_F32, kernel, out, in));
        target.bias   = bias(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, out));
    };
    auto snake = [&](synth::omnivoice::SnakeWeights & target, int64_t channels) {
        target.alpha = alpha(ggml_new_tensor_3d(pctx, GGML_TYPE_F32, 1, channels, 1));
    };

    // The fill order is the reference script's, module by module in forward
    // order. Reordering any line silently changes every weight after it.
    synth::omnivoice::ModelWeights & w = fixture.weights;
    w.quantizers.resize(kQuantizers);
    for (uint32_t level = 0; level < kQuantizers; ++level) {
        synth::omnivoice::RvqQuantizerWeights & quantizer = w.quantizers[level];
        quantizer.codebook           = weight(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kCodebookDim, kCodebookSize));
        quantizer.output_proj.weight = weight(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kCodebookDim, kConcat));
        quantizer.output_proj.bias   = bias(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kConcat));
    }
    w.fc2.weight = weight(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kConcat, kAcoustic));
    w.fc2.bias   = bias(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kAcoustic));

    synth::omnivoice::AcousticDecoderWeights & decoder = w.acoustic_decoder;
    conv(decoder.conv1, 7, kAcoustic, kDecoderHidden);
    int64_t width = kDecoderHidden;
    decoder.blocks.resize(std::size(kRatios));
    for (size_t index = 0; index < decoder.blocks.size(); ++index) {
        synth::omnivoice::AcousticDecoderBlock & block    = decoder.blocks[index];
        const int64_t                            narrower = width / 2;
        snake(block.snake1, width);
        transpose_conv(block.conv_t1, 2 * int64_t(kRatios[index]), width, narrower);
        block.res_units.resize(kDacUnits);
        for (synth::omnivoice::DacResidualUnit & unit : block.res_units) {
            snake(unit.snake1, narrower);
            conv(unit.conv1, 7, narrower, narrower);
            snake(unit.snake2, narrower);
            conv(unit.conv2, 1, narrower, narrower);
        }
        width = narrower;
    }
    snake(decoder.snake1, width);
    conv(decoder.conv2, 7, width, 1);

    fixture.codes = ggml_new_tensor_2d(pctx, GGML_TYPE_I32, kFrames, kQuantizers);

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

// The dequantizer alone, so a wrong sum order or a dropped projection bias is
// attributable before the decoder stack has a chance to smear it.
bool check_rvq(Fixture & fixture, float & worst) {
    Context       graph_ctx = make_graph_context();
    ggml_cgraph * graph     = ggml_new_graph_custom(graph_ctx.get(), kNodeBudget, false);

    ggml_tensor * latent =
        synth::omnivoice::codec_rvq_decode(graph_ctx.get(), fixture.weights.quantizers, fixture.codes);
    if (latent == nullptr || latent->ne[0] != int64_t(kConcat) || latent->ne[1] != int64_t(kFrames)) {
        return false;
    }

    std::vector<std::vector<float>> values;
    if (!compute(fixture, graph, { latent }, values)) {
        return false;
    }
    worst = deviation(values[0], kExpectedLatent, std::size(kExpectedLatent));
    std::printf("  rvq %.3g\n", double(worst));
    return true;
}

// The whole decode, with the builder's own RVQ and fc2 tensors probed so a
// failure lands on a stage rather than on "the waveform is wrong".
bool check_decode(Fixture & fixture, float & worst) {
    Context       graph_ctx = make_graph_context();
    ggml_cgraph * graph     = ggml_new_graph_custom(graph_ctx.get(), kNodeBudget, false);

    ggml_tensor * latent   = nullptr;
    ggml_tensor * acoustic = nullptr;
    ggml_tensor * wave     = synth::omnivoice::build_codec_decoder(graph_ctx.get(), fixture.codes, fixture.weights,
                                                                   make_hparams(), &latent, &acoustic);
    if (wave == nullptr || latent == nullptr || acoustic == nullptr) {
        // The commonest way to land here is the length assertion at the end of
        // the builder, which is what a missing output_padding trips.
        std::printf("  decode refused to build\n");
        return false;
    }
    // One dimension, exactly hop samples per frame. The odd ratio is what makes
    // this a real check: without output_padding the stride-3 block loses a
    // sample and this count comes out at 23.
    if (ggml_n_dims(wave) != 1 || wave->ne[0] != int64_t(kSamples)) {
        std::printf("  decode produced %lld samples, expected %u\n", (long long) wave->ne[0], kSamples);
        return false;
    }
    if (acoustic->ne[0] != int64_t(kAcoustic) || acoustic->ne[1] != int64_t(kFrames)) {
        return false;
    }

    std::vector<std::vector<float>> values;
    if (!compute(fixture, graph, { latent, acoustic, wave }, values)) {
        return false;
    }
    const float d0 = deviation(values[0], kExpectedLatent, std::size(kExpectedLatent));
    const float d1 = deviation(values[1], kExpectedAcoustic, std::size(kExpectedAcoustic));
    const float d2 = deviation(values[2], kExpectedWave, std::size(kExpectedWave));
    std::printf("  latent %.3g  acoustic %.3g  wave %.3g\n", double(d0), double(d1), double(d2));
    worst = std::fmax(std::fmax(d0, d1), d2);
    return true;
}

bool run_case(ggml_backend_dev_t device, float & max_diff) {
    Fixture fixture;
    if (!build_fixture(device, fixture)) {
        return false;
    }
    float rvq    = 0.0f;
    float decode = 0.0f;
    if (!check_rvq(fixture, rvq) || !check_decode(fixture, decode)) {
        return false;
    }
    max_diff = std::fmax(rvq, decode);
    return true;
}

// A shape the builders cannot serve is a wiring defect, so they return nullptr
// rather than aborting inside ggml on an assertion the caller cannot catch.
int check_rejections() {
    Context        context = make_context(ggml_tensor_overhead() * 256 + 4096);
    ggml_context * ctx     = context.get();
    SYNTH_TEST_CHECK(ctx != nullptr);

    std::vector<synth::omnivoice::RvqQuantizerWeights> quantizers(kQuantizers);
    for (synth::omnivoice::RvqQuantizerWeights & quantizer : quantizers) {
        quantizer.codebook           = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kCodebookDim, kCodebookSize);
        quantizer.output_proj.weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kCodebookDim, kConcat);
        quantizer.output_proj.bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kConcat);
    }
    ggml_tensor * codes = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, kFrames, kQuantizers);

    // Codes are indices; an F32 grid would be read as garbage rows.
    ggml_tensor * float_codes = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kFrames, kQuantizers);
    SYNTH_TEST_CHECK(synth::omnivoice::codec_rvq_decode(ctx, quantizers, float_codes) == nullptr);

    // One row per level, no more and no fewer: a grid with a spare row would
    // have that row's codes silently dropped.
    ggml_tensor * wide_codes = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, kFrames, kQuantizers + 1);
    SYNTH_TEST_CHECK(synth::omnivoice::codec_rvq_decode(ctx, quantizers, wide_codes) == nullptr);

    // A level whose projection never resolved.
    std::vector<synth::omnivoice::RvqQuantizerWeights> unbound = quantizers;
    unbound[1].output_proj.bias                                = nullptr;
    SYNTH_TEST_CHECK(synth::omnivoice::codec_rvq_decode(ctx, unbound, codes) == nullptr);

    // Higgs's output_padding is stride % 2 and its padding ceil(stride / 2), so
    // output_padding never exceeds padding; the crop recipe cannot represent it
    // if it ever did, and guessing would put the samples in the wrong place.
    synth::omnivoice::Conv1dWeights conv_t;
    conv_t.weight        = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 4, 8);
    conv_t.bias          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor * signal = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, kFrames);
    SYNTH_TEST_CHECK(synth::omnivoice::codec_transpose_conv1d(ctx, signal, conv_t, 2, 1, 0) != nullptr);
    SYNTH_TEST_CHECK(synth::omnivoice::codec_transpose_conv1d(ctx, signal, conv_t, 2, 1, 2) == nullptr);

    // A bias that is not one value per output channel would broadcast onto the
    // wrong ones; ggml_add would abort on it rather than return.
    synth::omnivoice::Conv1dWeights narrow_bias = conv_t;
    narrow_bias.bias                            = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
    SYNTH_TEST_CHECK(synth::omnivoice::codec_transpose_conv1d(ctx, signal, narrow_bias, 2, 1, 0) == nullptr);
    synth::omnivoice::Conv1dWeights forward;
    forward.weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 7, 8, 8);
    forward.bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 7);
    SYNTH_TEST_CHECK(synth::omnivoice::codec_conv1d(ctx, signal, forward, 1, 3) == nullptr);

    // A curve whose width is not the signal's would broadcast onto the wrong
    // channels, and a missing one is an unresolved package.
    synth::omnivoice::SnakeWeights snake;
    SYNTH_TEST_CHECK(synth::omnivoice::codec_snake(ctx, signal, snake) == nullptr);
    snake.alpha = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 4, 1);
    SYNTH_TEST_CHECK(synth::omnivoice::codec_snake(ctx, signal, snake) == nullptr);

    // The right width at the wrong rank: [channels, channels, 1] has the same
    // ne[1] the check reads, and reshaping it to one value per channel would
    // silently drop most of it.
    synth::omnivoice::SnakeWeights fat;
    fat.alpha = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 8, 1);
    SYNTH_TEST_CHECK(synth::omnivoice::codec_snake(ctx, signal, fat) == nullptr);

    // The whole builder refuses a grid whose row count disagrees with the
    // package's codebook count rather than decoding a truncated one.
    synth::omnivoice::ModelWeights weights;
    weights.quantizers                    = quantizers;
    weights.fc2.weight                    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kConcat, kAcoustic);
    weights.fc2.bias                      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kAcoustic);
    weights.acoustic_decoder.conv1.weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 7, kAcoustic, kDecoderHidden);
    weights.acoustic_decoder.conv1.bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kDecoderHidden);
    SYNTH_TEST_CHECK(synth::omnivoice::build_codec_decoder(ctx, wide_codes, weights, make_hparams()) == nullptr);

    // A decoder whose block count disagrees with the upsampling ratios: the
    // ratio a block upsamples by comes from the hyper-parameters, so a package
    // that resolved a different number of them would read past the list. This
    // is the EMPTY half of the compound condition (`decoder.blocks.empty() ||
    // decoder.blocks.size() != ratios.size()`) -- weights.acoustic_decoder.blocks
    // was never assigned above.
    SYNTH_TEST_CHECK(synth::omnivoice::build_codec_decoder(ctx, codes, weights, make_hparams()) == nullptr);

    // The NON-empty half of that same condition: a resolved block count that
    // disagrees with the ratio list without being zero. make_hparams() pins
    // two ratios; one resolved block is neither empty nor two.
    synth::omnivoice::ModelWeights mismatched_blocks = weights;
    mismatched_blocks.acoustic_decoder.blocks.resize(1);
    SYNTH_TEST_CHECK(synth::omnivoice::build_codec_decoder(ctx, codes, mismatched_blocks, make_hparams()) == nullptr);

    // The residual-length guard: DacResidualUnit's own conv1 is kernel 7 with
    // padding 3*dilation, which is exactly length-preserving (2*padding ==
    // dilation*(kernel-1)). A package whose conv1 resolved to kernel 5 with
    // that SAME padding formula -- the builder's own constant, not something
    // this fixture controls -- grows the branch by 2*dilation samples instead
    // of preserving it, and without the guard the mismatched ggml_add would
    // abort on a shape assertion rather than return nullptr.
    synth::omnivoice::HParams residual_hparams = make_hparams();
    residual_hparams.codec.upsampling_ratios.assign({ 2 });  // one block, stride 2
    synth::omnivoice::ModelWeights residual_weights;
    residual_weights.quantizers = quantizers;
    residual_weights.fc2.weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kConcat, kAcoustic);
    residual_weights.fc2.bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kAcoustic);
    residual_weights.acoustic_decoder.conv1.weight =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 7, kAcoustic, kDecoderHidden);
    residual_weights.acoustic_decoder.conv1.bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kDecoderHidden);
    residual_weights.acoustic_decoder.blocks.resize(1);
    synth::omnivoice::AcousticDecoderBlock & residual_block = residual_weights.acoustic_decoder.blocks[0];
    const int64_t                            narrower       = int64_t(kDecoderHidden) / 2;
    residual_block.snake1.alpha   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, kDecoderHidden, 1);
    // conv_t1 is [kernel, out, in]; stride 2 -> kernel 2*stride=4, out=narrower, in=kDecoderHidden.
    residual_block.conv_t1.weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, narrower, kDecoderHidden);
    residual_block.conv_t1.bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, narrower);
    residual_block.res_units.resize(kDacUnits);
    // Only unit 0 needs real weights: the mutated kernel-5 conv1 makes the
    // builder return nullptr on unit 0's own residual add, before units 1 and
    // 2 (left default-constructed) are ever reached.
    synth::omnivoice::DacResidualUnit & mutated_unit = residual_block.res_units[0];
    mutated_unit.snake1.alpha                        = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, narrower, 1);
    mutated_unit.conv1.weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, /* kernel */ 5, narrower, narrower);
    mutated_unit.conv1.bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, narrower);
    mutated_unit.snake2.alpha = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, narrower, 1);
    mutated_unit.conv2.weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, narrower, narrower);
    mutated_unit.conv2.bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, narrower);
    SYNTH_TEST_CHECK(synth::omnivoice::build_codec_decoder(ctx, codes, residual_weights, residual_hparams) == nullptr);
    return 0;
}

// The host guards, which never ride the graph: the grid is discrete, and the
// volume branch is two float operations over an already-materialized buffer.
int check_host_guards() {
    const std::vector<int32_t> good(std::begin(kCodes), std::end(kCodes));
    SYNTH_TEST_CHECK(synth::omnivoice::validate_code_grid(good, kFrames, kQuantizers, kCodebookSize) == SYNTH_OK);

    // The mask id is codebook_size, one past the last real code: it is what the
    // canvas starts filled with, so a loop that failed to commit a slot would
    // hand it straight to get_rows, which would happily read a row that exists
    // in the stacked canvas table but means nothing in the codebook.
    std::vector<int32_t> masked = good;
    masked[3]                   = int32_t(kCodebookSize);
    SYNTH_TEST_CHECK(synth::omnivoice::validate_code_grid(masked, kFrames, kQuantizers, kCodebookSize) ==
                     SYNTH_ERR_INVALID_ARG);

    std::vector<int32_t> negative = good;
    negative[0]                   = -1;
    SYNTH_TEST_CHECK(synth::omnivoice::validate_code_grid(negative, kFrames, kQuantizers, kCodebookSize) ==
                     SYNTH_ERR_INVALID_ARG);

    std::vector<int32_t> short_grid(good.begin(), good.end() - 1);
    SYNTH_TEST_CHECK(synth::omnivoice::validate_code_grid(short_grid, kFrames, kQuantizers, kCodebookSize) ==
                     SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(synth::omnivoice::validate_code_grid(good, 0, kQuantizers, kCodebookSize) ==
                     SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(synth::omnivoice::validate_code_grid(good, kFrames, 0, kCodebookSize) == SYNTH_ERR_INVALID_ARG);

    // Peak 0.4 scaled to 0.5: both values land exactly, since 0.2/0.4 is 0.5
    // and 0.5 * 0.5 is 0.25 in binary.
    std::vector<float> audio = { 0.2f, -0.4f };
    synth::omnivoice::apply_no_reference_volume(audio);
    SYNTH_TEST_CHECK(audio.size() == 2);
    SYNTH_TEST_CHECK(audio[0] == 0.25f);
    SYNTH_TEST_CHECK(audio[1] == -0.5f);

    // Silence stays silence rather than being amplified into noise by a
    // division the guard exists to prevent.
    std::vector<float> silence(8, 0.0f);
    synth::omnivoice::apply_no_reference_volume(silence);
    for (float value : silence) {
        SYNTH_TEST_CHECK(value == 0.0f);
    }
    std::vector<float> empty;
    synth::omnivoice::apply_no_reference_volume(empty);
    SYNTH_TEST_CHECK(empty.empty());

    // The guard's own boundary (`peak <= 1e-6`), one ULP on each side. At
    // exactly 1e-6 the guard fires (inclusive) and the signal is untouched;
    // one ULP above it, the guard does not fire and the peak is scaled to
    // exactly 0.5 -- x / x is exact for any finite nonzero x, so the lone
    // sample IS the peak and 1.0 * 0.5 rounds to nothing.
    std::vector<float> at_threshold = { 1e-6f, -1e-6f };
    synth::omnivoice::apply_no_reference_volume(at_threshold);
    SYNTH_TEST_CHECK(at_threshold[0] == 1e-6f);
    SYNTH_TEST_CHECK(at_threshold[1] == -1e-6f);

    const float        below      = std::nextafter(1e-6f, 0.0f);
    std::vector<float> just_below = { below };
    synth::omnivoice::apply_no_reference_volume(just_below);
    SYNTH_TEST_CHECK(just_below[0] == below);

    const float        above      = std::nextafter(1e-6f, 1.0f);
    std::vector<float> just_above = { above };
    synth::omnivoice::apply_no_reference_volume(just_above);
    SYNTH_TEST_CHECK(just_above[0] == 0.5f);
    return 0;
}

// The wave array is what catches the two defects no shape reveals -- a
// surviving tanh and a time shift -- and it can only do that if its samples
// are big enough and different enough from each other. A re-dump from a
// quieter fixture would leave every check above passing while proving nothing,
// so the same two margins the reference script asserts are re-asserted here
// against the loosest tolerance any device runs at.
int check_wave_has_power() {
    SYNTH_TEST_CHECK(std::size(kExpectedWave) == kSamples);
    float tanh_gap  = 0.0f;
    float shift_gap = 0.0f;
    for (size_t index = 0; index < std::size(kExpectedWave); ++index) {
        const float value = kExpectedWave[index];
        tanh_gap          = std::fmax(tanh_gap, std::fabs(std::tanh(value) - value));
        if (index > 0) {
            shift_gap = std::fmax(shift_gap, std::fabs(value - kExpectedWave[index - 1]));
        }
    }
    std::printf("omnivoice-codec: tanh gap %.3g, one-sample shift gap %.3g\n", double(tanh_gap), double(shift_gap));
    SYNTH_TEST_CHECK(tanh_gap > 10 * kAcceleratorTolerance);
    SYNTH_TEST_CHECK(shift_gap > 10 * kAcceleratorTolerance);
    return 0;
}

}  // namespace

int main() {
    SYNTH_TEST_CHECK(check_rejections() == 0);
    SYNTH_TEST_CHECK(check_host_guards() == 0);
    SYNTH_TEST_CHECK(check_wave_has_power() == 0);

    // Every registered device. The decoder is the family's second hot path, so
    // a backend that disagrees is worth catching here rather than in a golden.
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

        std::printf("omnivoice-codec: %s (device type %d)\n", ggml_backend_dev_name(device), int(type));
        float max_diff = 0.0f;
        SYNTH_TEST_CHECK(run_case(device, max_diff));

        const float tolerance = type == GGML_BACKEND_DEVICE_TYPE_CPU ? kCpuTolerance : kAcceleratorTolerance;
        std::printf("  max_diff %.3g (tolerance %.3g)\n", double(max_diff), double(tolerance));
        SYNTH_TEST_CHECK(max_diff < tolerance);
        ++exercised;
    }
    SYNTH_TEST_CHECK(exercised > 0);
    return 0;
}
