// The generator's residual block is compared against a host reference, and the
// decoder graph is checked for the frame arithmetic that ties its stages
// together.
//
// The decoder's numerical agreement with the oracle is a Stage 5 concern, where
// the real checkpoint and the recorded reference tensors are available. What
// belongs here is the part the oracle cannot catch cheaply: whether the
// upsampled features and the source spectrum still line up frame for frame at
// every stage, since a mismatch there is a shape error rather than a drift.

#include "arch/kokoro/decoder-host.h"
#include "arch/kokoro/decoder.h"
#include "arch/kokoro/generator.h"
#include "arch/kokoro/operations.h"
#include "arch/kokoro/weights.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "test-assert.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
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

Context make_context(size_t tensors) {
    ggml_init_params parameters{};
    parameters.mem_size = ggml_tensor_overhead() * tensors + ggml_graph_overhead_custom(8192, false);
    parameters.no_alloc = true;
    return Context(ggml_init(parameters));
}

// Tensors are [features, time]: element (c, t) sits at t * channels + c.
size_t at(int64_t channel, int64_t time, int64_t channels) {
    return size_t(time) * size_t(channels) + size_t(channel);
}

// AdaIN: normalize each channel over time without learned affine, then apply a
// scale and shift projected from the style, with the scale offset by one.
void reference_adain(std::vector<float> &       value,
                     int64_t                    channels,
                     int64_t                    length,
                     const std::vector<float> & gamma,
                     const std::vector<float> & beta,
                     float                      epsilon) {
    for (int64_t channel = 0; channel < channels; ++channel) {
        double mean = 0.0;
        for (int64_t time = 0; time < length; ++time) {
            mean += double(value[at(channel, time, channels)]);
        }
        mean /= double(length);
        double variance = 0.0;
        for (int64_t time = 0; time < length; ++time) {
            const double centred = double(value[at(channel, time, channels)]) - mean;
            variance += centred * centred;
        }
        variance /= double(length);
        const double scale = 1.0 / std::sqrt(variance + double(epsilon));
        for (int64_t time = 0; time < length; ++time) {
            const double normalized = (double(value[at(channel, time, channels)]) - mean) * scale;
            value[at(channel, time, channels)] =
                float(normalized * (1.0 + double(gamma[size_t(channel)])) + double(beta[size_t(channel)]));
        }
    }
}

void reference_snake(std::vector<float> & value, int64_t channels, int64_t length, const std::vector<float> & alpha) {
    for (int64_t channel = 0; channel < channels; ++channel) {
        const double a = double(alpha[size_t(channel)]);
        for (int64_t time = 0; time < length; ++time) {
            const double x                     = double(value[at(channel, time, channels)]);
            const double term                  = std::sin(a * x);
            value[at(channel, time, channels)] = float(x + term * term / a);
        }
    }
}

// Padded, dilated convolution that keeps the length. The weight is
// [kernel, in_channels, out_channels] in GGML's reported order.
std::vector<float> reference_conv(const std::vector<float> & input,
                                  const std::vector<float> & weight,
                                  const std::vector<float> & bias,
                                  int64_t                    channels,
                                  int64_t                    length,
                                  int64_t                    kernel,
                                  int64_t                    dilation) {
    const int64_t      padding = dilation * (kernel - 1) / 2;
    std::vector<float> output(size_t(length) * size_t(channels), 0.0f);
    for (int64_t out_channel = 0; out_channel < channels; ++out_channel) {
        for (int64_t time = 0; time < length; ++time) {
            double sum = double(bias[size_t(out_channel)]);
            for (int64_t in_channel = 0; in_channel < channels; ++in_channel) {
                for (int64_t tap = 0; tap < kernel; ++tap) {
                    const int64_t source = time + tap * dilation - padding;
                    if (source < 0 || source >= length) {
                        continue;
                    }
                    const size_t weight_index = size_t(out_channel) * size_t(channels) * size_t(kernel) +
                                                size_t(in_channel) * size_t(kernel) + size_t(tap);
                    sum += double(input[at(in_channel, source, channels)]) * double(weight[weight_index]);
                }
            }
            output[at(out_channel, time, channels)] = float(sum);
        }
    }
    return output;
}

// One branch of AdaINResBlock1, straight from the upstream definition.
std::vector<float> reference_resblock(const std::vector<float> &              input,
                                      const std::vector<float> &              style,
                                      const std::vector<std::vector<float>> & parameters,
                                      int64_t                                 channels,
                                      int64_t                                 length,
                                      int64_t                                 style_dim,
                                      const std::vector<int64_t> &            dilations,
                                      int64_t                                 kernel,
                                      float                                   epsilon) {
    // Projections turn the style into a scale and shift, stacked in that order.
    auto project = [&](const std::vector<float> & weight, const std::vector<float> & bias, std::vector<float> & gamma,
                       std::vector<float> & beta) {
        gamma.assign(size_t(channels), 0.0f);
        beta.assign(size_t(channels), 0.0f);
        for (int64_t row = 0; row < 2 * channels; ++row) {
            double sum = double(bias[size_t(row)]);
            for (int64_t column = 0; column < style_dim; ++column) {
                sum += double(style[size_t(column)]) * double(weight[size_t(row) * size_t(style_dim) + size_t(column)]);
            }
            if (row < channels) {
                gamma[size_t(row)] = float(sum);
            } else {
                beta[size_t(row - channels)] = float(sum);
            }
        }
    };

    std::vector<float> current = input;
    for (size_t branch = 0; branch < dilations.size(); ++branch) {
        const size_t       base = branch * 8;
        std::vector<float> gamma;
        std::vector<float> beta;

        std::vector<float> residual = current;
        project(parameters[base + 0], parameters[base + 1], gamma, beta);
        reference_adain(residual, channels, length, gamma, beta, epsilon);
        reference_snake(residual, channels, length, parameters[base + 4]);
        residual = reference_conv(residual, parameters[base + 2], parameters[base + 3], channels, length, kernel,
                                  dilations[branch]);

        project(parameters[base + 5], parameters[base + 6], gamma, beta);
        reference_adain(residual, channels, length, gamma, beta, epsilon);
        reference_snake(residual, channels, length, parameters[base + 7]);
        // The second convolution is never dilated. Its pairs are appended after
        // every branch's own parameters, two per branch.
        const size_t convs2 = dilations.size() * 8 + branch * 2;
        residual =
            reference_conv(residual, parameters[convs2 + 0], parameters[convs2 + 1], channels, length, kernel, 1);

        for (size_t index = 0; index < current.size(); ++index) {
            current[index] = residual[index] + current[index];
        }
    }
    return current;
}

}  // namespace

// Checks build_adain_resblock1 against the reference above.
static bool run_resblock_case(float & max_diff) {
    constexpr int64_t channels  = 4;
    constexpr int64_t length    = 7;
    constexpr int64_t style_dim = 3;
    constexpr int64_t kernel    = 3;
    constexpr float   epsilon   = 1e-5f;

    const std::vector<int64_t>  dilations64 = { 1, 3 };
    const std::vector<uint32_t> dilations   = { 1, 3 };

    std::mt19937                    rng(20260727u);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    auto                            sample = [&](size_t count) {
        std::vector<float> values(count);
        for (float & value : values) {
            value = dist(rng);
        }
        return values;
    };

    std::vector<float> input = sample(size_t(length) * size_t(channels));
    std::vector<float> style = sample(size_t(style_dim));

    // Per branch, in the order the reference reads them: adain1 weight and
    // bias, convs1 weight and bias, alpha1, adain2 weight and bias, alpha2.
    // The convs2 pairs follow all of the branches.
    std::vector<std::vector<float>> parameters;
    for (size_t branch = 0; branch < dilations.size(); ++branch) {
        parameters.push_back(sample(size_t(2 * channels) * size_t(style_dim)));
        parameters.push_back(sample(size_t(2 * channels)));
        parameters.push_back(sample(size_t(kernel) * size_t(channels) * size_t(channels)));
        parameters.push_back(sample(size_t(channels)));
        parameters.push_back(sample(size_t(channels)));
        parameters.push_back(sample(size_t(2 * channels) * size_t(style_dim)));
        parameters.push_back(sample(size_t(2 * channels)));
        parameters.push_back(sample(size_t(channels)));
    }
    const size_t convs2_base = parameters.size();
    for (size_t branch = 0; branch < dilations.size(); ++branch) {
        parameters.push_back(sample(size_t(kernel) * size_t(channels) * size_t(channels)));
        parameters.push_back(sample(size_t(channels)));
    }
    // The reference indexes convs2 as base + 2 + 8 * branches, which lands on
    // the block appended above only when the two agree on where it starts.
    if (convs2_base != dilations.size() * 8) {
        return false;
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) {
        return false;
    }
    Context        holder = make_context(256);
    ggml_context * ctx    = holder.get();

    ggml_tensor * input_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, channels, length);
    ggml_tensor * style_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, style_dim);

    synth::kokoro::AdaINResBlock1Weights weights;
    std::vector<ggml_tensor *>           handles;
    auto make_tensor = [&](const std::vector<float> & values, int64_t d0, int64_t d1, int64_t d2) {
        ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d0, d1, d2);
        handles.push_back(tensor);
        return tensor;
    };

    for (size_t branch = 0; branch < dilations.size(); ++branch) {
        const size_t                base = branch * 8;
        synth::kokoro::AdaINWeights adain1;
        adain1.fc.weight = make_tensor(parameters[base + 0], style_dim, 2 * channels, 1);
        adain1.fc.bias   = make_tensor(parameters[base + 1], 2 * channels, 1, 1);
        synth::kokoro::AdaINWeights adain2;
        adain2.fc.weight = make_tensor(parameters[base + 5], style_dim, 2 * channels, 1);
        adain2.fc.bias   = make_tensor(parameters[base + 6], 2 * channels, 1, 1);
        weights.adain1.push_back(adain1);
        weights.adain2.push_back(adain2);

        synth::kokoro::Conv1dWeights conv1;
        conv1.weight = make_tensor(parameters[base + 2], kernel, channels, channels);
        conv1.bias   = make_tensor(parameters[base + 3], channels, 1, 1);
        weights.convs1.push_back(conv1);

        synth::kokoro::Conv1dWeights conv2;
        conv2.weight = make_tensor(parameters[convs2_base + branch * 2 + 0], kernel, channels, channels);
        conv2.bias   = make_tensor(parameters[convs2_base + branch * 2 + 1], channels, 1, 1);
        weights.convs2.push_back(conv2);

        // Snake alphas are stored as [1, channels, 1], the shape the checkpoint
        // holds them in.
        weights.alpha1.push_back(make_tensor(parameters[base + 4], 1, channels, 1));
        weights.alpha2.push_back(make_tensor(parameters[base + 7], 1, channels, 1));
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_backend_free(backend);
        return false;
    }
    ggml_backend_tensor_set(input_tensor, input.data(), 0, ggml_nbytes(input_tensor));
    ggml_backend_tensor_set(style_tensor, style.data(), 0, ggml_nbytes(style_tensor));

    // Handles were pushed in the same order the values were built, so this
    // fills each tensor from its own vector.
    std::vector<const std::vector<float> *> ordered;
    for (size_t branch = 0; branch < dilations.size(); ++branch) {
        const size_t base = branch * 8;
        ordered.push_back(&parameters[base + 0]);
        ordered.push_back(&parameters[base + 1]);
        ordered.push_back(&parameters[base + 5]);
        ordered.push_back(&parameters[base + 6]);
        ordered.push_back(&parameters[base + 2]);
        ordered.push_back(&parameters[base + 3]);
        ordered.push_back(&parameters[convs2_base + branch * 2 + 0]);
        ordered.push_back(&parameters[convs2_base + branch * 2 + 1]);
        ordered.push_back(&parameters[base + 4]);
        ordered.push_back(&parameters[base + 7]);
    }
    if (ordered.size() != handles.size()) {
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        return false;
    }
    for (size_t index = 0; index < handles.size(); ++index) {
        ggml_backend_tensor_set(handles[index], ordered[index]->data(), 0, ggml_nbytes(handles[index]));
    }

    Context       graph_holder = make_context(1024);
    ggml_cgraph * graph        = ggml_new_graph_custom(graph_holder.get(), 4096, false);
    ggml_tensor * out =
        synth::kokoro::build_adain_resblock1(graph_holder.get(), input_tensor, style_tensor, weights, dilations.data(),
                                             uint32_t(dilations.size()), uint32_t(kernel), epsilon);
    if (out == nullptr || out->ne[0] != channels || out->ne[1] != length) {
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        return false;
    }
    ggml_build_forward_expand(graph, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, graph);
    ggml_backend_cpu_set_n_threads(backend, 2);
    const bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;

    std::vector<float> produced(size_t(length) * size_t(channels));
    if (ok) {
        ggml_backend_tensor_get(out, produced.data(), 0, ggml_nbytes(out));
    }
    const std::vector<float> expected =
        reference_resblock(input, style, parameters, channels, length, style_dim, dilations64, kernel, epsilon);

    max_diff = 0.0f;
    for (size_t index = 0; index < expected.size() && ok; ++index) {
        max_diff = std::max(max_diff, std::fabs(expected[index] - produced[index]));
    }

    ggml_gallocr_free(alloc);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    return ok;
}

namespace {

// A reduced configuration with the same structure as the real one: two
// upsampling stages whose kernels exceed their rates by an even margin, an even
// transform size, and a decoder whose last block doubles the frame rate.
synth::kokoro::HParams make_hparams() {
    synth::kokoro::HParams hparams;
    hparams.dim_in                         = 4;
    hparams.style_dim                      = 3;
    hparams.adain_eps                      = 1e-5f;
    hparams.istftnet.upsample_rates        = { 2, 3 };
    hparams.istftnet.upsample_kernel_sizes = { 4, 9 };
    hparams.istftnet.resblock_kernel_sizes = { 3, 5 };
    hparams.istftnet.resblock_dilations    = {
        { 1, 3 },
        { 1, 3 }
    };
    hparams.istftnet.upsample_initial_channel = 8;
    hparams.istftnet.gen_istft_n_fft          = 4;
    hparams.istftnet.gen_istft_hop_size       = 2;
    return hparams;
}

struct Builder {
    ggml_context * ctx;

    ggml_tensor * conv(int64_t kernel, int64_t in_channels, int64_t out_channels) const {
        return ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kernel, in_channels, out_channels);
    }

    ggml_tensor * vector(int64_t size) const { return ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size); }

    ggml_tensor * matrix(int64_t columns, int64_t rows) const {
        return ggml_new_tensor_2d(ctx, GGML_TYPE_F32, columns, rows);
    }

    synth::kokoro::Conv1dWeights conv_weights(int64_t kernel, int64_t in_channels, int64_t out_channels) const {
        synth::kokoro::Conv1dWeights weights;
        weights.weight = conv(kernel, in_channels, out_channels);
        weights.bias   = vector(out_channels);
        return weights;
    }

    // A transposed convolution stores its axes the other way round: PyTorch
    // holds [in, out, kernel], which GGML reports as [kernel, out, in].
    synth::kokoro::Conv1dWeights transpose_weights(int64_t kernel, int64_t in_channels, int64_t out_channels) const {
        synth::kokoro::Conv1dWeights weights;
        weights.weight = conv(kernel, out_channels, in_channels);
        weights.bias   = vector(out_channels);
        return weights;
    }

    synth::kokoro::AdaINWeights adain_weights(int64_t style_dim, int64_t channels) const {
        synth::kokoro::AdaINWeights weights;
        weights.fc.weight = matrix(style_dim, 2 * channels);
        weights.fc.bias   = vector(2 * channels);
        return weights;
    }

    synth::kokoro::AdainResBlockWeights block(int64_t style_dim,
                                              int64_t in_channels,
                                              int64_t out_channels,
                                              bool    upsample) const {
        synth::kokoro::AdainResBlockWeights weights;
        weights.conv1 = conv_weights(3, in_channels, out_channels);
        weights.conv2 = conv_weights(3, out_channels, out_channels);
        weights.norm1 = adain_weights(style_dim, in_channels);
        weights.norm2 = adain_weights(style_dim, out_channels);
        if (in_channels != out_channels) {
            weights.conv1x1 = conv(1, in_channels, out_channels);
        }
        if (upsample) {
            weights.pool = conv_weights(3, 1, in_channels);
        }
        return weights;
    }

    synth::kokoro::AdaINResBlock1Weights snake_block(int64_t                       style_dim,
                                                     int64_t                       channels,
                                                     int64_t                       kernel,
                                                     const std::vector<uint32_t> & dilations) const {
        synth::kokoro::AdaINResBlock1Weights weights;
        for (size_t branch = 0; branch < dilations.size(); ++branch) {
            weights.convs1.push_back(conv_weights(kernel, channels, channels));
            weights.convs2.push_back(conv_weights(kernel, channels, channels));
            weights.adain1.push_back(adain_weights(style_dim, channels));
            weights.adain2.push_back(adain_weights(style_dim, channels));
            weights.alpha1.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, channels, 1));
            weights.alpha2.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, channels, 1));
        }
        return weights;
    }
};

synth::kokoro::DecoderWeights make_weights(const Builder & make, const synth::kokoro::HParams & hparams) {
    const int64_t style  = hparams.style_dim;
    const int64_t asr    = hparams.dim_in;
    const int64_t wide   = 8;
    const int64_t narrow = 2;

    synth::kokoro::DecoderWeights weights;
    weights.f0_conv = make.conv_weights(3, 1, 1);
    weights.n_conv  = make.conv_weights(3, 1, 1);
    weights.asr_res = make.conv_weights(1, asr, narrow);
    weights.encode  = make.block(style, asr + 2, wide, false);
    for (int block = 0; block < 4; ++block) {
        const bool last = block == 3;
        weights.decode.push_back(make.block(style, wide + narrow + 2, last ? wide : wide, last));
    }

    const std::vector<uint32_t> noise_dilations = { 1, 3, 5 };
    int64_t                     channels        = hparams.istftnet.upsample_initial_channel;
    for (size_t stage = 0; stage < hparams.istftnet.upsample_rates.size(); ++stage) {
        const int64_t next = channels / 2;
        weights.generator.ups.push_back(
            make.transpose_weights(int64_t(hparams.istftnet.upsample_kernel_sizes[stage]), channels, next));
        const bool last = stage + 1 == hparams.istftnet.upsample_rates.size();
        if (last) {
            weights.generator.noise_convs.push_back(
                make.conv_weights(1, int64_t(hparams.istftnet.gen_istft_n_fft) + 2, next));
        } else {
            int64_t source_stride = 1;
            for (size_t later = stage + 1; later < hparams.istftnet.upsample_rates.size(); ++later) {
                source_stride *= int64_t(hparams.istftnet.upsample_rates[later]);
            }
            weights.generator.noise_convs.push_back(
                make.conv_weights(source_stride * 2, int64_t(hparams.istftnet.gen_istft_n_fft) + 2, next));
        }
        weights.generator.noise_res.push_back(make.snake_block(style, next, last ? 5 : 3, noise_dilations));
        for (size_t branch = 0; branch < hparams.istftnet.resblock_kernel_sizes.size(); ++branch) {
            weights.generator.resblocks.push_back(
                make.snake_block(style, next, int64_t(hparams.istftnet.resblock_kernel_sizes[branch]),
                                 hparams.istftnet.resblock_dilations[branch]));
        }
        channels = next;
    }
    weights.generator.conv_post = make.conv_weights(3, channels, int64_t(hparams.istftnet.gen_istft_n_fft) + 2);
    return weights;
}

}  // namespace

int main() {
    float max_diff = 0.0f;
    SYNTH_TEST_CHECK(run_resblock_case(max_diff));
    SYNTH_TEST_CHECK(max_diff < 1e-4f);

    const synth::kokoro::HParams hparams = make_hparams();

    // Two rates of 2 and 3 multiply the length by six, and the last stage's
    // reflected frame adds one. That extra frame is what makes the upsampled
    // features and the source spectrum the same length.
    SYNTH_TEST_CHECK(synth::kokoro::generator_output_frames(hparams, 6) == 37);
    SYNTH_TEST_CHECK(synth::kokoro::generator_output_frames(hparams, 0) == 1);
    // A hop of two turns those frames back into samples.
    SYNTH_TEST_CHECK(synth::kokoro::inverse_stft_length(hparams, 37) == 72);

    Context                             holder = make_context(4096);
    Builder                             make{ holder.get() };
    const synth::kokoro::DecoderWeights weights = make_weights(make, hparams);

    constexpr uint32_t          frame_count  = 3;
    Context                     graph_holder = make_context(8192);
    synth::kokoro::DecoderGraph built =
        synth::kokoro::build_decoder_graph(graph_holder.get(), weights, hparams, frame_count);
    SYNTH_TEST_CHECK(built.graph != nullptr);
    SYNTH_TEST_CHECK(built.spectrum != nullptr);

    // The curves arrive at twice the frame rate and are halved inside.
    SYNTH_TEST_CHECK(built.f0->ne[1] == 2 * frame_count);
    SYNTH_TEST_CHECK(built.energy->ne[1] == 2 * frame_count);
    SYNTH_TEST_CHECK(built.asr->ne[0] == hparams.dim_in);
    SYNTH_TEST_CHECK(built.asr->ne[1] == frame_count);

    const uint64_t frames = synth::kokoro::generator_output_frames(hparams, uint64_t(frame_count) * 2);
    SYNTH_TEST_CHECK(frames == 37);
    SYNTH_TEST_CHECK(built.har->ne[0] == hparams.istftnet.gen_istft_n_fft + 2);
    SYNTH_TEST_CHECK(built.har->ne[1] == int64_t(frames));
    SYNTH_TEST_CHECK(built.spectrum->ne[0] == hparams.istftnet.gen_istft_n_fft + 2);
    SYNTH_TEST_CHECK(built.spectrum->ne[1] == int64_t(frames));

    // The node budget has to cover what the builder actually emitted, or a
    // longer input would silently overflow the graph.
    SYNTH_TEST_CHECK(ggml_graph_n_nodes(built.graph) > 0);
    SYNTH_TEST_CHECK(uint64_t(ggml_graph_n_nodes(built.graph)) <=
                     synth::kokoro::decoder_graph_node_count(hparams, frame_count) + 256);

    // A longer input costs the same nodes, since only the shapes grow.
    Context                     longer_holder = make_context(8192);
    synth::kokoro::DecoderGraph longer =
        synth::kokoro::build_decoder_graph(longer_holder.get(), weights, hparams, frame_count * 4);
    SYNTH_TEST_CHECK(longer.graph != nullptr);
    SYNTH_TEST_CHECK(ggml_graph_n_nodes(longer.graph) == ggml_graph_n_nodes(built.graph));
    SYNTH_TEST_CHECK(longer.spectrum->ne[1] ==
                     int64_t(synth::kokoro::generator_output_frames(hparams, uint64_t(frame_count) * 8)));

    // Contract failures return an empty graph rather than a wrong shape.
    Context reject_holder = make_context(8192);
    SYNTH_TEST_CHECK(synth::kokoro::build_decoder_graph(nullptr, weights, hparams, frame_count).graph == nullptr);
    SYNTH_TEST_CHECK(synth::kokoro::build_decoder_graph(reject_holder.get(), weights, hparams, 0).graph == nullptr);

    synth::kokoro::DecoderWeights short_decode = weights;
    short_decode.decode.pop_back();
    SYNTH_TEST_CHECK(
        synth::kokoro::build_decoder_graph(reject_holder.get(), short_decode, hparams, frame_count).graph == nullptr);

    // An upsampling kernel that does not exceed its rate by an even margin has
    // no padding that preserves the length, which the builder refuses.
    synth::kokoro::HParams odd_margin            = hparams;
    odd_margin.istftnet.upsample_kernel_sizes[1] = 8;
    SYNTH_TEST_CHECK(synth::kokoro::build_decoder_graph(reject_holder.get(), weights, odd_margin, frame_count).graph ==
                     nullptr);

    // A catalog whose branch count disagrees with the configured dilations is a
    // conversion defect.
    synth::kokoro::HParams extra_branch = hparams;
    extra_branch.istftnet.resblock_kernel_sizes.push_back(7);
    SYNTH_TEST_CHECK(
        synth::kokoro::build_decoder_graph(reject_holder.get(), weights, extra_branch, frame_count).graph == nullptr);

    SYNTH_TEST_CHECK(synth::kokoro::build_adain_resblock1(reject_holder.get(), nullptr, nullptr,
                                                          weights.generator.noise_res.front(), nullptr, 0, 3,
                                                          1e-5f) == nullptr);
    return 0;
}
