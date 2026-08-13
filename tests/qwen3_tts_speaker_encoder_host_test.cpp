// encode_speaker_reference (src/arch/qwen3-tts/speaker-encoder-host.cpp): the
// host wrapper that joins compute_log_mel and build_speaker_encoder behind
// one PCM-in, x-vector-out call.
//
// Task 4's own test (qwen3_tts_speaker_encoder_test.cpp) already covers the
// graph's own rejections -- a mismatched mel width, a scale the channel count
// does not divide, a resolver that dropped a pointer -- against a mel tensor
// built by hand. This file does not repeat any of that. What it covers is
// everything this task's host wrapper adds on top of the graph: the
// ref_rms == 0 silent-reference gate (this family's own, not upstream's --
// see speaker-encoder-host.cpp's own header comment), the enc_dim shape
// assertion carryover Section 3 Task 2(a) named ("No runtime assertion that
// the speaker embedding is 1024 elements, unlike the shape[1] == 16 checks on
// codes"), and that a failure at either stage underneath (compute_log_mel or
// build_speaker_encoder) actually reaches the caller as a status rather than
// being swallowed.
//
// A real 76-tensor package and a real oracle x-vector are Task 5's other
// half, and belong to the integration tier (docs/testing.md: a unit-labelled
// test may not depend on the ~2.5 GB real package) -- see
// scripts/validate-qwen3-tts-replay.py and this task's own report for that
// measurement.

#include "arch/qwen3-tts/catalog.h"
#include "arch/qwen3-tts/mel.h"
#include "arch/qwen3-tts/speaker-encoder-host.h"
#include "arch/qwen3-tts/weights.h"
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

using synth::qwen3tts::Conv1dWeights;
using synth::qwen3tts::HParams;
using synth::qwen3tts::SpeakerEncoderBlockWeights;
using synth::qwen3tts::SpeakerEncoderWeights;
using synth::qwen3tts::XVectorEncoding;

// Small and distinct from the checkpoint's own widths, the same reasoning
// tests/qwen3_tts_speaker_encoder_test.cpp gives: a swapped extent is only
// visible when nothing lines up by coincidence.
constexpr int64_t  kMelBins           = 6;
constexpr int64_t  kChannels          = 16;
constexpr int64_t  kAggregated        = 3 * kChannels;
constexpr int64_t  kRes2NetScale      = 8;
constexpr int64_t  kSeChannels        = 5;
constexpr int64_t  kAttentionChannels = 7;
constexpr int64_t  kEncDim            = 11;
constexpr uint64_t kSeed              = 20260812u;

// A tiny mel front end, chosen only so compute_log_mel accepts the fixed pcm
// clip below and produces a mel whose bin count matches kMelBins. Nothing
// about the ECAPA graph depends on the sample rate or the mel scale
// parameters, so fmin/fmax/sample_rate are arbitrary within compute_log_mel's
// own domain.
constexpr uint32_t kNFft       = 64;
constexpr uint32_t kHopLength  = 16;
constexpr uint32_t kWinLength  = 64;
constexpr uint32_t kSampleRate = 16000;

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

HParams make_hparams() {
    HParams hparams;
    hparams.has_speaker_encoder         = true;
    hparams.speaker_encoder.enc_dim     = uint32_t(kEncDim);
    hparams.speaker_encoder.sample_rate = kSampleRate;
    hparams.speaker_encoder.mel_bins    = uint32_t(kMelBins);
    hparams.speaker_encoder.n_fft       = kNFft;
    hparams.speaker_encoder.hop_length  = kHopLength;
    hparams.speaker_encoder.win_length  = kWinLength;
    hparams.speaker_encoder.fmin        = 0.0f;
    hparams.speaker_encoder.fmax        = float(kSampleRate) / 2.0f;
    return hparams;
}

void add_conv(ggml_context *               context,
              Conv1dWeights &              target,
              int64_t                      kernel,
              int64_t                      in,
              int64_t                      out,
              std::vector<ggml_tensor *> & order,
              std::vector<float> &         scale) {
    // ggml reports a Conv1d kernel stored as [out, in, kernel] in reverse --
    // the same layout tests/qwen3_tts_speaker_encoder_test.cpp's own
    // add_conv uses.
    target.weight = ggml_new_tensor_3d(context, GGML_TYPE_F32, kernel, in, out);
    target.bias   = ggml_new_tensor_1d(context, GGML_TYPE_F32, out);
    order.push_back(target.weight);
    scale.push_back(0.5f);
    order.push_back(target.bias);
    scale.push_back(0.25f);
}

// Owns the backend and buffer the weights live on. Deliberately a SEPARATE
// ggml_backend_t from whatever encode_speaker_reference opens internally for
// its own mel input and graph compute -- exactly the split production has:
// Model::load allocates weights_buffer once, on its own BackendPlan's CPU
// backend, and encode_speaker_reference opens a fresh CPU-only BackendPlan
// per call. A test that put both on the same backend instance would not
// notice a bug in that split.
struct Fixture {
    ggml_backend_t        backend = nullptr;
    Context               persistent;
    ggml_backend_buffer_t buffer = nullptr;
    SpeakerEncoderWeights weights;

    ~Fixture() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
};

bool build_fixture(Fixture & fixture) {
    ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (device == nullptr) {
        return false;
    }
    fixture.backend = ggml_backend_dev_init(device, nullptr);
    if (fixture.backend == nullptr) {
        return false;
    }
    fixture.persistent  = make_context(ggml_tensor_overhead() * 256);
    ggml_context * pctx = fixture.persistent.get();

    std::vector<ggml_tensor *> order;
    std::vector<float>         scale;
    add_conv(pctx, fixture.weights.stem, 5, kMelBins, kChannels, order, scale);
    fixture.weights.blocks.assign(3, SpeakerEncoderBlockWeights{});
    for (SpeakerEncoderBlockWeights & block : fixture.weights.blocks) {
        const int64_t width = kChannels / kRes2NetScale;
        add_conv(pctx, block.tdnn1, 1, kChannels, kChannels, order, scale);
        block.res2net.assign(size_t(kRes2NetScale - 1), Conv1dWeights{});
        for (Conv1dWeights & split : block.res2net) {
            add_conv(pctx, split, 3, width, width, order, scale);
        }
        add_conv(pctx, block.se1, 1, kChannels, kSeChannels, order, scale);
        add_conv(pctx, block.se2, 1, kSeChannels, kChannels, order, scale);
        add_conv(pctx, block.tdnn2, 1, kChannels, kChannels, order, scale);
    }
    add_conv(pctx, fixture.weights.mfa, 1, kAggregated, kAggregated, order, scale);
    add_conv(pctx, fixture.weights.asp_tdnn, 1, 3 * kAggregated, kAttentionChannels, order, scale);
    add_conv(pctx, fixture.weights.asp, 1, kAttentionChannels, kAggregated, order, scale);
    add_conv(pctx, fixture.weights.fc, 1, 2 * kAggregated, kEncDim, order, scale);

    fixture.buffer = ggml_backend_alloc_ctx_tensors(pctx, fixture.backend);
    if (fixture.buffer == nullptr) {
        return false;
    }
    LcgStream stream(kSeed);
    for (size_t index = 0; index < order.size(); ++index) {
        const std::vector<float> values = stream.fill(size_t(ggml_nelements(order[index])), scale[index]);
        ggml_backend_tensor_set(order[index], values.data(), 0, ggml_nbytes(order[index]));
    }
    return true;
}

// 800 samples clears compute_log_mel's own min_pcm_samples ((n_fft -
// hop_length) / 2 + 1 = 25 at this file's parameters) by more than an order
// of magnitude, and produces 50 frames -- far past
// kSpeakerEncoderDeepestReflection (4), so the graph's own reflect-pad limit
// is never what a case here is exercising.
std::vector<float> make_pcm(size_t count = 800) {
    LcgStream stream(kSeed + 100);
    return stream.fill(count, 0.2f);
}

bool all_finite(const std::vector<float> & values) {
    for (float value : values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

// A single NaN weight, written directly onto an already-allocated tensor --
// the state a corrupt package or an upstream converter bug can produce, not
// something this file constructs through the ordinary fill path.
void poison_with_nan(ggml_tensor * tensor) {
    const float nan_value = std::numeric_limits<float>::quiet_NaN();
    ggml_backend_tensor_set(tensor, &nan_value, 0, sizeof(nan_value));
}

int check_success() {
    Fixture fixture;
    SYNTH_TEST_CHECK(build_fixture(fixture));
    const HParams            hparams = make_hparams();
    const std::vector<float> pcm     = make_pcm();

    synth::qwen3tts::MelSpectrogram expected_mel;
    SYNTH_TEST_CHECK(synth::qwen3tts::compute_log_mel(hparams.speaker_encoder, pcm, expected_mel) == SYNTH_OK);

    XVectorEncoding      output;
    const char *         code    = "unset";
    const char *         message = "unset";
    const synth_status_t status =
        synth::qwen3tts::encode_speaker_reference(hparams, fixture.weights, pcm, 1, output, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_OK);
    SYNTH_TEST_CHECK(code == nullptr);
    SYNTH_TEST_CHECK(message == nullptr);
    SYNTH_TEST_CHECK(output.x_vector.size() == size_t(kEncDim));
    SYNTH_TEST_CHECK(all_finite(output.x_vector));
    SYNTH_TEST_CHECK(output.ref_rms > 0.0f);
    SYNTH_TEST_CHECK(output.mel_frames == expected_mel.frames);
    return 0;
}

// The rule this task's header comment names: a digitally silent reference
// still produces a finite, stable x-vector, so it is refused by name rather
// than silently cloned.
int check_silent_reference_refused() {
    Fixture fixture;
    SYNTH_TEST_CHECK(build_fixture(fixture));
    const HParams            hparams = make_hparams();
    const std::vector<float> silence(800, 0.0f);

    XVectorEncoding      output;
    const char *         code    = nullptr;
    const char *         message = nullptr;
    const synth_status_t status =
        synth::qwen3tts::encode_speaker_reference(hparams, fixture.weights, silence, 1, output, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(code != nullptr);
    SYNTH_TEST_CHECK(std::string(code) == "voice_profile.reference_silent");
    SYNTH_TEST_CHECK(message != nullptr);
    SYNTH_TEST_CHECK(output.ref_rms == 0.0f);
    SYNTH_TEST_CHECK(output.x_vector.empty());
    SYNTH_TEST_CHECK(output.mel_frames == 0);
    return 0;
}

int check_empty_pcm_refused() {
    Fixture fixture;
    SYNTH_TEST_CHECK(build_fixture(fixture));
    const HParams            hparams = make_hparams();
    const std::vector<float> empty;

    XVectorEncoding      output;
    const char *         code    = nullptr;
    const char *         message = nullptr;
    const synth_status_t status =
        synth::qwen3tts::encode_speaker_reference(hparams, fixture.weights, empty, 1, output, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(code == nullptr);
    SYNTH_TEST_CHECK(message == nullptr);
    SYNTH_TEST_CHECK(output.x_vector.empty());
    return 0;
}

// Carryover Section 3 Task 2(a): "No runtime assertion that the speaker
// embedding is 1024 elements, unlike the shape[1] == 16 checks on codes."
// A package whose declared enc_dim disagrees with what the graph's own fc
// layer actually produces must be refused, not handed back as a
// wrong-width vector that still looks like a plausible embedding.
int check_enc_dim_mismatch_refused() {
    Fixture fixture;
    SYNTH_TEST_CHECK(build_fixture(fixture));
    HParams hparams                 = make_hparams();
    hparams.speaker_encoder.enc_dim = uint32_t(kEncDim) + 1;  // the graph still produces kEncDim
    const std::vector<float> pcm    = make_pcm();

    XVectorEncoding      output;
    const char *         code    = nullptr;
    const char *         message = nullptr;
    const synth_status_t status =
        synth::qwen3tts::encode_speaker_reference(hparams, fixture.weights, pcm, 1, output, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INTERNAL);
    SYNTH_TEST_CHECK(output.x_vector.empty());
    SYNTH_TEST_CHECK(output.mel_frames == 0);
    // ref_rms is instrumentation, measured before the shape ever comes into
    // question, and survives this failure -- see XVectorEncoding's own doc
    // comment.
    SYNTH_TEST_CHECK(output.ref_rms > 0.0f);
    return 0;
}

// compute_log_mel's own domain refusal (params.n_fft == 0) must reach the
// caller as a status, not be swallowed or turned into something else.
int check_mel_failure_propagates() {
    Fixture fixture;
    SYNTH_TEST_CHECK(build_fixture(fixture));
    HParams hparams               = make_hparams();
    hparams.speaker_encoder.n_fft = 0;
    const std::vector<float> pcm  = make_pcm();

    XVectorEncoding      output;
    const char *         code    = nullptr;
    const char *         message = nullptr;
    const synth_status_t status =
        synth::qwen3tts::encode_speaker_reference(hparams, fixture.weights, pcm, 1, output, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_UNSUPPORTED_INPUT);
    SYNTH_TEST_CHECK(output.x_vector.empty());
    return 0;
}

// build_speaker_encoder's own domain refusal -- a res2net split count the
// channel width does not divide (catalog.h's own note: dropping one
// convolution turns scale 8 into scale 7, which 16 channels do not divide
// by) -- must likewise reach the caller as a status.
int check_graph_build_failure_propagates() {
    Fixture fixture;
    SYNTH_TEST_CHECK(build_fixture(fixture));
    fixture.weights.blocks[1].res2net.pop_back();
    const HParams            hparams = make_hparams();
    const std::vector<float> pcm     = make_pcm();

    XVectorEncoding      output;
    const char *         code    = nullptr;
    const char *         message = nullptr;
    const synth_status_t status =
        synth::qwen3tts::encode_speaker_reference(hparams, fixture.weights, pcm, 1, output, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INTERNAL);
    SYNTH_TEST_CHECK(output.x_vector.empty());
    return 0;
}

// The other half of the shape assertion: even a right-sized embedding must
// be refused if a value in it is not finite. The fc bias is the very last
// tensor read before the embedding leaves the graph, so poisoning it reaches
// this check directly rather than getting lost after thirty-eight
// convolutions and an exponential.
int check_non_finite_embedding_refused() {
    Fixture fixture;
    SYNTH_TEST_CHECK(build_fixture(fixture));
    poison_with_nan(fixture.weights.fc.bias);
    const HParams            hparams = make_hparams();
    const std::vector<float> pcm     = make_pcm();

    XVectorEncoding      output;
    const char *         code    = nullptr;
    const char *         message = nullptr;
    const synth_status_t status =
        synth::qwen3tts::encode_speaker_reference(hparams, fixture.weights, pcm, 1, output, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INTERNAL);
    SYNTH_TEST_CHECK(output.x_vector.empty());
    return 0;
}

}  // namespace

int main() {
    SYNTH_TEST_CHECK(check_success() == 0);
    SYNTH_TEST_CHECK(check_silent_reference_refused() == 0);
    SYNTH_TEST_CHECK(check_empty_pcm_refused() == 0);
    SYNTH_TEST_CHECK(check_enc_dim_mismatch_refused() == 0);
    SYNTH_TEST_CHECK(check_mel_failure_propagates() == 0);
    SYNTH_TEST_CHECK(check_graph_build_failure_propagates() == 0);
    SYNTH_TEST_CHECK(check_non_finite_embedding_refused() == 0);
    std::printf("qwen3-tts-speaker-encoder-host: all checks passed\n");
    return 0;
}
