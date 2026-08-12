// create_x_vector_profile (src/arch/qwen3-tts/profile.cpp): the seam that
// turns a normalized reference clip into the in-memory Voice Profile a
// caller synthesizes with -- the object Task 9's public-seam dispatch will
// hand back from synth_voice_profile_create_from_reference once this
// family's capability snapshot stops being all-zero.
//
// Synthetic HParams and in-memory weights drawn from a shared LCG, no GGUF
// file and no loaded Model involved -- tests/omnivoice_reference_encoder_test.cpp's
// own fixture style, and the same shape
// tests/qwen3_tts_speaker_encoder_host_test.cpp (Task 5) already uses for the
// layer immediately underneath this one. profile.h's own header comment
// records why this function takes `HParams`/`SpeakerEncoderWeights` directly
// rather than a `Model &`: `synth::qwen3tts::Model` can only be built from a
// real GGUF (Model::load/load_cpu), and this family has no synthetic-package
// test harness yet.
//
// What this file covers: the two rejections create_x_vector_profile itself
// names (a transcript Plan 2 cannot honour; a package with no speaker
// encoder), that encode_speaker_reference's own silent-reference refusal
// propagates unchanged, and that a successful call produces a Profile fixed
// to CloneMode::XVector carrying the declared enc_dim width and the caller's
// language tag verbatim. encode_speaker_reference's own graph-level
// rejections (a mismatched mel width, a resolver that dropped a pointer, a
// non-finite embedding) are already Task 5's own coverage
// (qwen3_tts_speaker_encoder_host_test.cpp) and are not repeated here.

#include "arch/qwen3-tts/catalog.h"
#include "arch/qwen3-tts/profile.h"
#include "arch/qwen3-tts/weights.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "test-assert.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

using synth::qwen3tts::CloneMode;
using synth::qwen3tts::Conv1dWeights;
using synth::qwen3tts::HParams;
using synth::qwen3tts::SpeakerEncoderBlockWeights;
using synth::qwen3tts::SpeakerEncoderWeights;
using synth::qwen3tts::XVectorProfile;

// Small and distinct from the checkpoint's own widths -- a swapped extent is
// only visible when nothing lines up by coincidence, the same reasoning
// tests/qwen3_tts_speaker_encoder_test.cpp and its host-level sibling give
// for their own fixtures.
constexpr int64_t  kMelBins           = 6;
constexpr int64_t  kChannels          = 16;
constexpr int64_t  kAggregated        = 3 * kChannels;
constexpr int64_t  kRes2NetScale      = 8;
constexpr int64_t  kSeChannels        = 5;
constexpr int64_t  kAttentionChannels = 7;
constexpr int64_t  kEncDim            = 11;
constexpr uint64_t kSeed              = 20260812u;

// The mel front end's own parameters. kSampleRate is this package's real
// declared Reference Audio rate (24 kHz) rather than an arbitrary synthetic
// value, purely so one_second_of_speech() below is truthfully named -- the
// ECAPA graph itself does not care what the sample rate is named, only that
// compute_log_mel and the fixed mel-bin/channel widths below agree with each
// other.
constexpr uint32_t kNFft       = 64;
constexpr uint32_t kHopLength  = 16;
constexpr uint32_t kWinLength  = 64;
constexpr uint32_t kSampleRate = 24000;

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

// ggml reports a Conv1d kernel stored as [out, in, kernel] in reverse -- the
// same layout tests/qwen3_tts_speaker_encoder_test.cpp's own add_conv uses.
void add_conv(ggml_context *               context,
              Conv1dWeights &              target,
              int64_t                      kernel,
              int64_t                      in,
              int64_t                      out,
              std::vector<ggml_tensor *> & order,
              std::vector<float> &         scale) {
    target.weight = ggml_new_tensor_3d(context, GGML_TYPE_F32, kernel, in, out);
    target.bias   = ggml_new_tensor_1d(context, GGML_TYPE_F32, out);
    order.push_back(target.weight);
    scale.push_back(0.5f);
    order.push_back(target.bias);
    scale.push_back(0.25f);
}

// Owns the backend and buffer the weights live on -- the same shape
// tests/qwen3_tts_speaker_encoder_host_test.cpp's own Fixture uses, for the
// same reason: a separate ggml_backend_t from whatever
// encode_speaker_reference opens internally, matching production's own
// split between Model::load's persistent weights buffer and
// encode_speaker_reference's own per-call BackendPlan.
struct Fixture {
    ggml_backend_t        backend = nullptr;
    Context               persistent;
    ggml_backend_buffer_t buffer = nullptr;
    HParams               hparams;
    SpeakerEncoderWeights weights;

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

bool build_fixture(Fixture & fixture) {
    ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (device == nullptr) {
        return false;
    }
    fixture.backend = ggml_backend_dev_init(device, nullptr);
    if (fixture.backend == nullptr) {
        return false;
    }
    fixture.hparams     = make_hparams();
    fixture.persistent  = make_context(ggml_tensor_overhead() * 256);
    ggml_context * pctx = fixture.persistent.get();
    if (pctx == nullptr) {
        return false;
    }

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

// Literally one second at this package's own declared 24 kHz Reference Audio
// rate (kSampleRate above) -- far past compute_log_mel's own minimum sample
// count and kSpeakerEncoderDeepestReflection's 4-frame floor, so nothing
// here exercises either domain edge; those are Task 5's own coverage.
std::vector<float> one_second_of_speech() {
    LcgStream stream(kSeed + 100);
    return stream.fill(size_t(kSampleRate), 0.2f);
}

// D4: a transcript names the transcript-assisted mode, which Plan 2 does not
// implement. Accepting it and silently building the x-vector Profile anyway
// would hand back a weaker clone than the caller asked for -- rejected by
// name rather than ignored.
int test_a_transcript_is_rejected_not_ignored() {
    Fixture fixture;
    SYNTH_TEST_CHECK(build_fixture(fixture));
    const char *                          code    = nullptr;
    const char *                          message = nullptr;
    std::shared_ptr<const XVectorProfile> profile;
    const synth_status_t                  status = synth::qwen3tts::create_x_vector_profile(
        fixture.hparams, fixture.weights, one_second_of_speech(), /*transcript=*/"hello there",
        /*language_tag=*/"", /*threads=*/1, profile, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(profile == nullptr);
    SYNTH_TEST_CHECK(code != nullptr && std::strcmp(code, "voice_profile.transcript_unsupported") == 0);
    SYNTH_TEST_CHECK(message != nullptr);
    return 0;
}

// The spec's section 9 error table: a transcript that is present but
// whitespace-only is still "a transcript was supplied", so it takes the same
// rejection as any other non-empty transcript in Plan 2. The distinction
// between "empty" and "whitespace-only" only starts to matter from Plan 3
// on, once a transcript is actually consumed -- pinned now rather than
// discovered then.
int test_a_whitespace_transcript_is_rejected_too() {
    Fixture fixture;
    SYNTH_TEST_CHECK(build_fixture(fixture));
    const char *                          code    = nullptr;
    const char *                          message = nullptr;
    std::shared_ptr<const XVectorProfile> profile;
    const synth_status_t                  status = synth::qwen3tts::create_x_vector_profile(
        fixture.hparams, fixture.weights, one_second_of_speech(), "   ", "", 1, profile, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(profile == nullptr);
    SYNTH_TEST_CHECK(code != nullptr && std::strcmp(code, "voice_profile.transcript_unsupported") == 0);
    return 0;
}

// jiangzhuo's 2026-08-12 ruling (speaker-encoder-host.cpp's own header
// comment): a digitally silent reference still produces a finite, stable
// x-vector, and that embedding is meaningless, not a cloned voice.
// encode_speaker_reference's own refusal must reach this caller unchanged.
int test_a_silent_reference_is_refused_by_name() {
    Fixture fixture;
    SYNTH_TEST_CHECK(build_fixture(fixture));
    const char *                          code    = nullptr;
    const char *                          message = nullptr;
    std::shared_ptr<const XVectorProfile> profile;
    const std::vector<float>              silence(size_t(kSampleRate), 0.0f);
    const synth_status_t status = synth::qwen3tts::create_x_vector_profile(fixture.hparams, fixture.weights, silence,
                                                                           "", "", 1, profile, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(profile == nullptr);
    SYNTH_TEST_CHECK(code != nullptr && std::strcmp(code, "voice_profile.reference_silent") == 0);
    SYNTH_TEST_CHECK(message != nullptr);
    return 0;
}

// Model::prepare_x_vector's own CustomVoice guard, reproduced here (this
// function does not have a Model to delegate to -- see profile.h's own
// comment): a package with no speaker encoder at all must be refused before
// encode_speaker_reference ever sees a weights struct with every pointer
// null, and before the transcript-less, otherwise-valid request in this test
// would otherwise succeed.
int test_a_package_with_no_speaker_encoder_is_refused() {
    Fixture fixture;
    SYNTH_TEST_CHECK(build_fixture(fixture));
    fixture.hparams.has_speaker_encoder           = false;
    const char *                          code    = nullptr;
    const char *                          message = nullptr;
    std::shared_ptr<const XVectorProfile> profile;
    const synth_status_t                  status = synth::qwen3tts::create_x_vector_profile(
        fixture.hparams, fixture.weights, one_second_of_speech(), "", "", 1, profile, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_UNSUPPORTED_VOICE);
    SYNTH_TEST_CHECK(profile == nullptr);
    return 0;
}

// The success path: the Profile is fixed to CloneMode::XVector, carries the
// package's declared enc_dim width, a positive ref_rms, and the caller's
// language tag verbatim.
int test_a_prepared_profile_carries_the_declared_width() {
    Fixture fixture;
    SYNTH_TEST_CHECK(build_fixture(fixture));
    const char *                          code    = nullptr;
    const char *                          message = nullptr;
    std::shared_ptr<const XVectorProfile> profile;
    const synth_status_t                  status = synth::qwen3tts::create_x_vector_profile(
        fixture.hparams, fixture.weights, one_second_of_speech(), "", "", 1, profile, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_OK);
    SYNTH_TEST_CHECK(profile != nullptr);
    SYNTH_TEST_CHECK(code == nullptr);
    SYNTH_TEST_CHECK(message == nullptr);
    SYNTH_TEST_CHECK(profile->x_vector.size() == fixture.hparams.speaker_encoder.enc_dim);
    SYNTH_TEST_CHECK(profile->mode == CloneMode::XVector);
    SYNTH_TEST_CHECK(profile->ref_rms > 0.0f);
    SYNTH_TEST_CHECK(profile->language_tag.empty());
    return 0;
}

// language_tag is stored verbatim, not validated or normalized -- that is
// this function's caller's job (Task 9's dispatch, mirroring how
// voice-profile.cpp validates a reference's language tag before ever
// reaching omnivoice::create_clone_prompt).
int test_a_language_tag_is_stored_verbatim() {
    Fixture fixture;
    SYNTH_TEST_CHECK(build_fixture(fixture));
    const char *                          code    = nullptr;
    const char *                          message = nullptr;
    std::shared_ptr<const XVectorProfile> profile;
    const synth_status_t                  status = synth::qwen3tts::create_x_vector_profile(
        fixture.hparams, fixture.weights, one_second_of_speech(), "", "en-US", 1, profile, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_OK);
    SYNTH_TEST_CHECK(profile != nullptr);
    SYNTH_TEST_CHECK(profile->language_tag == "en-US");
    return 0;
}

}  // namespace

int main() {
    SYNTH_TEST_CHECK(test_a_transcript_is_rejected_not_ignored() == 0);
    SYNTH_TEST_CHECK(test_a_whitespace_transcript_is_rejected_too() == 0);
    SYNTH_TEST_CHECK(test_a_silent_reference_is_refused_by_name() == 0);
    SYNTH_TEST_CHECK(test_a_package_with_no_speaker_encoder_is_refused() == 0);
    SYNTH_TEST_CHECK(test_a_prepared_profile_carries_the_declared_width() == 0);
    SYNTH_TEST_CHECK(test_a_language_tag_is_stored_verbatim() == 0);
    std::printf("qwen3-tts-profile: all checks passed\n");
    return 0;
}
