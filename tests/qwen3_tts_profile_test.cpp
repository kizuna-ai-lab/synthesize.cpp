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
// names (a transcript it does not prepare; a package with no speaker
// encoder), that encode_speaker_reference's own silent-reference refusal
// propagates unchanged, and that a successful call produces a Profile fixed
// to CloneMode::XVector carrying the declared enc_dim width and the caller's
// language tag verbatim. encode_speaker_reference's own graph-level
// rejections (a mismatched mel width, a resolver that dropped a pointer, a
// non-finite embedding) are already Task 5's own coverage
// (qwen3_tts_speaker_encoder_host_test.cpp) and are not repeated here.
//
// Plan 3's Task 8 adds create_icl_profile's own section below, with a second
// fixture that carries a codec encoder beside the ECAPA stack; the envelope
// sections after it are Plan 2's and are unchanged, save for one added check
// that a Plan 2 x-vector Profile still round-trips now that a second clone
// mode exists.

#include "arch/qwen3-tts/catalog.h"
#include "arch/qwen3-tts/codec-encoder-host.h"
#include "arch/qwen3-tts/codec-encoder.h"
#include "arch/qwen3-tts/profile.h"
#include "arch/qwen3-tts/weights.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "test-assert.h"
#include "voice-profile-handle.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

using synth::qwen3tts::CloneMode;
using synth::qwen3tts::CodecEncoderWeights;
using synth::qwen3tts::Conv1dWeights;
using synth::qwen3tts::HParams;
using synth::qwen3tts::IclProfile;
using synth::qwen3tts::kPrescanKnownKeyCount;
using synth::qwen3tts::kPrescanKnownKeys;
using synth::qwen3tts::kPrescanKvCountXVector;
using synth::qwen3tts::PrescanKeyScope;
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

    std::vector<float> fill(size_t count, float scale, float offset = 0.0f) {
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

// One tensor of a fixture's weights and the LCG draw that fills it. `offset`
// is non-zero only for a LayerNorm gain, which a trained network draws about
// one rather than about zero.
struct WeightDraw {
    ggml_tensor * tensor = nullptr;
    float         scale  = 0.0f;
    float         offset = 0.0f;
};

void fill_weights(uint64_t seed, const std::vector<WeightDraw> & draws) {
    LcgStream stream(seed);
    for (const WeightDraw & draw : draws) {
        const std::vector<float> values = stream.fill(size_t(ggml_nelements(draw.tensor)), draw.scale, draw.offset);
        ggml_backend_tensor_set(draw.tensor, values.data(), 0, ggml_nbytes(draw.tensor));
    }
}

// ggml reports a Conv1d kernel stored as [out, in, kernel] in reverse -- the
// same layout tests/qwen3_tts_speaker_encoder_test.cpp's own add_conv uses.
void add_conv(ggml_context *            context,
              Conv1dWeights &           target,
              int64_t                   kernel,
              int64_t                   in,
              int64_t                   out,
              std::vector<WeightDraw> & draws) {
    target.weight = ggml_new_tensor_3d(context, GGML_TYPE_F32, kernel, in, out);
    target.bias   = ggml_new_tensor_1d(context, GGML_TYPE_F32, out);
    draws.push_back({ target.weight, 0.5f, 0.0f });
    draws.push_back({ target.bias, 0.25f, 0.0f });
}

// The ECAPA-TDNN half of a fixture, on `context`. Extracted from
// build_fixture below when the ICL fixture needed the same stack beside a
// codec encoder: the two fixtures must agree on the speaker weights exactly,
// or an ICL Profile and an x-vector Profile prepared from the same clip would
// stop being comparable for reasons that have nothing to do with the mode.
void add_speaker_weights(ggml_context * context, SpeakerEncoderWeights & weights, std::vector<WeightDraw> & draws) {
    add_conv(context, weights.stem, 5, kMelBins, kChannels, draws);
    weights.blocks.assign(3, SpeakerEncoderBlockWeights{});
    for (SpeakerEncoderBlockWeights & block : weights.blocks) {
        const int64_t width = kChannels / kRes2NetScale;
        add_conv(context, block.tdnn1, 1, kChannels, kChannels, draws);
        block.res2net.assign(size_t(kRes2NetScale - 1), Conv1dWeights{});
        for (Conv1dWeights & split : block.res2net) {
            add_conv(context, split, 3, width, width, draws);
        }
        add_conv(context, block.se1, 1, kChannels, kSeChannels, draws);
        add_conv(context, block.se2, 1, kSeChannels, kChannels, draws);
        add_conv(context, block.tdnn2, 1, kChannels, kChannels, draws);
    }
    add_conv(context, weights.mfa, 1, kAggregated, kAggregated, draws);
    add_conv(context, weights.asp_tdnn, 1, 3 * kAggregated, kAttentionChannels, draws);
    add_conv(context, weights.asp, 1, kAttentionChannels, kAggregated, draws);
    add_conv(context, weights.fc, 1, 2 * kAggregated, kEncDim, draws);
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

    std::vector<WeightDraw> draws;
    add_speaker_weights(pctx, fixture.weights, draws);

    fixture.buffer = ggml_backend_alloc_ctx_tensors(pctx, fixture.backend);
    if (fixture.buffer == nullptr) {
        return false;
    }
    fill_weights(kSeed, draws);
    return true;
}

// `samples` samples of drawn "speech" at this package's own declared 24 kHz
// Reference Audio rate. Same stream at every length, so a longer clip is the
// shorter one with samples appended and nothing else changed.
std::vector<float> speech_of(size_t samples) {
    LcgStream stream(kSeed + 100);
    return stream.fill(samples, 0.2f);
}

// Literally one second at this package's own declared 24 kHz Reference Audio
// rate (kSampleRate above) -- far past compute_log_mel's own minimum sample
// count and kSpeakerEncoderDeepestReflection's 4-frame floor, so nothing
// here exercises either domain edge; those are Task 5's own coverage.
std::vector<float> one_second_of_speech() {
    return speech_of(size_t(kSampleRate));
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

// =============================================================================
// Plan 3 Task 8: create_icl_profile and the IclProfile payload.
//
// The fixture below carries a codec encoder BESIDE the ECAPA stack above, on
// the same context, because an ICL Profile is prepared from both: D5
// tabulates three rows for this mode -- the `[1024]` speaker
// embedding, which that table marks `yes` in BOTH columns, plus the reference
// codes and the reference text token ids -- and create_icl_profile runs both
// encode chains to fill them.
//
// STILL NO Model, AND STILL NO 2.5 GB PACKAGE. This file is registered as a
// `unit` test (tests/CMakeLists.txt:180), so it may not depend on one, and
// synth::qwen3tts::Model has no constructor that avoids one. That is exactly
// why create_icl_profile takes `(HParams, SpeakerEncoderWeights,
// CodecEncoderWeights, ...)` rather than a `Model &` (its own header comment),
// and why `reference_text_ids` arrives already tokenized: the BPE tables live
// on Model, so tokenizing inside that function would drag one in. The tests
// below supply ids directly, exactly as Task 10's dispatch site will after
// running Model::tokenize_reference_transcript.
//
// The one assertion that does NOT fit here and moved rather than shrinking is
// the SELECTOR -- the same public call producing an ICL Profile with a
// transcript and an x-vector Profile without one. That needs a live Model and
// lands in tests/qwen3_tts_base_load_real.cpp as part of Task 10. What stays
// at the `unit` layer is the half that pins D4 at the family seam: each
// create_* function refuses the input naming the other mode
// (test_a_transcript_is_rejected_not_ignored above for one direction,
// test_icl_refuses_a_blank_transcript below for the other).
//
// EVERY RULE BELOW WAS INVERTED AND RE-RUN, and each reported at the check it
// was aimed at. Named by the DIMENSION each perturbation moves, because eight
// inversions of the same dimension are one inversion:
//
//   kind      CloneMode::Icl written as XVector    -> the mode check
//   kind      `speaker` moved off offset 0          -> the profile.h
//                                                      static_assert, at COMPILE
//                                                      time, not this file
//   kind      the loader labels an x-vector
//             envelope Icl                          -> the Plan 2 round trip
//   presence  the blank-transcript check removed    -> the "" case
//   value     blankness weakened to emptiness       -> the "   " case
//   presence  the reference_text_ids guard removed  -> the half-present case
//   presence  the ids not copied into the payload   -> D5's third row
//   order     a stage-major transpose on the way in -> codes_equal
//   length    the frame trim as a floor divide      -> 13 frames
//   length    the language_tag ceiling removed      -> the oversized tag
//   presence  the has_speaker_encoder guard removed -> UNSUPPORTED_VOICE
//
// TWO MASKINGS, recorded rather than left to be rediscovered. The
// all-three-rows check used to assert the literal frame count as well, so a
// frame-trim fault reported ITSELF there instead of at the check aimed at it
// (measured); the literal moved out, and only the self-consistent
// `codes.size() == groups * frames` stayed. And the loader-mislabels
// inversion trips test_round_trip_of_a_well_formed_profile first, since that
// arm has pinned the same field since Plan 2 -- with it commented out, the
// Plan 2 round trip below reports it, which is how it was confirmed rather
// than assumed.
// =============================================================================

// The real Base checkpoint's encode geometry, at a fraction of its width.
// `upsampling_ratios` is [8, 6, 5, 4] and the encoder walks it REVERSED
// (modeling_mimi.py:456), so its strides are 4, 5, 6, 8 and its kernels are
// twice those; the frame downsampler adds one more stride of 2, for
// 4 * 5 * 6 * 8 * 2 = 1920 samples per frame.
//
// THE STRIDES ARE THE PRODUCTION ONES AND THE CHANNEL WIDTHS ARE NOT, on
// purpose. codec_encoder_geometry reads each convolution's kernel extent and
// never its channel count, so this narrow stack lands on exactly the frame
// counts a real reference clip does -- which is what lets the frame-count
// check below name 13 and 101 rather than two numbers peculiar to a fixture.
// tests/qwen3_tts_codec_encoder_test.cpp takes the opposite trade (strides
// 2, 3, 2, 2, so 48 samples per frame) because it runs many more clips
// through the graph and does not care what a frame is worth.
constexpr int64_t kCodecStem            = 4;
constexpr int64_t kCodecStageWidths[]   = { 8, 16, 32, 64 };
constexpr int64_t kCodecStageKernels[]  = { 8, 10, 12, 16 };
constexpr int64_t kCodecHidden          = 32;
constexpr int64_t kCodecIntermediate    = 128;
constexpr int64_t kCodecLayers          = 8;
constexpr int64_t kCodecSamplesPerFrame = 1920;
// `kProjected` is `codebook_dim / 2` exactly as catalog.cpp derives it.
constexpr int64_t kProjected            = kCodecHidden / 2;
constexpr int64_t kCodebookSize         = 8;
constexpr int64_t kGroups               = 16;
constexpr int64_t kSemanticGroups       = 1;

// The two reference lengths D5's frame arithmetic is stated against: the
// shortest clip this package accepts, and base-icl-en's own. 24,000 / 1920 is
// 12.5, so the CEILING divide the frame trim applies makes it 13 -- the one
// number a floor divide gets wrong.
constexpr size_t   kShortReferenceSamples = 24000;
constexpr size_t   kLongReferenceSamples  = 193920;
constexpr uint64_t kShortReferenceFrames  = 13;
constexpr uint64_t kLongReferenceFrames   = 101;

// The ids Task 10's dispatch will pass down from
// Model::tokenize_reference_transcript. Opaque here by construction:
// create_icl_profile never touches a vocabulary, so nothing below depends on
// these being real ids for the transcript beside them -- only on them being
// carried through unchanged.
std::vector<int32_t> reference_text_ids() {
    return { 9707, 11, 1879, 13 };
}

void add_codec_conv(ggml_context *            context,
                    Conv1dWeights &           target,
                    int64_t                   kernel,
                    int64_t                   in,
                    int64_t                   out,
                    std::vector<WeightDraw> & draws) {
    // Scaled down by the fan-in so an eleven-convolution stack neither
    // saturates nor decays to nothing before the transformer sees it -- the
    // same scaling tests/qwen3_tts_codec_encoder_test.cpp's own fixture uses.
    const float scale = 2.0f / float(std::sqrt(double(kernel * in)));
    target.weight     = ggml_new_tensor_3d(context, GGML_TYPE_F32, kernel, in, out);
    target.bias       = ggml_new_tensor_1d(context, GGML_TYPE_F32, out);
    draws.push_back({ target.weight, scale, 0.0f });
    draws.push_back({ target.bias, 0.05f, 0.0f });
}

void add_codec_weights(ggml_context * context, CodecEncoderWeights & weights, std::vector<WeightDraw> & draws) {
    add_codec_conv(context, weights.stem, 7, 1, kCodecStem, draws);
    weights.stages.assign(4, synth::qwen3tts::CodecEncoderStage{});
    int64_t width = kCodecStem;
    for (size_t stage = 0; stage < weights.stages.size(); ++stage) {
        synth::qwen3tts::CodecEncoderStage & into = weights.stages[stage];
        add_codec_conv(context, into.bottleneck_in, 3, width, width / 2, draws);
        add_codec_conv(context, into.bottleneck_out, 1, width / 2, width, draws);
        add_codec_conv(context, into.stride_conv, kCodecStageKernels[stage], width, kCodecStageWidths[stage], draws);
        width = kCodecStageWidths[stage];
    }
    add_codec_conv(context, weights.tail, 3, width, kCodecHidden, draws);
    // No bias: the frame downsampler is the one convolution here built with
    // `bias=False`, which is why the catalog holds it as a bare tensor.
    weights.downsample = ggml_new_tensor_3d(context, GGML_TYPE_F32, 4, kCodecHidden, kCodecHidden);
    draws.push_back({ weights.downsample, 0.15f, 0.0f });

    weights.layers.assign(kCodecLayers, synth::qwen3tts::CodecEncoderTransformerLayerWeights{});
    for (synth::qwen3tts::CodecEncoderTransformerLayerWeights & layer : weights.layers) {
        // LayerNorm here carries a weight AND a bias, unlike the decoder's
        // RMSNorm; the gain is drawn about one, which is what a trained one
        // looks like.
        layer.input_layernorm.weight = ggml_new_tensor_1d(context, GGML_TYPE_F32, kCodecHidden);
        draws.push_back({ layer.input_layernorm.weight, 0.1f, 1.0f });
        layer.input_layernorm.bias = ggml_new_tensor_1d(context, GGML_TYPE_F32, kCodecHidden);
        draws.push_back({ layer.input_layernorm.bias, 0.05f, 0.0f });
        layer.q_proj = ggml_new_tensor_2d(context, GGML_TYPE_F32, kCodecHidden, kCodecHidden);
        draws.push_back({ layer.q_proj, 0.2f, 0.0f });
        layer.k_proj = ggml_new_tensor_2d(context, GGML_TYPE_F32, kCodecHidden, kCodecHidden);
        draws.push_back({ layer.k_proj, 0.2f, 0.0f });
        layer.v_proj = ggml_new_tensor_2d(context, GGML_TYPE_F32, kCodecHidden, kCodecHidden);
        draws.push_back({ layer.v_proj, 0.2f, 0.0f });
        layer.o_proj = ggml_new_tensor_2d(context, GGML_TYPE_F32, kCodecHidden, kCodecHidden);
        draws.push_back({ layer.o_proj, 0.2f, 0.0f });
        // MimiLayerScale initialises at 0.01 and stays small after training.
        layer.self_attn_layer_scale = ggml_new_tensor_1d(context, GGML_TYPE_F32, kCodecHidden);
        draws.push_back({ layer.self_attn_layer_scale, 0.02f, 0.0f });
        layer.post_attention_layernorm.weight = ggml_new_tensor_1d(context, GGML_TYPE_F32, kCodecHidden);
        draws.push_back({ layer.post_attention_layernorm.weight, 0.1f, 1.0f });
        layer.post_attention_layernorm.bias = ggml_new_tensor_1d(context, GGML_TYPE_F32, kCodecHidden);
        draws.push_back({ layer.post_attention_layernorm.bias, 0.05f, 0.0f });
        layer.fc1 = ggml_new_tensor_2d(context, GGML_TYPE_F32, kCodecHidden, kCodecIntermediate);
        draws.push_back({ layer.fc1, 0.2f, 0.0f });
        layer.fc2 = ggml_new_tensor_2d(context, GGML_TYPE_F32, kCodecIntermediate, kCodecHidden);
        draws.push_back({ layer.fc2, 0.1f, 0.0f });
        layer.mlp_layer_scale = ggml_new_tensor_1d(context, GGML_TYPE_F32, kCodecHidden);
        draws.push_back({ layer.mlp_layer_scale, 0.02f, 0.0f });
    }

    // `input_proj` is a kernel-one convolution with no bias, so ggml reports
    // it [1, in, out]; `output_proj` is deliberately left null, because it is
    // a DECODE-time tensor the encode path must never read.
    auto add_quantizer = [&](synth::qwen3tts::CodecQuantizerWeights & target, int64_t stages) {
        target.input_proj = ggml_new_tensor_3d(context, GGML_TYPE_F32, 1, kCodecHidden, kProjected);
        draws.push_back({ target.input_proj, 0.25f, 0.0f });
        target.codebooks.assign(size_t(stages), nullptr);
        for (int64_t stage = 0; stage < stages; ++stage) {
            target.codebooks[size_t(stage)] = ggml_new_tensor_2d(context, GGML_TYPE_F32, kProjected, kCodebookSize);
            draws.push_back({ target.codebooks[size_t(stage)], 0.6f, 0.0f });
        }
    };
    add_quantizer(weights.semantic, kSemanticGroups);
    // Exactly the fifteen the grid reads. The real checkpoint carries
    // thirty-one, and that a wrapper must take the group count from
    // `quantizer_count` rather than from the resolved codebook list is Task
    // 5's own coverage (tests/qwen3_tts_codec_encoder_test.cpp), not this
    // file's to repeat.
    add_quantizer(weights.acoustic, kGroups - kSemanticGroups);
}

struct IclFixture {
    ggml_backend_t        backend = nullptr;
    Context               persistent;
    ggml_backend_buffer_t buffer = nullptr;
    HParams               hparams;
    SpeakerEncoderWeights speaker;
    CodecEncoderWeights   codec;

    IclFixture()                               = default;
    IclFixture(const IclFixture &)             = delete;
    IclFixture & operator=(const IclFixture &) = delete;

    ~IclFixture() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
};

bool build_icl_fixture(IclFixture & fixture) {
    ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (device == nullptr) {
        return false;
    }
    fixture.backend = ggml_backend_dev_init(device, nullptr);
    if (fixture.backend == nullptr) {
        return false;
    }
    fixture.hparams                                        = make_hparams();
    // `codebook_dim` is this port's name for the LATENT width; the projected
    // space the quantizer decides in is half of it. catalog.cpp derives both
    // the same way, so the fixture must too.
    fixture.hparams.codec.decoder.codebook_dim             = uint32_t(kCodecHidden);
    fixture.hparams.codec.decoder.codebook_size            = uint32_t(kCodebookSize);
    fixture.hparams.codec.decoder.quantizer_count          = uint32_t(kGroups);
    fixture.hparams.codec.decoder.semantic_quantizer_count = uint32_t(kSemanticGroups);

    fixture.persistent  = make_context(ggml_tensor_overhead() * 512);
    ggml_context * pctx = fixture.persistent.get();
    if (pctx == nullptr) {
        return false;
    }
    std::vector<WeightDraw> draws;
    add_speaker_weights(pctx, fixture.speaker, draws);
    add_codec_weights(pctx, fixture.codec, draws);

    fixture.buffer = ggml_backend_alloc_ctx_tensors(pctx, fixture.backend);
    if (fixture.buffer == nullptr) {
        return false;
    }
    fill_weights(kSeed, draws);
    return true;
}

// Prepares an ICL Profile from `pcm` and the fixture's own weights, with the
// arguments every success-path check below shares.
synth_status_t prepare_icl(const IclFixture &                  fixture,
                           const std::vector<float> &          pcm,
                           std::shared_ptr<const IclProfile> & profile,
                           const char *&                       code,
                           const char *&                       message) {
    return synth::qwen3tts::create_icl_profile(fixture.hparams, fixture.speaker, fixture.codec, pcm,
                                               /*transcript=*/"hello there", reference_text_ids(),
                                               /*language_tag=*/"english", /*threads=*/1, profile, code, message);
}

// --- The mode the Profile names. This is the rule Step 2's inversion targets
// (make create_icl_profile write CloneMode::XVector and this check is what
// fails), and it is the whole point of D4: the mode is decided here, at
// preparation, so a Serialized Profile has one unambiguous meaning.
int test_an_icl_profile_reports_the_icl_mode() {
    IclFixture fixture;
    SYNTH_TEST_CHECK(build_icl_fixture(fixture));
    const char *                      code    = nullptr;
    const char *                      message = nullptr;
    std::shared_ptr<const IclProfile> profile;
    SYNTH_TEST_CHECK(prepare_icl(fixture, speech_of(kShortReferenceSamples), profile, code, message) == SYNTH_OK);
    SYNTH_TEST_CHECK(profile != nullptr);
    SYNTH_TEST_CHECK(code == nullptr);
    SYNTH_TEST_CHECK(message == nullptr);
    SYNTH_TEST_CHECK(profile->speaker.mode == CloneMode::Icl);
    SYNTH_TEST_CHECK(profile->speaker.language_tag == "english");
    return 0;
}

// --- The layout contract IclProfile's own header comment states, exercised
// the exact way src/synthesize.cpp:1073-1086 exercises it: recover the
// type-erased payload as an XVectorProfile and read `.mode` off it BEFORE
// knowing which mode it is. One ProfileFamilyTag covers both of this family's
// clone modes precisely because that read works
// (voice-profile-handle.h:21-26), so it has to be well-defined for an ICL
// payload and not merely usually-right.
//
// The compile-time half of this rule is the static_assert set in profile.h:
// move `speaker` off offset 0 and the BUILD fails rather than this check.
int test_an_icl_payload_reads_back_through_an_x_vector_pointer() {
    IclFixture fixture;
    SYNTH_TEST_CHECK(build_icl_fixture(fixture));
    const char *                      code    = nullptr;
    const char *                      message = nullptr;
    std::shared_ptr<const IclProfile> profile;
    SYNTH_TEST_CHECK(prepare_icl(fixture, speech_of(kShortReferenceSamples), profile, code, message) == SYNTH_OK);
    SYNTH_TEST_CHECK(profile != nullptr);

    const std::shared_ptr<const void> erased      = profile;
    const auto *                      as_x_vector = static_cast<const XVectorProfile *>(erased.get());
    SYNTH_TEST_CHECK(as_x_vector->mode == CloneMode::Icl);
    SYNTH_TEST_CHECK(as_x_vector->x_vector == profile->speaker.x_vector);
    SYNTH_TEST_CHECK(as_x_vector->ref_rms == profile->speaker.ref_rms);
    SYNTH_TEST_CHECK(as_x_vector->language_tag == profile->speaker.language_tag);
    return 0;
}

// --- All three of D5's ICL rows are present, INCLUDING the speaker
// embedding: the design's table marks the `[1024]` embedding `yes` in both
// columns because upstream inserts it regardless of mode, so an ICL Profile
// is an x-vector Profile plus two things rather than an alternative to one.
int test_an_icl_profile_carries_all_three_of_d5s_rows() {
    IclFixture fixture;
    SYNTH_TEST_CHECK(build_icl_fixture(fixture));
    const char *                      code    = nullptr;
    const char *                      message = nullptr;
    std::shared_ptr<const IclProfile> profile;
    SYNTH_TEST_CHECK(prepare_icl(fixture, speech_of(kShortReferenceSamples), profile, code, message) == SYNTH_OK);
    SYNTH_TEST_CHECK(profile != nullptr);

    // Row 1: the speaker embedding, at the package's own declared width, and
    // a positive ref_rms behind it.
    SYNTH_TEST_CHECK(profile->speaker.x_vector.size() == fixture.hparams.speaker_encoder.enc_dim);
    SYNTH_TEST_CHECK(profile->speaker.ref_rms > 0.0f);
    // Row 2: the reference codes, `groups * frames` of them, every one inside
    // the package's own codebook. What `frames` should BE for this clip is
    // deliberately not asserted here -- that is
    // test_the_icl_frame_count_follows_the_reference_length's own rule, and
    // repeating it here would only mean a frame-count fault reported itself
    // at this check instead of at the one aimed at it (measured: it did).
    SYNTH_TEST_CHECK(profile->groups == uint64_t(kGroups));
    SYNTH_TEST_CHECK(profile->frames > 0);
    SYNTH_TEST_CHECK(profile->codes.size() == size_t(profile->groups) * size_t(profile->frames));
    for (int32_t value : profile->codes) {
        SYNTH_TEST_CHECK(value >= 0 && value < int32_t(kCodebookSize));
    }
    // Row 3: the reference text token ids, carried through unchanged.
    SYNTH_TEST_CHECK(profile->reference_text_ids == reference_text_ids());
    return 0;
}

// --- The grid the Profile carries is the ENCODER'S grid, element for
// element, in its own GROUP-FASTEST order. Checked through the production
// entry point codec-encoder-host.h keeps for exactly this
// (`codes_equal`, whose header comment names this call site), against a
// second, independent encode of the same clip -- and against a STAGE-MAJOR
// permutation of the same numbers, which has the right element count, the
// right value range and the wrong order. That permutation is the buffer a
// reader who takes `[16, T]` for a numpy shape would build, and it is what
// any transpose added between the encoder and this payload would produce.
int test_the_icl_code_grid_is_the_encoders_own_grid() {
    IclFixture fixture;
    SYNTH_TEST_CHECK(build_icl_fixture(fixture));
    const std::vector<float>          pcm     = speech_of(kShortReferenceSamples);
    const char *                      code    = nullptr;
    const char *                      message = nullptr;
    std::shared_ptr<const IclProfile> profile;
    SYNTH_TEST_CHECK(prepare_icl(fixture, pcm, profile, code, message) == SYNTH_OK);
    SYNTH_TEST_CHECK(profile != nullptr);

    synth::qwen3tts::CodecEncoding encoding;
    const char *                   encode_code    = nullptr;
    const char *                   encode_message = nullptr;
    SYNTH_TEST_CHECK(synth::qwen3tts::encode_codec_reference(fixture.hparams, fixture.codec, pcm, 1, encoding,
                                                             encode_code, encode_message) == SYNTH_OK);
    SYNTH_TEST_CHECK(encoding.groups == profile->groups);
    SYNTH_TEST_CHECK(encoding.frames == profile->frames);
    SYNTH_TEST_CHECK(synth::qwen3tts::codes_equal(encoding, profile->codes.data(), int64_t(profile->frames)));

    // Without a spread the permutation below is the identity and the
    // inequality that follows would hold for the wrong reason.
    bool spread = false;
    for (size_t index = 1; index < profile->codes.size(); ++index) {
        spread = spread || profile->codes[index] != profile->codes[0];
    }
    SYNTH_TEST_CHECK(spread);

    std::vector<int32_t> stage_major(profile->codes.size(), 0);
    for (uint64_t frame = 0; frame < profile->frames; ++frame) {
        for (uint64_t group = 0; group < profile->groups; ++group) {
            stage_major[size_t(group * profile->frames + frame)] =
                profile->codes[size_t(frame * profile->groups + group)];
        }
    }
    SYNTH_TEST_CHECK(stage_major != profile->codes);
    SYNTH_TEST_CHECK(!synth::qwen3tts::codes_equal(encoding, stage_major.data(), int64_t(profile->frames)));
    return 0;
}

// --- The codes and the reference audio agree on length. Pure geometry, which
// is why drawn weights serve: codec_encoder_geometry walks the convolutions'
// own kernel extents and the channel widths never enter the arithmetic, so
// this narrow fixture lands on the same frame counts a real clip does. 24,000
// samples is 12.5 frames and becomes 13, which is the number a floor divide
// gets wrong; 193,920 is a whole 101.
int test_the_icl_frame_count_follows_the_reference_length() {
    IclFixture fixture;
    SYNTH_TEST_CHECK(build_icl_fixture(fixture));

    // The two literals above are anchored to the fixture's own stride product
    // rather than left as bare numbers: if this ever stops being 1920, the
    // frame counts below stop meaning what their comment says they mean.
    synth::qwen3tts::CodecEncoderGeometry geometry;
    SYNTH_TEST_CHECK(synth::qwen3tts::codec_encoder_geometry(fixture.codec, int64_t(kShortReferenceSamples), geometry));
    SYNTH_TEST_CHECK(geometry.samples_per_frame == kCodecSamplesPerFrame);

    const struct {
        size_t   samples;
        uint64_t frames;
    } cases[] = {
        { kShortReferenceSamples, kShortReferenceFrames },
        { kLongReferenceSamples,  kLongReferenceFrames  },
    };

    for (const auto & one : cases) {
        const char *                      code    = nullptr;
        const char *                      message = nullptr;
        std::shared_ptr<const IclProfile> profile;
        SYNTH_TEST_CHECK(prepare_icl(fixture, speech_of(one.samples), profile, code, message) == SYNTH_OK);
        SYNTH_TEST_CHECK(profile != nullptr);
        SYNTH_TEST_CHECK(profile->frames == one.frames);
        SYNTH_TEST_CHECK(profile->codes.size() == size_t(kGroups) * size_t(one.frames));
        const uint64_t ceiling =
            (uint64_t(one.samples) + uint64_t(kCodecSamplesPerFrame) - 1) / uint64_t(kCodecSamplesPerFrame);
        SYNTH_TEST_CHECK(profile->frames == ceiling);
        std::printf("    icl: %llu samples -> %llu frames x %llu groups\n", (unsigned long long) one.samples,
                    (unsigned long long) profile->frames, (unsigned long long) profile->groups);
    }
    return 0;
}

// --- D4 from the ICL side: a blank transcript names the OTHER mode, and this
// function refuses it rather than quietly preparing the weaker Profile the
// caller did not ask for. "Blank" is empty OR whitespace-only, which the
// design's section 9 error table gives one row and one status -- the
// distinction Plan 2's test_a_whitespace_transcript_is_rejected_too was
// written early to preserve, now landing on the function for which a blank
// transcript is a malformed input rather than simply the wrong mode.
//
// The fourth case is the REVERSE half-present state -- ids supplied with no
// transcript to have produced them -- which this same check refuses.
int test_icl_refuses_a_blank_transcript() {
    IclFixture fixture;
    SYNTH_TEST_CHECK(build_icl_fixture(fixture));

    const struct {
        const char *         transcript;
        std::vector<int32_t> ids;
    } cases[] = {
        { "",         reference_text_ids() },
        { "   ",      reference_text_ids() },
        { " \t\r\n ", reference_text_ids() },
        { "",         {}                   },
    };

    for (const auto & one : cases) {
        const char *                      code    = nullptr;
        const char *                      message = nullptr;
        std::shared_ptr<const IclProfile> profile;
        const synth_status_t              status = synth::qwen3tts::create_icl_profile(
            fixture.hparams, fixture.speaker, fixture.codec, speech_of(kShortReferenceSamples), one.transcript, one.ids,
            "english", 1, profile, code, message);
        SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(profile == nullptr);
        SYNTH_TEST_CHECK(code != nullptr && std::strcmp(code, "voice_profile.transcript_blank") == 0);
        SYNTH_TEST_CHECK(message != nullptr);
    }
    return 0;
}

// --- The half-present state, in the direction only the ids guard catches: a
// real transcript with no ids behind it. Refused rather than trusted -- an
// ICL prompt built from reference codes with no reference text is a
// plausible-sounding wrong prompt, not something anything downstream reports.
// This is the arm Step 2's second inversion targets (drop the guard and this
// check is what fails; the blank check above cannot cover it, because the
// transcript here is not blank).
int test_icl_refuses_a_transcript_without_its_token_ids() {
    IclFixture fixture;
    SYNTH_TEST_CHECK(build_icl_fixture(fixture));
    const char *                      code    = nullptr;
    const char *                      message = nullptr;
    std::shared_ptr<const IclProfile> profile;
    const synth_status_t              status = synth::qwen3tts::create_icl_profile(
        fixture.hparams, fixture.speaker, fixture.codec, speech_of(kShortReferenceSamples), "hello there",
        /*reference_text_ids=*/{}, "english", 1, profile, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(profile == nullptr);
    SYNTH_TEST_CHECK(code != nullptr && std::strcmp(code, "voice_profile.reference_text_ids_missing") == 0);
    SYNTH_TEST_CHECK(message != nullptr);
    return 0;
}

// --- The tie kMaxLanguageTagLength documents applies to both kinds or it
// applies to neither. The reviewer-measured defect it closes -- our own
// writer emitting an envelope our own reader refuses -- does not care which
// kind wrote the envelope, so the ICL creation path carries the same refusal
// the x-vector one does (test_language_tag_too_long_is_rejected_at_creation).
int test_icl_refuses_an_oversized_language_tag() {
    IclFixture fixture;
    SYNTH_TEST_CHECK(build_icl_fixture(fixture));
    const std::string                 oversized(size_t(synth::qwen3tts::kMaxLanguageTagLength) + 8, 'x');
    const char *                      code    = nullptr;
    const char *                      message = nullptr;
    std::shared_ptr<const IclProfile> profile;
    const synth_status_t              status = synth::qwen3tts::create_icl_profile(
        fixture.hparams, fixture.speaker, fixture.codec, speech_of(kShortReferenceSamples), "hello there",
        reference_text_ids(), oversized, 1, profile, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(profile == nullptr);
    SYNTH_TEST_CHECK(code != nullptr && std::strcmp(code, "voice_profile.language_tag_too_long") == 0);
    SYNTH_TEST_CHECK(message != nullptr);
    return 0;
}

// --- The two refusals this function inherits rather than names: a package
// with no encoders at all, and a digitally silent reference. Both reach the
// ICL path exactly as they reach the x-vector one -- `has_speaker_encoder` is
// the single flag the catalog resolves BOTH `speaker_encoder.*` and
// `codec.encoder.*` under (Model::prepare_codec_reference's own guard), and
// the silent-reference rejection is raised by name by both encode chains.
int test_icl_inherits_the_shared_refusals() {
    IclFixture fixture;
    SYNTH_TEST_CHECK(build_icl_fixture(fixture));

    IclFixture no_encoder;
    SYNTH_TEST_CHECK(build_icl_fixture(no_encoder));
    no_encoder.hparams.has_speaker_encoder    = false;
    const char *                      code    = nullptr;
    const char *                      message = nullptr;
    std::shared_ptr<const IclProfile> profile;
    SYNTH_TEST_CHECK(prepare_icl(no_encoder, speech_of(kShortReferenceSamples), profile, code, message) ==
                     SYNTH_ERR_UNSUPPORTED_VOICE);
    SYNTH_TEST_CHECK(profile == nullptr);

    const std::vector<float> silence(kShortReferenceSamples, 0.0f);
    SYNTH_TEST_CHECK(prepare_icl(fixture, silence, profile, code, message) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(profile == nullptr);
    SYNTH_TEST_CHECK(code != nullptr && std::strcmp(code, "voice_profile.reference_silent") == 0);
    return 0;
}

// =============================================================================
// Task 8: the v1 Serialized Profile envelope -- round trip and tamper matrix.
//
// Unlike tests/omnivoice_serialize_test.cpp's own tamper matrix, every arm
// below drives arch/qwen3-tts/profile.cpp's own
// serialize_x_vector_profile/load_profile_from_memory DIRECTLY, never a
// public synth_voice_profile_serialize/synth_voice_profile_load_from_memory
// seam: Task 9 is what opens that seam for this family, and this task's own
// brief is explicit that it "does not open the public seam." There is also
// no synthetic-package harness for this family yet (unlike OmniVoice's
// tests/omnivoice_synthetic_package.h), so every payload here is hand-built,
// the same "no real forward pass, no real Model" style
// tests/omnivoice_serialize_test.cpp's own header comment describes for its
// OWN family-level fixtures.
//
// The byte-level surgery helpers below (find_u8_32_value/find_string_value/
// find_u32_value, and the make_header/put_kv_*/common_kv_bytes hand-built-
// buffer builders) are this file's own second, independent transcription of
// arch/qwen3-tts/profile.cpp's own on-disk encoding -- the same tolerance
// tests/omnivoice_serialize_test.cpp's own header comment already accepts
// for a small, stable helper duplicated for a different purpose (constructing
// adversarial buffers, never deciding what the loader does with them: every
// actual PASS/FAIL verdict below still runs through the real
// load_profile_from_memory).
// =============================================================================

void put_bytes(std::vector<uint8_t> & out, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    out.insert(out.end(), bytes, bytes + size);
}

template <typename T> void put(std::vector<uint8_t> & out, T value) {
    put_bytes(out, &value, sizeof(value));
}

void put_gguf_string(std::vector<uint8_t> & out, const std::string & value) {
    put<uint64_t>(out, uint64_t(value.size()));
    put_bytes(out, value.data(), value.size());
}

void pad_to_alignment(std::vector<uint8_t> & out, size_t alignment) {
    while (out.size() % alignment != 0) {
        out.push_back(0);
    }
}

// Locates the byte OFFSET of a 32-element uint8 array KV's own VALUE bytes,
// by searching for that entry's on-disk encoding prefix -- a test-side
// mirror of profile.cpp's own find_u8_32_value_offset, used here only to
// locate fields to tamper with.
bool find_u8_32_value(const std::vector<uint8_t> & bytes, const std::string & key, size_t & out_offset) {
    std::vector<uint8_t> needle;
    put<uint64_t>(needle, uint64_t(key.size()));
    put_bytes(needle, key.data(), key.size());
    put<int32_t>(needle, int32_t(GGUF_TYPE_ARRAY));
    put<int32_t>(needle, int32_t(GGUF_TYPE_UINT8));
    put<uint64_t>(needle, uint64_t(32));
    const auto found = std::search(bytes.begin(), bytes.end(), needle.begin(), needle.end());
    if (found == bytes.end()) {
        return false;
    }
    const size_t offset = size_t(found - bytes.begin()) + needle.size();
    if (offset + 32 > bytes.size()) {
        return false;
    }
    out_offset = offset;
    return true;
}

bool find_string_value(const std::vector<uint8_t> & bytes,
                       const std::string &          key,
                       size_t &                     out_offset,
                       size_t &                     out_length) {
    std::vector<uint8_t> needle;
    put<uint64_t>(needle, uint64_t(key.size()));
    put_bytes(needle, key.data(), key.size());
    put<int32_t>(needle, int32_t(GGUF_TYPE_STRING));
    const auto found = std::search(bytes.begin(), bytes.end(), needle.begin(), needle.end());
    if (found == bytes.end()) {
        return false;
    }
    size_t offset = size_t(found - bytes.begin()) + needle.size();
    if (offset + sizeof(uint64_t) > bytes.size()) {
        return false;
    }
    uint64_t length = 0;
    std::memcpy(&length, bytes.data() + offset, sizeof(length));
    offset += sizeof(length);
    if (offset + length > bytes.size()) {
        return false;
    }
    out_offset = offset;
    out_length = size_t(length);
    return true;
}

bool find_u32_value(const std::vector<uint8_t> & bytes, const std::string & key, size_t & out_offset) {
    std::vector<uint8_t> needle;
    put<uint64_t>(needle, uint64_t(key.size()));
    put_bytes(needle, key.data(), key.size());
    put<int32_t>(needle, int32_t(GGUF_TYPE_UINT32));
    const auto found = std::search(bytes.begin(), bytes.end(), needle.begin(), needle.end());
    if (found == bytes.end()) {
        return false;
    }
    const size_t offset = size_t(found - bytes.begin()) + needle.size();
    if (offset + sizeof(uint32_t) > bytes.size()) {
        return false;
    }
    out_offset = offset;
    return true;
}

// ---------------------------------------------------------------------------
// Hand-built raw GGUF buffers, for the structural tamper arms (an unknown
// key, a known key with the wrong type, an unrecognized `kind`) that a
// simple byte-flip of a valid envelope cannot produce cleanly: `common_kv_bytes`
// transcribes set_common_metadata's own 8-key order (profile.cpp), NOT
// guessed -- the same discipline tests/omnivoice_serialize_test.cpp's own
// common_kv_bytes holds itself to.
// ---------------------------------------------------------------------------

std::vector<uint8_t> make_header(int64_t n_tensors, int64_t n_kv) {
    std::vector<uint8_t> bytes;
    put_bytes(bytes, GGUF_MAGIC, 4);
    put<uint32_t>(bytes, uint32_t(GGUF_VERSION));
    put<int64_t>(bytes, n_tensors);
    put<int64_t>(bytes, n_kv);
    return bytes;
}

void put_kv_string(std::vector<uint8_t> & out, const std::string & key, const std::string & value) {
    put_gguf_string(out, key);
    put<int32_t>(out, int32_t(GGUF_TYPE_STRING));
    put_gguf_string(out, value);
}

void put_kv_u32(std::vector<uint8_t> & out, const std::string & key, uint32_t value) {
    put_gguf_string(out, key);
    put<int32_t>(out, int32_t(GGUF_TYPE_UINT32));
    put<uint32_t>(out, value);
}

void put_kv_f32(std::vector<uint8_t> & out, const std::string & key, float value) {
    put_gguf_string(out, key);
    put<int32_t>(out, int32_t(GGUF_TYPE_FLOAT32));
    put<float>(out, value);
}

// A metadata entry whose declared type tag is an arbitrary raw int32 rather
// than one of the GGUF_TYPE_* enumerators -- the one shape the put_kv_*
// helpers above cannot express, since each of them writes a tag this project
// itself considers valid. The four payload bytes are never read: the tag is
// compared against the key's own kPrescanKnownKeys type first.
void put_kv_raw_type(std::vector<uint8_t> & out, const std::string & key, int32_t type_tag) {
    put_gguf_string(out, key);
    put<int32_t>(out, type_tag);
    put<float>(out, 0.5f);
}

void put_kv_u8_array32(std::vector<uint8_t> & out, const std::string & key) {
    put_gguf_string(out, key);
    put<int32_t>(out, int32_t(GGUF_TYPE_ARRAY));
    put<int32_t>(out, int32_t(GGUF_TYPE_UINT8));
    put<uint64_t>(out, uint64_t(32));
    const std::vector<uint8_t> value(32, 0);
    put_bytes(out, value.data(), value.size());
}

// The 8 metadata keys set_common_metadata (arch/qwen3-tts/profile.cpp)
// writes for every envelope, in that function's own order, ending with
// `kind` itself.
std::vector<uint8_t> common_kv_bytes(const std::string & kind) {
    std::vector<uint8_t> bytes;
    put_kv_string(bytes, "general.architecture", "synthprofile");
    put_kv_u32(bytes, "synthesize.voice_profile.format_version", 1);
    put_kv_string(bytes, "synthesize.voice_profile.model_family", "qwen3-tts");
    put_kv_string(bytes, "synthesize.voice_profile.schema", "qwen3-tts-voice-clone");
    put_kv_u32(bytes, "synthesize.voice_profile.schema_version", 1);
    put_kv_u8_array32(bytes, "synthesize.voice_profile.compatibility_id");
    put_kv_u8_array32(bytes, "synthesize.voice_profile.content_sha256");
    put_kv_string(bytes, "synthesize.voice_profile.kind", kind);
    return bytes;
}

// Appends a well-formed "profile.x_vector" tensor-info entry (the writer's
// own exact shape: n_dims=1, type F32, offset 0) plus zero-filled payload
// bytes and alignment padding, onto a buffer whose header already declares
// n_tensors=1.
void append_x_vector_tensor(std::vector<uint8_t> & bytes, int64_t element_count) {
    put_gguf_string(bytes, "profile.x_vector");
    put<uint32_t>(bytes, uint32_t(1));  // n_dims
    put<int64_t>(bytes, element_count);
    put<int32_t>(bytes, int32_t(GGML_TYPE_F32));
    put<uint64_t>(bytes, uint64_t(0));  // offset 0, the only tensor
    pad_to_alignment(bytes, GGUF_DEFAULT_ALIGNMENT);
    const std::vector<uint8_t> payload(size_t(element_count) * sizeof(float), 0);
    put_bytes(bytes, payload.data(), payload.size());
    pad_to_alignment(bytes, GGUF_DEFAULT_ALIGNMENT);
}

// Assembles a complete, hand-built envelope from a raw KV byte blob (built
// from common_kv_bytes plus whatever else a given arm appends) and the
// x-vector's own element count. The content_sha256 stored inside `kv` is
// whatever put_kv_u8_array32 wrote (32 zero bytes) -- deliberately never a
// real digest, since every structural arm below is rejected before
// load_profile_from_memory's own digest check ever runs.
std::vector<uint8_t> assemble_hand_built(const std::vector<uint8_t> & kv, int64_t n_kv, int64_t element_count) {
    std::vector<uint8_t> bytes = make_header(/*n_tensors=*/1, n_kv);
    put_bytes(bytes, kv.data(), kv.size());
    append_x_vector_tensor(bytes, element_count);
    return bytes;
}

void fill_compatibility_id(uint8_t (&id)[32]) {
    // Arbitrary but non-zero, so a reviewer scanning a hex dump never
    // mistakes this for an all-zero placeholder that was never set.
    for (size_t index = 0; index < sizeof(id); ++index) {
        id[index] = uint8_t(index + 7);
    }
}

std::shared_ptr<XVectorProfile> make_serializable_profile(uint32_t            enc_dim,
                                                          float               ref_rms,
                                                          const std::string & language_tag) {
    auto profile  = std::make_shared<XVectorProfile>();
    profile->mode = CloneMode::XVector;
    LcgStream stream(kSeed + 900);
    profile->x_vector     = stream.fill(enc_dim, 1.0f);
    profile->ref_rms      = ref_rms;
    profile->language_tag = language_tag;
    return profile;
}

// Builds a real, well-formed round-trippable envelope through the REAL
// writer -- the base every byte-flip tamper arm below starts from.
std::vector<uint8_t> build_valid_bytes(uint32_t enc_dim, const uint8_t (&compatibility_id)[32]) {
    const std::shared_ptr<XVectorProfile> profile = make_serializable_profile(enc_dim, 0.5f, "en-US");
    std::vector<uint8_t>                  bytes;
    if (synth::qwen3tts::serialize_x_vector_profile(*profile, compatibility_id, bytes) != SYNTH_OK) {
        return {};
    }
    return bytes;
}

// --- Round trip: a hand-built payload -> the real writer -> the real
// reader -> every field matches.
int test_round_trip_of_a_well_formed_profile() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    const std::shared_ptr<XVectorProfile> profile = make_serializable_profile(kProfileEncDim, 0.5f, "en-US");

    std::vector<uint8_t> bytes;
    SYNTH_TEST_CHECK(synth::qwen3tts::serialize_x_vector_profile(*profile, compatibility_id, bytes) == SYNTH_OK);
    SYNTH_TEST_CHECK(!bytes.empty());

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        kProfileEncDim, bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_OK);
    SYNTH_TEST_CHECK(family_tag == synth::ProfileFamilyTag::Qwen3TtsClone);
    SYNTH_TEST_CHECK(payload != nullptr);
    SYNTH_TEST_CHECK(code == nullptr);
    SYNTH_TEST_CHECK(message == nullptr);

    const auto * reloaded = static_cast<const XVectorProfile *>(payload.get());
    SYNTH_TEST_CHECK(reloaded->mode == CloneMode::XVector);
    SYNTH_TEST_CHECK(reloaded->x_vector == profile->x_vector);
    SYNTH_TEST_CHECK(reloaded->ref_rms == profile->ref_rms);
    SYNTH_TEST_CHECK(reloaded->language_tag == profile->language_tag);
    return 0;
}

// --- Determinism: two serialize calls over the identical payload produce
// byte-identical output.
int test_serialize_is_deterministic() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    const std::shared_ptr<XVectorProfile> profile = make_serializable_profile(kProfileEncDim, 0.5f, "en-US");
    std::vector<uint8_t>                  first, second;
    SYNTH_TEST_CHECK(synth::qwen3tts::serialize_x_vector_profile(*profile, compatibility_id, first) == SYNTH_OK);
    SYNTH_TEST_CHECK(synth::qwen3tts::serialize_x_vector_profile(*profile, compatibility_id, second) == SYNTH_OK);
    SYNTH_TEST_CHECK(first == second);
    return 0;
}

// --- `model_family` is "omnivoice" -> UNSUPPORTED_VOICE: another family's
// envelope is "not mine", not "corrupt". "omnivoice" and "qwen3-tts" are
// both exactly 9 bytes, so the field is overwritten with the literal string
// the row names rather than an arbitrary XOR-garbled one.
int test_wrong_model_family_is_unsupported_voice() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    std::vector<uint8_t> bytes = build_valid_bytes(kProfileEncDim, compatibility_id);
    SYNTH_TEST_CHECK(!bytes.empty());

    size_t            offset = 0, length = 0;
    const std::string replacement = "omnivoice";
    SYNTH_TEST_CHECK(find_string_value(bytes, "synthesize.voice_profile.model_family", offset, length));
    SYNTH_TEST_CHECK(length == replacement.size());
    std::memcpy(bytes.data() + offset, replacement.data(), replacement.size());

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        kProfileEncDim, bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_UNSUPPORTED_VOICE);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// --- `schema` is a different string -> UNSUPPORTED_VOICE: a future/other
// schema this build cannot read. A single-byte flip is enough (any value
// other than the exact expected string trips the same check the row names);
// tests/omnivoice_serialize_test.cpp's own schema arm uses the identical
// technique.
int test_wrong_schema_is_unsupported_voice() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    std::vector<uint8_t> bytes = build_valid_bytes(kProfileEncDim, compatibility_id);
    SYNTH_TEST_CHECK(!bytes.empty());

    size_t offset = 0, length = 0;
    SYNTH_TEST_CHECK(find_string_value(bytes, "synthesize.voice_profile.schema", offset, length));
    SYNTH_TEST_CHECK(length > 0);
    bytes[offset] ^= 0xFF;

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        kProfileEncDim, bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_UNSUPPORTED_VOICE);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// --- `schema_version` is 2 -> UNSUPPORTED_VOICE: a future version this
// build cannot read.
int test_wrong_schema_version_is_unsupported_voice() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    std::vector<uint8_t> bytes = build_valid_bytes(kProfileEncDim, compatibility_id);
    SYNTH_TEST_CHECK(!bytes.empty());

    size_t offset = 0;
    SYNTH_TEST_CHECK(find_u32_value(bytes, "synthesize.voice_profile.schema_version", offset));
    const uint32_t future_version = 2;
    std::memcpy(bytes.data() + offset, &future_version, sizeof(future_version));

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        kProfileEncDim, bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_UNSUPPORTED_VOICE);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// --- `kind` is "icl" -> INVALID_ARG, deliberately not UNSUPPORTED_VOICE: an
// unrecognized kind inside a schema this build owns is malformed, the exact
// case Plan 3 turns into a success. "icl" (3 bytes) is shorter than
// "x-vector" (8 bytes), so an in-place byte patch of a valid envelope cannot
// produce it without breaking every offset downstream -- this envelope is
// hand-built instead. Everything past the `kind` check (compatibility_id,
// content_sha256, the tensor payload) is checked LATER than `kind` in
// load_profile_from_memory's own order, so it never needs to be genuinely
// valid for this arm: reaching INVALID_ARG here has to be the `kind` check,
// not anything downstream of it.
int test_unrecognized_kind_is_invalid_arg() {
    constexpr int64_t    kProfileEncDim = 11;
    std::vector<uint8_t> kv             = common_kv_bytes("icl");
    put_kv_f32(kv, "synthesize.voice_profile.ref_rms", 0.5f);
    put_kv_string(kv, "synthesize.voice_profile.language_tag", "en");
    const std::vector<uint8_t> bytes = assemble_hand_built(kv, kPrescanKvCountXVector, kProfileEncDim);

    uint8_t compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        uint32_t(kProfileEncDim), bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// --- compatibility id one byte off -> UNSUPPORTED_VOICE: prepared for a
// different package.
int test_wrong_compatibility_id_is_unsupported_voice() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    std::vector<uint8_t> bytes = build_valid_bytes(kProfileEncDim, compatibility_id);
    SYNTH_TEST_CHECK(!bytes.empty());

    size_t offset = 0;
    SYNTH_TEST_CHECK(find_u8_32_value(bytes, "synthesize.voice_profile.compatibility_id", offset));
    bytes[offset] ^= 0xFF;

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    // The expected compatibility_id passed here is the ORIGINAL, unchanged
    // one -- only the envelope's own embedded copy was tampered with.
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        kProfileEncDim, bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_UNSUPPORTED_VOICE);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// --- content_sha256 one byte off -> INVALID_ARG: the bytes were altered.
int test_tampered_content_sha256_is_invalid_arg() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    std::vector<uint8_t> bytes = build_valid_bytes(kProfileEncDim, compatibility_id);
    SYNTH_TEST_CHECK(!bytes.empty());

    size_t offset = 0;
    SYNTH_TEST_CHECK(find_u8_32_value(bytes, "synthesize.voice_profile.content_sha256", offset));
    bytes[offset] ^= 0xFF;

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        kProfileEncDim, bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// --- Truncated buffer -> INVALID_ARG.
int test_truncated_buffer_is_invalid_arg() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    const std::vector<uint8_t> bytes = build_valid_bytes(kProfileEncDim, compatibility_id);
    SYNTH_TEST_CHECK(!bytes.empty());
    const std::vector<uint8_t> truncated(bytes.begin(), bytes.begin() + long(bytes.size() / 2));

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        kProfileEncDim, truncated.data(), truncated.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// --- Alignment not GGUF_DEFAULT_ALIGNMENT -> INVALID_ARG.
//
// The mechanism here is NOT load_profile_from_memory's own post-parse
// `gguf_get_alignment(g) != GGUF_DEFAULT_ALIGNMENT` comparison: ggml only
// ever sets a non-default alignment from a "general.alignment" metadata KV
// (ggml/src/gguf.cpp), and that key is never a member of kPrescanKnownKeys
// (this writer never emits it), so prescan_buffer's own unknown-key
// whitelist rejects any buffer that declares it before gguf_init_from_buffer
// -- and therefore that post-parse comparison -- ever runs. This arm still
// proves the row's own black-box contract (a declared non-default alignment
// is refused with INVALID_ARG); it is caught one layer earlier than the
// specific comparison that would otherwise enforce it, exactly the "no
// special case needed" property profile.cpp's own prescan_buffer header
// comment claims for the identical "general.alignment" hole in ggml's own
// eager reader.
int test_non_default_alignment_is_invalid_arg() {
    constexpr int64_t    kProfileEncDim = 11;
    std::vector<uint8_t> kv             = common_kv_bytes("x-vector");
    put_kv_f32(kv, "synthesize.voice_profile.ref_rms", 0.5f);
    put_kv_u32(kv, "general.alignment", 64);  // in place of language_tag, keeping n_kv == 10
    const std::vector<uint8_t> bytes = assemble_hand_built(kv, kPrescanKvCountXVector, kProfileEncDim);

    uint8_t compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        uint32_t(kProfileEncDim), bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// --- Tensor count 0 -> INVALID_ARG: exactly one x-vector tensor. Patches
// the header's own n_tensors field (byte offset 8: magic[4] + version[4]) on
// a real, otherwise-valid envelope.
//
// As written, prescan_buffer's own `n_tensors != 1` gate rejects this
// immediately. Mutation-tested by temporarily disabling just that
// comparison: `gguf_get_n_tensors(g) != 1`, checked later, still catches
// it -- gguf_init_from_buffer itself parses this buffer fine (it declares
// zero tensor-info entries and finds none, consistent with itself), so the
// mismatch against a REQUIRED width of 1 is a semantic check this project
// owns, not something ggml's own parser would ever refuse on its own. A
// third, independent layer sits behind even that one: with BOTH checks
// disabled, `gguf_find_tensor(g, "profile.x_vector") < 0` still catches it,
// since a context that parsed zero tensor-info entries has no tensor by
// that name to find.
int test_tensor_count_zero_is_invalid_arg() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    std::vector<uint8_t> bytes = build_valid_bytes(kProfileEncDim, compatibility_id);
    SYNTH_TEST_CHECK(bytes.size() > 16);
    const int64_t zero_tensors = 0;
    std::memcpy(bytes.data() + 8, &zero_tensors, sizeof(zero_tensors));

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        kProfileEncDim, bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// --- Tensor count 2 -> INVALID_ARG: same field, the other direction.
//
// Different mechanism from the n_tensors=0 arm above, confirmed by the same
// mutation-testing experiment: with prescan_buffer's own gate disabled,
// gguf_init_from_buffer ITSELF returns null for this buffer -- it tries to
// parse two tensor-info entries, and the "second" one is actually this
// envelope's own tensor DATA bytes misread as tensor-info, which fails
// ggml's own internal length sanity check ("string length ... exceeds
// maximum"). `gguf_get_n_tensors(g) != 1`, checked later in this function,
// is consequently NEVER REACHED for this specific input -- it is not a
// backstop for this arm, unlike the n_tensors=0 arm above where it is.
int test_tensor_count_two_is_invalid_arg() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    std::vector<uint8_t> bytes = build_valid_bytes(kProfileEncDim, compatibility_id);
    SYNTH_TEST_CHECK(bytes.size() > 16);
    const int64_t two_tensors = 2;
    std::memcpy(bytes.data() + 8, &two_tensors, sizeof(two_tensors));

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        kProfileEncDim, bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// --- x-vector length != enc_dim -> INVALID_ARG: a Profile of the wrong
// width builds a wrong-shaped prompt. The envelope itself is genuinely
// valid (built by the real writer for width 11); only the `enc_dim` the
// CALLER declares differs, exactly the way a caller would present a
// serialized Profile from one package to a Model expecting a different
// speaker-encoder width.
int test_x_vector_length_mismatch_is_invalid_arg() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    const std::vector<uint8_t> bytes = build_valid_bytes(kProfileEncDim, compatibility_id);
    SYNTH_TEST_CHECK(!bytes.empty());

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        kProfileEncDim + 1, bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// --- A key outside the prescan whitelist -> INVALID_ARG: refused before
// gguf_init_from_buffer runs. "language_tag" is replaced with a bogus key,
// keeping n_kv == 10.
int test_key_outside_whitelist_is_invalid_arg() {
    constexpr int64_t    kProfileEncDim = 11;
    std::vector<uint8_t> kv             = common_kv_bytes("x-vector");
    put_kv_f32(kv, "synthesize.voice_profile.ref_rms", 0.5f);
    put_kv_string(kv, "synthesize.voice_profile.bogus", "z");
    const std::vector<uint8_t> bytes = assemble_hand_built(kv, kPrescanKvCountXVector, kProfileEncDim);

    uint8_t compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        uint32_t(kProfileEncDim), bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// --- A whitelisted key with the wrong gguf type -> INVALID_ARG: `ref_rms`
// (FLOAT32 in kPrescanKnownKeys) declared as UINT32 instead.
int test_whitelisted_key_wrong_type_is_invalid_arg() {
    constexpr int64_t    kProfileEncDim = 11;
    std::vector<uint8_t> kv             = common_kv_bytes("x-vector");
    put_kv_u32(kv, "synthesize.voice_profile.ref_rms", 42);  // wrong type: should be FLOAT32
    put_kv_string(kv, "synthesize.voice_profile.language_tag", "en");
    const std::vector<uint8_t> bytes = assemble_hand_built(kv, kPrescanKvCountXVector, kProfileEncDim);

    uint8_t compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        uint32_t(kProfileEncDim), bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// --- A type tag no key's spec declares -> INVALID_ARG, at five values
// chosen around gguf_type's own range.
//
// WHICH IS WHICH, since they are not all the same kind of wrong. `gguf_type`
// enumerates 0..13 (GGUF_TYPE_COUNT is 13, a real enumerator), so its
// representable range is 0..15 and only a value outside THAT is undefined
// behaviour to convert. Of the five below, -1, 16, INT32_MAX and INT32_MIN
// are outside it; 13 is inside it -- GGUF_TYPE_COUNT itself, well-defined to
// convert and merely not a type any key declares. So this arm is not five
// demonstrations of the undefined behaviour; it is five values around the
// boundary, four of which would be UB under a converting reader.
//
// WHAT THIS PINS, EXACTLY: the black-box rejection behaviour, at five tag
// values, and nothing more. It is a value-space regression net over the arm
// directly above, not new branch coverage -- prescan_buffer's rule is a plain
// integer inequality against the key's own kPrescanKnownKeys type, so
// INT32_MIN takes the identical branch that arm's UINT32-instead-of-FLOAT32
// already takes. It is kept because the values are the interesting ones and
// they cost nothing to carry, not because it reaches anything new.
//
// WHAT THIS DOES NOT PIN, AND CANNOT: the absence of the enum conversion,
// which is the actual reason these values are interesting. `gguf_type` is an
// unscoped enum with no fixed underlying type, so its value range is 0..15
// and converting anything outside it is undefined behaviour -- on bytes that
// arrive through the public synth_voice_profile_load_from_memory with no
// prior validation. prescan_buffer therefore keeps the tag an `int32_t` and
// only ever compares it. Reinstating `gguf_type type = gguf_type(type_raw);`
// was MEASURED (2026-08-13) to leave this test passing in both the plain and
// the sanitizer build: GCC 13.3's UBSan instruments loads of enum-typed
// lvalues, not that register-resident conversion, and the value is never used
// to index, switch or dispatch, so no defined behaviour changes either. Clang's
// -fsanitize=enum would catch it; this tree has no clang leg.
//
// So if you are here to "simplify" the int32 walk back into a cast: this test
// will not stop you, and it is not evidence that the cast is safe. The reason
// not to is [expr.static.cast]/10, and it lives in prescan_buffer's own
// comment.
//
// The rule that IS pinned here is a POSITIVE one: prescan_buffer compares the
// raw tag against the key's own kPrescanKnownKeys type and refuses anything
// else, rather than range-testing against GGUF_TYPE_COUNT (which would be a
// blacklist against ggml's own enum extent -- see prescan_buffer's own header
// comment and docs/porting/families/omnivoice.md's untrusted-bytes section).
// `ref_rms` is the target because it is whitelisted, so the KV count, the key
// set and the duplicate check all pass and the type comparison is the one
// rule left to do the rejecting.
int test_out_of_range_type_tag_is_invalid_arg() {
    constexpr int64_t kProfileEncDim = 11;
    for (int32_t type_tag :
         { int32_t(-1), int32_t(13), int32_t(16), int32_t(0x7FFFFFFF), std::numeric_limits<int32_t>::min() }) {
        std::vector<uint8_t> kv = common_kv_bytes("x-vector");
        put_kv_raw_type(kv, "synthesize.voice_profile.ref_rms", type_tag);
        put_kv_string(kv, "synthesize.voice_profile.language_tag", "en");
        const std::vector<uint8_t> bytes = assemble_hand_built(kv, kPrescanKvCountXVector, kProfileEncDim);

        uint8_t compatibility_id[32];
        fill_compatibility_id(compatibility_id);
        synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
        std::shared_ptr<const void> payload;
        const char *                code    = nullptr;
        const char *                message = nullptr;
        const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
            uint32_t(kProfileEncDim), bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
        SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(payload == nullptr);
    }
    return 0;
}

// --- n_kv other than the pinned count -> INVALID_ARG. Patches the header's
// own n_kv field (byte offset 16: magic[4] + version[4] + n_tensors[8]) on a
// real, otherwise-valid envelope to one less than kPrescanKvCountXVector. As
// written, prescan_buffer's own KV-count check rejects this immediately,
// before a single KV entry is read. Mutation-tested by temporarily
// disabling just that comparison: the walk still ends up rejected even
// then, because trusting a too-small n_kv stops the KV loop one entry
// early and misreads the real "language_tag" KV bytes that follow as the
// start of the tensor-info section, tripping the tensor-name check instead
// (this exact buffer's own KV entries are otherwise perfectly well-formed,
// so nothing else fires first). The two checks overlap on THIS buffer;
// they do not overlap in general -- a buffer whose entries are still
// well-formed one-for-one with a legitimately different `kind` shape,
// which this format cannot describe until Plan 3 adds a second n_kv value,
// would only be caught by the count check itself.
int test_wrong_n_kv_count_is_invalid_arg() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    std::vector<uint8_t> bytes = build_valid_bytes(kProfileEncDim, compatibility_id);
    SYNTH_TEST_CHECK(bytes.size() > 24);
    const int64_t wrong_n_kv = kPrescanKvCountXVector - 1;
    std::memcpy(bytes.data() + 16, &wrong_n_kv, sizeof(wrong_n_kv));

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        kProfileEncDim, bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// =============================================================================
// Writer-agreement test (the shape of
// tests/omnivoice_serialize_writer_agreement_test.cpp): pins the REAL
// serialize_x_vector_profile's emitted key set/count against the REAL
// kPrescanKnownKeys whitelist (profile.h), independent of profile.cpp's own
// hand-rolled prescan_buffer walk -- parsed here with ggml's own
// gguf_init_from_buffer instead. A key REMOVED from the writer while left in
// kPrescanKnownKeys is a silently too-permissive whitelist that nothing else
// in this file would notice: every tamper arm above starts from either a
// real writer round trip or a hand-built buffer that already targets a
// SPECIFIC known key, so none of them would catch a whitelist entry the
// writer stopped emitting.
// =============================================================================

int test_writer_emits_exactly_the_whitelisted_keys() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    const std::shared_ptr<XVectorProfile> profile = make_serializable_profile(kProfileEncDim, 0.5f, "en-US");
    std::vector<uint8_t>                  bytes;
    SYNTH_TEST_CHECK(synth::qwen3tts::serialize_x_vector_profile(*profile, compatibility_id, bytes) == SYNTH_OK);
    SYNTH_TEST_CHECK(!bytes.empty());

    gguf_init_params init_params{};
    init_params.no_alloc = true;
    init_params.ctx      = nullptr;
    gguf_context * ctx   = gguf_init_from_buffer(bytes.data(), bytes.size(), init_params);
    SYNTH_TEST_CHECK(ctx != nullptr);
    std::set<std::string> emitted_keys;
    const int64_t         n_kv = gguf_get_n_kv(ctx);
    for (int64_t index = 0; index < n_kv; ++index) {
        emitted_keys.insert(gguf_get_key(ctx, index));
    }
    gguf_free(ctx);

    // Every kPrescanKnownKeys entry applies to the "x-vector" kind (Plan 2
    // has no other kind yet), filtered by `scope` rather than re-typed as
    // fresh string literals here -- the same discipline
    // tests/omnivoice_serialize_writer_agreement_test.cpp's own
    // expected_keys_for holds itself to.
    std::set<std::string> expected_keys;
    for (size_t index = 0; index < kPrescanKnownKeyCount; ++index) {
        const auto & spec = kPrescanKnownKeys[index];
        if (spec.scope == PrescanKeyScope::kCommon || spec.scope == PrescanKeyScope::kXVectorOnly) {
            expected_keys.insert(spec.key);
        }
    }

    SYNTH_TEST_CHECK(emitted_keys == expected_keys);
    SYNTH_TEST_CHECK(int64_t(emitted_keys.size()) == kPrescanKvCountXVector);
    return 0;
}

// =============================================================================
// Reviewer follow-up, IMPORTANT 1: the writer must never be able to emit an
// envelope its own reader refuses. Before kMaxLanguageTagLength existed and
// was tied to profile.cpp's own kPrescanMaxStringLength, a profile with a 1
// MiB + 8 byte language_tag serialized fine (1,049,376 bytes) and then
// failed to load with a bare INVALID_ARG -- measured on exactly this shape
// below.
// =============================================================================

// --- The reviewer's own measured buffer, at the creation entry point:
// create_x_vector_profile now refuses a language_tag over the tied ceiling
// before the (expensive) encode chain ever runs.
int test_language_tag_too_long_is_rejected_at_creation() {
    Fixture fixture;
    SYNTH_TEST_CHECK(build_fixture(fixture));
    const std::string                     oversized_tag(size_t(synth::qwen3tts::kMaxLanguageTagLength) + 8, 'x');
    const char *                          code    = nullptr;
    const char *                          message = nullptr;
    std::shared_ptr<const XVectorProfile> profile;
    const synth_status_t                  status = synth::qwen3tts::create_x_vector_profile(
        fixture.hparams, fixture.weights, one_second_of_speech(), "", oversized_tag, 1, profile, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(profile == nullptr);
    SYNTH_TEST_CHECK(code != nullptr && std::strcmp(code, "voice_profile.language_tag_too_long") == 0);
    SYNTH_TEST_CHECK(message != nullptr);
    return 0;
}

// --- The reviewer's own measured buffer, at the writer entry point: a
// hand-built XVectorProfile (bypassing create_x_vector_profile entirely,
// the same way every round-trip test in this file does) with a 1 MiB + 8
// byte language_tag. Before this task's fix, serialize_x_vector_profile
// returned SYNTH_OK here (1,049,376 bytes) and load_profile_from_memory
// then refused the result with a bare INVALID_ARG -- "your writer emits
// envelopes your own reader refuses." The writer itself now refuses this
// input before producing any bytes at all.
int test_oversized_language_tag_is_rejected_by_the_writer() {
    constexpr uint32_t                    kProfileEncDim = 11;
    const std::shared_ptr<XVectorProfile> profile        = make_serializable_profile(
        kProfileEncDim, 0.5f, std::string(size_t(synth::qwen3tts::kMaxLanguageTagLength) + 8, 'x'));

    uint8_t compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    std::vector<uint8_t> bytes;
    const synth_status_t status = synth::qwen3tts::serialize_x_vector_profile(*profile, compatibility_id, bytes);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(bytes.empty());
    return 0;
}

// --- Boundary proof that the tie is correct, not merely stricter: a
// language_tag at EXACTLY kMaxLanguageTagLength (not one byte over) still
// serializes and loads back byte-for-byte, so the fix closes the gap
// without narrowing what a legitimate (if pathological) caller could
// already do.
int test_language_tag_at_the_max_length_round_trips() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    const std::string                     tag(size_t(synth::qwen3tts::kMaxLanguageTagLength), 'x');
    const std::shared_ptr<XVectorProfile> profile = make_serializable_profile(kProfileEncDim, 0.5f, tag);

    std::vector<uint8_t> bytes;
    SYNTH_TEST_CHECK(synth::qwen3tts::serialize_x_vector_profile(*profile, compatibility_id, bytes) == SYNTH_OK);

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        kProfileEncDim, bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_OK);
    const auto * reloaded = static_cast<const XVectorProfile *>(payload.get());
    SYNTH_TEST_CHECK(reloaded->language_tag == tag);
    return 0;
}

// =============================================================================
// Reviewer follow-up, IMPORTANT 2: a loaded XVectorProfile must satisfy the
// same invariants a CREATED one does, not just the same structure. Before
// this task's fix, all five buffers below loaded with SYNTH_OK.
// =============================================================================

// --- ref_rms == 0.0f: the exact silent-reference state
// encode_speaker_reference itself refuses at creation with
// "voice_profile.reference_silent". A hand-built envelope claiming it is
// the untrusted-bytes counterpart of that same rejection.
int test_zero_ref_rms_is_rejected_by_the_reader() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    const std::shared_ptr<XVectorProfile> profile = make_serializable_profile(kProfileEncDim, 0.0f, "en-US");
    std::vector<uint8_t>                  bytes;
    SYNTH_TEST_CHECK(synth::qwen3tts::serialize_x_vector_profile(*profile, compatibility_id, bytes) == SYNTH_OK);

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        kProfileEncDim, bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// --- ref_rms < 0: reference_rms() (a sqrt of a sum of squares) can never
// itself produce a negative result, so a stored negative value can only
// come from a corrupted or hand-forged envelope.
int test_negative_ref_rms_is_rejected_by_the_reader() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    const std::shared_ptr<XVectorProfile> profile = make_serializable_profile(kProfileEncDim, -1.0f, "en-US");
    std::vector<uint8_t>                  bytes;
    SYNTH_TEST_CHECK(synth::qwen3tts::serialize_x_vector_profile(*profile, compatibility_id, bytes) == SYNTH_OK);

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        kProfileEncDim, bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// --- ref_rms == +inf: another state reference_rms() can never itself
// produce for real PCM. `ref_rms > 0.0f` alone would NOT catch this
// (+inf > 0.0f is true) -- the reader's own check pairs it with
// std::isfinite specifically because of this case.
int test_infinite_ref_rms_is_rejected_by_the_reader() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    const std::shared_ptr<XVectorProfile> profile =
        make_serializable_profile(kProfileEncDim, std::numeric_limits<float>::infinity(), "en-US");
    std::vector<uint8_t> bytes;
    SYNTH_TEST_CHECK(synth::qwen3tts::serialize_x_vector_profile(*profile, compatibility_id, bytes) == SYNTH_OK);

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        kProfileEncDim, bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// --- ref_rms == NaN: `ref_rms > 0.0f` alone WOULD catch this (any NaN
// comparison is false), but is checked here anyway so a future change to
// that check's shape cannot silently drop NaN coverage unnoticed.
int test_nan_ref_rms_is_rejected_by_the_reader() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    const std::shared_ptr<XVectorProfile> profile =
        make_serializable_profile(kProfileEncDim, std::numeric_limits<float>::quiet_NaN(), "en-US");
    std::vector<uint8_t> bytes;
    SYNTH_TEST_CHECK(synth::qwen3tts::serialize_x_vector_profile(*profile, compatibility_id, bytes) == SYNTH_OK);

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        kProfileEncDim, bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// --- An all-zero x-vector: finite in every element (so the isfinite check
// alone would accept it), but this network's own bias terms make an
// exactly-all-zero output indistinguishable from a corrupted or hand-forged
// payload rather than a real embedding.
int test_all_zero_x_vector_is_rejected_by_the_reader() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    auto profile  = std::make_shared<XVectorProfile>();
    profile->mode = CloneMode::XVector;
    profile->x_vector.assign(kProfileEncDim, 0.0f);
    profile->ref_rms      = 0.5f;
    profile->language_tag = "en-US";

    std::vector<uint8_t> bytes;
    SYNTH_TEST_CHECK(synth::qwen3tts::serialize_x_vector_profile(*profile, compatibility_id, bytes) == SYNTH_OK);

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        kProfileEncDim, bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// =============================================================================
// Plan 3 Task 8, the claim that has to be ASSERTED rather than assumed: adding
// a second clone mode did not disturb the first one.
//
// The Profile Schema does not change identity or version for this -- it stays
// "qwen3-tts-voice-clone" at version 1, and the in-envelope
// `synthesize.voice_profile.kind` gains a second value instead. Discriminating
// on `kind` rather than on `schema_version` is the entire point of that field
// (profile.h's own header comment on the envelope section), and what it BUYS
// is exactly this: an x-vector Profile written before ICL existed still loads,
// with no package re-cut. So the claim is checked here, in the commit that
// lands the second mode, rather than left to Task 9's own round trip.
//
// Distinct from test_round_trip_of_a_well_formed_profile above, which starts
// from a HAND-BUILT XVectorProfile: this one starts from the real
// create_x_vector_profile, so the whole Plan 2 path -- prepare, serialize,
// load -- is what is asserted still intact.
// =============================================================================

int test_a_plan_2_x_vector_profile_still_round_trips() {
    Fixture fixture;
    SYNTH_TEST_CHECK(build_fixture(fixture));
    const char *                          code    = nullptr;
    const char *                          message = nullptr;
    std::shared_ptr<const XVectorProfile> prepared;
    SYNTH_TEST_CHECK(synth::qwen3tts::create_x_vector_profile(fixture.hparams, fixture.weights, one_second_of_speech(),
                                                              /*transcript=*/"", /*language_tag=*/"english", 1,
                                                              prepared, code, message) == SYNTH_OK);
    SYNTH_TEST_CHECK(prepared != nullptr);
    SYNTH_TEST_CHECK(prepared->mode == CloneMode::XVector);

    uint8_t compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    std::vector<uint8_t> bytes;
    SYNTH_TEST_CHECK(synth::qwen3tts::serialize_x_vector_profile(*prepared, compatibility_id, bytes) == SYNTH_OK);

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                load_code    = nullptr;
    const char *                load_message = nullptr;
    SYNTH_TEST_CHECK(synth::qwen3tts::load_profile_from_memory(fixture.hparams.speaker_encoder.enc_dim, bytes.data(),
                                                               bytes.size(), compatibility_id, family_tag, payload,
                                                               load_code, load_message) == SYNTH_OK);
    SYNTH_TEST_CHECK(family_tag == synth::ProfileFamilyTag::Qwen3TtsClone);
    SYNTH_TEST_CHECK(payload != nullptr);

    const auto * reloaded = static_cast<const XVectorProfile *>(payload.get());
    // Still the FIRST mode, not the new one: a Plan 2 Profile read under a
    // build that knows about `icl` must not come back meaning `icl`.
    SYNTH_TEST_CHECK(reloaded->mode == CloneMode::XVector);
    SYNTH_TEST_CHECK(reloaded->x_vector == prepared->x_vector);
    SYNTH_TEST_CHECK(reloaded->ref_rms == prepared->ref_rms);
    SYNTH_TEST_CHECK(reloaded->language_tag == "english");
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

    SYNTH_TEST_CHECK(test_an_icl_profile_reports_the_icl_mode() == 0);
    SYNTH_TEST_CHECK(test_an_icl_payload_reads_back_through_an_x_vector_pointer() == 0);
    SYNTH_TEST_CHECK(test_an_icl_profile_carries_all_three_of_d5s_rows() == 0);
    SYNTH_TEST_CHECK(test_the_icl_code_grid_is_the_encoders_own_grid() == 0);
    SYNTH_TEST_CHECK(test_the_icl_frame_count_follows_the_reference_length() == 0);
    SYNTH_TEST_CHECK(test_icl_refuses_a_blank_transcript() == 0);
    SYNTH_TEST_CHECK(test_icl_refuses_a_transcript_without_its_token_ids() == 0);
    SYNTH_TEST_CHECK(test_icl_refuses_an_oversized_language_tag() == 0);
    SYNTH_TEST_CHECK(test_icl_inherits_the_shared_refusals() == 0);

    SYNTH_TEST_CHECK(test_round_trip_of_a_well_formed_profile() == 0);
    SYNTH_TEST_CHECK(test_serialize_is_deterministic() == 0);
    SYNTH_TEST_CHECK(test_wrong_model_family_is_unsupported_voice() == 0);
    SYNTH_TEST_CHECK(test_wrong_schema_is_unsupported_voice() == 0);
    SYNTH_TEST_CHECK(test_wrong_schema_version_is_unsupported_voice() == 0);
    SYNTH_TEST_CHECK(test_unrecognized_kind_is_invalid_arg() == 0);
    SYNTH_TEST_CHECK(test_wrong_compatibility_id_is_unsupported_voice() == 0);
    SYNTH_TEST_CHECK(test_tampered_content_sha256_is_invalid_arg() == 0);
    SYNTH_TEST_CHECK(test_truncated_buffer_is_invalid_arg() == 0);
    SYNTH_TEST_CHECK(test_non_default_alignment_is_invalid_arg() == 0);
    SYNTH_TEST_CHECK(test_tensor_count_zero_is_invalid_arg() == 0);
    SYNTH_TEST_CHECK(test_tensor_count_two_is_invalid_arg() == 0);
    SYNTH_TEST_CHECK(test_x_vector_length_mismatch_is_invalid_arg() == 0);
    SYNTH_TEST_CHECK(test_key_outside_whitelist_is_invalid_arg() == 0);
    SYNTH_TEST_CHECK(test_whitelisted_key_wrong_type_is_invalid_arg() == 0);
    SYNTH_TEST_CHECK(test_out_of_range_type_tag_is_invalid_arg() == 0);
    SYNTH_TEST_CHECK(test_wrong_n_kv_count_is_invalid_arg() == 0);
    SYNTH_TEST_CHECK(test_writer_emits_exactly_the_whitelisted_keys() == 0);

    SYNTH_TEST_CHECK(test_language_tag_too_long_is_rejected_at_creation() == 0);
    SYNTH_TEST_CHECK(test_oversized_language_tag_is_rejected_by_the_writer() == 0);
    SYNTH_TEST_CHECK(test_language_tag_at_the_max_length_round_trips() == 0);
    SYNTH_TEST_CHECK(test_zero_ref_rms_is_rejected_by_the_reader() == 0);
    SYNTH_TEST_CHECK(test_negative_ref_rms_is_rejected_by_the_reader() == 0);
    SYNTH_TEST_CHECK(test_infinite_ref_rms_is_rejected_by_the_reader() == 0);
    SYNTH_TEST_CHECK(test_nan_ref_rms_is_rejected_by_the_reader() == 0);
    SYNTH_TEST_CHECK(test_all_zero_x_vector_is_rejected_by_the_reader() == 0);

    SYNTH_TEST_CHECK(test_a_plan_2_x_vector_profile_still_round_trips() == 0);

    std::printf("qwen3-tts-profile: all checks passed\n");
    return 0;
}
