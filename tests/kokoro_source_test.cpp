// The harmonic source seam against reference values from the pinned upstream
// module.
//
// The values below were produced by driving upstream's SourceModuleHnNSF and
// TorchSTFT with the same injected random draws this seam accepts, at reduced
// dimensions. They pin the parts most easily got wrong: PyTorch's
// align_corners=false resampling, the phase accumulation that runs on the
// coarse grid and is rescaled, and torch.stft's reflect-padded centred framing.

#include "arch/kokoro/source.h"
#include "arch/kokoro/weights.h"
#include "test-assert.h"

#include <cmath>
#include <vector>

namespace {

// Reduced stand-ins for the real 300, 9 harmonics, n_fft 20 and hop 5.
constexpr uint32_t kScale     = 4;
constexpr uint32_t kHarmonics = 3;
constexpr uint32_t kNfft      = 8;
constexpr uint32_t kHop       = 2;
constexpr uint32_t kRate      = 24000;

// upsampled=20 frames=11 bins=5
const float kMergeWeight[] = { 0.662775099f, -0.780040801f, 1.1835531f };
const float kMergeBias     = -0.268448353f;
const float kRandIni[]     = { 0.0f, 0.206319392f, 0.445090175f };
const float kNoise[] = { -0.320476919f,  -1.58266461f, -0.324566692f, 1.92636728f,   -0.330011725f,  0.198444054f,
                         0.782072663f,   1.03910768f,  -0.887490094f, -0.209340423f, -0.215341449f,  -1.81572962f,
                         -0.345241964f,  -2.0614779f,  0.674100697f,  -1.32334578f,  0.503563821f,   -0.0835282356f,
                         -0.0234779324f, 0.174375355f, 2.2983427f,    0.95710206f,   -0.661867559f,  -0.828451931f,
                         1.40585291f,    -1.40125096f, 1.29733586f,   1.64093328f,   -1.05668414f,   -0.261587203f,
                         -0.25013262f,   0.50112164f,  -1.06539893f,  -0.178192616f, -0.259498864f,  -0.0144881476f,
                         -0.383890986f,  -2.96616983f, -1.06055498f,  -0.308996379f, -0.0916771665f, 1.62431645f,
                         0.00156733266f, -0.43754071f, -2.28581882f,  -0.367664188f, -0.88218081f,   0.546012104f,
                         0.148511663f,   -2.25778222f, 0.404298425f,  0.572199047f,  1.37977231f,    1.28771985f,
                         0.868438303f,   -1.38219845f, -0.963228643f, -0.398458838f, -0.173167616f,  0.756880879f };
const float kF0[]    = { 0.0f, 120.0f, 300.0f, 5.0f, 440.0f };
const float kExpectedSource[] = { -0.242267683f, -0.206469044f, -0.303346992f, -0.32668063f,  -0.236713976f,
                                  -0.239232972f, -0.215493858f, -0.206484035f, -0.184139013f, -0.181402221f,
                                  -0.188822731f, -0.183303744f, -0.237049654f, -0.205826193f, -0.333904833f,
                                  -0.228012472f, -0.19632858f,  -0.223289132f, -0.237193912f, -0.236477271f };
const float kExpectedHar[]    = { 0.993761897f, 0.423840106f,    0.0610793084f,   0.0606952757f,  0.0974674523f,
                                  3.14159274f,  9.53668149e-08f, 4.87928276e-07f, 6.6595436e-07f, 3.14159274f,
                                  1.06318104f,  0.58358258f,     0.11680828f,     0.0834604129f,  0.0224946737f,
                                  3.14159274f,  -0.125711605f,   2.14918447f,     -1.23227406f,   3.14159274f,
                                  1.03964746f,  0.544171929f,    0.0780206993f,   0.0627102107f,  0.0473786592f,
                                  3.14159274f,  0.178658172f,    -1.27549195f,    2.99997425f,    0.0f,
                                  0.880770624f, 0.436194271f,    0.00838249922f,  0.00857043732f, 0.0289298892f,
                                  3.14159274f,  0.140531182f,    -2.21995568f,    1.67675769f,    0.0f,
                                  0.779258013f, 0.37605831f,     0.022347508f,    0.00981386844f, 0.00666335225f,
                                  3.14159274f,  0.0912439898f,   -0.632889032f,   2.25671434f,    0.0f,
                                  0.771094441f, 0.367279768f,    0.0218393821f,   0.0278186295f,  0.0277396441f,
                                  3.14159274f,  -0.0750400424f,  0.078809537f,    1.14859152f,    3.14159274f,
                                  0.890514016f, 0.439040184f,    0.0272927117f,   0.0701207444f,  0.106312901f,
                                  3.14159274f,  -0.20869343f,    0.471550673f,    0.881706357f,   3.14159274f,
                                  0.980442524f, 0.553652763f,    0.117943421f,    0.120288104f,   0.120745391f,
                                  3.14159274f,  0.00511034578f,  3.03045106f,     -0.320465207f,  3.14159274f,
                                  0.931861758f, 0.425630063f,    0.0896267071f,   0.0573142059f,  0.0318941474f,
                                  3.14159274f,  0.113090687f,    -0.0952088013f,  -2.12687826f,   3.14159274f,
                                  0.912481904f, 0.468784869f,    0.0236698203f,   0.0144711258f,  0.00457155704f,
                                  3.14159274f,  -0.0595577322f,  2.61244392f,     1.1100471f,     0.0f,
                                  0.912481904f, 0.468784869f,    0.0236698203f,   0.0144710522f,  0.00457155704f,
                                  3.14159274f,  0.0595579073f,   -2.61244392f,    -1.1100446f,    0.0f };

synth::kokoro::HParams hparams() {
    synth::kokoro::HParams h;
    h.source.sampling_rate        = kRate;
    h.source.harmonic_num         = kHarmonics - 1;
    h.source.upsample_scale       = kScale;
    h.source.sine_amp             = 0.1f;
    h.source.noise_std            = 0.003f;
    h.source.voiced_threshold     = 10.0f;
    h.istftnet.gen_istft_n_fft    = kNfft;
    h.istftnet.gen_istft_hop_size = kHop;
    return h;
}

template <size_t N> std::vector<float> as_vector(const float (&values)[N]) {
    return std::vector<float>(values, values + N);
}

}  // namespace

int main() {
    const synth::kokoro::HParams h = hparams();

    synth::kokoro::SourceRandomInputs random;
    random.rand_ini = as_vector(kRandIni);
    random.noise    = as_vector(kNoise);

    synth::kokoro::SourceResult result;
    SYNTH_TEST_CHECK(synth::kokoro::build_harmonic_source(h, as_vector(kF0), as_vector(kMergeWeight), kMergeBias,
                                                          random, result) == SYNTH_OK);

    SYNTH_TEST_CHECK(synth::kokoro::source_harmonic_count(h) == kHarmonics);
    SYNTH_TEST_CHECK(result.har_source.size() == sizeof(kExpectedSource) / sizeof(float));
    SYNTH_TEST_CHECK(result.bins == kNfft / 2 + 1);
    SYNTH_TEST_CHECK(result.frames == result.har_source.size() / kHop + 1);
    SYNTH_TEST_CHECK(result.har.size() == size_t(result.frames) * 2 * result.bins);

    float source_diff = 0.0f;
    for (size_t index = 0; index < result.har_source.size(); ++index) {
        source_diff = std::max(source_diff, std::fabs(result.har_source[index] - kExpectedSource[index]));
    }
    SYNTH_TEST_CHECK(source_diff < 1e-5f);

    // Magnitudes compare directly; phases are compared on the circle so the
    // wrap at +/- pi is not read as a mismatch.
    float magnitude_diff = 0.0f;
    float phase_diff     = 0.0f;
    for (uint64_t frame = 0; frame < result.frames; ++frame) {
        for (uint64_t bin = 0; bin < result.bins; ++bin) {
            const size_t base      = size_t(frame) * 2 * result.bins;
            const float  magnitude = result.har[base + bin];
            const float  phase     = result.har[base + result.bins + bin];
            magnitude_diff         = std::max(magnitude_diff, std::fabs(magnitude - kExpectedHar[base + bin]));
            const float delta      = std::atan2(std::sin(phase - kExpectedHar[base + result.bins + bin]),
                                                std::cos(phase - kExpectedHar[base + result.bins + bin]));
            phase_diff             = std::max(phase_diff, std::fabs(delta));
        }
    }
    SYNTH_TEST_CHECK(magnitude_diff < 1e-5f);
    SYNTH_TEST_CHECK(phase_diff < 1e-4f);

    // Contract failures.
    synth::kokoro::SourceResult rejected;
    SYNTH_TEST_CHECK(synth::kokoro::build_harmonic_source(h, {}, as_vector(kMergeWeight), kMergeBias, random,
                                                          rejected) == SYNTH_ERR_INVALID_ARG);

    synth::kokoro::SourceRandomInputs short_noise = random;
    short_noise.noise.pop_back();
    SYNTH_TEST_CHECK(synth::kokoro::build_harmonic_source(h, as_vector(kF0), as_vector(kMergeWeight), kMergeBias,
                                                          short_noise, rejected) == SYNTH_ERR_INVALID_ARG);

    synth::kokoro::SourceRandomInputs short_phase = random;
    short_phase.rand_ini.pop_back();
    SYNTH_TEST_CHECK(synth::kokoro::build_harmonic_source(h, as_vector(kF0), as_vector(kMergeWeight), kMergeBias,
                                                          short_phase, rejected) == SYNTH_ERR_INVALID_ARG);

    std::vector<float> short_weight = as_vector(kMergeWeight);
    short_weight.pop_back();
    SYNTH_TEST_CHECK(synth::kokoro::build_harmonic_source(h, as_vector(kF0), short_weight, kMergeBias, random,
                                                          rejected) == SYNTH_ERR_INVALID_ARG);
    return 0;
}
