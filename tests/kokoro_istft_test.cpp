// The inverse short-time transform is compared against torch.istft.
//
// The reference is generated from the raw form the generator emits, so the
// exponential and the sine that turn the graph's output into a magnitude and a
// phase are part of what is checked, not applied beforehand.

#include "arch/kokoro/decoder-host.h"
#include "arch/kokoro/weights.h"
#include "test-assert.h"

#include <cmath>
#include <vector>

namespace {

// n_fft 8, hop 2, 9 frames, so 16 output samples.
constexpr uint32_t kNfft   = 8;
constexpr uint32_t kHop    = 2;
constexpr uint64_t kFrames = 9;

constexpr float kSpectrum[] = {
    -0.7019165754318237f,  0.8496788740158081f,    0.607083797454834f,    -0.11508476734161377f, 0.9800618886947632f,
    0.12778544425964355f,  -0.033782124519348145f, -0.45408570766448975f, 0.1767435073852539f,   -0.9821900129318237f,
    -0.0268937349319458f,  -0.46132445335388184f,  0.5305935144424438f,   -0.47565221786499023f, -0.6111619472503662f,
    0.22498619556427002f,  0.3503941297531128f,    0.5387789011001587f,   0.6004284620285034f,   -0.31678903102874756f,
    0.9713230133056641f,   0.5649137496948242f,    0.7418065071105957f,   0.2601414918899536f,   -0.1488969326019287f,
    -0.5471979379653931f,  -0.4950772523880005f,   0.6034935712814331f,   0.08367359638214111f,  -0.5518691539764404f,
    -0.6631971597671509f,  -0.45622479915618896f,  0.4461100101470947f,   0.7550157308578491f,   -0.12357330322265625f,
    0.170548677444458f,    0.05431365966796875f,   -0.74320387840271f,    -0.5986857414245605f,  0.8516738414764404f,
    0.1678868532180786f,   -0.07353639602661133f,  0.6728682518005371f,   -0.6747560501098633f,  -0.40245091915130615f,
    -0.6298302412033081f,  0.6562089920043945f,    0.375515341758728f,    -0.18416500091552734f, 0.944022536277771f,
    0.3871983289718628f,   0.5342617034912109f,    0.16404736042022705f,  -0.5123910903930664f,  0.6947115659713745f,
    -0.01598799228668213f, -0.008729934692382812f, -0.179482102394104f,   -0.2980821132659912f,  0.5162678956985474f,
    0.16616976261138916f,  0.004851937294006348f,  0.6789333820343018f,   -0.32365214824676514f, -0.043018341064453125f,
    0.43417882919311523f,  0.6583589315414429f,    0.28979504108428955f,  0.7751045227050781f,   -0.08532202243804932f,
    -0.7431244850158691f,  -0.7949943542480469f,   -0.8809770345687866f,  0.7694782018661499f,   -0.6129122972488403f,
    -0.08432137966156006f, -0.9035348892211914f,   0.9404277801513672f,   0.89328932762146f,     -0.9641091823577881f,
    0.14424526691436768f,  0.5199943780899048f,    -0.13702523708343506f, -0.8474953174591064f,  -0.5522581338882446f,
    -0.2867262363433838f,  -0.8347512483596802f,   -0.9681240320205688f,  -0.39254534244537354f, 0.07183802127838135f
};

constexpr float kExpectedAudio[] = { -0.1448516696691513f,  -0.025538688525557518f, 0.17494265735149384f,
                                     0.28591296076774597f,  -0.15135395526885986f,  -0.0464189350605011f,
                                     -0.34755584597587585f, 0.26592400670051575f,   0.28406867384910583f,
                                     -0.28042498230934143f, -0.039361562579870224f, -0.18007618188858032f,
                                     0.47102880477905273f,  0.12626740336418152f,   -0.15285880863666534f,
                                     0.2414003312587738f };

synth::kokoro::HParams make_hparams() {
    synth::kokoro::HParams hparams;
    hparams.istftnet.gen_istft_n_fft    = kNfft;
    hparams.istftnet.gen_istft_hop_size = kHop;
    return hparams;
}

}  // namespace

int main() {
    const synth::kokoro::HParams hparams = make_hparams();

    SYNTH_TEST_CHECK(synth::kokoro::inverse_stft_length(hparams, kFrames) == 16);
    SYNTH_TEST_CHECK(synth::kokoro::inverse_stft_length(hparams, 1) == 0);
    SYNTH_TEST_CHECK(synth::kokoro::inverse_stft_length(hparams, 0) == 0);

    const std::vector<float> spectrum(std::begin(kSpectrum), std::end(kSpectrum));
    SYNTH_TEST_CHECK(spectrum.size() == size_t(kFrames) * (kNfft + 2));

    std::vector<float> audio;
    SYNTH_TEST_CHECK(synth::kokoro::inverse_stft(hparams, spectrum, kFrames, audio) == SYNTH_OK);
    SYNTH_TEST_CHECK(audio.size() == std::size(kExpectedAudio));

    float max_diff = 0.0f;
    for (size_t index = 0; index < audio.size(); ++index) {
        max_diff = std::max(max_diff, std::fabs(audio[index] - kExpectedAudio[index]));
    }
    SYNTH_TEST_CHECK(max_diff < 1e-5f);

    // A spectrum whose size disagrees with the frame count is a wiring defect,
    // not something to reinterpret.
    std::vector<float> truncated(spectrum.begin(), spectrum.end() - 1);
    SYNTH_TEST_CHECK(synth::kokoro::inverse_stft(hparams, truncated, kFrames, audio) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(audio.empty());

    SYNTH_TEST_CHECK(synth::kokoro::inverse_stft(hparams, spectrum, 0, audio) == SYNTH_ERR_INVALID_ARG);

    // The transform needs an even window, since the onesided spectrum's last
    // bin is only the Nyquist bin when the window length is even.
    synth::kokoro::HParams odd   = hparams;
    odd.istftnet.gen_istft_n_fft = 7;
    SYNTH_TEST_CHECK(synth::kokoro::inverse_stft(odd, spectrum, kFrames, audio) == SYNTH_ERR_INVALID_ARG);

    synth::kokoro::HParams no_hop      = hparams;
    no_hop.istftnet.gen_istft_hop_size = 0;
    SYNTH_TEST_CHECK(synth::kokoro::inverse_stft(no_hop, spectrum, kFrames, audio) == SYNTH_ERR_INVALID_ARG);
    return 0;
}
