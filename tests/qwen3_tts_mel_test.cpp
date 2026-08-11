// The mel front end has no GGUF and no Model -- the governing rule is stated
// in-tree at tests/qwen3_tts_voice_required_test.cpp:25-30. Every case below
// is a rule with an input that fails it.
//
// kExpectedFramesForOneSecond and kExpectedFramesForOneSecondAtHop512 are
// taken from Task 1's oracle dump (conventions.json's `frames_for_one_second`
// and `frames_for_one_second_at_hop_512`), so the port's frame arithmetic is
// pinned to what upstream actually did rather than to a formula this task
// chose. Neither of the two formulas a reader might guess -- a naive centred
// STFT or a plainly uncentred one -- produces them; see conventions.json's
// `padding_note`.
#include "arch/qwen3-tts/mel.h"
#include "arch/qwen3-tts/weights.h"
#include "test-assert.h"

#include <cmath>
#include <vector>

namespace {

constexpr uint64_t kExpectedFramesForOneSecond         = 93;
constexpr uint64_t kExpectedFramesForOneSecondAtHop512 = 46;

synth::qwen3tts::SpeakerEncoderParams production_params() {
    synth::qwen3tts::SpeakerEncoderParams p;
    p.enc_dim     = 1024;
    p.sample_rate = 24000;
    p.mel_bins    = 128;
    p.n_fft       = 1024;
    p.hop_length  = 256;
    p.win_length  = 1024;
    p.fmin        = 0.0f;
    p.fmax        = 12000.0f;
    return p;
}

// Every one of the eight fields reaches the front end. Six of them had no
// consumer at all after Plan 1 -- read_speaker_encoder validated them and
// nothing else looked -- so this is the test that makes them live.
int test_the_frame_count_follows_hop_length() {
    const synth::qwen3tts::SpeakerEncoderParams p = production_params();
    std::vector<float>                          pcm(24000, 0.5f);
    synth::qwen3tts::MelSpectrogram             mel;
    SYNTH_TEST_CHECK(synth::qwen3tts::compute_log_mel(p, pcm, mel) == SYNTH_OK);
    SYNTH_TEST_CHECK(mel.bins == 128);
    SYNTH_TEST_CHECK(mel.frames == kExpectedFramesForOneSecond);  // from Task 1's conventions.json
    SYNTH_TEST_CHECK(mel.values.size() == size_t(mel.bins) * size_t(mel.frames));
    return 0;
}

// A different hop must produce a different frame count, or hop_length is
// being ignored and the production value is right by coincidence.
int test_a_different_hop_produces_a_different_frame_count() {
    synth::qwen3tts::SpeakerEncoderParams p = production_params();
    p.hop_length                            = 512;
    std::vector<float>              pcm(24000, 0.5f);
    synth::qwen3tts::MelSpectrogram mel;
    SYNTH_TEST_CHECK(synth::qwen3tts::compute_log_mel(p, pcm, mel) == SYNTH_OK);
    SYNTH_TEST_CHECK(mel.frames == kExpectedFramesForOneSecondAtHop512);
    return 0;
}

// fmin/fmax select which filters exist at all. Narrowing the band must move
// the output, or the filterbank is being built from the sample rate alone.
int test_the_band_limits_change_the_filterbank() {
    const synth::qwen3tts::SpeakerEncoderParams wide   = production_params();
    synth::qwen3tts::SpeakerEncoderParams       narrow = wide;
    narrow.fmin                                        = 300.0f;
    narrow.fmax                                        = 6000.0f;

    std::vector<float> pcm(24000);
    for (size_t i = 0; i < pcm.size(); ++i) {
        pcm[i] = std::sin(float(i) * 0.05f) * 0.4f;
    }
    synth::qwen3tts::MelSpectrogram a;
    synth::qwen3tts::MelSpectrogram b;
    SYNTH_TEST_CHECK(synth::qwen3tts::compute_log_mel(wide, pcm, a) == SYNTH_OK);
    SYNTH_TEST_CHECK(synth::qwen3tts::compute_log_mel(narrow, pcm, b) == SYNTH_OK);
    SYNTH_TEST_CHECK(a.values != b.values);
    return 0;
}

// win_length shorter than n_fft is a zero-padded window, not a shorter
// transform. Kokoro's transforms have no such concept, which is exactly why
// this one is new code.
int test_a_short_window_is_zero_padded_not_truncated() {
    synth::qwen3tts::SpeakerEncoderParams p = production_params();
    p.win_length                            = 512;
    std::vector<float>              pcm(24000, 0.25f);
    synth::qwen3tts::MelSpectrogram mel;
    SYNTH_TEST_CHECK(synth::qwen3tts::compute_log_mel(p, pcm, mel) == SYNTH_OK);
    SYNTH_TEST_CHECK(mel.frames == kExpectedFramesForOneSecond);  // unchanged: win_length is not the hop
    return 0;
}

// Digital silence must not produce -inf or NaN: the log floor is what stops
// it, and a floor nobody tested is a floor nobody has.
int test_digital_silence_stays_finite() {
    const synth::qwen3tts::SpeakerEncoderParams p = production_params();
    std::vector<float>                          pcm(24000, 0.0f);
    synth::qwen3tts::MelSpectrogram             mel;
    SYNTH_TEST_CHECK(synth::qwen3tts::compute_log_mel(p, pcm, mel) == SYNTH_OK);
    for (float value : mel.values) {
        SYNTH_TEST_CHECK(std::isfinite(value));
    }
    return 0;
}

int test_a_non_finite_sample_is_refused() {
    const synth::qwen3tts::SpeakerEncoderParams p = production_params();
    std::vector<float>                          pcm(24000, 0.1f);
    pcm[7] = std::nanf("");
    synth::qwen3tts::MelSpectrogram mel;
    SYNTH_TEST_CHECK(synth::qwen3tts::compute_log_mel(p, pcm, mel) == SYNTH_ERR_INVALID_ARG);
    return 0;
}

int test_a_clip_shorter_than_one_frame_is_refused() {
    const synth::qwen3tts::SpeakerEncoderParams p = production_params();
    std::vector<float>                          pcm(16, 0.1f);
    synth::qwen3tts::MelSpectrogram             mel;
    SYNTH_TEST_CHECK(synth::qwen3tts::compute_log_mel(p, pcm, mel) == SYNTH_ERR_INVALID_ARG);
    return 0;
}

// The radix-2 transform is the whole reason n_fft must be a power of two.
// The load-time rule below is what keeps this unreachable in practice; this
// is the belt to that braces.
int test_a_non_power_of_two_n_fft_is_refused() {
    synth::qwen3tts::SpeakerEncoderParams p = production_params();
    p.n_fft                                 = 1000;
    std::vector<float>              pcm(24000, 0.1f);
    synth::qwen3tts::MelSpectrogram mel;
    SYNTH_TEST_CHECK(synth::qwen3tts::compute_log_mel(p, pcm, mel) == SYNTH_ERR_UNSUPPORTED_INPUT);
    return 0;
}

}  // namespace

int main() {
    SYNTH_TEST_CHECK(test_the_frame_count_follows_hop_length() == 0);
    SYNTH_TEST_CHECK(test_a_different_hop_produces_a_different_frame_count() == 0);
    SYNTH_TEST_CHECK(test_the_band_limits_change_the_filterbank() == 0);
    SYNTH_TEST_CHECK(test_a_short_window_is_zero_padded_not_truncated() == 0);
    SYNTH_TEST_CHECK(test_digital_silence_stays_finite() == 0);
    SYNTH_TEST_CHECK(test_a_non_finite_sample_is_refused() == 0);
    SYNTH_TEST_CHECK(test_a_clip_shorter_than_one_frame_is_refused() == 0);
    SYNTH_TEST_CHECK(test_a_non_power_of_two_n_fft_is_refused() == 0);
    return 0;
}
