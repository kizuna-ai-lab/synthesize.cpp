// Kokoro through the public C interface.
//
// The staged validators compare tensors against the oracle; this covers what
// only the public seam can show — that the family is reachable through
// synth_model_load at all, that its declared capabilities are the ones the
// package contract states, and that the manifest's public-request relations
// hold: a seed determines the audio, different seeds differ, and the speaking
// rate orders the output length.

#include "synthesize.h"
#include "test-assert.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

// The upstream example, resolved to token IDs by the manifest.
const int32_t kTokens[] = { 0,   81,  83,  16,  61,  53,  156, 25,  16, 83,  44,  156, 138, 64, 16, 81,
                            83,  16,  58,  156, 76,  123, 62,  16,  65, 138, 68,  16,  81,  83, 16, 53,
                            156, 138, 54,  83,  123, 16,  138, 64,  16, 62,  156, 86,  54,  83, 64, 157,
                            102, 147, 83,  56,  3,   16,  62,  156, 63, 56,  46,  16,  62,  83, 16, 70,
                            16,  46,  156, 86,  46,  16,  133, 156, 72, 56,  42,  54,  4,   0 };

constexpr const char * kVoice          = "af_heart";
constexpr uint64_t     kVoiceSize      = 8;
// One predicted duration step is exactly this many output samples.
constexpr uint64_t     kSamplesPerStep = 600;

synth_request_t make_request(uint64_t seed, float speaking_rate) {
    synth_request_t request;
    synth_request_init(&request, sizeof(request));
    request.input_kind    = SYNTH_INPUT_TOKEN_IDS;
    request.input_data    = kTokens;
    request.input_count   = sizeof(kTokens) / sizeof(kTokens[0]);
    request.voice_id      = kVoice;
    request.voice_id_size = kVoiceSize;
    request.seed          = seed;
    request.speaking_rate = speaking_rate;
    return request;
}

bool synthesize(synth_context_t *    context,
                uint64_t             seed,
                float                speaking_rate,
                std::vector<float> & pcm,
                uint32_t &           sample_rate) {
    synth_request_t        request = make_request(seed, speaking_rate);
    synth_audio_buffer_t * audio   = nullptr;
    synth_result_t         result;
    synth_result_init(&result, sizeof(result));
    if (synth_synthesize_to_buffer(context, &request, &audio, &result) != SYNTH_OK || audio == nullptr) {
        return false;
    }
    pcm.assign(audio->samples, audio->samples + audio->frame_count);
    sample_rate = audio->sample_rate;
    synth_audio_buffer_free(audio);
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    SYNTH_TEST_CHECK(argc == 2);

    synth_model_t * model = nullptr;
    SYNTH_TEST_CHECK(synth_model_load(argv[1], nullptr, &model) == SYNTH_OK);
    SYNTH_TEST_CHECK(model != nullptr);

    // The declared capabilities are the package contract, not a restatement of
    // whatever the loader happened to read.
    synth_model_capabilities_t capabilities;
    synth_model_capabilities_init(&capabilities, sizeof(capabilities));
    SYNTH_TEST_CHECK(synth_model_get_capabilities(model, &capabilities) == SYNTH_OK);
    SYNTH_TEST_CHECK(capabilities.input_flags == (SYNTH_INPUT_SUPPORT_PHONEMES_UTF8 | SYNTH_INPUT_SUPPORT_TOKEN_IDS));
    SYNTH_TEST_CHECK(capabilities.output_sample_rate == 24000 && capabilities.output_channel_count == 1);
    SYNTH_TEST_CHECK(capabilities.min_speaking_rate == 0.8f && capabilities.max_speaking_rate == 1.25f);
    SYNTH_TEST_CHECK(capabilities.max_input_tokens == 512 && capabilities.max_output_frames == 1440000);

    // All 54 preset voices are exposed, and the catalog has no default: a
    // Kokoro request must name a voice.
    uint64_t voice_count = 0;
    SYNTH_TEST_CHECK(synth_model_get_preset_voice_count(model, &voice_count) == SYNTH_OK);
    SYNTH_TEST_CHECK(voice_count == 54);
    synth_preset_voice_t voice;
    synth_preset_voice_init(&voice, sizeof(voice));
    SYNTH_TEST_CHECK(synth_model_get_preset_voice(model, 3, &voice) == SYNTH_OK);
    SYNTH_TEST_CHECK(std::strcmp(voice.id, kVoice) == 0);

    synth_context_t * context = nullptr;
    SYNTH_TEST_CHECK(synth_context_create(model, &context) == SYNTH_OK);

    std::vector<float> first;
    uint32_t           sample_rate = 0;
    SYNTH_TEST_CHECK(synthesize(context, 1234, 1.0f, first, sample_rate));
    SYNTH_TEST_CHECK(sample_rate == 24000);
    SYNTH_TEST_CHECK(!first.empty() && first.size() % kSamplesPerStep == 0);
    for (float sample : first) {
        SYNTH_TEST_CHECK(std::isfinite(sample));
    }

    // A seed determines the whole draw, so the same request repeats exactly.
    std::vector<float> repeated;
    SYNTH_TEST_CHECK(synthesize(context, 1234, 1.0f, repeated, sample_rate));
    SYNTH_TEST_CHECK(repeated == first);

    // A different seed reaches the harmonic source and changes the audio, but
    // not its length: the randomness enters after the durations are resolved.
    std::vector<float> other_seed;
    SYNTH_TEST_CHECK(synthesize(context, 4242, 1.0f, other_seed, sample_rate));
    SYNTH_TEST_CHECK(other_seed.size() == first.size());
    SYNTH_TEST_CHECK(other_seed != first);

    // A faster rate shortens the output and a slower one lengthens it.
    std::vector<float> fast;
    std::vector<float> slow;
    SYNTH_TEST_CHECK(synthesize(context, 1234, 1.25f, fast, sample_rate));
    SYNTH_TEST_CHECK(synthesize(context, 1234, 0.8f, slow, sample_rate));
    SYNTH_TEST_CHECK(fast.size() < first.size() && first.size() < slow.size());
    SYNTH_TEST_CHECK(fast.size() % kSamplesPerStep == 0 && slow.size() % kSamplesPerStep == 0);

    // A rate outside the validated range is refused rather than clamped.
    synth_request_t        out_of_range = make_request(1234, 2.0f);
    synth_audio_buffer_t * audio        = nullptr;
    synth_result_t         result;
    synth_result_init(&result, sizeof(result));
    SYNTH_TEST_CHECK(synth_synthesize_to_buffer(context, &out_of_range, &audio, &result) != SYNTH_OK);
    SYNTH_TEST_CHECK(audio == nullptr);

    // The catalog has no default voice, so a request without one is refused.
    synth_request_t no_voice = make_request(1234, 1.0f);
    no_voice.voice_id        = nullptr;
    no_voice.voice_id_size   = 0;
    SYNTH_TEST_CHECK(synth_synthesize_to_buffer(context, &no_voice, &audio, &result) == SYNTH_ERR_UNSUPPORTED_VOICE);
    SYNTH_TEST_CHECK(audio == nullptr);

    synth_context_free(context);
    synth_model_free(model);
    return 0;
}
