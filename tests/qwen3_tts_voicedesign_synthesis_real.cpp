// A sentence of description in, audio out -- entirely through the public C
// seam (synth_voice_profile_create_from_description, then
// synth_synthesize_to_buffer), against the real 1.7B VoiceDesign package.
//
// THIS FILE EXISTS BECAUSE OF A DEFECT THIS TASK'S OWN COMMIT EXPOSED, NOT
// CREATED. Before Task 4, src/synthesize.cpp refused every Description Text
// Profile at the family_tag check with a clean SYNTH_ERR_UNSUPPORTED_VOICE,
// so nothing had ever driven Model::run_synthesis's decode loop for this
// variant -- tests/qwen3_tts_voicedesign_prefill_real.cpp only ever builds
// the PREFILL, and never calls the loop that follows it. Task 4 made the
// path reachable, and the first real run of it aborted the process:
//
//   ggml.c:2599: GGML_ASSERT(a->ne[d] == b->ne[d]) failed
//   #3 ggml_concat  #4 Model::run_synthesis  #5 synth_synthesize
//
// model.cpp's per-decode-step `t_hidden` was allocated at
// hparams.code_predictor.hidden_size instead of hparams.talker.hidden_size.
// It holds the talker's own per-step hidden state (talker-width) and is
// concatenated with a row of the talker's own codec_embedding (also
// talker-width) before code-predictor.cpp's build_code_predictor applies
// small_to_mtp_projection -- a real, non-identity Linear(talker.hidden_size,
// code_predictor.hidden_size) exactly on this package, since it is the first
// one this port has loaded whose talker (2048) and code predictor (1024)
// widths disagree (catalog.cpp's resolve_code_predictor, and
// docs/porting/families/qwen3-tts.md's own record of the same invariant).
// Every earlier package coincidentally has talker.hidden_size ==
// code_predictor.hidden_size, so the undersized allocation silently
// truncated `hidden_state` into `t_hidden` rather than erroring, and the
// concat two lines later never saw a shape it could not build -- until this
// package, where the truncation itself becomes a shape mismatch. Fixed by
// widening the allocation to hparams.talker.hidden_size; this file is the
// regression test for that fix reachable only against the real package,
// because the coincidence that hid it for two rungs is a property of real
// GGUF metadata, not of anything a synthetic HParams fixture chooses to set
// unless it deliberately sets the two widths apart -- which is exactly what
// tests/qwen3_tts_talker_test.cpp and friends never had a reason to do
// before this package existed. See this task's own report for why a
// unit-tier test could not have caught this on its own.
//
// Two cases, both required by design D3: an EMPTY instruct (upstream's own
// unconditioned path, `instruct_ids.append(None)`) and a NON-EMPTY one --
// the crash reproduced on both, so both are asserted here rather than
// trusting that fixing one fixes the other.
//
// Stage 3 Plan 2's Task 5 adds a third case: create -> serialize -> load ->
// synthesize, entirely through the public seam, against the SAME real
// package. tests/qwen3_tts_design_profile_test.cpp's own
// test_serialize_and_load_round_trip_through_the_public_seam pins the
// dispatch-routing half of this (a hand-built `synth_model`, no real weights
// needed) at the `unit` tier; this file is what proves the loaded-back
// Profile is not just structurally equal to the original but ACTUALLY
// SYNTHESIZES, which needs the real talker/codec weights a `unit` test
// cannot carry (docs/testing.md).

#include "synthesize.h"
#include "test-assert.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Synthesizes `text` against `profile` and returns the raw PCM plus its
// frame count. Mirrors tests/qwen3_tts_clone_real.cpp's own synthesize_pcm,
// duplicated rather than shared -- every integration driver in this
// directory is its own translation unit with no shared test-support library
// (the family's established convention).
bool synthesize_pcm(synth_model_t *         model,
                    synth_voice_profile_t * profile,
                    const char *            text,
                    std::vector<float> &    out_samples,
                    uint64_t &              out_frame_count,
                    uint32_t &              out_channels,
                    uint32_t &              out_sample_rate) {
    synth_context_t * context = nullptr;
    if (synth_context_create(model, &context) != SYNTH_OK) {
        return false;
    }

    synth_request_t request;
    synth_request_init(&request, sizeof(request));
    request.input_kind        = SYNTH_INPUT_TEXT_UTF8;
    request.input_data        = text;
    request.input_count       = std::strlen(text);
    request.voice_profile     = profile;
    request.language_tag      = "en";
    request.language_tag_size = std::strlen("en");
    request.seed              = 0;

    synth_audio_buffer_t * audio = nullptr;
    synth_result_t         result;
    synth_result_init(&result, sizeof(result));
    const synth_status_t status = synth_synthesize_to_buffer(context, &request, &audio, &result);
    std::fprintf(stderr, "synth_synthesize_to_buffer -> %d\n", int(status));
    const bool ok = status == SYNTH_OK && audio != nullptr;
    if (ok) {
        out_channels                = audio->channel_count;
        out_sample_rate             = audio->sample_rate;
        out_frame_count             = audio->frame_count;
        const uint64_t sample_count = audio->frame_count * audio->channel_count;
        out_samples.assign(audio->samples, audio->samples + sample_count);
    }
    if (audio != nullptr) {
        synth_audio_buffer_free(audio);
    }
    synth_context_free(context);
    return ok;
}

bool run_case(synth_model_t * model, const char * label, const std::string & instruct) {
    synth_voice_description_params_t params;
    synth_voice_description_params_init(&params, sizeof(params));
    params.description      = instruct.data();
    params.description_size = instruct.size();

    synth_voice_profile_t * profile = nullptr;
    const synth_status_t    created = synth_voice_profile_create_from_description(model, &params, &profile);
    std::fprintf(stderr, "%s: create_from_description -> %d\n", label, int(created));
    if (created != SYNTH_OK || profile == nullptr) {
        return false;
    }

    std::vector<float> pcm;
    uint64_t           frame_count = 0;
    uint32_t           channels    = 0;
    uint32_t           sample_rate = 0;
    const bool         synthesized =
        synthesize_pcm(model, profile, "Qwen3-TTS is awesome!", pcm, frame_count, channels, sample_rate);
    std::fprintf(stderr, "%s: frame_count=%llu channels=%u sample_rate=%u samples=%zu\n", label,
                 (unsigned long long) frame_count, channels, sample_rate, pcm.size());
    synth_voice_profile_free(profile);

    if (!synthesized || frame_count == 0 || pcm.empty()) {
        return false;
    }
    // Not merely "some samples exist" -- at least one must be nonzero, or a
    // build that produced silence (e.g. a synthesis that stopped on frame
    // zero and zero-filled the rest) would still pass a bare non-empty check.
    bool any_nonzero = false;
    for (float sample : pcm) {
        if (sample != 0.0f) {
            any_nonzero = true;
            break;
        }
    }
    return any_nonzero;
}

// Task 5's own round trip: create a design Profile, serialize it, load the
// bytes back through synth_voice_profile_load_from_memory (the call this
// task's own capability-bit fix is what makes succeed at all against this
// package -- before it, this returned SYNTH_ERR_UNSUPPORTED_VOICE
// unconditionally, since the dispatch never reached this family's loader for
// a VoiceDesign Model), and confirm the LOADED profile synthesizes non-silent
// audio too -- not merely that loading returns SYNTH_OK.
bool run_round_trip_case(synth_model_t * model) {
    const std::string                instruct = "A cheerful, bright female voice speaking with fast pacing.";
    synth_voice_description_params_t create_params;
    synth_voice_description_params_init(&create_params, sizeof(create_params));
    create_params.description      = instruct.data();
    create_params.description_size = instruct.size();

    synth_voice_profile_t * created     = nullptr;
    const synth_status_t created_status = synth_voice_profile_create_from_description(model, &create_params, &created);
    std::fprintf(stderr, "round-trip: create_from_description -> %d\n", int(created_status));
    if (created_status != SYNTH_OK || created == nullptr) {
        return false;
    }

    synth_voice_profile_serialize_params_t serialize_params;
    synth_voice_profile_serialize_params_init(&serialize_params, sizeof(serialize_params));
    synth_byte_buffer_t * bytes            = nullptr;
    const synth_status_t  serialize_status = synth_voice_profile_serialize(created, &serialize_params, &bytes);
    std::fprintf(stderr, "round-trip: serialize -> %d (data_size=%llu)\n", int(serialize_status),
                 bytes != nullptr ? (unsigned long long) bytes->data_size : 0ull);
    synth_voice_profile_free(created);
    if (serialize_status != SYNTH_OK || bytes == nullptr || bytes->data_size == 0) {
        return false;
    }

    synth_voice_profile_load_params_t load_params;
    synth_voice_profile_load_params_init(&load_params, sizeof(load_params));
    load_params.data      = bytes->data;
    load_params.data_size = bytes->data_size;

    synth_voice_profile_t * loaded        = nullptr;
    const synth_status_t    loaded_status = synth_voice_profile_load_from_memory(model, &load_params, &loaded);
    std::fprintf(stderr, "round-trip: load_from_memory -> %d\n", int(loaded_status));
    synth_byte_buffer_free(bytes);
    if (loaded_status != SYNTH_OK || loaded == nullptr) {
        return false;
    }

    std::vector<float> pcm;
    uint64_t           frame_count = 0;
    uint32_t           channels    = 0;
    uint32_t           sample_rate = 0;
    const bool         synthesized = synthesize_pcm(model, loaded, "This is the loaded-back profile speaking.", pcm,
                                                    frame_count, channels, sample_rate);
    std::fprintf(stderr, "round-trip: synthesize(loaded) frame_count=%llu channels=%u sample_rate=%u samples=%zu\n",
                 (unsigned long long) frame_count, channels, sample_rate, pcm.size());
    synth_voice_profile_free(loaded);

    if (!synthesized || frame_count == 0 || pcm.empty()) {
        return false;
    }
    bool any_nonzero = false;
    for (float sample : pcm) {
        if (sample != 0.0f) {
            any_nonzero = true;
            break;
        }
    }
    return any_nonzero;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];

    synth_model_load_params_t load_params;
    synth_model_load_params_init(&load_params, sizeof(load_params));
    load_params.backend   = SYNTH_BACKEND_CPU;
    synth_model_t * model = nullptr;
    SYNTH_TEST_CHECK(synth_model_load(model_path.c_str(), &load_params, &model) == SYNTH_OK);

    // Case 1: empty instruct, design D3's unconditioned path -- the same
    // shape tests/qwen3_tts_voicedesign_prefill_real.cpp's own prefill-only
    // gate exercises, now carried all the way through the decode loop.
    SYNTH_TEST_CHECK(run_case(model, "empty-instruct", ""));

    // Case 2: a non-empty instruct -- the shape Task 4's own oracle
    // comparison (tests/tolerances/qwen3-tts.json's "prefill" probe,
    // voicedesign-nonempty-instruct-en) measures at the prefill layer only.
    // This is the first place ANYTHING drives that instruct past the
    // prefill into a real decode loop.
    SYNTH_TEST_CHECK(run_case(model, "nonempty-instruct",
                              "A cheerful, bright female voice speaking with fast pacing and high energy."));

    // Case 3 (Task 5): create -> serialize -> load -> synthesize, entirely
    // through the public seam.
    SYNTH_TEST_CHECK(run_round_trip_case(model));

    synth_model_free(model);
    return 0;
}
