/* Drives one public OmniVoice request and reports what came back.
 *
 * Task 5 validates the public seam this family's Model::synthesize lands
 * behind: seed reporting, same-seed repeatability, and language resolution.
 * None of it injects tensors -- that is what separates this driver from the
 * replay one (omnivoice_replay_real.cpp).
 *
 * An adapter, not a test: it asserts nothing and prints what it observed.
 * The relations and their meaning live in Task 6's
 * scripts/validate-omnivoice-public.py. Transcribed from
 * tests/qwen3_tts_public_real.c, with two differences that follow directly
 * from this family's shape: no voice-id positional (the Preset Voice
 * Catalog is empty and the package default is unnamed auto-voice) and no
 * backend selector (Plan 2's placement note: every graph in this family
 * runs on the CPU).
 */

#include "synthesize.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Monotonic, because a real-time factor divides by it and a wall clock that
 * can step backwards would report a negative synthesis. */
static double now_seconds(void) {
    struct timespec moment;
    clock_gettime(CLOCK_MONOTONIC, &moment);
    return (double) moment.tv_sec + (double) moment.tv_nsec * 1e-9;
}

/* Returns false on any failure. Swallowing it printed the success JSON over a
 * missing or truncated file, and the validator then compared against that
 * instead of seeing an error. */
static int write_pcm(const char * path, const float * samples, uint64_t count) {
    FILE * file = fopen(path, "wb");
    if (file == NULL) {
        return 0;
    }
    const size_t written = fwrite(samples, sizeof(float), (size_t) count, file);
    const int    flushed = fclose(file) == 0;
    return written == (size_t) count && flushed;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <model.gguf> <out.pcm> <language-tag|-> <seed|random> [max-frames] [threads]\n",
                argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    const char * out_path   = argv[2];
    const char * language   = strcmp(argv[3], "-") == 0 ? NULL : argv[3];
    const char * seed_text  = argv[4];

    /* The text arrives on stdin so a case can carry any UTF-8 without
     * quoting. */
    static char  text[65536];
    const size_t text_size = fread(text, 1, sizeof text - 1, stdin);
    text[text_size]        = '\0';

    /* No backend selector: this family has no accelerator path yet (see the
     * placement note at the top of src/arch/omnivoice/model.cpp), so the
     * default the params initializer chooses is the only one there is. */
    synth_model_load_params_t load_params;
    synth_model_load_params_init(&load_params, sizeof load_params);

    const double    load_started = now_seconds();
    synth_model_t * model        = NULL;
    synth_status_t  status       = synth_model_load(model_path, &load_params, &model);
    const double    load_seconds = now_seconds() - load_started;
    if (status != SYNTH_OK) {
        fprintf(stderr, "load -> %d\n", (int) status);
        return 1;
    }

    synth_context_t * context = NULL;
    status                    = synth_context_create(model, &context);
    if (status != SYNTH_OK) {
        fprintf(stderr, "context -> %d\n", (int) status);
        synth_model_free(model);
        return 1;
    }

    /* 0 keeps whatever the context chose for itself, which is what an
     * embedder that never calls the setter gets. */
    if (argc > 6) {
        status = synth_context_set_threads(context, (int32_t) strtol(argv[6], NULL, 10));
        if (status != SYNTH_OK) {
            fprintf(stderr, "set_threads -> %d\n", (int) status);
            synth_context_free(context);
            synth_model_free(model);
            return 1;
        }
    }
    int32_t threads_used = 0;
    synth_context_get_threads(context, &threads_used);

    synth_request_t request;
    synth_request_init(&request, sizeof request);
    request.input_kind  = SYNTH_INPUT_TEXT_UTF8;
    request.input_data  = text;
    request.input_count = text_size;
    if (language != NULL) {
        request.language_tag      = language;
        request.language_tag_size = strlen(language);
    }
    /* No voice_id and no voice_profile: the catalog is empty and the package
     * default is the unnamed auto-voice, which is what leaving both null
     * asks for. */
    request.seed              = strcmp(seed_text, "random") == 0 ? SYNTH_SEED_RANDOM : strtoull(seed_text, NULL, 10);
    /* PCM frames, per docs/c-interface.md; the family converts to its own
     * native codec frames internally. */
    request.max_output_frames = argc > 5 ? strtoull(argv[5], NULL, 10) : 0;

    synth_audio_buffer_t * audio = NULL;
    synth_result_t         result;
    memset(&result, 0, sizeof result);
    result.struct_size = sizeof result;

    const double synthesis_started = now_seconds();
    status                         = synth_synthesize_to_buffer(context, &request, &audio, &result);
    const double synthesis_seconds = now_seconds() - synthesis_started;
    if (status != SYNTH_OK) {
        fprintf(stderr, "synthesize -> %d\n", (int) status);
        synth_context_free(context);
        synth_model_free(model);
        return 1;
    }

    if (!write_pcm(out_path, audio->samples, audio->frame_count)) {
        fprintf(stderr, "cannot write %s\n", out_path);
        synth_audio_buffer_free(audio);
        synth_context_free(context);
        synth_model_free(model);
        return 1;
    }
    printf(
        "{\"status\": %d, \"frames\": %llu, \"sample_rate\": %u, \"actual_seed\": \"%llu\", "
        "\"load_seconds\": %.4f, \"synthesis_seconds\": %.4f, \"threads\": %d, "
        "\"resolved_language\": \"%.*s\"}\n",
        (int) status, (unsigned long long) audio->frame_count, audio->sample_rate,
        (unsigned long long) result.actual_seed, load_seconds, synthesis_seconds, (int) threads_used,
        (int) result.resolved_language_tag_size,
        result.resolved_language_tag == NULL ? "" : result.resolved_language_tag);

    synth_audio_buffer_free(audio);
    synth_context_free(context);
    synth_model_free(model);
    return 0;
}
