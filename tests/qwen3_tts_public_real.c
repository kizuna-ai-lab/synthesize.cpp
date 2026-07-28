/* Drives one public request and reports what came back.
 *
 * Stage 5 phase 3 validates the public seam rather than the graphs: seed
 * reporting, same-seed repeatability, Voice and language resolution, and that a
 * repeated load leaves nothing behind. None of it injects tensors, which is what
 * separates this phase from the replay one.
 *
 * An adapter, not a test: it asserts nothing and prints what it observed. The
 * relations and their meaning live in scripts/validate-qwen3-tts-public.py.
 */

#include "synthesize.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Monotonic, because a real-time factor divides by it and a wall clock that can
 * step backwards would report a negative synthesis. */
static double now_seconds(void) {
    struct timespec moment;
    clock_gettime(CLOCK_MONOTONIC, &moment);
    return (double) moment.tv_sec + (double) moment.tv_nsec * 1e-9;
}

static void write_pcm(const char * path, const float * samples, uint64_t count) {
    FILE * file = fopen(path, "wb");
    if (file == NULL) {
        return;
    }
    fwrite(samples, sizeof(float), (size_t) count, file);
    fclose(file);
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr,
                "usage: %s <model.gguf> <out.pcm> <voice-id> <language-tag|-> <seed|random> "
                "[max-frames] [cpu|cuda]\n",
                argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    const char * out_path   = argv[2];
    const char * voice_id   = argv[3];
    const char * language   = strcmp(argv[4], "-") == 0 ? NULL : argv[4];
    const char * seed_text  = argv[5];

    /* The text arrives on stdin so a case can carry any UTF-8 without quoting. */
    static char  text[65536];
    const size_t text_size = fread(text, 1, sizeof text - 1, stdin);
    text[text_size]        = '\0';

    /* Selected through the public enum rather than the family's own device
     * lookup: this phase validates the seam a caller actually has, so which
     * device the core resolves CUDA to is part of what is under test. */
    const char * backend_text = argc > 7 ? argv[7] : "cpu";

    synth_model_load_params_t load_params;
    synth_model_load_params_init(&load_params, sizeof load_params);
    if (strcmp(backend_text, "cuda") == 0) {
        load_params.backend = SYNTH_BACKEND_CUDA;
    } else if (strcmp(backend_text, "cpu") == 0) {
        load_params.backend = SYNTH_BACKEND_CPU;
    } else {
        fprintf(stderr, "unknown backend %s\n", backend_text);
        return 2;
    }

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

    synth_request_t request;
    synth_request_init(&request, sizeof request);
    request.input_kind    = SYNTH_INPUT_TEXT_UTF8;
    request.input_data    = text;
    request.input_count   = text_size;
    request.voice_id      = voice_id;
    request.voice_id_size = strlen(voice_id);
    if (language != NULL) {
        request.language_tag      = language;
        request.language_tag_size = strlen(language);
    }
    request.seed              = strcmp(seed_text, "random") == 0 ? SYNTH_SEED_RANDOM : strtoull(seed_text, NULL, 10);
    request.max_output_frames = argc > 6 ? strtoull(argv[6], NULL, 10) : 512;

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

    write_pcm(out_path, audio->samples, audio->frame_count);
    printf(
        "{\"status\": %d, \"frames\": %llu, \"sample_rate\": %u, \"actual_seed\": \"%llu\", "
        "\"load_seconds\": %.4f, \"synthesis_seconds\": %.4f, "
        "\"resolved_voice\": \"%.*s\", \"resolved_language\": \"%.*s\"}\n",
        (int) status, (unsigned long long) audio->frame_count, audio->sample_rate,
        (unsigned long long) result.actual_seed, load_seconds, synthesis_seconds, (int) result.resolved_voice_id_size,
        result.resolved_voice_id == NULL ? "" : result.resolved_voice_id, (int) result.resolved_language_tag_size,
        result.resolved_language_tag == NULL ? "" : result.resolved_language_tag);

    synth_audio_buffer_free(audio);
    synth_context_free(context);
    synth_model_free(model);
    return 0;
}
