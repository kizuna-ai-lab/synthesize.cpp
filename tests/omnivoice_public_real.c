/* Drives one public OmniVoice request and reports what came back.
 *
 * Task 5 validates the public seam this family's Model::synthesize lands
 * behind: seed reporting, same-seed repeatability, and language resolution.
 * Task 14 adds the cloning path: `--reference <pcm.f32> --transcript <text>`
 * build a Reference Audio Voice Profile before the request and thread it
 * through `request.voice_profile`. Task 15 adds the voice-design path:
 * `--instruct <text>` builds a Description Text Voice Profile the same way.
 * `--reference`/`--transcript` and `--instruct` are mutually exclusive --
 * one synth_request_t carries at most one Voice Profile. None of it injects
 * tensors -- that is what separates this driver from the replay one
 * (omnivoice_replay_real.cpp).
 *
 * An adapter, not a test: it asserts nothing and prints what it observed.
 * The relations and their meaning live in scripts/validate-omnivoice-public.py.
 * Transcribed from tests/qwen3_tts_public_real.c, with one difference that
 * follows directly from this family's shape: no voice-id positional (the
 * Preset Voice Catalog is empty and the package default is unnamed
 * auto-voice).
 *
 * Plan 4 Task 11 adds the optional `[cpu|cuda]` backend positional
 * qwen3-tts's own driver already carries. Before this, `--backend cuda` on
 * scripts/validate-omnivoice-public.py was accepted but INERT: the flag
 * landed in the report dict and never reached this process, so a CUDA claim
 * could report a false green against a run that was actually CPU the whole
 * time. `resolved_device` in the printed JSON is this driver's own positive
 * assertion (accumulated requirement 1): a forgotten backend-claim flip and a
 * correct refusal both leave `synth_model_load` failing the same way, so
 * nothing downstream of a successful load can tell them apart by absence
 * alone -- the field says what `synth_model_get_device` reports AFTER a
 * successful load actually happened, not merely that one was requested.
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

/* Reads a whole file into a freshly malloc'd buffer; `*out_size` receives the
 * byte count. Returns NULL on any failure, leaving `*out_size` untouched. */
static void * read_whole_file(const char * path, size_t * out_size) {
    FILE * file = fopen(path, "rb");
    if (file == NULL) {
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return NULL;
    }
    const long size = ftell(file);
    if (size < 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }
    void * buffer = malloc((size_t) size > 0 ? (size_t) size : 1);
    if (buffer == NULL) {
        fclose(file);
        return NULL;
    }
    const size_t read = fread(buffer, 1, (size_t) size, file);
    const int    ok   = fclose(file) == 0;
    if (!ok || read != (size_t) size) {
        free(buffer);
        return NULL;
    }
    *out_size = (size_t) size;
    return buffer;
}

int main(int argc, char ** argv) {
    /* `--reference <pcm.f32>` and `--transcript <text>` (Task 14) may sit
     * anywhere among the arguments, so they are scanned out first and the
     * remaining positionals keep their original numbering -- the same
     * pattern tests/omnivoice_replay_real.cpp uses for its own trailing
     * flags. */
    const char * reference_path = NULL;
    const char * transcript     = NULL;
    const char * instruct       = NULL;
    const char * positional[8];
    int          positional_count = 0;
    for (int index = 1; index < argc; ++index) {
        if (strcmp(argv[index], "--reference") == 0) {
            if (index + 1 >= argc) {
                fprintf(stderr, "--reference needs a path\n");
                return 2;
            }
            reference_path = argv[++index];
            continue;
        }
        if (strcmp(argv[index], "--transcript") == 0) {
            if (index + 1 >= argc) {
                fprintf(stderr, "--transcript needs text\n");
                return 2;
            }
            transcript = argv[++index];
            continue;
        }
        if (strcmp(argv[index], "--instruct") == 0) {
            if (index + 1 >= argc) {
                fprintf(stderr, "--instruct needs text\n");
                return 2;
            }
            instruct = argv[++index];
            continue;
        }
        if (positional_count >= (int) (sizeof(positional) / sizeof(positional[0]))) {
            fprintf(stderr, "too many positional arguments\n");
            return 2;
        }
        positional[positional_count++] = argv[index];
    }
    if (positional_count < 4) {
        fprintf(stderr,
                "usage: %s <model.gguf> <out.pcm> <language-tag|-> <seed|random> [max-frames] [cpu|cuda] [threads] "
                "[--reference <pcm.f32> --transcript <text>] [--instruct <text>]\n",
                argv[0]);
        return 2;
    }
    if ((reference_path == NULL) != (transcript == NULL)) {
        fprintf(stderr, "--reference and --transcript must be given together\n");
        return 2;
    }
    if (instruct != NULL && reference_path != NULL) {
        fprintf(stderr, "--instruct and --reference are mutually exclusive: one request, one Voice Profile\n");
        return 2;
    }
    const char * model_path = positional[0];
    const char * out_path   = positional[1];
    const char * language   = strcmp(positional[2], "-") == 0 ? NULL : positional[2];
    const char * seed_text  = positional[3];

    /* The text arrives on stdin so a case can carry any UTF-8 without
     * quoting. */
    static char  text[65536];
    const size_t text_size = fread(text, 1, sizeof text - 1, stdin);
    text[text_size]        = '\0';

    /* Selected through the public enum, mirroring qwen3-tts's own driver
     * (tests/qwen3_tts_public_real.c): this phase validates the seam a
     * caller actually has, so which device the core resolves CUDA to is
     * part of what is under test. Defaults to "cpu", so a caller that omits
     * it keeps the old behaviour.
     *
     * Slot 5 was REPURPOSED, not added: it used to be the thread count, and
     * the usage string before Task 11 read "[max-frames] [threads]". A caller
     * that still passes a number there now takes the unknown-backend branch
     * and exits 2 -- loudly, not silently, but it is a break. The only caller
     * in the tree, scripts/validate-omnivoice-public.py, was moved in the same
     * change and passes no thread count at all, so nothing is currently
     * broken. Corrected 2026-08-10: this comment used to claim the slot "did
     * not exist" before, which would have told a future reader there was no
     * compatibility break to reason about. */
    const char * backend_text = positional_count > 5 ? positional[5] : "cpu";

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

    /* Accumulated requirement 1 (Plan 4 progress.md): the amended cleanup
     * test tolerates a forgotten backend-claim flip, because a correct
     * refusal and a forgotten flip both leave `synth_model_load` failing the
     * same way. Nothing downstream of THIS load succeeding can be trusted to
     * prove the claim by itself, so this driver asks the model directly what
     * device it actually landed on and prints it unconditionally --
     * `resolved_device` is not a diagnostic, it is the assertion. A caller
     * that requested "cuda" and gets back "cpu" here has found the exact bug
     * this requirement exists to catch. `memory_total`/`memory_shared` ride
     * along because they are this same query's other two fields and because
     * they are this family's own answer to the UMA peak-memory question
     * (docs/c-interface.md's SYNTH_DEVICE_MEMORY_SHARED, not nvidia-smi,
     * which the Interface deliberately never parses). */
    synth_backend_device_t resolved_device;
    synth_backend_device_init(&resolved_device, sizeof resolved_device);
    status = synth_model_get_device(model, &resolved_device);
    if (status != SYNTH_OK) {
        fprintf(stderr, "get_device -> %d\n", (int) status);
        synth_model_free(model);
        return 1;
    }

    /* The Reference Audio Voice Profile, built once before the request it
     * decorates. `reference_pcm` outlives the profile-creation call (the
     * public interface only borrows it for that synchronous call), so it is
     * freed only after the profile itself no longer needs it. */
    synth_voice_profile_t * profile       = NULL;
    void *                  reference_pcm = NULL;
    if (reference_path != NULL) {
        size_t reference_bytes = 0;
        reference_pcm          = read_whole_file(reference_path, &reference_bytes);
        if (reference_pcm == NULL || reference_bytes % sizeof(float) != 0) {
            fprintf(stderr, "cannot read --reference %s\n", reference_path);
            free(reference_pcm);
            synth_model_free(model);
            return 1;
        }

        synth_voice_reference_t reference;
        synth_voice_reference_init(&reference, sizeof reference);
        reference.samples           = (const float *) reference_pcm;
        reference.frame_count       = reference_bytes / sizeof(float);
        reference.sample_rate       = 24000; /* this family's declared Reference Audio target format */
        reference.channel_count     = 1;
        reference.transcript        = transcript;
        reference.transcript_size   = strlen(transcript);
        /* Fixed rather than a new flag: this driver's only clone caller
         * (scripts/validate-omnivoice-public.py) always pairs --reference
         * with the pinned English reference clip. */
        reference.language_tag      = "en";
        reference.language_tag_size = 2;

        synth_voice_reference_params_t reference_params;
        synth_voice_reference_params_init(&reference_params, sizeof reference_params);
        reference_params.references       = &reference;
        reference_params.reference_count  = 1;
        reference_params.reference_stride = sizeof reference;

        status = synth_voice_profile_create_from_reference(model, &reference_params, &profile);
        if (status != SYNTH_OK) {
            fprintf(stderr, "create_from_reference -> %d\n", (int) status);
            free(reference_pcm);
            synth_model_free(model);
            return 1;
        }
    }

    /* The Description Text ("voice design") Voice Profile (Task 15):
     * mutually exclusive with --reference above (checked before either
     * profile-building block runs), so `profile` is still NULL here on this
     * branch. Language and seed stay at the params initializer's defaults
     * (package default description language, concrete seed zero) -- this
     * driver's only design caller (scripts/validate-omnivoice-public.py)
     * has no need to vary either. */
    if (instruct != NULL) {
        synth_voice_description_params_t description_params;
        synth_voice_description_params_init(&description_params, sizeof description_params);
        description_params.description      = instruct;
        description_params.description_size = strlen(instruct);

        status = synth_voice_profile_create_from_description(model, &description_params, &profile);
        if (status != SYNTH_OK) {
            fprintf(stderr, "create_from_description -> %d\n", (int) status);
            synth_model_free(model);
            return 1;
        }
    }

    synth_context_t * context = NULL;
    status                    = synth_context_create(model, &context);
    if (status != SYNTH_OK) {
        fprintf(stderr, "context -> %d\n", (int) status);
        synth_voice_profile_free(profile);
        free(reference_pcm);
        synth_model_free(model);
        return 1;
    }

    /* 0 keeps whatever the context chose for itself, which is what an
     * embedder that never calls the setter gets. Positional 6, after the new
     * [cpu|cuda] slot at 5 -- qwen3-tts's own driver orders them the same way
     * (max-frames, backend, threads). */
    if (positional_count > 6) {
        status = synth_context_set_threads(context, (int32_t) strtol(positional[6], NULL, 10));
        if (status != SYNTH_OK) {
            fprintf(stderr, "set_threads -> %d\n", (int) status);
            synth_context_free(context);
            synth_voice_profile_free(profile);
            free(reference_pcm);
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
    /* No voice_id: the catalog is empty and the package default is the
     * unnamed auto-voice, which is what leaving it null asks for.
     * `voice_profile` carries the Reference Audio clone or the Description
     * Text design built above (mutually exclusive), or stays null for the
     * same auto-voice path Task 5's checks already cover. */
    request.voice_profile     = profile;
    request.seed              = strcmp(seed_text, "random") == 0 ? SYNTH_SEED_RANDOM : strtoull(seed_text, NULL, 10);
    /* PCM frames, per docs/c-interface.md; the family converts to its own
     * native codec frames internally. */
    request.max_output_frames = positional_count > 4 ? strtoull(positional[4], NULL, 10) : 0;

    synth_audio_buffer_t * audio = NULL;
    synth_result_t         result;
    memset(&result, 0, sizeof result);
    result.struct_size = sizeof result;

    const double synthesis_started = now_seconds();
    status                         = synth_synthesize_to_buffer(context, &request, &audio, &result);
    const double synthesis_seconds = now_seconds() - synthesis_started;
    if (status != SYNTH_OK) {
        fprintf(stderr, "synthesize -> %d\n", (int) status);
        /* synth_synthesize_to_buffer can return a non-null *out_audio
         * alongside a non-OK status (SYNTH_ERR_CANCELLED,
         * SYNTH_ERR_OUTPUT_LIMIT) if samples were already collected when the
         * error fired. Not reachable for OmniVoice today -- its cancellation
         * and output-limit checks both fire before any audio is generated --
         * but freeing unconditionally (a documented no-op on NULL) keeps this
         * robust against a future OmniVoice change to synthesize mid-generation. */
        synth_audio_buffer_free(audio);
        synth_context_free(context);
        synth_voice_profile_free(profile);
        free(reference_pcm);
        synth_model_free(model);
        return 1;
    }

    if (!write_pcm(out_path, audio->samples, audio->frame_count * audio->channel_count)) {
        fprintf(stderr, "cannot write %s\n", out_path);
        synth_audio_buffer_free(audio);
        synth_context_free(context);
        synth_voice_profile_free(profile);
        free(reference_pcm);
        synth_model_free(model);
        return 1;
    }
    printf(
        "{\"status\": %d, \"frames\": %llu, \"sample_rate\": %u, \"actual_seed\": \"%llu\", "
        "\"load_seconds\": %.4f, \"synthesis_seconds\": %.4f, \"threads\": %d, "
        "\"resolved_language\": \"%.*s\", \"resolved_device\": \"%s\", "
        "\"device_memory_total\": %llu, \"device_memory_shared\": %s}\n",
        (int) status, (unsigned long long) audio->frame_count, audio->sample_rate,
        (unsigned long long) result.actual_seed, load_seconds, synthesis_seconds, (int) threads_used,
        (int) result.resolved_language_tag_size,
        result.resolved_language_tag == NULL ? "" : result.resolved_language_tag,
        resolved_device.kind == NULL ? "" : resolved_device.kind, (unsigned long long) resolved_device.memory_total,
        (resolved_device.flags & SYNTH_DEVICE_MEMORY_SHARED) != 0 ? "true" : "false");

    synth_audio_buffer_free(audio);
    synth_context_free(context);
    synth_voice_profile_free(profile);
    free(reference_pcm);
    synth_model_free(model);
    return 0;
}
