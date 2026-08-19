/* Drives one public request and reports what came back.
 *
 * Stage 5 phase 3 validates the public seam rather than the graphs: seed
 * reporting, same-seed repeatability, Voice and language resolution, and that a
 * repeated load leaves nothing behind. None of it injects tensors, which is what
 * separates this phase from the replay one.
 *
 * An adapter, not a test: it asserts nothing and prints what it observed. The
 * relations and their meaning live in scripts/validate-qwen3-tts-public.py.
 *
 * Stage 3 Plan 2 Task 6 added two more things this adapter can be asked to do,
 * both still adapters rather than tests:
 *
 *   - A `desc:TEXT` voice creates a Voice Profile from a Description Text
 *     instruct through synth_voice_profile_create_from_description, the
 *     VoiceDesign counterpart to `ref:`'s reference-audio path. `TEXT` may be
 *     empty (`desc:` alone), which is design decision D3's legal unconditioned
 *     instruct, not a malformed request.
 *   - A `probe:KIND` voice does not synthesize at all: it attempts exactly one
 *     Voice Profile creation call -- `probe:reference` calls
 *     create_from_reference with zero references, `probe:description` calls
 *     create_from_description with an empty instruct, and
 *     `probe:description-language` calls create_from_description with a
 *     language tag attached -- and prints the synth_status_t it observed on
 *     stdout, succeeding (exit 0) whichever way that call went. Both calls'
 *     own package-support gate runs before either would need real content
 *     (see src/voice-profile.cpp's dispatchers), so an empty/zero-count params
 *     struct reaches that gate exactly as a populated one would. This adapter
 *     still asserts nothing; scripts/validate-qwen3-tts-public.py is what
 *     decides whether a given status means the refusal design section 6.4
 *     promises actually held.
 */

/* This driver reaches outside ISO C for two things -- `clock_gettime` for the
 * real-time factor and `strdup` for the `ref:` spec -- and CMake asks for C11.
 * They are declared today only because CMake leaves C_EXTENSIONS on and the
 * build lands on gnu11, so the macro states the dependency instead of relying
 * on that default. Note tests/omnivoice_public_real.c has the same
 * `clock_gettime` dependency and no macro; the tree declares none anywhere. */
#define _POSIX_C_SOURCE 200809L

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

/* Raw little-endian float32 mono samples, which is what this adapter already
 * WRITES for its output. Reading the same shape avoids a fifth WAV parser in
 * this directory -- and in C, where the four that exist are C++ -- while
 * keeping the reference an ordinary buffer of audio. The caller converts
 * whatever it has; scripts/validate-qwen3-tts-public.py does it with soundfile,
 * from the manifest's own pinned artifact. */
static float * read_f32(const char * path, uint64_t * out_count) {
    FILE * file = fopen(path, "rb");
    if (file == NULL) {
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return NULL;
    }
    const long size = ftell(file);
    if (size < 0 || (size % (long) sizeof(float)) != 0 || size == 0) {
        fclose(file);
        return NULL;
    }
    rewind(file);
    float * samples = (float *) malloc((size_t) size);
    if (samples == NULL) {
        fclose(file);
        return NULL;
    }
    const size_t count = (size_t) size / sizeof(float);
    const size_t read  = fread(samples, sizeof(float), count, file);
    fclose(file);
    if (read != count) {
        free(samples);
        return NULL;
    }
    *out_count = (uint64_t) count;
    return samples;
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr,
                "usage: %s <model.gguf> <out.pcm> "
                "<voice-id|ref:PATH.f32[:TRANSCRIPT]|desc:TEXT|probe:KIND> <language-tag|-> "
                "<seed|random> [max-frames] [cpu|cuda] [threads]\n"
                "  a `ref:` voice creates a Voice Profile from reference audio instead of naming a\n"
                "  Preset Voice, which is the only way to synthesize from a package that catalogues\n"
                "  none -- qwen3-tts-12hz-0-6b-base declares preset_ids: []\n"
                "  a `desc:` voice creates a Voice Profile from a Description Text instruct (which\n"
                "  may be empty) -- the only way to synthesize from a package whose declared Voice\n"
                "  Profile source is description_text, qwen3-tts-12hz-1-7b-voicedesign\n"
                "  a `probe:KIND` voice attempts one Voice Profile creation call (KIND is `reference`,\n"
                "  `description` or `description-language`) and reports its status instead of\n"
                "  synthesizing; see this file's own header comment\n",
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

    /* 0 keeps whatever the context chose for itself, which is what an embedder
     * that never calls the setter gets. */
    if (argc > 8) {
        status = synth_context_set_threads(context, (int32_t) strtol(argv[8], NULL, 10));
        if (status != SYNTH_OK) {
            fprintf(stderr, "set_threads -> %d\n", (int) status);
            synth_context_free(context);
            synth_model_free(model);
            return 1;
        }
    }
    int32_t threads_used = 0;
    synth_context_get_threads(context, &threads_used);

    /* `probe:KIND` short-circuits before any request is built: it exists only
     * to observe the status one Voice Profile creation call returns against
     * *this* loaded package, refusal included, and has nothing to synthesize
     * either way. Checked first so it never falls through into the `ref:`/
     * `desc:` handling below, whose job is to prepare a Profile that then
     * feeds a real synthesis. */
    if (strncmp(voice_id, "probe:", 6) == 0) {
        const char *            probe_kind    = voice_id + 6;
        synth_status_t          probe_status  = SYNTH_ERR_INVALID_ARG;
        synth_voice_profile_t * probe_profile = NULL;
        if (strcmp(probe_kind, "reference") == 0) {
            /* Zero references, not an omitted or malformed params struct: the
             * package-support gate in synth_voice_profile_create_from_reference
             * runs BEFORE reference_count is ever consulted (src/voice-
             * profile.cpp), so a package that does not support reference audio
             * at all is refused here without this adapter needing to read any
             * actual recording from disk. */
            synth_voice_reference_params_t reference_params;
            synth_voice_reference_params_init(&reference_params, sizeof reference_params);
            reference_params.references      = NULL;
            reference_params.reference_count = 0;
            probe_status = synth_voice_profile_create_from_reference(model, &reference_params, &probe_profile);
        } else if (strcmp(probe_kind, "description") == 0 || strcmp(probe_kind, "description-language") == 0) {
            /* Same reasoning as the reference arm above, mirrored: an empty
             * instruct is itself a legal request (D3), so this is not a
             * degenerate params struct being abused for the probe -- it is the
             * same shape a real empty-instruct request already legitimately
             * takes, with a language tag attached only for the second kind,
             * which docs/superpowers/specs/2026-08-18-qwen3-tts-stage-3-
             * design.md's D5 and Task 5's `voice_profile.
             * description_language_unsupported` refuse regardless of package. */
            synth_voice_description_params_t description_params;
            synth_voice_description_params_init(&description_params, sizeof description_params);
            description_params.seed = 0;
            if (strcmp(probe_kind, "description-language") == 0) {
                description_params.language_tag      = "en";
                description_params.language_tag_size = 2;
            }
            probe_status = synth_voice_profile_create_from_description(model, &description_params, &probe_profile);
        } else {
            fprintf(stderr, "unknown probe kind %s\n", probe_kind);
            synth_context_free(context);
            synth_model_free(model);
            return 2;
        }
        /* Whatever came back -- SYNTH_OK or a refusal -- is reported, not
         * judged: scripts/validate-qwen3-tts-public.py decides what a given
         * status means for a given package. A successful probe still frees the
         * Profile it made rather than leaking it. */
        printf("{\"probe\": \"%s\", \"status\": %d}\n", probe_kind, (int) probe_status);
        synth_voice_profile_free(probe_profile);
        synth_context_free(context);
        synth_model_free(model);
        return 0;
    }

    /* A `ref:` voice becomes a Voice Profile prepared through the public seam,
     * which is the path a Base-package caller actually has; a `desc:` voice
     * does the same for a VoiceDesign package's Description Text source.
     * Everything after this point is identical for every kind of Voice -- the
     * request carries either a voice_id or a voice_profile and nothing else
     * changes -- so the checks the validator runs are the same checks. */
    synth_voice_profile_t * profile          = NULL;
    float *                 reference_pcm    = NULL;
    uint64_t                reference_frames = 0;
    if (strncmp(voice_id, "ref:", 4) == 0) {
        char * spec = strdup(voice_id + 4);
        if (spec == NULL) {
            fprintf(stderr, "out of memory\n");
            synth_context_free(context);
            synth_model_free(model);
            return 1;
        }
        /* An optional transcript after a colon selects transcript-assisted
         * (ICL) mode; without one the Profile is prepared in x-vector mode. */
        char * transcript = strchr(spec, ':');
        if (transcript != NULL) {
            *transcript = '\0';
            ++transcript;
        }
        reference_pcm = read_f32(spec, &reference_frames);
        if (reference_pcm == NULL) {
            fprintf(stderr, "cannot read %s as raw float32 mono samples\n", spec);
            free(spec);
            synth_context_free(context);
            synth_model_free(model);
            return 1;
        }

        synth_voice_reference_t reference;
        synth_voice_reference_init(&reference, sizeof reference);
        reference.samples       = reference_pcm;
        reference.frame_count   = reference_frames;
        reference.sample_rate   = 24000;
        reference.channel_count = 1;
        if (transcript != NULL && transcript[0] != '\0') {
            reference.transcript      = transcript;
            reference.transcript_size = strlen(transcript);
            if (language != NULL) {
                reference.language_tag      = language;
                reference.language_tag_size = strlen(language);
            }
        }

        synth_voice_reference_params_t reference_params;
        synth_voice_reference_params_init(&reference_params, sizeof reference_params);
        reference_params.references       = &reference;
        reference_params.reference_count  = 1;
        reference_params.reference_stride = sizeof reference;

        status = synth_voice_profile_create_from_reference(model, &reference_params, &profile);
        free(spec);
        if (status != SYNTH_OK) {
            fprintf(stderr, "voice_profile_create_from_reference -> %d\n", (int) status);
            free(reference_pcm);
            synth_context_free(context);
            synth_model_free(model);
            return 1;
        }
    } else if (strncmp(voice_id, "desc:", 5) == 0) {
        const char * description = voice_id + 5;

        synth_voice_description_params_t description_params;
        synth_voice_description_params_init(&description_params, sizeof description_params);
        description_params.description      = description;
        description_params.description_size = strlen(description);
        /* This family's Description Text preparation has nothing seed-
         * dependent to fix -- unlike OmniVoice's own arm, whose params.seed
         * selects something at preparation time, Qwen3-TTS's payload does not
         * depend on it at all (design D-series decisions; src/voice-
         * profile.cpp's create_qwen3_tts_profile_from_description says so
         * directly). The ABI-wide contract still refuses SYNTH_SEED_RANDOM
         * regardless of family, though, so a fixed concrete placeholder is
         * passed here rather than threading the *synthesis* seed (`seed_text`
         * below, which may legitimately BE the random sentinel) into a field
         * this family never reads. */
        description_params.seed             = 0;

        status = synth_voice_profile_create_from_description(model, &description_params, &profile);
        if (status != SYNTH_OK) {
            fprintf(stderr, "voice_profile_create_from_description -> %d\n", (int) status);
            synth_context_free(context);
            synth_model_free(model);
            return 1;
        }
    }

    synth_request_t request;
    synth_request_init(&request, sizeof request);
    request.input_kind  = SYNTH_INPUT_TEXT_UTF8;
    request.input_data  = text;
    request.input_count = text_size;
    if (profile != NULL) {
        request.voice_profile = profile;
    } else {
        request.voice_id      = voice_id;
        request.voice_id_size = strlen(voice_id);
    }
    if (language != NULL) {
        request.language_tag      = language;
        request.language_tag_size = strlen(language);
    }
    request.seed              = strcmp(seed_text, "random") == 0 ? SYNTH_SEED_RANDOM : strtoull(seed_text, NULL, 10);
    /* PCM frames. The old default of 512 meant codec frames, which is what the
     * public field was wrongly compared against; 512 PCM frames is 21 ms. */
    request.max_output_frames = argc > 6 ? strtoull(argv[6], NULL, 10) : 983040;

    synth_audio_buffer_t * audio = NULL;
    synth_result_t         result;
    memset(&result, 0, sizeof result);
    result.struct_size = sizeof result;

    const double synthesis_started = now_seconds();
    status                         = synth_synthesize_to_buffer(context, &request, &audio, &result);
    const double synthesis_seconds = now_seconds() - synthesis_started;
    if (status != SYNTH_OK) {
        fprintf(stderr, "synthesize -> %d\n", (int) status);
        synth_voice_profile_free(profile);
        free(reference_pcm);
        synth_context_free(context);
        synth_model_free(model);
        return 1;
    }

    if (!write_pcm(out_path, audio->samples, audio->frame_count)) {
        fprintf(stderr, "cannot write %s\n", out_path);
        synth_audio_buffer_free(audio);
        synth_voice_profile_free(profile);
        free(reference_pcm);
        synth_context_free(context);
        synth_model_free(model);
        return 1;
    }
    printf(
        "{\"status\": %d, \"frames\": %llu, \"sample_rate\": %u, \"actual_seed\": \"%llu\", "
        "\"load_seconds\": %.4f, \"synthesis_seconds\": %.4f, \"threads\": %d, "
        "\"resolved_voice\": \"%.*s\", \"resolved_language\": \"%.*s\"}\n",
        (int) status, (unsigned long long) audio->frame_count, audio->sample_rate,
        (unsigned long long) result.actual_seed, load_seconds, synthesis_seconds, (int) threads_used,
        (int) result.resolved_voice_id_size, result.resolved_voice_id == NULL ? "" : result.resolved_voice_id,
        (int) result.resolved_language_tag_size,
        result.resolved_language_tag == NULL ? "" : result.resolved_language_tag);

    synth_audio_buffer_free(audio);
    /* Freed in the order a caller would: the Profile is independent of the
     * Context that consumed it, and the samples it was prepared from are the
     * caller's for the whole of its life. */
    synth_voice_profile_free(profile);
    free(reference_pcm);
    synth_context_free(context);
    synth_model_free(model);
    return 0;
}
