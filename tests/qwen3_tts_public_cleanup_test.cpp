// Qwen3-TTS repeated-run and resource-cleanup check, through the public seam.
//
// docs/backends.md gate 6 asks a claimed Execution Backend to "Pass repeated-run
// and resource-cleanup checks", and docs/port-validation.md phase 5 says every
// claimed backend "reruns the same cases, proves real placement, records
// operational measurements, and passes repeated-run cleanup". A single load that
// synthesizes once cannot show either half: a handle that never releases its
// weights, a context that outlives its free, or a backend buffer that is
// abandoned per cycle all pass a one-shot run and then sink a long-lived
// embedder.
//
// So this drives whole cycles -- load, context, synthesize, free -- on every
// backend the build claims, and asserts two things a one-shot run cannot:
//
//   * the cycles are interchangeable: identical frame count and bit-identical
//     PCM at a fixed seed, cycle after cycle, so nothing accumulates in the
//     process that changes what the next load produces;
//   * the process comes back to where it started: the floor of the resident set
//     between cycles stops moving.
//
// Neither the process start nor the first free is the baseline. The first few
// cycles legitimately keep what a later one does not pay again -- allocator
// arenas, the frozen backend registry, and on CUDA the device context and its
// kernel images -- and v1 never unloads a backend module, so those are
// process-lifetime costs by design rather than leaks. What is left after that
// warm-up oscillates: measured here, the post-free resident set swings through a
// 15 MB band on CPU as glibc thread arenas are taken and given back, with no
// trend. A last-minus-first comparison reads that band as growth. The floor does
// not move, so the statistic is the minimum over a window: a retained model,
// context, or backend buffer raises the floor, arena churn only raises peaks.
//
// Run this under the sanitizer build as well: LeakSanitizer answers at
// allocation granularity what the resident-set bound can only approximate.

#include "synthesize.h"
#include "test-assert.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Enough cycles that the warm-up window and both measurement windows are real,
// few enough that the test stays inside an ordinary integration timeout. The
// sanitizer build passes a smaller count: there the leak verdict comes from
// LeakSanitizer, which needs a couple of cycles rather than a trend.
constexpr int      kDefaultCycles = 12;
constexpr int      kWarmupCycles  = 2;
constexpr int      kWindowCycles  = 5;
constexpr uint64_t kMaxFrames     = 24;
constexpr uint64_t kSeed         = 1;
// The manifest's minimal case, which is the shortest real synthesis the family
// has.
constexpr const char * kText     = "Hi.";
constexpr const char * kVoice    = "aiden";
constexpr const char * kLanguage = "en";

// Allowed rise of the post-free resident-set floor between the two windows.
// Measured drift with the floor statistic is 4 KB on CPU and 364 KB on CUDA over
// 30 cycles; the smallest thing a cycle could actually retain -- one codec
// backend buffer -- is two orders of magnitude above this bound, and a retained
// model is three.
constexpr long kAllowedGrowthKb = 4096;

#if defined(__linux__)
long resident_kb() {
    std::FILE * file = std::fopen("/proc/self/status", "r");
    if (file == nullptr) {
        return -1;
    }
    char line[256];
    long value = -1;
    while (std::fgets(line, sizeof(line), file) != nullptr) {
        if (std::strncmp(line, "VmRSS:", 6) == 0) {
            value = std::strtol(line + 6, nullptr, 10);
            break;
        }
    }
    std::fclose(file);
    return value;
}
#else
// Where the resident set is not readable through a documented interface the
// cycle-stability half still runs; the sanitizer build carries the leak half.
long resident_kb() {
    return -1;
}
#endif

// FNV-1a over the PCM bytes. Bit-identical is the claim, so a hash of the bytes
// is the assertion; a tolerance here would defeat the point.
uint64_t digest(const float * samples, uint64_t count) {
    const unsigned char * bytes = reinterpret_cast<const unsigned char *>(samples);
    uint64_t              hash  = 1469598103934665603ULL;
    for (uint64_t i = 0; i < count * sizeof(float); ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

// One full public cycle. Everything it opens, it closes.
bool run_cycle(const char *              model_path,
               synth_backend_request_t   backend,
               uint64_t &                out_frames,
               uint64_t &                out_digest) {
    synth_model_load_params_t load_params;
    synth_model_load_params_init(&load_params, sizeof(load_params));
    load_params.backend = backend;

    synth_model_t * model = nullptr;
    if (synth_model_load(model_path, &load_params, &model) != SYNTH_OK || model == nullptr) {
        return false;
    }

    synth_context_t * context = nullptr;
    if (synth_context_create(model, &context) != SYNTH_OK || context == nullptr) {
        synth_model_free(model);
        return false;
    }

    synth_request_t request;
    synth_request_init(&request, sizeof(request));
    request.input_kind        = SYNTH_INPUT_TEXT_UTF8;
    request.input_data        = kText;
    request.input_count       = std::strlen(kText);
    request.voice_id          = kVoice;
    request.voice_id_size     = std::strlen(kVoice);
    request.language_tag      = kLanguage;
    request.language_tag_size = std::strlen(kLanguage);
    request.seed              = kSeed;
    request.max_output_frames = kMaxFrames;

    synth_audio_buffer_t * audio = nullptr;
    synth_result_t         result;
    synth_result_init(&result, sizeof(result));
    if (synth_synthesize_to_buffer(context, &request, &audio, &result) != SYNTH_OK || audio == nullptr) {
        synth_context_free(context);
        synth_model_free(model);
        return false;
    }

    out_frames = audio->frame_count;
    out_digest = digest(audio->samples, audio->frame_count);

    synth_audio_buffer_free(audio);
    // The Loaded Model must outlive every Synthesis Context created from it
    // (docs/c-interface.md), so the context goes first.
    synth_context_free(context);
    synth_model_free(model);
    return true;
}

// Lowest post-free resident set in [begin, end), or -1 when unreadable.
long floor_kb(const std::vector<long> & resident, int begin, int end) {
    long lowest = -1;
    for (int i = begin; i < end && i < static_cast<int>(resident.size()); ++i) {
        if (resident[i] > 0 && (lowest < 0 || resident[i] < lowest)) {
            lowest = resident[i];
        }
    }
    return lowest;
}

int run_backend(const char * model_path, synth_backend_request_t backend, const char * label, int cycles) {
    uint64_t          first_frames = 0;
    uint64_t          first_digest = 0;
    std::vector<long> resident;

    for (int cycle = 0; cycle < cycles; ++cycle) {
        uint64_t frames = 0;
        uint64_t hash   = 0;
        SYNTH_TEST_CHECK(run_cycle(model_path, backend, frames, hash));
        SYNTH_TEST_CHECK(frames > 0);
        if (cycle == 0) {
            first_frames = frames;
            first_digest = hash;
        } else {
            // A repeated run is a repeat: same length, same samples.
            SYNTH_TEST_CHECK(frames == first_frames);
            SYNTH_TEST_CHECK(hash == first_digest);
        }
        resident.push_back(resident_kb());
    }

    // Two windows separated by the warm-up: the floor of the first against the
    // floor of the last. Too few cycles to form both windows leaves the cycle
    // comparison above as the whole check, which is what the sanitizer run wants.
    if (cycles >= kWarmupCycles + 2 * kWindowCycles) {
        const long early = floor_kb(resident, kWarmupCycles, kWarmupCycles + kWindowCycles);
        const long late  = floor_kb(resident, cycles - kWindowCycles, cycles);
        if (early > 0 && late > 0) {
            std::fprintf(stderr, "%s: post-free resident floor %ld -> %ld KB over %d cycles (%+ld KB)\n", label,
                         early, late, cycles, late - early);
            SYNTH_TEST_CHECK(late - early <= kAllowedGrowthKb);
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    SYNTH_TEST_CHECK(argc == 2 || argc == 3);
    const int cycles = argc == 3 ? static_cast<int>(std::strtol(argv[2], nullptr, 10)) : kDefaultCycles;
    SYNTH_TEST_CHECK(cycles >= 2);

    // CPU is the baseline every package claims.
    SYNTH_TEST_CHECK(run_backend(argv[1], SYNTH_BACKEND_CPU, "cpu", cycles) == 0);

    // Every further backend the build claims runs the same cycles. A backend
    // that is absent from this build is not being claimed by it.
    if (synth_backend_available(SYNTH_BACKEND_CUDA) == SYNTH_TRUE) {
        SYNTH_TEST_CHECK(run_backend(argv[1], SYNTH_BACKEND_CUDA, "cuda", cycles) == 0);
    }
    return 0;
}
