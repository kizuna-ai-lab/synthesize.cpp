// The repeated-run and resource-cleanup check, through the public seam.
//
// Family-independent on purpose. Every Golden Manifest in this repository
// declares `resource_cleanup` on every case and nothing has ever read the field,
// so the gap this closes is not one family's -- what differs between families is
// only the input a request carries, which arrives on the command line.
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
#include <cstdlib>
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
constexpr uint64_t kSeed          = 1;

// What a request carries, which is the only family-specific thing here. VITS and
// Kokoro take token ids through a symbol-map frontend; Qwen3-TTS takes UTF-8
// text through a byte-pair one. A Voice and a language are optional because a
// fixed-default package refuses a Voice it does not have.
struct RequestSpec {
    // A frame means something different in each family -- a hop of 256 samples
    // in VITS, a duration step of 600 in Kokoro, a codec frame of 1920 here --
    // so a limit that is generous for one refuses the others outright.
    uint64_t             max_frames = 0;
    bool                 tokens     = false;
    std::string          text;
    std::vector<int32_t> token_ids;
    std::string          voice;
    std::string          language;
};

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
bool run_cycle(const char *            model_path,
               const RequestSpec &     spec,
               synth_backend_request_t backend,
               uint64_t &              out_frames,
               uint64_t &              out_digest) {
    synth_model_load_params_t load_params;
    synth_model_load_params_init(&load_params, sizeof(load_params));
    load_params.backend = backend;

    synth_model_t *      model  = nullptr;
    const synth_status_t loaded = synth_model_load(model_path, &load_params, &model);
    if (loaded != SYNTH_OK || model == nullptr) {
        std::fprintf(stderr, "cleanup: load -> %d\n", (int) loaded);
        return false;
    }

    synth_context_t * context = nullptr;
    if (synth_context_create(model, &context) != SYNTH_OK || context == nullptr) {
        synth_model_free(model);
        return false;
    }

    synth_request_t request;
    synth_request_init(&request, sizeof(request));
    if (spec.tokens) {
        request.input_kind  = SYNTH_INPUT_TOKEN_IDS;
        request.input_data  = spec.token_ids.data();
        request.input_count = spec.token_ids.size();
    } else {
        request.input_kind  = SYNTH_INPUT_TEXT_UTF8;
        request.input_data  = spec.text.c_str();
        request.input_count = spec.text.size();
    }
    if (!spec.voice.empty()) {
        request.voice_id      = spec.voice.c_str();
        request.voice_id_size = spec.voice.size();
    }
    if (!spec.language.empty()) {
        request.language_tag      = spec.language.c_str();
        request.language_tag_size = spec.language.size();
    }
    request.seed              = kSeed;
    request.max_output_frames = spec.max_frames;

    synth_audio_buffer_t * audio = nullptr;
    synth_result_t         result;
    synth_result_init(&result, sizeof(result));
    const synth_status_t synthesized = synth_synthesize_to_buffer(context, &request, &audio, &result);
    if (synthesized != SYNTH_OK || audio == nullptr) {
        std::fprintf(stderr, "cleanup: synthesize -> %d\n", (int) synthesized);
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

int run_backend(const char *            model_path,
                const RequestSpec &     spec,
                synth_backend_request_t backend,
                const char *            label,
                int                     cycles) {
    uint64_t          first_frames = 0;
    uint64_t          first_digest = 0;
    std::vector<long> resident;

    for (int cycle = 0; cycle < cycles; ++cycle) {
        uint64_t frames = 0;
        uint64_t hash   = 0;
        SYNTH_TEST_CHECK(run_cycle(model_path, spec, backend, frames, hash));
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
            std::fprintf(stderr, "%s: post-free resident floor %ld -> %ld KB over %d cycles (%+ld KB)\n", label, early,
                         late, cycles, late - early);
            SYNTH_TEST_CHECK(late - early <= kAllowedGrowthKb);
        }
    }
    return 0;
}

}  // namespace

// usage: <model.gguf> <cycles> <max-frames> <input> [voice] [language]
//   <input> is "text:Hi." or "tokens:1,2,3" -- which one a family takes is a
//   property of its frontend, not of this check.
int main(int argc, char ** argv) {
    SYNTH_TEST_CHECK(argc >= 5 && argc <= 7);
    const int cycles = static_cast<int>(std::strtol(argv[2], nullptr, 10));
    SYNTH_TEST_CHECK(cycles >= 2);

    RequestSpec spec;
    spec.max_frames = std::strtoull(argv[3], nullptr, 10);
    SYNTH_TEST_CHECK(spec.max_frames > 0);
    const std::string input(argv[4]);
    if (input.rfind("tokens:", 0) == 0) {
        spec.tokens = true;
        for (size_t at = 7; at <= input.size();) {
            const size_t comma = input.find(',', at);
            const size_t end   = comma == std::string::npos ? input.size() : comma;
            if (end > at) {
                spec.token_ids.push_back(
                    static_cast<int32_t>(std::strtol(input.substr(at, end - at).c_str(), nullptr, 10)));
            }
            if (comma == std::string::npos) {
                break;
            }
            at = comma + 1;
        }
        SYNTH_TEST_CHECK(!spec.token_ids.empty());
    } else if (input.rfind("tokensfile:", 0) == 0) {
        // VITS's tokens come from a Golden payload rather than a literal: its
        // frontend is a symbol map, and an arbitrary id sequence makes the
        // duration predictor ask for more frames than any limit allows.
        spec.tokens      = true;
        std::FILE * file = std::fopen(input.substr(11).c_str(), "rb");
        SYNTH_TEST_CHECK(file != nullptr);
        int32_t token = 0;
        while (std::fread(&token, sizeof(token), 1, file) == 1) {
            spec.token_ids.push_back(token);
        }
        std::fclose(file);
        SYNTH_TEST_CHECK(!spec.token_ids.empty());
    } else if (input.rfind("text:", 0) == 0) {
        spec.text = input.substr(5);
        SYNTH_TEST_CHECK(!spec.text.empty());
    } else {
        std::fprintf(stderr, "input must begin with text: or tokens:\n");
        return 1;
    }
    if (argc >= 6) {
        spec.voice = argv[5];
    }
    if (argc >= 7) {
        spec.language = argv[6];
    }

    // CPU is the baseline every package claims.
    SYNTH_TEST_CHECK(run_backend(argv[1], spec, SYNTH_BACKEND_CPU, "cpu", cycles) == 0);

    // Every further backend the build claims runs the same cycles. A backend
    // that is absent from this build is not being claimed by it.
    if (synth_backend_available(SYNTH_BACKEND_CUDA) == SYNTH_TRUE) {
        SYNTH_TEST_CHECK(run_backend(argv[1], spec, SYNTH_BACKEND_CUDA, "cuda", cycles) == 0);
    }
    return 0;
}
