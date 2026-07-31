// Replays the oracle's cases through the port and writes the artifacts the
// Stage 5 validator compares.
//
// This is an adapter, not a test: it asserts nothing numeric and reports what
// it produced. The comparison and its thresholds live in
// scripts/validate-omnivoice-replay.py and tests/tolerances/omnivoice.json.
//
// Three parity channels ride one binary. The step-0 conditional forward's
// probes and the replayed-grid waveform are threshold comparisons; the
// free-running greedy grid is this family's structural_exactness and is
// compared exactly, never under a tolerance.
//
// The third channel is the free-running path's OWN waveform (`pcm_freerun`),
// which is what run_synthesis hands a caller: it is the only artifact that can
// tell a wired-up codec from a codec that merely exists. It is not the same
// comparison as `pcm`. `pcm` replays the ORACLE's grid, so it isolates the
// codec; `pcm_freerun` decodes the grid this port chose, so it is only
// comparable to the oracle's waveform when the two grids agree. When they do
// not -- the dual-admissible case, where the port matches a committed
// ALTERNATE grid -- `--alt-grid` decodes that same alternate through the same
// seam, and the pair is compared for decode determinism instead of for oracle
// parity. Which comparison a case gets is the validator's call, from which
// grid the exact comparison matched.

#include "arch/omnivoice/codec-host.h"
#include "arch/omnivoice/generator-host.h"
#include "arch/omnivoice/omnivoice.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kCodebooks = 8;
constexpr uint32_t kVocab     = 1025;
// The generator's hidden width. Used only to check the read-back size against
// the canvas length this runner already knows from the prompt layout, so a
// silently reshaped probe buffer is caught before it is written out.
constexpr uint32_t kHidden    = 1024;

bool read_file(const std::string & path, std::vector<char> & bytes) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        return false;
    }
    const std::streamsize size = input.tellg();
    input.seekg(0);
    bytes.resize(size_t(size));
    return bool(input.read(bytes.data(), size));
}

bool read_i32(const std::string & path, std::vector<int32_t> & values) {
    std::vector<char> bytes;
    if (!read_file(path, bytes) || bytes.size() % sizeof(int32_t) != 0) {
        return false;
    }
    values.resize(bytes.size() / sizeof(int32_t));
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return true;
}

bool write_f32(const std::string & path, const std::vector<float> & values) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        return false;
    }
    output.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size() * sizeof(float)));
    return bool(output);
}

bool write_i32(const std::string & path, const std::vector<int32_t> & values) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        return false;
    }
    output.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size() * sizeof(int32_t)));
    return bool(output);
}

// A margin that is not finite would print as `nan` or `inf`, which is not JSON
// and which Python's own parser rejects -- an unreadable report, minutes into a
// sweep, instead of a visible anomaly. Emit the spellings that parser accepts.
// Both values are reachable: +inf means a committed token had no rival at all,
// and NaN means the scoring broke.
std::string margin_value(float value) {
    if (std::isnan(value)) {
        return "NaN";
    }
    if (std::isinf(value)) {
        return value > 0.0f ? "Infinity" : "-Infinity";
    }
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.9g", double(value));
    return buffer;
}

// The margin report as one JSON value: `null` when nothing measured it, so the
// key's presence never depends on the flag and the validator can read it
// unconditionally.
std::string margin_json(const synth::omnivoice::MarginReport & margin) {
    if (!margin.measured) {
        return "null";
    }
    char buffer[256];
    std::snprintf(buffer, sizeof(buffer),
                  "{\"kind\": \"%s\", \"value\": %s, \"step\": %u, \"codebook\": %u, \"frame\": %llu}",
                  margin.kind == synth::omnivoice::MarginReport::Kind::selection ? "selection" : "argmax",
                  margin_value(margin.value).c_str(), margin.step, margin.codebook, (unsigned long long) margin.frame);
    return buffer;
}

}  // namespace

int main(int argc, char ** argv) {
    // --margin-report may sit anywhere among the arguments: the probe layers
    // are trailing positionals, so a flag pinned to a fixed slot would have to
    // go before them and would renumber every caller's positional arguments.
    std::vector<std::string> positional;
    bool                     margin_report = false;
    std::string              alt_grid_path;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--margin-report") {
            margin_report = true;
            continue;
        }
        // Takes a value, so a missing one is refused rather than swallowing the
        // next argument -- which for this runner would be a probe layer.
        if (argument == "--alt-grid") {
            if (index + 1 >= argc) {
                std::fprintf(stderr, "--alt-grid needs a path\n");
                return 2;
            }
            alt_grid_path = argv[++index];
            continue;
        }
        // An unrecognized flag is refused rather than parsed as a positional.
        // Falling through would send `--margin-reprot` to std::atoi, which
        // reads it as probe layer 0 and silently produces a run with the
        // report disabled -- the validator refuses an unknown --cases name for
        // the same reason, and a typo must not read as a measurement.
        if (argument.rfind("--", 0) == 0) {
            std::fprintf(stderr, "unknown option %s\n", argument.c_str());
            return 2;
        }
        positional.push_back(argument);
    }
    if (positional.size() < 7) {
        std::fprintf(stderr,
                     "usage: %s <model.gguf> <case-dir> <out-dir> <num-step> <run-greedy 0|1> "
                     "<decode-replay 0|1> <volume peak|none> [probe-layers...] [--margin-report] "
                     "[--alt-grid <grid.i32>]\n",
                     argv[0]);
        return 2;
    }
    const std::string model_path(positional[0]);
    const std::string case_dir(positional[1]);
    const std::string out_dir(positional[2]);
    const uint32_t    num_step   = uint32_t(std::atoi(positional[3].c_str()));
    const bool        run_greedy = positional[4][0] == '1';
    const bool        decode     = positional[5][0] == '1';
    const std::string volume(positional[6]);
    if (volume != "peak" && volume != "none") {
        std::fprintf(stderr, "volume must be peak or none, got %s\n", volume.c_str());
        return 2;
    }

    std::vector<int32_t> text_ids;
    std::vector<int32_t> oracle_grid;
    std::vector<int32_t> oracle_prompt;
    if (!read_i32(case_dir + "/input/token_ids.i32", text_ids) ||
        !read_i32(case_dir + "/codes/grid.i32", oracle_grid) ||
        !read_i32(case_dir + "/input/prompt_grid.i32", oracle_prompt)) {
        std::fprintf(stderr, "cannot read the oracle's artifacts under %s\n", case_dir.c_str());
        return 2;
    }
    if (oracle_grid.empty() || oracle_grid.size() % kCodebooks != 0) {
        std::fprintf(stderr, "codes/grid.i32 holds %zu values, not a whole 8-codebook grid\n", oracle_grid.size());
        return 2;
    }
    const uint64_t frames = oracle_grid.size() / kCodebooks;

    std::vector<int32_t> reference_tokens;
    // Absence is expected, not an error: only the two cloning cases dump a
    // reference stream, and the rest are auto-voice or voice-design.
    const bool           has_reference = read_i32(case_dir + "/ref/tokens.i32", reference_tokens);

    // The prompt this port would assemble must BE the oracle's step-0 input.
    // Byte equality here is a free structural check on the grid layout before
    // a single forward runs. (The orientation trap is real: a transposed grid
    // decodes to audio rather than to an error.)
    synth::omnivoice::PromptLayout prompt;
    if (synth::omnivoice::build_prompt_grid(text_ids, reference_tokens, frames, kCodebooks, kVocab - 1, prompt) !=
        SYNTH_OK) {
        std::fprintf(stderr, "cannot assemble the prompt grid\n");
        return 2;
    }
    if (prompt.grid.size() != oracle_prompt.size() ||
        std::memcmp(prompt.grid.data(), oracle_prompt.data(), oracle_prompt.size() * sizeof(int32_t)) != 0) {
        // Whether a reference stream was loaded is the first thing to check
        // when the grids disagree, so it goes in the message rather than
        // needing a second run to discover.
        std::fprintf(stderr,
                     "assembled prompt grid differs from input/prompt_grid.i32 "
                     "(%zu vs %zu values, reference stream %s)\n",
                     prompt.grid.size(), oracle_prompt.size(), has_reference ? "loaded" : "absent");
        return 2;
    }

    std::unique_ptr<synth::omnivoice::Model> model;
    synth_status_t                           status = synth::omnivoice::Model::load_cpu(model_path, model);
    if (status != SYNTH_OK) {
        std::fprintf(stderr, "load -> %d\n", int(status));
        return 1;
    }

    synth::omnivoice::SynthesisRequest request;
    request.prompt_text_ids  = text_ids;
    request.reference_tokens = reference_tokens;
    request.target_frames    = frames;
    request.num_step         = num_step;
    request.probe_only       = !run_greedy;
    request.margin_report    = margin_report;
    request.threads          = 0;
    for (size_t index = 7; index < positional.size(); ++index) {
        request.probe_layers.push_back(uint32_t(std::atoi(positional[index].c_str())));
    }

    synth::omnivoice::SynthesisOutput output;
    const auto                        started = std::chrono::steady_clock::now();
    status                                    = model->run_synthesis(request, output);
    if (status != SYNTH_OK) {
        std::fprintf(stderr, "run_synthesis -> %d\n", int(status));
        return 1;
    }

    // Probes: hidden buffers are already the oracle's [S, hidden] order; the
    // logits are read back position-major and the oracle stores them C-major
    // [C, S, V], so reorder on write.
    bool ok = true;
    if (!request.probe_layers.empty()) {
        const uint64_t positions = prompt.total();
        if (output.final_hidden.size() != size_t(positions) * kHidden ||
            output.logits_step0.size() != size_t(positions) * kCodebooks * kVocab) {
            std::fprintf(stderr, "probe buffers are %zu hidden and %zu logits for %llu positions\n",
                         output.final_hidden.size(), output.logits_step0.size(), (unsigned long long) positions);
            return 1;
        }
        std::vector<float> reordered(output.logits_step0.size());
        for (uint32_t codebook = 0; codebook < kCodebooks; ++codebook) {
            for (uint64_t position = 0; position < positions; ++position) {
                std::memcpy(reordered.data() + (size_t(codebook) * positions + position) * kVocab,
                            output.logits_step0.data() + (size_t(position) * kCodebooks + codebook) * kVocab,
                            kVocab * sizeof(float));
            }
        }
        ok = write_f32(out_dir + "/logits_step0.f32", reordered) &&
             write_f32(out_dir + "/final.f32", output.final_hidden);
        for (size_t index = 0; index < output.layer_hidden.size() && ok; ++index) {
            ok = write_f32(out_dir + "/hidden_l" + std::to_string(request.probe_layers[index]) + ".f32",
                           output.layer_hidden[index]);
        }
    }
    if (ok && run_greedy) {
        ok = write_i32(out_dir + "/grid.i32", output.codes);
    }

    size_t samples           = 0;
    size_t freerun_samples   = 0;
    size_t alternate_samples = 0;
    if (ok && decode) {
        std::vector<float> audio;
        status = model->decode_codes(oracle_grid, frames, 0, audio);
        if (status != SYNTH_OK) {
            std::fprintf(stderr, "decode_codes -> %d\n", int(status));
            return 1;
        }
        if (volume == "peak") {
            // One formula, one home: the same helper run_synthesis applies, so
            // the replay path cannot drift from the shipped branch.
            synth::omnivoice::apply_no_reference_volume(audio);
        }
        samples = audio.size();
        ok      = write_f32(out_dir + "/pcm.f32", audio);

        // What run_synthesis itself produced, volume branch already applied
        // inside it. Written only alongside the replay decode so the two are
        // always comparable, and only when the greedy path actually ran --
        // a probe-only request returns before there is any audio.
        if (ok && run_greedy) {
            freerun_samples = output.audio.size();
            ok              = write_f32(out_dir + "/pcm_freerun.f32", output.audio);
        }
        if (ok && !alt_grid_path.empty()) {
            std::vector<int32_t> alternate_grid;
            if (!read_i32(alt_grid_path, alternate_grid)) {
                std::fprintf(stderr, "cannot read the alternate grid %s\n", alt_grid_path.c_str());
                return 2;
            }
            std::vector<float> alternate;
            status = model->decode_codes(alternate_grid, alternate_grid.size() / kCodebooks, 0, alternate);
            if (status != SYNTH_OK) {
                std::fprintf(stderr, "decode_codes(--alt-grid) -> %d\n", int(status));
                return 1;
            }
            // The no-reference branch unconditionally, because that is what
            // run_synthesis applied to the waveform this one is compared with.
            // The case's own `volume` argument belongs to the oracle's waveform
            // and says nothing about the port's.
            synth::omnivoice::apply_no_reference_volume(alternate);
            alternate_samples = alternate.size();
            ok                = write_f32(out_dir + "/pcm_alt.f32", alternate);
        }
    }
    if (!ok) {
        std::fprintf(stderr, "cannot write under %s\n", out_dir.c_str());
        return 2;
    }

    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::printf(
        "{\"frames\": %llu, \"samples\": %zu, \"freerun_samples\": %zu, \"alternate_samples\": %zu, "
        "\"probe_layers\": %zu, "
        "\"generator_seconds\": %.4f, \"generator_setup_seconds\": %.4f, \"codec_seconds\": %.4f, "
        "\"placement\": {\"generator\": [%llu, %llu], \"codec\": [%llu, %llu]}, \"margin\": %s, "
        "\"wall_seconds\": %.4f}\n",
        (unsigned long long) frames, samples, freerun_samples, alternate_samples, output.layer_hidden.size(),
        output.generator_seconds, output.generator_setup_seconds, output.codec_seconds,
        (unsigned long long) output.generator_placement.nodes,
        (unsigned long long) output.generator_placement.accelerator_nodes,
        (unsigned long long) output.codec_placement.nodes,
        (unsigned long long) output.codec_placement.accelerator_nodes, margin_json(output.margin).c_str(), wall);
    return 0;
}
