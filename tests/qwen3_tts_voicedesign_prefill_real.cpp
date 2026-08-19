// Plan 1's completion gate: the talker's assembled prefill at an empty
// instruct, against the real 1.7B VoiceDesign package, with the speaker slot
// correctly ABSENT.
//
// Prints what it observed, always -- an oracle-relative comparison (when an
// oracle path is given) and, since Task 7's fix round 2, the enforcement
// decision (when a max-relative bound is also given). Without a bound this
// binary behaves exactly like qwen3_tts_replay_real.cpp and
// qwen3_tts_codec_encoder_driver.cpp: an adapter that asserts nothing, useful
// for manual runs. WITH a bound -- which is how tests/CMakeLists.txt's
// registration below always invokes it -- it exits non-zero when the shapes
// disagree or the measured p95-relative distance exceeds the bound, so it is
// no longer merely an adapter in that mode: it is the thing that makes
// tests/tolerances/qwen3-tts.json's committed cell actually enforced, the
// same shape tests/qwen3_tts_icl_prompt_real.cpp already has for
// prompt.icl_embed. The earlier revision of this file printed the number and
// let a human copy it into the JSON by hand; a review found that left the
// completion gate enforcing nothing -- the faulted binary exited 0 -- and
// docs/superpowers/plans/2026-08-18-qwen3-tts-stage-3-plan-1-voicedesign-package.md's
// Task 7 Step 3 carries the erratum recording the plan text this corrects.
// Still registered with synth_register_integration_target only and still an
// integration target behind -DSYNTH_BUILD_INTEGRATION_TESTS=ON -- only the
// exit code changed, not the tier.
//
// WHY THIS TAKES TEXT AND LANGUAGE RATHER THAN THE ORACLE'S OWN TOKEN IDS.
// tests/qwen3_tts_replay_real.cpp and tests/qwen3_tts_icl_prompt_real.cpp both
// read pre-tokenized ids out of an oracle dump, because their job is to
// replay a KNOWN sequence through a later stage without also exercising the
// port's own tokenizer. This driver instead calls Model::tokenize_request
// itself -- the same BPE tables and the same qwen_assistant_turn wrapper
// tests/qwen3_tts_bpe_test.cpp and tests/qwen3_tts_reference_transcript_real.cpp
// already exercise against a real package's vocabulary, shared byte-for-byte
// across every Qwen3-TTS variant (Task 6's report). What Plan 1's completion
// gate has to prove is end-to-end: the same text and language a caller (or
// the oracle dumper) would pass produces the same assembled prefill, which
// only holds if this port's tokenizer AND its prompt assembly both agree with
// upstream -- and a divergence in either belongs in this gate's number, not
// hidden behind a shortcut that only checks the second.
//
// THE BRANCH THIS DRIVER EXISTS TO EXERCISE. Design section 5.3
// (docs/superpowers/specs/2026-08-18-qwen3-tts-stage-3-design.md): today
// src/arch/qwen3-tts/model.cpp:979 sets `prompt_request.has_speaker = true`
// unconditionally, because every case it has handled so far -- a Preset
// Voice, an x-vector or ICL clone -- really does carry a codec-vocabulary
// slot, substituted or not. VoiceDesign carries no speaker at all: the codec
// prefix is `[codec_think, codec_think_bos, language_token, codec_think_eos,
// codec_pad, codec_bos]` with nothing spliced between the last two.
// talker-host.cpp's `if (request.has_speaker) {...}` block has no `else`, so
// the code for that has always existed -- Task 5 proved the branch structurally
// correct by fault injection against a synthetic HParams fixture -- but
// nothing on the model.cpp call path has ever reached it, and this file is
// still the first thing anywhere to build that branch against a REAL
// package's weights and compare the resulting floats to a real oracle.
// model.cpp itself is not touched here: teaching it this third case is
// Plan 2's job, once a Description Text Voice Profile exists to select it.
//
// STAGE 3 PLAN 2 TASK 4 EXTENSION. model.cpp now IS touched (has_speaker is
// conditional, instruct_tokens are threaded in), so this driver gained a
// fourth positional argument, `<instruct>`, tokenized through
// Model::tokenize_instruct -- the same wrap-and-tokenize step
// SynthesisRequest::instruct now drives at synthesis -- and threaded into
// TalkerPromptRequest::instruct_tokens before build_talker_prompt runs.
// Passing "" reproduces Plan 1's own empty-instruct, no-block case exactly
// (tokenize_instruct's own D3 rule: an empty instruct tokenizes to an EMPTY
// vector, so request.instruct_tokens stays empty and this driver's behaviour
// is unchanged from before this task for that one input) -- which is why
// tests/CMakeLists.txt keeps registering Plan 1's empty-instruct case against
// this same binary rather than replacing it: it is the CONTROL that shows a
// non-empty instruct is what moves the number, not a vestige to prune.

#include "arch/qwen3-tts/bpe.h"
#include "arch/qwen3-tts/qwen3-tts.h"
#include "arch/qwen3-tts/talker-host.h"
#include "arch/qwen3-tts/talker.h"
#include "arch/qwen3-tts/weights.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "qwen3_tts_percentile.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

struct ContextDeleter {
    void operator()(ggml_context * context) const {
        if (context != nullptr) {
            ggml_free(context);
        }
    }
};

using Context = std::unique_ptr<ggml_context, ContextDeleter>;

Context make_context(size_t bytes) {
    ggml_init_params parameters{};
    parameters.mem_size = bytes;
    parameters.no_alloc = true;
    return Context(ggml_init(parameters));
}

bool read_f32(const std::string & path, std::vector<float> & values) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        return false;
    }
    const std::streamoff bytes = input.tellg();
    if (bytes <= 0 || bytes % std::streamoff(sizeof(float)) != 0) {
        return false;
    }
    values.resize(size_t(bytes) / sizeof(float));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(values.data()), bytes);
    return input.good();
}

bool write_f32(const std::string & path, const std::vector<float> & values) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        return false;
    }
    output.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size() * sizeof(float)));
    return bool(output);
}

// Folds to lower case and matches against the package's own declared
// language names. The same helper tests/qwen3_tts_icl_prompt_real.cpp
// carries under this name, duplicated rather than shared: every integration
// driver in this directory is its own translation unit with no shared
// test-support library (the family's established convention -- see
// tests/qwen3_tts_codec_encoder_driver.cpp's own note on the four WAV
// readers).
bool resolve_language(const synth::qwen3tts::HParams & hparams, const std::string & name, uint32_t & token) {
    std::string folded = name;
    for (char & character : folded) {
        if (character >= 'A' && character <= 'Z') {
            character = char(character - 'A' + 'a');
        }
    }
    for (size_t index = 0; index < hparams.language_names.size(); ++index) {
        if (hparams.language_names[index] == folded) {
            token = hparams.language_token_ids[index];
            return true;
        }
    }
    return false;
}

bool fetch(ggml_tensor * tensor, std::vector<float> & out) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_F32) {
        return false;
    }
    out.resize(size_t(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, out.data(), 0, ggml_nbytes(tensor));
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 6 || argc > 8) {
        std::fprintf(stderr,
                     "usage: %s <model.gguf> <text> <instruct> <language> <out-prefill.f32> "
                     "[oracle-prefill.f32] [max-relative]\n"
                     "  instruct may be \"\" -- design D3's legal, unconditioned path; matches Plan\n"
                     "  1's own case exactly, since an empty instruct tokenizes to no block at all.\n"
                     "  oracle-prefill.f32 is optional -- when given, this driver also prints the\n"
                     "  p95-relative distance between its own prefill and the oracle's.\n"
                     "  max-relative is optional and requires oracle-prefill.f32 -- when given, this\n"
                     "  driver exits non-zero if the shapes disagree or the measured p95-relative\n"
                     "  distance exceeds it, which is what makes this the completion gate rather than\n"
                     "  an observation of it.\n",
                     argv[0]);
        return 2;
    }
    const std::string model_path(argv[1]);
    const std::string text(argv[2]);
    const std::string instruct(argv[3]);
    const std::string language(argv[4]);
    const std::string out_path(argv[5]);
    const bool        have_oracle  = argc >= 7;
    const std::string oracle_path  = have_oracle ? argv[6] : std::string();
    // Requires argc == 8, which is only reachable once argc >= 7 already
    // held, so a bound is never accepted without an oracle to score it
    // against.
    const bool        have_bound   = argc == 8;
    double            max_relative = 0.0;
    if (have_bound) {
        max_relative = std::strtod(argv[7], nullptr);
        // A missing or malformed bound must not silently become 0 (every
        // comparison fails) or a huge number (every comparison passes) --
        // the same guard tests/qwen3_tts_icl_prompt_real.cpp applies to the
        // bound it reads.
        if (!(max_relative > 0.0 && max_relative < 1.0)) {
            std::fprintf(stderr, "max-relative must be in (0, 1), got '%s'\n", argv[7]);
            return 2;
        }
    }

    std::unique_ptr<synth::qwen3tts::Model> model;
    synth_status_t                          status = synth::qwen3tts::Model::load_cpu(model_path, model);
    if (status != SYNTH_OK) {
        std::fprintf(stderr, "load_cpu -> %d\n", int(status));
        return 1;
    }

    std::vector<int32_t> token_ids;
    status = model->tokenize_request(text, token_ids);
    if (status != SYNTH_OK) {
        std::fprintf(stderr, "tokenize_request -> %d\n", int(status));
        return 1;
    }
    if (token_ids.size() <= synth::qwen3tts::kAssistantRolePrefixTokens + synth::qwen3tts::kAssistantSuffixTokens) {
        std::fprintf(stderr, "tokenized text holds only %zu ids, too few to hold the turn's own markers\n",
                     token_ids.size());
        return 1;
    }

    const synth::qwen3tts::HParams & hparams        = model->hparams();
    uint32_t                         language_token = 0;
    if (!resolve_language(hparams, language, language_token)) {
        std::fprintf(stderr, "'%s' is not among this package's %zu declared languages\n", language.c_str(),
                     hparams.language_names.size());
        return 1;
    }

    // The same wrap-and-tokenize step Model::run_synthesis now calls when a
    // request carries a design Profile -- see this file's own Task 4
    // extension note above. Design D3: an empty `instruct` returns an EMPTY
    // vector, so `request.instruct_tokens` below stays empty and this
    // driver's prefill is byte-identical to Plan 1's own, unmodified case.
    std::vector<int32_t> instruct_ids;
    status = model->tokenize_instruct(instruct, instruct_ids);
    if (status != SYNTH_OK) {
        std::fprintf(stderr, "tokenize_instruct -> %d\n", int(status));
        return 1;
    }

    // The split model.cpp:972-975 already applies to a real request's token
    // ids: the first kAssistantRolePrefixTokens are the role prefix, the last
    // kAssistantSuffixTokens are the turn's closing markers, and everything
    // between is the text to speak.
    synth::qwen3tts::TalkerPromptRequest request;
    // The narrowing is safe for the same reason model.cpp's own is: this
    // family's BPE frontend only ever produces ids in
    // `[0, talker.text_vocab_size)`.
    request.instruct_tokens.reserve(instruct_ids.size());
    for (int32_t id : instruct_ids) {
        request.instruct_tokens.push_back(uint32_t(id));
    }
    request.role_tokens.assign(token_ids.begin(),
                               token_ids.begin() + std::ptrdiff_t(synth::qwen3tts::kAssistantRolePrefixTokens));
    request.text_tokens.assign(token_ids.begin() + std::ptrdiff_t(synth::qwen3tts::kAssistantRolePrefixTokens),
                               token_ids.end() - std::ptrdiff_t(synth::qwen3tts::kAssistantSuffixTokens));
    request.has_language   = true;
    request.language_token = language_token;
    // has_speaker left at its default (false): no codec-vocabulary slot,
    // matching upstream's own no-speaker path for this variant (design
    // section 5.3's third row) regardless of whether an instruct block is
    // present. request.has_reference also stays at its default (false): no
    // ICL block.

    synth::qwen3tts::TalkerPrompt prompt;
    status = synth::qwen3tts::build_talker_prompt(hparams, request, prompt);
    if (status != SYNTH_OK) {
        std::fprintf(stderr, "build_talker_prompt -> %d\n", int(status));
        return 1;
    }

    std::vector<int32_t> text_tokens_flat;
    std::vector<int32_t> codec_tokens_flat;
    std::vector<int32_t> acoustic_codes;
    int64_t              codec_offset    = -1;
    int64_t              acoustic_offset = -1;
    status = synth::qwen3tts::flatten_talker_prompt(hparams, prompt, text_tokens_flat, codec_tokens_flat, codec_offset,
                                                    acoustic_codes, acoustic_offset);
    if (status != SYNTH_OK) {
        std::fprintf(stderr, "flatten_talker_prompt -> %d\n", int(status));
        return 1;
    }

    ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (device == nullptr) {
        std::fprintf(stderr, "no CPU device\n");
        return 1;
    }
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (backend == nullptr) {
        std::fprintf(stderr, "no CPU backend\n");
        return 1;
    }

    constexpr size_t kNodeBudget = 4096;
    Context          graph_ctx =
        make_context(ggml_tensor_overhead() * (kNodeBudget + 64) + ggml_graph_overhead_custom(kNodeBudget, false));
    ggml_context * gctx  = graph_ctx.get();
    ggml_cgraph *  graph = ggml_new_graph_custom(gctx, kNodeBudget, false);

    ggml_tensor *         t_text  = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, int64_t(text_tokens_flat.size()));
    ggml_tensor *         t_codec = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, int64_t(codec_tokens_flat.size()));
    ggml_backend_buffer_t inputs  = ggml_backend_alloc_ctx_tensors(gctx, backend);
    if (inputs == nullptr) {
        std::fprintf(stderr, "input alloc failed\n");
        return 1;
    }
    ggml_backend_tensor_set(t_text, text_tokens_flat.data(), 0, ggml_nbytes(t_text));
    ggml_backend_tensor_set(t_codec, codec_tokens_flat.data(), 0, ggml_nbytes(t_codec));

    const synth::qwen3tts::TalkerWeights & talker = model->talker_weights();
    // No speaker embedding and no acoustic embedding passed: the graph-level
    // defaults, which is what "the slot is absent" means at this seam --
    // build_talker_prefill_input's header comment states that left at their
    // defaults this call is byte-identical to the Stage 1 path.
    ggml_tensor * prefill = synth::qwen3tts::build_talker_prefill_input(gctx, talker, t_text, t_codec, codec_offset);
    if (prefill == nullptr) {
        std::fprintf(stderr, "build_talker_prefill_input returned nullptr\n");
        return 1;
    }
    ggml_set_output(prefill);
    ggml_build_forward_expand(graph, prefill);

    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (allocator == nullptr || !ggml_gallocr_alloc_graph(allocator, graph)) {
        std::fprintf(stderr, "graph alloc failed\n");
        return 1;
    }
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "compute failed\n");
        return 1;
    }

    std::vector<float> got_prefill;
    if (!fetch(prefill, got_prefill)) {
        std::fprintf(stderr, "could not read the prefill tensor\n");
        return 1;
    }
    ggml_gallocr_free(allocator);
    ggml_backend_buffer_free(inputs);
    ggml_backend_free(backend);

    if (!write_f32(out_path, got_prefill)) {
        std::fprintf(stderr, "could not write %s\n", out_path.c_str());
        return 1;
    }

    const size_t hidden_size = size_t(hparams.talker.hidden_size);
    const size_t positions   = prompt.positions.size();

    // The port's own shape, checked with the same rigor the oracle's is
    // checked below. This gate's whole subject is "a shape mismatch is what
    // this comparison catches" (see the fault-injection record in
    // tests/tolerances/qwen3-tts.json), so the driver should not trust its
    // own tensor came out the size build_talker_prompt/flatten_talker_prompt
    // promised without checking. Guaranteed today -- ggml_backend_graph_compute
    // already succeeded on a tensor build_talker_prefill_input sized itself --
    // and checked anyway, the same discipline
    // tests/qwen3_tts_codec_encoder_driver.cpp applies to its own two-run
    // bit-equality check on a guarantee it could also have trusted silently.
    if (hidden_size == 0 || got_prefill.size() != positions * hidden_size) {
        std::fprintf(stderr, "this port's own prefill holds %zu floats, expected %zu positions x %zu hidden = %zu\n",
                     got_prefill.size(), positions, hidden_size, positions * hidden_size);
        return 1;
    }

    std::printf(
        "{\"variant\": \"%s\", \"positions\": %zu, \"hidden_size\": %zu, \"codec_offset\": %lld, "
        "\"external_speaker_index\": %lld, \"has_speaker\": %s, \"has_reference\": %s, "
        "\"instruct_tokens\": %zu",
        hparams.model_variant.c_str(), positions, hidden_size, (long long) codec_offset,
        (long long) prompt.external_speaker_index, request.has_speaker ? "true" : "false",
        request.has_reference ? "true" : "false", request.instruct_tokens.size());

    // Set only inside the have_oracle branch below; used after the closing
    // brace is printed to decide the exit code, which is why it is declared
    // out here rather than staying a temporary inside that block.
    bool   shapes_match = true;
    double p95_relative = 0.0;

    if (have_oracle) {
        std::vector<float> oracle_prefill;
        if (!read_f32(oracle_path, oracle_prefill)) {
            std::fprintf(stderr, "could not read %s as a raw float32 buffer\n", oracle_path.c_str());
            return 1;
        }
        if (hidden_size == 0 || oracle_prefill.size() % hidden_size != 0) {
            std::fprintf(stderr, "oracle prefill holds %zu floats, not a multiple of hidden_size %zu\n",
                         oracle_prefill.size(), hidden_size);
            return 1;
        }
        const size_t oracle_positions   = oracle_prefill.size() / hidden_size;
        shapes_match                    = oracle_positions == positions;
        // A shift-shaped fault -- retaining an extra codec slot, or dropping
        // one -- changes the POSITION COUNT, not just the values at a fixed
        // shape: build_talker_prompt's prefix run is a single vector that
        // every later position's index depends on, so an extra or missing
        // entry there shifts everything after it. reconstruction_p95_relative
        // is defined position-by-position and has no notion of alignment, so
        // it cannot be asked to score two different lengths against each
        // other. Comparing over the overlap -- the shorter of the two lengths,
        // aligned from the front -- still gives a real, reproducible number
        // rather than refusing outright: the leading positions this port and
        // the oracle agree on (role tokens, and the codec-prefix positions
        // before wherever the two sequences first diverge) compare like
        // normal, and a fault that shifts the tail shows up as elevated
        // deviation over the compared positions, not as a missing figure.
        // The shape mismatch itself is printed alongside the number (and, with
        // a bound, is a failure on its own -- see below), so a reader is never
        // left thinking the two prefills were the same length when they were
        // not, and a shape-changing fault cannot hide by producing a small
        // number over whatever positions happened to overlap.
        const size_t compared_positions = oracle_positions < positions ? oracle_positions : positions;
        // The same statistic, same operand order (oracle is the right operand
        // and the denominator) tests/qwen3_tts_icl_real.cpp and
        // scripts/validate-qwen3-tts-codec_encoder.py already use -- see
        // tests/qwen3_tts_percentile.h's own header comment.
        p95_relative = synth::qwen3_tts::testing::reconstruction_p95_relative(got_prefill.data(), oracle_prefill.data(),
                                                                              compared_positions, hidden_size);
        std::printf(
            ", \"oracle_path\": \"%s\", \"oracle_positions\": %zu, \"shapes_match\": %s, "
            "\"compared_positions\": %zu, \"p95_relative\": %.6f",
            oracle_path.c_str(), oracle_positions, shapes_match ? "true" : "false", compared_positions, p95_relative);
        if (have_bound) {
            // exceeds, not "is not less than": AT the bound passes, matching
            // "exits non-zero when the measured p95 exceeds it" -- the exact
            // phrasing this gate was specified against.
            const bool within_bound = p95_relative <= max_relative;
            std::printf(", \"max_relative\": %.6f, \"gate_passed\": %s", max_relative,
                        shapes_match && within_bound ? "true" : "false");
        }
    }
    std::printf("}\n");

    if (have_bound) {
        // BOTH signals fail the gate, independently -- a shape mismatch is
        // exactly what Task 7's own fault injection produced first (19 port
        // positions against the oracle's 18), and a port that happened to
        // keep the p95 comparison's own bookkeeping quiet on a truncated
        // overlap must not be read as passing. This is the enforcement Task 7
        // Fix Round 2 wires up: without it, a build carrying the fault this
        // file exists to catch prints "shapes_match: false" and a 365x p95
        // breach and still exits 0.
        if (!shapes_match) {
            std::fprintf(stderr,
                         "FAILED: the port's prefill has a different position count than the oracle's -- a "
                         "shape mismatch is refused outright, no tolerance applies\n");
            return 1;
        }
        if (p95_relative > max_relative) {
            std::fprintf(stderr, "FAILED: p95_relative %.6f exceeds max_relative %.6f\n", p95_relative, max_relative);
            return 1;
        }
    }
    return 0;
}
