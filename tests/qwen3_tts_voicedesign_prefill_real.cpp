// Plan 1's completion gate: the talker's assembled prefill at an empty
// instruct, against the real 1.7B VoiceDesign package, with the speaker slot
// correctly ABSENT.
//
// Adapter, not a test -- the same role qwen3_tts_replay_real.cpp and
// qwen3_tts_codec_encoder_driver.cpp already play in this directory (see
// their own header comments): it asserts nothing and prints what it
// observed, including the p95-relative distance against a supplied oracle
// prefill when one is given. Registered with synth_register_integration_target
// only, no CTest add_test -- the tolerance cell this driver's output feeds
// lives in tests/tolerances/qwen3-tts.json, recorded by hand from this
// binary's own printed "p95_relative", not enforced inside it.
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
    if (argc != 5 && argc != 6) {
        std::fprintf(stderr,
                     "usage: %s <model.gguf> <text> <language> <out-prefill.f32> [oracle-prefill.f32]\n"
                     "  the fifth argument is optional -- when given, this driver also prints the\n"
                     "  p95-relative distance between its own prefill and the oracle's\n",
                     argv[0]);
        return 2;
    }
    const std::string model_path(argv[1]);
    const std::string text(argv[2]);
    const std::string language(argv[3]);
    const std::string out_path(argv[4]);
    const bool        have_oracle = argc == 6;
    const std::string oracle_path = have_oracle ? argv[5] : std::string();

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

    // The split model.cpp:972-975 already applies to a real request's token
    // ids: the first kAssistantRolePrefixTokens are the role prefix, the last
    // kAssistantSuffixTokens are the turn's closing markers, and everything
    // between is the text to speak.
    synth::qwen3tts::TalkerPromptRequest request;
    request.role_tokens.assign(token_ids.begin(),
                               token_ids.begin() + std::ptrdiff_t(synth::qwen3tts::kAssistantRolePrefixTokens));
    request.text_tokens.assign(token_ids.begin() + std::ptrdiff_t(synth::qwen3tts::kAssistantRolePrefixTokens),
                               token_ids.end() - std::ptrdiff_t(synth::qwen3tts::kAssistantSuffixTokens));
    request.has_language   = true;
    request.language_token = language_token;
    // has_speaker left at its default (false): no codec-vocabulary slot and
    // no instruct block at all, matching upstream's own empty-instruct,
    // no-speaker path (design decisions D3 and section 5.3's third row).
    // request.has_reference also stays at its default (false): no ICL block.

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

    std::printf(
        "{\"variant\": \"%s\", \"positions\": %zu, \"hidden_size\": %zu, \"codec_offset\": %lld, "
        "\"external_speaker_index\": %lld, \"has_speaker\": %s, \"has_reference\": %s",
        hparams.model_variant.c_str(), positions, hidden_size, (long long) codec_offset,
        (long long) prompt.external_speaker_index, request.has_speaker ? "true" : "false",
        request.has_reference ? "true" : "false");

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
        // The shape mismatch itself is printed alongside the number, so a
        // reader is never left thinking the two prefills were the same length
        // when they were not.
        const size_t compared_positions = oracle_positions < positions ? oracle_positions : positions;
        // The same statistic, same operand order (oracle is the right operand
        // and the denominator) tests/qwen3_tts_icl_real.cpp and
        // scripts/validate-qwen3-tts-codec_encoder.py already use -- see
        // tests/qwen3_tts_percentile.h's own header comment.
        const double p95                = synth::qwen3_tts::testing::reconstruction_p95_relative(
            got_prefill.data(), oracle_prefill.data(), compared_positions, hidden_size);
        std::printf(
            ", \"oracle_path\": \"%s\", \"oracle_positions\": %zu, \"shapes_match\": %s, "
            "\"compared_positions\": %zu, \"p95_relative\": %.6f",
            oracle_path.c_str(), oracle_positions, oracle_positions == positions ? "true" : "false", compared_positions,
            p95);
    }
    std::printf("}\n");
    return 0;
}
