// The ICL prompt block, against the two tracks upstream actually summed.
//
// PER TRACK FIRST, THEN SUMMED, in that order. The summed comparison alone
// cannot separate a text-track error from a compensating codec-track error --
// which is the whole reason Task 2 dumped the two tracks separately rather than
// only the block. A block that is right for the wrong reasons is exactly the
// failure this plan exists to make visible, because nothing downstream of it
// can: the talker samples, so the audio was never going to match the oracle's
// sample for sample, and a misaligned block still produces fluent speech in
// approximately the right voice.
//
// Inputs come from the oracle rather than from the port wherever the port is
// not what is under test: `ref_text_ids.i32` and `target_text_ids.i32` are the
// ids upstream itself passed (Task 6's test is what compares the port's
// tokenizer against them), and `codes/reference.i32` is the reference grid
// (Task 5's validator is what compares the port's encoder against it). What is
// under test here is only the layout and the two embedding lookups on top of it.
//
// The alignment arm is NOT recomputed here. `alignment.json` reads it out of
// the line of the `return` that executed inside upstream, and the build passes
// that reading in; this test checks the port's trailing schedule against it.
// Recomputing `T1 > T2` would restate the assumption instead of checking it.
//
// TOLERANCE. The oracle's tensors are bfloat16 widened to float32 and the
// port's arithmetic is float32 over the package's own bfloat16 tables, so the
// two disagree at the bfloat16 representation scale and not at float32
// accumulation scale. The summed comparison additionally rounds the port's
// float32 sum back through bfloat16 first, because upstream summed in bfloat16
// and rounded once: prompt_conventions.json's `numerics` block measures that
// gap at 2**-8 = 0.00390625 relative, per case, and says to take the figure
// from there rather than from any report.

#include "arch/qwen3-tts/code-predictor.h"
#include "arch/qwen3-tts/qwen3-tts.h"
#include "arch/qwen3-tts/talker-host.h"
#include "arch/qwen3-tts/talker.h"
#include "arch/qwen3-tts/weights.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "synthesize.h"
#include "test-assert.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
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

bool read_bytes(const std::string & path, std::string & out) {
    std::FILE * file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        std::printf("  cannot open %s\n", path.c_str());
        return false;
    }
    out.clear();
    char   buffer[65536];
    size_t got = 0;
    while ((got = std::fread(buffer, 1, sizeof(buffer), file)) > 0) {
        out.append(buffer, got);
    }
    const bool ok = std::ferror(file) == 0;
    std::fclose(file);
    return ok;
}

bool read_int32_file(const std::string & path, std::vector<int32_t> & out) {
    std::string bytes;
    if (!read_bytes(path, bytes)) {
        return false;
    }
    if (bytes.empty() || bytes.size() % sizeof(int32_t) != 0) {
        std::printf("  %s is %zu bytes, not a non-empty int32 array\n", path.c_str(), bytes.size());
        return false;
    }
    out.resize(bytes.size() / sizeof(int32_t));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return true;
}

bool read_float_file(const std::string & path, std::vector<float> & out) {
    std::string bytes;
    if (!read_bytes(path, bytes)) {
        return false;
    }
    if (bytes.empty() || bytes.size() % sizeof(float) != 0) {
        std::printf("  %s is %zu bytes, not a non-empty float32 array\n", path.c_str(), bytes.size());
        return false;
    }
    out.resize(bytes.size() / sizeof(float));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return true;
}

// Round-to-nearest-even into bfloat16 and straight back out, which is what
// upstream's own sum did once and a float32 add does not do at all.
float round_to_bfloat16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        return value;  // NaN: leave it alone rather than turning it into another
    }
    const uint32_t rounded = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    float          out     = 0.0f;
    std::memcpy(&out, &rounded, sizeof(out));
    return out;
}

// The normalizer Tasks 4 and 5 settled on: max|delta| over the reference's own
// absmax, so one number covers a tensor whose scale this test does not know.
struct Deviation {
    double max_abs   = 0.0;
    double reference = 0.0;

    double relative() const { return reference > 0.0 ? max_abs / reference : max_abs; }
};

Deviation compare(const std::vector<float> & got, const std::vector<float> & expected) {
    Deviation deviation;
    for (size_t index = 0; index < expected.size(); ++index) {
        deviation.reference = std::fmax(deviation.reference, std::fabs(double(expected[index])));
        deviation.max_abs   = std::fmax(deviation.max_abs, std::fabs(double(got[index]) - double(expected[index])));
    }
    return deviation;
}

// Writes the port's two tracks where the float32 decomposition can read them.
// A no-op with the variable unset, which is how CTest runs this: nothing about
// the test's outcome depends on it, and a directory that cannot be written is
// reported rather than failing the comparison it exists to explain.
void dump_tracks(const std::string & case_id, const std::vector<float> & text, const std::vector<float> & codec) {
    const char * root = std::getenv("SYNTH_QWEN3_TTS_ICL_DUMP_DIR");
    if (root == nullptr || root[0] == '\0') {
        return;
    }
    const std::pair<const char *, const std::vector<float> *> outputs[] = {
        { "text",  &text  },
        { "codec", &codec }
    };
    for (const auto & output : outputs) {
        const std::string path = std::string(root) + "/" + case_id + "-" + output.first + ".f32";
        std::FILE *       file = std::fopen(path.c_str(), "wb");
        if (file == nullptr) {
            std::printf("  cannot write %s\n", path.c_str());
            continue;
        }
        std::fwrite(output.second->data(), sizeof(float), output.second->size(), file);
        std::fclose(file);
    }
}

struct CaseSpec {
    std::string case_id;
    std::string prompt_dir;
    std::string codes_path;
    std::string language;
    size_t      t1 = 0;
    size_t      t2 = 0;
    std::string branch;
    size_t      trailing_positions = 0;
    size_t      reference_frames   = 0;
    size_t      hidden_size        = 0;
    size_t      prefix_positions   = 0;
};

// The package names its languages in lower case ("english"); alignment.json
// records upstream's own spelling of the input ("English"). Folded here rather
// than in either of them, because both are right about their own side.
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
    std::printf("  no language token for '%s' among %zu names\n", folded.c_str(), hparams.language_names.size());
    return false;
}

// Reads a [hidden, count] F32 tensor out of a computed graph into a flat vector
// laid out the way the oracle writes its own: C-order [count, hidden], i.e.
// hidden fastest -- which is GGML's [hidden, count] byte for byte.
bool fetch(ggml_tensor * tensor, std::vector<float> & out) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_F32) {
        return false;
    }
    out.resize(size_t(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, out.data(), 0, ggml_nbytes(tensor));
    return true;
}

int check_case(const synth::qwen3tts::Model & model, ggml_backend_t backend, const CaseSpec & spec) {
    const synth::qwen3tts::HParams & hparams = model.hparams();
    const size_t                     groups  = hparams.talker.code_group_count;

    std::vector<int32_t> reference_ids;
    std::vector<int32_t> target_ids;
    std::vector<int32_t> reference_codes;
    SYNTH_TEST_CHECK(read_int32_file(spec.prompt_dir + "/ref_text_ids.i32", reference_ids));
    SYNTH_TEST_CHECK(read_int32_file(spec.prompt_dir + "/target_text_ids.i32", target_ids));
    SYNTH_TEST_CHECK(read_int32_file(spec.codes_path, reference_codes));

    std::vector<float> oracle_text;
    std::vector<float> oracle_codec;
    std::vector<float> oracle_block;
    SYNTH_TEST_CHECK(read_float_file(spec.prompt_dir + "/text_track.f32", oracle_text));
    SYNTH_TEST_CHECK(read_float_file(spec.prompt_dir + "/codec_track.f32", oracle_codec));
    SYNTH_TEST_CHECK(read_float_file(spec.prompt_dir + "/icl_embed.f32", oracle_block));

    // The oracle's own arithmetic, restated from its files rather than trusted:
    // T1 and T2 must be what the ids and the grid say they are.
    SYNTH_TEST_CHECK(reference_ids.size() + target_ids.size() + 1 == spec.t1);
    SYNTH_TEST_CHECK(reference_codes.size() == spec.reference_frames * groups);
    SYNTH_TEST_CHECK(spec.reference_frames + 1 == spec.t2);
    SYNTH_TEST_CHECK(oracle_text.size() == spec.t2 * spec.hidden_size);
    SYNTH_TEST_CHECK(oracle_codec.size() == oracle_text.size());
    SYNTH_TEST_CHECK(oracle_block.size() == oracle_text.size());
    SYNTH_TEST_CHECK(spec.hidden_size == hparams.talker.hidden_size);

    synth::qwen3tts::TalkerPromptRequest request;
    request.role_tokens.assign(3, 0);  // the assistant turn's own three markers
    request.text_tokens.assign(target_ids.begin(), target_ids.end());
    request.has_speaker         = true;
    request.speaker_is_external = true;
    uint32_t language_token     = 0;
    SYNTH_TEST_CHECK(resolve_language(hparams, spec.language, language_token));
    request.has_language   = true;
    request.language_token = language_token;
    request.has_reference  = true;
    request.reference_text_tokens.assign(reference_ids.begin(), reference_ids.end());
    request.reference_codes  = reference_codes;
    request.reference_frames = spec.reference_frames;

    synth::qwen3tts::TalkerPrompt prompt;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(hparams, request, prompt) == SYNTH_OK);
    SYNTH_TEST_CHECK(prompt.positions.size() == spec.prefix_positions + spec.t2);
    SYNTH_TEST_CHECK(prompt.trailing.size() == spec.trailing_positions);

    // The arm, checked against the one upstream's own executed `return`
    // reported. Both arms emit T2 positions, so this is the only thing that
    // separates them.
    if (spec.branch == "truncate") {
        SYNTH_TEST_CHECK(spec.trailing_positions == spec.t1 - spec.t2);
        SYNTH_TEST_CHECK(prompt.trailing.back().text == synth::qwen3tts::TalkerInputPosition::Text::TtsEos);
    } else {
        SYNTH_TEST_CHECK(spec.branch == "pad");
        SYNTH_TEST_CHECK(spec.trailing_positions == 1);
        SYNTH_TEST_CHECK(prompt.trailing[0].text == synth::qwen3tts::TalkerInputPosition::Text::TtsPad);
    }

    std::vector<int32_t> text_ids;
    std::vector<int32_t> codec_ids;
    std::vector<int32_t> acoustic_ids;
    int64_t              codec_offset    = -1;
    int64_t              acoustic_offset = -1;
    SYNTH_TEST_CHECK(synth::qwen3tts::flatten_talker_prompt(hparams, prompt, text_ids, codec_ids, codec_offset,
                                                            acoustic_ids, acoustic_offset) == SYNTH_OK);
    SYNTH_TEST_CHECK(acoustic_offset == int64_t(spec.prefix_positions + 1));

    // The block's own slices, so the two tracks can be compared where upstream
    // dumped them: the last T2 positions of each stream.
    const size_t         block_start = spec.prefix_positions;
    std::vector<int32_t> block_text(text_ids.end() - std::ptrdiff_t(spec.t2), text_ids.end());
    std::vector<int32_t> block_codec(codec_ids.end() - std::ptrdiff_t(spec.t2), codec_ids.end());
    SYNTH_TEST_CHECK(block_text.size() == spec.t2);
    SYNTH_TEST_CHECK(block_codec.size() == spec.t2);
    SYNTH_TEST_CHECK(block_codec[0] == int32_t(hparams.tokens.codec_bos));

    constexpr size_t kNodeBudget = 4096;
    Context          graph_ctx =
        make_context(ggml_tensor_overhead() * (kNodeBudget + 64) + ggml_graph_overhead_custom(kNodeBudget, false));
    ggml_context * gctx  = graph_ctx.get();
    ggml_cgraph *  graph = ggml_new_graph_custom(gctx, kNodeBudget, false);

    ggml_tensor * t_all_text    = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, int64_t(text_ids.size()));
    ggml_tensor * t_all_codec   = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, int64_t(codec_ids.size()));
    ggml_tensor * t_block_text  = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, int64_t(spec.t2));
    ggml_tensor * t_block_codec = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, int64_t(spec.t2));
    ggml_tensor * t_acoustic =
        ggml_new_tensor_2d(gctx, GGML_TYPE_I32, int64_t(spec.reference_frames), int64_t(groups) - 1);
    ggml_backend_buffer_t inputs = ggml_backend_alloc_ctx_tensors(gctx, backend);
    SYNTH_TEST_CHECK(inputs != nullptr);
    ggml_backend_tensor_set(t_all_text, text_ids.data(), 0, ggml_nbytes(t_all_text));
    ggml_backend_tensor_set(t_all_codec, codec_ids.data(), 0, ggml_nbytes(t_all_codec));
    ggml_backend_tensor_set(t_block_text, block_text.data(), 0, ggml_nbytes(t_block_text));
    ggml_backend_tensor_set(t_block_codec, block_codec.data(), 0, ggml_nbytes(t_block_codec));
    ggml_backend_tensor_set(t_acoustic, acoustic_ids.data(), 0, ggml_nbytes(t_acoustic));

    const synth::qwen3tts::TalkerWeights &        talker    = model.talker_weights();
    const synth::qwen3tts::CodePredictorWeights & predictor = model.code_predictor_weights();

    // Track one: text_projection(text_embeddings(cat([ref_id, text_id]))) then
    // tts_eos, padded to T2 in the pad arm (:1978-1981, :2018).
    ggml_tensor * text_track   = synth::qwen3tts::build_text_projection(gctx, talker, t_block_text);
    // Track two: codec_bos through the talker's own table, then per frame the
    // sum of sixteen -- group 0 from that same table, groups 1..15 from the
    // predictor's (:1983-1998).
    ggml_tensor * acoustic_sum = synth::qwen3tts::sum_code_embeddings(gctx, predictor, t_acoustic);
    SYNTH_TEST_CHECK(text_track != nullptr && acoustic_sum != nullptr);
    ggml_tensor * codec_track = ggml_get_rows(gctx, talker.codec_embedding, t_block_codec);
    codec_track = ggml_acc(gctx, codec_track, acoustic_sum, codec_track->nb[1], codec_track->nb[2], codec_track->nb[3],
                           codec_track->nb[1]);

    // And the production entry point, whole: the block must be its tail.
    ggml_tensor * prefill = synth::qwen3tts::build_talker_prefill_input(
        gctx, talker, t_all_text, t_all_codec, codec_offset, nullptr, -1, acoustic_sum, acoustic_offset);
    SYNTH_TEST_CHECK(prefill != nullptr);

    for (ggml_tensor * output : { text_track, codec_track, prefill }) {
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
    }
    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    SYNTH_TEST_CHECK(allocator != nullptr);
    SYNTH_TEST_CHECK(ggml_gallocr_alloc_graph(allocator, graph));
    SYNTH_TEST_CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> got_text;
    std::vector<float> got_codec;
    std::vector<float> got_prefill;
    SYNTH_TEST_CHECK(fetch(text_track, got_text));
    SYNTH_TEST_CHECK(fetch(codec_track, got_codec));
    SYNTH_TEST_CHECK(fetch(prefill, got_prefill));
    ggml_gallocr_free(allocator);
    ggml_backend_buffer_free(inputs);

    SYNTH_TEST_CHECK(got_text.size() == oracle_text.size());
    SYNTH_TEST_CHECK(got_codec.size() == oracle_codec.size());
    SYNTH_TEST_CHECK(got_prefill.size() == prompt.positions.size() * spec.hidden_size);

    // The port's own two tracks, for the float32 decomposition. Off unless
    // SYNTH_QWEN3_TTS_ICL_DUMP_DIR names a directory: the tolerances below are
    // justified by port-vs-upstream-f32 rather than by port-vs-bf16-oracle, and
    // that half is only reproducible if the port's tracks can be got out.
    // scripts/dump_reference_qwen3_tts_icl_prompt_float32.py --port-root reads
    // exactly these two files.
    dump_tracks(spec.case_id, got_text, got_codec);

    // 1. The text track, on its own.
    const Deviation text_deviation  = compare(got_text, oracle_text);
    // 2. The codec track, on its own. Only with both of these settled does the
    //    summed comparison below mean anything.
    const Deviation codec_deviation = compare(got_codec, oracle_codec);

    // 3. The sum, rounded back through bfloat16 the way upstream computed it.
    std::vector<float> summed(got_text.size());
    for (size_t index = 0; index < summed.size(); ++index) {
        summed[index] = round_to_bfloat16(got_text[index] + got_codec[index]);
    }
    const Deviation block_deviation = compare(summed, oracle_block);

    // 4. And the same block as it comes out of build_talker_prefill_input --
    //    the tail of the whole prefill, not a hand-assembled pair of tracks.
    std::vector<float> tail(got_prefill.end() - std::ptrdiff_t(oracle_block.size()), got_prefill.end());
    for (float & value : tail) {
        value = round_to_bfloat16(value);
    }
    const Deviation tail_deviation = compare(tail, oracle_block);

    std::printf("  %s (%s, T1=%zu T2=%zu, block at %zu): text %.3e  codec %.3e  block %.3e  prefill-tail %.3e\n",
                spec.case_id.c_str(), spec.branch.c_str(), spec.t1, spec.t2, block_start, text_deviation.relative(),
                codec_deviation.relative(), block_deviation.relative(), tail_deviation.relative());

    // bfloat16's unit roundoff is 2**-8 = 3.90625e-3 and is what
    // prompt_conventions.json's `numerics` block measures the raw-float32 gap
    // at. The port's own float32 arithmetic over the package's bfloat16 tables
    // adds accumulation on top of that, so these bounds are one bfloat16
    // rounding with room -- not a scale this test invented, and not one wide
    // enough for a shifted track to hide in: a one-position shift moves these
    // by O(1) relative, which Task 7's rule-deletion round measured.
    SYNTH_TEST_CHECK(text_deviation.relative() < 2.0e-2);
    SYNTH_TEST_CHECK(codec_deviation.relative() < 2.0e-2);
    SYNTH_TEST_CHECK(block_deviation.relative() < 2.0e-2);
    SYNTH_TEST_CHECK(tail_deviation.relative() < 2.0e-2);
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    // <model> then ten-field records per case:
    //   <case-id> <prompt-dir> <codes> <language> <T1> <T2> <branch>
    //   <trailing> <frames> <hidden> <prefix>
    // The numeric fields come out of alignment.json at configure time, so the
    // arm this test checks against is the one upstream's own executed return
    // reported -- never something recomputed here.
    constexpr int kFields = 11;
    if (argc < 2 + kFields || (argc - 2) % kFields != 0) {
        std::printf(
            "usage: %s <model> (<case-id> <prompt-dir> <codes> <language> <T1> <T2> <branch> <trailing> "
            "<frames> <hidden> <prefix>)...\n",
            argv[0]);
        return 1;
    }

    std::unique_ptr<synth::qwen3tts::Model> model;
    SYNTH_TEST_CHECK(synth::qwen3tts::Model::load_cpu(argv[1], model) == SYNTH_OK);
    SYNTH_TEST_CHECK(model != nullptr);

    ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    SYNTH_TEST_CHECK(device != nullptr);
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    SYNTH_TEST_CHECK(backend != nullptr);

    int status = 0;
    for (int index = 2; index + kFields - 1 < argc; index += kFields) {
        CaseSpec spec;
        spec.case_id            = argv[index];
        spec.prompt_dir         = argv[index + 1];
        spec.codes_path         = argv[index + 2];
        spec.language           = argv[index + 3];
        spec.t1                 = size_t(std::strtoull(argv[index + 4], nullptr, 10));
        spec.t2                 = size_t(std::strtoull(argv[index + 5], nullptr, 10));
        spec.branch             = argv[index + 6];
        spec.trailing_positions = size_t(std::strtoull(argv[index + 7], nullptr, 10));
        spec.reference_frames   = size_t(std::strtoull(argv[index + 8], nullptr, 10));
        spec.hidden_size        = size_t(std::strtoull(argv[index + 9], nullptr, 10));
        spec.prefix_positions   = size_t(std::strtoull(argv[index + 10], nullptr, 10));
        if (check_case(*model, backend, spec) != 0) {
            status = 1;
            break;
        }
    }
    ggml_backend_free(backend);
    SYNTH_TEST_CHECK(status == 0);
    return 0;
}
