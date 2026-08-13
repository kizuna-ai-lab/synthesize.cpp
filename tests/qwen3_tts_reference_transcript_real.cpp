// The reference transcript's ids, against the ones upstream actually passed.
//
// Task 2's ICL dumper records `ref_id` -- the value upstream handed
// generate_icl_prompt, after its own `_build_ref_text` wrapper and its own
// [3:-2] slice (qwen3_tts_model.py:272-273 and :598, sliced at
// modeling_qwen3_tts.py:2191) -- as `<case>/prompt/ref_text_ids.i32`. This
// compares the port's ids against those, exactly. Token ids are discrete;
// there is no tolerance to carry and none is invented.
//
// It needs the real Base package because the wrap-then-slice idiom is only
// meaningful against the package's own 151k-entry vocabulary: what the
// wrapping changes is which merges are reachable across the turn's boundary,
// and a synthetic vocabulary can only show that the mechanism exists (see
// tests/qwen3_tts_bpe_test.cpp's check_reference_transcript_ids), never that
// this vocabulary spends 3 and 2 tokens on the turn's markers.
//
// The transcript is read from a file the build writes out of the oracle's own
// alignment.json rather than pinned here, so a re-dump with a different
// reference clip cannot leave this test comparing ids to a transcript that no
// longer produced them.

#include "arch/qwen3-tts/qwen3-tts.h"
#include "synthesize.h"
#include "test-assert.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

bool read_file(const std::string & path, std::string & out) {
    std::FILE * file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        std::printf("  cannot open %s\n", path.c_str());
        return false;
    }
    out.clear();
    char   buffer[4096];
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
    if (!read_file(path, bytes)) {
        return false;
    }
    if (bytes.size() % sizeof(int32_t) != 0 || bytes.empty()) {
        std::printf("  %s is %zu bytes, not a non-empty int32 array\n", path.c_str(), bytes.size());
        return false;
    }
    out.resize(bytes.size() / sizeof(int32_t));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return true;
}

int check_case(const synth::qwen3tts::Model & model,
               const std::string &            case_id,
               const std::string &            transcript_path,
               const std::string &            ids_path) {
    std::string transcript;
    SYNTH_TEST_CHECK(read_file(transcript_path, transcript));
    SYNTH_TEST_CHECK(!transcript.empty());

    std::vector<int32_t> expected;
    SYNTH_TEST_CHECK(read_int32_file(ids_path, expected));

    std::vector<int32_t> got;
    SYNTH_TEST_CHECK(model.tokenize_reference_transcript(transcript, got) == SYNTH_OK);

    if (got.size() != expected.size()) {
        std::printf("  %s: %zu ids, oracle has %zu\n", case_id.c_str(), got.size(), expected.size());
    }
    SYNTH_TEST_CHECK(got.size() == expected.size());
    for (size_t index = 0; index < expected.size(); ++index) {
        if (got[index] != expected[index]) {
            std::printf("  %s: id %zu is %d, oracle has %d\n", case_id.c_str(), index, got[index], expected[index]);
        }
        SYNTH_TEST_CHECK(got[index] == expected[index]);
    }
    std::printf("  %s: %zu reference ids match the oracle exactly\n", case_id.c_str(), got.size());
    return 0;
}

// The wrapper is not decoration, on this package's own vocabulary.
//
// The pre-tokenizer's newline branch is greedy, so a transcript that opens with
// a newline has that newline absorbed into the turn's own -- the pair is a
// single token, and the slice takes it away with the role prefix. The
// observable consequence is that two different transcripts reach the talker as
// the same ids. Tokenizing the bare transcript instead keeps them apart, which
// is the whole reason this port cannot take that shortcut.
int check_wrapping_changes_the_ids(const synth::qwen3tts::Model & model) {
    std::vector<int32_t> with_newline;
    std::vector<int32_t> without_newline;
    SYNTH_TEST_CHECK(model.tokenize_reference_transcript("\nHello", with_newline) == SYNTH_OK);
    SYNTH_TEST_CHECK(model.tokenize_reference_transcript("Hello", without_newline) == SYNTH_OK);
    SYNTH_TEST_CHECK(!without_newline.empty());
    SYNTH_TEST_CHECK(with_newline == without_newline);

    // And the count that says so: bare "Hello" is one token on this vocabulary,
    // so "\nHello" reaching the same one token means the newline is gone rather
    // than merged into something wider.
    SYNTH_TEST_CHECK(without_newline.size() == 1);
    std::printf("  \"\\nHello\" and \"Hello\" both reach the talker as id %d\n", without_newline[0]);
    return 0;
}

// The design's §9 row, on the real package rather than a fixture: a transcript
// that names no speech is refused instead of tokenizing to a real id.
int check_blank_transcript_is_refused(const synth::qwen3tts::Model & model) {
    std::vector<int32_t> ids;
    SYNTH_TEST_CHECK(model.tokenize_reference_transcript("", ids) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(model.tokenize_reference_transcript("   ", ids) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(ids.empty());
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    // <model> then triples of <case-id> <transcript-file> <ids-file>.
    if (argc < 5 || (argc - 2) % 3 != 0) {
        std::printf("usage: %s <model> (<case-id> <transcript-file> <ids-file>)...\n", argv[0]);
        return 1;
    }

    std::unique_ptr<synth::qwen3tts::Model> model;
    SYNTH_TEST_CHECK(synth::qwen3tts::Model::load_cpu(argv[1], model) == SYNTH_OK);
    SYNTH_TEST_CHECK(model != nullptr);

    for (int index = 2; index + 2 < argc; index += 3) {
        SYNTH_TEST_CHECK(check_case(*model, argv[index], argv[index + 1], argv[index + 2]) == 0);
    }
    SYNTH_TEST_CHECK(check_wrapping_changes_the_ids(*model) == 0);
    SYNTH_TEST_CHECK(check_blank_transcript_is_refused(*model) == 0);
    return 0;
}
