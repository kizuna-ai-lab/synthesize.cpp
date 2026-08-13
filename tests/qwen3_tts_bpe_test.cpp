// The byte-pair pre-tokenizer, against the reference pattern's own splits.
//
// The split is what decides the tokenization, and it is the only stage that can
// be checked without the package's 151k-entry vocabulary. A wrong split produces
// different ids and therefore different, still-plausible speech -- there is no
// error anywhere.
//
// The cases reach every branch of the pattern: contractions, letters with and
// without a leading space, digits that do not group, punctuation runs, newline
// handling, whitespace that gives up its last character to the word after it,
// three non-Latin scripts, an emoji outside the basic plane, and the prompt
// template's own markers.
//
// Reference splits come from scripts/dump_reference_qwen3_tts_tokenizer.py's
// cases run through the pattern itself.

#include "arch/qwen3-tts/bpe.h"
#include "test-assert.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {

struct SplitCase {
    const char *         text;
    const char * const * pieces;
    size_t               count;
};

constexpr const char * kSplit0[] = { "hello", " world" };
constexpr const char * kSplit1[] = { "It", "'s", " a", " test", ",",   " isn", "'t",   " it",
                                     "?",  " I", "'d", " say",  " we", "'ve",  " won", "." };
constexpr const char * kSplit2[] = { "Numbers", " ", "1", "2", "3",   "4",    " and",   " ",
                                     "5",       ".", "6", "7", " do", " not", " group", "." };
constexpr const char * kSplit3[] = { " ", " leading", " and", " trailing", "   " };
constexpr const char * kSplit4[] = { "line",
                                     " one",
                                     "\x0a"
                                     "",
                                     "line",
                                     " two",
                                     "\x0d"
                                     "\x0a"
                                     "",
                                     "line",
                                     " three",
                                     "\x0a"
                                     "\x0a"
                                     "" };
constexpr const char * kSplit5[] = { "punctuation", "!!!", " ...---", " ???" };
constexpr const char * kSplit6[] = {
    "\xe4"
    "\xbd"
    "\xa0"
    "\xe5"
    "\xa5"
    "\xbd"
    "",
    "\xef"
    "\xbc"
    "\x8c"
    "\xe4"
    "\xb8"
    "\x96"
    "\xe7"
    "\x95"
    "\x8c"
    "",
    "\xe3"
    "\x80"
    "\x82"
    "\xe8"
    "\xbf"
    "\x99"
    "\xe6"
    "\x98"
    "\xaf"
    "\xe4"
    "\xb8"
    "\x80"
    "\xe4"
    "\xb8"
    "\xaa"
    "\xe6"
    "\xb5"
    "\x8b"
    "\xe8"
    "\xaf"
    "\x95"
    "",
    "\xe3"
    "\x80"
    "\x82"
    ""
};
constexpr const char * kSplit7[] = {
    "\xe3"
    "\x81"
    "\x93"
    "\xe3"
    "\x82"
    "\x93"
    "\xe3"
    "\x81"
    "\xab"
    "\xe3"
    "\x81"
    "\xa1"
    "\xe3"
    "\x81"
    "\xaf"
    "",
    "\xe3"
    "\x80"
    "\x81"
    "\xe4"
    "\xb8"
    "\x96"
    "\xe7"
    "\x95"
    "\x8c"
    "",
    "\xe3"
    "\x80"
    "\x82"
    ""
};
constexpr const char * kSplit8[] = {
    "\xec"
    "\x95"
    "\x88"
    "\xeb"
    "\x85"
    "\x95"
    "\xed"
    "\x95"
    "\x98"
    "\xec"
    "\x84"
    "\xb8"
    "\xec"
    "\x9a"
    "\x94"
    ""
};
constexpr const char * kSplit9[]  = { "Emoji",
                                      " \xf0"
                                      "\x9f"
                                      "\x98"
                                      "\x80"
                                      "",
                                      " and",
                                      " \xc3"
                                      "\xa9"
                                      "\xc3"
                                      "\xa8"
                                      "\xc3"
                                      "\xaa"
                                      "",
                                      " accents" };
constexpr const char * kSplit10[] = { "<|",
                                      "im",
                                      "_start",
                                      "|>",
                                      "assistant",
                                      "\x0a"
                                      "",
                                      "Hello",
                                      ".<|",
                                      "im",
                                      "_end",
                                      "|>\x0a"
                                      "",
                                      "<|",
                                      "im",
                                      "_start",
                                      "|>",
                                      "assistant",
                                      "\x0a"
                                      "" };
constexpr const char * kSplit11[] = { nullptr };
constexpr const char * kSplit12[] = { " " };

constexpr SplitCase kSplitCases[] = {
    { "hello world",                               kSplit0,  2  },
    { "It's a test, isn't it? I'd say we've won.", kSplit1,  16 },
    { "Numbers 1234 and 5.67 do not group.",       kSplit2,  16 },
    { "  leading and trailing   ",                 kSplit3,  5  },
    { "line one\x0a"
      "line two\x0d"
      "\x0a"
      "line three\x0a"
      "\x0a"
      "",                                     kSplit4,  9  },
    { "punctuation!!! ...--- ???",                 kSplit5,  4  },
    { "\xe4"
      "\xbd"
      "\xa0"
      "\xe5"
      "\xa5"
      "\xbd"
      "\xef"
      "\xbc"
      "\x8c"
      "\xe4"
      "\xb8"
      "\x96"
      "\xe7"
      "\x95"
      "\x8c"
      "\xe3"
      "\x80"
      "\x82"
      "\xe8"
      "\xbf"
      "\x99"
      "\xe6"
      "\x98"
      "\xaf"
      "\xe4"
      "\xb8"
      "\x80"
      "\xe4"
      "\xb8"
      "\xaa"
      "\xe6"
      "\xb5"
      "\x8b"
      "\xe8"
      "\xaf"
      "\x95"
      "\xe3"
      "\x80"
      "\x82"
      "",                                     kSplit6,  4  },
    { "\xe3"
      "\x81"
      "\x93"
      "\xe3"
      "\x82"
      "\x93"
      "\xe3"
      "\x81"
      "\xab"
      "\xe3"
      "\x81"
      "\xa1"
      "\xe3"
      "\x81"
      "\xaf"
      "\xe3"
      "\x80"
      "\x81"
      "\xe4"
      "\xb8"
      "\x96"
      "\xe7"
      "\x95"
      "\x8c"
      "\xe3"
      "\x80"
      "\x82"
      "",                                     kSplit7,  3  },
    { "\xec"
      "\x95"
      "\x88"
      "\xeb"
      "\x85"
      "\x95"
      "\xed"
      "\x95"
      "\x98"
      "\xec"
      "\x84"
      "\xb8"
      "\xec"
      "\x9a"
      "\x94"
      "",                                     kSplit8,  1  },
    { "Emoji \xf0"
      "\x9f"
      "\x98"
      "\x80"
      " and \xc3"
      "\xa9"
      "\xc3"
      "\xa8"
      "\xc3"
      "\xaa"
      " accents",                             kSplit9,  5  },
    { "<|im_start|>assistant\x0a"
      "Hello.<|im_end|>\x0a"
      "<|im_start|>assistant\x0a"
      "",                                     kSplit10, 17 },
    { "",                                          kSplit11, 0  },
    { " ",                                         kSplit12, 1  },
};

int check_pretokenize() {
    for (const SplitCase & test : kSplitCases) {
        const std::string              text(test.text);
        const std::vector<std::string> pieces = synth::qwen3tts::qwen_pretokenize(text);
        if (pieces.size() != test.count) {
            std::printf("  %s: %zu pieces, expected %zu\n", test.text, pieces.size(), test.count);
            SYNTH_TEST_CHECK(pieces.size() == test.count);
        }
        for (size_t index = 0; index < test.count; ++index) {
            if (pieces[index] != test.pieces[index]) {
                std::printf("  piece %zu: got %s, expected %s\n", index, pieces[index].c_str(), test.pieces[index]);
            }
            SYNTH_TEST_CHECK(pieces[index] == test.pieces[index]);
        }
        // The pieces must reassemble into the input exactly: the pattern
        // partitions the text rather than filtering it.
        std::string rejoined;
        for (const std::string & piece : pieces) {
            rejoined += piece;
        }
        SYNTH_TEST_CHECK(rejoined == text);
    }
    return 0;
}

// A vocabulary small enough to write out, in the byte-level alphabet: printable
// ASCII maps to itself and a space is U+0120, which is what makes " a" spell
// "\xc4\xa0a" rather than " a".
constexpr const char * kSpace = "\xc4\xa0";

int check_frontend() {
    synth::qwen3tts::BpeFrontendConfig config;
    config.provider_id      = "synthesize.qwen_bpe";
    config.contract_version = 1;
    config.vocab            = {
        "a",          "b", "c", kSpace, std::string(kSpace) + "a", "ab", std::string(kSpace) + "ab", "<|im_start|>",
        "<|im_end|>", "\n"
    };
    // Ranked most preferred first, which is what makes "ab" beat a space merge.
    config.merges         = { "a b", std::string(kSpace) + " a", std::string(kSpace) + "a b" };
    config.special_tokens = {
        { "<|im_start|>", 7 },
        { "<|im_end|>",   8 }
    };

    std::unique_ptr<synth::TextFrontend> frontend;
    SYNTH_TEST_CHECK(synth::qwen3tts::make_bpe_frontend(config, frontend) == SYNTH_OK);
    SYNTH_TEST_CHECK(frontend != nullptr);
    SYNTH_TEST_CHECK(frontend->input_flags() == SYNTH_INPUT_SUPPORT_TEXT_UTF8);

    auto encode = [&](const std::string & text, std::vector<int32_t> & ids) {
        return frontend->prepare(SYNTH_INPUT_TEXT_UTF8, text.data(), text.size(), 0, ids);
    };

    std::vector<int32_t> ids;
    const auto           equals = [](const std::vector<int32_t> & got, std::vector<int32_t> want) {
        return got == want;
    };
    // "ab" is one merge; the highest-ranked pair wins wherever it sits.
    SYNTH_TEST_CHECK(encode("ab", ids) == SYNTH_OK);
    SYNTH_TEST_CHECK(equals(ids, { 5 }));

    // A leading space belongs to the word after it, so " ab" is one piece. Rank
    // decides what merges inside it, not position: "a b" outranks the space
    // merge, so the result is the space and "ab" rather than "\u0120a" and "b".
    SYNTH_TEST_CHECK(encode(" ab", ids) == SYNTH_OK);
    SYNTH_TEST_CHECK(equals(ids, { 3, 5 }));

    // With the whole piece rankable, it folds to one token. This is the path a
    // real vocabulary takes for a common word.
    synth::qwen3tts::BpeFrontendConfig folded = config;
    folded.merges.push_back(std::string(kSpace) + " ab");
    std::unique_ptr<synth::TextFrontend> folding;
    SYNTH_TEST_CHECK(synth::qwen3tts::make_bpe_frontend(folded, folding) == SYNTH_OK);
    SYNTH_TEST_CHECK(folding->prepare(SYNTH_INPUT_TEXT_UTF8, " ab", 3, 0, ids) == SYNTH_OK);
    SYNTH_TEST_CHECK(equals(ids, { 6 }));

    // Special markers are matched literally: "<|im_start|>" is one token, not the
    // several its characters would merge into.
    SYNTH_TEST_CHECK(encode("<|im_start|>ab<|im_end|>", ids) == SYNTH_OK);
    SYNTH_TEST_CHECK(equals(ids, { 7, 5, 8 }));

    // Empty input is an empty token sequence, not a failure.
    SYNTH_TEST_CHECK(encode("", ids) == SYNTH_OK);
    SYNTH_TEST_CHECK(ids.empty());

    // A byte the vocabulary cannot name is a package defect rather than
    // something to drop, because dropping it would silently change the text.
    SYNTH_TEST_CHECK(encode("z", ids) == SYNTH_ERR_INVALID_ARG);

    // The declared limit is enforced here rather than after the graph is built.
    SYNTH_TEST_CHECK(frontend->prepare(SYNTH_INPUT_TEXT_UTF8, "ab", 2, 0, ids) == SYNTH_OK);
    SYNTH_TEST_CHECK(frontend->prepare(SYNTH_INPUT_TEXT_UTF8, "<|im_start|>ab<|im_end|>", 24, 2, ids) ==
                     SYNTH_ERR_INPUT_TOO_LONG);
    SYNTH_TEST_CHECK(ids.empty());

    // This family consumes raw text; phonemes are a different contract.
    SYNTH_TEST_CHECK(frontend->prepare(SYNTH_INPUT_PHONEMES_UTF8, "ab", 2, 0, ids) == SYNTH_ERR_INVALID_ARG);
    return 0;
}

int check_configuration_rejections() {
    std::unique_ptr<synth::TextFrontend> frontend;

    synth::qwen3tts::BpeFrontendConfig empty;
    empty.contract_version = 1;
    SYNTH_TEST_CHECK(synth::qwen3tts::make_bpe_frontend(empty, frontend) == SYNTH_ERR_GGUF);

    synth::qwen3tts::BpeFrontendConfig future;
    future.contract_version = 2;
    future.vocab            = { "a" };
    SYNTH_TEST_CHECK(synth::qwen3tts::make_bpe_frontend(future, frontend) == SYNTH_ERR_GGUF);

    // A merge is a pair separated by one space. Anything else ranks a pair that
    // can never occur, which silently disables that merge.
    synth::qwen3tts::BpeFrontendConfig malformed;
    malformed.contract_version = 1;
    malformed.vocab            = { "a", "b" };
    malformed.merges           = { "ab" };
    SYNTH_TEST_CHECK(synth::qwen3tts::make_bpe_frontend(malformed, frontend) == SYNTH_ERR_GGUF);

    // A special marker with no text, or with no id, names nothing.
    synth::qwen3tts::BpeFrontendConfig nameless;
    nameless.contract_version = 1;
    nameless.vocab            = { "a" };
    nameless.special_tokens   = {
        { "", 5 }
    };
    SYNTH_TEST_CHECK(synth::qwen3tts::make_bpe_frontend(nameless, frontend) == SYNTH_ERR_GGUF);

    synth::qwen3tts::BpeFrontendConfig negative;
    negative.contract_version = 1;
    negative.vocab            = { "a" };
    negative.special_tokens   = {
        { "<|im_start|>", -1 }
    };
    SYNTH_TEST_CHECK(synth::qwen3tts::make_bpe_frontend(negative, frontend) == SYNTH_ERR_GGUF);
    return 0;
}

int check_turn_wrapping() {
    // The frontend applies the turn wrapper itself, because the core runs the
    // frontend and this family's model consumes the wrapped ids rather than the
    // bare text.
    synth::qwen3tts::BpeFrontendConfig config;
    config.contract_version = 1;
    config.vocab            = { "a", "b", "<|im_start|>", "<|im_end|>", "\n" };
    config.special_tokens   = {
        { "<|im_start|>", 2 },
        { "<|im_end|>",   3 }
    };
    config.prefix = "<|im_start|>";
    config.suffix = "<|im_end|>";

    std::unique_ptr<synth::TextFrontend> frontend;
    SYNTH_TEST_CHECK(synth::qwen3tts::make_bpe_frontend(config, frontend) == SYNTH_OK);

    std::vector<int32_t> ids;
    SYNTH_TEST_CHECK(frontend->prepare(SYNTH_INPUT_TEXT_UTF8, "a", 1, 0, ids) == SYNTH_OK);
    // The markers come back as their own ids, so the wrapper is tokenized as the
    // special tokens it is made of rather than as text.
    SYNTH_TEST_CHECK(ids.size() == 3);
    SYNTH_TEST_CHECK(ids[0] == 2 && ids[2] == 3);

    // An empty request still carries the turn, which is what makes the talker's
    // fixed slice counts meaningful.
    SYNTH_TEST_CHECK(frontend->prepare(SYNTH_INPUT_TEXT_UTF8, "", 0, 0, ids) == SYNTH_OK);
    SYNTH_TEST_CHECK(ids.size() == 2);

    // Without a wrapper the ids are the text's alone.
    synth::qwen3tts::BpeFrontendConfig bare = config;
    bare.prefix.clear();
    bare.suffix.clear();
    std::unique_ptr<synth::TextFrontend> unwrapped;
    SYNTH_TEST_CHECK(synth::qwen3tts::make_bpe_frontend(bare, unwrapped) == SYNTH_OK);
    SYNTH_TEST_CHECK(unwrapped->prepare(SYNTH_INPUT_TEXT_UTF8, "a", 1, 0, ids) == SYNTH_OK);
    SYNTH_TEST_CHECK(ids.size() == 1 && ids[0] == 0);
    return 0;
}

int check_assistant_turn() {
    // A fixed string, not a template the package carries.
    SYNTH_TEST_CHECK(synth::qwen3tts::qwen_assistant_turn("hi") ==
                     "<|im_start|>assistant\nhi<|im_end|>\n<|im_start|>assistant\n");
    // The talker's layout slices the tokenized turn at these counts, and
    // upstream slices at the same two. "Upstream", not "the reference": the
    // reference *turn* below slices at 3 and -2.
    SYNTH_TEST_CHECK(synth::qwen3tts::kAssistantRolePrefixTokens == 3);
    SYNTH_TEST_CHECK(synth::qwen3tts::kAssistantSuffixTokens == 5);
    return 0;
}

// ---------------------------------------------------------------------------
// The reference transcript's own turn.
//
// Transcript-assisted cloning wraps the *reference transcript* in a second,
// shorter turn than the one every request gets, and slices the tokenized result
// at different counts. Upstream:
//
//     _build_ref_text        <|im_start|>assistant\n{text}<|im_end|>\n
//                            qwen3_tts_model.py:272-273, tokenized at :598
//     _build_assistant_text  <|im_start|>assistant\n{text}<|im_end|>\n
//                                                     <|im_start|>assistant\n
//                            qwen3_tts_model.py:269-270
//     the two slices         ref_id[:, 3:-2] and text_id[:, 3:-5], both at
//                            modeling_qwen3_tts.py:2190-2191
//
// Getting this wrong produces no error anywhere. A prompt off by one token
// still synthesizes speech, in the wrong voice or the wrong language
// (arch/qwen3-tts/talker-host.h).
// ---------------------------------------------------------------------------

// The byte-level alphabet again: 0x0A is U+010A, the way 0x20 is U+0120 above.
constexpr const char * kNewline = "\xc4\x8a";

// A vocabulary with the two properties the reference turn's arithmetic rests
// on, and nothing else: the role word folds into ONE token, and a newline pair
// folds into one token too. Both hold in the real package --
// models/qwen3-tts-12hz-0-6b-base spells "assistant" as the single id 77091,
// "\n" as 198 and "\n\n" as the single id 271 -- and the second is what makes
// wrapping observable at all.
//
// A vocabulary entry is unreachable without merges that build it, so the role
// word is spelled out as the chain that reaches it, the way a real vocabulary
// reaches it.
synth::qwen3tts::BpeFrontendConfig reference_turn_vocabulary() {
    synth::qwen3tts::BpeFrontendConfig config;
    config.provider_id      = "synthesize.qwen_bpe";
    config.contract_version = 1;
    config.vocab            = {
        "a",                               // 0
        "b",                               // 1
        "ab",                              // 2
        "s",                               // 3
        "i",                               // 4
        "t",                               // 5
        "n",                               // 6
        "as",                              // 7
        "ass",                             // 8
        "assi",                            // 9
        "assis",                           // 10
        "assist",                          // 11
        "assista",                         // 12
        "assistan",                        // 13
        "assistant",                       // 14
        kNewline,                          // 15  "\n"
        std::string(kNewline) + kNewline,  // 16  "\n\n"
        kSpace,                            // 17  " "
    };
    // Ranked most preferred first; the role-word chain has to run in order.
    config.merges         = { "a s",      "as s",      "ass i",      "assi s", "assis t",
                              "assist a", "assista n", "assistan t", "a b",    std::string(kNewline) + " " + kNewline };
    config.special_tokens = {
        { "<|im_start|>", 100 },
        { "<|im_end|>",   101 }
    };
    // No turn of its own: qwen_reference_transcript_ids applies the reference
    // turn itself, and a frontend carrying the assistant turn would wrap the
    // result a second time. This is what Model holds beside the wrapped one.
    config.prefix = "";
    config.suffix = "";
    return config;
}

// A frontend that hands back fewer tokens than the turn's own markers. Nothing
// this family loads can do that -- see the proof in bpe.cpp -- so this is how
// the bounds branch is reached at all. Without that branch, this case aborts:
// the slice's two iterators cross and assign() raises length_error, measured in
// both the Release and the sanitizer tree.
class ShortFrontend : public synth::TextFrontend {
  public:
    synth_input_flags_t input_flags() const noexcept override { return SYNTH_INPUT_SUPPORT_TEXT_UTF8; }

    synth_status_t prepare(synth_input_kind_t,
                           const void *,
                           uint64_t,
                           uint64_t,
                           std::vector<int32_t> & output) const override {
        output = { 100, 14, 15, 101 };  // the turn's markers, and no transcript
        return SYNTH_OK;
    }
};

int check_reference_turn() {
    // 1. The reference turn's exact string, which stops at the closing markers.
    //
    //    This is the only place the turn's ROLE WORD is checked, and it has to
    //    be: measured on the real vocabulary, wrapping in "<|im_start|>user\n"
    //    instead gives the identical sliced ids (prefix 151644, 872, 198 rather
    //    than 151644, 77091, 198 -- three tokens either way, all three sliced
    //    off). No id comparison anywhere, oracle included, can see a wrong role
    //    word. A bare `!= qwen_assistant_turn("hi")` used to sit here and was
    //    removed: with both turns pinned to exact strings it could never fail
    //    on its own.
    SYNTH_TEST_CHECK(synth::qwen3tts::qwen_reference_turn("hi") == "<|im_start|>assistant\nhi<|im_end|>\n");
    // The assistant turn is the reference turn plus a second opening; that is
    // the whole difference, and it is what the two suffix counts price. Derived
    // from the two exact strings rather than independent of them, and kept for
    // the same reason as the 3 + 2 arithmetic below: it names the relationship
    // the constants encode.
    SYNTH_TEST_CHECK(synth::qwen3tts::qwen_assistant_turn("hi") ==
                     synth::qwen3tts::qwen_reference_turn("hi") + "<|im_start|>assistant\n");

    // 2. The reference turn's own suffix count. The contrast with the assistant
    //    turn's 5 is what matters, but it is not re-asserted here: check_
    //    assistant_turn already pins that 5, and a `kReferenceSuffixTokens !=
    //    kAssistantSuffixTokens` beside these two exact values could only fail
    //    when one of them already had.
    SYNTH_TEST_CHECK(synth::qwen3tts::kReferenceSuffixTokens == 2);
    return 0;
}

int check_reference_transcript_ids() {
    std::unique_ptr<synth::TextFrontend> frontend;
    SYNTH_TEST_CHECK(synth::qwen3tts::make_bpe_frontend(reference_turn_vocabulary(), frontend) == SYNTH_OK);

    std::vector<int32_t> ids;
    const auto           reference = [&](const std::string & transcript, uint64_t max_tokens = 0) {
        return synth::qwen3tts::qwen_reference_transcript_ids(*frontend, transcript, max_tokens, ids);
    };
    // What the same frontend makes of the bare transcript, with no turn at all.
    std::vector<int32_t> bare;
    const auto           tokenize_bare = [&](const std::string & transcript) {
        SYNTH_TEST_CHECK(frontend->prepare(SYNTH_INPUT_TEXT_UTF8, transcript.data(), transcript.size(), 0, bare) ==
                         SYNTH_OK);
        return 0;
    };

    // 3. Round trip. The wrapped turn is [im_start, "assistant", "\n", <the
    //    transcript's own ids>, im_end, "\n"], and slicing 3 and 2 off it
    //    recovers exactly the transcript's contribution -- which is the count
    //    Task 7's T1 arithmetic adds up.
    SYNTH_TEST_CHECK(reference("ab") == SYNTH_OK);
    SYNTH_TEST_CHECK(ids == std::vector<int32_t>({ 2 }));
    SYNTH_TEST_CHECK(tokenize_bare("ab") == 0);
    SYNTH_TEST_CHECK(ids == bare);
    // The markers cost exactly the two constants, measured rather than assumed:
    // the whole turn is the transcript's ids plus 3 + 2.
    std::vector<int32_t> whole;
    const std::string    turn = synth::qwen3tts::qwen_reference_turn("ab");
    SYNTH_TEST_CHECK(frontend->prepare(SYNTH_INPUT_TEXT_UTF8, turn.data(), turn.size(), 0, whole) == SYNTH_OK);
    SYNTH_TEST_CHECK(whole.size() == ids.size() + synth::qwen3tts::kAssistantRolePrefixTokens +
                                         synth::qwen3tts::kReferenceSuffixTokens);

    // 4. The boundary case the wrap-then-slice idiom exists for, on a named
    //    input: "\nab". The pre-tokenizer's newline branch is greedy, so the
    //    transcript's leading newline joins the turn's own newline into one
    //    piece, that piece is one token, and the slice takes it away with the
    //    role prefix. Wrapping therefore does not merely add tokens around the
    //    transcript's -- it changes them.
    //
    //    The same input on the real package's vocabulary: "\nHello" through
    //    this path is the single id 9707, while tokenizing "\nHello" on its own
    //    is 198, 9707 (measured against models/qwen3-tts-12hz-0-6b-base with
    //    the pinned transformers 4.57.3). The integration test asserts that one
    //    against the real vocabulary; this one needs no package.
    //
    //    A LEADING NEWLINE IS THE ONLY INPUT THAT SHOWS THIS -- a leading
    //    space, tab, U+00A0, U+3000, an accented letter, and even a leading
    //    CRLF all tokenize identically wrapped and bare. The measured sweep is
    //    in tests/qwen3_tts_reference_transcript_real.cpp; swapping "\nab" here
    //    for any of them leaves an assertion that cannot fail.
    SYNTH_TEST_CHECK(reference("\nab") == SYNTH_OK);
    SYNTH_TEST_CHECK(tokenize_bare("\nab") == 0);
    SYNTH_TEST_CHECK(bare == std::vector<int32_t>({ 15, 2 }));
    SYNTH_TEST_CHECK(ids != bare);
    // And the sharper statement of the same fact: two different transcripts
    // reach the talker as the same ids. Tokenizing the bare transcript would
    // keep them apart, which is exactly why it is not equivalent.
    SYNTH_TEST_CHECK(ids == std::vector<int32_t>({ 2 }));

    // 5. An empty or whitespace-only transcript names no speech -- the design's
    //    §9 error table. " " is the case that needs the rule: it tokenizes
    //    perfectly well, to the one id below, so nothing downstream refuses it.
    SYNTH_TEST_CHECK(reference("") == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(ids.empty());
    SYNTH_TEST_CHECK(reference(" ") == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(reference("  \t\n ") == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(tokenize_bare(" ") == 0);
    SYNTH_TEST_CHECK(bare == std::vector<int32_t>({ 17 }));

    // The declared limit counts the turn's markers, the way tokenize_request's
    // does: a one-token transcript is a six-token turn, so a limit of six
    // passes and five does not.
    SYNTH_TEST_CHECK(reference("ab", 6) == SYNTH_OK);
    SYNTH_TEST_CHECK(reference("ab", 5) == SYNTH_ERR_INPUT_TOO_LONG);
    SYNTH_TEST_CHECK(ids.empty());

    // A byte the vocabulary cannot name is refused rather than dropped, the
    // same rule the request path follows.
    SYNTH_TEST_CHECK(reference("z") == SYNTH_ERR_INVALID_ARG);

    // The precondition on `frontend`, which no package can violate but a
    // future caller could.
    const ShortFrontend too_short;
    SYNTH_TEST_CHECK(synth::qwen3tts::qwen_reference_transcript_ids(too_short, "ab", 0, ids) ==
                     SYNTH_ERR_TEXT_FRONTEND);
    SYNTH_TEST_CHECK(ids.empty());
    return 0;
}

}  // namespace

// The merge order, and that a long run terminates in reasonable time.
//
// The rescan this replaced was quadratic in the symbols of one piece, and four of
// the pre-tokenizer's branches return an unbounded piece: measured on the real
// vocabulary, 32 KB of full stops took 27 seconds and 64 KB took 107, while
// tokenizing to 520 tokens against a declared limit of 1024. Text length is a
// post-conversion check, so nothing could refuse the request first.
//
// Two properties of the old loop were implicit and are easy to lose in a rewrite.
// Equivalence on the real vocabulary was established separately, by comparing
// synthesized audio across eight inputs before and after; what a synthetic
// vocabulary can pin is the ordering rule itself.
int check_merge_order() {
    synth::qwen3tts::BpeFrontendConfig config;
    config.provider_id      = "synthesize.qwen_bpe";
    config.contract_version = 1;
    // "xx" doubles: x -> xx -> xxxx, which is the shape that makes a long run of
    // one character merge at all.
    config.vocab            = {
        "x", "xx", "xxxx", "xxxxxxxx", "a", "b", "c", "ab", "bc", "abc", "\n", "<|im_start|>", "<|im_end|>"
    };
    // Ranked most preferred first. "a b" and "b c" share no rank; "ab c" is last.
    config.merges         = { "x x", "xx xx", "xxxx xxxx", "a b", "b c", "ab c" };
    config.special_tokens = {
        { "<|im_start|>", 11 },
        { "<|im_end|>",   12 }
    };
    config.prefix = "";
    config.suffix = "";

    std::unique_ptr<synth::TextFrontend> frontend;
    SYNTH_TEST_CHECK(synth::qwen3tts::make_bpe_frontend(config, frontend) == SYNTH_OK);

    // Lowest rank first, not left to right: "a b" outranks "b c", so "abc" comes
    // from merging "ab" and then "c" rather than "a" and "bc".
    {
        std::vector<int32_t> ids;
        SYNTH_TEST_CHECK(frontend->prepare(SYNTH_INPUT_TEXT_UTF8, "abc", 3, 0, ids) == SYNTH_OK);
        SYNTH_TEST_CHECK(ids.size() == 1 && ids[0] == 9);  // "abc"
    }

    const std::string run16(16, 'x');
    const std::string run11(11, 'x');
    const std::string run65536(65536, 'x');

    // A run of sixteen collapses through the doubling merges to two symbols of
    // eight, because no merge names sixteen. This is the case the old loop paid
    // fifteen full rescans for.
    {
        std::vector<int32_t> ids;
        SYNTH_TEST_CHECK(frontend->prepare(SYNTH_INPUT_TEXT_UTF8, run16.data(), run16.size(), 0, ids) == SYNTH_OK);
        SYNTH_TEST_CHECK(ids.size() == 2 && ids[0] == 3 && ids[1] == 3);  // "xxxxxxxx" twice
    }

    // An odd length leaves the remainder as smaller symbols rather than dropping
    // it: eleven is eight, two and one.
    {
        std::vector<int32_t> ids;
        SYNTH_TEST_CHECK(frontend->prepare(SYNTH_INPUT_TEXT_UTF8, run11.data(), run11.size(), 0, ids) == SYNTH_OK);
        SYNTH_TEST_CHECK(ids.size() == 3 && ids[0] == 3 && ids[1] == 1 && ids[2] == 0);
    }

    // The load-bearing case: a long unbounded piece must finish. At 64 KB the old
    // loop took about a hundred seconds; anything in that region fails the test's
    // own timeout rather than passing slowly.
    {
        std::vector<int32_t> ids;
        SYNTH_TEST_CHECK(frontend->prepare(SYNTH_INPUT_TEXT_UTF8, run65536.data(), run65536.size(), 0, ids) ==
                         SYNTH_OK);
        SYNTH_TEST_CHECK(ids.size() == 65536 / 8);
        for (const int32_t id : ids) {
            SYNTH_TEST_CHECK(id == 3);
        }
    }
    return 0;
}

int main() {
    SYNTH_TEST_CHECK(check_pretokenize() == 0);
    SYNTH_TEST_CHECK(check_frontend() == 0);
    SYNTH_TEST_CHECK(check_merge_order() == 0);
    SYNTH_TEST_CHECK(check_configuration_rejections() == 0);
    SYNTH_TEST_CHECK(check_turn_wrapping() == 0);
    SYNTH_TEST_CHECK(check_assistant_turn() == 0);
    SYNTH_TEST_CHECK(check_reference_turn() == 0);
    SYNTH_TEST_CHECK(check_reference_transcript_ids() == 0);
    return 0;
}
