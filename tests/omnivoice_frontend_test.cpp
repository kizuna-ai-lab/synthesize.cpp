// The OmniVoice frontend host: prompt text construction and the duration
// estimator, against the reference's own values.
//
// Nothing here can fail loudly at run time. A wrong cleanup rule, a wrong
// character weight or a wrong truncation produces a prompt the model still
// accepts and a canvas the decoder still fills -- different, still-plausible
// speech, with no error anywhere. So the expectations are transcribed rather
// than derived, and they are asserted at exact equality: doubles included,
// because the estimate feeds an integer frame count whose boundary is one ulp
// wide.
//
// Provenance of every transcribed value:
//
//   * duration rows, prompt strings and the anchor pair come from
//     scripts/dump_reference_omnivoice_tokenizer.py run against the pinned
//     package (source revision 468e927ba3716cd8dd86421148dfb3046e9f9d7b,
//     omnivoice 0.2.1) and the pinned weights revision
//     c5fdb5ccb189668d56333f77ba2629f4cd7535f4, written to
//     build/goldens/omnivoice/tokenizer/cases.json. That file is not committed
//     -- these literals are the committed statement of it.
//   * the character weights and the classification order come from
//     omnivoice/utils/duration.py at the same source revision.
//   * the combine_text cases were run through that revision's `_combine_text`
//     to obtain the expected strings; the two clone cases are additionally the
//     `combined_text` field of the corresponding cases.json rows.

#include "arch/omnivoice/frontend-host.h"
#include "bpe-frontend.h"
#include "test-assert.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

// --------------------------------------------------------------------------
// Shared text constants. The reference's anchor pair is contract, not a
// default: a request with no reference audio is estimated against it.
// --------------------------------------------------------------------------

constexpr const char * kAnchorText = "Nice to meet you.";

constexpr const char * kCloneRefText =
    "Some call me nature. Others call me Mother Nature. I've been here for over four point and "
    "five billion years, twenty-two thousand five hundred times longer than you.";

constexpr const char * kMediumEnText =
    "The engine reads a whole paragraph without pausing for breath. It holds one voice from the "
    "first word to the last. Nothing in the timing depends on the machine that runs it. It sounds "
    "the same everywhere.";

constexpr const char * kLongBoundaryText =
    "A golden case near the frame limit earns its place, because the cost of running it is paid on "
    "every backend and in every quantization profile this family ships. It sits here to prove that "
    "the canvas length the estimator computes on the host lands inside the ceiling the package "
    "declares, and that nothing in the decode loop degrades as the sequence grows longer. The "
    "shorter cases cannot show that, because a defect that accumulates with length has no room to "
    "appear in them.";

// --------------------------------------------------------------------------
// combine_text
// --------------------------------------------------------------------------

struct CombineCase {
    const char * ref;
    const char * text;
    const char * expected;
};

// The four cleanup rules run in upstream's order over the joined string, which
// is why the CJK-adjacency rule sees spaces the collapse rule already merged.
constexpr CombineCase kCombineCases[] = {
    // A reference joins with exactly one space, whatever padding either side had.
    { kAnchorText,            "OmniVoice speaks with one voice.",             "Nice to meet you. OmniVoice speaks with one voice." },
    { "  Nice to meet you. ", "  OmniVoice speaks with one voice.  ",
     "Nice to meet you. OmniVoice speaks with one voice."                                                                          },
    // No reference: the target alone, stripped.
    { "",                     "This is a sentence without any voice prompt.", "This is a sentence without any voice prompt."       },
    { "",                     "  padded text  ",                              "padded text"                                        },
    // Newlines are deleted, not turned into spaces -- the words run together.
    { "",                     "line one\r\nline two\nline three",             "line oneline twoline three"                         },
    // Full-width parentheses become ASCII ones.
    { "",
     "\xef\xbc\x88"
      "aside"
      "\xef\xbc\x89"
      " and "
      "\xef\xbc\x88"
      "more"
      "\xef\xbc\x89",                                                         "(aside) and (more)"                                 },
    // Runs of spaces and tabs collapse to one space.
    { "",                     "a\t\tb   c \t d",                              "a b c d"                                            },
    // Whitespace touching a CJK ideograph is removed on either side.
    { "",                     "你好 世界",                                    "你好世界"                                           },
    { "",                     "hello 你好 world",
     "hello你好"
      "world"                                                                                                                      },
    { "你好",                 "世界",                                         "你好世界"                                           },
    // The rule reads the whole whitespace class, not just U+0020: an ideographic
    // space next to an ideograph goes too.
    { "",                     "tail space before cjk 　你",                   "tail space before cjk你"                            },
    // Kana is outside U+4E00-U+9FFF, so the space between two kana survives.
    { "",                     "あ い",                                        "あ い"                                              },
    // The two clone cases, whose expected strings are also the `combined_text`
    // field of the cases.json rows of the same name.
    { kCloneRefText,          "OmniVoice speaks with one voice.",
     "Some call me nature. Others call me Mother Nature. I've been here for over four point and "
      "five billion years, twenty-two thousand five hundred times longer than you. OmniVoice "
      "speaks with one voice."                                                                                                     },
    { kCloneRefText,          "克隆的声音也要说中文的句子。",
     "Some call me nature. Others call me Mother Nature. I've been here for over four point and "
      "five billion years, twenty-two thousand five hundred times longer than you."
      "克隆的声音也要说中文的句子。"                                                                                               },
};

int check_combine_text() {
    for (const CombineCase & item : kCombineCases) {
        SYNTH_TEST_CHECK(synth::omnivoice::combine_text(item.ref, item.text) == item.expected);
    }
    // Both sides empty is an empty prompt, not a stray space.
    SYNTH_TEST_CHECK(synth::omnivoice::combine_text("", "").empty());
    // A reference that is only whitespace is still a reference: upstream tests
    // the raw string for truthiness before stripping it, so the join happens
    // and the leading space is then collapsed away by the strip of the whole.
    SYNTH_TEST_CHECK(synth::omnivoice::combine_text(" ", "text") == " text");
    return 0;
}

// --------------------------------------------------------------------------
// add_punctuation
// --------------------------------------------------------------------------

// Every expected string below was cross-checked against the pinned package
// (scripts/envs/omnivoice/.venv, omnivoice/utils/text.py at source revision
// 468e927ba3716cd8dd86421148dfb3046e9f9d7b): `add_punctuation` run directly
// against each `transcript` reproduces `expected` byte for byte.
int check_add_punctuation() {
    // No trailing punctuation: English gets ".", Chinese gets "。" -- and the
    // Chinese test scans the WHOLE string, not just its last character.
    SYNTH_TEST_CHECK(synth::omnivoice::add_punctuation("Hello world") == "Hello world.");
    SYNTH_TEST_CHECK(synth::omnivoice::add_punctuation("\xe6\xac\xa2\xe8\xbf\x8e\xe4\xbd\xbf\xe7\x94\xa8") ==
                     "\xe6\xac\xa2\xe8\xbf\x8e\xe4\xbd\xbf\xe7\x94\xa8\xe3\x80\x82");
    // Mixed script: one CJK ideograph anywhere in the (stripped) text is
    // enough to select the Chinese mark, even with Latin text and a Latin
    // trailing character.
    SYNTH_TEST_CHECK(synth::omnivoice::add_punctuation("hello \xe4\xbd\xa0\xe5\xa5\xbd") ==
                     "hello \xe4\xbd\xa0\xe5\xa5\xbd\xe3\x80\x82");

    // Already terminated: every END_PUNCTUATION member (text.py:38-65) is a
    // no-op. Each row is `text[-1]` unchanged from what `strip()` leaves it,
    // whether that codepoint is one or several bytes.
    struct AlreadyTerminated {
        const char * text;
    };

    constexpr AlreadyTerminated kAlreadyTerminated[] = {
        { "hello;" },
        { "hello:" },
        { "hello," },
        { "hello." },
        { "hello!" },
        { "hello?" },
        // … U+2026
        { "hello\xe2\x80\xa6" },
        { "hello)" },
        { "hello]" },
        { "hello}" },
        { "hello\"" },
        { "hello'" },
        // “ ” ‘ ’
        { "hello\xe2\x80\x9c" },
        { "hello\xe2\x80\x9d" },
        { "hello\xe2\x80\x98" },
        { "hello\xe2\x80\x99" },
        // ； ： ， 。 ！ ？ 、 ） 】 (fullwidth / CJK punctuation)
        { "hello\xef\xbc\x9b" },
        { "hello\xef\xbc\x9a" },
        { "hello\xef\xbc\x8c" },
        { "hello\xe3\x80\x82" },
        { "hello\xef\xbc\x81" },
        { "hello\xef\xbc\x9f" },
        { "hello\xe3\x80\x81" },
        { "hello\xef\xbc\x89" },
        { "hello\xe3\x80\x91" },
        // The set's one two-character member, "……" (a doubled U+2026):
        // unreachable through `text[-1]` on its own two-character identity,
        // but its trailing codepoint IS the lone "…" member above, so a text
        // ending in it is unchanged for that reason.
        { "hello\xe2\x80\xa6\xe2\x80\xa6" },
    };
    for (const AlreadyTerminated & item : kAlreadyTerminated) {
        SYNTH_TEST_CHECK(synth::omnivoice::add_punctuation(item.text) == item.text);
    }

    // Empty, and whitespace-only (which strips to empty): upstream's own
    // `if not text: return text` fires AFTER the strip, so both come back
    // empty rather than gaining a mark of their own.
    SYNTH_TEST_CHECK(synth::omnivoice::add_punctuation("").empty());
    SYNTH_TEST_CHECK(synth::omnivoice::add_punctuation("   ").empty());

    // Leading/trailing whitespace is stripped before the check and before the
    // mark is appended -- the mark lands immediately after the trimmed text.
    SYNTH_TEST_CHECK(synth::omnivoice::add_punctuation("  trailing space  ") == "trailing space.");

    return 0;
}

// --------------------------------------------------------------------------
// style_text
// --------------------------------------------------------------------------

int check_style_text() {
    // `denoise` is already the reference's `denoise and ref_audio_tokens is not
    // None`: the marker appears only for a clone request.
    SYNTH_TEST_CHECK(synth::omnivoice::style_text(false, "en", "") ==
                     "<|lang_start|>en<|lang_end|><|instruct_start|>None<|instruct_end|>");
    SYNTH_TEST_CHECK(synth::omnivoice::style_text(true, "", "female, young") ==
                     "<|denoise|><|lang_start|>None<|lang_end|><|instruct_start|>female, "
                     "young<|instruct_end|>");
    // Both slots empty is the literal "None" twice, which is what the
    // language-agnostic path sends.
    SYNTH_TEST_CHECK(synth::omnivoice::style_text(false, "", "") ==
                     "<|lang_start|>None<|lang_end|><|instruct_start|>None<|instruct_end|>");
    // A clone request that also names both slots keeps all three parts, in
    // upstream's order.
    SYNTH_TEST_CHECK(synth::omnivoice::style_text(true, "zh", "\xe7\x94\xb7\xef\xbc\x8c\xe8\x80\x81\xe5\xb9\xb4") ==
                     "<|denoise|><|lang_start|>zh<|lang_end|><|instruct_start|>"
                     "\xe7\x94\xb7\xef\xbc\x8c\xe8\x80\x81\xe5\xb9\xb4<|instruct_end|>");
    return 0;
}

// --------------------------------------------------------------------------
// tokenize_wrapped_text
// --------------------------------------------------------------------------

// A vocabulary small enough to write out, in the byte-level alphabet: printable
// ASCII maps to itself and a space is U+0120, which is what makes " a" spell
// "\xc4\xa0a" rather than " a".
constexpr const char * kSpace = "\xc4\xa0";

// The fixture reproduces the one merge the oracle shows the split defeating.
// In cases.json's `omni-nonverbal` row the whole string tokenizes " [" as one
// piece -- id 508 there, id 14 here -- while the tag-aware path emits the space
// and the bracket separately. `tag_split_changes_ids` is true for exactly that
// row and false for every other one, and this is the same difference in
// miniature. The tags are deliberately absent from the vocabulary as single
// tokens: they are plain text, which is why they need splitting at all.
//
// The letters are individual entries with no merges among them, so a tag
// tokenizes to one id per character. That keeps the expectations readable while
// leaving the only interesting merge -- the one that crosses the tag boundary --
// in place.
int check_nonverbal_split() {
    synth::BpeFrontendConfig config;
    config.provider_id      = "synthesize.qwen_bpe";
    config.contract_version = 1;
    config.vocab            = { "a",
                                "b",
                                "e",
                                "g",
                                "h",
                                "i",
                                "l",
                                "r",
                                "s",
                                "t",
                                "u",
                                "[",
                                "]",
                                kSpace,
                                std::string(kSpace) + "[",
                                std::string(kSpace) + "a",
                                "<|text_start|>",
                                "<|text_end|>" };
    config.merges           = { std::string(kSpace) + " [", std::string(kSpace) + " a" };
    config.special_tokens   = {
        { "<|text_start|>", 16 },
        { "<|text_end|>",   17 }
    };

    std::unique_ptr<synth::TextFrontend> frontend;
    SYNTH_TEST_CHECK(synth::make_bpe_frontend(config, frontend) == SYNTH_OK);
    SYNTH_TEST_CHECK(frontend != nullptr);

    const auto plain = [&](const std::string & text, std::vector<int32_t> & ids) {
        return frontend->prepare(SYNTH_INPUT_TEXT_UTF8, text.data(), text.size(), 0, ids);
    };

    // "laughter" and "sigh", one id per letter.
    const std::vector<int32_t> laughter = { 6, 0, 10, 3, 4, 9, 2, 7 };

    // Without the split, " [" merges across the tag boundary into id 14.
    const std::string    tagged = "<|text_start|>a [laughter] a<|text_end|>";
    std::vector<int32_t> whole;
    SYNTH_TEST_CHECK(plain(tagged, whole) == SYNTH_OK);
    SYNTH_TEST_CHECK((whole == std::vector<int32_t>{ 16, 0, 14, 6, 0, 10, 3, 4, 9, 2, 7, 12, 15, 17 }));

    // With it, the tag is tokenized standalone -- the space stays with the
    // segment before it (id 13) and the bracket opens the tag (id 11) -- and it
    // sits between the neighbours' own tokenizations.
    std::vector<int32_t> split;
    SYNTH_TEST_CHECK(synth::omnivoice::tokenize_wrapped_text(*frontend, tagged, split) == SYNTH_OK);
    SYNTH_TEST_CHECK((split == std::vector<int32_t>{ 16, 0, 13, 11, 6, 0, 10, 3, 4, 9, 2, 7, 12, 15, 17 }));
    SYNTH_TEST_CHECK(split != whole);

    // The tag's ids are exactly what the tag alone tokenizes to, which is the
    // property the reference splits for: the same ids whatever surrounds them.
    std::vector<int32_t> standalone;
    SYNTH_TEST_CHECK(plain("[laughter]", standalone) == SYNTH_OK);
    std::vector<int32_t> expected_tag = { 11 };
    expected_tag.insert(expected_tag.end(), laughter.begin(), laughter.end());
    expected_tag.push_back(12);
    SYNTH_TEST_CHECK(standalone == expected_tag);
    SYNTH_TEST_CHECK(std::vector<int32_t>(split.begin() + 3, split.begin() + 13) == expected_tag);

    // Text with no tag takes the same path as a plain prepare, byte for byte.
    const std::string    untagged = "<|text_start|>a a<|text_end|>";
    std::vector<int32_t> reference;
    std::vector<int32_t> through;
    SYNTH_TEST_CHECK(plain(untagged, reference) == SYNTH_OK);
    SYNTH_TEST_CHECK(synth::omnivoice::tokenize_wrapped_text(*frontend, untagged, through) == SYNTH_OK);
    SYNTH_TEST_CHECK(through == reference);
    SYNTH_TEST_CHECK((through == std::vector<int32_t>{ 16, 0, 15, 17 }));

    // A bracketed word that is not one of the thirteen is ordinary text.
    const std::string    unknown = "<|text_start|>a [b] a<|text_end|>";
    std::vector<int32_t> unknown_plain;
    std::vector<int32_t> unknown_split;
    SYNTH_TEST_CHECK(plain(unknown, unknown_plain) == SYNTH_OK);
    SYNTH_TEST_CHECK(synth::omnivoice::tokenize_wrapped_text(*frontend, unknown, unknown_split) == SYNTH_OK);
    SYNTH_TEST_CHECK(unknown_split == unknown_plain);

    // Two tags back to back, with nothing before or after either.
    std::vector<int32_t> doubled;
    SYNTH_TEST_CHECK(synth::omnivoice::tokenize_wrapped_text(*frontend, "[laughter][sigh]", doubled) == SYNTH_OK);
    SYNTH_TEST_CHECK((doubled == std::vector<int32_t>{ 11, 6, 0, 10, 3, 4, 9, 2, 7, 12, 11, 8, 5, 3, 4, 12 }));

    // A byte the vocabulary cannot name is still a package defect, and the
    // tag-aware path must not swallow the failure -- on either side of the tag,
    // and on the no-tag path, which is the one most requests take.
    //
    // Each call is seeded with junk first, so an empty result proves the vector
    // was cleared rather than merely never written to.
    const auto rejects = [&](const std::string & text) {
        std::vector<int32_t> ids = { 999, 998 };
        return synth::omnivoice::tokenize_wrapped_text(*frontend, text, ids) != SYNTH_OK && ids.empty();
    };
    SYNTH_TEST_CHECK(rejects("z [laughter]"));
    SYNTH_TEST_CHECK(rejects("[laughter] z"));
    // No tag at all, failing partway through the one span. The frontend emits
    // "a" and the space before it reaches the byte it cannot name, so this is
    // the case where a status returned straight through would carry two ids out
    // with it.
    SYNTH_TEST_CHECK(rejects("a z"));
    SYNTH_TEST_CHECK(rejects("<|text_start|>a z<|text_end|>"));

    // The guarantee is this function's, not one it inherits: a plain prepare
    // over the same input does leave its partial output behind. If that ever
    // changes the clear above becomes redundant rather than wrong, and this
    // line is what will say so.
    std::vector<int32_t> partial;
    SYNTH_TEST_CHECK(plain("a z", partial) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK((partial == std::vector<int32_t>{ 0, 13 }));

    // The exact status is passed through unchanged.
    std::vector<int32_t> rejected;
    SYNTH_TEST_CHECK(synth::omnivoice::tokenize_wrapped_text(*frontend, "a z", rejected) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(rejected.empty());

    // Empty input is an empty sequence, not a failure.
    std::vector<int32_t> empty;
    SYNTH_TEST_CHECK(synth::omnivoice::tokenize_wrapped_text(*frontend, "", empty) == SYNTH_OK);
    SYNTH_TEST_CHECK(empty.empty());
    return 0;
}

// --------------------------------------------------------------------------
// assemble_prompt_ids
// --------------------------------------------------------------------------

// Composition, not tokenization: check_nonverbal_split already proves the
// frontend tokenizes a wrapped string correctly, so this fixture exists only
// to let style_text/combine_text/tokenize_wrapped_text run for real, each
// character mapping to exactly one id and no merge rule to muddy which ids
// the hand-built expectation and the call under test share.
int check_assemble_prompt_ids() {
    synth::BpeFrontendConfig config;
    config.provider_id      = "synthesize.qwen_bpe";
    config.contract_version = 1;
    config.vocab            = { "H", "i", ".", "N", "o", "n", "e", "S", "m", "c", "a", "l", "t", "u", "r", kSpace };
    config.special_tokens   = {
        { "<|denoise|>",        100 },
        { "<|lang_start|>",     101 },
        { "<|lang_end|>",       102 },
        { "<|instruct_start|>", 103 },
        { "<|instruct_end|>",   104 },
        { "<|text_start|>",     105 },
        { "<|text_end|>",       106 },
    };
    std::unique_ptr<synth::TextFrontend> frontend;
    SYNTH_TEST_CHECK(synth::make_bpe_frontend(config, frontend) == SYNTH_OK);
    SYNTH_TEST_CHECK(frontend != nullptr);

    synth::omnivoice::SpecialTokens tokens;
    tokens.denoise        = 100;
    tokens.lang_start     = 101;
    tokens.lang_end       = 102;
    tokens.instruct_start = 103;
    tokens.instruct_end   = 104;
    tokens.text_start     = 105;
    tokens.text_end       = 106;

    // Plain request: style_text ids + text_start + tokenize("Hi.") + text_end,
    // built from the same three helpers this file already exercises on their
    // own, concatenated by hand rather than re-derived from scratch.
    std::vector<int32_t> expected;
    std::vector<int32_t> piece;
    SYNTH_TEST_CHECK(synth::omnivoice::tokenize_wrapped_text(*frontend, synth::omnivoice::style_text(false, "en", ""),
                                                             piece) == SYNTH_OK);
    expected = piece;
    SYNTH_TEST_CHECK(synth::omnivoice::tokenize_wrapped_text(*frontend, "<|text_start|>", piece) == SYNTH_OK);
    expected.insert(expected.end(), piece.begin(), piece.end());
    SYNTH_TEST_CHECK(synth::omnivoice::tokenize_wrapped_text(*frontend, synth::omnivoice::combine_text("", "Hi."),
                                                             piece) == SYNTH_OK);
    expected.insert(expected.end(), piece.begin(), piece.end());
    SYNTH_TEST_CHECK(synth::omnivoice::tokenize_wrapped_text(*frontend, "<|text_end|>", piece) == SYNTH_OK);
    expected.insert(expected.end(), piece.begin(), piece.end());

    std::vector<int32_t> actual;
    SYNTH_TEST_CHECK(synth::omnivoice::assemble_prompt_ids(*frontend, tokens, /*denoise=*/false, "en", "",
                                                           synth::omnivoice::combine_text("", "Hi."),
                                                           actual) == SYNTH_OK);
    SYNTH_TEST_CHECK(actual == expected);

    // Clone-shaped call: the denoise marker leads, and the combined
    // (space-joined) reference-plus-text follows the text_start marker.
    std::vector<int32_t> clone_expected;
    SYNTH_TEST_CHECK(synth::omnivoice::tokenize_wrapped_text(*frontend, synth::omnivoice::style_text(true, "en", ""),
                                                             piece) == SYNTH_OK);
    clone_expected = piece;
    SYNTH_TEST_CHECK(synth::omnivoice::tokenize_wrapped_text(*frontend, "<|text_start|>", piece) == SYNTH_OK);
    clone_expected.insert(clone_expected.end(), piece.begin(), piece.end());
    const std::string combined = synth::omnivoice::combine_text("Some call me nature.", "Hi.");
    SYNTH_TEST_CHECK(combined == "Some call me nature. Hi.");
    SYNTH_TEST_CHECK(synth::omnivoice::tokenize_wrapped_text(*frontend, combined, piece) == SYNTH_OK);
    clone_expected.insert(clone_expected.end(), piece.begin(), piece.end());
    SYNTH_TEST_CHECK(synth::omnivoice::tokenize_wrapped_text(*frontend, "<|text_end|>", piece) == SYNTH_OK);
    clone_expected.insert(clone_expected.end(), piece.begin(), piece.end());

    std::vector<int32_t> clone_actual;
    SYNTH_TEST_CHECK(synth::omnivoice::assemble_prompt_ids(*frontend, tokens, /*denoise=*/true, "en", "", combined,
                                                           clone_actual) == SYNTH_OK);
    SYNTH_TEST_CHECK(clone_actual == clone_expected);

    // The denoise marker is the composed string's very first token: style_text
    // places it before the language slot whenever `denoise` is set.
    std::vector<int32_t> denoise_marker;
    SYNTH_TEST_CHECK(synth::omnivoice::tokenize_wrapped_text(*frontend, "<|denoise|>", denoise_marker) == SYNTH_OK);
    SYNTH_TEST_CHECK(clone_actual.size() >= denoise_marker.size());
    SYNTH_TEST_CHECK(std::equal(denoise_marker.begin(), denoise_marker.end(), clone_actual.begin()));

    // Empty language and empty instruct both fall to the literal "None": the
    // resulting ids start with exactly what style_text("", "") tokenizes to,
    // rather than a hand-rederived guess at what "None" spells.
    std::vector<int32_t> none_style;
    SYNTH_TEST_CHECK(synth::omnivoice::tokenize_wrapped_text(*frontend, synth::omnivoice::style_text(false, "", ""),
                                                             none_style) == SYNTH_OK);
    std::vector<int32_t> none_actual;
    SYNTH_TEST_CHECK(synth::omnivoice::assemble_prompt_ids(*frontend, tokens, /*denoise=*/false, "", "",
                                                           synth::omnivoice::combine_text("", "Hi."),
                                                           none_actual) == SYNTH_OK);
    SYNTH_TEST_CHECK(none_actual.size() >= none_style.size());
    SYNTH_TEST_CHECK(std::equal(none_style.begin(), none_style.end(), none_actual.begin()));

    // Tokenizer failure (an out-of-vocabulary byte in the text) is one way
    // this function reports non-SYNTH_OK: the status is tokenize_wrapped_text's
    // own, propagated verbatim -- the BPE frontend's prepare() returns
    // SYNTH_ERR_INVALID_ARG for an unencodable span.
    std::vector<int32_t> rejected = { 999 };
    SYNTH_TEST_CHECK(synth::omnivoice::assemble_prompt_ids(*frontend, tokens, false, "en", "",
                                                           synth::omnivoice::combine_text("", "zzz not in vocab"),
                                                           rejected) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(rejected.empty());

    // The other way: a `tokens` whose text_end id disagrees with what the
    // FRONTEND actually emits for "<|text_end|>" (still 106 -- the frontend
    // fixture above is untouched). This is the release-path corruption guard
    // firing on a frontend/tokens mismatch, the failure mode a dropped or
    // renumbered closing marker would otherwise produce silently (see the
    // .cpp's comment on the composed string always closing on text_end).
    // Cheapest trigger available with the existing fixture: no real package
    // can misconfigure this -- SpecialTokens and the frontend's special
    // tokens are read from the same GGUF metadata at load time -- but a
    // regression that changed one without the other is exactly what this
    // guards against.
    synth::omnivoice::SpecialTokens mismatched_tokens = tokens;
    mismatched_tokens.text_end                        = 999;
    std::vector<int32_t> mismatched                   = { 111, 222 };
    SYNTH_TEST_CHECK(synth::omnivoice::assemble_prompt_ids(*frontend, mismatched_tokens, false, "en", "",
                                                           synth::omnivoice::combine_text("", "Hi."),
                                                           mismatched) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(mismatched.empty());
    return 0;
}

// --------------------------------------------------------------------------
// Character weights
// --------------------------------------------------------------------------

struct WeightCase {
    const char * utf8;
    double       weight;
    const char * why;
};

// Values from omnivoice/utils/duration.py's `_get_char_weight` at the pinned
// source revision, one per branch of its classification order.
constexpr WeightCase kWeightCases[] = {
    { "a",          1.0, "ASCII letter, tested before anything else"             },
    { "A",          1.0, "the upper half of the same test"                       },
    { " ",          0.2, "U+0020, tested by value"                               },
    { "ـ",          0.0, "Arabic tatweel, tested by value and weighed as a mark" },
    { "́",           0.0, "combining acute: general category Mn"                  },
    { "7",          3.5, "category Nd -- digits cost more than letters"          },
    { "!",          0.5, "category Po"                                           },
    { "。",         0.5, "ideographic full stop: category Po, not CJK"           },
    { "\U0001F600", 0.5, "emoji: category So"                                    },
    { " ",          0.2, "no-break space: category Zs"                           },
    { "　",         0.2, "ideographic space: category Zs"                        },
    { "你",         3.0, "CJK Unified Ideographs"                                },
    { "あ",         2.2, "Hiragana"                                              },
    { "한",         2.5, "Hangul Syllables"                                      },
    { "я",          1.0, "Cyrillic"                                              },
    { "ई",          1.8, "Devanagari"                                            },
    { "\U00021000", 3.0, "above U+20000 and past the range table: CJK"           },
    // The Ext-B boundary is exclusive upstream: U+20000 itself is `default`.
    { "\U00020000", 1.0, "U+20000 itself: not > 0x20000, so default"             },
    { "\U00020001", 3.0, "U+20001: one past the boundary, so CJK"                },
    // The two that look like whitespace and are not. Neither is category Z, so
    // both fall through to the range table's first entry and weigh as Latin.
    { "\t",         1.0, "U+0009 is category Cc, not Z"                          },
    { "\n",         1.0, "U+000A is category Cc, not Z"                          },
};

int check_char_weights() {
    const synth::omnivoice::DurationEstimator estimator;
    for (const WeightCase & item : kWeightCases) {
        SYNTH_TEST_CHECK(estimator.total_weight(item.utf8) == item.weight);
    }
    // The sum is over characters, not bytes: a three-byte ideograph counts once.
    SYNTH_TEST_CHECK(estimator.total_weight("你好") == 6.0);
    SYNTH_TEST_CHECK(estimator.total_weight("") == 0.0);
    // The anchor's weight is recorded in cases.json's duration_anchor block.
    SYNTH_TEST_CHECK(estimator.total_weight(kAnchorText) == 14.1);
    SYNTH_TEST_CHECK(estimator.total_weight(kCloneRefText) == 140.4);
    return 0;
}

// --------------------------------------------------------------------------
// Estimates against the oracle
// --------------------------------------------------------------------------

struct DurationCase {
    const char * name;
    const char * text;
    const char * ref_text;
    bool         ref_is_anchor;
    double       total_weight;
    double       ref_total_weight;
    uint64_t     ref_frames;
    float        speed;
    uint64_t     expected_frames;
};

// Every row of cases.json's `duration` array, in file order. Six of them land
// in the boost arm -- the estimate below the 50-frame threshold is pulled back
// up the cube-root curve -- and they are the reason the arm is not optional.
constexpr DurationCase kDurationCases[] = {
    { "omni-upstream-readme",        "This is a sentence without any voice prompt.",      kAnchorText,   true,  37.9,  14.1,  25,  1.0f,
     67                                                                                                                                      },
    { "omni-short-en",               "OmniVoice speaks with one voice.",                  kAnchorText,   true,  28.3,  14.1,  25,  1.0f, 50  },
    { "omni-short-zh",               "欢迎使用语音合成引擎。",                            kAnchorText,   true,  30.5,  14.1,  25,  1.0f, 54  },
    { "omni-short-ja",               "音声合成へようこそ。",                              kAnchorText,   true,  23.5,  14.1,  25,  1.0f, 47  },
    { "omni-lang-none",              "This request names no language.",                   kAnchorText,   true,  27.3,  14.1,  25,  1.0f, 49  },
    { "omni-punctuation",            "Wait — really?! Yes; truly: it works... (mostly).", kAnchorText,   true,  37.9,  14.1,  25,  1.0f,
     67                                                                                                                                      },
    { "omni-digits",                 "In 2026 the model counted 1234567890 samples.",     kAnchorText,   true,  74.7,  14.1,  25,  1.0f, 132 },
    { "omni-nonverbal",              "That is funny [laughter] but let me think.",        kAnchorText,   true,  34.9,  14.1,  25,  1.0f, 61  },
    { "omni-medium-en",              kMediumEnText,                                       kAnchorText,   true,  173.2, 14.1,  25,  1.0f, 307 },
    { "omni-long-boundary",          kLongBoundaryText,                                   kAnchorText,   true,  405.6, 14.1,  25,  1.0f, 719 },
    { "omni-rate-slow",              "OmniVoice speaks with one voice.",                  kAnchorText,   true,  28.3,  14.1,  25,  0.5f, 100 },
    { "omni-rate-slow",              "OmniVoice speaks with one voice.",                  kAnchorText,   true,  28.3,  14.1,  25,  1.0f, 50  },
    { "omni-rate-slow",              "OmniVoice speaks with one voice.",                  kAnchorText,   true,  28.3,  14.1,  25,  2.0f, 25  },
    { "omni-rate-fast",              "OmniVoice speaks with one voice.",                  kAnchorText,   true,  28.3,  14.1,  25,  0.5f, 100 },
    { "omni-rate-fast",              "OmniVoice speaks with one voice.",                  kAnchorText,   true,  28.3,  14.1,  25,  1.0f, 50  },
    { "omni-rate-fast",              "OmniVoice speaks with one voice.",                  kAnchorText,   true,  28.3,  14.1,  25,  2.0f, 25  },
    { "omni-design-en",              "OmniVoice speaks with one voice.",                  kAnchorText,   true,  28.3,  14.1,  25,  1.0f, 50  },
    { "omni-design-zh",              "语音设计决定说话人。",                              kAnchorText,   true,  27.5,  14.1,  25,  1.0f, 49  },
    { "omni-clone-en",               "OmniVoice speaks with one voice.",                  kCloneRefText, false, 28.3,  140.4, 351, 1.0f, 70  },
    { "omni-clone-zh",               "克隆的声音也要说中文的句子。",                      kCloneRefText, false, 39.5,  140.4, 351, 1.0f, 98  },
    { "omni-fast-mode",              "OmniVoice speaks with one voice.",                  kAnchorText,   true,  28.3,  14.1,  25,  1.0f, 50  },
    { "omni-sampled-seed-zero",      "Sampling follows the seed.",                        kAnchorText,   true,  23.1,  14.1,  25,  1.0f, 46  },
    { "omni-sampled-seed-one",       "Sampling follows the seed.",                        kAnchorText,   true,  23.1,  14.1,  25,  1.0f, 46  },
    { "omni-sampled-seed-forty-two", "Sampling follows the seed.",                        kAnchorText,   true,  23.1,  14.1,  25,  1.0f, 46  },
};

int check_estimates_match_oracle() {
    const synth::omnivoice::DurationEstimator estimator;

    size_t boosted = 0;
    for (const DurationCase & item : kDurationCases) {
        SYNTH_TEST_CHECK(estimator.total_weight(item.text) == item.total_weight);
        SYNTH_TEST_CHECK(estimator.total_weight(item.ref_text) == item.ref_total_weight);

        // The reference is always spelled out here, so the proportion is
        // checked without the anchor substitution in the way.
        SYNTH_TEST_CHECK(estimator.estimate_target_frames(item.text, item.ref_text, item.ref_frames, item.speed) ==
                         item.expected_frames);

        // Deliberate divergence, recorded in docs/porting/families/omnivoice.md:
        // upstream's estimator divides by zero reference tokens and its max(1, int(...))
        // returns 1; this port treats zero reference frames as "no reference" and
        // uses the anchor pair instead, because a reference with no audio behind it
        // is refused at load and can only mean a caller bug.
        //
        // A row the oracle marked as anchored must reach the same answer from
        // a request that carries no reference at all.
        if (item.ref_is_anchor) {
            SYNTH_TEST_CHECK(estimator.estimate_target_frames(item.text, "", 0, item.speed) == item.expected_frames);
        }

        const double raw = item.total_weight / (item.ref_total_weight / static_cast<double>(item.ref_frames));
        if (raw < 50.0) {
            ++boosted;
        }
    }
    // If a future edit drops the boost arm, most rows still pass. These six are
    // the ones that would not.
    SYNTH_TEST_CHECK(boosted == 6);

    return 0;
}

int check_boost_curve() {
    const synth::omnivoice::DurationEstimator estimator;

    // Full double precision, from Python's repr of the same computation at the
    // pinned source revision. The boosted value is
    // 50 * (est/50)^(1/3) with est = 41.66666666666667.
    SYNTH_TEST_CHECK(estimator.estimate_duration("音声合成へようこそ。", kAnchorText, 25.0) == 47.05180144405143);
    SYNTH_TEST_CHECK(estimator.estimate_duration("This request names no language.", kAnchorText, 25.0) ==
                     49.462323920353285);
    SYNTH_TEST_CHECK(estimator.estimate_duration("Sampling follows the seed.", kAnchorText, 25.0) == 46.78331171084225);
    SYNTH_TEST_CHECK(estimator.estimate_duration("语音设计决定说话人。", kAnchorText, 25.0) == 49.58281726838853);

    // Above the threshold the proportion is returned untouched.
    SYNTH_TEST_CHECK(estimator.estimate_duration("OmniVoice speaks with one voice.", kAnchorText, 25.0) ==
                     50.17730496453901);
    SYNTH_TEST_CHECK(estimator.estimate_duration("OmniVoice speaks with one voice.", kCloneRefText, 351.0) == 70.75);

    // The reference's own guards: no duration and no reference both yield zero
    // rather than a division.
    SYNTH_TEST_CHECK(estimator.estimate_duration("hello", "", 25.0) == 0.0);
    SYNTH_TEST_CHECK(estimator.estimate_duration("hello", kAnchorText, 0.0) == 0.0);
    SYNTH_TEST_CHECK(estimator.estimate_duration("hello", kAnchorText, -5.0) == 0.0);
    // A reference whose every character weighs nothing is the same case.
    SYNTH_TEST_CHECK(estimator.estimate_duration("hello", "́", 25.0) == 0.0);
    // An empty target is zero, and the boost curve leaves zero alone.
    SYNTH_TEST_CHECK(estimator.estimate_duration("", kAnchorText, 25.0) == 0.0);

    return 0;
}

int check_frame_truncation() {
    const synth::omnivoice::DurationEstimator estimator;

    // The result truncates toward zero rather than rounding: 50.177 is 50, and
    // halving the rate gives 100 rather than 101.
    SYNTH_TEST_CHECK(estimator.estimate_target_frames("OmniVoice speaks with one voice.", kAnchorText, 25, 1.0f) == 50);
    SYNTH_TEST_CHECK(estimator.estimate_target_frames("OmniVoice speaks with one voice.", kAnchorText, 25, 0.5f) ==
                     100);

    // The rate divides, and only when it is positive and not one -- upstream
    // leaves a zero rate alone rather than treating it as an error.
    SYNTH_TEST_CHECK(estimator.estimate_target_frames("OmniVoice speaks with one voice.", kAnchorText, 25, 0.0f) == 50);

    // The floor is one frame: an empty request still gets a canvas.
    SYNTH_TEST_CHECK(estimator.estimate_target_frames("", "", 0, 1.0f) == 1);
    SYNTH_TEST_CHECK(estimator.estimate_target_frames(".", "", 0, 20.0f) == 1);
    // A single letter is boosted well clear of the floor.
    SYNTH_TEST_CHECK(estimator.estimate_target_frames("a", "", 0, 1.0f) == 16);

    // A reference text with no frames behind it is not a reference: the anchor
    // takes over, which is what an auto-voice request looks like.
    SYNTH_TEST_CHECK(estimator.estimate_target_frames("OmniVoice speaks with one voice.", kCloneRefText, 0, 1.0f) ==
                     50);

    return 0;
}

}  // namespace

int main() {
    SYNTH_TEST_CHECK(check_combine_text() == 0);
    SYNTH_TEST_CHECK(check_add_punctuation() == 0);
    SYNTH_TEST_CHECK(check_style_text() == 0);
    SYNTH_TEST_CHECK(check_nonverbal_split() == 0);
    SYNTH_TEST_CHECK(check_assemble_prompt_ids() == 0);
    SYNTH_TEST_CHECK(check_char_weights() == 0);
    SYNTH_TEST_CHECK(check_estimates_match_oracle() == 0);
    SYNTH_TEST_CHECK(check_boost_curve() == 0);
    SYNTH_TEST_CHECK(check_frame_truncation() == 0);
    return 0;
}
