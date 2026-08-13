// The two-track ICL prompt block and its min(T1, T2) alignment.
//
// This is the piece of transcript-assisted cloning that nothing downstream can
// check. A block whose two tracks are misaligned by one position produces
// fluent speech, in approximately the right voice, in the right language, at
// the right length -- talker-host.h says so at the top of build_talker_prompt
// and it is the reason this file exists. The end-to-end audio cannot catch it
// either: the talker samples, so the audio was never going to match the
// oracle's sample for sample.
//
// Every rule below is one the oracle's own alignment.json supplies the target
// for (scripts/dump_reference_qwen3_tts_icl_prompt.py, one alignment.json per
// materialized case under build/goldens/.../<case>/prompt/). The geometries
// driven here are the measured ones:
//
//     case              T1   T2    arm         trailing
//     base-icl-en       46   102   pad         1
//     base-ref-min      46   14    truncate    32
//     base-text-short   33   102   pad         1
//
// BOTH ARMS ARE TESTED HERE UNCONDITIONALLY. The table above is bookkeeping
// recorded after Task 2 measured it -- it is not what decides which of the two
// tests below gets written, and neither test is presumed covered by the other
// or by an integration case. `alignment.json` reads the arm out of the line of
// the `return` that executed rather than recomputing `T1 > T2`, so it is the
// authority here; as of that measurement both arms already have a materialized
// case, so neither test is the sole exerciser of its arm -- but that is a fact
// about coverage, never a licence to drop one.
//
// Synthetic HParams and in-memory LCG weights throughout: no GGUF, no 2.5 GB
// package. tests/qwen3_tts_icl_prompt_real.cpp is the half that needs the real
// one, and it compares against the oracle's own text_track.f32,
// codec_track.f32 and icl_embed.f32.

#include "arch/qwen3-tts/code-predictor.h"
#include "arch/qwen3-tts/talker-host.h"
#include "arch/qwen3-tts/talker.h"
#include "arch/qwen3-tts/weights.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "test-assert.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

namespace {

using Position = synth::qwen3tts::TalkerInputPosition;
using Text     = Position::Text;

// The production group count: 1 semantic + 15 acoustic. Written as the real
// number rather than a small stand-in, because "sixteen embeddings per
// reference position" is one of the rules under test.
constexpr uint32_t kGroups = 16;

// base-ref-min's and base-icl-en's own geometry. T1 = 30 + 15 + 1 = 46 in both;
// the frame counts are what separate the arms.
constexpr size_t kReferenceTextIds  = 30;
constexpr size_t kTargetTextIds     = 15;
constexpr size_t kTruncateArmFrames = 13;   // T2 = 14  < T1 = 46
constexpr size_t kPadArmFrames      = 101;  // T2 = 102 > T1 = 46

// The two tables a reference frame is read through, and therefore the two
// bounds a reference code has to satisfy. The production package's own numbers.
constexpr uint32_t kTalkerCodecVocab   = 2200;
constexpr uint32_t kPredictorCodeVocab = 2048;

synth::qwen3tts::HParams base_hparams() {
    synth::qwen3tts::HParams h;
    h.voice_mode                = synth::qwen3tts::VoiceMode::ProfileSources;
    h.has_speaker_encoder       = true;
    h.talker.code_group_count   = kGroups;
    h.talker.codec_vocab_size   = kTalkerCodecVocab;
    h.code_predictor.vocab_size = kPredictorCodeVocab;
    h.tokens.tts_bos            = 100;
    h.tokens.tts_eos            = 101;
    h.tokens.tts_pad            = 102;
    h.tokens.codec_bos          = 2149;  // synthesize.qwen3-tts.token.codec_bos_id
    h.tokens.codec_eos          = 201;
    h.tokens.codec_pad          = 202;
    h.tokens.codec_think        = 203;
    h.tokens.codec_nothink      = 204;
    h.tokens.codec_think_bos    = 205;
    h.tokens.codec_think_eos    = 206;
    return h;
}

// Distinct, position-dependent ids so an off-by-one in either track is a value
// mismatch rather than a coincidence. Reference ids and target ids come from
// disjoint ranges for the same reason.
std::vector<uint32_t> reference_text_ids(size_t count) {
    std::vector<uint32_t> ids(count);
    for (size_t index = 0; index < count; ++index) {
        ids[index] = uint32_t(1000 + index);
    }
    return ids;
}

std::vector<uint32_t> target_text_ids(size_t count) {
    std::vector<uint32_t> ids(count);
    for (size_t index = 0; index < count; ++index) {
        ids[index] = uint32_t(2000 + index);
    }
    return ids;
}

// A [kGroups, frames] grid, GROUP-FASTEST: frame f's sixteen codes are
// contiguous at f * kGroups. Every code is distinct across (frame, group) below
// the vocabulary ceiling the caller passes, so a transposed or stage-major read
// lands on a different value rather than on the same one by luck.
std::vector<int32_t> reference_grid(size_t frames, int32_t group0_vocab, int32_t acoustic_vocab) {
    std::vector<int32_t> grid(frames * kGroups);
    for (size_t frame = 0; frame < frames; ++frame) {
        for (size_t group = 0; group < kGroups; ++group) {
            const int32_t vocab           = group == 0 ? group0_vocab : acoustic_vocab;
            grid[frame * kGroups + group] = int32_t((frame * 7 + group * 3 + 1) % size_t(vocab));
        }
    }
    return grid;
}

synth::qwen3tts::TalkerPromptRequest icl_request(size_t frames, size_t target_ids = kTargetTextIds) {
    synth::qwen3tts::TalkerPromptRequest request;
    request.role_tokens           = { 10, 11, 12 };
    request.text_tokens           = target_text_ids(target_ids);
    request.has_language          = true;
    request.language_token        = 300;
    request.has_speaker           = true;
    request.speaker_token         = 4242;
    request.speaker_is_external   = true;
    request.has_reference         = true;
    request.reference_text_tokens = reference_text_ids(kReferenceTextIds);
    request.reference_frames      = frames;
    request.reference_codes       = reference_grid(frames, int32_t(kPredictorCodeVocab), int32_t(kPredictorCodeVocab));
    return request;
}

// The same request with the reference taken away -- the Plan 2 x-vector prompt.
synth::qwen3tts::TalkerPromptRequest x_vector_request() {
    synth::qwen3tts::TalkerPromptRequest request = icl_request(kTruncateArmFrames);
    request.has_reference                        = false;
    request.reference_text_tokens.clear();
    request.reference_codes.clear();
    request.reference_frames = 0;
    return request;
}

// The text track at its own length, before the alignment sees it: the
// reference ids, then the target ids, then one tts_eos
// (modeling_qwen3_tts.py:1978-1981). Built here from the request's own inputs,
// so the tests below compare the port against this rather than against a
// second copy of the port.
std::vector<Position> text_track(const synth::qwen3tts::TalkerPromptRequest & request) {
    std::vector<Position> track;
    for (uint32_t token : request.reference_text_tokens) {
        Position position;
        position.text       = Text::Token;
        position.text_token = token;
        track.push_back(position);
    }
    for (uint32_t token : request.text_tokens) {
        Position position;
        position.text       = Text::Token;
        position.text_token = token;
        track.push_back(position);
    }
    Position eos;
    eos.text = Text::TtsEos;
    track.push_back(eos);
    return track;
}

bool same_text_side(const Position & got, const Position & expected) {
    if (got.text != expected.text) {
        return false;
    }
    return got.text != Text::Token || got.text_token == expected.text_token;
}

// The prefix both modes share: the role tokens, then one position per codec
// entry before codec_bos. Nine for a request that names a language and carries
// a speaker, which is what alignment.json's placement.prefix_positions reports.
size_t prefix_positions(const synth::qwen3tts::TalkerPromptRequest & request) {
    size_t codec = 2 + 1 + 1;  // think/nothink, think_bos, think_eos, codec_pad
    if (request.has_language) {
        codec += 1;
    }
    if (request.has_speaker) {
        codec += 1;
    }
    return request.role_tokens.size() + codec;
}

// Rule 1: T1 and T2 come from the inputs, not from the block. The oracle's
// checks `T1_is_ref_plus_target_plus_eos` and `T2_is_one_plus_ref_frames` are
// the same two statements.
int check_the_two_track_lengths() {
    const synth::qwen3tts::HParams h = base_hparams();

    for (size_t frames : { kTruncateArmFrames, kPadArmFrames }) {
        const synth::qwen3tts::TalkerPromptRequest request = icl_request(frames);
        const size_t t1 = request.reference_text_tokens.size() + request.text_tokens.size() + 1;
        const size_t t2 = 1 + frames;
        SYNTH_TEST_CHECK(text_track(request).size() == t1);
        SYNTH_TEST_CHECK(t1 == 46);
        SYNTH_TEST_CHECK(t2 == frames + 1);

        synth::qwen3tts::TalkerPrompt prompt;
        SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, request, prompt) == SYNTH_OK);
        SYNTH_TEST_CHECK(prompt.positions.size() == prefix_positions(request) + t2);
    }

    // base-text-short's geometry: a shorter target text shortens T1 and nothing
    // else, so the same reference lands in the other arm's neighbourhood.
    const synth::qwen3tts::TalkerPromptRequest short_text = icl_request(kPadArmFrames, /*target_ids=*/2);
    SYNTH_TEST_CHECK(text_track(short_text).size() == 33);
    return 0;
}

// Rule 2, the TRUNCATING arm (modeling_qwen3_tts.py:2016), at base-ref-min's
// geometry: T1 = 46 > T2 = 14. The block takes the text track's first T2
// entries -- element for element, not merely T2 of them -- and the remaining
// T1 - T2 = 32 become the decode loop's schedule, tts_eos last.
int check_the_truncating_arm() {
    const synth::qwen3tts::HParams             h       = base_hparams();
    const synth::qwen3tts::TalkerPromptRequest request = icl_request(kTruncateArmFrames);
    const std::vector<Position>                track   = text_track(request);
    const size_t                               t1      = track.size();
    const size_t                               t2      = kTruncateArmFrames + 1;
    SYNTH_TEST_CHECK(t1 > t2);

    synth::qwen3tts::TalkerPrompt prompt;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, request, prompt) == SYNTH_OK);

    const size_t prefix = prefix_positions(request);
    SYNTH_TEST_CHECK(prompt.positions.size() == prefix + t2);
    for (size_t index = 0; index < t2; ++index) {
        SYNTH_TEST_CHECK(same_text_side(prompt.positions[prefix + index], track[index]));
    }

    SYNTH_TEST_CHECK(prompt.trailing.size() == t1 - t2);
    SYNTH_TEST_CHECK(prompt.trailing.size() == 32);
    for (size_t index = 0; index < prompt.trailing.size(); ++index) {
        SYNTH_TEST_CHECK(same_text_side(prompt.trailing[index], track[t2 + index]));
    }
    SYNTH_TEST_CHECK(prompt.trailing.back().text == Text::TtsEos);
    // And what the decode loop actually reads out of it, in order: the schedule
    // is consumed by index, so an entry in the right list at the wrong index
    // would still be the wrong text at that step.
    for (size_t step = 0; step < prompt.trailing.size(); ++step) {
        const uint32_t expected =
            track[t2 + step].text == Text::TtsEos ? h.tokens.tts_eos : track[t2 + step].text_token;
        SYNTH_TEST_CHECK(synth::qwen3tts::talker_step_text_token(h, prompt, step) == expected);
    }
    // Past its end the schedule pads forever, which is what lets the talker keep
    // emitting frames after the text has run out.
    SYNTH_TEST_CHECK(synth::qwen3tts::talker_step_text_token(h, prompt, prompt.trailing.size()) == h.tokens.tts_pad);
    return 0;
}

// Rule 3, the PADDING arm (modeling_qwen3_tts.py:2018-2019), at base-icl-en's
// geometry: T1 = 46 <= T2 = 102. The text track is padded to T2 with tts_pad
// and the trailing schedule is a single bare tts_pad.
int check_the_padding_arm() {
    const synth::qwen3tts::HParams             h       = base_hparams();
    const synth::qwen3tts::TalkerPromptRequest request = icl_request(kPadArmFrames);
    const std::vector<Position>                track   = text_track(request);
    const size_t                               t1      = track.size();
    const size_t                               t2      = kPadArmFrames + 1;
    SYNTH_TEST_CHECK(t1 <= t2);

    synth::qwen3tts::TalkerPrompt prompt;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, request, prompt) == SYNTH_OK);

    const size_t prefix = prefix_positions(request);
    SYNTH_TEST_CHECK(prompt.positions.size() == prefix + t2);
    for (size_t index = 0; index < t1; ++index) {
        SYNTH_TEST_CHECK(same_text_side(prompt.positions[prefix + index], track[index]));
    }
    for (size_t index = t1; index < t2; ++index) {
        SYNTH_TEST_CHECK(prompt.positions[prefix + index].text == Text::TtsPad);
    }

    SYNTH_TEST_CHECK(prompt.trailing.size() == 1);
    SYNTH_TEST_CHECK(prompt.trailing[0].text == Text::TtsPad);
    SYNTH_TEST_CHECK(!prompt.trailing[0].has_codec);
    SYNTH_TEST_CHECK(synth::qwen3tts::talker_step_text_token(h, prompt, 0) == h.tokens.tts_pad);
    return 0;
}

// Rule 4, on its own because it is the property that makes icl_embed.f32's
// size useless as a branch discriminator: the block is T2 positions in BOTH
// arms. A future reader will otherwise try to read the arm off the block, and
// alignment.json's `arms_are_indistinguishable_by_length` exists for the same
// reason.
int check_the_block_is_t2_positions_in_both_arms() {
    const synth::qwen3tts::HParams h = base_hparams();

    for (size_t frames : { kTruncateArmFrames, kPadArmFrames }) {
        const synth::qwen3tts::TalkerPromptRequest request = icl_request(frames);
        synth::qwen3tts::TalkerPrompt              prompt;
        SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, request, prompt) == SYNTH_OK);
        SYNTH_TEST_CHECK(prompt.positions.size() - prefix_positions(request) == frames + 1);
    }

    // The two arms at the same T2 would be indistinguishable by block length;
    // only the trailing schedule separates them. Driven at T2 = 14 against a
    // text track short enough to pad.
    const synth::qwen3tts::TalkerPromptRequest truncating = icl_request(kTruncateArmFrames);
    synth::qwen3tts::TalkerPromptRequest       padding    = truncating;
    padding.reference_text_tokens.resize(5);
    padding.text_tokens.resize(5);

    synth::qwen3tts::TalkerPrompt a;
    synth::qwen3tts::TalkerPrompt b;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, truncating, a) == SYNTH_OK);
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, padding, b) == SYNTH_OK);
    SYNTH_TEST_CHECK(a.positions.size() == b.positions.size());
    SYNTH_TEST_CHECK(text_track(padding).size() <= kTruncateArmFrames + 1);
    SYNTH_TEST_CHECK(a.trailing.size() == 32);
    SYNTH_TEST_CHECK(b.trailing.size() == 1);
    return 0;
}

// Rule 5, host half: the codec track opens with codec_bos_id, a lone token
// through the TALKER's own codec embedding (modeling_qwen3_tts.py:1990-1998) --
// so it carries no acoustic groups, and every position after it carries all
// fifteen. The graph half, which pins that it really is the talker's table and
// not the predictor's, is check_the_graph_sums_sixteen_in_group_order below.
int check_the_codec_track_opens_with_codec_bos() {
    const synth::qwen3tts::HParams             h       = base_hparams();
    const synth::qwen3tts::TalkerPromptRequest request = icl_request(kTruncateArmFrames);
    synth::qwen3tts::TalkerPrompt              prompt;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, request, prompt) == SYNTH_OK);

    const size_t     prefix = prefix_positions(request);
    const Position & first  = prompt.positions[prefix];
    SYNTH_TEST_CHECK(first.has_codec);
    SYNTH_TEST_CHECK(first.codec_token == h.tokens.codec_bos);
    SYNTH_TEST_CHECK(first.codec_token == 2149);
    SYNTH_TEST_CHECK(first.acoustic_codes.empty());

    // Every reference position is a whole frame, group for group, read out of
    // the grid at its own offset. This is the element-wise check: a grid read
    // stage-major, or one frame late, lands on a different value here.
    for (size_t frame = 0; frame < kTruncateArmFrames; ++frame) {
        const Position & position = prompt.positions[prefix + 1 + frame];
        SYNTH_TEST_CHECK(position.has_codec);
        SYNTH_TEST_CHECK(position.codec_token == uint32_t(request.reference_codes[frame * kGroups]));
        SYNTH_TEST_CHECK(position.acoustic_codes.size() == kGroups - 1);
        for (size_t group = 1; group < kGroups; ++group) {
            SYNTH_TEST_CHECK(position.acoustic_codes[group - 1] ==
                             uint32_t(request.reference_codes[frame * kGroups + group]));
        }
    }
    return 0;
}

// What the graph is handed: the flattened streams. The codec run is still one
// contiguous tail starting where it always did, and the acoustic grid is
// group-major over the reference positions only.
int check_the_flattened_streams() {
    const synth::qwen3tts::HParams             h       = base_hparams();
    const synth::qwen3tts::TalkerPromptRequest request = icl_request(kTruncateArmFrames);
    synth::qwen3tts::TalkerPrompt              prompt;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, request, prompt) == SYNTH_OK);

    std::vector<int32_t> text;
    std::vector<int32_t> codec;
    std::vector<int32_t> acoustic;
    int64_t              codec_offset    = -1;
    int64_t              acoustic_offset = -1;
    SYNTH_TEST_CHECK(synth::qwen3tts::flatten_talker_prompt(h, prompt, text, codec, codec_offset, acoustic,
                                                            acoustic_offset) == SYNTH_OK);

    const size_t prefix = prefix_positions(request);
    SYNTH_TEST_CHECK(text.size() == prompt.positions.size());
    // The codec run starts after the role prefix, exactly where the non-ICL
    // prompt's does, and reaches the end.
    SYNTH_TEST_CHECK(codec_offset == int64_t(request.role_tokens.size()));
    SYNTH_TEST_CHECK(size_t(codec_offset) + codec.size() == prompt.positions.size());
    // The acoustic run starts one position INTO the block -- codec_bos has no
    // frame -- and also reaches the end.
    SYNTH_TEST_CHECK(acoustic_offset == int64_t(prefix + 1));
    SYNTH_TEST_CHECK(acoustic.size() == kTruncateArmFrames * (kGroups - 1));

    // Group-major: all frames of group 1, then all of group 2, ...
    for (size_t group = 1; group < kGroups; ++group) {
        for (size_t frame = 0; frame < kTruncateArmFrames; ++frame) {
            SYNTH_TEST_CHECK(acoustic[(group - 1) * kTruncateArmFrames + frame] ==
                             request.reference_codes[frame * kGroups + group]);
        }
    }
    return 0;
}

// Rule 7: the block is concatenated AFTER the shared prefix
// (modeling_qwen3_tts.py:2197) in place of the single first-text-token position
// the non-ICL branch appends (:2199-2202). So the prefix is untouched, position
// for position, and the total is that non-ICL total plus T2 - 1.
int check_the_block_is_appended_after_the_prefix() {
    const synth::qwen3tts::HParams             h   = base_hparams();
    const synth::qwen3tts::TalkerPromptRequest icl = icl_request(kTruncateArmFrames);
    const size_t                               t2  = kTruncateArmFrames + 1;

    synth::qwen3tts::TalkerPrompt with_reference;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, icl, with_reference) == SYNTH_OK);

    // The same request without the reference, cut back to the one text token
    // upstream's non-ICL branch puts in the prefill. That prompt is
    // prefix + 1 positions: the token against codec_bos, and nothing after it.
    synth::qwen3tts::TalkerPromptRequest single = x_vector_request();
    single.text_tokens.resize(1);
    synth::qwen3tts::TalkerPrompt without_reference;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, single, without_reference) == SYNTH_OK);

    const size_t prefix = prefix_positions(icl);
    SYNTH_TEST_CHECK(prefix == 9);
    SYNTH_TEST_CHECK(with_reference.positions.size() == prefix + t2);

    // Position for position, the prefix is the same object in both modes --
    // text side and codec side. A prefix that shifted by one would still be the
    // right length.
    for (size_t index = 0; index < prefix; ++index) {
        const Position & a = with_reference.positions[index];
        const Position & b = without_reference.positions[index];
        SYNTH_TEST_CHECK(same_text_side(a, b));
        SYNTH_TEST_CHECK(a.has_codec == b.has_codec);
        SYNTH_TEST_CHECK(a.codec_token == b.codec_token);
        SYNTH_TEST_CHECK(a.acoustic_codes.empty());
    }
    // The non-ICL prompt's own length, which is what "plus T2 - 1" is measured
    // FROM: prefix, one text position, then the closing tts_eos and codec_bos
    // positions the non-streaming layout adds. Written as its own assertion
    // rather than as `(prefix + 1) + (t2 - 1)` on the ICL side, which is
    // arithmetically the same statement as `prefix + t2` above and would read
    // as a second, independent check without being one.
    SYNTH_TEST_CHECK(without_reference.positions.size() == prefix + 3);
    return 0;
}

// Rule 8, host half: the x-vector mechanism is untouched by ICL. The slot's
// index is an index into the flattened codec run, and the ICL block extends
// that run -- so the risk is precisely that the index moves. It must not.
int check_the_x_vector_path_is_unchanged() {
    const synth::qwen3tts::HParams h = base_hparams();

    synth::qwen3tts::TalkerPrompt plain;
    synth::qwen3tts::TalkerPrompt cloned;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, x_vector_request(), plain) == SYNTH_OK);
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, icl_request(kTruncateArmFrames), cloned) == SYNTH_OK);

    // think, think_bos, language, think_eos, then the speaker: index 4 into the
    // codec run, in both modes.
    SYNTH_TEST_CHECK(plain.external_speaker_index == 4);
    SYNTH_TEST_CHECK(cloned.external_speaker_index == 4);

    std::vector<int32_t> text;
    std::vector<int32_t> codec;
    std::vector<int32_t> acoustic;
    int64_t              codec_offset    = -1;
    int64_t              acoustic_offset = -1;
    SYNTH_TEST_CHECK(synth::qwen3tts::flatten_talker_prompt(h, plain, text, codec, codec_offset, acoustic,
                                                            acoustic_offset) == SYNTH_OK);

    // Exactly the streams a Plan 2 prompt produced before the ICL block
    // existed: fifteen target tokens between tts_bos and tts_eos, one codec_pad
    // each, codec_bos closing against a pad.
    SYNTH_TEST_CHECK(codec_offset == 3);
    SYNTH_TEST_CHECK(acoustic.empty());
    SYNTH_TEST_CHECK(acoustic_offset == -1);
    SYNTH_TEST_CHECK(text.size() == 3 + 6 + kTargetTextIds + 2);
    const std::vector<int32_t> expected_head = { 10, 11, 12, 102, 102, 102, 102, 102, 100, 2000 };
    for (size_t index = 0; index < expected_head.size(); ++index) {
        SYNTH_TEST_CHECK(text[index] == expected_head[index]);
    }
    SYNTH_TEST_CHECK(text[text.size() - 2] == 101);
    SYNTH_TEST_CHECK(text.back() == 102);
    SYNTH_TEST_CHECK(codec[0] == 203);
    SYNTH_TEST_CHECK(codec[2] == 300);
    SYNTH_TEST_CHECK(codec[4] == 202);  // the external speaker's inert placeholder
    SYNTH_TEST_CHECK(codec.back() == 2149);
    for (const Position & position : plain.positions) {
        SYNTH_TEST_CHECK(position.acoustic_codes.empty());
    }
    SYNTH_TEST_CHECK(plain.trailing.empty());
    return 0;
}

// What must be refused rather than built into a prompt nothing downstream can
// question.
int check_malformed_references_are_refused() {
    const synth::qwen3tts::HParams h = base_hparams();
    synth::qwen3tts::TalkerPrompt  prompt;

    synth::qwen3tts::TalkerPromptRequest no_frames = icl_request(kTruncateArmFrames);
    no_frames.reference_frames                     = 0;
    no_frames.reference_codes.clear();
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, no_frames, prompt) == SYNTH_ERR_INVALID_ARG);

    // A grid whose element count does not match frames * groups: the right
    // range, the wrong shape.
    synth::qwen3tts::TalkerPromptRequest short_grid = icl_request(kTruncateArmFrames);
    short_grid.reference_codes.pop_back();
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, short_grid, prompt) == SYNTH_ERR_INVALID_ARG);

    // ggml_get_rows asserts `i01 >= 0 && i01 < ne01` (ggml-cpu/ops.cpp:4779)
    // and ABORTS the process on either side, so BOTH sides have to be refused
    // here. A blacklist of one side is the shape this project's own rule --
    // whitelist your own format -- exists to prevent.
    synth::qwen3tts::TalkerPromptRequest negative = icl_request(kTruncateArmFrames);
    negative.reference_codes[7]                   = -1;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, negative, prompt) == SYNTH_ERR_INVALID_ARG);

    // Group 0 is read through the TALKER's codec embedding and groups 1..15
    // through the code predictor's, and the two tables are different heights.
    // A code that is in range for one and not the other is the case a single
    // shared bound would wave through: 2100 is a real talker codec row and is
    // past the end of every predictor table.
    SYNTH_TEST_CHECK(kTalkerCodecVocab > kPredictorCodeVocab);
    synth::qwen3tts::TalkerPromptRequest wide_group_0 = icl_request(kTruncateArmFrames);
    wide_group_0.reference_codes[3 * kGroups]         = int32_t(kPredictorCodeVocab) + 52;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, wide_group_0, prompt) == SYNTH_OK);

    synth::qwen3tts::TalkerPromptRequest past_group_0 = icl_request(kTruncateArmFrames);
    past_group_0.reference_codes[3 * kGroups]         = int32_t(kTalkerCodecVocab);
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, past_group_0, prompt) == SYNTH_ERR_INVALID_ARG);

    synth::qwen3tts::TalkerPromptRequest past_acoustic = icl_request(kTruncateArmFrames);
    past_acoustic.reference_codes[3 * kGroups + 9]     = int32_t(kPredictorCodeVocab);
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, past_acoustic, prompt) == SYNTH_ERR_INVALID_ARG);

    // A package that declares no table height at all cannot be range-checked,
    // and building a prompt whose codes nothing bounded is how the abort gets
    // reached anyway.
    synth::qwen3tts::HParams unbounded  = h;
    unbounded.code_predictor.vocab_size = 0;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(unbounded, icl_request(kTruncateArmFrames), prompt) ==
                     SYNTH_ERR_INVALID_ARG);

    // A refusal must leave nothing half-built behind it.
    SYNTH_TEST_CHECK(prompt.positions.empty());
    SYNTH_TEST_CHECK(prompt.trailing.empty());

    // And the flattener's own invariants, on prompts hand-damaged past what
    // build_talker_prompt can produce.
    synth::qwen3tts::TalkerPrompt good;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, icl_request(kTruncateArmFrames), good) == SYNTH_OK);

    std::vector<int32_t> text;
    std::vector<int32_t> codec;
    std::vector<int32_t> acoustic;
    int64_t              codec_offset    = -1;
    int64_t              acoustic_offset = -1;

    // An acoustic run that stops short of the last position is not a tail, and
    // the graph adds it as one.
    synth::qwen3tts::TalkerPrompt short_run = good;
    short_run.positions.back().acoustic_codes.clear();
    SYNTH_TEST_CHECK(synth::qwen3tts::flatten_talker_prompt(h, short_run, text, codec, codec_offset, acoustic,
                                                            acoustic_offset) == SYNTH_ERR_INVALID_ARG);

    // A gap inside the run has the same consequence.
    synth::qwen3tts::TalkerPrompt gapped = good;
    gapped.positions[gapped.positions.size() - 3].acoustic_codes.clear();
    SYNTH_TEST_CHECK(synth::qwen3tts::flatten_talker_prompt(h, gapped, text, codec, codec_offset, acoustic,
                                                            acoustic_offset) == SYNTH_ERR_INVALID_ARG);

    // A position carrying fourteen groups instead of fifteen.
    synth::qwen3tts::TalkerPrompt ragged = good;
    ragged.positions.back().acoustic_codes.pop_back();
    SYNTH_TEST_CHECK(synth::qwen3tts::flatten_talker_prompt(h, ragged, text, codec, codec_offset, acoustic,
                                                            acoustic_offset) == SYNTH_ERR_INVALID_ARG);

    // Acoustic groups with no group 0 under them is a frame with no semantic
    // code -- and the codec run would no longer be a tail either.
    synth::qwen3tts::TalkerPrompt headless = good;
    headless.positions.back().has_codec    = false;
    SYNTH_TEST_CHECK(synth::qwen3tts::flatten_talker_prompt(h, headless, text, codec, codec_offset, acoustic,
                                                            acoustic_offset) == SYNTH_ERR_INVALID_ARG);
    return 0;
}

class LcgStream {
  public:
    explicit LcgStream(uint64_t seed) : state_(seed) {}

    float next() {
        state_ = state_ * 6364136223846793005ull + 1442695040888963407ull;
        return float(state_ >> 40) / 8388608.0f - 1.0f;
    }

    std::vector<float> fill(size_t count, float scale) {
        std::vector<float> values(count);
        for (float & value : values) {
            value = next() * scale;
        }
        return values;
    }

  private:
    uint64_t state_;
};

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

constexpr uint32_t kHidden         = 3;
constexpr uint32_t kTextWidth      = 4;
constexpr uint32_t kTextVocab      = 3000;
constexpr uint32_t kCodecVocab     = 2200;  // codec_bos is 2149
constexpr uint32_t kPredictorVocab = 11;
constexpr uint32_t kGraphFrames    = 6;
constexpr uint64_t kGraphSeed      = 20260814u;

// The talker's codec embedding and the predictor's fifteen tables, all
// materially different from each other, so summing the wrong table -- or the
// right tables in the wrong order -- moves the answer.
struct GraphFixture {
    ggml_backend_t                        backend = nullptr;
    Context                               persistent;
    ggml_backend_buffer_t                 buffer = nullptr;
    synth::qwen3tts::TalkerWeights        talker;
    synth::qwen3tts::CodePredictorWeights predictor;
    std::vector<float>                    codec_table;
    std::vector<std::vector<float>>       predictor_tables;

    ~GraphFixture() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
};

bool build_graph_fixture(GraphFixture & fixture) {
    ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (device == nullptr) {
        return false;
    }
    fixture.backend = ggml_backend_dev_init(device, nullptr);
    if (fixture.backend == nullptr) {
        return false;
    }
    fixture.persistent  = make_context(ggml_tensor_overhead() * 64);
    ggml_context * pctx = fixture.persistent.get();

    std::vector<ggml_tensor *> ordered;
    std::vector<float>         scales;
    auto                       add = [&](ggml_tensor * tensor, float scale) {
        ordered.push_back(tensor);
        scales.push_back(scale);
        return tensor;
    };

    synth::qwen3tts::TalkerWeights & w = fixture.talker;
    w.text_embedding                   = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kTextWidth, kTextVocab), 0.5f);
    w.text_projection_1.weight         = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kTextWidth, kTextWidth), 0.5f);
    w.text_projection_1.bias           = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kTextWidth), 0.25f);
    w.text_projection_2.weight         = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kTextWidth, kHidden), 0.5f);
    w.text_projection_2.bias           = add(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kHidden), 0.25f);
    w.codec_embedding                  = add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kCodecVocab), 0.5f);

    fixture.predictor.codec_embedding.resize(kGroups - 1);
    for (size_t table = 0; table < kGroups - 1; ++table) {
        // A different scale per table, so a permuted sum is not merely
        // improbable but arithmetically impossible to coincide.
        fixture.predictor.codec_embedding[table] =
            add(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kHidden, kPredictorVocab), 0.25f + 0.1f * float(table));
    }

    fixture.buffer = ggml_backend_alloc_ctx_tensors(pctx, fixture.backend);
    if (fixture.buffer == nullptr) {
        return false;
    }

    LcgStream stream(kGraphSeed);
    for (size_t index = 0; index < ordered.size(); ++index) {
        const std::vector<float> values = stream.fill(size_t(ggml_nelements(ordered[index])), scales[index]);
        ggml_backend_tensor_set(ordered[index], values.data(), 0, ggml_nbytes(ordered[index]));
    }

    // Read the tables back so the expectation below is computed on the host,
    // independently of any graph this file also builds.
    fixture.codec_table.resize(size_t(kHidden) * kCodecVocab);
    ggml_backend_tensor_get(w.codec_embedding, fixture.codec_table.data(), 0, ggml_nbytes(w.codec_embedding));
    fixture.predictor_tables.resize(kGroups - 1);
    for (size_t table = 0; table < kGroups - 1; ++table) {
        ggml_tensor * tensor = fixture.predictor.codec_embedding[table];
        fixture.predictor_tables[table].resize(size_t(kHidden) * kPredictorVocab);
        ggml_backend_tensor_get(tensor, fixture.predictor_tables[table].data(), 0, ggml_nbytes(tensor));
    }
    return true;
}

// Builds the whole prefill for a prompt and hands back the text tower's own
// output beside it, so the caller can subtract one from the other and be left
// with exactly the codec side.
bool run_prefill(GraphFixture &                        fixture,
                 const synth::qwen3tts::HParams &      hparams,
                 const synth::qwen3tts::TalkerPrompt & prompt,
                 ggml_tensor *                         speaker_source,
                 int64_t                               speaker_index,
                 std::vector<float> &                  text_values,
                 std::vector<float> &                  prefill_values) {
    std::vector<int32_t> text;
    std::vector<int32_t> codec;
    std::vector<int32_t> acoustic;
    int64_t              codec_offset    = -1;
    int64_t              acoustic_offset = -1;
    if (synth::qwen3tts::flatten_talker_prompt(hparams, prompt, text, codec, codec_offset, acoustic, acoustic_offset) !=
        SYNTH_OK) {
        return false;
    }

    constexpr size_t kNodeBudget = 512;
    Context          graph_ctx =
        make_context(ggml_tensor_overhead() * (kNodeBudget + 64) + ggml_graph_overhead_custom(kNodeBudget, false));
    ggml_context * gctx  = graph_ctx.get();
    ggml_cgraph *  graph = ggml_new_graph_custom(gctx, kNodeBudget, false);

    ggml_tensor * t_text     = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, int64_t(text.size()));
    ggml_tensor * t_codec    = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, int64_t(codec.size()));
    ggml_tensor * t_acoustic = nullptr;
    if (!acoustic.empty()) {
        // [frames, groups-1]: the layout sum_code_embeddings reads.
        t_acoustic = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, int64_t(acoustic.size()) / int64_t(kGroups - 1),
                                        int64_t(kGroups) - 1);
    }
    ggml_backend_buffer_t inputs = ggml_backend_alloc_ctx_tensors(gctx, fixture.backend);
    if (inputs == nullptr) {
        return false;
    }
    ggml_backend_tensor_set(t_text, text.data(), 0, ggml_nbytes(t_text));
    ggml_backend_tensor_set(t_codec, codec.data(), 0, ggml_nbytes(t_codec));
    if (t_acoustic != nullptr) {
        ggml_backend_tensor_set(t_acoustic, acoustic.data(), 0, ggml_nbytes(t_acoustic));
    }

    ggml_tensor * summed = nullptr;
    if (t_acoustic != nullptr) {
        summed = synth::qwen3tts::sum_code_embeddings(gctx, fixture.predictor, t_acoustic);
        if (summed == nullptr) {
            ggml_backend_buffer_free(inputs);
            return false;
        }
    }
    ggml_tensor * projected = synth::qwen3tts::build_text_projection(gctx, fixture.talker, t_text);
    ggml_tensor * prefill   = synth::qwen3tts::build_talker_prefill_input(
        gctx, fixture.talker, t_text, t_codec, codec_offset, speaker_source, speaker_index, summed, acoustic_offset);
    if (projected == nullptr || prefill == nullptr) {
        ggml_backend_buffer_free(inputs);
        return false;
    }
    for (ggml_tensor * output : { projected, prefill }) {
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
    }

    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(fixture.backend));
    bool           ok        = allocator != nullptr && ggml_gallocr_alloc_graph(allocator, graph);
    if (ok) {
        ok = ggml_backend_graph_compute(fixture.backend, graph) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        text_values.resize(size_t(ggml_nelements(projected)));
        ggml_backend_tensor_get(projected, text_values.data(), 0, ggml_nbytes(projected));
        prefill_values.resize(size_t(ggml_nelements(prefill)));
        ggml_backend_tensor_get(prefill, prefill_values.data(), 0, ggml_nbytes(prefill));
    }
    if (allocator != nullptr) {
        ggml_gallocr_free(allocator);
    }
    ggml_backend_buffer_free(inputs);
    return ok;
}

// The graph fixture's own token geometry: codec_bos is the real 2149 so the
// rule-5 assertion is about the real constant, and the text ids stay inside a
// 3000-entry vocabulary.
synth::qwen3tts::HParams graph_hparams() {
    synth::qwen3tts::HParams h  = base_hparams();
    // The two bounds have to be the fixture's OWN table heights, not the
    // production ones: build_talker_prompt refuses a code that ggml_get_rows
    // would abort on, and here the tables really are this small.
    h.talker.codec_vocab_size   = kCodecVocab;
    h.code_predictor.vocab_size = kPredictorVocab;
    h.tokens.tts_bos            = 1;
    h.tokens.tts_eos            = 2;
    h.tokens.tts_pad            = 3;
    h.tokens.codec_eos          = 5;
    h.tokens.codec_pad          = 6;
    h.tokens.codec_think        = 7;
    h.tokens.codec_nothink      = 8;
    h.tokens.codec_think_bos    = 9;
    h.tokens.codec_think_eos    = 10;
    return h;
}

synth::qwen3tts::TalkerPromptRequest graph_request(size_t frames) {
    synth::qwen3tts::TalkerPromptRequest request;
    request.role_tokens           = { 10, 11, 12 };
    request.text_tokens           = { 20, 21, 22 };
    request.has_language          = true;
    request.language_token        = 11;
    request.has_speaker           = true;
    request.speaker_token         = 12;
    request.has_reference         = true;
    request.reference_text_tokens = { 30, 31, 32, 33 };
    request.reference_frames      = frames;
    request.reference_codes       = reference_grid(frames, int32_t(kCodecVocab), int32_t(kPredictorVocab));
    return request;
}

// Rules 5 and 6, on values: every reference position is the sum of SIXTEEN
// embeddings -- group 0 from the talker's own codec table, groups 1..15 from
// the code predictor's, IN GROUP ORDER -- and the block's first position is a
// lone codec_bos through the talker's table.
//
// The expectation is computed on the host from the raw tables, so it is an
// independent implementation of the sum rather than a second call into the
// one under test. Reading the predictor's table for group 0, or the talker's
// for any other, or the fifteen tables in any other order, all land somewhere
// else.
int check_the_graph_sums_sixteen_in_group_order() {
    GraphFixture fixture;
    SYNTH_TEST_CHECK(build_graph_fixture(fixture));

    const synth::qwen3tts::HParams             h       = graph_hparams();
    const synth::qwen3tts::TalkerPromptRequest request = graph_request(kGraphFrames);
    synth::qwen3tts::TalkerPrompt              prompt;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, request, prompt) == SYNTH_OK);

    std::vector<float> text_values;
    std::vector<float> prefill_values;
    SYNTH_TEST_CHECK(run_prefill(fixture, h, prompt, nullptr, -1, text_values, prefill_values));

    const size_t positions = prompt.positions.size();
    SYNTH_TEST_CHECK(prefill_values.size() == positions * kHidden);
    SYNTH_TEST_CHECK(text_values.size() == prefill_values.size());

    const size_t prefix = prefix_positions(request);
    double       worst  = 0.0;
    for (size_t position = 0; position < positions; ++position) {
        const Position & layout = prompt.positions[position];
        for (size_t lane = 0; lane < kHidden; ++lane) {
            double expected = text_values[position * kHidden + lane];
            if (layout.has_codec) {
                expected += fixture.codec_table[size_t(layout.codec_token) * kHidden + lane];
            }
            for (size_t group = 0; group < layout.acoustic_codes.size(); ++group) {
                expected += fixture.predictor_tables[group][size_t(layout.acoustic_codes[group]) * kHidden + lane];
            }
            worst = std::fmax(worst, std::fabs(double(prefill_values[position * kHidden + lane]) - expected));
        }
    }
    std::printf("  sixteen-group sum: max |port - host| = %.3e over %zu positions\n", worst, positions);
    SYNTH_TEST_CHECK(worst < 1e-6);

    // The block's first position took the talker's codec_bos row and nothing
    // from the predictor: its codec side is exactly one table lookup.
    for (size_t lane = 0; lane < kHidden; ++lane) {
        const double codec_side =
            double(prefill_values[prefix * kHidden + lane]) - double(text_values[prefix * kHidden + lane]);
        const double bos_row = fixture.codec_table[size_t(h.tokens.codec_bos) * kHidden + lane];
        SYNTH_TEST_CHECK(std::fabs(codec_side - bos_row) < 1e-6);
    }

    // And the order is content, not decoration. Swapping two groups of every
    // frame leaves the same sixteen ids at every position -- the same multiset,
    // read through a different pair of tables -- and must not land on the same
    // vector. Without this, a code grid handed over in the wrong group order
    // would be invisible.
    synth::qwen3tts::TalkerPromptRequest permuted = request;
    for (size_t frame = 0; frame < kGraphFrames; ++frame) {
        std::swap(permuted.reference_codes[frame * kGroups + 3], permuted.reference_codes[frame * kGroups + 9]);
    }
    SYNTH_TEST_CHECK(permuted.reference_codes != request.reference_codes);
    synth::qwen3tts::TalkerPrompt permuted_prompt;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, permuted, permuted_prompt) == SYNTH_OK);

    std::vector<float> permuted_text;
    std::vector<float> permuted_prefill;
    SYNTH_TEST_CHECK(run_prefill(fixture, h, permuted_prompt, nullptr, -1, permuted_text, permuted_prefill));
    SYNTH_TEST_CHECK(permuted_prefill.size() == prefill_values.size());

    double separation = 0.0;
    for (size_t index = 0; index < prefill_values.size(); ++index) {
        separation = std::fmax(separation, std::fabs(double(permuted_prefill[index]) - double(prefill_values[index])));
    }
    std::printf("  permuting groups 3 and 9 moves the block by %.3e\n", separation);
    SYNTH_TEST_CHECK(separation > 1e-3);
    return 0;
}

// Rule 8, graph half: the x-vector substitution and the ICL block compose. With
// both in play, every position outside the speaker slot is still the ordinary
// text-plus-codec-plus-acoustic sum, and the slot itself is still the x-vector
// instead of its placeholder's row -- at the same index it had without ICL.
int check_the_x_vector_still_substitutes_under_icl() {
    GraphFixture fixture;
    SYNTH_TEST_CHECK(build_graph_fixture(fixture));

    const synth::qwen3tts::HParams       h       = graph_hparams();
    synth::qwen3tts::TalkerPromptRequest request = graph_request(kGraphFrames);
    request.speaker_is_external                  = true;
    synth::qwen3tts::TalkerPrompt prompt;
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prompt(h, request, prompt) == SYNTH_OK);
    SYNTH_TEST_CHECK(prompt.external_speaker_index == 4);

    Context               speaker_ctx    = make_context(ggml_tensor_overhead() * 4);
    ggml_tensor *         speaker        = ggml_new_tensor_2d(speaker_ctx.get(), GGML_TYPE_F32, kHidden, 1);
    ggml_backend_buffer_t speaker_buffer = ggml_backend_alloc_ctx_tensors(speaker_ctx.get(), fixture.backend);
    SYNTH_TEST_CHECK(speaker_buffer != nullptr);
    const std::vector<float> x_vector = { 0.125f, -0.375f, 0.75f };
    ggml_backend_tensor_set(speaker, x_vector.data(), 0, ggml_nbytes(speaker));

    std::vector<float> text_values;
    std::vector<float> prefill_values;
    const bool         ok =
        run_prefill(fixture, h, prompt, speaker, prompt.external_speaker_index, text_values, prefill_values);
    ggml_backend_buffer_free(speaker_buffer);
    SYNTH_TEST_CHECK(ok);

    // codec_offset is 3 (the role prefix), so the speaker's POSITION is 3 + 4.
    const size_t speaker_position = 3 + size_t(prompt.external_speaker_index);
    double       worst            = 0.0;
    for (size_t position = 0; position < prompt.positions.size(); ++position) {
        const Position & layout = prompt.positions[position];
        for (size_t lane = 0; lane < kHidden; ++lane) {
            double expected = text_values[position * kHidden + lane];
            if (position == speaker_position) {
                expected += x_vector[lane];
            } else if (layout.has_codec) {
                expected += fixture.codec_table[size_t(layout.codec_token) * kHidden + lane];
            }
            for (size_t group = 0; group < layout.acoustic_codes.size(); ++group) {
                expected += fixture.predictor_tables[group][size_t(layout.acoustic_codes[group]) * kHidden + lane];
            }
            worst = std::fmax(worst, std::fabs(double(prefill_values[position * kHidden + lane]) - expected));
        }
    }
    std::printf("  x-vector under ICL: max |port - host| = %.3e\n", worst);
    SYNTH_TEST_CHECK(worst < 1e-6);
    // The placeholder's own row must not be what landed there.
    const double placeholder = fixture.codec_table[size_t(request.speaker_token) * kHidden];
    SYNTH_TEST_CHECK(std::fabs(placeholder - x_vector[0]) > 1e-3);
    return 0;
}

// sum_code_embeddings reads each group's ids as a contiguous 1-D run, so a
// strided grid would silently take the wrong ones. This needs at least two
// tables to express: ggml_is_contiguous skips dimensions of extent 1, which is
// why tests/qwen3_tts_code_predictor_test.cpp's one-table fixture cannot host
// this check.
int check_the_sum_refuses_a_strided_grid() {
    Context        storage = make_context(ggml_tensor_overhead() * 32);
    ggml_context * ctx     = storage.get();

    synth::qwen3tts::CodePredictorWeights weights;
    for (size_t table = 0; table < 2; ++table) {
        weights.codec_embedding.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, kPredictorVocab));
    }

    ggml_tensor * grid = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, 2);
    SYNTH_TEST_CHECK(synth::qwen3tts::sum_code_embeddings(ctx, weights, grid) != nullptr);

    // The same two groups, taken two frames at a time out of a four-frame grid:
    // right shape, wrong stride.
    ggml_tensor * strided = ggml_view_2d(ctx, grid, 2, 2, grid->nb[1], 0);
    SYNTH_TEST_CHECK(strided->ne[0] == 2 && strided->ne[1] == 2);
    SYNTH_TEST_CHECK(!ggml_is_contiguous(strided));
    SYNTH_TEST_CHECK(synth::qwen3tts::sum_code_embeddings(ctx, weights, strided) == nullptr);

    // And the group count is the SLOW axis: a [2, 4] grid has the same eight
    // ids and four groups' worth of them.
    ggml_tensor * transposed = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 4);
    SYNTH_TEST_CHECK(synth::qwen3tts::sum_code_embeddings(ctx, weights, transposed) == nullptr);
    return 0;
}

// The graph refuses an acoustic block it cannot place as a tail, rather than
// accumulating it wherever the offset lands.
int check_the_graph_refuses_a_misplaced_acoustic_block() {
    // Seven build attempts, each of which lays out a text tower before it
    // reaches whatever it is going to refuse.
    Context        storage = make_context(ggml_tensor_overhead() * 256);
    ggml_context * ctx     = storage.get();

    synth::qwen3tts::TalkerWeights w;
    w.text_embedding           = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kTextWidth, 16);
    w.text_projection_1.weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kTextWidth, kTextWidth);
    w.text_projection_1.bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kTextWidth);
    w.text_projection_2.weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kTextWidth, kHidden);
    w.text_projection_2.bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHidden);
    w.codec_embedding          = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, 16);

    ggml_tensor * text     = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 8);
    ggml_tensor * codec    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 8);
    ggml_tensor * acoustic = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, 4);
    ggml_tensor * narrow   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden - 1, 4);

    // Placed as a tail: offset 4 plus four positions reaches the eighth.
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prefill_input(ctx, w, text, codec, 0, nullptr, -1, acoustic, 4) !=
                     nullptr);
    // One short of the end: the last reference frame would go unread.
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prefill_input(ctx, w, text, codec, 0, nullptr, -1, acoustic, 3) ==
                     nullptr);
    // Past the end.
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prefill_input(ctx, w, text, codec, 0, nullptr, -1, acoustic, 5) ==
                     nullptr);
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prefill_input(ctx, w, text, codec, 0, nullptr, -1, acoustic, -1) ==
                     nullptr);
    // The wrong width is not a row of the prefill.
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prefill_input(ctx, w, text, codec, 0, nullptr, -1, narrow, 4) ==
                     nullptr);
    // An offset with nothing to place at it: an ICL prompt flattened and then
    // half-dropped.
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prefill_input(ctx, w, text, codec, 0, nullptr, -1, nullptr, 4) ==
                     nullptr);
    // And the Stage 1 call, at the defaults, still builds.
    SYNTH_TEST_CHECK(synth::qwen3tts::build_talker_prefill_input(ctx, w, text, codec, 0) != nullptr);
    return 0;
}

}  // namespace

int main() {
    SYNTH_TEST_CHECK(check_the_two_track_lengths() == 0);
    SYNTH_TEST_CHECK(check_the_truncating_arm() == 0);
    SYNTH_TEST_CHECK(check_the_padding_arm() == 0);
    SYNTH_TEST_CHECK(check_the_block_is_t2_positions_in_both_arms() == 0);
    SYNTH_TEST_CHECK(check_the_codec_track_opens_with_codec_bos() == 0);
    SYNTH_TEST_CHECK(check_the_flattened_streams() == 0);
    SYNTH_TEST_CHECK(check_the_block_is_appended_after_the_prefix() == 0);
    SYNTH_TEST_CHECK(check_the_x_vector_path_is_unchanged() == 0);
    SYNTH_TEST_CHECK(check_malformed_references_are_refused() == 0);
    SYNTH_TEST_CHECK(check_the_graph_sums_sixteen_in_group_order() == 0);
    SYNTH_TEST_CHECK(check_the_x_vector_still_substitutes_under_icl() == 0);
    SYNTH_TEST_CHECK(check_the_sum_refuses_a_strided_grid() == 0);
    SYNTH_TEST_CHECK(check_the_graph_refuses_a_misplaced_acoustic_block() == 0);
    std::printf("qwen3-tts icl prompt: ok\n");
    return 0;
}
