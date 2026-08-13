#pragma once

#include "synthesize.h"

#include <cstdint>
#include <vector>

namespace synth::qwen3tts {

struct HParams;

// What one talker input position is made of.
//
// Two towers meet at every position and are summed. The text side is a token
// embedded through the text tower and brought down to the talker's width by the
// text projection, or one of three projected specials. The codec side is a row of
// the talker's own codec embedding. Most prompt positions carry both; the role
// prefix carries only text, and nothing carries only codec.
//
// A transcript-assisted (ICL) reference position carries a whole FRAME on its
// codec side rather than a single token: `code_group_count` codes, summed.
// Group 0 reads the talker's own codec embedding -- the same table an ordinary
// codec token reads -- so it stays in `codec_token`, and only groups 1..15 need
// somewhere new to live. That is why widening this struct left the x-vector
// path, `codec_offset` and the contiguous-run invariant untouched: to every
// existing consumer an ICL position still looks like one codec token.
struct TalkerInputPosition {
    enum class Text : uint8_t {
        None,
        Token,
        TtsBos,
        TtsEos,
        TtsPad,
    };

    Text                  text        = Text::None;
    uint32_t              text_token  = 0;  // meaningful when text == Text::Token
    bool                  has_codec   = false;
    uint32_t              codec_token = 0;  // group 0, through the talker's codec embedding
    // Groups 1..code_group_count-1 of a reference frame, IN GROUP ORDER, each
    // read from the code predictor's own table for that group. Empty at every
    // position of a non-ICL prompt and at the ICL block's opening codec_bos
    // position, which is a lone token like any other.
    std::vector<uint32_t> acoustic_codes;
};

// The talker's prefill, plus what each later step adds to its frame's codes.
struct TalkerPrompt {
    std::vector<TalkerInputPosition> positions;
    // One entry per decode step, consumed in order. A step past the end adds the
    // projected tts_pad embedding instead, which is what lets the talker keep
    // emitting frames after the text has run out.
    std::vector<TalkerInputPosition> trailing;
    // Index into the flattened codec run whose embedding row the x-vector
    // replaces, or -1 when the speaker is an ordinary codec token. It is an
    // index into `codec_tokens`, not into `positions`: the codec run starts at
    // `codec_offset` and the graph adds it as a tail.
    int64_t                          external_speaker_index = -1;
};

// What the caller asks for. The text is already tokenized: the chat template and
// the byte-pair merges belong to the text frontend, not here.
struct TalkerPromptRequest {
    // The chat-template prefix that opens the assistant turn. The reference's is
    // three tokens, `<|im_start|>assistant\n`, but the count is the template's
    // business and this takes whatever it is given.
    std::vector<uint32_t> role_tokens;
    // The text to speak, which must not be empty: its first token shares a
    // position with the codec stream's opening and the rest becomes `trailing`.
    std::vector<uint32_t> text_tokens;
    // Absent selects the reference's no-think path, which emits no language token
    // at all rather than a default one.
    bool                  has_language        = false;
    uint32_t              language_token      = 0;
    // Absent when no preset Voice was selected. A speaker is a codec-vocabulary
    // token, not an embedding, which is why it sits in this stream.
    bool                  has_speaker         = false;
    uint32_t              speaker_token       = 0;
    // The speaker slot's embedding comes from a prepared Voice Profile rather
    // than from the codec vocabulary. The POSITION is unchanged -- upstream
    // substitutes the embedding, it does not move or remove the slot -- so
    // `has_speaker` must also be set and `speaker_token` is written into the
    // stream as an inert placeholder whose row the graph never reads.
    bool                  speaker_is_external = false;

    // Transcript-assisted (In-Context Learning) cloning. The three fields move
    // together: absent, the prompt is exactly what it was before this existed.
    //
    // `reference_codes` is the reference audio's code grid in the layout
    // CodecEncoding settled -- `code_group_count * reference_frames` values,
    // GROUP-FASTEST, so frame `f`'s sixteen codes are contiguous at
    // `f * code_group_count`. There is no transpose at this boundary and none
    // is wanted: a stage-major grid has the right element count, the right
    // value range and the wrong order.
    //
    // `reference_text_tokens` is what upstream calls `ref_id` -- the reference
    // transcript through its own turn wrapper and slice (Task 6's
    // Model::tokenize_reference_transcript), NOT the target text.
    bool                  has_reference = false;
    std::vector<uint32_t> reference_text_tokens;
    std::vector<int32_t>  reference_codes;
    uint64_t              reference_frames = 0;
};

// Lays out the prefill and the trailing schedule.
//
// The layout is easy to get subtly wrong and impossible to notice afterwards:
// the codec stream is one longer than the text stream beside it, the last codec
// token pairs with the first text token rather than with a pad, and the text
// stream's single tts_bos sits at the position before that. A prompt off by one
// still synthesizes speech, in the wrong voice or the wrong language.
//
// With `request.has_reference`, the tail after the shared prefix is instead the
// two-track ICL block (modeling_qwen3_tts.py:1968-2019 `generate_icl_prompt`,
// concatenated at :2197 in place of the single first-text-token position the
// non-ICL branch appends at :2200-2202). Two tracks of different lengths are
// summed position by position:
//
//   T1 = reference_text_tokens + text_tokens + 1 (a closing tts_eos)
//   T2 = 1 + reference_frames  (a leading codec_bos)
//
// and the block is T2 positions in BOTH arms, so its length never says which
// arm ran -- only the trailing schedule does:
//
//   T1 >  T2  "truncate": the block takes the text track's first T2 entries and
//             the trailing schedule is the remaining T1 - T2, tts_eos last.
//   T1 <= T2  "pad":      the text track is padded to T2 with tts_pad and the
//             trailing schedule is a single tts_pad.
//
// Returns SYNTH_ERR_INVALID_ARG for a request with no text to speak, or for a
// reference whose code grid is not `code_group_count * reference_frames`
// non-negative values with at least one frame.
synth_status_t build_talker_prompt(const HParams & hparams, const TalkerPromptRequest & request, TalkerPrompt & out);

// Flattens a layout into the token streams the graph reads, resolving each
// special to its text-vocabulary id.
//
// Also checks what the graph then relies on: every position carries a text token,
// and the positions carrying a codec token are a contiguous run ending at the
// last position. `codec_offset` is where that run starts. Checking rather than
// assuming it is what lets the graph add one tensor into the tail of another
// instead of scattering rows.
//
// `acoustic_codes` is the same for the ICL block's groups 1..15: the positions
// carrying acoustic codes must themselves be a contiguous run ending at the last
// position, and `acoustic_offset` is where it starts (-1 when there is none).
// The layout is GROUP-MAJOR -- `acoustic_codes[group * frames + frame]`, i.e.
// GGML `[frames, code_group_count - 1]` with the frame index fastest -- because
// that is what lets sum_code_embeddings take each group's ids as a contiguous
// 1-D view. `ggml_get_rows` cannot read a strided index tensor, so the choice is
// this gather or a copy in the graph, and the gather is the one a test can see.
synth_status_t flatten_talker_prompt(const HParams &        hparams,
                                     const TalkerPrompt &   prompt,
                                     std::vector<int32_t> & text_tokens,
                                     std::vector<int32_t> & codec_tokens,
                                     int64_t &              codec_offset,
                                     std::vector<int32_t> & acoustic_codes,
                                     int64_t &              acoustic_offset);

// The text token a decode step contributes on top of its frame's codes: the
// schedule's entry while it lasts, then tts_pad forever.
uint32_t talker_step_text_token(const HParams & hparams, const TalkerPrompt & prompt, uint64_t step);

// A frame whose semantic code is the codec end token ends the utterance, and the
// frame itself is not part of it.
bool talker_frame_ends_utterance(const HParams & hparams, uint32_t semantic_code);

}  // namespace synth::qwen3tts
