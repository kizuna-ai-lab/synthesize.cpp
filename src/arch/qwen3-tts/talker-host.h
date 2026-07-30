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
struct TalkerInputPosition {
    enum class Text : uint8_t {
        None,
        Token,
        TtsBos,
        TtsEos,
        TtsPad,
    };

    Text     text        = Text::None;
    uint32_t text_token  = 0;  // meaningful when text == Text::Token
    bool     has_codec   = false;
    uint32_t codec_token = 0;
};

// The talker's prefill, plus what each later step adds to its frame's codes.
struct TalkerPrompt {
    std::vector<TalkerInputPosition> positions;
    // One entry per decode step, consumed in order. A step past the end adds the
    // projected tts_pad embedding instead, which is what lets the talker keep
    // emitting frames after the text has run out.
    std::vector<TalkerInputPosition> trailing;
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
    bool                  has_language   = false;
    uint32_t              language_token = 0;
    // Absent when no preset Voice was selected. A speaker is a codec-vocabulary
    // token, not an embedding, which is why it sits in this stream.
    bool                  has_speaker    = false;
    uint32_t              speaker_token  = 0;
};

// Lays out the prefill and the trailing schedule.
//
// The layout is easy to get subtly wrong and impossible to notice afterwards:
// the codec stream is one longer than the text stream beside it, the last codec
// token pairs with the first text token rather than with a pad, and the text
// stream's single tts_bos sits at the position before that. A prompt off by one
// still synthesizes speech, in the wrong voice or the wrong language.
//
// Returns SYNTH_ERR_INVALID_ARG for a request with no text to speak.
synth_status_t build_talker_prompt(const HParams & hparams, const TalkerPromptRequest & request, TalkerPrompt & out);

// Flattens a layout into the two token streams the graph reads, resolving each
// special to its text-vocabulary id.
//
// Also checks what the graph then relies on: every position carries a text token,
// and the positions carrying a codec token are a contiguous run ending at the
// last position. `codec_offset` is where that run starts. Checking rather than
// assuming it is what lets the graph add one tensor into the tail of another
// instead of scattering rows.
synth_status_t flatten_talker_prompt(const HParams &        hparams,
                                     const TalkerPrompt &   prompt,
                                     std::vector<int32_t> & text_tokens,
                                     std::vector<int32_t> & codec_tokens,
                                     int64_t &              codec_offset);

// The text token a decode step contributes on top of its frame's codes: the
// schedule's entry while it lasts, then tts_pad forever.
uint32_t talker_step_text_token(const HParams & hparams, const TalkerPrompt & prompt, uint64_t step);

// A frame whose semantic code is the codec end token ends the utterance, and the
// frame itself is not part of it.
bool talker_frame_ends_utterance(const HParams & hparams, uint32_t semantic_code);

}  // namespace synth::qwen3tts
