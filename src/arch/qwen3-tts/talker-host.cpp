// The talker's prompt layout.
//
// This is a host seam because it is all discrete: token ids and which embedding
// table each position reads. Nothing here is arithmetic that a backend could
// disagree about, and everything here is a decision that changes the voice, the
// language or the timing if it lands one position off.

#include "talker-host.h"

#include "weights.h"

namespace synth::qwen3tts {

namespace {

TalkerInputPosition text_only(TalkerInputPosition::Text kind, uint32_t token = 0) {
    TalkerInputPosition position;
    position.text       = kind;
    position.text_token = token;
    return position;
}

TalkerInputPosition paired(TalkerInputPosition::Text kind, uint32_t codec_token, uint32_t text_token = 0) {
    TalkerInputPosition position = text_only(kind, text_token);
    position.has_codec           = true;
    position.codec_token         = codec_token;
    return position;
}

}  // namespace

synth_status_t build_talker_prompt(const HParams & hparams, const TalkerPromptRequest & request, TalkerPrompt & out) {
    out = TalkerPrompt{};
    if (request.text_tokens.empty()) {
        return SYNTH_ERR_INVALID_ARG;
    }
    const SpecialTokens & tokens = hparams.tokens;

    // The codec stream, in order. Naming a language opens with think and carries
    // the language token between the think delimiters; asking for auto opens with
    // nothink and carries no language token at all, which is a shorter prompt
    // rather than the same prompt with a default.
    std::vector<uint32_t> codec;
    codec.push_back(request.has_language ? tokens.codec_think : tokens.codec_nothink);
    codec.push_back(tokens.codec_think_bos);
    if (request.has_language) {
        codec.push_back(request.language_token);
    }
    codec.push_back(tokens.codec_think_eos);
    // The index recorded here is an index into `codec`, not into `positions`,
    // and it survives unchanged into the flattened codec run: every entry
    // before codec_pad below maps 1:1, in order, onto flatten_talker_prompt's
    // output, because nothing before it is ever duplicated or skipped.
    int64_t speaker_codec_index = -1;
    if (request.has_speaker) {
        if (request.speaker_is_external) {
            speaker_codec_index = static_cast<int64_t>(codec.size());
            // codec_pad is a real codec-vocabulary token, so this position
            // still round-trips through flatten_talker_prompt like any other;
            // build_talker_prefill_input is what discards the row it produces
            // in favour of the x-vector at this same index.
            codec.push_back(tokens.codec_pad);
        } else {
            codec.push_back(request.speaker_token);
        }
    }
    codec.push_back(tokens.codec_pad);
    codec.push_back(tokens.codec_bos);

    // The whole text sits in the prefill. `generate_custom_voice` defaults to
    // the reference's non-streaming mode, which puts every text token in the
    // prompt and leaves the decode loop nothing but padding to add -- the
    // streaming layout, which feeds the text one token per frame, is a different
    // entry point this variant does not use. The two produce different prefill
    // lengths and different audio, and neither errors.
    out.positions.reserve(request.role_tokens.size() + codec.size() + request.text_tokens.size() + 2);
    for (uint32_t token : request.role_tokens) {
        out.positions.push_back(text_only(TalkerInputPosition::Text::Token, token));
    }

    // The text stream beside the codec stream is padding all the way to its last
    // entry, which is tts_bos. Only the codec stream's final token, codec_bos,
    // belongs to a position after this block -- tts_bos pairs with codec_pad.
    for (size_t index = 0; index + 1 < codec.size(); ++index) {
        const bool last = index + 2 == codec.size();
        out.positions.push_back(
            paired(last ? TalkerInputPosition::Text::TtsBos : TalkerInputPosition::Text::TtsPad, codec[index]));
    }

    // Then the text itself, every token against a codec pad, and one tts_eos
    // closing it.
    const uint32_t codec_pad = codec[codec.size() - 2];
    for (uint32_t token : request.text_tokens) {
        out.positions.push_back(paired(TalkerInputPosition::Text::Token, codec_pad, token));
    }
    out.positions.push_back(paired(TalkerInputPosition::Text::TtsEos, codec_pad));
    // codec_bos closes the prompt against a pad, not against a text token.
    out.positions.push_back(paired(TalkerInputPosition::Text::TtsPad, codec.back()));

    out.external_speaker_index = speaker_codec_index;

    // Nothing is left for the decode loop to contribute, so every step adds the
    // projected tts_pad embedding. The schedule is empty rather than holding one
    // pad, because talker_step_text_token already pads past its end.
    return SYNTH_OK;
}

namespace {

// Every special in the layout is itself a token of the text vocabulary, which is
// why the text side needs no separate path for them.
bool resolve_text_token(const HParams & hparams, const TalkerInputPosition & position, int32_t & token) {
    switch (position.text) {
        case TalkerInputPosition::Text::Token:
            token = static_cast<int32_t>(position.text_token);
            return true;
        case TalkerInputPosition::Text::TtsBos:
            token = static_cast<int32_t>(hparams.tokens.tts_bos);
            return true;
        case TalkerInputPosition::Text::TtsEos:
            token = static_cast<int32_t>(hparams.tokens.tts_eos);
            return true;
        case TalkerInputPosition::Text::TtsPad:
            token = static_cast<int32_t>(hparams.tokens.tts_pad);
            return true;
        case TalkerInputPosition::Text::None:
            break;
    }
    return false;
}

}  // namespace

synth_status_t flatten_talker_prompt(const HParams &        hparams,
                                     const TalkerPrompt &   prompt,
                                     std::vector<int32_t> & text_tokens,
                                     std::vector<int32_t> & codec_tokens,
                                     int64_t &              codec_offset) {
    text_tokens.clear();
    codec_tokens.clear();
    codec_offset = 0;
    if (prompt.positions.empty()) {
        return SYNTH_ERR_INVALID_ARG;
    }

    text_tokens.reserve(prompt.positions.size());
    codec_tokens.reserve(prompt.positions.size());
    bool started = false;
    for (size_t index = 0; index < prompt.positions.size(); ++index) {
        const TalkerInputPosition & position = prompt.positions[index];
        int32_t                     token    = 0;
        if (!resolve_text_token(hparams, position, token)) {
            return SYNTH_ERR_INVALID_ARG;
        }
        text_tokens.push_back(token);
        if (position.has_codec) {
            if (!started) {
                started      = true;
                codec_offset = static_cast<int64_t>(index);
            }
            codec_tokens.push_back(static_cast<int32_t>(position.codec_token));
        } else if (started) {
            // A gap would mean the codec stream is not a tail, and the graph adds
            // it as one.
            return SYNTH_ERR_INVALID_ARG;
        }
    }
    if (!started) {
        return SYNTH_ERR_INVALID_ARG;
    }
    return SYNTH_OK;
}

uint32_t talker_step_text_token(const HParams & hparams, const TalkerPrompt & prompt, uint64_t step) {
    if (step >= prompt.trailing.size()) {
        return hparams.tokens.tts_pad;
    }
    int32_t token = 0;
    if (!resolve_text_token(hparams, prompt.trailing[step], token)) {
        return hparams.tokens.tts_pad;
    }
    return static_cast<uint32_t>(token);
}

bool talker_frame_ends_utterance(const HParams & hparams, uint32_t semantic_code) {
    return semantic_code == hparams.tokens.codec_eos;
}

}  // namespace synth::qwen3tts
