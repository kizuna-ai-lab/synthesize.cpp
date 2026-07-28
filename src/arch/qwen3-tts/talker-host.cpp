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
    if (request.has_speaker) {
        codec.push_back(request.speaker_token);
    }
    codec.push_back(tokens.codec_pad);
    codec.push_back(tokens.codec_bos);

    out.positions.reserve(request.role_tokens.size() + codec.size());
    for (uint32_t token : request.role_tokens) {
        out.positions.push_back(text_only(TalkerInputPosition::Text::Token, token));
    }

    // The text stream beside the codec stream is padding all the way to its last
    // entry, which is tts_bos. It is one shorter than the codec stream, because
    // the codec stream's final token belongs to the position after it.
    for (size_t index = 0; index + 1 < codec.size(); ++index) {
        const bool last = index + 2 == codec.size();
        out.positions.push_back(
            paired(last ? TalkerInputPosition::Text::TtsBos : TalkerInputPosition::Text::TtsPad, codec[index]));
    }
    // The first token of the text shares its position with codec_bos. Everything
    // after it is handed to the decode loop one step at a time.
    out.positions.push_back(
        paired(TalkerInputPosition::Text::Token, codec.back(), request.text_tokens.front()));

    out.trailing.reserve(request.text_tokens.size());
    for (size_t index = 1; index < request.text_tokens.size(); ++index) {
        out.trailing.push_back(text_only(TalkerInputPosition::Text::Token, request.text_tokens[index]));
    }
    out.trailing.push_back(text_only(TalkerInputPosition::Text::TtsEos));
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
