// The talker's prompt layout.
//
// This is a host seam because it is all discrete: token ids and which embedding
// table each position reads. Nothing here is arithmetic that a backend could
// disagree about, and everything here is a decision that changes the voice, the
// language or the timing if it lands one position off.

#include "talker-host.h"

#include "weights.h"

#include <cstddef>

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

// Checked before anything is built, so a malformed reference leaves `out` as
// the caller found it rather than half a prompt.
bool reference_is_well_formed(const HParams & hparams, const TalkerPromptRequest & request) {
    const size_t groups = hparams.talker.code_group_count;
    if (groups < 2 || request.reference_frames == 0) {
        return false;
    }
    if (request.reference_codes.size() != size_t(request.reference_frames) * groups) {
        return false;
    }
    // Every code becomes a ggml_get_rows index, and that op asserts
    // `i01 >= 0 && i01 < ne01` (ggml/src/ggml-cpu/ops.cpp:4779) -- it ABORTS
    // the process on either side of the range, so both sides have to be
    // refused here or a malformed reference stops being a status the caller
    // can map. The two bounds differ because the two sides of a frame read
    // different tables: group 0 reads the talker's codec embedding, groups
    // 1..15 the code predictor's.
    const int64_t group_0_rows  = int64_t(hparams.talker.codec_vocab_size);
    const int64_t acoustic_rows = int64_t(hparams.code_predictor.vocab_size);
    if (group_0_rows <= 0 || acoustic_rows <= 0) {
        return false;
    }
    for (size_t index = 0; index < request.reference_codes.size(); ++index) {
        const int32_t code = request.reference_codes[index];
        const int64_t rows = index % groups == 0 ? group_0_rows : acoustic_rows;
        if (code < 0 || int64_t(code) >= rows) {
            return false;
        }
    }
    return true;
}

// The reference-conditioned tail: `generate_icl_prompt`
// (modeling_qwen3_tts.py:1968-2019), concatenated after the shared prefix at
// :2197 in place of the single first-text-token position the non-ICL branch
// appends at :2200-2202. Preconditions are reference_is_well_formed's.
//
// The two tracks are built at their own natural lengths and then aligned. The
// text track is T1 = ref ids + target ids + one tts_eos; the codec track is
// T2 = one codec_bos + one position per reference frame. Neither is a prefix
// of the other, and the block that comes out is T2 positions whichever arm
// runs -- so the block's length cannot say which one did. The trailing
// schedule can, and does.
void append_icl_block(const HParams &             hparams,
                      const TalkerPromptRequest & request,
                      uint32_t                    codec_bos,
                      TalkerPrompt &              out) {
    const size_t groups = hparams.talker.code_group_count;

    // The text track, at its own length. `tts_eos` closes it -- upstream
    // concatenates tts_eos_embed onto the projected ids at :1981, so the
    // closing position is part of T1 rather than something appended after the
    // alignment.
    std::vector<TalkerInputPosition> track;
    track.reserve(request.reference_text_tokens.size() + request.text_tokens.size() + 1);
    for (uint32_t token : request.reference_text_tokens) {
        track.push_back(text_only(TalkerInputPosition::Text::Token, token));
    }
    for (uint32_t token : request.text_tokens) {
        track.push_back(text_only(TalkerInputPosition::Text::Token, token));
    }
    track.push_back(text_only(TalkerInputPosition::Text::TtsEos));

    const size_t text_lens  = track.size();
    const size_t codec_lens = size_t(request.reference_frames) + 1;

    for (size_t index = 0; index < codec_lens; ++index) {
        // Past the text track the pad arm keeps going with tts_pad; the
        // truncate arm never gets here, because it has more text than block.
        TalkerInputPosition position = index < text_lens ? track[index] : text_only(TalkerInputPosition::Text::TtsPad);
        position.has_codec           = true;
        if (index == 0) {
            // codec_bos opens the codec track through the TALKER's own codec
            // embedding (:1990-1998) -- one token, not a frame.
            position.codec_token = codec_bos;
        } else {
            const int32_t * frame = request.reference_codes.data() + (index - 1) * groups;
            position.codec_token  = uint32_t(frame[0]);
            position.acoustic_codes.resize(groups - 1);
            for (size_t group = 1; group < groups; ++group) {
                position.acoustic_codes[group - 1] = uint32_t(frame[group]);
            }
        }
        out.positions.push_back(position);
    }

    if (text_lens > codec_lens) {
        // "truncate" (:2016): what the block could not take becomes the decode
        // loop's schedule, in order, tts_eos last.
        out.trailing.assign(track.begin() + std::ptrdiff_t(codec_lens), track.end());
    } else {
        // "pad" (:2018-2019): a single bare tts_pad. talker_step_text_token
        // already pads past the end of the schedule, so this entry changes no
        // behaviour -- it is materialized because it is the only thing that
        // distinguishes the two arms after the fact.
        out.trailing.push_back(text_only(TalkerInputPosition::Text::TtsPad));
    }
}

}  // namespace

synth_status_t build_talker_prompt(const HParams & hparams, const TalkerPromptRequest & request, TalkerPrompt & out) {
    out = TalkerPrompt{};
    if (request.text_tokens.empty()) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (request.has_reference && !reference_is_well_formed(hparams, request)) {
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
    out.positions.reserve(request.role_tokens.size() + codec.size() + request.text_tokens.size() +
                          size_t(request.reference_frames) + 2);
    for (uint32_t token : request.role_tokens) {
        out.positions.push_back(text_only(TalkerInputPosition::Text::Token, token));
    }

    // The text stream beside the codec stream is padding all the way to its last
    // entry, which is tts_bos. Only the codec stream's final token, codec_bos,
    // belongs to a position after this block -- tts_bos pairs with codec_pad.
    // Both modes share this prefix exactly; upstream builds it once, before it
    // knows which branch it is in (modeling_qwen3_tts.py:2182-2186).
    for (size_t index = 0; index + 1 < codec.size(); ++index) {
        const bool last = index + 2 == codec.size();
        out.positions.push_back(
            paired(last ? TalkerInputPosition::Text::TtsBos : TalkerInputPosition::Text::TtsPad, codec[index]));
    }

    if (request.has_reference) {
        append_icl_block(hparams, request, codec.back(), out);
        out.external_speaker_index = speaker_codec_index;
        return SYNTH_OK;
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
                                     int64_t &              codec_offset,
                                     std::vector<int32_t> & acoustic_codes,
                                     int64_t &              acoustic_offset) {
    text_tokens.clear();
    codec_tokens.clear();
    acoustic_codes.clear();
    codec_offset    = 0;
    acoustic_offset = -1;
    if (prompt.positions.empty()) {
        return SYNTH_ERR_INVALID_ARG;
    }
    // Zero when the package declares fewer than two code groups, which makes
    // the size check below refuse any position that carries acoustic codes --
    // a package/layout disagreement, not something to drop quietly.
    const size_t acoustic_groups =
        hparams.talker.code_group_count >= 2 ? size_t(hparams.talker.code_group_count) - 1 : 0;

    text_tokens.reserve(prompt.positions.size());
    codec_tokens.reserve(prompt.positions.size());
    // Gathered per position, then transposed into the group-major order the
    // graph reads. `frames` counts the acoustic positions seen so far.
    std::vector<const TalkerInputPosition *> acoustic;
    bool                                     started = false;
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
        if (!position.acoustic_codes.empty()) {
            // Groups 1..15 sit on top of group 0; a position carrying them
            // without one is a frame with no semantic code.
            if (!position.has_codec || position.acoustic_codes.size() != acoustic_groups) {
                return SYNTH_ERR_INVALID_ARG;
            }
            if (acoustic.empty()) {
                acoustic_offset = static_cast<int64_t>(index);
            } else if (index != size_t(acoustic_offset) + acoustic.size()) {
                // The graph adds the summed acoustic embeddings as a tail too,
                // so a gap here has the same consequence a codec gap has.
                return SYNTH_ERR_INVALID_ARG;
            }
            acoustic.push_back(&position);
        }
    }
    if (!started) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (!acoustic.empty() && size_t(acoustic_offset) + acoustic.size() != prompt.positions.size()) {
        return SYNTH_ERR_INVALID_ARG;
    }

    // Group-major: all `frames` ids of group 1, then all of group 2, ... Each
    // group's run is then a contiguous 1-D view for ggml_get_rows.
    const size_t frames = acoustic.size();
    acoustic_codes.resize(frames * acoustic_groups);
    for (size_t group = 0; group < acoustic_groups; ++group) {
        for (size_t frame = 0; frame < frames; ++frame) {
            acoustic_codes[group * frames + frame] = static_cast<int32_t>(acoustic[frame]->acoustic_codes[group]);
        }
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
