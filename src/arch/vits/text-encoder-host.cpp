#include "text-encoder-host.h"

#include "vits.h"
#include "weights.h"

#include <limits>

namespace synth::vits {

synth_status_t prepare_text_encoder_input(const HParams &              hparams,
                                          const std::vector<int32_t> & token_ids,
                                          int                          threads,
                                          PreparedTextEncoderInput &   output) {
    output = {};
    if (threads <= 0 || token_ids.empty()) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (hparams.vocab_size == 0 || hparams.max_input_tokens == 0 || hparams.text_attention_window == 0 ||
        hparams.text_attention_window > static_cast<uint32_t>((std::numeric_limits<int32_t>::max() - 1) / 2)) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (token_ids.size() > hparams.max_input_tokens) {
        return SYNTH_ERR_INPUT_TOO_LONG;
    }
    for (int32_t token : token_ids) {
        if (token < 0 || static_cast<uint32_t>(token) >= hparams.vocab_size) {
            return SYNTH_ERR_INVALID_ARG;
        }
    }
    if (token_ids.size() > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        return SYNTH_ERR_INPUT_TOO_LONG;
    }
    if (token_ids.size() > std::numeric_limits<size_t>::max() / token_ids.size()) {
        return SYNTH_ERR_INPUT_TOO_LONG;
    }

    const int64_t token_count = static_cast<int64_t>(token_ids.size());
    const int32_t zero_row    = static_cast<int32_t>(2ULL * hparams.text_attention_window + 1ULL);
    output.relative_indices.resize(token_ids.size() * token_ids.size());
    for (int64_t query = 0; query < token_count; ++query) {
        for (int64_t key = 0; key < token_count; ++key) {
            const int64_t relative = key - query;
            int32_t       row      = zero_row;
            if (relative >= -static_cast<int64_t>(hparams.text_attention_window) &&
                relative <= static_cast<int64_t>(hparams.text_attention_window)) {
                row = static_cast<int32_t>(relative + hparams.text_attention_window);
            }
            output.relative_indices[static_cast<size_t>(key + token_count * query)] = row;
        }
    }
    output.token_count = token_count;
    return SYNTH_OK;
}

synth_status_t finalize_text_encoder_output(const HParams &            hparams,
                                            size_t                     token_count,
                                            const std::vector<float> & m_p_channel_major,
                                            const std::vector<float> & logs_p_channel_major,
                                            TextEncoderOutput &        output) {
    output = {};
    if (token_count == 0 || hparams.inter_channels == 0 ||
        token_count > std::numeric_limits<size_t>::max() / hparams.inter_channels) {
        return SYNTH_ERR_INVALID_ARG;
    }
    const size_t elements = static_cast<size_t>(hparams.inter_channels) * token_count;
    if (m_p_channel_major.size() != elements || logs_p_channel_major.size() != elements) {
        return SYNTH_ERR_INTERNAL;
    }

    output.channels    = hparams.inter_channels;
    output.token_count = token_count;
    output.m_p.resize(elements);
    output.logs_p.resize(elements);
    output.mask.assign(token_count, 1.0f);
    for (size_t channel = 0; channel < hparams.inter_channels; ++channel) {
        for (size_t token = 0; token < token_count; ++token) {
            const size_t source        = channel + static_cast<size_t>(hparams.inter_channels) * token;
            const size_t destination   = token + token_count * channel;
            output.m_p[destination]    = m_p_channel_major[source];
            output.logs_p[destination] = logs_p_channel_major[source];
        }
    }
    return SYNTH_OK;
}

}  // namespace synth::vits
