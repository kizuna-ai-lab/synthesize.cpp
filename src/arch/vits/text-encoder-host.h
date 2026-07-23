#pragma once

#include "synthesize.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace synth::vits {

struct HParams;
struct TextEncoderOutput;

struct PreparedTextEncoderInput {
    int64_t              token_count = 0;
    std::vector<int32_t> relative_indices;
};

synth_status_t prepare_text_encoder_input(const HParams &              hparams,
                                          const std::vector<int32_t> & token_ids,
                                          int                          threads,
                                          PreparedTextEncoderInput &   output);

synth_status_t finalize_text_encoder_output(const HParams &            hparams,
                                            size_t                     token_count,
                                            const std::vector<float> & m_p_channel_major,
                                            const std::vector<float> & logs_p_channel_major,
                                            TextEncoderOutput &        output);

}  // namespace synth::vits
