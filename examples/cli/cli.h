#pragma once

#include "synthesize.h"

#include <cstdint>
#include <string>
#include <vector>

namespace synth_cli {

struct Options {
    bool                    show_help         = false;
    std::string             model_path;
    std::string             output_path;
    synth_input_kind_t      input_kind        = SYNTH_INPUT_TEXT_UTF8;
    std::string             linguistic_input;
    std::vector<int32_t>    token_ids;
    std::string             language_tag;
    std::string             voice_id;
    synth_backend_request_t backend           = SYNTH_BACKEND_AUTO;
    int32_t                 device_index      = -1;
    uint64_t                seed              = 0;
    float                   speaking_rate     = 1.0f;
    uint64_t                max_output_frames = 0;
};

bool parse_arguments(int argc, const char * const * argv, Options & output, std::string & error);

const char * usage_text();

bool write_f32_wav(const std::string & path,
                   const float *       samples,
                   uint64_t            frame_count,
                   uint32_t            sample_rate,
                   uint32_t            channel_count,
                   std::string &       error);

}  // namespace synth_cli
