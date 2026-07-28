#pragma once

#include "synthesize.h"
#include "text-frontend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_backend_device;

namespace synth::qwen3tts {

struct ModelInfo {
    std::string family = "qwen3-tts";
    std::string variant;
    std::string quantization_profile;
    uint32_t    architecture_version = 0;

    synth_input_flags_t input_flags          = 0;
    uint32_t            capability_flags     = 0;
    uint32_t            output_sample_rate   = 0;
    uint32_t            output_channel_count = 0;
    uint64_t            max_input_tokens     = 0;
    uint64_t            max_output_frames    = 0;
    float               min_speaking_rate    = 0.0f;
    float               max_speaking_rate    = 0.0f;

    bool                     has_package_default = false;
    std::vector<std::string> preset_voice_ids;
    std::vector<std::string> language_names;

    bool        frontend_present = false;
    std::string frontend_provider;
};

// One synthesis. `codes` is kept because the Port Validation Contract compares
// and replays them: they are the seam between the autoregressive half, which
// draws from a random stream, and the codec, which is deterministic.
struct SynthesisOutput {
    uint64_t             frame_count = 0;
    std::vector<int32_t> codes;  // [frame_count, code_group_count], level-major
    std::vector<float>   audio;  // frame_count * hop_length samples, one channel
};

struct SynthesisRequest {
    std::vector<int32_t> token_ids;  // the tokenized assistant turn
    std::string          voice_id;
    std::string          language;  // "auto" selects the no-think prompt
    uint64_t             seed        = 0;
    bool                 sample      = true;
    float                temperature = 0.9f;
    uint32_t             top_k       = 50;
    float                top_p       = 1.0f;
    int                  threads     = 0;
};

class Model {
  public:
    static synth_status_t load_cpu(const std::string & path, std::unique_ptr<Model> & output);
    static synth_status_t load(const std::string &      path,
                               ggml_backend_device *    primary_device,
                               bool                     include_accelerators,
                               std::unique_ptr<Model> & output);

    ~Model();
    Model(const Model &)             = delete;
    Model & operator=(const Model &) = delete;

    synth_status_t                      get_info(ModelInfo & output) const;
    ggml_backend_device *               primary_device() const;
    std::shared_ptr<const TextFrontend> text_frontend() const;

    // Wraps text in the assistant turn the reference uses and tokenizes it. The
    // template is a fixed string rather than something the package carries, so it
    // lives with the family rather than with the caller.
    synth_status_t tokenize_request(const std::string & text, std::vector<int32_t> & token_ids) const;

    // Resolves a preset Voice and the codec language token for a request. A
    // speaker carrying a dialect override wins over the requested language.
    synth_status_t resolve_voice(const std::string & voice_id,
                                 const std::string & language,
                                 uint32_t &          speaker_token,
                                 bool &              has_language,
                                 uint32_t &          language_token) const;

    synth_status_t run_synthesis(const SynthesisRequest & request, SynthesisOutput & output) const;

    // Turns a finished code stream into audio. Split out because the Port
    // Validation Contract replays codes: the autoregressive half draws from a
    // random stream and the codec does not, so this is the deterministic side of
    // the seam and is compared on its own.
    synth_status_t decode_codes(const std::vector<int32_t> & codes,
                                uint64_t                     frame_count,
                                int                          threads,
                                std::vector<float> &         audio) const;

  private:
    struct Impl;
    explicit Model(std::unique_ptr<Impl> implementation);
    std::unique_ptr<Impl> implementation_;
};

}  // namespace synth::qwen3tts
