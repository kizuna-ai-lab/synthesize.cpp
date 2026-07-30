#pragma once

#include "synthesize.h"
#include "text-frontend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_backend_device;

namespace synth::omnivoice {

struct ModelInfo {
    std::string family = "omnivoice";
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

    // Auto-voice is the unnamed package default; the Preset Voice Catalog is
    // deliberately empty. Voice identity arrives through profiles.
    bool                     has_package_default = true;
    std::vector<std::string> language_tags;  // already BCP-47; no name bridge

    bool        frontend_present = false;
    std::string frontend_provider;
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
    // The device the Loaded Model actually holds, which the public interface
    // publishes. Present in Plan 1 even though every graph is Plan 2's: a model
    // that loads and cannot say where it lives fails a documented query.
    ggml_backend_device *               primary_device() const;
    std::shared_ptr<const TextFrontend> text_frontend() const;

    // The frame geometry the core reports, and the vocabulary the frontend's
    // ids index — the text tower's, which the core range-checks against.
    uint32_t samples_per_frame() const;  // the codec hop: 960
    uint32_t text_vocab_size() const;

  private:
    struct Impl;
    explicit Model(std::unique_ptr<Impl> implementation);
    std::unique_ptr<Impl> implementation_;
};

}  // namespace synth::omnivoice
