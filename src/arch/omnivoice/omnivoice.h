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

// A family-internal synthesis request. Plan 2 reaches this through the replay
// runner only: the public seam stays on the synthesis.not_implemented stub
// until Plan 3 lands the sampled path the public defaults select. The prompt
// arrives as ids (the oracle's input/token_ids.i32, or Plan 3's own frontend
// output) and the canvas length arrives fixed -- the duration estimator's
// output, or the oracle grid's frame count under replay.
struct SynthesisRequest {
    // Row 0's text region: style markers, language/instruct slots and the
    // wrapped text, already tokenized. Every codebook row repeats these ids.
    std::vector<int32_t>  prompt_text_ids;
    // Reference audio tokens, codebook-major [num_codebooks * frames]; empty
    // for auto-voice and voice-design requests. Plan 3's cloning encoder
    // produces these from audio; Plan 2's runner replays the oracle's.
    std::vector<int32_t>  reference_tokens;
    uint64_t              target_frames = 0;
    uint32_t              num_step      = 0;  // 0 = the package's embedded default
    // Stop after the step-0 conditional forward with the probe buffers filled;
    // the sampled golden cases compare only that forward in Plan 2.
    bool                  probe_only    = false;
    int                   threads       = 0;  // 0 = default_synthesis_threads()
    std::vector<uint32_t> probe_layers;       // layer indices probed at step 0
};

struct SynthesisOutput {
    uint64_t             frame_count = 0;
    // The committed grid, codebook-major [num_codebooks * frame_count] --
    // codebook c, frame t at c * frame_count + t, the oracle's codes/grid.i32
    // layout exactly.
    std::vector<int32_t> codes;
    // The decoded waveform with the no-reference volume branch applied
    // (peak-normalise-to-0.5), which is what the oracle returns to its caller.
    std::vector<float>   audio;

    // Step-0 conditional probes, in ggml read-back order; the runner reorders
    // logits into the oracle's [C, S, V] layout on write.
    std::vector<float>              logits_step0;  // [positions][codebooks][vocab]
    std::vector<float>              final_hidden;  // [positions][hidden]
    std::vector<std::vector<float>> layer_hidden;  // one per requested layer

    struct StagePlacement {
        uint64_t nodes             = 0;
        uint64_t accelerator_nodes = 0;
    };

    double         generator_seconds       = 0.0;
    double         generator_setup_seconds = 0.0;
    double         codec_seconds           = 0.0;
    StagePlacement generator_placement;
    StagePlacement codec_placement;
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

    // The greedy mask-predict synthesis path. Deterministic: with the golden
    // parameters it makes no random draw at all, which is what the exact-token
    // gate stands on. The sampled path is Plan 3.
    synth_status_t run_synthesis(const SynthesisRequest & request, SynthesisOutput & output);

    // Decodes a committed grid (codebook-major [num_codebooks * frame_count],
    // values in [0, codebook_size)) to the RAW waveform -- no volume branch,
    // so the replay seam can apply the oracle's branch per case.
    synth_status_t decode_codes(const std::vector<int32_t> & codes,
                                uint64_t                     frame_count,
                                int                          threads,
                                std::vector<float> &         audio);

  private:
    struct Impl;
    explicit Model(std::unique_ptr<Impl> implementation);
    std::unique_ptr<Impl> implementation_;
};

}  // namespace synth::omnivoice
