#pragma once

#include "synthesize.h"
#include "text-frontend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_backend_device;

namespace synth::kokoro {

// Each stage output uses the axis order its reference probe uses, which is the
// order upstream's own tensor has at that point rather than one rule for the
// whole family. Where a comment says a shape, the last axis is the contiguous
// one, so [tokens, features] means the feature index runs fastest. The Stage 5
// validators compare these arrays against the probes directly.

struct PLBertOutput {
    uint32_t           hidden_size = 0;
    uint64_t           token_count = 0;
    std::vector<float> hidden;     // [token_count, hidden_size]
    std::vector<float> projected;  // [hidden_dim, token_count], the encoder projection
};

struct DurationOutput {
    uint64_t             token_count = 0;
    uint64_t             frame_count = 0;
    std::vector<float>   logits;     // [token_count, max_dur]
    std::vector<int64_t> durations;  // one rounded step count per token
    // [token_count, frame_count] one-hot expansion.
    std::vector<float>   alignment;
    // [token_count, hidden_dim + style_dim], the state the prosody stage expands.
    std::vector<float>   encoded;
};

struct ProsodyOutput {
    uint64_t           frame_count = 0;
    // [hidden_dim + style_dim, frame_count], the alignment-expanded state this
    // stage consumes.
    std::vector<float> expanded;
    // Both curves leave the prosody stage at twice the frame rate.
    std::vector<float> f0;
    std::vector<float> energy;
};

struct TextEncoderOutput {
    uint32_t           channels    = 0;
    uint64_t           token_count = 0;
    std::vector<float> encoded;  // [channels, token_count]
    // [channels, frame_count]: the same features after alignment expansion.
    uint64_t           frame_count = 0;
    std::vector<float> aligned;
};

struct SourceOutput {
    uint64_t           frames = 0;
    uint64_t           bins   = 0;
    std::vector<float> excitation;  // [upsampled_samples]
    std::vector<float> spectrum;    // [2 * bins, frames], magnitudes then phases
};

struct DecoderOutput {
    uint64_t           frames = 0;
    // [n_fft + 2, frames] as the graph leaves it: log magnitudes then pre-sine
    // angles, both still unapplied.
    std::vector<float> spectrum;
};

struct WaveformOutput {
    uint64_t           sample_count = 0;
    std::vector<float> pcm;
};

struct ModelInfo {
    bool                                has_package_default = false;
    std::vector<std::string>            preset_voice_ids;
    std::vector<uint32_t>               preset_voice_flags;
    std::shared_ptr<const TextFrontend> text_frontend;
    uint32_t                            input_flags          = 0;
    uint32_t                            capability_flags     = 0;
    uint32_t                            output_sample_rate   = 0;
    uint32_t                            output_channel_count = 0;
    uint32_t                            vocab_size           = 0;
    uint32_t                            samples_per_frame    = 0;
    uint64_t                            max_input_tokens     = 0;
    uint64_t                            max_output_frames    = 0;
    float                               min_speaking_rate    = 0.0f;
    float                               max_speaking_rate    = 0.0f;
};

// The two random draws the harmonic source consumes, in the order the oracle
// records them. Ordinary synthesis fills them from the project's own seeded
// stream; port validation replays the recorded tensors instead.
struct SourceRandom {
    std::vector<float> rand_ini;
    std::vector<float> noise;
};

class Model {
  public:
    static synth_status_t load_cpu(const std::string & path, std::unique_ptr<Model> & output);
    static synth_status_t load_cpu(const std::string &      path,
                                   ggml_backend_device *    device,
                                   std::unique_ptr<Model> & output);
    static synth_status_t load(const std::string &      path,
                               ggml_backend_device *    primary_device,
                               bool                     include_accelerators,
                               std::unique_ptr<Model> & output);

    synth_status_t        get_info(ModelInfo & output) const;
    ggml_backend_device * primary_device() const;

    // Resolves the Voice row from the final token count, which is the family's
    // stochastic contract: a Kokoro Voice is a table indexed by input length,
    // not a single vector. Returns false when the count has no row.
    bool resolve_voice_row(uint64_t final_token_count, uint32_t & row) const;

    // Samples the harmonic source needs for a resolved frame count, so a caller
    // can size its random draw before running any stage.
    uint64_t source_sample_count(uint64_t frame_count) const;
    uint32_t source_harmonics() const;

    ~Model();
    Model(const Model &)             = delete;
    Model & operator=(const Model &) = delete;
    Model(Model &&)                  = delete;
    Model & operator=(Model &&)      = delete;

    synth_status_t run_plbert(const std::vector<int32_t> & token_ids, int threads, PLBertOutput & output) const;

    synth_status_t run_duration(const std::vector<int32_t> & token_ids,
                                uint32_t                     voice_index,
                                uint32_t                     voice_row,
                                float                        speaking_rate,
                                int                          threads,
                                DurationOutput &             output) const;

    synth_status_t run_prosody(const std::vector<int32_t> & token_ids,
                               uint32_t                     voice_index,
                               uint32_t                     voice_row,
                               float                        speaking_rate,
                               int                          threads,
                               ProsodyOutput &              output) const;

    synth_status_t run_text_encoder(const std::vector<int32_t> & token_ids,
                                    uint32_t                     voice_index,
                                    uint32_t                     voice_row,
                                    float                        speaking_rate,
                                    int                          threads,
                                    TextEncoderOutput &          output) const;

    synth_status_t run_source(const std::vector<int32_t> & token_ids,
                              uint32_t                     voice_index,
                              uint32_t                     voice_row,
                              float                        speaking_rate,
                              const SourceRandom &         random,
                              int                          threads,
                              SourceOutput &               output) const;

    synth_status_t run_decoder(const std::vector<int32_t> & token_ids,
                               uint32_t                     voice_index,
                               uint32_t                     voice_row,
                               float                        speaking_rate,
                               const SourceRandom &         random,
                               int                          threads,
                               DecoderOutput &              output) const;

    synth_status_t run_waveform(const std::vector<int32_t> & token_ids,
                                uint32_t                     voice_index,
                                uint32_t                     voice_row,
                                float                        speaking_rate,
                                const SourceRandom &         random,
                                int                          threads,
                                WaveformOutput &             output) const;

  private:
    struct Impl;
    // Everything the staged entry points share. Each stage recomputes its
    // predecessors, which is what lets a validator drive one stage at a time
    // without the family holding synthesis state between calls.
    struct Pipeline;

    explicit Model(std::unique_ptr<Impl> implementation);

    synth_status_t compute_plbert(const std::vector<int32_t> & token_ids, int threads, Pipeline & state) const;
    synth_status_t compute_duration(const std::vector<int32_t> & token_ids,
                                    uint32_t                     voice_index,
                                    uint32_t                     voice_row,
                                    float                        speaking_rate,
                                    int                          threads,
                                    Pipeline &                   state) const;
    synth_status_t compute_prosody(const std::vector<int32_t> & token_ids,
                                   uint32_t                     voice_index,
                                   uint32_t                     voice_row,
                                   float                        speaking_rate,
                                   int                          threads,
                                   Pipeline &                   state) const;
    synth_status_t compute_text_encoder(const std::vector<int32_t> & token_ids,
                                        uint32_t                     voice_index,
                                        uint32_t                     voice_row,
                                        float                        speaking_rate,
                                        int                          threads,
                                        Pipeline &                   state) const;
    synth_status_t compute_source(const std::vector<int32_t> & token_ids,
                                  uint32_t                     voice_index,
                                  uint32_t                     voice_row,
                                  float                        speaking_rate,
                                  const SourceRandom &         random,
                                  int                          threads,
                                  Pipeline &                   state) const;
    synth_status_t compute_decoder(const std::vector<int32_t> & token_ids,
                                   uint32_t                     voice_index,
                                   uint32_t                     voice_row,
                                   float                        speaking_rate,
                                   const SourceRandom &         random,
                                   int                          threads,
                                   Pipeline &                   state,
                                   std::vector<float> &         spectrum) const;

    std::unique_ptr<Impl> implementation_;
};

}  // namespace synth::kokoro
