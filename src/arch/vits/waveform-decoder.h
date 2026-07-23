#pragma once

#include <cstdint>

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

namespace synth::vits {

struct DecoderWeights;
struct HParams;
struct VoiceWeights;

struct WaveformDecoderGraph {
    ggml_cgraph * graph         = nullptr;
    ggml_tensor * speaker_index = nullptr;
    ggml_tensor * z             = nullptr;
    ggml_tensor * pcm           = nullptr;
};

WaveformDecoderGraph build_waveform_decoder_graph(ggml_context *         context,
                                                  const DecoderWeights & weights,
                                                  const HParams &        hparams,
                                                  int64_t                frame_count);
WaveformDecoderGraph build_waveform_decoder_graph(ggml_context *         context,
                                                  const DecoderWeights & weights,
                                                  const VoiceWeights &   voice_weights,
                                                  const HParams &        hparams,
                                                  uint32_t               speaker_index,
                                                  int64_t                frame_count);

}  // namespace synth::vits
