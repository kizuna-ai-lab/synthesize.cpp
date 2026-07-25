#pragma once

#include <cstdint>

struct ggml_context;
struct ggml_tensor;

namespace synth::kokoro {

struct AdaINResBlock1Weights;
struct GeneratorWeights;
struct HParams;

// Applies one AdaINResBlock1, the generator's residual block.
//
// This is a different block from the decoder's AdainResBlk1d even though the
// names nearly collide upstream. Here each of the three branches is AdaIN,
// Snake, dilated convolution, AdaIN, Snake, convolution, added straight back
// onto the running value with no shortcut projection and no variance scaling.
ggml_tensor * build_adain_resblock1(ggml_context *                context,
                                    ggml_tensor *                 input,
                                    ggml_tensor *                 style,
                                    const AdaINResBlock1Weights & weights,
                                    const uint32_t *              dilations,
                                    uint32_t                      dilation_count,
                                    uint32_t                      kernel_size,
                                    float                         epsilon);

// Builds the iSTFTNet generator body, from the decoder's features to the
// magnitude and phase the inverse transform consumes.
//
// `har` is the harmonic source's spectrum from the host seam, laid out
// [2 * bins, frames] with magnitudes first. The generator's own noise
// convolutions resample it to each upsampling stage's rate.
//
// Returns the [n_fft + 2, frames] pre-activation. The caller splits it into an
// exponentiated magnitude half and a sine phase half; keeping that split
// outside the builder lets the host apply it next to the inverse transform.
ggml_tensor * build_generator(ggml_context *           context,
                              ggml_tensor *            input,
                              ggml_tensor *            style,
                              ggml_tensor *            har,
                              const GeneratorWeights & weights,
                              const HParams &          hparams);

// Frames the generator produces for a decoder feature length, which is also the
// number of short-time frames the inverse transform receives.
uint64_t generator_output_frames(const HParams & hparams, uint64_t length);

}  // namespace synth::kokoro
