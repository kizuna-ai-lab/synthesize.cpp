#!/usr/bin/env python3
"""Reference values for tests/omnivoice_codec_test.cpp.

Composes the Higgs decode path at toy dimensions from the REAL classes: the
transformers DacDecoder with HiggsAudioV2TokenizerModel._adjust_dac_decoder
applied (output_padding = stride % 2, tanh removed), and the RVQ/fc2 stages as
the exact F.embedding / F.linear arithmetic the reference modules perform.

The upstream decode path this mirrors is HiggsAudioV2TokenizerModel.decode:
the RVQ sum over levels, fc2 across the channel axis, then the adjusted
acoustic decoder. The two Higgs adjustments are applied by the upstream
staticmethod rather than restated here, so a transformers release that changes
either of them changes these numbers instead of hiding the drift.

Usage:
    uv run --project scripts/envs/omnivoice --locked python \
        scripts/dump_reference_omnivoice_codec.py
"""

import torch
import torch.nn.functional as F

SEED = 20260732
QUANTIZERS = 2
CODEBOOK_SIZE = 5
CODEBOOK_DIM = 3
CONCAT = 4  # RVQ width = fc2 input
ACOUSTIC = 4  # fc2 output = decoder input channels
DECODER_HIDDEN = 8
RATIOS = [2, 3]  # toy hop = 6
FRAMES = 4
CODES = [[0, 3, 1, 4], [2, 2, 0, 1]]  # level-major [QUANTIZERS][FRAMES]
HOP = RATIOS[0] * RATIOS[1]

# The three draw roles. Every scale is a power of two so the C++ twin, which
# multiplies in float where this multiplies in double, rounds identically --
# except the bias scale, whose one-ulp disagreement is 1e-8 on a 0.1 bias.
#
# The weight scale is 0.5 rather than something smaller because the decoder
# stack has enough gain that it decides the output amplitude: at 0.25 the wave
# peaks near 0.1, where tanh(x) and x agree to 4e-4 and a surviving final tanh
# would slip past the accelerator tolerance. At 0.5 the peak is near 1.8 and
# the two disagree by 0.87. Amplitude above 1 is not unrealistic either --
# removing the tanh is exactly why Higgs output needs peak normalization.
WEIGHT_SCALE = 0.5
BIAS_SCALE = 0.1
ALPHA_SCALE, ALPHA_OFFSET = 0.25, 1.0


class LcgStream:
    """The 64-bit LCG the C++ fixture mirrors; every draw is a 24-bit ratio."""

    def __init__(self, seed):
        self.state = seed

    def next(self):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) % 2**64
        return (self.state >> 40) / 8388608.0 - 1.0

    def fill(self, count, scale, offset):
        return torch.tensor(
            [self.next() * scale + offset for _ in range(count)], dtype=torch.float32
        )


def dump(name, tensor):
    flat = tensor.reshape(-1).tolist()
    print(f"constexpr float {name}[] = {{")
    for start in range(0, len(flat), 4):
        row = ", ".join(f"{value:.9g}f" for value in flat[start : start + 4])
        print(f"    {row},")
    print("};")


def main():
    from transformers.models.dac.configuration_dac import DacConfig
    from transformers.models.dac.modeling_dac import DacDecoder
    from transformers.models.higgs_audio_v2_tokenizer.modeling_higgs_audio_v2_tokenizer import (
        HiggsAudioV2TokenizerModel,
    )

    config = DacConfig(
        hidden_size=ACOUSTIC,
        decoder_hidden_size=DECODER_HIDDEN,
        upsampling_ratios=RATIOS,
    )
    decoder = DacDecoder(config).eval()
    HiggsAudioV2TokenizerModel._adjust_dac_decoder(decoder)

    # The adjustment is the whole point of dumping against the real class, so
    # assert it landed rather than trusting the call.
    assert isinstance(decoder.tanh, torch.nn.Identity), "the final tanh survived"
    for index, block in enumerate(decoder.block):
        assert block.conv_t1.output_padding == (RATIOS[index] % 2,), (
            f"block {index} output_padding {block.conv_t1.output_padding}"
        )

    stream = LcgStream(SEED)
    # RVQ + fc2 weights first, then the decoder's, module by module in forward
    # order. The C++ fixture mirrors this order line for line.
    weight = (WEIGHT_SCALE, 0.0)
    bias = (BIAS_SCALE, 0.0)
    alpha = (ALPHA_SCALE, ALPHA_OFFSET)
    codebooks, out_weights, out_biases = [], [], []
    for _ in range(QUANTIZERS):
        codebooks.append(
            stream.fill(CODEBOOK_SIZE * CODEBOOK_DIM, *weight).view(
                CODEBOOK_SIZE, CODEBOOK_DIM
            )
        )
        out_weights.append(
            stream.fill(CONCAT * CODEBOOK_DIM, *weight).view(CONCAT, CODEBOOK_DIM)
        )
        out_biases.append(stream.fill(CONCAT, *bias))
    fc2_weight = stream.fill(ACOUSTIC * CONCAT, *weight).view(ACOUSTIC, CONCAT)
    fc2_bias = stream.fill(ACOUSTIC, *bias)

    assignments = [(decoder.conv1.weight, *weight), (decoder.conv1.bias, *bias)]
    for block in decoder.block:
        assignments += [
            (block.snake1.alpha, *alpha),
            (block.conv_t1.weight, *weight),
            (block.conv_t1.bias, *bias),
        ]
        for unit in (block.res_unit1, block.res_unit2, block.res_unit3):
            assignments += [
                (unit.snake1.alpha, *alpha),
                (unit.conv1.weight, *weight),
                (unit.conv1.bias, *bias),
                (unit.snake2.alpha, *alpha),
                (unit.conv2.weight, *weight),
                (unit.conv2.bias, *bias),
            ]
    assignments += [
        (decoder.snake1.alpha, *alpha),
        (decoder.conv2.weight, *weight),
        (decoder.conv2.bias, *bias),
    ]
    with torch.no_grad():
        for parameter, scale, offset in assignments:
            parameter.copy_(
                stream.fill(parameter.numel(), scale, offset).view_as(parameter)
            )

    codes = torch.tensor(CODES)
    latent = torch.zeros(CONCAT, FRAMES)
    for level in range(QUANTIZERS):
        rows = F.embedding(codes[level], codebooks[level])  # [FRAMES, DIM]
        latent = latent + F.linear(rows, out_weights[level], out_biases[level]).T
    acoustic = F.linear(latent.T, fc2_weight, fc2_bias).T  # [ACOUSTIC, FRAMES]
    with torch.no_grad():
        wave = decoder(acoustic.unsqueeze(0))[0, 0]  # [FRAMES * HOP]

    assert wave.shape == (FRAMES * HOP,), f"wave shape {tuple(wave.shape)}"
    for name, tensor in (("latent", latent), ("acoustic", acoustic), ("wave", wave)):
        assert torch.isfinite(tensor).all(), f"{name} is not finite"

    # The wave array is the only thing that catches a surviving tanh or a
    # time-shifting crop, and it can only do that if its own samples are big
    # enough and different enough from each other. Both margins are asserted
    # here against the accelerator tolerance the test applies, so a re-tuned
    # fixture that quietly lost the power fails at dump time.
    tolerance = 5e-3
    tanh_gap = (torch.tanh(wave) - wave).abs().max().item()
    shift_gap = (wave[1:] - wave[:-1]).abs().max().item()
    assert tanh_gap > 10 * tolerance, f"a surviving tanh would move the wave by only {tanh_gap:.3g}"
    assert shift_gap > 10 * tolerance, f"a one-sample shift would move the wave by only {shift_gap:.3g}"
    print(f"// tanh gap {tanh_gap:.4g}, one-sample shift gap {shift_gap:.4g}, peak {wave.abs().max():.4g}")

    # Dump in the C++ read-back order: channel-major tensors flatten
    # frame-major with channels contiguous per frame -> transpose first.
    dump("kExpectedLatent", latent.T)
    dump("kExpectedAcoustic", acoustic.T)
    dump("kExpectedWave", wave)


if __name__ == "__main__":
    main()
