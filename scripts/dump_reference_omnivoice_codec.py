#!/usr/bin/env python3
"""Reference values for tests/omnivoice_codec_test.cpp and (--encoder)
tests/omnivoice_reference_encoder_test.cpp.

Default mode composes the Higgs DECODE path at toy dimensions from the REAL
classes: the transformers DacDecoder with
HiggsAudioV2TokenizerModel._adjust_dac_decoder applied (output_padding =
stride % 2, tanh removed), and the RVQ/fc2 stages as the exact F.embedding /
F.linear arithmetic the reference modules perform.

The upstream decode path this mirrors is HiggsAudioV2TokenizerModel.decode:
the RVQ sum over levels, fc2 across the channel axis, then the adjusted
acoustic decoder. The two Higgs adjustments are applied by the upstream
staticmethod rather than restated here, so a transformers release that changes
either of them changes these numbers instead of hiding the drift.

`--encoder` mode (Task 12) instead composes the ENCODE half's DAC-only stage:
a real transformers DacEncoder at toy dimensions, mirroring this file's own
decoder fixture widths (ENCODER_HIDDEN doubles through the SAME RATIOS list
the decoder halves through, landing back on ACOUSTIC -- this checkpoint's own
config states downsampling_ratios and upsampling_ratios as the literal same
array, [8, 5, 4, 2, 3], and DacEncoder/DacDecoder both walk their own ratio
list in plain enumeration order (modeling_dac.py's DacEncoder.__init__ /
DacDecoder.__init__), confirmed directly against the real checkpoint's own
instantiated block shapes -- see reference-encoder.h's own citation of this
evidence), plus a synthetic (not-HuBERT) semantic tensor and a plain
F.linear standing in for codec.fc, pinning build_reference_fusion's
concat+per-frame-Linear arithmetic independently of Task 11's own HuBERT
fixture (this mode never touches HuBERT or the RVQ -- no codes, no
quantizer -- both belong to a different task).

Usage:
    uv run --project scripts/envs/omnivoice --locked python \
        scripts/dump_reference_omnivoice_codec.py [--encoder]
"""

import argparse

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

# --encoder mode only (Task 12): the acoustic encoder mirrors the decoder's own
# toy widths in reverse -- ENCODER_HIDDEN doubles through RATIOS (the SAME
# list, in the SAME order the decoder halves through) and lands back on
# ACOUSTIC, the decoder's own input width, so the two toy fixtures describe
# the two ends of one consistent bottleneck. SEM_WIDTH is a synthetic stand-in
# for Task 11's HuBERT output (no HuBERT here -- this mode only exercises
# build_reference_fusion's concat+Linear arithmetic) and is deliberately a
# DIFFERENT width from ACOUSTIC so a fusion bug that mixed up which half goes
# where could not hide behind equal widths.
ENCODER_HIDDEN = 2
SEM_WIDTH = 3
FUSED_WIDTH = ACOUSTIC + SEM_WIDTH  # codec.fc is square: concat width -> itself
PCM_SCALE, PCM_OFFSET = 0.5, 0.0


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


def main_decoder():
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
    # The list above is built by hand from the module names this script knows
    # about; a transformers release that grows DacDecoder would leave the new
    # parameter at its (unseeded) random initialization and silently change
    # the dump on every run. Refuse to dump around such a parameter.
    assigned = {id(parameter) for parameter, _scale, _offset in assignments}
    missed = [
        name
        for name, parameter in decoder.named_parameters()
        if id(parameter) not in assigned
    ]
    assert not missed, (
        f"decoder parameters keeping their random initialization: {missed}; "
        "the dump would not be reproducible"
    )
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


def main_encoder():
    from transformers.models.dac.configuration_dac import DacConfig
    from transformers.models.dac.modeling_dac import DacEncoder

    # hidden_size is not a declared DacConfig dataclass field, but the same
    # override-after-__post_init__ mechanism main_decoder() already relies on
    # for `hidden_size`/`upsampling_ratios` applies here too (confirmed
    # directly against the real checkpoint: DacConfig.__post_init__
    # unconditionally derives hidden_size = encoder_hidden_size * 2**len(
    # ratios), but the real acoustic_model_config's own hidden_size=256 wins
    # over that derivation's 64*2**5=2048, because the raw config dict's
    # hidden_size is applied as a plain kwarg AFTER __post_init__ runs) --
    # this is what lets ENCODER_HIDDEN double through RATIOS and land on the
    # SAME ACOUSTIC width main_decoder()'s DECODER_HIDDEN halves down to,
    # rather than whatever ENCODER_HIDDEN * 2**len(RATIOS) happens to be.
    config = DacConfig(
        encoder_hidden_size=ENCODER_HIDDEN,
        hidden_size=ACOUSTIC,
        downsampling_ratios=RATIOS,
    )
    encoder = DacEncoder(config).eval()
    assert config.downsampling_ratios == RATIOS and config.hidden_size == ACOUSTIC, (
        f"DacConfig resolved downsampling_ratios={config.downsampling_ratios}, "
        f"hidden_size={config.hidden_size}; the override did not take"
    )

    stream = LcgStream(SEED)
    weight = (WEIGHT_SCALE, 0.0)
    bias = (BIAS_SCALE, 0.0)
    alpha = (ALPHA_SCALE, ALPHA_OFFSET)

    # The encoder's own forward order: entry conv, then per block the three
    # residual units (dilations 1/3/9) FIRST and the block's own Snake +
    # strided conv LAST (DacEncoderBlock.forward: res_unit1, res_unit2,
    # snake1(res_unit3(...)), conv1) -- the mirror image of the decoder's
    # snake-then-conv_t-then-residual-units order, not the same order
    # repeated. The C++ fixture's fill order below must match this exactly.
    assignments = [(encoder.conv1.weight, *weight), (encoder.conv1.bias, *bias)]
    for block in encoder.block:
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
            (block.snake1.alpha, *alpha),
            (block.conv1.weight, *weight),
            (block.conv1.bias, *bias),
        ]
    assignments += [
        (encoder.snake1.alpha, *alpha),
        (encoder.conv2.weight, *weight),
        (encoder.conv2.bias, *bias),
    ]
    # Same parameter-coverage assert as main_decoder(), hardened against a
    # transformers release that grows DacEncoder leaving a fresh parameter at
    # its unseeded random initialization.
    assigned = {id(parameter) for parameter, _scale, _offset in assignments}
    missed = [
        name
        for name, parameter in encoder.named_parameters()
        if id(parameter) not in assigned
    ]
    assert not missed, (
        f"encoder parameters keeping their random initialization: {missed}; "
        "the dump would not be reproducible"
    )
    with torch.no_grad():
        for parameter, scale, offset in assignments:
            parameter.copy_(
                stream.fill(parameter.numel(), scale, offset).view_as(parameter)
            )

    # A fixed, code-free (no RVQ, no HuBERT) raw waveform: FRAMES * HOP
    # samples, hop-aligned by construction exactly like every input this
    # port's own build_acoustic_encoder ever sees.
    pcm = stream.fill(FRAMES * HOP, PCM_SCALE, PCM_OFFSET)
    with torch.no_grad():
        acoustic = encoder(pcm.view(1, 1, -1))[0]  # [ACOUSTIC, FRAMES]
    assert acoustic.shape == (ACOUSTIC, FRAMES), f"acoustic shape {tuple(acoustic.shape)}"

    # A synthetic semantic tensor (no HuBERT here -- see this file's module
    # docstring) and a plain nn.Linear-equivalent (F.linear over explicit
    # weight/bias tensors, the same technique main_decoder() already uses for
    # fc2) standing in for codec.fc, continuing the SAME LCG stream.
    #
    # UNLIKE every other tensor in this function, `semantic` is not a native
    # torch module parameter (Conv1d.weight, Snake1d.alpha, ...), so there is
    # no torch-native shape dictating its .view() -- the C++ side must match
    # whatever choice is made here exactly. It is drawn POSITION-MAJOR,
    # FEATURE-MINOR (.view(FRAMES, SEM_WIDTH): each contiguous SEM_WIDTH-run
    # is one frame's full channel vector) and then transposed, which is what
    # makes the SAME flat draw sequence, read straight into a ggml
    # ne=[SEM_WIDTH, FRAMES] tensor (channel-fastest, so ALSO a contiguous
    # SEM_WIDTH-run per frame), represent the identical matrix -- the same
    # "position-major, feature-minor" rule this file's sibling semantic
    # dumper documents for its own kExpectedMean. Drawing it channel-major
    # instead (.view(SEM_WIDTH, FRAMES) with no transpose) would flip which
    # axis is contiguous and silently draw a DIFFERENT matrix on each side
    # whenever SEM_WIDTH != FRAMES.
    semantic = stream.fill(FRAMES * SEM_WIDTH, *weight).view(FRAMES, SEM_WIDTH).T
    fc_weight = stream.fill(FUSED_WIDTH * FUSED_WIDTH, *weight).view(FUSED_WIDTH, FUSED_WIDTH)
    fc_bias = stream.fill(FUSED_WIDTH, *bias)

    # HiggsAudioV2TokenizerModel.encode: torch.cat([e_acoustic, e_semantic],
    # dim=1) (the channel axis in torch's [B, C, T] layout, dim=0 here since
    # there is no batch axis) then
    # self.fc(embeddings.transpose(1, 2)).transpose(1, 2) -- Linear applied
    # per frame over the channel axis.
    embeddings = torch.cat([acoustic, semantic], dim=0)  # [FUSED_WIDTH, FRAMES]
    fused = F.linear(embeddings.T, fc_weight, fc_bias).T  # [FUSED_WIDTH, FRAMES]

    for name, tensor in (("acoustic", acoustic), ("semantic", semantic), ("fused", fused)):
        assert torch.isfinite(tensor).all(), f"{name} is not finite"

    print(
        f"// --encoder: acoustic {tuple(acoustic.shape)}, semantic {tuple(semantic.shape)}, "
        f"fused {tuple(fused.shape)}"
    )
    # Weights are not carried in either file, same as main_decoder()'s own
    # dump: pcm/semantic/fc_weight/fc_bias are all plain LCG draws (no
    # weight-norm folding or other transform, unlike the semantic dumper's
    # kPosConvWeight), so the C++ fixture draws them from the identical
    # stream instead of pasting them -- only the OUTPUT arrays are pinned.
    dump("kExpectedAcoustic", acoustic.T)
    dump("kExpectedFused", fused.T)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--encoder", action="store_true",
        help="dump the acoustic encoder + reference fusion miniature fixture "
             "(tests/omnivoice_reference_encoder_test.cpp) instead of the "
             "decoder's (tests/omnivoice_codec_test.cpp)",
    )
    arguments = parser.parse_args()
    if arguments.encoder:
        main_encoder()
    else:
        main_decoder()


if __name__ == "__main__":
    main()
