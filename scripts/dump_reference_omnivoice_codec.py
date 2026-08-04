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
fixture.

`--encoder` mode ALSO now prints two RVQ-encode fixtures (Task 13),
independent of the DAC-encoder/fusion fixture above (their own LCG streams,
their own toy widths -- no codes, no quantizer flows through the DAC/fusion
half at all): the REAL HiggsAudioV2TokenizerResidualVectorQuantization /
VectorQuantization / EuclideanCodebook classes, run against a
`types.SimpleNamespace` standing in for HiggsAudioV2TokenizerConfig (that
config's own `hidden_size`/`num_quantizers` are COMPUTED PROPERTIES derived
from acoustic+semantic sub-configs and target_bandwidths/frame_rate, not
plain settable fields, so a stand-in exposing just the four attributes these
three classes actually read -- hidden_size, codebook_dim, codebook_size,
num_quantizers, plus frame_rate for ResidualVectorQuantization.__init__'s own
unconditional read of it -- is the direct way to pin a toy num_quantizers=2
without fighting the real config's property graph). `main_rvq()`'s first
fixture is the general-correctness case (2 levels, residual chaining, no
crafted geometry) with one codebook row pair placed a KNOWN small distance
apart at a specific (level, frame) to give `narrowest_gap` a real, non-zero,
non-trivial value to report; its second fixture crafts an EXACT tie (two
codebook rows equidistant from every frame's projected point, verified
bit-exact via `torch.equal` before a single value is printed) to pin the
tie-break rule -- lower id wins. Every weight, not just the expected outputs,
is printed verbatim rather than LCG-mirrored on the C++ side: unlike the
DAC/fusion fixture above, part of each RVQ fixture's codebook is hand-placed
geometry, not a LCG draw, so keeping the whole fixture in one printed,
verifiable place beats splitting it across a partially-replayed stream.

Usage:
    uv run --project scripts/envs/omnivoice --locked python \
        scripts/dump_reference_omnivoice_codec.py [--encoder]
"""

import argparse
import types

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


def format_float(value):
    """`{value:.9g}f`, guaranteed to contain a decimal point or exponent.

    `f"{10.0:.9g}"` is `"10"` -- a bare integer -- and `"10f"` is not a valid
    C++ floating literal (no decimal point, no exponent, so the compiler reads
    it as the integer `10` followed by a stray identifier `f`). Every OTHER
    fixture in this file draws from an LCG, whose outputs are irrational-
    looking floats that never hit this edge; `main_rvq`'s hand-placed "far
    away" codebook rows are deliberately round numbers and would trip it.
    """
    text = f"{value:.9g}"
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return f"{text}f"


def dump(name, tensor):
    flat = tensor.reshape(-1).tolist()
    print(f"constexpr float {name}[] = {{")
    for start in range(0, len(flat), 4):
        row = ", ".join(format_float(value) for value in flat[start : start + 4])
        print(f"    {row},")
    print("};")


def dump_i32(name, tensor):
    flat = [int(value) for value in tensor.reshape(-1).tolist()]
    print(f"constexpr int32_t {name}[] = {{")
    for start in range(0, len(flat), 4):
        row = ", ".join(str(value) for value in flat[start : start + 4])
        print(f"    {row},")
    print("};")


def dump_scalar(name, value):
    print(f"constexpr float {name} = {format_float(value)};")


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


# --encoder mode, continued (Task 13): two independent RVQ-encode fixtures.
# See this file's module docstring for why the real transformers classes run
# against a `types.SimpleNamespace` stand-in config, and why every weight
# (not just the expected outputs) is printed verbatim.
RVQ_CONCAT        = 4
RVQ_DIM           = 2
RVQ_CODEBOOK_SIZE = 4
RVQ_LEVELS        = 2
RVQ_FRAMES        = 4
RVQ_SEED          = 20260813
RVQ_WEIGHT_SCALE, RVQ_WEIGHT_OFFSET = 0.4, 0.0
RVQ_BIAS_SCALE, RVQ_BIAS_OFFSET     = 0.1, 0.0
RVQ_CODE_SCALE, RVQ_CODE_OFFSET     = 0.5, 0.0

# The near-tie geometry: level 1's codebook rows 0 and 1 are placed at radii
# R1 < R2 from frame TARGET_FRAME's own projected point (z1[TARGET_FRAME]),
# in a direction (+x) orthogonal to nothing in particular -- just a fixed,
# reproducible offset. For e = z + delta, the EuclideanCodebook dist formula
# reduces algebraically to exactly -||delta||^2 regardless of z (the z terms
# cancel), so the two rows' dist values differ by exactly R2^2 - R1^2 up to
# floating-point rounding in computing z+delta and the dot products -- a
# real, controllable, non-zero gap, not the accidental ~1e-9 rounding noise a
# SYMMETRIC z+/-delta placement around the same base would give (verified
# empirically while building this fixture: a symmetric placement's true
# mathematical gap is EXACTLY zero for any delta, since the z terms cancel
# identically for +delta and -delta too -- that construction is fixture 2's
# job, not this one's).
RVQ_TARGET_FRAME = 2
RVQ_NEAR_R1, RVQ_NEAR_R2 = 0.1, 0.11


def main_rvq():
    from transformers.models.higgs_audio_v2_tokenizer.modeling_higgs_audio_v2_tokenizer import (
        HiggsAudioV2TokenizerResidualVectorQuantization,
    )

    # --- Fixture 1: general correctness (2 levels, residual chaining) plus a
    # crafted near-tie at (level 1, frame RVQ_TARGET_FRAME) that ends up the
    # narrowest gap over the whole grid (asserted below, not assumed). ---
    config = types.SimpleNamespace(
        hidden_size=RVQ_CONCAT, codebook_dim=RVQ_DIM, codebook_size=RVQ_CODEBOOK_SIZE,
        num_quantizers=RVQ_LEVELS, frame_rate=25.0,
    )
    rvq = HiggsAudioV2TokenizerResidualVectorQuantization(config).eval()
    stream = LcgStream(RVQ_SEED)
    weight = (RVQ_WEIGHT_SCALE, RVQ_WEIGHT_OFFSET)
    bias = (RVQ_BIAS_SCALE, RVQ_BIAS_OFFSET)
    code = (RVQ_CODE_SCALE, RVQ_CODE_OFFSET)

    with torch.no_grad():
        for level in range(RVQ_LEVELS):
            q = rvq.quantizers[level]
            q.project_in.weight.copy_(stream.fill(RVQ_DIM * RVQ_CONCAT, *weight).view(RVQ_DIM, RVQ_CONCAT))
            q.project_in.bias.copy_(stream.fill(RVQ_DIM, *bias))
            q.project_out.weight.copy_(stream.fill(RVQ_CONCAT * RVQ_DIM, *weight).view(RVQ_CONCAT, RVQ_DIM))
            q.project_out.bias.copy_(stream.fill(RVQ_CONCAT, *bias))
            q.codebook.embed.copy_(stream.fill(RVQ_CODEBOOK_SIZE * RVQ_DIM, *code).view(RVQ_CODEBOOK_SIZE, RVQ_DIM))

        # Position-major, feature-minor -- the same convention main_encoder's
        # own `semantic` draw documents, and what a ggml ne=[RVQ_CONCAT,
        # RVQ_FRAMES] tensor (channel-fastest) reads as without a transpose.
        latent_pm = stream.fill(RVQ_FRAMES * RVQ_CONCAT, *weight).view(RVQ_FRAMES, RVQ_CONCAT)
        embeddings = latent_pm.T.unsqueeze(0)  # [1, RVQ_CONCAT, RVQ_FRAMES]

        # Run level 0 by hand (rather than rvq.encode() directly) so frame
        # RVQ_TARGET_FRAME's level-1 input (z1) is available to place the
        # near-tie against before level 1's codebook is finalized.
        q0 = rvq.quantizers[0]
        hs0 = q0.project_in(embeddings.permute(0, 2, 1)).reshape(-1, RVQ_DIM)
        embed0 = q0.codebook.embed.t()
        scaled0 = hs0.pow(2).sum(1, keepdim=True)
        dist0 = -(scaled0 - 2 * hs0 @ embed0 + embed0.pow(2).sum(0, keepdim=True))
        idx0 = dist0.max(dim=-1).indices
        quant0 = q0.project_out(q0.codebook.decode(idx0.view(1, RVQ_FRAMES))).permute(0, 2, 1)
        residual = embeddings - quant0

        q1 = rvq.quantizers[1]
        z1 = q1.project_in(residual.permute(0, 2, 1)).reshape(-1, RVQ_DIM)
        base = z1[RVQ_TARGET_FRAME].clone()
        near_tie_codebook = q1.codebook.embed.clone()
        near_tie_codebook[0] = base + torch.tensor([RVQ_NEAR_R1, 0.0])
        near_tie_codebook[1] = base + torch.tensor([RVQ_NEAR_R2, 0.0])
        # Rows 2 and 3 stay clearly far from every frame's z1 (row 2/3's own
        # random draw already is; overwritten with round literals anyway so
        # the printed fixture does not depend on readers re-deriving "far
        # enough" from an LCG draw they cannot see at a glance).
        near_tie_codebook[2] = torch.tensor([9.5, -9.5])
        near_tie_codebook[3] = torch.tensor([-9.5, 9.5])
        q1.codebook.embed.copy_(near_tie_codebook)

        final_indices = rvq.encode(embeddings).squeeze(1)  # [RVQ_LEVELS, RVQ_FRAMES]

        # Independently recompute every (level, frame) gap -- not just the
        # crafted one -- with the SAME dist formula rvq_encode's own host loop
        # uses, so kRvqExpectedNarrowestGap is a measured minimum over the
        # WHOLE grid, not an assumption that nothing else came out smaller.
        def all_gaps(rvq, embeddings):
            residual = embeddings.clone()
            gaps = []
            for level in range(RVQ_LEVELS):
                q = rvq.quantizers[level]
                hs = q.project_in(residual.permute(0, 2, 1)).reshape(-1, RVQ_DIM)
                embed = q.codebook.embed.t()
                scaled = hs.pow(2).sum(1, keepdim=True)
                dist = -(scaled - 2 * hs @ embed + embed.pow(2).sum(0, keepdim=True))
                sorted_dist, sorted_idx = torch.sort(dist, dim=-1, descending=True)
                gaps.append(sorted_dist[:, 0] - sorted_dist[:, 1])
                best = sorted_idx[:, 0]
                quant = q.project_out(q.codebook.decode(best.view(1, RVQ_FRAMES))).permute(0, 2, 1)
                residual = residual - quant
            return torch.stack(gaps)

        gaps = all_gaps(rvq, embeddings)
        narrowest = gaps.min().item()
        narrowest_level, narrowest_frame = (index.item() for index in torch.unravel_index(gaps.argmin(), gaps.shape))
        assert torch.isfinite(gaps).all(), "a non-finite gap means a degenerate codebook row"
        # Comfortably inside the crafted near-tie's own R2^2-R1^2 = 0.0021
        # neighbourhood and comfortably below every OTHER gap in this grid
        # (measured >= 0.0769 elsewhere) -- see the module-level comment above
        # RVQ_TARGET_FRAME for why this is not the accidental near-zero a
        # symmetric placement would give.
        assert 1e-4 < narrowest < 1e-2, f"narrowest gap {narrowest} is not the crafted near-tie"
        assert (narrowest_level, narrowest_frame) == (1, RVQ_TARGET_FRAME), (
            f"narrowest gap landed at level {narrowest_level} frame {narrowest_frame}, "
            f"not the crafted (1, {RVQ_TARGET_FRAME})"
        )

        print(f"// --rvq fixture 1 (primary + near-tie): indices {final_indices.tolist()}")
        print(f"// narrowest gap {narrowest:.9g} at level {narrowest_level} frame {narrowest_frame}")
        dump("kRvqLatent", latent_pm)
        for level in range(RVQ_LEVELS):
            q = rvq.quantizers[level]
            dump(f"kRvqProjectInWeight{level}", q.project_in.weight)
            dump(f"kRvqProjectInBias{level}", q.project_in.bias)
            dump(f"kRvqProjectOutWeight{level}", q.project_out.weight)
            dump(f"kRvqProjectOutBias{level}", q.project_out.bias)
            dump(f"kRvqCodebook{level}", q.codebook.embed)
        dump_i32("kRvqExpectedTokens", final_indices)
        dump_scalar("kRvqExpectedNarrowestGap", narrowest)

    # --- Fixture 2: a crafted EXACT tie -- two codebook rows equidistant from
    # every frame's projected point, lower id wins, gap == 0.0 to the bit. A
    # single level: the tie-break rule is level-independent, and a second
    # level would only restate fixture 1's own residual-chaining coverage. ---
    print()
    tie_seed = RVQ_SEED + 1  # a distinct stream; the tie's geometry is
    #                          independent of fixture 1's weights.
    tie_stream = LcgStream(tie_seed)
    tie_config = types.SimpleNamespace(
        hidden_size=RVQ_CONCAT, codebook_dim=RVQ_DIM, codebook_size=RVQ_CODEBOOK_SIZE,
        num_quantizers=1, frame_rate=25.0,
    )
    tie_rvq = HiggsAudioV2TokenizerResidualVectorQuantization(tie_config).eval()
    tie_frames = 2

    with torch.no_grad():
        q = tie_rvq.quantizers[0]
        # project_in.weight is FORCED to zero (not drawn from the LCG at all,
        # deliberately skipping past those `RVQ_DIM * RVQ_CONCAT` draws the
        # C++ side must skip too): z = project_in(x) = 0@x + bias = bias for
        # EVERY frame regardless of x, which is what lets codebook rows 0/1
        # below be placed as a literal, base-independent exact tie rather
        # than one derived from (and therefore only as exact as) a computed
        # projection.
        q.project_in.weight.zero_()
        q.project_in.bias.copy_(torch.tensor([0.5, -0.25]))
        q.project_out.weight.copy_(tie_stream.fill(RVQ_CONCAT * RVQ_DIM, *weight).view(RVQ_CONCAT, RVQ_DIM))
        q.project_out.bias.copy_(tie_stream.fill(RVQ_CONCAT, *bias))
        # Row 0 and row 1 sit at +-0.25 from bias=[0.5,-0.25] along the first
        # axis only -- every value here is an exact binary fraction (0.75,
        # 0.25, -0.25 are all exactly representable in float32), so `z - e0`
        # and `z - e1` round to exactly -0.25 and +0.25 with no cancellation
        # error, and dist(e0) == dist(e1) to the bit (asserted below, not
        # assumed). Rows 2/3 are far enough that they never contend.
        codebook = torch.tensor([[0.75, -0.25], [0.25, -0.25], [9.5, 9.5], [-9.5, 9.5]])
        q.codebook.embed.copy_(codebook)

        tie_latent_pm = tie_stream.fill(tie_frames * RVQ_CONCAT, *weight).view(tie_frames, RVQ_CONCAT)
        tie_embeddings = tie_latent_pm.T.unsqueeze(0)

        hs = q.project_in(tie_embeddings.permute(0, 2, 1)).reshape(-1, RVQ_DIM)
        embed = q.codebook.embed.t()
        scaled = hs.pow(2).sum(1, keepdim=True)
        dist = -(scaled - 2 * hs @ embed + embed.pow(2).sum(0, keepdim=True))
        assert torch.equal(dist[:, 0], dist[:, 1]), (
            f"the crafted tie is not bit-exact: dist[:,0]={dist[:, 0].tolist()} "
            f"dist[:,1]={dist[:, 1].tolist()}"
        )

        tie_indices = tie_rvq.encode(tie_embeddings).squeeze(1)
        assert torch.equal(tie_indices[0], torch.zeros(tie_frames, dtype=tie_indices.dtype)), (
            f"the tie-break rule should pick the lower id (0) at every frame, got {tie_indices.tolist()}"
        )

        print(f"// --rvq fixture 2 (exact tie): indices {tie_indices.tolist()}, gap 0.0 exactly")
        dump("kRvqTieLatent", tie_latent_pm)
        dump("kRvqTieProjectInBias", q.project_in.bias)
        dump("kRvqTieProjectOutWeight", q.project_out.weight)
        dump("kRvqTieProjectOutBias", q.project_out.bias)
        dump("kRvqTieCodebook", q.codebook.embed)
        dump_i32("kRvqTieExpectedTokens", tie_indices)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--encoder", action="store_true",
        help="dump the acoustic encoder + reference fusion miniature fixture "
             "plus the two RVQ-encode fixtures "
             "(tests/omnivoice_reference_encoder_test.cpp) instead of the "
             "decoder's (tests/omnivoice_codec_test.cpp)",
    )
    arguments = parser.parse_args()
    if arguments.encoder:
        main_encoder()
        main_rvq()
    else:
        main_decoder()


if __name__ == "__main__":
    main()
