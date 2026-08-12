#!/usr/bin/env python3
"""Emit the reference values for tests/qwen3_tts_speaker_encoder_test.cpp.

Runs the reference implementation's own ``Qwen3TTSSpeakerEncoder`` end to end --
a mel spectrogram in, an x-vector out -- over a deliberately tiny configuration
that keeps the real structure: a reflect-padded TDNN stem, three SE-Res2Net
blocks at dilations 2/3/4 with the production res2net scale of eight, the
multi-layer aggregation of blocks 1..3 only, attentive statistics pooling, and
the final projection.

End to end on purpose, and against upstream's own modules rather than a
transcription of them. Almost nothing in this architecture is declared by the
package: the dilations, the res2net accumulation order, the two activations in
the pooling attention branch, the axis the softmax runs over and the reflect
padding mode are all read off the source. Every one of them, guessed wrong,
still yields a finite embedding of the right width whose cosine against the
right answer is around 0.9 -- close enough to read as a tolerance problem. Only
a comparison against the reference implementation separates them.

Widths are deliberately all distinct (6 mel bins, 16 channels, 48 aggregated, 5
squeeze-excite, 7 attention, 11 output) so that a swapped extent cannot pass by
coincidence, and every one of them is smaller than the checkpoint's. The one
production number kept is the res2net scale of eight, because the split
arithmetic is what it governs.

Weights come from the same generator as the other reference scripts, drawn in
the order tests/qwen3_tts_speaker_encoder_test.cpp adds them. Run from
``scripts/envs/qwen3-tts``:

    uv run python ../../dump_reference_qwen3_tts_speaker_encoder.py
"""

from __future__ import annotations

import argparse

import torch

from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder

MEL_BINS = 6
CHANNELS = 16
AGGREGATED = 3 * CHANNELS
RES2NET_SCALE = 8
SE_CHANNELS = 5
ATTENTION_CHANNELS = 7
ENC_DIM = 11
FRAMES = 13
SEED = 20260812

WEIGHT_SCALE = 0.5
BIAS_SCALE = 0.25

LCG_MULTIPLIER = 6364136223846793005
LCG_INCREMENT = 1442695040888963407
LCG_MASK = (1 << 64) - 1


class LcgStream:
    def __init__(self, seed: int) -> None:
        self.state = seed & LCG_MASK

    def next(self) -> float:
        self.state = (self.state * LCG_MULTIPLIER + LCG_INCREMENT) & LCG_MASK
        return float(self.state >> 40) / 8388608.0 - 1.0

    def fill(self, count: int, scale: float) -> torch.Tensor:
        return torch.tensor([self.next() * scale for _ in range(count)], dtype=torch.float32)


def build_config() -> Qwen3TTSSpeakerEncoderConfig:
    # enc_channels[-1] is the aggregation width and must be three times the block
    # width, because the aggregation concatenates the three blocks' outputs.
    return Qwen3TTSSpeakerEncoderConfig(
        mel_dim=MEL_BINS,
        enc_channels=[CHANNELS, CHANNELS, CHANNELS, CHANNELS, AGGREGATED],
        enc_kernel_sizes=[5, 3, 3, 3, 1],
        enc_dilations=[1, 2, 3, 4, 1],
        enc_res2net_scale=RES2NET_SCALE,
        enc_se_channels=SE_CHANNELS,
        enc_attention_channels=ATTENTION_CHANNELS,
        enc_dim=ENC_DIM,
    )


def fill_order(model: Qwen3TTSSpeakerEncoder) -> list[str]:
    """The order tests/qwen3_tts_speaker_encoder_test.cpp adds its tensors in.

    Both sides draw from one stream, so the order is part of the contract: a
    different order gives every tensor different values and nothing else.
    """
    names = ["blocks.0.conv"]
    for block in range(1, 4):
        base = f"blocks.{block}."
        names.append(base + "tdnn1.conv")
        names += [base + f"res2net_block.blocks.{i}.conv" for i in range(RES2NET_SCALE - 1)]
        names.append(base + "se_block.conv1")
        names.append(base + "se_block.conv2")
        names.append(base + "tdnn2.conv")
    names += ["mfa.conv", "asp.tdnn.conv", "asp.conv", "fc"]
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    arguments = parser.parse_args()

    torch.manual_seed(0)
    model = Qwen3TTSSpeakerEncoder(build_config()).eval()
    state = model.state_dict()

    stream = LcgStream(arguments.seed)
    for name in fill_order(model):
        weight = state[name + ".weight"]
        bias = state[name + ".bias"]
        # A ggml Conv1d kernel is [kernel, in, out] against torch's
        # [out, in, kernel] -- the same bytes in the same order, so a linear fill
        # lands identically on both sides.
        weight.copy_(stream.fill(weight.numel(), WEIGHT_SCALE).reshape(weight.shape))
        bias.copy_(stream.fill(bias.numel(), BIAS_SCALE).reshape(bias.shape))
    model.load_state_dict(state)

    # The C++ mel tensor is ggml [bins, frames], which is torch [frames, bins] in
    # memory -- exactly the layout this module's forward transposes.
    mel = stream.fill(FRAMES * MEL_BINS, 1.0).reshape(1, FRAMES, MEL_BINS)
    with torch.no_grad():
        embedding = model(mel)[0]

    print(f"// {ENC_DIM} values, seed {arguments.seed}.")
    print("constexpr float kExpectedEmbedding[] = {")
    for index in range(0, ENC_DIM, 4):
        row = ", ".join(f"{value:.9g}f" for value in embedding[index : index + 4].tolist())
        print(f"    {row},")
    print("};")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
