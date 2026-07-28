#!/usr/bin/env python3
"""Emit the reference values for tests/qwen3_tts_codec_ops_test.cpp.

The codec decoder's convolution stack is built from four operators, and every
one of them has a way to be wrong that produces audio rather than an error:

- ``Qwen3TTSTokenizerV2CausalConvNet`` pads on the **left only**. ggml pads
  symmetrically, so the port convolves wide and keeps the prefix; off by one and
  the waveform shifts in time.
- The same operator depthwise, where the kernel's input extent is one rather than
  the channel count. ggml's own depthwise convolution carries an upstream "very
  likely wrong for some cases" warning, which is reason enough to pin it.
- ``Qwen3TTSTokenizerV2CausalTransConvNet`` convolves and then drops the trailing
  ``kernel - stride`` samples. Cropping the wrong end is still causal-looking.
- ``SnakeBeta`` stores both curves as logarithms, so neither is near one and
  neither exponential can be folded away.

Run from ``scripts/envs/qwen3-tts``:

    uv run python ../../dump_reference_qwen3_tts_codec_ops.py
"""

from __future__ import annotations

import argparse
import sys

import torch

LENGTH = 7
CHANNELS = 4
OUT_CHANNELS = 6
KERNEL = 3
DILATION = 2
STRIDE = 2

LCG_MULTIPLIER = 6364136223846793005
LCG_INCREMENT = 1442695040888963407
LCG_MASK = (1 << 64) - 1


class LcgStream:
    def __init__(self, seed: int) -> None:
        self.state = seed & LCG_MASK

    def next(self) -> float:
        self.state = (self.state * LCG_MULTIPLIER + LCG_INCREMENT) & LCG_MASK
        return float(self.state >> 40) / 8388608.0 - 1.0

    def fill(self, shape, scale: float, offset: float) -> torch.Tensor:
        count = 1
        for extent in shape:
            count *= extent
        values = [self.next() * scale + offset for _ in range(count)]
        return torch.tensor(values, dtype=torch.float32).view(*shape)


def run(seed: int):
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2CausalConvNet,
        Qwen3TTSTokenizerV2CausalTransConvNet,
        Qwen3TTSTokenizerV2ConvNeXtBlock,
        SnakeBeta,
    )

    stream = LcgStream(seed)
    # The fill order below is the test's.
    signal = stream.fill((1, CHANNELS, LENGTH), 0.5, 0.0)

    conv = Qwen3TTSTokenizerV2CausalConvNet(CHANNELS, OUT_CHANNELS, KERNEL, dilation=DILATION)
    with torch.no_grad():
        conv.conv.weight.copy_(stream.fill((OUT_CHANNELS, CHANNELS, KERNEL), 0.5, 0.0))
        conv.conv.bias.copy_(stream.fill((OUT_CHANNELS,), 0.25, 0.0))

    depthwise = Qwen3TTSTokenizerV2CausalConvNet(CHANNELS, CHANNELS, KERNEL, groups=CHANNELS)
    with torch.no_grad():
        depthwise.conv.weight.copy_(stream.fill((CHANNELS, 1, KERNEL), 0.5, 0.0))
        depthwise.conv.bias.copy_(stream.fill((CHANNELS,), 0.25, 0.0))

    # Kernel twice the stride, which is what the residual stack uses.
    transposed = Qwen3TTSTokenizerV2CausalTransConvNet(CHANNELS, OUT_CHANNELS, 2 * STRIDE, STRIDE)
    with torch.no_grad():
        transposed.conv.weight.copy_(stream.fill((CHANNELS, OUT_CHANNELS, 2 * STRIDE), 0.5, 0.0))
        transposed.conv.bias.copy_(stream.fill((OUT_CHANNELS,), 0.25, 0.0))

    snake = SnakeBeta(CHANNELS)
    with torch.no_grad():
        snake.alpha.copy_(stream.fill((CHANNELS,), 0.5, 0.0))
        snake.beta.copy_(stream.fill((CHANNELS,), 0.5, 0.0))

    convnext = Qwen3TTSTokenizerV2ConvNeXtBlock(CHANNELS)
    with torch.no_grad():
        convnext.dwconv.conv.weight.copy_(stream.fill((CHANNELS, 1, 7), 0.5, 0.0))
        convnext.dwconv.conv.bias.copy_(stream.fill((CHANNELS,), 0.25, 0.0))
        convnext.norm.weight.copy_(stream.fill((CHANNELS,), 0.25, 1.0))
        convnext.norm.bias.copy_(stream.fill((CHANNELS,), 0.25, 0.0))
        convnext.pwconv1.weight.copy_(stream.fill((4 * CHANNELS, CHANNELS), 0.5, 0.0))
        convnext.pwconv1.bias.copy_(stream.fill((4 * CHANNELS,), 0.25, 0.0))
        convnext.pwconv2.weight.copy_(stream.fill((CHANNELS, 4 * CHANNELS), 0.5, 0.0))
        convnext.pwconv2.bias.copy_(stream.fill((CHANNELS,), 0.25, 0.0))
        convnext.gamma.copy_(stream.fill((CHANNELS,), 0.5, 1.0))

    with torch.no_grad():
        results = {
            "kExpectedCausalConv": conv(signal),
            "kExpectedDepthwise": depthwise(signal),
            "kExpectedTransposed": transposed(signal),
            "kExpectedSnakeBeta": snake(signal),
            "kExpectedConvNeXt": convnext(signal),
        }
    # The port's layout is ne = [channels, length], so channels are contiguous
    # and this has to be flattened with time on the outside -- the transpose of
    # the [C, T] the reference produces.
    return {name: value[0].transpose(0, 1).contiguous() for name, value in results.items()}


def emit(name: str, values: torch.Tensor) -> None:
    print(f"// ne = [{values.shape[1]} channels, {values.shape[0]} length].")
    print(f"constexpr float {name}[] = {{")
    flat = values.reshape(-1).tolist()
    for start in range(0, len(flat), 4):
        print("    " + ", ".join(f"{value:.9g}f" for value in flat[start : start + 4]) + ",")
    print("};")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260728)
    arguments = parser.parse_args()
    for name, values in run(arguments.seed).items():
        emit(name, values)
    return 0


if __name__ == "__main__":
    sys.exit(main())
