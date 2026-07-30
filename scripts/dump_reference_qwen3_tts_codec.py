#!/usr/bin/env python3
"""Emit the reference values for tests/qwen3_tts_codec_test.cpp.

Runs the reference implementation's own ``Qwen3TTSTokenizerV2Decoder`` end to
end -- codes in, waveform out -- over a deliberately tiny configuration with the
real structure: a split residual quantizer, a sliding-window transformer, ConvNeXt
upsampling stages and dilated residual stages ending in one audio channel.

End to end on purpose. Every stage of this decoder feeds the next, so a length
that drifts by one, a channel width that halves at the wrong place or a window
that reaches one frame too far shows up in the waveform and nowhere earlier.

Weights come from the same generator as the other reference scripts. Run from
``scripts/envs/qwen3-tts``:

    uv run python ../../dump_reference_qwen3_tts_codec.py
"""

from __future__ import annotations

import argparse
import sys

import torch

FRAMES = 6
QUANTIZERS = 3
SEMANTIC_QUANTIZERS = 1
CODEBOOK_SIZE = 5
CODEBOOK_DIM = 8
LATENT_DIM = 6
DECODER_DIM = 8
HIDDEN = 4
INTERMEDIATE = 6
LAYERS = 2
HEADS = 2
HEAD_DIM = 4
SLIDING_WINDOW = 3
UPSAMPLE_RATES = [2, 3]
UPSAMPLING_RATIOS = [2]
RMS_NORM_EPS = 1e-5
ROPE_THETA = 10000.0

LCG_MULTIPLIER = 6364136223846793005
LCG_INCREMENT = 1442695040888963407
LCG_MASK = (1 << 64) - 1


class LcgStream:
    def __init__(self, seed: int) -> None:
        self.state = seed & LCG_MASK

    def next(self) -> float:
        self.state = (self.state * LCG_MULTIPLIER + LCG_INCREMENT) & LCG_MASK
        return float(self.state >> 40) / 8388608.0 - 1.0

    def fill(self, count: int, scale: float, offset: float) -> torch.Tensor:
        return torch.tensor([self.next() * scale + offset for _ in range(count)], dtype=torch.float32)


def build(seed: int):
    from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2DecoderConfig,
    )
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Decoder,
    )

    config = Qwen3TTSTokenizerV2DecoderConfig(
        latent_dim=LATENT_DIM,
        codebook_dim=CODEBOOK_DIM,
        codebook_size=CODEBOOK_SIZE,
        decoder_dim=DECODER_DIM,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        head_dim=HEAD_DIM,
        num_quantizers=QUANTIZERS,
        num_semantic_quantizers=SEMANTIC_QUANTIZERS,
        rms_norm_eps=RMS_NORM_EPS,
        rope_theta=ROPE_THETA,
        sliding_window=SLIDING_WINDOW,
        upsample_rates=UPSAMPLE_RATES,
        upsampling_ratios=UPSAMPLING_RATIOS,
        max_position_embeddings=64,
        attention_bias=False,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    model = Qwen3TTSTokenizerV2Decoder(config).eval()

    stream = LcgStream(seed)
    # The order below is the test's; a line out of place shifts every weight
    # after it.
    assignments = []

    def quantizer(rvq, levels):
        # The checkpoint stores EMA accumulators rather than the table, and the
        # converter divides one by the other. Holding the usage at exactly one
        # makes the accumulator the table, so the test can fill the table
        # directly -- and it must not draw from the stream, or every weight after
        # it would shift.
        for level in range(levels):
            book = rvq.vq.layers[level]._codebook
            assignments.append((book.embedding_sum, 0.5, 0.0))
            with torch.no_grad():
                book.cluster_usage.fill_(1.0)
        assignments.append((rvq.output_proj.weight, 0.5, 0.0))

    quantizer(model.quantizer.rvq_first, SEMANTIC_QUANTIZERS)
    quantizer(model.quantizer.rvq_rest, QUANTIZERS - SEMANTIC_QUANTIZERS)

    assignments += [(model.pre_conv.conv.weight, 0.5, 0.0), (model.pre_conv.conv.bias, 0.25, 0.0)]

    transformer = model.pre_transformer
    assignments += [
        (transformer.input_proj.weight, 0.5, 0.0),
        (transformer.input_proj.bias, 0.25, 0.0),
        (transformer.output_proj.weight, 0.5, 0.0),
        (transformer.output_proj.bias, 0.25, 0.0),
    ]
    for layer in transformer.layers:
        assignments += [
            (layer.input_layernorm.weight, 0.25, 1.0),
            (layer.self_attn.q_proj.weight, 0.5, 0.0),
            (layer.self_attn.k_proj.weight, 0.5, 0.0),
            (layer.self_attn.v_proj.weight, 0.5, 0.0),
            (layer.self_attn.o_proj.weight, 0.5, 0.0),
            (layer.self_attn_layer_scale.scale, 0.1, 0.2),
            (layer.post_attention_layernorm.weight, 0.25, 1.0),
            (layer.mlp.gate_proj.weight, 0.5, 0.0),
            (layer.mlp.up_proj.weight, 0.5, 0.0),
            (layer.mlp.down_proj.weight, 0.5, 0.0),
            (layer.mlp_layer_scale.scale, 0.1, 0.2),
        ]
    assignments.append((transformer.norm.weight, 0.25, 1.0))

    for stage in model.upsample:
        assignments += [
            (stage[0].conv.weight, 0.5, 0.0),
            (stage[0].conv.bias, 0.25, 0.0),
            (stage[1].dwconv.conv.weight, 0.5, 0.0),
            (stage[1].dwconv.conv.bias, 0.25, 0.0),
            (stage[1].norm.weight, 0.25, 1.0),
            (stage[1].norm.bias, 0.25, 0.0),
            (stage[1].pwconv1.weight, 0.5, 0.0),
            (stage[1].pwconv1.bias, 0.25, 0.0),
            (stage[1].pwconv2.weight, 0.5, 0.0),
            (stage[1].pwconv2.bias, 0.25, 0.0),
            (stage[1].gamma, 0.5, 1.0),
        ]

    assignments += [(model.decoder[0].conv.weight, 0.5, 0.0), (model.decoder[0].conv.bias, 0.25, 0.0)]
    for index in range(len(UPSAMPLE_RATES)):
        block = model.decoder[index + 1].block
        assignments += [(block[0].alpha, 0.25, 0.0), (block[0].beta, 0.25, 0.0),
                        (block[1].conv.weight, 0.5, 0.0), (block[1].conv.bias, 0.25, 0.0)]
        for unit in range(3):
            target = block[unit + 2]
            assignments += [
                (target.act1.alpha, 0.25, 0.0), (target.act1.beta, 0.25, 0.0),
                (target.conv1.conv.weight, 0.5, 0.0), (target.conv1.conv.bias, 0.25, 0.0),
                (target.act2.alpha, 0.25, 0.0), (target.act2.beta, 0.25, 0.0),
                (target.conv2.conv.weight, 0.5, 0.0), (target.conv2.conv.bias, 0.25, 0.0),
            ]
    tail = len(UPSAMPLE_RATES) + 1
    # The final convolution is drawn small on purpose. At the same scale as the
    # rest, random weights drive most of the waveform into the clamp, and a wrong
    # sample clamped to -1 matches a right one clamped to -1.
    assignments += [(model.decoder[tail].alpha, 0.25, 0.0), (model.decoder[tail].beta, 0.25, 0.0),
                    (model.decoder[tail + 1].conv.weight, 0.06, 0.0),
                    (model.decoder[tail + 1].conv.bias, 0.05, 0.0)]

    with torch.no_grad():
        for parameter, scale, offset in assignments:
            parameter.copy_(stream.fill(parameter.numel(), scale, offset).view(parameter.shape))
    return model


def run(seed: int):
    model = build(seed)
    # One code per level per frame, cycling so no level repeats the last.
    codes = torch.tensor(
        [[[(level * 2 + frame) % CODEBOOK_SIZE for frame in range(FRAMES)] for level in range(QUANTIZERS)]],
        dtype=torch.long,
    )
    with torch.no_grad():
        wav = model(codes)
    total = 1
    for factor in UPSAMPLE_RATES + UPSAMPLING_RATIOS:
        total *= factor
    assert wav.shape == (1, 1, FRAMES * total), (wav.shape, FRAMES * total)
    return codes[0], wav.view(-1)


def literal(value: float) -> str:
    """A C++ float literal. %.9g renders 1.0 as "1", and "1f" does not compile."""
    text = f"{value:.9g}"
    if not any(mark in text for mark in (".", "e", "n", "i")):
        text += ".0"
    return text + "f"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260728)
    arguments = parser.parse_args()
    codes, wav = run(arguments.seed)
    rows = codes.transpose(0, 1).reshape(-1).tolist()
    print(f"// {FRAMES} frames x {QUANTIZERS} levels, level-major as the graph reads it.")
    print("constexpr int32_t kCodes[] = { " + ", ".join(str(int(v)) for v in codes.reshape(-1).tolist()) + " };")
    print()
    print(f"// {wav.numel()} samples: {FRAMES} frames at a hop of {wav.numel() // FRAMES}.")
    print("constexpr float kExpectedWaveform[] = {")
    flat = wav.tolist()
    for start in range(0, len(flat), 4):
        print("    " + ", ".join(literal(value) for value in flat[start : start + 4]) + ",")
    print("};")
    return 0


if __name__ == "__main__":
    sys.exit(main())
