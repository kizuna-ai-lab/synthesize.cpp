#!/usr/bin/env python3
"""Emit the reference values for tests/qwen3_tts_decoder_layer_test.cpp.

The Qwen3 decoder block is shared by the talker and the code predictor, so it is
worth pinning against the reference implementation itself rather than against a
host re-derivation of what the block is believed to do. This runs the actual
``Qwen3TTSDecoderLayer`` from the ``qwen-tts`` package over a deliberately tiny
configuration and prints the resulting hidden states as C++ literals.

Weights are not printed. Both sides draw them from the same integer generator in
the same order, so the test file carries only the outputs; see ``lcg_stream``
below and its twin in the test.

Run from ``scripts/envs/qwen3-tts``:

    uv run python ../../dump_reference_qwen3_tts_decoder_layer.py
"""

from __future__ import annotations

import argparse
import sys

import torch

# The shape under test is the real block's, scaled down: the same 2:1 grouped
# attention ratio, and a head_dim that does not divide the hidden size, which is
# how the real predictor is built (16 heads x 128 != 1024).
HIDDEN = 8
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 4
INTERMEDIATE = 16
POSITIONS = 4
RMS_NORM_EPS = 1e-6
ROPE_THETA = 1000000.0

# Any 64-bit LCG would do; this one is Knuth's MMIX multiplier. The point is that
# the sequence is reproducible in C++ without shipping a random-number library,
# and that every value is exactly representable in binary32 (a 24-bit numerator
# over 2^23), so the two sides agree bit for bit before any arithmetic runs.
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
        return torch.tensor(
            [self.next() * scale + offset for _ in range(count)], dtype=torch.float32
        )


def load_reference():
    from qwen_tts.core.models.configuration_qwen3_tts import (
        Qwen3TTSTalkerCodePredictorConfig,
    )
    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSDecoderLayer,
        Qwen3TTSRotaryEmbedding,
    )

    return Qwen3TTSTalkerCodePredictorConfig, Qwen3TTSDecoderLayer, Qwen3TTSRotaryEmbedding


def build_layer(seed: int):
    config_class, layer_class, rotary_class = load_reference()
    config = config_class(
        vocab_size=32,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=1,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        rms_norm_eps=RMS_NORM_EPS,
        rope_theta=ROPE_THETA,
        max_position_embeddings=64,
        attention_bias=False,
        attention_dropout=0.0,
        num_code_groups=16,
    )
    # The real predictor declares every layer full_attention and no sliding
    # window; asserting it here keeps this reference honest if that ever moves.
    assert set(config.layer_types) == {"full_attention"}, config.layer_types
    assert getattr(config, "sliding_window", None) is None

    config._attn_implementation = "eager"
    layer = layer_class(config, layer_idx=0).eval()
    rotary = rotary_class(config).eval()

    stream = LcgStream(seed)
    # Canonical order, mirrored exactly by the test. Norm gains sit near one
    # because a gain of zero would hide a wiring defect behind a zero output.
    assignments = [
        (layer.input_layernorm.weight, (HIDDEN,), 0.25, 1.0),
        (layer.self_attn.q_proj.weight, (HEADS * HEAD_DIM, HIDDEN), 0.5, 0.0),
        (layer.self_attn.k_proj.weight, (KV_HEADS * HEAD_DIM, HIDDEN), 0.5, 0.0),
        (layer.self_attn.v_proj.weight, (KV_HEADS * HEAD_DIM, HIDDEN), 0.5, 0.0),
        (layer.self_attn.o_proj.weight, (HIDDEN, HEADS * HEAD_DIM), 0.5, 0.0),
        (layer.self_attn.q_norm.weight, (HEAD_DIM,), 0.25, 1.0),
        (layer.self_attn.k_norm.weight, (HEAD_DIM,), 0.25, 1.0),
        (layer.post_attention_layernorm.weight, (HIDDEN,), 0.25, 1.0),
        (layer.mlp.gate_proj.weight, (INTERMEDIATE, HIDDEN), 0.5, 0.0),
        (layer.mlp.up_proj.weight, (INTERMEDIATE, HIDDEN), 0.5, 0.0),
        (layer.mlp.down_proj.weight, (HIDDEN, INTERMEDIATE), 0.5, 0.0),
    ]
    with torch.no_grad():
        for parameter, shape, scale, offset in assignments:
            assert tuple(parameter.shape) == shape, (parameter.shape, shape)
            parameter.copy_(stream.fill(parameter.numel(), scale, offset).view(shape))

    hidden = stream.fill(POSITIONS * HIDDEN, 0.5, 0.0).view(1, POSITIONS, HIDDEN)
    return layer, rotary, hidden


def run(seed: int) -> torch.Tensor:
    layer, rotary, hidden = build_layer(seed)
    position_ids = torch.arange(POSITIONS, dtype=torch.long).view(1, POSITIONS)

    # Additive causal mask, which is the form eager attention adds to the scores.
    mask = torch.full((POSITIONS, POSITIONS), float("-inf"), dtype=torch.float32)
    mask = torch.triu(mask, diagonal=1).view(1, 1, POSITIONS, POSITIONS)

    with torch.no_grad():
        position_embeddings = rotary(hidden, position_ids)
        output = layer(
            hidden_states=hidden,
            attention_mask=mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
    return output[0].view(POSITIONS, HIDDEN)


def literal(value: float) -> str:
    """A C++ float literal. %.9g renders 1.0 as "1", and "1f" does not compile."""
    text = f"{value:.9g}"
    if not any(mark in text for mark in (".", "e", "n", "i")):
        text += ".0"
    return text + "f"


def emit(values: torch.Tensor) -> None:
    flat = values.reshape(-1).tolist()
    print(f"// {POSITIONS} positions x {HIDDEN} hidden, position-major.")
    print("constexpr float kExpectedHidden[] = {")
    for start in range(0, len(flat), 4):
        row = ", ".join(literal(value) for value in flat[start : start + 4])
        print(f"    {row},")
    print("};")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260728)
    arguments = parser.parse_args()
    emit(run(arguments.seed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
