#!/usr/bin/env python3
"""Emit the reference values for tests/qwen3_tts_talker_test.cpp.

Two things are pinned, both against the reference implementation's own classes.

The **text tower**: ``Qwen3TTSTalkerResizeMLP`` over the wide text embedding,
which is what brings a text token down to the talker's width. It has biases and a
SiLU between its two layers, and the port has to agree on all three.

The **step**: ``Qwen3TTSTalkerModel``'s layers and final norm, then ``codec_head``.
The talker's attention goes through the reference's *multimodal* rope helper
rather than the plain one the code predictor uses, so running the real class is
how the claim that the two collapse to the same thing gets checked rather than
argued.

Weights come from the same generator as the other reference scripts. Run from
``scripts/envs/qwen3-tts``:

    uv run python ../../dump_reference_qwen3_tts_talker.py
"""

from __future__ import annotations

import argparse
import sys

import torch

HIDDEN = 8
TEXT_HIDDEN = 12
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 8
INTERMEDIATE = 16
LAYERS = 2
TEXT_VOCAB = 20
CODEC_VOCAB = 14
CODE_GROUPS = 4
POSITIONS = 5
RMS_NORM_EPS = 1e-6
ROPE_THETA = 1000000.0

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


def build(seed: int):
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig
    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSTalkerModel,
        Qwen3TTSTalkerResizeMLP,
    )

    config = Qwen3TTSTalkerConfig(
        vocab_size=CODEC_VOCAB,
        text_vocab_size=TEXT_VOCAB,
        text_hidden_size=TEXT_HIDDEN,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        rms_norm_eps=RMS_NORM_EPS,
        rope_theta=ROPE_THETA,
        max_position_embeddings=64,
        attention_bias=False,
        attention_dropout=0.0,
        num_code_groups=CODE_GROUPS,
        # The real checkpoint's shape, scaled down: three sections summing to half
        # the head dimension, interleaved. This is the whole reason the reference
        # class is run here rather than a plain-rope stand-in.
        rope_scaling={"interleaved": True, "mrope_section": [2, 1, 1],
                      "rope_type": "default", "type": "default"},
    )
    config._attn_implementation = "eager"
    assert config.rope_scaling["interleaved"] is True, config.rope_scaling
    assert sum(config.rope_scaling["mrope_section"]) * 2 == HEAD_DIM, config.rope_scaling

    model = Qwen3TTSTalkerModel._from_config(config).eval()
    projection = Qwen3TTSTalkerResizeMLP(
        TEXT_HIDDEN, TEXT_HIDDEN, HIDDEN, config.hidden_act, bias=True
    ).eval()
    codec_head = torch.nn.Linear(HIDDEN, CODEC_VOCAB, bias=False).eval()

    stream = LcgStream(seed)
    assignments = [
        (model.text_embedding.weight, 0.5, 0.0),
        (projection.linear_fc1.weight, 0.5, 0.0),
        (projection.linear_fc1.bias, 0.25, 0.0),
        (projection.linear_fc2.weight, 0.5, 0.0),
        (projection.linear_fc2.bias, 0.25, 0.0),
        (model.codec_embedding.weight, 0.5, 0.0),
        (codec_head.weight, 0.5, 0.0),
    ]
    for layer in model.layers:
        assignments += [
            (layer.input_layernorm.weight, 0.25, 1.0),
            (layer.self_attn.q_proj.weight, 0.5, 0.0),
            (layer.self_attn.k_proj.weight, 0.5, 0.0),
            (layer.self_attn.v_proj.weight, 0.5, 0.0),
            (layer.self_attn.o_proj.weight, 0.5, 0.0),
            (layer.self_attn.q_norm.weight, 0.25, 1.0),
            (layer.self_attn.k_norm.weight, 0.25, 1.0),
            (layer.post_attention_layernorm.weight, 0.25, 1.0),
            (layer.mlp.gate_proj.weight, 0.5, 0.0),
            (layer.mlp.up_proj.weight, 0.5, 0.0),
            (layer.mlp.down_proj.weight, 0.5, 0.0),
        ]
    assignments.append((model.norm.weight, 0.25, 1.0))
    with torch.no_grad():
        for parameter, scale, offset in assignments:
            parameter.copy_(stream.fill(parameter.numel(), scale, offset).view(parameter.shape))
    return model, projection, codec_head


def run(seed: int):
    model, projection, codec_head = build(seed)

    # A prefill in the real shape: every position carries a text token, and the
    # trailing positions also carry a codec token.
    text_ids = torch.tensor([[3, 11, 4, 0, 7]], dtype=torch.long)
    codec_ids = torch.tensor([[2, 9, 5]], dtype=torch.long)
    codec_offset = POSITIONS - codec_ids.shape[1]

    with torch.no_grad():
        text = projection(model.text_embedding(text_ids))
        codec = model.codec_embedding(codec_ids)
        inputs = text.clone()
        inputs[:, codec_offset:] = inputs[:, codec_offset:] + codec

        position_ids = torch.arange(POSITIONS, dtype=torch.long).view(1, 1, POSITIONS).expand(3, 1, POSITIONS)
        attention_mask = torch.ones(1, POSITIONS, dtype=torch.long)
        out = model(inputs_embeds=inputs, attention_mask=attention_mask, position_ids=position_ids,
                    use_cache=False)
        hidden = out.last_hidden_state
        logits = codec_head(hidden[:, -1:])

    return text.view(POSITIONS, HIDDEN), inputs.view(POSITIONS, HIDDEN), hidden[0, -1], logits.view(-1)


def emit_block(name: str, values: torch.Tensor, comment: str) -> None:
    print(f"// {comment}")
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
    text, inputs, hidden, logits = run(arguments.seed)
    emit_block("kExpectedTextProjection", text,
               f"{POSITIONS} positions x {HIDDEN} hidden: the text tower alone.")
    emit_block("kExpectedPrefillInput", inputs,
               "The same, with the codec stream summed into its trailing positions.")
    emit_block("kExpectedLastHidden", hidden,
               "The last position's hidden state after the final norm.")
    emit_block("kExpectedLogits", logits, f"codec_head over that: {CODEC_VOCAB} logits.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
