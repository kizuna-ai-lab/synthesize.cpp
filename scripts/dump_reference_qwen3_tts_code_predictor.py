#!/usr/bin/env python3
"""Emit the reference values for tests/qwen3_tts_code_predictor_test.cpp.

Runs the reference implementation's own
``Qwen3TTSTalkerCodePredictorModelForConditionalGeneration`` over one frame of a
deliberately tiny configuration, greedily, and prints the per-step logits and the
codes they select.

Greedy, not sampled: the codes have to be reproducible from the logits alone.
The reference draws from PyTorch's generator and this port draws from its own
seeded stream, so a sampled sequence would differ for reasons that say nothing
about whether the port is correct. What the port must reproduce is the
distribution the draw is made from, which is what the logits are.

Weights come from the same generator as the C++ side; see the decoder-layer
script for why. Run from ``scripts/envs/qwen3-tts``:

    uv run python ../../dump_reference_qwen3_tts_code_predictor.py
"""

from __future__ import annotations

import argparse
import sys

import torch

HIDDEN = 8
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 4
INTERMEDIATE = 16
LAYERS = 2
VOCAB = 12
CODE_GROUPS = 5
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
    from qwen_tts.core.models.configuration_qwen3_tts import (
        Qwen3TTSTalkerCodePredictorConfig,
        Qwen3TTSTalkerConfig,
    )
    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
    )

    config = Qwen3TTSTalkerCodePredictorConfig(
        vocab_size=VOCAB,
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
    )
    config._attn_implementation = "eager"
    talker_config = Qwen3TTSTalkerConfig(hidden_size=HIDDEN)

    model = Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(config, talker_config).eval()
    # The widths agree, so the reference's projection is an Identity and this
    # port binds no tensor for it. A rung where they differ would need both.
    assert isinstance(model.small_to_mtp_projection, torch.nn.Identity)
    assert len(model.lm_head) == CODE_GROUPS - 1
    assert len(model.model.codec_embedding) == CODE_GROUPS - 1

    stream = LcgStream(seed)
    assignments = []
    for layer in model.model.layers:
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
    assignments.append((model.model.norm.weight, 0.25, 1.0))
    for embedding in model.model.codec_embedding:
        assignments.append((embedding.weight, 0.5, 0.0))
    for head in model.lm_head:
        assignments.append((head.weight, 0.5, 0.0))

    with torch.no_grad():
        for parameter, scale, offset in assignments:
            parameter.copy_(stream.fill(parameter.numel(), scale, offset).view(parameter.shape))

    # The talker hands the predictor its own hidden state and its own embedding
    # of the semantic code, in that order.
    prefill = stream.fill(2 * HIDDEN, 0.5, 0.0).view(1, 2, HIDDEN)
    return model, prefill


def run(seed: int):
    model, prefill = build(seed)
    with torch.no_grad():
        result = model.generate(
            inputs_embeds=prefill,
            max_new_tokens=CODE_GROUPS - 1,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
    codes = result.sequences.view(-1).tolist()
    # generate() reports the processed scores, which for greedy decoding are the
    # raw logits; stacking them gives one row per step.
    logits = torch.stack([score.view(-1) for score in result.scores], dim=0)
    assert len(codes) == CODE_GROUPS - 1, codes
    assert logits.shape == (CODE_GROUPS - 1, VOCAB), logits.shape

    # The talker's next input is the sum of the acoustic embeddings, one per
    # group, plus its own embedding of the semantic code, which it adds itself.
    with torch.no_grad():
        summed = sum(
            model.model.codec_embedding[index](torch.tensor([code]))
            for index, code in enumerate(codes)
        ).view(-1)
    return codes, logits, summed


def emit(codes, logits, summed) -> None:
    print(f"// {CODE_GROUPS - 1} steps x {VOCAB} vocabulary, step-major.")
    print("constexpr float kExpectedLogits[] = {")
    flat = logits.reshape(-1).tolist()
    for start in range(0, len(flat), 4):
        print("    " + ", ".join(f"{value:.9g}f" for value in flat[start : start + 4]) + ",")
    print("};")
    print()
    print("constexpr int32_t kExpectedCodes[] = { " + ", ".join(str(code) for code in codes) + " };")
    print()
    print("// Sum of each acoustic code's embedding from its own group's table.")
    print("constexpr float kExpectedSummedEmbedding[] = {")
    values = summed.tolist()
    for start in range(0, len(values), 4):
        print("    " + ", ".join(f"{value:.9g}f" for value in values[start : start + 4]) + ",")
    print("};")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260728)
    arguments = parser.parse_args()
    emit(*run(arguments.seed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
