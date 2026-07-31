#!/usr/bin/env python3
"""Reference values for tests/omnivoice_generator_test.cpp.

Runs a one-layer transformers Qwen3Model bidirectionally (4-D all-True boolean
attention mask -- exactly how OmniVoice.forward drives it, see
``_prepare_batch``'s ``batch_attention_mask``) over LCG-drawn weights, plus the
embedding merge and the full-canvas heads, and prints the expected arrays. Fill
ORDER is the contract with the C++ test: reordering any line silently changes
every weight.

The merge is upstream ``_prepare_embed_inputs`` verbatim in shape: it ends in
``torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)``, a SELECT.
An audio position therefore carries the summed offset codebook embeddings
ALONE; the text embedding computed at that position is discarded.

Usage:
    uv run --project scripts/envs/omnivoice --locked python \
        scripts/dump_reference_omnivoice_generator.py
"""
import torch
import torch.nn.functional as F

SEED = 20260731
HIDDEN = 8
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 8
INTERMEDIATE = 16
# Two layers, not one. At depth 1 a stack that fed the embeddings to every layer
# instead of the running hidden state would be indistinguishable from a correct
# one, and so would a probe list that pushed the wrong tensor.
LAYERS = 2
EPS = 1e-6
THETA = 1e6
TEXT_VOCAB = 20
AUDIO_VOCAB = 6          # toy canvas vocabulary; mask id 5
CODEBOOKS = 3
TEXT_POSITIONS = 2
AUDIO_POSITIONS = 3
POSITIONS = TEXT_POSITIONS + AUDIO_POSITIONS

TEXT_IDS = [3, 7]
# Codebook-major [CODEBOOKS][AUDIO_POSITIONS]; 5 is the toy mask id, a real row
# of the audio table like any other.
AUDIO_GRID = [[5, 0, 2], [5, 1, 5], [4, 5, 3]]


class LcgStream:
    def __init__(self, seed):
        self.state = seed

    def next(self):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) % 2**64
        return (self.state >> 40) / 8388608.0 - 1.0

    def fill(self, count, scale, offset):
        return torch.tensor([self.next() * scale + offset for _ in range(count)],
                            dtype=torch.float32)


def dump(name, tensor):
    flat = tensor.reshape(-1).tolist()
    print(f"constexpr float {name}[] = {{")
    for start in range(0, len(flat), 4):
        row = ", ".join(f"{value:.9g}f" for value in flat[start:start + 4])
        print(f"    {row},")
    print("};")


def main():
    import transformers
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

    config = Qwen3Config(
        hidden_size=HIDDEN, num_attention_heads=HEADS, num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM, intermediate_size=INTERMEDIATE, rms_norm_eps=EPS,
        rope_theta=THETA, vocab_size=TEXT_VOCAB, num_hidden_layers=LAYERS,
        attention_dropout=0.0, attn_implementation="eager",
    )
    model = Qwen3Model(config).eval()
    assert model.config._attn_implementation == "eager", model.config._attn_implementation

    stream = LcgStream(SEED)
    # The C++ fixture fills its tensors in EXACTLY this order with these
    # scales/offsets (the qwen3-tts convention: projections {0.5, 0}, norm
    # gains {0.25, 1}, tables/inputs {0.5, 0}).
    text_table = stream.fill(TEXT_VOCAB * HIDDEN, 0.5, 0.0).view(TEXT_VOCAB, HIDDEN)
    audio_table = stream.fill(CODEBOOKS * AUDIO_VOCAB * HIDDEN, 0.5, 0.0).view(
        CODEBOOKS * AUDIO_VOCAB, HIDDEN)
    heads_table = stream.fill(CODEBOOKS * AUDIO_VOCAB * HIDDEN, 0.5, 0.0).view(
        CODEBOOKS * AUDIO_VOCAB, HIDDEN)
    # The final norm first, then every layer's eleven in order -- so layer 1's
    # weights differ from layer 0's and the two blocks cannot be confused.
    assignments = [(model.norm.weight, 0.25, 1.0)]
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
    with torch.no_grad():
        for parameter, scale, offset in assignments:
            parameter.copy_(stream.fill(parameter.numel(), scale, offset).view_as(parameter))

    # The embedding merge, exactly _prepare_embed_inputs' arithmetic: text
    # embedding for the text region, the SUM of the shifted codebook rows for
    # the audio region, where-selected -- which over contiguous regions is
    # concatenation.
    text_embed = F.embedding(torch.tensor(TEXT_IDS), text_table)
    grid = torch.tensor(AUDIO_GRID)
    offsets = (torch.arange(CODEBOOKS) * AUDIO_VOCAB).view(CODEBOOKS, 1)
    audio_embed = F.embedding(grid + offsets, audio_table).sum(dim=0)
    merged = torch.cat([text_embed, audio_embed], dim=0)

    mask = torch.ones(1, 1, POSITIONS, POSITIONS, dtype=torch.bool)
    position_ids = torch.arange(POSITIONS).unsqueeze(0)

    # Per-layer outputs come off hooks, NOT off `output_hidden_states`.
    # transformers 5.x records hidden states through the generic output-recorder
    # decorator, and its last recorded entry is the state AFTER `model.norm` --
    # identical to `last_hidden_state`, which would silently turn the deepest
    # layer probe into a second copy of the final probe.
    captured = {}

    def record(index):
        return lambda module, args, output: captured.__setitem__(index, output)

    handles = [layer.register_forward_hook(record(index))
               for index, layer in enumerate(model.layers)]
    with torch.no_grad():
        outputs = model(inputs_embeds=merged.unsqueeze(0), attention_mask=mask,
                        position_ids=position_ids)
    for handle in handles:
        handle.remove()

    layer_outs = [captured[index][0] for index in range(LAYERS)]  # pre-norm, per layer
    final = outputs.last_hidden_state[0]         # after the final norm
    logits = F.linear(final, heads_table)        # [POSITIONS, CODEBOOKS * AUDIO_VOCAB]

    # Every probe must be distinguishable from every other. If two layers agreed,
    # a stack that fed the embeddings to both instead of chaining them would pass;
    # if the deepest layer agreed with `final`, the probe was captured after the
    # norm and the final norm would go unchecked.
    for index in range(LAYERS - 1):
        assert not torch.equal(layer_outs[index], layer_outs[index + 1]), \
            f"layer {index} and {index + 1} probes are identical"
    assert not torch.equal(layer_outs[-1], final), "layer probe was captured post-norm"

    # The mask must actually be bidirectional: perturbing the LAST position has
    # to move the FIRST one. A transformers release that quietly re-imposed a
    # causal mask would otherwise hand the port causal reference values, and the
    # port -- which has no causal mode at all -- would look wrong forever.
    perturbed = merged.clone()
    perturbed[-1] += 1.0
    with torch.no_grad():
        other = model(inputs_embeds=perturbed.unsqueeze(0), attention_mask=mask,
                      position_ids=position_ids).last_hidden_state[0]
    moved = (other[0] - final[0]).abs().max().item()
    assert moved > 1e-3, f"attention is not bidirectional: position 0 moved {moved}"

    spread = min((layer_outs[i] - layer_outs[i + 1]).abs().max().item()
                 for i in range(LAYERS - 1))
    print(f"// transformers {transformers.__version__}, torch {torch.__version__};"
          f" bidirectionality probe moved position 0 by {moved:.4g};"
          f" closest two layer probes differ by {spread:.4g}")
    dump("kExpectedMerged", merged)
    for index, layer_out in enumerate(layer_outs):
        dump(f"kExpectedLayer{index}Output", layer_out)
    dump("kExpectedFinal", final)
    dump("kExpectedLogits", logits)


if __name__ == "__main__":
    main()
