#!/usr/bin/env python3
"""Reference values for tests/omnivoice_reference_encoder_test.cpp.

Runs a miniature HuBERT (transformers HubertModel, do_stable_layer_norm=False,
the same variant OmniVoice's checkpoint carries) plus the codec's own
SemanticEncoder over LCG-drawn weights, through the REAL transformers classes
-- HubertModel and, from
transformers.models.higgs_audio_v2_tokenizer.modeling_higgs_audio_v2_tokenizer,
SemanticEncoder itself (which in turn builds real
HiggsAudioV2TokenizerSemanticEncoderBlock / HiggsAudioV2TokenizerResidualUnit
instances) -- and prints the expected C++ fixture arrays.

This mirrors HiggsAudioV2TokenizerModel._extract_semantic_features +
.encode's `e_semantic = self.encoder_semantic(e_semantic_input.transpose(1, 2))`
(modeling_higgs_audio_v2_tokenizer.py:490-510, 544-545 in the pinned venv):
pad(160, 160) -> HubertModel(output_hidden_states=True) -> mean over all
hidden states -> [::2] downsample -> SemanticEncoder. The (160, 160) pad is a
literal constant in the reference (not derived from any config field -- see
the comment immediately above it in the pinned source, which shows the
ORIGINAL formula, `self.pad = hop_length // 2`, was replaced by this literal),
so it is applied here unconditionally too, at the SAME width the miniature
scale uses as at real scale: src/arch/omnivoice/reference-encoder.cpp hardcodes
it rather than deriving it from HParams, and this script must pad the same way
to stay comparable.

SemanticEncoder's own `__init__` only reads `config.semantic_hidden_size`,
`.kernel_size`, `.strides`, `.channel_ratios`, `.unit_kernel_size` and
`.block_dilations` (via the sub-modules it builds) -- not the full
HiggsAudioV2TokenizerConfig -- so a small attribute bag stands in for it here
rather than constructing the much larger acoustic+semantic config tree Task 12
needs for the DAC half. That is still the real module classes verbatim, only a
lighter-weight config object.

HuBERT's positional convolution carries a weight-norm parametrization
(`nn.utils.parametrizations.weight_norm(conv, name="weight", dim=2)`,
HubertPositionalConvEmbedding.__init__): this script sets the magnitude/
direction pair (`original0`/`original1`) via the LCG, then folds them with
`torch.nn.utils.parametrize.remove_parametrizations(..., leave_parametrized=True)`
-- which bakes in exactly `torch._weight_norm(direction, magnitude, dim=2)`,
the same operator scripts/convert-omnivoice.py's `fold_weight_norm` calls on
the real checkpoint (see its citation of `torch._weight_norm` as "the operator
the reference calls on every forward"). The C++ fixture's `kPosConvWeight`
is this folded tensor: the plain kernel the catalog resolves, never the
parametrized pair.

The parameter-coverage assert below (walking `named_parameters()` for both the
HuBERT tree and the SemanticEncoder tree) is the pattern hardened in
scripts/dump_reference_omnivoice_codec.py's own review: a transformers release
that grows either module tree would otherwise leave a fresh parameter at its
unseeded random initialization and silently change the dump on every run.

Usage:
    uv run --project scripts/envs/omnivoice --locked python \
        scripts/dump_reference_omnivoice_semantic.py
"""

import types

import torch

SEED = 20260801

# --- HuBERT (transformers HubertConfig), miniature scale --------------------
# Three feature-extractor layers, not seven: layer 0 keeps the real
# checkpoint's GroupNorm structure (feat_extract_norm="group" normalizes only
# the first layer), and every layer keeps a real checkpoint's stride-2
# pattern, which is what exercises the length arithmetic a stride-5 or a
# stride-1 layer would not.
CONV_DIM = [4, 4, 4]
CONV_KERNEL = [3, 3, 2]
CONV_STRIDE = [2, 2, 2]
HIDDEN = 8
HEADS = 2
HEAD_DIM = HIDDEN // HEADS
INTERMEDIATE = 12
LAYERS = 2
LAYER_NORM_EPS = 1e-5
# Even, like the real checkpoint's 128: exercises HubertSamePadLayer's
# drop-the-last-sample trim, which an odd kernel would never touch.
POS_KERNEL = 4
POS_GROUPS = 2
# Raw, PRE-pad samples. The (160, 160) pad dominates every downstream length
# at this scale (as it does at real scale, where the checkpoint's own 224640
# padded to 224960 samples), so shrinking this further buys nothing.
PCM_SAMPLES = 8

# --- codec.encoder_semantic (HiggsAudioV2TokenizerConfig fields), already
# minimal at real scale: two stride-1 blocks of two dilation-1 residual units,
# unchanged here.
SEM_KERNEL_SIZE = 3
SEM_STRIDES = [1, 1]
SEM_CHANNEL_RATIOS = [1, 1]
SEM_UNIT_KERNEL_SIZE = 3
SEM_BLOCK_DILATIONS = [1, 1]

# Roles this script draws with: projections/convolutions, biases, and the
# near-1 gain / near-0 shift a norm's affine pair gets so neither degenerates
# (an all-zero gain would erase everything downstream of it, silently passing
# a builder that dropped the affine step entirely).
WEIGHT_SCALE, WEIGHT_OFFSET = 0.3, 0.0
BIAS_SCALE, BIAS_OFFSET = 0.1, 0.0
GAIN_SCALE, GAIN_OFFSET = 0.25, 1.0
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


def build_hubert():
    from transformers.models.hubert.configuration_hubert import HubertConfig
    from transformers.models.hubert.modeling_hubert import HubertModel

    config = HubertConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        intermediate_size=INTERMEDIATE,
        hidden_act="gelu",
        hidden_dropout=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        feat_proj_layer_norm=True,
        feat_proj_dropout=0.0,
        final_dropout=0.0,
        layerdrop=0.0,
        layer_norm_eps=LAYER_NORM_EPS,
        feat_extract_norm="group",
        feat_extract_activation="gelu",
        conv_dim=CONV_DIM,
        conv_stride=CONV_STRIDE,
        conv_kernel=CONV_KERNEL,
        conv_bias=False,
        num_conv_pos_embeddings=POS_KERNEL,
        num_conv_pos_embedding_groups=POS_GROUPS,
        conv_pos_batch_norm=False,
        do_stable_layer_norm=False,
        apply_spec_augment=True,
        mask_time_prob=0.0,
        mask_feature_prob=0.0,
    )
    return HubertModel(config).eval()


def build_semantic_encoder():
    from transformers.models.higgs_audio_v2_tokenizer.modeling_higgs_audio_v2_tokenizer import (
        SemanticEncoder,
    )

    config = types.SimpleNamespace(
        semantic_hidden_size=HIDDEN,
        kernel_size=SEM_KERNEL_SIZE,
        strides=SEM_STRIDES,
        channel_ratios=SEM_CHANNEL_RATIOS,
        unit_kernel_size=SEM_UNIT_KERNEL_SIZE,
        block_dilations=SEM_BLOCK_DILATIONS,
    )
    return SemanticEncoder(config).eval()


def fill_hubert(model, stream):
    """Every HuBERT parameter, in the catalog's own resolution order
    (catalog.cpp's resolve_semantic_model): feature-extractor convs, the
    layer-0 GroupNorm, feature_projection, the positional convolution
    (magnitude then direction then bias), each encoder layer, and the
    encoder's own output LayerNorm.
    """
    assignments = []

    def weight(tensor):
        assignments.append((tensor, WEIGHT_SCALE, WEIGHT_OFFSET))
        return tensor

    def bias(tensor):
        assignments.append((tensor, BIAS_SCALE, BIAS_OFFSET))
        return tensor

    def gain(tensor):
        assignments.append((tensor, GAIN_SCALE, GAIN_OFFSET))
        return tensor

    for conv_layer in model.feature_extractor.conv_layers:
        weight(conv_layer.conv.weight)
    group_norm = model.feature_extractor.conv_layers[0].layer_norm
    gain(group_norm.weight)
    bias(group_norm.bias)

    gain(model.feature_projection.layer_norm.weight)
    bias(model.feature_projection.layer_norm.bias)
    weight(model.feature_projection.projection.weight)
    bias(model.feature_projection.projection.bias)

    pos_conv = model.encoder.pos_conv_embed.conv
    gain(pos_conv.parametrizations.weight.original0)  # magnitude, dim=2
    weight(pos_conv.parametrizations.weight.original1)  # direction
    bias(pos_conv.bias)

    for layer in model.encoder.layers:
        weight(layer.attention.q_proj.weight)
        bias(layer.attention.q_proj.bias)
        weight(layer.attention.k_proj.weight)
        bias(layer.attention.k_proj.bias)
        weight(layer.attention.v_proj.weight)
        bias(layer.attention.v_proj.bias)
        weight(layer.attention.out_proj.weight)
        bias(layer.attention.out_proj.bias)
        gain(layer.layer_norm.weight)
        bias(layer.layer_norm.bias)
        weight(layer.feed_forward.intermediate_dense.weight)
        bias(layer.feed_forward.intermediate_dense.bias)
        weight(layer.feed_forward.output_dense.weight)
        bias(layer.feed_forward.output_dense.bias)
        gain(layer.final_layer_norm.weight)
        bias(layer.final_layer_norm.bias)

    gain(model.encoder.layer_norm.weight)
    bias(model.encoder.layer_norm.bias)

    assigned = {id(parameter) for parameter, _scale, _offset in assignments}
    missed = [
        name for name, parameter in model.named_parameters() if id(parameter) not in assigned
    ]
    assert not missed, (
        f"HubertModel parameters keeping their random initialization: {missed}; "
        "the dump would not be reproducible"
    )
    with torch.no_grad():
        for parameter, scale, offset in assignments:
            parameter.copy_(stream.fill(parameter.numel(), scale, offset).view_as(parameter))

    # Fold the positional convolution's weight-norm parametrization into a
    # plain kernel, bit-identically to scripts/convert-omnivoice.py's
    # fold_weight_norm (both call torch._weight_norm(direction, magnitude,
    # dim)) -- the C++ fixture is this folded tensor, never the pair.
    torch.nn.utils.parametrize.remove_parametrizations(pos_conv, "weight", leave_parametrized=True)
    assert not hasattr(pos_conv, "parametrizations"), "pos_conv weight-norm survived folding"
    return pos_conv.weight.detach().clone()


def fill_semantic_encoder(encoder, stream):
    """Every SemanticEncoder parameter, in catalog.cpp's resolve_encoder_semantic
    order: the entry convolution, then per block the res units' two
    convolutions each, then the block's own (biased) exit convolution.
    """
    assignments = []

    def weight(tensor):
        assignments.append((tensor, WEIGHT_SCALE, WEIGHT_OFFSET))

    def bias(tensor):
        assignments.append((tensor, BIAS_SCALE, BIAS_OFFSET))

    weight(encoder.conv.weight)
    for block in encoder.conv_blocks:
        for unit in block.res_units:
            weight(unit.conv1.weight)
            weight(unit.conv2.weight)
        weight(block.conv.weight)
        bias(block.conv.bias)

    assigned = {id(parameter) for parameter, _scale, _offset in assignments}
    missed = [
        name for name, parameter in encoder.named_parameters() if id(parameter) not in assigned
    ]
    assert not missed, (
        f"SemanticEncoder parameters keeping their random initialization: {missed}; "
        "the dump would not be reproducible"
    )
    with torch.no_grad():
        for parameter, scale, offset in assignments:
            parameter.copy_(stream.fill(parameter.numel(), scale, offset).view_as(parameter))


def main():
    import transformers

    hubert = build_hubert()
    semantic_encoder = build_semantic_encoder()

    stream = LcgStream(SEED)
    pos_conv_weight = fill_hubert(hubert, stream)
    fill_semantic_encoder(semantic_encoder, stream)
    pcm = stream.fill(PCM_SAMPLES, PCM_SCALE, PCM_OFFSET)

    padded = torch.nn.functional.pad(pcm.unsqueeze(0), (160, 160))
    with torch.no_grad():
        outputs = hubert(padded, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    assert len(hidden_states) == LAYERS + 1, (
        f"expected {LAYERS + 1} hidden states (embedding + {LAYERS} layers), got {len(hidden_states)}"
    )
    stacked = torch.stack(hidden_states, dim=1)  # [1, LAYERS+1, T, HIDDEN]
    mean = stacked.mean(dim=1)[0]  # [T, HIDDEN]
    downsampled = mean[::2, :]  # [T', HIDDEN]

    with torch.no_grad():
        encoded = semantic_encoder(downsampled.T.unsqueeze(0))  # [1, HIDDEN, T'']
    final = encoded[0]  # [HIDDEN, T'']

    for name, tensor in (("mean", mean), ("downsampled", downsampled), ("final", final)):
        assert torch.isfinite(tensor).all(), f"{name} is not finite"

    print(
        f"// transformers {transformers.__version__}, torch {torch.__version__}; "
        f"mean {tuple(mean.shape)}, downsampled {tuple(downsampled.shape)}, "
        f"final {tuple(final.shape)}"
    )
    dump("kPcm16k", pcm)
    dump("kPosConvWeight", pos_conv_weight)
    # kExpectedMean is already [T, HIDDEN] -- position-major, feature-minor --
    # which is exactly how a ggml [HIDDEN, T] channel-major tensor (HIDDEN the
    # fastest-varying axis) flattens, so no transpose is needed to match the
    # C++ read-back order (the resampler/generator probes' own convention).
    dump("kExpectedMean", mean)
    dump("kExpectedDownsampled", downsampled)
    # kExpectedFinal comes off a [HIDDEN, T''] channel-first torch tensor, the
    # OPPOSITE order, so it is transposed first -- the codec script's own
    # "C++ read-back order" rule for latent/acoustic.
    dump("kExpectedFinal", final.T)


if __name__ == "__main__":
    main()
