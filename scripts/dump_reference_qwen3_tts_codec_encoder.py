#!/usr/bin/env python3
"""Dump every stage of Qwen3-TTS-Base's codec encoder, not just the reference codes.

``scripts/dump_reference_qwen3_tts_base.py`` already dumps this subsystem's
finished output -- ``codes/reference.i32``, asserted ``[frames, 16]`` -- and
nothing between the reference waveform and it. That output is *discrete*, and
discreteness is what makes an error here invisible: a wrong SEANet stride and a
wrong transformer scale can cancel into latents that are only *nearly* right,
and the quantizer's argmin then rounds them to codes that are *exactly* right
for one clip and wrong for the next. This script captures the eleven stages in
between so a port can be compared where the error still has a magnitude.

**Provenance: this subsystem has a third upstream, and it is not a Qwen file.**
Every earlier stage of this family was read off ``QwenLM/Qwen3-TTS@022e286``
(``modeling_qwen3_tts.py``) or off the checkpoint. The codec encoder is
``transformers``' own ``MimiModel``: ``Qwen3TTSTokenizerV2Encoder`` subclasses
it and nulls the decoder halves (``upsample``, ``decoder_transformer``,
``decoder`` are set to ``None`` --
``qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py:899-908``),
and Qwen contributes only two post-steps around it (``:983-984``). Reading a
convention off the wrong one of the three -- causal padding, ELU placement,
residual dilations, ``trim_right_ratio``, the split RVQ's stage order --
produces finite codes that decode to plausible-sounding audio. So every
convention this script records carries its ``file:line``, and the ones that can
be *measured* rather than read are measured (see ``conventions.json``'s
``provenance`` block, and the assertions in ``install_rvq_probes``).

What is dumped, in pipeline order (``modeling_mimi.py:1442-1471``):

    waveform -> SEANet stack (4 strided stages + tail) -> transformer (8 layers)
             -> downsample (frame rate halving) -> split RVQ -> codes

Note that the *downsample sits after the transformer, not before it*. The
design's file list reads the other way round; the order above is the one
``_encode_frame`` actually walks, and it is what the artifacts follow.

The RVQ gets more than a hook. For each of the 16 stages whose codes survive
Qwen's slice, this dumps the residual *entering* the stage and, per frame, the
gap between the best and second-best codebook distance -- the tie margin the
design's section 6 requires MEASURED before the equality gate's final form is
chosen.

**No seed is needed.** The codec encoder is deterministic: fixed convolutions,
a fixed transformer, and an argmin over fixed codebooks. The seeded-sampling
rule applies only to scripts that run the talker, and this one does not.

Every case is gated on reproducing the Base oracle's own ``codes/reference.i32``
byte for byte, and that check -- like every other check here -- runs *before*
anything reaches disk, because Tasks 3/4/5/12 read the ``.f32`` files rather
than ``result.json``: a failed run must leave nothing consumable behind, not
merely nothing announced. A case with no Base artifacts to check against is a
hard failure unless ``--allow-unverified`` is passed, which downgrades the run's
status to ``unverified`` in both ``result.json`` and ``conventions.json``.

Usage (explicit-argument form -- exploration only; there is no Base oracle
output at an ad-hoc ``--out-dir``, hence ``--allow-unverified``):

    uv run --project scripts/envs/qwen3-tts --locked python \\
      scripts/dump_reference_qwen3_tts_codec_encoder.py \\
      --weights-dir models/qwen3-tts-12hz-0-6b-base \\
      --ref-audio models/qwen3-tts-reference-audio/clone.wav \\
      --out-dir build/scratch/codec-encoder-probe \\
      --allow-unverified

Manifest form (the primary interface):

    uv run --project scripts/envs/qwen3-tts --locked python \\
      scripts/dump_reference_qwen3_tts_codec_encoder.py \\
      --weights-dir models/qwen3-tts-12hz-0-6b-base \\
      --manifest tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-base.manifest.json \\
      --case base-icl-en
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
import time
import urllib.request
from typing import Any, Optional

import librosa
import numpy as np
import torch

MANIFEST_SCHEMA = "synthesize-golden-manifest-v1"
FAMILY = "qwen3-tts"

# Upstream #1: the Qwen package. Owns the two post-steps around MimiModel and
# the entry point that turns a clip into ref_code.
UPSTREAM_REPOSITORY = "https://github.com/QwenLM/Qwen3-TTS"
UPSTREAM_REVISION = "022e286b98fbec7e1e916cb940cdf532cd9f488e"
QWEN_TOKENIZER_FILE = "qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py"
QWEN_INFERENCE_FILE = "qwen_tts/inference/qwen3_tts_tokenizer.py"
QWEN_MODEL_FILE = "qwen_tts/inference/qwen3_tts_model.py"
QWEN_MODELING_FILE = "qwen_tts/core/models/modeling_qwen3_tts.py"

# Upstream #2: transformers' Mimi. Owns everything between the waveform and the
# codes. No prior stage of this family has read this file.
TRANSFORMERS_VERSION = "4.57.3"
MIMI_FILE = "transformers/models/mimi/modeling_mimi.py"
MIMI_CONFIG_FILE = "transformers/models/mimi/configuration_mimi.py"

# Upstream #3 is the checkpoint itself: speech_tokenizer/config.json.

# How many RVQ stages survive Qwen's slice (top-level `encoder_valid_num_quantizers`,
# NOT `encoder_config.num_quantizers`, which is 32). 1 semantic + 15 acoustic.
KEPT_QUANTIZER_COUNT = 16


def write_f32(path: pathlib.Path, array: np.ndarray) -> dict:
    data = np.ascontiguousarray(array, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.tobytes())
    return {"path": path.name, "shape": list(data.shape), "elements": int(data.size)}


def write_i32(path: pathlib.Path, array: np.ndarray) -> dict:
    data = np.ascontiguousarray(array, dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.tobytes())
    return {"path": path.name, "shape": list(data.shape), "elements": int(data.size)}


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def to_numpy(value: Any, dtype: Optional[torch.dtype] = None) -> np.ndarray:
    """Detach and move a possibly-GPU, possibly-bf16 tensor to a plain host array.

    ``write_f32`` calls ``np.ascontiguousarray`` directly, which cannot see a
    CUDA tensor or a dtype numpy has no analogue for (bf16); every captured
    value passes through here first. Widening bf16 to f32 is lossless -- it is
    storage, not a second rounding -- but it does not undo the first one; see
    ``artifact_dtypes`` in ``conventions.json``.
    """
    if torch.is_tensor(value):
        tensor = value.detach()
        if dtype is not None:
            tensor = tensor.to(dtype)
        return tensor.cpu().numpy()
    return np.asarray(value)


# --------------------------------------------------------------------------
# Taps
# --------------------------------------------------------------------------


def locate_seanet_stages(mimi_encoder) -> list[int]:
    """Find the four strided convolutions by their stride, not by their index.

    ``MimiEncoder.__init__`` (modeling_mimi.py:444-473) builds the stack by
    appending, so the stage convolutions land at layers.3/6/9/12 today -- but
    that is an emergent property of ``num_residual_layers == 1``, not a
    contract. Selecting on ``stride > 1`` states the intent, and the count
    assertion below turns a structural change into a failure here rather than
    into a mislabelled artifact downstream.
    """
    indices = [
        index
        for index, layer in enumerate(mimi_encoder.layers)
        if hasattr(layer, "conv") and int(layer.conv.stride[0]) > 1
    ]
    if len(indices) != 4:
        raise SystemExit(
            f"expected 4 strided SEANet convolutions, found {len(indices)} at {indices}; "
            "the encoder topology moved and the taps are stale"
        )
    return indices


def install_taps(mimi_model, captured: dict) -> list:
    """Capture the waveform and every stage output on the way past.

    A forward hook is a read, not a change: nothing here alters the forward.
    The waveform uses a *pre*-hook on the SEANet stack, which is the exact
    tensor ``_encode_frame`` hands it at modeling_mimi.py:1456 -- after the
    feature extractor, after the dtype/device cast, and before any padding.
    """
    handles = []
    encoder = mimi_model.encoder

    # Every tap counts its firings. MimiModel.encode calls _encode_frame exactly
    # once in 4.57.3 (modeling_mimi.py:1577), so a second firing cannot happen
    # today -- but if streaming were ever enabled, last-wins would leave the
    # stage artifacts describing the final chunk while codes.i32 described the
    # whole clip, and the codes check would still pass because the codes come
    # from upstream's own output. run_case turns a second firing into a failure.
    captured["tap_firings"] = {}

    def capture(key):
        def hook(_module, _inputs, output):
            # A transformer layer returns a tuple whose first element is the
            # hidden state (modeling_mimi.py:989); a convolution returns the
            # tensor itself.
            captured[key] = to_numpy(output[0] if isinstance(output, tuple) else output, torch.float32)
            captured["tap_firings"][key] = captured["tap_firings"].get(key, 0) + 1

        return hook

    def capture_input(key):
        def hook(_module, inputs):
            captured[key] = to_numpy(inputs[0], torch.float32)
            captured["tap_firings"][key] = captured["tap_firings"].get(key, 0) + 1

        return hook

    handles.append(encoder.register_forward_pre_hook(capture_input("waveform")))
    handles.append(encoder.register_forward_hook(capture("seanet_tail")))

    for stage, index in enumerate(locate_seanet_stages(encoder)):
        handles.append(encoder.layers[index].register_forward_hook(capture(f"seanet_stage{stage}")))

    for index, layer in enumerate(mimi_model.encoder_transformer.layers):
        handles.append(layer.register_forward_hook(capture(f"transformer_l{index}")))

    handles.append(mimi_model.downsample.register_forward_hook(capture("downsample")))

    # Input and output lengths of every MimiConv1d on the encode path, so the
    # published padding arithmetic can be checked against upstream's own
    # functions at the lengths this case actually produced rather than asserted
    # from a reading. See check_padding_arithmetic.
    captured["conv_lengths"] = {}

    def capture_length(name, module):
        def pre_hook(_module, inputs):
            captured["conv_lengths"].setdefault(name, {})["module"] = module
            captured["conv_lengths"][name]["input_length"] = int(inputs[0].shape[-1])

        def post_hook(_module, _inputs, output):
            captured["conv_lengths"][name]["output_length"] = int(output.shape[-1])

        return pre_hook, post_hook

    conv_modules = [
        (f"encoder.{name}", module)
        for name, module in encoder.named_modules()
        if type(module).__name__ == "MimiConv1d"
    ] + [("downsample", mimi_model.downsample)]
    for name, module in conv_modules:
        pre_hook, post_hook = capture_length(name, module)
        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))

    return handles


def published_extra_padding(length: int, kernel_effective: int, padding_total: int,
                            stride: int) -> int:
    """The formula ``conventions.json`` publishes, implemented exactly as written.

    Kept as its own function so ``check_padding_arithmetic`` compares the
    PUBLISHED rule against upstream, not a second private copy that could drift
    from the string. Upstream writes ``ceil(x + 1) - 1`` (modeling_mimi.py:269-270);
    that equals ``ceil(x)`` for integer 1, and this is the reduced form.
    """
    n_frames = -(-(length - kernel_effective + padding_total) // stride)  # ceiling divide
    ideal_length = n_frames * stride + kernel_effective - padding_total
    return ideal_length - length


def check_padding_arithmetic(conv_lengths: dict) -> list[dict]:
    """Run the published padding formula against upstream's own two functions.

    The causal-padding convention is one of the five the plan names as
    dangerous, and it is the one a reader of ``conventions.json`` implements
    from. So it is not transcribed and left: for every convolution on the encode
    path, at the real input length this case produced, the published rule is
    evaluated and compared against ``_get_extra_padding_for_conv1d`` and
    ``_get_output_length`` -- upstream's own arithmetic -- and against the output
    length the forward actually produced. A mismatch fails the run.
    """
    rows = []
    for name in sorted(conv_lengths):
        entry = conv_lengths[name]
        module = entry["module"]
        length = entry["input_length"]
        kernel_effective = int(module.kernel_size)
        padding_total = int(module.padding_total)
        stride = int(module.stride)

        published = published_extra_padding(length, kernel_effective, padding_total, stride)
        upstream = int(module._get_extra_padding_for_conv1d(
            torch.zeros(1, 1, length, device="meta")
        ))
        if published != upstream:
            raise SystemExit(
                f"{name}: the extra_padding formula conventions.json publishes gives "
                f"{published} at input length {length}, upstream's own "
                f"_get_extra_padding_for_conv1d gives {upstream}. Refusing to publish a "
                "padding convention that does not reproduce the padding actually applied."
            )

        upstream_output = int(module._get_output_length(length))
        actual_output = entry["output_length"]
        if upstream_output != actual_output:
            raise SystemExit(
                f"{name}: _get_output_length says {upstream_output} but the forward "
                f"produced {actual_output}"
            )
        rows.append({
            "module": name,
            "input_length": length,
            "kernel_effective": kernel_effective,
            "stride": stride,
            "padding_total": padding_total,
            "pad_left": padding_total,
            "pad_right": published,
            "extra_padding": published,
            "output_length": actual_output,
            "agrees_with_upstream": True,
        })
    return rows


def install_rvq_probes(mimi_model, captured: dict) -> list:
    """Capture, per RVQ stage, the residual entering it and its two best distances.

    A forward hook cannot see this: ``MimiEuclideanCodebook.quantize``
    (modeling_mimi.py:1204-1209) is a plain method called from ``encode``, and
    the residual it receives never leaves the loop at
    modeling_mimi.py:1279-1286. Each codebook instance therefore gets its
    ``quantize`` shadowed for the duration of the call, carrying its own
    ``(branch, stage)`` identity -- so the stage order recorded here is *read
    off the module graph*, never inferred from the order the calls happen to
    arrive in.

    The authoritative indices still come from upstream's own call: the wrapper
    delegates, and recomputes the distances only for the margin, asserting the
    recomputation reproduces upstream's argmin exactly. A divergence there
    means this probe -- not the port -- is the thing that is wrong, and it
    fails loudly instead of quietly publishing a margin for a different
    assignment.
    """
    restores = []
    quantizer = mimi_model.quantizer
    branches = (
        ("semantic", quantizer.semantic_residual_vector_quantizer),
        ("acoustic", quantizer.acoustic_residual_vector_quantizer),
    )
    captured["rvq_call_order"] = []
    captured["rvq"] = {}

    # The quantizer's own input -- the 512-wide latent, before either branch's
    # input_proj. It cannot be taken with a forward hook: _encode_frame calls
    # `self.quantizer.encode(...)` directly (modeling_mimi.py:1469), not through
    # __call__, so nn.Module's hook machinery never runs. Shadowing `encode` on
    # the instance is the same technique used for the codebooks below.
    quantizer_encode = quantizer.encode

    def encode_wrapper(embeddings, *args, _original=quantizer_encode, **kwargs):
        captured["latents"] = to_numpy(embeddings, torch.float32)
        return _original(embeddings, *args, **kwargs)

    quantizer.encode = encode_wrapper
    restores.append(lambda: quantizer.__dict__.pop("encode", None))

    for branch, rvq in branches:
        for stage, layer in enumerate(rvq.layers):
            codebook = layer.codebook
            original = codebook.quantize

            def wrapper(hidden_states, _original=original, _codebook=codebook,
                        _branch=branch, _stage=stage):
                embed_ind = _original(hidden_states)
                captured["rvq_call_order"].append([_branch, _stage])
                # Only the stages Qwen keeps are dumped; upstream evaluates all
                # 32 regardless (see conventions.json, quantizers_evaluated).
                position = _stage if _branch == "semantic" else _stage + 1
                if position < KEPT_QUANTIZER_COUNT:
                    # The same expression upstream uses at modeling_mimi.py:1207,
                    # recomputed because the original returns only the argmin.
                    dists = torch.cdist(
                        hidden_states[None].float(), _codebook.embed[None].float(), p=2
                    )[0]
                    if not torch.equal(dists.argmin(dim=-1), embed_ind):
                        raise SystemExit(
                            f"{_branch} stage {_stage}: the probe's recomputed distances do not "
                            "reproduce upstream's argmin -- the margin would describe a "
                            "different assignment than the codes"
                        )
                    best_two = torch.topk(dists, 2, dim=-1, largest=False).values
                    captured["rvq"][position] = {
                        "branch": _branch,
                        "stage": _stage,
                        "residual": to_numpy(hidden_states, torch.float32),
                        "indices": to_numpy(embed_ind).astype(np.int64),
                        # Euclidean, as upstream computes them (cdist p=2).
                        "d1": to_numpy(best_two[..., 0], torch.float32),
                        "d2": to_numpy(best_two[..., 1], torch.float32),
                    }
                return embed_ind

            codebook.quantize = wrapper
            restores.append(lambda cb=codebook: cb.__dict__.pop("quantize", None))
    return restores


# --------------------------------------------------------------------------
# Conventions
# --------------------------------------------------------------------------


def build_conventions(observed: dict) -> dict:
    """Every convention this encoder has that leaves no trace in the checkpoint.

    Each entry carries the ``file:line`` it was read from, in the pinned
    ``transformers==4.57.3`` wheel or the pinned Qwen revision, or -- where the
    value is something a reading cannot settle -- the measurement that settled
    it. A convention that is not here was guessed. This file, not
    ``result.json``, is what Tasks 3 and 4 implement against.
    """
    return {
        "schema": "synthesize-qwen3-tts-codec-encoder-conventions-v1",
        "upstream": {
            "note": (
                "THREE provenances, which is the trap this file exists to close. The "
                "codec encoder's reference implementation is transformers' MimiModel, "
                "not a Qwen source file; Qwen contributes a subclass that nulls the "
                "decoder halves and two post-steps; the checkpoint contributes the "
                "widths. Reading a convention off the wrong one produces finite codes "
                "that decode to plausible audio, so every line below says which file."
            ),
            "qwen": {
                "repository": UPSTREAM_REPOSITORY,
                "revision": UPSTREAM_REVISION,
                "files": [QWEN_TOKENIZER_FILE, QWEN_INFERENCE_FILE, QWEN_MODEL_FILE, QWEN_MODELING_FILE],
            },
            "transformers": {
                "version": TRANSFORMERS_VERSION,
                "files": [MIMI_FILE, MIMI_CONFIG_FILE],
            },
            "checkpoint": {
                "file": "models/qwen3-tts-12hz-0-6b-base/speech_tokenizer/config.json",
                "revision": "5d83992436eae1d760afd27aff78a71d676296fc",
            },
            "class_chain": (
                f"{QWEN_TOKENIZER_FILE}:899-908 -- "
                "`class Qwen3TTSTokenizerV2Encoder(MimiModel)` whose __init__ sets "
                "self.upsample = None, self.decoder_transformer = None, self.decoder = "
                "None. Observed on the loaded model: all three are None. Everything "
                "from the waveform to the codes is therefore unmodified MimiModel."
            ),
        },
        # ------------------------------------------------------------------
        # 1. Padding
        # ------------------------------------------------------------------
        "convolution_padding": {
            "causal": True,
            "pad_mode": "constant",
            "pad_value": 0.0,
            "padding_total_formula": "((kernel_size - 1) * dilation + 1) - stride",
            "causal_split": "all of padding_total on the LEFT; only extra_padding on the right",
            "extra_padding_formula": (
                "extra_padding = ideal_length - length, where "
                "n_frames = ceil((length - effective_kernel + padding_total) / stride) and "
                "ideal_length = n_frames * stride + effective_kernel - padding_total"
            ),
            "extra_padding_formula_note": (
                "Upstream writes this as `ceil((L - k + p)/s + 1) - 1` "
                f"({MIMI_FILE}:269-270). The `+ 1` is INSIDE the ceil and the `- 1` is "
                "outside it; since 1 is an integer, ceil(x + 1) - 1 == ceil(x), and the "
                "two reduce to the single ceil recorded above. Transcribing the outer "
                "`- 1` without the inner `+ 1` yields a formula one `stride` short -- it "
                "makes extra_padding come out -1 for the stem and -4 for stage 0 where "
                "the true value is 0, i.e. it turns a pad into a truncation. The form "
                "above is not merely transcribed: extra_padding_check below runs it "
                "against upstream's own _get_extra_padding_for_conv1d and "
                "_get_output_length, for every convolution on the encode path, at the "
                "real input lengths this case produced."
            ),
            "extra_padding_check": observed["extra_padding_check"],
            "source": (
                f"{MIMI_FILE}:221 (self.causal = config.use_causal_conv); :222 "
                "(self.pad_mode = config.pad_mode unless the constructor overrides it); "
                ":237-246 (effective kernel_size = (k-1)*dilation+1, registered "
                "padding_total = kernel_size - stride); :263-273 "
                "(_get_extra_padding_for_conv1d); :331-333 -- the causal branch calls "
                "_pad1d(hidden_states, (self.padding_total, extra_padding), "
                "mode=self.pad_mode), i.e. ALL of the fixed padding goes on the left and "
                "the right side receives only the ragged remainder. :249-250 computes a "
                "symmetric padding_left/padding_right split, but that is the NON-causal "
                "branch (:336-338) and this checkpoint never reaches it "
                "(use_causal_conv true). A port that pads symmetrically builds the same "
                "shapes from the same weights and a different encoder. :275-293 (_pad1d) "
                "special-cases 'reflect' only; for 'constant' it forwards straight to "
                "nn.functional.pad with value 0.0."
            ),
            "checkpoint_declares": {"use_causal_conv": True, "pad_mode": "constant"},
            "observed": observed["conv_padding"],
        },
        # ------------------------------------------------------------------
        # 2. Activation
        # ------------------------------------------------------------------
        "activation": {
            "kind": "elu",
            "alpha": 1.0,
            "placement": "BEFORE each convolution, never after",
            "source": (
                f"{MIMI_FILE}:418-419 -- inside MimiResnetBlock the block list is built as "
                "`block += [nn.ELU()]` then `block += [MimiConv1d(...)]`, per convolution, "
                "so the ELU precedes its convolution; :463 -- an nn.ELU() is appended "
                "immediately BEFORE each downsampling MimiConv1d at :465; :468 -- one more "
                "before the tail convolution at :470. There is no activation before the "
                "stem convolution at :449 and none after the tail. nn.ELU()'s alpha "
                "defaults to 1.0 (no argument is passed at any of the three sites)."
            ),
            "not_gelu": (
                "The checkpoint's encoder_config.hidden_act is 'gelu', and it belongs to "
                "the TRANSFORMER's MLP (MimiMLP, modeling_mimi.py:577-587, "
                "ACT2FN[config.hidden_act]) -- not to the SEANet stack, which is ELU "
                "everywhere and reads no config field to decide that."
            ),
            "observed": observed["activation_modules"],
        },
        # ------------------------------------------------------------------
        # 3 & 4. The residual unit
        # ------------------------------------------------------------------
        "residual_unit": {
            "kernel_sizes": [3, 1],
            "dilations": [1, 1],
            "dilation_expansion": (
                "dilations are [dilation_growth_rate ** j, 1] for j in "
                "range(num_residual_layers). This checkpoint has num_residual_layers = 1, "
                "so j is only ever 0 and dilation_growth_rate = 2 expands to 2**0 = 1: "
                "EVERY residual convolution in this encoder is dilation 1, at every "
                "stage. dilation_growth_rate is declared and inert. A port that reads "
                "'growth rate 2' as per-stage dilations 1/2/4/8 gets a different encoder "
                "from the same weights, and every shape still resolves."
            ),
            "bottleneck": "dim // compress = dim // 2",
            "use_conv_shortcut": False,
            "shortcut_module": "nn.Identity",
            "residual_added": "after the block, before the stride",
            "source": (
                f"{MIMI_FILE}:409 (kernel_sizes = (config.residual_kernel_size, 1) = (3, 1)); "
                ":413 (hidden = dim // config.compress, compress = 2); :415-419 (in_chs is "
                "dim for the first convolution and hidden thereafter; out_chs is dim for "
                "the last and hidden otherwise -- so the pair is dim->dim/2 at kernel 3 "
                "then dim/2->dim at kernel 1); :461 (MimiResnetBlock(config, current_scale, "
                "[config.dilation_growth_rate ** j, 1]) with j from "
                "range(config.num_residual_layers)); :422-425 (use_conv_shortcut False "
                "makes self.shortcut an nn.Identity, so the skip is the unmodified input, "
                "NOT a 1x1 convolution); :441 (`return residual + hidden_states` -- the "
                "add closes the block); :459-465 (the residual blocks are appended BEFORE "
                "the ELU and the strided convolution of the same stage, so the residual "
                "is added at the stage's INPUT rate and the stride is applied to the sum)."
            ),
            "observed": observed["residual_units"],
        },
        # ------------------------------------------------------------------
        # 5 & 6. The split RVQ
        # ------------------------------------------------------------------
        "split_rvq": {
            "concatenated_order": "semantic stages first, then acoustic stages",
            "semantic_stage_count": 1,
            "acoustic_stage_count_available": 31,
            "acoustic_stage_count_kept": 15,
            "acoustic_input_is_the_original_latent": True,
            "acoustic_input_note": (
                "THE TRAP IN THIS SUBSYSTEM. The acoustic RVQ is fed `embeddings` -- the "
                "SAME 512-wide latent the semantic RVQ was fed -- and NOT the semantic "
                "branch's leftover residual. The two branches are independent residual "
                "chains over one shared input, each with its own input_proj. A port that "
                "chains them (acoustic starting from the semantic residual) produces "
                "finite codes for every frame and a different voice."
            ),
            "residual_update": "residual = residual - codebook.decode(indices)",
            "residual_space": (
                "the input_proj'ed space (512 -> 256, 1x1 convolution, no bias), per "
                "branch. The subtrahend is the DEQUANTIZED codebook row, looked up by "
                "index; output_proj is NOT applied -- it exists only on the decode path."
            ),
            "distance_space": "the same 256-wide projected space, per frame",
            "distance_metric": "euclidean (torch.cdist p=2), computed in float32",
            "distance_is_squared": False,
            "codebooks_l2_normalized": False,
            "source": (
                f"{MIMI_FILE}:1318-1345 (MimiSplitResidualVectorQuantizer.encode): :1337 "
                "encodes the semantic branch, :1340-1342 encodes the acoustic branch "
                "**passing `embeddings` again, not the semantic residual**, :1343 "
                "concatenates on dim 0 with the semantic codes leading; "
                ":1269-1287 (MimiResidualVectorQuantizer.encode): :1274-1275 applies "
                "input_proj when vector_quantization_hidden_dimension != hidden_size "
                "(256 != 512, so it always applies here), :1281-1285 is the residual "
                "loop -- `quantized = layer.decode(indices)` then `residual = residual - "
                "quantized`; :1243-1246 (MimiVectorQuantization.decode) is a plain "
                "codebook lookup plus a permute, with NO output_proj, so the subtrahend "
                "is the dequantized 256-wide row; :1259-1267 constructs input_proj and "
                "output_proj as bias-free 1x1 Conv1d, and :1298-1299 shows output_proj "
                "is applied only in decode(); :1238-1241 permutes [B, C, T] -> [B, T, C] "
                "before quantizing, so distances are per frame; :1204-1209 "
                "(MimiEuclideanCodebook.quantize) is "
                "`torch.cdist(hidden_states[None].float(), self.embed[None].float(), p=2)` "
                "then `.argmin(dim=-1)` -- plain Euclidean distance on unnormalized "
                "vectors, cast to float32 for the distance itself. There is no "
                "normalization, no cosine similarity, and no temperature anywhere on "
                "this path."
            ),
            "order_verified_by": (
                "measurement, not reading. Each codebook module was shadowed with its own "
                "(branch, stage) identity, and run_case asserts that the concatenated "
                "codes column j equals the argmin of the semantic stage 0 probe for j = 0 "
                "and of acoustic stage j-1 for j >= 1. See conventions.json's "
                "rvq_call_order_observed and the equality check in run_case."
            ),
            "rvq_call_order_observed": observed["rvq_call_order_summary"],
            "quantizers_evaluated": {
                "evaluated_by_upstream": 32,
                "kept": KEPT_QUANTIZER_COUNT,
                "note": (
                    f"{QWEN_TOKENIZER_FILE}:981-982 calls self.encoder.encode(...) with no "
                    "num_quantizers, and modeling_mimi.py:1543 then defaults it to "
                    "config.num_quantizers = 32. Upstream therefore evaluates all 32 "
                    "stages and Qwen discards 16 of them one line later (:983). A port "
                    "that evaluates only the 16 it keeps is still correct: within each "
                    "branch the residual chain is strictly forward, so stages 16..31 "
                    "cannot affect the codes of stages 0..15. The catalog resolves 1 + 31 "
                    "codebooks (catalog.cpp:434, kCodecEncoderAcousticQuantizerCount = 31) "
                    "of which sixteen are never evaluated -- intended, not a bug: the "
                    "package carries the whole checkpoint rather than a pruned copy."
                ),
            },
            "ema_at_inference": {
                "ema_update_live": False,
                "codebook_is_derived_not_stored": True,
                "derivation": "embed = embed_sum / cluster_usage.clamp(min=1e-5)[:, None]",
                "source": (
                    f"{MIMI_FILE}:1186-1202 -- the module registers `initialized`, "
                    "`cluster_usage` and `embed_sum` as BUFFERS and has no `embed` "
                    "parameter at all; `embed` is a cached @property that divides the two "
                    "accumulators, with epsilon 1e-5 (the __init__ default at :1186). "
                    "Nothing writes back to either buffer on the encode path: there is no "
                    "cluster update, no decay, no dead-code re-initialization anywhere in "
                    "MimiEuclideanCodebook. The accumulators are frozen training state "
                    "that happens to be the only place the codebook is stored -- which is "
                    "exactly why scripts/convert-qwen3-tts.py:370-399 collapses each "
                    "embed_sum/cluster_usage pair into one `.codebook` tensor (with the "
                    "same RVQ_EPS = 1e-5 clamp, convert-qwen3-tts.py:92) and :486-489 "
                    "drops the shape-(1,) `.initialized` flags. `initialized` is read by "
                    "nothing at inference."
                ),
                "clamp_binds": observed["cluster_usage_clamp_binds"],
                "cluster_usage_observed_min": observed["cluster_usage_min"],
                "clamp_note": (
                    "The 1e-5 clamp is inert for this checkpoint -- the smallest "
                    "cluster_usage across all 32 codebooks (1 semantic + 31 acoustic) "
                    "is orders of magnitude above "
                    "it -- so the converter's baked codebook and upstream's derived one "
                    "differ only by arithmetic precision, not by the clamp. See "
                    "artifact_dtypes: the precision difference is NOT negligible."
                ),
            },
        },
        # ------------------------------------------------------------------
        # 7. trim_right_ratio
        # ------------------------------------------------------------------
        "trim_right_ratio": {
            "value": 1.0,
            "applies_to": "MimiConvTranspose1d only -- the decoder's transposed convolutions",
            "reaches_the_encode_path": False,
            "source": (
                f"{MIMI_FILE}:359 (self.trim_right_ratio = config.trim_right_ratio is set in "
                "MimiConvTranspose1d.__init__, and nowhere else in the file); :373-381 "
                "(causal branch: padding_right = ceil(padding_total * trim_right_ratio), so "
                "1.0 trims the entire fixed padding from the right and padding_left becomes "
                "0); :393-399 (the trim is applied after the transposed convolution). "
                "MimiConv1d -- every convolution on the encode path -- never mentions it. "
                "The only MimiConvTranspose1d in MimiModel is `self.upsample` "
                "(:1417-1425), which Qwen3TTSTokenizerV2Encoder sets to None "
                f"({QWEN_TOKENIZER_FILE}:904). Observed on the loaded model: "
                "upsample/decoder_transformer/decoder are all None. A port of the ENCODER "
                "must not implement trim_right_ratio at all; it is decoder geometry that "
                "the shared config happens to carry."
            ),
        },
        # ------------------------------------------------------------------
        # 8. Qwen's two post-steps
        # ------------------------------------------------------------------
        "qwen_post_steps": {
            "quantizer_slice": {
                "expression": "audio_codes[:, :encoder_valid_num_quantizers]",
                "value": KEPT_QUANTIZER_COUNT,
                "source": f"{QWEN_TOKENIZER_FILE}:983",
                "reads_from": (
                    "the TOP-LEVEL key `encoder_valid_num_quantizers` of "
                    "speech_tokenizer/config.json (16), read at "
                    f"{QWEN_TOKENIZER_FILE}:933 from the top-level config. It is NOT "
                    "`encoder_config.num_quantizers`, which is 32 -- and 32 is exactly the "
                    "wrong answer this slice exists to prevent."
                ),
            },
            "frame_trim": {
                "expression": "code[..., :-(-mask.sum() // encode_downsample_rate)]",
                "arithmetic": "ceiling divide: -(-n // d) == ceil(n / d)",
                "source": f"{QWEN_TOKENIZER_FILE}:984",
                "reads_from": (
                    "the TOP-LEVEL key `encode_downsample_rate` (1920), read at "
                    f"{QWEN_TOKENIZER_FILE}:939. `encoder_config` has no downsample rate "
                    "at all -- an implementer who looks for one there finds nothing and "
                    "invents it."
                ),
                "layout_change": (
                    "the same line transposes (0, 1): the codes leave as [frames, "
                    "quantizers], not [quantizers, frames]."
                ),
                "observed": observed["frames"],
            },
        },
        # ------------------------------------------------------------------
        # Geometry, reconciled against the package
        # ------------------------------------------------------------------
        "geometry": {
            "note": (
                "The package publishes NO codec.encoder.* metadata namespace; the "
                "catalog's encoder widths are compiled-in literals. This block is the "
                "reconciliation between the checkpoint's declared values and what the "
                "port derives, checked here rather than discovered in Task 3."
            ),
            "stride_order": {
                "encoder_walk_order": [4, 5, 6, 8],
                "checkpoint_key": "encoder_config.upsampling_ratios",
                "checkpoint_value": [8, 6, 5, 4],
                "resolved_by": (
                    f"{MIMI_FILE}:456 -- `for ratio in reversed(config.upsampling_ratios)`. "
                    "The key is named for the DECODER's walk; the encoder walks it "
                    "backwards. So the encoder's strides, in order, are 4, 5, 6, 8, its "
                    "kernels (ratio * 2 at :465) are 8, 10, 12, 16, and its widths "
                    "(current_scale doubling at :457/:466 from num_filters = 64) are 128, "
                    "256, 512, 1024. That matches catalog.cpp:430-431 exactly -- the port "
                    "already has the right order, and this records WHY rather than "
                    "leaving it to coincidence. A reversed order gives the same 960x "
                    "total downsampling and a different everything else, which the "
                    "frame-count test cannot see."
                ),
                "verified_by": "measurement on the instantiated modules; see observed below",
                "observed": observed["seanet_stages"],
            },
            "total_downsampling": {
                "seanet": 960,
                "frame_downsampler": 2,
                "samples_per_frame": 1920,
                "arithmetic": "4 * 5 * 6 * 8 = 960, times the downsample conv's stride 2 = 1920",
                "cross_check": (
                    f"{MIMI_CONFIG_FILE}:247-269 (MimiConfig.frame_size multiplies exactly "
                    "these strides) -- observed frame_size = "
                    f"{observed['config_frame_size']}, frame_rate = "
                    f"{observed['config_frame_rate']} Hz, encodec_frame_rate = "
                    f"{observed['config_encodec_frame_rate']} Hz "
                    f"({MIMI_CONFIG_FILE}:237-240)."
                ),
            },
            "frame_downsampler": {
                "kernel": 4,
                "stride": 2,
                "in_channels": 512,
                "out_channels": 512,
                "bias": False,
                "groups": 1,
                "pad_mode": "replicate",
                "pad_mode_note": (
                    "NOT 'constant'. modeling_mimi.py:1406-1415 constructs the downsample "
                    "MimiConv1d with an explicit pad_mode='replicate' keyword, which "
                    "modeling_mimi.py:222 lets override the config's 'constant'. This is "
                    "the ONLY convolution in the encode path that does not zero-pad: it "
                    "repeats its first frame padding_total (= 2) times on the left. The "
                    "catalog records kernel 4 / 512->512 / no bias "
                    "(catalog.cpp:432, :546-547) but cannot record a padding mode, so a "
                    "port that pads it with zeros like its neighbours corrupts exactly "
                    "the first frame of every clip -- one frame in 101, which no summary "
                    "statistic will show."
                ),
                "kernel_derivation": (
                    "kernel_size = 2 * int(encodec_frame_rate / frame_rate) = 2 * int(25 / "
                    "12.5) = 4, stride = 2, at modeling_mimi.py:1406-1415; the module is "
                    "created at all only because frame_rate != encodec_frame_rate (:1405)."
                ),
                "source": f"{MIMI_FILE}:1406-1415",
                "observed": observed["downsample"],
            },
            "transformer": {
                "layers": 8,
                "hidden": 512,
                "heads": 8,
                "head_dim": 64,
                "intermediate": 2048,
                "norm": "LayerNorm (weight and bias), eps 1e-05",
                "layer_scale": "per-branch learnt diagonal scale, applied to the branch output before the add",
                "activation": "gelu (encoder_config.hidden_act)",
                "position": "AFTER the SEANet stack and BEFORE the frame downsampler",
                "position_source": (
                    f"{MIMI_FILE}:1456-1467 -- encoder, then encoder_transformer on the "
                    "transposed [B, T, C] view, then downsample on the transposed-back "
                    "[B, C, T]. The design's file list implies the downsample comes "
                    "first; it does not."
                ),
                "source": (
                    f"{MIMI_FILE}:922-994 (MimiTransformerLayer: input_layernorm, "
                    "self_attn, `residual + self_attn_layer_scale(hidden_states)` at :981, "
                    "post_attention_layernorm, mlp, `residual + "
                    "mlp_layer_scale(hidden_states)` at :987); :489-501 (MimiLayerScale is "
                    "an elementwise learnt vector, initial value "
                    "layer_scale_initial_scale = 0.01); :577-587 (MimiMLP: fc1, "
                    "ACT2FN[hidden_act], fc2 -- a plain two-layer MLP, not a gated one)."
                ),
                "attn_implementation_observed": observed["attn_implementation"],
                "attn_implementation_note": (
                    "The codec encoder does NOT inherit the talker's "
                    "attn_implementation='eager'. "
                    f"{QWEN_MODELING_FILE}:1872 POPS attn_implementation out of kwargs "
                    "before :1915-1918 forwards what is left to "
                    "Qwen3TTSTokenizer.from_pretrained, so the speech tokenizer is loaded "
                    "with whatever transformers defaults to -- observed as "
                    f"{observed['attn_implementation']!r}. Recorded because a comparison "
                    "against these transformer artifacts is comparing against SDPA's "
                    "accumulation order, not eager's."
                ),
                "sliding_window": observed["sliding_window"],
                "sliding_window_note": (
                    "encoder_config.sliding_window is 250 frames. Every clip in the "
                    "current manifest is well under that (101 frames at most), so the "
                    "window has not been exercised by any artifact this script has "
                    "produced -- a port must not conclude from these dumps that it can "
                    "skip the window."
                ),
            },
            "quantizer_widths": {
                "latent_width": 512,
                "projected_width": 256,
                "codebook_size": 2048,
                "checkpoint_keys": {
                    "encoder_config.hidden_size": 512,
                    "encoder_config.codebook_dim": 256,
                    "encoder_config.vector_quantization_hidden_dimension": 256,
                    "encoder_config.codebook_size": 2048,
                },
                "port_derivation": (
                    "the port derives the projected width as codebook_dim / 2 from the "
                    "DECODER's codebook_dim of 512 (catalog.cpp:409, "
                    "resolve_codec_encoder_quantizer's `const int64_t inner = "
                    "p.codebook_dim / 2`), and the codebook as {inner, codebook_size} = "
                    "{256, 2048} (catalog.cpp:413). The encoder's own config states 256 "
                    "directly; the two agree, and this is where that is checked."
                ),
                "observed": observed["quantizer_widths"],
            },
            "keys_that_are_top_level_not_encoder_config": {
                "encoder_valid_num_quantizers": 16,
                "encode_downsample_rate": 1920,
                "decoy_values_inside_encoder_config": {
                    "num_quantizers": 32,
                    "codebook_dim": 256,
                },
                "note": (
                    "Both quantities live at the TOP LEVEL of "
                    "speech_tokenizer/config.json. Inside encoder_config sit "
                    "num_quantizers = 32 and no downsample rate at all."
                ),
            },
            "hop_length_metadata_caveat": (
                "The package's synthesize.qwen3-tts.codec.hop_length is written from the "
                "checkpoint's top-level `decode_upsample_rate`, not "
                "`encode_downsample_rate` (scripts/convert-qwen3-tts.py:636-638 asserts "
                "int(codec_config['decode_upsample_rate']) == SAMPLES_PER_FRAME, and :643 "
                "writes it). Both are 1920 in this checkpoint, so hop_length is the right "
                "number for the encoder's frame trim today -- but it is the DECODE rate "
                "by construction, and a future checkpoint where the two differ would make "
                "the encoder's trim silently wrong. The encoder's trim divisor is "
                "`encode_downsample_rate` and nothing else."
            ),
        },
        # ------------------------------------------------------------------
        # Numerics
        # ------------------------------------------------------------------
        "artifact_dtypes": {
            "runtime_dtype": observed["runtime_dtype"],
            "device": observed["device"],
            "note": (
                "write_f32 always emits float32 on disk, and that hides the history. The "
                "codec encoder runs in the talker's bfloat16 -- "
                f"{QWEN_MODELING_FILE}:1915-1918 forwards dtype=torch.bfloat16 to "
                "Qwen3TTSTokenizer.from_pretrained -- even though "
                "encoder_config declares \"dtype\": \"float32\". Every stage artifact here "
                "is a bfloat16 value widened losslessly to float32, so a tolerance tighter "
                "than bfloat16's ~2^-8 relative precision is chasing the oracle's own "
                "rounding. The one exception is the distances, which upstream casts to "
                "float32 before computing (modeling_mimi.py:1207)."
            ),
            "waveform": {
                "runtime_dtype": observed["waveform_dtype"],
                "note": (
                    "The feature extractor emits float32 and "
                    f"{QWEN_INFERENCE_FILE}:248 then casts it to the model dtype with "
                    ".to(self.model.dtype) BEFORE model.encode sees it. waveform.f32 is "
                    "captured at the SEANet stack's input, i.e. after that cast, so it is "
                    "the exact tensor the first convolution consumed."
                ),
            },
            "codebook_precision_divergence": {
                "oracle_codebook_dtype": observed["codebook_dtype"],
                "port_codebook_dtype": "float32",
                "note": (
                    "THE SHARPEST NUMERICAL DIFFERENCE ON THIS PATH, and it is not in the "
                    "port's favour to ignore. Upstream derives the codebook at runtime as "
                    "embed_sum / cluster_usage.clamp(1e-5) (modeling_mimi.py:1198-1202) "
                    "from buffers that from_pretrained has already cast to bfloat16 -- so "
                    "the DIVISION ITSELF is done in bfloat16 and its result is a bfloat16 "
                    "table, later widened by .float() at :1207 for the distance. The "
                    "converter performs the same division in float32 from the float32 "
                    "safetensors (convert-qwen3-tts.py:376-377, :388) and bakes a float32 "
                    "table. The port's codebook is therefore MORE accurate than the "
                    "oracle's, and the two tables differ by roughly bfloat16 rounding. "
                    "That difference lands directly in the argmin, which is why "
                    "min_distance_margin below is the number that decides whether the "
                    "gate can be plain equality -- and why a port that reproduces this "
                    "oracle's codes bit-exactly is not thereby proven correct on a "
                    "different clip."
                ),
            },
            "distances": {
                "computed_in": "float32",
                "source": f"{MIMI_FILE}:1207 -- .float() on both operands inside cdist",
            },
            "bfloat16_round_trip_evidence": (
                "Observed, not inferred from the model's stated dtype: casting "
                "waveform.f32, seanet_tail.f32, latents.f32 and rvq_residual_s00.f32 to "
                "bfloat16 and back to float32 round-trips EXACTLY (max abs diff 0.0) -- "
                "they carry bfloat16 precision and nothing more. The same round-trip on "
                "rvq_distance_margin.f32 does not (max abs diff ~99.7), because the "
                "distances are genuinely float32. Budget tolerances accordingly: ~2^-8 "
                "relative for every stage artifact, float32 for the margin."
            ),
        },
        # ------------------------------------------------------------------
        # The tie margin
        # ------------------------------------------------------------------
        "tie_margin": {
            "why": (
                "RVQ assignment is an argmin over codebook distances. Two implementations "
                "that differ only in float32 accumulation order flip an assignment exactly "
                "when the gap between the best and second-best entry is smaller than that "
                "difference. docs/port-validation.md handles flips with "
                "oracle.alternate_grids, but the design (section 6) requires MEASURING "
                "whether they occur before choosing a gate -- so this is an artifact, not "
                "a diagnostic print."
            ),
            "artifact": "rvq_distance_margin.f32",
            "layout": "[kept_stages, frames] float32, row-major (frames contiguous)",
            "definition": "squared-distance margin: d2**2 - d1**2, over the 2048 codebook entries",
            "computed_as": (
                "(d2 - d1) * (d2 + d1), which is algebraically d2**2 - d1**2 but far "
                "better conditioned: the margin is a gap between two nearly equal "
                "distances, so evaluating it as a difference of two float32 squares of "
                "similar magnitude loses most of the significant digits at exactly the "
                "frames that matter most. Measured on this family's own artifacts, the "
                "naive form is off by roughly half a percent at the tightest margins."
            ),
            "why_squared": (
                "Upstream computes NON-squared Euclidean distances (torch.cdist p=2, "
                "modeling_mimi.py:1207); a port will almost certainly expand ||x-e||^2 "
                "instead, since the argmin is identical and the square root is wasted "
                "work. The flip SET is the same either way, but the numerical margin a "
                "port's own error must be compared against is the one in the space the "
                "port accumulates in -- squared. Both are recorded below; the artifact is "
                "the squared one."
            ),
            "min_distance_margin": observed["min_distance_margin"],
            "min_distance_margin_euclidean": observed["min_distance_margin_euclidean"],
            "min_relative_margin": observed["min_relative_margin"],
            "min_relative_margin_definition": "(d2**2 - d1**2) / d1**2, minimum over all stages and frames",
            "exact_ties_observed": observed["exact_ties"],
            "per_stage_min_margin": observed["per_stage_min_margin"],
            "measurement_coverage": (
                "The three ICL cases this script is run on give TWO distinct measurements, "
                "not three -- and, stated precisely, they are ONE SOURCE FILE AT TWO "
                "LENGTHS, not two clips. All three cases point at the same "
                "models/qwen3-tts-reference-audio/clone.wav. base-icl-en and "
                "base-text-short take it whole and differ only in synthesis text, which "
                "the codec encoder never reads: all 34 artifacts are byte-identical "
                "between them (confirmed by sha256 across every .f32 and .i32). That "
                "repeat is worth having as proof the path is deterministic case to case "
                "with no hidden state leaking in, but it is one measurement taken twice. "
                "base-ref-min is that same file's first 1.0 s. It is nonetheless a "
                "genuinely independent measurement rather than a prefix of the first: the "
                "encoder transformer is not causal, so shortening the input changes the "
                "interior -- 37 of its 208 codes differ from the corresponding prefix of "
                "the full-clip codes, and latents differ by up to 17.2 against an rms of "
                "4.1. So the numbers below rest on two real points, both drawn from one "
                "speaker in one recording. No second speaker, no second recording "
                "condition, no second sample rate. A gate chosen from them is chosen on "
                "that basis."
            ),
            "interpretation": (
                "NOT a conclusion, a measurement -- the gate is Task 12's decision. What "
                "the numbers say: no exact tie occurs, but the smallest margin is very "
                "tight in relative terms (min_relative_margin ~8.5e-6 of the winning "
                "squared distance), and it is tight in the ACOUSTIC stages -- the "
                "semantic stage's smallest margin is four orders of magnitude larger. "
                "Weigh that against codebook_precision_divergence above: the port's "
                "float32 codebook is not the same table as this oracle's bfloat16 one, "
                "and that difference is not obviously smaller than the margin it has to "
                "clear. Plain equality may well hold on these two measurements -- one "
                "recording at two lengths, see measurement_coverage -- and not on a "
                "different speaker."
            ),
        },
        # ------------------------------------------------------------------
        # On-disk layout
        # ------------------------------------------------------------------
        "artifact_layout": {
            "note": (
                "Every .f32 here is written by write_f32, which np.ascontiguousarray()s "
                "(row-major / C order) then calls .tobytes() once: no header, native "
                "little-endian float32. codes.i32 is the same with int32. The LAST axis "
                "listed is the contiguous one."
            ),
            "waveform.f32": ["samples"],
            "seanet_stage0.f32": ["128 channels", "frames"],
            "seanet_stage1.f32": ["256 channels", "frames"],
            "seanet_stage2.f32": ["512 channels", "frames"],
            "seanet_stage3.f32": ["1024 channels", "frames"],
            "seanet_tail.f32": ["512 channels", "frames"],
            "transformer_l0..l7.f32": ["frames", "512 hidden"],
            "transformer_layout_note": (
                "channel-LAST, unlike every convolution artifact here, because "
                "_encode_frame transposes to [B, T, C] before the transformer "
                "(modeling_mimi.py:1460) and back afterwards (:1466). These artifacts are "
                "in the transformer's own layout, not renormalized to the convolutions'."
            ),
            "downsample.f32": ["512 channels", "frames"],
            "latents.f32": ["512 channels", "frames"],
            "latents_note": (
                "The quantizer's input, captured at the quantizer boundary. Upstream "
                "hands it the downsample output with nothing in between "
                "(modeling_mimi.py:1467-1469), so this is byte-identical to "
                "downsample.f32 -- asserted per case on the exact bytes written, not "
                "assumed, with the outcome published as "
                "latents_equals_downsample_observed immediately below. Kept as a separate "
                "file because it is the seam Tasks 3/4 compare at. The 256-wide PROJECTED "
                "latents each branch actually quantizes are rvq_residual_s00 (semantic "
                "input_proj) and rvq_residual_s01 (acoustic input_proj); they are not the "
                "same tensor as this one and not the same as each other."
            ),
            "latents_equals_downsample_observed": observed["latents_equals_downsample"],
            "rvq_residual_s00..s15.f32": ["frames", "256 projected"],
            "rvq_residual_note": (
                "The residual ENTERING each kept stage, in concatenated-code order: s00 "
                "is the semantic branch's stage 0 and s01..s15 are the acoustic branch's "
                "stages 0..14. s00 and s01 are each branch's input_proj applied to the "
                "same latents, which is why they differ."
            ),
            "rvq_distance_margin.f32": ["16 stages", "frames"],
            "codes.i32": ["frames", "16 quantizers"],
            "codes_note": (
                "Post-slice and post-trim -- the same tensor "
                "dump_reference_qwen3_tts_base.py writes as codes/reference.i32, and "
                "asserted byte-identical to it BEFORE any artifact in this directory is "
                "written. See codes_verified_against_base_oracle: when it is false, the "
                "check was skipped under --allow-unverified and every file here is "
                "unvouched-for."
            ),
            "codes_verified_against_base_oracle": observed["codes_verified_against_base_oracle"],
            "base_oracle_reference_codes": observed["base_oracle_reference_codes"],
        },
        # ------------------------------------------------------------------
        # Frame counts
        # ------------------------------------------------------------------
        "frames": observed["frames"],
        # ------------------------------------------------------------------
        # Provenance for the things a reading could not settle
        # ------------------------------------------------------------------
        "provenance": {
            "stride_order": (
                "Observed, not read off either list: enumerated the instantiated "
                "MimiEncoder.layers on the loaded checkpoint and recorded each strided "
                "MimiConv1d's (kernel, stride, in_channels, out_channels). The result is "
                "in geometry.stride_order.observed and it is what the encoder walks, "
                "independent of how upsampling_ratios is written down."
            ),
            "rvq_stage_order": (
                "Observed in three independent parts, none of them tautological. (a) "
                "num_semantic_quantizers is read off the module and asserted to be 1 -- "
                "that, not a hardcoded rule, is what makes column 0 semantic. (b) The "
                "firing order is recorded as the calls arrive and asserted to be one "
                "semantic stage followed by the acoustic stages in index order. (c) The "
                "load-bearing one: each MimiEuclideanCodebook was shadowed carrying its "
                "own (branch, stage) identity read off the module graph, and run_case "
                "asserts that upstream's OWN codes[:, position] equal the argmin this "
                "script probed for the stage it claims that column is, for all 16. A "
                "concatenation the other way round fails (c); a reordered branch walk "
                "fails (b); a checkpoint with a wider semantic branch fails (a)."
            ),
            "acoustic_branch_input": (
                "Read at modeling_mimi.py:1340-1342 and corroborated numerically: "
                "rvq_residual_s01 (the acoustic branch's first residual) is NOT the "
                "semantic branch's post-subtraction leftover -- it is a different "
                "projection of the same latents. The two files are dumped separately so "
                "a port that chained the branches would disagree on s01 immediately, at "
                "the stage where the disagreement still has a magnitude."
            ),
            "distance_margin": (
                "Measured, not estimated: for each kept stage the probe recomputes the "
                "exact expression upstream uses (torch.cdist(..., p=2) on float32 "
                "operands), asserts its argmin reproduces upstream's returned indices "
                "bit-for-bit, and takes the two smallest by torch.topk(k=2, "
                "largest=False). The recorded minima are over every stage and every frame "
                "of this case."
            ),
            "codebook_precision_divergence": (
                "Observed: MimiEuclideanCodebook.embed's dtype was read off the loaded "
                "model and is bfloat16, because from_pretrained cast embed_sum and "
                "cluster_usage before the @property divided them. Cross-read against "
                "scripts/convert-qwen3-tts.py:376-377 (.to(torch.float32), one line per "
                "operand -- :376 embedding_sum, :377 cluster_usage) and :388 (the "
                "division), which is float32. The two codebooks are not the same table."
            ),
            "cluster_usage_clamp": (
                "Observed: the minimum cluster_usage over all 32 encoder codebooks "
                "(1 semantic + 31 acoustic) was recorded and compared against the 1e-5 "
                "clamp; see split_rvq.ema_at_inference.clamp_binds."
            ),
            "attn_implementation": (
                "Observed on the loaded model "
                "(encoder_transformer._attn_implementation), not assumed from the "
                "attn_implementation argument passed to Qwen3TTSModel.from_pretrained -- "
                "which does not reach this submodel, because "
                f"{QWEN_MODELING_FILE}:1872 pops it out of kwargs first."
            ),
            "latents_equals_downsample": (
                "Asserted per case on the exact bytes written, and published as the "
                "boolean artifact_layout.latents_equals_downsample_observed in THIS file "
                "rather than only in result.json. A mismatch aborts the run before any "
                "artifact reaches disk, so the claim in artifact_layout.latents_note "
                "cannot outlive the fact it describes."
            ),
            "frames": (
                "Observed from this case's own artifacts: the encoder's untrimmed output "
                "frame count from downsample.f32, the post-trim count from codes.i32, and "
                "the trim divisor's arithmetic recomputed from the sample count this case "
                "actually produced. The ceiling divide and the observed frame count are "
                "not merely both recorded -- run_case asserts they agree."
            ),
            "extra_padding_formula": (
                "Not transcribed and left: the formula string published under "
                "convolution_padding is implemented verbatim by published_extra_padding() "
                "and run, for every convolution on the encode path at this case's real "
                "input lengths, against upstream's own "
                "_get_extra_padding_for_conv1d and _get_output_length. Any disagreement "
                "aborts the run before anything is written. Per-convolution results are "
                "in convolution_padding.extra_padding_check. This check exists because an "
                "earlier revision of this file published the formula with upstream's "
                "outer '- 1' but without its inner '+ 1', which reads as a plausible "
                "transcription and turns every pad into a truncation."
            ),
            "codes_verified_against_base_oracle": (
                "The run fails outright when codes/reference.i32 is absent, unless "
                "--allow-unverified is passed -- in which case this flag is false and "
                "result.json's status is 'unverified' rather than 'ok'. A check that is "
                "skipped must not read like a check that passed."
            ),
        },
    }


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------


@dataclasses.dataclass
class RunCase:
    id: str
    ref_audio: str
    trim_seconds: Optional[float] = None
    # Set only by the manifest form, from the matching `source.artifacts` entry
    # (role reference-audio). The explicit-argument form has no manifest to pin
    # a digest against, so this stays None there.
    ref_sha256: Optional[str] = None


def build_case_from_args(args: argparse.Namespace) -> tuple[RunCase, pathlib.Path]:
    missing = [
        name for name, value in (("--ref-audio", args.ref_audio), ("--out-dir", args.out_dir))
        if value is None
    ]
    if missing:
        raise SystemExit(
            "explicit-argument form requires " + ", ".join(missing) +
            " (or pass --manifest together with --case)"
        )
    case = RunCase(id=args.out_dir.name, ref_audio=args.ref_audio, trim_seconds=args.trim_seconds)
    return case, args.out_dir


def load_cases_from_manifest(args: argparse.Namespace) -> list[tuple[RunCase, pathlib.Path]]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise SystemExit(f"{args.manifest}: unexpected manifest schema {manifest.get('schema')!r}")
    if manifest.get("family") != FAMILY:
        raise SystemExit(f"{args.manifest}: not a {FAMILY} manifest")

    output_root = args.output_root or pathlib.Path(manifest["case_artifact_root"])
    wanted = set(args.case) if args.case else None
    cases = [c for c in manifest["cases"] if wanted is None or c["id"] in wanted]
    if wanted and len(cases) != len(wanted):
        missing = wanted - {c["id"] for c in cases}
        raise SystemExit(f"unknown case id(s): {sorted(missing)}")

    specs: list[tuple[RunCase, pathlib.Path]] = []
    for case in cases:
        params = case.get("oracle", {}).get("parameters", {})
        reference = case.get("input", {}).get("reference", {})
        ref_audio = reference.get("artifact") or params.get("ref_audio")
        if ref_audio is None:
            raise SystemExit(
                f"{case['id']}: manifest case has no input.reference.artifact / "
                "oracle.parameters.ref_audio -- this script only dumps the codec "
                "encoder's reference-audio path, not text-only synthesis."
            )
        ref_sha256 = None
        for artifact in manifest.get("source", {}).get("artifacts", []):
            if artifact.get("role") == "reference-audio" and artifact.get("locator") == ref_audio:
                ref_sha256 = artifact.get("sha256")
                break
        specs.append((
            RunCase(
                id=case["id"],
                ref_audio=ref_audio,
                trim_seconds=params.get("trim_seconds"),
                ref_sha256=ref_sha256,
            ),
            output_root / case["id"],
        ))
    return specs


def resolve_reference_locator(
    locator: str, expected_sha256: Optional[str], cache_dir: pathlib.Path
) -> str:
    """Fetch an http(s) reference-audio locator to a local cache, verified by digest.

    Mirrors the sibling scripts' function of the same name: cache under a fixed
    local directory, skip re-fetching what is already there, verify content by
    digest rather than trusting the URL to keep serving the same bytes. A local
    path passes straight through unchanged.
    """
    if not (locator.startswith("http://") or locator.startswith("https://")):
        return locator
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / locator.rsplit("/", 1)[-1]
    if not destination.exists():
        print(f"fetching {locator}", flush=True)
        with urllib.request.urlopen(locator, timeout=60) as response:  # noqa: S310 - pinned https locator
            destination.write_bytes(response.read())
    if expected_sha256 is not None:
        actual = hashlib.sha256(destination.read_bytes()).hexdigest()
        if actual != expected_sha256:
            raise SystemExit(
                f"{destination}: sha256 {actual} does not match the manifest's "
                f"{expected_sha256}. Refusing to dump against an unpinned reference. "
                "If a stale cached file is the cause, delete it and re-run to re-fetch."
            )
    return str(destination)


def load_reference_audio(source: str, trim_seconds: Optional[float]) -> tuple[np.ndarray, int, dict]:
    """Load and optionally re-length the reference clip.

    Identical in behaviour to dump_reference_qwen3_tts_base.py's function of
    the same name -- a case like base-ref-min (trim_seconds 1.0) must reach the
    codec encoder with exactly the samples the Base oracle fed it, or codes.i32
    cannot be compared against codes/reference.i32 at all.
    """
    wav, sr = librosa.load(source, sr=None, mono=True)
    wav = wav.astype(np.float32)
    info: dict = {"source_samples": int(wav.shape[0]), "source_sample_rate": int(sr)}
    if trim_seconds is not None:
        target = int(round(trim_seconds * sr))
        if target <= 0:
            raise SystemExit(f"--trim-seconds must be positive, got {trim_seconds}")
        if target <= wav.shape[0]:
            wav = wav[:target]
            info["looped"] = False
            info["loop_repeats"] = 1
        else:
            repeats = -(-target // wav.shape[0])  # ceil division
            wav = np.tile(wav, repeats)[:target]
            info["looped"] = True
            info["loop_repeats"] = int(repeats)
        info["requested_seconds"] = trim_seconds
        info["applied_samples"] = int(wav.shape[0])
    return wav, sr, info


# --------------------------------------------------------------------------
# Observation
# --------------------------------------------------------------------------


def observe_static(mimi_model, tokenizer_model) -> dict:
    """Read the geometry off the instantiated modules, before any forward runs.

    Everything here could in principle be read off a config file; none of it is,
    because a config file states what was requested and a module states what was
    built. The two disagreeing is exactly the failure this file exists to catch.
    """
    encoder = mimi_model.encoder
    config = mimi_model.config

    stages = []
    for stage, index in enumerate(locate_seanet_stages(encoder)):
        conv = encoder.layers[index].conv
        stages.append({
            "stage": stage,
            "module": f"encoder.layers.{index}",
            "kernel": int(conv.kernel_size[0]),
            "stride": int(conv.stride[0]),
            "in_channels": int(conv.in_channels),
            "out_channels": int(conv.out_channels),
            "bias": conv.bias is not None,
        })

    residual_units = []
    for index, layer in enumerate(encoder.layers):
        if type(layer).__name__ != "MimiResnetBlock":
            continue
        convs = [
            {
                "sub_index": sub,
                "kernel": int(module.conv.kernel_size[0]),
                "dilation": int(module.conv.dilation[0]),
                "stride": int(module.conv.stride[0]),
                "in_channels": int(module.conv.in_channels),
                "out_channels": int(module.conv.out_channels),
            }
            for sub, module in enumerate(layer.block)
            if hasattr(module, "conv")
        ]
        residual_units.append({
            "module": f"encoder.layers.{index}",
            "shortcut": type(layer.shortcut).__name__,
            "convolutions": convs,
            "activations": [type(m).__name__ for m in layer.block if not hasattr(m, "conv")],
        })

    conv_padding = []
    for name, module in encoder.named_modules():
        if type(module).__name__ != "MimiConv1d":
            continue
        conv_padding.append({
            "module": f"encoder.{name}",
            "causal": bool(module.causal),
            "pad_mode": module.pad_mode,
            "padding_total": int(module.padding_total),
            "kernel_effective": int(module.kernel_size),
            "stride": int(module.stride),
        })
    downsample = mimi_model.downsample
    conv_padding.append({
        "module": "downsample",
        "causal": bool(downsample.causal),
        "pad_mode": downsample.pad_mode,
        "padding_total": int(downsample.padding_total),
        "kernel_effective": int(downsample.kernel_size),
        "stride": int(downsample.stride),
    })

    quantizer = mimi_model.quantizer
    usage_min = float("inf")
    codebook_dtype = None
    for rvq in (quantizer.semantic_residual_vector_quantizer,
                quantizer.acoustic_residual_vector_quantizer):
        for layer in rvq.layers:
            usage_min = min(usage_min, float(layer.codebook.cluster_usage.min().item()))
            codebook_dtype = str(layer.codebook.embed.dtype).replace("torch.", "")
    epsilon = quantizer.semantic_residual_vector_quantizer.layers[0].codebook.epsilon

    semantic_proj = quantizer.semantic_residual_vector_quantizer.input_proj
    acoustic_proj = quantizer.acoustic_residual_vector_quantizer.input_proj

    return {
        "seanet_stages": stages,
        "residual_units": residual_units,
        "conv_padding": conv_padding,
        "activation_modules": {
            "seanet_activation_types": sorted({
                type(module).__name__
                for module in encoder.modules()
                if type(module).__name__ in {"ELU", "GELU", "SiLU", "ReLU"}
            }),
            "elu_alpha": float(next(
                (module.alpha for module in encoder.modules() if type(module).__name__ == "ELU"),
                float("nan"),
            )),
        },
        "downsample": {
            "kernel": int(downsample.conv.kernel_size[0]),
            "stride": int(downsample.conv.stride[0]),
            "in_channels": int(downsample.conv.in_channels),
            "out_channels": int(downsample.conv.out_channels),
            "groups": int(downsample.conv.groups),
            "bias": downsample.conv.bias is not None,
            "pad_mode": downsample.pad_mode,
            "causal": bool(downsample.causal),
        },
        "quantizer_widths": {
            "semantic_stage_count": len(quantizer.semantic_residual_vector_quantizer.layers),
            "acoustic_stage_count": len(quantizer.acoustic_residual_vector_quantizer.layers),
            "max_num_quantizers": int(quantizer.max_num_quantizers),
            "semantic_input_proj": [int(semantic_proj.in_channels), int(semantic_proj.out_channels),
                                    int(semantic_proj.kernel_size[0]), semantic_proj.bias is not None],
            "acoustic_input_proj": [int(acoustic_proj.in_channels), int(acoustic_proj.out_channels),
                                    int(acoustic_proj.kernel_size[0]), acoustic_proj.bias is not None],
            "input_projs_are_distinct_modules": semantic_proj is not acoustic_proj,
            "codebook_shape": list(
                quantizer.semantic_residual_vector_quantizer.layers[0].codebook.embed.shape
            ),
        },
        "cluster_usage_min": usage_min,
        "cluster_usage_clamp_binds": bool(usage_min < epsilon),
        "codebook_dtype": codebook_dtype,
        "attn_implementation": mimi_model.encoder_transformer._attn_implementation,
        "sliding_window": int(config.sliding_window) if config.sliding_window else None,
        "config_frame_size": int(config.frame_size),
        "config_frame_rate": float(config.frame_rate),
        "config_encodec_frame_rate": int(config.encodec_frame_rate),
        "runtime_dtype": str(tokenizer_model.dtype).replace("torch.", ""),
        "decoder_halves_nulled": {
            "upsample": mimi_model.upsample is None,
            "decoder_transformer": mimi_model.decoder_transformer is None,
            "decoder": mimi_model.decoder is None,
        },
    }


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------


def run_case(model, case: RunCase, case_dir: pathlib.Path, reference_audio_dir: pathlib.Path,
             static: dict, device: str, allow_unverified: bool = False) -> dict:
    local_ref_audio = resolve_reference_locator(case.ref_audio, case.ref_sha256, reference_audio_dir)
    ref_wav, ref_sr, ref_info = load_reference_audio(local_ref_audio, case.trim_seconds)

    speech_tokenizer = model.model.speech_tokenizer
    mimi = speech_tokenizer.model.encoder

    sink: dict[str, Any] = {}
    handles = install_taps(mimi, sink)
    restores = install_rvq_probes(mimi, sink)
    started = time.time()
    try:
        # The exact call Qwen3TTSModel.create_voice_clone_prompt makes to produce
        # ref_code (qwen_tts/inference/qwen3_tts_model.py:427-428, then :451) --
        # a list of one waveform at its own sample rate. Going through this entry
        # point rather than reconstructing the feature-extractor call keeps the
        # padding mask, the dtype cast and the inference_mode context identical
        # to the Base oracle's, which is what makes codes.i32 comparable to
        # codes/reference.i32 at all. No text is read: the transcript affects the
        # two-track prompt, never the codec encoder.
        encoded = speech_tokenizer.encode([ref_wav], sr=ref_sr)
    finally:
        for handle in handles:
            handle.remove()
        for restore in restores:
            restore()
    wall_seconds = time.time() - started

    required = (
        ["waveform", "seanet_tail", "downsample", "latents"]
        + [f"seanet_stage{i}" for i in range(4)]
        + [f"transformer_l{i}" for i in range(len(mimi.encoder_transformer.layers))]
    )
    missing = [key for key in required if key not in sink]
    if missing:
        raise SystemExit(
            f"{case.id}: never observed {missing} -- the taps are stale or the forward "
            "path changed"
        )
    repeated = {key: count for key, count in sink["tap_firings"].items() if count != 1}
    if repeated:
        raise SystemExit(
            f"{case.id}: taps fired more than once ({repeated}). The stage artifacts "
            "would describe the last chunk while codes.i32 described the whole clip, and "
            "the codes check would still pass because the codes come from upstream's own "
            "output. Refusing to dump a chunked forward."
        )
    missing_stages = [i for i in range(KEPT_QUANTIZER_COUNT) if i not in sink["rvq"]]
    if missing_stages:
        raise SystemExit(
            f"{case.id}: RVQ stages {missing_stages} never fired -- the quantizer's "
            "structure changed"
        )

    # The published padding arithmetic, run against upstream's own functions at
    # this case's real input lengths. Before any write: a padding convention
    # that does not reproduce upstream is the single most damaging thing this
    # file could publish.
    padding_check = check_padding_arithmetic(sink["conv_lengths"])

    codes = to_numpy(encoded.audio_codes[0])
    if codes.ndim != 2 or codes.shape[1] != KEPT_QUANTIZER_COUNT:
        raise SystemExit(
            f"{case.id}: expected [frames, {KEPT_QUANTIZER_COUNT}] codes, got {codes.shape}"
        )
    frames_kept = int(codes.shape[0])

    # The concatenation order, checked rather than read, in three parts that
    # each have content:
    #
    #  (a) the semantic branch's width, which is what makes position 0 semantic
    #      and positions 1.. acoustic. Read off the module, not hardcoded.
    #  (b) the firing order, which is a real property of upstream's encode
    #      (semantic branch first, each branch walking its layers in order) and
    #      is recorded independently of the (branch, stage) identities.
    #  (c) the load-bearing one: each column of upstream's own codes equals the
    #      argmin this script probed for the stage it claims that column is.
    #
    # An earlier revision also compared each probe's (branch, stage) against the
    # rule that had assigned its position -- which could not fail under any
    # input. That tautology is gone; these three can each trip.
    semantic_count = int(mimi.quantizer.num_semantic_quantizers)
    if semantic_count != 1:
        raise SystemExit(
            f"{case.id}: num_semantic_quantizers is {semantic_count}, not 1. The "
            "position mapping this script publishes (column 0 semantic, columns 1.. "
            "acoustic) assumes 1; re-derive it before dumping."
        )
    expected_order = (
        [["semantic", stage] for stage in range(semantic_count)]
        + [["acoustic", stage] for stage in range(KEPT_QUANTIZER_COUNT - semantic_count)]
    )
    if sink["rvq_call_order"][:KEPT_QUANTIZER_COUNT] != expected_order:
        raise SystemExit(
            f"{case.id}: the quantizer fired in the order "
            f"{sink['rvq_call_order'][:KEPT_QUANTIZER_COUNT]}, not {expected_order} -- "
            "the split RVQ's branch order or per-branch stage order changed"
        )
    for position in range(KEPT_QUANTIZER_COUNT):
        probe = sink["rvq"][position]
        if not np.array_equal(probe["indices"][:frames_kept], codes[:, position]):
            raise SystemExit(
                f"{case.id}: code column {position} does not match the probed argmin of "
                f"{probe['branch']} stage {probe['stage']} -- the concatenated stage "
                "order is not what this script recorded"
            )

    # Byte-identity of latents and downsample. Upstream hands the downsample
    # output straight to the quantizer (modeling_mimi.py:1467-1469), and
    # conventions.json states that identity as fact -- so it is asserted here,
    # on the exact bytes that are about to be written, and the result is
    # published in conventions.json rather than only in result.json. If upstream
    # ever inserts a step between the two, this fails instead of quietly
    # contradicting the contract.
    latents_bytes = np.ascontiguousarray(sink["latents"][0], dtype=np.float32).tobytes()
    downsample_bytes = np.ascontiguousarray(sink["downsample"][0], dtype=np.float32).tobytes()
    latents_equals_downsample = latents_bytes == downsample_bytes
    if not latents_equals_downsample:
        raise SystemExit(
            f"{case.id}: the quantizer's input is no longer byte-identical to the "
            "downsample output. Something now sits between them, and "
            "conventions.json's artifact_layout.latents_note -- which states the "
            "identity as established fact -- is stale. Fix the contract before dumping."
        )

    # Frame geometry, and the trim arithmetic checked rather than merely
    # recorded: the ceiling divide must land on the frame count upstream's own
    # slice actually produced.
    frames_untrimmed = int(sink["downsample"][0].shape[-1])
    samples = int(sink["waveform"].reshape(-1).shape[0])
    trim_divisor = int(speech_tokenizer.model.encode_downsample_rate)
    trim_target = -(-samples // trim_divisor)
    if trim_target != frames_kept:
        raise SystemExit(
            f"{case.id}: the published trim arithmetic gives ceil({samples} / "
            f"{trim_divisor}) = {trim_target} frames, upstream produced {frames_kept}. "
            "Refusing to publish a frame rule that does not describe the codes."
        )

    # The strongest check available, and it runs BEFORE anything reaches disk:
    # the Base oracle already dumped this exact tensor for these cases. If the
    # hooks changed the forward, or the clip is being loaded differently, this
    # is where it shows -- and every tolerance measured downstream would
    # otherwise inherit the difference. A failed run must leave nothing
    # consumable behind, because Tasks 3/4/5/12 read the .f32 files, not
    # result.json.
    existing = case_dir / "codes" / "reference.i32"
    codes_bytes = np.ascontiguousarray(codes, dtype=np.int32).tobytes()
    if existing.exists():
        if existing.read_bytes() != codes_bytes:
            raise SystemExit(
                f"{case.id}: codes.i32 would differ from the Base oracle's {existing} -- "
                "the taps changed the forward, or the reference clip is not being loaded "
                "the same way. Nothing was written. Refusing to publish a stage dump "
                "that does not reproduce the codes it is supposed to explain."
            )
        codes_verified = True
    elif allow_unverified:
        codes_verified = False
        print(f"    WARNING: {existing} is absent; this case is UNVERIFIED", flush=True)
    else:
        raise SystemExit(
            f"{case.id}: {existing} does not exist, so codes.i32 cannot be checked "
            "against the Base oracle -- the one check that would catch a tap changing "
            "the forward. Nothing was written. Run "
            "scripts/dump_reference_qwen3_tts_base.py for this case first, or pass "
            "--allow-unverified to dump anyway (the run is then recorded as "
            "status 'unverified', not 'ok', in both result.json and conventions.json)."
        )

    artifacts: dict[str, dict] = {}
    artifacts["waveform"] = write_f32(case_dir / "codec_encoder" / "waveform.f32",
                                      sink["waveform"].reshape(-1))
    for index in range(4):
        artifacts[f"seanet_stage{index}"] = write_f32(
            case_dir / "codec_encoder" / f"seanet_stage{index}.f32", sink[f"seanet_stage{index}"][0]
        )
    artifacts["seanet_tail"] = write_f32(case_dir / "codec_encoder" / "seanet_tail.f32",
                                         sink["seanet_tail"][0])
    for index in range(len(mimi.encoder_transformer.layers)):
        artifacts[f"transformer_l{index}"] = write_f32(
            case_dir / "codec_encoder" / f"transformer_l{index}.f32", sink[f"transformer_l{index}"][0]
        )
    artifacts["downsample"] = write_f32(case_dir / "codec_encoder" / "downsample.f32",
                                        sink["downsample"][0])
    artifacts["latents"] = write_f32(case_dir / "codec_encoder" / "latents.f32", sink["latents"][0])

    margins = np.empty((KEPT_QUANTIZER_COUNT, sink["rvq"][0]["d1"].shape[-1]), dtype=np.float32)
    margins_euclidean = np.empty_like(margins)
    relative = np.empty_like(margins)
    for position in range(KEPT_QUANTIZER_COUNT):
        probe = sink["rvq"][position]
        artifacts[f"rvq_residual_s{position:02d}"] = write_f32(
            case_dir / "codec_encoder" / f"rvq_residual_s{position:02d}.f32", probe["residual"]
        )
        d1 = probe["d1"].astype(np.float32).reshape(-1)
        d2 = probe["d2"].astype(np.float32).reshape(-1)
        # (d2 - d1) * (d2 + d1) rather than d2*d2 - d1*d1. Algebraically the
        # same; numerically much better conditioned. The margin is a gap between
        # two nearly equal distances -- exactly the case where subtracting two
        # squares of similar magnitude in float32 loses most of the significant
        # digits, and this artifact IS the measurement, so ~0.5% of avoidable
        # error in it is not acceptable rounding.
        margins[position] = (d2 - d1) * (d2 + d1)
        margins_euclidean[position] = d2 - d1
        # d1 is a distance to a 256-wide codebook row; it is zero only if the
        # residual sits exactly on a centroid, which does not happen in float32
        # here -- but guard rather than emit an inf into the contract.
        denominator = np.where(d1 > 0, d1 * d1, np.nan)
        relative[position] = margins[position] / denominator

    artifacts["rvq_distance_margin"] = write_f32(
        case_dir / "codec_encoder" / "rvq_distance_margin.f32", margins
    )
    artifacts["codes"] = write_i32(case_dir / "codec_encoder" / "codes.i32", codes)

    observed = dict(static)
    observed.update({
        "waveform_dtype": "float32 on disk; runtime " + observed["runtime_dtype"],
        "device": device,
        "rvq_call_order_summary": {
            "calls": len(sink["rvq_call_order"]),
            "first_16": sink["rvq_call_order"][:KEPT_QUANTIZER_COUNT],
            "note": (
                "one semantic call followed by 31 acoustic calls; only the first 16 "
                "positions survive Qwen's slice."
            ),
        },
        "min_distance_margin": float(np.nanmin(margins)),
        "min_distance_margin_euclidean": float(np.nanmin(margins_euclidean)),
        "min_relative_margin": float(np.nanmin(relative)),
        "exact_ties": int(np.count_nonzero(margins <= 0.0)),
        "per_stage_min_margin": [float(np.nanmin(margins[i])) for i in range(KEPT_QUANTIZER_COUNT)],
        "frames": {
            "samples_into_encoder": samples,
            "encoder_output_frames": frames_untrimmed,
            "trim_divisor": trim_divisor,
            "trim_target_frames": trim_target,
            "frames_after_trim": frames_kept,
            "trim_was_a_noop": frames_untrimmed == frames_kept,
            "note": (
                "samples / 1920, rounded UP. The stage artifacts above are UNTRIMMED -- "
                "they are what the graph produces before Qwen's slice -- while codes.i32 "
                "is post-trim. When trim_was_a_noop is true the two counts coincide and "
                "no artifact here is affected; when it is false, only codes.i32 is "
                "shorter than the stage tensors."
            ),
        },
        "latents_equals_downsample": latents_equals_downsample,
        "extra_padding_check": padding_check,
        "codes_verified_against_base_oracle": codes_verified,
        "base_oracle_reference_codes": str(existing) if codes_verified else None,
    })
    conventions = build_conventions(observed)
    write_json(case_dir / "codec_encoder" / "conventions.json", conventions)
    artifacts["conventions"] = {"path": "conventions.json"}

    result = {
        "case": case.id,
        # Not an unconditional literal: a run that could not check its codes
        # against the Base oracle is not an "ok" run, and says so where any
        # consumer looks first.
        "status": "ok" if codes_verified else "unverified",
        "samples": samples,
        "encoder_output_frames": frames_untrimmed,
        "frames": frames_kept,
        "quantizers": KEPT_QUANTIZER_COUNT,
        "min_distance_margin": observed["min_distance_margin"],
        "exact_ties": observed["exact_ties"],
        "latents_equals_downsample": latents_equals_downsample,
        "matches_base_oracle_reference_codes": codes_verified,
        "reference_audio": ref_info,
        "wall_seconds": round(wall_seconds, 3),
        "artifacts": artifacts,
    }
    write_json(case_dir / "codec_encoder" / "result.json", result)
    artifacts["result"] = {"path": "result.json"}
    return {"id": case.id, "result": result}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--weights-dir", required=True, type=pathlib.Path)
    parser.add_argument(
        "--device", default="cuda:0",
        help="device_map forwarded to Qwen3TTSModel.from_pretrained (default: cuda:0). "
             "See docs/port-validation.md, 'Choosing the Oracle's dtype and Device'.",
    )

    # Explicit-argument form.
    parser.add_argument("--ref-audio", default=None,
                        help="Reference clip: local wav path, URL, or base64 string.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=None,
                        help="Case output directory for the explicit-argument form.")
    parser.add_argument(
        "--trim-seconds", type=float, default=None,
        help="Make the reference clip exactly this many seconds before encoding it, by "
             "truncating it or, if shorter, looping it.",
    )

    # Manifest form.
    parser.add_argument("--manifest", type=pathlib.Path, default=None,
                        help="Golden Manifest to read cases from. Never opened unless passed.")
    parser.add_argument("--case", action="append", default=None,
                        help="--manifest form only: restrict to this case id (repeatable).")
    parser.add_argument("--output-root", type=pathlib.Path, default=None,
                        help="--manifest form only: root directory for case subdirectories; "
                             "defaults to the manifest's case_artifact_root.")
    parser.add_argument(
        "--reference-audio-dir",
        type=pathlib.Path,
        default=pathlib.Path("models/qwen3-tts-reference-audio"),
        help="Where an http(s) input.reference.artifact locator is fetched and cached "
             "(git-ignored). Unused for a local-path locator.",
    )

    parser.add_argument(
        "--allow-unverified", action="store_true",
        help="Dump even when the case has no codes/reference.i32 from "
             "dump_reference_qwen3_tts_base.py to check codes.i32 against. Without it, a "
             "missing reference is a hard failure -- a skipped check must not read as a "
             "passing run. With it, the case is recorded as status 'unverified' in "
             "result.json and codes_verified_against_base_oracle false in "
             "conventions.json, so a consumer can tell the two apart.",
    )
    parser.add_argument("--report", type=pathlib.Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.manifest is not None:
        run_specs = load_cases_from_manifest(args)
    else:
        run_specs = [build_case_from_args(args)]

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"--device {args.device!r} requests CUDA but no CUDA device is available")

    from qwen_tts import Qwen3TTSModel

    # The same reference configuration the sibling dump scripts use, and -- more
    # to the point -- the same one the Base oracle used to produce
    # codes/reference.i32, which this script's codes.i32 is checked against.
    # Note that attn_implementation does NOT reach the speech tokenizer; see
    # conventions.json, geometry.transformer.attn_implementation_note.
    model = Qwen3TTSModel.from_pretrained(
        str(args.weights_dir),
        device_map=args.device,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    mimi = model.model.speech_tokenizer.model.encoder
    static = observe_static(mimi, model.model.speech_tokenizer.model)
    nulled = static["decoder_halves_nulled"]
    if not all(nulled.values()):
        raise SystemExit(
            f"the codec encoder still carries decoder halves ({nulled}); this script "
            "assumes Qwen3TTSTokenizerV2Encoder nulled them"
        )
    print(f"reference: {args.device} {static['runtime_dtype']} "
          f"{static['attn_implementation']} (codec encoder)", flush=True)

    records = []
    for index, (case, case_dir) in enumerate(run_specs, 1):
        print(f"[{index}/{len(run_specs)}] {case.id}", flush=True)
        records.append(run_case(model, case, case_dir, args.reference_audio_dir, static,
                                args.device, args.allow_unverified))
        result = records[-1]["result"]
        print(f"    {result['frames']} frames, min margin {result['min_distance_margin']:.6g}, "
              f"ties {result['exact_ties']}, {result['status']}, "
              f"wall {result['wall_seconds']}s", flush=True)

    report = {
        "schema": "synthesize-oracle-dump-v1",
        "family": FAMILY,
        "variant": "qwen3-tts-12hz-0-6b-base",
        "stage": "codec_encoder",
        "reference": {
            "device": args.device,
            "dtype": static["runtime_dtype"],
            "attn_implementation": static["attn_implementation"],
        },
        "case_count": len(records),
        "cases": records,
    }
    if args.report:
        write_json(args.report, report)
    print(f"dumped {len(records)} case(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
