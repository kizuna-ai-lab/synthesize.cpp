#!/usr/bin/env python3
"""Convert one explicitly supported VITS checkpoint to source-F32 GGUF."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any

from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.gguf_common import (  # noqa: E402
    add_general_identity,
    atomic_output_path,
    f32_numpy,
    git_revision,
    project_relative,
    sha256_file,
    write_json_atomic,
)


ARCH_KEY = "vits"
FORMAT_VERSION = 1
ARCHITECTURE_VERSION = 1
PROFILE_NAME = "F32"
PROFILE_VERSION = 1
SUPPORTED_VARIANTS = ("vits-ljspeech", "vits-vctk")
SUPPORTED_SOURCE_REVISION = "2e561ba58618d021b5b8323d3765880f7e0ecfdb"
EXPECTED_EMITTED_TENSORS = 460

INPUT_PHONEMES_UTF8 = 1 << 1
INPUT_TOKEN_IDS = 1 << 2
CAPABILITY_SPEAKING_RATE = 1 << 0
CAPABILITY_STOCHASTIC = 1 << 1
LANGUAGE_DEFAULT = 1 << 0
LANGUAGE_REGIONAL_FALLBACK = 1 << 1


class ConverterError(RuntimeError):
    """A source, manifest, tensor-map, or GGUF contract violation."""


@dataclass(frozen=True)
class LogicalTensor:
    name: str
    source_names: tuple[str, ...]
    tensor: torch.Tensor
    transform: str


@dataclass(frozen=True)
class OutputTensor:
    name: str
    source_names: tuple[str, ...]
    source_name: str
    array: np.ndarray
    transform: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--outfile", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConverterError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ConverterError(f"{path} must contain a JSON object")
    return value


def source_artifact(manifest: dict[str, Any], role: str) -> dict[str, Any]:
    matches = [
        artifact
        for artifact in manifest.get("source", {}).get("artifacts", [])
        if artifact.get("role") == role
    ]
    if len(matches) != 1:
        raise ConverterError(
            f"manifest must contain exactly one source artifact with role {role!r}"
        )
    return matches[0]


def require_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ConverterError(
            f"{label} SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )
    return actual


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema") != "synthesize-golden-manifest-v1":
        raise ConverterError("unsupported Golden Manifest schema")
    if manifest.get("family") != ARCH_KEY:
        raise ConverterError(f"this converter requires family={ARCH_KEY}")
    variant = manifest.get("variant")
    if variant not in SUPPORTED_VARIANTS:
        raise ConverterError(
            "this converter supports only variants " + ", ".join(SUPPORTED_VARIANTS)
        )
    source = manifest.get("source", {})
    if source.get("repository") != "https://github.com/jaywalnut310/vits":
        raise ConverterError("unsupported VITS source repository")
    if source.get("revision") != SUPPORTED_SOURCE_REVISION:
        raise ConverterError("unsupported VITS source revision")
    reference = manifest.get("reference", {})
    if reference.get("dtype") != "float32":
        raise ConverterError("source GGUF conversion requires reference dtype float32")

    package = manifest.get("package_contract")
    if not isinstance(package, dict):
        raise ConverterError("manifest package_contract must be an object")
    if package.get("input_kinds") != ["phonemes_utf8", "token_ids"]:
        raise ConverterError(
            "VITS package input_kinds must be ['phonemes_utf8', 'token_ids']"
        )
    if package.get("frontend") != {
        "provider": "synthesize.symbol_map",
        "contract_version": 1,
        "phoneme_mapping": "unicode_scalar",
    }:
        raise ConverterError("VITS package requires synthesize.symbol_map v1")
    if package.get("language_tags") != ["en"]:
        raise ConverterError("initial vits-ljspeech language catalog must be ['en']")
    voices = package.get("voices", {})
    if variant == "vits-ljspeech":
        if (
            voices.get("mode") != "fixed-default"
            or voices.get("default_id") is not None
            or voices.get("preset_ids") != []
        ):
            raise ConverterError(
                "vits-ljspeech requires one unnamed package-default Voice"
            )
    else:
        expected_ids = [f"speaker-{index:03d}" for index in range(109)]
        if (
            voices.get("mode") != "preset-catalog"
            or voices.get("default_id") is not None
            or voices.get("preset_ids") != expected_ids
        ):
            raise ConverterError(
                "vits-vctk requires the speaker-000 through speaker-108 preset catalog"
            )
    if package.get("stochastic") is not True:
        raise ConverterError("vits-ljspeech must declare stochastic synthesis")
    if package.get("speaking_rate_range") != [0.8, 1.25]:
        raise ConverterError("unsupported vits-ljspeech speaking-rate range")
    for name in ("max_input_tokens", "max_output_frames"):
        value = package.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConverterError(f"package_contract.{name} must be a positive integer")
    audio = package.get("native_audio", {})
    if audio != {
        "sample_rate_hz": 22050,
        "channels": 1,
        "sample_format": "f32le",
    }:
        raise ConverterError("unsupported vits-ljspeech native_audio contract")
    return package


def resolve_config_path(
    source_dir: Path, manifest: dict[str, Any]
) -> tuple[Path, str]:
    artifact = source_artifact(manifest, "config")
    repository_path = artifact.get("repository_path")
    if not isinstance(repository_path, str) or not repository_path:
        raise ConverterError("config source artifact requires repository_path")
    config_path = (source_dir / repository_path).resolve()
    if not config_path.is_relative_to(source_dir):
        raise ConverterError("config repository_path escapes source-dir")
    if not config_path.is_file():
        raise ConverterError(f"config does not exist: {config_path}")
    return config_path, require_hash(config_path, artifact["sha256"], "config")


def load_symbols(source_dir: Path) -> list[str]:
    symbols_path = source_dir / "text" / "symbols.py"
    if not symbols_path.is_file():
        raise ConverterError(f"VITS symbol source does not exist: {symbols_path}")
    spec = importlib.util.spec_from_file_location(
        "_synthesize_vits_symbols", symbols_path
    )
    if spec is None or spec.loader is None:
        raise ConverterError(f"cannot create import spec for {symbols_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise ConverterError(f"cannot load VITS symbols from {symbols_path}: {error}") from error
    symbols = getattr(module, "symbols", None)
    if (
        not isinstance(symbols, list)
        or not symbols
        or any(not isinstance(symbol, str) for symbol in symbols)
    ):
        raise ConverterError("pinned VITS symbols must be a non-empty string list")
    return symbols


def require_config_value(
    container: dict[str, Any], key: str, expected_type: type, label: str
) -> Any:
    value = container.get(key)
    if not isinstance(value, expected_type) or (
        expected_type is int and isinstance(value, bool)
    ):
        raise ConverterError(f"{label}.{key} must be {expected_type.__name__}")
    return value


def validate_config(
    config: dict[str, Any], symbols: list[str], package: dict[str, Any]
) -> dict[str, Any]:
    data = config.get("data")
    model = config.get("model")
    train = config.get("train")
    if not all(isinstance(value, dict) for value in (data, model, train)):
        raise ConverterError("VITS config requires data, model, and train objects")
    assert isinstance(data, dict) and isinstance(model, dict) and isinstance(train, dict)

    speaker_count = require_config_value(data, "n_speakers", int, "data")
    gin_channels = model.get("gin_channels", 0)
    if not isinstance(gin_channels, int) or isinstance(gin_channels, bool):
        raise ConverterError("model.gin_channels must be int")
    voices = package["voices"]
    if voices["mode"] == "fixed-default":
        if speaker_count != 0 or gin_channels != 0:
            raise ConverterError(
                "vits-ljspeech must be single-speaker without global conditioning"
            )
    elif (
        speaker_count != len(voices["preset_ids"])
        or speaker_count <= 0
        or gin_channels <= 0
    ):
        raise ConverterError(
            "vits-vctk speaker catalog, embedding count, and conditioning channels disagree"
        )
    if model.get("use_sdp", True) is not True:
        raise ConverterError("initial VITS converter requires use_sdp=true")
    if str(model.get("resblock")) != "1":
        raise ConverterError("initial VITS converter requires decoder resblock type 1")
    if data.get("sampling_rate") != package["native_audio"]["sample_rate_hz"]:
        raise ConverterError("VITS config sampling rate disagrees with package contract")

    integer_keys = (
        "inter_channels",
        "hidden_channels",
        "filter_channels",
        "n_heads",
        "n_layers",
        "kernel_size",
        "upsample_initial_channel",
    )
    values = {
        key: require_config_value(model, key, int, "model") for key in integer_keys
    }
    if values["hidden_channels"] % values["n_heads"] != 0:
        raise ConverterError("hidden_channels must be divisible by n_heads")

    array_keys = (
        "resblock_kernel_sizes",
        "resblock_dilation_sizes",
        "upsample_rates",
        "upsample_kernel_sizes",
    )
    arrays: dict[str, list[Any]] = {}
    for key in array_keys:
        value = model.get(key)
        if not isinstance(value, list) or not value:
            raise ConverterError(f"model.{key} must be a non-empty array")
        arrays[key] = value
    if len(arrays["upsample_rates"]) != len(arrays["upsample_kernel_sizes"]):
        raise ConverterError("upsample rate/kernel arrays must have equal lengths")
    if len(arrays["resblock_kernel_sizes"]) != len(
        arrays["resblock_dilation_sizes"]
    ):
        raise ConverterError("resblock kernel/dilation arrays must have equal lengths")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for key in ("upsample_rates", "upsample_kernel_sizes", "resblock_kernel_sizes")
        for value in arrays[key]
    ):
        raise ConverterError("VITS kernel and upsample arrays must contain positive ints")
    for dilations in arrays["resblock_dilation_sizes"]:
        if (
            not isinstance(dilations, list)
            or len(dilations) != 3
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                for value in dilations
            )
        ):
            raise ConverterError("resblock type 1 requires three positive dilations")

    hop_length = require_config_value(data, "hop_length", int, "data")
    upsample_product = int(np.prod(arrays["upsample_rates"], dtype=np.int64))
    if upsample_product != hop_length:
        raise ConverterError(
            f"upsample product {upsample_product} != data.hop_length {hop_length}"
        )
    segment_size = require_config_value(train, "segment_size", int, "train")
    if segment_size % hop_length != 0:
        raise ConverterError("train.segment_size must be divisible by hop_length")

    return {
        **values,
        **arrays,
        "vocab_size": len(symbols),
        "sample_rate_hz": int(data["sampling_rate"]),
        "hop_length": hop_length,
        "add_blank": bool(data.get("add_blank")),
        "speaker_count": speaker_count,
        "gin_channels": gin_channels,
        "text_attention_window": 4,
        "layer_norm_epsilon": 1e-5,
        "duration_dds_layers": 3,
        "duration_flow_count": 3,
        "duration_spline_bins": 10,
        "duration_spline_tail_bound": 5.0,
        "duration_spline_min_bin_width": 1e-3,
        "duration_spline_min_bin_height": 1e-3,
        "duration_spline_min_derivative": 1e-3,
        "flow_block_count": 4,
        "flow_kernel_size": 5,
        "flow_dilation_rate": 1,
        "flow_wn_layers": 4,
        "flow_mean_only": True,
        "decoder_leaky_relu_slope": 0.1,
    }


def load_checkpoint(
    checkpoint_path: Path, manifest: dict[str, Any]
) -> tuple[dict[str, torch.Tensor], dict[str, Any], str]:
    artifact = source_artifact(manifest, "checkpoint")
    checkpoint_sha256 = require_hash(
        checkpoint_path, artifact["sha256"], "checkpoint"
    )
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except Exception as error:
        raise ConverterError(f"cannot load checkpoint {checkpoint_path}: {error}") from error
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model"), dict
    ):
        raise ConverterError("checkpoint must contain a model state dictionary")
    state = checkpoint["model"]
    if any(not isinstance(name, str) for name in state):
        raise ConverterError("checkpoint model tensor names must be strings")
    if any(not isinstance(tensor, torch.Tensor) for tensor in state.values()):
        raise ConverterError("checkpoint model entries must all be tensors")
    info = {
        "iteration": checkpoint.get("iteration"),
        "learning_rate": checkpoint.get("learning_rate"),
        "source_tensor_count": len(state),
        "source_parameter_count": sum(tensor.numel() for tensor in state.values()),
    }
    return state, info, checkpoint_sha256


def normalize_weight_norm(
    state: dict[str, torch.Tensor],
) -> list[LogicalTensor]:
    g_bases = {
        name[: -len("weight_g")]
        for name in state
        if name.endswith("weight_g")
    }
    v_bases = {
        name[: -len("weight_v")]
        for name in state
        if name.endswith("weight_v")
    }
    if g_bases != v_bases:
        missing_g = sorted(v_bases - g_bases)
        missing_v = sorted(g_bases - v_bases)
        raise ConverterError(
            f"unpaired weight_norm tensors: missing_g={missing_g}, missing_v={missing_v}"
        )

    logical: list[LogicalTensor] = []
    for name in sorted(state):
        if name.endswith(("weight_g", "weight_v")):
            continue
        if name.endswith("weight") and name[: -len("weight")] in g_bases:
            raise ConverterError(f"checkpoint contains both fused and weight_norm {name}")
        logical.append(LogicalTensor(name, (name,), state[name], "identity"))

    for base in sorted(g_bases):
        g_name = base + "weight_g"
        v_name = base + "weight_v"
        g = state[g_name]
        v = state[v_name]
        if g.ndim != v.ndim or g.shape[0] != v.shape[0]:
            raise ConverterError(f"incompatible weight_norm shapes for {base}weight")
        if any(dimension != 1 for dimension in g.shape[1:]):
            raise ConverterError(f"unsupported weight_norm dimension for {base}weight")
        fused = torch._weight_norm(v, g, 0)
        logical.append(
            LogicalTensor(
                base + "weight",
                (g_name, v_name),
                fused,
                "weight_norm_fusion",
            )
        )

    logical.sort(key=lambda item: item.name)
    names = [item.name for item in logical]
    if len(names) != len(set(names)):
        raise ConverterError("weight_norm normalization produced duplicate names")
    return logical


def skip_reason(name: str) -> str | None:
    if name.startswith("enc_q."):
        return "training-only posterior encoder"
    if name.startswith(
        ("dp.post_pre.", "dp.post_proj.", "dp.post_convs.", "dp.post_flows.")
    ):
        return "training-only stochastic-duration posterior"
    if name.startswith("dp.flows.1."):
        return "unused first ConvFlow removed by upstream reverse inference"
    return None


def map_dds(name: str, source_prefix: str, destination_prefix: str) -> str | None:
    if not name.startswith(source_prefix + "."):
        return None
    suffix = name[len(source_prefix) + 1 :]
    patterns = (
        (r"convs_sep\.(\d+)\.(weight|bias)", "depthwise"),
        (r"convs_1x1\.(\d+)\.(weight|bias)", "pointwise"),
    )
    for pattern, module in patterns:
        match = re.fullmatch(pattern, suffix)
        if match:
            layer, parameter = match.groups()
            return f"{destination_prefix}.blocks.{layer}.{module}.{parameter}"
    match = re.fullmatch(r"norms_([12])\.(\d+)\.(gamma|beta)", suffix)
    if match:
        norm, layer, parameter = match.groups()
        module = "depthwise_norm" if norm == "1" else "pointwise_norm"
        mapped_parameter = "weight" if parameter == "gamma" else "bias"
        return (
            f"{destination_prefix}.blocks.{layer}.{module}.{mapped_parameter}"
        )
    return None


def map_text_encoder(name: str) -> str | None:
    if name == "enc_p.emb.weight":
        return "text_encoder.token_embedding.weight"
    match = re.fullmatch(
        r"enc_p\.encoder\.attn_layers\.(\d+)\.conv_([qkvo])\.(weight|bias)",
        name,
    )
    if match:
        layer, projection, parameter = match.groups()
        projection_name = {
            "q": "query",
            "k": "key",
            "v": "value",
            "o": "output",
        }[projection]
        return (
            f"text_encoder.blocks.{layer}.attention.{projection_name}.{parameter}"
        )
    match = re.fullmatch(
        r"enc_p\.encoder\.attn_layers\.(\d+)\.emb_rel_([kv])", name
    )
    if match:
        layer, kind = match.groups()
        kind_name = "relative_key" if kind == "k" else "relative_value"
        return f"text_encoder.blocks.{layer}.attention.{kind_name}.weight"
    match = re.fullmatch(
        r"enc_p\.encoder\.norm_layers_([12])\.(\d+)\.(gamma|beta)", name
    )
    if match:
        norm, layer, parameter = match.groups()
        module = "attention_norm" if norm == "1" else "ffn_norm"
        mapped_parameter = "weight" if parameter == "gamma" else "bias"
        return f"text_encoder.blocks.{layer}.{module}.{mapped_parameter}"
    match = re.fullmatch(
        r"enc_p\.encoder\.ffn_layers\.(\d+)\.conv_([12])\.(weight|bias)",
        name,
    )
    if match:
        layer, projection, parameter = match.groups()
        module = "input" if projection == "1" else "output"
        return f"text_encoder.blocks.{layer}.ffn.{module}.{parameter}"
    match = re.fullmatch(r"enc_p\.proj\.(weight|bias)", name)
    if match:
        return f"text_encoder.projection.{match.group(1)}"
    return None


def map_duration_predictor(name: str) -> str | None:
    match = re.fullmatch(r"dp\.cond\.(weight|bias)", name)
    if match:
        return f"duration_predictor.conditioning.{match.group(1)}"
    match = re.fullmatch(r"dp\.(pre|proj)\.(weight|bias)", name)
    if match:
        module, parameter = match.groups()
        module = "pre" if module == "pre" else "projection"
        return f"duration_predictor.{module}.{parameter}"
    mapped = map_dds(name, "dp.convs", "duration_predictor.dds")
    if mapped is not None:
        return mapped
    match = re.fullmatch(r"dp\.flows\.0\.(m|logs)", name)
    if match:
        parameter = "bias" if match.group(1) == "m" else "log_scale"
        return f"duration_predictor.affine.{parameter}"
    match = re.fullmatch(r"dp\.flows\.([357])\.(pre|proj)\.(weight|bias)", name)
    if match:
        source_flow, module, parameter = match.groups()
        flow = {"3": 0, "5": 1, "7": 2}[source_flow]
        module = "pre" if module == "pre" else "projection"
        return f"duration_predictor.flows.{flow}.{module}.{parameter}"
    match = re.match(r"dp\.flows\.([357])\.convs\.", name)
    if match:
        source_flow = match.group(1)
        flow = {"3": 0, "5": 1, "7": 2}[source_flow]
        return map_dds(
            name,
            f"dp.flows.{source_flow}.convs",
            f"duration_predictor.flows.{flow}.dds",
        )
    return None


def map_flow(name: str) -> str | None:
    match = re.fullmatch(
        r"flow\.flows\.([0246])\.enc\.cond_layer\.(weight|bias)", name
    )
    if match:
        source_flow, parameter = match.groups()
        return f"flow.blocks.{int(source_flow) // 2}.conditioning.{parameter}"
    match = re.fullmatch(
        r"flow\.flows\.([0246])\.(pre|post)\.(weight|bias)", name
    )
    if match:
        source_flow, module, parameter = match.groups()
        block = int(source_flow) // 2
        module = "pre" if module == "pre" else "projection"
        return f"flow.blocks.{block}.{module}.{parameter}"
    match = re.fullmatch(
        r"flow\.flows\.([0246])\.enc\.(in_layers|res_skip_layers)\."
        r"(\d+)\.(weight|bias)",
        name,
    )
    if match:
        source_flow, module, layer, parameter = match.groups()
        block = int(source_flow) // 2
        module = "input" if module == "in_layers" else "residual_skip"
        return f"flow.blocks.{block}.wn.layers.{layer}.{module}.{parameter}"
    return None


def map_decoder(name: str, resblock_branches: int) -> str | None:
    match = re.fullmatch(r"dec\.cond\.(weight|bias)", name)
    if match:
        return f"decoder.conditioning.{match.group(1)}"
    match = re.fullmatch(r"dec\.conv_(pre|post)\.(weight|bias)", name)
    if match:
        module, parameter = match.groups()
        return f"decoder.{module}.{parameter}"
    match = re.fullmatch(r"dec\.ups\.(\d+)\.(weight|bias)", name)
    if match:
        stage, parameter = match.groups()
        return f"decoder.upsample.{stage}.transpose_conv.{parameter}"
    match = re.fullmatch(
        r"dec\.resblocks\.(\d+)\.convs([12])\.(\d+)\.(weight|bias)", name
    )
    if match:
        flat_block, convolution, layer, parameter = match.groups()
        flat = int(flat_block)
        stage = flat // resblock_branches
        branch = flat % resblock_branches
        return (
            f"decoder.upsample.{stage}.resblocks.{branch}."
            f"conv{convolution}.{layer}.{parameter}"
        )
    return None


def map_runtime_tensor(name: str, hparams: dict[str, Any]) -> str:
    if name == "emb_g.weight":
        return "voice.embedding.weight"
    for mapper in (
        map_text_encoder,
        map_duration_predictor,
        map_flow,
    ):
        mapped = mapper(name)
        if mapped is not None:
            return mapped
    mapped = map_decoder(name, len(hparams["resblock_kernel_sizes"]))
    if mapped is not None:
        return mapped
    raise ConverterError(f"unmapped runtime tensor: {name}")


def prepare_output_tensors(
    logical: list[LogicalTensor], hparams: dict[str, Any]
) -> tuple[list[OutputTensor], list[dict[str, Any]]]:
    outputs: list[OutputTensor] = []
    skipped: list[dict[str, Any]] = []
    accounted_sources: set[str] = set()
    for item in logical:
        reason = skip_reason(item.name)
        if reason is not None:
            skipped.append(
                {
                    "logical_source_name": item.name,
                    "source_names": list(item.source_names),
                    "transform": item.transform,
                    "reason": reason,
                }
            )
            accounted_sources.update(item.source_names)
            continue
        destination = map_runtime_tensor(item.name, hparams)
        try:
            array = f32_numpy(item.tensor)
        except (TypeError, ValueError) as error:
            raise ConverterError(f"{item.name}: {error}") from error
        outputs.append(
            OutputTensor(
                destination,
                item.source_names,
                item.name,
                array,
                item.transform,
            )
        )
        accounted_sources.update(item.source_names)

    outputs.sort(key=lambda item: item.name)
    output_names = [item.name for item in outputs]
    if len(output_names) != len(set(output_names)):
        raise ConverterError("canonical mapping produced duplicate GGUF tensor names")
    expected_tensor_count = EXPECTED_EMITTED_TENSORS
    if hparams.get("speaker_count", 0) > 0:
        expected_tensor_count += 13
    if len(outputs) != expected_tensor_count:
        raise ConverterError(
            f"expected {expected_tensor_count} emitted tensors, got {len(outputs)}"
        )
    if not outputs or not accounted_sources:
        raise ConverterError("converter did not account for checkpoint tensors")
    return outputs, skipped


def compute_size_label(parameter_count: int) -> str:
    if parameter_count >= 1_000_000_000:
        return f"{parameter_count / 1_000_000_000:.1f}B"
    if parameter_count >= 1_000_000:
        return f"{parameter_count / 1_000_000:.0f}M"
    return f"{parameter_count / 1_000:.0f}K"


def add_package_metadata(
    writer: GGUFWriter,
    manifest: dict[str, Any],
    package: dict[str, Any],
    checkpoint_sha256: str,
    config_sha256: str,
    checkpoint_info: dict[str, Any],
) -> dict[str, Any]:
    variant = manifest["variant"]
    is_vctk = variant == "vits-vctk"
    general_name = "VITS VCTK" if is_vctk else "VITS LJSpeech"
    add_general_identity(
        writer,
        name=general_name,
        basename=variant,
        size_label=compute_size_label(checkpoint_info["source_parameter_count"]),
        languages=package["language_tags"],
        tags=["text-to-speech", "vits", "synthesize.cpp"],
        author="jaywalnut310",
        organization="jaywalnut310",
        source_url=manifest["source"]["repository"],
        description=(
            "Source-F32 synthesize.cpp conversion of the pinned VITS "
            + (
                "multi-speaker VCTK checkpoint."
                if is_vctk
                else "single-speaker LJSpeech checkpoint."
            )
        ),
        license_id="other",
        license_name="checkpoint redistribution terms not explicitly stated",
    )
    writer.add_repo_url(manifest["source"]["repository"])

    writer.add_uint32("synthesize.format_version", FORMAT_VERSION)
    writer.add_string("synthesize.model_family", ARCH_KEY)
    writer.add_string("synthesize.model_variant", manifest["variant"])
    writer.add_string("synthesize.quantization.profile", PROFILE_NAME)
    writer.add_uint32(
        "synthesize.quantization.profile_version", PROFILE_VERSION
    )
    writer.add_string(
        "synthesize.source.repository", manifest["source"]["repository"]
    )
    writer.add_string(
        "synthesize.source.revision", manifest["source"]["revision"]
    )
    writer.add_string(
        "synthesize.source.checkpoint.sha256", checkpoint_sha256
    )
    writer.add_string("synthesize.source.config.sha256", config_sha256)
    writer.add_string(
        "synthesize.source.checkpoint.license_status", "not-explicitly-stated"
    )
    writer.add_string("synthesize.converter", "scripts/convert-vits.py")

    writer.add_uint32(
        "synthesize.capabilities.input_flags",
        INPUT_PHONEMES_UTF8 | INPUT_TOKEN_IDS,
    )
    writer.add_uint32(
        "synthesize.capabilities.flags",
        CAPABILITY_SPEAKING_RATE | CAPABILITY_STOCHASTIC,
    )
    writer.add_uint64(
        "synthesize.capabilities.max_input_tokens",
        package["max_input_tokens"],
    )
    writer.add_uint64(
        "synthesize.capabilities.max_output_frames",
        package["max_output_frames"],
    )
    writer.add_float32(
        "synthesize.capabilities.min_speaking_rate",
        float(package["speaking_rate_range"][0]),
    )
    writer.add_float32(
        "synthesize.capabilities.max_speaking_rate",
        float(package["speaking_rate_range"][1]),
    )
    writer.add_uint32(
        "synthesize.audio.sample_rate_hz",
        package["native_audio"]["sample_rate_hz"],
    )
    writer.add_uint32(
        "synthesize.audio.channels", package["native_audio"]["channels"]
    )
    writer.add_string(
        "synthesize.audio.sample_format",
        package["native_audio"]["sample_format"],
    )

    writer.add_uint32("synthesize.language.count", 1)
    writer.add_string("synthesize.language.0.tag", "en")
    writer.add_uint32(
        "synthesize.language.0.flags",
        LANGUAGE_DEFAULT | LANGUAGE_REGIONAL_FALLBACK,
    )
    voices = package["voices"]
    has_package_default = voices["mode"] == "fixed-default"
    preset_ids = voices["preset_ids"]
    writer.add_string("synthesize.voice.mode", voices["mode"])
    writer.add_bool(
        "synthesize.voice.has_package_default", has_package_default
    )
    writer.add_uint32("synthesize.voice.preset_count", len(preset_ids))
    for index, voice_id in enumerate(preset_ids):
        writer.add_string(f"synthesize.voice.{index}.id", voice_id)
        writer.add_uint32(f"synthesize.voice.{index}.flags", 0)
    frontend = package["frontend"]
    writer.add_bool("synthesize.frontend.present", True)
    writer.add_string("synthesize.frontend.provider", frontend["provider"])
    writer.add_uint32(
        "synthesize.frontend.contract_version", frontend["contract_version"]
    )
    writer.add_string(
        "synthesize.frontend.phoneme_mapping", frontend["phoneme_mapping"]
    )
    writer.add_uint32("synthesize.sidecar.count", 0)

    expected = {
        "general.architecture": ARCH_KEY,
        "general.name": general_name,
        "general.file_type": 0,
        "synthesize.format_version": FORMAT_VERSION,
        "synthesize.model_family": ARCH_KEY,
        "synthesize.model_variant": manifest["variant"],
        "synthesize.source.checkpoint.sha256": checkpoint_sha256,
        "synthesize.source.config.sha256": config_sha256,
        "synthesize.capabilities.input_flags": (
            INPUT_PHONEMES_UTF8 | INPUT_TOKEN_IDS
        ),
        "synthesize.capabilities.flags": (
            CAPABILITY_SPEAKING_RATE | CAPABILITY_STOCHASTIC
        ),
        "synthesize.capabilities.max_input_tokens": package["max_input_tokens"],
        "synthesize.capabilities.max_output_frames": package["max_output_frames"],
        "synthesize.audio.sample_rate_hz": package["native_audio"]["sample_rate_hz"],
        "synthesize.audio.channels": 1,
        "synthesize.audio.sample_format": "f32le",
        "synthesize.language.count": 1,
        "synthesize.voice.mode": voices["mode"],
        "synthesize.voice.has_package_default": has_package_default,
        "synthesize.voice.preset_count": len(preset_ids),
        "synthesize.frontend.present": True,
        "synthesize.frontend.provider": frontend["provider"],
        "synthesize.frontend.contract_version": frontend["contract_version"],
        "synthesize.frontend.phoneme_mapping": frontend["phoneme_mapping"],
    }
    for index, voice_id in enumerate(preset_ids):
        expected[f"synthesize.voice.{index}.id"] = voice_id
        expected[f"synthesize.voice.{index}.flags"] = 0
    return expected


def add_vits_metadata(
    writer: GGUFWriter, hparams: dict[str, Any], symbols: list[str]
) -> dict[str, Any]:
    writer.add_uint32(
        "synthesize.vits.architecture_version", ARCHITECTURE_VERSION
    )
    writer.add_uint32("synthesize.vits.vocab_size", hparams["vocab_size"])
    writer.add_array("synthesize.vits.symbols", symbols)
    writer.add_string(
        "synthesize.vits.symbols.lookup_policy", "last_index_wins"
    )
    writer.add_uint32("synthesize.vits.symbols.blank_id", 0)
    writer.add_bool(
        "synthesize.vits.symbols.upstream_add_blank", hparams["add_blank"]
    )
    writer.add_uint32(
        "synthesize.vits.inter_channels", hparams["inter_channels"]
    )
    writer.add_uint32(
        "synthesize.vits.hidden_channels", hparams["hidden_channels"]
    )
    writer.add_uint32(
        "synthesize.vits.filter_channels", hparams["filter_channels"]
    )
    writer.add_uint32("synthesize.vits.hop_length", hparams["hop_length"])
    writer.add_uint32(
        "synthesize.vits.speaker_count", hparams["speaker_count"]
    )
    writer.add_uint32(
        "synthesize.vits.conditioning_channels", hparams["gin_channels"]
    )

    writer.add_uint32(
        "synthesize.vits.text.layer_count", hparams["n_layers"]
    )
    writer.add_uint32(
        "synthesize.vits.text.head_count", hparams["n_heads"]
    )
    writer.add_uint32(
        "synthesize.vits.text.head_channels",
        hparams["hidden_channels"] // hparams["n_heads"],
    )
    writer.add_uint32(
        "synthesize.vits.text.ffn_kernel_size", hparams["kernel_size"]
    )
    writer.add_uint32(
        "synthesize.vits.text.attention_window",
        hparams["text_attention_window"],
    )
    writer.add_float32(
        "synthesize.vits.text.layer_norm_epsilon",
        hparams["layer_norm_epsilon"],
    )
    writer.add_float32(
        "synthesize.vits.text.embedding_scale",
        float(np.sqrt(hparams["hidden_channels"])),
    )

    writer.add_string(
        "synthesize.vits.duration.predictor_type", "stochastic"
    )
    writer.add_uint32(
        "synthesize.vits.duration.dds_layer_count",
        hparams["duration_dds_layers"],
    )
    writer.add_uint32(
        "synthesize.vits.duration.flow_count",
        hparams["duration_flow_count"],
    )
    writer.add_uint32(
        "synthesize.vits.duration.spline_bin_count",
        hparams["duration_spline_bins"],
    )
    writer.add_float32(
        "synthesize.vits.duration.spline_tail_bound",
        hparams["duration_spline_tail_bound"],
    )
    writer.add_float32(
        "synthesize.vits.duration.spline_min_bin_width",
        hparams["duration_spline_min_bin_width"],
    )
    writer.add_float32(
        "synthesize.vits.duration.spline_min_bin_height",
        hparams["duration_spline_min_bin_height"],
    )
    writer.add_float32(
        "synthesize.vits.duration.spline_min_derivative",
        hparams["duration_spline_min_derivative"],
    )
    writer.add_float32("synthesize.vits.inference.noise_scale", 0.667)
    writer.add_float32("synthesize.vits.inference.noise_scale_w", 0.8)

    writer.add_uint32(
        "synthesize.vits.flow.block_count", hparams["flow_block_count"]
    )
    writer.add_uint32(
        "synthesize.vits.flow.kernel_size", hparams["flow_kernel_size"]
    )
    writer.add_uint32(
        "synthesize.vits.flow.dilation_rate", hparams["flow_dilation_rate"]
    )
    writer.add_uint32(
        "synthesize.vits.flow.wn_layer_count", hparams["flow_wn_layers"]
    )
    writer.add_bool(
        "synthesize.vits.flow.mean_only", hparams["flow_mean_only"]
    )

    writer.add_string("synthesize.vits.decoder.resblock_type", "1")
    writer.add_array(
        "synthesize.vits.decoder.resblock_kernel_sizes",
        hparams["resblock_kernel_sizes"],
    )
    for index, dilations in enumerate(hparams["resblock_dilation_sizes"]):
        writer.add_array(
            f"synthesize.vits.decoder.resblock_dilations.{index}", dilations
        )
    writer.add_array(
        "synthesize.vits.decoder.upsample_rates",
        hparams["upsample_rates"],
    )
    writer.add_array(
        "synthesize.vits.decoder.upsample_kernel_sizes",
        hparams["upsample_kernel_sizes"],
    )
    writer.add_uint32(
        "synthesize.vits.decoder.upsample_initial_channels",
        hparams["upsample_initial_channel"],
    )
    writer.add_float32(
        "synthesize.vits.decoder.leaky_relu_slope",
        hparams["decoder_leaky_relu_slope"],
    )
    writer.add_string("synthesize.vits.decoder.output_activation", "tanh")

    return {
        "synthesize.vits.architecture_version": ARCHITECTURE_VERSION,
        "synthesize.vits.vocab_size": hparams["vocab_size"],
        "synthesize.vits.symbols": symbols,
        "synthesize.vits.symbols.lookup_policy": "last_index_wins",
        "synthesize.vits.hop_length": hparams["hop_length"],
        "synthesize.vits.text.layer_count": hparams["n_layers"],
        "synthesize.vits.duration.flow_count": hparams["duration_flow_count"],
        "synthesize.vits.flow.block_count": hparams["flow_block_count"],
    }


def write_gguf(
    path: Path,
    outputs: list[OutputTensor],
    manifest: dict[str, Any],
    package: dict[str, Any],
    hparams: dict[str, Any],
    symbols: list[str],
    checkpoint_sha256: str,
    config_sha256: str,
    checkpoint_info: dict[str, Any],
) -> dict[str, Any]:
    writer = GGUFWriter(str(path), ARCH_KEY)
    expected_fields = add_package_metadata(
        writer,
        manifest,
        package,
        checkpoint_sha256,
        config_sha256,
        checkpoint_info,
    )
    expected_fields.update(add_vits_metadata(writer, hparams, symbols))
    for output in outputs:
        writer.add_tensor(
            output.name,
            output.array,
            raw_dtype=GGMLQuantizationType.F32,
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return expected_fields


def field_value(reader: GGUFReader, key: str) -> Any:
    field = reader.fields.get(key)
    if field is None:
        raise ConverterError(f"written GGUF is missing metadata {key}")
    return field.contents()


def verify_gguf(
    path: Path,
    outputs: list[OutputTensor],
    expected_fields: dict[str, Any],
) -> None:
    reader = GGUFReader(path)
    try:
        for key, expected in expected_fields.items():
            actual = field_value(reader, key)
            if isinstance(expected, float):
                if not np.isclose(actual, expected, rtol=0.0, atol=1e-6):
                    raise ConverterError(
                        f"written GGUF metadata {key}: expected {expected}, got {actual}"
                    )
            elif actual != expected:
                raise ConverterError(
                    f"written GGUF metadata {key}: expected {expected}, got {actual}"
                )

        by_name = {tensor.name: tensor for tensor in reader.tensors}
        expected_names = {output.name for output in outputs}
        if set(by_name) != expected_names:
            raise ConverterError("written GGUF tensor-name set differs from mapping")
        for output in outputs:
            tensor = by_name[output.name]
            if tensor.tensor_type != GGMLQuantizationType.F32:
                raise ConverterError(f"{output.name}: written type is not F32")
            expected_shape = tuple(reversed(output.array.shape))
            if tuple(int(value) for value in tensor.shape) != expected_shape:
                raise ConverterError(
                    f"{output.name}: expected GGML shape {expected_shape}, "
                    f"got {tuple(tensor.shape)}"
                )
            expected_payload = output.array.tobytes(order="C")
            actual_payload = tensor.data.tobytes(order="C")
            if actual_payload != expected_payload:
                raise ConverterError(f"{output.name}: written tensor bytes differ")
    finally:
        mmap = getattr(reader.data, "_mmap", None)
        if mmap is not None:
            mmap.close()


def tensor_report(output: OutputTensor) -> dict[str, Any]:
    payload = output.array.tobytes(order="C")
    return {
        "name": output.name,
        "source_name": output.source_name,
        "source_names": list(output.source_names),
        "transform": output.transform,
        "source_shape": list(output.array.shape),
        "ggml_shape": list(reversed(output.array.shape)),
        "dtype": "F32",
        "elements": int(output.array.size),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    manifest_path = args.manifest.resolve()
    source_dir = args.source_dir.resolve()
    checkpoint_path = args.checkpoint.resolve()
    outfile = args.outfile.resolve()
    if outfile.suffix.lower() != ".gguf":
        raise ConverterError("--outfile must end in .gguf")
    if not manifest_path.is_file():
        raise ConverterError(f"manifest does not exist: {manifest_path}")
    if not source_dir.is_dir():
        raise ConverterError(f"source directory does not exist: {source_dir}")
    if not checkpoint_path.is_file():
        raise ConverterError(f"checkpoint does not exist: {checkpoint_path}")
    if outfile in (manifest_path, checkpoint_path):
        raise ConverterError("--outfile must not overwrite an input file")

    manifest = load_json(manifest_path)
    package = validate_manifest(manifest)
    revision = git_revision(source_dir)
    if revision != manifest["source"]["revision"]:
        raise ConverterError(
            "source-dir must be the pinned VITS Git checkout at "
            f"{manifest['source']['revision']}; got {revision}"
        )
    config_path, config_sha256 = resolve_config_path(source_dir, manifest)
    if outfile == config_path:
        raise ConverterError("--outfile must not overwrite an input file")
    config = load_json(config_path)
    symbols = load_symbols(source_dir)
    hparams = validate_config(config, symbols, package)
    state, checkpoint_info, checkpoint_sha256 = load_checkpoint(
        checkpoint_path, manifest
    )
    if state.get("enc_p.emb.weight") is None:
        raise ConverterError("checkpoint is missing text embedding")
    if tuple(state["enc_p.emb.weight"].shape) != (
        hparams["vocab_size"],
        hparams["hidden_channels"],
    ):
        raise ConverterError("symbol table/config disagree with text embedding shape")

    logical = normalize_weight_norm(state)
    outputs, skipped = prepare_output_tensors(logical, hparams)
    source_names = {name for item in logical for name in item.source_names}
    if source_names != set(state):
        raise ConverterError("not every source tensor was accounted for")

    with atomic_output_path(outfile) as temporary:
        expected_fields = write_gguf(
            temporary,
            outputs,
            manifest,
            package,
            hparams,
            symbols,
            checkpoint_sha256,
            config_sha256,
            checkpoint_info,
        )
        verify_gguf(temporary, outputs, expected_fields)

    output_sha256 = sha256_file(outfile)
    report_path = (
        project_root
        / "reports"
        / "convert"
        / manifest["family"]
        / f"{manifest['variant']}-{PROFILE_NAME}.json"
    )
    converter_path = Path(__file__).resolve()
    helper_path = converter_path.parent / "lib" / "gguf_common.py"
    lock_path = project_root / manifest["reference"]["environment_lock"]
    emitted_source_count = sum(len(output.source_names) for output in outputs)
    skipped_source_count = sum(
        len(item["source_names"]) for item in skipped
    )
    if emitted_source_count + skipped_source_count != len(state):
        raise ConverterError("converter report source accounting is incomplete")
    report = {
        "schema": "synthesize-converter-report-v1",
        "family": manifest["family"],
        "variant": manifest["variant"],
        "profile": PROFILE_NAME,
        "profile_version": PROFILE_VERSION,
        "source": {
            "repository": manifest["source"]["repository"],
            "revision": manifest["source"]["revision"],
            "config_path": project_relative(config_path, project_root),
            "config_sha256": config_sha256,
            "checkpoint_path": project_relative(checkpoint_path, project_root),
            "checkpoint_sha256": checkpoint_sha256,
            **checkpoint_info,
        },
        "converter": {
            "path": project_relative(converter_path, project_root),
            "sha256": sha256_file(converter_path),
            "helper_path": project_relative(helper_path, project_root),
            "helper_sha256": sha256_file(helper_path),
            "environment_lock": project_relative(lock_path, project_root),
            "environment_lock_sha256": sha256_file(lock_path),
            "project_revision": git_revision(project_root),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "gguf": importlib_metadata.version("gguf"),
        },
        "output": {
            "path": project_relative(outfile, project_root),
            "bytes": outfile.stat().st_size,
            "sha256": output_sha256,
            "tensor_count": len(outputs),
            "logical_source_tensor_count": len(logical),
            "emitted_source_tensor_count": emitted_source_count,
            "skipped_logical_tensor_count": len(skipped),
            "skipped_source_tensor_count": skipped_source_count,
            "weight_norm_fusion_count": sum(
                output.transform == "weight_norm_fusion" for output in outputs
            ),
        },
        "tensors": [tensor_report(output) for output in outputs],
        "skipped": skipped,
    }
    write_json_atomic(report_path, report)
    print(
        json.dumps(
            {
                "status": "ok",
                "family": manifest["family"],
                "variant": manifest["variant"],
                "profile": PROFILE_NAME,
                "tensors": len(outputs),
                "bytes": outfile.stat().st_size,
                "sha256": output_sha256,
                "outfile": project_relative(outfile, project_root),
                "report": project_relative(report_path, project_root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ConverterError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
