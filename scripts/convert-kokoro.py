#!/usr/bin/env python3
"""Convert one explicitly supported Kokoro checkpoint to source-F32 GGUF.

The converter is manifest-driven. The manifest selects the supported Model
Variant and pins every source artifact digest; callers do not control tensor
mappings, layouts, metadata, or quantization. Later F16 and mixed-quantization
packages are produced by the C++ quantizer from this artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
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
    project_relative,
    sha256_file,
    write_json_atomic,
)

ARCH_KEY = "kokoro"
FORMAT_VERSION = 1
ARCHITECTURE_VERSION = 1
PROFILE_NAME = "F32"
PROFILE_VERSION = 1
SUPPORTED_VARIANTS = ("kokoro-v1-0",)
SUPPORTED_SOURCE_REPOSITORY = "https://github.com/hexgrad/kokoro"
SUPPORTED_SOURCE_REVISION = "dfb907a02bba8152ca444717ca5d78747ccb4bec"

EXPECTED_CHECKPOINT_ENTRIES = 548
EXPECTED_MODEL_TENSORS = 457
EXPECTED_WEIGHT_NORM_PAIRS = 89
EXPECTED_SKIPPED = 2
EXPECTED_VOICES = 54
VOICE_ROWS = 510
STYLE_DIM = 128
SAMPLES_PER_FRAME = 600

# Harmonic-plus-noise source constants fixed by the pinned Generator.
SOURCE_SAMPLING_RATE = 24000
SOURCE_HARMONIC_NUM = 8
SOURCE_SINE_AMP = 0.1
SOURCE_NOISE_STD = 0.003
SOURCE_VOICED_THRESHOLD = 10
SOURCE_UPSAMPLE_SCALE = 300

INPUT_PHONEMES_UTF8 = 1 << 1
INPUT_TOKEN_IDS = 1 << 2
CAPABILITY_SPEAKING_RATE = 1 << 0
CAPABILITY_STOCHASTIC = 1 << 1
LANGUAGE_DEFAULT = 1 << 0
LANGUAGE_REGIONAL_FALLBACK = 1 << 1

CHECKPOINT_NAMESPACES = (
    "bert",
    "bert_encoder",
    "predictor",
    "decoder",
    "text_encoder",
)


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
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--voices-dir", required=True, type=Path)
    parser.add_argument("--outfile", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
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
    if manifest.get("variant") not in SUPPORTED_VARIANTS:
        raise ConverterError(
            "this converter supports only variants " + ", ".join(SUPPORTED_VARIANTS)
        )
    source = manifest.get("source", {})
    if source.get("repository") != SUPPORTED_SOURCE_REPOSITORY:
        raise ConverterError("unsupported Kokoro source repository")
    if source.get("revision") != SUPPORTED_SOURCE_REVISION:
        raise ConverterError("unsupported Kokoro source revision")
    if manifest.get("reference", {}).get("dtype") != "float32":
        raise ConverterError("source GGUF conversion requires reference dtype float32")

    package = manifest.get("package_contract")
    if not isinstance(package, dict):
        raise ConverterError("manifest package_contract must be an object")
    if package.get("input_kinds") != ["phonemes_utf8", "token_ids"]:
        raise ConverterError(
            "Kokoro package input_kinds must be ['phonemes_utf8', 'token_ids']"
        )
    if package.get("frontend") != {
        "provider": "synthesize.symbol_map",
        "contract_version": 1,
        "phoneme_mapping": "unicode_scalar",
        "padding_rule": "wrap_pad_token",
    }:
        raise ConverterError(
            "Kokoro package requires synthesize.symbol_map v1 with unicode_scalar "
            "mapping and the wrap_pad_token padding rule"
        )
    if package.get("language_tags") != ["en"]:
        raise ConverterError("initial kokoro language catalog must be ['en']")
    if not package.get("stochastic"):
        raise ConverterError("Kokoro synthesis consumes randomness")
    audio = package.get("native_audio", {})
    if audio.get("sample_rate_hz") != SOURCE_SAMPLING_RATE or audio.get("channels") != 1:
        raise ConverterError("Kokoro native audio must be 24000 Hz mono")
    voices = package.get("voices", {})
    if voices.get("mode") != "preset-catalog" or voices.get("default_id") is not None:
        raise ConverterError(
            "Kokoro exposes a preset catalog and invents no package default Voice"
        )
    if len(voices.get("preset_ids", [])) != EXPECTED_VOICES:
        raise ConverterError(f"expected {EXPECTED_VOICES} preset voices")
    return package


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    def require(key: str, expected: Any) -> Any:
        actual = config.get(key)
        if actual != expected:
            raise ConverterError(f"config {key}: expected {expected!r}, got {actual!r}")
        return actual

    require("n_token", 178)
    require("hidden_dim", 512)
    require("style_dim", STYLE_DIM)
    require("n_layer", 3)
    require("n_mels", 80)
    require("max_dur", 50)
    require("dim_in", 64)
    require("max_conv_dim", 512)
    require("text_encoder_kernel_size", 5)
    require("multispeaker", True)

    plbert = config.get("plbert")
    if not isinstance(plbert, dict):
        raise ConverterError("config plbert must be an object")
    for key, expected in (
        ("hidden_size", 768),
        ("num_attention_heads", 12),
        ("intermediate_size", 2048),
        ("max_position_embeddings", 512),
        ("num_hidden_layers", 12),
    ):
        if plbert.get(key) != expected:
            raise ConverterError(f"config plbert.{key}: expected {expected}")

    istftnet = config.get("istftnet")
    if not isinstance(istftnet, dict):
        raise ConverterError("config istftnet must be an object")
    for key, expected in (
        ("upsample_rates", [10, 6]),
        ("upsample_kernel_sizes", [20, 12]),
        ("gen_istft_n_fft", 20),
        ("gen_istft_hop_size", 5),
        ("resblock_kernel_sizes", [3, 7, 11]),
        ("resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]]),
        ("upsample_initial_channel", 512),
    ):
        if istftnet.get(key) != expected:
            raise ConverterError(f"config istftnet.{key}: expected {expected!r}")

    upsample_product = 1
    for rate in istftnet["upsample_rates"]:
        upsample_product *= rate
    derived = 2 * upsample_product * istftnet["gen_istft_hop_size"]
    if derived != SAMPLES_PER_FRAME:
        raise ConverterError(
            f"derived samples per duration step {derived} != {SAMPLES_PER_FRAME}"
        )
    return config


def build_symbol_table(config: dict[str, Any]) -> list[str]:
    vocab = config.get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        raise ConverterError("config vocab must be a non-empty object")
    n_token = config["n_token"]
    symbols = [""] * n_token
    for symbol, token_id in sorted(vocab.items(), key=lambda item: item[1]):
        if not isinstance(token_id, int) or not 0 <= token_id < n_token:
            raise ConverterError(f"vocab id for {symbol!r} outside [0, {n_token})")
        if len(symbol) != 1:
            raise ConverterError(f"vocab key {symbol!r} is not a single Unicode scalar")
        if token_id == 0:
            raise ConverterError("vocab must not remap the reserved pad id 0")
        if symbols[token_id]:
            raise ConverterError(f"vocab id {token_id} is claimed twice")
        symbols[token_id] = symbol
    mapped = sum(1 for value in symbols if value)
    if mapped != len(vocab):
        raise ConverterError("symbol table lost entries while densifying the vocabulary")
    return symbols


def load_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001 - torch raises many unrelated types
        raise ConverterError(f"cannot load checkpoint {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ConverterError("checkpoint must be a namespace dictionary")
    if tuple(raw.keys()) != CHECKPOINT_NAMESPACES:
        raise ConverterError(
            f"unexpected checkpoint namespaces {tuple(raw.keys())}; "
            f"expected {CHECKPOINT_NAMESPACES}"
        )

    flat: dict[str, torch.Tensor] = {}
    entries = 0
    parameters = 0
    for namespace, state in raw.items():
        if not isinstance(state, dict):
            raise ConverterError(f"checkpoint namespace {namespace} is not a dictionary")
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                raise ConverterError(f"{namespace}.{key} is not a tensor")
            entries += 1
            parameters += int(value.numel())
            local = key[len("module.") :] if key.startswith("module.") else key
            flat_name = f"{namespace}.{local}"
            if flat_name in flat:
                raise ConverterError(f"duplicate checkpoint tensor {flat_name}")
            flat[flat_name] = value
    if entries != EXPECTED_CHECKPOINT_ENTRIES:
        raise ConverterError(
            f"expected {EXPECTED_CHECKPOINT_ENTRIES} checkpoint entries, got {entries}"
        )
    info = {
        "namespaces": list(raw.keys()),
        "entries": entries,
        "source_parameter_count": parameters,
    }
    return flat, info


def normalize_weight_norm(state: dict[str, torch.Tensor]) -> list[LogicalTensor]:
    """Fuse PyTorch weight-normalization pairs with the upstream dim-0 formula."""
    g_bases = {name[: -len("weight_g")] for name in state if name.endswith("weight_g")}
    v_bases = {name[: -len("weight_v")] for name in state if name.endswith("weight_v")}
    if g_bases != v_bases:
        raise ConverterError(
            "unpaired weight_norm tensors: "
            f"missing_g={sorted(v_bases - g_bases)}, missing_v={sorted(g_bases - v_bases)}"
        )
    if len(g_bases) != EXPECTED_WEIGHT_NORM_PAIRS:
        raise ConverterError(
            f"expected {EXPECTED_WEIGHT_NORM_PAIRS} weight_norm pairs, got {len(g_bases)}"
        )

    logical: list[LogicalTensor] = []
    for name in sorted(state):
        if name.endswith(("weight_g", "weight_v")):
            continue
        if name.endswith("weight") and name[: -len("weight")] in g_bases:
            raise ConverterError(f"checkpoint contains both fused and weight_norm {name}")
        logical.append(LogicalTensor(name, (name,), state[name], "identity"))

    for base in sorted(g_bases):
        g_name, v_name = base + "weight_g", base + "weight_v"
        g, v = state[g_name], state[v_name]
        if g.ndim != v.ndim or g.shape[0] != v.shape[0]:
            raise ConverterError(f"incompatible weight_norm shapes for {base}weight")
        if any(dimension != 1 for dimension in g.shape[1:]):
            raise ConverterError(f"unsupported weight_norm dimension for {base}weight")
        logical.append(
            LogicalTensor(
                base + "weight",
                (g_name, v_name),
                torch._weight_norm(v, g, 0),
                "weight_norm_fusion",
            )
        )

    logical.sort(key=lambda item: item.name)
    names = [item.name for item in logical]
    if len(names) != len(set(names)):
        raise ConverterError("weight_norm normalization produced duplicate names")
    return logical


def skip_reason(name: str) -> str | None:
    if name.startswith("bert.pooler."):
        return "unused ALBERT pooler; CustomAlbert returns last_hidden_state"
    return None


def prepare_output_tensors(
    logical: list[LogicalTensor],
) -> tuple[list[OutputTensor], list[dict[str, Any]]]:
    outputs: list[OutputTensor] = []
    skipped: list[dict[str, Any]] = []
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
            continue
        try:
            array = f32_numpy(item.tensor)
        except (TypeError, ValueError) as error:
            raise ConverterError(f"{item.name}: {error}") from error
        outputs.append(
            OutputTensor(item.name, item.source_names, item.name, array, item.transform)
        )

    if len(skipped) != EXPECTED_SKIPPED:
        raise ConverterError(f"expected {EXPECTED_SKIPPED} skipped tensors, got {len(skipped)}")
    if len(outputs) != EXPECTED_MODEL_TENSORS:
        raise ConverterError(
            f"expected {EXPECTED_MODEL_TENSORS} emitted model tensors, got {len(outputs)}"
        )
    outputs.sort(key=lambda item: item.name)
    if len({item.name for item in outputs}) != len(outputs):
        raise ConverterError("canonical mapping produced duplicate GGUF tensor names")
    return outputs, skipped


def load_voicepacks(
    manifest: dict[str, Any], voices_dir: Path, preset_ids: list[str]
) -> tuple[list[OutputTensor], list[dict[str, Any]]]:
    """Load each Preset Voice pack and verify it against its manifest digest."""
    by_locator = {
        artifact["locator"].rsplit("/", 1)[-1]: artifact
        for artifact in manifest["source"]["artifacts"]
        if artifact["role"] == "frontend-resource"
    }
    if len(by_locator) != EXPECTED_VOICES:
        raise ConverterError(
            f"manifest must pin {EXPECTED_VOICES} voicepack artifacts, got {len(by_locator)}"
        )

    outputs: list[OutputTensor] = []
    records: list[dict[str, Any]] = []
    for voice_id in preset_ids:
        path = voices_dir / f"{voice_id}.pt"
        if not path.is_file():
            raise ConverterError(f"missing voicepack {path}")
        artifact = by_locator.get(f"{voice_id}.pt")
        if artifact is None:
            raise ConverterError(f"manifest does not pin voicepack {voice_id}")
        digest = require_hash(path, artifact["sha256"], f"voicepack {voice_id}")
        try:
            pack = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:  # noqa: BLE001
            raise ConverterError(f"cannot load voicepack {path}: {error}") from error
        if not isinstance(pack, torch.Tensor):
            raise ConverterError(f"voicepack {voice_id} is not a tensor")
        if tuple(pack.shape) != (VOICE_ROWS, 1, 2 * STYLE_DIM):
            raise ConverterError(
                f"voicepack {voice_id} shape {tuple(pack.shape)} != "
                f"({VOICE_ROWS}, 1, {2 * STYLE_DIM})"
            )
        array = f32_numpy(pack.squeeze(1))
        outputs.append(
            OutputTensor(
                f"voice.{voice_id}",
                (f"voices/{voice_id}.pt",),
                f"voices/{voice_id}.pt",
                array,
                "squeeze_singleton_batch",
            )
        )
        records.append({"id": voice_id, "sha256": digest, "shape": list(array.shape)})
    return outputs, records


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
    general_name = "Kokoro v1.0"
    add_general_identity(
        writer,
        name=general_name,
        basename=manifest["variant"],
        size_label=compute_size_label(checkpoint_info["source_parameter_count"]),
        languages=package["language_tags"],
        tags=["text-to-speech", "kokoro", "styletts2", "synthesize.cpp"],
        author="hexgrad",
        organization="hexgrad",
        source_url=manifest["source"]["repository"],
        description=(
            "Source-F32 synthesize.cpp conversion of the pinned Kokoro-82M "
            "StyleTTS 2 checkpoint with its 54 preset voicepacks."
        ),
        license_id="apache-2.0",
        license_name="Apache License 2.0",
        license_link="https://huggingface.co/hexgrad/Kokoro-82M",
    )
    writer.add_repo_url(manifest["source"]["repository"])

    writer.add_uint32("synthesize.format_version", FORMAT_VERSION)
    writer.add_string("synthesize.model_family", ARCH_KEY)
    writer.add_string("synthesize.model_variant", manifest["variant"])
    writer.add_string("synthesize.quantization.profile", PROFILE_NAME)
    writer.add_uint32("synthesize.quantization.profile_version", PROFILE_VERSION)
    writer.add_string("synthesize.source.repository", manifest["source"]["repository"])
    writer.add_string("synthesize.source.revision", manifest["source"]["revision"])
    writer.add_string("synthesize.source.checkpoint.sha256", checkpoint_sha256)
    writer.add_string("synthesize.source.config.sha256", config_sha256)
    writer.add_string("synthesize.source.checkpoint.license_status", "apache-2.0")
    writer.add_string("synthesize.converter", "scripts/convert-kokoro.py")

    writer.add_uint32(
        "synthesize.capabilities.input_flags", INPUT_PHONEMES_UTF8 | INPUT_TOKEN_IDS
    )
    writer.add_uint32(
        "synthesize.capabilities.flags", CAPABILITY_SPEAKING_RATE | CAPABILITY_STOCHASTIC
    )
    writer.add_uint64(
        "synthesize.capabilities.max_input_tokens", package["max_input_tokens"]
    )
    writer.add_uint64(
        "synthesize.capabilities.max_output_frames", package["max_output_frames"]
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
        "synthesize.audio.sample_rate_hz", package["native_audio"]["sample_rate_hz"]
    )
    writer.add_uint32("synthesize.audio.channels", package["native_audio"]["channels"])
    writer.add_string(
        "synthesize.audio.sample_format", package["native_audio"]["sample_format"]
    )

    writer.add_uint32("synthesize.language.count", 1)
    writer.add_string("synthesize.language.0.tag", "en")
    writer.add_uint32(
        "synthesize.language.0.flags", LANGUAGE_DEFAULT | LANGUAGE_REGIONAL_FALLBACK
    )

    voices = package["voices"]
    preset_ids = voices["preset_ids"]
    writer.add_string("synthesize.voice.mode", voices["mode"])
    writer.add_bool("synthesize.voice.has_package_default", False)
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
    writer.add_string("synthesize.frontend.padding_rule", frontend["padding_rule"])
    writer.add_uint32("synthesize.sidecar.count", 0)

    return {
        "general.architecture": ARCH_KEY,
        "general.name": general_name,
        "general.file_type": 0,
        "synthesize.format_version": FORMAT_VERSION,
        "synthesize.model_family": ARCH_KEY,
        "synthesize.model_variant": manifest["variant"],
        "synthesize.source.checkpoint.sha256": checkpoint_sha256,
        "synthesize.source.config.sha256": config_sha256,
        "synthesize.capabilities.input_flags": INPUT_PHONEMES_UTF8 | INPUT_TOKEN_IDS,
        "synthesize.capabilities.flags": (
            CAPABILITY_SPEAKING_RATE | CAPABILITY_STOCHASTIC
        ),
        "synthesize.capabilities.max_input_tokens": package["max_input_tokens"],
        "synthesize.capabilities.max_output_frames": package["max_output_frames"],
        "synthesize.audio.sample_rate_hz": package["native_audio"]["sample_rate_hz"],
        "synthesize.audio.channels": 1,
        "synthesize.audio.sample_format": "f32le",
        "synthesize.voice.preset_count": len(preset_ids),
        "synthesize.frontend.padding_rule": frontend["padding_rule"],
    }


def add_kokoro_metadata(
    writer: GGUFWriter, config: dict[str, Any], symbols: list[str]
) -> dict[str, Any]:
    istftnet = config["istftnet"]
    plbert = config["plbert"]

    writer.add_uint32("synthesize.kokoro.architecture_version", ARCHITECTURE_VERSION)
    writer.add_uint32("synthesize.kokoro.n_token", config["n_token"])
    writer.add_array("synthesize.kokoro.symbols", symbols)
    writer.add_uint32("synthesize.kokoro.symbols.mapped_count", sum(1 for s in symbols if s))
    writer.add_string("synthesize.kokoro.symbols.lookup", "unique_scalar")
    writer.add_uint32("synthesize.kokoro.symbols.pad_id", 0)
    writer.add_uint32("synthesize.kokoro.hidden_dim", config["hidden_dim"])
    writer.add_uint32("synthesize.kokoro.style_dim", config["style_dim"])
    writer.add_uint32("synthesize.kokoro.n_layer", config["n_layer"])
    writer.add_uint32("synthesize.kokoro.n_mels", config["n_mels"])
    writer.add_uint32("synthesize.kokoro.max_dur", config["max_dur"])
    writer.add_uint32("synthesize.kokoro.dim_in", config["dim_in"])
    writer.add_uint32("synthesize.kokoro.text_encoder_kernel_size", config["text_encoder_kernel_size"])
    writer.add_uint32("synthesize.kokoro.samples_per_frame", SAMPLES_PER_FRAME)

    writer.add_uint32("synthesize.kokoro.plbert.hidden_size", plbert["hidden_size"])
    writer.add_uint32("synthesize.kokoro.plbert.num_attention_heads", plbert["num_attention_heads"])
    writer.add_uint32("synthesize.kokoro.plbert.intermediate_size", plbert["intermediate_size"])
    writer.add_uint32("synthesize.kokoro.plbert.num_hidden_layers", plbert["num_hidden_layers"])
    writer.add_uint32(
        "synthesize.kokoro.plbert.max_position_embeddings", plbert["max_position_embeddings"]
    )
    writer.add_uint32("synthesize.kokoro.plbert.shared_layer_groups", 1)
    writer.add_float32("synthesize.kokoro.plbert.layer_norm_eps", 1e-12)

    writer.add_array("synthesize.kokoro.istftnet.upsample_rates", istftnet["upsample_rates"])
    writer.add_array(
        "synthesize.kokoro.istftnet.upsample_kernel_sizes", istftnet["upsample_kernel_sizes"]
    )
    writer.add_array(
        "synthesize.kokoro.istftnet.resblock_kernel_sizes", istftnet["resblock_kernel_sizes"]
    )
    for index, dilations in enumerate(istftnet["resblock_dilation_sizes"]):
        writer.add_array(f"synthesize.kokoro.istftnet.resblock_dilations.{index}", dilations)
    writer.add_uint32(
        "synthesize.kokoro.istftnet.upsample_initial_channel",
        istftnet["upsample_initial_channel"],
    )
    writer.add_uint32("synthesize.kokoro.istftnet.gen_istft_n_fft", istftnet["gen_istft_n_fft"])
    writer.add_uint32(
        "synthesize.kokoro.istftnet.gen_istft_hop_size", istftnet["gen_istft_hop_size"]
    )
    writer.add_string("synthesize.kokoro.istftnet.window", "hann_periodic")
    writer.add_bool("synthesize.kokoro.istftnet.center", True)

    writer.add_uint32("synthesize.kokoro.source.sampling_rate", SOURCE_SAMPLING_RATE)
    writer.add_uint32("synthesize.kokoro.source.harmonic_num", SOURCE_HARMONIC_NUM)
    writer.add_float32("synthesize.kokoro.source.sine_amp", SOURCE_SINE_AMP)
    writer.add_float32("synthesize.kokoro.source.noise_std", SOURCE_NOISE_STD)
    writer.add_float32("synthesize.kokoro.source.voiced_threshold", float(SOURCE_VOICED_THRESHOLD))
    writer.add_uint32("synthesize.kokoro.source.upsample_scale", SOURCE_UPSAMPLE_SCALE)

    writer.add_uint32("synthesize.kokoro.voice.rows", VOICE_ROWS)
    writer.add_uint32("synthesize.kokoro.voice.dim", 2 * STYLE_DIM)
    writer.add_string("synthesize.kokoro.voice.row_rule", "final_token_count_minus_three")
    writer.add_uint32("synthesize.kokoro.voice.decoder_offset", 0)
    writer.add_uint32("synthesize.kokoro.voice.prosody_offset", STYLE_DIM)
    writer.add_bool("synthesize.kokoro.adain.instance_norm_affine", False)
    writer.add_float32("synthesize.kokoro.adain.eps", 1e-5)

    return {
        "synthesize.kokoro.architecture_version": ARCHITECTURE_VERSION,
        "synthesize.kokoro.n_token": config["n_token"],
        "synthesize.kokoro.style_dim": config["style_dim"],
        "synthesize.kokoro.samples_per_frame": SAMPLES_PER_FRAME,
        "synthesize.kokoro.voice.rows": VOICE_ROWS,
        "synthesize.kokoro.voice.dim": 2 * STYLE_DIM,
        "synthesize.kokoro.source.harmonic_num": SOURCE_HARMONIC_NUM,
        "synthesize.kokoro.istftnet.gen_istft_n_fft": istftnet["gen_istft_n_fft"],
        "synthesize.kokoro.istftnet.gen_istft_hop_size": istftnet["gen_istft_hop_size"],
    }


def write_gguf(
    path: Path,
    manifest: dict[str, Any],
    package: dict[str, Any],
    config: dict[str, Any],
    symbols: list[str],
    outputs: list[OutputTensor],
    checkpoint_sha256: str,
    config_sha256: str,
    checkpoint_info: dict[str, Any],
) -> dict[str, Any]:
    writer = GGUFWriter(str(path), ARCH_KEY)
    expected = add_package_metadata(
        writer, manifest, package, checkpoint_sha256, config_sha256, checkpoint_info
    )
    expected.update(add_kokoro_metadata(writer, config, symbols))
    for output in outputs:
        writer.add_tensor(output.name, output.array, raw_dtype=GGMLQuantizationType.F32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return expected


def verify_gguf(
    path: Path, outputs: list[OutputTensor], expected_fields: dict[str, Any]
) -> None:
    reader = GGUFReader(path)
    try:
        for key, expected in expected_fields.items():
            field = reader.fields.get(key)
            if field is None:
                raise ConverterError(f"written GGUF is missing metadata {key}")
            actual = field.contents()
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
        if set(by_name) != {output.name for output in outputs}:
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
            if tensor.data.tobytes(order="C") != output.array.tobytes(order="C"):
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
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    voices_dir = args.voices_dir.resolve()
    outfile = args.outfile.resolve()

    if outfile.suffix.lower() != ".gguf":
        raise ConverterError("--outfile must end in .gguf")
    for label, path in (
        ("manifest", manifest_path),
        ("config", config_path),
        ("checkpoint", checkpoint_path),
    ):
        if not path.is_file():
            raise ConverterError(f"{label} does not exist: {path}")
    if not voices_dir.is_dir():
        raise ConverterError(f"voices directory does not exist: {voices_dir}")
    if outfile in (manifest_path, config_path, checkpoint_path):
        raise ConverterError("--outfile must not overwrite an input file")

    manifest = load_json(manifest_path)
    package = validate_manifest(manifest)

    checkpoint_sha256 = require_hash(
        checkpoint_path, source_artifact(manifest, "checkpoint")["sha256"], "checkpoint"
    )
    config_sha256 = require_hash(
        config_path, source_artifact(manifest, "config")["sha256"], "config"
    )

    config = validate_config(load_json(config_path))
    symbols = build_symbol_table(config)

    state, checkpoint_info = load_checkpoint(checkpoint_path)
    logical = normalize_weight_norm(state)
    model_outputs, skipped = prepare_output_tensors(logical)
    voice_outputs, voice_records = load_voicepacks(
        manifest, voices_dir, package["voices"]["preset_ids"]
    )

    outputs = sorted(model_outputs + voice_outputs, key=lambda item: item.name)
    accounted = sum(len(item.source_names) for item in logical)
    if accounted != checkpoint_info["entries"]:
        raise ConverterError(
            f"accounted for {accounted} of {checkpoint_info['entries']} checkpoint entries"
        )

    with atomic_output_path(outfile) as temporary:
        expected_fields = write_gguf(
            temporary,
            manifest,
            package,
            config,
            symbols,
            outputs,
            checkpoint_sha256,
            config_sha256,
            checkpoint_info,
        )
        verify_gguf(temporary, outputs, expected_fields)

    output_sha256 = sha256_file(outfile)
    report = {
        "schema": "synthesize-converter-report-v1",
        "family": ARCH_KEY,
        "variant": manifest["variant"],
        "profile": PROFILE_NAME,
        "profile_version": PROFILE_VERSION,
        "converter": "scripts/convert-kokoro.py",
        "source": {
            "repository": manifest["source"]["repository"],
            "revision": manifest["source"]["revision"],
            "checkpoint": {
                "path": project_relative(checkpoint_path, project_root),
                "sha256": checkpoint_sha256,
                "entries": checkpoint_info["entries"],
                "parameter_count": checkpoint_info["source_parameter_count"],
            },
            "config": {
                "path": project_relative(config_path, project_root),
                "sha256": config_sha256,
            },
            "voicepacks": voice_records,
        },
        "output": {
            "path": project_relative(outfile, project_root),
            "sha256": output_sha256,
            "bytes": outfile.stat().st_size,
            "tensor_count": len(outputs),
            "model_tensor_count": len(model_outputs),
            "voice_tensor_count": len(voice_outputs),
            "weight_norm_pairs_fused": EXPECTED_WEIGHT_NORM_PAIRS,
        },
        "skipped": skipped,
        "tensors": [tensor_report(output) for output in outputs],
    }
    report_path = args.report or (
        project_root / "reports/convert/kokoro" / f"{manifest['variant']}-F32.json"
    )
    write_json_atomic(report_path, report)

    print(
        f"wrote {project_relative(outfile, project_root)} "
        f"({outfile.stat().st_size:,} bytes, {len(outputs)} tensors)"
    )
    print(f"  sha256 {output_sha256}")
    print(f"  report {project_relative(report_path, project_root)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConverterError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
