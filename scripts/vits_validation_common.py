"""Shared manifest and speaker-selection rules for VITS Golden validators."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gguf import GGUFReader


SUPPORTED_VARIANTS = ("vits-ljspeech", "vits-vctk")
SUPPORTED_QUANTIZATION_PROFILES = ("F32", "F16", "Q8_MIXED")


def manifest_variant(manifest: dict[str, Any]) -> str:
    variant = manifest.get("variant")
    if manifest.get("family") != "vits" or variant not in SUPPORTED_VARIANTS:
        raise ValueError(
            "validator requires a supported VITS reference variant: "
            + ", ".join(SUPPORTED_VARIANTS)
        )
    return variant


def speaker_index_for_case(case: dict[str, Any], variant: str) -> int | None:
    if variant == "vits-ljspeech":
        return None
    if variant != "vits-vctk":
        raise ValueError(f"unsupported VITS variant: {variant}")

    voice = case.get("voice")
    case_id = case.get("id", "<unknown>")
    if not isinstance(voice, dict) or voice.get("kind") != "preset_voice":
        raise ValueError(f"{case_id}: vits-vctk requires voice.kind=preset_voice")
    oracle_id = voice.get("oracle_id")
    try:
        speaker_index = int(oracle_id)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{case_id}: voice.oracle_id must be an integer") from error
    if str(speaker_index) != str(oracle_id) or not 0 <= speaker_index <= 0xFFFFFFFF:
        raise ValueError(f"{case_id}: voice.oracle_id must be a non-negative decimal integer")
    return speaker_index


def with_speaker_argument(
    command: list[str], case: dict[str, Any], variant: str
) -> list[str]:
    result = list(command)
    speaker_index = speaker_index_for_case(case, variant)
    if speaker_index is not None:
        result.append(str(speaker_index))
    return result


def model_quantization_identity(model_path: Path) -> tuple[str, int]:
    reader = GGUFReader(model_path)
    profile_field = reader.fields.get("synthesize.quantization.profile")
    version_field = reader.fields.get("synthesize.quantization.profile_version")
    if profile_field is None or version_field is None:
        raise ValueError("model is missing quantization profile metadata")
    profile = profile_field.contents()
    version = version_field.contents()
    if profile not in SUPPORTED_QUANTIZATION_PROFILES:
        raise ValueError(f"unsupported quantization profile: {profile}")
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        raise ValueError("quantization profile version must be a positive integer")
    return profile, version


def validation_report_name(
    variant: str, slice_name: str, backend: str, profile: str
) -> str:
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"unsupported VITS variant: {variant}")
    if profile not in SUPPORTED_QUANTIZATION_PROFILES:
        raise ValueError(f"unsupported quantization profile: {profile}")
    if backend not in ("cpu", "cuda"):
        raise ValueError(f"unsupported validation backend: {backend}")
    profile_component = "" if profile == "F32" else f"-{profile}"
    backend_component = "" if backend == "cpu" else f"-{backend}"
    return f"{variant}{profile_component}-{slice_name}{backend_component}.json"
