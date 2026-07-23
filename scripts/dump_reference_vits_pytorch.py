#!/usr/bin/env python3
"""Materialize deterministic VITS oracle artifacts from a Golden Manifest.

This runner is intentionally a validation tool, not a public synthesis API. It
captures the stochastic tensors consumed by the pinned PyTorch graph so another
runtime can replay the graph without reproducing PyTorch's random generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import torch


UINT64_MAX = (1 << 64) - 1
REQUIRED_PROBES = {
    "input.token_ids",
    "random.duration_noise",
    "random.latent_noise",
    "text.m_p",
    "text.logs_p",
    "text.mask",
    "duration.logw",
    "duration.w_ceil",
    "duration.y_length",
    "duration.attention",
    "prior.m_p_expanded",
    "prior.logs_p_expanded",
    "latent.z_p",
    "flow.z",
    "audio.pcm",
    "result",
    "metadata",
}
ARTIFACT_FORMATS = {
    "input.token_ids": "i32le",
    "random.duration_noise": "f32le",
    "random.latent_noise": "f32le",
    "text.m_p": "f32le",
    "text.logs_p": "f32le",
    "text.mask": "f32le",
    "duration.logw": "f32le",
    "duration.w_ceil": "f32le",
    "duration.y_length": "i64le",
    "duration.attention": "f32le",
    "prior.m_p_expanded": "f32le",
    "prior.logs_p_expanded": "f32le",
    "latent.z_p": "f32le",
    "flow.z": "f32le",
    "voice.embedding": "f32le",
    "audio.pcm": "f32le",
    "result": "json",
    "metadata": "json",
}


class DumperError(RuntimeError):
    """A manifest, source, inference, or artifact contract violation."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump deterministic PyTorch VITS Golden Manifest artifacts."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        metavar="CASE_ID",
        help="Materialize only this case; repeat the option to select several.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Override the manifest's case_artifact_root.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="PyTorch CPU intra-op thread count (default: 1).",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DumperError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise DumperError(f"{path} must contain a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise DumperError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def require_sha256(path: Path, expected: str, role: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise DumperError(
            f"{role} SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )
    return actual


def source_artifact(
    manifest: dict[str, Any], role: str, *, require_repository_path: bool = False
) -> dict[str, Any]:
    matches = [
        artifact
        for artifact in manifest.get("source", {}).get("artifacts", [])
        if artifact.get("role") == role
    ]
    if len(matches) != 1:
        raise DumperError(
            f"manifest must contain exactly one source artifact with role {role!r}"
        )
    artifact = matches[0]
    if require_repository_path and "repository_path" not in artifact:
        raise DumperError(f"{role} source artifact requires repository_path")
    return artifact


def verify_source_revision(source_dir: Path, expected: str) -> None:
    if not (source_dir / ".git").exists():
        return
    try:
        result = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise DumperError(f"cannot resolve source revision in {source_dir}: {error}") from error
    actual = result.stdout.strip()
    if actual != expected:
        raise DumperError(f"source revision mismatch: expected {expected}, got {actual}")


def parse_seed(value: Any, case_id: str) -> int:
    if not isinstance(value, str) or not value.isdecimal():
        raise DumperError(f"{case_id}: seed_u64 must be a decimal string")
    seed = int(value, 10)
    if seed > UINT64_MAX:
        raise DumperError(f"{case_id}: seed_u64 exceeds UINT64_MAX")
    return seed


def case_artifact_specs(case: dict[str, Any]) -> dict[str, dict[str, str]]:
    candidates: list[dict[str, str]] = []
    input_artifact = case.get("input", {}).get("artifact")
    if input_artifact is not None:
        candidates.append(input_artifact)
    candidates.extend(case.get("oracle", {}).get("stochastic_inputs", []))
    candidates.extend(case.get("expected", {}).get("artifacts", []))

    specs: dict[str, dict[str, str]] = {}
    paths: dict[str, str] = {}
    for spec in candidates:
        name = spec.get("name")
        if not isinstance(name, str):
            raise DumperError(f"{case['id']}: artifact without a valid name")
        prior = specs.get(name)
        if prior is not None and prior != spec:
            raise DumperError(f"{case['id']}: conflicting declarations for artifact {name}")
        expected_format = ARTIFACT_FORMATS.get(name)
        if expected_format is not None and spec.get("format") != expected_format:
            raise DumperError(
                f"{case['id']}: {name} requires format {expected_format}, "
                f"got {spec.get('format')}"
            )
        path = spec.get("path")
        if not isinstance(path, str) or not path:
            raise DumperError(f"{case['id']}: {name} requires a non-empty path")
        prior_name = paths.get(path)
        if prior_name is not None and prior_name != name:
            raise DumperError(
                f"{case['id']}: artifacts {prior_name} and {name} share path {path}"
            )
        specs[name] = spec
        paths[path] = name
    return specs


def validate_manifest(
    manifest: dict[str, Any], selected_ids: list[str] | None
) -> list[dict[str, Any]]:
    if manifest.get("schema") != "synthesize-golden-manifest-v1":
        raise DumperError("unsupported manifest schema")
    if manifest.get("family") != "vits":
        raise DumperError("this runner only accepts the vits family")
    if manifest.get("reference", {}).get("device") != "cpu":
        raise DumperError("the VITS oracle runner requires reference.device=cpu")
    if manifest.get("reference", {}).get("dtype") != "float32":
        raise DumperError("the VITS oracle runner requires reference.dtype=float32")

    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise DumperError("manifest cases must be a non-empty array")
    by_id: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise DumperError("each case requires a non-empty id")
        if case_id in by_id:
            raise DumperError(f"duplicate case id {case_id}")
        by_id[case_id] = case

        request = case.get("request", {})
        parse_seed(request.get("seed_u64"), case_id)
        if request.get("max_output_frames") != 0:
            raise DumperError(
                f"{case_id}: oracle success cases require max_output_frames=0"
            )
        rate = request.get("speaking_rate")
        if (
            not isinstance(rate, (int, float))
            or isinstance(rate, bool)
            or rate <= 0
        ):
            raise DumperError(f"{case_id}: speaking_rate must be positive")
        if case.get("expected", {}).get("status") != "ok":
            raise DumperError(f"{case_id}: this runner only materializes status=ok cases")
        if case.get("input", {}).get("kind") != "token_ids":
            raise DumperError(f"{case_id}: initial VITS oracle cases require token_ids")

        specs = case_artifact_specs(case)
        missing = REQUIRED_PROBES - specs.keys()
        if missing:
            raise DumperError(
                f"{case_id}: missing required artifact declarations: "
                + ", ".join(sorted(missing))
            )

    if selected_ids is None:
        return cases
    if len(selected_ids) != len(set(selected_ids)):
        raise DumperError("--case contains duplicate case IDs")
    unknown = [case_id for case_id in selected_ids if case_id not in by_id]
    if unknown:
        raise DumperError("unknown case ID(s): " + ", ".join(unknown))
    selected = set(selected_ids)
    return [case for case in cases if case["id"] in selected]


def import_upstream(source_dir: Path) -> tuple[Any, Any, Any, Any, Any]:
    source_text = str(source_dir)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    try:
        import commons  # type: ignore[import-not-found]
        import models  # type: ignore[import-not-found]
        import utils  # type: ignore[import-not-found]
        from text import text_to_sequence  # type: ignore[import-not-found]
        from text.symbols import symbols  # type: ignore[import-not-found]
    except Exception as error:
        raise DumperError(
            "cannot import pinned VITS source; build its monotonic_align extension "
            f"inside {source_dir} first: {error}"
        ) from error
    return commons, models, utils, text_to_sequence, symbols


def configure_torch(threads: int) -> None:
    if threads < 1:
        raise DumperError("--threads must be at least 1")
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as error:
        raise DumperError(f"cannot set PyTorch inter-op thread count: {error}") from error
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.deterministic = True


def load_model(
    config_path: Path,
    checkpoint_path: Path,
    models: Any,
    utils: Any,
    symbols: list[str],
) -> tuple[Any, Any, dict[str, Any]]:
    hps = utils.get_hparams_from_file(str(config_path))
    model = models.SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **dict(hps.model.items()),
    )
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except Exception as error:
        raise DumperError(f"cannot load checkpoint {checkpoint_path}: {error}") from error
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model"), dict
    ):
        raise DumperError("checkpoint must contain a model state dictionary")
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
    except RuntimeError as error:
        raise DumperError(
            f"checkpoint does not strictly match the VITS model: {error}"
        ) from error
    model.cpu().float().eval()
    checkpoint_info = {
        "iteration": checkpoint.get("iteration"),
        "learning_rate": checkpoint.get("learning_rate"),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    return model, hps, checkpoint_info


def resolve_token_ids(
    case: dict[str, Any], hps: Any, text_to_sequence: Any, commons: Any
) -> list[int]:
    input_value = case["input"]
    inline = input_value.get("token_ids")
    if inline is not None:
        if not isinstance(inline, list) or not inline:
            raise DumperError(f"{case['id']}: token_ids must be a non-empty array")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in inline
        ):
            raise DumperError(
                f"{case['id']}: token_ids must contain non-negative integers"
            )
        return inline

    source_text = input_value.get("source_text")
    if not isinstance(source_text, str) or not source_text:
        raise DumperError(
            f"{case['id']}: token_ids cases need inline token_ids or source_text"
        )
    token_ids = list(text_to_sequence(source_text, hps.data.text_cleaners))
    if hps.data.add_blank:
        token_ids = commons.intersperse(token_ids, 0)
    if not token_ids:
        raise DumperError(f"{case['id']}: upstream frontend produced no tokens")
    return token_ids


def resolve_voice(case: dict[str, Any], model: Any) -> torch.Tensor | None:
    voice = case["voice"]
    if model.n_speakers == 0:
        if voice.get("kind") != "package_default":
            raise DumperError(
                f"{case['id']}: a single-speaker VITS model requires package_default"
            )
        return None
    if voice.get("kind") != "preset_voice":
        raise DumperError(
            f"{case['id']}: a multi-speaker VITS model requires preset_voice"
        )
    oracle_id = voice.get("oracle_id")
    try:
        speaker_id = int(oracle_id)
    except (TypeError, ValueError) as error:
        raise DumperError(f"{case['id']}: oracle_id must be an integer") from error
    if speaker_id < 0 or speaker_id >= model.n_speakers:
        raise DumperError(
            f"{case['id']}: oracle_id {speaker_id} is outside [0, {model.n_speakers})"
        )
    return torch.tensor([speaker_id], dtype=torch.long, device="cpu")


def require_number(parameters: dict[str, Any], name: str) -> float:
    value = parameters.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DumperError(f"oracle parameter {name} must be numeric")
    return float(value)


def captured_sdp_reverse(
    predictor: Any,
    encoded: torch.Tensor,
    text_mask: torch.Tensor,
    conditioning: torch.Tensor | None,
    duration_noise: torch.Tensor,
    noise_scale_w: float,
) -> torch.Tensor:
    hidden = torch.detach(encoded)
    hidden = predictor.pre(hidden)
    if conditioning is not None:
        hidden = hidden + predictor.cond(torch.detach(conditioning))
    hidden = predictor.convs(hidden, text_mask)
    hidden = predictor.proj(hidden) * text_mask

    flows = list(reversed(predictor.flows))
    flows = flows[:-2] + [flows[-1]]
    latent = duration_noise * noise_scale_w
    for flow in flows:
        latent = flow(latent, text_mask, g=hidden, reverse=True)
    logw, _ = torch.split(latent, [1, 1], 1)
    return logw


def infer_with_captured_randomness(
    model: Any,
    token_ids: list[int],
    speaker_id: torch.Tensor | None,
    seed: int,
    noise_scale: float,
    noise_scale_w: float,
    length_scale: float,
    commons: Any,
) -> dict[str, torch.Tensor]:
    tokens = torch.tensor([token_ids], dtype=torch.long, device="cpu")
    token_lengths = torch.tensor([len(token_ids)], dtype=torch.long, device="cpu")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    encoded, text_m_p, text_logs_p, text_mask = model.enc_p(tokens, token_lengths)
    if model.n_speakers > 0:
        if speaker_id is None:
            raise DumperError("speaker_id is required for a multi-speaker model")
        conditioning = model.emb_g(speaker_id).unsqueeze(-1)
    else:
        conditioning = None

    if not model.use_sdp:
        raise DumperError("the initial VITS validation contract requires use_sdp=true")
    duration_noise = torch.randn(
        (tokens.size(0), 2, tokens.size(1)),
        generator=generator,
        device="cpu",
        dtype=encoded.dtype,
    )
    logw = captured_sdp_reverse(
        model.dp,
        encoded,
        text_mask,
        conditioning,
        duration_noise,
        noise_scale_w,
    )
    durations = torch.exp(logw) * text_mask * length_scale
    w_ceil = torch.ceil(durations)
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
        text_mask.dtype
    )
    attention_mask = torch.unsqueeze(text_mask, 2) * torch.unsqueeze(y_mask, -1)
    attention = commons.generate_path(w_ceil, attention_mask)

    expanded_m_p = torch.matmul(
        attention.squeeze(1), text_m_p.transpose(1, 2)
    ).transpose(1, 2)
    expanded_logs_p = torch.matmul(
        attention.squeeze(1), text_logs_p.transpose(1, 2)
    ).transpose(1, 2)
    # Upstream uses randn_like(m_p). The expanded prior is a transposed,
    # non-contiguous tensor, so preserving its strides is part of reproducing
    # which random value lands at each logical tensor index.
    latent_noise = torch.empty_like(expanded_m_p).normal_(generator=generator)
    z_p = expanded_m_p + latent_noise * torch.exp(expanded_logs_p) * noise_scale
    z = model.flow(z_p, y_mask, g=conditioning, reverse=True)
    pcm = model.dec(z * y_mask, g=conditioning)
    if pcm.ndim != 3 or pcm.shape[0] != 1 or pcm.shape[1] != 1:
        raise DumperError(
            f"VITS oracle produced non-mono output with shape {tuple(pcm.shape)}"
        )

    capture = {
        "input.token_ids": tokens.to(torch.int32),
        "random.duration_noise": duration_noise,
        "random.latent_noise": latent_noise,
        "text.m_p": text_m_p,
        "text.logs_p": text_logs_p,
        "text.mask": text_mask,
        "duration.logw": logw,
        "duration.w_ceil": w_ceil,
        "duration.y_length": y_lengths,
        "duration.attention": attention,
        "prior.m_p_expanded": expanded_m_p,
        "prior.logs_p_expanded": expanded_logs_p,
        "latent.z_p": z_p,
        "flow.z": z,
        "audio.pcm": pcm.reshape(-1),
    }
    if conditioning is not None:
        capture["voice.embedding"] = conditioning.squeeze(-1)
    return capture


def assert_tensor_equal(
    case_id: str, name: str, actual: torch.Tensor, expected: torch.Tensor
) -> None:
    if torch.equal(actual, expected):
        return
    if actual.shape != expected.shape:
        detail = f"shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    else:
        difference = torch.max(torch.abs(actual - expected)).item()
        detail = f"maximum absolute difference {difference}"
    raise DumperError(
        f"{case_id}: manual capture disagrees with model.infer at {name}: {detail}"
    )


def verify_against_upstream_infer(
    case_id: str,
    model: Any,
    token_ids: list[int],
    speaker_id: torch.Tensor | None,
    seed: int,
    noise_scale: float,
    noise_scale_w: float,
    length_scale: float,
    capture: dict[str, torch.Tensor],
) -> None:
    tokens = torch.tensor([token_ids], dtype=torch.long, device="cpu")
    token_lengths = torch.tensor([len(token_ids)], dtype=torch.long, device="cpu")
    torch.manual_seed(seed)
    pcm, attention, _y_mask, latent = model.infer(
        tokens,
        token_lengths,
        sid=speaker_id,
        noise_scale=noise_scale,
        noise_scale_w=noise_scale_w,
        length_scale=length_scale,
        max_len=None,
    )
    z, z_p, expanded_m_p, expanded_logs_p = latent
    assert_tensor_equal(
        case_id, "duration.attention", capture["duration.attention"], attention
    )
    assert_tensor_equal(
        case_id, "prior.m_p_expanded", capture["prior.m_p_expanded"], expanded_m_p
    )
    assert_tensor_equal(
        case_id,
        "prior.logs_p_expanded",
        capture["prior.logs_p_expanded"],
        expanded_logs_p,
    )
    assert_tensor_equal(case_id, "latent.z_p", capture["latent.z_p"], z_p)
    assert_tensor_equal(case_id, "flow.z", capture["flow.z"], z)
    assert_tensor_equal(case_id, "audio.pcm", capture["audio.pcm"], pcm.reshape(-1))
def safe_artifact_path(case_dir: Path, relative_path: str) -> Path:
    candidate = (case_dir / relative_path).resolve()
    root = case_dir.resolve()
    if not candidate.is_relative_to(root) or candidate == root:
        raise DumperError(f"artifact path escapes its case directory: {relative_path}")
    return candidate


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except OSError as error:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
        raise DumperError(f"cannot write artifact {path}: {error}") from error


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def tensor_payload(
    tensor: torch.Tensor, artifact_format: str
) -> tuple[bytes, list[int]]:
    detached = tensor.detach().cpu().contiguous()
    if artifact_format == "f32le":
        array = detached.to(torch.float32).numpy().astype("<f4", copy=False)
    elif artifact_format == "i32le":
        array = detached.to(torch.int32).numpy().astype("<i4", copy=False)
    elif artifact_format == "i64le":
        array = detached.to(torch.int64).numpy().astype("<i8", copy=False)
    else:
        raise DumperError(f"{artifact_format} is not a tensor artifact format")
    return array.tobytes(order="C"), list(array.shape)


class ArtifactWriter:
    def __init__(self, case_dir: Path, specs: dict[str, dict[str, str]]) -> None:
        self.case_dir = case_dir
        self.specs = specs
        self.records: list[dict[str, Any]] = []

    def write_tensor(self, name: str, tensor: torch.Tensor) -> None:
        spec = self._spec(name)
        payload, shape = tensor_payload(tensor, spec["format"])
        self._write(name, spec, payload, shape)

    def write_json(self, name: str, value: Any, *, record: bool = True) -> None:
        spec = self._spec(name)
        if spec["format"] != "json":
            raise DumperError(f"{name}: declared format is not json")
        payload = json_bytes(value)
        if record:
            self._write(name, spec, payload, None)
        else:
            atomic_write(safe_artifact_path(self.case_dir, spec["path"]), payload)

    def _spec(self, name: str) -> dict[str, str]:
        try:
            return self.specs[name]
        except KeyError as error:
            raise DumperError(f"artifact {name} is not declared by the case") from error

    def _write(
        self,
        name: str,
        spec: dict[str, str],
        payload: bytes,
        shape: list[int] | None,
    ) -> None:
        path = safe_artifact_path(self.case_dir, spec["path"])
        atomic_write(path, payload)
        self.records.append(
            {
                "name": name,
                "path": spec["path"],
                "format": spec["format"],
                "shape": shape,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )


def materialize_case(
    case: dict[str, Any],
    output_root: Path,
    model: Any,
    hps: Any,
    checkpoint_info: dict[str, Any],
    checkpoint_sha256: str,
    config_sha256: str,
    manifest: dict[str, Any],
    commons: Any,
    text_to_sequence: Any,
    threads: int,
) -> dict[str, Any]:
    case_id = case["id"]
    specs = case_artifact_specs(case)
    if model.n_speakers > 0:
        if "voice.embedding" not in specs:
            raise DumperError(
                f"{case_id}: multi-speaker case must declare voice.embedding"
            )
    elif "voice.embedding" in specs:
        raise DumperError(f"{case_id}: single-speaker case cannot declare voice.embedding")

    seed = parse_seed(case["request"]["seed_u64"], case_id)
    speaking_rate = float(case["request"]["speaking_rate"])
    length_scale = 1.0 / speaking_rate
    parameters = case["oracle"]["parameters"]
    noise_scale = require_number(parameters, "noise_scale")
    noise_scale_w = require_number(parameters, "noise_scale_w")
    declared_length_scale = require_number(parameters, "length_scale")
    if not np.isclose(declared_length_scale, length_scale, rtol=0.0, atol=1e-12):
        raise DumperError(
            f"{case_id}: oracle length_scale must equal 1/speaking_rate "
            f"({length_scale}), got {declared_length_scale}"
        )

    token_ids = resolve_token_ids(case, hps, text_to_sequence, commons)
    speaker_id = resolve_voice(case, model)
    with torch.inference_mode():
        capture = infer_with_captured_randomness(
            model,
            token_ids,
            speaker_id,
            seed,
            noise_scale,
            noise_scale_w,
            length_scale,
            commons,
        )
        verify_against_upstream_infer(
            case_id,
            model,
            token_ids,
            speaker_id,
            seed,
            noise_scale,
            noise_scale_w,
            length_scale,
            capture,
        )

    pcm = capture["audio.pcm"]
    if not bool(torch.isfinite(pcm).all()):
        raise DumperError(f"{case_id}: generated PCM contains NaN or infinity")
    frames = int(pcm.numel())
    sample_rate = int(hps.data.sampling_rate)
    writer = ArtifactWriter(output_root / case_id, specs)

    tensor_order = [
        "input.token_ids",
        "random.duration_noise",
        "random.latent_noise",
        "text.m_p",
        "text.logs_p",
        "text.mask",
        "duration.logw",
        "duration.w_ceil",
        "duration.y_length",
        "duration.attention",
        "prior.m_p_expanded",
        "prior.logs_p_expanded",
        "latent.z_p",
        "flow.z",
    ]
    if "voice.embedding" in capture:
        tensor_order.append("voice.embedding")
    tensor_order.append("audio.pcm")
    for name in tensor_order:
        writer.write_tensor(name, capture[name])

    result = {
        "status": "ok",
        "sample_rate_hz": sample_rate,
        "channels": 1,
        "sample_format": "f32le",
        "frames": frames,
        "duration_seconds": frames / sample_rate,
        "seed_u64": str(seed),
        "speaking_rate": speaking_rate,
    }
    writer.write_json("result", result)
    metadata = {
        "schema": "synthesize-vits-reference-artifacts-v1",
        "family": manifest["family"],
        "variant": manifest["variant"],
        "case_id": case_id,
        "source": {
            "repository": manifest["source"]["repository"],
            "revision": manifest["source"]["revision"],
            "config_sha256": config_sha256,
            "checkpoint_sha256": checkpoint_sha256,
        },
        "reference": {
            "implementation": manifest["reference"]["implementation"],
            "framework": manifest["reference"]["framework"],
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "dtype": "float32",
            "intra_op_threads": threads,
            "deterministic_algorithms": True,
            "upstream_infer_exact_match": True,
        },
        "checkpoint": checkpoint_info,
        "input": {
            "kind": "token_ids",
            "language_tag": case["input"].get("language_tag"),
            "source_text": case["input"].get("source_text"),
            "token_count": len(token_ids),
            "text_cleaners": list(hps.data.text_cleaners),
            "add_blank": bool(hps.data.add_blank),
        },
        "voice": case["voice"],
        "request": case["request"],
        "oracle_parameters": {
            "noise_scale": noise_scale,
            "noise_scale_w": noise_scale_w,
            "length_scale": length_scale,
        },
        "result": result,
        "artifacts": writer.records,
    }
    written_names = {record["name"] for record in writer.records}
    unwritten_names = set(specs) - written_names - {"metadata"}
    if unwritten_names:
        raise DumperError(
            f"{case_id}: declared artifacts were not written: "
            + ", ".join(sorted(unwritten_names))
        )
    writer.write_json("metadata", metadata, record=False)
    return {
        "case_id": case_id,
        "status": "ok",
        "frames": frames,
        "sample_rate_hz": sample_rate,
        "pcm_sha256": next(
            record["sha256"]
            for record in writer.records
            if record["name"] == "audio.pcm"
        ),
        "output_dir": str(output_root / case_id),
    }


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    source_dir = args.source_dir.resolve()
    checkpoint_path = args.checkpoint.resolve()
    if not source_dir.is_dir():
        raise DumperError(f"source directory does not exist: {source_dir}")
    if not checkpoint_path.is_file():
        raise DumperError(f"checkpoint does not exist: {checkpoint_path}")

    manifest = load_json(manifest_path)
    selected_cases = validate_manifest(manifest, args.case_ids)
    configure_torch(args.threads)

    revision = manifest["source"]["revision"]
    verify_source_revision(source_dir, revision)
    config_artifact = source_artifact(manifest, "config", require_repository_path=True)
    checkpoint_artifact = source_artifact(manifest, "checkpoint")
    config_path = (source_dir / config_artifact["repository_path"]).resolve()
    if not config_path.is_relative_to(source_dir):
        raise DumperError("config repository_path escapes source-dir")
    config_sha256 = require_sha256(config_path, config_artifact["sha256"], "config")
    checkpoint_sha256 = require_sha256(
        checkpoint_path, checkpoint_artifact["sha256"], "checkpoint"
    )

    commons, models, utils, text_to_sequence, symbols = import_upstream(source_dir)
    model, hps, checkpoint_info = load_model(
        config_path, checkpoint_path, models, utils, symbols
    )
    native_audio = manifest["package_contract"]["native_audio"]
    if (
        int(hps.data.sampling_rate) != native_audio["sample_rate_hz"]
        or native_audio["channels"] != 1
        or native_audio["sample_format"] != "f32le"
    ):
        raise DumperError("manifest native_audio disagrees with the pinned VITS config")

    output_root_value = (
        args.output_root
        if args.output_root is not None
        else Path(manifest["case_artifact_root"])
    )
    output_root = output_root_value.resolve()
    for case in selected_cases:
        summary = materialize_case(
            case,
            output_root,
            model,
            hps,
            checkpoint_info,
            checkpoint_sha256,
            config_sha256,
            manifest,
            commons,
            text_to_sequence,
            args.threads,
        )
        print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DumperError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
