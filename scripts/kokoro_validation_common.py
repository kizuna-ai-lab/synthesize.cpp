"""Shared driver for the per-stage Kokoro Golden validators.

Each `validate-kokoro-<stage>.py` is a thin wrapper over `run_stage` below. The
stages take identical arguments and differ only in which probes they compare, so
the parts that could drift between them — how a case is resolved, how the runner
is invoked, how a report is written — live here once.

Two comparison rules are specific to this family and are the reason a plain
element-wise difference is not enough everywhere:

- The harmonic source's spectrum is stored as magnitudes followed by phases. A
  phase is meaningless where its magnitude is near zero, so the two halves are
  recombined into a complex value and compared as one quantity.
- The decoder's spectrum leaves the graph as a logarithm and a pre-sine angle.
  Its logarithms reach about -69, which is a bin that is numerically silent, and
  a flat difference over those rows measures nothing. The two halves are given
  their activations and compared as the complex value the inverse transform
  actually consumes.
- The waveform cannot be compared sample-wise. The excitation's phase is a
  cumulative sum over the whole utterance, which makes it chaotic in F0 by
  construction, so agreement is stated as correlation and spectral distance.
  See reports/porting/kokoro/kokoro-v1-0/_porting-log.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable

import numpy as np
from gguf import GGUFReader


SUPPORTED_VARIANTS = ("kokoro-v1-0",)
SUPPORTED_QUANTIZATION_PROFILES = ("F32", "F16", "Q8_MIXED")


class ValidationError(RuntimeError):
    pass


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValidationError(f"{path} must contain a JSON object")
    return value


def display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def manifest_variant(manifest: dict[str, Any]) -> str:
    variant = manifest.get("variant")
    if manifest.get("family") != "kokoro" or variant not in SUPPORTED_VARIANTS:
        raise ValidationError(
            "validator requires a supported Kokoro reference variant: "
            + ", ".join(SUPPORTED_VARIANTS)
        )
    return variant


def model_quantization_identity(model_path: Path) -> tuple[str, int]:
    reader = GGUFReader(model_path)
    profile_field = reader.fields.get("synthesize.quantization.profile")
    version_field = reader.fields.get("synthesize.quantization.profile_version")
    if profile_field is None or version_field is None:
        raise ValidationError("model is missing quantization profile metadata")
    profile = profile_field.contents()
    version = version_field.contents()
    if profile not in SUPPORTED_QUANTIZATION_PROFILES:
        raise ValidationError(f"unsupported quantization profile: {profile}")
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        raise ValidationError("quantization profile version must be a positive integer")
    return profile, version


def validation_report_name(
    variant: str, slice_name: str, backend: str, profile: str
) -> str:
    if variant not in SUPPORTED_VARIANTS:
        raise ValidationError(f"unsupported Kokoro variant: {variant}")
    if profile not in SUPPORTED_QUANTIZATION_PROFILES:
        raise ValidationError(f"unsupported quantization profile: {profile}")
    if backend not in ("cpu", "cuda"):
        raise ValidationError(f"unsupported validation backend: {backend}")
    profile_component = "" if profile == "F32" else f"-{profile}"
    backend_component = "" if backend == "cpu" else f"-{backend}"
    return f"{variant}{profile_component}-{slice_name}{backend_component}.json"


def artifact_path(case: dict[str, Any], name: str) -> str:
    for artifact in case.get("expected", {}).get("artifacts", []):
        if artifact.get("name") == name and isinstance(artifact.get("path"), str):
            return artifact["path"]
    for artifact in case.get("oracle", {}).get("stochastic_inputs", []):
        if artifact.get("name") == name and isinstance(artifact.get("path"), str):
            return artifact["path"]
    raise ValidationError(f"case {case.get('id')} has no artifact {name}")


def voice_index(manifest: dict[str, Any], case: dict[str, Any]) -> int:
    preset_ids = manifest["package_contract"]["voices"]["preset_ids"]
    voice = case.get("voice")
    if not isinstance(voice, dict) or voice.get("kind") != "preset_voice":
        raise ValidationError(f"{case.get('id')}: every case requires a preset voice")
    try:
        return preset_ids.index(voice["id"])
    except ValueError as error:
        raise ValidationError(
            f"{case.get('id')}: voice {voice.get('id')} is not in the preset catalog"
        ) from error


def speaking_rate(case: dict[str, Any]) -> float:
    rate = case.get("request", {}).get("speaking_rate", 1.0)
    if not isinstance(rate, (int, float)) or isinstance(rate, bool):
        raise ValidationError(f"{case.get('id')}: speaking_rate must be a number")
    return float(rate)


def _finite(actual: np.ndarray, candidate: Path) -> None:
    if not bool(np.isfinite(actual).all()):
        raise ValidationError(f"non-finite candidate values in {candidate}")


def _load_pair(reference: Path, candidate: Path, dtype: str) -> tuple[np.ndarray, np.ndarray]:
    expected = np.fromfile(reference, dtype=dtype)
    actual = np.fromfile(candidate, dtype=dtype)
    if expected.shape != actual.shape:
        raise ValidationError(
            f"shape mismatch: {reference} has {expected.shape}, {candidate} has {actual.shape}"
        )
    return expected, actual


def compare_f32(reference: Path, candidate: Path) -> dict[str, Any]:
    expected, actual = _load_pair(reference, candidate, "<f4")
    _finite(actual, candidate)
    difference = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    return {
        "elements": int(expected.size),
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()) if difference.size else 0.0,
        "exact": bool(np.array_equal(expected, actual)),
    }


def compare_i64(reference: Path, candidate: Path) -> dict[str, Any]:
    expected, actual = _load_pair(reference, candidate, "<i8")
    return {
        "elements": int(expected.size),
        "max_abs": float(np.abs(expected - actual).max(initial=0)),
        "mean_abs": float(np.abs(expected - actual).mean()) if expected.size else 0.0,
        "exact": bool(np.array_equal(expected, actual)),
    }


def compare_spectrum(reference: Path, candidate: Path) -> dict[str, Any]:
    """Compares magnitude and phase halves as one complex quantity."""
    expected, actual = _load_pair(reference, candidate, "<f4")
    _finite(actual, candidate)
    rows = 22
    if expected.size % rows:
        raise ValidationError(f"{reference} is not a {rows}-row spectrum")
    bins = rows // 2
    a = actual.astype(np.float64).reshape(rows, -1)
    b = expected.astype(np.float64).reshape(rows, -1)
    difference = np.abs(a[:bins] * np.exp(1j * a[bins:]) - b[:bins] * np.exp(1j * b[bins:]))
    magnitude = np.abs(b[:bins] - a[:bins])
    return {
        "elements": int(expected.size),
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()) if difference.size else 0.0,
        "magnitude_max_abs": float(magnitude.max(initial=0.0)),
        "exact": bool(np.array_equal(expected, actual)),
    }


def compare_log_spectrum(reference: Path, candidate: Path) -> dict[str, Any]:
    """Applies the exponential and the sine, then compares as one complex value."""
    expected, actual = _load_pair(reference, candidate, "<f4")
    _finite(actual, candidate)
    rows = 22
    if expected.size % rows:
        raise ValidationError(f"{reference} is not a {rows}-row spectrum")
    bins = rows // 2
    a = actual.astype(np.float64).reshape(rows, -1)
    b = expected.astype(np.float64).reshape(rows, -1)
    ca = np.exp(a[:bins]) * np.exp(1j * np.sin(a[bins:]))
    cb = np.exp(b[:bins]) * np.exp(1j * np.sin(b[bins:]))
    difference = np.abs(ca - cb)
    return {
        "elements": int(expected.size),
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()) if difference.size else 0.0,
        "reference_magnitude_max": float(np.abs(cb).max(initial=0.0)),
        # The raw halves are still reported, so a change in the graph's output
        # is visible even when the applied value hides it.
        "log_magnitude_max_abs": float(np.abs(a[:bins] - b[:bins]).max(initial=0.0)),
        "angle_max_abs": float(np.abs(a[bins:] - b[bins:]).max(initial=0.0)),
        "exact": bool(np.array_equal(expected, actual)),
    }


def compare_waveform(reference: Path, candidate: Path) -> dict[str, Any]:
    expected, actual = _load_pair(reference, candidate, "<f4")
    _finite(actual, candidate)
    a = actual.astype(np.float64)
    b = expected.astype(np.float64)
    difference = np.abs(a - b)
    window = np.hanning(1024)
    frames = range(0, max(len(a) - 1024, 0), 256)
    spec_a = np.stack([np.abs(np.fft.rfft(a[i : i + 1024] * window)) for i in frames])
    spec_b = np.stack([np.abs(np.fft.rfft(b[i : i + 1024] * window)) for i in frames])
    return {
        "elements": int(expected.size),
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()) if difference.size else 0.0,
        "rms_error": float(np.sqrt((difference**2).mean())) if difference.size else 0.0,
        "reference_rms": float(np.sqrt((b**2).mean())) if b.size else 0.0,
        "correlation": float(np.corrcoef(a, b)[0, 1]),
        "spectrogram_correlation": float(
            np.corrcoef(spec_a.ravel(), spec_b.ravel())[0, 1]
        ),
        "exact": bool(np.array_equal(expected, actual)),
    }


COMPARERS: dict[str, Callable[[Path, Path], dict[str, Any]]] = {
    "f32": compare_f32,
    "i64": compare_i64,
    "spectrum": compare_spectrum,
    "log_spectrum": compare_log_spectrum,
    "waveform": compare_waveform,
}


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def run_stage(
    stage: str,
    slice_name: str,
    probes: tuple[tuple[str, str, str, bool], ...],
    description: str,
) -> int:
    """Runs one stage over every manifest case and writes its report.

    Each probe is (logical name, runner filename, comparer, structural). A
    structural probe must match exactly: it determines a shape or a length, so a
    difference there is a defect rather than drift.
    """
    args = parse_args(description)
    if args.threads <= 0:
        raise ValidationError("--threads must be positive")
    project_root = Path(__file__).resolve().parent.parent
    manifest_path = args.manifest.resolve()
    model_path = args.model.resolve()
    runner_path = args.runner.resolve()
    if not manifest_path.is_file() or not model_path.is_file() or not runner_path.is_file():
        raise ValidationError("manifest, model, and runner must be files")

    manifest = load_json(manifest_path)
    variant = manifest_variant(manifest)
    profile, profile_version = model_quantization_identity(model_path)
    case_root = project_root / manifest["case_artifact_root"]
    output_root = project_root / "build" / "validate" / "kokoro" / variant / profile

    case_reports = []
    for case in manifest.get("cases", []):
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValidationError("every case requires an ID")
        reference_dir = case_root / case_id
        if not reference_dir.is_dir():
            raise ValidationError(f"{case_id}: reference artifacts have not been dumped")
        output_dir = output_root / case_id / "cpp" / (
            slice_name if args.backend == "cpu" else f"{slice_name}_{args.backend}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Token IDs are inline in the manifest, so the runner's input is
        # materialized here rather than read from a committed artifact.
        token_ids = case.get("input", {}).get("token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise ValidationError(f"{case_id}: the manifest carries no inline token IDs")
        tokens_path = output_dir / "token_ids.i32"
        tokens_path.write_bytes(np.asarray(token_ids, dtype="<i4").tobytes())

        command = [
            str(runner_path),
            stage,
            str(model_path),
            str(tokens_path),
            str(reference_dir / "random"),
            str(output_dir),
            str(voice_index(manifest, case)),
            repr(speaking_rate(case)),
            str(args.threads),
            args.backend,
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise ValidationError(
                f"{case_id}: runner failed with {completed.returncode}: {completed.stderr.strip()}"
            )

        results = {}
        for logical_name, filename, comparer, structural in probes:
            results[logical_name] = COMPARERS[comparer](
                reference_dir / artifact_path(case, logical_name),
                output_dir / filename,
            )
            if structural and not results[logical_name]["exact"]:
                raise ValidationError(f"{case_id}: structural probe {logical_name} differs")
        case_reports.append({"id": case_id, "probes": results})

    if not case_reports:
        raise ValidationError("the manifest declares no cases")

    worst = {}
    for logical_name, _, _, _ in probes:
        worst_case = max(case_reports, key=lambda item: item["probes"][logical_name]["max_abs"])
        worst[logical_name] = {"case_id": worst_case["id"], **worst_case["probes"][logical_name]}

    report = {
        "schema": "synthesize-slice-validation-v1",
        "family": "kokoro",
        "variant": variant,
        "slice": slice_name,
        "status": "numeric_drift_measured",
        "tolerance_gate": "not_finalized",
        "reference_dtype": "F32",
        "profile": profile,
        "profile_version": profile_version,
        "backend": args.backend.upper(),
        "threads": args.threads,
        "manifest": display_path(manifest_path, project_root),
        "manifest_sha256": sha256_file(manifest_path),
        "model": display_path(model_path, project_root),
        "model_sha256": sha256_file(model_path),
        "runner": display_path(runner_path, project_root),
        # The manifest and the model are hashed, and the runner was not: a
        # report could name a binary and describe a different one. A stale
        # accelerator build once produced a decoder divergence of 1e14 that
        # way, and nothing in the report could have shown it.
        "runner_sha256": sha256_file(runner_path),
        "runner_mtime": int(runner_path.stat().st_mtime),
        "case_count": len(case_reports),
        "worst": worst,
        "cases": case_reports,
    }
    report_name = validation_report_name(variant, slice_name, args.backend, profile)
    report_path = project_root / "reports" / "validate" / "kokoro" / report_name
    write_json_atomic(report_path, report)
    print(
        json.dumps(
            {"status": report["status"], "cases": len(case_reports), "worst": worst},
            sort_keys=True,
        )
    )
    return 0


def main_wrapper(
    stage: str, slice_name: str, probes: tuple[tuple[str, str, str, bool], ...], description: str
) -> None:
    try:
        raise SystemExit(run_stage(stage, slice_name, probes, description))
    except (OSError, ValueError, KeyError, json.JSONDecodeError, ValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
