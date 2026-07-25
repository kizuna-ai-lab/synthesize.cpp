#!/usr/bin/env python3
"""Measure the in-progress C++ VITS text-encoder slice against its oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

from vits_validation_common import (
    manifest_variant,
    model_quantization_identity,
    validation_report_name,
)


class ValidationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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


def artifact_path(case: dict[str, Any], name: str) -> str:
    if name == "input.token_ids":
        artifact = case.get("input", {}).get("artifact", {})
        if artifact.get("name") == name and isinstance(artifact.get("path"), str):
            return artifact["path"]
    for artifact in case.get("expected", {}).get("artifacts", []):
        if artifact.get("name") == name and isinstance(artifact.get("path"), str):
            return artifact["path"]
    raise ValidationError(f"case {case.get('id')} has no artifact {name}")


def compare_f32(reference: Path, candidate: Path) -> dict[str, Any]:
    expected = np.fromfile(reference, dtype="<f4")
    actual = np.fromfile(candidate, dtype="<f4")
    if expected.shape != actual.shape:
        raise ValidationError(
            f"shape mismatch: {reference} has {expected.shape}, {candidate} has {actual.shape}"
        )
    if not bool(np.isfinite(actual).all()):
        raise ValidationError(f"non-finite candidate values in {candidate}")
    difference = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    return {
        "elements": int(expected.size),
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()) if difference.size else 0.0,
        "exact": bool(np.array_equal(expected, actual)),
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


def main() -> int:
    args = parse_args()
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
    output_root = project_root / "build" / "validate" / "vits" / variant / profile

    case_reports = []
    for case in manifest.get("cases", []):
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValidationError("every case requires an ID")
        reference_dir = case_root / case_id
        output_dir = output_root / case_id / "cpp" / (
            "text" if args.backend == "cpu" else f"text_{args.backend}"
        )
        command = [
            str(runner_path),
            str(model_path),
            str(reference_dir / artifact_path(case, "input.token_ids")),
            str(output_dir),
            str(args.threads),
            args.backend,
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise ValidationError(
                f"{case_id}: runner failed with {completed.returncode}: {completed.stderr.strip()}"
            )

        probes = {}
        for logical_name, filename in (
            ("text.m_p", "m_p.f32"),
            ("text.logs_p", "logs_p.f32"),
            ("text.mask", "mask.f32"),
        ):
            probes[logical_name] = compare_f32(
                reference_dir / artifact_path(case, logical_name),
                output_dir / filename,
            )
        if not probes["text.mask"]["exact"]:
            raise ValidationError(f"{case_id}: structural text.mask probe differs")
        case_reports.append({"id": case_id, "probes": probes})

    worst = {}
    for logical_name in ("text.m_p", "text.logs_p", "text.mask"):
        worst_case = max(
            case_reports,
            key=lambda item: item["probes"][logical_name]["max_abs"],
        )
        worst[logical_name] = {
            "case_id": worst_case["id"],
            **worst_case["probes"][logical_name],
        }

    report = {
        "schema": "synthesize-slice-validation-v1",
        "family": "vits",
        "variant": variant,
        "slice": "text_encoder",
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
        # Hashed for the same reason the model is: a report has to identify the
        # binary it drove, not just name a path that may since have been rebuilt.
        "runner_sha256": sha256_file(runner_path),
        "case_count": len(case_reports),
        "worst": worst,
        "cases": case_reports,
    }
    report_name = validation_report_name(variant, "text-encoder", args.backend, profile)
    report_path = project_root / "reports" / "validate" / "vits" / report_name
    write_json_atomic(report_path, report)
    print(json.dumps({"status": report["status"], "cases": len(case_reports), "worst": worst}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError, ValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
