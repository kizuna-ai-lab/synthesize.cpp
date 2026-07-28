#!/usr/bin/env python3
"""Stage 5 phase 2: compare the port's replay against the oracle, per case.

The oracle sampled its codes from PyTorch's generator and this port draws from
its own, so parity replays the oracle's codes rather than reproducing them --
see `docs/port-validation.md`, "Stochastic Replay and Public Seeds". Everything
downstream of that draw is then compared on identical inputs.

Two comparison rules, and the difference matters:

- **Deep talker layers use cosine similarity, not max-abs.** Reading the
  reference port showed per-layer max-abs of 13.07 / 13.09 / 13.42 at L7 / L14 /
  L21 and 66.8 at L27 while cosine held at or above 0.9998. Those are outlier
  channels ahead of the final norm; a max-abs threshold reports catastrophic
  failure on a correct port.
- **Audio uses max-abs**, because a waveform is what a listener gets and a
  cosine over it would hide a constant offset.

Nothing here decides pass or fail against a committed threshold yet: stage 5's
first job is to *measure*, so the numbers land in tests/tolerances/qwen3-tts.json
after review. `--check` compares against that file once it is filled.

    uv run --project scripts/envs/qwen3-tts --locked python \
      scripts/validate-qwen3-tts-replay.py --report reports/validate/qwen3-tts/replay.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

PROBE_LAYERS = (0, 7, 14, 21, 27)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path,
                        default=pathlib.Path("tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-customvoice.manifest.json"))
    parser.add_argument("--model", type=pathlib.Path,
                        default=pathlib.Path("models/qwen3-tts-12hz-0-6b-customvoice/"
                                             "qwen3-tts-12hz-0-6b-customvoice-BF16.gguf"))
    parser.add_argument("--runner", type=pathlib.Path,
                        default=pathlib.Path("build/unit/bin/synthesize-qwen3-tts-replay-real"))
    parser.add_argument("--work", type=pathlib.Path, default=pathlib.Path("build/goldens/qwen3-tts-replay"))
    parser.add_argument("--report", type=pathlib.Path, default=None)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--check", action="store_true",
                        help="fail when a measurement exceeds the committed tolerance")
    parser.add_argument("--tolerances", type=pathlib.Path,
                        default=pathlib.Path("tests/tolerances/qwen3-tts.json"))
    parser.add_argument("--stage", default="source-bf16-oracle-vs-f32-cpu")
    return parser.parse_args()


# The public interface speaks BCP-47 and the manifest names languages in full,
# the same bridge the family makes internally.
LANGUAGE_TAGS = {
    "english": "en", "german": "de", "spanish": "es", "chinese": "zh", "japanese": "ja",
    "french": "fr", "korean": "ko", "russian": "ru", "italian": "it", "portuguese": "pt",
}


def read_f32(path: pathlib.Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype=np.float32)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left)) * float(np.linalg.norm(right))
    if denominator == 0.0:
        return 1.0 if float(np.abs(left - right).max(initial=0.0)) == 0.0 else 0.0
    return float(np.dot(left, right) / denominator)


def compare(expected: np.ndarray, actual: np.ndarray) -> dict:
    if expected.shape != actual.shape:
        return {"shape_mismatch": [list(expected.shape), list(actual.shape)]}
    difference = np.abs(expected - actual)
    return {
        "elements": int(expected.size),
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()) if expected.size else 0.0,
        "cosine": cosine(expected, actual),
    }


def run_case(arguments: argparse.Namespace, case: dict, oracle_root: pathlib.Path) -> dict | None:
    case_id = case["id"]
    oracle = oracle_root / case_id
    if not (oracle / "codes" / "semantic.i32").exists():
        return None
    work = arguments.work / case_id
    work.mkdir(parents=True, exist_ok=True)

    language = case["input"].get("language_tag", "english")
    command = [
        str(arguments.runner), str(arguments.model), str(oracle), str(work),
        case["voice"]["id"], LANGUAGE_TAGS.get(language, "en"),
        *[str(layer) for layer in PROBE_LAYERS],
    ]
    finished = subprocess.run(command, capture_output=True, text=True)
    if finished.returncode != 0:
        return {"case": case_id, "status": "runner-failed", "stderr": finished.stderr.strip()[:400]}

    measurements = {"audio.pcm": compare(read_f32(oracle / "audio" / "pcm.f32"), read_f32(work / "pcm.f32"))}
    measurements["talker.logits"] = compare(read_f32(oracle / "talker" / "logits.f32"),
                                            read_f32(work / "talker_logits.f32"))
    measurements["talker.final"] = compare(read_f32(oracle / "talker" / "final.f32"),
                                           read_f32(work / "talker_final.f32"))
    for layer in PROBE_LAYERS:
        probe = oracle / "talker" / f"hidden_l{layer}.f32"
        if probe.exists():
            measurements[f"talker.hidden_l{layer}"] = compare(read_f32(probe),
                                                              read_f32(work / f"talker_l{layer}.f32"))
    return {"case": case_id, "status": "ok", "probes": measurements}


def main() -> int:
    arguments = parse_args()
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    oracle_root = pathlib.Path(manifest["case_artifact_root"])

    results = []
    for case in manifest["cases"]:
        if arguments.cases and case["id"] not in arguments.cases:
            continue
        outcome = run_case(arguments, case, oracle_root)
        if outcome is None:
            print(f"  {case['id']}: no oracle artifacts, skipped")
            continue
        results.append(outcome)
        if outcome["status"] != "ok":
            print(f"  {case['id']}: {outcome['status']}: {outcome.get('stderr', '')}")
            continue
        audio = outcome["probes"]["audio.pcm"]
        final = outcome["probes"]["talker.final"]
        print(f"  {case['id']}: audio max_abs {audio.get('max_abs', float('nan')):.4g}"
              f"  talker.final cosine {final.get('cosine', float('nan')):.6f}")

    # The worst case across the suite is what a tolerance has to cover.
    worst: dict[str, dict] = {}
    for outcome in results:
        for probe, measurement in outcome.get("probes", {}).items():
            if "max_abs" not in measurement:
                continue
            current = worst.setdefault(probe, {"max_abs": 0.0, "min_cosine": 1.0})
            current["max_abs"] = max(current["max_abs"], measurement["max_abs"])
            current["min_cosine"] = min(current["min_cosine"], measurement["cosine"])

    print("\nworst across the suite:")
    for probe in sorted(worst):
        entry = worst[probe]
        print(f"  {probe:24s} max_abs {entry['max_abs']:.6g}  min_cosine {entry['min_cosine']:.6f}")

    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(json.dumps({
            "schema": "synthesize-validation-report-v1",
            "family": "qwen3-tts",
            "variant": manifest["variant"],
            "suite_version": manifest["suite_version"],
            "phase": "oracle_replay",
            "cases": results,
            "worst": worst,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"\nreport: {arguments.report}")

    failures = [outcome["case"] for outcome in results if outcome["status"] != "ok"]
    if failures:
        print(f"\n{len(failures)} case(s) failed to run: {failures}")
        return 1
    if not arguments.check:
        return 0

    # A tolerance is a reviewed number in a committed file. Refusing to run
    # rather than inventing one is the point: a threshold the suite writes for
    # itself proves nothing.
    tolerances = json.loads(arguments.tolerances.read_text(encoding="utf-8"))
    stage = tolerances.get("stages", {}).get(arguments.stage)
    if not stage:
        print(f"\ntolerance stage {arguments.stage!r} is not recorded in {arguments.tolerances}")
        return 1

    breaches = []
    for outcome in results:
        for probe, measurement in outcome.get("probes", {}).items():
            limits = stage.get("probes", {}).get(probe)
            if limits is None:
                breaches.append(f"{outcome['case']}/{probe}: no tolerance recorded")
                continue
            if "max_abs" not in measurement:
                breaches.append(f"{outcome['case']}/{probe}: {measurement}")
                continue
            if "max_abs" in limits and measurement["max_abs"] > limits["max_abs"]:
                breaches.append(f"{outcome['case']}/{probe}: max_abs {measurement['max_abs']:.6g} "
                                f"exceeds {limits['max_abs']:.6g}")
            if "min_cosine" in limits and measurement["cosine"] < limits["min_cosine"]:
                breaches.append(f"{outcome['case']}/{probe}: cosine {measurement['cosine']:.6f} "
                                f"below {limits['min_cosine']:.6f}")
    if breaches:
        print(f"\n{len(breaches)} tolerance breach(es):")
        for breach in breaches[:20]:
            print(f"  {breach}")
        return 1
    print(f"\nall probes within the {arguments.stage} tolerances")
    return 0


if __name__ == "__main__":
    sys.exit(main())
