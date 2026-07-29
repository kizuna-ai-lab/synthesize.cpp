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
import os
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
    # The cell of the tolerance grid this run fills. Three coordinates rather
    # than one name: the grid is keyed on Quantization Profile, Execution
    # Backend and stage, so a single string cannot address it.
    parser.add_argument("--profile", default="BF16")
    parser.add_argument("--backend", default="CPU")
    parser.add_argument("--stage", default="replay")
    # Places the codec on the primary backend, which is what stage 7 measured.
    # It also changes what `backend_placement` must see: on a CPU run every node
    # of every stage has to be on the CPU, and with this set the codec's must all
    # have left it.
    parser.add_argument("--accelerate", action="store_true",
                        help="run the codec on the primary backend and require it to land there")
    return parser.parse_args()


# The public interface speaks BCP-47 and the manifest names languages in full,
# the same bridge the family makes internally. "auto" passes through: it is the
# reference's no-think prompt, which is a position shorter than any named
# language, and a fallback that quietly turned it into "en" made the port build a
# prompt one position longer than the oracle's.
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


# The Golden Manifest's per-case `checks` array, finally read.
#
# Every case of every family has declared all eight values since the schema was
# written, and nothing anywhere consumed the field: `backend_placement` and
# `resource_cleanup` named obligations no code enforced, and the other six were
# satisfied by validators that decide what to run from their own stage rather
# than from the manifest. A declaration nothing reads is not a contract.
#
# Only the checks this validator is in a position to decide are dispatched here.
# `resource_cleanup` belongs to a process that loads and frees repeatedly, which
# is a C++ test rather than a replay comparison, so it is reported as delegated
# instead of silently passing.
PLACEMENT_CHECKS = {"backend_placement"}
DELEGATED_CHECKS = {"resource_cleanup"}


def evaluate_case_checks(case: dict, observed: dict, arguments: argparse.Namespace) -> dict:
    """Decide the declared checks this stage owns, and say which it does not."""
    declared = list(case.get("checks", []))
    verdicts: dict[str, str] = {}

    for name in declared:
        if name in DELEGATED_CHECKS:
            verdicts[name] = "delegated"
        elif name not in PLACEMENT_CHECKS:
            # tensor_parity, structural_exactness, waveform_regression,
            # finite_pcm, request_repeatability and result_metadata are decided
            # by the probe comparison and the phases around it, not here.
            verdicts[name] = "covered-by-probes"

    if "backend_placement" not in declared:
        return verdicts

    placement = observed.get("placement")
    if not placement:
        verdicts["backend_placement"] = "unobserved: the runner reported no placement"
        return verdicts

    failures = []
    # The rule from docs/backends.md: a sampled code is a discrete output, so the
    # stages feeding it stay on the CPU however the request asks for a backend.
    for stage in ("talker", "predictor"):
        nodes, off_cpu = placement[stage]
        if off_cpu != 0:
            failures.append(f"{stage} placed {off_cpu} of {nodes} nodes off the CPU")
    codec_nodes, codec_off_cpu = placement["codec"]
    if arguments.accelerate:
        # Claiming a backend means proving the work reached it, not that a device
        # was present: a graph placed on an accelerator and silently fell back
        # looks identical from outside.
        if codec_off_cpu != codec_nodes:
            failures.append(f"codec placed {codec_off_cpu} of {codec_nodes} nodes off the CPU, expected all")
    elif codec_off_cpu != 0:
        failures.append(f"codec placed {codec_off_cpu} of {codec_nodes} nodes off the CPU on a CPU run")

    verdicts["backend_placement"] = "ok" if not failures else "; ".join(failures)
    return verdicts


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
        case["voice"]["id"], "auto" if language == "auto" else LANGUAGE_TAGS[language],
        *[str(layer) for layer in PROBE_LAYERS],
    ]
    environment = dict(os.environ)
    if arguments.accelerate:
        environment["SYNTH_QWEN3_TTS_ACCELERATE"] = "1"
    finished = subprocess.run(command, capture_output=True, text=True, env=environment)
    if finished.returncode != 0:
        return {"case": case_id, "status": "runner-failed", "stderr": finished.stderr.strip()[:400]}

    # The runner's own report, which carries where each stage's nodes ran.
    observed = {}
    for line in reversed(finished.stdout.strip().splitlines()):
        if line.startswith("{"):
            observed = json.loads(line)
            break

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
    return {"case": case_id, "status": "ok", "probes": measurements,
            "checks": evaluate_case_checks(case, observed, arguments)}


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
        placement_verdict = outcome["checks"].get("backend_placement")
        if placement_verdict not in (None, "ok"):
            print(f"    backend_placement: {placement_verdict}")

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

    placement_failures = [
        (outcome["case"], outcome["checks"]["backend_placement"])
        for outcome in results
        if outcome.get("status") == "ok"
        and outcome.get("checks", {}).get("backend_placement") not in (None, "ok")
    ]
    if placement_failures:
        print(f"\n{len(placement_failures)} case(s) failed the declared backend_placement check:")
        for case_id, verdict in placement_failures:
            print(f"  {case_id}: {verdict}")
        return 1
    if any(outcome.get("checks", {}).get("backend_placement") == "ok" for outcome in results):
        decided = sum(1 for o in results if o.get("checks", {}).get("backend_placement") == "ok")
        print(f"\nbackend_placement: {decided} case(s) placed as declared")

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
    cell = tolerances.get("profiles", {}).get(arguments.profile, {})
    # CPU is the profile's own entry; anything else hangs off `backends`, which
    # is the shape the coverage test walks.
    if arguments.backend.upper() != "CPU":
        cell = cell.get("backends", {}).get(arguments.backend.upper(), {})
    stage = cell.get("stages", {}).get(arguments.stage)
    if not stage:
        print(f"\ntolerance cell {arguments.profile}/{arguments.backend}/{arguments.stage} "
              f"is not recorded in {arguments.tolerances}")
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
