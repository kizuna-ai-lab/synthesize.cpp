#!/usr/bin/env python3
"""Compare the omnivoice port's replay outputs against the oracle dumps.

Deep generator probes gate on cosine similarity, not max-abs; the waveform
gates on both, because a listener hears the waveform and a cosine over it
would hide a constant offset. The token grid is compared EXACTLY: greedy
decoding makes no RNG call, docs/porting/families/omnivoice.md defines
structural_exactness for this family as equality of the 8 x T grid, and no
tolerance file entry exists or ever will for it.

Without --check this script measures; with --check it gates against
tests/tolerances/omnivoice.json (profiles.<PROFILE>.stages.<STAGE>), refusing
to run when the cell is absent -- a threshold the suite writes for itself
proves nothing.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import numpy as np

PROBE_LAYERS = (0, 7, 14, 21, 27)
VOLUME_BY_BRANCH = {"peak_normalise_to_0.5": "peak", "none": "none"}

# The refusals Plan 2's unbuilt stages print, and the stage each one names.
# These strings are the contract between src/arch/omnivoice/model.cpp's
# not-built-yet message and this script: without it `--require all` before
# Task 12 lands reports a generic `runner-failed` on all 20 cases, which is
# indistinguishable from a parity regression. A slice deletes its message and
# its row here together -- Task 10 took the greedy loop's, so only the codec's
# remains, and `--require grid` now has no not-built stage left to name.
NOT_BUILT_MARKERS = (("omnivoice: codec decode is slice 6 and not built yet", "decode_codes"),)


def unbuilt_stage(stderr: str) -> str | None:
    """The stage a runner failure blames on an unbuilt slice, or None."""
    for marker, stage in NOT_BUILT_MARKERS:
        if marker in stderr:
            return stage
    return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--manifest", type=pathlib.Path,
                        default=pathlib.Path("tests/golden/omnivoice/omnivoice-0-6b.manifest.json"))
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--runner", type=pathlib.Path,
                        default=pathlib.Path("build/bin/synthesize-omnivoice-replay-real"))
    parser.add_argument("--work", type=pathlib.Path,
                        default=pathlib.Path("build/goldens/omnivoice-replay"))
    parser.add_argument("--report", type=pathlib.Path, default=None)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--require", choices=("probes", "grid", "all"), default="all",
                        help="probes: step-0 forward only; grid: + the greedy free-run; "
                             "all: + the replayed-grid waveform")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--tolerances", type=pathlib.Path,
                        default=pathlib.Path("tests/tolerances/omnivoice.json"))
    parser.add_argument("--profile", default="F32")
    parser.add_argument("--backend", default="CPU")
    parser.add_argument("--stage", default="replay")
    return parser.parse_args(argv)


def read_f32(path: pathlib.Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype=np.float32)


def read_i32(path: pathlib.Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype=np.int32)


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


def is_greedy(case: dict) -> bool:
    return float(case["oracle"]["parameters"]["position_temperature"]) == 0.0


def run_case(arguments, case: dict, oracle_root: pathlib.Path) -> dict | None:
    case_id = case["id"]
    oracle = oracle_root / case_id
    if not (oracle / "codes/grid.i32").is_file():
        print(f"{case_id}: no oracle artifacts, skipped")
        return None

    # The probe set is a contract between the manifest, the dumper and this
    # script; drift dumps a file nothing compares or compares one nothing dumps.
    names = {artifact["name"] for artifact in case["expected"]["artifacts"]}
    wanted = {f"generator.hidden_l{layer}" for layer in PROBE_LAYERS}
    probed = {name for name in names if name.startswith("generator.hidden_l")}
    if probed != wanted:
        raise SystemExit(f"{case_id}: manifest probes {sorted(probed)} != validator's {sorted(wanted)}")

    greedy = is_greedy(case)
    run_greedy = greedy and arguments.require in ("grid", "all")
    decode = arguments.require == "all"
    branch = json.loads((oracle / "result.json").read_text(encoding="utf-8"))["volume_branch"]
    if branch not in VOLUME_BY_BRANCH:
        raise SystemExit(f"{case_id}: unsupported volume branch {branch!r}")

    work = arguments.work / case_id
    work.mkdir(parents=True, exist_ok=True)
    command = [str(arguments.runner), str(arguments.model), str(oracle), str(work),
               str(case["oracle"]["parameters"]["num_step"]),
               "1" if run_greedy else "0", "1" if decode else "0",
               VOLUME_BY_BRANCH[branch]] + [str(layer) for layer in PROBE_LAYERS]
    finished = subprocess.run(command, capture_output=True)
    if finished.returncode != 0:
        stderr = finished.stderr.decode("utf-8", "replace")
        # A stage this plan has not built yet is not a parity regression, and
        # saying "runner-failed" for it would read as one.
        stage = unbuilt_stage(stderr)
        if stage is not None:
            return {"case": case_id, "status": "stage-not-built", "stage": stage}
        return {"case": case_id, "status": "runner-failed", "stderr": stderr[:400]}
    stats = None
    for line in reversed(finished.stdout.decode("utf-8", "replace").splitlines()):
        if line.startswith("{"):
            stats = json.loads(line)
            break
    if stats is None:
        return {"case": case_id, "status": "runner-said-nothing"}

    measurements = {}
    for layer in PROBE_LAYERS:
        measurements[f"generator.hidden_l{layer}"] = compare(
            read_f32(oracle / f"generator/hidden_l{layer}.f32"), read_f32(work / f"hidden_l{layer}.f32"))
    measurements["generator.final"] = compare(read_f32(oracle / "generator/final.f32"),
                                              read_f32(work / "final.f32"))
    measurements["generator.logits_step0"] = compare(read_f32(oracle / "generator/logits_step0.f32"),
                                                     read_f32(work / "logits_step0.f32"))
    grid = None
    if run_greedy:
        expected = read_i32(oracle / "codes/grid.i32")
        actual = read_i32(work / "grid.i32")
        mismatches = (int((expected != actual).sum()) if expected.shape == actual.shape
                      else int(expected.size))
        grid = {"elements": int(expected.size), "mismatches": mismatches, "exact": mismatches == 0}
    finite = None
    if decode:
        pcm = read_f32(work / "pcm.f32")
        measurements["audio.pcm"] = compare(read_f32(oracle / "audio/pcm.f32"), pcm)
        finite = bool(np.isfinite(pcm).all())

    return {"case": case_id, "status": "ok", "greedy": greedy, "stats": stats,
            "measurements": measurements, "grid": grid, "finite_pcm": finite}


def main(argv=None) -> int:
    arguments = parse_args(argv)
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    oracle_root = pathlib.Path(manifest["case_artifact_root"])
    known = {case["id"] for case in manifest["cases"]}
    # A typo in --cases used to select nothing and then pass, which is the same
    # false green as running no cases at all.
    if arguments.cases is not None:
        unknown = sorted(set(arguments.cases) - known)
        if unknown:
            print(f"--cases names {unknown}, which {arguments.manifest} does not define")
            return 1
    cases = [case for case in manifest["cases"]
             if arguments.cases is None or case["id"] in arguments.cases]

    results, failures, structural_failures = [], 0, 0
    not_built: dict[str, int] = {}
    worst: dict[str, dict] = {}
    for case in cases:
        result = run_case(arguments, case, oracle_root)
        if result is None:
            continue
        results.append(result)
        if result["status"] == "stage-not-built":
            not_built[result["stage"]] = not_built.get(result["stage"], 0) + 1
            print(f"{result['case']}: stage not built yet: {result['stage']}")
            continue
        if result["status"] != "ok":
            failures += 1
            print(f"{result['case']}: {result['status']}")
            continue
        for name, measurement in result["measurements"].items():
            if "shape_mismatch" in measurement:
                failures += 1
                print(f"{result['case']}: {name} shape mismatch {measurement['shape_mismatch']}")
                continue
            slot = worst.setdefault(name, {"max_abs": 0.0, "min_cosine": 1.0})
            slot["max_abs"] = max(slot["max_abs"], measurement["max_abs"])
            slot["min_cosine"] = min(slot["min_cosine"], measurement["cosine"])
        if result["grid"] is not None and not result["grid"]["exact"]:
            structural_failures += 1
            print(f"{result['case']}: token grid differs at {result['grid']['mismatches']} "
                  f"of {result['grid']['elements']} positions")
        if result["finite_pcm"] is False:
            failures += 1
            print(f"{result['case']}: non-finite PCM")
        # Plan 2 is CPU-only: a node on an accelerator is a placement bug.
        placement = result["stats"]["placement"]
        if placement["generator"][1] != 0 or placement["codec"][1] != 0:
            failures += 1
            print(f"{result['case']}: nodes left the CPU: {placement}")

    compared = [result for result in results if result["status"] == "ok"]
    print(f"\n{'probe':32} {'max_abs':>12} {'min_cosine':>12}")
    for name in sorted(worst):
        print(f"{name:32} {worst[name]['max_abs']:12.6g} {worst[name]['min_cosine']:12.8f}")
    exact = [r for r in compared if r.get("grid") is not None]
    if exact:
        good = sum(1 for r in exact if r["grid"]["exact"])
        print(f"token grids exact: {good}/{len(exact)}")

    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(json.dumps({
            "schema": "synthesize-validation-report-v1", "family": "omnivoice",
            "variant": manifest["variant"], "suite_version": manifest["suite_version"],
            "phase": "oracle_replay", "profile": arguments.profile,
            "backend": arguments.backend.upper(), "require": arguments.require,
            "cases": results, "worst": worst,
        }, indent=2) + "\n", encoding="utf-8")

    if not_built:
        named = ", ".join(f"{stage} ({count} case(s))" for stage, count in sorted(not_built.items()))
        print(f"\nstage not built yet: {named} -- this plan has not reached it, "
              f"which is not a parity result. Re-run with --require probes for what slice 4 gates.")
        return 1
    # Comparing nothing is not passing. Both shapes of "nothing" reach here: a
    # --cases filter that selected no case, and a case_artifact_root whose
    # payload was never materialized, in which case every case was skipped.
    if not compared:
        print(f"\nno case produced a comparison: {len(cases)} selected, "
              f"{len(cases) - len(results)} skipped for missing oracle artifacts under {oracle_root}")
        return 1
    if failures or structural_failures:
        return 1
    if not arguments.check:
        return 0

    # A tolerance is a reviewed number in a committed file. Refusing to run
    # rather than inventing one is the point.
    tolerances = json.loads(arguments.tolerances.read_text(encoding="utf-8"))
    cell = tolerances.get("profiles", {}).get(arguments.profile, {})
    if arguments.backend.upper() != "CPU":
        cell = cell.get("backends", {}).get(arguments.backend.upper(), {})
    stage = cell.get("stages", {}).get(arguments.stage)
    if not stage:
        print(f"\ntolerance cell {arguments.profile}/{arguments.backend}/{arguments.stage} "
              f"is not recorded in {arguments.tolerances}")
        return 1
    probes = stage.get("probes")
    if not probes:
        print(f"\ntolerance cell {arguments.profile}/{arguments.backend}/{arguments.stage} "
              f"records no probes in {arguments.tolerances}")
        return 1
    breaches = 0
    for name, observed in sorted(worst.items()):
        limits = probes.get(name)
        if limits is None:
            print(f"{name}: no tolerance recorded")
            breaches += 1
            continue
        if "max_abs" in limits and observed["max_abs"] > limits["max_abs"]:
            print(f"{name}: max_abs {observed['max_abs']:.6g} breaches {limits['max_abs']:.6g}")
            breaches += 1
        if "min_cosine" in limits and observed["min_cosine"] < limits["min_cosine"]:
            print(f"{name}: cosine {observed['min_cosine']:.8f} breaches {limits['min_cosine']:.8f}")
            breaches += 1
    if breaches:
        return 1
    print(f"\nall probes within the {arguments.profile}/{arguments.backend}/{arguments.stage} tolerances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
