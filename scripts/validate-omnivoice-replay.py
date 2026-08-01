#!/usr/bin/env python3
"""Compare the omnivoice port's replay outputs against the oracle dumps.

Deep generator probes gate on cosine similarity, not max-abs; the waveform
gates on both, because a listener hears the waveform and a cosine over it
would hide a constant offset. The token grid is compared EXACTLY: greedy
decoding makes no RNG call, docs/porting/families/omnivoice.md defines
structural_exactness for this family as equality of the 8 x T grid, and no
tolerance file entry exists or ever will for it.

A case may pin more than one admissible grid. `oracle.alternate_grids` names
committed, digest-pinned grids the reference itself produced under a different
configuration; the port passes by equalling ANY of them, byte for byte. That
widens the target SET by enumeration and never the comparison: there is still
no tolerance anywhere on this path, and a grid enters the set only by being
committed with its provenance recorded.

Two waveforms are compared, and they are not the same claim. `audio.pcm`
replays the ORACLE's grid through the port's codec, which isolates the codec
from the decode loop; `audio.pcm_freerun` is what run_synthesis returned for
the grid the port itself chose, which is the only artifact that proves the
codec is wired into the synthesis path at all. The free-run waveform is
comparable to the oracle's only when the port matched the PRIMARY grid. When it
matched an alternate, no oracle waveform for that grid exists, so the case is
exempt from oracle parity on this channel and gets a decode-determinism
comparison instead: the port's own decode of the same committed alternate must
equal it exactly. That is a weaker claim, deliberately, and it is labelled as
such in the report rather than folded into the parity numbers.

Without --check this script measures; with --check it gates against
tests/tolerances/omnivoice.json (profiles.<PROFILE>.stages.<STAGE>), refusing
to run when the cell is absent -- a threshold the suite writes for itself
proves nothing.

The two clone cases additionally carry an encode-reference channel (Task 11):
the runner's `--encode-reference` resamples and runs the HuBERT semantic
branch plus the codec's own SemanticEncoder over `ref/pcm_24k.f32`, and this
script compares the result against the oracle's own `ref/semantic_mean.f32`
probe. That comparison is REPORTED here (max_abs and cosine, printed and
carried into --report's JSON) but not folded into `measurements`/`worst`, so
it never participates in --check's gate: thresholds for it are Task 13's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess

import numpy as np

PROBE_LAYERS = (0, 7, 14, 21, 27)
VOLUME_BY_BRANCH = {"peak_normalise_to_0.5": "peak", "none": "none"}

# The margin below which a candidate golden case is a coin flip rather than a
# demonstration; docs/porting/families/omnivoice.md carries the derivation.
# Reported, never enforced here: it screens cases being CHOSEN, and the two
# cases already below it are retained under a recorded ruling. A validator that
# failed on it would be re-litigating that ruling on every run.
MARGIN_SCREEN = 1e-4

# This script used to carry a table of the refusals Plan 2's unbuilt stages
# printed, so that `--require all` before the codec landed said "stage not built
# yet" rather than a generic `runner-failed` on all 20 cases. Every stage the
# three --require levels name is now built -- Task 10 took the greedy loop's row
# and Task 12 the codec's -- so the table would be empty and the branch reading
# it unreachable. It is deleted rather than left empty: a runner failure under
# any --require level is now a real failure, and there is no third answer.
# (`git log -S "NOT_BUILT_MARKERS"` has the mechanism if a later plan's unbuilt
# stage wants it back.)


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
                             "all: + the replayed-grid waveform and the free-run path's own")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--margin-report", action="store_true",
                        help="measure how narrowly each greedy case's decisions were made; "
                             "the screen for new golden cases (see the family doc)")
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


def peak_normalise(audio: np.ndarray) -> np.ndarray:
    """The no-reference volume branch, in float32, as the port applies it.

    Needed because Plan 2's run_synthesis applies this branch unconditionally
    (the other two arms key off a reference RMS that only Plan 3's cloning path
    can supply), while the oracle applied it only to its no-reference cases. So
    the free-run waveform's expected value is the oracle's waveform put through
    the same branch: for a case the oracle already normalised this is the
    identity to the bit -- x / 0.5 * 0.5 is exact in binary floating point --
    and for the clone cases it is the one transform that makes the two
    comparable at all.
    """
    peak = np.float32(np.abs(audio).max(initial=np.float32(0.0)))
    if peak <= np.float32(1e-6):
        return audio
    return audio / peak * np.float32(0.5)


def is_greedy(case: dict) -> bool:
    return float(case["oracle"]["parameters"]["position_temperature"]) == 0.0


def admissible_grids(manifest_path: pathlib.Path, case: dict,
                     oracle: pathlib.Path) -> list[tuple[str, np.ndarray]]:
    """Every grid this case's port output may equal, the dumped one first.

    An alternate is read from the committed file and checked against the digest
    the manifest pins before a single element is compared. Skipping that check
    would let an edited or truncated witness silently widen the target -- the
    one failure mode a set of admissible answers has that a single answer does
    not.
    """
    grids = [("primary", read_i32(oracle / "codes/grid.i32"))]
    for entry in case["oracle"].get("alternate_grids", []):
        path = manifest_path.parent / entry["file"]
        if not path.is_file():
            raise SystemExit(f"{case['id']}: alternate grid {path} is not committed")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != entry["sha256"]:
            raise SystemExit(f"{case['id']}: alternate grid {path} has sha256 {digest}, "
                             f"but the manifest pins {entry['sha256']}")
        grids.append((entry["file"], np.frombuffer(payload, dtype=np.int32)))
    return grids


def compare_grid(admissible: list[tuple[str, np.ndarray]], actual: np.ndarray) -> dict:
    """Exact equality against the admissible set, naming which one matched.

    `elements` reports the size of the grid `mismatches` was actually counted
    against -- the CLOSEST admissible grid, primary or an alternate -- rather
    than always the primary's. A primary-sized report is only correct when the
    closest grid happens to be the primary; a same-shaped suite never
    exercises the difference, but a mismatched-size alternate would otherwise
    print an element count that was never compared.
    """
    matched, closest, fewest, closest_elements = None, None, None, None
    for name, expected in admissible:
        count = (int((expected != actual).sum()) if expected.shape == actual.shape
                 else int(expected.size))
        if fewest is None or count < fewest:
            fewest, closest, closest_elements = count, name, int(expected.size)
        if count == 0:
            matched = name
            break
    return {"elements": closest_elements, "mismatches": fewest,
            "exact": matched is not None, "matched": matched, "closest": closest,
            "admissible": [name for name, _ in admissible]}


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

    admissible = admissible_grids(arguments.manifest, case, oracle) if run_greedy else []
    work = arguments.work / case_id
    work.mkdir(parents=True, exist_ok=True)
    command = [str(arguments.runner), str(arguments.model), str(oracle), str(work),
               str(case["oracle"]["parameters"]["num_step"]),
               "1" if run_greedy else "0", "1" if decode else "0",
               VOLUME_BY_BRANCH[branch]] + [str(layer) for layer in PROBE_LAYERS]
    if arguments.margin_report:
        command.append("--margin-report")
    # Asked for whenever the case pins one, not only when the port turns out to
    # need it: which grid the port matches is not known until it has run, and a
    # second full greedy run to find out would cost minutes to save one codec
    # pass. Unused when the port matches the primary.
    alternates = case["oracle"].get("alternate_grids", [])
    if decode and run_greedy and alternates:
        if len(alternates) > 1:
            raise SystemExit(f"{case_id}: {len(alternates)} alternate grids, but the runner "
                             f"takes one --alt-grid; the free-run waveform channel needs widening")
        command += ["--alt-grid", str(arguments.manifest.parent / alternates[0]["file"])]
    # The encode-reference channel: only the two clone cases dump the
    # pcm_24k/semantic_mean pair this needs, so its presence in the manifest's
    # own artifact list -- not --require -- decides whether this case carries
    # it. Independent of the greedy grid/decode machinery above.
    has_encode_reference = "ref.pcm_24k" in names and "ref.semantic_mean" in names
    if has_encode_reference:
        command += ["--encode-reference", str(oracle / "ref/pcm_24k.f32")]
    finished = subprocess.run(command, capture_output=True)
    if finished.returncode != 0:
        stderr = finished.stderr.decode("utf-8", "replace")
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
        grid = compare_grid(admissible, read_i32(work / "grid.i32"))
    finite = None
    freerun = None
    if decode:
        pcm = read_f32(work / "pcm.f32")
        measurements["audio.pcm"] = compare(read_f32(oracle / "audio/pcm.f32"), pcm)
        finite = bool(np.isfinite(pcm).all())
    if decode and run_greedy:
        # Read the count the runner reported, not just the file: the work
        # directory is reused across runs, so a file left by an earlier
        # invocation would otherwise be compared as if this one had written it.
        # Absence goes down the same road as a size disagreement -- a missing
        # waveform is a runner that did not produce one, which is a result to
        # report, not a traceback to raise.
        waveform = work / "pcm_freerun.f32"
        produced = read_f32(waveform) if waveform.is_file() else np.empty(0, dtype=np.float32)
        if int(stats.get("freerun_samples", 0)) != produced.size:
            return {"case": case_id, "status": "stale-freerun-waveform",
                    "stderr": f"the runner reported {stats.get('freerun_samples')} free-run samples "
                              f"but pcm_freerun.f32 holds {produced.size}"
                              f"{'' if waveform.is_file() else ' (the file is not there at all)'}"}
        finite = finite and bool(np.isfinite(produced).all())
        if grid["matched"] == "primary":
            # The port chose the oracle's grid, so its own waveform is owed the
            # oracle's, under the branch run_synthesis applied.
            freerun = {"mode": "oracle-parity", "grid": "primary"}
            measurements["audio.pcm_freerun"] = compare(
                peak_normalise(read_f32(oracle / "audio/pcm.f32")), produced)
        elif grid["matched"] is not None:
            # An alternate: no oracle waveform exists for the grid the port
            # produced, and comparing against the primary's would report a
            # difference the case does not claim. What is still owed is that
            # decoding that same committed alternate reproduces it.
            reported = int(stats.get("alternate_samples", 0))
            reference = read_f32(work / "pcm_alt.f32") if reported and (work / "pcm_alt.f32").is_file() else None
            # Same stale-file rule, and here it also catches the case where the
            # port matched an alternate that no --alt-grid was passed for: then
            # there is nothing to compare and saying so beats reporting a pass.
            comparable = (reference is not None and reference.size == reported
                          and reference.shape == produced.shape)
            freerun = {
                "mode": "decode-determinism", "grid": grid["matched"],
                "max_abs": float(np.abs(reference - produced).max(initial=0.0)) if comparable else None,
                "identical": comparable and bool(np.array_equal(reference, produced)),
            }
        else:
            # The grid matched nothing, which is already a structural failure;
            # the waveform it decoded to is a consequence, not a second finding.
            freerun = {"mode": "not-compared", "grid": None,
                       "reason": "the free-run grid matched no admissible grid"}

    # Kept OUT of `measurements`/`worst` on purpose: Task 11 reports this
    # channel, it does not gate it (see this script's own top comment and
    # scripts/dump_reference_omnivoice_pytorch.py's semantic_mean note --
    # captured before the [::2] downsample, which is exactly what
    # build_semantic_branch's `semantic_mean` out-parameter is too).
    encode_reference = None
    if has_encode_reference:
        produced_path = work / "semantic_mean.f32"
        if int(stats.get("encode_reference_elements", 0)) == 0 or not produced_path.is_file():
            encode_reference = {"status": "missing"}
        else:
            encode_reference = compare(read_f32(oracle / "ref/semantic_mean.f32"), read_f32(produced_path))

    return {"case": case_id, "status": "ok", "greedy": greedy, "stats": stats,
            "measurements": measurements, "grid": grid, "finite_pcm": finite,
            "freerun": freerun, "margin": stats.get("margin"), "encode_reference": encode_reference}


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
    worst: dict[str, dict] = {}
    for case in cases:
        result = run_case(arguments, case, oracle_root)
        if result is None:
            continue
        results.append(result)
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
                  f"of {result['grid']['elements']} positions "
                  f"(closest of {len(result['grid']['admissible'])} admissible: "
                  f"{result['grid']['closest']})")
        # Which grid a case matched is a fact about the contract, not a detail:
        # a case that starts passing only via its alternate has changed what it
        # demonstrates, and that must be visible without opening the report.
        elif result["grid"] is not None and len(result["grid"]["admissible"]) > 1:
            print(f"{result['case']}: token grid exact against {result['grid']['matched']} "
                  f"of {len(result['grid']['admissible'])} admissible grids")
        if result["margin"] is not None:
            margin = result["margin"]
            print(f"{result['case']}: min {margin['kind']} margin {margin['value']:.6g} "
                  f"at step {margin['step']}, codebook {margin['codebook']}, "
                  f"frame {margin['frame']}")
        if result["finite_pcm"] is False:
            failures += 1
            print(f"{result['case']}: non-finite PCM")
        # The exempt half of the free-run waveform channel. Its claim is
        # exactness, so it is counted with the structural failures rather than
        # measured into the worst-table, and it is printed either way -- a case
        # quietly dropping out of oracle parity is exactly what needs saying.
        freerun = result.get("freerun")
        if freerun is not None and freerun["mode"] == "decode-determinism":
            if freerun["identical"]:
                print(f"{result['case']}: free-run waveform exempt from oracle parity "
                      f"(port matched {freerun['grid']}); identical to this port's own "
                      f"decode of that grid")
            elif freerun["max_abs"] is None:
                structural_failures += 1
                print(f"{result['case']}: matched {freerun['grid']}, but no decode of that grid "
                      f"was produced to compare its free-run waveform against")
            else:
                structural_failures += 1
                print(f"{result['case']}: free-run waveform differs from this port's own "
                      f"decode of {freerun['grid']} (max_abs {freerun['max_abs']:.6g})")
        # Plan 2 is CPU-only: a node on an accelerator is a placement bug.
        placement = result["stats"]["placement"]
        if placement["generator"][1] != 0 or placement["codec"][1] != 0:
            failures += 1
            print(f"{result['case']}: nodes left the CPU: {placement}")
        # REPORTED, not gated (see this script's top comment): this never
        # touches `failures`/`structural_failures`, regardless of what it
        # prints -- Task 13 is what turns this into a real gate.
        encode_reference = result.get("encode_reference")
        if encode_reference is not None:
            if "shape_mismatch" in encode_reference:
                print(f"{result['case']}: ref.semantic_mean shape mismatch "
                      f"{encode_reference['shape_mismatch']} (reported, not gated -- Task 13)")
            elif encode_reference.get("status") == "missing":
                print(f"{result['case']}: ref.semantic_mean requested but semantic_mean.f32 "
                      f"is missing or empty (reported, not gated -- Task 13)")
            else:
                print(f"{result['case']}: ref.semantic_mean max_abs {encode_reference['max_abs']:.6g} "
                      f"cosine {encode_reference['cosine']:.8f} (reported, not gated -- Task 13)")

    compared = [result for result in results if result["status"] == "ok"]
    print(f"\n{'probe':32} {'max_abs':>12} {'min_cosine':>12}")
    for name in sorted(worst):
        print(f"{name:32} {worst[name]['max_abs']:12.6g} {worst[name]['min_cosine']:12.8f}")
    exact = [r for r in compared if r.get("grid") is not None]
    if exact:
        good = sum(1 for r in exact if r["grid"]["exact"])
        print(f"token grids exact: {good}/{len(exact)}")
    # The free-run waveform channel's own headline: how many cases carried it as
    # oracle parity, and how many only as decode determinism. The split is the
    # number a reader has to see, because the second kind is the weaker claim.
    walked = [r for r in compared if r.get("freerun") is not None]
    if walked:
        parity = sum(1 for r in walked if r["freerun"]["mode"] == "oracle-parity")
        determinism = sum(1 for r in walked if r["freerun"]["mode"] == "decode-determinism")
        print(f"free-run waveforms: {parity} against the oracle, "
              f"{determinism} exempt (decode determinism), {len(walked) - parity - determinism} not compared")
    measured = [r for r in compared if r.get("margin") is not None]
    if measured:
        narrowest = min(measured, key=lambda r: r["margin"]["value"])
        below = sum(1 for r in measured if r["margin"]["value"] < MARGIN_SCREEN)
        print(f"narrowest margin: {narrowest['margin']['value']:.6g} "
              f"({narrowest['margin']['kind']}) in {narrowest['case']}; "
              f"{below} of {len(measured)} case(s) under the {MARGIN_SCREEN:g} screen")

    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(json.dumps({
            "schema": "synthesize-validation-report-v1", "family": "omnivoice",
            "variant": manifest["variant"], "suite_version": manifest["suite_version"],
            "phase": "oracle_replay", "profile": arguments.profile,
            "backend": arguments.backend.upper(), "require": arguments.require,
            "margin_screen": MARGIN_SCREEN if arguments.margin_report else None,
            "cases": results, "worst": worst,
        }, indent=2) + "\n", encoding="utf-8")

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
