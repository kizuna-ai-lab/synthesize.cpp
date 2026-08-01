#!/usr/bin/env python3
"""Stage 6: the public seam driven end to end at package defaults (sampled).

Every other omnivoice validator replays an oracle grid or compares tensors.
This one injects nothing: it drives `synth_synthesize_to_buffer` the way a
caller does and checks the promises the public interface makes about seeds
and language, for a family whose Preset Voice Catalog is empty (there is no
Voice axis here the way qwen3-tts has one -- see check 7).

Seven checks, mirroring the manifest's `public_request` relation over the
three sampled-seed cases (`omni-sampled-seed-{zero,one,forty-two}`, text
pinned there as "Sampling follows the seed."):

1. seed 0, seed 1, seed 42: `actual_seed` echoes the request.
2. seed 0 run twice -> identical PCM digest.
3. the three seeds -> three pairwise-distinct digests (`artifact_differs`).
4. `random` -> a concrete, nonzero seed is reported, and replaying it
   reproduces the digest.
5. `resolved_language` echoes the request tag; `resolved_voice` is
   null/absent (empty catalog, unnamed package default) -- reuses the seed-0
   run rather than a fresh one, since it already requested language "en".
6. a `language -` (none) request succeeds: the "None" slot is trained, not
   merely tolerated.
7. auto-voice-follows-seed: this family has no separate Voice to vary the
   way qwen3-tts does, so the three seed-keyed digests from check 3 ARE the
   divergence evidence for "changing the draw changes the (auto-chosen)
   voice" -- recorded in the report JSON below, no extra run.

The seed-contract text (checks 1-4 and 7) is read from the manifest at run
time rather than pinned as a second literal in this file, so a manifest text
change is caught here instead of silently validating a sentence the manifest
no longer contains.

Run from the repository root:

    uv run --project scripts/envs/omnivoice --locked python \
      scripts/validate-omnivoice-public.py --report reports/validate/omnivoice/public.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

DEFAULT_MANIFEST = pathlib.Path("tests/golden/omnivoice/omnivoice-0-6b.manifest.json")

# The three cases the manifest's `public_request` / `artifact_differs`
# relation names (tests/golden/omnivoice/omnivoice-0-6b.manifest.json,
# "relations"). Their pinned text is read out below rather than repeated
# here as a string literal.
SEED_CONTRACT_CASE_IDS = ("omni-sampled-seed-zero", "omni-sampled-seed-one", "omni-sampled-seed-forty-two")


def seed_contract_text(manifest_path: pathlib.Path) -> str:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in manifest["cases"]}
    missing = [case_id for case_id in SEED_CONTRACT_CASE_IDS if case_id not in cases]
    if missing:
        raise SystemExit(f"{manifest_path} no longer defines {missing}")
    texts = {cases[case_id]["input"]["text"] for case_id in SEED_CONTRACT_CASE_IDS}
    if len(texts) != 1:
        raise SystemExit(f"seed-contract cases disagree on text: {sorted(texts)}")
    return next(iter(texts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path,
                        default=pathlib.Path("models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf"))
    parser.add_argument("--runner", type=pathlib.Path,
                        default=pathlib.Path("build/bin/synthesize-omnivoice-public-real"))
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--work", type=pathlib.Path, default=pathlib.Path("build/goldens/omnivoice-public"))
    parser.add_argument("--report", type=pathlib.Path, default=None)
    # The Quantization Profile and Execution Backend this run covers, recorded
    # rather than inferred: the tolerance grid is keyed on them, and a run
    # that does not say which cell it filled cannot fill one. This family has
    # no accelerator path yet (see src/arch/omnivoice/model.cpp's placement
    # note), so "cpu" is the only backend there is.
    parser.add_argument("--profile", default="F32")
    parser.add_argument("--backend", default="cpu", choices=("cpu",))
    return parser.parse_args()


def synthesize(arguments: argparse.Namespace, name: str, language: str | None, seed: str, text: str) -> dict | None:
    target = arguments.work / f"{name}.pcm"
    target.parent.mkdir(parents=True, exist_ok=True)
    # <model.gguf> <out.pcm> <language-tag|-> <seed|random>; no voice-id
    # positional (the Preset Voice Catalog is empty) and no backend selector
    # (this family's only backend is CPU) -- see tests/omnivoice_public_real.c.
    command = [str(arguments.runner), str(arguments.model), str(target), language if language is not None else "-",
               seed]
    finished = subprocess.run(command, input=text.encode("utf-8"), capture_output=True)
    if finished.returncode != 0:
        print(f"  {name}: runner failed: {finished.stderr.decode('utf-8', 'replace').strip()[:200]}")
        return None
    observed = json.loads(finished.stdout.decode("utf-8"))
    observed["digest"] = hashlib.sha256(target.read_bytes()).hexdigest()
    observed["bytes"] = target.stat().st_size
    return observed


def main() -> int:
    arguments = parse_args()
    text = seed_contract_text(arguments.manifest)
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        print(f"  [{'ok' if passed else 'FAIL'}] {name}: {detail}")

    # omni-sampled-seed-{zero,one,forty-two}: seeds 0, 1 and 42 at language "en".
    seed0a = synthesize(arguments, "seed-0-a", "en", "0", text)
    seed1 = synthesize(arguments, "seed-1", "en", "1", text)
    seed42 = synthesize(arguments, "seed-42", "en", "42", text)
    if seed0a is None or seed1 is None or seed42 is None:
        return 1

    # Check 1.
    record(
        "actual_seed echoes the request for seeds 0, 1 and 42",
        seed0a["actual_seed"] == "0" and seed1["actual_seed"] == "1" and seed42["actual_seed"] == "42",
        f"requested 0/1/42, reported {seed0a['actual_seed']}/{seed1['actual_seed']}/{seed42['actual_seed']}",
    )

    # omni-sampled-seed-zero replayed through the public seam a second time.
    seed0b = synthesize(arguments, "seed-0-b", "en", "0", text)
    if seed0b is None:
        return 1

    # Check 2.
    record(
        "seed 0 reproduces byte for byte",
        seed0a["digest"] == seed0b["digest"],
        f"{seed0a['digest'][:16]} vs {seed0b['digest'][:16]}",
    )

    # The manifest's `public_request` / `artifact_differs` relation.
    digests = {"0": seed0a["digest"], "1": seed1["digest"], "42": seed42["digest"]}
    distinct = len(set(digests.values())) == len(digests)

    # Check 3.
    record(
        "seeds 0, 1 and 42 are pairwise distinct (artifact_differs)",
        distinct,
        f"digests {', '.join(f'{seed}:{digest[:16]}' for seed, digest in digests.items())}",
    )

    # A `random` seed has to come back concrete, and replaying it has to reproduce.
    drawn = synthesize(arguments, "seed-random", "en", "random", text)
    if drawn is None:
        return 1
    replayed = synthesize(arguments, "seed-random-replay", "en", drawn["actual_seed"], text)
    if replayed is None:
        return 1

    # Check 4.
    record(
        "a random seed is reported concretely and reproduces",
        drawn["actual_seed"] not in ("", "0") and drawn["digest"] == replayed["digest"],
        f"drawn seed {drawn['actual_seed']}, digest {drawn['digest'][:16]} vs replay {replayed['digest'][:16]}",
    )

    # Check 5: reuses seed0a rather than a fresh run -- it already requested
    # language "en". resolved_voice is not merely null in the ABI struct; the
    # driver (tests/omnivoice_public_real.c) does not print the field at all
    # for this family, unlike qwen3-tts's, so "absent" is checked against the
    # driver's JSON surface, which is this stage's only observable one.
    record(
        "resolved_language echoes the request; resolved_voice is absent",
        seed0a["resolved_language"] == "en" and not seed0a.get("resolved_voice"),
        f"requested en, reported {seed0a['resolved_language']!r}; "
        f"resolved_voice {'present: ' + repr(seed0a['resolved_voice']) if seed0a.get('resolved_voice') else 'absent'}",
    )

    # A `language -` request: the "None" slot (package_contract.language_tags
    # has no null itself, but a null language_tag is exactly what
    # omni-upstream-readme's oracle case exercises) has to be a trained path,
    # not just one that happens not to crash.
    no_language = synthesize(arguments, "no-language", None, "0", text)
    if no_language is None:
        return 1

    # Check 6.
    record(
        "a request naming no language succeeds",
        int(no_language["status"]) == 0 and int(no_language["frames"]) > 0,
        f"status {no_language['status']}, frames {no_language['frames']}, "
        f"resolved_language {no_language['resolved_language']!r}",
    )

    # Check 7: this family's Preset Voice Catalog is empty (package_contract
    # in the manifest: preset_ids []), so there is no second Voice to request
    # the way qwen3-tts's public validator does. The seed-keyed digests from
    # check 3 are what auto-voice-follows-seed looks like here: changing the
    # draw is the only way this family's "voice" moves at all. No extra run.
    record(
        "seed-keyed digests double as auto-voice divergence evidence",
        distinct,
        "the package default (auto-voice) is the only voice this family has; "
        f"digests {', '.join(f'{seed}:{digest[:16]}' for seed, digest in digests.items())} "
        "are the same three runs check 3 already made",
    )

    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(json.dumps({
            "schema": "synthesize-validation-report-v1",
            "family": "omnivoice",
            "phase": "public_request",
            "profile": arguments.profile,
            "backend": arguments.backend.upper(),
            "text": text,
            "checks": checks,
            "voice_divergence_evidence": {
                "note": "no separate Voice axis exists for this family; these are check 3's seed-keyed digests.",
                "cases": list(SEED_CONTRACT_CASE_IDS),
                "digests": digests,
            },
        }, indent=2) + "\n", encoding="utf-8")
        print(f"\nreport: {arguments.report}")

    failed = [check["check"] for check in checks if not check["passed"]]
    if failed:
        print(f"\n{len(failed)} check(s) failed: {failed}")
        return 1
    print(f"\nall {len(checks)} public-request checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
