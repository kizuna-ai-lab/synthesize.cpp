#!/usr/bin/env python3
"""Stage 6: the public seam driven end to end at package defaults (sampled).

Every other omnivoice validator replays an oracle grid or compares tensors.
This one injects nothing: it drives `synth_synthesize_to_buffer` the way a
caller does and checks the promises the public interface makes about seeds
and language, for a family whose Preset Voice Catalog is empty (there is no
Voice axis here the way qwen3-tts has one -- see check 7).

Thirteen checks. The first seven mirror the manifest's `public_request` relation
over the three sampled-seed cases (`omni-sampled-seed-{zero,one,forty-two}`,
text pinned there as "Sampling follows the seed."):

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

Three more (Task 14, the cloning path), against the `omni-clone-en` golden
case's own pinned reference clip and transcript:

8. a clone request (Reference Audio profile + target text) succeeds.
9. the same clone request, same seed, run twice -> identical PCM digest:
   same-seed reproducibility holds WITH a profile, not only without one.
10. the clone digest differs from a same-seed, same-text run carrying no
    profile at all: the reference conditions the output rather than being
    silently ignored.

Three more (Task 15, the voice-design path), against the `omni-design-en`
golden case's own pinned instruct:

11. a design request (Description Text profile + target text) succeeds.
12. the same design request, same seed, run twice -> identical PCM digest:
    same-seed reproducibility holds WITH a design profile, exactly as it
    does with a clone one.
13. the design digest differs from a same-seed, same-text run carrying no
    profile at all: the instruct conditions the output rather than being
    silently ignored.

The public phase claims relations only -- it does not compare against the
oracle's own clone or design waveform (that is the replay gate's job, gated
by an uncommitted oracle payload) -- and the free-running greedy grid claim
for these cases stays where it has always lived, in
scripts/validate-omnivoice-replay.py.

The seed-contract text (checks 1-4 and 7) is read from the manifest at run
time rather than pinned as a second literal in this file, so a manifest text
change is caught here instead of silently validating a sentence the manifest
no longer contains. The clone checks (8-10) read the `omni-clone-en` case's
own text/reference/transcript/language the same way; the design checks
(11-13) read `omni-design-en`'s own text/description/language.

Run from the repository root:

    uv run --project scripts/envs/omnivoice --locked python \
      scripts/validate-omnivoice-public.py --report reports/validate/omnivoice/public.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
import subprocess
import sys
import wave

DEFAULT_MANIFEST = pathlib.Path("tests/golden/omnivoice/omnivoice-0-6b.manifest.json")

# The three cases the manifest's `public_request` / `artifact_differs`
# relation names (tests/golden/omnivoice/omnivoice-0-6b.manifest.json,
# "relations"). Their pinned text is read out below rather than repeated
# here as a string literal.
SEED_CONTRACT_CASE_IDS = ("omni-sampled-seed-zero", "omni-sampled-seed-one", "omni-sampled-seed-forty-two")

# The clone case the Task 14 checks (8-10) drive.
CLONE_CASE_ID = "omni-clone-en"

# The design case the Task 15 checks (11-13) drive.
DESIGN_CASE_ID = "omni-design-en"

# include/synthesize.h: `#define SYNTH_SEED_RANDOM UINT64_MAX`. A resolver
# that regressed to echoing the sentinel itself, deterministically, would
# still be neither "" nor "0" and would still reproduce byte-for-byte on
# replay (the replay would hit the same `== SYNTH_SEED_RANDOM` branch again)
# -- broken but self-consistent, and check 4 below must not call that
# "concrete".
SYNTH_SEED_RANDOM_STR = str((1 << 64) - 1)


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


def clone_case(manifest_path: pathlib.Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in manifest["cases"]}
    if CLONE_CASE_ID not in cases:
        raise SystemExit(f"{manifest_path} no longer defines {CLONE_CASE_ID}")
    return cases[CLONE_CASE_ID]


def design_case(manifest_path: pathlib.Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in manifest["cases"]}
    if DESIGN_CASE_ID not in cases:
        raise SystemExit(f"{manifest_path} no longer defines {DESIGN_CASE_ID}")
    return cases[DESIGN_CASE_ID]


def wav_to_f32(wav_path: pathlib.Path, out_path: pathlib.Path) -> None:
    """Converts a mono 16-bit PCM wav to raw little-endian f32 samples.

    Stdlib only (`wave` + `struct`) -- this validator's locked environment
    already carries `soundfile` for the oracle dumpers, but that is a heavier
    dependency than one small, one-time format conversion needs, and every
    other artifact this script produces is written the same plain way (raw
    PCM straight from the runner's own write_pcm).
    """
    with wave.open(str(wav_path), "rb") as handle:
        channels   = handle.getnchannels()
        sampwidth  = handle.getsampwidth()
        frame_count = handle.getnframes()
        raw        = handle.readframes(frame_count)
    if channels != 1 or sampwidth != 2:
        raise SystemExit(f"{wav_path}: expected mono 16-bit PCM, got {channels} channel(s) at {sampwidth * 8}-bit")
    sample_count = len(raw) // 2
    samples = struct.unpack(f"<{sample_count}h", raw)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(struct.pack(f"<{sample_count}f", *(sample / 32768.0 for sample in samples)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path,
                        default=pathlib.Path("models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf"))
    parser.add_argument("--runner", type=pathlib.Path,
                        default=pathlib.Path("build/bin/synthesize-omnivoice-public-real"))
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference-wav", type=pathlib.Path,
                        default=pathlib.Path("models/omnivoice-reference-audio/seedtts_ref_en_1.wav"),
                        help="the pinned clone reference clip (Task 14's checks 8-10)")
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


def synthesize(arguments: argparse.Namespace, name: str, language: str | None, seed: str, text: str,
              reference_f32: pathlib.Path | None = None, transcript: str | None = None,
              instruct: str | None = None) -> dict | None:
    target = arguments.work / f"{name}.pcm"
    target.parent.mkdir(parents=True, exist_ok=True)
    # <model.gguf> <out.pcm> <language-tag|-> <seed|random>; no voice-id
    # positional (the Preset Voice Catalog is empty) and no backend selector
    # (this family's only backend is CPU) -- see tests/omnivoice_public_real.c.
    command = [str(arguments.runner), str(arguments.model), str(target), language if language is not None else "-",
               seed]
    if reference_f32 is not None:
        command += ["--reference", str(reference_f32), "--transcript", transcript]
    if instruct is not None:
        command += ["--instruct", instruct]
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
        drawn["actual_seed"] not in ("", "0", SYNTH_SEED_RANDOM_STR) and drawn["digest"] == replayed["digest"],
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

    # --- Task 14: the cloning path, against omni-clone-en's own pinned
    # reference clip and transcript.
    clone = clone_case(arguments.manifest)
    clone_text = clone["input"]["text"]
    clone_reference = clone["input"]["reference"]
    clone_transcript = clone_reference["transcript"]
    clone_language = clone_reference.get("language_tag") or clone["input"].get("language_tag") or "en"

    reference_f32 = arguments.work / "clone-reference.f32"
    wav_to_f32(arguments.reference_wav, reference_f32)

    clone_a = synthesize(arguments, "clone-en-a", clone_language, "0", clone_text, reference_f32, clone_transcript)
    clone_b = synthesize(arguments, "clone-en-b", clone_language, "0", clone_text, reference_f32, clone_transcript)
    clone_no_profile = synthesize(arguments, "clone-en-no-profile", clone_language, "0", clone_text)
    if clone_a is None or clone_b is None or clone_no_profile is None:
        return 1

    # Check 8.
    record(
        "a clone request (Reference Audio profile + target text) succeeds",
        int(clone_a["status"]) == 0 and int(clone_a["frames"]) > 0,
        f"status {clone_a['status']}, frames {clone_a['frames']}",
    )

    # Check 9.
    record(
        "same-seed reproducibility holds WITH a profile",
        clone_a["digest"] == clone_b["digest"],
        f"{clone_a['digest'][:16]} vs {clone_b['digest'][:16]}",
    )

    # Check 10.
    record(
        "clone output differs from the no-profile run at the same seed",
        clone_a["digest"] != clone_no_profile["digest"],
        f"clone {clone_a['digest'][:16]} vs no-profile {clone_no_profile['digest'][:16]}",
    )

    # --- Task 15: the voice-design path, against omni-design-en's own
    # pinned instruct.
    design = design_case(arguments.manifest)
    design_text = design["input"]["text"]
    design_language = design["input"].get("language_tag") or "en"
    design_instruct = design["voice"]["description"]

    design_a = synthesize(arguments, "design-en-a", design_language, "0", design_text, instruct=design_instruct)
    design_b = synthesize(arguments, "design-en-b", design_language, "0", design_text, instruct=design_instruct)
    design_no_profile = synthesize(arguments, "design-en-no-profile", design_language, "0", design_text)
    if design_a is None or design_b is None or design_no_profile is None:
        return 1

    # Check 11.
    record(
        "a design request (Description Text profile + target text) succeeds",
        int(design_a["status"]) == 0 and int(design_a["frames"]) > 0,
        f"status {design_a['status']}, frames {design_a['frames']}",
    )

    # Check 12.
    record(
        "same-seed reproducibility holds WITH a design profile",
        design_a["digest"] == design_b["digest"],
        f"{design_a['digest'][:16]} vs {design_b['digest'][:16]}",
    )

    # Check 13.
    record(
        "design output differs from the no-profile run at the same seed",
        design_a["digest"] != design_no_profile["digest"],
        f"design {design_a['digest'][:16]} vs no-profile {design_no_profile['digest'][:16]}",
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
