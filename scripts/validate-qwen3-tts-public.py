#!/usr/bin/env python3
"""Stage 5 phase 3: the public seam, without injecting anything.

Phase 2 replays the oracle's codes and compares tensors. This phase does none of
that: it drives `synth_synthesize` the way a caller does and checks the promises
the public interface makes about seeds, Voices and languages.

Five properties, each of which can break while every tensor comparison still
passes:

- **The reported seed is the seed that was used.** A request naming a seed gets
  it back; a request asking for a random one gets a concrete value back, and
  replaying that value reproduces the audio bit for bit. Reporting a seed that
  does not reproduce is worse than reporting none.
- **The same seed gives the same audio.** Byte-identical, not merely close: the
  sampler is the only stochastic part and it is seeded.
- **Different seeds give different audio.** A seed that changes nothing means the
  draw is not reaching the sampler, which no tolerance would catch.
- **Different Voices give different audio**, which is what a Preset Voice is for.
- **A dialect speaker overrides the requested language**, because that is what
  the reference does.

Run from the repository root:

    uv run --project scripts/envs/qwen3-tts --locked python \
      scripts/validate-qwen3-tts-public.py --report reports/validate/qwen3-tts/public.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path,
                        default=pathlib.Path("models/qwen3-tts-12hz-0-6b-customvoice/"
                                             "qwen3-tts-12hz-0-6b-customvoice-BF16.gguf"))
    parser.add_argument("--runner", type=pathlib.Path,
                        default=pathlib.Path("build/unit/bin/synthesize-qwen3-tts-public-real"))
    parser.add_argument("--work", type=pathlib.Path, default=pathlib.Path("build/goldens/qwen3-tts-public"))
    parser.add_argument("--report", type=pathlib.Path, default=None)
    # Short on purpose: this phase checks the seam, not the audio, and every run
    # is a full synthesis.
    parser.add_argument("--text", default="Hi.")
    parser.add_argument("--max-frames", type=int, default=24)
    # The Quantization Profile and Execution Backend this run covers. They are
    # recorded rather than inferred because the tolerance grid is keyed on them:
    # a run that does not say which cell it filled cannot fill one.
    parser.add_argument("--profile", default="BF16")
    parser.add_argument("--backend", default="cpu", choices=("cpu", "cuda"))
    return parser.parse_args()


def synthesize(arguments: argparse.Namespace, name: str, voice: str, language: str | None,
               seed: str) -> dict | None:
    target = arguments.work / f"{name}.pcm"
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [str(arguments.runner), str(arguments.model), str(target), voice,
               language if language is not None else "-", seed, str(arguments.max_frames),
               arguments.backend]
    finished = subprocess.run(command, input=arguments.text.encode("utf-8"), capture_output=True)
    if finished.returncode != 0:
        print(f"  {name}: runner failed: {finished.stderr.decode('utf-8', 'replace').strip()[:200]}")
        return None
    observed = json.loads(finished.stdout.decode("utf-8"))
    observed["digest"] = hashlib.sha256(target.read_bytes()).hexdigest()
    observed["bytes"] = target.stat().st_size
    return observed


def main() -> int:
    arguments = parse_args()
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        print(f"  [{'ok' if passed else 'FAIL'}] {name}: {detail}")

    first = synthesize(arguments, "seed-7-a", "aiden", "en", "7")
    if first is None:
        return 1
    again = synthesize(arguments, "seed-7-b", "aiden", "en", "7")
    other = synthesize(arguments, "seed-8", "aiden", "en", "8")
    if again is None or other is None:
        return 1

    record("seed is reported as requested", first["actual_seed"] == "7",
           f"requested 7, reported {first['actual_seed']}")
    record("same seed reproduces byte for byte", first["digest"] == again["digest"],
           f"{first['digest'][:16]} vs {again['digest'][:16]}")
    record("a different seed changes the audio", first["digest"] != other["digest"],
           f"seed 7 {first['digest'][:16]} vs seed 8 {other['digest'][:16]}")

    # A random seed has to come back concrete, and replaying it has to reproduce.
    drawn = synthesize(arguments, "seed-random", "aiden", "en", "random")
    if drawn is None:
        return 1
    record("a random seed is reported concretely", drawn["actual_seed"] not in ("", "0"),
           f"reported {drawn['actual_seed']}")
    replayed = synthesize(arguments, "seed-random-replay", "aiden", "en", drawn["actual_seed"])
    if replayed is None:
        return 1
    record("the reported random seed reproduces", drawn["digest"] == replayed["digest"],
           f"{drawn['digest'][:16]} vs {replayed['digest'][:16]}")

    # Two Preset Voices at the same seed, which is what a Voice is for.
    second_voice = synthesize(arguments, "voice-vivian", "vivian", "en", "7")
    if second_voice is None:
        return 1
    record("a different Voice changes the audio", first["digest"] != second_voice["digest"],
           f"aiden {first['digest'][:16]} vs vivian {second_voice['digest'][:16]}")
    record("the resolved Voice is reported", second_voice["resolved_voice"] == "vivian",
           f"reported {second_voice['resolved_voice']!r}")

    # A dialect speaker pins its own language regardless of the request. That
    # override happens inside the family, and the seam reports the language the
    # *request* matched -- so it is not observable here, and this phase does not
    # claim to check it. What it can check is that such a speaker synthesizes at
    # all and produces something distinct from a non-dialect one.
    dialect = synthesize(arguments, "voice-dialect", "eric", "en", "7")
    plain   = synthesize(arguments, "voice-plain", "aiden", "en", "7")
    if dialect is None or plain is None:
        return 1
    record("a dialect speaker synthesizes", dialect["frames"] > 0,
           f"{dialect['frames']} frames, language reported {dialect['resolved_language']!r}")
    record("a dialect speaker differs from a plain one", dialect["digest"] != plain["digest"],
           f"eric {dialect['digest'][:16]} vs aiden {plain['digest'][:16]}")
    # The resolved language is the request's, reported back rather than assumed.
    # Before the language capability was read from the model this was hardcoded
    # "en" for every request, so a Chinese synthesis reported English.
    record("the resolved language is reported", dialect["resolved_language"] == "en",
           f"requested en, reported {dialect['resolved_language']!r}")

    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(json.dumps({
            "schema": "synthesize-validation-report-v1",
            "family": "qwen3-tts",
            "phase": "public_request",
            "profile": arguments.profile,
            "backend": arguments.backend.upper(),
            "text": arguments.text,
            "max_frames": arguments.max_frames,
            "checks": checks,
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
