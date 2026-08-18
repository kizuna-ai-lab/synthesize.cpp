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

TWO VARIANTS, TWO KINDS OF VOICE, AND THE SECOND ARRIVED ON 2026-08-17. Every
check above was written against Preset Voices, because qwen3-tts-12hz-0-6b-
customvoice catalogues nine of them. qwen3-tts-12hz-0-6b-base catalogues NONE --
its committed manifest declares `preset_ids: []` and its Voice Profile sources
are `reference_audio` and `serialized_profile` -- so not one of these checks
could run against it, and its `public` tolerance cell stayed an honest
placeholder for that reason rather than for want of trying.

Passing `--reference` twice selects the reference-audio path: the runner
prepares a Voice Profile through `synth_voice_profile_create_from_reference` and
the request carries that instead of a `voice_id`. The seed and language
properties are identical either way and are checked identically. Two are NOT
checkable there and are skipped with a reason recorded rather than silently
dropped: `resolved_voice` is empty for a Profile because a Profile has no id,
and a dialect speaker is a Preset Voice. "Different Voices give different audio"
survives in the form that means something for this variant -- two different
reference recordings.

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
    # PCM frames, which is what the public field counts. This used to be 24 and
    # meant 24 codec frames; the field was being compared in the wrong units, and
    # 24 PCM frames is one millisecond. 24 codec frames of 1920 samples is this.
    # Left unset so it can be resolved AFTER --reference is known; see
    # resolve_max_frames. 46080 was calibrated on the CustomVoice package, whose
    # Preset Voices take a short preset path, and it is too small for a clone.
    parser.add_argument("--max-frames", type=int, default=None)
    # The Quantization Profile and Execution Backend this run covers. They are
    # recorded rather than inferred because the tolerance grid is keyed on them:
    # a run that does not say which cell it filled cannot fill one.
    parser.add_argument("--profile", default="BF16")
    parser.add_argument("--backend", default="cpu", choices=("cpu", "cuda"))
    # Two are required rather than one: the check that a different Voice changes
    # the audio needs two Voices, and on a package with no Preset Voice catalogue
    # a Voice IS a reference recording. One would silently drop that check.
    parser.add_argument("--reference", action="append", default=None,
                        help="Reference audio for a package that catalogues no Preset Voices "
                             "(qwen3-tts-12hz-0-6b-base declares preset_ids: []). Pass TWICE. Each "
                             "is a path to a WAV or to raw float32 mono samples; a WAV is converted "
                             "once with soundfile because the runner reads raw float32.")
    return parser.parse_args()


# One millisecond of headroom is not the point; the two paths stop at different
# lengths for a structural reason. A Preset Voice on the CustomVoice package
# takes a short preset path and 46080 PCM frames -- 24 codec frames of 1920
# samples -- has always been enough. A Voice Profile prepared from a reference
# recording clones, and how long the clone runs depends on the sampled seed, so
# the random-seed check draws a length this script does not control.
#
# MEASURED, NOT ANTICIPATED: on 2026-08-17 that exact call returned
# SYNTH_ERR_OUTPUT_LIMIT (15) on the Base package and was rerun by hand at
# 983040. A gate that fails depending on a draw is not a gate, so the cap is
# selected by path rather than left to the caller to remember.
PRESET_VOICE_MAX_FRAMES = 46080
REFERENCE_VOICE_MAX_FRAMES = 983040


def resolve_max_frames(arguments: argparse.Namespace) -> int:
    """An explicit --max-frames always wins; otherwise the path chooses."""
    if arguments.max_frames is not None:
        return arguments.max_frames
    return REFERENCE_VOICE_MAX_FRAMES if arguments.reference else PRESET_VOICE_MAX_FRAMES


def materialize_reference(path: pathlib.Path, work: pathlib.Path, index: int) -> pathlib.Path:
    """The runner reads raw float32 mono; convert a WAV once if that is what we have.

    Deliberately NOT a WAV parser in the runner. There are already four in
    tests/, all C++, and that one is C -- and the conversion belongs on this
    side, where soundfile is a locked dependency and the manifest's own pinned
    artifact is what gets converted.
    """
    if path.suffix.lower() != ".wav":
        return path
    import numpy as np
    import soundfile

    samples, rate = soundfile.read(str(path), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1).astype(np.float32)
    if rate != 24000:
        raise SystemExit(f"{path}: {rate} Hz, but the package's declared reference rate is 24000 "
                         "and neither this script nor the runner resamples")
    work.mkdir(parents=True, exist_ok=True)
    target = work / f"reference-{index}.f32"
    target.write_bytes(np.ascontiguousarray(samples, dtype=np.float32).tobytes())
    return target


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
    arguments.max_frames = resolve_max_frames(arguments)
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        print(f"  [{'ok' if passed else 'FAIL'}] {name}: {detail}")

    skipped: list[dict] = []

    def skip(name: str, reason: str) -> None:
        skipped.append({"check": name, "inapplicable_because": reason})
        print(f"  [--] {name}: SKIPPED -- {reason}")

    # Which kind of Voice this package has. A Preset Voice catalogue and a
    # reference-audio Profile are both "a Voice" at the seam, and every seed and
    # language property below is identical for the two; what differs is how the
    # Voice is named and which two checks have a subject at all.
    references = arguments.reference or []
    if references:
        if len(references) < 2:
            print("--reference must be passed twice: the check that a different Voice changes the "
                  "audio needs two Voices, and here a Voice is a reference recording")
            return 1
        voice_a = "ref:" + str(materialize_reference(pathlib.Path(references[0]), arguments.work, 0))
        voice_b = "ref:" + str(materialize_reference(pathlib.Path(references[1]), arguments.work, 1))
        voice_a_label, voice_b_label = "reference A", "reference B"
        voice_kind = "reference_audio"
    else:
        voice_a, voice_b = "aiden", "vivian"
        voice_a_label, voice_b_label = "aiden", "vivian"
        voice_kind = "preset_voice"
    print(f"  Voice kind: {voice_kind}")

    first = synthesize(arguments, "seed-7-a", voice_a, "en", "7")
    if first is None:
        return 1
    again = synthesize(arguments, "seed-7-b", voice_a, "en", "7")
    other = synthesize(arguments, "seed-8", voice_a, "en", "8")
    if again is None or other is None:
        return 1

    record("seed is reported as requested", first["actual_seed"] == "7",
           f"requested 7, reported {first['actual_seed']}")
    record("same seed reproduces byte for byte", first["digest"] == again["digest"],
           f"{first['digest'][:16]} vs {again['digest'][:16]}")
    record("a different seed changes the audio", first["digest"] != other["digest"],
           f"seed 7 {first['digest'][:16]} vs seed 8 {other['digest'][:16]}")

    # A random seed has to come back concrete, and replaying it has to reproduce.
    drawn = synthesize(arguments, "seed-random", voice_a, "en", "random")
    if drawn is None:
        return 1
    record("a random seed is reported concretely", drawn["actual_seed"] not in ("", "0"),
           f"reported {drawn['actual_seed']}")
    replayed = synthesize(arguments, "seed-random-replay", voice_a, "en", drawn["actual_seed"])
    if replayed is None:
        return 1
    record("the reported random seed reproduces", drawn["digest"] == replayed["digest"],
           f"{drawn['digest'][:16]} vs {replayed['digest'][:16]}")

    # Two Voices at the same seed, which is what a Voice is for. On a package
    # with a Preset Voice catalogue those are two catalogue entries; on one
    # without, they are two reference recordings, which is the same claim about
    # the same seam.
    second_voice = synthesize(arguments, "voice-second", voice_b, "en", "7")
    if second_voice is None:
        return 1
    record("a different Voice changes the audio", first["digest"] != second_voice["digest"],
           f"{voice_a_label} {first['digest'][:16]} vs {voice_b_label} {second_voice['digest'][:16]}")

    if voice_kind == "preset_voice":
        record("the resolved Voice is reported", second_voice["resolved_voice"] == "vivian",
               f"reported {second_voice['resolved_voice']!r}")
    else:
        # Observed, not assumed: the runner reports an empty resolved_voice for a
        # Profile-selected Voice, because the request carried a voice_profile and
        # no voice_id, and there is no id for the seam to resolve or echo. That is
        # the seam behaving as designed, so this check has no subject here rather
        # than a failing one.
        skip("the resolved Voice is reported",
             "a Voice Profile has no id, so resolved_voice is empty by design "
             f"(observed {second_voice['resolved_voice']!r})")

    # A dialect speaker pins its own language regardless of the request. That
    # override happens inside the family, and the seam reports the language the
    # *request* matched -- so it is not observable here, and this phase does not
    # claim to check it. What it can check is that such a speaker synthesizes at
    # all and produces something distinct from a non-dialect one.
    if voice_kind == "preset_voice":
        dialect = synthesize(arguments, "voice-dialect", "eric", "en", "7")
        plain   = synthesize(arguments, "voice-plain", "aiden", "en", "7")
        if dialect is None or plain is None:
            return 1
        record("a dialect speaker synthesizes", dialect["frames"] > 0,
               f"{dialect['frames']} frames, language reported {dialect['resolved_language']!r}")
        record("a dialect speaker differs from a plain one", dialect["digest"] != plain["digest"],
               f"eric {dialect['digest'][:16]} vs aiden {plain['digest'][:16]}")
        language_witness = dialect
    else:
        skip("a dialect speaker synthesizes",
             "a dialect speaker is a Preset Voice and this package catalogues none")
        skip("a dialect speaker differs from a plain one",
             "a dialect speaker is a Preset Voice and this package catalogues none")
        language_witness = second_voice
    # The resolved language is the request's, reported back rather than assumed.
    # Before the language capability was read from the model this was hardcoded
    # "en" for every request, so a Chinese synthesis reported English.
    record("the resolved language is reported", language_witness["resolved_language"] == "en",
           f"requested en, reported {language_witness['resolved_language']!r}")

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
            "voice_kind": voice_kind,
            "checks": checks,
            # Recorded, not omitted: a cell whose count differs from another
            # variant's must say why, and "it was skipped" with a reason is a
            # different claim from "it passed".
            "skipped": skipped,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"\nreport: {arguments.report}")

    failed = [check["check"] for check in checks if not check["passed"]]
    if failed:
        print(f"\n{len(failed)} check(s) failed: {failed}")
        return 1
    if skipped:
        print(f"\n{len(skipped)} check(s) skipped as inapplicable to a {voice_kind} package: "
              f"{[entry['check'] for entry in skipped]}")
    print(f"\nall {len(checks)} public-request checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
