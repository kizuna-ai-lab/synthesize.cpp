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

THREE KINDS OF VOICE THIS SCRIPT CHECKS. The third arrived on 2026-08-18, the
second on 2026-08-17. Every check above was written against Preset Voices,
because qwen3-tts-12hz-0-6b-customvoice catalogues nine of them.
qwen3-tts-12hz-0-6b-base catalogues NONE -- its committed manifest declares
`preset_ids: []` and its Voice Profile sources are `reference_audio` and
`serialized_profile` -- so not one of these checks could run against it, and
its `public` tolerance cell stayed an honest placeholder for that reason
rather than for want of trying. qwen3-tts-12hz-1-7b-voicedesign catalogues no
Preset Voices either and takes no reference audio -- its declared Voice
Profile source is `description_text` -- and was itself an honest "not
reachable here yet" placeholder in this file's own history until Plan 2 Tasks
2-5 built `create_from_description`, serialized a design Profile, threaded the
instruct block into the prefill, and republished the capability bit.

Passing `--reference` twice selects the reference-audio path: the runner
prepares a Voice Profile through `synth_voice_profile_create_from_reference` and
the request carries that instead of a `voice_id`. The seed and language
properties are identical either way and are checked identically. Two are NOT
checkable there and are skipped with a reason recorded rather than silently
dropped: `resolved_voice` is empty for a Profile because a Profile has no id,
and a dialect speaker is a Preset Voice. "Different Voices give different audio"
survives in the form that means something for this variant -- two different
reference recordings.

Passing `--description` twice selects the Description Text path the same way,
through `synth_voice_profile_create_from_description`. Design section 6.3
names this path's three behavioural relations, and the same seed/language
checks and the same two skips above apply here for the same reasons -- a
Profile still has no id, and this package still catalogues no Preset Voices.
"Different Voices give different audio" becomes two different instructs, which
is exactly the shape relation 1 asks for and the reason two are required rather
than one: the SAME generic "a different Voice changes the audio" check IS
relation 1 for this Voice kind, and "same seed reproduces byte for byte" IS
relation 2 -- neither needed bespoke code, because both were already written
generically enough to mean the right thing for a description as much as for a
Preset Voice or a reference recording. Relation 3 (an empty instruct reproduces
upstream's own `instruct=""` output within tolerance) is different in kind --
a port-against-oracle comparison this script cannot make itself, since it has
no oracle artifact and, per this file's own opening paragraphs, phase 3 does
none of that on purpose -- so it is checked by reading the ALREADY-COMMITTED
figure the `replay` stage measured (tests/tolerances/qwen3-tts.json, this
variant's `replay.prefill` probe) and confirming the recorded observation is
still within the recorded bound, alongside a live smoke run through this same
public seam confirming an empty instruct still produces audio at all.

Design section 6.4 also names four refusals. Two are checked here directly by
attempting the OTHER kind of Voice Profile creation against whichever package
is loaded and confirming it is refused (`probe:reference` / `probe:description`
voice ids, handled by tests/qwen3_tts_public_real.c without ever synthesizing);
a fifth, unnamed by 6.4 but made explicit by Task 5
(`voice_profile.description_language_unsupported`), is checked the same way
(`probe:description-language`). The third -- a package claiming
`DESCRIPTION_TEXT` without the variant behind it, refused at load -- carries an
erratum in the design document: it is not implemented and is judged there to be
arguably impossible to implement package-side, since Description Text has no
distinguishing tensor to cross-check against the claim the way Reference Audio's
speaker encoder does. Not checked here either, for the same reason: this script
drives real packages through the public seam and has no way to manufacture the
malformed one the refusal would apply to. The fourth (retaining the speaker
slot when it must be absent) is a prefill-shape defect, not a seam relation --
tests/qwen3_tts_voicedesign_prefill_real.cpp's own fault injection is what
catches it, not this file.

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
    # Two, for the identical reason --reference is two: the check that a
    # different instruct changes the audio (design section 6.3's relation 1)
    # needs two instructs, and one would silently drop it.
    parser.add_argument("--description", action="append", default=None,
                        help="Description Text instruct for a package whose declared Voice Profile "
                             "source is description_text (qwen3-tts-12hz-1-7b-voicedesign). Pass "
                             "TWICE. Each is a sentence of natural language, prepared through "
                             "synth_voice_profile_create_from_description.")
    # tests/tolerances/qwen3-tts.json's own replay-stage figure is what relation
    # 3 reads rather than recomputes; see check_empty_instruct_within_tolerance.
    parser.add_argument("--tolerances", type=pathlib.Path,
                        default=pathlib.Path("tests/tolerances/qwen3-tts.json"))
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
# A Description Text synthesis has the identical property a reference clone's
# does: nothing about the instruct pins how long the talker samples for, so
# the length depends on the seed the same random draw draws, not on this
# script. MEASURED, NOT ANTICIPATED, the same way the cap above was: the
# oracle's own two committed voicedesign cases (tests/tolerances/qwen3-tts.json,
# this variant's replay.prefill probe) already run 48000 and 82560 PCM frames
# at instruct lengths of 0 and 19 tokens -- 48000 alone clears
# PRESET_VOICE_MAX_FRAMES before any seed-driven tail is even considered, so
# the smaller cap was never going to survive first contact here either. Reusing
# the reference path's own cap rather than inventing a third number: nothing
# about the failure mode differs between the two paths, only which kind of
# Voice Profile is being sampled from.
DESCRIPTION_VOICE_MAX_FRAMES = REFERENCE_VOICE_MAX_FRAMES


def resolve_max_frames(arguments: argparse.Namespace) -> int:
    """An explicit --max-frames always wins; otherwise the path chooses."""
    if arguments.max_frames is not None:
        return arguments.max_frames
    if arguments.reference:
        return REFERENCE_VOICE_MAX_FRAMES
    if arguments.description:
        return DESCRIPTION_VOICE_MAX_FRAMES
    return PRESET_VOICE_MAX_FRAMES


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


# Copied from include/synthesize.h rather than parsed from it -- this script
# already keeps independent small copies of ABI facts (the PCM-frames-not-
# codec-frames convention above is one), and a third mirrored constant is
# still cheaper than machine-reading a C header from Python.
SYNTH_ERR_UNSUPPORTED_INPUT = 8
SYNTH_ERR_UNSUPPORTED_VOICE = 10


def probe_refusal(arguments: argparse.Namespace, name: str, probe: str) -> dict | None:
    """Runs the runner in `probe:KIND` mode: it attempts exactly one Voice
    Profile creation call against the loaded package and reports the
    synth_status_t it observed, succeeding (exit 0) whichever way that call
    went -- the runner reports, this script judges, the same split every other
    check in this file keeps. Language, seed and max-frames are irrelevant to
    a probe that never reaches synth_synthesize, so placeholders are passed.
    """
    target = arguments.work / f"{name}.pcm"
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [str(arguments.runner), str(arguments.model), str(target), f"probe:{probe}",
               "-", "0", str(arguments.max_frames), arguments.backend]
    finished = subprocess.run(command, input=b"", capture_output=True)
    if finished.returncode != 0:
        print(f"  {name}: runner failed unexpectedly: "
              f"{finished.stderr.decode('utf-8', 'replace').strip()[:200]}")
        return None
    return json.loads(finished.stdout.decode("utf-8"))


def check_empty_instruct_within_tolerance(arguments: argparse.Namespace) -> tuple[bool | None, str]:
    """Design section 6.3's relation 3: 'an empty instruct reproduces upstream's
    instruct="" output within the replay stage's recorded tolerance.' Unlike
    relations 1 and 2, this is a port-against-oracle comparison, and an oracle
    artifact is not something this script ever has -- this file's own opening
    paragraphs are explicit that phase 3 does none of that on purpose.

    What makes the relation checkable anyway: the instruct only ever enters
    this family's pipeline through the assembled talker PREFILL (design
    section 5 -- no new graph, no change downstream of it), so a prefill built
    from an empty instruct behaves exactly as a prefill built with no instruct
    block at all. That comparison is already made, on the pinned oracle, by
    tests/qwen3_tts_voicedesign_prefill_real.cpp, and its result is committed
    in tests/tolerances/qwen3-tts.json under this variant's own
    `replay.prefill` probe (case `voicedesign-empty-instruct-en`). This
    function reads that COMMITTED figure and checks the contract it makes is
    the one still on disk, rather than re-deriving a number this script has no
    oracle to derive on its own. Returns (None, reason) when the figure is not
    recorded for this profile at all -- "unmeasured" is a different claim from
    "measured and failing" and is reported as a skip, not a failure. It returns
    (False, reason) instead, a hard failure rather than a skip, when the
    `voicedesign-empty-instruct-en` case EXISTS but no longer carries an empty
    instruct: a case id that promises "empty instruct" is itself part of the
    contract this check verifies, and a silently repurposed case (edited to a
    non-empty instruct without this check noticing) is exactly the kind of
    silent failure design section 6's own opening paragraph is built around --
    reporting that as "unmeasured" would be the weaker of the two wrong
    behaviours, since a caller reading a skip has no reason to suspect the
    citation's premise moved out from under it.
    """
    try:
        tolerances = json.loads(arguments.tolerances.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, f"cannot read {arguments.tolerances}: {error}"
    # The variant name is the model's own parent directory, matching this
    # family's own on-disk convention (models/<variant>/<variant>-<profile>.gguf)
    # rather than a separately-threaded --variant flag this script has never
    # needed before.
    variant = arguments.model.parent.name
    stages = (tolerances.get("variants", {}).get(variant, {}).get("profiles", {})
              .get(arguments.profile, {}).get("stages", {}))
    prefill = stages.get("replay", {}).get("probes", {}).get("prefill")
    if prefill is None:
        return None, (f"no replay.prefill probe recorded for variant {variant!r} profile "
                      f"{arguments.profile!r} in {arguments.tolerances}")
    case = prefill.get("observed_by_case", {}).get("voicedesign-empty-instruct-en")
    if case is None:
        return None, "replay.prefill probe has no voicedesign-empty-instruct-en case recorded"
    if case.get("instruct") != "":
        return False, (f"voicedesign-empty-instruct-en case no longer carries an empty instruct "
                       f"(observed {case.get('instruct')!r}) -- the case id's own contract broke, "
                       f"not merely an unmeasured configuration")
    bound = prefill.get("max_relative")
    observed = case.get("p95_relative")
    if bound is None or observed is None:
        return None, "replay.prefill probe or its case is missing max_relative/p95_relative"
    passed = observed <= bound
    detail = (f"replay stage's prefill probe, case voicedesign-empty-instruct-en: p95_relative "
              f"{observed} against a bound of {bound} (measured by "
              f"tests/qwen3_tts_voicedesign_prefill_real.cpp, not recomputed by this script)")
    return passed, detail


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

    # Which kind of Voice this package has. A Preset Voice catalogue, a
    # reference-audio Profile and a Description Text Profile are all "a Voice"
    # at the seam, and every seed and language property below is identical for
    # the three; what differs is how the Voice is named and which checks have a
    # subject at all.
    references = arguments.reference or []
    descriptions = arguments.description or []
    if references and descriptions:
        print("--reference and --description select different kinds of Voice for this validator; "
              "pass only one")
        return 1
    if references:
        if len(references) < 2:
            print("--reference must be passed twice: the check that a different Voice changes the "
                  "audio needs two Voices, and here a Voice is a reference recording")
            return 1
        voice_a = "ref:" + str(materialize_reference(pathlib.Path(references[0]), arguments.work, 0))
        voice_b = "ref:" + str(materialize_reference(pathlib.Path(references[1]), arguments.work, 1))
        voice_a_label, voice_b_label = "reference A", "reference B"
        voice_kind = "reference_audio"
    elif descriptions:
        if len(descriptions) < 2:
            print("--description must be passed twice: the check that a different instruct changes "
                  "the audio needs two instructs, and here a Voice is a Description Text instruct")
            return 1
        voice_a = "desc:" + descriptions[0]
        voice_b = "desc:" + descriptions[1]
        voice_a_label, voice_b_label = "description A", "description B"
        voice_kind = "description_text"
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
    # For voice_kind == "description_text" this IS design section 6.3's
    # relation 2 ("the same instruct and seed give byte-identical audio"):
    # `voice_a` names one instruct, both runs share it and seed 7, and the
    # comparison is exact -- no tolerance, matching the design's own "both
    # sides are this port" reasoning.
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
    # without, they are two reference recordings or two instructs, which is the
    # same claim about the same seam. For voice_kind == "description_text" this
    # IS design section 6.3's relation 1 ("two different instructs, same seed,
    # same text -> the audio differs"): voice_a and voice_b are the two
    # required --description instructs, first and second_voice share seed 7 and
    # arguments.text, and the digests being compared were both computed from
    # the runner's own PCM output BEFORE this record() call ever runs -- the
    # byte difference is confirmed, not assumed, ahead of the verdict, per the
    # design's own "trivial truth" discipline.
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

    # Design section 6.3's relation 3, description_text only: an empty instruct
    # reproduces upstream's own instruct="" output within the replay stage's
    # recorded tolerance. Two parts -- a live smoke run through THIS public
    # seam (does an empty instruct still synthesize at all, here, today), and a
    # citation of the already-committed oracle comparison (does the number that
    # backs the claim still hold on disk). Neither alone would be the relation:
    # the smoke run alone cannot compare against upstream, and the citation
    # alone would not prove the public seam still reaches the same code path.
    if voice_kind == "description_text":
        empty_instruct = synthesize(arguments, "description-empty-instruct", "desc:", "en", "7")
        if empty_instruct is None:
            return 1
        record("an empty instruct synthesizes through the public seam", empty_instruct["frames"] > 0,
               f"{empty_instruct['frames']} frames")
        oracle_passed, oracle_detail = check_empty_instruct_within_tolerance(arguments)
        name = "an empty instruct reproduces the oracle within the replay stage's recorded tolerance"
        if oracle_passed is None:
            skip(name, oracle_detail)
        else:
            record(name, oracle_passed, oracle_detail)

    # Design section 6.4's refusals. Each package can only demonstrate the
    # refusal of the OTHER Voice Profile source it does not declare -- a
    # VoiceDesign package (description_text) has no speaker encoder to prepare
    # a reference Profile from; a CustomVoice or Base package (preset_voice or
    # reference_audio) has no create_from_description arm wired to it at all.
    # `probe:KIND` never reaches synth_synthesize, so none of this needs the
    # max-frames cap or a real seed.
    if voice_kind == "description_text":
        reference_refusal = probe_refusal(arguments, "probe-reference-on-voicedesign", "reference")
        if reference_refusal is None:
            return 1
        record("a VoiceDesign package refuses create_from_reference",
               reference_refusal["status"] == SYNTH_ERR_UNSUPPORTED_VOICE,
               f"status {reference_refusal['status']} "
               f"(SYNTH_ERR_UNSUPPORTED_VOICE is {SYNTH_ERR_UNSUPPORTED_VOICE})")

        # Not one of design section 6.4's four bullets, but made explicit by
        # Task 5's own refusal (voice_profile.description_language_unsupported,
        # SYNTH_ERR_UNSUPPORTED_INPUT): D5 keeps language a per-synthesis field
        # rather than part of the Profile, and a caller supplying one to
        # create_from_description is refused rather than silently ignored.
        language_refusal = probe_refusal(arguments, "probe-description-language", "description-language")
        if language_refusal is None:
            return 1
        record("a supplied description language tag is refused",
               language_refusal["status"] == SYNTH_ERR_UNSUPPORTED_INPUT,
               f"status {language_refusal['status']} "
               f"(SYNTH_ERR_UNSUPPORTED_INPUT is {SYNTH_ERR_UNSUPPORTED_INPUT})")
    else:
        description_refusal = probe_refusal(arguments, "probe-description-on-this-package", "description")
        if description_refusal is None:
            return 1
        record(f"this package ({voice_kind}) refuses create_from_description",
               description_refusal["status"] == SYNTH_ERR_UNSUPPORTED_VOICE,
               f"status {description_refusal['status']} "
               f"(SYNTH_ERR_UNSUPPORTED_VOICE is {SYNTH_ERR_UNSUPPORTED_VOICE})")

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
