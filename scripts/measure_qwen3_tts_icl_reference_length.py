#!/usr/bin/env python3
"""Measure how ICL output length responds to REFERENCE length, everything else fixed.

WHY THIS EXISTS. Plan 1 recorded a 30 s reference producing 9 codec frames for a
sentence that takes ~45 at the clip's native length, and it sat unexplained
across three plans as "the 9-frame anomaly" -- a suspected defect. The Golden
cases hint it is not an anomaly at all but the end of a monotone trend: with the
SAME target text, a 1 s reference gave 127 frames, 8 s gave 45, and 30 s gave 9.
Three points do not establish a curve, so this script adds enough to say whether
the relation is real, monotone, and what shape it has.

WHAT IS HELD FIXED, because the entire claim rests on it. Every run below shares:

  * target text      "This is a test of Qwen three T T S base voice cloning."
  * reference clip   the one committed clone.wav (193,920 samples @ 24 kHz)
  * reference text   the clip's own single-repetition transcript (arm A)
  * language         English
  * seed             0, applied via torch.manual_seed immediately before each
                     generate_voice_clone call (dump_reference_qwen3_tts_base
                     .run_case does this, so seeding is identical to the Golden
                     dumps by construction, not by imitation)
  * sampling         do_sample and subtalker_dosample both True, max_new_tokens
                     2048 -- the manifest's own parameters
  * model            one Qwen3TTSModel, bfloat16, eager attention, loaded ONCE
                     and reused, so no run can differ by a reload

The ONLY thing that varies in arm A is ``trim_seconds``, which sets how much
reference audio is fed and therefore ``reference_code_frames``.

THE DUMPER IS IMPORTED, NOT REIMPLEMENTED. Both this sweep and the committed
Golden dumps run the identical ``run_case``, the same technique
scripts/dump_reference_qwen3_tts_codec_encoder_float32.py uses -- so a point
measured here is directly comparable to a manifest case, and a change to the
dumper cannot silently make the two disagree.

ARM B IS THE CONTROL THAT SEPARATES TWO MECHANISMS. Past the clip's own 8.08 s,
``trim_seconds`` is reached by LOOPING the clip, so the reference audio says the
sentence several times while the reference transcript still says it once. That is
a transcript-audio mismatch introduced by the harness, and Task 11's review found
that a mismatched transcript can make ICL run away to the frame ceiling. Arm B
repeats the transcript to match the number of loops. If output length recovers,
the driver is the MISMATCH; if it does not, the driver is reference LENGTH and
the mismatch hypothesis is refuted for this direction.

Artifacts are written to a scratch tree and are not committed; the report this
prints is the result. Usage:

    uv run --project scripts/envs/qwen3-tts --locked python \\
      scripts/measure_qwen3_tts_icl_reference_length.py \\
      --weights-dir models/qwen3-tts-12hz-0-6b-base \\
      --out build/qwen3-tts-icl-length-sweep
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402

import dump_reference_qwen3_tts_base as dumper  # noqa: E402

TEXT = "This is a test of Qwen three T T S base voice cloning."
REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
    "You blew it! And thanks to you."
)
REF_AUDIO = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"

# The committed clip, needed to predict how many times a trim loops it.
SOURCE_SAMPLES = 193_920
SAMPLE_RATE = 24_000
SOURCE_SECONDS = SOURCE_SAMPLES / SAMPLE_RATE  # 8.08

# Arm A: reference length varies, transcript stays as the clip's own.
# Chosen to bracket the three Golden points (1 s, native, 30 s) on both sides
# and to sample densely enough below the native length that a knee would show.
TRIMS = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, None, 10.0, 12.0, 16.0, 20.0, 24.0, 30.0]

# Arm B: only the looped trims, with the transcript repeated to match.
MATCHED_TRIMS = [10.0, 16.0, 20.0, 24.0, 30.0]

# Arm C: the seed scatter, which separates "a function of reference length" from
# "the stopping decision has become unstable". Every point in arms A and B is
# seeded 0, so each is deterministic -- but determinism per point says nothing
# about whether a neighbouring seed lands somewhere else entirely. A trim in the
# clean region and a trim in the erratic region get the same five seeds; if the
# clean one holds steady while the erratic one scatters, the erraticism is
# instability rather than a curve this sweep has undersampled.
SEED_SCATTER_TRIMS = [None, 30.0]
SEEDS = [0, 1, 2, 3, 4]


def loop_repeats(trim_seconds: float | None) -> int:
    if trim_seconds is None or trim_seconds <= SOURCE_SECONDS:
        return 1
    return math.ceil(trim_seconds * SAMPLE_RATE / SOURCE_SAMPLES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", required=True, type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("build/qwen3-tts-icl-length-sweep"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--reference-audio-dir", type=pathlib.Path, default=pathlib.Path("models/qwen3-tts-reference-audio")
    )
    parser.add_argument("--report", type=pathlib.Path, default=None)
    parser.add_argument("--arms", nargs="*", default=None,
                        help="Restrict to these arm letters, e.g. --arms C.")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"--device {args.device!r} requests CUDA but none is available")

    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        str(args.weights_dir), device_map=args.device, dtype=torch.bfloat16, attn_implementation="eager"
    )
    print(f"reference: {args.device} bfloat16 eager (loaded once, reused for every point)", flush=True)

    rows = []
    plan = [("A_transcript_x1", t, REF_TEXT, 0) for t in TRIMS]
    plan += [("B_transcript_matched", t, " ".join([REF_TEXT] * loop_repeats(t)), 0) for t in MATCHED_TRIMS]
    plan += [("C_seed_scatter", t, REF_TEXT, s) for t in SEED_SCATTER_TRIMS for s in SEEDS]
    if args.arms:
        plan = [entry for entry in plan if entry[0].split("_")[0] in args.arms]

    for index, (arm, trim, ref_text, seed) in enumerate(plan, 1):
        case_id = f"{arm}-trim{'native' if trim is None else trim}-seed{seed}"
        case = dumper.RunCase(
            id=case_id,
            ref_audio=REF_AUDIO,
            text=TEXT,
            language="English",
            x_vector_only=False,
            ref_text=ref_text,
            trim_seconds=trim,
            seed=seed,
            max_new_tokens=2048,
        )
        print(f"[{index}/{len(plan)}] {case_id} (loops={loop_repeats(trim)})", flush=True)
        try:
            record = dumper.run_case(model, case, args.out / case_id, args.reference_audio_dir)
            result = record["result"]
            row = {
                "arm": arm,
                "seed": seed,
                "trim_seconds": trim,
                "loop_repeats": loop_repeats(trim),
                "ref_text_repeats": len(ref_text) // len(REF_TEXT),
                "reference_code_frames": result.get("reference_code_frames"),
                "generated_code_frames": result.get("generated_code_frames"),
                "duration_seconds": result.get("duration_seconds"),
                "status": result.get("status"),
            }
        except SystemExit as error:
            # A non-terminating input is a datum, not a crash: record it and go on.
            row = {
                "arm": arm,
                "seed": seed,
                "trim_seconds": trim,
                "loop_repeats": loop_repeats(trim),
                "ref_text_repeats": len(ref_text) // len(REF_TEXT),
                "reference_code_frames": None,
                "generated_code_frames": None,
                "duration_seconds": None,
                "status": f"refused: {error}",
            }
        rows.append(row)
        print(
            f"    ref_frames={row['reference_code_frames']} "
            f"generated={row['generated_code_frames']} status={row['status']}",
            flush=True,
        )

    report = {
        "schema": "synthesize-qwen3-tts-icl-length-sweep-v1",
        "family": "qwen3-tts",
        "variant": "qwen3-tts-12hz-0-6b-base",
        "held_fixed": {
            "text": TEXT,
            "language": "English",
            "seed": "0 for arms A and B; arm C varies it deliberately",
            "max_new_tokens": 2048,
            "do_sample": True,
            "subtalker_dosample": True,
            "reference_clip_samples": SOURCE_SAMPLES,
            "model_dtype": "bfloat16",
            "attn_implementation": "eager",
            "model_loaded_once": True,
        },
        "rows": rows,
    }
    destination = args.report or (args.out / "report.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
