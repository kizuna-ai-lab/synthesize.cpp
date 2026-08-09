#!/usr/bin/env python3
"""The speaker-identity F0 proxy, and the degeneracy screen beside it.

Tier 3 of this family's quantization programme asks a question the exact-token
gate cannot answer: a quantized (or differently placed) generator produces a
*different valid realization*, not a wrong one, so token agreement is data and
never a verdict. What a listener actually notices is whether the voice changed
register -- whether it still sounds like the same person. The instrument for
that, established on 2026-08-08 and confirmed against a human ear in BOTH
directions on the same day, is:

  median F0 over voiced frames from two independent estimators (YIN with the
  cumulative mean normalised difference function, and a normalised
  cross-correlation tracker), plus the fraction of voiced frames below 165 Hz.

  A case is called **changed** only on complete register separation -- the
  below-165 Hz fraction going 1.00 to 0.00, or 0.00 to 1.00, on *both*
  trackers -- **and** a >40% median-F0 shift on *both*.

That conjunction is deliberately hard to trip. It is why the 2026-08-08 sweep
reported 1 of 17 for the CPU->CUDA generator placement move rather than a
number that tracks token drift: `omni-digits` flips 98.3% of its tokens and
keeps its speaker, `omni-short-en` flips 94.0% and changes. Never substitute a
flip percentage for this measure.

**Why this file exists.** Every F0 number published in this family's porting
log before 2026-08-09 came from code that was written, used and thrown away --
three profiles measured by three ad-hoc reimplementations are not comparable to
each other or to the 1-of-17 baseline they must be read against. The numbers
are only citable if the instrument is.

The same pass emits the **degeneracy screen**, which answers a different and
cruder question: is this render speech at all? Speech-shaped output from this
family sits at a zero-crossing rate of 3.4-4.2 kHz with a 20 ms-frame envelope
ratio of 218-1988. The known failure mode is a sub-50 Hz near-DC buzz with no
envelope at all. A render that fails this screen is not "a different speaker";
it is not a voice, and its F0 verdict must be read as *not comparable* rather
than as *unchanged*.

`--reference`/`--candidate` each take either a single raw float32 render or a
directory of `<case>/pcm_freerun.f32` renders (what
`scripts/validate-omnivoice-replay.py --work` writes). WAV is accepted too.

    python scripts/omnivoice-speaker-proxy.py \
      --reference build/goldens/f32-cpu --candidate build/goldens/q8gen-cpu \
      --name-reference F32/CPU --name-candidate Q8/CPU \
      --report speaker-proxy.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys
import wave

import numpy as np

# This family's output rate. Every render compared here comes from the same
# package, so a mismatch is a caller error rather than something to resample.
SAMPLE_RATE = 24000

# The 2026-08-08 parameters, recorded so a later run is comparable rather than
# merely plausible. The 1024-sample YIN frame at a 10 ms hop is what that
# sweep used; the 0.15 absolute threshold is YIN's own published default.
YIN_FRAME = 1024
HOP_SECONDS = 0.010
YIN_THRESHOLD = 0.15

# The NCC tracker gets its own window rather than reusing YIN's, so the two
# estimators are independent in what they see as well as in how they decide.
# They must be able to disagree: the 2026-08-08 sweep's one borderline case
# (`omni-design-zh`, NCC 160.0 -> 147.2 with YIN unchanged) is the reason the
# criterion demands agreement from both, and an instrument whose two halves
# always agree could not have produced that case.
NCC_FRAME = 1536
NCC_VOICING = 0.40
# Sub-harmonic suppression. For a strongly periodic frame the correlation at
# 2*tau and 3*tau is very nearly as high as at tau, so a plain argmax reports
# half or a third of the true F0 -- measured directly: a 150 Hz sawtooth came
# back as 75 Hz and a 300 Hz one as 100 Hz before this was added. Among the
# lags within this factor of the peak, the SHORTEST wins.
NCC_SUBHARMONIC = 0.90

# A frame with no energy has no pitch, and YIN in particular will happily call
# digital silence perfectly periodic (its normalised difference is 0/0 there).
# Reporting silence as voiced is how an instrument turns a dead render into
# "unchanged"; both estimators gate on this before deciding anything.
SILENCE_FLOOR = 1e-5
SILENCE_RELATIVE = 1e-3

# Human speech F0 search range. 60 Hz is below any voice this family produces;
# 400 Hz is above. Widening it would let the tracker lock onto a harmonic or
# onto the sub-50 Hz buzz the degeneracy screen exists to catch.
F0_MIN = 60.0
F0_MAX = 400.0

# The register boundary. Not a claim about sex -- it is the split that made the
# 2026-08-08 sweep's one positive case separate completely on both trackers,
# and the listener agreed with it.
REGISTER_HZ = 165.0

# The verdict's two halves, both required.
MEDIAN_SHIFT = 0.40

# The degeneracy screen's published speech-shaped band for this family.
ZCR_BAND = (3400.0, 4200.0)
ENVELOPE_BAND = (218.0, 1988.0)
ENVELOPE_FRAME_SECONDS = 0.020
# Below these, the render is the known failure mode rather than a quiet voice.
DEGENERATE_ZCR = 500.0
DEGENERATE_ENVELOPE = 10.0


def read_render(path: pathlib.Path) -> np.ndarray:
    """Raw float32 (what the replay runner writes) or a 16-bit/float WAV."""
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as handle:
            frames = handle.readframes(handle.getnframes())
            width = handle.getsampwidth()
            channels = handle.getnchannels()
        if width == 2:
            audio = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
        elif width == 4:
            audio = np.frombuffer(frames, dtype="<f4").astype(np.float64)
        else:
            raise SystemExit(f"{path}: unsupported WAV sample width {width}")
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        return audio
    return np.frombuffer(path.read_bytes(), dtype="<f4").astype(np.float64)


def _frames(signal: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """Non-copying strided view of the frames, oldest first."""
    if signal.size < frame:
        return np.zeros((0, frame), dtype=signal.dtype)
    count = 1 + (signal.size - frame) // hop
    return np.lib.stride_tricks.as_strided(
        signal, shape=(count, frame),
        strides=(signal.strides[0] * hop, signal.strides[0]), writeable=False)


def _correlations(block: np.ndarray, window: int, tau_max: int):
    """r(0,0), r(tau,tau) and r(0,tau) for every frame in `block`.

    The three quantities both estimators are built from. r(0,tau) is the
    cross-correlation of the frame's first `window` samples against itself,
    taken by FFT because a direct loop over 341 lags on a 2,876-frame render
    is minutes rather than milliseconds.
    """
    block = block - block.mean(axis=1, keepdims=True)
    power = np.concatenate([np.zeros((block.shape[0], 1)),
                            np.cumsum(block * block, axis=1)], axis=1)
    r00 = power[:, window] - power[:, 0]
    rtt = power[:, np.arange(tau_max + 1) + window] - power[:, np.arange(tau_max + 1)]
    size = 1
    while size < block.shape[1] + window:
        size *= 2
    head = np.zeros((block.shape[0], size))
    head[:, :window] = block[:, :window]
    full = np.zeros((block.shape[0], size))
    full[:, :block.shape[1]] = block
    spectrum = np.fft.rfft(full, axis=1) * np.conj(np.fft.rfft(head, axis=1))
    r0t = np.fft.irfft(spectrum, n=size, axis=1)[:, :tau_max + 1]
    return r00, rtt, r0t


def _silence_gate(signal: np.ndarray) -> float:
    """The per-frame RMS below which a frame carries no pitch to find."""
    peak = float(np.abs(signal).max(initial=0.0))
    return max(SILENCE_FLOOR, SILENCE_RELATIVE * peak)


def _parabolic(values: np.ndarray, index: int) -> float:
    """Sub-sample refinement of a lag, guarded at the ends."""
    if index <= 0 or index >= values.size - 1:
        return float(index)
    left, middle, right = values[index - 1], values[index], values[index + 1]
    denominator = 2.0 * (2.0 * middle - left - right)
    if denominator == 0.0:
        return float(index)
    return float(index) + float((right - left) / denominator)


def estimate_yin(signal: np.ndarray, sample_rate: int) -> dict:
    """YIN with the cumulative mean normalised difference function.

    d(tau) is built from the three correlation terms rather than a direct
    squared-difference loop; the identity
    d(tau) = r(0,0) + r(tau,tau) - 2 r(0,tau) is exact, not an approximation.
    """
    hop = int(round(sample_rate * HOP_SECONDS))
    tau_min = max(2, int(np.floor(sample_rate / F0_MAX)))
    tau_max = int(np.ceil(sample_rate / F0_MIN))
    window = YIN_FRAME - tau_max
    if window < tau_max // 2:
        raise SystemExit("YIN frame too short for the F0 range")
    block = _frames(signal, YIN_FRAME, hop)
    if block.shape[0] == 0:
        return {"frames": 0, "voiced_frames": 0, "median_f0_hz": None,
                "below_register_fraction": None}
    f0 = []
    gate = _silence_gate(signal)
    # Chunked so a 12-minute render does not need the whole FFT plane resident.
    for start in range(0, block.shape[0], 512):
        chunk = np.ascontiguousarray(block[start:start + 512], dtype=np.float64)
        loud = np.sqrt((chunk * chunk).mean(axis=1)) > gate
        r00, rtt, r0t = _correlations(chunk, window, tau_max)
        difference = r00[:, None] + rtt - 2.0 * r0t
        difference[:, 0] = 0.0
        running = np.cumsum(difference[:, 1:], axis=1)
        lags = np.arange(1, tau_max + 1)
        normalised = np.ones_like(difference)
        with np.errstate(divide="ignore", invalid="ignore"):
            normalised[:, 1:] = difference[:, 1:] * lags / np.maximum(running, 1e-30)
        normalised[~np.isfinite(normalised)] = 1.0
        for row, audible in zip(normalised, loud):
            if not audible:
                continue
            search = row[tau_min:tau_max + 1]
            under = np.flatnonzero(search < YIN_THRESHOLD)
            if under.size:
                # YIN's absolute threshold: the FIRST dip under it, then walk
                # down to that dip's local minimum, not the global minimum --
                # taking the global one is the classic octave error.
                index = tau_min + int(under[0])
                while index + 1 <= tau_max and row[index + 1] < row[index]:
                    index += 1
            else:
                continue
            lag = _parabolic(row, index)
            if lag > 0:
                candidate = sample_rate / lag
                if F0_MIN <= candidate <= F0_MAX:
                    f0.append(candidate)
    return _summarise(f0, int(block.shape[0]))


def estimate_ncc(signal: np.ndarray, sample_rate: int) -> dict:
    """A normalised cross-correlation tracker: peak-picking, not thresholding.

    Independent of YIN in window length, in what it maximises (a normalised
    correlation, not a normalised difference) and in its voicing decision (a
    peak height, not an absolute dip threshold).
    """
    hop = int(round(sample_rate * HOP_SECONDS))
    tau_min = max(2, int(np.floor(sample_rate / F0_MAX)))
    tau_max = int(np.ceil(sample_rate / F0_MIN))
    window = NCC_FRAME - tau_max
    block = _frames(signal, NCC_FRAME, hop)
    if block.shape[0] == 0:
        return {"frames": 0, "voiced_frames": 0, "median_f0_hz": None,
                "below_register_fraction": None}
    f0 = []
    gate = _silence_gate(signal)
    for start in range(0, block.shape[0], 512):
        chunk = np.ascontiguousarray(block[start:start + 512], dtype=np.float64)
        loud = np.sqrt((chunk * chunk).mean(axis=1)) > gate
        r00, rtt, r0t = _correlations(chunk, window, tau_max)
        with np.errstate(divide="ignore", invalid="ignore"):
            ncc = r0t / np.sqrt(np.maximum(r00[:, None] * rtt, 1e-30))
        ncc[~np.isfinite(ncc)] = 0.0
        for row, audible in zip(ncc, loud):
            if not audible:
                continue
            search = row[tau_min:tau_max + 1]
            peak = float(search.max())
            if peak < NCC_VOICING:
                continue
            # Shortest lag within NCC_SUBHARMONIC of the peak, then climb to
            # that peak's own local maximum: argmax alone reports tau/2.
            acceptable = np.flatnonzero(search >= NCC_SUBHARMONIC * peak)
            index = tau_min + int(acceptable[0])
            while index + 1 <= tau_max and row[index + 1] > row[index]:
                index += 1
            lag = _parabolic(row, index)
            if lag > 0:
                candidate = sample_rate / lag
                if F0_MIN <= candidate <= F0_MAX:
                    f0.append(candidate)
    return _summarise(f0, int(block.shape[0]))


def _summarise(f0: list[float], frames: int) -> dict:
    if not f0:
        return {"frames": frames, "voiced_frames": 0, "median_f0_hz": None,
                "below_register_fraction": None}
    values = np.asarray(f0)
    return {"frames": frames, "voiced_frames": int(values.size),
            "median_f0_hz": float(np.median(values)),
            "below_register_fraction": float(np.mean(values < REGISTER_HZ))}


def degeneracy(signal: np.ndarray, sample_rate: int, voiced_fraction: float | None) -> dict:
    """Is this a voice at all: DC, zero-crossing rate, envelope, voicing."""
    if signal.size == 0:
        return {"samples": 0, "degenerate": True, "reason": "empty render"}
    dc = float(signal.mean())
    centred = signal - dc
    crossings = int(np.count_nonzero(np.diff(np.signbit(centred))))
    zcr = crossings * sample_rate / max(1, signal.size - 1)
    frame = int(round(sample_rate * ENVELOPE_FRAME_SECONDS))
    usable = signal.size // frame * frame
    if usable >= 2 * frame:
        rms = np.sqrt((centred[:usable].reshape(-1, frame) ** 2).mean(axis=1))
        floor = max(float(rms.min()), 1e-12)
        ratio = float(rms.max() / floor)
    else:
        ratio = 0.0
    degenerate = zcr < DEGENERATE_ZCR or ratio < DEGENERATE_ENVELOPE
    return {"samples": int(signal.size), "dc_offset": dc,
            "zero_crossing_rate_hz": float(zcr),
            "zcr_in_published_band": bool(ZCR_BAND[0] <= zcr <= ZCR_BAND[1]),
            "envelope_ratio": ratio,
            "envelope_in_published_band": bool(ENVELOPE_BAND[0] <= ratio <= ENVELOPE_BAND[1]),
            "voiced_frame_fraction": voiced_fraction,
            "degenerate": bool(degenerate),
            "peak": float(np.abs(signal).max()), "rms": float(np.sqrt((signal ** 2).mean()))}


def measure(signal: np.ndarray, sample_rate: int) -> dict:
    yin = estimate_yin(signal, sample_rate)
    ncc = estimate_ncc(signal, sample_rate)
    voiced = None
    if yin["frames"]:
        voiced = yin["voiced_frames"] / yin["frames"]
    return {"yin": yin, "ncc": ncc,
            "degeneracy": degeneracy(signal, sample_rate, voiced)}


def _separated(left: float | None, right: float | None) -> bool:
    """Complete register separation: 1.00 to 0.00, in either direction."""
    if left is None or right is None:
        return False
    return (left == 1.0 and right == 0.0) or (left == 0.0 and right == 1.0)


def _shift(reference: float | None, candidate: float | None) -> float | None:
    if reference is None or candidate is None or reference == 0.0:
        return None
    return abs(candidate / reference - 1.0)


def _extremity(reference: dict, candidate: dict) -> tuple[float | None, float | None]:
    """How far this case moved, as the two quantities the criterion thresholds.

    `separation` is the SMALLER of the two trackers' absolute change in the
    below-165 Hz fraction, and `shift` the SMALLER of their median-F0 shifts,
    so both are "the weaker tracker's evidence". Reported alongside the
    pass/fail verdict because the criterion is a conjunction of two hard
    thresholds and a reader deciding whether to ship needs to see the
    near-misses, not only which side of the line each case fell.
    """
    separations, shifts = [], []
    for estimator in ("yin", "ncc"):
        low, high = (reference[estimator]["below_register_fraction"],
                     candidate[estimator]["below_register_fraction"])
        separations.append(None if low is None or high is None else abs(high - low))
        shifts.append(_shift(reference[estimator]["median_f0_hz"],
                             candidate[estimator]["median_f0_hz"]))
    if any(value is None for value in separations) or any(value is None for value in shifts):
        return None, None
    return min(separations), min(shifts)


def verdict(reference: dict, candidate: dict) -> dict:
    """The committed criterion, and the reasons a case is not counted at all.

    A render the degeneracy screen rejects, or an estimator with no voiced
    frame to take a median over, yields **not comparable** -- never
    "unchanged". Counting a silence as unchanged is how an instrument reports
    a broken profile as a passing one.
    """
    reasons = []
    if reference["degeneracy"]["degenerate"]:
        reasons.append("reference degenerate")
    if candidate["degeneracy"]["degenerate"]:
        reasons.append("candidate degenerate")
    for estimator in ("yin", "ncc"):
        if reference[estimator]["voiced_frames"] == 0:
            reasons.append(f"reference has no voiced frame under {estimator.upper()}")
        if candidate[estimator]["voiced_frames"] == 0:
            reasons.append(f"candidate has no voiced frame under {estimator.upper()}")
    shifts = {estimator: _shift(reference[estimator]["median_f0_hz"],
                                candidate[estimator]["median_f0_hz"])
              for estimator in ("yin", "ncc")}
    separations = {estimator: _separated(reference[estimator]["below_register_fraction"],
                                         candidate[estimator]["below_register_fraction"])
                   for estimator in ("yin", "ncc")}
    separation_min, shift_min = _extremity(reference, candidate)
    if reasons:
        return {"verdict": "not comparable", "reasons": reasons,
                "median_shift": shifts, "register_separated": separations,
                "separation_min": separation_min, "shift_min": shift_min}
    changed = (all(separations.values())
               and all(value is not None and value > MEDIAN_SHIFT for value in shifts.values()))
    return {"verdict": "changed" if changed else "unchanged", "reasons": [],
            "median_shift": shifts, "register_separated": separations,
            "separation_min": separation_min, "shift_min": shift_min}


def collect(root: pathlib.Path, cases: list[str] | None) -> dict[str, pathlib.Path]:
    if root.is_file():
        return {root.stem: root}
    found = {}
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        render = entry / "pcm_freerun.f32"
        if render.is_file():
            found[entry.name] = render
    if cases:
        missing = [case for case in cases if case not in found]
        if missing:
            raise SystemExit(f"{root}: no render for {missing}")
        found = {case: found[case] for case in cases}
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--reference", type=pathlib.Path, required=True)
    parser.add_argument("--candidate", type=pathlib.Path, required=True)
    parser.add_argument("--name-reference", default="reference")
    parser.add_argument("--name-candidate", default="candidate")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--report", type=pathlib.Path, default=None)
    arguments = parser.parse_args(argv)

    left = collect(arguments.reference, arguments.cases)
    right = collect(arguments.candidate, arguments.cases)
    shared = [case for case in left if case in right]
    if not shared:
        raise SystemExit("no case is present on both sides")

    results = []
    for case in shared:
        reference = measure(read_render(left[case]), arguments.sample_rate)
        candidate = measure(read_render(right[case]), arguments.sample_rate)
        results.append({"case": case, "reference": reference, "candidate": candidate,
                        **verdict(reference, candidate)})

    changed = [r for r in results if r["verdict"] == "changed"]
    excluded = [r for r in results if r["verdict"] == "not comparable"]
    counted = [r for r in results if r["verdict"] != "not comparable"]

    header = (f"{'case':24} {'YIN ref':>9} {'YIN cand':>9} {'NCC ref':>9} {'NCC cand':>9} "
              f"{'sep':>5} {'shift':>7}  verdict")
    print(f"{arguments.name_reference}  ->  {arguments.name_candidate}")
    print(header)
    results.sort(key=lambda r: (-(r["separation_min"] or -1.0), -(r["shift_min"] or -1.0)))
    for r in results:
        ry, cy = r["reference"]["yin"], r["candidate"]["yin"]
        rn, cn = r["reference"]["ncc"], r["candidate"]["ncc"]

        def hz(value):
            return f"{value:9.1f}" if value is not None else "        -"

        def fr(value):
            return f"{value:9.2f}" if value is not None else "        -"

        note = r["verdict"]
        if r["reasons"]:
            note += " (" + "; ".join(r["reasons"]) + ")"

        def small(value, width):
            return f"{value:{width}.2f}" if value is not None else " " * (width - 1) + "-"

        print(f"{r['case']:24} {hz(ry['median_f0_hz'])} {hz(cy['median_f0_hz'])} "
              f"{hz(rn['median_f0_hz'])} {hz(cn['median_f0_hz'])} "
              f"{small(r['separation_min'], 5)} {small(r['shift_min'], 7)}  {note}")
    print(f"\n{len(changed)} of {len(counted)} changed "
          f"({len(excluded)} not comparable, excluded from the count)")
    for r in excluded:
        print(f"  excluded: {r['case']} -- {'; '.join(r['reasons'])}")

    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(json.dumps(
            {"reference": arguments.name_reference, "candidate": arguments.name_candidate,
             "sample_rate": arguments.sample_rate,
             "criterion": {"register_hz": REGISTER_HZ, "median_shift": MEDIAN_SHIFT,
                           "yin_threshold": YIN_THRESHOLD, "yin_frame": YIN_FRAME,
                           "ncc_voicing": NCC_VOICING, "ncc_frame": NCC_FRAME,
                           "hop_seconds": HOP_SECONDS},
             "changed": len(changed), "counted": len(counted), "excluded": len(excluded),
             "cases": results}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
