#!/usr/bin/env python3
"""Validate the Qwen3-TTS codec encoder and its split RVQ, stage by stage.

THE CODES ARE NOT THE GATE, and this file exists because of that. The design's
fourth erratum (2026-08-13) dropped the plain-equality gate on
``codes/reference.i32``: the oracle's RVQ codebook is bfloat16 and the
converter's is float32, so the two tables disagree on 4.04% of frame-stage
decisions, upstream disagrees with ITSELF depending on whether ``dtype`` is
passed at load, and equality here would test a load-time keyword argument. What
gates instead is the continuous chain behind the codes, compared stage by stage.

THREE ARTIFACT SETS, ALL IN THE SAME ON-DISK ELEMENT ORDER, so nothing here
transposes anything:

    port          tests/qwen3_tts_codec_encoder_driver.cpp, F32 throughout
    upstream-f32  scripts/dump_reference_qwen3_tts_codec_encoder_float32.py
    oracle        the committed bfloat16 dump

and therefore two differences that mean different things:

    port vs upstream-f32     this port's own arithmetic. A defect lives here.
    upstream-f32 vs oracle   the oracle's own bf16 rounding. Nothing to fix.

EVERY NUMBER NAMES ITS NORMALIZER. ``rel_absmax`` is ``max|a-b|`` divided by the
REFERENCE tensor's own absolute maximum -- one scalar scale for the whole
tensor, not a per-element ``|d|/|x|``. ``rel_rms`` is ``rms(a-b) /
rms(reference)``. They are different quantities and quoting one against the
other's threshold is a review round this file exists to avoid repeating.

WHAT THE RECONSTRUCTION'S EXACT ZERO DOES AND DOES NOT SAY. ``port vs
upstream-f32`` on ``rvq_reconstruction`` measures 0 exactly, and that is a real
measurement of a comparison that can fail -- under an injected fault it reads
1.115. But it decomposes into three conjuncts at once: the codes agree, the two
float32 codebook tables are bit-identical **by construction** (convert-qwen3-tts
.py:388 bakes ``embed_sum.to(f32) / cluster_usage.to(f32).clamp(1e-5)``, and
upstream-f32 performs that identical division on the identical float32
safetensors at runtime -- same operands, same IEEE op), and the accumulation
order matches. So the zero is a CONSISTENCY CHECK, not independent evidence
about the codebook conversion: the two tables are the same arithmetic and could
not have differed. The load-bearing independent evidence is the continuous
chain, where the worst stage sits 42x below a single bf16 rounding across an
eleven-convolution SEANet stack, an eight-layer transformer, a downsampler and
sixteen RVQ residual stages. Quote that, not the zero.

THE UNDERSCORE IN THIS FILENAME IS LOAD-BEARING, and it cost six tasks to learn.
tests/python/test_tolerance_coverage.py globs ``scripts/validate-qwen3-tts-*.py``
and matches each stem, minus the family prefix, against the measured stage names
in tests/tolerances/qwen3-tts.json. The stage is ``codec_encoder``, so the file
must be ``validate-qwen3-tts-codec_encoder.py``. It first landed (6279830) as
``...-codec-encoder.py``, whose stem ``codec-encoder`` matched no stage and left
that check red for BOTH variants until 2026-08-14 -- unnoticed because the check
was not registered with CTest. It is registered now (synthesize-tolerance-
coverage), so a future separator slip fails in the standard unit gate instead.

Usage:

    scripts/envs/qwen3-tts/.venv/bin/python \\
        scripts/validate-qwen3-tts-codec_encoder.py \\
        --port <driver output root> --upstream-f32 <f32 dump root> \\
        --oracle build/goldens/qwen3-tts/qwen3-tts-12hz-0-6b-base \\
        --case base-icl-en --case base-ref-min --case base-text-short
"""

import argparse
import json
import pathlib
import sys

import numpy as np

# 2^-8: bfloat16 carries 8 significand bits, so a value it represents sits
# within this relative distance of the float32 one. The scale a port/oracle
# difference is EXPECTED to land at, and the scale a port/upstream-f32
# difference must land far below.
BF16_UNIT = 2.0 ** -8

# ---------------------------------------------------------------------------
# THE THRESHOLDS, and the reasoning, because whoever wires this into CTest
# inherits both.
#
# Each is stated with the measurement it clears, and the injected fault it
# still catches. The fault is the one Task 5 used: upstream's own symmetric
# (non-causal) MimiConv1d padding split, which builds the same shapes from the
# same weights and a different encoder.
#
# THE PRIMARY GATE IS THE CONTINUOUS CHAIN, not either code or reconstruction
# comparison. It is the only one of the three that is independent of the
# codebook table being shared by construction.
# THESE THREE CONSTANTS ARE FALLBACKS, NOT THE SOURCE. Since 2026-08-14 the
# enforced values are READ FROM tests/tolerances/qwen3-tts.json (--tolerances),
# the way scripts/validate-qwen3-tts-replay.py has always resolved its cell.
# They were module constants first, and the tolerance grid was then written by
# transcribing them -- so the JSON was a copy of the enforced numbers rather
# than their source, and editing the published gate changed nothing. Now the
# JSON is the source and these are used only when no tolerance file is
# supplied; load_gates() refuses to run if the two disagree, so a transcription
# can no longer drift silently.
STAGE_REL_ABSMAX = 1.0e-3
#   measured worst 9.303e-05 (transformer_l7, all three cases) -> 10.7x headroom
#   injected fault 1.649e+00                                   -> 1650x discrimination
#   and 1e-3 is still a quarter of one bf16 unit, so it cannot pass a port that
#   merely "rounds like bf16" -- it demands float32 agreement with upstream.
#
# THE RECONSTRUCTION, per branch, as a PERCENTILE and never a maximum.
#
# A maximum at bf16 scale is unsatisfiable by a correct port and this is
# measured, not predicted: a single flipped code displaces the reconstruction
# by about one full codebook-row separation, so the semantic branch's per-frame
# maximum is 235.8 absolute against a median of 0.55. Four of 101 frames flip.
# No bf16-scale maximum survives that, and widening one to fit would destroy the
# gate. The statistic is therefore the 95th percentile of the per-frame
# L2 deviation, RELATIVE to the reference frame's own L2 norm.
#
# THE TWO BRANCHES GET SEPARATE KEYS AND ARE NEVER AVERAGED. They are an order
# of magnitude apart (rms 13.5 against 3.11) and, more importantly, they have
# very different dynamic range. Under the injected fault the semantic branch's
# p95 moves 0.004103 -> 1.601, a 390x move, while the acoustic branch's moves
# only 0.2491 -> 1.741, a 7.0x move. Averaging them would blunt the sharper
# instrument with the duller one. (This read "0.004 -> 1.20 (300x)" and
# "0.25 -> 1.15 (4.6x)" until 2026-08-14, two figures that match neither the
# fault p95 recorded three lines below nor anything else in the file; they are
# replaced by the recorded ones rather than left as an unattributable
# statistic.)
# REDESIGNED 2026-08-17 BY PLAN 4 TASK 3, AND THE MASK IS THE WHOLE CHANGE.
#
# The statistic below is no longer the p95 over every frame. It is the p95 over
# the frames the ORACLE'S OWN CODEBOOK did not flip -- the frames where
# upstream-f32 and the bf16 oracle select the same code. The threshold numbers
# did not have to move to make that work; what moved is which frames they are
# computed over.
#
# WHY THE OLD ONE HAD TO GO. It sat on a cliff: a 95th percentile only enters
# the flipped-frame tail once more than 5% of frames carry a flip, so
# base-ref-max at 5.333% read 0.3478 against this same 2.0e-2 while the three
# calibration cases at 0.00/3.96/3.96% read 0.004. That was recorded as
# `gate_scope_warning` and left standing, because widening 2.0e-2 to 0.35 would
# have left the injected fault 4.6x above the gate instead of 80x.
#
# WHY THE MASK IS TAKEN FROM upstream-f32 VS THE ORACLE AND NOT FROM THE PORT.
# This is the half that decides whether the redesign works at all, and the
# obvious mask -- condition on the frames where the PORT and the oracle agree --
# was measured and FAILS. Under the injected symmetric-pad fault the port flips
# 89-100% of frames, so conditioning on port-vs-oracle agreement selects exactly
# the frames the fault did not reach: the masked statistic reads 0.003511
# against a clean 0.003576, i.e. discrimination ~1x, and on base-ref-min and on
# every acoustic branch it is undefined because no frame survives. A mask that
# mentions the port cannot see a fault in the port. The upstream-f32-vs-oracle
# mask contains no port at all, so the fault cannot move it: it drops the same
# ~5% of frames clean and faulted, and the fault's frames stay in.
#
# MEASURED, over five cases and two recordings, port fed the oracle's own
# bfloat16 waveform:
#
#   branch     clean p95 (worst)   faulted p95 (best)   discrimination
#   semantic   0.004103            1.601                390x - 474x
#   acoustic   0.003818            1.734                586x - 644x
#
# against the >=80x the redesign was required to keep on the semantic branch and
# the 3.5x it was required not to lose on the acoustic one. The acoustic branch
# improves by a factor of ~170, because the old unmasked figure (0.2491) sat in
# the flip tail at the MEDIAN, not only at the p95 -- that is what the removed
# comment below described, and the mask is what fixes it rather than a wider
# bound.
#
# THE TWO BRANCHES STILL GET SEPARATE KEYS AND ARE STILL NEVER AVERAGED. They
# remain an order of magnitude apart in rms (13.5 against 3.11).
RECONSTRUCTION_P95_RELATIVE = {
    "semantic": 2.0e-2,
    #   codebook-masked p95 0.004103 (worst of five) -> 4.9x headroom
    #   injected fault      1.601 (best of five)     -> 80.05x discrimination
    #   Unchanged from the pre-redesign value, and that is not a coincidence:
    #   base-ref-min has no codebook flips at all, so its masked and unmasked
    #   figures are the same number. What changed is that base-ref-max now
    #   reads 0.003756 here instead of 0.3478, so the gate holds on all five.
    "acoustic": 5.0e-1,
    #   codebook-masked p95 0.003818 here (worst of five) -> 131x headroom
    #   codebook-masked p95 0.181149 in the OTHER consumer -> 2.76x headroom
    #   injected fault      1.734 (best of five)          -> 3.5x discrimination
    #
    #   A 25x TIGHTENING TO 2.0e-2 WAS MEASURED, ATTEMPTED AND WITHDRAWN. The
    #   masked figure this script reads (0.003818) is at bf16 scale and would
    #   justify it. tests/qwen3_tts_icl_real.cpp reads 0.181149 for the same
    #   statistic on the same case, because it feeds the port the WAV where this
    #   script feeds it the oracle's own bfloat16-rounded waveform, and this
    #   branch aggregates fifteen stages so a 2.954e-03 input rounding compounds
    #   down the cascade. The semantic branch does not have that problem: the two
    #   consumers agree there to six digits, 0.003576 against 0.00357636.
    #
    #   The tightening passed all five cases here and failed the C++ arm on its
    #   first run, which is exactly what `registered_consumers` in the tolerance
    #   file exists to catch.
}

# Gated separately from the reconstruction, which is Plan 3's ruling carried out:
# "gate the flip rate separately from the reconstruction error on non-flipped
# frames".
#
# MEASURED, same five cases:
#   semantic  clean 0.000 / 1.705 / 3.960 / 3.960 / 5.333 %
#             fault 89.205 / 96.040 / 96.040 / 97.333 / 100.000 %
# No overlap, and the gap is 16.7x. The threshold below sits between them:
# 3.75x above the worst clean case and 4.46x below the best faulted one.
#
# ONLY THE SEMANTIC BRANCH IS GATED. The acoustic flip rate is RECORDED AND
# UNGATED, and the reason is a measurement rather than caution: it reads
# 75.568 / 79.208 / 79.208 / 83.200 / 84.615 % clean against 100.000% faulted,
# a separation of 1.18x. Fifteen acoustic columns mean a frame counts as
# flipped if ANY of them does, so the clean rate is already near saturation and
# no threshold can separate the two populations. A gate there would be one that
# cannot fail, which this suite refuses to commit.
#
# A flip rate is bounded above by 100%, so its discrimination is structurally
# limited in a way the reconstruction's is not. It is a second instrument, not
# a replacement for the first.
SEMANTIC_FLIP_RATE = 0.20
#
# HEADROOM IS ~2x ON THE ACOUSTIC BRANCH AND IT IS DRAWN FROM ONE RECORDING.
# base-ref-min is a byte-exact prefix of base-icl-en and base-text-short is the
# same clip again at full length: three cases, one speaker, one microphone. A
# second speaker would be independent corroboration; there is not one, so these
# two numbers should be revisited the first time a second reference clip exists
# rather than treated as settled by three passing cases.
# ---------------------------------------------------------------------------

STAGES = (
    ["waveform"]
    + [f"seanet_stage{i}" for i in range(4)]
    + ["seanet_tail"]
    + [f"transformer_l{i}" for i in range(8)]
    + ["downsample", "latents"]
    + [f"rvq_residual_s{i:02d}" for i in range(16)]
    + ["rvq_reconstruction"]
)


def read(root: pathlib.Path, case: str, name: str, dtype=np.float32) -> np.ndarray:
    suffix = "i32" if dtype is np.int32 else "f32"
    path = root / case / "codec_encoder" / f"{name}.{suffix}"
    if not path.exists():
        # The driver writes flat into whatever output directory it is handed,
        # so a tree produced by pointing it at `<root>/<case>` looks like this.
        path = root / case / f"{name}.{suffix}"
    if not path.exists():
        raise SystemExit(f"{case}: neither layout has {name}.{suffix} under {root / case}")
    return np.fromfile(path, dtype=dtype)


def widths(oracle: pathlib.Path, case: str) -> dict:
    """The quantizer geometry, read from the oracle's own conventions.json.

    Not hardcoded: `projected_width` is 256 for this checkpoint and a hardcoded
    256 is exactly the sort of constant that survives a re-cut and silently
    reshapes every comparison below it.
    """
    with open(oracle / case / "codec_encoder" / "conventions.json") as handle:
        conventions = json.load(handle)
    geometry = conventions["geometry"]["quantizer_widths"]
    return {
        "projected": int(geometry["projected_width"]),
        "latent": int(geometry["latent_width"]),
        "codebook_size": int(geometry["codebook_size"]),
        "min_distance_margin": float(conventions["tie_margin"]["min_distance_margin"]),
    }


def deviation(a: np.ndarray, reference: np.ndarray) -> dict:
    if a.shape != reference.shape:
        raise SystemExit(f"shape mismatch: {a.shape} vs {reference.shape}")
    delta = a.astype(np.float64) - reference.astype(np.float64)
    absmax = float(np.max(np.abs(reference))) or 1.0
    rms_ref = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2))) or 1.0
    return {
        "rel_absmax": float(np.max(np.abs(delta))) / absmax,
        "rel_rms": float(np.sqrt(np.mean(delta ** 2))) / rms_ref,
    }


def check_case(port: pathlib.Path, upstream: pathlib.Path, oracle: pathlib.Path, case: str) -> bool:
    print(f"\n{'=' * 78}\n{case}\n{'=' * 78}")
    geometry = widths(oracle, case)
    projected = geometry["projected"]
    ok = True

    # ---- the continuous chain: THE GATE --------------------------------
    print(f"{'stage':<22} {'port vs upstream-f32':>28}   {'upstream-f32 vs oracle':>28}")
    print(f"{'':<22} {'rel_absmax':>13} {'rel_rms':>14}   {'rel_absmax':>13} {'rel_rms':>14}")
    worst = ("", 0.0)
    for stage in STAGES:
        ref_oracle = read(oracle, case, stage)
        ref_upstream = read(upstream, case, stage)
        # The port READ waveform.f32; it is its input, not its output, so the
        # port column is the file itself and the row is excluded from the gate.
        values = ref_oracle if stage == "waveform" else read(port, case, stage)
        against_upstream = deviation(values, ref_upstream)
        against_oracle = deviation(ref_upstream, ref_oracle)
        if stage != "waveform" and against_upstream["rel_absmax"] > worst[1]:
            worst = (stage, against_upstream["rel_absmax"])
        print(f"{stage:<22} {against_upstream['rel_absmax']:>13.3e} "
              f"{against_upstream['rel_rms']:>14.3e}   "
              f"{against_oracle['rel_absmax']:>13.3e} {against_oracle['rel_rms']:>14.3e}")
    verdict = "PASS" if worst[1] <= STAGE_REL_ABSMAX else "FAIL"
    ok = ok and worst[1] <= STAGE_REL_ABSMAX
    print(f"  [{verdict}] worst stage {worst[0]} at rel_absmax {worst[1]:.3e} "
          f"= {worst[1] / BF16_UNIT:.4g} x one bf16 unit; gate {STAGE_REL_ABSMAX:.1e}")

    # ---- the reconstruction, per branch, per frame ----------------------
    codes_oracle = read(oracle, case, "codes", np.int32)
    frames = codes_oracle.size // 16
    shape = (2, frames, projected)
    recon = {
        "port": read(port, case, "rvq_reconstruction").reshape(shape),
        "upstream-f32": read(upstream, case, "rvq_reconstruction").reshape(shape),
        "oracle": read(oracle, case, "rvq_reconstruction").reshape(shape),
    }
    # Read BEFORE the reconstruction loop rather than after it, because the loop
    # now needs the flip mask. Nothing else about the codes block moved; it is
    # still recorded as evidence below and still gates nothing on its own.
    codes = {
        "port": read(port, case, "codes", np.int32).reshape(frames, 16),
        "upstream-f32": read(upstream, case, "codes", np.int32).reshape(frames, 16),
        "oracle": codes_oracle.reshape(frames, 16),
    }

    # The per-frame flip mask, per branch.
    #
    # The reconstruction is [2, frames, projected] with branch 0 semantic and
    # branch 1 acoustic; the codes are [frames, 16] with column 0 semantic and
    # columns 1..15 acoustic. So a frame's semantic branch is flipped when its
    # column 0 differs, and its acoustic branch is flipped when ANY of the
    # fifteen acoustic columns does. `differ` is exactly what the codes block
    # below already computes -- what was missing was the mask, not the data.
    def flip_mask(left: str, right: str, branch: int) -> np.ndarray:
        differ = codes[left] != codes[right]
        return differ[:, 0] if branch == 0 else differ[:, 1:].any(axis=1)

    # The mask is the SAME for every pairing and for the whole case: it is the
    # bf16 codebook's own disagreement with float32, and no pairing's operands
    # change it. Computed once, per branch, so it cannot accidentally be taken
    # from a pairing that mentions the port.
    for branch, name in ((0, "semantic"), (1, "acoustic")):
        reference_rms = np.sqrt(np.mean(recon["oracle"][branch].astype(np.float64) ** 2))
        codebook_mask = flip_mask("upstream-f32", "oracle", branch)
        dropped = int(codebook_mask.sum())
        print(f"\n  reconstruction[{name}]  oracle rms {reference_rms:.4g}  "
              f"codebook mask drops {dropped} of {codebook_mask.size} frames")
        gated = float("nan")
        for label, left, right in (("port vs upstream-f32", "port", "upstream-f32"),
                                   ("port vs oracle      ", "port", "oracle"),
                                   ("upstream-f32 vs orcl", "upstream-f32", "oracle")):
            a = recon[left][branch].astype(np.float64)
            b = recon[right][branch].astype(np.float64)
            per_frame = np.linalg.norm(a - b, axis=-1)
            norms = np.linalg.norm(b, axis=-1)
            relative = per_frame / norms
            kept = relative[~codebook_mask]
            masked = float(np.percentile(kept, 95)) if kept.size else float("nan")
            print(f"    {label}  per-frame L2 abs: median {np.median(per_frame):.4g} "
                  f"p95 {np.percentile(per_frame, 95):.4g} max {per_frame.max():.4g} "
                  f"| /||ref||: median {np.median(relative):.4g} "
                  f"p95 {np.percentile(relative, 95):.4g} "
                  f"| CODEBOOK-MASKED p95 {masked:.4g} over {kept.size}")
            if right == "oracle" and left == "port":
                gated = masked
        bound = RECONSTRUCTION_P95_RELATIVE[name]
        # A case in which the codebook flips EVERY frame leaves the statistic
        # undefined, and that is a refusal rather than a pass. It has never been
        # observed -- the mask drops 0 to 20 of 13 to 375 frames across the five
        # committed cases -- but a NaN compares false against every bound, so
        # without this the gate would silently stop gating.
        if not np.isfinite(gated):
            print(f"    [FAIL] {name}: the codebook mask left no frames, so the masked p95 is "
                  "undefined; this is a refusal, not a pass")
            ok = False
        else:
            verdict = "PASS" if gated <= bound else "FAIL"
            ok = ok and gated <= bound
            print(f"    [{verdict}] port-vs-oracle CODEBOOK-MASKED p95 relative {gated:.4g}; "
                  f"gate {bound:.3g}")

        # The flip rate, gated on the semantic branch and recorded on the other.
        port_flips = flip_mask("port", "oracle", branch)
        rate = float(port_flips.sum()) / float(port_flips.size)
        if name == "semantic":
            verdict = "PASS" if rate <= SEMANTIC_FLIP_RATE else "FAIL"
            ok = ok and rate <= SEMANTIC_FLIP_RATE
            print(f"    [{verdict}] semantic flip rate {100.0 * rate:.3f}%; "
                  f"gate {100.0 * SEMANTIC_FLIP_RATE:.1f}%")
        else:
            print(f"    [--] acoustic flip rate {100.0 * rate:.3f}% -- RECORDED, GATING NOTHING. "
                  "Fifteen columns mean a frame counts as flipped if any one does, so the clean "
                  "rate is already near saturation (75.6-84.6%) against 100.0% under the injected "
                  "fault: a separation of 1.18x, which no threshold can use.")

    # ---- codes: RECORDED, GATING NOTHING --------------------------------
    print("\n  codes -- recorded as evidence; NOTHING below branches on it, and no")
    print("  threshold is derived from it. The erratum's 4.04%/12.73% bound the")
    print("  codebook's contribution alone and are a floor on divergence, not a budget.")
    for label, left, right in (("port vs oracle      ", "port", "oracle"),
                               ("port vs upstream-f32", "port", "upstream-f32"),
                               ("upstream-f32 vs orcl", "upstream-f32", "oracle")):
        differ = codes[left] != codes[right]
        per_stage = differ.sum(axis=0)
        print(f"    {label}: agree {100.0 * (1 - differ.sum() / differ.size):.3f}% "
              f"({differ.sum()} of {differ.size} differ); semantic column "
              f"{per_stage[0]} of {frames}; per column {list(per_stage)}")

    # ---- the tie margin: also evidence ----------------------------------
    #
    # The port's grid is GROUP-FASTEST like its codes; the oracle's
    # rvq_distance_margin.f32 is C-order [stages, frames], i.e. stage-major.
    # The driver transposes on write, so both are stage-major here. Compared
    # per stage rather than elementwise: near the narrowest decisions the margin
    # is a difference of two nearly equal float32 squared distances and is not
    # resolvable, which is itself why ~4% of codes flip.
    margin_port = read(port, case, "rvq_distance_margin").reshape(16, frames)
    margin_oracle = read(oracle, case, "rvq_distance_margin").reshape(16, frames)
    stage_ratio = np.median(margin_port, axis=1) / np.median(margin_oracle, axis=1)
    print(f"\n  tie margin: port narrowest {margin_port.min():.6g}, oracle narrowest "
          f"{margin_oracle.min():.6g}, conventions.json min_distance_margin "
          f"{geometry['min_distance_margin']:.6g}")
    print(f"    per-stage median-margin ratio port/oracle: min {stage_ratio.min():.4g} "
          f"max {stage_ratio.max():.4g} (a transposed grid on either side would "
          f"scatter this away from 1)")
    differ = (codes["port"] != codes["oracle"]).T  # -> [stages, frames]
    if differ.any():
        # NAMED PRECISELY, because the brief's phrase "sit inside their frame's
        # margin" and this statistic are not the same thing. This is: the
        # oracle's own decision margin AT the disagreeing decisions, against the
        # same margin where the two agree. A flip concentrated at narrow margins
        # is the signature of rounding rather than of a wrong computation.
        print(f"    oracle margin at the {int(differ.sum())} disagreeing decisions: median "
              f"{np.median(margin_oracle[differ]):.6g}; where they agree: median "
              f"{np.median(margin_oracle[~differ]):.6g}")
    return ok


def load_gates(path: pathlib.Path, variant: str, profile: str, stage_name: str):
    """Resolve the enforced gates from the committed tolerance grid.

    The grid is the source; this module's constants are a fallback. Both are
    checked against each other and a disagreement is fatal rather than silently
    resolved in favour of either -- the failure this guards against is a
    transcribed copy drifting from the number a reviewer actually approved,
    which is how the grid came to exist in the first place.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    try:
        probes = document["variants"][variant]["profiles"][profile]["stages"][stage_name]["probes"]
    except KeyError as missing:
        raise SystemExit(f"{path}: no {variant}/{profile}/{stage_name} cell ({missing})") from None

    chain = float(probes["codec.chain"]["rel_absmax"])
    reconstruction = {
        branch: float(probes[f"codec.rvq_reconstruction.{branch}"]["p95_relative_l2"])
        for branch in ("semantic", "acoustic")
    }
    # Added by Plan 4 Task 3 and read through the same fatal-disagreement
    # discipline as the other two, because a threshold that only one side knows
    # about is the drift this function exists to catch.
    flip_rate = float(probes["codec.semantic_flip_rate"]["max_fraction"])

    drift = []
    if chain != STAGE_REL_ABSMAX:
        drift.append(f"chain rel_absmax: file {chain} vs module {STAGE_REL_ABSMAX}")
    for branch, value in reconstruction.items():
        if value != RECONSTRUCTION_P95_RELATIVE[branch]:
            drift.append(
                f"{branch} p95: file {value} vs module {RECONSTRUCTION_P95_RELATIVE[branch]}"
            )
    if flip_rate != SEMANTIC_FLIP_RATE:
        drift.append(f"semantic flip rate: file {flip_rate} vs module {SEMANTIC_FLIP_RATE}")
    if drift:
        raise SystemExit(
            "the committed grid and this module's fallback constants disagree, which means one of "
            "them was edited alone:\n  " + "\n  ".join(drift)
        )
    return chain, reconstruction, flip_rate


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", required=True, type=pathlib.Path)
    parser.add_argument("--upstream-f32", required=True, type=pathlib.Path, dest="upstream")
    parser.add_argument("--oracle", required=True, type=pathlib.Path)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument(
        "--tolerances",
        type=pathlib.Path,
        default=pathlib.Path("tests/tolerances/qwen3-tts.json"),
        help="The committed grid these gates come from. Pass --no-tolerances to fall back to "
             "this module's constants.",
    )
    parser.add_argument(
        "--no-tolerances",
        action="store_true",
        help="Ignore the tolerance file and use this module's fallback constants.",
    )
    parser.add_argument("--variant", default="qwen3-tts-12hz-0-6b-base")
    parser.add_argument("--profile", default="BF16")
    parser.add_argument("--stage", default="codec_encoder")
    args = parser.parse_args()

    global STAGE_REL_ABSMAX, RECONSTRUCTION_P95_RELATIVE, SEMANTIC_FLIP_RATE
    if not args.no_tolerances:
        STAGE_REL_ABSMAX, RECONSTRUCTION_P95_RELATIVE, SEMANTIC_FLIP_RATE = load_gates(
            args.tolerances, args.variant, args.profile, args.stage
        )

    ok = True
    for case in args.case:
        ok = check_case(args.port, args.upstream, args.oracle, case) and ok
    print(f"\n{'ALL CASES PASS' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
