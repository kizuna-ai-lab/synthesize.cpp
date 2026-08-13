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
STAGE_REL_ABSMAX = 1.0e-3
#   measured worst 9.303e-05 (transformer_l7, all three cases) -> 10.8x headroom
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
# very different dynamic range: the semantic branch moves 0.004 -> 1.20 under
# the fault (300x) and the acoustic branch only 0.25 -> 1.15 (4.6x). Averaging
# them would blunt the sharper instrument with the duller one.
RECONSTRUCTION_P95_RELATIVE = {
    "semantic": 2.0e-2,
    #   measured p95 0.004103 (worst case) -> 4.9x headroom
    #   injected fault p95 1.601           -> 78x discrimination
    #   5x one bf16 unit, which is where the design says this branch's median
    #   belongs; it flips on only 4 of 101 frames so the tail stays contained.
    "acoustic": 5.0e-1,
    #   measured p95 0.2491 (worst case)   -> 2.0x headroom
    #   injected fault p95 1.741           -> 3.5x discrimination
    #   NOT at bf16 scale, and it cannot be. This branch aggregates fifteen
    #   stages, disagreement with the bf16 oracle grows monotonically down the
    #   cascade (4, 6, 20, 34, ... 74 of 101 frames), so almost every frame
    #   carries at least one flip and the branch sits in the FLIP TAIL at the
    #   MEDIAN (0.1317), not only at its p95. The design's predicted 0.18
    #   absolute L2 for this branch is 39x too small against a measured median
    #   of 7.014 absolute and must not be used.
}
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
    for branch, name in ((0, "semantic"), (1, "acoustic")):
        reference_rms = np.sqrt(np.mean(recon["oracle"][branch].astype(np.float64) ** 2))
        print(f"\n  reconstruction[{name}]  oracle rms {reference_rms:.4g}")
        gated = 0.0
        for label, left, right in (("port vs upstream-f32", "port", "upstream-f32"),
                                   ("port vs oracle      ", "port", "oracle"),
                                   ("upstream-f32 vs orcl", "upstream-f32", "oracle")):
            a = recon[left][branch].astype(np.float64)
            b = recon[right][branch].astype(np.float64)
            per_frame = np.linalg.norm(a - b, axis=-1)
            norms = np.linalg.norm(b, axis=-1)
            relative = per_frame / norms
            print(f"    {label}  per-frame L2 abs: median {np.median(per_frame):.4g} "
                  f"p95 {np.percentile(per_frame, 95):.4g} max {per_frame.max():.4g} "
                  f"| /||ref||: median {np.median(relative):.4g} "
                  f"p95 {np.percentile(relative, 95):.4g}")
            if right == "oracle" and left == "port":
                gated = float(np.percentile(relative, 95))
        bound = RECONSTRUCTION_P95_RELATIVE[name]
        verdict = "PASS" if gated <= bound else "FAIL"
        ok = ok and gated <= bound
        print(f"    [{verdict}] port-vs-oracle p95 relative {gated:.4g}; gate {bound:.3g}")

    # ---- codes: RECORDED, GATING NOTHING --------------------------------
    codes = {
        "port": read(port, case, "codes", np.int32).reshape(frames, 16),
        "upstream-f32": read(upstream, case, "codes", np.int32).reshape(frames, 16),
        "oracle": codes_oracle.reshape(frames, 16),
    }
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", required=True, type=pathlib.Path)
    parser.add_argument("--upstream-f32", required=True, type=pathlib.Path, dest="upstream")
    parser.add_argument("--oracle", required=True, type=pathlib.Path)
    parser.add_argument("--case", action="append", required=True)
    args = parser.parse_args()

    ok = True
    for case in args.case:
        ok = check_case(args.port, args.upstream, args.oracle, case) and ok
    print(f"\n{'ALL CASES PASS' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
