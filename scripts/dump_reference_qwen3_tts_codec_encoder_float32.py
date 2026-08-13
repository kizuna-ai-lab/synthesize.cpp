#!/usr/bin/env python3
"""Dump Qwen3-TTS-Base's codec encoder from upstream's OWN module, in float32.

WHY THIS EXISTS, AND WHY IT IS NOT A FLAG ON ITS SIBLING.

``dump_reference_qwen3_tts_codec_encoder.py`` publishes the committed oracle,
and that oracle is **bfloat16 end to end** -- activations and the RVQ codebook
alike. A port/oracle difference therefore mixes two unrelated quantities: what
this port got wrong, and what bf16 rounding did. Neither is recoverable from the
other, and the design's fourth erratum (2026-08-13) removed the plain-equality
gate on the codes precisely because of it.

This script runs the identical upstream module in float32 on the identical
input, which splits the difference in two:

    port          vs  upstream-f32    this port's own arithmetic. A defect here.
    upstream-f32  vs  oracle          the oracle's own bf16 rounding. Nothing to fix.

It is a separate file rather than a ``--dtype`` flag on the sibling because the
sibling cannot be made to do this by adding one:

  * its ``from_pretrained`` call hardcodes ``dtype=torch.bfloat16`` and
    ``parse_args`` exposes no dtype argument at all;
  * it compares its own ``codes.i32`` against the committed
    ``codes/reference.i32`` and raises ``SystemExit`` on any difference, writing
    nothing. A float32 run MUST trip that -- 760 of 1616 codes differ on
    ``base-icl-en`` -- and ``--allow-unverified`` does not help, because it
    covers only the case where the reference file is *absent*, not where it
    differs;
  * writing into the committed oracle's own tree would overwrite the bf16
    artifacts the rest of the plan reads.

Making the sibling capable would need all three of a dtype argument, a
dtype-conditional codes gate, and a mandatory separate output root. This file
needs none of them: it imports the sibling as a MODULE and reuses its taps
(``install_taps``, ``install_rvq_probes``, ``build_reconstruction``,
``write_f32``, ``write_i32``), so both runs are instrumented by the same code,
and builds the float32 model itself.

THE INPUT IS THE ORACLE'S OWN ``waveform.f32``, NOT A WAV, and that is a
correctness requirement. ``waveform.f32`` is the clip *after* upstream's
``.to(model.dtype)`` bf16 cast, widened losslessly back to float32; feeding it
to a float32 model makes that cast a no-op, so the two runs differ only in the
dtype the arithmetic is done in. Feeding the WAV instead would put a
2.6%-of-rms input perturbation through eleven convolutions and confound exactly
the measurement this exists to isolate. Bit-identity of that input is asserted
below rather than assumed, and nothing is written if it fails.

Usage:

    scripts/envs/qwen3-tts/.venv/bin/python \\
        scripts/dump_reference_qwen3_tts_codec_encoder_float32.py \\
        --repo . \\
        --weights-dir models/qwen3-tts-12hz-0-6b-base \\
        --oracle-root build/goldens/qwen3-tts/qwen3-tts-12hz-0-6b-base \\
        --output-root <somewhere outside the oracle tree> \\
        --case base-icl-en --case base-ref-min --case base-text-short
"""

import argparse
import importlib.util
import pathlib
import sys

import numpy as np
import torch

# The `encoder_valid_num_quantizers` slice: 1 semantic + 15 acoustic. Read off
# the imported dumper rather than restated, so the two cannot drift.
SAMPLE_RATE = 24000


def load_dumper(repo: pathlib.Path):
    """Import the committed bf16 dumper as a module, for its taps and writers."""
    path = repo / "scripts" / "dump_reference_qwen3_tts_codec_encoder.py"
    spec = importlib.util.spec_from_file_location("qwen3_tts_codec_encoder_dumper", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: the module defines @dataclasses.dataclass types,
    # and dataclasses resolves each class's own module out of sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path("."),
                        help="Repository root, used to import the sibling dumper.")
    parser.add_argument("--weights-dir", required=True, type=pathlib.Path)
    parser.add_argument("--oracle-root", required=True, type=pathlib.Path,
                        help="Directory holding <case>/codec_encoder/waveform.f32.")
    parser.add_argument("--output-root", required=True, type=pathlib.Path,
                        help="Where this run's artifacts go. MUST NOT be the oracle root: "
                             "these are float32 artifacts and the committed oracle is bf16.")
    parser.add_argument("--case", action="append", required=True,
                        help="Case id under --oracle-root (repeatable).")
    parser.add_argument("--device", default="cpu",
                        help="device_map forwarded to from_pretrained (default: cpu).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_root.resolve() == args.oracle_root.resolve():
        raise SystemExit(
            "--output-root is the oracle root. These are float32 artifacts; writing them "
            "there would overwrite the committed bfloat16 oracle the rest of the plan reads."
        )
    dumper = load_dumper(args.repo)

    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        str(args.weights_dir),
        device_map=args.device,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    speech_tokenizer = model.model.speech_tokenizer
    mimi = speech_tokenizer.model.encoder
    runtime_dtypes = {parameter.dtype for parameter in mimi.parameters()}
    if runtime_dtypes != {torch.float32}:
        raise SystemExit(
            f"the encoder's parameters are {runtime_dtypes}, not float32 -- "
            "from_pretrained did not honour the dtype and this run would measure nothing"
        )
    print(f"upstream reference: {args.device} float32 eager (codec encoder)", flush=True)

    for case in args.case:
        source = args.oracle_root / case / "codec_encoder" / "waveform.f32"
        if not source.exists():
            raise SystemExit(f"{case}: {source} does not exist")
        wav = np.fromfile(source, dtype=np.float32)

        sink: dict = {}
        handles = dumper.install_taps(mimi, sink)
        restores = dumper.install_rvq_probes(mimi, sink)
        try:
            speech_tokenizer.encode([wav], sr=SAMPLE_RATE)
        finally:
            for handle in handles:
                handle.remove()
            for restore in restores:
                restore()

        # The load-bearing precondition. After the feature extractor and the
        # dtype cast, the float32 run must be looking at the EXACT samples the
        # bf16 run and the port both saw. If it is not, every number derived
        # from this dump measures an input difference rather than an arithmetic
        # one -- so nothing is written.
        seen = np.ascontiguousarray(sink["waveform"].reshape(-1), dtype=np.float32)
        if seen.shape != wav.shape or not np.array_equal(seen, wav):
            raise SystemExit(
                f"{case}: the float32 run's own waveform tap is not bit-identical to "
                f"{source} ({seen.shape} vs {wav.shape}). The decomposition would be "
                "invalid; nothing was written."
            )

        semantic_count = int(mimi.quantizer.num_semantic_quantizers)
        reconstruction, _ = dumper.build_reconstruction(sink, semantic_count)

        out = args.output_root / case / "codec_encoder"
        out.mkdir(parents=True, exist_ok=True)
        dumper.write_f32(out / "waveform.f32", sink["waveform"].reshape(-1))
        for index in range(4):
            dumper.write_f32(out / f"seanet_stage{index}.f32", sink[f"seanet_stage{index}"][0])
        dumper.write_f32(out / "seanet_tail.f32", sink["seanet_tail"][0])
        for index in range(len(mimi.encoder_transformer.layers)):
            dumper.write_f32(out / f"transformer_l{index}.f32", sink[f"transformer_l{index}"][0])
        dumper.write_f32(out / "downsample.f32", sink["downsample"][0])
        dumper.write_f32(out / "latents.f32", sink["latents"][0])

        kept = dumper.KEPT_QUANTIZER_COUNT
        frames = int(sink["downsample"][0].shape[-1])
        codes = np.empty((frames, kept), dtype=np.int32)
        margins = np.empty((kept, frames), dtype=np.float32)
        for position in range(kept):
            probe = sink["rvq"][position]
            dumper.write_f32(out / f"rvq_residual_s{position:02d}.f32", probe["residual"])
            codes[:, position] = np.asarray(probe["indices"]).reshape(-1)[:frames]
            d1 = probe["d1"].astype(np.float32).reshape(-1)
            d2 = probe["d2"].astype(np.float32).reshape(-1)
            # (d2 - d1) * (d2 + d1) rather than d2*d2 - d1*d1: algebraically the
            # same, numerically far better conditioned, and the same expression
            # the committed dumper publishes.
            margins[position] = (d2 - d1) * (d2 + d1)
        dumper.write_f32(out / "rvq_distance_margin.f32", margins)
        dumper.write_f32(out / "rvq_reconstruction.f32", reconstruction)
        dumper.write_i32(out / "codes.i32", codes)
        print(f"{case}: {frames} frames -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
