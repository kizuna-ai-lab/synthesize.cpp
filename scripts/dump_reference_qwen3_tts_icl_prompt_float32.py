#!/usr/bin/env python3
"""Dump the ICL prompt's two tracks from upstream's OWN modules, in float32.

WHY THIS EXISTS, AND WHY IT IS NOT A FLAG ON ITS SIBLING.

``dump_reference_qwen3_tts_icl_prompt.py`` publishes the committed oracle, and
that oracle is **bfloat16**: ``text_track.f32``, ``codec_track.f32`` and
``icl_embed.f32`` are bf16 tensors widened to float32. A port/oracle difference
therefore mixes two unrelated quantities -- what the port got wrong, and what
bf16 rounding did -- and neither is recoverable from the other. That is the
same problem Task 4 hit on the codec encoder, and the answer is the same one:
run the identical upstream module in float32 on the identical inputs, which
splits the difference in two.

    port          vs  upstream-f32    this port's own arithmetic. A defect here.
    upstream-f32  vs  oracle          the oracle's own bf16 rounding. Nothing to fix.

It is a separate file rather than a ``--dtype`` flag on the sibling because the
sibling's whole job is different: it drives a real ``generate_voice_clone``
call, traces which ``return`` inside ``generate_icl_prompt`` executed, and
writes the alignment record. None of that is needed here and all of it costs a
full voice-clone run. This script imports the sibling as a MODULE and reuses its
two rebuild helpers (``rebuild_text_track``, ``rebuild_codec_track``), so both
runs construct the tracks through the same code, and takes its inputs from the
oracle's own ``ref_text_ids.i32`` / ``target_text_ids.i32`` /
``codes/reference.i32`` -- the ids and codes upstream itself passed.

THE ALIGNMENT IS RECOMPUTED HERE, and that is safe in a way it is not in the
tests: this script is not checking which arm upstream took (``alignment.json``
already recorded that from the executed ``return``), it is reproducing a
float32 twin of an artifact whose arm is already known. The two are cross-
checked anyway -- the arm this script computes is compared against
``alignment.json``'s and the run aborts if they disagree.

Writes ``text_track.f32``, ``codec_track.f32`` and ``icl_embed.f32`` per case
under ``--output-root``, which MUST NOT overlap the oracle root, and prints the
upstream-f32-vs-oracle deviation for each. Pass ``--port-root`` to also report
port-vs-upstream-f32, against tracks the integration test wrote there (see
``SYNTH_QWEN3_TTS_ICL_DUMP_DIR`` in tests/qwen3_tts_icl_prompt_real.cpp).

Usage:

    uv run --project scripts/envs/qwen3-tts --locked python \\
        scripts/dump_reference_qwen3_tts_icl_prompt_float32.py \\
        --repo . \\
        --weights-dir models/qwen3-tts-12hz-0-6b-base \\
        --oracle-root build/goldens/qwen3-tts/qwen3-tts-12hz-0-6b-base \\
        --output-root build/icl-prompt-f32 \\
        --case base-icl-en --case base-ref-min --case base-text-short
"""

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np
import torch


def load_dumper(repo: pathlib.Path):
    """Import the committed bf16 dumper as a module, for its rebuild helpers."""
    path = repo / "scripts" / "dump_reference_qwen3_tts_icl_prompt.py"
    spec = importlib.util.spec_from_file_location("qwen3_tts_icl_prompt_dumper", path)
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
    parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--weights-dir", required=True, type=pathlib.Path)
    parser.add_argument("--oracle-root", required=True, type=pathlib.Path)
    parser.add_argument("--output-root", required=True, type=pathlib.Path,
                        help="Where this run's float32 tracks go. MUST NOT overlap the oracle "
                             "root in either direction.")
    parser.add_argument("--port-root", type=pathlib.Path, default=None,
                        help="Directory holding <case>-text.f32 / <case>-codec.f32 written by "
                             "tests/qwen3_tts_icl_prompt_real.cpp under "
                             "SYNTH_QWEN3_TTS_ICL_DUMP_DIR. Adds the port-vs-upstream-f32 half.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--case", action="append", required=True)
    return parser.parse_args()


def relative(got: np.ndarray, reference: np.ndarray) -> float:
    """max|delta| over the reference's own absmax -- Tasks 4 and 5's normalizer."""
    got = np.asarray(got, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    scale = float(np.max(np.abs(reference)))
    worst = float(np.max(np.abs(got - reference)))
    return worst / scale if scale > 0.0 else worst


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    # Containment in EITHER direction, not just equality: `--output-root
    # <oracle-root>/f32` writes one directory deeper, so it collides with nothing
    # and an equality test waves it through. Equality is the degenerate case both
    # `is_relative_to` calls already cover.
    output_root = args.output_root.resolve()
    oracle_root = args.oracle_root.resolve()
    if output_root.is_relative_to(oracle_root) or oracle_root.is_relative_to(output_root):
        raise SystemExit(
            f"--output-root {output_root} overlaps --oracle-root {oracle_root}. These are "
            "float32 artifacts and the oracle is bfloat16; one tree holding both, with "
            "nothing in the layout saying which files are which, is a confounded "
            "measurement -- the decomposition this script exists to produce would read "
            "whichever dtype it happened to find. Telling them apart afterwards costs a "
            "full regeneration. Point --output-root somewhere disjoint."
        )
    dumper = load_dumper(args.repo)

    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        str(args.weights_dir), device_map=args.device, dtype=torch.float32,
        attn_implementation="eager",
    )
    talker = model.model.talker
    dtypes = {parameter.dtype for parameter in talker.text_projection.parameters()}
    if dtypes != {torch.float32}:
        raise SystemExit(
            f"text_projection is {dtypes}, not float32 -- from_pretrained did not honour "
            "the dtype and this run would measure nothing"
        )
    config = model.model.config
    print(f"upstream reference: {args.device} float32 eager (icl prompt)", flush=True)

    specials = torch.tensor(
        [[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]],
        device=talker.device, dtype=torch.long,
    )
    _bos, tts_eos, tts_pad = talker.text_projection(
        talker.get_text_embeddings()(specials)
    ).chunk(3, dim=1)

    for case in args.case:
        prompt_dir = args.oracle_root / case / "prompt"
        alignment = json.loads((prompt_dir / "alignment.json").read_text())

        # Onto the talker's device, not left on the host. `--device cuda:0` puts the
        # embedding tables there, and an id tensor that stayed behind makes the lookup
        # inside the rebuild helpers a device mismatch rather than a slow path.
        ref_id = torch.from_numpy(
            np.fromfile(prompt_dir / "ref_text_ids.i32", dtype=np.int32).astype(np.int64)
        ).unsqueeze(0).to(talker.device)
        text_id = torch.from_numpy(
            np.fromfile(prompt_dir / "target_text_ids.i32", dtype=np.int32).astype(np.int64)
        ).unsqueeze(0).to(talker.device)
        grid = np.fromfile(args.oracle_root / case / "codes" / "reference.i32", dtype=np.int32)
        group_count = int(talker.config.num_code_groups)
        frames = grid.size // group_count
        if frames != int(alignment["ref_frames"]):
            raise SystemExit(
                f"{case}: codes/reference.i32 holds {frames} frames, alignment.json says "
                f"{alignment['ref_frames']}"
            )
        ref_code = torch.from_numpy(
            grid.reshape(frames, group_count).astype(np.int64)
        ).to(talker.device)

        text_track = dumper.rebuild_text_track(talker, ref_id, text_id, tts_eos)
        codec_track = dumper.rebuild_codec_track(talker, config, ref_code, torch.long)
        text_lens = int(text_track.shape[1])
        codec_lens = int(codec_track.shape[1])
        if text_lens != int(alignment["T1"]) or codec_lens != int(alignment["T2"]):
            raise SystemExit(
                f"{case}: rebuilt T1/T2 are {text_lens}/{codec_lens}, alignment.json says "
                f"{alignment['T1']}/{alignment['T2']}"
            )
        if text_lens > codec_lens:
            arm = "truncate"
            aligned_text = text_track[:, :codec_lens]
        else:
            arm = "pad"
            aligned_text = torch.cat([text_track] + [tts_pad] * (codec_lens - text_lens), dim=1)
        if arm != alignment["branch"]:
            raise SystemExit(
                f"{case}: this run's arm is '{arm}', but alignment.json read '{alignment['branch']}' "
                "out of the return upstream actually executed. The oracle is the authority; "
                "something about these inputs is not what produced it."
            )
        block = aligned_text + codec_track

        case_dir = args.output_root / case / "prompt"
        case_dir.mkdir(parents=True, exist_ok=True)
        # `.cpu()` before `.numpy()`: numpy cannot see a CUDA tensor, and under
        # `--device cuda:0` every one of these is one.
        got_text = np.ascontiguousarray(aligned_text[0].to(torch.float32).cpu().numpy(), dtype=np.float32)
        got_codec = np.ascontiguousarray(codec_track[0].to(torch.float32).cpu().numpy(), dtype=np.float32)
        got_block = np.ascontiguousarray(block[0].to(torch.float32).cpu().numpy(), dtype=np.float32)
        got_text.tofile(case_dir / "text_track.f32")
        got_codec.tofile(case_dir / "codec_track.f32")
        got_block.tofile(case_dir / "icl_embed.f32")

        shape = (codec_lens, -1)
        oracle_text = np.fromfile(prompt_dir / "text_track.f32", dtype=np.float32).reshape(shape)
        oracle_codec = np.fromfile(prompt_dir / "codec_track.f32", dtype=np.float32).reshape(shape)
        oracle_block = np.fromfile(prompt_dir / "icl_embed.f32", dtype=np.float32).reshape(shape)
        print(
            f"{case:16s} T1={text_lens:3d} T2={codec_lens:3d} {arm:8s}"
            f"  upstream-f32 vs bf16 oracle:"
            f"  text {relative(got_text, oracle_text):.3e}"
            f"  codec {relative(got_codec, oracle_codec):.3e}"
            f"  block {relative(got_block, oracle_block):.3e}",
            flush=True,
        )

        if args.port_root is not None:
            port_text = np.fromfile(args.port_root / f"{case}-text.f32", dtype=np.float32).reshape(shape)
            port_codec = np.fromfile(args.port_root / f"{case}-codec.f32", dtype=np.float32).reshape(shape)
            print(
                f"{case:16s} {'':22s}  port vs upstream-f32:"
                f"  text {relative(port_text, got_text):.3e}"
                f"  codec {relative(port_codec, got_codec):.3e}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
