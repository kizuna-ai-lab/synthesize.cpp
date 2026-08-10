#!/usr/bin/env python3
"""Drive the pinned PyTorch oracle over a couple of ad-hoc short-canvas probes.

Same generation parameters as the Golden suite's greedy cases (metadata.json of
build/goldens/omnivoice/omni-rate-fast), CPU F32, deterministic, one thread.
Writes raw f32 PCM into --out.
"""
import argparse, pathlib
import numpy as np
import torch


PARAMS = dict(
    num_step=32,
    guidance_scale=2.0,
    t_shift=0.1,
    layer_penalty_factor=5.0,
    position_temperature=0.0,
    class_temperature=0.0,
    postprocess_output=False,
    pad_duration=0.0,
    fade_duration=0.0,
    audio_chunk_duration=15.0,
    audio_chunk_threshold=30.0,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", default="models/omnivoice-0-6b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe", action="append", required=True, help="text|speed|label")
    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    from omnivoice.models.omnivoice import OmniVoice
    model = OmniVoice.from_pretrained(args.weights_dir, device_map="cpu", dtype=torch.float32)

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    for spec in args.probe:
        text, speed, label = spec.split("|")
        speed = float(speed)
        audios = model.generate(text=text, speed=speed, language="en", instruct=None, **PARAMS)
        audio = np.asarray(audios[0], dtype=np.float32).reshape(-1)
        path = outdir / f"{label}.f32"
        audio.tofile(path)
        print(f"{label}: text={text!r} speed={speed} samples={audio.size} "
              f"frames={audio.size/960:.3f} peak={np.abs(audio).max():.4f} "
              f"mean={audio.mean():.5f} -> {path}", flush=True)


if __name__ == "__main__":
    main()
