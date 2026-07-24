#!/usr/bin/env python3
"""Dump Kokoro reference probes, stochastic replay inputs, and PCM for every
Golden Manifest case.

The dumper does not reimplement any Kokoro arithmetic. It drives the pinned
upstream modules stage by stage so intermediate tensors can be captured, and it
injects the harmonic-source random draws so the same values can later be
replayed through the C++ graph. Each case is verified against an unmodified
``KModel.forward_with_tokens`` call under the same injected randomness; a
mismatch aborts the dump.

Usage:

    uv run --project scripts/envs/kokoro --locked python \
      scripts/dump_reference_kokoro_pytorch.py \
      --manifest tests/golden/kokoro/kokoro-v1-0.manifest.json \
      --source-dir models/upstream/kokoro-source \
      --weights-dir models/upstream/kokoro-v1_0
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import pathlib
import sys

import numpy as np
import torch

MANIFEST_SCHEMA = "synthesize-golden-manifest-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--source-dir", required=True, type=pathlib.Path)
    parser.add_argument("--weights-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output-root", type=pathlib.Path, default=None)
    parser.add_argument("--case", action="append", default=None,
                        help="Restrict the dump to the given case id (repeatable).")
    parser.add_argument("--report", type=pathlib.Path, default=None)
    return parser.parse_args()


def load_manifest(path: pathlib.Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise SystemExit(f"{path}: unexpected manifest schema {manifest.get('schema')!r}")
    if manifest.get("family") != "kokoro":
        raise SystemExit(f"{path}: not a kokoro manifest")
    return manifest


def write_f32(path: pathlib.Path, tensor: torch.Tensor) -> dict:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(array.tobytes())
    return {
        "shape": list(array.shape),
        "bytes": array.nbytes,
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def write_i64(path: pathlib.Path, tensor: torch.Tensor) -> dict:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(array.tobytes())
    return {
        "shape": list(array.shape),
        "bytes": array.nbytes,
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def write_json(path: pathlib.Path, payload: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return {
        "bytes": len(text.encode("utf-8")),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def draw_source_randomness(seed: int, upsampled_frames: int, harmonics: int) -> dict:
    """Draw the exact tensors the pinned harmonic source module consumes.

    ``rand_ini`` index zero is forced to zero the way ``SineGen._f02sine`` does,
    so the stored artifact is the value actually used rather than a raw draw.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rand_ini = torch.rand(1, harmonics, generator=generator)
    rand_ini[:, 0] = 0
    sine_noise = torch.randn(1, upsampled_frames, harmonics, generator=generator)
    unused_noise = torch.randn(1, upsampled_frames, 1, generator=generator)
    return {"rand_ini": rand_ini, "sine_noise": sine_noise, "unused_noise": unused_noise}


@contextlib.contextmanager
def injected_source_randomness(draws: dict):
    """Replace the three source-module draws with pre-generated tensors.

    ``SineGen`` calls ``torch.rand`` once for the initial phase, then
    ``torch.randn_like`` for the additive sine noise, and ``SourceModuleHnNSF``
    calls ``torch.randn_like`` once more for a tensor the Generator discards.
    """
    real_rand = torch.rand
    real_randn_like = torch.randn_like
    state = {"rand": 0, "randn_like": 0}

    def fake_rand(*args, **kwargs):
        if kwargs.get("generator") is not None:
            return real_rand(*args, **kwargs)
        state["rand"] += 1
        if state["rand"] != 1:
            raise AssertionError(f"unexpected torch.rand call #{state['rand']}")
        return draws["rand_ini"].clone()

    def fake_randn_like(tensor, *args, **kwargs):
        state["randn_like"] += 1
        if state["randn_like"] == 1:
            value = draws["sine_noise"]
        elif state["randn_like"] == 2:
            value = draws["unused_noise"]
        else:
            raise AssertionError(f"unexpected torch.randn_like call #{state['randn_like']}")
        if tuple(value.shape) != tuple(tensor.shape):
            raise AssertionError(
                f"injected tensor {tuple(value.shape)} does not match {tuple(tensor.shape)}"
            )
        return value.clone()

    torch.rand = fake_rand
    torch.randn_like = fake_randn_like
    try:
        yield state
    finally:
        torch.rand = real_rand
        torch.randn_like = real_randn_like


@torch.no_grad()
def run_case(model, case: dict, voices_dir: pathlib.Path, out_dir: pathlib.Path) -> dict:
    token_ids = case["input"]["token_ids"]
    speed = float(case["request"]["speaking_rate"])
    seed = int(case["request"]["seed_u64"])
    voice_id = case["voice"]["id"]

    input_ids = torch.LongTensor([token_ids])
    style_row = len(token_ids) - 3
    pack = torch.load(voices_dir / f"{voice_id}.pt", weights_only=True)
    if not 0 <= style_row < pack.shape[0]:
        raise SystemExit(f"{case['id']}: style row {style_row} outside voicepack")
    ref_s = pack[style_row]

    # ---- stage 1: PL-BERT and the prosody text encoder -------------------
    input_lengths = torch.full((1,), input_ids.shape[-1], dtype=torch.long)
    text_mask = torch.arange(int(input_lengths.max())).unsqueeze(0)
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))

    bert_hidden = model.bert(input_ids, attention_mask=(~text_mask).int())
    d_en = model.bert_encoder(bert_hidden).transpose(-1, -2)

    style_prosody = ref_s[:, 128:]
    style_decoder = ref_s[:, :128]

    d = model.predictor.text_encoder(d_en, style_prosody, input_lengths, text_mask)

    # ---- stage 2: duration ------------------------------------------------
    lstm_out, _ = model.predictor.lstm(d)
    duration_logits = model.predictor.duration_proj(lstm_out)
    summed = torch.sigmoid(duration_logits).sum(axis=-1) / speed
    pred_dur = torch.round(summed).clamp(min=1).long().squeeze()
    y_length = int(pred_dur.sum())

    indices = torch.repeat_interleave(torch.arange(input_ids.shape[1]), pred_dur)
    alignment = torch.zeros((input_ids.shape[1], indices.shape[0]))
    alignment[indices, torch.arange(indices.shape[0])] = 1
    alignment = alignment.unsqueeze(0)

    # ---- stage 3: prosody expansion and F0/energy -------------------------
    en = d.transpose(-1, -2) @ alignment
    f0_pred, n_pred = model.predictor.F0Ntrain(en, style_prosody)

    # ---- stage 4: text encoder and alignment expansion --------------------
    t_en = model.text_encoder(input_ids, input_lengths, text_mask)
    asr = t_en @ alignment

    # ---- stage 5: harmonic source with injected randomness ----------------
    generator_module = model.decoder.generator
    upsampled = f0_pred.shape[-1] * 300
    draws = draw_source_randomness(seed, upsampled, generator_module.m_source.l_sin_gen.dim)

    with injected_source_randomness(draws) as calls:
        f0_up = generator_module.f0_upsamp(f0_pred[:, None]).transpose(1, 2)
        har_source, _, _ = generator_module.m_source(f0_up)
        har_source_flat = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = generator_module.stft.transform(har_source_flat)
        har = torch.cat([har_spec, har_phase], dim=1)
    if calls["rand"] != 1 or calls["randn_like"] != 2:
        raise SystemExit(f"{case['id']}: unexpected source draw counts {calls}")

    # ---- stage 6: full decoder through the same injected randomness -------
    with injected_source_randomness(draws):
        audio = model.decoder(asr, f0_pred, n_pred, style_decoder).squeeze()

    # ---- verification against the unmodified upstream entry point ---------
    with injected_source_randomness(draws):
        upstream_audio, upstream_dur = model.forward_with_tokens(input_ids, ref_s, speed)
    upstream_audio = upstream_audio.squeeze()
    if not torch.equal(pred_dur, upstream_dur.squeeze()):
        raise SystemExit(f"{case['id']}: staged durations differ from forward_with_tokens")
    if not torch.equal(audio, upstream_audio):
        delta = (audio - upstream_audio).abs().max().item()
        raise SystemExit(f"{case['id']}: staged audio differs from upstream by {delta:.6e}")

    expected_samples = y_length * 600
    if audio.shape[-1] != expected_samples:
        raise SystemExit(
            f"{case['id']}: produced {audio.shape[-1]} samples, expected {expected_samples}"
        )
    if not torch.isfinite(audio).all():
        raise SystemExit(f"{case['id']}: produced non-finite PCM")

    # ---- write artifacts ---------------------------------------------------
    written = {
        "bert/hidden.f32": write_f32(out_dir / "bert/hidden.f32", bert_hidden),
        "text/d_en.f32": write_f32(out_dir / "text/d_en.f32", d_en),
        "duration/d.f32": write_f32(out_dir / "duration/d.f32", d),
        "duration/logits.f32": write_f32(out_dir / "duration/logits.f32", duration_logits),
        "duration/pred_dur.i64": write_i64(out_dir / "duration/pred_dur.i64", pred_dur),
        "duration/y_length.i64": write_i64(
            out_dir / "duration/y_length.i64", torch.tensor([y_length], dtype=torch.long)
        ),
        "duration/alignment.f32": write_f32(out_dir / "duration/alignment.f32", alignment),
        "prosody/en.f32": write_f32(out_dir / "prosody/en.f32", en),
        "prosody/f0.f32": write_f32(out_dir / "prosody/f0.f32", f0_pred),
        "prosody/n.f32": write_f32(out_dir / "prosody/n.f32", n_pred),
        "text/t_en.f32": write_f32(out_dir / "text/t_en.f32", t_en),
        "text/asr.f32": write_f32(out_dir / "text/asr.f32", asr),
        "source/har.f32": write_f32(out_dir / "source/har.f32", har),
        "audio/pcm.f32": write_f32(out_dir / "audio/pcm.f32", audio),
        "random/rand_ini.f32": write_f32(out_dir / "random/rand_ini.f32", draws["rand_ini"]),
        "random/source_noise.f32": write_f32(
            out_dir / "random/source_noise.f32", draws["sine_noise"]
        ),
    }

    result = {
        "frames_emitted": int(audio.shape[-1]),
        "actual_seed": str(seed),
        "sample_rate": 24000,
        "channel_count": 1,
        "resolved_language_tag": case["input"]["language_tag"],
        "resolved_voice_id": voice_id,
        "seed_used": True,
        "native_streaming_used": False,
    }
    written["result.json"] = write_json(out_dir / "result.json", result)

    metadata = {
        "case_id": case["id"],
        "token_count": len(token_ids),
        "style_row": style_row,
        "speaking_rate": speed,
        "y_length": y_length,
        "pred_dur": pred_dur.tolist(),
        "upsampled_source_frames": upsampled,
        "shapes": {
            "bert.hidden": list(bert_hidden.shape),
            "text.d_en": list(d_en.shape),
            "duration.d": list(d.shape),
            "duration.logits": list(duration_logits.shape),
            "duration.alignment": list(alignment.shape),
            "prosody.en": list(en.shape),
            "prosody.f0": list(f0_pred.shape),
            "prosody.n": list(n_pred.shape),
            "text.t_en": list(t_en.shape),
            "text.asr": list(asr.shape),
            "source.har": list(har.shape),
            "audio.pcm": list(audio.shape),
        },
    }
    written["metadata.json"] = write_json(out_dir / "metadata.json", metadata)

    return {
        "case": case["id"],
        "voice": voice_id,
        "token_count": len(token_ids),
        "style_row": style_row,
        "y_length": y_length,
        "frames": int(audio.shape[-1]),
        "pcm_sha256": written["audio/pcm.f32"]["sha256"],
        "artifacts": written,
    }


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.source_dir.resolve()))
    from kokoro import KModel  # noqa: E402  (import needs the pinned source on the path)

    manifest = load_manifest(args.manifest)
    output_root = args.output_root or pathlib.Path(manifest["case_artifact_root"])
    voices_dir = args.weights_dir / "voices"

    model = KModel(
        repo_id="hexgrad/Kokoro-82M",
        config=str(args.weights_dir / "config.json"),
        model=str(args.weights_dir / "kokoro-v1_0.pth"),
    ).eval()

    wanted = set(args.case) if args.case else None
    reports = []
    for case in manifest["cases"]:
        if wanted is not None and case["id"] not in wanted:
            continue
        out_dir = output_root / case["id"]
        report = run_case(model, case, voices_dir, out_dir)
        reports.append(report)
        print(
            f"{report['case']:26s} voice={report['voice']:12s} "
            f"tok={report['token_count']:4d} row={report['style_row']:4d} "
            f"Y={report['y_length']:5d} frames={report['frames']:7d} "
            f"pcm={report['pcm_sha256'][:16]}"
        )

    if wanted is not None:
        missing = wanted - {r["case"] for r in reports}
        if missing:
            raise SystemExit(f"unknown case ids: {sorted(missing)}")

    summary = {
        "schema": "synthesize-reference-dump-v1",
        "family": manifest["family"],
        "variant": manifest["variant"],
        "suite_version": manifest["suite_version"],
        "reference": manifest["reference"],
        "output_root": str(output_root),
        "cases": reports,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.report}")
    print(f"dumped {len(reports)} case(s) under {output_root}")


if __name__ == "__main__":
    main()
