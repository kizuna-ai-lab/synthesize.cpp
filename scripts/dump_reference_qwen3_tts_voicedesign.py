#!/usr/bin/env python3
"""Dump a Qwen3-TTS VoiceDesign reference prefill, replay codes, and PCM.

The dumper does not reimplement any Qwen3-TTS arithmetic. It drives the pinned
upstream ``Qwen3TTSModel.generate_voice_design`` entry point and captures the
tensors a C++ port must reproduce by wrapping two plain (non-``nn.Module``)
methods for the duration of one call, the same technique
``scripts/dump_reference_qwen3_tts_base.py`` uses for
``generate_icl_prompt``/``generate`` and ``scripts/dump_reference_qwen3_tts_pytorch.py``
uses for ``speech_tokenizer.decode``.

THE PREFILL PROBE, AND WHY IT IS CAPTURED WHERE IT IS. Design section 6.1
(docs/superpowers/specs/2026-08-18-qwen3-tts-stage-3-design.md) calls the
assembled talker prefill "the load-bearing" comparison: comparing only the
projected instruct/text embeddings on their own would largely re-test
``text_projection``, which the assistant text already exercises everywhere
else in this family. What VoiceDesign's own path contributes independently is
the WHOLE prefill's shape and content once the speaker slot is dropped
(``generate_voice_design`` -> ``modeling_qwen3_tts.py``'s ``generate``, lines
~2088-2089 leaving ``speaker_embed`` as ``None``) and, once Plan 2 lands, the
instruct block's placement ahead of it. There is exactly one place in
upstream's own code where the whole assembled prefill exists as a value before
being consumed: the ``inputs_embeds`` keyword argument of
``self.talker.generate(...)`` (``modeling_qwen3_tts.py:2271``), so that call is
what this file wraps -- there is no smaller boundary to intercept the same
tensor at.

DETERMINISM SETTINGS ARE TAKEN FROM scripts/dump_reference_qwen3_tts_base.py,
NOT CHOSEN HERE, and NOT the literal "greedy decoding" an earlier draft of the
task brief that requested this file described. Reading that file's own module
docstring: greedy decoding was tried first, because it is self-reproducible
without a seed, and abandoned within an hour of GPU time -- "it degenerates
into a fixed point on some speaker-and-input pairings", running one case to
its 8191-frame cap without ever sampling the codec end token. That failure
mode is exactly as available to a brand-new VoiceDesign input as it was to a
Base one; nothing about this variant makes greedy safer. So this file matches
the base dumper's ACTUAL protocol instead: ``torch.manual_seed(seed)``
immediately before the call, ``do_sample=True``, ``subtalker_dosample=True``.
This also costs nothing for what this dumper's completion gate actually
compares: the assembled prefill is a pure function of the tokenized text, the
instruct block and the model's weights -- no sampling happens before it is
captured, only in the autoregressive codes that follow it -- so the seeded
sampling protocol governs reproducibility of ``codes.i32``/``waveform.f32``
alone, and is adopted here for exactly the reason it was adopted there:
avoiding the degenerate run, not anything the prefill comparison needs.

Usage (the only form this file has -- Plan 1's Golden Manifest carries no
cases yet, so unlike the Base dumper there is no ``--manifest`` form to add):

    uv run --project scripts/envs/qwen3-tts --locked python \\
      scripts/dump_reference_qwen3_tts_voicedesign.py \\
      --checkpoint models/qwen3-tts-12hz-1-7b-voicedesign-src \\
      --text "Qwen3-TTS is awesome!" --instruct "" --language English \\
      --out reports/porting/qwen3-tts/qwen3-tts-12hz-1-7b-voicedesign/oracle/
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any, Optional

import numpy as np
import torch

FAMILY = "qwen3-tts"
VARIANT = "qwen3-tts-12hz-1-7b-voicedesign"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path,
                        help="Local VoiceDesign checkpoint directory (config.json, "
                             "model.safetensors, speech_tokenizer/).")
    parser.add_argument(
        "--device", default="cuda:0",
        help="device_map forwarded to Qwen3TTSModel.from_pretrained (default: cuda:0). "
             "The talker checkpoint is stored bfloat16; see docs/port-validation.md, "
             "'Choosing the Oracle's dtype and Device'.",
    )
    parser.add_argument("--text", required=True, help="Text to synthesize.")
    parser.add_argument(
        "--instruct", required=True,
        help="Instruction describing the desired voice/style. An empty string is a "
             "legal input (design decision D3): upstream treats it as no instruction "
             "at all -- instruct_ids.append(None), no instruct block prepended to the "
             "prefill -- which is the path Plan 1's completion gate exercises. "
             "Required rather than defaulted, so a caller states the choice rather "
             "than getting it by omission.",
    )
    parser.add_argument("--language", required=True, help="Target language, e.g. English.")
    parser.add_argument("--out", required=True, type=pathlib.Path,
                        help="Output directory for prefill.f32, codes.i32, waveform.f32, "
                             "result.json and metadata.json.")
    parser.add_argument("--seed", type=int, default=0,
                        help="torch.manual_seed applied immediately before "
                             "generate_voice_design. Decoding samples (do_sample=True, "
                             "subtalker_dosample=True) -- see the module docstring for why "
                             "this is not greedy -- so this seed is what makes the "
                             "codes.i32/waveform.f32 half of the dump regeneratable. It has "
                             "no bearing on prefill.f32, which is assembled before any "
                             "sampling happens.")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Codec-frame cap forwarded to generate_voice_design. Defaults "
                             "to 2048, matching every Stage 1/2 manifest case in this family.")
    parser.add_argument("--report", type=pathlib.Path, default=None)
    return parser.parse_args()


def write_f32(path: pathlib.Path, array: np.ndarray) -> dict:
    data = np.ascontiguousarray(array, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.tobytes())
    return {"path": path.name, "shape": list(data.shape), "elements": int(data.size)}


def write_i32(path: pathlib.Path, array: np.ndarray) -> dict:
    data = np.ascontiguousarray(array, dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.tobytes())
    return {"path": path.name, "shape": list(data.shape), "elements": int(data.size)}


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def to_numpy(value: Any, dtype: Optional[torch.dtype] = None) -> np.ndarray:
    """Detach and move a possibly-GPU, possibly-bf16 tensor to a plain host array.

    ``write_f32``/``write_i32`` call ``np.ascontiguousarray`` directly, which
    cannot see a CUDA tensor or a dtype numpy has no analogue for (bf16); every
    captured value passes through here first. Copied from
    scripts/dump_reference_qwen3_tts_base.py rather than imported: each dumper
    in this family is an independent CLI entry point and already carries its
    own copy of this helper.
    """
    if torch.is_tensor(value):
        tensor = value.detach()
        if dtype is not None:
            tensor = tensor.to(dtype)
        return tensor.cpu().numpy()
    return np.asarray(value)


def capture_talker_prefill(model, sink: dict):
    """Wraps the talker's own ``generate`` to capture the assembled prefill.

    ``Qwen3TTSForConditionalGeneration.generate`` (modeling_qwen3_tts.py:2022)
    builds ``talker_input_embeds`` -- the (absent, at empty instruct) instruct
    block, the role prefix, the codec-tag block with the speaker slot dropped
    for this variant, and the tts-text block -- entirely as a local variable,
    then hands it to ``self.talker.generate(inputs_embeds=talker_input_embeds,
    ...)`` (:2271). There is no smaller call boundary to intercept it at: this
    is the module docstring's "one place... where the whole assembled prefill
    exists as a value".
    """
    target = model.model.talker
    original = target.generate

    def wrapper(*args, **kwargs):
        if "prefill" not in sink:
            inputs_embeds = kwargs.get("inputs_embeds")
            if inputs_embeds is None:
                raise SystemExit("talker.generate was called without an inputs_embeds kwarg")
            # Batch size is always 1 here (one case, one text), so the batch
            # dimension carries no left-padding to strip -- unlike the
            # multi-sample path inside the outer generate() (:2233-2249),
            # which pads across samples of DIFFERENT lengths before this call.
            if inputs_embeds.shape[0] != 1:
                raise SystemExit(f"expected batch size 1, got {inputs_embeds.shape[0]}")
            sink["prefill"] = to_numpy(inputs_embeds[0], torch.float32)
        return original(*args, **kwargs)

    target.generate = wrapper
    return lambda: setattr(target, "generate", original)


def capture_generated_codes(model, sink: dict):
    """Wraps the outer ``generate`` to capture the newly generated codes.

    Same call site scripts/dump_reference_qwen3_tts_base.py's own
    ``capture_generated_codes`` wraps (that file's ``model.model.generate`` is
    this file's ``target.generate``); duplicated rather than imported for the
    same reason ``to_numpy`` is.
    """
    target = model.model
    original = target.generate

    def wrapper(*args, **kwargs):
        talker_codes_list, talker_hidden_states_list = original(*args, **kwargs)
        if "codes" not in sink:
            sink["codes"] = to_numpy(talker_codes_list[0])
        return talker_codes_list, talker_hidden_states_list

    target.generate = wrapper
    return lambda: setattr(target, "generate", original)


def run(args: argparse.Namespace) -> dict:
    from qwen_tts import Qwen3TTSModel

    # Driven by the checkpoint's own stored dtype rather than hardcoded as a
    # blanket default: the talker weights are bf16, so float32 would upcast
    # every talker weight and capture a reference for a model that does not
    # exist. See docs/port-validation.md, "Choosing the Oracle's dtype and
    # Device" -- the same reasoning scripts/dump_reference_qwen3_tts_base.py
    # applies.
    model = Qwen3TTSModel.from_pretrained(
        str(args.checkpoint),
        device_map=args.device,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    print(f"reference: {args.device} bfloat16 eager", flush=True)

    sink: dict = {}
    restore_prefill = capture_talker_prefill(model, sink)
    restore_codes = capture_generated_codes(model, sink)
    started = time.time()
    try:
        # Sampled, not greedy -- seeding immediately before the call is what
        # makes codes.i32/waveform.f32 regeneratable. See the module
        # docstring for why this matches scripts/dump_reference_qwen3_tts_base.py
        # rather than the literal greedy decoding an earlier draft of this
        # task asked for.
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        wavs, sample_rate = model.generate_voice_design(
            text=args.text,
            instruct=args.instruct,
            language=args.language,
            do_sample=True,
            subtalker_dosample=True,
            max_new_tokens=args.max_new_tokens,
        )
    finally:
        restore_prefill()
        restore_codes()
    wall_seconds = time.time() - started

    if "prefill" not in sink:
        raise SystemExit(
            "the talker's assembled prefill was never seen leaving "
            "Qwen3TTSForConditionalGeneration.generate"
        )
    if "codes" not in sink:
        raise SystemExit("generated talker codes were never seen leaving model.generate")

    prefill = sink["prefill"]
    if prefill.ndim != 2:
        raise SystemExit(f"expected a [T, hidden] prefill, got shape {prefill.shape}")

    codes = sink["codes"]
    if codes.ndim != 2 or codes.shape[1] != 16:
        raise SystemExit(f"expected [frames, 16] generated codes, got {codes.shape}")

    # A run that reaches the cap never emitted EOS, so it is not oracle
    # evidence regardless of how plausible the tail looks -- the same guard
    # scripts/dump_reference_qwen3_tts_base.py applies, for the exact failure
    # mode its module docstring documents.
    effective_cap = args.max_new_tokens
    if codes.shape[0] >= effective_cap - 1:
        tail = codes[-min(400, codes.shape[0]):, 0]
        raise SystemExit(
            f"generation reached max_new_tokens ({codes.shape[0]} of {effective_cap} "
            f"frames) without emitting EOS; last {tail.size} frames hold "
            f"{len(set(tail.tolist()))} distinct semantic code(s). This dump is not oracle "
            "evidence. Choose a different --seed or an input this variant terminates on."
        )

    wav = np.asarray(wavs[0], dtype=np.float32)
    if not np.isfinite(wav).all():
        raise SystemExit("reference produced non-finite PCM")

    artifacts: dict[str, dict] = {}
    artifacts["prefill"] = write_f32(args.out / "prefill.f32", prefill)
    artifacts["codes"] = write_i32(args.out / "codes.i32", codes)
    artifacts["waveform"] = write_f32(args.out / "waveform.f32", wav)

    duration = float(wav.shape[0]) / float(sample_rate) if wav.size else 0.0
    result: dict[str, Any] = {
        "schema": "synthesize-oracle-dump-v1",
        "family": FAMILY,
        "variant": VARIANT,
        "reference": {"device": args.device, "dtype": "bfloat16", "attn_implementation": "eager"},
        "status": "ok",
        "text": args.text,
        "instruct": args.instruct,
        "language": args.language,
        "sample_rate": int(sample_rate),
        "prefill_positions": int(prefill.shape[0]),
        "hidden_size": int(prefill.shape[1]),
        "generated_code_frames": int(codes.shape[0]),
        "frames": int(wav.shape[0]),
        "duration_seconds": duration,
        "peak_abs": float(np.abs(wav).max()) if wav.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(wav, dtype=np.float64)))) if wav.size else 0.0,
        "wall_seconds": round(wall_seconds, 3),
        "real_time_factor": round(wall_seconds / duration, 3) if duration else None,
        "decoding": {
            "do_sample": True,
            "subtalker_dosample": True,
            "max_new_tokens": effective_cap,
            "seed": args.seed,
        },
        "artifacts": artifacts,
    }
    write_json(args.out / "result.json", result)
    return result


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"--device {args.device!r} requests CUDA but no CUDA device is available")

    result = run(args)
    print(json.dumps(result, indent=2))
    if args.report:
        write_json(args.report, result)
    print(
        f"dumped 1 case: {result['prefill_positions']} prefill positions, "
        f"{result['generated_code_frames']} code frames, rtf {result['real_time_factor']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
