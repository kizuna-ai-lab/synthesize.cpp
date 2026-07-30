#!/usr/bin/env python3
"""Dump Qwen3-TTS reference probes, replay codes, and PCM for every Golden
Manifest case.

The dumper does not reimplement any Qwen3-TTS arithmetic. It drives the pinned
upstream ``Qwen3TTSModel.generate_custom_voice`` entry point and captures
intermediate tensors through forward hooks, so the C++ port can be compared
stage by stage rather than only at the waveform.

Two properties of this family shape the script.

The oracle samples, and seeds the reference so the capture is regeneratable.
Greedy decoding was tried first because it is self-reproducible, and rejected:
it degenerates into a fixed point on some speaker-and-input pairings -- two of
the manifest's cases ran to the token cap emitting one repeated code, with the
shipped repetition penalty active -- and it validates a mode no user runs.
Seeding PyTorch and sampling with the shipped defaults reproduces exactly on
re-run, terminates on every case tried, and is three to seven times faster on
the inputs greedy could not finish.

The replay tensors are the codes, not a generator. ``docs/port-validation.md``
requires captured model inputs "rather than relying on two frameworks to
implement the same pseudorandom generator", so the seed here only makes the
*reference* capture repeatable. ``codes.semantic`` and ``codes.acoustic`` are
what the C++ graph replays, and synthesize.cpp never reproduces MT19937.

Usage:

    uv run --project scripts/envs/qwen3-tts --locked python \
      scripts/dump_reference_qwen3_tts_pytorch.py \
      --manifest tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-customvoice.manifest.json \
      --weights-dir models/qwen3-tts-12hz-0-6b-customvoice
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch

MANIFEST_SCHEMA = "synthesize-golden-manifest-v1"
FAMILY = "qwen3-tts"
# The manifest names these; the hooks below must produce exactly this set.
TALKER_PROBE_LAYERS = (0, 7, 14, 21, 27)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
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
    if manifest.get("family") != FAMILY:
        raise SystemExit(f"{path}: not a {FAMILY} manifest")
    return manifest


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


class TalkerProbes:
    """Captures the first forward's hidden states for the named layers.

    Generation calls each layer once per decoded frame. The probe keeps the
    prefill pass -- the one that consumes the whole prompt -- because that is
    the pass a C++ port reproduces without needing the sampled history, and
    comparing it isolates the encoder side from the autoregressive loop.
    """

    def __init__(self, talker_model) -> None:
        self.captured: dict[str, np.ndarray] = {}
        self.handles = []
        for index in TALKER_PROBE_LAYERS:
            if index >= len(talker_model.layers):
                raise SystemExit(
                    f"talker has {len(talker_model.layers)} layers; probe layer {index} is out of range"
                )
            self.handles.append(
                talker_model.layers[index].register_forward_hook(self._make_hook(f"talker.hidden_l{index}"))
            )
        self.handles.append(talker_model.register_forward_hook(self._make_hook("talker.final")))

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            if name in self.captured:
                return  # prefill only; later calls are single-token decode steps
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            tensor = getattr(tensor, "last_hidden_state", tensor)
            if torch.is_tensor(tensor):
                self.captured[name] = tensor.detach().to(torch.float32).cpu().numpy()
        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def capture_codes(tts, sink: dict):
    """Intercept the codes on their way into the codec.

    `generate_custom_voice` returns only audio, and `speech_tokenizer` is not an
    `nn.Module`, so there is nothing to hook. Wrapping `decode` for the duration
    of the call captures exactly the tensor the codec consumes without altering
    the entry point the manifest pins.
    """
    tokenizer = tts.model.speech_tokenizer
    original = tokenizer.decode

    def wrapper(batch, *args, **kwargs):
        if batch and isinstance(batch[0], dict) and "audio_codes" in batch[0]:
            codes = batch[0]["audio_codes"]
            sink["codes"] = (codes.detach().cpu().numpy() if torch.is_tensor(codes)
                             else np.asarray(codes))
        return original(batch, *args, **kwargs)

    tokenizer.decode = wrapper
    return lambda: setattr(tokenizer, "decode", original)


class ExtraProbes:
    """The text projection's output and the codec head's logits.

    Both fire once per decoded frame; as with the layer probes only the first
    call is kept, so these describe the prefill rather than an arbitrary step of
    the loop.
    """

    def __init__(self, talker) -> None:
        self.captured: dict[str, np.ndarray] = {}
        self.handles = []
        for attribute, name in (("text_projection", "text.embed"), ("codec_head", "talker.logits")):
            module = getattr(talker, attribute, None)
            if module is None:
                raise SystemExit(f"talker has no {attribute}; the probe set is out of date")
            self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            if name in self.captured:
                return
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if torch.is_tensor(tensor):
                self.captured[name] = tensor.detach().to(torch.float32).cpu().numpy()
        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def run_case(tts, case: dict, output_root: pathlib.Path) -> dict:
    parameters = case["oracle"]["parameters"]
    case_dir = output_root / case["id"]
    artifacts: dict[str, dict] = {}

    probes = TalkerProbes(tts.model.talker.model)
    extra = ExtraProbes(tts.model.talker)
    sink: dict = {}
    restore_codes = capture_codes(tts, sink)
    started = time.time()
    try:
        # Seeding immediately before the call is what makes the dump
        # regeneratable; the case's own seed is used so the seeded cases produce
        # genuinely distinct references rather than three copies of one.
        seed = int(case["request"]["seed_u64"])
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        kwargs = dict(
            text=case["input"]["text"],
            language=parameters["language"],
            speaker=parameters["speaker"],
            do_sample=parameters["do_sample"],
            subtalker_dosample=parameters["subtalker_dosample"],
            max_new_tokens=parameters["max_new_tokens"],
        )
        if "instruct" in parameters:
            kwargs["instruct"] = parameters["instruct"]
        wavs, sample_rate = tts.generate_custom_voice(**kwargs)
    finally:
        probes.close()
        extra.close()
        restore_codes()
    wall_seconds = time.time() - started

    for name, array in probes.captured.items():
        artifacts[name] = write_f32(case_dir / f"talker/{name.split('.')[-1]}.f32", array)
    for name, array in extra.captured.items():
        leaf = name.split(".")[-1]
        folder = "text" if name.startswith("text.") else "talker"
        artifacts[name] = write_f32(case_dir / f"{folder}/{leaf}.f32", array)

    # The resolved token ids are the Text Frontend's output, so they are recorded
    # from the same helpers upstream uses rather than re-tokenised independently.
    token_ids = tts._tokenize_texts([tts._build_assistant_text(case["input"]["text"])])[0]
    artifacts["input.token_ids"] = write_i32(
        case_dir / "input/token_ids.i32", token_ids.detach().cpu().numpy().reshape(-1))

    if "codes" not in sink:
        raise SystemExit(f"{case['id']}: codes were never seen entering the codec")
    codes = np.asarray(sink["codes"])
    if codes.ndim > 2:
        codes = codes.reshape(codes.shape[-2], codes.shape[-1])
    # The codec consumes [frames, code_groups] -- time major, 16 groups per frame.
    # Asserting the orientation rather than trusting it: the two axes are easy to
    # transpose silently, and a wrong split still writes plausible-looking files.
    if codes.ndim != 2 or codes.shape[1] != 16:
        raise SystemExit(
            f"{case['id']}: expected [frames, 16] codes, got {codes.shape}"
        )
    # Column 0 is the semantic codebook (rvq_first); columns 1..15 are the
    # acoustic ones (rvq_rest), matching the 1 + 15 quantizer split.
    artifacts["codes.semantic"] = write_i32(case_dir / "codes/semantic.i32", codes[:, 0])
    artifacts["codes.acoustic"] = write_i32(case_dir / "codes/acoustic.i32", codes[:, 1:])

    audio = np.asarray(wavs[0], dtype=np.float32)
    artifacts["audio.pcm"] = write_f32(case_dir / "audio/pcm.f32", audio)

    if not np.isfinite(audio).all():
        raise SystemExit(f"{case['id']}: reference produced non-finite PCM")

    # A run that reaches max_new_tokens never emitted EOS. Sampling has not been
    # observed to degenerate the way greedy did, but the guard stays: a capped
    # dump is not oracle evidence, and accepting one silently would put thousands
    # of frames of a repeated code into the parity baseline.
    cap = int(parameters["max_new_tokens"])
    if codes.shape[0] >= cap - 1:
        tail = codes[-min(400, codes.shape[0]):, 0]
        raise SystemExit(
            f"{case['id']}: generation reached max_new_tokens ({codes.shape[0]} of {cap} frames) "
            f"without emitting EOS; last {tail.size} frames hold {len(set(tail.tolist()))} "
            f"distinct semantic code(s). Choose an input this variant terminates on, "
            f"or record the case as a known non-terminating input."
        )

    # One codec frame is exactly 1920 samples at 12.5 Hz and 24 kHz. Checking it
    # here ties the captured codes to the captured audio: a transposed or
    # mis-sliced code array still produces files, and this is what notices.
    samples_per_frame = 1920
    if audio.shape[0] != codes.shape[0] * samples_per_frame:
        raise SystemExit(
            f"{case['id']}: {codes.shape[0]} code frames imply "
            f"{codes.shape[0] * samples_per_frame} samples, but PCM has {audio.shape[0]}"
        )

    duration = float(audio.shape[0]) / float(sample_rate)
    result = {
        "case": case["id"],
        "status": "ok",
        "sample_rate": int(sample_rate),
        "frames": int(audio.shape[0]),
        "duration_seconds": duration,
        "peak": float(np.abs(audio).max()),
        "wall_seconds": round(wall_seconds, 3),
        "real_time_factor": round(wall_seconds / duration, 3) if duration else None,
    }
    metadata = {
        "seed_u64": case["request"]["seed_u64"],
        "language": parameters["language"],
        "speaker": parameters["speaker"],
        "instruct": parameters.get("instruct"),
        "decoding": {
            "do_sample": parameters["do_sample"],
            "subtalker_dosample": parameters["subtalker_dosample"],
            "max_new_tokens": parameters["max_new_tokens"],
        },
        "probes_captured": sorted(probes.captured),
    }
    write_json(case_dir / "result.json", result)
    write_json(case_dir / "metadata.json", metadata)
    artifacts["result"] = {"path": "result.json"}
    artifacts["metadata"] = {"path": "metadata.json"}
    return {"id": case["id"], "result": result, "artifacts": artifacts}


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    output_root = args.output_root or pathlib.Path(manifest["case_artifact_root"])
    reference = manifest["reference"]
    # Driven by the manifest rather than hardcoded. This family's talker weights
    # are stored bf16, so float32 would upcast them and capture a reference for a
    # model that does not exist.
    dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if reference["dtype"] not in dtypes:
        raise SystemExit(f"unsupported reference dtype {reference['dtype']!r}")
    device = reference["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("manifest pins a cuda reference but no CUDA device is available")

    from qwen_tts import Qwen3TTSModel

    tts = Qwen3TTSModel.from_pretrained(
        str(args.weights_dir),
        device_map="cuda:0" if device == "cuda" else device,
        dtype=dtypes[reference["dtype"]],
        attn_implementation="eager",
    )
    print(f"reference: {device} {reference['dtype']} eager", flush=True)

    wanted = set(args.case) if args.case else None
    cases = [c for c in manifest["cases"] if wanted is None or c["id"] in wanted]
    if wanted and len(cases) != len(wanted):
        missing = wanted - {c["id"] for c in cases}
        raise SystemExit(f"unknown case id(s): {sorted(missing)}")

    records = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case['id']}", flush=True)
        records.append(run_case(tts, case, output_root))
        print(f"    {records[-1]['result']['frames']} frames, "
              f"rtf {records[-1]['result']['real_time_factor']}", flush=True)

    report = {
        "schema": "synthesize-oracle-dump-v1",
        "family": FAMILY,
        "variant": manifest["variant"],
        "suite_version": manifest["suite_version"],
        "manifest": str(args.manifest),
        "reference": reference,
        "case_count": len(records),
        "cases": records,
    }
    if args.report:
        write_json(args.report, report)
    print(f"dumped {len(records)} case(s) to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
