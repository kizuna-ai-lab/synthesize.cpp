#!/usr/bin/env python3
"""Dump Qwen3-TTS-Base reference voice-clone probes, replay codes, and PCM.

The dumper does not reimplement any Qwen3-TTS arithmetic. It drives the pinned
upstream ``Qwen3TTSModel.create_voice_clone_prompt`` / ``generate_voice_clone``
entry points and captures the tensors a C++ port must reproduce by wrapping
two plain (non-``nn.Module``) methods for the duration of one call, the same
technique ``scripts/dump_reference_qwen3_tts_pytorch.py`` uses to intercept
``speech_tokenizer.decode``.

Two properties of this family shape the script.

**The oracle samples, seeded, exactly like the CustomVoice sibling -- it does
not go greedy.** That was tried here first and abandoned within an hour of GPU
time: with both sampling switches forced to ``False``, the first case run ran
away to 8191 frames (655 s of audio) for one short sentence, never sampling
the codec end token. `docs/porting/families/qwen3-tts.md` (around line 780)
already documents this exact failure mode -- greedy decoding "degenerates on
some speaker-and-input pairings" -- and every case in the Stage 1
(CustomVoice) Golden Manifest runs ``do_sample``/``subtalker_dosample`` both
``True`` at ``max_new_tokens=2048`` for exactly that reason. Reproducibility
does not require greedy: ``torch.manual_seed`` immediately before the call
reproduces the sampled trajectory run to run, and ``docs/port-validation.md``'s
stochastic-replay rule is what carries it into the port -- the oracle records
the sequence it sampled (``codes.semantic``/``codes.acoustic``) as the replay
input, and a validation-only seam replays those exact codes rather than
asking a C++ RNG to reproduce PyTorch's generator.

**Voice cloning has two modes with different artifact sets.** x-vector-only
mode ignores the reference transcript and reference codes entirely; only the
speaker embedding is used. ICL (in-context learning) mode requires the
reference transcript and conditions on the reference audio's own codes, so it
also has a reference code sequence and an ICL prompt embedding to capture.
``generate_icl_prompt`` -- the method that builds that embedding -- is called
only in ICL mode, which is what the case-shape checks below assert.

Usage (explicit-argument form -- the primary interface, and the only one
guaranteed to work until Task 3 lands the Golden Manifest):

    uv run --project scripts/envs/qwen3-tts --locked python \\
      scripts/dump_reference_qwen3_tts_base.py \\
      --weights-dir models/qwen3-tts-12hz-0-6b-base \\
      --ref-audio models/qwen3-tts-reference-audio/clone.wav \\
      --ref-text "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you." \\
      --text "This is a test of Qwen three T T S base voice cloning." \\
      --language English \\
      --out-dir build/goldens/qwen3-tts/base-icl-en

    # x-vector-only mode ignores --ref-text entirely.
    uv run ... --x-vector-only --ref-audio ... --text ... --language English \\
      --out-dir build/goldens/qwen3-tts/base-xvector-en

Manifest form (the primary interface now that Task 3's Golden Manifest exists;
a missing ``--manifest`` file is never required -- it is only opened when
``--manifest`` is actually passed). Field mapping: ``input.reference.artifact``
/ ``.transcript`` name the reference clip and its transcript, the same shape
OmniVoice's manifest uses for ``voice.kind: reference_audio``; ``oracle.parameters``
carries the mode switch (``x_vector_only``), the target ``language``, and
``trim_seconds`` for the two reference-duration-edge cases:

    uv run ... --manifest tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-base.manifest.json \\
      --weights-dir models/qwen3-tts-12hz-0-6b-base --case base-xvector-en
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
import time
import urllib.request
from typing import Any, Optional

import librosa
import numpy as np
import torch

MANIFEST_SCHEMA = "synthesize-golden-manifest-v1"
FAMILY = "qwen3-tts"
# 12.5 Hz codec at 24 kHz: encode_downsample_rate == decode_upsample_rate == 1920.
SAMPLES_PER_CODEC_FRAME = 1920


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--weights-dir", required=True, type=pathlib.Path)
    parser.add_argument(
        "--device", default="cuda:0",
        help="device_map forwarded to Qwen3TTSModel.from_pretrained (default: cuda:0). "
             "The talker checkpoint is stored bfloat16; see docs/port-validation.md, "
             "'Choosing the Oracle's dtype and Device'.",
    )

    # Explicit-argument form -- the primary interface.
    parser.add_argument("--ref-audio", default=None,
                         help="Reference clip: local wav path, URL, or base64 string.")
    parser.add_argument("--ref-text", default=None,
                         help="Reference transcript. Required unless --x-vector-only.")
    parser.add_argument("--text", default=None, help="Text to synthesize.")
    parser.add_argument("--language", default=None, help="Target language, e.g. English.")
    parser.add_argument("--x-vector-only", action="store_true",
                         help="Clone with only the speaker embedding; --ref-text is ignored.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=None,
                         help="Case output directory for the explicit-argument form.")
    parser.add_argument(
        "--trim-seconds", type=float, default=None,
        help="Make the reference clip exactly this many seconds before cloning, by "
             "truncating it or, if it is shorter than requested, looping it. Used to "
             "measure the reference-duration bounds without sourcing new audio.",
    )
    parser.add_argument("--seed", type=int, default=0,
                         help="torch.manual_seed applied immediately before generate_voice_clone. "
                              "Decoding samples (do_sample=True, subtalker_dosample=True), so this "
                              "seed is what makes the dump regeneratable -- see docs/port-validation.md's "
                              "stochastic-replay rule.")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                         help="Codec-frame cap forwarded to generate_voice_clone. Defaults to 2048, "
                              "matching every Stage 1 (CustomVoice) manifest case.")

    # Manifest form -- forward-compatible, not required to exist.
    parser.add_argument("--manifest", type=pathlib.Path, default=None,
                         help="Golden Manifest to read cases from. Never opened unless passed.")
    parser.add_argument("--case", action="append", default=None,
                         help="--manifest form only: restrict to this case id (repeatable).")
    parser.add_argument("--output-root", type=pathlib.Path, default=None,
                         help="--manifest form only: root directory for case subdirectories; "
                              "defaults to the manifest's case_artifact_root.")

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
    captured value passes through here first.
    """
    if torch.is_tensor(value):
        tensor = value.detach()
        if dtype is not None:
            tensor = tensor.to(dtype)
        return tensor.cpu().numpy()
    return np.asarray(value)


@dataclasses.dataclass
class RunCase:
    id: str
    ref_audio: str
    text: str
    language: str
    x_vector_only: bool
    ref_text: Optional[str] = None
    trim_seconds: Optional[float] = None
    seed: int = 0
    max_new_tokens: int = 2048
    # Set only by the manifest form, from the matching `source.artifacts`
    # entry (role reference-audio). The explicit-argument form has no
    # manifest to pin a digest against, so this stays None there -- same as
    # before this field existed.
    ref_sha256: Optional[str] = None


def build_case_from_args(args: argparse.Namespace) -> tuple[RunCase, pathlib.Path]:
    missing = [
        name for name, value in (
            ("--ref-audio", args.ref_audio),
            ("--text", args.text),
            ("--language", args.language),
            ("--out-dir", args.out_dir),
        ) if value is None
    ]
    if missing:
        raise SystemExit(
            "explicit-argument form requires " + ", ".join(missing) +
            " (or pass --manifest together with --case)"
        )
    if not args.x_vector_only and not args.ref_text:
        raise SystemExit(
            "--ref-text is required unless --x-vector-only is set: ICL mode conditions on "
            "the reference transcript, and x_vector_only_mode=False without one is a "
            "upstream ValueError, not a usable oracle run."
        )
    case = RunCase(
        id=args.out_dir.name,
        ref_audio=args.ref_audio,
        text=args.text,
        language=args.language,
        x_vector_only=bool(args.x_vector_only),
        ref_text=None if args.x_vector_only else args.ref_text,
        trim_seconds=args.trim_seconds,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
    )
    return case, args.out_dir


def load_cases_from_manifest(args: argparse.Namespace) -> list[tuple[RunCase, pathlib.Path]]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise SystemExit(f"{args.manifest}: unexpected manifest schema {manifest.get('schema')!r}")
    if manifest.get("family") != FAMILY:
        raise SystemExit(f"{args.manifest}: not a {FAMILY} manifest")

    output_root = args.output_root or pathlib.Path(manifest["case_artifact_root"])
    wanted = set(args.case) if args.case else None
    cases = [c for c in manifest["cases"] if wanted is None or c["id"] in wanted]
    if wanted and len(cases) != len(wanted):
        missing = wanted - {c["id"] for c in cases}
        raise SystemExit(f"unknown case id(s): {sorted(missing)}")

    specs: list[tuple[RunCase, pathlib.Path]] = []
    for case in cases:
        # Task 3 landed the real Base manifest schema. Every reference-audio
        # case names its clip and transcript under `input.reference` -- the
        # same shape OmniVoice's manifest uses for `voice.kind: reference_audio`
        # -- not under `voice` or `oracle.parameters`; this mapping was
        # corrected against that real shape instead of the blind guess the
        # comment here used to describe. `oracle.parameters.ref_audio` remains
        # as a fallback for a manifest that has no `input.reference` at all.
        params = case.get("oracle", {}).get("parameters", {})
        reference = case.get("input", {}).get("reference", {})
        ref_audio = reference.get("artifact") or params.get("ref_audio")
        if ref_audio is None:
            raise SystemExit(
                f"{case['id']}: manifest case has no input.reference.artifact / "
                "oracle.parameters.ref_audio"
            )
        x_vector_only = bool(params.get("x_vector_only", False))
        ref_sha256 = None
        for artifact in manifest.get("source", {}).get("artifacts", []):
            if artifact.get("role") == "reference-audio" and artifact.get("locator") == ref_audio:
                ref_sha256 = artifact.get("sha256")
                break
        run_case = RunCase(
            id=case["id"],
            ref_audio=ref_audio,
            text=case["input"]["text"],
            language=params.get("language", "Auto"),
            x_vector_only=x_vector_only,
            ref_text=None if x_vector_only else params.get("ref_text"),
            trim_seconds=params.get("trim_seconds"),
            seed=int(case.get("request", {}).get("seed_u64", 0)),
            max_new_tokens=params.get("max_new_tokens", 2048),
            ref_sha256=ref_sha256,
        )
        specs.append((run_case, output_root / case["id"]))
    return specs


def resolve_reference_locator(
    locator: str, expected_sha256: Optional[str], cache_dir: pathlib.Path
) -> str:
    """Fetch an http(s) reference-audio locator to a local cache, verified by digest.

    ``librosa.load`` cannot open a bare URL -- it hands unrecognised paths to
    ``soundfile``/``audioread``, both of which expect a local file and raise
    ``FileNotFoundError``. The Golden Manifest's ``input.reference.artifact``
    is the canonical upstream URL (``test_reference_audio_cases_name_a_pinned_source_artifact``
    requires it to match a pinned ``source.artifacts`` locator exactly), so
    that URL has to become a local path before ``load_reference_audio`` sees
    it. This mirrors ``materialise_reference`` in
    ``scripts/dump_reference_omnivoice_pytorch.py``, the established pattern
    in this repo for the identical problem: cache under a fixed local
    directory, skip re-fetching what is already there, and verify content by
    digest rather than trusting the URL to keep serving the same bytes.
    A local path (the explicit-argument form's normal case) passes straight
    through unchanged.
    """
    if not (locator.startswith("http://") or locator.startswith("https://")):
        return locator
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / locator.rsplit("/", 1)[-1]
    if not destination.exists():
        print(f"fetching {locator}", flush=True)
        with urllib.request.urlopen(locator, timeout=60) as response:  # noqa: S310 - pinned https locator
            destination.write_bytes(response.read())
    if expected_sha256 is not None:
        actual = hashlib.sha256(destination.read_bytes()).hexdigest()
        if actual != expected_sha256:
            raise SystemExit(
                f"{destination}: sha256 {actual} does not match the manifest's "
                f"{expected_sha256}. Refusing to dump a clone case against an "
                "unpinned reference. If a stale cached file is the cause, delete "
                "it and re-run to re-fetch."
            )
    return str(destination)


def load_reference_audio(source: str, trim_seconds: Optional[float]) -> tuple[np.ndarray, int, dict]:
    wav, sr = librosa.load(source, sr=None, mono=True)
    wav = wav.astype(np.float32)
    info: dict = {"source_samples": int(wav.shape[0]), "source_sample_rate": int(sr)}
    if trim_seconds is not None:
        target = int(round(trim_seconds * sr))
        if target <= 0:
            raise SystemExit(f"--trim-seconds must be positive, got {trim_seconds}")
        if target <= wav.shape[0]:
            wav = wav[:target]
            info["looped"] = False
            info["loop_repeats"] = 1
        else:
            # The clip is shorter than requested. Loop it rather than sourcing new
            # audio, per the brief: the measurement is about the requested
            # duration, not about finding a longer recording.
            repeats = -(-target // wav.shape[0])  # ceil division
            wav = np.tile(wav, repeats)[:target]
            info["looped"] = True
            info["loop_repeats"] = int(repeats)
        info["requested_seconds"] = trim_seconds
        info["applied_samples"] = int(wav.shape[0])
    return wav, sr, info


def capture_icl_embed(model, sink: dict):
    """Wrap ``generate_icl_prompt`` for the duration of one call.

    It is a plain method on the underlying ``Qwen3TTSForConditionalGeneration``,
    not an ``nn.Module``, so there is nothing to register a forward hook on;
    wrapping it for the call is the same technique the CustomVoice sibling uses
    for ``speech_tokenizer.decode``. It fires only in ICL mode -- x-vector-only
    cases never call it, which is what the case-shape check in ``run_case``
    verifies.
    """
    target = model.model
    original = target.generate_icl_prompt

    def wrapper(*args, **kwargs):
        icl_input_embed, trailing_text_hidden = original(*args, **kwargs)
        if "icl_embed" not in sink:
            sink["icl_embed"] = to_numpy(icl_input_embed, torch.float32)
        return icl_input_embed, trailing_text_hidden

    target.generate_icl_prompt = wrapper
    return lambda: setattr(target, "generate_icl_prompt", original)


def capture_generated_codes(model, sink: dict):
    """Wrap the talker's ``generate`` to capture the newly generated codes.

    ``generate_voice_clone`` concatenates the reference codes onto the front of
    this before handing it to the codec, so intercepting the codec's ``decode``
    input (as the CustomVoice sibling does) would mix the reference codes back
    into ``codes.semantic``/``codes.acoustic`` -- this family already has a
    separate probe for those (``codes/reference.i32``, captured from
    ``prompt_items[0].ref_code`` before synthesis). Capturing here instead keeps
    the two probes disjoint: reference codes describe the clip that was cloned,
    ``codes.semantic``/``codes.acoustic`` describe what was newly synthesized.
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


def run_case(model, case: RunCase, case_dir: pathlib.Path) -> dict:
    local_ref_audio = resolve_reference_locator(
        case.ref_audio,
        case.ref_sha256,
        pathlib.Path("models/qwen3-tts-reference-audio"),
    )
    ref_wav, ref_sr, ref_info = load_reference_audio(local_ref_audio, case.trim_seconds)

    prompt_items = model.create_voice_clone_prompt(
        ref_audio=(ref_wav, ref_sr),
        ref_text=None if case.x_vector_only else case.ref_text,
        x_vector_only_mode=case.x_vector_only,
    )

    artifacts: dict[str, dict] = {}

    # Dumped before synthesis, in this order, so a failure downstream still
    # leaves the deterministic stages on disk.
    artifacts["speaker.x_vector"] = write_f32(
        case_dir / "speaker" / "x_vector.f32",
        to_numpy(prompt_items[0].ref_spk_embedding, torch.float32),
    )
    if not case.x_vector_only:
        reference_codes = to_numpy(prompt_items[0].ref_code)
        if reference_codes.ndim != 2 or reference_codes.shape[1] != 16:
            raise SystemExit(
                f"{case.id}: expected [frames, 16] reference codes, got {reference_codes.shape}"
            )
        artifacts["codes.reference"] = write_i32(case_dir / "codes" / "reference.i32", reference_codes)

    sink: dict = {}
    restore_icl = capture_icl_embed(model, sink)
    restore_codes = capture_generated_codes(model, sink)
    started = time.time()
    try:
        # Sampled, not greedy -- seeding immediately before the call is what
        # makes the dump regeneratable. See the module docstring: greedy was
        # tried first and abandoned after one case ran away to 8191 frames.
        torch.manual_seed(case.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(case.seed)
        gen_kwargs: dict[str, Any] = dict(
            text=case.text,
            language=case.language,
            voice_clone_prompt=prompt_items,
            do_sample=True,
            subtalker_dosample=True,
            max_new_tokens=case.max_new_tokens,
        )
        wavs, sample_rate = model.generate_voice_clone(**gen_kwargs)
    finally:
        restore_icl()
        restore_codes()
    wall_seconds = time.time() - started

    if case.x_vector_only:
        if "icl_embed" in sink:
            raise SystemExit(
                f"{case.id}: x-vector-only case unexpectedly built an ICL prompt embedding"
            )
    else:
        if "icl_embed" not in sink:
            raise SystemExit(f"{case.id}: ICL case never built an ICL prompt embedding")
        artifacts["prompt.icl_embed"] = write_f32(case_dir / "prompt" / "icl_embed.f32", sink["icl_embed"])

    if "codes" not in sink:
        raise SystemExit(f"{case.id}: generated talker codes were never seen leaving model.generate")
    codes = sink["codes"]
    if codes.ndim != 2 or codes.shape[1] != 16:
        raise SystemExit(f"{case.id}: expected [frames, 16] generated codes, got {codes.shape}")
    # Column 0 is the semantic codebook (rvq_first); columns 1..15 are the
    # acoustic ones (rvq_rest), matching the CustomVoice dumper's convention.
    artifacts["codes.semantic"] = write_i32(case_dir / "codes" / "semantic.i32", codes[:, 0])
    artifacts["codes.acoustic"] = write_i32(case_dir / "codes" / "acoustic.i32", codes[:, 1:])

    # A run that reaches the cap never emitted EOS, so it is not oracle evidence
    # regardless of how plausible the tail looks -- same guard as the sibling.
    # This is exactly the failure mode greedy decoding hit during development
    # (8191 of an 8192 cap, no EOS ever sampled): the guard exists so a runaway
    # case is caught immediately rather than shipped as a "long" golden case.
    effective_cap = case.max_new_tokens
    if codes.shape[0] >= effective_cap - 1:
        tail = codes[-min(400, codes.shape[0]):, 0]
        raise SystemExit(
            f"{case.id}: generation reached max_new_tokens ({codes.shape[0]} of {effective_cap} "
            f"frames) without emitting EOS; last {tail.size} frames hold "
            f"{len(set(tail.tolist()))} distinct semantic code(s). Choose an input this variant "
            f"terminates on, or record the case as a known non-terminating input."
        )

    audio = np.asarray(wavs[0], dtype=np.float32)
    if not np.isfinite(audio).all():
        raise SystemExit(f"{case.id}: reference produced non-finite PCM")
    artifacts["audio.pcm"] = write_f32(case_dir / "audio.pcm", audio)

    # generate_voice_clone already trims the reference clip's own audio off the
    # front for ICL cases, so this is the same one-frame-per-1920-samples
    # identity the sibling checks -- it ties codes.semantic/acoustic to audio.pcm
    # for both modes, not just x-vector-only.
    if audio.shape[0] != codes.shape[0] * SAMPLES_PER_CODEC_FRAME:
        raise SystemExit(
            f"{case.id}: {codes.shape[0]} generated code frames imply "
            f"{codes.shape[0] * SAMPLES_PER_CODEC_FRAME} samples, but PCM has {audio.shape[0]}"
        )

    duration = float(audio.shape[0]) / float(sample_rate)
    semantic_codes = codes[:, 0]
    counts = np.bincount(semantic_codes.astype(np.int64))
    result: dict[str, Any] = {
        "case": case.id,
        "status": "ok",
        "sample_rate": int(sample_rate),
        "frames": int(audio.shape[0]),
        "generated_code_frames": int(codes.shape[0]),
        "duration_seconds": duration,
        "peak_abs": float(np.abs(audio).max()) if audio.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0,
        # Objective degeneracy signals, not a perceptual judgement -- see this
        # family's documented greedy trap (a fixed point repeating one code).
        "distinct_semantic_codes": int(np.count_nonzero(counts)),
        "most_common_semantic_code_fraction": (
            float(counts.max()) / float(semantic_codes.size) if semantic_codes.size else 0.0
        ),
        "wall_seconds": round(wall_seconds, 3),
        "real_time_factor": round(wall_seconds / duration, 3) if duration else None,
    }
    if not case.x_vector_only:
        reference_elements = artifacts["codes.reference"]["elements"]
        if reference_elements % 16 != 0:
            raise SystemExit(
                f"{case.id}: reference codes element count {reference_elements} is not a "
                "multiple of 16"
            )
        result["reference_code_frames"] = reference_elements // 16
        result["reference_audio"] = ref_info

    metadata = {
        "ref_audio": case.ref_audio,
        "ref_text": case.ref_text,
        "text": case.text,
        "language": case.language,
        "x_vector_only": case.x_vector_only,
        "seed": case.seed,
        "decoding": {
            "do_sample": True,
            "subtalker_dosample": True,
            "max_new_tokens": effective_cap,
        },
    }
    write_json(case_dir / "result.json", result)
    write_json(case_dir / "metadata.json", metadata)
    artifacts["result"] = {"path": "result.json"}
    artifacts["metadata"] = {"path": "metadata.json"}
    return {"id": case.id, "result": result, "artifacts": artifacts}


def main() -> int:
    args = parse_args()

    if args.manifest is not None:
        run_specs = load_cases_from_manifest(args)
    else:
        run_specs = [build_case_from_args(args)]

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"--device {args.device!r} requests CUDA but no CUDA device is available")

    from qwen_tts import Qwen3TTSModel

    # Driven by the checkpoint's own stored dtype rather than hardcoded as a
    # blanket default: the talker weights are bf16, so float32 would upcast
    # every talker weight and capture a reference for a model that does not
    # exist. See docs/port-validation.md, "Choosing the Oracle's dtype and
    # Device".
    model = Qwen3TTSModel.from_pretrained(
        str(args.weights_dir),
        device_map=args.device,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    print(f"reference: {args.device} bfloat16 eager", flush=True)

    records = []
    for index, (case, case_dir) in enumerate(run_specs, 1):
        print(f"[{index}/{len(run_specs)}] {case.id}", flush=True)
        records.append(run_case(model, case, case_dir))
        print(f"    {records[-1]['result']['frames']} frames, "
              f"rtf {records[-1]['result']['real_time_factor']}", flush=True)

    report = {
        "schema": "synthesize-oracle-dump-v1",
        "family": FAMILY,
        "variant": "qwen3-tts-12hz-0-6b-base",
        "reference": {"device": args.device, "dtype": "bfloat16", "attn_implementation": "eager"},
        "case_count": len(records),
        "cases": records,
    }
    if args.report:
        write_json(args.report, report)
    print(f"dumped {len(records)} case(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
