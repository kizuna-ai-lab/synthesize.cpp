#!/usr/bin/env python3
"""Dump Qwen3-TTS-Base's mel front end and every ECAPA-TDNN stage, not just the x-vector.

The eight pre-existing ``scripts/dump_reference_qwen3_tts_*.py`` scripts dump
only the finished ``[1024]`` speaker embedding (``speaker/x_vector.f32``) and
nothing between the reference waveform and it. Until this script exists, a
wrong mel and a wrong ECAPA forward can cancel to a plausible x-vector and
nothing says which stage moved. This script drives the pinned upstream
``Qwen3TTSModel.create_voice_clone_prompt(..., x_vector_only_mode=True)`` --
the same entry point ``scripts/dump_reference_qwen3_tts_base.py`` uses for its
``speaker.x_vector`` artifact -- and captures every intermediate on the way
past, by two techniques:

  * The six ECAPA-TDNN stages (``blocks[0..3]``, ``mfa``, ``asp``) are plain
    ``nn.Module`` instances, so a forward hook captures each one's output
    without changing the forward at all.
  * The mel spectrogram itself is computed by a bare module-level function
    (``mel_spectrogram`` in ``modeling_qwen3_tts.py``), called inline from
    ``Qwen3TTSModel.extract_speaker_embedding`` rather than through a module --
    there is nothing to register a forward hook on. It is wrapped for the
    duration of one call instead, the same technique
    ``dump_reference_qwen3_tts_base.py`` uses to intercept
    ``generate_icl_prompt``/``generate``.

Every mel convention that silently changes the answer -- the mel-scale
formula, filterbank normalization, magnitude vs. power, the log base and
floor, the window type and its zero-padding rule, and the padding/centering
scheme -- is recorded in ``conventions.json`` beside the artifacts, each
traced to its upstream ``file:line`` (or, where upstream's behaviour is not
observable from the outside, to a direct run of upstream's own
``mel_spectrogram`` on a synthetic input). ``docs/port-validation.md``'s rule
for choosing the oracle's dtype and device applies here exactly as it does to
the sibling script: this checkpoint's own bfloat16, on CUDA, with
``attn_implementation="eager"`` -- CPU/F32 would describe a model that does
not exist.

This front end has no sampling anywhere in it (``torch.inference_mode()``,
eval-mode Conv1d/BatchNorm-free ECAPA blocks, an argmin-nearest quantizer
encode) -- unlike ``dump_reference_qwen3_tts_base.py``'s full
``generate_voice_clone`` call, there is no seed to pin for reproducibility.

Usage (explicit-argument form):

    uv run --project scripts/envs/qwen3-tts --locked python \\
      scripts/dump_reference_qwen3_tts_speaker.py \\
      --weights-dir models/qwen3-tts-12hz-0-6b-base \\
      --ref-audio models/qwen3-tts-reference-audio/clone.wav \\
      --out-dir build/goldens/qwen3-tts/qwen3-tts-12hz-0-6b-base/base-xvector-en

Manifest form (the primary interface -- the Base Golden Manifest already
names the reference clip under ``input.reference.artifact``, the same shape
``dump_reference_qwen3_tts_base.py`` reads):

    uv run ... scripts/dump_reference_qwen3_tts_speaker.py \\
      --weights-dir models/qwen3-tts-12hz-0-6b-base \\
      --manifest tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-base.manifest.json \\
      --case base-xvector-en
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

UPSTREAM_REPOSITORY = "https://github.com/QwenLM/Qwen3-TTS"
UPSTREAM_REVISION = "022e286b98fbec7e1e916cb940cdf532cd9f488e"
UPSTREAM_FILE = "qwen_tts/core/models/modeling_qwen3_tts.py"

# Each stage's output on the way past. Upstream returns only the finished
# embedding, so a stage-wise comparison has no reference at all without
# these -- which is why the port's mel and its ECAPA forward could each be
# wrong and still agree at the output. A hook is a read, not a change:
# nothing here alters the forward.
SPEAKER_TAPS = {
    "speaker.blocks0": "speaker_encoder.blocks.0",
    "speaker.block1": "speaker_encoder.blocks.1",
    "speaker.block2": "speaker_encoder.blocks.2",
    "speaker.block3": "speaker_encoder.blocks.3",
    "speaker.mfa": "speaker_encoder.mfa",
    "speaker.asp": "speaker_encoder.asp",
}

# speaker/<tap-key-without-"speaker."-prefix>.f32
TAP_FILENAMES = {
    "speaker.blocks0": "blocks0.f32",
    "speaker.block1": "block1.f32",
    "speaker.block2": "block2.f32",
    "speaker.block3": "block3.f32",
    "speaker.mfa": "mfa.f32",
    "speaker.asp": "asp.f32",
}


def write_f32(path: pathlib.Path, array: np.ndarray) -> dict:
    data = np.ascontiguousarray(array, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.tobytes())
    return {"path": path.name, "shape": list(data.shape), "elements": int(data.size)}


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def to_numpy(value: Any, dtype: Optional[torch.dtype] = None) -> np.ndarray:
    """Detach and move a possibly-GPU, possibly-bf16 tensor to a plain host array.

    ``write_f32`` calls ``np.ascontiguousarray`` directly, which cannot see a
    CUDA tensor or a dtype numpy has no analogue for (bf16); every captured
    value passes through here first.
    """
    if torch.is_tensor(value):
        tensor = value.detach()
        if dtype is not None:
            tensor = tensor.to(dtype)
        return tensor.cpu().numpy()
    return np.asarray(value)


def install_taps(model, captured):
    """Capture each ECAPA stage's output on the way past.

    Upstream returns only the finished embedding, so a stage-wise comparison
    has no reference at all without these -- which is why the port's mel and
    its ECAPA forward could each be wrong and still agree at the output. A
    hook is a read, not a change: nothing here alters the forward.
    """
    handles = []
    modules = dict(model.named_modules())
    for name, path in SPEAKER_TAPS.items():
        module = modules.get(path)
        if module is None:
            raise SystemExit(f"upstream has no module {path!r}; the taps are stale")
        handles.append(module.register_forward_hook(
            lambda _m, _i, out, key=name: captured.__setitem__(key, to_numpy(out, torch.float32))))
    return handles


def capture_mel(sink: dict):
    """Wrap the module-level ``mel_spectrogram`` function for one call.

    Upstream computes the mel spectrogram inline inside
    ``Qwen3TTSModel.extract_speaker_embedding`` (modeling_qwen3_tts.py:1943)
    rather than through an ``nn.Module`` -- there is nothing to register a
    forward hook on. Wrapping it for the call's duration is the same
    technique ``dump_reference_qwen3_tts_base.py`` uses for
    ``generate_icl_prompt``/``generate``: a plain function is monkey-patched
    on the module that defines it, called, and restored.
    """
    import qwen_tts.core.models.modeling_qwen3_tts as qwen3_tts_modeling
    original = qwen3_tts_modeling.mel_spectrogram

    def wrapper(*args, **kwargs):
        mel = original(*args, **kwargs)
        # Last-wins, matching install_taps's forward hooks (captured.__setitem__
        # unconditionally overwrites). Only one call happens per case today, so
        # this can't yet be observed to matter -- but if a case ever cloned more
        # than one reference in a single create_voice_clone_prompt call, a
        # first-wins guard here would silently pair this case's mel with a
        # different item's ECAPA taps than the last-wins hooks captured.
        sink["mel"] = to_numpy(mel, torch.float32)
        y = args[0] if args else kwargs["y"]
        sink["mel_input_samples"] = int(y.shape[-1])
        return mel

    qwen3_tts_modeling.mel_spectrogram = wrapper
    return lambda: setattr(qwen3_tts_modeling, "mel_spectrogram", original)


def build_conventions(frames_observed: int, frames_observed_pcm_samples: Optional[int]) -> dict:
    """The six mel conventions this script observed off the pinned upstream
    source, plus the frame-count constants Task 2's unit tests need.

    Every value here was either read verbatim from ``modeling_qwen3_tts.py``
    at the cited line, confirmed against the installed library's documented
    default (librosa 0.11.0, torch's own docstring), or directly observed by
    running upstream's own ``mel_spectrogram`` on a synthetic input -- never
    assumed. A convention not recorded here, with its provenance, was
    guessed.
    """
    return {
        "schema": "synthesize-qwen3-tts-speaker-conventions-v1",
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "revision": UPSTREAM_REVISION,
            "file": UPSTREAM_FILE,
        },
        # The five numbers the package already names -- these alone do not
        # determine a mel spectrogram; the fields below do.
        "mel_bins": 128,
        "n_fft": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "fmin": 0,
        "fmax": 12000,
        "sample_rate": 24000,
        # The six conventions.
        "mel_scale": "slaney",
        "filterbank_normalization": "slaney_area",
        "spectrum_magnitude": "amplitude",
        "magnitude_epsilon": 1e-9,
        "log_base": "natural",
        "log_floor": 1e-5,
        "log_scale_before_log": 1.0,
        "window_type": "hann",
        "window_periodic": True,
        "window_length_lt_n_fft_rule": "zero_padded_centered",
        "centered": False,  # the literal `center` argument torch.stft receives; see padding_note.
        "manual_pad_each_side_formula": "(n_fft - hop_length) // 2",
        "manual_pad_mode": "reflect",
        "frames_formula": (
            "1 + ((len(pcm) + 2 * ((n_fft - hop_length) // 2) - n_fft) // hop_length)"
        ),
        "padding_note": (
            "This is neither a naive center=True STFT (reflect-pad n_fft//2 each side, "
            "giving frames = 1 + len(pcm)//hop_length) nor an uncentered one (no pad, "
            "frames = (len(pcm) - n_fft)//hop_length + 1). Upstream pads manually by "
            "(n_fft - hop_length)//2 samples on each side with reflect padding, then "
            "calls torch.stft with center=False on the already-padded signal. For "
            "hop_length=256 that pad is 384 samples per side, not the 512 a naive "
            "center=True would use -- so neither formula a reader might guess from "
            "'centered' alone is right. Settled by directly running upstream's "
            "mel_spectrogram on synthetic zero-signal inputs, not by formula alone; see "
            "frames_for_one_second / frames_for_one_second_at_hop_512 / frames_observed."
        ),
        "mel_computed_by": {
            "kind": "function",
            "qualname": "qwen_tts.core.models.modeling_qwen3_tts.mel_spectrogram",
            "note": (
                "Computed inline by a plain module-level function, not an nn.Module -- "
                "there is nothing to register a forward hook on, so this script wraps "
                "the function itself for the duration of the call (capture_mel)."
            ),
        },
        # Frame counts. frames_for_one_second and frames_for_one_second_at_hop_512
        # were each observed by calling upstream's own mel_spectrogram on a
        # synthetic all-zero input, not derived from a formula this script chose;
        # frames_observed is this case's own dumped mel.f32, likewise observed
        # rather than predicted.
        "frames_for_one_second": 93,
        "frames_for_one_second_at_hop_512": 46,
        "frames_observed": frames_observed,
        "frames_observed_pcm_samples": frames_observed_pcm_samples,
        # On-disk layout: every artifact here is written by write_f32, which
        # np.ascontiguousarray()s the array (row-major / C order) and then calls
        # .tobytes() once -- no header, native little-endian float32.
        "mel_layout": {
            "shape": ["mel_bins", "frames"],
            "order": "row_major_c",
            "note": (
                "frames is the fast-varying (contiguous) axis: the element at "
                "(bin, frame) is float32 #(bin * frames + frame), i.e. all frames "
                "of bin 0 precede all frames of bin 1. This is only in result.json "
                "as a shape list today ([128, 757]); recorded here too because "
                "conventions.json, not result.json, is the file Task 2 implements "
                "against."
            ),
        },
        "min_pcm_samples": {
            "formula": "(n_fft - hop_length) // 2 + 1",
            "value_at_production_hop": 385,
            "note": (
                "Below this, upstream's own manual reflect pad -- "
                "F.pad(y, (pad, pad), mode='reflect') with pad = (n_fft-hop_length)//2 "
                "-- fails outright: torch requires the padded dimension's length to "
                "exceed the pad width on each side, not merely reach it. Verified "
                "directly: torch.nn.functional.pad on a synthetic input raises at "
                "pcm_samples in {383, 384} ('Padding size should be less than the "
                "corresponding input dimension') and succeeds at 385, for the "
                "production hop_length=256 (pad=384). A C++ mel front end must "
                "reject an input at or below this length itself -- with a domain "
                "error, not by letting the reflect-pad step fail however it fails -- "
                "before it ever reaches a frame-count check."
            ),
        },
        "measurement_coverage": {
            "note": (
                "base-xvector-en and base-xvector-zh -- the only two Plan-2-runnable "
                "cases in this manifest -- both reference the same clip "
                "(models/qwen3-tts-reference-audio/clone.wav) with no --trim-seconds; "
                "only the synthesis text and language differ, and the speaker path "
                "this script dumps never reads either. Every file under speaker/ is "
                "therefore byte-identical between the two cases (confirmed by sha256 "
                "across all eight files). A second case run does not give Task 6 a "
                "second independent point on the speaker path: it is one measurement, "
                "taken twice. What the repeat does prove, and is worth having, is that "
                "the whole pipeline (load -> resample -> encode -> mel -> ECAPA -> "
                "x-vector) is deterministic case to case given the same input -- "
                "there is no hidden per-case state leaking in. A genuine second "
                "measurement would need a different reference clip, or this same clip "
                "trimmed to a different length via --trim-seconds (the encoder sees a "
                "different mel length either way); neither current case varies that. "
                "Not fixed here: the Golden Manifest is Plan 1's artifact, and adding "
                "a differently-scoped case is a later slice's decision, not this dump "
                "script's."
            ),
            "case_ids_observed_byte_identical": ["base-xvector-en", "base-xvector-zh"],
        },
        "artifact_dtypes": {
            "note": (
                "write_f32 always emits float32 on disk, but that on-disk dtype "
                "hides two different histories. mel.f32 never touched bfloat16 at "
                "all; every ECAPA/x-vector artifact is a bfloat16 value widened "
                "losslessly to float32. Tasks 4/5/6 must not budget the same "
                "tolerance for both -- a tolerance against the ECAPA/x-vector files "
                "that is tighter than bfloat16's own ~2^-8 relative precision is "
                "chasing the oracle's rounding, not the port's."
            ),
            "mel": {
                "runtime_dtype": "float32",
                "device": "cpu",
                "ever_bfloat16": False,
                "comparison_implication": (
                    "Bit-exact target. mel_spectrogram runs entirely on the CPU in "
                    "float32 (device = y.device at :433, and "
                    "torch.from_numpy(audio).unsqueeze(0) at :1944 is a CPU tensor "
                    "with no dtype cast). The .to(self.device).to(self.dtype) bf16 "
                    "cast at :1953 is applied to mel_spectrogram's RETURN VALUE, "
                    "after this script's capture_mel wrapper already recorded it. A "
                    "C++ mel front end may be compared against mel.f32 exactly "
                    "(modulo ordinary cross-implementation floating-point rounding "
                    "from a different FFT/DFT), not to a tolerance budgeted for "
                    "bf16 quantization -- there is none here to budget for."
                ),
            },
            "blocks0": {"runtime_dtype": "bfloat16", "ever_bfloat16": True},
            "block1": {"runtime_dtype": "bfloat16", "ever_bfloat16": True},
            "block2": {"runtime_dtype": "bfloat16", "ever_bfloat16": True},
            "block3": {"runtime_dtype": "bfloat16", "ever_bfloat16": True},
            "mfa": {"runtime_dtype": "bfloat16", "ever_bfloat16": True},
            "asp": {"runtime_dtype": "bfloat16", "ever_bfloat16": True},
            "x_vector": {"runtime_dtype": "bfloat16", "ever_bfloat16": True},
            "ecapa_and_x_vector_comparison_implication": (
                "Captured via forward hooks on speaker_encoder.<module>, which runs "
                "in the model's runtime dtype (torch.bfloat16 -- "
                "Qwen3TTSModel.from_pretrained(..., dtype=torch.bfloat16, ...)). "
                "to_numpy() upcasts to float32 for storage -- a lossless widening, "
                "not a new rounding -- but the value itself already carries "
                "bfloat16's ~2^-8 relative precision (7 explicit mantissa bits). "
                "Verified directly: casting each of blocks0/block1/block2/block3/"
                "mfa/asp/x_vector.f32 to bfloat16 and back to float32 round-trips "
                "exactly (max abs diff 0.0) for all seven; the same round-trip on "
                "mel.f32 does not (max abs diff ~0.031). A tolerance compared "
                "against these seven files should budget for bfloat16 quantization "
                "-- roughly 2^-8 relative error is the oracle's own, not the port's, "
                "and tightening a tolerance below that chases noise that isn't "
                "there to find."
            ),
        },
        # Extra: the ECAPA-TDNN topology behind the six taps this script installs,
        # for Task 4. Not part of the mel front end Task 2 implements against.
        "ecapa_topology": {
            "blocks": [
                {"tap": "speaker.blocks0", "module": "speaker_encoder.blocks.0",
                 "kind": "TimeDelayNetBlock", "in_channels": 128, "out_channels": 512,
                 "kernel_size": 5, "dilation": 1},
                {"tap": "speaker.block1", "module": "speaker_encoder.blocks.1",
                 "kind": "SqueezeExcitationRes2NetBlock", "in_channels": 512,
                 "out_channels": 512, "kernel_size": 3, "dilation": 2},
                {"tap": "speaker.block2", "module": "speaker_encoder.blocks.2",
                 "kind": "SqueezeExcitationRes2NetBlock", "in_channels": 512,
                 "out_channels": 512, "kernel_size": 3, "dilation": 3},
                {"tap": "speaker.block3", "module": "speaker_encoder.blocks.3",
                 "kind": "SqueezeExcitationRes2NetBlock", "in_channels": 512,
                 "out_channels": 512, "kernel_size": 3, "dilation": 4},
            ],
            "mfa": {"tap": "speaker.mfa", "module": "speaker_encoder.mfa",
                     "kind": "TimeDelayNetBlock", "in_channels": 1536, "out_channels": 1536,
                     "kernel_size": 1, "dilation": 1,
                     "note": "input is torch.cat(blocks[1:], dim=1) = 512*3 = 1536 channels"},
            "asp": {"tap": "speaker.asp", "module": "speaker_encoder.asp",
                    "kind": "AttentiveStatisticsPooling", "attention_channels": 128,
                    "output_channels": 3072, "output_frames": 1},
            "fc": {"module": "speaker_encoder.fc", "kind": "Conv1d",
                   "in_channels": 3072, "out_channels": 1024,
                   "note": ("not tapped separately -- its output, squeezed, is the "
                             "x_vector artifact already dumped.")},
        },
        "provenance": {
            "mel_scale": (
                f"{UPSTREAM_FILE}:435-437 calls librosa.filters.mel(sr=..., n_fft=..., "
                "n_mels=..., fmin=..., fmax=...) with no htk kwarg; confirmed against "
                "the installed librosa (0.11.0) filters.mel signature, which defaults "
                "htk=False -- the Slaney mel scale, not HTK's 2595*log10(1+f/700)."
            ),
            "filterbank_normalization": (
                f"{UPSTREAM_FILE}:435-437, the same call, no norm kwarg; librosa 0.11.0's "
                "filters.mel defaults norm='slaney' (area normalization), not norm=None "
                "or a per-filter peak norm."
            ),
            "spectrum_magnitude": (
                f"{UPSTREAM_FILE}:459 -- spec = torch.sqrt(torch.view_as_real(spec)."
                "pow(2).sum(-1) + 1e-9): amplitude |X|, not power |X|^2, with epsilon "
                "1e-9 inside the sqrt."
            ),
            "log_base_and_floor": (
                f"{UPSTREAM_FILE}:396-397,462 -- dynamic_range_compression_torch(x, C=1, "
                "clip_val=1e-5) = torch.log(torch.clamp(x, min=clip_val) * C); natural "
                "log (torch.log, base e), floor 1e-5, called at line 462 with the "
                "default C=1."
            ),
            "window": (
                f"{UPSTREAM_FILE}:440 -- torch.hann_window(win_size) with no periodic "
                "kwarg; torch.hann_window defaults periodic=True (confirmed: "
                "torch.hann_window(N) equals torch.hann_window(N, periodic=True) and "
                "differs from torch.hann_window(N, periodic=False)). win_length == "
                "n_fft == 1024 always in this checkpoint's own call "
                "(extract_speaker_embedding passes win_size=n_fft=1024), so the "
                "win_length<n_fft zero-pad-centered rule is torch.stft's own documented "
                "contract (\"window will be padded on both sides to length n_fft\") "
                "rather than something this checkpoint's call exercises."
            ),
            "padding_and_centering": (
                f"{UPSTREAM_FILE}:442-445 (manual reflect pad of (n_fft-hop_length)//2 "
                "samples each side); :408 (`center: bool = False` -- the signature "
                "default that actually supplies the False, since "
                "extract_speaker_embedding's call at :1943-1952 never passes a "
                "center= argument); :453 (`center=center` -- merely forwards that "
                "already-resolved value into torch.stft, it is not itself where the "
                "False comes from). See padding_note above -- confirmed by directly "
                "running upstream's own mel_spectrogram on synthetic inputs, not by "
                "formula alone."
            ),
            "mel_computed_by": (
                f"Inline function call at {UPSTREAM_FILE}:1943-1952 "
                "(Qwen3TTSModel.extract_speaker_embedding), not an nn.Module -- wrapped "
                "for the call's duration by this script's capture_mel(), the same "
                "technique dump_reference_qwen3_tts_base.py uses for "
                "generate_icl_prompt/generate."
            ),
            "frames_for_one_second": (
                "Observed, not derived: called "
                "qwen_tts.core.models.modeling_qwen3_tts.mel_spectrogram("
                "torch.zeros(1, 24000), n_fft=1024, num_mels=128, sampling_rate=24000, "
                "hop_size=256, win_size=1024, fmin=0, fmax=12000) directly and read "
                "mel.shape[-1] == 93."
            ),
            "frames_for_one_second_at_hop_512": (
                "Observed, not derived: the same call with hop_size=512 (n_fft and "
                "win_size unchanged); mel.shape[-1] == 46."
            ),
            "frames_observed": (
                "Read directly off this case's own dumped speaker/mel.f32 "
                "(elements // mel_bins), for the reference clip this case actually "
                "cloned -- a target Task 2 must reproduce, not a formula it chose."
            ),
            "ecapa_topology": (
                f"{UPSTREAM_FILE}:311-393 (Qwen3TTSSpeakerEncoder.__init__/forward); "
                "configuration_qwen3_tts.py:47-67 (Qwen3TTSSpeakerEncoderConfig "
                "defaults: enc_channels=[512,512,512,512,1536], "
                "enc_kernel_sizes=[5,3,3,3,1], enc_dilations=[1,2,3,4,1], "
                "enc_attention_channels=128); "
                "models/qwen3-tts-12hz-0-6b-base/config.json's speaker_encoder_config "
                "overrides only enc_dim=1024 and sample_rate=24000."
            ),
            "mel_layout": (
                "Read from this script's own write path: dump_front_end_artifacts "
                "passes sink['mel'][0] (shape [mel_bins, frames]) to write_f32, "
                "which np.ascontiguousarray()s (row-major/C order) then calls "
                ".tobytes() once -- frames is therefore the contiguous axis."
            ),
            "min_pcm_samples": (
                "Observed, not derived from the frames_formula alone: called "
                "torch.nn.functional.pad(torch.zeros(1, 1, N), (384, 384), "
                "mode='reflect') directly for N in {383, 384, 385, 386, 400} at the "
                "production hop_length=256 (pad=(1024-256)//2=384) -- N in {383, 384} "
                "raised, N>=385 succeeded. The general formula (pad + 1) is read off "
                "that boundary, not assumed from reflect-padding rules in the "
                "abstract."
            ),
            "measurement_coverage": (
                "Observed: sha256 of every speaker/*.f32 file matched pairwise "
                "between build/goldens/.../base-xvector-en/speaker/ and "
                "base-xvector-zh/speaker/ after running this script on both cases "
                "from the same manifest; cross-checked against both cases' "
                "input.reference in the manifest naming the identical clip and "
                "carrying no trim_seconds parameter."
            ),
            "artifact_dtypes": (
                "Observed, not inferred from the model's stated dtype alone: cast "
                "each dumped speaker/*.f32 file to bfloat16 and back to float32 "
                "and compared to the original with torch.equal. mel.f32 differed "
                "(max abs diff ~0.031); blocks0/block1/block2/block3/mfa/asp/"
                "x_vector.f32 were all bit-identical to their round-tripped copies "
                "(max abs diff 0.0). Combined with the source reading in "
                "padding_and_centering and mel_computed_by above (the bf16 cast in "
                "extract_speaker_embedding applies to mel_spectrogram's return "
                "value, not to anything inside it) to explain why."
            ),
        },
    }


@dataclasses.dataclass
class RunCase:
    id: str
    ref_audio: str
    trim_seconds: Optional[float] = None
    # Set only by the manifest form, from the matching `source.artifacts` entry
    # (role reference-audio). The explicit-argument form has no manifest to pin
    # a digest against, so this stays None there.
    ref_sha256: Optional[str] = None


def build_case_from_args(args: argparse.Namespace) -> tuple[RunCase, pathlib.Path]:
    missing = [
        name for name, value in (("--ref-audio", args.ref_audio), ("--out-dir", args.out_dir))
        if value is None
    ]
    if missing:
        raise SystemExit(
            "explicit-argument form requires " + ", ".join(missing) +
            " (or pass --manifest together with --case)"
        )
    case = RunCase(id=args.out_dir.name, ref_audio=args.ref_audio, trim_seconds=args.trim_seconds)
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
        params = case.get("oracle", {}).get("parameters", {})
        reference = case.get("input", {}).get("reference", {})
        ref_audio = reference.get("artifact") or params.get("ref_audio")
        if ref_audio is None:
            raise SystemExit(
                f"{case['id']}: manifest case has no input.reference.artifact / "
                "oracle.parameters.ref_audio -- this script only dumps the "
                "reference-audio speaker path, not text-only synthesis."
            )
        ref_sha256 = None
        for artifact in manifest.get("source", {}).get("artifacts", []):
            if artifact.get("role") == "reference-audio" and artifact.get("locator") == ref_audio:
                ref_sha256 = artifact.get("sha256")
                break
        run_case = RunCase(
            id=case["id"],
            ref_audio=ref_audio,
            trim_seconds=params.get("trim_seconds"),
            ref_sha256=ref_sha256,
        )
        specs.append((run_case, output_root / case["id"]))
    return specs


def resolve_reference_locator(
    locator: str, expected_sha256: Optional[str], cache_dir: pathlib.Path
) -> str:
    """Fetch an http(s) reference-audio locator to a local cache, verified by digest.

    Mirrors ``dump_reference_qwen3_tts_base.py``'s function of the same name:
    cache under a fixed local directory, skip re-fetching what is already
    there, verify content by digest rather than trusting the URL to keep
    serving the same bytes. A local path passes straight through unchanged.
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
                f"{expected_sha256}. Refusing to dump against an unpinned reference. "
                "If a stale cached file is the cause, delete it and re-run to re-fetch."
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
            repeats = -(-target // wav.shape[0])  # ceil division
            wav = np.tile(wav, repeats)[:target]
            info["looped"] = True
            info["loop_repeats"] = int(repeats)
        info["requested_seconds"] = trim_seconds
        info["applied_samples"] = int(wav.shape[0])
    return wav, sr, info


def dump_front_end_artifacts(case_dir: pathlib.Path, sink: dict) -> dict:
    """Write mel.f32 and every ECAPA tap that has actually fired.

    Called both on the normal path and (with whatever partial ``sink`` a
    failed forward left behind) from an exception handler -- so a failure in
    the pooling stage or later still leaves the front end's reference on
    disk, per the brief.
    """
    artifacts: dict[str, dict] = {}
    if "mel" in sink:
        artifacts["speaker.mel"] = write_f32(case_dir / "speaker" / "mel.f32", sink["mel"][0])
    for key, filename in TAP_FILENAMES.items():
        if key in sink:
            artifacts[key] = write_f32(case_dir / "speaker" / filename, sink[key][0])
    return artifacts


def write_conventions_if_possible(case_dir: pathlib.Path, sink: dict) -> Optional[dict]:
    """Write conventions.json beside whatever front-end artifacts exist.

    Depends only on the mel capture (frames_observed derives from its
    shape) -- so it is safe to call from a partial-failure path where later
    stages (blocks/mfa/asp/fc) never ran. Without this, a pooling failure
    would leave mel.f32 and some tap files on disk with no contract file
    beside them, undoing the point of dumping the front end first.
    """
    if "mel" not in sink:
        return None
    frames_observed = int(sink["mel"].shape[-1])
    conventions = build_conventions(frames_observed, sink.get("mel_input_samples"))
    write_json(case_dir / "speaker" / "conventions.json", conventions)
    return {"path": "conventions.json"}


def run_case(model, case: RunCase, case_dir: pathlib.Path, reference_audio_dir: pathlib.Path) -> dict:
    local_ref_audio = resolve_reference_locator(case.ref_audio, case.ref_sha256, reference_audio_dir)
    ref_wav, ref_sr, ref_info = load_reference_audio(local_ref_audio, case.trim_seconds)

    sink: dict[str, Any] = {}
    restore_mel = capture_mel(sink)
    hook_handles = install_taps(model.model, sink)
    started = time.time()
    try:
        prompt_items = model.create_voice_clone_prompt(
            ref_audio=(ref_wav, ref_sr),
            ref_text=None,
            x_vector_only_mode=True,
        )
    except Exception:
        # Even a failure past the pooling stage leaves whatever the hooks
        # already saw as the front end's reference -- write it, and the
        # contract file beside it, before re-raising. Without the second
        # call, a pooling failure would leave mel.f32/tap files on disk with
        # no conventions.json next to them.
        dump_front_end_artifacts(case_dir, sink)
        write_conventions_if_possible(case_dir, sink)
        raise
    finally:
        restore_mel()
        for handle in hook_handles:
            handle.remove()
    wall_seconds = time.time() - started

    required = ["mel"] + list(SPEAKER_TAPS.keys())
    missing = [key for key in required if key not in sink]
    if missing:
        raise SystemExit(
            f"{case.id}: never observed {missing} -- the taps are stale or the "
            "forward path changed"
        )

    # Front end before the x-vector, per the brief: a failure extracting the
    # x-vector below still leaves these on disk.
    artifacts = dump_front_end_artifacts(case_dir, sink)

    x_vector = to_numpy(prompt_items[0].ref_spk_embedding, torch.float32)
    artifacts["speaker.x_vector"] = write_f32(case_dir / "speaker" / "x_vector.f32", x_vector)

    mel_bins = sink["mel"].shape[1]
    frames_observed = int(sink["mel"].shape[-1])
    conventions_artifact = write_conventions_if_possible(case_dir, sink)
    if conventions_artifact:
        artifacts["conventions"] = conventions_artifact

    result = {
        "case": case.id,
        "status": "ok",
        "mel_bins": int(mel_bins),
        "frames": frames_observed,
        "mel_input_samples": sink.get("mel_input_samples"),
        "x_vector_elements": int(x_vector.size),
        "reference_audio": ref_info,
        "wall_seconds": round(wall_seconds, 3),
        "artifacts": artifacts,
    }
    metadata = {
        "ref_audio": case.ref_audio,
        "x_vector_only": True,
        "reference": {"device": "cuda:0", "dtype": "bfloat16", "attn_implementation": "eager"},
    }
    write_json(case_dir / "result.json", result)
    write_json(case_dir / "metadata.json", metadata)
    artifacts["result"] = {"path": "result.json"}
    artifacts["metadata"] = {"path": "metadata.json"}
    return {"id": case.id, "result": result, "artifacts": artifacts}


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

    # Explicit-argument form.
    parser.add_argument("--ref-audio", default=None,
                         help="Reference clip: local wav path, URL, or base64 string.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=None,
                         help="Case output directory for the explicit-argument form.")
    parser.add_argument(
        "--trim-seconds", type=float, default=None,
        help="Make the reference clip exactly this many seconds before extracting the "
             "speaker embedding, by truncating it or, if shorter, looping it.",
    )

    # Manifest form.
    parser.add_argument("--manifest", type=pathlib.Path, default=None,
                         help="Golden Manifest to read cases from. Never opened unless passed.")
    parser.add_argument("--case", action="append", default=None,
                         help="--manifest form only: restrict to this case id (repeatable).")
    parser.add_argument("--output-root", type=pathlib.Path, default=None,
                         help="--manifest form only: root directory for case subdirectories; "
                              "defaults to the manifest's case_artifact_root.")
    parser.add_argument(
        "--reference-audio-dir",
        type=pathlib.Path,
        default=pathlib.Path("models/qwen3-tts-reference-audio"),
        help="Where an http(s) input.reference.artifact locator is fetched and cached "
             "(git-ignored). Unused for a local-path locator.",
    )

    parser.add_argument("--report", type=pathlib.Path, default=None)
    return parser.parse_args()


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
    # blanket default -- see docs/port-validation.md, "Choosing the Oracle's
    # dtype and Device". This is the same reference configuration
    # dump_reference_qwen3_tts_base.py uses.
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
        records.append(run_case(model, case, case_dir, args.reference_audio_dir))
        print(f"    mel {records[-1]['result']['frames']} frames, "
              f"wall {records[-1]['result']['wall_seconds']}s", flush=True)

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
