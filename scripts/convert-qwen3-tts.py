#!/usr/bin/env python3
"""Convert the pinned Qwen3-TTS CustomVoice checkpoint to a source-dtype GGUF.

Stage 3 of the porting pipeline. This emits the source/reference-dtype artifact
only; F16 and every Quantization Profile are produced later by the C++
quantizer from this file, so no quantization policy belongs here.

The reference dtype for this family is mixed and that is not an accident of
packaging: the talker checkpoint stores 402 BF16 tensors and the speech
tokenizer stores 496 F32 ones. Both are carried through unchanged. Upcasting
the talker to F32 would produce a package for a model that does not exist --
which is exactly what an earlier oracle run did before
`docs/port-validation.md` grew its "Choosing the Oracle's dtype and Device"
section.

Three conversion rules here fail silently rather than loudly if they are
dropped. Each is asserted, not assumed:

1. The RVQ codebooks are not codebooks. The checkpoint stores EMA accumulators
   and the table has to be reconstructed as
   `embedding_sum / clip(cluster_usage, RVQ_EPS)[:, None]`. Skip it and the
   nearest-neighbour lookup returns noise with no error anywhere.
2. The two sides name that pair differently. The decoder uses `embedding_sum`
   and the encoder uses `embed_sum`, both alongside `cluster_usage`. A rule
   keyed on one silently passes over all sixteen codebooks of the other.
3. `initialized` is a shape-(1,) flag on encoder codebooks, not a weight.

Usage:

    uv run --project scripts/envs/qwen3-tts --locked python \
      scripts/convert-qwen3-tts.py \
      --manifest tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-customvoice.manifest.json \
      --weights-dir models/qwen3-tts-12hz-0-6b-customvoice \
      --output models/qwen3-tts-12hz-0-6b-customvoice/qwen3-tts-12hz-0-6b-customvoice-BF16.gguf
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.gguf_common import (  # noqa: E402
    add_general_identity,
    atomic_output_path,
    project_relative,
    sha256_file,
    write_json_atomic,
)

ARCH_KEY = "qwen3-tts"
FORMAT_VERSION = 1
PROFILE_NAME = "BF16"
PROFILE_VERSION = 1
ARCHITECTURE_VERSION = 1
REPORT_SCHEMA = "synthesize-converter-report-v1"

# The reconstruction epsilon upstream uses when dividing by cluster usage.
RVQ_EPS = 1e-5
# Codec frame geometry, asserted against the tokenizer config rather than assumed.
FRAME_RATE_HZ = 12.5
SAMPLES_PER_FRAME = 1920

CAPABILITY_STOCHASTIC = 1 << 1
INPUT_TEXT_UTF8 = 1 << 0


class ConverterError(RuntimeError):
    pass


@dataclass
class OutputTensor:
    name: str
    array: np.ndarray
    dtype: GGMLQuantizationType
    origin: str
    note: str = ""

    def report(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ggml_shape": list(self.array.shape[::-1]),
            "dtype": self.dtype.name,
            "elements": int(self.array.size),
            "bytes": int(self.array.nbytes),
            "sha256": hashlib.sha256(self.array.tobytes()).hexdigest(),
            "origin": self.origin,
            **({"note": self.note} if self.note else {}),
        }


@dataclass
class Conversion:
    outputs: list[OutputTensor] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    transformed: list[dict[str, str]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--weights-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def source_artifact(manifest: dict[str, Any], role: str, needle: str) -> dict[str, Any]:
    for artifact in manifest["source"]["artifacts"]:
        if artifact["role"] == role and needle in artifact["locator"]:
            return artifact
    raise ConverterError(f"manifest has no {role} artifact matching {needle!r}")


def numpy_of(tensor: torch.Tensor) -> tuple[np.ndarray, GGMLQuantizationType]:
    """Carry the stored dtype through unchanged.

    numpy has no bfloat16, so BF16 is viewed as raw uint16 pairs. GGUF stores
    the same bit pattern, and `GGMLQuantizationType.BF16` tells the reader how
    to interpret it.
    """
    if tensor.dtype is torch.bfloat16:
        raw = tensor.detach().contiguous().view(torch.uint16).cpu().numpy()
        return raw, GGMLQuantizationType.BF16
    if tensor.dtype is torch.float32:
        return tensor.detach().contiguous().cpu().numpy().astype(np.float32), GGMLQuantizationType.F32
    if tensor.dtype is torch.float16:
        return tensor.detach().contiguous().cpu().numpy().astype(np.float16), GGMLQuantizationType.F16
    raise ConverterError(f"unhandled source dtype {tensor.dtype}")


def reconstruct_codebooks(tensors: dict[str, torch.Tensor], conversion: Conversion) -> dict[str, torch.Tensor]:
    """Turn EMA accumulator pairs into the codebooks the graph actually needs.

    Both naming conventions are handled and both are *required* to appear. If a
    checkpoint ever stops using one of them this raises rather than quietly
    emitting half the tables, because that is the failure this function exists
    to prevent.
    """
    resolved: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    seen_conventions: set[str] = set()

    for name, tensor in tensors.items():
        for sum_field in ("embedding_sum", "embed_sum"):
            if not name.endswith("." + sum_field):
                continue
            usage_name = name[: -len(sum_field)] + "cluster_usage"
            if usage_name not in tensors:
                raise ConverterError(f"{name} has no matching {usage_name}")
            embedding_sum = tensors[name].to(torch.float32)
            cluster_usage = tensors[usage_name].to(torch.float32)
            if embedding_sum.ndim != 2 or cluster_usage.ndim != 1:
                raise ConverterError(
                    f"{name}: expected [entries, dim] and [entries], got "
                    f"{tuple(embedding_sum.shape)} and {tuple(cluster_usage.shape)}"
                )
            if embedding_sum.shape[0] != cluster_usage.shape[0]:
                raise ConverterError(
                    f"{name}: {embedding_sum.shape[0]} entries against "
                    f"{cluster_usage.shape[0]} usage counts"
                )
            codebook = embedding_sum / cluster_usage.clamp(min=RVQ_EPS).unsqueeze(1)
            target = name[: -len(sum_field)] + "codebook"
            resolved[target] = codebook
            consumed.update({name, usage_name})
            seen_conventions.add(sum_field)
            conversion.transformed.append({
                "output": target,
                "from": f"{name} / clamp({usage_name}, {RVQ_EPS})",
                "reason": "the checkpoint stores EMA accumulators, not the codebook",
            })

    if seen_conventions != {"embedding_sum", "embed_sum"}:
        raise ConverterError(
            "expected both RVQ naming conventions in this checkpoint, saw "
            f"{sorted(seen_conventions)}. A rule keyed on one name silently skips "
            "the other side's codebooks, which is the defect this check exists for."
        )

    remaining = {k: v for k, v in tensors.items() if k not in consumed}
    remaining.update(resolved)
    return remaining


def convert_file(path: Path, prefix: str, conversion: Conversion, reconstruct: bool) -> None:
    # Imported here rather than at module scope so the pure-python rules above
    # stay importable in environments without safetensors -- the registered
    # python unit suite runs under a different family's locked environment.
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}

    if reconstruct:
        tensors = reconstruct_codebooks(tensors, conversion)

    for name in sorted(tensors):
        tensor = tensors[name]
        if name.endswith(".initialized"):
            conversion.skipped.append({
                "logical_source_name": f"{prefix}{name}",
                "reason": "shape-(1,) codebook-initialised flag, not a weight",
            })
            continue
        array, dtype = numpy_of(tensor)
        conversion.outputs.append(OutputTensor(
            name=f"{prefix}{name}", array=array, dtype=dtype,
            origin=project_relative(path, Path.cwd()),
        ))


def add_metadata(writer: GGUFWriter, manifest: dict[str, Any], config: dict[str, Any],
                 tokenizer_config: dict[str, Any], codec_config: dict[str, Any],
                 vocab: dict[str, int], merges: list[str], digests: dict[str, str]) -> None:
    talker = config["talker_config"]
    predictor = talker["code_predictor_config"]

    add_general_identity(
        writer,
        name="Qwen3-TTS 12Hz 0.6B CustomVoice",
        basename=manifest["variant"],
        size_label="0.6B",
        languages=[t for t in manifest["package_contract"]["language_tags"] if t != "auto"],
        tags=["text-to-speech", "qwen3-tts", "synthesize.cpp"],
        author="Alibaba Qwen",
        organization="Qwen",
        source_url=manifest["source"]["repository"],
        description=(
            "Source-dtype synthesize.cpp conversion of the pinned Qwen3-TTS 12Hz "
            "0.6B CustomVoice checkpoint: BF16 talker, F32 speech tokenizer."
        ),
        license_id="apache-2.0",
        license_name="Apache License 2.0",
        license_link="https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    )
    writer.add_repo_url(manifest["source"]["repository"])

    writer.add_uint32("synthesize.format_version", FORMAT_VERSION)
    writer.add_string("synthesize.model_family", ARCH_KEY)
    writer.add_string("synthesize.model_variant", manifest["variant"])
    writer.add_string("synthesize.quantization.profile", PROFILE_NAME)
    writer.add_uint32("synthesize.quantization.profile_version", PROFILE_VERSION)
    writer.add_string("synthesize.source.repository", manifest["source"]["repository"])
    writer.add_string("synthesize.source.revision", manifest["source"]["revision"])
    writer.add_string("synthesize.source.checkpoint.sha256", digests["talker"])
    writer.add_string("synthesize.source.codec.sha256", digests["codec"])
    writer.add_string("synthesize.source.config.sha256", digests["config"])
    writer.add_string("synthesize.source.checkpoint.license_status", "apache-2.0")
    writer.add_string("synthesize.converter", "scripts/convert-qwen3-tts.py")

    writer.add_uint32("synthesize.qwen3-tts.architecture_version", ARCHITECTURE_VERSION)

    package = manifest["package_contract"]
    writer.add_uint32("synthesize.capabilities.input_flags", INPUT_TEXT_UTF8)
    writer.add_uint32("synthesize.capabilities.flags", CAPABILITY_STOCHASTIC)
    writer.add_uint64("synthesize.capabilities.max_input_tokens", int(package["max_input_tokens"]))
    writer.add_uint64("synthesize.capabilities.max_output_frames", int(package["max_output_frames"]))
    # This family exposes no speaking-rate control: upstream's entry point has no
    # speed parameter, so the range is pinned rather than merely defaulted.
    low, high = package["speaking_rate_range"]
    writer.add_float32("synthesize.capabilities.min_speaking_rate", float(low))
    writer.add_float32("synthesize.capabilities.max_speaking_rate", float(high))

    audio = package["native_audio"]
    writer.add_uint32("synthesize.audio.sample_rate_hz", int(audio["sample_rate_hz"]))
    writer.add_uint32("synthesize.audio.channels", int(audio["channels"]))
    writer.add_string("synthesize.audio.sample_format", audio["sample_format"])

    voices = package["voices"]
    writer.add_string("synthesize.voice.mode", voices["mode"])
    writer.add_bool("synthesize.voice.has_package_default", voices.get("default_id") is not None)
    writer.add_uint32("synthesize.voice.preset_count", len(voices["preset_ids"]))
    for index, voice_id in enumerate(voices["preset_ids"]):
        writer.add_string(f"synthesize.voice.{index}.id", voice_id)
        writer.add_uint32(f"synthesize.voice.{index}.flags", 0)

    # Talker geometry.
    for key, value in (
        ("layer_count", talker["num_hidden_layers"]),
        ("hidden_size", talker["hidden_size"]),
        ("attention_head_count", talker["num_attention_heads"]),
        ("key_value_head_count", talker["num_key_value_heads"]),
        ("head_dim", talker["head_dim"]),
        ("intermediate_size", talker["intermediate_size"]),
        ("codec_vocab_size", talker["vocab_size"]),
        ("text_vocab_size", talker["text_vocab_size"]),
        ("text_hidden_size", talker["text_hidden_size"]),
        ("code_group_count", talker["num_code_groups"]),
    ):
        writer.add_uint32(f"synthesize.qwen3-tts.talker.{key}", int(value))
    writer.add_float32("synthesize.qwen3-tts.talker.rms_norm_eps", float(talker["rms_norm_eps"]))
    writer.add_float32("synthesize.qwen3-tts.talker.rope_theta", float(talker["rope_theta"]))
    # The declared mrope collapses exactly to 1-D rope for this model: every
    # position_ids path yields three identical rows. Recorded so the runtime does
    # not reimplement sectioning that the reference never exercises.
    writer.add_string("synthesize.qwen3-tts.talker.rope_type", "1d")
    writer.add_string(
        "synthesize.qwen3-tts.talker.rope_note",
        "config declares mrope_section with interleaved=true; all three position rows "
        "are always identical, so it is exactly plain 1-D rope",
    )

    for key, value in (
        ("layer_count", predictor["num_hidden_layers"]),
        ("hidden_size", predictor["hidden_size"]),
        ("attention_head_count", predictor["num_attention_heads"]),
        ("key_value_head_count", predictor["num_key_value_heads"]),
        ("head_dim", predictor["head_dim"]),
        ("vocab_size", predictor["vocab_size"]),
        ("code_group_count", predictor["num_code_groups"]),
    ):
        writer.add_uint32(f"synthesize.qwen3-tts.code_predictor.{key}", int(value))

    # Codec geometry, checked rather than copied.
    hop = int(codec_config["decode_upsample_rate"])
    if hop != SAMPLES_PER_FRAME:
        raise ConverterError(f"codec hop is {hop}, expected {SAMPLES_PER_FRAME}")
    sample_rate = int(codec_config["input_sample_rate"])
    if abs(sample_rate / hop - FRAME_RATE_HZ) > 1e-9:
        raise ConverterError(f"{sample_rate} Hz over hop {hop} is not {FRAME_RATE_HZ} Hz")
    writer.add_uint32("synthesize.qwen3-tts.codec.sample_rate", sample_rate)
    writer.add_uint32("synthesize.qwen3-tts.codec.hop_length", hop)
    writer.add_float32("synthesize.qwen3-tts.codec.frame_rate_hz", FRAME_RATE_HZ)

    # Special token ids the graph needs.
    for key in ("tts_bos_token_id", "tts_eos_token_id", "tts_pad_token_id",
                "im_start_token_id", "im_end_token_id", "assistant_token_id"):
        writer.add_uint32(f"synthesize.qwen3-tts.token.{key}", int(config[key]))
    for key in ("codec_bos_id", "codec_eos_token_id", "codec_pad_id"):
        writer.add_uint32(f"synthesize.qwen3-tts.token.{key}", int(talker[key]))

    # Preset Voice Catalog: speakers are codec-vocabulary token ids, not
    # embeddings, which is why this variant carries no speaker encoder.
    speakers = sorted(talker["spk_id"], key=lambda n: talker["spk_id"][n])
    writer.add_array("synthesize.qwen3-tts.speakers.names", speakers)
    writer.add_array("synthesize.qwen3-tts.speakers.token_ids", [int(talker["spk_id"][n]) for n in speakers])
    writer.add_array(
        "synthesize.qwen3-tts.speakers.dialect_override",
        [str(talker["spk_is_dialect"][n]) if talker["spk_is_dialect"][n] else "" for n in speakers],
    )
    languages = sorted(talker["codec_language_id"], key=lambda n: talker["codec_language_id"][n])
    writer.add_array("synthesize.qwen3-tts.languages.names", languages)
    writer.add_array("synthesize.qwen3-tts.languages.token_ids",
                     [int(talker["codec_language_id"][n]) for n in languages])

    # Text Frontend payload: vocabulary and merges travel in the package so the
    # provider needs no executable per-model mapping logic.
    ordered = sorted(vocab, key=lambda token: vocab[token])
    writer.add_array("synthesize.qwen3-tts.frontend.vocab", ordered)
    writer.add_array("synthesize.qwen3-tts.frontend.merges", merges)
    writer.add_bool("synthesize.frontend.present", True)
    writer.add_string("synthesize.frontend.provider", "synthesize.qwen_bpe")
    writer.add_uint32("synthesize.frontend.contract_version", 1)
    chat_template = tokenizer_config.get("chat_template")
    if isinstance(chat_template, str):
        writer.add_string("synthesize.qwen3-tts.frontend.chat_template", chat_template)


def verify_gguf(path: Path, outputs: list[OutputTensor]) -> None:
    reader = GGUFReader(str(path))
    found = {t.name: t for t in reader.tensors}
    if len(found) != len(outputs):
        raise ConverterError(f"wrote {len(outputs)} tensors but read back {len(found)}")
    for output in outputs:
        tensor = found.get(output.name)
        if tensor is None:
            raise ConverterError(f"{output.name} missing from the written file")
        if GGMLQuantizationType(tensor.tensor_type) is not output.dtype:
            raise ConverterError(
                f"{output.name}: wrote {output.dtype.name}, read "
                f"{GGMLQuantizationType(tensor.tensor_type).name}"
            )
        if int(np.prod(tensor.shape)) != output.array.size:
            raise ConverterError(f"{output.name}: element count changed on write")


def main() -> int:
    args = parse_args()
    project_root = Path.cwd()
    manifest = load_json(args.manifest)
    if manifest.get("family") != ARCH_KEY:
        raise ConverterError(f"{args.manifest}: not a {ARCH_KEY} manifest")

    weights = args.weights_dir
    talker_path = weights / "model.safetensors"
    codec_path = weights / "speech_tokenizer" / "model.safetensors"
    config = load_json(weights / "config.json")
    codec_config = load_json(weights / "speech_tokenizer" / "config.json")
    tokenizer_config = load_json(weights / "tokenizer_config.json")
    vocab = load_json(weights / "vocab.json")
    merges = [
        line for line in (weights / "merges.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#version")
    ]

    digests = {
        "talker": sha256_file(talker_path),
        "codec": sha256_file(codec_path),
        "config": sha256_file(weights / "config.json"),
    }
    expected = source_artifact(manifest, "checkpoint", "model.safetensors")["sha256"]
    if digests["talker"] != expected and digests["codec"] != expected:
        raise ConverterError("neither checkpoint matches the manifest's pinned digest")

    conversion = Conversion()
    convert_file(talker_path, "", conversion, reconstruct=False)
    convert_file(codec_path, "codec.", conversion, reconstruct=True)

    names = [o.name for o in conversion.outputs]
    if len(set(names)) != len(names):
        duplicates = sorted({n for n in names if names.count(n) > 1})
        raise ConverterError(f"tensor names collide after prefixing: {duplicates}")

    with atomic_output_path(args.output) as staging:
        writer = GGUFWriter(str(staging), ARCH_KEY)
        add_metadata(writer, manifest, config, tokenizer_config, codec_config,
                     vocab, merges, digests)
        for output in conversion.outputs:
            writer.add_tensor(output.name, output.array, raw_dtype=output.dtype)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    verify_gguf(args.output, conversion.outputs)

    by_dtype: dict[str, int] = {}
    for output in conversion.outputs:
        by_dtype[output.dtype.name] = by_dtype.get(output.dtype.name, 0) + 1

    report = {
        "schema": REPORT_SCHEMA,
        "family": ARCH_KEY,
        "variant": manifest["variant"],
        "profile": PROFILE_NAME,
        "profile_version": PROFILE_VERSION,
        "converter": {
            "path": "scripts/convert-qwen3-tts.py",
            "environment_lock": "scripts/envs/qwen3-tts/uv.lock",
            "environment_lock_sha256": sha256_file(Path("scripts/envs/qwen3-tts/uv.lock")),
        },
        "source": {
            "repository": manifest["source"]["repository"],
            "revision": manifest["source"]["revision"],
            "talker_path": project_relative(talker_path, project_root),
            "talker_sha256": digests["talker"],
            "codec_path": project_relative(codec_path, project_root),
            "codec_sha256": digests["codec"],
            "config_sha256": digests["config"],
        },
        "output": {
            "path": project_relative(args.output, project_root),
            "bytes": args.output.stat().st_size,
            "sha256": sha256_file(args.output),
            "emitted_tensor_count": len(conversion.outputs),
            "tensor_count_by_dtype": by_dtype,
        },
        "dtype_policy": (
            "the checkpoint's own dtypes are carried through unchanged: BF16 talker, "
            "F32 speech tokenizer. Reconstructed RVQ codebooks are F32 because they are "
            "computed here rather than stored."
        ),
        "transformed": conversion.transformed,
        "skipped": conversion.skipped,
        "tensors": [o.report() for o in conversion.outputs],
    }
    report_path = args.report or Path(f"reports/convert/{ARCH_KEY}/{manifest['variant']}-{PROFILE_NAME}.json")
    write_json_atomic(report_path, report)

    print(f"wrote {args.output} ({report['output']['bytes']:,} bytes)")
    print(f"  tensors: {len(conversion.outputs)} {by_dtype}")
    print(f"  reconstructed codebooks: {len(conversion.transformed)}")
    print(f"  skipped: {len(conversion.skipped)}")
    print(f"  report: {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConverterError as error:
        print(f"convert-qwen3-tts: {error}", file=sys.stderr)
        raise SystemExit(1)
