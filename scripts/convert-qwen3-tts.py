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

For CustomVoice, the speech tokenizer's *encoder* half is deliberately not
carried. Synthesis runs one way -- codes to audio -- so nothing in this
package can reach it, and this checkpoint could not use it anyway:
`model.safetensors` is 402 tensors, all `talker.`, with no speaker encoder at
all, and the reference builds voice-clone prompts from the Base variant
instead. Dropping it removes 161 tensors and 225 MB that no graph reads.

For Base, the encoder half is carried in full: it is what turns a reference
clip into codes for voice cloning. 16 of its 32 codebooks -- exactly the ones
with a same-index decoder counterpart -- are measured bit-identical to the
decoder's at this revision by `measure_shared_codebooks` below, which compares
every `.codebook` tensor on both sides at every conversion. The four
input/output projections on those same two quantizer stages were found
identical too, by hand at Stage 1 and again during Stage 2's review; nothing
in this file re-measures them, and that is deliberate, because nothing acts on
either measurement.

They are all carried, in full, rather than stored once and aliased under the
decoder's name: an alias with no in-package record of where it points is a
name a consumer cannot resolve, and this project chose the ~35 MB of duplicate
bytes (~1.4% of the 2.52 GB package) over that.

Two conversion rules here fail silently rather than loudly if they are dropped.
Each is asserted, not assumed:

1. The RVQ codebooks are not codebooks. The checkpoint stores EMA accumulators
   and the table has to be reconstructed as
   `embedding_sum / clip(cluster_usage, RVQ_EPS)[:, None]`. Skip it and the
   nearest-neighbour lookup returns noise with no error anywhere. The two halves
   of the tokenizer name that pair differently -- the decoder uses
   `embedding_sum`, the encoder `embed_sum` -- so the rule matches either and
   then checks that no accumulator survived it, which holds whichever halves a
   future package carries.
2. `initialized` is a shape-(1,) flag on a codebook, not a weight.

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
from typing import Any, Sequence

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

# GGML stores a tensor name in a fixed 64-byte field and truncates silently past
# it. A truncated name is not findable by the name the catalog asks for, and two
# names that differ only past the cut become the same tensor. This table has two
# independent reasons to grow: CustomVoice's conversion (talker + codec decoder
# only) overflowed on five paths; Base's conversion additionally carries the
# codec's encoder half, whose own paths are longer for reasons specific to that
# half (`encoder_transformer` is four characters longer than the decoder's
# `pre_transformer`, and its residual-vector-quantizer names spell out
# "residual_vector" where the decoder already abbreviates to
# `rvq_first`/`rvq_rest`). The components that made them long are shortened --
# the shortest edit that fixes it, not a renaming scheme.
GGML_MAX_NAME = 64
NAME_SHORTENINGS = (
    (".post_attention_layernorm.", ".post_attn_norm."),
    # Renamed for symmetry with its sibling below rather than for length; one
    # long and one short would read as a mistake.
    (".self_attn_layer_scale.", ".self_attn_scale."),
    (".mlp_layer_scale.", ".mlp_scale."),
    # `_codebook` is the module that holds the table, and the table is the only
    # thing left in it after reconstruction.
    ("._codebook.codebook", ".codebook"),
    # The encoder's own transformer block, distinct from the decoder's
    # `pre_transformer` -- shortened on its own terms, not aliased to a
    # different module's name.
    (".encoder_transformer.", ".enc_transformer."),
    # The decoder already spells its residual vector quantizers `rvq_first` /
    # `rvq_rest`; the encoder's two quantizer stages get the same abbreviation
    # rather than a second, inconsistent one.
    (".quantizer.acoustic_residual_vector_quantizer.", ".quantizer.acoustic_rvq."),
    (".quantizer.semantic_residual_vector_quantizer.", ".quantizer.semantic_rvq."),
    # Mirrors `._codebook.codebook` above for the encoder's un-underscored
    # module name: after reconstruction, the table is the only thing left.
    (".codebook.codebook", ".codebook"),
)


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
    renamed: list[dict[str, str]] = field(default_factory=list)
    # Not tensors that were removed -- both halves are always carried in full.
    # This records which encoder codebook is bit-identical to which decoder
    # one, measured at this revision, so that fact is on the record even
    # though nothing acts on it.
    measured_shared_codebooks: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class VariantProfile:
    model_type: str
    carries_speaker_encoder: bool
    carries_codec_encoder: bool
    display_name: str
    size_label: str


def shorten_name(name: str, conversion: Conversion) -> str:
    """Bring a name under GGML's fixed-width limit, or refuse to emit it."""
    shortened = name
    for long_form, short_form in NAME_SHORTENINGS:
        shortened = shortened.replace(long_form, short_form)
    if shortened != name:
        conversion.renamed.append({"output": shortened, "from": name})
    if len(shortened) >= GGML_MAX_NAME:
        raise ConverterError(
            f"tensor name {shortened!r} is {len(shortened)} characters; GGML truncates at "
            f"{GGML_MAX_NAME} and the catalog would never find it"
        )
    return shortened


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--weights-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def variant_profile(config: dict[str, Any]) -> VariantProfile:
    """Decide what this checkpoint carries from what it declares.

    The three supported variants differ by a whole subsystem: Base ships a
    76-tensor ECAPA-TDNN speaker encoder and needs the tokenizer's encoder half
    to turn reference audio into codes; CustomVoice and VoiceDesign ship neither
    and resolve speakers as codec-vocabulary token ids. Keying that on the
    declared type and then checking the declaration against the config is what
    keeps a future variant from silently converting as whichever branch it fell
    into.
    """
    model_type = str(config.get("tts_model_type", ""))
    has_encoder_config = "speaker_encoder_config" in config
    if model_type == "base":
        if not has_encoder_config:
            raise ConverterError(
                "config declares tts_model_type=base but carries no speaker_encoder_config; "
                "a Base package without a speaker encoder cannot prepare a Voice Profile"
            )
        return VariantProfile("base", True, True, "Qwen3-TTS 12Hz 0.6B Base", "0.6B")
    if model_type == "custom_voice":
        if has_encoder_config:
            raise ConverterError(
                "config declares tts_model_type=custom_voice but carries a speaker_encoder_config"
            )
        return VariantProfile("custom_voice", False, False, "Qwen3-TTS 12Hz 0.6B CustomVoice", "0.6B")
    if model_type == "voice_design":
        # No encoder of either kind: this checkpoint has no speaker_encoder_config
        # and there is no reference audio on its path, so the tensor set is
        # CustomVoice's shape rather than Base's. See the Stage 3 design, section 4.
        if has_encoder_config:
            raise ConverterError(
                "config declares tts_model_type=voice_design but carries a speaker_encoder_config"
            )
        return VariantProfile("voice_design", False, False,
                              "Qwen3-TTS 12Hz 1.7B VoiceDesign", "1.7B")
    raise ConverterError(f"unsupported tts_model_type {model_type!r}")


def profile_source_names(profile: VariantProfile) -> list[str]:
    """Which Voice Profile sources this variant implements.

    Named rather than derived at the read side, because after Stage 3 the Voice
    Mode no longer determines this: `profile-sources` covers both Base, which
    clones from a recording, and VoiceDesign, which cannot clone at all. The
    runtime refuses a package whose declared sources do not match the blocks it
    carries, so this is a claim the package has to earn.
    """
    if profile.carries_speaker_encoder:
        return ["reference-audio"]
    if profile.model_type == "voice_design":
        return ["description-text"]
    return []


def profile_schema_name(profile: VariantProfile) -> str | None:
    """The Profile Schema this variant serializes under, or None if it prepares nothing."""
    if profile.carries_speaker_encoder:
        return "qwen3-tts-voice-clone"
    if profile.model_type == "voice_design":
        return "qwen3-tts-voice-design"
    return None


# modeling_qwen3_tts.py:1941 -- the speaker encoder consumes a 128-bin mel, not
# a waveform. These five numbers are the front end's whole contract and they are
# carried in the package rather than written into the C++ port.
SPEAKER_MEL = {"mel_bins": 128, "n_fft": 1024, "hop_length": 256,
               "win_length": 1024, "fmin": 0.0, "fmax": 12000.0}


def speaker_encoder_metadata(config: dict[str, Any]) -> dict[str, Any]:
    enc_dim = int(config["enc_dim"])
    sample_rate = int(config["sample_rate"])
    if sample_rate != 24000:
        raise ConverterError(f"speaker encoder declares {sample_rate} Hz; the mel front end is pinned to 24000")
    return {"enc_dim": enc_dim, "sample_rate": sample_rate, **SPEAKER_MEL}


def speaker_catalog(talker: dict[str, Any], preset_ids: list[str]) -> tuple[list[str], list[int], list[str]]:
    """Cross-check the manifest's Catalog against the checkpoint's speaker table.

    Stage 1's hazard was ordering; this variant's is emptiness. Both directions
    are checked, so a Base package cannot inherit a CustomVoice catalog and a
    CustomVoice package cannot lose one.
    """
    if set(preset_ids) != set(talker["spk_id"]):
        raise ConverterError(
            f"the manifest lists {sorted(preset_ids)} but the checkpoint carries "
            f"{sorted(talker['spk_id'])}"
        )
    names = list(preset_ids)
    return (names,
            [int(talker["spk_id"][n]) for n in names],
            [str(talker["spk_is_dialect"][n]) if talker["spk_is_dialect"][n] else "" for n in names])


def source_artifact(manifest: dict[str, Any], role: str, needle: str) -> dict[str, Any]:
    for artifact in manifest["source"]["artifacts"]:
        if artifact["role"] == role and needle in artifact["locator"]:
            return artifact
    raise ConverterError(f"manifest has no {role} artifact matching {needle!r}")


def talker_checkpoint_locator(manifest: dict[str, Any]) -> str:
    """The manifest's talker checkpoint artifact, matched unambiguously.

    Both variants' manifests carry two "checkpoint"-role artifacts -- the
    talker's `model.safetensors` and the codec's
    `speech_tokenizer/model.safetensors` -- and both locators contain the
    substring "model.safetensors", so a plain substring search (as
    `source_artifact` above does, for the unrelated pinned-digest check in
    `main()`) finds the right one only because it happens to be listed first.
    This matches on the talker's own path shape -- it has no
    `speech_tokenizer/` segment -- so the result does not depend on manifest
    ordering.
    """
    matches = [
        artifact["locator"] for artifact in manifest["source"]["artifacts"]
        if artifact["role"] == "checkpoint"
        and artifact["locator"].endswith("/model.safetensors")
        and "/speech_tokenizer/" not in artifact["locator"]
    ]
    if len(matches) != 1:
        raise ConverterError(
            f"expected exactly one talker checkpoint artifact, found {len(matches)}"
        )
    return matches[0]


def license_link_for(manifest: dict[str, Any]) -> str:
    """The Hugging Face model page to link as this package's license source.

    Derived from the pinned talker checkpoint's own locator, not from
    `manifest["source"]["repository"]`: CustomVoice's `repository` field is a
    GitHub URL (where the code lives), not the Hugging Face model page its
    checkpoint is actually pinned under. Every variant's package must link
    its own page, not whichever variant's happened to be converted first.
    """
    return talker_checkpoint_locator(manifest).split("/resolve/", 1)[0]


def compatibility_id(schema: str, version: int, digests: Sequence[str]) -> str:
    """Profile Compatibility ID: sha256 over the family compatibility manifest.

    Schema identity plus the fingerprints of every weight and config that
    changes what prepared conditioning means. `schema` and `version` are
    hashed together as one `"{schema}/{version}"` piece, then each digest is
    hashed as its own separate piece (not joined into a single string first),
    mirroring `scripts/convert-omnivoice.py`'s construction exactly so both
    families compute a Compatibility ID the same way. The digest order is
    part of the contract: reordering, rejoining, or hashing a different
    representation changes every already-shipped package's id silently
    unless a test like `CompatibilityIdTests` pins the exact output.
    """
    compat = hashlib.sha256()
    compat.update(f"{schema}/{version}".encode())
    for digest in digests:
        compat.update(digest.encode())
    return compat.hexdigest()


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

    Both naming conventions are matched, and afterwards no accumulator may be
    left over. Checking the leftovers rather than checking that both spellings
    appeared is what keeps this honest for a package that carries only one half
    of the tokenizer: a rule keyed on one spelling would pass over the other
    half's tables in silence, and this catches that whichever halves are
    present.
    """
    resolved: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()

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
            conversion.transformed.append({
                "output": target,
                "from": f"{name} / clamp({usage_name}, {RVQ_EPS})",
                "reason": "the checkpoint stores EMA accumulators, not the codebook",
            })

    if not resolved:
        raise ConverterError("no RVQ codebook was reconstructed; the accumulator names moved")

    leftover = sorted(
        name for name in tensors
        if name not in consumed
        and name.rsplit(".", 1)[-1] in {"embedding_sum", "embed_sum", "cluster_usage"}
    )
    if leftover:
        raise ConverterError(
            f"{len(leftover)} EMA accumulator(s) were not reconstructed, starting with "
            f"{leftover[0]}. Emitting them as weights would give the graph noise instead "
            "of a codebook, with no error anywhere."
        )

    remaining = {k: v for k, v in tensors.items() if k not in consumed}
    remaining.update(resolved)
    return remaining


def measure_shared_codebooks(tensors: dict[str, torch.Tensor], conversion: Conversion) -> None:
    """Record which encoder codebooks are bit-identical to a decoder one.

    Measured, not assumed: equality is tested at this revision rather than
    carried over from Stage 1's measurement. Nothing is removed here -- an
    alias with no in-package record of where it points is a name a consumer
    cannot resolve, so both halves are always carried in full regardless of
    what this finds. The measurement is kept anyway because it is real and
    worth having on the record.
    """
    decoder_tables = {
        name: tensor for name, tensor in tensors.items()
        if name.startswith("decoder.") and name.endswith(".codebook")
    }
    for name, tensor in tensors.items():
        if not (name.startswith("encoder.") and name.endswith(".codebook")):
            continue
        twin = next(
            (d for d, table in decoder_tables.items()
             if table.shape == tensor.shape and torch.equal(table, tensor)),
            None,
        )
        if twin is not None:
            conversion.measured_shared_codebooks.append({
                "encoder_tensor": name,
                "decoder_tensor": twin,
                "note": "bit-identical at this revision; both are carried in the package",
            })


def convert_file(path: Path, prefix: str, conversion: Conversion, reconstruct: bool,
                 drop_prefix: str | None = None, measure_duplicates: bool = False) -> None:
    # Imported here rather than at module scope so the pure-python rules above
    # stay importable in environments without safetensors -- the registered
    # python unit suite runs under a different family's locked environment.
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}

    if drop_prefix is not None:
        dropped = sorted(key for key in tensors if key.startswith(drop_prefix))
        if not dropped:
            raise ConverterError(
                f"nothing under {drop_prefix!r} to drop; the checkpoint's layout moved and "
                "this package would silently start carrying a half it never carried before"
            )
        for key in dropped:
            del tensors[key]
        conversion.skipped.append({
            "logical_source_name": f"{prefix}{drop_prefix}*",
            "count": len(dropped),
            "reason": (
                "CustomVoice carries no speaker encoder to clone with, so nothing in this "
                "package can reach the tokenizer's encoder half; synthesis only runs "
                "codes-to-audio anyway. (Base carries this half in full instead of dropping "
                "it -- see convert_file's caller.)"
            ),
        })

    if reconstruct:
        tensors = reconstruct_codebooks(tensors, conversion)

    if measure_duplicates:
        measure_shared_codebooks(tensors, conversion)

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
            name=shorten_name(f"{prefix}{name}", conversion), array=array, dtype=dtype,
            origin=project_relative(path, Path.cwd()),
        ))


def add_metadata(writer: GGUFWriter, manifest: dict[str, Any], config: dict[str, Any],
                 tokenizer_config: dict[str, Any], codec_config: dict[str, Any],
                 generation_config: dict[str, Any],
                 vocab: dict[str, int], merges: list[str], digests: dict[str, str],
                 profile: VariantProfile) -> None:
    talker = config["talker_config"]
    predictor = talker["code_predictor_config"]

    add_general_identity(
        writer,
        name=profile.display_name,
        basename=manifest["variant"],
        size_label=profile.size_label,
        languages=[t for t in manifest["package_contract"]["language_tags"] if t != "auto"],
        tags=["text-to-speech", "qwen3-tts", "synthesize.cpp"],
        author="Alibaba Qwen",
        organization="Qwen",
        source_url=manifest["source"]["repository"],
        description=(
            f"Source-dtype synthesize.cpp conversion of the pinned {profile.display_name} "
            "checkpoint: BF16 talker, F32 speech tokenizer."
        ),
        license_id="apache-2.0",
        license_name="Apache License 2.0",
        license_link=license_link_for(manifest),
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
    writer.add_string("synthesize.source.generation_config.sha256", digests["generation_config"])
    writer.add_string("synthesize.source.checkpoint.license_status", "apache-2.0")
    writer.add_string("synthesize.converter", "scripts/convert-qwen3-tts.py")

    writer.add_uint32("synthesize.qwen3-tts.architecture_version", ARCHITECTURE_VERSION)

    # The checkpoint's shipped decoding defaults, carried rather than hardcoded.
    #
    # These decide what the model says, not merely how it sounds: the talker ends
    # an utterance by sampling the codec end token, so the filter chain in front
    # of that draw is part of the model's contract. Reimplementing it from memory
    # is how repetition_penalty went missing, which truncated long inputs
    # mid-sentence for a month without any test noticing.
    #
    # The talker and the code predictor are configured separately upstream and
    # are carried separately here. Only the talker has a repetition penalty.
    def sampling_float(key: str, minimum: float) -> float:
        value = float(generation_config[key])
        if not value > minimum:
            raise ConverterError(f"generation_config.{key} is {value}, which cannot be sampled with")
        return value

    if not bool(generation_config["do_sample"]):
        raise ConverterError("generation_config declares do_sample false; this family samples")

    writer.add_float32("synthesize.qwen3-tts.sampling.temperature", sampling_float("temperature", 0.0))
    writer.add_uint32("synthesize.qwen3-tts.sampling.top_k", int(generation_config["top_k"]))
    writer.add_float32("synthesize.qwen3-tts.sampling.top_p", sampling_float("top_p", 0.0))
    writer.add_float32("synthesize.qwen3-tts.sampling.repetition_penalty",
                       sampling_float("repetition_penalty", 0.0))
    writer.add_float32("synthesize.qwen3-tts.sampling.predictor.temperature",
                       sampling_float("subtalker_temperature", 0.0))
    writer.add_uint32("synthesize.qwen3-tts.sampling.predictor.top_k",
                      int(generation_config["subtalker_top_k"]))
    writer.add_float32("synthesize.qwen3-tts.sampling.predictor.top_p",
                       sampling_float("subtalker_top_p", 0.0))

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

    # The decoder's own geometry. Every codec tensor's shape is derived from
    # these at load, so a package whose metadata and tensors disagree is refused
    # instead of building a graph around whichever one the reader trusted.
    decoder = codec_config["decoder_config"]
    rates = [int(rate) for rate in decoder["upsample_rates"]]
    ratios = [int(ratio) for ratio in decoder["upsampling_ratios"]]
    total = 1
    for factor in rates + ratios:
        if factor <= 0:
            raise ConverterError(f"codec upsample factor {factor} is not positive")
        total *= factor
    if total != hop:
        raise ConverterError(
            f"codec upsample factors {rates} x {ratios} multiply to {total}, not the hop {hop}"
        )
    # The residual stages halve the channel width each time, so the last one
    # must land on a whole number of channels.
    if decoder["decoder_dim"] % (2 ** len(rates)) != 0:
        raise ConverterError(
            f"decoder_dim {decoder['decoder_dim']} does not halve {len(rates)} times"
        )
    if decoder["codebook_dim"] % 2 != 0:
        raise ConverterError(f"codebook_dim {decoder['codebook_dim']} is not even")
    if int(decoder["num_semantic_quantizers"]) >= int(decoder["num_quantizers"]):
        raise ConverterError("the semantic quantizer count must leave acoustic groups behind it")
    if int(decoder["num_quantizers"]) != int(talker["num_code_groups"]):
        raise ConverterError(
            f"the codec takes {decoder['num_quantizers']} code groups but the talker emits "
            f"{talker['num_code_groups']}"
        )

    for key, value in (
        ("latent_dim", decoder["latent_dim"]),
        ("dim", decoder["decoder_dim"]),
        # The quantizer works at half the codebook dimension; the projections on
        # either side of it are what change width.
        ("codebook_dim", decoder["codebook_dim"]),
        ("codebook_size", decoder["codebook_size"]),
        ("quantizer_count", decoder["num_quantizers"]),
        ("semantic_quantizer_count", decoder["num_semantic_quantizers"]),
        ("hidden_size", decoder["hidden_size"]),
        ("intermediate_size", decoder["intermediate_size"]),
        ("layer_count", decoder["num_hidden_layers"]),
        ("attention_head_count", decoder["num_attention_heads"]),
        ("key_value_head_count", decoder["num_key_value_heads"]),
        ("head_dim", decoder["head_dim"]),
        ("sliding_window", decoder["sliding_window"]),
    ):
        writer.add_uint32(f"synthesize.qwen3-tts.codec.decoder.{key}", int(value))
    writer.add_float32("synthesize.qwen3-tts.codec.decoder.rms_norm_eps", float(decoder["rms_norm_eps"]))
    writer.add_float32("synthesize.qwen3-tts.codec.decoder.rope_theta", float(decoder["rope_theta"]))
    writer.add_array("synthesize.qwen3-tts.codec.decoder.upsample_rates", rates)
    writer.add_array("synthesize.qwen3-tts.codec.decoder.upsampling_ratios", ratios)

    # Special token ids the graph needs.
    for key in ("tts_bos_token_id", "tts_eos_token_id", "tts_pad_token_id",
                "im_start_token_id", "im_end_token_id", "assistant_token_id"):
        writer.add_uint32(f"synthesize.qwen3-tts.token.{key}", int(config[key]))
    # The think/nothink pair opens the codec side of the prompt: a request that
    # names a language emits think, its language token, and think_eos; one that
    # asks for auto emits nothink instead and no language token at all.
    for key in ("codec_bos_id", "codec_eos_token_id", "codec_pad_id", "codec_think_id",
                "codec_nothink_id", "codec_think_bos_id", "codec_think_eos_id"):
        writer.add_uint32(f"synthesize.qwen3-tts.token.{key}", int(talker[key]))

    # Preset Voice Catalog: speakers are codec-vocabulary token ids, not
    # embeddings, which is why only a variant with no speaker encoder carries
    # one at all.
    #
    # The order is the Voice catalog's, not the checkpoint's token order. The two
    # arrays describe one catalog and the loader reads them index by index, so
    # ordering them differently -- alphabetically here, by token id there -- makes
    # every entry name one speaker and select another.
    #
    # A Base package's checkpoint and manifest both carry an empty catalog, so
    # `speaker_catalog` returns three empty lists and nothing is emitted below:
    # an empty `synthesize.qwen3-tts.speakers.*` array would be a Catalog that
    # exists but selects nothing, not the absence this variant means to declare.
    speaker_names, speaker_token_ids, speaker_dialects = speaker_catalog(talker, list(voices["preset_ids"]))
    if speaker_names:
        writer.add_array("synthesize.qwen3-tts.speakers.names", speaker_names)
        writer.add_array("synthesize.qwen3-tts.speakers.token_ids", speaker_token_ids)
        writer.add_array("synthesize.qwen3-tts.speakers.dialect_override", speaker_dialects)

    sources = profile_source_names(profile)
    schema = profile_schema_name(profile)
    if sources:
        writer.add_array("synthesize.voice.profile_sources", sources)
        writer.add_string("synthesize.profile.schema", schema)
        writer.add_uint32("synthesize.profile.schema_version", 1)
        writer.add_string(
            "synthesize.profile.compatibility_id",
            compatibility_id(schema, 1, (digests["talker"], digests["codec"], digests["config"])),
        )
    # The reference limits and the encoder's own front-end contract describe
    # REFERENCE AUDIO. A package that cannot take a recording has nothing to
    # say here, and writing zeros would be a claim rather than a silence.
    if profile.carries_speaker_encoder:
        reference = package["profile"]["reference"]
        for key in ("target_sample_rate", "target_channels"):
            writer.add_uint32(f"synthesize.reference.{key}", int(reference[key]))
        for key in ("min_frames_per_clip", "max_frames_per_clip",
                    "max_total_frames", "max_reference_count"):
            writer.add_uint64(f"synthesize.reference.{key}", int(reference[key]))
        for key, value in speaker_encoder_metadata(config["speaker_encoder_config"]).items():
            if isinstance(value, float):
                writer.add_float32(f"synthesize.qwen3-tts.speaker_encoder.{key}", value)
            else:
                writer.add_uint32(f"synthesize.qwen3-tts.speaker_encoder.{key}", int(value))

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
    generation_config = load_json(weights / "generation_config.json")
    vocab = load_json(weights / "vocab.json")
    merges = [
        line for line in (weights / "merges.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#version")
    ]

    digests = {
        "talker": sha256_file(talker_path),
        "codec": sha256_file(codec_path),
        "config": sha256_file(weights / "config.json"),
        "generation_config": sha256_file(weights / "generation_config.json"),
    }
    expected = source_artifact(manifest, "checkpoint", "model.safetensors")["sha256"]
    if digests["talker"] != expected and digests["codec"] != expected:
        raise ConverterError("neither checkpoint matches the manifest's pinned digest")

    profile = variant_profile(config)
    conversion = Conversion()
    convert_file(talker_path, "", conversion, reconstruct=False)
    convert_file(codec_path, "codec.", conversion, reconstruct=True,
                 drop_prefix=None if profile.carries_codec_encoder else "encoder.",
                 measure_duplicates=profile.carries_codec_encoder)

    names = [o.name for o in conversion.outputs]
    if len(set(names)) != len(names):
        duplicates = sorted({n for n in names if names.count(n) > 1})
        raise ConverterError(f"tensor names collide after prefixing: {duplicates}")

    with atomic_output_path(args.output) as staging:
        writer = GGUFWriter(str(staging), ARCH_KEY)
        add_metadata(writer, manifest, config, tokenizer_config, codec_config,
                     generation_config, vocab, merges, digests, profile)
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
        "renamed": conversion.renamed,
        "skipped": conversion.skipped,
        "measured_shared_codebooks": conversion.measured_shared_codebooks,
        "tensors": [o.report() for o in conversion.outputs],
    }
    report_path = args.report or Path(f"reports/convert/{ARCH_KEY}/{manifest['variant']}-{PROFILE_NAME}.json")
    write_json_atomic(report_path, report)

    print(f"wrote {args.output} ({report['output']['bytes']:,} bytes)")
    print(f"  tensors: {len(conversion.outputs)} {by_dtype}")
    print(f"  reconstructed codebooks: {len(conversion.transformed)}")
    print(f"  skipped: {len(conversion.skipped)}")
    print(f"  measured shared codebooks (both carried): {len(conversion.measured_shared_codebooks)}")
    print(f"  report: {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConverterError as error:
        print(f"convert-qwen3-tts: {error}", file=sys.stderr)
        raise SystemExit(1)
