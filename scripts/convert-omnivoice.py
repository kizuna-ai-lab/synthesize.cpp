#!/usr/bin/env python3
"""Convert the pinned OmniVoice checkpoint to a source-dtype GGUF.

Stage 3 of the porting pipeline. This emits the source/reference-dtype artifact
only; F16 and every Quantization Profile are produced later by the C++
quantizer from this file, so no quantization policy belongs here.

The reference dtype is F32 throughout and that is not a choice this script
makes: the generator stores 312 F32 tensors plus one I64 index buffer, and the
Higgs Audio V2 codec stores 527 F32 ones. `docs/port-validation.md`'s
dtype-follows-the-checkpoint rule and its least-approximating-device rule pick
the same configuration here, which is the opposite of the qwen3-tts BF16
situation.

Unlike qwen3-tts, the codec's *encoder* half ships. Voice cloning from
reference audio is part of this family's product surface, so the HuBERT
semantic branch, the acoustic encoder, `encoder_semantic` and `fc` are all
carried; only the two modules that no code path reaches are dropped.

Four conversion rules here fail silently rather than loudly if they are wrong.
Each is asserted, not assumed:

1. `fc1.*` and `decoder_semantic.*` feed a training-only semantic
   reconstruction loss. `HiggsAudioV2TokenizerModel.encode` and `.decode` reach
   neither, so they are dropped -- and a checkpoint where the drop pattern
   matches nothing stops the conversion rather than quietly widening the
   package.
2. The RVQ codebooks *are* codebooks. `HiggsAudioV2TokenizerEuclideanCodebook`
   decodes with `F.embedding(embed_ind, self.embed)`, reading the stored table
   directly, so nothing is reconstructed. What the checkpoint also stores per
   quantizer is k-means training state -- `embed_avg`, `cluster_size`,
   `inited` -- and emitting those as weights would double the quantizer for
   nothing. A quantizer that carries the accumulators but no table is refused,
   because that would be the qwen3-tts situation and would need reconstruction.
3. HuBERT's positional convolution is stored as a weight-norm parametrization
   (`original0` = magnitude, `original1` = direction). Emitted verbatim the
   graph gets a magnitude and a direction where it expects a kernel. The fold
   goes through `torch._weight_norm`, the operator the reference itself calls
   on every forward, so the result is bit-identical rather than merely close.
4. GGML truncates a tensor name at 64 bytes. 195 of the codec's names sit
   within six characters of that once prefixed and the longest overruns it by
   eighteen, so the components that make them long are shortened -- the
   shortest edit that fixes each, not a renaming scheme.

Codebook counts and dimensions are measured from the tensors, never copied from
`audio_tokenizer/config.json`, which disagrees with its own weights three ways
(`n_codebooks` 9 against 8 tables, `acoustic_model_config.codebook_dim` 8
against [1024, 64] tables, `sampling_rate` 16000 against a 24 kHz output). The
disagreements are recorded in the conversion report rather than silently
resolved.

Usage:

    uv run --project scripts/envs/omnivoice --locked python \
      scripts/convert-omnivoice.py \
      --manifest tests/golden/omnivoice/omnivoice-0-6b.manifest.json \
      --weights-dir models/omnivoice-0-6b \
      --output models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
import hashlib
import importlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable

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

ARCH_KEY = "omnivoice"
FORMAT_VERSION = 1
PROFILE_NAME = "F32"
PROFILE_VERSION = 1
ARCHITECTURE_VERSION = 1
REPORT_SCHEMA = "synthesize-converter-report-v1"

# GGML stores a tensor name in a fixed 64-byte field and truncates silently past
# it. A truncated name is not findable by the name the catalog asks for, and two
# names that differ only past the cut become the same tensor.
#
# Every rule below is load-bearing: dropping any one of them puts at least one
# emitted name back over the limit, which `test_every_shortening_rule_is_load_bearing`
# checks against the committed inventory. The two other candidates considered --
# shortening `final_layer_norm` and `pos_conv_embed` -- were dropped because
# neither name overflows once the weight-norm fold has collapsed the
# parametrization pair, and a rename nobody needs only makes the catalog harder
# to read.
GGML_MAX_NAME = 64
NAME_SHORTENINGS = (
    (".attention.", ".attn."),
    (".feed_forward.", ".ff."),
    (".intermediate_dense.", ".inter_dense."),
    (".feature_extractor.conv_layers.", ".feat_conv."),
)

SAMPLE_RATE = 24000
HOP_LENGTH = 960
FRAME_RATE_HZ = 25.0
NUM_CODEBOOKS = 8
AUDIO_VOCAB = 1025
AUDIO_MASK_ID = 1024

CAPABILITY_SPEAKING_RATE = 1 << 0
CAPABILITY_STOCHASTIC = 1 << 1
INPUT_TEXT_UTF8 = 1 << 0

# The two modules nothing reads. Kept as a name->reason table so no skip can
# reach the report without saying why it happened.
DROP_PREFIX_REASONS = {
    "fc1.": (
        "projects the quantized latent into the semantic decoder used only by the "
        "training reconstruction loss; neither HiggsAudioV2TokenizerModel.encode nor "
        ".decode reaches it"
    ),
    "decoder_semantic.": (
        "reconstructs HuBERT features for the training loss; the synthesis path goes "
        "quantizer -> fc2 -> acoustic_decoder and never enters this module"
    ),
}
DROP_PREFIXES = tuple(DROP_PREFIX_REASONS)

# Per-quantizer k-means training state that sits beside the live table.
CODEBOOK_TABLE_LEAF = "embed"
CODEBOOK_TRAINING_BUFFERS = {
    "embed_avg": (
        "EMA numerator maintained by k-means training; decode reads codebook.embed "
        "directly, so carrying this would double the quantizer for nothing"
    ),
    "cluster_size": (
        "EMA denominator maintained by k-means training; the stored table is already "
        "the quotient this family's decode path uses"
    ),
    "inited": "shape-(1,) codebook-initialised flag, not a weight",
}

PARAMETRIZATION_INFIX = ".parametrizations."

OFFSETS_BUFFER = "codebook_layer_offsets"
OFFSETS_REASON = (
    f"derivable: arange({NUM_CODEBOOKS}) * {AUDIO_VOCAB}; I64 tensors are not "
    "GGUF-portable"
)

CODEC_LICENSE_NAME = "LICENSE-higgs-audio-2.txt"
# Quoted from the '## License' section of the model card at the pinned weights
# revision. Upstream names no CC-BY-NC version; this project must not invent one.
CHECKPOINT_LICENSE_STATEMENT = (
    "Our code is released under the Apache 2.0 License. The pre-trained model is "
    "licensed under the CC-BY-NC due to constraints from its training data (e.g., Emilia)."
)

GENERATION_DEFAULT_KEYS = (
    "num_step",
    "guidance_scale",
    "t_shift",
    "layer_penalty_factor",
    "position_temperature",
    "class_temperature",
)


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
    skipped: list[dict[str, Any]] = field(default_factory=list)
    transformed: list[dict[str, str]] = field(default_factory=list)
    renamed: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class CodecGeometry:
    """What the RVQ tables actually are, as opposed to what the config claims."""

    quantizer_count: int
    codebook_size: int
    codebook_dim: int


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


def pinned_digest(manifest: dict[str, Any], role: str, locator_suffix: str,
                  *, excluding: str | None = None) -> str:
    matches = [
        artifact for artifact in manifest["source"]["artifacts"]
        if artifact["role"] == role
        and artifact["locator"].endswith(locator_suffix)
        and (excluding is None or excluding not in artifact["locator"])
    ]
    if len(matches) != 1:
        raise ConverterError(
            f"the manifest names {len(matches)} {role} artifacts ending in "
            f"{locator_suffix!r}; exactly one is required to pin the conversion"
        )
    return matches[0]["sha256"]


@dataclass(frozen=True)
class PinnedInput:
    """A file the manifest pins by digest, and where its pin lives."""

    label: str
    path: Path
    role: str
    locator_suffix: str
    excluding: str | None = None


def pinned_inputs(weights_dir: Path) -> tuple[PinnedInput, ...]:
    """Every local file the conversion reads or republishes.

    The codec's LICENSE belongs here and not merely in the copy step. It is
    republished beside the artifact and its digest is recorded in the report as
    authoritative, so a locally edited grant would be carried into the package
    and vouched for. Hashing the copy against its own source is a tautology
    after `shutil.copyfile`; the manifest is the only outside witness.
    """
    return (
        PinnedInput("generator", weights_dir / "model.safetensors",
                    "checkpoint", "/model.safetensors", "/audio_tokenizer/"),
        PinnedInput("codec", weights_dir / "audio_tokenizer" / "model.safetensors",
                    "checkpoint", "/audio_tokenizer/model.safetensors"),
        PinnedInput("config", weights_dir / "config.json",
                    "config", "/config.json", "/audio_tokenizer/"),
        PinnedInput("codec_config", weights_dir / "audio_tokenizer" / "config.json",
                    "config", "/audio_tokenizer/config.json"),
        PinnedInput("tokenizer", weights_dir / "tokenizer.json",
                    "frontend-resource", "/tokenizer.json"),
        PinnedInput("codec_license", weights_dir / "audio_tokenizer" / "LICENSE",
                    "license", "/audio_tokenizer/LICENSE"),
    )


def verify_pinned_inputs(manifest: dict[str, Any],
                         inputs: Iterable[PinnedInput]) -> dict[str, str]:
    """Hash every input against its manifest pin before anything is read.

    Runs first, so a missing or altered input costs nothing rather than being
    discovered after a 3.2 GB file has been written.
    """
    digests: dict[str, str] = {}
    for entry in inputs:
        if not entry.path.is_file():
            raise ConverterError(
                f"{entry.path} is missing; the manifest pins it and the conversion cannot "
                "proceed without it"
            )
        digest = sha256_file(entry.path)
        expected = pinned_digest(manifest, entry.role, entry.locator_suffix,
                                 excluding=entry.excluding)
        if digest != expected:
            raise ConverterError(
                f"{entry.path}: the local file hashes to {digest} but the manifest pins "
                f"{expected}; converting it would produce a package nothing validated"
            )
        digests[entry.label] = digest
    return digests


def numpy_of(tensor: torch.Tensor) -> tuple[np.ndarray, GGMLQuantizationType]:
    """Carry the stored dtype through unchanged.

    This checkpoint is F32 throughout apart from one derivable I64 buffer that
    is dropped before it reaches here, so any other dtype means the layout
    moved and a silent upcast would produce a package for a model that does not
    exist.
    """
    if tensor.dtype is torch.float32:
        array = tensor.detach().contiguous().cpu().numpy()
        return np.ascontiguousarray(array, dtype=np.float32), GGMLQuantizationType.F32
    raise ConverterError(
        f"unhandled source dtype {tensor.dtype}; this family's checkpoint is F32 throughout "
        "and coercing anything else would hide a layout change"
    )


def drop_by_prefix(tensors: dict[str, torch.Tensor], prefixes: Iterable[str],
                   conversion: Conversion, prefix: str) -> dict[str, torch.Tensor]:
    """Remove whole modules the synthesis path cannot reach, loudly."""
    remaining = dict(tensors)
    for drop_prefix in prefixes:
        matched = sorted(name for name in remaining if name.startswith(drop_prefix))
        if not matched:
            raise ConverterError(
                f"nothing under {drop_prefix!r} to drop; the checkpoint's layout moved and "
                "this package would silently start carrying a module it never carried before"
            )
        reason = DROP_PREFIX_REASONS.get(drop_prefix)
        if reason is None:
            raise ConverterError(f"no recorded reason for dropping {drop_prefix!r}")
        for name in matched:
            del remaining[name]
        conversion.skipped.append({
            "logical_source_name": f"{prefix}{drop_prefix}*",
            "count": len(matched),
            "reason": reason,
        })
    return remaining


def drop_derivable_offsets(tensors: dict[str, torch.Tensor],
                           conversion: Conversion) -> dict[str, torch.Tensor]:
    """Drop the per-codebook offset buffer after proving it is derivable.

    It is this checkpoint's only non-F32 tensor. Dropping it without checking
    would delete real information the moment upstream changes the layout, so
    the values are compared against the rule that replaces them.
    """
    if OFFSETS_BUFFER not in tensors:
        raise ConverterError(
            f"the generator carries no {OFFSETS_BUFFER}; the checkpoint's layout moved"
        )
    expected = torch.arange(NUM_CODEBOOKS, dtype=torch.int64) * AUDIO_VOCAB
    actual = tensors[OFFSETS_BUFFER]
    if tuple(actual.shape) != tuple(expected.shape) or not torch.equal(
        actual.to(torch.int64), expected
    ):
        raise ConverterError(
            f"{OFFSETS_BUFFER} is {actual.tolist()}, not "
            f"arange({NUM_CODEBOOKS}) * {AUDIO_VOCAB} = {expected.tolist()}; it is no longer "
            "derivable and dropping it would delete information the graph needs"
        )
    remaining = {name: value for name, value in tensors.items() if name != OFFSETS_BUFFER}
    conversion.skipped.append({
        "logical_source_name": OFFSETS_BUFFER,
        "count": 1,
        "reason": OFFSETS_REASON,
    })
    return remaining


def skip_codebook_training_buffers(tensors: dict[str, torch.Tensor], conversion: Conversion,
                                   prefix: str) -> dict[str, torch.Tensor]:
    """Keep the live RVQ table, drop the k-means state that sits beside it."""
    table_suffix = f".codebook.{CODEBOOK_TABLE_LEAF}"
    bases = {name[: -len(CODEBOOK_TABLE_LEAF)] for name in tensors if name.endswith(table_suffix)}

    remaining: dict[str, torch.Tensor] = {}
    orphans: set[str] = set()
    skipped: list[dict[str, Any]] = []
    for name, tensor in tensors.items():
        head, _, leaf = name.rpartition(".")
        if leaf in CODEBOOK_TRAINING_BUFFERS and head.endswith(".codebook"):
            base = f"{head}."
            if base not in bases:
                orphans.add(base)
                continue
            skipped.append({
                "logical_source_name": f"{prefix}{name}",
                "count": 1,
                "reason": CODEBOOK_TRAINING_BUFFERS[leaf],
            })
            continue
        remaining[name] = tensor

    if orphans:
        first = sorted(orphans)[0]
        raise ConverterError(
            f"{first}* carries k-means accumulators but no {CODEBOOK_TABLE_LEAF} table. "
            "This checkpoint would need the table reconstructed from them; emitting nothing "
            "for that quantizer loses it with no error anywhere"
        )
    if not bases:
        raise ConverterError(
            f"no RVQ codebook table ({table_suffix}) is present; the quantizer's layout moved"
        )

    conversion.skipped.extend(sorted(skipped, key=lambda entry: entry["logical_source_name"]))
    return remaining


def folded_weight_name(name: str) -> str:
    """`a.b.parametrizations.weight.original0` -> `a.b.weight`."""
    base, _, rest = name.partition(PARAMETRIZATION_INFIX)
    parameter = rest.rsplit(".", 1)[0]
    if not parameter:
        raise ConverterError(f"{name} is not a parametrized weight")
    return f"{base}.{parameter}"


def _weight_norm_dim(magnitude: torch.Tensor, direction: torch.Tensor, target: str) -> int:
    """Recover the dimension weight norm was applied over from g's shape."""
    if magnitude.ndim != direction.ndim:
        raise ConverterError(
            f"{target}: magnitude {tuple(magnitude.shape)} and direction "
            f"{tuple(direction.shape)} disagree on rank, so the norm dimension is unknown"
        )
    axes = [axis for axis, extent in enumerate(magnitude.shape) if extent != 1]
    if len(axes) != 1:
        raise ConverterError(
            f"{target}: magnitude shape {tuple(magnitude.shape)} does not name exactly one "
            "non-singleton dimension, so the weight-norm dimension is ambiguous"
        )
    dim = axes[0]
    if magnitude.shape[dim] != direction.shape[dim]:
        raise ConverterError(
            f"{target}: magnitude and direction disagree along dimension {dim} "
            f"({magnitude.shape[dim]} against {direction.shape[dim]})"
        )
    return dim


def fold_weight_norm(tensors: dict[str, torch.Tensor], conversion: Conversion,
                     prefix: str) -> dict[str, torch.Tensor]:
    """Collapse weight-norm parametrizations into the weights they stand for.

    `torch._weight_norm` is the operator the reference calls on every forward,
    so folding through it is bit-identical rather than close. A hand-written
    `g * v / ||v||` agrees only to a few ulps, which is enough to move the
    argmax this family commits at every mask-predict step.
    """
    if not hasattr(torch, "_weight_norm"):
        raise ConverterError(
            "this torch build offers no torch._weight_norm; folding by hand would agree with "
            "the reference only to a few ulps, which is not enough for an exact-token family"
        )

    groups: dict[str, dict[str, str]] = {}
    for name in tensors:
        if PARAMETRIZATION_INFIX not in name:
            continue
        if not (name.endswith(".original0") or name.endswith(".original1")):
            raise ConverterError(
                f"{name} is a parametrization this converter does not understand; only "
                "weight norm's original0/original1 pair is handled"
            )
        groups.setdefault(folded_weight_name(name), {})[name[-1]] = name

    if not groups:
        raise ConverterError(
            "no weight norm parametrization is present; HuBERT's positional convolution "
            "always carries one, so the checkpoint's layout moved"
        )

    consumed: set[str] = set()
    folded: dict[str, torch.Tensor] = {}
    for target in sorted(groups):
        pair = groups[target]
        magnitude_name = pair.get("0")
        direction_name = pair.get("1")
        if magnitude_name is None:
            raise ConverterError(
                f"{target}: the weight-norm direction is present but original0, its "
                "magnitude, is missing"
            )
        if direction_name is None:
            raise ConverterError(
                f"{target}: the weight-norm magnitude is present but original1, its "
                "direction, is missing"
            )
        magnitude = tensors[magnitude_name]
        direction = tensors[direction_name]
        dim = _weight_norm_dim(magnitude, direction, target)
        folded[target] = torch._weight_norm(direction, magnitude, dim)
        consumed.update({magnitude_name, direction_name})
        conversion.transformed.append({
            "output": target,
            "from": f"{prefix}{magnitude_name} * {prefix}{direction_name} / "
                    f"||direction||(dim={dim})",
            "reason": (
                "the checkpoint stores a weight norm parametrization, not a kernel; folded "
                "through torch._weight_norm so the result is bit-identical to what the "
                "reference recomputes on every forward"
            ),
        })

    remaining = {name: value for name, value in tensors.items() if name not in consumed}
    remaining.update(folded)
    return remaining


def measure_codec_geometry(tensors: dict[str, torch.Tensor]) -> CodecGeometry:
    """Size the RVQ from its tables. `audio_tokenizer/config.json` is wrong."""
    suffix = f".codebook.{CODEBOOK_TABLE_LEAF}"
    tables: dict[int, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if not name.endswith(suffix):
            continue
        index = name[: -len(suffix)].rsplit(".", 1)[-1]
        if not index.isdigit():
            raise ConverterError(f"{name} does not carry a numeric quantizer index")
        tables[int(index)] = tensor

    if not tables:
        raise ConverterError(f"no RVQ codebook table ({suffix}) to measure the codec from")
    if sorted(tables) != list(range(len(tables))):
        raise ConverterError(
            f"quantizer indices {sorted(tables)} are not contiguous from zero; the residual "
            "stages would be read in the wrong order"
        )

    shapes = {tuple(int(extent) for extent in tensor.shape) for tensor in tables.values()}
    if len(shapes) != 1:
        raise ConverterError(f"RVQ codebook tables disagree on shape: {sorted(shapes)}")
    shape = shapes.pop()
    if len(shape) != 2:
        raise ConverterError(f"an RVQ codebook table has shape {shape}, expected [entries, dim]")
    return CodecGeometry(quantizer_count=len(tables), codebook_size=shape[0], codebook_dim=shape[1])


def config_disagreements(codec_config: dict[str, Any],
                         geometry: CodecGeometry) -> list[dict[str, Any]]:
    """Record where the codec config contradicts its own weights."""
    acoustic = codec_config["acoustic_model_config"]
    found: list[dict[str, Any]] = []

    def note(field_name: str, declared: int, measured: int, resolution: str) -> None:
        if declared != measured:
            found.append({
                "field": field_name,
                "declared": declared,
                "measured": measured,
                "resolution": resolution,
            })

    note("acoustic_model_config.n_codebooks", int(acoustic["n_codebooks"]),
         geometry.quantizer_count,
         "the tensors win: the package is sized from the codebook tables that exist")
    note("acoustic_model_config.codebook_dim", int(acoustic["codebook_dim"]),
         geometry.codebook_dim,
         "the tensors win: the top-level codebook_dim agrees with the tables and is written")
    note("acoustic_model_config.sampling_rate", int(acoustic["sampling_rate"]),
         int(codec_config["sample_rate"]),
         "the top-level sample_rate is the output rate; 16000 is the semantic branch's rate")
    return found


def validate_generation_defaults(defaults: dict[str, Any]) -> None:
    for key in GENERATION_DEFAULT_KEYS:
        if key not in defaults:
            raise ConverterError(
                f"the package's generation config carries no {key}; the mask-predict schedule "
                "cannot be written from a partial reading"
            )
    num_step = int(defaults["num_step"])
    if num_step <= 0:
        raise ConverterError(f"num_step is {num_step}; the mask-predict loop would never run")
    for key in GENERATION_DEFAULT_KEYS[1:]:
        value = float(defaults[key])
        if not math.isfinite(value):
            raise ConverterError(f"{key} is {value}, which cannot be decoded with")


def read_generation_defaults(module_name: str = "omnivoice") -> dict[str, Any]:
    """Read the decoding schedule from the pinned package, never a second copy.

    `num_step`, `guidance_scale` and `t_shift` decide what the model says, not
    merely how it sounds. A table restated here drifts from the package the
    oracle ran, and nothing would report the drift.
    """
    try:
        module = importlib.import_module(module_name)
        config_type = module.OmniVoiceGenerationConfig
    except (ImportError, AttributeError) as error:
        raise ConverterError(
            f"the pinned {module_name} package does not offer OmniVoiceGenerationConfig "
            f"({error}); run this converter under scripts/envs/omnivoice. The defaults are "
            "read from upstream and are deliberately not restated in this script"
        ) from error
    defaults = dataclasses.asdict(config_type())
    validate_generation_defaults(defaults)
    return defaults


def carry_licenses(weights_dir: Path, output: Path, project_root: Path,
                   pinned_license_sha256: str) -> list[dict[str, Any]]:
    """Copy the codec's grant beside the converted artifact and pin its hash.

    `audio_tokenizer/README.md` is an unedited template whose License field
    reads "[More Information Needed]", so the bundled LICENSE is the sole grant
    for the codec weights -- and the agreement requires redistribution to carry
    its text. The declarative Sidecar Resource descriptors are assembled at the
    ship stage; what this guarantees is that the text never separates from the
    artifact and its hash is pinned from the first cut.

    The copy is checked against the digest the manifest pins, not against its
    own source: re-hashing the source after `shutil.copyfile` compares a file
    with itself and would vouch for a locally edited grant.
    """
    source = weights_dir / "audio_tokenizer" / "LICENSE"
    if not source.is_file():
        raise ConverterError(
            f"{source} is missing; it is the codec weights' only grant and must travel "
            "with every converted artifact"
        )
    destination = output.parent / CODEC_LICENSE_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    digest = sha256_file(destination)
    if digest != pinned_license_sha256:
        raise ConverterError(
            f"{destination} hashes to {digest} but the manifest pins "
            f"{pinned_license_sha256}; the grant that would ship is not the audited one"
        )

    return [
        {
            "role": "codec-weights",
            "spdx": "LicenseRef-Boson-Higgs-Audio-2-Community",
            "source": project_relative(source, project_root),
            "path": project_relative(destination, project_root),
            "sha256": digest,
            "bytes": destination.stat().st_size,
            "statement": (
                "BOSON HIGGS AUDIO 2 COMMUNITY LICENSE AGREEMENT -- redistribution must "
                "carry the agreement text, commercial use is capped at 100k monthly active "
                "users, and outputs may not be used to train other models"
            ),
        },
        {
            "role": "checkpoint-weights",
            "spdx": "CC-BY-NC (no version stated by upstream)",
            "statement": CHECKPOINT_LICENSE_STATEMENT,
            "evidence": (
                "the '## License' section of the model card at the pinned weights revision; "
                "the card's frontmatter carries no license key, so this prose is the only "
                "statement upstream makes"
            ),
        },
        {
            "role": "source-code",
            "spdx": "Apache-2.0",
            "statement": "Our code is released under the Apache 2.0 License.",
            "evidence": "LICENSE at the pinned k2-fsa/OmniVoice source revision",
        },
    ]


def load_tensors(path: Path) -> dict[str, torch.Tensor]:
    # Imported here rather than at module scope so the pure-python rules above
    # stay importable in environments without safetensors -- the registered
    # python unit suite runs under a different family's locked environment.
    from safetensors import safe_open  # noqa: PLC0415

    with safe_open(str(path), framework="pt") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def convert_file(path: Path, prefix: str, conversion: Conversion, *,
                 drop_offsets: bool = False,
                 drop_prefixes: Iterable[str] = (),
                 skip_codebook_buffers: bool = False,
                 fold: bool = False) -> dict[str, torch.Tensor]:
    tensors = load_tensors(path)
    if drop_offsets:
        tensors = drop_derivable_offsets(tensors, conversion)
    if drop_prefixes:
        tensors = drop_by_prefix(tensors, drop_prefixes, conversion, prefix)
    if skip_codebook_buffers:
        tensors = skip_codebook_training_buffers(tensors, conversion, prefix)
    if fold:
        tensors = fold_weight_norm(tensors, conversion, prefix)

    origin = project_relative(path, Path.cwd())
    for name in sorted(tensors):
        array, dtype = numpy_of(tensors[name])
        conversion.outputs.append(OutputTensor(
            name=shorten_name(f"{prefix}{name}", conversion), array=array, dtype=dtype,
            origin=origin,
        ))
    return tensors


def add_metadata(writer: GGUFWriter, manifest: dict[str, Any], config: dict[str, Any],
                 codec_config: dict[str, Any], tokenizer_json: dict[str, Any],
                 gen_defaults: dict[str, Any], digests: dict[str, str],
                 geometry: CodecGeometry) -> None:
    llm = config["llm_config"]
    add_general_identity(
        writer,
        name="OmniVoice 0.6B",
        basename=manifest["variant"],
        size_label="0.6B",
        languages=[t for t in manifest["package_contract"]["language_tags"] if t != "auto"],
        tags=["text-to-speech", "omnivoice", "synthesize.cpp"],
        author="k2-fsa (Xiaomi)",
        organization="k2-fsa",
        source_url=manifest["source"]["repository"],
        description=(
            "Source-dtype synthesize.cpp conversion of the pinned k2-fsa/OmniVoice "
            "checkpoint: F32 mask-predict generator plus F32 Higgs Audio V2 codec."
        ),
        license_id="cc-by-nc-4.0",
        license_name="CC-BY-NC (version unstated upstream) + Boson Higgs Audio 2 Community License (codec)",
        license_link="https://huggingface.co/k2-fsa/OmniVoice",
    )
    writer.add_repo_url(manifest["source"]["repository"])

    writer.add_uint32("synthesize.format_version", FORMAT_VERSION)
    writer.add_string("synthesize.model_family", ARCH_KEY)
    writer.add_string("synthesize.model_variant", manifest["variant"])
    writer.add_string("synthesize.quantization.profile", PROFILE_NAME)
    writer.add_uint32("synthesize.quantization.profile_version", PROFILE_VERSION)
    writer.add_string("synthesize.source.repository", manifest["source"]["repository"])
    writer.add_string("synthesize.source.revision", manifest["source"]["revision"])
    writer.add_string("synthesize.source.checkpoint.sha256", digests["generator"])
    writer.add_string("synthesize.source.codec.sha256", digests["codec"])
    writer.add_string("synthesize.source.config.sha256", digests["config"])
    writer.add_string("synthesize.source.checkpoint.license_status",
                      "cc-by-nc (LM, version unstated) + boson-higgs-audio-2-community (codec) + apache-2.0 (code)")
    writer.add_string("synthesize.converter", "scripts/convert-omnivoice.py")
    writer.add_uint32("synthesize.omnivoice.architecture_version", ARCHITECTURE_VERSION)

    # Generation defaults: read from the pinned package, never restated.
    writer.add_uint32("synthesize.omnivoice.generation.num_step", int(gen_defaults["num_step"]))
    writer.add_float32("synthesize.omnivoice.generation.guidance_scale", float(gen_defaults["guidance_scale"]))
    writer.add_float32("synthesize.omnivoice.generation.t_shift", float(gen_defaults["t_shift"]))
    writer.add_float32("synthesize.omnivoice.generation.layer_penalty_factor", float(gen_defaults["layer_penalty_factor"]))
    writer.add_float32("synthesize.omnivoice.generation.position_temperature", float(gen_defaults["position_temperature"]))
    writer.add_float32("synthesize.omnivoice.generation.class_temperature", float(gen_defaults["class_temperature"]))

    package = manifest["package_contract"]
    writer.add_uint32("synthesize.capabilities.input_flags", INPUT_TEXT_UTF8)
    writer.add_uint32("synthesize.capabilities.flags",
                      CAPABILITY_SPEAKING_RATE | CAPABILITY_STOCHASTIC)
    writer.add_uint64("synthesize.capabilities.max_input_tokens", int(package["max_input_tokens"]))
    writer.add_uint64("synthesize.capabilities.max_output_frames", int(package["max_output_frames"]))
    low, high = package["speaking_rate_range"]
    writer.add_float32("synthesize.capabilities.min_speaking_rate", float(low))
    writer.add_float32("synthesize.capabilities.max_speaking_rate", float(high))

    audio = package["native_audio"]
    writer.add_uint32("synthesize.audio.sample_rate_hz", int(audio["sample_rate_hz"]))
    writer.add_uint32("synthesize.audio.channels", int(audio["channels"]))
    writer.add_string("synthesize.audio.sample_format", audio["sample_format"])

    # Voice: no preset catalog; the package default is auto-voice.
    writer.add_string("synthesize.voice.mode", "profile-sources")
    writer.add_bool("synthesize.voice.has_package_default", True)
    writer.add_uint32("synthesize.voice.preset_count", 0)

    # Generator geometry (checked against llm_config).
    for key, value in (
        ("layer_count", llm["num_hidden_layers"]),
        ("hidden_size", llm["hidden_size"]),
        ("attention_head_count", llm["num_attention_heads"]),
        ("key_value_head_count", llm["num_key_value_heads"]),
        ("head_dim", llm["head_dim"]),
        ("intermediate_size", llm["intermediate_size"]),
        ("text_vocab_size", llm["vocab_size"]),
    ):
        writer.add_uint32(f"synthesize.omnivoice.generator.{key}", int(value))
    writer.add_float32("synthesize.omnivoice.generator.rms_norm_eps", float(llm["rms_norm_eps"]))
    writer.add_float32("synthesize.omnivoice.generator.rope_theta", float(llm["rope_parameters"]["rope_theta"]))
    if any(t != "full_attention" for t in llm["layer_types"]):
        raise ConverterError("a layer_type is not full_attention; this port builds bidirectional graphs only")
    writer.add_string("synthesize.omnivoice.generator.attention", "bidirectional")
    if not bool(llm["tie_word_embeddings"]):
        raise ConverterError("tie_word_embeddings is false; the catalog expects no separate lm_head")

    # Audio canvas contract. The codebook count is cross-checked against the
    # tables that exist rather than trusted, because the codec config's own
    # n_codebooks says nine.
    if int(config["num_audio_codebook"]) != geometry.quantizer_count:
        raise ConverterError(
            f"config declares {config['num_audio_codebook']} audio codebooks but the codec "
            f"carries {geometry.quantizer_count} RVQ tables"
        )
    if int(config["audio_vocab_size"]) != AUDIO_VOCAB or int(config["audio_mask_id"]) != AUDIO_MASK_ID:
        raise ConverterError(
            f"the audio canvas moved: vocab {config['audio_vocab_size']} mask "
            f"{config['audio_mask_id']}, expected {AUDIO_VOCAB} and {AUDIO_MASK_ID}"
        )
    writer.add_uint32("synthesize.omnivoice.audio.num_codebooks", int(config["num_audio_codebook"]))
    writer.add_uint32("synthesize.omnivoice.audio.vocab_size", int(config["audio_vocab_size"]))
    writer.add_uint32("synthesize.omnivoice.audio.mask_id", int(config["audio_mask_id"]))

    # Codec geometry, checked rather than copied (size codebooks from tensors, not config).
    writer.add_uint32("synthesize.omnivoice.codec.sample_rate", int(codec_config["sample_rate"]))
    writer.add_uint32("synthesize.omnivoice.codec.hop_length", HOP_LENGTH)
    writer.add_float32("synthesize.omnivoice.codec.frame_rate_hz", FRAME_RATE_HZ)
    if int(codec_config["sample_rate"]) != SAMPLE_RATE:
        raise ConverterError("codec sample rate moved")
    if abs(SAMPLE_RATE / HOP_LENGTH - FRAME_RATE_HZ) > 1e-9:
        raise ConverterError(f"{SAMPLE_RATE} Hz over hop {HOP_LENGTH} is not {FRAME_RATE_HZ} Hz")
    acoustic = codec_config["acoustic_model_config"]
    ratios = [int(r) for r in acoustic["upsampling_ratios"]]
    total = 1
    for r in ratios:
        total *= r
    if total != HOP_LENGTH:
        raise ConverterError(f"upsampling ratios {ratios} multiply to {total}, not hop {HOP_LENGTH}")
    writer.add_array("synthesize.omnivoice.codec.upsampling_ratios", ratios)
    writer.add_uint32("synthesize.omnivoice.codec.decoder_hidden_size", int(acoustic["decoder_hidden_size"]))
    writer.add_uint32("synthesize.omnivoice.codec.encoder_hidden_size", int(acoustic["encoder_hidden_size"]))
    writer.add_uint32("synthesize.omnivoice.codec.hidden_size", int(acoustic["hidden_size"]))
    if int(codec_config["codebook_dim"]) != geometry.codebook_dim:
        raise ConverterError(
            f"codebook_dim {codec_config['codebook_dim']} contradicts the "
            f"[{geometry.codebook_size}, {geometry.codebook_dim}] tables the codec carries"
        )
    if int(codec_config["codebook_size"]) != geometry.codebook_size:
        raise ConverterError(
            f"codebook_size {codec_config['codebook_size']} contradicts the "
            f"[{geometry.codebook_size}, {geometry.codebook_dim}] tables the codec carries"
        )
    writer.add_uint32("synthesize.omnivoice.codec.codebook_dim", int(codec_config["codebook_dim"]))
    writer.add_uint32("synthesize.omnivoice.codec.codebook_size", int(codec_config["codebook_size"]))
    writer.add_uint32("synthesize.omnivoice.codec.semantic_sample_rate", int(codec_config["semantic_sample_rate"]))
    semantic = codec_config["semantic_model_config"]
    writer.add_uint32("synthesize.omnivoice.semantic.hidden_size", int(semantic["hidden_size"]))
    writer.add_uint32("synthesize.omnivoice.semantic.layer_count", int(semantic["num_hidden_layers"]))
    writer.add_uint32("synthesize.omnivoice.semantic.attention_head_count", int(semantic["num_attention_heads"]))
    writer.add_uint32("synthesize.omnivoice.semantic.intermediate_size", int(semantic["intermediate_size"]))
    writer.add_array("synthesize.omnivoice.semantic.conv_dim", [int(x) for x in semantic["conv_dim"]])
    writer.add_array("synthesize.omnivoice.semantic.conv_kernel", [int(x) for x in semantic["conv_kernel"]])
    writer.add_array("synthesize.omnivoice.semantic.conv_stride", [int(x) for x in semantic["conv_stride"]])
    writer.add_float32("synthesize.omnivoice.semantic.layer_norm_eps", float(semantic["layer_norm_eps"]))

    # Special token ids the prompt builder needs by value.
    specials = {t["content"]: int(t["id"]) for t in tokenizer_json["added_tokens"]}
    for name, key in (("<|denoise|>", "denoise"), ("<|lang_start|>", "lang_start"), ("<|lang_end|>", "lang_end"),
                      ("<|instruct_start|>", "instruct_start"), ("<|instruct_end|>", "instruct_end"),
                      ("<|text_start|>", "text_start"), ("<|text_end|>", "text_end")):
        if name not in specials:
            raise ConverterError(f"tokenizer carries no {name}")
        writer.add_uint32(f"synthesize.omnivoice.token.{key}", specials[name])
    writer.add_uint32("synthesize.omnivoice.token.eos", int(config["eos_token_id"]))
    writer.add_uint32("synthesize.omnivoice.token.pad", int(config["pad_token_id"]))

    # Language Capability Catalog: validated tags only; prompt text == tag. Taken
    # from the manifest's own contract rather than restated, which is the same
    # filtered list `general.languages` above carries.
    writer.add_array("synthesize.omnivoice.languages.tags",
                     [t for t in package["language_tags"] if t != "auto"])

    # Voice Profile contract (consumed from Plan 3 on; declared now so one
    # package serves the whole cycle).
    writer.add_string("synthesize.profile.schema", "omnivoice-clone-prompt")
    writer.add_uint32("synthesize.profile.schema_version", 1)
    writer.add_uint32("synthesize.reference.target_sample_rate", SAMPLE_RATE)
    writer.add_uint32("synthesize.reference.target_channels", 1)
    writer.add_uint64("synthesize.reference.min_frames_per_clip", 24000)      # 1 s
    writer.add_uint64("synthesize.reference.max_frames_per_clip", 480000)     # 20 s (upstream warns beyond)
    writer.add_uint64("synthesize.reference.max_total_frames", 480000)
    writer.add_uint64("synthesize.reference.max_reference_count", 1)

    # Profile Compatibility ID: sha256 over the family compatibility manifest.
    compat = hashlib.sha256()
    for piece in ("omnivoice-clone-prompt/1", digests["generator"], digests["codec"], digests["config"]):
        compat.update(piece.encode())
    writer.add_string("synthesize.profile.compatibility_id", compat.hexdigest())

    # Text Frontend payload from tokenizer.json.
    vocab = tokenizer_json["model"]["vocab"]
    ordered = sorted(vocab, key=lambda token: vocab[token])
    merges = [" ".join(m) if isinstance(m, list) else m for m in tokenizer_json["model"]["merges"]]
    writer.add_array("synthesize.omnivoice.frontend.vocab", ordered)
    writer.add_array("synthesize.omnivoice.frontend.merges", merges)
    writer.add_bool("synthesize.frontend.present", True)
    writer.add_string("synthesize.frontend.provider", "synthesize.qwen_bpe")
    writer.add_uint32("synthesize.frontend.contract_version", 1)


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
    generator_path = weights / "model.safetensors"
    codec_path = weights / "audio_tokenizer" / "model.safetensors"

    # Every pinned input -- weights, both configs, the tokenizer and the codec's
    # LICENSE -- is verified before a single tensor is read.
    digests = verify_pinned_inputs(manifest, pinned_inputs(weights))

    config = load_json(weights / "config.json")
    codec_config = load_json(weights / "audio_tokenizer" / "config.json")
    tokenizer_json = load_json(weights / "tokenizer.json")

    gen_defaults = read_generation_defaults()

    conversion = Conversion()
    convert_file(generator_path, "", conversion, drop_offsets=True)
    generator_count = len(conversion.outputs)
    codec_tensors = convert_file(codec_path, "codec.", conversion,
                                 drop_prefixes=DROP_PREFIXES,
                                 skip_codebook_buffers=True, fold=True)
    codec_count = len(conversion.outputs) - generator_count

    geometry = measure_codec_geometry(codec_tensors)
    disagreements = config_disagreements(codec_config, geometry)

    names = [o.name for o in conversion.outputs]
    if len(set(names)) != len(names):
        duplicates = sorted({n for n in names if names.count(n) > 1})
        raise ConverterError(f"tensor names collide after prefixing: {duplicates}")

    with atomic_output_path(args.output) as staging:
        writer = GGUFWriter(str(staging), ARCH_KEY)
        add_metadata(writer, manifest, config, codec_config, tokenizer_json, gen_defaults,
                     digests, geometry)
        for output in conversion.outputs:
            writer.add_tensor(output.name, output.array, raw_dtype=output.dtype)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    verify_gguf(args.output, conversion.outputs)
    licenses = carry_licenses(weights, args.output, project_root, digests["codec_license"])

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
            "path": "scripts/convert-omnivoice.py",
            "environment_lock": "scripts/envs/omnivoice/uv.lock",
            "environment_lock_sha256": sha256_file(Path("scripts/envs/omnivoice/uv.lock")),
        },
        "source": {
            "repository": manifest["source"]["repository"],
            "revision": manifest["source"]["revision"],
            "generator_path": project_relative(generator_path, project_root),
            "generator_sha256": digests["generator"],
            "codec_path": project_relative(codec_path, project_root),
            "codec_sha256": digests["codec"],
            "config_sha256": digests["config"],
            "codec_config_sha256": digests["codec_config"],
            "tokenizer_sha256": digests["tokenizer"],
            "codec_license_sha256": digests["codec_license"],
        },
        "output": {
            "path": project_relative(args.output, project_root),
            "bytes": args.output.stat().st_size,
            "sha256": sha256_file(args.output),
            "emitted_tensor_count": len(conversion.outputs),
            "generator_tensor_count": generator_count,
            "codec_tensor_count": codec_count,
            "tensor_count_by_dtype": by_dtype,
        },
        "dtype_policy": "F32 checkpoint carried through unchanged",
        "codec_geometry": dataclasses.asdict(geometry),
        "config_disagreements": disagreements,
        "generation_defaults": {key: gen_defaults[key] for key in GENERATION_DEFAULT_KEYS},
        "generation_defaults_source": "omnivoice.OmniVoiceGenerationConfig at the pinned revision",
        "licenses": licenses,
        "transformed": conversion.transformed,
        "renamed": conversion.renamed,
        "skipped": conversion.skipped,
        "tensors": [o.report() for o in conversion.outputs],
    }
    report_path = args.report or Path(f"reports/convert/{ARCH_KEY}/{manifest['variant']}-{PROFILE_NAME}.json")
    write_json_atomic(report_path, report)

    print(f"wrote {args.output} ({report['output']['bytes']:,} bytes)")
    print(f"  tensors: {len(conversion.outputs)} {by_dtype} "
          f"(generator {generator_count}, codec {codec_count})")
    print(f"  folded: {len(conversion.transformed)}  renamed: {len(conversion.renamed)}  "
          f"skipped: {len(conversion.skipped)}")
    print(f"  codec geometry: {geometry}  config disagreements: {len(disagreements)}")
    print(f"  licence: {licenses[0]['path']}")
    print(f"  report: {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConverterError as error:
        print(f"convert-omnivoice: {error}", file=sys.stderr)
        raise SystemExit(1)
