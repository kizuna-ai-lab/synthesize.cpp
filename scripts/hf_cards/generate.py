#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "jinja2>=3.1",
#   "pyyaml>=6.0",
# ]
# ///
"""Generate and verify a synthesize.cpp Hugging Face model card."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
REQUIRED_KEYS = (
    "model_slug",
    "display_name",
    "target_repo",
    "source",
    "license",
    "license_name",
    "license_link",
    "library_name",
    "pipeline_tag",
    "languages",
    "tags",
    "summary",
    "capabilities",
    "validation",
    "publication_note",
    "license_note",
    "architecture_label",
    "usage",
    "quants",
)


def load_spec(path: Path) -> dict:
    with path.open(encoding="utf-8") as spec_file:
        spec = yaml.safe_load(spec_file)
    if not isinstance(spec, dict):
        raise ValueError(f"model-card spec must be a mapping: {path}")
    return spec


def validate_spec(spec: dict) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in spec]
    if missing:
        raise ValueError(f"model-card spec is missing required keys: {', '.join(missing)}")
    if not isinstance(spec["source"].get("repository_label"), str):
        raise ValueError("source.repository_label must name the upstream repository")
    columns = spec["validation"].get("columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError("validation.columns must list at least one measured column")
    for column in columns:
        if not isinstance(column, dict) or "key" not in column or "title" not in column:
            raise ValueError("each validation column needs a key and a title")
    if not isinstance(spec["validation"].get("metric_note"), str):
        raise ValueError("validation.metric_note must say what the measured column means")
    if spec["library_name"] != "synthesize.cpp":
        raise ValueError("library_name must be synthesize.cpp")
    if spec["pipeline_tag"] != "text-to-speech":
        raise ValueError("pipeline_tag must be text-to-speech")
    if re.fullmatch(r"[a-z0-9.-]+", spec["license_name"]) is None:
        raise ValueError("license_name must match the Hugging Face lowercase slug format")
    if spec["validation"].get("quality_evaluation") not in {"not_run", "complete"}:
        raise ValueError("quality_evaluation must be not_run or complete")
    # docs/model-porting.md requires a Model Page and its generated card to report
    # exactly one of these. The field is separate from quality_evaluation: an
    # audit is one maintainer listening for obvious regressions, and the automated
    # grid is a different claim that this project cannot make at all yet.
    if spec["validation"].get("listening_audit", "not_run") not in {
        "no_obvious_regression",
        "regression",
        "not_run",
    }:
        raise ValueError("listening_audit must be no_obvious_regression, regression or not_run")
    if spec["validation"].get("level") not in {"port_validated", "quality_evaluated"}:
        raise ValueError("validation level must be port_validated or quality_evaluated")

    # Three voice modes. `preset_catalog` selects a named Preset Voice from a
    # non-empty catalog. `fixed_default` has one fixed, untrained-to-vary
    # Voice and no catalog at all. `seed_default_with_profiles` is a third
    # shape neither of those describes: an empty catalog whose unnamed
    # package default has no fixed identity of its own -- with no Voice
    # Profile supplied, which speaker a request produces follows the
    # synthesis seed -- and which also accepts caller-built Voice Profiles
    # (Reference Audio and/or Description Text) to pin a specific identity
    # instead of leaving it to the seed.
    voice_mode = spec["capabilities"].get("voice_mode", "preset_catalog")
    if voice_mode not in {"preset_catalog", "fixed_default", "seed_default_with_profiles"}:
        raise ValueError(
            "voice_mode must be preset_catalog, fixed_default or seed_default_with_profiles"
        )
    if voice_mode in {"fixed_default", "seed_default_with_profiles"} and spec["usage"].get("voice"):
        raise ValueError("fixed-default and seed-default Voice packages must not specify a usage voice")
    if voice_mode == "preset_catalog" and not spec["usage"].get("voice"):
        raise ValueError("preset-catalog Voice packages must specify a usage voice")
    if voice_mode == "seed_default_with_profiles":
        sources = spec["capabilities"].get("voice_profile_sources") or []
        if not sources or not set(sources) <= {"reference_audio", "description_text"}:
            raise ValueError(
                "seed_default_with_profiles requires capabilities.voice_profile_sources drawn "
                "from reference_audio and/or description_text"
            )

    # The built-in frontend either tokenizes raw text directly (no G2P step
    # exists or is needed) or consumes phonemes a caller's own G2P frontend
    # produced. Which one this package is decides both the "Voices and
    # input" prose and which CLI flag the usage example must demonstrate.
    input_kinds = spec["capabilities"].get("input_kinds", [])
    if "text_utf8" in input_kinds:
        if not spec["usage"].get("text"):
            raise ValueError("usage must specify a non-empty text example for a text_utf8 frontend")
    elif not spec["usage"].get("phonemes"):
        raise ValueError("usage must specify a non-empty phoneme example")

    # The CUDA placement sentence is a positive assertion about every node in
    # the replayed graph; it must not be printed for a package that never
    # made that claim, and it must say which stage stayed on CPU when the
    # claim is partial rather than silently generalizing a codec-only result
    # to the whole package.
    #
    # This field is REQUIRED, and deliberately has no usable default. It used
    # to default to `full`, which meant four of the five shipped cards asserted
    # "CUDA placement contained zero executable CPU fallback nodes" without any
    # spec ever saying so. Three of those four were false: VITS, Kokoro and
    # Qwen3-TTS each hold stages on CPU by design under `docs/backends.md`'s
    # discrete-outputs rule. `docs/backends.md` line 9 forbids exactly this --
    # placement "must not be presented as full GPU execution" -- and a default
    # is how the strongest possible claim got made by nobody's decision. A
    # package with no placement measurement declares `not_recorded`; saying so
    # is honest, and inventing a placement description would be worse.
    if "cuda_placement" not in spec["validation"]:
        raise ValueError(
            "validation.cuda_placement is required: declare full, partial, none or "
            "not_recorded rather than relying on a default"
        )
    cuda_placement = spec["validation"]["cuda_placement"]
    if cuda_placement not in {"full", "partial", "none", "not_recorded"}:
        raise ValueError(
            "validation.cuda_placement must be full, partial, none or not_recorded"
        )
    if cuda_placement == "partial" and not spec["validation"].get("cuda_placement_note"):
        raise ValueError("validation.cuda_placement 'partial' requires a cuda_placement_note")

    sidecar_filenames: set[str] = set()
    for sidecar in spec.get("sidecars", []):
        for key in ("filename", "role", "size", "size_bytes", "sha256"):
            if key not in sidecar:
                raise ValueError(f"sidecar entry is missing {key}")
        filename = sidecar["filename"]
        if Path(filename).name != filename:
            raise ValueError(f"sidecar filename must be a flat repository path: {filename}")
        if filename in sidecar_filenames:
            raise ValueError(f"sidecar filenames must be unique: {filename}")
        if not isinstance(sidecar["size_bytes"], int) or sidecar["size_bytes"] <= 0:
            raise ValueError(f"sidecar size_bytes must be positive: {filename}")
        sha256 = sidecar["sha256"]
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ValueError(f"sidecar SHA-256 must have 64 hexadecimal characters: {filename}")
        try:
            int(sha256, 16)
        except ValueError as exc:
            raise ValueError(f"sidecar SHA-256 is not hexadecimal: {filename}") from exc
        sidecar_filenames.add(filename)

    slug = spec["model_slug"]
    names: set[str] = set()
    filenames: set[str] = set()
    for quant in spec["quants"]:
        for key in ("name", "filename", "size", "size_bytes", "sha256", "tensor_types"):
            if key not in quant:
                raise ValueError(f"quant entry is missing {key}")
        name = quant["name"]
        filename = quant["filename"]
        if name in names or filename in filenames:
            raise ValueError("quant names and filenames must be unique")
        if filename != f"{slug}-{name}.gguf":
            raise ValueError(f"quant filename does not match profile: {filename}")
        if Path(filename).name != filename:
            raise ValueError(f"quant filename must be a flat repository path: {filename}")
        if not isinstance(quant["size_bytes"], int) or quant["size_bytes"] <= 0:
            raise ValueError(f"quant size_bytes must be positive: {filename}")
        sha256 = quant["sha256"]
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ValueError(f"quant SHA-256 must have 64 hexadecimal characters: {filename}")
        try:
            int(sha256, 16)
        except ValueError as exc:
            raise ValueError(f"quant SHA-256 is not hexadecimal: {filename}") from exc
        names.add(name)
        filenames.add(filename)

    if sidecar_filenames & filenames:
        raise ValueError("sidecar filenames must be distinct from quant filenames")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        while chunk := model_file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifacts(spec: dict, model_dir: Path) -> None:
    for quant in spec["quants"]:
        path = model_dir / quant["filename"]
        if not path.is_file():
            raise ValueError(f"missing GGUF artifact: {path}")
        actual_size = path.stat().st_size
        if actual_size != quant["size_bytes"]:
            raise ValueError(
                f"size mismatch for {path.name}: expected {quant['size_bytes']}, got {actual_size}"
            )
        actual_sha256 = file_sha256(path)
        if actual_sha256 != quant["sha256"]:
            raise ValueError(
                f"SHA-256 mismatch for {path.name}: expected {quant['sha256']}, got {actual_sha256}"
            )
    for sidecar in spec.get("sidecars", []):
        path = model_dir / sidecar["filename"]
        if not path.is_file():
            raise ValueError(f"missing sidecar artifact: {path}")
        actual_size = path.stat().st_size
        if actual_size != sidecar["size_bytes"]:
            raise ValueError(
                f"size mismatch for {path.name}: expected {sidecar['size_bytes']}, got {actual_size}"
            )
        actual_sha256 = file_sha256(path)
        if actual_sha256 != sidecar["sha256"]:
            raise ValueError(
                f"SHA-256 mismatch for {path.name}: expected {sidecar['sha256']}, got {actual_sha256}"
            )


def strip_frontmatter(card: str) -> str:
    if card.startswith("---\n"):
        end = card.find("\n---\n", 4)
        if end != -1:
            return card[end + len("\n---\n") :].strip()
    return card.strip()


def load_upstream_card(spec: dict) -> str:
    card_path = REPO_ROOT / spec["source"]["card_path"]
    if not card_path.is_file():
        raise ValueError(f"missing pinned upstream card: {card_path}")
    return strip_frontmatter(card_path.read_text(encoding="utf-8"))


# Prose blocks in a spec may reference the spec's own fields, so a licence note
# can cite the licence link without repeating the URL. They are rendered first,
# and only they: the result is inserted into the card as plain text.
PROSE_KEYS = ("license_note", "publication_note", "summary")


def render(spec: dict, upstream_card: str) -> str:
    environment = Environment(
        loader=FileSystemLoader(HERE),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        autoescape=False,
    )
    resolved = dict(spec)
    for key in PROSE_KEYS:
        if isinstance(resolved.get(key), str):
            resolved[key] = environment.from_string(resolved[key]).render(**spec)
    template = environment.get_template("template.md.j2")
    return template.render(upstream_card=upstream_card, **resolved)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="YAML model-card specification")
    parser.add_argument("-o", "--output", type=Path, help="output README.md")
    parser.add_argument("--stdout", action="store_true", help="write the rendered card to stdout")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that the output exists and exactly matches the rendered card",
    )
    args = parser.parse_args()

    spec = load_spec(args.spec)
    validate_spec(spec)
    model_dir = REPO_ROOT / "models" / spec["model_slug"]
    validate_artifacts(spec, model_dir)
    card = render(spec, load_upstream_card(spec))
    output = args.output or (model_dir / "README.md")

    if args.stdout:
        sys.stdout.write(card)
        return 0
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != card:
            print(f"generated model card is stale or missing: {output}", file=sys.stderr)
            return 1
        print(f"verified {output}", file=sys.stderr)
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(card, encoding="utf-8")
    print(f"wrote {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
