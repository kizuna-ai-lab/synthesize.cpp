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
    if spec["library_name"] != "synthesize.cpp":
        raise ValueError("library_name must be synthesize.cpp")
    if spec["pipeline_tag"] != "text-to-speech":
        raise ValueError("pipeline_tag must be text-to-speech")
    if re.fullmatch(r"[a-z0-9.-]+", spec["license_name"]) is None:
        raise ValueError("license_name must match the Hugging Face lowercase slug format")
    if spec["validation"].get("quality_evaluation") not in {"not_run", "complete"}:
        raise ValueError("quality_evaluation must be not_run or complete")
    if spec["validation"].get("level") not in {"port_validated", "quality_evaluated"}:
        raise ValueError("validation level must be port_validated or quality_evaluated")

    voice_mode = spec["capabilities"].get("voice_mode", "preset_catalog")
    if voice_mode not in {"preset_catalog", "fixed_default"}:
        raise ValueError("voice_mode must be preset_catalog or fixed_default")
    if voice_mode == "fixed_default" and spec["usage"].get("voice"):
        raise ValueError("fixed-default Voice packages must not specify a usage voice")
    if voice_mode == "preset_catalog" and not spec["usage"].get("voice"):
        raise ValueError("preset-catalog Voice packages must specify a usage voice")
    if not spec["usage"].get("phonemes"):
        raise ValueError("usage must specify a non-empty phoneme example")

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


def render(spec: dict, upstream_card: str) -> str:
    environment = Environment(
        loader=FileSystemLoader(HERE),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        autoescape=False,
    )
    template = environment.get_template("template.md.j2")
    return template.render(upstream_card=upstream_card, **spec)


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
