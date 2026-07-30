#!/usr/bin/env python3
"""Drive the pinned OmniVoice oracle over every Golden Manifest case.

**The dump half of this script does not exist yet.** What is committed here is
the manifest reader and the argument surface: it loads
``tests/golden/omnivoice/omnivoice-0-6b.manifest.json``, checks the structure
the dumper will depend on, resolves the case selection, and reports what it
would produce. Asking it to actually dump exits non-zero with a message saying
so, rather than writing an empty tree that a later comparison would read as
agreement.

That split is deliberate. The manifest names this file as
``reference.runner``, and tests/python/test_golden_manifests.py requires a
named runner to be committed -- a contract that points at a missing file is not
a contract. Meeting it with an empty stub would be worse than not meeting it,
so the part that can be written and exercised now is written now, and the part
that needs the weights arrives with them.

What the dumper will do, from the manifest this script already reads: drive the
pinned ``OmniVoice.generate`` entry point at F32 on CPU, capture the step-0
conditional forward's generator probes, the resulting 8 x T token grid, and the
waveform, and write each case's artifacts under ``case_artifact_root``. Greedy
cases make no RNG call at all -- ``position_temperature`` and
``class_temperature`` are both zero, so neither Gumbel branch is taken -- which
is why their grid is an exact-equality target rather than a replay input. The
sampled cases draw from the global torch RNG, which upstream exposes no seed
for; their grid is a replay artifact feeding the codec comparison, and the
port's own seed contract is what validates the sampled path.

Usage once the dump half lands:

    uv run --project scripts/envs/omnivoice --locked python \\
      scripts/dump_reference_omnivoice_pytorch.py \\
      --manifest tests/golden/omnivoice/omnivoice-0-6b.manifest.json \\
      --weights-dir models/omnivoice-0-6b

Today, and until then:

    python scripts/dump_reference_omnivoice_pytorch.py \\
      --manifest tests/golden/omnivoice/omnivoice-0-6b.manifest.json \\
      --validate-only
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

MANIFEST_SCHEMA = "synthesize-golden-manifest-v1"
FAMILY = "omnivoice"
# The manifest names these; the hooks the dumper installs must produce exactly
# this set, captured on the step-0 conditional forward.
GENERATOR_PROBE_LAYERS = (0, 7, 14, 21, 27)

NOT_IMPLEMENTED = (
    "the OmniVoice reference dumper is not implemented yet: this script "
    "currently reads and checks the manifest but captures nothing. Re-run with "
    "--validate-only to exercise what exists."
)


class ManifestError(Exception):
    """The manifest does not describe something this runner could dump."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--weights-dir", type=pathlib.Path, default=None)
    parser.add_argument("--output-root", type=pathlib.Path, default=None)
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="Restrict the dump to the given case id (repeatable).",
    )
    parser.add_argument("--report", type=pathlib.Path, default=None)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Read and check the manifest, print the planned work, and stop.",
    )
    return parser.parse_args(argv)


def load_manifest(path: pathlib.Path) -> dict:
    """Read the manifest and check the structure the dumper depends on.

    These checks overlap tests/python/test_golden_manifests.py on purpose. That
    suite gates the committed manifest; this one gates whatever file is handed
    to --manifest, which may be an edited copy.
    """
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ManifestError(f"no manifest at {path}") from error
    except json.JSONDecodeError as error:
        raise ManifestError(f"{path} is not valid JSON: {error}") from error

    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ManifestError(
            f"{path} declares schema {manifest.get('schema')!r}, expected {MANIFEST_SCHEMA!r}"
        )
    if manifest.get("family") != FAMILY:
        raise ManifestError(
            f"{path} is a {manifest.get('family')!r} manifest; this runner drives {FAMILY!r}"
        )

    reference = manifest.get("reference", {})
    if reference.get("dtype") != "float32" or reference.get("device") != "cpu":
        raise ManifestError(
            "this runner drives the oracle at float32 on cpu; the manifest asks for "
            f"{reference.get('dtype')!r} on {reference.get('device')!r}"
        )

    cases = manifest.get("cases")
    if not cases:
        raise ManifestError(f"{path} declares no cases")

    seen: set[str] = set()
    for case in cases:
        case_id = case["id"]
        if case_id in seen:
            raise ManifestError(f"duplicate case id {case_id!r}")
        seen.add(case_id)
        for artifact in case["expected"]["artifacts"]:
            relative = pathlib.PurePosixPath(artifact["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ManifestError(
                    f"{case_id}: artifact path {artifact['path']!r} escapes the case root"
                )

    return manifest


def select_cases(manifest: dict, requested: list[str] | None) -> list[dict]:
    if not requested:
        return list(manifest["cases"])
    by_id = {case["id"]: case for case in manifest["cases"]}
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ManifestError(f"no such case(s) in the manifest: {', '.join(unknown)}")
    return [by_id[case_id] for case_id in requested]


def describe(manifest: dict, cases: list[dict], output_root: pathlib.Path) -> None:
    """Print the work the dump half will do, so the plan is inspectable now."""
    greedy = [c for c in cases if c["oracle"]["parameters"]["position_temperature"] == 0.0]
    sampled = [c for c in cases if c["oracle"]["parameters"]["position_temperature"] != 0.0]
    clone = [c for c in cases if "reference" in c["input"]]

    print(f"manifest      {manifest['family']}/{manifest['variant']}")
    print(f"oracle        {manifest['reference']['implementation']}")
    print(f"              {manifest['reference']['dtype']} on {manifest['reference']['device']}")
    print(f"output root   {output_root}")
    print(
        f"cases         {len(cases)} selected "
        f"({len(greedy)} greedy, {len(sampled)} sampled, {len(clone)} clone)"
    )
    print(f"probe layers  {', '.join(str(n) for n in GENERATOR_PROBE_LAYERS)}")
    for case in cases:
        artifacts = len(case["expected"]["artifacts"])
        print(f"  {case['id']:<28} {artifacts:>2} artifacts -> {output_root / case['id']}")


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)

    try:
        manifest = load_manifest(arguments.manifest)
        cases = select_cases(manifest, arguments.case)
    except ManifestError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    output_root = arguments.output_root or pathlib.Path(manifest["case_artifact_root"])
    describe(manifest, cases, output_root)

    if arguments.validate_only:
        print("\nmanifest ok; no dump was requested")
        return 0

    print(f"\nerror: {NOT_IMPLEMENTED}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
