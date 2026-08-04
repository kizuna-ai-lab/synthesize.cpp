#!/usr/bin/env python3
"""The local inputs a conversion or an oracle dump must digest-verify first.

One pinned set, two consumers (converter, oracle dumper); Plan 3 carryover
item 4. `scripts/convert-omnivoice.py` used to carry its own six-entry list
and `scripts/dump_reference_omnivoice_pytorch.py` inferred a private idea of
the same set from "/resolve/" markers in the manifest's locator URLs -- two
lists nothing ever checked against each other, so a drift between them would
have gone unnoticed until a converted package and a dumped baseline quietly
came from different weights. `PINNED_INPUTS` below is the converter's six
entries verbatim (the richer of the two lists), and both scripts now walk it.

Stdlib only. `dump_reference_omnivoice_pytorch.py` is imported directly by
`tests/python/test_dump_reference_omnivoice_pytorch.py` via a plain
`sys.path.insert` + `import`, with no package install and no locked
environment, so nothing in this module may require a dependency outside the
standard library.
"""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path

PinnedInput = namedtuple("PinnedInput", "role relative_path sha256_key")

# `role` is the manifest artifact role that names the input, and is also what
# tells apart the two "checkpoint" entries and the two "config" entries below
# (locator suffix matching alone cannot: "audio_tokenizer/model.safetensors"
# ends in "/model.safetensors" too). `relative_path` is where the file lives
# under a weights directory, and also what a HuggingFace
# ".../resolve/<revision>/<relative_path>" locator resolves to once the
# revision segment is dropped -- the technique the oracle dumper already used
# and the one both consumers now share. `sha256_key` is the label the verified
# digest is filed under: converter report fields and `digests[...]` lookups in
# both scripts.
PINNED_INPUTS: list[PinnedInput] = [
    PinnedInput("checkpoint", "model.safetensors", "generator"),
    PinnedInput("checkpoint", "audio_tokenizer/model.safetensors", "codec"),
    PinnedInput("config", "config.json", "config"),
    PinnedInput("config", "audio_tokenizer/config.json", "codec_config"),
    PinnedInput("frontend-resource", "tokenizer.json", "tokenizer"),
    PinnedInput("license", "audio_tokenizer/LICENSE", "codec_license"),
]


def resolve_local(weights_dir: Path, pin: PinnedInput) -> Path:
    """Where `pin` lives under a materialised weights directory."""
    return weights_dir / pin.relative_path


# A HuggingFace resolve URL names a file's path in the weights repository
# after the revision segment: ".../resolve/<revision>/<relative_path>".
# `convert-omnivoice.py` and `dump_reference_omnivoice_pytorch.py` both used
# to define this marker and split on it themselves (Plan 3 close-out,
# task-17 triage) -- one copy each, nothing checking the two agreed.
RESOLVE_MARKER = "/resolve/"


def relative_path_from_locator(locator: str) -> str | None:
    """The weights-repository relative path a HuggingFace resolve `locator` names.

    Drops everything up to and including the revision segment. Returns
    `None` if `locator` carries no "/resolve/" marker at all (a
    source-repository artifact, e.g. the Apache LICENSE, or anything else
    that is not a weights-repository resolve URL), or if the marker is
    present but nothing follows the revision segment -- a truncated locator
    that names no file. Both callers treat `None` the same as "does not
    match this pin": a truncated locator now surfaces as zero matches
    through the same `ConverterError`/`SystemExit` path a wrong or missing
    locator already takes, rather than as a raw `IndexError` from indexing
    past a one-element `split("/", 1)` result.
    """
    if RESOLVE_MARKER not in locator:
        return None
    remainder = locator.split(RESOLVE_MARKER, 1)[1]
    if "/" not in remainder:
        return None
    return remainder.split("/", 1)[1]
