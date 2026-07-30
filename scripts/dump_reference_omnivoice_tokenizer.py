#!/usr/bin/env python3
"""Dump the OmniVoice text frontend and duration estimator as one JSON contract.

The C++ port reimplements two pieces of pure host arithmetic: the byte-level
BPE that turns a request into prompt token ids, and the ``RuleDurationEstimator``
that fixes the canvas length before the first forward. Neither needs a GPU, a
checkpoint tensor, or a synthesis run to check -- but both are exactly the kind
of code where a silent one-character divergence shifts every downstream token
index. This script captures the oracle's answers once, so the port's unit tests
compare against measured values rather than against a second reading of the
upstream source.

What lands in ``build/goldens/omnivoice/tokenizer/cases.json``:

1. ``pre_tokenizer_splits`` -- the exact piece list the HF pre-tokenizer emits
   for a fixed set of awkward strings. The pre-tokenizer regex is the part of
   the frontend most likely to diverge, and its output is checkable without
   any merge-table involvement.
2. ``token_ids`` -- ids for those same strings, plus, for every Golden Manifest
   case, the two prompt strings ``_prepare_inference_inputs`` actually builds:
   the style block and the wrapped text block. The wrapped block is recorded
   both plainly and through ``_tokenize_with_nonverbal_tags``, because the
   split-out-the-tag path is a separate contract.
3. ``duration`` -- ``calculate_total_weight`` and the full
   ``_estimate_target_tokens`` input/output row for every case, including the
   0.5/1.0/2.0 speed sweep for the two rate cases.
4. ``special_tokens`` -- the seven OmniVoice markers plus eos/pad, with the
   marker ids asserted against the pinned 151669-151675.

Everything upstream that can be imported is imported and called rather than
reimplemented. The two exceptions are the style-block and wrapped-text
f-strings, which live inside ``OmniVoice._prepare_inference_inputs`` and cannot
be reached without instantiating the model; they are transcribed here with the
source pinned in a comment, and ``docs/porting/families/omnivoice.md`` carries
the same layout.

Usage:

    uv run --project scripts/envs/omnivoice --locked python \\
      scripts/dump_reference_omnivoice_tokenizer.py \\
      --manifest tests/golden/omnivoice/omnivoice-0-6b.manifest.json \\
      --weights-dir models/omnivoice-0-6b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import urllib.request

MANIFEST_SCHEMA = "synthesize-golden-manifest-v1"
FAMILY = "omnivoice"
DUMP_SCHEMA = "synthesize-oracle-tokenizer-dump-v1"

# The seven OmniVoice markers, in the order docs/porting/families/omnivoice.md
# documents them, against the ids intake measured. A checkpoint whose added
# tokens moved is a different frontend, and every prompt id below would be
# wrong in a way no later stage would attribute to the tokenizer.
MARKER_TOKENS = (
    "<|denoise|>",
    "<|lang_start|>",
    "<|lang_end|>",
    "<|instruct_start|>",
    "<|instruct_end|>",
    "<|text_start|>",
    "<|text_end|>",
)
MARKER_IDS = tuple(range(151669, 151676))
# 151,643 base entries plus 33 added tokens; llm_config.vocab_size agrees.
EXPECTED_VOCAB_TOTAL = 151676
EXPECTED_VOCAB_BASE = 151643

# The no-reference duration anchor, verbatim from
# OmniVoice._estimate_target_tokens at the pinned revision. It is part of this
# family's contract, not an implementation detail: every auto-voice and
# voice-design canvas length is a proportion against these two numbers.
NOREF_ANCHOR_TEXT = "Nice to meet you."
NOREF_ANCHOR_TOKENS = 25

SAMPLES_PER_FRAME = 960
NATIVE_SAMPLE_RATE = 24000

# Strings chosen so each one can fail on its own: a piece-boundary rule that is
# wrong for exactly one of them still shows up.
PRE_TOKENIZER_STRINGS = (
    ("ascii", "Hello, world!"),
    ("contractions", "It's the model's output -- don't re-tokenise it."),
    ("digits", "In 2026 the model counted 1234567890 samples."),
    ("cjk", "欢迎使用语音合成引擎。"),
    ("kana", "音声合成へようこそ。"),
    ("punctuation", "Wait — really?! Yes; truly: it works... (mostly)."),
    ("leading_space", "   leading spaces"),
    ("trailing_space", "trailing spaces   "),
    ("newlines", "line one\nline two\r\nline three"),
    ("tabs", "tab\tseparated\tvalues"),
    ("nonverbal", "That is funny [laughter] but let me think."),
    ("mixed_script", "Mixed 中文 and English 123 混排"),
    ("empty", ""),
)


class ManifestError(Exception):
    """The manifest does not describe something this dumper could read."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--weights-dir", required=True, type=pathlib.Path)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Destination JSON (default: <case_artifact_root>/tokenizer/cases.json).",
    )
    parser.add_argument(
        "--reference-audio-dir",
        type=pathlib.Path,
        default=pathlib.Path("models/omnivoice-reference-audio"),
        help="Where the digest-pinned clone reference wav is cached (git-ignored).",
    )
    return parser.parse_args(argv)


def require(mapping, key, where: str):
    """Read a manifest key or explain which one is missing.

    A bare KeyError from deep inside a dump would name the key and nothing
    else. The dumper is run against edited manifest copies often enough that
    "which case, which field" is the part worth printing.
    """
    if not isinstance(mapping, dict):
        raise ManifestError(f"{where}: expected an object, found {type(mapping).__name__}")
    if key not in mapping:
        raise ManifestError(f"{where}: missing required key {key!r}")
    return mapping[key]


def load_manifest(path: pathlib.Path) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ManifestError(f"no manifest at {path}") from error
    except json.JSONDecodeError as error:
        raise ManifestError(f"{path} is not valid JSON: {error}") from error
    if require(manifest, "schema", str(path)) != MANIFEST_SCHEMA:
        raise ManifestError(f"{path}: unexpected manifest schema {manifest['schema']!r}")
    if require(manifest, "family", str(path)) != FAMILY:
        raise ManifestError(f"{path}: not a {FAMILY} manifest")
    cases = require(manifest, "cases", str(path))
    if not cases:
        raise ManifestError(f"{path} declares no cases")
    for case in cases:
        case_id = require(case, "id", f"{path}: case")
        where = f"{path}: case {case_id}"
        require(require(case, "input", where), "text", f"{where}.input")
        require(require(case, "request", where), "speaking_rate", f"{where}.request")
        parameters = require(require(case, "oracle", where), "parameters", f"{where}.oracle")
        for key in ("language", "instruct"):
            require(parameters, key, f"{where}.oracle.parameters")
    return manifest


def reference_digest(manifest: dict, locator: str) -> str:
    """The pinned sha256 for a reference artifact, read from the manifest.

    Upstream ships no reference audio at any pinned revision, so the clone
    reference is pinned by content instead. Taking the digest from the manifest
    rather than a constant here keeps one copy of that pin.
    """
    for artifact in manifest.get("source", {}).get("artifacts", []):
        if artifact.get("locator") == locator:
            digest = artifact.get("sha256")
            if not digest:
                raise ManifestError(f"manifest artifact {locator} carries no sha256")
            return digest
    raise ManifestError(f"manifest source.artifacts has no entry for {locator}")


def materialise_reference(locator: str, digest: str, directory: pathlib.Path) -> pathlib.Path:
    """Fetch the clone reference if absent, and refuse to use a wrong one."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / locator.rsplit("/", 1)[-1]
    if not destination.exists():
        print(f"fetching {locator}", flush=True)
        with urllib.request.urlopen(locator) as response:  # noqa: S310 - pinned https locator
            destination.write_bytes(response.read())
    actual = hashlib.sha256(destination.read_bytes()).hexdigest()
    if actual != digest:
        raise SystemExit(
            f"{destination}: sha256 {actual} does not match the manifest's {digest}. "
            "Refusing to derive a duration anchor from an unpinned reference."
        )
    return destination


def reference_audio_tokens(path: pathlib.Path) -> dict:
    """Reproduce create_voice_clone_prompt's arithmetic up to the token count.

    The estimator needs the reference token count, and the count is a property
    of the waveform rather than of the model: mono-mix, resample to the codec
    rate, the quiet-reference boost gate, then clip the tail to a whole number
    of hops. Every one of those steps is transcribed from
    ``OmniVoice.create_voice_clone_prompt``; the main dumper measures the same
    number from the real encode and the two are cross-checked there.
    """
    import numpy as np
    import soundfile
    import torch
    import torchaudio

    data, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    waveform = data.T
    if waveform.shape[0] > 1:
        waveform = np.mean(waveform, axis=0, keepdims=True)
    if sample_rate != NATIVE_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            torch.from_numpy(waveform),
            orig_freq=sample_rate,
            new_freq=NATIVE_SAMPLE_RATE,
        ).numpy()
    rms = float(np.sqrt(np.mean(waveform**2)))
    boosted = 0 < rms < 0.1
    if boosted:
        waveform = waveform * 0.1 / rms
    clip = int(waveform.shape[-1] % SAMPLES_PER_FRAME)
    clipped = waveform[:, :-clip] if clip > 0 else waveform
    return {
        "path": str(path),
        "source_sample_rate": int(sample_rate),
        "samples_at_24k": int(waveform.shape[-1]),
        "ref_rms": rms,
        "quiet_reference_boost_applied": boosted,
        "hop_clipped_samples": clip,
        "num_ref_audio_tokens": int(clipped.shape[-1] // SAMPLES_PER_FRAME),
    }


def build_estimator():
    """Bind upstream's ``_estimate_target_tokens`` without loading a checkpoint.

    The method reads only ``self.duration_estimator``, so a stand-in carrying
    that one attribute lets the dump call the real upstream code path -- anchor
    substitution, speed division and the ``max(1, int(...))`` truncation
    included -- instead of a local paraphrase of it.
    """
    from omnivoice.models.omnivoice import OmniVoice
    from omnivoice.utils.duration import RuleDurationEstimator

    class _EstimatorHost:
        duration_estimator = RuleDurationEstimator()

    host = _EstimatorHost()
    return host.duration_estimator, OmniVoice._estimate_target_tokens.__get__(host, _EstimatorHost)


def style_text_for(language, instruct, has_reference: bool, denoise: bool) -> str:
    """The style block, transcribed from ``_prepare_inference_inputs``.

    Source: omnivoice/models/omnivoice.py at 468e927b, inside
    ``_prepare_inference_inputs``. Absent fields are the literal four-character
    string ``None`` rather than an omission, and ``<|denoise|>`` appears only
    when reference audio tokens are present.
    """
    style = ""
    if denoise and has_reference:
        style += "<|denoise|>"
    style += f"<|lang_start|>{language if language else 'None'}<|lang_end|>"
    style += f"<|instruct_start|>{instruct if instruct else 'None'}<|instruct_end|>"
    return style


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        manifest = load_manifest(arguments.manifest)
    except ManifestError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if not arguments.weights_dir.is_dir():
        print(f"error: no weights directory at {arguments.weights_dir}", file=sys.stderr)
        return 1

    output = arguments.output or (
        pathlib.Path(manifest.get("case_artifact_root", "build/goldens/omnivoice"))
        / "tokenizer"
        / "cases.json"
    )

    from transformers import AutoTokenizer

    from omnivoice.models.omnivoice import (
        _combine_text,
        _resolve_instruct,
        _resolve_language,
        _tokenize_with_nonverbal_tags,
    )
    from omnivoice.utils.voice_design import _ZH_RE

    tokenizer = AutoTokenizer.from_pretrained(str(arguments.weights_dir))
    estimator, estimate_target_tokens = build_estimator()

    # --- Special tokens -------------------------------------------------
    marker_ids = [tokenizer.convert_tokens_to_ids(token) for token in MARKER_TOKENS]
    if tuple(marker_ids) != MARKER_IDS:
        raise SystemExit(
            "marker ids moved: expected "
            f"{list(MARKER_IDS)} for {list(MARKER_TOKENS)}, measured {marker_ids}"
        )
    if len(tokenizer) != EXPECTED_VOCAB_TOTAL or tokenizer.vocab_size != EXPECTED_VOCAB_BASE:
        raise SystemExit(
            f"vocabulary moved: {tokenizer.vocab_size} base + added = {len(tokenizer)}, "
            f"expected {EXPECTED_VOCAB_BASE} and {EXPECTED_VOCAB_TOTAL}"
        )
    if tokenizer.bos_token_id is not None:
        raise SystemExit(
            f"tokenizer declares a BOS token ({tokenizer.bos_token_id}); the family "
            "contract says no BOS is added on either side"
        )

    special_tokens = {
        "markers": [
            {"token": token, "id": int(identifier)}
            for token, identifier in zip(MARKER_TOKENS, marker_ids, strict=True)
        ],
        "eos": {"token": tokenizer.eos_token, "id": int(tokenizer.eos_token_id)},
        "pad": {"token": tokenizer.pad_token, "id": int(tokenizer.pad_token_id)},
        "bos": None,
        "vocab_base": int(tokenizer.vocab_size),
        "vocab_total": len(tokenizer),
    }

    # --- Pre-tokenizer splits ------------------------------------------
    pre_tokenizer = tokenizer._tokenizer.pre_tokenizer
    pre_tokenizer_splits = [
        {
            "label": label,
            "text": text,
            "pieces": [
                {"piece": piece, "start": int(span[0]), "end": int(span[1])}
                for piece, span in pre_tokenizer.pre_tokenize_str(text)
            ],
        }
        for label, text in PRE_TOKENIZER_STRINGS
    ]

    # --- Token ids for the fixed strings --------------------------------
    def ids_of(text: str) -> list[int]:
        without = tokenizer(text, add_special_tokens=False).input_ids
        # `_prepare_inference_inputs` tokenises the style block with the
        # tokeniser's default, which is add_special_tokens=True. Asserting the
        # two agree is what lets every id below be recorded once: if a future
        # checkpoint ever added a BOS, the prompt would gain a token the port
        # does not emit, and this is where that shows up.
        with_special = tokenizer(text).input_ids
        if without != with_special:
            raise SystemExit(
                f"add_special_tokens changes the ids for {text!r}: "
                f"{without} vs {with_special}"
            )
        return [int(i) for i in without]

    token_ids = [
        {"label": label, "kind": "fixed", "text": text, "ids": ids_of(text)}
        for label, text in PRE_TOKENIZER_STRINGS
    ]

    # --- Per-case prompt strings and duration rows ----------------------
    reference_cache: dict[str, dict] = {}
    prompt_strings = []
    duration_rows = []
    instruct_checks = []

    for case in manifest["cases"]:
        case_id = case["id"]
        parameters = case["oracle"]["parameters"]
        text = case["input"]["text"]
        language = parameters["language"]
        instruct = parameters["instruct"]
        reference = case["input"].get("reference")

        resolved_language = _resolve_language(language)
        if resolved_language != language:
            raise SystemExit(
                f"{case_id}: _resolve_language({language!r}) returned "
                f"{resolved_language!r}; the manifest pins already-resolved codes"
            )
        if instruct is not None:
            use_zh = bool(text and _ZH_RE.search(text))
            resolved_instruct = _resolve_instruct(instruct, use_zh=use_zh)
            if resolved_instruct != instruct:
                raise SystemExit(
                    f"{case_id}: _resolve_instruct({instruct!r}, use_zh={use_zh}) returned "
                    f"{resolved_instruct!r}; the manifest pins already-normalised instructs"
                )
            instruct_checks.append(
                {
                    "case": case_id,
                    "instruct": instruct,
                    "use_zh": use_zh,
                    "resolved": resolved_instruct,
                    "is_noop": True,
                }
            )

        ref_text = parameters.get("ref_text") if reference else None
        num_ref_tokens = None
        if reference:
            locator = require(reference, "artifact", f"case {case_id}: input.reference")
            if locator not in reference_cache:
                digest = reference_digest(manifest, locator)
                path = materialise_reference(locator, digest, arguments.reference_audio_dir)
                reference_cache[locator] = reference_audio_tokens(path) | {"sha256": digest}
            num_ref_tokens = reference_cache[locator]["num_ref_audio_tokens"]

        style_text = style_text_for(
            language,
            instruct,
            has_reference=reference is not None,
            denoise=bool(parameters.get("denoise", True)),
        )
        full_text = _combine_text(ref_text=ref_text, text=text)
        wrapped_text = f"<|text_start|>{full_text}<|text_end|>"
        nonverbal_ids = [
            int(i) for i in _tokenize_with_nonverbal_tags(wrapped_text, tokenizer)[0].tolist()
        ]
        plain_ids = ids_of(wrapped_text)
        prompt_strings.append(
            {
                "case": case_id,
                "style_text": style_text,
                "style_ids": ids_of(style_text),
                "combined_text": full_text,
                "wrapped_text": wrapped_text,
                # What `_prepare_inference_inputs` actually uses.
                "wrapped_ids": nonverbal_ids,
                # The same string straight through the tokeniser. Equal unless
                # the text carries a bracketed nonverbal tag, which is the whole
                # point of the split-out path.
                "wrapped_ids_without_tag_split": plain_ids,
                "tag_split_changes_ids": nonverbal_ids != plain_ids,
            }
        )
        token_ids.append(
            {"label": f"{case_id}.style", "kind": "case_style", "text": style_text,
             "ids": ids_of(style_text)}
        )
        token_ids.append(
            {"label": f"{case_id}.wrapped", "kind": "case_wrapped", "text": wrapped_text,
             "ids": nonverbal_ids}
        )

        # The rate cases get the whole sweep so the port can check the division
        # and the max(1, int(...)) truncation at both declared ends, not only at
        # the speed the case happens to run.
        speeds = [case["request"]["speaking_rate"]]
        if "duration-scaling" in case.get("coverage", []):
            speeds = [0.5, 1.0, 2.0]
        for speed in speeds:
            duration_rows.append(
                {
                    "case": case_id,
                    "text": text,
                    "total_weight": estimator.calculate_total_weight(text),
                    "ref_text": ref_text if ref_text else NOREF_ANCHOR_TEXT,
                    "ref_text_is_anchor": ref_text is None,
                    "ref_text_total_weight": estimator.calculate_total_weight(
                        ref_text if ref_text else NOREF_ANCHOR_TEXT
                    ),
                    "num_ref_audio_tokens": num_ref_tokens
                    if num_ref_tokens is not None
                    else NOREF_ANCHOR_TOKENS,
                    "speed": speed,
                    "estimated_frames": int(
                        estimate_target_tokens(text, ref_text, num_ref_tokens, speed=speed)
                    ),
                }
            )

    payload = {
        "schema": DUMP_SCHEMA,
        "family": FAMILY,
        "variant": manifest.get("variant"),
        "suite_version": manifest.get("suite_version"),
        "manifest": str(arguments.manifest),
        "weights_dir": str(arguments.weights_dir),
        "tokenizer_class": type(tokenizer).__name__,
        "add_special_tokens_is_noop": True,
        "special_tokens": special_tokens,
        "duration_anchor": {
            "text": NOREF_ANCHOR_TEXT,
            "num_audio_tokens": NOREF_ANCHOR_TOKENS,
            "total_weight": estimator.calculate_total_weight(NOREF_ANCHOR_TEXT),
        },
        "clone_references": list(reference_cache.values()),
        "instruct_resolver_checks": instruct_checks,
        "pre_tokenizer_splits": pre_tokenizer_splits,
        "token_ids": token_ids,
        "prompt_strings": prompt_strings,
        "duration": duration_rows,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"wrote {output}: {len(pre_tokenizer_splits)} pre-tokenizer strings, "
        f"{len(token_ids)} id rows, {len(prompt_strings)} prompts, "
        f"{len(duration_rows)} duration rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
