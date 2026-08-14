#!/usr/bin/env python3
"""Dump the Qwen3-TTS-Base ICL prompt's two tracks, its alignment branch, and its
trailing schedule.

``scripts/dump_reference_qwen3_tts_base.py`` already captures ``prompt/icl_embed.f32``
-- the finished block ``generate_icl_prompt`` hands back. That block is the
*elementwise sum* of two tracks, and a sum is a lossy record of an alignment: a
text track one position early plus a codec track one position late is a
different sum, but so are a hundred other pairings, and the sum alone says only
that something moved. ``src/arch/qwen3-tts/talker-host.h`` (the comment above
``build_talker_prompt``) already writes the consequence down for the non-ICL
prompt: "A prompt off by one still synthesizes speech, in the wrong voice or the
wrong language." Nothing a listener hears, and no end-to-end audio tolerance,
catches it -- the talker samples, so the audio was never going to match
sample-for-sample anyway. This script is what stands between a wrong alignment
and a shipped defect, so it dumps the two summands separately, records which
alignment arm upstream actually took, and dumps the trailing schedule that arm
produced.

**The branch is not recoverable from any artifact that already exists.**
Upstream's rule (``generate_icl_prompt``, ``non_streaming_mode=False``) has two
arms: when ``T1 > T2`` the text track is truncated to ``T2`` and its tail becomes
the trailing schedule; otherwise the text track is padded up to ``T2`` and the
trailing schedule is a bare ``tts_pad_embed``. Both arms emit a summed block of
exactly ``T2`` positions, so ``icl_embed.f32``'s size is silent about which one
ran, and the discriminator -- the trailing schedule -- was recorded nowhere.
``alignment.json`` is that record, and it is *observed*, never recomputed: the
arm is read out of the line number of the ``return`` statement that actually
executed, and cross-checked against two independent signals (whether the
returned trailing schedule *is* the ``tts_pad_embed`` object upstream was handed,
and how long the ``text_embed`` local is at return). All three must agree or the
run aborts.

**Every convention this script records is extracted from the installed source at
run time, never transcribed by hand.** ``prompt_conventions.json`` carries
verbatim source spans with the line numbers they were found at, located by
walking the AST of the live ``qwen_tts`` package rather than by hardcoded
offsets, so a citation cannot go stale: if upstream moves or reshapes a
construct, the locator fails loudly instead of emitting a line number that
points at something else. Where a convention can be *executed* rather than
asserted, it is: the text and codec tracks are rebuilt by calling upstream's own
modules and compared bitwise against the captured tensors; the reference and
target token id slices are rebuilt by calling upstream's own turn builders and
tokenizer; the x-vector's survival into the ICL prompt is proved by finding the
position in the assembled prompt that equals ``tts_pad_embed + speaker_embed``
bitwise, not by reading the ``if speaker_embed is None`` branch and believing it.

Every gate runs before the first byte is written, and the final gate re-reads
the bytes from disk; a run that fails leaves no consumable artifacts behind.

**This script never samples.** ``generate_icl_prompt`` runs during prompt
assembly, before the talker generates anything, so the dump does not depend on
the seed, the sampler, or ``max_new_tokens``. The script lets upstream assemble
the whole prompt and then halts at ``talker.generate`` with a sentinel that
derives from ``BaseException`` (so no ``except Exception`` upstream can swallow
it). Halting there rather than inside ``generate_icl_prompt`` is deliberate: the
sentinel's own call captures the finished prompt and the batched trailing
schedule, which is what makes the placement claim checkable instead of merely
transcribed.

Usage (explicit-argument form):

    uv run --project scripts/envs/qwen3-tts --locked python \\
      scripts/dump_reference_qwen3_tts_icl_prompt.py \\
      --weights-dir models/qwen3-tts-12hz-0-6b-base \\
      --ref-audio models/qwen3-tts-reference-audio/clone.wav \\
      --ref-text "Okay. Yeah. I resent you. ..." \\
      --text "This is a test of Qwen three T T S base voice cloning." \\
      --language English \\
      --out-dir build/goldens/qwen3-tts/qwen3-tts-12hz-0-6b-base/base-icl-en

Manifest form (the normal one):

    uv run --project scripts/envs/qwen3-tts --locked python \\
      scripts/dump_reference_qwen3_tts_icl_prompt.py \\
      --weights-dir models/qwen3-tts-12hz-0-6b-base \\
      --manifest tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-base.manifest.json \\
      --case base-icl-en

x-vector-only cases have no ICL prompt at all and are refused rather than
skipped -- the mirror image of the assertion in
``dump_reference_qwen3_tts_base.py`` that an x-vector case must *not* build one.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
import pathlib
import sys
import textwrap
import time
import urllib.request
from typing import Any, Optional

import librosa
import numpy as np
import torch

MANIFEST_SCHEMA = "synthesize-golden-manifest-v1"
FAMILY = "qwen3-tts"
CONVENTIONS_SCHEMA = "synthesize-qwen3-tts-icl-prompt-conventions-v1"
ALIGNMENT_SCHEMA = "synthesize-qwen3-tts-icl-prompt-alignment-v1"

# The two arms of `generate_icl_prompt`'s `non_streaming_mode=False` path, named
# the way the plan and every downstream task name them. "truncate" is the
# `T1 > T2` arm; "pad" is the other one. `stream` is the `non_streaming_mode=True`
# path, which `generate_voice_clone` never takes by default -- if it ever fires
# here the run aborts, because the artifacts would describe a different layout.
ARM_TRUNCATE = "truncate"
ARM_PAD = "pad"
ARM_STREAM = "stream"


class _HaltBeforeSampling(BaseException):
    """Raised to stop upstream once the prompt is assembled.

    Derives from ``BaseException`` on purpose: a bare ``except Exception``
    anywhere in the upstream call chain would otherwise turn a deliberate halt
    into a silent partial run.
    """


# --------------------------------------------------------------------------
# artifact writers (kept local, as in every sibling dump script in this family)
# --------------------------------------------------------------------------

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
    """Detach and move a possibly-GPU, possibly-bf16 tensor to a plain host array."""
    if torch.is_tensor(value):
        tensor = value.detach()
        if dtype is not None:
            tensor = tensor.to(dtype)
        return tensor.cpu().numpy()
    return np.asarray(value)


# --------------------------------------------------------------------------
# source location: every citation this script emits is found, not remembered
# --------------------------------------------------------------------------

def installed_upstream() -> dict:
    """Read the pinned qwen-tts repository and commit out of the installed wheel.

    uv records a VCS install's resolved commit in the distribution's
    ``direct_url.json`` (PEP 610), so the revision the artifacts claim is the
    revision that actually produced them. A missing or non-VCS record is a hard
    failure rather than a fallback to a remembered string: an artifact naming the
    wrong upstream commit is worse than one that was never written.
    """
    try:
        raw = importlib.metadata.distribution("qwen-tts").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError as error:  # pragma: no cover - env guard
        raise SystemExit(f"qwen-tts is not installed in this environment: {error}")
    if raw is None:
        raise SystemExit(
            "the installed qwen-tts has no direct_url.json, so its upstream commit cannot be "
            "read; the artifacts would have to name a revision nothing verified"
        )
    record = json.loads(raw)
    vcs = record.get("vcs_info") or {}
    if not vcs.get("commit_id"):
        raise SystemExit(
            f"the installed qwen-tts was not installed from a VCS ({record.get('url')!r}), so "
            "there is no commit to record"
        )
    return {"repository": record["url"], "revision": vcs["commit_id"]}


def function_ast(func) -> tuple[pathlib.Path, ast.FunctionDef]:
    """Parse one function out of the *installed* package, in file line numbers.

    ``inspect.getsourcelines`` hands back the function's own lines and the line
    the definition starts on; ``ast.increment_lineno`` shifts the parsed tree
    back into file coordinates so every ``lineno`` this module reports can be
    opened in an editor and land on the construct it names.

    The unwrapping is load-bearing. ``getsourcelines`` follows ``__wrapped__``
    but ``getsourcefile`` does not, so asking both about a decorated method --
    ``generate`` carries ``@torch.no_grad()`` -- pairs upstream's line numbers
    with ``torch/utils/_contextlib.py``, and every citation lands in the wrong
    file.
    """
    func = inspect.unwrap(getattr(func, "__func__", func))
    path = pathlib.Path(inspect.getsourcefile(func))
    lines, start = inspect.getsourcelines(func)
    tree = ast.parse(textwrap.dedent("".join(lines)))
    ast.increment_lineno(tree, start - 1)
    node = tree.body[0]
    if not isinstance(node, ast.FunctionDef):
        raise SystemExit(f"{path}: {func!r} did not parse as a plain function definition")
    return path, node


def span_source(path: pathlib.Path, first: int, last: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not (1 <= first <= last <= len(lines)):
        raise SystemExit(f"{path}: line span {first}-{last} is outside the file")
    return "\n".join(lines[first - 1:last])


def cite(path: pathlib.Path, first: int, last: int, claim: str, verified_by: str) -> dict:
    """One transcription entry: the claim, where it lives, and its verbatim text."""
    return {
        "claim": claim,
        "file": path.name,
        "lines": [int(first), int(last)],
        "citation": f"{path.name}:{first}-{last}",
        "source": span_source(path, first, last),
        "verified_by": verified_by,
    }


def statement_span(nodes: list[ast.stmt]) -> tuple[int, int]:
    return min(n.lineno for n in nodes), max(n.end_lineno for n in nodes)


def enclosing_body(root: ast.AST, first: ast.stmt, second: ast.stmt):
    """Find the statement list that holds both nodes, and their indices in it."""
    for node in ast.walk(root):
        for field in ("body", "orelse", "finalbody"):
            seq = getattr(node, field, None)
            if isinstance(seq, list) and first in seq and second in seq:
                return seq, seq.index(first), seq.index(second)
    return None, None, None


def second_dim_slice(node: ast.AST) -> tuple[Optional[int], Optional[int]]:
    """Read the ``[:, lo:hi]`` bounds off a subscript, e.g. ``input_id[:, 3:-5]``."""
    if not isinstance(node, ast.Subscript) or not isinstance(node.slice, ast.Tuple):
        raise SystemExit(f"expected a `[:, lo:hi]` subscript, got {ast.dump(node)[:200]}")
    dims = node.slice.elts
    if len(dims) != 2 or not isinstance(dims[1], ast.Slice):
        raise SystemExit(f"expected two subscript dimensions ending in a slice, got {len(dims)}")

    def literal(value: Optional[ast.expr]) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, ast.Constant) and isinstance(value.value, int):
            return int(value.value)
        if (isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub)
                and isinstance(value.operand, ast.Constant)):
            return -int(value.operand.value)
        raise SystemExit(f"slice bound is not an integer literal: {ast.dump(value)[:200]}")

    return literal(dims[1].lower), literal(dims[1].upper)


def describe_icl_prompt_source(func) -> dict:
    """Locate every construct inside ``generate_icl_prompt`` and map arm -> return lines.

    The arm map is the load-bearing part: it is what turns the line number of the
    ``return`` that actually executed into the name of the arm that executed,
    without ever comparing ``T1`` to ``T2`` in this script. The structural
    assertions along the way are the transcription's guard -- if upstream ever
    changes the discriminating test, this raises rather than mislabelling a
    branch.
    """
    path, fn = function_ast(func)

    body = list(fn.body)
    text_stmts: list[ast.stmt] = []
    index = 0
    while index < len(body) and isinstance(body[index], ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "text_embed" for t in body[index].targets
    ):
        text_stmts.append(body[index])
        index += 1
    if not text_stmts:
        raise SystemExit(f"{path}: {fn.name} no longer opens by assigning `text_embed`")

    codec_stmts: list[ast.stmt] = []
    while index < len(body) and not (
        isinstance(body[index], ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "text_lens" for t in body[index].targets)
    ):
        codec_stmts.append(body[index])
        index += 1
    if not codec_stmts:
        raise SystemExit(f"{path}: {fn.name} has no codec-track construction before `text_lens`")

    lens_stmts = [body[index], body[index + 1]]
    index += 2

    outer = body[index]
    if not (isinstance(outer, ast.If) and isinstance(outer.test, ast.Name)
            and outer.test.id == "non_streaming_mode"):
        raise SystemExit(f"{path}: {fn.name} no longer branches on `non_streaming_mode`")
    inner = outer.orelse[0] if outer.orelse else None
    if not isinstance(inner, ast.If):
        raise SystemExit(f"{path}: {fn.name}'s non-streaming path is no longer an `if`")

    test = inner.test
    ok = (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1 and isinstance(test.ops[0], ast.Gt)
        and isinstance(test.left, ast.Name) and test.left.id == "text_lens"
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Name) and test.comparators[0].id == "codec_lens"
    )
    if not ok:
        raise SystemExit(
            f"{path}:{inner.lineno}: the alignment branch no longer tests "
            f"`text_lens > codec_lens` but `{ast.unparse(test)}`. The plan's arm names "
            "('truncate' when T1 > T2, 'pad' otherwise) describe a rule that has changed; "
            "re-derive the arms before dumping anything."
        )

    def returns(statements: list[ast.stmt]) -> list[ast.Return]:
        found: list[ast.Return] = []
        for statement in statements:
            for node in ast.walk(statement):
                if isinstance(node, ast.Return):
                    found.append(node)
        return found

    arm_returns = {
        ARM_TRUNCATE: returns(inner.body),
        ARM_PAD: returns(inner.orelse),
        ARM_STREAM: returns(outer.body),
    }
    for arm, nodes in arm_returns.items():
        if not nodes:
            raise SystemExit(f"{path}: the {arm!r} arm of {fn.name} has no return statement")

    return {
        "path": path,
        "function": fn,
        "arm_line_ranges": {
            arm: [[int(n.lineno), int(n.end_lineno)] for n in nodes]
            for arm, nodes in arm_returns.items()
        },
        "spans": {
            "text_track": statement_span(text_stmts),
            "codec_track": statement_span(codec_stmts),
            "lens": statement_span(lens_stmts),
            "alignment": (inner.lineno, inner.end_lineno),
        },
    }


def describe_generate_source(func) -> dict:
    """Locate the ICL call site, the placement, and the speaker-embedding slot."""
    path, fn = function_ast(func)

    call_if = call_index = call_stmt = None
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        for position, statement in enumerate(node.body):
            if (isinstance(statement, ast.Assign)
                    and isinstance(statement.value, ast.Call)
                    and isinstance(statement.value.func, ast.Attribute)
                    and statement.value.func.attr == "generate_icl_prompt"):
                call_if, call_index, call_stmt = node, position, statement
                break
        if call_stmt is not None:
            break
    if call_stmt is None:
        raise SystemExit(f"{path}: {fn.name} no longer calls generate_icl_prompt")

    if call_index + 1 >= len(call_if.body):
        raise SystemExit(f"{path}: nothing follows the generate_icl_prompt call in its branch")
    placement = call_if.body[call_index + 1]
    placed_ok = (
        isinstance(placement, ast.Assign)
        and len(placement.targets) == 1
        and isinstance(placement.targets[0], ast.Name)
        and placement.targets[0].id == "talker_input_embed"
        and isinstance(placement.value, ast.Call)
        and isinstance(placement.value.func, ast.Attribute)
        and placement.value.func.attr == "cat"
        and placement.value.args
        and isinstance(placement.value.args[0], ast.List)
        and [getattr(e, "id", None) for e in placement.value.args[0].elts]
        == ["talker_input_embed", "icl_input_embed"]
    )
    if not placed_ok:
        raise SystemExit(
            f"{path}:{placement.lineno}: the ICL block is no longer appended to the prefix "
            f"as `cat([talker_input_embed, icl_input_embed])` but as "
            f"`{ast.unparse(placement)[:160]}`"
        )

    keywords = {kw.arg: kw.value for kw in call_stmt.value.keywords}
    for name in ("text_id", "ref_id"):
        if name not in keywords:
            raise SystemExit(f"{path}: generate_icl_prompt is no longer passed {name!r} by keyword")
    text_slice = second_dim_slice(keywords["text_id"])
    ref_slice = second_dim_slice(keywords["ref_id"])

    speaker_if = None
    for node in ast.walk(fn):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name) and node.test.left.id == "speaker_embed"
                and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Is)):
            speaker_if = node
            break
    if speaker_if is None:
        raise SystemExit(f"{path}: {fn.name} no longer branches on `speaker_embed is None`")

    body, speaker_position, icl_position = enclosing_body(fn, speaker_if, call_if)
    if body is None:
        raise SystemExit(
            f"{path}: the speaker-embedding branch and the ICL branch are no longer siblings; "
            "the claim that ICL adds to the x-vector path rather than replacing it depends on it"
        )
    if not speaker_position < icl_position:
        raise SystemExit(
            f"{path}: the speaker-embedding branch no longer precedes the ICL branch"
        )

    non_icl = call_if.orelse
    if not non_icl:
        raise SystemExit(f"{path}: the ICL conditional has no non-ICL branch to compare against")

    return {
        "path": path,
        "spans": {
            "call": (call_stmt.lineno, call_stmt.end_lineno),
            "placement": (placement.lineno, placement.end_lineno),
            "speaker_slot": (speaker_if.lineno, speaker_if.end_lineno),
            "non_icl": statement_span([non_icl[0]]),
        },
        "text_id_slice": text_slice,
        "ref_id_slice": ref_slice,
    }


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------

def capture_icl_prompt(model, sink: dict, arm_line_ranges: dict):
    """Wrap ``generate_icl_prompt`` and read its frame at the moment it returns.

    ``generate_icl_prompt`` is a plain method with nothing to hang a forward hook
    on, so ``dump_reference_qwen3_tts_base.py`` wraps it for the duration of one
    call; this reuses that mechanism and adds the one thing a wrapper alone
    cannot give: the function's *locals*, and the line of the ``return`` that
    produced them. The two summed tracks are locals, never returned, and the arm
    that ran is not a value at all -- it is control flow. A trace function
    installed for exactly this one call reads both without reimplementing a line
    of upstream arithmetic.

    The trace function is installed inside the wrapper rather than around the
    whole synthesis call, so only ``generate_icl_prompt``'s own frame and its
    callees are traced.
    """
    target = model.model
    original = target.generate_icl_prompt
    code = original.__func__.__code__
    if code.co_name != "generate_icl_prompt":
        raise SystemExit(
            "generate_icl_prompt is wrapped by a decorator "
            f"(its code object is {code.co_name!r} at {code.co_filename}); the frame this "
            "script reads locals from would be the decorator's, not upstream's"
        )

    def wrapper(*args, **kwargs):
        captured: dict = {}

        def local_trace(frame, event, arg):
            if event == "return":
                captured["locals"] = dict(frame.f_locals)
                captured["return_line"] = int(frame.f_lineno)
            return local_trace

        def global_trace(frame, event, arg):
            if event == "call" and frame.f_code is code:
                return local_trace
            return None

        previous = sys.gettrace()
        sys.settrace(global_trace)
        try:
            icl_input_embed, trailing_text_hidden = original(*args, **kwargs)
        finally:
            sys.settrace(previous)

        if "sink" not in sink:
            if "locals" not in captured:
                raise SystemExit(
                    "generate_icl_prompt returned without the trace function seeing its frame; "
                    "the two tracks and the executed arm cannot be observed. (A debugger or "
                    "coverage tool holding sys.settrace is the usual cause.)"
                )
            frame_locals = captured["locals"]
            line = captured["return_line"]
            arm = None
            for name, ranges in arm_line_ranges.items():
                if any(low <= line <= high for low, high in ranges):
                    arm = name
                    break
            if arm is None:
                raise SystemExit(
                    f"generate_icl_prompt returned from line {line}, which is not inside any "
                    f"return statement this script located ({arm_line_ranges}); the arm that "
                    "executed cannot be named, so nothing is dumped"
                )
            sink["sink"] = True
            sink["arm_from_return_line"] = arm
            sink["return_line"] = line
            sink["locals"] = frame_locals
            sink["icl_input_embed"] = icl_input_embed
            sink["trailing_text_hidden"] = trailing_text_hidden
            sink["args"] = kwargs
        return icl_input_embed, trailing_text_hidden

    target.generate_icl_prompt = wrapper
    return lambda: setattr(target, "generate_icl_prompt", original)


def halt_before_sampling(model, sink: dict):
    """Capture the assembled prompt at ``talker.generate`` and stop there.

    This is the last thing upstream does with the prompt before it starts
    sampling, so its keyword arguments are the finished article: ``inputs_embeds``
    is the whole prefill with the ICL block already concatenated onto it, and
    ``trailing_text_hidden`` is the trailing schedule after upstream's own batch
    padding. Capturing here is what makes "the block is concatenated after the
    existing prefix" a checkable statement rather than a transcribed one.
    """
    talker = model.model.talker
    original = talker.generate

    def wrapper(*args, **kwargs):
        sink["talker_kwargs"] = kwargs
        raise _HaltBeforeSampling()

    talker.generate = wrapper
    return lambda: setattr(talker, "generate", original)


# --------------------------------------------------------------------------
# case plumbing (mirrors the sibling dump scripts; ICL-only fields)
# --------------------------------------------------------------------------

@dataclasses.dataclass
class RunCase:
    id: str
    ref_audio: str
    ref_text: str
    text: str
    language: str
    trim_seconds: Optional[float] = None
    ref_sha256: Optional[str] = None


def build_case_from_args(args: argparse.Namespace) -> tuple[RunCase, pathlib.Path]:
    missing = [
        name for name, value in (
            ("--ref-audio", args.ref_audio),
            ("--ref-text", args.ref_text),
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
    case = RunCase(
        id=args.out_dir.name,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        text=args.text,
        language=args.language,
        trim_seconds=args.trim_seconds,
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
        raise SystemExit(f"unknown case id(s): {sorted(wanted - {c['id'] for c in cases})}")

    specs: list[tuple[RunCase, pathlib.Path]] = []
    for case in cases:
        params = case.get("oracle", {}).get("parameters", {})
        reference = case.get("input", {}).get("reference", {})
        ref_audio = reference.get("artifact") or params.get("ref_audio")
        ref_text = params.get("ref_text") or reference.get("transcript")
        if params.get("x_vector_only", False):
            if wanted is None:
                continue
            raise SystemExit(
                f"{case['id']}: x-vector-only cases build no ICL prompt at all, so there is "
                "nothing here to dump. Select an ICL case instead."
            )
        if ref_audio is None or ref_text is None:
            raise SystemExit(
                f"{case['id']}: an ICL case needs both input.reference.artifact and a "
                "reference transcript (oracle.parameters.ref_text)"
            )
        ref_sha256 = None
        for artifact in manifest.get("source", {}).get("artifacts", []):
            if artifact.get("role") == "reference-audio" and artifact.get("locator") == ref_audio:
                ref_sha256 = artifact.get("sha256")
                break
        specs.append((
            RunCase(
                id=case["id"],
                ref_audio=ref_audio,
                ref_text=ref_text,
                text=case["input"]["text"],
                language=params.get("language", "Auto"),
                trim_seconds=params.get("trim_seconds"),
                ref_sha256=ref_sha256,
            ),
            output_root / case["id"],
        ))
    if not specs:
        raise SystemExit(f"{args.manifest}: no ICL cases selected")
    return specs


def resolve_reference_locator(
    locator: str, expected_sha256: Optional[str], cache_dir: pathlib.Path
) -> str:
    """Fetch an http(s) reference-audio locator to a local cache, verified by digest.

    The digest is REQUIRED for a remote locator rather than checked when one
    happens to be present. A missing digest used to mean the file was downloaded
    and the comparison skipped entirely, while the refusal sentence below went on
    naming a protection that had not run -- the failure mode is a reference clip
    changing under a stable URL, which is exactly the case no check covered.

    LATENT, NOT LIVE. Verified against
    tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-base.manifest.json: all 12 cases
    name the one reference-audio locator
    https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav, and
    that artifact carries a sha256, so every case in the tree today already took
    the verified branch and none of them reaches the new exit.
    """
    if not (locator.startswith("http://") or locator.startswith("https://")):
        return locator
    if expected_sha256 is None:
        raise SystemExit(
            f"{locator} is remote and carries no pinned sha256, so nothing could say what "
            "was downloaded. Add a `sha256` to the manifest's source.artifacts entry whose "
            'role is "reference-audio" and whose locator is this URL. (The explicit '
            "--ref-audio form has no manifest to read a digest from; pass a local path "
            "there.) Refusing to dump an ICL prompt against an unpinned reference."
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / locator.rsplit("/", 1)[-1]
    if not destination.exists():
        print(f"fetching {locator}", flush=True)
        with urllib.request.urlopen(locator, timeout=60) as response:  # noqa: S310
            destination.write_bytes(response.read())
    actual = hashlib.sha256(destination.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise SystemExit(
            f"{destination}: sha256 {actual} does not match the manifest's "
            f"{expected_sha256}. Refusing to dump an ICL prompt against an unpinned "
            "reference. If a stale cached file is the cause, delete it and re-run."
        )
    return str(destination)


def load_reference_audio(source: str, trim_seconds: Optional[float]) -> tuple[np.ndarray, int, dict]:
    """Load and optionally re-length the reference clip, exactly as the Base dumper does.

    The trimming rule is copied deliberately rather than approximated: the
    reference frame count -- and therefore ``T2``, and therefore which alignment
    arm runs -- is a direct function of the clip length this produces, so a
    different rule here would dump the alignment of a case that does not exist.
    """
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
            repeats = -(-target // wav.shape[0])
            wav = np.tile(wav, repeats)[:target]
            info["looped"] = True
            info["loop_repeats"] = int(repeats)
        info["requested_seconds"] = trim_seconds
        info["applied_samples"] = int(wav.shape[0])
    return wav, sr, info


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------

def gate(checks: list[dict], name: str, ok: bool, detail: str) -> None:
    """Record a check and abort on failure, before anything is written.

    ``detail`` is the sentence printed when the check *fails*, so it is recorded
    under ``on_failure`` rather than as a bare ``detail``. Storing a failure
    sentence next to ``"ok": true`` reads as an assertion that the thing failed:
    the sum gate's message says the tracks "do not reproduce the block", which is
    the opposite of what a passing run means. Downstream tasks are written from
    these files, so the key has to say which outcome the sentence describes.
    """
    checks.append({"check": name, "ok": bool(ok), "on_failure": detail})
    if not ok:
        raise SystemExit(f"{name}: {detail}")


def rebuild_text_track(talker, ref_id, text_id, tts_eos_embed):
    """Re-run upstream's own two text-track statements, through upstream's own modules."""
    embed = talker.text_projection(talker.get_text_embeddings()(torch.cat([ref_id, text_id], dim=-1)))
    return torch.cat([embed, tts_eos_embed], dim=1)


def rebuild_codec_track(talker, config, ref_code, id_dtype):
    """Re-run upstream's own codec-track loop, through upstream's own modules."""
    groups = []
    for i in range(talker.config.num_code_groups):
        if i == 0:
            groups.append(talker.get_input_embeddings()(ref_code[:, :1]))
        else:
            groups.append(talker.code_predictor.get_input_embeddings()[i - 1](ref_code[:, i:i + 1]))
    stacked = torch.cat(groups, dim=1).sum(1).unsqueeze(0)
    bos = talker.get_input_embeddings()(
        torch.tensor([[config.talker_config.codec_bos_id]], device=talker.device, dtype=id_dtype)
    )
    return torch.cat([bos, stacked], dim=1)


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

@torch.inference_mode()
def run_case(model, case: RunCase, case_dir: pathlib.Path, reference_audio_dir: pathlib.Path,
             icl_source: dict, generate_source: dict) -> dict:
    # Under inference_mode because the checks below re-run upstream's own embedding
    # modules on tensors upstream created inside its own `@torch.inference_mode()`
    # (`create_voice_clone_prompt`), and an inference tensor cannot be fed to an
    # autograd-tracked op.
    started = time.time()
    checks: list[dict] = []

    local_ref_audio = resolve_reference_locator(case.ref_audio, case.ref_sha256, reference_audio_dir)
    ref_wav, ref_sr, ref_info = load_reference_audio(local_ref_audio, case.trim_seconds)

    prompt_items = model.create_voice_clone_prompt(
        ref_audio=(ref_wav, ref_sr), ref_text=case.ref_text, x_vector_only_mode=False
    )

    sink: dict = {}
    restore_icl = capture_icl_prompt(model, sink, icl_source["arm_line_ranges"])
    restore_halt = halt_before_sampling(model, sink)
    try:
        model.generate_voice_clone(
            text=case.text, language=case.language, voice_clone_prompt=prompt_items
        )
    except _HaltBeforeSampling:
        pass
    finally:
        restore_icl()
        restore_halt()

    if "sink" not in sink:
        raise SystemExit(
            f"{case.id}: generate_icl_prompt was never called, so this case has no ICL prompt. "
            "An x-vector-only case reaches synthesis without one; this script only dumps ICL."
        )
    if "talker_kwargs" not in sink:
        raise SystemExit(f"{case.id}: prompt assembly never reached talker.generate")

    frame_locals = sink["locals"]
    call_kwargs = sink["args"]
    talker = model.model.talker
    config = model.model.config

    for name in ("text_embed", "codec_embed", "text_lens", "codec_lens"):
        if name not in frame_locals:
            raise SystemExit(
                f"{case.id}: generate_icl_prompt's frame has no local {name!r} at return; "
                f"it holds {sorted(frame_locals)}"
            )

    # T1 and T2 are upstream's own locals, computed by upstream before the branch
    # and never reassigned -- not this script's arithmetic.
    t1 = int(frame_locals["text_lens"])
    t2 = int(frame_locals["codec_lens"])
    text_embed = frame_locals["text_embed"]
    codec_embed = frame_locals["codec_embed"]
    icl = sink["icl_input_embed"]
    trailing = sink["trailing_text_hidden"]

    ref_id = call_kwargs["ref_id"]
    text_id = call_kwargs["text_id"]
    ref_code = call_kwargs["ref_code"]
    tts_pad_embed = call_kwargs["tts_pad_embed"]
    tts_eos_embed = call_kwargs["tts_eos_embed"]

    gate(checks, "non_streaming_mode",
         call_kwargs.get("non_streaming_mode") is False,
         f"generate_voice_clone must default non_streaming_mode to False; saw "
         f"{call_kwargs.get('non_streaming_mode')!r}. The streaming layout is a different "
         "prompt and these artifacts would not describe it.")

    # ---- the arm, observed three independent ways -------------------------
    arm_from_line = sink["arm_from_return_line"]
    # `trailing_text_hidden` in the pad arm *is* the object upstream was handed;
    # in the truncate arm it is a fresh slice of text_embed. Object identity is
    # not a length comparison, so this is genuinely a second observation.
    arm_from_identity = ARM_PAD if trailing is tts_pad_embed else ARM_TRUNCATE
    # The truncate arm leaves `text_embed` at its full T1; the pad arm rebinds it
    # to the padded T2-length tensor.
    text_local_positions = int(text_embed.shape[1])
    if text_local_positions == t1 == t2:
        arm_from_length = arm_from_line  # indistinguishable when T1 == T2; see below
    elif text_local_positions == t1:
        arm_from_length = ARM_TRUNCATE
    elif text_local_positions == t2:
        arm_from_length = ARM_PAD
    else:
        arm_from_length = None

    # Ordered before the agreement gate on purpose. `arm_from_identity` can only
    # ever be pad or truncate, so a streaming return would trip the agreement
    # gate first and be reported as "the three readings disagree" -- a
    # misdiagnosis of a case that is really "this script does not describe that
    # layout". Checking the one reading that can actually say `stream` first is
    # what makes this gate reachable rather than decorative.
    gate(checks, "arm_is_not_streaming", arm_from_line != ARM_STREAM,
         f"generate_icl_prompt returned from its streaming arm (line "
         f"{sink['return_line']}); these artifacts describe the non-streaming layout only, "
         "in which the two tracks are summed rather than concatenated")
    gate(checks, "arm_observations_agree",
         arm_from_line == arm_from_identity == arm_from_length,
         f"the three independent readings of the alignment arm disagree: return line "
         f"{sink['return_line']} says {arm_from_line!r}, trailing-object identity says "
         f"{arm_from_identity!r}, text_embed length {text_local_positions} says "
         f"{arm_from_length!r} (T1={t1}, T2={t2})")
    arm = arm_from_line

    # ---- the transcribed shapes, made executable --------------------------
    gate(checks, "T1_is_ref_plus_target_plus_eos",
         t1 == int(ref_id.shape[1]) + int(text_id.shape[1]) + 1,
         f"T1={t1} but len(ref_id)={int(ref_id.shape[1])} + len(text_id)="
         f"{int(text_id.shape[1])} + 1 = {int(ref_id.shape[1]) + int(text_id.shape[1]) + 1}")
    ref_frames = int(ref_code.shape[0])
    gate(checks, "T2_is_one_plus_ref_frames", t2 == 1 + ref_frames,
         f"T2={t2} but 1 + ref_frames = {1 + ref_frames}")
    gate(checks, "ref_code_groups_match_config",
         int(ref_code.shape[1]) == int(talker.config.num_code_groups),
         f"reference codes carry {int(ref_code.shape[1])} groups, config says "
         f"{int(talker.config.num_code_groups)}")

    hidden = int(icl.shape[-1])
    gate(checks, "icl_block_is_T2_positions",
         tuple(icl.shape) == (1, t2, hidden),
         f"the summed block is {tuple(icl.shape)}, expected (1, {t2}, {hidden})")
    gate(checks, "codec_track_is_T2_positions",
         tuple(codec_embed.shape) == (1, t2, hidden),
         f"the codec track is {tuple(codec_embed.shape)}, expected (1, {t2}, {hidden})")

    # ---- the tracks, rebuilt through upstream's own modules ---------------
    rebuilt_text = rebuild_text_track(talker, ref_id, text_id, tts_eos_embed)
    gate(checks, "text_track_matches_upstream_construction",
         tuple(rebuilt_text.shape) == (1, t1, hidden)
         and torch.equal(rebuilt_text, text_embed[:, :t1]),
         "re-running text_projection(text_embeddings(cat([ref_id, text_id]))) then "
         "cat([., tts_eos_embed]) through upstream's own modules does not reproduce the "
         "captured text track; the transcription in prompt_conventions.json is wrong")
    rebuilt_codec = rebuild_codec_track(talker, config, ref_code, ref_id.dtype)
    gate(checks, "codec_track_matches_upstream_construction",
         torch.equal(rebuilt_codec, codec_embed),
         "re-running the 16-group embedding sum with a codec_bos_id row prepended through "
         "upstream's own modules does not reproduce the captured codec track; the "
         "transcription in prompt_conventions.json is wrong")

    # ---- the aligned pair, and the sum it must reproduce ------------------
    if arm == ARM_TRUNCATE:
        text_track = text_embed[:, :t2]
        expected_trailing = text_embed[:, t2:]
        gate(checks, "trailing_is_the_text_tail",
             torch.equal(trailing, expected_trailing),
             "the truncate arm's trailing schedule is not text_embed[:, T2:]")
    else:
        text_track = text_embed
        gate(checks, "trailing_is_the_pad_embedding",
             torch.equal(trailing, tts_pad_embed) and tuple(trailing.shape) == (1, 1, hidden),
             f"the pad arm's trailing schedule should be a bare tts_pad_embed, got "
             f"{tuple(trailing.shape)}")
        gate(checks, "padded_text_track_reaches_T2",
             int(text_track.shape[1]) == t2,
             f"the padded text track is {int(text_track.shape[1])} positions, expected {t2}")

    gate(checks, "tracks_sum_to_the_block_exactly",
         torch.equal(text_track + codec_embed, icl),
         "text_track + codec_track does not reproduce the captured ICL block bitwise in the "
         "model's own dtype; the capture is not capturing what it claims")

    # ---- the ids, rebuilt through upstream's own builders and tokenizer ----
    ref_lo, ref_hi = generate_source["ref_id_slice"]
    text_lo, text_hi = generate_source["text_id_slice"]
    gate(checks, "id_slices_are_the_transcribed_ones",
         (ref_lo, ref_hi) == (3, -2) and (text_lo, text_hi) == (3, -5),
         f"upstream now slices ref_id[{ref_lo}:{ref_hi}] and text_id[{text_lo}:{text_hi}]; "
         "the transcribed [3:-2] / [3:-5] no longer hold")
    rebuilt_ref = model._tokenize_texts([model._build_ref_text(case.ref_text)])[0][:, ref_lo:ref_hi]
    rebuilt_target = model._tokenize_texts(
        [model._build_assistant_text(case.text)])[0][:, text_lo:text_hi]
    gate(checks, "ref_ids_match_the_reference_turn_wrapper",
         torch.equal(rebuilt_ref.cpu(), ref_id.cpu()),
         "tokenizing _build_ref_text(ref_text) and slicing [3:-2] does not reproduce the "
         "reference ids upstream passed; the reference turn wrapper is not what was recorded")
    gate(checks, "target_ids_match_the_assistant_turn_wrapper",
         torch.equal(rebuilt_target.cpu(), text_id.cpu()),
         "tokenizing _build_assistant_text(text) and slicing [3:-5] does not reproduce the "
         "target ids upstream passed; the target turn wrapper is not what was recorded")

    # ---- placement, and the x-vector's survival, both observed ------------
    talker_kwargs = sink["talker_kwargs"]
    assembled = talker_kwargs["inputs_embeds"]
    gate(checks, "block_is_appended_at_the_end_of_the_prefill",
         int(assembled.shape[1]) > t2 and torch.equal(assembled[:, -t2:], icl),
         f"the assembled prefill's last {t2} positions are not the ICL block, so the block "
         "is not simply concatenated after the existing prefix")
    prefix_positions = int(assembled.shape[1]) - t2

    # The x-vector claim, proved rather than read: upstream builds the speaker
    # slot as tts_pad_embed + speaker_embed, so exactly one position of the
    # prefill equals that sum bitwise. Finding it proves the speaker embedding
    # survived into the ICL prompt; finding none would falsify D5's table.
    speaker_embed = prompt_items[0].ref_spk_embedding.to(talker.device).to(talker.dtype).view(1, 1, -1)
    speaker_row = (tts_pad_embed + speaker_embed)[0, 0]
    matches = [
        position for position in range(prefix_positions)
        if torch.equal(assembled[0, position], speaker_row)
    ]
    gate(checks, "x_vector_is_present_in_the_icl_prompt", len(matches) == 1,
         f"expected exactly one prefill position equal to tts_pad_embed + speaker_embed, "
         f"found {len(matches)} ({matches}). ICL is supposed to add to the x-vector path, "
         "not replace it (design D5).")
    speaker_slot = matches[0]

    # ---- the trailing schedule's own contents ----------------------------
    trailing_positions = int(trailing.shape[1])
    trailing_token_ids: list[int] = []
    trailing_ends_with_tts_eos = False
    trailing_id_margin: Optional[dict] = None
    if arm == ARM_TRUNCATE:
        gate(checks, "trailing_length_is_T1_minus_T2", trailing_positions == t1 - t2,
             f"the truncate arm's trailing schedule is {trailing_positions} positions, "
             f"expected T1 - T2 = {t1 - t2}")
        # Positions T2..T1-2 of the text track are token embeddings of
        # cat([ref_id, text_id]); position T1-1 is tts_eos_embed.
        all_ids = torch.cat([ref_id, text_id], dim=-1)
        trailing_token_ids = [int(v) for v in all_ids[0, t2:].tolist()]
        trailing_ends_with_tts_eos = torch.equal(trailing[:, -1:], tts_eos_embed)
        gate(checks, "trailing_ends_with_tts_eos", trailing_ends_with_tts_eos,
             "the truncate arm's trailing schedule does not end with tts_eos_embed, so the "
             "recorded token ids do not describe it")
        # Runs whenever the schedule has a head at all -- a 2-position tail
        # leaves one head position, and one position is enough to separate the
        # aligned ids from a shift. The threshold used to be 2 and would have
        # skipped that case silently.
        if trailing_positions > 1:
            # Corroborating the recorded ids cannot be a bitwise check, and the
            # reason is worth stating: `text_projection` is a matmul, and running
            # it over a short slice of ids reassociates differently from running
            # it over the full sequence, so the same ids give bf16 results that
            # differ in the last bits. The ids are already justified bitwise --
            # `text_track_matches_upstream_construction` above rebuilt the whole
            # track from `cat([ref_id, text_id])` and matched exactly, and a
            # projection is position-wise, so position p of the track is the
            # projection of id p by construction. What this adds is a guard on
            # THIS script's index arithmetic: project the ids recorded for the
            # trailing schedule and confirm they land far closer to it than the
            # same ids shifted by one position do. An off-by-one in the recorded
            # ids is the failure this catches, and it is exactly the failure the
            # plan warns is inaudible.
            def projected(start: int) -> torch.Tensor:
                window = all_ids[:, start:start + trailing_positions - 1]
                return talker.text_projection(talker.get_text_embeddings()(window))

            head = trailing[:, :-1].to(torch.float32)

            def deviation(start: int) -> float:
                probe = projected(start).to(torch.float32)
                if probe.shape[1] != head.shape[1]:
                    return float("inf")
                return float((probe - head).abs().max())

            aligned = deviation(t2)
            shifted = min(deviation(t2 - 1), deviation(t2 + 1))
            trailing_id_margin = {
                "aligned_max_abs": aligned,
                "nearest_shift_max_abs": shifted,
                "head_positions": trailing_positions - 1,
            }
            gate(checks, "trailing_head_is_those_token_ids",
                 aligned * 4.0 < shifted,
                 f"projecting the recorded trailing token ids lands {aligned:.4g} from the "
                 f"trailing schedule, no better than the same ids shifted one position "
                 f"({shifted:.4g}); the recorded ids do not describe the schedule")
    else:
        gate(checks, "trailing_length_is_one", trailing_positions == 1,
             f"the pad arm's trailing schedule is {trailing_positions} positions, expected 1")
        trailing_id_margin = {
            "skipped": "the pad arm's schedule is a single tts_pad_embed, so it carries no "
                       "token ids to corroborate",
        }

    # ---- the arrays that will be written, checked as f32 ------------------
    text_f32 = to_numpy(text_track, torch.float32)
    codec_f32 = to_numpy(codec_embed, torch.float32)
    trailing_f32 = to_numpy(trailing, torch.float32)
    icl_f32 = to_numpy(icl, torch.float32)
    ref_ids_i32 = to_numpy(ref_id).astype(np.int32)
    target_ids_i32 = to_numpy(text_id).astype(np.int32)

    # These three catch infinities and nothing else, which is worth stating
    # because the bitwise gates below look like they subsume them and only half
    # do. `torch.equal` returns False for NaN (NaN != NaN elementwise), so a NaN
    # anywhere in a track already fails the sum gates -- but it returns True for
    # inf, and inf + finite = inf on both sides of the sum, so an infinity would
    # pass every bitwise check in this script. `trailing` has no sum gate at all.
    for name, array in (("text_track", text_f32), ("codec_track", codec_f32),
                        ("trailing", trailing_f32)):
        gate(checks, f"{name}_is_finite", bool(np.isfinite(array).all()),
             f"{name} holds non-finite values (an infinity survives every bitwise gate here, "
             "because inf == inf)")

    def sum_round_trips(text: np.ndarray, codec: np.ndarray,
                        block: np.ndarray) -> tuple[bool, dict]:
        """Does f32(text) + f32(codec), rounded back to the model dtype, equal the block?

        The written files are float32 widenings of bfloat16 tensors, so a
        consumer that adds them in float32 gets the exact real sum, one bfloat16
        rounding away from the block upstream computed. Rounding the f32 sum back
        through the model's dtype is the exact statement, and it is the statement
        Tasks 6, 7 and 12 can rely on.

        The deviations returned alongside it are measured here rather than
        derived so that the bound published in the artifacts is a number this run
        observed. A raw float32 comparison against ``icl_embed.f32`` -- the
        obvious thing a downstream test does -- has to allow this much.
        """
        summed = (torch.from_numpy(text) + torch.from_numpy(codec)).numpy()
        rounded = torch.from_numpy(summed).to(icl.dtype).to(torch.float32).numpy()
        absolute = np.abs(summed - block)
        nonzero = block != 0
        stats = {
            "max_abs": float(absolute.max()) if block.size else 0.0,
            "max_rel": float((absolute[nonzero] / np.abs(block[nonzero])).max())
                       if bool(nonzero.any()) else 0.0,
            # A purely relative bound is unusable where the block is exactly
            # zero, so the residue there is published separately.
            "zero_positions": int((~nonzero).sum()),
            "max_abs_where_block_is_zero": float(absolute[~nonzero].max())
                                           if bool((~nonzero).any()) else 0.0,
        }
        return bool(np.array_equal(rounded, block)), stats

    exact, sum_stats = sum_round_trips(text_f32, codec_f32, icl_f32)
    gate(checks, "written_tracks_sum_to_the_block",
         exact,
         f"adding the float32 tracks and rounding back to {icl.dtype} does not reproduce the "
         f"block (max float32 deviation {sum_stats['max_abs']:.3e})")

    # ---- the sibling artifact the Base dumper already wrote ---------------
    icl_embed_path = case_dir / "prompt" / "icl_embed.f32"
    if not icl_embed_path.exists():
        raise SystemExit(
            f"{case.id}: {icl_embed_path} does not exist. This script's tracks are dumped to "
            "sum to that file, so there is nothing to check them against. Run "
            "scripts/dump_reference_qwen3_tts_base.py for this case first."
        )
    on_disk = np.fromfile(icl_embed_path, dtype=np.float32)
    gate(checks, "block_matches_the_base_dumper_artifact",
         on_disk.size == icl_f32.size and bool(np.array_equal(on_disk, icl_f32.reshape(-1))),
         f"the ICL block this run built ({icl_f32.size} floats) does not match "
         f"{icl_embed_path} ({on_disk.size} floats) bitwise. The prompt is deterministic, so "
         "a mismatch means the two runs saw different inputs -- not a tolerance question.")

    # ---- everything that can still raise, resolved before the first write ---
    # `span_source` reads the file again and `build_conventions` resolves spans
    # the startup pass did not; doing both here means the only work left after
    # the first byte is written is writing the rest of the bytes.
    executed_return_source = span_source(
        icl_source["path"], sink["return_line"], sink["return_line"]).strip()
    numerics = {
        "storage_dtype": "float32",
        "model_dtype": str(icl.dtype).replace("torch.", ""),
        "rule": "text_track.f32 + codec_track.f32 reproduces icl_embed.f32 only after the "
                "float32 sum is rounded back to the model dtype. Compare "
                "round_to_bfloat16(text_track + codec_track) against icl_embed for a bitwise "
                "result; a raw float32 comparison must carry the tolerance below.",
        "why": "The three files are float32 widenings of bfloat16 tensors, so adding two of "
               "them in float32 produces the exact real sum, while upstream computed the sum "
               "in bfloat16 and rounded once. The gap is one bfloat16 rounding, not an error.",
        "raw_f32_sum_max_abs_deviation": sum_stats["max_abs"],
        "raw_f32_sum_max_rel_deviation": sum_stats["max_rel"],
        "raw_f32_sum_max_rel_deviation_as_power_of_two": (
            round(float(np.log2(sum_stats["max_rel"])), 6) if sum_stats["max_rel"] > 0 else None
        ),
        "block_zero_positions": sum_stats["zero_positions"],
        "raw_f32_sum_max_abs_deviation_where_block_is_zero":
            sum_stats["max_abs_where_block_is_zero"],
        "tolerance_note": "The relative figure is bfloat16's unit roundoff, 2**-8 == "
                          "0.00390625 -- NOT the 2**-9 half-ulp figure, which a raw float32 "
                          "comparison will exceed. Measured on this case's own bytes, not "
                          "derived. Where the block is exactly zero a relative bound does not "
                          "apply; use the absolute figure recorded beside it.",
    }
    conventions = build_conventions(icl_source, generate_source, model, case,
                                    {"T1": t1, "T2": t2, "branch": arm,
                                     "trailing_positions": trailing_positions},
                                    numerics)

    # ---- write, then verify the bytes on disk -----------------------------
    # Every write lives inside one guard, including the two JSON files. Leaving
    # them outside it was a real hole: a raise between the binaries and
    # prompt_conventions.json would have left five binaries and alignment.json on
    # disk -- a complete, usable, and by then unverified artifact set.
    prompt_dir = case_dir / "prompt"
    written: list[pathlib.Path] = []

    def record(name: str, writer, payload) -> dict:
        path = prompt_dir / name
        info = writer(path, payload)
        written.append(path)
        return info

    try:
        artifacts = {
            "text_track": record("text_track.f32", write_f32, text_f32),
            "codec_track": record("codec_track.f32", write_f32, codec_f32),
            "trailing": record("trailing.f32", write_f32, trailing_f32),
            "ref_text_ids": record("ref_text_ids.i32", write_i32, ref_ids_i32),
            "target_text_ids": record("target_text_ids.i32", write_i32, target_ids_i32),
        }
        reread_exact, reread_stats = sum_round_trips(
            np.fromfile(prompt_dir / "text_track.f32", dtype=np.float32),
            np.fromfile(prompt_dir / "codec_track.f32", dtype=np.float32),
            on_disk,
        )
        gate(checks, "bytes_on_disk_sum_to_icl_embed_f32", reread_exact,
             f"re-reading the written tracks and adding them does not reproduce "
             f"{icl_embed_path} (max float32 deviation {reread_stats['max_abs']:.3e})")
        alignment = build_alignment(
            case, t1, t2, arm, ref_frames, trailing_positions, hidden, numerics,
            sink, executed_return_source, arm_from_line, arm_from_identity, arm_from_length,
            text_local_positions, trailing_token_ids, trailing_ends_with_tts_eos,
            trailing_id_margin, prefix_positions, assembled, speaker_slot,
            ref_ids_i32, target_ids_i32, (ref_lo, ref_hi), (text_lo, text_hi),
            artifacts, ref_info, checks, started)
        record("alignment.json", write_json_artifact, alignment)
        record("prompt_conventions.json", write_json_artifact, conventions)
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return {"id": case.id, "alignment": alignment}


def write_json_artifact(path: pathlib.Path, payload: dict) -> dict:
    write_json(path, payload)
    return {"path": path.name}


def build_alignment(case, t1, t2, arm, ref_frames, trailing_positions, hidden, numerics,
                    sink, executed_return_source, arm_from_line, arm_from_identity,
                    arm_from_length, text_local_positions, trailing_token_ids,
                    trailing_ends_with_tts_eos, trailing_id_margin, prefix_positions,
                    assembled, speaker_slot, ref_ids_i32, target_ids_i32, ref_slice,
                    text_slice, artifacts, ref_info, checks, started) -> dict:
    ref_lo, ref_hi = ref_slice
    text_lo, text_hi = text_slice
    alignment = {
        "schema": ALIGNMENT_SCHEMA,
        "case": case.id,
        "T1": t1,
        "T2": t2,
        "branch": arm,
        "ref_frames": ref_frames,
        "trailing_positions": trailing_positions,
        "hidden_size": hidden,
        "branch_observation": {
            "note": "The arm is read out of the return statement that executed, never "
                    "recomputed from T1 and T2. Two further independent signals must agree.",
            "executed_return_line": sink["return_line"],
            "executed_return_source": executed_return_source,
            "arm_from_return_line": arm_from_line,
            "arm_from_trailing_object_identity": arm_from_identity,
            "arm_from_text_embed_length": arm_from_length,
            "text_embed_positions_at_return": text_local_positions,
            "arms_are_indistinguishable_by_length": t1 == t2,
        },
        "trailing": {
            "kind": "text_track_tail" if arm == ARM_TRUNCATE else "tts_pad_embed",
            "positions": trailing_positions,
            "token_ids": trailing_token_ids,
            "ends_with_tts_eos": trailing_ends_with_tts_eos,
            "token_id_margin": trailing_id_margin,
            "note": "In the truncate arm the schedule is text_embed[:, T2:]: the projected "
                    "embeddings of the token ids listed here, followed by tts_eos_embed. In "
                    "the pad arm it is a single bare tts_pad_embed and token_ids is empty.",
        },
        "placement": {
            "prefix_positions": prefix_positions,
            "assembled_prompt_positions": int(assembled.shape[1]),
            "block_at": [prefix_positions, int(assembled.shape[1])],
            "speaker_slot_index": speaker_slot,
            "note": "speaker_slot_index is the prefill position found to equal "
                    "tts_pad_embed + speaker_embed bitwise -- observed, not derived. Its "
                    "presence is what shows ICL adds to the x-vector path rather than "
                    "replacing it.",
        },
        "ids": {
            "ref_text_ids": len(ref_ids_i32.reshape(-1)),
            "target_text_ids": len(target_ids_i32.reshape(-1)),
            "ref_id_slice": [ref_lo, ref_hi],
            "text_id_slice": [text_lo, text_hi],
        },
        "artifacts": artifacts,
        "inputs": {
            "ref_audio": case.ref_audio,
            "ref_text": case.ref_text,
            "text": case.text,
            "language": case.language,
            "trim_seconds": case.trim_seconds,
            "reference_audio": ref_info,
        },
        "numerics": numerics,
        "checks": checks,
        "wall_seconds": round(time.time() - started, 3),
    }
    return alignment


def build_conventions(icl_source: dict, generate_source: dict, model, case: RunCase,
                      observed: dict, numerics: dict) -> dict:
    """The transcription, with every span extracted from the installed source."""
    icl_path = icl_source["path"]
    gen_path = generate_source["path"]
    spans = icl_source["spans"]
    gen_spans = generate_source["spans"]
    model_path = pathlib.Path(inspect.getsourcefile(type(model)))

    def method_citation(name: str, claim: str) -> dict:
        method = getattr(type(model), name)
        lines, start = inspect.getsourcelines(method)
        return cite(model_path, start, start + len(lines) - 1, claim, "executed")

    return {
        "schema": CONVENTIONS_SCHEMA,
        "note": "Every span below was located by walking the AST of the installed qwen_tts "
                "package at run time and its text read straight out of that file, so a "
                "citation here cannot be stale: if upstream moves or reshapes a construct, "
                "the locator fails and no artifacts are written. `verified_by` says how each "
                "claim was checked -- 'executed' means this script re-ran upstream's own code "
                "and compared bitwise, 'observed' means it was read out of a running frame or "
                "the assembled prompt, 'ast' means the structure itself was asserted.",
        "numerics": numerics,
        "upstream": {
            # Read out of the installed distribution rather than written down.
            # A hardcoded revision is the one fact that can go stale silently in
            # a file whose whole premise is that nothing here is hardcoded.
            **installed_upstream(),
            "files": [
                str(icl_path.relative_to(icl_path.parents[3])),
                str(model_path.relative_to(model_path.parents[2])),
            ],
        },
        "text_track": cite(
            icl_path, *spans["text_track"],
            "text_projection(text_embeddings(cat([ref_id, text_id]))), then "
            "cat([., tts_eos_embed]). T1 = len(ref_id) + len(text_id) + 1.",
            "executed"),
        "codec_track": cite(
            icl_path, *spans["codec_track"],
            "Per reference frame, the sum of num_code_groups embeddings: group 0 from "
            "talker.get_input_embeddings(), groups 1..15 from "
            "talker.code_predictor.get_input_embeddings()[i-1]. One codec_bos_id row is "
            "prepended through the talker's own codec embedding. T2 = 1 + ref_frames.",
            "executed"),
        "lens": cite(icl_path, *spans["lens"],
                     "T1 and T2 are upstream's own text_lens / codec_lens, computed before "
                     "the branch and never reassigned; alignment.json reports these locals.",
                     "observed"),
        "alignment": cite(
            icl_path, *spans["alignment"],
            "With non_streaming_mode=False: if T1 > T2 the block is text_embed[:, :T2] + "
            "codec_embed and text_embed[:, T2:] becomes the trailing schedule ('truncate'); "
            "otherwise the text track is padded with tts_pad_embed up to T2 and the trailing "
            "schedule is a bare tts_pad_embed ('pad'). Both arms emit T2 positions, so the "
            "block's size does not identify the arm -- the trailing schedule does.",
            "observed"),
        "alignment_arm_return_lines": {
            "claim": "Which arm ran is control flow, not a value. alignment.json names it from "
                     "the line of the return that executed; these are the line ranges each arm "
                     "can return from, located by AST.",
            "file": icl_path.name,
            "ranges": icl_source["arm_line_ranges"],
            "verified_by": "ast",
        },
        "call_site": cite(
            gen_path, *gen_spans["call"],
            "generate_icl_prompt is called with text_id=input_id[:, 3:-5] and "
            "ref_id=ref_ids[index][:, 3:-2]; the two turn wrappers below are what those "
            "slices cut into.",
            "executed"),
        "placement": cite(
            gen_path, *gen_spans["placement"],
            "The block is concatenated AFTER the existing prefix. It replaces the "
            "tts_text_first_token position the non-ICL branch builds.",
            "observed"),
        "non_icl_branch": cite(
            gen_path, *gen_spans["non_icl"],
            "What the ICL block replaces: the non-ICL branch appends a single "
            "tts_text_first_token position instead.",
            "ast"),
        "speaker_slot": cite(
            gen_path, *gen_spans["speaker_slot"],
            "speaker_embed is inserted into codec_input_emebdding regardless of mode, and "
            "this branch is a sibling that precedes the ICL branch rather than an alternative "
            "to it. ICL ADDS to the x-vector path; it does not substitute for it. This matches "
            "the design's D5 table, where the speaker embedding is 'yes' in both columns.",
            "observed"),
        "non_streaming_default": {
            "claim": "generate_voice_clone defaults non_streaming_mode to False, which is the "
                     "arm pair described above; the observed keyword is recorded in "
                     "alignment.json's checks.",
            "file": model_path.name,
            "verified_by": "observed",
        },
        "reference_turn_wrapper": method_citation(
            "_build_ref_text",
            "The reference transcript uses a different turn wrapper from the target text: it "
            "has no trailing '<|im_start|>assistant\\n', which is why it is sliced [3:-2] "
            "where the target text is sliced [3:-5]."),
        "target_turn_wrapper": method_citation(
            "_build_assistant_text",
            "The target text's wrapper, tokenized whole and sliced [3:-5]."),
        "observed": {"case": case.id, **observed},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--weights-dir", required=True, type=pathlib.Path)
    parser.add_argument("--device", default="cuda:0",
                        help="device_map forwarded to Qwen3TTSModel.from_pretrained.")
    parser.add_argument("--ref-audio", default=None)
    parser.add_argument("--ref-text", default=None)
    parser.add_argument("--text", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--trim-seconds", type=float, default=None)
    parser.add_argument("--out-dir", type=pathlib.Path, default=None)
    parser.add_argument("--manifest", type=pathlib.Path, default=None,
                        help="Golden Manifest to read cases from. Never opened unless passed.")
    parser.add_argument("--case", action="append", default=None,
                        help="--manifest form only: restrict to this case id (repeatable).")
    parser.add_argument("--output-root", type=pathlib.Path, default=None)
    parser.add_argument("--reference-audio-dir", type=pathlib.Path,
                        default=pathlib.Path("models/qwen3-tts-reference-audio"))
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

    # The same reference configuration the Base dumper used to produce the
    # prompt/icl_embed.f32 these tracks are checked against.
    model = Qwen3TTSModel.from_pretrained(
        str(args.weights_dir),
        device_map=args.device,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    # Locate every construct before touching a case: a stale citation should stop
    # the run, not decorate an artifact.
    icl_source = describe_icl_prompt_source(model.model.generate_icl_prompt)
    generate_source = describe_generate_source(model.model.generate)
    print(f"reference: {args.device} bfloat16 eager (ICL prompt)", flush=True)
    print(f"    alignment arms at {icl_source['path'].name}: "
          f"{icl_source['arm_line_ranges']}", flush=True)

    records = []
    for index, (case, case_dir) in enumerate(run_specs, 1):
        print(f"[{index}/{len(run_specs)}] {case.id}", flush=True)
        records.append(run_case(model, case, case_dir, args.reference_audio_dir,
                                icl_source, generate_source))
        alignment = records[-1]["alignment"]
        print(f"    T1={alignment['T1']} T2={alignment['T2']} branch={alignment['branch']} "
              f"ref_frames={alignment['ref_frames']} "
              f"trailing={alignment['trailing_positions']} "
              f"(return line {alignment['branch_observation']['executed_return_line']})",
              flush=True)

    report = {
        "schema": "synthesize-oracle-dump-v1",
        "family": FAMILY,
        "variant": "qwen3-tts-12hz-0-6b-base",
        "stage": "icl_prompt",
        "reference": {"device": args.device, "dtype": "bfloat16",
                      "attn_implementation": "eager"},
        "case_count": len(records),
        "cases": records,
    }
    if args.report:
        write_json(args.report, report)
    print(f"dumped {len(records)} case(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
