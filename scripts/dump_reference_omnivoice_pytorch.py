#!/usr/bin/env python3
"""Drive the pinned OmniVoice oracle over every Golden Manifest case.

The dumper reimplements no OmniVoice arithmetic. It drives the pinned
``OmniVoice.generate`` entry point at F32 on CPU and captures intermediates
through forward hooks and two narrowly scoped method wraps, so the C++ port can
be compared stage by stage rather than only at the waveform.

Three properties of this family shape the script.

**Greedy runs make no RNG call at all.** ``position_temperature`` and
``class_temperature`` are both zero on seventeen of the twenty cases, so the
decode loop takes neither Gumbel branch. That is stronger than a seeded
reproduction: the token grid is bit-reproducible across processes, which is why
those cases are an exact-equality target rather than a replay input. The dumper
does not merely assume it -- it snapshots the torch RNG state around every
greedy generation and fails if a single draw moved it, and it runs each greedy
case twice in one process and fails unless the two grids are identical.

**The canvas length is fixed before the first forward.** ``RuleDurationEstimator``
decides how many frames to paint, so a determinism check that compared output
*lengths* would call this family deterministic even when it is not. Everything
here compares content.

**One output scaling survives the contract's switches.** With
``postprocess_output=False, pad_duration=0.0, fade_duration=0.0`` the volume
branch inside ``_post_process_audio`` is still ungated: auto-voice and
voice-design output is peak-normalised to exactly 0.5, and a clone whose
reference is quiet is scaled by ``ref_rms / 0.1``. ``audio/pcm.f32`` is the
waveform **as returned to the caller**, scaling included, because that is what
the port must reproduce; ``result.json`` records the per-case peak and the
branch that produced it so the scaling is visible rather than folded silently
into the tolerance budget.

Usage:

    uv run --project scripts/envs/omnivoice --locked python \\
      scripts/dump_reference_omnivoice_pytorch.py \\
      --manifest tests/golden/omnivoice/omnivoice-0-6b.manifest.json \\
      --weights-dir models/omnivoice-0-6b \\
      --report build/goldens/omnivoice/dump-report.json

The manifest can be checked on its own, with no weights and no torch import:

    python scripts/dump_reference_omnivoice_pytorch.py \\
      --manifest tests/golden/omnivoice/omnivoice-0-6b.manifest.json \\
      --validate-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import omnivoice_pinned_inputs  # noqa: E402

MANIFEST_SCHEMA = "synthesize-golden-manifest-v1"
FAMILY = "omnivoice"
DUMP_SCHEMA = "synthesize-oracle-dump-v1"

# The manifest names these; the hooks the dumper installs must produce exactly
# this set, captured on the step-0 conditional forward. load_manifest() checks
# the two agree rather than letting a renamed artifact dump nothing.
GENERATOR_PROBE_LAYERS = (0, 7, 14, 21, 27)
PROBE_ARTIFACT_PREFIX = "generator.hidden_l"

# 24 kHz mono at a 25 Hz frame rate: one codec frame is exactly 960 samples.
SAMPLES_PER_FRAME = 960
NATIVE_SAMPLE_RATE = 24000
# Derived, not restated: the frame rate feeds the audio_chunk_threshold gate,
# and a restated value would drift silently if a revision ever moved the hop.
FRAME_RATE_HZ = NATIVE_SAMPLE_RATE / SAMPLES_PER_FRAME
AUDIO_MASK_ID = 1024
NUM_CODEBOOKS = 8

# Keys of OmniVoiceGenerationConfig. Everything else in oracle.parameters is
# either a named generate() argument or manifest bookkeeping; splitting them by
# an explicit list rather than by "whatever is left" means a manifest key the
# oracle would silently drop is reported instead.
GEN_CONFIG_KEYS = frozenset(
    {
        "num_step",
        "guidance_scale",
        "t_shift",
        "layer_penalty_factor",
        "position_temperature",
        "class_temperature",
        "denoise",
        "preprocess_prompt",
        "postprocess_output",
        "audio_chunk_duration",
        "audio_chunk_threshold",
        "pad_duration",
        "fade_duration",
    }
)
GENERATE_ARGUMENT_KEYS = frozenset({"language", "instruct"})
# Consumed by the dumper rather than passed to generate(): ref_text builds the
# clone prompt, seeded_by is a note about where the seed comes from.
DUMPER_ONLY_KEYS = frozenset({"ref_text", "seeded_by"})


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
        "--reference-audio-dir",
        type=pathlib.Path,
        default=pathlib.Path("models/omnivoice-reference-audio"),
        help="Where the digest-pinned clone reference wav is cached (git-ignored).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Read and check the manifest, print the planned work, and stop.",
    )
    return parser.parse_args(argv)


def require(mapping, key, where: str):
    """Read a manifest key, or say which one is missing and where.

    Manifest structure is walked through this rather than indexed directly. A
    bare ``KeyError`` traceback from four frames into a dump names the key and
    nothing else; the case and the field are the part worth printing, and the
    caller turns a ManifestError into the same ``error: ...`` exit-1 contract
    every other manifest problem uses.
    """
    if not isinstance(mapping, dict):
        raise ManifestError(f"{where}: expected an object, found {type(mapping).__name__}")
    if key not in mapping:
        raise ManifestError(f"{where}: missing required key {key!r}")
    return mapping[key]


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

    if require(manifest, "schema", str(path)) != MANIFEST_SCHEMA:
        raise ManifestError(
            f"{path} declares schema {manifest['schema']!r}, expected {MANIFEST_SCHEMA!r}"
        )
    if require(manifest, "family", str(path)) != FAMILY:
        raise ManifestError(
            f"{path} is a {manifest['family']!r} manifest; this runner drives {FAMILY!r}"
        )

    reference = require(manifest, "reference", str(path))
    if reference.get("dtype") != "float32" or reference.get("device") != "cpu":
        raise ManifestError(
            "this runner drives the oracle at float32 on cpu; the manifest asks for "
            f"{reference.get('dtype')!r} on {reference.get('device')!r}"
        )

    require(manifest, "case_artifact_root", str(path))
    contract = require(manifest, "package_contract", str(path))
    require(contract, "max_output_frames", f"{path}: package_contract")

    cases = require(manifest, "cases", str(path))
    if not cases:
        raise ManifestError(f"{path} declares no cases")

    seen: set[str] = set()
    for case in cases:
        case_id = require(case, "id", f"{path}: case")
        where = f"{path}: case {case_id}"
        if case_id in seen:
            raise ManifestError(f"duplicate case id {case_id!r}")
        seen.add(case_id)

        case_input = require(case, "input", where)
        require(case_input, "text", f"{where}.input")
        reference_input = case_input.get("reference")
        if reference_input is not None:
            require(reference_input, "artifact", f"{where}.input.reference")

        request = require(case, "request", where)
        for key in ("seed_u64", "speaking_rate"):
            require(request, key, f"{where}.request")

        parameters = require(require(case, "oracle", where), "parameters", f"{where}.oracle")
        if reference_input is not None:
            # Resolving the digest HERE turns an unpinned reference into an
            # `error:` + exit 1 before any model loads, instead of a
            # ManifestError escaping mid-dump with three cases already written.
            reference_digest(manifest, reference_input["artifact"])
            if parameters.get("preprocess_prompt") is not False:
                raise ManifestError(
                    f"{where}: a clone case must pin preprocess_prompt=false; anything else "
                    "lets silence stripping into a parity baseline"
                )

        for key in ("num_step", "position_temperature", "class_temperature", "language",
                    "instruct", "postprocess_output",
                    "audio_chunk_duration", "audio_chunk_threshold"):
            require(parameters, key, f"{where}.oracle.parameters")
        threshold_frames = float(parameters["audio_chunk_threshold"]) * FRAME_RATE_HZ
        if threshold_frames < float(contract["max_output_frames"]):
            raise ManifestError(
                f"{where}: audio_chunk_threshold {parameters['audio_chunk_threshold']} s is "
                f"{threshold_frames:.0f} frames, below max_output_frames "
                f"{contract['max_output_frames']}; a golden case could silently take the "
                "chunked long-form path, which is out of scope"
            )
        unknown = set(parameters) - GEN_CONFIG_KEYS - GENERATE_ARGUMENT_KEYS - DUMPER_ONLY_KEYS
        if unknown:
            raise ManifestError(
                f"{where}.oracle.parameters: {sorted(unknown)} would be silently dropped by "
                "OmniVoiceGenerationConfig.from_dict; the dump would not be driving what the "
                "manifest describes"
            )
        if reference_input is not None and not parameters.get("ref_text"):
            raise ManifestError(
                f"{where}: a clone case must pin oracle.parameters.ref_text; without it the "
                "oracle would transcribe the reference with Whisper"
            )

        expected = require(case, "expected", where)
        artifacts = require(expected, "artifacts", f"{where}.expected")
        names: set[str] = set()
        for artifact in artifacts:
            name = require(artifact, "name", f"{where}.expected.artifacts[]")
            relative = pathlib.PurePosixPath(
                require(artifact, "path", f"{where}.expected.artifacts[{name}]")
            )
            require(artifact, "format", f"{where}.expected.artifacts[{name}]")
            if relative.is_absolute() or ".." in relative.parts:
                raise ManifestError(
                    f"{case_id}: artifact path {artifact['path']!r} escapes the case root"
                )
            if name in names:
                raise ManifestError(f"{case_id}: duplicate artifact name {name!r}")
            names.add(name)

        # The probe set is a shared constant between the manifest and the hook
        # wiring below. If they drift, the dump either writes a file nothing
        # asked for or silently skips one the comparison expects.
        probed = {
            name for name in names if name.startswith(PROBE_ARTIFACT_PREFIX)
        }
        wanted = {f"{PROBE_ARTIFACT_PREFIX}{index}" for index in GENERATOR_PROBE_LAYERS}
        if probed != wanted:
            raise ManifestError(
                f"{case_id}: generator probe artifacts {sorted(probed)} do not match this "
                f"runner's GENERATOR_PROBE_LAYERS {sorted(wanted)}"
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


def is_greedy(case: dict) -> bool:
    parameters = case["oracle"]["parameters"]
    return (
        parameters["position_temperature"] == 0.0 and parameters["class_temperature"] == 0.0
    )


def describe(manifest: dict, cases: list[dict], output_root: pathlib.Path) -> None:
    """Print the work the dump will do, so the plan is inspectable up front."""
    greedy = [case for case in cases if is_greedy(case)]
    sampled = [case for case in cases if not is_greedy(case)]
    clone = [case for case in cases if case["input"].get("reference")]

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


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------


def write_f32(path: pathlib.Path, array) -> dict:
    import numpy as np

    data = np.ascontiguousarray(array, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.tobytes())
    return {"shape": list(data.shape), "elements": int(data.size)}


def write_i32(path: pathlib.Path, array) -> dict:
    import numpy as np

    data = np.ascontiguousarray(array, dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.tobytes())
    return {"shape": list(data.shape), "elements": int(data.size)}


def write_json(path: pathlib.Path, payload: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"keys": sorted(payload)}


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def _first_tensor(output):
    tensor = output[0] if isinstance(output, (tuple, list)) else output
    return getattr(tensor, "last_hidden_state", tensor)


class GeneratorProbes:
    """Hidden states at the named backbone layers, and the final norm.

    Only the first forward is kept. The decode loop runs ``num_step`` batched
    forwards, each conditioned on the tokens committed so far; step 0 is the one
    a port reproduces without needing the commit history, so comparing it
    isolates the backbone from the diffusion loop.

    Nothing here guesses where the layers live: the attribute path is resolved
    against the loaded module tree, and the number of hooks installed is
    asserted against the probe list plus the one norm.
    """

    def __init__(self, model) -> None:
        import torch

        self.torch = torch
        self.captured: dict[str, object] = {}
        self.handles = []

        llm = getattr(model, "llm", None)
        if llm is None:
            raise SystemExit("model has no .llm attribute; the probe wiring is out of date")
        # Qwen3Model exposes .layers/.norm directly; Qwen3ForCausalLM nests them
        # one level down under .model. Accept either rather than assuming.
        host = llm if hasattr(llm, "layers") else getattr(llm, "model", None)
        if host is None or not hasattr(host, "layers"):
            raise SystemExit(
                f"cannot find the backbone layers on {type(llm).__name__}; "
                "the probe wiring is out of date"
            )
        self.layer_path = "llm.layers" if host is llm else "llm.model.layers"
        norm = getattr(host, "norm", None)
        if norm is None:
            raise SystemExit(
                f"cannot find the final norm on {type(host).__name__}; "
                "the probe wiring is out of date"
            )
        self.norm_path = self.layer_path.rsplit(".", 1)[0] + ".norm"
        self.layer_count = len(host.layers)

        for index in GENERATOR_PROBE_LAYERS:
            if index >= self.layer_count:
                raise SystemExit(
                    f"backbone has {self.layer_count} layers; probe layer {index} is out of range"
                )
            self.handles.append(
                host.layers[index].register_forward_hook(
                    self._make_hook(f"{PROBE_ARTIFACT_PREFIX}{index}")
                )
            )
        self.handles.append(norm.register_forward_hook(self._make_hook("generator.final")))
        if len(self.handles) != len(GENERATOR_PROBE_LAYERS) + 1:
            raise SystemExit(
                f"installed {len(self.handles)} hooks, expected "
                f"{len(GENERATOR_PROBE_LAYERS) + 1} (probe layers plus the final norm)"
            )

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            if name in self.captured:
                return  # step 0 only; later calls carry committed tokens
            tensor = _first_tensor(output)
            if self.torch.is_tensor(tensor):
                self.captured[name] = tensor.detach().to(self.torch.float32).cpu().numpy()

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


class LogitsProbe:
    """The step-0 audio-head logits, taken from the model's own return value.

    ``OmniVoice.forward`` already reshapes ``audio_heads``'s flat output into
    ``[B, C, S, V]``. Hooking the model rather than the head captures the
    permutation as upstream performs it instead of re-deriving it here, which is
    exactly the sort of reshape a port gets wrong silently.
    """

    def __init__(self, model) -> None:
        import torch

        self.torch = torch
        self.captured = None
        self.handle = model.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        if self.captured is not None:
            return
        logits = getattr(output, "logits", None)
        if logits is None and isinstance(output, (tuple, list)):
            logits = output[-1]
        if self.torch.is_tensor(logits):
            self.captured = logits.detach().to(self.torch.float32).cpu().numpy()

    def close(self) -> None:
        self.handle.remove()


class MethodWrap:
    """Temporarily replace a bound method, restoring it on close.

    Two of the tensors this dump needs never cross a module boundary: the prompt
    grid is a local of ``_prepare_inference_inputs``, and the committed token
    grid is an argument to ``audio_tokenizer.decode``. Wrapping the method for
    the duration of the call captures them without altering the entry point the
    manifest pins.
    """

    def __init__(self, owner, name: str, factory) -> None:
        self.owner = owner
        self.name = name
        self.original = getattr(owner, name)
        self.calls: list = []
        setattr(owner, name, factory(self.original, self.calls))

    def close(self) -> None:
        setattr(self.owner, self.name, self.original)


def _prompt_wrap(original, calls):
    def wrapper(text, num_target_tokens, ref_text=None, ref_audio_tokens=None,
                lang=None, instruct=None, denoise=True):
        prepared = original(
            text, num_target_tokens, ref_text, ref_audio_tokens, lang, instruct, denoise
        )
        calls.append(
            {
                "input_ids": prepared["input_ids"].detach().cpu().numpy(),
                "num_target_tokens": int(num_target_tokens),
                "num_ref_tokens": 0 if ref_audio_tokens is None else int(ref_audio_tokens.size(-1)),
                "text": text,
                "ref_text": ref_text,
                "lang": lang,
                "instruct": instruct,
                "denoise": bool(denoise),
            }
        )
        return prepared

    return wrapper


def _decode_wrap(original, calls):
    def wrapper(audio_codes, *args, **kwargs):
        calls.append(audio_codes.detach().cpu().numpy())
        return original(audio_codes, *args, **kwargs)

    return wrapper


def _encode_wrap(original, calls):
    def wrapper(input_values, *args, **kwargs):
        calls.append(input_values.detach().cpu().numpy())
        return original(input_values, *args, **kwargs)

    return wrapper


class SemanticProbe:
    """The HuBERT branch's final-layer hidden states during a reference encode.

    Note what this is and is not. ``_extract_semantic_features`` feeds the
    quantiser the *mean over all thirteen* HuBERT hidden states, not the last
    one; this probe is the last layer alone, which is the stage boundary a port
    can compare cheaply before the averaging. ``metadata.json`` records the
    distinction so nobody later reads a passing comparison here as proof the
    semantic branch as a whole agrees.
    """

    def __init__(self, audio_tokenizer) -> None:
        import torch

        self.torch = torch
        self.captured = None
        semantic = getattr(audio_tokenizer, "semantic_model", None)
        if semantic is None:
            raise SystemExit(
                "audio tokenizer has no .semantic_model; the reference probe is out of date"
            )
        self.module_type = type(semantic).__name__
        self.handle = semantic.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        if self.captured is not None:
            return
        tensor = _first_tensor(output)
        if self.torch.is_tensor(tensor):
            self.captured = tensor.detach().to(self.torch.float32).cpu().numpy()

    def close(self) -> None:
        self.handle.remove()


class Pcm16kProbe:
    """The resampled, mono, PRE-pad 16 kHz waveform HuBERT is actually called on.

    ``_extract_semantic_features`` (transformers'
    ``modeling_higgs_audio_v2_tokenizer.py:490-499``, pinned in this
    environment) resamples 24 kHz to 16 kHz with ``torchaudio.functional.resample``
    at its defaults, keeps channel 0, then pads a *fixed* 160 samples onto each
    end before calling HuBERT -- a constant the pinned source itself flags as
    differing from boson's original ``hop_length // 2 == 480`` (see the
    in-source TODO at :497, referencing boson-ai/higgs-audio's
    ``higgs_audio_v2_tokenizer.py:173-174``). Wrapping
    ``torchaudio.functional.resample`` directly is fragile (it is called by
    fully-qualified reference from inside the module, not looked up on an
    instance), so this probe instead pre-hooks HuBERT's own forward and strips
    the known 160-sample pad back off both ends. The length check below is a
    self-consistency affirmation, not an independent proof: `stripped_length`
    is `padded_length - 2 * PAD` by construction of the slice bounds, so the
    two can never disagree; what it actually guards is arithmetic sanity if
    the slicing above is ever edited, not a live risk from anything this
    dump encounters today. A `SystemExit` on disagreement stops the dump
    rather than silently writing a mis-aligned artifact.
    """

    PAD = 160

    def __init__(self, semantic_model) -> None:
        self.captured = None
        self.handle = semantic_model.register_forward_pre_hook(self._hook)

    def _hook(self, _module, args):
        import torch

        if self.captured is not None:
            return
        input_values = args[0]
        padded_length = int(input_values.shape[-1])
        stripped = input_values[:, self.PAD : padded_length - self.PAD]
        stripped_length = int(stripped.shape[-1])
        if padded_length != stripped_length + 2 * self.PAD:
            raise SystemExit(
                f"pcm_16k probe: padded length {padded_length} does not equal stripped "
                f"length {stripped_length} + {2 * self.PAD}; the fixed 160-sample pad "
                "strip is wrong and the artifact would not be the pre-pad waveform"
            )
        self.captured = stripped.detach().to(torch.float32).cpu().numpy()

    def close(self) -> None:
        self.handle.remove()


class SemanticMeanProbe:
    """The mean over all thirteen HuBERT hidden states, before the [::2] downsample.

    ``_extract_semantic_features`` (``modeling_higgs_audio_v2_tokenizer.py:501-505``)
    stacks HuBERT's ``output_hidden_states`` tuple along a NEW dimension 1 --
    ``torch.stack([h.to(input_values.device) for h in hidden_states], dim=1)``,
    giving ``[batch, num_layers, seq, hidden]`` -- and then averages over that
    same dim 1, producing ``[batch, seq, hidden]``. This is the feature the
    codec's quantiser actually consumes; ``SemanticProbe`` above captures only
    the last of the thirteen layers, a cheaper stage boundary, not this value.
    The hook transcribes the pinned source's exact stack-then-mean rather than
    assuming a shortcut (e.g. averaging along dim 0) that would only coincide
    with it because both reduce over the same number of elements. Captured
    BEFORE the ``[::2]`` stride-2 downsample at :507-508: this probe is that
    slice's input, not its output.
    """

    def __init__(self, semantic_model) -> None:
        self.captured = None
        self.handle = semantic_model.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        import torch

        if self.captured is not None:
            return
        hidden_states = getattr(output, "hidden_states", None)
        if not hidden_states:
            raise SystemExit(
                "semantic_mean probe: HuBERT forward returned no hidden_states; "
                "output_hidden_states=True must be in effect for _extract_semantic_features"
            )
        stacked = torch.stack([h.to(hidden_states[0].device) for h in hidden_states], dim=1)
        mean = stacked.mean(dim=1)
        self.captured = mean.detach().to(torch.float32).cpu().numpy()

    def close(self) -> None:
        self.handle.remove()


class FusedLatentProbe:
    """``audio_tokenizer.fc``'s own output, first call: the RVQ quantiser's input.

    ``HiggsAudioV2TokenizerModel.encode`` (``modeling_higgs_audio_v2_tokenizer.py:544-556``)
    concatenates the acoustic and (downsampled) semantic branches and feeds the
    result through ``self.fc`` (``nn.Linear(1024, 1024)``); exactly that
    tensor -- reshaped back to channel-first, but not otherwise touched -- is
    what ``self.quantizer.encode`` receives. Hooking ``fc`` directly captures
    the per-frame latent the port must reproduce without re-deriving the
    branch concatenation this dumper does not otherwise need to know about.
    """

    def __init__(self, fc) -> None:
        self.captured = None
        self.handle = fc.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        import torch

        if self.captured is not None:
            return
        if not torch.is_tensor(output):
            raise SystemExit(
                "fused_latent probe: audio_tokenizer.fc did not return a plain tensor"
            )
        self.captured = output.detach().to(torch.float32).cpu().numpy()

    def close(self) -> None:
        self.handle.remove()


# ---------------------------------------------------------------------------
# Clone reference
# ---------------------------------------------------------------------------


def verify_pinned_inputs(manifest: dict, weights_dir: pathlib.Path) -> list[str]:
    """sha256-verify every weights-repository input the manifest pins.

    Walks `omnivoice_pinned_inputs.PINNED_INPUTS` -- the same six-entry table
    `convert-omnivoice.py` verifies at startup, rather than a private idea of
    which local files count as pinned inputs inferred solely from which
    manifest artifacts happen to carry a "/resolve/" marker. Until this table
    existed only the clone reference audio was checked with anything like this
    rigor; the weights, configs and tokenizer the dump actually reads were
    trusted. A parity baseline dumped from silently different inputs would be
    wrong in a way no later gate could localise, so a mismatch stops the dump.

    Source-repository files (the Apache LICENSE) and the clone reference
    (checked separately by `materialise_reference`) carry no "/resolve/"
    marker or are not in the table, and are not dump inputs.
    """
    verified = []
    for pin in omnivoice_pinned_inputs.PINNED_INPUTS:
        matches = [
            artifact for artifact in manifest["source"]["artifacts"]
            if artifact["role"] == pin.role
            and omnivoice_pinned_inputs.relative_path_from_locator(artifact["locator"]) == pin.relative_path
        ]
        if len(matches) != 1:
            raise SystemExit(
                f"the manifest names {len(matches)} {pin.role} artifacts resolving to "
                f"{pin.relative_path!r}; exactly one is required to pin the dump"
            )
        local = omnivoice_pinned_inputs.resolve_local(weights_dir, pin)
        if not local.is_file():
            raise SystemExit(f"{local}: the manifest pins this input and it is missing")
        # Chunked: model.safetensors is multi-gigabyte and this runs before
        # the model loads, so a whole-file read_bytes() would add a transient
        # allocation of the same size on top of the dump's own peak.
        digest = hashlib.sha256()
        with local.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        expected = matches[0]["sha256"]
        if actual != expected:
            raise SystemExit(
                f"{local}: sha256 {actual} does not match the manifest's "
                f"{expected}; refusing to dump against unpinned inputs"
            )
        verified.append(pin.relative_path)
    return verified


def reference_digest(manifest: dict, locator: str) -> str:
    for artifact in manifest.get("source", {}).get("artifacts", []):
        if artifact.get("locator") == locator:
            digest = artifact.get("sha256")
            if not digest:
                raise ManifestError(f"manifest artifact {locator} carries no sha256")
            return digest
    raise ManifestError(f"manifest source.artifacts has no entry for {locator}")


def materialise_reference(locator: str, digest: str, directory: pathlib.Path) -> pathlib.Path:
    """Fetch the clone reference if absent, and refuse to dump a wrong one.

    Upstream ships no reference audio at any pinned revision, so this artifact
    is pinned by content instead. A digest mismatch stops the clone cases rather
    than substituting a different voice into a parity baseline.
    """
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / locator.rsplit("/", 1)[-1]
    if not destination.exists():
        print(f"fetching {locator}", flush=True)
        with urllib.request.urlopen(locator, timeout=60) as response:  # noqa: S310 - pinned https locator
            destination.write_bytes(response.read())
    actual = hashlib.sha256(destination.read_bytes()).hexdigest()
    if actual != digest:
        raise SystemExit(
            f"{destination}: sha256 {actual} does not match the manifest's {digest}. "
            "Refusing to dump a clone case against an unpinned reference. "
            "If a stale cached file is the cause, delete it and re-run to re-fetch."
        )
    return destination


def build_clone_prompt(model, path: pathlib.Path, ref_text: str, preprocess_prompt: bool) -> dict:
    """Encode the reference, capturing the PCM, the tokens and the HuBERT states.

    Three additional probes were added by jiangzhuo's ruling of 2026-08-01 to
    look inside the semantic branch that ``SemanticProbe`` alone only bounds:
    ``Pcm16kProbe`` (the waveform HuBERT reads), ``SemanticMeanProbe`` (the
    feature the quantiser actually consumes) and ``FusedLatentProbe`` (the
    quantiser's input after the acoustic/semantic branches are combined).
    """
    import soundfile
    import torch

    data, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T.copy())

    semantic = SemanticProbe(model.audio_tokenizer)
    pcm_16k = Pcm16kProbe(model.audio_tokenizer.semantic_model)
    semantic_mean = SemanticMeanProbe(model.audio_tokenizer.semantic_model)
    fused_latent = FusedLatentProbe(model.audio_tokenizer.fc)
    encode = MethodWrap(model.audio_tokenizer, "encode", _encode_wrap)
    try:
        prompt = model.create_voice_clone_prompt(
            (waveform, int(sample_rate)),
            ref_text=ref_text,
            preprocess_prompt=preprocess_prompt,
        )
    finally:
        semantic.close()
        pcm_16k.close()
        semantic_mean.close()
        fused_latent.close()
        encode.close()

    if not encode.calls:
        raise SystemExit(f"{path.name}: the reference never reached audio_tokenizer.encode")
    if semantic.captured is None:
        raise SystemExit(f"{path.name}: the HuBERT branch produced no hidden states")
    if pcm_16k.captured is None:
        raise SystemExit(f"{path.name}: the pcm_16k probe never saw a HuBERT forward")
    if semantic_mean.captured is None:
        raise SystemExit(f"{path.name}: the semantic_mean probe never saw a HuBERT forward")
    if fused_latent.captured is None:
        raise SystemExit(f"{path.name}: the fused_latent probe never saw audio_tokenizer.fc run")

    # (1, 1, T) as passed to encode -- already mono, resampled and hop-clipped.
    encoded_pcm = encode.calls[0].reshape(-1)
    tokens = prompt.ref_audio_tokens.detach().cpu().numpy()
    if tokens.shape[0] != NUM_CODEBOOKS:
        raise SystemExit(
            f"{path.name}: expected {NUM_CODEBOOKS} reference codebooks, got {tokens.shape}"
        )
    if encoded_pcm.shape[0] % SAMPLES_PER_FRAME != 0:
        raise SystemExit(
            f"{path.name}: {encoded_pcm.shape[0]} reference samples is not a whole number "
            f"of {SAMPLES_PER_FRAME}-sample hops"
        )
    return {
        "prompt": prompt,
        "pcm_24k": encoded_pcm,
        "tokens": tokens,
        "semantic_hidden": semantic.captured[0],
        "semantic_module": semantic.module_type,
        "pcm_16k": pcm_16k.captured[0],
        "semantic_mean": semantic_mean.captured[0],
        "fused_latent": fused_latent.captured[0],
        "source_sample_rate": int(sample_rate),
        "ref_rms": float(prompt.ref_rms),
        "ref_text": prompt.ref_text,
    }


# ---------------------------------------------------------------------------
# The dump
# ---------------------------------------------------------------------------


def torch_environment(torch) -> dict:
    """The execution configuration a baseline depends on, recorded per dump."""
    return {
        "torch_version": torch.__version__,
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
    }


def volume_branch(ref_rms) -> str:
    """Which arm of the ungated volume branch this case took.

    Recorded per case because the branch is invisible in the parameters: it
    keys off whether a reference exists and how loud it was, and it is applied
    with every post-processing switch off.
    """
    if ref_rms is None:
        return "peak_normalise_to_0.5"
    if ref_rms < 0.1:
        return "scale_by_ref_rms_over_0.1"
    return "none"


def assert_request_is_a_resolver_fixed_point(case: dict) -> dict:
    """The manifest's language and instruct strings must survive the resolvers.

    v1 does not reimplement ``_resolve_instruct``'s normalisation: the oracle
    cases pin already-normalised strings, and the port passes Description Text
    through unchanged. That is only honest if running the resolver over the
    pinned strings changes nothing -- otherwise the prompt the oracle builds
    would differ from the string the manifest, the tolerances and the port all
    name. Checked per case rather than assumed, because it is the assumption
    the whole passthrough decision rests on.
    """
    from omnivoice.models.omnivoice import _resolve_instruct, _resolve_language
    from omnivoice.utils.voice_design import _ZH_RE

    parameters = case["oracle"]["parameters"]
    resolved_language = _resolve_language(parameters["language"])
    if resolved_language != parameters["language"]:
        raise SystemExit(
            f"{case['id']}: _resolve_language({parameters['language']!r}) returned "
            f"{resolved_language!r}; the manifest pins already-resolved codes"
        )
    instruct = parameters["instruct"]
    if instruct is None:
        return {"language_is_fixed_point": True, "instruct_is_fixed_point": None}
    use_zh = bool(case["input"]["text"] and _ZH_RE.search(case["input"]["text"]))
    resolved_instruct = _resolve_instruct(instruct, use_zh=use_zh)
    if resolved_instruct != instruct:
        raise SystemExit(
            f"{case['id']}: _resolve_instruct({instruct!r}, use_zh={use_zh}) returned "
            f"{resolved_instruct!r}; the manifest pins already-normalised instructs, and "
            "the port's passthrough of Description Text depends on that being a no-op"
        )
    return {
        "language_is_fixed_point": True,
        "instruct_is_fixed_point": True,
        "instruct_resolver_use_zh": use_zh,
    }


def generate_once(model, case: dict, clone, speed: float):
    """One oracle call, with the case's parameters passed through verbatim."""
    parameters = case["oracle"]["parameters"]
    kwargs = {key: value for key, value in parameters.items() if key in GEN_CONFIG_KEYS}
    kwargs.update(
        {key: parameters[key] for key in GENERATE_ARGUMENT_KEYS if key in parameters}
    )
    if clone is not None:
        kwargs["voice_clone_prompt"] = clone["prompt"]
    return model.generate(text=case["input"]["text"], speed=speed, **kwargs)


class CaseCaptures:
    """The four interception points one case needs, installed and torn down together.

    Two are module hooks and two are method wraps, and all four must come off
    even when the generation raises -- a leaked hook would silently poison the
    next case's step-0 capture with this case's first forward.
    """

    def __init__(self, model) -> None:
        self.probes = GeneratorProbes(model)
        self.logits = LogitsProbe(model)
        self.prompt = MethodWrap(model, "_prepare_inference_inputs", _prompt_wrap)
        self.codes = MethodWrap(model.audio_tokenizer, "decode", _decode_wrap)

    def close(self) -> None:
        for capture in (self.probes, self.logits, self.prompt, self.codes):
            capture.close()


def read_prompt(case_id: str, calls: list) -> dict:
    """The 8 x S conditional grid and the three regions it divides into."""
    if len(calls) != 1:
        raise SystemExit(
            f"{case_id}: _prepare_inference_inputs ran {len(calls)} times; the dump "
            "assumes one un-chunked sequence per case"
        )
    prepared = calls[0]
    grid_ids = prepared["input_ids"]
    if grid_ids.shape[0] != 1 or grid_ids.shape[1] != NUM_CODEBOOKS:
        raise SystemExit(f"{case_id}: prompt grid has shape {grid_ids.shape}, expected [1, 8, S]")
    grid = grid_ids[0]
    conditional_length = int(grid.shape[1])
    num_target = prepared["num_target_tokens"]
    num_ref = prepared["num_ref_tokens"]
    text_region = conditional_length - num_target - num_ref
    if text_region <= 0:
        raise SystemExit(
            f"{case_id}: text region computed as {text_region} tokens from "
            f"S={conditional_length}, target={num_target}, ref={num_ref}"
        )
    # Every row of the 8-codebook dimension repeats the same text ids; asserting
    # it here is what makes writing row 0 alone an honest summary of the region.
    if not (grid[:, :text_region] == grid[0, :text_region]).all():
        raise SystemExit(f"{case_id}: the 8 prompt rows disagree over the text region")
    return {
        "grid": grid,
        "conditional_length": conditional_length,
        "num_target": num_target,
        "num_ref": num_ref,
        "text_region": text_region,
    }


def read_probes(case_id: str, probes: "GeneratorProbes", logits: "LogitsProbe",
                conditional_length: int) -> tuple:
    """Step-0 hidden states and audio-head logits, both on the conditional row."""
    missing = {f"{PROBE_ARTIFACT_PREFIX}{n}" for n in GENERATOR_PROBE_LAYERS} | {"generator.final"}
    missing -= set(probes.captured)
    if missing:
        raise SystemExit(f"{case_id}: probes captured nothing for {sorted(missing)}")
    if logits.captured is None:
        raise SystemExit(f"{case_id}: the audio-head logits were never seen")

    hidden: dict[str, object] = {}
    for name, array in probes.captured.items():
        if array.ndim != 3 or array.shape[1] < conditional_length:
            raise SystemExit(
                f"{case_id}: probe {name} has shape {array.shape}, expected "
                f"[batch, >= {conditional_length}, hidden]"
            )
        # Row 0 is the conditional branch; row 1 is the unconditional one, which
        # carries the target region only and is padded to the same width.
        hidden[name] = array[0, :conditional_length, :]

    step0 = logits.captured
    if step0.ndim != 4 or step0.shape[1] != NUM_CODEBOOKS:
        raise SystemExit(
            f"{case_id}: step-0 logits have shape {step0.shape}, expected [B, 8, S, V]"
        )
    return hidden, step0[0, :, :conditional_length, :]


def read_committed_grid(case_id: str, calls: list, num_target: int):
    """The 8 x T grid on its way into the codec, checked for shape and range."""
    if len(calls) != 1:
        raise SystemExit(
            f"{case_id}: audio_tokenizer.decode ran {len(calls)} times; the dump assumes "
            "one un-chunked decode per case"
        )
    committed = calls[0]
    if committed.ndim == 3:
        committed = committed[0]
    if committed.ndim != 2 or committed.shape[0] != NUM_CODEBOOKS:
        raise SystemExit(f"{case_id}: expected an [8, T] committed grid, got {calls[0].shape}")
    if committed.shape[1] != num_target:
        raise SystemExit(
            f"{case_id}: the estimator asked for {num_target} frames but "
            f"{committed.shape[1]} were committed"
        )
    low, high = int(committed.min()), int(committed.max())
    if low < 0 or high > AUDIO_MASK_ID - 1:
        raise SystemExit(
            f"{case_id}: committed grid holds values in [{low}, {high}]; the mask id "
            f"{AUDIO_MASK_ID} must never survive and the vocabulary ends at {AUDIO_MASK_ID - 1}"
        )
    return committed


def read_waveform(case_id: str, audios, frames: int, ceiling: int):
    """The PCM as returned to the caller, tied back to the committed grid."""
    import numpy as np

    audio = np.asarray(audios[0], dtype=np.float32)
    if audio.ndim != 1:
        raise SystemExit(f"{case_id}: expected 1-D PCM, got shape {audio.shape}")
    if not np.isfinite(audio).all():
        raise SystemExit(f"{case_id}: reference produced non-finite PCM")
    # One codec frame is exactly 960 samples. Checking it here ties the captured
    # grid to the captured audio: a transposed or mis-sliced grid still writes
    # plausible-looking files, and this is what notices.
    if audio.shape[0] != frames * SAMPLES_PER_FRAME:
        raise SystemExit(
            f"{case_id}: {frames} committed frames imply {frames * SAMPLES_PER_FRAME} samples, "
            f"but the returned PCM has {audio.shape[0]}"
        )
    if frames > ceiling:
        raise SystemExit(
            f"{case_id}: {frames} frames exceeds the package contract's {ceiling}-frame ceiling"
        )
    return audio


def assert_greedy_replays(model, case: dict, clone, speed: float, committed) -> bool:
    """Run a greedy case a second time in-process and require the same grid.

    Cheap next to the value: the whole validation strategy for this family
    rests on the committed grid being reproducible, and a run that quietly
    depended on some hidden state would otherwise become the baseline every
    later comparison is measured against.
    """
    import numpy as np

    second = MethodWrap(model.audio_tokenizer, "decode", _decode_wrap)
    try:
        generate_once(model, case, clone, speed)
    finally:
        second.close()
    replay = second.calls[0]
    if replay.ndim == 3:
        replay = replay[0]
    if not np.array_equal(replay, committed):
        differing = int((replay != committed).sum())
        raise SystemExit(
            f"{case['id']}: two greedy generations in one process disagreed on "
            f"{differing} of {committed.size} grid entries"
        )
    return True


def write_case(case: dict, case_dir: pathlib.Path, produced: dict) -> dict:
    """Write exactly the manifest's artifact set for this case, and prove it.

    The manifest drives the writing rather than the other way round, so a name
    the dump invents and a name the manifest expects but nothing produced are
    both errors here instead of a surprise at comparison time.
    """
    case_id = case["id"]
    expected = {artifact["name"]: artifact for artifact in case["expected"]["artifacts"]}
    unexpected = sorted(set(produced) - set(expected))
    unproduced = sorted(set(expected) - set(produced))
    if unexpected or unproduced:
        raise SystemExit(
            f"{case_id}: the dump produced {unexpected or 'nothing extra'} that the manifest "
            f"does not expect, and is missing {unproduced or 'nothing'}"
        )

    # The manifest declares each artifact's format independently of which
    # writer produces it; a mismatch here means the two drifted and the dump
    # would write bytes the comparison script reads under the wrong dtype.
    writer_formats = {write_i32: "i32le", write_f32: "f32le", write_json: "json"}
    for name, (writer, _payload) in produced.items():
        declared = expected[name]["format"]
        if writer_formats[writer] != declared:
            raise SystemExit(
                f"{case_id}: {name} is declared {declared!r} but the dump would write "
                f"{writer_formats[writer]!r}; the manifest and the writer registry drifted"
            )

    # Clear the case directory first. A re-run after the artifact set changed
    # would otherwise leave the previous run's files beside the new ones, and
    # the exactness check below would report a stale file as an extra product.
    if case_dir.exists():
        for stale in sorted(case_dir.rglob("*"), reverse=True):
            stale.unlink() if stale.is_file() else stale.rmdir()

    artifacts = {}
    for name, artifact in expected.items():
        writer, payload = produced[name]
        artifacts[name] = {
            "path": artifact["path"],
            **writer(case_dir / artifact["path"], payload),
        }

    on_disk = sorted(str(p.relative_to(case_dir)) for p in case_dir.rglob("*") if p.is_file())
    wanted = sorted(artifact["path"] for artifact in expected.values())
    if on_disk != wanted:
        raise SystemExit(
            f"{case_id}: {case_dir} holds {on_disk}, which is not exactly the manifest's {wanted}"
        )
    return artifacts


def resolve_clone(model, case: dict, clones: dict, manifest: dict,
                  reference_dir: pathlib.Path):
    """The reference-audio prompt for a clone case, built once and reused.

    Both clone cases point at the same reference with the same transcript, so
    the HuBERT encode runs once; keying the cache on the transcript and the
    preprocess flag as well as the locator means a case that varied either
    would get its own prompt rather than silently inherit this one.
    """
    reference_input = case["input"].get("reference")
    if reference_input is None:
        return None
    parameters = case["oracle"]["parameters"]
    locator = reference_input["artifact"]
    key = (locator, parameters["ref_text"], bool(parameters.get("preprocess_prompt", True)))
    if key not in clones:
        path = materialise_reference(locator, reference_digest(manifest, locator), reference_dir)
        clones[key] = build_clone_prompt(model, path, key[1], key[2])
    return clones[key]


def run_case(model, case: dict, output_root: pathlib.Path, clones: dict,
             manifest: dict, reference_dir: pathlib.Path) -> dict:
    import numpy as np
    import torch

    case_id = case["id"]
    parameters = case["oracle"]["parameters"]
    greedy = is_greedy(case)
    speed = float(case["request"]["speaking_rate"])
    reference_input = case["input"].get("reference")

    resolver = assert_request_is_a_resolver_fixed_point(case)
    clone = resolve_clone(model, case, clones, manifest, reference_dir)

    captures = CaseCaptures(model)
    rng_before = torch.random.get_rng_state()
    if not greedy:
        # The oracle exposes no seed parameter, so the reference draw is pinned
        # by seeding the global generator immediately before the call. The
        # case's own seed is used, so the three sampled cases produce genuinely
        # distinct references rather than three copies of one.
        torch.manual_seed(int(case["request"]["seed_u64"]))

    started = time.time()
    try:
        audios = generate_once(model, case, clone, speed)
    finally:
        captures.close()
    wall_seconds = time.time() - started

    if greedy and not torch.equal(rng_before, torch.random.get_rng_state()):
        raise SystemExit(
            f"{case_id}: the torch RNG state moved during a greedy run. Both temperatures are "
            "zero, so the decode loop should take neither Gumbel branch and make no draw; the "
            "family's cross-process determinism claim does not hold as written."
        )

    prompt = read_prompt(case_id, captures.prompt.calls)
    conditional_length = prompt["conditional_length"]
    hidden, step0_logits = read_probes(
        case_id, captures.probes, captures.logits, conditional_length
    )
    committed = read_committed_grid(case_id, captures.codes.calls, prompt["num_target"])
    frames = int(committed.shape[1])
    audio = read_waveform(
        case_id, audios, frames, int(manifest["package_contract"]["max_output_frames"])
    )

    double_run = None
    if greedy:
        double_run = assert_greedy_replays(model, case, clone, speed, committed)

    ref_rms = None if clone is None else clone["ref_rms"]
    peak = float(np.abs(audio).max())
    duration = float(audio.shape[0]) / NATIVE_SAMPLE_RATE
    result = {
        "case": case_id,
        "status": "ok",
        "sample_rate": NATIVE_SAMPLE_RATE,
        "samples": int(audio.shape[0]),
        "frames": frames,
        "duration_seconds": duration,
        "peak": peak,
        "rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))),
        "volume_branch": volume_branch(ref_rms),
        "ref_rms": ref_rms,
        "greedy": greedy,
        "double_run_identical": double_run,
        "rng_untouched": bool(greedy),
        "wall_seconds": round(wall_seconds, 3),
        "real_time_factor": round(wall_seconds / duration, 3) if duration else None,
    }
    metadata = {
        "case": case_id,
        "seed_u64": case["request"]["seed_u64"],
        "speaking_rate": speed,
        "language": parameters["language"],
        "instruct": parameters["instruct"],
        "ref_text": parameters.get("ref_text"),
        "resolver_fixed_point": resolver,
        "decoding": {key: parameters[key] for key in sorted(set(parameters) & GEN_CONFIG_KEYS)},
        "prompt": {
            "conditional_length": conditional_length,
            "text_region_tokens": prompt["text_region"],
            "reference_audio_tokens": prompt["num_ref"],
            "target_frames": prompt["num_target"],
            "codebooks": NUM_CODEBOOKS,
            "style_and_text_rows_identical": True,
        },
        "probes": {
            "backbone_layers": list(GENERATOR_PROBE_LAYERS),
            "layer_module_path": captures.probes.layer_path,
            "norm_module_path": captures.probes.norm_path,
            "backbone_layer_count": captures.probes.layer_count,
            "step": 0,
            "batch_row": "conditional (row 0 of the CFG pair)",
        },
        "environment": torch_environment(torch),
        "shapes": {
            "prompt_grid": [NUM_CODEBOOKS, conditional_length],
            "token_ids": [prompt["text_region"]],
            "logits_step0": list(step0_logits.shape),
            "hidden": {name: list(array.shape) for name, array in sorted(hidden.items())},
            "codes_grid": [NUM_CODEBOOKS, frames],
            "audio_pcm": [int(audio.shape[0])],
        },
        "notes": [
            "generator/logits_step0.f32 is [C, S, V] flattened C-major, the permutation "
            "OmniVoice.forward itself performs.",
            "audio/pcm.f32 is the waveform as returned to the caller, carrying the ungated "
            f"volume branch: {volume_branch(ref_rms)}.",
        ],
    }
    if clone is not None:
        metadata["reference"] = {
            "artifact": reference_input["artifact"],
            "sha256": reference_digest(manifest, reference_input["artifact"]),
            "source_sample_rate": clone["source_sample_rate"],
            "ref_rms": clone["ref_rms"],
            "quiet_reference_boost_applied": 0 < clone["ref_rms"] < 0.1,
            "semantic_module": clone["semantic_module"],
            "semantic_hidden_shape": list(clone["semantic_hidden"].shape),
            "pcm_24k_samples": int(clone["pcm_24k"].shape[0]),
            "pcm_16k_shape": list(clone["pcm_16k"].shape),
            "semantic_mean_shape": list(clone["semantic_mean"].shape),
            "fused_latent_shape": list(clone["fused_latent"].shape),
        }
        metadata["notes"].append(
            "ref/semantic_hidden.f32 is the HuBERT branch's FINAL layer output. The codec "
            "feeds the quantiser the mean over all hidden states, not this tensor; the probe "
            "is a stage boundary, not the consumed feature."
        )
        metadata["notes"].append(
            "ref/pcm_16k.f32 is the 24->16 kHz resampled, channel-0 waveform HuBERT reads, "
            "captured BEFORE the fixed 160-sample pad each side; the pad itself is not in "
            "this file (clone-encode probe added by jiangzhuo's ruling of 2026-08-01)."
        )
        metadata["notes"].append(
            "ref/semantic_mean.f32 IS the feature the quantiser consumes: the mean over all "
            "thirteen HuBERT hidden states, captured BEFORE the [::2] stride-2 downsample "
            "(clone-encode probe added by jiangzhuo's ruling of 2026-08-01)."
        )
        metadata["notes"].append(
            "ref/fused_latent.f32 is audio_tokenizer.fc's own output, first call: the RVQ "
            "quantiser's input after the acoustic and semantic branches are combined "
            "(clone-encode probe added by jiangzhuo's ruling of 2026-08-01)."
        )

    produced = {
        "input.token_ids": (write_i32, prompt["grid"][0, : prompt["text_region"]]),
        "prompt.token_grid": (write_i32, prompt["grid"]),
        "generator.final": (write_f32, hidden["generator.final"]),
        "generator.logits_step0": (write_f32, step0_logits),
        "codes.grid": (write_i32, committed),
        "audio.pcm": (write_f32, audio),
        "result": (write_json, result),
        "metadata": (write_json, metadata),
    }
    for index in GENERATOR_PROBE_LAYERS:
        name = f"{PROBE_ARTIFACT_PREFIX}{index}"
        produced[name] = (write_f32, hidden[name])
    if clone is not None:
        produced["ref.pcm_24k"] = (write_f32, clone["pcm_24k"])
        produced["ref.tokens"] = (write_i32, clone["tokens"])
        produced["ref.semantic_hidden"] = (write_f32, clone["semantic_hidden"])
        produced["ref.pcm_16k"] = (write_f32, clone["pcm_16k"])
        produced["ref.semantic_mean"] = (write_f32, clone["semantic_mean"])
        produced["ref.fused_latent"] = (write_f32, clone["fused_latent"])

    artifacts = write_case(case, output_root / case_id, produced)
    return {"id": case_id, "result": result, "artifacts": artifacts}


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

    if arguments.weights_dir is None:
        print("error: --weights-dir is required unless --validate-only is given", file=sys.stderr)
        return 1
    if not arguments.weights_dir.is_dir():
        print(f"error: no weights directory at {arguments.weights_dir}", file=sys.stderr)
        return 1

    verified_inputs = verify_pinned_inputs(manifest, arguments.weights_dir)
    print(f"verified {len(verified_inputs)} pinned inputs against the manifest", flush=True)

    import torch

    # Exact-token baselines must be reproducible off this machine, not merely on
    # it. The thread pool is pinned because oneDNN/MKL reduction order can move
    # with pool size, and deterministic algorithms are demanded rather than
    # hoped for: an op with no deterministic CPU path aborts the dump instead of
    # quietly varying. set_num_interop_threads must run before any parallel op,
    # which is why this sits directly under the import.
    torch.set_num_interop_threads(1)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    from omnivoice.models.omnivoice import OmniVoice

    reference = manifest["reference"]
    print(f"\nloading {arguments.weights_dir} ({reference['dtype']} on {reference['device']})",
          flush=True)
    model = OmniVoice.from_pretrained(
        str(arguments.weights_dir), device_map="cpu", dtype=torch.float32
    )
    if str(model.device) != "cpu":
        raise SystemExit(f"model loaded on {model.device}; the manifest pins cpu")
    if model.sampling_rate != NATIVE_SAMPLE_RATE:
        raise SystemExit(
            f"audio tokenizer runs at {model.sampling_rate} Hz; the contract says "
            f"{NATIVE_SAMPLE_RATE}"
        )
    if model.audio_tokenizer.config.hop_length != SAMPLES_PER_FRAME:
        raise SystemExit(
            f"codec hop is {model.audio_tokenizer.config.hop_length}, expected {SAMPLES_PER_FRAME}"
        )

    # Module-path discovery, printed once. The hook wiring is resolved against
    # the real tree rather than a guessed attribute path, and the count of hooks
    # installed is asserted inside GeneratorProbes.
    discovery = GeneratorProbes(model)
    print(
        f"backbone      {type(model.llm).__name__}, {discovery.layer_count} layers at "
        f"{discovery.layer_path}, final norm at {discovery.norm_path}"
    )
    print(
        f"codec         {type(model.audio_tokenizer).__name__}, semantic branch "
        f"{type(model.audio_tokenizer.semantic_model).__name__}, hop "
        f"{model.audio_tokenizer.config.hop_length}"
    )
    print(f"hooks         {len(discovery.handles)} "
          f"({len(GENERATOR_PROBE_LAYERS)} probe layers + 1 final norm)")
    discovery.close()

    report = {
        "schema": DUMP_SCHEMA,
        "family": FAMILY,
        "variant": manifest["variant"],
        "suite_version": manifest["suite_version"],
        "manifest": str(arguments.manifest),
        "weights_dir": str(arguments.weights_dir),
        "reference": reference,
        "probe_layers": list(GENERATOR_PROBE_LAYERS),
        "module_paths": {
            "backbone": type(model.llm).__name__,
            "layers": discovery.layer_path,
            "norm": discovery.norm_path,
            "semantic": type(model.audio_tokenizer.semantic_model).__name__,
        },
        "environment": torch_environment(torch),
        "verified_inputs": verified_inputs,
        "case_count": 0,
        "cases": [],
    }

    clones: dict = {}
    started = time.time()
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case['id']}", flush=True)
        record = run_case(
            model, case, output_root, clones, manifest, arguments.reference_audio_dir
        )
        report["cases"].append(record)
        report["case_count"] = len(report["cases"])
        outcome = record["result"]
        print(
            f"    {outcome['frames']} frames, {outcome['duration_seconds']:.2f}s, "
            f"peak {outcome['peak']:.4f}, {outcome['wall_seconds']:.1f}s wall, "
            f"rtf {outcome['real_time_factor']}",
            flush=True,
        )
        # Written after every case so a crash leaves the finished ones described
        # rather than only on disk.
        if arguments.report:
            report["total_wall_seconds"] = round(time.time() - started, 3)
            write_json(arguments.report, report)

    report["total_wall_seconds"] = round(time.time() - started, 3)
    if arguments.report:
        write_json(arguments.report, report)
    print(
        f"dumped {len(report['cases'])} case(s) to {output_root} in "
        f"{report['total_wall_seconds']:.0f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
